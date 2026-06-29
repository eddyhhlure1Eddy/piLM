"""Qwen3.5 weight loading: maps HF safetensors names to model params, chunked for large tensors."""
import torch
import torch.nn as nn
import json
import os
import hashlib
import gc
import re
from pathlib import Path
from runtime.loader.safetensors_loader import SafetensorsFile
from models.qwen3.model import Qwen3Model
from models.base.linear import BackendLinear, QuantizedW8A32Linear, QuantizedW4A16Linear
from runtime.ckernel import (
    quantize_weight_i8_per_row_chunked,
    quantize_weight_i4_per_row_chunked,
    quantize_weight_i4_grouped_chunked,
)


def _build_name_map(model: Qwen3Model, weight_map: dict) -> dict:
    name_map = {}
    for name, _ in model.named_parameters():
        if name.startswith("embed_tokens"):
            src = f"model.language_model.{name}"
        elif name.startswith("layers."):
            parts = name.split(".")
            layer_idx = int(parts[1])
            remainder = ".".join(parts[2:])
            if remainder == "mlp.gate.weight":
                remainder = "mlp.gate_proj.weight"
            elif remainder == "mlp.up.weight":
                remainder = "mlp.up_proj.weight"
            elif remainder == "mlp.down.weight":
                remainder = "mlp.down_proj.weight"
            src = f"model.language_model.layers.{layer_idx}.{remainder}"
        elif name.startswith("norm."):
            src = f"model.language_model.norm.{name.split('.', 1)[1]}"
        elif name.startswith("lm_head."):
            src = "lm_head.weight"
        else:
            src = name
        if src in weight_map:
            name_map[src] = name
    return name_map


def _set_parameter(model: nn.Module, name: str, tensor: torch.Tensor, requires_grad: bool) -> None:
    parent_name, param_name = name.rsplit(".", 1) if "." in name else ("", name)
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, param_name, nn.Parameter(tensor, requires_grad=requires_grad))


def _set_module(model: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, module)


def _quant_cache_path(cache_dir: Path, src_name: str) -> Path:
    digest = hashlib.sha1(src_name.encode("utf-8")).hexdigest()[:16]
    leaf = src_name.replace("/", "_").replace(".", "_")
    return cache_dir / f"{digest}-{leaf}.safetensors"


def _load_quant_bundle_manifest(model_path: Path) -> dict | None:
    manifest_path = model_path / "pilm_quant_manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _bundle_weight_map(model_path: Path, manifest: dict) -> dict:
    from safetensors import safe_open

    weight_map = {}
    for export in manifest.get("residual_exports", []):
        rel_file = export["file"]
        with safe_open(str(model_path / rel_file), framework="pt", device="cpu") as sf:
            for name in sf.keys():
                weight_map[name] = rel_file
    for name in manifest.get("quantized_linear_sources", []):
        weight_map[name] = "__pilm_linear_cache__"
    return weight_map


def _load_tensor_copy_from_safetensors(shard_path: Path, src_name: str) -> torch.Tensor:
    with SafetensorsFile(str(shard_path)) as sf:
        info = sf.tensor_info(src_name)
        raw = sf.get_tensor_bytes(src_name)
    if info.dtype == "BF16":
        tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint16).clone().view(torch.bfloat16)
    elif info.dtype == "F16":
        tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint16).clone().view(torch.float16)
    elif info.dtype == "F32":
        tensor = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()
    elif info.dtype == "I64":
        tensor = torch.frombuffer(bytearray(raw), dtype=torch.int64).clone()
    elif info.dtype == "I32":
        tensor = torch.frombuffer(bytearray(raw), dtype=torch.int32).clone()
    elif info.dtype == "I8":
        tensor = torch.frombuffer(bytearray(raw), dtype=torch.int8).clone()
    elif info.dtype == "U8":
        tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
    else:
        raise ValueError(f"unsupported safetensors dtype for copy load: {info.dtype}")
    return tensor.reshape(info.shape).contiguous()


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _layer_index(module_name: str) -> int | None:
    match = _LAYER_RE.search(module_name)
    if not match:
        return None
    return int(match.group(1))


def _static_protect_layers() -> int:
    try:
        return max(0, int(os.environ.get("PILM_STATIC_PROTECT_LAYERS", "2")))
    except ValueError:
        return 2


def _should_quantize_linear(model: nn.Module, module_name: str, quantize_policy: str = "all") -> bool:
    if quantize_policy in {"all", "w8a32", "w8a32-all"}:
        return True
    if quantize_policy != "static":
        raise ValueError(f"unsupported quantize policy: {quantize_policy}")

    if module_name == "lm_head":
        return False

    layer_idx = _layer_index(module_name)
    if layer_idx is not None:
        try:
            num_layers = len(model.layers)  # type: ignore[attr-defined]
        except Exception:
            num_layers = 0
        protected = _static_protect_layers()
        if protected and num_layers and (layer_idx < protected or layer_idx >= num_layers - protected):
            return False

    # Full attention Q/K/V/O projections are more sensitive for retrieval and
    # long-context routing, so the static policy keeps them in BF16.
    if ".self_attn." in module_name:
        return False

    # Linear-attention gates/decay are small but control state dynamics.
    sensitive_linear_attn = (
        ".linear_attn.in_proj_z",
        ".linear_attn.in_proj_a",
        ".linear_attn.in_proj_b",
    )
    if any(module_name.endswith(suffix) for suffix in sensitive_linear_attn):
        return False

    return True


def _quantizable_linear_module(
    model: nn.Module,
    tgt_name: str,
    skip_lm_head: bool = True,
    quantize_policy: str = "all",
) -> tuple[BackendLinear | None, str]:
    if not tgt_name.endswith(".weight"):
        return None, ""
    module_name = tgt_name[:-len(".weight")]
    if skip_lm_head and module_name == "lm_head":
        return None, ""
    if not _should_quantize_linear(model, module_name, quantize_policy=quantize_policy):
        return None, ""
    try:
        module = model.get_submodule(module_name)
    except AttributeError:
        return None, ""
    if not isinstance(module, BackendLinear) or module.bias is not None:
        return None, ""
    return module, module_name


def _set_quantized_linear(
    model: nn.Module,
    module_name: str,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    quant_format: str = "w8a32",
    in_features: int | None = None,
) -> None:
    if quant_format in {"w4a16", "w4a16g32", "w4a16g128"}:
        if in_features is None:
            raise ValueError("in_features is required for w4a16")
        _set_module(model, module_name, QuantizedW4A16Linear(qweight, scales, in_features))
    else:
        _set_module(model, module_name, QuantizedW8A32Linear(qweight, scales))


def _maybe_load_quantized_linear_cache(
    model: nn.Module,
    module_name: str,
    module: BackendLinear,
    cache_file: Path,
    quant_format: str = "w8a32",
) -> bool:
    if not cache_file.exists():
        return False
    try:
        from safetensors.torch import load_file
        payload = load_file(str(cache_file), device="cpu")
        qweight = payload["qweight"]
        scales = payload["scales"]
        if quant_format in {"w4a16", "w4a16g32", "w4a16g128"}:
            expected_shape = (module.out_features, (module.in_features + 1) // 2)
            expected_scales_shape = (
                (module.out_features, (module.in_features + 31) // 32)
                if quant_format == "w4a16g32"
                else (module.out_features, (module.in_features + 127) // 128)
                if quant_format == "w4a16g128"
                else (module.out_features,)
            )
        else:
            expected_shape = (module.out_features, module.in_features)
            expected_scales_shape = (module.out_features,)
        if tuple(qweight.shape) != expected_shape or tuple(scales.shape) != expected_scales_shape:
            return False
        expected_dtype = torch.uint8 if quant_format in {"w4a16", "w4a16g32", "w4a16g128"} else torch.int8
        if qweight.device.type != "cpu" or qweight.dtype != expected_dtype:
            return False
        if scales.device.type != "cpu" or scales.dtype != torch.float32:
            return False
        if quant_format in {"w4a16", "w4a16g32", "w4a16g128"}:
            cached_in_features = payload.get("in_features")
            if cached_in_features is not None and int(cached_in_features.reshape(-1)[0].item()) != module.in_features:
                return False
        qweight = qweight.contiguous().clone()
        scales = scales.contiguous().clone()
        _set_quantized_linear(
            model,
            module_name,
            qweight,
            scales,
            quant_format=quant_format,
            in_features=module.in_features,
        )
        return True
    except Exception:
        return False


def _save_quantized_linear_cache(
    cache_file: Path,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    in_features: int | None = None,
) -> None:
    from safetensors.torch import save_file

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
    payload = {"qweight": qweight.cpu(), "scales": scales.cpu()}
    if in_features is not None:
        payload["in_features"] = torch.tensor([int(in_features)], dtype=torch.int32)
    save_file(payload, str(tmp_file))
    os.replace(tmp_file, cache_file)


def _maybe_set_quantized_linear(
    model: nn.Module,
    tgt_name: str,
    tensor: torch.Tensor,
    skip_lm_head: bool = True,
    cache_file: Path | None = None,
    quantize_policy: str = "all",
    quant_format: str = "w8a32",
) -> bool:
    if tensor.ndim != 2:
        return False
    module, module_name = _quantizable_linear_module(
        model,
        tgt_name,
        skip_lm_head=skip_lm_head,
        quantize_policy=quantize_policy,
    )
    if module is None:
        return False
    rows_per_chunk = int(os.environ.get("PILM_QUANT_ROWS_PER_CHUNK", "1024"))
    if quant_format == "w4a16":
        qweight, scales = quantize_weight_i4_per_row_chunked(tensor, rows_per_chunk=rows_per_chunk)
    elif quant_format == "w4a16g32":
        qweight, scales = quantize_weight_i4_grouped_chunked(tensor, group_size=32, rows_per_chunk=rows_per_chunk)
    elif quant_format == "w4a16g128":
        qweight, scales = quantize_weight_i4_grouped_chunked(tensor, group_size=128, rows_per_chunk=rows_per_chunk)
    else:
        qweight, scales = quantize_weight_i8_per_row_chunked(tensor, rows_per_chunk=rows_per_chunk)
    _set_quantized_linear(
        model,
        module_name,
        qweight,
        scales,
        quant_format=quant_format,
        in_features=module.in_features,
    )
    if cache_file is not None:
        try:
            _save_quantized_linear_cache(
                cache_file,
                qweight,
                scales,
                in_features=module.in_features if quant_format in {"w4a16", "w4a16g32", "w4a16g128"} else None,
            )
        except Exception:
            pass
    return True


def _load_with_safe_open(model: Qwen3Model, model_path: Path, weight_map: dict, name_map: dict):
    from safetensors import safe_open

    shard_set = sorted(set(weight_map.values()))
    loaded = 0
    missing = []

    for shard_name in shard_set:
        shard_path = model_path / shard_name
        if not shard_path.exists():
            missing.append(f"{shard_name}: shard file not found at {shard_path}")
            continue

        with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
            for src_name in sf.keys():
                if src_name not in name_map:
                    continue
                tgt_name = name_map[src_name]
                try:
                    param = model.get_parameter(tgt_name)
                    tensor = sf.get_tensor(src_name)
                    if tuple(tensor.shape) != tuple(param.shape):
                        tensor = tensor.reshape(param.shape)
                    if tensor.dtype != param.dtype:
                        tensor = tensor.to(param.dtype)
                    tensor = tensor.contiguous().clone()
                    if param.is_meta:
                        _set_parameter(model, tgt_name, tensor, param.requires_grad)
                    else:
                        param.data.copy_(tensor)
                    loaded += 1
                except Exception as e:
                    missing.append(f"{tgt_name}: {e}")

    return loaded, missing


def _load_with_mmap_reader(model: Qwen3Model, model_path: Path, weight_map: dict, name_map: dict):
    shard_set = sorted(set(weight_map.values()))
    loaded = 0
    missing = []

    TORCH_DTYPE = {
        "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
        "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
        "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
        "U8": torch.uint8,
    }

    for shard_name in shard_set:
        shard_path = model_path / shard_name
        if not shard_path.exists():
            missing.append(f"{shard_name}: shard file not found at {shard_path}")
            continue
        sf = SafetensorsFile(str(shard_path))
        for src_name in sf.list_tensors():
            if src_name not in name_map:
                continue
            tgt_name = name_map[src_name]
            info = sf.tensor_info(src_name)
            dtype = TORCH_DTYPE.get(info.dtype, torch.float32)
            try:
                param = model.get_parameter(tgt_name)
                if param.is_meta:
                    raise RuntimeError("meta parameters require the safetensors package")
                nbytes = info.nbytes
                elem_bytes = 2 if info.dtype in ("BF16", "F16") else (4 if info.dtype == "F32" else 1)

                def _copy_into(view_mv, dst, shape, src_dtype):
                    """Zero-intermediate-copy: torch.frombuffer shares mmap, then one copy_ into param."""
                    if src_dtype in ("BF16", "F16"):
                        t = torch.frombuffer(view_mv, dtype=torch.bfloat16 if src_dtype == "BF16" else torch.float16)
                    elif src_dtype == "F32":
                        t = torch.frombuffer(view_mv, dtype=torch.float32)
                    elif src_dtype == "I64":
                        t = torch.frombuffer(view_mv, dtype=torch.int64)
                    elif src_dtype == "I32":
                        t = torch.frombuffer(view_mv, dtype=torch.int32)
                    else:
                        t = torch.frombuffer(view_mv, dtype=torch.uint8).to(dtype)
                    t = t.reshape(shape)
                    if t.dtype != dst.dtype:
                        dst.copy_(t.to(dst.dtype))
                    else:
                        dst.copy_(t)

                if nbytes > 256 * 1024 * 1024:
                    chunk_rows = max(1, 128 * 1024 * 1024 // (info.shape[-1] * elem_bytes))
                    row_bytes = info.shape[-1] * elem_bytes
                    for start_row in range(0, info.shape[0], chunk_rows):
                        end_row = min(start_row + chunk_rows, info.shape[0])
                        rbs = start_row * row_bytes
                        rbe = end_row * row_bytes
                        chunk_view = sf.get_tensor_byte_range(src_name, rbs, rbe)
                        _copy_into(chunk_view, param.data[start_row:end_row],
                                   (end_row - start_row, info.shape[-1]), info.dtype)
                        del chunk_view
                    loaded += 1
                else:
                    view = sf.get_tensor_view(src_name)
                    shape = param.shape if param.shape != tuple(info.shape) else info.shape
                    _copy_into(view, param.data, shape, info.dtype)
                    loaded += 1
                    del view
            except Exception as e:
                missing.append(f"{tgt_name}: {e}")
        sf.close()

    return loaded, missing


def load_weights_from_safetensors(model: Qwen3Model, model_dir: str):
    """Load weights using our own reader. Maps HF names to model parameter names.

    Returns (loaded_count, missing_list).
    """
    model_path = Path(model_dir)
    index_file = model_path / "model.safetensors.index.json"

    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
    else:
        sf = SafetensorsFile(str(model_path / "model.safetensors"))
        weight_map = {k: "model.safetensors" for k in sf.list_tensors()}
        sf.close()

    name_map = _build_name_map(model, weight_map)
    missing_model_params = [
        name for name, _ in model.named_parameters()
        if name not in set(name_map.values())
    ]
    missing = [f"{name}: no checkpoint tensor mapping" for name in missing_model_params]

    try:
        loaded, load_issues = _load_with_safe_open(model, model_path, weight_map, name_map)
    except ImportError:
        loaded, load_issues = _load_with_mmap_reader(model, model_path, weight_map, name_map)

    return loaded, missing + load_issues


def load_weights_w8a32_from_safetensors(
    model: Qwen3Model,
    model_dir: str,
    skip_lm_head: bool = True,
    quantize_policy: str = "all",
    quant_format: str = "w8a32",
):
    """Load checkpoint and quantize eligible Linear weights directly.

    Eligible bias-free `BackendLinear.weight` tensors are converted to
    `QuantizedW8A32Linear` as each safetensors tensor is read, avoiding a live
    BF16 parameter for those Linear weights.
    """
    from safetensors import safe_open

    model_path = Path(model_dir)
    bundle_manifest = _load_quant_bundle_manifest(model_path)
    index_file = model_path / "model.safetensors.index.json"
    if bundle_manifest is not None:
        weight_map = _bundle_weight_map(model_path, bundle_manifest)
    elif index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
    else:
        sf = SafetensorsFile(str(model_path / "model.safetensors"))
        weight_map = {k: "model.safetensors" for k in sf.list_tensors()}
        sf.close()

    name_map = _build_name_map(model, weight_map)
    missing_model_params = [
        name for name, _ in model.named_parameters()
        if name not in set(name_map.values())
    ]
    missing = [f"{name}: no checkpoint tensor mapping" for name in missing_model_params]

    loaded = 0
    quantized = 0
    cached = 0
    load_issues = []
    if quant_format in {"w4a16", "w4a16g32", "w4a16g128"}:
        cache_enabled = os.environ.get("PILM_W4A16_CACHE", os.environ.get("PILM_W8A32_CACHE", "0")) == "1"
        if quant_format == "w4a16g32":
            default_cache = "w4a16_g32"
        elif quant_format == "w4a16g128":
            default_cache = "w4a16_g128"
        else:
            default_cache = "w4a16"
        if quantize_policy == "all":
            default_cache = f"{default_cache}_all"
        if bundle_manifest is not None and quant_format == "w4a16":
            cache_enabled = True
            cache_dir = model_path / "linear_w4a16"
        else:
            cache_dir = Path(os.environ.get("PILM_W4A16_CACHE_DIR", str(model_path / ".pilm_cache" / default_cache)))
    else:
        cache_enabled = os.environ.get("PILM_W8A32_CACHE", "0") == "1"
        if bundle_manifest is not None and quant_format == "w8a32":
            # A W8A32 bundle ships pre-quantized per-Linear caches under
            # linear_w8a32/, so we can load without re-quantizing the 18 GB
            # checkpoint (avoids Windows page-file "os error 1455").
            cache_enabled = True
            linear_subdir = bundle_manifest.get("linear_subdir", "linear_w8a32")
            cache_dir = model_path / linear_subdir
        else:
            cache_dir = Path(os.environ.get("PILM_W8A32_CACHE_DIR", str(model_path / ".pilm_cache" / "w8a32")))
    debug = os.environ.get("PILM_LOAD_DEBUG") == "1"
    debug_limit = int(os.environ.get("PILM_LOAD_DEBUG_LIMIT", "80"))
    debug_count = 0

    def _debug(message: str, always: bool = False) -> None:
        nonlocal debug_count
        if not debug:
            return
        if always or debug_count < debug_limit:
            print(f"[weights:w8a32] {message}", flush=True)

    if bundle_manifest is not None:
        for src_name in bundle_manifest.get("quantized_linear_sources", []):
            if src_name not in name_map:
                continue
            tgt_name = name_map[src_name]
            try:
                module, module_name = _quantizable_linear_module(
                    model,
                    tgt_name,
                    skip_lm_head=skip_lm_head,
                    quantize_policy=quantize_policy,
                )
                if module is None:
                    continue
                cache_file = _quant_cache_path(cache_dir, src_name)
                if not _maybe_load_quantized_linear_cache(
                    model,
                    module_name,
                    module,
                    cache_file,
                    quant_format=quant_format,
                ):
                    load_issues.append(f"{tgt_name}: missing or invalid quant cache {cache_file}")
                    continue
                loaded += 1
                quantized += 1
                cached += 1
            except Exception as e:
                load_issues.append(f"{tgt_name}: {e}")

    for shard_name in sorted(set(weight_map.values())):
        if shard_name == "__pilm_linear_cache__":
            continue
        shard_path = model_path / shard_name
        if not shard_path.exists():
            load_issues.append(f"{shard_name}: shard file not found at {shard_path}")
            continue
        _debug(f"shard {shard_name}", always=True)
        with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
            for src_name in sf.keys():
                if src_name not in name_map:
                    continue
                tgt_name = name_map[src_name]
                debug_count += 1
                try:
                    module, module_name = _quantizable_linear_module(
                        model,
                        tgt_name,
                        skip_lm_head=skip_lm_head,
                        quantize_policy=quantize_policy,
                    )
                    cache_file = _quant_cache_path(cache_dir, src_name) if cache_enabled and module is not None else None
                    if cache_file is not None and _maybe_load_quantized_linear_cache(
                        model,
                        module_name,
                        module,
                        cache_file,
                        quant_format=quant_format,
                    ):
                        _debug(f"{debug_count}: cached {src_name} -> {tgt_name}")
                        loaded += 1
                        quantized += 1
                        cached += 1
                        continue
                    copy_load = quantize_policy == "static"
                    _debug(f"{debug_count}: get {src_name} -> {tgt_name}")
                    if copy_load:
                        tensor = _load_tensor_copy_from_safetensors(shard_path, src_name)
                    else:
                        tensor = sf.get_tensor(src_name)
                    _debug(f"{debug_count}: got shape={tuple(tensor.shape)} dtype={tensor.dtype}")
                    if _maybe_set_quantized_linear(
                        model,
                        tgt_name,
                        tensor,
                        skip_lm_head=skip_lm_head,
                        cache_file=cache_file,
                        quantize_policy=quantize_policy,
                        quant_format=quant_format,
                    ):
                        _debug(f"{debug_count}: quantized {tgt_name}")
                        del tensor
                        loaded += 1
                        quantized += 1
                        continue
                    param = model.get_parameter(tgt_name)
                    if tuple(tensor.shape) != tuple(param.shape):
                        tensor = tensor.reshape(param.shape)
                    if tensor.dtype != param.dtype:
                        tensor = tensor.to(param.dtype)
                    if param.is_meta:
                        _set_parameter(model, tgt_name, tensor.contiguous(), param.requires_grad)
                    else:
                        param.data.copy_(tensor)
                    _debug(f"{debug_count}: loaded param {tgt_name}")
                    del tensor
                    loaded += 1
                except Exception as e:
                    load_issues.append(f"{tgt_name}: {e}")
        gc.collect()
        if os.environ.get("PILM_TRIM_WORKING_SET", "1") != "0":
            try:
                from runtime.memory import trim_process_working_set
                trim_process_working_set()
            except Exception:
                pass

    return loaded, missing + load_issues, quantized, cached
