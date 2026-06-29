"""Export a piLM quantized asset bundle.

The bundle keeps quantized Linear cache files plus the BF16 tensors that are not
represented by those cache files. It is an intermediate artifact: the runtime
loader still needs bundle support before it can load this directory directly.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from runtime.loader import load_model  # noqa: E402
from models import detect_arch, get_model  # noqa: E402
from models.qwen3.weights import (  # noqa: E402
    _build_name_map,
    _quant_cache_path,
    _quantizable_linear_module,
)


SMALL_MODEL_FILES = [
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
]


def _link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def _copy_small_files(model_dir: Path, out_dir: Path) -> list[str]:
    copied = []
    for name in SMALL_MODEL_FILES:
        src = model_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
            copied.append(name)
    return copied


def _load_index(model_dir: Path) -> dict:
    index_file = model_dir / "model.safetensors.index.json"
    if not index_file.exists():
        raise FileNotFoundError(f"missing index file: {index_file}")
    return json.loads(index_file.read_text(encoding="utf-8"))["weight_map"]


def _build_meta_model(model_dir: Path):
    loaded = load_model(str(model_dir))
    arch = detect_arch(loaded.config)
    model_cls = get_model(arch)
    with torch.device("meta"):
        return model_cls(loaded.config)


def _tensor_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def export_bundle(args: argparse.Namespace) -> dict:
    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    weight_map = _load_index(model_dir)
    model = _build_meta_model(model_dir)
    name_map = _build_name_map(model, weight_map)

    quantized_src = []
    residual_src = []
    for src_name, tgt_name in sorted(name_map.items()):
        module, _module_name = _quantizable_linear_module(
            model,
            tgt_name,
            skip_lm_head=False,
            quantize_policy="all",
        )
        if module is not None:
            quantized_src.append(src_name)
        else:
            residual_src.append(src_name)

    mtp_src = sorted(src for src in weight_map if src.startswith("mtp."))

    copied_files = _copy_small_files(model_dir, out_dir)

    linear_dir = out_dir / "linear_w4a16"
    link_modes = {"hardlink": 0, "copy": 0}
    missing_cache = []
    for src_name in quantized_src:
        cache_file = _quant_cache_path(cache_dir, src_name)
        if not cache_file.exists():
            missing_cache.append(src_name)
            continue
        mode = _link_or_copy(cache_file, linear_dir / cache_file.name)
        link_modes[mode] = link_modes.get(mode, 0) + 1

    def export_tensor_set(names: list[str], target_dir: Path, prefix: str) -> list[dict]:
        target_dir.mkdir(parents=True, exist_ok=True)
        exports = []
        by_shard: dict[str, list[str]] = {}
        for name in names:
            by_shard.setdefault(weight_map[name], []).append(name)
        for idx, (shard_name, shard_tensors) in enumerate(sorted(by_shard.items()), start=1):
            shard_path = model_dir / shard_name
            payload = {}
            with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
                for tensor_name in sorted(shard_tensors):
                    payload[tensor_name] = sf.get_tensor(tensor_name).contiguous().clone()
            out_file = target_dir / f"{prefix}-{idx:04d}.safetensors"
            save_file(payload, str(out_file))
            exports.append({
                "file": str(out_file.relative_to(out_dir)),
                "source_shard": shard_name,
                "tensors": len(payload),
                "bytes": out_file.stat().st_size,
            })
            del payload
        return exports

    residual_exports = export_tensor_set(residual_src, out_dir / "residual_bf16", "residual")
    mtp_exports = export_tensor_set(mtp_src, out_dir / "mtp_bf16", "mtp") if args.include_mtp else []

    manifest = {
        "format": "pilm-quant-bundle-v1",
        "source_model_dir": str(model_dir),
        "quantize": "w4a16-all",
        "linear_cache_dir": str(cache_dir),
        "small_files": copied_files,
        "quantized_linear_tensors": len(quantized_src),
        "residual_bf16_tensors": len(residual_src),
        "mtp_bf16_tensors": len(mtp_src) if args.include_mtp else 0,
        "quantized_linear_sources": quantized_src,
        "residual_sources": residual_src,
        "mtp_sources": mtp_src if args.include_mtp else [],
        "missing_linear_cache": missing_cache,
        "linear_cache_link_modes": link_modes,
        "residual_exports": residual_exports,
        "mtp_exports": mtp_exports,
    }
    (out_dir / "pilm_quant_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["bundle_bytes"] = _tensor_bytes(out_dir)
    manifest["bundle_gib"] = round(manifest["bundle_bytes"] / (1024 ** 3), 4)
    (out_dir / "pilm_quant_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--include-mtp", action="store_true")
    args = parser.parse_args()
    manifest = export_bundle(args)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
