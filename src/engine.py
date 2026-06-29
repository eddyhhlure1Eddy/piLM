"""Inference engine: ties together loader, model, scheduler, KV cache, tokenizer.

Entry point: Engine(model_dir) -> generate(prompt) -> text
"""
import torch
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Generator, Union

os.environ.setdefault("HF_HOME", os.path.join(os.environ.get("LOCALAPPDATA", "/tmp"), "hf_cache"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(os.environ.get("LOCALAPPDATA", "/tmp"), "torch_cache"))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime.loader import load_model, ModelConfig
from runtime.scheduler import Scheduler, KVCacheManager, Request, RequestStatus, DEFAULT_BLOCK_SIZE
from runtime.kvcache import PhysicalKVCache
from models import get_model, get_weight_loader, detect_arch


class SimpleTokenizer:
    """Wraps HF AutoTokenizer if available; falls back to basic lookup."""

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self._tok = None
        try:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        except Exception:
            pass
        self._load_chat_template()
        if self._tok and self.chat_template and not getattr(self._tok, "chat_template", None):
            self._tok.chat_template = self.chat_template

    def _load_chat_template(self):
        tc_path = self.model_dir / "tokenizer_config.json"
        template_path = self.model_dir / "chat_template.jinja"
        self.chat_template = ""
        if tc_path.exists():
            with open(tc_path, encoding="utf-8") as f:
                tc = json.load(f)
            self.chat_template = tc.get("chat_template") or ""
            self.eos_token_id = tc.get("eos_token_id", tc.get("eos_token"))
        else:
            self.eos_token_id = 248046
        if template_path.exists():
            self.chat_template = template_path.read_text(encoding="utf-8")

    def format_prompt(self, prompt: Union[str, list]) -> str:
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": prompt}]
        return self.apply_chat(messages)

    def encode(self, text: str) -> List[int]:
        if self._tok:
            return self._tok(text, return_tensors=None)["input_ids"]
        raise RuntimeError("No tokenizer available")

    def decode(self, ids: List[int]) -> str:
        if self._tok:
            return self._tok.decode(ids, skip_special_tokens=True)
        raise RuntimeError("No tokenizer available")

    def apply_chat(self, messages: list) -> str:
        if self._tok and hasattr(self._tok, "apply_chat_template"):
            return self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return messages[-1]["content"] if messages else ""

    @property
    def eos_id(self) -> int:
        if self._tok:
            return self._tok.eos_token_id
        return 248046


@dataclass
class EngineSessionState:
    request: Optional[Request] = None
    conv_states: Optional[List[torch.Tensor]] = None
    recurrent_states: Optional[List[torch.Tensor]] = None
    last_text: str = ""
    last_reused_tokens: int = 0
    last_appended_prompt_tokens: int = 0


def _common_prefix_len(left: List[int], right: List[int]) -> int:
    limit = min(len(left), len(right))
    for idx in range(limit):
        if left[idx] != right[idx]:
            return idx
    return limit


class Engine:
    def __init__(
        self,
        model_dir: str,
        dtype: str = "bfloat16",
        quantize: Optional[str] = None,
        kv_cache_gb: Optional[float] = None,
    ):
        self.model_dir = model_dir
        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
        self.quantize = quantize or "none"
        self.quantization_policy = "none"
        self.quantized_linear_modules = 0
        self.cached_linear_modules = 0
        self.bf16_linear_modules = 0
        self.fused_swiglu_modules = 0
        self.mtp_draft = None
        if self.quantize.startswith("w4") and not os.environ.get("PILM_ECPU_LIB"):
            w4_lib = Path(__file__).resolve().parent / "ecpu" / "build_w4" / "libecpu.dll"
            if w4_lib.exists():
                os.environ["PILM_ECPU_LIB"] = str(w4_lib)
        self.ckernel_info = {}
        self.ckernel_available = self._detect_ckernel()

        print(f"[engine] loading config from {model_dir} ...")
        self.loaded = load_model(model_dir)
        self.config = self.loaded.config
        print(f"[engine] model: {self.config.architectures}, "
              f"layers={self.config.num_layers}, hidden={self.config.hidden_size}, "
              f"head_dim={self.config.head_dim}, kv_heads={self.config.num_kv_heads}")

        print(f"[engine] building model ...")
        arch = detect_arch(self.config)
        ModelClass = get_model(arch)
        weight_loader = get_weight_loader(arch)
        with torch.device("meta"):
            self.model = ModelClass(self.config)

        print(f"[engine] loading weights ({self.loaded.num_tensors} tensors, "
              f"{self.loaded.total_weight_bytes / (1024**3):.2f} GB) ...")
        if quantize in {"w8a32", "w8a32-all", "w8a32-static", "w4a16-all", "w4a16-static", "w4a16g32-static", "w4a16g128-static"}:
            from models.qwen3.weights import load_weights_w8a32_from_safetensors
            skip_lm_head = quantize not in {"w8a32-all", "w4a16-all"}
            quantize_policy = "static" if quantize in {"w8a32-static", "w4a16-static", "w4a16g32-static", "w4a16g128-static"} else "all"
            if quantize in {"w4a16-all", "w4a16-static"}:
                quant_format = "w4a16"
            elif quantize == "w4a16g32-static":
                quant_format = "w4a16g32"
            elif quantize == "w4a16g128-static":
                quant_format = "w4a16g128"
            else:
                quant_format = "w8a32"
            self.quantization_policy = quantize_policy
            with torch.no_grad():
                loaded, missing, quantized, cached = load_weights_w8a32_from_safetensors(
                    self.model,
                    model_dir,
                    skip_lm_head=skip_lm_head,
                    quantize_policy=quantize_policy,
                    quant_format=quant_format,
            )
            self.quantized_linear_modules = quantized
            self.cached_linear_modules = cached
            self.bf16_linear_modules = self._count_bf16_linear_modules()
            print(
                f"[engine] directly quantized {quantized} Linear tensors to {quant_format.upper()} "
                f"({cached} from cache, policy={quantize_policy}, bf16_linear={self.bf16_linear_modules})"
            )
            if quant_format in {"w8a32", "w4a16"} and os.environ.get("PILM_FUSE_SWIGLU", "1") != "0":
                from runtime.quantize import fuse_quantized_swiglu_modules
                fused = fuse_quantized_swiglu_modules(self.model)
                self.fused_swiglu_modules = fused
                if fused:
                    print(f"[engine] fused {fused} quantized SwiGLU blocks")
        else:
            with torch.no_grad():
                loaded, missing = weight_loader(self.model, model_dir)
        print(f"[engine] loaded {loaded} weight tensors")
        if missing:
            print(f"[engine] WARNING: {len(missing)} issues: {missing[:3]}")
        if quantize:
            if quantize not in {"w8a32", "w8a32-all", "w8a32-static", "w4a16-all", "w4a16-static", "w4a16g32-static", "w4a16g128-static"}:
                raise ValueError(f"unsupported quantize mode: {quantize}")
            if os.environ.get("PILM_TRIM_WORKING_SET", "1") != "0":
                from runtime.memory import trim_process_working_set
                trimmed = trim_process_working_set()
                if trimmed:
                    print("[engine] trimmed process working set after quantization")
            else:
                print("[engine] skipped process working set trim after quantization")

        self.tokenizer = SimpleTokenizer(model_dir)
        if os.environ.get("PILM_MTP_DRAFT", "0") == "1":
            try:
                from models.qwen3.mtp import load_mtp_draft
                self.mtp_draft = load_mtp_draft(model_dir, self.config)
                mtp_q = int(getattr(self.mtp_draft, "quantized_linear_modules", 0))
                print(f"[engine] loaded MTP draft module (quantized_linears={mtp_q})")
            except Exception as exc:
                self.mtp_draft = None
                print(f"[engine] WARNING: failed to load MTP draft module: {type(exc).__name__}: {exc}")

        tc = self.config.text_config
        block_size = DEFAULT_BLOCK_SIZE
        if kv_cache_gb is None:
            kv_cache_gb = float(os.environ.get("PILM_KV_CACHE_GB", "1"))
        self.kv_cache_gb = kv_cache_gb
        avail_bytes = max(1, int(kv_cache_gb * (1024**3)))
        per_block_per_layer = block_size * tc.num_key_value_heads * tc.head_dim * 2 * 2
        num_blocks = max(64, avail_bytes // (per_block_per_layer * tc.num_hidden_layers))
        print(f"[engine] KV cache: {num_blocks} blocks x {block_size} tokens, "
              f"{num_blocks * per_block_per_layer * tc.num_hidden_layers / (1024**3):.2f} GB")

        self.kv_manager = KVCacheManager(
            num_blocks,
            block_size,
            enable_caching=os.environ.get("PILM_PREFIX_CACHE", "0") == "1",
        )
        self.scheduler = Scheduler(self.kv_manager, max_num_batched_tokens=block_size)
        self.physical_kv = PhysicalKVCache(
            num_layers=tc.num_hidden_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=tc.num_key_value_heads,
            head_dim=tc.head_dim,
            dtype=self.dtype,
        )
        self.block_size = block_size

        self._req_counter = 0
        self.last_stats = {}
        print(f"[engine] ready.")

    def _detect_ckernel(self) -> bool:
        try:
            import _abi as abi
        except ImportError:
            try:
                from . import _abi as abi
            except ImportError:
                return False
        try:
            info = abi.runtime_info()
        except Exception as exc:
            self.ckernel_info = {"error": f"{type(exc).__name__}: {exc}"}
            return False
        self.ckernel_info = info
        symbols = info.get("symbols", {})
        if self.quantize in {"w4a16-all", "w4a16-static"}:
            return bool(symbols.get("ekernel_linear_w4a16_bf16"))
        if self.quantize == "w4a16g32-static":
            return bool(symbols.get("ekernel_linear_w4a16g32_bf16"))
        if self.quantize == "w4a16g128-static":
            return bool(symbols.get("ekernel_linear_w4a16g128_bf16"))
        if self.quantize in {"w8a32", "w8a32-all", "w8a32-static"} and self.dtype == torch.bfloat16:
            return bool(symbols.get("ekernel_linear_w8a16_bf16"))
        if self.quantize in {"w8a32", "w8a32-all", "w8a32-static"}:
            return bool(symbols.get("ekernel_linear_w8a32"))
        return bool(symbols.get("ekernel_gemm"))

    def _count_bf16_linear_modules(self) -> int:
        try:
            from models.base.linear import BackendLinear
        except ImportError:
            from .models.base.linear import BackendLinear
        return sum(1 for module in self.model.modules() if isinstance(module, BackendLinear))

    @property
    def kernel_backend(self) -> str:
        quantize = getattr(self, "quantize", "none")
        dtype = getattr(self, "dtype", torch.float32)
        ckernel_available = getattr(self, "ckernel_available", False)
        if quantize in {"w4a16-all", "w4a16-static"} and dtype == torch.bfloat16 and ckernel_available:
            return "ckernel_w4a16_bf16"
        if quantize == "w4a16g32-static" and dtype == torch.bfloat16 and ckernel_available:
            return "ckernel_w4a16g32_bf16"
        if quantize == "w4a16g128-static" and dtype == torch.bfloat16 and ckernel_available:
            return "ckernel_w4a16g128_bf16"
        if quantize in {"w8a32", "w8a32-all", "w8a32-static"} and dtype == torch.bfloat16 and ckernel_available:
            return "ckernel_w8a16_bf16"
        if quantize in {"w8a32", "w8a32-all", "w8a32-static"} and ckernel_available:
            return "ckernel_w8a32"
        return os.environ.get("PILM_LINEAR_BACKEND", "torch")

    def runtime_status(self) -> dict:
        return {
            "dtype": str(getattr(self, "dtype", torch.float32)).replace("torch.", ""),
            "quantize": getattr(self, "quantize", "none"),
            "quantization_policy": getattr(self, "quantization_policy", "none"),
            "kernel_backend": self.kernel_backend,
            "ckernel_available": getattr(self, "ckernel_available", False),
            "ckernel_info": getattr(self, "ckernel_info", {}),
            "quantized_linear_modules": getattr(self, "quantized_linear_modules", 0),
            "cached_linear_modules": getattr(self, "cached_linear_modules", 0),
            "bf16_linear_modules": getattr(self, "bf16_linear_modules", 0),
            "fused_swiglu_modules": getattr(self, "fused_swiglu_modules", 0),
            "fuse_swiglu": os.environ.get("PILM_FUSE_SWIGLU", "1") != "0",
            "norm_backend": os.environ.get("PILM_NORM_BACKEND", "torch") or "torch",
            "w8a16_m_flat": os.environ.get("ECPU_W8A16_M_FLAT", "0") == "1",
            "trim_working_set": os.environ.get("PILM_TRIM_WORKING_SET", "1") != "0",
            "prefix_cache_enabled": os.environ.get("PILM_PREFIX_CACHE", "0") == "1",
            "kv_cache_gb": getattr(self, "kv_cache_gb", None),
            "mtp_draft_enabled": self.mtp_draft is not None,
            "mtp_quantized_linear_modules": int(getattr(self.mtp_draft, "quantized_linear_modules", 0)) if self.mtp_draft is not None else 0,
        }

    def new_session_state(self) -> EngineSessionState:
        return EngineSessionState()

    def _can_use_greedy_lm_head_argmax(self, temperature: float, top_k: int) -> bool:
        if temperature > 0 or top_k != 0:
            return False
        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is None or not hasattr(lm_head, "argmax_bf16"):
            return False
        if self.dtype != torch.bfloat16:
            return False
        return os.environ.get("PILM_LM_HEAD_ARGMAX", "0") == "1"

    def generate(
        self,
        prompt: Union[str, list],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 40,
        session_state: Optional[EngineSessionState] = None,
    ) -> Generator[str, None, None]:
        if isinstance(prompt, list) and all(isinstance(item, int) for item in prompt):
            input_ids = list(prompt)
        else:
            formatted_prompt = self.tokenizer.format_prompt(prompt)
            input_ids = self.tokenizer.encode(formatted_prompt)
        if not input_ids:
            return

        num_layers = self.config.num_layers
        reused_tokens = 0
        appended_prompt_tokens = len(input_ids)
        keep_state = session_state is not None

        if session_state and session_state.request is not None:
            existing = session_state.request
            reused_tokens = _common_prefix_len(existing.all_token_ids, input_ids)
            if reused_tokens == len(existing.all_token_ids) and reused_tokens <= len(input_ids):
                req = existing
                suffix = input_ids[reused_tokens:]
                state_reused_tokens = req.num_computed_tokens
                req.prompt_token_ids = list(input_ids)
                req._all_token_ids.extend(suffix)
                req._output_token_ids = []
                req.max_new_tokens = max_new_tokens
                req.eos_token_id = self.tokenizer.eos_id
                req.status = RequestStatus.RUNNING
                req._update_block_hashes(self.block_size)
                reused_tokens = state_reused_tokens
                appended_prompt_tokens = len(input_ids) - reused_tokens
                conv_states = session_state.conv_states or [None] * num_layers
                recurrent_states = session_state.recurrent_states or [None] * num_layers
                self.scheduler.running.append(req)
                self.scheduler.requests[req.req_id] = req
            else:
                self._release_session_state(session_state)
                reused_tokens = 0
                appended_prompt_tokens = len(input_ids)
                req, conv_states, recurrent_states = self._new_request(input_ids, max_new_tokens, num_layers)
                session_state.request = req
        else:
            req, conv_states, recurrent_states = self._new_request(input_ids, max_new_tokens, num_layers)
            if session_state is not None:
                session_state.request = req

        generated_text_parts = []
        use_lm_head_argmax = self._can_use_greedy_lm_head_argmax(temperature, top_k)
        use_mtp_draft = self.mtp_draft is not None and temperature == 0 and top_k == 0
        use_mtp_speculative = (
            use_mtp_draft
            and os.environ.get("PILM_MTP_SPECULATIVE", "0") == "1"
            and session_state is None
            and not use_lm_head_argmax
        )
        mtp_speculative_steps = max(1, int(os.environ.get("PILM_MTP_SPECULATIVE_STEPS", "3")))
        last_mtp_draft_id = None
        pending_mtp_draft_ids: List[int] = []
        mtp_draft_attempts = 0
        mtp_draft_matches = 0
        mtp_speculative_accepts = 0
        mtp_speculative_rejects = 0
        mtp_speculative_appended_drafts = 0
        for step in range(max_new_tokens + 1):
            speculative_draft_appended = False
            appended_mtp_draft_ids: List[int] = []
            if (
                use_mtp_speculative
                and pending_mtp_draft_ids
                and len(generated_text_parts) < max_new_tokens
                and req in self.scheduler.running
                and req.num_uncomputed == 1
                and req.num_computed_tokens >= req.num_prompt_tokens
            ):
                block_remaining = self.block_size - (req.num_computed_tokens % self.block_size)
                remaining_output = max_new_tokens - len(generated_text_parts)
                draft_limit = max(0, min(remaining_output, block_remaining - 1, len(pending_mtp_draft_ids)))
                if draft_limit > 0:
                    appended_mtp_draft_ids = [int(tok) for tok in pending_mtp_draft_ids[:draft_limit]]
                    req._all_token_ids.extend(appended_mtp_draft_ids)
                    req._update_block_hashes(self.block_size)
                    pending_mtp_draft_ids = []
                    speculative_draft_appended = True
                    mtp_speculative_appended_drafts += len(appended_mtp_draft_ids)

            out = self.scheduler.schedule()
            if out.is_empty:
                break

            generated_tokens = {}
            stop_generation = False
            for sreq in out.scheduled:
                r = next(r for r in self.scheduler.running if r.req_id == sreq.req_id)
                is_decode = not sreq.is_prefill
                start_pos = sreq.positions[0] if sreq.positions else r.num_computed_tokens - sreq.num_tokens
                token_ids = r.all_token_ids[start_pos:start_pos + sreq.num_tokens]
                if is_decode and not token_ids:
                    token_ids = [r.output_token_ids[-1]] if r.output_token_ids else [input_ids[-1]]

                ids_tensor = torch.tensor(token_ids, dtype=torch.long)
                pos_tensor = torch.tensor(sreq.positions, dtype=torch.long)

                if not sreq.block_ids:
                    continue

                if is_decode:
                    block_id = sreq.block_ids[-1]
                    slot_offset = (r.num_computed_tokens - 1) % self.block_size if sreq.num_tokens == 1 else start_pos % self.block_size
                else:
                    block_index = start_pos // self.block_size
                    block_id = sreq.block_ids[block_index] if block_index < len(sreq.block_ids) else sreq.block_ids[-1]
                    slot_offset = start_pos % self.block_size

                speculative_verify = (
                    use_mtp_speculative
                    and speculative_draft_appended
                    and appended_mtp_draft_ids
                    and sreq.num_tokens == 1 + len(appended_mtp_draft_ids)
                    and len(token_ids) == 1 + len(appended_mtp_draft_ids)
                    and [int(tok) for tok in token_ids[1:]] == appended_mtp_draft_ids
                )
                saved_conv_states = None
                saved_recurrent_states = None
                if speculative_verify:
                    saved_conv_states = [
                        state.clone() if isinstance(state, torch.Tensor) else None
                        for state in conv_states
                    ]
                    saved_recurrent_states = [
                        state.clone() if isinstance(state, torch.Tensor) else None
                        for state in recurrent_states
                    ]

                with torch.no_grad():
                    model_out, new_convs, new_recs = self.model(
                        input_ids=ids_tensor,
                        positions=pos_tensor,
                        kv_caches_k=self.physical_kv.k_cache,
                        kv_caches_v=self.physical_kv.v_cache,
                        block_id=block_id,
                        slot_offset=slot_offset,
                        is_decode=is_decode,
                        block_ids=sreq.block_ids,
                        conv_states=conv_states,
                        recurrent_states=recurrent_states,
                        logits_last_only=not speculative_verify,
                        return_last_hidden=use_lm_head_argmax or use_mtp_draft,
                    )
                    conv_states = new_convs
                    recurrent_states = new_recs

                if sreq.is_prefill_chunk:
                    continue

                if speculative_verify:
                    logits = self.model.lm_head(model_out).to(torch.float32)
                    accepted_count = 0
                    rejected_id = None
                    for draft_idx, draft_id in enumerate(appended_mtp_draft_ids):
                        verified_id = int(torch.argmax(logits[draft_idx]).item())
                        mtp_draft_attempts += 1
                        if verified_id != int(draft_id):
                            rejected_id = verified_id
                            mtp_speculative_rejects += 1
                            break
                        mtp_draft_matches += 1
                        mtp_speculative_accepts += 1
                        accepted_count += 1
                        r._output_token_ids.append(int(draft_id))
                        draft_text = self.tokenizer.decode([int(draft_id)])
                        generated_text_parts.append(draft_text)
                        yield draft_text
                        if int(draft_id) == self.tokenizer.eos_id or len(generated_text_parts) >= max_new_tokens:
                            stop_generation = True
                            break

                    if stop_generation:
                        pending_mtp_draft_ids = []
                        break

                    if rejected_id is not None:
                        prefix_len = 1 + accepted_count
                        del r._all_token_ids[start_pos + prefix_len:]
                        r.block_hashes = []
                        r._update_block_hashes(self.block_size)
                        r.num_computed_tokens = start_pos + prefix_len
                        with torch.no_grad():
                            _hidden_after_prefix, conv_states, recurrent_states = self.model(
                                input_ids=ids_tensor[:prefix_len],
                                positions=pos_tensor[:prefix_len],
                                kv_caches_k=self.physical_kv.k_cache,
                                kv_caches_v=self.physical_kv.v_cache,
                                block_id=block_id,
                                slot_offset=slot_offset,
                                is_decode=is_decode,
                                block_ids=sreq.block_ids,
                                conv_states=saved_conv_states,
                                recurrent_states=saved_recurrent_states,
                                logits_last_only=True,
                                return_last_hidden=True,
                            )
                        next_id = int(rejected_id)
                        with torch.no_grad():
                            drafts, _ = self.mtp_draft.draft_tokens(
                                _hidden_after_prefix[-1],
                                next_id,
                                int(pos_tensor[prefix_len - 1].item()) + 1,
                                self.model.embed_tokens,
                                self.model.lm_head,
                                num_tokens=mtp_speculative_steps,
                            )
                            pending_mtp_draft_ids = [int(tok) for tok in drafts]
                            last_mtp_draft_id = pending_mtp_draft_ids[-1] if pending_mtp_draft_ids else None
                        generated_tokens.setdefault(r.req_id, []).append(next_id)
                        text = self.tokenizer.decode([next_id])
                        generated_text_parts.append(text)
                        yield text
                        if next_id == self.tokenizer.eos_id:
                            stop_generation = True
                            break
                        continue

                    if len(generated_text_parts) >= max_new_tokens:
                        pending_mtp_draft_ids = []
                        continue

                    next_id = int(torch.argmax(logits[len(appended_mtp_draft_ids)]).item())
                    with torch.no_grad():
                        drafts, _ = self.mtp_draft.draft_tokens(
                            model_out[-1],
                            next_id,
                            int(pos_tensor[-1].item()) + 1,
                            self.model.embed_tokens,
                            self.model.lm_head,
                            num_tokens=mtp_speculative_steps,
                        )
                        pending_mtp_draft_ids = [int(tok) for tok in drafts]
                        last_mtp_draft_id = pending_mtp_draft_ids[-1] if pending_mtp_draft_ids else None
                    generated_tokens.setdefault(r.req_id, []).append(next_id)
                    text = self.tokenizer.decode([next_id])
                    generated_text_parts.append(text)
                    yield text
                    if next_id == self.tokenizer.eos_id:
                        stop_generation = True
                        break
                    continue

                if use_lm_head_argmax:
                    next_id = self.model.lm_head.argmax_bf16(model_out[-1])
                else:
                    if use_mtp_draft:
                        next_logits = self.model.lm_head(model_out)[-1].to(torch.float32)
                    else:
                        next_logits = model_out[-1].to(torch.float32)
                    if temperature > 0:
                        next_logits = next_logits / temperature
                        if top_k > 0:
                            topk_vals, topk_idx = torch.topk(next_logits, min(top_k, next_logits.shape[-1]))
                            probs = torch.softmax(topk_vals, dim=-1)
                            choice = torch.multinomial(probs, 1)
                            next_id = topk_idx[choice].item()
                        else:
                            probs = torch.softmax(next_logits, dim=-1)
                            next_id = torch.multinomial(probs, 1).item()
                    else:
                        next_id = torch.argmax(next_logits).item()

                if use_mtp_draft:
                    if pending_mtp_draft_ids and not use_mtp_speculative:
                        mtp_draft_attempts += 1
                        if int(pending_mtp_draft_ids[0]) == int(next_id):
                            mtp_draft_matches += 1
                    with torch.no_grad():
                        if use_mtp_speculative:
                            drafts, _ = self.mtp_draft.draft_tokens(
                                model_out[-1],
                                int(next_id),
                                int(pos_tensor[-1].item()) + 1,
                                self.model.embed_tokens,
                                self.model.lm_head,
                                num_tokens=mtp_speculative_steps,
                            )
                            pending_mtp_draft_ids = [int(tok) for tok in drafts]
                            last_mtp_draft_id = pending_mtp_draft_ids[-1] if pending_mtp_draft_ids else None
                        else:
                            token_tensor = torch.tensor([next_id], dtype=torch.long)
                            token_embedding = self.model.embed_tokens(token_tensor)[0]
                            mtp_position = torch.tensor([int(pos_tensor[-1].item()) + 1], dtype=torch.long)
                            last_mtp_draft_id = self.mtp_draft.argmax(
                                model_out[-1],
                                token_embedding,
                                mtp_position,
                                self.model.lm_head,
                            )
                            pending_mtp_draft_ids = [int(last_mtp_draft_id)]

                generated_tokens.setdefault(r.req_id, []).append(next_id)

                text = self.tokenizer.decode([next_id])
                generated_text_parts.append(text)
                yield text

                if next_id == self.tokenizer.eos_id:
                    stop_generation = True
                    break

            self.scheduler.update_from_output(out, generated_tokens, release_finished=not keep_state)
            if stop_generation:
                break

        self.scheduler.running.clear()
        self.scheduler.waiting = type(self.scheduler.waiting)()
        self.last_stats = {
            "prompt_tokens": len(input_ids),
            "generated_tokens": len(generated_text_parts),
            "reused_tokens": reused_tokens,
            "appended_prompt_tokens": appended_prompt_tokens,
            "kv_cache_blocks": self.kv_manager.pool.num_cached_blocks(),
            "kernel_backend": self.kernel_backend,
            "lm_head_argmax": use_lm_head_argmax,
            "mtp_draft": use_mtp_draft,
            "mtp_last_draft_id": last_mtp_draft_id,
            "mtp_draft_attempts": mtp_draft_attempts,
            "mtp_draft_matches": mtp_draft_matches,
            "mtp_draft_acceptance": (
                mtp_draft_matches / mtp_draft_attempts if mtp_draft_attempts else 0.0
            ),
            "mtp_speculative": use_mtp_speculative,
            "mtp_speculative_steps": mtp_speculative_steps if use_mtp_speculative else 0,
            "mtp_speculative_appended_drafts": mtp_speculative_appended_drafts,
            "mtp_speculative_accepts": mtp_speculative_accepts,
            "mtp_speculative_rejects": mtp_speculative_rejects,
        }
        if session_state is not None:
            session_state.request = req
            session_state.conv_states = conv_states
            session_state.recurrent_states = recurrent_states
            session_state.last_text = "".join(generated_text_parts)
            session_state.last_reused_tokens = reused_tokens
            session_state.last_appended_prompt_tokens = appended_prompt_tokens

    def _new_request(self, input_ids: List[int], max_new_tokens: int, num_layers: int):
        self._req_counter += 1
        req = Request(
            req_id=self._req_counter,
            prompt_token_ids=input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_id,
        )
        self.scheduler.add_request(req)
        return req, [None] * num_layers, [None] * num_layers

    def _release_session_state(self, session_state: EngineSessionState) -> None:
        if session_state.request is not None:
            self.kv_manager.free_request_blocks(session_state.request)
        session_state.request = None
        session_state.conv_states = None
        session_state.recurrent_states = None
