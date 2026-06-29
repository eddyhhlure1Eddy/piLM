"""Local persistent piLM HTTP server.

The server loads one Engine at startup and reuses it for all requests.
Generation is serialized because the current Engine owns one scheduler/KV cache.
"""
import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from .engine import Engine
except ImportError:
    from engine import Engine


class PiLMApp:
    def __init__(self, model_dir: str, max_new_tokens: int, quantize: str = "none", kv_cache_gb: float | None = None):
        load_start = time.perf_counter()
        self.engine = Engine(
            model_dir,
            dtype="bfloat16",
            quantize=None if quantize == "none" else quantize,
            kv_cache_gb=kv_cache_gb,
        )
        self.load_seconds = time.perf_counter() - load_start
        self.model_dir = model_dir
        self.quantize = quantize
        self.default_max_new_tokens = max_new_tokens
        self.lock = threading.Lock()
        self.sessions: Dict[str, List[dict]] = {}
        self.session_states: Dict[str, Any] = {}

    def generate(self, prompt_or_messages, max_new_tokens: int, temperature: float, top_k: int, session_id: str | None = None) -> dict:
        with self.lock:
            return self._generate_unlocked(prompt_or_messages, max_new_tokens, temperature, top_k, session_id)

    def _generate_unlocked(self, prompt_or_messages, max_new_tokens: int, temperature: float, top_k: int, session_id: str | None = None) -> dict:
        start = time.perf_counter()
        first_chunk_seconds = None
        chunks = []
        session_state = None
        if session_id is not None:
            session_state = self.session_states.setdefault(session_id, self.engine.new_session_state())
        for chunk in self.engine.generate(
            prompt_or_messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            session_state=session_state,
        ):
            if first_chunk_seconds is None:
                first_chunk_seconds = time.perf_counter() - start
            chunks.append(chunk)
        total_seconds = time.perf_counter() - start
        text = "".join(chunks)
        return {
            "text": text,
            "generated_chunks": len(chunks),
            "first_chunk_seconds": first_chunk_seconds,
            "total_seconds": total_seconds,
            "chunks_per_second": (len(chunks) / total_seconds) if total_seconds > 0 else 0.0,
            "engine_stats": dict(self.engine.last_stats),
        }

    def clear_session(self, session_id: str) -> None:
        state = self.session_states.pop(session_id, None)
        if state is not None:
            self.engine._release_session_state(state)
        self.sessions.pop(session_id, None)

    def profile(
        self,
        prompt_or_messages,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_modules: int = 12,
        linear_stages: bool = False,
    ) -> dict:
        stats = defaultdict(float)
        counts = defaultdict(int)
        restore = []
        linear_stage_profile = None
        reset_linear_stage_profile = None
        get_linear_stage_profile = None
        attention_stage_profile = None
        reset_attention_stage_profile = None
        get_attention_stage_profile = None
        if linear_stages:
            try:
                from models.base.linear import reset_linear_stage_profile, linear_stage_profile as get_linear_stage_profile
                from models.qwen3.attention import reset_attention_stage_profile, attention_stage_profile as get_attention_stage_profile
            except ImportError:
                from .models.base.linear import reset_linear_stage_profile, linear_stage_profile as get_linear_stage_profile
                from .models.qwen3.attention import reset_attention_stage_profile, attention_stage_profile as get_attention_stage_profile

        def wrap(name, module):
            original_forward = module.forward
            restore.append((module, original_forward))

            def timed_forward(*args, _name=name, _original=original_forward, **kwargs):
                start = time.perf_counter()
                try:
                    return _original(*args, **kwargs)
                finally:
                    stats[_name] += time.perf_counter() - start
                    counts[_name] += 1

            module.forward = timed_forward

        with self.lock:
            try:
                if reset_linear_stage_profile is not None:
                    reset_linear_stage_profile(True)
                if reset_attention_stage_profile is not None:
                    reset_attention_stage_profile(True)
                for idx, layer in enumerate(self.engine.model.layers):
                    if hasattr(layer, "self_attn"):
                        wrap(f"layers.{idx}.full_attention", layer.self_attn)
                    if hasattr(layer, "linear_attn"):
                        wrap(f"layers.{idx}.linear_attention", layer.linear_attn)
                    wrap(f"layers.{idx}.mlp", layer.mlp)
                wrap("lm_head", self.engine.model.lm_head)
                result = self._generate_unlocked(prompt_or_messages, max_new_tokens, temperature, top_k)
                if get_linear_stage_profile is not None:
                    linear_stage_profile = get_linear_stage_profile()
                if get_attention_stage_profile is not None:
                    attention_stage_profile = get_attention_stage_profile()
            finally:
                for module, original_forward in restore:
                    module.forward = original_forward
                if reset_linear_stage_profile is not None:
                    reset_linear_stage_profile(False)
                if reset_attention_stage_profile is not None:
                    reset_attention_stage_profile(False)

        by_kind = defaultdict(float)
        profile = []
        for name, seconds in sorted(stats.items(), key=lambda item: item[1], reverse=True):
            if ".full_attention" in name:
                kind = "full_attention"
            elif ".linear_attention" in name:
                kind = "linear_attention"
            elif ".mlp" in name:
                kind = "mlp"
            else:
                kind = name
            by_kind[kind] += seconds
            profile.append({
                "module": name,
                "calls": counts[name],
                "seconds": round(seconds, 4),
                "avg_seconds": round(seconds / counts[name], 4) if counts[name] else 0.0,
            })
        result["module_profile"] = profile[:top_modules]
        result["profiled_module_seconds"] = round(sum(stats.values()), 4)
        result["module_profile_by_kind"] = {
            key: round(value, 4)
            for key, value in sorted(by_kind.items(), key=lambda item: item[1], reverse=True)
        }
        if linear_stage_profile is not None:
            result["linear_stage_profile"] = linear_stage_profile
        if attention_stage_profile is not None:
            result["attention_stage_profile"] = attention_stage_profile
        return result


APP: PiLMApp | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "piLMHTTP/0.1"

    def _send_json(self, status: int, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health":
            assert APP is not None
            try:
                import torch
                torch_threads = torch.get_num_threads()
                torch_interop_threads = torch.get_num_interop_threads()
            except Exception:
                torch_threads = None
                torch_interop_threads = None
            self._send_json(200, {
                "ok": True,
                "model_dir": APP.model_dir,
                "quantize": APP.quantize,
                "omp_threads": os.environ.get("OMP_NUM_THREADS"),
                "torch_threads": torch_threads,
                "torch_interop_threads": torch_interop_threads,
                "load_seconds": round(APP.load_seconds, 4),
                "kv_cache_gb": APP.engine.kv_cache_gb,
                "sessions": len(APP.sessions),
                "session_states": len(APP.session_states),
                "prefix_cache_blocks": APP.engine.kv_manager.pool.num_cached_blocks(),
                "free_blocks": APP.engine.kv_manager.num_free_blocks,
                "runtime": APP.engine.runtime_status(),
            })
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        assert APP is not None
        try:
            payload = self._read_json()
            if self.path == "/generate":
                self._handle_generate(payload)
            elif self.path == "/chat":
                self._handle_chat(payload)
            elif self.path == "/clear":
                session_id = str(payload.get("session_id", "default"))
                APP.clear_session(session_id)
                self._send_json(200, {"ok": True, "session_id": session_id})
            elif self.path == "/profile":
                self._handle_profile(payload)
            elif self.path == "/v1/chat/completions":
                self._handle_chat_completions(payload)
            else:
                self._send_json(404, {"error": "not found"})
        except Exception as exc:
            self._send_json(500, {"error": type(exc).__name__, "message": str(exc)})

    def _sampling_args(self, payload: dict) -> tuple[int, float, int]:
        max_new_tokens = int(payload.get("max_new_tokens", payload.get("max_tokens", APP.default_max_new_tokens)))
        temperature = float(payload.get("temperature", 0.7))
        top_k = int(payload.get("top_k", 40))
        return max_new_tokens, temperature, top_k

    def _handle_generate(self, payload: dict) -> None:
        prompt = payload.get("prompt")
        messages = payload.get("messages")
        if messages is None and prompt is None:
            self._send_json(400, {"error": "prompt or messages is required"})
            return
        max_new_tokens, temperature, top_k = self._sampling_args(payload)
        result = APP.generate(messages if messages is not None else str(prompt), max_new_tokens, temperature, top_k)
        self._send_json(200, {
            "ok": True,
            **_rounded_timings(result),
        })

    def _handle_chat(self, payload: dict) -> None:
        message = payload.get("message")
        if not message:
            self._send_json(400, {"error": "message is required"})
            return
        session_id = str(payload.get("session_id", "default"))
        history = APP.sessions.setdefault(session_id, [])
        history.append({"role": "user", "content": str(message)})
        max_new_tokens, temperature, top_k = self._sampling_args(payload)
        prompt_or_ids = history
        state = APP.session_states.get(session_id)
        if state is not None and state.request is not None:
            suffix = _chat_turn_suffix(str(message))
            prompt_or_ids = state.request.all_token_ids + APP.engine.tokenizer.encode(suffix)
        result = APP.generate(prompt_or_ids, max_new_tokens, temperature, top_k, session_id=session_id)
        history.append({"role": "assistant", "content": _assistant_history_content(result["text"])})
        self._send_json(200, {
            "ok": True,
            "session_id": session_id,
            "messages": len(history),
            **_rounded_timings(result),
        })

    def _handle_profile(self, payload: dict) -> None:
        prompt = payload.get("prompt")
        messages = payload.get("messages")
        if messages is None and prompt is None:
            self._send_json(400, {"error": "prompt or messages is required"})
            return
        max_new_tokens, temperature, top_k = self._sampling_args(payload)
        top_modules = int(payload.get("top_modules", 12))
        linear_stages = bool(payload.get("linear_stages", False))
        result = APP.profile(
            messages if messages is not None else str(prompt),
            max_new_tokens,
            temperature,
            top_k,
            top_modules,
            linear_stages=linear_stages,
        )
        self._send_json(200, {
            "ok": True,
            **_rounded_timings(result),
        })

    def _handle_chat_completions(self, payload: dict) -> None:
        messages = payload.get("messages")
        if not isinstance(messages, list):
            self._send_json(400, {"error": "messages list is required"})
            return
        max_new_tokens, temperature, top_k = self._sampling_args(payload)
        result = APP.generate(messages, max_new_tokens, temperature, top_k)
        created = int(time.time())
        self._send_json(200, {
            "id": f"chatcmpl-{created}",
            "object": "chat.completion",
            "created": created,
            "model": payload.get("model", "piLM-local"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": "stop",
            }],
            "piLM": _rounded_timings(result),
        })

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[server] " + (fmt % args) + "\n")


def _rounded_timings(result: dict) -> dict:
    out = dict(result)
    for key in ("first_chunk_seconds", "total_seconds", "chunks_per_second"):
        if out.get(key) is not None:
            out[key] = round(out[key], 4)
    if isinstance(out.get("engine_stats"), dict):
        out["engine_stats"] = {
            key: round(value, 4) if isinstance(value, float) else value
            for key, value in out["engine_stats"].items()
        }
    return out


def _assistant_history_content(text: str) -> str:
    if text.startswith("<think>") or text.startswith("<|im_start|>"):
        return text
    return "<think>\n" + text


def _chat_turn_suffix(user_text: str) -> str:
    return (
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_text}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m piLM.server")
    parser.add_argument("model_dir", nargs="?", default=r"D:\Qwen3.5-9B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8028)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--quantize", choices=["none", "w8a32", "w8a32-all", "w8a32-static", "w4a16-all", "w4a16-static", "w4a16g32-static", "w4a16g128-static"], default="none")
    parser.add_argument("--norm-backend", choices=["torch", "ckernel", "ckernel-all"], default="torch")
    parser.add_argument("--omp-threads", default="16")
    parser.add_argument("--torch-interop-threads", default=os.environ.get("PILM_TORCH_INTEROP_THREADS", "1"))
    parser.add_argument("--w8a16-m-flat", action="store_true", help="enable experimental flat OpenMP scheduling for M>1 W8A16")
    parser.add_argument("--kv-cache-gb", type=float, default=None)
    parser.add_argument("--w8a32-cache", action="store_true", help="load/save quantized Linear safetensors cache")
    parser.add_argument("--w8a32-cache-dir", default=None)
    parser.add_argument("--w4a16-cache", action="store_true", help="load/save W4A16 quantized Linear safetensors cache")
    parser.add_argument("--w4a16-cache-dir", default=None)
    return parser.parse_args()


def main() -> None:
    global APP
    args = _parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)
    if args.w8a32_cache:
        os.environ["PILM_W8A32_CACHE"] = "1"
    if args.w8a32_cache_dir:
        os.environ["PILM_W8A32_CACHE_DIR"] = args.w8a32_cache_dir
    if args.w4a16_cache:
        os.environ["PILM_W4A16_CACHE"] = "1"
    if args.w4a16_cache_dir:
        os.environ["PILM_W4A16_CACHE_DIR"] = args.w4a16_cache_dir
    if args.norm_backend == "torch":
        os.environ.pop("PILM_NORM_BACKEND", None)
    else:
        os.environ["PILM_NORM_BACKEND"] = args.norm_backend
    if args.w8a16_m_flat:
        os.environ["ECPU_W8A16_M_FLAT"] = "1"
    try:
        import torch
        torch.set_num_threads(int(args.omp_threads))
        torch.set_num_interop_threads(int(args.torch_interop_threads))
    except Exception:
        pass
    APP = PiLMApp(args.model_dir, args.max_new_tokens, args.quantize, args.kv_cache_gb)
    httpd = HTTPServer((args.host, args.port), Handler)
    print(f"[server] listening on http://{args.host}:{args.port}")
    print("[server] endpoints: GET /health, POST /generate, POST /chat, POST /profile, POST /v1/chat/completions")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] stopped")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
