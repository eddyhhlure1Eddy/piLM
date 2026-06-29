"""piLM CLI entry point.

Usage:
    python -m piLM [model_dir] [prompt]
    python -m piLM [model_dir] --interactive
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from .engine import Engine
except ImportError:
    from engine import Engine


def _configure_console() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="piLM")
    parser.add_argument("model_dir", nargs="?", default=r"D:\Qwen3.5-9B")
    parser.add_argument("prompt", nargs="*", help="single-turn prompt")
    parser.add_argument("-i", "--interactive", action="store_true")
    parser.add_argument("--bench", action="store_true", help="run a local load/decode benchmark")
    parser.add_argument("--serve", action="store_true", help="start the persistent local HTTP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8028)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
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


def _generate_once(engine: Engine, messages, args: argparse.Namespace) -> str:
    full = ""
    for chunk in engine.generate(
        messages,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    ):
        full += chunk
        print(chunk, end="", flush=True)
    print()
    return full


def main():
    _configure_console()
    args = _parse_args()
    prompt = " ".join(args.prompt).strip()

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

    if args.bench:
        try:
            from .bench import run_benchmark
        except ImportError:
            from bench import run_benchmark
        result = run_benchmark(args)
        for key, value in result.items():
            print(f"{key}: {value}")
        return

    if args.serve:
        try:
            from . import server as server_module
        except ImportError:
            import server as server_module
        from http.server import HTTPServer

        server_module.APP = server_module.PiLMApp(args.model_dir, args.max_new_tokens, args.quantize, args.kv_cache_gb)
        httpd = HTTPServer((args.host, args.port), server_module.Handler)
        print(f"[server] listening on http://{args.host}:{args.port}")
        httpd.serve_forever()

    print(f"[piLM] model_dir={args.model_dir}")
    print("[piLM] initializing engine ...\n")
    engine = Engine(
        args.model_dir,
        dtype="bfloat16",
        quantize=None if args.quantize == "none" else args.quantize,
        kv_cache_gb=args.kv_cache_gb,
    )

    if prompt and not args.interactive:
        messages = [{"role": "user", "content": prompt}]
        print("[piLM] generating ...\n")
        _generate_once(engine, messages, args)
        return

    messages = []
    print("[piLM] interactive mode. Commands: /exit, /clear")
    while True:
        try:
            user_text = input("\nuser> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            break
        if user_text == "/clear":
            messages.clear()
            print("[piLM] history cleared")
            continue

        messages.append({"role": "user", "content": user_text})
        print("assistant> ", end="", flush=True)
        assistant_text = _generate_once(engine, messages, args)
        messages.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
    main()
