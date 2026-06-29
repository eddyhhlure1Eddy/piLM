"""Model registry: each model family lives in its own subpackage.

Register new models here:
    MODEL_REGISTRY[arch] = (model_class, weight_loader, config_parser)

Usage:
    from models import get_model, get_weight_loader
    ModelClass = get_model("qwen3")
    loader = get_weight_loader("qwen3")
"""
from typing import Callable, Tuple, Type
import torch.nn as nn


MODEL_REGISTRY: dict = {
    "qwen3": "models.qwen3",
    "qwen3_5": "models.qwen3",
}


def get_model(arch: str) -> Type[nn.Module]:
    if arch in ("qwen3", "qwen3_5"):
        from .qwen3 import Qwen3Model
        return Qwen3Model
    raise KeyError(f"Unknown model arch: {arch!r}. Registered: {list(MODEL_REGISTRY)}")


def get_weight_loader(arch: str) -> Callable:
    if arch in ("qwen3", "qwen3_5"):
        from .qwen3 import load_weights_from_safetensors
        return load_weights_from_safetensors
    raise KeyError(f"Unknown model arch: {arch!r}. Registered: {list(MODEL_REGISTRY)}")


def detect_arch(config) -> str:
    """Auto-detect model arch from parsed config."""
    tc = getattr(config, "text_config", config)
    if hasattr(tc, "linear_num_key_heads") or hasattr(tc, "layer_types"):
        return "qwen3"
    if getattr(tc, "model_type", "") in ("qwen3", "qwen3_5"):
        return "qwen3"
    raise ValueError(f"Cannot detect arch from config: {type(config).__name__}")


__all__ = ["MODEL_REGISTRY", "get_model", "get_weight_loader", "detect_arch"]