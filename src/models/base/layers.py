"""Shared base layers: RMSNorm, GemmaRMSNorm, RoPE, SwiGLU - model-agnostic."""
import os
import torch
import torch.nn as nn
import math
from collections import OrderedDict
from typing import Optional, Tuple
from models.base.linear import BackendLinear

_ROPE_INV_FREQ_CACHE: dict[tuple[str, int, float], torch.Tensor] = {}
_ROPE_SINGLE_POS_CACHE: OrderedDict[tuple[str, int, float, int], tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
_ROPE_SINGLE_POS_CACHE_LIMIT = 8192


def _ckernel_norm_mode() -> str:
    return os.environ.get("PILM_NORM_BACKEND", "").strip().lower()


def _rmsnorm_ckernel_or_none(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    add_one: bool,
) -> torch.Tensor | None:
    mode = _ckernel_norm_mode()
    if mode not in {"ckernel", "ckernel-all"}:
        return None
    if x.device.type != "cpu" or x.dtype != torch.bfloat16:
        return None
    if weight.device.type != "cpu" or weight.dtype != torch.bfloat16:
        return None
    rows = x.numel() // x.shape[-1]
    if mode == "ckernel" and rows != 1:
        return None
    try:
        try:
            from runtime.ckernel import rmsnorm_bf16
        except ImportError:
            from ...runtime.ckernel import rmsnorm_bf16
    except ImportError:
        return None
    try:
        return rmsnorm_bf16(x, weight, eps, add_one=add_one)
    except RuntimeError as exc:
        if "ekernel_rmsnorm_bf16" in str(exc):
            return None
        raise


def _rope_inv_freq(device: torch.device, n_freq: int, theta: float) -> torch.Tensor:
    key = (str(device), int(n_freq), float(theta))
    cached = _ROPE_INV_FREQ_CACHE.get(key)
    if cached is not None:
        return cached
    inv_freq = 1.0 / (theta ** (torch.arange(0, n_freq, dtype=torch.float32, device=device) / n_freq))
    _ROPE_INV_FREQ_CACHE[key] = inv_freq
    return inv_freq


def _rope_cos_sin(
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    pos = positions.to(torch.float32).reshape(-1)
    if pos.numel() == 1:
        pos_int = int(positions.reshape(-1)[0].item())
        key = (str(inv_freq.device), int(inv_freq.numel()), float(theta), pos_int)
        cached = _ROPE_SINGLE_POS_CACHE.get(key)
        if cached is not None:
            _ROPE_SINGLE_POS_CACHE.move_to_end(key)
            return cached
        freqs = pos * inv_freq
        out = (torch.cos(freqs).reshape(1, -1), torch.sin(freqs).reshape(1, -1))
        _ROPE_SINGLE_POS_CACHE[key] = out
        if len(_ROPE_SINGLE_POS_CACHE) > _ROPE_SINGLE_POS_CACHE_LIMIT:
            _ROPE_SINGLE_POS_CACHE.popitem(last=False)
        return out
    freqs = pos.unsqueeze(1) * inv_freq.unsqueeze(0)
    return torch.cos(freqs), torch.sin(freqs)


class RMSNorm(nn.Module):
    """Standard RMSNorm: x * weight / rms(x). Used for ssm_norm (linear attn output)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c_out = _rmsnorm_ckernel_or_none(x, self.weight, self.eps, add_one=False)
        if c_out is not None:
            return c_out
        orig = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x.to(orig))


class GemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm: x * (weight + 1) / rms(x). Used for all norms except ssm_norm.

    Checkpoint stores weight centered around 0; effective scale = weight + 1.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c_out = _rmsnorm_ckernel_or_none(x, self.weight, self.eps, add_one=True)
        if c_out is not None:
            return c_out
        orig = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return ((self.weight + 1.0) * x.to(orig))


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    theta: float = 10000000.0,
    partial_rotary_factor: float = 0.25,
    mrope_section: Optional[list] = None,
    interleaved: bool = True,
    use_mrope: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply partial RoPE (NeoX half-split style for Qwen3.5 text-only).

    Standard RoPE on first rot_dim = head_dim*partial_rotary_factor dims.
    Half-split: first half and second half form pairs (x[i], x[i+n_freq]).
    """
    squeeze = False
    if q.dim() == 2:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        squeeze = True

    seq, n_heads, head_dim = q.shape
    rot_dim = int(head_dim * partial_rotary_factor)
    n_freq = rot_dim // 2

    inv_freq = _rope_inv_freq(q.device, n_freq, theta)
    cos, sin = _rope_cos_sin(positions, inv_freq, theta)

    q_rot = q[..., :rot_dim].to(torch.float32)
    k_rot = k[..., :rot_dim].to(torch.float32)
    q_pass = q[..., rot_dim:]
    k_pass = k[..., rot_dim:]

    cos_b = cos.unsqueeze(1)
    sin_b = sin.unsqueeze(1)

    q_first = q_rot[..., :n_freq]
    q_second = q_rot[..., n_freq:]
    k_first = k_rot[..., :n_freq]
    k_second = k_rot[..., n_freq:]

    q_out_first = q_first * cos_b - q_second * sin_b
    q_out_second = q_first * sin_b + q_second * cos_b
    k_out_first = k_first * cos_b - k_second * sin_b
    k_out_second = k_first * sin_b + k_second * cos_b

    q_rot_out = torch.cat([q_out_first, q_out_second], dim=-1)
    k_rot_out = torch.cat([k_out_first, k_out_second], dim=-1)

    q_out = torch.cat([q_rot_out.to(q.dtype), q_pass], dim=-1)
    k_out = torch.cat([k_rot_out.to(k.dtype), k_pass], dim=-1)

    if squeeze:
        return q_out.squeeze(0), k_out.squeeze(0)
    return q_out, k_out


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate = BackendLinear(hidden_size, intermediate_size, bias=False)
        self.up = BackendLinear(hidden_size, intermediate_size, bias=False)
        self.down = BackendLinear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))
