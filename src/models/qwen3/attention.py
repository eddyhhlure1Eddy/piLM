"""Attention layers for Qwen3.5 hybrid architecture.

FullAttention: q_proj outputs 2x (query + sigmoid gate), GQA with RoPE, q_norm/k_norm.
LinearAttention: Gated DeltaNet (ported from llama.cpp + vLLM).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time
from collections import defaultdict
from typing import Optional, Tuple
from models.base.linear import BackendLinear
from models.base.layers import RMSNorm, GemmaRMSNorm, apply_rope

_PROFILE_ATTENTION_STAGES = False
_ATTENTION_STAGE_SECONDS = defaultdict(float)
_ATTENTION_STAGE_COUNTS = defaultdict(int)


def reset_attention_stage_profile(enabled: bool = True) -> None:
    global _PROFILE_ATTENTION_STAGES
    _PROFILE_ATTENTION_STAGES = enabled
    _ATTENTION_STAGE_SECONDS.clear()
    _ATTENTION_STAGE_COUNTS.clear()


def attention_stage_profile() -> dict:
    profile = {}
    for name, seconds in sorted(_ATTENTION_STAGE_SECONDS.items(), key=lambda item: item[1], reverse=True):
        count = _ATTENTION_STAGE_COUNTS[name]
        profile[name] = {
            "calls": count,
            "seconds": round(seconds, 4),
            "avg_seconds": round(seconds / count, 6) if count else 0.0,
        }
    return profile


def _record_attention_stage(name: str, start: float) -> None:
    _ATTENTION_STAGE_SECONDS[name] += time.perf_counter() - start
    _ATTENTION_STAGE_COUNTS[name] += 1


class FullAttention(nn.Module):
    """Qwen3.5 full attention: q_proj outputs n_heads*head_dim*2 (query + gate)."""

    def __init__(self, hidden_size: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 rope_theta: float = 1e7, partial_rotary: float = 0.25,
                 mrope_section: Optional[list] = None, mrope_interleaved: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.partial_rotary = partial_rotary
        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved

        self.q_proj = BackendLinear(hidden_size, n_heads * head_dim * 2, bias=False)
        self.k_proj = BackendLinear(hidden_size, n_kv_heads * head_dim, bias=False)
        self.v_proj = BackendLinear(hidden_size, n_kv_heads * head_dim, bias=False)
        self.o_proj = BackendLinear(n_heads * head_dim, hidden_size, bias=False)
        self.q_norm = GemmaRMSNorm(head_dim)
        self.k_norm = GemmaRMSNorm(head_dim)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache_k: Optional[torch.Tensor] = None,
        kv_cache_v: Optional[torch.Tensor] = None,
        block_id: int = -1,
        slot_offset: int = 0,
        is_decode: bool = False,
        block_ids: Optional[list] = None,
    ) -> torch.Tensor:
        seq = x.shape[0]

        qg = self.q_proj(x).view(seq, self.n_heads, self.head_dim * 2)
        query_states, gate = torch.chunk(qg, 2, dim=-1)
        gate = gate.reshape(seq, -1)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(self.k_proj(x).view(seq, self.n_kv_heads, self.head_dim))
        value_states = self.v_proj(x).view(seq, self.n_kv_heads, self.head_dim)

        query_states, key_states = apply_rope(
            query_states, key_states, positions, self.rope_theta,
            self.partial_rotary, self.mrope_section, self.mrope_interleaved,
            use_mrope=True
        )

        if kv_cache_k is not None and block_id >= 0:
            kv_cache_k[block_id, slot_offset:slot_offset + seq] = key_states
            kv_cache_v[block_id, slot_offset:slot_offset + seq] = value_states
            if block_ids:
                cache_len = int(positions[-1].item()) + 1
                block_size = kv_cache_k.shape[1]
                needed = (cache_len + block_size - 1) // block_size
                cache_block_ids = block_ids[:needed]
                if len(cache_block_ids) == 1:
                    full_k = kv_cache_k[cache_block_ids[0], :cache_len]
                    full_v = kv_cache_v[cache_block_ids[0], :cache_len]
                else:
                    full_k = torch.cat([kv_cache_k[bid] for bid in cache_block_ids], dim=0)[:cache_len]
                    full_v = torch.cat([kv_cache_v[bid] for bid in cache_block_ids], dim=0)[:cache_len]
            elif is_decode and slot_offset > 0:
                full_k = kv_cache_k[block_id, :slot_offset + seq]
                full_v = kv_cache_v[block_id, :slot_offset + seq]
            else:
                full_k = key_states
                full_v = value_states
        else:
            full_k = key_states
            full_v = value_states

        if self.n_kv_heads < self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            full_k = full_k.unsqueeze(2).expand(-1, -1, rep, -1).reshape(full_k.shape[0], self.n_heads, self.head_dim)
            full_v = full_v.unsqueeze(2).expand(-1, -1, rep, -1).reshape(full_v.shape[0], self.n_heads, self.head_dim)

        q = query_states.transpose(0, 1)
        k_t = full_k.transpose(0, 1)
        v_t = full_v.transpose(0, 1)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.bmm(q, k_t.transpose(1, 2)) * scale

        if not is_decode or seq > 1:
            kv_len = full_k.shape[0]
            mask = torch.triu(torch.ones(seq, kv_len, dtype=torch.bool), diagonal=kv_len - seq + 1)
            scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))

        attn = torch.softmax(scores.to(torch.float32), dim=-1).to(v_t.dtype)
        out = torch.bmm(attn, v_t)
        out = out.transpose(0, 1).reshape(seq, -1)
        out = out * torch.sigmoid(gate)
        out = self.o_proj(out)
        return out


class LinearAttention(nn.Module):
    """Qwen3.5 Gated DeltaNet (ported from llama.cpp qwen35.cpp + ops.cpp gated_delta_net).

    Weights: in_proj_qkv (wqkv), in_proj_z (wqkv_gate), in_proj_a (ssm_alpha),
             in_proj_b (ssm_beta), conv1d (ssm_conv1d), A_log (ssm_a), dt_bias (ssm_dt),
             norm (ssm_norm), out_proj (ssm_out)
    """

    def __init__(self, hidden_size: int, n_k_heads: int, k_head_dim: int,
                 n_v_heads: int, v_head_dim: int, conv_kernel: int = 4,
                 rms_eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_k_heads = n_k_heads
        self.k_head_dim = k_head_dim
        self.n_v_heads = n_v_heads
        self.v_head_dim = v_head_dim
        self.key_dim = n_k_heads * k_head_dim
        self.value_dim = n_v_heads * v_head_dim
        self.conv_kernel = conv_kernel

        conv_dim = self.key_dim * 2 + self.value_dim
        self.in_proj_qkv = BackendLinear(hidden_size, conv_dim, bias=False)
        self.in_proj_z = BackendLinear(hidden_size, self.value_dim, bias=False)
        self.in_proj_b = BackendLinear(hidden_size, self.n_v_heads, bias=False)
        self.in_proj_a = BackendLinear(hidden_size, self.n_v_heads, bias=False)
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, conv_kernel, padding=conv_kernel - 1, groups=conv_dim, bias=False)
        _prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            self.A_log = nn.Parameter(torch.zeros(self.n_v_heads))
            self.dt_bias = nn.Parameter(torch.ones(self.n_v_heads))
        finally:
            torch.set_default_dtype(_prev)
        self.norm = RMSNorm(self.v_head_dim, rms_eps)
        self.out_proj = BackendLinear(self.value_dim, hidden_size, bias=False)

    def _l2_norm(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True)
        return x / (norm + eps)

    def _conv1d_causal(self, mixed_qkv: torch.Tensor, conv_state: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = mixed_qkv.shape[0]
        w = self.conv1d.weight.squeeze(1)
        if conv_state is not None and seq == 1:
            out = mixed_qkv * w[:, -1].unsqueeze(0)
            for i in range(self.conv_kernel - 1):
                out = out + conv_state[i:i + 1] * w[:, i].unsqueeze(0)
            if self.conv_kernel > 2:
                tail = conv_state[1:].clone()
                conv_state[:-1].copy_(tail)
            conv_state[-1:].copy_(mixed_qkv)
            new_state = conv_state
            return out, new_state

        if conv_state is not None:
            pad = conv_state
        else:
            pad = torch.zeros(self.conv_kernel - 1, mixed_qkv.shape[1], dtype=mixed_qkv.dtype, device=mixed_qkv.device)
        padded = torch.cat([pad, mixed_qkv], dim=0)
        conv_out = torch.zeros_like(mixed_qkv)
        for i in range(self.conv_kernel):
            conv_out += padded[i:i + seq] * w[:, i].unsqueeze(0)
        new_state = padded[-(self.conv_kernel - 1):].clone()
        return conv_out, new_state

    def _recurrent_step(
        self,
        recurrent_state: torch.Tensor,
        q_t: torch.Tensor,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        beta_t: torch.Tensor,
        decay_t: torch.Tensor,
        scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if os.environ.get("PILM_LINEAR_ATTN_RECURRENT_BACKEND", "torch").lower() == "ckernel":
            try:
                try:
                    from runtime.ckernel import gated_delta_recurrent_f32
                except ImportError:
                    from ...runtime.ckernel import gated_delta_recurrent_f32
                return gated_delta_recurrent_f32(
                    recurrent_state,
                    q_t,
                    k_t,
                    v_t,
                    beta_t.to(torch.float32),
                    decay_t.to(torch.float32),
                    scale,
                )
            except (ImportError, RuntimeError, ValueError):
                pass
        recurrent_state.mul_(decay_t.view(-1, 1, 1))
        state_t = recurrent_state.transpose(1, 2)
        s_t_t_k = torch.bmm(state_t, k_t.unsqueeze(-1)).squeeze(-1)
        delta = (v_t - s_t_t_k) * beta_t.to(torch.float32).unsqueeze(-1)
        recurrent_state.add_(k_t.unsqueeze(-1) * delta.unsqueeze(1))
        y_t = torch.bmm(recurrent_state.transpose(1, 2), q_t.unsqueeze(-1)).squeeze(-1)
        return y_t * scale, recurrent_state

    def forward(
        self,
        x: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
        recurrent_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = x.shape[0]

        stage_start = time.perf_counter() if _PROFILE_ATTENTION_STAGES else 0.0
        mixed_qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x)
        if _PROFILE_ATTENTION_STAGES:
            _record_attention_stage(f"linear_attn.proj_qkv_z.seq{seq}", stage_start)

        stage_start = time.perf_counter() if _PROFILE_ATTENTION_STAGES else 0.0
        beta = torch.sigmoid(self.in_proj_b(x))
        alpha = self.in_proj_a(x)

        dt = F.softplus(alpha.float() + self.dt_bias.unsqueeze(0).float())
        A = -torch.exp(self.A_log.float())
        gate = dt * A.unsqueeze(0)
        if _PROFILE_ATTENTION_STAGES:
            _record_attention_stage(f"linear_attn.gate.seq{seq}", stage_start)

        stage_start = time.perf_counter() if _PROFILE_ATTENTION_STAGES else 0.0
        conv_out, new_conv_state = self._conv1d_causal(mixed_qkv, conv_state)
        conv_out = F.silu(conv_out)
        if _PROFILE_ATTENTION_STAGES:
            _record_attention_stage(f"linear_attn.conv_silu.seq{seq}", stage_start)

        q = conv_out[..., :self.key_dim]
        k = conv_out[..., self.key_dim:self.key_dim * 2]
        v = conv_out[..., self.key_dim * 2:]

        stage_start = time.perf_counter() if _PROFILE_ATTENTION_STAGES else 0.0
        q_h = q.view(seq, self.n_k_heads, self.k_head_dim)
        k_h = k.view(seq, self.n_k_heads, self.k_head_dim)
        v_h = v.view(seq, self.n_v_heads, self.v_head_dim)

        q_h = self._l2_norm(q_h.to(torch.float32)).to(x.dtype)
        k_h = self._l2_norm(k_h.to(torch.float32)).to(x.dtype)

        rep = self.n_v_heads // self.n_k_heads
        if rep > 1:
            q_h = q_h.unsqueeze(2).expand(-1, -1, rep, -1).reshape(seq, self.n_v_heads, self.k_head_dim)
            k_h = k_h.unsqueeze(2).expand(-1, -1, rep, -1).reshape(seq, self.n_v_heads, self.k_head_dim)
        if _PROFILE_ATTENTION_STAGES:
            _record_attention_stage(f"linear_attn.l2_repeat.seq{seq}", stage_start)

        if recurrent_state is None:
            recurrent_state = torch.zeros(self.n_v_heads, self.k_head_dim, self.v_head_dim,
                                          dtype=torch.float32, device=x.device)

        scale = 1.0 / math.sqrt(self.v_head_dim)
        stage_start = time.perf_counter() if _PROFILE_ATTENTION_STAGES else 0.0
        if seq == 1:
            q_t = q_h[0].to(torch.float32)
            k_t = k_h[0].to(torch.float32)
            v_t = v_h[0].to(torch.float32)
            beta_t = beta[0]
            decay_t = torch.exp(gate[0])

            y_t, recurrent_state = self._recurrent_step(recurrent_state, q_t, k_t, v_t, beta_t, decay_t, scale)
            y = y_t.unsqueeze(0)
        else:
            outputs = []
            for t in range(seq):
                q_t = q_h[t].to(torch.float32)
                k_t = k_h[t].to(torch.float32)
                v_t = v_h[t].to(torch.float32)
                beta_t = beta[t]
                decay_t = torch.exp(gate[t])

                y_t, recurrent_state = self._recurrent_step(recurrent_state, q_t, k_t, v_t, beta_t, decay_t, scale)
                outputs.append(y_t)

            y = torch.stack(outputs, dim=0)
        if _PROFILE_ATTENTION_STAGES:
            _record_attention_stage(f"linear_attn.recurrent.seq{seq}", stage_start)
        recurrent_state_out = recurrent_state
        y = y.to(x.dtype)

        stage_start = time.perf_counter() if _PROFILE_ATTENTION_STAGES else 0.0
        y = y.view(seq, self.n_v_heads, self.v_head_dim)
        y_normed = self.norm(y.reshape(seq, -1, self.v_head_dim))
        y = y_normed.reshape(seq, -1) * F.silu(z)

        out = self.out_proj(y)
        if _PROFILE_ATTENTION_STAGES:
            _record_attention_stage(f"linear_attn.norm_out.seq{seq}", stage_start)
        return out, new_conv_state, recurrent_state_out
