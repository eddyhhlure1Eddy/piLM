"""Qwen3.5 transformer model: embedding + 32 hybrid layers + norm + lm_head."""
import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from models.base.linear import BackendLinear
from models.base.layers import RMSNorm, GemmaRMSNorm, SwiGLU
from models.qwen3.attention import FullAttention, LinearAttention


class TransformerBlock(nn.Module):
    def __init__(self, layer_idx: int, layer_type: str, config):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        tc = config.text_config
        self.input_layernorm = GemmaRMSNorm(tc.hidden_size, tc.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(tc.hidden_size, tc.rms_norm_eps)

        if layer_type == "full_attention":
            self.self_attn = FullAttention(
                hidden_size=tc.hidden_size,
                n_heads=tc.num_attention_heads,
                n_kv_heads=tc.num_key_value_heads,
                head_dim=tc.head_dim,
                rope_theta=tc.rope.rope_theta,
                partial_rotary=tc.rope.partial_rotary_factor,
                mrope_section=tc.rope.mrope_section,
                mrope_interleaved=tc.rope.mrope_interleaved,
            )
        else:
            self.linear_attn = LinearAttention(
                hidden_size=tc.hidden_size,
                n_k_heads=tc.linear_num_key_heads,
                k_head_dim=tc.linear_key_head_dim,
                n_v_heads=tc.linear_num_value_heads,
                v_head_dim=tc.linear_value_head_dim,
                conv_kernel=tc.linear_conv_kernel_dim,
                rms_eps=tc.rms_norm_eps,
            )

        self.mlp = SwiGLU(tc.hidden_size, tc.intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache_k: Optional[torch.Tensor] = None,
        kv_cache_v: Optional[torch.Tensor] = None,
        block_id: int = -1,
        slot_offset: int = 0,
        is_decode: bool = False,
        block_ids: Optional[List[int]] = None,
        conv_state: Optional[torch.Tensor] = None,
        recurrent_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        residual = x
        x = self.input_layernorm(x)

        new_conv = None
        new_rec = None
        if self.layer_type == "full_attention":
            attn_out = self.self_attn(
                x, positions, kv_cache_k, kv_cache_v, block_id, slot_offset, is_decode, block_ids
            )
        else:
            attn_out, new_conv, new_rec = self.linear_attn(x, conv_state, recurrent_state)

        x = residual + attn_out
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x, new_conv, new_rec


class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        tc = config.text_config
        self.dtype = torch.bfloat16 if tc.dtype == "bfloat16" else torch.float32
        _prev = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)
        try:
            self.embed_tokens = nn.Embedding(tc.vocab_size, tc.hidden_size)
            self.layers = nn.ModuleList([
                TransformerBlock(i, tc.layer_types[i] if i < len(tc.layer_types) else "full_attention", config)
                for i in range(tc.num_hidden_layers)
            ])
            self.norm = GemmaRMSNorm(tc.hidden_size, tc.rms_norm_eps)
            self.lm_head = BackendLinear(tc.hidden_size, tc.vocab_size, bias=False)
        finally:
            torch.set_default_dtype(_prev)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches_k: Optional[List[torch.Tensor]] = None,
        kv_caches_v: Optional[List[torch.Tensor]] = None,
        block_id: int = -1,
        slot_offset: int = 0,
        is_decode: bool = False,
        block_ids: Optional[List[int]] = None,
        conv_states: Optional[List[torch.Tensor]] = None,
        recurrent_states: Optional[List[torch.Tensor]] = None,
        logits_last_only: bool = False,
        return_last_hidden: bool = False,
    ) -> Tuple[torch.Tensor, List, List]:
        x = self.embed_tokens(input_ids)
        new_convs = []
        new_recs = []
        for i, layer in enumerate(self.layers):
            kv_k = kv_caches_k[i] if kv_caches_k else None
            kv_v = kv_caches_v[i] if kv_caches_v else None
            cv = conv_states[i] if conv_states and layer.layer_type == "linear_attention" else None
            rc = recurrent_states[i] if recurrent_states and layer.layer_type == "linear_attention" else None
            x, nc, nr = layer(x, positions, kv_k, kv_v, block_id, slot_offset, is_decode, block_ids, cv, rc)
            new_convs.append(nc)
            new_recs.append(nr)
        x = self.norm(x)
        if logits_last_only:
            x = x[-1:]
        if return_last_hidden:
            return x, new_convs, new_recs
        logits = self.lm_head(x)
        return logits, new_convs, new_recs
