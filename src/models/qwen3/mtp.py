from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open

from models.base.layers import GemmaRMSNorm
from models.base.linear import BackendLinear, QuantizedW4A16Linear
from models.qwen3.model import TransformerBlock


class Qwen3MTPDraft(nn.Module):
    def __init__(self, config):
        super().__init__()
        tc = config.text_config
        self.pre_fc_norm_embedding = GemmaRMSNorm(tc.hidden_size, tc.rms_norm_eps)
        self.pre_fc_norm_hidden = GemmaRMSNorm(tc.hidden_size, tc.rms_norm_eps)
        self.fc = BackendLinear(tc.hidden_size * 2, tc.hidden_size, bias=False)
        self.layers = nn.ModuleList([TransformerBlock(0, "full_attention", config)])
        self.norm = GemmaRMSNorm(tc.hidden_size, tc.rms_norm_eps)

    def hidden(
        self,
        hidden: torch.Tensor,
        token_embedding: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.ndim == 1:
            hidden = hidden.unsqueeze(0)
        if token_embedding.ndim == 1:
            token_embedding = token_embedding.unsqueeze(0)
        h = self.pre_fc_norm_hidden(hidden)
        e = self.pre_fc_norm_embedding(token_embedding)
        x = self.fc(torch.cat([e, h], dim=-1))
        x, _nc, _nr = self.layers[0](x, position)
        return self.norm(x)

    def forward(
        self,
        hidden: torch.Tensor,
        token_embedding: torch.Tensor,
        position: torch.Tensor,
        lm_head: nn.Module,
    ) -> torch.Tensor:
        return lm_head(self.hidden(hidden, token_embedding, position))

    def argmax(
        self,
        hidden: torch.Tensor,
        token_embedding: torch.Tensor,
        position: torch.Tensor,
        lm_head: nn.Module,
    ) -> int:
        logits = self.forward(hidden, token_embedding, position, lm_head)
        return int(torch.argmax(logits[-1].to(torch.float32)).item())

    def draft_tokens(
        self,
        hidden: torch.Tensor,
        first_token_id: int,
        first_position: int,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        num_tokens: int = 3,
    ) -> tuple[list[int], torch.Tensor]:
        drafts: list[int] = []
        token_id = int(first_token_id)
        state_hidden = hidden
        last_hidden = hidden
        for offset in range(max(0, int(num_tokens))):
            token_tensor = torch.tensor([token_id], dtype=torch.long)
            token_embedding = embed_tokens(token_tensor)[0]
            position = torch.tensor([int(first_position) + offset], dtype=torch.long)
            last_hidden = self.hidden(state_hidden, token_embedding, position)
            logits = lm_head(last_hidden)[-1].to(torch.float32)
            token_id = int(torch.argmax(logits).item())
            drafts.append(token_id)
            state_hidden = last_hidden[-1]
        return drafts, last_hidden[-1]


def _quantize_mtp_linears(module: nn.Module) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, BackendLinear) and child.bias is None:
            setattr(module, name, QuantizedW4A16Linear.from_linear(child))
            replaced += 1
        else:
            replaced += _quantize_mtp_linears(child)
    return replaced


def _set_parameter(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
    parent_name, param_name = name.rsplit(".", 1) if "." in name else ("", name)
    parent = module.get_submodule(parent_name) if parent_name else module
    param = parent.get_parameter(param_name)
    if tuple(tensor.shape) != tuple(param.shape):
        tensor = tensor.reshape(param.shape)
    if tensor.dtype != param.dtype:
        tensor = tensor.to(param.dtype)
    setattr(parent, param_name, nn.Parameter(tensor.contiguous(), requires_grad=False))


def _target_name(src_name: str) -> str:
    if not src_name.startswith("mtp."):
        raise ValueError(f"not an MTP tensor: {src_name}")
    name = src_name[len("mtp."):]
    name = name.replace(".mlp.gate_proj.weight", ".mlp.gate.weight")
    name = name.replace(".mlp.up_proj.weight", ".mlp.up.weight")
    name = name.replace(".mlp.down_proj.weight", ".mlp.down.weight")
    return name


def load_mtp_draft(model_dir: str, config) -> Qwen3MTPDraft:
    model_path = Path(model_dir)
    mtp_dir = model_path / "mtp_bf16"
    if not mtp_dir.exists():
        raise FileNotFoundError(f"MTP bundle directory not found: {mtp_dir}")
    _prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        mtp = Qwen3MTPDraft(config)
    finally:
        torch.set_default_dtype(_prev)
    loaded = 0
    for shard in sorted(mtp_dir.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as sf:
            for src_name in sf.keys():
                tensor = sf.get_tensor(src_name)
                _set_parameter(mtp, _target_name(src_name), tensor)
                loaded += 1
    if loaded == 0:
        raise RuntimeError(f"no MTP tensors loaded from {mtp_dir}")
    if os.environ.get("PILM_MTP_QUANTIZE", "bf16").lower() == "w4a16":
        mtp.quantized_linear_modules = _quantize_mtp_linears(mtp)
    else:
        mtp.quantized_linear_modules = 0
    mtp.eval()
    return mtp
