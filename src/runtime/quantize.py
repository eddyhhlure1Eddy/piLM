"""Runtime quantization helpers."""
from __future__ import annotations

import torch.nn as nn
import torch

try:
    from models.base.layers import SwiGLU
    from models.base.linear import BackendLinear, QuantizedW8A32Linear, QuantizedW8A32SwiGLU, QuantizedW4A16Linear, QuantizedW4A16SwiGLU
except ImportError:
    from ..models.base.layers import SwiGLU
    from ..models.base.linear import BackendLinear, QuantizedW8A32Linear, QuantizedW8A32SwiGLU, QuantizedW4A16Linear, QuantizedW4A16SwiGLU


def quantize_linear_modules_w8a32(module: nn.Module, skip_lm_head: bool = True, prefix: str = "") -> int:
    """Replace bias-free Linear modules with QuantizedW8A32Linear in-place.

    Returns the number of replaced modules.
    """
    replaced = 0
    for name, child in list(module.named_children()):
        child_prefix = f"{prefix}.{name}" if prefix else name
        if skip_lm_head and child_prefix == "lm_head":
            continue
        if isinstance(child, (BackendLinear, nn.Linear)) and child.bias is None:
            setattr(module, name, QuantizedW8A32Linear.from_linear(child))
            replaced += 1
        else:
            replaced += quantize_linear_modules_w8a32(child, skip_lm_head=skip_lm_head, prefix=child_prefix)
    return replaced


def fuse_quantized_swiglu_modules(module: nn.Module) -> int:
    """Replace SwiGLU blocks whose three Linear children are already quantized.

    Returns the number of fused MLP blocks.
    """
    fused = 0
    for name, child in list(module.named_children()):
        if (
            isinstance(child, SwiGLU)
            and isinstance(child.gate, QuantizedW8A32Linear)
            and isinstance(child.up, QuantizedW8A32Linear)
            and isinstance(child.down, QuantizedW8A32Linear)
        ):
            setattr(
                module,
                name,
                QuantizedW8A32SwiGLU(
                    gate_up_qweight=torch.cat([child.gate.qweight, child.up.qweight], dim=0),
                    gate_up_scales=torch.cat([child.gate.scales, child.up.scales], dim=0),
                    down_qweight=child.down.qweight,
                    down_scales=child.down.scales,
                    intermediate_size=child.gate.out_features,
                ),
            )
            fused += 1
        elif (
            isinstance(child, SwiGLU)
            and isinstance(child.gate, QuantizedW4A16Linear)
            and isinstance(child.up, QuantizedW4A16Linear)
            and isinstance(child.down, QuantizedW4A16Linear)
            and child.gate.scales.ndim == 1
            and child.up.scales.ndim == 1
            and child.down.scales.ndim == 1
        ):
            setattr(
                module,
                name,
                QuantizedW4A16SwiGLU(
                    gate_up_qweight=torch.cat([child.gate.qweight, child.up.qweight], dim=0),
                    gate_up_scales=torch.cat([child.gate.scales, child.up.scales], dim=0),
                    down_qweight=child.down.qweight,
                    down_scales=child.down.scales,
                    intermediate_size=child.gate.out_features,
                    gate_up_in_features=child.gate.in_features,
                    down_in_features=child.down.in_features,
                ),
            )
            fused += 1
        else:
            fused += fuse_quantized_swiglu_modules(child)
    return fused
