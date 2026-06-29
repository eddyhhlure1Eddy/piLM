"""Swappable Linear layer backend.

Default path is PyTorch. Set `PILM_LINEAR_BACKEND=ckernel_f32` to route eligible
CPU F32 bias-free Linear calls through the C GEMM wrapper.
"""
import os
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

_CKERNELS = None
_PROFILE_LINEAR_STAGES = False
_LINEAR_STAGE_SECONDS = defaultdict(float)
_LINEAR_STAGE_COUNTS = defaultdict(int)


def _w4a16_q8_enabled() -> bool:
    """Experimental: route W4 per-row Linear through the int8-activation
    (pmaddwd) kernel `linear_w4a16_bf16_q8` instead of `linear_w4a16_bf16`.

    Requires a loaded eCPU library that exposes
    `ekernel_linear_w4a16_bf16_q8` (e.g. build_w4q8 via PILM_ECPU_LIB).
    Only affects per-row W4 scales (ndim == 1); grouped W4 (g32/g128) and all
    W8 paths are unchanged.  Default off."""
    return os.environ.get("PILM_W4A16_Q8", "0") == "1"


def _w4_b8_enabled() -> bool:
    """Experimental: route W4 SwiGLU MLP linears through blocked8xK16 layout."""
    return os.environ.get("PILM_W4_B8", "0") == "1"


def _w4_b8_linear_enabled() -> bool:
    """Experimental: route standalone per-row W4 Linear through blocked8xK16."""
    return os.environ.get("PILM_W4_B8_LINEAR", "0") == "1"


def _w8a16_q8_enabled() -> bool:
    """Experimental: route the BF16 W8 Linear path through the int8-activation
    (madd_epi16) kernel `linear_w8a16_bf16_q8` instead of `linear_w8a16_bf16`.

    Requires a loaded eCPU library that exposes
    `ekernel_linear_w8a16_bf16_q8` (e.g. build_w8q8 via PILM_ECPU_LIB).
    Only affects the BF16-activation W8 path (QuantizedW8A32Linear BF16 branch
    and QuantizedW8A32SwiGLU); the F32 W8A32 path, argmax, and i8b8 paths are
    unchanged.  Default off."""
    return os.environ.get("PILM_W8A16_Q8", "0") == "1"


def _w4_swiglu_fused_enabled() -> bool:
    """Experimental: route M=1 per-row W4 SwiGLU through one fused C kernel.

    Requires a loaded eCPU library exposing `ekernel_swiglu_w4a16_bf16`.
    Default off; the stable W4 service should keep using the measured two-step
    W4 SwiGLU path unless this wins in isolated and full-model benches."""
    return os.environ.get("PILM_W4_SWIGLU_FUSED", "0") == "1"


def _get_ckernels():
    global _CKERNELS
    if _CKERNELS is not None:
        return _CKERNELS
    try:
        from runtime.ckernel import (
            linear_f32,
            quantize_weight_i8_per_row,
            quantize_weight_i4_per_row_chunked,
            quantize_weight_i4_grouped_chunked,
            linear_w8a32,
            linear_w8a16_bf16,
            linear_w8a16_bf16_argmax,
            linear_w4a16_bf16,
            linear_w4a16g32_bf16,
            linear_w4a16g128_bf16,
            swiglu_bf16,
            swiglu_w4a16_bf16,
            pack_i4_rows_blocked8k16,
            linear_w4a16_bf16_b8,
        )
    except ImportError:
        from ...runtime.ckernel import (
            linear_f32,
            quantize_weight_i8_per_row,
            quantize_weight_i4_per_row_chunked,
            quantize_weight_i4_grouped_chunked,
            linear_w8a32,
            linear_w8a16_bf16,
            linear_w8a16_bf16_argmax,
            linear_w4a16_bf16,
            linear_w4a16g32_bf16,
            linear_w4a16g128_bf16,
            swiglu_bf16,
            swiglu_w4a16_bf16,
            pack_i4_rows_blocked8k16,
            linear_w4a16_bf16_b8,
        )
    # Optional experimental int8-activation W4 kernel.  Missing symbol -> None;
    # callers must check before using it.  Lives at index 11.
    linear_w4a16_bf16_q8 = None
    try:
        try:
            from runtime.ckernel import linear_w4a16_bf16_q8 as _q8
        except ImportError:
            from ...runtime.ckernel import linear_w4a16_bf16_q8 as _q8
        linear_w4a16_bf16_q8 = _q8
    except Exception:
        linear_w4a16_bf16_q8 = None
    # Optional experimental int8-activation W8 kernel.  Lives at index 12.
    linear_w8a16_bf16_q8 = None
    try:
        try:
            from runtime.ckernel import linear_w8a16_bf16_q8 as _q8w8
        except ImportError:
            from ...runtime.ckernel import linear_w8a16_bf16_q8 as _q8w8
        linear_w8a16_bf16_q8 = _q8w8
    except Exception:
        linear_w8a16_bf16_q8 = None
    # Optional experimental fused W4 SwiGLU MLP kernel.  Lives at index 13.
    fused_w4_swiglu = None
    try:
        fused_w4_swiglu = swiglu_w4a16_bf16
    except Exception:
        fused_w4_swiglu = None
    _CKERNELS = (
        linear_f32,
        quantize_weight_i8_per_row,
        quantize_weight_i4_per_row_chunked,
        quantize_weight_i4_grouped_chunked,
        linear_w8a32,
        linear_w8a16_bf16,
        linear_w8a16_bf16_argmax,
        linear_w4a16_bf16,
        linear_w4a16g32_bf16,
        linear_w4a16g128_bf16,
        swiglu_bf16,
        linear_w4a16_bf16_q8,
        linear_w8a16_bf16_q8,
        fused_w4_swiglu,
        pack_i4_rows_blocked8k16,
        linear_w4a16_bf16_b8,
    )
    return _CKERNELS


def reset_linear_stage_profile(enabled: bool = True) -> None:
    global _PROFILE_LINEAR_STAGES
    _PROFILE_LINEAR_STAGES = enabled
    _LINEAR_STAGE_SECONDS.clear()
    _LINEAR_STAGE_COUNTS.clear()


def linear_stage_profile() -> dict:
    profile = {}
    for name, seconds in sorted(_LINEAR_STAGE_SECONDS.items(), key=lambda item: item[1], reverse=True):
        count = _LINEAR_STAGE_COUNTS[name]
        profile[name] = {
            "calls": count,
            "seconds": round(seconds, 4),
            "avg_seconds": round(seconds / count, 6) if count else 0.0,
        }
    return profile


def _record_stage(name: str, start: float) -> None:
    _LINEAR_STAGE_SECONDS[name] += time.perf_counter() - start
    _LINEAR_STAGE_COUNTS[name] += 1


class BackendLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        backend = os.environ.get("PILM_LINEAR_BACKEND", "torch").lower()
        if (
            backend == "ckernel_f32"
            and self.bias is None
            and input.device.type == "cpu"
            and self.weight.device.type == "cpu"
            and input.dtype == torch.float32
            and self.weight.dtype == torch.float32
        ):
            linear_f32 = _get_ckernels()[0]
            return linear_f32(input, self.weight)
        return F.linear(input, self.weight, self.bias)


class QuantizedW8A32Linear(nn.Module):
    """Weight-only int8 Linear.

    This is an inference-only module. It stores int8 per-output-channel weights
    plus F32 scales, and currently computes with the W8A32 C kernel.
    """

    def __init__(self, qweight: torch.Tensor, scales: torch.Tensor):
        super().__init__()
        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("scales", scales.contiguous())
        self.in_features = qweight.shape[1]
        self.out_features = qweight.shape[0]

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "QuantizedW8A32Linear":
        if linear.bias is not None:
            raise ValueError("QuantizedW8A32Linear only supports bias-free Linear")
        quantize_weight_i8_per_row = _get_ckernels()[1]
        qweight, scales = quantize_weight_i8_per_row(linear.weight.detach().to(torch.float32))
        return cls(qweight, scales)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        linear_w8a32 = _get_ckernels()[4]
        if input.dtype == torch.bfloat16 and input.device.type == "cpu":
            original_shape = tuple(input.shape)
            if input.ndim == 1:
                x2d = input.reshape(1, -1)
                squeeze = True
            elif input.ndim == 2:
                x2d = input
                squeeze = False
            else:
                x2d = input.reshape(-1, input.shape[-1])
                squeeze = False
            stage_start = time.perf_counter() if _PROFILE_LINEAR_STAGES else 0.0
            kernels = _get_ckernels()
            if _w8a16_q8_enabled() and kernels[12] is not None:
                out = kernels[12](x2d, self.qweight, self.scales)
                stage_name = "linear_w8a16_bf16_q8"
            else:
                out = kernels[5](x2d, self.qweight, self.scales)
                stage_name = "linear_w8a16_bf16"
            if _PROFILE_LINEAR_STAGES:
                _record_stage(f"{stage_name}.M{x2d.shape[0]}.N{self.out_features}.K{self.in_features}", stage_start)
            if squeeze:
                return out.reshape(-1)
            if len(original_shape) > 2:
                return out.reshape(*original_shape[:-1], out.shape[-1])
            return out
        original_dtype = input.dtype
        original_shape = tuple(input.shape)
        x = input.to(torch.float32)
        if x.ndim == 1:
            x2d = x.reshape(1, -1)
            squeeze = True
        elif x.ndim == 2:
            x2d = x
            squeeze = False
        else:
            x2d = x.reshape(-1, x.shape[-1])
            squeeze = False
        stage_start = time.perf_counter() if _PROFILE_LINEAR_STAGES else 0.0
        out = linear_w8a32(x2d, self.qweight, self.scales)
        if _PROFILE_LINEAR_STAGES:
            _record_stage(f"linear_w8a32.M{x2d.shape[0]}.N{self.out_features}.K{self.in_features}", stage_start)
        if squeeze:
            out = out.reshape(-1)
        elif len(original_shape) > 2:
            out = out.reshape(*original_shape[:-1], out.shape[-1])
        return out.to(original_dtype)

    def argmax_bf16(self, input: torch.Tensor) -> int:
        if input.dtype != torch.bfloat16 or input.device.type != "cpu":
            raise ValueError("argmax_bf16 requires CPU BF16 input")
        linear_w8a16_bf16_argmax = _get_ckernels()[6]
        return linear_w8a16_bf16_argmax(input, self.qweight, self.scales)


class QuantizedW4A16Linear(nn.Module):
    """Weight-only signed int4 Linear for BF16 CPU inference.

    Two signed int4 weights are packed into one uint8. Each output row has a
    float32 scale. This is intentionally simple and static so it can run on
    Windows, Linux, and ARM before architecture-specific kernels are added.
    """

    def __init__(self, qweight: torch.Tensor, scales: torch.Tensor, in_features: int):
        super().__init__()
        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("scales", scales.contiguous())
        self.in_features = int(in_features)
        self.out_features = qweight.shape[0]
        self.group_size = None
        self._qweight_b8 = None
        if scales.ndim == 2:
            groups = int(scales.shape[1])
            if groups == (self.in_features + 127) // 128:
                self.group_size = 128
            else:
                self.group_size = 32

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "QuantizedW4A16Linear":
        if linear.bias is not None:
            raise ValueError("QuantizedW4A16Linear only supports bias-free Linear")
        quantize_weight_i4_per_row_chunked = _get_ckernels()[2]
        qweight, scales = quantize_weight_i4_per_row_chunked(linear.weight.detach().to(torch.float32))
        return cls(qweight, scales, linear.in_features)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dtype != torch.bfloat16 or input.device.type != "cpu":
            raise ValueError("QuantizedW4A16Linear currently requires CPU BF16 input")
        original_shape = tuple(input.shape)
        if input.ndim == 1:
            x2d = input.reshape(1, -1)
            squeeze = True
        elif input.ndim == 2:
            x2d = input
            squeeze = False
        else:
            x2d = input.reshape(-1, input.shape[-1])
            squeeze = False
        stage_start = time.perf_counter() if _PROFILE_LINEAR_STAGES else 0.0
        if self.scales.ndim == 2:
            if self.group_size == 128:
                linear_w4a16g128_bf16 = _get_ckernels()[9]
                out = linear_w4a16g128_bf16(x2d, self.qweight, self.scales, self.in_features)
                stage_name = "linear_w4a16g128_bf16"
            else:
                linear_w4a16g32_bf16 = _get_ckernels()[8]
                out = linear_w4a16g32_bf16(x2d, self.qweight, self.scales, self.in_features)
                stage_name = "linear_w4a16g32_bf16"
        else:
            kernels = _get_ckernels()
            if _w4a16_q8_enabled() and kernels[11] is not None:
                out = kernels[11](x2d, self.qweight, self.scales, self.in_features)
                stage_name = "linear_w4a16_bf16_q8"
            elif _w4_b8_linear_enabled() and x2d.shape[0] == 1 and kernels[15] is not None:
                if self._qweight_b8 is None:
                    self._qweight_b8 = kernels[14](self.qweight, self.in_features)
                out = kernels[15](x2d, self._qweight_b8, self.scales, self.out_features, self.in_features)
                stage_name = "linear_w4a16_bf16_b8"
            else:
                out = kernels[7](x2d, self.qweight, self.scales, self.in_features)
                stage_name = "linear_w4a16_bf16"
        if _PROFILE_LINEAR_STAGES:
            _record_stage(f"{stage_name}.M{x2d.shape[0]}.N{self.out_features}.K{self.in_features}", stage_start)
        if squeeze:
            return out.reshape(-1)
        if len(original_shape) > 2:
            return out.reshape(*original_shape[:-1], out.shape[-1])
        return out


class QuantizedW8A32SwiGLU(nn.Module):
    """Fused quantized SwiGLU.

    Computes gate and up projection in one quantized kernel by concatenating
    their row-wise int8 weights. The down projection remains a second quantized
    kernel.
    """

    def __init__(
        self,
        gate_up_qweight: torch.Tensor,
        gate_up_scales: torch.Tensor,
        down_qweight: torch.Tensor,
        down_scales: torch.Tensor,
        intermediate_size: int,
    ):
        super().__init__()
        self.register_buffer("gate_up_qweight", gate_up_qweight.contiguous())
        self.register_buffer("gate_up_scales", gate_up_scales.contiguous())
        self.register_buffer("down_qweight", down_qweight.contiguous())
        self.register_buffer("down_scales", down_scales.contiguous())
        self.intermediate_size = intermediate_size

    @classmethod
    def from_swiglu(cls, swiglu: nn.Module) -> "QuantizedW8A32SwiGLU":
        if swiglu.gate.bias is not None or swiglu.up.bias is not None or swiglu.down.bias is not None:
            raise ValueError("QuantizedW8A32SwiGLU only supports bias-free Linear")
        quantize_weight_i8_per_row = _get_ckernels()[1]
        gate_q, gate_scales = quantize_weight_i8_per_row(swiglu.gate.weight.detach().to(torch.float32))
        up_q, up_scales = quantize_weight_i8_per_row(swiglu.up.weight.detach().to(torch.float32))
        down_q, down_scales = quantize_weight_i8_per_row(swiglu.down.weight.detach().to(torch.float32))
        return cls(
            torch.cat([gate_q, up_q], dim=0),
            torch.cat([gate_scales, up_scales], dim=0),
            down_q,
            down_scales,
            gate_q.shape[0],
        )

    def _linear(self, x2d: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        linear_w8a32 = _get_ckernels()[4]
        linear_w8a16_bf16 = _get_ckernels()[5]
        if x2d.dtype == torch.bfloat16:
            kernels = _get_ckernels()
            if _w8a16_q8_enabled() and kernels[12] is not None:
                return kernels[12](x2d, qweight, scales)
            return linear_w8a16_bf16(x2d, qweight, scales)
        return linear_w8a32(x2d.to(torch.float32), qweight, scales).to(x2d.dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        original_shape = tuple(input.shape)
        if input.ndim == 1:
            x2d = input.reshape(1, -1)
            squeeze = True
        elif input.ndim == 2:
            x2d = input
            squeeze = False
        else:
            x2d = input.reshape(-1, input.shape[-1])
            squeeze = False

        stage_start = time.perf_counter() if _PROFILE_LINEAR_STAGES else 0.0
        gate_up = self._linear(x2d, self.gate_up_qweight, self.gate_up_scales)
        if _PROFILE_LINEAR_STAGES:
            _record_stage(f"swiglu.gate_up.M{x2d.shape[0]}.N{self.intermediate_size * 2}.K{x2d.shape[1]}", stage_start)
        gate, up = gate_up.split(self.intermediate_size, dim=-1)
        if (
            os.environ.get("PILM_SWIGLU_ACT_BACKEND", "torch").lower() == "ckernel"
            and gate.dtype == torch.bfloat16
            and gate.device.type == "cpu"
        ):
            stage_start = time.perf_counter() if _PROFILE_LINEAR_STAGES else 0.0
            hidden = _get_ckernels()[10](gate, up)
            if _PROFILE_LINEAR_STAGES:
                _record_stage(f"swiglu.activation_ckernel.M{x2d.shape[0]}.N{self.intermediate_size}", stage_start)
        else:
            stage_start = time.perf_counter() if _PROFILE_LINEAR_STAGES else 0.0
            hidden = F.silu(gate) * up
            if _PROFILE_LINEAR_STAGES:
                _record_stage(f"swiglu.activation_torch.M{x2d.shape[0]}.N{self.intermediate_size}", stage_start)
        stage_start = time.perf_counter() if _PROFILE_LINEAR_STAGES else 0.0
        out = self._linear(hidden, self.down_qweight, self.down_scales)
        if _PROFILE_LINEAR_STAGES:
            _record_stage(f"swiglu.down.M{x2d.shape[0]}.N{self.down_qweight.shape[0]}.K{self.down_qweight.shape[1]}", stage_start)

        if squeeze:
            return out.reshape(-1)
        if len(original_shape) > 2:
            return out.reshape(*original_shape[:-1], out.shape[-1])
        return out


class QuantizedW4A16SwiGLU(nn.Module):
    """Fused W4A16 SwiGLU gate/up projection plus quantized down projection."""

    def __init__(
        self,
        gate_up_qweight: torch.Tensor,
        gate_up_scales: torch.Tensor,
        down_qweight: torch.Tensor,
        down_scales: torch.Tensor,
        intermediate_size: int,
        gate_up_in_features: int,
        down_in_features: int,
    ):
        super().__init__()
        self.register_buffer("gate_up_qweight", gate_up_qweight.contiguous())
        self.register_buffer("gate_up_scales", gate_up_scales.contiguous())
        self.register_buffer("down_qweight", down_qweight.contiguous())
        self.register_buffer("down_scales", down_scales.contiguous())
        self.intermediate_size = int(intermediate_size)
        self.gate_up_in_features = int(gate_up_in_features)
        self.down_in_features = int(down_in_features)
        self._gate_up_qweight_b8 = None
        self._down_qweight_b8 = None

    def _linear(
        self,
        x2d: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        in_features: int,
        cache_attr: str | None = None,
    ) -> torch.Tensor:
        if x2d.dtype != torch.bfloat16:
            raise ValueError("QuantizedW4A16SwiGLU requires CPU BF16 input")
        if scales.ndim != 1:
            raise ValueError("QuantizedW4A16SwiGLU currently supports per-row W4 scales only")
        kernels = _get_ckernels()
        if _w4a16_q8_enabled() and kernels[11] is not None:
            return kernels[11](x2d, qweight, scales, in_features)
        if _w4_b8_enabled() and x2d.shape[0] == 1 and kernels[15] is not None:
            blocked = getattr(self, cache_attr) if cache_attr else None
            if blocked is None:
                blocked = kernels[14](qweight, in_features)
                if cache_attr:
                    setattr(self, cache_attr, blocked)
            return kernels[15](x2d, blocked, scales, qweight.shape[0], in_features)
        return kernels[7](x2d, qweight, scales, in_features)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        original_shape = tuple(input.shape)
        if input.ndim == 1:
            x2d = input.reshape(1, -1)
            squeeze = True
        elif input.ndim == 2:
            x2d = input
            squeeze = False
        else:
            x2d = input.reshape(-1, input.shape[-1])
            squeeze = False

        kernels = _get_ckernels()
        if (
            _w4_swiglu_fused_enabled()
            and not _w4a16_q8_enabled()
            and kernels[13] is not None
            and x2d.shape[0] == 1
        ):
            stage_start = time.perf_counter() if _PROFILE_LINEAR_STAGES else 0.0
            out = kernels[13](
                x2d,
                self.gate_up_qweight,
                self.gate_up_scales,
                self.down_qweight,
                self.down_scales,
                self.intermediate_size,
                self.gate_up_in_features,
                self.down_in_features,
            )
            if _PROFILE_LINEAR_STAGES:
                _record_stage(
                    f"swiglu_w4.fused_c.M{x2d.shape[0]}.H{self.down_qweight.shape[0]}.I{self.intermediate_size}.K{x2d.shape[1]}",
                    stage_start,
                )
            if squeeze:
                return out.reshape(-1)
            if len(original_shape) > 2:
                return out.reshape(*original_shape[:-1], out.shape[-1])
            return out

        stage_start = time.perf_counter() if _PROFILE_LINEAR_STAGES else 0.0
        gate_up = self._linear(
            x2d,
            self.gate_up_qweight,
            self.gate_up_scales,
            self.gate_up_in_features,
            "_gate_up_qweight_b8",
        )
        if _PROFILE_LINEAR_STAGES:
            _record_stage(f"swiglu_w4.gate_up.M{x2d.shape[0]}.N{self.intermediate_size * 2}.K{x2d.shape[1]}", stage_start)
        gate, up = gate_up.split(self.intermediate_size, dim=-1)
        stage_start = time.perf_counter() if _PROFILE_LINEAR_STAGES else 0.0
        if (
            os.environ.get("PILM_SWIGLU_ACT_BACKEND", "torch").lower() == "ckernel"
            and gate.dtype == torch.bfloat16
            and gate.device.type == "cpu"
        ):
            hidden = _get_ckernels()[10](gate, up)
            act_name = "activation_ckernel"
        else:
            hidden = F.silu(gate) * up
            act_name = "activation_torch"
        if _PROFILE_LINEAR_STAGES:
            _record_stage(f"swiglu_w4.{act_name}.M{x2d.shape[0]}.N{self.intermediate_size}", stage_start)
        stage_start = time.perf_counter() if _PROFILE_LINEAR_STAGES else 0.0
        out = self._linear(hidden, self.down_qweight, self.down_scales, self.down_in_features, "_down_qweight_b8")
        if _PROFILE_LINEAR_STAGES:
            _record_stage(f"swiglu_w4.down.M{x2d.shape[0]}.N{self.down_qweight.shape[0]}.K{self.down_in_features}", stage_start)

        if squeeze:
            return out.reshape(-1)
        if len(original_shape) > 2:
            return out.reshape(*original_shape[:-1], out.shape[-1])
        return out
