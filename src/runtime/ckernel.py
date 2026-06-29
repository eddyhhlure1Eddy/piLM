"""Python wrappers for piLM C kernels."""
from __future__ import annotations

from typing import Tuple

import torch

try:
    from .. import _abi as abi
except ImportError:
    import _abi as abi


def _require_f32_cpu_2d(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must be torch.float32, got {tensor.dtype}")
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={tuple(tensor.shape)}")
    return tensor.contiguous()


def gemm_f32(a: torch.Tensor, b: torch.Tensor, transpose_b: bool = False) -> torch.Tensor:
    """Run C ekernel_gemm for F32 matrices.

    If transpose_b is False:
        a[M, K] @ b[K, N] -> out[M, N]
    If transpose_b is True:
        a[M, K] @ b[N, K].T -> out[M, N]
    """
    a = _require_f32_cpu_2d("a", a)
    b = _require_f32_cpu_2d("b", b)
    m, k = a.shape
    if transpose_b:
        n, kb = b.shape
    else:
        kb, n = b.shape
    if kb != k:
        raise ValueError(f"incompatible GEMM shapes: a={tuple(a.shape)} b={tuple(b.shape)} transpose_b={transpose_b}")

    out = torch.empty((m, n), dtype=torch.float32)
    rc = abi.gemm_f32_ptr(a.data_ptr(), b.data_ptr(), out.data_ptr(), m, n, k, False, transpose_b)
    if rc != 0:
        raise RuntimeError(f"ekernel_gemm failed: rc={rc}")
    return out


def linear_f32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """F32 linear without bias using C GEMM: x @ weight.T."""
    original_shape: Tuple[int, ...] = tuple(x.shape)
    if x.ndim == 1:
        x2d = x.reshape(1, -1)
        squeeze = True
    elif x.ndim == 2:
        x2d = x
        squeeze = False
    else:
        x2d = x.reshape(-1, x.shape[-1])
        squeeze = False

    out = gemm_f32(x2d, weight, transpose_b=True)
    if squeeze:
        return out.reshape(-1)
    if len(original_shape) > 2:
        return out.reshape(*original_shape[:-1], out.shape[-1])
    return out


def quantize_weight_i8_per_row(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight = _require_f32_cpu_2d("weight", weight)
    max_abs = weight.abs().amax(dim=1)
    scales = torch.clamp(max_abs / 127.0, min=1e-8).to(torch.float32)
    qweight = torch.round(weight / scales.unsqueeze(1)).clamp(-127, 127).to(torch.int8).contiguous()
    return qweight, scales.contiguous()


def quantize_weight_i8_per_row_chunked(weight: torch.Tensor, rows_per_chunk: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.device.type != "cpu":
        raise ValueError("weight must be a CPU tensor")
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D, got shape={tuple(weight.shape)}")
    rows, cols = weight.shape
    rows_per_chunk = max(1, int(rows_per_chunk))
    qweight = torch.empty((rows, cols), dtype=torch.int8)
    scales = torch.empty((rows,), dtype=torch.float32)
    for start in range(0, rows, rows_per_chunk):
        end = min(start + rows_per_chunk, rows)
        chunk = weight[start:end].to(torch.float32).contiguous()
        max_abs = chunk.abs().amax(dim=1)
        chunk_scales = torch.clamp(max_abs / 127.0, min=1e-8).to(torch.float32)
        qweight[start:end].copy_(torch.round(chunk / chunk_scales.unsqueeze(1)).clamp(-127, 127).to(torch.int8))
        scales[start:end].copy_(chunk_scales)
        del chunk, max_abs, chunk_scales
    return qweight.contiguous(), scales.contiguous()


def quantize_weight_i4_per_row_chunked(weight: torch.Tensor, rows_per_chunk: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.device.type != "cpu":
        raise ValueError("weight must be a CPU tensor")
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D, got shape={tuple(weight.shape)}")
    rows, cols = weight.shape
    packed_cols = (cols + 1) // 2
    rows_per_chunk = max(1, int(rows_per_chunk))
    qweight = torch.empty((rows, packed_cols), dtype=torch.uint8)
    scales = torch.empty((rows,), dtype=torch.float32)
    for start in range(0, rows, rows_per_chunk):
        end = min(start + rows_per_chunk, rows)
        chunk = weight[start:end].to(torch.float32).contiguous()
        max_abs = chunk.abs().amax(dim=1)
        chunk_scales = torch.clamp(max_abs / 7.0, min=1e-8).to(torch.float32)
        q = torch.round(chunk / chunk_scales.unsqueeze(1)).clamp(-7, 7).to(torch.int16)
        if cols % 2:
            q = torch.nn.functional.pad(q, (0, 1))
        low = torch.bitwise_and(q[:, 0::2], 0x0F).to(torch.uint8)
        high = torch.bitwise_left_shift(torch.bitwise_and(q[:, 1::2], 0x0F), 4).to(torch.uint8)
        qweight[start:end].copy_(torch.bitwise_or(low, high))
        scales[start:end].copy_(chunk_scales)
        del chunk, max_abs, chunk_scales, q, low, high
    return qweight.contiguous(), scales.contiguous()


def quantize_weight_i4_grouped_chunked(
    weight: torch.Tensor,
    group_size: int = 32,
    rows_per_chunk: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    if group_size not in {32, 128}:
        raise ValueError("only group_size=32 or 128 is currently supported by the C kernel")
    if weight.device.type != "cpu":
        raise ValueError("weight must be a CPU tensor")
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D, got shape={tuple(weight.shape)}")
    rows, cols = weight.shape
    packed_cols = (cols + 1) // 2
    groups = (cols + group_size - 1) // group_size
    rows_per_chunk = max(1, int(rows_per_chunk))
    qweight = torch.empty((rows, packed_cols), dtype=torch.uint8)
    scales = torch.empty((rows, groups), dtype=torch.float32)
    for start in range(0, rows, rows_per_chunk):
        end = min(start + rows_per_chunk, rows)
        chunk = weight[start:end].to(torch.float32).contiguous()
        if cols % group_size:
            pad_cols = groups * group_size - cols
            chunk_for_quant = torch.nn.functional.pad(chunk, (0, pad_cols))
        else:
            chunk_for_quant = chunk
        grouped = chunk_for_quant.reshape(end - start, groups, group_size)
        max_abs = grouped.abs().amax(dim=2)
        chunk_scales = torch.clamp(max_abs / 7.0, min=1e-8).to(torch.float32)
        q = torch.round(grouped / chunk_scales.unsqueeze(2)).clamp(-7, 7).to(torch.int16)
        q = q.reshape(end - start, groups * group_size)[:, :cols]
        if cols % 2:
            q = torch.nn.functional.pad(q, (0, 1))
        low = torch.bitwise_and(q[:, 0::2], 0x0F).to(torch.uint8)
        high = torch.bitwise_left_shift(torch.bitwise_and(q[:, 1::2], 0x0F), 4).to(torch.uint8)
        qweight[start:end].copy_(torch.bitwise_or(low, high))
        scales[start:end].copy_(chunk_scales)
        del chunk, chunk_for_quant, grouped, max_abs, chunk_scales, q, low, high
    return qweight.contiguous(), scales.contiguous()


def linear_w8a32(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    x = _require_f32_cpu_2d("x", x)
    if qweight.device.type != "cpu" or qweight.dtype != torch.int8 or qweight.ndim != 2:
        raise ValueError("qweight must be a 2D CPU torch.int8 tensor")
    if scales.device.type != "cpu" or scales.dtype != torch.float32 or scales.ndim != 1:
        raise ValueError("scales must be a 1D CPU torch.float32 tensor")
    qweight = qweight.contiguous()
    scales = scales.contiguous()
    m, k = x.shape
    n, wk = qweight.shape
    if wk != k or scales.shape[0] != n:
        raise ValueError(f"incompatible W8A32 shapes: x={tuple(x.shape)} qweight={tuple(qweight.shape)} scales={tuple(scales.shape)}")
    out = torch.empty((m, n), dtype=torch.float32)
    rc = abi.linear_w8a32_ptr(x.data_ptr(), qweight.data_ptr(), scales.data_ptr(), out.data_ptr(), m, n, k)
    if rc != 0:
        raise RuntimeError(f"ekernel_linear_w8a32 failed: rc={rc}")
    return out


def linear_w8a16_bf16(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if x.device.type != "cpu" or x.dtype != torch.bfloat16 or x.ndim != 2:
        raise ValueError("x must be a 2D CPU torch.bfloat16 tensor")
    if qweight.device.type != "cpu" or qweight.dtype != torch.int8 or qweight.ndim != 2:
        raise ValueError("qweight must be a 2D CPU torch.int8 tensor")
    if scales.device.type != "cpu" or scales.dtype != torch.float32 or scales.ndim != 1:
        raise ValueError("scales must be a 1D CPU torch.float32 tensor")
    x = x.contiguous()
    qweight = qweight.contiguous()
    scales = scales.contiguous()
    m, k = x.shape
    n, wk = qweight.shape
    if wk != k or scales.shape[0] != n:
        raise ValueError(f"incompatible W8A16 shapes: x={tuple(x.shape)} qweight={tuple(qweight.shape)} scales={tuple(scales.shape)}")
    out = torch.empty((m, n), dtype=torch.bfloat16)
    rc = abi.linear_w8a16_bf16_ptr(x.data_ptr(), qweight.data_ptr(), scales.data_ptr(), out.data_ptr(), m, n, k)
    if rc != 0:
        raise RuntimeError(f"ekernel_linear_w8a16_bf16 failed: rc={rc}")
    return out


def linear_w8a16_bf16_q8(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Experimental W8A16 BF16 linear with on-the-fly int8 activation quantization.

    Accepts the same int8 per-output-row-scaled weights as `linear_w8a16_bf16`;
    the activation row is quantized to signed int8 inside the C kernel and the
    dot product uses an AVX2 integer (madd_epi16) path.  The only added error
    versus `linear_w8a16_bf16` is the int8 activation rounding.  Requires a
    loaded eCPU library that exposes `ekernel_linear_w8a16_bf16_q8`.
    """
    if x.device.type != "cpu" or x.dtype != torch.bfloat16 or x.ndim != 2:
        raise ValueError("x must be a 2D CPU torch.bfloat16 tensor")
    if qweight.device.type != "cpu" or qweight.dtype != torch.int8 or qweight.ndim != 2:
        raise ValueError("qweight must be a 2D CPU torch.int8 tensor")
    if scales.device.type != "cpu" or scales.dtype != torch.float32 or scales.ndim != 1:
        raise ValueError("scales must be a 1D CPU torch.float32 tensor")
    x = x.contiguous()
    qweight = qweight.contiguous()
    scales = scales.contiguous()
    m, k = x.shape
    n, wk = qweight.shape
    if wk != k or scales.shape[0] != n:
        raise ValueError(f"incompatible W8A16-Q8 shapes: x={tuple(x.shape)} qweight={tuple(qweight.shape)} scales={tuple(scales.shape)}")
    out = torch.empty((m, n), dtype=torch.bfloat16)
    rc = abi.linear_w8a16_bf16_q8_ptr(x.data_ptr(), qweight.data_ptr(), scales.data_ptr(), out.data_ptr(), m, n, k)
    if rc != 0:
        raise RuntimeError(f"ekernel_linear_w8a16_bf16_q8 failed: rc={rc}")
    return out


def linear_w8a16_bf16_argmax(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor) -> int:
    if x.device.type != "cpu" or x.dtype != torch.bfloat16:
        raise ValueError("x must be a CPU torch.bfloat16 tensor")
    if qweight.device.type != "cpu" or qweight.dtype != torch.int8 or qweight.ndim != 2:
        raise ValueError("qweight must be a 2D CPU torch.int8 tensor")
    if scales.device.type != "cpu" or scales.dtype != torch.float32 or scales.ndim != 1:
        raise ValueError("scales must be a 1D CPU torch.float32 tensor")
    if x.ndim == 1:
        x2d = x.reshape(1, -1)
    elif x.ndim == 2 and x.shape[0] == 1:
        x2d = x
    else:
        raise ValueError(f"x must be a single row, got shape={tuple(x.shape)}")
    x2d = x2d.contiguous()
    qweight = qweight.contiguous()
    scales = scales.contiguous()
    m, k = x2d.shape
    n, wk = qweight.shape
    if wk != k or scales.shape[0] != n:
        raise ValueError(f"incompatible W8A16 argmax shapes: x={tuple(x2d.shape)} qweight={tuple(qweight.shape)} scales={tuple(scales.shape)}")
    rc, index, _value = abi.linear_w8a16_bf16_argmax_ptr(
        x2d.data_ptr(),
        qweight.data_ptr(),
        scales.data_ptr(),
        m,
        n,
        k,
    )
    if rc != 0:
        raise RuntimeError(f"ekernel_linear_w8a16_bf16_argmax failed: rc={rc}")
    return index


def pack_i8_rows_interleaved8(qweight: torch.Tensor) -> torch.Tensor:
    if qweight.device.type != "cpu" or qweight.dtype != torch.int8 or qweight.ndim != 2:
        raise ValueError("qweight must be a 2D CPU torch.int8 tensor")
    n, k = qweight.shape
    blocks = (n + 7) // 8
    if n % 8:
        pad = torch.zeros((blocks * 8 - n, k), dtype=torch.int8)
        padded = torch.cat([qweight.contiguous(), pad], dim=0)
    else:
        padded = qweight.contiguous()
    return padded.reshape(blocks, 8, k).permute(0, 2, 1).contiguous()


def unpack_i4_rows_to_i8(qweight: torch.Tensor, in_features: int | None = None) -> torch.Tensor:
    if qweight.device.type != "cpu" or qweight.dtype != torch.uint8 or qweight.ndim != 2:
        raise ValueError("qweight must be a 2D CPU torch.uint8 tensor")
    qweight = qweight.contiguous()
    n, packed_k = qweight.shape
    k = packed_k * 2 if in_features is None else int(in_features)
    if k <= 0 or packed_k != (k + 1) // 2:
        raise ValueError(f"incompatible W4 packed shape: qweight={tuple(qweight.shape)} in_features={in_features}")

    low = torch.bitwise_and(qweight, 0x0F).to(torch.int16)
    high = torch.bitwise_and(torch.bitwise_right_shift(qweight, 4), 0x0F).to(torch.int16)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)

    unpacked = torch.empty((n, packed_k * 2), dtype=torch.int8)
    unpacked[:, 0::2].copy_(low.to(torch.int8))
    unpacked[:, 1::2].copy_(high.to(torch.int8))
    return unpacked[:, :k].contiguous()


def pack_i4_rows_to_i8_interleaved8(qweight: torch.Tensor, in_features: int | None = None) -> torch.Tensor:
    unpacked = unpack_i4_rows_to_i8(qweight, in_features=in_features)
    n, k = unpacked.shape
    blocks = (n + 7) // 8
    if n % 8:
        pad = torch.zeros((blocks * 8 - n, k), dtype=torch.int8)
        unpacked = torch.cat([unpacked, pad], dim=0)
    return unpacked.reshape(blocks, 8, k).permute(0, 2, 1).contiguous()


def pack_i4_rows_blocked8k16(qweight: torch.Tensor, in_features: int | None = None) -> torch.Tensor:
    if qweight.device.type != "cpu" or qweight.dtype != torch.uint8 or qweight.ndim != 2:
        raise ValueError("qweight must be a 2D CPU torch.uint8 tensor")
    qweight = qweight.contiguous()
    n, packed_k = qweight.shape
    k = packed_k * 2 if in_features is None else int(in_features)
    if k <= 0 or packed_k != (k + 1) // 2:
        raise ValueError(f"incompatible W4 packed shape: qweight={tuple(qweight.shape)} in_features={in_features}")
    row_blocks = (n + 7) // 8
    k_blocks = (k + 15) // 16
    padded_rows = row_blocks * 8
    padded_packed_k = k_blocks * 8
    padded = qweight
    if n != padded_rows or packed_k != padded_packed_k:
        padded_full = torch.zeros((padded_rows, padded_packed_k), dtype=torch.uint8)
        padded_full[:n, :packed_k].copy_(qweight)
        padded = padded_full
    return padded.reshape(row_blocks, 8, k_blocks, 8).permute(0, 2, 1, 3).contiguous()


def linear_w8a16_bf16_i8b8(
    x: torch.Tensor,
    qweight_interleaved: torch.Tensor,
    scales: torch.Tensor,
    out_features: int,
) -> torch.Tensor:
    if x.device.type != "cpu" or x.dtype != torch.bfloat16:
        raise ValueError("x must be a CPU torch.bfloat16 tensor")
    if x.ndim == 1:
        x2d = x.reshape(1, -1)
    elif x.ndim == 2 and x.shape[0] == 1:
        x2d = x
    else:
        raise ValueError(f"x must be a single row, got shape={tuple(x.shape)}")
    if qweight_interleaved.device.type != "cpu" or qweight_interleaved.dtype != torch.int8 or qweight_interleaved.ndim != 3:
        raise ValueError("qweight_interleaved must be a 3D CPU torch.int8 tensor")
    if scales.device.type != "cpu" or scales.dtype != torch.float32 or scales.ndim != 1:
        raise ValueError("scales must be a 1D CPU torch.float32 tensor")
    x2d = x2d.contiguous()
    qweight_interleaved = qweight_interleaved.contiguous()
    scales = scales.contiguous()
    blocks, k, lanes = qweight_interleaved.shape
    if lanes != 8:
        raise ValueError(f"interleaved lanes must be 8, got {lanes}")
    if x2d.shape[1] != k or scales.shape[0] != out_features or blocks != (out_features + 7) // 8:
        raise ValueError(
            "incompatible W8A16 I8B8 shapes: "
            f"x={tuple(x2d.shape)} qweight={tuple(qweight_interleaved.shape)} "
            f"scales={tuple(scales.shape)} out_features={out_features}"
        )
    out = torch.empty((1, out_features), dtype=torch.bfloat16)
    rc = abi.linear_w8a16_bf16_i8b8_ptr(
        x2d.data_ptr(),
        qweight_interleaved.data_ptr(),
        scales.data_ptr(),
        out.data_ptr(),
        1,
        out_features,
        k,
    )
    if rc != 0:
        raise RuntimeError(f"ekernel_linear_w8a16_bf16_i8b8 failed: rc={rc}")
    return out


def linear_w4a16_bf16_i4b8(
    x: torch.Tensor,
    qweight_interleaved: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    if x.device.type != "cpu" or x.dtype != torch.bfloat16:
        raise ValueError("x must be a CPU torch.bfloat16 tensor")
    if x.ndim == 1:
        x2d = x.reshape(1, -1)
    elif x.ndim == 2 and x.shape[0] == 1:
        x2d = x
    else:
        raise ValueError(f"x must be a single row, got shape={tuple(x.shape)}")
    if qweight_interleaved.device.type != "cpu" or qweight_interleaved.dtype != torch.int8 or qweight_interleaved.ndim != 3:
        raise ValueError("qweight_interleaved must be a 3D CPU torch.int8 tensor")
    if scales.device.type != "cpu" or scales.dtype != torch.float32 or scales.ndim != 1:
        raise ValueError("scales must be a 1D CPU torch.float32 tensor")
    x2d = x2d.contiguous()
    qweight_interleaved = qweight_interleaved.contiguous()
    scales = scales.contiguous()
    blocks, k, lanes = qweight_interleaved.shape
    out_features = int(scales.shape[0])
    if lanes != 8:
        raise ValueError(f"interleaved lanes must be 8, got {lanes}")
    if x2d.shape[1] != k or blocks != (out_features + 7) // 8:
        raise ValueError(
            "incompatible W4A16 I4B8 shapes: "
            f"x={tuple(x2d.shape)} qweight={tuple(qweight_interleaved.shape)} "
            f"scales={tuple(scales.shape)}"
        )
    out = torch.empty((1, out_features), dtype=torch.bfloat16)
    rc = abi.linear_w4a16_bf16_i4b8_ptr(
        x2d.data_ptr(),
        qweight_interleaved.data_ptr(),
        scales.data_ptr(),
        out.data_ptr(),
        1,
        out_features,
        k,
    )
    if rc != 0:
        raise RuntimeError(f"ekernel_linear_w4a16_bf16_i4b8 failed: rc={rc}")
    return out


def linear_w4a16_bf16_b8(
    x: torch.Tensor,
    qweight_blocked: torch.Tensor,
    scales: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    if x.device.type != "cpu" or x.dtype != torch.bfloat16:
        raise ValueError("x must be a CPU torch.bfloat16 tensor")
    if x.ndim == 1:
        x2d = x.reshape(1, -1)
    elif x.ndim == 2 and x.shape[0] == 1:
        x2d = x
    else:
        raise ValueError(f"experimental b8 W4 kernel currently supports a single row, got shape={tuple(x.shape)}")
    if qweight_blocked.device.type != "cpu" or qweight_blocked.dtype != torch.uint8 or qweight_blocked.ndim != 4:
        raise ValueError("qweight_blocked must be a 4D CPU torch.uint8 tensor")
    if scales.device.type != "cpu" or scales.dtype != torch.float32 or scales.ndim != 1:
        raise ValueError("scales must be a 1D CPU torch.float32 tensor")
    x2d = x2d.contiguous()
    qweight_blocked = qweight_blocked.contiguous()
    scales = scales.contiguous()
    m, k = x2d.shape
    n = int(out_features)
    expected_shape = ((n + 7) // 8, (int(in_features) + 15) // 16, 8, 8)
    if int(in_features) != k or tuple(qweight_blocked.shape) != expected_shape or scales.shape[0] != n:
        raise ValueError(
            "incompatible W4 b8 shapes: "
            f"x={tuple(x2d.shape)} qweight_blocked={tuple(qweight_blocked.shape)} "
            f"scales={tuple(scales.shape)} out_features={out_features} in_features={in_features}"
        )
    out = torch.empty((m, n), dtype=torch.bfloat16)
    rc = abi.linear_w4a16_bf16_b8_ptr(
        x2d.data_ptr(),
        qweight_blocked.data_ptr(),
        scales.data_ptr(),
        out.data_ptr(),
        m,
        n,
        k,
    )
    if rc != 0:
        raise RuntimeError(f"ekernel_linear_w4a16_bf16_b8 failed: rc={rc}")
    return out


def linear_w4a16_bf16(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, in_features: int | None = None) -> torch.Tensor:
    if x.device.type != "cpu" or x.dtype != torch.bfloat16 or x.ndim != 2:
        raise ValueError("x must be a 2D CPU torch.bfloat16 tensor")
    if qweight.device.type != "cpu" or qweight.dtype != torch.uint8 or qweight.ndim != 2:
        raise ValueError("qweight must be a 2D CPU torch.uint8 tensor")
    if scales.device.type != "cpu" or scales.dtype != torch.float32 or scales.ndim != 1:
        raise ValueError("scales must be a 1D CPU torch.float32 tensor")
    x = x.contiguous()
    qweight = qweight.contiguous()
    scales = scales.contiguous()
    m, k = x.shape
    n, packed_k = qweight.shape
    expected_packed = (k + 1) // 2
    if in_features is not None and int(in_features) != k:
        raise ValueError(f"in_features={in_features} does not match x K={k}")
    if packed_k != expected_packed or scales.shape[0] != n:
        raise ValueError(f"incompatible W4A16 shapes: x={tuple(x.shape)} qweight={tuple(qweight.shape)} scales={tuple(scales.shape)}")
    out = torch.empty((m, n), dtype=torch.bfloat16)
    rc = abi.linear_w4a16_bf16_ptr(x.data_ptr(), qweight.data_ptr(), scales.data_ptr(), out.data_ptr(), m, n, k)
    if rc != 0:
        raise RuntimeError(f"ekernel_linear_w4a16_bf16 failed: rc={rc}")
    return out


def linear_w4a16_bf16_q8(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, in_features: int | None = None) -> torch.Tensor:
    """Experimental W4A16 BF16 linear with on-the-fly int8 activation quantization.

    Accepts the same packed int4 qweight and per-row fp32 scales as
    `linear_w4a16_bf16`; the activation row is quantized to signed int8 inside
    the C kernel and the dot product uses an AVX2 integer (pmaddwd) path.  The
    only added error versus `linear_w4a16_bf16` is the int8 activation rounding.
    Requires a loaded eCPU library that exposes `ekernel_linear_w4a16_bf16_q8`.
    """
    if x.device.type != "cpu" or x.dtype != torch.bfloat16 or x.ndim != 2:
        raise ValueError("x must be a 2D CPU torch.bfloat16 tensor")
    if qweight.device.type != "cpu" or qweight.dtype != torch.uint8 or qweight.ndim != 2:
        raise ValueError("qweight must be a 2D CPU torch.uint8 tensor")
    if scales.device.type != "cpu" or scales.dtype != torch.float32 or scales.ndim != 1:
        raise ValueError("scales must be a 1D CPU torch.float32 tensor")
    x = x.contiguous()
    qweight = qweight.contiguous()
    scales = scales.contiguous()
    m, k = x.shape
    n, packed_k = qweight.shape
    expected_packed = (k + 1) // 2
    if in_features is not None and int(in_features) != k:
        raise ValueError(f"in_features={in_features} does not match x K={k}")
    if packed_k != expected_packed or scales.shape[0] != n:
        raise ValueError(f"incompatible W4A16-Q8 shapes: x={tuple(x.shape)} qweight={tuple(qweight.shape)} scales={tuple(scales.shape)}")
    out = torch.empty((m, n), dtype=torch.bfloat16)
    rc = abi.linear_w4a16_bf16_q8_ptr(x.data_ptr(), qweight.data_ptr(), scales.data_ptr(), out.data_ptr(), m, n, k)
    if rc != 0:
        raise RuntimeError(f"ekernel_linear_w4a16_bf16_q8 failed: rc={rc}")
    return out


def linear_w4a16g32_bf16(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, in_features: int | None = None) -> torch.Tensor:
    if x.device.type != "cpu" or x.dtype != torch.bfloat16 or x.ndim != 2:
        raise ValueError("x must be a 2D CPU torch.bfloat16 tensor")
    if qweight.device.type != "cpu" or qweight.dtype != torch.uint8 or qweight.ndim != 2:
        raise ValueError("qweight must be a 2D CPU torch.uint8 tensor")
    if scales.device.type != "cpu" or scales.dtype != torch.float32 or scales.ndim != 2:
        raise ValueError("scales must be a 2D CPU torch.float32 tensor")
    x = x.contiguous()
    qweight = qweight.contiguous()
    scales = scales.contiguous()
    m, k = x.shape
    n, packed_k = qweight.shape
    groups = (k + 31) // 32
    if in_features is not None and int(in_features) != k:
        raise ValueError(f"in_features={in_features} does not match x K={k}")
    if packed_k != (k + 1) // 2 or tuple(scales.shape) != (n, groups):
        raise ValueError(f"incompatible W4A16G32 shapes: x={tuple(x.shape)} qweight={tuple(qweight.shape)} scales={tuple(scales.shape)}")
    out = torch.empty((m, n), dtype=torch.bfloat16)
    rc = abi.linear_w4a16g32_bf16_ptr(x.data_ptr(), qweight.data_ptr(), scales.data_ptr(), out.data_ptr(), m, n, k)
    if rc != 0:
        raise RuntimeError(f"ekernel_linear_w4a16g32_bf16 failed: rc={rc}")
    return out


def linear_w4a16g128_bf16(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, in_features: int | None = None) -> torch.Tensor:
    if x.device.type != "cpu" or x.dtype != torch.bfloat16 or x.ndim != 2:
        raise ValueError("x must be a 2D CPU torch.bfloat16 tensor")
    if qweight.device.type != "cpu" or qweight.dtype != torch.uint8 or qweight.ndim != 2:
        raise ValueError("qweight must be a 2D CPU torch.uint8 tensor")
    if scales.device.type != "cpu" or scales.dtype != torch.float32 or scales.ndim != 2:
        raise ValueError("scales must be a 2D CPU torch.float32 tensor")
    x = x.contiguous()
    qweight = qweight.contiguous()
    scales = scales.contiguous()
    m, k = x.shape
    n, packed_k = qweight.shape
    groups = (k + 127) // 128
    if in_features is not None and int(in_features) != k:
        raise ValueError(f"in_features={in_features} does not match x K={k}")
    if packed_k != (k + 1) // 2 or tuple(scales.shape) != (n, groups):
        raise ValueError(f"incompatible W4A16G128 shapes: x={tuple(x.shape)} qweight={tuple(qweight.shape)} scales={tuple(scales.shape)}")
    out = torch.empty((m, n), dtype=torch.bfloat16)
    rc = abi.linear_w4a16g128_bf16_ptr(x.data_ptr(), qweight.data_ptr(), scales.data_ptr(), out.data_ptr(), m, n, k)
    if rc != 0:
        raise RuntimeError(f"ekernel_linear_w4a16g128_bf16 failed: rc={rc}")
    return out


def swiglu_bf16(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if gate.device.type != "cpu" or up.device.type != "cpu":
        raise ValueError("gate and up must be CPU tensors")
    if gate.dtype != torch.bfloat16 or up.dtype != torch.bfloat16:
        raise ValueError("gate and up must be torch.bfloat16")
    if tuple(gate.shape) != tuple(up.shape):
        raise ValueError(f"gate/up shapes differ: gate={tuple(gate.shape)} up={tuple(up.shape)}")
    gate = gate.contiguous()
    up = up.contiguous()
    out = torch.empty_like(gate)
    rc = abi.swiglu_bf16_ptr(gate.data_ptr(), up.data_ptr(), out.data_ptr(), gate.numel())
    if rc != 0:
        raise RuntimeError(f"ekernel_swiglu_bf16 failed: rc={rc}")
    return out


def swiglu_w4a16_bf16(
    x: torch.Tensor,
    gate_up_qweight: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_qweight: torch.Tensor,
    down_scales: torch.Tensor,
    intermediate_size: int,
    gate_up_in_features: int | None = None,
    down_in_features: int | None = None,
) -> torch.Tensor:
    """Experimental decode-only fused W4 SwiGLU MLP kernel."""
    if x.device.type != "cpu" or x.dtype != torch.bfloat16 or x.ndim != 2:
        raise ValueError("x must be a 2D CPU torch.bfloat16 tensor")
    if gate_up_qweight.device.type != "cpu" or gate_up_qweight.dtype != torch.uint8 or gate_up_qweight.ndim != 2:
        raise ValueError("gate_up_qweight must be a 2D CPU torch.uint8 tensor")
    if down_qweight.device.type != "cpu" or down_qweight.dtype != torch.uint8 or down_qweight.ndim != 2:
        raise ValueError("down_qweight must be a 2D CPU torch.uint8 tensor")
    if gate_up_scales.device.type != "cpu" or gate_up_scales.dtype != torch.float32 or gate_up_scales.ndim != 1:
        raise ValueError("gate_up_scales must be a 1D CPU torch.float32 tensor")
    if down_scales.device.type != "cpu" or down_scales.dtype != torch.float32 or down_scales.ndim != 1:
        raise ValueError("down_scales must be a 1D CPU torch.float32 tensor")
    x = x.contiguous()
    gate_up_qweight = gate_up_qweight.contiguous()
    gate_up_scales = gate_up_scales.contiguous()
    down_qweight = down_qweight.contiguous()
    down_scales = down_scales.contiguous()
    m, k = x.shape
    intermediate_size = int(intermediate_size)
    gate_up_in_features = k if gate_up_in_features is None else int(gate_up_in_features)
    down_in_features = intermediate_size if down_in_features is None else int(down_in_features)
    hidden_size = int(down_qweight.shape[0])
    if m != 1:
        raise ValueError("experimental fused W4 SwiGLU currently supports M=1 only")
    if gate_up_in_features != k:
        raise ValueError(f"gate_up_in_features={gate_up_in_features} does not match x K={k}")
    if down_in_features != intermediate_size:
        raise ValueError("down_in_features must equal intermediate_size")
    if gate_up_qweight.shape != (intermediate_size * 2, (gate_up_in_features + 1) // 2):
        raise ValueError(
            "gate_up_qweight shape must be "
            f"{(intermediate_size * 2, (gate_up_in_features + 1) // 2)}, got {tuple(gate_up_qweight.shape)}"
        )
    if down_qweight.shape[1] != (down_in_features + 1) // 2:
        raise ValueError(f"down_qweight packed K does not match down_in_features={down_in_features}")
    if gate_up_scales.shape[0] != intermediate_size * 2 or down_scales.shape[0] != hidden_size:
        raise ValueError("scale shapes do not match qweight rows")
    out = torch.empty((m, hidden_size), dtype=torch.bfloat16)
    rc = abi.swiglu_w4a16_bf16_ptr(
        x.data_ptr(),
        gate_up_qweight.data_ptr(),
        gate_up_scales.data_ptr(),
        down_qweight.data_ptr(),
        down_scales.data_ptr(),
        out.data_ptr(),
        m,
        hidden_size,
        intermediate_size,
        gate_up_in_features,
        down_in_features,
    )
    if rc != 0:
        raise RuntimeError(f"ekernel_swiglu_w4a16_bf16 failed: rc={rc}")
    return out


def rmsnorm_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    add_one: bool = False,
) -> torch.Tensor:
    if x.device.type != "cpu" or x.dtype != torch.bfloat16:
        raise ValueError("x must be a CPU torch.bfloat16 tensor")
    if weight.device.type != "cpu" or weight.dtype != torch.bfloat16 or weight.ndim != 1:
        raise ValueError("weight must be a 1D CPU torch.bfloat16 tensor")
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(f"incompatible RMSNorm shapes: x={tuple(x.shape)} weight={tuple(weight.shape)}")

    original_shape: Tuple[int, ...] = tuple(x.shape)
    x2d = x.reshape(-1, x.shape[-1]).contiguous()
    weight = weight.contiguous()
    rows, n = x2d.shape
    out = torch.empty_like(x2d)
    rc = abi.rmsnorm_bf16_ptr(
        x2d.data_ptr(),
        out.data_ptr(),
        weight.data_ptr(),
        rows,
        n,
        float(eps),
        bool(add_one),
    )
    if rc != 0:
        raise RuntimeError(f"ekernel_rmsnorm_bf16 failed: rc={rc}")
    return out.reshape(original_shape)


def gated_delta_recurrent_f32(
    state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if state.device.type != "cpu" or state.dtype != torch.float32 or state.ndim != 3:
        raise ValueError("state must be a 3D CPU torch.float32 tensor")
    if q.device.type != "cpu" or k.device.type != "cpu" or v.device.type != "cpu":
        raise ValueError("q, k, and v must be CPU tensors")
    if q.dtype != torch.float32 or k.dtype != torch.float32 or v.dtype != torch.float32:
        raise ValueError("q, k, and v must be torch.float32")
    if beta.device.type != "cpu" or decay.device.type != "cpu":
        raise ValueError("beta and decay must be CPU tensors")
    if beta.dtype != torch.float32 or decay.dtype != torch.float32:
        raise ValueError("beta and decay must be torch.float32")
    n_heads, k_dim, v_dim = state.shape
    if tuple(q.shape) != (n_heads, k_dim) or tuple(k.shape) != (n_heads, k_dim):
        raise ValueError(f"q/k shapes must be {(n_heads, k_dim)}, got q={tuple(q.shape)} k={tuple(k.shape)}")
    if tuple(v.shape) != (n_heads, v_dim):
        raise ValueError(f"v shape must be {(n_heads, v_dim)}, got {tuple(v.shape)}")
    if tuple(beta.shape) != (n_heads,) or tuple(decay.shape) != (n_heads,):
        raise ValueError(f"beta/decay shapes must be {(n_heads,)}, got beta={tuple(beta.shape)} decay={tuple(decay.shape)}")

    state = state.contiguous()
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    beta = beta.contiguous()
    decay = decay.contiguous()
    out = torch.empty((n_heads, v_dim), dtype=torch.float32)
    rc = abi.gated_delta_recurrent_f32_ptr(
        state.data_ptr(),
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        beta.data_ptr(),
        decay.data_ptr(),
        out.data_ptr(),
        n_heads,
        k_dim,
        v_dim,
        float(scale),
    )
    if rc != 0:
        raise RuntimeError(f"ekernel_gated_delta_recurrent_f32 failed: rc={rc}")
    return out, state
