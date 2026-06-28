from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch


FP4_E2M1_MAX = 6.0
FP4_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP8_MAX = {
    "fp8": 448.0,
    "fp8_e4m3": 448.0,
    "fp8_e4m3fn": 448.0,
    "fp8_e5m2": 57344.0,
}

SCALE_GRANULARITIES = (
    "per_block",
    "per_channel_block",
    "per_channel",
    "per_tensor",
)

SCALE_GRANULARITY_ALIASES = {
    "position": "position",
    "column": "per_channel",
    "channel": "per_channel",
    "per_channel": "per_channel",
    "channel_block": "per_channel_block",
    "per_channel_block": "per_channel_block",
    "row": "row",
    "batch": "batch",
    "tensor": "per_tensor",
    "per_tensor": "per_tensor",
    "block": "per_block",
    "per_block": "per_block",
}


def default_int4_quantile() -> float:
    q = float(os.environ.get("KVARN_RTN_QUANTILE", "0.005") or 0.005)
    return q if 0.0 < q < 0.5 else 0.005


def fp4_pow2_scale_exp(absmax: torch.Tensor) -> torch.Tensor:
    return pow2_scale_exp(absmax / FP4_E2M1_MAX)


def pow2_scale_exp(scale: torch.Tensor) -> torch.Tensor:
    mant, exp = torch.frexp(scale)
    exp2 = torch.where(mant <= 0.5, exp - 1, exp)
    return torch.where(scale > 0, exp2, torch.zeros_like(exp2)).to(torch.int64)


def fp4_e2m1_nearest(x: torch.Tensor) -> torch.Tensor:
    grid = torch.tensor(FP4_E2M1_GRID, dtype=x.dtype, device=x.device)
    sign = torch.sign(x)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    mag = x.abs().clamp(max=FP4_E2M1_MAX)
    idx = (mag.unsqueeze(-1) - grid).abs().argmin(dim=-1)
    return sign * grid[idx]


def compute_fp4_scale(absmax: torch.Tensor, dtype_name: str) -> torch.Tensor:
    return compute_scale(absmax, FP4_E2M1_MAX, dtype_name)


def compute_scale(
    absmax: torch.Tensor, quant_max: float, dtype_name: str
) -> torch.Tensor:
    if dtype_name == "pow2":
        exp = pow2_scale_exp(absmax / quant_max)
        return torch.exp2(exp.float())
    scale = (absmax / quant_max).clamp_min(1e-12)
    if dtype_name == "fp32":
        return scale.float()
    if dtype_name == "fp16":
        return scale.to(torch.float16).float()
    if dtype_name == "bf16":
        return scale.to(torch.bfloat16).float()
    if dtype_name in {"fp8", "fp8_e4m3", "fp8_e4m3fn"}:
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("This torch build does not expose float8_e4m3fn.")
        return scale.to(torch.float8_e4m3fn).float()
    if dtype_name == "fp8_e5m2":
        if not hasattr(torch, "float8_e5m2"):
            raise RuntimeError("This torch build does not expose float8_e5m2.")
        return scale.to(torch.float8_e5m2).float()
    if dtype_name == "fp4":
        return fp4_e2m1_nearest(scale.float()).clamp_min(1e-12)
    raise ValueError(f"unknown scale dtype: {dtype_name}")


def fp8_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name in {"fp8", "fp8_e4m3", "fp8_e4m3fn"}:
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("This torch build does not expose float8_e4m3fn.")
        return torch.float8_e4m3fn
    if dtype_name == "fp8_e5m2":
        if not hasattr(torch, "float8_e5m2"):
            raise RuntimeError("This torch build does not expose float8_e5m2.")
        return torch.float8_e5m2
    raise ValueError(f"unknown FP8 dtype: {dtype_name}")


def normalize_granularity(granularity: str) -> str:
    if granularity not in SCALE_GRANULARITY_ALIASES:
        raise ValueError(f"unknown scale granularity: {granularity}")
    return SCALE_GRANULARITY_ALIASES[granularity]


def scale_reduce_dims(x: torch.Tensor, granularity: str) -> tuple[int, ...]:
    granularity = normalize_granularity(granularity)
    if x.ndim != 3:
        if granularity in {"position", "per_channel"}:
            return (0,)
        if granularity == "row":
            return (1,)
        if granularity == "per_tensor":
            return tuple(range(x.ndim))
        raise ValueError(f"{granularity} granularity expects a 3D tensor")
    if granularity == "position":
        return (0,)
    if granularity == "per_channel":
        return (0, 1)
    if granularity == "row":
        return (0, 2)
    if granularity == "batch":
        return (1, 2)
    if granularity == "per_tensor":
        return (0, 1, 2)
    if granularity in {"per_block", "per_channel_block"}:
        raise ValueError(f"{granularity} does not use reduce dims")
    raise ValueError(f"unknown scale granularity: {granularity}")


def absmax_for_scale(x: torch.Tensor, granularity: str) -> torch.Tensor:
    return x.abs().amax(dim=scale_reduce_dims(x, granularity), keepdim=True)


def aminmax_for_scale(
    x: torch.Tensor, granularity: str
) -> tuple[torch.Tensor, torch.Tensor]:
    dims = scale_reduce_dims(x, granularity)
    return x.amin(dim=dims, keepdim=True), x.amax(dim=dims, keepdim=True)


def quantile_range(
    x: torch.Tensor, q: float, dims: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < q < 0.5:
        raise ValueError(f"quantile must be in (0, 0.5), got {q}")
    dims = tuple(dim if dim >= 0 else x.ndim + dim for dim in dims)
    keep_dims = tuple(dim for dim in range(x.ndim) if dim not in dims)
    permuted = x.permute(*keep_dims, *dims)
    keep_shape = [x.shape[dim] for dim in keep_dims]
    reduce_numel = math.prod(x.shape[dim] for dim in dims)
    flattened = permuted.reshape(*keep_shape, reduce_numel)
    reduce_numel = flattened.shape[-1]
    lo_k = max(1, min(reduce_numel, math.ceil(q * reduce_numel)))
    hi_k = max(1, min(reduce_numel, math.ceil((1.0 - q) * reduce_numel)))
    lo = flattened.kthvalue(lo_k, dim=-1).values
    hi = flattened.kthvalue(hi_k, dim=-1).values
    out_shape = [1] * x.ndim
    for index, dim in enumerate(keep_dims):
        out_shape[dim] = keep_shape[index]
    return lo.reshape(out_shape), hi.reshape(out_shape)


def quantile_absmax(x: torch.Tensor, q: float, dims: tuple[int, ...]) -> torch.Tensor:
    if not 0.0 < q < 0.5:
        raise ValueError(f"quantile must be in (0, 0.5), got {q}")
    dims = tuple(dim if dim >= 0 else x.ndim + dim for dim in dims)
    keep_dims = tuple(dim for dim in range(x.ndim) if dim not in dims)
    permuted = x.abs().permute(*keep_dims, *dims)
    keep_shape = [x.shape[dim] for dim in keep_dims]
    reduce_numel = math.prod(x.shape[dim] for dim in dims)
    flattened = permuted.reshape(*keep_shape, reduce_numel)
    k = max(1, min(reduce_numel, math.ceil((1.0 - q) * reduce_numel)))
    absmax = flattened.kthvalue(k, dim=-1).values
    out_shape = [1] * x.ndim
    for index, dim in enumerate(keep_dims):
        out_shape[dim] = keep_shape[index]
    return absmax.reshape(out_shape)


def block_view(x: torch.Tensor, block_size: int) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError("per_block requires a 3D tensor")
    batch_size, rows, cols = x.shape
    if (batch_size * rows) % block_size != 0 or cols % block_size != 0:
        raise ValueError(
            "per_block requires batch_size*rows and cols divisible by block size"
        )
    return x.reshape(
        batch_size * rows // block_size,
        block_size,
        cols // block_size,
        block_size,
    ).permute(0, 2, 1, 3)


def unblock_view(view: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    batch_size, rows, cols = shape
    return view.permute(0, 2, 1, 3).reshape(batch_size, rows, cols)


def channel_block_view(x: torch.Tensor, block_size: int) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError("per_channel_block requires a 3D tensor")
    batch_size, rows, cols = x.shape
    if cols % block_size != 0:
        raise ValueError(
            "per_channel_block requires cols divisible by block size"
        )
    return x.reshape(batch_size, rows, cols // block_size, block_size)


def unchannel_block_view(view: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    batch_size, rows, cols = shape
    return view.reshape(batch_size, rows, cols)


def quant_int4(
    x: torch.Tensor,
    granularity: str,
    block_size: int,
    quant_scheme: str = "asym",
    quantile: float | None = None,
) -> torch.Tensor:
    granularity = normalize_granularity(granularity)
    if granularity == "per_block":
        view = block_view(x, block_size)
        if quant_scheme == "sym":
            qmax = 7
            absmax = view.abs().amax(dim=(2, 3), keepdim=True)
            scale = (absmax / qmax).clamp_min(1e-12)
            q = torch.clamp(torch.round(view / scale), -qmax, qmax)
            return unblock_view(q * scale, x.shape)
        if quant_scheme == "asym":
            qmax = 15
            lo = view.amin(dim=(2, 3), keepdim=True)
            hi = view.amax(dim=(2, 3), keepdim=True)
            scale = ((hi - lo) / qmax).clamp_min(1e-12)
            q = torch.clamp(torch.round((view - lo) / scale), 0, qmax)
            return unblock_view(q * scale + lo, x.shape)
        if quant_scheme == "asym_quantile":
            qmax = 15
            q = 0.005 if quantile is None else quantile
            lo, hi = quantile_range(view, q, dims=(2, 3))
            scale = ((hi - lo) / qmax).clamp_min(1e-10)
            quant = torch.clamp(torch.round((view - lo) / scale), 0, qmax)
            return unblock_view(quant * scale + lo, x.shape)
        raise ValueError(f"unknown int4 quant scheme: {quant_scheme}")
    if granularity == "per_channel_block":
        view = channel_block_view(x, block_size)
        if quant_scheme == "sym":
            qmax = 7
            absmax = view.abs().amax(dim=(0, 1, 3), keepdim=True)
            scale = (absmax / qmax).clamp_min(1e-12)
            q = torch.clamp(torch.round(view / scale), -qmax, qmax)
            return unchannel_block_view(q * scale, x.shape)
        if quant_scheme == "asym":
            qmax = 15
            lo = view.amin(dim=(0, 1, 3), keepdim=True)
            hi = view.amax(dim=(0, 1, 3), keepdim=True)
            scale = ((hi - lo) / qmax).clamp_min(1e-12)
            q = torch.clamp(torch.round((view - lo) / scale), 0, qmax)
            return unchannel_block_view(q * scale + lo, x.shape)
        if quant_scheme == "asym_quantile":
            qmax = 15
            q = 0.005 if quantile is None else quantile
            lo, hi = quantile_range(view, q, dims=(0, 1, 3))
            scale = ((hi - lo) / qmax).clamp_min(1e-10)
            quant = torch.clamp(torch.round((view - lo) / scale), 0, qmax)
            return unchannel_block_view(quant * scale + lo, x.shape)
        raise ValueError(f"unknown int4 quant scheme: {quant_scheme}")
    if quant_scheme == "sym":
        qmax = 7
        scale = (absmax_for_scale(x, granularity) / qmax).clamp_min(1e-12)
        q = torch.clamp(torch.round(x / scale), -qmax, qmax)
        return q * scale
    if quant_scheme == "asym":
        qmax = 15
        lo, hi = aminmax_for_scale(x, granularity)
        scale = ((hi - lo) / qmax).clamp_min(1e-12)
        q = torch.clamp(torch.round((x - lo) / scale), 0, qmax)
        return q * scale + lo
    if quant_scheme == "asym_quantile":
        qmax = 15
        q = 0.005 if quantile is None else quantile
        lo, hi = quantile_range(x, q, dims=scale_reduce_dims(x, granularity))
        scale = ((hi - lo) / qmax).clamp_min(1e-10)
        quant = torch.clamp(torch.round((x - lo) / scale), 0, qmax)
        return quant * scale + lo
    raise ValueError(f"unknown int4 quant scheme: {quant_scheme}")


def quant_int8(
    x: torch.Tensor,
    granularity: str,
    scale_dtype: str = "fp32",
    quant_scheme: str = "sym",
    block_size: int = 128,
) -> torch.Tensor:
    granularity = normalize_granularity(granularity)
    if granularity == "per_block":
        view = block_view(x, block_size)
        if quant_scheme == "sym":
            qmax = 127
            absmax = view.abs().amax(dim=(2, 3), keepdim=True)
            scale = compute_scale(absmax, qmax, scale_dtype)
            q = torch.clamp(torch.round(view / scale), -qmax, qmax)
            return unblock_view((q * scale).to(x.dtype), x.shape)
        if quant_scheme == "asym":
            qmin = -128.0
            qmax = 127.0
            lo = view.amin(dim=(2, 3), keepdim=True)
            hi = view.amax(dim=(2, 3), keepdim=True)
            scale = compute_scale(hi - lo, qmax - qmin, scale_dtype)
            zp = torch.round(qmin - lo / scale).clamp(qmin, qmax)
            q = torch.clamp(torch.round(view / scale + zp), qmin, qmax)
            return unblock_view(((q - zp) * scale).to(x.dtype), x.shape)
        raise ValueError(f"unknown int quant scheme: {quant_scheme}")
    if granularity == "per_channel_block":
        view = channel_block_view(x, block_size)
        if quant_scheme == "sym":
            qmax = 127
            absmax = view.abs().amax(dim=(0, 1, 3), keepdim=True)
            scale = compute_scale(absmax, qmax, scale_dtype)
            q = torch.clamp(torch.round(view / scale), -qmax, qmax)
            return unchannel_block_view((q * scale).to(x.dtype), x.shape)
        if quant_scheme == "asym":
            qmin = -128.0
            qmax = 127.0
            lo = view.amin(dim=(0, 1, 3), keepdim=True)
            hi = view.amax(dim=(0, 1, 3), keepdim=True)
            scale = compute_scale(hi - lo, qmax - qmin, scale_dtype)
            zp = torch.round(qmin - lo / scale).clamp(qmin, qmax)
            q = torch.clamp(torch.round(view / scale + zp), qmin, qmax)
            return unchannel_block_view(((q - zp) * scale).to(x.dtype), x.shape)
        raise ValueError(f"unknown int quant scheme: {quant_scheme}")
    if quant_scheme == "sym":
        qmax = 127
        scale = compute_scale(absmax_for_scale(x, granularity), qmax, scale_dtype)
        q = torch.clamp(torch.round(x / scale), -qmax, qmax)
        return (q * scale).to(x.dtype)
    if quant_scheme == "asym":
        qmin = -128.0
        qmax = 127.0
        lo, hi = aminmax_for_scale(x, granularity)
        scale = compute_scale(hi - lo, qmax - qmin, scale_dtype)
        zp = torch.round(qmin - lo / scale).clamp(qmin, qmax)
        q = torch.clamp(torch.round(x / scale + zp), qmin, qmax)
        return ((q - zp) * scale).to(x.dtype)
    raise ValueError(f"unknown int quant scheme: {quant_scheme}")


def quant_fp4(
    x: torch.Tensor,
    granularity: str,
    scale_dtype: str,
    block_size: int,
    quantile: float | None = None,
) -> torch.Tensor:
    granularity = normalize_granularity(granularity)
    if granularity == "per_block":
        view = block_view(x, block_size)
        if quantile is None:
            absmax = view.abs().amax(dim=(2, 3), keepdim=True)
        else:
            absmax = quantile_absmax(view, quantile, dims=(2, 3))
        scale = compute_fp4_scale(absmax, scale_dtype)
        scaled = view.float() / scale
        q = fp4_e2m1_nearest(scaled)
        return unblock_view((q * scale).to(x.dtype), x.shape)
    if granularity == "per_channel_block":
        view = channel_block_view(x, block_size)
        if quantile is None:
            absmax = view.abs().amax(dim=(0, 1, 3), keepdim=True)
        else:
            absmax = quantile_absmax(view, quantile, dims=(0, 1, 3))
        scale = compute_fp4_scale(absmax, scale_dtype)
        scaled = view.float() / scale
        q = fp4_e2m1_nearest(scaled)
        return unchannel_block_view((q * scale).to(x.dtype), x.shape)
    if quantile is None:
        absmax = absmax_for_scale(x, granularity)
    else:
        absmax = quantile_absmax(x, quantile, dims=scale_reduce_dims(x, granularity))
    scale = compute_fp4_scale(absmax, scale_dtype)
    scaled = x.float() / scale
    q = fp4_e2m1_nearest(scaled)
    return (q * scale).to(x.dtype)


def quant_fp8(
    x: torch.Tensor,
    value_dtype: str,
    granularity: str,
    scale_dtype: str,
    block_size: int,
    quantile: float | None = None,
) -> torch.Tensor:
    granularity = normalize_granularity(granularity)
    if granularity == "per_block":
        view = block_view(x, block_size)
        if quantile is None:
            absmax = view.abs().amax(dim=(2, 3), keepdim=True)
        else:
            absmax = quantile_absmax(view, quantile, dims=(2, 3))
        scale = compute_scale(absmax, FP8_MAX[value_dtype], scale_dtype)
        scaled = (view.float() / scale).clamp(
            -FP8_MAX[value_dtype], FP8_MAX[value_dtype]
        )
        q = scaled.to(fp8_dtype(value_dtype)).float()
        return unblock_view((q * scale).to(x.dtype), x.shape)
    if granularity == "per_channel_block":
        view = channel_block_view(x, block_size)
        if quantile is None:
            absmax = view.abs().amax(dim=(0, 1, 3), keepdim=True)
        else:
            absmax = quantile_absmax(view, quantile, dims=(0, 1, 3))
        scale = compute_scale(absmax, FP8_MAX[value_dtype], scale_dtype)
        scaled = (view.float() / scale).clamp(
            -FP8_MAX[value_dtype], FP8_MAX[value_dtype]
        )
        q = scaled.to(fp8_dtype(value_dtype)).float()
        return unchannel_block_view((q * scale).to(x.dtype), x.shape)
    if quantile is None:
        absmax = absmax_for_scale(x, granularity)
    else:
        absmax = quantile_absmax(x, quantile, dims=scale_reduce_dims(x, granularity))
    scale = compute_scale(absmax, FP8_MAX[value_dtype], scale_dtype)
    scaled = (x.float() / scale).clamp(
        -FP8_MAX[value_dtype], FP8_MAX[value_dtype]
    )
    q = scaled.to(fp8_dtype(value_dtype)).float()
    return (q * scale).to(x.dtype)


def quant_fp4_zp(
    x: torch.Tensor, granularity: str, scale_dtype: str, block_size: int
) -> torch.Tensor:
    granularity = normalize_granularity(granularity)
    if granularity == "per_block":
        view = block_view(x, block_size)
        lo = view.amin(dim=(2, 3), keepdim=True)
        hi = view.amax(dim=(2, 3), keepdim=True)
        center = 0.5 * (lo + hi)
        shifted = view - center
        absmax = shifted.abs().amax(dim=(2, 3), keepdim=True)
        scale = compute_fp4_scale(absmax, scale_dtype)
        q = fp4_e2m1_nearest(shifted.float() / scale)
        return unblock_view((q * scale + center).to(x.dtype), x.shape)
    if granularity == "per_channel_block":
        view = channel_block_view(x, block_size)
        lo = view.amin(dim=(0, 1, 3), keepdim=True)
        hi = view.amax(dim=(0, 1, 3), keepdim=True)
        center = 0.5 * (lo + hi)
        shifted = view - center
        absmax = shifted.abs().amax(dim=(0, 1, 3), keepdim=True)
        scale = compute_fp4_scale(absmax, scale_dtype)
        q = fp4_e2m1_nearest(shifted.float() / scale)
        return unchannel_block_view((q * scale + center).to(x.dtype), x.shape)
    lo, hi = aminmax_for_scale(x, granularity)
    center = 0.5 * (lo + hi)
    shifted = x - center
    return quant_fp4(shifted, granularity, scale_dtype, block_size) + center


def metrics(ref: torch.Tensor, got: torch.Tensor) -> dict[str, float]:
    err = got.float() - ref.float()
    mse = float(err.square().mean().item())
    signal = float(ref.float().square().mean().item())
    return {
        "mse": mse,
        "relative_mse": mse / signal if signal > 0 else math.nan,
        "rmse": math.sqrt(mse),
        "mae": float(err.abs().mean().item()),
        "max_abs": float(err.abs().max().item()),
        "sqnr_db": 10.0 * math.log10(signal / mse) if mse > 0 else math.inf,
    }


def tensor_summary(tensor: torch.Tensor) -> dict[str, float | int]:
    finite = torch.isfinite(tensor)
    summary: dict[str, float | int] = {
        "numel": tensor.numel(),
        "finite": int(finite.sum().item()),
        "nan": int(torch.isnan(tensor).sum().item()),
        "posinf": int(torch.isposinf(tensor).sum().item()),
        "neginf": int(torch.isneginf(tensor).sum().item()),
        "zero": int((tensor == 0).sum().item()),
    }
    if finite.any():
        finite_values = tensor[finite].float()
        nonzero = finite_values[finite_values != 0]
        summary.update(
            {
                "min": float(finite_values.min().item()),
                "max": float(finite_values.max().item()),
                "mean": float(finite_values.mean().item()),
            }
        )
        if nonzero.numel() > 0:
            summary["min_abs_nonzero"] = float(nonzero.abs().min().item())
    return summary


def print_tensor_summary(label: str, tensor: torch.Tensor) -> None:
    summary = tensor_summary(tensor)
    stats = " ".join(f"{key}={value}" for key, value in summary.items())
    print(f"[DEBUG] {label}: {stats}")


def fp_absmax_for_quant(
    x: torch.Tensor,
    granularity: str,
    block_size: int,
    zero_point: bool = False,
) -> torch.Tensor:
    granularity = normalize_granularity(granularity)
    if granularity == "per_block":
        view = block_view(x, block_size)
        if zero_point:
            lo = view.amin(dim=(2, 3), keepdim=True)
            hi = view.amax(dim=(2, 3), keepdim=True)
            view = view - 0.5 * (lo + hi)
        return view.abs().amax(dim=(2, 3), keepdim=True)
    if granularity == "per_channel_block":
        view = channel_block_view(x, block_size)
        if zero_point:
            lo = view.amin(dim=(0, 1, 3), keepdim=True)
            hi = view.amax(dim=(0, 1, 3), keepdim=True)
            view = view - 0.5 * (lo + hi)
        return view.abs().amax(dim=(0, 1, 3), keepdim=True)
    if zero_point:
        lo, hi = aminmax_for_scale(x, granularity)
        x = x - 0.5 * (lo + hi)
    return absmax_for_scale(x, granularity)


def print_fp_nan_diagnostics(
    name: str,
    x: torch.Tensor,
    got: torch.Tensor,
    granularity: str,
    block_size: int,
    quant_max: float,
    scale_dtype: str,
    zero_point: bool = False,
) -> None:
    if torch.isfinite(got).all():
        return
    absmax = fp_absmax_for_quant(x, granularity, block_size, zero_point)
    raw_scale = (absmax / quant_max).clamp_min(1e-12)
    stored_scale = compute_scale(absmax, quant_max, scale_dtype)
    print(f"[DEBUG] non-finite output from {name}")
    print_tensor_summary(f"{name} absmax", absmax)
    print_tensor_summary(f"{name} raw_scale", raw_scale)
    print_tensor_summary(f"{name} stored_scale_{scale_dtype}", stored_scale)
    print_tensor_summary(f"{name} output", got)


def print_error_nan_diagnostics(
    name: str, ref: torch.Tensor, got: torch.Tensor
) -> None:
    err = got.float() - ref.float()
    if torch.isfinite(err).all():
        return
    print(f"[DEBUG] non-finite error from {name}")
    print_tensor_summary(f"{name} ref", ref)
    print_tensor_summary(f"{name} got", got)
    print_tensor_summary(f"{name} error", err)


def make_matrix(args: argparse.Namespace, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=args.device)
    gen.manual_seed(seed)
    shape = (args.batch_size, args.rows, args.cols)
    if args.distribution == "normal":
        x = torch.randn(shape, generator=gen, device=args.device)
        x = x * args.normal_sigma + args.normal_mean
    elif args.distribution == "uniform":
        x = torch.rand(shape, generator=gen, device=args.device) * 2.0 - 1.0
    elif args.distribution == "shifted_normal":
        x = torch.randn(shape, generator=gen, device=args.device)
        x = x + args.shift
    elif args.distribution == "lognormal_signed":
        normal = torch.randn(shape, generator=gen, device=args.device)
        mag = (normal * args.normal_sigma + args.normal_mean).exp()
        sign = torch.randint(0, 2, shape, generator=gen, device=args.device)
        x = mag * (sign.float() * 2.0 - 1.0)
    elif args.distribution == "column_scaled_normal":
        x = torch.randn(shape, generator=gen, device=args.device)
        x = x * args.normal_sigma + args.normal_mean
        col_log_scale = torch.randn(
            (1, 1, args.cols), generator=gen, device=args.device
        )
        col_scale = (col_log_scale * args.column_scale_sigma).exp()
        x = x * col_scale
    else:
        raise ValueError(f"unknown distribution: {args.distribution}")
    if args.outlier_frac > 0:
        mask = torch.rand(shape, generator=gen, device=args.device) < args.outlier_frac
        signs = torch.randint(0, 2, shape, generator=gen, device=args.device)
        outliers = args.outlier_scale * (signs.float() * 2.0 - 1.0)
        x = torch.where(mask, x + outliers, x)
    return x.to(torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare per-column INT/FP quantization MSE on a matrix."
    )
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Run seeds [seed, seed + num_seeds) and report aggregate stats.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--distribution",
        choices=(
            "normal",
            "uniform",
            "shifted_normal",
            "lognormal_signed",
            "column_scaled_normal",
        ),
        default="normal",
    )
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--outlier-frac", type=float, default=0.0)
    parser.add_argument("--outlier-scale", type=float, default=10.0)
    parser.add_argument("--normal_mean", type=float, default=0.0)
    parser.add_argument("--normal_sigma", type=float, default=1.0)
    parser.add_argument(
        "--column-scale-sigma",
        type=float,
        default=3.0,
        help="Lognormal column scale sigma for column_scaled_normal.",
    )
    parser.add_argument(
        "--scale-granularities",
        default="per_block,per_channel_block,per_channel,per_tensor",
        help=(
            "Comma-separated scale granularities: "
            f"{','.join(SCALE_GRANULARITIES)}. "
            "Aliases column/tensor are accepted for per_channel/per_tensor; "
            "channel_block is accepted for per_channel_block."
        ),
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help=(
            "Block size for per_block square tiles and per_channel_block "
            "channel chunks."
        ),
    )
    parser.add_argument(
        "--scale-dtypes",
        default="pow2,fp8,fp4,fp16",
        help="Comma-separated FP4 scale storage dtypes.",
    )
    parser.add_argument(
        "--fp8-dtypes",
        default="fp8",
        help="Comma-separated FP8 value dtypes: fp8/fp8_e4m3/fp8_e4m3fn, fp8_e5m2.",
    )
    parser.add_argument(
        "--fp8-scale-dtypes",
        default="pow2,fp16",
        help="Comma-separated FP8 and INT8 scale storage dtypes.",
    )
    parser.add_argument(
        "--int-quant-scheme",
        choices=("sym", "asym"),
        default="sym",
        help=(
            "INT8 quantization scheme. sym uses signed symmetric [-127,127]; "
            "asym uses affine min/max with quantized range [-128,127]."
        ),
    )
    parser.add_argument(
        "--int4-quantile",
        type=float,
        default=default_int4_quantile(),
        help=(
            "Quantile q for the KVarN-style int4_asym_quantile baseline, "
            "using [q, 1-q] instead of min/max."
        ),
    )
    parser.add_argument(
        "--fp-quantile",
        type=float,
        default=None,
        help=(
            "Deprecated shared quantile q for FP quantile baselines. "
            "Use --fp4-quantile and --fp8-quantile to tune them separately."
        ),
    )
    parser.add_argument(
        "--fp4-quantile",
        type=float,
        default=None,
        help=(
            "Quantile q for FP4 quantile baselines, using abs(x) quantile "
            "1-q instead of absmax. Defaults to --fp-quantile or 0.005."
        ),
    )
    parser.add_argument(
        "--fp8-quantile",
        type=float,
        default=None,
        help=(
            "Quantile q for FP8 quantile baselines, using abs(x) quantile "
            "1-q instead of absmax. Defaults to --fp-quantile or 0.0001."
        ),
    )
    parser.add_argument(
        "--fp4-quantile-sweep",
        default="",
        help=(
            "Comma-separated FP4 quantiles to sweep. If set, emits one "
            "fp4_quantile_q*_scale_* baseline per q."
        ),
    )
    parser.add_argument(
        "--fp8-quantile-sweep",
        default="",
        help=(
            "Comma-separated FP8 quantiles to sweep. If set, emits one "
            "fp8_quantile_q*_scale_* baseline per q."
        ),
    )
    parser.add_argument(
        "--compare-quantile-only",
        action="store_true",
        help="Only include quantile FP/INT variants in the pairwise report.",
    )
    parser.add_argument(
        "--output-json",
        default="myscript/output/evalscope/profile/matrix_quant_mse.json",
    )
    parser.add_argument(
        "--debug-fp-nan-scales",
        action="store_true",
        help="Print scale/output summaries for FP quant configs with non-finite output.",
    )
    parser.add_argument(
        "--histogram-dir",
        default="",
        help="If set, write average histogram PNGs to this directory.",
    )
    parser.add_argument(
        "--histogram-kind",
        choices=("value", "rel_error", "abs_error", "signed_error"),
        default="rel_error",
    )
    parser.add_argument("--histogram-bins", type=int, default=120)
    parser.add_argument(
        "--histogram-alpha",
        type=float,
        default=0.55,
        help="Transparency for histogram strokes; lower values make overlays easier to compare.",
    )
    parser.add_argument(
        "--histogram-y-scale",
        choices=("linear", "log"),
        default="linear",
        help="Y-axis scale for histogram plots.",
    )
    parser.add_argument(
        "--histogram-y-min",
        type=float,
        default=1e-12,
        help="Minimum plotted probability for log-y histograms.",
    )
    parser.add_argument(
        "--histogram-log-min",
        type=float,
        default=-8.0,
        help="Min log10 bin edge for rel_error/abs_error histograms.",
    )
    parser.add_argument(
        "--histogram-log-max",
        type=float,
        default=1.0,
        help="Max log10 bin edge for rel_error/abs_error histograms.",
    )
    parser.add_argument(
        "--histogram-signed-max",
        type=float,
        default=1.0,
        help="Symmetric max bin edge for signed_error histograms.",
    )
    parser.add_argument(
        "--histogram-value-max",
        type=float,
        default=0.0,
        help="Symmetric max bin edge for value histograms. Use <=0 for auto.",
    )
    parser.add_argument(
        "--histogram-value-quantile",
        type=float,
        default=0.999,
        help="Auto value histogram range uses this abs(x) quantile across seeds.",
    )
    return parser.parse_args()


def run_once(args: argparse.Namespace, seed: int) -> dict[str, dict[str, float]]:
    x = make_matrix(args, seed)
    results = {}
    scale_dtypes = [item.strip() for item in args.scale_dtypes.split(",") if item]
    fp8_dtypes = [item.strip() for item in args.fp8_dtypes.split(",") if item]
    fp8_scale_dtypes = [
        item.strip() for item in args.fp8_scale_dtypes.split(",") if item
    ]
    granularities = [
        item.strip() for item in args.scale_granularities.split(",") if item
    ]
    for granularity in granularities:
        granularity = normalize_granularity(granularity)
        prefix = granularity
        for int4_scheme in ("sym", "asym", "asym_quantile"):
            results[f"{prefix}_int4_{int4_scheme}"] = metrics(
                x,
                quant_int4(
                    x,
                    granularity,
                    args.block_size,
                    quant_scheme=int4_scheme,
                    quantile=args.int4_quantile,
                ),
            )
        results[f"{prefix}_int8_{args.int_quant_scheme}"] = metrics(
            x,
            quant_int8(
                x,
                granularity,
                quant_scheme=args.int_quant_scheme,
                block_size=args.block_size,
            ),
        )
        for scale_dtype in scale_dtypes:
            name = f"{prefix}_fp4_scale_{scale_dtype}"
            got = quant_fp4(x, granularity, scale_dtype, args.block_size)
            results[name] = metrics(x, got)
            if args.debug_fp_nan_scales:
                print_error_nan_diagnostics(name, x, got)
                print_fp_nan_diagnostics(
                    name,
                    x,
                    got,
                    granularity,
                    args.block_size,
                    FP4_E2M1_MAX,
                    scale_dtype,
                )
            for q in args.fp4_quantiles:
                name = fp_quantile_name(
                    prefix,
                    "fp4",
                    scale_dtype,
                    q,
                    args.fp4_quantile_use_q_label,
                )
                got = quant_fp4(
                    x,
                    granularity,
                    scale_dtype,
                    args.block_size,
                    quantile=q,
                )
                results[name] = metrics(x, got)
                if args.debug_fp_nan_scales:
                    print_error_nan_diagnostics(name, x, got)
                    print_fp_nan_diagnostics(
                        name,
                        x,
                        got,
                        granularity,
                        args.block_size,
                        FP4_E2M1_MAX,
                        scale_dtype,
                    )
            name = f"{prefix}_fp4_zp_scale_{scale_dtype}"
            got = quant_fp4_zp(x, granularity, scale_dtype, args.block_size)
            results[name] = metrics(x, got)
            if args.debug_fp_nan_scales:
                print_error_nan_diagnostics(name, x, got)
                print_fp_nan_diagnostics(
                    name,
                    x,
                    got,
                    granularity,
                    args.block_size,
                    FP4_E2M1_MAX,
                    scale_dtype,
                    zero_point=True,
                )
        for scale_dtype in fp8_scale_dtypes:
            results[
                f"{prefix}_int8_{args.int_quant_scheme}_scale_{scale_dtype}"
            ] = metrics(
                x,
                quant_int8(
                    x,
                    granularity,
                    scale_dtype,
                    quant_scheme=args.int_quant_scheme,
                    block_size=args.block_size,
                ),
            )
        for fp8_value_dtype in fp8_dtypes:
            for scale_dtype in fp8_scale_dtypes:
                name = f"{prefix}_{fp8_value_dtype}_scale_{scale_dtype}"
                got = quant_fp8(
                    x,
                    fp8_value_dtype,
                    granularity,
                    scale_dtype,
                    args.block_size,
                )
                results[name] = metrics(x, got)
                if args.debug_fp_nan_scales:
                    print_error_nan_diagnostics(name, x, got)
                    print_fp_nan_diagnostics(
                        name,
                        x,
                        got,
                        granularity,
                        args.block_size,
                        FP8_MAX[fp8_value_dtype],
                        scale_dtype,
                    )
                for q in args.fp8_quantiles:
                    name = fp_quantile_name(
                        prefix,
                        fp8_value_dtype,
                        scale_dtype,
                        q,
                        args.fp8_quantile_use_q_label,
                    )
                    got = quant_fp8(
                        x,
                        fp8_value_dtype,
                        granularity,
                        scale_dtype,
                        args.block_size,
                        quantile=q,
                    )
                    results[name] = metrics(x, got)
                    if args.debug_fp_nan_scales:
                        print_error_nan_diagnostics(name, x, got)
                        print_fp_nan_diagnostics(
                            name,
                            x,
                            got,
                            granularity,
                            args.block_size,
                            FP8_MAX[fp8_value_dtype],
                            scale_dtype,
                        )
    return results


def aggregate_runs(
    runs: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    names = sorted(runs[0])
    out: dict[str, dict[str, float]] = {}
    for name in names:
        out[name] = {}
        metric_names = sorted(runs[0][name])
        for metric_name in metric_names:
            values = torch.tensor(
                [run[name][metric_name] for run in runs],
                dtype=torch.float64,
            )
            out[name][metric_name] = float(values.mean().item())
            out[name][f"{metric_name}_std"] = (
                float(values.std(unbiased=False).item()) if len(runs) > 1 else 0.0
            )
    return out


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_csv(value: str) -> list[float]:
    return [float(item) for item in parse_csv(value)]


def validate_quantile(name: str, q: float) -> None:
    if not 0.0 < q < 0.5:
        raise ValueError(f"{name} must be in (0, 0.5), got {q}")


def quantile_label(q: float) -> str:
    return f"q{q:g}".replace("-", "m").replace(".", "p")


def fp_quantile_name(
    prefix: str,
    fp_kind: str,
    scale_dtype: str,
    q: float,
    use_q_label: bool,
) -> str:
    if use_q_label:
        return (
            f"{prefix}_{fp_kind}_quantile_{quantile_label(q)}_scale_"
            f"{scale_dtype}"
        )
    return f"{prefix}_{fp_kind}_quantile_scale_{scale_dtype}"


def parse_quantile_label(label: str) -> float:
    value = label.removeprefix("q").replace("p", ".").replace("m", "-")
    return float(value)


def fp_quantile_from_name(name: str) -> float | None:
    parts = name.split("_")
    if "quantile" not in parts:
        return None
    index = parts.index("quantile")
    if index + 1 >= len(parts):
        return None
    label = parts[index + 1]
    if not label.startswith("q"):
        return None
    try:
        return parse_quantile_label(label)
    except ValueError:
        return None


def winner_from_relative_mse(fp_rel: float, int_rel: float) -> tuple[float, str]:
    fp_valid = math.isfinite(fp_rel)
    int_valid = math.isfinite(int_rel)
    if fp_valid and not int_valid:
        return 0.0, "fp_win"
    if int_valid and not fp_valid:
        return math.inf, "int_win"
    if not fp_valid and not int_valid:
        return math.nan, "invalid"

    ratio = fp_rel / int_rel if int_rel > 0 else math.inf
    if fp_rel < int_rel:
        return ratio, "fp_win"
    if int_rel < fp_rel:
        return ratio, "int_win"
    return ratio, "tie"


def build_pairwise_report(
    avg: dict[str, dict[str, float]], args: argparse.Namespace
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    granularities = [
        normalize_granularity(item) for item in parse_csv(args.scale_granularities)
    ]

    fp4_kinds = ["fp4", "fp4_zp"]
    fp4_quantile_kinds = [
        ("fp4", q, args.fp4_quantile_use_q_label) for q in args.fp4_quantiles
    ]
    int4_schemes = ["sym", "asym", "asym_quantile"]
    if args.compare_quantile_only:
        fp4_kinds = []
        int4_schemes = ["asym_quantile"]

    for granularity in granularities:
        for fp_kind in fp4_kinds:
            for int4_scheme in int4_schemes:
                int4_name = f"{granularity}_int4_{int4_scheme}"
                if int4_name not in avg:
                    continue
                int4_rel = avg[int4_name]["relative_mse"]
                for scale_dtype in parse_csv(args.scale_dtypes):
                    fp_name = f"{granularity}_{fp_kind}_scale_{scale_dtype}"
                    if fp_name not in avg:
                        continue
                    fp_rel = avg[fp_name]["relative_mse"]
                    ratio, winner = winner_from_relative_mse(fp_rel, int4_rel)
                    rows.append(
                        {
                            "granularity": granularity,
                            "bits": 4,
                            "scale_dtype": scale_dtype,
                            "fp_name": fp_name,
                            "int_name": int4_name,
                            "fp_relative_mse": fp_rel,
                            "int_relative_mse": int4_rel,
                            "ratio": ratio,
                            "winner": winner,
                        }
                    )

        for fp_kind, q, use_q_label in fp4_quantile_kinds:
            for int4_scheme in int4_schemes:
                int4_name = f"{granularity}_int4_{int4_scheme}"
                if int4_name not in avg:
                    continue
                int4_rel = avg[int4_name]["relative_mse"]
                for scale_dtype in parse_csv(args.scale_dtypes):
                    fp_name = fp_quantile_name(
                        granularity, fp_kind, scale_dtype, q, use_q_label
                    )
                    if fp_name not in avg:
                        continue
                    fp_rel = avg[fp_name]["relative_mse"]
                    ratio, winner = winner_from_relative_mse(fp_rel, int4_rel)
                    rows.append(
                        {
                            "granularity": granularity,
                            "bits": 4,
                            "scale_dtype": scale_dtype,
                            "fp_name": fp_name,
                            "int_name": int4_name,
                            "fp_relative_mse": fp_rel,
                            "int_relative_mse": int4_rel,
                            "ratio": ratio,
                            "winner": winner,
                        }
                    )

        for scale_dtype in parse_csv(args.fp8_scale_dtypes):
            int8_name = (
                f"{granularity}_int8_{args.int_quant_scheme}_scale_{scale_dtype}"
            )
            if int8_name not in avg:
                continue
            int8_rel = avg[int8_name]["relative_mse"]
            for fp8_value_dtype in parse_csv(args.fp8_dtypes):
                if not args.compare_quantile_only:
                    fp_name = f"{granularity}_{fp8_value_dtype}_scale_{scale_dtype}"
                    if fp_name in avg:
                        fp_rel = avg[fp_name]["relative_mse"]
                        ratio, winner = winner_from_relative_mse(fp_rel, int8_rel)
                        rows.append(
                            {
                                "granularity": granularity,
                                "bits": 8,
                                "scale_dtype": scale_dtype,
                                "fp_name": fp_name,
                                "int_name": int8_name,
                                "fp_relative_mse": fp_rel,
                                "int_relative_mse": int8_rel,
                                "ratio": ratio,
                                "winner": winner,
                            }
                        )
                for q in args.fp8_quantiles:
                    fp_name = fp_quantile_name(
                        granularity,
                        fp8_value_dtype,
                        scale_dtype,
                        q,
                        args.fp8_quantile_use_q_label,
                    )
                    if fp_name not in avg:
                        continue
                    fp_rel = avg[fp_name]["relative_mse"]
                    ratio, winner = winner_from_relative_mse(fp_rel, int8_rel)
                    rows.append(
                        {
                            "granularity": granularity,
                            "bits": 8,
                            "scale_dtype": scale_dtype,
                            "fp_name": fp_name,
                            "int_name": int8_name,
                            "fp_relative_mse": fp_rel,
                            "int_relative_mse": int8_rel,
                            "ratio": ratio,
                            "winner": winner,
                        }
                    )
    return rows


def print_pairwise_report(rows: list[dict[str, float | str]]) -> None:
    print("\n[REPORT] FP / INT relative MSE ratio. ratio < 1 means fp_win.")
    for row in rows:
        print(
            f"{row['granularity']:12s} bits={row['bits']} "
            f"scale={row['scale_dtype']:5s} "
            f"fp={row['fp_name']:34s} "
            f"int={row['int_name']:34s} "
            f"fp_rel={row['fp_relative_mse']:.6e} "
            f"int_rel={row['int_relative_mse']:.6e} "
            f"ratio={row['ratio']:.3f} "
            f"winner={row['winner']}"
        )


def histogram_values(
    ref: torch.Tensor, got: torch.Tensor, hist_kind: str
) -> torch.Tensor:
    err = got.float() - ref.float()
    if hist_kind == "value":
        return got.float().flatten()
    if hist_kind == "abs_error":
        return err.abs().flatten()
    if hist_kind == "rel_error":
        denom = ref.float().abs().clamp_min(1e-12)
        return (err.abs() / denom).flatten()
    if hist_kind == "signed_error":
        return err.flatten()
    raise ValueError(f"unknown histogram kind: {hist_kind}")


def print_histogram_nan_diagnostics(
    name: str,
    ref: torch.Tensor,
    got: torch.Tensor,
    values: torch.Tensor,
    hist_kind: str,
) -> None:
    err = got.float() - ref.float()
    print(f"[DEBUG] non-finite {hist_kind} histogram values from {name}")
    print_tensor_summary(f"{name} ref", ref)
    print_tensor_summary(f"{name} got", got)
    print_tensor_summary(f"{name} error", err)
    print_tensor_summary(f"{name} {hist_kind}", values)


def histogram_bins(args: argparse.Namespace) -> torch.Tensor:
    if args.histogram_kind == "value":
        if args.histogram_value_max <= 0:
            raise ValueError("value histogram bins must be built from a positive max")
        return torch.linspace(
            -args.histogram_value_max,
            args.histogram_value_max,
            args.histogram_bins + 1,
            device=args.device,
        )
    if args.histogram_kind in {"abs_error", "rel_error"}:
        return torch.logspace(
            args.histogram_log_min,
            args.histogram_log_max,
            args.histogram_bins + 1,
            device=args.device,
        )
    return torch.linspace(
        -args.histogram_signed_max,
        args.histogram_signed_max,
        args.histogram_bins + 1,
        device=args.device,
    )


def infer_value_histogram_max(args: argparse.Namespace, seeds: list[int]) -> float:
    quantile_values = []
    for seed in seeds:
        x = make_matrix(args, seed)
        quantile_values.append(
            torch.quantile(x.float().abs().flatten(), args.histogram_value_quantile)
        )
    value_max = float(torch.stack(quantile_values).amax().item())
    return max(value_max, 1e-12)


def iter_histogram_quantizers(args: argparse.Namespace):
    for granularity in parse_csv(args.scale_granularities):
        granularity = normalize_granularity(granularity)
        for int4_scheme in ("sym", "asym", "asym_quantile"):
            yield (
                f"{granularity}_int4_{int4_scheme}",
                lambda x, g=granularity, s=int4_scheme: quant_int4(
                    x,
                    g,
                    args.block_size,
                    quant_scheme=s,
                    quantile=args.int4_quantile,
                ),
            )
        for scale_dtype in parse_csv(args.scale_dtypes):
            yield (
                f"{granularity}_fp4_scale_{scale_dtype}",
                lambda x, g=granularity, sd=scale_dtype: quant_fp4(
                    x, g, sd, args.block_size
                ),
            )
            for q in args.fp4_quantiles:
                yield (
                    fp_quantile_name(
                        granularity,
                        "fp4",
                        scale_dtype,
                        q,
                        args.fp4_quantile_use_q_label,
                    ),
                    lambda x, g=granularity, sd=scale_dtype, q=q: quant_fp4(
                        x,
                        g,
                        sd,
                        args.block_size,
                        quantile=q,
                    ),
                )
        for scale_dtype in parse_csv(args.fp8_scale_dtypes):
            yield (
                f"{granularity}_int8_{args.int_quant_scheme}_scale_{scale_dtype}",
                lambda x, g=granularity, sd=scale_dtype: quant_int8(
                    x,
                    g,
                    sd,
                    quant_scheme=args.int_quant_scheme,
                    block_size=args.block_size,
                ),
            )
            for fp8_value_dtype in parse_csv(args.fp8_dtypes):
                yield (
                    f"{granularity}_{fp8_value_dtype}_scale_{scale_dtype}",
                    lambda x, g=granularity, sd=scale_dtype, fd=fp8_value_dtype: (
                        quant_fp8(
                            x,
                            fd,
                            g,
                            sd,
                            args.block_size,
                        )
                    ),
                )
                for q in args.fp8_quantiles:
                    def quantizer(
                        x,
                        g=granularity,
                        sd=scale_dtype,
                        fd=fp8_value_dtype,
                        q=q,
                    ):
                        return quant_fp8(
                            x,
                            fd,
                            g,
                            sd,
                            args.block_size,
                            quantile=q,
                        )

                    yield (
                        fp_quantile_name(
                            granularity,
                            fp8_value_dtype,
                            scale_dtype,
                            q,
                            args.fp8_quantile_use_q_label,
                        ),
                        quantizer,
                    )


def average_histograms(
    args: argparse.Namespace, seeds: list[int]
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None]:
    quantizers = list(iter_histogram_quantizers(args))
    if args.histogram_kind == "value" and args.histogram_value_max <= 0:
        args.histogram_value_max = infer_value_histogram_max(args, seeds)
    bins = histogram_bins(args)
    counts = {
        name: torch.zeros(args.histogram_bins, dtype=torch.float64)
        for name, _ in quantizers
    }
    ref_counts = (
        torch.zeros(args.histogram_bins, dtype=torch.float64)
        if args.histogram_kind == "value"
        else None
    )
    for seed in seeds:
        x = make_matrix(args, seed)
        if ref_counts is not None:
            ref_values = x.float().flatten()
            ref_values = ref_values.clamp(float(bins[0]), float(bins[-1]))
            ref_bucket = torch.bucketize(ref_values, bins) - 1
            ref_bucket = ref_bucket.clamp(0, args.histogram_bins - 1)
            ref_hist = torch.bincount(ref_bucket, minlength=args.histogram_bins)
            ref_counts += ref_hist.cpu().double()
        for name, quantizer in quantizers:
            got = quantizer(x)
            values = histogram_values(x, got, args.histogram_kind)
            finite = torch.isfinite(values)
            if not finite.all():
                bad = values.numel() - int(finite.sum().item())
                print(
                    f"[WARN] skipped {bad} non-finite "
                    f"{args.histogram_kind} histogram values from {name}"
                )
                if args.debug_fp_nan_scales:
                    print_histogram_nan_diagnostics(
                        name, x, got, values, args.histogram_kind
                    )
                values = values[finite]
                if values.numel() == 0:
                    continue
            values = values.float().clamp(float(bins[0]), float(bins[-1]))
            bucket = torch.bucketize(values, bins) - 1
            bucket = bucket.clamp(0, args.histogram_bins - 1)
            hist = torch.bincount(bucket, minlength=args.histogram_bins)
            counts[name] += hist.cpu().double()
    for name in counts:
        total = counts[name].sum().clamp_min(1.0)
        counts[name] = counts[name] / total
    if ref_counts is not None:
        ref_counts = ref_counts / ref_counts.sum().clamp_min(1.0)
    return counts, bins.cpu(), ref_counts


def plot_histogram_panel(
    args: argparse.Namespace,
    bins: torch.Tensor,
    counts: dict[str, torch.Tensor],
    ref_counts: torch.Tensor | None,
    names: list[str],
    title: str,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(names), figsize=(5.2 * len(names), 4.8), sharey=True
    )
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        if name not in counts:
            continue
        if ref_counts is not None:
            ax.stairs(
                ref_counts.clamp_min(args.histogram_y_min).numpy(),
                bins.numpy(),
                linewidth=1.6,
                alpha=args.histogram_alpha,
                label="original x",
            )
            ax.stairs(
                counts[name].clamp_min(args.histogram_y_min).numpy(),
                bins.numpy(),
                linewidth=1.6,
                alpha=args.histogram_alpha,
                label="dequantized",
            )
            ax.legend(fontsize=8)
        else:
            ax.stairs(
                counts[name].clamp_min(args.histogram_y_min).numpy(),
                bins.numpy(),
                linewidth=1.8,
                alpha=args.histogram_alpha,
            )
        if args.histogram_kind in {"abs_error", "rel_error"}:
            ax.set_xscale("log")
        if args.histogram_y_scale == "log":
            ax.set_yscale("log")
        ax.set_xlabel(args.histogram_kind)
        ax.set_title(name)
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("average probability")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[INFO] wrote histogram {path}")


def plot_histogram_comparison_panel(
    args: argparse.Namespace,
    bins: torch.Tensor,
    counts: dict[str, torch.Tensor],
    pairs: list[tuple[str, str]],
    title: str,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(pairs), figsize=(5.2 * len(pairs), 4.8), sharey=True
    )
    if len(pairs) == 1:
        axes = [axes]
    for ax, (int_name, fp_name) in zip(axes, pairs):
        if int_name in counts:
            ax.stairs(
                counts[int_name].clamp_min(args.histogram_y_min).numpy(),
                bins.numpy(),
                linewidth=1.8,
                alpha=args.histogram_alpha,
                label=int_name,
                color="tab:orange",
            )
        if fp_name in counts:
            ax.stairs(
                counts[fp_name].clamp_min(args.histogram_y_min).numpy(),
                bins.numpy(),
                linewidth=1.8,
                alpha=args.histogram_alpha,
                label=fp_name,
                color="tab:blue",
            )
        if args.histogram_y_scale == "log":
            ax.set_yscale("log")
        ax.set_xlabel(args.histogram_kind)
        ax.set_title(f"{fp_name}\nvs {int_name}", fontsize=9)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("average probability")
    fig.suptitle(title, y=1.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[INFO] wrote histogram {path}")


def plot_histogram_overlay_panel(
    args: argparse.Namespace,
    bins: torch.Tensor,
    counts: dict[str, torch.Tensor],
    names: list[str],
    title: str,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    baseline_colors = (
        ("int4_sym", "black"),
        ("int4_asym_quantile", "tab:purple"),
        ("int4_asym", "tab:green"),
        ("fp4_scale", "dimgray"),
    )
    fallback_colors = ("tab:brown", "tab:cyan", "tab:olive", "tab:pink")
    quantile_values = {
        name: q for name in names if (q := fp_quantile_from_name(name)) is not None
    }
    sorted_quantiles = sorted(set(quantile_values.values()))
    quantile_index = {q: index for index, q in enumerate(sorted_quantiles)}
    quantile_cmap = plt.get_cmap("turbo")

    def color_for_name(name: str, fallback_index: int) -> str | tuple[float, ...]:
        quantile = quantile_values.get(name)
        if quantile is not None:
            if len(sorted_quantiles) == 1:
                return quantile_cmap(0.5)
            ratio = quantile_index[quantile] / (len(sorted_quantiles) - 1)
            return quantile_cmap(0.12 + 0.76 * ratio)
        for pattern, color in baseline_colors:
            if pattern in name:
                return color
        return fallback_colors[fallback_index % len(fallback_colors)]

    fig, ax = plt.subplots(1, 1, figsize=(5.8, 4.8))
    for index, name in enumerate(names):
        if name not in counts:
            continue
        color = color_for_name(name, index)
        ax.stairs(
            counts[name].clamp_min(args.histogram_y_min).numpy(),
            bins.numpy(),
            linewidth=1.8,
            alpha=args.histogram_alpha,
            label=name,
            color=color,
        )
    if args.histogram_y_scale == "log":
        ax.set_yscale("log")
    ax.set_xlabel(args.histogram_kind)
    ax.set_ylabel("average probability")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[INFO] wrote histogram {path}")


def plot_average_histograms(args: argparse.Namespace, seeds: list[int]) -> None:
    try:
        import matplotlib.pyplot  # noqa: F401
    except ImportError:
        print("[WARN] matplotlib is not installed; skipping histogram plots.")
        return

    output_dir = Path(args.histogram_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts, bins, ref_counts = average_histograms(args, seeds)

    for granularity in parse_csv(args.scale_granularities):
        granularity = normalize_granularity(granularity)
        for scale_dtype in parse_csv(args.scale_dtypes):
            names = [
                f"{granularity}_int4_sym",
                f"{granularity}_int4_asym",
                f"{granularity}_int4_asym_quantile",
                f"{granularity}_fp4_scale_{scale_dtype}",
            ]
            names.extend(
                fp_quantile_name(
                    granularity,
                    "fp4",
                    scale_dtype,
                    q,
                    args.fp4_quantile_use_q_label,
                )
                for q in args.fp4_quantiles
            )
            path = (
                output_dir
                / (
                    f"hist_{granularity}_4bit_scale_{scale_dtype}_"
                    f"{args.histogram_kind}.png"
                )
            )
            if args.histogram_kind == "signed_error":
                plot_histogram_overlay_panel(
                    args,
                    bins,
                    counts,
                    names,
                    f"{granularity} 4-bit scale={scale_dtype}",
                    path,
                )
            else:
                plot_histogram_panel(
                    args,
                    bins,
                    counts,
                    ref_counts,
                    names,
                    f"{granularity} 4-bit scale={scale_dtype}",
                    path,
                )

        for scale_dtype in parse_csv(args.fp8_scale_dtypes):
            names = [
                f"{granularity}_int8_{args.int_quant_scheme}_scale_{scale_dtype}"
            ]
            names.extend(
                f"{granularity}_{fp8_name}_scale_{scale_dtype}"
                for fp8_name in parse_csv(args.fp8_dtypes)
            )
            names.extend(
                fp_quantile_name(
                    granularity,
                    fp8_name,
                    scale_dtype,
                    q,
                    args.fp8_quantile_use_q_label,
                )
                for fp8_name in parse_csv(args.fp8_dtypes)
                for q in args.fp8_quantiles
            )
            path = (
                output_dir
                / (
                    f"hist_{granularity}_8bit_scale_{scale_dtype}_"
                    f"{args.histogram_kind}.png"
                )
            )
            if args.histogram_kind == "signed_error":
                pairs = [
                    (
                        (
                            f"{granularity}_int8_{args.int_quant_scheme}_scale_"
                            f"{scale_dtype}"
                        ),
                        f"{granularity}_{fp8_name}_scale_{scale_dtype}",
                    )
                    for fp8_name in parse_csv(args.fp8_dtypes)
                ]
                pairs.extend(
                    (
                        (
                            f"{granularity}_int8_{args.int_quant_scheme}_scale_"
                            f"{scale_dtype}"
                        ),
                        fp_quantile_name(
                            granularity,
                            fp8_name,
                            scale_dtype,
                            q,
                            args.fp8_quantile_use_q_label,
                        ),
                    )
                    for fp8_name in parse_csv(args.fp8_dtypes)
                    for q in args.fp8_quantiles
                )
                plot_histogram_comparison_panel(
                    args,
                    bins,
                    counts,
                    pairs,
                    f"{granularity} 8-bit scale={scale_dtype}",
                    path,
                )
            else:
                plot_histogram_panel(
                    args,
                    bins,
                    counts,
                    ref_counts,
                    names,
                    f"{granularity} 8-bit scale={scale_dtype}",
                    path,
                )


def main() -> None:
    args = parse_args()
    if args.fp4_quantile is None:
        args.fp4_quantile = (
            args.fp_quantile if args.fp_quantile is not None else 0.005
        )
    if args.fp8_quantile is None:
        args.fp8_quantile = (
            args.fp_quantile if args.fp_quantile is not None else 0.0001
        )
    validate_quantile("--int4-quantile", args.int4_quantile)
    if args.fp_quantile is not None:
        validate_quantile("--fp-quantile", args.fp_quantile)
    validate_quantile("--fp4-quantile", args.fp4_quantile)
    validate_quantile("--fp8-quantile", args.fp8_quantile)
    args.fp4_quantiles = (
        parse_float_csv(args.fp4_quantile_sweep)
        if args.fp4_quantile_sweep
        else [args.fp4_quantile]
    )
    args.fp8_quantiles = (
        parse_float_csv(args.fp8_quantile_sweep)
        if args.fp8_quantile_sweep
        else [args.fp8_quantile]
    )
    for q in args.fp4_quantiles:
        validate_quantile("--fp4-quantile-sweep", q)
    for q in args.fp8_quantiles:
        validate_quantile("--fp8-quantile-sweep", q)
    args.fp4_quantile_use_q_label = len(args.fp4_quantiles) > 1
    args.fp8_quantile_use_q_label = len(args.fp8_quantiles) > 1
    seeds = list(range(args.seed, args.seed + args.num_seeds))
    runs = [run_once(args, seed) for seed in seeds]
    aggregate = aggregate_runs(runs)
    pairwise_report = build_pairwise_report(aggregate, args)
    results = {
        "config": vars(args),
        "seeds": seeds,
        "per_seed": {str(seed): run for seed, run in zip(seeds, runs)},
        "average": aggregate,
        "pairwise_report": pairwise_report,
    }
    print(
        f"[INFO] shape=({args.batch_size}, {args.rows}, {args.cols}) "
        f"seeds={seeds}"
    )
    for name, values in aggregate.items():
        print(
            f"{name:28s} mse={values['mse']:.6e} "
            f"mse_std={values['mse_std']:.2e} "
            f"rel_mse={values['relative_mse']:.6e} "
            f"rel_std={values['relative_mse_std']:.2e} "
            f"sqnr={values['sqnr_db']:.2f}dB "
            f"mae={values['mae']:.6e}"
        )
    print_pairwise_report(pairwise_report)
    if args.histogram_dir:
        plot_average_histograms(args, seeds)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[INFO] wrote {output}")


if __name__ == "__main__":
    main()
