"""Ascend key-mask broadcast addition and packed Q scaling Triton kernels."""
import numbers
import torch
import triton
import triton.language as tl
from flag_gems.utils import libentry


@libentry()
@triton.jit
def _keymask(X, M, Y, ROWS, K, B0, M0, BR: tl.constexpr, BC: tl.constexpr):
    col = tl.program_id(0) * BC + tl.arange(0, BC)
    row = tl.program_id(1) * BR + tl.arange(0, BR)
    b = tl.program_id(2).to(tl.int64)
    off = row[:, None] * K + col[None, :]
    valid = (row[:, None] < ROWS) & (col[None, :] < K)
    x = tl.load(X + b * B0 + off, valid, 0).to(tl.float32)
    m = tl.load(M + b * M0 + col, col < K, 0).to(tl.float32)
    tl.store(Y + b * B0 + off, x + m[None, :], valid)


@libentry()
@triton.jit
def _scale_q(X, Y, S, W, B0, T0, VAL: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr):
    c = tl.program_id(0) * BC + tl.arange(0, BC)
    s = tl.program_id(1) * BR + tl.arange(0, BR)
    b = tl.program_id(2).to(tl.int64)
    ok = (s[:, None] < S) & (c[None, :] < W)
    v = tl.load(X + b * B0 + s[:, None] * T0 + c[None, :], ok, 0).to(tl.float32)
    tl.store(Y + b * (S * W) + s[:, None] * W + c[None, :], v * VAL, ok)


def keymask_add(a, b, alpha=1):
    if (
        alpha != 1
        or torch.is_grad_enabled()
        or not isinstance(a, torch.Tensor)
        or not isinstance(b, torch.Tensor)
        or a.device.type != "npu"
        or a.device != b.device
        or a.dtype != torch.float16
        or b.dtype != a.dtype
        or a.ndim != 4
        or b.ndim != 4
    ):
        return None
    if a.numel() < b.numel():
        a, b = b, a
    n, h, q, k = a.shape
    if not (
        k
        and h
        and q
        and n
        and a.is_contiguous()
        and b.shape[0] in (1, n)
        and b.shape[1:] == (1, 1, k)
        and b.stride(-1) == 1
        and h * q * k < 2**31 - 2048
    ):
        return None
    y = torch.empty_like(a)
    _keymask[(triton.cdiv(k, 256), triton.cdiv(q * h, 8), n)](
        a, b, y, q * h, k, a.stride(0), b.stride(0) if b.size(0) > 1 else 0, 8, 256
    )
    return y


def packed_scale(a, b):
    if isinstance(a, numbers.Real) and isinstance(b, torch.Tensor):
        a, b = b, a
    if (
        not isinstance(a, torch.Tensor)
        or not isinstance(b, numbers.Real)
        or b not in (0.0625, 0.125, 0.25)
        or torch.is_grad_enabled()
        or a.device.type != "npu"
        or a.dtype != torch.float16
        or a.ndim != 4
    ):
        return None
    n, s, h, d = a.shape
    w = h * d
    if not (
        n
        and s
        and h
        and d
        and a.stride(-1) == 1
        and a.stride(-2) == d
        and s * max(a.stride(1), w) < 2**31 - 2048
    ):
        return None
    y = torch.empty(a.shape, device=a.device, dtype=a.dtype)
    _scale_q[(triton.cdiv(w, 256), triton.cdiv(s, 8), n)](
        a, y, s, w, a.stride(0), a.stride(1), float(b), 8, 256
    )
    return y
