"""Shape-agnostic 64-bit persistent Ascend Triton middle-axis and trailing-scalar broadcast mul.
Eliminates the _launch_generic meta-tensor fallback on:
  1) Vision-RoPE middle-axis broadcast shapes:
     - 3D: (S, H, D) * (S, 1, D)
     - 4D: (B, S, H, D) * (B|1, S, 1, D)
  2) Qwen3.5 torch_chunk_gated_delta_rule trailing-scalar broadcast shapes:
     - Contiguous N-D: (..., K) or (..., K1, K2) with trailing 1s on C (e.g. i=1..63, (B,H,Dk,Dv)*(B,H,1,1))
     - Strided 4D: (B, H, S, D) * (B, H, S, 1) (e.g. q_i = query[:, :, i])
"""
from __future__ import annotations

import functools
import torch
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn


@triton.jit(do_not_specialize=[
    "M_total", "S_len", "H", "D", "XB", "XS", "XH", "CB", "CS", "n_progs"
])
def _mul_mid_broadcast_kernel(
    X, C, Y,
    M_total, S_len, H, D,
    XB, XS, XH,
    CB, CS,
    n_progs,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    for m_idx in range(pid, M_total, n_progs):
        m_i = tl.cast(m_idx, tl.int64)
        b_i = m_i // S_len
        s_i = m_i % S_len
        x_base = b_i * XB + s_i * XS
        c_base = b_i * CB + s_i * CS
        y_base = m_i * (tl.cast(H, tl.int64) * tl.cast(D, tl.int64))

        for d0 in range(0, D, BLOCK_D):
            d_off = d0 + tl.arange(0, BLOCK_D)
            d_mask = d_off < D
            c_val = tl.load(
                C + c_base + tl.cast(d_off, tl.int64),
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)[None, :]

            for h0 in range(0, H, BLOCK_H):
                h_off = h0 + tl.arange(0, BLOCK_H)
                h_mask = h_off < H
                mask = h_mask[:, None] & d_mask[None, :]
                x_ptr = (
                    X
                    + x_base
                    + tl.cast(h_off[:, None], tl.int64) * XH
                    + tl.cast(d_off[None, :], tl.int64)
                )
                y_ptr = (
                    Y
                    + y_base
                    + tl.cast(h_off[:, None], tl.int64) * D
                    + tl.cast(d_off[None, :], tl.int64)
                )
                x_val = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
                tl.store(y_ptr, x_val * c_val, mask=mask)


@triton.jit(do_not_specialize=["M", "K", "n_progs"])
def _mul_row_scalar_smallk_kernel(
    A, B, Y,
    M, K, n_progs,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    K_i64 = tl.cast(K, tl.int64)
    step = n_progs * BLOCK_M
    for m0 in range(pid * BLOCK_M, M, step):
        m_off = m0 + tl.arange(0, BLOCK_M)
        m_mask = m_off < M
        b_val = tl.load(
            B + tl.cast(m_off, tl.int64),
            mask=m_mask,
            other=0.0,
        ).to(tl.float32)[:, None]
        row_base = tl.cast(m_off[:, None], tl.int64) * K_i64
        for k0 in range(0, K, BLOCK_K):
            k_off = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_off < K
            mask = m_mask[:, None] & k_mask[None, :]
            offs = row_base + tl.cast(k_off[None, :], tl.int64)
            a_val = tl.load(A + offs, mask=mask, other=0.0).to(tl.float32)
            tl.store(Y + offs, a_val * b_val, mask=mask)


@triton.jit(do_not_specialize=["M", "K", "n_progs"])
def _mul_row_scalar_largek_kernel(
    A, B, Y,
    M, K, n_progs,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    K_i64 = tl.cast(K, tl.int64)
    for m_idx in range(pid, M, n_progs):
        m_i = tl.cast(m_idx, tl.int64)
        b_val = tl.load(B + m_i).to(tl.float32)
        row_base = m_i * K_i64
        for k0 in range(0, K, BLOCK_K):
            k_off = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_off < K
            offs = row_base + tl.cast(k_off, tl.int64)
            a_val = tl.load(A + offs, mask=k_mask, other=0.0).to(tl.float32)
            tl.store(Y + offs, a_val * b_val, mask=k_mask)


@triton.jit(do_not_specialize=[
    "B_dim", "H_dim", "S_dim", "D_dim", "XB", "XH", "XS", "CB", "CH", "CS", "n_progs"
])
def _mul_4d_last1_strided_kernel(
    X, C, Y,
    B_dim, H_dim, S_dim, D_dim,
    XB, XH, XS,
    CB, CH, CS,
    n_progs,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    BH = B_dim * H_dim
    S_i64 = tl.cast(S_dim, tl.int64)
    D_i64 = tl.cast(D_dim, tl.int64)
    SD_i64 = S_i64 * D_i64
    for bh in range(pid, BH, n_progs):
        b_i = tl.cast(bh // H_dim, tl.int64)
        h_i = tl.cast(bh % H_dim, tl.int64)
        x_bh = b_i * XB + h_i * XH
        c_bh = b_i * CB + h_i * CH
        y_bh = tl.cast(bh, tl.int64) * SD_i64

        for s0 in range(0, S_dim, BLOCK_S):
            s_off = s0 + tl.arange(0, BLOCK_S)
            s_mask = s_off < S_dim
            s_i64 = tl.cast(s_off, tl.int64)
            c_val = tl.load(
                C + c_bh + s_i64 * CS,
                mask=s_mask,
                other=0.0,
            ).to(tl.float32)[:, None]
            x_row_base = x_bh + s_i64[:, None] * XS
            y_row_base = y_bh + s_i64[:, None] * D_i64

            for d0 in range(0, D_dim, BLOCK_D):
                d_off = d0 + tl.arange(0, BLOCK_D)
                d_mask = d_off < D_dim
                mask = s_mask[:, None] & d_mask[None, :]
                d_i64 = tl.cast(d_off[None, :], tl.int64)
                x_val = tl.load(X + x_row_base + d_i64, mask=mask, other=0.0).to(tl.float32)
                tl.store(Y + y_row_base + d_i64, x_val * c_val, mask=mask)


@triton.jit(do_not_specialize=[
    "B_dim", "H_dim", "S_dim", "D_dim", "AB", "AH", "AS", "BB", "BH", "BS", "n_progs"
])
def _mul_4d_same_strided_kernel(
    A, B, Y,
    B_dim, H_dim, S_dim, D_dim,
    AB, AH, AS,
    BB, BH, BS,
    n_progs,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    bh_total = B_dim * H_dim
    S_i64 = tl.cast(S_dim, tl.int64)
    D_i64 = tl.cast(D_dim, tl.int64)
    SD_i64 = S_i64 * D_i64
    AB_i64 = tl.cast(AB, tl.int64)
    AH_i64 = tl.cast(AH, tl.int64)
    AS_i64 = tl.cast(AS, tl.int64)
    BB_i64 = tl.cast(BB, tl.int64)
    BH_i64 = tl.cast(BH, tl.int64)
    BS_i64 = tl.cast(BS, tl.int64)
    for bh in range(pid, bh_total, n_progs):
        b_i = tl.cast(bh // H_dim, tl.int64)
        h_i = tl.cast(bh % H_dim, tl.int64)
        a_bh = b_i * AB_i64 + h_i * AH_i64
        b_bh = b_i * BB_i64 + h_i * BH_i64
        y_bh = tl.cast(bh, tl.int64) * SD_i64

        for s0 in range(0, S_dim, BLOCK_S):
            s_off = s0 + tl.arange(0, BLOCK_S)
            s_mask = s_off < S_dim
            s_i64 = tl.cast(s_off[:, None], tl.int64)
            a_row = a_bh + s_i64 * AS_i64
            b_row = b_bh + s_i64 * BS_i64
            y_row = y_bh + s_i64 * D_i64

            for d0 in range(0, D_dim, BLOCK_D):
                d_off = d0 + tl.arange(0, BLOCK_D)
                d_mask = d_off < D_dim
                mask = s_mask[:, None] & d_mask[None, :]
                d_i64 = tl.cast(d_off[None, :], tl.int64)
                a_val = tl.load(A + a_row + d_i64, mask=mask, other=0.0).to(tl.float32)
                b_val = tl.load(B + b_row + d_i64, mask=mask, other=0.0).to(tl.float32)
                tl.store(Y + y_row + d_i64, a_val * b_val, mask=mask)

_ALLOWED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _match_mid_broadcast(x, c):
    if (
        not isinstance(x, torch.Tensor)
        or not isinstance(c, torch.Tensor)
        or x.device != c.device
        or x.device.type != "npu"
        or x.dtype not in _ALLOWED_DTYPES
        or c.dtype not in _ALLOWED_DTYPES
        or x.ndim != c.ndim
        or x.numel() == 0
        or c.numel() == 0
        or x.stride(-1) != 1
        or c.stride(-1) != 1
    ):
        return None
    if x.ndim == 3:
        S, H, D = x.shape
        if c.shape == (S, 1, D) and H > 1 and S > 0 and D > 0:
            return (S, S, H, D, 0, x.stride(0), x.stride(1), 0, c.stride(0))
    elif x.ndim == 4:
        B, S, H, D = x.shape
        if (
            (c.shape == (B, S, 1, D) or c.shape == (1, S, 1, D))
            and H > 1
            and B > 0
            and S > 0
            and D > 0
        ):
            cb = 0 if c.shape[0] == 1 else c.stride(0)
            return (B * S, S, H, D, x.stride(0), x.stride(1), x.stride(2), cb, c.stride(1))
    return None


def _try_mid_broadcast_mul(a, b):
    spec = _match_mid_broadcast(a, b)
    if spec is not None:
        x, c = a, b
    else:
        spec = _match_mid_broadcast(b, a)
        if spec is None:
            return None
        x, c = b, a

    M_total, S_len, H, D, XB, XS, XH, CB, CS = spec
    out_dtype = torch.promote_types(x.dtype, c.dtype)
    out = torch.empty(x.shape, dtype=out_dtype, device=x.device)
    n_progs = min(int(M_total), 2048)
    with torch_device_fn.device(x.device):
        _mul_mid_broadcast_kernel[(n_progs,)](
            x,
            c,
            out,
            int(M_total),
            int(S_len),
            int(H),
            int(D),
            int(XB),
            int(XS),
            int(XH),
            int(CB),
            int(CS),
            int(n_progs),
            BLOCK_H=16,
            BLOCK_D=128,
        )
    return out


def _try_trailing_scalar_mul(a, b):
    if (
        not isinstance(a, torch.Tensor)
        or not isinstance(b, torch.Tensor)
        or a.device != b.device
        or a.device.type != "npu"
        or a.dtype not in _ALLOWED_DTYPES
        or b.dtype not in _ALLOWED_DTYPES
        or a.ndim != b.ndim
        or a.ndim < 2
        or a.numel() == 0
        or b.numel() == 0
    ):
        return None

    x, c = (a, b) if a.numel() >= b.numel() else (b, a)

    # Path 1: Contiguous N-D tensors with trailing size-1 dimensions on c
    if x.numel() > c.numel() and x.is_contiguous() and c.is_contiguous():
        t = 0
        for dim in reversed(c.shape):
            if dim == 1:
                t += 1
            else:
                break
        if t >= 1 and x.shape[: x.ndim - t] == c.shape[: c.ndim - t]:
            M = int(c.numel())
            K = int(x.numel()) // M
            if M > 0 and K > 1:
                out_dtype = torch.promote_types(x.dtype, c.dtype)
                out = torch.empty(x.shape, dtype=out_dtype, device=x.device)
                with torch_device_fn.device(x.device):
                    if K <= 256:
                        block_k = 64 if K <= 64 else 256
                        block_m = 32
                        n_progs = min((M + block_m - 1) // block_m, 2048)
                        _mul_row_scalar_smallk_kernel[(n_progs,)](
                            x, c, out, M, K, int(n_progs),
                            BLOCK_M=block_m, BLOCK_K=block_k,
                        )
                    else:
                        n_progs = min(M, 2048)
                        _mul_row_scalar_largek_kernel[(n_progs,)](
                            x, c, out, M, K, int(n_progs),
                            BLOCK_K=1024,
                        )
                return out

    # Path 2: Strided 4D (B, H, S, D) * (B, H, S, 1) with unit inner stride
    if (
        x.ndim == 4
        and c.shape == (x.shape[0], x.shape[1], x.shape[2], 1)
        and x.shape[3] > 1
        and x.stride(-1) == 1
        and c.stride(-1) == 1
    ):
        B_dim, H_dim, S_dim, D_dim = x.shape
        out_dtype = torch.promote_types(x.dtype, c.dtype)
        out = torch.empty(x.shape, dtype=out_dtype, device=x.device)
        n_progs = min(int(B_dim * H_dim), 2048)
        with torch_device_fn.device(x.device):
            _mul_4d_last1_strided_kernel[(n_progs,)](
                x, c, out,
                int(B_dim), int(H_dim), int(S_dim), int(D_dim),
                int(x.stride(0)), int(x.stride(1)), int(x.stride(2)),
                int(c.stride(0)), int(c.stride(1)), int(c.stride(2)),
                int(n_progs),
                BLOCK_S=16,
                BLOCK_D=128,
            )
        return out

    return None


def _try_4d_same_strided_mul(a, b):
    if (
        not isinstance(a, torch.Tensor)
        or not isinstance(b, torch.Tensor)
        or a.device != b.device
        or a.device.type != "npu"
        or a.dtype not in _ALLOWED_DTYPES
        or b.dtype not in _ALLOWED_DTYPES
        or a.ndim != 4
        or b.ndim != 4
        or a.shape != b.shape
        or a.numel() == 0
        or (a.is_contiguous() and b.is_contiguous())
        or a.stride(-1) != 1
        or b.stride(-1) != 1
    ):
        return None
    B_dim, H_dim, S_dim, D_dim = a.shape
    if B_dim * H_dim <= 0 or B_dim * H_dim > 2**31 - 1:
        return None
    out_dtype = torch.promote_types(a.dtype, b.dtype)
    out = torch.empty(a.shape, dtype=out_dtype, device=a.device)
    n_progs = min(int(B_dim * H_dim), 2048)
    with torch_device_fn.device(a.device):
        _mul_4d_same_strided_kernel[(n_progs,)](
            a, b, out,
            int(B_dim), int(H_dim), int(S_dim), int(D_dim),
            int(a.stride(0)), int(a.stride(1)), int(a.stride(2)),
            int(b.stride(0)), int(b.stride(1)), int(b.stride(2)),
            int(n_progs),
            BLOCK_S=32,
            BLOCK_D=64,
        )
    return out

def make_patched_mul_vision_rope(base_mul):
    if getattr(base_mul, "_wave3_vision_mul_patched", False):
        return base_mul

    @functools.wraps(base_mul)
    def patched_mul_vision_rope(a, b):
        out = _try_mid_broadcast_mul(a, b)
        if out is not None:
            return out
        out = _try_trailing_scalar_mul(a, b)
        if out is not None:
            return out
        out = _try_4d_same_strided_mul(a, b)
        if out is not None:
            return out
        return base_mul(a, b)

    patched_mul_vision_rope._wave3_patched = True
    patched_mul_vision_rope._wave3_vision_mul_patched = True
    return patched_mul_vision_rope
