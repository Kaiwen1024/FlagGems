"""Shape-agnostic 64-bit persistent Ascend Triton pad / constant_pad_nd repair.

Eliminates:
  1) Per-shape Triton recompilations caused by @libentry + tl.constexpr elem counts
     in upstream flag_gems.ops.pad.
  2) Heavy 1D multi-division elementwise overhead in upstream constant_pad_nd / pad
     on Qwen3.5 GatedDeltaNet F.pad calls.
"""
from __future__ import annotations

import functools
import math
import torch
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn

_ALLOWED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_CALL_STATS = {"inner1": 0, "pad2d": 0, "fallback": 0}


@triton.jit(do_not_specialize=[
    "Outer", "InLen", "OutLen", "pad_l", "fill_val", "n_progs"
])
def _pad_1axis_inner1_kernel(
    X, Y,
    Outer, InLen, OutLen,
    pad_l, fill_val, n_progs,
    BLOCK_L: tl.constexpr,
):
    pid = tl.program_id(0)
    OutLen_i64 = tl.cast(OutLen, tl.int64)
    InLen_i64 = tl.cast(InLen, tl.int64)
    pad_l_i64 = tl.cast(pad_l, tl.int64)
    n_l_tiles = (OutLen + BLOCK_L - 1) // BLOCK_L
    n_l_tiles_i64 = tl.cast(n_l_tiles, tl.int64)
    total_tiles = Outer * n_l_tiles
    fill_f32 = tl.cast(fill_val, tl.float32)

    for t_idx in range(pid, total_tiles, n_progs):
        t_i = tl.cast(t_idx, tl.int64)
        o = t_i // n_l_tiles_i64
        lt = t_i % n_l_tiles_i64
        l_off = lt * BLOCK_L + tl.cast(tl.arange(0, BLOCK_L), tl.int64)
        dst_mask = l_off < OutLen_i64
        src_l = l_off - pad_l_i64
        in_mask = dst_mask & (src_l >= 0) & (src_l < InLen_i64)
        safe_src_l = tl.where(in_mask, src_l, 0)
        x_ptr = X + o * InLen_i64 + safe_src_l
        y_ptr = Y + o * OutLen_i64 + l_off
        val = tl.load(x_ptr, mask=in_mask, other=0.0).to(tl.float32)
        val = tl.where(in_mask, val, fill_f32)
        tl.store(y_ptr, val, mask=dst_mask)


@triton.jit(do_not_specialize=[
    "Outer", "InLen", "OutLen", "Inner", "pad_l", "fill_val", "n_progs"
])
def _pad_1axis_2d_kernel(
    X, Y,
    Outer, InLen, OutLen, Inner,
    pad_l, fill_val, n_progs,
    BLOCK_L: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid = tl.program_id(0)
    OutLen_i64 = tl.cast(OutLen, tl.int64)
    InLen_i64 = tl.cast(InLen, tl.int64)
    Inner_i64 = tl.cast(Inner, tl.int64)
    pad_l_i64 = tl.cast(pad_l, tl.int64)
    n_l_tiles = (OutLen + BLOCK_L - 1) // BLOCK_L
    n_l_tiles_i64 = tl.cast(n_l_tiles, tl.int64)
    total_tiles = Outer * n_l_tiles
    fill_f32 = tl.cast(fill_val, tl.float32)

    for t_idx in range(pid, total_tiles, n_progs):
        t_i = tl.cast(t_idx, tl.int64)
        o = t_i // n_l_tiles_i64
        lt = t_i % n_l_tiles_i64
        l_off = lt * BLOCK_L + tl.cast(tl.arange(0, BLOCK_L), tl.int64)
        l_mask = l_off < OutLen_i64
        src_l = l_off - pad_l_i64
        in_l_mask = l_mask & (src_l >= 0) & (src_l < InLen_i64)
        safe_src_l = tl.where(in_l_mask, src_l, 0)
        x_base = o * InLen_i64 * Inner_i64 + safe_src_l[:, None] * Inner_i64
        y_base = o * OutLen_i64 * Inner_i64 + l_off[:, None] * Inner_i64
        for i0 in range(0, Inner, BLOCK_I):
            i_off = i0 + tl.cast(tl.arange(0, BLOCK_I), tl.int64)
            i_mask = i_off < Inner_i64
            load_mask = in_l_mask[:, None] & i_mask[None, :]
            store_mask = l_mask[:, None] & i_mask[None, :]
            val = tl.load(X + x_base + i_off[None, :], mask=load_mask, other=0.0).to(tl.float32)
            val = tl.where(load_mask, val, fill_f32)
            tl.store(Y + y_base + i_off[None, :], val, mask=store_mask)


def _try_fast_constant_pad_1axis(self, pad_list, value=0):
    if (
        not isinstance(self, torch.Tensor)
        or self.device.type != "npu"
        or not self.is_contiguous()
        or self.dtype not in _ALLOWED_DTYPES
        or self.numel() == 0
        or self.ndim == 0
        or len(pad_list) % 2 != 0
        or (len(pad_list) // 2) > self.ndim
    ):
        return None

    try:
        pads = [int(p) for p in pad_list]
    except Exception:
        return None

    if any(p < 0 for p in pads):
        return None

    n_pairs = len(pads) // 2
    nz = [
        (self.ndim - 1 - i, pads[2 * i], pads[2 * i + 1])
        for i in range(n_pairs)
        if pads[2 * i] != 0 or pads[2 * i + 1] != 0
    ]
    if len(nz) > 1:
        return None

    if len(nz) == 0:
        ax, pad_l, pad_r = self.ndim - 1, 0, 0
    else:
        ax, pad_l, pad_r = nz[0]

    Outer = int(math.prod(self.shape[:ax]))
    InLen = int(self.shape[ax])
    OutLen = InLen + pad_l + pad_r
    Inner = int(math.prod(self.shape[ax + 1 :]))
    block_l = 1024 if Inner == 1 else 16
    n_l_tiles = (OutLen + block_l - 1) // block_l
    total_tiles = Outer * n_l_tiles
    if OutLen <= 0 or total_tiles <= 0 or total_tiles > 2**31 - 1:
        return None

    out_shape = list(self.shape)
    out_shape[ax] = OutLen
    out = torch.empty(out_shape, dtype=self.dtype, device=self.device)
    fill_val = 0.0 if value is None else float(value)
    n_progs = min(int(total_tiles), 2048)

    with torch_device_fn.device(self.device):
        if Inner == 1:
            _CALL_STATS["inner1"] += 1
            _pad_1axis_inner1_kernel[(n_progs,)](
                self, out,
                int(Outer), int(InLen), int(OutLen),
                int(pad_l), fill_val, int(n_progs),
                BLOCK_L=1024,
            )
        else:
            _CALL_STATS["pad2d"] += 1
            _pad_1axis_2d_kernel[(n_progs,)](
                self, out,
                int(Outer), int(InLen), int(OutLen), int(Inner),
                int(pad_l), fill_val, int(n_progs),
                BLOCK_L=16,
                BLOCK_I=128,
            )
    return out


def make_patched_constant_pad_nd(base_pad):
    if getattr(base_pad, "_wave3_const_pad_patched", False):
        return base_pad

    @functools.wraps(base_pad)
    def patched_constant_pad_nd(self, pad_list, value=0):
        out = _try_fast_constant_pad_1axis(self, pad_list, value)
        if out is not None:
            return out
        _CALL_STATS["fallback"] += 1
        return base_pad(self, pad_list, value)

    patched_constant_pad_nd._wave3_patched = True
    patched_constant_pad_nd._wave3_const_pad_patched = True
    return patched_constant_pad_nd


def make_patched_pad(base_pad):
    if getattr(base_pad, "_wave3_pad_patched", False):
        return base_pad

    @functools.wraps(base_pad)
    def patched_pad(self, pad, mode="constant", value=None):
        if mode == "constant":
            out = _try_fast_constant_pad_1axis(self, pad, value)
            if out is not None:
                return out
        _CALL_STATS["fallback"] += 1
        return base_pad(self, pad, mode=mode, value=value)

    patched_pad._wave3_patched = True
    patched_pad._wave3_pad_patched = True
    return patched_pad
