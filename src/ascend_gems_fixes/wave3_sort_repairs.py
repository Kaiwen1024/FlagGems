"""Process-local UB-safe Ascend Triton sort / argsort repair for Wave 3.

Eliminates BiShengIR UB overflow (4.33 Mbits > 1.57 Mbits) in generic
`flag_gems/ops/sort.py` `compute_global_hist_kernel` (which allocates a 2D
(TILE_R=16, TILE_N=1024) int64 tensor) and avoids `sweep` decoupled-lookback
spinlock risks by implementing a 2-kernel persistent 1D-grid 4-bit radix sort
using TILE_N=256 and NPU-native `torch.cumsum` global offset prefixing.

Preserves stability across all 4-bit radix passes, supports N >= 0, all
integer/float/bool dtypes, arbitrary dim, ascending/descending, and handles
Qwen2.5-VL `torch.argsort(window_index)` without dynamic-shape UB overflow.
"""
from __future__ import annotations

import functools
import importlib
import sys

_SORT_KERNELS = None


def _get_sort_kernels():
    global _SORT_KERNELS
    if _SORT_KERNELS is None:
        import triton
        import triton.language as tl
        from flag_gems.utils import libentry

        @tl.constexpr
        def _get_int_t(num_bits: tl.constexpr, signed: tl.constexpr) -> tl.dtype:
            nb = num_bits.value if isinstance(num_bits, tl.constexpr) else num_bits
            sg = signed.value if isinstance(signed, tl.constexpr) else signed
            return tl.core.get_int_dtype(nb, sg)

        @tl.constexpr
        def _sign_mask_val(num_bits: tl.constexpr) -> int:
            nb = num_bits.value if isinstance(num_bits, tl.constexpr) else num_bits
            return 1 << (nb - 1)

        @tl.constexpr
        def _low_mask_val(num_bits: tl.constexpr) -> int:
            nb = num_bits.value if isinstance(num_bits, tl.constexpr) else num_bits
            return (1 << (nb - 1)) - 1

        @triton.jit
        def _convert_to_uint(x, descending: tl.constexpr = False):
            if x.dtype.is_floating():
                num_bits: tl.constexpr = x.dtype.primitive_bitwidth
                sdtype = _get_int_t(num_bits, True)
                udtype = _get_int_t(num_bits, False)
                sx = x.to(sdtype, bitcast=True)
                ux = x.to(udtype, bitcast=True)
                sign_bit = tl.full((), value=_sign_mask_val(num_bits), dtype=udtype)
                rshift = tl.full((), value=num_bits - 1, dtype=sdtype)
                mask = sign_bit | (sx >> rshift).to(udtype, bitcast=True)
                return ux ^ (~mask) if descending else ux ^ mask
            elif x.dtype.is_int_signed():
                num_bits: tl.constexpr = x.dtype.primitive_bitwidth
                udtype = _get_int_t(num_bits, False)
                ux = tl.cast(x, udtype, bitcast=True)
                if descending:
                    bmask = tl.full((), value=_low_mask_val(num_bits), dtype=udtype)
                    return ux ^ bmask
                else:
                    smask = tl.full((), value=_sign_mask_val(num_bits), dtype=udtype)
                    return ux ^ smask
            else:
                return ~x if descending else x

        @libentry()
        @triton.jit(do_not_specialize=["bit_offset", "M", "N", "grid_n", "total_tiles", "n_progs"])
        def _radix_block_hist_kernel(
            arr_ptr,
            block_hist_ptr,
            bit_offset,
            M,
            N,
            grid_n,
            total_tiles,
            n_progs,
            TILE_N: tl.constexpr,
            DESCENDING: tl.constexpr,
        ):
            pid = tl.program_id(0)
            n_i64 = tl.zeros([], dtype=tl.int64) + N
            grid_n_i64 = tl.zeros([], dtype=tl.int64) + grid_n
            for tile_id in range(pid, total_tiles, n_progs):
                pid_m = tile_id // grid_n
                pid_n = tile_id % grid_n
                m_i64 = pid_m.to(tl.int64)
                n_off = pid_n.to(tl.int64) * TILE_N + tl.arange(0, TILE_N).to(tl.int64)
                mask = n_off < n_i64
                arr = tl.load(arr_ptr + m_i64 * n_i64 + n_off, mask=mask)
                arr_u = _convert_to_uint(arr, DESCENDING)
                shift = tl.zeros([], dtype=arr_u.dtype) + bit_offset
                mask15 = tl.full((), 15, dtype=arr_u.dtype)
                key = ((arr_u >> shift) & mask15).to(tl.int32)
                hist_base = m_i64 * (16 * grid_n_i64) + pid_n.to(tl.int64)
                for b in range(16):
                    m_b = mask & (key == b)
                    cnt = tl.sum(m_b.to(tl.int32), axis=0)
                    tl.store(block_hist_ptr + hist_base + b * grid_n_i64, cnt)

        @libentry()
        @triton.jit(do_not_specialize=["bit_offset", "M", "N", "grid_n", "total_tiles", "n_progs"])
        def _radix_block_scatter_kernel(
            arr_in_ptr,
            idx_in_ptr,
            arr_out_ptr,
            idx_out_ptr,
            ex_offsets_ptr,
            bit_offset,
            M,
            N,
            grid_n,
            total_tiles,
            n_progs,
            TILE_N: tl.constexpr,
            DESCENDING: tl.constexpr,
        ):
            pid = tl.program_id(0)
            n_i64 = tl.zeros([], dtype=tl.int64) + N
            grid_n_i64 = tl.zeros([], dtype=tl.int64) + grid_n
            for tile_id in range(pid, total_tiles, n_progs):
                pid_m = tile_id // grid_n
                pid_n = tile_id % grid_n
                m_i64 = pid_m.to(tl.int64)
                n_off = pid_n.to(tl.int64) * TILE_N + tl.arange(0, TILE_N).to(tl.int64)
                mask = n_off < n_i64
                src_off = m_i64 * n_i64 + n_off
                arr = tl.load(arr_in_ptr + src_off, mask=mask)
                idx = tl.load(idx_in_ptr + src_off, mask=mask)
                arr_u = _convert_to_uint(arr, DESCENDING)
                shift = tl.zeros([], dtype=arr_u.dtype) + bit_offset
                mask15 = tl.full((), 15, dtype=arr_u.dtype)
                key = ((arr_u >> shift) & mask15).to(tl.int32)
                off_base = m_i64 * (16 * grid_n_i64) + pid_n.to(tl.int64)
                row_out_base = m_i64 * n_i64
                for b in range(16):
                    m_b = mask & (key == b)
                    base = tl.load(ex_offsets_ptr + off_base + b * grid_n_i64).to(tl.int64)
                    local_rank = tl.cumsum(m_b.to(tl.int32), axis=0) - 1
                    safe_pos = tl.where(m_b, base + local_rank.to(tl.int64), 0)
                    dst = row_out_base + safe_pos
                    tl.store(arr_out_ptr + dst, arr, mask=m_b)
                    tl.store(idx_out_ptr + dst, idx, mask=m_b)

        _SORT_KERNELS = (_radix_block_hist_kernel, _radix_block_scatter_kernel)
    return _SORT_KERNELS


def wave3_radix_sort(arr, k_bits=4, descending=False):
    import torch
    import triton
    from flag_gems.runtime import torch_device_fn

    assert k_bits == 4, f"wave3_radix_sort expects k_bits=4, got {k_bits}"
    orig_shape = arr.shape
    n = orig_shape[-1]
    if n == 0:
        return arr.clone(), torch.empty(orig_shape, dtype=torch.int64, device=arr.device)
    if n == 1:
        return arr.clone(), torch.zeros(orig_shape, dtype=torch.int64, device=arr.device)

    orig_dtype = arr.dtype
    work_in = arr.to(torch.uint8) if orig_dtype == torch.bool else arr.clone()
    num_bits = 1 if orig_dtype == torch.bool else (work_in.element_size() * 8)
    n_passes = 1 if orig_dtype == torch.bool else ((num_bits + 3) // 4)

    m = work_in.numel() // n
    if m == 0:
        return arr.clone(), torch.empty(orig_shape, dtype=torch.int64, device=arr.device)

    hist_kernel, scatter_kernel = _get_sort_kernels()
    TILE_N = 256
    grid_n = triton.cdiv(n, TILE_N)
    total_tiles = m * grid_n
    n_progs = min(total_tiles, 4096)

    with torch_device_fn.device(arr.device):
        idx_in = (
            torch.arange(0, n, dtype=torch.int64, device=arr.device)
            .broadcast_to(orig_shape)
            .contiguous()
        )
        work_out = torch.empty_like(work_in)
        idx_out = torch.empty_like(idx_in)
        block_hist = torch.empty((m, 16, grid_n), device=arr.device, dtype=torch.int32)

        for p in range(n_passes):
            bit_offset = p * 4
            hist_kernel[(n_progs,)](
                work_in,
                block_hist,
                bit_offset,
                m,
                n,
                grid_n,
                total_tiles,
                n_progs,
                TILE_N,
                descending,
            )
            flat = block_hist.reshape(m, 16 * grid_n)
            if n >= (1 << 30):
                flat = flat.to(torch.int64)
            ex_offsets = (torch.cumsum(flat, dim=-1) - flat).to(torch.int64).reshape(m, 16, grid_n)

            scatter_kernel[(n_progs,)](
                work_in,
                idx_in,
                work_out,
                idx_out,
                ex_offsets,
                bit_offset,
                m,
                n,
                grid_n,
                total_tiles,
                n_progs,
                TILE_N,
                descending,
            )
            work_in, work_out = work_out, work_in
            idx_in, idx_out = idx_out, idx_in

        if orig_dtype == torch.bool:
            work_in = work_in.to(torch.bool)
    return work_in, idx_in


def wave3_sort_stable(inp, *, stable=False, dim=-1, descending=False):
    import torch

    _ = stable
    if dim < 0:
        dim = dim + inp.ndim
    if inp.shape[dim] == 0:
        return inp.clone(), torch.empty(inp.shape, dtype=torch.int64, device=inp.device)
    if inp.shape[dim] == 1:
        return inp.clone(), torch.zeros(inp.shape, dtype=torch.int64, device=inp.device)

    moved = False
    if dim != inp.ndim - 1:
        work = torch.movedim(inp, dim, -1).contiguous()
        moved = True
    else:
        work = inp.contiguous()

    out, out_index = wave3_radix_sort(work, k_bits=4, descending=descending)
    if moved:
        out = torch.movedim(out, -1, dim).contiguous()
        out_index = torch.movedim(out_index, -1, dim).contiguous()
    return out, out_index


def wave3_sort(inp, dim=-1, descending=False):
    return wave3_sort_stable(inp, stable=False, dim=dim, descending=descending)


def wave3_argsort(inp, dim=-1, descending=False):
    _, indices = wave3_sort_stable(inp, stable=True, dim=dim, descending=descending)
    return indices


wave3_sort._wave3_patched = True
wave3_sort_stable._wave3_patched = True
wave3_argsort._wave3_patched = True


def install_sort_repairs(repairs_log: list[str] | None = None) -> dict:
    """Patch FlagGems sort/argsort modules with UB-safe Ascend Triton radix sort."""
    patched = []
    for mod_name in ('flag_gems.ops.sort',):
        try:
            mod = importlib.import_module(mod_name)
            mod.radix_sort = wave3_radix_sort
            mod.sort = wave3_sort
            mod.sort_stable = wave3_sort_stable
            patched.append(mod_name)
        except Exception:
            pass
    for mod_name in ('flag_gems.ops.argsort', 'flag_gems'):
        try:
            mod = importlib.import_module(mod_name)
            if mod_name == 'flag_gems.ops.argsort' or hasattr(mod, 'argsort'):
                mod.argsort = wave3_argsort
                patched.append(mod_name)
        except Exception:
            pass
    for mod_name in ('_ascend.ops.sort', 'flag_gems.runtime.backend._ascend.ops.sort'):
        try:
            mod = sys.modules.get(mod_name)
            if mod is not None and hasattr(mod, 'sort'):
                mod.sort = wave3_sort
                patched.append(mod_name)
        except Exception:
            pass
    if repairs_log is not None and patched:
        repairs_log.append(
            "Replace FlagGems generic sort/argsort (UB-overflowing 2D compute_global_hist_kernel + decoupled-lookback sweep) with UB-safe 2-kernel persistent 1D-grid 4-bit Ascend Triton radix sort + torch.cumsum prefixing."
        )
    return {'patched': patched}
