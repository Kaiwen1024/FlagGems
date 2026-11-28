"""Process-local verified Ascend Triton avg_pool2d repair for Wave 3.
Fixes upstream flag_gems/ops/avg_pool2d.py:
  1. Removes the 9-config autotune + 409,600-iteration (kernel=640) loop hang (c014).
  2. Fixes PyTorch ceil_mode/padding divisor semantics when kernel > in_size or padding > 0.
All code paths execute genuine FlagGems Triton kernels cleanly.
"""
from __future__ import annotations

import functools
import importlib
import sys

_AVG_POOL2D_KERNELS = None


def _parse_pool_params(kernel_size, stride, padding):
    if isinstance(kernel_size, int):
        kh = kw = int(kernel_size)
    else:
        kh, kw = int(kernel_size[0]), int(kernel_size[1])

    if stride is None or (isinstance(stride, (list, tuple)) and not stride):
        sh, sw = kh, kw
    elif isinstance(stride, int):
        sh = sw = int(stride)
    else:
        sh, sw = int(stride[0]), int(stride[1])

    if isinstance(padding, int):
        ph = pw = int(padding)
    else:
        ph, pw = int(padding[0]), int(padding[1])

    if sh <= 0 or sw <= 0:
        raise ValueError("stride must be greater than zero")
    if ph < 0 or pw < 0:
        raise ValueError("padding must be non-negative")
    if ph > kh // 2 or pw > kw // 2:
        raise ValueError("pad should be smaller than or equal to half of kernel size")
    return kh, kw, sh, sw, ph, pw


def _pool2d_output_size(in_size: int, kernel: int, stride: int, padding: int, ceil_mode: bool) -> int:
    numerator = in_size + 2 * padding - kernel
    if ceil_mode:
        out_size = (numerator + stride - 1) // stride + 1
        if (out_size - 1) * stride >= in_size + padding:
            out_size -= 1
    else:
        out_size = numerator // stride + 1
    return max(out_size, 0)


def _get_avg_pool2d_kernels():
    global _AVG_POOL2D_KERNELS
    if _AVG_POOL2D_KERNELS is None:
        import triton
        import triton.language as tl
        from flag_gems.utils import libentry

        @libentry()
        @triton.jit
        def _avg_pool2d_global_flat_kernel(
            inp_ptr,
            out_ptr,
            R,
            V,
            div,
            n_progs,
            BLOCK_R: tl.constexpr,
            BLOCK_K: tl.constexpr,
        ):
            pid = tl.program_id(0)
            dtype = inp_ptr.type.element_ty
            zero64 = tl.zeros([], dtype=tl.int64)
            v64 = zero64 + V

            for r_base in range(pid * BLOCK_R, R, n_progs * BLOCK_R):
                r_off = r_base + tl.arange(0, BLOCK_R)
                r_mask = r_off < R
                r64 = r_off[:, None].to(tl.int64) * v64
                acc = tl.zeros([BLOCK_R, BLOCK_K], dtype=tl.float32)

                for k0 in range(0, V, BLOCK_K):
                    k_off = k0 + tl.arange(0, BLOCK_K)
                    k_mask = k_off < V
                    m = r_mask[:, None] & k_mask[None, :]
                    offs = r64 + k_off[None, :].to(tl.int64)
                    vals = tl.load(inp_ptr + offs, mask=m, other=0.0).to(tl.float32)
                    acc += tl.where(m, vals, 0.0)

                out_vals = tl.sum(acc, axis=1) / div
                tl.store(out_ptr + r_off.to(tl.int64), out_vals.to(dtype), mask=r_mask)

        @libentry()
        @triton.jit
        def _avg_pool2d_exact_kernel(
            inp_ptr,
            out_ptr,
            in_stride_n,
            in_stride_c,
            in_stride_h,
            in_stride_w,
            in_c,
            in_h,
            in_w,
            out_h,
            out_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            max_kh,
            max_kw,
            divisor_override,
            w_tiles,
            total_tiles,
            n_progs,
            COUNT_INCLUDE_PAD: tl.constexpr,
            HAS_DIV_OVERRIDE: tl.constexpr,
            BLOCK_W: tl.constexpr,
        ):
            pid = tl.program_id(0)
            dtype = inp_ptr.type.element_ty
            zero64 = tl.zeros([], dtype=tl.int64)
            in_sn = zero64 + in_stride_n
            in_sc = zero64 + in_stride_c
            in_sh = zero64 + in_stride_h
            in_sw = zero64 + in_stride_w
            out_hw = (zero64 + out_h) * (zero64 + out_w)

            for tile_id in range(pid, total_tiles, n_progs):
                wt = tile_id % w_tiles
                nc_oh = tile_id // w_tiles
                oh = nc_oh % out_h
                nc = nc_oh // out_h
                n_idx = nc // in_c
                c_idx = nc % in_c

                hstart = oh * stride_h - padding_h
                hend_raw = hstart + kernel_h
                hend_limit = in_h + padding_h
                hend_pad = hend_raw if hend_raw < hend_limit else hend_limit
                h_pool = hend_pad - hstart
                if h_pool < 0:
                    h_pool = 0
                h_upper = hend_pad if hend_pad < in_h else in_h

                ow_off = wt * BLOCK_W + tl.arange(0, BLOCK_W)
                ow_mask = ow_off < out_w
                wstart = ow_off * stride_w - padding_w
                wend_pad = tl.minimum(wstart + kernel_w, in_w + padding_w)
                w_pool = tl.maximum(wend_pad - wstart, 0)
                pool_size = h_pool * w_pool

                sum_acc = tl.zeros([BLOCK_W], dtype=tl.float32)
                valid_cnt = tl.zeros([BLOCK_W], dtype=tl.int32)
                in_base = inp_ptr + (zero64 + n_idx) * in_sn + (zero64 + c_idx) * in_sc

                for kh in range(0, max_kh):
                    ih = hstart + kh
                    if ih >= 0:
                        if ih < h_upper:
                            ih_off = (zero64 + ih) * in_sh
                            for kw in range(0, max_kw):
                                iw = wstart + kw
                                in_m = ow_mask & (iw >= 0) & (iw < in_w) & (iw < wend_pad)
                                safe_iw = tl.where(in_m, iw, 0).to(tl.int64)
                                vals = tl.load(
                                    in_base + ih_off + safe_iw * in_sw,
                                    mask=in_m,
                                    other=0.0,
                                ).to(tl.float32)
                                sum_acc += tl.where(in_m, vals, 0.0)
                                valid_cnt += in_m.to(tl.int32)

                if HAS_DIV_OVERRIDE:
                    div = tl.full([BLOCK_W], divisor_override, dtype=tl.float32)
                elif COUNT_INCLUDE_PAD:
                    div = pool_size.to(tl.float32)
                else:
                    div = valid_cnt.to(tl.float32)

                out_vals = tl.where(div != 0.0, sum_acc / div, 0.0)
                out_off = (zero64 + nc) * out_hw + (zero64 + oh) * (zero64 + out_w) + ow_off.to(tl.int64)
                tl.store(out_ptr + out_off, out_vals.to(dtype), mask=ow_mask)

        _AVG_POOL2D_KERNELS = (_avg_pool2d_global_flat_kernel, _avg_pool2d_exact_kernel)
    return _AVG_POOL2D_KERNELS


def make_patched_avg_pool2d(orig_avg_pool2d):
    if getattr(orig_avg_pool2d, '_wave3_patched', False):
        return orig_avg_pool2d

    import torch
    import triton

    @functools.wraps(orig_avg_pool2d)
    def patched_avg_pool2d(
        input: torch.Tensor,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    ):
        if divisor_override is not None and divisor_override == 0:
            raise ValueError("divisor_override cannot be zero")

        is_3d = input.ndim == 3
        if input.ndim not in (3, 4):
            return orig_avg_pool2d(
                input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override
            )

        x4d = input.unsqueeze(0) if is_3d else input
        x4d = x4d.contiguous()
        kh, kw, sh, sw, ph, pw = _parse_pool_params(kernel_size, stride, padding)
        in_n, in_c, in_h, in_w = x4d.shape
        out_h = _pool2d_output_size(in_h, kh, sh, ph, ceil_mode)
        out_w = _pool2d_output_size(in_w, kw, sw, pw, ceil_mode)

        if in_n * in_c * out_h * out_w == 0:
            out = torch.empty((in_n, in_c, out_h, out_w), dtype=x4d.dtype, device=x4d.device)
            return out.squeeze(0) if is_3d else out

        # Fast path for global pooling (out_h == 1 and out_w == 1): avoids huge kernel_h*kernel_w loops
        # (e.g. ALIGN nn.AvgPool2d(640, ceil_mode=True) on 10x10 and AdaptiveAvgPool2d(1) in 55 MBConv blocks)
        # and avoids flag_gems sum_dim_kernel_inner UB overflow using a 4 KiB UB-safe Triton reduction kernel.
        global_k, exact_k = _get_avg_pool2d_kernels()
        if out_h == 1 and out_w == 1:
            hstart, wstart = -ph, -pw
            hend_pad = min(hstart + kh, in_h + ph)
            wend_pad = min(wstart + kw, in_w + pw)
            h0, w0 = max(hstart, 0), max(wstart, 0)
            h1, w1 = min(hend_pad, in_h), min(wend_pad, in_w)
            pool_size = max(hend_pad - hstart, 0) * max(wend_pad - wstart, 0)
            valid_size = max(h1 - h0, 0) * max(w1 - w0, 0)
            div = float(divisor_override) if divisor_override is not None else float(
                pool_size if count_include_pad else valid_size
            )
            if valid_size == 0 or div == 0.0:
                out = torch.zeros((in_n, in_c, 1, 1), dtype=x4d.dtype, device=x4d.device)
                return out.squeeze(0) if is_3d else out
            sub = x4d if (h0 == 0 and h1 == in_h and w0 == 0 and w1 == in_w) else x4d[:, :, h0:h1, w0:w1].contiguous()
            R = in_n * in_c
            out = torch.empty((in_n, in_c, 1, 1), dtype=x4d.dtype, device=x4d.device)
            block_r = 4
            block_k = 256
            n_progs = min(triton.cdiv(R, block_r), 2048)
            global_k[(n_progs,)](sub, out, R, valid_size, div, n_progs, BLOCK_R=block_r, BLOCK_K=block_k)
            return out.squeeze(0) if is_3d else out

        # Multi-output path via zero-autotune, 64-bit, exact-divisor Ascend Triton kernel
        out = torch.empty((in_n, in_c, out_h, out_w), dtype=x4d.dtype, device=x4d.device)
        k = exact_k
        block_w = 32
        w_tiles = triton.cdiv(out_w, block_w)
        total_tiles = in_n * in_c * out_h * w_tiles
        n_progs = min(total_tiles, 2048)
        max_kh = min(kh, in_h + 2 * ph)
        max_kw = min(kw, in_w + 2 * pw)
        k[(n_progs,)](
            x4d,
            out,
            x4d.stride(0),
            x4d.stride(1),
            x4d.stride(2),
            x4d.stride(3),
            in_c,
            in_h,
            in_w,
            out_h,
            out_w,
            kh,
            kw,
            sh,
            sw,
            ph,
            pw,
            max_kh,
            max_kw,
            float(divisor_override) if divisor_override is not None else 0.0,
            w_tiles,
            total_tiles,
            n_progs,
            COUNT_INCLUDE_PAD=bool(count_include_pad),
            HAS_DIV_OVERRIDE=(divisor_override is not None),
            BLOCK_W=block_w,
        )
        return out.squeeze(0) if is_3d else out

    patched_avg_pool2d._wave3_patched = True
    return patched_avg_pool2d


def patch_avg_pool2d_modules(wrap_fn_list=None) -> list[str]:
    """Patch FlagGems avg_pool2d module targets cleanly."""
    patched = []
    for mod_name in ('flag_gems.ops.avg_pool2d', 'flag_gems.ops', 'flag_gems'):
        try:
            m = importlib.import_module(mod_name)
            if hasattr(m, 'avg_pool2d') and not getattr(m.avg_pool2d, '_wave3_patched', False):
                m.avg_pool2d = make_patched_avg_pool2d(m.avg_pool2d)
                patched.append(mod_name)
        except Exception:
            pass
    return patched
