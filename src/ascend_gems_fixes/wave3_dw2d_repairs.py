"""Process-local persistent 1D-grid Ascend Triton depthwise conv2d repair for Wave 3.

Fixes 3D grid overflow (total_tiles / blockDim > 65,535 -> ACL error 507035)
in wave3_repairs._dw2d_kernel when batch_size * C >= 65536 (e.g. c014 align-base,
c015 CLIP-convnext_large_d_320, c016 CLIP-convnext_xxlarge).
"""
from __future__ import annotations

import functools
import math

_DW2D_PERSISTENT_KERNEL = None


def _pair(v) -> tuple[int, int]:
    if isinstance(v, int):
        iv = int(v)
        return (iv, iv)
    vals = tuple(int(x) for x in v)
    if len(vals) == 1:
        return (vals[0], vals[0])
    return (vals[0], vals[1])


def _get_dw2d_persistent_kernel():
    global _DW2D_PERSISTENT_KERNEL
    if _DW2D_PERSISTENT_KERNEL is None:
        import triton
        import triton.language as tl
        from flag_gems.utils import libentry

        @libentry()
        @triton.jit
        def _dw2d_persistent_kernel(
            X,
            W,
            Y,
            BIAS,
            CO,
            H,
            IW,
            OH,
            OW,
            SN,
            SC,
            SH,
            SW,
            spatial_tiles,
            total_tiles,
            n_progs,
            KH: tl.constexpr,
            KW: tl.constexpr,
            STEP_H: tl.constexpr,
            STEP_W: tl.constexpr,
            PH: tl.constexpr,
            PW: tl.constexpr,
            MULT: tl.constexpr,
            HAS_BIAS: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            pid = tl.program_id(0)
            y_dtype = Y.type.element_ty
            zero64 = tl.zeros([], dtype=tl.int64)
            co64 = zero64 + CO
            ohow64 = (zero64 + OH) * (zero64 + OW)
            sn64 = zero64 + SN
            sc64 = zero64 + SC
            sh64 = zero64 + SH
            sw64 = zero64 + SW
            khkw64 = zero64 + (KH * KW)

            for tile_idx in range(pid, total_tiles, n_progs):
                sp = tile_idx % spatial_tiles
                rem = tile_idx // spatial_tiles
                ch = rem % CO
                b = rem // CO

                ch64 = zero64 + ch
                b64 = zero64 + b
                ch_in64 = zero64 + (ch // MULT)

                i = sp * BLOCK + tl.arange(0, BLOCK)
                i_mask = i < OH * OW
                oh = i // OW
                ow = i % OW

                acc = tl.full((BLOCK,), 0.0, tl.float32)
                for ky in tl.static_range(KH):
                    iy = oh * STEP_H - PH + ky
                    y_ok = i_mask & (iy >= 0) & (iy < H)
                    iy64 = iy.to(tl.int64) * sh64
                    for kx in tl.static_range(KW):
                        ix = ow * STEP_W - PW + kx
                        active = y_ok & (ix >= 0) & (ix < IW)
                        x_off = b64 * sn64 + ch_in64 * sc64 + iy64 + ix.to(tl.int64) * sw64
                        safe_x_off = tl.where(active, x_off, 0)
                        val = tl.load(X + safe_x_off, mask=active, other=0.0).to(tl.float32)
                        w_off = ch64 * khkw64 + (ky * KW + kx)
                        coeff = tl.load(W + w_off).to(tl.float32)
                        acc = tl.fma(val, coeff, acc)

                if HAS_BIAS:
                    acc += tl.load(BIAS + ch64).to(tl.float32)

                y_off = (b64 * co64 + ch64) * ohow64 + i.to(tl.int64)
                safe_y_off = tl.where(i_mask, y_off, 0)
                tl.store(Y + safe_y_off, acc.to(y_dtype), mask=i_mask)

        _DW2D_PERSISTENT_KERNEL = _dw2d_persistent_kernel
    return _DW2D_PERSISTENT_KERNEL


def make_patched_conv2d_persistent(orig_conv2d, fallback_patched_conv2d):
    if getattr(orig_conv2d, '_wave3_dw2d_persistent_patched', False):
        return orig_conv2d

    import torch
    import triton
    from torch.nn import functional as F

    _ALLOWED_K = (3, 5, 7, 9, 11, 13, 15)

    @functools.wraps(orig_conv2d)
    def patched_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        sh, sw = _pair(stride)
        dh, dw = _pair(dilation)

        x_in = input
        if isinstance(padding, str):
            if padding == 'same':
                ih, iw = int(x_in.shape[-2]), int(x_in.shape[-1])
                kh, kw = int(weight.shape[-2]), int(weight.shape[-1])
                eff_kh = dh * (kh - 1) + 1
                eff_kw = dw * (kw - 1) + 1
                out_h = math.ceil(ih / sh)
                out_w = math.ceil(iw / sw)
                pad_h_total = max((out_h - 1) * sh + eff_kh - ih, 0)
                pad_w_total = max((out_w - 1) * sw + eff_kw - iw, 0)
                pad_top = pad_h_total // 2
                pad_bottom = pad_h_total - pad_top
                pad_left = pad_w_total // 2
                pad_right = pad_w_total - pad_left
                if pad_top == pad_bottom and pad_left == pad_right:
                    ph, pw = pad_top, pad_left
                else:
                    x_in = F.pad(x_in, (pad_left, pad_right, pad_top, pad_bottom))
                    ph, pw = 0, 0
            elif padding == 'valid':
                ph, pw = 0, 0
            else:
                raise ValueError(f"Unsupported padding string: {padding}, only 'valid'/'same' are allowed.")
        else:
            ph, pw = _pair(padding)

        if (
            isinstance(x_in, torch.Tensor)
            and isinstance(weight, torch.Tensor)
            and x_in.ndim == 4
            and weight.ndim == 4
            and x_in.device.type == 'npu'
            and torch.is_floating_point(x_in)
        ):
            n, cin, ih, iw = (int(v) for v in x_in.shape)
            cout, w_cin, kh, kw = (int(v) for v in weight.shape)
            eff_kh = dh * (kh - 1) + 1
            eff_kw = dw * (kw - 1) + 1
            oh = (ih + 2 * ph - eff_kh) // sh + 1
            ow = (iw + 2 * pw - eff_kw) // sw + 1

            if (
                groups == cin
                and cin > 1
                and cout % cin == 0
                and dh == 1
                and dw == 1
                and kh in _ALLOWED_K
                and kw in _ALLOWED_K
                and min(n, cin, oh, ow) > 0
            ):
                x = x_in.contiguous()
                w = weight.contiguous()
                y = torch.empty((n, cout, oh, ow), device=x.device, dtype=x.dtype)
                block = 256
                spatial_tiles = triton.cdiv(oh * ow, block)
                total_tiles = n * cout * spatial_tiles
                n_progs = min(total_tiles, 4096)
                bias_t = bias.contiguous() if bias is not None else w
                dw_kernel = _get_dw2d_persistent_kernel()
                dw_kernel[(n_progs,)](
                    x,
                    w,
                    y,
                    bias_t,
                    cout,
                    ih,
                    iw,
                    oh,
                    ow,
                    *x.stride(),
                    spatial_tiles,
                    total_tiles,
                    n_progs,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    cout // cin,
                    bias is not None,
                    block,
                )
                return y

        return fallback_patched_conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

    patched_conv2d._wave3_patched = True
    patched_conv2d._wave3_dw2d_persistent_patched = True
    return patched_conv2d
