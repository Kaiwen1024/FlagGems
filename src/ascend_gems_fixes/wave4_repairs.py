"""Wave 4 FlagGems pre-enable operator fixes (layered cleanly on top of wave3_repairs).
All fixes keep genuine FlagGems Triton kernels active."""
from __future__ import annotations

import functools
import importlib
import os
import sys


def make_patched_conv1d_m2(orig_conv1d):
    if getattr(orig_conv1d, '_wave4_m2_conv1d', False):
        return orig_conv1d
    from depthwise_npu import run as dw_run

    @functools.wraps(orig_conv1d)
    def patched_conv1d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if os.environ.get('GEMS_ASCEND_DEPTHWISE_1D') == '1' or os.environ.get('GEMS_M2_DEPTHWISE_FP32') == '1':
            res = dw_run(x, weight, bias, stride, padding, dilation, groups)
            if res is not None:
                return res
        return orig_conv1d(x, weight, bias, stride, padding, dilation, groups)

    patched_conv1d._wave4_m2_conv1d = True
    patched_conv1d._wave4_patched = True
    return patched_conv1d


def make_patched_pad_m2(base_pad_fn, has_mode_arg: bool = False):
    marker = '_wave4_pad_m2' if has_mode_arg else '_wave4_const_pad_m2'
    if getattr(base_pad_fn, marker, False):
        return base_pad_fn
    from ascend_fft_padding import run as fft_pad_run

    if has_mode_arg:
        @functools.wraps(base_pad_fn)
        def patched_pad(self, pad, mode="constant", value=None):
            if mode == "constant" and os.environ.get('GEMS_ASCEND_M2_FFT_PADDING') == '1' and not self.is_contiguous():
                res = fft_pad_run(self, pad, value)
                if res is not None:
                    return res
            return base_pad_fn(self, pad, mode=mode, value=value)
        fn = patched_pad
    else:
        @functools.wraps(base_pad_fn)
        def patched_const_pad(self, pad, value=0):
            if os.environ.get('GEMS_ASCEND_M2_FFT_PADDING') == '1' and not self.is_contiguous():
                res = fft_pad_run(self, pad, value)
                if res is not None:
                    return res
            return base_pad_fn(self, pad, value)
        fn = patched_const_pad

    setattr(fn, marker, True)
    fn._wave4_patched = True
    return fn


_FLIP_KERNELS = None


def _get_flip_kernels():
    global _FLIP_KERNELS
    if _FLIP_KERNELS is None:
        import triton
        import triton.language as tl
        from flag_gems.utils import libentry

        @libentry()
        @triton.jit
        def _flip_4d_kernel(
            X,
            Y,
            N,
            D0,
            D1,
            D2,
            D3,
            S0,
            S1,
            S2,
            S3,
            F0: tl.constexpr,
            F1: tl.constexpr,
            F2: tl.constexpr,
            F3: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            pid = tl.program_id(0).to(tl.int64)
            num_progs = tl.num_programs(0).to(tl.int64)
            step = num_progs * BLOCK
            offs_base = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
            num_iters = (N + step - 1) // step
            for it in range(num_iters):
                idx = offs_base + it * step
                mask = idx < N
                safe_idx = tl.where(mask, idx, 0)
                rem = safe_idx
                c3 = rem % D3
                rem = rem // D3
                c2 = rem % D2
                rem = rem // D2
                c1 = rem % D1
                rem = rem // D1
                c0 = rem
                x3 = (D3 - 1 - c3) if F3 else c3
                x2 = (D2 - 1 - c2) if F2 else c2
                x1 = (D1 - 1 - c1) if F1 else c1
                x0 = (D0 - 1 - c0) if F0 else c0
                in_off = x0 * S0 + x1 * S1 + x2 * S2 + x3 * S3
                val = tl.load(X + in_off, mask=mask, other=0)
                tl.store(Y + idx, val, mask=mask)

        @libentry()
        @triton.jit
        def _flip_8d_kernel(
            X,
            Y,
            N,
            D0,
            D1,
            D2,
            D3,
            D4,
            D5,
            D6,
            D7,
            S0,
            S1,
            S2,
            S3,
            S4,
            S5,
            S6,
            S7,
            F0: tl.constexpr,
            F1: tl.constexpr,
            F2: tl.constexpr,
            F3: tl.constexpr,
            F4: tl.constexpr,
            F5: tl.constexpr,
            F6: tl.constexpr,
            F7: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            pid = tl.program_id(0).to(tl.int64)
            num_progs = tl.num_programs(0).to(tl.int64)
            step = num_progs * BLOCK
            offs_base = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
            num_iters = (N + step - 1) // step
            for it in range(num_iters):
                idx = offs_base + it * step
                mask = idx < N
                safe_idx = tl.where(mask, idx, 0)
                rem = safe_idx
                c7 = rem % D7
                rem = rem // D7
                c6 = rem % D6
                rem = rem // D6
                c5 = rem % D5
                rem = rem // D5
                c4 = rem % D4
                rem = rem // D4
                c3 = rem % D3
                rem = rem // D3
                c2 = rem % D2
                rem = rem // D2
                c1 = rem % D1
                rem = rem // D1
                c0 = rem
                x7 = (D7 - 1 - c7) if F7 else c7
                x6 = (D6 - 1 - c6) if F6 else c6
                x5 = (D5 - 1 - c5) if F5 else c5
                x4 = (D4 - 1 - c4) if F4 else c4
                x3 = (D3 - 1 - c3) if F3 else c3
                x2 = (D2 - 1 - c2) if F2 else c2
                x1 = (D1 - 1 - c1) if F1 else c1
                x0 = (D0 - 1 - c0) if F0 else c0
                in_off = (
                    x0 * S0 + x1 * S1 + x2 * S2 + x3 * S3
                    + x4 * S4 + x5 * S5 + x6 * S6 + x7 * S7
                )
                val = tl.load(X + in_off, mask=mask, other=0)
                tl.store(Y + idx, val, mask=mask)

        _FLIP_KERNELS = (_flip_4d_kernel, _flip_8d_kernel)
    return _FLIP_KERNELS


def make_patched_flip(orig_flip):
    if getattr(orig_flip, '_wave4_flip', False):
        return orig_flip

    import torch
    import triton
    from flag_gems.runtime import torch_device_fn

    @functools.wraps(orig_flip)
    def patched_flip(A: torch.Tensor, dims):
        ndim = A.dim()
        flip_dims_b = [False] * ndim
        for dim in dims:
            assert (
                dim >= -ndim and dim < ndim
            ), "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                -ndim, ndim - 1, dim
            )
            assert not flip_dims_b[
                dim
            ], "dim {} appears multiple times in the list of dims".format(dim)
            flip_dims_b[dim] = True

        n = 0
        for i in range(ndim):
            if flip_dims_b[i] and A.size(i) > 1 and A.stride(i) != 0:
                n += 1
        if n == 0 or A.numel() <= 1:
            return A.clone()
        if A.device.type != 'npu' or ndim > 8:
            return orig_flip(A, dims)

        if A.dtype == torch.bool:
            return patched_flip(A.to(torch.uint8), dims).to(torch.bool)

        src = A.contiguous() if any(s < 0 for s in A.stride()) else A
        out = torch.empty(src.shape, dtype=src.dtype, device=src.device)
        numel = src.numel()
        block = 256
        grid = (min(triton.cdiv(numel, block), 48),)
        k4, k8 = _get_flip_kernels()

        with torch_device_fn.device(src.device):
            if ndim <= 4:
                pad = 4 - ndim
                ds = [1] * pad + list(src.shape)
                ss = [0] * pad + list(src.stride())
                fs = [False] * pad + flip_dims_b
                k4[grid](
                    src,
                    out,
                    numel,
                    ds[0], ds[1], ds[2], ds[3],
                    ss[0], ss[1], ss[2], ss[3],
                    fs[0], fs[1], fs[2], fs[3],
                    block,
                )
            else:
                pad = 8 - ndim
                ds = [1] * pad + list(src.shape)
                ss = [0] * pad + list(src.stride())
                fs = [False] * pad + flip_dims_b
                k8[grid](
                    src,
                    out,
                    numel,
                    ds[0], ds[1], ds[2], ds[3], ds[4], ds[5], ds[6], ds[7],
                    ss[0], ss[1], ss[2], ss[3], ss[4], ss[5], ss[6], ss[7],
                    fs[0], fs[1], fs[2], fs[3], fs[4], fs[5], fs[6], fs[7],
                    block,
                )
        return out

    patched_flip._wave4_flip = True
    patched_flip._wave4_patched = True
    return patched_flip


def make_patched_sdpa_math(orig_sdpa):
    if getattr(orig_sdpa, '_wave4_sdpa', False):
        return orig_sdpa

    import torch

    @functools.wraps(orig_sdpa)
    def patched_sdpa(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
    ):
        # Leave explicit GEMS_PACKED_SDPA wide unmasked 3D path to upstream helper
        if (
            os.environ.get('GEMS_PACKED_SDPA') == '1'
            and attn_mask is None
            and not is_causal
            and query.ndim == key.ndim == value.ndim == 3
            and query.shape[-1] > 256
        ):
            return orig_sdpa(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )

        squeezed = False
        if query.ndim == 3 and key.ndim == 3 and value.ndim == 3:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            if attn_mask is not None:
                if attn_mask.ndim == 2:
                    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                elif attn_mask.ndim == 3:
                    attn_mask = attn_mask.unsqueeze(0)
            squeezed = True

        if attn_mask is not None and attn_mask.dtype == torch.bool:
            mask_dtype = (
                query.dtype
                if query.dtype in (torch.float16, torch.bfloat16, torch.float32)
                else torch.float32
            )
            mask_f = torch.zeros(attn_mask.shape, dtype=mask_dtype, device=attn_mask.device)
            attn_mask = mask_f.masked_fill(~attn_mask, float('-inf'))

        out, weights = orig_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )
        if squeezed:
            return out.squeeze(0), weights.squeeze(0)
        return out, weights

    patched_sdpa._wave4_sdpa = True
    patched_sdpa._wave4_patched = True
    return patched_sdpa


def install_wave4_pre_enable_fixes(repairs_log: list[str]) -> dict:
    """Install Wave 3 fixes plus Wave 4 additions BEFORE flag_gems.enable()."""
    import flag_gems  # noqa: F401
    from flag_gems.runtime.op_registrar import GeneralOpRegistrar
    from wave3_repairs import install_pre_enable_fixes

    meta = install_pre_enable_fixes(repairs_log)

    # Also patch conv1d, flip, and _scaled_dot_product_attention_math modules directly for internal callers
    try:
        importlib.import_module('flag_gems.ops.conv1d')
        c1d_mod = sys.modules['flag_gems.ops.conv1d']
        if hasattr(c1d_mod, 'conv1d') and not getattr(c1d_mod.conv1d, '_wave4_m2_conv1d', False):
            c1d_mod.conv1d = make_patched_conv1d_m2(c1d_mod.conv1d)
    except Exception:
        pass
    for flip_mod_name in ('flag_gems.runtime.backend._ascend.ops.flip', 'flag_gems.ops.flip'):
        try:
            importlib.import_module(flip_mod_name)
            f_mod = sys.modules[flip_mod_name]
            if hasattr(f_mod, 'flip') and not getattr(f_mod.flip, '_wave4_flip', False):
                f_mod.flip = make_patched_flip(f_mod.flip)
        except Exception:
            pass
    try:
        importlib.import_module('flag_gems.ops._scaled_dot_product_attention_math')
        sdpa_mod = sys.modules['flag_gems.ops._scaled_dot_product_attention_math']
        if (
            hasattr(sdpa_mod, '_scaled_dot_product_attention_math')
            and not getattr(sdpa_mod._scaled_dot_product_attention_math, '_wave4_sdpa', False)
        ):
            sdpa_mod._scaled_dot_product_attention_math = make_patched_sdpa_math(
                sdpa_mod._scaled_dot_product_attention_math
            )
    except Exception:
        pass

    def _wrap_w4(k_str: str, fn):
        if k_str in ('conv1d', 'conv1d.Tensor', 'conv1d.padding') and not getattr(fn, '_wave4_m2_conv1d', False):
            return make_patched_conv1d_m2(fn)
        if k_str == 'constant_pad_nd' and not getattr(fn, '_wave4_const_pad_m2', False):
            return make_patched_pad_m2(fn, has_mode_arg=False)
        if k_str == 'pad' and not getattr(fn, '_wave4_pad_m2', False):
            return make_patched_pad_m2(fn, has_mode_arg=True)
        if k_str == 'flip' and not getattr(fn, '_wave4_flip', False):
            return make_patched_flip(fn)
        if k_str == '_scaled_dot_product_attention_math' and not getattr(fn, '_wave4_sdpa', False):
            return make_patched_sdpa_math(fn)
        return fn

    w4_patched = []
    if hasattr(flag_gems, '_FULL_CONFIG'):
        new_cfg = []
        for item in flag_gems._FULL_CONFIG:
            if item and len(item) >= 2:
                k_str = str(item[0])
                wfn = _wrap_w4(k_str, item[1])
                if wfn is not item[1]:
                    item = (item[0], wfn, *item[2:])
                    w4_patched.append(k_str)
            new_cfg.append(item)
        flag_gems._FULL_CONFIG = tuple(new_cfg)
        if hasattr(flag_gems, 'FULL_CONFIG_BY_FUNC'):
            by_fn = {}
            for _it in flag_gems._FULL_CONFIG:
                if _it and len(_it) >= 2:
                    _fn = _it[1]
                    _name = _fn.__name__ if hasattr(_fn, '__name__') else str(_fn)
                    by_fn.setdefault(_name, []).append(_it)
            flag_gems.FULL_CONFIG_BY_FUNC = by_fn
    meta['wave4_patched_config'] = w4_patched

    orig_reg_impl = GeneralOpRegistrar.register_impl
    if not getattr(GeneralOpRegistrar, '_wave4_hooked', False):
        def w4_register_impl(self, key, fn, extra_dispatch_keys=()):
            new_fn = _wrap_w4(str(key), fn)
            return orig_reg_impl(self, key, new_fn, extra_dispatch_keys)
        GeneralOpRegistrar.register_impl = w4_register_impl
        GeneralOpRegistrar._wave4_hooked = True

    repairs_log.append(
        'Wave 4 hooks: wrap M2-BERT depthwise 3-tap FP32 conv1d (depthwise_npu), non-contiguous FP32 constant_pad_nd/pad (ascend_fft_padding), persistent 1D-grid 64-bit ND Ascend Triton flip, and 3D+boolean-mask _scaled_dot_product_attention_math cleanly via FlagGems registrar.'
    )
    return meta
