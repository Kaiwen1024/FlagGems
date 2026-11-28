"""Process-local verified FlagGems operator & offline-hub repairs for Wave 3 (45 multimodal/vision models).
All operator repairs keep genuine FlagGems Triton kernels active."""
from __future__ import annotations

from collections import namedtuple
import functools
import importlib
import math
import sys

_DW2D_KERNEL = None
_MAX_DIM_KERNELS = None
_MASKED_SCATTER_KERNELS = None
_MAX_OUT_TUPLE = namedtuple("max", ["values", "indices"])


def _get_dw2d_kernel():
    global _DW2D_KERNEL
    if _DW2D_KERNEL is None:
        import triton
        import triton.language as tl
        from flag_gems.utils import libentry

        @libentry()
        @triton.jit
        def _dw2d_kernel(
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
            K: tl.constexpr,
            STEP: tl.constexpr,
            PH: tl.constexpr,
            PW: tl.constexpr,
            MULT: tl.constexpr,
            HAS_BIAS: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            ch = tl.program_id(1)
            b = tl.program_id(2)
            oh = i // OW
            ow = i % OW
            acc = tl.full((BLOCK,), 0, tl.float32)
            for ky in tl.static_range(K):
                for kx in tl.static_range(K):
                    iy = oh * STEP - PH + ky
                    ix = ow * STEP - PW + kx
                    active = (i < OH * OW) & (iy >= 0) & (ix >= 0) & (iy < H) & (ix < IW)
                    val = tl.load(X + b * SN + (ch // MULT) * SC + iy * SH + ix * SW, active, other=0).to(tl.float32)
                    coeff = tl.load(W + ch * K * K + ky * K + kx).to(tl.float32)
                    acc = tl.fma(val, coeff, acc)
            if HAS_BIAS:
                acc += tl.load(BIAS + ch).to(tl.float32)
            tl.store(Y + ((b * CO + ch) * OH * OW) + i, acc, i < OH * OW)

        _DW2D_KERNEL = _dw2d_kernel
    return _DW2D_KERNEL


def make_patched_conv2d(orig_conv2d):
    if getattr(orig_conv2d, '_wave3_patched', False):
        return orig_conv2d

    import torch
    import triton
    from torch.nn import functional as F

    importlib.import_module('flag_gems.ops.conv2d')
    conv_mod = sys.modules['flag_gems.ops.conv2d']
    Conv2d = conv_mod.Conv2d

    @functools.wraps(orig_conv2d)
    def patched_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        sh, sw = (int(stride), int(stride)) if isinstance(stride, int) else (int(stride[0]), int(stride[1]))
        dh, dw = (int(dilation), int(dilation)) if isinstance(dilation, int) else (int(dilation[0]), int(dilation[1]))

        if isinstance(padding, str):
            if padding == 'same':
                ih, iw = int(input.shape[-2]), int(input.shape[-1])
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
                    input = F.pad(input, (pad_left, pad_right, pad_top, pad_bottom))
                    ph, pw = 0, 0
            elif padding == 'valid':
                ph, pw = 0, 0
            else:
                raise ValueError(f"Unsupported padding string: {padding}, only 'valid'/'same' are allowed.")
        elif isinstance(padding, int):
            ph, pw = int(padding), int(padding)
        else:
            ph, pw = int(padding[0]), int(padding[1])

        if (
            isinstance(input, torch.Tensor)
            and isinstance(weight, torch.Tensor)
            and input.ndim == 4
            and weight.ndim == 4
            and input.device.type == 'npu'
            and torch.is_floating_point(input)
        ):
            n, cin, ih, iw = (int(v) for v in input.shape)
            cout, w_cin, kh, kw = (int(v) for v in weight.shape)
            eff_kh = dh * (kh - 1) + 1
            eff_kw = dw * (kw - 1) + 1
            oh = (ih + 2 * ph - eff_kh) // sh + 1
            ow = (iw + 2 * pw - eff_kw) // sw + 1

            # Case 1: Depthwise 2D convolution via Ascend Triton kernel (require kh == kw > 1 so conv1d kw==1 is excluded)
            if (
                groups == cin
                and cin > 1
                and cout % cin == 0
                and dh == 1
                and dw == 1
                and kh == kw
                and kh in (3, 5, 7, 9, 11, 13, 15)
                and sh == sw
                and min(n, cin, oh, ow) > 0
            ):
                x = input.contiguous()
                w = weight.contiguous()
                y = torch.empty((n, cout, oh, ow), device=x.device, dtype=x.dtype)
                if max(x.numel(), y.numel()) < 2**31 - 2048:
                    bias_t = bias.contiguous() if bias is not None else w
                    dw_kernel = _get_dw2d_kernel()
                    dw_kernel[(triton.cdiv(oh * ow, 256), cout, n)](
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
                        kh,
                        sh,
                        ph,
                        pw,
                        cout // cin,
                        bias is not None,
                        256,
                    )
                    return y

            # Case 2: Pointwise 1x1 convolution lowered to FlagGems Triton linear/mm
            if groups == 1 and kh == 1 and kw == 1 and ph == 0 and pw == 0 and min(n, cin, oh, ow) > 0:
                val = (input[:, :, ::sh, ::sw] if (sh > 1 or sw > 1) else input).permute(0, 2, 3, 1).contiguous()
                y = F.linear(val.reshape(n * oh * ow, cin), weight.reshape(cout, cin), bias)
                return y.reshape(n, oh, ow, cout).permute(0, 3, 1, 2).contiguous()

            # Case 3: Non-overlapping patch / ConvNeXt downsample convolution lowered to FlagGems linear/mm
            if (
                groups == 1
                and ph == 0
                and pw == 0
                and dh == 1
                and dw == 1
                and kh == sh
                and kw == sw
                and ih == oh * kh
                and iw == ow * kw
                and min(n, cin, oh, ow) > 0
            ):
                val = (
                    input.reshape(n, cin, oh, kh, ow, kw)
                    .permute(0, 2, 4, 1, 3, 5)
                    .contiguous()
                    .reshape(n * oh * ow, cin * kh * kw)
                )
                y = F.linear(val, weight.reshape(cout, cin * kh * kw), bias)
                return y.reshape(n, oh, ow, cout).permute(0, 3, 1, 2).contiguous()

            # Case 4: Small-channel stem convolution (e.g. EfficientNet RGB 3x3 stride 2) lowered to FlagGems linear/mm
            if groups == 1 and cin <= 4 and dh == 1 and dw == 1 and kh * kw <= 25 and min(n, cin, oh, ow) > 0:
                padded = F.pad(input, (pw, pw, ph, ph)) if (ph or pw) else input
                vals = torch.stack(
                    [
                        padded[:, :, ky : ky + oh * sh : sh, kx : kx + ow * sw : sw]
                        for ky in range(kh)
                        for kx in range(kw)
                    ],
                    dim=-1,
                )
                vals = vals.permute(0, 2, 3, 1, 4).reshape(n * oh * ow, cin * kh * kw).contiguous()
                y = F.linear(vals, weight.reshape(cout, cin * kh * kw), bias)
                return y.reshape(n, oh, ow, cout).permute(0, 3, 1, 2).contiguous()

        return Conv2d.apply(input, weight, bias, (sh, sw), (ph, pw), (dh, dw), groups)

    patched_conv2d._wave3_patched = True
    return patched_conv2d


def make_patched_conv3d(orig_conv3d):
    if getattr(orig_conv3d, '_wave3_patched', False):
        return orig_conv3d

    import torch
    from torch.nn import functional as F

    def _triple(v):
        if isinstance(v, int):
            iv = int(v)
            return (iv, iv, iv)
        vals = tuple(int(x) for x in v)
        if len(vals) == 1:
            return (vals[0], vals[0], vals[0])
        return (vals[0], vals[1], vals[2])

    @functools.wraps(orig_conv3d)
    def patched_conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        sd, sh, sw = _triple(stride)
        dd, dh, dw = _triple(dilation)
        if isinstance(padding, str):
            is_zero_pad = padding == 'valid'
        elif isinstance(padding, int):
            is_zero_pad = int(padding) == 0
        else:
            pd, ph, pw = _triple(padding)
            is_zero_pad = pd == 0 and ph == 0 and pw == 0

        if (
            is_zero_pad
            and int(groups) == 1
            and (dd, dh, dw) == (1, 1, 1)
            and isinstance(input, torch.Tensor)
            and isinstance(weight, torch.Tensor)
            and input.ndim == 5
            and weight.ndim == 5
            and input.device.type == 'npu'
            and torch.is_floating_point(input)
        ):
            n, cin, id_, ih, iw = (int(v) for v in input.shape)
            cout, w_cin, kd, kh, kw = (int(v) for v in weight.shape)
            if w_cin == cin and min(n, cin, cout, id_, ih, iw, kd, kh, kw) > 0:
                if id_ == kd and ih == kh and iw == kw:
                    x2d = input.contiguous().reshape(n, cin * kd * kh * kw)
                    w2d = weight.contiguous().reshape(cout, cin * kd * kh * kw)
                    y = F.linear(x2d, w2d, bias)
                    return y.reshape(n, cout, 1, 1, 1)
                if (
                    kd == sd
                    and kh == sh
                    and kw == sw
                    and id_ % kd == 0
                    and ih % kh == 0
                    and iw % kw == 0
                ):
                    od, oh, ow = id_ // kd, ih // kh, iw // kw
                    x = (
                        input.contiguous()
                        .reshape(n, cin, od, kd, oh, kh, ow, kw)
                        .permute(0, 2, 4, 6, 1, 3, 5, 7)
                        .contiguous()
                        .reshape(n * od * oh * ow, cin * kd * kh * kw)
                    )
                    w2d = weight.contiguous().reshape(cout, cin * kd * kh * kw)
                    y = F.linear(x, w2d, bias)
                    return y.reshape(n, od, oh, ow, cout).permute(0, 4, 1, 2, 3).contiguous()

        return orig_conv3d(input, weight, bias, stride, padding, dilation, groups)

    patched_conv3d._wave3_patched = True
    return patched_conv3d


def make_patched_sum(orig_sum_fn):
    if getattr(orig_sum_fn, '_wave3_patched', False):
        return orig_sum_fn
    import torch

    @functools.wraps(orig_sum_fn)
    def patched_sum(inp, *args, **kwargs):
        if isinstance(inp, torch.Tensor) and inp.dtype in (torch.bool, torch.uint8, torch.int8):
            target_dtype = kwargs.get('dtype') or torch.int64
            inp = inp.to(target_dtype)
        return orig_sum_fn(inp, *args, **kwargs)

    patched_sum._wave3_patched = True
    return patched_sum


def make_patched_mul(orig_mul):
    if getattr(orig_mul, '_wave3_patched', False):
        return orig_mul
    import torch

    @functools.wraps(orig_mul)
    def patched_mul(a, b):
        if (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.device.type == 'npu'
            and b.device.type == 'npu'
            and a.ndim == 4
            and b.ndim == 4
            and a.shape[:2] == b.shape[:2]
            and a.is_contiguous()
            and b.is_contiguous()
        ):
            if b.shape[2] == 1 and b.shape[3] == 1 and a.shape[2] * a.shape[3] >= 16:
                n, c, h, w = a.shape
                return orig_mul(a.view(n * c, h * w), b.view(n * c, 1)).view(n, c, h, w)
            if a.shape[2] == 1 and a.shape[3] == 1 and b.shape[2] * b.shape[3] >= 16:
                n, c, h, w = b.shape
                return orig_mul(b.view(n * c, h * w), a.view(n * c, 1)).view(n, c, h, w)
        return orig_mul(a, b)

    patched_mul._wave3_patched = True
    return patched_mul


def make_patched_arange(orig_fn):
    if getattr(orig_fn, '_wave3_patched', False):
        return orig_fn
    import torch

    @functools.wraps(orig_fn)
    def patched_arange(*args, **kwargs):
        if kwargs.get('dtype') is None:
            vals = list(args) + [kwargs.get(k) for k in ('start', 'end', 'step') if k in kwargs]
            has_float = any(
                isinstance(x, float) or (isinstance(x, torch.Tensor) and torch.is_floating_point(x))
                for x in vals
            )
            if has_float:
                kwargs = dict(kwargs)
                kwargs['dtype'] = torch.get_default_dtype()
        return orig_fn(*args, **kwargs)

    patched_arange._wave3_patched = True
    return patched_arange


def _get_max_dim_kernels():
    global _MAX_DIM_KERNELS
    if _MAX_DIM_KERNELS is None:
        import triton
        import triton.language as tl
        from flag_gems.utils import libentry
        from flag_gems.utils.limits import get_dtype_min

        @libentry()
        @triton.jit
        def _max_dim_strided_k_kernel(
            inp,
            out_val,
            out_idx,
            M,
            N,
            K,
            k_tiles,
            total_tiles,
            n_progs,
            IS_FLOAT: tl.constexpr,
            BLOCK_K: tl.constexpr,
        ):
            pid = tl.program_id(0)
            dtype = inp.type.element_ty
            acc_type = tl.float32 if (dtype is tl.bfloat16 or dtype is tl.float16) else dtype
            min_val = get_dtype_min(dtype)
            zero64 = tl.zeros([], dtype=tl.int64)
            stride_n = zero64 + K
            stride_m = (zero64 + N) * stride_n

            for tile_id in range(pid, total_tiles, n_progs):
                m = tile_id // k_tiles
                kt = tile_id % k_tiles
                k_off = kt * BLOCK_K + tl.arange(0, BLOCK_K)
                k_mask = k_off < K
                k64 = k_off.to(tl.int64)
                base = (zero64 + m) * stride_m + k64

                best_val = tl.load(inp + base, mask=k_mask, other=min_val).to(acc_type)
                best_idx = tl.zeros([BLOCK_K], dtype=tl.int64)

                cur_off = base
                for n in range(1, N):
                    cur_off += stride_n
                    cur = tl.load(inp + cur_off, mask=k_mask, other=min_val).to(acc_type)
                    if IS_FLOAT:
                        better = (cur > best_val) | ((cur != cur) & (best_val == best_val))
                    else:
                        better = cur > best_val
                    best_val = tl.where(better, cur, best_val)
                    best_idx = tl.where(better, tl.zeros([BLOCK_K], dtype=tl.int64) + n, best_idx)

                out_off = (zero64 + m) * stride_n + k64
                tl.store(out_val + out_off, best_val.to(dtype), mask=k_mask)
                tl.store(out_idx + out_off, best_idx, mask=k_mask)

        @libentry()
        @triton.jit
        def _max_dim_inner_kernel(
            inp,
            out_val,
            out_idx,
            M,
            N,
            n_progs,
            IS_FLOAT: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
        ):
            pid = tl.program_id(0)
            dtype = inp.type.element_ty
            acc_type = tl.float32 if (dtype is tl.bfloat16 or dtype is tl.float16) else dtype
            min_val = get_dtype_min(dtype)
            zero64 = tl.zeros([], dtype=tl.int64)
            stride_m = zero64 + N

            for m_base in range(pid * BLOCK_M, M, n_progs * BLOCK_M):
                m_off = m_base + tl.arange(0, BLOCK_M)
                row_mask = m_off < M
                m64 = m_off[:, None].to(tl.int64) * stride_m
                result_value = tl.full([BLOCK_M], value=min_val, dtype=acc_type)
                result_index = tl.zeros([BLOCK_M], dtype=tl.int64)
                first_chunk = tl.full([BLOCK_M], value=1, dtype=tl.int1)

                for i in range(0, N, BLOCK_N):
                    n_off = i + tl.arange(0, BLOCK_N)
                    col_mask = n_off < N
                    mask = row_mask[:, None] & col_mask[None, :]
                    offs = m64 + n_off[None, :].to(tl.int64)
                    inp_vals = tl.load(inp + offs, mask=mask, other=min_val).to(acc_type)
                    inp_vals = tl.where(mask, inp_vals, min_val)
                    max_val, max_idx = tl.max(inp_vals, axis=1, return_indices=True)
                    if IS_FLOAT:
                        better = first_chunk | (max_val > result_value) | ((max_val != max_val) & (result_value == result_value))
                    else:
                        better = first_chunk | (max_val > result_value)
                    result_value = tl.where(better, max_val, result_value)
                    result_index = tl.where(better, zero64 + i + max_idx.to(tl.int64), result_index)
                    first_chunk = tl.zeros([BLOCK_M], dtype=tl.int1)

                out_off = m_off.to(tl.int64)
                tl.store(out_val + out_off, result_value.to(dtype), mask=row_mask)
                tl.store(out_idx + out_off, result_index, mask=row_mask)

        _MAX_DIM_KERNELS = (_max_dim_strided_k_kernel, _max_dim_inner_kernel)
    return _MAX_DIM_KERNELS


def make_patched_max_dim(orig_max_dim):
    if getattr(orig_max_dim, '_wave3_patched', False):
        return orig_max_dim

    import torch
    import triton
    from flag_gems.runtime import torch_device_fn

    @functools.wraps(orig_max_dim)
    def patched_max_dim(inp, dim=None, keepdim=False):
        if dim is None or not isinstance(inp, torch.Tensor) or inp.device.type != 'npu':
            return orig_max_dim(inp, dim=dim, keepdim=keepdim)

        if inp.ndim == 0:
            assert -1 <= dim <= 0, 'Invalid dim'
            out_val = inp.clone()
            out_idx = torch.zeros((), dtype=torch.int64, device=inp.device)
            return _MAX_OUT_TUPLE(values=out_val, indices=out_idx)

        assert -inp.ndim <= dim < inp.ndim, 'Invalid dim'
        shape = list(inp.shape)
        dim_idx = int(dim) % inp.ndim
        N = int(shape[dim_idx])
        if N == 0:
            return orig_max_dim(inp, dim=dim, keepdim=keepdim)

        out_shape = list(shape)
        out_shape[dim_idx] = 1
        orig_dtype = inp.dtype

        if inp.numel() == 0:
            out_value = torch.empty(out_shape, dtype=orig_dtype, device=inp.device)
            out_index = torch.empty(out_shape, dtype=torch.int64, device=inp.device)
            if not keepdim:
                out_value = torch.squeeze(out_value, dim_idx)
                out_index = torch.squeeze(out_index, dim_idx)
            return _MAX_OUT_TUPLE(values=out_value, indices=out_index)

        work_dtype = torch.int32 if orig_dtype in (torch.bool, torch.int8, torch.uint8, torch.int16) else orig_dtype
        work_inp = inp.to(work_dtype) if orig_dtype != work_dtype else inp
        work_inp = work_inp.contiguous()

        M = math.prod(shape[:dim_idx]) if dim_idx > 0 else 1
        K = work_inp.numel() // (M * N)

        out_value = torch.empty(out_shape, dtype=work_dtype, device=inp.device)
        out_index = torch.empty(out_shape, dtype=torch.int64, device=inp.device)

        strided_k_kernel, inner_kernel = _get_max_dim_kernels()
        is_float = work_inp.is_floating_point()

        with torch_device_fn.device(inp.device):
            if K == 1:
                block_m = 4
                block_n = 256
                n_progs = min(triton.cdiv(M, block_m), 2048)
                inner_kernel[(n_progs,)](
                    work_inp,
                    out_value,
                    out_index,
                    M,
                    N,
                    n_progs,
                    IS_FLOAT=is_float,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                )
            else:
                block_k = 256
                k_tiles = triton.cdiv(K, block_k)
                total_tiles = M * k_tiles
                n_progs = min(total_tiles, 2048)
                strided_k_kernel[(n_progs,)](
                    work_inp,
                    out_value,
                    out_index,
                    M,
                    N,
                    K,
                    k_tiles,
                    total_tiles,
                    n_progs,
                    IS_FLOAT=is_float,
                    BLOCK_K=block_k,
                )

        if orig_dtype != work_dtype:
            out_value = out_value.to(orig_dtype)
        if not keepdim:
            out_value = torch.squeeze(out_value, dim_idx)
            out_index = torch.squeeze(out_index, dim_idx)
        return _MAX_OUT_TUPLE(values=out_value, indices=out_index)

    patched_max_dim._wave3_patched = True
    return patched_max_dim


def _get_masked_scatter_kernels():
    global _MASKED_SCATTER_KERNELS
    if _MASKED_SCATTER_KERNELS is None:
        import triton
        import triton.language as tl
        from flag_gems.utils import libentry

        @libentry()
        @triton.jit
        def _ms_count_blocks_kernel(
            mask_ptr,
            counts_ptr,
            N,
            n_blocks,
            n_progs,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            zero64 = tl.zeros([], dtype=tl.int64)
            for b in range(pid, n_blocks, n_progs):
                off = (zero64 + b) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
                m = off < N
                sel = tl.load(mask_ptr + off, mask=m, other=0).to(tl.int1)
                cnt = tl.sum(tl.where(m & sel, 1, 0).to(tl.int32), axis=0).to(tl.int64)
                tl.store(
                    counts_ptr + (zero64 + b) + tl.arange(0, 1),
                    tl.full([1], cnt, dtype=tl.int64),
                )

        @libentry()
        @triton.jit
        def _ms_build_row_map_kernel(
            mask_ptr,
            part_sums_ptr,
            row_src_ptr,
            R,
            n_blocks,
            n_progs,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            zero64 = tl.zeros([], dtype=tl.int64)
            for b in range(pid, n_blocks, n_progs):
                off = (zero64 + b) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
                m = off < R
                sel = tl.load(mask_ptr + off, mask=m, other=0).to(tl.int1)
                active = m & sel
                sel_i = tl.where(active, 1, 0).to(tl.int32)
                adv = tl.sum(tl.load(part_sums_ptr + (zero64 + b) + tl.arange(0, 1)).to(tl.int64), axis=0)
                c = adv + tl.cumsum(sel_i, axis=0).to(tl.int64) - 1
                row_map = tl.where(active, c, -1)
                tl.store(row_src_ptr + off, row_map, mask=m)

        @libentry()
        @triton.jit
        def _ms_copy_rows_kernel(
            out_ptr,
            src_ptr,
            row_src_ptr,
            R,
            D,
            n_progs,
            BLOCK_D: tl.constexpr,
        ):
            pid = tl.program_id(0)
            zero64 = tl.zeros([], dtype=tl.int64)
            d64 = zero64 + D
            for r in range(pid, R, n_progs):
                r64 = zero64 + r
                src_r = tl.sum(tl.load(row_src_ptr + r64 + tl.arange(0, 1)).to(tl.int64), axis=0)
                if src_r >= 0:
                    src_base = src_r * d64
                    out_base = r64 * d64
                    for d0 in range(0, D, BLOCK_D):
                        d_off = (zero64 + d0) + tl.arange(0, BLOCK_D).to(tl.int64)
                        d_m = d_off < D
                        vals = tl.load(src_ptr + src_base + d_off, mask=d_m, other=0)
                        tl.store(out_ptr + out_base + d_off, vals, mask=d_m)

        @libentry()
        @triton.jit
        def _ms_scatter_elem_kernel(
            out_ptr,
            mask_ptr,
            src_ptr,
            part_sums_ptr,
            N,
            n_blocks,
            n_progs,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            zero64 = tl.zeros([], dtype=tl.int64)
            for b in range(pid, n_blocks, n_progs):
                off = (zero64 + b) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
                m = off < N
                sel = tl.load(mask_ptr + off, mask=m, other=0).to(tl.int1)
                active = m & sel
                sel_i = tl.where(active, 1, 0).to(tl.int32)
                cnt = tl.sum(sel_i, axis=0)
                if cnt > 0:
                    adv = tl.sum(tl.load(part_sums_ptr + (zero64 + b) + tl.arange(0, 1)).to(tl.int64), axis=0)
                    global_idx = adv + tl.cumsum(sel_i, axis=0).to(tl.int64) - 1
                    safe_idx = tl.where(active, global_idx, 0)
                    vals = tl.load(src_ptr + safe_idx, mask=active, other=0)
                    tl.store(out_ptr + off, vals, mask=active)

        _MASKED_SCATTER_KERNELS = (
            _ms_count_blocks_kernel,
            _ms_build_row_map_kernel,
            _ms_copy_rows_kernel,
            _ms_scatter_elem_kernel,
        )
    return _MASKED_SCATTER_KERNELS


def _masked_scatter_core(out, mask, source):
    """In-place verified Ascend Triton masked_scatter on contiguous `out`."""
    import torch
    import triton
    from flag_gems.utils import broadcastable, broadcastable_to

    if not broadcastable(out.shape, mask.shape):
        raise RuntimeError(
            f"The shapes of mask {tuple(mask.shape)} and input {tuple(out.shape)} are not broadcastable"
        )
    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)

    N = out.numel()
    if N == 0:
        return out
    if out.ndim == 0:
        if mask.numel() == 1 and bool(mask.item()):
            if source.numel() < 1:
                raise RuntimeError("Number of elements of source < number of ones in mask")
            out.fill_(source.reshape(-1)[0].to(out.dtype))
        return out

    if mask.ndim <= out.ndim and broadcastable_to(mask.shape, out.shape):
        bmask = mask.expand(out.shape)
    else:
        _, bmask = torch.broadcast_tensors(out, mask)
    split = bmask.ndim
    while split > 0 and (bmask.stride(split - 1) == 0 or bmask.shape[split - 1] == 1):
        split -= 1
    D = math.prod(bmask.shape[split:]) if split < bmask.ndim else 1
    R = N // D

    count_k, row_map_k, copy_rows_k, scatter_elem_k = _get_masked_scatter_kernels()
    BLOCK_SIZE = 1024

    if 32 <= D <= 8192 and R >= 1:
        row_idx = (slice(None),) * split + (0,) * (bmask.ndim - split)
        row_mask = bmask[row_idx].reshape(R).contiguous()
        n_blocks = triton.cdiv(R, BLOCK_SIZE)
        block_counts = torch.empty(n_blocks, dtype=torch.int64, device=out.device)
        n_progs = min(n_blocks, 2048)
        count_k[(n_progs,)](row_mask, block_counts, R, n_blocks, n_progs, BLOCK_SIZE=BLOCK_SIZE)

        counts_cpu = block_counts.cpu().to(torch.int64)
        true_rows = int(counts_cpu.sum().item())
        if true_rows == 0:
            return out
        need = true_rows * D
        if source.numel() < need:
            raise RuntimeError(f"masked_scatter: source numel {source.numel()} < mask ones {need}")

        prefix_sum = torch.zeros(n_blocks, dtype=torch.int64)
        if n_blocks > 1:
            torch.cumsum(counts_cpu[:-1], dim=0, out=prefix_sum[1:])
        part_sums = prefix_sum.to(out.device)

        row_src = torch.empty(R, dtype=torch.int64, device=out.device)
        row_map_k[(n_progs,)](row_mask, part_sums, row_src, R, n_blocks, n_progs, BLOCK_SIZE=BLOCK_SIZE)

        src_flat = source.reshape(-1)[:need]
        if not src_flat.is_contiguous():
            src_flat = src_flat.contiguous()
        if src_flat.dtype != out.dtype:
            src_flat = src_flat.to(out.dtype)

        copy_progs = min(R, 2048)
        copy_rows_k[(copy_progs,)](out, src_flat, row_src, R, D, copy_progs, BLOCK_D=1024)
        return out

    mask_flat = bmask if bmask.is_contiguous() else bmask.contiguous()
    n_blocks = triton.cdiv(N, BLOCK_SIZE)
    block_counts = torch.empty(n_blocks, dtype=torch.int64, device=out.device)
    n_progs = min(n_blocks, 2048)
    count_k[(n_progs,)](mask_flat, block_counts, N, n_blocks, n_progs, BLOCK_SIZE=BLOCK_SIZE)

    counts_cpu = block_counts.cpu().to(torch.int64)
    true_count = int(counts_cpu.sum().item())
    if true_count == 0:
        return out
    if source.numel() < true_count:
        raise RuntimeError(f"masked_scatter: source numel {source.numel()} < mask ones {true_count}")

    prefix_sum = torch.zeros(n_blocks, dtype=torch.int64)
    if n_blocks > 1:
        torch.cumsum(counts_cpu[:-1], dim=0, out=prefix_sum[1:])
    part_sums = prefix_sum.to(out.device)

    src_flat = source.reshape(-1)[:true_count]
    if not src_flat.is_contiguous():
        src_flat = src_flat.contiguous()
    if src_flat.dtype != out.dtype:
        src_flat = src_flat.to(out.dtype)

    scatter_elem_k[(n_progs,)](out, mask_flat, src_flat, part_sums, N, n_blocks, n_progs, BLOCK_SIZE=BLOCK_SIZE)
    return out


def make_patched_masked_scatter(orig_fn, inplace: bool = False):
    if getattr(orig_fn, '_wave3_patched', False):
        return orig_fn

    @functools.wraps(orig_fn)
    def patched_masked_scatter(inp, mask, source):
        if inplace:
            if inp.is_contiguous():
                return _masked_scatter_core(inp, mask, source)
            tmp = inp.contiguous()
            _masked_scatter_core(tmp, mask, source)
            inp.copy_(tmp)
            return inp
        out = inp.clone()
        if not out.is_contiguous():
            out = out.contiguous()
        return _masked_scatter_core(out, mask, source)

    patched_masked_scatter._wave3_patched = True
    return patched_masked_scatter


def install_pre_enable_fixes(repairs_log: list[str]) -> dict:
    """Install registrar-level hooks BEFORE flag_gems.enable() so CountingRegistrar wraps the fixed ops!"""
    import flag_gems  # noqa: F401
    from flag_gems.runtime.op_registrar import GeneralOpRegistrar

    meta = {}

    # 0. Fix O(N_tables) SQLAlchemy full-DB reflection in FlagGems SQLPersistantModel.build_sql_model_by_db
    try:
        import sqlalchemy
        import sqlalchemy.ext.automap
        from flag_gems.utils.models.sql import SQLPersistantModel

        if not getattr(SQLPersistantModel, '_wave3_fast_reflect', False):
            def _fast_build_sql_model_by_db(name: str, engine):
                if not sqlalchemy.inspect(engine).has_table(name):
                    return None
                md = sqlalchemy.MetaData()
                md.reflect(bind=engine, only=[name])
                auto_base = sqlalchemy.ext.automap.automap_base(metadata=md)
                auto_base.prepare()
                return auto_base.classes.get(name)

            SQLPersistantModel.build_sql_model_by_db = staticmethod(_fast_build_sql_model_by_db)
            SQLPersistantModel._wave3_fast_reflect = True
            meta['sql_single_table_reflect'] = True
    except Exception:
        pass

    # 1. Bound Philox rand_kernel BLOCK=256, num_stages=1 on Ascend 910 (fixes c045 UB overflow)
    importlib.import_module('flag_gems.ops.rand')
    rand_mod = sys.modules['flag_gems.ops.rand']
    rk = rand_mod.rand_kernel
    if hasattr(rk, 'values') and isinstance(rk.values, dict):
        rk.values['BLOCK'] = lambda args: 256
        rk.values['num_warps'] = lambda args: 4
        rk.values['num_stages'] = lambda args: 1
        meta['rand_kernel_ub'] = {'BLOCK': 256, 'num_warps': 4, 'num_stages': 1}

    # 2. Patch conv_mod.conv2d and conv3d_mod.conv3d directly so internal module callers use them too
    importlib.import_module('flag_gems.ops.conv2d')
    conv_mod = sys.modules['flag_gems.ops.conv2d']
    if not getattr(conv_mod.conv2d, '_wave3_patched', False):
        conv_mod.conv2d = make_patched_conv2d(conv_mod.conv2d)

    importlib.import_module('flag_gems.ops.conv3d')
    conv3d_mod = sys.modules['flag_gems.ops.conv3d']
    if not getattr(conv3d_mod.conv3d, '_wave3_patched', False):
        conv3d_mod.conv3d = make_patched_conv3d(conv3d_mod.conv3d)

    for _max_mod_name in (
        'flag_gems.runtime.backend._ascend.ops.max',
        'flag_gems.runtime.backend._ascend.ops',
        'flag_gems.ops.max',
        'flag_gems.ops',
        'flag_gems',
    ):
        try:
            _m = importlib.import_module(_max_mod_name)
            if hasattr(_m, 'max_dim') and not getattr(_m.max_dim, '_wave3_patched', False):
                _m.max_dim = make_patched_max_dim(_m.max_dim)
        except Exception:
            pass

    for _ms_mod_name in (
        'flag_gems.runtime.backend._ascend.ops.masked_scatter',
        'flag_gems.runtime.backend._ascend.ops',
        'flag_gems.ops.masked_scatter',
        'flag_gems.ops',
        'flag_gems',
    ):
        try:
            _m = importlib.import_module(_ms_mod_name)
            if hasattr(_m, 'masked_scatter') and not getattr(_m.masked_scatter, '_wave3_patched', False):
                _m.masked_scatter = make_patched_masked_scatter(_m.masked_scatter, inplace=False)
            if hasattr(_m, 'masked_scatter_') and not getattr(_m.masked_scatter_, '_wave3_patched', False):
                _m.masked_scatter_ = make_patched_masked_scatter(_m.masked_scatter_, inplace=True)
        except Exception:
            pass

    try:
        from wave3_pool_repairs import make_patched_avg_pool2d, patch_avg_pool2d_modules
        patch_avg_pool2d_modules()
    except Exception:
        make_patched_avg_pool2d = None

    try:
        from wave3_mem_repairs import install_libtuner_memory_fix
        meta['libtuner_mem_fix'] = install_libtuner_memory_fix(repairs_log)
    except Exception as _mem_exc:
        meta['libtuner_mem_fix_error'] = repr(_mem_exc)

    try:
        from wave3_dw2d_repairs import make_patched_conv2d_persistent
    except Exception:
        make_patched_conv2d_persistent = None

    import os as _os
    _os.environ.setdefault('GEMS_NAFLEX_AA_RESIZE', '1')
    meta['naflex_aa_resize'] = _os.environ.get('GEMS_NAFLEX_AA_RESIZE')

    try:
        from wave3_index_repairs import install_adjacent_index_fix
        meta['adj_index_fix'] = install_adjacent_index_fix(repairs_log)
    except Exception as _idx_exc:
        meta['adj_index_fix_error'] = repr(_idx_exc)

    try:
        from wave3_sort_repairs import install_sort_repairs, wave3_argsort, wave3_sort, wave3_sort_stable
        meta['sort_fix'] = install_sort_repairs(repairs_log)
    except Exception as _sort_exc:
        meta['sort_fix_error'] = repr(_sort_exc)
        wave3_argsort = wave3_sort = wave3_sort_stable = None

    try:
        from wave3_mul_repairs import make_patched_mul_vision_rope
        meta['vision_rope_mul_fix'] = 'installed'
    except Exception as _mul_exc:
        meta['vision_rope_mul_fix_error'] = repr(_mul_exc)
        make_patched_mul_vision_rope = None

    try:
        from wave3_broadcast_repairs import install_broadcast_runtime_error_fix
        meta['broadcast_runtime_err_fix'] = install_broadcast_runtime_error_fix(repairs_log)
    except Exception as _bcast_exc:
        meta['broadcast_runtime_err_fix_error'] = repr(_bcast_exc)

    try:
        from wave3_pad_repairs import make_patched_constant_pad_nd, make_patched_pad
        meta['constant_pad_nd_fix'] = 'installed'
    except Exception as _pad_exc:
        meta['constant_pad_nd_fix_error'] = repr(_pad_exc)
        make_patched_constant_pad_nd = make_patched_pad = None

    def _wrap_registered_fn(k_str: str, fn):
        if 'conv2d' in k_str and getattr(fn, '__name__', '') == 'conv2d' and not getattr(fn, '_wave3_patched', False):
            base_conv = make_patched_conv2d(fn)
            return make_patched_conv2d_persistent(fn, base_conv) if make_patched_conv2d_persistent is not None else base_conv
        if 'conv3d' in k_str and getattr(fn, '__name__', '') == 'conv3d' and not getattr(fn, '_wave3_patched', False):
            return make_patched_conv3d(fn)
        if (k_str.startswith('sum') or '.sum' in k_str) and not getattr(fn, '_wave3_patched', False):
            return make_patched_sum(fn)
        if k_str == 'mul.Tensor' and not getattr(fn, '_wave3_patched', False):
            base_mul = make_patched_mul(fn)
            return make_patched_mul_vision_rope(base_mul) if make_patched_mul_vision_rope is not None else base_mul
        if k_str in ('arange', 'arange.start', 'arange.start_step') and not getattr(fn, '_wave3_patched', False):
            return make_patched_arange(fn)
        if k_str == 'max.dim' and not getattr(fn, '_wave3_patched', False):
            return make_patched_max_dim(fn)
        if k_str == 'masked_scatter' and not getattr(fn, '_wave3_patched', False):
            return make_patched_masked_scatter(fn, inplace=False)
        if k_str == 'masked_scatter_' and not getattr(fn, '_wave3_patched', False):
            return make_patched_masked_scatter(fn, inplace=True)
        if k_str == 'avg_pool2d' and make_patched_avg_pool2d is not None and not getattr(fn, '_wave3_patched', False):
            return make_patched_avg_pool2d(fn)
        if k_str == 'argsort' and wave3_argsort is not None and not getattr(fn, '_wave3_patched', False):
            return wave3_argsort
        if k_str == 'sort' and wave3_sort is not None and not getattr(fn, '_wave3_patched', False):
            return wave3_sort
        if k_str == 'sort.stable' and wave3_sort_stable is not None and not getattr(fn, '_wave3_patched', False):
            return wave3_sort_stable
        if k_str == 'constant_pad_nd' and make_patched_constant_pad_nd is not None and not getattr(fn, '_wave3_patched', False):
            return make_patched_constant_pad_nd(fn)
        if k_str == 'pad' and make_patched_pad is not None and not getattr(fn, '_wave3_patched', False):
            return make_patched_pad(fn)
        return fn

    # Patch flag_gems._FULL_CONFIG BEFORE enable() so CountingRegistrar wraps patched functions directly
    patched_config_keys = []
    if hasattr(flag_gems, '_FULL_CONFIG'):
        new_cfg = []
        for item in flag_gems._FULL_CONFIG:
            if item and len(item) >= 2:
                k_str = str(item[0])
                wrapped_fn = _wrap_registered_fn(k_str, item[1])
                if wrapped_fn is not item[1]:
                    item = (item[0], wrapped_fn, *item[2:])
                    patched_config_keys.append(k_str)
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
    meta['patched_full_config'] = patched_config_keys

    # 3. Hook GeneralOpRegistrar.register_impl as fallback for any custom registrar config
    if not getattr(GeneralOpRegistrar, '_wave3_hooked', False):
        orig_register_impl = GeneralOpRegistrar.register_impl
        hooked_keys = []

        def hooked_register_impl(self, key, fn, extra_dispatch_keys=()):
            k_str = str(key)
            new_fn = _wrap_registered_fn(k_str, fn)
            if new_fn is not fn:
                fn = new_fn
                hooked_keys.append(k_str)
            return orig_register_impl(self, key, fn, extra_dispatch_keys)

        GeneralOpRegistrar.register_impl = hooked_register_impl
        GeneralOpRegistrar._wave3_hooked = True
        GeneralOpRegistrar._wave3_hooked_keys = hooked_keys

    meta['registrar_hook'] = 'installed'
    repairs_log.append(
        "Hook FlagGems GeneralOpRegistrar for conv2d (padding='same', Ascend Triton _dw2d_kernel for depthwise 2D, FlagGems F.linear lowering for pointwise/non-overlapping/RGB stem), conv3d (exact non-overlapping 3D patch -> FlagGems F.linear), sum(bool/uint8/int8->int64 before multi-axis Triton reduction), 4D squeeze-excite mul.Tensor, arange(float start/end/step -> default float dtype), max.dim (64-bit coalesced Ascend Triton inner/strided-K kernels without autotune or axis0//2 truncation), masked_scatter/masked_scatter_ (64-bit UB-safe Triton row-broadcast & elementwise kernels clamping masked indices to 0), avg_pool2d (exact ceil_mode/padding divisor & FlagGems reduction fast path for large/global kernels), GEMS_NAFLEX_AA_RESIZE=1 (separable Triton linear-MM antialiased bilinear resize via naflex_resize.py), LibTuner/Autotuner HBM cleanup, persistent 1D _dw2d_persistent_kernel, and Philox rand_kernel BLOCK=256 UB cap."
    )
    return meta


def install_offline_hub_no_exist_shim(complete_repo_ids: set[str]) -> None:
    """In HF_HUB_OFFLINE=1, manually populated snapshots/<sha> lack .no_exist/<sha>/<optional_file> markers
    and hf_hub_download raises LocalEntryNotFoundError (a subclass of EntryNotFoundError), causing
    transformers 5.x cached_files() to raise OSError instead of falling through cleanly when e.g. config.json
    is absent from a PEFT adapter repo. Since all files of our pinned revisions are completely downloaded,
    treat absent optional files as _CACHED_NO_EXIST / plain EntryNotFoundError."""
    import huggingface_hub
    import huggingface_hub.file_download as fd
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    repo_set = getattr(fd, '_wave3_complete_repos', None)
    if repo_set is None:
        repo_set = set()
        fd._wave3_complete_repos = repo_set
    repo_set.update(complete_repo_ids)

    thub = None
    try:
        import transformers.utils.hub as thub
    except Exception:
        pass

    no_exist_sentinel = getattr(fd, '_CACHED_NO_EXIST', None)
    if no_exist_sentinel is not None and not getattr(fd, '_wave3_no_exist_shimmed', False):
        orig_try = fd.try_to_load_from_cache

        @functools.wraps(orig_try)
        def shimmed_try(repo_id: str, filename: str, *args, **kwargs):
            res = orig_try(repo_id, filename, *args, **kwargs)
            if res is None and repo_id in repo_set and kwargs.get('repo_type', 'model') in (None, 'model'):
                return no_exist_sentinel
            return res

        fd.try_to_load_from_cache = shimmed_try
        if hasattr(huggingface_hub, 'try_to_load_from_cache'):
            huggingface_hub.try_to_load_from_cache = shimmed_try
        if thub is not None and hasattr(thub, 'try_to_load_from_cache'):
            thub.try_to_load_from_cache = shimmed_try
        fd._wave3_no_exist_shimmed = True

    if not getattr(fd, '_wave3_hf_dl_shimmed', False):
        orig_dl = fd.hf_hub_download

        @functools.wraps(orig_dl)
        def shimmed_dl(repo_id: str, filename: str, *args, **kwargs):
            try:
                return orig_dl(repo_id, filename, *args, **kwargs)
            except LocalEntryNotFoundError as e:
                if repo_id in repo_set and kwargs.get('repo_type', 'model') in (None, 'model'):
                    raise EntryNotFoundError(
                        f"{filename} does not exist in complete pinned snapshot of {repo_id}"
                    ) from e
                raise

        fd.hf_hub_download = shimmed_dl
        if hasattr(huggingface_hub, 'hf_hub_download'):
            huggingface_hub.hf_hub_download = shimmed_dl
        if thub is not None and hasattr(thub, 'hf_hub_download'):
            thub.hf_hub_download = shimmed_dl
        fd._wave3_hf_dl_shimmed = True


def _wrap_cpu_method(cls, attr_name: str) -> str | None:
    import torch

    if cls is None or cls is object or (isinstance(cls, type) and issubclass(cls, torch.nn.Module)):
        return None
    orig = getattr(cls, attr_name, None)
    if orig is None or not callable(orig) or getattr(orig, '_wave3_cpu_guarded', False):
        return None

    @functools.wraps(orig)
    def _cpu_guarded(*args, **kwargs):
        with torch.device('cpu'):
            return orig(*args, **kwargs)

    _cpu_guarded._wave3_cpu_guarded = True
    setattr(cls, attr_name, _cpu_guarded)
    return f"{cls.__module__}.{cls.__name__}.{attr_name}"


def install_cpu_preprocessing_guard(repairs_log: list[str] | None = None) -> list[str]:
    """Run transformers processor / image_processor / tokenizer calls under CPU default device
    so torch.set_default_device('npu:0') does not send >8D patchify/preprocess tensors to NPU CANN."""
    wrapped: list[str] = []
    targets = [
        ('transformers.processing_utils', 'ProcessorMixin', ('__call__', 'apply_chat_template')),
        ('transformers.image_processing_utils', 'BaseImageProcessor', ('__call__',)),
        ('transformers.image_processing_utils', 'ImageProcessingMixin', ('__call__',)),
        ('transformers.image_processing_utils_fast', 'BaseImageProcessorFast', ('__call__',)),
        ('transformers.tokenization_utils_base', 'PreTrainedTokenizerBase', ('__call__',)),
    ]
    for mod_name, cls_name, attrs in targets:
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name, None)
            if cls is not None:
                for attr in attrs:
                    hit = _wrap_cpu_method(cls, attr)
                    if hit:
                        wrapped.append(hit)
        except Exception:
            pass
    if repairs_log is not None and wrapped:
        repairs_log.append(
            "Guard transformers ProcessorMixin / BaseImageProcessor / PreTrainedTokenizerBase preprocessing under torch.device('cpu') so torch.set_default_device('npu:0') does not push >8D image patchify tensors to NPU."
        )
    return wrapped


def guard_adapter_processors(adapter) -> list[str]:
    """Also guard concrete subclass __call__/apply_chat_template on loaded processor objects (e.g. Qwen3VLProcessor)."""
    wrapped: list[str] = []
    objs = []
    for attr in ('processor', 'image_processor', 'tokenizer'):
        obj = getattr(adapter, attr, None)
        if obj is not None:
            objs.append(obj)
    mod = getattr(adapter, 'model', None)
    if mod is not None:
        try:
            if len(mod) > 0:
                first_mod = mod[0]
                for attr in ('processor', 'tokenizer'):
                    obj = getattr(first_mod, attr, None)
                    if obj is not None:
                        objs.append(obj)
        except Exception:
            pass

    expanded = []
    for obj in objs:
        expanded.append(obj)
        for sub_attr in ('image_processor', 'tokenizer'):
            sub = getattr(obj, sub_attr, None)
            if sub is not None:
                expanded.append(sub)

    seen_types = set()
    for obj in expanded:
        cls = type(obj)
        if cls in seen_types:
            continue
        seen_types.add(cls)
        for attr in ('__call__', 'apply_chat_template'):
            try:
                hit = _wrap_cpu_method(cls, attr)
                if hit:
                    wrapped.append(hit)
            except Exception:
                pass
    return wrapped
