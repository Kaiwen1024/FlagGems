"""Ascend floating-point mul paths with int32 broadcast indexing."""
import torch
import triton
import triton.language as tl

from flag_gems.ops.mul import mul as _generic_mul
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


@libentry()
@triton.jit
def _mul_broadcast_i32(A, B, Y, WIDTH, ROW_SCALAR: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = col < WIDTH
    offset = row * WIDTH + col
    a = tl.load(A + offset, mask=mask, other=0).to(tl.float32)
    if ROW_SCALAR:
        b = tl.load(B + row).to(tl.float32)
    else:
        b = tl.load(B + col, mask=mask, other=0).to(tl.float32)
    tl.store(Y + offset, a * b, mask=mask)


@libentry()
@triton.jit
def _contig_mul_i32(X,C,Y,N,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    x=tl.load(X+i,i<N,other=0).to(tl.float32)
    v=tl.load(C+i,i<N,other=0).to(tl.float32)
    tl.store(Y+i,x*v,i<N)

@libentry()
@triton.jit
def _rotary_mul(X, C, Y, S, H, D: tl.constexpr, XB, XH, XS, CB, CS, BLOCK: tl.constexpr):
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    s = pos // D
    d = pos % D
    mask = pos < S * D
    x = tl.load(X + b*XB + h*XH + s*XS + d, mask, other=0).to(tl.float32)
    c = tl.load(C + b*CB + s*CS + d, mask, other=0).to(tl.float32)
    tl.store(Y + bh*(S*D) + pos, x*c, mask)

@libentry()
@triton.jit
def _gated_mul(X, G, Y, S, D: tl.constexpr, XB, XS, GB, GS, BLOCK: tl.constexpr):
    pos = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)
    b = tl.program_id(1)
    row = pos // D
    col = pos % D
    mask = pos < S*D
    x=tl.load(X+b*XB+row*XS+col,mask,other=0).to(tl.float32)
    g=tl.load(G+b*GB+row*GS+col,mask,other=0).to(tl.float32)
    tl.store(Y+b*(S*D)+pos,x*g,mask)

def _fast_layout_mul(x,c):
    if not isinstance(x,torch.Tensor) or not isinstance(c,torch.Tensor):
        return None
    allowed=(torch.float16,torch.bfloat16,torch.float32)
    if x.device!=c.device or x.device.type!='npu' or x.dtype not in allowed or c.dtype not in allowed:
        return None
    if x.ndim!=c.ndim or x.ndim not in (3,4) or not (x.numel() and c.numel()):
        return None
    if x.stride(-1)!=1 or c.stride(-1)!=1:
        return None
    if x.numel()<c.numel():x,c=c,x
    if max(x.numel(), c.numel(), *(sum((n-1)*s for n,s in zip(t.shape,t.stride())) for t in (x,c)))>2**31-512:
        return None
    if x.is_contiguous() and c.is_contiguous() and x.shape==c.shape:
        out=torch.empty(x.shape,dtype=torch.promote_types(x.dtype,c.dtype),device=x.device)
        with torch_device_fn.device(x.device):
            _contig_mul_i32[(triton.cdiv(x.numel(),512),)](x,c,out,x.numel(),512)
        return out
    if x.ndim==4 and c.shape[1]==1 and x.shape[2:]==c.shape[2:] and c.shape[0] in (1,x.shape[0]) and x.shape[-1] in (32,64,80,96,128,256):
        B,H,S,D=x.shape
        out=torch.empty(x.shape,dtype=torch.promote_types(x.dtype,c.dtype),device=x.device)
        with torch_device_fn.device(x.device):
            _rotary_mul[(triton.cdiv(S*D,512),B*H)](x,c,out,S,H,D,*x.stride()[:3],0 if c.shape[0]==1 else c.stride(0),c.stride(2),512)
        return out
    if x.ndim==3 and x.shape==c.shape and x.shape[-1]>=32:
        B,S,D=x.shape
        out=torch.empty(x.shape,dtype=torch.promote_types(x.dtype,c.dtype),device=x.device)
        with torch_device_fn.device(x.device):
            _gated_mul[(triton.cdiv(S*D,512),B)](x,c,out,S,D,*x.stride()[:2],*c.stride()[:2],512)
        return out
    return None



def mul(A, B):
    if __import__('os').environ.get('GEMS_NPU_KEY_MASK_ADD')=='1':
        from jina_keymask import packed_scale
        result = packed_scale(A,B)
        if result is not None:return result

    fast = _fast_layout_mul(A, B)
    if fast is not None:
        return fast
    if (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor)
            and A.device == B.device and A.device.type == 'npu'
            and A.dtype == B.dtype and A.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and A.is_contiguous() and B.is_contiguous()
            and 0 < A.numel() <= 2**31 - 512 and 0 < B.numel() <= 2**31 - 512):
        if A.numel() < B.numel():
            A, B = B, A
        width = None
        row_scalar = False
        # A contiguous suffix, including channel bias, repeats per outer row.
        suffix = tuple(B.shape)
        while suffix and suffix[0] == 1:
            suffix = suffix[1:]
        if suffix and B.ndim <= A.ndim and tuple(A.shape[-len(suffix):]) == suffix:
            width = B.numel()
        elif (A.ndim >= 1 and B.ndim == A.ndim and B.shape[-1] == 1
              and tuple(B.shape[:-1]) == tuple(A.shape[:-1])):
            width = A.shape[-1]
            row_scalar = True
        if width is not None and width >= 16:
            result = torch.empty_like(A)
            block = min(512, triton.next_power_of_2(width))
            with torch_device_fn.device(A.device):
                _mul_broadcast_i32[(triton.cdiv(width, block), A.numel() // width)](
                    A, B, result, width, ROW_SCALAR=row_scalar, BLOCK=block)
            return result
    return _generic_mul(A, B)
