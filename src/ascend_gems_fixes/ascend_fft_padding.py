"""Safe final-axis padding for FP32 real-FFT inputs, including [B,T,H] transposed to [B,H,T]."""
import torch,triton
import triton.language as tl
from flag_gems.utils import libentry
@libentry()
@triton.jit
def _pad_last_rows(X,Y,N,L,OL,LEFT,VALUE,BLOCK:tl.constexpr):
    off=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=tl.minimum(off//OL,(N-1)//OL);col=off%OL-LEFT
    valid=(col>=0)&(col<L)
    src=row*L+tl.minimum(tl.maximum(col,0),L-1)
    v=tl.load(X+src,mask=off<N,other=0.)
    tl.store(Y+off,tl.where(valid,v,VALUE),mask=off<N)
def run(x,pad,value):
    if (x.device.type!='npu' or x.dtype!=torch.float32 or x.ndim not in (2,3) or not 2<=len(pad)<=2*x.ndim
        or len(pad)%2 or any(v!=0 for v in pad[2:]) or not all(isinstance(v,int) and v>=0 for v in pad[:2])
        or not x.numel() or torch.is_grad_enabled()):return None
    left,right=pad[:2];l=x.shape[-1];o=l+left+right;n=x.numel()//l*o
    if n>2**31-512 or left+right==0:return None
    xc=x.contiguous();out=torch.empty((*x.shape[:-1],o),dtype=x.dtype,device=x.device)
    with torch.npu.device(x.device):
        _pad_last_rows[(triton.cdiv(n,256),)](xc,out,n,l,o,left,float(value or 0.),BLOCK=256,num_warps=4,num_stages=1)
    return out
