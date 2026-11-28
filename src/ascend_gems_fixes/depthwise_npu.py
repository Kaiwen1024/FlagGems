"""Ascend inference kernels for the short M2, LFM and Qwen depthwise conv1d layers."""
import torch
import triton
import triton.language as tl
from flag_gems.utils import libentry

@libentry()
@triton.jit
def _four_tap_channels(X,W,BIAS,Y,C,L,OL, P:tl.constexpr, BI:tl.constexpr, BC:tl.constexpr):
    # Storage for X and Y is [batch,time,channel]. All channel loads are consecutive.
    pid=tl.program_id(0)
    groups=tl.cdiv(C,BC);ch=(pid%groups)*BC+tl.arange(0,BC)
    bt=pid//groups;b=bt//OL;t=bt%OL
    acc=tl.full((BC,),0,tl.float32)
    for k in tl.static_range(4):
        i=t-P+k
        valid=(i>=0)&(i<L)
        safe=tl.minimum(tl.maximum(i,0),L-1)
        x=tl.load(X+(b*L+safe)*C+ch,ch<C,other=0).to(tl.float32)
        a=tl.load(W+ch*4+k,ch<C,other=0).to(tl.float32)
        acc=tl.fma(tl.where(valid,x,0.),a,acc)
    if BI:acc+=tl.load(BIAS+ch,ch<C,other=0).to(tl.float32)
    tl.store(Y+bt*C+ch,acc,ch<C)

@libentry()
@triton.jit
def _dw3(X,W,BIAS,Y,C,L,OL,S0,S1,S2,P:tl.constexpr,HAS_BIAS:tl.constexpr,
         BC:tl.constexpr, BT:tl.constexpr, K:tl.constexpr=3):
    b=tl.program_id(2)
    ch=tl.program_id(1)*BC+tl.arange(0,BC)
    t=tl.program_id(0)*BT+tl.arange(0,BT)
    acc=tl.full((BC,BT),0,tl.float32)
    for k in tl.static_range(K):
        s=t-P+k
        v=tl.load(X+b*S0+ch[:,None]*S1+s[None,:]*S2,(ch[:,None]<C)&(s[None,:]>=0)&(s[None,:]<L),other=0).to(tl.float32)
        w=tl.load(W+ch*K+k,ch<C,other=0).to(tl.float32)
        acc=tl.fma(v,w[:,None],acc)
    if HAS_BIAS:
        acc+=tl.load(BIAS+ch,ch<C,other=0).to(tl.float32)[:,None]
    tl.store(Y+(b*C+ch[:,None])*OL+t[None,:],acc,(ch[:,None]<C)&(t[None,:]<OL))

def run(x,w,bias,stride,padding,dilation,groups):
    def one(v):return v[0] if isinstance(v,(tuple,list)) and len(v)==1 else v
    pad=one(padding)
    env=__import__('os').environ
    allowed=(torch.float16,torch.float32) if env.get('GEMS_M2_DEPTHWISE_FP32')=='1' or env.get('GEMS_QWEN4_DEPTHWISE')=='1' else (torch.float16,)
    if w.ndim!=3:return None
    k=w.shape[-1]
    if k!=3 and not(k==4 and env.get('GEMS_QWEN4_DEPTHWISE')=='1'):return None
    if (torch.is_grad_enabled() or x.device.type!='npu' or x.dtype not in allowed or w.dtype!=x.dtype or w.device!=x.device
        or x.ndim!=3 or x.shape[1]!=groups or tuple(w.shape)!=(groups,1,k)
        or not w.is_contiguous() or one(stride)!=1 or one(dilation)!=1 or not isinstance(pad,int) or not 0<=pad<k
        or (bias is not None and (bias.device!=x.device or bias.dtype!=x.dtype or bias.shape!=(groups,) or not bias.is_contiguous()))):
        return None
    n,c,l=x.shape;ol=l+2*pad-k+1
    if not (n and c and l and ol>0) or n*c*ol>2**31-512 or sum((a-1)*b for a,b in zip(x.shape,x.stride()))>2**31-512:return None
    if k==4 and env.get('GEMS_FOUR_TAP_CHANNELS')=='1':
        src=x.transpose(1,2).contiguous();dest=torch.empty((n,ol,c),dtype=x.dtype,device=x.device)
        with torch.npu.device(x.device):
            _four_tap_channels[(n*ol*triton.cdiv(c,256),)](src,w,bias if bias is not None else w,dest,c,l,ol,
                                                          P=pad,BI=bias is not None,BC=256,num_warps=4,num_stages=1)
        return dest.transpose(1,2).contiguous()
    y=torch.empty((n,c,ol),device=x.device,dtype=x.dtype)
    with torch.npu.device(x.device):
        _dw3[(triton.cdiv(ol,32),triton.cdiv(c,32),n)](x,w,bias if bias is not None else w,y,c,l,ol,*x.stride(),pad,bias is not None,32,32,k)
    return y
