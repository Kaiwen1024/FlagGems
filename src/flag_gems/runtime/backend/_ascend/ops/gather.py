# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.shape_utils import restride_dim

logger = logging.getLogger(__name__)
UB_SIZE_BYTES = 192 * 1024


def compute_base_offset(shape, strides, dim):
    idx = torch.arange(int(torch.prod(torch.tensor(shape))), device="cpu")
    coord = torch.empty((len(shape), idx.numel()), dtype=torch.long, device="cpu")
    for i in reversed(range(len(shape))):
        coord[i] = idx % shape[i]
        idx = idx // shape[i]

    offset = torch.zeros_like(coord[0])
    for i in range(len(shape)):
        if i != dim:
            offset += coord[i] * strides[i]
    return offset


@libentry()
@triton.heuristics({"BLOCK_SIZE": lambda args: 1024})
@triton.jit
def _gather_flat_kernel_fixed(
    inp,
    index,
    out,
    base_offset,
    inp_dim_stride,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    cur_index = tl.load(index + offset, mask=mask, other=0)
    base = tl.load(base_offset + offset, mask=mask, other=0)

    inp_offset = base + cur_index * inp_dim_stride

    val = tl.load(inp + inp_offset, mask=mask, other=0)
    tl.store(out + offset, val, mask=mask)



@libentry()
@triton.jit
def _gather_strided_3d(X, I, Y, N, S1, S2, XS0,XS1,XS2, IS0,IS1,IS2, YS0,YS1,YS2,
                       DIM:tl.constexpr, BLOCK:tl.constexpr):
    offs=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    t=offs//S2
    k=offs%S2
    i=t//S1
    j=t%S1
    valid=offs<N
    idx=tl.load(I+i*IS0+j*IS1+k*IS2,valid,other=0).to(tl.int32)
    if DIM==0:
        xidx=idx*XS0+j*XS1+k*XS2
    elif DIM==1:
        xidx=i*XS0+idx*XS1+k*XS2
    else:
        xidx=i*XS0+j*XS1+idx*XS2
    val=tl.load(X+xidx,valid,other=0)
    tl.store(Y+i*YS0+j*YS1+k*YS2,val,valid)

def gather_flat_fixed(inp: torch.Tensor, dim: int, index: torch.Tensor, out=None):
    logger.debug("GEMS_ASCEND GATHER")

    # The flat kernel addresses index linearly; expand() may give it zero strides.
    if inp.ndim==3 and index.ndim==3 and index.dtype==torch.int64 and (out is None or out.shape==index.shape):
        if out is None:
            out=torch.empty(index.shape,dtype=inp.dtype,device=inp.device)
        if index.numel()==0:return out
        if (0<inp.numel() and index.numel()<=2**31-1024 and
            all(sum((n-1)*st for n,st in zip(v.shape,v.stride()))<=2**31-1024 for v in (inp,index,out))
            and all(index.shape[j]<=inp.shape[j] for j in range(3) if j!=dim%3)):
            _gather_strided_3d[(triton.cdiv(index.numel(),512),)](inp,index,out,index.numel(),index.shape[1],index.shape[2],
                *inp.stride(),*index.stride(),*out.stride(),dim%3,512)
            return out
    index = index.contiguous()

    if out is None:
        out = torch.empty_like(index, dtype=inp.dtype, device=inp.device)

    N = index.numel()
    dim_stride = inp.stride(dim)
    inp_strided = restride_dim(inp, dim, index.shape)
    if dim == -1:
        dim = inp_strided.dim() - 1
    base_offset = compute_base_offset(index.shape, inp_strided.stride(), dim).to(
        torch.int64
    )
    base_offset = base_offset.npu()
    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
    _gather_flat_kernel_fixed[grid](
        inp_strided,
        index,
        out,
        base_offset,
        dim_stride,
        N,
    )
    return out


def gather(inp, dim, index, out=None, sparse_grad=False):
    logger.debug("GEMS_ASCEND GATHER")
    if inp.ndim != index.ndim:
        raise IndexError(
            f"self and index must have the same number of dimensions, "
            f"got self.ndim = {inp.ndim} and index.ndim = {index.ndim}"
        )
    if out is None:
        out = torch.empty_like(index, dtype=inp.dtype, device=inp.device)

    dim = dim % inp.dim()
    return gather_flat_fixed(inp, dim, index, out)


def gather_backward(grad, self, dim, index, sparse_grad):
    logger.debug("GEMS_ASCEND GATHER_BACKWARD")
    from .scatter import scatter_

    result = grad.new_zeros(self.shape)
    return scatter_(result, dim, index, grad, reduce="add")
