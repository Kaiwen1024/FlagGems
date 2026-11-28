"""Ascend add paths that keep small-tensor indexing in int32."""
import torch
import triton
import triton.language as tl

from flag_gems.ops.add import add as _generic_add
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


@libentry()
@triton.jit
def _add_contiguous_i32(A, B, Y, N, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    a = tl.load(A + offsets, mask=mask, other=0).to(tl.float32)
    b = tl.load(B + offsets, mask=mask, other=0).to(tl.float32)
    tl.store(Y + offsets, a + b, mask=mask)


@libentry()
@triton.jit
def _add_attention_mask_i32(A, B, Y, PLANE, HEADS, BATCH_STRIDE, BLOCK: tl.constexpr):
    plane_id = tl.program_id(1)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < PLANE
    a_offsets = plane_id * PLANE + offsets
    b_offsets = (plane_id // HEADS) * BATCH_STRIDE + offsets
    a = tl.load(A + a_offsets, mask=mask, other=0).to(tl.float32)
    b = tl.load(B + b_offsets, mask=mask, other=0).to(tl.float32)
    tl.store(Y + a_offsets, a + b, mask=mask)


@libentry()
@triton.jit
def _add_suffix_bias_i32(A, B, Y, PERIOD, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.program_id(1)
    mask = offsets < PERIOD
    positions = row * PERIOD + offsets
    a = tl.load(A + positions, mask=mask, other=0).to(tl.float32)
    b = tl.load(B + offsets, mask=mask, other=0).to(tl.float32)
    tl.store(Y + positions, a + b, mask=mask)


@libentry()
@triton.jit
def _add_head_bias_i32(A, B, Y, SEQUENCE, HEADS, AS0, AS1, AS2, BS1,
                       WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    plane = tl.program_id(1)
    batch = plane // HEADS
    head = plane % HEADS
    seq = offsets // WIDTH
    channel = offsets % WIDTH
    mask = seq < SEQUENCE
    a_offsets = batch * AS0 + head * AS1 + seq * AS2 + channel
    b_offsets = head * BS1 + channel
    y_offsets = plane * (SEQUENCE * WIDTH) + offsets
    a = tl.load(A + a_offsets, mask=mask, other=0).to(tl.float32)
    b = tl.load(B + b_offsets, mask=mask, other=0).to(tl.float32)
    tl.store(Y + y_offsets, a + b, mask=mask)


def _suffix_broadcast(a_shape, b_shape):
    if len(b_shape) > len(a_shape):
        return False
    padded = (1,) * (len(a_shape) - len(b_shape)) + tuple(b_shape)
    started = False
    for a_dim, b_dim in zip(a_shape, padded):
        if b_dim != 1:
            started = True
        if started and a_dim != b_dim:
            return False
    return started


def _i32_span(tensor):
    return (all(s >= 0 for s in tensor.stride())
            and sum((dim - 1) * stride for dim, stride in zip(tensor.shape, tensor.stride())) <= 2**31 - 512)


def add(A, B, *, alpha=1):
    if __import__('os').environ.get('GEMS_NPU_KEY_MASK_ADD')=='1':
        from jina_keymask import keymask_add
        result = keymask_add(A,B,alpha)
        if result is not None:return result

    # Keep large attention additions in bounded launch-local index ranges.
    # This splits pointwise work only; the model's batch and sequence stay intact.
    if (isinstance(A,torch.Tensor) and isinstance(B,torch.Tensor) and alpha == 1
            and A.device == B.device and A.device.type == 'npu'
            and A.dtype == B.dtype == torch.float16
            and A.is_contiguous() and B.is_contiguous()
            and A.ndim == B.ndim == 4 and A.numel() > 2**31 - 512
            and B.shape[0] in (1,A.shape[0]) and B.shape[1] == 1
            and B.shape[2:] == A.shape[2:]):
        plane = A.shape[2] * A.shape[3]
        batch_elements = A.shape[1] * plane
        chunk_batches = (2**31 - 512) // batch_elements
        if chunk_batches > 0:
            result = torch.empty_like(A)
            with torch_device_fn.device(A.device):
                for start in range(0,A.shape[0],chunk_batches):
                    stop = min(start+chunk_batches,A.shape[0])
                    a = A[start:stop]
                    b = B if B.shape[0] == 1 else B[start:stop]
                    y = result[start:stop]
                    _add_attention_mask_i32[(triton.cdiv(plane,512),(stop-start)*A.shape[1])](
                        a,b,y,plane,A.shape[1],0 if B.shape[0] == 1 else plane,BLOCK=512)
            return result
    if (isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor)
            and alpha == 1 and A.device == B.device and A.device.type == 'npu'
            and A.dtype == B.dtype and A.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and 0 < A.numel() <= 2**31 - 512 and 0 < B.numel() <= 2**31 - 512):
        if A.is_contiguous() and B.is_contiguous() and A.shape == B.shape:
            result = torch.empty_like(A)
            with torch_device_fn.device(A.device):
                _add_contiguous_i32[(triton.cdiv(A.numel(), 512),)](A, B, result, A.numel(), BLOCK=512)
            return result
        if (A.is_contiguous() and B.is_contiguous()
                and A.ndim == B.ndim == 4 and B.shape[0] in (1, A.shape[0])
                and B.shape[1] == 1 and B.shape[2:] == A.shape[2:]):
            plane = A.shape[2] * A.shape[3]
            result = torch.empty_like(A)
            with torch_device_fn.device(A.device):
                _add_attention_mask_i32[(triton.cdiv(plane, 512), A.shape[0]*A.shape[1])](
                    A, B, result, plane, A.shape[1], 0 if B.shape[0] == 1 else plane, BLOCK=512)
            return result
        if (A.is_contiguous() and B.is_contiguous()
                and _suffix_broadcast(A.shape, B.shape)
                and B.numel() >= 16 and A.numel() // B.numel() <= 65535):
            period = B.numel()
            result = torch.empty_like(A)
            with torch_device_fn.device(A.device):
                _add_suffix_bias_i32[(triton.cdiv(period,512), A.numel() // period)](
                    A, B, result, period, BLOCK=512)
            return result
        if (A.ndim == B.ndim == 4 and A.shape[3] == 64
                and B.shape == (1, A.shape[1], 1, A.shape[3])
                and A.stride(-1) == B.stride(-1) == 1 and _i32_span(A) and _i32_span(B)):
            result = torch.empty(A.shape, device=A.device, dtype=A.dtype)
            with torch_device_fn.device(A.device):
                _add_head_bias_i32[(triton.cdiv(A.shape[2]*A.shape[3],512),A.shape[0]*A.shape[1])](
                    A, B, result, A.shape[2], A.shape[1], A.stride(0), A.stride(1),
                    A.stride(2), B.stride(1), WIDTH=A.shape[3], BLOCK=512)
            return result
    return _generic_add(A, B, alpha=alpha)
