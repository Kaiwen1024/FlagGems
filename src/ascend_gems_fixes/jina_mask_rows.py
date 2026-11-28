"""Preserve Boolean indexing order and avoid rebuilding adjacent-index kernels per sequence length."""
import torch

hits = 0


def gather(inp, indices, kernel):
    global hits
    if (
        inp.ndim != 3
        or inp.device.type != "npu"
        or not inp.is_contiguous()
        or inp.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or inp.shape[-1] not in (768, 1024, 1536)
        or not 0 < inp.numel() <= 2**31 - 512
        or not indices
        or len(indices) > 3
    ):
        return None
    mask = indices[0]
    if (
        mask is None
        or mask.dtype != torch.bool
        or mask.device != inp.device
        or tuple(mask.shape) != tuple(inp.shape[:-1])
        or any(x is not None for x in indices[1:])
    ):
        return None
    ids = mask.reshape(-1).nonzero().reshape(-1).contiguous()
    width = inp.shape[-1]
    out = torch.empty((ids.numel(), width), device=inp.device, dtype=inp.dtype)
    if out.numel():
        import triton

        with torch.npu.device(inp.device):
            kernel[(triton.cdiv(out.numel(), 256),)](
                inp, ids, out, out.numel(), width, 256, num_warps=4, num_stages=1
            )
    hits += 1
    return out
