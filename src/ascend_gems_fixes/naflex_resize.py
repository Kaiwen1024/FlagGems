"""Separable antialiased bilinear resize; preserve NCHW logical order on strided tensors."""
from functools import lru_cache
import numpy as np
import torch
from torch.nn import functional as F


@lru_cache(maxsize=32)
def axis(n, m, scale, device):
    center = (np.arange(m, dtype=np.float64) + 0.5) * scale - 0.5
    dist = np.abs(np.arange(n, dtype=np.float64)[None, :] - center[:, None])
    w = np.maximum(0.0, 1.0 - dist / max(1.0, scale))
    w /= w.sum(axis=1, keepdims=True)
    return torch.from_numpy(w.astype("float32")).to(device)


def resize(x, output_size, scales_h=None, scales_w=None):
    n, c, ih, iw = x.shape
    oh, ow = output_size
    sh = 1.0 / scales_h if scales_h is not None and scales_h > 0 else ih / oh
    sw = 1.0 / scales_w if scales_w is not None and scales_w > 0 else iw / ow
    wx = axis(iw, ow, sw, str(x.device))
    wy = axis(ih, oh, sh, str(x.device))
    v = F.linear(x.float().contiguous().reshape(-1, iw), wx).reshape(n, c, ih, ow)
    if x.dtype == torch.uint8:
        v = v.round().clamp_(0, 255)
    v = F.linear(v.transpose(-1, -2).contiguous().reshape(-1, ih), wy)
    v = v.reshape(n, c, ow, oh).transpose(-1, -2).contiguous()
    if x.dtype == torch.uint8:
        v = v.round().clamp_(0, 255)
    return v.to(x.dtype)
