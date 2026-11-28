# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Ascend linear with fixed matrix strides and a runtime batch extent."""
import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=['M'])
def linear_kernel(
    X, W, B, Y, M,
    N: tl.constexpr, K: tl.constexpr, HAS_BIAS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    row = (pid // tl.cdiv(N, BN)) * BM + tl.arange(0, BM)
    col = (pid % tl.cdiv(N, BN)) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    xp = X + row[:, None] * K + kk[None, :]
    wp = W + col[None, :] * K + kk[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    # The Ascend compiler can mishandle a single Cube accumulation followed
    # by FP32 bias addition. An extra masked iteration contributes zero.
    for step in range(max(2, tl.cdiv(K, BK))):
        remaining = K - step * BK
        x = tl.load(xp, (row < M)[:, None] & (kk < remaining)[None, :], other=0)
        w = tl.load(wp, (kk < remaining)[:, None] & (col < N)[None, :], other=0)
        acc += tl.dot(x, w, allow_tf32=False)
        xp += BK
        wp += BK
    if HAS_BIAS:
        bias = tl.load(B + col, col < N, other=0).to(tl.float32)
        acc += tl.broadcast_to(bias[None, :], (BM, BN))
    tl.store(Y + row[:, None] * N + col[None, :], acc.to(Y.dtype.element_ty),
             (row < M)[:, None] & (col < N)[None, :])


def linear(input, weight, bias=None):
    logger.debug('GEMS_ASCEND LINEAR')
    if input.ndim < 1 or weight.ndim != 2 or input.shape[-1] != weight.shape[1]:
        raise RuntimeError('linear expects input (..., K) and weight (N, K)')
    if input.dtype != weight.dtype or input.device != weight.device:
        raise RuntimeError('linear input and weight must have the same dtype and device')
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError('Ascend linear supports floating-point inputs')
    N, K = weight.shape
    if bias is not None and (bias.shape != (N,) or bias.dtype != input.dtype or bias.device != input.device):
        raise RuntimeError('linear bias must have shape (N,) and the input dtype/device')
    M = math.prod(input.shape[:-1])
    x = input.contiguous().view(M, K)
    w = weight.contiguous()
    b = bias.contiguous() if bias is not None else w
    result = torch.empty((*input.shape[:-1], N), device=input.device, dtype=input.dtype)
    if M == 0 or N == 0:
        return result
    with torch_device_fn.device(input.device):
        linear_kernel[(triton.cdiv(M, 128) * triton.cdiv(N, 128),)](
            x, w, b, result, M, N, K, bias is not None, 128, 128, 128,
            num_warps=4, num_stages=1,
        )
    return result
