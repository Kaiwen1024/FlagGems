"""Process-local NPU sparse-sparse similarity shim for Wave 3 (e.g. c018 naver/v-splade-quality).

On Ascend NPU, `sentence_transformers.util.similarity.dot_score` / `cos_sim`
calls `torch.mm(a, b.transpose(0, 1)).to_dense()` directly on NPU `sparse_coo`
tensors, raising:
  RuntimeError: CAUTION: The operator 'aten::_sparse_sparse_matmul' is not currently supported on the NPU backend. [ERR01007]

This helper wraps `dot_score`, `cos_sim`, and `pytorch_cos_sim` identically
for both `native` and `gems` modes ONLY when at least one operand is sparse
and on an NPU device: it materializes bounded `(<=512, V)` float32 row blocks
(preferring NPU `to_dense()` when small, with CPU `.coalesce()` sparse-slice
fallback) and executes `torch.mm(a_blk, b_blk.transpose(0, 1))` on the NPU.
"""
from __future__ import annotations

import functools
import importlib


def _materialize_dense_block_f32(t, start: int, end: int, dev, state: dict):
    import torch

    n_rows = end - start
    if not (t.is_sparse or getattr(t, 'is_sparse_csr', False)):
        return t[start:end].to(device=dev, dtype=torch.float32)

    if t.shape[0] <= n_rows and start == 0:
        if 'full_dense' not in state:
            try:
                state['full_dense'] = t.to_dense().to(device=dev, dtype=torch.float32)
            except Exception:
                state['full_dense'] = None
        if state['full_dense'] is not None:
            return state['full_dense']

    if 'cpu_coo' not in state:
        tc = t.to_sparse() if getattr(t, 'is_sparse_csr', False) else t
        state['cpu_coo'] = tc.detach().cpu().coalesce()

    coo = state['cpu_coo']
    idx = coo.indices()
    val = coo.values().to(dtype=torch.float32)
    mask = (idx[0] >= start) & (idx[0] < end)
    sub_idx = torch.stack((idx[0, mask] - start, idx[1, mask]), dim=0)
    dense_cpu = torch.sparse_coo_tensor(
        sub_idx,
        val[mask],
        size=(n_rows, t.shape[1]),
        dtype=torch.float32,
    ).coalesce().to_dense()
    return dense_cpu.to(device=dev)


def _chunked_npu_sparse_mm(a, b, normalize: bool = False, q_chunk: int = 512, d_chunk: int = 512):
    import torch
    import torch.nn.functional as F

    dev = a.device if a.device.type == 'npu' else b.device
    q_total = a.shape[0]
    d_total = b.shape[0]
    out = torch.empty((q_total, d_total), device=dev, dtype=torch.float32)
    if q_total == 0 or d_total == 0:
        return out

    a_state: dict = {}
    b_state: dict = {}

    # Pre-materialize dense blocks cleanly with L2 normalization if cosine
    b_blocks: list[tuple[int, int, object]] = []
    for ds in range(0, d_total, d_chunk):
        de = min(ds + d_chunk, d_total)
        b_blk = _materialize_dense_block_f32(b, ds, de, dev, b_state)
        if normalize:
            b_blk = F.normalize(b_blk, p=2, dim=-1)
        b_blocks.append((ds, de, b_blk))

    for qs in range(0, q_total, q_chunk):
        qe = min(qs + q_chunk, q_total)
        a_blk = _materialize_dense_block_f32(a, qs, qe, dev, a_state)
        if normalize:
            a_blk = F.normalize(a_blk, p=2, dim=-1)
        for ds, de, b_blk in b_blocks:
            out[qs:qe, ds:de] = torch.mm(a_blk, b_blk.transpose(0, 1))

    return out


def install_npu_sparse_similarity_fix(repairs_log: list[str] | None = None) -> dict:
    sim_mod = importlib.import_module('sentence_transformers.util.similarity')
    util_mod = None
    try:
        util_mod = importlib.import_module('sentence_transformers.util')
    except Exception:
        pass

    from sentence_transformers.util.tensor import _convert_to_batch_tensor

    patched = []

    def _should_intercept(ta, tb) -> bool:
        return (
            (ta.is_sparse or tb.is_sparse or getattr(ta, 'is_sparse_csr', False) or getattr(tb, 'is_sparse_csr', False))
            and (ta.device.type == 'npu' or tb.device.type == 'npu')
        )

    orig_dot = sim_mod.dot_score
    if not getattr(orig_dot, '_wave3_sparse_sim_patched', False):
        @functools.wraps(orig_dot)
        def wrapped_dot_score(a, b):
            ta = _convert_to_batch_tensor(a)
            tb = _convert_to_batch_tensor(b)
            if _should_intercept(ta, tb):
                return _chunked_npu_sparse_mm(ta, tb, normalize=False)
            return orig_dot(ta, tb)

        wrapped_dot_score._wave3_sparse_sim_patched = True
        sim_mod.dot_score = wrapped_dot_score
        if util_mod is not None and hasattr(util_mod, 'dot_score'):
            util_mod.dot_score = wrapped_dot_score
        patched.append('dot_score')

    orig_cos = sim_mod.cos_sim
    if not getattr(orig_cos, '_wave3_sparse_sim_patched', False):
        @functools.wraps(orig_cos)
        def wrapped_cos_sim(a, b):
            ta = _convert_to_batch_tensor(a)
            tb = _convert_to_batch_tensor(b)
            if _should_intercept(ta, tb):
                return _chunked_npu_sparse_mm(ta, tb, normalize=True)
            return orig_cos(ta, tb)

        wrapped_cos_sim._wave3_sparse_sim_patched = True
        sim_mod.cos_sim = wrapped_cos_sim
        if util_mod is not None and hasattr(util_mod, 'cos_sim'):
            util_mod.cos_sim = wrapped_cos_sim
        patched.append('cos_sim')

        @functools.wraps(sim_mod.pytorch_cos_sim)
        def wrapped_pytorch_cos_sim(a, b):
            return sim_mod.cos_sim(a, b)

        wrapped_pytorch_cos_sim._wave3_sparse_sim_patched = True
        sim_mod.pytorch_cos_sim = wrapped_pytorch_cos_sim
        if util_mod is not None and hasattr(util_mod, 'pytorch_cos_sim'):
            util_mod.pytorch_cos_sim = wrapped_pytorch_cos_sim
        patched.append('pytorch_cos_sim')

    if repairs_log is not None and patched:
        repairs_log.append(
            "Wrap sentence_transformers dot_score/cos_sim on NPU sparse_coo tensors using chunked float32 dense NPU torch.mm (avoiding unsupported aten::_sparse_sparse_matmul ERR01007)."
        )
    return {'patched': patched}
