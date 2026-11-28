"""Process-local shape-agnostic Ascend Triton index.Tensor repair for Wave 3.

Eliminates per-shape dynamic Python codegen & LibTuner compilation storms
(e.g. 493 recompilations in c012, 358 in c011) in `_ascend/ops/index.py`
`AscendAdjacentIndexFunction` by replacing `_adjacent_index_func` with
static, zero-autotune, 64-bit persistent 1D-grid Triton kernels compiled
once per dtype regardless of sequence length or input shape.

Crucially, `_adj_index_gather_vec_kernel` keeps `INP + src_base + n_off`
as a linear scalar+arange expression so BiShengIR lowers it to contiguous
1D MTE DMA copies (~0.17 ms instead of ~42 ms vector gather).
"""
from __future__ import annotations

import importlib
import sys

_ADJ_INDEX_KERNELS = None


def _get_adj_index_kernels():
    global _ADJ_INDEX_KERNELS
    if _ADJ_INDEX_KERNELS is None:
        import triton
        import triton.language as tl
        from flag_gems.utils import libentry

        @libentry()
        @triton.jit
        def _adj_prep_1idx_kernel(I0, LIN, M, D0, n_blocks, n_progs, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            zero64 = tl.zeros([], dtype=tl.int64)
            d0 = zero64 + D0
            for b in range(pid, n_blocks, n_progs):
                off = (zero64 + b) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
                m = off < M
                raw = tl.load(I0 + off, mask=m, other=0).to(tl.int64)
                k = tl.where(raw < 0, raw + d0, raw)
                ok = (k >= 0) & (k < d0)
                tl.store(LIN + off, tl.where(ok, k, -1), mask=m)

        @libentry()
        @triton.jit
        def _adj_prep_2idx_kernel(I0, I1, LIN, M, D0, D1, n_blocks, n_progs, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            zero64 = tl.zeros([], dtype=tl.int64)
            d0 = zero64 + D0
            d1 = zero64 + D1
            for b in range(pid, n_blocks, n_progs):
                off = (zero64 + b) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
                m = off < M
                r0 = tl.load(I0 + off, mask=m, other=0).to(tl.int64)
                r1 = tl.load(I1 + off, mask=m, other=0).to(tl.int64)
                k0 = tl.where(r0 < 0, r0 + d0, r0)
                k1 = tl.where(r1 < 0, r1 + d1, r1)
                ok = (k0 >= 0) & (k0 < d0) & (k1 >= 0) & (k1 < d1)
                tl.store(LIN + off, tl.where(ok, k0 * d1 + k1, -1), mask=m)

        @libentry()
        @triton.jit
        def _adj_index_gather_vec_kernel(
            INP,
            LIN,
            OUT,
            V,
            M,
            N,
            R,
            n_progs,
            BLOCK_N: tl.constexpr,
        ):
            pid = tl.program_id(0)
            zero64 = tl.zeros([], dtype=tl.int64)
            v64 = zero64 + V
            M64 = zero64 + M
            n64 = zero64 + N

            for r in range(pid, R, n_progs):
                p = r // M
                m = r % M
                src_r = tl.sum(tl.load(LIN + (zero64 + m) + tl.arange(0, 1)), axis=0)
                if src_r >= 0:
                    src_base = ((zero64 + p) * v64 + src_r) * n64
                    out_base = ((zero64 + p) * M64 + (zero64 + m)) * n64
                    for n0 in range(0, N, BLOCK_N):
                        n_off = (zero64 + n0) + tl.arange(0, BLOCK_N).to(tl.int64)
                        nm = n_off < n64
                        vals = tl.load(INP + src_base + n_off, mask=nm, other=0)
                        tl.store(OUT + out_base + n_off, vals, mask=nm)

        @libentry()
        @triton.jit
        def _adj_index_gather_small_n_kernel(
            INP,
            LIN,
            OUT,
            V,
            M,
            N,
            m_tiles,
            total_tasks,
            n_progs,
            BLOCK_M: tl.constexpr,
        ):
            pid = tl.program_id(0)
            zero64 = tl.zeros([], dtype=tl.int64)
            zero_vec64 = tl.zeros([BLOCK_M], dtype=tl.int64)
            v64 = zero64 + V
            M64 = zero64 + M
            n64 = zero64 + N

            for task_idx in range(pid, total_tasks, n_progs):
                mt = task_idx % m_tiles
                p = task_idx // m_tiles
                p64 = zero64 + p

                m_off = mt * BLOCK_M + tl.arange(0, BLOCK_M)
                m_mask = m_off < M
                m_off64 = m_off.to(tl.int64)
                safe_m = tl.where(m_mask, m_off64, zero64)

                lin_vec = tl.load(LIN + safe_m, mask=m_mask, other=zero_vec64 - 1)
                active = m_mask & (lin_vec >= 0)

                safe_lin = tl.where(active, lin_vec, zero64)
                base_src = (p64 * v64 + safe_lin) * n64
                base_dst = (p64 * M64 + m_off64) * n64

                for ni in range(0, N):
                    ni64 = zero64 + ni
                    s_off = tl.where(active, base_src + ni64, zero64)
                    d_off = tl.where(active, base_dst + ni64, zero64)
                    val = tl.load(INP + s_off, mask=active, other=0)
                    tl.store(OUT + d_off, val, mask=active)

        _ADJ_INDEX_KERNELS = (
            _adj_prep_1idx_kernel,
            _adj_prep_2idx_kernel,
            _adj_index_gather_vec_kernel,
            _adj_index_gather_small_n_kernel,
        )
    return _ADJ_INDEX_KERNELS


def _shape_product(dims) -> int:
    prod = 1
    for d in dims:
        prod *= int(d)
    return prod


class StaticAdjacentIndexFunction:
    """Drop-in replacement for `_ascend.ops.index.AscendAdjacentIndexFunction`."""

    def __init__(self):
        self.overloads = {}

    def __call__(self, inp, start_dim, tensor_indices, out):
        import torch
        import triton

        if not tensor_indices or inp.numel() == 0 or out.numel() == 0:
            return inp

        indices_len = len(tensor_indices)
        M = int(tensor_indices[0].numel())
        P = _shape_product(inp.shape[:start_dim])
        V = _shape_product(inp.shape[start_dim : start_dim + indices_len])
        N = _shape_product(inp.shape[start_dim + indices_len :])
        if M == 0 or P == 0 or V == 0 or N == 0:
            return inp

        dims = [int(inp.shape[start_dim + i]) for i in range(indices_len)]
        prep1_k, prep2_k, vec_kernel, small_n_kernel = _get_adj_index_kernels()

        lin_i64 = torch.empty((M,), dtype=torch.int64, device=inp.device)
        if indices_len == 1:
            f0 = tensor_indices[0].reshape(-1).contiguous()
            nb = triton.cdiv(M, 1024)
            np_ = min(nb, 2048)
            prep1_k[(np_,)](f0, lin_i64, M, dims[0], nb, np_, 1024)
        elif indices_len == 2:
            f0 = tensor_indices[0].reshape(-1).contiguous()
            f1 = tensor_indices[1].reshape(-1).contiguous()
            nb = triton.cdiv(M, 1024)
            np_ = min(nb, 2048)
            prep2_k[(np_,)](f0, f1, lin_i64, M, dims[0], dims[1], nb, np_, 1024)
        else:
            lin_idx = None
            valid = None
            for idx_t, d in zip(tensor_indices, dims):
                flat = idx_t.reshape(-1)
                if flat.dtype != torch.int64:
                    flat = flat.to(torch.int64)
                norm = torch.where(flat < 0, flat + d, flat)
                in_bounds = (norm >= 0) & (norm < d)
                norm = torch.where(in_bounds, norm, 0)
                if lin_idx is None:
                    lin_idx = norm
                    valid = in_bounds
                else:
                    lin_idx = lin_idx * d + norm
                    valid = valid & in_bounds
            lin_i64 = torch.where(valid, lin_idx, -1).contiguous()

        if N >= 32:
            R = P * M
            n_progs = min(R, 2048)
            vec_kernel[(n_progs,)](
                inp,
                lin_i64,
                out,
                V,
                M,
                N,
                R,
                n_progs,
                1024,
            )
        else:
            block_m = 256
            m_tiles = triton.cdiv(M, block_m)
            total_tasks = P * m_tiles
            n_progs = min(total_tasks, 4096)
            small_n_kernel[(n_progs,)](
                inp,
                lin_i64,
                out,
                V,
                M,
                N,
                m_tiles,
                total_tasks,
                n_progs,
                block_m,
            )
        return inp


def install_adjacent_index_fix(repairs_log: list[str] | None = None) -> dict:
    """Replace AscendAdjacentIndexFunction with StaticAdjacentIndexFunction on active modules."""
    import flag_gems  # noqa: F401 - ensures SpecOpRegistrar puts _ascend on sys.path

    patched = []
    static_fn = StaticAdjacentIndexFunction()
    for mod_name in ('_ascend.ops.index', 'flag_gems.runtime.backend._ascend.ops.index'):
        try:
            mod = sys.modules.get(mod_name)
            if mod is None and mod_name == '_ascend.ops.index':
                mod = importlib.import_module(mod_name)
            if mod is not None and hasattr(mod, '_adjacent_index_func'):
                mod._adjacent_index_func = static_fn
                patched.append(mod_name)
        except Exception:
            pass

    if repairs_log is not None and patched:
        repairs_log.append(
            "Replace per-shape dynamic AscendAdjacentIndexFunction codegen/LibTuner with static contiguous-DMA persistent 1D-grid 64-bit Triton gather kernels compiled once per dtype."
        )
    return {'patched': patched}
