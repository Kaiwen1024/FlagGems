"""Process-local FlagGems LibTuner & Triton Autotuner HBM leak fix for Wave 3.

Root cause (verified on Qwen3-VL-Embedding-2B c007):
  1. flag_gems/utils/libentry.py LibTuner.run assigns:
       self._last_benchmark_args = tuple(args)
       self._last_benchmark_meta = dict(kwargs)
       self.nargs = dict(zip(self.arg_names, args))
     `_last_benchmark_args` and `_last_benchmark_meta` are NEVER cleared after
     `run()` returns, and `self.nargs` is not cleared on exception paths.
  2. When ops like `_ascend/ops/index.py` (AscendAdjacentIndexFunction) generate
     one dynamic Python module + LibTuner instance per input shape, each instance
     permanently pins all input/output NPU tensors in `_last_benchmark_args`,
     leaking ~1.16 GiB/batch (~60 GiB total before OOM). Global single-instance
     LibTuner kernels (e.g. mul) also pin the last large activation tensor (~0.9 GiB).
"""
from __future__ import annotations

import functools
import importlib
import sys


def install_libtuner_memory_fix(repairs_log: list[str] | None = None) -> dict:
    """Clear LibTuner / Autotuner tensor references in a finally block after run()."""
    patched: list[str] = []

    importlib.import_module('flag_gems.utils.libentry')
    le = sys.modules['flag_gems.utils.libentry']
    lt_cls = getattr(le, 'LibTuner', None)
    if lt_cls is not None and not getattr(lt_cls.run, '_wave3_mem_patched', False):
        orig_lt_run = lt_cls.run

        @functools.wraps(orig_lt_run)
        def _lt_run_clean(self, *args, **kwargs):
            try:
                return orig_lt_run(self, *args, **kwargs)
            finally:
                self.nargs = None
                self._last_benchmark_args = None
                self._last_benchmark_meta = None

        _lt_run_clean._wave3_mem_patched = True
        lt_cls.run = _lt_run_clean
        patched.append('flag_gems.utils.libentry.LibTuner.run')

    seen_auto: set[int] = set()
    for mod_name in ('triton.runtime.autotuner', 'triton.spec.ascend.runtime.autotuner'):
        try:
            mod = importlib.import_module(mod_name)
            at_cls = getattr(mod, 'Autotuner', None)
            if at_cls is not None and id(at_cls) not in seen_auto:
                seen_auto.add(id(at_cls))
                if not getattr(at_cls.run, '_wave3_mem_patched', False):
                    orig_at_run = at_cls.run

                    @functools.wraps(orig_at_run)
                    def _at_run_clean(self, *args, **kwargs):
                        try:
                            return orig_at_run(self, *args, **kwargs)
                        finally:
                            self.nargs = None

                    _at_run_clean._wave3_mem_patched = True
                    at_cls.run = _at_run_clean
                    patched.append(f'{mod_name}.Autotuner.run')
        except Exception:
            pass

    if repairs_log is not None and patched:
        repairs_log.append(
            "Release FlagGems LibTuner (_last_benchmark_args/_last_benchmark_meta/nargs) and Triton Autotuner (nargs) tensor references after kernel launch to eliminate dynamic-shape HBM leaks."
        )
    return {'patched': patched}
