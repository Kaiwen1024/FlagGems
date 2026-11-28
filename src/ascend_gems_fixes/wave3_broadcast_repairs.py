"""Wave 3 broadcast exception semantics repair.

Matches PyTorch ATen semantics by raising RuntimeError instead of ValueError
when FlagGems shape_utils.broadcast detects unbroadcastable tensor shapes
(e.g. allowing downstream_eval vidore._encode_texts to catch RuntimeError and
clear stale Qwen3.5 rope_deltas).
"""
from __future__ import annotations

import functools
import sys


def install_broadcast_runtime_error_fix(repairs_log: list[str] | None = None) -> dict:
    """Wrap flag_gems.utils.shape_utils.broadcast so broadcast errors raise RuntimeError."""
    import flag_gems.utils as fg_utils
    import flag_gems.utils.shape_utils as shape_utils

    orig_broadcast = getattr(shape_utils, 'broadcast', None)
    if orig_broadcast is None:
        return {'patched': [], 'reason': 'missing_broadcast'}

    if not getattr(orig_broadcast, '_wave3_runtime_err_patched', False):
        @functools.wraps(orig_broadcast)
        def _broadcast_runtime_err(s1, s2):
            try:
                return orig_broadcast(s1, s2)
            except ValueError as exc:
                raise RuntimeError(
                    f"The size of tensor shapes {s1} and {s2} are not broadcastable: {exc}"
                ) from exc

        _broadcast_runtime_err._wave3_runtime_err_patched = True
        shape_utils.broadcast = _broadcast_runtime_err

        updated = ['flag_gems.utils.shape_utils.broadcast']
        for mod_name, mod in list(sys.modules.items()):
            if mod is None or not (mod_name == 'flag_gems' or mod_name.startswith('flag_gems.')):
                continue
            if mod is shape_utils:
                continue
            if getattr(mod, 'broadcast', None) is orig_broadcast:
                setattr(mod, 'broadcast', _broadcast_runtime_err)
                updated.append(f'{mod_name}.broadcast')
    else:
        updated = ['flag_gems.utils.shape_utils.broadcast(already)']

    if getattr(fg_utils, 'broadcast', None) is orig_broadcast:
        setattr(fg_utils, 'broadcast', shape_utils.broadcast)

    if repairs_log is not None:
        repairs_log.append(
            "Wrap flag_gems.utils.shape_utils.broadcast to raise RuntimeError instead of ValueError on unbroadcastable shapes (matching PyTorch ATen semantics)."
        )
    return {'patched': updated}
