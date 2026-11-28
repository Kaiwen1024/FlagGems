"""Unified Ascend FlagGems Plan-B runtime repair hooks (524-model verified stack).

Automatically invoked by ``flag_gems.enable()`` / ``flag_gems.use_gems()`` when
``FLAGGEMS_ASCEND_FIXES=1`` (default inside this image). Set
``FLAGGEMS_ASCEND_FIXES=0`` to bypass these hooks for baseline comparison.
"""
from __future__ import annotations

import os
from typing import Any

_PRE_APPLIED = False
_POST_APPLIED = False
_LAST_META: dict[str, Any] = {}


def is_enabled() -> bool:
    return os.environ.get("FLAGGEMS_ASCEND_FIXES", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def apply_pre_enable_fixes() -> dict[str, Any]:
    global _PRE_APPLIED, _LAST_META
    if _PRE_APPLIED or not is_enabled():
        return _LAST_META
    from wave4_repairs import install_wave4_pre_enable_fixes

    repairs_log: list[str] = []
    meta = install_wave4_pre_enable_fixes(repairs_log)
    meta["repairs_log"] = repairs_log
    _LAST_META = meta
    _PRE_APPLIED = True
    return meta


def apply_post_enable_fixes() -> dict[str, Any]:
    global _POST_APPLIED
    if _POST_APPLIED or not is_enabled():
        return _LAST_META
    from wave3_repairs import install_cpu_preprocessing_guard
    from wave3_sparse_sim import install_npu_sparse_similarity_fix

    post_log: list[str] = []
    try:
        install_cpu_preprocessing_guard(post_log)
    except Exception:
        pass
    try:
        install_npu_sparse_similarity_fix(post_log)
    except Exception:
        pass
    _LAST_META["post_repairs_log"] = post_log
    _POST_APPLIED = True
    return _LAST_META


def enable_all() -> dict[str, Any]:
    apply_pre_enable_fixes()
    import flag_gems

    flag_gems.enable()
    apply_post_enable_fixes()
    return _LAST_META

