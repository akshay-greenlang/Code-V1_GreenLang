# -*- coding: utf-8 -*-
"""Process-wide method-pack registry (Phase F2)."""
from __future__ import annotations

import logging
import threading
from typing import Dict, List

from greenlang.data.canonical_v2 import MethodProfile
from greenlang.factors.method_packs.base import MethodPack

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_packs: Dict[MethodProfile, MethodPack] = {}


class MethodPackNotFound(KeyError):
    """Raised when :func:`get_pack` is asked for an unregistered profile."""


def register_pack(pack: MethodPack) -> None:
    """Register a method pack.  Idempotent on (profile, pack_version)."""
    with _lock:
        existing = _packs.get(pack.profile)
        if existing is not None and existing.pack_version != pack.pack_version:
            logger.warning(
                "Method pack %s version bumped from %s to %s",
                pack.profile.value, existing.pack_version, pack.pack_version,
            )
        _packs[pack.profile] = pack
        logger.info("Registered method pack: %s v%s", pack.profile.value, pack.pack_version)


def get_pack(profile: MethodProfile) -> MethodPack:
    """Retrieve a registered method pack.  Raises on unknown profile."""
    with _lock:
        pack = _packs.get(profile)
    if pack is None:
        raise MethodPackNotFound(
            "no method pack registered for profile %r; available: %s"
            % (profile, sorted(p.value for p in _packs))
        )
    return pack


def list_packs() -> List[MethodPack]:
    """Return every registered pack, sorted by profile name."""
    with _lock:
        return sorted(_packs.values(), key=lambda p: p.profile.value)


def registered_profiles() -> List[MethodProfile]:
    with _lock:
        return sorted(_packs.keys(), key=lambda p: p.value)


__all__ = [
    "MethodPackNotFound",
    "register_pack",
    "get_pack",
    "list_packs",
    "registered_profiles",
]
