# -*- coding: utf-8 -*-
"""
Standards page / document diff hook (U2).

Production deployments may plug in HTML/PDF extractors; this module keeps a stable
deterministic interface for watch pipelines.
"""

from __future__ import annotations

import hashlib
from typing import Tuple


def fingerprint_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def diff_text_versions(old_text: str, new_text: str) -> Tuple[bool, str]:
    """
    Return (changed, summary). When bodies differ, summary is a short machine note.
    """
    if old_text == new_text:
        return False, "unchanged"
    return True, f"length_delta={len(new_text) - len(old_text)}"
