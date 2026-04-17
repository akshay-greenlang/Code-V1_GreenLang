# -*- coding: utf-8 -*-
"""Change classification routing (U3)."""

from __future__ import annotations

from typing import Any, Dict


def classify_change(*, old_hash: str, new_hash: str, old_row: Dict[str, Any], new_row: Dict[str, Any]) -> str:
    if old_hash != new_hash:
        return "numeric_or_vectors"
    if old_row != new_row:
        return "metadata"
    return "docs_only"
