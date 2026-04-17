# -*- coding: utf-8 -*-
"""Methodology tags separate from raw numeric vectors (S4)."""

from __future__ import annotations

from typing import Any, List


def methodology_tags_for_record(record: Any) -> List[str]:
    tags: List[str] = []
    prov = getattr(record, "provenance", None)
    if prov is not None:
        m = getattr(prov, "methodology", None)
        if m is not None:
            tags.append(f"method:{m.value}")
        b = getattr(record, "boundary", None)
        if b is not None:
            tags.append(f"boundary:{b.value}")
    return tags
