# -*- coding: utf-8 -*-
"""Geography tokens (S3): ISO-3166 alpha-2 and common grid / corridor hints."""

from __future__ import annotations

import re

_ISO2 = re.compile(r"^[A-Z]{2}$")


def is_iso3166_alpha2(code: str) -> bool:
    c = (code or "").strip().upper()
    return bool(_ISO2.match(c))


def normalize_grid_token(token: str) -> str:
    """Normalize eGRID-style or market names for indexing (lossy, display-safe)."""
    return re.sub(r"[^a-zA-Z0-9_]+", "_", (token or "").strip()).strip("_").lower() or "unknown"
