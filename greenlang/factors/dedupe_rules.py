# -*- coding: utf-8 -*-
"""
Canonical merge / deduplication rules when the same physical activity appears
in multiple upstream sources (per FY27 Factors plan).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

# Same priority as EmissionFactorService.DEFAULT_SOURCE_PRIORITY in Agent Factory.
SOURCE_PRIORITY: List[str] = [
    "epa",
    "defra",
    "iea",
    "ipcc",
    "ecoinvent",
]

MERGE_RULES: Dict[str, Any] = {
    "description": (
        "When two records describe the same activity, geography, and validity window, "
        "keep the record from the highest-priority source in SOURCE_PRIORITY."
    ),
    "tie_breakers": [
        "Prefer more specific geography_level (facility > grid_zone > country > global).",
        "Prefer newer source_year when priority and geography match.",
        "Prefer lower uncertainty_95ci when still tied.",
    ],
    "source_priority": SOURCE_PRIORITY,
}


def duplicate_fingerprint(record: Any) -> str:
    """
    Stable fingerprint for duplicate detection (D5): fuel, geo, scope, boundary,
    methodology, unit, validity start, optional source_record_id.
    """
    key = {
        "fuel": str(getattr(record, "fuel_type", "")).lower(),
        "geo": str(getattr(record, "geography", "")),
        "scope": getattr(getattr(record, "scope", None), "value", str(getattr(record, "scope", ""))),
        "boundary": getattr(getattr(record, "boundary", None), "value", str(getattr(record, "boundary", ""))),
        "methodology": getattr(
            getattr(getattr(record, "provenance", None), "methodology", None), "value", ""
        ),
        "unit": str(getattr(record, "unit", "")),
        "valid_from": str(getattr(record, "valid_from", "")),
        "source_record_id": getattr(record, "source_record_id", None),
    }
    raw = json.dumps(key, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


def sort_sources_for_merge(sources: List[str]) -> List[str]:
    """Return sources ordered by merge preference (best first)."""

    def key(s: str) -> tuple:
        s_low = s.lower()
        try:
            return (0, SOURCE_PRIORITY.index(s_low))
        except ValueError:
            return (1, s_low)

    return sorted(sources, key=key)
