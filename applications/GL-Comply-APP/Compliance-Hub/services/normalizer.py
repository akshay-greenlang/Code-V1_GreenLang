# -*- coding: utf-8 -*-
"""Cross-framework input normalizer.

Deduplicates shared inputs across frameworks so data is submitted once. Current
implementation:
- Activity dedup by (activity_id, year, fuel_type, unit) composite key
- Lowercase trim on entity jurisdiction
- No-op entity resolution (full entity_graph wiring in PLATFORM 1, task #25)
- No-op unit normalization (reuse AGENT-FOUND-003 wiring to be added)
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from schemas.models import ComplianceRequest

logger = logging.getLogger(__name__)


def normalize(request: ComplianceRequest) -> ComplianceRequest:
    """Return a canonicalized ComplianceRequest (new object; input untouched)."""
    normalized = request.model_copy(deep=True)
    normalized = _normalize_entity(normalized)
    normalized = _dedupe_activities(normalized)
    return normalized


def _normalize_entity(request: ComplianceRequest) -> ComplianceRequest:
    jurisdiction = request.entity.jurisdiction.strip().upper()
    request.entity.jurisdiction = jurisdiction
    return request


def _dedupe_activities(request: ComplianceRequest) -> ComplianceRequest:
    if not request.data_sources:
        return request
    activities = request.data_sources.get("activities")
    if not activities or not isinstance(activities, list):
        return request

    seen: dict[tuple, dict[str, Any]] = {}
    for item in activities:
        key = _activity_key(item)
        if key in seen:
            logger.debug("Deduplicated activity %s", key)
            continue
        seen[key] = item

    data_sources = deepcopy(request.data_sources)
    data_sources["activities"] = list(seen.values())
    request.data_sources = data_sources
    return request


def _activity_key(item: dict[str, Any]) -> tuple:
    return (
        item.get("activity_id"),
        item.get("year"),
        item.get("fuel_type") or item.get("activity_type"),
        item.get("unit"),
    )
