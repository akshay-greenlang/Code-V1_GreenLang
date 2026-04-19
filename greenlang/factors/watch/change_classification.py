# -*- coding: utf-8 -*-
"""
Change classification routing (U3).

Classifies changes between two versions of an emission factor into
well-defined categories so that downstream consumers (changelog generator,
release orchestrator, notification dispatcher) can act on them
appropriately.

Each ChangeType carries a severity flag:
- breaking: consumers MUST update (numeric corrections, deprecations)
- non-breaking: informational only (metadata updates, new source years)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ChangeType(str, Enum):
    """Classification of a factor-level change between editions."""

    NUMERIC_CORRECTION = "numeric_correction"
    METADATA_UPDATE = "metadata_update"
    METHODOLOGY_CHANGE = "methodology_change"
    NEW_SOURCE_YEAR = "new_source_year"
    DEPRECATION = "deprecation"
    SCOPE_EXPANSION = "scope_expansion"


# Fields whose change signals a numeric correction
_NUMERIC_FIELDS = {"co2e_total", "CO2", "CH4", "N2O", "HFCs", "PFCs", "SF6", "NF3"}


@dataclass
class ChangeClassification:
    """Result of classifying a single factor change."""

    change_type: ChangeType
    is_breaking: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON-friendly output."""
        return {
            "change_type": self.change_type.value,
            "is_breaking": self.is_breaking,
            "reason": self.reason,
        }


def _get_nested(row: Dict[str, Any], dotted_key: str) -> Any:
    """Safely retrieve a value from a nested dict using a dotted path."""
    parts = dotted_key.split(".")
    current: Any = row
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _has_numeric_change(old_row: Dict[str, Any], new_row: Dict[str, Any]) -> bool:
    """Return True if any numeric emission value field differs."""
    for field_name in _NUMERIC_FIELDS:
        # Check top-level
        old_val = old_row.get(field_name)
        new_val = new_row.get(field_name)
        if old_val != new_val and old_val is not None and new_val is not None:
            return True
        # Check inside gwp_100yr / gwp_20yr / vectors sub-dicts
        for container in ("gwp_100yr", "gwp_20yr", "vectors"):
            old_nested = _get_nested(old_row, f"{container}.{field_name}")
            new_nested = _get_nested(new_row, f"{container}.{field_name}")
            if old_nested != new_nested and old_nested is not None and new_nested is not None:
                return True
    return False


def _has_methodology_change(old_row: Dict[str, Any], new_row: Dict[str, Any]) -> bool:
    """Return True if methodology or gwp_set changed."""
    old_meth = _get_nested(old_row, "provenance.methodology") or old_row.get("methodology")
    new_meth = _get_nested(new_row, "provenance.methodology") or new_row.get("methodology")
    if old_meth != new_meth and old_meth is not None and new_meth is not None:
        return True

    old_gwp = old_row.get("gwp_set")
    new_gwp = new_row.get("gwp_set")
    if old_gwp != new_gwp and old_gwp is not None and new_gwp is not None:
        return True

    return False


def _has_source_year_change(old_row: Dict[str, Any], new_row: Dict[str, Any]) -> bool:
    """Return True if source_year changed (indicating a new publication year)."""
    old_year = _get_nested(old_row, "provenance.source_year") or old_row.get("source_year")
    new_year = _get_nested(new_row, "provenance.source_year") or new_row.get("source_year")
    if old_year != new_year and old_year is not None and new_year is not None:
        return True
    return False


def _is_deprecation(old_row: Dict[str, Any], new_row: Dict[str, Any]) -> bool:
    """Return True if the factor has been deprecated (replacement set or status changed)."""
    old_status = old_row.get("factor_status", "certified")
    new_status = new_row.get("factor_status", "certified")
    if new_status == "deprecated" and old_status != "deprecated":
        return True

    old_replacement = old_row.get("replacement_factor_id")
    new_replacement = new_row.get("replacement_factor_id")
    if new_replacement and not old_replacement:
        return True

    return False


def _is_scope_expansion(old_row: Dict[str, Any], new_row: Dict[str, Any]) -> bool:
    """Return True if the factor's scope or boundary expanded."""
    old_scope = old_row.get("scope")
    new_scope = new_row.get("scope")
    if old_scope != new_scope and old_scope is not None and new_scope is not None:
        return True

    old_boundary = old_row.get("boundary")
    new_boundary = new_row.get("boundary")
    if old_boundary != new_boundary and old_boundary is not None and new_boundary is not None:
        return True

    return False


def classify_change(
    *,
    old_hash: str,
    new_hash: str,
    old_row: Dict[str, Any],
    new_row: Dict[str, Any],
) -> ChangeClassification:
    """
    Classify a change between two versions of the same factor.

    Priority order (first match wins):
        1. Deprecation (breaking)
        2. Numeric correction (breaking)
        3. Methodology change (breaking)
        4. Scope expansion (non-breaking)
        5. New source year (non-breaking)
        6. Metadata update (non-breaking, fallback)

    Args:
        old_hash: Content hash of the old factor version.
        new_hash: Content hash of the new factor version.
        old_row: Dict representation of the old factor.
        new_row: Dict representation of the new factor.

    Returns:
        ChangeClassification with type, severity, and reason.
    """
    if old_hash == new_hash and old_row == new_row:
        # Identical -- treat as metadata update with no real diff
        return ChangeClassification(
            change_type=ChangeType.METADATA_UPDATE,
            is_breaking=False,
            reason="No detectable change (hashes and content identical)",
        )

    # 1. Deprecation check
    if _is_deprecation(old_row, new_row):
        replacement = new_row.get("replacement_factor_id", "none")
        logger.info(
            "Factor classified as DEPRECATION: replacement=%s",
            replacement,
        )
        return ChangeClassification(
            change_type=ChangeType.DEPRECATION,
            is_breaking=True,
            reason=f"Factor deprecated; replacement={replacement}",
        )

    # 2. Numeric correction check
    if _has_numeric_change(old_row, new_row):
        logger.info("Factor classified as NUMERIC_CORRECTION")
        return ChangeClassification(
            change_type=ChangeType.NUMERIC_CORRECTION,
            is_breaking=True,
            reason="Emission factor numeric values changed",
        )

    # 3. Methodology change check
    if _has_methodology_change(old_row, new_row):
        logger.info("Factor classified as METHODOLOGY_CHANGE")
        return ChangeClassification(
            change_type=ChangeType.METHODOLOGY_CHANGE,
            is_breaking=True,
            reason="Methodology or GWP set changed",
        )

    # 4. Scope expansion check
    if _is_scope_expansion(old_row, new_row):
        logger.info("Factor classified as SCOPE_EXPANSION")
        return ChangeClassification(
            change_type=ChangeType.SCOPE_EXPANSION,
            is_breaking=False,
            reason="Scope or boundary expanded",
        )

    # 5. New source year check
    if _has_source_year_change(old_row, new_row):
        logger.info("Factor classified as NEW_SOURCE_YEAR")
        return ChangeClassification(
            change_type=ChangeType.NEW_SOURCE_YEAR,
            is_breaking=False,
            reason="Source publication year updated",
        )

    # 6. Fallback: metadata update
    logger.info("Factor classified as METADATA_UPDATE")
    return ChangeClassification(
        change_type=ChangeType.METADATA_UPDATE,
        is_breaking=False,
        reason="Non-numeric metadata fields changed",
    )
