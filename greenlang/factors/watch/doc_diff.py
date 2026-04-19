# -*- coding: utf-8 -*-
"""
Standards page / document diff hook (U2).

Provides structured diffing between two factor versions at the field level,
deterministic fingerprinting for change detection, and human-readable
summaries of multi-factor change sets.

Production deployments may plug in HTML/PDF extractors; this module keeps a
stable deterministic interface for watch pipelines.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FieldDiff:
    """A single field-level difference between two factor versions."""

    field_path: str
    diff_type: str  # "added", "removed", "changed"
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        d: Dict[str, Any] = {
            "field_path": self.field_path,
            "diff_type": self.diff_type,
        }
        if self.diff_type in ("removed", "changed"):
            d["old_value"] = self.old_value
        if self.diff_type in ("added", "changed"):
            d["new_value"] = self.new_value
        return d


@dataclass
class DocDiff:
    """Structured diff between two factor versions."""

    factor_id: str
    left_fingerprint: str
    right_fingerprint: str
    is_changed: bool
    field_diffs: List[FieldDiff] = field(default_factory=list)

    @property
    def changed_field_count(self) -> int:
        """Number of fields that differ."""
        return len(self.field_diffs)

    @property
    def has_numeric_changes(self) -> bool:
        """True if any diff involves a numeric emission value."""
        numeric_paths = {
            "co2e_total", "CO2", "CH4", "N2O", "HFCs", "PFCs", "SF6", "NF3",
            "gwp_100yr.CO2", "gwp_100yr.CH4", "gwp_100yr.N2O",
            "gwp_100yr.co2e_total",
            "gwp_20yr.CO2", "gwp_20yr.CH4", "gwp_20yr.N2O",
            "gwp_20yr.co2e_total",
            "vectors.CO2", "vectors.CH4", "vectors.N2O",
        }
        return any(fd.field_path in numeric_paths for fd in self.field_diffs)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "factor_id": self.factor_id,
            "left_fingerprint": self.left_fingerprint,
            "right_fingerprint": self.right_fingerprint,
            "is_changed": self.is_changed,
            "changed_field_count": self.changed_field_count,
            "has_numeric_changes": self.has_numeric_changes,
            "field_diffs": [fd.to_dict() for fd in self.field_diffs],
        }


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


def fingerprint_text(text: str) -> str:
    """Compute a SHA-256 fingerprint of raw text content."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def fingerprint_factor(factor: Any) -> str:
    """
    Compute a deterministic SHA-256 fingerprint for a factor record.

    Works with both dict and dataclass/object factor representations.
    Fields are sorted alphabetically to ensure identical factors always
    produce the same fingerprint regardless of insertion order.

    Args:
        factor: An EmissionFactorRecord, dict, or any object with a
                ``to_dict()`` method.

    Returns:
        64-character hex SHA-256 digest string.
    """
    if hasattr(factor, "to_dict") and callable(factor.to_dict):
        data = factor.to_dict()
    elif isinstance(factor, dict):
        data = factor
    else:
        data = {"repr": repr(factor)}

    payload = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Plain-text diff (backward-compatible)
# ---------------------------------------------------------------------------


def diff_text_versions(old_text: str, new_text: str) -> Tuple[bool, str]:
    """
    Return (changed, summary). When bodies differ, summary is a short machine note.

    Retained for backward compatibility with source_watch and scheduler modules.
    """
    if old_text == new_text:
        return False, "unchanged"
    return True, f"length_delta={len(new_text) - len(old_text)}"


# ---------------------------------------------------------------------------
# Structured diff engine
# ---------------------------------------------------------------------------


def _diff_dicts(
    left: Dict[str, Any],
    right: Dict[str, Any],
    prefix: str = "",
) -> List[FieldDiff]:
    """Recursively diff two dicts, returning a list of FieldDiff objects."""
    diffs: List[FieldDiff] = []
    all_keys = set(left.keys()) | set(right.keys())

    for key in sorted(all_keys):
        path = f"{prefix}.{key}" if prefix else key
        lv = left.get(key)
        rv = right.get(key)

        if key not in left:
            diffs.append(FieldDiff(
                field_path=path,
                diff_type="added",
                new_value=rv,
            ))
        elif key not in right:
            diffs.append(FieldDiff(
                field_path=path,
                diff_type="removed",
                old_value=lv,
            ))
        elif isinstance(lv, dict) and isinstance(rv, dict):
            diffs.extend(_diff_dicts(lv, rv, prefix=path))
        elif lv != rv:
            diffs.append(FieldDiff(
                field_path=path,
                diff_type="changed",
                old_value=lv,
                new_value=rv,
            ))

    return diffs


def _factor_to_dict(factor: Any) -> Dict[str, Any]:
    """Normalize a factor (object or dict) into a dict."""
    if isinstance(factor, dict):
        return factor
    if hasattr(factor, "to_dict") and callable(factor.to_dict):
        return factor.to_dict()
    return {}


def generate_doc_diff(left_factor: Any, right_factor: Any) -> DocDiff:
    """
    Generate a structured field-level diff between two factor versions.

    Computes fingerprints for both sides and recursively compares all
    fields.  Works with EmissionFactorRecord objects, dicts, or any
    object that implements ``to_dict()``.

    Args:
        left_factor: The older factor version.
        right_factor: The newer factor version.

    Returns:
        DocDiff with field-level differences and fingerprints.
    """
    left_dict = _factor_to_dict(left_factor)
    right_dict = _factor_to_dict(right_factor)

    left_fp = fingerprint_factor(left_factor)
    right_fp = fingerprint_factor(right_factor)

    factor_id = (
        left_dict.get("factor_id")
        or right_dict.get("factor_id")
        or "unknown"
    )

    if left_fp == right_fp:
        return DocDiff(
            factor_id=factor_id,
            left_fingerprint=left_fp,
            right_fingerprint=right_fp,
            is_changed=False,
        )

    field_diffs = _diff_dicts(left_dict, right_dict)

    logger.info(
        "Doc diff for %s: %d field changes detected",
        factor_id,
        len(field_diffs),
    )

    return DocDiff(
        factor_id=factor_id,
        left_fingerprint=left_fp,
        right_fingerprint=right_fp,
        is_changed=True,
        field_diffs=field_diffs,
    )


# ---------------------------------------------------------------------------
# Multi-diff summary
# ---------------------------------------------------------------------------


def summarize_changes(diffs: List[DocDiff]) -> str:
    """
    Generate a human-readable summary from a list of DocDiff objects.

    Groups changes into numeric vs non-numeric and provides per-factor
    detail lines.

    Args:
        diffs: List of DocDiff objects (one per factor).

    Returns:
        Multi-line string summary.
    """
    if not diffs:
        return "No changes detected."

    changed = [d for d in diffs if d.is_changed]
    unchanged = len(diffs) - len(changed)

    lines: List[str] = []
    lines.append(f"Change summary: {len(changed)} changed, {unchanged} unchanged")

    numeric_changes: List[DocDiff] = []
    metadata_changes: List[DocDiff] = []

    for d in changed:
        if d.has_numeric_changes:
            numeric_changes.append(d)
        else:
            metadata_changes.append(d)

    if numeric_changes:
        lines.append("")
        lines.append(f"Numeric corrections ({len(numeric_changes)}):")
        for d in numeric_changes:
            numeric_fields = [
                fd for fd in d.field_diffs
                if fd.diff_type == "changed"
                and any(nf in fd.field_path for nf in ("co2e", "CO2", "CH4", "N2O"))
            ]
            detail_parts = []
            for fd in numeric_fields[:3]:  # Limit detail lines per factor
                detail_parts.append(
                    f"{fd.field_path}: {fd.old_value} -> {fd.new_value}"
                )
            detail = "; ".join(detail_parts) if detail_parts else f"{d.changed_field_count} fields"
            lines.append(f"  - {d.factor_id}: {detail}")

    if metadata_changes:
        lines.append("")
        lines.append(f"Metadata updates ({len(metadata_changes)}):")
        for d in metadata_changes:
            lines.append(f"  - {d.factor_id}: {d.changed_field_count} fields changed")

    return "\n".join(lines)
