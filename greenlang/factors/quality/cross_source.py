# -*- coding: utf-8 -*-
"""
Cross-source consistency checker (F022).

Compares emission factors from different sources for the same activity
(fuel_type + geography + scope). Flags discrepancies:
- >20% divergence: WARNING (log, include in report)
- >50% divergence: REVIEW_REQUIRED (methodology lead must approve)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Thresholds
WARNING_THRESHOLD = 0.20
REVIEW_THRESHOLD = 0.50


@dataclass
class SourceValue:
    """A single factor value from a specific source."""

    factor_id: str
    source_id: str
    co2_value: float
    co2e_total: float
    unit: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "source_id": self.source_id,
            "co2_value": self.co2_value,
            "co2e_total": self.co2e_total,
            "unit": self.unit,
        }


@dataclass
class ConsistencyCheck:
    """Result of comparing factors across sources for one activity."""

    activity_key: str
    fuel_type: str
    geography: str
    scope: str
    sources: List[SourceValue] = field(default_factory=list)
    max_divergence: float = 0.0
    severity: str = "ok"  # "ok" | "warning" | "review_required"
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "activity_key": self.activity_key,
            "fuel_type": self.fuel_type,
            "geography": self.geography,
            "scope": self.scope,
            "sources": [s.to_dict() for s in self.sources],
            "max_divergence": round(self.max_divergence, 4),
            "severity": self.severity,
            "detail": self.detail,
        }


@dataclass
class ConsistencyReport:
    """Full cross-source consistency report for an edition."""

    edition_id: str
    total_activities: int = 0
    total_ok: int = 0
    total_warnings: int = 0
    total_reviews: int = 0
    checks: List[ConsistencyCheck] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edition_id": self.edition_id,
            "total_activities": self.total_activities,
            "total_ok": self.total_ok,
            "total_warnings": self.total_warnings,
            "total_reviews": self.total_reviews,
            "checks": [c.to_dict() for c in self.checks],
        }

    @property
    def has_issues(self) -> bool:
        return self.total_warnings > 0 or self.total_reviews > 0

    @property
    def needs_review(self) -> bool:
        return self.total_reviews > 0


def _extract_activity_key(factor: Any) -> str:
    """Build activity key from factor (dict or record)."""
    if isinstance(factor, dict):
        fuel = str(factor.get("fuel_type", "")).lower().strip()
        geo = str(factor.get("geography", "")).upper().strip()
        scope = str(factor.get("scope", "")).strip()
    else:
        fuel = str(getattr(factor, "fuel_type", "")).lower().strip()
        geo = str(getattr(factor, "geography", "")).upper().strip()
        scope = str(getattr(getattr(factor, "scope", None), "value", getattr(factor, "scope", ""))).strip()
    return f"{fuel}|{geo}|{scope}"


def _extract_source_value(factor: Any) -> SourceValue:
    """Extract source value from a factor dict or record."""
    if isinstance(factor, dict):
        fid = factor.get("factor_id", "")
        sid = factor.get("source_id") or ""
        if not sid:
            parts = fid.split(":")
            sid = parts[1].lower() if len(parts) >= 2 else ""
        vectors = factor.get("vectors") or {}
        co2 = float(vectors.get("CO2") or 0)
        gwp = factor.get("gwp_100yr") or {}
        co2e = float(gwp.get("co2e_total") or co2)
        unit = str(factor.get("unit", ""))
    else:
        fid = getattr(factor, "factor_id", "")
        sid = getattr(factor, "source_id", "") or ""
        if not sid:
            parts = fid.split(":")
            sid = parts[1].lower() if len(parts) >= 2 else ""
        vectors = getattr(factor, "vectors", None)
        co2 = float(getattr(vectors, "CO2", 0) if vectors else 0)
        gwp = getattr(factor, "gwp_100yr", None)
        co2e = float(getattr(gwp, "co2e_total", co2) if gwp else co2)
        unit = str(getattr(factor, "unit", ""))

    return SourceValue(
        factor_id=fid,
        source_id=sid,
        co2_value=co2,
        co2e_total=co2e,
        unit=unit,
    )


def _compute_divergence(values: List[float]) -> float:
    """
    Compute maximum pairwise relative divergence.

    Returns 0.0 if all values are zero or only one value.
    """
    if len(values) < 2:
        return 0.0
    non_zero = [v for v in values if v > 0]
    if len(non_zero) < 2:
        return 0.0
    min_v = min(non_zero)
    max_v = max(non_zero)
    if min_v == 0:
        return 0.0
    return (max_v - min_v) / min_v


def check_cross_source_consistency(
    factors: Sequence[Any],
    *,
    edition_id: str = "unknown",
    warning_threshold: float = WARNING_THRESHOLD,
    review_threshold: float = REVIEW_THRESHOLD,
    min_sources: int = 2,
) -> ConsistencyReport:
    """
    Compare factors from different sources for the same activity.

    Args:
        factors: List of factor dicts or EmissionFactorRecord objects.
        edition_id: Edition label for reporting.
        warning_threshold: Relative divergence threshold for warnings (default 0.20).
        review_threshold: Relative divergence threshold for human review (default 0.50).
        min_sources: Minimum number of distinct sources to compare (default 2).

    Returns:
        ConsistencyReport with per-activity comparisons.
    """
    report = ConsistencyReport(edition_id=edition_id)

    # Group by activity key
    groups: Dict[str, List[Any]] = defaultdict(list)
    for f in factors:
        ak = _extract_activity_key(f)
        groups[ak].append(f)

    for ak, group in sorted(groups.items()):
        # Extract source values
        source_values = [_extract_source_value(f) for f in group]

        # Only compare if multiple distinct sources
        distinct_sources = set(sv.source_id for sv in source_values if sv.source_id)
        if len(distinct_sources) < min_sources:
            continue

        report.total_activities += 1

        # Parse activity key
        parts = ak.split("|")
        fuel = parts[0] if len(parts) > 0 else ""
        geo = parts[1] if len(parts) > 1 else ""
        scope = parts[2] if len(parts) > 2 else ""

        # Compute divergence on CO2 values
        co2_values = [sv.co2_value for sv in source_values if sv.co2_value > 0]
        divergence = _compute_divergence(co2_values)

        # Classify severity
        if divergence >= review_threshold:
            severity = "review_required"
            detail = (
                f"CO2 divergence {divergence:.1%} exceeds review threshold {review_threshold:.0%}. "
                f"Sources: {', '.join(sorted(distinct_sources))}"
            )
            report.total_reviews += 1
        elif divergence >= warning_threshold:
            severity = "warning"
            detail = (
                f"CO2 divergence {divergence:.1%} exceeds warning threshold {warning_threshold:.0%}. "
                f"Sources: {', '.join(sorted(distinct_sources))}"
            )
            report.total_warnings += 1
        else:
            severity = "ok"
            detail = f"CO2 divergence {divergence:.1%} within tolerance"
            report.total_ok += 1

        check = ConsistencyCheck(
            activity_key=ak,
            fuel_type=fuel,
            geography=geo,
            scope=scope,
            sources=source_values,
            max_divergence=divergence,
            severity=severity,
            detail=detail,
        )
        report.checks.append(check)

    logger.info(
        "Cross-source consistency: edition=%s activities=%d ok=%d warnings=%d reviews=%d",
        edition_id, report.total_activities, report.total_ok,
        report.total_warnings, report.total_reviews,
    )
    return report
