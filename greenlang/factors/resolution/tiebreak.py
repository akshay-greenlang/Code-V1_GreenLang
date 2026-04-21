# -*- coding: utf-8 -*-
"""Tie-break scoring for the Resolution Engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional


@dataclass
class TieBreakReasons:
    """Structured record of why the engine picked this factor over its alternates.

    Lower ``score`` wins.  Each sub-score is 0 (best) to 10 (worst).
    """

    geography_distance: int = 0       # 0 = exact match, up to 10 = GLOBAL fallback
    time_distance: int = 0            # 0 = current vintage, 10 = very stale
    time_granularity_penalty: int = 0 # 0 = matches requested granularity, 10 = coarser than requested
    technology_match: int = 0
    unit_compatibility: int = 0
    methodology_compatibility: int = 0
    source_authority_rank: int = 0    # 0 = authoritative, 10 = aggregator
    verification_penalty: int = 0     # 0 verified, 10 unverified
    uncertainty_penalty: int = 0
    recency_penalty: int = 0
    license_penalty: int = 0          # 0 preferred class, 10 connector-only
    notes: List[str] = field(default_factory=list)

    def score(self) -> int:
        return (
            self.geography_distance
            + self.time_distance
            + self.time_granularity_penalty
            + self.technology_match
            + self.unit_compatibility
            + self.methodology_compatibility
            + self.source_authority_rank
            + self.verification_penalty
            + self.uncertainty_penalty
            + self.recency_penalty
            + self.license_penalty
        )

    def one_liner(self) -> str:
        parts: List[str] = []
        if self.geography_distance == 0:
            parts.append("exact-geography")
        else:
            parts.append(f"geo-distance={self.geography_distance}")
        if self.time_distance <= 1:
            parts.append("current-vintage")
        if self.verification_penalty == 0:
            parts.append("verified")
        if self.license_penalty == 0:
            parts.append("open-license")
        if self.notes:
            parts.append("; ".join(self.notes))
        return ", ".join(parts)


# --------------------------------------------------------------------------
# Scoring helpers
# --------------------------------------------------------------------------


def score_geography(record_geo: Optional[str], request_geo: Optional[str]) -> int:
    if not record_geo:
        return 5
    if not request_geo:
        return 2
    rg = record_geo.upper()
    req = request_geo.upper()
    if rg == req:
        return 0
    if rg in req or req in rg:
        return 2
    if rg == "GLOBAL":
        return 8
    return 6


def score_time(valid_to: Optional[date], request_date: date) -> int:
    if valid_to is None:
        return 0
    if valid_to >= request_date:
        return 0
    # Stale — 1 point per full year.
    delta_days = (request_date - valid_to).days
    return min(10, delta_days // 365 + 1)


def score_source_authority(source_id: Optional[str]) -> int:
    """Cheap heuristic: authoritative regulators rank best."""
    if not source_id:
        return 5
    sid = source_id.lower()
    if sid in {"epa_hub", "desnz_ghg_conversion", "india_cea_co2_baseline", "ipcc_defaults"}:
        return 0
    if sid in {"egrid", "aib_residual_mix_eu", "green_e"}:
        return 1
    if sid in {"ghg_protocol", "tcr_grp_defaults"}:
        return 2
    return 4


def score_verification(record: Any) -> int:
    v = getattr(record, "verification", None)
    if v is None:
        return 5
    status = str(getattr(v, "status", "unverified"))
    mapping = {
        "regulator_approved": 0,
        "external_verified": 1,
        "internal_review": 4,
        "unverified": 8,
    }
    return mapping.get(status, 5)


_GRANULARITY_ORDER = {
    "hourly": 0,
    "daily": 1,
    "seasonal": 1,
    "monthly": 2,
    "quarterly": 3,
    "annual": 4,
    "multi_year": 5,
}


def score_time_granularity(
    record_granularity: Optional[str],
    requested_granularity: Optional[str],
) -> int:
    """Penalty when the record's time granularity is coarser than requested.

    Zero when the record is at least as fine-grained as requested; rises
    linearly as the record becomes coarser.  Unknown record granularity
    is treated as "annual".
    """
    if not requested_granularity:
        return 0
    req = requested_granularity.lower()
    rec = (record_granularity or "annual").lower()
    req_rank = _GRANULARITY_ORDER.get(req, 4)
    rec_rank = _GRANULARITY_ORDER.get(rec, 4)
    # Record finer than requested — that's fine, no penalty.
    if rec_rank <= req_rank:
        return 0
    # Coarser — each step adds 2 points, capped at 10.
    return min(10, (rec_rank - req_rank) * 2)


def score_license(redistribution_class: Optional[str]) -> int:
    if not redistribution_class:
        return 2
    mapping = {
        "open": 0,
        "restricted": 2,
        "oem_redistributable": 2,
        "customer_private": 3,
        "licensed": 4,
    }
    return mapping.get(str(redistribution_class), 5)


def build_tiebreak(
    record: Any,
    *,
    request_geo: Optional[str],
    request_date: date,
    request_granularity: Optional[str] = None,
) -> TieBreakReasons:
    tb = TieBreakReasons()
    tb.geography_distance = score_geography(
        getattr(record, "geography", None), request_geo
    )
    tb.time_distance = score_time(getattr(record, "valid_to", None), request_date)
    tb.time_granularity_penalty = score_time_granularity(
        getattr(record, "time_granularity", None), request_granularity
    )
    tb.source_authority_rank = score_source_authority(
        getattr(record, "source_id", None)
    )
    tb.verification_penalty = score_verification(record)
    tb.license_penalty = score_license(getattr(record, "redistribution_class", None))

    # Uncertainty: lower 95% CI is better.
    ci = getattr(record, "uncertainty_95ci", None)
    if ci is not None:
        try:
            tb.uncertainty_penalty = min(10, int(float(ci) * 100))
        except (TypeError, ValueError):
            tb.uncertainty_penalty = 5

    return tb


__all__ = [
    "TieBreakReasons",
    "build_tiebreak",
    "score_geography",
    "score_time",
    "score_time_granularity",
    "score_source_authority",
    "score_verification",
    "score_license",
]
