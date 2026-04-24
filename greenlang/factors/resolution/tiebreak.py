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
    # Wave 3 (gold-eval): activity-text alignment penalty. 0 when the
    # factor's id/fuel_type/tags carry every query token, 20 when the
    # record looks semantically unrelated. Weighted heavily (x2) vs. the
    # other 10 signals so that a semantic miss can outrank a one-notch
    # geography or vintage advantage — the gold-eval showed the tie-
    # break was picking electricity factors for freight queries because
    # all the other signals were tied at 0.
    activity_match_penalty: int = 0
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
            + self.activity_match_penalty
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


def score_time(
    valid_to: Optional[date],
    request_date: date,
    valid_from: Optional[date] = None,
) -> int:
    """Year-proximity vintage scoring (Wave 5 — CEA FY27 fix).

    Rule order:
      1. Window CONTAINS the request date  -> 0 (best).
      2. Window is in the PAST (valid_to < request_date)
         -> 1 + whole_years_stale, capped at 10.
      3. Window is in the FUTURE (valid_from > request_date)
         -> 2 + whole_years_ahead, capped at 10. Future vintages
         always score strictly worse than a past vintage of the same
         calendar-year distance because they represent data the
         caller could not yet have used at the time of the event —
         this prevents the FY28 sibling from winning a FY27 request
         when both are "valid" by the old one-sided ``valid_to``
         check.
      4. Fallback (legacy behaviour, only ``valid_to`` known)
         -> ``valid_to >= request_date`` is 0, otherwise
         ``1 + years_stale``.  Keeps SimpleNamespace test fixtures
         that never declared a ``valid_from`` backward compatible.

    The signal is soft — upstream cascade + method-pack filtering
    are still responsible for hard eligibility. A future-dated
    factor can still win if nothing else is available (the score
    caps at 10, matching the other vintage-style signals).
    """
    if valid_to is None and valid_from is None:
        return 0
    # Case 1: window contains the request date.
    if (
        valid_from is not None
        and valid_to is not None
        and valid_from <= request_date <= valid_to
    ):
        return 0
    # Case 2: window is in the past.
    if valid_to is not None and valid_to < request_date:
        delta_days = (request_date - valid_to).days
        return min(10, delta_days // 365 + 1)
    # Case 3: window is in the future.
    if valid_from is not None and valid_from > request_date:
        delta_days = (valid_from - request_date).days
        return min(10, delta_days // 365 + 2)
    # Case 4: legacy fallback — only valid_to known.
    if valid_to is not None and valid_to >= request_date:
        return 0
    return 0


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


def score_activity_match(
    record: Any,
    activity_tokens: Optional[List[str]],
) -> int:
    """Penalty for poor alignment between the activity text and the factor.

    0  — every query token appears in the factor's id/fuel/tags blob
    4  — at least one token matches
    12 — no tokens match but we have a token list (means the record is
         semantically unrelated to the request)
    0  — no tokens at all (pure geography resolution)

    The asymmetric scale keeps the penalty competitive with the
    ``geography_distance`` signal (0-10) — a semantic miss must
    decisively lose to a semantic hit even when geography and source
    authority are tied.
    """
    if not activity_tokens:
        return 0
    blob_parts: List[str] = []
    for attr in ("factor_id", "fuel_type", "unit", "notes"):
        val = getattr(record, attr, None)
        if val:
            blob_parts.append(str(val).lower())
    tags = getattr(record, "tags", None) or []
    for t in tags:
        blob_parts.append(str(t).lower())
    blob = " ".join(blob_parts)
    if not blob:
        return 12
    hits = sum(1 for tok in activity_tokens if tok and tok in blob)
    if hits == 0:
        return 12
    # Proportional reward: 0 penalty when all tokens match, ramping up
    # to 4 when only one matches. Cap at 4 so that activity mismatch
    # never dominates a vintage-mismatch gap of 10.
    miss_ratio = 1.0 - (hits / max(1, len(activity_tokens)))
    return int(round(miss_ratio * 4))


def build_tiebreak(
    record: Any,
    *,
    request_geo: Optional[str],
    request_date: date,
    request_granularity: Optional[str] = None,
    activity_tokens: Optional[List[str]] = None,
) -> TieBreakReasons:
    tb = TieBreakReasons()
    tb.geography_distance = score_geography(
        getattr(record, "geography", None), request_geo
    )
    tb.time_distance = score_time(
        getattr(record, "valid_to", None),
        request_date,
        getattr(record, "valid_from", None),
    )
    tb.time_granularity_penalty = score_time_granularity(
        getattr(record, "time_granularity", None), request_granularity
    )
    tb.source_authority_rank = score_source_authority(
        getattr(record, "source_id", None)
    )
    tb.verification_penalty = score_verification(record)
    tb.license_penalty = score_license(getattr(record, "redistribution_class", None))
    tb.activity_match_penalty = score_activity_match(record, activity_tokens)

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
