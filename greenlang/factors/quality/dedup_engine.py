# -*- coding: utf-8 -*-
"""
Enhanced duplicate detection engine (F021).

Detects exact and near-duplicate factors within an edition using
fingerprint-based hashing and activity-key matching. Resolution
follows SOURCE_PRIORITY, geography specificity, temporal recency,
and DQS score per the canonical merge rules in ``dedupe_rules.py``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from greenlang.factors.dedupe_rules import (
    SOURCE_PRIORITY,
    duplicate_fingerprint as _record_fingerprint,
    sort_sources_for_merge,
)

logger = logging.getLogger(__name__)


def _fingerprint(factor: Any) -> str:
    """
    Compute duplicate fingerprint, working with both dicts and record objects.

    Uses the canonical ``duplicate_fingerprint`` for record objects, and a
    dict-aware equivalent for plain dicts.
    """
    if not isinstance(factor, dict):
        return _record_fingerprint(factor)
    import hashlib
    import json as _json
    key = {
        "fuel": str(factor.get("fuel_type", "")).lower(),
        "geo": str(factor.get("geography", "")),
        "scope": str(factor.get("scope", "")),
        "boundary": str(factor.get("boundary", "")),
        "methodology": str((factor.get("provenance") or {}).get("methodology", "")),
        "unit": str(factor.get("unit", "")),
        "valid_from": str(factor.get("valid_from", "")),
        "source_record_id": factor.get("source_record_id"),
    }
    raw = _json.dumps(key, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]

# Geography specificity ranking (higher = more specific)
_GEO_LEVEL_RANK: Dict[str, int] = {
    "facility": 5,
    "grid_zone": 4,
    "state": 3,
    "country": 2,
    "region": 1,
    "global": 0,
}


@dataclass
class DuplicatePair:
    """A pair of factors identified as duplicates or near-duplicates."""

    factor_id_a: str
    factor_id_b: str
    match_type: str  # "exact" | "near"
    fingerprint: str
    resolution: str  # "keep_a" | "keep_b" | "human_review"
    reason: str
    score_a: float = 0.0
    score_b: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_id_a": self.factor_id_a,
            "factor_id_b": self.factor_id_b,
            "match_type": self.match_type,
            "fingerprint": self.fingerprint,
            "resolution": self.resolution,
            "reason": self.reason,
            "score_a": self.score_a,
            "score_b": self.score_b,
        }


@dataclass
class DedupReport:
    """Result of duplicate detection on an edition."""

    edition_id: str
    total_factors: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    auto_resolved: int = 0
    human_review: int = 0
    pairs: List[DuplicatePair] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edition_id": self.edition_id,
            "total_factors": self.total_factors,
            "exact_duplicates": self.exact_duplicates,
            "near_duplicates": self.near_duplicates,
            "auto_resolved": self.auto_resolved,
            "human_review": self.human_review,
            "pairs": [p.to_dict() for p in self.pairs],
        }

    @property
    def has_duplicates(self) -> bool:
        return len(self.pairs) > 0


def _extract_source(factor: Any) -> str:
    """Get source_id from a factor (record or dict)."""
    if isinstance(factor, dict):
        sid = factor.get("source_id") or ""
        if not sid:
            fid = factor.get("factor_id", "")
            parts = fid.split(":")
            if len(parts) >= 2:
                sid = parts[1].lower()
        return sid.lower()
    sid = getattr(factor, "source_id", "") or ""
    if not sid:
        fid = getattr(factor, "factor_id", "")
        parts = fid.split(":")
        if len(parts) >= 2:
            sid = parts[1].lower()
    return sid.lower()


def _geo_level_rank(factor: Any) -> int:
    """Get geography level specificity rank."""
    if isinstance(factor, dict):
        gl = str(factor.get("geography_level", "")).lower()
    else:
        gl = str(getattr(getattr(factor, "geography_level", None), "value", getattr(factor, "geography_level", ""))).lower()
    return _GEO_LEVEL_RANK.get(gl, 1)


def _source_year(factor: Any) -> int:
    """Get source year from provenance."""
    if isinstance(factor, dict):
        prov = factor.get("provenance") or {}
        return int(prov.get("source_year") or 0)
    prov = getattr(factor, "provenance", None)
    if prov is None:
        return 0
    return int(getattr(prov, "source_year", 0) or 0)


def _dqs_score(factor: Any) -> float:
    """Compute average DQS score (0-5)."""
    if isinstance(factor, dict):
        dqs = factor.get("dqs") or {}
    else:
        dqs = getattr(factor, "dqs", None)
        if dqs is not None and not isinstance(dqs, dict):
            dqs = {
                "temporal": getattr(dqs, "temporal", 0),
                "geographical": getattr(dqs, "geographical", 0),
                "technological": getattr(dqs, "technological", 0),
                "representativeness": getattr(dqs, "representativeness", 0),
                "methodological": getattr(dqs, "methodological", 0),
            }
        elif dqs is None:
            dqs = {}
    vals = [float(v) for v in dqs.values() if isinstance(v, (int, float))]
    return sum(vals) / max(len(vals), 1)


def _uncertainty(factor: Any) -> float:
    """Get uncertainty_95ci (lower is better)."""
    if isinstance(factor, dict):
        return float(factor.get("uncertainty_95ci") or 1.0)
    return float(getattr(factor, "uncertainty_95ci", 1.0) or 1.0)


def _factor_id(factor: Any) -> str:
    if isinstance(factor, dict):
        return factor.get("factor_id", "")
    return getattr(factor, "factor_id", "")


def _content_hash(factor: Any) -> str:
    if isinstance(factor, dict):
        return factor.get("content_hash", "")
    return getattr(factor, "content_hash", "")


def _merge_score(factor: Any) -> Tuple[int, int, int, float, float]:
    """
    Composite merge priority score (higher tuple = better candidate to keep).

    Order: source_priority, geo_specificity, source_year, dqs_avg, -uncertainty
    """
    src = _extract_source(factor)
    try:
        src_rank = len(SOURCE_PRIORITY) - SOURCE_PRIORITY.index(src)
    except ValueError:
        src_rank = 0
    return (
        src_rank,
        _geo_level_rank(factor),
        _source_year(factor),
        _dqs_score(factor),
        -_uncertainty(factor),
    )


def _resolve_pair(
    factor_a: Any,
    factor_b: Any,
    match_type: str,
    fingerprint: str,
) -> DuplicatePair:
    """Determine which factor to keep in a duplicate pair."""
    fid_a = _factor_id(factor_a)
    fid_b = _factor_id(factor_b)
    score_a = _merge_score(factor_a)
    score_b = _merge_score(factor_b)

    # If scores are identical, flag for human review
    if score_a == score_b:
        return DuplicatePair(
            factor_id_a=fid_a,
            factor_id_b=fid_b,
            match_type=match_type,
            fingerprint=fingerprint,
            resolution="human_review",
            reason="identical merge scores",
            score_a=_dqs_score(factor_a),
            score_b=_dqs_score(factor_b),
        )

    if score_a > score_b:
        return DuplicatePair(
            factor_id_a=fid_a,
            factor_id_b=fid_b,
            match_type=match_type,
            fingerprint=fingerprint,
            resolution="keep_a",
            reason=f"higher merge priority ({_extract_source(factor_a)} > {_extract_source(factor_b)})",
            score_a=_dqs_score(factor_a),
            score_b=_dqs_score(factor_b),
        )
    else:
        return DuplicatePair(
            factor_id_a=fid_a,
            factor_id_b=fid_b,
            match_type=match_type,
            fingerprint=fingerprint,
            resolution="keep_b",
            reason=f"higher merge priority ({_extract_source(factor_b)} > {_extract_source(factor_a)})",
            score_a=_dqs_score(factor_a),
            score_b=_dqs_score(factor_b),
        )


def _activity_key(factor: Any) -> str:
    """
    Near-duplicate key: fuel_type + geography + scope + boundary.

    Near-dupes have the same activity but may come from different sources
    with different values.
    """
    if isinstance(factor, dict):
        fuel = str(factor.get("fuel_type", "")).lower().strip()
        geo = str(factor.get("geography", "")).upper().strip()
        scope = str(factor.get("scope", "")).strip()
        boundary = str(factor.get("boundary", "")).lower().strip()
    else:
        fuel = str(getattr(factor, "fuel_type", "")).lower().strip()
        geo = str(getattr(factor, "geography", "")).upper().strip()
        scope = str(getattr(getattr(factor, "scope", None), "value", getattr(factor, "scope", ""))).strip()
        boundary = str(getattr(getattr(factor, "boundary", None), "value", getattr(factor, "boundary", ""))).lower().strip()
    return f"{fuel}|{geo}|{scope}|{boundary}"


def detect_duplicates(
    factors: Sequence[Any],
    *,
    edition_id: str = "unknown",
    detect_near: bool = True,
) -> DedupReport:
    """
    Detect exact and near-duplicate factors.

    Args:
        factors: List of EmissionFactorRecord objects or factor dicts.
        edition_id: Edition label for reporting.
        detect_near: If True, also detect near-duplicates (same activity key).

    Returns:
        DedupReport with pairs and resolution recommendations.
    """
    report = DedupReport(edition_id=edition_id, total_factors=len(factors))

    # Phase 1: Exact duplicates (same fingerprint)
    fp_groups: Dict[str, List[Any]] = defaultdict(list)
    for f in factors:
        fp = _fingerprint(f)
        fp_groups[fp].append(f)

    for fp, group in fp_groups.items():
        if len(group) < 2:
            continue
        # Compare all pairs in the group
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                # Check content hash — if identical, truly exact
                ch_a = _content_hash(group[i])
                ch_b = _content_hash(group[j])
                match_type = "exact" if (ch_a and ch_b and ch_a == ch_b) else "near"

                pair = _resolve_pair(group[i], group[j], match_type, fp)
                report.pairs.append(pair)

                if match_type == "exact":
                    report.exact_duplicates += 1
                else:
                    report.near_duplicates += 1

                if pair.resolution == "human_review":
                    report.human_review += 1
                else:
                    report.auto_resolved += 1

    # Phase 2: Near-duplicates (same activity key, different fingerprint)
    if detect_near:
        ak_groups: Dict[str, List[Any]] = defaultdict(list)
        for f in factors:
            ak = _activity_key(f)
            ak_groups[ak].append(f)

        seen_pairs: set = set()
        for _ak, group in ak_groups.items():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    fid_a = _factor_id(group[i])
                    fid_b = _factor_id(group[j])
                    pair_key = (min(fid_a, fid_b), max(fid_a, fid_b))
                    if pair_key in seen_pairs:
                        continue
                    # Skip if already detected as exact duplicate
                    fp_a = _fingerprint(group[i])
                    fp_b = _fingerprint(group[j])
                    if fp_a == fp_b:
                        seen_pairs.add(pair_key)
                        continue

                    pair = _resolve_pair(group[i], group[j], "near", f"activity:{_ak}")
                    report.pairs.append(pair)
                    report.near_duplicates += 1
                    seen_pairs.add(pair_key)

                    if pair.resolution == "human_review":
                        report.human_review += 1
                    else:
                        report.auto_resolved += 1

    logger.info(
        "Dedup complete: edition=%s factors=%d exact=%d near=%d auto_resolved=%d human_review=%d",
        edition_id, len(factors), report.exact_duplicates, report.near_duplicates,
        report.auto_resolved, report.human_review,
    )
    return report
