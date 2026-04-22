# -*- coding: utf-8 -*-
"""
Composite Factor Quality Score (FQS) on a 0-100 scale.

CTO non-negotiable (2026-04-22): expose a single 0-100 composite quality
score alongside the 5 per-dimension scores so developers, consultants,
and platform buyers can filter/sort/gate on one number.

Inputs
------
The engine already stores a GHG-Protocol-aligned 5-dimension
:class:`greenlang.data.emission_factor_record.DataQualityScore` on a
1-5 scale.  This module converts it into a 0-100 surface + a categorical
rating, without changing the underlying DQS (which stays 1-5 for
backwards compatibility with existing records and DQS emitters).

Design decisions
----------------
* **Version the formula.**  Weights can shift over time as we learn
  which dimensions predict audit outcomes.  Callers always receive
  ``formula_version`` so historical responses remain interpretable.
* **Expose components on 0-100.**  "Temporal=3" reads as "3/5 = 60/100";
  the UI, docs, and SDK consume the 0-100 view, the internal engine
  continues to operate on 1-5.
* **Rating bands are policy, not data.**  The Certified release gate
  uses ``RATING_BAND_CERTIFIED_MIN`` (75/100 by default); the Preview
  gate uses ``RATING_BAND_PREVIEW_MIN`` (50/100).  Ops can tighten
  either band without a schema change.
* **No hallucinated dimensions.**  We keep the DQS field names
  (temporal, geographical, technological, representativeness,
  methodological) and publish a CTO-spec alias map so external callers
  can read under either vocabulary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


FORMULA_VERSION = "fqs-1.0.0"


# ------------------------------------------------------------------ #
# Weights — weighted average across the 5 DQS dimensions.
#
# Rationale: temporal + geographic are the two dimensions that most
# often break compliance defensibility ("that factor is from 2018 in
# France, applied to a 2026 US reporter"), so they carry higher weight.
# Technological + representativeness are the primary-vs-secondary-data
# signals.  Methodological is a process-quality check; it carries the
# lowest weight because a poor methodology score rarely flips a regulator
# outcome on its own.
# ------------------------------------------------------------------ #

DEFAULT_WEIGHTS: Dict[str, float] = {
    "temporal": 0.25,
    "geographical": 0.25,
    "technological": 0.20,
    "representativeness": 0.15,
    "methodological": 0.15,
}

assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9, (
    "DEFAULT_WEIGHTS must sum to 1.0; got "
    f"{sum(DEFAULT_WEIGHTS.values()):.6f}"
)


# ------------------------------------------------------------------ #
# CTO-spec alias map.  External docs reference the right-hand names.
# Internal storage uses the DQS names (left).  Any external response
# includes both vocabularies so the spec is honoured without forcing
# an internal rename.
# ------------------------------------------------------------------ #

CTO_SPEC_ALIASES: Dict[str, str] = {
    "temporal": "temporal_representativeness",
    "geographical": "geographic_representativeness",
    "technological": "technology_representativeness",
    "representativeness": "verification",
    "methodological": "completeness",
}


# ------------------------------------------------------------------ #
# Rating bands.
# ------------------------------------------------------------------ #

RATING_BAND_EXCELLENT_MIN = 85  # 0-100
RATING_BAND_GOOD_MIN = 70
RATING_BAND_FAIR_MIN = 50
# Below RATING_BAND_FAIR_MIN -> "poor".

RATING_BAND_CERTIFIED_MIN = 75
RATING_BAND_PREVIEW_MIN = 50
# Below RATING_BAND_PREVIEW_MIN -> connector-only eligibility only.


def rating_label(composite_fqs: float) -> str:
    """Map a 0-100 FQS to a rating label."""
    if composite_fqs >= RATING_BAND_EXCELLENT_MIN:
        return "excellent"
    if composite_fqs >= RATING_BAND_GOOD_MIN:
        return "good"
    if composite_fqs >= RATING_BAND_FAIR_MIN:
        return "fair"
    return "poor"


def promotion_eligibility(composite_fqs: float) -> str:
    """
    Map a 0-100 FQS to a promotion label.

    Returns one of ``certified``, ``preview``, ``connector_only``.
    The release signoff pipeline consumes this directly:
    ``release_signoff_checklist`` refuses to promote a record to
    ``certified`` if the composite is below ``RATING_BAND_CERTIFIED_MIN``.
    """
    if composite_fqs >= RATING_BAND_CERTIFIED_MIN:
        return "certified"
    if composite_fqs >= RATING_BAND_PREVIEW_MIN:
        return "preview"
    return "connector_only"


# ------------------------------------------------------------------ #
# Dataclass outputs.
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class ComponentScore100:
    """One DQS dimension rendered on a 0-100 scale, with CTO alias."""

    name: str              # DQS internal name, e.g. "temporal"
    cto_alias: str         # CTO-spec alias, e.g. "temporal_representativeness"
    score_5: float         # original DQS 1-5 score (float to allow half-steps)
    score_100: float       # 0-100 rendering

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "cto_alias": self.cto_alias,
            "score_5": self.score_5,
            "score_100": self.score_100,
        }


@dataclass(frozen=True)
class CompositeFQS:
    """
    Public-surface composite quality score.

    ``components`` is a list (stable order: temporal, geographical,
    technological, representativeness, methodological) of 5 score-100
    entries so the SDK/UI can render a bar chart without re-ordering.
    ``composite_fqs`` is the single 0-100 number suitable for filters,
    sort orders, and release-gate thresholds.
    """

    composite_fqs: float
    rating: str
    promotion_eligibility: str
    components: list  # list[ComponentScore100]
    formula_version: str = FORMULA_VERSION
    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "composite_fqs": self.composite_fqs,
            "rating": self.rating,
            "promotion_eligibility": self.promotion_eligibility,
            "components": [c.to_dict() for c in self.components],
            "formula_version": self.formula_version,
            "weights": dict(self.weights),
        }


# ------------------------------------------------------------------ #
# Core computation.
# ------------------------------------------------------------------ #


def _to_100(score_5: float) -> float:
    """Map a 1-5 DQS dimension to a 0-100 scale.

    DQS is defined on [1, 5].  Maps linearly so that:
        5 -> 100   (best available)
        4 -> 80
        3 -> 60
        2 -> 40
        1 -> 20    (lowest valid score)
    We do NOT clamp to [0, 100]; callers should validate DQS inputs are
    in [1, 5] before calling (the DataQualityScore dataclass already
    enforces this in __post_init__).
    """
    return float(score_5) * 20.0


def compute_fqs(
    dqs: Any,
    *,
    weights: Optional[Mapping[str, float]] = None,
) -> CompositeFQS:
    """
    Compute the 0-100 composite FQS from a
    :class:`greenlang.data.emission_factor_record.DataQualityScore`.

    Args:
        dqs: DataQualityScore instance (or any object exposing 5 integer
            attributes: temporal, geographical, technological,
            representativeness, methodological — each in [1, 5]).
        weights: Optional override weights mapping.  Defaults to
            :data:`DEFAULT_WEIGHTS`.  Must sum to 1.0 within 1e-6.

    Returns:
        :class:`CompositeFQS` with ``composite_fqs`` in [20, 100].

    Raises:
        ValueError: if ``weights`` is supplied but does not sum to 1.0,
            or if a required DQS dimension is missing.
    """
    w = dict(weights) if weights is not None else dict(DEFAULT_WEIGHTS)
    if abs(sum(w.values()) - 1.0) > 1e-6:
        raise ValueError(
            f"weights must sum to 1.0; got {sum(w.values()):.6f}"
        )

    components = []
    weighted_sum = 0.0
    for dim in (
        "temporal",
        "geographical",
        "technological",
        "representativeness",
        "methodological",
    ):
        if not hasattr(dqs, dim):
            raise ValueError(
                f"DQS is missing required dimension {dim!r}"
            )
        score_5 = float(getattr(dqs, dim))
        score_100 = _to_100(score_5)
        weight = w[dim]
        weighted_sum += score_100 * weight
        components.append(
            ComponentScore100(
                name=dim,
                cto_alias=CTO_SPEC_ALIASES[dim],
                score_5=score_5,
                score_100=score_100,
            )
        )

    composite = round(weighted_sum, 2)
    return CompositeFQS(
        composite_fqs=composite,
        rating=rating_label(composite),
        promotion_eligibility=promotion_eligibility(composite),
        components=components,
        weights=w,
    )


def compute_fqs_from_dict(
    dqs_dict: Mapping[str, Any],
    *,
    weights: Optional[Mapping[str, float]] = None,
) -> CompositeFQS:
    """
    Convenience: compute FQS from a plain dict, e.g. a JSON payload.

    Expected keys: ``temporal``, ``geographical``, ``technological``,
    ``representativeness``, ``methodological`` (all 1-5 int or float).
    """
    class _Shim:
        temporal = dqs_dict["temporal"]
        geographical = dqs_dict["geographical"]
        technological = dqs_dict["technological"]
        representativeness = dqs_dict["representativeness"]
        methodological = dqs_dict["methodological"]

    return compute_fqs(_Shim, weights=weights)


__all__ = [
    "FORMULA_VERSION",
    "DEFAULT_WEIGHTS",
    "CTO_SPEC_ALIASES",
    "RATING_BAND_EXCELLENT_MIN",
    "RATING_BAND_GOOD_MIN",
    "RATING_BAND_FAIR_MIN",
    "RATING_BAND_CERTIFIED_MIN",
    "RATING_BAND_PREVIEW_MIN",
    "ComponentScore100",
    "CompositeFQS",
    "compute_fqs",
    "compute_fqs_from_dict",
    "rating_label",
    "promotion_eligibility",
]
