# -*- coding: utf-8 -*-
"""
GreenLang Factors — Resolution Engine (Phase F3).

This is the brain of the product.  Callers pass a
:class:`ResolutionRequest` (activity + method profile + jurisdiction +
date + optional tenant context) and receive a :class:`ResolvedFactor`
that includes not just the winning factor but the full reasoning chain:
alternates considered, why this one won, assumptions, gas breakdown,
uncertainty band, CO2e basis, deprecation status.

Non-negotiables enforced here:

- **#3** — "never hide fallback logic" — every resolution returns the
  full list of alternates plus the tie-break reasons that picked the
  winner.
- **#6** — "never let policy workflows call raw factors" —
  :class:`ResolutionRequest.method_profile` is required; the engine
  raises if the caller tries to bypass it.

Quick start::

    from greenlang.factors.resolution import ResolutionEngine, ResolutionRequest
    from greenlang.data.canonical_v2 import MethodProfile

    engine = ResolutionEngine()
    resolved = engine.resolve(
        ResolutionRequest(
            activity="12,500 kWh",
            method_profile=MethodProfile.CORPORATE_SCOPE2_LOCATION,
            jurisdiction="IN",
            reporting_date="2027-06-01",
        )
    )
    print(resolved.chosen_factor.factor_id)
    print(resolved.explain())

The REST surface for explain is ``GET /api/v1/factors/{factor_id}/explain``.
CLI surface: ``gl factors resolve --profile ... --activity ... --explain``.
"""
from __future__ import annotations

from greenlang.factors.resolution.engine import ResolutionEngine
from greenlang.factors.resolution.request import ResolutionRequest
from greenlang.factors.resolution.result import (
    AlternateCandidate,
    GasBreakdown,
    ResolvedFactor,
    UncertaintyBand,
)
from greenlang.factors.resolution.tiebreak import TieBreakReasons

__all__ = [
    "ResolutionEngine",
    "ResolutionRequest",
    "ResolvedFactor",
    "AlternateCandidate",
    "GasBreakdown",
    "UncertaintyBand",
    "TieBreakReasons",
]
