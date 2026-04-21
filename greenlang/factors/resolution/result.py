# -*- coding: utf-8 -*-
"""ResolvedFactor payload — non-negotiable #3 ("never hide fallback logic")."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import Field

from greenlang.schemas.base import GreenLangBase


class GasBreakdown(GreenLangBase):
    """Per-gas emission amounts for the resolved factor."""

    co2_kg: float = 0.0
    ch4_kg: float = 0.0
    n2o_kg: float = 0.0
    hfcs_kg: float = 0.0
    pfcs_kg: float = 0.0
    sf6_kg: float = 0.0
    nf3_kg: float = 0.0
    biogenic_co2_kg: float = 0.0
    co2e_total_kg: float = 0.0
    gwp_basis: str = "IPCC_AR6_100"


class UncertaintyBand(GreenLangBase):
    """Structured uncertainty disclosure."""

    distribution: str = "unknown"               # enum UncertaintyDistribution
    ci_95_percent: Optional[float] = None       # ±x fractional (e.g. 0.05)
    low: Optional[float] = None
    high: Optional[float] = None
    note: Optional[str] = None


class AlternateCandidate(GreenLangBase):
    """A candidate factor that the engine considered but did not select."""

    factor_id: str
    tie_break_score: float                      # lower is better (rank-like)
    why_not_chosen: str                         # short phrase
    source_id: Optional[str] = None
    vintage: Optional[int] = None
    redistribution_class: Optional[str] = None


class ResolvedFactor(GreenLangBase):
    """Full return value of :meth:`ResolutionEngine.resolve`.

    Every field here maps to one of the CTO's explain-endpoint requirements.
    """

    # Chosen factor.
    chosen_factor_id: str
    chosen_factor_name: Optional[str] = None
    source_id: Optional[str] = None
    source_version: Optional[str] = None
    factor_version: Optional[str] = None
    vintage: Optional[int] = None
    method_profile: str
    formula_type: Optional[str] = None
    redistribution_class: Optional[str] = None

    # Explain payload.
    fallback_rank: int                          # 1..7 from MethodPack region hierarchy
    step_label: str                             # e.g. "customer_override"
    why_chosen: str                             # human-readable one-liner
    alternates: List[AlternateCandidate] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    deprecation_status: Optional[str] = None
    deprecation_replacement: Optional[str] = None

    # Quality + uncertainty.
    quality_score: Optional[float] = None       # 0-100 composite DQS
    uncertainty: UncertaintyBand = Field(default_factory=UncertaintyBand)
    verification_status: Optional[str] = None

    # Numeric payload.
    gas_breakdown: GasBreakdown = Field(default_factory=GasBreakdown)
    factor_unit_denominator: Optional[str] = None
    primary_data_flag: Optional[str] = None

    # Unit conversion (populated when the request supplies ``target_unit``).
    target_unit: Optional[str] = None
    converted_co2e_per_unit: Optional[float] = None
    unit_conversion_factor: Optional[float] = None
    unit_conversion_path: List[str] = Field(default_factory=list)
    unit_conversion_note: Optional[str] = None

    # Tracing.
    resolved_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    method_pack_version: Optional[str] = None
    engine_version: str = "resolution-1.0.0"

    def explain(self) -> Dict[str, Any]:
        """Compact dict form suitable for the ``/explain`` endpoint."""
        return {
            "chosen": {
                "factor_id": self.chosen_factor_id,
                "factor_name": self.chosen_factor_name,
                "source": self.source_id,
                "source_version": self.source_version,
                "factor_version": self.factor_version,
                "vintage": self.vintage,
                "method_profile": self.method_profile,
                "redistribution_class": self.redistribution_class,
            },
            "derivation": {
                "fallback_rank": self.fallback_rank,
                "step_label": self.step_label,
                "why_chosen": self.why_chosen,
                "assumptions": self.assumptions,
                "deprecation_status": self.deprecation_status,
                "deprecation_replacement": self.deprecation_replacement,
            },
            "quality": {
                "score": self.quality_score,
                "verification_status": self.verification_status,
                "primary_data_flag": self.primary_data_flag,
            },
            "uncertainty": self.uncertainty.model_dump(),
            "emissions": self.gas_breakdown.model_dump(),
            "unit_conversion": (
                None
                if self.target_unit is None
                else {
                    "target_unit": self.target_unit,
                    "factor": self.unit_conversion_factor,
                    "converted_co2e_per_unit": self.converted_co2e_per_unit,
                    "path": list(self.unit_conversion_path),
                    "note": self.unit_conversion_note,
                }
            ),
            "alternates": [a.model_dump() for a in self.alternates],
            "meta": {
                "resolved_at": self.resolved_at.isoformat(),
                "method_pack_version": self.method_pack_version,
                "engine_version": self.engine_version,
            },
        }


__all__ = ["ResolvedFactor", "AlternateCandidate", "GasBreakdown", "UncertaintyBand"]
