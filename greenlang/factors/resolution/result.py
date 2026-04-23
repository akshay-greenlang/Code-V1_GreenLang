# -*- coding: utf-8 -*-
"""ResolvedFactor payload — non-negotiable #3 ("never hide fallback logic").

This module carries the full CTO 16-element resolver contract.  Each new
envelope model (``ChosenFactor``, ``SourceDescriptor``, ``QualityEnvelope``,
``UncertaintyEnvelope``, ``LicensingEnvelope``, ``DeprecationStatus``) is
projected by :meth:`ResolutionEngine._build_resolved_factor` and surfaced
on :class:`ResolvedFactor` alongside the legacy flat fields.  Legacy
fields (``chosen_factor_id``, ``source_id``, ``why_chosen``, ``quality_score``,
...) are retained for one deprecation window and are marked with
``Field(deprecated=True)``.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field

from greenlang.schemas.base import GreenLangBase


# ---------------------------------------------------------------------------
# Envelope models — CTO 16-element contract
# ---------------------------------------------------------------------------


class GasBreakdown(GreenLangBase):
    """Per-gas emission amounts for the resolved factor (CTO element #8).

    Both ``*_kg`` (legacy) and unsuffixed names are emitted so spec-compliant
    consumers can read either vocabulary.  ``co2e_total`` is exposed as a
    first-class field.
    """

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

    # CTO-spec aliases (unsuffixed names).  Populated on model_dump via
    # @property.  Stored as computed fields so FastAPI/Pydantic includes
    # them in JSON output.
    @property
    def co2(self) -> float:  # pragma: no cover - trivial alias
        return self.co2_kg

    @property
    def ch4(self) -> float:  # pragma: no cover
        return self.ch4_kg

    @property
    def n2o(self) -> float:  # pragma: no cover
        return self.n2o_kg

    @property
    def hfcs(self) -> float:  # pragma: no cover
        return self.hfcs_kg

    @property
    def pfcs(self) -> float:  # pragma: no cover
        return self.pfcs_kg

    @property
    def sf6(self) -> float:  # pragma: no cover
        return self.sf6_kg

    @property
    def nf3(self) -> float:  # pragma: no cover
        return self.nf3_kg

    @property
    def biogenic_co2(self) -> float:  # pragma: no cover
        return self.biogenic_co2_kg

    @property
    def co2e_total(self) -> float:  # pragma: no cover
        return self.co2e_total_kg


class ChosenFactor(GreenLangBase):
    """CTO contract element #1 — compact nested envelope for the chosen factor."""

    id: str
    name: str
    version: str
    factor_family: str


class SourceDescriptor(GreenLangBase):
    """CTO contract element #4 — 4-field source envelope."""

    id: str
    version: str
    name: str
    authority: str


class QualityEnvelope(GreenLangBase):
    """CTO contract element #10 — composite FQS + 5 per-dimension 0-100 scores."""

    composite_fqs_0_100: float = Field(..., ge=0.0, le=100.0)
    temporal_score: float = Field(..., ge=0.0, le=100.0)
    geographic_score: float = Field(..., ge=0.0, le=100.0)
    technology_score: float = Field(..., ge=0.0, le=100.0)
    verification_score: float = Field(..., ge=0.0, le=100.0)
    completeness_score: float = Field(..., ge=0.0, le=100.0)
    rating: Optional[str] = None
    formula_version: Optional[str] = None
    weights: Optional[Dict[str, float]] = None


class UncertaintyEnvelope(GreenLangBase):
    """CTO contract element #11 — typed uncertainty with numeric bounds."""

    type: Literal["95_percent_ci", "qualitative"]
    low: Optional[float] = None
    high: Optional[float] = None
    distribution: Optional[str] = None
    note: Optional[str] = None


class LicensingEnvelope(GreenLangBase):
    """CTO contract element #12 — licensing/redistribution envelope."""

    redistribution_class: Literal[
        "open",
        "licensed_embedded",
        "customer_private",
        "oem_redistributable",
    ]
    customer_entitlement_required: bool = False
    watermark_required: bool = False


class DeprecationStatus(GreenLangBase):
    """CTO contract element #15 — deprecation envelope, always present."""

    status: Literal["active", "deprecated", "superseded"] = "active"
    replacement_pointer_factor_id: Optional[str] = None
    deprecated_at: Optional[date] = None
    reason: Optional[str] = None


class UncertaintyBand(GreenLangBase):
    """Legacy structured uncertainty disclosure.

    Kept for backwards compatibility with existing consumers; new callers
    should read :class:`UncertaintyEnvelope` via ``resolved.uncertainty``.
    """

    distribution: str = "unknown"               # enum UncertaintyDistribution
    ci_95_percent: Optional[float] = None       # ±x fractional (e.g. 0.05)
    low: Optional[float] = None
    high: Optional[float] = None
    note: Optional[str] = None


class AlternateCandidate(GreenLangBase):
    """A candidate factor that the engine considered but did not select."""

    factor_id: str
    tie_break_score: float                      # lower is better (rank-like)
    why_not_chosen: str                         # short phrase (legacy name)
    reason_lost: Optional[str] = None           # CTO-spec alias of why_not_chosen
    source_id: Optional[str] = None
    vintage: Optional[int] = None
    redistribution_class: Optional[str] = None


# ---------------------------------------------------------------------------
# ResolvedFactor — the full 16-element envelope
# ---------------------------------------------------------------------------


class ResolvedFactor(GreenLangBase):
    """Full return value of :meth:`ResolutionEngine.resolve`.

    Carries every CTO-required field in the 16-element contract via nested
    envelopes, plus legacy flat fields (marked deprecated) for one release
    cycle of backwards compatibility.
    """

    # --- Contract envelopes (CTO 16-element) ---
    chosen_factor: Optional[ChosenFactor] = None                    # #1
    source: Optional[SourceDescriptor] = None                       # #4
    quality: Optional[QualityEnvelope] = None                       # #10
    licensing: Optional[LicensingEnvelope] = None                   # #12
    deprecation_status: DeprecationStatus = Field(                  # #15
        default_factory=lambda: DeprecationStatus(status="active")
    )

    # --- Contract scalars ---
    why_this_won: Optional[str] = None                              # #3
    release_version: Optional[str] = None                           # #5 (distinct)
    method_pack: Optional[str] = None                               # #6 (pack id)
    valid_from: Optional[date] = None                               # #7
    valid_to: Optional[date] = None                                 # #7
    co2e_basis: Optional[str] = None                                # #9

    # --- Legacy flat fields (deprecated; kept for one release) ---
    chosen_factor_id: str = Field(..., description="Deprecated: read chosen_factor.id instead.")
    chosen_factor_name: Optional[str] = None
    source_id: Optional[str] = Field(None, description="Deprecated: read source.id instead.")
    source_version: Optional[str] = None
    factor_version: Optional[str] = None
    vintage: Optional[int] = None
    method_profile: str
    formula_type: Optional[str] = None
    redistribution_class: Optional[str] = Field(
        None, description="Deprecated: read licensing.redistribution_class."
    )

    # Explain payload (legacy + canonical).
    fallback_rank: int                          # 1..7 — CTO #14 (already compliant)
    step_label: str
    why_chosen: str = Field(..., description="Deprecated alias of why_this_won.")
    alternates: List[AlternateCandidate] = Field(default_factory=list)   # #2
    assumptions: List[str] = Field(default_factory=list)                 # #13
    deprecation_replacement: Optional[str] = Field(
        None,
        description="Deprecated: read deprecation_status.replacement_pointer_factor_id.",
    )

    # Legacy quality + uncertainty.
    quality_score: Optional[float] = Field(
        None, description="Deprecated: read quality.composite_fqs_0_100."
    )
    uncertainty: UncertaintyEnvelope = Field(
        default_factory=lambda: UncertaintyEnvelope(type="qualitative")
    )
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

    # Audit text (CTO contract — /explain payload).  Populated by the
    # resolution engine via the method-pack audit-text renderer.  When the
    # template is unapproved (``approved: false`` in its frontmatter), the
    # rendered paragraph is prefixed with the SAFE-DRAFT banner and
    # ``audit_text_draft`` is set to True so SDK/UI consumers can flag it.
    # See ``docs/specs/audit_text_template_policy.md`` for the policy.
    audit_text: Optional[str] = None
    audit_text_draft: Optional[bool] = None

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
                "factor_family": (
                    self.chosen_factor.factor_family if self.chosen_factor else None
                ),
                "source": (
                    self.source.model_dump() if self.source else self.source_id
                ),
                "source_version": self.source_version,
                "factor_version": self.factor_version,
                "release_version": self.release_version,
                "vintage": self.vintage,
                "method_profile": self.method_profile,
                "method_pack": self.method_pack,
                "method_pack_version": self.method_pack_version,
                "valid_from": self.valid_from.isoformat() if self.valid_from else None,
                "valid_to": self.valid_to.isoformat() if self.valid_to else None,
                "licensing": self.licensing.model_dump() if self.licensing else None,
                "redistribution_class": self.redistribution_class,
            },
            "derivation": {
                "fallback_rank": self.fallback_rank,
                "step_label": self.step_label,
                "why_this_won": self.why_this_won or self.why_chosen,
                "why_chosen": self.why_chosen,
                "assumptions": self.assumptions,
                "deprecation_status": self.deprecation_status.model_dump(),
                "deprecation_replacement": self.deprecation_replacement,
            },
            "quality": (
                self.quality.model_dump()
                if self.quality
                else {
                    "score": self.quality_score,
                    "verification_status": self.verification_status,
                    "primary_data_flag": self.primary_data_flag,
                }
            ),
            "uncertainty": self.uncertainty.model_dump(),
            "emissions": self.gas_breakdown.model_dump(),
            "co2e_basis": self.co2e_basis,
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
            "audit_text": self.audit_text,
            "audit_text_draft": self.audit_text_draft,
            "meta": {
                "resolved_at": self.resolved_at.isoformat(),
                "method_pack_version": self.method_pack_version,
                "engine_version": self.engine_version,
            },
        }


__all__ = [
    "AlternateCandidate",
    "ChosenFactor",
    "DeprecationStatus",
    "GasBreakdown",
    "LicensingEnvelope",
    "QualityEnvelope",
    "ResolvedFactor",
    "SourceDescriptor",
    "UncertaintyBand",
    "UncertaintyEnvelope",
]
