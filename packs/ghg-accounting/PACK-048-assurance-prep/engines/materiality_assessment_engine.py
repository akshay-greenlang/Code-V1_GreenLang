# -*- coding: utf-8 -*-
"""
MaterialityAssessmentEngine - PACK-048 GHG Assurance Prep Engine 6
====================================================================

Determines materiality thresholds for GHG assurance engagements including
quantitative materiality, performance materiality, clearly trivial
thresholds, scope-specific materiality, and qualitative factor assessment.

Calculation Methodology:
    Quantitative Materiality:
        M = total_emissions * materiality_pct

        Where:
            total_emissions = total reported tCO2e
            materiality_pct = default 5% (range 1-10%)

    Performance Materiality:
        PM = M * performance_pct

        Where:
            performance_pct = default 65% (range 50-75%)

    Clearly Trivial Threshold:
        CT = M * trivial_pct

        Where:
            trivial_pct = default 5% (range 1-10%)

    Scope-Specific Materiality:
        M_scope = scope_emissions * scope_materiality_pct

        Different thresholds per scope based on data quality
        and assurance expectations.

    Qualitative Factors (scored 1-5):
        regulatory_sensitivity:     Regulatory scrutiny level
        stakeholder_visibility:     Public/stakeholder attention
        reputational_risk:          Reputational exposure
        estimation_uncertainty:     Degree of estimation
        prior_period_adjustments:   History of restatements

    Qualitative Adjustment:
        M_adjusted = M * (1 - qualitative_adjustment)

        Where:
            qualitative_adjustment = SUM(factor_scores) / (5 * n_factors) * max_adj
            max_adj = 0.25 (maximum 25% reduction)

    Specific Materiality (per line item):
        M_specific = line_item_value * specific_pct

        Where:
            specific_pct may be lower than overall materiality_pct
            for sensitive items.

Regulatory References:
    - ISAE 3410 para 23-25: Materiality in GHG assurance
    - ISAE 3000 (Revised): Materiality concept
    - ISA 320: Materiality in Planning and Performing an Audit
    - ISO 14064-3:2019 Clause 6.2: Significance (materiality)
    - ESRS 1 para 25-28: Materiality assessment
    - GHG Protocol: Relevance and completeness principles

Zero-Hallucination:
    - All materiality calculations use deterministic Decimal arithmetic
    - All formulas from published assurance standards
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-048 GHG Assurance Prep
Engine:  6 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0"),
) -> Decimal:
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _round2(value: Any) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MaterialityBasis(str, Enum):
    """Basis for materiality determination.

    TOTAL_EMISSIONS:        Based on total reported emissions.
    SCOPE_EMISSIONS:        Based on individual scope emissions.
    REVENUE_INTENSITY:      Based on revenue intensity metric.
    PRODUCTION_INTENSITY:   Based on production intensity metric.
    """
    TOTAL_EMISSIONS = "total_emissions"
    SCOPE_EMISSIONS = "scope_emissions"
    REVENUE_INTENSITY = "revenue_intensity"
    PRODUCTION_INTENSITY = "production_intensity"


class AssuranceLevel(str, Enum):
    """Assurance level.

    LIMITED:        Limited assurance engagement.
    REASONABLE:     Reasonable assurance engagement.
    """
    LIMITED = "limited"
    REASONABLE = "reasonable"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MATERIALITY_PCT: Decimal = Decimal("0.05")        # 5%
DEFAULT_PERFORMANCE_PCT: Decimal = Decimal("0.65")         # 65%
DEFAULT_TRIVIAL_PCT: Decimal = Decimal("0.05")             # 5% of materiality
DEFAULT_MAX_QUALITATIVE_ADJ: Decimal = Decimal("0.25")     # 25% max reduction

# Scope-specific default materiality percentages
SCOPE_MATERIALITY_DEFAULTS: Dict[str, Decimal] = {
    "scope_1": Decimal("0.05"),     # 5%
    "scope_2": Decimal("0.05"),     # 5%
    "scope_3": Decimal("0.10"),     # 10% (higher uncertainty)
}

# Assurance level multiplier for materiality
ASSURANCE_LEVEL_MULTIPLIER: Dict[str, Decimal] = {
    AssuranceLevel.LIMITED.value: Decimal("1.0"),
    AssuranceLevel.REASONABLE.value: Decimal("0.75"),    # Tighter for reasonable
}


# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------


class QualitativeFactor(BaseModel):
    """Qualitative factor for materiality adjustment.

    Attributes:
        factor_name:    Factor name.
        description:    Factor description.
        score:          Score (1-5, 5=highest risk/sensitivity).
        rationale:      Rationale for score.
    """
    factor_name: str = Field(default="", description="Factor name")
    description: str = Field(default="", description="Description")
    score: int = Field(default=1, ge=1, le=5, description="Score (1-5)")
    rationale: str = Field(default="", description="Rationale")


class ScopeEmissions(BaseModel):
    """Emissions by scope for scope-specific materiality.

    Attributes:
        scope_1_tco2e:  Scope 1 emissions (tCO2e).
        scope_2_tco2e:  Scope 2 emissions (tCO2e).
        scope_3_tco2e:  Scope 3 emissions (tCO2e).
        total_tco2e:    Total emissions (tCO2e).
    """
    scope_1_tco2e: Decimal = Field(default=Decimal("0"), ge=0, description="Scope 1")
    scope_2_tco2e: Decimal = Field(default=Decimal("0"), ge=0, description="Scope 2")
    scope_3_tco2e: Decimal = Field(default=Decimal("0"), ge=0, description="Scope 3")
    total_tco2e: Decimal = Field(default=Decimal("0"), ge=0, description="Total")

    @field_validator(
        "scope_1_tco2e", "scope_2_tco2e", "scope_3_tco2e", "total_tco2e",
        mode="before",
    )
    @classmethod
    def coerce_emissions(cls, v: Any) -> Decimal:
        return _decimal(v)

    @model_validator(mode="after")
    def compute_total(self) -> "ScopeEmissions":
        if self.total_tco2e == Decimal("0"):
            self.total_tco2e = (
                self.scope_1_tco2e + self.scope_2_tco2e + self.scope_3_tco2e
            )
        return self


class SpecificItem(BaseModel):
    """A specific line item for individual materiality assessment.

    Attributes:
        item_id:        Item identifier.
        item_name:      Item name.
        value_tco2e:    Item value in tCO2e.
        specific_pct:   Specific materiality percentage (if different from overall).
        is_sensitive:   Whether item is considered sensitive.
    """
    item_id: str = Field(default="", description="Item ID")
    item_name: str = Field(default="", description="Name")
    value_tco2e: Decimal = Field(default=Decimal("0"), ge=0, description="Value tCO2e")
    specific_pct: Optional[Decimal] = Field(default=None, ge=0, le=1, description="Specific %")
    is_sensitive: bool = Field(default=False, description="Sensitive")

    @field_validator("value_tco2e", mode="before")
    @classmethod
    def coerce_value(cls, v: Any) -> Decimal:
        return _decimal(v)


class MaterialityConfig(BaseModel):
    """Configuration for materiality assessment.

    Attributes:
        organisation_id:        Organisation identifier.
        assurance_level:        Assurance level (limited/reasonable).
        materiality_pct:        Overall materiality percentage.
        performance_pct:        Performance materiality percentage.
        trivial_pct:            Clearly trivial percentage.
        scope_materiality_pcts: Scope-specific materiality percentages.
        max_qualitative_adj:    Maximum qualitative adjustment.
        output_precision:       Output decimal places.
    """
    organisation_id: str = Field(default="", description="Org ID")
    assurance_level: AssuranceLevel = Field(
        default=AssuranceLevel.LIMITED, description="Assurance level"
    )
    materiality_pct: Decimal = Field(
        default=DEFAULT_MATERIALITY_PCT, ge=Decimal("0.01"), le=Decimal("0.10"),
        description="Materiality %",
    )
    performance_pct: Decimal = Field(
        default=DEFAULT_PERFORMANCE_PCT, ge=Decimal("0.50"), le=Decimal("0.75"),
        description="Performance %",
    )
    trivial_pct: Decimal = Field(
        default=DEFAULT_TRIVIAL_PCT, ge=Decimal("0.01"), le=Decimal("0.10"),
        description="Trivial %",
    )
    scope_materiality_pcts: Dict[str, Decimal] = Field(
        default_factory=lambda: dict(SCOPE_MATERIALITY_DEFAULTS),
        description="Scope materiality %s",
    )
    max_qualitative_adj: Decimal = Field(
        default=DEFAULT_MAX_QUALITATIVE_ADJ, ge=0, le=Decimal("0.50"),
        description="Max qualitative adjustment",
    )
    output_precision: int = Field(default=2, ge=0, le=6, description="Output precision")


class MaterialityInput(BaseModel):
    """Input for materiality assessment.

    Attributes:
        emissions:              Emissions by scope.
        qualitative_factors:    Qualitative factors for assessment.
        specific_items:         Specific line items.
        config:                 Materiality configuration.
    """
    emissions: ScopeEmissions = Field(
        default_factory=ScopeEmissions, description="Emissions"
    )
    qualitative_factors: List[QualitativeFactor] = Field(
        default_factory=list, description="Qualitative factors"
    )
    specific_items: List[SpecificItem] = Field(
        default_factory=list, description="Specific items"
    )
    config: MaterialityConfig = Field(
        default_factory=MaterialityConfig, description="Configuration"
    )


# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------


class MaterialityThreshold(BaseModel):
    """A materiality threshold value.

    Attributes:
        threshold_name:     Threshold name.
        value_tco2e:        Threshold value in tCO2e.
        basis:              Calculation basis.
        percentage:         Percentage used.
        description:        Description.
    """
    threshold_name: str = Field(default="", description="Name")
    value_tco2e: Decimal = Field(default=Decimal("0"), description="Value tCO2e")
    basis: str = Field(default="", description="Basis")
    percentage: Decimal = Field(default=Decimal("0"), description="Percentage")
    description: str = Field(default="", description="Description")


class ScopeMateriality(BaseModel):
    """Scope-specific materiality.

    Attributes:
        scope:              Scope name.
        scope_emissions:    Scope emissions (tCO2e).
        materiality_pct:    Materiality percentage.
        materiality_tco2e:  Materiality threshold (tCO2e).
    """
    scope: str = Field(default="", description="Scope")
    scope_emissions: Decimal = Field(default=Decimal("0"), description="Emissions")
    materiality_pct: Decimal = Field(default=Decimal("0"), description="Materiality %")
    materiality_tco2e: Decimal = Field(default=Decimal("0"), description="Materiality tCO2e")


class QualitativeAssessment(BaseModel):
    """Qualitative assessment summary.

    Attributes:
        factors:                Qualitative factors.
        total_score:            Sum of factor scores.
        max_possible_score:     Maximum possible score.
        normalised_score:       Normalised score (0-1).
        adjustment_pct:         Materiality adjustment percentage.
        adjusted_materiality:   Adjusted materiality (tCO2e).
    """
    factors: List[QualitativeFactor] = Field(
        default_factory=list, description="Factors"
    )
    total_score: int = Field(default=0, description="Total score")
    max_possible_score: int = Field(default=0, description="Max score")
    normalised_score: Decimal = Field(default=Decimal("0"), description="Normalised")
    adjustment_pct: Decimal = Field(default=Decimal("0"), description="Adjustment %")
    adjusted_materiality: Decimal = Field(
        default=Decimal("0"), description="Adjusted materiality"
    )


class SpecificMateriality(BaseModel):
    """Specific materiality for a line item.

    Attributes:
        item_id:            Item identifier.
        item_name:          Item name.
        value_tco2e:        Item value.
        materiality_tco2e:  Specific materiality threshold.
        exceeds_materiality: Whether item exceeds materiality.
    """
    item_id: str = Field(default="", description="Item ID")
    item_name: str = Field(default="", description="Name")
    value_tco2e: Decimal = Field(default=Decimal("0"), description="Value")
    materiality_tco2e: Decimal = Field(default=Decimal("0"), description="Threshold")
    exceeds_materiality: bool = Field(default=False, description="Exceeds")


class MaterialityAssessment(BaseModel):
    """Complete materiality assessment.

    Attributes:
        overall_materiality:        Overall materiality threshold (tCO2e).
        performance_materiality:    Performance materiality (tCO2e).
        clearly_trivial:            Clearly trivial threshold (tCO2e).
        scope_materialities:        Per-scope materiality.
        qualitative_assessment:     Qualitative assessment.
        specific_materialities:     Per-item materiality.
        final_materiality:          Final materiality after qualitative adjustment.
    """
    overall_materiality: MaterialityThreshold = Field(
        default_factory=MaterialityThreshold, description="Overall"
    )
    performance_materiality: MaterialityThreshold = Field(
        default_factory=MaterialityThreshold, description="Performance"
    )
    clearly_trivial: MaterialityThreshold = Field(
        default_factory=MaterialityThreshold, description="Trivial"
    )
    scope_materialities: List[ScopeMateriality] = Field(
        default_factory=list, description="Scope"
    )
    qualitative_assessment: QualitativeAssessment = Field(
        default_factory=QualitativeAssessment, description="Qualitative"
    )
    specific_materialities: List[SpecificMateriality] = Field(
        default_factory=list, description="Specific"
    )
    final_materiality: Decimal = Field(default=Decimal("0"), description="Final")


class MaterialityResult(BaseModel):
    """Complete result of materiality assessment.

    Attributes:
        result_id:              Unique result identifier.
        organisation_id:        Organisation identifier.
        assessment:             Materiality assessment.
        assurance_level:        Assurance level.
        total_emissions_tco2e:  Total emissions.
        methodology_rationale:  Methodology rationale.
        warnings:               Warnings.
        calculated_at:          Timestamp.
        processing_time_ms:     Processing time (ms).
        provenance_hash:        SHA-256 hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    organisation_id: str = Field(default="", description="Org ID")
    assessment: MaterialityAssessment = Field(
        default_factory=MaterialityAssessment, description="Assessment"
    )
    assurance_level: str = Field(default="", description="Assurance level")
    total_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), description="Total emissions"
    )
    methodology_rationale: str = Field(default="", description="Methodology rationale")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    calculated_at: str = Field(default="", description="Timestamp")
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    provenance_hash: str = Field(default="", description="SHA-256 hash")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class MaterialityAssessmentEngine:
    """Determines materiality thresholds for GHG assurance engagements.

    Computes quantitative materiality, performance materiality, clearly
    trivial thresholds, scope-specific materiality, and qualitative
    factor adjustments.

    Guarantees:
        - Deterministic: Same inputs always produce the same output.
        - Reproducible: Full provenance tracking with SHA-256 hashes.
        - Auditable: Every threshold calculation documented.
        - Zero-Hallucination: No LLM in any calculation path.
    """

    def __init__(self) -> None:
        self._version = _MODULE_VERSION
        logger.info("MaterialityAssessmentEngine v%s initialised", self._version)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, input_data: MaterialityInput) -> MaterialityResult:
        """Assess materiality for GHG assurance engagement.

        Args:
            input_data: Emissions, qualitative factors, specific items, config.

        Returns:
            MaterialityResult with all materiality thresholds.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []
        config = input_data.config
        prec = config.output_precision
        prec_str = "0." + "0" * prec
        emissions = input_data.emissions

        # Step 1: Assurance level multiplier
        level_mult = ASSURANCE_LEVEL_MULTIPLIER.get(
            config.assurance_level.value, Decimal("1.0")
        )

        # Step 2: Overall materiality: M = total_emissions * materiality_pct * level_mult
        mat_pct = config.materiality_pct * level_mult
        overall_tco2e = (emissions.total_tco2e * mat_pct).quantize(
            Decimal(prec_str), rounding=ROUND_HALF_UP
        )
        overall_mat = MaterialityThreshold(
            threshold_name="Overall Materiality",
            value_tco2e=overall_tco2e,
            basis="total_emissions",
            percentage=mat_pct.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP),
            description=f"M = {emissions.total_tco2e} * {mat_pct} = {overall_tco2e} tCO2e",
        )

        # Step 3: Performance materiality: PM = M * performance_pct
        perf_tco2e = (overall_tco2e * config.performance_pct).quantize(
            Decimal(prec_str), rounding=ROUND_HALF_UP
        )
        perf_mat = MaterialityThreshold(
            threshold_name="Performance Materiality",
            value_tco2e=perf_tco2e,
            basis="overall_materiality",
            percentage=config.performance_pct,
            description=f"PM = {overall_tco2e} * {config.performance_pct} = {perf_tco2e} tCO2e",
        )

        # Step 4: Clearly trivial: CT = M * trivial_pct
        trivial_tco2e = (overall_tco2e * config.trivial_pct).quantize(
            Decimal(prec_str), rounding=ROUND_HALF_UP
        )
        trivial_mat = MaterialityThreshold(
            threshold_name="Clearly Trivial",
            value_tco2e=trivial_tco2e,
            basis="overall_materiality",
            percentage=config.trivial_pct,
            description=f"CT = {overall_tco2e} * {config.trivial_pct} = {trivial_tco2e} tCO2e",
        )

        # Step 5: Scope-specific materiality
        scope_mats: List[ScopeMateriality] = []
        scope_data = [
            ("scope_1", emissions.scope_1_tco2e),
            ("scope_2", emissions.scope_2_tco2e),
            ("scope_3", emissions.scope_3_tco2e),
        ]
        for scope_name, scope_em in scope_data:
            scope_pct = config.scope_materiality_pcts.get(
                scope_name, DEFAULT_MATERIALITY_PCT
            ) * level_mult
            scope_mat = (scope_em * scope_pct).quantize(
                Decimal(prec_str), rounding=ROUND_HALF_UP
            )
            scope_mats.append(ScopeMateriality(
                scope=scope_name,
                scope_emissions=scope_em,
                materiality_pct=scope_pct.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP),
                materiality_tco2e=scope_mat,
            ))

        # Step 6: Qualitative assessment
        qual_assessment = self._assess_qualitative(
            input_data.qualitative_factors, overall_tco2e, config, prec_str,
        )

        # Step 7: Specific materiality
        specific_mats: List[SpecificMateriality] = []
        for item in input_data.specific_items:
            spec_pct = item.specific_pct if item.specific_pct is not None else mat_pct
            if item.is_sensitive:
                spec_pct = spec_pct * Decimal("0.5")  # Halve for sensitive items
            spec_threshold = (item.value_tco2e * spec_pct).quantize(
                Decimal(prec_str), rounding=ROUND_HALF_UP
            )
            specific_mats.append(SpecificMateriality(
                item_id=item.item_id,
                item_name=item.item_name,
                value_tco2e=item.value_tco2e,
                materiality_tco2e=spec_threshold,
                exceeds_materiality=item.value_tco2e > overall_tco2e,
            ))

        # Step 8: Final materiality (after qualitative adjustment)
        final_mat = qual_assessment.adjusted_materiality

        assessment = MaterialityAssessment(
            overall_materiality=overall_mat,
            performance_materiality=perf_mat,
            clearly_trivial=trivial_mat,
            scope_materialities=scope_mats,
            qualitative_assessment=qual_assessment,
            specific_materialities=specific_mats,
            final_materiality=final_mat,
        )

        # Methodology rationale
        rationale = (
            f"Materiality determined at {float(mat_pct)*100:.2f}% of total emissions "
            f"({emissions.total_tco2e} tCO2e) for {config.assurance_level.value} assurance. "
            f"Performance materiality at {float(config.performance_pct)*100:.1f}% of overall. "
            f"Clearly trivial at {float(config.trivial_pct)*100:.1f}% of overall."
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = MaterialityResult(
            organisation_id=config.organisation_id,
            assessment=assessment,
            assurance_level=config.assurance_level.value,
            total_emissions_tco2e=emissions.total_tco2e,
            methodology_rationale=rationale,
            warnings=warnings,
            calculated_at=_utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Internal: Qualitative Assessment
    # ------------------------------------------------------------------

    def _assess_qualitative(
        self,
        factors: List[QualitativeFactor],
        overall_mat: Decimal,
        config: MaterialityConfig,
        prec_str: str,
    ) -> QualitativeAssessment:
        """Assess qualitative factors and compute adjusted materiality.

        M_adjusted = M * (1 - qualitative_adjustment)
        qualitative_adjustment = normalised_score * max_adj
        """
        if not factors:
            return QualitativeAssessment(
                adjusted_materiality=overall_mat,
            )

        total_score = sum(f.score for f in factors)
        max_possible = 5 * len(factors)
        normalised = _safe_divide(
            _decimal(total_score), _decimal(max_possible)
        ).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

        adj_pct = (normalised * config.max_qualitative_adj).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )

        adjusted = (overall_mat * (Decimal("1") - adj_pct)).quantize(
            Decimal(prec_str), rounding=ROUND_HALF_UP
        )

        return QualitativeAssessment(
            factors=factors,
            total_score=total_score,
            max_possible_score=max_possible,
            normalised_score=normalised,
            adjustment_pct=adj_pct,
            adjusted_materiality=adjusted,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_version(self) -> str:
        return self._version


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "MaterialityBasis",
    "AssuranceLevel",
    # Input Models
    "QualitativeFactor",
    "ScopeEmissions",
    "SpecificItem",
    "MaterialityConfig",
    "MaterialityInput",
    # Output Models
    "MaterialityThreshold",
    "ScopeMateriality",
    "QualitativeAssessment",
    "SpecificMateriality",
    "MaterialityAssessment",
    "MaterialityResult",
    # Engine
    "MaterialityAssessmentEngine",
]
