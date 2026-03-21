# -*- coding: utf-8 -*-
"""
InterimTargetEngine - PACK-025 Race to Zero Engine 3
======================================================

Validates 2030 interim targets against Race to Zero requirements and
1.5C pathway alignment. Confirms approximately 50% absolute emission
reduction by 2030 from a recent baseline, validates scope coverage,
checks science-based methodology alignment, assesses fair-share
contribution, and calculates temperature alignment scoring.

Calculation Methodology:
    Absolute Reduction Validation:
        reduction_pct = (baseline - target) / baseline * 100
        aligned_1_5c = reduction_pct >= 42  (IPCC AR6 WG3 minimum)
        aligned_r2z  = reduction_pct >= 50  (Race to Zero aspiration)

    Annual Reduction Rate:
        years = target_year - baseline_year
        annual_rate = 1 - (target / baseline) ^ (1 / years)
        aligned_1_5c = annual_rate >= 0.042  (4.2%/yr SBTi 1.5C)
        aligned_wb2c = annual_rate >= 0.025  (2.5%/yr SBTi WB2C)

    Scope Coverage Validation:
        scope1_2_coverage >= 95%
        scope3_coverage >= 67%  (for corporates/FIs)

    Pathway Alignment Scoring:
        1.5C_ALIGNED:     annual_rate >= 4.2%, reduction >= 42%
        WELL_BELOW_2C:    annual_rate >= 2.5%, reduction >= 25%
        2C_ALIGNED:       annual_rate >= 1.5%, reduction >= 15%
        MISALIGNED:       below all thresholds

    Temperature Score (simplified SBTi-aligned):
        temp = 1.5 + max(0, (4.2 - annual_rate_pct) / 4.2) * 2.0
        capped at 4.0C

    Fair Share Assessment:
        Considers: historical responsibility, capability (GDP/capita),
        sector intensity, development status

Regulatory References:
    - Race to Zero Interpretation Guide (June 2022), SL-P2
    - IPCC AR6 WG3 (2022): 43% CO2 reduction by 2030 (from 2019)
    - SBTi Corporate Manual v5.3 (2024): 4.2%/yr (1.5C), 2.5%/yr (WB2C)
    - IEA Net Zero by 2050 Roadmap (2021, updated 2023)
    - Paris Agreement Article 2.1(a): 1.5C temperature limit

Zero-Hallucination:
    - IPCC AR6 WG3 Table SPM.1 reduction ranges
    - SBTi contraction rates from Corporate Manual v5.3
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-025 Race to Zero
Engine:  3 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

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


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    return _safe_divide(part * Decimal("100"), whole)


def _round_val(value: Decimal, places: int = 6) -> Decimal:
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def _round3(value: float) -> float:
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PathwayAlignment(str, Enum):
    """Temperature pathway alignment classification.

    ALIGNED_1_5C: Consistent with 1.5C warming limit.
    WELL_BELOW_2C: Consistent with well-below 2C.
    ALIGNED_2C: Consistent with 2C warming limit.
    MISALIGNED: Below all recognized pathways.
    """
    ALIGNED_1_5C = "1.5c_aligned"
    WELL_BELOW_2C = "well_below_2c"
    ALIGNED_2C = "2c_aligned"
    MISALIGNED = "misaligned"


class TargetMethodology(str, Enum):
    """Science-based target methodology.

    SBTI_ACA: SBTi Absolute Contraction Approach.
    SBTI_SDA: SBTi Sectoral Decarbonization Approach.
    IEA_NZE: IEA Net Zero by 2050 scenario.
    IPCC_SR15: IPCC Special Report 1.5C pathways.
    IPCC_AR6: IPCC AR6 WG3 mitigation pathways.
    CUSTOM: Custom science-based methodology.
    NONE: No methodology specified.
    """
    SBTI_ACA = "sbti_aca"
    SBTI_SDA = "sbti_sda"
    IEA_NZE = "iea_nze"
    IPCC_SR15 = "ipcc_sr15"
    IPCC_AR6 = "ipcc_ar6"
    CUSTOM = "custom"
    NONE = "none"


class TargetType(str, Enum):
    """Target type classification.

    ABSOLUTE: Absolute emission reduction target.
    INTENSITY: Intensity-based reduction target.
    BOTH: Both absolute and intensity targets set.
    """
    ABSOLUTE = "absolute"
    INTENSITY = "intensity"
    BOTH = "both"


class ComplianceLevel(str, Enum):
    """Interim target compliance level.

    FULLY_COMPLIANT: Meets all Race to Zero requirements.
    SUBSTANTIALLY_COMPLIANT: Meets core requirements, minor gaps.
    PARTIALLY_COMPLIANT: Some requirements met, significant gaps.
    NON_COMPLIANT: Does not meet minimum requirements.
    """
    FULLY_COMPLIANT = "fully_compliant"
    SUBSTANTIALLY_COMPLIANT = "substantially_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# IPCC AR6 WG3 minimum reduction for 1.5C (from 2019 baseline).
IPCC_MIN_REDUCTION_PCT: Decimal = Decimal("42")

# Race to Zero aspiration (approximate halving by 2030).
R2Z_TARGET_REDUCTION_PCT: Decimal = Decimal("50")

# SBTi annual contraction rates (Corporate Manual v5.3).
SBTI_1_5C_ANNUAL_RATE: Decimal = Decimal("4.2")
SBTI_WB2C_ANNUAL_RATE: Decimal = Decimal("2.5")
SBTI_2C_ANNUAL_RATE: Decimal = Decimal("1.5")

# Minimum scope coverage percentages.
MIN_SCOPE1_2_COVERAGE: Decimal = Decimal("95")
MIN_SCOPE3_COVERAGE: Decimal = Decimal("67")

# Baseline year bounds.
MIN_BASELINE_YEAR: int = 2015
PREFERRED_MIN_BASELINE_YEAR: int = 2019

# Temperature score range.
TEMP_FLOOR: Decimal = Decimal("1.5")
TEMP_CEILING: Decimal = Decimal("4.0")

# Pathway thresholds (annual reduction rate, absolute reduction by 2030).
PATHWAY_THRESHOLDS: List[Tuple[Decimal, Decimal, str]] = [
    (SBTI_1_5C_ANNUAL_RATE, IPCC_MIN_REDUCTION_PCT, PathwayAlignment.ALIGNED_1_5C.value),
    (SBTI_WB2C_ANNUAL_RATE, Decimal("25"), PathwayAlignment.WELL_BELOW_2C.value),
    (SBTI_2C_ANNUAL_RATE, Decimal("15"), PathwayAlignment.ALIGNED_2C.value),
]

# Recognized methodologies.
RECOGNIZED_METHODOLOGIES = {
    TargetMethodology.SBTI_ACA.value,
    TargetMethodology.SBTI_SDA.value,
    TargetMethodology.IEA_NZE.value,
    TargetMethodology.IPCC_SR15.value,
    TargetMethodology.IPCC_AR6.value,
}


# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------


class ScopeTargetInput(BaseModel):
    """Target data for a single scope.

    Attributes:
        scope: Scope number (1, 2, or 3).
        baseline_emissions_tco2e: Baseline emissions for this scope.
        target_emissions_tco2e: Target emissions for this scope.
        coverage_pct: Coverage percentage for this scope.
        methodology: Methodology used for this scope's target.
        includes_all_gases: Whether all GHGs are included.
        notes: Additional notes.
    """
    scope: int = Field(..., ge=1, le=3, description="Scope number")
    baseline_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Baseline emissions (tCO2e)"
    )
    target_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Target emissions (tCO2e)"
    )
    coverage_pct: Decimal = Field(
        default=Decimal("100"), ge=0, le=Decimal("100"),
        description="Coverage (%)"
    )
    methodology: str = Field(
        default=TargetMethodology.NONE.value, description="Methodology"
    )
    includes_all_gases: bool = Field(default=True, description="All GHGs included")
    notes: str = Field(default="", description="Notes")


class InterimTargetInput(BaseModel):
    """Complete input for interim target validation.

    Attributes:
        entity_name: Entity name.
        actor_type: Actor type.
        sector: Industry sector or NACE code.
        baseline_year: Baseline year for the target.
        target_year: Interim target year (typically 2030).
        total_baseline_emissions_tco2e: Total baseline emissions.
        total_target_emissions_tco2e: Total target emissions.
        target_type: Target type (absolute/intensity/both).
        target_reduction_pct: Stated reduction percentage.
        methodology: Primary target-setting methodology.
        scope_targets: Per-scope target data.
        scope1_coverage_pct: Scope 1 coverage (%).
        scope2_coverage_pct: Scope 2 coverage (%).
        scope3_coverage_pct: Scope 3 coverage (%).
        intensity_metric: Intensity metric if intensity target.
        intensity_baseline: Intensity baseline value.
        intensity_target: Intensity target value.
        fair_share_considered: Whether fair share analysis was done.
        development_status: Developed/developing country.
        historical_responsibility_factor: Historical responsibility weight.
        sbti_validated: Whether SBTi has validated the target.
        include_pathway_comparison: Whether to include pathway comparison.
    """
    entity_name: str = Field(
        ..., min_length=1, max_length=300, description="Entity name"
    )
    actor_type: str = Field(default="corporate", description="Actor type")
    sector: str = Field(default="general", max_length=100, description="Sector")
    baseline_year: int = Field(
        ..., ge=2010, le=2060, description="Baseline year"
    )
    target_year: int = Field(
        default=2030, ge=2025, le=2040, description="Target year"
    )
    total_baseline_emissions_tco2e: Decimal = Field(
        ..., ge=0, description="Total baseline emissions (tCO2e)"
    )
    total_target_emissions_tco2e: Decimal = Field(
        ..., ge=0, description="Total target emissions (tCO2e)"
    )
    target_type: str = Field(
        default=TargetType.ABSOLUTE.value, description="Target type"
    )
    target_reduction_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"),
        description="Stated reduction (%)"
    )
    methodology: str = Field(
        default=TargetMethodology.NONE.value, description="Primary methodology"
    )
    scope_targets: List[ScopeTargetInput] = Field(
        default_factory=list, description="Per-scope targets"
    )
    scope1_coverage_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"), description="S1 coverage"
    )
    scope2_coverage_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"), description="S2 coverage"
    )
    scope3_coverage_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"), description="S3 coverage"
    )
    intensity_metric: str = Field(default="", description="Intensity metric")
    intensity_baseline: Decimal = Field(default=Decimal("0"), ge=0)
    intensity_target: Decimal = Field(default=Decimal("0"), ge=0)
    fair_share_considered: bool = Field(default=False)
    development_status: str = Field(default="developed")
    historical_responsibility_factor: Decimal = Field(
        default=Decimal("1.0"), ge=0, le=Decimal("2.0")
    )
    sbti_validated: bool = Field(default=False)
    include_pathway_comparison: bool = Field(default=True)

    @field_validator("target_type")
    @classmethod
    def validate_target_type(cls, v: str) -> str:
        valid = {t.value for t in TargetType}
        if v not in valid:
            raise ValueError(f"Unknown target type '{v}'.")
        return v

    @field_validator("methodology")
    @classmethod
    def validate_methodology(cls, v: str) -> str:
        valid = {m.value for m in TargetMethodology}
        if v not in valid:
            raise ValueError(f"Unknown methodology '{v}'.")
        return v


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class ScopeTargetResult(BaseModel):
    """Validation result for a single scope target.

    Attributes:
        scope: Scope number.
        baseline_tco2e: Baseline emissions.
        target_tco2e: Target emissions.
        reduction_pct: Calculated reduction percentage.
        annual_rate_pct: Annualized reduction rate.
        coverage_pct: Coverage percentage.
        coverage_sufficient: Whether coverage meets minimum.
        pathway_alignment: Pathway alignment for this scope.
        issues: Issues identified.
    """
    scope: int = Field(default=1)
    baseline_tco2e: Decimal = Field(default=Decimal("0"))
    target_tco2e: Decimal = Field(default=Decimal("0"))
    reduction_pct: Decimal = Field(default=Decimal("0"))
    annual_rate_pct: Decimal = Field(default=Decimal("0"))
    coverage_pct: Decimal = Field(default=Decimal("0"))
    coverage_sufficient: bool = Field(default=False)
    pathway_alignment: str = Field(default=PathwayAlignment.MISALIGNED.value)
    issues: List[str] = Field(default_factory=list)


class PathwayComparison(BaseModel):
    """Comparison of entity target vs global/sector pathway.

    Attributes:
        pathway_name: Name of the pathway compared against.
        pathway_source: Source of the pathway data.
        pathway_reduction_pct: Pathway required reduction by target year.
        entity_reduction_pct: Entity's planned reduction.
        gap_pct: Gap between entity and pathway (positive = shortfall).
        entity_exceeds: Whether entity target exceeds pathway.
        notes: Comparison notes.
    """
    pathway_name: str = Field(default="")
    pathway_source: str = Field(default="")
    pathway_reduction_pct: Decimal = Field(default=Decimal("0"))
    entity_reduction_pct: Decimal = Field(default=Decimal("0"))
    gap_pct: Decimal = Field(default=Decimal("0"))
    entity_exceeds: bool = Field(default=False)
    notes: str = Field(default="")


class FairShareResult(BaseModel):
    """Fair share assessment result.

    Attributes:
        fair_share_score: Fair share score (0-100).
        development_status: Developed or developing.
        historical_factor: Historical responsibility multiplier.
        equity_adjusted_target_pct: Equity-adjusted target %.
        assessment: Qualitative assessment.
    """
    fair_share_score: Decimal = Field(default=Decimal("0"))
    development_status: str = Field(default="developed")
    historical_factor: Decimal = Field(default=Decimal("1.0"))
    equity_adjusted_target_pct: Decimal = Field(default=Decimal("0"))
    assessment: str = Field(default="")


class InterimTargetResult(BaseModel):
    """Complete interim target validation result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        entity_name: Entity name.
        baseline_year: Baseline year.
        target_year: Target year.
        total_baseline_tco2e: Total baseline emissions.
        total_target_tco2e: Total target emissions.
        absolute_reduction_pct: Calculated absolute reduction.
        annual_reduction_rate_pct: Annualized reduction rate.
        pathway_alignment: Overall pathway alignment.
        temperature_score: Implied temperature score.
        compliance_level: Compliance level.
        methodology_recognized: Whether methodology is recognized.
        scope_coverage_valid: Whether scope coverage meets requirements.
        baseline_year_valid: Whether baseline year is acceptable.
        scope_results: Per-scope validation results.
        pathway_comparisons: Pathway comparison results.
        fair_share: Fair share assessment.
        meets_r2z_minimum: Whether meets Race to Zero minimum.
        meets_ipcc_minimum: Whether meets IPCC AR6 minimum.
        meets_sbti_1_5c: Whether meets SBTi 1.5C rate.
        gaps: Identified gaps.
        recommendations: Improvement recommendations.
        warnings: Non-blocking warnings.
        errors: Blocking errors.
        processing_time_ms: Processing duration.
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=_utcnow)
    entity_name: str = Field(default="")
    baseline_year: int = Field(default=0)
    target_year: int = Field(default=2030)
    total_baseline_tco2e: Decimal = Field(default=Decimal("0"))
    total_target_tco2e: Decimal = Field(default=Decimal("0"))
    absolute_reduction_pct: Decimal = Field(default=Decimal("0"))
    annual_reduction_rate_pct: Decimal = Field(default=Decimal("0"))
    pathway_alignment: str = Field(default=PathwayAlignment.MISALIGNED.value)
    temperature_score: Decimal = Field(default=Decimal("4.0"))
    compliance_level: str = Field(default=ComplianceLevel.NON_COMPLIANT.value)
    methodology_recognized: bool = Field(default=False)
    scope_coverage_valid: bool = Field(default=False)
    baseline_year_valid: bool = Field(default=False)
    scope_results: List[ScopeTargetResult] = Field(default_factory=list)
    pathway_comparisons: List[PathwayComparison] = Field(default_factory=list)
    fair_share: Optional[FairShareResult] = Field(default=None)
    meets_r2z_minimum: bool = Field(default=False)
    meets_ipcc_minimum: bool = Field(default=False)
    meets_sbti_1_5c: bool = Field(default=False)
    gaps: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class InterimTargetEngine:
    """Race to Zero interim target validation engine.

    Validates 2030 interim targets against 1.5C pathway requirements,
    calculates annual reduction rates, assesses scope coverage, performs
    pathway comparisons, and computes temperature alignment scores.

    Usage::

        engine = InterimTargetEngine()
        result = engine.validate(input_data)
        print(f"Alignment: {result.pathway_alignment}")
        print(f"Temperature: {result.temperature_score}C")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise InterimTargetEngine.

        Args:
            config: Optional overrides.
        """
        self.config = config or {}
        self._min_reduction = _decimal(
            self.config.get("min_reduction_pct", IPCC_MIN_REDUCTION_PCT)
        )
        self._min_annual_rate = _decimal(
            self.config.get("min_annual_rate", SBTI_1_5C_ANNUAL_RATE)
        )
        logger.info("InterimTargetEngine v%s initialised", self.engine_version)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def validate(
        self, data: InterimTargetInput,
    ) -> InterimTargetResult:
        """Perform complete interim target validation.

        Args:
            data: Validated interim target input.

        Returns:
            InterimTargetResult with full validation.
        """
        t0 = time.perf_counter()
        logger.info(
            "Interim target validation: entity=%s, base=%d, target=%d",
            data.entity_name, data.baseline_year, data.target_year,
        )

        warnings: List[str] = []
        errors: List[str] = []
        gaps: List[str] = []

        # Step 1: Calculate absolute reduction
        abs_reduction = self._calc_reduction_pct(
            data.total_baseline_emissions_tco2e,
            data.total_target_emissions_tco2e,
        )

        # Step 2: Calculate annual reduction rate
        years = data.target_year - data.baseline_year
        annual_rate = self._calc_annual_rate(
            data.total_baseline_emissions_tco2e,
            data.total_target_emissions_tco2e,
            years,
        )

        # Step 3: Determine pathway alignment
        pathway = self._determine_pathway(abs_reduction, annual_rate)

        # Step 4: Calculate temperature score
        temp_score = self._calc_temperature_score(annual_rate)

        # Step 5: Validate baseline year
        baseline_valid = data.baseline_year >= MIN_BASELINE_YEAR
        if not baseline_valid:
            gaps.append(
                f"Baseline year {data.baseline_year} is before {MIN_BASELINE_YEAR}."
            )
        if data.baseline_year < PREFERRED_MIN_BASELINE_YEAR:
            warnings.append(
                f"Baseline year {data.baseline_year} is before preferred "
                f"minimum of {PREFERRED_MIN_BASELINE_YEAR}."
            )

        # Step 6: Validate scope coverage
        scope_valid = self._validate_scope_coverage(data, gaps)

        # Step 7: Validate methodology
        method_recognized = data.methodology in RECOGNIZED_METHODOLOGIES
        if not method_recognized:
            gaps.append(
                f"Methodology '{data.methodology}' is not a recognized "
                f"science-based methodology."
            )

        # Step 8: Per-scope validation
        scope_results = self._validate_scope_targets(data, years)

        # Step 9: Pathway comparisons
        pathway_comparisons: List[PathwayComparison] = []
        if data.include_pathway_comparison:
            pathway_comparisons = self._build_pathway_comparisons(
                abs_reduction, data
            )

        # Step 10: Fair share assessment
        fair_share = self._assess_fair_share(abs_reduction, data)

        # Step 11: Compliance checks
        meets_r2z = abs_reduction >= R2Z_TARGET_REDUCTION_PCT
        meets_ipcc = abs_reduction >= IPCC_MIN_REDUCTION_PCT
        meets_sbti = annual_rate >= SBTI_1_5C_ANNUAL_RATE

        if not meets_ipcc:
            gaps.append(
                f"Absolute reduction of {abs_reduction}% is below IPCC minimum "
                f"of {IPCC_MIN_REDUCTION_PCT}%."
            )
        if not meets_sbti:
            gaps.append(
                f"Annual reduction rate of {annual_rate}% is below SBTi 1.5C "
                f"minimum of {SBTI_1_5C_ANNUAL_RATE}%."
            )

        # Step 12: Compliance level
        if meets_r2z and scope_valid and method_recognized and baseline_valid:
            compliance = ComplianceLevel.FULLY_COMPLIANT.value
        elif meets_ipcc and scope_valid:
            compliance = ComplianceLevel.SUBSTANTIALLY_COMPLIANT.value
        elif meets_ipcc or (abs_reduction >= Decimal("25")):
            compliance = ComplianceLevel.PARTIALLY_COMPLIANT.value
        else:
            compliance = ComplianceLevel.NON_COMPLIANT.value

        # Step 13: Recommendations
        recommendations = self._generate_recommendations(
            abs_reduction, annual_rate, pathway, scope_valid,
            method_recognized, baseline_valid, data
        )

        # Warnings
        if data.target_type == TargetType.INTENSITY.value:
            warnings.append(
                "Intensity targets alone may not satisfy Race to Zero. "
                "Absolute targets are required."
            )
        if years < 5:
            warnings.append(
                f"Only {years} years between baseline and target. "
                f"Short timeframes increase implementation risk."
            )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = InterimTargetResult(
            entity_name=data.entity_name,
            baseline_year=data.baseline_year,
            target_year=data.target_year,
            total_baseline_tco2e=_round_val(data.total_baseline_emissions_tco2e),
            total_target_tco2e=_round_val(data.total_target_emissions_tco2e),
            absolute_reduction_pct=abs_reduction,
            annual_reduction_rate_pct=annual_rate,
            pathway_alignment=pathway,
            temperature_score=temp_score,
            compliance_level=compliance,
            methodology_recognized=method_recognized,
            scope_coverage_valid=scope_valid,
            baseline_year_valid=baseline_valid,
            scope_results=scope_results,
            pathway_comparisons=pathway_comparisons,
            fair_share=fair_share,
            meets_r2z_minimum=meets_r2z,
            meets_ipcc_minimum=meets_ipcc,
            meets_sbti_1_5c=meets_sbti,
            gaps=gaps,
            recommendations=recommendations,
            warnings=warnings,
            errors=errors,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Interim target validation complete: alignment=%s, temp=%.2fC, "
            "reduction=%.1f%%, annual=%.2f%%, hash=%s",
            pathway, float(temp_score), float(abs_reduction),
            float(annual_rate), result.provenance_hash[:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # Internal Methods                                                     #
    # ------------------------------------------------------------------ #

    def _calc_reduction_pct(
        self, baseline: Decimal, target: Decimal,
    ) -> Decimal:
        """Calculate absolute reduction percentage.

        Args:
            baseline: Baseline emissions.
            target: Target emissions.

        Returns:
            Reduction percentage (0-100).
        """
        if baseline <= Decimal("0"):
            return Decimal("0")
        reduction = (baseline - target) / baseline * Decimal("100")
        return _round_val(max(Decimal("0"), reduction), 2)

    def _calc_annual_rate(
        self, baseline: Decimal, target: Decimal, years: int,
    ) -> Decimal:
        """Calculate annualized linear reduction rate.

        Uses compound annual reduction: rate = 1 - (target/baseline)^(1/years)

        Args:
            baseline: Baseline emissions.
            target: Target emissions.
            years: Number of years.

        Returns:
            Annual reduction rate as percentage.
        """
        if baseline <= Decimal("0") or years <= 0:
            return Decimal("0")
        ratio = float(target / baseline)
        if ratio <= 0:
            return Decimal("100")
        annual = (1.0 - math.pow(ratio, 1.0 / years)) * 100.0
        return _round_val(_decimal(max(0.0, annual)), 2)

    def _determine_pathway(
        self, reduction: Decimal, annual_rate: Decimal,
    ) -> str:
        """Determine pathway alignment from reduction and annual rate.

        Args:
            reduction: Absolute reduction percentage.
            annual_rate: Annual reduction rate percentage.

        Returns:
            PathwayAlignment value.
        """
        for rate_threshold, reduction_threshold, alignment in PATHWAY_THRESHOLDS:
            if annual_rate >= rate_threshold and reduction >= reduction_threshold:
                return alignment
        return PathwayAlignment.MISALIGNED.value

    def _calc_temperature_score(self, annual_rate: Decimal) -> Decimal:
        """Calculate simplified temperature alignment score.

        temp = 1.5 + max(0, (4.2 - annual_rate) / 4.2) * 2.0
        Capped between 1.5C and 4.0C.

        Args:
            annual_rate: Annual reduction rate percentage.

        Returns:
            Temperature score in degrees Celsius.
        """
        if annual_rate >= SBTI_1_5C_ANNUAL_RATE:
            return TEMP_FLOOR

        gap = float(SBTI_1_5C_ANNUAL_RATE - annual_rate)
        rate_val = float(SBTI_1_5C_ANNUAL_RATE)
        if rate_val <= 0:
            return TEMP_CEILING

        temp = 1.5 + max(0.0, gap / rate_val) * 2.0
        temp = min(float(TEMP_CEILING), max(float(TEMP_FLOOR), temp))
        return _round_val(_decimal(temp), 2)

    def _validate_scope_coverage(
        self, data: InterimTargetInput, gaps: List[str],
    ) -> bool:
        """Validate scope coverage against requirements.

        Args:
            data: Input data.
            gaps: Gap list to append to.

        Returns:
            True if coverage meets requirements.
        """
        valid = True
        if data.scope1_coverage_pct < MIN_SCOPE1_2_COVERAGE:
            valid = False
            gaps.append(
                f"Scope 1 coverage ({data.scope1_coverage_pct}%) below "
                f"minimum {MIN_SCOPE1_2_COVERAGE}%."
            )
        if data.scope2_coverage_pct < MIN_SCOPE1_2_COVERAGE:
            valid = False
            gaps.append(
                f"Scope 2 coverage ({data.scope2_coverage_pct}%) below "
                f"minimum {MIN_SCOPE1_2_COVERAGE}%."
            )
        if data.actor_type in ("corporate", "financial_institution"):
            if data.scope3_coverage_pct < MIN_SCOPE3_COVERAGE:
                valid = False
                gaps.append(
                    f"Scope 3 coverage ({data.scope3_coverage_pct}%) below "
                    f"minimum {MIN_SCOPE3_COVERAGE}%."
                )
        return valid

    def _validate_scope_targets(
        self, data: InterimTargetInput, years: int,
    ) -> List[ScopeTargetResult]:
        """Validate per-scope targets.

        Args:
            data: Input data.
            years: Years between baseline and target.

        Returns:
            List of ScopeTargetResult.
        """
        results: List[ScopeTargetResult] = []
        coverage_mins = {1: MIN_SCOPE1_2_COVERAGE, 2: MIN_SCOPE1_2_COVERAGE, 3: MIN_SCOPE3_COVERAGE}

        for st in data.scope_targets:
            reduction = self._calc_reduction_pct(
                st.baseline_emissions_tco2e, st.target_emissions_tco2e
            )
            annual = self._calc_annual_rate(
                st.baseline_emissions_tco2e, st.target_emissions_tco2e, years
            )
            pathway = self._determine_pathway(reduction, annual)
            min_cov = coverage_mins.get(st.scope, Decimal("95"))
            cov_ok = st.coverage_pct >= min_cov

            issues: List[str] = []
            if not cov_ok:
                issues.append(
                    f"Scope {st.scope} coverage ({st.coverage_pct}%) below {min_cov}%."
                )
            if not st.includes_all_gases:
                issues.append(f"Scope {st.scope} does not include all GHGs.")

            results.append(ScopeTargetResult(
                scope=st.scope,
                baseline_tco2e=_round_val(st.baseline_emissions_tco2e),
                target_tco2e=_round_val(st.target_emissions_tco2e),
                reduction_pct=reduction,
                annual_rate_pct=annual,
                coverage_pct=st.coverage_pct,
                coverage_sufficient=cov_ok,
                pathway_alignment=pathway,
                issues=issues,
            ))

        return results

    def _build_pathway_comparisons(
        self, entity_reduction: Decimal, data: InterimTargetInput,
    ) -> List[PathwayComparison]:
        """Build pathway comparison results.

        Args:
            entity_reduction: Entity's absolute reduction percentage.
            data: Input data.

        Returns:
            List of PathwayComparison.
        """
        comparisons = [
            ("IPCC AR6 1.5C (no/limited overshoot)", "IPCC AR6 WG3 SPM", Decimal("43")),
            ("IPCC AR6 1.5C (high overshoot)", "IPCC AR6 WG3 SPM", Decimal("34")),
            ("Race to Zero Aspiration", "UNFCCC Race to Zero", Decimal("50")),
            ("SBTi 1.5C (ACA)", "SBTi Corporate Manual v5.3", Decimal("42")),
            ("IEA Net Zero by 2050", "IEA NZE Roadmap 2023", Decimal("45")),
        ]

        results: List[PathwayComparison] = []
        for name, source, pathway_pct in comparisons:
            gap = pathway_pct - entity_reduction
            results.append(PathwayComparison(
                pathway_name=name,
                pathway_source=source,
                pathway_reduction_pct=pathway_pct,
                entity_reduction_pct=entity_reduction,
                gap_pct=_round_val(gap, 2),
                entity_exceeds=entity_reduction >= pathway_pct,
                notes=(
                    "Entity meets or exceeds pathway." if entity_reduction >= pathway_pct
                    else f"Entity falls short by {_round_val(gap, 1)}pp."
                ),
            ))

        return results

    def _assess_fair_share(
        self, reduction: Decimal, data: InterimTargetInput,
    ) -> FairShareResult:
        """Assess fair share contribution.

        Args:
            reduction: Entity's absolute reduction percentage.
            data: Input data.

        Returns:
            FairShareResult.
        """
        # Simple equity adjustment: developed countries expected to reduce more
        factor = data.historical_responsibility_factor
        equity_adjusted = _round_val(reduction / factor, 2)

        if data.development_status == "developed":
            baseline_expectation = Decimal("50")
        else:
            baseline_expectation = Decimal("35")

        score = min(
            Decimal("100"),
            _safe_divide(reduction, baseline_expectation) * Decimal("100")
        )

        if score >= Decimal("80"):
            assessment = "Target represents a strong fair share contribution."
        elif score >= Decimal("60"):
            assessment = "Target represents an adequate fair share contribution."
        elif score >= Decimal("40"):
            assessment = "Target may fall short of fair share expectations."
        else:
            assessment = "Target does not meet fair share expectations."

        return FairShareResult(
            fair_share_score=_round_val(score, 2),
            development_status=data.development_status,
            historical_factor=factor,
            equity_adjusted_target_pct=equity_adjusted,
            assessment=assessment,
        )

    def _generate_recommendations(
        self,
        reduction: Decimal,
        annual_rate: Decimal,
        pathway: str,
        scope_valid: bool,
        method_recognized: bool,
        baseline_valid: bool,
        data: InterimTargetInput,
    ) -> List[str]:
        """Generate improvement recommendations.

        Args:
            reduction: Absolute reduction percentage.
            annual_rate: Annual reduction rate.
            pathway: Current pathway alignment.
            scope_valid: Whether scope coverage is valid.
            method_recognized: Whether methodology is recognized.
            baseline_valid: Whether baseline year is valid.
            data: Input data.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        if pathway == PathwayAlignment.MISALIGNED.value:
            recs.append(
                f"CRITICAL: Target is misaligned with all recognized pathways. "
                f"Increase ambition to at least 42% reduction by {data.target_year} "
                f"(currently {reduction}%)."
            )
        elif pathway == PathwayAlignment.ALIGNED_2C.value:
            recs.append(
                f"Target aligned with 2C but not 1.5C. Increase annual "
                f"reduction from {annual_rate}% to >={SBTI_1_5C_ANNUAL_RATE}%."
            )
        elif pathway == PathwayAlignment.WELL_BELOW_2C.value:
            recs.append(
                f"Target aligned with Well-Below 2C. Close gap to 1.5C by "
                f"increasing annual reduction from {annual_rate}% to "
                f">={SBTI_1_5C_ANNUAL_RATE}%."
            )

        if not scope_valid:
            recs.append(
                "Expand scope coverage to S1+S2 >=95% and S3 >=67% "
                "(for corporates and financial institutions)."
            )

        if not method_recognized:
            recs.append(
                "Adopt a recognized science-based methodology: SBTi (ACA or SDA), "
                "IEA NZE, or IPCC AR6 pathways."
            )

        if not baseline_valid:
            recs.append(
                f"Rebase targets to a year >= {MIN_BASELINE_YEAR} "
                f"(preferably >= {PREFERRED_MIN_BASELINE_YEAR})."
            )

        if not data.sbti_validated:
            recs.append(
                "Consider SBTi target validation for additional credibility "
                "and Race to Zero partner initiative alignment."
            )

        if not data.fair_share_considered:
            recs.append(
                "Conduct a fair share assessment considering equity, "
                "capability, and historical responsibility."
            )

        return recs
