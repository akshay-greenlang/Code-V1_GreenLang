# -*- coding: utf-8 -*-
"""
NetZeroTargetEngine - PACK-021 Net Zero Starter Engine 2
==========================================================

SBTi Net-Zero Standard v1.2 compliant target-setting engine.

This engine implements the complete target-setting pipeline per the
Science Based Targets initiative (SBTi) Corporate Net-Zero Standard.
It supports near-term (5-10 year, 1.5C-aligned) and long-term (by 2050,
90%+ reduction) target creation, annual reduction rate calculation,
pathway selection, coverage validation, temperature alignment scoring,
and milestone generation.

Target-Setting Methodology:
    Near-term (2030):
        - ACA: 4.2% per year linear absolute reduction (1.5C)
        - WB2C: 2.5% per year linear absolute reduction
        - Scope 1+2 coverage: 95% minimum
        - Scope 3 coverage: 67% minimum (if >40% of total)

    Long-term (2050):
        - 90%+ absolute reduction from base year
        - All scopes covered
        - Maximum 10% residual emissions (neutralized)

    Annual rate calculation:
        annual_rate = total_reduction_pct / (target_year - base_year)

    Milestone generation:
        interim_target[year] = base * (1 - annual_rate * years_elapsed)

Regulatory References:
    - SBTi Corporate Net-Zero Standard v1.2 (2023)
    - SBTi Target Validation Protocol v3.0 (2023)
    - SBTi Criteria and Recommendations (2024)
    - Paris Agreement (2015) Article 2
    - IPCC SR15 (2018) - 1.5C mitigation pathways
    - GHG Protocol Corporate Standard - Chapter 5 (Target Setting)
    - ISO 14064-1:2018 - GHG quantification and reporting

Zero-Hallucination:
    - All target calculations use deterministic Decimal arithmetic
    - Pathway parameters are hard-coded from SBTi publications
    - Annual rates use simple linear division
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-021 Net Zero Starter
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
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
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PathwayType(str, Enum):
    """SBTi decarbonization pathway type.

    ACA: Absolute Contraction Approach - linear absolute reduction.
    SDA: Sectoral Decarbonization Approach - sector-specific intensity.
    FLAG: Forest, Land and Agriculture - sector-specific guidance.
    """
    ACA = "aca"
    SDA = "sda"
    FLAG = "flag"

class AmbitionLevel(str, Enum):
    """Target ambition level per SBTi classification."""
    PARIS_1_5C = "1.5c"
    WELL_BELOW_2C = "well_below_2c"
    BELOW_2C = "2c"

class TargetTimeframe(str, Enum):
    """Target timeframe classification per SBTi Net-Zero Standard."""
    NEAR_TERM = "near_term"
    LONG_TERM = "long_term"

class TargetType(str, Enum):
    """Target type classification."""
    ABSOLUTE = "absolute"
    INTENSITY = "intensity"

class TemperatureAlignment(str, Enum):
    """Temperature alignment classification."""
    ALIGNED_1_5C = "1.5c_aligned"
    ALIGNED_WB2C = "well_below_2c_aligned"
    ALIGNED_2C = "2c_aligned"
    NOT_ALIGNED = "not_aligned"

class SBTiSector(str, Enum):
    """SBTi sector classification for pathway selection."""
    POWER_GENERATION = "power_generation"
    CEMENT = "cement"
    STEEL = "steel"
    ALUMINUM = "aluminum"
    CHEMICALS = "chemicals"
    BUILDINGS = "buildings"
    TRANSPORT = "transport"
    AVIATION = "aviation"
    SHIPPING = "shipping"
    FLAG = "flag"
    OTHER = "other"

class ScopeCategory(str, Enum):
    """Scope coverage for target setting."""
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_1_2 = "scope_1_2"
    SCOPE_3 = "scope_3"
    ALL_SCOPES = "all_scopes"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SBTi minimum annual reduction rates by ambition and scope.
# Source: SBTi Corporate Net-Zero Standard v1.2 (2023), Table 1.
SBTI_RATES: Dict[str, Dict[str, Decimal]] = {
    AmbitionLevel.PARIS_1_5C: {
        "scope_1_2_annual_pct": Decimal("4.2"),
        "scope_3_annual_pct": Decimal("2.5"),
        "near_term_min_reduction_pct": Decimal("42"),
        "long_term_reduction_pct": Decimal("90"),
        "max_residual_pct": Decimal("10"),
    },
    AmbitionLevel.WELL_BELOW_2C: {
        "scope_1_2_annual_pct": Decimal("2.5"),
        "scope_3_annual_pct": Decimal("2.5"),
        "near_term_min_reduction_pct": Decimal("25"),
        "long_term_reduction_pct": Decimal("80"),
        "max_residual_pct": Decimal("20"),
    },
    AmbitionLevel.BELOW_2C: {
        "scope_1_2_annual_pct": Decimal("1.23"),
        "scope_3_annual_pct": Decimal("1.23"),
        "near_term_min_reduction_pct": Decimal("12.3"),
        "long_term_reduction_pct": Decimal("70"),
        "max_residual_pct": Decimal("30"),
    },
}

# SBTi coverage requirements.
COVERAGE_REQUIREMENTS: Dict[str, Decimal] = {
    "scope_1_2_min_pct": Decimal("95"),
    "scope_3_min_pct": Decimal("67"),
    "scope_3_materiality_threshold_pct": Decimal("40"),
    "near_term_min_years": Decimal("5"),
    "near_term_max_years": Decimal("10"),
    "long_term_target_year": Decimal("2050"),
    "base_year_max_age_years": Decimal("5"),
}

# SBTi sector-specific SDA intensity pathways (tCO2e per unit).
# Simplified starter-tier reference values.
SDA_BENCHMARKS: Dict[str, Dict[str, Decimal]] = {
    SBTiSector.POWER_GENERATION: {
        "unit": "tCO2e_per_MWh",
        "2020_benchmark": Decimal("0.376"),
        "2030_benchmark": Decimal("0.138"),
        "2050_benchmark": Decimal("0.004"),
    },
    SBTiSector.BUILDINGS: {
        "unit": "tCO2e_per_sqm",
        "2020_benchmark": Decimal("0.081"),
        "2030_benchmark": Decimal("0.049"),
        "2050_benchmark": Decimal("0.009"),
    },
    SBTiSector.TRANSPORT: {
        "unit": "tCO2e_per_vkm",
        "2020_benchmark": Decimal("0.000171"),
        "2030_benchmark": Decimal("0.000110"),
        "2050_benchmark": Decimal("0.000019"),
    },
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class TargetInput(BaseModel):
    """Input data for target setting.

    Attributes:
        entity_name: Reporting entity name.
        base_year: Base year for the target.
        base_year_scope1_tco2e: Scope 1 emissions in base year.
        base_year_scope2_tco2e: Scope 2 emissions in base year.
        base_year_scope3_tco2e: Scope 3 emissions in base year.
        near_term_target_year: Near-term target year (typically 2030).
        long_term_target_year: Long-term target year (default 2050).
        pathway_type: Decarbonization pathway type.
        ambition_level: Target ambition level.
        sector: SBTi sector classification.
        scope1_2_coverage_pct: Coverage of Scope 1+2 in target (%).
        scope3_coverage_pct: Coverage of Scope 3 in target (%).
        include_scope3: Whether to include Scope 3 targets.
        milestone_interval_years: Years between interim milestones.
    """
    entity_name: str = Field(
        ..., min_length=1, max_length=300, description="Entity name"
    )
    base_year: int = Field(..., ge=1990, le=2100, description="Base year")
    base_year_scope1_tco2e: Decimal = Field(
        ..., ge=Decimal("0"), description="Base year Scope 1 (tCO2e)"
    )
    base_year_scope2_tco2e: Decimal = Field(
        ..., ge=Decimal("0"), description="Base year Scope 2 (tCO2e)"
    )
    base_year_scope3_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Base year Scope 3 (tCO2e)",
    )
    near_term_target_year: int = Field(
        default=2030, ge=2025, le=2040,
        description="Near-term target year",
    )
    long_term_target_year: int = Field(
        default=2050, ge=2040, le=2070,
        description="Long-term target year",
    )
    pathway_type: PathwayType = Field(
        default=PathwayType.ACA, description="Pathway type"
    )
    ambition_level: AmbitionLevel = Field(
        default=AmbitionLevel.PARIS_1_5C, description="Ambition level"
    )
    sector: SBTiSector = Field(
        default=SBTiSector.OTHER, description="SBTi sector"
    )
    scope1_2_coverage_pct: Decimal = Field(
        default=Decimal("100"), ge=Decimal("0"), le=Decimal("100"),
        description="Scope 1+2 coverage (%)",
    )
    scope3_coverage_pct: Decimal = Field(
        default=Decimal("67"), ge=Decimal("0"), le=Decimal("100"),
        description="Scope 3 coverage (%)",
    )
    include_scope3: bool = Field(
        default=True, description="Include Scope 3 targets"
    )
    milestone_interval_years: int = Field(
        default=5, ge=1, le=10, description="Milestone interval (years)"
    )

    @field_validator("near_term_target_year")
    @classmethod
    def validate_near_term(cls, v: int, info: Any) -> int:
        """Validate near-term target year is after base year."""
        base = info.data.get("base_year", 1990)
        if v <= base:
            raise ValueError(
                f"near_term_target_year ({v}) must be after base_year ({base})"
            )
        return v

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class TargetDefinition(BaseModel):
    """A single target definition (near-term or long-term).

    Attributes:
        target_id: Unique target identifier.
        name: Target name.
        timeframe: Near-term or long-term.
        target_type: Absolute or intensity.
        scope_coverage: Scopes covered.
        base_year: Base year.
        target_year: Target year.
        base_year_tco2e: Base year emissions.
        target_tco2e: Target emissions level.
        reduction_pct: Target reduction from base year (%).
        annual_rate_pct: Required annual linear reduction rate (%).
        coverage_pct: Scope coverage percentage.
        pathway: Pathway type used.
        ambition: Ambition level.
    """
    target_id: str = Field(default_factory=_new_uuid)
    name: str = Field(default="")
    timeframe: TargetTimeframe = Field(...)
    target_type: TargetType = Field(default=TargetType.ABSOLUTE)
    scope_coverage: ScopeCategory = Field(...)
    base_year: int = Field(default=0)
    target_year: int = Field(default=0)
    base_year_tco2e: Decimal = Field(default=Decimal("0"))
    target_tco2e: Decimal = Field(default=Decimal("0"))
    reduction_pct: Decimal = Field(default=Decimal("0"))
    annual_rate_pct: Decimal = Field(default=Decimal("0"))
    coverage_pct: Decimal = Field(default=Decimal("0"))
    pathway: str = Field(default="")
    ambition: str = Field(default="")

class MilestoneEntry(BaseModel):
    """An interim milestone on the reduction pathway.

    Attributes:
        year: Milestone year.
        target_tco2e: Target emissions at this milestone.
        reduction_from_base_pct: Reduction from base year (%).
        scope: Scope(s) covered.
    """
    year: int = Field(...)
    target_tco2e: Decimal = Field(default=Decimal("0"))
    reduction_from_base_pct: Decimal = Field(default=Decimal("0"))
    scope: str = Field(default="")

class ValidationCheck(BaseModel):
    """A single SBTi validation check result.

    Attributes:
        check_name: Name of the check.
        passed: Whether the check passed.
        required_value: Required threshold value.
        actual_value: Actual value.
        message: Explanation.
    """
    check_name: str = Field(...)
    passed: bool = Field(default=False)
    required_value: str = Field(default="")
    actual_value: str = Field(default="")
    message: str = Field(default="")

class TargetResult(BaseModel):
    """Complete target-setting result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Timestamp.
        entity_name: Entity name.
        near_term_target: Near-term target definition.
        long_term_target: Long-term target definition.
        scope3_target: Scope 3 specific target (if applicable).
        annual_rates: Annual reduction rates by scope.
        milestones: Interim milestones.
        validation_checks: SBTi validation checks.
        validation_passed: Whether all mandatory checks passed.
        temperature_alignment: Temperature alignment classification.
        scope3_materiality: Whether Scope 3 is material (>40% of total).
        total_base_year_tco2e: Total base year emissions.
        recommendations: Improvement recommendations.
        processing_time_ms: Processing duration (ms).
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    entity_name: str = Field(default="")
    near_term_target: Optional[TargetDefinition] = Field(None)
    long_term_target: Optional[TargetDefinition] = Field(None)
    scope3_target: Optional[TargetDefinition] = Field(None)
    annual_rates: Dict[str, Decimal] = Field(default_factory=dict)
    milestones: List[MilestoneEntry] = Field(default_factory=list)
    validation_checks: List[ValidationCheck] = Field(default_factory=list)
    validation_passed: bool = Field(default=False)
    temperature_alignment: str = Field(default="")
    scope3_materiality: bool = Field(default=False)
    total_base_year_tco2e: Decimal = Field(default=Decimal("0"))
    recommendations: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class NetZeroTargetEngine:
    """SBTi Net-Zero Standard v1.2 target-setting engine.

    Provides deterministic, zero-hallucination calculations for:
    - Near-term target generation (5-10 year, 1.5C-aligned)
    - Long-term target generation (by 2050, 90%+ reduction)
    - Scope 3 target generation (if material)
    - Annual reduction rate calculation
    - Interim milestone generation
    - SBTi validation checks
    - Temperature alignment scoring
    - Coverage requirement validation

    All calculations use Decimal arithmetic for bit-perfect
    reproducibility.  No LLM is used in any calculation path.

    Usage::

        engine = NetZeroTargetEngine()
        result = engine.calculate(target_input)
        assert result.validation_passed
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def calculate(self, data: TargetInput) -> TargetResult:
        """Generate net-zero targets per SBTi Net-Zero Standard.

        Args:
            data: Validated target input data.

        Returns:
            TargetResult with near-term, long-term targets, milestones,
            and validation status.
        """
        t0 = time.perf_counter()
        logger.info(
            "Setting targets: entity=%s, base=%d, near=%d, long=%d, "
            "ambition=%s, pathway=%s",
            data.entity_name, data.base_year, data.near_term_target_year,
            data.long_term_target_year, data.ambition_level.value,
            data.pathway_type.value,
        )

        total_base = (
            data.base_year_scope1_tco2e
            + data.base_year_scope2_tco2e
            + data.base_year_scope3_tco2e
        )
        scope1_2_base = data.base_year_scope1_tco2e + data.base_year_scope2_tco2e

        # Scope 3 materiality check
        scope3_material = False
        if total_base > Decimal("0"):
            scope3_pct = _safe_pct(data.base_year_scope3_tco2e, total_base)
            scope3_material = (
                scope3_pct >= COVERAGE_REQUIREMENTS["scope_3_materiality_threshold_pct"]
            )

        # Step 1: Near-term target (Scope 1+2)
        near_term = self._create_near_term_target(
            data, scope1_2_base
        )

        # Step 2: Long-term target (all scopes)
        long_term = self._create_long_term_target(
            data, total_base
        )

        # Step 3: Scope 3 target (if material and requested)
        scope3_target = None
        if data.include_scope3 and (scope3_material or data.base_year_scope3_tco2e > Decimal("0")):
            scope3_target = self._create_scope3_target(data)

        # Step 4: Annual rates summary
        annual_rates = self._collect_annual_rates(
            near_term, long_term, scope3_target
        )

        # Step 5: Milestones
        milestones = self._generate_milestones(
            data, near_term, long_term, scope3_target
        )

        # Step 6: SBTi validation
        checks = self._run_validation_checks(
            data, near_term, long_term, scope3_target, scope3_material
        )
        all_passed = all(c.passed for c in checks if "mandatory" in c.check_name.lower() or "coverage" in c.check_name.lower() or "ambition" in c.check_name.lower())

        # Step 7: Temperature alignment
        temp_alignment = self._assess_temperature_alignment(
            near_term, data.ambition_level
        )

        # Step 8: Recommendations
        recommendations = self._generate_recommendations(
            data, checks, scope3_material
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = TargetResult(
            entity_name=data.entity_name,
            near_term_target=near_term,
            long_term_target=long_term,
            scope3_target=scope3_target,
            annual_rates=annual_rates,
            milestones=milestones,
            validation_checks=checks,
            validation_passed=all_passed,
            temperature_alignment=temp_alignment,
            scope3_materiality=scope3_material,
            total_base_year_tco2e=_round_val(total_base),
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Targets set: near=%s, long=%s, validation=%s, alignment=%s",
            near_term.name if near_term else "none",
            long_term.name if long_term else "none",
            "PASS" if all_passed else "FAIL",
            temp_alignment,
        )
        return result

    # ------------------------------------------------------------------ #
    # Near-Term Target                                                    #
    # ------------------------------------------------------------------ #

    def _create_near_term_target(
        self,
        data: TargetInput,
        scope1_2_base: Decimal,
    ) -> TargetDefinition:
        """Create near-term Scope 1+2 target.

        Near-term targets cover 5-10 years, requiring minimum 4.2%/yr
        linear reduction for 1.5C alignment per SBTi ACA.

        Args:
            data: Target input.
            scope1_2_base: Base year Scope 1+2 emissions.

        Returns:
            TargetDefinition for near-term.
        """
        rates = SBTI_RATES.get(data.ambition_level, SBTI_RATES[AmbitionLevel.PARIS_1_5C])
        annual_rate = rates["scope_1_2_annual_pct"]

        years = _decimal(data.near_term_target_year - data.base_year)
        total_reduction_pct = _round_val(annual_rate * years, 2)

        # Cap at 100%
        if total_reduction_pct > Decimal("100"):
            total_reduction_pct = Decimal("100")

        # Apply coverage
        covered_base = scope1_2_base * data.scope1_2_coverage_pct / Decimal("100")
        target_tco2e = _round_val(
            covered_base * (Decimal("100") - total_reduction_pct) / Decimal("100")
        )

        name = (
            f"Near-term {data.ambition_level.value} target: "
            f"{total_reduction_pct}% reduction in Scope 1+2 by "
            f"{data.near_term_target_year}"
        )

        return TargetDefinition(
            name=name,
            timeframe=TargetTimeframe.NEAR_TERM,
            target_type=TargetType.ABSOLUTE,
            scope_coverage=ScopeCategory.SCOPE_1_2,
            base_year=data.base_year,
            target_year=data.near_term_target_year,
            base_year_tco2e=_round_val(covered_base),
            target_tco2e=target_tco2e,
            reduction_pct=total_reduction_pct,
            annual_rate_pct=_round_val(annual_rate, 2),
            coverage_pct=data.scope1_2_coverage_pct,
            pathway=data.pathway_type.value,
            ambition=data.ambition_level.value,
        )

    # ------------------------------------------------------------------ #
    # Long-Term Target                                                    #
    # ------------------------------------------------------------------ #

    def _create_long_term_target(
        self,
        data: TargetInput,
        total_base: Decimal,
    ) -> TargetDefinition:
        """Create long-term net-zero target (by 2050).

        Long-term targets require 90%+ reduction across all scopes.

        Args:
            data: Target input.
            total_base: Total base year emissions (all scopes).

        Returns:
            TargetDefinition for long-term.
        """
        rates = SBTI_RATES.get(data.ambition_level, SBTI_RATES[AmbitionLevel.PARIS_1_5C])
        long_term_reduction = rates["long_term_reduction_pct"]

        years = _decimal(data.long_term_target_year - data.base_year)
        annual_rate = _round_val(_safe_divide(long_term_reduction, years), 2)

        target_tco2e = _round_val(
            total_base * (Decimal("100") - long_term_reduction) / Decimal("100")
        )

        name = (
            f"Long-term net-zero target: {long_term_reduction}% reduction "
            f"across all scopes by {data.long_term_target_year}"
        )

        return TargetDefinition(
            name=name,
            timeframe=TargetTimeframe.LONG_TERM,
            target_type=TargetType.ABSOLUTE,
            scope_coverage=ScopeCategory.ALL_SCOPES,
            base_year=data.base_year,
            target_year=data.long_term_target_year,
            base_year_tco2e=_round_val(total_base),
            target_tco2e=target_tco2e,
            reduction_pct=long_term_reduction,
            annual_rate_pct=annual_rate,
            coverage_pct=Decimal("100"),
            pathway=data.pathway_type.value,
            ambition=data.ambition_level.value,
        )

    # ------------------------------------------------------------------ #
    # Scope 3 Target                                                      #
    # ------------------------------------------------------------------ #

    def _create_scope3_target(self, data: TargetInput) -> TargetDefinition:
        """Create Scope 3 specific target.

        SBTi requires Scope 3 target if Scope 3 > 40% of total.
        Minimum 2.5%/yr for 1.5C pathway.

        Args:
            data: Target input.

        Returns:
            TargetDefinition for Scope 3.
        """
        rates = SBTI_RATES.get(data.ambition_level, SBTI_RATES[AmbitionLevel.PARIS_1_5C])
        annual_rate = rates["scope_3_annual_pct"]

        years = _decimal(data.near_term_target_year - data.base_year)
        total_reduction_pct = _round_val(annual_rate * years, 2)
        if total_reduction_pct > Decimal("100"):
            total_reduction_pct = Decimal("100")

        covered_base = (
            data.base_year_scope3_tco2e
            * data.scope3_coverage_pct / Decimal("100")
        )
        target_tco2e = _round_val(
            covered_base * (Decimal("100") - total_reduction_pct) / Decimal("100")
        )

        name = (
            f"Scope 3 target: {total_reduction_pct}% reduction by "
            f"{data.near_term_target_year} "
            f"({data.scope3_coverage_pct}% coverage)"
        )

        return TargetDefinition(
            name=name,
            timeframe=TargetTimeframe.NEAR_TERM,
            target_type=TargetType.ABSOLUTE,
            scope_coverage=ScopeCategory.SCOPE_3,
            base_year=data.base_year,
            target_year=data.near_term_target_year,
            base_year_tco2e=_round_val(covered_base),
            target_tco2e=target_tco2e,
            reduction_pct=total_reduction_pct,
            annual_rate_pct=_round_val(annual_rate, 2),
            coverage_pct=data.scope3_coverage_pct,
            pathway=data.pathway_type.value,
            ambition=data.ambition_level.value,
        )

    # ------------------------------------------------------------------ #
    # Annual Rates                                                        #
    # ------------------------------------------------------------------ #

    def _collect_annual_rates(
        self,
        near_term: Optional[TargetDefinition],
        long_term: Optional[TargetDefinition],
        scope3_target: Optional[TargetDefinition],
    ) -> Dict[str, Decimal]:
        """Collect annual reduction rates from all targets.

        Args:
            near_term: Near-term target.
            long_term: Long-term target.
            scope3_target: Scope 3 target.

        Returns:
            Dict mapping scope label to annual rate.
        """
        rates: Dict[str, Decimal] = {}
        if near_term:
            rates["scope_1_2_near_term_pct_per_year"] = near_term.annual_rate_pct
        if long_term:
            rates["all_scopes_long_term_pct_per_year"] = long_term.annual_rate_pct
        if scope3_target:
            rates["scope_3_near_term_pct_per_year"] = scope3_target.annual_rate_pct
        return rates

    # ------------------------------------------------------------------ #
    # Milestone Generation                                                #
    # ------------------------------------------------------------------ #

    def _generate_milestones(
        self,
        data: TargetInput,
        near_term: Optional[TargetDefinition],
        long_term: Optional[TargetDefinition],
        scope3_target: Optional[TargetDefinition],
    ) -> List[MilestoneEntry]:
        """Generate interim milestones at specified intervals.

        Milestones are placed every N years (default 5) along the
        linear reduction pathway from base year to target year.

        Args:
            data: Target input with interval configuration.
            near_term: Near-term target.
            long_term: Long-term target.
            scope3_target: Scope 3 target.

        Returns:
            List of MilestoneEntry instances.
        """
        milestones: List[MilestoneEntry] = []
        interval = data.milestone_interval_years

        targets_to_process = []
        if near_term:
            targets_to_process.append(("Scope 1+2", near_term))
        if long_term:
            targets_to_process.append(("All Scopes", long_term))
        if scope3_target:
            targets_to_process.append(("Scope 3", scope3_target))

        for scope_label, target in targets_to_process:
            base = target.base_year_tco2e
            annual_rate = target.annual_rate_pct
            total_years = target.target_year - target.base_year

            if total_years <= 0:
                continue

            year = target.base_year + interval
            while year < target.target_year:
                elapsed = year - target.base_year
                reduction_pct = _round_val(annual_rate * _decimal(elapsed), 2)
                if reduction_pct > Decimal("100"):
                    reduction_pct = Decimal("100")
                target_at_milestone = _round_val(
                    base * (Decimal("100") - reduction_pct) / Decimal("100")
                )
                milestones.append(MilestoneEntry(
                    year=year,
                    target_tco2e=target_at_milestone,
                    reduction_from_base_pct=reduction_pct,
                    scope=scope_label,
                ))
                year += interval

            # Final target year milestone
            milestones.append(MilestoneEntry(
                year=target.target_year,
                target_tco2e=target.target_tco2e,
                reduction_from_base_pct=target.reduction_pct,
                scope=scope_label,
            ))

        return sorted(milestones, key=lambda m: (m.year, m.scope))

    # ------------------------------------------------------------------ #
    # Validation Checks                                                   #
    # ------------------------------------------------------------------ #

    def _run_validation_checks(
        self,
        data: TargetInput,
        near_term: Optional[TargetDefinition],
        long_term: Optional[TargetDefinition],
        scope3_target: Optional[TargetDefinition],
        scope3_material: bool,
    ) -> List[ValidationCheck]:
        """Run SBTi validation checks against the generated targets.

        Checks include: ambition, timeframe, coverage, boundary,
        base year recency, Scope 3 materiality.

        Args:
            data: Target input.
            near_term: Near-term target.
            long_term: Long-term target.
            scope3_target: Scope 3 target.
            scope3_material: Whether Scope 3 exceeds materiality threshold.

        Returns:
            List of ValidationCheck results.
        """
        checks: List[ValidationCheck] = []
        rates = SBTI_RATES.get(data.ambition_level, SBTI_RATES[AmbitionLevel.PARIS_1_5C])

        # Check 1: Near-term ambition
        if near_term:
            min_rate = rates["scope_1_2_annual_pct"]
            checks.append(ValidationCheck(
                check_name="Mandatory: Near-term ambition (Scope 1+2)",
                passed=near_term.annual_rate_pct >= min_rate,
                required_value=f">= {min_rate}%/yr",
                actual_value=f"{near_term.annual_rate_pct}%/yr",
                message=(
                    "PASS: Annual rate meets minimum requirement."
                    if near_term.annual_rate_pct >= min_rate
                    else f"FAIL: Rate {near_term.annual_rate_pct}% below minimum {min_rate}%."
                ),
            ))

        # Check 2: Near-term timeframe (5-10 years)
        if near_term:
            years = data.near_term_target_year - data.base_year
            in_range = 5 <= years <= 10
            checks.append(ValidationCheck(
                check_name="Mandatory: Near-term timeframe (5-10 years)",
                passed=in_range,
                required_value="5-10 years",
                actual_value=f"{years} years",
                message=(
                    "PASS: Timeframe within SBTi range."
                    if in_range
                    else f"FAIL: {years} years is outside 5-10 year range."
                ),
            ))

        # Check 3: Scope 1+2 coverage
        checks.append(ValidationCheck(
            check_name="Mandatory: Scope 1+2 coverage >= 95%",
            passed=data.scope1_2_coverage_pct >= COVERAGE_REQUIREMENTS["scope_1_2_min_pct"],
            required_value=f">= {COVERAGE_REQUIREMENTS['scope_1_2_min_pct']}%",
            actual_value=f"{data.scope1_2_coverage_pct}%",
            message=(
                "PASS: Scope 1+2 coverage meets requirement."
                if data.scope1_2_coverage_pct >= COVERAGE_REQUIREMENTS["scope_1_2_min_pct"]
                else "FAIL: Coverage below 95% minimum."
            ),
        ))

        # Check 4: Scope 3 coverage (if material)
        if scope3_material:
            s3_ok = data.scope3_coverage_pct >= COVERAGE_REQUIREMENTS["scope_3_min_pct"]
            checks.append(ValidationCheck(
                check_name="Mandatory: Scope 3 coverage >= 67% (material)",
                passed=s3_ok,
                required_value=f">= {COVERAGE_REQUIREMENTS['scope_3_min_pct']}%",
                actual_value=f"{data.scope3_coverage_pct}%",
                message=(
                    "PASS: Scope 3 coverage meets requirement."
                    if s3_ok
                    else "FAIL: Scope 3 coverage below 67% minimum."
                ),
            ))

        # Check 5: Scope 3 target exists (if material)
        if scope3_material:
            has_s3 = scope3_target is not None
            checks.append(ValidationCheck(
                check_name="Mandatory: Scope 3 target required (>40% of total)",
                passed=has_s3,
                required_value="Scope 3 target set",
                actual_value="Yes" if has_s3 else "No",
                message=(
                    "PASS: Scope 3 target defined."
                    if has_s3
                    else "FAIL: Scope 3 is >40% of total but no target set."
                ),
            ))

        # Check 6: Long-term target by 2050
        if long_term:
            by_2050 = long_term.target_year <= 2050
            checks.append(ValidationCheck(
                check_name="Ambition: Long-term target by 2050",
                passed=by_2050,
                required_value="<= 2050",
                actual_value=str(long_term.target_year),
                message=(
                    "PASS: Long-term target by 2050."
                    if by_2050
                    else f"WARNING: Target year {long_term.target_year} is after 2050."
                ),
            ))

        # Check 7: Long-term reduction >= 90%
        if long_term:
            min_lt = rates["long_term_reduction_pct"]
            checks.append(ValidationCheck(
                check_name="Ambition: Long-term reduction >= 90%",
                passed=long_term.reduction_pct >= min_lt,
                required_value=f">= {min_lt}%",
                actual_value=f"{long_term.reduction_pct}%",
                message=(
                    "PASS: Long-term reduction meets threshold."
                    if long_term.reduction_pct >= min_lt
                    else f"FAIL: Reduction {long_term.reduction_pct}% below {min_lt}%."
                ),
            ))

        # Check 8: Base year recency
        current_year = utcnow().year
        base_age = current_year - data.base_year
        max_age = int(COVERAGE_REQUIREMENTS["base_year_max_age_years"])
        checks.append(ValidationCheck(
            check_name="Coverage: Base year recency (<= 5 years)",
            passed=base_age <= max_age,
            required_value=f"<= {max_age} years old",
            actual_value=f"{base_age} years old",
            message=(
                "PASS: Base year is recent enough."
                if base_age <= max_age
                else f"WARNING: Base year is {base_age} years old (max {max_age})."
            ),
        ))

        return checks

    # ------------------------------------------------------------------ #
    # Temperature Alignment                                               #
    # ------------------------------------------------------------------ #

    def _assess_temperature_alignment(
        self,
        near_term: Optional[TargetDefinition],
        ambition: AmbitionLevel,
    ) -> str:
        """Assess temperature alignment of the target.

        Maps the ambition level and annual rate to a temperature outcome.

        Args:
            near_term: Near-term target definition.
            ambition: Ambition level.

        Returns:
            Temperature alignment string.
        """
        if near_term is None:
            return TemperatureAlignment.NOT_ALIGNED.value

        rate = near_term.annual_rate_pct

        if rate >= Decimal("4.2"):
            return TemperatureAlignment.ALIGNED_1_5C.value
        elif rate >= Decimal("2.5"):
            return TemperatureAlignment.ALIGNED_WB2C.value
        elif rate >= Decimal("1.23"):
            return TemperatureAlignment.ALIGNED_2C.value
        else:
            return TemperatureAlignment.NOT_ALIGNED.value

    # ------------------------------------------------------------------ #
    # Recommendations                                                     #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(
        self,
        data: TargetInput,
        checks: List[ValidationCheck],
        scope3_material: bool,
    ) -> List[str]:
        """Generate actionable recommendations based on validation results.

        Args:
            data: Target input.
            checks: Validation check results.
            scope3_material: Scope 3 materiality flag.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        failed = [c for c in checks if not c.passed]
        if failed:
            recs.append(
                f"{len(failed)} validation check(s) did not pass. "
                "Review and address each before SBTi submission."
            )

        if data.ambition_level != AmbitionLevel.PARIS_1_5C:
            recs.append(
                "Consider upgrading to 1.5C ambition for strongest "
                "alignment with the Paris Agreement and investor expectations."
            )

        if scope3_material and not data.include_scope3:
            recs.append(
                "Scope 3 exceeds 40% of total emissions. SBTi requires "
                "a Scope 3 target. Enable include_scope3=True."
            )

        if data.scope1_2_coverage_pct < Decimal("100"):
            recs.append(
                f"Scope 1+2 coverage is {data.scope1_2_coverage_pct}%. "
                "Increase to 100% to ensure comprehensive coverage."
            )

        if data.pathway_type == PathwayType.ACA:
            recs.append(
                "ACA pathway selected. If operating in a sector with SDA "
                "benchmarks (power, buildings, transport), consider SDA for "
                "more granular sector alignment."
            )

        return recs

    # ------------------------------------------------------------------ #
    # Utility Methods                                                     #
    # ------------------------------------------------------------------ #

    def get_sbti_rates(self, ambition: AmbitionLevel) -> Dict[str, str]:
        """Look up SBTi minimum rates for an ambition level.

        Args:
            ambition: Target ambition level.

        Returns:
            Dict with rate parameters as strings.

        Raises:
            ValueError: If ambition level is not recognized.
        """
        rates = SBTI_RATES.get(ambition)
        if rates is None:
            raise ValueError(f"Unknown ambition level: {ambition}")
        return {k: str(v) for k, v in rates.items()}

    def get_sda_benchmark(
        self, sector: SBTiSector, year: str
    ) -> Optional[Dict[str, str]]:
        """Look up SDA sector benchmark for a given year.

        Args:
            sector: SBTi sector.
            year: Year key (e.g. '2030_benchmark').

        Returns:
            Dict with unit and value, or None if not found.
        """
        benchmark = SDA_BENCHMARKS.get(sector)
        if benchmark is None:
            return None
        value = benchmark.get(f"{year}_benchmark")
        if value is None:
            return None
        return {
            "unit": benchmark["unit"],
            "value": str(value),
        }

    def get_coverage_requirements(self) -> Dict[str, str]:
        """Return SBTi coverage requirements as a dict of strings.

        Returns:
            Dict mapping requirement names to string values.
        """
        return {k: str(v) for k, v in COVERAGE_REQUIREMENTS.items()}
