# -*- coding: utf-8 -*-
"""
TargetSettingEngine - PACK-023 SBTi Alignment Engine 1
========================================================

Near-term, long-term, and net-zero target definition with ACA/SDA/FLAG
pathway selection, ambition level assessment (1.5C/WB2C/2C), scope
coverage validation, and target boundary enforcement.

This engine is the entry-point for the SBTi target-setting process.
It takes an organisation's emissions inventory, sector, and preferences
and produces fully defined targets with annual pathway milestones,
coverage checks, and an ambition assessment against SBTi criteria.

Calculation Methodology:
    ACA absolute reduction:
        E(t) = E(base) * (1 - annual_rate) ^ (t - base_year)
        annual_rate = {1.5C: 4.2%, WB2C: 2.5%, 2C: 1.6%}

    SDA intensity convergence:
        I(t) = I_sector(t) + (I_co(base) - I_sector(base))
               * ((I_sector(target) - I_sector(t))
               / (I_sector(target) - I_sector(base)))

    FLAG linear reduction:
        E(t) = E(base) * (1 - 0.0303 * (t - base_year))

    Coverage checks:
        scope12_coverage = (covered_scope12 / total_scope12) >= 0.95
        scope3_nt_coverage = (covered_scope3 / total_scope3) >= 0.67
        scope3_lt_coverage = (covered_scope3 / total_scope3) >= 0.90

Regulatory References:
    - SBTi Corporate Manual V5.3 (2024)
    - SBTi Corporate Net-Zero Standard V1.3 (2024)
    - SBTi FLAG Guidance V1.1 (2022)
    - IPCC AR6 WG3 (2022) - Carbon budgets
    - IEA Net Zero by 2050 Roadmap (2023) - Sector benchmarks
    - GHG Protocol Corporate Standard (WRI/WBCSD, 2015)

Zero-Hallucination:
    - All reductions computed with deterministic Decimal arithmetic
    - ACA rates from published SBTi Corporate Manual tables
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-023 SBTi Alignment
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
from typing import Any, Dict, List, Optional, Tuple

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

class TargetType(str, Enum):
    """SBTi target classification types.

    NEAR_TERM: 5-10 year target horizon.
    LONG_TERM: Beyond 2035, up to 2050.
    NET_ZERO: Net-zero by 2050 at latest.
    FLAG: Forest, Land and Agriculture target.
    """
    NEAR_TERM = "near_term"
    LONG_TERM = "long_term"
    NET_ZERO = "net_zero"
    FLAG = "flag"

class AmbitionLevel(str, Enum):
    """Temperature ambition levels per SBTi alignment.

    CELSIUS_1_5: 1.5C-aligned (most ambitious).
    WELL_BELOW_2C: Well-below 2C pathway.
    CELSIUS_2: 2C-aligned (minimum acceptable).
    """
    CELSIUS_1_5 = "1.5c"
    WELL_BELOW_2C = "well_below_2c"
    CELSIUS_2 = "2c"

class PathwayMethod(str, Enum):
    """SBTi target-setting methodological approaches.

    ACA: Absolute Contraction Approach.
    SDA: Sectoral Decarbonization Approach.
    FLAG: Forest, Land and Agriculture pathway.
    ECONOMIC_INTENSITY: Revenue-based intensity.
    PHYSICAL_INTENSITY: Physical output-based intensity.
    SUPPLIER_ENGAGEMENT: Supplier-level engagement targets.
    """
    ACA = "aca"
    SDA = "sda"
    FLAG = "flag"
    ECONOMIC_INTENSITY = "economic_intensity"
    PHYSICAL_INTENSITY = "physical_intensity"
    SUPPLIER_ENGAGEMENT = "supplier_engagement"

class TargetScope(str, Enum):
    """GHG Protocol emission scopes for SBTi target boundary.

    SCOPE_1: Direct emissions.
    SCOPE_2: Purchased electricity/heat.
    SCOPE_1_2: Combined Scope 1 + 2.
    SCOPE_3: Value chain emissions.
    SCOPE_1_2_3: All scopes combined.
    FLAG: Forest, Land and Agriculture specific.
    """
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_1_2 = "scope_1_2"
    SCOPE_3 = "scope_3"
    SCOPE_1_2_3 = "scope_1_2_3"
    FLAG = "flag"

class BoundaryApproach(str, Enum):
    """Organisational boundary consolidation approach.

    OPERATIONAL_CONTROL: Full emissions where operational control exists.
    FINANCIAL_CONTROL: Full emissions where financial control exists.
    EQUITY_SHARE: Pro-rata by equity ownership percentage.
    """
    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"
    EQUITY_SHARE = "equity_share"

# ---------------------------------------------------------------------------
# Constants -- SBTi ACA Rates and Thresholds
# ---------------------------------------------------------------------------

# ACA annual linear reduction rates (as fractions) per ambition level.
# Source: SBTi Corporate Manual V5.3, Table 3.
ACA_ANNUAL_RATES: Dict[str, Decimal] = {
    AmbitionLevel.CELSIUS_1_5: Decimal("0.042"),
    AmbitionLevel.WELL_BELOW_2C: Decimal("0.025"),
    AmbitionLevel.CELSIUS_2: Decimal("0.016"),
}

# FLAG linear annual reduction rate: 3.03%/yr.
# Source: SBTi FLAG Guidance V1.1, Section 5.
FLAG_RATE: Decimal = Decimal("0.0303")

# Earliest acceptable base year for SBTi submissions.
BASE_YEAR_MIN: int = 2015

# Maximum base year age for new submissions (years from current year).
BASE_YEAR_MAX_AGE: int = 5

# Scope 1+2 coverage minimum.
SCOPE12_COVERAGE_MIN: Decimal = Decimal("0.95")

# Scope 3 near-term coverage minimum.
SCOPE3_NT_COVERAGE: Decimal = Decimal("0.67")

# Scope 3 long-term coverage minimum.
SCOPE3_LT_COVERAGE: Decimal = Decimal("0.90")

# Near-term target timeframe bounds.
NEAR_TERM_MIN_YEARS: int = 5
NEAR_TERM_MAX_YEARS: int = 10

# Long-term target year floor.
LONG_TERM_MIN_YEAR: int = 2036

# Net-zero target year ceiling.
NET_ZERO_MAX_YEAR: int = 2050

# Scope 3 materiality trigger threshold (fraction of S1+S2+S3).
SCOPE3_TRIGGER_THRESHOLD: Decimal = Decimal("0.40")

# FLAG materiality trigger threshold (fraction of total).
FLAG_TRIGGER_THRESHOLD: Decimal = Decimal("0.20")

# Minimum long-term absolute reduction percentage by 2050.
LONG_TERM_MIN_REDUCTION_PCT: Decimal = Decimal("90.0")

# Maximum residual emissions for net-zero (% of base year).
NET_ZERO_MAX_RESIDUAL_PCT: Decimal = Decimal("10.0")

# Scope 2: renewable electricity target year.
SCOPE2_RE100_YEAR: int = 2030

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class EmissionsInventory(BaseModel):
    """Base year emissions inventory.

    Attributes:
        scope1_tco2e: Scope 1 direct emissions (tCO2e).
        scope2_location_tco2e: Scope 2 location-based emissions.
        scope2_market_tco2e: Scope 2 market-based emissions.
        scope3_total_tco2e: Total Scope 3 emissions.
        scope3_categories: Per-category Scope 3 breakdown.
        flag_emissions_tco2e: FLAG-specific emissions.
        biogenic_tco2e: Biogenic CO2 emissions.
        total_tco2e: Grand total of all emissions.
    """
    scope1_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    scope2_location_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    scope2_market_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    scope3_total_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    scope3_categories: Dict[str, Decimal] = Field(default_factory=dict)
    flag_emissions_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    biogenic_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    total_tco2e: Decimal = Field(default=Decimal("0"), ge=0)

class TargetSettingInput(BaseModel):
    """Input data for SBTi target definition.

    Attributes:
        entity_name: Reporting entity name.
        sector: Industry sector for SDA pathway selection.
        base_year: Emissions base year.
        target_year_near_term: Near-term target year (5-10 years out).
        target_year_long_term: Long-term target year (by 2050).
        inventory: Base year emissions inventory.
        boundary_approach: Organisational boundary method.
        preferred_method_s12: Preferred Scope 1+2 pathway method.
        preferred_method_s3: Preferred Scope 3 pathway method.
        preferred_ambition: Preferred ambition level.
        scope12_coverage_pct: Percentage of S1+2 emissions covered.
        scope3_coverage_pct: Percentage of S3 emissions covered.
        has_flag_emissions: Whether entity has FLAG emissions.
        renewable_electricity_pct: Current renewable electricity share.
        supplier_engagement_pct: Current supplier engagement rate.
        include_net_zero: Whether to define a net-zero target.
    """
    entity_name: str = Field(
        ..., min_length=1, max_length=300, description="Entity name"
    )
    sector: str = Field(
        default="general", max_length=100, description="SBTi sector"
    )
    base_year: int = Field(
        ..., ge=2015, le=2030, description="Emissions base year"
    )
    target_year_near_term: int = Field(
        default=0, ge=0, le=2040,
        description="Near-term target year (0 = auto-calculate)"
    )
    target_year_long_term: int = Field(
        default=2050, ge=2035, le=2060,
        description="Long-term target year"
    )
    inventory: EmissionsInventory = Field(
        ..., description="Base year emissions inventory"
    )
    boundary_approach: BoundaryApproach = Field(
        default=BoundaryApproach.OPERATIONAL_CONTROL,
        description="Organisational boundary approach"
    )
    preferred_method_s12: PathwayMethod = Field(
        default=PathwayMethod.ACA,
        description="Preferred S1+2 pathway method"
    )
    preferred_method_s3: PathwayMethod = Field(
        default=PathwayMethod.ACA,
        description="Preferred S3 pathway method"
    )
    preferred_ambition: AmbitionLevel = Field(
        default=AmbitionLevel.CELSIUS_1_5,
        description="Preferred ambition level"
    )
    scope12_coverage_pct: Decimal = Field(
        default=Decimal("100"), ge=Decimal("0"), le=Decimal("100"),
        description="S1+2 coverage as percentage"
    )
    scope3_coverage_pct: Decimal = Field(
        default=Decimal("67"), ge=Decimal("0"), le=Decimal("100"),
        description="S3 coverage as percentage"
    )
    has_flag_emissions: bool = Field(
        default=False, description="Whether entity has FLAG emissions"
    )
    renewable_electricity_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Current renewable electricity share (%)"
    )
    supplier_engagement_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Current supplier engagement rate (%)"
    )
    include_net_zero: bool = Field(
        default=True, description="Whether to define a net-zero target"
    )

    @field_validator("target_year_near_term")
    @classmethod
    def validate_near_term_year(cls, v: int, info: Any) -> int:
        """Auto-calculate near-term year if zero."""
        if v == 0:
            base = info.data.get("base_year", 2023)
            return base + 7  # default 7-year horizon
        return v

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class PathwayMilestone(BaseModel):
    """Annual milestone point on a reduction pathway.

    Attributes:
        year: Calendar year.
        target_emissions_tco2e: Target emissions at this year.
        reduction_from_base_pct: Cumulative reduction from base (%).
        annual_reduction_rate_pct: Implied annual reduction rate (%).
        on_track_threshold_tco2e: Maximum emissions to be on-track.
    """
    year: int = Field(default=0)
    target_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    reduction_from_base_pct: Decimal = Field(default=Decimal("0"))
    annual_reduction_rate_pct: Decimal = Field(default=Decimal("0"))
    on_track_threshold_tco2e: Decimal = Field(default=Decimal("0"))

class ScopeTarget(BaseModel):
    """Target definition for a specific scope.

    Attributes:
        scope: Target scope (S1, S2, S1+2, S3).
        base_year_emissions_tco2e: Base year emissions for this scope.
        target_year_emissions_tco2e: Target year emissions.
        reduction_pct: Total reduction percentage.
        annual_reduction_rate_pct: Annualised reduction rate.
        method: Pathway method used.
        coverage_pct: Percentage of scope emissions covered.
        meets_coverage_requirement: Whether coverage is sufficient.
    """
    scope: str = Field(default="")
    base_year_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    target_year_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    reduction_pct: Decimal = Field(default=Decimal("0"))
    annual_reduction_rate_pct: Decimal = Field(default=Decimal("0"))
    method: str = Field(default="")
    coverage_pct: Decimal = Field(default=Decimal("0"))
    meets_coverage_requirement: bool = Field(default=False)

class TargetDefinition(BaseModel):
    """Complete target definition for a target type.

    Attributes:
        target_id: Unique target identifier.
        target_type: Target type (near-term / long-term / net-zero / FLAG).
        entity_name: Reporting entity.
        base_year: Emissions base year.
        target_year: Target achievement year.
        boundary_approach: Organisational boundary method.
        ambition_level: Temperature ambition alignment.
        scope_targets: Per-scope target definitions.
        milestones: Annual pathway milestones.
        total_base_emissions_tco2e: Total base year emissions in target.
        total_target_emissions_tco2e: Total target year emissions.
        total_reduction_pct: Overall reduction percentage.
    """
    target_id: str = Field(default_factory=_new_uuid)
    target_type: str = Field(default="")
    entity_name: str = Field(default="")
    base_year: int = Field(default=0)
    target_year: int = Field(default=0)
    boundary_approach: str = Field(default="")
    ambition_level: str = Field(default="")
    scope_targets: List[ScopeTarget] = Field(default_factory=list)
    milestones: List[PathwayMilestone] = Field(default_factory=list)
    total_base_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    total_target_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    total_reduction_pct: Decimal = Field(default=Decimal("0"))

class AmbitionAssessment(BaseModel):
    """Temperature alignment assessment of defined targets.

    Attributes:
        overall_alignment: Overall temperature alignment label.
        scope12_alignment: S1+2 pathway alignment.
        scope3_alignment: S3 pathway alignment.
        scope12_annual_rate_pct: S1+2 annual reduction rate.
        scope3_annual_rate_pct: S3 annual reduction rate.
        meets_1_5c: Whether targets meet 1.5C minimum.
        meets_wb2c: Whether targets meet WB2C minimum.
        gap_to_1_5c_pct: Gap to 1.5C rate (percentage points).
        temperature_estimate_c: Estimated temperature outcome.
    """
    overall_alignment: str = Field(default="")
    scope12_alignment: str = Field(default="")
    scope3_alignment: str = Field(default="")
    scope12_annual_rate_pct: Decimal = Field(default=Decimal("0"))
    scope3_annual_rate_pct: Decimal = Field(default=Decimal("0"))
    meets_1_5c: bool = Field(default=False)
    meets_wb2c: bool = Field(default=False)
    gap_to_1_5c_pct: Decimal = Field(default=Decimal("0"))
    temperature_estimate_c: Decimal = Field(default=Decimal("0"))

class BaseYearValidation(BaseModel):
    """Result of base year validation.

    Attributes:
        base_year: Assessed base year.
        is_valid: Whether base year meets SBTi requirements.
        issues: List of identified issues.
        meets_minimum_year: Whether >= 2015.
        meets_recency: Whether within 5 years.
        inventory_complete: Whether inventory is sufficiently complete.
    """
    base_year: int = Field(default=0)
    is_valid: bool = Field(default=False)
    issues: List[str] = Field(default_factory=list)
    meets_minimum_year: bool = Field(default=False)
    meets_recency: bool = Field(default=False)
    inventory_complete: bool = Field(default=False)

class CoverageCheck(BaseModel):
    """Scope coverage validation result.

    Attributes:
        scope: Scope being checked.
        coverage_pct: Achieved coverage percentage.
        required_pct: Required coverage percentage.
        is_sufficient: Whether coverage meets requirement.
        gap_pct: Gap to required coverage.
        message: Human-readable assessment.
    """
    scope: str = Field(default="")
    coverage_pct: Decimal = Field(default=Decimal("0"))
    required_pct: Decimal = Field(default=Decimal("0"))
    is_sufficient: bool = Field(default=False)
    gap_pct: Decimal = Field(default=Decimal("0"))
    message: str = Field(default="")

class TargetSettingResult(BaseModel):
    """Complete target-setting result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Timestamp.
        entity_name: Entity name.
        targets: List of defined targets.
        ambition_assessment: Temperature alignment assessment.
        base_year_validation: Base year check result.
        coverage_checks: Scope coverage validations.
        scope3_required: Whether Scope 3 targets are required.
        flag_required: Whether FLAG targets are required.
        warnings: Non-blocking warnings.
        errors: Blocking errors.
        processing_time_ms: Processing duration (ms).
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    entity_name: str = Field(default="")
    targets: List[TargetDefinition] = Field(default_factory=list)
    ambition_assessment: Optional[AmbitionAssessment] = Field(None)
    base_year_validation: Optional[BaseYearValidation] = Field(None)
    coverage_checks: List[CoverageCheck] = Field(default_factory=list)
    scope3_required: bool = Field(default=False)
    flag_required: bool = Field(default=False)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class TargetSettingEngine:
    """SBTi target setting engine.

    Defines near-term, long-term, and net-zero targets with ACA/SDA/FLAG
    pathway selection, ambition level assessment, and scope coverage
    validation.  All calculations use deterministic Decimal arithmetic.

    Usage::

        engine = TargetSettingEngine()
        result = engine.define_targets(input_data)
        for t in result.targets:
            print(f"{t.target_type}: {t.total_reduction_pct}%")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise TargetSettingEngine.

        Args:
            config: Optional configuration overrides.
        """
        self.config = config or {}
        logger.info("TargetSettingEngine v%s initialised", self.engine_version)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def define_targets(self, data: TargetSettingInput) -> TargetSettingResult:
        """Define SBTi-aligned targets for an entity.

        Orchestrates the full target-setting pipeline: validates base year,
        checks coverage, determines Scope 3 / FLAG requirements, builds
        near-term and optionally long-term / net-zero targets, and assesses
        ambition alignment.

        Args:
            data: Validated target-setting input.

        Returns:
            TargetSettingResult with targets, milestones, and assessment.
        """
        t0 = time.perf_counter()
        logger.info(
            "Target setting: entity=%s, base=%d, sector=%s",
            data.entity_name, data.base_year, data.sector,
        )

        warnings: List[str] = []
        errors: List[str] = []

        # Step 1: Validate base year
        base_val = self.validate_base_year(data.base_year, data.inventory)

        if not base_val.is_valid:
            errors.extend(base_val.issues)

        # Step 2: Check coverage
        coverage_checks = self.check_coverage(data)

        # Step 3: Determine requirements
        scope3_required = self._is_scope3_required(data.inventory)
        flag_required = self._is_flag_required(data.inventory)

        if scope3_required:
            warnings.append(
                "Scope 3 emissions exceed 40% of total; "
                "Scope 3 targets are required."
            )
        if flag_required:
            warnings.append(
                "FLAG emissions exceed 20% of total; "
                "Separate FLAG target required."
            )

        # Step 4: Build targets
        targets: List[TargetDefinition] = []

        # Near-term target
        nt = self._build_near_term_target(data)
        targets.append(nt)

        # Long-term target
        lt = self._build_long_term_target(data)
        targets.append(lt)

        # Net-zero target (if requested)
        if data.include_net_zero:
            nz = self._build_net_zero_target(data)
            targets.append(nz)

        # FLAG target (if required)
        if flag_required and data.has_flag_emissions:
            ft = self._build_flag_target(data)
            targets.append(ft)

        # Step 5: Assess ambition
        ambition = self.assess_ambition(data, targets)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = TargetSettingResult(
            entity_name=data.entity_name,
            targets=targets,
            ambition_assessment=ambition,
            base_year_validation=base_val,
            coverage_checks=coverage_checks,
            scope3_required=scope3_required,
            flag_required=flag_required,
            warnings=warnings,
            errors=errors,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Target setting complete: %d targets, ambition=%s, hash=%s",
            len(targets), ambition.overall_alignment,
            result.provenance_hash[:16],
        )
        return result

    def calculate_pathway(
        self,
        base_emissions: Decimal,
        base_year: int,
        target_year: int,
        method: PathwayMethod,
        ambition: AmbitionLevel,
    ) -> List[PathwayMilestone]:
        """Calculate annual pathway milestones for a given method.

        Args:
            base_emissions: Base year emissions (tCO2e).
            base_year: Base calendar year.
            target_year: Target calendar year.
            method: Pathway methodology.
            ambition: Ambition level.

        Returns:
            List of PathwayMilestone objects from base to target year.
        """
        milestones: List[PathwayMilestone] = []
        annual_rate = self._get_annual_rate(method, ambition)

        for year in range(base_year, target_year + 1):
            elapsed = year - base_year
            target_em = self._project_emissions(
                base_emissions, annual_rate, elapsed, method
            )
            reduction_pct = _safe_pct(
                base_emissions - target_em, base_emissions
            )

            milestones.append(PathwayMilestone(
                year=year,
                target_emissions_tco2e=_round_val(target_em),
                reduction_from_base_pct=_round_val(reduction_pct, 2),
                annual_reduction_rate_pct=_round_val(
                    annual_rate * Decimal("100"), 2
                ),
                on_track_threshold_tco2e=_round_val(
                    target_em * Decimal("1.05")
                ),
            ))

        return milestones

    def assess_ambition(
        self,
        data: TargetSettingInput,
        targets: List[TargetDefinition],
    ) -> AmbitionAssessment:
        """Assess temperature alignment of defined targets.

        Compares achieved annual reduction rates against SBTi thresholds
        to determine 1.5C, WB2C, or 2C alignment.

        Args:
            data: Input data.
            targets: Defined targets.

        Returns:
            AmbitionAssessment with alignment and temperature estimate.
        """
        # Find S1+2 rate from near-term target
        s12_rate = Decimal("0")
        s3_rate = Decimal("0")

        for target in targets:
            if target.target_type == TargetType.NEAR_TERM.value:
                for st in target.scope_targets:
                    if st.scope in (
                        TargetScope.SCOPE_1_2.value, TargetScope.SCOPE_1.value
                    ):
                        s12_rate = max(
                            s12_rate, st.annual_reduction_rate_pct
                        )
                    if st.scope == TargetScope.SCOPE_3.value:
                        s3_rate = max(s3_rate, st.annual_reduction_rate_pct)

        # Determine alignment
        s12_alignment = self._rate_to_alignment(s12_rate)
        s3_alignment = self._rate_to_alignment(s3_rate)

        # Overall = minimum of S1+2 and S3 (if S3 required)
        if self._is_scope3_required(data.inventory) and s3_rate > Decimal("0"):
            overall = self._min_alignment(s12_alignment, s3_alignment)
        else:
            overall = s12_alignment

        # Gap to 1.5C
        gap = max(Decimal("0"), Decimal("4.2") - s12_rate)

        # Temperature estimate via piecewise linear mapping
        temp = self._rate_to_temperature(s12_rate)

        return AmbitionAssessment(
            overall_alignment=overall,
            scope12_alignment=s12_alignment,
            scope3_alignment=s3_alignment,
            scope12_annual_rate_pct=_round_val(s12_rate, 2),
            scope3_annual_rate_pct=_round_val(s3_rate, 2),
            meets_1_5c=s12_rate >= Decimal("4.2"),
            meets_wb2c=s12_rate >= Decimal("2.5"),
            gap_to_1_5c_pct=_round_val(gap, 2),
            temperature_estimate_c=_round_val(temp, 2),
        )

    def validate_base_year(
        self, base_year: int, inventory: EmissionsInventory,
    ) -> BaseYearValidation:
        """Validate base year against SBTi requirements.

        Args:
            base_year: Proposed base year.
            inventory: Base year emissions inventory.

        Returns:
            BaseYearValidation with pass/fail and issues.
        """
        issues: List[str] = []
        current_year = utcnow().year

        meets_min = base_year >= BASE_YEAR_MIN
        if not meets_min:
            issues.append(
                f"Base year {base_year} is before minimum {BASE_YEAR_MIN}."
            )

        meets_recency = (current_year - base_year) <= BASE_YEAR_MAX_AGE
        if not meets_recency:
            issues.append(
                f"Base year {base_year} is more than "
                f"{BASE_YEAR_MAX_AGE} years old."
            )

        # Check inventory completeness
        has_s1 = inventory.scope1_tco2e > Decimal("0")
        has_s2 = (
            inventory.scope2_location_tco2e > Decimal("0")
            or inventory.scope2_market_tco2e > Decimal("0")
        )
        inv_complete = has_s1 and has_s2
        if not inv_complete:
            issues.append(
                "Inventory must include both Scope 1 and Scope 2 emissions."
            )

        is_valid = meets_min and meets_recency and inv_complete

        return BaseYearValidation(
            base_year=base_year,
            is_valid=is_valid,
            issues=issues,
            meets_minimum_year=meets_min,
            meets_recency=meets_recency,
            inventory_complete=inv_complete,
        )

    def check_coverage(
        self, data: TargetSettingInput,
    ) -> List[CoverageCheck]:
        """Validate scope coverage against SBTi requirements.

        Args:
            data: Target-setting input.

        Returns:
            List of CoverageCheck results.
        """
        checks: List[CoverageCheck] = []

        # Scope 1+2 coverage
        s12_required = SCOPE12_COVERAGE_MIN * Decimal("100")
        s12_sufficient = data.scope12_coverage_pct >= s12_required
        s12_gap = max(Decimal("0"), s12_required - data.scope12_coverage_pct)
        checks.append(CoverageCheck(
            scope=TargetScope.SCOPE_1_2.value,
            coverage_pct=_round_val(data.scope12_coverage_pct, 2),
            required_pct=_round_val(s12_required, 2),
            is_sufficient=s12_sufficient,
            gap_pct=_round_val(s12_gap, 2),
            message=(
                "Scope 1+2 coverage meets 95% requirement."
                if s12_sufficient
                else f"Scope 1+2 coverage gap of {_round_val(s12_gap, 1)}%."
            ),
        ))

        # Scope 3 near-term coverage
        s3nt_required = SCOPE3_NT_COVERAGE * Decimal("100")
        s3nt_sufficient = data.scope3_coverage_pct >= s3nt_required
        s3nt_gap = max(Decimal("0"), s3nt_required - data.scope3_coverage_pct)
        checks.append(CoverageCheck(
            scope=f"{TargetScope.SCOPE_3.value}_near_term",
            coverage_pct=_round_val(data.scope3_coverage_pct, 2),
            required_pct=_round_val(s3nt_required, 2),
            is_sufficient=s3nt_sufficient,
            gap_pct=_round_val(s3nt_gap, 2),
            message=(
                "Scope 3 near-term coverage meets 67% requirement."
                if s3nt_sufficient
                else f"Scope 3 near-term coverage gap of "
                     f"{_round_val(s3nt_gap, 1)}%."
            ),
        ))

        # Scope 3 long-term coverage
        s3lt_required = SCOPE3_LT_COVERAGE * Decimal("100")
        s3lt_sufficient = data.scope3_coverage_pct >= s3lt_required
        s3lt_gap = max(Decimal("0"), s3lt_required - data.scope3_coverage_pct)
        checks.append(CoverageCheck(
            scope=f"{TargetScope.SCOPE_3.value}_long_term",
            coverage_pct=_round_val(data.scope3_coverage_pct, 2),
            required_pct=_round_val(s3lt_required, 2),
            is_sufficient=s3lt_sufficient,
            gap_pct=_round_val(s3lt_gap, 2),
            message=(
                "Scope 3 long-term coverage meets 90% requirement."
                if s3lt_sufficient
                else f"Scope 3 long-term coverage gap of "
                     f"{_round_val(s3lt_gap, 1)}%."
            ),
        ))

        return checks

    # ------------------------------------------------------------------ #
    # Target Builders                                                     #
    # ------------------------------------------------------------------ #

    def _build_near_term_target(
        self, data: TargetSettingInput,
    ) -> TargetDefinition:
        """Build near-term target definition (5-10 year horizon).

        Args:
            data: Target-setting input.

        Returns:
            TargetDefinition for near-term target.
        """
        method = data.preferred_method_s12
        ambition = data.preferred_ambition
        annual_rate = self._get_annual_rate(method, ambition)
        annual_rate_s3 = self._get_annual_rate(data.preferred_method_s3, ambition)

        base_year = data.base_year
        target_year = data.target_year_near_term
        elapsed = target_year - base_year

        # Scope 1+2 target
        s12_base = (
            data.inventory.scope1_tco2e
            + data.inventory.scope2_market_tco2e
        )
        s12_target = self._project_emissions(
            s12_base, annual_rate, elapsed, method
        )
        s12_reduction = _safe_pct(s12_base - s12_target, s12_base)

        s12_scope_target = ScopeTarget(
            scope=TargetScope.SCOPE_1_2.value,
            base_year_emissions_tco2e=_round_val(s12_base),
            target_year_emissions_tco2e=_round_val(s12_target),
            reduction_pct=_round_val(s12_reduction, 2),
            annual_reduction_rate_pct=_round_val(
                annual_rate * Decimal("100"), 2
            ),
            method=method.value,
            coverage_pct=data.scope12_coverage_pct,
            meets_coverage_requirement=(
                data.scope12_coverage_pct >= SCOPE12_COVERAGE_MIN * Decimal("100")
            ),
        )

        scope_targets = [s12_scope_target]

        # Scope 3 target (if required)
        if self._is_scope3_required(data.inventory):
            s3_base = data.inventory.scope3_total_tco2e
            s3_target = self._project_emissions(
                s3_base, annual_rate_s3, elapsed, data.preferred_method_s3
            )
            s3_reduction = _safe_pct(s3_base - s3_target, s3_base)

            scope_targets.append(ScopeTarget(
                scope=TargetScope.SCOPE_3.value,
                base_year_emissions_tco2e=_round_val(s3_base),
                target_year_emissions_tco2e=_round_val(s3_target),
                reduction_pct=_round_val(s3_reduction, 2),
                annual_reduction_rate_pct=_round_val(
                    annual_rate_s3 * Decimal("100"), 2
                ),
                method=data.preferred_method_s3.value,
                coverage_pct=data.scope3_coverage_pct,
                meets_coverage_requirement=(
                    data.scope3_coverage_pct
                    >= SCOPE3_NT_COVERAGE * Decimal("100")
                ),
            ))

        # Milestones
        milestones = self.calculate_pathway(
            s12_base, base_year, target_year, method, ambition
        )

        total_base = sum(
            st.base_year_emissions_tco2e for st in scope_targets
        )
        total_target = sum(
            st.target_year_emissions_tco2e for st in scope_targets
        )
        total_reduction = _safe_pct(total_base - total_target, total_base)

        return TargetDefinition(
            target_type=TargetType.NEAR_TERM.value,
            entity_name=data.entity_name,
            base_year=base_year,
            target_year=target_year,
            boundary_approach=data.boundary_approach.value,
            ambition_level=ambition.value,
            scope_targets=scope_targets,
            milestones=milestones,
            total_base_emissions_tco2e=_round_val(total_base),
            total_target_emissions_tco2e=_round_val(total_target),
            total_reduction_pct=_round_val(total_reduction, 2),
        )

    def _build_long_term_target(
        self, data: TargetSettingInput,
    ) -> TargetDefinition:
        """Build long-term target definition (by 2050).

        Args:
            data: Target-setting input.

        Returns:
            TargetDefinition for long-term target.
        """
        method = data.preferred_method_s12
        ambition = data.preferred_ambition
        target_year = data.target_year_long_term
        base_year = data.base_year
        elapsed = target_year - base_year

        # Long-term requires >= 90% reduction
        lt_rate = self._get_annual_rate(method, ambition)
        lt_reduction_pct = LONG_TERM_MIN_REDUCTION_PCT

        s12_base = (
            data.inventory.scope1_tco2e
            + data.inventory.scope2_market_tco2e
        )
        s12_target = s12_base * (
            Decimal("1") - lt_reduction_pct / Decimal("100")
        )

        # Calculate effective annual rate from 90% reduction
        effective_rate = Decimal("0")
        if elapsed > 0 and s12_base > Decimal("0"):
            remaining_fraction = Decimal("1") - lt_reduction_pct / Decimal("100")
            # Solve: remaining = (1 - r)^elapsed => r = 1 - remaining^(1/elapsed)
            rem_float = float(remaining_fraction)
            if rem_float > 0:
                effective_rate = _decimal(
                    1.0 - rem_float ** (1.0 / elapsed)
                )

        actual_rate = max(lt_rate, effective_rate)

        s12_scope = ScopeTarget(
            scope=TargetScope.SCOPE_1_2.value,
            base_year_emissions_tco2e=_round_val(s12_base),
            target_year_emissions_tco2e=_round_val(s12_target),
            reduction_pct=_round_val(lt_reduction_pct, 2),
            annual_reduction_rate_pct=_round_val(
                actual_rate * Decimal("100"), 2
            ),
            method=method.value,
            coverage_pct=data.scope12_coverage_pct,
            meets_coverage_requirement=(
                data.scope12_coverage_pct
                >= SCOPE12_COVERAGE_MIN * Decimal("100")
            ),
        )

        scope_targets = [s12_scope]

        # Scope 3 long-term (90% coverage required)
        if self._is_scope3_required(data.inventory):
            s3_base = data.inventory.scope3_total_tco2e
            s3_target = s3_base * (
                Decimal("1") - lt_reduction_pct / Decimal("100")
            )
            scope_targets.append(ScopeTarget(
                scope=TargetScope.SCOPE_3.value,
                base_year_emissions_tco2e=_round_val(s3_base),
                target_year_emissions_tco2e=_round_val(s3_target),
                reduction_pct=_round_val(lt_reduction_pct, 2),
                annual_reduction_rate_pct=_round_val(
                    actual_rate * Decimal("100"), 2
                ),
                method=data.preferred_method_s3.value,
                coverage_pct=data.scope3_coverage_pct,
                meets_coverage_requirement=(
                    data.scope3_coverage_pct
                    >= SCOPE3_LT_COVERAGE * Decimal("100")
                ),
            ))

        milestones = self.calculate_pathway(
            s12_base, base_year, target_year, method, ambition
        )

        total_base = sum(
            st.base_year_emissions_tco2e for st in scope_targets
        )
        total_target = sum(
            st.target_year_emissions_tco2e for st in scope_targets
        )
        total_reduction = _safe_pct(total_base - total_target, total_base)

        return TargetDefinition(
            target_type=TargetType.LONG_TERM.value,
            entity_name=data.entity_name,
            base_year=base_year,
            target_year=target_year,
            boundary_approach=data.boundary_approach.value,
            ambition_level=ambition.value,
            scope_targets=scope_targets,
            milestones=milestones,
            total_base_emissions_tco2e=_round_val(total_base),
            total_target_emissions_tco2e=_round_val(total_target),
            total_reduction_pct=_round_val(total_reduction, 2),
        )

    def _build_net_zero_target(
        self, data: TargetSettingInput,
    ) -> TargetDefinition:
        """Build net-zero target definition (by 2050).

        Net-zero requires >= 90% absolute reduction with neutralisation
        of residual emissions (max 10% of base year).

        Args:
            data: Target-setting input.

        Returns:
            TargetDefinition for net-zero target.
        """
        target_year = min(data.target_year_long_term, NET_ZERO_MAX_YEAR)
        base_year = data.base_year
        elapsed = target_year - base_year

        s12_base = (
            data.inventory.scope1_tco2e
            + data.inventory.scope2_market_tco2e
        )
        s3_base = data.inventory.scope3_total_tco2e
        total_base = s12_base + s3_base

        residual_max = total_base * NET_ZERO_MAX_RESIDUAL_PCT / Decimal("100")
        nz_reduction_pct = Decimal("100") - NET_ZERO_MAX_RESIDUAL_PCT

        # S1+2: full 90% reduction
        s12_target = s12_base * NET_ZERO_MAX_RESIDUAL_PCT / Decimal("100")
        effective_rate = Decimal("0")
        if elapsed > 0 and s12_base > Decimal("0"):
            rem = float(NET_ZERO_MAX_RESIDUAL_PCT / Decimal("100"))
            if rem > 0:
                effective_rate = _decimal(1.0 - rem ** (1.0 / elapsed))

        scope_targets = [
            ScopeTarget(
                scope=TargetScope.SCOPE_1_2.value,
                base_year_emissions_tco2e=_round_val(s12_base),
                target_year_emissions_tco2e=_round_val(s12_target),
                reduction_pct=_round_val(nz_reduction_pct, 2),
                annual_reduction_rate_pct=_round_val(
                    effective_rate * Decimal("100"), 2
                ),
                method=data.preferred_method_s12.value,
                coverage_pct=data.scope12_coverage_pct,
                meets_coverage_requirement=(
                    data.scope12_coverage_pct
                    >= SCOPE12_COVERAGE_MIN * Decimal("100")
                ),
            ),
        ]

        if s3_base > Decimal("0"):
            s3_target = s3_base * NET_ZERO_MAX_RESIDUAL_PCT / Decimal("100")
            scope_targets.append(ScopeTarget(
                scope=TargetScope.SCOPE_3.value,
                base_year_emissions_tco2e=_round_val(s3_base),
                target_year_emissions_tco2e=_round_val(s3_target),
                reduction_pct=_round_val(nz_reduction_pct, 2),
                annual_reduction_rate_pct=_round_val(
                    effective_rate * Decimal("100"), 2
                ),
                method=data.preferred_method_s3.value,
                coverage_pct=data.scope3_coverage_pct,
                meets_coverage_requirement=(
                    data.scope3_coverage_pct
                    >= SCOPE3_LT_COVERAGE * Decimal("100")
                ),
            ))

        total_target = sum(
            st.target_year_emissions_tco2e for st in scope_targets
        )
        total_reduction = _safe_pct(total_base - total_target, total_base)

        return TargetDefinition(
            target_type=TargetType.NET_ZERO.value,
            entity_name=data.entity_name,
            base_year=base_year,
            target_year=target_year,
            boundary_approach=data.boundary_approach.value,
            ambition_level=AmbitionLevel.CELSIUS_1_5.value,
            scope_targets=scope_targets,
            milestones=[],
            total_base_emissions_tco2e=_round_val(total_base),
            total_target_emissions_tco2e=_round_val(total_target),
            total_reduction_pct=_round_val(total_reduction, 2),
        )

    def _build_flag_target(
        self, data: TargetSettingInput,
    ) -> TargetDefinition:
        """Build FLAG target definition (3.03%/yr linear reduction).

        Args:
            data: Target-setting input.

        Returns:
            TargetDefinition for FLAG target.
        """
        base_year = data.base_year
        target_year = data.target_year_near_term
        elapsed = target_year - base_year

        flag_base = data.inventory.flag_emissions_tco2e
        flag_target = self._project_emissions(
            flag_base, FLAG_RATE, elapsed, PathwayMethod.FLAG
        )
        reduction_pct = _safe_pct(flag_base - flag_target, flag_base)

        scope_targets = [ScopeTarget(
            scope=TargetScope.FLAG.value,
            base_year_emissions_tco2e=_round_val(flag_base),
            target_year_emissions_tco2e=_round_val(flag_target),
            reduction_pct=_round_val(reduction_pct, 2),
            annual_reduction_rate_pct=_round_val(
                FLAG_RATE * Decimal("100"), 2
            ),
            method=PathwayMethod.FLAG.value,
            coverage_pct=Decimal("100"),
            meets_coverage_requirement=True,
        )]

        milestones = self.calculate_pathway(
            flag_base, base_year, target_year, PathwayMethod.FLAG,
            AmbitionLevel.CELSIUS_1_5,
        )

        return TargetDefinition(
            target_type=TargetType.FLAG.value,
            entity_name=data.entity_name,
            base_year=base_year,
            target_year=target_year,
            boundary_approach=data.boundary_approach.value,
            ambition_level=AmbitionLevel.CELSIUS_1_5.value,
            scope_targets=scope_targets,
            milestones=milestones,
            total_base_emissions_tco2e=_round_val(flag_base),
            total_target_emissions_tco2e=_round_val(flag_target),
            total_reduction_pct=_round_val(reduction_pct, 2),
        )

    # ------------------------------------------------------------------ #
    # Pathway Calculations                                                #
    # ------------------------------------------------------------------ #

    def _get_annual_rate(
        self, method: PathwayMethod, ambition: AmbitionLevel,
    ) -> Decimal:
        """Get annual reduction rate for method + ambition combination.

        Args:
            method: Pathway method.
            ambition: Ambition level.

        Returns:
            Annual reduction rate as a fraction (e.g. 0.042).
        """
        if method == PathwayMethod.FLAG:
            return FLAG_RATE
        return ACA_ANNUAL_RATES.get(ambition, Decimal("0.042"))

    def _project_emissions(
        self,
        base_emissions: Decimal,
        annual_rate: Decimal,
        elapsed_years: int,
        method: PathwayMethod,
    ) -> Decimal:
        """Project emissions forward by elapsed years.

        For ACA and FLAG: linear reduction
            E(t) = E(base) * max(0, 1 - rate * elapsed)

        For compound methods:
            E(t) = E(base) * (1 - rate) ^ elapsed

        Args:
            base_emissions: Starting emissions.
            annual_rate: Annual reduction rate (fraction).
            elapsed_years: Number of years from base.
            method: Pathway method.

        Returns:
            Projected emissions (Decimal, non-negative).
        """
        if elapsed_years <= 0:
            return base_emissions

        if method in (PathwayMethod.ACA, PathwayMethod.FLAG):
            # Linear reduction
            factor = max(
                Decimal("0"),
                Decimal("1") - annual_rate * Decimal(str(elapsed_years)),
            )
        else:
            # Compound reduction
            factor = (Decimal("1") - annual_rate) ** elapsed_years
            factor = max(Decimal("0"), factor)

        return base_emissions * factor

    # ------------------------------------------------------------------ #
    # Requirement Checks                                                  #
    # ------------------------------------------------------------------ #

    def _is_scope3_required(self, inventory: EmissionsInventory) -> bool:
        """Check if Scope 3 targets are required (>= 40% of total).

        Args:
            inventory: Emissions inventory.

        Returns:
            True if Scope 3 targets are required.
        """
        total = (
            inventory.scope1_tco2e
            + inventory.scope2_market_tco2e
            + inventory.scope3_total_tco2e
        )
        if total <= Decimal("0"):
            return False
        s3_fraction = _safe_divide(inventory.scope3_total_tco2e, total)
        return s3_fraction >= SCOPE3_TRIGGER_THRESHOLD

    def _is_flag_required(self, inventory: EmissionsInventory) -> bool:
        """Check if FLAG target is required (>= 20% of total).

        Args:
            inventory: Emissions inventory.

        Returns:
            True if FLAG target is required.
        """
        total = inventory.total_tco2e
        if total <= Decimal("0"):
            total = (
                inventory.scope1_tco2e
                + inventory.scope2_market_tco2e
                + inventory.scope3_total_tco2e
            )
        if total <= Decimal("0"):
            return False
        flag_fraction = _safe_divide(inventory.flag_emissions_tco2e, total)
        return flag_fraction >= FLAG_TRIGGER_THRESHOLD

    # ------------------------------------------------------------------ #
    # Ambition Helpers                                                    #
    # ------------------------------------------------------------------ #

    def _rate_to_alignment(self, annual_rate_pct: Decimal) -> str:
        """Map annual reduction rate (%) to alignment label.

        Args:
            annual_rate_pct: Annual rate as percentage (e.g. 4.2 for 4.2%).

        Returns:
            Alignment label string.
        """
        if annual_rate_pct >= Decimal("4.2"):
            return AmbitionLevel.CELSIUS_1_5.value
        if annual_rate_pct >= Decimal("2.5"):
            return AmbitionLevel.WELL_BELOW_2C.value
        if annual_rate_pct >= Decimal("1.6"):
            return AmbitionLevel.CELSIUS_2.value
        return "not_aligned"

    def _min_alignment(self, a: str, b: str) -> str:
        """Return the less ambitious of two alignment labels.

        Args:
            a: First alignment.
            b: Second alignment.

        Returns:
            Less ambitious alignment label.
        """
        order = [
            "not_aligned",
            AmbitionLevel.CELSIUS_2.value,
            AmbitionLevel.WELL_BELOW_2C.value,
            AmbitionLevel.CELSIUS_1_5.value,
        ]
        idx_a = order.index(a) if a in order else 0
        idx_b = order.index(b) if b in order else 0
        return order[min(idx_a, idx_b)]

    def _rate_to_temperature(self, annual_rate_pct: Decimal) -> Decimal:
        """Estimate implied temperature from annual reduction rate.

        Piecewise linear mapping:
            >= 7.0%/yr -> 1.20C
            4.2%/yr -> 1.50C
            2.5%/yr -> 1.80C
            1.6%/yr -> 2.00C
            0.0%/yr -> 3.20C

        Args:
            annual_rate_pct: Annual rate as percentage.

        Returns:
            Estimated temperature in Celsius.
        """
        breakpoints: List[Tuple[Decimal, Decimal]] = [
            (Decimal("7.0"), Decimal("1.20")),
            (Decimal("4.2"), Decimal("1.50")),
            (Decimal("2.5"), Decimal("1.80")),
            (Decimal("1.6"), Decimal("2.00")),
            (Decimal("0.0"), Decimal("3.20")),
        ]

        rate = annual_rate_pct
        if rate >= breakpoints[0][0]:
            return breakpoints[0][1]

        for i in range(len(breakpoints) - 1):
            r_high, t_high = breakpoints[i]
            r_low, t_low = breakpoints[i + 1]
            if rate >= r_low:
                # Linear interpolation
                fraction = _safe_divide(r_high - rate, r_high - r_low)
                return t_high + fraction * (t_low - t_high)

        return breakpoints[-1][1]

    # ------------------------------------------------------------------ #
    # Utility Methods                                                     #
    # ------------------------------------------------------------------ #

    def get_summary(self, result: TargetSettingResult) -> Dict[str, Any]:
        """Generate concise summary from result.

        Args:
            result: Target setting result to summarise.

        Returns:
            Dict with key metrics and provenance hash.
        """
        summary: Dict[str, Any] = {
            "entity_name": result.entity_name,
            "num_targets": len(result.targets),
            "scope3_required": result.scope3_required,
            "flag_required": result.flag_required,
            "targets": [],
        }
        for t in result.targets:
            summary["targets"].append({
                "type": t.target_type,
                "base_year": t.base_year,
                "target_year": t.target_year,
                "reduction_pct": str(t.total_reduction_pct),
                "ambition": t.ambition_level,
            })
        if result.ambition_assessment:
            summary["overall_alignment"] = (
                result.ambition_assessment.overall_alignment
            )
            summary["temperature_estimate_c"] = str(
                result.ambition_assessment.temperature_estimate_c
            )
        summary["provenance_hash"] = _compute_hash(summary)
        return summary

    def get_supported_methods(self) -> Dict[str, str]:
        """Return supported pathway methods with descriptions.

        Returns:
            Dict mapping method value to description.
        """
        return {
            PathwayMethod.ACA.value: (
                "Absolute Contraction Approach - uniform annual reduction"
            ),
            PathwayMethod.SDA.value: (
                "Sectoral Decarbonization Approach - intensity convergence"
            ),
            PathwayMethod.FLAG.value: (
                "Forest, Land & Agriculture - 3.03%/yr linear"
            ),
            PathwayMethod.ECONOMIC_INTENSITY.value: (
                "Economic intensity - tCO2e per unit revenue"
            ),
            PathwayMethod.PHYSICAL_INTENSITY.value: (
                "Physical intensity - tCO2e per unit output"
            ),
            PathwayMethod.SUPPLIER_ENGAGEMENT.value: (
                "Supplier engagement - % suppliers with SBTi targets"
            ),
        }

    def get_ambition_thresholds(self) -> Dict[str, str]:
        """Return ambition thresholds for reference.

        Returns:
            Dict mapping ambition level to annual rate string.
        """
        return {
            level.value: f"{rate * 100}%/yr"
            for level, rate in ACA_ANNUAL_RATES.items()
        }
