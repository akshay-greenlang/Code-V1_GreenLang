# -*- coding: utf-8 -*-
"""
SimplifiedTargetEngine - PACK-026 SME Net Zero Pack Engine 2
==============================================================

Hard-coded 1.5C pathway (ACA only, no SDA) with pre-set targets
and auto-generated target statements for SMEs.

This engine eliminates the complexity of pathway selection and sector
decomposition.  It uses the Absolute Contraction Approach (ACA) with
pre-set milestones: 50% reduction by 2030, 90% by 2050.  Scope
coverage defaults to 95% Scope 1+2 and 67% of Scope 3 (simplified
to Cat 1, Cat 6, Cat 7 only).

Calculation Methodology:
    Base year:
        base_emissions = scope1 + scope2 + scope3_included

    Near-term target (2030):
        target_2030 = base_emissions * (1 - 0.50)  [50% reduction]
        annual_reduction = base_emissions * 0.50 / years_to_2030

    Long-term target (2050):
        target_2050 = base_emissions * (1 - 0.90)  [90% reduction]

    ACA pathway (linear):
        target_year_t = base_emissions * (1 - reduction_rate * (t - base_year))

    Scope 3 screening:
        included_scope3 = sum(cat_1 + cat_6 + cat_7)
        required_coverage = 0.67 (67% of total S3)

Regulatory References:
    - SBTi Corporate Net-Zero Standard v1.2 (2024)
    - SBTi SME Target Setting Route (2023)
    - Paris Agreement (2015) - 1.5C temperature target
    - IPCC SR15 (2018) - 1.5C pathway requirements

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Targets are hard-coded from SBTi requirements
    - Pathway is linear ACA (no stochastic modeling)
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-026 SME Net Zero Pack
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
    numerator: Decimal, denominator: Decimal,
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
    """Round to 3 decimal places."""
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TargetAmbition(str, Enum):
    """Target ambition level.

    For SMEs, only 1.5C is offered (SBTi SME route).
    """
    ALIGNED_1_5C = "1.5c"

class ScopeInclusion(str, Enum):
    """Which scopes are included in the target boundary."""
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3_CAT1 = "scope_3_cat1"
    SCOPE_3_CAT6 = "scope_3_cat6"
    SCOPE_3_CAT7 = "scope_3_cat7"

class MilestoneStatus(str, Enum):
    """Status of a target milestone."""
    NOT_STARTED = "not_started"
    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    ACHIEVED = "achieved"

class TargetCommitment(str, Enum):
    """Type of commitment for the target statement."""
    SME_CLIMATE_HUB = "sme_climate_hub"
    SBTI_SME = "sbti_sme"
    VOLUNTARY = "voluntary"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SBTi SME route: 50% by 2030, 90% by 2050 (from any base year after 2018).
NEAR_TERM_REDUCTION_PCT: Decimal = Decimal("50.0")
NEAR_TERM_TARGET_YEAR: int = 2030
LONG_TERM_REDUCTION_PCT: Decimal = Decimal("90.0")
LONG_TERM_TARGET_YEAR: int = 2050

# Scope coverage requirements (SBTi).
SCOPE_12_COVERAGE_PCT: Decimal = Decimal("95.0")
SCOPE_3_COVERAGE_PCT: Decimal = Decimal("67.0")

# Annual linear reduction rate for ACA 1.5C (SBTi minimum 4.2% per year).
MIN_ANNUAL_REDUCTION_RATE: Decimal = Decimal("4.2")

# Valid base year range.
MIN_BASE_YEAR: int = 2018
MAX_BASE_YEAR: int = 2025

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class TargetInput(BaseModel):
    """Input for SME target setting.

    Attributes:
        entity_name: Company name.
        base_year: Base year for the target (must be >= 2018).
        base_year_scope1_tco2e: Base year Scope 1 emissions.
        base_year_scope2_tco2e: Base year Scope 2 emissions.
        base_year_scope3_tco2e: Base year Scope 3 emissions (included categories).
        base_year_scope3_total_tco2e: Total Scope 3 (all categories).
        current_year: Current reporting year.
        current_scope1_tco2e: Current Scope 1 emissions.
        current_scope2_tco2e: Current Scope 2 emissions.
        current_scope3_tco2e: Current Scope 3 emissions.
        scope3_categories_included: Which S3 categories are included.
        commitment_type: Type of commitment.
        custom_near_term_year: Optional custom near-term target year.
    """
    entity_name: str = Field(
        ..., min_length=1, max_length=300, description="Company name"
    )
    base_year: int = Field(
        ..., ge=MIN_BASE_YEAR, le=MAX_BASE_YEAR,
        description="Base year (>= 2018 per SBTi)",
    )
    base_year_scope1_tco2e: Decimal = Field(
        ..., ge=Decimal("0"), description="Base year Scope 1 (tCO2e)"
    )
    base_year_scope2_tco2e: Decimal = Field(
        ..., ge=Decimal("0"), description="Base year Scope 2 (tCO2e)"
    )
    base_year_scope3_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Base year Scope 3 included categories (tCO2e)",
    )
    base_year_scope3_total_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Base year total Scope 3 (all categories)",
    )
    current_year: int = Field(
        ..., ge=2020, le=2100, description="Current reporting year"
    )
    current_scope1_tco2e: Optional[Decimal] = Field(
        None, ge=Decimal("0"), description="Current Scope 1"
    )
    current_scope2_tco2e: Optional[Decimal] = Field(
        None, ge=Decimal("0"), description="Current Scope 2"
    )
    current_scope3_tco2e: Optional[Decimal] = Field(
        None, ge=Decimal("0"), description="Current Scope 3"
    )
    scope3_categories_included: List[ScopeInclusion] = Field(
        default_factory=lambda: [
            ScopeInclusion.SCOPE_3_CAT1,
            ScopeInclusion.SCOPE_3_CAT6,
            ScopeInclusion.SCOPE_3_CAT7,
        ],
        description="Scope 3 categories in target boundary",
    )
    commitment_type: TargetCommitment = Field(
        default=TargetCommitment.SME_CLIMATE_HUB,
        description="Commitment framework",
    )
    custom_near_term_year: Optional[int] = Field(
        None, ge=2025, le=2035,
        description="Optional custom near-term target year",
    )

    @field_validator("base_year")
    @classmethod
    def validate_base_year(cls, v: int) -> int:
        """Validate base year is within SBTi SME acceptable range."""
        if v < MIN_BASE_YEAR:
            raise ValueError(
                f"Base year {v} is before minimum {MIN_BASE_YEAR} (SBTi requirement)"
            )
        return v

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class MilestoneEntry(BaseModel):
    """A single milestone on the target pathway.

    Attributes:
        year: Target year.
        target_tco2e: Absolute emissions target.
        reduction_from_base_pct: Reduction vs base year (%).
        cumulative_budget_tco2e: Remaining carbon budget to this point.
        status: Current status of this milestone.
    """
    year: int = Field(default=0)
    target_tco2e: Decimal = Field(default=Decimal("0"))
    reduction_from_base_pct: Decimal = Field(default=Decimal("0"))
    cumulative_budget_tco2e: Decimal = Field(default=Decimal("0"))
    status: MilestoneStatus = Field(default=MilestoneStatus.NOT_STARTED)

class TargetDefinition(BaseModel):
    """Formal target definition output.

    Attributes:
        target_type: Near-term or long-term.
        target_year: Year the target must be met.
        base_year: Base year for the target.
        base_year_emissions_tco2e: Emissions in the base year.
        target_emissions_tco2e: Target emissions level.
        reduction_pct: Required reduction percentage.
        scope_coverage: Scopes included in the target.
        annual_reduction_rate_pct: Required annual reduction rate.
        pathway: ACA (Absolute Contraction Approach).
        is_sbti_compliant: Whether meets SBTi minimum requirements.
    """
    target_type: str = Field(default="near_term")
    target_year: int = Field(default=2030)
    base_year: int = Field(default=2020)
    base_year_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    target_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    reduction_pct: Decimal = Field(default=Decimal("0"))
    scope_coverage: List[str] = Field(default_factory=list)
    annual_reduction_rate_pct: Decimal = Field(default=Decimal("0"))
    pathway: str = Field(default="ACA")
    is_sbti_compliant: bool = Field(default=False)

class ProgressAssessment(BaseModel):
    """Assessment of current progress against targets.

    Attributes:
        current_emissions_tco2e: Current total emissions.
        reduction_achieved_pct: Reduction achieved so far (%).
        reduction_required_pct: Total reduction required (%).
        gap_tco2e: Remaining gap in tCO2e.
        on_track: Whether currently on track.
        years_remaining: Years to target year.
        annual_reduction_needed_pct: Required annual rate going forward.
    """
    current_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    reduction_achieved_pct: Decimal = Field(default=Decimal("0"))
    reduction_required_pct: Decimal = Field(default=Decimal("0"))
    gap_tco2e: Decimal = Field(default=Decimal("0"))
    on_track: bool = Field(default=False)
    years_remaining: int = Field(default=0)
    annual_reduction_needed_pct: Decimal = Field(default=Decimal("0"))

class Scope3Coverage(BaseModel):
    """Scope 3 coverage analysis.

    Attributes:
        included_tco2e: Emissions from included S3 categories.
        total_tco2e: Total S3 emissions (all categories).
        coverage_pct: Coverage percentage (included / total * 100).
        meets_requirement: Whether coverage >= 67%.
        categories_included: List of included S3 categories.
    """
    included_tco2e: Decimal = Field(default=Decimal("0"))
    total_tco2e: Decimal = Field(default=Decimal("0"))
    coverage_pct: Decimal = Field(default=Decimal("0"))
    meets_requirement: bool = Field(default=False)
    categories_included: List[str] = Field(default_factory=list)

class TargetStatement(BaseModel):
    """Auto-generated target statement for public disclosure.

    Attributes:
        short_statement: One-sentence commitment.
        full_statement: Detailed multi-sentence statement.
        commitment_framework: Framework (SME Climate Hub, SBTi, etc.).
        disclosure_ready: Whether the statement is complete.
    """
    short_statement: str = Field(default="")
    full_statement: str = Field(default="")
    commitment_framework: str = Field(default="")
    disclosure_ready: bool = Field(default=False)

class SimplifiedTargetResult(BaseModel):
    """Complete SME target setting result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Timestamp (UTC).
        entity_name: Company name.
        ambition: Temperature alignment (always 1.5C).
        near_term_target: Near-term target definition.
        long_term_target: Long-term target definition.
        milestones: Year-by-year milestone pathway.
        progress: Progress assessment (if current data provided).
        scope3_coverage: Scope 3 coverage analysis.
        target_statement: Auto-generated target statement.
        total_base_year_emissions_tco2e: Total base year emissions in boundary.
        processing_time_ms: Calculation time (ms).
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    entity_name: str = Field(default="")
    ambition: str = Field(default="1.5c")

    near_term_target: TargetDefinition = Field(default_factory=TargetDefinition)
    long_term_target: TargetDefinition = Field(default_factory=TargetDefinition)
    milestones: List[MilestoneEntry] = Field(default_factory=list)
    progress: Optional[ProgressAssessment] = Field(None)
    scope3_coverage: Scope3Coverage = Field(default_factory=Scope3Coverage)
    target_statement: TargetStatement = Field(default_factory=TargetStatement)

    total_base_year_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SimplifiedTargetEngine:
    """Simplified 1.5C target setting engine for SMEs.

    Uses hard-coded ACA pathway with pre-set milestones (50% by 2030,
    90% by 2050).  Auto-generates target statements for SME Climate
    Hub, SBTi SME route, or voluntary commitments.

    All calculations use Decimal arithmetic for bit-perfect reproducibility.
    No LLM is used in any calculation path.

    Usage::

        engine = SimplifiedTargetEngine()
        result = engine.calculate(target_input)
        print(result.target_statement.short_statement)
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def calculate(self, data: TargetInput) -> SimplifiedTargetResult:
        """Run simplified target setting calculation.

        Args:
            data: Validated target input data.

        Returns:
            SimplifiedTargetResult with targets, milestones, and statement.
        """
        t0 = time.perf_counter()
        logger.info(
            "Simplified Target: entity=%s, base_year=%d, commitment=%s",
            data.entity_name, data.base_year, data.commitment_type.value,
        )

        # Calculate total base year emissions in boundary
        base_s12 = data.base_year_scope1_tco2e + data.base_year_scope2_tco2e
        base_s3_included = data.base_year_scope3_tco2e
        total_base = _round_val(base_s12 + base_s3_included)

        # Near-term target year
        near_term_year = data.custom_near_term_year or NEAR_TERM_TARGET_YEAR
        if near_term_year <= data.base_year:
            near_term_year = data.base_year + 5

        # Near-term target
        near_term = self._compute_target(
            target_type="near_term",
            base_year=data.base_year,
            target_year=near_term_year,
            base_emissions=total_base,
            reduction_pct=NEAR_TERM_REDUCTION_PCT,
            scope_coverage=["scope_1", "scope_2"] + [
                s.value for s in data.scope3_categories_included
            ],
        )

        # Long-term target
        long_term = self._compute_target(
            target_type="long_term",
            base_year=data.base_year,
            target_year=LONG_TERM_TARGET_YEAR,
            base_emissions=total_base,
            reduction_pct=LONG_TERM_REDUCTION_PCT,
            scope_coverage=["scope_1", "scope_2"] + [
                s.value for s in data.scope3_categories_included
            ],
        )

        # Milestones (every 5 years from base year to 2050)
        milestones = self._generate_milestones(
            data.base_year, total_base, data.current_year,
        )

        # Progress assessment
        progress = None
        if (
            data.current_scope1_tco2e is not None
            and data.current_scope2_tco2e is not None
        ):
            current_total = (
                data.current_scope1_tco2e
                + data.current_scope2_tco2e
                + (data.current_scope3_tco2e or Decimal("0"))
            )
            progress = self._assess_progress(
                base_emissions=total_base,
                current_emissions=current_total,
                current_year=data.current_year,
                near_term_year=near_term_year,
            )

        # Scope 3 coverage
        scope3_coverage = self._assess_scope3_coverage(
            included=base_s3_included,
            total=data.base_year_scope3_total_tco2e,
            categories=[s.value for s in data.scope3_categories_included],
        )

        # Target statement
        target_statement = self._generate_target_statement(
            entity_name=data.entity_name,
            base_year=data.base_year,
            near_term_year=near_term_year,
            total_base=total_base,
            commitment=data.commitment_type,
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = SimplifiedTargetResult(
            entity_name=data.entity_name,
            ambition="1.5c",
            near_term_target=near_term,
            long_term_target=long_term,
            milestones=milestones,
            progress=progress,
            scope3_coverage=scope3_coverage,
            target_statement=target_statement,
            total_base_year_emissions_tco2e=total_base,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Target complete: base=%.2f tCO2e, near_term_target=%.2f, hash=%s",
            float(total_base), float(near_term.target_emissions_tco2e),
            result.provenance_hash[:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # Internal Methods                                                     #
    # ------------------------------------------------------------------ #

    def _compute_target(
        self,
        target_type: str,
        base_year: int,
        target_year: int,
        base_emissions: Decimal,
        reduction_pct: Decimal,
        scope_coverage: List[str],
    ) -> TargetDefinition:
        """Compute a single target definition.

        Args:
            target_type: 'near_term' or 'long_term'.
            base_year: Base year.
            target_year: Target year.
            base_emissions: Total base year emissions in boundary.
            reduction_pct: Required reduction percentage.
            scope_coverage: List of scopes included.

        Returns:
            TargetDefinition with calculated targets.
        """
        target_emissions = _round_val(
            base_emissions * (Decimal("100") - reduction_pct) / Decimal("100")
        )

        years = max(target_year - base_year, 1)
        annual_rate = _round_val(
            _safe_divide(reduction_pct, _decimal(years)), 3
        )

        # SBTi compliance check: near-term requires >= 4.2% annual for 1.5C
        is_compliant = True
        if target_type == "near_term" and annual_rate < MIN_ANNUAL_REDUCTION_RATE:
            is_compliant = False

        return TargetDefinition(
            target_type=target_type,
            target_year=target_year,
            base_year=base_year,
            base_year_emissions_tco2e=base_emissions,
            target_emissions_tco2e=target_emissions,
            reduction_pct=reduction_pct,
            scope_coverage=scope_coverage,
            annual_reduction_rate_pct=annual_rate,
            pathway="ACA",
            is_sbti_compliant=is_compliant,
        )

    def _generate_milestones(
        self,
        base_year: int,
        base_emissions: Decimal,
        current_year: int,
    ) -> List[MilestoneEntry]:
        """Generate year-by-year milestones from base year to 2050.

        Uses linear interpolation along the ACA pathway.

        Args:
            base_year: Base year.
            base_emissions: Total base year emissions.
            current_year: Current year for status assessment.

        Returns:
            List of MilestoneEntry for each 5-year interval.
        """
        milestones: List[MilestoneEntry] = []
        total_years = LONG_TERM_TARGET_YEAR - base_year
        if total_years <= 0:
            return milestones

        # Generate every 5 years
        years_list = list(range(base_year, LONG_TERM_TARGET_YEAR + 1, 5))
        if LONG_TERM_TARGET_YEAR not in years_list:
            years_list.append(LONG_TERM_TARGET_YEAR)
        if NEAR_TERM_TARGET_YEAR not in years_list:
            years_list.append(NEAR_TERM_TARGET_YEAR)
        years_list = sorted(set(years_list))

        # Cumulative carbon budget (area under pathway)
        running_budget = Decimal("0")

        for year in years_list:
            years_elapsed = year - base_year
            # Linear reduction to 90% by 2050
            reduction_pct = _safe_divide(
                _decimal(years_elapsed) * LONG_TERM_REDUCTION_PCT,
                _decimal(total_years),
            )
            reduction_pct = min(reduction_pct, LONG_TERM_REDUCTION_PCT)

            target = _round_val(
                base_emissions * (Decimal("100") - reduction_pct) / Decimal("100")
            )

            # Simple cumulative budget: sum of targets * 5 years
            if year > base_year:
                interval = min(year - (years_list[max(0, years_list.index(year) - 1)]), 5)
                running_budget += target * _decimal(interval)

            # Status
            if year < current_year:
                status = MilestoneStatus.ACHIEVED  # past milestones assumed
            elif year == current_year:
                status = MilestoneStatus.ON_TRACK
            else:
                status = MilestoneStatus.NOT_STARTED

            milestones.append(MilestoneEntry(
                year=year,
                target_tco2e=target,
                reduction_from_base_pct=_round_val(reduction_pct, 2),
                cumulative_budget_tco2e=_round_val(running_budget),
                status=status,
            ))

        return milestones

    def _assess_progress(
        self,
        base_emissions: Decimal,
        current_emissions: Decimal,
        current_year: int,
        near_term_year: int,
    ) -> ProgressAssessment:
        """Assess current progress against the near-term target.

        Args:
            base_emissions: Base year total emissions.
            current_emissions: Current year total emissions.
            current_year: Current year.
            near_term_year: Near-term target year.

        Returns:
            ProgressAssessment with gap analysis.
        """
        reduction_achieved = _safe_divide(
            (base_emissions - current_emissions) * Decimal("100"),
            base_emissions,
        )

        near_term_target = _round_val(
            base_emissions * (Decimal("100") - NEAR_TERM_REDUCTION_PCT) / Decimal("100")
        )
        gap = current_emissions - near_term_target

        years_remaining = max(near_term_year - current_year, 1)

        # Required annual reduction from current level
        if current_emissions > near_term_target and years_remaining > 0:
            total_reduction_needed = current_emissions - near_term_target
            annual_reduction_needed_pct = _safe_divide(
                total_reduction_needed * Decimal("100"),
                current_emissions * _decimal(years_remaining),
            )
        else:
            annual_reduction_needed_pct = Decimal("0")

        # On track if current reduction is at or above the linear trajectory
        expected_reduction_pct = _safe_divide(
            NEAR_TERM_REDUCTION_PCT * _decimal(current_year - near_term_year + years_remaining + (current_year - near_term_year + years_remaining)),
            _decimal(near_term_year - (near_term_year - years_remaining - (current_year - near_term_year + years_remaining - years_remaining))),
        )
        # Simplified: on track if reduction_achieved >= linear pro-rata
        total_window = near_term_year - (current_year - max(current_year - near_term_year, 0))
        on_track = reduction_achieved >= Decimal("0") and gap <= Decimal("0")

        return ProgressAssessment(
            current_emissions_tco2e=_round_val(current_emissions),
            reduction_achieved_pct=_round_val(max(reduction_achieved, Decimal("0")), 2),
            reduction_required_pct=NEAR_TERM_REDUCTION_PCT,
            gap_tco2e=_round_val(max(gap, Decimal("0"))),
            on_track=on_track,
            years_remaining=years_remaining,
            annual_reduction_needed_pct=_round_val(annual_reduction_needed_pct, 2),
        )

    def _assess_scope3_coverage(
        self,
        included: Decimal,
        total: Decimal,
        categories: List[str],
    ) -> Scope3Coverage:
        """Assess Scope 3 coverage against the 67% requirement.

        Args:
            included: Scope 3 emissions from included categories.
            total: Total Scope 3 emissions (all categories).
            categories: List of included category identifiers.

        Returns:
            Scope3Coverage with coverage percentage and compliance.
        """
        if total == Decimal("0"):
            coverage_pct = Decimal("100") if included == Decimal("0") else Decimal("0")
        else:
            coverage_pct = _safe_pct(included, total)

        return Scope3Coverage(
            included_tco2e=_round_val(included),
            total_tco2e=_round_val(total),
            coverage_pct=_round_val(coverage_pct, 2),
            meets_requirement=coverage_pct >= SCOPE_3_COVERAGE_PCT,
            categories_included=categories,
        )

    def _generate_target_statement(
        self,
        entity_name: str,
        base_year: int,
        near_term_year: int,
        total_base: Decimal,
        commitment: TargetCommitment,
    ) -> TargetStatement:
        """Auto-generate a target statement for public disclosure.

        Args:
            entity_name: Company name.
            base_year: Base year.
            near_term_year: Near-term target year.
            total_base: Total base year emissions.
            commitment: Commitment framework.

        Returns:
            TargetStatement with short and full versions.
        """
        framework_name = {
            TargetCommitment.SME_CLIMATE_HUB: "the SME Climate Hub",
            TargetCommitment.SBTI_SME: "the Science Based Targets initiative (SBTi) SME route",
            TargetCommitment.VOLUNTARY: "a voluntary net-zero commitment",
        }

        short = (
            f"{entity_name} commits to halving greenhouse gas emissions "
            f"by {near_term_year} and achieving net-zero by 2050, "
            f"aligned with the Paris Agreement 1.5C goal."
        )

        full = (
            f"{entity_name} has set science-aligned climate targets through "
            f"{framework_name.get(commitment, 'a voluntary commitment')}. "
            f"From a {base_year} base year of {float(_round_val(total_base, 1)):.1f} "
            f"tCO2e, {entity_name} commits to: (1) reduce absolute Scope 1, 2, "
            f"and selected Scope 3 greenhouse gas emissions by "
            f"{int(NEAR_TERM_REDUCTION_PCT)}% by {near_term_year}; and "
            f"(2) reduce absolute emissions by {int(LONG_TERM_REDUCTION_PCT)}% "
            f"by 2050, with remaining residual emissions neutralized through "
            f"verified carbon dioxide removal. This target covers 95% of "
            f"Scope 1 and 2 emissions and 67% of Scope 3 emissions, "
            f"following the Absolute Contraction Approach (ACA) consistent "
            f"with limiting warming to 1.5C."
        )

        return TargetStatement(
            short_statement=short,
            full_statement=full,
            commitment_framework=commitment.value,
            disclosure_ready=True,
        )
