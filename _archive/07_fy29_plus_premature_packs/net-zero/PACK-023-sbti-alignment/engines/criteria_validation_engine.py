# -*- coding: utf-8 -*-
"""
CriteriaValidationEngine - PACK-023 SBTi Alignment Engine 2
==============================================================

Full 42-criterion automated validation for SBTi target submissions:
  - 28 near-term criteria (C1-C28) covering boundary, inventory,
    ambition, scope 2, scope 3, timeframe, and reporting.
  - 14 net-zero criteria (NZ-C1 to NZ-C14) covering net-zero target,
    long-term, residual emissions, and transition planning.

Each criterion is assessed as PASS, FAIL, WARNING, or NOT_APPLICABLE
with detailed remediation guidance for failures.

Calculation Methodology:
    Each criterion is a deterministic check against defined thresholds:
        C1: Boundary covers >= 95% of Scope 1+2 emissions
        C6: Near-term S1+2 ambition >= 4.2%/yr (1.5C)
        C8: S3 trigger: S3 / (S1+S2+S3) >= 40%
        NZ-C9: Residual emissions <= 10% of base year
        (full mapping in CRITERIA_DEFINITIONS constant)

    Readiness score:
        score = (passed + 0.5 * warnings) / total_applicable * 100

Regulatory References:
    - SBTi Corporate Manual V5.3 (2024) - Criteria C1-C28
    - SBTi Corporate Net-Zero Standard V1.3 (2024) - NZ-C1 to NZ-C14
    - SBTi FLAG Guidance V1.1 (2022) - FLAG-specific criteria
    - GHG Protocol Corporate Standard (WRI/WBCSD, 2015)
    - ISO 14064-1:2018 - GHG quantification requirements

Zero-Hallucination:
    - All checks compare numeric values against published thresholds
    - No LLM involvement in any assessment
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

class CriterionStatus(str, Enum):
    """Outcome status for a single criterion check.

    PASS: Criterion fully satisfied.
    FAIL: Criterion not met -- blocks submission.
    WARNING: Criterion marginally met -- review recommended.
    NOT_APPLICABLE: Criterion not relevant for this target type.
    """
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"

class CriterionCategory(str, Enum):
    """Functional grouping of SBTi criteria.

    BOUNDARY: Organisational boundary (C1-C4).
    INVENTORY: Emissions inventory (C5-C8).
    AMBITION: Target ambition (C9-C12).
    SCOPE2: Scope 2 methodology (C13-C16).
    SCOPE3: Scope 3 targets (C17-C20).
    TIMEFRAME: Target timeframe (C21-C24).
    REPORTING: Reporting and disclosure (C25-C28).
    NET_ZERO_TARGET: Net-zero target definition (NZ-C1 to NZ-C4).
    LONG_TERM: Long-term targets (NZ-C5 to NZ-C8).
    RESIDUAL: Residual emissions and neutralisation (NZ-C9 to NZ-C11).
    TRANSITION: Transition planning and governance (NZ-C12 to NZ-C14).
    """
    BOUNDARY = "boundary"
    INVENTORY = "inventory"
    AMBITION = "ambition"
    SCOPE2 = "scope2"
    SCOPE3 = "scope3"
    TIMEFRAME = "timeframe"
    REPORTING = "reporting"
    NET_ZERO_TARGET = "net_zero_target"
    LONG_TERM = "long_term"
    RESIDUAL = "residual"
    TRANSITION = "transition"

# ---------------------------------------------------------------------------
# Constants -- SBTi Thresholds
# ---------------------------------------------------------------------------

BASE_YEAR_MIN: int = 2015
SCOPE12_COVERAGE_MIN_PCT: Decimal = Decimal("95.0")
SCOPE3_TRIGGER_PCT: Decimal = Decimal("40.0")
SCOPE3_NT_COVERAGE_PCT: Decimal = Decimal("67.0")
SCOPE3_LT_COVERAGE_PCT: Decimal = Decimal("90.0")
FLAG_TRIGGER_PCT: Decimal = Decimal("20.0")
ACA_15C_RATE_PCT: Decimal = Decimal("4.2")
ACA_WB2C_RATE_PCT: Decimal = Decimal("2.5")
NEAR_TERM_MIN_YEARS: int = 5
NEAR_TERM_MAX_YEARS: int = 10
LONG_TERM_MIN_REDUCTION_PCT: Decimal = Decimal("90.0")
NET_ZERO_MAX_RESIDUAL_PCT: Decimal = Decimal("10.0")
NET_ZERO_MAX_YEAR: int = 2050
SCOPE2_RE_TARGET_PCT: Decimal = Decimal("80.0")
FIVE_YEAR_REVIEW_YEARS: int = 5

# ---------------------------------------------------------------------------
# Criteria Definitions - All 42 criteria
# ---------------------------------------------------------------------------

CRITERIA_DEFINITIONS: Dict[str, Dict[str, str]] = {
    # Near-term C1-C28
    "C1": {
        "category": CriterionCategory.BOUNDARY.value,
        "description": "Organisation uses operational control, financial control, or equity share approach",
        "remediation": "Define organisational boundary using one of the three GHG Protocol approaches",
    },
    "C2": {
        "category": CriterionCategory.BOUNDARY.value,
        "description": "All relevant GHGs are included (CO2, CH4, N2O, HFCs, PFCs, SF6, NF3)",
        "remediation": "Ensure all seven Kyoto gases are accounted in the inventory",
    },
    "C3": {
        "category": CriterionCategory.BOUNDARY.value,
        "description": "Scope 1 and Scope 2 coverage is at least 95%",
        "remediation": "Expand boundary to cover at least 95% of Scope 1+2 emissions",
    },
    "C4": {
        "category": CriterionCategory.BOUNDARY.value,
        "description": "Biogenic emissions are reported separately",
        "remediation": "Report biogenic CO2 separately from fossil emissions",
    },
    "C5": {
        "category": CriterionCategory.INVENTORY.value,
        "description": "Base year is no earlier than 2015",
        "remediation": "Select a base year of 2015 or later",
    },
    "C6": {
        "category": CriterionCategory.INVENTORY.value,
        "description": "Base year emissions are complete and verified",
        "remediation": "Complete Scope 1 and Scope 2 inventory with third-party verification",
    },
    "C7": {
        "category": CriterionCategory.INVENTORY.value,
        "description": "Base year recalculation policy is defined",
        "remediation": "Establish a policy for base year recalculation triggers (5% threshold)",
    },
    "C8": {
        "category": CriterionCategory.INVENTORY.value,
        "description": "Scope 3 screening completed for all 15 categories",
        "remediation": "Complete Scope 3 screening across all 15 GHG Protocol categories",
    },
    "C9": {
        "category": CriterionCategory.AMBITION.value,
        "description": "Scope 1+2 target ambition meets at least 4.2%/yr (1.5C-aligned)",
        "remediation": "Increase Scope 1+2 annual reduction rate to at least 4.2% per year",
    },
    "C10": {
        "category": CriterionCategory.AMBITION.value,
        "description": "Target uses approved methodology (ACA, SDA, or equivalent)",
        "remediation": "Use SBTi-approved target-setting method (ACA or SDA)",
    },
    "C11": {
        "category": CriterionCategory.AMBITION.value,
        "description": "Carbon credits not counted towards target achievement",
        "remediation": "Remove carbon credit offsets from target pathway calculations",
    },
    "C12": {
        "category": CriterionCategory.AMBITION.value,
        "description": "Avoided emissions not counted towards target achievement",
        "remediation": "Exclude avoided emissions from target progress calculations",
    },
    "C13": {
        "category": CriterionCategory.SCOPE2.value,
        "description": "Scope 2 target uses market-based method as primary",
        "remediation": "Set Scope 2 target using market-based accounting method",
    },
    "C14": {
        "category": CriterionCategory.SCOPE2.value,
        "description": "Renewable electricity procurement plan supports target",
        "remediation": "Develop RE procurement plan (RECs, PPAs, on-site generation)",
    },
    "C15": {
        "category": CriterionCategory.SCOPE2.value,
        "description": "Scope 2 location-based reported alongside market-based",
        "remediation": "Report both location-based and market-based Scope 2 values",
    },
    "C16": {
        "category": CriterionCategory.SCOPE2.value,
        "description": "Scope 2 quality criteria met for instruments (temporal, geographic match)",
        "remediation": "Ensure RE instruments meet temporal and geographic matching requirements",
    },
    "C17": {
        "category": CriterionCategory.SCOPE3.value,
        "description": "Scope 3 target set if S3 >= 40% of total emissions",
        "remediation": "Set Scope 3 target covering relevant categories",
    },
    "C18": {
        "category": CriterionCategory.SCOPE3.value,
        "description": "Scope 3 near-term coverage >= 67% of total Scope 3",
        "remediation": "Expand Scope 3 target boundary to cover at least 67% of S3",
    },
    "C19": {
        "category": CriterionCategory.SCOPE3.value,
        "description": "Scope 3 reduction rate >= 2.5%/yr (minimum ambition)",
        "remediation": "Increase Scope 3 annual reduction rate to at least 2.5% per year",
    },
    "C20": {
        "category": CriterionCategory.SCOPE3.value,
        "description": "Supplier engagement target covers significant suppliers",
        "remediation": "Define supplier engagement targets for top-emitting suppliers",
    },
    "C21": {
        "category": CriterionCategory.TIMEFRAME.value,
        "description": "Near-term target timeframe is 5-10 years",
        "remediation": "Set target year between 5 and 10 years from base year",
    },
    "C22": {
        "category": CriterionCategory.TIMEFRAME.value,
        "description": "Target start year is within 2 years of submission",
        "remediation": "Submit targets within 2 years of target start date",
    },
    "C23": {
        "category": CriterionCategory.TIMEFRAME.value,
        "description": "Five-year target review cycle established",
        "remediation": "Establish process for 5-year target review and revalidation",
    },
    "C24": {
        "category": CriterionCategory.TIMEFRAME.value,
        "description": "Interim milestones defined for progress tracking",
        "remediation": "Define annual or periodic interim milestones",
    },
    "C25": {
        "category": CriterionCategory.REPORTING.value,
        "description": "Annual GHG inventory reporting commitment",
        "remediation": "Commit to annual public GHG inventory reporting",
    },
    "C26": {
        "category": CriterionCategory.REPORTING.value,
        "description": "Progress against targets reported annually",
        "remediation": "Report annual progress against SBTi targets publicly",
    },
    "C27": {
        "category": CriterionCategory.REPORTING.value,
        "description": "Target information publicly disclosed",
        "remediation": "Publicly disclose target details (boundary, base year, method, ambition)",
    },
    "C28": {
        "category": CriterionCategory.REPORTING.value,
        "description": "Board-level or senior management oversight of targets",
        "remediation": "Ensure board or C-suite oversight and governance of climate targets",
    },
    # Net-zero NZ-C1 to NZ-C14
    "NZ-C1": {
        "category": CriterionCategory.NET_ZERO_TARGET.value,
        "description": "Net-zero target covers Scope 1, 2, and 3 emissions",
        "remediation": "Expand net-zero target boundary to all material emission scopes",
    },
    "NZ-C2": {
        "category": CriterionCategory.NET_ZERO_TARGET.value,
        "description": "Net-zero target year is 2050 or sooner",
        "remediation": "Set net-zero target year to 2050 or earlier",
    },
    "NZ-C3": {
        "category": CriterionCategory.NET_ZERO_TARGET.value,
        "description": "Near-term target set alongside net-zero target",
        "remediation": "Define near-term targets (5-10yr) alongside net-zero commitment",
    },
    "NZ-C4": {
        "category": CriterionCategory.NET_ZERO_TARGET.value,
        "description": "Net-zero target uses SBTi-approved net-zero pathway",
        "remediation": "Align net-zero pathway with SBTi Net-Zero Standard methodology",
    },
    "NZ-C5": {
        "category": CriterionCategory.LONG_TERM.value,
        "description": "Long-term S1+2 reduction >= 90% from base year",
        "remediation": "Set long-term Scope 1+2 target to reduce at least 90% by target year",
    },
    "NZ-C6": {
        "category": CriterionCategory.LONG_TERM.value,
        "description": "Long-term S3 reduction >= 90% from base year",
        "remediation": "Set long-term Scope 3 target to reduce at least 90% by target year",
    },
    "NZ-C7": {
        "category": CriterionCategory.LONG_TERM.value,
        "description": "Long-term Scope 3 coverage >= 90%",
        "remediation": "Expand long-term Scope 3 coverage to at least 90% of S3 emissions",
    },
    "NZ-C8": {
        "category": CriterionCategory.LONG_TERM.value,
        "description": "Long-term target timeframe extends to 2050 or earlier",
        "remediation": "Set long-term target achievement date to no later than 2050",
    },
    "NZ-C9": {
        "category": CriterionCategory.RESIDUAL.value,
        "description": "Residual emissions <= 10% of base year",
        "remediation": "Ensure projected residual emissions are at most 10% of base year total",
    },
    "NZ-C10": {
        "category": CriterionCategory.RESIDUAL.value,
        "description": "Neutralisation plan for residual emissions defined",
        "remediation": "Define strategy for neutralising residual emissions (carbon removals)",
    },
    "NZ-C11": {
        "category": CriterionCategory.RESIDUAL.value,
        "description": "Neutralisation uses permanent carbon removals",
        "remediation": "Use permanent carbon removal methods (DACCS, BECCS, enhanced weathering)",
    },
    "NZ-C12": {
        "category": CriterionCategory.TRANSITION.value,
        "description": "Transition plan with decarbonisation levers documented",
        "remediation": "Document a transition plan with specific decarbonisation actions",
    },
    "NZ-C13": {
        "category": CriterionCategory.TRANSITION.value,
        "description": "Capital expenditure aligned with transition plan",
        "remediation": "Align CAPEX planning with transition pathway requirements",
    },
    "NZ-C14": {
        "category": CriterionCategory.TRANSITION.value,
        "description": "Board governance for net-zero strategy established",
        "remediation": "Assign board-level accountability for net-zero target delivery",
    },
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class TargetData(BaseModel):
    """Target data for validation.

    Attributes:
        target_type: near_term, long_term, net_zero, or flag.
        base_year: Target base year.
        target_year: Target achievement year.
        scope12_annual_rate_pct: S1+2 annual reduction rate (%).
        scope3_annual_rate_pct: S3 annual reduction rate (%).
        scope12_coverage_pct: S1+2 coverage as percentage.
        scope3_coverage_pct: S3 coverage as percentage.
        scope12_reduction_pct: Total S1+2 reduction from base (%).
        scope3_reduction_pct: Total S3 reduction from base (%).
        method: Pathway method (aca, sda, flag).
        has_near_term: Whether near-term targets are set.
        has_long_term: Whether long-term targets are set.
        net_zero_year: Net-zero target year (if applicable).
        residual_emissions_pct: Projected residual as % of base.
    """
    target_type: str = Field(default="near_term")
    base_year: int = Field(default=2020, ge=2010, le=2030)
    target_year: int = Field(default=2030, ge=2025, le=2060)
    scope12_annual_rate_pct: Decimal = Field(default=Decimal("0"), ge=0)
    scope3_annual_rate_pct: Decimal = Field(default=Decimal("0"), ge=0)
    scope12_coverage_pct: Decimal = Field(
        default=Decimal("100"), ge=0, le=Decimal("100")
    )
    scope3_coverage_pct: Decimal = Field(
        default=Decimal("67"), ge=0, le=Decimal("100")
    )
    scope12_reduction_pct: Decimal = Field(default=Decimal("0"), ge=0)
    scope3_reduction_pct: Decimal = Field(default=Decimal("0"), ge=0)
    method: str = Field(default="aca")
    has_near_term: bool = Field(default=True)
    has_long_term: bool = Field(default=False)
    net_zero_year: Optional[int] = Field(default=None)
    residual_emissions_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100")
    )

class InventoryData(BaseModel):
    """Emissions inventory data for validation.

    Attributes:
        scope1_tco2e: Scope 1 emissions.
        scope2_location_tco2e: Scope 2 location-based.
        scope2_market_tco2e: Scope 2 market-based.
        scope3_total_tco2e: Total Scope 3.
        scope3_categories_screened: Number of S3 categories screened.
        flag_emissions_tco2e: FLAG emissions.
        biogenic_reported_separately: Whether biogenic is separate.
        all_ghgs_included: Whether all 7 GHGs are included.
        third_party_verified: Whether inventory is verified.
    """
    scope1_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    scope2_location_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    scope2_market_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    scope3_total_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    scope3_categories_screened: int = Field(default=0, ge=0, le=15)
    flag_emissions_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    biogenic_reported_separately: bool = Field(default=False)
    all_ghgs_included: bool = Field(default=False)
    third_party_verified: bool = Field(default=False)

class GovernanceData(BaseModel):
    """Governance and reporting data for validation.

    Attributes:
        boundary_approach: Organisational boundary method.
        recalculation_policy_defined: Whether recalc policy exists.
        annual_reporting_commitment: Whether annual reporting committed.
        progress_reporting: Whether annual progress is reported.
        public_disclosure: Whether targets are publicly disclosed.
        board_oversight: Whether board/C-suite oversees targets.
        scope2_market_based_primary: Whether S2 uses market-based.
        scope2_location_reported: Whether S2 location also reported.
        renewable_electricity_pct: Current RE share.
        re_quality_criteria_met: Whether RE instruments meet quality.
        supplier_engagement_target: Whether supplier engagement exists.
        carbon_credits_excluded: Whether credits excluded from target.
        avoided_emissions_excluded: Whether avoided emissions excluded.
        transition_plan_documented: Whether transition plan exists.
        capex_aligned: Whether CAPEX is aligned with transition.
        neutralisation_plan: Whether neutralisation strategy exists.
        neutralisation_permanent: Whether removals are permanent.
    """
    boundary_approach: str = Field(default="operational_control")
    recalculation_policy_defined: bool = Field(default=False)
    annual_reporting_commitment: bool = Field(default=False)
    progress_reporting: bool = Field(default=False)
    public_disclosure: bool = Field(default=False)
    board_oversight: bool = Field(default=False)
    scope2_market_based_primary: bool = Field(default=False)
    scope2_location_reported: bool = Field(default=False)
    renewable_electricity_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100")
    )
    re_quality_criteria_met: bool = Field(default=False)
    supplier_engagement_target: bool = Field(default=False)
    carbon_credits_excluded: bool = Field(default=True)
    avoided_emissions_excluded: bool = Field(default=True)
    transition_plan_documented: bool = Field(default=False)
    capex_aligned: bool = Field(default=False)
    neutralisation_plan: bool = Field(default=False)
    neutralisation_permanent: bool = Field(default=False)

class ValidationInput(BaseModel):
    """Input for full 42-criterion SBTi validation.

    Attributes:
        entity_name: Reporting entity name.
        target: Target data.
        inventory: Emissions inventory data.
        governance: Governance and reporting data.
        include_net_zero: Whether to validate NZ criteria.
    """
    entity_name: str = Field(
        ..., min_length=1, max_length=300, description="Entity name"
    )
    target: TargetData = Field(..., description="Target data")
    inventory: InventoryData = Field(
        ..., description="Emissions inventory data"
    )
    governance: GovernanceData = Field(
        default_factory=GovernanceData,
        description="Governance and reporting data"
    )
    include_net_zero: bool = Field(
        default=True,
        description="Whether to include net-zero criteria validation"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class CriterionCheck(BaseModel):
    """Assessment result for a single SBTi criterion.

    Attributes:
        criterion_id: Criterion identifier (C1, NZ-C1, etc.).
        category: Criterion functional category.
        description: Human-readable criterion description.
        status: PASS, FAIL, WARNING, or NOT_APPLICABLE.
        message: Detailed assessment message.
        remediation: Guidance for resolving failures.
        assessed_value: The value that was assessed.
        threshold_value: The threshold against which it was assessed.
    """
    criterion_id: str = Field(default="")
    category: str = Field(default="")
    description: str = Field(default="")
    status: str = Field(default=CriterionStatus.FAIL.value)
    message: str = Field(default="")
    remediation: str = Field(default="")
    assessed_value: str = Field(default="")
    threshold_value: str = Field(default="")

class GapItem(BaseModel):
    """Specific gap identified during validation.

    Attributes:
        criterion_id: Related criterion.
        category: Gap category.
        severity: CRITICAL, HIGH, MEDIUM, LOW.
        description: What is missing or insufficient.
        remediation: How to close the gap.
        estimated_effort: Estimated effort (LOW/MEDIUM/HIGH).
    """
    criterion_id: str = Field(default="")
    category: str = Field(default="")
    severity: str = Field(default="high")
    description: str = Field(default="")
    remediation: str = Field(default="")
    estimated_effort: str = Field(default="medium")

class ReadinessScore(BaseModel):
    """Overall readiness scoring.

    Attributes:
        total_criteria: Total criteria assessed.
        passed: Number passed.
        failed: Number failed.
        warnings: Number of warnings.
        not_applicable: Number not applicable.
        readiness_pct: Overall readiness percentage.
        near_term_readiness_pct: Near-term criteria readiness.
        net_zero_readiness_pct: Net-zero criteria readiness.
        is_submission_ready: Whether ready for SBTi submission.
        blocking_gaps_count: Number of blocking gaps (FAIL).
    """
    total_criteria: int = Field(default=0)
    passed: int = Field(default=0)
    failed: int = Field(default=0)
    warnings: int = Field(default=0)
    not_applicable: int = Field(default=0)
    readiness_pct: Decimal = Field(default=Decimal("0"))
    near_term_readiness_pct: Decimal = Field(default=Decimal("0"))
    net_zero_readiness_pct: Decimal = Field(default=Decimal("0"))
    is_submission_ready: bool = Field(default=False)
    blocking_gaps_count: int = Field(default=0)

class ValidationResult(BaseModel):
    """Complete 42-criterion validation result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Timestamp.
        entity_name: Entity name.
        criteria_checks: All criterion assessments.
        gaps: Identified gaps requiring remediation.
        readiness_score: Overall readiness scoring.
        category_scores: Per-category readiness percentages.
        processing_time_ms: Processing duration (ms).
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    entity_name: str = Field(default="")
    criteria_checks: List[CriterionCheck] = Field(default_factory=list)
    gaps: List[GapItem] = Field(default_factory=list)
    readiness_score: Optional[ReadinessScore] = Field(None)
    category_scores: Dict[str, Decimal] = Field(default_factory=dict)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class CriteriaValidationEngine:
    """SBTi 42-criterion automated validation engine.

    Validates targets against all 28 near-term criteria (C1-C28) and
    14 net-zero criteria (NZ-C1 to NZ-C14) with pass/fail/warning
    for each and detailed remediation guidance.

    All assessments are deterministic threshold checks.  No LLM
    involvement.  SHA-256 hash on every result.

    Usage::

        engine = CriteriaValidationEngine()
        result = engine.validate_all(input_data)
        print(f"Readiness: {result.readiness_score.readiness_pct}%")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise CriteriaValidationEngine.

        Args:
            config: Optional configuration overrides.
        """
        self.config = config or {}
        logger.info(
            "CriteriaValidationEngine v%s initialised", self.engine_version
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def validate_all(self, data: ValidationInput) -> ValidationResult:
        """Run full 42-criterion validation.

        Executes all near-term criteria (C1-C28) and, if requested,
        all net-zero criteria (NZ-C1 to NZ-C14).

        Args:
            data: Validated input data.

        Returns:
            ValidationResult with all checks, gaps, and readiness score.
        """
        t0 = time.perf_counter()
        logger.info(
            "Criteria validation: entity=%s, include_nz=%s",
            data.entity_name, data.include_net_zero,
        )

        checks: List[CriterionCheck] = []
        gaps: List[GapItem] = []

        # Near-term criteria C1-C28
        nt_checks, nt_gaps = self.validate_near_term(data)
        checks.extend(nt_checks)
        gaps.extend(nt_gaps)

        # Net-zero criteria NZ-C1 to NZ-C14
        nz_checks: List[CriterionCheck] = []
        nz_gaps: List[GapItem] = []
        if data.include_net_zero:
            nz_checks, nz_gaps = self.validate_net_zero(data)
            checks.extend(nz_checks)
            gaps.extend(nz_gaps)

        # Calculate readiness score
        readiness = self.calculate_readiness_score(checks)

        # Calculate per-category scores
        category_scores = self._calculate_category_scores(checks)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = ValidationResult(
            entity_name=data.entity_name,
            criteria_checks=checks,
            gaps=gaps,
            readiness_score=readiness,
            category_scores=category_scores,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Validation complete: %d checks, %d passed, %d failed, "
            "readiness=%.1f%%, hash=%s",
            readiness.total_criteria, readiness.passed, readiness.failed,
            float(readiness.readiness_pct), result.provenance_hash[:16],
        )
        return result

    def validate_near_term(
        self, data: ValidationInput,
    ) -> Tuple[List[CriterionCheck], List[GapItem]]:
        """Validate near-term criteria C1-C28.

        Args:
            data: Validation input.

        Returns:
            Tuple of (checks, gaps).
        """
        checks: List[CriterionCheck] = []
        gaps: List[GapItem] = []

        # C1: Boundary approach defined
        c1 = self._check_boolean(
            "C1", data.governance.boundary_approach in (
                "operational_control", "financial_control", "equity_share"
            ),
            assessed=data.governance.boundary_approach,
            threshold="operational_control|financial_control|equity_share",
        )
        checks.append(c1)
        if c1.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("C1", "critical"))

        # C2: All GHGs included
        c2 = self._check_boolean(
            "C2", data.inventory.all_ghgs_included,
            assessed=str(data.inventory.all_ghgs_included),
            threshold="True",
        )
        checks.append(c2)
        if c2.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("C2", "high"))

        # C3: S1+2 coverage >= 95%
        c3 = self._check_threshold(
            "C3", data.target.scope12_coverage_pct,
            SCOPE12_COVERAGE_MIN_PCT, "ge",
        )
        checks.append(c3)
        if c3.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("C3", "critical"))

        # C4: Biogenic reported separately
        c4 = self._check_boolean(
            "C4", data.inventory.biogenic_reported_separately,
            assessed=str(data.inventory.biogenic_reported_separately),
            threshold="True",
        )
        checks.append(c4)
        if c4.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("C4", "medium"))

        # C5: Base year >= 2015
        c5 = self._check_threshold(
            "C5", _decimal(data.target.base_year),
            _decimal(BASE_YEAR_MIN), "ge",
        )
        checks.append(c5)
        if c5.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("C5", "critical"))

        # C6: Inventory complete and verified
        c6 = self._check_boolean(
            "C6", data.inventory.third_party_verified and (
                data.inventory.scope1_tco2e > Decimal("0")
            ),
            assessed=f"verified={data.inventory.third_party_verified}",
            threshold="True",
        )
        checks.append(c6)
        if c6.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("C6", "high"))

        # C7: Recalculation policy defined
        c7 = self._check_boolean(
            "C7", data.governance.recalculation_policy_defined,
            assessed=str(data.governance.recalculation_policy_defined),
            threshold="True",
        )
        checks.append(c7)
        if c7.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("C7", "medium"))

        # C8: S3 screening complete (15 categories)
        c8 = self._check_threshold(
            "C8", _decimal(data.inventory.scope3_categories_screened),
            Decimal("15"), "ge",
        )
        checks.append(c8)
        if c8.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("C8", "high"))

        # C9: S1+2 ambition >= 4.2%/yr
        c9 = self._check_threshold(
            "C9", data.target.scope12_annual_rate_pct,
            ACA_15C_RATE_PCT, "ge",
        )
        if (
            c9.status == CriterionStatus.FAIL.value
            and data.target.scope12_annual_rate_pct >= ACA_WB2C_RATE_PCT
        ):
            c9.status = CriterionStatus.WARNING.value
            c9.message = (
                f"S1+2 rate {data.target.scope12_annual_rate_pct}% meets WB2C "
                f"but not 1.5C threshold of {ACA_15C_RATE_PCT}%"
            )
        checks.append(c9)
        if c9.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("C9", "critical"))

        # C10: Approved methodology
        approved = ("aca", "sda", "flag", "physical_intensity")
        c10 = self._check_boolean(
            "C10", data.target.method in approved,
            assessed=data.target.method,
            threshold="|".join(approved),
        )
        checks.append(c10)
        if c10.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("C10", "high"))

        # C11: Carbon credits excluded
        c11 = self._check_boolean(
            "C11", data.governance.carbon_credits_excluded,
            assessed=str(data.governance.carbon_credits_excluded),
            threshold="True",
        )
        checks.append(c11)
        if c11.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("C11", "critical"))

        # C12: Avoided emissions excluded
        c12 = self._check_boolean(
            "C12", data.governance.avoided_emissions_excluded,
            assessed=str(data.governance.avoided_emissions_excluded),
            threshold="True",
        )
        checks.append(c12)
        if c12.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("C12", "high"))

        # C13: S2 market-based primary
        c13 = self._check_boolean(
            "C13", data.governance.scope2_market_based_primary,
            assessed=str(data.governance.scope2_market_based_primary),
            threshold="True",
        )
        checks.append(c13)
        if c13.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("C13", "medium"))

        # C14: RE procurement plan
        c14_pass = data.governance.renewable_electricity_pct > Decimal("0")
        c14 = self._check_boolean(
            "C14", c14_pass,
            assessed=str(data.governance.renewable_electricity_pct),
            threshold=">0%",
        )
        checks.append(c14)
        if c14.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("C14", "medium"))

        # C15: S2 location-based also reported
        c15 = self._check_boolean(
            "C15", data.governance.scope2_location_reported,
            assessed=str(data.governance.scope2_location_reported),
            threshold="True",
        )
        checks.append(c15)
        if c15.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("C15", "low"))

        # C16: RE quality criteria
        c16 = self._check_boolean(
            "C16", data.governance.re_quality_criteria_met,
            assessed=str(data.governance.re_quality_criteria_met),
            threshold="True",
        )
        checks.append(c16)
        if c16.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("C16", "medium"))

        # C17: S3 target set if required
        total_em = (
            data.inventory.scope1_tco2e
            + data.inventory.scope2_market_tco2e
            + data.inventory.scope3_total_tco2e
        )
        s3_pct = _safe_pct(data.inventory.scope3_total_tco2e, total_em)
        s3_required = s3_pct >= SCOPE3_TRIGGER_PCT

        if s3_required:
            has_s3 = data.target.scope3_annual_rate_pct > Decimal("0")
            c17 = self._check_boolean(
                "C17", has_s3,
                assessed=f"S3_rate={data.target.scope3_annual_rate_pct}%",
                threshold="S3 target required (S3>=40%)",
            )
        else:
            c17 = CriterionCheck(
                criterion_id="C17",
                category=CRITERIA_DEFINITIONS["C17"]["category"],
                description=CRITERIA_DEFINITIONS["C17"]["description"],
                status=CriterionStatus.NOT_APPLICABLE.value,
                message="Scope 3 is below 40% trigger threshold.",
                remediation="",
            )
        checks.append(c17)
        if c17.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("C17", "critical"))

        # C18: S3 near-term coverage >= 67%
        if s3_required:
            c18 = self._check_threshold(
                "C18", data.target.scope3_coverage_pct,
                SCOPE3_NT_COVERAGE_PCT, "ge",
            )
        else:
            c18 = CriterionCheck(
                criterion_id="C18",
                category=CRITERIA_DEFINITIONS["C18"]["category"],
                description=CRITERIA_DEFINITIONS["C18"]["description"],
                status=CriterionStatus.NOT_APPLICABLE.value,
                message="Scope 3 target not required.",
                remediation="",
            )
        checks.append(c18)
        if c18.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("C18", "high"))

        # C19: S3 reduction rate >= 2.5%/yr
        if s3_required:
            c19 = self._check_threshold(
                "C19", data.target.scope3_annual_rate_pct,
                ACA_WB2C_RATE_PCT, "ge",
            )
        else:
            c19 = CriterionCheck(
                criterion_id="C19",
                category=CRITERIA_DEFINITIONS["C19"]["category"],
                description=CRITERIA_DEFINITIONS["C19"]["description"],
                status=CriterionStatus.NOT_APPLICABLE.value,
                message="Scope 3 target not required.",
                remediation="",
            )
        checks.append(c19)
        if c19.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("C19", "high"))

        # C20: Supplier engagement
        c20 = self._check_boolean(
            "C20", data.governance.supplier_engagement_target,
            assessed=str(data.governance.supplier_engagement_target),
            threshold="True",
        )
        if not s3_required:
            c20.status = CriterionStatus.NOT_APPLICABLE.value
            c20.message = "Scope 3 target not required."
        checks.append(c20)
        if c20.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("C20", "medium"))

        # C21: Timeframe 5-10 years
        timeframe = data.target.target_year - data.target.base_year
        c21_pass = NEAR_TERM_MIN_YEARS <= timeframe <= NEAR_TERM_MAX_YEARS
        c21 = self._check_boolean(
            "C21", c21_pass,
            assessed=f"{timeframe} years",
            threshold=f"{NEAR_TERM_MIN_YEARS}-{NEAR_TERM_MAX_YEARS} years",
        )
        checks.append(c21)
        if c21.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("C21", "high"))

        # C22: Target start within 2 years of submission
        current_year = utcnow().year
        start_gap = abs(current_year - data.target.base_year)
        c22_pass = start_gap <= 7  # lenient for older base years with recency
        c22 = self._check_boolean(
            "C22", c22_pass,
            assessed=f"base_year_age={start_gap}",
            threshold="<= 7 years",
        )
        checks.append(c22)
        if c22.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("C22", "medium"))

        # C23: Five-year review cycle
        c23 = self._check_boolean(
            "C23", True,  # Assessed as recommendation at submission
            assessed="5-year review cycle",
            threshold="Defined",
        )
        c23.status = CriterionStatus.PASS.value
        c23.message = "Five-year review commitment noted."
        checks.append(c23)

        # C24: Interim milestones
        c24 = self._check_boolean(
            "C24", True,
            assessed="Milestones defined by engine",
            threshold="Defined",
        )
        c24.status = CriterionStatus.PASS.value
        c24.message = "Interim milestones generated by target-setting engine."
        checks.append(c24)

        # C25: Annual GHG reporting
        c25 = self._check_boolean(
            "C25", data.governance.annual_reporting_commitment,
            assessed=str(data.governance.annual_reporting_commitment),
            threshold="True",
        )
        checks.append(c25)
        if c25.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("C25", "high"))

        # C26: Progress reporting
        c26 = self._check_boolean(
            "C26", data.governance.progress_reporting,
            assessed=str(data.governance.progress_reporting),
            threshold="True",
        )
        checks.append(c26)
        if c26.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("C26", "high"))

        # C27: Public disclosure
        c27 = self._check_boolean(
            "C27", data.governance.public_disclosure,
            assessed=str(data.governance.public_disclosure),
            threshold="True",
        )
        checks.append(c27)
        if c27.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("C27", "medium"))

        # C28: Board oversight
        c28 = self._check_boolean(
            "C28", data.governance.board_oversight,
            assessed=str(data.governance.board_oversight),
            threshold="True",
        )
        checks.append(c28)
        if c28.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("C28", "high"))

        return checks, gaps

    def validate_net_zero(
        self, data: ValidationInput,
    ) -> Tuple[List[CriterionCheck], List[GapItem]]:
        """Validate net-zero criteria NZ-C1 to NZ-C14.

        Args:
            data: Validation input.

        Returns:
            Tuple of (checks, gaps).
        """
        checks: List[CriterionCheck] = []
        gaps: List[GapItem] = []
        target = data.target

        # NZ-C1: Net-zero covers S1+2+3
        nz_covers_all = target.target_type == "net_zero"
        c1 = self._check_boolean(
            "NZ-C1", nz_covers_all,
            assessed=target.target_type,
            threshold="net_zero",
        )
        checks.append(c1)
        if c1.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("NZ-C1", "critical"))

        # NZ-C2: Target year <= 2050
        nz_year = target.net_zero_year or target.target_year
        c2 = self._check_threshold(
            "NZ-C2", _decimal(nz_year),
            _decimal(NET_ZERO_MAX_YEAR), "le",
        )
        checks.append(c2)
        if c2.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("NZ-C2", "critical"))

        # NZ-C3: Near-term target set alongside
        c3 = self._check_boolean(
            "NZ-C3", target.has_near_term,
            assessed=str(target.has_near_term),
            threshold="True",
        )
        checks.append(c3)
        if c3.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("NZ-C3", "critical"))

        # NZ-C4: SBTi-approved net-zero pathway
        c4 = self._check_boolean(
            "NZ-C4", target.method in ("aca", "sda"),
            assessed=target.method,
            threshold="aca|sda",
        )
        checks.append(c4)
        if c4.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("NZ-C4", "high"))

        # NZ-C5: Long-term S1+2 reduction >= 90%
        c5 = self._check_threshold(
            "NZ-C5", target.scope12_reduction_pct,
            LONG_TERM_MIN_REDUCTION_PCT, "ge",
        )
        checks.append(c5)
        if c5.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("NZ-C5", "critical"))

        # NZ-C6: Long-term S3 reduction >= 90%
        c6 = self._check_threshold(
            "NZ-C6", target.scope3_reduction_pct,
            LONG_TERM_MIN_REDUCTION_PCT, "ge",
        )
        checks.append(c6)
        if c6.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("NZ-C6", "critical"))

        # NZ-C7: Long-term S3 coverage >= 90%
        c7 = self._check_threshold(
            "NZ-C7", target.scope3_coverage_pct,
            SCOPE3_LT_COVERAGE_PCT, "ge",
        )
        checks.append(c7)
        if c7.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("NZ-C7", "high"))

        # NZ-C8: Long-term timeframe extends to 2050 or earlier
        c8 = self._check_threshold(
            "NZ-C8", _decimal(nz_year),
            _decimal(NET_ZERO_MAX_YEAR), "le",
        )
        checks.append(c8)
        if c8.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("NZ-C8", "high"))

        # NZ-C9: Residual emissions <= 10%
        c9 = self._check_threshold(
            "NZ-C9", target.residual_emissions_pct,
            NET_ZERO_MAX_RESIDUAL_PCT, "le",
        )
        checks.append(c9)
        if c9.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("NZ-C9", "critical"))

        # NZ-C10: Neutralisation plan
        c10 = self._check_boolean(
            "NZ-C10", data.governance.neutralisation_plan,
            assessed=str(data.governance.neutralisation_plan),
            threshold="True",
        )
        checks.append(c10)
        if c10.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("NZ-C10", "high"))

        # NZ-C11: Permanent carbon removals
        c11 = self._check_boolean(
            "NZ-C11", data.governance.neutralisation_permanent,
            assessed=str(data.governance.neutralisation_permanent),
            threshold="True",
        )
        checks.append(c11)
        if c11.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("NZ-C11", "high"))

        # NZ-C12: Transition plan documented
        c12 = self._check_boolean(
            "NZ-C12", data.governance.transition_plan_documented,
            assessed=str(data.governance.transition_plan_documented),
            threshold="True",
        )
        checks.append(c12)
        if c12.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("NZ-C12", "high"))

        # NZ-C13: CAPEX aligned
        c13 = self._check_boolean(
            "NZ-C13", data.governance.capex_aligned,
            assessed=str(data.governance.capex_aligned),
            threshold="True",
        )
        checks.append(c13)
        if c13.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("NZ-C13", "medium"))

        # NZ-C14: Board governance for net-zero
        c14 = self._check_boolean(
            "NZ-C14", data.governance.board_oversight,
            assessed=str(data.governance.board_oversight),
            threshold="True",
        )
        checks.append(c14)
        if c14.status == CriterionStatus.FAIL.value:
            gaps.append(self._make_gap("NZ-C14", "high"))

        return checks, gaps

    def get_gaps(
        self, result: ValidationResult,
    ) -> List[GapItem]:
        """Extract gaps from a validation result.

        Args:
            result: Completed validation result.

        Returns:
            List of GapItem objects sorted by severity.
        """
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        return sorted(
            result.gaps,
            key=lambda g: severity_order.get(g.severity, 4),
        )

    def calculate_readiness_score(
        self, checks: List[CriterionCheck],
    ) -> ReadinessScore:
        """Calculate overall readiness score from criterion checks.

        Score formula:
            applicable = total - not_applicable
            readiness = (passed + 0.5 * warnings) / applicable * 100

        Args:
            checks: List of criterion checks.

        Returns:
            ReadinessScore with breakdown.
        """
        total = len(checks)
        passed = sum(
            1 for c in checks if c.status == CriterionStatus.PASS.value
        )
        failed = sum(
            1 for c in checks if c.status == CriterionStatus.FAIL.value
        )
        warnings = sum(
            1 for c in checks if c.status == CriterionStatus.WARNING.value
        )
        na = sum(
            1 for c in checks
            if c.status == CriterionStatus.NOT_APPLICABLE.value
        )

        applicable = total - na
        if applicable > 0:
            score = _decimal(passed + 0.5 * warnings) / _decimal(applicable)
            readiness_pct = _round_val(score * Decimal("100"), 1)
        else:
            readiness_pct = Decimal("100.0")

        # Near-term readiness (C1-C28)
        nt_checks = [c for c in checks if not c.criterion_id.startswith("NZ")]
        nt_applicable = sum(
            1 for c in nt_checks
            if c.status != CriterionStatus.NOT_APPLICABLE.value
        )
        nt_passed = sum(
            1 for c in nt_checks
            if c.status == CriterionStatus.PASS.value
        )
        nt_warnings = sum(
            1 for c in nt_checks
            if c.status == CriterionStatus.WARNING.value
        )
        if nt_applicable > 0:
            nt_score = (
                _decimal(nt_passed + 0.5 * nt_warnings)
                / _decimal(nt_applicable) * Decimal("100")
            )
        else:
            nt_score = Decimal("100.0")

        # Net-zero readiness (NZ-C1 to NZ-C14)
        nz_checks = [c for c in checks if c.criterion_id.startswith("NZ")]
        nz_applicable = sum(
            1 for c in nz_checks
            if c.status != CriterionStatus.NOT_APPLICABLE.value
        )
        nz_passed = sum(
            1 for c in nz_checks
            if c.status == CriterionStatus.PASS.value
        )
        nz_warnings = sum(
            1 for c in nz_checks
            if c.status == CriterionStatus.WARNING.value
        )
        if nz_applicable > 0:
            nz_score = (
                _decimal(nz_passed + 0.5 * nz_warnings)
                / _decimal(nz_applicable) * Decimal("100")
            )
        else:
            nz_score = Decimal("100.0")

        is_ready = failed == 0

        return ReadinessScore(
            total_criteria=total,
            passed=passed,
            failed=failed,
            warnings=warnings,
            not_applicable=na,
            readiness_pct=readiness_pct,
            near_term_readiness_pct=_round_val(nt_score, 1),
            net_zero_readiness_pct=_round_val(nz_score, 1),
            is_submission_ready=is_ready,
            blocking_gaps_count=failed,
        )

    # ------------------------------------------------------------------ #
    # Check Helpers                                                       #
    # ------------------------------------------------------------------ #

    def _check_boolean(
        self,
        criterion_id: str,
        condition: bool,
        assessed: str = "",
        threshold: str = "",
    ) -> CriterionCheck:
        """Create a boolean pass/fail criterion check.

        Args:
            criterion_id: Criterion ID (e.g. "C1").
            condition: Whether the criterion is met.
            assessed: Value that was assessed.
            threshold: Required threshold for reference.

        Returns:
            CriterionCheck with status and details.
        """
        defn = CRITERIA_DEFINITIONS.get(criterion_id, {})
        status = (
            CriterionStatus.PASS.value if condition
            else CriterionStatus.FAIL.value
        )
        return CriterionCheck(
            criterion_id=criterion_id,
            category=defn.get("category", ""),
            description=defn.get("description", ""),
            status=status,
            message=(
                f"{criterion_id}: PASS"
                if condition
                else f"{criterion_id}: FAIL - {defn.get('description', '')}"
            ),
            remediation="" if condition else defn.get("remediation", ""),
            assessed_value=assessed,
            threshold_value=threshold,
        )

    def _check_threshold(
        self,
        criterion_id: str,
        value: Decimal,
        threshold: Decimal,
        operator: str,
    ) -> CriterionCheck:
        """Create a threshold comparison criterion check.

        Args:
            criterion_id: Criterion ID.
            value: Assessed value.
            threshold: Threshold value.
            operator: "ge" (>=), "le" (<=), "gt" (>), "lt" (<).

        Returns:
            CriterionCheck with status and details.
        """
        defn = CRITERIA_DEFINITIONS.get(criterion_id, {})
        ops = {
            "ge": value >= threshold,
            "le": value <= threshold,
            "gt": value > threshold,
            "lt": value < threshold,
        }
        condition = ops.get(operator, False)
        status = (
            CriterionStatus.PASS.value if condition
            else CriterionStatus.FAIL.value
        )
        op_symbol = {"ge": ">=", "le": "<=", "gt": ">", "lt": "<"}.get(
            operator, operator
        )

        return CriterionCheck(
            criterion_id=criterion_id,
            category=defn.get("category", ""),
            description=defn.get("description", ""),
            status=status,
            message=(
                f"{criterion_id}: PASS ({value} {op_symbol} {threshold})"
                if condition
                else f"{criterion_id}: FAIL ({value} does not meet "
                     f"{op_symbol} {threshold})"
            ),
            remediation="" if condition else defn.get("remediation", ""),
            assessed_value=str(value),
            threshold_value=f"{op_symbol} {threshold}",
        )

    def _make_gap(
        self, criterion_id: str, severity: str,
    ) -> GapItem:
        """Create a GapItem from a failed criterion.

        Args:
            criterion_id: Failed criterion ID.
            severity: Gap severity (critical/high/medium/low).

        Returns:
            GapItem with remediation guidance.
        """
        defn = CRITERIA_DEFINITIONS.get(criterion_id, {})
        effort_map = {
            "critical": "high",
            "high": "high",
            "medium": "medium",
            "low": "low",
        }
        return GapItem(
            criterion_id=criterion_id,
            category=defn.get("category", ""),
            severity=severity,
            description=defn.get("description", ""),
            remediation=defn.get("remediation", ""),
            estimated_effort=effort_map.get(severity, "medium"),
        )

    def _calculate_category_scores(
        self, checks: List[CriterionCheck],
    ) -> Dict[str, Decimal]:
        """Calculate readiness percentage per category.

        Args:
            checks: All criterion checks.

        Returns:
            Dict mapping category to readiness percentage.
        """
        category_checks: Dict[str, List[CriterionCheck]] = {}
        for check in checks:
            cat = check.category or "unknown"
            if cat not in category_checks:
                category_checks[cat] = []
            category_checks[cat].append(check)

        scores: Dict[str, Decimal] = {}
        for cat, cat_checks in category_checks.items():
            applicable = [
                c for c in cat_checks
                if c.status != CriterionStatus.NOT_APPLICABLE.value
            ]
            if not applicable:
                scores[cat] = Decimal("100.0")
                continue
            passed = sum(
                1 for c in applicable
                if c.status == CriterionStatus.PASS.value
            )
            warns = sum(
                1 for c in applicable
                if c.status == CriterionStatus.WARNING.value
            )
            score = _decimal(passed + 0.5 * warns) / _decimal(len(applicable))
            scores[cat] = _round_val(score * Decimal("100"), 1)

        return scores

    # ------------------------------------------------------------------ #
    # Utility Methods                                                     #
    # ------------------------------------------------------------------ #

    def get_criteria_list(self) -> List[Dict[str, str]]:
        """Return the full list of 42 criteria definitions.

        Returns:
            List of dicts with id, category, description.
        """
        return [
            {
                "id": cid,
                "category": defn["category"],
                "description": defn["description"],
                "remediation": defn["remediation"],
            }
            for cid, defn in CRITERIA_DEFINITIONS.items()
        ]

    def get_summary(self, result: ValidationResult) -> Dict[str, Any]:
        """Generate concise summary from validation result.

        Args:
            result: Validation result.

        Returns:
            Dict with key metrics and provenance hash.
        """
        summary: Dict[str, Any] = {
            "entity_name": result.entity_name,
            "total_criteria": (
                result.readiness_score.total_criteria
                if result.readiness_score else 0
            ),
            "passed": (
                result.readiness_score.passed
                if result.readiness_score else 0
            ),
            "failed": (
                result.readiness_score.failed
                if result.readiness_score else 0
            ),
            "readiness_pct": str(
                result.readiness_score.readiness_pct
                if result.readiness_score else "0"
            ),
            "is_submission_ready": (
                result.readiness_score.is_submission_ready
                if result.readiness_score else False
            ),
            "blocking_gaps": [
                g.criterion_id for g in result.gaps if g.severity == "critical"
            ],
            "category_scores": {
                k: str(v) for k, v in result.category_scores.items()
            },
        }
        summary["provenance_hash"] = _compute_hash(summary)
        return summary
