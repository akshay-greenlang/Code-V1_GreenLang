# -*- coding: utf-8 -*-
"""
GoodGovernanceEngine - PACK-010 SFDR Article 8 Engine 4
=========================================================

Verifies that investee companies meet the good governance requirements of
SFDR Article 2(17) as a prerequisite for qualifying as a "sustainable
investment."

Article 2(17) requires that investee companies follow good governance
practices, in particular with respect to:
    1. Sound management structures
    2. Employee relations
    3. Remuneration of staff
    4. Tax compliance

This engine also checks additional governance standards including adherence
to the UN Global Compact, anti-corruption policies, and anti-bribery
measures.

Key Regulatory References:
    - Regulation (EU) 2019/2088 (SFDR) Article 2(17)
    - Delegated Regulation (EU) 2022/1288 (SFDR RTS) Recital 33
    - UN Global Compact Ten Principles
    - OECD Guidelines for Multinational Enterprises

Zero-Hallucination:
    - All scoring uses deterministic arithmetic (no LLM in scoring path)
    - Configurable weights and pass/fail thresholds
    - SHA-256 provenance hashing on every result
    - No LLM involvement in any numeric or scoring path

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-010 SFDR Article 8
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
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
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _round_val(value: float, places: int = 2) -> float:
    """Round a float to specified decimal places."""
    return round(value, places)


def _clamp(value: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """Clamp a value to the given range."""
    return max(min_val, min(max_val, value))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class GovernanceArea(str, Enum):
    """The four Article 2(17) good governance assessment areas."""

    MANAGEMENT_STRUCTURES = "management_structures"
    EMPLOYEE_RELATIONS = "employee_relations"
    REMUNERATION = "remuneration"
    TAX_COMPLIANCE = "tax_compliance"


class GovernanceCheckType(str, Enum):
    """Type of governance check."""

    BOOLEAN = "BOOLEAN"             # Yes/No flag
    THRESHOLD_MIN = "THRESHOLD_MIN"  # Fail if below threshold
    THRESHOLD_MAX = "THRESHOLD_MAX"  # Fail if above threshold
    SCORE = "SCORE"                 # 0-100 score input


class GovernanceStatus(str, Enum):
    """Governance assessment outcome."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ViolationType(str, Enum):
    """Types of governance violations."""

    UNGC_VIOLATION = "ungc_violation"
    OECD_VIOLATION = "oecd_violation"
    CORRUPTION = "corruption"
    BRIBERY = "bribery"
    TAX_EVASION = "tax_evasion"
    LABOR_RIGHTS = "labor_rights"
    HEALTH_SAFETY = "health_safety"
    BOARD_INDEPENDENCE = "board_independence"
    REMUNERATION_EXCESS = "remuneration_excess"
    OTHER = "other"


AREA_NAMES: Dict[str, str] = {
    GovernanceArea.MANAGEMENT_STRUCTURES.value: "Sound Management Structures",
    GovernanceArea.EMPLOYEE_RELATIONS.value: "Employee Relations",
    GovernanceArea.REMUNERATION.value: "Remuneration of Staff",
    GovernanceArea.TAX_COMPLIANCE.value: "Tax Compliance",
}


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class GovernanceCriterion(BaseModel):
    """A single governance assessment criterion.

    Defines one check within a governance area, including its type,
    threshold, weight in the composite score, and whether it is mandatory.

    Attributes:
        criterion_id: Unique criterion identifier.
        area: Which governance area this criterion belongs to.
        name: Human-readable criterion name.
        description: Detailed description of the check.
        check_type: Type of check (BOOLEAN, THRESHOLD_MIN, etc.).
        metric_key: Key in the input data dictionary.
        threshold_value: Numeric threshold for THRESHOLD checks.
        weight: Weight in the area composite score (0-1).
        is_mandatory: If True, failure causes automatic area FAIL.
        max_score: Maximum points this criterion contributes.
    """

    criterion_id: str = Field(
        ..., min_length=1, description="Unique criterion identifier",
    )
    area: GovernanceArea = Field(
        ..., description="Governance area",
    )
    name: str = Field(
        ..., description="Criterion name",
    )
    description: str = Field(
        default="", description="Detailed description",
    )
    check_type: GovernanceCheckType = Field(
        ..., description="Type of governance check",
    )
    metric_key: str = Field(
        ..., description="Key in the input data dictionary",
    )
    threshold_value: Optional[float] = Field(
        None, description="Threshold for pass/fail boundary",
    )
    weight: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Weight in area composite score (0-1)",
    )
    is_mandatory: bool = Field(
        default=False,
        description="Mandatory criterion (failure = area FAIL)",
    )
    max_score: float = Field(
        default=100.0, ge=0, description="Maximum score contribution",
    )


class GovernanceConfig(BaseModel):
    """Configuration for the Good Governance Engine.

    Attributes:
        pass_threshold: Minimum composite score for overall PASS (0-100).
        area_pass_threshold: Minimum score per area for area PASS (0-100).
        area_weights: Weights for each governance area in composite score.
        criteria: List of governance criteria to evaluate.
        ungc_adherence_required: Whether UNGC adherence is a hard requirement.
        anti_corruption_required: Whether anti-corruption policy is required.
        anti_bribery_required: Whether anti-bribery measures are required.
    """

    pass_threshold: float = Field(
        default=60.0, ge=0.0, le=100.0,
        description="Minimum composite score for PASS",
    )
    area_pass_threshold: float = Field(
        default=40.0, ge=0.0, le=100.0,
        description="Minimum per-area score for area PASS",
    )
    area_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            GovernanceArea.MANAGEMENT_STRUCTURES.value: 0.30,
            GovernanceArea.EMPLOYEE_RELATIONS.value: 0.25,
            GovernanceArea.REMUNERATION.value: 0.20,
            GovernanceArea.TAX_COMPLIANCE.value: 0.25,
        },
        description="Area weights for composite score (must sum to 1.0)",
    )
    criteria: List[GovernanceCriterion] = Field(
        default_factory=list,
        description="Governance criteria to evaluate",
    )
    ungc_adherence_required: bool = Field(
        default=True,
        description="UNGC adherence is a hard requirement for PASS",
    )
    anti_corruption_required: bool = Field(
        default=True,
        description="Anti-corruption policy is required for PASS",
    )
    anti_bribery_required: bool = Field(
        default=True,
        description="Anti-bribery measures are required for PASS",
    )

    @model_validator(mode="after")
    def validate_weights(self) -> "GovernanceConfig":
        """Validate that area weights sum approximately to 1.0."""
        total = sum(self.area_weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"area_weights must sum to 1.0, got {total}"
            )
        return self


class ManagementStructureData(BaseModel):
    """Data for sound management structures assessment.

    Attributes:
        has_independent_board: Whether the board has independent members.
        independent_board_pct: Percentage of independent board members.
        has_audit_committee: Whether an audit committee exists.
        has_risk_committee: Whether a risk committee exists.
        has_nomination_committee: Whether a nomination committee exists.
        has_sustainability_committee: Whether sustainability committee exists.
        ceo_chair_separation: Whether CEO and board chair are separate.
        board_size: Total number of board members.
        board_meetings_per_year: Number of board meetings per year.
        has_whistleblower_mechanism: Whether a whistleblower mechanism exists.
    """

    has_independent_board: Optional[bool] = Field(None)
    independent_board_pct: Optional[float] = Field(None, ge=0.0, le=100.0)
    has_audit_committee: Optional[bool] = Field(None)
    has_risk_committee: Optional[bool] = Field(None)
    has_nomination_committee: Optional[bool] = Field(None)
    has_sustainability_committee: Optional[bool] = Field(None)
    ceo_chair_separation: Optional[bool] = Field(None)
    board_size: Optional[int] = Field(None, ge=0)
    board_meetings_per_year: Optional[int] = Field(None, ge=0)
    has_whistleblower_mechanism: Optional[bool] = Field(None)


class EmployeeRelationsData(BaseModel):
    """Data for employee relations assessment.

    Attributes:
        ilo_core_conventions_compliance: Adherence to ILO core conventions.
        has_health_safety_policy: Formal H&S policy exists.
        lost_time_injury_rate: Lost time injury frequency rate.
        employee_turnover_pct: Annual employee turnover rate.
        has_training_programs: Formal training/development programs.
        training_hours_per_employee: Average training hours per employee per year.
        has_diversity_policy: Formal diversity and inclusion policy.
        has_collective_bargaining: Collective bargaining agreements.
        living_wage_compliance: Whether company pays living wage.
        has_grievance_mechanism: Employee grievance mechanism exists.
    """

    ilo_core_conventions_compliance: Optional[bool] = Field(None)
    has_health_safety_policy: Optional[bool] = Field(None)
    lost_time_injury_rate: Optional[float] = Field(None, ge=0.0)
    employee_turnover_pct: Optional[float] = Field(None, ge=0.0, le=100.0)
    has_training_programs: Optional[bool] = Field(None)
    training_hours_per_employee: Optional[float] = Field(None, ge=0.0)
    has_diversity_policy: Optional[bool] = Field(None)
    has_collective_bargaining: Optional[bool] = Field(None)
    living_wage_compliance: Optional[bool] = Field(None)
    has_grievance_mechanism: Optional[bool] = Field(None)


class RemunerationData(BaseModel):
    """Data for remuneration assessment.

    Attributes:
        has_remuneration_policy: Formal remuneration policy exists.
        remuneration_policy_disclosed: Whether policy is publicly disclosed.
        ceo_to_median_pay_ratio: CEO pay to median employee pay ratio.
        has_clawback_provisions: Clawback provisions in executive contracts.
        performance_linked_pay_pct: Percentage of exec pay linked to performance.
        esg_linked_remuneration: Whether ESG metrics in executive compensation.
        shareholder_vote_on_pay: Whether say-on-pay vote exists.
        excessive_severance_provisions: Whether excessive golden parachutes exist.
    """

    has_remuneration_policy: Optional[bool] = Field(None)
    remuneration_policy_disclosed: Optional[bool] = Field(None)
    ceo_to_median_pay_ratio: Optional[float] = Field(None, ge=0.0)
    has_clawback_provisions: Optional[bool] = Field(None)
    performance_linked_pay_pct: Optional[float] = Field(None, ge=0.0, le=100.0)
    esg_linked_remuneration: Optional[bool] = Field(None)
    shareholder_vote_on_pay: Optional[bool] = Field(None)
    excessive_severance_provisions: Optional[bool] = Field(None)


class TaxComplianceData(BaseModel):
    """Data for tax compliance assessment.

    Attributes:
        has_tax_strategy_disclosure: Public tax strategy disclosed.
        country_by_country_reporting: CBCR in place.
        aggressive_tax_planning_flag: Whether flagged for aggressive tax planning.
        tax_haven_exposure: Whether company has significant tax haven operations.
        tax_controversies: Number of ongoing tax controversies.
        effective_tax_rate: Company's effective tax rate (%).
        tax_transparency_score: Tax transparency rating (0-100).
    """

    has_tax_strategy_disclosure: Optional[bool] = Field(None)
    country_by_country_reporting: Optional[bool] = Field(None)
    aggressive_tax_planning_flag: Optional[bool] = Field(None)
    tax_haven_exposure: Optional[bool] = Field(None)
    tax_controversies: Optional[int] = Field(None, ge=0)
    effective_tax_rate: Optional[float] = Field(None, ge=0.0, le=100.0)
    tax_transparency_score: Optional[float] = Field(None, ge=0.0, le=100.0)


class CompanyGovernanceData(BaseModel):
    """Complete governance data for a single company.

    Combines all four assessment areas plus additional governance checks
    (UNGC, anti-corruption, anti-bribery) into a single input model.

    Attributes:
        company_id: Unique company identifier.
        company_name: Company name.
        management_data: Sound management structures data.
        employee_data: Employee relations data.
        remuneration_data: Remuneration data.
        tax_data: Tax compliance data.
        ungc_signatory: Whether company is a UNGC signatory.
        ungc_violations: Whether company has UNGC violations.
        oecd_violations: Whether company has OECD guideline violations.
        has_anti_corruption_policy: Anti-corruption policy exists.
        has_anti_bribery_measures: Anti-bribery measures in place.
        corruption_controversies: Number of corruption-related controversies.
    """

    company_id: str = Field(
        ..., min_length=1, description="Unique company identifier",
    )
    company_name: str = Field(
        ..., min_length=1, description="Company name",
    )
    management_data: Optional[ManagementStructureData] = Field(None)
    employee_data: Optional[EmployeeRelationsData] = Field(None)
    remuneration_data: Optional[RemunerationData] = Field(None)
    tax_data: Optional[TaxComplianceData] = Field(None)
    ungc_signatory: Optional[bool] = Field(None)
    ungc_violations: Optional[bool] = Field(None)
    oecd_violations: Optional[bool] = Field(None)
    has_anti_corruption_policy: Optional[bool] = Field(None)
    has_anti_bribery_measures: Optional[bool] = Field(None)
    corruption_controversies: Optional[int] = Field(None, ge=0)


class AreaResult(BaseModel):
    """Assessment result for a single governance area."""

    area: GovernanceArea = Field(..., description="Governance area")
    area_name: str = Field(..., description="Full area name")
    status: GovernanceStatus = Field(..., description="Area assessment outcome")
    score: float = Field(
        ..., ge=0.0, le=100.0, description="Area score (0-100)",
    )
    weight: float = Field(
        ..., ge=0.0, le=1.0, description="Weight in composite score",
    )
    weighted_score: float = Field(
        ..., ge=0.0, le=100.0,
        description="Score * weight contribution",
    )
    criteria_total: int = Field(..., ge=0, description="Total criteria evaluated")
    criteria_passed: int = Field(..., ge=0, description="Criteria passed")
    criteria_failed: int = Field(..., ge=0, description="Criteria failed")
    criteria_no_data: int = Field(..., ge=0, description="Criteria without data")
    mandatory_failures: List[str] = Field(
        default_factory=list,
        description="Mandatory criteria that failed",
    )
    details: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-criterion detail records",
    )


class GovernanceViolation(BaseModel):
    """A specific governance violation found during assessment."""

    violation_type: ViolationType = Field(..., description="Type of violation")
    description: str = Field(..., description="Violation description")
    severity: str = Field(
        default="medium",
        description="high, medium, or low",
    )
    source: str = Field(
        default="assessment", description="Source of violation data",
    )


class GovernanceResult(BaseModel):
    """Complete good governance assessment result for a company.

    Contains per-area scores, composite score, pass/fail determination,
    violations list, and provenance tracking.
    """

    result_id: str = Field(
        default_factory=_new_uuid, description="Unique result identifier",
    )
    company_id: str = Field(..., description="Company identifier")
    company_name: str = Field(..., description="Company name")
    overall_status: GovernanceStatus = Field(
        ..., description="Overall governance assessment outcome",
    )
    composite_score: float = Field(
        ..., ge=0.0, le=100.0,
        description="Weighted composite governance score (0-100)",
    )
    pass_threshold: float = Field(
        ..., ge=0.0, le=100.0,
        description="Configured pass threshold",
    )
    area_results: Dict[str, AreaResult] = Field(
        default_factory=dict,
        description="Per-area assessment results",
    )
    violations: List[GovernanceViolation] = Field(
        default_factory=list,
        description="List of governance violations found",
    )
    ungc_adherence: Optional[bool] = Field(
        None, description="Whether UNGC adherence is confirmed",
    )
    anti_corruption_status: Optional[bool] = Field(
        None, description="Whether anti-corruption requirements are met",
    )
    anti_bribery_status: Optional[bool] = Field(
        None, description="Whether anti-bribery requirements are met",
    )
    data_coverage_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Percentage of criteria with available data",
    )
    assessed_at: datetime = Field(
        default_factory=_utcnow, description="Assessment timestamp",
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing time in milliseconds",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION, description="Engine version",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash",
    )


class PortfolioGovernanceResult(BaseModel):
    """Portfolio-level governance assessment aggregation."""

    portfolio_name: Optional[str] = Field(None, description="Portfolio name")
    total_companies: int = Field(..., ge=0)
    passing_companies: int = Field(..., ge=0)
    failing_companies: int = Field(..., ge=0)
    warning_companies: int = Field(..., ge=0)
    insufficient_data_companies: int = Field(..., ge=0)
    compliance_rate_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Percentage of companies passing governance",
    )
    average_composite_score: float = Field(
        ..., ge=0.0, le=100.0,
        description="Average composite score across portfolio",
    )
    area_averages: Dict[str, float] = Field(
        default_factory=dict,
        description="Average score per governance area",
    )
    company_results: List[GovernanceResult] = Field(
        default_factory=list,
    )
    common_violations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Most common violation types across portfolio",
    )
    provenance_hash: str = Field(default="")
    assessed_at: datetime = Field(default_factory=_utcnow)
    processing_time_ms: float = Field(default=0.0)
    engine_version: str = Field(default=_MODULE_VERSION)


# ---------------------------------------------------------------------------
# Default Criteria
# ---------------------------------------------------------------------------


def _build_default_criteria() -> List[GovernanceCriterion]:
    """Build default governance criteria for all four assessment areas.

    Returns:
        List of GovernanceCriterion covering all four Article 2(17) areas.
    """
    criteria = [
        # ============================================================
        # Area 1: Sound Management Structures
        # ============================================================
        GovernanceCriterion(
            criterion_id="MGT-01",
            area=GovernanceArea.MANAGEMENT_STRUCTURES,
            name="Board independence",
            description="Board has at least one-third independent members",
            check_type=GovernanceCheckType.THRESHOLD_MIN,
            metric_key="independent_board_pct",
            threshold_value=33.3,
            weight=0.20,
            is_mandatory=False,
            max_score=100.0,
        ),
        GovernanceCriterion(
            criterion_id="MGT-02",
            area=GovernanceArea.MANAGEMENT_STRUCTURES,
            name="Audit committee",
            description="Independent audit committee exists",
            check_type=GovernanceCheckType.BOOLEAN,
            metric_key="has_audit_committee",
            weight=0.20,
            is_mandatory=True,
        ),
        GovernanceCriterion(
            criterion_id="MGT-03",
            area=GovernanceArea.MANAGEMENT_STRUCTURES,
            name="Risk committee",
            description="Risk management committee exists",
            check_type=GovernanceCheckType.BOOLEAN,
            metric_key="has_risk_committee",
            weight=0.15,
        ),
        GovernanceCriterion(
            criterion_id="MGT-04",
            area=GovernanceArea.MANAGEMENT_STRUCTURES,
            name="CEO/Chair separation",
            description="CEO and board chair roles are separated",
            check_type=GovernanceCheckType.BOOLEAN,
            metric_key="ceo_chair_separation",
            weight=0.15,
        ),
        GovernanceCriterion(
            criterion_id="MGT-05",
            area=GovernanceArea.MANAGEMENT_STRUCTURES,
            name="Sustainability committee",
            description="Dedicated sustainability/ESG committee at board level",
            check_type=GovernanceCheckType.BOOLEAN,
            metric_key="has_sustainability_committee",
            weight=0.10,
        ),
        GovernanceCriterion(
            criterion_id="MGT-06",
            area=GovernanceArea.MANAGEMENT_STRUCTURES,
            name="Whistleblower mechanism",
            description="Whistleblower mechanism available for employees",
            check_type=GovernanceCheckType.BOOLEAN,
            metric_key="has_whistleblower_mechanism",
            weight=0.10,
        ),
        GovernanceCriterion(
            criterion_id="MGT-07",
            area=GovernanceArea.MANAGEMENT_STRUCTURES,
            name="Board meeting frequency",
            description="Board meets at least 4 times per year",
            check_type=GovernanceCheckType.THRESHOLD_MIN,
            metric_key="board_meetings_per_year",
            threshold_value=4.0,
            weight=0.10,
        ),
        # ============================================================
        # Area 2: Employee Relations
        # ============================================================
        GovernanceCriterion(
            criterion_id="EMP-01",
            area=GovernanceArea.EMPLOYEE_RELATIONS,
            name="ILO core conventions",
            description="Compliance with ILO core labor conventions",
            check_type=GovernanceCheckType.BOOLEAN,
            metric_key="ilo_core_conventions_compliance",
            weight=0.25,
            is_mandatory=True,
        ),
        GovernanceCriterion(
            criterion_id="EMP-02",
            area=GovernanceArea.EMPLOYEE_RELATIONS,
            name="Health and safety policy",
            description="Formal health and safety policy in place",
            check_type=GovernanceCheckType.BOOLEAN,
            metric_key="has_health_safety_policy",
            weight=0.20,
            is_mandatory=True,
        ),
        GovernanceCriterion(
            criterion_id="EMP-03",
            area=GovernanceArea.EMPLOYEE_RELATIONS,
            name="Lost time injury rate",
            description="Lost time injury frequency rate below 5.0",
            check_type=GovernanceCheckType.THRESHOLD_MAX,
            metric_key="lost_time_injury_rate",
            threshold_value=5.0,
            weight=0.15,
        ),
        GovernanceCriterion(
            criterion_id="EMP-04",
            area=GovernanceArea.EMPLOYEE_RELATIONS,
            name="Training programs",
            description="Formal employee training and development programs",
            check_type=GovernanceCheckType.BOOLEAN,
            metric_key="has_training_programs",
            weight=0.10,
        ),
        GovernanceCriterion(
            criterion_id="EMP-05",
            area=GovernanceArea.EMPLOYEE_RELATIONS,
            name="Living wage compliance",
            description="Company pays at least living wage to all employees",
            check_type=GovernanceCheckType.BOOLEAN,
            metric_key="living_wage_compliance",
            weight=0.15,
        ),
        GovernanceCriterion(
            criterion_id="EMP-06",
            area=GovernanceArea.EMPLOYEE_RELATIONS,
            name="Grievance mechanism",
            description="Employee grievance mechanism exists",
            check_type=GovernanceCheckType.BOOLEAN,
            metric_key="has_grievance_mechanism",
            weight=0.15,
        ),
        # ============================================================
        # Area 3: Remuneration of Staff
        # ============================================================
        GovernanceCriterion(
            criterion_id="REM-01",
            area=GovernanceArea.REMUNERATION,
            name="Remuneration policy",
            description="Formal remuneration policy in place",
            check_type=GovernanceCheckType.BOOLEAN,
            metric_key="has_remuneration_policy",
            weight=0.25,
            is_mandatory=True,
        ),
        GovernanceCriterion(
            criterion_id="REM-02",
            area=GovernanceArea.REMUNERATION,
            name="Policy disclosure",
            description="Remuneration policy is publicly disclosed",
            check_type=GovernanceCheckType.BOOLEAN,
            metric_key="remuneration_policy_disclosed",
            weight=0.15,
        ),
        GovernanceCriterion(
            criterion_id="REM-03",
            area=GovernanceArea.REMUNERATION,
            name="CEO pay ratio",
            description="CEO-to-median pay ratio below 200x",
            check_type=GovernanceCheckType.THRESHOLD_MAX,
            metric_key="ceo_to_median_pay_ratio",
            threshold_value=200.0,
            weight=0.15,
        ),
        GovernanceCriterion(
            criterion_id="REM-04",
            area=GovernanceArea.REMUNERATION,
            name="Clawback provisions",
            description="Clawback provisions in executive compensation",
            check_type=GovernanceCheckType.BOOLEAN,
            metric_key="has_clawback_provisions",
            weight=0.15,
        ),
        GovernanceCriterion(
            criterion_id="REM-05",
            area=GovernanceArea.REMUNERATION,
            name="Performance-linked pay",
            description="At least 30% of executive pay linked to performance",
            check_type=GovernanceCheckType.THRESHOLD_MIN,
            metric_key="performance_linked_pay_pct",
            threshold_value=30.0,
            weight=0.15,
        ),
        GovernanceCriterion(
            criterion_id="REM-06",
            area=GovernanceArea.REMUNERATION,
            name="ESG-linked remuneration",
            description="ESG targets included in executive compensation",
            check_type=GovernanceCheckType.BOOLEAN,
            metric_key="esg_linked_remuneration",
            weight=0.15,
        ),
        # ============================================================
        # Area 4: Tax Compliance
        # ============================================================
        GovernanceCriterion(
            criterion_id="TAX-01",
            area=GovernanceArea.TAX_COMPLIANCE,
            name="Tax strategy disclosure",
            description="Public disclosure of tax strategy",
            check_type=GovernanceCheckType.BOOLEAN,
            metric_key="has_tax_strategy_disclosure",
            weight=0.25,
            is_mandatory=True,
        ),
        GovernanceCriterion(
            criterion_id="TAX-02",
            area=GovernanceArea.TAX_COMPLIANCE,
            name="Country-by-country reporting",
            description="CBCR (public or to tax authorities)",
            check_type=GovernanceCheckType.BOOLEAN,
            metric_key="country_by_country_reporting",
            weight=0.20,
        ),
        GovernanceCriterion(
            criterion_id="TAX-03",
            area=GovernanceArea.TAX_COMPLIANCE,
            name="No aggressive tax planning",
            description="No aggressive tax planning structures identified",
            check_type=GovernanceCheckType.BOOLEAN,
            metric_key="aggressive_tax_planning_flag",
            threshold_value=None,
            weight=0.20,
            is_mandatory=True,
        ),
        GovernanceCriterion(
            criterion_id="TAX-04",
            area=GovernanceArea.TAX_COMPLIANCE,
            name="No tax haven exposure",
            description="No significant operations in tax havens",
            check_type=GovernanceCheckType.BOOLEAN,
            metric_key="tax_haven_exposure",
            weight=0.15,
        ),
        GovernanceCriterion(
            criterion_id="TAX-05",
            area=GovernanceArea.TAX_COMPLIANCE,
            name="Tax transparency score",
            description="Tax transparency score at least 40/100",
            check_type=GovernanceCheckType.THRESHOLD_MIN,
            metric_key="tax_transparency_score",
            threshold_value=40.0,
            weight=0.20,
        ),
    ]

    return criteria


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class GoodGovernanceEngine:
    """Good Governance Assessment Engine per SFDR Article 2(17).

    Evaluates investee companies across four governance areas:
    sound management structures, employee relations, remuneration of staff,
    and tax compliance. Produces per-area scores, a weighted composite
    score, and a pass/fail determination.

    Additional checks include UNGC adherence, anti-corruption policies,
    and anti-bribery measures, which can be configured as hard requirements.

    Zero-Hallucination Guarantees:
        - All scoring uses deterministic arithmetic
        - Configurable weights and thresholds
        - SHA-256 provenance hashing on every result
        - No LLM involvement in scoring

    Attributes:
        config: Engine configuration with criteria and thresholds.
        _criteria_by_area: Criteria grouped by governance area.
        _assessment_count: Running count of assessments.

    Example:
        >>> config = GovernanceConfig(criteria=_build_default_criteria())
        >>> engine = GoodGovernanceEngine(config)
        >>> data = CompanyGovernanceData(
        ...     company_id="ISIN001", company_name="Example Corp",
        ...     management_data=ManagementStructureData(has_audit_committee=True),
        ... )
        >>> result = engine.assess_governance(data)
        >>> print(f"Score: {result.composite_score}, Status: {result.overall_status}")
    """

    def __init__(self, config: Optional[GovernanceConfig] = None) -> None:
        """Initialize the Good Governance Engine.

        Args:
            config: Configuration. If None, uses default criteria and thresholds.
        """
        if config is None:
            config = GovernanceConfig(criteria=_build_default_criteria())

        if not config.criteria:
            config.criteria = _build_default_criteria()

        self.config = config
        self._criteria_by_area: Dict[str, List[GovernanceCriterion]] = (
            self._group_criteria_by_area()
        )
        self._assessment_count: int = 0

        logger.info(
            "GoodGovernanceEngine initialized (v%s, %d criteria, "
            "pass_threshold=%.1f, areas=%d)",
            _MODULE_VERSION,
            len(config.criteria),
            config.pass_threshold,
            len(self._criteria_by_area),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess_governance(
        self,
        company_data: CompanyGovernanceData,
    ) -> GovernanceResult:
        """Assess good governance for a single company.

        Evaluates all four governance areas, computes area scores and
        composite score, checks additional requirements (UNGC, anti-corruption),
        and determines overall pass/fail status.

        Args:
            company_data: Complete governance data for the company.

        Returns:
            GovernanceResult with scores, status, and violations.

        Raises:
            ValueError: If company_id is empty.
        """
        start = _utcnow()
        self._assessment_count += 1

        if not company_data.company_id:
            raise ValueError("company_id is required")

        logger.debug(
            "Assessing governance for %s (%s)",
            company_data.company_id, company_data.company_name,
        )

        # Step 1: Flatten company data into a single lookup dict
        flat_data = self._flatten_company_data(company_data)

        # Step 2: Assess each governance area
        area_results: Dict[str, AreaResult] = {}
        for area in GovernanceArea:
            area_result = self._assess_area(area, flat_data)
            area_results[area.value] = area_result

        # Step 3: Compute composite score
        composite_score = self._compute_composite_score(area_results)

        # Step 4: Check additional requirements
        ungc_ok = self._check_ungc_adherence(company_data)
        corruption_ok = self._check_anti_corruption(company_data)
        bribery_ok = self._check_anti_bribery(company_data)

        # Step 5: Collect violations
        violations = self._collect_violations(
            company_data, area_results, ungc_ok, corruption_ok, bribery_ok
        )

        # Step 6: Determine overall status
        overall_status = self._determine_overall_status(
            composite_score, area_results, ungc_ok, corruption_ok, bribery_ok
        )

        # Step 7: Data coverage
        total_criteria = sum(ar.criteria_total for ar in area_results.values())
        no_data_criteria = sum(ar.criteria_no_data for ar in area_results.values())
        data_coverage = (
            ((total_criteria - no_data_criteria) / total_criteria * 100.0)
            if total_criteria > 0
            else 0.0
        )

        elapsed_ms = (_utcnow() - start).total_seconds() * 1000

        result = GovernanceResult(
            company_id=company_data.company_id,
            company_name=company_data.company_name,
            overall_status=overall_status,
            composite_score=_round_val(composite_score),
            pass_threshold=self.config.pass_threshold,
            area_results=area_results,
            violations=violations,
            ungc_adherence=ungc_ok,
            anti_corruption_status=corruption_ok,
            anti_bribery_status=bribery_ok,
            data_coverage_pct=_round_val(data_coverage),
            processing_time_ms=round(elapsed_ms, 2),
        )

        result.provenance_hash = _compute_hash({
            "company_id": company_data.company_id,
            "composite_score": composite_score,
            "overall_status": overall_status.value,
            "area_scores": {
                k: v.score for k, v in area_results.items()
            },
        })

        logger.info(
            "Governance assessment for %s: status=%s, score=%.1f, "
            "violations=%d, time=%.1fms",
            company_data.company_id, overall_status.value,
            composite_score, len(violations), elapsed_ms,
        )

        return result

    def assess_portfolio_governance(
        self,
        companies: List[CompanyGovernanceData],
        portfolio_name: Optional[str] = None,
    ) -> PortfolioGovernanceResult:
        """Assess governance for an entire portfolio of companies.

        Args:
            companies: List of company governance data.
            portfolio_name: Optional portfolio name.

        Returns:
            PortfolioGovernanceResult with aggregate metrics.

        Raises:
            ValueError: If companies list is empty.
        """
        start = _utcnow()

        if not companies:
            raise ValueError("Companies list cannot be empty")

        logger.info(
            "Assessing portfolio governance for %d companies", len(companies)
        )

        results: List[GovernanceResult] = []
        for company in companies:
            try:
                result = self.assess_governance(company)
                results.append(result)
            except Exception as exc:
                logger.error(
                    "Governance assessment failed for %s: %s",
                    company.company_id, exc,
                )
                raise

        passing = sum(
            1 for r in results if r.overall_status == GovernanceStatus.PASS
        )
        failing = sum(
            1 for r in results if r.overall_status == GovernanceStatus.FAIL
        )
        warning = sum(
            1 for r in results if r.overall_status == GovernanceStatus.WARNING
        )
        no_data = sum(
            1 for r in results
            if r.overall_status == GovernanceStatus.INSUFFICIENT_DATA
        )

        assessable = len(results) - no_data
        compliance_rate = (
            (passing / assessable * 100.0) if assessable > 0 else 0.0
        )

        avg_score = (
            sum(r.composite_score for r in results) / len(results)
            if results else 0.0
        )

        area_averages = self._compute_area_averages(results)
        common_violations = self._compute_common_violations(results)

        elapsed_ms = (_utcnow() - start).total_seconds() * 1000

        portfolio_result = PortfolioGovernanceResult(
            portfolio_name=portfolio_name,
            total_companies=len(results),
            passing_companies=passing,
            failing_companies=failing,
            warning_companies=warning,
            insufficient_data_companies=no_data,
            compliance_rate_pct=_round_val(compliance_rate),
            average_composite_score=_round_val(avg_score),
            area_averages=area_averages,
            company_results=results,
            common_violations=common_violations,
            processing_time_ms=round(elapsed_ms, 2),
        )

        portfolio_result.provenance_hash = _compute_hash({
            "portfolio_name": portfolio_name,
            "total": len(results),
            "passing": passing,
            "compliance_rate": compliance_rate,
            "avg_score": avg_score,
        })

        logger.info(
            "Portfolio governance complete: %d companies, compliance=%.1f%%, "
            "avg_score=%.1f, time=%.1fms",
            len(results), compliance_rate, avg_score, elapsed_ms,
        )

        return portfolio_result

    def get_violation_report(
        self,
        result: GovernanceResult,
    ) -> List[Dict[str, Any]]:
        """Generate a structured violation report from a governance result.

        Args:
            result: A completed governance assessment result.

        Returns:
            List of violation detail dictionaries.
        """
        report: List[Dict[str, Any]] = []

        for violation in result.violations:
            report.append({
                "company_id": result.company_id,
                "company_name": result.company_name,
                "violation_type": violation.violation_type.value,
                "description": violation.description,
                "severity": violation.severity,
                "source": violation.source,
                "overall_status": result.overall_status.value,
                "composite_score": result.composite_score,
            })

        return report

    def governance_score(
        self,
        company_data: CompanyGovernanceData,
    ) -> float:
        """Quick composite governance score for a company.

        Convenience method returning just the numeric score without
        the full assessment result.

        Args:
            company_data: Company governance data.

        Returns:
            Composite governance score (0-100).
        """
        result = self.assess_governance(company_data)
        return result.composite_score

    # ------------------------------------------------------------------
    # Private: Area Assessment
    # ------------------------------------------------------------------

    def _assess_area(
        self,
        area: GovernanceArea,
        flat_data: Dict[str, Any],
    ) -> AreaResult:
        """Assess a single governance area.

        Evaluates each criterion for the area, computes the area score,
        and determines pass/fail status.

        Args:
            area: Governance area to assess.
            flat_data: Flattened company data dictionary.

        Returns:
            AreaResult with score and status.
        """
        criteria = self._criteria_by_area.get(area.value, [])
        area_weight = self.config.area_weights.get(area.value, 0.25)

        if not criteria:
            return AreaResult(
                area=area,
                area_name=AREA_NAMES.get(area.value, area.value),
                status=GovernanceStatus.NOT_APPLICABLE,
                score=0.0,
                weight=area_weight,
                weighted_score=0.0,
                criteria_total=0,
                criteria_passed=0,
                criteria_failed=0,
                criteria_no_data=0,
            )

        details: List[Dict[str, Any]] = []
        total_weight = sum(c.weight for c in criteria)
        weighted_score_sum = 0.0
        passed = 0
        failed = 0
        no_data = 0
        mandatory_failures: List[str] = []

        for criterion in criteria:
            check_result = self._evaluate_criterion(criterion, flat_data)
            details.append(check_result)

            if check_result["status"] == "PASS":
                passed += 1
                criterion_score = criterion.max_score
            elif check_result["status"] == "FAIL":
                failed += 1
                criterion_score = 0.0
                if criterion.is_mandatory:
                    mandatory_failures.append(criterion.criterion_id)
            else:
                no_data += 1
                criterion_score = 0.0

            normalized_weight = (
                criterion.weight / total_weight if total_weight > 0 else 0.0
            )
            weighted_score_sum += criterion_score * normalized_weight

        area_score = _clamp(weighted_score_sum)

        # Determine area status
        if mandatory_failures:
            area_status = GovernanceStatus.FAIL
        elif no_data > len(criteria) * 0.5:
            area_status = GovernanceStatus.INSUFFICIENT_DATA
        elif area_score >= self.config.area_pass_threshold:
            area_status = GovernanceStatus.PASS
        elif area_score >= self.config.area_pass_threshold * 0.7:
            area_status = GovernanceStatus.WARNING
        else:
            area_status = GovernanceStatus.FAIL

        return AreaResult(
            area=area,
            area_name=AREA_NAMES.get(area.value, area.value),
            status=area_status,
            score=_round_val(area_score),
            weight=area_weight,
            weighted_score=_round_val(area_score * area_weight),
            criteria_total=len(criteria),
            criteria_passed=passed,
            criteria_failed=failed,
            criteria_no_data=no_data,
            mandatory_failures=mandatory_failures,
            details=details,
        )

    def _evaluate_criterion(
        self,
        criterion: GovernanceCriterion,
        flat_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate a single governance criterion.

        Args:
            criterion: The criterion to evaluate.
            flat_data: Flattened company data dictionary.

        Returns:
            Detail dict with status, actual value, and explanation.
        """
        value = flat_data.get(criterion.metric_key)

        detail: Dict[str, Any] = {
            "criterion_id": criterion.criterion_id,
            "name": criterion.name,
            "area": criterion.area.value,
            "check_type": criterion.check_type.value,
            "metric_key": criterion.metric_key,
            "actual_value": value,
            "threshold_value": criterion.threshold_value,
            "is_mandatory": criterion.is_mandatory,
            "weight": criterion.weight,
        }

        if value is None:
            detail["status"] = "NO_DATA"
            detail["explanation"] = f"No data for {criterion.metric_key}"
            return detail

        if criterion.check_type == GovernanceCheckType.BOOLEAN:
            is_pass = self._evaluate_boolean(criterion, value)
        elif criterion.check_type == GovernanceCheckType.THRESHOLD_MIN:
            is_pass = float(value) >= (criterion.threshold_value or 0.0)
        elif criterion.check_type == GovernanceCheckType.THRESHOLD_MAX:
            is_pass = float(value) <= (criterion.threshold_value or float("inf"))
        elif criterion.check_type == GovernanceCheckType.SCORE:
            is_pass = float(value) >= (criterion.threshold_value or 0.0)
        else:
            is_pass = bool(value)

        detail["status"] = "PASS" if is_pass else "FAIL"
        detail["explanation"] = (
            f"{criterion.name}: {'passed' if is_pass else 'failed'} "
            f"(value={value}, threshold={criterion.threshold_value})"
        )

        return detail

    def _evaluate_boolean(
        self,
        criterion: GovernanceCriterion,
        value: Any,
    ) -> bool:
        """Evaluate a boolean criterion.

        Some boolean criteria are "negative" flags (e.g., aggressive_tax_planning_flag)
        where True means FAIL. These are identified by their metric_key suffix.

        Args:
            criterion: The criterion to evaluate.
            value: The actual value.

        Returns:
            True if the criterion passes.
        """
        negative_flags = {
            "aggressive_tax_planning_flag",
            "tax_haven_exposure",
            "excessive_severance_provisions",
        }

        if criterion.metric_key in negative_flags:
            # For negative flags, True = FAIL (so pass if value is False)
            return not bool(value)

        return bool(value)

    # ------------------------------------------------------------------
    # Private: Composite Score
    # ------------------------------------------------------------------

    def _compute_composite_score(
        self,
        area_results: Dict[str, AreaResult],
    ) -> float:
        """Compute weighted composite governance score.

        Args:
            area_results: Per-area assessment results.

        Returns:
            Composite score (0-100).
        """
        total = 0.0
        for area_result in area_results.values():
            total += area_result.weighted_score

        return _clamp(total)

    # ------------------------------------------------------------------
    # Private: Additional Checks
    # ------------------------------------------------------------------

    def _check_ungc_adherence(
        self,
        company_data: CompanyGovernanceData,
    ) -> Optional[bool]:
        """Check UNGC adherence.

        Args:
            company_data: Company data.

        Returns:
            True if UNGC compliant, False if violations, None if no data.
        """
        if company_data.ungc_violations is True:
            return False
        if company_data.oecd_violations is True:
            return False
        if company_data.ungc_signatory is True:
            return True
        if company_data.ungc_violations is False:
            return True
        return None

    def _check_anti_corruption(
        self,
        company_data: CompanyGovernanceData,
    ) -> Optional[bool]:
        """Check anti-corruption requirements.

        Args:
            company_data: Company data.

        Returns:
            True if compliant, False if not, None if no data.
        """
        if company_data.has_anti_corruption_policy is None:
            return None
        if not company_data.has_anti_corruption_policy:
            return False
        if (
            company_data.corruption_controversies is not None
            and company_data.corruption_controversies > 0
        ):
            return False
        return True

    def _check_anti_bribery(
        self,
        company_data: CompanyGovernanceData,
    ) -> Optional[bool]:
        """Check anti-bribery requirements.

        Args:
            company_data: Company data.

        Returns:
            True if compliant, False if not, None if no data.
        """
        if company_data.has_anti_bribery_measures is None:
            return None
        return company_data.has_anti_bribery_measures

    # ------------------------------------------------------------------
    # Private: Violations Collection
    # ------------------------------------------------------------------

    def _collect_violations(
        self,
        company_data: CompanyGovernanceData,
        area_results: Dict[str, AreaResult],
        ungc_ok: Optional[bool],
        corruption_ok: Optional[bool],
        bribery_ok: Optional[bool],
    ) -> List[GovernanceViolation]:
        """Collect all governance violations from the assessment.

        Args:
            company_data: Company data.
            area_results: Per-area results.
            ungc_ok: UNGC adherence status.
            corruption_ok: Anti-corruption status.
            bribery_ok: Anti-bribery status.

        Returns:
            List of GovernanceViolation objects.
        """
        violations: List[GovernanceViolation] = []

        # UNGC violations
        if company_data.ungc_violations is True:
            violations.append(GovernanceViolation(
                violation_type=ViolationType.UNGC_VIOLATION,
                description="Company has UN Global Compact principle violations",
                severity="high",
                source="company_data",
            ))

        if company_data.oecd_violations is True:
            violations.append(GovernanceViolation(
                violation_type=ViolationType.OECD_VIOLATION,
                description="Company has OECD Guidelines violations",
                severity="high",
                source="company_data",
            ))

        # Corruption
        if corruption_ok is False:
            violations.append(GovernanceViolation(
                violation_type=ViolationType.CORRUPTION,
                description=(
                    "Anti-corruption policy missing or corruption controversies exist"
                ),
                severity="high",
                source="company_data",
            ))

        # Bribery
        if bribery_ok is False:
            violations.append(GovernanceViolation(
                violation_type=ViolationType.BRIBERY,
                description="Anti-bribery measures not in place",
                severity="high",
                source="company_data",
            ))

        # Area-level mandatory failures
        for area_key, area_result in area_results.items():
            for mandatory_id in area_result.mandatory_failures:
                violation_type = self._map_criterion_to_violation(mandatory_id)
                violations.append(GovernanceViolation(
                    violation_type=violation_type,
                    description=(
                        f"Mandatory criterion {mandatory_id} failed "
                        f"in area {area_result.area_name}"
                    ),
                    severity="high",
                    source="area_assessment",
                ))

        return violations

    def _map_criterion_to_violation(
        self,
        criterion_id: str,
    ) -> ViolationType:
        """Map a criterion ID to a violation type.

        Args:
            criterion_id: Criterion identifier (e.g., 'MGT-02').

        Returns:
            Appropriate ViolationType.
        """
        prefix = criterion_id.split("-")[0] if "-" in criterion_id else ""
        mapping = {
            "MGT": ViolationType.BOARD_INDEPENDENCE,
            "EMP": ViolationType.LABOR_RIGHTS,
            "REM": ViolationType.REMUNERATION_EXCESS,
            "TAX": ViolationType.TAX_EVASION,
        }
        return mapping.get(prefix, ViolationType.OTHER)

    # ------------------------------------------------------------------
    # Private: Overall Status Determination
    # ------------------------------------------------------------------

    def _determine_overall_status(
        self,
        composite_score: float,
        area_results: Dict[str, AreaResult],
        ungc_ok: Optional[bool],
        corruption_ok: Optional[bool],
        bribery_ok: Optional[bool],
    ) -> GovernanceStatus:
        """Determine overall governance status.

        Fail conditions (any triggers FAIL):
            1. Composite score below pass_threshold
            2. Any area has mandatory criterion failure
            3. UNGC adherence required but not met
            4. Anti-corruption required but not met
            5. Anti-bribery required but not met

        Args:
            composite_score: Weighted composite score.
            area_results: Per-area results.
            ungc_ok: UNGC adherence status.
            corruption_ok: Anti-corruption status.
            bribery_ok: Anti-bribery status.

        Returns:
            Overall GovernanceStatus.
        """
        # Hard fail checks
        if self.config.ungc_adherence_required and ungc_ok is False:
            return GovernanceStatus.FAIL

        if self.config.anti_corruption_required and corruption_ok is False:
            return GovernanceStatus.FAIL

        if self.config.anti_bribery_required and bribery_ok is False:
            return GovernanceStatus.FAIL

        # Check for any area with mandatory failures
        has_mandatory_fail = any(
            ar.mandatory_failures for ar in area_results.values()
        )
        if has_mandatory_fail:
            return GovernanceStatus.FAIL

        # Check for insufficient data across areas
        insufficient_areas = sum(
            1 for ar in area_results.values()
            if ar.status == GovernanceStatus.INSUFFICIENT_DATA
        )
        if insufficient_areas >= len(area_results) / 2:
            return GovernanceStatus.INSUFFICIENT_DATA

        # Score-based determination
        if composite_score >= self.config.pass_threshold:
            return GovernanceStatus.PASS
        elif composite_score >= self.config.pass_threshold * 0.7:
            return GovernanceStatus.WARNING
        else:
            return GovernanceStatus.FAIL

    # ------------------------------------------------------------------
    # Private: Data Flattening
    # ------------------------------------------------------------------

    def _flatten_company_data(
        self,
        company_data: CompanyGovernanceData,
    ) -> Dict[str, Any]:
        """Flatten nested company data into a single lookup dictionary.

        Merges all sub-model fields into a single dict keyed by field name,
        enabling criterion evaluation without knowing which sub-model a
        field belongs to.

        Args:
            company_data: Nested company governance data.

        Returns:
            Flat dictionary of all governance data fields.
        """
        flat: Dict[str, Any] = {}

        if company_data.management_data:
            for field_name, value in company_data.management_data.model_dump().items():
                flat[field_name] = value

        if company_data.employee_data:
            for field_name, value in company_data.employee_data.model_dump().items():
                flat[field_name] = value

        if company_data.remuneration_data:
            for field_name, value in company_data.remuneration_data.model_dump().items():
                flat[field_name] = value

        if company_data.tax_data:
            for field_name, value in company_data.tax_data.model_dump().items():
                flat[field_name] = value

        # Add top-level governance fields
        flat["ungc_signatory"] = company_data.ungc_signatory
        flat["ungc_violations"] = company_data.ungc_violations
        flat["oecd_violations"] = company_data.oecd_violations
        flat["has_anti_corruption_policy"] = company_data.has_anti_corruption_policy
        flat["has_anti_bribery_measures"] = company_data.has_anti_bribery_measures
        flat["corruption_controversies"] = company_data.corruption_controversies

        return flat

    # ------------------------------------------------------------------
    # Private: Grouping and Aggregation Helpers
    # ------------------------------------------------------------------

    def _group_criteria_by_area(
        self,
    ) -> Dict[str, List[GovernanceCriterion]]:
        """Group criteria by governance area.

        Returns:
            Dict mapping area value to list of criteria.
        """
        grouped: Dict[str, List[GovernanceCriterion]] = defaultdict(list)
        for criterion in self.config.criteria:
            grouped[criterion.area.value].append(criterion)
        return dict(grouped)

    def _compute_area_averages(
        self,
        results: List[GovernanceResult],
    ) -> Dict[str, float]:
        """Compute average scores per governance area across portfolio.

        Args:
            results: List of company governance results.

        Returns:
            Dict mapping area value to average score.
        """
        area_sums: Dict[str, float] = defaultdict(float)
        area_counts: Dict[str, int] = defaultdict(int)

        for result in results:
            for area_key, area_result in result.area_results.items():
                area_sums[area_key] += area_result.score
                area_counts[area_key] += 1

        return {
            area: _round_val(
                area_sums[area] / area_counts[area]
            ) if area_counts[area] > 0 else 0.0
            for area in area_sums
        }

    def _compute_common_violations(
        self,
        results: List[GovernanceResult],
    ) -> List[Dict[str, Any]]:
        """Compute most common violation types across portfolio.

        Args:
            results: List of company governance results.

        Returns:
            List of violation frequency dicts, sorted descending.
        """
        violation_counts: Dict[str, int] = defaultdict(int)

        for result in results:
            for violation in result.violations:
                violation_counts[violation.violation_type.value] += 1

        sorted_violations = sorted(
            violation_counts.items(), key=lambda x: x[1], reverse=True
        )

        return [
            {
                "violation_type": vtype,
                "count": count,
                "prevalence_pct": _round_val(
                    count / len(results) * 100.0
                ) if results else 0.0,
            }
            for vtype, count in sorted_violations
        ]

    # ------------------------------------------------------------------
    # Read-only Properties
    # ------------------------------------------------------------------

    @property
    def assessment_count(self) -> int:
        """Number of governance assessments performed since initialization."""
        return self._assessment_count

    @property
    def governance_areas(self) -> List[str]:
        """List of governance area values."""
        return [a.value for a in GovernanceArea]

    @property
    def total_criteria(self) -> int:
        """Total number of configured governance criteria."""
        return len(self.config.criteria)

    @property
    def mandatory_criteria(self) -> List[str]:
        """List of mandatory criterion IDs."""
        return [
            c.criterion_id
            for c in self.config.criteria
            if c.is_mandatory
        ]
