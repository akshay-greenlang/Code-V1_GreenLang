# -*- coding: utf-8 -*-
"""
Portfolio Screening Workflow
================================

Four-phase workflow for investment screening of SFDR Article 8 financial
products. Orchestrates universe definition, negative screening, positive
screening, and compliance checking into a single auditable pipeline.

Regulatory Context:
    Per EU SFDR Regulation 2019/2088 and Delegated Regulation 2022/1288 (RTS):
    - Article 8: Products that promote E/S characteristics must disclose
      how those characteristics are met, including binding elements.
    - Binding elements typically include exclusion criteria and minimum
      standards that constitute the "negative screening" component.
    - Positive screening criteria define the E/S promotion aspect.
    - Pre-contractual commitments define minimum thresholds that must
      be maintained through continuous monitoring.

    Common Exclusion Criteria (Negative Screening):
    - Controversial weapons (cluster munitions, anti-personnel mines,
      biological/chemical weapons): Zero tolerance
    - Tobacco production: >5% revenue threshold
    - Thermal coal mining/power generation: >30% revenue threshold
    - Oil and gas exploration: Varies by product
    - UNGC/OECD Guidelines violators: Zero tolerance
    - Severe environmental controversies: Case-by-case

    Common Positive Criteria:
    - Minimum ESG rating threshold
    - Environmental performance metrics
    - Social impact indicators
    - Governance quality scores

Phases:
    1. UniverseDefinition - Define investment universe, import holdings,
       set screening scope (new investments vs full portfolio)
    2. NegativeScreening - Apply exclusion criteria, flag violations,
       calculate exclusion impact
    3. PositiveScreening - Apply E/S promotion criteria (ESG ratings,
       sustainability indicators, environmental performance thresholds)
    4. ComplianceCheck - Verify portfolio meets Article 8 binding elements,
       calculate compliance percentage, generate screening report

Author: GreenLang Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# UTILITIES
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC time with timezone info."""
    return datetime.now(timezone.utc)


def _hash_data(data: Any) -> str:
    """Compute SHA-256 provenance hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


class ScreeningScope(str, Enum):
    """Screening scope."""
    FULL_PORTFOLIO = "FULL_PORTFOLIO"
    NEW_INVESTMENTS = "NEW_INVESTMENTS"
    REBALANCE = "REBALANCE"


class ViolationSeverity(str, Enum):
    """Severity of a screening violation."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ScreeningVerdict(str, Enum):
    """Screening outcome verdict."""
    PASS = "PASS"
    FAIL = "FAIL"
    WATCHLIST = "WATCHLIST"
    EXEMPT = "EXEMPT"


# =============================================================================
# DATA MODELS - SHARED
# =============================================================================


class WorkflowContext(BaseModel):
    """Shared state passed between workflow phases."""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str = Field(..., description="Organization identifier")
    execution_timestamp: datetime = Field(default_factory=_utcnow)
    config: Dict[str, Any] = Field(default_factory=dict)
    phase_states: Dict[str, PhaseStatus] = Field(default_factory=dict)
    phase_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def set_phase_output(self, phase_name: str, outputs: Dict[str, Any]) -> None:
        """Store phase outputs for downstream consumption."""
        self.phase_outputs[phase_name] = outputs

    def get_phase_output(self, phase_name: str) -> Dict[str, Any]:
        """Retrieve outputs from a previous phase."""
        return self.phase_outputs.get(phase_name, {})

    def mark_phase(self, phase_name: str, status: PhaseStatus) -> None:
        """Record phase status for checkpoint/resume."""
        self.phase_states[phase_name] = status

    def is_phase_completed(self, phase_name: str) -> bool:
        """Check if a phase has already completed."""
        return self.phase_states.get(phase_name) == PhaseStatus.COMPLETED


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    records_processed: int = Field(default=0)


class WorkflowResult(BaseModel):
    """Complete result from a multi-phase workflow execution."""
    workflow_id: str = Field(..., description="Unique workflow execution ID")
    workflow_name: str = Field(..., description="Workflow type identifier")
    status: WorkflowStatus = Field(..., description="Overall workflow status")
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    phases: List[PhaseResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


# =============================================================================
# DATA MODELS - PORTFOLIO SCREENING
# =============================================================================


class ScreeningHolding(BaseModel):
    """A holding to be screened."""
    holding_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    issuer_name: str = Field(..., description="Issuer name")
    isin: Optional[str] = Field(None)
    sector: str = Field(default="")
    country: str = Field(default="")
    portfolio_weight_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    market_value_eur: float = Field(default=0.0, ge=0.0)
    esg_rating: Optional[float] = Field(None, ge=0.0, le=100.0)
    controversial_weapons: bool = Field(default=False)
    tobacco_revenue_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    thermal_coal_revenue_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    fossil_fuel_exploration: bool = Field(default=False)
    ungc_violator: bool = Field(default=False)
    environmental_controversy_severity: Optional[str] = Field(None)
    carbon_intensity_tco2e_per_m_eur: Optional[float] = Field(None, ge=0.0)
    renewable_energy_pct: Optional[float] = Field(None, ge=0.0, le=100.0)
    water_stress_exposure: bool = Field(default=False)
    is_new_investment: bool = Field(default=False)


class ExclusionRule(BaseModel):
    """A single exclusion rule for negative screening."""
    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Rule name")
    field: str = Field(..., description="Field to check on holding")
    operator: str = Field(
        ..., description="Comparison: gt, gte, lt, lte, eq, bool_true"
    )
    threshold: float = Field(default=0.0, description="Threshold value")
    severity: ViolationSeverity = Field(default=ViolationSeverity.CRITICAL)
    binding: bool = Field(default=True)


class PositiveRule(BaseModel):
    """A single positive screening rule."""
    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Rule name")
    field: str = Field(..., description="Field to check on holding")
    operator: str = Field(
        ..., description="Comparison: gt, gte, lt, lte, eq"
    )
    threshold: float = Field(default=0.0)
    weight: float = Field(
        default=1.0, ge=0.0, description="Weighting for scoring"
    )


class PortfolioScreeningInput(BaseModel):
    """Input configuration for the portfolio screening workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    product_name: str = Field(..., description="Financial product name")
    screening_date: str = Field(
        ..., description="Screening date YYYY-MM-DD"
    )
    screening_scope: ScreeningScope = Field(
        default=ScreeningScope.FULL_PORTFOLIO
    )
    holdings: List[ScreeningHolding] = Field(
        default_factory=list, description="Holdings to screen"
    )
    exclusion_rules: List[ExclusionRule] = Field(
        default_factory=lambda: [
            ExclusionRule(
                name="Controversial Weapons",
                field="controversial_weapons",
                operator="bool_true",
                severity=ViolationSeverity.CRITICAL,
            ),
            ExclusionRule(
                name="Tobacco >5% Revenue",
                field="tobacco_revenue_pct",
                operator="gt",
                threshold=5.0,
                severity=ViolationSeverity.HIGH,
            ),
            ExclusionRule(
                name="Thermal Coal >30% Revenue",
                field="thermal_coal_revenue_pct",
                operator="gt",
                threshold=30.0,
                severity=ViolationSeverity.HIGH,
            ),
            ExclusionRule(
                name="Fossil Fuel Exploration",
                field="fossil_fuel_exploration",
                operator="bool_true",
                severity=ViolationSeverity.MEDIUM,
            ),
            ExclusionRule(
                name="UNGC Violators",
                field="ungc_violator",
                operator="bool_true",
                severity=ViolationSeverity.CRITICAL,
            ),
        ]
    )
    positive_rules: List[PositiveRule] = Field(
        default_factory=lambda: [
            PositiveRule(
                name="Minimum ESG Rating",
                field="esg_rating",
                operator="gte",
                threshold=50.0,
                weight=2.0,
            ),
        ]
    )
    minimum_esg_rating: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Minimum ESG rating threshold"
    )
    minimum_compliance_pct: float = Field(
        default=90.0, ge=0.0, le=100.0,
        description="Minimum portfolio compliance percentage"
    )
    skip_phases: List[str] = Field(default_factory=list)

    @field_validator("screening_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate screening date is valid ISO format."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("screening_date must be YYYY-MM-DD format")
        return v


class PortfolioScreeningResult(WorkflowResult):
    """Complete result from the portfolio screening workflow."""
    product_name: str = Field(default="")
    total_holdings_screened: int = Field(default=0)
    exclusion_violations: int = Field(default=0)
    critical_violations: int = Field(default=0)
    positive_screen_pass: int = Field(default=0)
    positive_screen_fail: int = Field(default=0)
    compliance_pct: float = Field(default=0.0)
    is_compliant: bool = Field(default=False)
    watchlist_count: int = Field(default=0)


# =============================================================================
# PHASE IMPLEMENTATIONS
# =============================================================================


class UniverseDefinitionPhase:
    """
    Phase 1: Universe Definition.

    Defines the investment universe, imports holdings data, and sets
    the screening scope.
    """

    PHASE_NAME = "universe_definition"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute universe definition phase.

        Args:
            context: Workflow context with holdings data.

        Returns:
            PhaseResult with defined universe and scope.
        """
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            holdings = config.get("holdings", [])
            scope = config.get(
                "screening_scope", ScreeningScope.FULL_PORTFOLIO.value
            )

            # Filter holdings based on scope
            if scope == ScreeningScope.NEW_INVESTMENTS.value:
                screened = [
                    h for h in holdings if h.get("is_new_investment", False)
                ]
            else:
                screened = holdings

            outputs["total_universe_size"] = len(holdings)
            outputs["screened_holdings_count"] = len(screened)
            outputs["screening_scope"] = scope
            outputs["screened_holdings"] = screened

            # Universe statistics
            total_value = sum(
                h.get("market_value_eur", 0.0) for h in screened
            )
            total_weight = sum(
                h.get("portfolio_weight_pct", 0.0) for h in screened
            )

            outputs["total_screened_value_eur"] = round(total_value, 2)
            outputs["total_screened_weight_pct"] = round(total_weight, 2)

            # Sector distribution
            sector_dist: Dict[str, int] = {}
            for h in screened:
                sector = h.get("sector", "Other")
                sector_dist[sector] = sector_dist.get(sector, 0) + 1
            outputs["sector_distribution"] = sector_dist

            # Country distribution
            country_dist: Dict[str, int] = {}
            for h in screened:
                country = h.get("country", "Other")
                country_dist[country] = country_dist.get(country, 0) + 1
            outputs["country_distribution"] = country_dist

            if len(screened) == 0:
                warnings.append(
                    "No holdings to screen in the defined scope"
                )

            status = PhaseStatus.COMPLETED
            records = len(screened)

        except Exception as exc:
            logger.error("UniverseDefinition failed: %s", exc, exc_info=True)
            errors.append(f"Universe definition failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )


class NegativeScreeningPhase:
    """
    Phase 2: Negative Screening.

    Applies exclusion criteria to each holding and flags violations
    with severity levels.
    """

    PHASE_NAME = "negative_screening"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute negative screening phase.

        Args:
            context: Workflow context with universe and exclusion rules.

        Returns:
            PhaseResult with violation details and exclusion impact.
        """
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            universe_output = context.get_phase_output("universe_definition")
            holdings = universe_output.get("screened_holdings", [])
            exclusion_rules = config.get("exclusion_rules", [])

            violations: List[Dict[str, Any]] = []
            pass_holdings: List[Dict[str, Any]] = []
            excluded_weight = 0.0
            excluded_value = 0.0

            for holding in holdings:
                holding_violations = []

                for rule in exclusion_rules:
                    field = rule.get("field", "")
                    operator = rule.get("operator", "")
                    threshold = rule.get("threshold", 0.0)
                    value = holding.get(field)

                    is_violated = False
                    if value is None:
                        continue

                    if operator == "bool_true" and value is True:
                        is_violated = True
                    elif operator == "gt" and isinstance(value, (int, float)):
                        is_violated = value > threshold
                    elif operator == "gte" and isinstance(value, (int, float)):
                        is_violated = value >= threshold
                    elif operator == "lt" and isinstance(value, (int, float)):
                        is_violated = value < threshold
                    elif operator == "eq":
                        is_violated = value == threshold

                    if is_violated:
                        holding_violations.append({
                            "rule_name": rule.get("name", ""),
                            "rule_id": rule.get("rule_id", ""),
                            "field": field,
                            "actual_value": value,
                            "threshold": threshold,
                            "severity": rule.get(
                                "severity",
                                ViolationSeverity.HIGH.value
                            ),
                            "binding": rule.get("binding", True),
                        })

                if holding_violations:
                    violations.append({
                        "holding_id": holding.get("holding_id", ""),
                        "issuer_name": holding.get("issuer_name", ""),
                        "isin": holding.get("isin", ""),
                        "portfolio_weight_pct": holding.get(
                            "portfolio_weight_pct", 0.0
                        ),
                        "market_value_eur": holding.get(
                            "market_value_eur", 0.0
                        ),
                        "violations": holding_violations,
                        "verdict": ScreeningVerdict.FAIL.value,
                    })
                    excluded_weight += holding.get(
                        "portfolio_weight_pct", 0.0
                    )
                    excluded_value += holding.get("market_value_eur", 0.0)
                else:
                    pass_holdings.append(holding)

            outputs["violations"] = violations
            outputs["violations_count"] = len(violations)
            outputs["pass_count"] = len(pass_holdings)
            outputs["pass_holdings"] = pass_holdings
            outputs["excluded_weight_pct"] = round(excluded_weight, 2)
            outputs["excluded_value_eur"] = round(excluded_value, 2)

            # Count by severity
            critical_count = sum(
                1 for v in violations
                if any(
                    viol.get("severity") == ViolationSeverity.CRITICAL.value
                    for viol in v.get("violations", [])
                )
            )
            high_count = sum(
                1 for v in violations
                if any(
                    viol.get("severity") == ViolationSeverity.HIGH.value
                    for viol in v.get("violations", [])
                )
            )
            outputs["critical_violations"] = critical_count
            outputs["high_violations"] = high_count

            if critical_count > 0:
                warnings.append(
                    f"{critical_count} holding(s) have CRITICAL exclusion "
                    f"violations requiring immediate action"
                )

            status = PhaseStatus.COMPLETED
            records = len(holdings)

        except Exception as exc:
            logger.error("NegativeScreening failed: %s", exc, exc_info=True)
            errors.append(f"Negative screening failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )


class PositiveScreeningPhase:
    """
    Phase 3: Positive Screening.

    Applies E/S promotion criteria including minimum ESG rating,
    sustainability indicators, and environmental performance thresholds.
    """

    PHASE_NAME = "positive_screening"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute positive screening phase.

        Args:
            context: Workflow context with holdings that passed negative screening.

        Returns:
            PhaseResult with positive screening scores and categorization.
        """
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            neg_output = context.get_phase_output("negative_screening")
            holdings = neg_output.get("pass_holdings", [])
            positive_rules = config.get("positive_rules", [])
            min_esg = config.get("minimum_esg_rating")

            pass_positive: List[Dict[str, Any]] = []
            fail_positive: List[Dict[str, Any]] = []
            watchlist: List[Dict[str, Any]] = []

            for holding in holdings:
                score = 0.0
                max_score = 0.0
                rule_results: List[Dict[str, Any]] = []

                for rule in positive_rules:
                    field = rule.get("field", "")
                    operator = rule.get("operator", "")
                    threshold = rule.get("threshold", 0.0)
                    weight = rule.get("weight", 1.0)
                    value = holding.get(field)
                    max_score += weight

                    passed = False
                    if value is not None:
                        if operator == "gte" and isinstance(
                            value, (int, float)
                        ):
                            passed = value >= threshold
                        elif operator == "gt" and isinstance(
                            value, (int, float)
                        ):
                            passed = value > threshold
                        elif operator == "lte" and isinstance(
                            value, (int, float)
                        ):
                            passed = value <= threshold
                        elif operator == "lt" and isinstance(
                            value, (int, float)
                        ):
                            passed = value < threshold
                        elif operator == "eq":
                            passed = value == threshold

                    if passed:
                        score += weight

                    rule_results.append({
                        "rule_name": rule.get("name", ""),
                        "field": field,
                        "actual_value": value,
                        "threshold": threshold,
                        "passed": passed,
                        "weight": weight,
                    })

                # ESG rating check
                esg_passed = True
                if min_esg is not None:
                    esg_rating = holding.get("esg_rating")
                    if esg_rating is None or esg_rating < min_esg:
                        esg_passed = False

                # Calculate overall score percentage
                score_pct = (score / max_score * 100) if max_score > 0 else 0.0

                screening_result = {
                    "holding_id": holding.get("holding_id", ""),
                    "issuer_name": holding.get("issuer_name", ""),
                    "isin": holding.get("isin", ""),
                    "portfolio_weight_pct": holding.get(
                        "portfolio_weight_pct", 0.0
                    ),
                    "positive_score": round(score, 2),
                    "max_score": round(max_score, 2),
                    "score_pct": round(score_pct, 1),
                    "esg_rating": holding.get("esg_rating"),
                    "esg_passed": esg_passed,
                    "rule_results": rule_results,
                }

                if score_pct >= 70.0 and esg_passed:
                    screening_result["verdict"] = ScreeningVerdict.PASS.value
                    pass_positive.append(screening_result)
                elif score_pct >= 40.0:
                    screening_result["verdict"] = (
                        ScreeningVerdict.WATCHLIST.value
                    )
                    watchlist.append(screening_result)
                else:
                    screening_result["verdict"] = ScreeningVerdict.FAIL.value
                    fail_positive.append(screening_result)

            outputs["pass_positive"] = pass_positive
            outputs["fail_positive"] = fail_positive
            outputs["watchlist"] = watchlist
            outputs["pass_count"] = len(pass_positive)
            outputs["fail_count"] = len(fail_positive)
            outputs["watchlist_count"] = len(watchlist)

            # Weight distribution
            pass_weight = sum(
                h.get("portfolio_weight_pct", 0.0) for h in pass_positive
            )
            fail_weight = sum(
                h.get("portfolio_weight_pct", 0.0) for h in fail_positive
            )
            watch_weight = sum(
                h.get("portfolio_weight_pct", 0.0) for h in watchlist
            )
            outputs["pass_weight_pct"] = round(pass_weight, 2)
            outputs["fail_weight_pct"] = round(fail_weight, 2)
            outputs["watchlist_weight_pct"] = round(watch_weight, 2)

            if len(fail_positive) > 0:
                warnings.append(
                    f"{len(fail_positive)} holding(s) ({fail_weight:.1f}% "
                    f"weight) fail positive screening criteria"
                )

            status = PhaseStatus.COMPLETED
            records = len(holdings)

        except Exception as exc:
            logger.error("PositiveScreening failed: %s", exc, exc_info=True)
            errors.append(f"Positive screening failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )


class ComplianceCheckPhase:
    """
    Phase 4: Compliance Check.

    Verifies the portfolio meets all Article 8 binding elements,
    calculates overall compliance percentage, and generates the
    screening report.
    """

    PHASE_NAME = "compliance_check"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute compliance check phase.

        Args:
            context: Workflow context with screening results.

        Returns:
            PhaseResult with compliance assessment and screening report.
        """
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            universe_output = context.get_phase_output("universe_definition")
            neg_output = context.get_phase_output("negative_screening")
            pos_output = context.get_phase_output("positive_screening")
            min_compliance = config.get("minimum_compliance_pct", 90.0)

            total_screened = universe_output.get(
                "screened_holdings_count", 0
            )
            neg_violations = neg_output.get("violations_count", 0)
            pos_pass = pos_output.get("pass_count", 0)
            pos_fail = pos_output.get("fail_count", 0)
            watchlist = pos_output.get("watchlist_count", 0)
            critical_violations = neg_output.get("critical_violations", 0)

            # Calculate compliance percentage
            compliant_count = pos_pass
            non_compliant_count = neg_violations + pos_fail
            total = max(total_screened, 1)
            compliance_pct = round(compliant_count / total * 100, 2)

            # Weight-based compliance
            excluded_weight = neg_output.get("excluded_weight_pct", 0.0)
            pass_weight = pos_output.get("pass_weight_pct", 0.0)
            total_weight = universe_output.get(
                "total_screened_weight_pct", 100.0
            )
            weight_compliance_pct = round(
                pass_weight / max(total_weight, 0.01) * 100, 2
            )

            outputs["compliance_pct"] = compliance_pct
            outputs["weight_compliance_pct"] = weight_compliance_pct
            outputs["minimum_compliance_pct"] = min_compliance
            outputs["is_compliant"] = (
                weight_compliance_pct >= min_compliance
                and critical_violations == 0
            )

            # Binding elements check
            binding_checks = []

            # Check: No critical exclusion violations
            binding_checks.append({
                "element": "No Controversial Weapons",
                "passed": critical_violations == 0,
                "detail": (
                    f"{critical_violations} critical violation(s)"
                ),
            })

            # Check: Minimum compliance threshold
            binding_checks.append({
                "element": "Minimum Compliance Threshold",
                "passed": weight_compliance_pct >= min_compliance,
                "detail": (
                    f"{weight_compliance_pct:.1f}% vs "
                    f"{min_compliance:.1f}% required"
                ),
            })

            # Check: ESG rating threshold
            min_esg = config.get("minimum_esg_rating")
            if min_esg is not None:
                esg_failures = sum(
                    1 for h in pos_output.get("fail_positive", [])
                    + pos_output.get("watchlist", [])
                    if not h.get("esg_passed", True)
                )
                binding_checks.append({
                    "element": f"Minimum ESG Rating ({min_esg})",
                    "passed": esg_failures == 0,
                    "detail": f"{esg_failures} holding(s) below threshold",
                })

            outputs["binding_checks"] = binding_checks
            outputs["all_binding_met"] = all(
                c["passed"] for c in binding_checks
            )

            # Generate screening report
            outputs["screening_report"] = {
                "report_id": str(uuid.uuid4()),
                "generated_at": _utcnow().isoformat(),
                "product_name": config.get("product_name", ""),
                "screening_date": config.get("screening_date", ""),
                "scope": config.get(
                    "screening_scope",
                    ScreeningScope.FULL_PORTFOLIO.value
                ),
                "universe_size": total_screened,
                "negative_screening": {
                    "violations": neg_violations,
                    "critical": critical_violations,
                    "excluded_weight_pct": excluded_weight,
                },
                "positive_screening": {
                    "pass": pos_pass,
                    "fail": pos_fail,
                    "watchlist": watchlist,
                    "pass_weight_pct": pass_weight,
                },
                "compliance": {
                    "by_count_pct": compliance_pct,
                    "by_weight_pct": weight_compliance_pct,
                    "minimum_required_pct": min_compliance,
                    "is_compliant": outputs["is_compliant"],
                    "binding_elements_met": outputs["all_binding_met"],
                },
                "action_required": not outputs["is_compliant"],
            }

            if not outputs["is_compliant"]:
                warnings.append(
                    f"Portfolio is NOT compliant. Compliance: "
                    f"{weight_compliance_pct:.1f}% "
                    f"(required: {min_compliance:.1f}%)"
                )

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("ComplianceCheck failed: %s", exc, exc_info=True)
            errors.append(f"Compliance check failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )


# =============================================================================
# WORKFLOW ORCHESTRATOR
# =============================================================================


class PortfolioScreeningWorkflow:
    """
    Four-phase portfolio screening workflow for SFDR Article 8.

    Orchestrates universe definition, negative screening, positive
    screening, and compliance checking. Supports configurable exclusion
    rules and positive criteria with weighted scoring.

    Example:
        >>> wf = PortfolioScreeningWorkflow()
        >>> input_data = PortfolioScreeningInput(
        ...     organization_id="org-123",
        ...     product_name="Green Bond Fund",
        ...     screening_date="2026-01-15",
        ... )
        >>> result = await wf.run(input_data)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    WORKFLOW_NAME = "portfolio_screening"

    PHASE_ORDER = [
        "universe_definition",
        "negative_screening",
        "positive_screening",
        "compliance_check",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """Initialize the portfolio screening workflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "universe_definition": UniverseDefinitionPhase(),
            "negative_screening": NegativeScreeningPhase(),
            "positive_screening": PositiveScreeningPhase(),
            "compliance_check": ComplianceCheckPhase(),
        }

    async def run(
        self, input_data: PortfolioScreeningInput
    ) -> PortfolioScreeningResult:
        """
        Execute the complete 4-phase portfolio screening workflow.

        Args:
            input_data: Validated workflow input configuration.

        Returns:
            PortfolioScreeningResult with per-phase details and summary.
        """
        started_at = _utcnow()
        logger.info(
            "Starting portfolio screening workflow %s for org=%s product=%s",
            self.workflow_id, input_data.organization_id,
            input_data.product_name,
        )

        context = WorkflowContext(
            workflow_id=self.workflow_id,
            organization_id=input_data.organization_id,
            config=self._build_config(input_data),
        )

        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        for idx, phase_name in enumerate(self.PHASE_ORDER):
            if phase_name in input_data.skip_phases:
                skip_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.SKIPPED,
                    provenance_hash=_hash_data({"skipped": True}),
                )
                completed_phases.append(skip_result)
                context.mark_phase(phase_name, PhaseStatus.SKIPPED)
                continue

            if context.is_phase_completed(phase_name):
                continue

            pct = idx / len(self.PHASE_ORDER)
            self._notify_progress(phase_name, f"Starting: {phase_name}", pct)
            context.mark_phase(phase_name, PhaseStatus.RUNNING)

            try:
                phase_executor = self._phases[phase_name]
                phase_result = await phase_executor.execute(context)
                completed_phases.append(phase_result)

                if phase_result.status == PhaseStatus.COMPLETED:
                    context.set_phase_output(phase_name, phase_result.outputs)
                    context.mark_phase(phase_name, PhaseStatus.COMPLETED)
                else:
                    context.mark_phase(phase_name, phase_result.status)
                    if phase_name == "universe_definition":
                        overall_status = WorkflowStatus.FAILED
                        break

                context.errors.extend(phase_result.errors)
                context.warnings.extend(phase_result.warnings)

            except Exception as exc:
                logger.error(
                    "Phase '%s' raised unhandled exception: %s",
                    phase_name, exc, exc_info=True,
                )
                error_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    started_at=_utcnow(),
                    errors=[str(exc)],
                    provenance_hash=_hash_data({"error": str(exc)}),
                )
                completed_phases.append(error_result)
                context.mark_phase(phase_name, PhaseStatus.FAILED)
                overall_status = WorkflowStatus.FAILED
                break

        if overall_status == WorkflowStatus.RUNNING:
            all_ok = all(
                p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                for p in completed_phases
            )
            overall_status = (
                WorkflowStatus.COMPLETED if all_ok else WorkflowStatus.PARTIAL
            )

        completed_at = _utcnow()
        total_duration = (completed_at - started_at).total_seconds()
        summary = self._build_summary(context)
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })

        self._notify_progress(
            "workflow", f"Workflow {overall_status.value}", 1.0
        )

        return PortfolioScreeningResult(
            workflow_id=self.workflow_id,
            workflow_name=self.WORKFLOW_NAME,
            status=overall_status,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=total_duration,
            phases=completed_phases,
            summary=summary,
            provenance_hash=provenance,
            product_name=summary.get("product_name", ""),
            total_holdings_screened=summary.get(
                "total_holdings_screened", 0
            ),
            exclusion_violations=summary.get("exclusion_violations", 0),
            critical_violations=summary.get("critical_violations", 0),
            positive_screen_pass=summary.get("positive_screen_pass", 0),
            positive_screen_fail=summary.get("positive_screen_fail", 0),
            compliance_pct=summary.get("compliance_pct", 0.0),
            is_compliant=summary.get("is_compliant", False),
            watchlist_count=summary.get("watchlist_count", 0),
        )

    def _build_config(
        self, input_data: PortfolioScreeningInput
    ) -> Dict[str, Any]:
        """Transform input model to config dict for phases."""
        config = input_data.model_dump()
        config["screening_scope"] = input_data.screening_scope.value
        if input_data.holdings:
            config["holdings"] = [
                h.model_dump() for h in input_data.holdings
            ]
        if input_data.exclusion_rules:
            config["exclusion_rules"] = [
                r.model_dump() for r in input_data.exclusion_rules
            ]
            for r in config["exclusion_rules"]:
                r["severity"] = r["severity"].value if isinstance(
                    r["severity"], ViolationSeverity
                ) else r["severity"]
        if input_data.positive_rules:
            config["positive_rules"] = [
                r.model_dump() for r in input_data.positive_rules
            ]
        return config

    def _build_summary(self, context: WorkflowContext) -> Dict[str, Any]:
        """Build workflow summary from phase outputs."""
        config = context.config
        universe = context.get_phase_output("universe_definition")
        neg = context.get_phase_output("negative_screening")
        pos = context.get_phase_output("positive_screening")
        compliance = context.get_phase_output("compliance_check")

        return {
            "product_name": config.get("product_name", ""),
            "total_holdings_screened": universe.get(
                "screened_holdings_count", 0
            ),
            "exclusion_violations": neg.get("violations_count", 0),
            "critical_violations": neg.get("critical_violations", 0),
            "positive_screen_pass": pos.get("pass_count", 0),
            "positive_screen_fail": pos.get("fail_count", 0),
            "watchlist_count": pos.get("watchlist_count", 0),
            "compliance_pct": compliance.get(
                "weight_compliance_pct", 0.0
            ),
            "is_compliant": compliance.get("is_compliant", False),
        }

    def _notify_progress(
        self, phase: str, message: str, pct: float
    ) -> None:
        """Send progress notification via callback if registered."""
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for phase=%s", phase)
