# -*- coding: utf-8 -*-
"""
Climate Risk Workflow
==============================

6-phase workflow for physical and transition risk assessment per ESRS E1-9.
Implements risk identification, quantification, opportunity assessment,
scenario analysis, financial aggregation, and report generation.

Phases:
    1. RiskIdentification     -- Identify physical and transition risks
    2. RiskQuantification     -- Quantify risk likelihood and magnitude
    3. OpportunityAssessment  -- Identify climate-related opportunities
    4. ScenarioAnalysis       -- Run scenario-based risk analysis
    5. FinancialAggregation   -- Aggregate financial effects
    6. ReportGeneration       -- Produce E1-9 disclosure data

Author: GreenLang Team
Version: 16.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    """Compute SHA-256 hex digest of *data*."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class WorkflowPhase(str, Enum):
    """Phases of the climate risk workflow."""
    RISK_IDENTIFICATION = "risk_identification"
    RISK_QUANTIFICATION = "risk_quantification"
    OPPORTUNITY_ASSESSMENT = "opportunity_assessment"
    SCENARIO_ANALYSIS = "scenario_analysis"
    FINANCIAL_AGGREGATION = "financial_aggregation"
    REPORT_GENERATION = "report_generation"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PhaseStatus(str, Enum):
    """Status of a single phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class RiskCategory(str, Enum):
    """Climate risk category."""
    PHYSICAL_ACUTE = "physical_acute"
    PHYSICAL_CHRONIC = "physical_chronic"
    TRANSITION_POLICY = "transition_policy"
    TRANSITION_TECHNOLOGY = "transition_technology"
    TRANSITION_MARKET = "transition_market"
    TRANSITION_REPUTATION = "transition_reputation"
    TRANSITION_LEGAL = "transition_legal"


class TimeHorizon(str, Enum):
    """Risk time horizon."""
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"


class LikelihoodLevel(str, Enum):
    """Risk likelihood level."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class MagnitudeLevel(str, Enum):
    """Risk magnitude level."""
    NEGLIGIBLE = "negligible"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class FinancialEffectType(str, Enum):
    """Financial effect type."""
    REVENUE_IMPACT = "revenue_impact"
    COST_INCREASE = "cost_increase"
    ASSET_IMPAIRMENT = "asset_impairment"
    CAPITAL_EXPENDITURE = "capital_expenditure"
    STRANDED_ASSETS = "stranded_assets"
    INSURANCE_COST = "insurance_cost"
    LITIGATION = "litigation"
    REVENUE_OPPORTUNITY = "revenue_opportunity"
    COST_SAVINGS = "cost_savings"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class ClimateRisk(BaseModel):
    """A climate risk record."""
    risk_id: str = Field(default_factory=lambda: f"cr-{_new_uuid()[:8]}")
    name: str = Field(..., description="Risk name")
    description: str = Field(default="")
    category: RiskCategory = Field(..., description="Risk category")
    time_horizon: TimeHorizon = Field(default=TimeHorizon.MEDIUM_TERM)
    likelihood: LikelihoodLevel = Field(default=LikelihoodLevel.MEDIUM)
    magnitude: MagnitudeLevel = Field(default=MagnitudeLevel.MODERATE)
    likelihood_score: float = Field(default=0.0, ge=0.0, le=5.0)
    magnitude_score: float = Field(default=0.0, ge=0.0, le=5.0)
    composite_risk_score: float = Field(default=0.0, ge=0.0, le=25.0)
    affected_assets_eur: float = Field(default=0.0, ge=0.0)
    estimated_financial_impact_eur: float = Field(default=0.0)
    mitigation_status: str = Field(default="unmitigated")
    location: str = Field(default="")
    scenario_sensitivity: Dict[str, float] = Field(default_factory=dict)


class ClimateOpportunity(BaseModel):
    """A climate-related opportunity."""
    opportunity_id: str = Field(default_factory=lambda: f"co-{_new_uuid()[:8]}")
    name: str = Field(..., description="Opportunity name")
    description: str = Field(default="")
    category: str = Field(default="", description="resource_efficiency, products, markets, resilience")
    time_horizon: TimeHorizon = Field(default=TimeHorizon.MEDIUM_TERM)
    likelihood: LikelihoodLevel = Field(default=LikelihoodLevel.MEDIUM)
    estimated_value_eur: float = Field(default=0.0, ge=0.0)
    investment_required_eur: float = Field(default=0.0, ge=0.0)
    roi_pct: float = Field(default=0.0)


class FinancialEffect(BaseModel):
    """Aggregated financial effect from climate risks/opportunities."""
    effect_type: FinancialEffectType = Field(...)
    time_horizon: TimeHorizon = Field(default=TimeHorizon.MEDIUM_TERM)
    amount_eur: float = Field(default=0.0)
    is_risk: bool = Field(default=True)
    source_ids: List[str] = Field(default_factory=list)
    description: str = Field(default="")


class ClimateRiskInput(BaseModel):
    """Input data model for ClimateRiskWorkflow."""
    risks: List[ClimateRisk] = Field(
        default_factory=list, description="Identified climate risks"
    )
    opportunities: List[ClimateOpportunity] = Field(
        default_factory=list, description="Identified opportunities"
    )
    total_assets_eur: float = Field(
        default=0.0, ge=0.0, description="Total assets for exposure calc"
    )
    revenue_eur: float = Field(
        default=0.0, ge=0.0, description="Annual revenue"
    )
    scenarios: List[str] = Field(
        default_factory=lambda: ["rcp2.6", "rcp4.5", "rcp8.5"],
        description="Climate scenarios to analyze"
    )
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)


class ClimateRiskResult(BaseModel):
    """Complete result from climate risk workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="climate_risk")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, description="Number of phases completed")
    duration_ms: float = Field(default=0.0, description="Total duration in milliseconds")
    total_duration_seconds: float = Field(default=0.0)
    risks: List[ClimateRisk] = Field(default_factory=list)
    opportunities: List[ClimateOpportunity] = Field(default_factory=list)
    financial_effects: List[FinancialEffect] = Field(default_factory=list)
    total_risks: int = Field(default=0)
    physical_risks: int = Field(default=0)
    transition_risks: int = Field(default=0)
    total_opportunities: int = Field(default=0)
    total_risk_exposure_eur: float = Field(default=0.0)
    total_opportunity_value_eur: float = Field(default=0.0)
    net_financial_impact_eur: float = Field(default=0.0)
    high_risk_count: int = Field(default=0)
    assets_at_risk_pct: float = Field(default=0.0)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")


# =============================================================================
# SCORING MAPS
# =============================================================================

LIKELIHOOD_SCORES: Dict[str, float] = {
    "very_low": 1.0,
    "low": 2.0,
    "medium": 3.0,
    "high": 4.0,
    "very_high": 5.0,
}

MAGNITUDE_SCORES: Dict[str, float] = {
    "negligible": 1.0,
    "low": 2.0,
    "moderate": 3.0,
    "high": 4.0,
    "very_high": 5.0,
}

# Scenario multipliers for risk scoring
SCENARIO_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "rcp2.6": {"physical_acute": 1.0, "physical_chronic": 1.0, "transition_policy": 1.5, "transition_technology": 1.3},
    "rcp4.5": {"physical_acute": 1.3, "physical_chronic": 1.4, "transition_policy": 1.2, "transition_technology": 1.1},
    "rcp8.5": {"physical_acute": 1.8, "physical_chronic": 2.0, "transition_policy": 0.8, "transition_technology": 0.9},
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class ClimateRiskWorkflow:
    """
    6-phase climate risk assessment workflow for ESRS E1-9.

    Implements physical and transition risk identification, quantification
    with likelihood/magnitude scoring, opportunity assessment, scenario
    analysis (RCP 2.6/4.5/8.5), financial aggregation, and disclosure-ready
    output generation.

    Zero-hallucination: all risk scores use deterministic scoring maps.

    Example:
        >>> wf = ClimateRiskWorkflow()
        >>> inp = ClimateRiskInput(risks=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.total_risks >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize ClimateRiskWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._risks: List[ClimateRisk] = []
        self._opportunities: List[ClimateOpportunity] = []
        self._financial_effects: List[FinancialEffect] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.RISK_IDENTIFICATION.value, "description": "Identify physical and transition risks"},
            {"name": WorkflowPhase.RISK_QUANTIFICATION.value, "description": "Quantify risk likelihood and magnitude"},
            {"name": WorkflowPhase.OPPORTUNITY_ASSESSMENT.value, "description": "Identify climate-related opportunities"},
            {"name": WorkflowPhase.SCENARIO_ANALYSIS.value, "description": "Run scenario-based risk analysis"},
            {"name": WorkflowPhase.FINANCIAL_AGGREGATION.value, "description": "Aggregate financial effects"},
            {"name": WorkflowPhase.REPORT_GENERATION.value, "description": "Produce E1-9 disclosure data"},
        ]

    def validate_inputs(self, input_data: ClimateRiskInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.risks and not input_data.opportunities:
            issues.append("No risks or opportunities provided")
        if input_data.total_assets_eur <= 0:
            issues.append("Total assets must be positive for exposure calculation")
        return issues

    async def execute(
        self,
        input_data: Optional[ClimateRiskInput] = None,
        risks: Optional[List[ClimateRisk]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ClimateRiskResult:
        """
        Execute the 6-phase climate risk workflow.

        Args:
            input_data: Full input model (preferred).
            risks: Climate risks (fallback).
            config: Configuration overrides.

        Returns:
            ClimateRiskResult with quantified risks, opportunities, and financial effects.
        """
        if input_data is None:
            input_data = ClimateRiskInput(
                risks=risks or [],
                config=config or {},
            )

        started_at = _utcnow()
        self.logger.info("Starting climate risk workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS

        try:
            phase_results.append(await self._phase_risk_identification(input_data))
            phase_results.append(await self._phase_risk_quantification(input_data))
            phase_results.append(await self._phase_opportunity_assessment(input_data))
            phase_results.append(await self._phase_scenario_analysis(input_data))
            phase_results.append(await self._phase_financial_aggregation(input_data))
            phase_results.append(await self._phase_report_generation(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Climate risk workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()
        completed_count = sum(1 for p in phase_results if p.status == PhaseStatus.COMPLETED)
        physical = sum(1 for r in self._risks if r.category.value.startswith("physical"))
        transition = sum(1 for r in self._risks if r.category.value.startswith("transition"))
        total_exposure = sum(abs(r.estimated_financial_impact_eur) for r in self._risks)
        total_opportunity = sum(o.estimated_value_eur for o in self._opportunities)
        high_risk = sum(1 for r in self._risks if r.composite_risk_score >= 16.0)
        assets_at_risk = sum(r.affected_assets_eur for r in self._risks)
        assets_pct = round(
            (assets_at_risk / input_data.total_assets_eur * 100)
            if input_data.total_assets_eur > 0 else 0.0, 2
        )

        result = ClimateRiskResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=completed_count,
            duration_ms=round(elapsed * 1000, 2),
            total_duration_seconds=elapsed,
            risks=self._risks,
            opportunities=self._opportunities,
            financial_effects=self._financial_effects,
            total_risks=len(self._risks),
            physical_risks=physical,
            transition_risks=transition,
            total_opportunities=len(self._opportunities),
            total_risk_exposure_eur=round(total_exposure, 2),
            total_opportunity_value_eur=round(total_opportunity, 2),
            net_financial_impact_eur=round(total_opportunity - total_exposure, 2),
            high_risk_count=high_risk,
            assets_at_risk_pct=assets_pct,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Climate risk %s completed in %.2fs: %d risks (%d high), %d opportunities",
            self.workflow_id, elapsed, len(self._risks), high_risk, len(self._opportunities),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Risk Identification
    # -------------------------------------------------------------------------

    async def _phase_risk_identification(
        self, input_data: ClimateRiskInput,
    ) -> PhaseResult:
        """Identify physical and transition climate risks."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._risks = list(input_data.risks)

        cat_counts: Dict[str, int] = {}
        horizon_counts: Dict[str, int] = {}
        for r in self._risks:
            cat_counts[r.category.value] = cat_counts.get(r.category.value, 0) + 1
            horizon_counts[r.time_horizon.value] = horizon_counts.get(r.time_horizon.value, 0) + 1

        outputs["risks_identified"] = len(self._risks)
        outputs["category_distribution"] = cat_counts
        outputs["time_horizon_distribution"] = horizon_counts

        physical = sum(1 for r in self._risks if r.category.value.startswith("physical"))
        transition = sum(1 for r in self._risks if r.category.value.startswith("transition"))
        outputs["physical_risks"] = physical
        outputs["transition_risks"] = transition

        if not self._risks:
            warnings.append("No climate risks identified")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 RiskIdentification: %d risks (%d physical, %d transition)",
            len(self._risks), physical, transition,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.RISK_IDENTIFICATION.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Risk Quantification
    # -------------------------------------------------------------------------

    async def _phase_risk_quantification(
        self, input_data: ClimateRiskInput,
    ) -> PhaseResult:
        """Quantify risk likelihood and magnitude scores."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        for risk in self._risks:
            risk.likelihood_score = LIKELIHOOD_SCORES.get(risk.likelihood.value, 3.0)
            risk.magnitude_score = MAGNITUDE_SCORES.get(risk.magnitude.value, 3.0)
            risk.composite_risk_score = round(risk.likelihood_score * risk.magnitude_score, 2)

        if self._risks:
            composites = [r.composite_risk_score for r in self._risks]
            outputs["avg_risk_score"] = round(sum(composites) / len(composites), 2)
            outputs["max_risk_score"] = max(composites)
            outputs["min_risk_score"] = min(composites)
            outputs["high_risk_count"] = sum(1 for c in composites if c >= 16.0)
            outputs["critical_risk_count"] = sum(1 for c in composites if c >= 20.0)
        else:
            outputs["avg_risk_score"] = 0.0
            outputs["max_risk_score"] = 0.0
            outputs["min_risk_score"] = 0.0
            outputs["high_risk_count"] = 0
            outputs["critical_risk_count"] = 0

        outputs["risks_quantified"] = len(self._risks)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Phase 2 RiskQuantification: %d risks scored", len(self._risks))
        return PhaseResult(
            phase_name=WorkflowPhase.RISK_QUANTIFICATION.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Opportunity Assessment
    # -------------------------------------------------------------------------

    async def _phase_opportunity_assessment(
        self, input_data: ClimateRiskInput,
    ) -> PhaseResult:
        """Identify and assess climate-related opportunities."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._opportunities = list(input_data.opportunities)

        cat_counts: Dict[str, int] = {}
        for o in self._opportunities:
            cat_counts[o.category] = cat_counts.get(o.category, 0) + 1

        total_value = sum(o.estimated_value_eur for o in self._opportunities)
        total_investment = sum(o.investment_required_eur for o in self._opportunities)

        outputs["opportunities_identified"] = len(self._opportunities)
        outputs["category_distribution"] = cat_counts
        outputs["total_estimated_value_eur"] = round(total_value, 2)
        outputs["total_investment_required_eur"] = round(total_investment, 2)
        outputs["avg_roi_pct"] = round(
            sum(o.roi_pct for o in self._opportunities) / len(self._opportunities)
            if self._opportunities else 0.0, 1
        )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 OpportunityAssessment: %d opportunities, %.0f EUR value",
            len(self._opportunities), total_value,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.OPPORTUNITY_ASSESSMENT.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Scenario Analysis
    # -------------------------------------------------------------------------

    async def _phase_scenario_analysis(
        self, input_data: ClimateRiskInput,
    ) -> PhaseResult:
        """Run scenario-based risk analysis."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        scenario_impacts: Dict[str, Dict[str, float]] = {}

        for scenario in input_data.scenarios:
            multipliers = SCENARIO_MULTIPLIERS.get(scenario, {})
            scenario_total = 0.0

            for risk in self._risks:
                cat_key = risk.category.value
                multiplier = multipliers.get(cat_key, 1.0)
                adjusted_impact = risk.estimated_financial_impact_eur * multiplier
                scenario_total += adjusted_impact

                # Store per-risk scenario sensitivity
                risk.scenario_sensitivity[scenario] = round(adjusted_impact, 2)

            scenario_impacts[scenario] = {
                "total_impact_eur": round(scenario_total, 2),
                "impact_as_pct_assets": round(
                    (scenario_total / input_data.total_assets_eur * 100)
                    if input_data.total_assets_eur > 0 else 0.0, 2
                ),
            }

        outputs["scenarios_analyzed"] = len(input_data.scenarios)
        outputs["scenario_impacts"] = scenario_impacts

        # Check worst-case scenario
        if scenario_impacts:
            worst_scenario = max(
                scenario_impacts.items(),
                key=lambda x: abs(x[1].get("total_impact_eur", 0))
            )
            outputs["worst_case_scenario"] = worst_scenario[0]
            outputs["worst_case_impact_eur"] = worst_scenario[1]["total_impact_eur"]

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Phase 4 ScenarioAnalysis: %d scenarios analyzed", len(input_data.scenarios))
        return PhaseResult(
            phase_name=WorkflowPhase.SCENARIO_ANALYSIS.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Financial Aggregation
    # -------------------------------------------------------------------------

    async def _phase_financial_aggregation(
        self, input_data: ClimateRiskInput,
    ) -> PhaseResult:
        """Aggregate financial effects from risks and opportunities."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._financial_effects = []

        # Aggregate risk financial effects by type
        risk_by_type: Dict[str, float] = {}
        risk_sources: Dict[str, List[str]] = {}
        for risk in self._risks:
            if risk.estimated_financial_impact_eur != 0:
                effect_type = "cost_increase" if risk.category.value.startswith("transition") else "asset_impairment"
                risk_by_type[effect_type] = risk_by_type.get(effect_type, 0.0) + risk.estimated_financial_impact_eur
                risk_sources.setdefault(effect_type, []).append(risk.risk_id)

        for effect_type, amount in risk_by_type.items():
            self._financial_effects.append(FinancialEffect(
                effect_type=FinancialEffectType(effect_type),
                amount_eur=round(amount, 2),
                is_risk=True,
                source_ids=risk_sources.get(effect_type, []),
                description=f"Aggregated {effect_type} from climate risks",
            ))

        # Aggregate opportunity financial effects
        for opp in self._opportunities:
            if opp.estimated_value_eur > 0:
                self._financial_effects.append(FinancialEffect(
                    effect_type=FinancialEffectType.REVENUE_OPPORTUNITY,
                    amount_eur=round(opp.estimated_value_eur, 2),
                    is_risk=False,
                    source_ids=[opp.opportunity_id],
                    description=f"Opportunity: {opp.name}",
                ))

        total_risk = sum(e.amount_eur for e in self._financial_effects if e.is_risk)
        total_opp = sum(e.amount_eur for e in self._financial_effects if not e.is_risk)

        outputs["financial_effects_count"] = len(self._financial_effects)
        outputs["total_risk_financial_impact_eur"] = round(total_risk, 2)
        outputs["total_opportunity_value_eur"] = round(total_opp, 2)
        outputs["net_impact_eur"] = round(total_opp - total_risk, 2)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 5 FinancialAggregation: risk=%.0f EUR, opportunity=%.0f EUR",
            total_risk, total_opp,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.FINANCIAL_AGGREGATION.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 6: Report Generation
    # -------------------------------------------------------------------------

    async def _phase_report_generation(
        self, input_data: ClimateRiskInput,
    ) -> PhaseResult:
        """Generate E1-9 disclosure-ready output."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        physical = sum(1 for r in self._risks if r.category.value.startswith("physical"))
        transition = sum(1 for r in self._risks if r.category.value.startswith("transition"))

        outputs["e1_9_disclosure"] = {
            "total_risks": len(self._risks),
            "physical_risks": physical,
            "transition_risks": transition,
            "total_opportunities": len(self._opportunities),
            "high_risk_count": sum(1 for r in self._risks if r.composite_risk_score >= 16.0),
            "total_risk_exposure_eur": round(
                sum(abs(r.estimated_financial_impact_eur) for r in self._risks), 2
            ),
            "total_opportunity_value_eur": round(
                sum(o.estimated_value_eur for o in self._opportunities), 2
            ),
            "assets_at_risk_pct": round(
                (sum(r.affected_assets_eur for r in self._risks)
                 / input_data.total_assets_eur * 100)
                if input_data.total_assets_eur > 0 else 0.0, 2
            ),
            "scenarios_analyzed": len(input_data.scenarios),
            "reporting_year": input_data.reporting_year,
        }

        outputs["report_ready"] = True

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Phase 6 ReportGeneration: E1-9 disclosure ready")
        return PhaseResult(
            phase_name=WorkflowPhase.REPORT_GENERATION.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: ClimateRiskResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
