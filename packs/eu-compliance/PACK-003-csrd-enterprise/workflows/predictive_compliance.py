# -*- coding: utf-8 -*-
"""
Predictive Compliance Workflow
=================================

5-phase AI-driven compliance forecasting workflow. Analyzes historical ESG
data, models trends, predicts compliance gaps, scores risks, and generates
intervention action plans with ROI estimates.

Phases:
    1. Historical Analysis: Load and normalize 3-5 years of ESG data
    2. Trend Modeling: Fit regression, ARIMA, and ensemble models to key metrics
    3. Gap Prediction: Forecast future compliance gaps with confidence intervals
    4. Risk Scoring: Score non-compliance risk (CRITICAL/HIGH/MEDIUM/LOW)
    5. Action Planning: Generate intervention recommendations with estimated ROI

Author: GreenLang Team
Version: 3.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class RiskLevel(str, Enum):
    """Compliance risk level classification."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class TrendDirection(str, Enum):
    """Direction of a metric trend."""

    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    VOLATILE = "volatile"


class InterventionPriority(str, Enum):
    """Priority classification for intervention actions."""

    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration in seconds")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class ComplianceTarget(BaseModel):
    """A compliance target to evaluate against."""

    metric_id: str = Field(..., description="Metric identifier")
    metric_name: str = Field(default="", description="Human-readable metric name")
    framework: str = Field(default="CSRD", description="Regulatory framework")
    target_value: float = Field(..., description="Target compliance value")
    target_year: int = Field(..., ge=2024, le=2060, description="Year target must be met")
    unit: str = Field(default="tCO2e", description="Measurement unit")
    direction: str = Field(
        default="decrease", description="decrease or increase to meet target"
    )


class ConfidenceInterval(BaseModel):
    """Confidence interval for a projection."""

    lower: float = Field(..., description="Lower bound")
    central: float = Field(..., description="Central estimate (median/mean)")
    upper: float = Field(..., description="Upper bound")
    confidence_level: float = Field(
        default=0.95, ge=0.5, le=0.99, description="Confidence level (e.g. 0.95 = 95%)"
    )


class MetricProjection(BaseModel):
    """Projected future value of a compliance metric."""

    metric_id: str = Field(..., description="Metric identifier")
    year: int = Field(..., description="Projection year")
    projection: ConfidenceInterval = Field(..., description="Projected value with CI")
    trend_direction: TrendDirection = Field(..., description="Trend direction")
    model_used: str = Field(default="ensemble", description="Model used for projection")
    r_squared: float = Field(default=0.0, ge=0.0, le=1.0, description="Model fit R-squared")


class GapPrediction(BaseModel):
    """Predicted compliance gap for a metric."""

    metric_id: str = Field(..., description="Metric identifier")
    target_year: int = Field(..., description="Target year")
    target_value: float = Field(..., description="Required target value")
    projected_value: ConfidenceInterval = Field(..., description="Projected value at target year")
    gap_size: float = Field(..., description="Gap between projected and target (positive = shortfall)")
    probability_of_non_compliance: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of missing target"
    )
    risk_level: RiskLevel = Field(..., description="Risk classification based on probability")


class InterventionAction(BaseModel):
    """Recommended intervention to close a compliance gap."""

    action_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    metric_id: str = Field(..., description="Target metric this addresses")
    description: str = Field(..., description="Action description")
    priority: InterventionPriority = Field(..., description="Priority classification")
    estimated_impact: float = Field(..., description="Estimated impact on metric")
    estimated_cost_eur: float = Field(default=0.0, ge=0.0, description="Estimated cost in EUR")
    estimated_roi_pct: float = Field(default=0.0, description="Estimated ROI percentage")
    implementation_months: int = Field(default=6, ge=1, description="Implementation timeline")
    confidence: float = Field(default=0.7, ge=0.0, le=1.0, description="Confidence in estimate")


class PredictiveComplianceInput(BaseModel):
    """Input for predictive compliance workflow."""

    entity_id: str = Field(..., description="Entity to analyze")
    tenant_id: str = Field(default="", description="Tenant isolation ID")
    years_of_data: int = Field(default=3, ge=1, le=10, description="Years of historical data")
    targets: List[ComplianceTarget] = Field(
        ..., min_length=1, description="Compliance targets to evaluate"
    )
    projection_horizon_years: int = Field(
        default=5, ge=1, le=30, description="Years to project forward"
    )
    confidence_level: float = Field(
        default=0.95, ge=0.5, le=0.99, description="Confidence level for intervals"
    )
    include_monte_carlo: bool = Field(
        default=True, description="Run Monte Carlo simulation for risk scoring"
    )
    monte_carlo_iterations: int = Field(
        default=10000, ge=1000, le=100000, description="Monte Carlo iterations"
    )


class PredictiveComplianceResult(BaseModel):
    """Complete result from predictive compliance workflow."""

    workflow_id: str = Field(..., description="Unique workflow execution ID")
    workflow_name: str = Field(default="predictive_compliance")
    status: WorkflowStatus = Field(..., description="Overall workflow status")
    phases: List[PhaseResult] = Field(default_factory=list, description="Per-phase results")
    total_duration_seconds: float = Field(default=0.0, description="Total duration")
    entity_id: str = Field(default="", description="Entity analyzed")
    projections: List[MetricProjection] = Field(
        default_factory=list, description="Metric projections"
    )
    gap_predictions: List[GapPrediction] = Field(
        default_factory=list, description="Gap predictions"
    )
    overall_risk_level: RiskLevel = Field(
        default=RiskLevel.LOW, description="Overall compliance risk"
    )
    intervention_actions: List[InterventionAction] = Field(
        default_factory=list, description="Recommended interventions"
    )
    total_estimated_investment_eur: float = Field(
        default=0.0, description="Total estimated investment needed"
    )
    weighted_average_roi_pct: float = Field(
        default=0.0, description="Weighted average ROI across interventions"
    )
    provenance_hash: str = Field(default="", description="SHA-256 of complete output")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class PredictiveComplianceWorkflow:
    """
    5-phase AI-driven compliance forecasting workflow.

    Analyzes historical ESG data, models trends, predicts future compliance
    gaps, scores non-compliance risk with confidence intervals, and generates
    intervention action plans with estimated ROI.

    Risk scoring thresholds:
        CRITICAL: >90% probability of non-compliance
        HIGH: 70-90% probability
        MEDIUM: 40-70% probability
        LOW: <40% probability

    Attributes:
        workflow_id: Unique execution identifier.
        config: Optional EnterprisePackConfig.
        _context: Accumulated workflow context.

    Example:
        >>> workflow = PredictiveComplianceWorkflow()
        >>> targets = [ComplianceTarget(
        ...     metric_id="scope1_emissions", target_value=1000.0,
        ...     target_year=2030, direction="decrease"
        ... )]
        >>> input_data = PredictiveComplianceInput(
        ...     entity_id="entity-001", years_of_data=5, targets=targets
        ... )
        >>> result = await workflow.execute(input_data)
        >>> assert result.overall_risk_level in RiskLevel
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """
        Initialize the predictive compliance workflow.

        Args:
            config: Optional EnterprisePackConfig for engine resolution.
        """
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._context: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self, input_data: PredictiveComplianceInput
    ) -> PredictiveComplianceResult:
        """
        Execute the 5-phase predictive compliance workflow.

        Args:
            input_data: Validated input with entity, targets, and parameters.

        Returns:
            PredictiveComplianceResult with projections, gaps, risk scores,
            and intervention actions.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting predictive compliance workflow %s for entity=%s targets=%d",
            self.workflow_id, input_data.entity_id, len(input_data.targets),
        )

        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        try:
            # Phase 1: Historical Analysis
            p1 = await self._phase_1_historical_analysis(input_data)
            phase_results.append(p1)
            if p1.status == PhaseStatus.FAILED:
                overall_status = WorkflowStatus.FAILED
                raise RuntimeError("Historical analysis failed")

            # Phase 2: Trend Modeling
            p2 = await self._phase_2_trend_modeling(input_data)
            phase_results.append(p2)
            if p2.status == PhaseStatus.FAILED:
                overall_status = WorkflowStatus.FAILED
                raise RuntimeError("Trend modeling failed")

            # Phase 3: Gap Prediction
            p3 = await self._phase_3_gap_prediction(input_data)
            phase_results.append(p3)
            if p3.status == PhaseStatus.FAILED:
                overall_status = WorkflowStatus.FAILED
                raise RuntimeError("Gap prediction failed")

            # Phase 4: Risk Scoring
            p4 = await self._phase_4_risk_scoring(input_data)
            phase_results.append(p4)

            # Phase 5: Action Planning
            p5 = await self._phase_5_action_planning(input_data)
            phase_results.append(p5)

            overall_status = WorkflowStatus.COMPLETED

        except RuntimeError:
            pass
        except Exception as exc:
            self.logger.critical(
                "Predictive compliance workflow %s failed: %s",
                self.workflow_id, str(exc), exc_info=True,
            )
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="workflow_error",
                status=PhaseStatus.FAILED,
                errors=[str(exc)],
                provenance_hash=self._hash_data({"error": str(exc)}),
            ))

        completed_at = datetime.utcnow()
        total_duration = (completed_at - started_at).total_seconds()

        projections = self._context.get("projections", [])
        gap_predictions = self._context.get("gap_predictions", [])
        interventions = self._context.get("interventions", [])
        overall_risk = self._context.get("overall_risk", RiskLevel.LOW)

        total_investment = sum(a.estimated_cost_eur for a in interventions)
        weighted_roi = self._compute_weighted_roi(interventions)

        provenance = self._hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in phase_results],
        })

        self.logger.info(
            "Predictive compliance workflow %s finished status=%s risk=%s in %.1fs",
            self.workflow_id, overall_status.value, overall_risk.value
            if isinstance(overall_risk, RiskLevel) else overall_risk, total_duration,
        )

        return PredictiveComplianceResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            total_duration_seconds=total_duration,
            entity_id=input_data.entity_id,
            projections=projections,
            gap_predictions=gap_predictions,
            overall_risk_level=overall_risk,
            intervention_actions=interventions,
            total_estimated_investment_eur=total_investment,
            weighted_average_roi_pct=weighted_roi,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 1: Historical Analysis
    # -------------------------------------------------------------------------

    async def _phase_1_historical_analysis(
        self, input_data: PredictiveComplianceInput
    ) -> PhaseResult:
        """
        Load and normalize historical ESG data for the entity.

        Retrieves 3-5 years of historical metric data, normalizes units,
        fills gaps using imputation, and validates data quality.

        Agents invoked:
            - greenlang.agents.data.time_series_gap_filler
            - greenlang.agents.data.data_quality_profiler
            - greenlang.agents.foundation.unit_normalizer

        Steps:
            1. Load historical data for each target metric
            2. Normalize units to standard measurement system
            3. Fill gaps using time series imputation
            4. Profile data quality and flag insufficient history
        """
        phase_name = "historical_analysis"
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Step 1: Load historical data
        historical_data: Dict[str, List[Dict[str, Any]]] = {}
        for target in input_data.targets:
            data = await self._load_historical_metric(
                input_data.entity_id, target.metric_id,
                input_data.years_of_data,
            )
            historical_data[target.metric_id] = data
            if len(data) < 2:
                warnings.append(
                    f"Insufficient history for {target.metric_id}: "
                    f"{len(data)} data points (minimum 2 required)"
                )

        outputs["metrics_loaded"] = len(historical_data)
        outputs["total_data_points"] = sum(len(d) for d in historical_data.values())

        # Step 2: Normalize units
        normalized = await self._normalize_units(historical_data)
        outputs["units_normalized"] = True

        # Step 3: Fill gaps
        gap_filled = await self._fill_time_series_gaps(normalized)
        outputs["gaps_filled"] = gap_filled.get("gaps_filled", 0)

        # Step 4: Quality profile
        quality = await self._profile_historical_quality(gap_filled)
        outputs["data_quality_score"] = quality.get("score", 0.0)
        outputs["metrics_with_sufficient_data"] = quality.get("sufficient_count", 0)

        if quality.get("score", 0.0) < 50.0:
            errors.append(
                f"Historical data quality too low for reliable forecasting: "
                f"{quality.get('score', 0.0):.1f}%"
            )

        self._context["historical_data"] = gap_filled.get("data", historical_data)

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        duration = (datetime.utcnow() - started_at).total_seconds()

        return PhaseResult(
            phase_name=phase_name,
            status=status,
            duration_seconds=duration,
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=self._hash_data(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Trend Modeling
    # -------------------------------------------------------------------------

    async def _phase_2_trend_modeling(
        self, input_data: PredictiveComplianceInput
    ) -> PhaseResult:
        """
        Fit statistical models to historical data for each metric.

        Trains regression, ARIMA, and ensemble models on each metric's
        historical data. Selects the best model based on cross-validation
        and generates multi-year projections with confidence intervals.

        Agents invoked:
            - greenlang.engines.predictive.trend_analyzer
            - greenlang.engines.predictive.model_selector

        Steps:
            1. Fit linear and polynomial regression models
            2. Fit ARIMA/seasonal decomposition models
            3. Train ensemble (weighted average of best models)
            4. Cross-validate and select best model per metric
            5. Generate projections with confidence intervals
        """
        phase_name = "trend_modeling"
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        historical = self._context.get("historical_data", {})
        current_year = datetime.utcnow().year
        projection_end = current_year + input_data.projection_horizon_years

        projections: List[MetricProjection] = []
        model_performance: Dict[str, Dict[str, float]] = {}

        for target in input_data.targets:
            metric_data = historical.get(target.metric_id, [])

            # Step 1-3: Fit models
            models = await self._fit_models(target.metric_id, metric_data)
            best_model = models.get("best_model", "ensemble")
            r_squared = models.get("r_squared", 0.0)
            model_performance[target.metric_id] = {
                "best_model": best_model,
                "r_squared": r_squared,
            }

            if r_squared < 0.3:
                warnings.append(
                    f"Low model fit for {target.metric_id} (R²={r_squared:.2f}). "
                    f"Projections may be unreliable."
                )

            # Step 4-5: Generate projections
            for year in range(current_year + 1, projection_end + 1):
                projection = await self._project_metric(
                    target.metric_id, year, best_model,
                    input_data.confidence_level,
                )
                projections.append(MetricProjection(
                    metric_id=target.metric_id,
                    year=year,
                    projection=ConfidenceInterval(
                        lower=projection["lower"],
                        central=projection["central"],
                        upper=projection["upper"],
                        confidence_level=input_data.confidence_level,
                    ),
                    trend_direction=TrendDirection(projection.get("trend", "stable")),
                    model_used=best_model,
                    r_squared=r_squared,
                ))

        outputs["projections_generated"] = len(projections)
        outputs["model_performance"] = model_performance
        outputs["projection_horizon"] = f"{current_year + 1}-{projection_end}"

        self._context["projections"] = projections
        self._context["model_performance"] = model_performance

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        duration = (datetime.utcnow() - started_at).total_seconds()

        return PhaseResult(
            phase_name=phase_name,
            status=status,
            duration_seconds=duration,
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=self._hash_data(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Gap Prediction
    # -------------------------------------------------------------------------

    async def _phase_3_gap_prediction(
        self, input_data: PredictiveComplianceInput
    ) -> PhaseResult:
        """
        Forecast future compliance gaps with confidence intervals.

        Compares projected metric values at target years against compliance
        targets to identify shortfalls. Calculates probability of non-compliance
        using the confidence interval distribution.

        Steps:
            1. Match projections to compliance targets
            2. Calculate gap size (projected vs. target)
            3. Compute probability of non-compliance from CI
            4. Classify risk level based on probability
        """
        phase_name = "gap_prediction"
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        projections = self._context.get("projections", [])
        gap_predictions: List[GapPrediction] = []

        for target in input_data.targets:
            # Step 1: Find projection at target year
            matching = [
                p for p in projections
                if p.metric_id == target.metric_id and p.year == target.target_year
            ]

            if not matching:
                warnings.append(
                    f"No projection found for {target.metric_id} at year {target.target_year}"
                )
                continue

            proj = matching[0]

            # Step 2: Calculate gap
            if target.direction == "decrease":
                gap = proj.projection.central - target.target_value
                prob = self._calculate_exceedance_probability(
                    proj.projection, target.target_value, "above"
                )
            else:
                gap = target.target_value - proj.projection.central
                prob = self._calculate_exceedance_probability(
                    proj.projection, target.target_value, "below"
                )

            # Step 3-4: Risk classification
            risk = self._classify_risk(prob)

            gap_predictions.append(GapPrediction(
                metric_id=target.metric_id,
                target_year=target.target_year,
                target_value=target.target_value,
                projected_value=proj.projection,
                gap_size=max(0.0, gap),
                probability_of_non_compliance=prob,
                risk_level=risk,
            ))

        outputs["gaps_predicted"] = len(gap_predictions)
        outputs["gaps_with_shortfall"] = sum(1 for g in gap_predictions if g.gap_size > 0)
        outputs["critical_gaps"] = sum(1 for g in gap_predictions if g.risk_level == RiskLevel.CRITICAL)
        outputs["high_gaps"] = sum(1 for g in gap_predictions if g.risk_level == RiskLevel.HIGH)

        self._context["gap_predictions"] = gap_predictions

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        duration = (datetime.utcnow() - started_at).total_seconds()

        return PhaseResult(
            phase_name=phase_name,
            status=status,
            duration_seconds=duration,
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=self._hash_data(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Risk Scoring
    # -------------------------------------------------------------------------

    async def _phase_4_risk_scoring(
        self, input_data: PredictiveComplianceInput
    ) -> PhaseResult:
        """
        Score overall and per-metric compliance risk.

        Aggregates gap predictions into an overall risk score using Monte Carlo
        simulation (when enabled) to account for correlation between metrics.

        Steps:
            1. Run Monte Carlo simulation across all metrics
            2. Compute joint probability of non-compliance
            3. Calculate portfolio risk score
            4. Determine overall risk level
        """
        phase_name = "risk_scoring"
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        gaps = self._context.get("gap_predictions", [])

        # Step 1: Monte Carlo (if enabled)
        if input_data.include_monte_carlo and gaps:
            mc_result = await self._run_monte_carlo(
                gaps, input_data.monte_carlo_iterations
            )
            outputs["monte_carlo_iterations"] = input_data.monte_carlo_iterations
            outputs["joint_non_compliance_probability"] = mc_result.get(
                "joint_probability", 0.0
            )
            outputs["worst_case_gap_tco2e"] = mc_result.get("worst_case_gap", 0.0)
            outputs["expected_gap_tco2e"] = mc_result.get("expected_gap", 0.0)
            joint_prob = mc_result.get("joint_probability", 0.0)
        else:
            # Simple aggregation fallback
            if gaps:
                joint_prob = 1.0 - (
                    1.0
                    if not gaps
                    else max(g.probability_of_non_compliance for g in gaps)
                )
                joint_prob = max(g.probability_of_non_compliance for g in gaps)
            else:
                joint_prob = 0.0
            outputs["monte_carlo_iterations"] = 0
            outputs["joint_non_compliance_probability"] = joint_prob

        # Step 2-3: Overall risk
        overall_risk = self._classify_risk(joint_prob)
        outputs["overall_risk_level"] = overall_risk.value
        outputs["overall_risk_probability"] = joint_prob

        # Per-metric risk summary
        outputs["per_metric_risk"] = {
            g.metric_id: {
                "risk_level": g.risk_level.value,
                "probability": g.probability_of_non_compliance,
                "gap_size": g.gap_size,
            }
            for g in gaps
        }

        self._context["overall_risk"] = overall_risk

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        duration = (datetime.utcnow() - started_at).total_seconds()

        return PhaseResult(
            phase_name=phase_name,
            status=status,
            duration_seconds=duration,
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=self._hash_data(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Action Planning
    # -------------------------------------------------------------------------

    async def _phase_5_action_planning(
        self, input_data: PredictiveComplianceInput
    ) -> PhaseResult:
        """
        Generate intervention recommendations with estimated ROI.

        For each predicted compliance gap, generates actionable recommendations
        with cost estimates, expected impact, implementation timeline, and
        estimated return on investment.

        Steps:
            1. Generate intervention options for each gap
            2. Estimate cost and impact for each option
            3. Calculate ROI and payback period
            4. Prioritize interventions by ROI and urgency
            5. Compile intervention portfolio summary
        """
        phase_name = "action_planning"
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        gaps = self._context.get("gap_predictions", [])
        interventions: List[InterventionAction] = []

        for gap in gaps:
            if gap.gap_size <= 0:
                continue

            # Step 1-3: Generate interventions
            actions = await self._generate_interventions(gap)
            for action_data in actions:
                action = InterventionAction(
                    metric_id=gap.metric_id,
                    description=action_data.get("description", ""),
                    priority=InterventionPriority(action_data.get("priority", "medium_term")),
                    estimated_impact=action_data.get("impact", 0.0),
                    estimated_cost_eur=action_data.get("cost_eur", 0.0),
                    estimated_roi_pct=action_data.get("roi_pct", 0.0),
                    implementation_months=action_data.get("months", 6),
                    confidence=action_data.get("confidence", 0.7),
                )
                interventions.append(action)

        # Step 4: Sort by ROI descending
        interventions.sort(key=lambda a: a.estimated_roi_pct, reverse=True)

        # Step 5: Portfolio summary
        outputs["interventions_generated"] = len(interventions)
        outputs["total_estimated_cost_eur"] = sum(a.estimated_cost_eur for a in interventions)
        outputs["total_estimated_impact"] = sum(a.estimated_impact for a in interventions)
        outputs["immediate_actions"] = sum(
            1 for a in interventions if a.priority == InterventionPriority.IMMEDIATE
        )
        outputs["short_term_actions"] = sum(
            1 for a in interventions if a.priority == InterventionPriority.SHORT_TERM
        )

        if not interventions and any(g.gap_size > 0 for g in gaps):
            warnings.append("Gaps identified but no interventions could be generated")

        self._context["interventions"] = interventions

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        duration = (datetime.utcnow() - started_at).total_seconds()

        return PhaseResult(
            phase_name=phase_name,
            status=status,
            duration_seconds=duration,
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=self._hash_data(outputs),
        )

    # -------------------------------------------------------------------------
    # Risk Classification
    # -------------------------------------------------------------------------

    def _classify_risk(self, probability: float) -> RiskLevel:
        """
        Classify risk level based on probability of non-compliance.

        Thresholds:
            CRITICAL: >90%
            HIGH: 70-90%
            MEDIUM: 40-70%
            LOW: <40%
        """
        if probability > 0.90:
            return RiskLevel.CRITICAL
        elif probability > 0.70:
            return RiskLevel.HIGH
        elif probability > 0.40:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _calculate_exceedance_probability(
        self, ci: ConfidenceInterval, threshold: float, direction: str
    ) -> float:
        """
        Calculate probability of exceeding (or falling below) a threshold.

        Uses the CI to estimate the probability assuming approximate normal
        distribution between lower and upper bounds.
        """
        if ci.upper == ci.lower:
            if direction == "above":
                return 1.0 if ci.central > threshold else 0.0
            else:
                return 1.0 if ci.central < threshold else 0.0

        # Approximate: fraction of CI range beyond threshold
        total_range = ci.upper - ci.lower
        if direction == "above":
            if threshold <= ci.lower:
                return 0.95
            elif threshold >= ci.upper:
                return 0.05
            else:
                fraction_above = (ci.upper - threshold) / total_range
                return min(0.95, max(0.05, fraction_above))
        else:
            if threshold >= ci.upper:
                return 0.95
            elif threshold <= ci.lower:
                return 0.05
            else:
                fraction_below = (threshold - ci.lower) / total_range
                return min(0.95, max(0.05, fraction_below))

    def _compute_weighted_roi(self, interventions: List[InterventionAction]) -> float:
        """Compute cost-weighted average ROI across interventions."""
        total_cost = sum(a.estimated_cost_eur for a in interventions)
        if total_cost == 0:
            return 0.0
        weighted = sum(
            a.estimated_roi_pct * a.estimated_cost_eur for a in interventions
        )
        return weighted / total_cost

    # -------------------------------------------------------------------------
    # Agent Simulation Stubs
    # -------------------------------------------------------------------------

    async def _load_historical_metric(
        self, entity_id: str, metric_id: str, years: int
    ) -> List[Dict[str, Any]]:
        """Load historical data points for a metric."""
        current_year = datetime.utcnow().year
        base_value = 5000.0
        data = []
        for y in range(current_year - years, current_year):
            data.append({
                "year": y,
                "value": base_value * (1.0 - 0.03 * (y - (current_year - years))),
                "unit": "tCO2e",
            })
        return data

    async def _normalize_units(
        self, data: Dict[str, List[Dict]]
    ) -> Dict[str, List[Dict]]:
        """Normalize measurement units to standard system."""
        return data

    async def _fill_time_series_gaps(self, data: Any) -> Dict[str, Any]:
        """Fill gaps in time series data."""
        return {"data": data, "gaps_filled": 0}

    async def _profile_historical_quality(self, data: Any) -> Dict[str, Any]:
        """Profile historical data quality."""
        return {"score": 88.0, "sufficient_count": 5}

    async def _fit_models(
        self, metric_id: str, data: List[Dict]
    ) -> Dict[str, Any]:
        """Fit regression, ARIMA, and ensemble models to a metric."""
        return {"best_model": "ensemble", "r_squared": 0.85}

    async def _project_metric(
        self, metric_id: str, year: int, model: str, confidence: float
    ) -> Dict[str, float]:
        """Project a metric value at a future year."""
        base = 4000.0
        offset = (year - datetime.utcnow().year) * 150.0
        central = base - offset
        margin = abs(central) * (1 - confidence) * 2
        return {
            "central": central,
            "lower": central - margin,
            "upper": central + margin,
            "trend": "declining",
        }

    async def _run_monte_carlo(
        self, gaps: List[GapPrediction], iterations: int
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation for joint risk assessment."""
        max_prob = max(g.probability_of_non_compliance for g in gaps) if gaps else 0.0
        return {
            "joint_probability": max_prob,
            "worst_case_gap": sum(g.gap_size * 1.5 for g in gaps),
            "expected_gap": sum(g.gap_size * g.probability_of_non_compliance for g in gaps),
            "iterations_run": iterations,
        }

    async def _generate_interventions(
        self, gap: GapPrediction
    ) -> List[Dict[str, Any]]:
        """Generate intervention options for a compliance gap."""
        actions = []
        if gap.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH):
            actions.append({
                "description": f"Immediate emission reduction program for {gap.metric_id}",
                "priority": "immediate",
                "impact": gap.gap_size * 0.5,
                "cost_eur": gap.gap_size * 100.0,
                "roi_pct": 150.0,
                "months": 3,
                "confidence": 0.8,
            })
        actions.append({
            "description": f"Technology upgrade to reduce {gap.metric_id}",
            "priority": "short_term",
            "impact": gap.gap_size * 0.3,
            "cost_eur": gap.gap_size * 200.0,
            "roi_pct": 250.0,
            "months": 12,
            "confidence": 0.7,
        })
        actions.append({
            "description": f"Supply chain engagement for {gap.metric_id} reduction",
            "priority": "medium_term",
            "impact": gap.gap_size * 0.2,
            "cost_eur": gap.gap_size * 50.0,
            "roi_pct": 300.0,
            "months": 18,
            "confidence": 0.6,
        })
        return actions

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _hash_data(self, data: Any) -> str:
        """Calculate SHA-256 hash for provenance tracking."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
