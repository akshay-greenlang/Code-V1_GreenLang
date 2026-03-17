# -*- coding: utf-8 -*-
"""
Financial Assessment Workflow
=================================

4-phase workflow for double materiality financial assessment within PACK-015
Double Materiality Pack. Implements ESRS 1 Chapter 3 outside-in
(financial materiality) analysis per EFRAG IG-1.

Phases:
    1. FinancialDataCollection  -- Gather financial exposure data
    2. RiskOpportunityMapping   -- Map to financial KPIs and risk categories
    3. FinancialScoring         -- Score magnitude, likelihood, time horizon
    4. FinancialRanking         -- Rank and filter by threshold

Author: GreenLang Team
Version: 15.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

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


class FinancialRiskType(str, Enum):
    """Financial risk classification aligned with TCFD/ESRS."""
    TRANSITION_POLICY = "transition_policy"
    TRANSITION_TECHNOLOGY = "transition_technology"
    TRANSITION_MARKET = "transition_market"
    TRANSITION_REPUTATION = "transition_reputation"
    PHYSICAL_ACUTE = "physical_acute"
    PHYSICAL_CHRONIC = "physical_chronic"
    OPPORTUNITY_RESOURCE = "opportunity_resource"
    OPPORTUNITY_ENERGY = "opportunity_energy"
    OPPORTUNITY_PRODUCTS = "opportunity_products"
    OPPORTUNITY_MARKETS = "opportunity_markets"
    OPPORTUNITY_RESILIENCE = "opportunity_resilience"


class TimeHorizon(str, Enum):
    """Time horizon for financial materiality."""
    SHORT_TERM = "short_term"       # 0-1 year
    MEDIUM_TERM = "medium_term"     # 1-5 years
    LONG_TERM = "long_term"         # 5+ years


class FinancialKPI(str, Enum):
    """Financial KPIs potentially affected by sustainability matters."""
    REVENUE = "revenue"
    COST_OF_GOODS_SOLD = "cogs"
    OPERATING_EXPENSES = "opex"
    CAPITAL_EXPENDITURE = "capex"
    ASSET_VALUE = "asset_value"
    ACCESS_TO_CAPITAL = "access_to_capital"
    INSURANCE_COST = "insurance_cost"
    LITIGATION_COST = "litigation_cost"
    MARKET_SHARE = "market_share"
    BRAND_VALUE = "brand_value"


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


class FinancialExposure(BaseModel):
    """Financial exposure data for a sustainability matter."""
    exposure_id: str = Field(default_factory=lambda: f"fe-{uuid.uuid4().hex[:8]}")
    matter_name: str = Field(..., description="Sustainability matter name")
    esrs_topic: str = Field(default="", description="ESRS topic code (E1-G1)")
    risk_type: FinancialRiskType = Field(default=FinancialRiskType.TRANSITION_POLICY)
    affected_kpis: List[FinancialKPI] = Field(default_factory=list)
    estimated_impact_eur: float = Field(default=0.0)
    time_horizon: TimeHorizon = Field(default=TimeHorizon.MEDIUM_TERM)
    description: str = Field(default="")
    source: str = Field(default="")
    confidence_pct: float = Field(default=50.0, ge=0.0, le=100.0)


class FinancialScore(BaseModel):
    """Financial materiality score for a single exposure."""
    exposure_id: str = Field(default="")
    magnitude_score: float = Field(default=0.0, ge=0.0, le=5.0)
    likelihood_score: float = Field(default=0.0, ge=0.0, le=5.0)
    time_horizon_score: float = Field(default=0.0, ge=0.0, le=5.0)
    composite_score: float = Field(default=0.0, ge=0.0, le=5.0)
    scoring_method: str = Field(default="weighted_average")
    justification: str = Field(default="")


class RankedFinancialItem(BaseModel):
    """A ranked financial materiality item after filtering."""
    rank: int = Field(default=0, ge=0)
    exposure_id: str = Field(default="")
    matter_name: str = Field(default="")
    esrs_topic: str = Field(default="")
    risk_type: str = Field(default="")
    composite_score: float = Field(default=0.0)
    estimated_impact_eur: float = Field(default=0.0)
    is_material: bool = Field(default=False)
    threshold_applied: float = Field(default=0.0)


class FinancialAssessmentInput(BaseModel):
    """Input data model for FinancialAssessmentWorkflow."""
    financial_exposures: List[FinancialExposure] = Field(
        default_factory=list, description="Financial exposure records"
    )
    company_revenue_eur: float = Field(
        default=0.0, ge=0.0, description="Annual revenue for relative sizing"
    )
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    materiality_threshold: float = Field(
        default=2.5, ge=0.0, le=5.0,
        description="Minimum composite score for financial materiality"
    )
    scoring_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "magnitude": 0.45, "likelihood": 0.35, "time_horizon": 0.20,
        }
    )
    config: Dict[str, Any] = Field(default_factory=dict)


class FinancialAssessmentResult(BaseModel):
    """Complete result from financial assessment workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="financial_assessment")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    exposures_assessed: int = Field(default=0, ge=0)
    material_items: int = Field(default=0, ge=0)
    non_material_items: int = Field(default=0, ge=0)
    total_exposure_eur: float = Field(default=0.0)
    financial_scores: List[FinancialScore] = Field(default_factory=list)
    ranked_items: List[RankedFinancialItem] = Field(default_factory=list)
    risk_type_distribution: Dict[str, int] = Field(default_factory=dict)
    kpi_impact_summary: Dict[str, float] = Field(default_factory=dict)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")


# =============================================================================
# MAGNITUDE REFERENCE DATA
# =============================================================================

# Magnitude thresholds as % of revenue
MAGNITUDE_THRESHOLDS: Dict[str, float] = {
    "very_high": 5.0,     # >5% of revenue
    "high": 2.0,          # 2-5%
    "moderate": 0.5,      # 0.5-2%
    "low": 0.1,           # 0.1-0.5%
    "very_low": 0.0,      # <0.1%
}

# Time horizon multipliers (closer = higher urgency)
TIME_HORIZON_MULTIPLIERS: Dict[str, float] = {
    "short_term": 1.0,
    "medium_term": 0.75,
    "long_term": 0.50,
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class FinancialAssessmentWorkflow:
    """
    4-phase financial materiality assessment workflow.

    Implements the outside-in (financial) dimension of double materiality
    per ESRS 1 Chapter 3. Collects financial exposure data, maps risks
    and opportunities to KPIs, scores magnitude/likelihood/time-horizon,
    and ranks by composite score against a configurable threshold.

    Zero-hallucination: all scoring uses deterministic formulas with
    revenue-relative magnitude sizing. No LLM in numeric paths.

    Example:
        >>> wf = FinancialAssessmentWorkflow()
        >>> inp = FinancialAssessmentInput(financial_exposures=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.material_items >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize FinancialAssessmentWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._exposures: List[FinancialExposure] = []
        self._financial_scores: List[FinancialScore] = []
        self._ranked_items: List[RankedFinancialItem] = []
        self._kpi_impact: Dict[str, float] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(
        self,
        input_data: Optional[FinancialAssessmentInput] = None,
        financial_exposures: Optional[List[FinancialExposure]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> FinancialAssessmentResult:
        """
        Execute the 4-phase financial assessment.

        Args:
            input_data: Full input model (preferred).
            financial_exposures: Financial exposure records (fallback).
            config: Configuration overrides.

        Returns:
            FinancialAssessmentResult with scores, rankings, KPI impact summary.
        """
        if input_data is None:
            input_data = FinancialAssessmentInput(
                financial_exposures=financial_exposures or [],
                config=config or {},
            )

        started_at = datetime.utcnow()
        self.logger.info("Starting financial assessment %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase_results.append(await self._phase_financial_data_collection(input_data))
            phase_results.append(await self._phase_risk_opportunity_mapping(input_data))
            phase_results.append(await self._phase_financial_scoring(input_data))
            phase_results.append(await self._phase_financial_ranking(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error(
                "Financial assessment workflow failed: %s", exc, exc_info=True,
            )
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()
        material_count = sum(1 for r in self._ranked_items if r.is_material)
        risk_dist: Dict[str, int] = {}
        for e in self._exposures:
            risk_dist[e.risk_type.value] = risk_dist.get(e.risk_type.value, 0) + 1

        result = FinancialAssessmentResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            total_duration_seconds=elapsed,
            exposures_assessed=len(self._exposures),
            material_items=material_count,
            non_material_items=len(self._ranked_items) - material_count,
            total_exposure_eur=sum(e.estimated_impact_eur for e in self._exposures),
            financial_scores=self._financial_scores,
            ranked_items=self._ranked_items,
            risk_type_distribution=risk_dist,
            kpi_impact_summary=self._kpi_impact,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Financial assessment %s completed in %.2fs: %d material of %d",
            self.workflow_id, elapsed, material_count, len(self._exposures),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Financial Data Collection
    # -------------------------------------------------------------------------

    async def _phase_financial_data_collection(
        self, input_data: FinancialAssessmentInput,
    ) -> PhaseResult:
        """Gather financial exposure data and validate completeness."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._exposures = list(input_data.financial_exposures)
        total_exposure = sum(e.estimated_impact_eur for e in self._exposures)

        outputs["exposures_collected"] = len(self._exposures)
        outputs["total_estimated_impact_eur"] = round(total_exposure, 2)
        outputs["company_revenue_eur"] = round(input_data.company_revenue_eur, 2)
        outputs["exposure_as_pct_revenue"] = round(
            (total_exposure / input_data.company_revenue_eur * 100)
            if input_data.company_revenue_eur > 0 else 0.0, 2,
        )
        outputs["time_horizon_distribution"] = self._count_by_field(
            self._exposures, "time_horizon",
        )

        if not self._exposures:
            warnings.append("No financial exposures provided; assessment will be empty")
        if input_data.company_revenue_eur <= 0:
            warnings.append(
                "Company revenue not provided; relative magnitude scoring will use defaults"
            )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 FinancialDataCollection: %d exposures, total=%.2f EUR",
            len(self._exposures), total_exposure,
        )
        return PhaseResult(
            phase_name="financial_data_collection", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Risk/Opportunity Mapping
    # -------------------------------------------------------------------------

    async def _phase_risk_opportunity_mapping(
        self, input_data: FinancialAssessmentInput,
    ) -> PhaseResult:
        """Map financial exposures to risk categories and KPIs."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._kpi_impact = {}
        risk_count = 0
        opportunity_count = 0

        for exposure in self._exposures:
            # Classify as risk or opportunity
            if exposure.risk_type.value.startswith("opportunity_"):
                opportunity_count += 1
            else:
                risk_count += 1

            # Aggregate impact by KPI
            for kpi in exposure.affected_kpis:
                kpi_key = kpi.value
                self._kpi_impact[kpi_key] = (
                    self._kpi_impact.get(kpi_key, 0.0) + exposure.estimated_impact_eur
                )

        # Round KPI impacts
        self._kpi_impact = {k: round(v, 2) for k, v in self._kpi_impact.items()}

        outputs["risks_identified"] = risk_count
        outputs["opportunities_identified"] = opportunity_count
        outputs["kpi_impact_summary"] = self._kpi_impact
        outputs["affected_kpi_count"] = len(self._kpi_impact)

        if not self._kpi_impact:
            warnings.append("No KPI impacts mapped; check affected_kpis in exposure data")

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 RiskOpportunityMapping: %d risks, %d opportunities, %d KPIs",
            risk_count, opportunity_count, len(self._kpi_impact),
        )
        return PhaseResult(
            phase_name="risk_opportunity_mapping", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Financial Scoring
    # -------------------------------------------------------------------------

    async def _phase_financial_scoring(
        self, input_data: FinancialAssessmentInput,
    ) -> PhaseResult:
        """Score each financial exposure on magnitude, likelihood, time horizon."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._financial_scores = []

        weights = input_data.scoring_weights
        revenue = input_data.company_revenue_eur

        for exposure in self._exposures:
            score = self._compute_financial_score(exposure, weights, revenue)
            self._financial_scores.append(score)

        if self._financial_scores:
            composites = [s.composite_score for s in self._financial_scores]
            outputs["average_composite"] = round(sum(composites) / len(composites), 3)
            outputs["max_composite"] = round(max(composites), 3)
            outputs["min_composite"] = round(min(composites), 3)
        else:
            outputs["average_composite"] = 0.0
            outputs["max_composite"] = 0.0
            outputs["min_composite"] = 0.0

        outputs["exposures_scored"] = len(self._financial_scores)
        outputs["weights_applied"] = weights

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 FinancialScoring: %d exposures scored, avg=%.3f",
            len(self._financial_scores), outputs["average_composite"],
        )
        return PhaseResult(
            phase_name="financial_scoring", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _compute_financial_score(
        self,
        exposure: FinancialExposure,
        weights: Dict[str, float],
        revenue: float,
    ) -> FinancialScore:
        """
        Compute deterministic financial materiality score.

        Magnitude is derived from exposure as % of revenue.
        Likelihood uses confidence_pct. Time horizon applies urgency weighting.
        """
        magnitude = self._score_magnitude(exposure.estimated_impact_eur, revenue)
        likelihood = self._score_likelihood(exposure.confidence_pct)
        time_horizon = self._score_time_horizon(exposure.time_horizon)

        w_mag = weights.get("magnitude", 0.45)
        w_lik = weights.get("likelihood", 0.35)
        w_th = weights.get("time_horizon", 0.20)
        w_sum = w_mag + w_lik + w_th

        composite = (
            magnitude * w_mag + likelihood * w_lik + time_horizon * w_th
        ) / w_sum if w_sum > 0 else 0.0

        return FinancialScore(
            exposure_id=exposure.exposure_id,
            magnitude_score=round(magnitude, 2),
            likelihood_score=round(likelihood, 2),
            time_horizon_score=round(time_horizon, 2),
            composite_score=round(composite, 2),
            scoring_method="weighted_average",
            justification=(
                f"Magnitude: {magnitude:.2f} (impact/revenue), "
                f"Likelihood: {likelihood:.2f}, "
                f"Time horizon: {exposure.time_horizon.value}"
            ),
        )

    def _score_magnitude(self, impact_eur: float, revenue: float) -> float:
        """Score magnitude based on impact as percentage of revenue."""
        if revenue <= 0:
            # Without revenue context, score by absolute amount
            if impact_eur >= 10_000_000:
                return 5.0
            elif impact_eur >= 5_000_000:
                return 4.0
            elif impact_eur >= 1_000_000:
                return 3.0
            elif impact_eur >= 100_000:
                return 2.0
            else:
                return 1.0

        pct = abs(impact_eur) / revenue * 100
        if pct >= 5.0:
            return 5.0
        elif pct >= 2.0:
            return 4.0
        elif pct >= 0.5:
            return 3.0
        elif pct >= 0.1:
            return 2.0
        else:
            return 1.0

    def _score_likelihood(self, confidence_pct: float) -> float:
        """Convert confidence percentage to 0-5 score."""
        return min(confidence_pct / 20.0, 5.0)

    def _score_time_horizon(self, time_horizon: TimeHorizon) -> float:
        """Score time horizon with closer horizons scoring higher."""
        horizon_scores: Dict[str, float] = {
            "short_term": 5.0,
            "medium_term": 3.5,
            "long_term": 2.0,
        }
        return horizon_scores.get(time_horizon.value, 3.0)

    # -------------------------------------------------------------------------
    # Phase 4: Financial Ranking
    # -------------------------------------------------------------------------

    async def _phase_financial_ranking(
        self, input_data: FinancialAssessmentInput,
    ) -> PhaseResult:
        """Rank financial items by composite score and apply threshold."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._ranked_items = []

        threshold = input_data.materiality_threshold
        exposure_lookup = {e.exposure_id: e for e in self._exposures}

        sorted_scores = sorted(
            self._financial_scores,
            key=lambda s: s.composite_score,
            reverse=True,
        )

        for rank, score in enumerate(sorted_scores, start=1):
            exposure = exposure_lookup.get(score.exposure_id)
            is_material = score.composite_score >= threshold

            self._ranked_items.append(RankedFinancialItem(
                rank=rank,
                exposure_id=score.exposure_id,
                matter_name=exposure.matter_name if exposure else "",
                esrs_topic=exposure.esrs_topic if exposure else "",
                risk_type=exposure.risk_type.value if exposure else "",
                composite_score=score.composite_score,
                estimated_impact_eur=exposure.estimated_impact_eur if exposure else 0.0,
                is_material=is_material,
                threshold_applied=threshold,
            ))

        material_count = sum(1 for r in self._ranked_items if r.is_material)
        material_exposure = sum(
            r.estimated_impact_eur for r in self._ranked_items if r.is_material
        )

        outputs["total_ranked"] = len(self._ranked_items)
        outputs["material_count"] = material_count
        outputs["non_material_count"] = len(self._ranked_items) - material_count
        outputs["threshold_applied"] = threshold
        outputs["material_exposure_eur"] = round(material_exposure, 2)

        if material_count == 0:
            warnings.append(
                "No financial items exceeded materiality threshold. "
                "Consider reviewing threshold or financial exposure data."
            )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 FinancialRanking: %d material of %d (threshold=%.1f)",
            material_count, len(self._ranked_items), threshold,
        )
        return PhaseResult(
            phase_name="financial_ranking", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: FinancialAssessmentResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _count_by_field(self, items: List[Any], field: str) -> Dict[str, int]:
        """Count items by a field value."""
        counts: Dict[str, int] = {}
        for item in items:
            val = getattr(item, field, None)
            key = val.value if hasattr(val, "value") else str(val)
            counts[key] = counts.get(key, 0) + 1
        return counts
