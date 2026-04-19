# -*- coding: utf-8 -*-
"""
Prioritization Workflow
===================================

3-phase workflow for prioritizing quick-win energy efficiency measures
within PACK-033 Quick Wins Identifier Pack.

Phases:
    1. FinancialAnalysis    -- Run PaybackCalculatorEngine on all measures
    2. CarbonAssessment     -- Run CarbonReductionEngine for emissions impact
    3. MultiCriteriaRanking -- Run ImplementationPrioritizerEngine for final ranking

The workflow follows GreenLang zero-hallucination principles: NPV, IRR,
payback, and carbon calculations use deterministic engineering economics
formulas. SHA-256 provenance hashes guarantee auditability.

Schedule: on-demand (follows facility scan)
Estimated duration: 20 minutes

Author: GreenLang Team
Version: 33.0.0
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime
from decimal import Decimal
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


class WeightProfile(str, Enum):
    """Pre-defined weight profiles for multi-criteria ranking."""

    COST_FOCUSED = "cost_focused"
    CARBON_FOCUSED = "carbon_focused"
    BALANCED = "balanced"
    QUICK_PAYBACK = "quick_payback"
    RISK_AVERSE = "risk_averse"


# =============================================================================
# WEIGHT PROFILES (Zero-Hallucination Reference Data)
# =============================================================================

WEIGHT_PROFILES: Dict[str, Dict[str, float]] = {
    "cost_focused": {
        "npv": 0.35,
        "payback": 0.30,
        "savings_cost": 0.20,
        "carbon": 0.10,
        "ease": 0.05,
    },
    "carbon_focused": {
        "npv": 0.15,
        "payback": 0.10,
        "savings_cost": 0.10,
        "carbon": 0.50,
        "ease": 0.15,
    },
    "balanced": {
        "npv": 0.25,
        "payback": 0.20,
        "savings_cost": 0.20,
        "carbon": 0.20,
        "ease": 0.15,
    },
    "quick_payback": {
        "npv": 0.15,
        "payback": 0.45,
        "savings_cost": 0.20,
        "carbon": 0.10,
        "ease": 0.10,
    },
    "risk_averse": {
        "npv": 0.20,
        "payback": 0.25,
        "savings_cost": 0.15,
        "carbon": 0.10,
        "ease": 0.30,
    },
}

# Default emission factor for grid electricity (kgCO2e/kWh)
DEFAULT_GRID_EF: Dict[str, float] = {
    "US": 0.390,
    "EU": 0.275,
    "UK": 0.207,
    "AU": 0.680,
    "IN": 0.820,
    "CN": 0.580,
    "DEFAULT": 0.400,
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_ms: float = Field(default=0.0, description="Phase duration in milliseconds")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class RankedMeasure(BaseModel):
    """A measure with its multi-criteria ranking score."""

    measure_id: str = Field(default="", description="Measure identifier")
    title: str = Field(default="", description="Measure title")
    rank: int = Field(default=0, ge=0, description="Overall rank (1 = best)")
    composite_score: Decimal = Field(default=Decimal("0"), description="Weighted score 0-100")
    npv: Decimal = Field(default=Decimal("0"), description="Net present value")
    irr_pct: Decimal = Field(default=Decimal("0"), description="Internal rate of return %")
    simple_payback_months: Decimal = Field(default=Decimal("0"), ge=0)
    annual_savings_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    annual_savings_cost: Decimal = Field(default=Decimal("0"), ge=0)
    co2e_reduction_tonnes: Decimal = Field(default=Decimal("0"), ge=0)
    implementation_cost: Decimal = Field(default=Decimal("0"), ge=0)
    ease_score: Decimal = Field(default=Decimal("50"), ge=0, le=100)


class ParetoPoint(BaseModel):
    """A point on the Pareto frontier (cost vs benefit)."""

    measure_id: str = Field(default="", description="Measure identifier")
    cost: Decimal = Field(default=Decimal("0"), ge=0)
    benefit: Decimal = Field(default=Decimal("0"), ge=0)
    is_dominated: bool = Field(default=False)


class PrioritizationInput(BaseModel):
    """Input data model for PrioritizationWorkflow."""

    scan_id: str = Field(default="", description="Originating scan ID")
    measures: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of measure dicts with savings/cost data",
    )
    weight_profile: str = Field(default="balanced", description="Weight profile name")
    budget_limit: Optional[Decimal] = Field(default=None, ge=0, description="Max budget")
    region: str = Field(default="DEFAULT", description="Region for emission factors")
    discount_rate_pct: Decimal = Field(default=Decimal("8.0"), ge=0, le=30)
    analysis_years: int = Field(default=10, ge=1, le=30, description="NPV analysis horizon")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("weight_profile")
    @classmethod
    def validate_weight_profile(cls, v: str) -> str:
        """Ensure weight profile is recognized."""
        if v not in WEIGHT_PROFILES:
            raise ValueError(f"Unknown weight_profile '{v}'. Valid: {list(WEIGHT_PROFILES.keys())}")
        return v


class PrioritizationResult(BaseModel):
    """Complete result from prioritization workflow."""

    result_id: str = Field(..., description="Unique result ID")
    ranked_measures: List[RankedMeasure] = Field(default_factory=list)
    pareto_frontier: List[ParetoPoint] = Field(default_factory=list)
    implementation_sequence: Dict[str, Any] = Field(default_factory=dict)
    total_npv: Decimal = Field(default=Decimal("0"))
    portfolio_irr: Decimal = Field(default=Decimal("0"))
    total_investment: Decimal = Field(default=Decimal("0"), ge=0)
    total_annual_savings: Decimal = Field(default=Decimal("0"), ge=0)
    total_co2e_reduction: Decimal = Field(default=Decimal("0"), ge=0)
    weight_profile_used: str = Field(default="balanced")
    phases_completed: List[str] = Field(default_factory=list)
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class PrioritizationWorkflow:
    """
    3-phase prioritization workflow for quick-win measures.

    Performs financial analysis (NPV, IRR, payback), carbon impact
    assessment, and multi-criteria ranking using configurable weight
    profiles. Produces a Pareto frontier and implementation sequence.

    Zero-hallucination: NPV uses standard discounted cash-flow formula,
    IRR uses bisection method, carbon uses region-specific grid emission
    factors. No LLM calls in the numeric computation path.

    Attributes:
        result_id: Unique execution identifier.
        _ranked_measures: Ranked measure list.
        _pareto_frontier: Pareto-optimal points.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = PrioritizationWorkflow()
        >>> inp = PrioritizationInput(measures=[...], weight_profile="balanced")
        >>> result = wf.run(inp)
        >>> assert len(result.ranked_measures) > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize PrioritizationWorkflow."""
        self.result_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._ranked_measures: List[RankedMeasure] = []
        self._pareto_frontier: List[ParetoPoint] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: PrioritizationInput) -> PrioritizationResult:
        """
        Execute the 3-phase prioritization workflow.

        Args:
            input_data: Validated prioritization input.

        Returns:
            PrioritizationResult with ranked measures and Pareto frontier.

        Raises:
            ValueError: If no measures provided.
        """
        t_start = time.perf_counter()
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting prioritization workflow %s measures=%d profile=%s",
            self.result_id, len(input_data.measures), input_data.weight_profile,
        )

        if not input_data.measures:
            raise ValueError("At least one measure is required for prioritization")

        self._phase_results = []
        self._ranked_measures = []
        self._pareto_frontier = []

        try:
            # Phase 1: Financial Analysis
            phase1 = self._phase_financial_analysis(input_data)
            self._phase_results.append(phase1)

            # Phase 2: Carbon Assessment
            phase2 = self._phase_carbon_assessment(input_data)
            self._phase_results.append(phase2)

            # Phase 3: Multi-Criteria Ranking
            phase3 = self._phase_multi_criteria_ranking(input_data)
            self._phase_results.append(phase3)

        except Exception as exc:
            self.logger.error("Prioritization workflow failed: %s", exc, exc_info=True)
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        total_npv = sum(m.npv for m in self._ranked_measures)
        total_investment = sum(m.implementation_cost for m in self._ranked_measures)
        total_savings = sum(m.annual_savings_cost for m in self._ranked_measures)
        total_co2 = sum(m.co2e_reduction_tonnes for m in self._ranked_measures)
        portfolio_irr = self._calculate_portfolio_irr(
            float(total_investment), float(total_savings),
            input_data.analysis_years,
        )

        # Build implementation sequence
        impl_sequence = self._build_implementation_sequence(input_data)

        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        result = PrioritizationResult(
            result_id=self.result_id,
            ranked_measures=self._ranked_measures,
            pareto_frontier=self._pareto_frontier,
            implementation_sequence=impl_sequence,
            total_npv=total_npv,
            portfolio_irr=Decimal(str(round(portfolio_irr, 2))),
            total_investment=total_investment,
            total_annual_savings=total_savings,
            total_co2e_reduction=total_co2,
            weight_profile_used=input_data.weight_profile,
            phases_completed=completed_phases,
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Prioritization workflow %s completed in %.0fms measures=%d NPV=%.0f",
            self.result_id, elapsed_ms, len(self._ranked_measures), float(total_npv),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Financial Analysis
    # -------------------------------------------------------------------------

    def _phase_financial_analysis(
        self, input_data: PrioritizationInput
    ) -> PhaseResult:
        """Run PaybackCalculatorEngine on all measures (NPV, IRR, payback)."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        discount = float(input_data.discount_rate_pct) / 100.0
        years = input_data.analysis_years

        for measure_dict in input_data.measures:
            measure_id = measure_dict.get("measure_id", f"m-{uuid.uuid4().hex[:8]}")
            title = measure_dict.get("title", "Unnamed Measure")
            savings_kwh = float(measure_dict.get("annual_savings_kwh", 0))
            savings_cost = float(measure_dict.get("annual_savings_cost", 0))
            impl_cost = float(measure_dict.get("implementation_cost", 0))
            payback_months = float(measure_dict.get("simple_payback_months", 0))

            # NPV: deterministic discounted cash-flow
            npv = -impl_cost
            for year in range(1, years + 1):
                npv += savings_cost / ((1.0 + discount) ** year)

            # IRR: bisection method
            irr = self._approximate_irr(impl_cost, savings_cost, years)

            # Recalculate payback if not provided
            if payback_months <= 0 and savings_cost > 0:
                payback_months = impl_cost / savings_cost * 12.0

            # Ease score heuristic: lower cost = easier
            ease = max(0.0, min(100.0, 100.0 - (impl_cost / max(savings_cost * 3, 1.0)) * 10.0))

            ranked = RankedMeasure(
                measure_id=measure_id,
                title=title,
                npv=Decimal(str(round(npv, 2))),
                irr_pct=Decimal(str(round(irr, 2))),
                simple_payback_months=Decimal(str(round(payback_months, 1))),
                annual_savings_kwh=Decimal(str(round(savings_kwh, 2))),
                annual_savings_cost=Decimal(str(round(savings_cost, 2))),
                implementation_cost=Decimal(str(round(impl_cost, 2))),
                ease_score=Decimal(str(round(ease, 1))),
            )
            self._ranked_measures.append(ranked)

        outputs["measures_analysed"] = len(self._ranked_measures)
        outputs["total_npv"] = str(sum(m.npv for m in self._ranked_measures))
        outputs["positive_npv_count"] = sum(1 for m in self._ranked_measures if m.npv > 0)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 FinancialAnalysis: %d measures, total NPV=%s",
            len(self._ranked_measures), outputs["total_npv"],
        )
        return PhaseResult(
            phase_name="financial_analysis", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Carbon Assessment
    # -------------------------------------------------------------------------

    def _phase_carbon_assessment(
        self, input_data: PrioritizationInput
    ) -> PhaseResult:
        """Run CarbonReductionEngine for emissions impact on each measure."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        ef = DEFAULT_GRID_EF.get(input_data.region, DEFAULT_GRID_EF["DEFAULT"])

        total_co2 = Decimal("0")
        for measure in self._ranked_measures:
            co2_tonnes = Decimal(str(
                round(float(measure.annual_savings_kwh) * ef / 1000.0, 4)
            ))
            measure.co2e_reduction_tonnes = co2_tonnes
            total_co2 += co2_tonnes

        outputs["emission_factor_kgco2_kwh"] = ef
        outputs["region"] = input_data.region
        outputs["total_co2e_reduction_tonnes"] = str(total_co2)
        outputs["measures_assessed"] = len(self._ranked_measures)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 CarbonAssessment: total CO2e reduction=%.2f tonnes, EF=%.3f",
            float(total_co2), ef,
        )
        return PhaseResult(
            phase_name="carbon_assessment", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Multi-Criteria Ranking
    # -------------------------------------------------------------------------

    def _phase_multi_criteria_ranking(
        self, input_data: PrioritizationInput
    ) -> PhaseResult:
        """Run ImplementationPrioritizerEngine with weighted scoring."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        weights = WEIGHT_PROFILES[input_data.weight_profile]

        # Normalize each criterion to 0-100 scale
        if not self._ranked_measures:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            return PhaseResult(
                phase_name="multi_criteria_ranking", phase_number=3,
                status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
                outputs={"ranked": 0}, warnings=["No measures to rank"],
                provenance_hash=self._hash_dict({"ranked": 0}),
            )

        # Find max values for normalization
        max_npv = max(float(m.npv) for m in self._ranked_measures) or 1.0
        max_savings = max(float(m.annual_savings_cost) for m in self._ranked_measures) or 1.0
        max_co2 = max(float(m.co2e_reduction_tonnes) for m in self._ranked_measures) or 1.0
        max_payback = max(float(m.simple_payback_months) for m in self._ranked_measures) or 1.0

        for measure in self._ranked_measures:
            # NPV score: higher is better
            npv_score = max(0.0, float(measure.npv) / abs(max_npv) * 100.0) if max_npv != 0 else 0.0

            # Payback score: lower payback is better (invert)
            payback_score = max(0.0, (1.0 - float(measure.simple_payback_months) / max_payback) * 100.0)

            # Savings score: higher is better
            savings_score = float(measure.annual_savings_cost) / max_savings * 100.0

            # Carbon score: higher is better
            carbon_score = float(measure.co2e_reduction_tonnes) / max_co2 * 100.0

            # Ease score: already 0-100
            ease_val = float(measure.ease_score)

            composite = (
                weights["npv"] * npv_score
                + weights["payback"] * payback_score
                + weights["savings_cost"] * savings_score
                + weights["carbon"] * carbon_score
                + weights["ease"] * ease_val
            )
            measure.composite_score = Decimal(str(round(composite, 2)))

        # Sort by composite score descending and assign ranks
        self._ranked_measures.sort(key=lambda m: m.composite_score, reverse=True)
        for idx, measure in enumerate(self._ranked_measures, start=1):
            measure.rank = idx

        # Apply budget filter if specified
        if input_data.budget_limit is not None:
            cumulative = Decimal("0")
            for measure in self._ranked_measures:
                cumulative += measure.implementation_cost
                if cumulative > input_data.budget_limit:
                    warnings.append(
                        f"Budget limit {input_data.budget_limit} reached at rank {measure.rank}"
                    )
                    break

        # Build Pareto frontier (cost vs NPV)
        self._pareto_frontier = self._compute_pareto_frontier()

        outputs["measures_ranked"] = len(self._ranked_measures)
        outputs["weight_profile"] = input_data.weight_profile
        outputs["weights"] = weights
        outputs["top_3"] = [
            {"rank": m.rank, "title": m.title, "score": str(m.composite_score)}
            for m in self._ranked_measures[:3]
        ]
        outputs["pareto_points"] = len(self._pareto_frontier)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 MultiCriteriaRanking: %d measures ranked, profile=%s",
            len(self._ranked_measures), input_data.weight_profile,
        )
        return PhaseResult(
            phase_name="multi_criteria_ranking", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Supporting Methods
    # -------------------------------------------------------------------------

    def _compute_pareto_frontier(self) -> List[ParetoPoint]:
        """Compute Pareto frontier of cost vs NPV benefit."""
        points: List[ParetoPoint] = []
        for m in self._ranked_measures:
            points.append(ParetoPoint(
                measure_id=m.measure_id,
                cost=m.implementation_cost,
                benefit=m.npv,
            ))

        # Mark dominated points
        for i, p in enumerate(points):
            for j, q in enumerate(points):
                if i == j:
                    continue
                # q dominates p if q has lower cost AND higher benefit
                if q.cost <= p.cost and q.benefit >= p.benefit and (
                    q.cost < p.cost or q.benefit > p.benefit
                ):
                    p.is_dominated = True
                    break

        return points

    def _build_implementation_sequence(
        self, input_data: PrioritizationInput
    ) -> Dict[str, Any]:
        """Build phased implementation sequence from ranked measures."""
        phases: Dict[str, List[str]] = {
            "immediate_0_3_months": [],
            "short_term_3_6_months": [],
            "medium_term_6_12_months": [],
            "long_term_12_plus_months": [],
        }

        for measure in self._ranked_measures:
            payback = float(measure.simple_payback_months)
            if payback <= 3:
                phases["immediate_0_3_months"].append(measure.measure_id)
            elif payback <= 6:
                phases["short_term_3_6_months"].append(measure.measure_id)
            elif payback <= 12:
                phases["medium_term_6_12_months"].append(measure.measure_id)
            else:
                phases["long_term_12_plus_months"].append(measure.measure_id)

        return phases

    def _approximate_irr(
        self, investment: float, annual_cashflow: float, years: int
    ) -> float:
        """Approximate IRR using bisection method (zero-hallucination)."""
        if investment <= 0 or annual_cashflow <= 0:
            return 0.0

        low, high = 0.0, 5.0
        mid = 0.0
        for _ in range(50):
            mid = (low + high) / 2.0
            npv = -investment + sum(
                annual_cashflow / ((1.0 + mid) ** y) for y in range(1, years + 1)
            )
            if npv > 0:
                low = mid
            else:
                high = mid
        return mid * 100.0

    def _calculate_portfolio_irr(
        self, total_investment: float, total_annual_savings: float, years: int
    ) -> float:
        """Calculate portfolio-level IRR."""
        return self._approximate_irr(total_investment, total_annual_savings, years)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: PrioritizationResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
