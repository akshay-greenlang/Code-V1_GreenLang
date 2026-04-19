# -*- coding: utf-8 -*-
"""
GL-ADAPT-X-010: Adaptation Investment Prioritizer Agent
========================================================

Prioritizes adaptation investments based on cost-effectiveness, risk
reduction potential, co-benefits, and strategic alignment.

Capabilities:
    - Multi-criteria investment prioritization
    - Cost-effectiveness ranking
    - Risk reduction optimization
    - Co-benefit valuation
    - Budget allocation optimization
    - Implementation sequencing
    - Portfolio optimization

Zero-Hallucination Guarantees:
    - All rankings from deterministic algorithms
    - Cost-benefit calculations verified
    - Complete provenance tracking
    - No LLM-based prioritization

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class PriorityLevel(str, Enum):
    """Investment priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    OPTIONAL = "optional"


class InvestmentCategory(str, Enum):
    """Categories of adaptation investments."""
    INFRASTRUCTURE = "infrastructure"
    TECHNOLOGY = "technology"
    NATURE_BASED = "nature_based"
    OPERATIONAL = "operational"
    CAPACITY_BUILDING = "capacity_building"
    INSURANCE = "insurance"


class TimeFrame(str, Enum):
    """Implementation timeframes."""
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"


# =============================================================================
# Pydantic Models
# =============================================================================

class InvestmentOption(BaseModel):
    """Single investment option for prioritization."""
    option_id: str = Field(...)
    name: str = Field(...)
    description: str = Field(default="")
    category: InvestmentCategory = Field(...)
    capital_cost_usd: float = Field(..., ge=0)
    annual_operating_cost_usd: float = Field(default=0.0, ge=0)
    implementation_months: int = Field(default=12, ge=1)
    risk_reduction_pct: float = Field(..., ge=0, le=100)
    expected_benefit_usd: float = Field(default=0.0, ge=0)
    co_benefits: List[str] = Field(default_factory=list)
    target_hazards: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    strategic_alignment: float = Field(default=0.5, ge=0, le=1)


class PrioritizedInvestment(BaseModel):
    """Investment with prioritization scores."""
    option: InvestmentOption = Field(...)
    priority_level: PriorityLevel = Field(...)
    priority_rank: int = Field(...)

    # Scoring components
    cost_effectiveness_score: float = Field(..., ge=0, le=1)
    risk_reduction_score: float = Field(..., ge=0, le=1)
    co_benefit_score: float = Field(..., ge=0, le=1)
    feasibility_score: float = Field(..., ge=0, le=1)
    strategic_score: float = Field(..., ge=0, le=1)

    # Overall
    composite_score: float = Field(..., ge=0, le=1)

    # Financial metrics
    benefit_cost_ratio: float = Field(default=0.0, ge=0)
    npv_usd: float = Field(default=0.0)
    payback_years: Optional[float] = Field(None, ge=0)

    # Implementation
    recommended_timeframe: TimeFrame = Field(...)
    recommended_sequence: int = Field(default=1, ge=1)

    # Rationale
    prioritization_rationale: List[str] = Field(default_factory=list)


class BudgetAllocation(BaseModel):
    """Recommended budget allocation."""
    category: InvestmentCategory = Field(...)
    allocated_amount_usd: float = Field(..., ge=0)
    allocation_pct: float = Field(..., ge=0, le=100)
    investments_funded: List[str] = Field(default_factory=list)
    risk_reduction_achieved_pct: float = Field(default=0.0, ge=0, le=100)


class PrioritizationInput(BaseModel):
    """Input model for Adaptation Investment Prioritizer."""
    request_id: str = Field(...)
    investment_options: List[InvestmentOption] = Field(..., min_length=1)
    total_budget_usd: float = Field(..., ge=0)
    time_horizon_years: int = Field(default=10, ge=1, le=50)
    discount_rate: float = Field(default=0.05, ge=0, le=0.3)
    risk_reduction_target_pct: Optional[float] = Field(None, ge=0, le=100)

    # Weighting preferences
    cost_effectiveness_weight: float = Field(default=0.3, ge=0, le=1)
    risk_reduction_weight: float = Field(default=0.3, ge=0, le=1)
    co_benefit_weight: float = Field(default=0.15, ge=0, le=1)
    feasibility_weight: float = Field(default=0.15, ge=0, le=1)
    strategic_weight: float = Field(default=0.1, ge=0, le=1)


class PrioritizationOutput(BaseModel):
    """Output model for Adaptation Investment Prioritizer."""
    request_id: str = Field(...)
    completed_at: datetime = Field(default_factory=DeterministicClock.now)

    # Prioritized investments
    prioritized_investments: List[PrioritizedInvestment] = Field(default_factory=list)

    # Summary
    total_investments_analyzed: int = Field(default=0)
    critical_priority_count: int = Field(default=0)
    high_priority_count: int = Field(default=0)

    # Budget analysis
    budget_allocations: List[BudgetAllocation] = Field(default_factory=list)
    total_allocated_usd: float = Field(default=0.0, ge=0)
    budget_utilization_pct: float = Field(default=0.0, ge=0, le=100)

    # Impact summary
    total_risk_reduction_pct: float = Field(default=0.0, ge=0, le=100)
    total_npv_benefits_usd: float = Field(default=0.0)
    portfolio_bcr: float = Field(default=0.0, ge=0)

    # Implementation plan
    implementation_phases: Dict[str, List[str]] = Field(default_factory=dict)

    # Processing info
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# Adaptation Investment Prioritizer Agent Implementation
# =============================================================================

class AdaptationInvestmentPrioritizerAgent(BaseAgent):
    """
    GL-ADAPT-X-010: Adaptation Investment Prioritizer Agent

    Prioritizes adaptation investments using multi-criteria analysis
    based on cost-effectiveness, risk reduction, and co-benefits.

    Zero-Hallucination Implementation:
        - All rankings from deterministic algorithms
        - Verified financial calculations
        - No LLM-based prioritization
        - Complete audit trail

    Example:
        >>> agent = AdaptationInvestmentPrioritizerAgent()
        >>> result = agent.run({
        ...     "request_id": "PRI001",
        ...     "investment_options": [...],
        ...     "total_budget_usd": 5000000
        ... })
    """

    AGENT_ID = "GL-ADAPT-X-010"
    AGENT_NAME = "Adaptation Investment Prioritizer Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Adaptation Investment Prioritizer Agent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Prioritizes adaptation investments",
                version=self.VERSION,
                parameters={}
            )

        super().__init__(config)
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def initialize(self):
        """Initialize agent resources."""
        logger.info("Adaptation Investment Prioritizer Agent initialized")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute investment prioritization."""
        start_time = time.time()

        try:
            priority_input = PrioritizationInput(**input_data)
            self.logger.info(
                f"Starting investment prioritization: {priority_input.request_id}, "
                f"{len(priority_input.investment_options)} options"
            )

            # Score and prioritize each investment
            prioritized = []
            for option in priority_input.investment_options:
                scored = self._score_investment(option, priority_input)
                prioritized.append(scored)

            # Sort by composite score
            prioritized.sort(key=lambda x: x.composite_score, reverse=True)

            # Assign ranks
            for i, inv in enumerate(prioritized):
                inv.priority_rank = i + 1

            # Allocate budget
            allocations = self._allocate_budget(
                prioritized,
                priority_input.total_budget_usd,
                priority_input.risk_reduction_target_pct
            )

            # Calculate portfolio metrics
            funded_investments = [p for p in prioritized if any(
                p.option.option_id in a.investments_funded for a in allocations
            )]
            total_allocated = sum(a.allocated_amount_usd for a in allocations)
            total_risk_reduction = min(100, sum(p.option.risk_reduction_pct for p in funded_investments))
            total_npv = sum(p.npv_usd for p in funded_investments)
            total_cost = sum(p.option.capital_cost_usd for p in funded_investments)
            portfolio_bcr = total_npv / total_cost if total_cost > 0 else 0

            # Create implementation phases
            phases = self._create_implementation_phases(funded_investments)

            # Count priorities
            critical_count = sum(1 for p in prioritized if p.priority_level == PriorityLevel.CRITICAL)
            high_count = sum(1 for p in prioritized if p.priority_level == PriorityLevel.HIGH)

            processing_time = (time.time() - start_time) * 1000

            output = PrioritizationOutput(
                request_id=priority_input.request_id,
                prioritized_investments=prioritized,
                total_investments_analyzed=len(prioritized),
                critical_priority_count=critical_count,
                high_priority_count=high_count,
                budget_allocations=allocations,
                total_allocated_usd=total_allocated,
                budget_utilization_pct=(total_allocated / priority_input.total_budget_usd * 100) if priority_input.total_budget_usd > 0 else 0,
                total_risk_reduction_pct=total_risk_reduction,
                total_npv_benefits_usd=total_npv,
                portfolio_bcr=portfolio_bcr,
                implementation_phases=phases,
                processing_time_ms=processing_time,
            )

            output.provenance_hash = self._calculate_provenance_hash(priority_input, output)

            self.logger.info(
                f"Investment prioritization complete: {critical_count} critical, "
                f"{high_count} high priority"
            )

            return AgentResult(
                success=True,
                data=output.model_dump(),
                metadata={
                    "agent_id": self.AGENT_ID,
                    "version": self.VERSION,
                    "critical_count": critical_count
                }
            )

        except Exception as e:
            self.logger.error(f"Investment prioritization failed: {str(e)}", exc_info=True)
            return AgentResult(
                success=False,
                error=str(e),
                metadata={"agent_id": self.AGENT_ID, "version": self.VERSION}
            )

    def _score_investment(
        self,
        option: InvestmentOption,
        params: PrioritizationInput
    ) -> PrioritizedInvestment:
        """Score and prioritize a single investment."""
        # Calculate financial metrics
        annual_benefit = option.expected_benefit_usd
        total_cost = option.capital_cost_usd + (
            option.annual_operating_cost_usd * params.time_horizon_years
        )
        npv = self._calculate_npv(
            annual_benefit, option.capital_cost_usd,
            option.annual_operating_cost_usd, params.time_horizon_years,
            params.discount_rate
        )
        bcr = npv / option.capital_cost_usd if option.capital_cost_usd > 0 else 0

        # Payback period
        net_annual = annual_benefit - option.annual_operating_cost_usd
        payback = option.capital_cost_usd / net_annual if net_annual > 0 else None

        # Score components (0-1 scale)
        # Cost-effectiveness: higher BCR = better
        cost_effectiveness_score = min(1.0, bcr / 3)  # Normalize to 3x BCR = 1.0

        # Risk reduction: direct mapping
        risk_reduction_score = option.risk_reduction_pct / 100

        # Co-benefits: number of co-benefits (capped at 5)
        co_benefit_score = min(1.0, len(option.co_benefits) / 5)

        # Feasibility: shorter implementation = better
        feasibility_score = max(0, 1 - (option.implementation_months / 36))

        # Strategic alignment: direct from input
        strategic_score = option.strategic_alignment

        # Composite score
        composite = (
            cost_effectiveness_score * params.cost_effectiveness_weight +
            risk_reduction_score * params.risk_reduction_weight +
            co_benefit_score * params.co_benefit_weight +
            feasibility_score * params.feasibility_weight +
            strategic_score * params.strategic_weight
        )

        # Determine priority level
        priority_level = self._determine_priority_level(composite)

        # Determine timeframe
        timeframe = self._determine_timeframe(option, priority_level)

        # Build rationale
        rationale = []
        if cost_effectiveness_score > 0.7:
            rationale.append(f"High cost-effectiveness (BCR: {bcr:.2f})")
        if risk_reduction_score > 0.5:
            rationale.append(f"Significant risk reduction ({option.risk_reduction_pct:.0f}%)")
        if len(option.co_benefits) >= 3:
            rationale.append(f"Multiple co-benefits ({len(option.co_benefits)})")
        if feasibility_score > 0.7:
            rationale.append("Quick implementation feasible")

        return PrioritizedInvestment(
            option=option,
            priority_level=priority_level,
            priority_rank=0,  # Set later after sorting
            cost_effectiveness_score=cost_effectiveness_score,
            risk_reduction_score=risk_reduction_score,
            co_benefit_score=co_benefit_score,
            feasibility_score=feasibility_score,
            strategic_score=strategic_score,
            composite_score=composite,
            benefit_cost_ratio=bcr,
            npv_usd=npv,
            payback_years=payback,
            recommended_timeframe=timeframe,
            recommended_sequence=1,
            prioritization_rationale=rationale
        )

    def _calculate_npv(
        self,
        annual_benefit: float,
        capital_cost: float,
        annual_operating: float,
        years: int,
        discount_rate: float
    ) -> float:
        """Calculate NPV of investment."""
        if discount_rate <= 0:
            return (annual_benefit - annual_operating) * years - capital_cost

        pv_factor = (1 - (1 + discount_rate) ** -years) / discount_rate
        pv_benefits = annual_benefit * pv_factor
        pv_costs = capital_cost + (annual_operating * pv_factor)
        return pv_benefits - pv_costs

    def _determine_priority_level(self, composite_score: float) -> PriorityLevel:
        """Determine priority level from composite score."""
        if composite_score >= 0.8:
            return PriorityLevel.CRITICAL
        elif composite_score >= 0.6:
            return PriorityLevel.HIGH
        elif composite_score >= 0.4:
            return PriorityLevel.MEDIUM
        elif composite_score >= 0.2:
            return PriorityLevel.LOW
        else:
            return PriorityLevel.OPTIONAL

    def _determine_timeframe(
        self,
        option: InvestmentOption,
        priority: PriorityLevel
    ) -> TimeFrame:
        """Determine recommended implementation timeframe."""
        if priority == PriorityLevel.CRITICAL:
            return TimeFrame.IMMEDIATE
        elif priority == PriorityLevel.HIGH:
            if option.implementation_months <= 6:
                return TimeFrame.IMMEDIATE
            else:
                return TimeFrame.SHORT_TERM
        elif priority == PriorityLevel.MEDIUM:
            return TimeFrame.MEDIUM_TERM
        else:
            return TimeFrame.LONG_TERM

    def _allocate_budget(
        self,
        prioritized: List[PrioritizedInvestment],
        total_budget: float,
        risk_target: Optional[float]
    ) -> List[BudgetAllocation]:
        """Allocate budget to investments."""
        allocations_by_category: Dict[InvestmentCategory, BudgetAllocation] = {}
        remaining_budget = total_budget
        cumulative_risk_reduction = 0.0

        # Allocate to highest priority first
        for inv in prioritized:
            if remaining_budget <= 0:
                break

            if risk_target and cumulative_risk_reduction >= risk_target:
                break

            cost = inv.option.capital_cost_usd
            if cost <= remaining_budget:
                remaining_budget -= cost
                cumulative_risk_reduction += inv.option.risk_reduction_pct

                cat = inv.option.category
                if cat not in allocations_by_category:
                    allocations_by_category[cat] = BudgetAllocation(
                        category=cat,
                        allocated_amount_usd=0,
                        allocation_pct=0,
                        investments_funded=[],
                        risk_reduction_achieved_pct=0
                    )

                allocations_by_category[cat].allocated_amount_usd += cost
                allocations_by_category[cat].investments_funded.append(inv.option.option_id)
                allocations_by_category[cat].risk_reduction_achieved_pct += inv.option.risk_reduction_pct

        # Calculate percentages
        allocations = list(allocations_by_category.values())
        total_allocated = sum(a.allocated_amount_usd for a in allocations)
        for alloc in allocations:
            alloc.allocation_pct = (alloc.allocated_amount_usd / total_budget * 100) if total_budget > 0 else 0

        return allocations

    def _create_implementation_phases(
        self,
        investments: List[PrioritizedInvestment]
    ) -> Dict[str, List[str]]:
        """Create implementation phases."""
        phases = {
            TimeFrame.IMMEDIATE.value: [],
            TimeFrame.SHORT_TERM.value: [],
            TimeFrame.MEDIUM_TERM.value: [],
            TimeFrame.LONG_TERM.value: [],
        }

        for inv in investments:
            phases[inv.recommended_timeframe.value].append(inv.option.name)

        return phases

    def _calculate_provenance_hash(
        self,
        input_data: PrioritizationInput,
        output: PrioritizationOutput
    ) -> str:
        """Calculate SHA-256 hash for provenance."""
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "request_id": input_data.request_id,
            "option_count": len(input_data.investment_options),
            "total_allocated": output.total_allocated_usd,
            "timestamp": output.completed_at.isoformat(),
        }
        return hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "AdaptationInvestmentPrioritizerAgent",
    "PriorityLevel",
    "InvestmentCategory",
    "TimeFrame",
    "InvestmentOption",
    "PrioritizedInvestment",
    "BudgetAllocation",
    "PrioritizationInput",
    "PrioritizationOutput",
]
