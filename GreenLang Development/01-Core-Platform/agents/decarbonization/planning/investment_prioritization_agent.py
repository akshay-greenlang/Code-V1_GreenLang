# -*- coding: utf-8 -*-
"""
GL-DECARB-X-005: Investment Prioritization Agent
=================================================

Ranks decarbonization investments based on financial metrics including
NPV, IRR, payback period, and carbon cost efficiency.

Capabilities:
    - Calculate Net Present Value (NPV) of investments
    - Calculate Internal Rate of Return (IRR)
    - Calculate simple and discounted payback periods
    - Rank investments by multiple criteria
    - Handle uncertainty in cost and savings estimates
    - Support multi-criteria decision analysis (MCDA)
    - Compare projects across different scales
    - Generate investment priority queues

Zero-Hallucination Principle:
    All financial calculations use deterministic formulas with
    documented assumptions. No AI-generated financial estimates.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import math
import time
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.agents.base_agents import DeterministicAgent, AuditEntry
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism import (
    DeterministicClock,
    content_hash,
    deterministic_id,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class RankingCriteria(str, Enum):
    """Criteria for ranking investments."""
    NPV = "npv"
    IRR = "irr"
    PAYBACK = "payback"
    COST_PER_TCO2E = "cost_per_tco2e"
    CARBON_ROI = "carbon_roi"
    MCDA_SCORE = "mcda_score"


class InvestmentCategory(str, Enum):
    """Categories of decarbonization investments."""
    ENERGY_EFFICIENCY = "energy_efficiency"
    RENEWABLE_ENERGY = "renewable_energy"
    ELECTRIFICATION = "electrification"
    PROCESS_CHANGE = "process_change"
    FUEL_SWITCHING = "fuel_switching"
    CARBON_CAPTURE = "carbon_capture"
    NATURE_BASED = "nature_based"


class RiskLevel(str, Enum):
    """Risk levels for investments."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


# =============================================================================
# Pydantic Models
# =============================================================================

class CashFlow(BaseModel):
    """Cash flow for a specific year."""
    year: int = Field(..., description="Year")
    capital_expenditure: float = Field(default=0, description="Capital expenditure (negative)")
    operating_cost_savings: float = Field(default=0, description="Operating cost savings")
    maintenance_cost: float = Field(default=0, description="Additional maintenance cost")
    carbon_savings_value: float = Field(default=0, description="Value of carbon savings at assumed price")
    net_cash_flow: float = Field(default=0, description="Net cash flow for year")


class InvestmentProject(BaseModel):
    """A decarbonization investment project."""
    project_id: str = Field(..., description="Unique project identifier")
    name: str = Field(..., description="Project name")
    description: str = Field(default="", description="Project description")
    category: InvestmentCategory = Field(..., description="Investment category")

    # Capital costs
    capital_cost_usd: float = Field(..., ge=0, description="Total capital investment")
    capital_cost_low: float = Field(default=0, ge=0, description="Low estimate")
    capital_cost_high: float = Field(default=0, ge=0, description="High estimate")

    # Operating impact
    annual_operating_savings_usd: float = Field(default=0, description="Annual operating cost savings")
    annual_maintenance_cost_usd: float = Field(default=0, ge=0, description="Additional annual maintenance")

    # Carbon impact
    annual_emission_reduction_tco2e: float = Field(..., ge=0, description="Annual emission reduction")

    # Timing
    implementation_years: int = Field(default=1, ge=1, le=10, description="Years to implement")
    economic_life_years: int = Field(default=15, ge=1, le=50, description="Economic life of investment")

    # Risk
    risk_level: RiskLevel = Field(default=RiskLevel.MEDIUM)
    technology_readiness: int = Field(default=7, ge=1, le=9, description="TRL")


class InvestmentMetrics(BaseModel):
    """Financial metrics for an investment project."""
    project_id: str = Field(..., description="Project ID")
    project_name: str = Field(..., description="Project name")

    # NPV
    npv_usd: float = Field(..., description="Net Present Value")
    npv_per_tco2e: float = Field(..., description="NPV per tonne abated")

    # IRR
    irr_percent: Optional[float] = Field(None, description="Internal Rate of Return (%)")

    # Payback
    simple_payback_years: Optional[float] = Field(None, description="Simple payback period")
    discounted_payback_years: Optional[float] = Field(None, description="Discounted payback period")

    # Cost effectiveness
    cost_per_tco2e: float = Field(..., description="Cost per tonne abated")
    lifetime_abatement_tco2e: float = Field(..., ge=0, description="Total lifetime abatement")

    # Carbon ROI
    carbon_roi: float = Field(..., description="Carbon ROI (tCO2e/$1000 invested)")

    # MCDA score
    mcda_score: float = Field(default=0, ge=0, le=100, description="Multi-criteria score")

    # Ranking
    rank_by_npv: int = Field(default=0, ge=0, description="Rank by NPV")
    rank_by_irr: int = Field(default=0, ge=0, description="Rank by IRR")
    rank_by_payback: int = Field(default=0, ge=0, description="Rank by payback")
    rank_by_cost_effectiveness: int = Field(default=0, ge=0, description="Rank by cost/tCO2e")
    overall_rank: int = Field(default=0, ge=0, description="Overall rank")

    # Cash flows
    cash_flows: List[CashFlow] = Field(default_factory=list, description="Year-by-year cash flows")


class InvestmentPrioritizationInput(BaseModel):
    """Input model for InvestmentPrioritizationAgent."""
    operation: str = Field(
        default="rank_investments",
        description="Operation: 'rank_investments', 'calculate_metrics', 'optimize_portfolio'"
    )

    # Projects to analyze
    projects: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of investment projects"
    )

    # Financial parameters
    discount_rate: float = Field(default=0.08, ge=0, le=0.3, description="Discount rate")
    carbon_price_usd: float = Field(default=50, ge=0, description="Assumed carbon price ($/tCO2e)")
    inflation_rate: float = Field(default=0.02, ge=0, le=0.2, description="Inflation rate")

    # Ranking criteria
    primary_criteria: RankingCriteria = Field(default=RankingCriteria.NPV)

    # MCDA weights (must sum to 1)
    weight_npv: float = Field(default=0.3, ge=0, le=1, description="Weight for NPV")
    weight_payback: float = Field(default=0.2, ge=0, le=1, description="Weight for payback")
    weight_carbon: float = Field(default=0.3, ge=0, le=1, description="Weight for carbon impact")
    weight_risk: float = Field(default=0.2, ge=0, le=1, description="Weight for risk")

    # Budget constraint for portfolio optimization
    budget_constraint_usd: Optional[float] = Field(None, ge=0, description="Total budget")


class InvestmentPrioritizationOutput(BaseModel):
    """Output model for InvestmentPrioritizationAgent."""
    operation: str = Field(..., description="Operation performed")
    success: bool = Field(..., description="Whether operation succeeded")

    # Results
    ranked_investments: List[InvestmentMetrics] = Field(
        default_factory=list,
        description="Investments with metrics, ranked"
    )

    # Portfolio summary
    portfolio_summary: Optional[Dict[str, Any]] = Field(None, description="Portfolio-level summary")

    # Metadata
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    error_message: Optional[str] = Field(None)


# =============================================================================
# Agent Implementation
# =============================================================================

class InvestmentPrioritizationAgent(DeterministicAgent):
    """
    GL-DECARB-X-005: Investment Prioritization Agent

    Ranks decarbonization investments using standard financial metrics
    and multi-criteria decision analysis.

    Zero-Hallucination Implementation:
        - All financial calculations use standard formulas
        - NPV, IRR, payback calculated deterministically
        - No AI-generated financial estimates
        - Complete audit trail for all calculations

    Example:
        >>> agent = InvestmentPrioritizationAgent()
        >>> result = agent.run({
        ...     "operation": "rank_investments",
        ...     "projects": project_list,
        ...     "discount_rate": 0.08,
        ...     "carbon_price_usd": 75
        ... })
    """

    AGENT_ID = "GL-DECARB-X-005"
    AGENT_NAME = "Investment Prioritization Agent"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="InvestmentPrioritizationAgent",
        category=AgentCategory.CRITICAL,
        description="Ranks investments by NPV, IRR, payback, and carbon efficiency"
    )

    def __init__(self, config: Optional[AgentConfig] = None, enable_audit_trail: bool = True):
        """Initialize the InvestmentPrioritizationAgent."""
        super().__init__(enable_audit_trail=enable_audit_trail)

        self.config = config or AgentConfig(
            name=self.AGENT_NAME,
            description="Ranks investments by financial and carbon metrics",
            version=self.VERSION
        )

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.AGENT_ID}: {self.AGENT_NAME}")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute investment prioritization operation."""
        start_time = time.time()
        calculation_trace = []

        try:
            # Parse input
            invest_input = InvestmentPrioritizationInput(**inputs)
            calculation_trace.append(f"Operation: {invest_input.operation}")

            if invest_input.operation == "rank_investments":
                result = self._rank_investments(invest_input, calculation_trace)
            elif invest_input.operation == "calculate_metrics":
                result = self._calculate_metrics(invest_input, calculation_trace)
            elif invest_input.operation == "optimize_portfolio":
                result = self._optimize_portfolio(invest_input, calculation_trace)
            else:
                raise ValueError(f"Unknown operation: {invest_input.operation}")

            processing_time = (time.time() - start_time) * 1000
            result["processing_time_ms"] = processing_time

            self._capture_audit_entry(
                operation=invest_input.operation,
                inputs={"operation": invest_input.operation, "projects_count": len(invest_input.projects)},
                outputs={"success": result["success"]},
                calculation_trace=calculation_trace
            )

            return result

        except Exception as e:
            self.logger.error(f"Investment prioritization failed: {str(e)}", exc_info=True)
            return {
                "operation": inputs.get("operation", "unknown"),
                "success": False,
                "error_message": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": DeterministicClock.now().isoformat()
            }

    def _rank_investments(
        self,
        invest_input: InvestmentPrioritizationInput,
        calculation_trace: List[str]
    ) -> Dict[str, Any]:
        """Rank investments by specified criteria."""
        # Calculate metrics for all projects
        metrics_list = []

        for proj_data in invest_input.projects:
            project = InvestmentProject(**proj_data)
            metrics = self._calculate_project_metrics(
                project,
                invest_input.discount_rate,
                invest_input.carbon_price_usd,
                calculation_trace
            )

            # Calculate MCDA score
            metrics.mcda_score = self._calculate_mcda_score(
                metrics,
                invest_input.weight_npv,
                invest_input.weight_payback,
                invest_input.weight_carbon,
                invest_input.weight_risk,
                project.risk_level
            )

            metrics_list.append(metrics)

        calculation_trace.append(f"Calculated metrics for {len(metrics_list)} projects")

        # Rank by different criteria
        # NPV ranking (higher is better)
        npv_sorted = sorted(metrics_list, key=lambda m: m.npv_usd, reverse=True)
        for i, m in enumerate(npv_sorted):
            m.rank_by_npv = i + 1

        # IRR ranking (higher is better)
        irr_sorted = sorted(
            [m for m in metrics_list if m.irr_percent is not None],
            key=lambda m: m.irr_percent or 0,
            reverse=True
        )
        for i, m in enumerate(irr_sorted):
            m.rank_by_irr = i + 1

        # Payback ranking (lower is better)
        payback_sorted = sorted(
            [m for m in metrics_list if m.simple_payback_years is not None],
            key=lambda m: m.simple_payback_years or 999
        )
        for i, m in enumerate(payback_sorted):
            m.rank_by_payback = i + 1

        # Cost effectiveness ranking (lower is better)
        cost_sorted = sorted(metrics_list, key=lambda m: m.cost_per_tco2e)
        for i, m in enumerate(cost_sorted):
            m.rank_by_cost_effectiveness = i + 1

        # Overall ranking based on primary criteria
        if invest_input.primary_criteria == RankingCriteria.NPV:
            ranked = npv_sorted
        elif invest_input.primary_criteria == RankingCriteria.IRR:
            ranked = irr_sorted
        elif invest_input.primary_criteria == RankingCriteria.PAYBACK:
            ranked = payback_sorted
        elif invest_input.primary_criteria == RankingCriteria.COST_PER_TCO2E:
            ranked = cost_sorted
        elif invest_input.primary_criteria == RankingCriteria.MCDA_SCORE:
            ranked = sorted(metrics_list, key=lambda m: m.mcda_score, reverse=True)
        else:
            ranked = npv_sorted

        for i, m in enumerate(ranked):
            m.overall_rank = i + 1

        calculation_trace.append(f"Ranked by {invest_input.primary_criteria.value}")

        # Portfolio summary
        portfolio_summary = {
            "total_projects": len(ranked),
            "total_capital_required": sum(
                proj.get("capital_cost_usd", 0)
                for proj in invest_input.projects
            ),
            "total_lifetime_abatement_tco2e": sum(m.lifetime_abatement_tco2e for m in ranked),
            "average_npv": sum(m.npv_usd for m in ranked) / len(ranked) if ranked else 0,
            "positive_npv_count": sum(1 for m in ranked if m.npv_usd > 0),
            "negative_cost_per_tco2e_count": sum(1 for m in ranked if m.cost_per_tco2e < 0),
        }

        return {
            "operation": "rank_investments",
            "success": True,
            "ranked_investments": [m.model_dump() for m in ranked],
            "portfolio_summary": portfolio_summary,
            "timestamp": DeterministicClock.now().isoformat()
        }

    def _calculate_metrics(
        self,
        invest_input: InvestmentPrioritizationInput,
        calculation_trace: List[str]
    ) -> Dict[str, Any]:
        """Calculate metrics without ranking."""
        metrics_list = []

        for proj_data in invest_input.projects:
            project = InvestmentProject(**proj_data)
            metrics = self._calculate_project_metrics(
                project,
                invest_input.discount_rate,
                invest_input.carbon_price_usd,
                calculation_trace
            )
            metrics_list.append(metrics)

        return {
            "operation": "calculate_metrics",
            "success": True,
            "ranked_investments": [m.model_dump() for m in metrics_list],
            "timestamp": DeterministicClock.now().isoformat()
        }

    def _optimize_portfolio(
        self,
        invest_input: InvestmentPrioritizationInput,
        calculation_trace: List[str]
    ) -> Dict[str, Any]:
        """Select optimal portfolio within budget constraint."""
        if not invest_input.budget_constraint_usd:
            raise ValueError("budget_constraint_usd required for portfolio optimization")

        # First rank all investments
        rank_result = self._rank_investments(invest_input, calculation_trace)
        all_investments = rank_result["ranked_investments"]

        # Greedy selection by MCDA score within budget
        budget = invest_input.budget_constraint_usd
        selected = []
        remaining_budget = budget

        # Sort by MCDA score
        sorted_investments = sorted(all_investments, key=lambda m: m.get("mcda_score", 0), reverse=True)

        for inv in sorted_investments:
            capital = next(
                (p.get("capital_cost_usd", 0) for p in invest_input.projects if p.get("project_id") == inv["project_id"]),
                0
            )
            if capital <= remaining_budget:
                selected.append(inv)
                remaining_budget -= capital

        calculation_trace.append(f"Selected {len(selected)} projects within ${budget:,.0f} budget")

        portfolio_summary = {
            "budget": budget,
            "budget_utilized": budget - remaining_budget,
            "budget_remaining": remaining_budget,
            "projects_selected": len(selected),
            "total_npv": sum(inv.get("npv_usd", 0) for inv in selected),
            "total_abatement_tco2e": sum(inv.get("lifetime_abatement_tco2e", 0) for inv in selected),
        }

        return {
            "operation": "optimize_portfolio",
            "success": True,
            "ranked_investments": selected,
            "portfolio_summary": portfolio_summary,
            "timestamp": DeterministicClock.now().isoformat()
        }

    def _calculate_project_metrics(
        self,
        project: InvestmentProject,
        discount_rate: float,
        carbon_price: float,
        calculation_trace: List[str]
    ) -> InvestmentMetrics:
        """Calculate all financial metrics for a project."""
        # Generate cash flows
        cash_flows = self._generate_cash_flows(
            project,
            carbon_price,
            calculation_trace
        )

        # Calculate NPV
        npv = self._calculate_npv(cash_flows, discount_rate)

        # Calculate IRR
        irr = self._calculate_irr(cash_flows)

        # Calculate payback
        simple_payback = self._calculate_simple_payback(cash_flows)
        discounted_payback = self._calculate_discounted_payback(cash_flows, discount_rate)

        # Calculate lifetime abatement
        lifetime_abatement = project.annual_emission_reduction_tco2e * project.economic_life_years

        # Calculate cost per tCO2e
        if lifetime_abatement > 0:
            cost_per_tco2e = project.capital_cost_usd / lifetime_abatement
            npv_per_tco2e = npv / lifetime_abatement
        else:
            cost_per_tco2e = float('inf')
            npv_per_tco2e = 0

        # Calculate carbon ROI
        carbon_roi = (lifetime_abatement / project.capital_cost_usd * 1000) if project.capital_cost_usd > 0 else 0

        return InvestmentMetrics(
            project_id=project.project_id,
            project_name=project.name,
            npv_usd=npv,
            npv_per_tco2e=npv_per_tco2e,
            irr_percent=irr,
            simple_payback_years=simple_payback,
            discounted_payback_years=discounted_payback,
            cost_per_tco2e=cost_per_tco2e,
            lifetime_abatement_tco2e=lifetime_abatement,
            carbon_roi=carbon_roi,
            cash_flows=cash_flows
        )

    def _generate_cash_flows(
        self,
        project: InvestmentProject,
        carbon_price: float,
        calculation_trace: List[str]
    ) -> List[CashFlow]:
        """Generate year-by-year cash flows."""
        cash_flows = []

        # Year 0: Initial investment (spread over implementation years)
        annual_capex = project.capital_cost_usd / project.implementation_years

        for year in range(project.economic_life_years + project.implementation_years):
            capex = -annual_capex if year < project.implementation_years else 0
            savings = project.annual_operating_savings_usd if year >= project.implementation_years else 0
            maintenance = project.annual_maintenance_cost_usd if year >= project.implementation_years else 0
            carbon_value = project.annual_emission_reduction_tco2e * carbon_price if year >= project.implementation_years else 0

            net_cf = capex + savings - maintenance + carbon_value

            cash_flows.append(CashFlow(
                year=year,
                capital_expenditure=capex,
                operating_cost_savings=savings,
                maintenance_cost=maintenance,
                carbon_savings_value=carbon_value,
                net_cash_flow=net_cf
            ))

        return cash_flows

    def _calculate_npv(self, cash_flows: List[CashFlow], discount_rate: float) -> float:
        """Calculate Net Present Value."""
        npv = 0
        for cf in cash_flows:
            discount_factor = 1 / ((1 + discount_rate) ** cf.year)
            npv += cf.net_cash_flow * discount_factor
        return npv

    def _calculate_irr(self, cash_flows: List[CashFlow], max_iterations: int = 100) -> Optional[float]:
        """Calculate Internal Rate of Return using Newton-Raphson method."""
        cf_values = [cf.net_cash_flow for cf in cash_flows]

        # Need at least one sign change for IRR to exist
        has_positive = any(cf > 0 for cf in cf_values)
        has_negative = any(cf < 0 for cf in cf_values)
        if not (has_positive and has_negative):
            return None

        # Initial guess
        irr = 0.1
        epsilon = 1e-7

        for _ in range(max_iterations):
            npv = sum(cf / ((1 + irr) ** i) for i, cf in enumerate(cf_values))
            derivative = sum(-i * cf / ((1 + irr) ** (i + 1)) for i, cf in enumerate(cf_values))

            if abs(derivative) < epsilon:
                break

            irr_new = irr - npv / derivative

            if abs(irr_new - irr) < epsilon:
                return irr_new * 100  # Return as percentage

            irr = irr_new

        return irr * 100 if -1 < irr < 10 else None  # Sanity check

    def _calculate_simple_payback(self, cash_flows: List[CashFlow]) -> Optional[float]:
        """Calculate simple payback period."""
        cumulative = 0
        for cf in cash_flows:
            cumulative += cf.net_cash_flow
            if cumulative >= 0:
                # Interpolate for partial year
                if cf.net_cash_flow > 0:
                    partial = (cumulative - cf.net_cash_flow) / (-cf.net_cash_flow)
                    return cf.year + partial
                return float(cf.year)
        return None  # Never pays back

    def _calculate_discounted_payback(
        self,
        cash_flows: List[CashFlow],
        discount_rate: float
    ) -> Optional[float]:
        """Calculate discounted payback period."""
        cumulative = 0
        for cf in cash_flows:
            discount_factor = 1 / ((1 + discount_rate) ** cf.year)
            discounted_cf = cf.net_cash_flow * discount_factor
            cumulative += discounted_cf
            if cumulative >= 0:
                return float(cf.year)
        return None

    def _calculate_mcda_score(
        self,
        metrics: InvestmentMetrics,
        weight_npv: float,
        weight_payback: float,
        weight_carbon: float,
        weight_risk: float,
        risk_level: RiskLevel
    ) -> float:
        """Calculate multi-criteria decision analysis score."""
        # Normalize NPV (0-100 scale, assuming max NPV of 1M)
        npv_score = min(max(metrics.npv_usd / 1000000 * 100, 0), 100)

        # Normalize payback (0-100 scale, lower is better)
        if metrics.simple_payback_years:
            payback_score = max(100 - metrics.simple_payback_years * 10, 0)
        else:
            payback_score = 0

        # Normalize carbon impact (0-100 scale)
        carbon_score = min(metrics.lifetime_abatement_tco2e / 10000 * 100, 100)

        # Risk score (0-100 scale)
        risk_scores = {
            RiskLevel.LOW: 100,
            RiskLevel.MEDIUM: 66,
            RiskLevel.HIGH: 33,
            RiskLevel.VERY_HIGH: 0,
        }
        risk_score = risk_scores.get(risk_level, 50)

        # Weighted sum
        mcda = (
            npv_score * weight_npv +
            payback_score * weight_payback +
            carbon_score * weight_carbon +
            risk_score * weight_risk
        )

        return mcda

    # =========================================================================
    # Public API Methods
    # =========================================================================

    def rank_projects(
        self,
        projects: List[Dict[str, Any]],
        discount_rate: float = 0.08,
        carbon_price: float = 50,
        criteria: RankingCriteria = RankingCriteria.NPV
    ) -> List[InvestmentMetrics]:
        """
        Rank investment projects.

        Args:
            projects: List of project dictionaries
            discount_rate: Discount rate for NPV calculation
            carbon_price: Assumed carbon price ($/tCO2e)
            criteria: Primary ranking criteria

        Returns:
            List of InvestmentMetrics ranked by criteria
        """
        result = self.execute({
            "operation": "rank_investments",
            "projects": projects,
            "discount_rate": discount_rate,
            "carbon_price_usd": carbon_price,
            "primary_criteria": criteria.value
        })

        if result["success"]:
            return [InvestmentMetrics(**m) for m in result["ranked_investments"]]
        else:
            raise ValueError(result.get("error_message", "Ranking failed"))
