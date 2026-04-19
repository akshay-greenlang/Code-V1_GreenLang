"""GL-087: Business Builder Agent (BUSINESS-BUILDER).

Builds business models for energy solutions.

Standards: Business Model Canvas, Lean Startup
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BusinessModelType(str, Enum):
    ESCO = "ESCO"
    PPA = "PPA"
    LEASE = "LEASE"
    SHARED_SAVINGS = "SHARED_SAVINGS"
    FEE_FOR_SERVICE = "FEE_FOR_SERVICE"


class RevenueStream(BaseModel):
    stream_id: str
    name: str
    type: str
    annual_value_usd: float = Field(ge=0)
    margin_pct: float = Field(ge=0, le=100)
    growth_rate_pct: float = Field(default=0)


class BusinessBuilderInput(BaseModel):
    solution_id: str
    solution_name: str = Field(default="Energy Solution")
    model_type: BusinessModelType = Field(default=BusinessModelType.SHARED_SAVINGS)
    revenue_streams: List[RevenueStream] = Field(default_factory=list)
    implementation_cost_usd: float = Field(default=100000, ge=0)
    annual_operating_cost_usd: float = Field(default=10000, ge=0)
    contract_term_years: int = Field(default=10, ge=1)
    customer_savings_share_pct: float = Field(default=50, ge=0, le=100)
    discount_rate_pct: float = Field(default=8, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class YearlyProjection(BaseModel):
    year: int
    revenue_usd: float
    costs_usd: float
    profit_usd: float
    cumulative_cash_flow_usd: float


class BusinessBuilderOutput(BaseModel):
    solution_id: str
    model_type: str
    total_revenue_usd: float
    total_costs_usd: float
    gross_margin_pct: float
    net_present_value_usd: float
    internal_rate_of_return_pct: float
    payback_years: float
    projections: List[YearlyProjection]
    break_even_year: int
    model_viability: str
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class BusinessBuilderAgent:
    AGENT_ID = "GL-087"
    AGENT_NAME = "BUSINESS-BUILDER"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"BusinessBuilderAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = BusinessBuilderInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _calculate_irr(self, cash_flows: List[float], max_iterations: int = 100) -> float:
        """Calculate IRR using Newton-Raphson method."""
        if not cash_flows or all(cf == 0 for cf in cash_flows):
            return 0

        rate = 0.1
        for _ in range(max_iterations):
            npv = sum(cf / (1 + rate) ** i for i, cf in enumerate(cash_flows))
            npv_derivative = sum(-i * cf / (1 + rate) ** (i + 1) for i, cf in enumerate(cash_flows))

            if abs(npv_derivative) < 1e-10:
                break

            new_rate = rate - npv / npv_derivative
            if abs(new_rate - rate) < 1e-6:
                break
            rate = new_rate

        return rate * 100

    def _process(self, inp: BusinessBuilderInput) -> BusinessBuilderOutput:
        recommendations = []
        projections = []
        cash_flows = [-inp.implementation_cost_usd]
        cumulative = -inp.implementation_cost_usd
        break_even_year = inp.contract_term_years + 1
        total_revenue = 0
        total_costs = inp.implementation_cost_usd

        for year in range(1, inp.contract_term_years + 1):
            # Calculate annual revenue with growth
            year_revenue = 0
            for stream in inp.revenue_streams:
                growth_factor = (1 + stream.growth_rate_pct / 100) ** (year - 1)
                year_revenue += stream.annual_value_usd * growth_factor

            # For shared savings model, adjust revenue
            if inp.model_type == BusinessModelType.SHARED_SAVINGS:
                year_revenue *= (1 - inp.customer_savings_share_pct / 100)

            year_costs = inp.annual_operating_cost_usd
            year_profit = year_revenue - year_costs
            cumulative += year_profit
            cash_flows.append(year_profit)

            total_revenue += year_revenue
            total_costs += year_costs

            projections.append(YearlyProjection(
                year=year,
                revenue_usd=round(year_revenue, 2),
                costs_usd=round(year_costs, 2),
                profit_usd=round(year_profit, 2),
                cumulative_cash_flow_usd=round(cumulative, 2)
            ))

            if cumulative >= 0 and break_even_year > inp.contract_term_years:
                break_even_year = year

        # Financial metrics
        discount = inp.discount_rate_pct / 100
        npv = sum(cf / (1 + discount) ** i for i, cf in enumerate(cash_flows))
        irr = self._calculate_irr(cash_flows)

        # Gross margin
        gross_margin = ((total_revenue - total_costs) / total_revenue * 100) if total_revenue > 0 else 0

        # Payback
        if any(p.cumulative_cash_flow_usd >= 0 for p in projections):
            for i, p in enumerate(projections):
                if p.cumulative_cash_flow_usd >= 0:
                    if i > 0:
                        prev = projections[i-1].cumulative_cash_flow_usd
                        frac = abs(prev) / (p.cumulative_cash_flow_usd - prev) if (p.cumulative_cash_flow_usd - prev) != 0 else 0
                        payback = i + frac
                    else:
                        payback = 1
                    break
        else:
            payback = inp.contract_term_years + 1

        # Viability assessment
        if npv > 0 and irr > inp.discount_rate_pct and payback < inp.contract_term_years * 0.5:
            viability = "HIGHLY VIABLE"
        elif npv > 0 and irr > inp.discount_rate_pct:
            viability = "VIABLE"
        elif npv > 0:
            viability = "MARGINAL"
        else:
            viability = "NOT VIABLE"

        # Recommendations
        if npv < 0:
            recommendations.append(f"Negative NPV ${npv:,.0f} - restructure pricing or reduce costs")
        if payback > inp.contract_term_years * 0.7:
            recommendations.append(f"Long payback {payback:.1f} years - consider shorter contract term")
        if gross_margin < 20:
            recommendations.append(f"Low gross margin {gross_margin:.1f}% - improve pricing")

        if inp.model_type == BusinessModelType.SHARED_SAVINGS and inp.customer_savings_share_pct > 60:
            recommendations.append("High customer share reduces provider returns")

        if len(inp.revenue_streams) == 1:
            recommendations.append("Single revenue stream - diversify for stability")

        calc_hash = hashlib.sha256(json.dumps({
            "solution": inp.solution_id,
            "npv": round(npv, 2),
            "irr": round(irr, 1)
        }).encode()).hexdigest()

        return BusinessBuilderOutput(
            solution_id=inp.solution_id,
            model_type=inp.model_type.value,
            total_revenue_usd=round(total_revenue, 2),
            total_costs_usd=round(total_costs, 2),
            gross_margin_pct=round(gross_margin, 1),
            net_present_value_usd=round(npv, 2),
            internal_rate_of_return_pct=round(irr, 1),
            payback_years=round(payback, 2),
            projections=projections,
            break_even_year=break_even_year,
            model_viability=viability,
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-087", "name": "BUSINESS-BUILDER", "version": "1.0.0",
    "summary": "Business model development for energy solutions",
    "standards": [{"ref": "Business Model Canvas"}, {"ref": "Lean Startup"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
