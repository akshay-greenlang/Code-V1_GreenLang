"""GL-087: Business Case Agent (BUSINESS-CASE).

Develops financial business cases for energy projects.

Standards: NPV Analysis, IRR, Payback Period
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProjectType(str, Enum):
    EFFICIENCY = "EFFICIENCY"
    RENEWABLE = "RENEWABLE"
    ELECTRIFICATION = "ELECTRIFICATION"
    STORAGE = "STORAGE"
    CONTROLS = "CONTROLS"


class CostItem(BaseModel):
    item_id: str
    description: str
    capital_cost_usd: float = Field(default=0, ge=0)
    annual_operating_cost_usd: float = Field(default=0, ge=0)


class BenefitItem(BaseModel):
    item_id: str
    description: str
    annual_savings_usd: float = Field(default=0, ge=0)
    growth_rate_pct: float = Field(default=0)


class BusinessCaseInput(BaseModel):
    project_id: str
    project_name: str = Field(default="Energy Project")
    project_type: ProjectType = Field(default=ProjectType.EFFICIENCY)
    costs: List[CostItem] = Field(default_factory=list)
    benefits: List[BenefitItem] = Field(default_factory=list)
    project_life_years: int = Field(default=15, ge=1)
    discount_rate_pct: float = Field(default=8, ge=0)
    inflation_rate_pct: float = Field(default=2, ge=0)
    incentive_usd: float = Field(default=0, ge=0)
    carbon_value_per_tonne: float = Field(default=0, ge=0)
    annual_carbon_reduction_tonnes: float = Field(default=0, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BusinessCaseOutput(BaseModel):
    project_id: str
    total_capital_cost_usd: float
    total_annual_savings_usd: float
    simple_payback_years: float
    net_present_value_usd: float
    internal_rate_of_return_pct: float
    benefit_cost_ratio: float
    levelized_cost_of_savings_usd: float
    lifetime_savings_usd: float
    lifetime_carbon_reduction_tonnes: float
    lifetime_carbon_value_usd: float
    investment_grade: str
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class BusinessCaseAgent:
    AGENT_ID = "GL-087B"
    AGENT_NAME = "BUSINESS-CASE"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"BusinessCaseAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = BusinessCaseInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _calculate_irr(self, cash_flows: List[float]) -> float:
        """Calculate IRR."""
        if not cash_flows or all(cf == 0 for cf in cash_flows):
            return 0

        rate = 0.1
        for _ in range(100):
            npv = sum(cf / (1 + rate) ** i for i, cf in enumerate(cash_flows))
            npv_deriv = sum(-i * cf / (1 + rate) ** (i + 1) for i, cf in enumerate(cash_flows))
            if abs(npv_deriv) < 1e-10:
                break
            new_rate = rate - npv / npv_deriv
            if abs(new_rate - rate) < 1e-6:
                break
            rate = max(-0.99, min(10, new_rate))

        return rate * 100

    def _process(self, inp: BusinessCaseInput) -> BusinessCaseOutput:
        recommendations = []

        # Calculate totals
        total_capital = sum(c.capital_cost_usd for c in inp.costs)
        total_annual_opex = sum(c.annual_operating_cost_usd for c in inp.costs)
        first_year_savings = sum(b.annual_savings_usd for b in inp.benefits)

        # Net capital after incentives
        net_capital = total_capital - inp.incentive_usd

        # Carbon value
        annual_carbon_value = inp.annual_carbon_reduction_tonnes * inp.carbon_value_per_tonne
        total_annual_benefit = first_year_savings + annual_carbon_value

        # Simple payback
        net_annual = total_annual_benefit - total_annual_opex
        simple_payback = net_capital / net_annual if net_annual > 0 else 99

        # Build cash flows for NPV/IRR
        discount = inp.discount_rate_pct / 100
        cash_flows = [-net_capital]
        lifetime_savings = 0
        pv_benefits = 0
        pv_costs = net_capital

        for year in range(1, inp.project_life_years + 1):
            # Apply growth and inflation
            year_savings = 0
            for benefit in inp.benefits:
                growth = (1 + benefit.growth_rate_pct / 100) ** (year - 1)
                year_savings += benefit.annual_savings_usd * growth

            year_carbon = annual_carbon_value * ((1 + inp.inflation_rate_pct / 100) ** (year - 1))
            year_opex = total_annual_opex * ((1 + inp.inflation_rate_pct / 100) ** (year - 1))

            net_cf = year_savings + year_carbon - year_opex
            cash_flows.append(net_cf)
            lifetime_savings += year_savings

            pv_benefits += (year_savings + year_carbon) / (1 + discount) ** year
            pv_costs += year_opex / (1 + discount) ** year

        # NPV
        npv = sum(cf / (1 + discount) ** i for i, cf in enumerate(cash_flows))

        # IRR
        irr = self._calculate_irr(cash_flows)

        # Benefit/Cost ratio
        bcr = pv_benefits / pv_costs if pv_costs > 0 else 0

        # Levelized cost of savings
        if lifetime_savings > 0:
            lcos = net_capital / lifetime_savings
        else:
            lcos = 0

        # Lifetime carbon
        lifetime_carbon = inp.annual_carbon_reduction_tonnes * inp.project_life_years
        lifetime_carbon_value = lifetime_carbon * inp.carbon_value_per_tonne

        # Investment grade
        if npv > 0 and irr > 15 and simple_payback < 3:
            grade = "A - Excellent"
        elif npv > 0 and irr > 10 and simple_payback < 5:
            grade = "B - Good"
        elif npv > 0 and irr > inp.discount_rate_pct:
            grade = "C - Acceptable"
        elif npv > 0:
            grade = "D - Marginal"
        else:
            grade = "F - Not Viable"

        # Recommendations
        if npv < 0:
            recommendations.append(f"Project has negative NPV ${npv:,.0f} - seek additional incentives")
        if simple_payback > 5:
            recommendations.append(f"Long payback {simple_payback:.1f} years - may not meet corporate hurdle")
        if inp.incentive_usd == 0:
            recommendations.append("No incentives applied - research available rebates and tax credits")
        if inp.carbon_value_per_tonne == 0 and inp.annual_carbon_reduction_tonnes > 0:
            recommendations.append("Carbon value not included - add internal carbon price")
        if bcr < 1.5:
            recommendations.append(f"Low benefit/cost ratio {bcr:.2f} - optimize project scope")

        if grade.startswith("A") or grade.startswith("B"):
            recommendations.append("Strong business case - proceed with implementation")

        calc_hash = hashlib.sha256(json.dumps({
            "project": inp.project_id,
            "npv": round(npv, 2),
            "irr": round(irr, 1),
            "payback": round(simple_payback, 2)
        }).encode()).hexdigest()

        return BusinessCaseOutput(
            project_id=inp.project_id,
            total_capital_cost_usd=round(total_capital, 2),
            total_annual_savings_usd=round(first_year_savings, 2),
            simple_payback_years=round(simple_payback, 2),
            net_present_value_usd=round(npv, 2),
            internal_rate_of_return_pct=round(irr, 1),
            benefit_cost_ratio=round(bcr, 2),
            levelized_cost_of_savings_usd=round(lcos, 4),
            lifetime_savings_usd=round(lifetime_savings, 2),
            lifetime_carbon_reduction_tonnes=round(lifetime_carbon, 1),
            lifetime_carbon_value_usd=round(lifetime_carbon_value, 2),
            investment_grade=grade,
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-087B", "name": "BUSINESS-CASE", "version": "1.0.0",
    "summary": "Financial business case development",
    "standards": [{"ref": "NPV Analysis"}, {"ref": "IRR"}, {"ref": "Payback Period"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
