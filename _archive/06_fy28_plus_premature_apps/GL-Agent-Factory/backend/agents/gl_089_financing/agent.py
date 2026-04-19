"""GL-089: Financing Agent (FINANCING).

Optimizes financing structures for energy projects.

Standards: Project Finance, Green Bonds
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FinancingType(str, Enum):
    CAPEX = "CAPEX"
    LEASE = "LEASE"
    PPA = "PPA"
    ESCO = "ESCO"
    GREEN_BOND = "GREEN_BOND"
    PACE = "PACE"


class FinancingOption(BaseModel):
    option_id: str
    financing_type: FinancingType
    provider: str
    interest_rate_pct: float = Field(ge=0)
    term_years: int = Field(ge=1)
    down_payment_pct: float = Field(ge=0, le=100)
    annual_payment_usd: float = Field(ge=0)
    total_cost_usd: float = Field(ge=0)
    ownership_transfer: bool = Field(default=True)


class FinancingInput(BaseModel):
    project_id: str
    project_name: str = Field(default="Energy Project")
    project_cost_usd: float = Field(..., gt=0)
    annual_savings_usd: float = Field(..., ge=0)
    project_life_years: int = Field(default=15, ge=1)
    cost_of_capital_pct: float = Field(default=8, ge=0)
    options: List[FinancingOption] = Field(default_factory=list)
    tax_rate_pct: float = Field(default=25, ge=0, le=50)
    depreciation_years: int = Field(default=7, ge=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FinancingComparison(BaseModel):
    option_id: str
    financing_type: str
    total_payments_usd: float
    net_present_cost_usd: float
    effective_rate_pct: float
    cash_flow_impact_year1_usd: float
    cumulative_savings_usd: float
    rank: int


class FinancingOutput(BaseModel):
    project_id: str
    project_cost_usd: float
    comparisons: List[FinancingComparison]
    recommended_option_id: Optional[str]
    recommended_type: Optional[str]
    npv_advantage_usd: float
    cash_outlay_savings_usd: float
    financing_summary: str
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class FinancingAgent:
    AGENT_ID = "GL-089"
    AGENT_NAME = "FINANCING"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"FinancingAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = FinancingInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: FinancingInput) -> FinancingOutput:
        recommendations = []
        comparisons = []
        discount = inp.cost_of_capital_pct / 100

        # Add cash purchase as baseline
        cash_npc = inp.project_cost_usd
        cash_total = inp.project_cost_usd

        for option in inp.options:
            # Total payments
            if option.financing_type == FinancingType.CAPEX:
                total_payments = inp.project_cost_usd
                down = inp.project_cost_usd
            else:
                down = inp.project_cost_usd * (option.down_payment_pct / 100)
                total_payments = down + option.annual_payment_usd * option.term_years

            # NPV of payments
            npc = down
            for year in range(1, option.term_years + 1):
                npc += option.annual_payment_usd / (1 + discount) ** year

            # Effective rate
            if inp.project_cost_usd > 0 and option.term_years > 0:
                effective = ((total_payments / inp.project_cost_usd) ** (1 / option.term_years) - 1) * 100
            else:
                effective = 0

            # Year 1 cash flow impact
            year1_impact = down + option.annual_payment_usd - inp.annual_savings_usd

            # Cumulative savings over term
            cumulative = inp.annual_savings_usd * option.term_years - total_payments

            comparisons.append(FinancingComparison(
                option_id=option.option_id,
                financing_type=option.financing_type.value,
                total_payments_usd=round(total_payments, 2),
                net_present_cost_usd=round(npc, 2),
                effective_rate_pct=round(effective, 2),
                cash_flow_impact_year1_usd=round(year1_impact, 2),
                cumulative_savings_usd=round(cumulative, 2),
                rank=0
            ))

        # Rank by NPC
        comparisons.sort(key=lambda x: x.net_present_cost_usd)
        for i, c in enumerate(comparisons):
            c.rank = i + 1

        # Recommendation
        if comparisons:
            best = comparisons[0]
            worst = comparisons[-1]
            npv_advantage = worst.net_present_cost_usd - best.net_present_cost_usd

            rec_id = best.option_id
            rec_type = best.financing_type
            summary = f"Recommended: {rec_type} with NPC ${best.net_present_cost_usd:,.0f}"

            # Cash outlay savings vs upfront
            capex_option = next((c for c in comparisons if c.financing_type == "CAPEX"), None)
            if capex_option and best.financing_type != "CAPEX":
                cash_savings = inp.project_cost_usd - (inp.project_cost_usd * inp.options[0].down_payment_pct / 100)
            else:
                cash_savings = 0
        else:
            rec_id = None
            rec_type = None
            npv_advantage = 0
            cash_savings = 0
            summary = "No financing options provided"

        # Recommendations
        if len(inp.options) < 3:
            recommendations.append("Limited financing options - solicit additional quotes")

        negative_cf = [c for c in comparisons if c.cash_flow_impact_year1_usd > 0]
        if negative_cf:
            recommendations.append(f"{len(negative_cf)} options have negative Year 1 cash flow")

        low_rate = [c for c in comparisons if c.effective_rate_pct < inp.cost_of_capital_pct]
        if low_rate:
            recommendations.append(f"{len(low_rate)} options beat cost of capital - favorable terms")

        green_options = [o for o in inp.options if o.financing_type in [FinancingType.GREEN_BOND, FinancingType.PACE]]
        if not green_options:
            recommendations.append("Consider green financing options for lower rates")

        calc_hash = hashlib.sha256(json.dumps({
            "project": inp.project_id,
            "cost": inp.project_cost_usd,
            "recommended": rec_id
        }).encode()).hexdigest()

        return FinancingOutput(
            project_id=inp.project_id,
            project_cost_usd=inp.project_cost_usd,
            comparisons=comparisons,
            recommended_option_id=rec_id,
            recommended_type=rec_type,
            npv_advantage_usd=round(npv_advantage, 2),
            cash_outlay_savings_usd=round(cash_savings, 2),
            financing_summary=summary,
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-089", "name": "FINANCING", "version": "1.0.0",
    "summary": "Project financing optimization",
    "standards": [{"ref": "Project Finance"}, {"ref": "Green Bonds"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
