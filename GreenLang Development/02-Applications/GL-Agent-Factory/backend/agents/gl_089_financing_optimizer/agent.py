"""GL-089: Financing Optimizer Agent (FINANCING-OPTIMIZER).

Optimizes project financing structures.

Standards: Project Finance, DCF Analysis
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DebtType(str, Enum):
    TERM_LOAN = "TERM_LOAN"
    REVOLVER = "REVOLVER"
    MEZZANINE = "MEZZANINE"
    GREEN_LOAN = "GREEN_LOAN"
    BONDS = "BONDS"


class FinancingOptimizerInput(BaseModel):
    project_id: str
    project_cost_usd: float = Field(..., gt=0)
    equity_available_usd: float = Field(default=0, ge=0)
    target_debt_ratio_pct: float = Field(default=70, ge=0, le=100)
    project_irr_pct: float = Field(default=12, ge=0)
    debt_interest_rate_pct: float = Field(default=6, ge=0)
    debt_term_years: int = Field(default=10, ge=1)
    annual_revenue_usd: float = Field(default=0, ge=0)
    annual_opex_usd: float = Field(default=0, ge=0)
    tax_rate_pct: float = Field(default=25, ge=0, le=50)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FinancingOptimizerOutput(BaseModel):
    project_id: str
    optimal_debt_usd: float
    optimal_equity_usd: float
    debt_to_equity_ratio: float
    dscr: float
    equity_irr_pct: float
    wacc_pct: float
    annual_debt_service_usd: float
    financing_feasibility: str
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class FinancingOptimizerAgent:
    AGENT_ID = "GL-089B"
    AGENT_NAME = "FINANCING-OPTIMIZER"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"FinancingOptimizerAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = FinancingOptimizerInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _calculate_debt_service(self, principal: float, rate: float, term: int) -> float:
        """Calculate annual debt service (principal + interest)."""
        if rate == 0:
            return principal / term if term > 0 else 0
        r = rate / 100
        pmt = principal * (r * (1 + r) ** term) / ((1 + r) ** term - 1)
        return pmt

    def _process(self, inp: FinancingOptimizerInput) -> FinancingOptimizerOutput:
        recommendations = []

        # Optimal financing structure
        target_debt = inp.project_cost_usd * (inp.target_debt_ratio_pct / 100)
        equity_needed = inp.project_cost_usd - target_debt

        # Check equity availability
        if inp.equity_available_usd < equity_needed:
            equity_gap = equity_needed - inp.equity_available_usd
            recommendations.append(f"Equity gap ${equity_gap:,.0f} - seek additional investors")
            actual_equity = inp.equity_available_usd
            actual_debt = inp.project_cost_usd - actual_equity
        else:
            actual_equity = equity_needed
            actual_debt = target_debt

        # Debt to equity ratio
        de_ratio = actual_debt / actual_equity if actual_equity > 0 else float('inf')

        # Annual debt service
        debt_service = self._calculate_debt_service(actual_debt, inp.debt_interest_rate_pct, inp.debt_term_years)

        # DSCR (Debt Service Coverage Ratio)
        noi = inp.annual_revenue_usd - inp.annual_opex_usd
        dscr = noi / debt_service if debt_service > 0 else float('inf')

        # Equity IRR (simplified - project IRR leveraged by debt)
        # Equity IRR ≈ Project IRR + (Project IRR - Debt Cost) * D/E
        leverage_boost = (inp.project_irr_pct - inp.debt_interest_rate_pct) * de_ratio if de_ratio < 10 else 0
        equity_irr = inp.project_irr_pct + leverage_boost

        # WACC
        debt_weight = actual_debt / inp.project_cost_usd if inp.project_cost_usd > 0 else 0
        equity_weight = actual_equity / inp.project_cost_usd if inp.project_cost_usd > 0 else 0
        cost_of_debt = inp.debt_interest_rate_pct * (1 - inp.tax_rate_pct / 100)
        cost_of_equity = inp.project_irr_pct + 5  # Risk premium
        wacc = debt_weight * cost_of_debt + equity_weight * cost_of_equity

        # Feasibility assessment
        if dscr >= 1.4 and de_ratio <= 4:
            feasibility = "HIGHLY FEASIBLE"
        elif dscr >= 1.2 and de_ratio <= 6:
            feasibility = "FEASIBLE"
        elif dscr >= 1.0:
            feasibility = "MARGINAL"
        else:
            feasibility = "NOT FEASIBLE"

        # Recommendations
        if dscr < 1.2:
            recommendations.append(f"Low DSCR {dscr:.2f} - lenders typically require >1.25")
        if dscr > 2.0:
            recommendations.append(f"High DSCR {dscr:.2f} - room for additional debt")

        if de_ratio > 4:
            recommendations.append(f"High leverage {de_ratio:.1f}x - consider more equity")

        if equity_irr > inp.project_irr_pct + 10:
            recommendations.append(f"Strong equity returns {equity_irr:.1f}% from leverage")

        if inp.debt_interest_rate_pct > 8:
            recommendations.append("High debt cost - explore green financing options")

        calc_hash = hashlib.sha256(json.dumps({
            "project": inp.project_id,
            "debt": round(actual_debt, 2),
            "dscr": round(dscr, 2)
        }).encode()).hexdigest()

        return FinancingOptimizerOutput(
            project_id=inp.project_id,
            optimal_debt_usd=round(actual_debt, 2),
            optimal_equity_usd=round(actual_equity, 2),
            debt_to_equity_ratio=round(de_ratio, 2),
            dscr=round(dscr, 2),
            equity_irr_pct=round(equity_irr, 1),
            wacc_pct=round(wacc, 2),
            annual_debt_service_usd=round(debt_service, 2),
            financing_feasibility=feasibility,
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-089B", "name": "FINANCING-OPTIMIZER", "version": "1.0.0",
    "summary": "Project financing structure optimization",
    "standards": [{"ref": "Project Finance"}, {"ref": "DCF Analysis"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
