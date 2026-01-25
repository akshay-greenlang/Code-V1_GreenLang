"""GL-083: Procurement Advisor Agent (PROCUREMENT-ADVISOR).

Optimizes energy procurement and sourcing strategies.

Standards: ISO 50001, NAPM Guidelines
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProcurementStrategy(str, Enum):
    FIXED_PRICE = "FIXED_PRICE"
    INDEX_PLUS = "INDEX_PLUS"
    BLOCK_AND_INDEX = "BLOCK_AND_INDEX"
    SPOT_MARKET = "SPOT_MARKET"
    PPA = "PPA"


class EnergyProduct(str, Enum):
    ELECTRICITY = "ELECTRICITY"
    NATURAL_GAS = "NATURAL_GAS"
    RENEWABLE = "RENEWABLE"
    REC = "REC"


class ProcurementOption(BaseModel):
    option_id: str
    supplier: str
    product: EnergyProduct
    strategy: ProcurementStrategy
    price_per_unit: float = Field(ge=0)
    contract_term_months: int = Field(ge=1)
    volume_commitment_pct: float = Field(ge=0, le=100)
    green_premium_pct: float = Field(default=0, ge=0)
    risk_score: float = Field(ge=0, le=100)


class ProcurementAdvisorInput(BaseModel):
    facility_id: str
    annual_electricity_mwh: float = Field(default=10000, ge=0)
    annual_gas_therms: float = Field(default=100000, ge=0)
    current_elec_rate_mwh: float = Field(default=80, ge=0)
    current_gas_rate_therm: float = Field(default=0.80, ge=0)
    risk_tolerance: str = Field(default="MODERATE")
    renewable_target_pct: float = Field(default=50, ge=0, le=100)
    budget_variance_tolerance_pct: float = Field(default=10, ge=0)
    options: List[ProcurementOption] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RecommendedStrategy(BaseModel):
    product: str
    recommended_strategy: str
    suppliers: List[str]
    blended_price: float
    annual_cost_usd: float
    savings_vs_current_usd: float
    risk_level: str
    renewable_content_pct: float


class ProcurementAdvisorOutput(BaseModel):
    facility_id: str
    current_annual_cost_usd: float
    recommended_strategies: List[RecommendedStrategy]
    total_recommended_cost_usd: float
    total_savings_usd: float
    savings_pct: float
    renewable_achievement_pct: float
    budget_risk_assessment: str
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class ProcurementAdvisorAgent:
    AGENT_ID = "GL-083"
    AGENT_NAME = "PROCUREMENT-ADVISOR"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"ProcurementAdvisorAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = ProcurementAdvisorInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _select_best_options(self, options: List[ProcurementOption], product: EnergyProduct,
                             risk_tolerance: str, renewable_target: float) -> List[ProcurementOption]:
        """Select best procurement options for a product."""
        product_options = [o for o in options if o.product == product]
        if not product_options:
            return []

        # Risk filter
        if risk_tolerance == "LOW":
            max_risk = 30
        elif risk_tolerance == "HIGH":
            max_risk = 80
        else:
            max_risk = 50

        filtered = [o for o in product_options if o.risk_score <= max_risk]
        if not filtered:
            filtered = product_options

        # Sort by price
        filtered.sort(key=lambda x: x.price_per_unit * (1 + x.green_premium_pct/100))

        return filtered[:3]

    def _process(self, inp: ProcurementAdvisorInput) -> ProcurementAdvisorOutput:
        recommendations = []

        # Current costs
        current_elec_cost = inp.annual_electricity_mwh * inp.current_elec_rate_mwh
        current_gas_cost = inp.annual_gas_therms * inp.current_gas_rate_therm
        current_total = current_elec_cost + current_gas_cost

        strategies = []
        total_recommended = 0
        renewable_achieved = 0

        # Electricity strategy
        elec_options = self._select_best_options(inp.options, EnergyProduct.ELECTRICITY,
                                                 inp.risk_tolerance, inp.renewable_target_pct)
        if elec_options:
            best = elec_options[0]
            cost = inp.annual_electricity_mwh * best.price_per_unit * (1 + best.green_premium_pct/100)
            savings = current_elec_cost - cost

            strategies.append(RecommendedStrategy(
                product="ELECTRICITY",
                recommended_strategy=best.strategy.value,
                suppliers=[best.supplier],
                blended_price=round(best.price_per_unit, 2),
                annual_cost_usd=round(cost, 2),
                savings_vs_current_usd=round(savings, 2),
                risk_level="LOW" if best.risk_score < 30 else "MEDIUM" if best.risk_score < 60 else "HIGH",
                renewable_content_pct=100 if best.product == EnergyProduct.RENEWABLE else best.green_premium_pct
            ))
            total_recommended += cost

        # Gas strategy
        gas_options = self._select_best_options(inp.options, EnergyProduct.NATURAL_GAS,
                                                inp.risk_tolerance, 0)
        if gas_options:
            best = gas_options[0]
            cost = inp.annual_gas_therms * best.price_per_unit
            savings = current_gas_cost - cost

            strategies.append(RecommendedStrategy(
                product="NATURAL_GAS",
                recommended_strategy=best.strategy.value,
                suppliers=[best.supplier],
                blended_price=round(best.price_per_unit, 4),
                annual_cost_usd=round(cost, 2),
                savings_vs_current_usd=round(savings, 2),
                risk_level="LOW" if best.risk_score < 30 else "MEDIUM" if best.risk_score < 60 else "HIGH",
                renewable_content_pct=0
            ))
            total_recommended += cost

        # Renewable strategy (RECs/PPAs)
        renewable_options = self._select_best_options(inp.options, EnergyProduct.RENEWABLE,
                                                      inp.risk_tolerance, inp.renewable_target_pct)
        if renewable_options and inp.renewable_target_pct > 0:
            best = renewable_options[0]
            renewable_mwh = inp.annual_electricity_mwh * (inp.renewable_target_pct / 100)
            cost = renewable_mwh * best.price_per_unit

            strategies.append(RecommendedStrategy(
                product="RENEWABLE",
                recommended_strategy=best.strategy.value,
                suppliers=[best.supplier],
                blended_price=round(best.price_per_unit, 2),
                annual_cost_usd=round(cost, 2),
                savings_vs_current_usd=0,
                risk_level="LOW",
                renewable_content_pct=100
            ))
            renewable_achieved = inp.renewable_target_pct

        # Calculate totals
        if not strategies:
            total_recommended = current_total

        total_savings = current_total - total_recommended
        savings_pct = (total_savings / current_total * 100) if current_total > 0 else 0

        # Budget risk
        if savings_pct < -inp.budget_variance_tolerance_pct:
            budget_risk = "HIGH - Over budget"
        elif savings_pct < 0:
            budget_risk = "MEDIUM - Slight increase"
        else:
            budget_risk = "LOW - Within budget"

        # Recommendations
        if not inp.options:
            recommendations.append("No procurement options provided - solicit supplier quotes")
        if savings_pct > 10:
            recommendations.append(f"Significant savings potential ({savings_pct:.1f}%) - proceed with procurement")
        if renewable_achieved < inp.renewable_target_pct:
            recommendations.append(f"Renewable target gap: {inp.renewable_target_pct - renewable_achieved:.0f}% shortfall")
        if inp.risk_tolerance == "LOW":
            recommendations.append("Low risk tolerance - prioritize fixed-price contracts")

        calc_hash = hashlib.sha256(json.dumps({
            "facility": inp.facility_id,
            "recommended_cost": round(total_recommended, 2),
            "savings_pct": round(savings_pct, 1)
        }).encode()).hexdigest()

        return ProcurementAdvisorOutput(
            facility_id=inp.facility_id,
            current_annual_cost_usd=round(current_total, 2),
            recommended_strategies=strategies,
            total_recommended_cost_usd=round(total_recommended, 2),
            total_savings_usd=round(total_savings, 2),
            savings_pct=round(savings_pct, 1),
            renewable_achievement_pct=round(renewable_achieved, 1),
            budget_risk_assessment=budget_risk,
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-083", "name": "PROCUREMENT-ADVISOR", "version": "1.0.0",
    "summary": "Energy procurement and sourcing optimization",
    "standards": [{"ref": "ISO 50001"}, {"ref": "NAPM Guidelines"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
