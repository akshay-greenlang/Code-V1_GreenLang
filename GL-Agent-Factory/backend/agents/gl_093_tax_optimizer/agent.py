"""GL-093: Tax Optimizer Agent (TAX-OPTIMIZER).

Optimizes tax strategies for energy projects.

Standards: IRS Guidelines, Tax Code
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TaxIncentive(str, Enum):
    ITC = "ITC"  # Investment Tax Credit
    PTC = "PTC"  # Production Tax Credit
    MACRS = "MACRS"  # Accelerated Depreciation
    BONUS_DEPRECIATION = "BONUS_DEPRECIATION"
    SECTION_179 = "SECTION_179"
    STATE_CREDIT = "STATE_CREDIT"


class EligibleAsset(BaseModel):
    asset_id: str
    asset_name: str
    cost_basis_usd: float = Field(ge=0)
    placed_in_service_year: int = Field(ge=2020)
    asset_class: str = Field(default="solar")
    eligible_incentives: List[TaxIncentive] = Field(default_factory=list)


class TaxOptimizerInput(BaseModel):
    entity_id: str
    entity_name: str = Field(default="Entity")
    tax_year: int = Field(default=2024)
    assets: List[EligibleAsset] = Field(default_factory=list)
    taxable_income_usd: float = Field(default=1000000, ge=0)
    marginal_tax_rate_pct: float = Field(default=25, ge=0, le=50)
    state_tax_rate_pct: float = Field(default=5, ge=0, le=15)
    can_monetize_credits: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaxBenefit(BaseModel):
    asset_id: str
    incentive_type: str
    benefit_value_usd: float
    year_claimed: int
    carryforward_available: bool


class TaxOptimizerOutput(BaseModel):
    entity_id: str
    tax_year: int
    total_tax_benefits_usd: float
    federal_credits_usd: float
    state_credits_usd: float
    depreciation_deduction_usd: float
    tax_savings_usd: float
    effective_project_cost_reduction_pct: float
    benefits_by_asset: List[TaxBenefit]
    optimization_strategy: str
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class TaxOptimizerAgent:
    AGENT_ID = "GL-093"
    AGENT_NAME = "TAX-OPTIMIZER"
    VERSION = "1.0.0"

    # ITC rates by asset class
    ITC_RATES = {"solar": 0.30, "storage": 0.30, "geothermal": 0.30, "fuel_cell": 0.30}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"TaxOptimizerAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = TaxOptimizerInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: TaxOptimizerInput) -> TaxOptimizerOutput:
        recommendations = []
        benefits = []

        total_cost_basis = sum(a.cost_basis_usd for a in inp.assets)
        federal_credits = 0
        state_credits = 0
        depreciation = 0

        for asset in inp.assets:
            # ITC
            if TaxIncentive.ITC in asset.eligible_incentives:
                itc_rate = self.ITC_RATES.get(asset.asset_class, 0.30)
                itc_value = asset.cost_basis_usd * itc_rate
                federal_credits += itc_value

                benefits.append(TaxBenefit(
                    asset_id=asset.asset_id,
                    incentive_type="ITC",
                    benefit_value_usd=round(itc_value, 2),
                    year_claimed=inp.tax_year,
                    carryforward_available=True
                ))

            # MACRS Depreciation (5-year for solar)
            if TaxIncentive.MACRS in asset.eligible_incentives:
                # First year MACRS rate
                macrs_rate = 0.20
                # Reduce basis by half of ITC if applicable
                itc_basis_reduction = 0.5 if TaxIncentive.ITC in asset.eligible_incentives else 0
                adjusted_basis = asset.cost_basis_usd * (1 - self.ITC_RATES.get(asset.asset_class, 0) * itc_basis_reduction)
                macrs_value = adjusted_basis * macrs_rate
                depreciation += macrs_value

                benefits.append(TaxBenefit(
                    asset_id=asset.asset_id,
                    incentive_type="MACRS",
                    benefit_value_usd=round(macrs_value, 2),
                    year_claimed=inp.tax_year,
                    carryforward_available=False
                ))

            # Bonus Depreciation
            if TaxIncentive.BONUS_DEPRECIATION in asset.eligible_incentives:
                bonus_rate = 0.60 if inp.tax_year >= 2024 else 0.80
                bonus_value = asset.cost_basis_usd * bonus_rate
                depreciation += bonus_value

                benefits.append(TaxBenefit(
                    asset_id=asset.asset_id,
                    incentive_type="BONUS_DEPRECIATION",
                    benefit_value_usd=round(bonus_value, 2),
                    year_claimed=inp.tax_year,
                    carryforward_available=False
                ))

            # State credit
            if TaxIncentive.STATE_CREDIT in asset.eligible_incentives:
                state_value = asset.cost_basis_usd * 0.10  # Typical state credit
                state_credits += state_value

                benefits.append(TaxBenefit(
                    asset_id=asset.asset_id,
                    incentive_type="STATE_CREDIT",
                    benefit_value_usd=round(state_value, 2),
                    year_claimed=inp.tax_year,
                    carryforward_available=True
                ))

        # Tax savings from depreciation
        combined_rate = (inp.marginal_tax_rate_pct + inp.state_tax_rate_pct) / 100
        depreciation_savings = depreciation * combined_rate

        # Total benefits
        total_benefits = federal_credits + state_credits + depreciation_savings

        # Tax savings
        tax_savings = federal_credits + state_credits + depreciation_savings

        # Cost reduction percentage
        cost_reduction = (total_benefits / total_cost_basis * 100) if total_cost_basis > 0 else 0

        # Strategy
        if federal_credits > inp.taxable_income_usd * combined_rate:
            strategy = "CREDIT_MONETIZATION_RECOMMENDED"
            recommendations.append("Tax credits exceed tax liability - consider credit monetization")
        else:
            strategy = "DIRECT_USE"

        # Recommendations
        if not any(TaxIncentive.ITC in a.eligible_incentives for a in inp.assets):
            recommendations.append("Review ITC eligibility - 30% credit available for qualifying assets")

        if cost_reduction > 40:
            recommendations.append(f"Strong tax benefits ({cost_reduction:.0f}% cost reduction)")

        if inp.can_monetize_credits and federal_credits > 100000:
            recommendations.append("Significant credits - consider tax equity partnership")

        calc_hash = hashlib.sha256(json.dumps({
            "entity": inp.entity_id,
            "year": inp.tax_year,
            "benefits": round(total_benefits, 2)
        }).encode()).hexdigest()

        return TaxOptimizerOutput(
            entity_id=inp.entity_id,
            tax_year=inp.tax_year,
            total_tax_benefits_usd=round(total_benefits, 2),
            federal_credits_usd=round(federal_credits, 2),
            state_credits_usd=round(state_credits, 2),
            depreciation_deduction_usd=round(depreciation, 2),
            tax_savings_usd=round(tax_savings, 2),
            effective_project_cost_reduction_pct=round(cost_reduction, 1),
            benefits_by_asset=benefits,
            optimization_strategy=strategy,
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-093", "name": "TAX-OPTIMIZER", "version": "1.0.0",
    "summary": "Tax optimization for energy projects",
    "standards": [{"ref": "IRS Guidelines"}, {"ref": "Tax Code"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
