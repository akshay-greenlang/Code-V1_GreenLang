"""GL-090: Asset Value Agent (ASSET-VALUE).

Determines asset value for energy infrastructure.

Standards: FASB ASC 820, IAS 36
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AssetCategory(str, Enum):
    BOILER = "BOILER"
    CHILLER = "CHILLER"
    SOLAR = "SOLAR"
    WIND = "WIND"
    BATTERY = "BATTERY"
    CHP = "CHP"


class AssetValueInput(BaseModel):
    asset_id: str
    asset_category: AssetCategory = Field(default=AssetCategory.BOILER)
    acquisition_cost_usd: float = Field(..., ge=0)
    acquisition_year: int = Field(..., ge=1980)
    expected_life_years: int = Field(default=20, ge=1)
    salvage_value_pct: float = Field(default=10, ge=0, le=50)
    annual_output_kwh: float = Field(default=100000, ge=0)
    annual_operating_cost_usd: float = Field(default=10000, ge=0)
    energy_price_kwh: float = Field(default=0.10, ge=0)
    discount_rate_pct: float = Field(default=8, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AssetValueOutput(BaseModel):
    asset_id: str
    asset_category: str
    current_book_value_usd: float
    economic_value_usd: float
    liquidation_value_usd: float
    value_in_use_usd: float
    accumulated_depreciation_usd: float
    remaining_useful_life_years: int
    value_per_kwh_capacity: float
    roi_to_date_pct: float
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class AssetValueAgent:
    AGENT_ID = "GL-090B"
    AGENT_NAME = "ASSET-VALUE"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"AssetValueAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = AssetValueInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: AssetValueInput) -> AssetValueOutput:
        recommendations = []
        current_year = datetime.utcnow().year

        # Age and remaining life
        age = current_year - inp.acquisition_year
        remaining_life = max(0, inp.expected_life_years - age)

        # Salvage value
        salvage = inp.acquisition_cost_usd * (inp.salvage_value_pct / 100)

        # Depreciation (straight-line)
        depreciable = inp.acquisition_cost_usd - salvage
        annual_depreciation = depreciable / inp.expected_life_years if inp.expected_life_years > 0 else 0
        accumulated_depreciation = min(depreciable, annual_depreciation * age)

        # Book value
        book_value = inp.acquisition_cost_usd - accumulated_depreciation

        # Value in use (DCF of future cash flows)
        discount = inp.discount_rate_pct / 100
        annual_revenue = inp.annual_output_kwh * inp.energy_price_kwh
        annual_ncf = annual_revenue - inp.annual_operating_cost_usd

        value_in_use = 0
        for year in range(1, remaining_life + 1):
            value_in_use += annual_ncf / (1 + discount) ** year

        # Add salvage
        if remaining_life > 0:
            value_in_use += salvage / (1 + discount) ** remaining_life

        # Economic value (higher of value in use and liquidation)
        liquidation = salvage * 0.8  # Quick sale discount
        economic_value = max(value_in_use, liquidation)

        # Value per kWh capacity
        value_per_kwh = economic_value / inp.annual_output_kwh if inp.annual_output_kwh > 0 else 0

        # ROI to date
        total_revenue_to_date = annual_revenue * age
        total_cost_to_date = inp.acquisition_cost_usd + inp.annual_operating_cost_usd * age
        roi = ((total_revenue_to_date - total_cost_to_date) / inp.acquisition_cost_usd * 100) if inp.acquisition_cost_usd > 0 else 0

        # Recommendations
        if book_value > economic_value:
            impairment = book_value - economic_value
            recommendations.append(f"Potential impairment ${impairment:,.0f} - book value exceeds recoverable amount")

        if remaining_life < 3:
            recommendations.append(f"Asset near end of life ({remaining_life} years) - plan replacement")

        if value_in_use < 0:
            recommendations.append("Negative value in use - consider early retirement")

        if roi < 0:
            recommendations.append(f"Negative ROI to date ({roi:.1f}%) - review operating performance")

        if annual_ncf < 0:
            recommendations.append("Negative cash flows - operating costs exceed revenue")

        calc_hash = hashlib.sha256(json.dumps({
            "asset": inp.asset_id,
            "book_value": round(book_value, 2),
            "economic_value": round(economic_value, 2)
        }).encode()).hexdigest()

        return AssetValueOutput(
            asset_id=inp.asset_id,
            asset_category=inp.asset_category.value,
            current_book_value_usd=round(book_value, 2),
            economic_value_usd=round(economic_value, 2),
            liquidation_value_usd=round(liquidation, 2),
            value_in_use_usd=round(value_in_use, 2),
            accumulated_depreciation_usd=round(accumulated_depreciation, 2),
            remaining_useful_life_years=remaining_life,
            value_per_kwh_capacity=round(value_per_kwh, 4),
            roi_to_date_pct=round(roi, 1),
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-090B", "name": "ASSET-VALUE", "version": "1.0.0",
    "summary": "Energy asset value determination",
    "standards": [{"ref": "FASB ASC 820"}, {"ref": "IAS 36"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
