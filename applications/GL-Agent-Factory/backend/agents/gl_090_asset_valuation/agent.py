"""GL-090: Asset Valuation Agent (ASSET-VALUATION).

Values energy assets for transactions and reporting.

Standards: IAS 36, IFRS 16, ASC 842
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ValuationMethod(str, Enum):
    DCF = "DCF"
    MARKET_COMPARABLE = "MARKET_COMPARABLE"
    REPLACEMENT_COST = "REPLACEMENT_COST"
    INCOME_APPROACH = "INCOME_APPROACH"


class AssetType(str, Enum):
    GENERATION = "GENERATION"
    STORAGE = "STORAGE"
    EFFICIENCY_EQUIPMENT = "EFFICIENCY_EQUIPMENT"
    INFRASTRUCTURE = "INFRASTRUCTURE"


class AssetValuationInput(BaseModel):
    asset_id: str
    asset_name: str = Field(default="Energy Asset")
    asset_type: AssetType = Field(default=AssetType.EFFICIENCY_EQUIPMENT)
    original_cost_usd: float = Field(..., ge=0)
    installation_year: int = Field(..., ge=1990)
    useful_life_years: int = Field(default=20, ge=1)
    annual_revenue_usd: float = Field(default=0, ge=0)
    annual_opex_usd: float = Field(default=0, ge=0)
    capacity_kw: float = Field(default=0, ge=0)
    capacity_factor_pct: float = Field(default=0, ge=0, le=100)
    discount_rate_pct: float = Field(default=8, ge=0)
    terminal_value_multiple: float = Field(default=5, ge=0)
    market_comparables_per_kw: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AssetValuationOutput(BaseModel):
    asset_id: str
    asset_name: str
    dcf_value_usd: float
    market_value_usd: Optional[float]
    replacement_value_usd: float
    book_value_usd: float
    fair_value_usd: float
    remaining_life_years: int
    annual_depreciation_usd: float
    impairment_indicator: bool
    valuation_confidence: str
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class AssetValuationAgent:
    AGENT_ID = "GL-090"
    AGENT_NAME = "ASSET-VALUATION"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"AssetValuationAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = AssetValuationInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: AssetValuationInput) -> AssetValuationOutput:
        recommendations = []
        current_year = datetime.utcnow().year

        # Age and remaining life
        age = current_year - inp.installation_year
        remaining_life = max(0, inp.useful_life_years - age)

        # Annual depreciation (straight-line)
        annual_depreciation = inp.original_cost_usd / inp.useful_life_years if inp.useful_life_years > 0 else 0

        # Book value
        accumulated_depreciation = annual_depreciation * age
        book_value = max(0, inp.original_cost_usd - accumulated_depreciation)

        # DCF valuation
        discount = inp.discount_rate_pct / 100
        annual_ncf = inp.annual_revenue_usd - inp.annual_opex_usd
        dcf = 0

        for year in range(1, remaining_life + 1):
            dcf += annual_ncf / (1 + discount) ** year

        # Terminal value
        if remaining_life > 0 and annual_ncf > 0:
            terminal = annual_ncf * inp.terminal_value_multiple / (1 + discount) ** remaining_life
        else:
            terminal = 0

        dcf_value = dcf + terminal

        # Market comparable
        if inp.market_comparables_per_kw and inp.capacity_kw > 0:
            market_value = inp.capacity_kw * inp.market_comparables_per_kw
        else:
            market_value = None

        # Replacement cost (adjusted for inflation ~3%/year)
        inflation_factor = 1.03 ** age
        replacement_value = inp.original_cost_usd * inflation_factor

        # Fair value (weighted average of available methods)
        values = [dcf_value]
        if market_value:
            values.append(market_value)

        fair_value = sum(values) / len(values)

        # Impairment indicator
        impairment = fair_value < book_value * 0.9

        # Confidence assessment
        if market_value and abs(dcf_value - market_value) / fair_value < 0.2:
            confidence = "HIGH"
        elif remaining_life > 5 and annual_ncf > 0:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # Recommendations
        if impairment:
            recommendations.append(f"Impairment indicator: Fair value ${fair_value:,.0f} < Book value ${book_value:,.0f}")

        if remaining_life < 3:
            recommendations.append(f"Asset near end of life ({remaining_life} years) - plan replacement")

        if dcf_value < 0:
            recommendations.append("Negative DCF value - review operating economics")

        if market_value and dcf_value > 0:
            premium = (dcf_value - market_value) / market_value * 100 if market_value > 0 else 0
            if premium > 20:
                recommendations.append(f"DCF {premium:.0f}% above market - verify assumptions")
            elif premium < -20:
                recommendations.append(f"DCF {abs(premium):.0f}% below market - potential undervaluation")

        if inp.capacity_factor_pct < 20 and inp.asset_type == AssetType.GENERATION:
            recommendations.append(f"Low capacity factor ({inp.capacity_factor_pct}%) impacts value")

        calc_hash = hashlib.sha256(json.dumps({
            "asset": inp.asset_id,
            "fair_value": round(fair_value, 2),
            "book_value": round(book_value, 2)
        }).encode()).hexdigest()

        return AssetValuationOutput(
            asset_id=inp.asset_id,
            asset_name=inp.asset_name,
            dcf_value_usd=round(dcf_value, 2),
            market_value_usd=round(market_value, 2) if market_value else None,
            replacement_value_usd=round(replacement_value, 2),
            book_value_usd=round(book_value, 2),
            fair_value_usd=round(fair_value, 2),
            remaining_life_years=remaining_life,
            annual_depreciation_usd=round(annual_depreciation, 2),
            impairment_indicator=impairment,
            valuation_confidence=confidence,
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-090", "name": "ASSET-VALUATION", "version": "1.0.0",
    "summary": "Energy asset valuation",
    "standards": [{"ref": "IAS 36"}, {"ref": "IFRS 16"}, {"ref": "ASC 842"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
