"""GL-092: Insurance Advisor Agent (INSURANCE-ADVISOR).

Advises on insurance for energy assets and projects.

Standards: ISO 31000, Insurance Industry Standards
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CoverageType(str, Enum):
    PROPERTY = "PROPERTY"
    BUSINESS_INTERRUPTION = "BUSINESS_INTERRUPTION"
    LIABILITY = "LIABILITY"
    EQUIPMENT_BREAKDOWN = "EQUIPMENT_BREAKDOWN"
    ENVIRONMENTAL = "ENVIRONMENTAL"
    CYBER = "CYBER"


class InsuranceNeed(BaseModel):
    coverage_type: CoverageType
    asset_value_usd: float = Field(ge=0)
    annual_revenue_at_risk_usd: float = Field(default=0, ge=0)
    risk_score: int = Field(ge=1, le=10)


class InsuranceAdvisorInput(BaseModel):
    facility_id: str
    facility_name: str = Field(default="Facility")
    insurance_needs: List[InsuranceNeed] = Field(default_factory=list)
    current_coverage_usd: float = Field(default=0, ge=0)
    current_premium_usd: float = Field(default=0, ge=0)
    deductible_usd: float = Field(default=10000, ge=0)
    loss_history_3yr_usd: float = Field(default=0, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CoverageRecommendation(BaseModel):
    coverage_type: str
    recommended_limit_usd: float
    estimated_premium_usd: float
    deductible_usd: float
    priority: str


class InsuranceAdvisorOutput(BaseModel):
    facility_id: str
    total_asset_value_usd: float
    total_recommended_coverage_usd: float
    coverage_gap_usd: float
    recommendations: List[CoverageRecommendation]
    total_estimated_premium_usd: float
    premium_as_pct_of_value: float
    risk_transfer_score: float
    advisory_notes: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class InsuranceAdvisorAgent:
    AGENT_ID = "GL-092"
    AGENT_NAME = "INSURANCE-ADVISOR"
    VERSION = "1.0.0"

    # Premium rates per coverage type (simplified)
    PREMIUM_RATES = {
        CoverageType.PROPERTY: 0.003,
        CoverageType.BUSINESS_INTERRUPTION: 0.004,
        CoverageType.LIABILITY: 0.002,
        CoverageType.EQUIPMENT_BREAKDOWN: 0.005,
        CoverageType.ENVIRONMENTAL: 0.006,
        CoverageType.CYBER: 0.008
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"InsuranceAdvisorAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = InsuranceAdvisorInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: InsuranceAdvisorInput) -> InsuranceAdvisorOutput:
        advisory_notes = []
        recommendations = []

        total_asset_value = sum(n.asset_value_usd for n in inp.insurance_needs)
        total_revenue_risk = sum(n.annual_revenue_at_risk_usd for n in inp.insurance_needs)

        total_recommended = 0
        total_premium = 0

        for need in inp.insurance_needs:
            # Recommended limit based on value and risk
            if need.coverage_type == CoverageType.BUSINESS_INTERRUPTION:
                limit = need.annual_revenue_at_risk_usd * 1.5  # 18 months coverage
            else:
                limit = need.asset_value_usd

            # Risk adjustment
            risk_factor = 1 + (need.risk_score - 5) * 0.1

            # Premium calculation
            base_rate = self.PREMIUM_RATES.get(need.coverage_type, 0.004)
            premium = limit * base_rate * risk_factor

            # Priority based on risk score
            if need.risk_score >= 8:
                priority = "HIGH"
            elif need.risk_score >= 5:
                priority = "MEDIUM"
            else:
                priority = "LOW"

            recommendations.append(CoverageRecommendation(
                coverage_type=need.coverage_type.value,
                recommended_limit_usd=round(limit, 2),
                estimated_premium_usd=round(premium, 2),
                deductible_usd=inp.deductible_usd,
                priority=priority
            ))

            total_recommended += limit
            total_premium += premium

        # Coverage gap
        gap = total_recommended - inp.current_coverage_usd

        # Premium as percentage
        premium_pct = (total_premium / total_asset_value * 100) if total_asset_value > 0 else 0

        # Risk transfer score (higher = more risk transferred)
        coverage_ratio = inp.current_coverage_usd / total_recommended if total_recommended > 0 else 0
        risk_transfer = min(100, coverage_ratio * 100)

        # Advisory notes
        if gap > 0:
            advisory_notes.append(f"Coverage gap of ${gap:,.0f} identified")
        if inp.loss_history_3yr_usd > inp.current_premium_usd * 3:
            advisory_notes.append("Loss history exceeds premiums - expect rate increase")
        if premium_pct > 1:
            advisory_notes.append(f"Premium cost {premium_pct:.2f}% of asset value - above typical")
        if inp.deductible_usd < total_asset_value * 0.001:
            advisory_notes.append("Low deductible increases premium - consider higher deductible")

        high_priority = [r for r in recommendations if r.priority == "HIGH"]
        if high_priority:
            advisory_notes.append(f"{len(high_priority)} high-priority coverage needs")

        calc_hash = hashlib.sha256(json.dumps({
            "facility": inp.facility_id,
            "recommended": round(total_recommended, 2),
            "gap": round(gap, 2)
        }).encode()).hexdigest()

        return InsuranceAdvisorOutput(
            facility_id=inp.facility_id,
            total_asset_value_usd=round(total_asset_value, 2),
            total_recommended_coverage_usd=round(total_recommended, 2),
            coverage_gap_usd=round(gap, 2),
            recommendations=recommendations,
            total_estimated_premium_usd=round(total_premium, 2),
            premium_as_pct_of_value=round(premium_pct, 3),
            risk_transfer_score=round(risk_transfer, 1),
            advisory_notes=advisory_notes,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-092", "name": "INSURANCE-ADVISOR", "version": "1.0.0",
    "summary": "Insurance advisory for energy assets",
    "standards": [{"ref": "ISO 31000"}, {"ref": "Insurance Industry Standards"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
