"""GL-086: Risk Assessment Agent (RISK).

Assesses energy and climate-related risks.

Standards: ISO 31000, TCFD
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RiskCategory(str, Enum):
    TRANSITION = "TRANSITION"
    PHYSICAL = "PHYSICAL"
    REGULATORY = "REGULATORY"
    MARKET = "MARKET"
    TECHNOLOGY = "TECHNOLOGY"
    REPUTATION = "REPUTATION"


class RiskItem(BaseModel):
    risk_id: str
    description: str
    category: RiskCategory
    likelihood: int = Field(ge=1, le=5)
    impact: int = Field(ge=1, le=5)
    time_horizon_years: int = Field(ge=1)
    financial_exposure_usd: float = Field(default=0, ge=0)


class RiskInput(BaseModel):
    facility_id: str
    facility_name: str = Field(default="Facility")
    risks: List[RiskItem] = Field(default_factory=list)
    annual_revenue_usd: float = Field(default=100000000, gt=0)
    carbon_intensity_tonnes_per_m: float = Field(default=100, ge=0)
    carbon_price_current_usd: float = Field(default=50, ge=0)
    carbon_price_projected_usd: float = Field(default=150, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RiskAssessment(BaseModel):
    risk_id: str
    description: str
    category: str
    risk_score: int
    risk_level: str
    financial_impact_usd: float
    mitigation_priority: str


class RiskOutput(BaseModel):
    facility_id: str
    total_risks_assessed: int
    high_risks: int
    medium_risks: int
    low_risks: int
    risk_assessments: List[RiskAssessment]
    total_financial_exposure_usd: float
    carbon_price_risk_usd: float
    risk_adjusted_exposure_usd: float
    overall_risk_score: float
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class RiskAssessmentAgent:
    AGENT_ID = "GL-086"
    AGENT_NAME = "RISK"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"RiskAssessmentAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = RiskInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: RiskInput) -> RiskOutput:
        recommendations = []
        assessments = []
        high = medium = low = 0
        total_exposure = 0

        for risk in inp.risks:
            # Risk score = likelihood * impact
            score = risk.likelihood * risk.impact

            # Risk level
            if score >= 16:
                level = "HIGH"
                priority = "IMMEDIATE"
                high += 1
            elif score >= 9:
                level = "MEDIUM"
                priority = "SHORT_TERM"
                medium += 1
            else:
                level = "LOW"
                priority = "MONITOR"
                low += 1

            # Financial impact (exposure adjusted by probability)
            probability = risk.likelihood / 5
            financial_impact = risk.financial_exposure_usd * probability
            total_exposure += financial_impact

            assessments.append(RiskAssessment(
                risk_id=risk.risk_id,
                description=risk.description,
                category=risk.category.value,
                risk_score=score,
                risk_level=level,
                financial_impact_usd=round(financial_impact, 2),
                mitigation_priority=priority
            ))

        # Sort by risk score
        assessments.sort(key=lambda x: -x.risk_score)

        # Carbon price risk
        revenue_m = inp.annual_revenue_usd / 1000000
        annual_emissions = inp.carbon_intensity_tonnes_per_m * revenue_m
        carbon_price_increase = inp.carbon_price_projected_usd - inp.carbon_price_current_usd
        carbon_risk = annual_emissions * carbon_price_increase

        # Risk-adjusted exposure
        risk_adjusted = total_exposure + carbon_risk

        # Overall risk score (weighted average)
        if inp.risks:
            overall = sum(r.likelihood * r.impact for r in inp.risks) / len(inp.risks)
        else:
            overall = 0

        # Recommendations
        if high > 0:
            recommendations.append(f"{high} HIGH risks require immediate mitigation")
        if carbon_risk > inp.annual_revenue_usd * 0.01:
            recommendations.append(f"Carbon price risk ${carbon_risk:,.0f} exceeds 1% of revenue")

        transition_risks = [r for r in inp.risks if r.category == RiskCategory.TRANSITION]
        if transition_risks:
            recommendations.append(f"{len(transition_risks)} transition risks - accelerate decarbonization")

        physical_risks = [r for r in inp.risks if r.category == RiskCategory.PHYSICAL]
        if physical_risks:
            recommendations.append(f"{len(physical_risks)} physical risks - assess climate adaptation needs")

        if overall > 15:
            recommendations.append("Overall risk score HIGH - comprehensive risk management required")
        elif overall > 10:
            recommendations.append("Overall risk score ELEVATED - prioritize mitigation actions")

        calc_hash = hashlib.sha256(json.dumps({
            "facility": inp.facility_id,
            "total_exposure": round(total_exposure, 2),
            "overall_score": round(overall, 1)
        }).encode()).hexdigest()

        return RiskOutput(
            facility_id=inp.facility_id,
            total_risks_assessed=len(inp.risks),
            high_risks=high,
            medium_risks=medium,
            low_risks=low,
            risk_assessments=assessments,
            total_financial_exposure_usd=round(total_exposure, 2),
            carbon_price_risk_usd=round(carbon_risk, 2),
            risk_adjusted_exposure_usd=round(risk_adjusted, 2),
            overall_risk_score=round(overall, 1),
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-086", "name": "RISK", "version": "1.0.0",
    "summary": "Energy and climate risk assessment",
    "standards": [{"ref": "ISO 31000"}, {"ref": "TCFD"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
