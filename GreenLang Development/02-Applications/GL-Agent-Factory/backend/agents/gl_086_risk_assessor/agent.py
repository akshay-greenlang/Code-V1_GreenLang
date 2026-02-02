"""GL-086: Risk Assessor Agent (RISK-ASSESSOR).

Comprehensive risk assessment for energy projects.

Standards: ISO 31000, PMI Risk Management
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProjectRiskType(str, Enum):
    TECHNICAL = "TECHNICAL"
    SCHEDULE = "SCHEDULE"
    COST = "COST"
    PERFORMANCE = "PERFORMANCE"
    SUPPLY_CHAIN = "SUPPLY_CHAIN"
    REGULATORY = "REGULATORY"


class ProjectRisk(BaseModel):
    risk_id: str
    name: str
    risk_type: ProjectRiskType
    probability_pct: float = Field(ge=0, le=100)
    cost_impact_usd: float = Field(default=0, ge=0)
    schedule_impact_days: int = Field(default=0, ge=0)
    mitigation_available: bool = Field(default=True)
    mitigation_cost_usd: float = Field(default=0, ge=0)


class RiskAssessorInput(BaseModel):
    project_id: str
    project_name: str = Field(default="Project")
    project_budget_usd: float = Field(default=1000000, gt=0)
    project_duration_days: int = Field(default=365, ge=1)
    risks: List[ProjectRisk] = Field(default_factory=list)
    contingency_pct: float = Field(default=10, ge=0, le=50)
    risk_tolerance: str = Field(default="MODERATE")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RiskMitigation(BaseModel):
    risk_id: str
    risk_name: str
    expected_loss_usd: float
    mitigation_action: str
    residual_risk_usd: float
    roi_of_mitigation: float


class RiskAssessorOutput(BaseModel):
    project_id: str
    total_risks: int
    expected_cost_overrun_usd: float
    expected_schedule_delay_days: float
    contingency_required_usd: float
    contingency_adequate: bool
    risk_mitigations: List[RiskMitigation]
    monte_carlo_p50_usd: float
    monte_carlo_p90_usd: float
    overall_risk_rating: str
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class RiskAssessorAgent:
    AGENT_ID = "GL-086B"
    AGENT_NAME = "RISK-ASSESSOR"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"RiskAssessorAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = RiskAssessorInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: RiskAssessorInput) -> RiskAssessorOutput:
        recommendations = []
        mitigations = []

        # Calculate expected values
        total_expected_cost = 0
        total_expected_delay = 0

        for risk in inp.risks:
            prob = risk.probability_pct / 100
            expected_cost = risk.cost_impact_usd * prob
            expected_delay = risk.schedule_impact_days * prob

            total_expected_cost += expected_cost
            total_expected_delay += expected_delay

            # Mitigation analysis
            if risk.mitigation_available and risk.mitigation_cost_usd > 0:
                # Assume mitigation reduces probability by 70%
                residual_prob = prob * 0.3
                residual_cost = risk.cost_impact_usd * residual_prob
                savings = expected_cost - residual_cost - risk.mitigation_cost_usd
                roi = (savings / risk.mitigation_cost_usd * 100) if risk.mitigation_cost_usd > 0 else 0

                action = "MITIGATE" if roi > 0 else "ACCEPT"
            else:
                residual_cost = expected_cost
                roi = 0
                action = "ACCEPT"

            mitigations.append(RiskMitigation(
                risk_id=risk.risk_id,
                risk_name=risk.name,
                expected_loss_usd=round(expected_cost, 2),
                mitigation_action=action,
                residual_risk_usd=round(residual_cost, 2),
                roi_of_mitigation=round(roi, 1)
            ))

        # Sort mitigations by ROI
        mitigations.sort(key=lambda x: -x.roi_of_mitigation)

        # Contingency analysis
        contingency_budget = inp.project_budget_usd * (inp.contingency_pct / 100)
        contingency_adequate = contingency_budget >= total_expected_cost

        # Simplified Monte Carlo approximation
        # P50 = expected value
        # P90 = expected + 1.28 * std_dev (assuming roughly normal)
        variance = sum((r.cost_impact_usd ** 2) * (r.probability_pct/100) * (1 - r.probability_pct/100) for r in inp.risks)
        std_dev = variance ** 0.5
        p50 = inp.project_budget_usd + total_expected_cost
        p90 = p50 + 1.28 * std_dev

        # Overall rating
        risk_ratio = total_expected_cost / inp.project_budget_usd if inp.project_budget_usd > 0 else 0
        if risk_ratio > 0.15:
            rating = "HIGH"
        elif risk_ratio > 0.08:
            rating = "MEDIUM"
        else:
            rating = "LOW"

        # Recommendations
        if not contingency_adequate:
            recommendations.append(f"Contingency ${contingency_budget:,.0f} insufficient - need ${total_expected_cost:,.0f}")

        high_roi = [m for m in mitigations if m.roi_of_mitigation > 100]
        if high_roi:
            recommendations.append(f"{len(high_roi)} mitigations with >100% ROI - implement immediately")

        if total_expected_delay > inp.project_duration_days * 0.1:
            recommendations.append(f"Schedule risk {total_expected_delay:.0f} days - add buffer time")

        tech_risks = [r for r in inp.risks if r.risk_type == ProjectRiskType.TECHNICAL]
        if len(tech_risks) > len(inp.risks) * 0.5:
            recommendations.append("High proportion of technical risks - conduct technical review")

        if rating == "HIGH":
            recommendations.append("Overall project risk HIGH - consider risk transfer or redesign")

        calc_hash = hashlib.sha256(json.dumps({
            "project": inp.project_id,
            "expected_overrun": round(total_expected_cost, 2),
            "rating": rating
        }).encode()).hexdigest()

        return RiskAssessorOutput(
            project_id=inp.project_id,
            total_risks=len(inp.risks),
            expected_cost_overrun_usd=round(total_expected_cost, 2),
            expected_schedule_delay_days=round(total_expected_delay, 1),
            contingency_required_usd=round(total_expected_cost, 2),
            contingency_adequate=contingency_adequate,
            risk_mitigations=mitigations,
            monte_carlo_p50_usd=round(p50, 2),
            monte_carlo_p90_usd=round(p90, 2),
            overall_risk_rating=rating,
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-086B", "name": "RISK-ASSESSOR", "version": "1.0.0",
    "summary": "Project risk assessment and mitigation planning",
    "standards": [{"ref": "ISO 31000"}, {"ref": "PMI Risk Management"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
