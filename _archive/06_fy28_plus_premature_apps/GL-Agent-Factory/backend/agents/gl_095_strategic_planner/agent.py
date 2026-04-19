"""GL-095: Strategic Planner Agent (STRATEGIC-PLANNER).

Develops strategic plans for energy programs.

Standards: Strategic Planning Best Practices
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StrategicPriority(str, Enum):
    DECARBONIZATION = "DECARBONIZATION"
    COST_REDUCTION = "COST_REDUCTION"
    RELIABILITY = "RELIABILITY"
    RESILIENCE = "RESILIENCE"
    COMPLIANCE = "COMPLIANCE"


class StrategicGoal(BaseModel):
    goal_id: str
    description: str
    priority: StrategicPriority
    target_year: int = Field(ge=2024)
    target_value: float
    current_value: float
    unit: str


class StrategicPlannerInput(BaseModel):
    organization_id: str
    planning_horizon_years: int = Field(default=5, ge=1, le=30)
    goals: List[StrategicGoal] = Field(default_factory=list)
    annual_budget_usd: float = Field(default=1000000, ge=0)
    risk_tolerance: str = Field(default="MODERATE")
    external_factors: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StrategicInitiative(BaseModel):
    initiative_id: str
    name: str
    aligned_goals: List[str]
    estimated_investment_usd: float
    expected_impact: str
    implementation_year: int
    priority_rank: int


class StrategicPlannerOutput(BaseModel):
    organization_id: str
    planning_horizon: str
    strategic_initiatives: List[StrategicInitiative]
    total_investment_required_usd: float
    budget_gap_usd: float
    goal_achievement_outlook: str
    key_risks: List[str]
    success_factors: List[str]
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class StrategicPlannerAgent:
    AGENT_ID = "GL-095"
    AGENT_NAME = "STRATEGIC-PLANNER"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"StrategicPlannerAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = StrategicPlannerInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _generate_initiatives(self, goals: List[StrategicGoal], budget: float) -> List[StrategicInitiative]:
        """Generate strategic initiatives based on goals."""
        initiatives = []
        current_year = datetime.utcnow().year

        for i, goal in enumerate(goals):
            gap = abs(goal.target_value - goal.current_value)
            years_to_target = goal.target_year - current_year

            if gap > 0 and years_to_target > 0:
                # Estimate investment needed
                investment = budget * 0.2 * (gap / max(goal.target_value, 1))

                # Determine impact
                progress_pct = (goal.current_value / goal.target_value * 100) if goal.target_value > 0 else 0
                if progress_pct < 30:
                    impact = "HIGH"
                elif progress_pct < 60:
                    impact = "MEDIUM"
                else:
                    impact = "LOW"

                initiatives.append(StrategicInitiative(
                    initiative_id=f"INIT-{goal.goal_id}",
                    name=f"Initiative for {goal.description}",
                    aligned_goals=[goal.goal_id],
                    estimated_investment_usd=round(investment, 2),
                    expected_impact=impact,
                    implementation_year=current_year + 1,
                    priority_rank=i + 1
                ))

        return initiatives

    def _process(self, inp: StrategicPlannerInput) -> StrategicPlannerOutput:
        recommendations = []
        risks = []
        success_factors = []

        # Generate initiatives
        initiatives = self._generate_initiatives(inp.goals, inp.annual_budget_usd)

        # Calculate total investment
        total_investment = sum(i.estimated_investment_usd for i in initiatives)
        total_budget = inp.annual_budget_usd * inp.planning_horizon_years
        budget_gap = total_investment - total_budget

        # Goal achievement outlook
        achievable_goals = sum(1 for g in inp.goals if g.current_value >= g.target_value * 0.5)
        if achievable_goals == len(inp.goals):
            outlook = "STRONG"
        elif achievable_goals >= len(inp.goals) * 0.7:
            outlook = "POSITIVE"
        elif achievable_goals >= len(inp.goals) * 0.5:
            outlook = "MODERATE"
        else:
            outlook = "CHALLENGING"

        # Risks based on external factors and goals
        if budget_gap > 0:
            risks.append(f"Funding gap of ${budget_gap:,.0f} may delay initiatives")

        decarb_goals = [g for g in inp.goals if g.priority == StrategicPriority.DECARBONIZATION]
        if decarb_goals:
            risks.append("Regulatory changes may accelerate decarbonization requirements")

        if "supply_chain" in str(inp.external_factors).lower():
            risks.append("Supply chain disruptions may impact equipment procurement")

        if inp.risk_tolerance == "LOW":
            risks.append("Conservative risk tolerance may limit innovation opportunities")

        # Success factors
        success_factors.append("Executive sponsorship and cross-functional alignment")
        success_factors.append("Adequate funding and resource allocation")
        success_factors.append("Clear metrics and accountability")
        if decarb_goals:
            success_factors.append("Technology partnerships for decarbonization")

        # Recommendations
        if budget_gap > 0:
            recommendations.append(f"Address ${budget_gap:,.0f} funding gap through phased implementation")
        if outlook == "CHALLENGING":
            recommendations.append("Reprioritize goals to focus on achievable targets")

        high_impact = [i for i in initiatives if i.expected_impact == "HIGH"]
        if high_impact:
            recommendations.append(f"Prioritize {len(high_impact)} high-impact initiatives")

        recommendations.append("Establish quarterly review cadence for plan monitoring")

        planning_horizon = f"{inp.planning_horizon_years} years ({datetime.utcnow().year}-{datetime.utcnow().year + inp.planning_horizon_years})"

        calc_hash = hashlib.sha256(json.dumps({
            "organization": inp.organization_id,
            "horizon": inp.planning_horizon_years,
            "outlook": outlook
        }).encode()).hexdigest()

        return StrategicPlannerOutput(
            organization_id=inp.organization_id,
            planning_horizon=planning_horizon,
            strategic_initiatives=initiatives,
            total_investment_required_usd=round(total_investment, 2),
            budget_gap_usd=round(max(0, budget_gap), 2),
            goal_achievement_outlook=outlook,
            key_risks=risks,
            success_factors=success_factors,
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-095", "name": "STRATEGIC-PLANNER", "version": "1.0.0",
    "summary": "Strategic planning for energy programs",
    "standards": [{"ref": "Strategic Planning Best Practices"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
