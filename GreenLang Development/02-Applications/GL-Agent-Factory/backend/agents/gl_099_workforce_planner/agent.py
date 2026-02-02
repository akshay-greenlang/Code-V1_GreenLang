"""GL-099: Workforce Planner Agent (WORKFORCE-PLANNER).

Plans workforce for energy programs and transitions.

Standards: HR Analytics, Workforce Planning
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SkillCategory(str, Enum):
    TECHNICAL = "TECHNICAL"
    DIGITAL = "DIGITAL"
    LEADERSHIP = "LEADERSHIP"
    SAFETY = "SAFETY"
    SUSTAINABILITY = "SUSTAINABILITY"


class SkillGap(BaseModel):
    skill_name: str
    category: SkillCategory
    current_headcount: int = Field(ge=0)
    required_headcount: int = Field(ge=0)
    criticality: str = Field(default="MEDIUM")


class WorkforcePlannerInput(BaseModel):
    organization_id: str
    planning_horizon_years: int = Field(default=3, ge=1, le=10)
    skill_gaps: List[SkillGap] = Field(default_factory=list)
    current_headcount: int = Field(default=500, ge=0)
    attrition_rate_pct: float = Field(default=10, ge=0, le=50)
    growth_target_pct: float = Field(default=5, ge=-20, le=50)
    training_budget_usd: float = Field(default=100000, ge=0)
    hiring_cost_per_fte_usd: float = Field(default=10000, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SkillAction(BaseModel):
    skill_name: str
    gap_count: int
    hire_count: int
    train_count: int
    estimated_cost_usd: float
    timeline_months: int


class WorkforcePlannerOutput(BaseModel):
    organization_id: str
    planning_horizon: str
    total_gap: int
    total_hire_needed: int
    total_train_needed: int
    skill_actions: List[SkillAction]
    total_investment_usd: float
    budget_gap_usd: float
    critical_gaps: List[str]
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class WorkforcePlannerAgent:
    AGENT_ID = "GL-099B"
    AGENT_NAME = "WORKFORCE-PLANNER"
    VERSION = "1.0.0"

    TRAINING_COST_PER_PERSON = 5000

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"WorkforcePlannerAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = WorkforcePlannerInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: WorkforcePlannerInput) -> WorkforcePlannerOutput:
        recommendations = []
        actions = []
        critical = []

        total_gap = 0
        total_hire = 0
        total_train = 0
        total_cost = 0

        for gap in inp.skill_gaps:
            deficit = gap.required_headcount - gap.current_headcount
            if deficit <= 0:
                continue

            total_gap += deficit

            # Strategy: 70% train, 30% hire for most skills
            if gap.criticality == "HIGH":
                hire_pct = 0.5
                train_pct = 0.5
                critical.append(gap.skill_name)
            else:
                hire_pct = 0.3
                train_pct = 0.7

            hire = int(deficit * hire_pct)
            train = deficit - hire

            # Costs
            hire_cost = hire * inp.hiring_cost_per_fte_usd
            train_cost = train * self.TRAINING_COST_PER_PERSON
            action_cost = hire_cost + train_cost

            # Timeline (months)
            timeline = 6 if gap.criticality == "HIGH" else 12

            actions.append(SkillAction(
                skill_name=gap.skill_name,
                gap_count=deficit,
                hire_count=hire,
                train_count=train,
                estimated_cost_usd=round(action_cost, 2),
                timeline_months=timeline
            ))

            total_hire += hire
            total_train += train
            total_cost += action_cost

        # Attrition adjustment
        annual_attrition = int(inp.current_headcount * inp.attrition_rate_pct / 100)
        total_hire += annual_attrition * inp.planning_horizon_years
        total_cost += annual_attrition * inp.planning_horizon_years * inp.hiring_cost_per_fte_usd

        # Budget gap
        budget_gap = total_cost - inp.training_budget_usd * inp.planning_horizon_years

        # Recommendations
        if critical:
            recommendations.append(f"Critical skill gaps: {', '.join(critical)} - prioritize")
        if budget_gap > 0:
            recommendations.append(f"Budget gap ${budget_gap:,.0f} - seek additional funding")
        if inp.attrition_rate_pct > 15:
            recommendations.append(f"High attrition ({inp.attrition_rate_pct}%) - improve retention")
        if total_train > total_hire:
            recommendations.append("Training-heavy strategy - ensure learning infrastructure")
        if total_gap == 0:
            recommendations.append("No skill gaps identified - validate workforce model")

        horizon = f"{inp.planning_horizon_years} years"

        calc_hash = hashlib.sha256(json.dumps({
            "organization": inp.organization_id,
            "gap": total_gap,
            "cost": round(total_cost, 2)
        }).encode()).hexdigest()

        return WorkforcePlannerOutput(
            organization_id=inp.organization_id,
            planning_horizon=horizon,
            total_gap=total_gap,
            total_hire_needed=total_hire,
            total_train_needed=total_train,
            skill_actions=actions,
            total_investment_usd=round(total_cost, 2),
            budget_gap_usd=round(max(0, budget_gap), 2),
            critical_gaps=critical,
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-099B", "name": "WORKFORCE-PLANNER", "version": "1.0.0",
    "summary": "Workforce planning for energy transitions",
    "standards": [{"ref": "HR Analytics"}, {"ref": "Workforce Planning"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
