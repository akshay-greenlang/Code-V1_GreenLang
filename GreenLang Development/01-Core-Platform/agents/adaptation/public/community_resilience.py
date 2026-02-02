# -*- coding: utf-8 -*-
"""
GL-ADAPT-PUB-006: Community Resilience Planner Agent
====================================================

Plans community-level climate resilience including social networks,
local assets, and participatory adaptation strategies.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class AssetType(str, Enum):
    """Types of community assets."""
    SOCIAL = "social"  # Community organizations
    PHYSICAL = "physical"  # Buildings, parks
    ECONOMIC = "economic"  # Local businesses
    NATURAL = "natural"  # Green spaces, water
    INSTITUTIONAL = "institutional"  # Government services


class ResilienceIndicator(BaseModel):
    """Community resilience indicator."""
    indicator_id: str = Field(...)
    name: str = Field(...)
    category: str = Field(...)  # social, economic, environmental, institutional
    current_value: float = Field(default=0.0)
    target_value: float = Field(default=0.0)
    unit: str = Field(default="score")
    data_source: Optional[str] = Field(None)


class CommunityAsset(BaseModel):
    """Community asset."""
    asset_id: str = Field(...)
    name: str = Field(...)
    asset_type: AssetType = Field(...)
    description: Optional[str] = Field(None)
    capacity: int = Field(default=0, ge=0)
    climate_vulnerable: bool = Field(default=False)
    community_importance: str = Field(default="medium")  # low, medium, high


class AdaptationProject(BaseModel):
    """Community adaptation project."""
    project_id: str = Field(...)
    name: str = Field(...)
    description: str = Field(...)
    target_indicators: List[str] = Field(default_factory=list)
    estimated_cost_usd: float = Field(default=0.0, ge=0)
    community_participation_required: bool = Field(default=True)
    implementation_months: int = Field(default=12, ge=1)
    status: str = Field(default="proposed")


class CommunityResiliencePlan(BaseModel):
    """Community resilience plan."""
    plan_id: str = Field(...)
    community_name: str = Field(...)
    plan_name: str = Field(...)
    population: int = Field(default=0, ge=0)
    assets: List[CommunityAsset] = Field(default_factory=list)
    indicators: List[ResilienceIndicator] = Field(default_factory=list)
    projects: List[AdaptationProject] = Field(default_factory=list)
    overall_resilience_score: float = Field(default=0.0, ge=0, le=100)
    total_project_budget_usd: float = Field(default=0.0, ge=0)
    created_at: datetime = Field(default_factory=DeterministicClock.now)
    updated_at: datetime = Field(default_factory=DeterministicClock.now)
    provenance_hash: Optional[str] = Field(None)


class CommunityResilienceInput(BaseModel):
    """Input for Community Resilience Agent."""
    action: str = Field(...)
    plan_id: Optional[str] = Field(None)
    community_name: Optional[str] = Field(None)
    population: Optional[int] = Field(None)
    asset: Optional[CommunityAsset] = Field(None)
    indicator: Optional[ResilienceIndicator] = Field(None)
    project: Optional[AdaptationProject] = Field(None)
    user_id: Optional[str] = Field(None)

    @field_validator('action')
    @classmethod
    def validate_action(cls, v: str) -> str:
        valid = {'create_plan', 'add_asset', 'add_indicator', 'add_project',
                 'calculate_resilience_score', 'identify_gaps', 'get_plan', 'list_plans'}
        if v not in valid:
            raise ValueError(f"Invalid action: {v}")
        return v


class CommunityResilienceOutput(BaseModel):
    """Output from Community Resilience Agent."""
    success: bool = Field(...)
    action: str = Field(...)
    plan: Optional[CommunityResiliencePlan] = Field(None)
    plans: Optional[List[CommunityResiliencePlan]] = Field(None)
    resilience_assessment: Optional[Dict[str, Any]] = Field(None)
    gap_analysis: Optional[Dict[str, Any]] = Field(None)
    provenance_hash: Optional[str] = Field(None)
    calculation_trace: List[str] = Field(default_factory=list)
    error: Optional[str] = Field(None)
    warnings: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    processing_time_ms: float = Field(default=0.0)


class CommunityResiliencePlannerAgent(BaseAgent):
    """
    GL-ADAPT-PUB-006: Community Resilience Planner Agent

    Plans community-level climate resilience.
    """

    AGENT_ID = "GL-ADAPT-PUB-006"
    AGENT_NAME = "Community Resilience Planner Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Community resilience planning",
                version=self.VERSION,
            )
        super().__init__(config)
        self._plans: Dict[str, CommunityResiliencePlan] = {}
        self.logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        import time
        start_time = time.time()
        try:
            agent_input = CommunityResilienceInput(**input_data)
            handlers = {
                'create_plan': self._create_plan,
                'add_asset': self._add_asset,
                'add_indicator': self._add_indicator,
                'add_project': self._add_project,
                'calculate_resilience_score': self._calculate_score,
                'identify_gaps': self._identify_gaps,
                'get_plan': self._get_plan,
                'list_plans': self._list_plans,
            }
            output = handlers[agent_input.action](agent_input)
            output.processing_time_ms = (time.time() - start_time) * 1000
            output.provenance_hash = self._hash_output(output)
            return AgentResult(success=output.success, data=output.model_dump(), error=output.error)
        except Exception as e:
            self.logger.error(f"Error: {e}", exc_info=True)
            return AgentResult(success=False, error=str(e))

    def _create_plan(self, inp: CommunityResilienceInput) -> CommunityResilienceOutput:
        if not inp.community_name:
            return CommunityResilienceOutput(success=False, action='create_plan', error="Community name required")
        plan_id = f"CRP-{inp.community_name.upper()[:3]}-{DeterministicClock.now().strftime('%Y%m%d')}"
        plan = CommunityResiliencePlan(
            plan_id=plan_id,
            community_name=inp.community_name,
            plan_name=f"{inp.community_name} Community Resilience Plan",
            population=inp.population or 0,
        )
        self._plans[plan_id] = plan
        return CommunityResilienceOutput(success=True, action='create_plan', plan=plan, calculation_trace=[f"Created plan {plan_id}"])

    def _add_asset(self, inp: CommunityResilienceInput) -> CommunityResilienceOutput:
        if not inp.plan_id or not inp.asset:
            return CommunityResilienceOutput(success=False, action='add_asset', error="Plan ID and asset required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return CommunityResilienceOutput(success=False, action='add_asset', error="Plan not found")
        plan.assets.append(inp.asset)
        plan.updated_at = DeterministicClock.now()
        return CommunityResilienceOutput(success=True, action='add_asset', plan=plan, calculation_trace=[f"Added asset {inp.asset.asset_id}"])

    def _add_indicator(self, inp: CommunityResilienceInput) -> CommunityResilienceOutput:
        if not inp.plan_id or not inp.indicator:
            return CommunityResilienceOutput(success=False, action='add_indicator', error="Plan ID and indicator required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return CommunityResilienceOutput(success=False, action='add_indicator', error="Plan not found")
        plan.indicators.append(inp.indicator)
        plan.updated_at = DeterministicClock.now()
        return CommunityResilienceOutput(success=True, action='add_indicator', plan=plan, calculation_trace=[f"Added indicator {inp.indicator.indicator_id}"])

    def _add_project(self, inp: CommunityResilienceInput) -> CommunityResilienceOutput:
        if not inp.plan_id or not inp.project:
            return CommunityResilienceOutput(success=False, action='add_project', error="Plan ID and project required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return CommunityResilienceOutput(success=False, action='add_project', error="Plan not found")
        plan.projects.append(inp.project)
        plan.total_project_budget_usd = sum(p.estimated_cost_usd for p in plan.projects)
        plan.updated_at = DeterministicClock.now()
        return CommunityResilienceOutput(success=True, action='add_project', plan=plan, calculation_trace=[f"Added project {inp.project.project_id}"])

    def _calculate_score(self, inp: CommunityResilienceInput) -> CommunityResilienceOutput:
        if not inp.plan_id:
            return CommunityResilienceOutput(success=False, action='calculate_resilience_score', error="Plan ID required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return CommunityResilienceOutput(success=False, action='calculate_resilience_score', error="Plan not found")
        # Calculate based on indicators
        if plan.indicators:
            achievement_rates = []
            for ind in plan.indicators:
                if ind.target_value > 0:
                    rate = min(100, (ind.current_value / ind.target_value) * 100)
                    achievement_rates.append(rate)
            plan.overall_resilience_score = sum(achievement_rates) / len(achievement_rates) if achievement_rates else 0
        # Factor in assets
        asset_bonus = min(20, len([a for a in plan.assets if a.community_importance == "high"]) * 2)
        plan.overall_resilience_score = min(100, plan.overall_resilience_score + asset_bonus)
        plan.updated_at = DeterministicClock.now()
        assessment = {
            "overall_score": plan.overall_resilience_score,
            "indicators_count": len(plan.indicators),
            "assets_count": len(plan.assets),
            "projects_count": len(plan.projects),
            "by_category": {},
        }
        return CommunityResilienceOutput(success=True, action='calculate_resilience_score', plan=plan, resilience_assessment=assessment, calculation_trace=[f"Resilience score: {plan.overall_resilience_score:.1f}"])

    def _identify_gaps(self, inp: CommunityResilienceInput) -> CommunityResilienceOutput:
        if not inp.plan_id:
            return CommunityResilienceOutput(success=False, action='identify_gaps', error="Plan ID required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return CommunityResilienceOutput(success=False, action='identify_gaps', error="Plan not found")
        underperforming = [i for i in plan.indicators if i.target_value > 0 and (i.current_value / i.target_value) < 0.7]
        vulnerable_assets = [a for a in plan.assets if a.climate_vulnerable]
        uncovered_categories = set(AssetType) - set(a.asset_type for a in plan.assets)
        gap_analysis = {
            "underperforming_indicators": [i.name for i in underperforming],
            "vulnerable_assets": [a.name for a in vulnerable_assets],
            "missing_asset_types": [t.value for t in uncovered_categories],
            "total_gaps": len(underperforming) + len(vulnerable_assets) + len(uncovered_categories),
        }
        return CommunityResilienceOutput(success=True, action='identify_gaps', plan=plan, gap_analysis=gap_analysis, calculation_trace=["Gap analysis completed"])

    def _get_plan(self, inp: CommunityResilienceInput) -> CommunityResilienceOutput:
        if not inp.plan_id:
            return CommunityResilienceOutput(success=False, action='get_plan', error="Plan ID required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return CommunityResilienceOutput(success=False, action='get_plan', error="Plan not found")
        return CommunityResilienceOutput(success=True, action='get_plan', plan=plan)

    def _list_plans(self, inp: CommunityResilienceInput) -> CommunityResilienceOutput:
        return CommunityResilienceOutput(success=True, action='list_plans', plans=list(self._plans.values()))

    def _hash_output(self, output: CommunityResilienceOutput) -> str:
        content = {"action": output.action, "success": output.success, "timestamp": output.timestamp.isoformat()}
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
