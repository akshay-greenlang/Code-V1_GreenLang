# -*- coding: utf-8 -*-
"""
GL-ADAPT-PUB-003: Critical Infrastructure Protection Agent
==========================================================

Assesses climate risks to critical infrastructure and plans
resilience improvements for essential services.

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


class AssetCategory(str, Enum):
    """Categories of critical infrastructure."""
    ENERGY = "energy"
    WATER = "water"
    TRANSPORTATION = "transportation"
    COMMUNICATIONS = "communications"
    HEALTHCARE = "healthcare"
    EMERGENCY_SERVICES = "emergency_services"
    GOVERNMENT = "government"


class ClimateHazard(str, Enum):
    """Climate-related hazards."""
    EXTREME_HEAT = "extreme_heat"
    FLOODING = "flooding"
    DROUGHT = "drought"
    WILDFIRE = "wildfire"
    SEVERE_STORM = "severe_storm"
    SEA_LEVEL_RISE = "sea_level_rise"
    EXTREME_COLD = "extreme_cold"


class RiskRating(str, Enum):
    """Risk ratings."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class CriticalAsset(BaseModel):
    """Critical infrastructure asset."""
    asset_id: str = Field(...)
    name: str = Field(...)
    category: AssetCategory = Field(...)
    location: str = Field(...)
    replacement_value_usd: float = Field(default=0.0, ge=0)
    population_served: int = Field(default=0, ge=0)
    year_built: Optional[int] = Field(None)
    last_major_upgrade: Optional[int] = Field(None)
    hazard_exposures: List[ClimateHazard] = Field(default_factory=list)
    current_risk_rating: RiskRating = Field(default=RiskRating.MODERATE)
    has_backup_power: bool = Field(default=False)
    flood_protection_level_ft: Optional[float] = Field(None)


class ResilienceAssessment(BaseModel):
    """Resilience assessment for an asset."""
    asset_id: str = Field(...)
    assessment_date: datetime = Field(default_factory=DeterministicClock.now)
    overall_risk_score: float = Field(default=0.0, ge=0, le=100)
    hazard_scores: Dict[str, float] = Field(default_factory=dict)
    recommended_improvements: List[str] = Field(default_factory=list)
    estimated_improvement_cost_usd: float = Field(default=0.0, ge=0)
    priority_rank: int = Field(default=0, ge=0)


class InfrastructureProtectionPlan(BaseModel):
    """Infrastructure protection plan."""
    plan_id: str = Field(...)
    jurisdiction_name: str = Field(...)
    plan_name: str = Field(...)
    critical_assets: List[CriticalAsset] = Field(default_factory=list)
    assessments: List[ResilienceAssessment] = Field(default_factory=list)
    total_assets: int = Field(default=0, ge=0)
    high_risk_assets: int = Field(default=0, ge=0)
    total_improvement_budget_usd: float = Field(default=0.0, ge=0)
    created_at: datetime = Field(default_factory=DeterministicClock.now)
    updated_at: datetime = Field(default_factory=DeterministicClock.now)
    provenance_hash: Optional[str] = Field(None)


class InfrastructureProtectionInput(BaseModel):
    """Input for Infrastructure Protection Agent."""
    action: str = Field(...)
    plan_id: Optional[str] = Field(None)
    jurisdiction_name: Optional[str] = Field(None)
    asset: Optional[CriticalAsset] = Field(None)
    asset_id: Optional[str] = Field(None)
    user_id: Optional[str] = Field(None)

    @field_validator('action')
    @classmethod
    def validate_action(cls, v: str) -> str:
        valid = {'create_plan', 'add_asset', 'assess_asset', 'prioritize_investments',
                 'calculate_risk_summary', 'get_plan', 'list_plans'}
        if v not in valid:
            raise ValueError(f"Invalid action: {v}")
        return v


class InfrastructureProtectionOutput(BaseModel):
    """Output from Infrastructure Protection Agent."""
    success: bool = Field(...)
    action: str = Field(...)
    plan: Optional[InfrastructureProtectionPlan] = Field(None)
    plans: Optional[List[InfrastructureProtectionPlan]] = Field(None)
    assessment: Optional[ResilienceAssessment] = Field(None)
    risk_summary: Optional[Dict[str, Any]] = Field(None)
    investment_priorities: Optional[List[Dict[str, Any]]] = Field(None)
    provenance_hash: Optional[str] = Field(None)
    calculation_trace: List[str] = Field(default_factory=list)
    error: Optional[str] = Field(None)
    warnings: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    processing_time_ms: float = Field(default=0.0)


class CriticalInfrastructureProtectionAgent(BaseAgent):
    """
    GL-ADAPT-PUB-003: Critical Infrastructure Protection Agent

    Assesses and improves infrastructure climate resilience.
    """

    AGENT_ID = "GL-ADAPT-PUB-003"
    AGENT_NAME = "Critical Infrastructure Protection Agent"
    VERSION = "1.0.0"

    HAZARD_WEIGHTS = {
        ClimateHazard.FLOODING: 0.25,
        ClimateHazard.EXTREME_HEAT: 0.20,
        ClimateHazard.SEVERE_STORM: 0.20,
        ClimateHazard.WILDFIRE: 0.15,
        ClimateHazard.SEA_LEVEL_RISE: 0.10,
        ClimateHazard.DROUGHT: 0.05,
        ClimateHazard.EXTREME_COLD: 0.05,
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Critical infrastructure climate resilience",
                version=self.VERSION,
            )
        super().__init__(config)
        self._plans: Dict[str, InfrastructureProtectionPlan] = {}
        self.logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        import time
        start_time = time.time()
        try:
            agent_input = InfrastructureProtectionInput(**input_data)
            handlers = {
                'create_plan': self._create_plan,
                'add_asset': self._add_asset,
                'assess_asset': self._assess_asset,
                'prioritize_investments': self._prioritize_investments,
                'calculate_risk_summary': self._calculate_risk_summary,
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

    def _create_plan(self, inp: InfrastructureProtectionInput) -> InfrastructureProtectionOutput:
        if not inp.jurisdiction_name:
            return InfrastructureProtectionOutput(success=False, action='create_plan', error="Jurisdiction name required")
        plan_id = f"CIP-{inp.jurisdiction_name.upper()[:3]}-{DeterministicClock.now().strftime('%Y%m%d')}"
        plan = InfrastructureProtectionPlan(
            plan_id=plan_id,
            jurisdiction_name=inp.jurisdiction_name,
            plan_name=f"{inp.jurisdiction_name} Infrastructure Protection Plan",
        )
        self._plans[plan_id] = plan
        return InfrastructureProtectionOutput(success=True, action='create_plan', plan=plan, calculation_trace=[f"Created plan {plan_id}"])

    def _add_asset(self, inp: InfrastructureProtectionInput) -> InfrastructureProtectionOutput:
        if not inp.plan_id or not inp.asset:
            return InfrastructureProtectionOutput(success=False, action='add_asset', error="Plan ID and asset required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return InfrastructureProtectionOutput(success=False, action='add_asset', error=f"Plan not found")
        plan.critical_assets.append(inp.asset)
        plan.total_assets = len(plan.critical_assets)
        plan.high_risk_assets = len([a for a in plan.critical_assets if a.current_risk_rating in (RiskRating.HIGH, RiskRating.CRITICAL)])
        plan.updated_at = DeterministicClock.now()
        return InfrastructureProtectionOutput(success=True, action='add_asset', plan=plan, calculation_trace=[f"Added asset {inp.asset.asset_id}"])

    def _assess_asset(self, inp: InfrastructureProtectionInput) -> InfrastructureProtectionOutput:
        if not inp.plan_id or not inp.asset_id:
            return InfrastructureProtectionOutput(success=False, action='assess_asset', error="Plan ID and asset ID required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return InfrastructureProtectionOutput(success=False, action='assess_asset', error=f"Plan not found")
        asset = next((a for a in plan.critical_assets if a.asset_id == inp.asset_id), None)
        if not asset:
            return InfrastructureProtectionOutput(success=False, action='assess_asset', error=f"Asset not found")
        # Calculate risk score
        hazard_scores = {}
        for hazard in asset.hazard_exposures:
            weight = self.HAZARD_WEIGHTS.get(hazard, 0.1)
            hazard_scores[hazard.value] = weight * 100
        overall_score = sum(hazard_scores.values())
        recommendations = []
        if not asset.has_backup_power:
            recommendations.append("Install backup power generation")
        if ClimateHazard.FLOODING in asset.hazard_exposures and not asset.flood_protection_level_ft:
            recommendations.append("Implement flood protection measures")
        if asset.year_built and asset.year_built < 2000:
            recommendations.append("Upgrade aging infrastructure components")
        assessment = ResilienceAssessment(
            asset_id=inp.asset_id,
            overall_risk_score=min(100, overall_score),
            hazard_scores=hazard_scores,
            recommended_improvements=recommendations,
            estimated_improvement_cost_usd=len(recommendations) * 500000,
        )
        plan.assessments.append(assessment)
        plan.updated_at = DeterministicClock.now()
        return InfrastructureProtectionOutput(success=True, action='assess_asset', plan=plan, assessment=assessment, calculation_trace=[f"Assessed asset {inp.asset_id}"])

    def _prioritize_investments(self, inp: InfrastructureProtectionInput) -> InfrastructureProtectionOutput:
        if not inp.plan_id:
            return InfrastructureProtectionOutput(success=False, action='prioritize_investments', error="Plan ID required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return InfrastructureProtectionOutput(success=False, action='prioritize_investments', error=f"Plan not found")
        priorities = []
        for assessment in plan.assessments:
            asset = next((a for a in plan.critical_assets if a.asset_id == assessment.asset_id), None)
            if asset:
                score = assessment.overall_risk_score * (asset.population_served / 10000 + 1)
                priorities.append({
                    "asset_id": assessment.asset_id,
                    "asset_name": asset.name,
                    "category": asset.category.value,
                    "risk_score": assessment.overall_risk_score,
                    "priority_score": score,
                    "estimated_cost_usd": assessment.estimated_improvement_cost_usd,
                    "recommendations": assessment.recommended_improvements,
                })
        priorities.sort(key=lambda x: x["priority_score"], reverse=True)
        for i, p in enumerate(priorities):
            p["rank"] = i + 1
        return InfrastructureProtectionOutput(success=True, action='prioritize_investments', plan=plan, investment_priorities=priorities, calculation_trace=[f"Prioritized {len(priorities)} investments"])

    def _calculate_risk_summary(self, inp: InfrastructureProtectionInput) -> InfrastructureProtectionOutput:
        if not inp.plan_id:
            return InfrastructureProtectionOutput(success=False, action='calculate_risk_summary', error="Plan ID required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return InfrastructureProtectionOutput(success=False, action='calculate_risk_summary', error=f"Plan not found")
        by_category = {}
        for asset in plan.critical_assets:
            cat = asset.category.value
            if cat not in by_category:
                by_category[cat] = {"count": 0, "high_risk": 0}
            by_category[cat]["count"] += 1
            if asset.current_risk_rating in (RiskRating.HIGH, RiskRating.CRITICAL):
                by_category[cat]["high_risk"] += 1
        summary = {
            "total_assets": plan.total_assets,
            "high_risk_assets": plan.high_risk_assets,
            "assessments_completed": len(plan.assessments),
            "by_category": by_category,
            "total_improvement_cost_usd": sum(a.estimated_improvement_cost_usd for a in plan.assessments),
        }
        return InfrastructureProtectionOutput(success=True, action='calculate_risk_summary', plan=plan, risk_summary=summary, calculation_trace=["Risk summary calculated"])

    def _get_plan(self, inp: InfrastructureProtectionInput) -> InfrastructureProtectionOutput:
        if not inp.plan_id:
            return InfrastructureProtectionOutput(success=False, action='get_plan', error="Plan ID required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return InfrastructureProtectionOutput(success=False, action='get_plan', error=f"Plan not found")
        return InfrastructureProtectionOutput(success=True, action='get_plan', plan=plan)

    def _list_plans(self, inp: InfrastructureProtectionInput) -> InfrastructureProtectionOutput:
        return InfrastructureProtectionOutput(success=True, action='list_plans', plans=list(self._plans.values()))

    def _hash_output(self, output: InfrastructureProtectionOutput) -> str:
        content = {"action": output.action, "success": output.success, "timestamp": output.timestamp.isoformat()}
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
