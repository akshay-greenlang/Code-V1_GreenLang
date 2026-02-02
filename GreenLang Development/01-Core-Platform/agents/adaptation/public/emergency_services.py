# -*- coding: utf-8 -*-
"""
GL-ADAPT-PUB-005: Emergency Services Adaptation Agent
=====================================================

Adapts emergency response capabilities to climate change impacts.

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


class EmergencyServiceType(str, Enum):
    """Types of emergency services."""
    FIRE = "fire"
    EMS = "ems"
    POLICE = "police"
    SEARCH_RESCUE = "search_rescue"
    HAZMAT = "hazmat"
    DISPATCH = "dispatch"


class EmergencyResource(BaseModel):
    """Emergency service resource."""
    resource_id: str = Field(...)
    name: str = Field(...)
    service_type: EmergencyServiceType = Field(...)
    location: str = Field(...)
    personnel_count: int = Field(default=0, ge=0)
    vehicle_count: int = Field(default=0, ge=0)
    has_climate_controlled_vehicles: bool = Field(default=False)
    has_flood_capable_vehicles: bool = Field(default=False)
    backup_power_hours: float = Field(default=0.0, ge=0)


class ResponseCapability(BaseModel):
    """Response capability assessment."""
    capability_id: str = Field(...)
    emergency_type: str = Field(...)
    current_capacity: str = Field(default="adequate")  # inadequate, adequate, surplus
    response_time_minutes: float = Field(default=0.0, ge=0)
    coverage_gap_areas: List[str] = Field(default_factory=list)
    improvement_recommendations: List[str] = Field(default_factory=list)


class EmergencyServicesPlan(BaseModel):
    """Emergency services adaptation plan."""
    plan_id: str = Field(...)
    jurisdiction_name: str = Field(...)
    plan_name: str = Field(...)
    resources: List[EmergencyResource] = Field(default_factory=list)
    capabilities: List[ResponseCapability] = Field(default_factory=list)
    total_personnel: int = Field(default=0, ge=0)
    total_vehicles: int = Field(default=0, ge=0)
    adaptation_budget_usd: float = Field(default=0.0, ge=0)
    created_at: datetime = Field(default_factory=DeterministicClock.now)
    updated_at: datetime = Field(default_factory=DeterministicClock.now)
    provenance_hash: Optional[str] = Field(None)


class EmergencyServicesInput(BaseModel):
    """Input for Emergency Services Agent."""
    action: str = Field(...)
    plan_id: Optional[str] = Field(None)
    jurisdiction_name: Optional[str] = Field(None)
    resource: Optional[EmergencyResource] = Field(None)
    capability: Optional[ResponseCapability] = Field(None)
    user_id: Optional[str] = Field(None)

    @field_validator('action')
    @classmethod
    def validate_action(cls, v: str) -> str:
        valid = {'create_plan', 'add_resource', 'assess_capability', 'analyze_gaps',
                 'generate_recommendations', 'get_plan', 'list_plans'}
        if v not in valid:
            raise ValueError(f"Invalid action: {v}")
        return v


class EmergencyServicesOutput(BaseModel):
    """Output from Emergency Services Agent."""
    success: bool = Field(...)
    action: str = Field(...)
    plan: Optional[EmergencyServicesPlan] = Field(None)
    plans: Optional[List[EmergencyServicesPlan]] = Field(None)
    gap_analysis: Optional[Dict[str, Any]] = Field(None)
    recommendations: Optional[List[Dict[str, Any]]] = Field(None)
    provenance_hash: Optional[str] = Field(None)
    calculation_trace: List[str] = Field(default_factory=list)
    error: Optional[str] = Field(None)
    warnings: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    processing_time_ms: float = Field(default=0.0)


class EmergencyServicesAdaptationAgent(BaseAgent):
    """
    GL-ADAPT-PUB-005: Emergency Services Adaptation Agent

    Adapts emergency services for climate resilience.
    """

    AGENT_ID = "GL-ADAPT-PUB-005"
    AGENT_NAME = "Emergency Services Adaptation Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Emergency services climate adaptation",
                version=self.VERSION,
            )
        super().__init__(config)
        self._plans: Dict[str, EmergencyServicesPlan] = {}
        self.logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        import time
        start_time = time.time()
        try:
            agent_input = EmergencyServicesInput(**input_data)
            handlers = {
                'create_plan': self._create_plan,
                'add_resource': self._add_resource,
                'assess_capability': self._assess_capability,
                'analyze_gaps': self._analyze_gaps,
                'generate_recommendations': self._generate_recommendations,
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

    def _create_plan(self, inp: EmergencyServicesInput) -> EmergencyServicesOutput:
        if not inp.jurisdiction_name:
            return EmergencyServicesOutput(success=False, action='create_plan', error="Jurisdiction name required")
        plan_id = f"ESA-{inp.jurisdiction_name.upper()[:3]}-{DeterministicClock.now().strftime('%Y%m%d')}"
        plan = EmergencyServicesPlan(
            plan_id=plan_id,
            jurisdiction_name=inp.jurisdiction_name,
            plan_name=f"{inp.jurisdiction_name} Emergency Services Adaptation Plan",
        )
        self._plans[plan_id] = plan
        return EmergencyServicesOutput(success=True, action='create_plan', plan=plan, calculation_trace=[f"Created plan {plan_id}"])

    def _add_resource(self, inp: EmergencyServicesInput) -> EmergencyServicesOutput:
        if not inp.plan_id or not inp.resource:
            return EmergencyServicesOutput(success=False, action='add_resource', error="Plan ID and resource required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return EmergencyServicesOutput(success=False, action='add_resource', error="Plan not found")
        plan.resources.append(inp.resource)
        plan.total_personnel = sum(r.personnel_count for r in plan.resources)
        plan.total_vehicles = sum(r.vehicle_count for r in plan.resources)
        plan.updated_at = DeterministicClock.now()
        return EmergencyServicesOutput(success=True, action='add_resource', plan=plan, calculation_trace=[f"Added resource {inp.resource.resource_id}"])

    def _assess_capability(self, inp: EmergencyServicesInput) -> EmergencyServicesOutput:
        if not inp.plan_id or not inp.capability:
            return EmergencyServicesOutput(success=False, action='assess_capability', error="Plan ID and capability required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return EmergencyServicesOutput(success=False, action='assess_capability', error="Plan not found")
        plan.capabilities.append(inp.capability)
        plan.updated_at = DeterministicClock.now()
        return EmergencyServicesOutput(success=True, action='assess_capability', plan=plan, calculation_trace=[f"Added capability {inp.capability.capability_id}"])

    def _analyze_gaps(self, inp: EmergencyServicesInput) -> EmergencyServicesOutput:
        if not inp.plan_id:
            return EmergencyServicesOutput(success=False, action='analyze_gaps', error="Plan ID required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return EmergencyServicesOutput(success=False, action='analyze_gaps', error="Plan not found")
        inadequate = [c for c in plan.capabilities if c.current_capacity == "inadequate"]
        no_flood_vehicles = sum(1 for r in plan.resources if not r.has_flood_capable_vehicles)
        low_backup_power = sum(1 for r in plan.resources if r.backup_power_hours < 24)
        gap_analysis = {
            "inadequate_capabilities": len(inadequate),
            "resources_without_flood_vehicles": no_flood_vehicles,
            "resources_with_low_backup_power": low_backup_power,
            "coverage_gaps": list(set(g for c in plan.capabilities for g in c.coverage_gap_areas)),
        }
        return EmergencyServicesOutput(success=True, action='analyze_gaps', plan=plan, gap_analysis=gap_analysis, calculation_trace=["Gap analysis completed"])

    def _generate_recommendations(self, inp: EmergencyServicesInput) -> EmergencyServicesOutput:
        if not inp.plan_id:
            return EmergencyServicesOutput(success=False, action='generate_recommendations', error="Plan ID required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return EmergencyServicesOutput(success=False, action='generate_recommendations', error="Plan not found")
        recommendations = []
        no_flood = [r for r in plan.resources if not r.has_flood_capable_vehicles]
        if no_flood:
            recommendations.append({"priority": "high", "category": "equipment", "recommendation": f"Acquire flood-capable vehicles for {len(no_flood)} stations", "estimated_cost_usd": len(no_flood) * 150000})
        low_power = [r for r in plan.resources if r.backup_power_hours < 24]
        if low_power:
            recommendations.append({"priority": "high", "category": "infrastructure", "recommendation": f"Upgrade backup power for {len(low_power)} stations", "estimated_cost_usd": len(low_power) * 50000})
        for cap in plan.capabilities:
            if cap.current_capacity == "inadequate":
                recommendations.extend([{"priority": "medium", "category": "capability", "recommendation": rec, "estimated_cost_usd": 100000} for rec in cap.improvement_recommendations])
        return EmergencyServicesOutput(success=True, action='generate_recommendations', plan=plan, recommendations=recommendations, calculation_trace=[f"Generated {len(recommendations)} recommendations"])

    def _get_plan(self, inp: EmergencyServicesInput) -> EmergencyServicesOutput:
        if not inp.plan_id:
            return EmergencyServicesOutput(success=False, action='get_plan', error="Plan ID required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return EmergencyServicesOutput(success=False, action='get_plan', error="Plan not found")
        return EmergencyServicesOutput(success=True, action='get_plan', plan=plan)

    def _list_plans(self, inp: EmergencyServicesInput) -> EmergencyServicesOutput:
        return EmergencyServicesOutput(success=True, action='list_plans', plans=list(self._plans.values()))

    def _hash_output(self, output: EmergencyServicesOutput) -> str:
        content = {"action": output.action, "success": output.success, "timestamp": output.timestamp.isoformat()}
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
