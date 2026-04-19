# -*- coding: utf-8 -*-
"""
GL-ADAPT-PUB-002: Flood Response Planning Agent
================================================

Plans flood emergency response including evacuation routes, shelters,
and resource deployment. Supports flood risk assessment and warning systems.

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


class FloodRiskLevel(str, Enum):
    """Flood risk levels."""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class FloodType(str, Enum):
    """Types of flooding."""
    RIVERINE = "riverine"
    COASTAL = "coastal"
    FLASH = "flash"
    URBAN = "urban"
    GROUNDWATER = "groundwater"


class FloodZone(BaseModel):
    """Flood zone definition."""
    zone_id: str = Field(...)
    zone_name: str = Field(...)
    flood_type: FloodType = Field(...)
    risk_level: FloodRiskLevel = Field(...)
    population: int = Field(default=0, ge=0)
    critical_facilities_count: int = Field(default=0, ge=0)
    base_flood_elevation_ft: Optional[float] = Field(None)
    return_period_years: Optional[int] = Field(None)


class EvacuationRoute(BaseModel):
    """Evacuation route definition."""
    route_id: str = Field(...)
    route_name: str = Field(...)
    origin_zone: str = Field(...)
    destination: str = Field(...)
    distance_miles: float = Field(..., gt=0)
    capacity_vehicles_per_hour: int = Field(..., ge=0)
    flood_prone_segments: int = Field(default=0, ge=0)
    alternate_routes: List[str] = Field(default_factory=list)


class FloodShelter(BaseModel):
    """Emergency flood shelter."""
    shelter_id: str = Field(...)
    name: str = Field(...)
    address: str = Field(...)
    capacity: int = Field(..., ge=0)
    elevation_ft: float = Field(...)
    generator_available: bool = Field(default=False)


class FloodResponsePlan(BaseModel):
    """Flood response plan."""
    plan_id: str = Field(...)
    municipality_name: str = Field(...)
    plan_name: str = Field(...)
    flood_zones: List[FloodZone] = Field(default_factory=list)
    evacuation_routes: List[EvacuationRoute] = Field(default_factory=list)
    shelters: List[FloodShelter] = Field(default_factory=list)
    total_at_risk_population: int = Field(default=0, ge=0)
    total_shelter_capacity: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=DeterministicClock.now)
    updated_at: datetime = Field(default_factory=DeterministicClock.now)
    provenance_hash: Optional[str] = Field(None)


class FloodResponseInput(BaseModel):
    """Input for Flood Response Agent."""
    action: str = Field(...)
    plan_id: Optional[str] = Field(None)
    municipality_name: Optional[str] = Field(None)
    flood_zone: Optional[FloodZone] = Field(None)
    evacuation_route: Optional[EvacuationRoute] = Field(None)
    shelter: Optional[FloodShelter] = Field(None)
    forecast_flood_level_ft: Optional[float] = Field(None)
    user_id: Optional[str] = Field(None)

    @field_validator('action')
    @classmethod
    def validate_action(cls, v: str) -> str:
        valid = {'create_plan', 'add_flood_zone', 'add_evacuation_route', 'add_shelter',
                 'assess_risk', 'calculate_evacuation_time', 'get_plan', 'list_plans'}
        if v not in valid:
            raise ValueError(f"Invalid action: {v}")
        return v


class FloodResponseOutput(BaseModel):
    """Output from Flood Response Agent."""
    success: bool = Field(...)
    action: str = Field(...)
    plan: Optional[FloodResponsePlan] = Field(None)
    plans: Optional[List[FloodResponsePlan]] = Field(None)
    risk_assessment: Optional[Dict[str, Any]] = Field(None)
    evacuation_analysis: Optional[Dict[str, Any]] = Field(None)
    provenance_hash: Optional[str] = Field(None)
    calculation_trace: List[str] = Field(default_factory=list)
    error: Optional[str] = Field(None)
    warnings: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    processing_time_ms: float = Field(default=0.0)


class FloodResponsePlanningAgent(BaseAgent):
    """
    GL-ADAPT-PUB-002: Flood Response Planning Agent

    Plans flood emergency response and evacuation.
    """

    AGENT_ID = "GL-ADAPT-PUB-002"
    AGENT_NAME = "Flood Response Planning Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Flood emergency response planning",
                version=self.VERSION,
            )
        super().__init__(config)
        self._plans: Dict[str, FloodResponsePlan] = {}
        self.logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        import time
        start_time = time.time()
        try:
            agent_input = FloodResponseInput(**input_data)
            handlers = {
                'create_plan': self._create_plan,
                'add_flood_zone': self._add_flood_zone,
                'add_evacuation_route': self._add_evacuation_route,
                'add_shelter': self._add_shelter,
                'assess_risk': self._assess_risk,
                'calculate_evacuation_time': self._calculate_evacuation_time,
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

    def _create_plan(self, inp: FloodResponseInput) -> FloodResponseOutput:
        if not inp.municipality_name:
            return FloodResponseOutput(success=False, action='create_plan', error="Municipality name required")
        plan_id = f"FRP-{inp.municipality_name.upper()[:3]}-{DeterministicClock.now().strftime('%Y%m%d')}"
        plan = FloodResponsePlan(
            plan_id=plan_id,
            municipality_name=inp.municipality_name,
            plan_name=f"{inp.municipality_name} Flood Response Plan",
        )
        self._plans[plan_id] = plan
        return FloodResponseOutput(success=True, action='create_plan', plan=plan, calculation_trace=[f"Created plan {plan_id}"])

    def _add_flood_zone(self, inp: FloodResponseInput) -> FloodResponseOutput:
        if not inp.plan_id or not inp.flood_zone:
            return FloodResponseOutput(success=False, action='add_flood_zone', error="Plan ID and flood zone required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return FloodResponseOutput(success=False, action='add_flood_zone', error=f"Plan not found")
        plan.flood_zones.append(inp.flood_zone)
        plan.total_at_risk_population = sum(z.population for z in plan.flood_zones if z.risk_level in (FloodRiskLevel.HIGH, FloodRiskLevel.EXTREME))
        plan.updated_at = DeterministicClock.now()
        return FloodResponseOutput(success=True, action='add_flood_zone', plan=plan, calculation_trace=[f"Added zone {inp.flood_zone.zone_id}"])

    def _add_evacuation_route(self, inp: FloodResponseInput) -> FloodResponseOutput:
        if not inp.plan_id or not inp.evacuation_route:
            return FloodResponseOutput(success=False, action='add_evacuation_route', error="Plan ID and route required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return FloodResponseOutput(success=False, action='add_evacuation_route', error=f"Plan not found")
        plan.evacuation_routes.append(inp.evacuation_route)
        plan.updated_at = DeterministicClock.now()
        return FloodResponseOutput(success=True, action='add_evacuation_route', plan=plan, calculation_trace=[f"Added route {inp.evacuation_route.route_id}"])

    def _add_shelter(self, inp: FloodResponseInput) -> FloodResponseOutput:
        if not inp.plan_id or not inp.shelter:
            return FloodResponseOutput(success=False, action='add_shelter', error="Plan ID and shelter required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return FloodResponseOutput(success=False, action='add_shelter', error=f"Plan not found")
        plan.shelters.append(inp.shelter)
        plan.total_shelter_capacity = sum(s.capacity for s in plan.shelters)
        plan.updated_at = DeterministicClock.now()
        return FloodResponseOutput(success=True, action='add_shelter', plan=plan, calculation_trace=[f"Added shelter {inp.shelter.shelter_id}"])

    def _assess_risk(self, inp: FloodResponseInput) -> FloodResponseOutput:
        if not inp.plan_id:
            return FloodResponseOutput(success=False, action='assess_risk', error="Plan ID required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return FloodResponseOutput(success=False, action='assess_risk', error=f"Plan not found")
        high_risk_zones = [z for z in plan.flood_zones if z.risk_level in (FloodRiskLevel.HIGH, FloodRiskLevel.EXTREME)]
        assessment = {
            "total_zones": len(plan.flood_zones),
            "high_risk_zones": len(high_risk_zones),
            "at_risk_population": plan.total_at_risk_population,
            "shelter_capacity": plan.total_shelter_capacity,
            "shelter_gap": max(0, plan.total_at_risk_population - plan.total_shelter_capacity),
            "evacuation_routes": len(plan.evacuation_routes),
        }
        return FloodResponseOutput(success=True, action='assess_risk', plan=plan, risk_assessment=assessment, calculation_trace=["Risk assessed"])

    def _calculate_evacuation_time(self, inp: FloodResponseInput) -> FloodResponseOutput:
        if not inp.plan_id:
            return FloodResponseOutput(success=False, action='calculate_evacuation_time', error="Plan ID required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return FloodResponseOutput(success=False, action='calculate_evacuation_time', error=f"Plan not found")
        total_capacity = sum(r.capacity_vehicles_per_hour for r in plan.evacuation_routes)
        vehicles_needed = plan.total_at_risk_population / 2.5  # avg persons per vehicle
        hours_needed = vehicles_needed / total_capacity if total_capacity > 0 else float('inf')
        analysis = {
            "total_evacuation_capacity_vph": total_capacity,
            "vehicles_needed": vehicles_needed,
            "estimated_hours": hours_needed,
            "routes_available": len(plan.evacuation_routes),
        }
        return FloodResponseOutput(success=True, action='calculate_evacuation_time', plan=plan, evacuation_analysis=analysis, calculation_trace=[f"Evacuation time: {hours_needed:.1f} hours"])

    def _get_plan(self, inp: FloodResponseInput) -> FloodResponseOutput:
        if not inp.plan_id:
            return FloodResponseOutput(success=False, action='get_plan', error="Plan ID required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return FloodResponseOutput(success=False, action='get_plan', error=f"Plan not found")
        return FloodResponseOutput(success=True, action='get_plan', plan=plan)

    def _list_plans(self, inp: FloodResponseInput) -> FloodResponseOutput:
        return FloodResponseOutput(success=True, action='list_plans', plans=list(self._plans.values()))

    def _hash_output(self, output: FloodResponseOutput) -> str:
        content = {"action": output.action, "success": output.success, "timestamp": output.timestamp.isoformat()}
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
