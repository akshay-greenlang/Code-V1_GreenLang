# -*- coding: utf-8 -*-
"""
GL-ADAPT-PUB-001: Urban Heat Action Agent
==========================================

Plans and manages urban heat wave response including cooling center
networks, vulnerable population identification, and heat action protocols.

Capabilities:
    - Heat vulnerability mapping
    - Cooling center network planning
    - Heat wave early warning protocols
    - Urban heat island analysis
    - Green infrastructure recommendations
    - Heat-health surveillance support

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


class HeatAlertLevel(str, Enum):
    """Heat alert levels."""
    NORMAL = "normal"
    WATCH = "watch"
    WARNING = "warning"
    EMERGENCY = "emergency"


class VulnerabilityLevel(str, Enum):
    """Vulnerability levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class HeatVulnerabilityZone(BaseModel):
    """Zone with heat vulnerability assessment."""
    zone_id: str = Field(...)
    zone_name: str = Field(...)
    population: int = Field(..., ge=0)
    vulnerability_level: VulnerabilityLevel = Field(...)
    elderly_population_percent: float = Field(default=0.0, ge=0, le=100)
    tree_canopy_cover_percent: float = Field(default=0.0, ge=0, le=100)
    impervious_surface_percent: float = Field(default=0.0, ge=0, le=100)
    ac_penetration_percent: float = Field(default=0.0, ge=0, le=100)
    heat_index_differential_f: float = Field(default=0.0)
    cooling_centers_available: int = Field(default=0, ge=0)


class CoolingCenter(BaseModel):
    """Cooling center facility."""
    center_id: str = Field(...)
    name: str = Field(...)
    address: str = Field(...)
    capacity: int = Field(..., ge=0)
    ada_accessible: bool = Field(default=True)
    hours_of_operation: str = Field(default="9am-9pm")
    has_backup_power: bool = Field(default=False)
    services_available: List[str] = Field(default_factory=list)
    latitude: Optional[float] = Field(None)
    longitude: Optional[float] = Field(None)


class HeatActionPlan(BaseModel):
    """Heat action plan for municipality."""
    plan_id: str = Field(...)
    municipality_name: str = Field(...)
    plan_name: str = Field(...)
    vulnerability_zones: List[HeatVulnerabilityZone] = Field(default_factory=list)
    cooling_centers: List[CoolingCenter] = Field(default_factory=list)
    alert_thresholds: Dict[str, float] = Field(default_factory=dict)
    response_protocols: Dict[str, List[str]] = Field(default_factory=dict)
    total_vulnerable_population: int = Field(default=0, ge=0)
    total_cooling_capacity: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=DeterministicClock.now)
    updated_at: datetime = Field(default_factory=DeterministicClock.now)
    provenance_hash: Optional[str] = Field(None)


class UrbanHeatActionInput(BaseModel):
    """Input for Urban Heat Action Agent."""
    action: str = Field(...)
    plan_id: Optional[str] = Field(None)
    municipality_name: Optional[str] = Field(None)
    zone: Optional[HeatVulnerabilityZone] = Field(None)
    cooling_center: Optional[CoolingCenter] = Field(None)
    forecast_temperature_f: Optional[float] = Field(None)
    user_id: Optional[str] = Field(None)

    @field_validator('action')
    @classmethod
    def validate_action(cls, v: str) -> str:
        valid = {'create_plan', 'add_zone', 'add_cooling_center', 'assess_heat_risk',
                 'activate_protocol', 'calculate_capacity_gap', 'get_plan', 'list_plans'}
        if v not in valid:
            raise ValueError(f"Invalid action: {v}")
        return v


class UrbanHeatActionOutput(BaseModel):
    """Output from Urban Heat Action Agent."""
    success: bool = Field(...)
    action: str = Field(...)
    plan: Optional[HeatActionPlan] = Field(None)
    plans: Optional[List[HeatActionPlan]] = Field(None)
    risk_assessment: Optional[Dict[str, Any]] = Field(None)
    capacity_analysis: Optional[Dict[str, Any]] = Field(None)
    alert_level: Optional[HeatAlertLevel] = Field(None)
    provenance_hash: Optional[str] = Field(None)
    calculation_trace: List[str] = Field(default_factory=list)
    error: Optional[str] = Field(None)
    warnings: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    processing_time_ms: float = Field(default=0.0)


class UrbanHeatActionAgent(BaseAgent):
    """
    GL-ADAPT-PUB-001: Urban Heat Action Agent

    Plans and manages urban heat wave response.
    """

    AGENT_ID = "GL-ADAPT-PUB-001"
    AGENT_NAME = "Urban Heat Action Agent"
    VERSION = "1.0.0"

    HEAT_THRESHOLDS = {
        HeatAlertLevel.WATCH: 95.0,
        HeatAlertLevel.WARNING: 100.0,
        HeatAlertLevel.EMERGENCY: 105.0,
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Urban heat wave response planning",
                version=self.VERSION,
            )
        super().__init__(config)
        self._plans: Dict[str, HeatActionPlan] = {}
        self.logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        import time
        start_time = time.time()
        try:
            agent_input = UrbanHeatActionInput(**input_data)
            handlers = {
                'create_plan': self._create_plan,
                'add_zone': self._add_zone,
                'add_cooling_center': self._add_cooling_center,
                'assess_heat_risk': self._assess_heat_risk,
                'activate_protocol': self._activate_protocol,
                'calculate_capacity_gap': self._calculate_capacity_gap,
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

    def _create_plan(self, inp: UrbanHeatActionInput) -> UrbanHeatActionOutput:
        if not inp.municipality_name:
            return UrbanHeatActionOutput(success=False, action='create_plan', error="Municipality name required")
        plan_id = f"HAP-{inp.municipality_name.upper()[:3]}-{DeterministicClock.now().strftime('%Y%m%d')}"
        plan = HeatActionPlan(
            plan_id=plan_id,
            municipality_name=inp.municipality_name,
            plan_name=f"{inp.municipality_name} Heat Action Plan",
            alert_thresholds={k.value: v for k, v in self.HEAT_THRESHOLDS.items()},
            response_protocols={
                "watch": ["Issue public advisory", "Open cooling centers", "Check on vulnerable"],
                "warning": ["Extend cooling center hours", "Deploy mobile units", "Activate buddy system"],
                "emergency": ["24/7 cooling centers", "Emergency medical standby", "Door-to-door checks"],
            }
        )
        self._plans[plan_id] = plan
        return UrbanHeatActionOutput(success=True, action='create_plan', plan=plan, calculation_trace=[f"Created plan {plan_id}"])

    def _add_zone(self, inp: UrbanHeatActionInput) -> UrbanHeatActionOutput:
        if not inp.plan_id or not inp.zone:
            return UrbanHeatActionOutput(success=False, action='add_zone', error="Plan ID and zone required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return UrbanHeatActionOutput(success=False, action='add_zone', error=f"Plan not found: {inp.plan_id}")
        plan.vulnerability_zones.append(inp.zone)
        plan.total_vulnerable_population = sum(z.population for z in plan.vulnerability_zones if z.vulnerability_level in (VulnerabilityLevel.HIGH, VulnerabilityLevel.EXTREME))
        plan.updated_at = DeterministicClock.now()
        return UrbanHeatActionOutput(success=True, action='add_zone', plan=plan, calculation_trace=[f"Added zone {inp.zone.zone_id}"])

    def _add_cooling_center(self, inp: UrbanHeatActionInput) -> UrbanHeatActionOutput:
        if not inp.plan_id or not inp.cooling_center:
            return UrbanHeatActionOutput(success=False, action='add_cooling_center', error="Plan ID and cooling center required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return UrbanHeatActionOutput(success=False, action='add_cooling_center', error=f"Plan not found: {inp.plan_id}")
        plan.cooling_centers.append(inp.cooling_center)
        plan.total_cooling_capacity = sum(c.capacity for c in plan.cooling_centers)
        plan.updated_at = DeterministicClock.now()
        return UrbanHeatActionOutput(success=True, action='add_cooling_center', plan=plan, calculation_trace=[f"Added center {inp.cooling_center.center_id}"])

    def _assess_heat_risk(self, inp: UrbanHeatActionInput) -> UrbanHeatActionOutput:
        if not inp.plan_id or inp.forecast_temperature_f is None:
            return UrbanHeatActionOutput(success=False, action='assess_heat_risk', error="Plan ID and forecast temperature required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return UrbanHeatActionOutput(success=False, action='assess_heat_risk', error=f"Plan not found: {inp.plan_id}")
        temp = inp.forecast_temperature_f
        alert = HeatAlertLevel.NORMAL
        if temp >= self.HEAT_THRESHOLDS[HeatAlertLevel.EMERGENCY]:
            alert = HeatAlertLevel.EMERGENCY
        elif temp >= self.HEAT_THRESHOLDS[HeatAlertLevel.WARNING]:
            alert = HeatAlertLevel.WARNING
        elif temp >= self.HEAT_THRESHOLDS[HeatAlertLevel.WATCH]:
            alert = HeatAlertLevel.WATCH
        risk = {
            "forecast_temperature_f": temp,
            "alert_level": alert.value,
            "vulnerable_population": plan.total_vulnerable_population,
            "cooling_capacity": plan.total_cooling_capacity,
            "high_risk_zones": [z.zone_name for z in plan.vulnerability_zones if z.vulnerability_level in (VulnerabilityLevel.HIGH, VulnerabilityLevel.EXTREME)],
        }
        return UrbanHeatActionOutput(success=True, action='assess_heat_risk', plan=plan, risk_assessment=risk, alert_level=alert, calculation_trace=[f"Alert level: {alert.value}"])

    def _activate_protocol(self, inp: UrbanHeatActionInput) -> UrbanHeatActionOutput:
        if not inp.plan_id:
            return UrbanHeatActionOutput(success=False, action='activate_protocol', error="Plan ID required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return UrbanHeatActionOutput(success=False, action='activate_protocol', error=f"Plan not found: {inp.plan_id}")
        return UrbanHeatActionOutput(success=True, action='activate_protocol', plan=plan, calculation_trace=["Protocol activation logged"])

    def _calculate_capacity_gap(self, inp: UrbanHeatActionInput) -> UrbanHeatActionOutput:
        if not inp.plan_id:
            return UrbanHeatActionOutput(success=False, action='calculate_capacity_gap', error="Plan ID required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return UrbanHeatActionOutput(success=False, action='calculate_capacity_gap', error=f"Plan not found: {inp.plan_id}")
        gap = plan.total_vulnerable_population - plan.total_cooling_capacity
        analysis = {
            "vulnerable_population": plan.total_vulnerable_population,
            "cooling_capacity": plan.total_cooling_capacity,
            "capacity_gap": max(0, gap),
            "coverage_percent": (plan.total_cooling_capacity / plan.total_vulnerable_population * 100) if plan.total_vulnerable_population > 0 else 100,
        }
        return UrbanHeatActionOutput(success=True, action='calculate_capacity_gap', plan=plan, capacity_analysis=analysis, calculation_trace=[f"Capacity gap: {gap}"])

    def _get_plan(self, inp: UrbanHeatActionInput) -> UrbanHeatActionOutput:
        if not inp.plan_id:
            return UrbanHeatActionOutput(success=False, action='get_plan', error="Plan ID required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return UrbanHeatActionOutput(success=False, action='get_plan', error=f"Plan not found: {inp.plan_id}")
        return UrbanHeatActionOutput(success=True, action='get_plan', plan=plan)

    def _list_plans(self, inp: UrbanHeatActionInput) -> UrbanHeatActionOutput:
        return UrbanHeatActionOutput(success=True, action='list_plans', plans=list(self._plans.values()))

    def _hash_output(self, output: UrbanHeatActionOutput) -> str:
        content = {"action": output.action, "success": output.success, "timestamp": output.timestamp.isoformat()}
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
