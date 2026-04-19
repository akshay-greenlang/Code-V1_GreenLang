# -*- coding: utf-8 -*-
"""
GL-ADAPT-PUB-004: Public Health & Climate Agent
================================================

Assesses climate-related health risks and plans public health adaptations.

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


class HealthOutcome(str, Enum):
    """Climate-related health outcomes."""
    HEAT_ILLNESS = "heat_illness"
    RESPIRATORY = "respiratory"
    CARDIOVASCULAR = "cardiovascular"
    INFECTIOUS_DISEASE = "infectious_disease"
    WATER_BORNE = "water_borne"
    MENTAL_HEALTH = "mental_health"
    INJURIES = "injuries"
    ALLERGIES = "allergies"


class VulnerablePopulation(BaseModel):
    """Vulnerable population group."""
    group_id: str = Field(...)
    group_name: str = Field(...)
    population_count: int = Field(..., ge=0)
    primary_health_risks: List[HealthOutcome] = Field(default_factory=list)
    geographic_concentration: Optional[str] = Field(None)
    access_to_healthcare: str = Field(default="moderate")  # low, moderate, high


class HealthInterventionProgram(BaseModel):
    """Health intervention program."""
    program_id: str = Field(...)
    name: str = Field(...)
    target_outcomes: List[HealthOutcome] = Field(default_factory=list)
    target_populations: List[str] = Field(default_factory=list)
    estimated_cost_usd: float = Field(default=0.0, ge=0)
    estimated_lives_saved_per_year: float = Field(default=0.0, ge=0)
    implementation_status: str = Field(default="planned")


class HealthAdaptationPlan(BaseModel):
    """Health adaptation plan."""
    plan_id: str = Field(...)
    jurisdiction_name: str = Field(...)
    plan_name: str = Field(...)
    vulnerable_populations: List[VulnerablePopulation] = Field(default_factory=list)
    intervention_programs: List[HealthInterventionProgram] = Field(default_factory=list)
    total_vulnerable_population: int = Field(default=0, ge=0)
    total_program_budget_usd: float = Field(default=0.0, ge=0)
    created_at: datetime = Field(default_factory=DeterministicClock.now)
    updated_at: datetime = Field(default_factory=DeterministicClock.now)
    provenance_hash: Optional[str] = Field(None)


class PublicHealthInput(BaseModel):
    """Input for Public Health Agent."""
    action: str = Field(...)
    plan_id: Optional[str] = Field(None)
    jurisdiction_name: Optional[str] = Field(None)
    vulnerable_population: Optional[VulnerablePopulation] = Field(None)
    intervention_program: Optional[HealthInterventionProgram] = Field(None)
    user_id: Optional[str] = Field(None)

    @field_validator('action')
    @classmethod
    def validate_action(cls, v: str) -> str:
        valid = {'create_plan', 'add_vulnerable_population', 'add_intervention',
                 'assess_health_risks', 'calculate_program_effectiveness', 'get_plan', 'list_plans'}
        if v not in valid:
            raise ValueError(f"Invalid action: {v}")
        return v


class PublicHealthOutput(BaseModel):
    """Output from Public Health Agent."""
    success: bool = Field(...)
    action: str = Field(...)
    plan: Optional[HealthAdaptationPlan] = Field(None)
    plans: Optional[List[HealthAdaptationPlan]] = Field(None)
    health_risk_assessment: Optional[Dict[str, Any]] = Field(None)
    program_effectiveness: Optional[Dict[str, Any]] = Field(None)
    provenance_hash: Optional[str] = Field(None)
    calculation_trace: List[str] = Field(default_factory=list)
    error: Optional[str] = Field(None)
    warnings: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    processing_time_ms: float = Field(default=0.0)


class PublicHealthClimateAgent(BaseAgent):
    """
    GL-ADAPT-PUB-004: Public Health & Climate Agent

    Plans public health adaptations to climate change.
    """

    AGENT_ID = "GL-ADAPT-PUB-004"
    AGENT_NAME = "Public Health & Climate Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Climate-health adaptation planning",
                version=self.VERSION,
            )
        super().__init__(config)
        self._plans: Dict[str, HealthAdaptationPlan] = {}
        self.logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        import time
        start_time = time.time()
        try:
            agent_input = PublicHealthInput(**input_data)
            handlers = {
                'create_plan': self._create_plan,
                'add_vulnerable_population': self._add_vulnerable_population,
                'add_intervention': self._add_intervention,
                'assess_health_risks': self._assess_health_risks,
                'calculate_program_effectiveness': self._calculate_effectiveness,
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

    def _create_plan(self, inp: PublicHealthInput) -> PublicHealthOutput:
        if not inp.jurisdiction_name:
            return PublicHealthOutput(success=False, action='create_plan', error="Jurisdiction name required")
        plan_id = f"PHC-{inp.jurisdiction_name.upper()[:3]}-{DeterministicClock.now().strftime('%Y%m%d')}"
        plan = HealthAdaptationPlan(
            plan_id=plan_id,
            jurisdiction_name=inp.jurisdiction_name,
            plan_name=f"{inp.jurisdiction_name} Health-Climate Plan",
        )
        self._plans[plan_id] = plan
        return PublicHealthOutput(success=True, action='create_plan', plan=plan, calculation_trace=[f"Created plan {plan_id}"])

    def _add_vulnerable_population(self, inp: PublicHealthInput) -> PublicHealthOutput:
        if not inp.plan_id or not inp.vulnerable_population:
            return PublicHealthOutput(success=False, action='add_vulnerable_population', error="Plan ID and population required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return PublicHealthOutput(success=False, action='add_vulnerable_population', error="Plan not found")
        plan.vulnerable_populations.append(inp.vulnerable_population)
        plan.total_vulnerable_population = sum(p.population_count for p in plan.vulnerable_populations)
        plan.updated_at = DeterministicClock.now()
        return PublicHealthOutput(success=True, action='add_vulnerable_population', plan=plan, calculation_trace=[f"Added population {inp.vulnerable_population.group_id}"])

    def _add_intervention(self, inp: PublicHealthInput) -> PublicHealthOutput:
        if not inp.plan_id or not inp.intervention_program:
            return PublicHealthOutput(success=False, action='add_intervention', error="Plan ID and program required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return PublicHealthOutput(success=False, action='add_intervention', error="Plan not found")
        plan.intervention_programs.append(inp.intervention_program)
        plan.total_program_budget_usd = sum(p.estimated_cost_usd for p in plan.intervention_programs)
        plan.updated_at = DeterministicClock.now()
        return PublicHealthOutput(success=True, action='add_intervention', plan=plan, calculation_trace=[f"Added program {inp.intervention_program.program_id}"])

    def _assess_health_risks(self, inp: PublicHealthInput) -> PublicHealthOutput:
        if not inp.plan_id:
            return PublicHealthOutput(success=False, action='assess_health_risks', error="Plan ID required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return PublicHealthOutput(success=False, action='assess_health_risks', error="Plan not found")
        by_outcome: Dict[str, int] = {}
        for pop in plan.vulnerable_populations:
            for risk in pop.primary_health_risks:
                by_outcome[risk.value] = by_outcome.get(risk.value, 0) + pop.population_count
        assessment = {
            "total_vulnerable": plan.total_vulnerable_population,
            "population_by_health_outcome": by_outcome,
            "intervention_coverage": len(plan.intervention_programs),
            "uncovered_outcomes": [o.value for o in HealthOutcome if o.value not in [t.value for p in plan.intervention_programs for t in p.target_outcomes]],
        }
        return PublicHealthOutput(success=True, action='assess_health_risks', plan=plan, health_risk_assessment=assessment, calculation_trace=["Risk assessment completed"])

    def _calculate_effectiveness(self, inp: PublicHealthInput) -> PublicHealthOutput:
        if not inp.plan_id:
            return PublicHealthOutput(success=False, action='calculate_program_effectiveness', error="Plan ID required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return PublicHealthOutput(success=False, action='calculate_program_effectiveness', error="Plan not found")
        total_lives_saved = sum(p.estimated_lives_saved_per_year for p in plan.intervention_programs)
        total_cost = plan.total_program_budget_usd
        cost_per_life = total_cost / total_lives_saved if total_lives_saved > 0 else 0
        effectiveness = {
            "total_programs": len(plan.intervention_programs),
            "total_budget_usd": total_cost,
            "estimated_lives_saved_per_year": total_lives_saved,
            "cost_per_life_saved_usd": cost_per_life,
        }
        return PublicHealthOutput(success=True, action='calculate_program_effectiveness', plan=plan, program_effectiveness=effectiveness, calculation_trace=["Effectiveness calculated"])

    def _get_plan(self, inp: PublicHealthInput) -> PublicHealthOutput:
        if not inp.plan_id:
            return PublicHealthOutput(success=False, action='get_plan', error="Plan ID required")
        plan = self._plans.get(inp.plan_id)
        if not plan:
            return PublicHealthOutput(success=False, action='get_plan', error="Plan not found")
        return PublicHealthOutput(success=True, action='get_plan', plan=plan)

    def _list_plans(self, inp: PublicHealthInput) -> PublicHealthOutput:
        return PublicHealthOutput(success=True, action='list_plans', plans=list(self._plans.values()))

    def _hash_output(self, output: PublicHealthOutput) -> str:
        content = {"action": output.action, "success": output.success, "timestamp": output.timestamp.isoformat()}
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
