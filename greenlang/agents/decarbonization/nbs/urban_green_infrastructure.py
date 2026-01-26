# -*- coding: utf-8 -*-
"""
GL-DECARB-NBS-007: Urban Green Infrastructure Agent
===================================================

Plans urban greening projects for carbon sequestration and co-benefits.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism.clock import DeterministicClock

logger = logging.getLogger(__name__)


class GreenInfrastructureType(str, Enum):
    URBAN_FOREST = "urban_forest"
    STREET_TREES = "street_trees"
    GREEN_ROOF = "green_roof"
    URBAN_PARK = "urban_park"
    GREEN_CORRIDOR = "green_corridor"
    COMMUNITY_GARDEN = "community_garden"
    BIOSWALE = "bioswale"
    VERTICAL_GARDEN = "vertical_garden"


# Annual carbon sequestration (kg CO2e per unit)
SEQUESTRATION_FACTORS = {
    GreenInfrastructureType.URBAN_FOREST: 20000,  # per hectare
    GreenInfrastructureType.STREET_TREES: 20,      # per tree
    GreenInfrastructureType.GREEN_ROOF: 3000,      # per 1000 m2
    GreenInfrastructureType.URBAN_PARK: 15000,     # per hectare
    GreenInfrastructureType.GREEN_CORRIDOR: 18000,  # per hectare
    GreenInfrastructureType.COMMUNITY_GARDEN: 8000, # per hectare
    GreenInfrastructureType.BIOSWALE: 500,         # per 100m
    GreenInfrastructureType.VERTICAL_GARDEN: 200,   # per 100 m2
}

# Costs (USD per unit)
UNIT_COSTS = {
    GreenInfrastructureType.URBAN_FOREST: 30000,
    GreenInfrastructureType.STREET_TREES: 500,
    GreenInfrastructureType.GREEN_ROOF: 15000,
    GreenInfrastructureType.URBAN_PARK: 50000,
    GreenInfrastructureType.GREEN_CORRIDOR: 25000,
    GreenInfrastructureType.COMMUNITY_GARDEN: 10000,
    GreenInfrastructureType.BIOSWALE: 2000,
    GreenInfrastructureType.VERTICAL_GARDEN: 5000,
}


class UrbanGreenProject(BaseModel):
    project_id: str = Field(...)
    infrastructure_type: GreenInfrastructureType = Field(...)
    quantity: float = Field(..., gt=0)  # hectares, trees, or m2 depending on type
    unit: str = Field(default="units")


class UrbanGreenPlan(BaseModel):
    project_id: str = Field(...)
    infrastructure_type: GreenInfrastructureType = Field(...)
    annual_sequestration_kg_co2e: float = Field(...)
    total_30yr_co2e_tonnes: float = Field(...)
    total_cost_usd: float = Field(...)
    cost_per_tonne_co2e: float = Field(...)
    heat_island_reduction_score: float = Field(..., ge=0, le=100)
    air_quality_benefit_score: float = Field(..., ge=0, le=100)
    stormwater_benefit_score: float = Field(..., ge=0, le=100)


class UrbanGreenInput(BaseModel):
    city_name: str = Field(...)
    projects: List[UrbanGreenProject] = Field(..., min_length=1)
    project_duration_years: int = Field(default=30)


class UrbanGreenOutput(BaseModel):
    city_name: str = Field(...)
    calculation_date: datetime = Field(default_factory=DeterministicClock.now)
    total_sequestration_30yr_co2e: float = Field(...)
    total_cost_usd: float = Field(...)
    average_cost_per_tonne: float = Field(...)
    plans: List[UrbanGreenPlan] = Field(...)
    aggregate_heat_island_score: float = Field(...)
    aggregate_air_quality_score: float = Field(...)
    provenance_hash: str = Field(...)
    warnings: List[str] = Field(default_factory=list)


class UrbanGreenInfrastructureAgent(BaseAgent):
    """GL-DECARB-NBS-007: Urban Green Infrastructure Agent"""

    AGENT_ID = "GL-DECARB-NBS-007"
    AGENT_NAME = "Urban Green Infrastructure Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(name=self.AGENT_NAME, description="Urban greening planning", version=self.VERSION)
        super().__init__(config)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        try:
            agent_input = UrbanGreenInput(**input_data)
            plans = []

            for project in agent_input.projects:
                plan = self._create_plan(project, agent_input.project_duration_years)
                plans.append(plan)

            total_seq = sum(p.total_30yr_co2e_tonnes for p in plans)
            total_cost = sum(p.total_cost_usd for p in plans)
            avg_cost = total_cost / total_seq if total_seq > 0 else 0

            agg_heat = sum(p.heat_island_reduction_score for p in plans) / len(plans)
            agg_air = sum(p.air_quality_benefit_score for p in plans) / len(plans)

            provenance_hash = hashlib.sha256(
                json.dumps({"agent": self.AGENT_ID, "ts": DeterministicClock.now().isoformat()}).encode()
            ).hexdigest()

            output = UrbanGreenOutput(
                city_name=agent_input.city_name,
                total_sequestration_30yr_co2e=total_seq,
                total_cost_usd=total_cost,
                average_cost_per_tonne=avg_cost,
                plans=plans,
                aggregate_heat_island_score=agg_heat,
                aggregate_air_quality_score=agg_air,
                provenance_hash=provenance_hash,
                warnings=[]
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    def _create_plan(self, project: UrbanGreenProject, duration: int) -> UrbanGreenPlan:
        infra_type = project.infrastructure_type
        seq_factor = SEQUESTRATION_FACTORS.get(infra_type, 1000)
        cost_factor = UNIT_COSTS.get(infra_type, 5000)

        annual_seq_kg = seq_factor * project.quantity
        total_seq_tonnes = (annual_seq_kg * duration) / 1000

        total_cost = cost_factor * project.quantity
        cost_per_tonne = total_cost / total_seq_tonnes if total_seq_tonnes > 0 else 0

        # Co-benefit scores
        heat_scores = {
            GreenInfrastructureType.URBAN_FOREST: 90,
            GreenInfrastructureType.STREET_TREES: 75,
            GreenInfrastructureType.GREEN_ROOF: 85,
            GreenInfrastructureType.URBAN_PARK: 80,
        }
        air_scores = {
            GreenInfrastructureType.URBAN_FOREST: 95,
            GreenInfrastructureType.STREET_TREES: 85,
            GreenInfrastructureType.GREEN_CORRIDOR: 80,
        }
        storm_scores = {
            GreenInfrastructureType.BIOSWALE: 95,
            GreenInfrastructureType.GREEN_ROOF: 80,
            GreenInfrastructureType.URBAN_PARK: 70,
        }

        return UrbanGreenPlan(
            project_id=project.project_id,
            infrastructure_type=infra_type,
            annual_sequestration_kg_co2e=round(annual_seq_kg, 2),
            total_30yr_co2e_tonnes=round(total_seq_tonnes, 2),
            total_cost_usd=round(total_cost, 2),
            cost_per_tonne_co2e=round(cost_per_tonne, 2),
            heat_island_reduction_score=heat_scores.get(infra_type, 50),
            air_quality_benefit_score=air_scores.get(infra_type, 50),
            stormwater_benefit_score=storm_scores.get(infra_type, 50)
        )
