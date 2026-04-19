# -*- coding: utf-8 -*-
"""
GL-ADAPT-NBS-009: Climate-Resilient Landscapes Agent
=====================================================

Integrated landscape planning for climate resilience.

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


class ResilienceLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class LandscapeType(str, Enum):
    AGRICULTURAL = "agricultural"
    FOREST = "forest"
    COASTAL = "coastal"
    URBAN_PERIURBAN = "urban_periurban"
    MOUNTAIN = "mountain"
    DRYLAND = "dryland"


class LandscapeUnit(BaseModel):
    unit_id: str = Field(...)
    area_km2: float = Field(..., gt=0)
    landscape_type: LandscapeType = Field(...)
    natural_land_cover_pct: float = Field(default=30, ge=0, le=100)
    biodiversity_index: float = Field(default=0.5, ge=0, le=1)
    water_retention_score: float = Field(default=0.5, ge=0, le=1)
    carbon_stock_tco2_per_ha: float = Field(default=100, ge=0)
    population: int = Field(default=10000, ge=0)
    economic_value_million: float = Field(default=50, ge=0)


class LandscapePlan(BaseModel):
    unit_id: str = Field(...)
    current_resilience: ResilienceLevel = Field(...)
    target_resilience: ResilienceLevel = Field(...)
    resilience_score_current: float = Field(...)
    resilience_score_projected: float = Field(...)
    interventions: List[Dict[str, Any]] = Field(...)
    natural_land_target_pct: float = Field(...)
    carbon_sequestration_potential_tco2: float = Field(...)
    ecosystem_services_value_million_annual: float = Field(...)
    implementation_cost_million: float = Field(...)
    benefit_cost_ratio: float = Field(...)


class LandscapeInput(BaseModel):
    project_id: str = Field(...)
    region_name: str = Field(...)
    units: List[LandscapeUnit] = Field(..., min_length=1)
    climate_scenario: str = Field(default="RCP4.5")
    planning_horizon_years: int = Field(default=30, ge=10, le=100)


class LandscapeOutput(BaseModel):
    project_id: str = Field(...)
    region_name: str = Field(...)
    calculation_date: datetime = Field(default_factory=DeterministicClock.now)
    total_area_km2: float = Field(...)
    total_population: int = Field(...)
    current_avg_resilience_score: float = Field(...)
    projected_avg_resilience_score: float = Field(...)
    plans: List[LandscapePlan] = Field(...)
    total_carbon_sequestration_tco2: float = Field(...)
    total_ecosystem_services_million_annual: float = Field(...)
    total_investment_million: float = Field(...)
    portfolio_bcr: float = Field(...)
    provenance_hash: str = Field(...)


class ClimateResilientLandscapesAgent(BaseAgent):
    """GL-ADAPT-NBS-009: Climate-Resilient Landscapes Agent"""

    AGENT_ID = "GL-ADAPT-NBS-009"
    AGENT_NAME = "Climate-Resilient Landscapes Agent"
    VERSION = "1.0.0"

    LANDSCAPE_TARGETS = {
        LandscapeType.AGRICULTURAL: {"natural_land": 25, "priority_interventions": ["agroforestry", "hedgerows", "buffer_strips"]},
        LandscapeType.FOREST: {"natural_land": 80, "priority_interventions": ["restoration", "fire_management", "connectivity"]},
        LandscapeType.COASTAL: {"natural_land": 40, "priority_interventions": ["wetland_restoration", "dune_stabilization", "mangroves"]},
        LandscapeType.URBAN_PERIURBAN: {"natural_land": 30, "priority_interventions": ["green_infrastructure", "urban_forests", "green_corridors"]},
        LandscapeType.MOUNTAIN: {"natural_land": 60, "priority_interventions": ["watershed_protection", "slope_stabilization", "pasture_management"]},
        LandscapeType.DRYLAND: {"natural_land": 35, "priority_interventions": ["rangeland_restoration", "water_harvesting", "windbreaks"]},
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Climate-resilient landscape planning",
                version=self.VERSION
            )
        super().__init__(config)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        try:
            agent_input = LandscapeInput(**input_data)
            plans = []

            for unit in agent_input.units:
                plan = self._create_landscape_plan(
                    unit, agent_input.climate_scenario, agent_input.planning_horizon_years
                )
                plans.append(plan)

            total_area = sum(u.area_km2 for u in agent_input.units)
            total_pop = sum(u.population for u in agent_input.units)

            current_avg_resilience = sum(p.resilience_score_current for p in plans) / len(plans)
            projected_avg_resilience = sum(p.resilience_score_projected for p in plans) / len(plans)

            total_carbon = sum(p.carbon_sequestration_potential_tco2 for p in plans)
            total_services = sum(p.ecosystem_services_value_million_annual for p in plans)
            total_investment = sum(p.implementation_cost_million for p in plans)

            portfolio_bcr = (total_services * agent_input.planning_horizon_years) / total_investment if total_investment > 0 else 0

            provenance_hash = hashlib.sha256(
                json.dumps({"agent": self.AGENT_ID, "ts": DeterministicClock.now().isoformat()}).encode()
            ).hexdigest()

            output = LandscapeOutput(
                project_id=agent_input.project_id,
                region_name=agent_input.region_name,
                total_area_km2=total_area,
                total_population=total_pop,
                current_avg_resilience_score=round(current_avg_resilience, 2),
                projected_avg_resilience_score=round(projected_avg_resilience, 2),
                plans=plans,
                total_carbon_sequestration_tco2=round(total_carbon, 0),
                total_ecosystem_services_million_annual=round(total_services, 2),
                total_investment_million=round(total_investment, 2),
                portfolio_bcr=round(portfolio_bcr, 2),
                provenance_hash=provenance_hash,
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    def _create_landscape_plan(
        self, unit: LandscapeUnit, scenario: str, horizon: int
    ) -> LandscapePlan:
        resilience_score = (
            unit.natural_land_cover_pct * 0.25 +
            unit.biodiversity_index * 100 * 0.25 +
            unit.water_retention_score * 100 * 0.25 +
            min(100, unit.carbon_stock_tco2_per_ha / 3) * 0.25
        )

        climate_stress = 0.8 if scenario == "RCP4.5" else 0.6
        current_resilience_score = resilience_score * climate_stress

        if current_resilience_score >= 70:
            current_resilience = ResilienceLevel.VERY_HIGH
        elif current_resilience_score >= 50:
            current_resilience = ResilienceLevel.HIGH
        elif current_resilience_score >= 30:
            current_resilience = ResilienceLevel.MODERATE
        else:
            current_resilience = ResilienceLevel.LOW

        targets = self.LANDSCAPE_TARGETS.get(unit.landscape_type, {"natural_land": 30, "priority_interventions": []})
        natural_land_target = targets["natural_land"]

        interventions = self._design_interventions(
            unit, natural_land_target, targets["priority_interventions"]
        )

        total_cost = sum(i["cost_million"] for i in interventions)

        natural_land_increase = max(0, natural_land_target - unit.natural_land_cover_pct)
        carbon_sequestration = unit.area_km2 * 100 * natural_land_increase / 100 * 5

        projected_resilience_score = min(100, current_resilience_score + natural_land_increase * 0.5 + 10)

        if projected_resilience_score >= 70:
            target_resilience = ResilienceLevel.VERY_HIGH
        elif projected_resilience_score >= 50:
            target_resilience = ResilienceLevel.HIGH
        elif projected_resilience_score >= 30:
            target_resilience = ResilienceLevel.MODERATE
        else:
            target_resilience = ResilienceLevel.LOW

        ecosystem_services = (
            unit.area_km2 * (natural_land_target / 100) * 200 +
            carbon_sequestration * 0.02 +
            unit.population * 0.005
        ) / 1_000_000

        bcr = (ecosystem_services * horizon) / total_cost if total_cost > 0 else 0

        return LandscapePlan(
            unit_id=unit.unit_id,
            current_resilience=current_resilience,
            target_resilience=target_resilience,
            resilience_score_current=round(current_resilience_score, 1),
            resilience_score_projected=round(projected_resilience_score, 1),
            interventions=interventions,
            natural_land_target_pct=natural_land_target,
            carbon_sequestration_potential_tco2=round(carbon_sequestration, 0),
            ecosystem_services_value_million_annual=round(ecosystem_services, 3),
            implementation_cost_million=round(total_cost, 2),
            benefit_cost_ratio=round(bcr, 2),
        )

    def _design_interventions(
        self, unit: LandscapeUnit, target_natural: float, priority_types: List[str]
    ) -> List[Dict[str, Any]]:
        interventions = []
        natural_deficit = max(0, target_natural - unit.natural_land_cover_pct)
        restoration_area_km2 = unit.area_km2 * (natural_deficit / 100)

        if restoration_area_km2 > 0:
            for i, intervention_type in enumerate(priority_types[:3]):
                allocation = [0.5, 0.3, 0.2][i] if i < 3 else 0.2
                area = restoration_area_km2 * allocation
                cost = area * 100 * self._get_intervention_cost(intervention_type) / 1_000_000

                interventions.append({
                    "type": intervention_type,
                    "area_km2": round(area, 2),
                    "cost_million": round(cost, 3),
                    "carbon_benefit_tco2": round(area * 100 * 5, 0),
                })

        if unit.water_retention_score < 0.5:
            water_cost = unit.area_km2 * 0.1 * 50000 / 1_000_000
            interventions.append({
                "type": "water_retention_improvement",
                "area_km2": round(unit.area_km2 * 0.1, 2),
                "cost_million": round(water_cost, 3),
                "carbon_benefit_tco2": 0,
            })

        return interventions

    def _get_intervention_cost(self, intervention_type: str) -> float:
        costs = {
            "agroforestry": 3000,
            "hedgerows": 5000,
            "buffer_strips": 2000,
            "restoration": 8000,
            "fire_management": 1000,
            "connectivity": 6000,
            "wetland_restoration": 15000,
            "dune_stabilization": 10000,
            "mangroves": 12000,
            "green_infrastructure": 50000,
            "urban_forests": 30000,
            "green_corridors": 20000,
            "watershed_protection": 5000,
            "slope_stabilization": 8000,
            "pasture_management": 1500,
            "rangeland_restoration": 2000,
            "water_harvesting": 4000,
            "windbreaks": 3000,
        }
        return costs.get(intervention_type, 5000)
