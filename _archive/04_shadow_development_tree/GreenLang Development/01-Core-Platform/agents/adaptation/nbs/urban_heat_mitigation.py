# -*- coding: utf-8 -*-
"""
GL-ADAPT-NBS-006: Urban Heat Island Mitigation Agent
=====================================================

Nature-based urban cooling strategies and heat island mitigation.

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


class HeatRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"


class GreenInfrastructureType(str, Enum):
    URBAN_FOREST = "urban_forest"
    GREEN_ROOF = "green_roof"
    GREEN_WALL = "green_wall"
    URBAN_PARK = "urban_park"
    STREET_TREES = "street_trees"
    RAIN_GARDEN = "rain_garden"
    BIOSWALE = "bioswale"


class UrbanZone(BaseModel):
    zone_id: str = Field(...)
    area_km2: float = Field(..., gt=0)
    population: int = Field(..., ge=0)
    current_green_cover_pct: float = Field(default=15, ge=0, le=100)
    avg_surface_temp_c: float = Field(default=35, ge=-20, le=60)
    impervious_surface_pct: float = Field(default=70, ge=0, le=100)
    building_density_per_km2: float = Field(default=5000, ge=0)


class HeatMitigationPlan(BaseModel):
    zone_id: str = Field(...)
    heat_risk_level: HeatRiskLevel = Field(...)
    temperature_reduction_potential_c: float = Field(...)
    recommended_green_cover_pct: float = Field(...)
    interventions: List[Dict[str, Any]] = Field(...)
    implementation_cost_million: float = Field(...)
    health_benefits_million_annual: float = Field(...)
    energy_savings_million_annual: float = Field(...)
    priority: str = Field(...)


class UrbanHeatInput(BaseModel):
    project_id: str = Field(...)
    city_name: str = Field(...)
    zones: List[UrbanZone] = Field(..., min_length=1)
    climate_scenario: str = Field(default="RCP4.5")
    target_year: int = Field(default=2050, ge=2024, le=2100)


class UrbanHeatOutput(BaseModel):
    project_id: str = Field(...)
    city_name: str = Field(...)
    calculation_date: datetime = Field(default_factory=DeterministicClock.now)
    total_area_km2: float = Field(...)
    total_population: int = Field(...)
    plans: List[HeatMitigationPlan] = Field(...)
    city_avg_temp_reduction_c: float = Field(...)
    total_investment_million: float = Field(...)
    total_annual_benefits_million: float = Field(...)
    provenance_hash: str = Field(...)


class UrbanHeatMitigationAgent(BaseAgent):
    """GL-ADAPT-NBS-006: Urban Heat Island Mitigation Agent"""

    AGENT_ID = "GL-ADAPT-NBS-006"
    AGENT_NAME = "Urban Heat Island Mitigation Agent"
    VERSION = "1.0.0"

    COOLING_EFFECTIVENESS = {
        GreenInfrastructureType.URBAN_FOREST: {"temp_reduction": 3.0, "cost_per_ha": 50000},
        GreenInfrastructureType.GREEN_ROOF: {"temp_reduction": 1.5, "cost_per_ha": 200000},
        GreenInfrastructureType.GREEN_WALL: {"temp_reduction": 1.0, "cost_per_ha": 300000},
        GreenInfrastructureType.URBAN_PARK: {"temp_reduction": 2.5, "cost_per_ha": 100000},
        GreenInfrastructureType.STREET_TREES: {"temp_reduction": 2.0, "cost_per_ha": 80000},
        GreenInfrastructureType.RAIN_GARDEN: {"temp_reduction": 1.0, "cost_per_ha": 60000},
        GreenInfrastructureType.BIOSWALE: {"temp_reduction": 0.8, "cost_per_ha": 40000},
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Urban heat island mitigation",
                version=self.VERSION
            )
        super().__init__(config)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        try:
            agent_input = UrbanHeatInput(**input_data)
            plans = []

            for zone in agent_input.zones:
                plan = self._create_mitigation_plan(zone, agent_input.climate_scenario)
                plans.append(plan)

            total_area = sum(z.area_km2 for z in agent_input.zones)
            total_pop = sum(z.population for z in agent_input.zones)
            total_investment = sum(p.implementation_cost_million for p in plans)
            total_benefits = sum(p.health_benefits_million_annual + p.energy_savings_million_annual for p in plans)

            weighted_temp_reduction = sum(
                p.temperature_reduction_potential_c * z.area_km2
                for p, z in zip(plans, agent_input.zones)
            ) / total_area if total_area > 0 else 0

            provenance_hash = hashlib.sha256(
                json.dumps({"agent": self.AGENT_ID, "ts": DeterministicClock.now().isoformat()}).encode()
            ).hexdigest()

            output = UrbanHeatOutput(
                project_id=agent_input.project_id,
                city_name=agent_input.city_name,
                total_area_km2=total_area,
                total_population=total_pop,
                plans=plans,
                city_avg_temp_reduction_c=round(weighted_temp_reduction, 2),
                total_investment_million=round(total_investment, 2),
                total_annual_benefits_million=round(total_benefits, 2),
                provenance_hash=provenance_hash,
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    def _create_mitigation_plan(self, zone: UrbanZone, scenario: str) -> HeatMitigationPlan:
        heat_score = (
            zone.avg_surface_temp_c * 0.3 +
            zone.impervious_surface_pct * 0.3 +
            (100 - zone.current_green_cover_pct) * 0.2 +
            (zone.building_density_per_km2 / 100) * 0.2
        )

        climate_factor = 1.2 if scenario == "RCP4.5" else 1.4
        heat_score *= climate_factor

        if heat_score >= 80:
            risk_level = HeatRiskLevel.EXTREME
            target_green = 30
        elif heat_score >= 60:
            risk_level = HeatRiskLevel.VERY_HIGH
            target_green = 25
        elif heat_score >= 45:
            risk_level = HeatRiskLevel.HIGH
            target_green = 22
        elif heat_score >= 30:
            risk_level = HeatRiskLevel.MODERATE
            target_green = 20
        else:
            risk_level = HeatRiskLevel.LOW
            target_green = 18

        green_increase_needed = max(0, target_green - zone.current_green_cover_pct)
        area_to_green_ha = (green_increase_needed / 100) * zone.area_km2 * 100

        interventions = self._recommend_interventions(zone, area_to_green_ha)

        total_cost = sum(i["cost_million"] for i in interventions)
        total_temp_reduction = sum(i["temp_reduction_c"] for i in interventions)

        population_density = zone.population / zone.area_km2 if zone.area_km2 > 0 else 0
        health_benefits = (total_temp_reduction * population_density * 50) / 1_000_000
        energy_savings = total_temp_reduction * zone.area_km2 * 0.5

        priority = "critical" if risk_level in (HeatRiskLevel.EXTREME, HeatRiskLevel.VERY_HIGH) else (
            "high" if risk_level == HeatRiskLevel.HIGH else "moderate"
        )

        return HeatMitigationPlan(
            zone_id=zone.zone_id,
            heat_risk_level=risk_level,
            temperature_reduction_potential_c=round(total_temp_reduction, 2),
            recommended_green_cover_pct=target_green,
            interventions=interventions,
            implementation_cost_million=round(total_cost, 2),
            health_benefits_million_annual=round(health_benefits, 3),
            energy_savings_million_annual=round(energy_savings, 3),
            priority=priority,
        )

    def _recommend_interventions(self, zone: UrbanZone, area_ha: float) -> List[Dict[str, Any]]:
        interventions = []

        street_tree_area = min(area_ha * 0.3, zone.area_km2 * 100 * 0.1)
        if street_tree_area > 0:
            eff = self.COOLING_EFFECTIVENESS[GreenInfrastructureType.STREET_TREES]
            interventions.append({
                "type": "street_trees",
                "area_ha": round(street_tree_area, 1),
                "cost_million": round(street_tree_area * eff["cost_per_ha"] / 1_000_000, 3),
                "temp_reduction_c": round(eff["temp_reduction"] * (street_tree_area / 10), 2),
            })

        park_area = area_ha * 0.4
        if park_area > 0:
            eff = self.COOLING_EFFECTIVENESS[GreenInfrastructureType.URBAN_PARK]
            interventions.append({
                "type": "urban_park",
                "area_ha": round(park_area, 1),
                "cost_million": round(park_area * eff["cost_per_ha"] / 1_000_000, 3),
                "temp_reduction_c": round(eff["temp_reduction"] * (park_area / 20), 2),
            })

        green_roof_area = area_ha * 0.2
        if green_roof_area > 0:
            eff = self.COOLING_EFFECTIVENESS[GreenInfrastructureType.GREEN_ROOF]
            interventions.append({
                "type": "green_roof",
                "area_ha": round(green_roof_area, 1),
                "cost_million": round(green_roof_area * eff["cost_per_ha"] / 1_000_000, 3),
                "temp_reduction_c": round(eff["temp_reduction"] * (green_roof_area / 5), 2),
            })

        rain_garden_area = area_ha * 0.1
        if rain_garden_area > 0:
            eff = self.COOLING_EFFECTIVENESS[GreenInfrastructureType.RAIN_GARDEN]
            interventions.append({
                "type": "rain_garden",
                "area_ha": round(rain_garden_area, 1),
                "cost_million": round(rain_garden_area * eff["cost_per_ha"] / 1_000_000, 3),
                "temp_reduction_c": round(eff["temp_reduction"] * (rain_garden_area / 5), 2),
            })

        return interventions
