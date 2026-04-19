# -*- coding: utf-8 -*-
"""
GL-ADAPT-NBS-007: Watershed Protection Agent
=============================================

Watershed adaptation planning for climate resilience.

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


class WatershedRisk(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class LandUseType(str, Enum):
    FOREST = "forest"
    AGRICULTURE = "agriculture"
    URBAN = "urban"
    WETLAND = "wetland"
    GRASSLAND = "grassland"


class WatershedSite(BaseModel):
    watershed_id: str = Field(...)
    area_km2: float = Field(..., gt=0)
    forest_cover_pct: float = Field(default=40, ge=0, le=100)
    wetland_cover_pct: float = Field(default=5, ge=0, le=100)
    impervious_surface_pct: float = Field(default=10, ge=0, le=100)
    avg_slope_pct: float = Field(default=10, ge=0, le=100)
    annual_precipitation_mm: float = Field(default=1000, ge=0)
    population_served: int = Field(default=50000, ge=0)


class WatershedProtectionPlan(BaseModel):
    watershed_id: str = Field(...)
    flood_risk_level: WatershedRisk = Field(...)
    water_quality_risk_level: WatershedRisk = Field(...)
    erosion_risk_level: WatershedRisk = Field(...)
    recommended_forest_cover_pct: float = Field(...)
    recommended_wetland_cover_pct: float = Field(...)
    interventions: List[Dict[str, Any]] = Field(...)
    implementation_cost_million: float = Field(...)
    ecosystem_services_value_million_annual: float = Field(...)


class WatershedInput(BaseModel):
    project_id: str = Field(...)
    watersheds: List[WatershedSite] = Field(..., min_length=1)
    climate_scenario: str = Field(default="RCP4.5")


class WatershedOutput(BaseModel):
    project_id: str = Field(...)
    calculation_date: datetime = Field(default_factory=DeterministicClock.now)
    total_watershed_area_km2: float = Field(...)
    total_population_served: int = Field(...)
    plans: List[WatershedProtectionPlan] = Field(...)
    total_investment_million: float = Field(...)
    total_ecosystem_services_million_annual: float = Field(...)
    provenance_hash: str = Field(...)


class WatershedProtectionAgent(BaseAgent):
    """GL-ADAPT-NBS-007: Watershed Protection Agent"""

    AGENT_ID = "GL-ADAPT-NBS-007"
    AGENT_NAME = "Watershed Protection Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Watershed protection planning",
                version=self.VERSION
            )
        super().__init__(config)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        try:
            agent_input = WatershedInput(**input_data)
            plans = []

            for watershed in agent_input.watersheds:
                plan = self._create_protection_plan(watershed, agent_input.climate_scenario)
                plans.append(plan)

            total_area = sum(w.area_km2 for w in agent_input.watersheds)
            total_pop = sum(w.population_served for w in agent_input.watersheds)
            total_investment = sum(p.implementation_cost_million for p in plans)
            total_services = sum(p.ecosystem_services_value_million_annual for p in plans)

            provenance_hash = hashlib.sha256(
                json.dumps({"agent": self.AGENT_ID, "ts": DeterministicClock.now().isoformat()}).encode()
            ).hexdigest()

            output = WatershedOutput(
                project_id=agent_input.project_id,
                total_watershed_area_km2=total_area,
                total_population_served=total_pop,
                plans=plans,
                total_investment_million=round(total_investment, 2),
                total_ecosystem_services_million_annual=round(total_services, 2),
                provenance_hash=provenance_hash,
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    def _create_protection_plan(self, watershed: WatershedSite, scenario: str) -> WatershedProtectionPlan:
        climate_factor = 1.3 if scenario == "RCP4.5" else 1.5

        flood_score = (
            (100 - watershed.forest_cover_pct) * 0.3 +
            watershed.impervious_surface_pct * 0.3 +
            (100 - watershed.wetland_cover_pct) * 0.2 +
            watershed.avg_slope_pct * 0.2
        ) * climate_factor

        water_quality_score = (
            watershed.impervious_surface_pct * 0.4 +
            (100 - watershed.forest_cover_pct) * 0.3 +
            (100 - watershed.wetland_cover_pct) * 0.3
        )

        erosion_score = (
            (100 - watershed.forest_cover_pct) * 0.4 +
            watershed.avg_slope_pct * 0.4 +
            watershed.impervious_surface_pct * 0.2
        )

        def score_to_risk(score: float) -> WatershedRisk:
            if score >= 70:
                return WatershedRisk.VERY_HIGH
            elif score >= 50:
                return WatershedRisk.HIGH
            elif score >= 30:
                return WatershedRisk.MODERATE
            return WatershedRisk.LOW

        flood_risk = score_to_risk(flood_score)
        water_quality_risk = score_to_risk(water_quality_score)
        erosion_risk = score_to_risk(erosion_score)

        target_forest = min(60, watershed.forest_cover_pct + 15)
        target_wetland = min(15, watershed.wetland_cover_pct + 5)

        interventions = []
        total_cost = 0

        forest_increase_km2 = (target_forest - watershed.forest_cover_pct) / 100 * watershed.area_km2
        if forest_increase_km2 > 0:
            cost = forest_increase_km2 * 100 * 5000 / 1_000_000
            interventions.append({
                "type": "riparian_reforestation",
                "area_km2": round(forest_increase_km2, 2),
                "cost_million": round(cost, 3),
            })
            total_cost += cost

        wetland_increase_km2 = (target_wetland - watershed.wetland_cover_pct) / 100 * watershed.area_km2
        if wetland_increase_km2 > 0:
            cost = wetland_increase_km2 * 100 * 15000 / 1_000_000
            interventions.append({
                "type": "wetland_restoration",
                "area_km2": round(wetland_increase_km2, 2),
                "cost_million": round(cost, 3),
            })
            total_cost += cost

        if erosion_risk in (WatershedRisk.HIGH, WatershedRisk.VERY_HIGH):
            cost = watershed.area_km2 * 0.05
            interventions.append({
                "type": "erosion_control_structures",
                "area_km2": round(watershed.area_km2 * 0.1, 2),
                "cost_million": round(cost, 3),
            })
            total_cost += cost

        ecosystem_services = (
            watershed.area_km2 * (target_forest / 100) * 500 +
            watershed.area_km2 * (target_wetland / 100) * 1500 +
            watershed.population_served * 0.01
        ) / 1_000_000

        return WatershedProtectionPlan(
            watershed_id=watershed.watershed_id,
            flood_risk_level=flood_risk,
            water_quality_risk_level=water_quality_risk,
            erosion_risk_level=erosion_risk,
            recommended_forest_cover_pct=target_forest,
            recommended_wetland_cover_pct=target_wetland,
            interventions=interventions,
            implementation_cost_million=round(total_cost, 2),
            ecosystem_services_value_million_annual=round(ecosystem_services, 3),
        )
