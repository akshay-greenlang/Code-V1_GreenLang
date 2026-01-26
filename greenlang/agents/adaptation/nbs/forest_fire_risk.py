# -*- coding: utf-8 -*-
"""
GL-ADAPT-NBS-001: Forest Fire Risk Agent
========================================

Assesses wildfire risk for forest and land areas under climate change.

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


class FireRiskLevel(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"


class VegetationType(str, Enum):
    CONIFEROUS = "coniferous"
    DECIDUOUS = "deciduous"
    MIXED = "mixed"
    SHRUBLAND = "shrubland"
    GRASSLAND = "grassland"
    SAVANNA = "savanna"


class FireSite(BaseModel):
    site_id: str = Field(...)
    area_ha: float = Field(..., gt=0)
    vegetation_type: VegetationType = Field(...)
    fuel_load_tonnes_ha: float = Field(default=25, ge=0)
    slope_percent: float = Field(default=10, ge=0, le=100)
    historical_fire_frequency_per_decade: float = Field(default=0.5, ge=0)
    annual_precipitation_mm: float = Field(default=800, ge=0)
    summer_temp_avg_c: float = Field(default=25, ge=-20, le=50)


class FireRiskAssessment(BaseModel):
    site_id: str = Field(...)
    current_risk_level: FireRiskLevel = Field(...)
    future_risk_level_2050: FireRiskLevel = Field(...)
    fire_weather_index: float = Field(..., ge=0, le=100)
    ignition_probability: float = Field(..., ge=0, le=1)
    spread_potential_score: float = Field(..., ge=0, le=100)
    adaptation_priority: str = Field(...)
    recommended_actions: List[str] = Field(...)


class ForestFireRiskInput(BaseModel):
    project_id: str = Field(...)
    sites: List[FireSite] = Field(..., min_length=1)
    climate_scenario: str = Field(default="RCP4.5")


class ForestFireRiskOutput(BaseModel):
    project_id: str = Field(...)
    calculation_date: datetime = Field(default_factory=DeterministicClock.now)
    total_area_ha: float = Field(...)
    high_risk_area_ha: float = Field(...)
    assessments: List[FireRiskAssessment] = Field(...)
    overall_risk_level: FireRiskLevel = Field(...)
    provenance_hash: str = Field(...)
    warnings: List[str] = Field(default_factory=list)


class ForestFireRiskAgent(BaseAgent):
    """GL-ADAPT-NBS-001: Forest Fire Risk Agent"""

    AGENT_ID = "GL-ADAPT-NBS-001"
    AGENT_NAME = "Forest Fire Risk Agent"
    VERSION = "1.0.0"

    # Fire Weather Index calculation factors
    VEGETATION_FUEL_FACTOR = {
        VegetationType.CONIFEROUS: 1.3,
        VegetationType.DECIDUOUS: 0.8,
        VegetationType.MIXED: 1.0,
        VegetationType.SHRUBLAND: 1.4,
        VegetationType.GRASSLAND: 1.2,
        VegetationType.SAVANNA: 1.1,
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(name=self.AGENT_NAME, description="Forest fire risk assessment", version=self.VERSION)
        super().__init__(config)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        try:
            agent_input = ForestFireRiskInput(**input_data)
            assessments = []
            high_risk_area = 0.0

            for site in agent_input.sites:
                assessment = self._assess_site(site, agent_input.climate_scenario)
                assessments.append(assessment)
                if assessment.current_risk_level in (FireRiskLevel.HIGH, FireRiskLevel.VERY_HIGH, FireRiskLevel.EXTREME):
                    high_risk_area += site.area_ha

            total_area = sum(s.area_ha for s in agent_input.sites)

            # Determine overall risk
            risk_counts = {level: 0 for level in FireRiskLevel}
            for a in assessments:
                risk_counts[a.current_risk_level] += 1

            if risk_counts[FireRiskLevel.EXTREME] > 0:
                overall = FireRiskLevel.EXTREME
            elif risk_counts[FireRiskLevel.VERY_HIGH] > len(assessments) * 0.3:
                overall = FireRiskLevel.VERY_HIGH
            elif risk_counts[FireRiskLevel.HIGH] > len(assessments) * 0.3:
                overall = FireRiskLevel.HIGH
            else:
                overall = FireRiskLevel.MODERATE

            provenance_hash = hashlib.sha256(
                json.dumps({"agent": self.AGENT_ID, "ts": DeterministicClock.now().isoformat()}).encode()
            ).hexdigest()

            output = ForestFireRiskOutput(
                project_id=agent_input.project_id,
                total_area_ha=total_area,
                high_risk_area_ha=high_risk_area,
                assessments=assessments,
                overall_risk_level=overall,
                provenance_hash=provenance_hash,
                warnings=[]
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    def _assess_site(self, site: FireSite, scenario: str) -> FireRiskAssessment:
        # Fire Weather Index calculation (simplified)
        fuel_factor = self.VEGETATION_FUEL_FACTOR.get(site.vegetation_type, 1.0)
        drought_factor = max(0, 1 - (site.annual_precipitation_mm / 1500))
        temp_factor = max(0, (site.summer_temp_avg_c - 15) / 30)
        slope_factor = site.slope_percent / 100

        fwi = (
            (site.fuel_load_tonnes_ha / 50) * fuel_factor * 0.3 +
            drought_factor * 0.3 +
            temp_factor * 0.25 +
            slope_factor * 0.15
        ) * 100

        fwi = min(100, max(0, fwi))

        # Current risk level
        if fwi >= 80:
            current_risk = FireRiskLevel.EXTREME
        elif fwi >= 65:
            current_risk = FireRiskLevel.VERY_HIGH
        elif fwi >= 50:
            current_risk = FireRiskLevel.HIGH
        elif fwi >= 35:
            current_risk = FireRiskLevel.MODERATE
        elif fwi >= 20:
            current_risk = FireRiskLevel.LOW
        else:
            current_risk = FireRiskLevel.VERY_LOW

        # Future risk (climate change increases by ~20% under RCP4.5)
        future_fwi = fwi * (1.2 if scenario == "RCP4.5" else 1.4)
        if future_fwi >= 80:
            future_risk = FireRiskLevel.EXTREME
        elif future_fwi >= 65:
            future_risk = FireRiskLevel.VERY_HIGH
        elif future_fwi >= 50:
            future_risk = FireRiskLevel.HIGH
        else:
            future_risk = FireRiskLevel.MODERATE

        # Ignition probability
        ignition_prob = (site.historical_fire_frequency_per_decade / 10) * (1 + drought_factor)
        ignition_prob = min(1, max(0, ignition_prob))

        # Spread potential
        spread_score = (fuel_factor * 30 + slope_factor * 40 + drought_factor * 30)

        # Priority and recommendations
        if current_risk in (FireRiskLevel.VERY_HIGH, FireRiskLevel.EXTREME):
            priority = "critical"
            actions = ["Fuel management", "Firebreaks", "Early warning system", "Evacuation planning"]
        elif current_risk == FireRiskLevel.HIGH:
            priority = "high"
            actions = ["Fuel reduction", "Fire-resistant species", "Monitoring"]
        else:
            priority = "moderate"
            actions = ["Routine monitoring", "Community awareness"]

        return FireRiskAssessment(
            site_id=site.site_id,
            current_risk_level=current_risk,
            future_risk_level_2050=future_risk,
            fire_weather_index=round(fwi, 1),
            ignition_probability=round(ignition_prob, 3),
            spread_potential_score=round(spread_score, 1),
            adaptation_priority=priority,
            recommended_actions=actions
        )
