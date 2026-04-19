# -*- coding: utf-8 -*-
"""
GL-DECARB-NBS-006: Avoided Deforestation (REDD+) Agent
======================================================

Plans REDD+ (Reducing Emissions from Deforestation and Degradation) projects.

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


class ForestType(str, Enum):
    TROPICAL_RAINFOREST = "tropical_rainforest"
    TROPICAL_DRY = "tropical_dry"
    SUBTROPICAL = "subtropical"
    TEMPERATE = "temperate"
    BOREAL = "boreal"


class DeforestationDriver(str, Enum):
    AGRICULTURE_EXPANSION = "agriculture_expansion"
    LOGGING = "logging"
    MINING = "mining"
    INFRASTRUCTURE = "infrastructure"
    FUELWOOD = "fuelwood"


CARBON_STOCKS_FOREST = {
    ForestType.TROPICAL_RAINFOREST: 300,
    ForestType.TROPICAL_DRY: 130,
    ForestType.SUBTROPICAL: 180,
    ForestType.TEMPERATE: 150,
    ForestType.BOREAL: 80,
}

BASELINE_DEFORESTATION_RATES = {
    ForestType.TROPICAL_RAINFOREST: 0.015,
    ForestType.TROPICAL_DRY: 0.02,
    ForestType.SUBTROPICAL: 0.01,
    ForestType.TEMPERATE: 0.005,
    ForestType.BOREAL: 0.003,
}


class REDDPlusSite(BaseModel):
    site_id: str = Field(...)
    area_ha: float = Field(..., gt=0)
    forest_type: ForestType = Field(...)
    historical_deforestation_rate: Optional[float] = Field(None, ge=0, le=0.5)
    primary_driver: DeforestationDriver = Field(...)


class REDDPlusPlan(BaseModel):
    site_id: str = Field(...)
    baseline_deforestation_rate: float = Field(...)
    projected_avoided_area_ha: float = Field(...)
    avoided_emissions_co2e_30yr: float = Field(...)
    annual_avoided_emissions_co2e: float = Field(...)
    total_project_cost_usd: float = Field(...)
    cost_per_tonne_co2e: float = Field(...)
    leakage_discount_percent: float = Field(...)


class REDDPlusInput(BaseModel):
    project_id: str = Field(...)
    sites: List[REDDPlusSite] = Field(..., min_length=1)
    project_duration_years: int = Field(default=30)
    leakage_discount: float = Field(default=0.15, ge=0, le=0.5)


class REDDPlusOutput(BaseModel):
    project_id: str = Field(...)
    calculation_date: datetime = Field(default_factory=DeterministicClock.now)
    total_area_ha: float = Field(...)
    total_avoided_emissions_co2e: float = Field(...)
    total_cost_usd: float = Field(...)
    average_cost_per_tonne: float = Field(...)
    plans: List[REDDPlusPlan] = Field(...)
    provenance_hash: str = Field(...)
    warnings: List[str] = Field(default_factory=list)


class AvoidedDeforestationAgent(BaseAgent):
    """GL-DECARB-NBS-006: Avoided Deforestation (REDD+) Agent"""

    AGENT_ID = "GL-DECARB-NBS-006"
    AGENT_NAME = "Avoided Deforestation Agent"
    VERSION = "1.0.0"
    CO2_TO_C_RATIO = 44.0 / 12.0

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(name=self.AGENT_NAME, description="REDD+ project planning", version=self.VERSION)
        super().__init__(config)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        try:
            agent_input = REDDPlusInput(**input_data)
            plans = []

            for site in agent_input.sites:
                plan = self._create_plan(site, agent_input.project_duration_years, agent_input.leakage_discount)
                plans.append(plan)

            total_area = sum(s.area_ha for s in agent_input.sites)
            total_avoided = sum(p.avoided_emissions_co2e_30yr for p in plans)
            total_cost = sum(p.total_project_cost_usd for p in plans)
            avg_cost = total_cost / total_avoided if total_avoided > 0 else 0

            provenance_hash = hashlib.sha256(
                json.dumps({"agent": self.AGENT_ID, "ts": DeterministicClock.now().isoformat()}).encode()
            ).hexdigest()

            output = REDDPlusOutput(
                project_id=agent_input.project_id,
                total_area_ha=total_area,
                total_avoided_emissions_co2e=total_avoided,
                total_cost_usd=total_cost,
                average_cost_per_tonne=avg_cost,
                plans=plans,
                provenance_hash=provenance_hash,
                warnings=[]
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    def _create_plan(self, site: REDDPlusSite, duration: int, leakage: float) -> REDDPlusPlan:
        # Deforestation rate
        rate = site.historical_deforestation_rate or BASELINE_DEFORESTATION_RATES.get(site.forest_type, 0.01)

        # Carbon stock
        stock = CARBON_STOCKS_FOREST.get(site.forest_type, 150)

        # Avoided area over project duration
        avoided_area = site.area_ha * (1 - (1 - rate) ** duration)

        # Avoided emissions (before leakage discount)
        avoided_c = avoided_area * stock
        avoided_co2e_gross = avoided_c * self.CO2_TO_C_RATIO

        # Apply leakage discount
        avoided_co2e_net = avoided_co2e_gross * (1 - leakage)
        annual_avoided = avoided_co2e_net / duration

        # Costs (including community engagement, monitoring, enforcement)
        cost_per_ha_yr = 15
        total_cost = cost_per_ha_yr * site.area_ha * duration
        cost_per_tonne = total_cost / avoided_co2e_net if avoided_co2e_net > 0 else 0

        return REDDPlusPlan(
            site_id=site.site_id,
            baseline_deforestation_rate=rate,
            projected_avoided_area_ha=round(avoided_area, 2),
            avoided_emissions_co2e_30yr=round(avoided_co2e_net, 2),
            annual_avoided_emissions_co2e=round(annual_avoided, 2),
            total_project_cost_usd=round(total_cost, 2),
            cost_per_tonne_co2e=round(cost_per_tonne, 2),
            leakage_discount_percent=leakage * 100
        )
