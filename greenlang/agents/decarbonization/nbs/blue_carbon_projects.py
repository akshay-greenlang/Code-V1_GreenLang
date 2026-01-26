# -*- coding: utf-8 -*-
"""
GL-DECARB-NBS-005: Blue Carbon Projects Agent
=============================================

Plans coastal blue carbon projects (mangroves, seagrass, salt marshes).

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


class BlueEcosystem(str, Enum):
    MANGROVE = "mangrove"
    SEAGRASS = "seagrass"
    SALT_MARSH = "salt_marsh"
    TIDAL_MARSH = "tidal_marsh"


class ProjectType(str, Enum):
    CONSERVATION = "conservation"
    RESTORATION = "restoration"
    SUSTAINABLE_USE = "sustainable_use"


SEQUESTRATION_RATES = {
    BlueEcosystem.MANGROVE: 2.5,
    BlueEcosystem.SEAGRASS: 1.4,
    BlueEcosystem.SALT_MARSH: 1.8,
    BlueEcosystem.TIDAL_MARSH: 1.5,
}

CARBON_STOCKS = {
    BlueEcosystem.MANGROVE: 400,
    BlueEcosystem.SEAGRASS: 140,
    BlueEcosystem.SALT_MARSH: 250,
    BlueEcosystem.TIDAL_MARSH: 200,
}


class BlueCarbonSite(BaseModel):
    site_id: str = Field(...)
    area_ha: float = Field(..., gt=0)
    ecosystem_type: BlueEcosystem = Field(...)
    project_type: ProjectType = Field(...)
    degradation_percent: float = Field(default=0, ge=0, le=100)


class BlueCarbonProjectPlan(BaseModel):
    site_id: str = Field(...)
    ecosystem_type: BlueEcosystem = Field(...)
    avoided_emissions_co2e: float = Field(...)
    sequestration_co2e_30yr: float = Field(...)
    total_benefit_co2e: float = Field(...)
    total_cost_usd: float = Field(...)
    cost_per_tonne_co2e: float = Field(...)


class BlueCarbonProjectInput(BaseModel):
    project_id: str = Field(...)
    sites: List[BlueCarbonSite] = Field(..., min_length=1)
    project_duration_years: int = Field(default=30)


class BlueCarbonProjectOutput(BaseModel):
    project_id: str = Field(...)
    calculation_date: datetime = Field(default_factory=DeterministicClock.now)
    total_area_ha: float = Field(...)
    total_benefit_co2e: float = Field(...)
    total_cost_usd: float = Field(...)
    plans: List[BlueCarbonProjectPlan] = Field(...)
    provenance_hash: str = Field(...)
    warnings: List[str] = Field(default_factory=list)


class BlueCarbonProjectsAgent(BaseAgent):
    """GL-DECARB-NBS-005: Blue Carbon Projects Agent"""

    AGENT_ID = "GL-DECARB-NBS-005"
    AGENT_NAME = "Blue Carbon Projects Agent"
    VERSION = "1.0.0"
    CO2_TO_C_RATIO = 44.0 / 12.0

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(name=self.AGENT_NAME, description="Blue carbon projects", version=self.VERSION)
        super().__init__(config)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        try:
            agent_input = BlueCarbonProjectInput(**input_data)
            plans = []

            for site in agent_input.sites:
                plan = self._create_plan(site, agent_input.project_duration_years)
                plans.append(plan)

            total_area = sum(s.area_ha for s in agent_input.sites)
            total_benefit = sum(p.total_benefit_co2e for p in plans)
            total_cost = sum(p.total_cost_usd for p in plans)

            provenance_hash = hashlib.sha256(
                json.dumps({"agent": self.AGENT_ID, "ts": DeterministicClock.now().isoformat()}).encode()
            ).hexdigest()

            output = BlueCarbonProjectOutput(
                project_id=agent_input.project_id,
                total_area_ha=total_area,
                total_benefit_co2e=total_benefit,
                total_cost_usd=total_cost,
                plans=plans,
                provenance_hash=provenance_hash,
                warnings=[]
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    def _create_plan(self, site: BlueCarbonSite, duration: int) -> BlueCarbonProjectPlan:
        eco = site.ecosystem_type
        seq_rate = SEQUESTRATION_RATES.get(eco, 1.5)
        stock = CARBON_STOCKS.get(eco, 200)

        # Sequestration benefit
        if site.project_type == ProjectType.RESTORATION:
            seq_c = seq_rate * site.area_ha * duration * (site.degradation_percent / 100)
        else:
            seq_c = seq_rate * site.area_ha * duration

        seq_co2e = seq_c * self.CO2_TO_C_RATIO

        # Avoided emissions (for conservation)
        avoided = 0.0
        if site.project_type == ProjectType.CONSERVATION:
            # Assume 1% annual loss rate avoided
            avoided = stock * 0.01 * site.area_ha * duration * self.CO2_TO_C_RATIO

        total_benefit = seq_co2e + avoided

        # Costs
        cost_per_ha = {
            ProjectType.CONSERVATION: 500,
            ProjectType.RESTORATION: 5000,
            ProjectType.SUSTAINABLE_USE: 300,
        }.get(site.project_type, 1000)

        total_cost = cost_per_ha * site.area_ha
        cost_per_tonne = total_cost / total_benefit if total_benefit > 0 else 0

        return BlueCarbonProjectPlan(
            site_id=site.site_id,
            ecosystem_type=eco,
            avoided_emissions_co2e=round(avoided, 2),
            sequestration_co2e_30yr=round(seq_co2e, 2),
            total_benefit_co2e=round(total_benefit, 2),
            total_cost_usd=round(total_cost, 2),
            cost_per_tonne_co2e=round(cost_per_tonne, 2)
        )
