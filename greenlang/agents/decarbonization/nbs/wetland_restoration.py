# -*- coding: utf-8 -*-
"""
GL-DECARB-NBS-004: Wetland Restoration Agent
============================================

Plans wetland and peatland restoration projects.

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


class WetlandType(str, Enum):
    PEATLAND = "peatland"
    FRESHWATER_MARSH = "freshwater_marsh"
    COASTAL_MARSH = "coastal_marsh"
    BOG = "bog"
    FEN = "fen"


class RestorationAction(str, Enum):
    REWETTING = "rewetting"
    DAM_REMOVAL = "dam_removal"
    DRAIN_BLOCKING = "drain_blocking"
    REVEGETATION = "revegetation"
    INVASIVE_REMOVAL = "invasive_removal"


class WetlandSite(BaseModel):
    site_id: str = Field(...)
    area_ha: float = Field(..., gt=0)
    wetland_type: WetlandType = Field(...)
    current_water_table_cm: float = Field(default=-50)
    peat_depth_m: Optional[float] = Field(None, ge=0)


class WetlandRestorationPlan(BaseModel):
    site_id: str = Field(...)
    recommended_actions: List[RestorationAction] = Field(...)
    target_water_table_cm: float = Field(...)
    avoided_emissions_co2e_yr: float = Field(...)
    total_30yr_benefit_co2e: float = Field(...)
    total_cost_usd: float = Field(...)
    cost_per_tonne_co2e: float = Field(...)


class WetlandRestorationInput(BaseModel):
    project_id: str = Field(...)
    sites: List[WetlandSite] = Field(..., min_length=1)
    project_duration_years: int = Field(default=30)


class WetlandRestorationOutput(BaseModel):
    project_id: str = Field(...)
    calculation_date: datetime = Field(default_factory=DeterministicClock.now)
    total_area_ha: float = Field(...)
    total_avoided_emissions_co2e: float = Field(...)
    total_cost_usd: float = Field(...)
    plans: List[WetlandRestorationPlan] = Field(...)
    provenance_hash: str = Field(...)
    warnings: List[str] = Field(default_factory=list)


class WetlandRestorationAgent(BaseAgent):
    """GL-DECARB-NBS-004: Wetland Restoration Agent"""

    AGENT_ID = "GL-DECARB-NBS-004"
    AGENT_NAME = "Wetland Restoration Agent"
    VERSION = "1.0.0"

    # Emission factors for drained peatlands (t CO2/ha/yr)
    DRAINED_PEAT_EF = {
        WetlandType.PEATLAND: 10.0,
        WetlandType.BOG: 8.0,
        WetlandType.FEN: 7.0,
        WetlandType.FRESHWATER_MARSH: 3.0,
        WetlandType.COASTAL_MARSH: 2.0,
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(name=self.AGENT_NAME, description="Wetland restoration", version=self.VERSION)
        super().__init__(config)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        try:
            agent_input = WetlandRestorationInput(**input_data)
            plans = []

            for site in agent_input.sites:
                plan = self._create_plan(site, agent_input.project_duration_years)
                plans.append(plan)

            total_area = sum(s.area_ha for s in agent_input.sites)
            total_benefit = sum(p.total_30yr_benefit_co2e for p in plans)
            total_cost = sum(p.total_cost_usd for p in plans)

            provenance_hash = hashlib.sha256(
                json.dumps({"agent": self.AGENT_ID, "ts": DeterministicClock.now().isoformat()}).encode()
            ).hexdigest()

            output = WetlandRestorationOutput(
                project_id=agent_input.project_id,
                total_area_ha=total_area,
                total_avoided_emissions_co2e=total_benefit,
                total_cost_usd=total_cost,
                plans=plans,
                provenance_hash=provenance_hash,
                warnings=[]
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    def _create_plan(self, site: WetlandSite, duration: int) -> WetlandRestorationPlan:
        actions = [RestorationAction.REWETTING]
        if site.current_water_table_cm < -30:
            actions.append(RestorationAction.DRAIN_BLOCKING)

        # Emission reduction calculation
        baseline_ef = self.DRAINED_PEAT_EF.get(site.wetland_type, 5.0)
        # Rewetting reduces emissions by 80%
        avoided_per_yr = baseline_ef * 0.8 * site.area_ha
        total_benefit = avoided_per_yr * duration

        # Costs
        cost_per_ha = 3000 if site.wetland_type == WetlandType.PEATLAND else 1500
        total_cost = cost_per_ha * site.area_ha
        cost_per_tonne = total_cost / total_benefit if total_benefit > 0 else 0

        return WetlandRestorationPlan(
            site_id=site.site_id,
            recommended_actions=actions,
            target_water_table_cm=-10,
            avoided_emissions_co2e_yr=round(avoided_per_yr, 2),
            total_30yr_benefit_co2e=round(total_benefit, 2),
            total_cost_usd=round(total_cost, 2),
            cost_per_tonne_co2e=round(cost_per_tonne, 2)
        )
