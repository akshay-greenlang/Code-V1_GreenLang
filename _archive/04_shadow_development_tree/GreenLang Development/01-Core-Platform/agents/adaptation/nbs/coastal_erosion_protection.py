# -*- coding: utf-8 -*-
"""
GL-ADAPT-NBS-005: Coastal Erosion Protection Agent
===================================================

Nature-based coastal defense and erosion protection assessment.

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


class ErosionRisk(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class CoastalHabitatType(str, Enum):
    MANGROVE = "mangrove"
    SALT_MARSH = "salt_marsh"
    SEAGRASS = "seagrass"
    CORAL_REEF = "coral_reef"
    DUNE_SYSTEM = "dune_system"
    OYSTER_REEF = "oyster_reef"


class CoastalSite(BaseModel):
    site_id: str = Field(...)
    coastline_km: float = Field(..., gt=0)
    habitat_type: CoastalHabitatType = Field(...)
    habitat_condition_pct: float = Field(default=70, ge=0, le=100)
    sea_level_rise_exposure_m: float = Field(default=0.5, ge=0)
    storm_frequency_per_year: float = Field(default=2, ge=0)
    current_erosion_rate_m_year: float = Field(default=0.5, ge=0)
    protected_assets_value_million: float = Field(default=10, ge=0)


class CoastalProtectionPlan(BaseModel):
    site_id: str = Field(...)
    erosion_risk_level: ErosionRisk = Field(...)
    wave_attenuation_pct: float = Field(..., ge=0, le=100)
    sediment_stabilization_score: float = Field(..., ge=0, le=100)
    restoration_area_ha: float = Field(...)
    restoration_cost_million: float = Field(...)
    protected_value_million: float = Field(...)
    benefit_cost_ratio: float = Field(...)
    recommended_interventions: List[str] = Field(...)


class CoastalProtectionInput(BaseModel):
    project_id: str = Field(...)
    sites: List[CoastalSite] = Field(..., min_length=1)
    climate_scenario: str = Field(default="RCP4.5")
    planning_horizon_years: int = Field(default=30, ge=10, le=100)


class CoastalProtectionOutput(BaseModel):
    project_id: str = Field(...)
    calculation_date: datetime = Field(default_factory=DeterministicClock.now)
    total_coastline_km: float = Field(...)
    high_risk_coastline_km: float = Field(...)
    plans: List[CoastalProtectionPlan] = Field(...)
    total_restoration_cost_million: float = Field(...)
    total_protected_value_million: float = Field(...)
    portfolio_bcr: float = Field(...)
    provenance_hash: str = Field(...)


class CoastalErosionProtectionAgent(BaseAgent):
    """GL-ADAPT-NBS-005: Coastal Erosion Protection Agent"""

    AGENT_ID = "GL-ADAPT-NBS-005"
    AGENT_NAME = "Coastal Erosion Protection Agent"
    VERSION = "1.0.0"

    HABITAT_EFFECTIVENESS = {
        CoastalHabitatType.MANGROVE: {"wave_attenuation": 0.7, "sediment": 0.8},
        CoastalHabitatType.SALT_MARSH: {"wave_attenuation": 0.5, "sediment": 0.7},
        CoastalHabitatType.SEAGRASS: {"wave_attenuation": 0.3, "sediment": 0.6},
        CoastalHabitatType.CORAL_REEF: {"wave_attenuation": 0.8, "sediment": 0.4},
        CoastalHabitatType.DUNE_SYSTEM: {"wave_attenuation": 0.6, "sediment": 0.9},
        CoastalHabitatType.OYSTER_REEF: {"wave_attenuation": 0.5, "sediment": 0.5},
    }

    RESTORATION_COST_PER_HA = {
        CoastalHabitatType.MANGROVE: 15000,
        CoastalHabitatType.SALT_MARSH: 25000,
        CoastalHabitatType.SEAGRASS: 50000,
        CoastalHabitatType.CORAL_REEF: 150000,
        CoastalHabitatType.DUNE_SYSTEM: 20000,
        CoastalHabitatType.OYSTER_REEF: 80000,
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Nature-based coastal protection",
                version=self.VERSION
            )
        super().__init__(config)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        try:
            agent_input = CoastalProtectionInput(**input_data)
            plans = []
            high_risk_km = 0.0

            for site in agent_input.sites:
                plan = self._create_protection_plan(
                    site, agent_input.climate_scenario, agent_input.planning_horizon_years
                )
                plans.append(plan)
                if plan.erosion_risk_level in (ErosionRisk.HIGH, ErosionRisk.VERY_HIGH):
                    high_risk_km += site.coastline_km

            total_coastline = sum(s.coastline_km for s in agent_input.sites)
            total_cost = sum(p.restoration_cost_million for p in plans)
            total_protected = sum(p.protected_value_million for p in plans)
            portfolio_bcr = total_protected / total_cost if total_cost > 0 else 0

            provenance_hash = hashlib.sha256(
                json.dumps({"agent": self.AGENT_ID, "ts": DeterministicClock.now().isoformat()}).encode()
            ).hexdigest()

            output = CoastalProtectionOutput(
                project_id=agent_input.project_id,
                total_coastline_km=total_coastline,
                high_risk_coastline_km=high_risk_km,
                plans=plans,
                total_restoration_cost_million=round(total_cost, 2),
                total_protected_value_million=round(total_protected, 2),
                portfolio_bcr=round(portfolio_bcr, 2),
                provenance_hash=provenance_hash,
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    def _create_protection_plan(
        self, site: CoastalSite, scenario: str, horizon: int
    ) -> CoastalProtectionPlan:
        effectiveness = self.HABITAT_EFFECTIVENESS.get(site.habitat_type, {"wave_attenuation": 0.5, "sediment": 0.5})
        condition_factor = site.habitat_condition_pct / 100

        wave_attenuation = effectiveness["wave_attenuation"] * condition_factor * 100
        sediment_score = effectiveness["sediment"] * condition_factor * 100

        erosion_risk_score = (
            site.current_erosion_rate_m_year * 20 +
            site.sea_level_rise_exposure_m * 30 +
            site.storm_frequency_per_year * 10 +
            (100 - site.habitat_condition_pct) * 0.4
        )
        climate_factor = 1.3 if scenario == "RCP4.5" else 1.6
        erosion_risk_score *= climate_factor

        if erosion_risk_score >= 80:
            risk_level = ErosionRisk.VERY_HIGH
        elif erosion_risk_score >= 60:
            risk_level = ErosionRisk.HIGH
        elif erosion_risk_score >= 40:
            risk_level = ErosionRisk.MODERATE
        elif erosion_risk_score >= 20:
            risk_level = ErosionRisk.LOW
        else:
            risk_level = ErosionRisk.VERY_LOW

        restoration_ha = site.coastline_km * 0.5 * (1 - condition_factor + 0.2)
        cost_per_ha = self.RESTORATION_COST_PER_HA.get(site.habitat_type, 50000)
        restoration_cost = (restoration_ha * cost_per_ha) / 1_000_000

        protected_value = site.protected_assets_value_million * (wave_attenuation / 100)
        bcr = protected_value / restoration_cost if restoration_cost > 0 else 0

        interventions = self._get_interventions(site.habitat_type, risk_level)

        return CoastalProtectionPlan(
            site_id=site.site_id,
            erosion_risk_level=risk_level,
            wave_attenuation_pct=round(wave_attenuation, 1),
            sediment_stabilization_score=round(sediment_score, 1),
            restoration_area_ha=round(restoration_ha, 1),
            restoration_cost_million=round(restoration_cost, 3),
            protected_value_million=round(protected_value, 2),
            benefit_cost_ratio=round(bcr, 2),
            recommended_interventions=interventions,
        )

    def _get_interventions(self, habitat: CoastalHabitatType, risk: ErosionRisk) -> List[str]:
        base_interventions = {
            CoastalHabitatType.MANGROVE: ["Mangrove restoration", "Hydrological restoration"],
            CoastalHabitatType.SALT_MARSH: ["Marsh creation", "Living shorelines"],
            CoastalHabitatType.SEAGRASS: ["Seagrass transplanting", "Water quality improvement"],
            CoastalHabitatType.CORAL_REEF: ["Coral gardening", "Artificial reef structures"],
            CoastalHabitatType.DUNE_SYSTEM: ["Dune revegetation", "Sand fencing"],
            CoastalHabitatType.OYSTER_REEF: ["Oyster reef restoration", "Shell recycling"],
        }

        interventions = base_interventions.get(habitat, ["Habitat restoration"])

        if risk in (ErosionRisk.HIGH, ErosionRisk.VERY_HIGH):
            interventions.extend(["Emergency stabilization", "Hybrid green-gray infrastructure"])

        return interventions
