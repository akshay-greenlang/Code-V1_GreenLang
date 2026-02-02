# -*- coding: utf-8 -*-
"""
GL-DECARB-NBS-008: Regenerative Agriculture Agent
=================================================

Plans regenerative agriculture practices for carbon sequestration.

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


class RegenerativePractice(str, Enum):
    COVER_CROPPING = "cover_cropping"
    NO_TILL = "no_till"
    ROTATIONAL_GRAZING = "rotational_grazing"
    AGROFORESTRY = "agroforestry"
    COMPOSTING = "composting"
    MULCHING = "mulching"
    INTEGRATED_PEST_MGMT = "integrated_pest_management"
    POLYCULTURE = "polyculture"
    SILVOPASTURE = "silvopasture"
    PERENNIAL_CROPS = "perennial_crops"


class FarmType(str, Enum):
    CROPLAND = "cropland"
    PASTURE = "pasture"
    MIXED = "mixed"
    ORCHARD = "orchard"
    VINEYARD = "vineyard"


# Carbon sequestration potential (tonnes CO2e/ha/yr)
PRACTICE_POTENTIALS = {
    RegenerativePractice.COVER_CROPPING: 2.5,
    RegenerativePractice.NO_TILL: 1.8,
    RegenerativePractice.ROTATIONAL_GRAZING: 3.0,
    RegenerativePractice.AGROFORESTRY: 5.0,
    RegenerativePractice.COMPOSTING: 2.0,
    RegenerativePractice.MULCHING: 1.5,
    RegenerativePractice.INTEGRATED_PEST_MGMT: 0.5,
    RegenerativePractice.POLYCULTURE: 1.2,
    RegenerativePractice.SILVOPASTURE: 6.0,
    RegenerativePractice.PERENNIAL_CROPS: 4.0,
}

# Annual costs (USD/ha)
PRACTICE_COSTS = {
    RegenerativePractice.COVER_CROPPING: 100,
    RegenerativePractice.NO_TILL: 50,
    RegenerativePractice.ROTATIONAL_GRAZING: 30,
    RegenerativePractice.AGROFORESTRY: 500,
    RegenerativePractice.COMPOSTING: 200,
    RegenerativePractice.MULCHING: 150,
    RegenerativePractice.INTEGRATED_PEST_MGMT: 80,
    RegenerativePractice.POLYCULTURE: 120,
    RegenerativePractice.SILVOPASTURE: 400,
    RegenerativePractice.PERENNIAL_CROPS: 600,
}


class FarmAssessment(BaseModel):
    farm_id: str = Field(...)
    area_ha: float = Field(..., gt=0)
    farm_type: FarmType = Field(...)
    current_practices: List[RegenerativePractice] = Field(default_factory=list)
    soil_health_score: float = Field(default=0.5, ge=0, le=1)


class RegenerativeAgPlan(BaseModel):
    farm_id: str = Field(...)
    recommended_practices: List[RegenerativePractice] = Field(...)
    annual_sequestration_co2e: float = Field(...)
    total_20yr_sequestration_co2e: float = Field(...)
    annual_cost_usd: float = Field(...)
    total_20yr_cost_usd: float = Field(...)
    cost_per_tonne_co2e: float = Field(...)
    soil_health_improvement: str = Field(...)
    yield_impact: str = Field(...)


class RegenerativeAgInput(BaseModel):
    project_id: str = Field(...)
    farms: List[FarmAssessment] = Field(..., min_length=1)
    project_duration_years: int = Field(default=20)
    max_practices_per_farm: int = Field(default=4)


class RegenerativeAgOutput(BaseModel):
    project_id: str = Field(...)
    calculation_date: datetime = Field(default_factory=DeterministicClock.now)
    total_area_ha: float = Field(...)
    total_sequestration_co2e: float = Field(...)
    total_cost_usd: float = Field(...)
    average_cost_per_tonne: float = Field(...)
    plans: List[RegenerativeAgPlan] = Field(...)
    provenance_hash: str = Field(...)
    warnings: List[str] = Field(default_factory=list)


class RegenerativeAgricultureAgent(BaseAgent):
    """GL-DECARB-NBS-008: Regenerative Agriculture Agent"""

    AGENT_ID = "GL-DECARB-NBS-008"
    AGENT_NAME = "Regenerative Agriculture Agent"
    VERSION = "1.0.0"

    # Recommended practices by farm type
    FARM_PRACTICES = {
        FarmType.CROPLAND: [
            RegenerativePractice.COVER_CROPPING,
            RegenerativePractice.NO_TILL,
            RegenerativePractice.COMPOSTING,
            RegenerativePractice.POLYCULTURE,
        ],
        FarmType.PASTURE: [
            RegenerativePractice.ROTATIONAL_GRAZING,
            RegenerativePractice.SILVOPASTURE,
            RegenerativePractice.MULCHING,
        ],
        FarmType.MIXED: [
            RegenerativePractice.AGROFORESTRY,
            RegenerativePractice.COVER_CROPPING,
            RegenerativePractice.ROTATIONAL_GRAZING,
        ],
        FarmType.ORCHARD: [
            RegenerativePractice.MULCHING,
            RegenerativePractice.COVER_CROPPING,
            RegenerativePractice.COMPOSTING,
        ],
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(name=self.AGENT_NAME, description="Regenerative agriculture planning", version=self.VERSION)
        super().__init__(config)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        try:
            agent_input = RegenerativeAgInput(**input_data)
            plans = []

            for farm in agent_input.farms:
                plan = self._create_plan(
                    farm,
                    agent_input.project_duration_years,
                    agent_input.max_practices_per_farm
                )
                plans.append(plan)

            total_area = sum(f.area_ha for f in agent_input.farms)
            total_seq = sum(p.total_20yr_sequestration_co2e for p in plans)
            total_cost = sum(p.total_20yr_cost_usd for p in plans)
            avg_cost = total_cost / total_seq if total_seq > 0 else 0

            provenance_hash = hashlib.sha256(
                json.dumps({"agent": self.AGENT_ID, "ts": DeterministicClock.now().isoformat()}).encode()
            ).hexdigest()

            output = RegenerativeAgOutput(
                project_id=agent_input.project_id,
                total_area_ha=total_area,
                total_sequestration_co2e=total_seq,
                total_cost_usd=total_cost,
                average_cost_per_tonne=avg_cost,
                plans=plans,
                provenance_hash=provenance_hash,
                warnings=[]
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    def _create_plan(
        self,
        farm: FarmAssessment,
        duration: int,
        max_practices: int
    ) -> RegenerativeAgPlan:
        # Get suitable practices for farm type
        suitable = self.FARM_PRACTICES.get(farm.farm_type, list(RegenerativePractice)[:4])

        # Remove already implemented practices
        new_practices = [p for p in suitable if p not in farm.current_practices]

        # Select top practices by cost-effectiveness
        selected = sorted(
            new_practices,
            key=lambda p: PRACTICE_COSTS[p] / PRACTICE_POTENTIALS[p]
        )[:max_practices]

        # Calculate sequestration (with diminishing returns for multiple practices)
        total_rate = sum(
            PRACTICE_POTENTIALS[p] * (0.85 ** i)
            for i, p in enumerate(selected)
        )

        # Adjust for soil health (better soil = better results)
        adjusted_rate = total_rate * (0.7 + 0.3 * farm.soil_health_score)
        annual_seq = adjusted_rate * farm.area_ha
        total_seq = annual_seq * duration

        # Costs
        annual_cost = sum(PRACTICE_COSTS[p] for p in selected) * farm.area_ha
        total_cost = annual_cost * duration
        cost_per_tonne = total_cost / total_seq if total_seq > 0 else 0

        # Soil health improvement estimate
        if len(selected) >= 3:
            soil_improvement = "significant"
        elif len(selected) >= 2:
            soil_improvement = "moderate"
        else:
            soil_improvement = "minor"

        # Yield impact
        if RegenerativePractice.COVER_CROPPING in selected or RegenerativePractice.COMPOSTING in selected:
            yield_impact = "positive (+5-15%)"
        else:
            yield_impact = "neutral to positive"

        return RegenerativeAgPlan(
            farm_id=farm.farm_id,
            recommended_practices=selected,
            annual_sequestration_co2e=round(annual_seq, 2),
            total_20yr_sequestration_co2e=round(total_seq, 2),
            annual_cost_usd=round(annual_cost, 2),
            total_20yr_cost_usd=round(total_cost, 2),
            cost_per_tonne_co2e=round(cost_per_tonne, 2),
            soil_health_improvement=soil_improvement,
            yield_impact=yield_impact
        )
