# -*- coding: utf-8 -*-
"""
GL-DECARB-NBS-003: Soil Carbon Enhancement Agent
=================================================

Plans soil carbon enhancement strategies for agricultural lands.

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


class EnhancementPractice(str, Enum):
    """Soil carbon enhancement practices."""
    NO_TILL = "no_till"
    REDUCED_TILL = "reduced_till"
    COVER_CROPS = "cover_crops"
    CROP_ROTATION = "crop_rotation"
    ORGANIC_AMENDMENTS = "organic_amendments"
    BIOCHAR = "biochar"
    COMPOST = "compost"
    RESIDUE_RETENTION = "residue_retention"


# Carbon sequestration rates by practice (tonnes C/ha/yr)
PRACTICE_RATES = {
    EnhancementPractice.NO_TILL: 0.5,
    EnhancementPractice.REDUCED_TILL: 0.3,
    EnhancementPractice.COVER_CROPS: 0.7,
    EnhancementPractice.CROP_ROTATION: 0.3,
    EnhancementPractice.ORGANIC_AMENDMENTS: 0.8,
    EnhancementPractice.BIOCHAR: 1.5,
    EnhancementPractice.COMPOST: 0.6,
    EnhancementPractice.RESIDUE_RETENTION: 0.4,
}

PRACTICE_COSTS = {
    EnhancementPractice.NO_TILL: 50,
    EnhancementPractice.REDUCED_TILL: 30,
    EnhancementPractice.COVER_CROPS: 100,
    EnhancementPractice.CROP_ROTATION: 20,
    EnhancementPractice.ORGANIC_AMENDMENTS: 150,
    EnhancementPractice.BIOCHAR: 500,
    EnhancementPractice.COMPOST: 200,
    EnhancementPractice.RESIDUE_RETENTION: 25,
}


class SoilSite(BaseModel):
    """Soil site assessment."""
    site_id: str = Field(...)
    area_ha: float = Field(..., gt=0)
    current_soc_tonnes_ha: float = Field(default=40, ge=0)
    soil_texture: str = Field(default="loam")
    current_practices: List[EnhancementPractice] = Field(default_factory=list)


class SoilEnhancementStrategy(BaseModel):
    """Soil enhancement strategy for a site."""
    site_id: str = Field(...)
    recommended_practices: List[EnhancementPractice] = Field(...)
    annual_sequestration_tonnes_c: float = Field(...)
    total_20yr_sequestration_co2e: float = Field(...)
    annual_cost_usd: float = Field(...)
    total_20yr_cost_usd: float = Field(...)
    cost_per_tonne_co2e: float = Field(...)
    permanence_risk: str = Field(...)


class SoilEnhancementInput(BaseModel):
    """Input for Soil Carbon Enhancement Agent."""
    project_id: str = Field(...)
    sites: List[SoilSite] = Field(..., min_length=1)
    available_practices: List[EnhancementPractice] = Field(
        default_factory=lambda: list(EnhancementPractice)
    )
    project_duration_years: int = Field(default=20)


class SoilEnhancementOutput(BaseModel):
    """Output from Soil Carbon Enhancement Agent."""
    project_id: str = Field(...)
    calculation_date: datetime = Field(default_factory=DeterministicClock.now)
    total_area_ha: float = Field(...)
    total_sequestration_co2e: float = Field(...)
    total_cost_usd: float = Field(...)
    average_cost_per_tonne: float = Field(...)
    strategies: List[SoilEnhancementStrategy] = Field(...)
    provenance_hash: str = Field(...)
    warnings: List[str] = Field(default_factory=list)


class SoilCarbonEnhancementAgent(BaseAgent):
    """
    GL-DECARB-NBS-003: Soil Carbon Enhancement Agent

    Plans soil carbon enhancement strategies with practice recommendations.
    """

    AGENT_ID = "GL-DECARB-NBS-003"
    AGENT_NAME = "Soil Carbon Enhancement Agent"
    VERSION = "1.0.0"
    CO2_TO_C_RATIO = 44.0 / 12.0

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Soil carbon enhancement planning",
                version=self.VERSION,
            )
        super().__init__(config)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute soil carbon planning."""
        try:
            agent_input = SoilEnhancementInput(**input_data)
            strategies = []
            warnings = []

            for site in agent_input.sites:
                strategy = self._create_strategy(
                    site, agent_input.available_practices, agent_input.project_duration_years
                )
                strategies.append(strategy)

            total_area = sum(s.area_ha for s in agent_input.sites)
            total_seq = sum(s.total_20yr_sequestration_co2e for s in strategies)
            total_cost = sum(s.total_20yr_cost_usd for s in strategies)
            avg_cost = total_cost / total_seq if total_seq > 0 else 0

            provenance_hash = hashlib.sha256(
                json.dumps({"agent": self.AGENT_ID, "ts": DeterministicClock.now().isoformat()}).encode()
            ).hexdigest()

            output = SoilEnhancementOutput(
                project_id=agent_input.project_id,
                total_area_ha=total_area,
                total_sequestration_co2e=total_seq,
                total_cost_usd=total_cost,
                average_cost_per_tonne=avg_cost,
                strategies=strategies,
                provenance_hash=provenance_hash,
                warnings=warnings
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            logger.error(f"Soil carbon planning failed: {str(e)}", exc_info=True)
            return AgentResult(success=False, error=str(e))

    def _create_strategy(
        self,
        site: SoilSite,
        available: List[EnhancementPractice],
        duration: int
    ) -> SoilEnhancementStrategy:
        """Create enhancement strategy for a site."""

        # Select top 3 practices not already in use
        new_practices = [p for p in available if p not in site.current_practices]
        # Sort by cost-effectiveness
        sorted_practices = sorted(
            new_practices,
            key=lambda p: PRACTICE_COSTS[p] / PRACTICE_RATES[p]
        )[:3]

        # Calculate combined effect (diminishing returns)
        total_rate = sum(PRACTICE_RATES[p] * (0.8 ** i) for i, p in enumerate(sorted_practices))
        annual_seq_c = total_rate * site.area_ha
        total_seq_c = annual_seq_c * duration
        total_co2e = total_seq_c * self.CO2_TO_C_RATIO

        annual_cost = sum(PRACTICE_COSTS[p] for p in sorted_practices) * site.area_ha
        total_cost = annual_cost * duration
        cost_per_tonne = total_cost / total_co2e if total_co2e > 0 else 0

        # Permanence risk
        if EnhancementPractice.BIOCHAR in sorted_practices:
            permanence = "low"
        elif EnhancementPractice.NO_TILL in sorted_practices:
            permanence = "medium"
        else:
            permanence = "high"

        return SoilEnhancementStrategy(
            site_id=site.site_id,
            recommended_practices=sorted_practices,
            annual_sequestration_tonnes_c=round(annual_seq_c, 2),
            total_20yr_sequestration_co2e=round(total_co2e, 2),
            annual_cost_usd=round(annual_cost, 2),
            total_20yr_cost_usd=round(total_cost, 2),
            cost_per_tonne_co2e=round(cost_per_tonne, 2),
            permanence_risk=permanence
        )
