# -*- coding: utf-8 -*-
"""
GL-DECARB-NBS-001: Afforestation Planner Agent
==============================================

Plans and optimizes afforestation (tree planting on non-forest land) projects
for carbon sequestration and co-benefits.

Capabilities:
    - Site suitability assessment
    - Species selection optimization
    - Planting density recommendations
    - Carbon sequestration projections
    - Cost-benefit analysis
    - Co-benefits quantification

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism.clock import DeterministicClock

logger = logging.getLogger(__name__)


class ClimateZone(str, Enum):
    """Climate zones for species selection."""
    TROPICAL = "tropical"
    SUBTROPICAL = "subtropical"
    TEMPERATE = "temperate"
    BOREAL = "boreal"
    ARID = "arid"
    MEDITERRANEAN = "mediterranean"


class LandType(str, Enum):
    """Land types for afforestation."""
    DEGRADED_AGRICULTURAL = "degraded_agricultural"
    ABANDONED_CROPLAND = "abandoned_cropland"
    GRASSLAND = "grassland"
    SHRUBLAND = "shrubland"
    BARREN = "barren"
    MINING_RECLAMATION = "mining_reclamation"


class PlantingObjective(str, Enum):
    """Primary planting objectives."""
    CARBON_SEQUESTRATION = "carbon_sequestration"
    TIMBER_PRODUCTION = "timber_production"
    BIODIVERSITY = "biodiversity"
    WATERSHED_PROTECTION = "watershed_protection"
    EROSION_CONTROL = "erosion_control"
    AGROFORESTRY = "agroforestry"


# Default carbon sequestration rates (tonnes C/ha/yr)
SEQUESTRATION_RATES = {
    ClimateZone.TROPICAL: {"fast_growing": 8.0, "native_mixed": 5.0, "slow_growing": 3.0},
    ClimateZone.SUBTROPICAL: {"fast_growing": 6.0, "native_mixed": 4.0, "slow_growing": 2.5},
    ClimateZone.TEMPERATE: {"fast_growing": 4.0, "native_mixed": 3.0, "slow_growing": 2.0},
    ClimateZone.BOREAL: {"fast_growing": 2.0, "native_mixed": 1.5, "slow_growing": 1.0},
    ClimateZone.ARID: {"fast_growing": 1.5, "native_mixed": 1.0, "slow_growing": 0.5},
}

# Planting costs (USD/ha)
PLANTING_COSTS = {
    "site_preparation": {"low": 200, "medium": 500, "high": 1000},
    "seedlings": {"low": 300, "medium": 600, "high": 1200},
    "planting_labor": {"low": 200, "medium": 400, "high": 800},
    "maintenance_annual": {"low": 50, "medium": 100, "high": 200},
}


class SiteAssessment(BaseModel):
    """Site assessment input."""
    site_id: str = Field(...)
    area_ha: float = Field(..., gt=0)
    climate_zone: ClimateZone = Field(...)
    land_type: LandType = Field(...)
    annual_rainfall_mm: Optional[float] = Field(None, ge=0)
    soil_quality_score: float = Field(default=0.5, ge=0, le=1)
    slope_percent: Optional[float] = Field(None, ge=0, le=100)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)


class AfforestationPlan(BaseModel):
    """Afforestation plan output."""
    site_id: str = Field(...)
    recommended_species_mix: str = Field(...)
    planting_density_stems_ha: int = Field(...)

    # Carbon projections
    annual_sequestration_tonnes_c: float = Field(...)
    total_sequestration_30yr_tonnes_c: float = Field(...)
    total_co2e_30yr_tonnes: float = Field(...)

    # Costs
    establishment_cost_usd: float = Field(...)
    annual_maintenance_cost_usd: float = Field(...)
    total_30yr_cost_usd: float = Field(...)
    cost_per_tonne_co2e_usd: float = Field(...)

    # Timeline
    planting_start_recommended: str = Field(...)
    years_to_canopy_closure: int = Field(...)

    # Co-benefits
    biodiversity_score: float = Field(..., ge=0, le=100)
    watershed_benefit_score: float = Field(..., ge=0, le=100)


class AfforestationInput(BaseModel):
    """Input for Afforestation Planner Agent."""
    project_id: str = Field(...)
    sites: List[SiteAssessment] = Field(..., min_length=1)
    primary_objective: PlantingObjective = Field(default=PlantingObjective.CARBON_SEQUESTRATION)
    budget_usd: Optional[float] = Field(None, ge=0)
    target_co2e_tonnes: Optional[float] = Field(None, ge=0)
    project_duration_years: int = Field(default=30, ge=1)
    species_preference: Optional[str] = Field(None)


class AfforestationOutput(BaseModel):
    """Output from Afforestation Planner Agent."""
    project_id: str = Field(...)
    calculation_date: datetime = Field(default_factory=DeterministicClock.now)

    # Summary
    total_area_ha: float = Field(...)
    total_30yr_sequestration_tonnes_co2e: float = Field(...)
    total_project_cost_usd: float = Field(...)
    average_cost_per_tonne_co2e: float = Field(...)

    # Plans
    site_plans: List[AfforestationPlan] = Field(...)

    # Feasibility
    meets_budget: Optional[bool] = Field(None)
    meets_carbon_target: Optional[bool] = Field(None)

    provenance_hash: str = Field(...)
    warnings: List[str] = Field(default_factory=list)


class AfforestationPlannerAgent(BaseAgent):
    """
    GL-DECARB-NBS-001: Afforestation Planner Agent

    Plans afforestation projects with optimized species selection,
    cost projections, and carbon sequestration estimates.
    """

    AGENT_ID = "GL-DECARB-NBS-001"
    AGENT_NAME = "Afforestation Planner Agent"
    VERSION = "1.0.0"

    CO2_TO_C_RATIO = 44.0 / 12.0

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Afforestation project planning and optimization",
                version=self.VERSION,
            )
        super().__init__(config)
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute afforestation planning."""
        warnings: List[str] = []

        try:
            agent_input = AfforestationInput(**input_data)

            total_area = sum(s.area_ha for s in agent_input.sites)
            site_plans: List[AfforestationPlan] = []

            for site in agent_input.sites:
                plan = self._create_site_plan(
                    site=site,
                    objective=agent_input.primary_objective,
                    duration_years=agent_input.project_duration_years,
                    species_preference=agent_input.species_preference,
                    warnings=warnings
                )
                site_plans.append(plan)

            # Aggregates
            total_co2e = sum(p.total_co2e_30yr_tonnes for p in site_plans)
            total_cost = sum(p.total_30yr_cost_usd for p in site_plans)
            avg_cost_per_tonne = total_cost / total_co2e if total_co2e > 0 else 0

            # Check constraints
            meets_budget = None
            meets_target = None
            if agent_input.budget_usd:
                meets_budget = total_cost <= agent_input.budget_usd
            if agent_input.target_co2e_tonnes:
                meets_target = total_co2e >= agent_input.target_co2e_tonnes

            provenance_hash = self._calculate_provenance_hash(input_data, site_plans)

            output = AfforestationOutput(
                project_id=agent_input.project_id,
                total_area_ha=total_area,
                total_30yr_sequestration_tonnes_co2e=total_co2e,
                total_project_cost_usd=total_cost,
                average_cost_per_tonne_co2e=avg_cost_per_tonne,
                site_plans=site_plans,
                meets_budget=meets_budget,
                meets_carbon_target=meets_target,
                provenance_hash=provenance_hash,
                warnings=warnings
            )

            return AgentResult(
                success=True,
                data=output.model_dump()
            )

        except Exception as e:
            logger.error(f"Afforestation planning failed: {str(e)}", exc_info=True)
            return AgentResult(success=False, error=str(e))

    def _create_site_plan(
        self,
        site: SiteAssessment,
        objective: PlantingObjective,
        duration_years: int,
        species_preference: Optional[str],
        warnings: List[str]
    ) -> AfforestationPlan:
        """Create plan for a single site."""

        # Determine species mix based on objective and climate
        if objective == PlantingObjective.CARBON_SEQUESTRATION:
            species_mix = "fast_growing"
            density = 1600
        elif objective == PlantingObjective.BIODIVERSITY:
            species_mix = "native_mixed"
            density = 1100
        else:
            species_mix = "native_mixed"
            density = 1200

        # Get sequestration rate
        climate_rates = SEQUESTRATION_RATES.get(site.climate_zone, SEQUESTRATION_RATES[ClimateZone.TEMPERATE])
        base_rate = climate_rates.get(species_mix, 3.0)

        # Adjust for soil quality
        rate = base_rate * (0.5 + 0.5 * site.soil_quality_score)
        annual_seq_c = rate * site.area_ha
        total_seq_c = annual_seq_c * duration_years
        total_co2e = total_seq_c * self.CO2_TO_C_RATIO

        # Calculate costs
        cost_level = "medium"  # Default
        establishment = (
            PLANTING_COSTS["site_preparation"][cost_level] +
            PLANTING_COSTS["seedlings"][cost_level] +
            PLANTING_COSTS["planting_labor"][cost_level]
        ) * site.area_ha

        annual_maintenance = PLANTING_COSTS["maintenance_annual"][cost_level] * site.area_ha
        total_cost = establishment + (annual_maintenance * duration_years)
        cost_per_tonne = total_cost / total_co2e if total_co2e > 0 else 0

        # Determine planting season
        if site.climate_zone in (ClimateZone.TROPICAL, ClimateZone.SUBTROPICAL):
            planting_start = "Start of rainy season"
        else:
            planting_start = "Spring (March-May)"

        # Years to canopy closure
        canopy_years = {
            "fast_growing": 5,
            "native_mixed": 8,
            "slow_growing": 12
        }.get(species_mix, 8)

        # Co-benefits
        biodiversity_score = 70 if species_mix == "native_mixed" else 45
        watershed_score = 65 + (site.slope_percent or 10) * 0.3

        return AfforestationPlan(
            site_id=site.site_id,
            recommended_species_mix=species_mix,
            planting_density_stems_ha=density,
            annual_sequestration_tonnes_c=round(annual_seq_c, 2),
            total_sequestration_30yr_tonnes_c=round(total_seq_c, 2),
            total_co2e_30yr_tonnes=round(total_co2e, 2),
            establishment_cost_usd=round(establishment, 2),
            annual_maintenance_cost_usd=round(annual_maintenance, 2),
            total_30yr_cost_usd=round(total_cost, 2),
            cost_per_tonne_co2e_usd=round(cost_per_tonne, 2),
            planting_start_recommended=planting_start,
            years_to_canopy_closure=canopy_years,
            biodiversity_score=min(100, biodiversity_score),
            watershed_benefit_score=min(100, watershed_score)
        )

    def _calculate_provenance_hash(
        self,
        inputs: Dict[str, Any],
        plans: List[AfforestationPlan]
    ) -> str:
        content = {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "inputs_hash": hashlib.sha256(
                json.dumps(inputs, sort_keys=True, default=str).encode()
            ).hexdigest(),
            "timestamp": DeterministicClock.now().isoformat(),
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()
