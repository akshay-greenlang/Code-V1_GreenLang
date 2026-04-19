# -*- coding: utf-8 -*-
"""
GL-ADAPT-NBS-003: Species Migration Agent
=========================================

Models climate-driven species range shifts and migration patterns.

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


class MigrationRisk(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class TaxonomicGroup(str, Enum):
    MAMMALS = "mammals"
    BIRDS = "birds"
    REPTILES = "reptiles"
    AMPHIBIANS = "amphibians"
    FISH = "fish"
    PLANTS = "plants"
    INVERTEBRATES = "invertebrates"


class SpeciesRecord(BaseModel):
    species_id: str = Field(...)
    common_name: str = Field(...)
    scientific_name: str = Field(...)
    taxonomic_group: TaxonomicGroup = Field(...)
    current_range_km2: float = Field(..., gt=0)
    elevation_range_m: tuple = Field(default=(0, 3000))
    thermal_tolerance_c: tuple = Field(default=(5, 35))
    dispersal_ability: str = Field(default="moderate")
    is_endemic: bool = Field(default=False)


class MigrationProjection(BaseModel):
    species_id: str = Field(...)
    species_name: str = Field(...)
    current_suitable_area_km2: float = Field(...)
    future_suitable_area_km2: float = Field(...)
    area_change_percent: float = Field(...)
    projected_shift_km: float = Field(...)
    shift_direction: str = Field(...)
    elevation_shift_m: float = Field(...)
    migration_risk: MigrationRisk = Field(...)
    range_overlap_percent: float = Field(...)
    corridor_needed: bool = Field(...)
    conservation_priority: str = Field(...)


class SpeciesMigrationInput(BaseModel):
    project_id: str = Field(...)
    species: List[SpeciesRecord] = Field(..., min_length=1)
    climate_scenario: str = Field(default="RCP4.5")
    time_horizon: int = Field(default=2050)
    temperature_increase_c: float = Field(default=1.5)


class SpeciesMigrationOutput(BaseModel):
    project_id: str = Field(...)
    calculation_date: datetime = Field(default_factory=DeterministicClock.now)
    total_species: int = Field(...)
    high_risk_species: int = Field(...)
    projections: List[MigrationProjection] = Field(...)
    corridor_recommendations: List[str] = Field(...)
    provenance_hash: str = Field(...)


class SpeciesMigrationAgent(BaseAgent):
    """GL-ADAPT-NBS-003: Species Migration Agent"""

    AGENT_ID = "GL-ADAPT-NBS-003"
    AGENT_NAME = "Species Migration Agent"
    VERSION = "1.0.0"

    # Migration velocity by group (km per decade)
    MIGRATION_VELOCITIES = {
        TaxonomicGroup.BIRDS: 35.0,
        TaxonomicGroup.MAMMALS: 20.0,
        TaxonomicGroup.REPTILES: 15.0,
        TaxonomicGroup.AMPHIBIANS: 10.0,
        TaxonomicGroup.FISH: 25.0,
        TaxonomicGroup.PLANTS: 5.0,
        TaxonomicGroup.INVERTEBRATES: 12.0,
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(name=self.AGENT_NAME, description="Species migration modeling", version=self.VERSION)
        super().__init__(config)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        try:
            agent_input = SpeciesMigrationInput(**input_data)
            projections = []
            high_risk = 0

            for species in agent_input.species:
                projection = self._project_migration(
                    species,
                    agent_input.temperature_increase_c,
                    agent_input.time_horizon
                )
                projections.append(projection)
                if projection.migration_risk in (MigrationRisk.HIGH, MigrationRisk.CRITICAL):
                    high_risk += 1

            # Corridor recommendations
            corridors = []
            if high_risk > 0:
                corridors.append("Establish north-south oriented corridors")
                corridors.append("Protect elevational gradients")
            if any(p.corridor_needed for p in projections):
                corridors.append("Connect fragmented habitats")

            provenance_hash = hashlib.sha256(
                json.dumps({"agent": self.AGENT_ID, "ts": DeterministicClock.now().isoformat()}).encode()
            ).hexdigest()

            output = SpeciesMigrationOutput(
                project_id=agent_input.project_id,
                total_species=len(agent_input.species),
                high_risk_species=high_risk,
                projections=projections,
                corridor_recommendations=corridors,
                provenance_hash=provenance_hash
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    def _project_migration(
        self,
        species: SpeciesRecord,
        temp_increase: float,
        target_year: int
    ) -> MigrationProjection:
        # Calculate required climate shift
        decades = (target_year - 2020) / 10
        climate_velocity_km = temp_increase * 6.7 * decades * 10  # ~6.7 km/yr per C

        # Species dispersal capacity
        base_velocity = self.MIGRATION_VELOCITIES.get(species.taxonomic_group, 15.0)
        dispersal_multiplier = {
            "high": 1.5,
            "moderate": 1.0,
            "low": 0.5,
        }.get(species.dispersal_ability, 1.0)

        species_velocity = base_velocity * dispersal_multiplier * decades

        # Range shift
        shift_km = min(species_velocity, climate_velocity_km)
        shift_direction = "poleward" if shift_km > 0 else "none"

        # Elevation shift (~150m per C)
        elevation_shift = temp_increase * 150

        # Area change estimation
        thermal_range = species.thermal_tolerance_c[1] - species.thermal_tolerance_c[0]
        tolerance_factor = thermal_range / 30  # Broader tolerance = less impact

        if tolerance_factor > 1.0:
            area_change = -10 * temp_increase
        else:
            area_change = -25 * temp_increase / tolerance_factor

        future_area = species.current_range_km2 * (1 + area_change / 100)
        future_area = max(0, future_area)

        # Range overlap
        if species_velocity >= climate_velocity_km:
            overlap = 80 - (10 * temp_increase)
        else:
            overlap = 50 - (20 * temp_increase)
        overlap = max(0, min(100, overlap))

        # Risk assessment
        if area_change < -50 or species.is_endemic:
            risk = MigrationRisk.CRITICAL
            priority = "critical"
        elif area_change < -30:
            risk = MigrationRisk.HIGH
            priority = "high"
        elif area_change < -15:
            risk = MigrationRisk.MODERATE
            priority = "moderate"
        else:
            risk = MigrationRisk.LOW
            priority = "low"

        # Corridor need
        corridor_needed = overlap < 60 or species.dispersal_ability == "low"

        return MigrationProjection(
            species_id=species.species_id,
            species_name=species.common_name,
            current_suitable_area_km2=species.current_range_km2,
            future_suitable_area_km2=round(future_area, 1),
            area_change_percent=round(area_change, 1),
            projected_shift_km=round(shift_km, 1),
            shift_direction=shift_direction,
            elevation_shift_m=round(elevation_shift, 0),
            migration_risk=risk,
            range_overlap_percent=round(overlap, 1),
            corridor_needed=corridor_needed,
            conservation_priority=priority
        )
