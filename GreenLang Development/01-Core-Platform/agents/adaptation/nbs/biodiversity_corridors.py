# -*- coding: utf-8 -*-
"""
GL-ADAPT-NBS-008: Biodiversity Corridors Agent
===============================================

Wildlife corridor planning for climate adaptation.

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


class ConnectivityLevel(str, Enum):
    ISOLATED = "isolated"
    POOR = "poor"
    MODERATE = "moderate"
    GOOD = "good"
    EXCELLENT = "excellent"


class HabitatType(str, Enum):
    FOREST = "forest"
    WETLAND = "wetland"
    GRASSLAND = "grassland"
    RIPARIAN = "riparian"
    MONTANE = "montane"


class HabitatPatch(BaseModel):
    patch_id: str = Field(...)
    area_ha: float = Field(..., gt=0)
    habitat_type: HabitatType = Field(...)
    species_richness: int = Field(default=50, ge=0)
    threatened_species_count: int = Field(default=5, ge=0)
    connectivity_score: float = Field(default=0.5, ge=0, le=1)


class CorridorPlan(BaseModel):
    corridor_id: str = Field(...)
    source_patch_id: str = Field(...)
    target_patch_id: str = Field(...)
    length_km: float = Field(...)
    width_m: float = Field(...)
    area_ha: float = Field(...)
    connectivity_improvement: float = Field(...)
    species_benefiting: int = Field(...)
    implementation_cost_million: float = Field(...)
    priority: str = Field(...)
    design_features: List[str] = Field(...)


class CorridorInput(BaseModel):
    project_id: str = Field(...)
    region_name: str = Field(...)
    patches: List[HabitatPatch] = Field(..., min_length=2)
    climate_scenario: str = Field(default="RCP4.5")
    target_connectivity: float = Field(default=0.7, ge=0, le=1)


class CorridorOutput(BaseModel):
    project_id: str = Field(...)
    region_name: str = Field(...)
    calculation_date: datetime = Field(default_factory=DeterministicClock.now)
    current_connectivity_score: float = Field(...)
    projected_connectivity_score: float = Field(...)
    total_patches: int = Field(...)
    total_patch_area_ha: float = Field(...)
    corridor_plans: List[CorridorPlan] = Field(...)
    total_corridor_area_ha: float = Field(...)
    total_investment_million: float = Field(...)
    species_protected: int = Field(...)
    provenance_hash: str = Field(...)


class BiodiversityCorridorsAgent(BaseAgent):
    """GL-ADAPT-NBS-008: Biodiversity Corridors Agent"""

    AGENT_ID = "GL-ADAPT-NBS-008"
    AGENT_NAME = "Biodiversity Corridors Agent"
    VERSION = "1.0.0"

    HABITAT_CORRIDOR_WIDTH = {
        HabitatType.FOREST: 200,
        HabitatType.WETLAND: 100,
        HabitatType.GRASSLAND: 150,
        HabitatType.RIPARIAN: 50,
        HabitatType.MONTANE: 300,
    }

    RESTORATION_COST_PER_HA = {
        HabitatType.FOREST: 8000,
        HabitatType.WETLAND: 15000,
        HabitatType.GRASSLAND: 3000,
        HabitatType.RIPARIAN: 10000,
        HabitatType.MONTANE: 5000,
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Biodiversity corridor planning",
                version=self.VERSION
            )
        super().__init__(config)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        try:
            agent_input = CorridorInput(**input_data)

            current_connectivity = sum(p.connectivity_score for p in agent_input.patches) / len(agent_input.patches)

            corridor_plans = self._plan_corridors(
                agent_input.patches,
                agent_input.target_connectivity,
                agent_input.climate_scenario
            )

            total_corridor_area = sum(c.area_ha for c in corridor_plans)
            total_investment = sum(c.implementation_cost_million for c in corridor_plans)

            projected_connectivity = min(
                1.0,
                current_connectivity + sum(c.connectivity_improvement for c in corridor_plans) / len(agent_input.patches)
            )

            all_species = set()
            for patch in agent_input.patches:
                all_species.add(patch.species_richness)
            species_protected = max(all_species) if all_species else 0

            provenance_hash = hashlib.sha256(
                json.dumps({"agent": self.AGENT_ID, "ts": DeterministicClock.now().isoformat()}).encode()
            ).hexdigest()

            output = CorridorOutput(
                project_id=agent_input.project_id,
                region_name=agent_input.region_name,
                current_connectivity_score=round(current_connectivity, 3),
                projected_connectivity_score=round(projected_connectivity, 3),
                total_patches=len(agent_input.patches),
                total_patch_area_ha=sum(p.area_ha for p in agent_input.patches),
                corridor_plans=corridor_plans,
                total_corridor_area_ha=round(total_corridor_area, 1),
                total_investment_million=round(total_investment, 2),
                species_protected=species_protected,
                provenance_hash=provenance_hash,
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            return AgentResult(success=False, error=str(e))

    def _plan_corridors(
        self,
        patches: List[HabitatPatch],
        target_connectivity: float,
        scenario: str
    ) -> List[CorridorPlan]:
        corridors = []
        corridor_id = 1

        sorted_patches = sorted(patches, key=lambda p: p.connectivity_score)

        for i, patch in enumerate(sorted_patches[:-1]):
            if patch.connectivity_score < target_connectivity:
                best_target = None
                best_score = 0

                for j, other in enumerate(sorted_patches):
                    if i != j and other.habitat_type == patch.habitat_type:
                        combined_score = other.species_richness + other.threatened_species_count * 5
                        if combined_score > best_score:
                            best_score = combined_score
                            best_target = other

                if best_target is None and len(sorted_patches) > i + 1:
                    best_target = sorted_patches[i + 1]

                if best_target:
                    corridor = self._design_corridor(
                        corridor_id, patch, best_target, scenario
                    )
                    corridors.append(corridor)
                    corridor_id += 1

        return corridors

    def _design_corridor(
        self,
        corridor_id: int,
        source: HabitatPatch,
        target: HabitatPatch,
        scenario: str
    ) -> CorridorPlan:
        length_km = ((source.area_ha + target.area_ha) / 2) ** 0.5 / 10
        length_km = max(1, min(50, length_km))

        width_m = self.HABITAT_CORRIDOR_WIDTH.get(source.habitat_type, 150)
        climate_factor = 1.2 if scenario == "RCP4.5" else 1.4
        width_m *= climate_factor

        area_ha = length_km * (width_m / 1000) * 100

        cost_per_ha = self.RESTORATION_COST_PER_HA.get(source.habitat_type, 8000)
        cost = (area_ha * cost_per_ha) / 1_000_000

        connectivity_improvement = min(0.3, area_ha / 500)

        species_benefiting = min(
            source.species_richness + target.species_richness,
            int(area_ha / 10) + source.threatened_species_count + target.threatened_species_count
        )

        total_threatened = source.threatened_species_count + target.threatened_species_count
        if total_threatened > 10:
            priority = "critical"
        elif total_threatened > 5:
            priority = "high"
        else:
            priority = "moderate"

        design_features = self._get_design_features(source.habitat_type)

        return CorridorPlan(
            corridor_id=f"COR-{corridor_id:03d}",
            source_patch_id=source.patch_id,
            target_patch_id=target.patch_id,
            length_km=round(length_km, 2),
            width_m=round(width_m, 0),
            area_ha=round(area_ha, 1),
            connectivity_improvement=round(connectivity_improvement, 3),
            species_benefiting=species_benefiting,
            implementation_cost_million=round(cost, 3),
            priority=priority,
            design_features=design_features,
        )

    def _get_design_features(self, habitat: HabitatType) -> List[str]:
        features = {
            HabitatType.FOREST: [
                "Native tree species planting",
                "Understory vegetation",
                "Wildlife crossings",
                "Snag retention"
            ],
            HabitatType.WETLAND: [
                "Wetland creation",
                "Buffer zones",
                "Amphibian tunnels",
                "Native aquatic plants"
            ],
            HabitatType.GRASSLAND: [
                "Native grass seeding",
                "Pollinator habitat",
                "Brush piles",
                "Grazing management"
            ],
            HabitatType.RIPARIAN: [
                "Streamside buffers",
                "Bank stabilization",
                "Fish passage",
                "Floodplain connectivity"
            ],
            HabitatType.MONTANE: [
                "Elevation gradient connectivity",
                "Rock outcrop preservation",
                "Alpine meadow linkages",
                "Thermal refugia"
            ],
        }
        return features.get(habitat, ["Habitat restoration", "Native species planting"])
