# -*- coding: utf-8 -*-
"""
GL-MRV-NBS-006: Land Use Change MRV Agent
=========================================

This agent measures, reports, and verifies emissions and removals from
land use, land use change, and forestry (LULUCF) following IPCC guidelines.

Capabilities:
    - Land use change emissions (deforestation, conversion)
    - Land use change removals (afforestation, reforestation)
    - Stock change method for carbon accounting
    - Gain-loss method for emission estimation
    - Historical land use tracking

Methodologies:
    - IPCC 2006 Guidelines Chapter 2 (Generic Methodologies)
    - IPCC 2019 Refinement
    - GHG Protocol Land Sector and Removals Guidance
    - UNFCCC LULUCF reporting guidelines

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import math
from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base_agents import DeterministicAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism.clock import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class LandUseCategory(str, Enum):
    """IPCC land use categories."""
    FOREST_LAND = "forest_land"
    CROPLAND = "cropland"
    GRASSLAND = "grassland"
    WETLAND = "wetland"
    SETTLEMENT = "settlement"
    OTHER_LAND = "other_land"


class TransitionType(str, Enum):
    """Land use transition types."""
    # Forest transitions
    DEFORESTATION_TO_CROPLAND = "deforestation_to_cropland"
    DEFORESTATION_TO_GRASSLAND = "deforestation_to_grassland"
    DEFORESTATION_TO_SETTLEMENT = "deforestation_to_settlement"
    AFFORESTATION_FROM_CROPLAND = "afforestation_from_cropland"
    AFFORESTATION_FROM_GRASSLAND = "afforestation_from_grassland"
    REFORESTATION = "reforestation"

    # Other transitions
    CROPLAND_TO_GRASSLAND = "cropland_to_grassland"
    GRASSLAND_TO_CROPLAND = "grassland_to_cropland"
    WETLAND_DRAINAGE = "wetland_drainage"
    WETLAND_RESTORATION = "wetland_restoration"

    # No change
    REMAINING = "remaining"


class ClimateZone(str, Enum):
    """Climate zones for emission factors."""
    TROPICAL_WET = "tropical_wet"
    TROPICAL_DRY = "tropical_dry"
    SUBTROPICAL = "subtropical"
    WARM_TEMPERATE = "warm_temperate"
    COOL_TEMPERATE = "cool_temperate"
    BOREAL = "boreal"


class IPCCTier(str, Enum):
    """IPCC methodology tiers."""
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"


# =============================================================================
# IPCC Default Values
# =============================================================================

# Carbon stocks by land use category (tonnes C/ha)
# Source: IPCC 2006 Guidelines, various tables
CARBON_STOCKS: Dict[LandUseCategory, Dict[ClimateZone, Dict[str, float]]] = {
    LandUseCategory.FOREST_LAND: {
        ClimateZone.TROPICAL_WET: {"biomass": 200.0, "soil": 65.0},
        ClimateZone.TROPICAL_DRY: {"biomass": 100.0, "soil": 45.0},
        ClimateZone.SUBTROPICAL: {"biomass": 110.0, "soil": 55.0},
        ClimateZone.WARM_TEMPERATE: {"biomass": 90.0, "soil": 70.0},
        ClimateZone.COOL_TEMPERATE: {"biomass": 60.0, "soil": 80.0},
        ClimateZone.BOREAL: {"biomass": 35.0, "soil": 100.0},
    },
    LandUseCategory.CROPLAND: {
        ClimateZone.TROPICAL_WET: {"biomass": 5.0, "soil": 35.0},
        ClimateZone.TROPICAL_DRY: {"biomass": 5.0, "soil": 25.0},
        ClimateZone.SUBTROPICAL: {"biomass": 5.0, "soil": 35.0},
        ClimateZone.WARM_TEMPERATE: {"biomass": 5.0, "soil": 50.0},
        ClimateZone.COOL_TEMPERATE: {"biomass": 5.0, "soil": 60.0},
        ClimateZone.BOREAL: {"biomass": 5.0, "soil": 70.0},
    },
    LandUseCategory.GRASSLAND: {
        ClimateZone.TROPICAL_WET: {"biomass": 8.0, "soil": 60.0},
        ClimateZone.TROPICAL_DRY: {"biomass": 5.0, "soil": 40.0},
        ClimateZone.SUBTROPICAL: {"biomass": 6.0, "soil": 50.0},
        ClimateZone.WARM_TEMPERATE: {"biomass": 4.0, "soil": 65.0},
        ClimateZone.COOL_TEMPERATE: {"biomass": 3.0, "soil": 75.0},
        ClimateZone.BOREAL: {"biomass": 2.0, "soil": 85.0},
    },
    LandUseCategory.WETLAND: {
        ClimateZone.TROPICAL_WET: {"biomass": 50.0, "soil": 200.0},
        ClimateZone.TROPICAL_DRY: {"biomass": 30.0, "soil": 150.0},
        ClimateZone.WARM_TEMPERATE: {"biomass": 25.0, "soil": 250.0},
        ClimateZone.COOL_TEMPERATE: {"biomass": 20.0, "soil": 300.0},
        ClimateZone.BOREAL: {"biomass": 15.0, "soil": 400.0},
    },
    LandUseCategory.SETTLEMENT: {
        ClimateZone.TROPICAL_WET: {"biomass": 10.0, "soil": 30.0},
        ClimateZone.TROPICAL_DRY: {"biomass": 8.0, "soil": 20.0},
        ClimateZone.SUBTROPICAL: {"biomass": 12.0, "soil": 35.0},
        ClimateZone.WARM_TEMPERATE: {"biomass": 15.0, "soil": 45.0},
        ClimateZone.COOL_TEMPERATE: {"biomass": 12.0, "soil": 50.0},
    },
    LandUseCategory.OTHER_LAND: {
        ClimateZone.TROPICAL_WET: {"biomass": 2.0, "soil": 15.0},
        ClimateZone.TROPICAL_DRY: {"biomass": 1.0, "soil": 10.0},
        ClimateZone.SUBTROPICAL: {"biomass": 2.0, "soil": 20.0},
        ClimateZone.WARM_TEMPERATE: {"biomass": 2.0, "soil": 25.0},
        ClimateZone.COOL_TEMPERATE: {"biomass": 1.0, "soil": 30.0},
    },
}

# Fraction of carbon released immediately upon land use change
IMMEDIATE_EMISSION_FRACTION: Dict[TransitionType, float] = {
    TransitionType.DEFORESTATION_TO_CROPLAND: 0.85,
    TransitionType.DEFORESTATION_TO_GRASSLAND: 0.70,
    TransitionType.DEFORESTATION_TO_SETTLEMENT: 0.90,
    TransitionType.WETLAND_DRAINAGE: 0.30,
    TransitionType.GRASSLAND_TO_CROPLAND: 0.20,
}

# Transition period for soil carbon changes (years)
SOIL_TRANSITION_YEARS = 20


# =============================================================================
# Pydantic Models
# =============================================================================

class LandUseTransition(BaseModel):
    """Land use transition record."""

    transition_id: str = Field(..., description="Transition identifier")
    area_ha: float = Field(..., gt=0, description="Area in hectares")

    # Land use categories
    initial_land_use: LandUseCategory = Field(..., description="Initial land use")
    final_land_use: LandUseCategory = Field(..., description="Final land use")
    transition_type: TransitionType = Field(..., description="Transition type")

    # Climate zone
    climate_zone: ClimateZone = Field(..., description="Climate zone")

    # Timing
    transition_year: int = Field(..., ge=1900, description="Year of transition")
    years_since_transition: Optional[int] = Field(None, ge=0)

    # Custom carbon stocks (Tier 2/3)
    initial_biomass_c_tonnes_ha: Optional[float] = Field(None, ge=0)
    initial_soil_c_tonnes_ha: Optional[float] = Field(None, ge=0)
    final_biomass_c_tonnes_ha: Optional[float] = Field(None, ge=0)
    final_soil_c_tonnes_ha: Optional[float] = Field(None, ge=0)


class LUCEmissionsEstimate(BaseModel):
    """Land use change emissions/removals estimate."""

    transition_id: str = Field(...)

    # Carbon stock changes (tonnes C)
    biomass_change_tonnes_c: float = Field(..., description="Biomass change")
    soil_change_tonnes_c: float = Field(..., description="Soil carbon change")
    total_change_tonnes_c: float = Field(..., description="Total carbon change")

    # Emissions (positive = emission, negative = removal)
    immediate_emissions_tonnes_co2: float = Field(...)
    delayed_emissions_tonnes_co2_yr: float = Field(default=0.0)
    total_emissions_tonnes_co2: float = Field(...)

    # Per hectare
    emissions_per_ha_tonnes_co2: float = Field(...)

    # Is it a source or sink?
    is_emission: bool = Field(..., description="True if net emission")

    # Uncertainty
    uncertainty_percent: float = Field(..., ge=0)
    lower_bound_tonnes: float = Field(...)
    upper_bound_tonnes: float = Field(...)

    # Methodology
    tier: IPCCTier = Field(...)
    methodology_reference: str = Field(
        default="IPCC 2006 Guidelines Vol. 4 Ch. 2"
    )

    calculation_timestamp: datetime = Field(default_factory=DeterministicClock.now)


class LandUseChangeInput(BaseModel):
    """Input for Land Use Change MRV Agent."""

    project_id: str = Field(..., description="Project identifier")
    reporting_year: int = Field(..., ge=1990)

    transitions: List[LandUseTransition] = Field(..., min_length=1)
    target_tier: IPCCTier = Field(default=IPCCTier.TIER_1)

    # Optional: track delayed emissions
    include_delayed_emissions: bool = Field(
        default=True,
        description="Include soil carbon decay emissions"
    )


class LandUseChangeOutput(BaseModel):
    """Output from Land Use Change MRV Agent."""

    project_id: str = Field(...)
    calculation_date: datetime = Field(default_factory=DeterministicClock.now)

    # Summary
    total_area_ha: float = Field(..., ge=0)

    # Net emissions (positive = source, negative = sink)
    net_emissions_tonnes_co2: float = Field(...)
    net_emissions_per_ha_tonnes_co2: float = Field(...)

    # Breakdown
    total_biomass_change_tonnes_c: float = Field(...)
    total_soil_change_tonnes_c: float = Field(...)

    # By transition type
    emissions_by_type: Dict[str, float] = Field(default_factory=dict)
    removals_by_type: Dict[str, float] = Field(default_factory=dict)

    # Estimates
    estimates: List[LUCEmissionsEstimate] = Field(...)

    # Summary flags
    is_net_source: bool = Field(...)
    is_net_sink: bool = Field(...)

    # Uncertainty
    total_uncertainty_percent: float = Field(...)
    lower_bound_tonnes: float = Field(...)
    upper_bound_tonnes: float = Field(...)

    # Methodology
    methodology_tier: IPCCTier = Field(...)
    provenance_hash: str = Field(...)
    calculation_trace: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# =============================================================================
# Land Use Change MRV Agent
# =============================================================================

class LandUseChangeMRVAgent(DeterministicAgent):
    """
    GL-MRV-NBS-006: Land Use Change MRV Agent

    Measures, reports, and verifies emissions and removals from land use change.
    CRITICAL PATH agent with zero-hallucination guarantee.

    Supports:
        - Deforestation emissions
        - Afforestation/reforestation removals
        - Land conversion emissions
        - Soil carbon changes

    Usage:
        agent = LandUseChangeMRVAgent()
        result = agent.execute({
            "project_id": "LUC-001",
            "transitions": [...],
        })
    """

    AGENT_ID = "GL-MRV-NBS-006"
    AGENT_NAME = "Land Use Change MRV Agent"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="LandUseChangeMRVAgent",
        category=AgentCategory.CRITICAL,
        uses_chat_session=False,
        uses_rag=False,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="Land use change emissions/removals MRV with IPCC compliance"
    )

    CO2_TO_C_RATIO = 44.0 / 12.0

    def __init__(self, enable_audit_trail: bool = True):
        """Initialize Land Use Change MRV Agent."""
        super().__init__(enable_audit_trail=enable_audit_trail)
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute land use change calculation."""
        calculation_trace: List[str] = []
        warnings: List[str] = []

        try:
            agent_input = LandUseChangeInput(**inputs)
            calculation_trace.append(
                f"Processing {len(agent_input.transitions)} land use transitions"
            )

            total_area = sum(t.area_ha for t in agent_input.transitions)
            estimates: List[LUCEmissionsEstimate] = []

            emissions_by_type: Dict[str, float] = {}
            removals_by_type: Dict[str, float] = {}

            for transition in agent_input.transitions:
                estimate = self._calculate_transition_emissions(
                    transition=transition,
                    tier=agent_input.target_tier,
                    include_delayed=agent_input.include_delayed_emissions,
                    calculation_trace=calculation_trace,
                    warnings=warnings
                )
                estimates.append(estimate)

                # Categorize
                t_type = transition.transition_type.value
                if estimate.is_emission:
                    emissions_by_type[t_type] = emissions_by_type.get(
                        t_type, 0.0
                    ) + estimate.total_emissions_tonnes_co2
                else:
                    removals_by_type[t_type] = removals_by_type.get(
                        t_type, 0.0
                    ) + abs(estimate.total_emissions_tonnes_co2)

            # Aggregate
            net_emissions = sum(e.total_emissions_tonnes_co2 for e in estimates)
            net_per_ha = net_emissions / total_area if total_area > 0 else 0.0
            total_biomass_change = sum(e.biomass_change_tonnes_c for e in estimates)
            total_soil_change = sum(e.soil_change_tonnes_c for e in estimates)

            is_net_source = net_emissions > 0
            is_net_sink = net_emissions < 0

            # Uncertainty
            uncertainty = self._calculate_combined_uncertainty(estimates)
            lower_bound = net_emissions * (1 - uncertainty / 100)
            upper_bound = net_emissions * (1 + uncertainty / 100)

            provenance_hash = self._calculate_provenance_hash(
                inputs, estimates, calculation_trace
            )

            output = LandUseChangeOutput(
                project_id=agent_input.project_id,
                total_area_ha=total_area,
                net_emissions_tonnes_co2=net_emissions,
                net_emissions_per_ha_tonnes_co2=net_per_ha,
                total_biomass_change_tonnes_c=total_biomass_change,
                total_soil_change_tonnes_c=total_soil_change,
                emissions_by_type=emissions_by_type,
                removals_by_type=removals_by_type,
                estimates=estimates,
                is_net_source=is_net_source,
                is_net_sink=is_net_sink,
                total_uncertainty_percent=uncertainty,
                lower_bound_tonnes=lower_bound,
                upper_bound_tonnes=upper_bound,
                methodology_tier=agent_input.target_tier,
                provenance_hash=provenance_hash,
                calculation_trace=calculation_trace,
                warnings=warnings
            )

            self._capture_audit_entry(
                operation="land_use_change_mrv",
                inputs=inputs,
                outputs=output.model_dump(),
                calculation_trace=calculation_trace,
                metadata={"agent_id": self.AGENT_ID, "version": self.VERSION}
            )

            return output.model_dump()

        except Exception as e:
            logger.error(f"Land use change calculation failed: {str(e)}", exc_info=True)
            raise

    def _calculate_transition_emissions(
        self,
        transition: LandUseTransition,
        tier: IPCCTier,
        include_delayed: bool,
        calculation_trace: List[str],
        warnings: List[str]
    ) -> LUCEmissionsEstimate:
        """Calculate emissions for a land use transition."""

        area = transition.area_ha
        climate = transition.climate_zone
        t_type = transition.transition_type

        # Get carbon stocks
        initial_stocks = CARBON_STOCKS.get(transition.initial_land_use, {}).get(
            climate, {"biomass": 50.0, "soil": 50.0}
        )
        final_stocks = CARBON_STOCKS.get(transition.final_land_use, {}).get(
            climate, {"biomass": 50.0, "soil": 50.0}
        )

        # Use custom values if provided (Tier 2/3)
        if tier != IPCCTier.TIER_1:
            if transition.initial_biomass_c_tonnes_ha is not None:
                initial_stocks["biomass"] = transition.initial_biomass_c_tonnes_ha
            if transition.initial_soil_c_tonnes_ha is not None:
                initial_stocks["soil"] = transition.initial_soil_c_tonnes_ha
            if transition.final_biomass_c_tonnes_ha is not None:
                final_stocks["biomass"] = transition.final_biomass_c_tonnes_ha
            if transition.final_soil_c_tonnes_ha is not None:
                final_stocks["soil"] = transition.final_soil_c_tonnes_ha

        # Calculate changes (negative = carbon loss = emission)
        biomass_change = (final_stocks["biomass"] - initial_stocks["biomass"]) * area
        soil_change = (final_stocks["soil"] - initial_stocks["soil"]) * area
        total_change = biomass_change + soil_change

        # Immediate emissions (biomass)
        immediate_fraction = IMMEDIATE_EMISSION_FRACTION.get(t_type, 0.5)
        immediate_biomass_loss = min(0, biomass_change) * immediate_fraction
        immediate_emissions = abs(immediate_biomass_loss) * self.CO2_TO_C_RATIO

        # Delayed emissions (soil carbon decay)
        delayed_emissions_yr = 0.0
        if include_delayed and soil_change < 0:
            years_elapsed = transition.years_since_transition or 0
            remaining_years = max(0, SOIL_TRANSITION_YEARS - years_elapsed)
            if remaining_years > 0:
                delayed_emissions_yr = (abs(soil_change) / SOIL_TRANSITION_YEARS) * self.CO2_TO_C_RATIO

        # For afforestation/reforestation (positive change = removal)
        if total_change > 0:
            # This is a removal, so "emission" is negative
            total_emissions = -total_change * self.CO2_TO_C_RATIO
            is_emission = False
        else:
            total_emissions = abs(total_change) * self.CO2_TO_C_RATIO
            is_emission = True

        emissions_per_ha = total_emissions / area if area > 0 else 0.0

        calculation_trace.append(
            f"Transition {transition.transition_id}: "
            f"{transition.initial_land_use.value} -> {transition.final_land_use.value}, "
            f"Biomass delta={biomass_change:.1f} t C, "
            f"Soil delta={soil_change:.1f} t C, "
            f"Net={'emission' if is_emission else 'removal'}={abs(total_emissions):.1f} t CO2"
        )

        # Uncertainty
        uncertainty_map = {
            IPCCTier.TIER_1: 90.0,
            IPCCTier.TIER_2: 50.0,
            IPCCTier.TIER_3: 25.0,
        }
        uncertainty = uncertainty_map.get(tier, 90.0)

        return LUCEmissionsEstimate(
            transition_id=transition.transition_id,
            biomass_change_tonnes_c=biomass_change,
            soil_change_tonnes_c=soil_change,
            total_change_tonnes_c=total_change,
            immediate_emissions_tonnes_co2=immediate_emissions,
            delayed_emissions_tonnes_co2_yr=delayed_emissions_yr,
            total_emissions_tonnes_co2=total_emissions,
            emissions_per_ha_tonnes_co2=emissions_per_ha,
            is_emission=is_emission,
            uncertainty_percent=uncertainty,
            lower_bound_tonnes=total_emissions * (1 - uncertainty / 100),
            upper_bound_tonnes=total_emissions * (1 + uncertainty / 100),
            tier=tier
        )

    def _calculate_combined_uncertainty(
        self,
        estimates: List[LUCEmissionsEstimate]
    ) -> float:
        """Calculate combined uncertainty."""
        if not estimates:
            return 0.0

        sum_squares = sum(
            (e.uncertainty_percent / 100 * abs(e.total_emissions_tonnes_co2)) ** 2
            for e in estimates
        )
        total = abs(sum(e.total_emissions_tonnes_co2 for e in estimates))

        if total == 0:
            return 90.0

        return math.sqrt(sum_squares) / total * 100

    def _calculate_provenance_hash(
        self,
        inputs: Dict[str, Any],
        estimates: List[LUCEmissionsEstimate],
        trace: List[str]
    ) -> str:
        """Calculate SHA-256 provenance hash."""
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


def create_land_use_change_agent(enable_audit_trail: bool = True) -> LandUseChangeMRVAgent:
    """Create a Land Use Change MRV Agent instance."""
    return LandUseChangeMRVAgent(enable_audit_trail=enable_audit_trail)
