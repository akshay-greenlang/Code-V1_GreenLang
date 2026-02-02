# -*- coding: utf-8 -*-
"""
GL-MRV-NBS-004: Blue Carbon MRV Agent
=====================================

This agent measures, reports, and verifies blue carbon (coastal/marine carbon)
following IPCC guidelines and Blue Carbon Initiative methodologies.

Capabilities:
    - Mangrove carbon stock estimation
    - Seagrass meadow carbon accounting
    - Salt marsh carbon measurement
    - Tidal wetland emissions/removals
    - Coastal ecosystem degradation emissions

Methodologies:
    - IPCC 2013 Wetlands Supplement (coastal wetlands)
    - Blue Carbon Initiative methods
    - IUCN Blue Carbon guidelines
    - Verified Carbon Standard (VCS) blue carbon methodologies

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

class BlueEcosystemType(str, Enum):
    """Blue carbon ecosystem types."""
    MANGROVE = "mangrove"
    SEAGRASS = "seagrass"
    SALT_MARSH = "salt_marsh"
    TIDAL_FRESHWATER = "tidal_freshwater"
    KELP_FOREST = "kelp_forest"
    MACROALGAE = "macroalgae"


class CoastalZone(str, Enum):
    """Coastal climate zones."""
    TROPICAL = "tropical"
    SUBTROPICAL = "subtropical"
    TEMPERATE = "temperate"
    BOREAL = "boreal"


class EcosystemCondition(str, Enum):
    """Ecosystem condition status."""
    PRISTINE = "pristine"
    DEGRADED = "degraded"
    CONVERTED = "converted"
    RESTORED = "restored"


class IPCCTier(str, Enum):
    """IPCC methodology tiers."""
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"


# =============================================================================
# IPCC Default Carbon Values (from 2013 Wetlands Supplement)
# =============================================================================

# Carbon stocks by ecosystem type (tonnes C/ha)
# Above-ground biomass
AGB_CARBON: Dict[BlueEcosystemType, Dict[CoastalZone, float]] = {
    BlueEcosystemType.MANGROVE: {
        CoastalZone.TROPICAL: 115.0,
        CoastalZone.SUBTROPICAL: 95.0,
        CoastalZone.TEMPERATE: 55.0,
    },
    BlueEcosystemType.SALT_MARSH: {
        CoastalZone.TROPICAL: 8.0,
        CoastalZone.SUBTROPICAL: 7.5,
        CoastalZone.TEMPERATE: 6.5,
        CoastalZone.BOREAL: 5.0,
    },
    BlueEcosystemType.SEAGRASS: {
        CoastalZone.TROPICAL: 2.5,
        CoastalZone.SUBTROPICAL: 2.0,
        CoastalZone.TEMPERATE: 1.5,
    },
}

# Soil carbon to 1m depth (tonnes C/ha)
SOIL_CARBON: Dict[BlueEcosystemType, Dict[CoastalZone, float]] = {
    BlueEcosystemType.MANGROVE: {
        CoastalZone.TROPICAL: 386.0,
        CoastalZone.SUBTROPICAL: 350.0,
        CoastalZone.TEMPERATE: 280.0,
    },
    BlueEcosystemType.SALT_MARSH: {
        CoastalZone.TROPICAL: 255.0,
        CoastalZone.SUBTROPICAL: 245.0,
        CoastalZone.TEMPERATE: 230.0,
        CoastalZone.BOREAL: 200.0,
    },
    BlueEcosystemType.SEAGRASS: {
        CoastalZone.TROPICAL: 140.0,
        CoastalZone.SUBTROPICAL: 125.0,
        CoastalZone.TEMPERATE: 110.0,
    },
}

# Carbon sequestration rates (tonnes C/ha/yr)
SEQUESTRATION_RATES: Dict[BlueEcosystemType, Dict[CoastalZone, float]] = {
    BlueEcosystemType.MANGROVE: {
        CoastalZone.TROPICAL: 2.3,
        CoastalZone.SUBTROPICAL: 2.0,
        CoastalZone.TEMPERATE: 1.5,
    },
    BlueEcosystemType.SALT_MARSH: {
        CoastalZone.TROPICAL: 1.8,
        CoastalZone.SUBTROPICAL: 1.5,
        CoastalZone.TEMPERATE: 1.2,
        CoastalZone.BOREAL: 0.8,
    },
    BlueEcosystemType.SEAGRASS: {
        CoastalZone.TROPICAL: 1.4,
        CoastalZone.SUBTROPICAL: 1.2,
        CoastalZone.TEMPERATE: 0.9,
    },
}

# Emissions from conversion (% of carbon lost as CO2)
CONVERSION_EMISSION_FACTORS: Dict[BlueEcosystemType, float] = {
    BlueEcosystemType.MANGROVE: 0.75,  # 75% of carbon released
    BlueEcosystemType.SALT_MARSH: 0.60,
    BlueEcosystemType.SEAGRASS: 0.50,
}


# =============================================================================
# Pydantic Models
# =============================================================================

class BlueCarbonMeasurement(BaseModel):
    """Blue carbon ecosystem measurement."""

    site_id: str = Field(..., description="Site identifier")
    measurement_date: date = Field(..., description="Measurement date")
    area_ha: float = Field(..., gt=0, description="Area in hectares")

    # Classification
    ecosystem_type: BlueEcosystemType = Field(..., description="Ecosystem type")
    coastal_zone: CoastalZone = Field(..., description="Coastal zone")
    condition: EcosystemCondition = Field(
        default=EcosystemCondition.PRISTINE,
        description="Ecosystem condition"
    )

    # Direct measurements (Tier 2/3)
    measured_agb_tonnes_c_ha: Optional[float] = Field(None, ge=0)
    measured_bgb_tonnes_c_ha: Optional[float] = Field(None, ge=0)
    measured_soil_c_tonnes_ha: Optional[float] = Field(None, ge=0)
    soil_sampling_depth_m: float = Field(default=1.0, gt=0)

    # Sequestration measurement
    measured_sequestration_tonnes_c_ha_yr: Optional[float] = Field(None)

    # For conversion/degradation
    years_since_conversion: Optional[int] = Field(None, ge=0)


class BlueCarbonEstimate(BaseModel):
    """Blue carbon estimate for a site."""

    # Carbon stocks
    agb_carbon_tonnes: float = Field(..., ge=0, description="Above-ground carbon")
    bgb_carbon_tonnes: float = Field(default=0.0, ge=0, description="Below-ground carbon")
    soil_carbon_tonnes: float = Field(..., ge=0, description="Soil carbon")
    total_carbon_tonnes: float = Field(..., ge=0, description="Total carbon stock")
    carbon_density_tonnes_ha: float = Field(..., ge=0)

    # Fluxes (annual)
    sequestration_tonnes_c_yr: float = Field(default=0.0, description="Annual sequestration")
    emission_tonnes_co2_yr: float = Field(default=0.0, description="Annual emissions")
    net_flux_tonnes_co2e_yr: float = Field(..., description="Net annual flux")

    # Uncertainty
    uncertainty_percent: float = Field(..., ge=0)
    lower_bound_tonnes: float = Field(...)
    upper_bound_tonnes: float = Field(...)

    # Methodology
    tier: IPCCTier = Field(...)
    methodology_reference: str = Field(
        default="IPCC 2013 Wetlands Supplement - Coastal Wetlands"
    )

    calculation_timestamp: datetime = Field(default_factory=DeterministicClock.now)


class BlueCarbonInput(BaseModel):
    """Input for Blue Carbon MRV Agent."""

    project_id: str = Field(..., description="Project identifier")
    reporting_year: int = Field(..., ge=1990)

    measurements: List[BlueCarbonMeasurement] = Field(..., min_length=1)
    target_tier: IPCCTier = Field(default=IPCCTier.TIER_1)

    # Project type
    is_conservation_project: bool = Field(
        default=True,
        description="Conservation (avoid emissions) or restoration"
    )
    project_crediting_period_years: int = Field(default=30, ge=1)


class BlueCarbonOutput(BaseModel):
    """Output from Blue Carbon MRV Agent."""

    project_id: str = Field(...)
    calculation_date: datetime = Field(default_factory=DeterministicClock.now)

    # Summary
    total_area_ha: float = Field(..., ge=0)
    total_carbon_stock_tonnes: float = Field(..., ge=0)
    average_carbon_density_tonnes_ha: float = Field(..., ge=0)

    # Annual fluxes
    annual_sequestration_tonnes_c: float = Field(...)
    annual_emissions_tonnes_co2: float = Field(...)
    net_annual_flux_tonnes_co2e: float = Field(...)

    # Project totals (over crediting period)
    total_sequestration_potential_tonnes_co2e: float = Field(...)
    total_avoided_emissions_tonnes_co2e: float = Field(...)

    # Estimates by site
    estimates: List[BlueCarbonEstimate] = Field(...)

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
# Blue Carbon MRV Agent
# =============================================================================

class BlueCarbonMRVAgent(DeterministicAgent):
    """
    GL-MRV-NBS-004: Blue Carbon MRV Agent

    Measures, reports, and verifies blue carbon (coastal/marine ecosystems).
    CRITICAL PATH agent with zero-hallucination guarantee.

    Supported Ecosystems:
        - Mangroves
        - Seagrass meadows
        - Salt marshes
        - Tidal freshwater wetlands

    Usage:
        agent = BlueCarbonMRVAgent()
        result = agent.execute({
            "project_id": "BLUE-001",
            "measurements": [...],
        })
    """

    AGENT_ID = "GL-MRV-NBS-004"
    AGENT_NAME = "Blue Carbon MRV Agent"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="BlueCarbonMRVAgent",
        category=AgentCategory.CRITICAL,
        uses_chat_session=False,
        uses_rag=False,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="Coastal/marine blue carbon MRV with IPCC compliance"
    )

    CO2_TO_C_RATIO = 44.0 / 12.0

    def __init__(self, enable_audit_trail: bool = True):
        """Initialize Blue Carbon MRV Agent."""
        super().__init__(enable_audit_trail=enable_audit_trail)
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute blue carbon calculation."""
        calculation_trace: List[str] = []
        warnings: List[str] = []

        try:
            agent_input = BlueCarbonInput(**inputs)
            calculation_trace.append(
                f"Processing {len(agent_input.measurements)} blue carbon sites"
            )

            total_area = sum(m.area_ha for m in agent_input.measurements)
            estimates: List[BlueCarbonEstimate] = []

            for measurement in agent_input.measurements:
                estimate = self._calculate_site_carbon(
                    measurement=measurement,
                    tier=agent_input.target_tier,
                    calculation_trace=calculation_trace,
                    warnings=warnings
                )
                estimates.append(estimate)

            # Aggregate
            total_carbon = sum(e.total_carbon_tonnes for e in estimates)
            avg_density = total_carbon / total_area if total_area > 0 else 0.0

            annual_seq = sum(e.sequestration_tonnes_c_yr for e in estimates)
            annual_emis = sum(e.emission_tonnes_co2_yr for e in estimates)
            net_flux = sum(e.net_flux_tonnes_co2e_yr for e in estimates)

            # Project totals
            crediting_years = agent_input.project_crediting_period_years
            total_seq_potential = annual_seq * self.CO2_TO_C_RATIO * crediting_years

            # Avoided emissions (if conservation)
            avoided_emissions = 0.0
            if agent_input.is_conservation_project:
                for m in agent_input.measurements:
                    if m.condition == EcosystemCondition.PRISTINE:
                        # Calculate what would be lost if converted
                        ef = CONVERSION_EMISSION_FACTORS.get(m.ecosystem_type, 0.5)
                        agb = AGB_CARBON.get(m.ecosystem_type, {}).get(m.coastal_zone, 50.0)
                        soil = SOIL_CARBON.get(m.ecosystem_type, {}).get(m.coastal_zone, 200.0)
                        avoided = (agb + soil) * ef * m.area_ha * self.CO2_TO_C_RATIO
                        avoided_emissions += avoided

            # Combined uncertainty
            uncertainty = self._calculate_combined_uncertainty(estimates)
            lower_bound = total_carbon * (1 - uncertainty / 100)
            upper_bound = total_carbon * (1 + uncertainty / 100)

            provenance_hash = self._calculate_provenance_hash(
                inputs, estimates, calculation_trace
            )

            output = BlueCarbonOutput(
                project_id=agent_input.project_id,
                total_area_ha=total_area,
                total_carbon_stock_tonnes=total_carbon,
                average_carbon_density_tonnes_ha=avg_density,
                annual_sequestration_tonnes_c=annual_seq,
                annual_emissions_tonnes_co2=annual_emis,
                net_annual_flux_tonnes_co2e=net_flux,
                total_sequestration_potential_tonnes_co2e=total_seq_potential,
                total_avoided_emissions_tonnes_co2e=avoided_emissions,
                estimates=estimates,
                total_uncertainty_percent=uncertainty,
                lower_bound_tonnes=lower_bound,
                upper_bound_tonnes=upper_bound,
                methodology_tier=agent_input.target_tier,
                provenance_hash=provenance_hash,
                calculation_trace=calculation_trace,
                warnings=warnings
            )

            self._capture_audit_entry(
                operation="blue_carbon_mrv",
                inputs=inputs,
                outputs=output.model_dump(),
                calculation_trace=calculation_trace,
                metadata={"agent_id": self.AGENT_ID, "version": self.VERSION}
            )

            return output.model_dump()

        except Exception as e:
            logger.error(f"Blue carbon calculation failed: {str(e)}", exc_info=True)
            raise

    def _calculate_site_carbon(
        self,
        measurement: BlueCarbonMeasurement,
        tier: IPCCTier,
        calculation_trace: List[str],
        warnings: List[str]
    ) -> BlueCarbonEstimate:
        """Calculate carbon for a blue carbon site."""

        area = measurement.area_ha
        eco_type = measurement.ecosystem_type
        zone = measurement.coastal_zone

        # Above-ground biomass carbon
        if tier != IPCCTier.TIER_1 and measurement.measured_agb_tonnes_c_ha:
            agb_per_ha = measurement.measured_agb_tonnes_c_ha
        else:
            agb_per_ha = AGB_CARBON.get(eco_type, {}).get(zone, 50.0)
        agb_total = agb_per_ha * area

        # Below-ground biomass (estimated from root:shoot)
        bgb_per_ha = agb_per_ha * 0.5  # Default 0.5 root:shoot for coastal
        if measurement.measured_bgb_tonnes_c_ha:
            bgb_per_ha = measurement.measured_bgb_tonnes_c_ha
        bgb_total = bgb_per_ha * area

        # Soil carbon
        if tier != IPCCTier.TIER_1 and measurement.measured_soil_c_tonnes_ha:
            soil_per_ha = measurement.measured_soil_c_tonnes_ha
        else:
            soil_per_ha = SOIL_CARBON.get(eco_type, {}).get(zone, 200.0)
            # Adjust for sampling depth (default values are to 1m)
            soil_per_ha = soil_per_ha * measurement.soil_sampling_depth_m
        soil_total = soil_per_ha * area

        total_carbon = agb_total + bgb_total + soil_total
        density = total_carbon / area if area > 0 else 0.0

        # Sequestration (for intact/restored ecosystems)
        sequestration = 0.0
        if measurement.condition in (EcosystemCondition.PRISTINE, EcosystemCondition.RESTORED):
            if measurement.measured_sequestration_tonnes_c_ha_yr:
                seq_rate = measurement.measured_sequestration_tonnes_c_ha_yr
            else:
                seq_rate = SEQUESTRATION_RATES.get(eco_type, {}).get(zone, 1.0)
            sequestration = seq_rate * area

        # Emissions (for converted/degraded ecosystems)
        emissions = 0.0
        if measurement.condition == EcosystemCondition.CONVERTED:
            ef = CONVERSION_EMISSION_FACTORS.get(eco_type, 0.5)
            years = measurement.years_since_conversion or 10
            # Linear decay model over 20 years
            if years <= 20:
                annual_rate = (total_carbon * ef) / 20
                emissions = annual_rate * self.CO2_TO_C_RATIO

        # Net flux (negative = sink)
        net_flux = emissions - (sequestration * self.CO2_TO_C_RATIO)

        calculation_trace.append(
            f"Site {measurement.site_id} ({eco_type.value}): "
            f"AGB={agb_total:.1f}, Soil={soil_total:.1f}, "
            f"Total={total_carbon:.1f} t C"
        )

        # Uncertainty
        uncertainty_map = {
            IPCCTier.TIER_1: 75.0,
            IPCCTier.TIER_2: 40.0,
            IPCCTier.TIER_3: 20.0,
        }
        uncertainty = uncertainty_map.get(tier, 75.0)

        return BlueCarbonEstimate(
            agb_carbon_tonnes=agb_total,
            bgb_carbon_tonnes=bgb_total,
            soil_carbon_tonnes=soil_total,
            total_carbon_tonnes=total_carbon,
            carbon_density_tonnes_ha=density,
            sequestration_tonnes_c_yr=sequestration,
            emission_tonnes_co2_yr=emissions,
            net_flux_tonnes_co2e_yr=net_flux,
            uncertainty_percent=uncertainty,
            lower_bound_tonnes=total_carbon * (1 - uncertainty / 100),
            upper_bound_tonnes=total_carbon * (1 + uncertainty / 100),
            tier=tier
        )

    def _calculate_combined_uncertainty(
        self,
        estimates: List[BlueCarbonEstimate]
    ) -> float:
        """Calculate combined uncertainty."""
        if not estimates:
            return 0.0

        sum_squares = sum(
            (e.uncertainty_percent / 100 * e.total_carbon_tonnes) ** 2
            for e in estimates
        )
        total = sum(e.total_carbon_tonnes for e in estimates)

        if total == 0:
            return 75.0

        return math.sqrt(sum_squares) / total * 100

    def _calculate_provenance_hash(
        self,
        inputs: Dict[str, Any],
        estimates: List[BlueCarbonEstimate],
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


def create_blue_carbon_agent(enable_audit_trail: bool = True) -> BlueCarbonMRVAgent:
    """Create a Blue Carbon MRV Agent instance."""
    return BlueCarbonMRVAgent(enable_audit_trail=enable_audit_trail)
