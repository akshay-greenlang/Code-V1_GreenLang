# -*- coding: utf-8 -*-
"""
GL-MRV-NBS-005: Agroforestry MRV Agent
======================================

This agent measures, reports, and verifies carbon in agroforestry systems
following IPCC guidelines and GHG Protocol guidance.

Capabilities:
    - Tree biomass carbon in agricultural landscapes
    - Soil carbon under agroforestry
    - Carbon sequestration rates by system type
    - Multi-strata agroforestry accounting
    - Silvopasture and alley cropping systems

Methodologies:
    - IPCC 2006 Guidelines Chapter 4 (Forest Land) for trees
    - IPCC Chapter 5 (Cropland) for soil carbon
    - GHG Protocol Land Sector Guidance
    - VCS AFOLU methodologies for agroforestry

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

class AgroforestrySystem(str, Enum):
    """Agroforestry system types."""
    SILVOARABLE = "silvoarable"           # Trees + crops
    SILVOPASTORAL = "silvopastoral"       # Trees + livestock/pasture
    AGROSILVOPASTORAL = "agrosilvopastoral"  # Trees + crops + livestock
    ALLEY_CROPPING = "alley_cropping"
    SHADE_GROWN_COFFEE = "shade_grown_coffee"
    SHADE_GROWN_COCOA = "shade_grown_cocoa"
    HOME_GARDEN = "home_garden"
    WINDBREAK = "windbreak"
    RIPARIAN_BUFFER = "riparian_buffer"
    IMPROVED_FALLOW = "improved_fallow"
    MULTISTRATA = "multistrata"


class TreeSpecies(str, Enum):
    """Common agroforestry tree species categories."""
    TROPICAL_FRUIT = "tropical_fruit"
    TROPICAL_TIMBER = "tropical_timber"
    NITROGEN_FIXING = "nitrogen_fixing"
    PALM = "palm"
    TEMPERATE_FRUIT = "temperate_fruit"
    TEMPERATE_TIMBER = "temperate_timber"
    CONIFER = "conifer"
    EUCALYPTUS = "eucalyptus"
    MIXED_SPECIES = "mixed_species"


class ClimateZone(str, Enum):
    """Climate zones for agroforestry."""
    TROPICAL_WET = "tropical_wet"
    TROPICAL_DRY = "tropical_dry"
    SUBTROPICAL = "subtropical"
    TEMPERATE = "temperate"
    BOREAL = "boreal"


class IPCCTier(str, Enum):
    """IPCC methodology tiers."""
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"


# =============================================================================
# Default Carbon Values
# =============================================================================

# Above-ground carbon by system type and climate (tonnes C/ha at maturity)
AGB_CARBON_DEFAULTS: Dict[AgroforestrySystem, Dict[ClimateZone, float]] = {
    AgroforestrySystem.SILVOARABLE: {
        ClimateZone.TROPICAL_WET: 35.0,
        ClimateZone.TROPICAL_DRY: 25.0,
        ClimateZone.SUBTROPICAL: 30.0,
        ClimateZone.TEMPERATE: 25.0,
    },
    AgroforestrySystem.SILVOPASTORAL: {
        ClimateZone.TROPICAL_WET: 45.0,
        ClimateZone.TROPICAL_DRY: 30.0,
        ClimateZone.SUBTROPICAL: 35.0,
        ClimateZone.TEMPERATE: 30.0,
    },
    AgroforestrySystem.SHADE_GROWN_COFFEE: {
        ClimateZone.TROPICAL_WET: 40.0,
        ClimateZone.TROPICAL_DRY: 25.0,
        ClimateZone.SUBTROPICAL: 30.0,
    },
    AgroforestrySystem.SHADE_GROWN_COCOA: {
        ClimateZone.TROPICAL_WET: 50.0,
        ClimateZone.TROPICAL_DRY: 30.0,
    },
    AgroforestrySystem.ALLEY_CROPPING: {
        ClimateZone.TROPICAL_WET: 25.0,
        ClimateZone.TROPICAL_DRY: 15.0,
        ClimateZone.SUBTROPICAL: 20.0,
        ClimateZone.TEMPERATE: 18.0,
    },
    AgroforestrySystem.MULTISTRATA: {
        ClimateZone.TROPICAL_WET: 80.0,
        ClimateZone.TROPICAL_DRY: 50.0,
        ClimateZone.SUBTROPICAL: 60.0,
    },
    AgroforestrySystem.HOME_GARDEN: {
        ClimateZone.TROPICAL_WET: 55.0,
        ClimateZone.TROPICAL_DRY: 35.0,
        ClimateZone.SUBTROPICAL: 45.0,
        ClimateZone.TEMPERATE: 35.0,
    },
    AgroforestrySystem.WINDBREAK: {
        ClimateZone.TROPICAL_WET: 20.0,
        ClimateZone.TROPICAL_DRY: 15.0,
        ClimateZone.SUBTROPICAL: 18.0,
        ClimateZone.TEMPERATE: 15.0,
    },
}

# Root-to-shoot ratios for agroforestry trees
ROOT_SHOOT_RATIOS: Dict[TreeSpecies, float] = {
    TreeSpecies.TROPICAL_FRUIT: 0.25,
    TreeSpecies.TROPICAL_TIMBER: 0.27,
    TreeSpecies.NITROGEN_FIXING: 0.30,
    TreeSpecies.PALM: 0.20,
    TreeSpecies.TEMPERATE_FRUIT: 0.30,
    TreeSpecies.TEMPERATE_TIMBER: 0.26,
    TreeSpecies.CONIFER: 0.29,
    TreeSpecies.EUCALYPTUS: 0.25,
    TreeSpecies.MIXED_SPECIES: 0.27,
}

# Annual carbon sequestration rates (tonnes C/ha/yr during growth)
SEQUESTRATION_RATES: Dict[AgroforestrySystem, Dict[ClimateZone, float]] = {
    AgroforestrySystem.SILVOARABLE: {
        ClimateZone.TROPICAL_WET: 3.5,
        ClimateZone.TROPICAL_DRY: 2.0,
        ClimateZone.SUBTROPICAL: 2.5,
        ClimateZone.TEMPERATE: 2.0,
    },
    AgroforestrySystem.SILVOPASTORAL: {
        ClimateZone.TROPICAL_WET: 4.0,
        ClimateZone.TROPICAL_DRY: 2.5,
        ClimateZone.SUBTROPICAL: 3.0,
        ClimateZone.TEMPERATE: 2.5,
    },
    AgroforestrySystem.SHADE_GROWN_COFFEE: {
        ClimateZone.TROPICAL_WET: 3.0,
        ClimateZone.TROPICAL_DRY: 2.0,
        ClimateZone.SUBTROPICAL: 2.5,
    },
    AgroforestrySystem.MULTISTRATA: {
        ClimateZone.TROPICAL_WET: 5.0,
        ClimateZone.TROPICAL_DRY: 3.0,
        ClimateZone.SUBTROPICAL: 4.0,
    },
}

# SOC under agroforestry (tonnes C/ha to 30cm)
SOC_AGROFORESTRY: Dict[ClimateZone, float] = {
    ClimateZone.TROPICAL_WET: 60.0,
    ClimateZone.TROPICAL_DRY: 45.0,
    ClimateZone.SUBTROPICAL: 55.0,
    ClimateZone.TEMPERATE: 65.0,
    ClimateZone.BOREAL: 80.0,
}


# =============================================================================
# Pydantic Models
# =============================================================================

class AgroforestryMeasurement(BaseModel):
    """Agroforestry site measurement."""

    site_id: str = Field(..., description="Site identifier")
    measurement_date: date = Field(..., description="Measurement date")
    area_ha: float = Field(..., gt=0, description="Area in hectares")

    # Classification
    system_type: AgroforestrySystem = Field(..., description="Agroforestry system type")
    climate_zone: ClimateZone = Field(..., description="Climate zone")
    dominant_tree_species: TreeSpecies = Field(
        default=TreeSpecies.MIXED_SPECIES,
        description="Dominant tree species category"
    )

    # Stand characteristics
    tree_age_years: Optional[int] = Field(None, ge=0, description="Tree age")
    tree_density_ha: Optional[int] = Field(None, ge=0, description="Trees per hectare")
    canopy_cover_percent: Optional[float] = Field(None, ge=0, le=100)

    # Direct measurements (Tier 2/3)
    measured_agb_tonnes_c_ha: Optional[float] = Field(None, ge=0)
    measured_bgb_tonnes_c_ha: Optional[float] = Field(None, ge=0)
    measured_soc_tonnes_c_ha: Optional[float] = Field(None, ge=0)


class AgroforestryEstimate(BaseModel):
    """Carbon estimate for agroforestry site."""

    # Carbon pools (tonnes C)
    tree_agb_carbon_tonnes: float = Field(..., ge=0)
    tree_bgb_carbon_tonnes: float = Field(..., ge=0)
    soil_carbon_tonnes: float = Field(..., ge=0)
    total_carbon_tonnes: float = Field(..., ge=0)
    carbon_density_tonnes_ha: float = Field(..., ge=0)

    # Sequestration (tonnes C/yr)
    annual_sequestration_tonnes_c: float = Field(default=0.0)
    cumulative_sequestration_tonnes_c: float = Field(default=0.0)

    # CO2e
    total_co2e_tonnes: float = Field(..., ge=0)
    annual_co2e_sequestration: float = Field(default=0.0)

    # Uncertainty
    uncertainty_percent: float = Field(..., ge=0)
    lower_bound_tonnes: float = Field(...)
    upper_bound_tonnes: float = Field(...)

    # Methodology
    tier: IPCCTier = Field(...)
    methodology_reference: str = Field(
        default="IPCC 2006 Guidelines + GHG Protocol Land Sector"
    )

    calculation_timestamp: datetime = Field(default_factory=DeterministicClock.now)


class AgroforestryInput(BaseModel):
    """Input for Agroforestry MRV Agent."""

    project_id: str = Field(..., description="Project identifier")
    baseline_year: int = Field(..., ge=1990)
    reporting_year: int = Field(..., ge=1990)

    measurements: List[AgroforestryMeasurement] = Field(..., min_length=1)
    target_tier: IPCCTier = Field(default=IPCCTier.TIER_1)

    # Baseline scenario
    baseline_carbon_tonnes_ha: Optional[float] = Field(
        None,
        description="Baseline carbon (e.g., from degraded land)"
    )


class AgroforestryOutput(BaseModel):
    """Output from Agroforestry MRV Agent."""

    project_id: str = Field(...)
    calculation_date: datetime = Field(default_factory=DeterministicClock.now)

    # Summary
    total_area_ha: float = Field(..., ge=0)
    total_carbon_stock_tonnes: float = Field(..., ge=0)
    average_carbon_density_tonnes_ha: float = Field(..., ge=0)

    # Sequestration
    annual_sequestration_tonnes_c: float = Field(...)
    cumulative_sequestration_tonnes_c: float = Field(...)

    # CO2e
    total_co2e_tonnes: float = Field(..., ge=0)
    annual_co2e_sequestration: float = Field(...)

    # Additionality (vs baseline)
    additional_carbon_tonnes: Optional[float] = Field(None)
    additional_co2e_tonnes: Optional[float] = Field(None)

    # Estimates
    estimates: List[AgroforestryEstimate] = Field(...)

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
# Agroforestry MRV Agent
# =============================================================================

class AgroforestryMRVAgent(DeterministicAgent):
    """
    GL-MRV-NBS-005: Agroforestry MRV Agent

    Measures, reports, and verifies carbon in agroforestry systems.
    CRITICAL PATH agent with zero-hallucination guarantee.

    Supported Systems:
        - Silvoarable (trees + crops)
        - Silvopastoral (trees + pasture)
        - Shade-grown coffee/cocoa
        - Alley cropping
        - Home gardens
        - Windbreaks/shelterbelts

    Usage:
        agent = AgroforestryMRVAgent()
        result = agent.execute({
            "project_id": "AF-001",
            "measurements": [...],
        })
    """

    AGENT_ID = "GL-MRV-NBS-005"
    AGENT_NAME = "Agroforestry MRV Agent"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="AgroforestryMRVAgent",
        category=AgentCategory.CRITICAL,
        uses_chat_session=False,
        uses_rag=False,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="Agroforestry carbon MRV with IPCC compliance"
    )

    CO2_TO_C_RATIO = 44.0 / 12.0

    def __init__(self, enable_audit_trail: bool = True):
        """Initialize Agroforestry MRV Agent."""
        super().__init__(enable_audit_trail=enable_audit_trail)
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agroforestry carbon calculation."""
        calculation_trace: List[str] = []
        warnings: List[str] = []

        try:
            agent_input = AgroforestryInput(**inputs)
            calculation_trace.append(
                f"Processing {len(agent_input.measurements)} agroforestry sites"
            )

            total_area = sum(m.area_ha for m in agent_input.measurements)
            estimates: List[AgroforestryEstimate] = []

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
            annual_seq = sum(e.annual_sequestration_tonnes_c for e in estimates)
            cumulative_seq = sum(e.cumulative_sequestration_tonnes_c for e in estimates)
            total_co2e = sum(e.total_co2e_tonnes for e in estimates)
            annual_co2e_seq = sum(e.annual_co2e_sequestration for e in estimates)

            # Additionality vs baseline
            additional_carbon = None
            additional_co2e = None
            if agent_input.baseline_carbon_tonnes_ha is not None:
                baseline_total = agent_input.baseline_carbon_tonnes_ha * total_area
                additional_carbon = total_carbon - baseline_total
                additional_co2e = additional_carbon * self.CO2_TO_C_RATIO

            # Uncertainty
            uncertainty = self._calculate_combined_uncertainty(estimates)
            lower_bound = total_carbon * (1 - uncertainty / 100)
            upper_bound = total_carbon * (1 + uncertainty / 100)

            provenance_hash = self._calculate_provenance_hash(
                inputs, estimates, calculation_trace
            )

            output = AgroforestryOutput(
                project_id=agent_input.project_id,
                total_area_ha=total_area,
                total_carbon_stock_tonnes=total_carbon,
                average_carbon_density_tonnes_ha=avg_density,
                annual_sequestration_tonnes_c=annual_seq,
                cumulative_sequestration_tonnes_c=cumulative_seq,
                total_co2e_tonnes=total_co2e,
                annual_co2e_sequestration=annual_co2e_seq,
                additional_carbon_tonnes=additional_carbon,
                additional_co2e_tonnes=additional_co2e,
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
                operation="agroforestry_mrv",
                inputs=inputs,
                outputs=output.model_dump(),
                calculation_trace=calculation_trace,
                metadata={"agent_id": self.AGENT_ID, "version": self.VERSION}
            )

            return output.model_dump()

        except Exception as e:
            logger.error(f"Agroforestry calculation failed: {str(e)}", exc_info=True)
            raise

    def _calculate_site_carbon(
        self,
        measurement: AgroforestryMeasurement,
        tier: IPCCTier,
        calculation_trace: List[str],
        warnings: List[str]
    ) -> AgroforestryEstimate:
        """Calculate carbon for an agroforestry site."""

        area = measurement.area_ha
        system = measurement.system_type
        climate = measurement.climate_zone
        species = measurement.dominant_tree_species

        # Tree AGB carbon
        if tier != IPCCTier.TIER_1 and measurement.measured_agb_tonnes_c_ha:
            agb_per_ha = measurement.measured_agb_tonnes_c_ha
        else:
            # Get default and adjust for age if available
            agb_at_maturity = AGB_CARBON_DEFAULTS.get(system, {}).get(climate, 30.0)

            if measurement.tree_age_years:
                # Assume 20 years to maturity, linear growth
                maturity_age = 20
                growth_factor = min(1.0, measurement.tree_age_years / maturity_age)
                agb_per_ha = agb_at_maturity * growth_factor
            else:
                agb_per_ha = agb_at_maturity * 0.6  # Assume 60% if age unknown

        agb_total = agb_per_ha * area

        # Tree BGB carbon
        if tier != IPCCTier.TIER_1 and measurement.measured_bgb_tonnes_c_ha:
            bgb_per_ha = measurement.measured_bgb_tonnes_c_ha
        else:
            rs_ratio = ROOT_SHOOT_RATIOS.get(species, 0.27)
            bgb_per_ha = agb_per_ha * rs_ratio
        bgb_total = bgb_per_ha * area

        # Soil carbon
        if tier != IPCCTier.TIER_1 and measurement.measured_soc_tonnes_c_ha:
            soc_per_ha = measurement.measured_soc_tonnes_c_ha
        else:
            soc_per_ha = SOC_AGROFORESTRY.get(climate, 55.0)
        soc_total = soc_per_ha * area

        total_carbon = agb_total + bgb_total + soc_total
        density = total_carbon / area if area > 0 else 0.0

        # Sequestration
        seq_rate = SEQUESTRATION_RATES.get(system, {}).get(climate, 2.5)
        annual_seq = seq_rate * area
        cumulative_seq = 0.0
        if measurement.tree_age_years:
            cumulative_seq = annual_seq * measurement.tree_age_years

        # CO2e
        total_co2e = total_carbon * self.CO2_TO_C_RATIO
        annual_co2e_seq = annual_seq * self.CO2_TO_C_RATIO

        calculation_trace.append(
            f"Site {measurement.site_id} ({system.value}): "
            f"AGB={agb_total:.1f}, BGB={bgb_total:.1f}, SOC={soc_total:.1f}, "
            f"Total={total_carbon:.1f} t C"
        )

        # Uncertainty
        uncertainty_map = {
            IPCCTier.TIER_1: 75.0,
            IPCCTier.TIER_2: 35.0,
            IPCCTier.TIER_3: 15.0,
        }
        uncertainty = uncertainty_map.get(tier, 75.0)

        return AgroforestryEstimate(
            tree_agb_carbon_tonnes=agb_total,
            tree_bgb_carbon_tonnes=bgb_total,
            soil_carbon_tonnes=soc_total,
            total_carbon_tonnes=total_carbon,
            carbon_density_tonnes_ha=density,
            annual_sequestration_tonnes_c=annual_seq,
            cumulative_sequestration_tonnes_c=cumulative_seq,
            total_co2e_tonnes=total_co2e,
            annual_co2e_sequestration=annual_co2e_seq,
            uncertainty_percent=uncertainty,
            lower_bound_tonnes=total_carbon * (1 - uncertainty / 100),
            upper_bound_tonnes=total_carbon * (1 + uncertainty / 100),
            tier=tier
        )

    def _calculate_combined_uncertainty(
        self,
        estimates: List[AgroforestryEstimate]
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
        estimates: List[AgroforestryEstimate],
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


def create_agroforestry_agent(enable_audit_trail: bool = True) -> AgroforestryMRVAgent:
    """Create an Agroforestry MRV Agent instance."""
    return AgroforestryMRVAgent(enable_audit_trail=enable_audit_trail)
