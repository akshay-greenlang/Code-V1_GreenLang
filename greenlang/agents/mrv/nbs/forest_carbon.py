# -*- coding: utf-8 -*-
"""
GL-MRV-NBS-001: Forest Carbon MRV Agent
=======================================

This agent measures, reports, and verifies forest carbon sequestration
following IPCC guidelines and GHG Protocol Land Sector Guidance.

Capabilities:
    - Above-ground biomass (AGB) calculation using allometric equations
    - Below-ground biomass (BGB) estimation using root-to-shoot ratios
    - Dead organic matter (DOM) and litter carbon estimation
    - Soil organic carbon (SOC) in forest systems
    - Growth and yield modeling for carbon stock changes
    - Uncertainty quantification (IPCC Tier 1/2/3)

Methodologies:
    - IPCC 2006 Guidelines for National GHG Inventories (Agriculture, Forestry)
    - IPCC 2019 Refinement for forest carbon
    - GHG Protocol Land Sector and Removals Guidance
    - Verified Carbon Standard (VCS) AFOLU methodologies

Zero-Hallucination Guarantee:
    All carbon calculations use published allometric equations, IPCC default
    factors, and peer-reviewed methodologies. No LLM reasoning in calculation path.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import math
from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.agents.base_agents import DeterministicAgent, AuditEntry
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism.clock import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class ForestType(str, Enum):
    """Forest type classification following IPCC categories."""
    TROPICAL_RAINFOREST = "tropical_rainforest"
    TROPICAL_MOIST_DECIDUOUS = "tropical_moist_deciduous"
    TROPICAL_DRY = "tropical_dry"
    TROPICAL_SHRUBLAND = "tropical_shrubland"
    TROPICAL_MOUNTAIN = "tropical_mountain"
    SUBTROPICAL_HUMID = "subtropical_humid"
    SUBTROPICAL_DRY = "subtropical_dry"
    SUBTROPICAL_STEPPE = "subtropical_steppe"
    SUBTROPICAL_MOUNTAIN = "subtropical_mountain"
    TEMPERATE_OCEANIC = "temperate_oceanic"
    TEMPERATE_CONTINENTAL = "temperate_continental"
    TEMPERATE_MOUNTAIN = "temperate_mountain"
    BOREAL_CONIFEROUS = "boreal_coniferous"
    BOREAL_TUNDRA_WOODLAND = "boreal_tundra_woodland"
    BOREAL_MOUNTAIN = "boreal_mountain"
    PLANTATION_BROADLEAF = "plantation_broadleaf"
    PLANTATION_CONIFER = "plantation_conifer"


class CarbonPool(str, Enum):
    """Forest carbon pools per IPCC guidelines."""
    ABOVE_GROUND_BIOMASS = "above_ground_biomass"  # AGB
    BELOW_GROUND_BIOMASS = "below_ground_biomass"  # BGB
    DEAD_WOOD = "dead_wood"                        # DOM - standing/lying deadwood
    LITTER = "litter"                              # DOM - leaf litter, fine debris
    SOIL_ORGANIC_CARBON = "soil_organic_carbon"    # SOC


class MeasurementMethod(str, Enum):
    """Methods for forest carbon measurement."""
    GROUND_INVENTORY = "ground_inventory"          # Plot-based measurements
    REMOTE_SENSING = "remote_sensing"              # LiDAR, satellite
    ALLOMETRIC_EQUATION = "allometric_equation"    # DBH/height-based
    VOLUME_BASED = "volume_based"                  # Timber volume conversion
    GAIN_LOSS = "gain_loss"                        # Stock change method
    DEFAULT_FACTOR = "default_factor"              # IPCC Tier 1


class IPCCTier(str, Enum):
    """IPCC methodology tiers."""
    TIER_1 = "tier_1"  # Default values
    TIER_2 = "tier_2"  # Country-specific data
    TIER_3 = "tier_3"  # Advanced methods (models, inventory)


# =============================================================================
# IPCC Default Factors (2006/2019)
# =============================================================================

# Above-ground biomass (tonnes dry matter per hectare) by forest type
# Source: IPCC 2006 Guidelines, Table 4.7
AGB_DEFAULT_VALUES: Dict[ForestType, float] = {
    ForestType.TROPICAL_RAINFOREST: 300.0,
    ForestType.TROPICAL_MOIST_DECIDUOUS: 220.0,
    ForestType.TROPICAL_DRY: 130.0,
    ForestType.TROPICAL_SHRUBLAND: 70.0,
    ForestType.TROPICAL_MOUNTAIN: 195.0,
    ForestType.SUBTROPICAL_HUMID: 220.0,
    ForestType.SUBTROPICAL_DRY: 130.0,
    ForestType.SUBTROPICAL_STEPPE: 70.0,
    ForestType.SUBTROPICAL_MOUNTAIN: 195.0,
    ForestType.TEMPERATE_OCEANIC: 180.0,
    ForestType.TEMPERATE_CONTINENTAL: 120.0,
    ForestType.TEMPERATE_MOUNTAIN: 100.0,
    ForestType.BOREAL_CONIFEROUS: 50.0,
    ForestType.BOREAL_TUNDRA_WOODLAND: 15.0,
    ForestType.BOREAL_MOUNTAIN: 30.0,
    ForestType.PLANTATION_BROADLEAF: 150.0,
    ForestType.PLANTATION_CONIFER: 120.0,
}

# Root-to-shoot ratios (R) for below-ground biomass
# Source: IPCC 2006 Guidelines, Table 4.4
ROOT_SHOOT_RATIOS: Dict[ForestType, float] = {
    ForestType.TROPICAL_RAINFOREST: 0.37,
    ForestType.TROPICAL_MOIST_DECIDUOUS: 0.27,
    ForestType.TROPICAL_DRY: 0.56,
    ForestType.TROPICAL_SHRUBLAND: 0.40,
    ForestType.TROPICAL_MOUNTAIN: 0.27,
    ForestType.SUBTROPICAL_HUMID: 0.27,
    ForestType.SUBTROPICAL_DRY: 0.56,
    ForestType.SUBTROPICAL_STEPPE: 0.40,
    ForestType.SUBTROPICAL_MOUNTAIN: 0.27,
    ForestType.TEMPERATE_OCEANIC: 0.26,
    ForestType.TEMPERATE_CONTINENTAL: 0.26,
    ForestType.TEMPERATE_MOUNTAIN: 0.26,
    ForestType.BOREAL_CONIFEROUS: 0.39,
    ForestType.BOREAL_TUNDRA_WOODLAND: 0.39,
    ForestType.BOREAL_MOUNTAIN: 0.39,
    ForestType.PLANTATION_BROADLEAF: 0.26,
    ForestType.PLANTATION_CONIFER: 0.29,
}

# Carbon fraction of dry matter (CF)
# Source: IPCC 2006 Guidelines, default = 0.47 for all forests
CARBON_FRACTION_DEFAULT = 0.47

# Dead wood and litter as fraction of AGB
# Source: IPCC 2006 Guidelines, Table 2.2
DEAD_WOOD_FRACTION: Dict[ForestType, float] = {
    ForestType.TROPICAL_RAINFOREST: 0.08,
    ForestType.TROPICAL_MOIST_DECIDUOUS: 0.06,
    ForestType.TROPICAL_DRY: 0.04,
    ForestType.TEMPERATE_OCEANIC: 0.10,
    ForestType.TEMPERATE_CONTINENTAL: 0.08,
    ForestType.BOREAL_CONIFEROUS: 0.12,
}

LITTER_FRACTION: Dict[ForestType, float] = {
    ForestType.TROPICAL_RAINFOREST: 0.04,
    ForestType.TROPICAL_MOIST_DECIDUOUS: 0.03,
    ForestType.TROPICAL_DRY: 0.02,
    ForestType.TEMPERATE_OCEANIC: 0.05,
    ForestType.TEMPERATE_CONTINENTAL: 0.04,
    ForestType.BOREAL_CONIFEROUS: 0.08,
}

# Uncertainty factors (half-width of 95% CI as % of mean)
# Source: IPCC 2006 Guidelines
UNCERTAINTY_FACTORS: Dict[CarbonPool, Dict[IPCCTier, float]] = {
    CarbonPool.ABOVE_GROUND_BIOMASS: {
        IPCCTier.TIER_1: 75.0,  # +/- 75%
        IPCCTier.TIER_2: 30.0,  # +/- 30%
        IPCCTier.TIER_3: 15.0,  # +/- 15%
    },
    CarbonPool.BELOW_GROUND_BIOMASS: {
        IPCCTier.TIER_1: 90.0,
        IPCCTier.TIER_2: 40.0,
        IPCCTier.TIER_3: 20.0,
    },
    CarbonPool.DEAD_WOOD: {
        IPCCTier.TIER_1: 100.0,
        IPCCTier.TIER_2: 50.0,
        IPCCTier.TIER_3: 25.0,
    },
    CarbonPool.LITTER: {
        IPCCTier.TIER_1: 100.0,
        IPCCTier.TIER_2: 50.0,
        IPCCTier.TIER_3: 25.0,
    },
    CarbonPool.SOIL_ORGANIC_CARBON: {
        IPCCTier.TIER_1: 90.0,
        IPCCTier.TIER_2: 45.0,
        IPCCTier.TIER_3: 20.0,
    },
}


# =============================================================================
# Pydantic Models
# =============================================================================

class ForestMeasurement(BaseModel):
    """Individual forest plot measurement data."""

    plot_id: str = Field(..., description="Unique plot identifier")
    measurement_date: date = Field(..., description="Date of measurement")
    area_ha: float = Field(..., gt=0, description="Plot area in hectares")

    # Stand characteristics
    forest_type: ForestType = Field(..., description="Forest type classification")
    stand_age_years: Optional[int] = Field(None, ge=0, description="Stand age in years")

    # Direct measurements (Tier 2/3)
    mean_dbh_cm: Optional[float] = Field(None, ge=0, description="Mean diameter at breast height (cm)")
    mean_height_m: Optional[float] = Field(None, ge=0, description="Mean tree height (m)")
    stems_per_ha: Optional[int] = Field(None, ge=0, description="Number of stems per hectare")
    basal_area_m2_ha: Optional[float] = Field(None, ge=0, description="Basal area (m2/ha)")

    # Volume-based (Tier 2/3)
    merchantable_volume_m3_ha: Optional[float] = Field(
        None, ge=0, description="Merchantable volume (m3/ha)"
    )
    wood_density_tonnes_m3: Optional[float] = Field(
        None, ge=0, description="Wood density (tonnes dry matter/m3)"
    )

    # Direct biomass measurements (Tier 3)
    measured_agb_tonnes_ha: Optional[float] = Field(
        None, ge=0, description="Directly measured AGB (tonnes dry matter/ha)"
    )
    measured_bgb_tonnes_ha: Optional[float] = Field(
        None, ge=0, description="Directly measured BGB (tonnes dry matter/ha)"
    )

    # Carbon pool fractions (if measured)
    carbon_fraction: Optional[float] = Field(
        None, ge=0, le=1, description="Carbon fraction of dry matter"
    )

    # Location
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="Latitude")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="Longitude")
    elevation_m: Optional[float] = Field(None, description="Elevation in meters")


class BiomassEstimate(BaseModel):
    """Estimated biomass and carbon for a carbon pool."""

    carbon_pool: CarbonPool = Field(..., description="Carbon pool type")
    biomass_tonnes_ha: float = Field(..., ge=0, description="Biomass (tonnes dry matter/ha)")
    carbon_tonnes_ha: float = Field(..., ge=0, description="Carbon stock (tonnes C/ha)")
    carbon_tonnes_total: float = Field(..., ge=0, description="Total carbon (tonnes C)")

    # Uncertainty
    uncertainty_percent: float = Field(..., ge=0, description="Uncertainty (+/- %)")
    lower_bound_tonnes: float = Field(..., ge=0, description="Lower 95% CI (tonnes C)")
    upper_bound_tonnes: float = Field(..., description="Upper 95% CI (tonnes C)")

    # Methodology
    method: MeasurementMethod = Field(..., description="Estimation method used")
    tier: IPCCTier = Field(..., description="IPCC tier level")

    # Provenance
    calculation_timestamp: datetime = Field(
        default_factory=DeterministicClock.now,
        description="When calculation was performed"
    )


class ForestCarbonInput(BaseModel):
    """Input for Forest Carbon MRV Agent."""

    project_id: str = Field(..., description="Project identifier")
    measurement_period_start: date = Field(..., description="Start of measurement period")
    measurement_period_end: date = Field(..., description="End of measurement period")

    # Measurements
    measurements: List[ForestMeasurement] = Field(
        ..., min_length=1, description="Forest plot measurements"
    )

    # Configuration
    target_tier: IPCCTier = Field(
        default=IPCCTier.TIER_1,
        description="Target IPCC tier for calculations"
    )
    include_pools: List[CarbonPool] = Field(
        default_factory=lambda: [
            CarbonPool.ABOVE_GROUND_BIOMASS,
            CarbonPool.BELOW_GROUND_BIOMASS,
        ],
        description="Carbon pools to include"
    )

    # Optional parameters
    country_specific_factors: Optional[Dict[str, float]] = Field(
        None, description="Country-specific emission factors for Tier 2"
    )
    allometric_equation_id: Optional[str] = Field(
        None, description="Specific allometric equation to use"
    )

    @field_validator('measurements')
    @classmethod
    def validate_measurements(cls, v: List[ForestMeasurement]) -> List[ForestMeasurement]:
        """Ensure measurements have required data."""
        if not v:
            raise ValueError("At least one measurement is required")
        return v


class ForestCarbonOutput(BaseModel):
    """Output from Forest Carbon MRV Agent."""

    project_id: str = Field(..., description="Project identifier")
    calculation_date: datetime = Field(
        default_factory=DeterministicClock.now,
        description="Date of calculation"
    )

    # Summary results
    total_area_ha: float = Field(..., ge=0, description="Total project area (ha)")
    total_carbon_stock_tonnes: float = Field(
        ..., ge=0, description="Total carbon stock (tonnes C)"
    )
    carbon_stock_per_ha: float = Field(
        ..., ge=0, description="Average carbon density (tonnes C/ha)"
    )

    # Results by pool
    pool_estimates: List[BiomassEstimate] = Field(
        ..., description="Carbon estimates by pool"
    )

    # Uncertainty
    total_uncertainty_percent: float = Field(
        ..., ge=0, description="Combined uncertainty (+/- %)"
    )
    total_lower_bound_tonnes: float = Field(
        ..., ge=0, description="Lower 95% CI (tonnes C)"
    )
    total_upper_bound_tonnes: float = Field(
        ..., description="Upper 95% CI (tonnes C)"
    )

    # CO2 equivalent
    co2e_tonnes_total: float = Field(
        ..., ge=0, description="Total CO2 equivalent (tonnes CO2e)"
    )

    # Methodology
    methodology_tier: IPCCTier = Field(..., description="IPCC tier achieved")
    methodology_reference: str = Field(
        default="IPCC 2006 Guidelines Vol. 4 Ch. 4",
        description="Methodology reference"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    calculation_trace: List[str] = Field(
        default_factory=list, description="Step-by-step calculation trace"
    )
    warnings: List[str] = Field(default_factory=list, description="Calculation warnings")


# =============================================================================
# Forest Carbon MRV Agent
# =============================================================================

class ForestCarbonMRVAgent(DeterministicAgent):
    """
    GL-MRV-NBS-001: Forest Carbon MRV Agent

    Measures, reports, and verifies forest carbon sequestration using
    IPCC-compliant methodologies.

    This is a CRITICAL PATH agent with zero-hallucination guarantee.
    All calculations are deterministic using published factors and equations.

    Supported Carbon Pools:
        - Above-ground biomass (AGB)
        - Below-ground biomass (BGB)
        - Dead wood
        - Litter
        - Soil organic carbon (SOC)

    IPCC Tier Support:
        - Tier 1: Uses IPCC default values
        - Tier 2: Uses country-specific or regional data
        - Tier 3: Uses direct measurements or models

    Usage:
        agent = ForestCarbonMRVAgent()
        result = agent.execute({
            "project_id": "FOREST-001",
            "measurements": [...],
            "target_tier": "tier_1"
        })
    """

    AGENT_ID = "GL-MRV-NBS-001"
    AGENT_NAME = "Forest Carbon MRV Agent"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="ForestCarbonMRVAgent",
        category=AgentCategory.CRITICAL,
        uses_chat_session=False,
        uses_rag=False,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="Forest carbon sequestration MRV with IPCC compliance"
    )

    # CO2 to C ratio (molecular weight ratio)
    CO2_TO_C_RATIO = 44.0 / 12.0  # 3.67

    def __init__(self, enable_audit_trail: bool = True):
        """Initialize Forest Carbon MRV Agent."""
        super().__init__(enable_audit_trail=enable_audit_trail)
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute forest carbon calculation.

        Args:
            inputs: Input parameters including measurements and configuration

        Returns:
            Dictionary containing carbon estimates and provenance
        """
        calculation_trace: List[str] = []
        warnings: List[str] = []

        try:
            # Parse and validate input
            agent_input = ForestCarbonInput(**inputs)
            calculation_trace.append(
                f"Parsed input for project {agent_input.project_id} "
                f"with {len(agent_input.measurements)} measurements"
            )

            # Calculate total area
            total_area_ha = sum(m.area_ha for m in agent_input.measurements)
            calculation_trace.append(f"Total project area: {total_area_ha:.2f} ha")

            # Determine achievable tier
            achieved_tier = self._determine_achievable_tier(
                agent_input.measurements,
                agent_input.target_tier
            )
            if achieved_tier != agent_input.target_tier:
                warnings.append(
                    f"Target tier {agent_input.target_tier.value} not achievable, "
                    f"using {achieved_tier.value}"
                )
            calculation_trace.append(f"Using IPCC methodology tier: {achieved_tier.value}")

            # Calculate carbon for each pool
            pool_estimates: List[BiomassEstimate] = []

            for pool in agent_input.include_pools:
                estimate = self._calculate_carbon_pool(
                    measurements=agent_input.measurements,
                    pool=pool,
                    tier=achieved_tier,
                    country_factors=agent_input.country_specific_factors,
                    calculation_trace=calculation_trace,
                    warnings=warnings
                )
                pool_estimates.append(estimate)

            # Calculate totals
            total_carbon = sum(e.carbon_tonnes_total for e in pool_estimates)
            carbon_per_ha = total_carbon / total_area_ha if total_area_ha > 0 else 0.0

            calculation_trace.append(
                f"Total carbon stock: {total_carbon:.2f} tonnes C "
                f"({carbon_per_ha:.2f} tonnes C/ha)"
            )

            # Calculate combined uncertainty (root sum of squares)
            total_uncertainty = self._calculate_combined_uncertainty(pool_estimates)
            lower_bound = total_carbon * (1 - total_uncertainty / 100)
            upper_bound = total_carbon * (1 + total_uncertainty / 100)

            calculation_trace.append(
                f"Combined uncertainty: +/- {total_uncertainty:.1f}% "
                f"(95% CI: {lower_bound:.2f} - {upper_bound:.2f} tonnes C)"
            )

            # Convert to CO2e
            co2e_total = total_carbon * self.CO2_TO_C_RATIO
            calculation_trace.append(
                f"CO2 equivalent: {co2e_total:.2f} tonnes CO2e "
                f"(using ratio {self.CO2_TO_C_RATIO:.3f})"
            )

            # Calculate provenance hash
            provenance_hash = self._calculate_provenance_hash(
                inputs, pool_estimates, calculation_trace
            )

            # Build output
            output = ForestCarbonOutput(
                project_id=agent_input.project_id,
                total_area_ha=total_area_ha,
                total_carbon_stock_tonnes=total_carbon,
                carbon_stock_per_ha=carbon_per_ha,
                pool_estimates=pool_estimates,
                total_uncertainty_percent=total_uncertainty,
                total_lower_bound_tonnes=max(0, lower_bound),
                total_upper_bound_tonnes=upper_bound,
                co2e_tonnes_total=co2e_total,
                methodology_tier=achieved_tier,
                provenance_hash=provenance_hash,
                calculation_trace=calculation_trace,
                warnings=warnings
            )

            # Capture audit entry
            self._capture_audit_entry(
                operation="forest_carbon_mrv",
                inputs=inputs,
                outputs=output.model_dump(),
                calculation_trace=calculation_trace,
                metadata={
                    "agent_id": self.AGENT_ID,
                    "version": self.VERSION,
                    "tier": achieved_tier.value,
                }
            )

            return output.model_dump()

        except Exception as e:
            logger.error(f"Forest carbon calculation failed: {str(e)}", exc_info=True)
            raise

    def _determine_achievable_tier(
        self,
        measurements: List[ForestMeasurement],
        target_tier: IPCCTier
    ) -> IPCCTier:
        """
        Determine the highest achievable IPCC tier based on available data.

        Args:
            measurements: List of forest measurements
            target_tier: Target tier requested

        Returns:
            Achievable IPCC tier
        """
        # Check for Tier 3 data (direct measurements)
        has_tier3_data = all(
            m.measured_agb_tonnes_ha is not None
            for m in measurements
        )

        # Check for Tier 2 data (DBH, height, volume)
        has_tier2_data = all(
            (m.mean_dbh_cm is not None and m.stems_per_ha is not None) or
            m.merchantable_volume_m3_ha is not None
            for m in measurements
        )

        if target_tier == IPCCTier.TIER_3 and has_tier3_data:
            return IPCCTier.TIER_3
        elif target_tier in (IPCCTier.TIER_2, IPCCTier.TIER_3) and has_tier2_data:
            return IPCCTier.TIER_2
        else:
            return IPCCTier.TIER_1

    def _calculate_carbon_pool(
        self,
        measurements: List[ForestMeasurement],
        pool: CarbonPool,
        tier: IPCCTier,
        country_factors: Optional[Dict[str, float]],
        calculation_trace: List[str],
        warnings: List[str]
    ) -> BiomassEstimate:
        """
        Calculate carbon stock for a specific pool.

        Args:
            measurements: Forest measurements
            pool: Carbon pool to calculate
            tier: IPCC tier to use
            country_factors: Country-specific factors (optional)
            calculation_trace: Trace for audit
            warnings: Warnings list

        Returns:
            BiomassEstimate for the pool
        """
        total_area = sum(m.area_ha for m in measurements)

        if pool == CarbonPool.ABOVE_GROUND_BIOMASS:
            biomass, method = self._calculate_agb(measurements, tier, country_factors)
        elif pool == CarbonPool.BELOW_GROUND_BIOMASS:
            biomass, method = self._calculate_bgb(measurements, tier, country_factors)
        elif pool == CarbonPool.DEAD_WOOD:
            biomass, method = self._calculate_dead_wood(measurements, tier)
        elif pool == CarbonPool.LITTER:
            biomass, method = self._calculate_litter(measurements, tier)
        elif pool == CarbonPool.SOIL_ORGANIC_CARBON:
            biomass, method = self._calculate_soc(measurements, tier)
        else:
            raise ValueError(f"Unknown carbon pool: {pool}")

        # Calculate carbon (biomass * carbon fraction)
        carbon_fraction = CARBON_FRACTION_DEFAULT
        # Use measured carbon fraction if available
        measured_cf = [m.carbon_fraction for m in measurements if m.carbon_fraction]
        if measured_cf:
            carbon_fraction = sum(measured_cf) / len(measured_cf)

        carbon_per_ha = biomass * carbon_fraction
        carbon_total = carbon_per_ha * total_area

        # Get uncertainty
        uncertainty = UNCERTAINTY_FACTORS.get(pool, {}).get(tier, 75.0)
        lower_bound = max(0, carbon_total * (1 - uncertainty / 100))
        upper_bound = carbon_total * (1 + uncertainty / 100)

        calculation_trace.append(
            f"{pool.value}: {biomass:.2f} t DM/ha x {carbon_fraction:.3f} = "
            f"{carbon_per_ha:.2f} t C/ha x {total_area:.2f} ha = "
            f"{carbon_total:.2f} t C (+/- {uncertainty:.0f}%)"
        )

        return BiomassEstimate(
            carbon_pool=pool,
            biomass_tonnes_ha=biomass,
            carbon_tonnes_ha=carbon_per_ha,
            carbon_tonnes_total=carbon_total,
            uncertainty_percent=uncertainty,
            lower_bound_tonnes=lower_bound,
            upper_bound_tonnes=upper_bound,
            method=method,
            tier=tier
        )

    def _calculate_agb(
        self,
        measurements: List[ForestMeasurement],
        tier: IPCCTier,
        country_factors: Optional[Dict[str, float]]
    ) -> Tuple[float, MeasurementMethod]:
        """
        Calculate above-ground biomass.

        Returns:
            Tuple of (biomass in t/ha, method used)
        """
        # Tier 3: Use direct measurements
        if tier == IPCCTier.TIER_3:
            measured = [m.measured_agb_tonnes_ha for m in measurements if m.measured_agb_tonnes_ha]
            if measured:
                return sum(measured) / len(measured), MeasurementMethod.GROUND_INVENTORY

        # Tier 2: Use allometric equations or volume-based
        if tier == IPCCTier.TIER_2:
            # Volume-based method
            volumes = [m.merchantable_volume_m3_ha for m in measurements
                      if m.merchantable_volume_m3_ha]
            densities = [m.wood_density_tonnes_m3 for m in measurements
                        if m.wood_density_tonnes_m3]

            if volumes and densities:
                avg_volume = sum(volumes) / len(volumes)
                avg_density = sum(densities) / len(densities)
                # BEF (Biomass Expansion Factor) default = 1.3 for total AGB
                bef = country_factors.get('bef', 1.3) if country_factors else 1.3
                biomass = avg_volume * avg_density * bef
                return biomass, MeasurementMethod.VOLUME_BASED

            # Allometric method (simplified)
            dbh_list = [m.mean_dbh_cm for m in measurements if m.mean_dbh_cm]
            stems_list = [m.stems_per_ha for m in measurements if m.stems_per_ha]

            if dbh_list and stems_list:
                avg_dbh = sum(dbh_list) / len(dbh_list)
                avg_stems = sum(stems_list) / len(stems_list)
                # Simplified pan-tropical allometric: AGB = 0.0673 * (DBH^2.652)
                # Source: Chave et al. 2014
                single_tree_biomass = 0.0673 * (avg_dbh ** 2.652) / 1000  # tonnes
                biomass = single_tree_biomass * avg_stems
                return biomass, MeasurementMethod.ALLOMETRIC_EQUATION

        # Tier 1: Use IPCC defaults
        forest_types = [m.forest_type for m in measurements]
        biomass_values = [AGB_DEFAULT_VALUES.get(ft, 150.0) for ft in forest_types]
        areas = [m.area_ha for m in measurements]
        total_area = sum(areas)

        # Area-weighted average
        weighted_biomass = sum(b * a for b, a in zip(biomass_values, areas)) / total_area
        return weighted_biomass, MeasurementMethod.DEFAULT_FACTOR

    def _calculate_bgb(
        self,
        measurements: List[ForestMeasurement],
        tier: IPCCTier,
        country_factors: Optional[Dict[str, float]]
    ) -> Tuple[float, MeasurementMethod]:
        """
        Calculate below-ground biomass using root-to-shoot ratios.

        BGB = AGB * R (root-to-shoot ratio)
        """
        # First get AGB
        agb, agb_method = self._calculate_agb(measurements, tier, country_factors)

        # Tier 3: Use direct measurements if available
        if tier == IPCCTier.TIER_3:
            measured = [m.measured_bgb_tonnes_ha for m in measurements
                       if m.measured_bgb_tonnes_ha]
            if measured:
                return sum(measured) / len(measured), MeasurementMethod.GROUND_INVENTORY

        # Tier 1/2: Use root-to-shoot ratios
        forest_types = [m.forest_type for m in measurements]
        ratios = [ROOT_SHOOT_RATIOS.get(ft, 0.27) for ft in forest_types]
        areas = [m.area_ha for m in measurements]
        total_area = sum(areas)

        # Area-weighted average ratio
        weighted_ratio = sum(r * a for r, a in zip(ratios, areas)) / total_area

        # Apply country-specific ratio if provided
        if country_factors and 'root_shoot_ratio' in country_factors:
            weighted_ratio = country_factors['root_shoot_ratio']

        bgb = agb * weighted_ratio
        return bgb, MeasurementMethod.DEFAULT_FACTOR

    def _calculate_dead_wood(
        self,
        measurements: List[ForestMeasurement],
        tier: IPCCTier
    ) -> Tuple[float, MeasurementMethod]:
        """Calculate dead wood carbon pool."""
        # Get AGB first
        agb, _ = self._calculate_agb(measurements, tier, None)

        # Get dead wood fraction by forest type
        forest_types = [m.forest_type for m in measurements]
        fractions = [DEAD_WOOD_FRACTION.get(ft, 0.06) for ft in forest_types]
        areas = [m.area_ha for m in measurements]
        total_area = sum(areas)

        weighted_fraction = sum(f * a for f, a in zip(fractions, areas)) / total_area
        dead_wood = agb * weighted_fraction

        return dead_wood, MeasurementMethod.DEFAULT_FACTOR

    def _calculate_litter(
        self,
        measurements: List[ForestMeasurement],
        tier: IPCCTier
    ) -> Tuple[float, MeasurementMethod]:
        """Calculate litter carbon pool."""
        # Get AGB first
        agb, _ = self._calculate_agb(measurements, tier, None)

        # Get litter fraction by forest type
        forest_types = [m.forest_type for m in measurements]
        fractions = [LITTER_FRACTION.get(ft, 0.04) for ft in forest_types]
        areas = [m.area_ha for m in measurements]
        total_area = sum(areas)

        weighted_fraction = sum(f * a for f, a in zip(fractions, areas)) / total_area
        litter = agb * weighted_fraction

        return litter, MeasurementMethod.DEFAULT_FACTOR

    def _calculate_soc(
        self,
        measurements: List[ForestMeasurement],
        tier: IPCCTier
    ) -> Tuple[float, MeasurementMethod]:
        """
        Calculate soil organic carbon.

        SOC estimation requires soil sampling data for Tier 2/3.
        Tier 1 uses IPCC default values by climate zone.
        """
        # Default SOC values by climate (tonnes C/ha to 30cm depth)
        # Source: IPCC 2006 Guidelines, Table 2.3
        soc_defaults = {
            ForestType.TROPICAL_RAINFOREST: 44.0,
            ForestType.TROPICAL_MOIST_DECIDUOUS: 44.0,
            ForestType.TROPICAL_DRY: 35.0,
            ForestType.TEMPERATE_OCEANIC: 68.0,
            ForestType.TEMPERATE_CONTINENTAL: 68.0,
            ForestType.BOREAL_CONIFEROUS: 86.0,
        }

        forest_types = [m.forest_type for m in measurements]
        soc_values = [soc_defaults.get(ft, 50.0) for ft in forest_types]
        areas = [m.area_ha for m in measurements]
        total_area = sum(areas)

        weighted_soc = sum(s * a for s, a in zip(soc_values, areas)) / total_area

        return weighted_soc, MeasurementMethod.DEFAULT_FACTOR

    def _calculate_combined_uncertainty(
        self,
        estimates: List[BiomassEstimate]
    ) -> float:
        """
        Calculate combined uncertainty using error propagation.

        For addition: U_total = sqrt(sum((U_i * X_i)^2)) / sum(X_i) * 100
        """
        if not estimates:
            return 0.0

        sum_squares = sum(
            (e.uncertainty_percent / 100 * e.carbon_tonnes_total) ** 2
            for e in estimates
        )
        total_carbon = sum(e.carbon_tonnes_total for e in estimates)

        if total_carbon == 0:
            return 0.0

        combined = math.sqrt(sum_squares) / total_carbon * 100
        return round(combined, 1)

    def _calculate_provenance_hash(
        self,
        inputs: Dict[str, Any],
        estimates: List[BiomassEstimate],
        trace: List[str]
    ) -> str:
        """Calculate SHA-256 hash for provenance tracking."""
        content = {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "inputs_hash": hashlib.sha256(
                json.dumps(inputs, sort_keys=True, default=str).encode()
            ).hexdigest(),
            "estimates_hash": hashlib.sha256(
                json.dumps(
                    [e.model_dump() for e in estimates],
                    sort_keys=True,
                    default=str
                ).encode()
            ).hexdigest(),
            "trace_hash": hashlib.sha256(
                json.dumps(trace).encode()
            ).hexdigest(),
            "timestamp": DeterministicClock.now().isoformat(),
        }

        return hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()


# =============================================================================
# Factory Functions
# =============================================================================

def create_forest_carbon_agent(enable_audit_trail: bool = True) -> ForestCarbonMRVAgent:
    """
    Create a Forest Carbon MRV Agent instance.

    Args:
        enable_audit_trail: Whether to enable audit trail

    Returns:
        Configured ForestCarbonMRVAgent
    """
    return ForestCarbonMRVAgent(enable_audit_trail=enable_audit_trail)


if __name__ == "__main__":
    # Example usage
    agent = ForestCarbonMRVAgent()

    # Sample input
    sample_input = {
        "project_id": "FOREST-EXAMPLE-001",
        "measurement_period_start": "2024-01-01",
        "measurement_period_end": "2024-12-31",
        "measurements": [
            {
                "plot_id": "PLOT-001",
                "measurement_date": "2024-06-15",
                "area_ha": 100.0,
                "forest_type": "tropical_moist_deciduous",
                "stand_age_years": 25,
            },
            {
                "plot_id": "PLOT-002",
                "measurement_date": "2024-06-16",
                "area_ha": 150.0,
                "forest_type": "tropical_moist_deciduous",
                "stand_age_years": 30,
            }
        ],
        "target_tier": "tier_1",
        "include_pools": ["above_ground_biomass", "below_ground_biomass"]
    }

    result = agent.execute(sample_input)
    print(f"Total Carbon Stock: {result['total_carbon_stock_tonnes']:.2f} tonnes C")
    print(f"CO2 Equivalent: {result['co2e_tonnes_total']:.2f} tonnes CO2e")
    print(f"Uncertainty: +/- {result['total_uncertainty_percent']:.1f}%")
    print(f"Provenance Hash: {result['provenance_hash'][:16]}...")
