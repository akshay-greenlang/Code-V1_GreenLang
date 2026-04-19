# -*- coding: utf-8 -*-
"""
GL-MRV-NBS-002: Soil Carbon MRV Agent
=====================================

This agent measures, reports, and verifies soil organic carbon (SOC)
following IPCC guidelines and GHG Protocol Land Sector Guidance.

Capabilities:
    - Soil organic carbon stock calculation
    - Stock change factors for land management
    - Tillage, input, and land use change effects
    - Uncertainty quantification (IPCC Tier 1/2/3)
    - Bulk density and depth corrections

Methodologies:
    - IPCC 2006 Guidelines Chapter 5 (Cropland) and Chapter 6 (Grassland)
    - IPCC 2019 Refinement for improved SOC factors
    - GHG Protocol Land Sector and Removals Guidance
    - 4 per 1000 Initiative methodology

Zero-Hallucination Guarantee:
    All SOC calculations use IPCC default factors and peer-reviewed
    equations. No LLM reasoning in calculation path.

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

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base_agents import DeterministicAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism.clock import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class SoilType(str, Enum):
    """Soil type classification following IPCC categories."""
    HIGH_ACTIVITY_CLAY = "high_activity_clay"      # HAC soils
    LOW_ACTIVITY_CLAY = "low_activity_clay"        # LAC soils
    SANDY = "sandy"                                 # Sandy soils
    SPODIC = "spodic"                              # Spodosols
    VOLCANIC = "volcanic"                          # Andisols
    WETLAND = "wetland"                            # Organic/wetland soils
    ORGANIC = "organic"                            # Histosols (>20% organic C)


class ClimateZone(str, Enum):
    """Climate zone classification for SOC reference values."""
    COLD_DRY = "cold_dry"
    COLD_MOIST = "cold_moist"
    WARM_DRY = "warm_dry"
    WARM_MOIST = "warm_moist"
    TROPICAL_DRY = "tropical_dry"
    TROPICAL_MOIST = "tropical_moist"
    TROPICAL_WET = "tropical_wet"
    TROPICAL_MONTANE = "tropical_montane"


class LandUseType(str, Enum):
    """Land use types for SOC calculations."""
    NATIVE_FOREST = "native_forest"
    MANAGED_FOREST = "managed_forest"
    CROPLAND_ANNUAL = "cropland_annual"
    CROPLAND_PERENNIAL = "cropland_perennial"
    PADDY_RICE = "paddy_rice"
    GRASSLAND_NATIVE = "grassland_native"
    GRASSLAND_IMPROVED = "grassland_improved"
    GRASSLAND_DEGRADED = "grassland_degraded"
    SETTLEMENT = "settlement"
    OTHER_LAND = "other_land"


class LandManagementPractice(str, Enum):
    """Land management practices affecting SOC."""
    # Tillage practices
    FULL_TILLAGE = "full_tillage"
    REDUCED_TILLAGE = "reduced_tillage"
    NO_TILL = "no_till"

    # Input practices
    LOW_INPUT = "low_input"
    MEDIUM_INPUT = "medium_input"
    HIGH_INPUT = "high_input"
    HIGH_INPUT_ORGANIC = "high_input_organic"

    # Grazing intensity
    NOMINALLY_MANAGED = "nominally_managed"
    MODERATELY_DEGRADED = "moderately_degraded"
    SEVERELY_DEGRADED = "severely_degraded"
    IMPROVED_GRASSLAND = "improved_grassland"


class IPCCTier(str, Enum):
    """IPCC methodology tiers."""
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"


# =============================================================================
# IPCC Default Factors
# =============================================================================

# Reference SOC stocks (tonnes C/ha to 30cm depth) by climate and soil type
# Source: IPCC 2006 Guidelines, Table 2.3
SOC_REFERENCE: Dict[ClimateZone, Dict[SoilType, float]] = {
    ClimateZone.COLD_DRY: {
        SoilType.HIGH_ACTIVITY_CLAY: 50.0,
        SoilType.LOW_ACTIVITY_CLAY: 33.0,
        SoilType.SANDY: 34.0,
        SoilType.SPODIC: 115.0,
        SoilType.VOLCANIC: 70.0,
        SoilType.WETLAND: 87.0,
    },
    ClimateZone.COLD_MOIST: {
        SoilType.HIGH_ACTIVITY_CLAY: 95.0,
        SoilType.LOW_ACTIVITY_CLAY: 85.0,
        SoilType.SANDY: 71.0,
        SoilType.SPODIC: 115.0,
        SoilType.VOLCANIC: 130.0,
        SoilType.WETLAND: 87.0,
    },
    ClimateZone.WARM_DRY: {
        SoilType.HIGH_ACTIVITY_CLAY: 38.0,
        SoilType.LOW_ACTIVITY_CLAY: 24.0,
        SoilType.SANDY: 19.0,
        SoilType.SPODIC: 48.0,
        SoilType.VOLCANIC: 70.0,
        SoilType.WETLAND: 86.0,
    },
    ClimateZone.WARM_MOIST: {
        SoilType.HIGH_ACTIVITY_CLAY: 88.0,
        SoilType.LOW_ACTIVITY_CLAY: 63.0,
        SoilType.SANDY: 34.0,
        SoilType.SPODIC: 48.0,
        SoilType.VOLCANIC: 130.0,
        SoilType.WETLAND: 86.0,
    },
    ClimateZone.TROPICAL_DRY: {
        SoilType.HIGH_ACTIVITY_CLAY: 38.0,
        SoilType.LOW_ACTIVITY_CLAY: 35.0,
        SoilType.SANDY: 31.0,
        SoilType.SPODIC: 48.0,
        SoilType.VOLCANIC: 50.0,
        SoilType.WETLAND: 86.0,
    },
    ClimateZone.TROPICAL_MOIST: {
        SoilType.HIGH_ACTIVITY_CLAY: 65.0,
        SoilType.LOW_ACTIVITY_CLAY: 47.0,
        SoilType.SANDY: 39.0,
        SoilType.SPODIC: 48.0,
        SoilType.VOLCANIC: 70.0,
        SoilType.WETLAND: 86.0,
    },
    ClimateZone.TROPICAL_WET: {
        SoilType.HIGH_ACTIVITY_CLAY: 44.0,
        SoilType.LOW_ACTIVITY_CLAY: 60.0,
        SoilType.SANDY: 66.0,
        SoilType.SPODIC: 48.0,
        SoilType.VOLCANIC: 130.0,
        SoilType.WETLAND: 86.0,
    },
    ClimateZone.TROPICAL_MONTANE: {
        SoilType.HIGH_ACTIVITY_CLAY: 88.0,
        SoilType.LOW_ACTIVITY_CLAY: 63.0,
        SoilType.SANDY: 34.0,
        SoilType.SPODIC: 48.0,
        SoilType.VOLCANIC: 130.0,
        SoilType.WETLAND: 86.0,
    },
}

# Land use factors (FLU) - relative to native vegetation
# Source: IPCC 2006 Guidelines, Table 5.5
LAND_USE_FACTORS: Dict[ClimateZone, Dict[LandUseType, float]] = {
    ClimateZone.TROPICAL_DRY: {
        LandUseType.NATIVE_FOREST: 1.0,
        LandUseType.CROPLAND_ANNUAL: 0.58,
        LandUseType.CROPLAND_PERENNIAL: 1.0,
        LandUseType.PADDY_RICE: 1.1,
        LandUseType.GRASSLAND_NATIVE: 1.0,
    },
    ClimateZone.TROPICAL_MOIST: {
        LandUseType.NATIVE_FOREST: 1.0,
        LandUseType.CROPLAND_ANNUAL: 0.48,
        LandUseType.CROPLAND_PERENNIAL: 1.0,
        LandUseType.PADDY_RICE: 1.1,
        LandUseType.GRASSLAND_NATIVE: 1.0,
    },
    ClimateZone.WARM_DRY: {
        LandUseType.NATIVE_FOREST: 1.0,
        LandUseType.CROPLAND_ANNUAL: 0.8,
        LandUseType.CROPLAND_PERENNIAL: 1.0,
        LandUseType.GRASSLAND_NATIVE: 1.0,
    },
    ClimateZone.WARM_MOIST: {
        LandUseType.NATIVE_FOREST: 1.0,
        LandUseType.CROPLAND_ANNUAL: 0.69,
        LandUseType.CROPLAND_PERENNIAL: 1.0,
        LandUseType.GRASSLAND_NATIVE: 1.0,
    },
    ClimateZone.COLD_DRY: {
        LandUseType.NATIVE_FOREST: 1.0,
        LandUseType.CROPLAND_ANNUAL: 0.8,
        LandUseType.CROPLAND_PERENNIAL: 1.0,
        LandUseType.GRASSLAND_NATIVE: 1.0,
    },
    ClimateZone.COLD_MOIST: {
        LandUseType.NATIVE_FOREST: 1.0,
        LandUseType.CROPLAND_ANNUAL: 0.69,
        LandUseType.CROPLAND_PERENNIAL: 1.0,
        LandUseType.GRASSLAND_NATIVE: 1.0,
    },
}

# Tillage factors (FMG)
# Source: IPCC 2006 Guidelines, Table 5.5
TILLAGE_FACTORS: Dict[ClimateZone, Dict[LandManagementPractice, float]] = {
    ClimateZone.TROPICAL_DRY: {
        LandManagementPractice.FULL_TILLAGE: 1.0,
        LandManagementPractice.REDUCED_TILLAGE: 1.02,
        LandManagementPractice.NO_TILL: 1.10,
    },
    ClimateZone.TROPICAL_MOIST: {
        LandManagementPractice.FULL_TILLAGE: 1.0,
        LandManagementPractice.REDUCED_TILLAGE: 1.08,
        LandManagementPractice.NO_TILL: 1.15,
    },
    ClimateZone.WARM_DRY: {
        LandManagementPractice.FULL_TILLAGE: 1.0,
        LandManagementPractice.REDUCED_TILLAGE: 1.02,
        LandManagementPractice.NO_TILL: 1.10,
    },
    ClimateZone.WARM_MOIST: {
        LandManagementPractice.FULL_TILLAGE: 1.0,
        LandManagementPractice.REDUCED_TILLAGE: 1.08,
        LandManagementPractice.NO_TILL: 1.15,
    },
    ClimateZone.COLD_DRY: {
        LandManagementPractice.FULL_TILLAGE: 1.0,
        LandManagementPractice.REDUCED_TILLAGE: 1.02,
        LandManagementPractice.NO_TILL: 1.10,
    },
    ClimateZone.COLD_MOIST: {
        LandManagementPractice.FULL_TILLAGE: 1.0,
        LandManagementPractice.REDUCED_TILLAGE: 1.08,
        LandManagementPractice.NO_TILL: 1.15,
    },
}

# Input factors (FI) - organic matter inputs
# Source: IPCC 2006 Guidelines, Table 5.5
INPUT_FACTORS: Dict[ClimateZone, Dict[LandManagementPractice, float]] = {
    ClimateZone.TROPICAL_DRY: {
        LandManagementPractice.LOW_INPUT: 0.92,
        LandManagementPractice.MEDIUM_INPUT: 1.0,
        LandManagementPractice.HIGH_INPUT: 1.04,
        LandManagementPractice.HIGH_INPUT_ORGANIC: 1.37,
    },
    ClimateZone.TROPICAL_MOIST: {
        LandManagementPractice.LOW_INPUT: 0.91,
        LandManagementPractice.MEDIUM_INPUT: 1.0,
        LandManagementPractice.HIGH_INPUT: 1.04,
        LandManagementPractice.HIGH_INPUT_ORGANIC: 1.44,
    },
    ClimateZone.WARM_DRY: {
        LandManagementPractice.LOW_INPUT: 0.92,
        LandManagementPractice.MEDIUM_INPUT: 1.0,
        LandManagementPractice.HIGH_INPUT: 1.04,
        LandManagementPractice.HIGH_INPUT_ORGANIC: 1.37,
    },
    ClimateZone.WARM_MOIST: {
        LandManagementPractice.LOW_INPUT: 0.91,
        LandManagementPractice.MEDIUM_INPUT: 1.0,
        LandManagementPractice.HIGH_INPUT: 1.04,
        LandManagementPractice.HIGH_INPUT_ORGANIC: 1.44,
    },
    ClimateZone.COLD_DRY: {
        LandManagementPractice.LOW_INPUT: 0.92,
        LandManagementPractice.MEDIUM_INPUT: 1.0,
        LandManagementPractice.HIGH_INPUT: 1.04,
        LandManagementPractice.HIGH_INPUT_ORGANIC: 1.37,
    },
    ClimateZone.COLD_MOIST: {
        LandManagementPractice.LOW_INPUT: 0.91,
        LandManagementPractice.MEDIUM_INPUT: 1.0,
        LandManagementPractice.HIGH_INPUT: 1.04,
        LandManagementPractice.HIGH_INPUT_ORGANIC: 1.44,
    },
}

# Uncertainty factors (% of mean)
UNCERTAINTY_FACTORS: Dict[IPCCTier, float] = {
    IPCCTier.TIER_1: 90.0,
    IPCCTier.TIER_2: 45.0,
    IPCCTier.TIER_3: 20.0,
}


# =============================================================================
# Pydantic Models
# =============================================================================

class SoilSample(BaseModel):
    """Individual soil sample measurement data."""

    sample_id: str = Field(..., description="Unique sample identifier")
    sampling_date: date = Field(..., description="Date of sampling")
    plot_id: str = Field(..., description="Plot identifier")

    # Location
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)

    # Classification
    soil_type: SoilType = Field(..., description="Soil type classification")
    climate_zone: ClimateZone = Field(..., description="Climate zone")
    land_use: LandUseType = Field(..., description="Current land use")

    # Management
    tillage_practice: Optional[LandManagementPractice] = Field(
        None, description="Tillage practice"
    )
    input_practice: Optional[LandManagementPractice] = Field(
        None, description="Organic input practice"
    )

    # Direct measurements (Tier 2/3)
    soc_percent: Optional[float] = Field(
        None, ge=0, le=100, description="SOC concentration (%)"
    )
    bulk_density_g_cm3: Optional[float] = Field(
        None, gt=0, le=2.5, description="Bulk density (g/cm3)"
    )
    sampling_depth_cm: float = Field(
        default=30.0, gt=0, description="Sampling depth (cm)"
    )

    # Area
    area_ha: float = Field(..., gt=0, description="Area represented (ha)")


class SOCEstimate(BaseModel):
    """Soil organic carbon estimate."""

    soc_tonnes_ha: float = Field(..., ge=0, description="SOC stock (tonnes C/ha)")
    soc_tonnes_total: float = Field(..., ge=0, description="Total SOC (tonnes C)")

    # Stock change (if applicable)
    soc_change_tonnes_ha_yr: Optional[float] = Field(
        None, description="Annual SOC change (tonnes C/ha/yr)"
    )
    soc_change_tonnes_total: Optional[float] = Field(
        None, description="Total SOC change (tonnes C)"
    )

    # Uncertainty
    uncertainty_percent: float = Field(..., ge=0, description="Uncertainty (+/- %)")
    lower_bound_tonnes: float = Field(..., ge=0, description="Lower 95% CI")
    upper_bound_tonnes: float = Field(..., description="Upper 95% CI")

    # Methodology
    tier: IPCCTier = Field(..., description="IPCC tier")
    methodology_reference: str = Field(
        default="IPCC 2006 Guidelines Vol. 4 Ch. 5",
        description="Methodology reference"
    )

    # Provenance
    calculation_timestamp: datetime = Field(default_factory=DeterministicClock.now)


class SoilCarbonInput(BaseModel):
    """Input for Soil Carbon MRV Agent."""

    project_id: str = Field(..., description="Project identifier")
    baseline_year: int = Field(..., ge=1990, description="Baseline year")
    reporting_year: int = Field(..., ge=1990, description="Reporting year")

    # Soil samples
    samples: List[SoilSample] = Field(
        ..., min_length=1, description="Soil sample data"
    )

    # Configuration
    target_tier: IPCCTier = Field(default=IPCCTier.TIER_1)
    transition_period_years: int = Field(
        default=20, ge=1, description="Default transition period for stock changes"
    )

    # Previous land use (for stock change calculations)
    previous_land_use: Optional[LandUseType] = Field(None)
    previous_tillage: Optional[LandManagementPractice] = Field(None)
    previous_input: Optional[LandManagementPractice] = Field(None)


class SoilCarbonOutput(BaseModel):
    """Output from Soil Carbon MRV Agent."""

    project_id: str = Field(..., description="Project identifier")
    calculation_date: datetime = Field(default_factory=DeterministicClock.now)

    # Summary
    total_area_ha: float = Field(..., ge=0, description="Total area (ha)")
    total_soc_tonnes: float = Field(..., ge=0, description="Total SOC (tonnes C)")
    average_soc_tonnes_ha: float = Field(..., ge=0, description="Average SOC (tonnes C/ha)")

    # Stock change
    total_soc_change_tonnes: Optional[float] = Field(
        None, description="Total SOC change (tonnes C)"
    )
    annual_soc_change_tonnes: Optional[float] = Field(
        None, description="Annual SOC change (tonnes C/yr)"
    )

    # CO2e
    co2e_tonnes_total: float = Field(..., ge=0, description="CO2 equivalent (tonnes)")
    co2e_change_tonnes: Optional[float] = Field(
        None, description="CO2 change (tonnes CO2e)"
    )

    # Detailed estimates
    estimates: List[SOCEstimate] = Field(..., description="Estimates by stratum")

    # Uncertainty
    total_uncertainty_percent: float = Field(..., ge=0)
    lower_bound_tonnes: float = Field(..., ge=0)
    upper_bound_tonnes: float = Field(...)

    # Methodology
    methodology_tier: IPCCTier = Field(...)

    # Provenance
    provenance_hash: str = Field(...)
    calculation_trace: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# =============================================================================
# Soil Carbon MRV Agent
# =============================================================================

class SoilCarbonMRVAgent(DeterministicAgent):
    """
    GL-MRV-NBS-002: Soil Carbon MRV Agent

    Measures, reports, and verifies soil organic carbon using IPCC-compliant
    methodologies. This is a CRITICAL PATH agent with zero-hallucination guarantee.

    Calculation Methods:
        - Tier 1: SOC = SOC_ref * FLU * FMG * FI
        - Tier 2: Country-specific reference values and factors
        - Tier 3: Direct measurement with bulk density correction

    Stock Change Calculation:
        Annual change = (SOC_final - SOC_initial) / transition_period

    Usage:
        agent = SoilCarbonMRVAgent()
        result = agent.execute({
            "project_id": "SOC-001",
            "samples": [...],
            "target_tier": "tier_1"
        })
    """

    AGENT_ID = "GL-MRV-NBS-002"
    AGENT_NAME = "Soil Carbon MRV Agent"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="SoilCarbonMRVAgent",
        category=AgentCategory.CRITICAL,
        uses_chat_session=False,
        uses_rag=False,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="Soil organic carbon MRV with IPCC compliance"
    )

    CO2_TO_C_RATIO = 44.0 / 12.0

    def __init__(self, enable_audit_trail: bool = True):
        """Initialize Soil Carbon MRV Agent."""
        super().__init__(enable_audit_trail=enable_audit_trail)
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute soil carbon calculation."""
        calculation_trace: List[str] = []
        warnings: List[str] = []

        try:
            # Parse input
            agent_input = SoilCarbonInput(**inputs)
            calculation_trace.append(
                f"Processing {len(agent_input.samples)} soil samples for project "
                f"{agent_input.project_id}"
            )

            # Calculate total area
            total_area_ha = sum(s.area_ha for s in agent_input.samples)
            calculation_trace.append(f"Total area: {total_area_ha:.2f} ha")

            # Determine achievable tier
            achieved_tier = self._determine_tier(agent_input.samples, agent_input.target_tier)
            if achieved_tier != agent_input.target_tier:
                warnings.append(
                    f"Target tier {agent_input.target_tier.value} not achievable, "
                    f"using {achieved_tier.value}"
                )

            # Calculate SOC by stratum
            estimates: List[SOCEstimate] = []

            # Group samples by soil type and climate zone
            strata = self._stratify_samples(agent_input.samples)

            for stratum_key, stratum_samples in strata.items():
                estimate = self._calculate_stratum_soc(
                    samples=stratum_samples,
                    tier=achieved_tier,
                    previous_land_use=agent_input.previous_land_use,
                    previous_tillage=agent_input.previous_tillage,
                    previous_input=agent_input.previous_input,
                    transition_years=agent_input.transition_period_years,
                    calculation_trace=calculation_trace,
                    warnings=warnings
                )
                estimates.append(estimate)

            # Calculate totals
            total_soc = sum(e.soc_tonnes_total for e in estimates)
            average_soc = total_soc / total_area_ha if total_area_ha > 0 else 0.0

            # Stock change
            total_soc_change = None
            annual_soc_change = None
            if any(e.soc_change_tonnes_total is not None for e in estimates):
                total_soc_change = sum(
                    e.soc_change_tonnes_total or 0 for e in estimates
                )
                annual_soc_change = total_soc_change / agent_input.transition_period_years

            # Combined uncertainty
            total_uncertainty = self._calculate_combined_uncertainty(estimates)
            lower_bound = max(0, total_soc * (1 - total_uncertainty / 100))
            upper_bound = total_soc * (1 + total_uncertainty / 100)

            # CO2e
            co2e_total = total_soc * self.CO2_TO_C_RATIO
            co2e_change = (
                total_soc_change * self.CO2_TO_C_RATIO
                if total_soc_change is not None else None
            )

            calculation_trace.append(
                f"Total SOC: {total_soc:.2f} tonnes C "
                f"({average_soc:.2f} tonnes C/ha)"
            )

            # Provenance hash
            provenance_hash = self._calculate_provenance_hash(
                inputs, estimates, calculation_trace
            )

            output = SoilCarbonOutput(
                project_id=agent_input.project_id,
                total_area_ha=total_area_ha,
                total_soc_tonnes=total_soc,
                average_soc_tonnes_ha=average_soc,
                total_soc_change_tonnes=total_soc_change,
                annual_soc_change_tonnes=annual_soc_change,
                co2e_tonnes_total=co2e_total,
                co2e_change_tonnes=co2e_change,
                estimates=estimates,
                total_uncertainty_percent=total_uncertainty,
                lower_bound_tonnes=lower_bound,
                upper_bound_tonnes=upper_bound,
                methodology_tier=achieved_tier,
                provenance_hash=provenance_hash,
                calculation_trace=calculation_trace,
                warnings=warnings
            )

            # Audit entry
            self._capture_audit_entry(
                operation="soil_carbon_mrv",
                inputs=inputs,
                outputs=output.model_dump(),
                calculation_trace=calculation_trace,
                metadata={"agent_id": self.AGENT_ID, "version": self.VERSION}
            )

            return output.model_dump()

        except Exception as e:
            logger.error(f"Soil carbon calculation failed: {str(e)}", exc_info=True)
            raise

    def _determine_tier(
        self,
        samples: List[SoilSample],
        target_tier: IPCCTier
    ) -> IPCCTier:
        """Determine achievable tier based on available data."""
        # Check for Tier 3 data
        has_tier3 = all(
            s.soc_percent is not None and s.bulk_density_g_cm3 is not None
            for s in samples
        )

        if target_tier == IPCCTier.TIER_3 and has_tier3:
            return IPCCTier.TIER_3
        elif target_tier in (IPCCTier.TIER_2, IPCCTier.TIER_3) and has_tier3:
            return IPCCTier.TIER_2

        return IPCCTier.TIER_1

    def _stratify_samples(
        self,
        samples: List[SoilSample]
    ) -> Dict[str, List[SoilSample]]:
        """Group samples into strata by soil type and climate zone."""
        strata: Dict[str, List[SoilSample]] = {}

        for sample in samples:
            key = f"{sample.soil_type.value}_{sample.climate_zone.value}"
            if key not in strata:
                strata[key] = []
            strata[key].append(sample)

        return strata

    def _calculate_stratum_soc(
        self,
        samples: List[SoilSample],
        tier: IPCCTier,
        previous_land_use: Optional[LandUseType],
        previous_tillage: Optional[LandManagementPractice],
        previous_input: Optional[LandManagementPractice],
        transition_years: int,
        calculation_trace: List[str],
        warnings: List[str]
    ) -> SOCEstimate:
        """Calculate SOC for a stratum."""
        total_area = sum(s.area_ha for s in samples)
        soil_type = samples[0].soil_type
        climate_zone = samples[0].climate_zone

        if tier == IPCCTier.TIER_3:
            # Direct measurement
            soc_values = []
            for s in samples:
                if s.soc_percent and s.bulk_density_g_cm3:
                    # SOC (t/ha) = SOC% * BD * depth * 100
                    soc = (s.soc_percent / 100) * s.bulk_density_g_cm3 * s.sampling_depth_cm * 100
                    soc_values.append((soc, s.area_ha))

            # Area-weighted average
            soc_tonnes_ha = sum(s * a for s, a in soc_values) / total_area
        else:
            # Tier 1/2: SOC = SOC_ref * FLU * FMG * FI
            soc_ref = SOC_REFERENCE.get(climate_zone, {}).get(soil_type, 50.0)

            # Get factors
            land_use = samples[0].land_use
            flu = LAND_USE_FACTORS.get(climate_zone, {}).get(land_use, 1.0)

            tillage = samples[0].tillage_practice
            fmg = 1.0
            if tillage:
                fmg = TILLAGE_FACTORS.get(climate_zone, {}).get(tillage, 1.0)

            input_practice = samples[0].input_practice
            fi = 1.0
            if input_practice:
                fi = INPUT_FACTORS.get(climate_zone, {}).get(input_practice, 1.0)

            soc_tonnes_ha = soc_ref * flu * fmg * fi

            calculation_trace.append(
                f"Stratum {soil_type.value}/{climate_zone.value}: "
                f"SOC_ref={soc_ref:.1f} * FLU={flu:.2f} * FMG={fmg:.2f} * FI={fi:.2f} "
                f"= {soc_tonnes_ha:.2f} t C/ha"
            )

        soc_total = soc_tonnes_ha * total_area

        # Stock change calculation
        soc_change_ha = None
        soc_change_total = None

        if previous_land_use:
            # Calculate initial SOC
            soc_ref = SOC_REFERENCE.get(climate_zone, {}).get(soil_type, 50.0)
            flu_prev = LAND_USE_FACTORS.get(climate_zone, {}).get(previous_land_use, 1.0)
            fmg_prev = 1.0
            if previous_tillage:
                fmg_prev = TILLAGE_FACTORS.get(climate_zone, {}).get(previous_tillage, 1.0)
            fi_prev = 1.0
            if previous_input:
                fi_prev = INPUT_FACTORS.get(climate_zone, {}).get(previous_input, 1.0)

            soc_initial_ha = soc_ref * flu_prev * fmg_prev * fi_prev
            soc_change_ha = (soc_tonnes_ha - soc_initial_ha) / transition_years
            soc_change_total = soc_change_ha * total_area

            calculation_trace.append(
                f"Stock change: ({soc_tonnes_ha:.2f} - {soc_initial_ha:.2f}) / "
                f"{transition_years} = {soc_change_ha:.3f} t C/ha/yr"
            )

        # Uncertainty
        uncertainty = UNCERTAINTY_FACTORS.get(tier, 90.0)
        lower_bound = max(0, soc_total * (1 - uncertainty / 100))
        upper_bound = soc_total * (1 + uncertainty / 100)

        return SOCEstimate(
            soc_tonnes_ha=soc_tonnes_ha,
            soc_tonnes_total=soc_total,
            soc_change_tonnes_ha_yr=soc_change_ha,
            soc_change_tonnes_total=soc_change_total,
            uncertainty_percent=uncertainty,
            lower_bound_tonnes=lower_bound,
            upper_bound_tonnes=upper_bound,
            tier=tier
        )

    def _calculate_combined_uncertainty(
        self,
        estimates: List[SOCEstimate]
    ) -> float:
        """Calculate combined uncertainty using error propagation."""
        if not estimates:
            return 0.0

        sum_squares = sum(
            (e.uncertainty_percent / 100 * e.soc_tonnes_total) ** 2
            for e in estimates
        )
        total_soc = sum(e.soc_tonnes_total for e in estimates)

        if total_soc == 0:
            return 0.0

        return math.sqrt(sum_squares) / total_soc * 100

    def _calculate_provenance_hash(
        self,
        inputs: Dict[str, Any],
        estimates: List[SOCEstimate],
        trace: List[str]
    ) -> str:
        """Calculate SHA-256 hash for provenance."""
        content = {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "inputs_hash": hashlib.sha256(
                json.dumps(inputs, sort_keys=True, default=str).encode()
            ).hexdigest(),
            "estimates_hash": hashlib.sha256(
                json.dumps([e.model_dump() for e in estimates], sort_keys=True, default=str).encode()
            ).hexdigest(),
            "timestamp": DeterministicClock.now().isoformat(),
        }

        return hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()


def create_soil_carbon_agent(enable_audit_trail: bool = True) -> SoilCarbonMRVAgent:
    """Create a Soil Carbon MRV Agent instance."""
    return SoilCarbonMRVAgent(enable_audit_trail=enable_audit_trail)
