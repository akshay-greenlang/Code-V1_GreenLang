# -*- coding: utf-8 -*-
"""
GL-MRV-NBS-003: Wetland Carbon MRV Agent
========================================

This agent measures, reports, and verifies wetland and peatland carbon
following IPCC Wetlands Supplement and GHG Protocol guidance.

Capabilities:
    - Peatland carbon stock estimation
    - Wetland GHG emissions (CO2, CH4, N2O)
    - Drainage and rewetting effects
    - Land use change on organic soils
    - Dissolved organic carbon (DOC) exports

Methodologies:
    - IPCC 2013 Wetlands Supplement
    - IPCC 2006 Guidelines (organic soils)
    - Ramsar Convention wetland classification

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

class WetlandType(str, Enum):
    """Wetland type classification per IPCC Wetlands Supplement."""
    PEATLAND_NATURAL = "peatland_natural"
    PEATLAND_DRAINED_CROPLAND = "peatland_drained_cropland"
    PEATLAND_DRAINED_GRASSLAND = "peatland_drained_grassland"
    PEATLAND_DRAINED_FOREST = "peatland_drained_forest"
    PEATLAND_REWETTED = "peatland_rewetted"
    PEATLAND_EXTRACTED = "peatland_extracted"
    FRESHWATER_MARSH = "freshwater_marsh"
    SALTWATER_MARSH = "saltwater_marsh"
    SWAMP_FOREST = "swamp_forest"
    BOG = "bog"
    FEN = "fen"
    FLOODED_LAND = "flooded_land"


class PeatlandCondition(str, Enum):
    """Peatland hydrological condition."""
    PRISTINE = "pristine"
    DEGRADED = "degraded"
    DRAINED = "drained"
    REWETTED = "rewetted"
    FLOODED = "flooded"


class ClimateRegion(str, Enum):
    """Climate region for wetland emission factors."""
    BOREAL = "boreal"
    TEMPERATE = "temperate"
    TROPICAL = "tropical"


class IPCCTier(str, Enum):
    """IPCC methodology tiers."""
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"


# =============================================================================
# IPCC Default Emission Factors (from 2013 Wetlands Supplement)
# =============================================================================

# CO2 emissions from drained organic soils (tonnes CO2/ha/yr)
DRAINED_PEAT_CO2_EF: Dict[ClimateRegion, Dict[str, float]] = {
    ClimateRegion.BOREAL: {
        "cropland": 7.9,
        "grassland": 5.7,
        "forest": 2.6,
        "extracted": 10.3,
    },
    ClimateRegion.TEMPERATE: {
        "cropland": 7.9,
        "grassland": 6.1,
        "forest": 2.6,
        "extracted": 10.3,
    },
    ClimateRegion.TROPICAL: {
        "cropland": 14.0,
        "grassland": 9.6,
        "forest": 5.3,
        "extracted": 15.0,
    },
}

# CH4 emissions from wetlands (kg CH4/ha/yr)
WETLAND_CH4_EF: Dict[WetlandType, Dict[ClimateRegion, float]] = {
    WetlandType.PEATLAND_NATURAL: {
        ClimateRegion.BOREAL: 50.0,
        ClimateRegion.TEMPERATE: 80.0,
        ClimateRegion.TROPICAL: 130.0,
    },
    WetlandType.FRESHWATER_MARSH: {
        ClimateRegion.BOREAL: 100.0,
        ClimateRegion.TEMPERATE: 200.0,
        ClimateRegion.TROPICAL: 350.0,
    },
    WetlandType.PEATLAND_REWETTED: {
        ClimateRegion.BOREAL: 200.0,
        ClimateRegion.TEMPERATE: 250.0,
        ClimateRegion.TROPICAL: 400.0,
    },
}

# Peat carbon content (tonnes C/ha per 10cm depth)
PEAT_CARBON_DENSITY: Dict[ClimateRegion, float] = {
    ClimateRegion.BOREAL: 52.0,
    ClimateRegion.TEMPERATE: 55.0,
    ClimateRegion.TROPICAL: 60.0,
}

# GWP values for CH4 and N2O (AR6)
GWP_CH4_AR6 = 27.9
GWP_N2O_AR6 = 273.0


# =============================================================================
# Pydantic Models
# =============================================================================

class WetlandMeasurement(BaseModel):
    """Wetland measurement data."""

    site_id: str = Field(..., description="Site identifier")
    measurement_date: date = Field(..., description="Measurement date")
    area_ha: float = Field(..., gt=0, description="Area in hectares")

    # Classification
    wetland_type: WetlandType = Field(..., description="Wetland type")
    climate_region: ClimateRegion = Field(..., description="Climate region")
    condition: PeatlandCondition = Field(..., description="Hydrological condition")

    # Peat characteristics
    peat_depth_m: Optional[float] = Field(None, ge=0, description="Peat depth (m)")
    water_table_depth_cm: Optional[float] = Field(
        None, description="Water table depth below surface (cm)"
    )
    bulk_density_g_cm3: Optional[float] = Field(None, gt=0, le=2.0)
    carbon_content_percent: Optional[float] = Field(None, ge=0, le=100)

    # Direct flux measurements (Tier 2/3)
    measured_co2_flux_tonnes_ha_yr: Optional[float] = Field(None)
    measured_ch4_flux_kg_ha_yr: Optional[float] = Field(None)
    measured_n2o_flux_kg_ha_yr: Optional[float] = Field(None)

    # DOC export
    doc_export_tonnes_c_ha_yr: Optional[float] = Field(None, ge=0)


class WetlandCarbonEstimate(BaseModel):
    """Wetland carbon/GHG estimate."""

    # Carbon stocks
    peat_carbon_stock_tonnes: Optional[float] = Field(None, ge=0)
    peat_carbon_density_tonnes_ha: Optional[float] = Field(None, ge=0)

    # Annual fluxes (as CO2e)
    co2_emissions_tonnes_yr: float = Field(..., description="CO2 emissions")
    ch4_emissions_co2e_tonnes_yr: float = Field(..., description="CH4 as CO2e")
    n2o_emissions_co2e_tonnes_yr: float = Field(default=0.0, description="N2O as CO2e")
    doc_export_co2e_tonnes_yr: float = Field(default=0.0, description="DOC as CO2e")

    # Total
    total_ghg_co2e_tonnes_yr: float = Field(..., description="Total GHG (CO2e/yr)")
    total_ghg_per_ha_co2e_yr: float = Field(..., description="GHG intensity (CO2e/ha/yr)")

    # Net balance (negative = sink)
    net_ghg_balance_co2e_tonnes_yr: float = Field(..., description="Net GHG balance")

    # Uncertainty
    uncertainty_percent: float = Field(..., ge=0)
    lower_bound_co2e: float = Field(...)
    upper_bound_co2e: float = Field(...)

    # Methodology
    tier: IPCCTier = Field(...)
    methodology_reference: str = Field(
        default="IPCC 2013 Wetlands Supplement"
    )

    calculation_timestamp: datetime = Field(default_factory=DeterministicClock.now)


class WetlandCarbonInput(BaseModel):
    """Input for Wetland Carbon MRV Agent."""

    project_id: str = Field(..., description="Project identifier")
    reporting_year: int = Field(..., ge=1990)

    measurements: List[WetlandMeasurement] = Field(..., min_length=1)
    target_tier: IPCCTier = Field(default=IPCCTier.TIER_1)

    # GWP selection
    gwp_ch4: float = Field(default=GWP_CH4_AR6, description="GWP for CH4")
    gwp_n2o: float = Field(default=GWP_N2O_AR6, description="GWP for N2O")


class WetlandCarbonOutput(BaseModel):
    """Output from Wetland Carbon MRV Agent."""

    project_id: str = Field(...)
    calculation_date: datetime = Field(default_factory=DeterministicClock.now)

    # Summary
    total_area_ha: float = Field(..., ge=0)
    total_peat_carbon_tonnes: Optional[float] = Field(None)
    total_ghg_co2e_tonnes_yr: float = Field(...)
    ghg_intensity_co2e_ha_yr: float = Field(...)

    # By source
    co2_emissions_tonnes_yr: float = Field(...)
    ch4_emissions_co2e_tonnes_yr: float = Field(...)
    n2o_emissions_co2e_tonnes_yr: float = Field(...)
    doc_export_co2e_tonnes_yr: float = Field(...)

    # Net balance
    net_ghg_balance_co2e_tonnes_yr: float = Field(...)

    # Estimates
    estimates: List[WetlandCarbonEstimate] = Field(...)

    # Uncertainty
    total_uncertainty_percent: float = Field(...)
    lower_bound_co2e: float = Field(...)
    upper_bound_co2e: float = Field(...)

    # Methodology
    methodology_tier: IPCCTier = Field(...)
    provenance_hash: str = Field(...)
    calculation_trace: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# =============================================================================
# Wetland Carbon MRV Agent
# =============================================================================

class WetlandCarbonMRVAgent(DeterministicAgent):
    """
    GL-MRV-NBS-003: Wetland Carbon MRV Agent

    Measures, reports, and verifies wetland/peatland carbon and GHG emissions.
    CRITICAL PATH agent with zero-hallucination guarantee.

    Key Features:
        - Peatland carbon stock estimation
        - CO2, CH4, N2O emission calculations
        - Drainage and rewetting effects
        - DOC export accounting

    Usage:
        agent = WetlandCarbonMRVAgent()
        result = agent.execute({
            "project_id": "WETLAND-001",
            "measurements": [...],
        })
    """

    AGENT_ID = "GL-MRV-NBS-003"
    AGENT_NAME = "Wetland Carbon MRV Agent"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="WetlandCarbonMRVAgent",
        category=AgentCategory.CRITICAL,
        uses_chat_session=False,
        uses_rag=False,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="Wetland/peatland carbon MRV with IPCC compliance"
    )

    CO2_TO_C_RATIO = 44.0 / 12.0

    def __init__(self, enable_audit_trail: bool = True):
        """Initialize Wetland Carbon MRV Agent."""
        super().__init__(enable_audit_trail=enable_audit_trail)
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute wetland carbon calculation."""
        calculation_trace: List[str] = []
        warnings: List[str] = []

        try:
            agent_input = WetlandCarbonInput(**inputs)
            calculation_trace.append(
                f"Processing {len(agent_input.measurements)} wetland sites"
            )

            total_area = sum(m.area_ha for m in agent_input.measurements)
            estimates: List[WetlandCarbonEstimate] = []

            for measurement in agent_input.measurements:
                estimate = self._calculate_site_ghg(
                    measurement=measurement,
                    tier=agent_input.target_tier,
                    gwp_ch4=agent_input.gwp_ch4,
                    gwp_n2o=agent_input.gwp_n2o,
                    calculation_trace=calculation_trace,
                    warnings=warnings
                )
                estimates.append(estimate)

            # Aggregate totals
            total_co2 = sum(e.co2_emissions_tonnes_yr for e in estimates)
            total_ch4_co2e = sum(e.ch4_emissions_co2e_tonnes_yr for e in estimates)
            total_n2o_co2e = sum(e.n2o_emissions_co2e_tonnes_yr for e in estimates)
            total_doc_co2e = sum(e.doc_export_co2e_tonnes_yr for e in estimates)
            total_ghg = sum(e.total_ghg_co2e_tonnes_yr for e in estimates)
            net_balance = sum(e.net_ghg_balance_co2e_tonnes_yr for e in estimates)

            total_peat_carbon = sum(
                e.peat_carbon_stock_tonnes or 0 for e in estimates
            ) or None

            ghg_intensity = total_ghg / total_area if total_area > 0 else 0.0

            # Combined uncertainty
            total_uncertainty = self._calculate_combined_uncertainty(estimates)
            lower_bound = total_ghg * (1 - total_uncertainty / 100)
            upper_bound = total_ghg * (1 + total_uncertainty / 100)

            provenance_hash = self._calculate_provenance_hash(
                inputs, estimates, calculation_trace
            )

            output = WetlandCarbonOutput(
                project_id=agent_input.project_id,
                total_area_ha=total_area,
                total_peat_carbon_tonnes=total_peat_carbon,
                total_ghg_co2e_tonnes_yr=total_ghg,
                ghg_intensity_co2e_ha_yr=ghg_intensity,
                co2_emissions_tonnes_yr=total_co2,
                ch4_emissions_co2e_tonnes_yr=total_ch4_co2e,
                n2o_emissions_co2e_tonnes_yr=total_n2o_co2e,
                doc_export_co2e_tonnes_yr=total_doc_co2e,
                net_ghg_balance_co2e_tonnes_yr=net_balance,
                estimates=estimates,
                total_uncertainty_percent=total_uncertainty,
                lower_bound_co2e=lower_bound,
                upper_bound_co2e=upper_bound,
                methodology_tier=agent_input.target_tier,
                provenance_hash=provenance_hash,
                calculation_trace=calculation_trace,
                warnings=warnings
            )

            self._capture_audit_entry(
                operation="wetland_carbon_mrv",
                inputs=inputs,
                outputs=output.model_dump(),
                calculation_trace=calculation_trace,
                metadata={"agent_id": self.AGENT_ID, "version": self.VERSION}
            )

            return output.model_dump()

        except Exception as e:
            logger.error(f"Wetland carbon calculation failed: {str(e)}", exc_info=True)
            raise

    def _calculate_site_ghg(
        self,
        measurement: WetlandMeasurement,
        tier: IPCCTier,
        gwp_ch4: float,
        gwp_n2o: float,
        calculation_trace: List[str],
        warnings: List[str]
    ) -> WetlandCarbonEstimate:
        """Calculate GHG emissions for a wetland site."""

        area = measurement.area_ha
        region = measurement.climate_region
        w_type = measurement.wetland_type

        # Peat carbon stock
        peat_carbon_stock = None
        peat_carbon_density = None
        if measurement.peat_depth_m:
            carbon_per_10cm = PEAT_CARBON_DENSITY.get(region, 55.0)
            peat_carbon_density = carbon_per_10cm * (measurement.peat_depth_m * 10)
            peat_carbon_stock = peat_carbon_density * area

            calculation_trace.append(
                f"Site {measurement.site_id}: Peat stock = "
                f"{peat_carbon_density:.1f} t C/ha x {area:.1f} ha = "
                f"{peat_carbon_stock:.1f} t C"
            )

        # CO2 emissions
        if tier in (IPCCTier.TIER_2, IPCCTier.TIER_3) and measurement.measured_co2_flux_tonnes_ha_yr:
            co2_per_ha = measurement.measured_co2_flux_tonnes_ha_yr
        else:
            co2_per_ha = self._get_co2_emission_factor(w_type, region, measurement.condition)

        co2_total = co2_per_ha * area

        # CH4 emissions
        if tier in (IPCCTier.TIER_2, IPCCTier.TIER_3) and measurement.measured_ch4_flux_kg_ha_yr:
            ch4_kg_per_ha = measurement.measured_ch4_flux_kg_ha_yr
        else:
            ch4_kg_per_ha = WETLAND_CH4_EF.get(w_type, {}).get(region, 100.0)

        ch4_co2e_total = (ch4_kg_per_ha / 1000) * gwp_ch4 * area

        # N2O (typically small for wetlands, use default or measured)
        n2o_co2e_total = 0.0
        if measurement.measured_n2o_flux_kg_ha_yr:
            n2o_co2e_total = (measurement.measured_n2o_flux_kg_ha_yr / 1000) * gwp_n2o * area

        # DOC export
        doc_co2e_total = 0.0
        if measurement.doc_export_tonnes_c_ha_yr:
            doc_co2e_total = measurement.doc_export_tonnes_c_ha_yr * self.CO2_TO_C_RATIO * area

        # Total GHG
        total_ghg = co2_total + ch4_co2e_total + n2o_co2e_total + doc_co2e_total
        ghg_per_ha = total_ghg / area if area > 0 else 0.0

        # Net balance (drained peatlands are sources, natural wetlands can be sinks)
        net_balance = total_ghg
        if measurement.condition == PeatlandCondition.PRISTINE:
            # Natural wetlands have some carbon uptake
            uptake_estimate = 2.0 * area  # ~2 t CO2/ha/yr uptake
            net_balance = total_ghg - uptake_estimate

        calculation_trace.append(
            f"Site {measurement.site_id}: CO2={co2_total:.1f}, "
            f"CH4={ch4_co2e_total:.1f}, Total={total_ghg:.1f} t CO2e/yr"
        )

        # Uncertainty
        uncertainty = {
            IPCCTier.TIER_1: 90.0,
            IPCCTier.TIER_2: 50.0,
            IPCCTier.TIER_3: 25.0,
        }.get(tier, 90.0)

        return WetlandCarbonEstimate(
            peat_carbon_stock_tonnes=peat_carbon_stock,
            peat_carbon_density_tonnes_ha=peat_carbon_density,
            co2_emissions_tonnes_yr=co2_total,
            ch4_emissions_co2e_tonnes_yr=ch4_co2e_total,
            n2o_emissions_co2e_tonnes_yr=n2o_co2e_total,
            doc_export_co2e_tonnes_yr=doc_co2e_total,
            total_ghg_co2e_tonnes_yr=total_ghg,
            total_ghg_per_ha_co2e_yr=ghg_per_ha,
            net_ghg_balance_co2e_tonnes_yr=net_balance,
            uncertainty_percent=uncertainty,
            lower_bound_co2e=total_ghg * (1 - uncertainty / 100),
            upper_bound_co2e=total_ghg * (1 + uncertainty / 100),
            tier=tier
        )

    def _get_co2_emission_factor(
        self,
        w_type: WetlandType,
        region: ClimateRegion,
        condition: PeatlandCondition
    ) -> float:
        """Get CO2 emission factor based on wetland type and condition."""

        # For drained peatlands, use IPCC defaults
        if condition == PeatlandCondition.DRAINED:
            land_use_map = {
                WetlandType.PEATLAND_DRAINED_CROPLAND: "cropland",
                WetlandType.PEATLAND_DRAINED_GRASSLAND: "grassland",
                WetlandType.PEATLAND_DRAINED_FOREST: "forest",
                WetlandType.PEATLAND_EXTRACTED: "extracted",
            }
            land_use = land_use_map.get(w_type, "cropland")
            return DRAINED_PEAT_CO2_EF.get(region, {}).get(land_use, 7.9)

        # Pristine peatlands are near carbon-neutral or slight sinks
        if condition == PeatlandCondition.PRISTINE:
            return 0.0

        # Rewetted peatlands have reduced emissions
        if condition == PeatlandCondition.REWETTED:
            base_ef = DRAINED_PEAT_CO2_EF.get(region, {}).get("cropland", 7.9)
            return base_ef * 0.2  # ~80% reduction

        return 5.0  # Default for degraded

    def _calculate_combined_uncertainty(
        self,
        estimates: List[WetlandCarbonEstimate]
    ) -> float:
        """Calculate combined uncertainty."""
        if not estimates:
            return 0.0

        sum_squares = sum(
            (e.uncertainty_percent / 100 * e.total_ghg_co2e_tonnes_yr) ** 2
            for e in estimates
        )
        total_ghg = sum(e.total_ghg_co2e_tonnes_yr for e in estimates)

        if total_ghg == 0:
            return 90.0

        return math.sqrt(sum_squares) / abs(total_ghg) * 100

    def _calculate_provenance_hash(
        self,
        inputs: Dict[str, Any],
        estimates: List[WetlandCarbonEstimate],
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


def create_wetland_carbon_agent(enable_audit_trail: bool = True) -> WetlandCarbonMRVAgent:
    """Create a Wetland Carbon MRV Agent instance."""
    return WetlandCarbonMRVAgent(enable_audit_trail=enable_audit_trail)
