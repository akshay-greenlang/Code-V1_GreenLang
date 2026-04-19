# -*- coding: utf-8 -*-
"""
MRV Energy Sector - Common Schemas and Data Models

This module defines Pydantic models shared across all MRV Energy agents.
All models include comprehensive validation, units tracking, and
provenance support for regulatory compliance.

Standards Reference:
    - GHG Protocol: Corporate Standard, Scope 2 Guidance
    - EPA Part 98: Subpart D (Electricity Generation)
    - ISO 14064-1: GHG accounting and verification
    - EU ETS: Monitoring and Reporting Regulation
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class FuelType(str, Enum):
    """Standard fuel types for power generation."""
    NATURAL_GAS = "natural_gas"
    COAL_BITUMINOUS = "coal_bituminous"
    COAL_SUBBITUMINOUS = "coal_subbituminous"
    COAL_LIGNITE = "coal_lignite"
    COAL_ANTHRACITE = "coal_anthracite"
    FUEL_OIL_NO2 = "fuel_oil_no2"
    FUEL_OIL_NO6 = "fuel_oil_no6"
    DIESEL = "diesel"
    LPG = "lpg"
    HYDROGEN = "hydrogen"
    BIOMASS_WOOD = "biomass_wood"
    BIOMASS_BIOGAS = "biomass_biogas"
    MUNICIPAL_SOLID_WASTE = "msw"
    LANDFILL_GAS = "landfill_gas"


class GenerationType(str, Enum):
    """Power generation technology types."""
    CCGT = "combined_cycle_gas_turbine"
    OCGT = "open_cycle_gas_turbine"
    COAL_SUBCRITICAL = "coal_subcritical"
    COAL_SUPERCRITICAL = "coal_supercritical"
    COAL_USC = "coal_ultra_supercritical"
    NUCLEAR_PWR = "nuclear_pwr"
    NUCLEAR_BWR = "nuclear_bwr"
    HYDRO_RUN_OF_RIVER = "hydro_ror"
    HYDRO_RESERVOIR = "hydro_reservoir"
    HYDRO_PUMPED = "hydro_pumped"
    SOLAR_PV_UTILITY = "solar_pv_utility"
    SOLAR_PV_ROOFTOP = "solar_pv_rooftop"
    SOLAR_CSP = "solar_csp"
    WIND_ONSHORE = "wind_onshore"
    WIND_OFFSHORE = "wind_offshore"
    GEOTHERMAL = "geothermal"
    BIOMASS = "biomass"
    CHP_GAS = "chp_gas"
    CHP_COAL = "chp_coal"
    CHP_BIOMASS = "chp_biomass"


class EmissionScope(str, Enum):
    """GHG Protocol emission scopes."""
    SCOPE_1 = "scope_1"
    SCOPE_2_LOCATION = "scope_2_location"
    SCOPE_2_MARKET = "scope_2_market"
    SCOPE_3_UPSTREAM = "scope_3_upstream"
    SCOPE_3_DOWNSTREAM = "scope_3_downstream"


class HydrogenProductionMethod(str, Enum):
    """Hydrogen production pathways."""
    SMR = "steam_methane_reforming"
    SMR_CCS = "steam_methane_reforming_with_ccs"
    ATR = "autothermal_reforming"
    ATR_CCS = "autothermal_reforming_with_ccs"
    COAL_GASIFICATION = "coal_gasification"
    COAL_GASIFICATION_CCS = "coal_gasification_with_ccs"
    ELECTROLYSIS_GRID = "electrolysis_grid"
    ELECTROLYSIS_RENEWABLE = "electrolysis_renewable"
    ELECTROLYSIS_NUCLEAR = "electrolysis_nuclear"
    BIOMASS_GASIFICATION = "biomass_gasification"


class StorageTechnology(str, Enum):
    """Energy storage technologies."""
    LITHIUM_ION_NMC = "li_ion_nmc"
    LITHIUM_ION_LFP = "li_ion_lfp"
    LITHIUM_ION_NCA = "li_ion_nca"
    LEAD_ACID = "lead_acid"
    FLOW_VANADIUM = "flow_vanadium"
    FLOW_ZINC_BROMINE = "flow_zinc_bromine"
    SODIUM_SULFUR = "sodium_sulfur"
    PUMPED_HYDRO = "pumped_hydro"
    COMPRESSED_AIR = "caes"
    FLYWHEEL = "flywheel"
    HYDROGEN = "hydrogen_storage"


class GridRegion(str, Enum):
    """Grid regions for emission factor lookups."""
    # US eGRID regions
    US_WECC = "us_wecc"
    US_MROE = "us_mroe"
    US_NPCC = "us_npcc"
    US_RFCE = "us_rfce"
    US_RFCM = "us_rfcm"
    US_RFCW = "us_rfcw"
    US_SRMW = "us_srmw"
    US_SRMV = "us_srmv"
    US_SRSO = "us_srso"
    US_SRTV = "us_srtv"
    US_SRVC = "us_srvc"
    US_SPNO = "us_spno"
    US_SPSO = "us_spso"
    US_CAMX = "us_camx"
    US_NWPP = "us_nwpp"
    US_AZNM = "us_aznm"
    US_RMPA = "us_rmpa"
    # EU zones
    EU_NORDIC = "eu_nordic"
    EU_CENTRAL_WEST = "eu_central_west"
    EU_CENTRAL_EAST = "eu_central_east"
    EU_IBERIAN = "eu_iberian"
    EU_ITALIAN = "eu_italian"
    EU_BRITISH = "eu_british"
    # Other major grids
    CHINA_NORTH = "china_north"
    CHINA_SOUTH = "china_south"
    INDIA_NORTH = "india_north"
    INDIA_SOUTH = "india_south"
    JAPAN_EAST = "japan_east"
    JAPAN_WEST = "japan_west"


class UncertaintyLevel(str, Enum):
    """Data quality and uncertainty levels per GHG Protocol."""
    MEASURED = "measured"  # Direct measurement (< 5% uncertainty)
    CALCULATED = "calculated"  # Engineering calculation (5-15%)
    ESTIMATED = "estimated"  # Default factors (15-30%)
    EXTRAPOLATED = "extrapolated"  # Limited data (30-50%)


# =============================================================================
# Base Input/Output Models
# =============================================================================

class MRVBaseInput(BaseModel):
    """Base input model for all MRV agents."""

    facility_id: str = Field(..., description="Unique facility identifier")
    reporting_period_start: datetime = Field(
        ..., description="Start of reporting period (UTC)"
    )
    reporting_period_end: datetime = Field(
        ..., description="End of reporting period (UTC)"
    )
    data_quality: UncertaintyLevel = Field(
        default=UncertaintyLevel.CALCULATED,
        description="Data quality level per GHG Protocol"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for context"
    )

    @validator("reporting_period_end")
    def end_after_start(cls, v, values):
        """Validate end date is after start date."""
        if "reporting_period_start" in values and v <= values["reporting_period_start"]:
            raise ValueError("reporting_period_end must be after reporting_period_start")
        return v


class MRVBaseOutput(BaseModel):
    """Base output model for all MRV agents."""

    facility_id: str = Field(..., description="Facility identifier from input")
    agent_id: str = Field(..., description="Agent identifier (e.g., GL-MRV-ENE-001)")
    calculation_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When calculation was performed"
    )
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    processing_time_ms: float = Field(..., description="Processing duration in ms")
    validation_status: str = Field(
        default="PASS",
        description="Validation status: PASS, WARN, FAIL"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal warnings during calculation"
    )
    calculation_trace: List[str] = Field(
        default_factory=list,
        description="Step-by-step calculation trace for audit"
    )
    data_quality: UncertaintyLevel = Field(
        ..., description="Overall data quality of result"
    )
    uncertainty_pct: float = Field(
        ..., ge=0, le=100,
        description="Estimated uncertainty percentage"
    )


# =============================================================================
# Power Generation MRV Schemas (GL-MRV-ENE-001)
# =============================================================================

class PowerGenerationInput(MRVBaseInput):
    """Input for power generation emissions MRV."""

    unit_id: str = Field(..., description="Generation unit identifier")
    generation_type: GenerationType = Field(
        ..., description="Power generation technology"
    )
    fuel_type: Optional[FuelType] = Field(
        None, description="Primary fuel type (for fossil units)"
    )
    fuel_consumption: Optional[float] = Field(
        None, ge=0, description="Fuel consumption in physical units"
    )
    fuel_consumption_unit: str = Field(
        default="MMBTU", description="Unit for fuel consumption"
    )
    net_generation_mwh: float = Field(
        ..., ge=0, description="Net electricity generation (MWh)"
    )
    capacity_factor: Optional[float] = Field(
        None, ge=0, le=1, description="Capacity factor (0-1)"
    )
    heat_rate_btu_kwh: Optional[float] = Field(
        None, ge=0, description="Heat rate (BTU/kWh)"
    )
    cems_co2_tons: Optional[float] = Field(
        None, ge=0, description="CEMS-measured CO2 emissions (tons)"
    )
    stack_measurements: Optional[Dict[str, float]] = Field(
        None, description="Stack measurements (O2%, CO ppm, NOx ppm)"
    )


class PowerGenerationOutput(MRVBaseOutput):
    """Output from power generation emissions MRV."""

    unit_id: str = Field(..., description="Generation unit identifier")
    generation_type: GenerationType = Field(...)

    # Emissions by gas
    co2_tonnes: float = Field(..., ge=0, description="CO2 emissions (metric tons)")
    ch4_tonnes_co2e: float = Field(
        default=0.0, ge=0, description="CH4 emissions (tonnes CO2e)"
    )
    n2o_tonnes_co2e: float = Field(
        default=0.0, ge=0, description="N2O emissions (tonnes CO2e)"
    )
    total_ghg_tonnes_co2e: float = Field(
        ..., ge=0, description="Total GHG emissions (tonnes CO2e)"
    )

    # Emission intensity
    emission_intensity_kg_mwh: float = Field(
        ..., ge=0, description="Emission intensity (kg CO2e/MWh)"
    )

    # Scope classification
    scope: EmissionScope = Field(
        default=EmissionScope.SCOPE_1,
        description="GHG Protocol scope"
    )

    # Verification info
    methodology: str = Field(
        ..., description="Calculation methodology reference"
    )
    emission_factors_used: Dict[str, float] = Field(
        default_factory=dict,
        description="Emission factors applied"
    )


# =============================================================================
# Grid Emissions Tracker Schemas (GL-MRV-ENE-002)
# =============================================================================

class GridEmissionsInput(MRVBaseInput):
    """Input for grid emissions tracking."""

    grid_region: GridRegion = Field(..., description="Grid region for factors")
    electricity_consumption_mwh: float = Field(
        ..., ge=0, description="Electricity consumption (MWh)"
    )
    accounting_method: str = Field(
        default="location_based",
        description="location_based or market_based"
    )
    renewable_certificates_mwh: Optional[float] = Field(
        None, ge=0, description="RECs/GOs applied (MWh)"
    )
    ppa_allocation_mwh: Optional[float] = Field(
        None, ge=0, description="PPA electricity allocation (MWh)"
    )
    supplier_emission_factor: Optional[float] = Field(
        None, ge=0, description="Supplier-specific factor (kg CO2e/MWh)"
    )
    residual_mix_factor: Optional[float] = Field(
        None, ge=0, description="Residual mix factor (kg CO2e/MWh)"
    )


class GridEmissionsOutput(MRVBaseOutput):
    """Output from grid emissions tracking."""

    grid_region: GridRegion = Field(...)

    # Location-based results
    location_based_co2e_tonnes: float = Field(
        ..., ge=0, description="Location-based emissions (tonnes CO2e)"
    )
    location_emission_factor_kg_mwh: float = Field(
        ..., ge=0, description="Grid average emission factor (kg/MWh)"
    )

    # Market-based results
    market_based_co2e_tonnes: float = Field(
        ..., ge=0, description="Market-based emissions (tonnes CO2e)"
    )
    market_emission_factor_kg_mwh: float = Field(
        ..., ge=0, description="Effective market-based factor (kg/MWh)"
    )

    # Instrument breakdown
    rec_reduction_tonnes: float = Field(
        default=0.0, ge=0, description="Reduction from RECs (tonnes)"
    )
    ppa_reduction_tonnes: float = Field(
        default=0.0, ge=0, description="Reduction from PPAs (tonnes)"
    )

    # Scope 2 summary
    scope_2_location: float = Field(
        ..., ge=0, description="Scope 2 location-based (tonnes)"
    )
    scope_2_market: float = Field(
        ..., ge=0, description="Scope 2 market-based (tonnes)"
    )


# =============================================================================
# Renewable Generation MRV Schemas (GL-MRV-ENE-003)
# =============================================================================

class RenewableGenerationInput(MRVBaseInput):
    """Input for renewable generation MRV."""

    asset_id: str = Field(..., description="Renewable asset identifier")
    technology: GenerationType = Field(
        ..., description="Renewable technology type"
    )
    installed_capacity_mw: float = Field(
        ..., gt=0, description="Installed capacity (MW)"
    )
    net_generation_mwh: float = Field(
        ..., ge=0, description="Net generation (MWh)"
    )
    curtailment_mwh: float = Field(
        default=0.0, ge=0, description="Curtailed generation (MWh)"
    )
    auxiliary_consumption_mwh: float = Field(
        default=0.0, ge=0, description="Auxiliary consumption (MWh)"
    )
    grid_region: GridRegion = Field(
        ..., description="Grid region for avoided emissions"
    )


class RenewableGenerationOutput(MRVBaseOutput):
    """Output from renewable generation MRV."""

    asset_id: str = Field(...)
    technology: GenerationType = Field(...)

    # Generation metrics
    gross_generation_mwh: float = Field(
        ..., ge=0, description="Gross generation (MWh)"
    )
    net_generation_mwh: float = Field(
        ..., ge=0, description="Net generation (MWh)"
    )
    capacity_factor: float = Field(
        ..., ge=0, le=1, description="Capacity factor"
    )

    # Emissions avoided
    avoided_co2e_tonnes: float = Field(
        ..., ge=0, description="Avoided emissions (tonnes CO2e)"
    )
    avoided_emission_factor_kg_mwh: float = Field(
        ..., ge=0, description="Grid factor used (kg CO2e/MWh)"
    )

    # Lifecycle emissions
    lifecycle_co2e_tonnes: float = Field(
        ..., ge=0, description="Lifecycle emissions (tonnes CO2e)"
    )
    lifecycle_intensity_g_kwh: float = Field(
        ..., ge=0, description="Lifecycle intensity (g CO2e/kWh)"
    )

    # Certificate eligibility
    rec_eligible_mwh: float = Field(
        ..., ge=0, description="REC-eligible generation (MWh)"
    )


# =============================================================================
# Storage Systems MRV Schemas (GL-MRV-ENE-004)
# =============================================================================

class StorageSystemsInput(MRVBaseInput):
    """Input for energy storage systems MRV."""

    storage_id: str = Field(..., description="Storage system identifier")
    technology: StorageTechnology = Field(
        ..., description="Storage technology type"
    )
    rated_capacity_mwh: float = Field(
        ..., gt=0, description="Rated energy capacity (MWh)"
    )
    rated_power_mw: float = Field(
        ..., gt=0, description="Rated power capacity (MW)"
    )
    energy_charged_mwh: float = Field(
        ..., ge=0, description="Energy charged during period (MWh)"
    )
    energy_discharged_mwh: float = Field(
        ..., ge=0, description="Energy discharged during period (MWh)"
    )
    round_trip_efficiency: float = Field(
        ..., gt=0, le=1, description="Round-trip efficiency"
    )
    grid_region: GridRegion = Field(
        ..., description="Grid region for charging emissions"
    )
    charging_emission_factor_kg_mwh: Optional[float] = Field(
        None, ge=0, description="Emission factor for charging energy"
    )


class StorageSystemsOutput(MRVBaseOutput):
    """Output from energy storage systems MRV."""

    storage_id: str = Field(...)
    technology: StorageTechnology = Field(...)

    # Operational metrics
    total_cycles: float = Field(
        ..., ge=0, description="Equivalent full cycles"
    )
    capacity_utilization: float = Field(
        ..., ge=0, le=1, description="Capacity utilization rate"
    )
    storage_losses_mwh: float = Field(
        ..., ge=0, description="Round-trip losses (MWh)"
    )

    # Emissions accounting
    charging_emissions_tonnes: float = Field(
        ..., ge=0, description="Emissions from charging (tonnes CO2e)"
    )
    avoided_emissions_tonnes: float = Field(
        ..., ge=0, description="Avoided emissions from dispatch (tonnes CO2e)"
    )
    net_emissions_tonnes: float = Field(
        ..., description="Net emissions impact (tonnes CO2e)"
    )
    emission_intensity_kg_mwh: float = Field(
        ..., description="Emission intensity of discharged energy"
    )

    # Lifecycle considerations
    embedded_emissions_tonnes: float = Field(
        default=0.0, ge=0,
        description="Annualized embedded emissions (tonnes CO2e)"
    )


# =============================================================================
# Transmission Loss MRV Schemas (GL-MRV-ENE-005)
# =============================================================================

class TransmissionLossInput(MRVBaseInput):
    """Input for transmission and distribution loss MRV."""

    network_id: str = Field(..., description="Network/utility identifier")
    voltage_level: str = Field(
        ..., description="Voltage level (transmission/distribution)"
    )
    energy_injected_mwh: float = Field(
        ..., ge=0, description="Energy injected into network (MWh)"
    )
    energy_delivered_mwh: float = Field(
        ..., ge=0, description="Energy delivered to customers (MWh)"
    )
    line_length_km: Optional[float] = Field(
        None, ge=0, description="Total line length (km)"
    )
    transformer_losses_mwh: Optional[float] = Field(
        None, ge=0, description="Transformer losses (MWh)"
    )
    grid_region: GridRegion = Field(
        ..., description="Grid region for emission factors"
    )


class TransmissionLossOutput(MRVBaseOutput):
    """Output from transmission and distribution loss MRV."""

    network_id: str = Field(...)

    # Loss metrics
    total_losses_mwh: float = Field(
        ..., ge=0, description="Total T&D losses (MWh)"
    )
    loss_percentage: float = Field(
        ..., ge=0, le=100, description="Loss percentage"
    )
    technical_losses_mwh: float = Field(
        ..., ge=0, description="Technical losses (MWh)"
    )
    non_technical_losses_mwh: float = Field(
        default=0.0, ge=0, description="Non-technical losses (MWh)"
    )

    # Emissions from losses
    loss_emissions_tonnes: float = Field(
        ..., ge=0, description="Emissions from losses (tonnes CO2e)"
    )
    emission_factor_kg_mwh: float = Field(
        ..., ge=0, description="Grid emission factor applied"
    )

    # Loss-adjusted factors
    loss_adjusted_factor_kg_mwh: float = Field(
        ..., ge=0, description="Loss-adjusted emission factor"
    )


# =============================================================================
# Fuel Supply Chain MRV Schemas (GL-MRV-ENE-006)
# =============================================================================

class FuelSupplyChainInput(MRVBaseInput):
    """Input for fuel supply chain (upstream) emissions MRV."""

    fuel_type: FuelType = Field(..., description="Fuel type")
    fuel_quantity: float = Field(..., gt=0, description="Fuel quantity")
    fuel_unit: str = Field(..., description="Fuel unit (tonnes, m3, MMBTU)")
    origin_country: str = Field(
        ..., min_length=2, max_length=3,
        description="Fuel origin country (ISO code)"
    )
    extraction_method: Optional[str] = Field(
        None, description="Extraction/production method"
    )
    transport_mode: str = Field(
        default="pipeline", description="Transport mode (pipeline, ship, rail)"
    )
    transport_distance_km: float = Field(
        ..., ge=0, description="Transport distance (km)"
    )
    supplier_id: Optional[str] = Field(
        None, description="Fuel supplier identifier"
    )


class FuelSupplyChainOutput(MRVBaseOutput):
    """Output from fuel supply chain emissions MRV."""

    fuel_type: FuelType = Field(...)

    # Upstream emission components
    extraction_emissions_tonnes: float = Field(
        ..., ge=0, description="Extraction/production emissions (tonnes CO2e)"
    )
    processing_emissions_tonnes: float = Field(
        ..., ge=0, description="Processing/refining emissions (tonnes CO2e)"
    )
    transport_emissions_tonnes: float = Field(
        ..., ge=0, description="Transport emissions (tonnes CO2e)"
    )
    fugitive_emissions_tonnes: float = Field(
        default=0.0, ge=0, description="Fugitive emissions (tonnes CO2e)"
    )
    total_upstream_emissions_tonnes: float = Field(
        ..., ge=0, description="Total upstream emissions (tonnes CO2e)"
    )

    # Intensity metrics
    upstream_intensity_kg_unit: float = Field(
        ..., ge=0, description="Upstream intensity per fuel unit"
    )
    wtt_factor_kg_kwh: float = Field(
        ..., ge=0, description="Well-to-tank factor (kg CO2e/kWh)"
    )

    # Scope classification
    scope: EmissionScope = Field(
        default=EmissionScope.SCOPE_3_UPSTREAM
    )


# =============================================================================
# CHP Systems MRV Schemas (GL-MRV-ENE-007)
# =============================================================================

class CHPSystemsInput(MRVBaseInput):
    """Input for combined heat and power (CHP) systems MRV."""

    chp_id: str = Field(..., description="CHP system identifier")
    fuel_type: FuelType = Field(..., description="Primary fuel")
    fuel_consumption: float = Field(
        ..., gt=0, description="Fuel consumption"
    )
    fuel_unit: str = Field(default="MMBTU")
    electricity_output_mwh: float = Field(
        ..., ge=0, description="Electricity output (MWh)"
    )
    heat_output_mmbtu: float = Field(
        ..., ge=0, description="Useful heat output (MMBTU)"
    )
    heat_output_mwh_thermal: Optional[float] = Field(
        None, ge=0, description="Heat output (MWh thermal)"
    )
    reference_electricity_efficiency: float = Field(
        default=0.38, gt=0, le=1,
        description="Reference efficiency for separate electricity"
    )
    reference_heat_efficiency: float = Field(
        default=0.90, gt=0, le=1,
        description="Reference efficiency for separate heat"
    )


class CHPSystemsOutput(MRVBaseOutput):
    """Output from CHP systems MRV."""

    chp_id: str = Field(...)

    # Efficiency metrics
    electrical_efficiency: float = Field(
        ..., ge=0, le=1, description="Electrical efficiency"
    )
    thermal_efficiency: float = Field(
        ..., ge=0, le=1, description="Thermal efficiency"
    )
    overall_efficiency: float = Field(
        ..., ge=0, le=1, description="Overall CHP efficiency"
    )
    power_to_heat_ratio: float = Field(
        ..., ge=0, description="Power-to-heat ratio"
    )

    # Emissions
    total_emissions_tonnes: float = Field(
        ..., ge=0, description="Total CHP emissions (tonnes CO2e)"
    )
    electricity_allocation_tonnes: float = Field(
        ..., ge=0, description="Emissions allocated to electricity"
    )
    heat_allocation_tonnes: float = Field(
        ..., ge=0, description="Emissions allocated to heat"
    )

    # Intensities
    electricity_intensity_kg_mwh: float = Field(
        ..., ge=0, description="Electricity emission intensity"
    )
    heat_intensity_kg_mwh: float = Field(
        ..., ge=0, description="Heat emission intensity"
    )

    # Primary Energy Savings
    primary_energy_savings_pct: float = Field(
        ..., description="Primary energy savings vs separate production"
    )
    avoided_emissions_tonnes: float = Field(
        ..., ge=0, description="Avoided emissions vs separate"
    )


# =============================================================================
# Hydrogen Production MRV Schemas (GL-MRV-ENE-008)
# =============================================================================

class HydrogenProductionInput(MRVBaseInput):
    """Input for hydrogen production emissions MRV."""

    production_id: str = Field(..., description="Production facility identifier")
    production_method: HydrogenProductionMethod = Field(
        ..., description="Hydrogen production method"
    )
    hydrogen_output_kg: float = Field(
        ..., gt=0, description="Hydrogen produced (kg)"
    )
    feedstock_consumption: float = Field(
        ..., ge=0, description="Feedstock consumption"
    )
    feedstock_unit: str = Field(
        ..., description="Feedstock unit (MMBTU, kg, kWh)"
    )
    electricity_consumption_kwh: float = Field(
        ..., ge=0, description="Electricity consumption (kWh)"
    )
    grid_region: Optional[GridRegion] = Field(
        None, description="Grid region for electricity"
    )
    electricity_emission_factor_kg_kwh: Optional[float] = Field(
        None, ge=0, description="Electricity emission factor"
    )
    ccs_capture_rate: Optional[float] = Field(
        None, ge=0, le=1, description="CCS capture rate (if applicable)"
    )


class HydrogenProductionOutput(MRVBaseOutput):
    """Output from hydrogen production MRV."""

    production_id: str = Field(...)
    production_method: HydrogenProductionMethod = Field(...)

    # Production metrics
    hydrogen_output_kg: float = Field(
        ..., gt=0, description="Hydrogen produced (kg)"
    )
    specific_energy_consumption_kwh_kg: float = Field(
        ..., ge=0, description="Specific energy consumption"
    )

    # Emissions by source
    process_emissions_tonnes: float = Field(
        ..., ge=0, description="Direct process emissions"
    )
    electricity_emissions_tonnes: float = Field(
        ..., ge=0, description="Electricity-related emissions"
    )
    feedstock_emissions_tonnes: float = Field(
        default=0.0, ge=0, description="Feedstock upstream emissions"
    )
    captured_emissions_tonnes: float = Field(
        default=0.0, ge=0, description="CO2 captured (if CCS)"
    )
    total_emissions_tonnes: float = Field(
        ..., ge=0, description="Total emissions (tonnes CO2e)"
    )

    # Carbon intensity
    carbon_intensity_kg_co2_kg_h2: float = Field(
        ..., ge=0, description="kg CO2e per kg H2"
    )

    # Hydrogen color classification
    hydrogen_color: str = Field(
        ..., description="Hydrogen color (grey, blue, green, etc.)"
    )

    # Certification eligibility
    low_carbon_eligible: bool = Field(
        ..., description="Eligible for low-carbon hydrogen certification"
    )
    certification_threshold_kg_co2_kg_h2: float = Field(
        default=3.0, description="Threshold for low-carbon (kg CO2/kg H2)"
    )
