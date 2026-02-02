# -*- coding: utf-8 -*-
"""
Decarbonization Energy Sector - Common Schemas and Data Models

This module defines Pydantic models shared across all Decarbonization
Energy agents for planning, strategy, and optimization.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class DecarbonizationPathway(str, Enum):
    """Standard decarbonization pathways."""
    ELECTRIFICATION = "electrification"
    RENEWABLE_INTEGRATION = "renewable_integration"
    ENERGY_STORAGE = "energy_storage"
    HYDROGEN = "hydrogen"
    NUCLEAR = "nuclear"
    CCUS = "ccus"
    EFFICIENCY = "efficiency"
    DEMAND_FLEXIBILITY = "demand_flexibility"
    GRID_MODERNIZATION = "grid_modernization"


class TimeHorizon(str, Enum):
    """Planning time horizons."""
    SHORT_TERM = "short_term"  # 1-3 years
    MEDIUM_TERM = "medium_term"  # 3-10 years
    LONG_TERM = "long_term"  # 10-30 years
    NET_ZERO_2050 = "net_zero_2050"


class TechnologyReadinessLevel(int, Enum):
    """Technology readiness levels (TRL)."""
    TRL_1 = 1  # Basic principles observed
    TRL_2 = 2  # Technology concept formulated
    TRL_3 = 3  # Experimental proof of concept
    TRL_4 = 4  # Technology validated in lab
    TRL_5 = 5  # Technology validated in relevant environment
    TRL_6 = 6  # Technology demonstrated in relevant environment
    TRL_7 = 7  # System prototype in operational environment
    TRL_8 = 8  # System complete and qualified
    TRL_9 = 9  # Actual system proven in operational environment


class RenewableTechnology(str, Enum):
    """Renewable energy technologies."""
    SOLAR_PV = "solar_pv"
    SOLAR_CSP = "solar_csp"
    WIND_ONSHORE = "wind_onshore"
    WIND_OFFSHORE = "wind_offshore"
    HYDRO = "hydro"
    GEOTHERMAL = "geothermal"
    BIOMASS = "biomass"
    OCEAN = "ocean"


class StorageApplication(str, Enum):
    """Energy storage applications."""
    FREQUENCY_REGULATION = "frequency_regulation"
    SPINNING_RESERVE = "spinning_reserve"
    PEAK_SHAVING = "peak_shaving"
    LOAD_SHIFTING = "load_shifting"
    RENEWABLE_FIRMING = "renewable_firming"
    TRANSMISSION_DEFERRAL = "transmission_deferral"
    MICROGRIDS = "microgrids"
    EV_CHARGING = "ev_charging"


class HydrogenApplication(str, Enum):
    """Hydrogen end-use applications."""
    POWER_GENERATION = "power_generation"
    INDUSTRIAL_FEEDSTOCK = "industrial_feedstock"
    INDUSTRIAL_HEAT = "industrial_heat"
    TRANSPORT_HEAVY_DUTY = "transport_heavy_duty"
    TRANSPORT_AVIATION = "transport_aviation"
    TRANSPORT_MARITIME = "transport_maritime"
    BUILDING_HEATING = "building_heating"
    ENERGY_STORAGE = "energy_storage"


class NuclearTechnology(str, Enum):
    """Nuclear technology types."""
    LARGE_LIGHT_WATER = "large_lwr"
    SMALL_MODULAR_REACTOR = "smr"
    ADVANCED_GEN_IV = "gen_iv"
    FUSION = "fusion"


class CCUSTechnology(str, Enum):
    """CCUS technology types."""
    POST_COMBUSTION = "post_combustion"
    PRE_COMBUSTION = "pre_combustion"
    OXY_COMBUSTION = "oxy_combustion"
    DIRECT_AIR_CAPTURE = "dac"
    BIOENERGY_CCS = "beccs"


# =============================================================================
# Base Models
# =============================================================================

class DecarbonizationBaseInput(BaseModel):
    """Base input for decarbonization planning agents."""

    organization_id: str = Field(..., description="Organization identifier")
    region: str = Field(..., description="Geographic region/market")
    baseline_year: int = Field(..., ge=2000, le=2030)
    target_year: int = Field(..., ge=2030, le=2100)
    current_emissions_mtco2e: float = Field(
        ..., gt=0, description="Current annual emissions (Mt CO2e)"
    )
    target_emissions_mtco2e: float = Field(
        ..., ge=0, description="Target annual emissions (Mt CO2e)"
    )
    budget_million_usd: Optional[float] = Field(
        None, ge=0, description="Available capital budget ($M)"
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Planning constraints"
    )


class DecarbonizationBaseOutput(BaseModel):
    """Base output for decarbonization planning agents."""

    organization_id: str = Field(...)
    agent_id: str = Field(...)
    calculation_timestamp: datetime = Field(...)
    provenance_hash: str = Field(...)
    processing_time_ms: float = Field(...)

    # Pathway summary
    recommended_pathways: List[str] = Field(...)
    total_abatement_mtco2e: float = Field(...)
    total_investment_million_usd: float = Field(...)
    levelized_abatement_cost_usd_tco2e: float = Field(...)

    # Timeline
    milestones: List[Dict[str, Any]] = Field(default_factory=list)

    # Risk and uncertainty
    confidence_level: float = Field(..., ge=0, le=1)
    key_risks: List[str] = Field(default_factory=list)


# =============================================================================
# Grid Decarbonization (GL-DECARB-ENE-001)
# =============================================================================

class GridDecarbonizationInput(DecarbonizationBaseInput):
    """Input for grid decarbonization planning."""

    grid_region: str = Field(..., description="Grid/balancing authority")
    current_generation_mix: Dict[str, float] = Field(
        ..., description="Current generation by source (%)"
    )
    peak_demand_gw: float = Field(..., gt=0)
    annual_demand_twh: float = Field(..., gt=0)
    renewable_potential: Dict[str, float] = Field(
        default_factory=dict, description="Technical potential by resource (GW)"
    )
    interconnection_capacity_gw: float = Field(default=0.0, ge=0)
    storage_capacity_gwh: float = Field(default=0.0, ge=0)


class GridDecarbonizationOutput(DecarbonizationBaseOutput):
    """Output from grid decarbonization planning."""

    # Future generation mix
    target_generation_mix: Dict[str, float] = Field(...)
    renewable_percentage: float = Field(..., ge=0, le=100)
    clean_energy_percentage: float = Field(..., ge=0, le=100)

    # Capacity additions
    renewable_additions_gw: Dict[str, float] = Field(...)
    storage_additions_gwh: float = Field(...)
    transmission_additions_gw: float = Field(...)

    # Retirements
    fossil_retirements_gw: Dict[str, float] = Field(...)

    # Costs
    capital_investment_billion_usd: float = Field(...)
    annual_operating_savings_million_usd: float = Field(...)


# =============================================================================
# Renewable Integration (GL-DECARB-ENE-002)
# =============================================================================

class RenewableIntegrationInput(DecarbonizationBaseInput):
    """Input for renewable energy integration planning."""

    technology: RenewableTechnology = Field(...)
    target_capacity_mw: float = Field(..., gt=0)
    site_characteristics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Site-specific parameters (irradiance, wind speed, etc.)"
    )
    grid_connection_point: str = Field(...)
    land_availability_km2: Optional[float] = Field(None, ge=0)
    interconnection_queue_position: Optional[int] = Field(None)


class RenewableIntegrationOutput(DecarbonizationBaseOutput):
    """Output from renewable integration planning."""

    technology: str = Field(...)
    installed_capacity_mw: float = Field(...)
    expected_generation_gwh_year: float = Field(...)
    capacity_factor: float = Field(..., ge=0, le=1)

    # Integration requirements
    curtailment_risk_pct: float = Field(..., ge=0, le=100)
    storage_requirement_mwh: float = Field(...)
    grid_upgrade_requirements: List[str] = Field(...)

    # Economics
    lcoe_usd_mwh: float = Field(...)
    capital_cost_million_usd: float = Field(...)
    payback_years: float = Field(...)

    # Environmental
    annual_emissions_avoided_tonnes: float = Field(...)
    land_use_km2: float = Field(...)


# =============================================================================
# Storage Optimization (GL-DECARB-ENE-003)
# =============================================================================

class StorageOptimizationInput(DecarbonizationBaseInput):
    """Input for energy storage optimization."""

    applications: List[StorageApplication] = Field(...)
    renewable_penetration_pct: float = Field(..., ge=0, le=100)
    peak_demand_mw: float = Field(..., gt=0)
    load_profile: Optional[List[float]] = Field(
        None, description="Hourly load profile (MW)"
    )
    existing_storage_mwh: float = Field(default=0.0, ge=0)
    electricity_price_profile: Optional[List[float]] = Field(
        None, description="Hourly prices ($/MWh)"
    )


class StorageOptimizationOutput(DecarbonizationBaseOutput):
    """Output from storage optimization."""

    # Optimal sizing
    recommended_power_mw: float = Field(...)
    recommended_energy_mwh: float = Field(...)
    recommended_duration_hours: float = Field(...)

    # Technology selection
    recommended_technology: str = Field(...)
    technology_rationale: str = Field(...)

    # Value streams
    annual_revenue_streams: Dict[str, float] = Field(...)
    total_annual_value_usd: float = Field(...)

    # Economics
    capital_cost_usd: float = Field(...)
    lcoe_usd_mwh: float = Field(...)
    npv_million_usd: float = Field(...)
    irr_pct: float = Field(...)


# =============================================================================
# Demand Flexibility (GL-DECARB-ENE-004)
# =============================================================================

class DemandFlexibilityInput(DecarbonizationBaseInput):
    """Input for demand-side flexibility planning."""

    sector: str = Field(..., description="Sector (industrial, commercial, residential)")
    current_peak_mw: float = Field(..., gt=0)
    annual_consumption_mwh: float = Field(..., gt=0)
    load_profile: Optional[List[float]] = Field(None)
    existing_dr_capacity_mw: float = Field(default=0.0, ge=0)
    process_flexibility: Dict[str, float] = Field(
        default_factory=dict,
        description="Flexible processes and their MW potential"
    )


class DemandFlexibilityOutput(DecarbonizationBaseOutput):
    """Output from demand flexibility planning."""

    # Flexibility potential
    total_flexible_load_mw: float = Field(...)
    flexibility_percentage: float = Field(..., ge=0, le=100)
    peak_reduction_mw: float = Field(...)

    # Programs
    recommended_programs: List[Dict[str, Any]] = Field(...)

    # Value
    annual_savings_usd: float = Field(...)
    grid_benefit_usd: float = Field(...)
    emissions_reduction_tonnes: float = Field(...)


# =============================================================================
# Hydrogen Strategy (GL-DECARB-ENE-005)
# =============================================================================

class HydrogenStrategyInput(DecarbonizationBaseInput):
    """Input for hydrogen transition planning."""

    applications: List[HydrogenApplication] = Field(...)
    current_hydrogen_demand_tonnes_year: float = Field(default=0.0, ge=0)
    target_hydrogen_demand_tonnes_year: float = Field(..., ge=0)
    production_pathway_preference: Optional[str] = Field(None)
    renewable_electricity_price_usd_mwh: Optional[float] = Field(None, ge=0)
    natural_gas_price_usd_mmbtu: Optional[float] = Field(None, ge=0)


class HydrogenStrategyOutput(DecarbonizationBaseOutput):
    """Output from hydrogen strategy planning."""

    # Production
    recommended_production_method: str = Field(...)
    production_capacity_tonnes_year: float = Field(...)
    electrolyzer_capacity_mw: Optional[float] = Field(None)

    # Infrastructure
    storage_capacity_tonnes: float = Field(...)
    pipeline_km: float = Field(default=0.0)

    # Economics
    hydrogen_production_cost_usd_kg: float = Field(...)
    capital_investment_million_usd: float = Field(...)
    annual_operating_cost_million_usd: float = Field(...)

    # Environmental
    hydrogen_carbon_intensity_kg_co2_kg_h2: float = Field(...)
    annual_emissions_avoided_tonnes: float = Field(...)


# =============================================================================
# Nuclear Assessment (GL-DECARB-ENE-006)
# =============================================================================

class NuclearAssessmentInput(DecarbonizationBaseInput):
    """Input for nuclear energy assessment."""

    technology_preference: Optional[NuclearTechnology] = Field(None)
    capacity_requirement_gw: float = Field(..., gt=0)
    site_requirements: Dict[str, Any] = Field(default_factory=dict)
    regulatory_environment: str = Field(
        default="standard", description="Regulatory readiness level"
    )
    public_acceptance_level: str = Field(
        default="moderate", description="low, moderate, high"
    )


class NuclearAssessmentOutput(DecarbonizationBaseOutput):
    """Output from nuclear energy assessment."""

    # Technology recommendation
    recommended_technology: str = Field(...)
    number_of_units: int = Field(...)
    total_capacity_gw: float = Field(...)

    # Timeline
    development_timeline_years: int = Field(...)
    construction_start_year: int = Field(...)
    commercial_operation_year: int = Field(...)

    # Economics
    overnight_cost_billion_usd: float = Field(...)
    lcoe_usd_mwh: float = Field(...)

    # Environmental
    annual_generation_twh: float = Field(...)
    annual_emissions_avoided_mt: float = Field(...)


# =============================================================================
# CCUS for Power (GL-DECARB-ENE-007)
# =============================================================================

class CCUSPowerInput(DecarbonizationBaseInput):
    """Input for power sector CCUS planning."""

    target_facilities: List[Dict[str, Any]] = Field(
        ..., description="Facilities for CCUS retrofit"
    )
    technology_preference: Optional[CCUSTechnology] = Field(None)
    storage_site_availability: bool = Field(default=True)
    co2_transport_distance_km: float = Field(default=50.0, ge=0)


class CCUSPowerOutput(DecarbonizationBaseOutput):
    """Output from power sector CCUS planning."""

    # Capture
    capture_technology: str = Field(...)
    capture_capacity_mtco2_year: float = Field(...)
    capture_rate_pct: float = Field(...)

    # Storage/utilization
    storage_solution: str = Field(...)
    storage_capacity_mt: float = Field(...)

    # Economics
    capture_cost_usd_tco2: float = Field(...)
    capital_investment_million_usd: float = Field(...)
    energy_penalty_pct: float = Field(...)

    # Environmental
    net_emissions_reduction_mtco2_year: float = Field(...)


# =============================================================================
# Distributed Generation (GL-DECARB-ENE-008)
# =============================================================================

class DistributedGenerationInput(DecarbonizationBaseInput):
    """Input for distributed generation planning."""

    sector: str = Field(..., description="residential, commercial, industrial")
    rooftop_area_km2: Optional[float] = Field(None, ge=0)
    average_building_consumption_kwh: float = Field(..., gt=0)
    number_of_buildings: int = Field(..., gt=0)
    existing_der_capacity_mw: float = Field(default=0.0, ge=0)
    net_metering_available: bool = Field(default=True)


class DistributedGenerationOutput(DecarbonizationBaseOutput):
    """Output from distributed generation planning."""

    # DER deployment
    solar_pv_capacity_mw: float = Field(...)
    battery_storage_mwh: float = Field(...)
    participation_rate_pct: float = Field(...)

    # Grid impact
    peak_reduction_mw: float = Field(...)
    grid_export_mwh_year: float = Field(...)

    # Economics
    average_payback_years: float = Field(...)
    lcoe_usd_kwh: float = Field(...)

    # Environmental
    annual_emissions_avoided_tonnes: float = Field(...)


# =============================================================================
# Grid Modernization (GL-DECARB-ENE-009)
# =============================================================================

class GridModernizationInput(DecarbonizationBaseInput):
    """Input for grid modernization planning."""

    grid_characteristics: Dict[str, Any] = Field(
        ..., description="Current grid infrastructure status"
    )
    smart_meter_penetration_pct: float = Field(default=0.0, ge=0, le=100)
    renewable_penetration_target_pct: float = Field(..., ge=0, le=100)
    ev_adoption_target_pct: float = Field(default=0.0, ge=0, le=100)


class GridModernizationOutput(DecarbonizationBaseOutput):
    """Output from grid modernization planning."""

    # Infrastructure upgrades
    transmission_investments: List[Dict[str, Any]] = Field(...)
    distribution_investments: List[Dict[str, Any]] = Field(...)
    smart_grid_investments: List[Dict[str, Any]] = Field(...)

    # Capabilities enabled
    hosting_capacity_increase_pct: float = Field(...)
    reliability_improvement_pct: float = Field(...)

    # Economics
    total_investment_billion_usd: float = Field(...)
    annual_benefits_billion_usd: float = Field(...)


# =============================================================================
# Just Transition (GL-DECARB-ENE-010)
# =============================================================================

class JustTransitionInput(DecarbonizationBaseInput):
    """Input for just transition planning."""

    affected_communities: List[Dict[str, Any]] = Field(
        ..., description="Communities affected by energy transition"
    )
    fossil_fuel_employment: int = Field(..., ge=0)
    planned_retirements: List[Dict[str, Any]] = Field(...)
    retraining_budget_million_usd: Optional[float] = Field(None, ge=0)


class JustTransitionOutput(DecarbonizationBaseOutput):
    """Output from just transition planning."""

    # Workforce impact
    jobs_at_risk: int = Field(...)
    new_clean_energy_jobs: int = Field(...)
    net_job_impact: int = Field(...)

    # Programs
    retraining_programs: List[Dict[str, Any]] = Field(...)
    economic_diversification: List[Dict[str, Any]] = Field(...)
    community_investments: List[Dict[str, Any]] = Field(...)

    # Timeline
    transition_timeline_years: int = Field(...)

    # Investment
    total_transition_investment_million_usd: float = Field(...)
