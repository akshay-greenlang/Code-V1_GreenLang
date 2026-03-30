# -*- coding: utf-8 -*-
"""
SectorPathwaySetupWizard - 7-Step Sector Configuration for PACK-028
=======================================================================

This module implements a 7-step sector pathway configuration wizard for
organizations requiring deep sector-specific decarbonization pathway
analysis. Guides users through sector selection, activity data
configuration, baseline setup, SDA/IEA pathway preferences, technology
inventory, convergence model selection, and final review & deployment.

Wizard Steps (7):
    1. sector_selection         -- Primary sector + SDA eligibility check
    2. activity_data_setup      -- Sector-specific activity data configuration
    3. baseline_configuration   -- Base year, emissions, intensity baseline
    4. pathway_preferences      -- SDA/IEA scenario + convergence model
    5. technology_inventory     -- Current technology portfolio assessment
    6. abatement_planning       -- Lever prioritization + roadmap setup
    7. review_and_deploy        -- Configuration review + deployment

Sector Selection Logic:
    The wizard first identifies whether the organization falls into one of
    the 12 SBTi SDA-eligible sectors. If so, the SDA convergence pathway
    is activated automatically. For all sectors (SDA and non-SDA), the IEA
    NZE pathway is available. The wizard also identifies applicable NACE
    Rev.2, GICS, and ISIC Rev.4 codes for the selected sector.

Activity Data Setup:
    Based on the selected sector, the wizard configures which activity data
    fields are required (e.g., electricity generation in MWh for power,
    crude steel production in tonnes for steel, passenger-km for aviation).
    It also identifies which MRV agents and DATA agents should be prioritized
    for the sector.

Pathway Preferences:
    Users select their preferred climate scenario (NZE 1.5C, WB2C, 2C,
    APS, STEPS), convergence model (linear, exponential, S-curve, stepped),
    and target year (2030 near-term, 2050 long-term). The wizard validates
    that preferences are compatible with the selected sector's SDA eligibility.

Technology Inventory:
    Users declare their current technology portfolio (e.g., blast furnaces
    vs. EAF for steel, coal plants vs. renewables for power). This feeds
    into the technology roadmap engine for transition planning.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-028 Sector Pathway Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SectorWizardStep(str, Enum):
    """The 7 steps of the sector pathway setup wizard."""
    SECTOR_SELECTION = "sector_selection"
    ACTIVITY_DATA_SETUP = "activity_data_setup"
    BASELINE_CONFIGURATION = "baseline_configuration"
    PATHWAY_PREFERENCES = "pathway_preferences"
    TECHNOLOGY_INVENTORY = "technology_inventory"
    ABATEMENT_PLANNING = "abatement_planning"
    REVIEW_AND_DEPLOY = "review_and_deploy"

class StepStatus(str, Enum):
    """Status of a wizard step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class SDAEligibility(str, Enum):
    """SDA eligibility classification."""
    SDA_ELIGIBLE = "sda_eligible"
    IEA_ONLY = "iea_only"
    CUSTOM_PATHWAY = "custom_pathway"

class ConvergencePreference(str, Enum):
    """User convergence model preference."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    S_CURVE = "s_curve"
    STEPPED = "stepped"
    AUTO = "auto"

class ScenarioPreference(str, Enum):
    """Climate scenario preference."""
    NZE_15C = "nze_1.5c"
    WB2C = "wb2c"
    C2 = "2c"
    APS = "aps"
    STEPS = "steps"

class TechnologyMaturity(str, Enum):
    """Current technology maturity assessment."""
    LEGACY = "legacy"
    TRANSITIONAL = "transitional"
    BEST_AVAILABLE = "best_available"
    EMERGING = "emerging"
    FRONTIER = "frontier"

class SectorRoutingGroup(str, Enum):
    """Sector routing group for pipeline configuration."""
    HEAVY_INDUSTRY = "heavy_industry"
    LIGHT_INDUSTRY = "light_industry"
    TRANSPORT = "transport"
    POWER = "power"
    BUILDINGS = "buildings"
    AGRICULTURE = "agriculture"
    CROSS_SECTOR = "cross_sector"

# ---------------------------------------------------------------------------
# Sector Selection Data
# ---------------------------------------------------------------------------

SDA_SECTOR_OPTIONS: Dict[str, Dict[str, Any]] = {
    "power_generation": {
        "display_name": "Power Generation & Electricity",
        "sda_eligible": True,
        "nace_rev2": ["D35.1"],
        "gics": "551010",
        "isic_rev4": ["3510"],
        "intensity_metric": "gCO2/kWh",
        "routing_group": "power",
        "description": "Electricity generation, transmission, and distribution",
        "key_activities": ["electricity_generated_mwh", "capacity_mw", "fuel_mix"],
    },
    "steel": {
        "display_name": "Iron & Steel Manufacturing",
        "sda_eligible": True,
        "nace_rev2": ["C24.1"],
        "gics": "151040",
        "isic_rev4": ["2410"],
        "intensity_metric": "tCO2e/tonne crude steel",
        "routing_group": "heavy_industry",
        "description": "Iron and steel production including primary and secondary routes",
        "key_activities": ["crude_steel_tonnes", "scrap_ratio_pct", "production_route"],
    },
    "cement": {
        "display_name": "Cement & Clinker Manufacturing",
        "sda_eligible": True,
        "nace_rev2": ["C23.51"],
        "gics": "151020",
        "isic_rev4": ["2394"],
        "intensity_metric": "kgCO2/tonne cite",
        "routing_group": "heavy_industry",
        "description": "Cement, clinker, and concrete production",
        "key_activities": ["clinker_tonnes", "cement_tonnes", "clinker_ratio"],
    },
    "aluminum": {
        "display_name": "Aluminum Smelting & Refining",
        "sda_eligible": True,
        "nace_rev2": ["C24.42"],
        "gics": "151040",
        "isic_rev4": ["2420"],
        "intensity_metric": "tCO2e/tonne primary aluminum",
        "routing_group": "heavy_industry",
        "description": "Primary and secondary aluminum production",
        "key_activities": ["primary_aluminum_tonnes", "secondary_aluminum_tonnes", "smelter_energy_kwh_per_tonne"],
    },
    "pulp_paper": {
        "display_name": "Pulp & Paper Manufacturing",
        "sda_eligible": True,
        "nace_rev2": ["C17.1", "C17.2"],
        "gics": "151050",
        "isic_rev4": ["1701", "1702"],
        "intensity_metric": "tCO2e/tonne product",
        "routing_group": "light_industry",
        "description": "Pulp, paper, and packaging production",
        "key_activities": ["pulp_tonnes", "paper_tonnes", "energy_source"],
    },
    "chemicals": {
        "display_name": "Chemical Manufacturing",
        "sda_eligible": True,
        "nace_rev2": ["C20"],
        "gics": "151010",
        "isic_rev4": ["2011", "2012", "2013"],
        "intensity_metric": "tCO2e/tonne product",
        "routing_group": "heavy_industry",
        "description": "Basic and specialty chemical production",
        "key_activities": ["production_tonnes", "product_type", "feedstock_type"],
    },
    "aviation": {
        "display_name": "Aviation & Air Transport",
        "sda_eligible": True,
        "nace_rev2": ["H51.1"],
        "gics": "203020",
        "isic_rev4": ["5110"],
        "intensity_metric": "gCO2/pkm",
        "routing_group": "transport",
        "description": "Passenger and cargo air transport",
        "key_activities": ["passenger_km", "cargo_tkm", "fuel_consumption_litres", "fleet_type"],
    },
    "shipping": {
        "display_name": "Maritime Shipping & Transport",
        "sda_eligible": True,
        "nace_rev2": ["H50.1", "H50.2"],
        "gics": "203010",
        "isic_rev4": ["5011", "5012"],
        "intensity_metric": "gCO2/tkm",
        "routing_group": "transport",
        "description": "Freight and passenger maritime transport",
        "key_activities": ["tonne_km", "deadweight_tonnes", "fuel_type", "vessel_type"],
    },
    "road_transport": {
        "display_name": "Road Transport & Logistics",
        "sda_eligible": True,
        "nace_rev2": ["H49.3", "H49.4"],
        "gics": "203030",
        "isic_rev4": ["4923"],
        "intensity_metric": "gCO2/vkm",
        "routing_group": "transport",
        "description": "Road freight and passenger transport",
        "key_activities": ["vehicle_km", "tonne_km", "fleet_size", "fuel_type", "ev_share_pct"],
    },
    "rail": {
        "display_name": "Rail Transport",
        "sda_eligible": True,
        "nace_rev2": ["H49.1", "H49.2"],
        "gics": "203040",
        "isic_rev4": ["4911", "4912"],
        "intensity_metric": "gCO2/tkm",
        "routing_group": "transport",
        "description": "Passenger and freight rail transport",
        "key_activities": ["passenger_km", "tonne_km", "electrification_pct"],
    },
    "buildings_residential": {
        "display_name": "Residential Buildings",
        "sda_eligible": True,
        "nace_rev2": ["L68.2"],
        "gics": "601010",
        "isic_rev4": ["6810"],
        "intensity_metric": "kgCO2/m2",
        "routing_group": "buildings",
        "description": "Residential property management and construction",
        "key_activities": ["floor_area_m2", "heating_type", "building_age", "insulation_rating"],
    },
    "buildings_commercial": {
        "display_name": "Commercial Buildings",
        "sda_eligible": True,
        "nace_rev2": ["L68.2"],
        "gics": "601020",
        "isic_rev4": ["6810"],
        "intensity_metric": "kgCO2/m2",
        "routing_group": "buildings",
        "description": "Commercial property management, offices, retail",
        "key_activities": ["floor_area_m2", "heating_type", "cooling_type", "occupancy_rate"],
    },
}

EXTENDED_SECTOR_OPTIONS: Dict[str, Dict[str, Any]] = {
    "agriculture": {
        "display_name": "Agriculture & Livestock",
        "sda_eligible": False,
        "nace_rev2": ["A01"],
        "gics": "301020",
        "isic_rev4": ["0111", "0112", "0121"],
        "intensity_metric": "tCO2e/ha",
        "routing_group": "agriculture",
        "description": "Crop production, livestock, mixed farming",
        "key_activities": ["hectares", "livestock_head", "crop_type", "fertilizer_tonnes"],
    },
    "food_beverage": {
        "display_name": "Food & Beverage Processing",
        "sda_eligible": False,
        "nace_rev2": ["C10", "C11"],
        "gics": "302010",
        "isic_rev4": ["1010", "1020", "1030"],
        "intensity_metric": "tCO2e/tonne product",
        "routing_group": "light_industry",
        "description": "Food processing, beverage manufacturing",
        "key_activities": ["production_tonnes", "refrigeration_kwh", "water_m3"],
    },
    "oil_gas_upstream": {
        "display_name": "Oil & Gas Upstream",
        "sda_eligible": False,
        "nace_rev2": ["B06", "B09.1"],
        "gics": "101020",
        "isic_rev4": ["0610", "0620"],
        "intensity_metric": "kgCO2e/boe",
        "routing_group": "heavy_industry",
        "description": "Oil and gas exploration, extraction, processing",
        "key_activities": ["barrels_oil_equivalent", "flaring_m3", "methane_intensity"],
    },
    "cross_sector": {
        "display_name": "Cross-Sector / Multi-Industry",
        "sda_eligible": False,
        "nace_rev2": [],
        "gics": "",
        "isic_rev4": [],
        "intensity_metric": "tCO2e/revenue_mUSD",
        "routing_group": "cross_sector",
        "description": "Diversified conglomerates, multi-sector organizations",
        "key_activities": ["revenue_usd", "employee_count", "office_area_m2"],
    },
}

ALL_SECTOR_OPTIONS: Dict[str, Dict[str, Any]] = {
    **SDA_SECTOR_OPTIONS,
    **EXTENDED_SECTOR_OPTIONS,
}

# ---------------------------------------------------------------------------
# Technology Profiles by Sector
# ---------------------------------------------------------------------------

SECTOR_TECHNOLOGY_PROFILES: Dict[str, List[Dict[str, Any]]] = {
    "power_generation": [
        {"tech_id": "coal_plant", "name": "Coal-Fired Power Plant", "maturity": "legacy", "emission_factor_relative": 1.0},
        {"tech_id": "ccgt", "name": "Combined Cycle Gas Turbine", "maturity": "best_available", "emission_factor_relative": 0.45},
        {"tech_id": "solar_pv", "name": "Solar PV (Utility Scale)", "maturity": "best_available", "emission_factor_relative": 0.05},
        {"tech_id": "onshore_wind", "name": "Onshore Wind", "maturity": "best_available", "emission_factor_relative": 0.03},
        {"tech_id": "offshore_wind", "name": "Offshore Wind", "maturity": "emerging", "emission_factor_relative": 0.04},
        {"tech_id": "nuclear", "name": "Nuclear (Gen III+)", "maturity": "best_available", "emission_factor_relative": 0.01},
        {"tech_id": "smr", "name": "Small Modular Reactor (SMR)", "maturity": "emerging", "emission_factor_relative": 0.01},
        {"tech_id": "battery_storage", "name": "Battery Energy Storage", "maturity": "emerging", "emission_factor_relative": 0.0},
        {"tech_id": "ccs_power", "name": "CCS on Fossil Generation", "maturity": "emerging", "emission_factor_relative": 0.10},
        {"tech_id": "hydrogen_turbine", "name": "Green Hydrogen Turbine", "maturity": "frontier", "emission_factor_relative": 0.02},
    ],
    "steel": [
        {"tech_id": "bf_bof", "name": "Blast Furnace / BOF", "maturity": "legacy", "emission_factor_relative": 1.0},
        {"tech_id": "eaf_scrap", "name": "Electric Arc Furnace (Scrap)", "maturity": "best_available", "emission_factor_relative": 0.25},
        {"tech_id": "dri_natural_gas", "name": "DRI with Natural Gas", "maturity": "best_available", "emission_factor_relative": 0.50},
        {"tech_id": "dri_hydrogen", "name": "DRI with Green Hydrogen", "maturity": "emerging", "emission_factor_relative": 0.05},
        {"tech_id": "bf_ccs", "name": "Blast Furnace with CCS", "maturity": "emerging", "emission_factor_relative": 0.30},
        {"tech_id": "electrolysis_iron", "name": "Molten Oxide Electrolysis", "maturity": "frontier", "emission_factor_relative": 0.03},
    ],
    "cement": [
        {"tech_id": "conventional_kiln", "name": "Conventional Rotary Kiln", "maturity": "legacy", "emission_factor_relative": 1.0},
        {"tech_id": "efficient_kiln", "name": "High-Efficiency Kiln", "maturity": "best_available", "emission_factor_relative": 0.85},
        {"tech_id": "alternative_fuels", "name": "Alternative Fuels (Biomass/Waste)", "maturity": "best_available", "emission_factor_relative": 0.70},
        {"tech_id": "blended_cement", "name": "Blended Cement (SCMs)", "maturity": "best_available", "emission_factor_relative": 0.65},
        {"tech_id": "ccus_cement", "name": "CCUS on Cement Plant", "maturity": "emerging", "emission_factor_relative": 0.15},
        {"tech_id": "geopolymer", "name": "Geopolymer Cement", "maturity": "frontier", "emission_factor_relative": 0.20},
    ],
    "aluminum": [
        {"tech_id": "hall_heroult", "name": "Hall-Heroult (Standard)", "maturity": "legacy", "emission_factor_relative": 1.0},
        {"tech_id": "hall_heroult_improved", "name": "Hall-Heroult (Improved)", "maturity": "best_available", "emission_factor_relative": 0.80},
        {"tech_id": "inert_anode", "name": "Inert Anode Technology", "maturity": "emerging", "emission_factor_relative": 0.15},
        {"tech_id": "secondary_aluminum", "name": "Secondary Aluminum (Recycling)", "maturity": "best_available", "emission_factor_relative": 0.05},
        {"tech_id": "renewable_smelting", "name": "Renewable-Powered Smelting", "maturity": "transitional", "emission_factor_relative": 0.20},
    ],
    "aviation": [
        {"tech_id": "conventional_jet", "name": "Conventional Jet Engine", "maturity": "legacy", "emission_factor_relative": 1.0},
        {"tech_id": "fuel_efficient_jet", "name": "Fuel-Efficient Aircraft (A321neo, 787)", "maturity": "best_available", "emission_factor_relative": 0.80},
        {"tech_id": "saf_blend", "name": "Sustainable Aviation Fuel (Blend)", "maturity": "transitional", "emission_factor_relative": 0.50},
        {"tech_id": "saf_100pct", "name": "100% SAF Operations", "maturity": "emerging", "emission_factor_relative": 0.15},
        {"tech_id": "hydrogen_aircraft", "name": "Hydrogen-Powered Aircraft", "maturity": "frontier", "emission_factor_relative": 0.02},
        {"tech_id": "electric_aircraft", "name": "Electric Aircraft (<500km)", "maturity": "frontier", "emission_factor_relative": 0.05},
    ],
    "shipping": [
        {"tech_id": "conventional_hfo", "name": "Conventional HFO Engine", "maturity": "legacy", "emission_factor_relative": 1.0},
        {"tech_id": "lng_dual_fuel", "name": "LNG Dual-Fuel Engine", "maturity": "transitional", "emission_factor_relative": 0.75},
        {"tech_id": "methanol_engine", "name": "Green Methanol Engine", "maturity": "emerging", "emission_factor_relative": 0.20},
        {"tech_id": "ammonia_engine", "name": "Green Ammonia Engine", "maturity": "emerging", "emission_factor_relative": 0.05},
        {"tech_id": "wind_assist", "name": "Wind-Assisted Propulsion", "maturity": "emerging", "emission_factor_relative": 0.70},
        {"tech_id": "hydrogen_fuel_cell", "name": "Hydrogen Fuel Cell Ship", "maturity": "frontier", "emission_factor_relative": 0.03},
    ],
    "road_transport": [
        {"tech_id": "diesel_ice", "name": "Diesel ICE Fleet", "maturity": "legacy", "emission_factor_relative": 1.0},
        {"tech_id": "gasoline_ice", "name": "Gasoline ICE Fleet", "maturity": "legacy", "emission_factor_relative": 0.95},
        {"tech_id": "hybrid", "name": "Hybrid Electric Vehicles", "maturity": "best_available", "emission_factor_relative": 0.60},
        {"tech_id": "phev", "name": "Plug-In Hybrid (PHEV)", "maturity": "best_available", "emission_factor_relative": 0.40},
        {"tech_id": "bev", "name": "Battery Electric Vehicles (BEV)", "maturity": "best_available", "emission_factor_relative": 0.15},
        {"tech_id": "hydrogen_fcev", "name": "Hydrogen Fuel Cell (FCEV)", "maturity": "emerging", "emission_factor_relative": 0.10},
    ],
    "buildings_residential": [
        {"tech_id": "gas_boiler", "name": "Natural Gas Boiler", "maturity": "legacy", "emission_factor_relative": 1.0},
        {"tech_id": "oil_boiler", "name": "Oil Boiler", "maturity": "legacy", "emission_factor_relative": 1.20},
        {"tech_id": "heat_pump_ashp", "name": "Air-Source Heat Pump", "maturity": "best_available", "emission_factor_relative": 0.25},
        {"tech_id": "heat_pump_gshp", "name": "Ground-Source Heat Pump", "maturity": "best_available", "emission_factor_relative": 0.20},
        {"tech_id": "district_heating", "name": "District Heating Connection", "maturity": "transitional", "emission_factor_relative": 0.30},
        {"tech_id": "rooftop_solar", "name": "Rooftop Solar PV", "maturity": "best_available", "emission_factor_relative": 0.0},
        {"tech_id": "deep_retrofit", "name": "Deep Energy Retrofit", "maturity": "best_available", "emission_factor_relative": 0.50},
    ],
    "buildings_commercial": [
        {"tech_id": "gas_hvac", "name": "Gas-Fired HVAC System", "maturity": "legacy", "emission_factor_relative": 1.0},
        {"tech_id": "electric_hvac", "name": "All-Electric HVAC", "maturity": "best_available", "emission_factor_relative": 0.30},
        {"tech_id": "vrf_system", "name": "Variable Refrigerant Flow (VRF)", "maturity": "best_available", "emission_factor_relative": 0.35},
        {"tech_id": "bems", "name": "Building Energy Management System", "maturity": "best_available", "emission_factor_relative": 0.80},
        {"tech_id": "smart_building", "name": "AI-Optimized Smart Building", "maturity": "emerging", "emission_factor_relative": 0.60},
        {"tech_id": "on_site_solar", "name": "On-Site Solar + Storage", "maturity": "best_available", "emission_factor_relative": 0.0},
    ],
    "agriculture": [
        {"tech_id": "conventional_farming", "name": "Conventional Farming", "maturity": "legacy", "emission_factor_relative": 1.0},
        {"tech_id": "precision_agriculture", "name": "Precision Agriculture", "maturity": "best_available", "emission_factor_relative": 0.75},
        {"tech_id": "low_emission_fertilizer", "name": "Low-Emission Fertilizer", "maturity": "emerging", "emission_factor_relative": 0.60},
        {"tech_id": "methane_digesters", "name": "Methane Digesters (Livestock)", "maturity": "best_available", "emission_factor_relative": 0.50},
        {"tech_id": "regenerative_ag", "name": "Regenerative Agriculture", "maturity": "transitional", "emission_factor_relative": 0.40},
    ],
}

# ---------------------------------------------------------------------------
# Abatement Lever Priorities by Sector
# ---------------------------------------------------------------------------

SECTOR_LEVER_PRIORITIES: Dict[str, List[Dict[str, Any]]] = {
    "power_generation": [
        {"lever": "renewable_expansion", "display": "Renewable Capacity Expansion", "default_priority": 1, "typical_abatement_pct": 40},
        {"lever": "coal_phaseout", "display": "Coal Plant Phase-Out", "default_priority": 2, "typical_abatement_pct": 25},
        {"lever": "grid_storage", "display": "Grid Energy Storage", "default_priority": 3, "typical_abatement_pct": 10},
        {"lever": "demand_response", "display": "Demand Response / Smart Grid", "default_priority": 4, "typical_abatement_pct": 5},
        {"lever": "nuclear_smr", "display": "Nuclear / SMR Deployment", "default_priority": 5, "typical_abatement_pct": 15},
        {"lever": "ccs_power", "display": "CCS on Fossil Generation", "default_priority": 6, "typical_abatement_pct": 5},
    ],
    "steel": [
        {"lever": "eaf_transition", "display": "Electric Arc Furnace Transition", "default_priority": 1, "typical_abatement_pct": 30},
        {"lever": "dri_hydrogen", "display": "Green Hydrogen DRI", "default_priority": 2, "typical_abatement_pct": 35},
        {"lever": "scrap_recycling", "display": "Scrap Recycling Increase", "default_priority": 3, "typical_abatement_pct": 10},
        {"lever": "bf_efficiency", "display": "Blast Furnace Efficiency", "default_priority": 4, "typical_abatement_pct": 8},
        {"lever": "ccs_steel", "display": "CCS for Integrated Plants", "default_priority": 5, "typical_abatement_pct": 12},
        {"lever": "waste_heat", "display": "Waste Heat Recovery", "default_priority": 6, "typical_abatement_pct": 5},
    ],
    "cement": [
        {"lever": "clinker_substitution", "display": "Clinker Substitution (SCMs)", "default_priority": 1, "typical_abatement_pct": 20},
        {"lever": "ccus_cement", "display": "Carbon Capture & Storage", "default_priority": 2, "typical_abatement_pct": 30},
        {"lever": "alternative_fuels", "display": "Alternative Fuels", "default_priority": 3, "typical_abatement_pct": 15},
        {"lever": "kiln_efficiency", "display": "High-Efficiency Kilns", "default_priority": 4, "typical_abatement_pct": 10},
        {"lever": "low_carbon_cement", "display": "Low-Carbon Cement Products", "default_priority": 5, "typical_abatement_pct": 15},
        {"lever": "circular_concrete", "display": "Concrete Reuse / Circular", "default_priority": 6, "typical_abatement_pct": 10},
    ],
    "aviation": [
        {"lever": "fleet_renewal", "display": "Fleet Renewal (Fuel-Efficient)", "default_priority": 1, "typical_abatement_pct": 20},
        {"lever": "saf_adoption", "display": "Sustainable Aviation Fuel", "default_priority": 2, "typical_abatement_pct": 40},
        {"lever": "operational_efficiency", "display": "Operational Efficiency", "default_priority": 3, "typical_abatement_pct": 10},
        {"lever": "load_factor", "display": "Load Factor Optimization", "default_priority": 4, "typical_abatement_pct": 5},
        {"lever": "hydrogen_aircraft", "display": "Hydrogen Aircraft (Short-Haul)", "default_priority": 5, "typical_abatement_pct": 15},
        {"lever": "carbon_offset", "display": "CORSIA Offset (Transitional)", "default_priority": 6, "typical_abatement_pct": 10},
    ],
    "shipping": [
        {"lever": "fleet_efficiency", "display": "Fleet Efficiency (Hull/Propulsion)", "default_priority": 1, "typical_abatement_pct": 15},
        {"lever": "slow_steaming", "display": "Slow Steaming & Route Optimization", "default_priority": 2, "typical_abatement_pct": 10},
        {"lever": "alternative_fuels_ship", "display": "Alternative Fuels (Methanol/Ammonia)", "default_priority": 3, "typical_abatement_pct": 40},
        {"lever": "wind_assist", "display": "Wind-Assisted Propulsion", "default_priority": 4, "typical_abatement_pct": 10},
        {"lever": "shore_power", "display": "Port Electrification (Shore Power)", "default_priority": 5, "typical_abatement_pct": 5},
        {"lever": "hydrogen_ship", "display": "Hydrogen Fuel Cell Vessels", "default_priority": 6, "typical_abatement_pct": 20},
    ],
    "buildings_residential": [
        {"lever": "heat_pump_transition", "display": "Heat Pump Transition", "default_priority": 1, "typical_abatement_pct": 35},
        {"lever": "envelope_retrofit", "display": "Building Envelope Retrofit", "default_priority": 2, "typical_abatement_pct": 25},
        {"lever": "rooftop_solar", "display": "Rooftop Solar PV", "default_priority": 3, "typical_abatement_pct": 15},
        {"lever": "district_heating", "display": "District Heating Integration", "default_priority": 4, "typical_abatement_pct": 10},
        {"lever": "smart_controls", "display": "Smart Energy Controls", "default_priority": 5, "typical_abatement_pct": 10},
        {"lever": "electrification", "display": "Full Electrification", "default_priority": 6, "typical_abatement_pct": 5},
    ],
    "buildings_commercial": [
        {"lever": "electric_hvac", "display": "All-Electric HVAC", "default_priority": 1, "typical_abatement_pct": 30},
        {"lever": "building_automation", "display": "Building Energy Management", "default_priority": 2, "typical_abatement_pct": 15},
        {"lever": "envelope_upgrade", "display": "Envelope Upgrade (Insulation/Glazing)", "default_priority": 3, "typical_abatement_pct": 20},
        {"lever": "onsite_generation", "display": "On-Site Solar + Storage", "default_priority": 4, "typical_abatement_pct": 15},
        {"lever": "green_lease", "display": "Green Lease Provisions", "default_priority": 5, "typical_abatement_pct": 10},
        {"lever": "net_zero_building", "display": "Net Zero Building Design", "default_priority": 6, "typical_abatement_pct": 10},
    ],
    "agriculture": [
        {"lever": "precision_farming", "display": "Precision Agriculture", "default_priority": 1, "typical_abatement_pct": 20},
        {"lever": "low_emission_fertilizer", "display": "Low-Emission Fertilizer", "default_priority": 2, "typical_abatement_pct": 15},
        {"lever": "methane_capture", "display": "Methane Capture (Livestock)", "default_priority": 3, "typical_abatement_pct": 20},
        {"lever": "soil_carbon", "display": "Soil Carbon Sequestration", "default_priority": 4, "typical_abatement_pct": 15},
        {"lever": "regenerative", "display": "Regenerative Practices", "default_priority": 5, "typical_abatement_pct": 20},
        {"lever": "agroforestry", "display": "Agroforestry Integration", "default_priority": 6, "typical_abatement_pct": 10},
    ],
}

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class SectorSelectionData(BaseModel):
    """Step 1: Sector selection data."""
    organization_name: str = Field(..., min_length=1, max_length=500)
    primary_sector: str = Field(...)
    sub_sector: str = Field(default="")
    nace_code: str = Field(default="")
    gics_code: str = Field(default="")
    isic_code: str = Field(default="")
    headquarters_country: str = Field(default="US")
    sda_eligible: bool = Field(default=False)
    eligibility_status: SDAEligibility = Field(default=SDAEligibility.IEA_ONLY)
    routing_group: str = Field(default="cross_sector")
    intensity_metric: str = Field(default="tCO2e/revenue_mUSD")

class ActivityDataSetup(BaseModel):
    """Step 2: Activity data configuration."""
    primary_activity_field: str = Field(default="")
    primary_activity_unit: str = Field(default="")
    primary_activity_value: float = Field(default=0.0)
    secondary_activities: Dict[str, float] = Field(default_factory=dict)
    preferred_mrv_agents: List[str] = Field(default_factory=list)
    preferred_data_agents: List[str] = Field(default_factory=list)
    data_collection_frequency: str = Field(default="monthly")
    data_quality_level: str = Field(default="measured")

class BaselineConfig(BaseModel):
    """Step 3: Baseline configuration."""
    base_year: int = Field(default=2023, ge=2015, le=2026)
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_location_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_market_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_tco2e: float = Field(default=0.0, ge=0.0)
    base_intensity: float = Field(default=0.0, ge=0.0)
    intensity_metric: str = Field(default="")
    activity_denominator: float = Field(default=0.0, ge=0.0)
    consolidation_approach: str = Field(default="operational_control")
    coverage_scope1_pct: float = Field(default=100.0, ge=0.0, le=100.0)
    coverage_scope2_pct: float = Field(default=100.0, ge=0.0, le=100.0)
    use_pack021_baseline: bool = Field(default=True)

class PathwayPreferences(BaseModel):
    """Step 4: Pathway preferences."""
    primary_scenario: ScenarioPreference = Field(default=ScenarioPreference.NZE_15C)
    comparison_scenarios: List[ScenarioPreference] = Field(
        default_factory=lambda: [ScenarioPreference.WB2C, ScenarioPreference.STEPS]
    )
    convergence_model: ConvergencePreference = Field(default=ConvergencePreference.AUTO)
    near_term_target_year: int = Field(default=2030, ge=2025, le=2035)
    long_term_target_year: int = Field(default=2050, ge=2040, le=2070)
    near_term_reduction_rate_pct: float = Field(default=4.2)
    enable_sda_pathway: bool = Field(default=True)
    enable_iea_pathway: bool = Field(default=True)
    enable_flag_pathway: bool = Field(default=False)
    regional_context: str = Field(default="global")

class TechnologyInventoryData(BaseModel):
    """Step 5: Technology inventory."""
    current_technologies: List[Dict[str, Any]] = Field(default_factory=list)
    technology_maturity_overall: TechnologyMaturity = Field(default=TechnologyMaturity.LEGACY)
    planned_investments: List[Dict[str, Any]] = Field(default_factory=list)
    technology_barriers: List[str] = Field(default_factory=list)
    capex_budget_eur: float = Field(default=0.0, ge=0.0)
    technology_readiness_score: float = Field(default=0.0, ge=0.0, le=100.0)

class AbatementPlanData(BaseModel):
    """Step 6: Abatement planning."""
    enabled_levers: List[str] = Field(default_factory=list)
    lever_priorities: Dict[str, int] = Field(default_factory=dict)
    total_target_abatement_tco2e: float = Field(default=0.0)
    abatement_budget_eur: float = Field(default=0.0)
    implementation_start_year: int = Field(default=2025)
    implementation_phases: int = Field(default=3)
    max_cost_per_tco2e_eur: float = Field(default=200.0)

class WizardStepState(BaseModel):
    """State of a single wizard step."""
    name: SectorWizardStep = Field(...)
    display_name: str = Field(default="")
    status: StepStatus = Field(default=StepStatus.PENDING)
    data: Dict[str, Any] = Field(default_factory=dict)
    validation_errors: List[str] = Field(default_factory=list)
    execution_time_ms: float = Field(default=0.0)

class WizardState(BaseModel):
    """Complete wizard state."""
    wizard_id: str = Field(default="")
    current_step: SectorWizardStep = Field(
        default=SectorWizardStep.SECTOR_SELECTION
    )
    steps: Dict[str, WizardStepState] = Field(default_factory=dict)
    is_complete: bool = Field(default=False)
    created_at: datetime = Field(default_factory=utcnow)
    completed_at: Optional[datetime] = Field(None)

class SectorPathwaySetupResult(BaseModel):
    """Complete setup configuration result."""
    result_id: str = Field(default_factory=_new_uuid)
    organization_name: str = Field(default="")
    primary_sector: str = Field(default="")
    sector_display_name: str = Field(default="")
    sda_eligible: bool = Field(default=False)
    eligibility_status: str = Field(default="iea_only")
    routing_group: str = Field(default="cross_sector")
    nace_code: str = Field(default="")
    gics_code: str = Field(default="")
    isic_code: str = Field(default="")
    intensity_metric: str = Field(default="")
    base_year: int = Field(default=2023)
    base_intensity: float = Field(default=0.0)
    scope1_tco2e: float = Field(default=0.0)
    scope2_tco2e: float = Field(default=0.0)
    scope3_tco2e: float = Field(default=0.0)
    primary_scenario: str = Field(default="nze_1.5c")
    convergence_model: str = Field(default="auto")
    near_term_target_year: int = Field(default=2030)
    long_term_target_year: int = Field(default=2050)
    near_term_reduction_rate_pct: float = Field(default=4.2)
    technology_maturity: str = Field(default="legacy")
    technology_readiness_score: float = Field(default=0.0)
    enabled_levers: List[str] = Field(default_factory=list)
    lever_count: int = Field(default=0)
    capex_budget_eur: float = Field(default=0.0)
    engines_enabled: List[str] = Field(default_factory=list)
    workflows_enabled: List[str] = Field(default_factory=list)
    mrv_priority_agents: List[str] = Field(default_factory=list)
    data_priority_agents: List[str] = Field(default_factory=list)
    total_steps_completed: int = Field(default=0)
    total_steps: int = Field(default=7)
    configuration_hash: str = Field(default="")
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Step Definitions
# ---------------------------------------------------------------------------

STEP_ORDER: List[SectorWizardStep] = [
    SectorWizardStep.SECTOR_SELECTION,
    SectorWizardStep.ACTIVITY_DATA_SETUP,
    SectorWizardStep.BASELINE_CONFIGURATION,
    SectorWizardStep.PATHWAY_PREFERENCES,
    SectorWizardStep.TECHNOLOGY_INVENTORY,
    SectorWizardStep.ABATEMENT_PLANNING,
    SectorWizardStep.REVIEW_AND_DEPLOY,
]

STEP_DISPLAY_NAMES: Dict[SectorWizardStep, str] = {
    SectorWizardStep.SECTOR_SELECTION: "Sector Selection & Classification",
    SectorWizardStep.ACTIVITY_DATA_SETUP: "Activity Data Configuration",
    SectorWizardStep.BASELINE_CONFIGURATION: "Baseline Configuration",
    SectorWizardStep.PATHWAY_PREFERENCES: "Pathway & Scenario Preferences",
    SectorWizardStep.TECHNOLOGY_INVENTORY: "Technology Portfolio Assessment",
    SectorWizardStep.ABATEMENT_PLANNING: "Abatement Lever Planning",
    SectorWizardStep.REVIEW_AND_DEPLOY: "Review & Deploy Configuration",
}

STEP_DESCRIPTIONS: Dict[SectorWizardStep, str] = {
    SectorWizardStep.SECTOR_SELECTION:
        "Select your primary sector and verify SBTi SDA eligibility. The system will "
        "automatically map NACE Rev.2, GICS, and ISIC Rev.4 codes and determine whether "
        "SDA convergence pathways apply.",
    SectorWizardStep.ACTIVITY_DATA_SETUP:
        "Configure sector-specific activity data fields (e.g., electricity in MWh for power, "
        "crude steel in tonnes for steel). This determines which MRV and DATA agents are "
        "prioritized for your sector.",
    SectorWizardStep.BASELINE_CONFIGURATION:
        "Set your GHG emissions base year, Scope 1+2+3 emissions, and sector intensity "
        "baseline. Optionally import baseline data from PACK-021 Net Zero Starter Pack.",
    SectorWizardStep.PATHWAY_PREFERENCES:
        "Choose your climate scenario (NZE 1.5C, WB2C, 2C, APS, STEPS), convergence model "
        "(linear, exponential, S-curve, stepped), and target years for near-term and long-term.",
    SectorWizardStep.TECHNOLOGY_INVENTORY:
        "Declare your current technology portfolio and assess technology maturity. This feeds "
        "into the technology roadmap engine for transition planning.",
    SectorWizardStep.ABATEMENT_PLANNING:
        "Select and prioritize decarbonization levers for your sector. Configure abatement "
        "budget, implementation timeline, and cost thresholds.",
    SectorWizardStep.REVIEW_AND_DEPLOY:
        "Review complete configuration and deploy the sector pathway analysis pipeline.",
}

# ---------------------------------------------------------------------------
# MRV/DATA Agent Priority by Routing Group
# ---------------------------------------------------------------------------

ROUTING_GROUP_MRV_PRIORITIES: Dict[str, List[str]] = {
    "heavy_industry": [
        "MRV-001", "MRV-003", "MRV-004", "MRV-005",  # Stationary, mobile, process, fugitive
        "MRV-002",  # Refrigerants
        "MRV-009", "MRV-010",  # Scope 2
        "MRV-014", "MRV-016", "MRV-018",  # Scope 3: Cat 1, 3, 5
    ],
    "light_industry": [
        "MRV-001", "MRV-002", "MRV-003",
        "MRV-009", "MRV-010",
        "MRV-014", "MRV-015", "MRV-018",  # Cat 1, 2, 5
    ],
    "transport": [
        "MRV-003",  # Mobile combustion (critical)
        "MRV-001", "MRV-005",  # Stationary, fugitive
        "MRV-009", "MRV-010",  # Scope 2
        "MRV-016", "MRV-017",  # Cat 3, 4
    ],
    "power": [
        "MRV-001",  # Stationary combustion (critical)
        "MRV-005", "MRV-006",  # Fugitive, land use
        "MRV-009", "MRV-010",  # Scope 2
        "MRV-016",  # Cat 3
    ],
    "buildings": [
        "MRV-001", "MRV-002",  # Stationary, refrigerants
        "MRV-009", "MRV-010",  # Scope 2 (critical for buildings)
        "MRV-011",  # Steam/heat
        "MRV-014", "MRV-020",  # Cat 1, Cat 7
    ],
    "agriculture": [
        "MRV-006", "MRV-008",  # Land use, agricultural (critical)
        "MRV-001", "MRV-003",  # Stationary, mobile
        "MRV-005",  # Fugitive
        "MRV-014", "MRV-018",  # Cat 1, Cat 5
    ],
    "cross_sector": [
        "MRV-001", "MRV-002", "MRV-003",
        "MRV-009", "MRV-010",
        "MRV-014", "MRV-019", "MRV-020",  # Cat 1, Cat 6, Cat 7
    ],
}

ROUTING_GROUP_DATA_PRIORITIES: Dict[str, List[str]] = {
    "heavy_industry": ["DATA-001", "DATA-002", "DATA-003", "DATA-010", "DATA-015"],
    "light_industry": ["DATA-001", "DATA-002", "DATA-003", "DATA-008", "DATA-010"],
    "transport": ["DATA-002", "DATA-003", "DATA-004", "DATA-010", "DATA-015"],
    "power": ["DATA-002", "DATA-003", "DATA-010", "DATA-014", "DATA-015"],
    "buildings": ["DATA-002", "DATA-003", "DATA-006", "DATA-010", "DATA-015"],
    "agriculture": ["DATA-002", "DATA-003", "DATA-006", "DATA-010", "DATA-020"],
    "cross_sector": ["DATA-001", "DATA-002", "DATA-003", "DATA-008", "DATA-010"],
}

# ---------------------------------------------------------------------------
# SectorPathwaySetupWizard
# ---------------------------------------------------------------------------

class SectorPathwaySetupWizard:
    """7-step sector pathway configuration wizard for PACK-028.

    Guides organizations through sector selection, activity data setup,
    baseline configuration, pathway preferences, technology inventory,
    abatement planning, and review/deployment.

    Example:
        >>> wizard = SectorPathwaySetupWizard()
        >>> state = wizard.start()
        >>> state = wizard.complete_step("sector_selection", {
        ...     "organization_name": "Steel Corp",
        ...     "primary_sector": "steel",
        ... })
        >>> result = wizard.generate_config()

    Demo Mode:
        >>> result = wizard.run_demo(sector="steel")
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self._state: Optional[WizardState] = None
        self._validators: Dict[SectorWizardStep, callable] = {
            SectorWizardStep.SECTOR_SELECTION: self._validate_sector_selection,
            SectorWizardStep.ACTIVITY_DATA_SETUP: self._validate_activity_data,
            SectorWizardStep.BASELINE_CONFIGURATION: self._validate_baseline,
            SectorWizardStep.PATHWAY_PREFERENCES: self._validate_pathway_prefs,
            SectorWizardStep.TECHNOLOGY_INVENTORY: self._validate_technology,
            SectorWizardStep.ABATEMENT_PLANNING: self._validate_abatement,
            SectorWizardStep.REVIEW_AND_DEPLOY: self._validate_review,
        }
        self.logger.info("SectorPathwaySetupWizard initialized: 7 steps")

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def start(self) -> WizardState:
        """Start a new wizard session."""
        wizard_id = _compute_hash(f"sector-wizard:{utcnow().isoformat()}")[:16]
        steps: Dict[str, WizardStepState] = {}
        for step in STEP_ORDER:
            steps[step.value] = WizardStepState(
                name=step,
                display_name=STEP_DISPLAY_NAMES.get(step, step.value),
            )
        self._state = WizardState(
            wizard_id=wizard_id, current_step=STEP_ORDER[0], steps=steps,
        )
        self.logger.info("Wizard session started: %s", wizard_id)
        return self._state

    def complete_step(self, step_name: str, data: Dict[str, Any]) -> WizardState:
        """Complete a wizard step with provided data."""
        if self._state is None:
            raise RuntimeError("Wizard must be started first (call start())")

        try:
            step_enum = SectorWizardStep(step_name)
        except ValueError:
            raise ValueError(f"Unknown step '{step_name}'. Valid: {[s.value for s in SectorWizardStep]}")

        step = self._state.steps.get(step_name)
        if not step:
            raise ValueError(f"Step '{step_name}' not found in wizard state")

        step.status = StepStatus.IN_PROGRESS
        start_time = time.monotonic()

        try:
            # Auto-enrich sector selection data
            if step_enum == SectorWizardStep.SECTOR_SELECTION:
                data = self._enrich_sector_selection(data)
            elif step_enum == SectorWizardStep.ACTIVITY_DATA_SETUP:
                data = self._enrich_activity_data(data)
            elif step_enum == SectorWizardStep.TECHNOLOGY_INVENTORY:
                data = self._enrich_technology_inventory(data)
            elif step_enum == SectorWizardStep.ABATEMENT_PLANNING:
                data = self._enrich_abatement_planning(data)

            # Validate
            validator = self._validators.get(step_enum)
            errors = validator(data) if validator else []

            step.data = data
            step.execution_time_ms = round((time.monotonic() - start_time) * 1000, 1)

            if errors:
                step.status = StepStatus.FAILED
                step.validation_errors = errors
            else:
                step.status = StepStatus.COMPLETED
                step.validation_errors = []
                self._advance_step(step_enum)

        except Exception as exc:
            step.status = StepStatus.FAILED
            step.validation_errors = [str(exc)]
            step.execution_time_ms = round((time.monotonic() - start_time) * 1000, 1)

        return self._state

    def generate_config(self) -> SectorPathwaySetupResult:
        """Generate the final configuration from wizard state."""
        if self._state is None:
            return SectorPathwaySetupResult()

        completed = sum(
            1 for s in self._state.steps.values() if s.status == StepStatus.COMPLETED
        )

        # Extract data from each step
        sector_data = self._get_step_data("sector_selection")
        activity_data = self._get_step_data("activity_data_setup")
        baseline_data = self._get_step_data("baseline_configuration")
        pathway_data = self._get_step_data("pathway_preferences")
        tech_data = self._get_step_data("technology_inventory")
        abatement_data = self._get_step_data("abatement_planning")

        primary_sector = sector_data.get("primary_sector", "cross_sector")
        routing_group = sector_data.get("routing_group", "cross_sector")

        # Determine engines to enable
        engines_enabled = [
            "sector_classification_engine",
            "intensity_calculator_engine",
            "pathway_generator_engine",
            "convergence_analyzer_engine",
            "technology_roadmap_engine",
            "abatement_waterfall_engine",
            "sector_benchmark_engine",
            "scenario_comparison_engine",
        ]

        # Determine workflows to enable
        workflows_enabled = [
            "sector_pathway_design_workflow",
            "pathway_validation_workflow",
            "technology_planning_workflow",
            "progress_monitoring_workflow",
            "multi_scenario_analysis_workflow",
        ]

        # Get priority agents
        mrv_priorities = ROUTING_GROUP_MRV_PRIORITIES.get(routing_group, ROUTING_GROUP_MRV_PRIORITIES["cross_sector"])
        data_priorities = ROUTING_GROUP_DATA_PRIORITIES.get(routing_group, ROUTING_GROUP_DATA_PRIORITIES["cross_sector"])

        result = SectorPathwaySetupResult(
            organization_name=sector_data.get("organization_name", ""),
            primary_sector=primary_sector,
            sector_display_name=sector_data.get("sector_display_name", primary_sector),
            sda_eligible=sector_data.get("sda_eligible", False),
            eligibility_status=sector_data.get("eligibility_status", "iea_only"),
            routing_group=routing_group,
            nace_code=sector_data.get("nace_code", ""),
            gics_code=sector_data.get("gics_code", ""),
            isic_code=sector_data.get("isic_code", ""),
            intensity_metric=sector_data.get("intensity_metric", ""),
            base_year=baseline_data.get("base_year", 2023),
            base_intensity=baseline_data.get("base_intensity", 0.0),
            scope1_tco2e=baseline_data.get("scope1_tco2e", 0.0),
            scope2_tco2e=baseline_data.get("scope2_location_tco2e", 0.0),
            scope3_tco2e=baseline_data.get("scope3_tco2e", 0.0),
            primary_scenario=pathway_data.get("primary_scenario", "nze_1.5c"),
            convergence_model=pathway_data.get("convergence_model", "auto"),
            near_term_target_year=pathway_data.get("near_term_target_year", 2030),
            long_term_target_year=pathway_data.get("long_term_target_year", 2050),
            near_term_reduction_rate_pct=pathway_data.get("near_term_reduction_rate_pct", 4.2),
            technology_maturity=tech_data.get("technology_maturity_overall", "legacy"),
            technology_readiness_score=tech_data.get("technology_readiness_score", 0.0),
            enabled_levers=abatement_data.get("enabled_levers", []),
            lever_count=len(abatement_data.get("enabled_levers", [])),
            capex_budget_eur=tech_data.get("capex_budget_eur", 0.0),
            engines_enabled=engines_enabled,
            workflows_enabled=workflows_enabled,
            mrv_priority_agents=mrv_priorities,
            data_priority_agents=data_priorities,
            total_steps_completed=completed,
            configuration_hash=_compute_hash(sector_data),
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "Config generated for %s (%s): %d/%d steps, SDA=%s",
            result.organization_name, result.primary_sector,
            completed, 7, result.sda_eligible,
        )
        return result

    def run_demo(self, sector: str = "steel") -> SectorPathwaySetupResult:
        """Run a complete demo with sector-specific defaults."""
        self.start()

        sector_info = ALL_SECTOR_OPTIONS.get(sector, ALL_SECTOR_OPTIONS["steel"])
        routing_group = sector_info.get("routing_group", "heavy_industry")
        key_activities = sector_info.get("key_activities", [])
        intensity_metric = sector_info.get("intensity_metric", "tCO2e/tonne")

        # Step 1: Sector selection
        self.complete_step("sector_selection", {
            "organization_name": f"Demo {sector_info.get('display_name', sector)} Corp",
            "primary_sector": sector,
        })

        # Step 2: Activity data
        activity_value = {
            "power_generation": 50000.0,
            "steel": 1000000.0,
            "cement": 500000.0,
            "aluminum": 200000.0,
            "aviation": 1e9,
            "shipping": 5e8,
            "road_transport": 1e7,
            "buildings_residential": 500000.0,
            "buildings_commercial": 300000.0,
            "agriculture": 10000.0,
        }.get(sector, 100000.0)

        self.complete_step("activity_data_setup", {
            "primary_activity_field": key_activities[0] if key_activities else "production_units",
            "primary_activity_value": activity_value,
            "data_collection_frequency": "monthly",
        })

        # Step 3: Baseline
        demo_baselines = {
            "power_generation": {"scope1": 2000000, "scope2": 50000, "intensity": 400},
            "steel": {"scope1": 1500000, "scope2": 200000, "intensity": 1.85},
            "cement": {"scope1": 800000, "scope2": 100000, "intensity": 0.63},
            "aluminum": {"scope1": 500000, "scope2": 300000, "intensity": 12.0},
            "aviation": {"scope1": 3000000, "scope2": 50000, "intensity": 90},
            "shipping": {"scope1": 1000000, "scope2": 20000, "intensity": 12},
            "road_transport": {"scope1": 400000, "scope2": 30000, "intensity": 200},
            "buildings_residential": {"scope1": 100000, "scope2": 200000, "intensity": 35},
            "buildings_commercial": {"scope1": 50000, "scope2": 150000, "intensity": 60},
            "agriculture": {"scope1": 200000, "scope2": 50000, "intensity": 5.0},
        }
        bl = demo_baselines.get(sector, {"scope1": 500000, "scope2": 100000, "intensity": 1.0})

        self.complete_step("baseline_configuration", {
            "base_year": 2023,
            "scope1_tco2e": bl["scope1"],
            "scope2_location_tco2e": bl["scope2"],
            "scope3_tco2e": bl["scope1"] * 0.5,
            "base_intensity": bl["intensity"],
            "intensity_metric": intensity_metric,
            "activity_denominator": activity_value,
        })

        # Step 4: Pathway preferences
        self.complete_step("pathway_preferences", {
            "primary_scenario": "nze_1.5c",
            "comparison_scenarios": ["wb2c", "steps"],
            "convergence_model": "auto",
            "near_term_target_year": 2030,
            "long_term_target_year": 2050,
            "near_term_reduction_rate_pct": 4.2,
            "enable_sda_pathway": sector_info.get("sda_eligible", False),
            "enable_iea_pathway": True,
        })

        # Step 5: Technology inventory
        sector_techs = SECTOR_TECHNOLOGY_PROFILES.get(sector, [])
        current_techs = [
            {"tech_id": t["tech_id"], "share_pct": 50.0 if i == 0 else 30.0 if i == 1 else 20.0 / max(len(sector_techs) - 2, 1)}
            for i, t in enumerate(sector_techs[:3])
        ]

        self.complete_step("technology_inventory", {
            "current_technologies": current_techs,
            "technology_maturity_overall": "transitional",
            "capex_budget_eur": 50_000_000,
            "technology_readiness_score": 45.0,
        })

        # Step 6: Abatement planning
        sector_levers = SECTOR_LEVER_PRIORITIES.get(sector, [])
        enabled_levers = [l["lever"] for l in sector_levers]
        lever_priorities = {l["lever"]: l["default_priority"] for l in sector_levers}

        self.complete_step("abatement_planning", {
            "enabled_levers": enabled_levers,
            "lever_priorities": lever_priorities,
            "total_target_abatement_tco2e": bl["scope1"] * 0.50,
            "abatement_budget_eur": 30_000_000,
            "implementation_start_year": 2025,
            "implementation_phases": 3,
        })

        # Step 7: Review
        self.complete_step("review_and_deploy", {
            "confirmed": True,
            "deployment_mode": "production",
        })

        return self.generate_config()

    def get_state(self) -> Optional[WizardState]:
        """Get current wizard state."""
        return self._state

    def get_step_info(self) -> List[Dict[str, Any]]:
        """Get step information with current status."""
        return [
            {
                "step": s.value,
                "display_name": STEP_DISPLAY_NAMES.get(s, ""),
                "description": STEP_DESCRIPTIONS.get(s, ""),
                "status": (
                    self._state.steps[s.value].status.value
                    if self._state and s.value in self._state.steps
                    else "pending"
                ),
            }
            for s in STEP_ORDER
        ]

    def get_sector_options(self) -> Dict[str, Dict[str, Any]]:
        """Get all available sector options with SDA eligibility."""
        return {
            sector: {
                "display_name": info["display_name"],
                "sda_eligible": info["sda_eligible"],
                "intensity_metric": info["intensity_metric"],
                "routing_group": info["routing_group"],
                "description": info["description"],
            }
            for sector, info in ALL_SECTOR_OPTIONS.items()
        }

    def get_technology_options(self, sector: str) -> List[Dict[str, Any]]:
        """Get available technology options for a sector."""
        return SECTOR_TECHNOLOGY_PROFILES.get(sector, [])

    def get_lever_options(self, sector: str) -> List[Dict[str, Any]]:
        """Get available abatement levers for a sector."""
        return SECTOR_LEVER_PRIORITIES.get(sector, [])

    # -------------------------------------------------------------------
    # Enrichment
    # -------------------------------------------------------------------

    def _enrich_sector_selection(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Auto-fill sector classification fields based on selected sector."""
        sector = data.get("primary_sector", "")
        sector_info = ALL_SECTOR_OPTIONS.get(sector, {})

        if sector_info:
            data.setdefault("sda_eligible", sector_info.get("sda_eligible", False))
            data.setdefault("routing_group", sector_info.get("routing_group", "cross_sector"))
            data.setdefault("intensity_metric", sector_info.get("intensity_metric", ""))
            data.setdefault("sector_display_name", sector_info.get("display_name", sector))

            nace_codes = sector_info.get("nace_rev2", [])
            data.setdefault("nace_code", nace_codes[0] if nace_codes else "")
            data.setdefault("gics_code", sector_info.get("gics", ""))

            isic_codes = sector_info.get("isic_rev4", [])
            data.setdefault("isic_code", isic_codes[0] if isic_codes else "")

            if data.get("sda_eligible"):
                data.setdefault("eligibility_status", "sda_eligible")
            else:
                data.setdefault("eligibility_status", "iea_only")

        return data

    def _enrich_activity_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Auto-fill activity data fields based on sector from step 1."""
        sector_data = self._get_step_data("sector_selection")
        sector = sector_data.get("primary_sector", "")
        routing_group = sector_data.get("routing_group", "cross_sector")
        sector_info = ALL_SECTOR_OPTIONS.get(sector, {})

        if sector_info:
            key_activities = sector_info.get("key_activities", [])
            if key_activities and not data.get("primary_activity_field"):
                data["primary_activity_field"] = key_activities[0]

            # Set preferred MRV agents
            data.setdefault(
                "preferred_mrv_agents",
                ROUTING_GROUP_MRV_PRIORITIES.get(routing_group, ROUTING_GROUP_MRV_PRIORITIES["cross_sector"]),
            )
            data.setdefault(
                "preferred_data_agents",
                ROUTING_GROUP_DATA_PRIORITIES.get(routing_group, ROUTING_GROUP_DATA_PRIORITIES["cross_sector"]),
            )

        return data

    def _enrich_technology_inventory(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Auto-calculate technology readiness score if not provided."""
        if not data.get("technology_readiness_score") and data.get("current_technologies"):
            # Calculate based on maturity distribution
            sector_data = self._get_step_data("sector_selection")
            sector = sector_data.get("primary_sector", "")
            profiles = SECTOR_TECHNOLOGY_PROFILES.get(sector, [])

            maturity_scores = {
                "legacy": 10, "transitional": 30,
                "best_available": 60, "emerging": 80, "frontier": 95,
            }

            total_share = 0.0
            weighted_score = 0.0
            for tech in data["current_technologies"]:
                tech_id = tech.get("tech_id", "")
                share = tech.get("share_pct", 0.0)
                # Find maturity from profile
                maturity = "legacy"
                for p in profiles:
                    if p["tech_id"] == tech_id:
                        maturity = p.get("maturity", "legacy")
                        break
                weighted_score += share * maturity_scores.get(maturity, 10)
                total_share += share

            if total_share > 0:
                data["technology_readiness_score"] = round(weighted_score / total_share, 1)

        return data

    def _enrich_abatement_planning(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Auto-populate lever list if not provided."""
        if not data.get("enabled_levers"):
            sector_data = self._get_step_data("sector_selection")
            sector = sector_data.get("primary_sector", "")
            sector_levers = SECTOR_LEVER_PRIORITIES.get(sector, [])
            data["enabled_levers"] = [l["lever"] for l in sector_levers]
            data["lever_priorities"] = {l["lever"]: l["default_priority"] for l in sector_levers}

        return data

    # -------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------

    def _validate_sector_selection(self, data: Dict[str, Any]) -> List[str]:
        """Validate sector selection step."""
        errors: List[str] = []

        if not data.get("organization_name"):
            errors.append("Organization name is required")

        primary_sector = data.get("primary_sector", "")
        if not primary_sector:
            errors.append("Primary sector must be selected")
        elif primary_sector not in ALL_SECTOR_OPTIONS:
            errors.append(
                f"Unknown sector '{primary_sector}'. Valid options: {list(ALL_SECTOR_OPTIONS.keys())}"
            )

        return errors

    def _validate_activity_data(self, data: Dict[str, Any]) -> List[str]:
        """Validate activity data setup step."""
        errors: List[str] = []

        if not data.get("primary_activity_field"):
            errors.append("Primary activity field must be specified")

        val = data.get("primary_activity_value", 0)
        if val <= 0:
            errors.append("Primary activity value must be > 0")

        return errors

    def _validate_baseline(self, data: Dict[str, Any]) -> List[str]:
        """Validate baseline configuration step."""
        errors: List[str] = []

        base_year = data.get("base_year", 0)
        if base_year < 2015 or base_year > 2026:
            errors.append("Base year must be between 2015 and 2026")

        scope1 = data.get("scope1_tco2e", 0)
        scope2 = data.get("scope2_location_tco2e", 0)
        if scope1 <= 0 and scope2 <= 0:
            errors.append("At least Scope 1 or Scope 2 emissions must be > 0")

        return errors

    def _validate_pathway_prefs(self, data: Dict[str, Any]) -> List[str]:
        """Validate pathway preferences step."""
        errors: List[str] = []

        scenario = data.get("primary_scenario", "")
        valid_scenarios = [s.value for s in ScenarioPreference]
        if scenario and scenario not in valid_scenarios:
            errors.append(f"Invalid scenario '{scenario}'. Valid: {valid_scenarios}")

        near = data.get("near_term_target_year", 2030)
        long = data.get("long_term_target_year", 2050)
        if near >= long:
            errors.append("Near-term target year must be before long-term target year")

        # SDA pathway validation
        sector_data = self._get_step_data("sector_selection")
        sda_eligible = sector_data.get("sda_eligible", False)
        enable_sda = data.get("enable_sda_pathway", False)
        if enable_sda and not sda_eligible:
            errors.append(
                f"SDA pathway not available for sector '{sector_data.get('primary_sector')}'. "
                f"SDA is only available for: {list(SDA_SECTOR_OPTIONS.keys())}"
            )

        return errors

    def _validate_technology(self, data: Dict[str, Any]) -> List[str]:
        """Validate technology inventory step."""
        errors: List[str] = []

        techs = data.get("current_technologies", [])
        if techs:
            total_share = sum(t.get("share_pct", 0) for t in techs)
            if abs(total_share - 100.0) > 5.0:
                errors.append(f"Technology shares sum to {total_share}%, should be ~100%")

        return errors

    def _validate_abatement(self, data: Dict[str, Any]) -> List[str]:
        """Validate abatement planning step."""
        errors: List[str] = []

        levers = data.get("enabled_levers", [])
        if not levers:
            errors.append("At least one abatement lever must be enabled")

        target = data.get("total_target_abatement_tco2e", 0)
        if target < 0:
            errors.append("Target abatement must be >= 0")

        return errors

    def _validate_review(self, data: Dict[str, Any]) -> List[str]:
        """Validate review and deploy step."""
        errors: List[str] = []

        if not data.get("confirmed", False):
            errors.append("Configuration must be confirmed before deployment")

        # Check prerequisite steps
        if self._state:
            required = [
                SectorWizardStep.SECTOR_SELECTION,
                SectorWizardStep.BASELINE_CONFIGURATION,
                SectorWizardStep.PATHWAY_PREFERENCES,
            ]
            for req in required:
                step = self._state.steps.get(req.value)
                if not step or step.status != StepStatus.COMPLETED:
                    errors.append(f"Step '{req.value}' must be completed before deployment")

        return errors

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _get_step_data(self, step_name: str) -> Dict[str, Any]:
        """Get data from a specific step."""
        if self._state is None:
            return {}
        step = self._state.steps.get(step_name)
        return step.data if step else {}

    def _advance_step(self, current: SectorWizardStep) -> None:
        """Advance to the next step after successful completion."""
        if self._state is None:
            return
        try:
            idx = STEP_ORDER.index(current)
            if idx < len(STEP_ORDER) - 1:
                self._state.current_step = STEP_ORDER[idx + 1]
            else:
                self._state.is_complete = True
                self._state.completed_at = utcnow()
        except ValueError:
            pass
