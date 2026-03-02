"""
Use of Sold Products Agent Models (AGENT-MRV-024)

This module provides comprehensive data models for GHG Protocol Scope 3 Category 11
(Use of Sold Products) emissions calculations.

Supports:
- 10 product use categories (vehicles, appliances, HVAC, lighting, IT, industrial,
  fuels/feedstocks, building products, consumer products, medical devices)
- 6 use-phase emission types (3 direct + 3 indirect)
- 8 calculation methods (direct fuel/refrigerant/chemical, indirect
  electricity/heating/steam, fuels sold, feedstocks sold)
- 24 product energy profiles with default lifetimes and annual consumption
- 15 fuel combustion emission factors with NCV values
- 10 refrigerant GWPs (AR5 and AR6) with charge and leak rate defaults
- 16 grid emission factors by region (aligned with MRV-023)
- 5 lifetime adjustment factors (standard/heavy/light/industrial/seasonal)
- 6 energy degradation rates by product category
- 8 double-counting prevention rules (DC-USP-001 through DC-USP-008)
- 7 compliance frameworks (GHG Protocol, ISO 14064, CSRD, CDP, SBTi, SB 253, GRI)
- Data quality indicators (DQI) with 5-dimension scoring
- Uncertainty quantification (Monte Carlo, analytical, IPCC Tier 2)
- SHA-256 provenance chain with 10-stage pipeline
- Product lifetime modeling with degradation curves

All numeric fields use Decimal for precision in regulatory calculations.
All models are frozen (immutable) for audit trail integrity.

Example:
    >>> from greenlang.use_of_sold_products.models import (
    ...     ProductInput, ProductUseCategory, VehicleType, FuelType, GridRegion
    ... )
    >>> product = ProductInput(
    ...     product_id="PROD-001",
    ...     category=ProductUseCategory.VEHICLES,
    ...     product_type="PASSENGER_CAR_GASOLINE",
    ...     units_sold=Decimal("10000"),
    ...     lifetime_years=Decimal("15"),
    ...     fuel_type=FuelType.GASOLINE,
    ...     use_region=GridRegion.US
    ... )
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
from pydantic import ConfigDict
import hashlib
import json

# ==============================================================================
# AGENT METADATA
# ==============================================================================

AGENT_ID: str = "GL-MRV-S3-011"
AGENT_COMPONENT: str = "AGENT-MRV-024"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_usp_"

# ==============================================================================
# ENUMERATIONS (22)
# ==============================================================================


class ProductUseCategory(str, Enum):
    """Product use categories for Scope 3 Category 11 per GHG Protocol."""

    VEHICLES = "vehicles"  # Cars, trucks, motorcycles (direct fuel combustion)
    APPLIANCES = "appliances"  # Refrigerators, washing machines, ovens (indirect)
    HVAC = "hvac"  # Air conditioners, heat pumps, furnaces (direct + indirect)
    LIGHTING = "lighting"  # LED bulbs, CFL bulbs (indirect electricity)
    IT_EQUIPMENT = "it_equipment"  # Laptops, desktops, servers, monitors (indirect)
    INDUSTRIAL_EQUIPMENT = "industrial_equipment"  # Generators, boilers, compressors
    FUELS_FEEDSTOCKS = "fuels_feedstocks"  # Gasoline, diesel, natural gas, coal
    BUILDING_PRODUCTS = "building_products"  # Windows, insulation, HVAC ducts
    CONSUMER_PRODUCTS = "consumer_products"  # Aerosols, solvents, fertilizers
    MEDICAL_DEVICES = "medical_devices"  # Imaging equipment, ventilators


class UsePhaseEmissionType(str, Enum):
    """Use-phase emission types per GHG Protocol Scope 3 Category 11."""

    DIRECT_FUEL_COMBUSTION = "direct_fuel_combustion"  # Fuel burned in product
    DIRECT_REFRIGERANT_LEAKAGE = "direct_refrigerant_leakage"  # Refrigerant/F-gas leaks
    DIRECT_CHEMICAL_RELEASE = "direct_chemical_release"  # Chemical GHG release during use
    INDIRECT_ELECTRICITY = "indirect_electricity"  # Electricity consumed during use
    INDIRECT_FUEL_HEATING = "indirect_fuel_heating"  # Fuel for heating during use
    INDIRECT_STEAM_COOLING = "indirect_steam_cooling"  # District heating/cooling consumed


class VehicleType(str, Enum):
    """Vehicle types for direct fuel combustion calculations."""

    PASSENGER_CAR_GASOLINE = "passenger_car_gasoline"  # Gasoline passenger car
    PASSENGER_CAR_DIESEL = "passenger_car_diesel"  # Diesel passenger car
    PASSENGER_CAR_EV = "passenger_car_ev"  # Electric vehicle (indirect)
    LIGHT_TRUCK = "light_truck"  # Light-duty truck / SUV
    HEAVY_TRUCK = "heavy_truck"  # Heavy-duty truck / commercial vehicle
    MOTORCYCLE = "motorcycle"  # Motorcycle / scooter


class ApplianceType(str, Enum):
    """Appliance types for indirect electricity calculations."""

    REFRIGERATOR = "refrigerator"  # Refrigerator / fridge-freezer
    WASHING_MACHINE = "washing_machine"  # Washing machine / washer
    DISHWASHER = "dishwasher"  # Dishwasher
    DRYER = "dryer"  # Tumble dryer / clothes dryer
    OVEN = "oven"  # Oven / range / cooker


class HVACType(str, Enum):
    """HVAC equipment types (direct refrigerant + indirect electricity)."""

    ROOM_AC = "room_ac"  # Room / window air conditioner
    CENTRAL_AC = "central_ac"  # Central air conditioning system
    HEAT_PUMP = "heat_pump"  # Heat pump (heating + cooling)
    GAS_FURNACE = "gas_furnace"  # Gas-fired furnace / boiler


class ITEquipmentType(str, Enum):
    """IT and office equipment types for indirect electricity calculations."""

    LAPTOP = "laptop"  # Laptop / notebook computer
    DESKTOP = "desktop"  # Desktop computer / workstation
    SERVER = "server"  # Data center server
    MONITOR = "monitor"  # Display / monitor


class IndustrialType(str, Enum):
    """Industrial equipment types (direct fuel + indirect electricity)."""

    DIESEL_GENERATOR = "diesel_generator"  # Diesel-powered generator set
    GAS_BOILER = "gas_boiler"  # Natural gas boiler
    COMPRESSOR = "compressor"  # Industrial compressor (electric)


class FuelType(str, Enum):
    """Fuel types for direct combustion and fuels-sold calculations."""

    GASOLINE = "gasoline"  # Motor gasoline / petrol
    DIESEL = "diesel"  # Diesel fuel
    NATURAL_GAS = "natural_gas"  # Natural gas (per m3)
    LPG = "lpg"  # Liquefied petroleum gas
    KEROSENE = "kerosene"  # Kerosene / paraffin
    HFO = "hfo"  # Heavy fuel oil / residual fuel
    JET_FUEL = "jet_fuel"  # Aviation turbine fuel / Jet A-1
    ETHANOL = "ethanol"  # Bioethanol (biogenic)
    BIODIESEL = "biodiesel"  # Biodiesel / FAME
    COAL = "coal"  # Coal (per kg)
    WOOD_PELLETS = "wood_pellets"  # Wood pellets / biomass (per kg)
    PROPANE = "propane"  # Propane
    HYDROGEN = "hydrogen"  # Hydrogen (per kg, zero direct CO2)
    CNG = "cng"  # Compressed natural gas (per m3)
    LNG = "lng"  # Liquefied natural gas (per kg)


class RefrigerantType(str, Enum):
    """Refrigerant types with GWP values for leakage calculations."""

    R134A = "r134a"  # HFC-134a (common auto AC)
    R410A = "r410a"  # HFC blend (residential HVAC)
    R32 = "r32"  # HFC-32 (lower GWP residential)
    R290 = "r290"  # Propane (natural refrigerant)
    R404A = "r404a"  # HFC blend (commercial refrigeration)
    R407C = "r407c"  # HFC blend (HVAC replacement)
    R507A = "r507a"  # HFC blend (low-temp commercial)
    R1234YF = "r1234yf"  # HFO (next-gen auto AC)
    R1234ZE = "r1234ze"  # HFO (chillers, heat pumps)
    R744 = "r744"  # CO2 (transcritical systems)


class GWPStandard(str, Enum):
    """IPCC Global Warming Potential assessment report version."""

    AR5 = "ar5"  # Fifth Assessment Report (100-year GWP)
    AR6 = "ar6"  # Sixth Assessment Report (100-year GWP)


class GridRegion(str, Enum):
    """Grid emission factor regions for indirect electricity calculations."""

    US = "US"  # United States average
    GB = "GB"  # Great Britain
    DE = "DE"  # Germany
    FR = "FR"  # France (nuclear-dominated, low EF)
    CN = "CN"  # China (coal-dominated, high EF)
    IN = "IN"  # India
    JP = "JP"  # Japan
    KR = "KR"  # South Korea
    BR = "BR"  # Brazil (hydro-dominated, low EF)
    CA = "CA"  # Canada
    AU = "AU"  # Australia (coal, high EF)
    MX = "MX"  # Mexico
    IT = "IT"  # Italy
    ES = "ES"  # Spain
    PL = "PL"  # Poland (coal-dominated)
    GLOBAL = "GLOBAL"  # Global weighted average


class LifetimeAdjustment(str, Enum):
    """Product lifetime adjustment factors for usage intensity."""

    STANDARD = "standard"  # Default assumption (multiplier 1.00)
    HEAVY = "heavy"  # Reduced lifetime, commercial/fleet use (0.80)
    LIGHT = "light"  # Extended lifetime, light residential use (1.20)
    INDUSTRIAL = "industrial"  # Continuous 24/7 industrial use (0.60)
    SEASONAL = "seasonal"  # Only used part of year, e.g. AC in temperate (0.50)


class CalculationMethod(str, Enum):
    """Calculation methods for use-phase emissions per GHG Protocol Cat 11."""

    DIRECT_FUEL = "direct_fuel"  # Direct fuel combustion in product
    DIRECT_REFRIGERANT = "direct_refrigerant"  # Direct refrigerant leakage
    DIRECT_CHEMICAL = "direct_chemical"  # Direct chemical GHG release
    INDIRECT_ELECTRICITY = "indirect_electricity"  # Indirect electricity consumption
    INDIRECT_HEATING = "indirect_heating"  # Indirect heating fuel consumption
    INDIRECT_STEAM = "indirect_steam"  # Indirect steam/cooling consumption
    FUELS_SOLD = "fuels_sold"  # Fuels sold for combustion by end users
    FEEDSTOCKS_SOLD = "feedstocks_sold"  # Feedstocks sold for oxidation


class DataQualityTier(str, Enum):
    """Data quality tiers affecting uncertainty ranges."""

    TIER_1 = "tier_1"  # Product-specific / primary data (best)
    TIER_2 = "tier_2"  # Category-average / secondary data
    TIER_3 = "tier_3"  # Global average / estimated data (worst)


class DQIDimension(str, Enum):
    """Data Quality Indicator dimensions per GHG Protocol guidance."""

    RELIABILITY = "reliability"  # Measurement / verification approach
    COMPLETENESS = "completeness"  # Fraction of data coverage
    TEMPORAL = "temporal"  # Temporal correlation to reporting year
    GEOGRAPHICAL = "geographical"  # Geographical correlation to activity
    TECHNOLOGICAL = "technological"  # Technological correlation to product


class ComplianceFramework(str, Enum):
    """Regulatory/reporting frameworks for compliance validation."""

    GHG_PROTOCOL = "ghg_protocol"  # GHG Protocol Scope 3 Standard Ch 6
    ISO_14064 = "iso_14064"  # ISO 14064-1:2018 Clause 5.2.4
    CSRD_ESRS = "csrd_esrs"  # CSRD ESRS E1 Climate Change
    CDP = "cdp"  # CDP Climate Change Questionnaire
    SBTI = "sbti"  # Science Based Targets initiative
    SB_253 = "sb_253"  # California SB 253 Climate Disclosure
    GRI = "gri"  # GRI 305 Emissions Standard


class ComplianceStatus(str, Enum):
    """Compliance check result status."""

    PASS = "pass"  # Fully compliant
    FAIL = "fail"  # Non-compliant
    WARNING = "warning"  # Partially compliant / needs attention
    NOT_APPLICABLE = "not_applicable"  # Framework not applicable


class PipelineStage(str, Enum):
    """Processing pipeline stages for the 10-stage orchestration."""

    VALIDATE = "validate"  # Input validation and schema checks
    CLASSIFY = "classify"  # Product classification (category, emission type)
    NORMALIZE = "normalize"  # Unit normalization (energy, volume, mass)
    RESOLVE_EFS = "resolve_efs"  # Emission factor resolution
    CALCULATE = "calculate"  # Core emissions calculation
    LIFETIME = "lifetime"  # Lifetime modeling and degradation
    AGGREGATE = "aggregate"  # Aggregation by category, period, type
    COMPLIANCE = "compliance"  # Compliance checks against frameworks
    PROVENANCE = "provenance"  # Provenance chain computation
    SEAL = "seal"  # Final chain sealing and output


class ProvenanceStage(str, Enum):
    """Provenance chain stages for SHA-256 hash tracking."""

    VALIDATE = "validate"  # Input validation hash
    CLASSIFY = "classify"  # Classification hash
    NORMALIZE = "normalize"  # Normalization hash
    RESOLVE_EFS = "resolve_efs"  # EF resolution hash
    CALCULATE = "calculate"  # Calculation hash
    LIFETIME = "lifetime"  # Lifetime modeling hash
    AGGREGATE = "aggregate"  # Aggregation hash
    COMPLIANCE = "compliance"  # Compliance hash
    PROVENANCE = "provenance"  # Provenance chain hash
    SEAL = "seal"  # Final seal hash


class UncertaintyMethod(str, Enum):
    """Uncertainty quantification methods."""

    MONTE_CARLO = "monte_carlo"  # Monte Carlo simulation
    ANALYTICAL = "analytical"  # Analytical error propagation
    IPCC_TIER_2 = "ipcc_tier_2"  # IPCC Tier 2 default ranges


class BatchStatus(str, Enum):
    """Batch calculation processing status."""

    PENDING = "pending"  # Awaiting processing
    PROCESSING = "processing"  # Currently processing
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Processing failed


class AuditAction(str, Enum):
    """Audit trail action types."""

    CREATE = "create"  # New calculation created
    UPDATE = "update"  # Calculation updated
    DELETE = "delete"  # Calculation deleted
    RECALCULATE = "recalculate"  # Recalculation triggered
    COMPLIANCE_CHECK = "compliance_check"  # Compliance check performed
    EXPORT = "export"  # Results exported


# ==============================================================================
# CONSTANT TABLES (16)
# ==============================================================================

# 1. Product Energy Profiles (24 products)
# Source: PRD Section 5.1 - Default lifetime and annual consumption
PRODUCT_ENERGY_PROFILES: Dict[str, Dict[str, Any]] = {
    # VEHICLES
    "PASSENGER_CAR_GASOLINE": {
        "category": ProductUseCategory.VEHICLES,
        "lifetime_years": Decimal("15"),
        "annual_consumption": Decimal("1200"),
        "unit": "liters/year",
        "description": "Gasoline passenger car",
    },
    "PASSENGER_CAR_DIESEL": {
        "category": ProductUseCategory.VEHICLES,
        "lifetime_years": Decimal("15"),
        "annual_consumption": Decimal("1000"),
        "unit": "liters/year",
        "description": "Diesel passenger car",
    },
    "PASSENGER_CAR_EV": {
        "category": ProductUseCategory.VEHICLES,
        "lifetime_years": Decimal("15"),
        "annual_consumption": Decimal("3500"),
        "unit": "kWh/year",
        "description": "Electric vehicle (BEV)",
    },
    "LIGHT_TRUCK": {
        "category": ProductUseCategory.VEHICLES,
        "lifetime_years": Decimal("15"),
        "annual_consumption": Decimal("1800"),
        "unit": "liters/year",
        "description": "Light-duty truck / SUV",
    },
    "HEAVY_TRUCK": {
        "category": ProductUseCategory.VEHICLES,
        "lifetime_years": Decimal("10"),
        "annual_consumption": Decimal("30000"),
        "unit": "liters/year",
        "description": "Heavy-duty truck / commercial vehicle",
    },
    "MOTORCYCLE": {
        "category": ProductUseCategory.VEHICLES,
        "lifetime_years": Decimal("12"),
        "annual_consumption": Decimal("500"),
        "unit": "liters/year",
        "description": "Motorcycle / scooter",
    },
    # APPLIANCES
    "REFRIGERATOR": {
        "category": ProductUseCategory.APPLIANCES,
        "lifetime_years": Decimal("15"),
        "annual_consumption": Decimal("400"),
        "unit": "kWh/year",
        "description": "Refrigerator / fridge-freezer",
    },
    "WASHING_MACHINE": {
        "category": ProductUseCategory.APPLIANCES,
        "lifetime_years": Decimal("12"),
        "annual_consumption": Decimal("200"),
        "unit": "kWh/year",
        "description": "Washing machine / washer",
    },
    "DISHWASHER": {
        "category": ProductUseCategory.APPLIANCES,
        "lifetime_years": Decimal("12"),
        "annual_consumption": Decimal("290"),
        "unit": "kWh/year",
        "description": "Dishwasher",
    },
    "DRYER": {
        "category": ProductUseCategory.APPLIANCES,
        "lifetime_years": Decimal("12"),
        "annual_consumption": Decimal("550"),
        "unit": "kWh/year",
        "description": "Tumble dryer / clothes dryer",
    },
    "OVEN": {
        "category": ProductUseCategory.APPLIANCES,
        "lifetime_years": Decimal("15"),
        "annual_consumption": Decimal("320"),
        "unit": "kWh/year",
        "description": "Oven / range / cooker",
    },
    # HVAC
    "ROOM_AC": {
        "category": ProductUseCategory.HVAC,
        "lifetime_years": Decimal("12"),
        "annual_consumption": Decimal("1200"),
        "unit": "kWh/year",
        "description": "Room / window air conditioner",
    },
    "CENTRAL_AC": {
        "category": ProductUseCategory.HVAC,
        "lifetime_years": Decimal("15"),
        "annual_consumption": Decimal("3500"),
        "unit": "kWh/year",
        "description": "Central air conditioning system",
    },
    "HEAT_PUMP": {
        "category": ProductUseCategory.HVAC,
        "lifetime_years": Decimal("15"),
        "annual_consumption": Decimal("4000"),
        "unit": "kWh/year",
        "description": "Heat pump (heating + cooling)",
    },
    "GAS_FURNACE": {
        "category": ProductUseCategory.HVAC,
        "lifetime_years": Decimal("20"),
        "annual_consumption": Decimal("1500"),
        "unit": "m3/year",
        "description": "Gas-fired furnace / boiler",
    },
    # LIGHTING
    "LED_BULB": {
        "category": ProductUseCategory.LIGHTING,
        "lifetime_years": Decimal("15"),
        "annual_consumption": Decimal("10"),
        "unit": "kWh/year",
        "description": "LED light bulb",
    },
    "CFL_BULB": {
        "category": ProductUseCategory.LIGHTING,
        "lifetime_years": Decimal("8"),
        "annual_consumption": Decimal("14"),
        "unit": "kWh/year",
        "description": "Compact fluorescent light bulb",
    },
    # IT EQUIPMENT
    "LAPTOP": {
        "category": ProductUseCategory.IT_EQUIPMENT,
        "lifetime_years": Decimal("5"),
        "annual_consumption": Decimal("50"),
        "unit": "kWh/year",
        "description": "Laptop / notebook computer",
    },
    "DESKTOP": {
        "category": ProductUseCategory.IT_EQUIPMENT,
        "lifetime_years": Decimal("6"),
        "annual_consumption": Decimal("200"),
        "unit": "kWh/year",
        "description": "Desktop computer / workstation",
    },
    "SERVER": {
        "category": ProductUseCategory.IT_EQUIPMENT,
        "lifetime_years": Decimal("5"),
        "annual_consumption": Decimal("4500"),
        "unit": "kWh/year",
        "description": "Data center server",
    },
    "MONITOR": {
        "category": ProductUseCategory.IT_EQUIPMENT,
        "lifetime_years": Decimal("7"),
        "annual_consumption": Decimal("80"),
        "unit": "kWh/year",
        "description": "Display / monitor",
    },
    # INDUSTRIAL EQUIPMENT
    "DIESEL_GENERATOR": {
        "category": ProductUseCategory.INDUSTRIAL_EQUIPMENT,
        "lifetime_years": Decimal("15"),
        "annual_consumption": Decimal("20000"),
        "unit": "liters/year",
        "description": "Diesel-powered generator set",
    },
    "GAS_BOILER": {
        "category": ProductUseCategory.INDUSTRIAL_EQUIPMENT,
        "lifetime_years": Decimal("20"),
        "annual_consumption": Decimal("25000"),
        "unit": "m3/year",
        "description": "Natural gas boiler",
    },
    "COMPRESSOR": {
        "category": ProductUseCategory.INDUSTRIAL_EQUIPMENT,
        "lifetime_years": Decimal("15"),
        "annual_consumption": Decimal("15000"),
        "unit": "kWh/year",
        "description": "Industrial compressor (electric)",
    },
}

# 2. Fuel Combustion Emission Factors (15 fuels)
# Source: PRD Section 5.2 - kgCO2e/unit and NCV in MJ/unit
FUEL_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    FuelType.GASOLINE.value: {
        "ef": Decimal("2.315"),
        "ncv": Decimal("34.2"),
    },
    FuelType.DIESEL.value: {
        "ef": Decimal("2.706"),
        "ncv": Decimal("38.6"),
    },
    FuelType.NATURAL_GAS.value: {
        "ef": Decimal("2.024"),
        "ncv": Decimal("38.3"),
    },
    FuelType.LPG.value: {
        "ef": Decimal("1.557"),
        "ncv": Decimal("26.1"),
    },
    FuelType.KEROSENE.value: {
        "ef": Decimal("2.541"),
        "ncv": Decimal("37.0"),
    },
    FuelType.HFO.value: {
        "ef": Decimal("3.114"),
        "ncv": Decimal("40.4"),
    },
    FuelType.JET_FUEL.value: {
        "ef": Decimal("2.548"),
        "ncv": Decimal("37.4"),
    },
    FuelType.ETHANOL.value: {
        "ef": Decimal("0.020"),
        "ncv": Decimal("26.7"),
    },
    FuelType.BIODIESEL.value: {
        "ef": Decimal("0.015"),
        "ncv": Decimal("37.0"),
    },
    FuelType.COAL.value: {
        "ef": Decimal("2.883"),
        "ncv": Decimal("25.8"),
    },
    FuelType.WOOD_PELLETS.value: {
        "ef": Decimal("0.015"),
        "ncv": Decimal("17.0"),
    },
    FuelType.PROPANE.value: {
        "ef": Decimal("1.530"),
        "ncv": Decimal("25.3"),
    },
    FuelType.HYDROGEN.value: {
        "ef": Decimal("0.000"),
        "ncv": Decimal("120.0"),
    },
    FuelType.CNG.value: {
        "ef": Decimal("2.024"),
        "ncv": Decimal("38.3"),
    },
    FuelType.LNG.value: {
        "ef": Decimal("2.750"),
        "ncv": Decimal("49.5"),
    },
}

# 3. Refrigerant GWPs (10 refrigerants)
# Source: PRD Section 5.3 - AR5/AR6 GWPs, typical charge, annual leak rate
REFRIGERANT_GWPS: Dict[str, Dict[str, Decimal]] = {
    RefrigerantType.R134A.value: {
        "gwp_ar5": Decimal("1430"),
        "gwp_ar6": Decimal("1530"),
        "typical_charge_kg": Decimal("1.5"),
        "annual_leak_rate": Decimal("0.05"),
    },
    RefrigerantType.R410A.value: {
        "gwp_ar5": Decimal("2088"),
        "gwp_ar6": Decimal("2088"),
        "typical_charge_kg": Decimal("3.0"),
        "annual_leak_rate": Decimal("0.04"),
    },
    RefrigerantType.R32.value: {
        "gwp_ar5": Decimal("675"),
        "gwp_ar6": Decimal("771"),
        "typical_charge_kg": Decimal("1.5"),
        "annual_leak_rate": Decimal("0.03"),
    },
    RefrigerantType.R290.value: {
        "gwp_ar5": Decimal("3"),
        "gwp_ar6": Decimal("0.02"),
        "typical_charge_kg": Decimal("0.3"),
        "annual_leak_rate": Decimal("0.02"),
    },
    RefrigerantType.R404A.value: {
        "gwp_ar5": Decimal("3922"),
        "gwp_ar6": Decimal("3922"),
        "typical_charge_kg": Decimal("5.0"),
        "annual_leak_rate": Decimal("0.10"),
    },
    RefrigerantType.R407C.value: {
        "gwp_ar5": Decimal("1774"),
        "gwp_ar6": Decimal("1774"),
        "typical_charge_kg": Decimal("3.0"),
        "annual_leak_rate": Decimal("0.05"),
    },
    RefrigerantType.R507A.value: {
        "gwp_ar5": Decimal("3985"),
        "gwp_ar6": Decimal("3985"),
        "typical_charge_kg": Decimal("5.0"),
        "annual_leak_rate": Decimal("0.10"),
    },
    RefrigerantType.R1234YF.value: {
        "gwp_ar5": Decimal("4"),
        "gwp_ar6": Decimal("0.501"),
        "typical_charge_kg": Decimal("0.6"),
        "annual_leak_rate": Decimal("0.03"),
    },
    RefrigerantType.R1234ZE.value: {
        "gwp_ar5": Decimal("7"),
        "gwp_ar6": Decimal("1.37"),
        "typical_charge_kg": Decimal("1.0"),
        "annual_leak_rate": Decimal("0.03"),
    },
    RefrigerantType.R744.value: {
        "gwp_ar5": Decimal("1"),
        "gwp_ar6": Decimal("1"),
        "typical_charge_kg": Decimal("2.5"),
        "annual_leak_rate": Decimal("0.05"),
    },
}

# 4. Grid Emission Factors (16 regions, kgCO2e/kWh)
# Source: PRD Section 5.4 - same as MRV-023
GRID_EMISSION_FACTORS: Dict[str, Decimal] = {
    GridRegion.US.value: Decimal("0.417"),
    GridRegion.GB.value: Decimal("0.233"),
    GridRegion.DE.value: Decimal("0.348"),
    GridRegion.FR.value: Decimal("0.052"),
    GridRegion.CN.value: Decimal("0.555"),
    GridRegion.IN.value: Decimal("0.708"),
    GridRegion.JP.value: Decimal("0.462"),
    GridRegion.KR.value: Decimal("0.424"),
    GridRegion.BR.value: Decimal("0.075"),
    GridRegion.CA.value: Decimal("0.120"),
    GridRegion.AU.value: Decimal("0.656"),
    GridRegion.MX.value: Decimal("0.431"),
    GridRegion.IT.value: Decimal("0.256"),
    GridRegion.ES.value: Decimal("0.175"),
    GridRegion.PL.value: Decimal("0.635"),
    GridRegion.GLOBAL.value: Decimal("0.475"),
}

# 5. Lifetime Adjustment Factors
# Source: PRD Section 5.5
LIFETIME_ADJUSTMENT_FACTORS: Dict[str, Decimal] = {
    LifetimeAdjustment.STANDARD.value: Decimal("1.00"),
    LifetimeAdjustment.HEAVY.value: Decimal("0.80"),
    LifetimeAdjustment.LIGHT.value: Decimal("1.20"),
    LifetimeAdjustment.INDUSTRIAL.value: Decimal("0.60"),
    LifetimeAdjustment.SEASONAL.value: Decimal("0.50"),
}

# 6. Energy Degradation Rates (annual fractional efficiency loss by category)
# Source: PRD Section 5.6
ENERGY_DEGRADATION_RATES: Dict[str, Decimal] = {
    ProductUseCategory.VEHICLES.value: Decimal("0.015"),
    ProductUseCategory.APPLIANCES.value: Decimal("0.005"),
    ProductUseCategory.HVAC.value: Decimal("0.010"),
    ProductUseCategory.LIGHTING.value: Decimal("0.020"),
    ProductUseCategory.IT_EQUIPMENT.value: Decimal("0.000"),
    ProductUseCategory.INDUSTRIAL_EQUIPMENT.value: Decimal("0.010"),
}

# 7. Steam/Cooling Emission Factors (kgCO2e/MJ by source)
STEAM_COOLING_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "district_heating": {
        "ef_kgco2e_per_mj": Decimal("0.060"),
        "description_key": "District heating (average)",
    },
    "district_cooling": {
        "ef_kgco2e_per_mj": Decimal("0.045"),
        "description_key": "District cooling (average)",
    },
    "industrial_steam": {
        "ef_kgco2e_per_mj": Decimal("0.072"),
        "description_key": "Industrial steam (natural gas boiler)",
    },
    "chp_waste_heat": {
        "ef_kgco2e_per_mj": Decimal("0.025"),
        "description_key": "CHP waste heat recovery",
    },
}

# 8. Chemical Products (example chemicals with GHG content and release fractions)
CHEMICAL_PRODUCTS: Dict[str, Dict[str, Decimal]] = {
    "aerosol_propellant": {
        "ghg_content_kg": Decimal("0.250"),
        "release_fraction": Decimal("1.00"),
        "gwp": Decimal("1430"),
    },
    "foam_blowing_agent": {
        "ghg_content_kg": Decimal("0.500"),
        "release_fraction": Decimal("0.10"),
        "gwp": Decimal("1430"),
    },
    "sf6_switchgear": {
        "ghg_content_kg": Decimal("5.000"),
        "release_fraction": Decimal("0.01"),
        "gwp": Decimal("23500"),
    },
    "fire_extinguisher": {
        "ghg_content_kg": Decimal("2.500"),
        "release_fraction": Decimal("0.02"),
        "gwp": Decimal("3922"),
    },
    "fertilizer_n2o": {
        "ghg_content_kg": Decimal("0.010"),
        "release_fraction": Decimal("0.01"),
        "gwp": Decimal("273"),
    },
}

# 9. DQI Scoring Matrix (5 dimensions x 3 levels)
DQI_SCORING: Dict[str, Dict[str, Decimal]] = {
    DQIDimension.RELIABILITY.value: {
        DataQualityTier.TIER_1.value: Decimal("5"),
        DataQualityTier.TIER_2.value: Decimal("3"),
        DataQualityTier.TIER_3.value: Decimal("1"),
    },
    DQIDimension.COMPLETENESS.value: {
        DataQualityTier.TIER_1.value: Decimal("5"),
        DataQualityTier.TIER_2.value: Decimal("3"),
        DataQualityTier.TIER_3.value: Decimal("1"),
    },
    DQIDimension.TEMPORAL.value: {
        DataQualityTier.TIER_1.value: Decimal("5"),
        DataQualityTier.TIER_2.value: Decimal("3"),
        DataQualityTier.TIER_3.value: Decimal("1"),
    },
    DQIDimension.GEOGRAPHICAL.value: {
        DataQualityTier.TIER_1.value: Decimal("5"),
        DataQualityTier.TIER_2.value: Decimal("3"),
        DataQualityTier.TIER_3.value: Decimal("1"),
    },
    DQIDimension.TECHNOLOGICAL.value: {
        DataQualityTier.TIER_1.value: Decimal("5"),
        DataQualityTier.TIER_2.value: Decimal("3"),
        DataQualityTier.TIER_3.value: Decimal("1"),
    },
}

# 10. Uncertainty Ranges by calculation method (min/default/max as fractions)
# Source: PRD Section 5.8
UNCERTAINTY_RANGES: Dict[str, Dict[str, Decimal]] = {
    CalculationMethod.DIRECT_FUEL.value: {
        "min": Decimal("0.10"),
        "default": Decimal("0.15"),
        "max": Decimal("0.25"),
    },
    CalculationMethod.DIRECT_REFRIGERANT.value: {
        "min": Decimal("0.10"),
        "default": Decimal("0.20"),
        "max": Decimal("0.35"),
    },
    CalculationMethod.DIRECT_CHEMICAL.value: {
        "min": Decimal("0.15"),
        "default": Decimal("0.25"),
        "max": Decimal("0.40"),
    },
    CalculationMethod.INDIRECT_ELECTRICITY.value: {
        "min": Decimal("0.10"),
        "default": Decimal("0.20"),
        "max": Decimal("0.30"),
    },
    CalculationMethod.INDIRECT_HEATING.value: {
        "min": Decimal("0.20"),
        "default": Decimal("0.30"),
        "max": Decimal("0.50"),
    },
    CalculationMethod.FUELS_SOLD.value: {
        "min": Decimal("0.05"),
        "default": Decimal("0.10"),
        "max": Decimal("0.15"),
    },
}

# 11. Double-Counting Prevention Rules (8 rules)
# Source: PRD Section 6
DC_RULES: Dict[str, Dict[str, str]] = {
    "DC-USP-001": {
        "rule": "vs Scope 1",
        "description": "Exclude direct emissions from own use of products",
    },
    "DC-USP-002": {
        "rule": "vs Scope 2",
        "description": "Exclude electricity from own product use",
    },
    "DC-USP-003": {
        "rule": "vs Cat 1",
        "description": "No overlap with upstream production",
    },
    "DC-USP-004": {
        "rule": "vs Cat 3",
        "description": "No overlap with fuel & energy activities",
    },
    "DC-USP-005": {
        "rule": "vs Cat 10",
        "description": "No overlap with processing (pre-use)",
    },
    "DC-USP-006": {
        "rule": "vs Cat 12",
        "description": "No overlap with end-of-life (post-use)",
    },
    "DC-USP-007": {
        "rule": "vs Cat 13",
        "description": "No overlap with downstream leased assets",
    },
    "DC-USP-008": {
        "rule": "Fuel double-count",
        "description": (
            "Don't count fuel sold if already counted in vehicle use-phase"
        ),
    },
}

# 12. Compliance Framework Rules (7 frameworks)
# Source: PRD Section 7
COMPLIANCE_FRAMEWORK_RULES: Dict[str, Dict[str, Any]] = {
    ComplianceFramework.GHG_PROTOCOL.value: {
        "name": "GHG Protocol Scope 3 Standard",
        "reference": "Chapter 6, Category 11",
        "required_disclosures": [
            "total_co2e",
            "direct_vs_indirect_split",
            "lifetime_assumptions",
            "method_used",
            "ef_sources",
            "exclusions",
            "dqi_score",
        ],
    },
    ComplianceFramework.ISO_14064.value: {
        "name": "ISO 14064-1:2018",
        "reference": "Clause 5.2.4",
        "required_disclosures": [
            "total_co2e",
            "methodology",
            "uncertainty_analysis",
            "base_year",
            "verification",
        ],
    },
    ComplianceFramework.CSRD_ESRS.value: {
        "name": "CSRD ESRS E1 Climate Change",
        "reference": "E1-6 Scope 3",
        "required_disclosures": [
            "total_co2e",
            "category_breakdown",
            "methodology",
            "targets",
            "actions",
            "dnsh_assessment",
        ],
    },
    ComplianceFramework.CDP.value: {
        "name": "CDP Climate Change Questionnaire",
        "reference": "C6.5 Category 11",
        "required_disclosures": [
            "total_co2e",
            "methodology",
            "data_quality",
            "product_breakdown",
            "verification_status",
        ],
    },
    ComplianceFramework.SBTI.value: {
        "name": "Science Based Targets initiative",
        "reference": "Corporate Net-Zero Standard",
        "required_disclosures": [
            "total_co2e",
            "target_coverage_67pct",
            "base_year_recalculation",
            "progress_tracking",
        ],
    },
    ComplianceFramework.SB_253.value: {
        "name": "California SB 253 Climate Disclosure Act",
        "reference": "Climate Corporate Data Accountability Act",
        "required_disclosures": [
            "total_co2e",
            "methodology",
            "assurance_opinion",
        ],
    },
    ComplianceFramework.GRI.value: {
        "name": "GRI 305 Emissions Standard",
        "reference": "305-3 Other indirect GHG",
        "required_disclosures": [
            "total_co2e",
            "gases_included",
            "base_year",
            "standards_used",
        ],
    },
}

# 13. Feedstock Properties (5 feedstocks)
FEEDSTOCK_PROPERTIES: Dict[str, Dict[str, Decimal]] = {
    "natural_gas_feedstock": {
        "carbon_content": Decimal("0.729"),
        "oxidation_factor": Decimal("0.995"),
    },
    "naphtha": {
        "carbon_content": Decimal("0.836"),
        "oxidation_factor": Decimal("0.990"),
    },
    "coal_feedstock": {
        "carbon_content": Decimal("0.716"),
        "oxidation_factor": Decimal("0.980"),
    },
    "petroleum_coke": {
        "carbon_content": Decimal("0.870"),
        "oxidation_factor": Decimal("0.990"),
    },
    "ethylene": {
        "carbon_content": Decimal("0.856"),
        "oxidation_factor": Decimal("0.995"),
    },
}

# 14. Product Type Map (maps ProductUseCategory to applicable sub-type enums)
PRODUCT_TYPE_MAP: Dict[str, List[str]] = {
    ProductUseCategory.VEHICLES.value: [
        VehicleType.PASSENGER_CAR_GASOLINE.value,
        VehicleType.PASSENGER_CAR_DIESEL.value,
        VehicleType.PASSENGER_CAR_EV.value,
        VehicleType.LIGHT_TRUCK.value,
        VehicleType.HEAVY_TRUCK.value,
        VehicleType.MOTORCYCLE.value,
    ],
    ProductUseCategory.APPLIANCES.value: [
        ApplianceType.REFRIGERATOR.value,
        ApplianceType.WASHING_MACHINE.value,
        ApplianceType.DISHWASHER.value,
        ApplianceType.DRYER.value,
        ApplianceType.OVEN.value,
    ],
    ProductUseCategory.HVAC.value: [
        HVACType.ROOM_AC.value,
        HVACType.CENTRAL_AC.value,
        HVACType.HEAT_PUMP.value,
        HVACType.GAS_FURNACE.value,
    ],
    ProductUseCategory.IT_EQUIPMENT.value: [
        ITEquipmentType.LAPTOP.value,
        ITEquipmentType.DESKTOP.value,
        ITEquipmentType.SERVER.value,
        ITEquipmentType.MONITOR.value,
    ],
    ProductUseCategory.INDUSTRIAL_EQUIPMENT.value: [
        IndustrialType.DIESEL_GENERATOR.value,
        IndustrialType.GAS_BOILER.value,
        IndustrialType.COMPRESSOR.value,
    ],
    ProductUseCategory.LIGHTING.value: [
        "led_bulb",
        "cfl_bulb",
    ],
    ProductUseCategory.FUELS_FEEDSTOCKS.value: [
        ft.value for ft in FuelType
    ],
    ProductUseCategory.BUILDING_PRODUCTS.value: [
        "windows",
        "insulation",
        "hvac_ducts",
    ],
    ProductUseCategory.CONSUMER_PRODUCTS.value: [
        "aerosol_propellant",
        "foam_blowing_agent",
        "sf6_switchgear",
        "fire_extinguisher",
        "fertilizer_n2o",
    ],
    ProductUseCategory.MEDICAL_DEVICES.value: [
        "imaging_equipment",
        "ventilator",
        "lab_equipment",
    ],
}

# 15. Emission Type Map (maps each product category to applicable emission types)
EMISSION_TYPE_MAP: Dict[str, List[str]] = {
    ProductUseCategory.VEHICLES.value: [
        UsePhaseEmissionType.DIRECT_FUEL_COMBUSTION.value,
    ],
    ProductUseCategory.APPLIANCES.value: [
        UsePhaseEmissionType.INDIRECT_ELECTRICITY.value,
    ],
    ProductUseCategory.HVAC.value: [
        UsePhaseEmissionType.DIRECT_REFRIGERANT_LEAKAGE.value,
        UsePhaseEmissionType.INDIRECT_ELECTRICITY.value,
    ],
    ProductUseCategory.LIGHTING.value: [
        UsePhaseEmissionType.INDIRECT_ELECTRICITY.value,
    ],
    ProductUseCategory.IT_EQUIPMENT.value: [
        UsePhaseEmissionType.INDIRECT_ELECTRICITY.value,
    ],
    ProductUseCategory.INDUSTRIAL_EQUIPMENT.value: [
        UsePhaseEmissionType.DIRECT_FUEL_COMBUSTION.value,
        UsePhaseEmissionType.INDIRECT_ELECTRICITY.value,
    ],
    ProductUseCategory.FUELS_FEEDSTOCKS.value: [
        UsePhaseEmissionType.DIRECT_FUEL_COMBUSTION.value,
    ],
    ProductUseCategory.BUILDING_PRODUCTS.value: [
        UsePhaseEmissionType.INDIRECT_ELECTRICITY.value,
    ],
    ProductUseCategory.CONSUMER_PRODUCTS.value: [
        UsePhaseEmissionType.DIRECT_CHEMICAL_RELEASE.value,
    ],
    ProductUseCategory.MEDICAL_DEVICES.value: [
        UsePhaseEmissionType.INDIRECT_ELECTRICITY.value,
    ],
}

# 16. Default Usage Hours (annual hours by product category)
DEFAULT_USAGE_HOURS: Dict[str, Decimal] = {
    ProductUseCategory.VEHICLES.value: Decimal("600"),
    ProductUseCategory.APPLIANCES.value: Decimal("2000"),
    ProductUseCategory.HVAC.value: Decimal("2500"),
    ProductUseCategory.LIGHTING.value: Decimal("1000"),
    ProductUseCategory.IT_EQUIPMENT.value: Decimal("2500"),
    ProductUseCategory.INDUSTRIAL_EQUIPMENT.value: Decimal("4000"),
    ProductUseCategory.FUELS_FEEDSTOCKS.value: Decimal("0"),
    ProductUseCategory.BUILDING_PRODUCTS.value: Decimal("8760"),
    ProductUseCategory.CONSUMER_PRODUCTS.value: Decimal("100"),
    ProductUseCategory.MEDICAL_DEVICES.value: Decimal("3000"),
}


# ==============================================================================
# PYDANTIC MODELS (14) - All frozen for audit trail integrity
# ==============================================================================


class ProductInput(BaseModel):
    """
    Input for a single product's use-phase emissions calculation.

    Represents one product type sold by the reporting company. Contains
    all parameters needed to calculate both direct and indirect use-phase
    emissions over the product's expected lifetime.

    Example:
        >>> product = ProductInput(
        ...     product_id="PROD-001",
        ...     category=ProductUseCategory.VEHICLES,
        ...     product_type="PASSENGER_CAR_GASOLINE",
        ...     units_sold=Decimal("10000"),
        ...     lifetime_years=Decimal("15"),
        ...     fuel_type=FuelType.GASOLINE,
        ...     use_region=GridRegion.US
        ... )
    """

    product_id: str = Field(
        ..., min_length=1, max_length=64,
        description="Unique product identifier"
    )
    category: ProductUseCategory = Field(
        ..., description="Product use category (determines emission type)"
    )
    product_type: str = Field(
        ..., min_length=1, max_length=64,
        description="Product sub-type within category"
    )
    units_sold: Decimal = Field(
        ..., gt=0,
        description="Number of units sold in the reporting period"
    )
    lifetime_years: Optional[Decimal] = Field(
        default=None, gt=0,
        description="Expected product lifetime in years (overrides default)"
    )
    annual_energy: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Annual energy consumption (kWh, liters, or m3 per year)"
    )
    fuel_type: Optional[FuelType] = Field(
        default=None,
        description="Fuel type for direct combustion products"
    )
    energy_type: Optional[UsePhaseEmissionType] = Field(
        default=None,
        description="Primary emission type override"
    )
    use_region: Optional[GridRegion] = Field(
        default=None,
        description="Region where product is used (for grid EF lookup)"
    )
    refrigerant_type: Optional[RefrigerantType] = Field(
        default=None,
        description="Refrigerant type for HVAC/refrigeration products"
    )
    charge_kg: Optional[Decimal] = Field(
        default=None, gt=0,
        description="Refrigerant charge per unit in kg"
    )
    leak_rate: Optional[Decimal] = Field(
        default=None, gt=0, le=1,
        description="Annual refrigerant leak rate (fraction, 0-1)"
    )
    chemical_content_kg: Optional[Decimal] = Field(
        default=None, ge=0,
        description="GHG-containing chemical content per unit in kg"
    )
    release_fraction: Optional[Decimal] = Field(
        default=None, ge=0, le=1,
        description="Fraction of chemical released during use (0-1)"
    )
    gwp_standard: GWPStandard = Field(
        default=GWPStandard.AR5,
        description="GWP assessment report version for refrigerants"
    )
    lifetime_adjustment: LifetimeAdjustment = Field(
        default=LifetimeAdjustment.STANDARD,
        description="Lifetime adjustment factor (standard/heavy/light/industrial/seasonal)"
    )

    model_config = ConfigDict(frozen=True)

    @validator("product_id")
    def validate_product_id(cls, v: str) -> str:
        """Validate product_id is non-empty after stripping whitespace."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("product_id must not be empty or whitespace-only")
        return stripped

    @validator("units_sold")
    def validate_units_sold(cls, v: Decimal) -> Decimal:
        """Validate units_sold is a positive number."""
        if v <= 0:
            raise ValueError(f"units_sold must be positive, got {v}")
        return v


class FuelSalesInput(BaseModel):
    """
    Input for fuels or feedstocks sold for end-user combustion/oxidation.

    Used when the reporting company sells fuels (gasoline, diesel, natural gas)
    or feedstocks that will release GHGs when combusted/oxidized by end users.

    Example:
        >>> fuel_sale = FuelSalesInput(
        ...     fuel_type=FuelType.GASOLINE,
        ...     volume_sold=Decimal("1000000"),
        ...     unit="liters",
        ...     region=GridRegion.US
        ... )
    """

    fuel_type: FuelType = Field(
        ..., description="Type of fuel or feedstock sold"
    )
    volume_sold: Decimal = Field(
        ..., gt=0,
        description="Volume or mass of fuel sold (in unit specified)"
    )
    unit: str = Field(
        default="liters",
        description="Unit of measurement (liters, m3, kg, tonnes)"
    )
    region: Optional[GridRegion] = Field(
        default=None,
        description="Sales region for geographic attribution"
    )

    model_config = ConfigDict(frozen=True)

    @validator("volume_sold")
    def validate_volume_sold(cls, v: Decimal) -> Decimal:
        """Validate volume_sold is a positive number."""
        if v <= 0:
            raise ValueError(f"volume_sold must be positive, got {v}")
        return v


class DirectEmissionsInput(BaseModel):
    """
    Input for direct use-phase emissions calculation.

    Groups one or more products whose direct emissions (fuel combustion,
    refrigerant leakage, chemical release) need to be calculated.

    Example:
        >>> direct_input = DirectEmissionsInput(
        ...     org_id="ORG-001",
        ...     reporting_year=2025,
        ...     products=[product1, product2]
        ... )
    """

    org_id: str = Field(
        ..., min_length=1, max_length=64,
        description="Organization identifier"
    )
    reporting_year: int = Field(
        ..., ge=2015, le=2035,
        description="Reporting year for the calculation"
    )
    products: List[ProductInput] = Field(
        ..., min_length=1,
        description="List of products with direct use-phase emissions"
    )

    model_config = ConfigDict(frozen=True)


class IndirectEmissionsInput(BaseModel):
    """
    Input for indirect use-phase emissions calculation.

    Groups one or more products whose indirect emissions (electricity
    consumption, heating fuel, steam/cooling) need to be calculated.

    Example:
        >>> indirect_input = IndirectEmissionsInput(
        ...     org_id="ORG-001",
        ...     reporting_year=2025,
        ...     products=[appliance1, it_product1]
        ... )
    """

    org_id: str = Field(
        ..., min_length=1, max_length=64,
        description="Organization identifier"
    )
    reporting_year: int = Field(
        ..., ge=2015, le=2035,
        description="Reporting year for the calculation"
    )
    products: List[ProductInput] = Field(
        ..., min_length=1,
        description="List of products with indirect use-phase emissions"
    )

    model_config = ConfigDict(frozen=True)


class FuelsAndFeedstocksInput(BaseModel):
    """
    Input for fuels and feedstocks sold calculation.

    Groups one or more fuel/feedstock sales records for emissions
    calculation based on end-user combustion/oxidation.

    Example:
        >>> fuels_input = FuelsAndFeedstocksInput(
        ...     org_id="ORG-001",
        ...     reporting_year=2025,
        ...     fuel_sales=[fuel_sale1, fuel_sale2]
        ... )
    """

    org_id: str = Field(
        ..., min_length=1, max_length=64,
        description="Organization identifier"
    )
    reporting_year: int = Field(
        ..., ge=2015, le=2035,
        description="Reporting year for the calculation"
    )
    fuel_sales: List[FuelSalesInput] = Field(
        ..., min_length=1,
        description="List of fuel/feedstock sales records"
    )

    model_config = ConfigDict(frozen=True)


class DirectEmissionDetail(BaseModel):
    """
    Detailed breakdown of a single direct emission calculation.

    Records the specific emission type, source parameter (fuel, refrigerant,
    or chemical), quantity used, emission factor applied, and resulting emissions.

    Example:
        >>> detail = DirectEmissionDetail(
        ...     product_id="PROD-001",
        ...     emission_type=UsePhaseEmissionType.DIRECT_FUEL_COMBUSTION,
        ...     fuel_type=FuelType.GASOLINE,
        ...     quantity=Decimal("18000000"),
        ...     ef_used=Decimal("2.315"),
        ...     emissions_kg=Decimal("41670000")
        ... )
    """

    product_id: str = Field(
        ..., description="Product identifier"
    )
    emission_type: UsePhaseEmissionType = Field(
        ..., description="Type of direct emission"
    )
    fuel_type: Optional[str] = Field(
        default=None,
        description="Fuel type (for DIRECT_FUEL_COMBUSTION)"
    )
    refrigerant_type: Optional[str] = Field(
        default=None,
        description="Refrigerant type (for DIRECT_REFRIGERANT_LEAKAGE)"
    )
    chemical_name: Optional[str] = Field(
        default=None,
        description="Chemical name (for DIRECT_CHEMICAL_RELEASE)"
    )
    quantity: Decimal = Field(
        ..., description="Quantity consumed/released over lifetime"
    )
    ef_used: Decimal = Field(
        ..., description="Emission factor applied (kgCO2e/unit or GWP)"
    )
    emissions_kg: Decimal = Field(
        ..., description="Calculated emissions in kgCO2e"
    )

    model_config = ConfigDict(frozen=True)


class IndirectEmissionDetail(BaseModel):
    """
    Detailed breakdown of a single indirect emission calculation.

    Records the specific emission type, energy consumption, grid or fuel
    emission factor applied, and resulting emissions.

    Example:
        >>> detail = IndirectEmissionDetail(
        ...     product_id="PROD-002",
        ...     emission_type=UsePhaseEmissionType.INDIRECT_ELECTRICITY,
        ...     energy_kwh=Decimal("90000000"),
        ...     grid_ef=Decimal("0.417"),
        ...     emissions_kg=Decimal("37530000")
        ... )
    """

    product_id: str = Field(
        ..., description="Product identifier"
    )
    emission_type: UsePhaseEmissionType = Field(
        ..., description="Type of indirect emission"
    )
    energy_kwh: Optional[Decimal] = Field(
        default=None,
        description="Total electricity consumed over lifetime (kWh)"
    )
    fuel_liters: Optional[Decimal] = Field(
        default=None,
        description="Total heating fuel consumed over lifetime (liters or m3)"
    )
    steam_mj: Optional[Decimal] = Field(
        default=None,
        description="Total steam/cooling consumed over lifetime (MJ)"
    )
    grid_ef: Optional[Decimal] = Field(
        default=None,
        description="Grid emission factor used (kgCO2e/kWh)"
    )
    fuel_ef: Optional[Decimal] = Field(
        default=None,
        description="Fuel emission factor used (kgCO2e/liter)"
    )
    steam_ef: Optional[Decimal] = Field(
        default=None,
        description="Steam/cooling EF used (kgCO2e/MJ)"
    )
    emissions_kg: Decimal = Field(
        ..., description="Calculated emissions in kgCO2e"
    )

    model_config = ConfigDict(frozen=True)


class DataQualityScore(BaseModel):
    """
    Data quality indicator score across 5 dimensions plus overall.

    Follows GHG Protocol DQI guidance with scores from 1 (lowest)
    to 5 (highest) per dimension.

    Example:
        >>> dqi = DataQualityScore(
        ...     reliability=Decimal("4"),
        ...     completeness=Decimal("5"),
        ...     temporal=Decimal("4"),
        ...     geographical=Decimal("3"),
        ...     technological=Decimal("4"),
        ...     overall=Decimal("4.0")
        ... )
    """

    reliability: Decimal = Field(
        ..., ge=1, le=5,
        description="Reliability dimension score (1-5)"
    )
    completeness: Decimal = Field(
        ..., ge=1, le=5,
        description="Completeness dimension score (1-5)"
    )
    temporal: Decimal = Field(
        ..., ge=1, le=5,
        description="Temporal correlation score (1-5)"
    )
    geographical: Decimal = Field(
        ..., ge=1, le=5,
        description="Geographical correlation score (1-5)"
    )
    technological: Decimal = Field(
        ..., ge=1, le=5,
        description="Technological correlation score (1-5)"
    )
    overall: Decimal = Field(
        ..., ge=1, le=5,
        description="Weighted overall DQI score (1-5)"
    )

    model_config = ConfigDict(frozen=True)


class ProductBreakdown(BaseModel):
    """
    Per-product emissions breakdown within a calculation result.

    Provides granular detail for each product type including units sold,
    lifetime, direct and indirect emissions, calculation method, and DQI.

    Example:
        >>> breakdown = ProductBreakdown(
        ...     product_id="PROD-001",
        ...     category=ProductUseCategory.VEHICLES,
        ...     units_sold=Decimal("10000"),
        ...     lifetime=Decimal("15"),
        ...     direct_emissions_kg=Decimal("41670000"),
        ...     indirect_emissions_kg=Decimal("0"),
        ...     total_emissions_kg=Decimal("41670000"),
        ...     method=CalculationMethod.DIRECT_FUEL,
        ...     dqi=dqi_score
        ... )
    """

    product_id: str = Field(
        ..., description="Product identifier"
    )
    category: ProductUseCategory = Field(
        ..., description="Product use category"
    )
    units_sold: Decimal = Field(
        ..., description="Units sold in reporting period"
    )
    lifetime: Decimal = Field(
        ..., description="Product lifetime used (years, after adjustment)"
    )
    direct_emissions_kg: Decimal = Field(
        ..., description="Total direct use-phase emissions (kgCO2e)"
    )
    indirect_emissions_kg: Decimal = Field(
        ..., description="Total indirect use-phase emissions (kgCO2e)"
    )
    total_emissions_kg: Decimal = Field(
        ..., description="Combined direct + indirect emissions (kgCO2e)"
    )
    method: CalculationMethod = Field(
        ..., description="Primary calculation method used"
    )
    dqi: Optional[DataQualityScore] = Field(
        default=None,
        description="Data quality indicator score"
    )

    model_config = ConfigDict(frozen=True)


class UncertaintyResult(BaseModel):
    """
    Uncertainty quantification result for an emissions estimate.

    Provides mean estimate with confidence interval bounds using
    the specified quantification method.

    Example:
        >>> uncertainty = UncertaintyResult(
        ...     method=UncertaintyMethod.MONTE_CARLO,
        ...     mean=Decimal("41670000"),
        ...     std_dev=Decimal("6250500"),
        ...     ci_lower=Decimal("29169500"),
        ...     ci_upper=Decimal("54170500")
        ... )
    """

    method: UncertaintyMethod = Field(
        ..., description="Uncertainty quantification method used"
    )
    mean: Decimal = Field(
        ..., description="Mean emissions estimate (kgCO2e)"
    )
    std_dev: Decimal = Field(
        ..., description="Standard deviation (kgCO2e)"
    )
    ci_lower: Decimal = Field(
        ..., description="95% confidence interval lower bound (kgCO2e)"
    )
    ci_upper: Decimal = Field(
        ..., description="95% confidence interval upper bound (kgCO2e)"
    )

    model_config = ConfigDict(frozen=True)


class ProvenanceRecord(BaseModel):
    """
    Single record in the SHA-256 provenance chain.

    Tracks data lineage through each pipeline stage for full
    audit trail and reproducibility.

    Example:
        >>> record = ProvenanceRecord(
        ...     stage=ProvenanceStage.CALCULATE,
        ...     input_hash="abc123...",
        ...     output_hash="def456...",
        ...     timestamp="2025-12-01T10:30:00Z",
        ...     metadata={"method": "direct_fuel"}
        ... )
    """

    stage: ProvenanceStage = Field(
        ..., description="Pipeline stage name"
    )
    input_hash: str = Field(
        ..., min_length=64, max_length=64,
        description="SHA-256 hash of stage input"
    )
    output_hash: str = Field(
        ..., min_length=64, max_length=64,
        description="SHA-256 hash of stage output"
    )
    timestamp: str = Field(
        ..., description="ISO 8601 timestamp"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Stage-specific metadata"
    )

    model_config = ConfigDict(frozen=True)


class CalculationResult(BaseModel):
    """
    Complete result from a use-of-sold-products emissions calculation.

    Contains total emissions split by direct/indirect/fuel-sales with
    per-product breakdowns, data quality, uncertainty, and provenance.

    Example:
        >>> result = CalculationResult(
        ...     calc_id="CALC-2025-001",
        ...     org_id="ORG-001",
        ...     reporting_year=2025,
        ...     direct_emissions_kg=Decimal("41670000"),
        ...     indirect_emissions_kg=Decimal("37530000"),
        ...     fuel_sales_emissions_kg=Decimal("2315000"),
        ...     total_emissions_kg=Decimal("81515000"),
        ...     total_tco2e=Decimal("81515"),
        ...     product_breakdowns=[breakdown1, breakdown2],
        ...     provenance_hash="abc123..."
        ... )
    """

    calc_id: str = Field(
        ..., description="Unique calculation identifier"
    )
    org_id: str = Field(
        ..., description="Organization identifier"
    )
    reporting_year: int = Field(
        ..., description="Reporting year"
    )
    direct_emissions_kg: Decimal = Field(
        ..., description="Total direct use-phase emissions (kgCO2e)"
    )
    indirect_emissions_kg: Decimal = Field(
        ..., description="Total indirect use-phase emissions (kgCO2e)"
    )
    fuel_sales_emissions_kg: Decimal = Field(
        ..., description="Total fuels/feedstocks sold emissions (kgCO2e)"
    )
    total_emissions_kg: Decimal = Field(
        ..., description="Grand total emissions (kgCO2e)"
    )
    total_tco2e: Decimal = Field(
        ..., description="Grand total in metric tonnes CO2e"
    )
    product_breakdowns: List[ProductBreakdown] = Field(
        default_factory=list,
        description="Per-product emissions breakdown"
    )
    dqi: Optional[DataQualityScore] = Field(
        default=None,
        description="Overall data quality indicator"
    )
    uncertainty: Optional[UncertaintyResult] = Field(
        default=None,
        description="Uncertainty quantification result"
    )
    provenance_hash: str = Field(
        ..., description="Final SHA-256 provenance chain hash"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="ISO 8601 calculation timestamp"
    )

    model_config = ConfigDict(frozen=True)


class AggregationResult(BaseModel):
    """
    Aggregated emissions result by various dimensions.

    Provides total emissions with breakdowns by product category,
    emission type, and reporting period.

    Example:
        >>> agg = AggregationResult(
        ...     period="2025",
        ...     total_tco2e=Decimal("81515"),
        ...     direct_tco2e=Decimal("41670"),
        ...     indirect_tco2e=Decimal("37530"),
        ...     fuels_tco2e=Decimal("2315"),
        ...     by_category={"vehicles": Decimal("41670")},
        ...     by_emission_type={"direct_fuel_combustion": Decimal("41670")}
        ... )
    """

    period: str = Field(
        ..., description="Reporting period (e.g., '2025', '2025-Q3')"
    )
    total_tco2e: Decimal = Field(
        ..., description="Total emissions in tCO2e"
    )
    direct_tco2e: Decimal = Field(
        ..., description="Direct use-phase emissions in tCO2e"
    )
    indirect_tco2e: Decimal = Field(
        ..., description="Indirect use-phase emissions in tCO2e"
    )
    fuels_tco2e: Decimal = Field(
        ..., description="Fuels/feedstocks sold emissions in tCO2e"
    )
    by_category: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by product category (tCO2e)"
    )
    by_emission_type: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by use-phase emission type (tCO2e)"
    )

    model_config = ConfigDict(frozen=True)


class ComplianceResult(BaseModel):
    """
    Result from compliance checking against a specific framework.

    Reports pass/fail status with the number of rules checked,
    passed, and failed, along with specific findings.

    Example:
        >>> compliance = ComplianceResult(
        ...     framework=ComplianceFramework.GHG_PROTOCOL,
        ...     status=ComplianceStatus.PASS,
        ...     rules_checked=12,
        ...     rules_passed=12,
        ...     rules_failed=0,
        ...     findings=[]
        ... )
    """

    framework: ComplianceFramework = Field(
        ..., description="Compliance framework checked"
    )
    status: ComplianceStatus = Field(
        ..., description="Overall compliance status"
    )
    rules_checked: int = Field(
        ..., ge=0,
        description="Total number of rules checked"
    )
    rules_passed: int = Field(
        ..., ge=0,
        description="Number of rules passed"
    )
    rules_failed: int = Field(
        ..., ge=0,
        description="Number of rules failed"
    )
    findings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Specific findings (gaps, issues, recommendations)"
    )

    model_config = ConfigDict(frozen=True)


# ==============================================================================
# HELPER FUNCTIONS (16)
# ==============================================================================

# Quantization constant: 8 decimal places for regulatory precision
_QUANT_8DP = Decimal("0.00000001")


def get_product_profile(
    category: ProductUseCategory, product_type: str
) -> Optional[Dict[str, Any]]:
    """
    Look up product energy profile by category and product type.

    Args:
        category: Product use category.
        product_type: Product sub-type key (e.g., 'PASSENGER_CAR_GASOLINE').

    Returns:
        Product profile dict with lifetime_years, annual_consumption, unit;
        or None if not found.

    Example:
        >>> profile = get_product_profile(
        ...     ProductUseCategory.VEHICLES, "PASSENGER_CAR_GASOLINE"
        ... )
        >>> profile["lifetime_years"]
        Decimal('15')
        >>> profile["annual_consumption"]
        Decimal('1200')
    """
    key = product_type.upper()
    profile = PRODUCT_ENERGY_PROFILES.get(key)
    if profile is not None and profile["category"] == category:
        return profile
    return None


def get_fuel_ef(fuel_type: FuelType) -> Optional[Dict[str, Decimal]]:
    """
    Get fuel combustion emission factor and NCV.

    Args:
        fuel_type: Fuel type enum value.

    Returns:
        Dict with 'ef' (kgCO2e/unit) and 'ncv' (MJ/unit); or None if not found.

    Example:
        >>> factors = get_fuel_ef(FuelType.GASOLINE)
        >>> factors["ef"]
        Decimal('2.315')
        >>> factors["ncv"]
        Decimal('34.2')
    """
    return FUEL_EMISSION_FACTORS.get(fuel_type.value)


def get_refrigerant_gwp(
    ref_type: RefrigerantType, standard: GWPStandard = GWPStandard.AR5
) -> Optional[Decimal]:
    """
    Get refrigerant GWP value for the specified assessment report version.

    Args:
        ref_type: Refrigerant type enum value.
        standard: GWP assessment report version (AR5 or AR6).

    Returns:
        GWP value as Decimal; or None if not found.

    Example:
        >>> get_refrigerant_gwp(RefrigerantType.R134A, GWPStandard.AR5)
        Decimal('1430')
        >>> get_refrigerant_gwp(RefrigerantType.R134A, GWPStandard.AR6)
        Decimal('1530')
    """
    entry = REFRIGERANT_GWPS.get(ref_type.value)
    if entry is None:
        return None
    key = f"gwp_{standard.value}"
    return entry.get(key)


def get_grid_ef(region: GridRegion) -> Optional[Decimal]:
    """
    Get grid emission factor for a region.

    Args:
        region: Grid region enum value.

    Returns:
        Grid emission factor in kgCO2e/kWh; or None if not found.

    Example:
        >>> get_grid_ef(GridRegion.US)
        Decimal('0.417')
        >>> get_grid_ef(GridRegion.FR)
        Decimal('0.052')
    """
    return GRID_EMISSION_FACTORS.get(region.value)


def get_lifetime_adjustment(adj: LifetimeAdjustment) -> Decimal:
    """
    Get lifetime adjustment multiplier.

    Args:
        adj: Lifetime adjustment type.

    Returns:
        Multiplier as Decimal (e.g., 1.00 for STANDARD, 0.80 for HEAVY).

    Example:
        >>> get_lifetime_adjustment(LifetimeAdjustment.HEAVY)
        Decimal('0.80')
        >>> get_lifetime_adjustment(LifetimeAdjustment.LIGHT)
        Decimal('1.20')
    """
    return LIFETIME_ADJUSTMENT_FACTORS.get(
        adj.value, Decimal("1.00")
    )


def get_degradation_rate(category: ProductUseCategory) -> Decimal:
    """
    Get annual energy degradation rate for a product category.

    Args:
        category: Product use category.

    Returns:
        Annual degradation rate as Decimal fraction (e.g., 0.015 for 1.5%).
        Returns Decimal("0.000") for categories without degradation data.

    Example:
        >>> get_degradation_rate(ProductUseCategory.VEHICLES)
        Decimal('0.015')
        >>> get_degradation_rate(ProductUseCategory.IT_EQUIPMENT)
        Decimal('0.000')
    """
    return ENERGY_DEGRADATION_RATES.get(
        category.value, Decimal("0.000")
    )


def get_steam_factor(source: str) -> Optional[Dict[str, Decimal]]:
    """
    Get steam/cooling emission factor by source type.

    Args:
        source: Steam/cooling source key (e.g., 'district_heating').

    Returns:
        Dict with 'ef_kgco2e_per_mj'; or None if not found.

    Example:
        >>> factor = get_steam_factor("district_heating")
        >>> factor["ef_kgco2e_per_mj"]
        Decimal('0.060')
    """
    return STEAM_COOLING_FACTORS.get(source)


def get_chemical_product(name: str) -> Optional[Dict[str, Decimal]]:
    """
    Get chemical product GHG properties.

    Args:
        name: Chemical product key (e.g., 'aerosol_propellant').

    Returns:
        Dict with 'ghg_content_kg', 'release_fraction', 'gwp';
        or None if not found.

    Example:
        >>> chem = get_chemical_product("aerosol_propellant")
        >>> chem["gwp"]
        Decimal('1430')
        >>> chem["release_fraction"]
        Decimal('1.00')
    """
    return CHEMICAL_PRODUCTS.get(name)


def get_dqi_score(dim: DQIDimension, level: DataQualityTier) -> Decimal:
    """
    Get DQI score for a specific dimension and quality tier.

    Args:
        dim: DQI dimension (reliability, completeness, etc.).
        level: Data quality tier (tier_1, tier_2, tier_3).

    Returns:
        Score value as Decimal (1-5). Returns Decimal("1") if not found.

    Example:
        >>> get_dqi_score(DQIDimension.RELIABILITY, DataQualityTier.TIER_1)
        Decimal('5')
        >>> get_dqi_score(DQIDimension.TEMPORAL, DataQualityTier.TIER_3)
        Decimal('1')
    """
    dim_scores = DQI_SCORING.get(dim.value)
    if dim_scores is None:
        return Decimal("1")
    return dim_scores.get(level.value, Decimal("1"))


def get_uncertainty_range(method: CalculationMethod) -> Optional[Dict[str, Decimal]]:
    """
    Get uncertainty range (min/default/max) for a calculation method.

    Args:
        method: Calculation method enum value.

    Returns:
        Dict with 'min', 'default', 'max' as Decimal fractions;
        or None if not found.

    Example:
        >>> rng = get_uncertainty_range(CalculationMethod.DIRECT_FUEL)
        >>> rng["default"]
        Decimal('0.15')
        >>> rng["max"]
        Decimal('0.25')
    """
    return UNCERTAINTY_RANGES.get(method.value)


def get_dc_rule(rule_id: str) -> Optional[Dict[str, str]]:
    """
    Get double-counting prevention rule by ID.

    Args:
        rule_id: Rule identifier (e.g., 'DC-USP-001').

    Returns:
        Dict with 'rule' and 'description'; or None if not found.

    Example:
        >>> rule = get_dc_rule("DC-USP-001")
        >>> rule["rule"]
        'vs Scope 1'
        >>> rule["description"]
        'Exclude direct emissions from own use of products'
    """
    return DC_RULES.get(rule_id)


def get_framework_rules(
    framework: ComplianceFramework,
) -> Optional[Dict[str, Any]]:
    """
    Get compliance framework requirements and disclosures.

    Args:
        framework: Compliance framework enum value.

    Returns:
        Dict with 'name', 'reference', 'required_disclosures';
        or None if not found.

    Example:
        >>> rules = get_framework_rules(ComplianceFramework.GHG_PROTOCOL)
        >>> rules["reference"]
        'Chapter 6, Category 11'
        >>> len(rules["required_disclosures"])
        7
    """
    return COMPLIANCE_FRAMEWORK_RULES.get(framework.value)


def get_feedstock_properties(feedstock: str) -> Optional[Dict[str, Decimal]]:
    """
    Get feedstock properties for oxidation-based emissions calculation.

    Args:
        feedstock: Feedstock key (e.g., 'naphtha', 'coal_feedstock').

    Returns:
        Dict with 'carbon_content' and 'oxidation_factor'; or None if not found.

    Example:
        >>> props = get_feedstock_properties("naphtha")
        >>> props["carbon_content"]
        Decimal('0.836')
        >>> props["oxidation_factor"]
        Decimal('0.990')
    """
    return FEEDSTOCK_PROPERTIES.get(feedstock)


def get_product_types(category: ProductUseCategory) -> List[str]:
    """
    Get list of valid product sub-types for a product category.

    Args:
        category: Product use category.

    Returns:
        List of product sub-type strings. Returns empty list if not found.

    Example:
        >>> get_product_types(ProductUseCategory.VEHICLES)
        ['passenger_car_gasoline', 'passenger_car_diesel', ...]
        >>> get_product_types(ProductUseCategory.HVAC)
        ['room_ac', 'central_ac', 'heat_pump', 'gas_furnace']
    """
    return PRODUCT_TYPE_MAP.get(category.value, [])


def get_emission_types(category: ProductUseCategory) -> List[str]:
    """
    Get applicable emission types for a product category.

    Args:
        category: Product use category.

    Returns:
        List of emission type strings (direct/indirect). Returns empty list
        if not found.

    Example:
        >>> get_emission_types(ProductUseCategory.VEHICLES)
        ['direct_fuel_combustion']
        >>> get_emission_types(ProductUseCategory.HVAC)
        ['direct_refrigerant_leakage', 'indirect_electricity']
    """
    return EMISSION_TYPE_MAP.get(category.value, [])


def get_default_usage_hours(category: ProductUseCategory) -> Decimal:
    """
    Get default annual usage hours for a product category.

    Args:
        category: Product use category.

    Returns:
        Annual usage hours as Decimal. Returns Decimal("0") if not found.

    Example:
        >>> get_default_usage_hours(ProductUseCategory.VEHICLES)
        Decimal('600')
        >>> get_default_usage_hours(ProductUseCategory.IT_EQUIPMENT)
        Decimal('2500')
    """
    return DEFAULT_USAGE_HOURS.get(category.value, Decimal("0"))


# ==============================================================================
# PROVENANCE HELPER
# ==============================================================================


def calculate_provenance_hash(*inputs: Any) -> str:
    """
    Calculate SHA-256 provenance hash from variable inputs.

    Supports Pydantic models (serialized to sorted JSON), Decimal values,
    and any other stringifiable objects. Used to build the 10-stage
    provenance chain for audit trail integrity.

    Args:
        *inputs: Variable number of input objects to hash.

    Returns:
        Hexadecimal SHA-256 hash string (64 characters).

    Example:
        >>> h = calculate_provenance_hash("PROD-001", Decimal("10000"))
        >>> len(h)
        64
    """
    hash_input = ""
    for inp in inputs:
        if isinstance(inp, BaseModel):
            hash_input += json.dumps(
                inp.model_dump(mode="json"), sort_keys=True, default=str
            )
        elif isinstance(inp, Decimal):
            hash_input += str(
                inp.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)
            )
        else:
            hash_input += str(inp)

    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Metadata
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",

    # Enumerations (22)
    "ProductUseCategory",
    "UsePhaseEmissionType",
    "VehicleType",
    "ApplianceType",
    "HVACType",
    "ITEquipmentType",
    "IndustrialType",
    "FuelType",
    "RefrigerantType",
    "GWPStandard",
    "GridRegion",
    "LifetimeAdjustment",
    "CalculationMethod",
    "DataQualityTier",
    "DQIDimension",
    "ComplianceFramework",
    "ComplianceStatus",
    "PipelineStage",
    "ProvenanceStage",
    "UncertaintyMethod",
    "BatchStatus",
    "AuditAction",

    # Constant Tables (16)
    "PRODUCT_ENERGY_PROFILES",
    "FUEL_EMISSION_FACTORS",
    "REFRIGERANT_GWPS",
    "GRID_EMISSION_FACTORS",
    "LIFETIME_ADJUSTMENT_FACTORS",
    "ENERGY_DEGRADATION_RATES",
    "STEAM_COOLING_FACTORS",
    "CHEMICAL_PRODUCTS",
    "DQI_SCORING",
    "UNCERTAINTY_RANGES",
    "DC_RULES",
    "COMPLIANCE_FRAMEWORK_RULES",
    "FEEDSTOCK_PROPERTIES",
    "PRODUCT_TYPE_MAP",
    "EMISSION_TYPE_MAP",
    "DEFAULT_USAGE_HOURS",

    # Pydantic Models (14)
    "ProductInput",
    "FuelSalesInput",
    "DirectEmissionsInput",
    "IndirectEmissionsInput",
    "FuelsAndFeedstocksInput",
    "CalculationResult",
    "ProductBreakdown",
    "DirectEmissionDetail",
    "IndirectEmissionDetail",
    "AggregationResult",
    "ComplianceResult",
    "ProvenanceRecord",
    "DataQualityScore",
    "UncertaintyResult",

    # Helper Functions (16 + 1 provenance)
    "get_product_profile",
    "get_fuel_ef",
    "get_refrigerant_gwp",
    "get_grid_ef",
    "get_lifetime_adjustment",
    "get_degradation_rate",
    "get_steam_factor",
    "get_chemical_product",
    "get_dqi_score",
    "get_uncertainty_range",
    "get_dc_rule",
    "get_framework_rules",
    "get_feedstock_properties",
    "get_product_types",
    "get_emission_types",
    "get_default_usage_hours",
    "calculate_provenance_hash",
]
