"""
Downstream Leased Assets Agent Models (AGENT-MRV-026)

This module provides comprehensive data models for GHG Protocol Scope 3 Category 13
(Downstream Leased Assets) emissions calculations.

Category 13 vs Category 8 Context:
    Cat 13 = Downstream Leased Assets = assets OWNED by the reporter and LEASED TO
    others (reporter is LESSOR). The reporter must account for Scope 1 and Scope 2
    emissions occurring from the operation of assets it owns but has leased to tenants.
    Cat 8 = Upstream Leased Assets = assets LEASED BY the reporter (reporter is LESSEE).
    Both categories share asset types and benchmark data, but Cat 13 additionally
    requires tenant data collection, vacancy handling, common-area allocation, and
    operational-control boundary determination.

Supports:
- 4 asset categories (buildings, vehicles, equipment, IT assets)
- 4 calculation methods (asset-specific, average-data, spend-based, hybrid)
- 8 building types x 5 climate zones with EUI benchmarks (kWh/m2/yr)
- 8 vehicle types x 7 fuel types with DEFRA 2024 emission factors
- 6 equipment types with fuel consumption and load-factor parameters
- 7 IT asset types with PUE-adjusted power ratings
- 12 country + 26 eGRID subregion grid emission factors
- 8 fuel types with direct and WTT emission factors (DEFRA 2024)
- 10 EEIO leasing/rental NAICS codes (spend-based fallback)
- Vacancy base-load fractions by building type
- 15 refrigerant GWPs (IPCC AR6)
- 30+ country-to-climate-zone mappings
- 6 allocation methods (floor area, headcount, revenue, FTE, equal, custom)
- 3 lease types (operating, finance, sale-leaseback)
- 3 consolidation approaches (financial control, equity share, operational control)
- 8 double-counting prevention rules (DC-DLA-001 through DC-DLA-008)
- 7 compliance frameworks (GHG Protocol, ISO 14064, CSRD, CDP, SBTi, SB 253, GRI)
- Data quality indicators (5-dimension DQI scoring, 3 tiers)
- Uncertainty quantification (Monte Carlo, analytical, IPCC Tier 2)
- SHA-256 provenance chain with 10-stage pipeline
- Avoided emissions tracking (energy efficiency, renewables, green leases)

All numeric fields use Decimal for precision in regulatory calculations.
All models are frozen (immutable) for audit trail integrity.

Example:
    >>> from greenlang.downstream_leased_assets.models import (
    ...     BuildingAssetInput, BuildingType, ClimateZone, OccupancyStatus
    ... )
    >>> building = BuildingAssetInput(
    ...     building_type=BuildingType.OFFICE,
    ...     floor_area_sqm=Decimal("5000"),
    ...     climate_zone=ClimateZone.TEMPERATE,
    ...     country_code="US",
    ...     lease_share=Decimal("1.0"),
    ...     occupancy_status=OccupancyStatus.OCCUPIED,
    ...     tenant_count=3
    ... )
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any
from datetime import datetime, date
from enum import Enum
from pydantic import BaseModel, Field, validator
from pydantic import ConfigDict
import hashlib
import json

# ==============================================================================
# AGENT METADATA
# ==============================================================================

AGENT_ID: str = "GL-MRV-S3-013"
AGENT_COMPONENT: str = "AGENT-MRV-026"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_dla_"

# ==============================================================================
# ENUMERATIONS
# ==============================================================================


class AssetCategory(str, Enum):
    """Primary asset categories for downstream leased assets."""

    BUILDING = "building"  # Commercial / residential buildings leased to tenants
    VEHICLE = "vehicle"  # Fleet vehicles leased to other organisations
    EQUIPMENT = "equipment"  # Industrial / construction equipment leased out
    IT_ASSET = "it_asset"  # IT infrastructure leased (servers, network gear)


class BuildingType(str, Enum):
    """Building types with distinct energy use intensity (EUI) profiles."""

    OFFICE = "office"  # Commercial office space
    RETAIL = "retail"  # Retail / shopping
    WAREHOUSE = "warehouse"  # Warehouse / distribution centre
    INDUSTRIAL = "industrial"  # Light industrial / manufacturing
    DATA_CENTER = "data_center"  # Data centre / colocation facility
    HOTEL = "hotel"  # Hotel / hospitality
    HEALTHCARE = "healthcare"  # Hospital / clinic
    RESIDENTIAL_MULTIFAMILY = "residential_multifamily"  # Multi-family residential


class VehicleType(str, Enum):
    """Vehicle types for leased fleet emissions calculations."""

    SMALL_CAR = "small_car"  # Small / compact car (< 1.4L)
    MEDIUM_CAR = "medium_car"  # Medium car (1.4-2.0L)
    LARGE_CAR = "large_car"  # Large car / executive (> 2.0L)
    SUV = "suv"  # Sport utility vehicle
    LIGHT_VAN = "light_van"  # Light commercial van (< 3.5t)
    HEAVY_VAN = "heavy_van"  # Heavy van / panel van (> 3.5t)
    LIGHT_TRUCK = "light_truck"  # Light-duty truck
    HEAVY_TRUCK = "heavy_truck"  # Heavy-duty truck / HGV


class FuelType(str, Enum):
    """Fuel types for vehicle and equipment emissions."""

    GASOLINE = "gasoline"  # Petrol / gasoline
    DIESEL = "diesel"  # Diesel
    LPG = "lpg"  # Liquefied petroleum gas
    CNG = "cng"  # Compressed natural gas
    HYBRID = "hybrid"  # Hybrid electric vehicle (HEV)
    PHEV = "phev"  # Plug-in hybrid electric vehicle
    BEV = "bev"  # Battery electric vehicle (zero tailpipe)


class EquipmentType(str, Enum):
    """Equipment types for leased industrial / construction assets."""

    MANUFACTURING = "manufacturing"  # Manufacturing machinery
    CONSTRUCTION = "construction"  # Construction equipment (excavators, cranes)
    GENERATOR = "generator"  # Diesel / gas generators
    AGRICULTURAL = "agricultural"  # Agricultural machinery (tractors, harvesters)
    MINING = "mining"  # Mining equipment
    HVAC = "hvac"  # HVAC systems / chillers


class ITAssetType(str, Enum):
    """IT asset types for leased technology infrastructure."""

    SERVER = "server"  # Rack-mount / blade server
    NETWORK_SWITCH = "network_switch"  # Network switch / router
    STORAGE = "storage"  # Storage array / SAN
    DESKTOP = "desktop"  # Desktop workstation
    LAPTOP = "laptop"  # Laptop / notebook
    PRINTER = "printer"  # Network printer
    COPIER = "copier"  # Photocopier / MFD


class ClimateZone(str, Enum):
    """Climate zones affecting building energy use intensity."""

    TROPICAL = "tropical"  # Tropical (high cooling demand)
    ARID = "arid"  # Arid / desert (extreme cooling)
    TEMPERATE = "temperate"  # Temperate (moderate heating and cooling)
    CONTINENTAL = "continental"  # Continental (high heating demand)
    POLAR = "polar"  # Polar / subarctic (extreme heating demand)


class CalculationMethod(str, Enum):
    """Calculation methods per GHG Protocol Scope 3 Technical Guidance."""

    ASSET_SPECIFIC = "asset_specific"  # Metered energy data from tenants
    AVERAGE_DATA = "average_data"  # EUI benchmarks by building/asset type
    SPEND_BASED = "spend_based"  # Lease revenue x EEIO factor
    HYBRID = "hybrid"  # Weighted combination of multiple methods


class AllocationMethod(str, Enum):
    """Methods for allocating emissions across tenants in multi-tenant assets."""

    FLOOR_AREA = "floor_area"  # Proportional to leased floor area (m2)
    HEADCOUNT = "headcount"  # Proportional to tenant headcount
    REVENUE = "revenue"  # Proportional to tenant revenue
    FTE = "fte"  # Proportional to full-time equivalents
    EQUAL_SHARE = "equal_share"  # Equal split across tenants
    CUSTOM = "custom"  # Custom weighting provided by reporter


class LeaseType(str, Enum):
    """Lease classification per IFRS 16 / ASC 842."""

    OPERATING = "operating"  # Operating lease (lessor retains ownership)
    FINANCE = "finance"  # Finance / capital lease (risks transfer)
    SALE_LEASEBACK = "sale_leaseback"  # Sale-and-leaseback arrangement


class ConsolidationApproach(str, Enum):
    """GHG Protocol organisational boundary approaches."""

    FINANCIAL_CONTROL = "financial_control"  # 100% if financial control
    EQUITY_SHARE = "equity_share"  # Proportional to equity ownership %
    OPERATIONAL_CONTROL = "operational_control"  # 100% if operational control


class EFSource(str, Enum):
    """Emission factor data sources."""

    DEFRA_2024 = "DEFRA_2024"  # UK DEFRA/DESNZ 2024 conversion factors
    EPA_2024 = "EPA_2024"  # US EPA emission factors 2024
    IEA_2024 = "IEA_2024"  # IEA world energy statistics 2024
    EGRID_2024 = "EGRID_2024"  # EPA eGRID subregional factors 2024
    IPCC_AR6 = "IPCC_AR6"  # IPCC Sixth Assessment Report
    CUSTOM = "CUSTOM"  # Organisation-specific / custom factors


class DataQualityTier(str, Enum):
    """Data quality tiers affecting uncertainty ranges and DQI scoring."""

    TIER_1 = "tier_1"  # Primary / metered / supplier-specific data
    TIER_2 = "tier_2"  # Regional / asset-type secondary data
    TIER_3 = "tier_3"  # Global average / spend-based estimates


class DQIDimension(str, Enum):
    """Data Quality Indicator dimensions per GHG Protocol guidance."""

    TEMPORAL = "temporal"  # Temporal correlation to reporting year
    GEOGRAPHICAL = "geographical"  # Geographical correlation to asset location
    TECHNOLOGICAL = "technological"  # Technological correlation to asset type
    COMPLETENESS = "completeness"  # Fraction of portfolio data coverage
    RELIABILITY = "reliability"  # Reliability of measurement / estimation


class ComplianceFramework(str, Enum):
    """Regulatory and voluntary reporting frameworks."""

    GHG_PROTOCOL = "ghg_protocol"  # GHG Protocol Scope 3 Standard
    ISO_14064 = "iso_14064"  # ISO 14064-1:2018
    CSRD_ESRS = "csrd_esrs"  # CSRD ESRS E1 Climate Change
    CDP = "cdp"  # CDP Climate Change Questionnaire
    SBTI = "sbti"  # Science Based Targets initiative
    SB_253 = "sb_253"  # California SB 253 (Climate Corporate Data Accountability)
    GRI = "gri"  # GRI 305 Emissions Standard


class ComplianceStatus(str, Enum):
    """Compliance check result status."""

    COMPLIANT = "compliant"  # Fully meets framework requirements
    NON_COMPLIANT = "non_compliant"  # Fails to meet requirements
    PARTIAL = "partial"  # Partially compliant / needs remediation
    NOT_APPLICABLE = "not_applicable"  # Framework requirement does not apply


class PipelineStage(str, Enum):
    """Processing pipeline stages for provenance tracking."""

    VALIDATE = "validate"  # Input validation and schema checks
    CLASSIFY = "classify"  # Asset classification (category, type, zone)
    NORMALIZE = "normalize"  # Unit normalization (area, distance, energy)
    RESOLVE_EFS = "resolve_efs"  # Emission factor resolution and lookup
    CALCULATE = "calculate"  # Core emissions calculation (zero-hallucination)
    ALLOCATE = "allocate"  # Tenant allocation and common-area split
    AGGREGATE = "aggregate"  # Portfolio-level aggregation
    COMPLIANCE = "compliance"  # Compliance checks across frameworks
    PROVENANCE = "provenance"  # Provenance chain construction
    SEAL = "seal"  # Final chain sealing with root hash


class UncertaintyMethod(str, Enum):
    """Uncertainty quantification methods."""

    MONTE_CARLO = "monte_carlo"  # Monte Carlo simulation
    ANALYTICAL = "analytical"  # Analytical error propagation
    IPCC_TIER2 = "ipcc_tier2"  # IPCC Tier 2 default uncertainty ranges


class BatchStatus(str, Enum):
    """Batch processing status for portfolio-level calculations."""

    PENDING = "pending"  # Awaiting processing
    PROCESSING = "processing"  # Currently processing
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Processing failed


class GWPSource(str, Enum):
    """IPCC Global Warming Potential assessment report version."""

    AR5 = "AR5"  # Fifth Assessment Report (100-year)
    AR6 = "AR6"  # Sixth Assessment Report (100-year)


class EnergyType(str, Enum):
    """Energy types consumed in leased buildings."""

    ELECTRICITY = "electricity"  # Grid or on-site electricity
    NATURAL_GAS = "natural_gas"  # Piped natural gas
    STEAM = "steam"  # District steam / heating
    CHILLED_WATER = "chilled_water"  # District cooling


class OccupancyStatus(str, Enum):
    """Occupancy status of a leased asset (Cat 13 specific)."""

    OCCUPIED = "occupied"  # Fully occupied by tenant(s)
    VACANT = "vacant"  # Unoccupied (lessor bears base-load emissions)
    PARTIALLY_OCCUPIED = "partially_occupied"  # Some space is vacant


# ==============================================================================
# CONSTANT TABLES
# ==============================================================================

# Quantization constant: 8 decimal places for Decimal precision
_QUANT_8DP = Decimal("0.00000001")

# 1. Building Energy Use Intensity benchmarks (kWh/m2/year)
# Source: ENERGY STAR Portfolio Manager, CIBSE TM46, IEA
# Rows: 8 building types, Columns: 5 climate zones
BUILDING_EUI_BENCHMARKS: Dict[str, Dict[str, Decimal]] = {
    BuildingType.OFFICE.value: {
        ClimateZone.TROPICAL.value: Decimal("220"),
        ClimateZone.ARID.value: Decimal("240"),
        ClimateZone.TEMPERATE.value: Decimal("180"),
        ClimateZone.CONTINENTAL.value: Decimal("210"),
        ClimateZone.POLAR.value: Decimal("260"),
    },
    BuildingType.RETAIL.value: {
        ClimateZone.TROPICAL.value: Decimal("270"),
        ClimateZone.ARID.value: Decimal("290"),
        ClimateZone.TEMPERATE.value: Decimal("220"),
        ClimateZone.CONTINENTAL.value: Decimal("250"),
        ClimateZone.POLAR.value: Decimal("310"),
    },
    BuildingType.WAREHOUSE.value: {
        ClimateZone.TROPICAL.value: Decimal("140"),
        ClimateZone.ARID.value: Decimal("150"),
        ClimateZone.TEMPERATE.value: Decimal("120"),
        ClimateZone.CONTINENTAL.value: Decimal("135"),
        ClimateZone.POLAR.value: Decimal("170"),
    },
    BuildingType.INDUSTRIAL.value: {
        ClimateZone.TROPICAL.value: Decimal("260"),
        ClimateZone.ARID.value: Decimal("280"),
        ClimateZone.TEMPERATE.value: Decimal("230"),
        ClimateZone.CONTINENTAL.value: Decimal("250"),
        ClimateZone.POLAR.value: Decimal("300"),
    },
    BuildingType.DATA_CENTER.value: {
        ClimateZone.TROPICAL.value: Decimal("4000"),
        ClimateZone.ARID.value: Decimal("3800"),
        ClimateZone.TEMPERATE.value: Decimal("3500"),
        ClimateZone.CONTINENTAL.value: Decimal("3600"),
        ClimateZone.POLAR.value: Decimal("3200"),
    },
    BuildingType.HOTEL.value: {
        ClimateZone.TROPICAL.value: Decimal("320"),
        ClimateZone.ARID.value: Decimal("340"),
        ClimateZone.TEMPERATE.value: Decimal("280"),
        ClimateZone.CONTINENTAL.value: Decimal("310"),
        ClimateZone.POLAR.value: Decimal("370"),
    },
    BuildingType.HEALTHCARE.value: {
        ClimateZone.TROPICAL.value: Decimal("400"),
        ClimateZone.ARID.value: Decimal("420"),
        ClimateZone.TEMPERATE.value: Decimal("350"),
        ClimateZone.CONTINENTAL.value: Decimal("380"),
        ClimateZone.POLAR.value: Decimal("450"),
    },
    BuildingType.RESIDENTIAL_MULTIFAMILY.value: {
        ClimateZone.TROPICAL.value: Decimal("170"),
        ClimateZone.ARID.value: Decimal("185"),
        ClimateZone.TEMPERATE.value: Decimal("150"),
        ClimateZone.CONTINENTAL.value: Decimal("175"),
        ClimateZone.POLAR.value: Decimal("220"),
    },
}

# 2. Vehicle emission factors (kgCO2e/km) by vehicle type x fuel type
# Source: DEFRA 2024, EPA SmartWay. BEV = 0.0 tailpipe (Scope 2 upstream).
VEHICLE_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    VehicleType.SMALL_CAR.value: {
        FuelType.GASOLINE.value: Decimal("0.14910"),
        FuelType.DIESEL.value: Decimal("0.13860"),
        FuelType.LPG.value: Decimal("0.12040"),
        FuelType.CNG.value: Decimal("0.11500"),
        FuelType.HYBRID.value: Decimal("0.10210"),
        FuelType.PHEV.value: Decimal("0.06450"),
        FuelType.BEV.value: Decimal("0.0"),
    },
    VehicleType.MEDIUM_CAR.value: {
        FuelType.GASOLINE.value: Decimal("0.18490"),
        FuelType.DIESEL.value: Decimal("0.16680"),
        FuelType.LPG.value: Decimal("0.14520"),
        FuelType.CNG.value: Decimal("0.13870"),
        FuelType.HYBRID.value: Decimal("0.12350"),
        FuelType.PHEV.value: Decimal("0.07800"),
        FuelType.BEV.value: Decimal("0.0"),
    },
    VehicleType.LARGE_CAR.value: {
        FuelType.GASOLINE.value: Decimal("0.27850"),
        FuelType.DIESEL.value: Decimal("0.22350"),
        FuelType.LPG.value: Decimal("0.19400"),
        FuelType.CNG.value: Decimal("0.18540"),
        FuelType.HYBRID.value: Decimal("0.16540"),
        FuelType.PHEV.value: Decimal("0.10450"),
        FuelType.BEV.value: Decimal("0.0"),
    },
    VehicleType.SUV.value: {
        FuelType.GASOLINE.value: Decimal("0.23150"),
        FuelType.DIESEL.value: Decimal("0.20920"),
        FuelType.LPG.value: Decimal("0.18170"),
        FuelType.CNG.value: Decimal("0.17360"),
        FuelType.HYBRID.value: Decimal("0.15470"),
        FuelType.PHEV.value: Decimal("0.09770"),
        FuelType.BEV.value: Decimal("0.0"),
    },
    VehicleType.LIGHT_VAN.value: {
        FuelType.GASOLINE.value: Decimal("0.23070"),
        FuelType.DIESEL.value: Decimal("0.20570"),
        FuelType.LPG.value: Decimal("0.17890"),
        FuelType.CNG.value: Decimal("0.17090"),
        FuelType.HYBRID.value: Decimal("0.15230"),
        FuelType.PHEV.value: Decimal("0.09620"),
        FuelType.BEV.value: Decimal("0.0"),
    },
    VehicleType.HEAVY_VAN.value: {
        FuelType.GASOLINE.value: Decimal("0.30400"),
        FuelType.DIESEL.value: Decimal("0.27130"),
        FuelType.LPG.value: Decimal("0.23590"),
        FuelType.CNG.value: Decimal("0.22540"),
        FuelType.HYBRID.value: Decimal("0.20090"),
        FuelType.PHEV.value: Decimal("0.12690"),
        FuelType.BEV.value: Decimal("0.0"),
    },
    VehicleType.LIGHT_TRUCK.value: {
        FuelType.GASOLINE.value: Decimal("0.37220"),
        FuelType.DIESEL.value: Decimal("0.33200"),
        FuelType.LPG.value: Decimal("0.28870"),
        FuelType.CNG.value: Decimal("0.27590"),
        FuelType.HYBRID.value: Decimal("0.24590"),
        FuelType.PHEV.value: Decimal("0.15530"),
        FuelType.BEV.value: Decimal("0.0"),
    },
    VehicleType.HEAVY_TRUCK.value: {
        FuelType.GASOLINE.value: Decimal("0.88450"),
        FuelType.DIESEL.value: Decimal("0.78920"),
        FuelType.LPG.value: Decimal("0.68620"),
        FuelType.CNG.value: Decimal("0.65580"),
        FuelType.HYBRID.value: Decimal("0.58430"),
        FuelType.PHEV.value: Decimal("0.36910"),
        FuelType.BEV.value: Decimal("0.0"),
    },
}

# 3. Equipment fuel consumption parameters by equipment type
# rated_power_kw: Typical rated power (kW)
# fuel_consumption_lph: Fuel consumption at rated power (litres/hour)
# load_factor: Default load factor (fraction of rated power)
EQUIPMENT_FUEL_CONSUMPTION: Dict[str, Dict[str, Decimal]] = {
    EquipmentType.MANUFACTURING.value: {
        "rated_power_kw": Decimal("150"),
        "fuel_consumption_lph": Decimal("38.0"),
        "load_factor": Decimal("0.55"),
    },
    EquipmentType.CONSTRUCTION.value: {
        "rated_power_kw": Decimal("200"),
        "fuel_consumption_lph": Decimal("50.0"),
        "load_factor": Decimal("0.50"),
    },
    EquipmentType.GENERATOR.value: {
        "rated_power_kw": Decimal("500"),
        "fuel_consumption_lph": Decimal("130.0"),
        "load_factor": Decimal("0.70"),
    },
    EquipmentType.AGRICULTURAL.value: {
        "rated_power_kw": Decimal("120"),
        "fuel_consumption_lph": Decimal("30.0"),
        "load_factor": Decimal("0.45"),
    },
    EquipmentType.MINING.value: {
        "rated_power_kw": Decimal("350"),
        "fuel_consumption_lph": Decimal("90.0"),
        "load_factor": Decimal("0.60"),
    },
    EquipmentType.HVAC.value: {
        "rated_power_kw": Decimal("75"),
        "fuel_consumption_lph": Decimal("20.0"),
        "load_factor": Decimal("0.65"),
    },
}

# 4. IT asset power ratings
# power_kw: Typical power draw per unit (kW)
# default_pue: Power Usage Effectiveness (data centre overhead multiplier)
# hours_per_year: Annual operating hours
IT_ASSET_POWER_RATINGS: Dict[str, Dict[str, Decimal]] = {
    ITAssetType.SERVER.value: {
        "power_kw": Decimal("0.500"),
        "default_pue": Decimal("1.58"),
        "hours_per_year": Decimal("8760"),
    },
    ITAssetType.NETWORK_SWITCH.value: {
        "power_kw": Decimal("0.150"),
        "default_pue": Decimal("1.58"),
        "hours_per_year": Decimal("8760"),
    },
    ITAssetType.STORAGE.value: {
        "power_kw": Decimal("0.800"),
        "default_pue": Decimal("1.58"),
        "hours_per_year": Decimal("8760"),
    },
    ITAssetType.DESKTOP.value: {
        "power_kw": Decimal("0.175"),
        "default_pue": Decimal("1.0"),
        "hours_per_year": Decimal("2080"),
    },
    ITAssetType.LAPTOP.value: {
        "power_kw": Decimal("0.065"),
        "default_pue": Decimal("1.0"),
        "hours_per_year": Decimal("2080"),
    },
    ITAssetType.PRINTER.value: {
        "power_kw": Decimal("0.300"),
        "default_pue": Decimal("1.0"),
        "hours_per_year": Decimal("2080"),
    },
    ITAssetType.COPIER.value: {
        "power_kw": Decimal("0.450"),
        "default_pue": Decimal("1.0"),
        "hours_per_year": Decimal("2080"),
    },
}

# 5. Grid emission factors (kgCO2e/kWh)
# 12 countries + 26 eGRID subregions
# Source: IEA 2024, EPA eGRID 2024
GRID_EMISSION_FACTORS: Dict[str, Decimal] = {
    # --- Country-level ---
    "US": Decimal("0.417"),
    "GB": Decimal("0.207"),
    "DE": Decimal("0.350"),
    "JP": Decimal("0.471"),
    "CN": Decimal("0.555"),
    "IN": Decimal("0.708"),
    "BR": Decimal("0.074"),
    "AU": Decimal("0.656"),
    "KR": Decimal("0.459"),
    "CA": Decimal("0.120"),
    "FR": Decimal("0.052"),
    "GLOBAL": Decimal("0.436"),
    # --- eGRID subregions (US) ---
    "AKGD": Decimal("0.437"),
    "AKMS": Decimal("0.211"),
    "AZNM": Decimal("0.402"),
    "CAMX": Decimal("0.225"),
    "ERCT": Decimal("0.396"),
    "FRCC": Decimal("0.393"),
    "HIMS": Decimal("0.507"),
    "HIOA": Decimal("0.683"),
    "MROE": Decimal("0.548"),
    "MROW": Decimal("0.448"),
    "NEWE": Decimal("0.227"),
    "NWPP": Decimal("0.280"),
    "NYCW": Decimal("0.255"),
    "NYLI": Decimal("0.480"),
    "NYUP": Decimal("0.115"),
    "PRMS": Decimal("0.600"),
    "RFCE": Decimal("0.320"),
    "RFCM": Decimal("0.540"),
    "RFCW": Decimal("0.470"),
    "RMPA": Decimal("0.530"),
    "SPNO": Decimal("0.500"),
    "SPSO": Decimal("0.430"),
    "SRMV": Decimal("0.374"),
    "SRMW": Decimal("0.610"),
    "SRSO": Decimal("0.420"),
    "SRTV": Decimal("0.440"),
}

# 6. Fuel emission factors (kgCO2e per litre, except CNG per kg)
# co2e_per_liter: Direct combustion emission factor
# wtt_factor: Well-to-tank upstream factor
# Source: DEFRA 2024
FUEL_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "gasoline": {
        "co2e_per_liter": Decimal("2.31480"),
        "wtt_factor": Decimal("0.58549"),
    },
    "diesel": {
        "co2e_per_liter": Decimal("2.70370"),
        "wtt_factor": Decimal("0.60927"),
    },
    "lpg": {
        "co2e_per_liter": Decimal("1.55370"),
        "wtt_factor": Decimal("0.32149"),
    },
    "cng": {
        "co2e_per_liter": Decimal("2.53970"),  # per kg
        "wtt_factor": Decimal("0.50870"),  # per kg
    },
    "kerosene": {
        "co2e_per_liter": Decimal("2.54070"),
        "wtt_factor": Decimal("0.59349"),
    },
    "fuel_oil": {
        "co2e_per_liter": Decimal("3.17920"),
        "wtt_factor": Decimal("0.66429"),
    },
    "natural_gas": {
        "co2e_per_liter": Decimal("2.02130"),  # per m3
        "wtt_factor": Decimal("0.34759"),  # per m3
    },
    "biodiesel": {
        "co2e_per_liter": Decimal("0.17830"),
        "wtt_factor": Decimal("0.52610"),
    },
}

# 7. EEIO spend-based factors (kgCO2e per USD) by NAICS leasing/rental codes
# Source: EPA USEEIO v2.0 / Exiobase 3
EEIO_SPEND_FACTORS: Dict[str, Dict[str, Any]] = {
    "531110": {
        "name": "Lessors of residential buildings",
        "ef": Decimal("0.15"),
    },
    "531120": {
        "name": "Lessors of nonresidential buildings",
        "ef": Decimal("0.18"),
    },
    "531190": {
        "name": "Lessors of other real estate",
        "ef": Decimal("0.16"),
    },
    "532111": {
        "name": "Passenger car rental and leasing",
        "ef": Decimal("0.22"),
    },
    "532112": {
        "name": "Truck, trailer and RV rental and leasing",
        "ef": Decimal("0.28"),
    },
    "532310": {
        "name": "General rental centres",
        "ef": Decimal("0.25"),
    },
    "532412": {
        "name": "Construction equipment rental and leasing",
        "ef": Decimal("0.35"),
    },
    "532420": {
        "name": "Office machinery and equipment rental",
        "ef": Decimal("0.12"),
    },
    "532490": {
        "name": "Other commercial equipment rental",
        "ef": Decimal("0.20"),
    },
    "518210": {
        "name": "Data processing and hosting (colocation)",
        "ef": Decimal("0.30"),
    },
}

# 8. Allocation defaults by building use type
# Default tenant share percentages when allocation data is unavailable
ALLOCATION_DEFAULTS: Dict[str, Dict[str, Decimal]] = {
    BuildingType.OFFICE.value: {
        "tenant_share": Decimal("0.85"),
        "common_area_share": Decimal("0.15"),
    },
    BuildingType.RETAIL.value: {
        "tenant_share": Decimal("0.80"),
        "common_area_share": Decimal("0.20"),
    },
    BuildingType.WAREHOUSE.value: {
        "tenant_share": Decimal("0.92"),
        "common_area_share": Decimal("0.08"),
    },
    BuildingType.INDUSTRIAL.value: {
        "tenant_share": Decimal("0.90"),
        "common_area_share": Decimal("0.10"),
    },
    BuildingType.DATA_CENTER.value: {
        "tenant_share": Decimal("0.75"),
        "common_area_share": Decimal("0.25"),
    },
    BuildingType.HOTEL.value: {
        "tenant_share": Decimal("0.70"),
        "common_area_share": Decimal("0.30"),
    },
    BuildingType.HEALTHCARE.value: {
        "tenant_share": Decimal("0.78"),
        "common_area_share": Decimal("0.22"),
    },
    BuildingType.RESIDENTIAL_MULTIFAMILY.value: {
        "tenant_share": Decimal("0.88"),
        "common_area_share": Decimal("0.12"),
    },
}

# 9. Vacancy base-load fractions by building type
# When a building is vacant, this fraction of normal energy is still consumed
# for HVAC set-back, lighting timers, security, fire suppression, elevators, etc.
VACANCY_BASE_LOAD: Dict[str, Decimal] = {
    BuildingType.OFFICE.value: Decimal("0.30"),
    BuildingType.RETAIL.value: Decimal("0.25"),
    BuildingType.WAREHOUSE.value: Decimal("0.15"),
    BuildingType.INDUSTRIAL.value: Decimal("0.20"),
    BuildingType.DATA_CENTER.value: Decimal("0.60"),
    BuildingType.HOTEL.value: Decimal("0.35"),
    BuildingType.HEALTHCARE.value: Decimal("0.40"),
    BuildingType.RESIDENTIAL_MULTIFAMILY.value: Decimal("0.25"),
}

# 10. Refrigerant GWPs (100-year, IPCC AR6)
# Used for fugitive emissions from leased HVAC / cooling equipment
REFRIGERANT_GWPS: Dict[str, Decimal] = {
    "R-410A": Decimal("2088"),
    "R-32": Decimal("675"),
    "R-134a": Decimal("1430"),
    "R-407C": Decimal("1774"),
    "R-404A": Decimal("3922"),
    "R-507A": Decimal("3985"),
    "R-22": Decimal("1810"),
    "R-290": Decimal("3"),
    "R-600a": Decimal("3"),
    "R-1234yf": Decimal("1"),
    "R-1234ze": Decimal("7"),
    "R-744": Decimal("1"),
    "R-717": Decimal("0"),
    "R-407A": Decimal("2107"),
    "R-448A": Decimal("1387"),
}

# 11. Country-to-climate-zone mappings
# Source: Koppen-Geiger classification (dominant zone per country)
COUNTRY_CLIMATE_ZONES: Dict[str, str] = {
    "US": ClimateZone.TEMPERATE.value,
    "GB": ClimateZone.TEMPERATE.value,
    "DE": ClimateZone.TEMPERATE.value,
    "FR": ClimateZone.TEMPERATE.value,
    "JP": ClimateZone.TEMPERATE.value,
    "CN": ClimateZone.CONTINENTAL.value,
    "IN": ClimateZone.TROPICAL.value,
    "BR": ClimateZone.TROPICAL.value,
    "AU": ClimateZone.ARID.value,
    "KR": ClimateZone.CONTINENTAL.value,
    "CA": ClimateZone.CONTINENTAL.value,
    "RU": ClimateZone.CONTINENTAL.value,
    "MX": ClimateZone.ARID.value,
    "ZA": ClimateZone.TEMPERATE.value,
    "SA": ClimateZone.ARID.value,
    "AE": ClimateZone.ARID.value,
    "SG": ClimateZone.TROPICAL.value,
    "TH": ClimateZone.TROPICAL.value,
    "MY": ClimateZone.TROPICAL.value,
    "ID": ClimateZone.TROPICAL.value,
    "PH": ClimateZone.TROPICAL.value,
    "NG": ClimateZone.TROPICAL.value,
    "KE": ClimateZone.TROPICAL.value,
    "EG": ClimateZone.ARID.value,
    "AR": ClimateZone.TEMPERATE.value,
    "CL": ClimateZone.TEMPERATE.value,
    "SE": ClimateZone.CONTINENTAL.value,
    "NO": ClimateZone.CONTINENTAL.value,
    "FI": ClimateZone.CONTINENTAL.value,
    "IS": ClimateZone.POLAR.value,
    "NZ": ClimateZone.TEMPERATE.value,
    "IT": ClimateZone.TEMPERATE.value,
    "ES": ClimateZone.TEMPERATE.value,
    "NL": ClimateZone.TEMPERATE.value,
    "BE": ClimateZone.TEMPERATE.value,
    "CH": ClimateZone.TEMPERATE.value,
    "AT": ClimateZone.CONTINENTAL.value,
    "PL": ClimateZone.CONTINENTAL.value,
    "DK": ClimateZone.TEMPERATE.value,
    "IE": ClimateZone.TEMPERATE.value,
}

# 12. Double-counting prevention rules (Cat 13 specific)
# Key: rule_id, Value: description and affected categories
DC_RULES: Dict[str, Dict[str, str]] = {
    "DC-DLA-001": {
        "description": "Building energy must not be double-counted with Scope 1 "
                       "(on-site combustion) if operational control is with lessor",
        "affected_categories": "Scope 1, Cat 13",
    },
    "DC-DLA-002": {
        "description": "Grid electricity must not be double-counted with Scope 2 "
                       "(purchased electricity) if already in lessor Scope 2 boundary",
        "affected_categories": "Scope 2, Cat 13",
    },
    "DC-DLA-003": {
        "description": "Vehicle emissions leased downstream must not be double-counted "
                       "with Cat 8 (upstream leased assets) by the lessee",
        "affected_categories": "Cat 8, Cat 13",
    },
    "DC-DLA-004": {
        "description": "IT asset emissions in colocation must not overlap with "
                       "data centre building emissions (avoid double energy count)",
        "affected_categories": "Cat 13 building, Cat 13 IT",
    },
    "DC-DLA-005": {
        "description": "Equipment emissions leased downstream must not be double-counted "
                       "with Cat 1 (purchased goods) or Cat 2 (capital goods) at tenant",
        "affected_categories": "Cat 1, Cat 2, Cat 13",
    },
    "DC-DLA-006": {
        "description": "Finance lease assets where risks transfer to lessee may already "
                       "be in lessee Scope 1/2 boundary per GHG Protocol",
        "affected_categories": "Scope 1, Scope 2, Cat 13",
    },
    "DC-DLA-007": {
        "description": "Common area energy in multi-tenant buildings must be allocated "
                       "to avoid double-counting across tenants",
        "affected_categories": "Cat 13 tenants",
    },
    "DC-DLA-008": {
        "description": "Vacancy base-load emissions must not overlap with "
                       "occupied period allocations for the same space",
        "affected_categories": "Cat 13 vacancy, Cat 13 occupied",
    },
}

# 13. Compliance framework requirements
# Each framework specifies a list of required disclosures for Cat 13
COMPLIANCE_FRAMEWORK_RULES: Dict[str, List[str]] = {
    ComplianceFramework.GHG_PROTOCOL.value: [
        "total_co2e_cat13",
        "calculation_method_disclosure",
        "ef_sources_cited",
        "exclusions_justified",
        "dqi_score",
        "double_counting_check",
        "consolidation_approach",
        "asset_category_breakdown",
    ],
    ComplianceFramework.ISO_14064.value: [
        "total_co2e_cat13",
        "uncertainty_analysis",
        "base_year_emissions",
        "methodology_description",
        "exclusions_justified",
        "consolidation_approach",
    ],
    ComplianceFramework.CSRD_ESRS.value: [
        "total_co2e_cat13",
        "category_breakdown",
        "methodology_description",
        "reduction_targets",
        "transition_actions",
        "value_chain_engagement",
    ],
    ComplianceFramework.CDP.value: [
        "total_co2e_cat13",
        "calculation_method_disclosure",
        "asset_category_breakdown",
        "verification_status",
        "reduction_initiatives",
    ],
    ComplianceFramework.SBTI.value: [
        "total_co2e_cat13",
        "target_coverage_pct",
        "base_year_emissions",
        "progress_tracking",
        "supplier_engagement",
    ],
    ComplianceFramework.SB_253.value: [
        "total_co2e_cat13",
        "methodology_description",
        "assurance_opinion",
        "consolidation_approach",
    ],
    ComplianceFramework.GRI.value: [
        "total_co2e_cat13",
        "gases_included",
        "base_year_disclosure",
        "standards_methodology",
        "calculation_tools",
    ],
}

# 14. DQI scoring matrix (score per dimension per tier, scale 1-5)
DQI_SCORING: Dict[str, Dict[str, Decimal]] = {
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
    DQIDimension.COMPLETENESS.value: {
        DataQualityTier.TIER_1.value: Decimal("5"),
        DataQualityTier.TIER_2.value: Decimal("3"),
        DataQualityTier.TIER_3.value: Decimal("2"),
    },
    DQIDimension.RELIABILITY.value: {
        DataQualityTier.TIER_1.value: Decimal("5"),
        DataQualityTier.TIER_2.value: Decimal("3"),
        DataQualityTier.TIER_3.value: Decimal("1"),
    },
}

# 15. Uncertainty ranges by calculation method and data quality tier
# Values represent the half-width of the 95% confidence interval as a fraction
UNCERTAINTY_RANGES: Dict[str, Dict[str, Decimal]] = {
    CalculationMethod.ASSET_SPECIFIC.value: {
        DataQualityTier.TIER_1.value: Decimal("0.05"),
        DataQualityTier.TIER_2.value: Decimal("0.10"),
        DataQualityTier.TIER_3.value: Decimal("0.15"),
    },
    CalculationMethod.AVERAGE_DATA.value: {
        DataQualityTier.TIER_1.value: Decimal("0.20"),
        DataQualityTier.TIER_2.value: Decimal("0.30"),
        DataQualityTier.TIER_3.value: Decimal("0.40"),
    },
    CalculationMethod.SPEND_BASED.value: {
        DataQualityTier.TIER_1.value: Decimal("0.40"),
        DataQualityTier.TIER_2.value: Decimal("0.50"),
        DataQualityTier.TIER_3.value: Decimal("0.60"),
    },
    CalculationMethod.HYBRID.value: {
        DataQualityTier.TIER_1.value: Decimal("0.10"),
        DataQualityTier.TIER_2.value: Decimal("0.18"),
        DataQualityTier.TIER_3.value: Decimal("0.25"),
    },
}


# ==============================================================================
# PYDANTIC INPUT MODELS
# ==============================================================================


class BuildingAssetInput(BaseModel):
    """
    Input for building-level emissions calculation (Cat 13 lessor perspective).

    Captures building characteristics, tenant occupancy, metered energy (if
    available), and vacancy information unique to downstream leased assets.

    Example:
        >>> building = BuildingAssetInput(
        ...     building_type=BuildingType.OFFICE,
        ...     floor_area_sqm=Decimal("5000"),
        ...     climate_zone=ClimateZone.TEMPERATE,
        ...     country_code="US",
        ...     lease_share=Decimal("1.0"),
        ...     occupancy_status=OccupancyStatus.OCCUPIED,
        ...     tenant_count=3
        ... )
    """

    building_type: BuildingType = Field(
        ..., description="Building use type for EUI benchmark selection"
    )
    floor_area_sqm: Decimal = Field(
        ..., gt=0,
        description="Total leasable floor area in square metres"
    )
    climate_zone: Optional[ClimateZone] = Field(
        default=None,
        description="Climate zone (auto-resolved from country_code if omitted)"
    )
    country_code: str = Field(
        default="US",
        description="ISO 3166-1 alpha-2 country code for grid EF and climate zone"
    )
    lease_share: Decimal = Field(
        default=Decimal("1.0"), ge=0, le=1,
        description="Fraction of building owned/leased (0-1, for equity share)"
    )
    metered_energy_kwh: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Total metered energy in kWh (for asset-specific method)"
    )
    occupancy_status: OccupancyStatus = Field(
        default=OccupancyStatus.OCCUPIED,
        description="Current occupancy status of the leased building"
    )
    vacancy_months: int = Field(
        default=0, ge=0, le=12,
        description="Number of months vacant in the reporting period"
    )
    tenant_count: int = Field(
        default=1, ge=0,
        description="Number of tenants occupying the building (0 if vacant)"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy isolation"
    )

    model_config = ConfigDict(frozen=True)

    @validator("country_code")
    def validate_country_code(cls, v: str) -> str:
        """Validate and uppercase country code."""
        return v.upper()


class VehicleAssetInput(BaseModel):
    """
    Input for vehicle fleet emissions calculation (Cat 13 lessor perspective).

    Captures vehicle type, fuel type, distance/fuel data, and fleet count
    for vehicles owned by the reporter and leased to third parties.

    Example:
        >>> vehicle = VehicleAssetInput(
        ...     vehicle_type=VehicleType.MEDIUM_CAR,
        ...     fuel_type=FuelType.DIESEL,
        ...     distance_km=Decimal("25000"),
        ...     lease_share=Decimal("1.0"),
        ...     fleet_count=10
        ... )
    """

    vehicle_type: VehicleType = Field(
        ..., description="Vehicle type / size category"
    )
    fuel_type: FuelType = Field(
        ..., description="Fuel / powertrain type"
    )
    distance_km: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Annual distance driven in km (distance-based method)"
    )
    fuel_consumed_l: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Annual fuel consumed in litres (fuel-based method)"
    )
    lease_share: Decimal = Field(
        default=Decimal("1.0"), ge=0, le=1,
        description="Fraction of vehicle owned/leased (0-1)"
    )
    fleet_count: int = Field(
        default=1, ge=1,
        description="Number of identical vehicles in this fleet segment"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy isolation"
    )

    model_config = ConfigDict(frozen=True)


class EquipmentAssetInput(BaseModel):
    """
    Input for equipment emissions calculation (Cat 13 lessor perspective).

    Captures equipment type, operating hours, load factor, and fuel type
    for industrial/construction equipment leased to third parties.

    Example:
        >>> equipment = EquipmentAssetInput(
        ...     equipment_type=EquipmentType.CONSTRUCTION,
        ...     fuel_type=FuelType.DIESEL,
        ...     operating_hours=Decimal("1500"),
        ...     load_factor=Decimal("0.50"),
        ...     lease_share=Decimal("1.0")
        ... )
    """

    equipment_type: EquipmentType = Field(
        ..., description="Equipment type / category"
    )
    fuel_type: FuelType = Field(
        default=FuelType.DIESEL,
        description="Fuel type consumed by the equipment"
    )
    operating_hours: Decimal = Field(
        ..., gt=0,
        description="Annual operating hours"
    )
    load_factor: Optional[Decimal] = Field(
        default=None, ge=0, le=1,
        description="Load factor override (fraction of rated power, 0-1)"
    )
    rated_power_kw: Optional[Decimal] = Field(
        default=None, gt=0,
        description="Rated power override in kW (uses default if omitted)"
    )
    lease_share: Decimal = Field(
        default=Decimal("1.0"), ge=0, le=1,
        description="Fraction of equipment owned/leased (0-1)"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy isolation"
    )

    model_config = ConfigDict(frozen=True)


class ITAssetInput(BaseModel):
    """
    Input for IT asset emissions calculation (Cat 13 lessor perspective).

    Captures IT asset type, power draw, PUE, and quantity for IT
    infrastructure leased to third parties (e.g., colocation, managed hosting).

    Example:
        >>> it_asset = ITAssetInput(
        ...     it_asset_type=ITAssetType.SERVER,
        ...     quantity=50,
        ...     lease_share=Decimal("1.0")
        ... )
    """

    it_asset_type: ITAssetType = Field(
        ..., description="IT asset type / category"
    )
    power_kw: Optional[Decimal] = Field(
        default=None, gt=0,
        description="Power draw per unit in kW (uses default if omitted)"
    )
    pue: Optional[Decimal] = Field(
        default=None, ge=1,
        description="Power Usage Effectiveness override (uses default if omitted)"
    )
    hours_per_year: Optional[Decimal] = Field(
        default=None, gt=0,
        description="Annual operating hours override (uses default if omitted)"
    )
    quantity: int = Field(
        default=1, ge=1,
        description="Number of identical IT assets"
    )
    lease_share: Decimal = Field(
        default=Decimal("1.0"), ge=0, le=1,
        description="Fraction of asset owned/leased (0-1)"
    )
    country_code: str = Field(
        default="US",
        description="Country code for grid emission factor lookup"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy isolation"
    )

    model_config = ConfigDict(frozen=True)

    @validator("country_code")
    def validate_country_code(cls, v: str) -> str:
        """Validate and uppercase country code."""
        return v.upper()


class LeaseInfo(BaseModel):
    """
    Lease metadata for consolidation boundary and reporting.

    Determines whether an asset falls within the reporter's GHG boundary
    based on the consolidation approach and lease classification.

    Example:
        >>> lease = LeaseInfo(
        ...     lease_type=LeaseType.OPERATING,
        ...     consolidation_approach=ConsolidationApproach.FINANCIAL_CONTROL,
        ...     lease_term_years=5,
        ...     lease_share=Decimal("1.0")
        ... )
    """

    lease_type: LeaseType = Field(
        ..., description="Lease classification (operating, finance, sale-leaseback)"
    )
    consolidation_approach: ConsolidationApproach = Field(
        default=ConsolidationApproach.OPERATIONAL_CONTROL,
        description="GHG Protocol organisational boundary approach"
    )
    lease_term_years: Optional[int] = Field(
        default=None, gt=0,
        description="Lease term in years"
    )
    start_date: Optional[date] = Field(
        default=None,
        description="Lease start date (ISO 8601)"
    )
    end_date: Optional[date] = Field(
        default=None,
        description="Lease end date (ISO 8601)"
    )
    lease_share: Decimal = Field(
        default=Decimal("1.0"), ge=0, le=1,
        description="Equity share fraction (used with equity_share approach)"
    )

    model_config = ConfigDict(frozen=True)


class AllocationInfo(BaseModel):
    """
    Tenant allocation parameters for multi-tenant buildings (Cat 13 specific).

    Determines how building-level emissions are split across tenants
    and common areas in the lessor's downstream leased asset portfolio.

    Example:
        >>> allocation = AllocationInfo(
        ...     method=AllocationMethod.FLOOR_AREA,
        ...     tenant_share=Decimal("0.40"),
        ...     common_area_share=Decimal("0.10")
        ... )
    """

    method: AllocationMethod = Field(
        default=AllocationMethod.FLOOR_AREA,
        description="Allocation method for multi-tenant buildings"
    )
    tenant_share: Optional[Decimal] = Field(
        default=None, ge=0, le=1,
        description="Tenant's share of total building emissions (0-1)"
    )
    common_area_share: Optional[Decimal] = Field(
        default=None, ge=0, le=1,
        description="Common area share allocated to this tenant (0-1)"
    )
    allocation_basis_value: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Numeric basis value (m2, headcount, revenue, FTE)"
    )

    model_config = ConfigDict(frozen=True)


# ==============================================================================
# PYDANTIC RESULT MODELS
# ==============================================================================


class AssetCalculationResult(BaseModel):
    """
    Result from a single asset emissions calculation.

    Contains the total CO2e, energy consumption, DQI, uncertainty,
    and provenance hash for audit trail.
    """

    asset_id: str = Field(
        ..., description="Unique asset identifier"
    )
    category: AssetCategory = Field(
        ..., description="Asset category (building, vehicle, equipment, IT)"
    )
    method: CalculationMethod = Field(
        ..., description="Calculation method used"
    )
    total_co2e: Decimal = Field(
        ..., description="Total CO2e emissions (kgCO2e)"
    )
    scope1_co2e: Decimal = Field(
        default=Decimal("0"), description="Scope 1 component (kgCO2e, on-site combustion)"
    )
    scope2_co2e: Decimal = Field(
        default=Decimal("0"), description="Scope 2 component (kgCO2e, purchased electricity)"
    )
    energy_consumption_kwh: Decimal = Field(
        default=Decimal("0"), description="Total energy consumption (kWh)"
    )
    dqi_score: Optional[Decimal] = Field(
        default=None, description="Data quality indicator score (1-5)"
    )
    uncertainty_pct: Optional[Decimal] = Field(
        default=None, description="Uncertainty half-width as fraction (e.g., 0.15 = 15%)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(frozen=True)


class PortfolioResult(BaseModel):
    """
    Aggregated portfolio-level result across all downstream leased assets.

    Provides total emissions plus breakdowns by category, method, building
    type, vehicle type, and weighted DQI.
    """

    total_co2e: Decimal = Field(
        ..., description="Total portfolio CO2e (kgCO2e)"
    )
    asset_count: int = Field(
        ..., description="Total number of assets in the portfolio"
    )
    by_category: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="CO2e breakdown by asset category"
    )
    by_method: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="CO2e breakdown by calculation method"
    )
    by_building_type: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="CO2e breakdown by building type (buildings only)"
    )
    by_vehicle_type: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="CO2e breakdown by vehicle type (vehicles only)"
    )
    weighted_dqi: Optional[Decimal] = Field(
        default=None,
        description="Emissions-weighted average DQI score (1-5)"
    )

    model_config = ConfigDict(frozen=True)


class AvoidedEmissions(BaseModel):
    """
    Avoided emissions from green building and energy efficiency measures.

    Captures credits from energy efficiency improvements, renewable energy
    procurement, and green lease clauses in downstream leased assets.
    """

    energy_efficiency_credit: Decimal = Field(
        default=Decimal("0"),
        description="CO2e avoided through energy efficiency (kgCO2e)"
    )
    renewable_energy_credit: Decimal = Field(
        default=Decimal("0"),
        description="CO2e avoided through on-site or procured renewables (kgCO2e)"
    )
    green_lease_benefit: Decimal = Field(
        default=Decimal("0"),
        description="CO2e avoided through green lease clauses (kgCO2e)"
    )

    model_config = ConfigDict(frozen=True)


class ComplianceResult(BaseModel):
    """
    Result from compliance check against a specific framework.

    Captures status, findings, score, and recommendations for
    Cat 13 downstream leased asset disclosures.
    """

    framework: ComplianceFramework = Field(
        ..., description="Framework checked"
    )
    status: ComplianceStatus = Field(
        ..., description="Compliance status"
    )
    findings: List[str] = Field(
        default_factory=list,
        description="Specific findings (gaps, issues, observations)"
    )
    score: Decimal = Field(
        default=Decimal("0"),
        description="Compliance score (0-100)"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improving compliance"
    )

    model_config = ConfigDict(frozen=True)


class ProvenanceRecord(BaseModel):
    """
    Single record in the provenance chain.

    Each pipeline stage produces a provenance record linking input
    and output hashes into a tamper-evident chain.
    """

    chain_id: str = Field(
        ..., description="Unique chain identifier"
    )
    stage: str = Field(
        ..., description="Pipeline stage name"
    )
    input_hash: str = Field(
        ..., description="SHA-256 hash of stage input"
    )
    output_hash: str = Field(
        ..., description="SHA-256 hash of stage output"
    )
    timestamp: str = Field(
        ..., description="ISO 8601 timestamp of stage execution"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Stage-specific metadata"
    )

    model_config = ConfigDict(frozen=True)


class DataQualityScore(BaseModel):
    """
    Composite data quality score across five dimensions.

    Dimensions are weighted and combined into an overall score
    used for DQI reporting and uncertainty range selection.
    """

    temporal: Decimal = Field(
        ..., ge=1, le=5, description="Temporal dimension score (1-5)"
    )
    geographical: Decimal = Field(
        ..., ge=1, le=5, description="Geographical dimension score (1-5)"
    )
    technological: Decimal = Field(
        ..., ge=1, le=5, description="Technological dimension score (1-5)"
    )
    completeness: Decimal = Field(
        ..., ge=1, le=5, description="Completeness dimension score (1-5)"
    )
    reliability: Decimal = Field(
        ..., ge=1, le=5, description="Reliability dimension score (1-5)"
    )
    overall: Decimal = Field(
        ..., ge=1, le=5, description="Weighted composite score (1-5)"
    )

    model_config = ConfigDict(frozen=True)


class UncertaintyResult(BaseModel):
    """
    Uncertainty quantification result.

    Provides confidence interval bounds around the emissions estimate,
    using one of the supported uncertainty methods.
    """

    method: UncertaintyMethod = Field(
        ..., description="Uncertainty quantification method used"
    )
    mean: Decimal = Field(
        ..., description="Mean emissions estimate (kgCO2e)"
    )
    lower_bound: Decimal = Field(
        ..., description="Lower bound of confidence interval (kgCO2e)"
    )
    upper_bound: Decimal = Field(
        ..., description="Upper bound of confidence interval (kgCO2e)"
    )
    confidence: Decimal = Field(
        default=Decimal("0.95"),
        description="Confidence level (e.g., 0.95 = 95%)"
    )

    model_config = ConfigDict(frozen=True)


class AggregationResult(BaseModel):
    """
    Aggregated emissions result for a reporting period.

    Provides total emissions with breakdowns by asset category,
    building type, and geographic region.
    """

    period: str = Field(
        ..., description="Reporting period (e.g., '2024', '2024-Q3')"
    )
    total_co2e: Decimal = Field(
        ..., description="Total CO2e for the period (kgCO2e)"
    )
    by_category: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="CO2e breakdown by asset category"
    )
    by_building_type: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="CO2e breakdown by building type"
    )
    by_region: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="CO2e breakdown by country / region"
    )

    model_config = ConfigDict(frozen=True)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def get_building_eui(
    building_type: str,
    climate_zone: str,
) -> Optional[Decimal]:
    """
    Look up building EUI benchmark (kWh/m2/year) by type and climate zone.

    Args:
        building_type: Building type string (e.g., "office").
        climate_zone: Climate zone string (e.g., "temperate").

    Returns:
        EUI in kWh/m2/year, or None if not found.

    Example:
        >>> get_building_eui("office", "temperate")
        Decimal('180')
    """
    bt_data = BUILDING_EUI_BENCHMARKS.get(building_type)
    if bt_data is None:
        return None
    return bt_data.get(climate_zone)


def get_vehicle_ef(
    vehicle_type: str,
    fuel_type: str,
) -> Optional[Decimal]:
    """
    Look up vehicle emission factor (kgCO2e/km) by vehicle and fuel type.

    Args:
        vehicle_type: Vehicle type string (e.g., "medium_car").
        fuel_type: Fuel type string (e.g., "diesel").

    Returns:
        Emission factor in kgCO2e/km, or None if not found.

    Example:
        >>> get_vehicle_ef("medium_car", "diesel")
        Decimal('0.16680')
    """
    vt_data = VEHICLE_EMISSION_FACTORS.get(vehicle_type)
    if vt_data is None:
        return None
    return vt_data.get(fuel_type)


def get_equipment_fuel(
    equipment_type: str,
) -> Optional[Dict[str, Decimal]]:
    """
    Look up equipment fuel consumption parameters by type.

    Args:
        equipment_type: Equipment type string (e.g., "construction").

    Returns:
        Dict with rated_power_kw, fuel_consumption_lph, load_factor; or None.

    Example:
        >>> params = get_equipment_fuel("construction")
        >>> params["fuel_consumption_lph"]
        Decimal('50.0')
    """
    return EQUIPMENT_FUEL_CONSUMPTION.get(equipment_type)


def get_it_power(
    it_asset_type: str,
) -> Optional[Dict[str, Decimal]]:
    """
    Look up IT asset power ratings by type.

    Args:
        it_asset_type: IT asset type string (e.g., "server").

    Returns:
        Dict with power_kw, default_pue, hours_per_year; or None.

    Example:
        >>> params = get_it_power("server")
        >>> params["power_kw"]
        Decimal('0.500')
    """
    return IT_ASSET_POWER_RATINGS.get(it_asset_type)


def get_grid_ef(
    region_code: str,
) -> Decimal:
    """
    Look up grid emission factor (kgCO2e/kWh) by country or eGRID subregion.

    Falls back to GLOBAL average if region code is not found.

    Args:
        region_code: ISO country code or eGRID subregion code.

    Returns:
        Grid emission factor in kgCO2e/kWh.

    Example:
        >>> get_grid_ef("US")
        Decimal('0.417')
        >>> get_grid_ef("CAMX")
        Decimal('0.225')
        >>> get_grid_ef("ZZ")
        Decimal('0.436')
    """
    code = region_code.upper()
    return GRID_EMISSION_FACTORS.get(code, GRID_EMISSION_FACTORS["GLOBAL"])


def get_fuel_ef(
    fuel_type: str,
) -> Optional[Dict[str, Decimal]]:
    """
    Look up fuel emission factors (kgCO2e/litre) by fuel type.

    Args:
        fuel_type: Fuel type string (e.g., "diesel").

    Returns:
        Dict with co2e_per_liter and wtt_factor; or None.

    Example:
        >>> factors = get_fuel_ef("diesel")
        >>> factors["co2e_per_liter"]
        Decimal('2.70370')
    """
    return FUEL_EMISSION_FACTORS.get(fuel_type)


def get_eeio_factor(
    naics_code: str,
) -> Optional[Decimal]:
    """
    Look up EEIO spend-based emission factor (kgCO2e/USD) by NAICS code.

    Args:
        naics_code: NAICS industry code string.

    Returns:
        EEIO factor in kgCO2e/USD, or None if not found.

    Example:
        >>> get_eeio_factor("531110")
        Decimal('0.15')
    """
    entry = EEIO_SPEND_FACTORS.get(naics_code)
    if entry is not None:
        return entry["ef"]
    return None


def get_allocation_default(
    building_type: str,
) -> Optional[Dict[str, Decimal]]:
    """
    Look up default allocation percentages by building type.

    Args:
        building_type: Building type string (e.g., "office").

    Returns:
        Dict with tenant_share and common_area_share; or None.

    Example:
        >>> defaults = get_allocation_default("office")
        >>> defaults["tenant_share"]
        Decimal('0.85')
    """
    return ALLOCATION_DEFAULTS.get(building_type)


def get_vacancy_base_load(
    building_type: str,
) -> Decimal:
    """
    Look up vacancy base-load fraction by building type.

    When a building is vacant, this fraction of normal energy use
    continues for HVAC set-back, security, fire systems, etc.
    Falls back to 0.25 if building type is not found.

    Args:
        building_type: Building type string (e.g., "office").

    Returns:
        Base-load fraction (0-1).

    Example:
        >>> get_vacancy_base_load("data_center")
        Decimal('0.60')
    """
    return VACANCY_BASE_LOAD.get(building_type, Decimal("0.25"))


def get_refrigerant_gwp(
    refrigerant: str,
) -> Optional[Decimal]:
    """
    Look up refrigerant GWP (100-year, IPCC AR6) by refrigerant name.

    Args:
        refrigerant: Refrigerant designation (e.g., "R-410A").

    Returns:
        GWP value, or None if not found.

    Example:
        >>> get_refrigerant_gwp("R-410A")
        Decimal('2088')
    """
    return REFRIGERANT_GWPS.get(refrigerant)


def get_country_climate_zone(
    country_code: str,
) -> str:
    """
    Look up the dominant climate zone for a country.

    Falls back to TEMPERATE if the country code is not mapped.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.

    Returns:
        Climate zone string value.

    Example:
        >>> get_country_climate_zone("IN")
        'tropical'
        >>> get_country_climate_zone("CA")
        'continental'
    """
    code = country_code.upper()
    return COUNTRY_CLIMATE_ZONES.get(code, ClimateZone.TEMPERATE.value)


def get_dc_rule(
    rule_id: str,
) -> Optional[Dict[str, str]]:
    """
    Look up a double-counting prevention rule by ID.

    Args:
        rule_id: Rule identifier (e.g., "DC-DLA-001").

    Returns:
        Dict with description and affected_categories; or None.

    Example:
        >>> rule = get_dc_rule("DC-DLA-003")
        >>> "Cat 8" in rule["affected_categories"]
        True
    """
    return DC_RULES.get(rule_id)


def get_framework_rules(
    framework: str,
) -> Optional[List[str]]:
    """
    Look up required disclosures for a compliance framework.

    Args:
        framework: Compliance framework string value.

    Returns:
        List of required disclosure items; or None.

    Example:
        >>> rules = get_framework_rules("ghg_protocol")
        >>> "total_co2e_cat13" in rules
        True
    """
    return COMPLIANCE_FRAMEWORK_RULES.get(framework)


def get_dqi_score(
    dimension: str,
    tier: str,
) -> Optional[Decimal]:
    """
    Look up DQI score for a specific dimension and data quality tier.

    Args:
        dimension: DQI dimension string (e.g., "temporal").
        tier: Data quality tier string (e.g., "tier_1").

    Returns:
        DQI score (1-5); or None if not found.

    Example:
        >>> get_dqi_score("temporal", "tier_1")
        Decimal('5')
    """
    dim_data = DQI_SCORING.get(dimension)
    if dim_data is None:
        return None
    return dim_data.get(tier)


def get_uncertainty_range(
    method: str,
    tier: str,
) -> Optional[Decimal]:
    """
    Look up uncertainty half-width for a calculation method and tier.

    Args:
        method: Calculation method string (e.g., "asset_specific").
        tier: Data quality tier string (e.g., "tier_1").

    Returns:
        Uncertainty half-width as fraction (e.g., 0.05 = +/- 5%); or None.

    Example:
        >>> get_uncertainty_range("asset_specific", "tier_1")
        Decimal('0.05')
    """
    method_data = UNCERTAINTY_RANGES.get(method)
    if method_data is None:
        return None
    return method_data.get(tier)


def calculate_provenance_hash(*inputs: Any) -> str:
    """
    Calculate SHA-256 provenance hash from variable inputs.

    Supports Pydantic models (serialised to sorted JSON), Decimal values,
    and any other stringifiable objects. Used throughout the pipeline
    for building tamper-evident provenance chains.

    Args:
        *inputs: Variable number of input objects to hash.

    Returns:
        Hexadecimal SHA-256 hash string (64 characters).

    Example:
        >>> h = calculate_provenance_hash("building_001", Decimal("1234.56"))
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


def validate_consolidation_approach(
    lease_type: str,
    consolidation_approach: str,
) -> bool:
    """
    Validate that a lease type is appropriately reported under the
    given consolidation approach for Cat 13 downstream leased assets.

    Per GHG Protocol:
    - Under operational control: include operating leases where lessor
      does NOT have operational control (tenant operates the asset).
    - Under financial control: include assets where lessor has financial
      control (owns asset on balance sheet).
    - Under equity share: include proportional to equity ownership.
    - Finance leases may shift reporting to lessee under some approaches.

    Args:
        lease_type: Lease type string value.
        consolidation_approach: Consolidation approach string value.

    Returns:
        True if the combination is valid for Cat 13 reporting.

    Example:
        >>> validate_consolidation_approach("operating", "financial_control")
        True
        >>> validate_consolidation_approach("finance", "operational_control")
        False
    """
    # Under operational control, finance leases transfer operational control
    # to the lessee, so they should NOT be in the lessor's Cat 13 boundary.
    if (consolidation_approach == ConsolidationApproach.OPERATIONAL_CONTROL.value
            and lease_type == LeaseType.FINANCE.value):
        return False

    # All other combinations are valid for Cat 13 downstream reporting.
    return True


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Metadata
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",

    # Enumerations
    "AssetCategory",
    "BuildingType",
    "VehicleType",
    "FuelType",
    "EquipmentType",
    "ITAssetType",
    "ClimateZone",
    "CalculationMethod",
    "AllocationMethod",
    "LeaseType",
    "ConsolidationApproach",
    "EFSource",
    "DataQualityTier",
    "DQIDimension",
    "ComplianceFramework",
    "ComplianceStatus",
    "PipelineStage",
    "UncertaintyMethod",
    "BatchStatus",
    "GWPSource",
    "EnergyType",
    "OccupancyStatus",

    # Constant tables
    "BUILDING_EUI_BENCHMARKS",
    "VEHICLE_EMISSION_FACTORS",
    "EQUIPMENT_FUEL_CONSUMPTION",
    "IT_ASSET_POWER_RATINGS",
    "GRID_EMISSION_FACTORS",
    "FUEL_EMISSION_FACTORS",
    "EEIO_SPEND_FACTORS",
    "ALLOCATION_DEFAULTS",
    "VACANCY_BASE_LOAD",
    "REFRIGERANT_GWPS",
    "COUNTRY_CLIMATE_ZONES",
    "DC_RULES",
    "COMPLIANCE_FRAMEWORK_RULES",
    "DQI_SCORING",
    "UNCERTAINTY_RANGES",

    # Input models
    "BuildingAssetInput",
    "VehicleAssetInput",
    "EquipmentAssetInput",
    "ITAssetInput",
    "LeaseInfo",
    "AllocationInfo",

    # Result models
    "AssetCalculationResult",
    "PortfolioResult",
    "AvoidedEmissions",
    "ComplianceResult",
    "ProvenanceRecord",
    "DataQualityScore",
    "UncertaintyResult",
    "AggregationResult",

    # Helper functions
    "get_building_eui",
    "get_vehicle_ef",
    "get_equipment_fuel",
    "get_it_power",
    "get_grid_ef",
    "get_fuel_ef",
    "get_eeio_factor",
    "get_allocation_default",
    "get_vacancy_base_load",
    "get_refrigerant_gwp",
    "get_country_climate_zone",
    "get_dc_rule",
    "get_framework_rules",
    "get_dqi_score",
    "get_uncertainty_range",
    "calculate_provenance_hash",
    "validate_consolidation_approach",
]
