"""
Employee Commuting Agent Models (AGENT-MRV-020)

This module provides comprehensive data models for GHG Protocol Scope 3 Category 7
(Employee Commuting) emissions calculations.

Supports:
- 3 calculation methods (employee-specific, average-data, spend-based)
- 14 commute modes (SOV, carpool, vanpool, bus, metro, light rail, commuter rail,
  ferry, motorcycle, e-bike, e-scooter, cycling, walking, telework)
- 12 vehicle types with per-vkm and per-pkm factors (DEFRA 2024)
- 5 fuel types with WTT (well-to-tank) factors
- 6 transit types with per-pkm factors
- Micro-mobility emission factors (e-bike, e-scooter)
- Telework home-office energy emissions (laptop, heating, lighting)
- Grid emission factors for 11 regions
- Working days defaults for 11 regions (holidays, PTO, sick)
- Average commute distances for 10 countries + global
- Default mode share distributions (US, UK, EU)
- EEIO spend-based factors (7 NAICS codes)
- CPI deflation and multi-currency conversion (12 currencies)
- Data quality indicators (DQI) with 5-dimension scoring
- Uncertainty quantification (Monte Carlo, analytical, IPCC Tier 2)
- Compliance checking for 7 frameworks (GHG Protocol, ISO 14064, CSRD, CDP,
  SBTi, SB 253, GRI)
- SHA-256 provenance chain with 10-stage pipeline
- Hot-spot analysis for mode and distance-band optimization
- Survey extrapolation (full census, stratified sample, random sample, convenience)
- Seasonal adjustment for telework energy consumption
- 4-band distance classification (short, medium, long, very long)

All numeric fields use Decimal for precision in regulatory calculations.
All models are frozen (immutable) for audit trail integrity.

Example:
    >>> from greenlang.agents.mrv.employee_commuting.models import CommuteInput, CommuteMode
    >>> commute = CommuteInput(
    ...     mode=CommuteMode.SOV,
    ...     vehicle_type=VehicleType.CAR_MEDIUM_PETROL,
    ...     one_way_distance_km=Decimal("15.0"),
    ...     commute_days_per_week=5
    ... )
    >>> from greenlang.agents.mrv.employee_commuting.models import TeleworkInput, TeleworkFrequency
    >>> telework = TeleworkInput(
    ...     frequency=TeleworkFrequency.HYBRID_3,
    ...     region=RegionCode.US
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

AGENT_ID: str = "GL-MRV-S3-007"
AGENT_COMPONENT: str = "AGENT-MRV-020"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_ec_"

# ==============================================================================
# ENUMERATIONS (25 total)
# ==============================================================================


class CalculationMethod(str, Enum):
    """Calculation method for employee commuting emissions per GHG Protocol."""

    EMPLOYEE_SPECIFIC = "employee_specific"  # Individual employee survey data
    AVERAGE_DATA = "average_data"  # Average commute distance and mode share
    SPEND_BASED = "spend_based"  # Commuting benefits spend x EEIO factor


class CommuteMode(str, Enum):
    """Commute modes for employee commuting (14 modes)."""

    SOV = "sov"  # Single-occupancy vehicle (driving alone)
    CARPOOL = "carpool"  # Carpool (2+ occupants sharing ride)
    VANPOOL = "vanpool"  # Vanpool (employer-organized, 7-15 passengers)
    BUS = "bus"  # Local bus / city bus
    METRO = "metro"  # Metro / underground / subway
    LIGHT_RAIL = "light_rail"  # Light rail / tram / streetcar
    COMMUTER_RAIL = "commuter_rail"  # Commuter rail / regional train
    FERRY = "ferry"  # Ferry / water taxi
    MOTORCYCLE = "motorcycle"  # Motorcycle / scooter
    E_BIKE = "e_bike"  # Electric bicycle
    E_SCOOTER = "e_scooter"  # Electric kick-scooter
    CYCLING = "cycling"  # Pedal bicycle (zero emissions)
    WALKING = "walking"  # Walking (zero emissions)
    TELEWORK = "telework"  # Remote work / work from home


class VehicleType(str, Enum):
    """Vehicle types for SOV and carpool commuting (12 types)."""

    CAR_AVERAGE = "car_average"  # Average car (unknown fuel type)
    CAR_SMALL_PETROL = "car_small_petrol"  # Small petrol car (< 1.4L)
    CAR_MEDIUM_PETROL = "car_medium_petrol"  # Medium petrol car (1.4-2.0L)
    CAR_LARGE_PETROL = "car_large_petrol"  # Large petrol car (> 2.0L)
    CAR_SMALL_DIESEL = "car_small_diesel"  # Small diesel car (< 1.7L)
    CAR_MEDIUM_DIESEL = "car_medium_diesel"  # Medium diesel car (1.7-2.0L)
    CAR_LARGE_DIESEL = "car_large_diesel"  # Large diesel car (> 2.0L)
    HYBRID = "hybrid"  # Hybrid electric vehicle (HEV)
    PLUGIN_HYBRID = "plugin_hybrid"  # Plug-in hybrid electric vehicle (PHEV)
    BEV = "bev"  # Battery electric vehicle (BEV)
    VAN_AVERAGE = "van_average"  # Van / minibus (vanpool default)
    MOTORCYCLE = "motorcycle"  # Motorcycle / scooter


class FuelType(str, Enum):
    """Fuel types for fuel-based commute calculations (5 types)."""

    PETROL = "petrol"  # Gasoline / petrol
    DIESEL = "diesel"  # Diesel
    LPG = "lpg"  # Liquefied petroleum gas
    E10 = "e10"  # Ethanol blend (10% ethanol, 90% petrol)
    B7 = "b7"  # Biodiesel blend (7% FAME, 93% diesel)


class TransitType(str, Enum):
    """Public transit types with distinct emission profiles (6 types)."""

    BUS_LOCAL = "bus_local"  # Local bus / city bus
    BUS_COACH = "bus_coach"  # Long-distance coach / intercity bus
    METRO = "metro"  # Metro / underground / subway
    LIGHT_RAIL = "light_rail"  # Light rail / tram / streetcar
    COMMUTER_RAIL = "commuter_rail"  # Commuter rail / regional train
    FERRY = "ferry"  # Ferry / water taxi


class TeleworkFrequency(str, Enum):
    """Telework frequency patterns (6 patterns)."""

    FULL_REMOTE = "full_remote"  # 5 days/week remote (100% telework)
    HYBRID_4 = "hybrid_4"  # 4 days/week remote, 1 day office
    HYBRID_3 = "hybrid_3"  # 3 days/week remote, 2 days office
    HYBRID_2 = "hybrid_2"  # 2 days/week remote, 3 days office
    HYBRID_1 = "hybrid_1"  # 1 day/week remote, 4 days office
    OFFICE_FULL = "office_full"  # 5 days/week in office (0% telework)


class WorkSchedule(str, Enum):
    """Work schedule types affecting annual commute days (4 schedules)."""

    FULL_TIME = "full_time"  # Full-time (100% of working days)
    PART_TIME_80 = "part_time_80"  # Part-time 80% (4 days/week)
    PART_TIME_60 = "part_time_60"  # Part-time 60% (3 days/week)
    PART_TIME_50 = "part_time_50"  # Part-time 50% (2.5 days/week)


class EFSource(str, Enum):
    """Emission factor data source (7 sources)."""

    EMPLOYEE = "employee"  # Employee-reported / survey data
    DEFRA = "defra"  # UK DEFRA/DESNZ conversion factors 2024
    EPA = "epa"  # US EPA emission factors
    IEA = "iea"  # IEA energy statistics (grid factors)
    CENSUS = "census"  # National census commute data
    EEIO = "eeio"  # Environmentally Extended Input-Output
    CUSTOM = "custom"  # Custom / organization-specific factors


class ComplianceFramework(str, Enum):
    """Regulatory/reporting framework for compliance checks (7 frameworks)."""

    GHG_PROTOCOL = "ghg_protocol"  # GHG Protocol Scope 3 Standard
    ISO_14064 = "iso_14064"  # ISO 14064-1:2018
    CSRD_ESRS = "csrd_esrs"  # CSRD ESRS E1 Climate Change
    CDP = "cdp"  # CDP Climate Change Questionnaire
    SBTI = "sbti"  # Science Based Targets initiative
    SB_253 = "sb_253"  # California SB 253 (Climate Corporate Data Accountability Act)
    GRI = "gri"  # GRI 305 Emissions Standard


class DataQualityTier(str, Enum):
    """Data quality tiers affecting uncertainty ranges (3 tiers)."""

    TIER_1 = "tier_1"  # Employee-specific / primary survey data
    TIER_2 = "tier_2"  # Regional / mode-specific secondary data
    TIER_3 = "tier_3"  # Global average / spend-based estimates


class ProvenanceStage(str, Enum):
    """Processing pipeline stages for provenance tracking (10 stages)."""

    VALIDATE = "validate"  # Input validation
    CLASSIFY = "classify"  # Mode and distance band classification
    NORMALIZE = "normalize"  # Unit normalization (currency, distance)
    RESOLVE_EFS = "resolve_efs"  # Emission factor resolution
    CALCULATE_COMMUTE = "calculate_commute"  # Commute emissions calculation
    CALCULATE_TELEWORK = "calculate_telework"  # Telework emissions calculation
    EXTRAPOLATE = "extrapolate"  # Survey sample extrapolation
    COMPLIANCE = "compliance"  # Compliance checks
    AGGREGATE = "aggregate"  # Aggregation by mode, period, department
    SEAL = "seal"  # Provenance chain sealing


class UncertaintyMethod(str, Enum):
    """Uncertainty quantification method (3 methods)."""

    MONTE_CARLO = "monte_carlo"  # Monte Carlo simulation
    ANALYTICAL = "analytical"  # Analytical error propagation
    IPCC_TIER_2 = "ipcc_tier_2"  # IPCC Tier 2 default ranges


class DQIDimension(str, Enum):
    """Data Quality Indicator dimensions per GHG Protocol (5 dimensions)."""

    REPRESENTATIVENESS = "representativeness"  # How well data represents the activity
    COMPLETENESS = "completeness"  # Fraction of data coverage
    TEMPORAL = "temporal"  # Temporal correlation to reporting year
    GEOGRAPHICAL = "geographical"  # Geographical correlation to activity
    TECHNOLOGICAL = "technological"  # Technological correlation to activity


class DQIScore(str, Enum):
    """Data Quality Indicator scores (5-point scale, 5 = best)."""

    VERY_HIGH = "very_high"  # 5 - Primary data, employee-specific survey
    HIGH = "high"  # 4 - Verified secondary data / census
    MEDIUM = "medium"  # 3 - Industry average data
    LOW = "low"  # 2 - Estimated / proxy data
    VERY_LOW = "very_low"  # 1 - Spend-based / generic data


class ComplianceStatus(str, Enum):
    """Compliance check result status (3 statuses)."""

    PASS = "pass"  # Fully compliant
    FAIL = "fail"  # Non-compliant
    WARNING = "warning"  # Partially compliant / needs attention


class GWPVersion(str, Enum):
    """IPCC Global Warming Potential assessment report version (4 versions)."""

    AR4 = "ar4"  # Fourth Assessment Report (100-year)
    AR5 = "ar5"  # Fifth Assessment Report (100-year)
    AR6 = "ar6"  # Sixth Assessment Report (100-year)
    AR6_20YR = "ar6_20yr"  # Sixth Assessment Report (20-year)


class EmissionGas(str, Enum):
    """Greenhouse gas types relevant to employee commuting (3 gases)."""

    CO2 = "co2"  # Carbon dioxide
    CH4 = "ch4"  # Methane
    N2O = "n2o"  # Nitrous oxide


class CurrencyCode(str, Enum):
    """ISO 4217 currency codes for spend-based calculations (12 currencies)."""

    USD = "USD"  # US Dollar
    EUR = "EUR"  # Euro
    GBP = "GBP"  # British Pound
    CAD = "CAD"  # Canadian Dollar
    AUD = "AUD"  # Australian Dollar
    JPY = "JPY"  # Japanese Yen
    CNY = "CNY"  # Chinese Yuan
    INR = "INR"  # Indian Rupee
    CHF = "CHF"  # Swiss Franc
    SGD = "SGD"  # Singapore Dollar
    BRL = "BRL"  # Brazilian Real
    ZAR = "ZAR"  # South African Rand


class ExportFormat(str, Enum):
    """Export format for results (4 formats)."""

    JSON = "json"  # JSON format
    CSV = "csv"  # CSV format
    EXCEL = "excel"  # Excel (XLSX) format
    PDF = "pdf"  # PDF report


class BatchStatus(str, Enum):
    """Batch calculation processing status (5 states)."""

    PENDING = "pending"  # Awaiting processing
    PROCESSING = "processing"  # Currently processing
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Processing failed
    PARTIAL = "partial"  # Some records failed


class RegionCode(str, Enum):
    """Region codes for working days and grid factors (11 regions)."""

    US = "US"  # United States
    GB = "GB"  # United Kingdom
    DE = "DE"  # Germany
    FR = "FR"  # France
    JP = "JP"  # Japan
    CA = "CA"  # Canada
    AU = "AU"  # Australia
    IN = "IN"  # India
    CN = "CN"  # China
    BR = "BR"  # Brazil
    GLOBAL = "GLOBAL"  # Global default


class DistanceBand(str, Enum):
    """Commute distance bands for segmentation (4 bands)."""

    SHORT_0_5 = "short_0_5"  # 0-5 km (walking / cycling / micro-mobility)
    MEDIUM_5_15 = "medium_5_15"  # 5-15 km (transit / driving)
    LONG_15_30 = "long_15_30"  # 15-30 km (driving / commuter rail)
    VERY_LONG_30_PLUS = "very_long_30_plus"  # 30+ km (long-distance commute)


class SurveyMethod(str, Enum):
    """Employee survey methodology for data collection (4 methods)."""

    FULL_CENSUS = "full_census"  # Survey all employees
    STRATIFIED_SAMPLE = "stratified_sample"  # Stratified random sample
    RANDOM_SAMPLE = "random_sample"  # Simple random sample
    CONVENIENCE = "convenience"  # Convenience / voluntary sample


class AllocationMethod(str, Enum):
    """Emissions allocation method for multi-entity reporting (5 methods)."""

    EQUAL = "equal"  # Equal allocation across entities
    HEADCOUNT = "headcount"  # Based on employee headcount
    SITE = "site"  # Based on office site / location
    DEPARTMENT = "department"  # Based on department attribution
    COST_CENTER = "cost_center"  # Based on cost center budgets


class SeasonalAdjustment(str, Enum):
    """Seasonal adjustment for telework energy consumption (4 adjustments)."""

    NONE = "none"  # No seasonal adjustment applied
    HEATING_ONLY = "heating_only"  # Adjust for heating season only
    COOLING_ONLY = "cooling_only"  # Adjust for cooling season only
    FULL_SEASONAL = "full_seasonal"  # Full heating + cooling seasonal adjustment


# ==============================================================================
# CONSTANT TABLES (16 total)
# ==============================================================================

# 1. Global Warming Potential values (100-year unless stated)
GWP_VALUES: Dict[GWPVersion, Dict[str, Decimal]] = {
    GWPVersion.AR4: {
        "co2": Decimal("1"),
        "ch4": Decimal("25"),
        "n2o": Decimal("298"),
    },
    GWPVersion.AR5: {
        "co2": Decimal("1"),
        "ch4": Decimal("28"),
        "n2o": Decimal("265"),
    },
    GWPVersion.AR6: {
        "co2": Decimal("1"),
        "ch4": Decimal("27.9"),
        "n2o": Decimal("273"),
    },
    GWPVersion.AR6_20YR: {
        "co2": Decimal("1"),
        "ch4": Decimal("81.2"),
        "n2o": Decimal("273"),
    },
}

# 2. Vehicle emission factors (kgCO2e) - DEFRA 2024
# ef_per_vkm: emissions per vehicle-km
# ef_per_pkm: emissions per passenger-km (vkm / occupancy)
# wtt_per_vkm: well-to-tank per vehicle-km
# occupancy: average occupancy factor
VEHICLE_EMISSION_FACTORS: Dict[VehicleType, Dict[str, Optional[Decimal]]] = {
    VehicleType.CAR_AVERAGE: {
        "ef_per_vkm": Decimal("0.27145"),
        "ef_per_pkm": Decimal("0.17082"),
        "wtt_per_vkm": Decimal("0.03965"),
        "occupancy": Decimal("1.59"),
    },
    VehicleType.CAR_SMALL_PETROL: {
        "ef_per_vkm": Decimal("0.20755"),
        "ef_per_pkm": Decimal("0.13053"),
        "wtt_per_vkm": Decimal("0.03301"),
        "occupancy": Decimal("1.59"),
    },
    VehicleType.CAR_MEDIUM_PETROL: {
        "ef_per_vkm": Decimal("0.25594"),
        "ef_per_pkm": Decimal("0.16106"),
        "wtt_per_vkm": Decimal("0.04074"),
        "occupancy": Decimal("1.59"),
    },
    VehicleType.CAR_LARGE_PETROL: {
        "ef_per_vkm": Decimal("0.35388"),
        "ef_per_pkm": Decimal("0.22258"),
        "wtt_per_vkm": Decimal("0.05631"),
        "occupancy": Decimal("1.59"),
    },
    VehicleType.CAR_SMALL_DIESEL: {
        "ef_per_vkm": Decimal("0.19290"),
        "ef_per_pkm": Decimal("0.12132"),
        "wtt_per_vkm": Decimal("0.02734"),
        "occupancy": Decimal("1.59"),
    },
    VehicleType.CAR_MEDIUM_DIESEL: {
        "ef_per_vkm": Decimal("0.23280"),
        "ef_per_pkm": Decimal("0.14642"),
        "wtt_per_vkm": Decimal("0.03299"),
        "occupancy": Decimal("1.59"),
    },
    VehicleType.CAR_LARGE_DIESEL: {
        "ef_per_vkm": Decimal("0.29610"),
        "ef_per_pkm": Decimal("0.18629"),
        "wtt_per_vkm": Decimal("0.04198"),
        "occupancy": Decimal("1.59"),
    },
    VehicleType.HYBRID: {
        "ef_per_vkm": Decimal("0.17830"),
        "ef_per_pkm": Decimal("0.11214"),
        "wtt_per_vkm": Decimal("0.02838"),
        "occupancy": Decimal("1.59"),
    },
    VehicleType.PLUGIN_HYBRID: {
        "ef_per_vkm": Decimal("0.10250"),
        "ef_per_pkm": Decimal("0.06447"),
        "wtt_per_vkm": Decimal("0.01363"),
        "occupancy": Decimal("1.59"),
    },
    VehicleType.BEV: {
        "ef_per_vkm": Decimal("0.07005"),
        "ef_per_pkm": Decimal("0.04406"),
        "wtt_per_vkm": Decimal("0.01479"),
        "occupancy": Decimal("1.59"),
    },
    VehicleType.VAN_AVERAGE: {
        "ef_per_vkm": Decimal("0.27439"),
        "ef_per_pkm": None,
        "wtt_per_vkm": Decimal("0.06184"),
        "occupancy": None,
    },
    VehicleType.MOTORCYCLE: {
        "ef_per_vkm": Decimal("0.11337"),
        "ef_per_pkm": Decimal("0.11337"),
        "wtt_per_vkm": Decimal("0.02867"),
        "occupancy": Decimal("1.0"),
    },
}

# 3. Fuel emission factors (kgCO2e per litre) - DEFRA 2024
FUEL_EMISSION_FACTORS: Dict[FuelType, Dict[str, Decimal]] = {
    FuelType.PETROL: {
        "ef_per_litre": Decimal("2.31480"),
        "wtt_per_litre": Decimal("0.58549"),
    },
    FuelType.DIESEL: {
        "ef_per_litre": Decimal("2.70370"),
        "wtt_per_litre": Decimal("0.60927"),
    },
    FuelType.LPG: {
        "ef_per_litre": Decimal("1.55370"),
        "wtt_per_litre": Decimal("0.32149"),
    },
    FuelType.E10: {
        "ef_per_litre": Decimal("2.09780"),
        "wtt_per_litre": Decimal("0.52400"),
    },
    FuelType.B7: {
        "ef_per_litre": Decimal("2.53090"),
        "wtt_per_litre": Decimal("0.57030"),
    },
}

# 4. Transit emission factors (kgCO2e per passenger-km) - DEFRA 2024
TRANSIT_EMISSION_FACTORS: Dict[TransitType, Dict[str, Decimal]] = {
    TransitType.BUS_LOCAL: {
        "ef_per_pkm": Decimal("0.10312"),
        "wtt_per_pkm": Decimal("0.01847"),
    },
    TransitType.BUS_COACH: {
        "ef_per_pkm": Decimal("0.02732"),
        "wtt_per_pkm": Decimal("0.00489"),
    },
    TransitType.METRO: {
        "ef_per_pkm": Decimal("0.02781"),
        "wtt_per_pkm": Decimal("0.00586"),
    },
    TransitType.LIGHT_RAIL: {
        "ef_per_pkm": Decimal("0.02904"),
        "wtt_per_pkm": Decimal("0.00612"),
    },
    TransitType.COMMUTER_RAIL: {
        "ef_per_pkm": Decimal("0.10500"),
        "wtt_per_pkm": Decimal("0.01300"),
    },
    TransitType.FERRY: {
        "ef_per_pkm": Decimal("0.01877"),
        "wtt_per_pkm": Decimal("0.00572"),
    },
}

# 5. Micro-mobility emission factors (kgCO2e per passenger-km)
# Accounts for electricity consumption for charging
MICRO_MOBILITY_EFS: Dict[str, Decimal] = {
    "e_bike": Decimal("0.00500"),
    "e_scooter": Decimal("0.00350"),
}

# 6. Grid emission factors (kgCO2e per kWh) by region - IEA 2024
GRID_EMISSION_FACTORS: Dict[RegionCode, Decimal] = {
    RegionCode.GB: Decimal("0.20707"),
    RegionCode.US: Decimal("0.37170"),
    RegionCode.DE: Decimal("0.33800"),
    RegionCode.FR: Decimal("0.05100"),
    RegionCode.JP: Decimal("0.43400"),
    RegionCode.CN: Decimal("0.53700"),
    RegionCode.IN: Decimal("0.70800"),
    RegionCode.CA: Decimal("0.12000"),
    RegionCode.AU: Decimal("0.65600"),
    RegionCode.BR: Decimal("0.07400"),
    RegionCode.GLOBAL: Decimal("0.43600"),
}

# 7. Working days defaults by region
# Accounts for weekends, public holidays, PTO, sick days
# net = 365 - weekends(104) - holidays - pto - sick
WORKING_DAYS_DEFAULTS: Dict[RegionCode, Dict[str, int]] = {
    RegionCode.US: {
        "holidays": 11,
        "pto": 15,
        "sick": 5,
        "net": 225,
    },
    RegionCode.GB: {
        "holidays": 8,
        "pto": 28,
        "sick": 5,
        "net": 212,
    },
    RegionCode.DE: {
        "holidays": 10,
        "pto": 30,
        "sick": 11,
        "net": 200,
    },
    RegionCode.FR: {
        "holidays": 11,
        "pto": 25,
        "sick": 8,
        "net": 209,
    },
    RegionCode.JP: {
        "holidays": 16,
        "pto": 10,
        "sick": 5,
        "net": 219,
    },
    RegionCode.CA: {
        "holidays": 10,
        "pto": 15,
        "sick": 6,
        "net": 220,
    },
    RegionCode.AU: {
        "holidays": 8,
        "pto": 20,
        "sick": 10,
        "net": 218,
    },
    RegionCode.IN: {
        "holidays": 15,
        "pto": 12,
        "sick": 5,
        "net": 233,
    },
    RegionCode.CN: {
        "holidays": 11,
        "pto": 5,
        "sick": 5,
        "net": 240,
    },
    RegionCode.BR: {
        "holidays": 12,
        "pto": 22,
        "sick": 5,
        "net": 217,
    },
    RegionCode.GLOBAL: {
        "holidays": 11,
        "pto": 15,
        "sick": 5,
        "net": 230,
    },
}

# 8. Average commute distances (one-way, km) by country
# Source: NHTS (US), NTS (UK), Eurostat, national census data
AVERAGE_COMMUTE_DISTANCES: Dict[str, Decimal] = {
    "US": Decimal("21.7"),
    "GB": Decimal("14.4"),
    "DE": Decimal("17.0"),
    "FR": Decimal("13.3"),
    "JP": Decimal("19.5"),
    "CA": Decimal("15.1"),
    "AU": Decimal("16.8"),
    "IN": Decimal("10.2"),
    "CN": Decimal("9.8"),
    "BR": Decimal("12.5"),
    "GLOBAL": Decimal("15.0"),
}

# 9. Default mode share distributions (fraction of commuters by mode)
# Source: US Census ACS, UK NTS, Eurostat
DEFAULT_MODE_SHARES: Dict[str, Dict[str, Decimal]] = {
    "US": {
        "sov": Decimal("0.7610"),
        "carpool": Decimal("0.0890"),
        "bus": Decimal("0.0260"),
        "metro": Decimal("0.0200"),
        "commuter_rail": Decimal("0.0100"),
        "light_rail": Decimal("0.0040"),
        "ferry": Decimal("0.0010"),
        "motorcycle": Decimal("0.0020"),
        "cycling": Decimal("0.0060"),
        "walking": Decimal("0.0280"),
        "telework": Decimal("0.0530"),
    },
    "GB": {
        "sov": Decimal("0.5800"),
        "carpool": Decimal("0.0500"),
        "bus": Decimal("0.0780"),
        "metro": Decimal("0.0420"),
        "commuter_rail": Decimal("0.0500"),
        "light_rail": Decimal("0.0150"),
        "ferry": Decimal("0.0020"),
        "motorcycle": Decimal("0.0080"),
        "cycling": Decimal("0.0250"),
        "walking": Decimal("0.1100"),
        "telework": Decimal("0.0400"),
    },
    "EU": {
        "sov": Decimal("0.5200"),
        "carpool": Decimal("0.0600"),
        "bus": Decimal("0.0900"),
        "metro": Decimal("0.0500"),
        "commuter_rail": Decimal("0.0400"),
        "light_rail": Decimal("0.0200"),
        "ferry": Decimal("0.0020"),
        "motorcycle": Decimal("0.0200"),
        "cycling": Decimal("0.0500"),
        "walking": Decimal("0.1100"),
        "telework": Decimal("0.0380"),
    },
}

# 10. Telework home-office energy defaults (kWh per day)
# Source: IEA analysis of home-office energy use
TELEWORK_ENERGY_DEFAULTS: Dict[str, Decimal] = {
    "laptop_monitor": Decimal("0.3"),  # Laptop + external monitor (8 hrs)
    "heating": Decimal("3.5"),  # Additional home heating (kWh/day avg)
    "cooling": Decimal("2.0"),  # Additional home cooling (kWh/day avg)
    "lighting": Decimal("0.2"),  # Additional home lighting (kWh/day)
    "total_typical": Decimal("4.0"),  # Typical total daily consumption
}

# 11. Van / minibus emission factors for vanpool calculations
# ef_per_vkm: emissions per vehicle-km (entire van)
# wtt_per_vkm: well-to-tank per vehicle-km
# default_occupancy: typical vanpool occupancy
VAN_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "van_small": {
        "ef_per_vkm": Decimal("0.22730"),
        "wtt_per_vkm": Decimal("0.05124"),
        "default_occupancy": Decimal("7"),
    },
    "van_medium": {
        "ef_per_vkm": Decimal("0.27439"),
        "wtt_per_vkm": Decimal("0.06184"),
        "default_occupancy": Decimal("10"),
    },
    "van_large": {
        "ef_per_vkm": Decimal("0.31210"),
        "wtt_per_vkm": Decimal("0.07033"),
        "default_occupancy": Decimal("12"),
    },
    "minibus": {
        "ef_per_vkm": Decimal("0.33490"),
        "wtt_per_vkm": Decimal("0.07547"),
        "default_occupancy": Decimal("15"),
    },
}

# 12. EEIO factors for spend-based calculation (kgCO2e per USD)
# Source: EPA USEEIO v2.0 / Exiobase 3
EEIO_FACTORS: Dict[str, Dict[str, Any]] = {
    "485000": {
        "name": "Ground passenger transport",
        "ef": Decimal("0.2600"),
    },
    "485110": {
        "name": "Mixed mode transit",
        "ef": Decimal("0.2200"),
    },
    "485210": {
        "name": "Interurban bus transport",
        "ef": Decimal("0.2400"),
    },
    "487110": {
        "name": "Scenic/sightseeing rail",
        "ef": Decimal("0.3100"),
    },
    "488490": {
        "name": "Other transport support",
        "ef": Decimal("0.1900"),
    },
    "532100": {
        "name": "Automotive rental/leasing",
        "ef": Decimal("0.1950"),
    },
    "811100": {
        "name": "Automotive repair/maintenance",
        "ef": Decimal("0.1500"),
    },
}

# 13. Currency exchange rates to USD (approximate mid-market rates)
CURRENCY_RATES: Dict[CurrencyCode, Decimal] = {
    CurrencyCode.USD: Decimal("1.0"),
    CurrencyCode.EUR: Decimal("1.0850"),
    CurrencyCode.GBP: Decimal("1.2650"),
    CurrencyCode.CAD: Decimal("0.7410"),
    CurrencyCode.AUD: Decimal("0.6520"),
    CurrencyCode.JPY: Decimal("0.006667"),
    CurrencyCode.CNY: Decimal("0.1378"),
    CurrencyCode.INR: Decimal("0.01198"),
    CurrencyCode.CHF: Decimal("1.1280"),
    CurrencyCode.SGD: Decimal("0.7440"),
    CurrencyCode.BRL: Decimal("0.1990"),
    CurrencyCode.ZAR: Decimal("0.05340"),
}

# 14. CPI deflators for spend-based calculation (base year 2021 = 1.0)
# Source: US BLS CPI-U / OECD CPI
CPI_DEFLATORS: Dict[int, Decimal] = {
    2015: Decimal("0.8490"),
    2016: Decimal("0.8597"),
    2017: Decimal("0.8781"),
    2018: Decimal("0.8997"),
    2019: Decimal("0.9153"),
    2020: Decimal("0.9271"),
    2021: Decimal("1.0000"),
    2022: Decimal("1.0800"),
    2023: Decimal("1.1152"),
    2024: Decimal("1.1490"),
    2025: Decimal("1.1780"),
}

# 15. Data Quality Indicator scoring and weights
# Scoring matrix: 1-5 scale per dimension; Weights sum to 1.0
DQI_SCORING: Dict[DQIDimension, Dict[DQIScore, Decimal]] = {
    DQIDimension.REPRESENTATIVENESS: {
        DQIScore.VERY_HIGH: Decimal("5"),
        DQIScore.HIGH: Decimal("4"),
        DQIScore.MEDIUM: Decimal("3"),
        DQIScore.LOW: Decimal("2"),
        DQIScore.VERY_LOW: Decimal("1"),
    },
    DQIDimension.COMPLETENESS: {
        DQIScore.VERY_HIGH: Decimal("5"),
        DQIScore.HIGH: Decimal("4"),
        DQIScore.MEDIUM: Decimal("3"),
        DQIScore.LOW: Decimal("2"),
        DQIScore.VERY_LOW: Decimal("1"),
    },
    DQIDimension.TEMPORAL: {
        DQIScore.VERY_HIGH: Decimal("5"),
        DQIScore.HIGH: Decimal("4"),
        DQIScore.MEDIUM: Decimal("3"),
        DQIScore.LOW: Decimal("2"),
        DQIScore.VERY_LOW: Decimal("1"),
    },
    DQIDimension.GEOGRAPHICAL: {
        DQIScore.VERY_HIGH: Decimal("5"),
        DQIScore.HIGH: Decimal("4"),
        DQIScore.MEDIUM: Decimal("3"),
        DQIScore.LOW: Decimal("2"),
        DQIScore.VERY_LOW: Decimal("1"),
    },
    DQIDimension.TECHNOLOGICAL: {
        DQIScore.VERY_HIGH: Decimal("5"),
        DQIScore.HIGH: Decimal("4"),
        DQIScore.MEDIUM: Decimal("3"),
        DQIScore.LOW: Decimal("2"),
        DQIScore.VERY_LOW: Decimal("1"),
    },
}

# DQI dimension weights (sum to 1.0)
DQI_WEIGHTS: Dict[DQIDimension, Decimal] = {
    DQIDimension.REPRESENTATIVENESS: Decimal("0.30"),
    DQIDimension.COMPLETENESS: Decimal("0.25"),
    DQIDimension.TEMPORAL: Decimal("0.15"),
    DQIDimension.GEOGRAPHICAL: Decimal("0.15"),
    DQIDimension.TECHNOLOGICAL: Decimal("0.15"),
}

# 16. Uncertainty ranges by calculation method and data quality tier
# Values represent the half-width of the 95% confidence interval as a fraction
UNCERTAINTY_RANGES: Dict[str, Dict[DataQualityTier, Decimal]] = {
    "employee_specific": {
        DataQualityTier.TIER_1: Decimal("0.05"),
        DataQualityTier.TIER_2: Decimal("0.10"),
        DataQualityTier.TIER_3: Decimal("0.15"),
    },
    "average_data": {
        DataQualityTier.TIER_1: Decimal("0.20"),
        DataQualityTier.TIER_2: Decimal("0.30"),
        DataQualityTier.TIER_3: Decimal("0.40"),
    },
    "spend_based": {
        DataQualityTier.TIER_1: Decimal("0.40"),
        DataQualityTier.TIER_2: Decimal("0.50"),
        DataQualityTier.TIER_3: Decimal("0.60"),
    },
    "telework": {
        DataQualityTier.TIER_1: Decimal("0.15"),
        DataQualityTier.TIER_2: Decimal("0.25"),
        DataQualityTier.TIER_3: Decimal("0.35"),
    },
    "transit": {
        DataQualityTier.TIER_1: Decimal("0.10"),
        DataQualityTier.TIER_2: Decimal("0.18"),
        DataQualityTier.TIER_3: Decimal("0.25"),
    },
    "vehicle_distance": {
        DataQualityTier.TIER_1: Decimal("0.08"),
        DataQualityTier.TIER_2: Decimal("0.15"),
        DataQualityTier.TIER_3: Decimal("0.22"),
    },
}

# Required disclosures per compliance framework for employee commuting (Cat 7)
FRAMEWORK_REQUIRED_DISCLOSURES: Dict[ComplianceFramework, List[str]] = {
    ComplianceFramework.GHG_PROTOCOL: [
        "total_co2e",
        "method_used",
        "ef_sources",
        "exclusions",
        "dqi_score",
        "mode_breakdown",
    ],
    ComplianceFramework.ISO_14064: [
        "total_co2e",
        "uncertainty_analysis",
        "base_year",
        "methodology",
        "data_sources",
    ],
    ComplianceFramework.CSRD_ESRS: [
        "total_co2e",
        "category_breakdown",
        "methodology",
        "targets",
        "actions",
        "telework_policy",
    ],
    ComplianceFramework.CDP: [
        "total_co2e",
        "mode_breakdown",
        "employee_count",
        "survey_coverage",
        "verification_status",
    ],
    ComplianceFramework.SBTI: [
        "total_co2e",
        "target_coverage",
        "reduction_initiatives",
        "progress_tracking",
    ],
    ComplianceFramework.SB_253: [
        "total_co2e",
        "methodology",
        "assurance_opinion",
    ],
    ComplianceFramework.GRI: [
        "total_co2e",
        "gases_included",
        "base_year",
        "standards_used",
        "intensity_ratios",
    ],
}

# Work schedule fractions (multiplier on working days)
WORK_SCHEDULE_FRACTIONS: Dict[WorkSchedule, Decimal] = {
    WorkSchedule.FULL_TIME: Decimal("1.0"),
    WorkSchedule.PART_TIME_80: Decimal("0.8"),
    WorkSchedule.PART_TIME_60: Decimal("0.6"),
    WorkSchedule.PART_TIME_50: Decimal("0.5"),
}

# Telework frequency fractions (fraction of days worked from home)
TELEWORK_FREQUENCY_FRACTIONS: Dict[TeleworkFrequency, Decimal] = {
    TeleworkFrequency.FULL_REMOTE: Decimal("1.0"),
    TeleworkFrequency.HYBRID_4: Decimal("0.8"),
    TeleworkFrequency.HYBRID_3: Decimal("0.6"),
    TeleworkFrequency.HYBRID_2: Decimal("0.4"),
    TeleworkFrequency.HYBRID_1: Decimal("0.2"),
    TeleworkFrequency.OFFICE_FULL: Decimal("0.0"),
}

# Seasonal adjustment multipliers for telework energy
SEASONAL_ADJUSTMENT_MULTIPLIERS: Dict[SeasonalAdjustment, Decimal] = {
    SeasonalAdjustment.NONE: Decimal("1.0"),
    SeasonalAdjustment.HEATING_ONLY: Decimal("1.15"),
    SeasonalAdjustment.COOLING_ONLY: Decimal("1.10"),
    SeasonalAdjustment.FULL_SEASONAL: Decimal("1.25"),
}


# ==============================================================================
# INPUT MODELS (11 total)
# ==============================================================================


class CommuteInput(BaseModel):
    """
    Input for individual commute emissions calculation (distance-based).

    Used for employee-specific method where individual commute data is known.
    Supports SOV, carpool, motorcycle, and other vehicle-based modes.

    Example:
        >>> commute = CommuteInput(
        ...     mode=CommuteMode.SOV,
        ...     vehicle_type=VehicleType.CAR_MEDIUM_PETROL,
        ...     one_way_distance_km=Decimal("15.0"),
        ...     commute_days_per_week=5
        ... )
    """

    mode: CommuteMode = Field(
        ..., description="Commute mode (SOV, carpool, etc.)"
    )
    vehicle_type: Optional[VehicleType] = Field(
        default=None,
        description="Vehicle type for SOV/carpool (None for transit/active)"
    )
    one_way_distance_km: Decimal = Field(
        ..., gt=0,
        description="One-way commute distance in kilometres"
    )
    commute_days_per_week: int = Field(
        default=5, ge=1, le=7,
        description="Number of commute days per week"
    )
    work_schedule: WorkSchedule = Field(
        default=WorkSchedule.FULL_TIME,
        description="Employee work schedule"
    )
    region: RegionCode = Field(
        default=RegionCode.GLOBAL,
        description="Region for working days lookup"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)

    @validator("one_way_distance_km")
    def validate_distance(cls, v: Decimal) -> Decimal:
        """Validate commute distance is positive and reasonable."""
        if v <= 0:
            raise ValueError(
                f"One-way distance must be positive, got {v}"
            )
        if v > Decimal("500"):
            raise ValueError(
                f"One-way commute distance exceeds 500 km: {v}. "
                "This may not be a daily commute."
            )
        return v


class FuelBasedCommuteInput(BaseModel):
    """
    Input for fuel-based commute emissions calculation.

    Used when fuel consumption is known directly (e.g., from fuel receipts).

    Example:
        >>> fuel_commute = FuelBasedCommuteInput(
        ...     fuel_type=FuelType.PETROL,
        ...     litres_per_week=Decimal("12.5"),
        ...     commute_weeks_per_year=48
        ... )
    """

    fuel_type: FuelType = Field(
        ..., description="Fuel type consumed"
    )
    litres_per_week: Decimal = Field(
        ..., gt=0,
        description="Litres of fuel consumed per week for commuting"
    )
    commute_weeks_per_year: int = Field(
        default=48, ge=1, le=52,
        description="Number of weeks commuting per year"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)

    @validator("litres_per_week")
    def validate_litres(cls, v: Decimal) -> Decimal:
        """Validate fuel quantity is positive."""
        if v <= 0:
            raise ValueError(
                f"Fuel quantity must be positive, got {v}"
            )
        return v


class CarpoolInput(BaseModel):
    """
    Input for carpool commute emissions calculation.

    Divides vehicle emissions by the number of occupants.

    Example:
        >>> carpool = CarpoolInput(
        ...     vehicle_type=VehicleType.CAR_AVERAGE,
        ...     one_way_distance_km=Decimal("20.0"),
        ...     occupants=3,
        ...     commute_days_per_week=5
        ... )
    """

    vehicle_type: VehicleType = Field(
        default=VehicleType.CAR_AVERAGE,
        description="Carpool vehicle type"
    )
    one_way_distance_km: Decimal = Field(
        ..., gt=0,
        description="One-way commute distance in kilometres"
    )
    occupants: int = Field(
        ..., ge=2, le=8,
        description="Number of occupants sharing the ride (including driver)"
    )
    commute_days_per_week: int = Field(
        default=5, ge=1, le=7,
        description="Number of commute days per week"
    )
    work_schedule: WorkSchedule = Field(
        default=WorkSchedule.FULL_TIME,
        description="Employee work schedule"
    )
    region: RegionCode = Field(
        default=RegionCode.GLOBAL,
        description="Region for working days lookup"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)

    @validator("one_way_distance_km")
    def validate_distance(cls, v: Decimal) -> Decimal:
        """Validate carpool distance is positive."""
        if v <= 0:
            raise ValueError(
                f"One-way distance must be positive, got {v}"
            )
        return v


class TransitInput(BaseModel):
    """
    Input for public transit commute emissions calculation.

    Supports bus, metro, light rail, commuter rail, and ferry.

    Example:
        >>> transit = TransitInput(
        ...     transit_type=TransitType.METRO,
        ...     one_way_distance_km=Decimal("8.5"),
        ...     commute_days_per_week=5
        ... )
    """

    transit_type: TransitType = Field(
        ..., description="Type of public transit service"
    )
    one_way_distance_km: Decimal = Field(
        ..., gt=0,
        description="One-way transit distance in kilometres"
    )
    commute_days_per_week: int = Field(
        default=5, ge=1, le=7,
        description="Number of commute days per week"
    )
    work_schedule: WorkSchedule = Field(
        default=WorkSchedule.FULL_TIME,
        description="Employee work schedule"
    )
    region: RegionCode = Field(
        default=RegionCode.GLOBAL,
        description="Region for working days lookup"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)

    @validator("one_way_distance_km")
    def validate_distance(cls, v: Decimal) -> Decimal:
        """Validate transit distance is positive."""
        if v <= 0:
            raise ValueError(
                f"One-way distance must be positive, got {v}"
            )
        return v


class TeleworkInput(BaseModel):
    """
    Input for telework (remote work) emissions calculation.

    Calculates home-office energy emissions from electricity consumption
    for laptop, heating, cooling, and lighting.

    Example:
        >>> telework = TeleworkInput(
        ...     frequency=TeleworkFrequency.HYBRID_3,
        ...     region=RegionCode.US,
        ...     seasonal_adjustment=SeasonalAdjustment.FULL_SEASONAL
        ... )
    """

    frequency: TeleworkFrequency = Field(
        ..., description="Telework frequency pattern"
    )
    region: RegionCode = Field(
        default=RegionCode.GLOBAL,
        description="Region for grid emission factor and working days"
    )
    daily_kwh_override: Optional[Decimal] = Field(
        default=None, gt=0,
        description="Override daily kWh consumption (default: TELEWORK_ENERGY_DEFAULTS)"
    )
    seasonal_adjustment: SeasonalAdjustment = Field(
        default=SeasonalAdjustment.NONE,
        description="Seasonal adjustment for energy consumption"
    )
    work_schedule: WorkSchedule = Field(
        default=WorkSchedule.FULL_TIME,
        description="Employee work schedule"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)


class SurveyResponseInput(BaseModel):
    """
    Individual employee survey response for commute data collection.

    Example:
        >>> response = SurveyResponseInput(
        ...     employee_id="EMP-1234",
        ...     mode=CommuteMode.SOV,
        ...     vehicle_type=VehicleType.CAR_MEDIUM_PETROL,
        ...     one_way_distance_km=Decimal("18.5"),
        ...     commute_days_per_week=5,
        ...     telework_frequency=TeleworkFrequency.HYBRID_2,
        ...     department="Engineering",
        ...     site="HQ-London"
        ... )
    """

    employee_id: str = Field(
        ..., min_length=1,
        description="Unique employee identifier"
    )
    mode: CommuteMode = Field(
        ..., description="Primary commute mode"
    )
    vehicle_type: Optional[VehicleType] = Field(
        default=None,
        description="Vehicle type (if mode is SOV/carpool/motorcycle)"
    )
    one_way_distance_km: Decimal = Field(
        ..., gt=0,
        description="One-way commute distance in kilometres"
    )
    commute_days_per_week: int = Field(
        default=5, ge=1, le=7,
        description="Number of commute days per week"
    )
    telework_frequency: TeleworkFrequency = Field(
        default=TeleworkFrequency.OFFICE_FULL,
        description="Telework frequency"
    )
    work_schedule: WorkSchedule = Field(
        default=WorkSchedule.FULL_TIME,
        description="Employee work schedule"
    )
    department: Optional[str] = Field(
        default=None,
        description="Employee department for allocation"
    )
    site: Optional[str] = Field(
        default=None,
        description="Office site / location"
    )
    cost_center: Optional[str] = Field(
        default=None,
        description="Cost center for allocation"
    )

    model_config = ConfigDict(frozen=True)

    @validator("one_way_distance_km")
    def validate_distance(cls, v: Decimal) -> Decimal:
        """Validate survey distance is positive."""
        if v <= 0:
            raise ValueError(
                f"One-way distance must be positive, got {v}"
            )
        return v


class SurveyInput(BaseModel):
    """
    Input for processing an employee commute survey.

    Contains survey metadata and all individual responses.

    Example:
        >>> survey = SurveyInput(
        ...     survey_method=SurveyMethod.STRATIFIED_SAMPLE,
        ...     total_employees=5000,
        ...     responses=[response1, response2, ...],
        ...     region=RegionCode.US,
        ...     reporting_period="2024"
        ... )
    """

    survey_method: SurveyMethod = Field(
        ..., description="Survey methodology used"
    )
    total_employees: int = Field(
        ..., gt=0,
        description="Total number of employees in scope"
    )
    responses: List[SurveyResponseInput] = Field(
        ..., min_length=1,
        description="List of survey responses"
    )
    region: RegionCode = Field(
        default=RegionCode.GLOBAL,
        description="Region for working days and grid factor defaults"
    )
    reporting_period: str = Field(
        ..., description="Reporting period (e.g., '2024', '2024-Q3')"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)

    @validator("total_employees")
    def validate_total_employees(cls, v: int) -> int:
        """Validate total employees is positive."""
        if v <= 0:
            raise ValueError(
                f"Total employees must be positive, got {v}"
            )
        return v


class AverageDataInput(BaseModel):
    """
    Input for average-data calculation method.

    Uses national census or default commute distance and mode share data
    to estimate emissions for the entire employee population.

    Example:
        >>> avg = AverageDataInput(
        ...     total_employees=5000,
        ...     region=RegionCode.US,
        ...     reporting_period="2024"
        ... )
    """

    total_employees: int = Field(
        ..., gt=0,
        description="Total number of employees in scope"
    )
    region: RegionCode = Field(
        default=RegionCode.GLOBAL,
        description="Region for default distance and mode share"
    )
    mode_share_override: Optional[Dict[str, Decimal]] = Field(
        default=None,
        description="Custom mode share distribution (overrides DEFAULT_MODE_SHARES)"
    )
    distance_override_km: Optional[Decimal] = Field(
        default=None, gt=0,
        description="Custom average one-way distance (overrides AVERAGE_COMMUTE_DISTANCES)"
    )
    telework_rate: Decimal = Field(
        default=Decimal("0.0"), ge=0, le=1,
        description="Fraction of employees teleworking (0.0-1.0)"
    )
    reporting_period: str = Field(
        ..., description="Reporting period (e.g., '2024', '2024-Q3')"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)

    @validator("total_employees")
    def validate_total_employees(cls, v: int) -> int:
        """Validate total employees is positive."""
        if v <= 0:
            raise ValueError(
                f"Total employees must be positive, got {v}"
            )
        return v


class SpendInput(BaseModel):
    """
    Input for spend-based emissions calculation using EEIO factors.

    Used when only commuting benefit spend data is available.

    Example:
        >>> spend = SpendInput(
        ...     naics_code="485000",
        ...     amount=Decimal("250000.00"),
        ...     currency=CurrencyCode.USD,
        ...     reporting_year=2024
        ... )
    """

    naics_code: str = Field(
        ..., description="NAICS code for EEIO factor lookup"
    )
    amount: Decimal = Field(
        ..., gt=0,
        description="Spend amount in specified currency"
    )
    currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="ISO 4217 currency code"
    )
    reporting_year: int = Field(
        default=2024, ge=2015, le=2030,
        description="Reporting year for CPI deflation"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)

    @validator("amount")
    def validate_amount(cls, v: Decimal) -> Decimal:
        """Validate spend amount is positive."""
        if v <= 0:
            raise ValueError(
                f"Spend amount must be positive, got {v}"
            )
        return v


class EmployeeInput(BaseModel):
    """
    Comprehensive input for a single employee's commuting emissions.

    Wraps commute and telework data with organizational metadata.

    Example:
        >>> employee = EmployeeInput(
        ...     employee_id="EMP-1234",
        ...     mode=CommuteMode.SOV,
        ...     vehicle_type=VehicleType.CAR_MEDIUM_PETROL,
        ...     one_way_distance_km=Decimal("18.5"),
        ...     commute_days_per_week=5,
        ...     telework_frequency=TeleworkFrequency.HYBRID_2,
        ...     region=RegionCode.US,
        ...     department="Engineering"
        ... )
    """

    employee_id: str = Field(
        ..., min_length=1,
        description="Unique employee identifier"
    )
    mode: CommuteMode = Field(
        ..., description="Primary commute mode"
    )
    vehicle_type: Optional[VehicleType] = Field(
        default=None,
        description="Vehicle type (if mode is SOV/carpool/motorcycle)"
    )
    one_way_distance_km: Decimal = Field(
        ..., gt=0,
        description="One-way commute distance in kilometres"
    )
    commute_days_per_week: int = Field(
        default=5, ge=1, le=7,
        description="Number of commute days per week"
    )
    telework_frequency: TeleworkFrequency = Field(
        default=TeleworkFrequency.OFFICE_FULL,
        description="Telework frequency pattern"
    )
    work_schedule: WorkSchedule = Field(
        default=WorkSchedule.FULL_TIME,
        description="Employee work schedule"
    )
    region: RegionCode = Field(
        default=RegionCode.GLOBAL,
        description="Region for grid factor and working days"
    )
    department: Optional[str] = Field(
        default=None,
        description="Department for allocation"
    )
    site: Optional[str] = Field(
        default=None,
        description="Office site / location"
    )
    cost_center: Optional[str] = Field(
        default=None,
        description="Cost center for allocation"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)

    @validator("one_way_distance_km")
    def validate_distance(cls, v: Decimal) -> Decimal:
        """Validate commute distance is positive and reasonable."""
        if v <= 0:
            raise ValueError(
                f"One-way distance must be positive, got {v}"
            )
        return v


class BatchEmployeeInput(BaseModel):
    """
    Batch input for processing multiple employees in a single request.

    Example:
        >>> batch = BatchEmployeeInput(
        ...     employees=[emp1, emp2, emp3],
        ...     reporting_period="2024",
        ...     allocation_method=AllocationMethod.DEPARTMENT
        ... )
    """

    employees: List[EmployeeInput] = Field(
        ..., min_length=1,
        description="List of employee commute inputs to process"
    )
    reporting_period: str = Field(
        ..., description="Reporting period (e.g., '2024', '2024-Q3')"
    )
    allocation_method: AllocationMethod = Field(
        default=AllocationMethod.EQUAL,
        description="Emissions allocation method"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)


# ==============================================================================
# RESULT MODELS (17 total)
# ==============================================================================


class CommuteResult(BaseModel):
    """
    Result from an individual commute emissions calculation.

    Contains direct (TTW) and well-to-tank (WTT) emissions with provenance.
    """

    mode: CommuteMode = Field(
        ..., description="Commute mode used"
    )
    vehicle_type: Optional[VehicleType] = Field(
        default=None,
        description="Vehicle type (if applicable)"
    )
    one_way_distance_km: Decimal = Field(
        ..., description="One-way commute distance (km)"
    )
    annual_commute_days: int = Field(
        ..., description="Annual commute days after adjustments"
    )
    annual_distance_km: Decimal = Field(
        ..., description="Total annual commute distance (km, round-trip)"
    )
    co2e: Decimal = Field(
        ..., description="Direct tank-to-wheel CO2e (kgCO2e/year)"
    )
    wtt_co2e: Decimal = Field(
        ..., description="Well-to-tank CO2e (kgCO2e/year)"
    )
    total_co2e: Decimal = Field(
        ..., description="Total CO2e including WTT (kgCO2e/year)"
    )
    ef_source: EFSource = Field(
        ..., description="Emission factor source"
    )
    distance_band: DistanceBand = Field(
        ..., description="Distance band classification"
    )
    method: CalculationMethod = Field(
        ..., description="Calculation method used"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(frozen=True)


class TeleworkResult(BaseModel):
    """
    Result from telework home-office emissions calculation.

    Calculates energy-based emissions from home-office electricity consumption.
    """

    frequency: TeleworkFrequency = Field(
        ..., description="Telework frequency pattern"
    )
    annual_telework_days: int = Field(
        ..., description="Annual telework days"
    )
    daily_kwh: Decimal = Field(
        ..., description="Daily electricity consumption (kWh)"
    )
    annual_kwh: Decimal = Field(
        ..., description="Annual electricity consumption (kWh)"
    )
    grid_ef: Decimal = Field(
        ..., description="Grid emission factor used (kgCO2e/kWh)"
    )
    seasonal_multiplier: Decimal = Field(
        ..., description="Seasonal adjustment multiplier applied"
    )
    co2e: Decimal = Field(
        ..., description="Total telework CO2e (kgCO2e/year)"
    )
    region: RegionCode = Field(
        ..., description="Region used for grid factor"
    )
    ef_source: EFSource = Field(
        ..., description="Emission factor source"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(frozen=True)


class EmployeeResult(BaseModel):
    """
    Combined result for a single employee (commute + telework).

    Provides total emissions and breakdown by commute and telework components.
    """

    employee_id: str = Field(
        ..., description="Employee identifier"
    )
    commute_result: Optional[CommuteResult] = Field(
        default=None,
        description="Commute emissions result"
    )
    telework_result: Optional[TeleworkResult] = Field(
        default=None,
        description="Telework emissions result"
    )
    total_co2e: Decimal = Field(
        ..., description="Total CO2e (commute + telework, kgCO2e/year)"
    )
    department: Optional[str] = Field(
        default=None,
        description="Employee department"
    )
    site: Optional[str] = Field(
        default=None,
        description="Employee office site"
    )
    method: CalculationMethod = Field(
        ..., description="Calculation method used"
    )
    dqi_score: Optional[Decimal] = Field(
        default=None,
        description="Data quality indicator score (1-5)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(frozen=True)


class SurveyResult(BaseModel):
    """
    Result from processing an employee commute survey.

    Includes per-respondent results and extrapolation to total employee population.
    """

    survey_method: SurveyMethod = Field(
        ..., description="Survey methodology used"
    )
    total_employees: int = Field(
        ..., description="Total employees in scope"
    )
    respondent_count: int = Field(
        ..., description="Number of survey respondents"
    )
    response_rate: Decimal = Field(
        ..., description="Survey response rate (0-1)"
    )
    sample_co2e: Decimal = Field(
        ..., description="Total CO2e from survey sample (kgCO2e/year)"
    )
    extrapolated_co2e: Decimal = Field(
        ..., description="Extrapolated total CO2e for all employees (kgCO2e/year)"
    )
    per_employee_avg_co2e: Decimal = Field(
        ..., description="Average CO2e per employee (kgCO2e/year)"
    )
    mode_breakdown: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="CO2e breakdown by commute mode"
    )
    employee_results: List[EmployeeResult] = Field(
        default_factory=list,
        description="Individual employee results from survey"
    )
    reporting_period: str = Field(
        ..., description="Reporting period"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(frozen=True)


class BatchResult(BaseModel):
    """Result from batch employee commute processing."""

    results: List[EmployeeResult] = Field(
        ..., description="Individual employee results"
    )
    total_co2e: Decimal = Field(
        ..., description="Total CO2e for all employees (kgCO2e/year)"
    )
    count: int = Field(
        ..., description="Total number of employees processed"
    )
    errors: List[dict] = Field(
        default_factory=list,
        description="Errors from failed employee calculations"
    )
    status: BatchStatus = Field(
        ..., description="Batch processing status"
    )
    reporting_period: str = Field(
        ..., description="Reporting period"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(frozen=True)


class AggregationResult(BaseModel):
    """Aggregated commuting emissions by various dimensions."""

    total_co2e: Decimal = Field(
        ..., description="Total CO2e (kgCO2e/year)"
    )
    by_mode: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="CO2e breakdown by commute mode"
    )
    by_distance_band: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="CO2e breakdown by distance band"
    )
    by_department: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="CO2e breakdown by department"
    )
    by_site: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="CO2e breakdown by office site"
    )
    commute_co2e: Decimal = Field(
        ..., description="Total commute-only CO2e (kgCO2e/year)"
    )
    telework_co2e: Decimal = Field(
        ..., description="Total telework-only CO2e (kgCO2e/year)"
    )
    period: str = Field(
        ..., description="Reporting period"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(frozen=True)


class ModeShareResult(BaseModel):
    """Mode share analysis result with emissions intensity per mode."""

    mode: str = Field(
        ..., description="Commute mode name"
    )
    employee_count: int = Field(
        ..., description="Number of employees using this mode"
    )
    employee_share: Decimal = Field(
        ..., description="Fraction of employees using this mode (0-1)"
    )
    total_co2e: Decimal = Field(
        ..., description="Total CO2e for this mode (kgCO2e/year)"
    )
    co2e_share: Decimal = Field(
        ..., description="Fraction of total CO2e from this mode (0-1)"
    )
    avg_co2e_per_employee: Decimal = Field(
        ..., description="Average CO2e per employee using this mode (kgCO2e/year)"
    )
    avg_distance_km: Decimal = Field(
        ..., description="Average one-way distance for this mode (km)"
    )

    model_config = ConfigDict(frozen=True)


class ComplianceCheckResult(BaseModel):
    """Result from compliance check against a specific framework."""

    framework: ComplianceFramework = Field(
        ..., description="Framework checked"
    )
    status: ComplianceStatus = Field(
        ..., description="Compliance status"
    )
    score: Decimal = Field(
        ..., description="Compliance score (0-100)"
    )
    findings: List[dict] = Field(
        default_factory=list,
        description="Specific findings (gaps, issues)"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improvement"
    )

    model_config = ConfigDict(frozen=True)


class UncertaintyResult(BaseModel):
    """Result from uncertainty quantification."""

    mean: Decimal = Field(
        ..., description="Mean emissions estimate (kgCO2e)"
    )
    std_dev: Decimal = Field(
        ..., description="Standard deviation (kgCO2e)"
    )
    ci_lower: Decimal = Field(
        ..., description="Confidence interval lower bound (kgCO2e)"
    )
    ci_upper: Decimal = Field(
        ..., description="Confidence interval upper bound (kgCO2e)"
    )
    method: UncertaintyMethod = Field(
        ..., description="Method used"
    )
    iterations: int = Field(
        ..., description="Number of iterations (Monte Carlo)"
    )
    confidence_level: Decimal = Field(
        ..., description="Confidence level (e.g., 0.95)"
    )

    model_config = ConfigDict(frozen=True)


class DataQualityResult(BaseModel):
    """Result from data quality assessment."""

    overall_score: Decimal = Field(
        ..., description="Weighted composite DQI score (1-5)"
    )
    dimensions: Dict[str, Decimal] = Field(
        ..., description="Score per dimension"
    )
    classification: str = Field(
        ..., description="Quality classification (Excellent/Good/Fair/Poor/Very Poor)"
    )
    tier: DataQualityTier = Field(
        ..., description="Data quality tier assignment"
    )

    model_config = ConfigDict(frozen=True)


class ProvenanceRecord(BaseModel):
    """Single record in the provenance chain."""

    stage: str = Field(
        ..., description="Pipeline stage name"
    )
    input_hash: str = Field(
        ..., description="SHA-256 hash of stage input"
    )
    output_hash: str = Field(
        ..., description="SHA-256 hash of stage output"
    )
    chain_hash: str = Field(
        ..., description="Cumulative chain hash (input_hash + previous chain_hash)"
    )
    timestamp: str = Field(
        ..., description="ISO 8601 timestamp"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Stage-specific metadata"
    )

    model_config = ConfigDict(frozen=True)


class ProvenanceChainResult(BaseModel):
    """Complete provenance chain for an emissions calculation."""

    records: List[ProvenanceRecord] = Field(
        ..., description="Ordered list of provenance records"
    )
    is_valid: bool = Field(
        ..., description="Whether chain integrity is verified"
    )
    chain_hash: str = Field(
        ..., description="Final chain hash"
    )

    model_config = ConfigDict(frozen=True)


class SpendResult(BaseModel):
    """Result from spend-based EEIO emissions calculation."""

    naics_code: str = Field(
        ..., description="NAICS code used"
    )
    spend_usd: Decimal = Field(
        ..., description="Spend amount in USD after currency conversion"
    )
    cpi_deflator: Decimal = Field(
        ..., description="CPI deflator applied"
    )
    eeio_factor: Decimal = Field(
        ..., description="EEIO factor (kgCO2e/USD)"
    )
    co2e: Decimal = Field(
        ..., description="Total CO2e (kgCO2e)"
    )
    ef_source: EFSource = Field(
        ..., description="Emission factor source"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(frozen=True)


class WorkingDaysResult(BaseModel):
    """Result from working days calculation for a region."""

    region: RegionCode = Field(
        ..., description="Region code"
    )
    gross_working_days: int = Field(
        ..., description="Total working days before schedule adjustments"
    )
    holidays: int = Field(
        ..., description="Public holidays deducted"
    )
    pto: int = Field(
        ..., description="Paid time off deducted"
    )
    sick: int = Field(
        ..., description="Sick days deducted"
    )
    net_working_days: int = Field(
        ..., description="Net working days after all deductions"
    )
    schedule_fraction: Decimal = Field(
        ..., description="Work schedule fraction applied"
    )
    adjusted_days: int = Field(
        ..., description="Final adjusted working days"
    )

    model_config = ConfigDict(frozen=True)


class GridEFResult(BaseModel):
    """Result from grid emission factor lookup."""

    region: RegionCode = Field(
        ..., description="Region code"
    )
    grid_ef: Decimal = Field(
        ..., description="Grid emission factor (kgCO2e/kWh)"
    )
    ef_source: EFSource = Field(
        ..., description="Emission factor source"
    )

    model_config = ConfigDict(frozen=True)


class HotSpotResult(BaseModel):
    """
    Hot-spot analysis result identifying top emission contributors
    and reduction opportunities for employee commuting.
    """

    top_modes: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by commute mode (ranked)"
    )
    top_distance_bands: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by distance band (ranked)"
    )
    top_departments: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Top emitting departments"
    )
    top_sites: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Top emitting office sites"
    )
    reduction_opportunities: List[dict] = Field(
        default_factory=list,
        description="Identified reduction opportunities"
    )
    sov_share: Decimal = Field(
        ..., description="Fraction of SOV commuters (0-1)"
    )
    avg_distance_km: Decimal = Field(
        ..., description="Average one-way commute distance (km)"
    )

    model_config = ConfigDict(frozen=True)


class MetricsSummary(BaseModel):
    """Summary metrics for monitoring and dashboarding."""

    total_calculations: int = Field(
        ..., description="Total number of calculations performed"
    )
    total_co2e: Decimal = Field(
        ..., description="Total CO2e across all calculations (kgCO2e)"
    )
    total_employees: int = Field(
        ..., description="Total number of employees processed"
    )
    total_commute_co2e: Decimal = Field(
        ..., description="Total commute emissions (kgCO2e)"
    )
    total_telework_co2e: Decimal = Field(
        ..., description="Total telework emissions (kgCO2e)"
    )
    avg_co2e_per_employee: Decimal = Field(
        ..., description="Average CO2e per employee (kgCO2e)"
    )
    avg_dqi: Decimal = Field(
        ..., description="Average data quality indicator score"
    )
    sov_rate: Decimal = Field(
        ..., description="Fraction of employees driving alone"
    )
    telework_rate: Decimal = Field(
        ..., description="Fraction of employees teleworking (any frequency)"
    )

    model_config = ConfigDict(frozen=True)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

# Quantization constant: 8 decimal places
_QUANT_8DP = Decimal("0.00000001")


def calculate_provenance_hash(*inputs: Any) -> str:
    """
    Calculate SHA-256 provenance hash from variable inputs.

    Supports Pydantic models (serialized to sorted JSON), Decimal values,
    and any other stringifiable objects.

    Args:
        *inputs: Variable number of input objects to hash.

    Returns:
        Hexadecimal SHA-256 hash string (64 characters).

    Example:
        >>> h = calculate_provenance_hash("EMP-1234", Decimal("1234.56"))
        >>> len(h)
        64
    """
    hash_input = ""
    for inp in inputs:
        if isinstance(inp, BaseModel):
            # Pydantic v2 model_dump_json() does not support sort_keys;
            # serialise via json.dumps with sort_keys for deterministic output.
            hash_input += json.dumps(
                inp.model_dump(mode="json"), sort_keys=True, default=str
            )
        elif isinstance(inp, Decimal):
            hash_input += str(inp.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP))
        else:
            hash_input += str(inp)

    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


def get_dqi_classification(score: Decimal) -> str:
    """
    Classify a composite DQI score into a human-readable label.

    Score range 1-5 (5 = best):
      >=4.5 -> Excellent
      >=3.5 -> Good
      >=2.5 -> Fair
      >=1.5 -> Poor
      <1.5  -> Very Poor

    Args:
        score: Composite DQI score (1-5).

    Returns:
        Classification string.

    Example:
        >>> get_dqi_classification(Decimal("4.2"))
        'Good'
        >>> get_dqi_classification(Decimal("4.8"))
        'Excellent'
    """
    if score >= Decimal("4.5"):
        return "Excellent"
    elif score >= Decimal("3.5"):
        return "Good"
    elif score >= Decimal("2.5"):
        return "Fair"
    elif score >= Decimal("1.5"):
        return "Poor"
    else:
        return "Very Poor"


def convert_currency_to_usd(
    amount: Decimal,
    currency: CurrencyCode,
    year: Optional[int] = None,
) -> Decimal:
    """
    Convert an amount from the given currency to USD using stored exchange rates.

    Optionally deflates to base year 2021 USD if a year is provided.

    Args:
        amount: Amount in the source currency.
        currency: Source currency code.
        year: Optional spend year for CPI deflation (base year 2021).

    Returns:
        Equivalent amount in USD, quantized to 8 decimal places.

    Raises:
        ValueError: If currency code is not found in CURRENCY_RATES.
        ValueError: If year is provided but not found in CPI_DEFLATORS.

    Example:
        >>> convert_currency_to_usd(Decimal("1000"), CurrencyCode.EUR)
        Decimal('1085.00000000')
        >>> convert_currency_to_usd(Decimal("1000"), CurrencyCode.USD, year=2024)
        Decimal('870.32201914')
    """
    rate = CURRENCY_RATES.get(currency)
    if rate is None:
        raise ValueError(
            f"Currency '{currency.value}' not found in CURRENCY_RATES"
        )
    usd_amount = amount * rate

    if year is not None:
        deflator = CPI_DEFLATORS.get(year)
        if deflator is None:
            raise ValueError(
                f"CPI deflator not available for year {year}. "
                f"Available years: {sorted(CPI_DEFLATORS.keys())}"
            )
        # Deflate to base year 2021 (deflator for 2021 = 1.0)
        usd_amount = usd_amount / deflator

    return usd_amount.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)


def get_working_days(
    region: RegionCode,
    work_schedule: WorkSchedule = WorkSchedule.FULL_TIME,
) -> int:
    """
    Get net annual working days for a region, adjusted by work schedule.

    Args:
        region: Region code for working days lookup.
        work_schedule: Employee work schedule for fractional adjustment.

    Returns:
        Net annual working days as an integer.

    Example:
        >>> get_working_days(RegionCode.US)
        225
        >>> get_working_days(RegionCode.DE, WorkSchedule.PART_TIME_80)
        160
    """
    region_data = WORKING_DAYS_DEFAULTS.get(region)
    if region_data is None:
        # Fall back to GLOBAL
        region_data = WORKING_DAYS_DEFAULTS[RegionCode.GLOBAL]

    net_days = region_data["net"]
    schedule_fraction = WORK_SCHEDULE_FRACTIONS[work_schedule]

    adjusted = Decimal(str(net_days)) * schedule_fraction
    return int(adjusted.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def classify_distance_band(one_way_km: Decimal) -> DistanceBand:
    """
    Classify a one-way commute distance into a distance band.

    Bands:
      - SHORT_0_5: 0-5 km
      - MEDIUM_5_15: 5-15 km
      - LONG_15_30: 15-30 km
      - VERY_LONG_30_PLUS: 30+ km

    Args:
        one_way_km: One-way commute distance in kilometres.

    Returns:
        DistanceBand classification.

    Example:
        >>> classify_distance_band(Decimal("3.5"))
        <DistanceBand.SHORT_0_5: 'short_0_5'>
        >>> classify_distance_band(Decimal("12.0"))
        <DistanceBand.MEDIUM_5_15: 'medium_5_15'>
        >>> classify_distance_band(Decimal("22.0"))
        <DistanceBand.LONG_15_30: 'long_15_30'>
        >>> classify_distance_band(Decimal("45.0"))
        <DistanceBand.VERY_LONG_30_PLUS: 'very_long_30_plus'>
    """
    if one_way_km <= Decimal("5"):
        return DistanceBand.SHORT_0_5
    elif one_way_km <= Decimal("15"):
        return DistanceBand.MEDIUM_5_15
    elif one_way_km <= Decimal("30"):
        return DistanceBand.LONG_15_30
    else:
        return DistanceBand.VERY_LONG_30_PLUS


def get_grid_ef(region: RegionCode) -> Decimal:
    """
    Get grid emission factor for a region, falling back to GLOBAL default.

    Args:
        region: Region code for grid emission factor lookup.

    Returns:
        Grid emission factor in kgCO2e per kWh.

    Example:
        >>> get_grid_ef(RegionCode.US)
        Decimal('0.37170')
        >>> get_grid_ef(RegionCode.GLOBAL)
        Decimal('0.43600')
    """
    return GRID_EMISSION_FACTORS.get(
        region, GRID_EMISSION_FACTORS[RegionCode.GLOBAL]
    )


def get_average_commute_distance(region: RegionCode) -> Decimal:
    """
    Get average one-way commute distance for a region.

    Falls back to GLOBAL default if region-specific data is unavailable.

    Args:
        region: Region code for commute distance lookup.

    Returns:
        Average one-way commute distance in kilometres.

    Example:
        >>> get_average_commute_distance(RegionCode.US)
        Decimal('21.7')
        >>> get_average_commute_distance(RegionCode.GLOBAL)
        Decimal('15.0')
    """
    return AVERAGE_COMMUTE_DISTANCES.get(
        region.value, AVERAGE_COMMUTE_DISTANCES["GLOBAL"]
    )


def get_default_mode_shares(region_key: str) -> Dict[str, Decimal]:
    """
    Get default commute mode share distribution for a region.

    Falls back to US distribution if region key is not found.

    Args:
        region_key: Region key ("US", "GB", or "EU").

    Returns:
        Dictionary mapping mode names to share fractions.

    Example:
        >>> shares = get_default_mode_shares("US")
        >>> shares["sov"]
        Decimal('0.7610')
    """
    return DEFAULT_MODE_SHARES.get(region_key, DEFAULT_MODE_SHARES["US"])


def get_eeio_factor(naics_code: str) -> Optional[Decimal]:
    """
    Get EEIO emission factor by NAICS code.

    Args:
        naics_code: NAICS industry code string.

    Returns:
        EEIO factor in kgCO2e per USD, or None if not found.

    Example:
        >>> get_eeio_factor("485000")
        Decimal('0.2600')
    """
    entry = EEIO_FACTORS.get(naics_code)
    if entry is not None:
        return entry["ef"]
    return None


def get_telework_daily_kwh(
    seasonal_adjustment: SeasonalAdjustment = SeasonalAdjustment.NONE,
) -> Decimal:
    """
    Get typical daily telework energy consumption with seasonal adjustment.

    Args:
        seasonal_adjustment: Seasonal adjustment type.

    Returns:
        Daily energy consumption in kWh.

    Example:
        >>> get_telework_daily_kwh()
        Decimal('4.0')
        >>> get_telework_daily_kwh(SeasonalAdjustment.FULL_SEASONAL)
        Decimal('5.00000000')
    """
    base_kwh = TELEWORK_ENERGY_DEFAULTS["total_typical"]
    multiplier = SEASONAL_ADJUSTMENT_MULTIPLIERS[seasonal_adjustment]
    adjusted = base_kwh * multiplier
    return adjusted.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)


def get_cpi_deflator(year: int, base_year: int = 2021) -> Decimal:
    """
    Get the CPI deflator to convert spend from a given year to the base year.

    The deflator converts nominal spend to real (base-year) USD:
      real_usd = nominal_usd / deflator(year) * deflator(base_year)

    Since base_year=2021 has deflator=1.0, the formula simplifies to:
      real_usd = nominal_usd / deflator(year)

    Args:
        year: Year of the spend data.
        base_year: Base year for deflation (default 2021).

    Returns:
        CPI deflator value.

    Raises:
        ValueError: If year is not found in CPI_DEFLATORS.

    Example:
        >>> get_cpi_deflator(2024)
        Decimal('1.1490')
    """
    deflator = CPI_DEFLATORS.get(year)
    if deflator is None:
        raise ValueError(
            f"CPI deflator not available for year {year}. "
            f"Available years: {sorted(CPI_DEFLATORS.keys())}"
        )
    base_deflator = CPI_DEFLATORS.get(base_year)
    if base_deflator is None:
        raise ValueError(
            f"CPI deflator not available for base year {base_year}"
        )
    return deflator


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Metadata
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",

    # Enums (25)
    "CalculationMethod",
    "CommuteMode",
    "VehicleType",
    "FuelType",
    "TransitType",
    "TeleworkFrequency",
    "WorkSchedule",
    "EFSource",
    "ComplianceFramework",
    "DataQualityTier",
    "ProvenanceStage",
    "UncertaintyMethod",
    "DQIDimension",
    "DQIScore",
    "ComplianceStatus",
    "GWPVersion",
    "EmissionGas",
    "CurrencyCode",
    "ExportFormat",
    "BatchStatus",
    "RegionCode",
    "DistanceBand",
    "SurveyMethod",
    "AllocationMethod",
    "SeasonalAdjustment",

    # Constants (16 constant tables + supplementary)
    "GWP_VALUES",
    "VEHICLE_EMISSION_FACTORS",
    "FUEL_EMISSION_FACTORS",
    "TRANSIT_EMISSION_FACTORS",
    "MICRO_MOBILITY_EFS",
    "GRID_EMISSION_FACTORS",
    "WORKING_DAYS_DEFAULTS",
    "AVERAGE_COMMUTE_DISTANCES",
    "DEFAULT_MODE_SHARES",
    "TELEWORK_ENERGY_DEFAULTS",
    "VAN_EMISSION_FACTORS",
    "EEIO_FACTORS",
    "CURRENCY_RATES",
    "CPI_DEFLATORS",
    "DQI_SCORING",
    "DQI_WEIGHTS",
    "UNCERTAINTY_RANGES",
    "FRAMEWORK_REQUIRED_DISCLOSURES",
    "WORK_SCHEDULE_FRACTIONS",
    "TELEWORK_FREQUENCY_FRACTIONS",
    "SEASONAL_ADJUSTMENT_MULTIPLIERS",

    # Input models (11)
    "CommuteInput",
    "FuelBasedCommuteInput",
    "CarpoolInput",
    "TransitInput",
    "TeleworkInput",
    "SurveyResponseInput",
    "SurveyInput",
    "AverageDataInput",
    "SpendInput",
    "EmployeeInput",
    "BatchEmployeeInput",

    # Result models (17)
    "CommuteResult",
    "TeleworkResult",
    "EmployeeResult",
    "SurveyResult",
    "BatchResult",
    "AggregationResult",
    "ModeShareResult",
    "ComplianceCheckResult",
    "UncertaintyResult",
    "DataQualityResult",
    "ProvenanceRecord",
    "ProvenanceChainResult",
    "SpendResult",
    "WorkingDaysResult",
    "GridEFResult",
    "HotSpotResult",
    "MetricsSummary",

    # Helper functions
    "calculate_provenance_hash",
    "get_dqi_classification",
    "convert_currency_to_usd",
    "get_working_days",
    "classify_distance_band",
    "get_grid_ef",
    "get_average_commute_distance",
    "get_default_mode_shares",
    "get_eeio_factor",
    "get_telework_daily_kwh",
    "get_cpi_deflator",
]
