"""
Franchises Agent Models (AGENT-MRV-027)

This module provides comprehensive data models for GHG Protocol Scope 3 Category 14
(Franchises) emissions calculations -- reported by the FRANCHISOR.

Supports:
- 4 calculation methods (franchise-specific, average-data, spend-based, hybrid)
- 10 franchise types (QSR, full-service restaurant, hotel, convenience store, etc.)
- 7 emission sources (stationary combustion, mobile, refrigerants, electricity, etc.)
- EUI benchmarks by franchise type and climate zone (50 combinations)
- Revenue intensity factors from EEIO for 10 franchise industry categories
- Refrigerant GWPs (IPCC AR6) for 10 common refrigerants
- Grid emission factors for 12 countries + 26 eGRID subregions
- Fuel emission factors for 8 fuel types (DEFRA/EPA 2024)
- Hotel energy benchmarks by class and climate zone
- 8 double-counting prevention rules (DC-FRN-001 through DC-FRN-008)
- Data quality indicators (DQI) with 5-dimension scoring
- Uncertainty quantification (Monte Carlo, analytical, IPCC Tier 2)
- Compliance checking for 7 frameworks (GHG Protocol, ISO 14064, CSRD, CDP, SBTi, SB 253, GRI)
- SHA-256 provenance chain with 10-stage pipeline
- Network-level aggregation and data-coverage reporting

Company-owned units MUST be excluded from Scope 3 Category 14 per GHG Protocol
guidance (they belong in Scope 1/2).  DC-FRN-001 enforces this boundary.

All numeric fields use Decimal for precision in regulatory calculations.
All models are frozen (immutable) for audit trail integrity.

Example:
    >>> from greenlang.franchises.models import FranchiseUnitInput, FranchiseType
    >>> unit = FranchiseUnitInput(
    ...     unit_id="FRN-001",
    ...     franchise_type=FranchiseType.QSR_RESTAURANT,
    ...     ownership_type=OwnershipType.FRANCHISED,
    ...     floor_area_m2=Decimal("250"),
    ...     country="US",
    ...     region="SRSO",
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

AGENT_ID: str = "GL-MRV-S3-014"
AGENT_COMPONENT: str = "AGENT-MRV-027"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_frn_"

# ==============================================================================
# ENUMERATIONS
# ==============================================================================


class FranchiseType(str, Enum):
    """Franchise business type affecting energy-use intensity benchmarks."""

    QSR_RESTAURANT = "qsr_restaurant"  # Quick-service restaurant (fast food)
    FULL_SERVICE_RESTAURANT = "full_service_restaurant"  # Sit-down / casual dining
    HOTEL = "hotel"  # Hotel / lodging
    CONVENIENCE_STORE = "convenience_store"  # Convenience store / gas station
    RETAIL_STORE = "retail_store"  # Retail merchandise store
    FITNESS_CENTER = "fitness_center"  # Gym / fitness center
    AUTOMOTIVE_SERVICE = "automotive_service"  # Auto repair / quick lube / car wash
    HEALTHCARE_CLINIC = "healthcare_clinic"  # Urgent care / dental / veterinary
    EDUCATION_CENTER = "education_center"  # Tutoring / training / childcare
    OTHER_SERVICE = "other_service"  # Other service-based franchise


class OwnershipType(str, Enum):
    """Ownership classification for a franchise unit.

    Company-owned units are excluded from Scope 3 Cat 14 (DC-FRN-001).
    """

    FRANCHISED = "franchised"  # Operated by an independent franchisee
    COMPANY_OWNED = "company_owned"  # Operated by the franchisor (Scope 1/2)
    JOINT_VENTURE = "joint_venture"  # Joint venture -- allocation required


class FranchiseAgreementType(str, Enum):
    """Franchise agreement structure affecting reporting obligations."""

    SINGLE_UNIT = "single_unit"  # Single franchise location
    MULTI_UNIT = "multi_unit"  # Multiple locations, one franchisee
    AREA_DEVELOPMENT = "area_development"  # Exclusive territory development rights
    MASTER_FRANCHISE = "master_franchise"  # Sub-franchisor rights in a region


class CalculationMethod(str, Enum):
    """Calculation method for franchise emissions per GHG Protocol."""

    FRANCHISE_SPECIFIC = "franchise_specific"  # Primary energy/refrigerant data per unit
    AVERAGE_DATA = "average_data"  # EUI benchmarks by type and climate zone
    SPEND_BASED = "spend_based"  # EEIO factors applied to revenue/royalty data
    HYBRID = "hybrid"  # Blend of methods across heterogeneous network


class EmissionSource(str, Enum):
    """Emission source categories within a franchise unit."""

    STATIONARY_COMBUSTION = "stationary_combustion"  # On-site fuel combustion (cooking, heating)
    MOBILE_COMBUSTION = "mobile_combustion"  # Delivery fleet / service vehicles
    REFRIGERANT_LEAKAGE = "refrigerant_leakage"  # HVAC and commercial refrigeration leaks
    PROCESS_EMISSIONS = "process_emissions"  # Franchise-specific process emissions
    PURCHASED_ELECTRICITY = "purchased_electricity"  # Grid electricity consumption
    PURCHASED_HEATING = "purchased_heating"  # District heating / steam purchases
    PURCHASED_COOLING = "purchased_cooling"  # District cooling purchases


class FuelType(str, Enum):
    """Fuel types consumed at franchise locations."""

    NATURAL_GAS = "natural_gas"  # Pipeline natural gas (m3 or therms)
    PROPANE = "propane"  # LPG / propane (gallons or litres)
    DIESEL = "diesel"  # Diesel fuel (litres)
    GASOLINE = "gasoline"  # Petrol / gasoline (litres)
    FUEL_OIL = "fuel_oil"  # Heating oil #2 (litres)
    LPG = "lpg"  # Liquefied petroleum gas (litres)
    BIOMASS = "biomass"  # Wood pellets / biomass (kg)
    ELECTRICITY = "electricity"  # Electricity (kWh) -- not a fuel but treated as energy input


class ClimateZone(str, Enum):
    """Koppen-Geiger climate classification zones affecting EUI benchmarks."""

    TROPICAL = "tropical"  # A -- high cooling demand
    ARID = "arid"  # B -- extreme cooling, low heating
    TEMPERATE = "temperate"  # C -- moderate heating/cooling
    CONTINENTAL = "continental"  # D -- high heating demand
    POLAR = "polar"  # E -- extreme heating demand


class EFSource(str, Enum):
    """Emission factor data source for audit trail."""

    DEFRA_2024 = "DEFRA_2024"  # UK DEFRA/DESNZ 2024 conversion factors
    EPA_2024 = "EPA_2024"  # US EPA GHG Emission Factors Hub 2024
    IEA_2024 = "IEA_2024"  # IEA CO2 Emissions from Fuel Combustion 2024
    EGRID_2024 = "EGRID_2024"  # US EPA eGRID 2024 subregional factors
    IPCC_AR6 = "IPCC_AR6"  # IPCC Sixth Assessment Report GWP values
    CUSTOM = "CUSTOM"  # Organization-supplied custom factors


class DataQualityTier(str, Enum):
    """Data quality tiers affecting uncertainty ranges."""

    TIER_1 = "tier_1"  # Primary data / metered / franchise-specific
    TIER_2 = "tier_2"  # Regional benchmarks / survey data
    TIER_3 = "tier_3"  # Global averages / spend-based / defaults


class DQIDimension(str, Enum):
    """Data Quality Indicator dimensions per GHG Protocol Scope 3 guidance."""

    TEMPORAL = "temporal"  # Temporal correlation to reporting year
    GEOGRAPHICAL = "geographical"  # Geographical correlation to franchise location
    TECHNOLOGICAL = "technological"  # Technological correlation to franchise type
    COMPLETENESS = "completeness"  # Fraction of franchise network covered
    RELIABILITY = "reliability"  # Source reliability and verification status


class ComplianceFramework(str, Enum):
    """Regulatory/reporting framework for compliance checks."""

    GHG_PROTOCOL = "ghg_protocol"  # GHG Protocol Scope 3 Standard (Cat 14)
    ISO_14064 = "iso_14064"  # ISO 14064-1:2018
    CSRD_ESRS = "csrd_esrs"  # CSRD ESRS E1 Climate Change
    CDP = "cdp"  # CDP Climate Change Questionnaire
    SBTI = "sbti"  # Science Based Targets initiative
    SB_253 = "sb_253"  # California SB 253
    GRI = "gri"  # GRI 305 Emissions Standard


class ComplianceStatus(str, Enum):
    """Compliance check result status."""

    COMPLIANT = "compliant"  # Fully compliant
    NON_COMPLIANT = "non_compliant"  # Non-compliant
    PARTIAL = "partial"  # Partially compliant / needs attention
    NOT_APPLICABLE = "not_applicable"  # Framework not applicable


class PipelineStage(str, Enum):
    """Processing pipeline stages for provenance tracking."""

    VALIDATE = "validate"  # Input validation and boundary checks
    CLASSIFY = "classify"  # Franchise type/zone classification
    NORMALIZE = "normalize"  # Unit and energy normalization
    RESOLVE_EFS = "resolve_efs"  # Emission factor resolution
    CALCULATE = "calculate"  # Emissions calculation (zero-hallucination)
    ALLOCATE = "allocate"  # JV allocation and pro-rata adjustments
    AGGREGATE = "aggregate"  # Network-level aggregation
    COMPLIANCE = "compliance"  # Compliance checks across frameworks
    PROVENANCE = "provenance"  # Provenance hash chain computation
    SEAL = "seal"  # Final chain sealing and output signing


class UncertaintyMethod(str, Enum):
    """Uncertainty quantification method."""

    MONTE_CARLO = "monte_carlo"  # Monte Carlo simulation (N iterations)
    ANALYTICAL = "analytical"  # Analytical error propagation (GUM)
    IPCC_TIER2 = "ipcc_tier2"  # IPCC Good Practice Tier 2 defaults


class BatchStatus(str, Enum):
    """Batch calculation processing status."""

    PENDING = "pending"  # Awaiting processing
    PROCESSING = "processing"  # Currently processing
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Processing failed


class GWPSource(str, Enum):
    """IPCC Global Warming Potential assessment report version."""

    AR5 = "AR5"  # Fifth Assessment Report (100-year GWP)
    AR6 = "AR6"  # Sixth Assessment Report (100-year GWP)


class DataCollectionMethod(str, Enum):
    """How energy/activity data was collected at the franchise unit."""

    METERED = "metered"  # Utility meter / sub-meter reading
    SURVEY = "survey"  # Franchisee self-reported survey
    ESTIMATED = "estimated"  # Estimated from proxies (floor area, revenue)
    DEFAULT = "default"  # Default benchmark applied (no primary data)


class UnitStatus(str, Enum):
    """Operational status of a franchise unit."""

    ACTIVE = "active"  # Currently operating
    TEMPORARILY_CLOSED = "temporarily_closed"  # Temporarily closed (seasonal, renovation)
    PERMANENTLY_CLOSED = "permanently_closed"  # Permanently closed
    UNDER_CONSTRUCTION = "under_construction"  # Not yet open


class ConsolidationApproach(str, Enum):
    """GHG Protocol organizational boundary approach for franchise reporting."""

    FINANCIAL_CONTROL = "financial_control"  # Financial control approach
    EQUITY_SHARE = "equity_share"  # Equity share approach
    OPERATIONAL_CONTROL = "operational_control"  # Operational control approach


class RefrigerantType(str, Enum):
    """Common refrigerants used in franchise HVAC and commercial refrigeration."""

    R_410A = "R_410A"  # R-410A (HFC blend, most common residential/commercial HVAC)
    R_32 = "R_32"  # R-32 (HFC, lower GWP replacement for R-410A)
    R_134A = "R_134a"  # R-134a (HFC, auto AC and commercial)
    R_404A = "R_404A"  # R-404A (HFC blend, commercial refrigeration)
    R_507A = "R_507A"  # R-507A (HFC blend, low-temp commercial)
    R_22 = "R_22"  # R-22 (HCFC, legacy systems being phased out)
    R_407C = "R_407C"  # R-407C (HFC blend, R-22 replacement)
    R_290 = "R_290"  # R-290 Propane (natural refrigerant)
    R_744 = "R_744"  # R-744 CO2 (natural refrigerant, transcritical)
    R_1234YF = "R_1234yf"  # R-1234yf (HFO, ultra-low GWP)


# ==============================================================================
# CONSTANT TABLES
# ==============================================================================

# Quantization constant: 8 decimal places for regulatory precision
_QUANT_8DP = Decimal("0.00000001")

# ---------------------------------------------------------------------------
# 1. FRANCHISE_EUI_BENCHMARKS
#    Energy Use Intensity (kWh/m2/year) by franchise type x climate zone.
#    Sources: CBECS 2018, ENERGY STAR Portfolio Manager, IEA EBC.
#    QSR ~400-800, hotel ~200-500, retail ~150-350, etc.
# ---------------------------------------------------------------------------
FRANCHISE_EUI_BENCHMARKS: Dict[str, Dict[str, Decimal]] = {
    FranchiseType.QSR_RESTAURANT.value: {
        ClimateZone.TROPICAL.value: Decimal("780"),
        ClimateZone.ARID.value: Decimal("720"),
        ClimateZone.TEMPERATE.value: Decimal("550"),
        ClimateZone.CONTINENTAL.value: Decimal("620"),
        ClimateZone.POLAR.value: Decimal("680"),
    },
    FranchiseType.FULL_SERVICE_RESTAURANT.value: {
        ClimateZone.TROPICAL.value: Decimal("620"),
        ClimateZone.ARID.value: Decimal("580"),
        ClimateZone.TEMPERATE.value: Decimal("440"),
        ClimateZone.CONTINENTAL.value: Decimal("500"),
        ClimateZone.POLAR.value: Decimal("560"),
    },
    FranchiseType.HOTEL.value: {
        ClimateZone.TROPICAL.value: Decimal("480"),
        ClimateZone.ARID.value: Decimal("440"),
        ClimateZone.TEMPERATE.value: Decimal("310"),
        ClimateZone.CONTINENTAL.value: Decimal("380"),
        ClimateZone.POLAR.value: Decimal("450"),
    },
    FranchiseType.CONVENIENCE_STORE.value: {
        ClimateZone.TROPICAL.value: Decimal("520"),
        ClimateZone.ARID.value: Decimal("480"),
        ClimateZone.TEMPERATE.value: Decimal("370"),
        ClimateZone.CONTINENTAL.value: Decimal("420"),
        ClimateZone.POLAR.value: Decimal("470"),
    },
    FranchiseType.RETAIL_STORE.value: {
        ClimateZone.TROPICAL.value: Decimal("340"),
        ClimateZone.ARID.value: Decimal("310"),
        ClimateZone.TEMPERATE.value: Decimal("220"),
        ClimateZone.CONTINENTAL.value: Decimal("260"),
        ClimateZone.POLAR.value: Decimal("300"),
    },
    FranchiseType.FITNESS_CENTER.value: {
        ClimateZone.TROPICAL.value: Decimal("420"),
        ClimateZone.ARID.value: Decimal("390"),
        ClimateZone.TEMPERATE.value: Decimal("300"),
        ClimateZone.CONTINENTAL.value: Decimal("350"),
        ClimateZone.POLAR.value: Decimal("400"),
    },
    FranchiseType.AUTOMOTIVE_SERVICE.value: {
        ClimateZone.TROPICAL.value: Decimal("290"),
        ClimateZone.ARID.value: Decimal("270"),
        ClimateZone.TEMPERATE.value: Decimal("200"),
        ClimateZone.CONTINENTAL.value: Decimal("240"),
        ClimateZone.POLAR.value: Decimal("280"),
    },
    FranchiseType.HEALTHCARE_CLINIC.value: {
        ClimateZone.TROPICAL.value: Decimal("380"),
        ClimateZone.ARID.value: Decimal("350"),
        ClimateZone.TEMPERATE.value: Decimal("270"),
        ClimateZone.CONTINENTAL.value: Decimal("310"),
        ClimateZone.POLAR.value: Decimal("360"),
    },
    FranchiseType.EDUCATION_CENTER.value: {
        ClimateZone.TROPICAL.value: Decimal("310"),
        ClimateZone.ARID.value: Decimal("280"),
        ClimateZone.TEMPERATE.value: Decimal("210"),
        ClimateZone.CONTINENTAL.value: Decimal("250"),
        ClimateZone.POLAR.value: Decimal("290"),
    },
    FranchiseType.OTHER_SERVICE.value: {
        ClimateZone.TROPICAL.value: Decimal("320"),
        ClimateZone.ARID.value: Decimal("290"),
        ClimateZone.TEMPERATE.value: Decimal("220"),
        ClimateZone.CONTINENTAL.value: Decimal("260"),
        ClimateZone.POLAR.value: Decimal("300"),
    },
}

# ---------------------------------------------------------------------------
# 2. FRANCHISE_REVENUE_INTENSITY
#    kgCO2e per dollar of franchise revenue, EEIO-derived.
#    Sources: EPA USEEIO v2.0, Exiobase 3.8.
# ---------------------------------------------------------------------------
FRANCHISE_REVENUE_INTENSITY: Dict[str, Decimal] = {
    FranchiseType.QSR_RESTAURANT.value: Decimal("0.3820"),
    FranchiseType.FULL_SERVICE_RESTAURANT.value: Decimal("0.3150"),
    FranchiseType.HOTEL.value: Decimal("0.1490"),
    FranchiseType.CONVENIENCE_STORE.value: Decimal("0.2730"),
    FranchiseType.RETAIL_STORE.value: Decimal("0.1650"),
    FranchiseType.FITNESS_CENTER.value: Decimal("0.1280"),
    FranchiseType.AUTOMOTIVE_SERVICE.value: Decimal("0.2410"),
    FranchiseType.HEALTHCARE_CLINIC.value: Decimal("0.1920"),
    FranchiseType.EDUCATION_CENTER.value: Decimal("0.1100"),
    FranchiseType.OTHER_SERVICE.value: Decimal("0.1750"),
}

# ---------------------------------------------------------------------------
# 3. COOKING_FUEL_CONSUMPTION
#    Typical cooking fuel profiles for restaurant franchise types.
#    Percentage breakdown of energy by fuel type (sums to ~1.0).
#    Sources: DOE Building Energy Codes, ENERGY STAR food-service studies.
# ---------------------------------------------------------------------------
COOKING_FUEL_CONSUMPTION: Dict[str, Dict[str, Decimal]] = {
    FranchiseType.QSR_RESTAURANT.value: {
        FuelType.NATURAL_GAS.value: Decimal("0.55"),
        FuelType.PROPANE.value: Decimal("0.10"),
        FuelType.ELECTRICITY.value: Decimal("0.35"),
    },
    FranchiseType.FULL_SERVICE_RESTAURANT.value: {
        FuelType.NATURAL_GAS.value: Decimal("0.60"),
        FuelType.PROPANE.value: Decimal("0.05"),
        FuelType.ELECTRICITY.value: Decimal("0.35"),
    },
}

# ---------------------------------------------------------------------------
# 4. REFRIGERATION_LEAKAGE_RATES
#    Annual refrigerant leakage rates by equipment type (fraction per year).
#    Sources: EPA GreenChill, IPCC/TEAP 2005, ASHRAE 15.
# ---------------------------------------------------------------------------
REFRIGERATION_LEAKAGE_RATES: Dict[str, Decimal] = {
    "walk_in_cooler": Decimal("0.15"),  # 15% annual leakage
    "walk_in_freezer": Decimal("0.18"),  # 18% -- higher stress, lower temp
    "reach_in_cooler": Decimal("0.08"),  # 8% -- sealed hermetic systems
    "reach_in_freezer": Decimal("0.10"),  # 10%
    "display_case": Decimal("0.12"),  # 12% -- frequent door cycling
    "ice_machine": Decimal("0.06"),  # 6% -- small charge, sealed
    "rooftop_hvac": Decimal("0.05"),  # 5% -- residential-style
    "split_system_hvac": Decimal("0.04"),  # 4% -- brazed connections
    "chiller_hvac": Decimal("0.02"),  # 2% -- well-maintained commercial
    "vending_machine": Decimal("0.03"),  # 3% -- hermetic, factory-sealed
}

# ---------------------------------------------------------------------------
# 5. GRID_EMISSION_FACTORS
#    Electricity grid factors (kgCO2e/kWh) for 12 countries + 26 US eGRID subregions.
#    Sources: IEA 2024, EPA eGRID 2024.
# ---------------------------------------------------------------------------
GRID_EMISSION_FACTORS: Dict[str, Decimal] = {
    # Countries (national average)
    "US": Decimal("0.3862"),
    "GB": Decimal("0.2071"),
    "CA": Decimal("0.1200"),
    "DE": Decimal("0.3380"),
    "FR": Decimal("0.0520"),
    "JP": Decimal("0.4570"),
    "AU": Decimal("0.6560"),
    "CN": Decimal("0.5550"),
    "IN": Decimal("0.7080"),
    "BR": Decimal("0.0740"),
    "KR": Decimal("0.4590"),
    "MX": Decimal("0.4310"),
    # US eGRID 2024 subregions
    "AKGD": Decimal("0.4277"),
    "AKMS": Decimal("0.2217"),
    "AZNM": Decimal("0.3863"),
    "CAMX": Decimal("0.2285"),
    "ERCT": Decimal("0.3738"),
    "FRCC": Decimal("0.3803"),
    "HIMS": Decimal("0.5118"),
    "HIOA": Decimal("0.6672"),
    "MROE": Decimal("0.5548"),
    "MROW": Decimal("0.4426"),
    "NEWE": Decimal("0.2131"),
    "NWPP": Decimal("0.2629"),
    "NYCW": Decimal("0.2367"),
    "NYLI": Decimal("0.3549"),
    "NYUP": Decimal("0.1259"),
    "PRMS": Decimal("0.5050"),
    "RFCE": Decimal("0.3083"),
    "RFCM": Decimal("0.5291"),
    "RFCW": Decimal("0.4551"),
    "RMPA": Decimal("0.5342"),
    "SPNO": Decimal("0.5107"),
    "SPSO": Decimal("0.4163"),
    "SRMV": Decimal("0.3478"),
    "SRMW": Decimal("0.6325"),
    "SRSO": Decimal("0.3813"),
    "SRTV": Decimal("0.4168"),
}

# ---------------------------------------------------------------------------
# 6. FUEL_EMISSION_FACTORS
#    kgCO2e per unit of fuel -- from DEFRA 2024 and EPA GHG Emission Factors Hub.
#    Units: natural_gas (per m3), propane (per litre), diesel (per litre),
#    gasoline (per litre), fuel_oil (per litre), lpg (per litre),
#    biomass (per kg, CO2 reported as memo), electricity (per kWh, placeholder 0).
# ---------------------------------------------------------------------------
FUEL_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    FuelType.NATURAL_GAS.value: {
        "ef": Decimal("2.02140"),  # kgCO2e per m3
        "wtt": Decimal("0.34360"),  # Well-to-tank per m3
        "unit": Decimal("1"),  # m3
    },
    FuelType.PROPANE.value: {
        "ef": Decimal("1.55370"),  # kgCO2e per litre
        "wtt": Decimal("0.32149"),
        "unit": Decimal("1"),
    },
    FuelType.DIESEL.value: {
        "ef": Decimal("2.70370"),  # kgCO2e per litre
        "wtt": Decimal("0.60927"),
        "unit": Decimal("1"),
    },
    FuelType.GASOLINE.value: {
        "ef": Decimal("2.31480"),  # kgCO2e per litre
        "wtt": Decimal("0.58549"),
        "unit": Decimal("1"),
    },
    FuelType.FUEL_OIL.value: {
        "ef": Decimal("2.96450"),  # kgCO2e per litre
        "wtt": Decimal("0.62370"),
        "unit": Decimal("1"),
    },
    FuelType.LPG.value: {
        "ef": Decimal("1.55370"),  # kgCO2e per litre
        "wtt": Decimal("0.32149"),
        "unit": Decimal("1"),
    },
    FuelType.BIOMASS.value: {
        "ef": Decimal("0.01539"),  # kgCO2e per kg (non-CO2 only, CO2 biogenic memo)
        "wtt": Decimal("0.02050"),
        "unit": Decimal("1"),
    },
    FuelType.ELECTRICITY.value: {
        "ef": Decimal("0"),  # Grid factor resolved separately
        "wtt": Decimal("0"),
        "unit": Decimal("1"),
    },
}

# ---------------------------------------------------------------------------
# 7. REFRIGERANT_GWPS
#    100-year Global Warming Potential values for common franchise refrigerants.
#    Source: IPCC AR6 (2021), Table 7.SM.7.
# ---------------------------------------------------------------------------
REFRIGERANT_GWPS: Dict[str, Decimal] = {
    RefrigerantType.R_410A.value: Decimal("2088"),
    RefrigerantType.R_32.value: Decimal("675"),
    RefrigerantType.R_134A.value: Decimal("1430"),
    RefrigerantType.R_404A.value: Decimal("3922"),
    RefrigerantType.R_507A.value: Decimal("3985"),
    RefrigerantType.R_22.value: Decimal("1810"),
    RefrigerantType.R_407C.value: Decimal("1774"),
    RefrigerantType.R_290.value: Decimal("3"),  # Propane -- very low GWP
    RefrigerantType.R_744.value: Decimal("1"),  # CO2 -- lowest possible
    RefrigerantType.R_1234YF.value: Decimal("4"),  # HFO -- ultra-low GWP
}

# ---------------------------------------------------------------------------
# 8. EEIO_SPEND_FACTORS
#    EEIO emission factors (kgCO2e per USD) by NAICS code for franchise industries.
#    Source: EPA USEEIO v2.0 / Exiobase 3.8.
# ---------------------------------------------------------------------------
EEIO_SPEND_FACTORS: Dict[str, Dict[str, Any]] = {
    "722513": {
        "name": "Limited-service restaurants (QSR)",
        "ef": Decimal("0.3820"),
    },
    "722511": {
        "name": "Full-service restaurants",
        "ef": Decimal("0.3150"),
    },
    "721110": {
        "name": "Hotels and motels",
        "ef": Decimal("0.1490"),
    },
    "445120": {
        "name": "Convenience stores",
        "ef": Decimal("0.2730"),
    },
    "448140": {
        "name": "Family clothing stores",
        "ef": Decimal("0.1650"),
    },
    "713940": {
        "name": "Fitness and recreational sports centers",
        "ef": Decimal("0.1280"),
    },
    "811111": {
        "name": "General automotive repair",
        "ef": Decimal("0.2410"),
    },
    "621111": {
        "name": "Offices of physicians / clinics",
        "ef": Decimal("0.1920"),
    },
    "611691": {
        "name": "Exam preparation and tutoring",
        "ef": Decimal("0.1100"),
    },
    "812990": {
        "name": "All other personal services",
        "ef": Decimal("0.1750"),
    },
}

# ---------------------------------------------------------------------------
# 9. HOTEL_ENERGY_BENCHMARKS
#    kWh/room/year by hotel class and climate zone.
#    Sources: Cornell HCMI, ENERGY STAR Portfolio Manager, IHG Green Engage.
# ---------------------------------------------------------------------------
HOTEL_ENERGY_BENCHMARKS: Dict[str, Dict[str, Decimal]] = {
    "economy": {
        ClimateZone.TROPICAL.value: Decimal("18500"),
        ClimateZone.ARID.value: Decimal("17200"),
        ClimateZone.TEMPERATE.value: Decimal("12800"),
        ClimateZone.CONTINENTAL.value: Decimal("15400"),
        ClimateZone.POLAR.value: Decimal("17800"),
    },
    "midscale": {
        ClimateZone.TROPICAL.value: Decimal("25600"),
        ClimateZone.ARID.value: Decimal("23800"),
        ClimateZone.TEMPERATE.value: Decimal("18200"),
        ClimateZone.CONTINENTAL.value: Decimal("21400"),
        ClimateZone.POLAR.value: Decimal("24200"),
    },
    "upscale": {
        ClimateZone.TROPICAL.value: Decimal("34800"),
        ClimateZone.ARID.value: Decimal("32100"),
        ClimateZone.TEMPERATE.value: Decimal("25000"),
        ClimateZone.CONTINENTAL.value: Decimal("29500"),
        ClimateZone.POLAR.value: Decimal("33200"),
    },
    "luxury": {
        ClimateZone.TROPICAL.value: Decimal("48500"),
        ClimateZone.ARID.value: Decimal("44800"),
        ClimateZone.TEMPERATE.value: Decimal("35600"),
        ClimateZone.CONTINENTAL.value: Decimal("41200"),
        ClimateZone.POLAR.value: Decimal("46500"),
    },
}

# ---------------------------------------------------------------------------
# 10. VEHICLE_EMISSION_FACTORS
#     Delivery fleet EFs by vehicle type (kgCO2e/km).
#     Sources: DEFRA 2024, EPA SmartWay.
# ---------------------------------------------------------------------------
VEHICLE_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "motorcycle": {
        "ef_per_km": Decimal("0.11337"),
        "wtt_per_km": Decimal("0.02867"),
    },
    "small_van": {
        "ef_per_km": Decimal("0.20910"),
        "wtt_per_km": Decimal("0.04845"),
    },
    "medium_van": {
        "ef_per_km": Decimal("0.25720"),
        "wtt_per_km": Decimal("0.05958"),
    },
    "large_van": {
        "ef_per_km": Decimal("0.31190"),
        "wtt_per_km": Decimal("0.07225"),
    },
    "light_truck": {
        "ef_per_km": Decimal("0.36850"),
        "wtt_per_km": Decimal("0.08540"),
    },
    "car_average": {
        "ef_per_km": Decimal("0.27145"),
        "wtt_per_km": Decimal("0.06291"),
    },
    "electric_van": {
        "ef_per_km": Decimal("0.07005"),
        "wtt_per_km": Decimal("0.01479"),
    },
    "cargo_bike": {
        "ef_per_km": Decimal("0.00000"),
        "wtt_per_km": Decimal("0.00000"),
    },
}

# ---------------------------------------------------------------------------
# 11. DC_RULES
#     Double-counting prevention rules for franchise emissions.
#     DC-FRN-001 is CRITICAL: company-owned units belong in Scope 1/2, not Cat 14.
# ---------------------------------------------------------------------------
DC_RULES: Dict[str, Dict[str, str]] = {
    "DC-FRN-001": {
        "rule": "Exclude company-owned units from Scope 3 Category 14",
        "scope": "organizational_boundary",
        "severity": "critical",
        "description": (
            "Company-owned franchise units must be reported under the "
            "franchisor's Scope 1 and Scope 2.  Including them in Cat 14 "
            "would double-count those emissions."
        ),
    },
    "DC-FRN-002": {
        "rule": "Avoid double-counting with Scope 2 purchased electricity",
        "scope": "scope2_overlap",
        "severity": "high",
        "description": (
            "If a franchise unit's electricity is purchased through the "
            "franchisor's consolidated billing, it may already appear in "
            "the franchisor's Scope 2.  Deduct accordingly."
        ),
    },
    "DC-FRN-003": {
        "rule": "Avoid overlap with Category 1 (Purchased Goods & Services)",
        "scope": "cat1_overlap",
        "severity": "high",
        "description": (
            "Franchise fees paid by franchisees to the franchisor are revenue, "
            "not a purchased good/service from the franchisor's perspective.  "
            "Do not report the same activity in both Cat 1 and Cat 14."
        ),
    },
    "DC-FRN-004": {
        "rule": "Joint venture pro-rata allocation",
        "scope": "joint_venture",
        "severity": "medium",
        "description": (
            "For joint-venture franchise units, only include the franchisee's "
            "share of emissions proportional to their equity/control share."
        ),
    },
    "DC-FRN-005": {
        "rule": "Avoid overlap with Category 8 (Upstream Leased Assets)",
        "scope": "cat8_overlap",
        "severity": "medium",
        "description": (
            "If the franchisor leases assets to franchisees, those asset "
            "emissions should be reported in Cat 8, not duplicated in Cat 14."
        ),
    },
    "DC-FRN-006": {
        "rule": "Avoid overlap with Category 13 (Downstream Leased Assets)",
        "scope": "cat13_overlap",
        "severity": "medium",
        "description": (
            "Properties leased by the franchisor to franchisees may fall "
            "under Cat 13 if the franchisor owns the building.  Ensure no "
            "double-counting between Cat 13 and Cat 14."
        ),
    },
    "DC-FRN-007": {
        "rule": "Consolidated billing electricity deduction",
        "scope": "consolidated_billing",
        "severity": "medium",
        "description": (
            "When the franchisor pays utility bills on behalf of franchisees "
            "and reports that electricity in its own Scope 2, deduct the "
            "franchisee portion from Cat 14 to avoid double-counting."
        ),
    },
    "DC-FRN-008": {
        "rule": "Waste from franchise operations vs Cat 5",
        "scope": "cat5_overlap",
        "severity": "low",
        "description": (
            "Waste generated at franchise locations is a franchisee Scope 1/3 "
            "item.  It should not appear in the franchisor's Cat 5 (Waste "
            "Generated in Operations) unless the franchisor controls waste "
            "collection contracts directly."
        ),
    },
}

# ---------------------------------------------------------------------------
# 12. COMPLIANCE_FRAMEWORK_RULES
#     Specific reporting requirements per compliance framework for Cat 14.
# ---------------------------------------------------------------------------
COMPLIANCE_FRAMEWORK_RULES: Dict[str, Dict[str, Any]] = {
    ComplianceFramework.GHG_PROTOCOL.value: {
        "name": "GHG Protocol Scope 3 Standard -- Category 14",
        "required_disclosures": [
            "total_co2e",
            "calculation_method",
            "ef_sources",
            "data_coverage_percentage",
            "boundary_description",
            "exclusions_justified",
            "dqi_score",
        ],
        "boundary_guidance": (
            "Include all franchises operating under the reporting company's "
            "brand(s) not already included in Scope 1 and Scope 2."
        ),
    },
    ComplianceFramework.ISO_14064.value: {
        "name": "ISO 14064-1:2018 -- Indirect GHG emissions",
        "required_disclosures": [
            "total_co2e",
            "uncertainty_analysis",
            "base_year",
            "methodology_reference",
            "organizational_boundary",
        ],
        "boundary_guidance": (
            "Define organizational boundary using financial control, "
            "operational control, or equity share approach."
        ),
    },
    ComplianceFramework.CSRD_ESRS.value: {
        "name": "CSRD ESRS E1 -- Climate Change",
        "required_disclosures": [
            "total_co2e",
            "category_breakdown",
            "methodology",
            "targets_and_progress",
            "transition_plan_alignment",
            "financial_exposure",
        ],
        "boundary_guidance": (
            "Report all material Scope 3 categories.  Franchise emissions "
            "are typically material for franchisor business models."
        ),
    },
    ComplianceFramework.CDP.value: {
        "name": "CDP Climate Change Questionnaire",
        "required_disclosures": [
            "total_co2e",
            "method_used",
            "franchise_count",
            "data_coverage",
            "verification_status",
            "engagement_activities",
        ],
        "boundary_guidance": (
            "Report Scope 3 Cat 14 if relevant.  CDP expects disclosure "
            "of engagement with franchisees on emissions reduction."
        ),
    },
    ComplianceFramework.SBTI.value: {
        "name": "Science Based Targets initiative",
        "required_disclosures": [
            "total_co2e",
            "target_coverage_percentage",
            "base_year_emissions",
            "progress_tracking",
            "engagement_target",
        ],
        "boundary_guidance": (
            "If Cat 14 exceeds the 40% Scope 3 threshold, a Scope 3 target "
            "covering franchises is required."
        ),
    },
    ComplianceFramework.SB_253.value: {
        "name": "California SB 253",
        "required_disclosures": [
            "total_co2e",
            "methodology",
            "assurance_opinion",
            "reporting_entity",
        ],
        "boundary_guidance": (
            "Report all Scope 3 categories.  Third-party assurance of "
            "Scope 3 required by 2030."
        ),
    },
    ComplianceFramework.GRI.value: {
        "name": "GRI 305 -- Emissions",
        "required_disclosures": [
            "total_co2e",
            "gases_included",
            "base_year",
            "standards_and_methodologies",
            "consolidation_approach",
        ],
        "boundary_guidance": (
            "Report Scope 3 emissions using a recognized methodology.  "
            "Disclose the consolidation approach and any exclusions."
        ),
    },
}

# ---------------------------------------------------------------------------
# 13. DQI_SCORING
#     5 dimensions x 3 tiers scoring matrix.
#     Score: 1 (low) to 5 (high).
# ---------------------------------------------------------------------------
DQI_SCORING: Dict[str, Dict[str, Decimal]] = {
    DQIDimension.TEMPORAL.value: {
        DataQualityTier.TIER_1.value: Decimal("5"),  # Same reporting year, metered
        DataQualityTier.TIER_2.value: Decimal("3"),  # Within 3 years
        DataQualityTier.TIER_3.value: Decimal("1"),  # Older than 3 years / default
    },
    DQIDimension.GEOGRAPHICAL.value: {
        DataQualityTier.TIER_1.value: Decimal("5"),  # Same country/region
        DataQualityTier.TIER_2.value: Decimal("3"),  # Same continent
        DataQualityTier.TIER_3.value: Decimal("1"),  # Global average
    },
    DQIDimension.TECHNOLOGICAL.value: {
        DataQualityTier.TIER_1.value: Decimal("5"),  # Same franchise type/equipment
        DataQualityTier.TIER_2.value: Decimal("3"),  # Similar building type
        DataQualityTier.TIER_3.value: Decimal("1"),  # Generic commercial average
    },
    DQIDimension.COMPLETENESS.value: {
        DataQualityTier.TIER_1.value: Decimal("5"),  # > 90% coverage
        DataQualityTier.TIER_2.value: Decimal("3"),  # 50-90% coverage
        DataQualityTier.TIER_3.value: Decimal("1"),  # < 50% coverage
    },
    DQIDimension.RELIABILITY.value: {
        DataQualityTier.TIER_1.value: Decimal("5"),  # Verified / third-party assured
        DataQualityTier.TIER_2.value: Decimal("3"),  # Peer reviewed / internal audit
        DataQualityTier.TIER_3.value: Decimal("1"),  # Non-verified estimate
    },
}

# DQI dimension weights (sum to 1.0)
DQI_WEIGHTS: Dict[str, Decimal] = {
    DQIDimension.TEMPORAL.value: Decimal("0.20"),
    DQIDimension.GEOGRAPHICAL.value: Decimal("0.20"),
    DQIDimension.TECHNOLOGICAL.value: Decimal("0.20"),
    DQIDimension.COMPLETENESS.value: Decimal("0.25"),
    DQIDimension.RELIABILITY.value: Decimal("0.15"),
}

# ---------------------------------------------------------------------------
# 14. UNCERTAINTY_RANGES
#     Half-width of 95% CI as fraction, by calculation method x data quality tier.
# ---------------------------------------------------------------------------
UNCERTAINTY_RANGES: Dict[str, Dict[str, Decimal]] = {
    CalculationMethod.FRANCHISE_SPECIFIC.value: {
        DataQualityTier.TIER_1.value: Decimal("0.10"),
        DataQualityTier.TIER_2.value: Decimal("0.15"),
        DataQualityTier.TIER_3.value: Decimal("0.20"),
    },
    CalculationMethod.AVERAGE_DATA.value: {
        DataQualityTier.TIER_1.value: Decimal("0.20"),
        DataQualityTier.TIER_2.value: Decimal("0.30"),
        DataQualityTier.TIER_3.value: Decimal("0.40"),
    },
    CalculationMethod.SPEND_BASED.value: {
        DataQualityTier.TIER_1.value: Decimal("0.35"),
        DataQualityTier.TIER_2.value: Decimal("0.50"),
        DataQualityTier.TIER_3.value: Decimal("0.60"),
    },
    CalculationMethod.HYBRID.value: {
        DataQualityTier.TIER_1.value: Decimal("0.15"),
        DataQualityTier.TIER_2.value: Decimal("0.25"),
        DataQualityTier.TIER_3.value: Decimal("0.35"),
    },
}

# ---------------------------------------------------------------------------
# 15. COUNTRY_CLIMATE_ZONES
#     Country ISO alpha-2 code -> dominant climate zone.
#     30+ countries mapped for EUI benchmark selection.
# ---------------------------------------------------------------------------
COUNTRY_CLIMATE_ZONES: Dict[str, str] = {
    "US": ClimateZone.TEMPERATE.value,  # Varies widely; temperate as default
    "GB": ClimateZone.TEMPERATE.value,
    "CA": ClimateZone.CONTINENTAL.value,
    "DE": ClimateZone.TEMPERATE.value,
    "FR": ClimateZone.TEMPERATE.value,
    "ES": ClimateZone.TEMPERATE.value,
    "IT": ClimateZone.TEMPERATE.value,
    "NL": ClimateZone.TEMPERATE.value,
    "BE": ClimateZone.TEMPERATE.value,
    "CH": ClimateZone.CONTINENTAL.value,
    "AT": ClimateZone.CONTINENTAL.value,
    "SE": ClimateZone.CONTINENTAL.value,
    "NO": ClimateZone.CONTINENTAL.value,
    "FI": ClimateZone.CONTINENTAL.value,
    "DK": ClimateZone.TEMPERATE.value,
    "JP": ClimateZone.TEMPERATE.value,
    "KR": ClimateZone.TEMPERATE.value,
    "CN": ClimateZone.TEMPERATE.value,
    "AU": ClimateZone.ARID.value,
    "IN": ClimateZone.TROPICAL.value,
    "BR": ClimateZone.TROPICAL.value,
    "MX": ClimateZone.TROPICAL.value,
    "SA": ClimateZone.ARID.value,
    "AE": ClimateZone.ARID.value,
    "QA": ClimateZone.ARID.value,
    "SG": ClimateZone.TROPICAL.value,
    "TH": ClimateZone.TROPICAL.value,
    "MY": ClimateZone.TROPICAL.value,
    "ID": ClimateZone.TROPICAL.value,
    "PH": ClimateZone.TROPICAL.value,
    "ZA": ClimateZone.TEMPERATE.value,
    "NG": ClimateZone.TROPICAL.value,
    "KE": ClimateZone.TROPICAL.value,
    "RU": ClimateZone.CONTINENTAL.value,
    "PL": ClimateZone.CONTINENTAL.value,
    "IS": ClimateZone.POLAR.value,
}


# ==============================================================================
# PYDANTIC INPUT MODELS
# ==============================================================================


class CookingEnergyInput(BaseModel):
    """
    Restaurant-specific cooking energy data for franchise-specific calculations.

    Captures fuel consumption by type for commercial kitchen equipment
    (fryers, ovens, grills, etc.).

    Example:
        >>> cooking = CookingEnergyInput(
        ...     cooking_type="commercial_kitchen",
        ...     natural_gas_therms=Decimal("850"),
        ...     propane_gallons=Decimal("120"),
        ...     electricity_kwh=Decimal("35000"),
        ...     fryer_count=4,
        ...     oven_count=2,
        ... )
    """

    cooking_type: str = Field(
        default="commercial_kitchen",
        description="Kitchen type identifier (commercial_kitchen, prep_only, etc.)"
    )
    natural_gas_therms: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Annual natural gas consumption in therms"
    )
    propane_gallons: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Annual propane consumption in US gallons"
    )
    electricity_kwh: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Annual electricity consumption for cooking equipment (kWh)"
    )
    fryer_count: int = Field(
        default=0, ge=0,
        description="Number of deep fryers (for benchmark validation)"
    )
    oven_count: int = Field(
        default=0, ge=0,
        description="Number of ovens (for benchmark validation)"
    )

    model_config = ConfigDict(frozen=True)


class RefrigerationInput(BaseModel):
    """
    Refrigerant system data for a franchise unit.

    Models a single refrigeration or HVAC system with its charge
    and estimated annual leakage rate.

    Example:
        >>> refrig = RefrigerationInput(
        ...     system_type="walk_in_cooler",
        ...     refrigerant_type=RefrigerantType.R_404A,
        ...     charge_kg=Decimal("12.5"),
        ...     annual_leakage_rate=Decimal("0.15"),
        ...     operating_hours=Decimal("8760"),
        ... )
    """

    system_type: str = Field(
        ..., description="Equipment type (walk_in_cooler, rooftop_hvac, etc.)"
    )
    refrigerant_type: RefrigerantType = Field(
        ..., description="Refrigerant gas type"
    )
    charge_kg: Decimal = Field(
        ..., gt=0,
        description="Total refrigerant charge in kilograms"
    )
    annual_leakage_rate: Optional[Decimal] = Field(
        default=None, ge=0, le=1,
        description=(
            "Annual leakage rate as fraction (0-1).  If None, the default "
            "rate from REFRIGERATION_LEAKAGE_RATES is used."
        )
    )
    operating_hours: Decimal = Field(
        default=Decimal("8760"), gt=0, le=8784,
        description="Annual operating hours (max 8784 for leap year)"
    )

    model_config = ConfigDict(frozen=True)

    @validator("system_type")
    def validate_system_type(cls, v: str) -> str:
        """Normalize system type to lowercase."""
        return v.strip().lower().replace(" ", "_")


class DeliveryFleetInput(BaseModel):
    """
    Delivery / service vehicle fleet data for a franchise unit.

    Models one vehicle class in the unit's delivery fleet.

    Example:
        >>> fleet = DeliveryFleetInput(
        ...     vehicle_type="small_van",
        ...     fuel_type=FuelType.DIESEL,
        ...     annual_distance_km=Decimal("18000"),
        ...     fuel_consumption_l=Decimal("2200"),
        ...     vehicle_count=3,
        ... )
    """

    vehicle_type: str = Field(
        ..., description="Vehicle type (motorcycle, small_van, medium_van, etc.)"
    )
    fuel_type: FuelType = Field(
        default=FuelType.DIESEL,
        description="Primary fuel type for this vehicle class"
    )
    annual_distance_km: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Annual distance driven per vehicle (km)"
    )
    fuel_consumption_l: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Annual fuel consumption per vehicle (litres)"
    )
    vehicle_count: int = Field(
        default=1, ge=1, le=500,
        description="Number of vehicles of this type"
    )

    model_config = ConfigDict(frozen=True)

    @validator("vehicle_type")
    def validate_vehicle_type(cls, v: str) -> str:
        """Normalize vehicle type to lowercase."""
        return v.strip().lower().replace(" ", "_")


class HotelOperationsInput(BaseModel):
    """
    Hotel-specific operational data for hotel franchise units.

    Enables benchmark-based estimation when metered data is unavailable.

    Example:
        >>> hotel = HotelOperationsInput(
        ...     class_type="upscale",
        ...     room_count=180,
        ...     occupancy_rate=Decimal("0.72"),
        ...     has_laundry=True,
        ...     has_pool=True,
        ...     has_restaurant=True,
        ...     has_spa=False,
        ... )
    """

    class_type: str = Field(
        default="midscale",
        description="Hotel class (economy, midscale, upscale, luxury)"
    )
    room_count: int = Field(
        ..., gt=0, le=5000,
        description="Number of guest rooms"
    )
    occupancy_rate: Decimal = Field(
        default=Decimal("0.65"), ge=0, le=1,
        description="Annual average occupancy rate (0-1)"
    )
    has_laundry: bool = Field(
        default=False,
        description="On-site laundry facility (adds ~12% energy)"
    )
    has_pool: bool = Field(
        default=False,
        description="On-site swimming pool (adds ~8% energy)"
    )
    has_restaurant: bool = Field(
        default=False,
        description="On-site restaurant / food service (adds ~15% energy)"
    )
    has_spa: bool = Field(
        default=False,
        description="On-site spa / wellness center (adds ~10% energy)"
    )

    model_config = ConfigDict(frozen=True)

    @validator("class_type")
    def validate_class_type(cls, v: str) -> str:
        """Normalize and validate hotel class."""
        v_lower = v.strip().lower()
        valid_classes = {"economy", "midscale", "upscale", "luxury"}
        if v_lower not in valid_classes:
            raise ValueError(
                f"Hotel class must be one of {valid_classes}, got '{v}'"
            )
        return v_lower


class FranchiseUnitInput(BaseModel):
    """
    Unit-level input for a single franchise location.

    Contains all data required for franchise-specific, average-data,
    or hybrid calculations.  The pipeline selects the appropriate method
    based on data availability.

    Example:
        >>> unit = FranchiseUnitInput(
        ...     unit_id="FRN-0042",
        ...     franchise_type=FranchiseType.QSR_RESTAURANT,
        ...     ownership_type=OwnershipType.FRANCHISED,
        ...     agreement_type=FranchiseAgreementType.SINGLE_UNIT,
        ...     floor_area_m2=Decimal("280"),
        ...     country="US",
        ...     region="SRSO",
        ...     electricity_kwh=Decimal("145000"),
        ...     natural_gas_m3=Decimal("8500"),
        ... )
    """

    unit_id: str = Field(
        ..., min_length=1, max_length=64,
        description="Unique identifier for the franchise unit"
    )
    franchise_type: FranchiseType = Field(
        ..., description="Business type classification"
    )
    ownership_type: OwnershipType = Field(
        ..., description="Ownership structure (franchised, company-owned, JV)"
    )
    agreement_type: FranchiseAgreementType = Field(
        default=FranchiseAgreementType.SINGLE_UNIT,
        description="Type of franchise agreement"
    )
    unit_status: UnitStatus = Field(
        default=UnitStatus.ACTIVE,
        description="Operational status of the unit"
    )
    floor_area_m2: Optional[Decimal] = Field(
        default=None, gt=0,
        description="Gross floor area in square metres"
    )
    country: str = Field(
        ..., min_length=2, max_length=3,
        description="Country ISO alpha-2 code (e.g., 'US', 'GB')"
    )
    region: Optional[str] = Field(
        default=None, max_length=10,
        description="Sub-national region or eGRID subregion code"
    )
    climate_zone: Optional[ClimateZone] = Field(
        default=None,
        description="Override climate zone (auto-detected from country if None)"
    )

    # Energy data -- franchise-specific inputs
    electricity_kwh: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Annual purchased electricity (kWh)"
    )
    natural_gas_m3: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Annual natural gas consumption (m3)"
    )
    propane_litres: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Annual propane consumption (litres)"
    )
    diesel_litres: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Annual diesel consumption (litres)"
    )
    fuel_oil_litres: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Annual fuel oil consumption (litres)"
    )
    district_heating_kwh: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Annual district heating purchased (kWh)"
    )
    district_cooling_kwh: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Annual district cooling purchased (kWh)"
    )

    # Cooking-specific (QSR/FSR)
    cooking_energy: Optional[CookingEnergyInput] = Field(
        default=None,
        description="Restaurant cooking energy breakdown (QSR/FSR only)"
    )

    # Refrigeration systems
    refrigeration_systems: Optional[List[RefrigerationInput]] = Field(
        default=None,
        description="List of refrigeration / HVAC systems with charge data"
    )

    # Delivery fleet
    delivery_vehicles: Optional[List[DeliveryFleetInput]] = Field(
        default=None,
        description="Delivery fleet vehicles for mobile combustion"
    )

    # Hotel-specific
    hotel_operations: Optional[HotelOperationsInput] = Field(
        default=None,
        description="Hotel operational data (hotel franchises only)"
    )

    # Financial / revenue data for spend-based method
    annual_revenue_usd: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Annual unit revenue in USD (for spend-based method)"
    )
    annual_royalty_usd: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Annual royalty fees paid to franchisor in USD"
    )
    occupancy_rate: Optional[Decimal] = Field(
        default=None, ge=0, le=1,
        description="Average occupancy rate (hotels) or seat utilization"
    )

    # Data collection metadata
    data_collection_method: DataCollectionMethod = Field(
        default=DataCollectionMethod.DEFAULT,
        description="How the activity data was collected"
    )
    reporting_year: int = Field(
        default=2024, ge=2015, le=2035,
        description="Reporting year for temporal alignment"
    )
    jv_equity_share: Optional[Decimal] = Field(
        default=None, ge=0, le=1,
        description="Equity share for joint venture units (0-1)"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)

    @validator("country")
    def validate_country(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        return v.strip().upper()

    @validator("region")
    def validate_region(cls, v: Optional[str]) -> Optional[str]:
        """Normalize region code to uppercase."""
        if v is not None:
            return v.strip().upper()
        return v


class FranchiseNetworkInput(BaseModel):
    """
    Network-level input for an entire franchise brand/system.

    Wraps multiple FranchiseUnitInput records with network metadata.

    Example:
        >>> network = FranchiseNetworkInput(
        ...     brand_name="BurgerQueen",
        ...     units=[unit1, unit2, unit3],
        ...     consolidation_approach=ConsolidationApproach.OPERATIONAL_CONTROL,
        ...     reporting_period="2024",
        ...     total_royalty_income_usd=Decimal("45000000"),
        ...     data_coverage_target=Decimal("0.80"),
        ... )
    """

    brand_name: str = Field(
        ..., min_length=1, max_length=256,
        description="Franchise brand / system name"
    )
    units: List[FranchiseUnitInput] = Field(
        ..., min_length=1,
        description="List of franchise unit inputs"
    )
    consolidation_approach: ConsolidationApproach = Field(
        default=ConsolidationApproach.OPERATIONAL_CONTROL,
        description="GHG Protocol organizational boundary approach"
    )
    reporting_period: str = Field(
        ..., min_length=4, max_length=16,
        description="Reporting period (e.g., '2024', '2024-Q4')"
    )
    total_royalty_income_usd: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Total annual royalty income in USD (network-wide)"
    )
    data_coverage_target: Decimal = Field(
        default=Decimal("0.80"), ge=0, le=1,
        description="Target data coverage for franchise-specific method (0-1)"
    )
    base_year: Optional[int] = Field(
        default=None, ge=2015, le=2035,
        description="Base year for SBTi / trend analysis"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy"
    )

    model_config = ConfigDict(frozen=True)

    @validator("brand_name")
    def validate_brand_name(cls, v: str) -> str:
        """Strip whitespace from brand name."""
        return v.strip()


# ==============================================================================
# PYDANTIC RESULT MODELS
# ==============================================================================


class DataQualityScore(BaseModel):
    """
    Data quality assessment result across 5 DQI dimensions.

    The composite score is a weighted average of individual dimension scores.
    """

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
    completeness: Decimal = Field(
        ..., ge=1, le=5,
        description="Data completeness score (1-5)"
    )
    reliability: Decimal = Field(
        ..., ge=1, le=5,
        description="Source reliability score (1-5)"
    )
    composite: Decimal = Field(
        ..., ge=1, le=5,
        description="Weighted composite DQI score (1-5)"
    )
    classification: str = Field(
        ..., description="Quality label (Excellent/Good/Fair/Poor/Very Poor)"
    )

    model_config = ConfigDict(frozen=True)


class UncertaintyResult(BaseModel):
    """Result from uncertainty quantification for an emissions estimate."""

    method: UncertaintyMethod = Field(
        ..., description="Uncertainty method used"
    )
    confidence_interval: Decimal = Field(
        ..., description="Confidence level (e.g., 0.95 for 95%)"
    )
    lower_bound: Decimal = Field(
        ..., description="Lower bound of CI (kgCO2e)"
    )
    upper_bound: Decimal = Field(
        ..., description="Upper bound of CI (kgCO2e)"
    )
    coefficient_of_variation: Decimal = Field(
        ..., ge=0,
        description="CV = std_dev / mean (dimensionless)"
    )
    mean: Decimal = Field(
        ..., description="Mean emissions estimate (kgCO2e)"
    )
    std_dev: Decimal = Field(
        ..., ge=0,
        description="Standard deviation (kgCO2e)"
    )

    model_config = ConfigDict(frozen=True)


class FranchiseCalculationResult(BaseModel):
    """
    Emissions calculation result for a single franchise unit.

    Contains the total and per-source breakdown, method used,
    data quality, and provenance hash.
    """

    unit_id: str = Field(
        ..., description="Franchise unit identifier"
    )
    franchise_type: FranchiseType = Field(
        ..., description="Business type of the unit"
    )
    ownership_type: OwnershipType = Field(
        ..., description="Ownership classification"
    )
    calculation_method: CalculationMethod = Field(
        ..., description="Calculation method applied"
    )
    total_emissions_kgco2e: Decimal = Field(
        ..., ge=0,
        description="Total emissions (kgCO2e)"
    )

    # Per-source breakdown
    stationary_combustion_kgco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Stationary combustion emissions (kgCO2e)"
    )
    mobile_combustion_kgco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Mobile combustion / delivery fleet emissions (kgCO2e)"
    )
    refrigerant_leakage_kgco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Refrigerant leakage emissions (kgCO2e)"
    )
    purchased_electricity_kgco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Purchased electricity emissions (kgCO2e)"
    )
    purchased_heating_kgco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Purchased heating / steam emissions (kgCO2e)"
    )
    purchased_cooling_kgco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Purchased cooling emissions (kgCO2e)"
    )
    process_emissions_kgco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Process emissions (kgCO2e)"
    )

    # Metadata
    ef_sources: List[str] = Field(
        default_factory=list,
        description="Emission factor sources used"
    )
    data_quality: Optional[DataQualityScore] = Field(
        default=None,
        description="DQI score for this calculation"
    )
    uncertainty: Optional[UncertaintyResult] = Field(
        default=None,
        description="Uncertainty quantification result"
    )
    dc_rules_applied: List[str] = Field(
        default_factory=list,
        description="Double-counting rules applied (e.g., DC-FRN-001)"
    )
    excluded: bool = Field(
        default=False,
        description="True if unit was excluded (company-owned, DC-FRN-001)"
    )
    exclusion_reason: Optional[str] = Field(
        default=None,
        description="Reason for exclusion"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash for audit trail"
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Calculation processing time in milliseconds"
    )

    model_config = ConfigDict(frozen=True)


class DataCoverageReport(BaseModel):
    """
    Summary of data collection methods across the franchise network.

    Used to assess the quality and reliability of the overall estimate.
    """

    total_units: int = Field(
        ..., ge=0,
        description="Total units in the network"
    )
    active_units: int = Field(
        ..., ge=0,
        description="Active units included in calculation"
    )
    metered_count: int = Field(
        default=0, ge=0,
        description="Units with metered / primary data"
    )
    survey_count: int = Field(
        default=0, ge=0,
        description="Units with survey-reported data"
    )
    estimated_count: int = Field(
        default=0, ge=0,
        description="Units with estimated / proxy data"
    )
    default_count: int = Field(
        default=0, ge=0,
        description="Units using default benchmarks"
    )
    excluded_count: int = Field(
        default=0, ge=0,
        description="Units excluded (company-owned, inactive)"
    )
    coverage_percentage: Decimal = Field(
        ..., ge=0, le=100,
        description="Percentage of active units with primary data"
    )
    meets_target: bool = Field(
        ..., description="Whether coverage meets the data_coverage_target"
    )

    model_config = ConfigDict(frozen=True)


class NetworkAggregationResult(BaseModel):
    """
    Network-level aggregated emissions for all franchise units.
    """

    brand_name: str = Field(
        ..., description="Franchise brand name"
    )
    reporting_period: str = Field(
        ..., description="Reporting period"
    )
    total_emissions_kgco2e: Decimal = Field(
        ..., ge=0,
        description="Total network emissions (kgCO2e)"
    )
    total_emissions_tco2e: Decimal = Field(
        ..., ge=0,
        description="Total network emissions (tCO2e)"
    )
    unit_count: int = Field(
        ..., ge=0,
        description="Number of units included"
    )
    average_per_unit_kgco2e: Decimal = Field(
        ..., ge=0,
        description="Average emissions per unit (kgCO2e)"
    )
    method_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of units by calculation method"
    )
    coverage_report: Optional[DataCoverageReport] = Field(
        default=None,
        description="Data coverage report"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 hash of the aggregated result"
    )

    model_config = ConfigDict(frozen=True)


class ComplianceResult(BaseModel):
    """Compliance check result for a single framework."""

    framework: ComplianceFramework = Field(
        ..., description="Framework evaluated"
    )
    status: ComplianceStatus = Field(
        ..., description="Compliance status"
    )
    score: Decimal = Field(
        ..., ge=0, le=100,
        description="Compliance score (0-100)"
    )
    findings: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Specific findings / gaps identified"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable recommendations"
    )
    required_disclosures: List[str] = Field(
        default_factory=list,
        description="Disclosures required by this framework"
    )
    disclosures_present: List[str] = Field(
        default_factory=list,
        description="Disclosures present in the calculation"
    )

    model_config = ConfigDict(frozen=True)


class ProvenanceRecord(BaseModel):
    """Single record in the SHA-256 provenance chain."""

    record_id: str = Field(
        ..., description="Unique record identifier"
    )
    sha256_hash: str = Field(
        ..., min_length=64, max_length=64,
        description="SHA-256 hash of this stage's output"
    )
    parent_hash: Optional[str] = Field(
        default=None, min_length=64, max_length=64,
        description="SHA-256 hash of the previous stage (None for first)"
    )
    chain_hash: str = Field(
        ..., min_length=64, max_length=64,
        description="Cumulative chain hash (parent_hash + sha256_hash)"
    )
    timestamp: str = Field(
        ..., description="ISO 8601 timestamp of computation"
    )
    operation: str = Field(
        ..., description="Pipeline stage / operation name"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Stage-specific metadata"
    )

    model_config = ConfigDict(frozen=True)


class AggregationResult(BaseModel):
    """
    Multi-dimensional aggregation of franchise emissions for reporting.
    """

    period: str = Field(
        ..., description="Reporting period"
    )
    total_emissions_kgco2e: Decimal = Field(
        ..., ge=0,
        description="Total emissions (kgCO2e)"
    )
    total_emissions_tco2e: Decimal = Field(
        ..., ge=0,
        description="Total emissions (tCO2e)"
    )
    by_franchise_type: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions breakdown by franchise type (kgCO2e)"
    )
    by_region: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions breakdown by country/region (kgCO2e)"
    )
    by_method: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions breakdown by calculation method (kgCO2e)"
    )
    by_emission_source: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions breakdown by source category (kgCO2e)"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 hash of the aggregation"
    )

    model_config = ConfigDict(frozen=True)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def validate_franchise_type(franchise_type: str) -> bool:
    """
    Check whether a string is a valid FranchiseType enum value.

    Args:
        franchise_type: String to validate.

    Returns:
        True if valid, False otherwise.

    Example:
        >>> validate_franchise_type("qsr_restaurant")
        True
        >>> validate_franchise_type("unknown")
        False
    """
    valid_values = {ft.value for ft in FranchiseType}
    return franchise_type in valid_values


def validate_ownership_boundary(ownership_type: OwnershipType) -> bool:
    """
    Validate that a unit should be included in Scope 3 Cat 14.

    Company-owned units are excluded per DC-FRN-001.

    Args:
        ownership_type: Ownership classification.

    Returns:
        True if the unit belongs in Cat 14, False if it should be excluded.

    Example:
        >>> validate_ownership_boundary(OwnershipType.FRANCHISED)
        True
        >>> validate_ownership_boundary(OwnershipType.COMPANY_OWNED)
        False
    """
    return ownership_type != OwnershipType.COMPANY_OWNED


def validate_consolidation_approach(approach: ConsolidationApproach) -> bool:
    """
    Validate that the consolidation approach is recognized.

    Args:
        approach: Consolidation approach enum value.

    Returns:
        True if valid.

    Example:
        >>> validate_consolidation_approach(ConsolidationApproach.OPERATIONAL_CONTROL)
        True
    """
    return approach in (
        ConsolidationApproach.FINANCIAL_CONTROL,
        ConsolidationApproach.EQUITY_SHARE,
        ConsolidationApproach.OPERATIONAL_CONTROL,
    )


def calculate_pro_rata_factor(
    equity_share: Optional[Decimal],
    consolidation: ConsolidationApproach,
) -> Decimal:
    """
    Calculate the pro-rata allocation factor for joint venture units.

    For operational-control and financial-control approaches, allocation
    is either 100% (if control) or 0%.  For equity-share, it equals the
    equity stake.

    Args:
        equity_share: Equity share fraction (0-1), required for equity_share approach.
        consolidation: Consolidation approach.

    Returns:
        Pro-rata factor as Decimal (0-1).

    Raises:
        ValueError: If equity_share is None when required.

    Example:
        >>> calculate_pro_rata_factor(Decimal("0.50"), ConsolidationApproach.EQUITY_SHARE)
        Decimal('0.50')
        >>> calculate_pro_rata_factor(None, ConsolidationApproach.OPERATIONAL_CONTROL)
        Decimal('1')
    """
    if consolidation == ConsolidationApproach.EQUITY_SHARE:
        if equity_share is None:
            raise ValueError(
                "equity_share is required when consolidation_approach is 'equity_share'"
            )
        return equity_share
    # Operational or financial control: full inclusion
    return Decimal("1")


def normalize_floor_area(area_sqft: Decimal) -> Decimal:
    """
    Convert floor area from square feet to square metres.

    Args:
        area_sqft: Area in square feet.

    Returns:
        Area in square metres, quantized to 8 dp.

    Example:
        >>> normalize_floor_area(Decimal("2500"))
        Decimal('232.25760000')
    """
    sqft_to_m2 = Decimal("0.09290304")
    return (area_sqft * sqft_to_m2).quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)


def convert_energy_units(
    value: Decimal,
    from_unit: str,
    to_unit: str = "kwh",
) -> Decimal:
    """
    Convert between common energy units.

    Supported conversions (to kWh):
        therms -> kWh (1 therm = 29.3001 kWh)
        mmbtu -> kWh (1 MMBtu = 293.0710 kWh)
        gj -> kWh (1 GJ = 277.7778 kWh)
        mj -> kWh (1 MJ = 0.27778 kWh)
        kwh -> kWh (identity)

    Args:
        value: Numeric energy value.
        from_unit: Source unit (therms, mmbtu, gj, mj, kwh).
        to_unit: Target unit (only 'kwh' currently supported).

    Returns:
        Converted value, quantized to 8 dp.

    Raises:
        ValueError: If from_unit is not recognized.

    Example:
        >>> convert_energy_units(Decimal("100"), "therms")
        Decimal('2930.01000000')
    """
    conversion_to_kwh: Dict[str, Decimal] = {
        "therms": Decimal("29.3001"),
        "mmbtu": Decimal("293.0710"),
        "gj": Decimal("277.7778"),
        "mj": Decimal("0.27778"),
        "kwh": Decimal("1"),
    }
    from_lower = from_unit.strip().lower()
    if from_lower not in conversion_to_kwh:
        raise ValueError(
            f"Unknown energy unit '{from_unit}'.  "
            f"Supported: {list(conversion_to_kwh.keys())}"
        )
    result = value * conversion_to_kwh[from_lower]
    return result.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP)


def get_climate_zone(country: str) -> str:
    """
    Look up the dominant climate zone for a country.

    Falls back to 'temperate' if the country is not in COUNTRY_CLIMATE_ZONES.

    Args:
        country: ISO alpha-2 country code (uppercase).

    Returns:
        ClimateZone value string.

    Example:
        >>> get_climate_zone("IN")
        'tropical'
        >>> get_climate_zone("ZZ")
        'temperate'
    """
    return COUNTRY_CLIMATE_ZONES.get(
        country.upper(), ClimateZone.TEMPERATE.value
    )


def get_grid_ef(country: str, region: Optional[str] = None) -> Decimal:
    """
    Get electricity grid emission factor for a location.

    Looks up the eGRID subregion first (if provided), then the country.
    Falls back to the US national average if neither is found.

    Args:
        country: ISO alpha-2 country code.
        region: Optional eGRID subregion code (US only).

    Returns:
        Grid emission factor in kgCO2e/kWh.

    Example:
        >>> get_grid_ef("US", "SRSO")
        Decimal('0.3813')
        >>> get_grid_ef("GB")
        Decimal('0.2071')
    """
    if region and region.upper() in GRID_EMISSION_FACTORS:
        return GRID_EMISSION_FACTORS[region.upper()]
    country_upper = country.upper()
    return GRID_EMISSION_FACTORS.get(country_upper, Decimal("0.3862"))


def get_fuel_ef(fuel_type: str) -> Decimal:
    """
    Get fuel emission factor (kgCO2e per unit).

    Args:
        fuel_type: FuelType value string.

    Returns:
        Emission factor from FUEL_EMISSION_FACTORS, or Decimal("0") if not found.

    Example:
        >>> get_fuel_ef("natural_gas")
        Decimal('2.02140')
    """
    entry = FUEL_EMISSION_FACTORS.get(fuel_type)
    if entry is not None:
        return entry["ef"]
    return Decimal("0")


def get_eui_benchmark(
    franchise_type: str,
    climate_zone: str,
) -> Decimal:
    """
    Get the Energy Use Intensity benchmark (kWh/m2/year).

    Args:
        franchise_type: FranchiseType value string.
        climate_zone: ClimateZone value string.

    Returns:
        EUI benchmark, or Decimal("300") default.

    Example:
        >>> get_eui_benchmark("qsr_restaurant", "temperate")
        Decimal('550')
    """
    type_data = FRANCHISE_EUI_BENCHMARKS.get(franchise_type)
    if type_data is None:
        return Decimal("300")
    return type_data.get(climate_zone, Decimal("300"))


def is_company_owned(ownership_type: OwnershipType) -> bool:
    """
    Check if a unit is company-owned (excluded from Cat 14).

    Args:
        ownership_type: Unit ownership classification.

    Returns:
        True if company-owned.

    Example:
        >>> is_company_owned(OwnershipType.COMPANY_OWNED)
        True
        >>> is_company_owned(OwnershipType.FRANCHISED)
        False
    """
    return ownership_type == OwnershipType.COMPANY_OWNED


def is_franchised(ownership_type: OwnershipType) -> bool:
    """
    Check if a unit is franchised (included in Cat 14 at 100%).

    Args:
        ownership_type: Unit ownership classification.

    Returns:
        True if independently franchised.

    Example:
        >>> is_franchised(OwnershipType.FRANCHISED)
        True
    """
    return ownership_type == OwnershipType.FRANCHISED


def get_franchise_label(franchise_type: FranchiseType) -> str:
    """
    Return a human-readable label for a franchise type.

    Args:
        franchise_type: FranchiseType enum value.

    Returns:
        Display label string.

    Example:
        >>> get_franchise_label(FranchiseType.QSR_RESTAURANT)
        'Quick-Service Restaurant'
    """
    labels: Dict[FranchiseType, str] = {
        FranchiseType.QSR_RESTAURANT: "Quick-Service Restaurant",
        FranchiseType.FULL_SERVICE_RESTAURANT: "Full-Service Restaurant",
        FranchiseType.HOTEL: "Hotel / Lodging",
        FranchiseType.CONVENIENCE_STORE: "Convenience Store",
        FranchiseType.RETAIL_STORE: "Retail Store",
        FranchiseType.FITNESS_CENTER: "Fitness Center",
        FranchiseType.AUTOMOTIVE_SERVICE: "Automotive Service",
        FranchiseType.HEALTHCARE_CLINIC: "Healthcare Clinic",
        FranchiseType.EDUCATION_CENTER: "Education Center",
        FranchiseType.OTHER_SERVICE: "Other Service",
    }
    return labels.get(franchise_type, franchise_type.value)


def format_emissions_kg(value: Decimal) -> str:
    """
    Format a kgCO2e value as a human-readable string with 2 decimal places.

    Args:
        value: Emissions in kgCO2e.

    Returns:
        Formatted string (e.g., '12,345.67 kgCO2e').

    Example:
        >>> format_emissions_kg(Decimal("12345.6789"))
        '12,345.68 kgCO2e'
    """
    rounded = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{rounded:,.2f} kgCO2e"


def format_emissions_tonnes(value: Decimal) -> str:
    """
    Convert kgCO2e to tCO2e and format with 3 decimal places.

    Args:
        value: Emissions in kgCO2e.

    Returns:
        Formatted string in tCO2e.

    Example:
        >>> format_emissions_tonnes(Decimal("12345.6789"))
        '12.346 tCO2e'
    """
    tonnes = (value / Decimal("1000")).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    )
    return f"{tonnes:,.3f} tCO2e"


def get_data_quality_tier(
    data_collection_method: DataCollectionMethod,
) -> DataQualityTier:
    """
    Map a data collection method to a data quality tier.

    Args:
        data_collection_method: How data was collected.

    Returns:
        Corresponding DataQualityTier.

    Example:
        >>> get_data_quality_tier(DataCollectionMethod.METERED)
        <DataQualityTier.TIER_1: 'tier_1'>
        >>> get_data_quality_tier(DataCollectionMethod.DEFAULT)
        <DataQualityTier.TIER_3: 'tier_3'>
    """
    mapping: Dict[DataCollectionMethod, DataQualityTier] = {
        DataCollectionMethod.METERED: DataQualityTier.TIER_1,
        DataCollectionMethod.SURVEY: DataQualityTier.TIER_2,
        DataCollectionMethod.ESTIMATED: DataQualityTier.TIER_2,
        DataCollectionMethod.DEFAULT: DataQualityTier.TIER_3,
    }
    return mapping.get(data_collection_method, DataQualityTier.TIER_3)


def classify_emission_source(
    source_description: str,
) -> Optional[EmissionSource]:
    """
    Classify a free-text source description into an EmissionSource enum.

    Args:
        source_description: Free text describing the emission source.

    Returns:
        Matched EmissionSource or None if not recognized.

    Example:
        >>> classify_emission_source("natural gas boiler")
        <EmissionSource.STATIONARY_COMBUSTION: 'stationary_combustion'>
        >>> classify_emission_source("delivery van diesel")
        <EmissionSource.MOBILE_COMBUSTION: 'mobile_combustion'>
    """
    desc_lower = source_description.strip().lower()

    stationary_keywords = {"boiler", "furnace", "heater", "oven", "fryer", "grill", "stove"}
    mobile_keywords = {"delivery", "van", "truck", "vehicle", "fleet", "car", "motorcycle"}
    refrigerant_keywords = {"refrigerant", "hvac", "cooler", "freezer", "r-", "r_"}
    electricity_keywords = {"electricity", "grid", "power", "electric"}
    heating_keywords = {"district heating", "steam", "hot water"}
    cooling_keywords = {"district cooling", "chilled water"}

    if any(kw in desc_lower for kw in stationary_keywords):
        return EmissionSource.STATIONARY_COMBUSTION
    if any(kw in desc_lower for kw in mobile_keywords):
        return EmissionSource.MOBILE_COMBUSTION
    if any(kw in desc_lower for kw in refrigerant_keywords):
        return EmissionSource.REFRIGERANT_LEAKAGE
    if any(kw in desc_lower for kw in electricity_keywords):
        return EmissionSource.PURCHASED_ELECTRICITY
    if any(kw in desc_lower for kw in heating_keywords):
        return EmissionSource.PURCHASED_HEATING
    if any(kw in desc_lower for kw in cooling_keywords):
        return EmissionSource.PURCHASED_COOLING
    return None


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
        >>> h = calculate_provenance_hash("FRN-001", Decimal("12345.67"))
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
            hash_input += str(inp.quantize(_QUANT_8DP, rounding=ROUND_HALF_UP))
        else:
            hash_input += str(inp)

    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


def get_dqi_classification(score: Decimal) -> str:
    """
    Map a composite DQI score (1-5) to a human-readable classification label.

    Thresholds:
        >= 4.5 -> "Excellent"
        >= 3.5 -> "Good"
        >= 2.5 -> "Fair"
        >= 1.5 -> "Poor"
        <  1.5 -> "Very Poor"

    Args:
        score: Composite DQI score (Decimal, 1-5 range).

    Returns:
        Classification string.

    Example:
        >>> get_dqi_classification(Decimal("4.8"))
        'Excellent'
        >>> get_dqi_classification(Decimal("3.0"))
        'Fair'
    """
    if score >= Decimal("4.5"):
        return "Excellent"
    if score >= Decimal("3.5"):
        return "Good"
    if score >= Decimal("2.5"):
        return "Fair"
    if score >= Decimal("1.5"):
        return "Poor"
    return "Very Poor"


# ---------------------------------------------------------------------------
# Currency conversion constants (approximate mid-market rates to USD)
# ---------------------------------------------------------------------------
_CURRENCY_TO_USD: Dict[str, Decimal] = {
    "USD": Decimal("1.0000"),
    "EUR": Decimal("1.0850"),
    "GBP": Decimal("1.2650"),
    "JPY": Decimal("0.00670"),
    "CAD": Decimal("0.7450"),
    "AUD": Decimal("0.6520"),
    "CHF": Decimal("1.1200"),
    "CNY": Decimal("0.1380"),
    "INR": Decimal("0.01200"),
    "BRL": Decimal("0.2000"),
    "KRW": Decimal("0.000750"),
    "MXN": Decimal("0.0580"),
}


def convert_currency_to_usd(amount: Decimal, currency: str) -> Decimal:
    """
    Convert a monetary amount to USD using approximate mid-market rates.

    Args:
        amount: Monetary value in source currency.
        currency: ISO 4217 currency code (e.g., 'EUR', 'GBP').

    Returns:
        Amount in USD, quantized to 2 decimal places.

    Raises:
        ValueError: If currency code is not recognized.

    Example:
        >>> convert_currency_to_usd(Decimal("1000"), "USD")
        Decimal('1000.00')
        >>> convert_currency_to_usd(Decimal("1000"), "EUR")
        Decimal('1085.00')
    """
    code = currency.strip().upper()
    rate = _CURRENCY_TO_USD.get(code)
    if rate is None:
        raise ValueError(
            f"Unknown currency '{currency}'.  "
            f"Supported: {list(_CURRENCY_TO_USD.keys())}"
        )
    return (amount * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# CPI deflator table (base year 2021 = 1.0000)
# Source: US BLS CPI-U All Items (approximate annual averages)
# ---------------------------------------------------------------------------
_CPI_DEFLATORS: Dict[int, Decimal] = {
    2015: Decimal("0.8850"),
    2016: Decimal("0.8960"),
    2017: Decimal("0.9150"),
    2018: Decimal("0.9370"),
    2019: Decimal("0.9530"),
    2020: Decimal("0.9650"),
    2021: Decimal("1.0000"),
    2022: Decimal("1.0800"),
    2023: Decimal("1.1150"),
    2024: Decimal("1.1450"),
    2025: Decimal("1.1700"),
    2026: Decimal("1.1950"),
    2027: Decimal("1.2200"),
    2028: Decimal("1.2450"),
    2029: Decimal("1.2700"),
    2030: Decimal("1.2950"),
}


def get_cpi_deflator(year: int) -> Decimal:
    """
    Get the CPI deflator for a given year (base year 2021 = 1.0).

    Used to deflate spend-based inputs to the EEIO base year.

    Args:
        year: Calendar year (2015-2030).

    Returns:
        CPI deflator as Decimal.

    Raises:
        ValueError: If year is not in the supported range.

    Example:
        >>> get_cpi_deflator(2021)
        Decimal('1.0000')
        >>> get_cpi_deflator(2024)
        Decimal('1.1450')
    """
    deflator = _CPI_DEFLATORS.get(year)
    if deflator is None:
        raise ValueError(
            f"CPI deflator not available for year {year}.  "
            f"Supported range: {min(_CPI_DEFLATORS)}-{max(_CPI_DEFLATORS)}"
        )
    return deflator


def get_revenue_intensity(franchise_type: str) -> Decimal:
    """
    Get the EEIO revenue intensity factor (kgCO2e/USD) for a franchise type.

    Args:
        franchise_type: FranchiseType value string (e.g., 'qsr_restaurant').

    Returns:
        Revenue intensity factor, or Decimal("0.1750") as default.

    Example:
        >>> get_revenue_intensity("qsr_restaurant")
        Decimal('0.3820')
    """
    return FRANCHISE_REVENUE_INTENSITY.get(franchise_type, Decimal("0.1750"))


def get_refrigerant_gwp(refrigerant_type: str) -> Decimal:
    """
    Get the 100-year GWP for a refrigerant type.

    Args:
        refrigerant_type: RefrigerantType value string (e.g., 'R_404A').

    Returns:
        GWP value as Decimal, or Decimal("0") if not found.

    Example:
        >>> get_refrigerant_gwp("R_404A")
        Decimal('3922')
    """
    return REFRIGERANT_GWPS.get(refrigerant_type, Decimal("0"))


def get_eeio_factor(naics_code: str) -> Optional[Decimal]:
    """
    Get the EEIO emission factor (kgCO2e/USD) for a NAICS code.

    Args:
        naics_code: 6-digit NAICS industry code.

    Returns:
        Emission factor as Decimal, or None if not found.

    Example:
        >>> get_eeio_factor("722513")
        Decimal('0.3820')
        >>> get_eeio_factor("000000") is None
        True
    """
    entry = EEIO_SPEND_FACTORS.get(naics_code)
    if entry is not None:
        return entry["ef"]
    return None


def get_hotel_benchmark(
    hotel_class: str,
    climate_zone: str,
) -> Decimal:
    """
    Get the hotel energy benchmark (kWh/room/year) by class and climate zone.

    Args:
        hotel_class: Hotel class (economy, midscale, upscale, luxury).
        climate_zone: ClimateZone value string.

    Returns:
        Benchmark value, or Decimal("18200") as default (midscale/temperate).

    Example:
        >>> get_hotel_benchmark("upscale", "temperate")
        Decimal('25000')
    """
    class_data = HOTEL_ENERGY_BENCHMARKS.get(hotel_class)
    if class_data is None:
        return Decimal("18200")
    return class_data.get(climate_zone, Decimal("18200"))


def get_vehicle_ef(vehicle_type: str) -> Optional[Dict[str, Decimal]]:
    """
    Get the delivery vehicle emission factors (kgCO2e/km) by vehicle type.

    Args:
        vehicle_type: Vehicle type key (e.g., 'small_van', 'motorcycle').

    Returns:
        Dict with 'ef_per_km' and 'wtt_per_km', or None if not found.

    Example:
        >>> get_vehicle_ef("small_van")
        {'ef_per_km': Decimal('0.20910'), 'wtt_per_km': Decimal('0.04845')}
    """
    return VEHICLE_EMISSION_FACTORS.get(vehicle_type)


def get_dc_rule(rule_id: str) -> Optional[Dict[str, str]]:
    """
    Get a double-counting prevention rule by its ID.

    Args:
        rule_id: Rule identifier (e.g., 'DC-FRN-001').

    Returns:
        Rule dict with 'rule', 'scope', 'severity', 'description',
        or None if not found.

    Example:
        >>> get_dc_rule("DC-FRN-001")["severity"]
        'critical'
    """
    return DC_RULES.get(rule_id)


def validate_ownership_for_cat14(ownership_type: str) -> bool:
    """
    Validate whether a string ownership type is included in Scope 3 Cat 14.

    Company-owned units are excluded per DC-FRN-001.

    Args:
        ownership_type: Ownership type string (e.g., 'franchised', 'company_owned').

    Returns:
        True if included in Cat 14, False if excluded.

    Example:
        >>> validate_ownership_for_cat14("franchised")
        True
        >>> validate_ownership_for_cat14("company_owned")
        False
        >>> validate_ownership_for_cat14("joint_venture")
        True
    """
    return ownership_type != "company_owned"


def get_franchise_type_label(franchise_type: str) -> str:
    """
    Return a human-readable label for a franchise type string value.

    Args:
        franchise_type: FranchiseType value string (e.g., 'qsr_restaurant').

    Returns:
        Display label string.

    Example:
        >>> get_franchise_type_label("qsr_restaurant")
        'Quick-Service Restaurant'
        >>> get_franchise_type_label("hotel")
        'Hotel / Lodging'
    """
    labels: Dict[str, str] = {
        "qsr_restaurant": "Quick-Service Restaurant",
        "full_service_restaurant": "Full-Service Restaurant",
        "hotel": "Hotel / Lodging",
        "convenience_store": "Convenience Store",
        "retail_store": "Retail Store",
        "fitness_center": "Fitness Center",
        "automotive_service": "Automotive Service",
        "healthcare_clinic": "Healthcare Clinic",
        "education_center": "Education Center",
        "other_service": "Other Service",
    }
    return labels.get(franchise_type, franchise_type.replace("_", " ").title())


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
    "FranchiseType",
    "OwnershipType",
    "FranchiseAgreementType",
    "CalculationMethod",
    "EmissionSource",
    "FuelType",
    "ClimateZone",
    "EFSource",
    "DataQualityTier",
    "DQIDimension",
    "ComplianceFramework",
    "ComplianceStatus",
    "PipelineStage",
    "UncertaintyMethod",
    "BatchStatus",
    "GWPSource",
    "DataCollectionMethod",
    "UnitStatus",
    "ConsolidationApproach",
    "RefrigerantType",

    # Constant tables
    "FRANCHISE_EUI_BENCHMARKS",
    "FRANCHISE_REVENUE_INTENSITY",
    "COOKING_FUEL_CONSUMPTION",
    "REFRIGERATION_LEAKAGE_RATES",
    "GRID_EMISSION_FACTORS",
    "FUEL_EMISSION_FACTORS",
    "REFRIGERANT_GWPS",
    "EEIO_SPEND_FACTORS",
    "HOTEL_ENERGY_BENCHMARKS",
    "VEHICLE_EMISSION_FACTORS",
    "DC_RULES",
    "COMPLIANCE_FRAMEWORK_RULES",
    "DQI_SCORING",
    "DQI_WEIGHTS",
    "UNCERTAINTY_RANGES",
    "COUNTRY_CLIMATE_ZONES",

    # Input models
    "CookingEnergyInput",
    "RefrigerationInput",
    "DeliveryFleetInput",
    "HotelOperationsInput",
    "FranchiseUnitInput",
    "FranchiseNetworkInput",

    # Result models
    "DataQualityScore",
    "UncertaintyResult",
    "FranchiseCalculationResult",
    "DataCoverageReport",
    "NetworkAggregationResult",
    "ComplianceResult",
    "ProvenanceRecord",
    "AggregationResult",

    # Helper functions
    "validate_franchise_type",
    "validate_ownership_boundary",
    "validate_consolidation_approach",
    "calculate_pro_rata_factor",
    "normalize_floor_area",
    "convert_energy_units",
    "get_climate_zone",
    "get_grid_ef",
    "get_fuel_ef",
    "get_eui_benchmark",
    "is_company_owned",
    "is_franchised",
    "get_franchise_label",
    "format_emissions_kg",
    "format_emissions_tonnes",
    "get_data_quality_tier",
    "classify_emission_source",
    "calculate_provenance_hash",
    "get_dqi_classification",
    "convert_currency_to_usd",
    "get_cpi_deflator",
    "get_revenue_intensity",
    "get_refrigerant_gwp",
    "get_eeio_factor",
    "get_hotel_benchmark",
    "get_vehicle_ef",
    "get_dc_rule",
    "validate_ownership_for_cat14",
    "get_franchise_type_label",
]
