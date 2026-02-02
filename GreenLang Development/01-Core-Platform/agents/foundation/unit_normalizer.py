# -*- coding: utf-8 -*-
"""
GL-FOUND-X-003: Unit & Reference Normalizer
============================================

The core unit conversion and reference standardization agent for GreenLang Climate OS.
This agent provides deterministic, auditable conversions with complete provenance tracking.

Capabilities:
    - Unit conversion across compatible dimensions (mass, energy, volume, area, distance)
    - GHG unit handling with GWP (Global Warming Potential) conversions
    - Fuel name standardization to canonical references
    - Material name standardization to canonical references
    - Reference ID management for cross-system mapping
    - Currency conversion with date-specific exchange rates
    - Dimensional analysis to prevent invalid conversions
    - Conversion lineage tracking for complete audit trails
    - Custom conversion factors per tenant

Zero-Hallucination Guarantees:
    - All conversions are deterministic mathematical operations
    - NO LLM involvement in any conversion calculations
    - All conversion factors are traceable to authoritative sources
    - Complete provenance hash for every conversion operation

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism.clock import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class UnitDimension(str, Enum):
    """Supported unit dimensions for dimensional analysis."""
    MASS = "mass"
    ENERGY = "energy"
    VOLUME = "volume"
    AREA = "area"
    DISTANCE = "distance"
    EMISSIONS = "emissions"
    CURRENCY = "currency"
    TIME = "time"


class GHGType(str, Enum):
    """Greenhouse gas types with GWP values."""
    CO2 = "CO2"
    CH4 = "CH4"
    N2O = "N2O"
    CO2E = "CO2e"  # CO2 equivalent


# Global Warming Potentials (AR6 100-year values)
# Source: IPCC AR6, Chapter 7, Table 7.SM.7
GWP_AR6_100 = {
    GHGType.CO2: Decimal("1"),
    GHGType.CH4: Decimal("29.8"),  # Fossil CH4 with climate-carbon feedback
    GHGType.N2O: Decimal("273"),
    GHGType.CO2E: Decimal("1"),
}

# Alternative GWP sets for different reporting standards
GWP_AR5_100 = {
    GHGType.CO2: Decimal("1"),
    GHGType.CH4: Decimal("28"),
    GHGType.N2O: Decimal("265"),
    GHGType.CO2E: Decimal("1"),
}

GWP_AR4_100 = {
    GHGType.CO2: Decimal("1"),
    GHGType.CH4: Decimal("25"),
    GHGType.N2O: Decimal("298"),
    GHGType.CO2E: Decimal("1"),
}


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class UnitDefinition(BaseModel):
    """Definition of a unit with its properties."""
    symbol: str = Field(..., description="Unit symbol (e.g., 'kg', 'MWh')")
    name: str = Field(..., description="Full unit name")
    dimension: UnitDimension = Field(..., description="Physical dimension")
    to_base_factor: Decimal = Field(..., description="Factor to convert to base unit")
    base_unit: str = Field(..., description="Base unit for this dimension")
    aliases: List[str] = Field(default_factory=list, description="Alternative names/symbols")
    source: str = Field(default="SI", description="Source standard (SI, US, etc.)")


class ConversionRequest(BaseModel):
    """Request for a unit conversion operation."""
    value: float = Field(..., description="Value to convert")
    from_unit: str = Field(..., description="Source unit")
    to_unit: str = Field(..., description="Target unit")
    precision: int = Field(default=6, ge=0, le=15, description="Decimal precision")
    tenant_id: Optional[str] = Field(None, description="Tenant ID for custom factors")

    @field_validator('value')
    @classmethod
    def validate_value(cls, v: float) -> float:
        """Ensure value is finite."""
        if not (-1e308 < v < 1e308):
            raise ValueError("Value must be finite")
        return v


class ConversionResult(BaseModel):
    """Result of a unit conversion operation."""
    original_value: float = Field(..., description="Original input value")
    original_unit: str = Field(..., description="Original unit")
    converted_value: float = Field(..., description="Converted value")
    target_unit: str = Field(..., description="Target unit")
    conversion_factor: float = Field(..., description="Applied conversion factor")
    dimension: str = Field(..., description="Unit dimension")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    source: str = Field(default="GreenLang", description="Conversion factor source")
    timestamp: datetime = Field(default_factory=DeterministicClock.now)


class GHGConversionRequest(BaseModel):
    """Request for GHG unit conversion with GWP."""
    value: float = Field(..., description="Value to convert")
    from_unit: str = Field(..., description="Source unit (e.g., kgCO2, tCH4)")
    to_unit: str = Field(..., description="Target unit (e.g., tCO2e)")
    gwp_source: str = Field(default="AR6", description="GWP source (AR4, AR5, AR6)")


class GHGConversionResult(BaseModel):
    """Result of a GHG conversion operation."""
    original_value: float = Field(..., description="Original value")
    original_unit: str = Field(..., description="Original unit")
    original_gas: str = Field(..., description="Original GHG type")
    converted_value: float = Field(..., description="Converted value")
    target_unit: str = Field(..., description="Target unit")
    target_gas: str = Field(..., description="Target GHG type")
    gwp_applied: float = Field(..., description="GWP factor applied")
    gwp_source: str = Field(..., description="GWP source standard")
    mass_conversion_factor: float = Field(..., description="Mass unit conversion factor")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


class FuelStandardizationRequest(BaseModel):
    """Request to standardize a fuel name."""
    fuel_name: str = Field(..., description="Input fuel name to standardize")
    tenant_id: Optional[str] = Field(None, description="Tenant ID for custom mappings")


class FuelStandardizationResult(BaseModel):
    """Result of fuel name standardization."""
    original_name: str = Field(..., description="Original fuel name")
    standardized_name: str = Field(..., description="Standardized fuel name")
    fuel_code: str = Field(..., description="Standard fuel code")
    fuel_category: str = Field(..., description="Fuel category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Match confidence")
    source: str = Field(..., description="Reference source")
    provenance_hash: str = Field(..., description="SHA-256 hash")


class MaterialStandardizationRequest(BaseModel):
    """Request to standardize a material name."""
    material_name: str = Field(..., description="Input material name")
    tenant_id: Optional[str] = Field(None, description="Tenant ID for custom mappings")


class MaterialStandardizationResult(BaseModel):
    """Result of material name standardization."""
    original_name: str = Field(..., description="Original material name")
    standardized_name: str = Field(..., description="Standardized name")
    material_code: str = Field(..., description="Standard material code")
    material_category: str = Field(..., description="Material category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Match confidence")
    source: str = Field(..., description="Reference source")
    provenance_hash: str = Field(..., description="SHA-256 hash")


class ReferenceIDRequest(BaseModel):
    """Request to manage cross-system reference IDs."""
    source_system: str = Field(..., description="Source system identifier")
    source_id: str = Field(..., description="ID in source system")
    target_system: Optional[str] = Field(None, description="Target system (if resolving)")


class ReferenceIDResult(BaseModel):
    """Result of reference ID resolution."""
    source_system: str = Field(..., description="Source system")
    source_id: str = Field(..., description="Source ID")
    canonical_id: str = Field(..., description="GreenLang canonical ID")
    mappings: Dict[str, str] = Field(default_factory=dict, description="All system mappings")
    provenance_hash: str = Field(..., description="SHA-256 hash")


class CurrencyConversionRequest(BaseModel):
    """Request for currency conversion."""
    value: float = Field(..., description="Amount to convert")
    from_currency: str = Field(..., description="Source currency (ISO 4217)")
    to_currency: str = Field(..., description="Target currency (ISO 4217)")
    conversion_date: Optional[date] = Field(None, description="Date for exchange rate")


class CurrencyConversionResult(BaseModel):
    """Result of currency conversion."""
    original_value: float = Field(..., description="Original amount")
    original_currency: str = Field(..., description="Source currency")
    converted_value: float = Field(..., description="Converted amount")
    target_currency: str = Field(..., description="Target currency")
    exchange_rate: float = Field(..., description="Applied exchange rate")
    rate_date: date = Field(..., description="Exchange rate date")
    rate_source: str = Field(..., description="Rate source")
    provenance_hash: str = Field(..., description="SHA-256 hash")


class NormalizerInput(BaseModel):
    """Input data model for UnitNormalizerAgent."""
    operation: str = Field(..., description="Operation type: convert, ghg_convert, standardize_fuel, standardize_material, resolve_reference, convert_currency")
    data: Dict[str, Any] = Field(..., description="Operation-specific data")
    tenant_id: Optional[str] = Field(None, description="Tenant ID for custom configs")


class NormalizerOutput(BaseModel):
    """Output data model for UnitNormalizerAgent."""
    success: bool = Field(..., description="Operation success status")
    operation: str = Field(..., description="Operation performed")
    result: Dict[str, Any] = Field(..., description="Operation result")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")
    processing_time_ms: float = Field(..., description="Processing duration")
    validation_status: str = Field(..., description="PASS or FAIL")
    errors: List[str] = Field(default_factory=list, description="Error messages if any")


# =============================================================================
# CONVERSION TABLES
# =============================================================================

# Mass units - base unit: kg
MASS_UNITS: Dict[str, Decimal] = {
    "g": Decimal("0.001"),
    "gram": Decimal("0.001"),
    "grams": Decimal("0.001"),
    "kg": Decimal("1"),
    "kilogram": Decimal("1"),
    "kilograms": Decimal("1"),
    "tonne": Decimal("1000"),
    "tonnes": Decimal("1000"),
    "metric_ton": Decimal("1000"),
    "t": Decimal("1000"),
    "lb": Decimal("0.45359237"),
    "lbs": Decimal("0.45359237"),
    "pound": Decimal("0.45359237"),
    "pounds": Decimal("0.45359237"),
    "short_ton": Decimal("907.18474"),
    "us_ton": Decimal("907.18474"),
    "long_ton": Decimal("1016.0469088"),
    "uk_ton": Decimal("1016.0469088"),
    "oz": Decimal("0.028349523125"),
    "ounce": Decimal("0.028349523125"),
    "mg": Decimal("0.000001"),
    "milligram": Decimal("0.000001"),
}

# Energy units - base unit: J (Joules)
ENERGY_UNITS: Dict[str, Decimal] = {
    "j": Decimal("1"),
    "joule": Decimal("1"),
    "joules": Decimal("1"),
    "kj": Decimal("1000"),
    "kilojoule": Decimal("1000"),
    "mj": Decimal("1000000"),
    "megajoule": Decimal("1000000"),
    "gj": Decimal("1000000000"),
    "gigajoule": Decimal("1000000000"),
    "kwh": Decimal("3600000"),
    "kilowatt_hour": Decimal("3600000"),
    "mwh": Decimal("3600000000"),
    "megawatt_hour": Decimal("3600000000"),
    "gwh": Decimal("3600000000000"),
    "gigawatt_hour": Decimal("3600000000000"),
    "btu": Decimal("1055.05585262"),
    "therm": Decimal("105505585.262"),
    "therms": Decimal("105505585.262"),
    "mmbtu": Decimal("1055055852.62"),
    "cal": Decimal("4.184"),
    "calorie": Decimal("4.184"),
    "kcal": Decimal("4184"),
    "kilocalorie": Decimal("4184"),
}

# Volume units - base unit: L (liters)
VOLUME_UNITS: Dict[str, Decimal] = {
    "l": Decimal("1"),
    "liter": Decimal("1"),
    "liters": Decimal("1"),
    "litre": Decimal("1"),
    "litres": Decimal("1"),
    "ml": Decimal("0.001"),
    "milliliter": Decimal("0.001"),
    "m3": Decimal("1000"),
    "cubic_meter": Decimal("1000"),
    "cubic_metre": Decimal("1000"),
    "gallon": Decimal("3.785411784"),
    "gallon_us": Decimal("3.785411784"),
    "gal": Decimal("3.785411784"),
    "gallon_uk": Decimal("4.54609"),
    "gallon_imperial": Decimal("4.54609"),
    "barrel": Decimal("158.987294928"),
    "bbl": Decimal("158.987294928"),
    "oil_barrel": Decimal("158.987294928"),
    "ft3": Decimal("28.316846592"),
    "cubic_foot": Decimal("28.316846592"),
    "cubic_feet": Decimal("28.316846592"),
    "ccf": Decimal("2831.6846592"),  # 100 cubic feet
    "mcf": Decimal("28316.846592"),  # 1000 cubic feet
}

# Area units - base unit: m2
AREA_UNITS: Dict[str, Decimal] = {
    "m2": Decimal("1"),
    "sqm": Decimal("1"),
    "square_meter": Decimal("1"),
    "square_metre": Decimal("1"),
    "hectare": Decimal("10000"),
    "ha": Decimal("10000"),
    "acre": Decimal("4046.8564224"),
    "acres": Decimal("4046.8564224"),
    "km2": Decimal("1000000"),
    "sqkm": Decimal("1000000"),
    "square_kilometer": Decimal("1000000"),
    "ft2": Decimal("0.09290304"),
    "sqft": Decimal("0.09290304"),
    "square_foot": Decimal("0.09290304"),
    "square_feet": Decimal("0.09290304"),
    "mi2": Decimal("2589988.110336"),
    "square_mile": Decimal("2589988.110336"),
}

# Distance units - base unit: m
DISTANCE_UNITS: Dict[str, Decimal] = {
    "m": Decimal("1"),
    "meter": Decimal("1"),
    "metre": Decimal("1"),
    "meters": Decimal("1"),
    "km": Decimal("1000"),
    "kilometer": Decimal("1000"),
    "kilometre": Decimal("1000"),
    "mi": Decimal("1609.344"),
    "mile": Decimal("1609.344"),
    "miles": Decimal("1609.344"),
    "nmi": Decimal("1852"),
    "nautical_mile": Decimal("1852"),
    "ft": Decimal("0.3048"),
    "foot": Decimal("0.3048"),
    "feet": Decimal("0.3048"),
    "yd": Decimal("0.9144"),
    "yard": Decimal("0.9144"),
    "yards": Decimal("0.9144"),
    "cm": Decimal("0.01"),
    "centimeter": Decimal("0.01"),
    "mm": Decimal("0.001"),
    "millimeter": Decimal("0.001"),
}

# Emissions units - base unit: kgCO2e
EMISSIONS_UNITS: Dict[str, Decimal] = {
    "kgco2e": Decimal("1"),
    "kgco2": Decimal("1"),
    "kg_co2e": Decimal("1"),
    "kg_co2": Decimal("1"),
    "tco2e": Decimal("1000"),
    "tco2": Decimal("1000"),
    "t_co2e": Decimal("1000"),
    "t_co2": Decimal("1000"),
    "tonneco2e": Decimal("1000"),
    "tonneco2": Decimal("1000"),
    "mtco2e": Decimal("1000"),  # Metric ton
    "gco2e": Decimal("0.001"),
    "gco2": Decimal("0.001"),
    "g_co2e": Decimal("0.001"),
    "lbco2e": Decimal("0.45359237"),
    "lb_co2e": Decimal("0.45359237"),
}

# Time units - base unit: seconds
TIME_UNITS: Dict[str, Decimal] = {
    "s": Decimal("1"),
    "second": Decimal("1"),
    "seconds": Decimal("1"),
    "min": Decimal("60"),
    "minute": Decimal("60"),
    "minutes": Decimal("60"),
    "h": Decimal("3600"),
    "hr": Decimal("3600"),
    "hour": Decimal("3600"),
    "hours": Decimal("3600"),
    "day": Decimal("86400"),
    "days": Decimal("86400"),
    "week": Decimal("604800"),
    "weeks": Decimal("604800"),
    "month": Decimal("2629746"),  # Average month (365.25/12 days)
    "months": Decimal("2629746"),
    "year": Decimal("31556952"),  # Average year (365.25 days)
    "years": Decimal("31556952"),
}

# Dimension to units mapping
DIMENSION_UNITS: Dict[UnitDimension, Dict[str, Decimal]] = {
    UnitDimension.MASS: MASS_UNITS,
    UnitDimension.ENERGY: ENERGY_UNITS,
    UnitDimension.VOLUME: VOLUME_UNITS,
    UnitDimension.AREA: AREA_UNITS,
    UnitDimension.DISTANCE: DISTANCE_UNITS,
    UnitDimension.EMISSIONS: EMISSIONS_UNITS,
    UnitDimension.TIME: TIME_UNITS,
}

# Base units per dimension
BASE_UNITS: Dict[UnitDimension, str] = {
    UnitDimension.MASS: "kg",
    UnitDimension.ENERGY: "j",
    UnitDimension.VOLUME: "l",
    UnitDimension.AREA: "m2",
    UnitDimension.DISTANCE: "m",
    UnitDimension.EMISSIONS: "kgco2e",
    UnitDimension.TIME: "s",
}


# =============================================================================
# FUEL STANDARDIZATION TABLES
# =============================================================================

FUEL_STANDARDIZATION: Dict[str, Dict[str, str]] = {
    # Natural Gas variants
    "natural gas": {"name": "Natural Gas", "code": "NG", "category": "gaseous"},
    "nat gas": {"name": "Natural Gas", "code": "NG", "category": "gaseous"},
    "natural_gas": {"name": "Natural Gas", "code": "NG", "category": "gaseous"},
    "methane": {"name": "Natural Gas", "code": "NG", "category": "gaseous"},
    "ng": {"name": "Natural Gas", "code": "NG", "category": "gaseous"},
    "pipeline gas": {"name": "Natural Gas", "code": "NG", "category": "gaseous"},
    "cng": {"name": "Compressed Natural Gas", "code": "CNG", "category": "gaseous"},
    "compressed natural gas": {"name": "Compressed Natural Gas", "code": "CNG", "category": "gaseous"},
    "lng": {"name": "Liquefied Natural Gas", "code": "LNG", "category": "gaseous"},
    "liquefied natural gas": {"name": "Liquefied Natural Gas", "code": "LNG", "category": "gaseous"},

    # Diesel variants
    "diesel": {"name": "Diesel", "code": "DSL", "category": "liquid"},
    "diesel fuel": {"name": "Diesel", "code": "DSL", "category": "liquid"},
    "diesel oil": {"name": "Diesel", "code": "DSL", "category": "liquid"},
    "gas oil": {"name": "Diesel", "code": "DSL", "category": "liquid"},
    "derv": {"name": "Diesel", "code": "DSL", "category": "liquid"},
    "red diesel": {"name": "Red Diesel", "code": "RDS", "category": "liquid"},
    "biodiesel": {"name": "Biodiesel", "code": "BDS", "category": "biofuel"},
    "b100": {"name": "Biodiesel", "code": "BDS", "category": "biofuel"},
    "b20": {"name": "Biodiesel Blend B20", "code": "B20", "category": "biofuel"},

    # Gasoline/Petrol variants
    "gasoline": {"name": "Gasoline", "code": "GAS", "category": "liquid"},
    "petrol": {"name": "Gasoline", "code": "GAS", "category": "liquid"},
    "motor gasoline": {"name": "Gasoline", "code": "GAS", "category": "liquid"},
    "mogas": {"name": "Gasoline", "code": "GAS", "category": "liquid"},
    "unleaded": {"name": "Gasoline", "code": "GAS", "category": "liquid"},
    "premium gasoline": {"name": "Premium Gasoline", "code": "PGS", "category": "liquid"},
    "e10": {"name": "Gasoline E10", "code": "E10", "category": "biofuel"},
    "e85": {"name": "Ethanol E85", "code": "E85", "category": "biofuel"},
    "ethanol": {"name": "Ethanol", "code": "ETH", "category": "biofuel"},

    # Propane/LPG variants
    "propane": {"name": "Propane", "code": "PRP", "category": "gaseous"},
    "lpg": {"name": "Liquefied Petroleum Gas", "code": "LPG", "category": "gaseous"},
    "liquefied petroleum gas": {"name": "Liquefied Petroleum Gas", "code": "LPG", "category": "gaseous"},
    "butane": {"name": "Butane", "code": "BUT", "category": "gaseous"},
    "autogas": {"name": "Liquefied Petroleum Gas", "code": "LPG", "category": "gaseous"},

    # Fuel Oil variants
    "fuel oil": {"name": "Fuel Oil", "code": "FO", "category": "liquid"},
    "heating oil": {"name": "Heating Oil", "code": "HO", "category": "liquid"},
    "hfo": {"name": "Heavy Fuel Oil", "code": "HFO", "category": "liquid"},
    "heavy fuel oil": {"name": "Heavy Fuel Oil", "code": "HFO", "category": "liquid"},
    "bunker fuel": {"name": "Heavy Fuel Oil", "code": "HFO", "category": "liquid"},
    "residual fuel oil": {"name": "Residual Fuel Oil", "code": "RFO", "category": "liquid"},
    "no. 2 oil": {"name": "Fuel Oil No. 2", "code": "FO2", "category": "liquid"},
    "no. 6 oil": {"name": "Fuel Oil No. 6", "code": "FO6", "category": "liquid"},
    "kerosene": {"name": "Kerosene", "code": "KER", "category": "liquid"},
    "jet fuel": {"name": "Jet Fuel", "code": "JET", "category": "liquid"},
    "jet a": {"name": "Jet Fuel A", "code": "JTA", "category": "liquid"},
    "aviation fuel": {"name": "Aviation Gasoline", "code": "AVG", "category": "liquid"},

    # Coal variants
    "coal": {"name": "Coal", "code": "COL", "category": "solid"},
    "bituminous coal": {"name": "Bituminous Coal", "code": "BCO", "category": "solid"},
    "anthracite": {"name": "Anthracite Coal", "code": "ANT", "category": "solid"},
    "lignite": {"name": "Lignite", "code": "LIG", "category": "solid"},
    "brown coal": {"name": "Lignite", "code": "LIG", "category": "solid"},
    "sub-bituminous": {"name": "Sub-bituminous Coal", "code": "SBC", "category": "solid"},
    "coke": {"name": "Coke", "code": "COK", "category": "solid"},
    "petroleum coke": {"name": "Petroleum Coke", "code": "PCK", "category": "solid"},

    # Biomass variants
    "biomass": {"name": "Biomass", "code": "BIO", "category": "biofuel"},
    "wood": {"name": "Wood", "code": "WOD", "category": "biofuel"},
    "wood chips": {"name": "Wood Chips", "code": "WCH", "category": "biofuel"},
    "wood pellets": {"name": "Wood Pellets", "code": "WPL", "category": "biofuel"},
    "firewood": {"name": "Firewood", "code": "FWD", "category": "biofuel"},
    "biogas": {"name": "Biogas", "code": "BGS", "category": "biofuel"},
    "landfill gas": {"name": "Landfill Gas", "code": "LFG", "category": "biofuel"},

    # Electricity (for completeness)
    "electricity": {"name": "Electricity", "code": "ELC", "category": "electricity"},
    "grid electricity": {"name": "Grid Electricity", "code": "GRD", "category": "electricity"},
    "renewable electricity": {"name": "Renewable Electricity", "code": "REN", "category": "electricity"},
    "solar": {"name": "Solar Electricity", "code": "SOL", "category": "electricity"},
    "wind": {"name": "Wind Electricity", "code": "WND", "category": "electricity"},

    # Hydrogen
    "hydrogen": {"name": "Hydrogen", "code": "H2", "category": "gaseous"},
    "green hydrogen": {"name": "Green Hydrogen", "code": "GH2", "category": "gaseous"},
    "blue hydrogen": {"name": "Blue Hydrogen", "code": "BH2", "category": "gaseous"},
    "grey hydrogen": {"name": "Grey Hydrogen", "code": "YH2", "category": "gaseous"},
}


# =============================================================================
# MATERIAL STANDARDIZATION TABLES
# =============================================================================

MATERIAL_STANDARDIZATION: Dict[str, Dict[str, str]] = {
    # Metals
    "steel": {"name": "Steel", "code": "STL", "category": "metals"},
    "carbon steel": {"name": "Carbon Steel", "code": "CST", "category": "metals"},
    "stainless steel": {"name": "Stainless Steel", "code": "SST", "category": "metals"},
    "aluminum": {"name": "Aluminum", "code": "ALU", "category": "metals"},
    "aluminium": {"name": "Aluminum", "code": "ALU", "category": "metals"},
    "copper": {"name": "Copper", "code": "COP", "category": "metals"},
    "iron": {"name": "Iron", "code": "IRN", "category": "metals"},
    "cast iron": {"name": "Cast Iron", "code": "CIR", "category": "metals"},
    "pig iron": {"name": "Pig Iron", "code": "PIR", "category": "metals"},
    "zinc": {"name": "Zinc", "code": "ZNC", "category": "metals"},
    "lead": {"name": "Lead", "code": "LED", "category": "metals"},
    "nickel": {"name": "Nickel", "code": "NIC", "category": "metals"},
    "titanium": {"name": "Titanium", "code": "TIT", "category": "metals"},
    "brass": {"name": "Brass", "code": "BRS", "category": "metals"},
    "bronze": {"name": "Bronze", "code": "BRZ", "category": "metals"},

    # Plastics
    "plastic": {"name": "Plastic (Generic)", "code": "PLS", "category": "plastics"},
    "pet": {"name": "PET (Polyethylene Terephthalate)", "code": "PET", "category": "plastics"},
    "polyethylene terephthalate": {"name": "PET (Polyethylene Terephthalate)", "code": "PET", "category": "plastics"},
    "hdpe": {"name": "HDPE (High-Density Polyethylene)", "code": "HDPE", "category": "plastics"},
    "high density polyethylene": {"name": "HDPE (High-Density Polyethylene)", "code": "HDPE", "category": "plastics"},
    "ldpe": {"name": "LDPE (Low-Density Polyethylene)", "code": "LDPE", "category": "plastics"},
    "low density polyethylene": {"name": "LDPE (Low-Density Polyethylene)", "code": "LDPE", "category": "plastics"},
    "pvc": {"name": "PVC (Polyvinyl Chloride)", "code": "PVC", "category": "plastics"},
    "polyvinyl chloride": {"name": "PVC (Polyvinyl Chloride)", "code": "PVC", "category": "plastics"},
    "pp": {"name": "PP (Polypropylene)", "code": "PP", "category": "plastics"},
    "polypropylene": {"name": "PP (Polypropylene)", "code": "PP", "category": "plastics"},
    "ps": {"name": "PS (Polystyrene)", "code": "PS", "category": "plastics"},
    "polystyrene": {"name": "PS (Polystyrene)", "code": "PS", "category": "plastics"},
    "abs": {"name": "ABS (Acrylonitrile Butadiene Styrene)", "code": "ABS", "category": "plastics"},
    "nylon": {"name": "Nylon", "code": "NYL", "category": "plastics"},
    "polyamide": {"name": "Nylon", "code": "NYL", "category": "plastics"},

    # Construction materials
    "cement": {"name": "Cement", "code": "CEM", "category": "construction"},
    "portland cement": {"name": "Portland Cement", "code": "PCM", "category": "construction"},
    "concrete": {"name": "Concrete", "code": "CON", "category": "construction"},
    "reinforced concrete": {"name": "Reinforced Concrete", "code": "RCO", "category": "construction"},
    "brick": {"name": "Brick", "code": "BRK", "category": "construction"},
    "glass": {"name": "Glass", "code": "GLS", "category": "construction"},
    "float glass": {"name": "Float Glass", "code": "FGL", "category": "construction"},
    "timber": {"name": "Timber", "code": "TMB", "category": "construction"},
    "lumber": {"name": "Timber", "code": "TMB", "category": "construction"},
    "plywood": {"name": "Plywood", "code": "PLY", "category": "construction"},
    "mdf": {"name": "MDF (Medium Density Fiberboard)", "code": "MDF", "category": "construction"},
    "gypsum": {"name": "Gypsum", "code": "GYP", "category": "construction"},
    "drywall": {"name": "Drywall", "code": "DRY", "category": "construction"},
    "asphalt": {"name": "Asphalt", "code": "ASP", "category": "construction"},
    "bitumen": {"name": "Bitumen", "code": "BIT", "category": "construction"},
    "gravel": {"name": "Gravel", "code": "GRV", "category": "construction"},
    "sand": {"name": "Sand", "code": "SND", "category": "construction"},
    "aggregate": {"name": "Aggregate", "code": "AGG", "category": "construction"},

    # Paper and packaging
    "paper": {"name": "Paper", "code": "PAP", "category": "paper"},
    "cardboard": {"name": "Cardboard", "code": "CBD", "category": "paper"},
    "corrugated cardboard": {"name": "Corrugated Cardboard", "code": "CCB", "category": "paper"},
    "kraft paper": {"name": "Kraft Paper", "code": "KFT", "category": "paper"},
    "newsprint": {"name": "Newsprint", "code": "NWS", "category": "paper"},
    "recycled paper": {"name": "Recycled Paper", "code": "RCP", "category": "paper"},

    # Chemicals
    "ammonia": {"name": "Ammonia", "code": "NH3", "category": "chemicals"},
    "urea": {"name": "Urea", "code": "URE", "category": "chemicals"},
    "fertilizer": {"name": "Fertilizer", "code": "FER", "category": "chemicals"},
    "sulfuric acid": {"name": "Sulfuric Acid", "code": "H2SO4", "category": "chemicals"},
    "nitric acid": {"name": "Nitric Acid", "code": "HNO3", "category": "chemicals"},
    "chlorine": {"name": "Chlorine", "code": "CL2", "category": "chemicals"},
    "sodium hydroxide": {"name": "Sodium Hydroxide", "code": "NaOH", "category": "chemicals"},
    "caustic soda": {"name": "Sodium Hydroxide", "code": "NaOH", "category": "chemicals"},
}


# =============================================================================
# SAMPLE EXCHANGE RATES (In production, these would come from a data service)
# =============================================================================

DEFAULT_EXCHANGE_RATES: Dict[str, Dict[str, float]] = {
    "USD": {"EUR": 0.92, "GBP": 0.79, "JPY": 149.50, "CNY": 7.24, "CAD": 1.36, "AUD": 1.53, "CHF": 0.88, "INR": 83.12},
    "EUR": {"USD": 1.09, "GBP": 0.86, "JPY": 162.50, "CNY": 7.87, "CAD": 1.48, "AUD": 1.66, "CHF": 0.96, "INR": 90.35},
    "GBP": {"USD": 1.27, "EUR": 1.16, "JPY": 189.00, "CNY": 9.16, "CAD": 1.72, "AUD": 1.93, "CHF": 1.11, "INR": 105.15},
}


# =============================================================================
# UNIT NORMALIZER AGENT
# =============================================================================

class UnitNormalizerAgent(BaseAgent):
    """
    GL-FOUND-X-003: Unit & Reference Normalizer Agent

    This agent provides deterministic unit conversion, reference standardization,
    and cross-system ID management with complete provenance tracking.

    Zero-Hallucination Guarantees:
        - All conversions use deterministic mathematical operations
        - NO LLM involvement in any calculation path
        - Complete audit trail with SHA-256 provenance hashes
        - All conversion factors are traceable to authoritative sources

    Capabilities:
        - Unit conversion across compatible dimensions
        - GHG unit handling with GWP conversions (AR4, AR5, AR6)
        - Fuel name standardization
        - Material name standardization
        - Reference ID management
        - Currency conversion with date-specific rates

    Usage:
        >>> agent = UnitNormalizerAgent()
        >>> result = agent.run({
        ...     "operation": "convert",
        ...     "data": {"value": 1000, "from_unit": "kg", "to_unit": "tonnes"}
        ... })
        >>> print(result.data["result"]["converted_value"])  # 1.0

    Attributes:
        AGENT_ID: Unique agent identifier
        AGENT_NAME: Human-readable agent name
        VERSION: Agent version
    """

    AGENT_ID = "GL-FOUND-X-003"
    AGENT_NAME = "Unit & Reference Normalizer"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize UnitNormalizerAgent.

        Args:
            config: Optional agent configuration
        """
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Unit conversion and reference standardization agent",
                version=self.VERSION,
                parameters={
                    "default_precision": 6,
                    "gwp_source": "AR6",
                    "strict_mode": True,
                }
            )
        super().__init__(config)

        # Custom conversion factors per tenant
        self._tenant_conversions: Dict[str, Dict[str, Dict[str, Decimal]]] = {}

        # Custom fuel mappings per tenant
        self._tenant_fuel_mappings: Dict[str, Dict[str, Dict[str, str]]] = {}

        # Custom material mappings per tenant
        self._tenant_material_mappings: Dict[str, Dict[str, Dict[str, str]]] = {}

        # Reference ID registry (in production, use database)
        self._reference_registry: Dict[str, Dict[str, str]] = {}

        # Exchange rate cache (in production, use external service)
        self._exchange_rates = DEFAULT_EXCHANGE_RATES.copy()

        self.logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def initialize(self):
        """Initialize agent resources."""
        # Pre-compile unit normalization patterns
        self._unit_pattern = re.compile(r'^([a-zA-Z_]+)(\d*)$')

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute a normalization operation.

        Args:
            input_data: Dictionary with 'operation' and 'data' keys

        Returns:
            AgentResult with conversion/standardization results

        Raises:
            ValueError: If operation is invalid or data is malformed
        """
        start_time = DeterministicClock.now()

        try:
            # Parse input
            normalizer_input = NormalizerInput(**input_data)
            operation = normalizer_input.operation.lower()
            data = normalizer_input.data
            tenant_id = normalizer_input.tenant_id

            # Route to appropriate handler
            if operation == "convert":
                result = self._handle_unit_conversion(data, tenant_id)
            elif operation == "ghg_convert":
                result = self._handle_ghg_conversion(data)
            elif operation == "standardize_fuel":
                result = self._handle_fuel_standardization(data, tenant_id)
            elif operation == "standardize_material":
                result = self._handle_material_standardization(data, tenant_id)
            elif operation == "resolve_reference":
                result = self._handle_reference_resolution(data)
            elif operation == "convert_currency":
                result = self._handle_currency_conversion(data)
            elif operation == "get_dimension":
                result = self._handle_get_dimension(data)
            elif operation == "validate_conversion":
                result = self._handle_validate_conversion(data)
            elif operation == "list_units":
                result = self._handle_list_units(data)
            else:
                return AgentResult(
                    success=False,
                    error=f"Unknown operation: {operation}",
                    data={"supported_operations": [
                        "convert", "ghg_convert", "standardize_fuel",
                        "standardize_material", "resolve_reference",
                        "convert_currency", "get_dimension",
                        "validate_conversion", "list_units"
                    ]}
                )

            # Calculate processing time
            end_time = DeterministicClock.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            # Build output
            output = NormalizerOutput(
                success=True,
                operation=operation,
                result=result,
                provenance_hash=self._compute_provenance_hash(input_data, result),
                processing_time_ms=processing_time_ms,
                validation_status="PASS"
            )

            return AgentResult(
                success=True,
                data=output.model_dump()
            )

        except Exception as e:
            self.logger.error(f"Normalization failed: {str(e)}", exc_info=True)
            end_time = DeterministicClock.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            return AgentResult(
                success=False,
                error=str(e),
                data={
                    "operation": input_data.get("operation", "unknown"),
                    "processing_time_ms": processing_time_ms,
                    "validation_status": "FAIL"
                }
            )

    # =========================================================================
    # UNIT CONVERSION
    # =========================================================================

    def _handle_unit_conversion(
        self,
        data: Dict[str, Any],
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle unit conversion operation.

        Args:
            data: Conversion request data
            tenant_id: Optional tenant ID for custom factors

        Returns:
            Conversion result dictionary
        """
        request = ConversionRequest(**data, tenant_id=tenant_id)

        # Normalize unit names
        from_unit = self._normalize_unit_name(request.from_unit)
        to_unit = self._normalize_unit_name(request.to_unit)

        # Determine dimensions (pass tenant_id for custom units)
        from_dimension = self._get_unit_dimension(from_unit, tenant_id)
        to_dimension = self._get_unit_dimension(to_unit, tenant_id)

        if from_dimension is None:
            raise ValueError(f"Unknown unit: {request.from_unit}")
        if to_dimension is None:
            raise ValueError(f"Unknown unit: {request.to_unit}")
        if from_dimension != to_dimension:
            raise ValueError(
                f"Cannot convert between different dimensions: "
                f"{from_dimension.value} ({request.from_unit}) to "
                f"{to_dimension.value} ({request.to_unit})"
            )

        # Get conversion factors
        units_table = self._get_units_table(from_dimension, tenant_id)
        from_factor = units_table[from_unit]
        to_factor = units_table[to_unit]

        # Calculate conversion factor and result
        conversion_factor = from_factor / to_factor
        value_decimal = Decimal(str(request.value))
        converted_decimal = value_decimal * conversion_factor

        # Round to requested precision
        precision_str = f"1e-{request.precision}"
        converted_rounded = converted_decimal.quantize(
            Decimal(precision_str),
            rounding=ROUND_HALF_UP
        )

        result = ConversionResult(
            original_value=request.value,
            original_unit=request.from_unit,
            converted_value=float(converted_rounded),
            target_unit=request.to_unit,
            conversion_factor=float(conversion_factor),
            dimension=from_dimension.value,
            provenance_hash=self._compute_provenance_hash(
                {"value": request.value, "from": from_unit, "to": to_unit},
                {"converted": float(converted_rounded)}
            ),
            source="GreenLang" if tenant_id is None else f"Tenant:{tenant_id}"
        )

        return result.model_dump()

    def convert(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        precision: int = 6,
        tenant_id: Optional[str] = None
    ) -> float:
        """
        Direct conversion method for programmatic use.

        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit
            precision: Decimal precision (default 6)
            tenant_id: Optional tenant ID

        Returns:
            Converted value as float

        Example:
            >>> agent = UnitNormalizerAgent()
            >>> agent.convert(1000, "kg", "tonnes")
            1.0
        """
        result = self._handle_unit_conversion({
            "value": value,
            "from_unit": from_unit,
            "to_unit": to_unit,
            "precision": precision
        }, tenant_id)
        return result["converted_value"]

    def _normalize_unit_name(self, unit: str) -> str:
        """
        Normalize unit name to standard form.

        Args:
            unit: Input unit name

        Returns:
            Normalized unit name (lowercase, standardized)
        """
        # Convert to lowercase and strip whitespace
        normalized = unit.lower().strip()

        # Replace common variations
        normalized = normalized.replace("-", "_").replace(" ", "_")
        normalized = normalized.replace("^2", "2").replace("^3", "3")

        # Handle subscript equivalents
        normalized = normalized.replace("co₂", "co2")

        return normalized

    def _get_unit_dimension(
        self,
        unit: str,
        tenant_id: Optional[str] = None
    ) -> Optional[UnitDimension]:
        """
        Determine the dimension of a unit.

        Args:
            unit: Normalized unit name
            tenant_id: Optional tenant ID for custom units

        Returns:
            UnitDimension enum or None if unknown
        """
        # Check standard units first
        for dimension, units_table in DIMENSION_UNITS.items():
            if unit in units_table:
                return dimension

        # Check tenant-specific units
        if tenant_id and tenant_id in self._tenant_conversions:
            for dim_str, units_table in self._tenant_conversions[tenant_id].items():
                if unit in units_table:
                    try:
                        return UnitDimension(dim_str)
                    except ValueError:
                        pass

        return None

    def _get_units_table(
        self,
        dimension: UnitDimension,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Decimal]:
        """
        Get units table for a dimension, including tenant customizations.

        Args:
            dimension: Unit dimension
            tenant_id: Optional tenant ID

        Returns:
            Dictionary of unit names to conversion factors
        """
        # Start with base table
        base_table = DIMENSION_UNITS[dimension].copy()

        # Merge tenant-specific conversions if available
        if tenant_id and tenant_id in self._tenant_conversions:
            tenant_table = self._tenant_conversions[tenant_id].get(
                dimension.value, {}
            )
            base_table.update(tenant_table)

        return base_table

    def is_convertible(self, from_unit: str, to_unit: str) -> bool:
        """
        Check if two units are convertible (same dimension).

        Args:
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            True if convertible, False otherwise
        """
        from_unit = self._normalize_unit_name(from_unit)
        to_unit = self._normalize_unit_name(to_unit)

        from_dim = self._get_unit_dimension(from_unit)
        to_dim = self._get_unit_dimension(to_unit)

        return from_dim is not None and from_dim == to_dim

    def _handle_validate_conversion(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle validation of conversion compatibility."""
        from_unit = data.get("from_unit", "")
        to_unit = data.get("to_unit", "")

        from_normalized = self._normalize_unit_name(from_unit)
        to_normalized = self._normalize_unit_name(to_unit)

        from_dim = self._get_unit_dimension(from_normalized)
        to_dim = self._get_unit_dimension(to_normalized)

        is_valid = from_dim is not None and from_dim == to_dim

        return {
            "from_unit": from_unit,
            "to_unit": to_unit,
            "from_dimension": from_dim.value if from_dim else None,
            "to_dimension": to_dim.value if to_dim else None,
            "is_convertible": is_valid,
            "reason": None if is_valid else (
                f"Unknown unit: {from_unit}" if from_dim is None else
                f"Unknown unit: {to_unit}" if to_dim is None else
                f"Incompatible dimensions: {from_dim.value} vs {to_dim.value}"
            )
        }

    def _handle_get_dimension(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get dimension operation."""
        unit = data.get("unit", "")
        normalized = self._normalize_unit_name(unit)
        dimension = self._get_unit_dimension(normalized)

        return {
            "unit": unit,
            "normalized_unit": normalized,
            "dimension": dimension.value if dimension else None,
            "base_unit": BASE_UNITS.get(dimension) if dimension else None,
            "is_known": dimension is not None
        }

    def _handle_list_units(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list units operation."""
        dimension_filter = data.get("dimension")

        if dimension_filter:
            try:
                dim = UnitDimension(dimension_filter.lower())
                return {
                    "dimension": dim.value,
                    "base_unit": BASE_UNITS[dim],
                    "units": list(DIMENSION_UNITS[dim].keys())
                }
            except ValueError:
                return {
                    "error": f"Unknown dimension: {dimension_filter}",
                    "available_dimensions": [d.value for d in UnitDimension]
                }

        return {
            "dimensions": {
                dim.value: {
                    "base_unit": BASE_UNITS[dim],
                    "unit_count": len(units),
                    "units": list(units.keys())
                }
                for dim, units in DIMENSION_UNITS.items()
            }
        }

    # =========================================================================
    # GHG CONVERSION
    # =========================================================================

    def _handle_ghg_conversion(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle GHG conversion with GWP.

        Args:
            data: GHG conversion request data

        Returns:
            GHG conversion result dictionary
        """
        request = GHGConversionRequest(**data)

        # Parse from_unit to extract gas type and mass unit
        from_gas, from_mass_unit = self._parse_ghg_unit(request.from_unit)
        to_gas, to_mass_unit = self._parse_ghg_unit(request.to_unit)

        # Get GWP table
        gwp_table = self._get_gwp_table(request.gwp_source)

        # Get GWP values
        from_gwp = gwp_table.get(from_gas, Decimal("1"))
        to_gwp = gwp_table.get(to_gas, Decimal("1"))

        # Calculate GWP factor (converting from one gas to another's equivalent)
        gwp_factor = from_gwp / to_gwp

        # Calculate mass conversion factor
        from_mass_normalized = self._normalize_unit_name(from_mass_unit)
        to_mass_normalized = self._normalize_unit_name(to_mass_unit)

        # Handle emission-specific units
        from_factor = EMISSIONS_UNITS.get(from_mass_normalized) or MASS_UNITS.get(from_mass_normalized)
        to_factor = EMISSIONS_UNITS.get(to_mass_normalized) or MASS_UNITS.get(to_mass_normalized)

        if from_factor is None:
            raise ValueError(f"Unknown mass unit in: {request.from_unit}")
        if to_factor is None:
            raise ValueError(f"Unknown mass unit in: {request.to_unit}")

        mass_conversion_factor = from_factor / to_factor

        # Calculate final result
        value_decimal = Decimal(str(request.value))
        converted = value_decimal * gwp_factor * mass_conversion_factor

        result = GHGConversionResult(
            original_value=request.value,
            original_unit=request.from_unit,
            original_gas=from_gas.value,
            converted_value=float(converted.quantize(Decimal("0.000001"))),
            target_unit=request.to_unit,
            target_gas=to_gas.value,
            gwp_applied=float(gwp_factor),
            gwp_source=request.gwp_source,
            mass_conversion_factor=float(mass_conversion_factor),
            provenance_hash=self._compute_provenance_hash(data, {"converted": float(converted)})
        )

        return result.model_dump()

    def convert_ghg(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        gwp_source: str = "AR6"
    ) -> float:
        """
        Convert GHG emissions with GWP.

        Args:
            value: Value to convert
            from_unit: Source unit (e.g., "kgCH4")
            to_unit: Target unit (e.g., "tCO2e")
            gwp_source: GWP source (AR4, AR5, AR6)

        Returns:
            Converted value

        Example:
            >>> agent = UnitNormalizerAgent()
            >>> agent.convert_ghg(1000, "kgCH4", "tCO2e", "AR6")
            29.8
        """
        result = self._handle_ghg_conversion({
            "value": value,
            "from_unit": from_unit,
            "to_unit": to_unit,
            "gwp_source": gwp_source
        })
        return result["converted_value"]

    def _parse_ghg_unit(self, unit: str) -> Tuple[GHGType, str]:
        """
        Parse a GHG unit string into gas type and mass unit.

        Args:
            unit: GHG unit string (e.g., "kgCO2e", "tCH4")

        Returns:
            Tuple of (GHGType, mass_unit_string)
        """
        unit_lower = unit.lower()

        # Check for each gas type
        for gas in [GHGType.CO2E, GHGType.CO2, GHGType.CH4, GHGType.N2O]:
            gas_str = gas.value.lower()
            if gas_str in unit_lower:
                # Extract mass unit prefix
                idx = unit_lower.find(gas_str)
                mass_part = unit_lower[:idx] if idx > 0 else "kg"
                # Map common prefixes
                if mass_part in ("", "kg"):
                    return gas, "kg"
                elif mass_part in ("t", "tonne", "mt"):
                    return gas, "tonne"
                elif mass_part == "g":
                    return gas, "g"
                elif mass_part == "lb":
                    return gas, "lb"
                else:
                    return gas, mass_part

        # Default to CO2e if no gas specified
        return GHGType.CO2E, self._normalize_unit_name(unit)

    def _get_gwp_table(self, source: str) -> Dict[GHGType, Decimal]:
        """
        Get GWP table for specified source.

        Args:
            source: GWP source (AR4, AR5, AR6)

        Returns:
            Dictionary of GHG type to GWP value
        """
        source_upper = source.upper()
        if source_upper == "AR6":
            return GWP_AR6_100
        elif source_upper == "AR5":
            return GWP_AR5_100
        elif source_upper == "AR4":
            return GWP_AR4_100
        else:
            self.logger.warning(f"Unknown GWP source: {source}, using AR6")
            return GWP_AR6_100

    # =========================================================================
    # FUEL STANDARDIZATION
    # =========================================================================

    def _handle_fuel_standardization(
        self,
        data: Dict[str, Any],
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle fuel name standardization.

        Args:
            data: Fuel standardization request data
            tenant_id: Optional tenant ID

        Returns:
            Fuel standardization result dictionary
        """
        request = FuelStandardizationRequest(**data, tenant_id=tenant_id)

        # Normalize input name
        normalized_name = request.fuel_name.lower().strip()

        # Check tenant-specific mappings first
        fuel_info = None
        source = "GreenLang"

        if tenant_id and tenant_id in self._tenant_fuel_mappings:
            fuel_info = self._tenant_fuel_mappings[tenant_id].get(normalized_name)
            if fuel_info:
                source = f"Tenant:{tenant_id}"

        # Fall back to standard mappings
        if fuel_info is None:
            fuel_info = FUEL_STANDARDIZATION.get(normalized_name)

        # Try fuzzy matching if exact match not found
        confidence = 1.0
        if fuel_info is None:
            fuel_info, confidence = self._fuzzy_match_fuel(normalized_name)

        if fuel_info is None:
            # Return unknown fuel with low confidence
            result = FuelStandardizationResult(
                original_name=request.fuel_name,
                standardized_name=request.fuel_name.title(),
                fuel_code="UNK",
                fuel_category="unknown",
                confidence=0.0,
                source="Unknown",
                provenance_hash=self._compute_provenance_hash(data, {"matched": False})
            )
        else:
            result = FuelStandardizationResult(
                original_name=request.fuel_name,
                standardized_name=fuel_info["name"],
                fuel_code=fuel_info["code"],
                fuel_category=fuel_info["category"],
                confidence=confidence,
                source=source,
                provenance_hash=self._compute_provenance_hash(data, fuel_info)
            )

        return result.model_dump()

    def standardize_fuel(
        self,
        fuel_name: str,
        tenant_id: Optional[str] = None
    ) -> str:
        """
        Standardize a fuel name.

        Args:
            fuel_name: Input fuel name
            tenant_id: Optional tenant ID

        Returns:
            Standardized fuel name

        Example:
            >>> agent = UnitNormalizerAgent()
            >>> agent.standardize_fuel("nat gas")
            'Natural Gas'
        """
        result = self._handle_fuel_standardization(
            {"fuel_name": fuel_name},
            tenant_id
        )
        return result["standardized_name"]

    def _fuzzy_match_fuel(
        self,
        name: str
    ) -> Tuple[Optional[Dict[str, str]], float]:
        """
        Attempt fuzzy matching for fuel names.

        Args:
            name: Normalized fuel name

        Returns:
            Tuple of (matched fuel info, confidence score)
        """
        best_match = None
        best_score = 0.0

        for key, info in FUEL_STANDARDIZATION.items():
            # Check for substring match
            if key in name or name in key:
                score = min(len(name), len(key)) / max(len(name), len(key))
                if score > best_score:
                    best_score = score
                    best_match = info

            # Check for word overlap
            name_words = set(name.split())
            key_words = set(key.split())
            overlap = len(name_words & key_words)
            if overlap > 0:
                score = overlap / max(len(name_words), len(key_words))
                if score > best_score:
                    best_score = score * 0.9  # Slightly lower confidence for word match
                    best_match = info

        # Only return if confidence is reasonable
        if best_score >= 0.5:
            return best_match, best_score
        return None, 0.0

    # =========================================================================
    # MATERIAL STANDARDIZATION
    # =========================================================================

    def _handle_material_standardization(
        self,
        data: Dict[str, Any],
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle material name standardization.

        Args:
            data: Material standardization request data
            tenant_id: Optional tenant ID

        Returns:
            Material standardization result dictionary
        """
        request = MaterialStandardizationRequest(**data, tenant_id=tenant_id)

        # Normalize input name
        normalized_name = request.material_name.lower().strip()

        # Check tenant-specific mappings first
        material_info = None
        source = "GreenLang"

        if tenant_id and tenant_id in self._tenant_material_mappings:
            material_info = self._tenant_material_mappings[tenant_id].get(normalized_name)
            if material_info:
                source = f"Tenant:{tenant_id}"

        # Fall back to standard mappings
        if material_info is None:
            material_info = MATERIAL_STANDARDIZATION.get(normalized_name)

        # Try fuzzy matching if exact match not found
        confidence = 1.0
        if material_info is None:
            material_info, confidence = self._fuzzy_match_material(normalized_name)

        if material_info is None:
            result = MaterialStandardizationResult(
                original_name=request.material_name,
                standardized_name=request.material_name.title(),
                material_code="UNK",
                material_category="unknown",
                confidence=0.0,
                source="Unknown",
                provenance_hash=self._compute_provenance_hash(data, {"matched": False})
            )
        else:
            result = MaterialStandardizationResult(
                original_name=request.material_name,
                standardized_name=material_info["name"],
                material_code=material_info["code"],
                material_category=material_info["category"],
                confidence=confidence,
                source=source,
                provenance_hash=self._compute_provenance_hash(data, material_info)
            )

        return result.model_dump()

    def standardize_material(
        self,
        material_name: str,
        tenant_id: Optional[str] = None
    ) -> str:
        """
        Standardize a material name.

        Args:
            material_name: Input material name
            tenant_id: Optional tenant ID

        Returns:
            Standardized material name

        Example:
            >>> agent = UnitNormalizerAgent()
            >>> agent.standardize_material("aluminium")
            'Aluminum'
        """
        result = self._handle_material_standardization(
            {"material_name": material_name},
            tenant_id
        )
        return result["standardized_name"]

    def _fuzzy_match_material(
        self,
        name: str
    ) -> Tuple[Optional[Dict[str, str]], float]:
        """
        Attempt fuzzy matching for material names.

        Args:
            name: Normalized material name

        Returns:
            Tuple of (matched material info, confidence score)
        """
        best_match = None
        best_score = 0.0

        for key, info in MATERIAL_STANDARDIZATION.items():
            # Check for substring match
            if key in name or name in key:
                score = min(len(name), len(key)) / max(len(name), len(key))
                if score > best_score:
                    best_score = score
                    best_match = info

        if best_score >= 0.5:
            return best_match, best_score
        return None, 0.0

    # =========================================================================
    # REFERENCE ID MANAGEMENT
    # =========================================================================

    def _handle_reference_resolution(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle reference ID resolution.

        Args:
            data: Reference ID request data

        Returns:
            Reference ID result dictionary
        """
        request = ReferenceIDRequest(**data)

        # Create canonical key
        canonical_key = f"{request.source_system}:{request.source_id}"

        # Check if we have this reference
        if canonical_key in self._reference_registry:
            mappings = self._reference_registry[canonical_key]
            canonical_id = mappings.get("canonical", canonical_key)
        else:
            # Create new canonical ID
            canonical_id = f"GL-{uuid.uuid4().hex[:12].upper()}"
            self._reference_registry[canonical_key] = {
                "canonical": canonical_id,
                request.source_system: request.source_id
            }
            mappings = self._reference_registry[canonical_key]

        result = ReferenceIDResult(
            source_system=request.source_system,
            source_id=request.source_id,
            canonical_id=canonical_id,
            mappings=mappings,
            provenance_hash=self._compute_provenance_hash(data, {"canonical": canonical_id})
        )

        return result.model_dump()

    def register_reference_mapping(
        self,
        source_system: str,
        source_id: str,
        target_system: str,
        target_id: str
    ) -> str:
        """
        Register a cross-system reference mapping.

        Args:
            source_system: Source system name
            source_id: ID in source system
            target_system: Target system name
            target_id: ID in target system

        Returns:
            Canonical GreenLang ID

        Example:
            >>> agent = UnitNormalizerAgent()
            >>> canonical = agent.register_reference_mapping(
            ...     "SAP", "MAT001", "Oracle", "M-00001"
            ... )
        """
        # Get or create canonical entry
        source_key = f"{source_system}:{source_id}"
        target_key = f"{target_system}:{target_id}"

        canonical_id = None

        # Check if either mapping exists
        if source_key in self._reference_registry:
            canonical_id = self._reference_registry[source_key].get("canonical")
        elif target_key in self._reference_registry:
            canonical_id = self._reference_registry[target_key].get("canonical")

        # Create new canonical if needed
        if canonical_id is None:
            canonical_id = f"GL-{uuid.uuid4().hex[:12].upper()}"

        # Update both mappings
        if source_key not in self._reference_registry:
            self._reference_registry[source_key] = {"canonical": canonical_id}
        self._reference_registry[source_key][target_system] = target_id

        if target_key not in self._reference_registry:
            self._reference_registry[target_key] = {"canonical": canonical_id}
        self._reference_registry[target_key][source_system] = source_id

        return canonical_id

    # =========================================================================
    # CURRENCY CONVERSION
    # =========================================================================

    def _handle_currency_conversion(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle currency conversion.

        Args:
            data: Currency conversion request data

        Returns:
            Currency conversion result dictionary
        """
        request = CurrencyConversionRequest(**data)

        # Normalize currency codes
        from_currency = request.from_currency.upper()
        to_currency = request.to_currency.upper()
        rate_date = request.conversion_date or date.today()

        # Get exchange rate
        exchange_rate = self._get_exchange_rate(from_currency, to_currency, rate_date)

        if exchange_rate is None:
            raise ValueError(
                f"No exchange rate available for {from_currency} to {to_currency}"
            )

        # Calculate conversion
        converted_value = request.value * exchange_rate

        result = CurrencyConversionResult(
            original_value=request.value,
            original_currency=from_currency,
            converted_value=round(converted_value, 2),
            target_currency=to_currency,
            exchange_rate=exchange_rate,
            rate_date=rate_date,
            rate_source="GreenLang Default Rates",
            provenance_hash=self._compute_provenance_hash(data, {"converted": converted_value})
        )

        return result.model_dump()

    def convert_currency(
        self,
        value: float,
        from_currency: str,
        to_currency: str,
        conversion_date: Optional[date] = None
    ) -> float:
        """
        Convert currency value.

        Args:
            value: Amount to convert
            from_currency: Source currency (ISO 4217)
            to_currency: Target currency (ISO 4217)
            conversion_date: Date for exchange rate

        Returns:
            Converted amount

        Example:
            >>> agent = UnitNormalizerAgent()
            >>> agent.convert_currency(100, "USD", "EUR")
            92.0
        """
        result = self._handle_currency_conversion({
            "value": value,
            "from_currency": from_currency,
            "to_currency": to_currency,
            "conversion_date": conversion_date.isoformat() if conversion_date else None
        })
        return result["converted_value"]

    def _get_exchange_rate(
        self,
        from_currency: str,
        to_currency: str,
        rate_date: date
    ) -> Optional[float]:
        """
        Get exchange rate between currencies.

        Args:
            from_currency: Source currency
            to_currency: Target currency
            rate_date: Date for rate (not used in default implementation)

        Returns:
            Exchange rate or None if not available
        """
        if from_currency == to_currency:
            return 1.0

        # Check direct rate
        if from_currency in self._exchange_rates:
            if to_currency in self._exchange_rates[from_currency]:
                return self._exchange_rates[from_currency][to_currency]

        # Check inverse rate
        if to_currency in self._exchange_rates:
            if from_currency in self._exchange_rates[to_currency]:
                return 1.0 / self._exchange_rates[to_currency][from_currency]

        # Try triangulation through USD
        if from_currency != "USD" and to_currency != "USD":
            rate_to_usd = self._get_exchange_rate(from_currency, "USD", rate_date)
            rate_from_usd = self._get_exchange_rate("USD", to_currency, rate_date)
            if rate_to_usd and rate_from_usd:
                return rate_to_usd * rate_from_usd

        return None

    def set_exchange_rate(
        self,
        from_currency: str,
        to_currency: str,
        rate: float
    ):
        """
        Set a custom exchange rate.

        Args:
            from_currency: Source currency
            to_currency: Target currency
            rate: Exchange rate
        """
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()

        if from_currency not in self._exchange_rates:
            self._exchange_rates[from_currency] = {}

        self._exchange_rates[from_currency][to_currency] = rate
        self.logger.info(f"Set exchange rate: {from_currency}/{to_currency} = {rate}")

    # =========================================================================
    # TENANT CUSTOMIZATION
    # =========================================================================

    def register_tenant_conversion(
        self,
        tenant_id: str,
        dimension: str,
        unit: str,
        to_base_factor: float
    ):
        """
        Register a tenant-specific unit conversion factor.

        Args:
            tenant_id: Tenant identifier
            dimension: Unit dimension (mass, energy, etc.)
            unit: Unit name
            to_base_factor: Factor to convert to base unit
        """
        if tenant_id not in self._tenant_conversions:
            self._tenant_conversions[tenant_id] = {}

        if dimension not in self._tenant_conversions[tenant_id]:
            self._tenant_conversions[tenant_id][dimension] = {}

        self._tenant_conversions[tenant_id][dimension][unit.lower()] = Decimal(str(to_base_factor))
        self.logger.info(f"Registered tenant conversion: {tenant_id}/{dimension}/{unit}")

    def register_tenant_fuel_mapping(
        self,
        tenant_id: str,
        fuel_alias: str,
        standard_name: str,
        fuel_code: str,
        category: str
    ):
        """
        Register a tenant-specific fuel name mapping.

        Args:
            tenant_id: Tenant identifier
            fuel_alias: Tenant's fuel name
            standard_name: Standard fuel name
            fuel_code: Standard fuel code
            category: Fuel category
        """
        if tenant_id not in self._tenant_fuel_mappings:
            self._tenant_fuel_mappings[tenant_id] = {}

        self._tenant_fuel_mappings[tenant_id][fuel_alias.lower()] = {
            "name": standard_name,
            "code": fuel_code,
            "category": category
        }
        self.logger.info(f"Registered tenant fuel mapping: {tenant_id}/{fuel_alias}")

    def register_tenant_material_mapping(
        self,
        tenant_id: str,
        material_alias: str,
        standard_name: str,
        material_code: str,
        category: str
    ):
        """
        Register a tenant-specific material name mapping.

        Args:
            tenant_id: Tenant identifier
            material_alias: Tenant's material name
            standard_name: Standard material name
            material_code: Standard material code
            category: Material category
        """
        if tenant_id not in self._tenant_material_mappings:
            self._tenant_material_mappings[tenant_id] = {}

        self._tenant_material_mappings[tenant_id][material_alias.lower()] = {
            "name": standard_name,
            "code": material_code,
            "category": category
        }
        self.logger.info(f"Registered tenant material mapping: {tenant_id}/{material_alias}")

    # =========================================================================
    # PROVENANCE AND UTILITIES
    # =========================================================================

    def _compute_provenance_hash(
        self,
        input_data: Any,
        output_data: Any
    ) -> str:
        """
        Compute SHA-256 provenance hash for audit trail.

        Args:
            input_data: Input data
            output_data: Output data

        Returns:
            SHA-256 hash string
        """
        provenance_str = json.dumps(
            {"input": input_data, "output": output_data},
            sort_keys=True,
            default=str
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def get_supported_dimensions(self) -> List[str]:
        """Get list of supported unit dimensions."""
        return [d.value for d in UnitDimension]

    def get_supported_units(self, dimension: str) -> List[str]:
        """
        Get list of supported units for a dimension.

        Args:
            dimension: Unit dimension

        Returns:
            List of unit names
        """
        try:
            dim = UnitDimension(dimension.lower())
            return list(DIMENSION_UNITS[dim].keys())
        except ValueError:
            return []

    def get_conversion_lineage(
        self,
        value: float,
        from_unit: str,
        to_unit: str
    ) -> Dict[str, Any]:
        """
        Get detailed conversion lineage for audit.

        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Detailed lineage information
        """
        from_normalized = self._normalize_unit_name(from_unit)
        to_normalized = self._normalize_unit_name(to_unit)

        from_dim = self._get_unit_dimension(from_normalized)
        to_dim = self._get_unit_dimension(to_normalized)

        if from_dim is None or to_dim is None or from_dim != to_dim:
            return {
                "valid": False,
                "error": "Invalid conversion"
            }

        units_table = DIMENSION_UNITS[from_dim]
        from_factor = units_table[from_normalized]
        to_factor = units_table[to_normalized]
        conversion_factor = from_factor / to_factor
        converted = Decimal(str(value)) * conversion_factor

        return {
            "valid": True,
            "input": {
                "value": value,
                "unit": from_unit,
                "normalized_unit": from_normalized
            },
            "output": {
                "value": float(converted),
                "unit": to_unit,
                "normalized_unit": to_normalized
            },
            "conversion": {
                "dimension": from_dim.value,
                "base_unit": BASE_UNITS[from_dim],
                "from_to_base_factor": str(from_factor),
                "to_to_base_factor": str(to_factor),
                "conversion_factor": str(conversion_factor)
            },
            "provenance": {
                "source": "GreenLang Unit Normalizer",
                "version": self.VERSION,
                "timestamp": DeterministicClock.now().isoformat(),
                "hash": self._compute_provenance_hash(
                    {"value": value, "from": from_unit, "to": to_unit},
                    {"converted": float(converted)}
                )
            }
        }
