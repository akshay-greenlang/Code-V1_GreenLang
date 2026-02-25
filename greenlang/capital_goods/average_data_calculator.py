# -*- coding: utf-8 -*-
"""
AverageDataCalculatorEngine - Engine 3: Capital Goods Agent (AGENT-MRV-015)

Physical quantity-based calculation engine for GHG Protocol Scope 3 Category 2
(Capital Goods) emissions.  Calculates cradle-to-gate embodied carbon using
average emission factors from ICE Database v3.0, ecoinvent v3.11, and DEFRA 2023.

The average-data method multiplies physical quantities (mass in kg, area in m2,
or unit counts) by industry-average emission factors expressed in kgCO2e per
physical unit.  This approach provides medium accuracy (+/- 30-60 %) and is
recommended when physical data is available but supplier-specific EPD/PCF data
is not.

Key Formulas (Zero-Hallucination - Deterministic Decimal Arithmetic):
    material_emissions = weight_kg x ef_kgco2e_per_kg
    area_emissions     = area_m2   x ef_kgco2e_per_m2
    unit_emissions     = quantity  x ef_kgco2e_per_unit
    transport          = weight_tonnes x distance_km x transport_ef_per_tonne_km
    total              = material_emissions + transport_emissions

Gas Breakdown (typical cradle-to-gate split by material family):
    CO2 : 90-97 % of CO2e for metals, concrete, glass
    CH4 :  1-5  % of CO2e (energy supply chain methane)
    N2O :  0.5-3 % of CO2e (industrial process N2O)

Data Quality:
    DQI scores follow GHG Protocol 5-dimension pedigree matrix.
    Average-data method yields base reliability score of 3.0 (Fair).

Thread Safety:
    Singleton pattern with threading.RLock.  All mutable state is
    protected by the lock.  Decimal arithmetic is inherently thread-safe.

Agent: GL-MRV-S3-002 (AGENT-MRV-015)
Purpose: Physical-quantity-based emissions for capital goods
Regulatory: GHG Protocol Scope 3, ISO 14064-1, CSRD E1, ICE v3.0, ecoinvent 3.11, DEFRA 2023

Example:
    >>> from greenlang.capital_goods.average_data_calculator import AverageDataCalculatorEngine
    >>> from greenlang.capital_goods.models import PhysicalRecord, AssetCategory
    >>> from decimal import Decimal
    >>> engine = AverageDataCalculatorEngine()
    >>> record = PhysicalRecord(
    ...     asset_id="ASSET-001",
    ...     material_type="structural_steel",
    ...     quantity=Decimal("5000"),
    ...     unit="kg",
    ...     weight_kg=Decimal("5000"),
    ...     asset_category=AssetCategory.MACHINERY,
    ... )
    >>> result = engine.calculate(record)
    >>> assert result.emissions_kg_co2e > 0

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-015 Capital Goods (GL-MRV-S3-002)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from greenlang.capital_goods.models import (
    AssetCategory,
    AssetSubCategory,
    AverageDataResult,
    CalculationMethod,
    CAPITAL_PHYSICAL_EMISSION_FACTORS,
    CoverageReport,
    DQIAssessment,
    DQI_QUALITY_TIERS,
    EmissionGas,
    PEDIGREE_UNCERTAINTY_FACTORS,
    PhysicalEF,
    PhysicalEFSource,
    PhysicalRecord,
    UNCERTAINTY_RANGES,
    ZERO,
    ONE,
    ONE_THOUSAND,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Decimal Precision
# ============================================================================

#: Quantization precision for Decimal rounding (8 decimal places).
_PRECISION = Decimal("0.00000001")

#: Newton's method convergence tolerance for square root.
_SQRT_TOLERANCE = Decimal("1E-20")

#: Maximum Newton's method iterations.
_SQRT_MAX_ITERATIONS: int = 100


# ============================================================================
# Embedded Data Tables - UNIT_CONVERSION_TO_KG (~50 entries)
# ============================================================================

UNIT_CONVERSION_TO_KG: Dict[str, Decimal] = {
    # Mass units
    "kg": Decimal("1.0"),
    "kilogram": Decimal("1.0"),
    "kilograms": Decimal("1.0"),
    "g": Decimal("0.001"),
    "gram": Decimal("0.001"),
    "grams": Decimal("0.001"),
    "mg": Decimal("0.000001"),
    "milligram": Decimal("0.000001"),
    "t": Decimal("1000.0"),
    "tonne": Decimal("1000.0"),
    "tonnes": Decimal("1000.0"),
    "metric_ton": Decimal("1000.0"),
    "mt": Decimal("1000.0"),
    "lb": Decimal("0.453592"),
    "lbs": Decimal("0.453592"),
    "pound": Decimal("0.453592"),
    "pounds": Decimal("0.453592"),
    "oz": Decimal("0.0283495"),
    "ounce": Decimal("0.0283495"),
    "ounces": Decimal("0.0283495"),
    "ton_us": Decimal("907.18474"),
    "short_ton": Decimal("907.18474"),
    "ton_uk": Decimal("1016.0469"),
    "long_ton": Decimal("1016.0469"),
    "cwt_us": Decimal("45.3592"),
    "cwt_uk": Decimal("50.8023"),
    "stone": Decimal("6.35029"),
    # Volume-to-mass conversions for common materials (m3 -> kg via density)
    "m3_concrete": Decimal("2400.0"),
    "m3_concrete_25mpa": Decimal("2400.0"),
    "m3_concrete_32mpa": Decimal("2400.0"),
    "m3_concrete_40mpa": Decimal("2400.0"),
    "m3_steel": Decimal("7850.0"),
    "m3_stainless_steel": Decimal("8000.0"),
    "m3_aluminum": Decimal("2700.0"),
    "m3_copper": Decimal("8960.0"),
    "m3_timber": Decimal("500.0"),
    "m3_timber_softwood": Decimal("450.0"),
    "m3_timber_hardwood": Decimal("700.0"),
    "m3_glass": Decimal("2500.0"),
    "m3_brick": Decimal("1900.0"),
    "m3_water": Decimal("1000.0"),
    "m3_asphalt": Decimal("2360.0"),
    "m3_gravel": Decimal("1680.0"),
    "m3_sand": Decimal("1600.0"),
    # Area-based units (passthrough: ef applied per m2)
    "m2": Decimal("1.0"),
    "m2_glass_6mm": Decimal("15.0"),
    "m2_glass_10mm": Decimal("25.0"),
    "m2_glass_double": Decimal("30.0"),
    # Count-based (passthrough: ef applied per unit)
    "unit": Decimal("1.0"),
    "piece": Decimal("1.0"),
    "each": Decimal("1.0"),
    "pcs": Decimal("1.0"),
    "item": Decimal("1.0"),
    "set": Decimal("1.0"),
    # Energy/power units (passthrough: ef applied per kW/kWh/MW)
    "kw": Decimal("1.0"),
    "kwh": Decimal("1.0"),
    "mw": Decimal("1000.0"),
    "mwh": Decimal("1000.0"),
    "hp": Decimal("0.7457"),
    # Length units (for linear materials like pipe, cable)
    "m": Decimal("1.0"),
    "km": Decimal("1000.0"),
    "ft": Decimal("0.3048"),
    "yd": Decimal("0.9144"),
    "mile": Decimal("1609.344"),
}


# ============================================================================
# Embedded Data Tables - TRANSPORT_EFS (~12 modes)
# ============================================================================

#: Transport emission factors in kgCO2e per tonne-km.
#: Sources: DEFRA 2023, GLEC Framework 2023.
TRANSPORT_EFS: Dict[str, Decimal] = {
    "road_truck": Decimal("0.062"),
    "road_van": Decimal("0.137"),
    "road_local": Decimal("0.200"),
    "rail_freight": Decimal("0.028"),
    "electric_rail": Decimal("0.005"),
    "sea_container": Decimal("0.016"),
    "sea_bulk": Decimal("0.005"),
    "air_freight": Decimal("0.602"),
    "air_short_haul": Decimal("0.673"),
    "inland_waterway": Decimal("0.032"),
    "pipeline": Decimal("0.015"),
    "multimodal": Decimal("0.045"),
}


# ============================================================================
# Embedded Data Tables - CATEGORY_DEFAULT_MATERIAL (~20 entries)
# ============================================================================

#: Maps AssetSubCategory values to default material_type for EF lookup.
CATEGORY_DEFAULT_MATERIAL: Dict[str, str] = {
    # Buildings
    "OFFICE_BUILDING": "concrete_25mpa",
    "WAREHOUSE": "structural_steel",
    "MANUFACTURING_FACILITY": "structural_steel",
    "RETAIL_STORE": "concrete_25mpa",
    # Machinery
    "CNC_MACHINE": "structural_steel",
    "PRESS": "structural_steel",
    "CRANE": "structural_steel",
    "CONVEYOR": "structural_steel",
    "INDUSTRIAL_ROBOT": "aluminum_extrusion",
    # Equipment
    "HVAC": "structural_steel",
    "ELECTRICAL_PANEL": "structural_steel",
    "GENERATOR": "structural_steel",
    "COMPRESSOR": "structural_steel",
    "TRANSFORMER": "transformer_per_unit",
    # Vehicles
    "PASSENGER_CAR": "structural_steel",
    "LIGHT_TRUCK": "structural_steel",
    "HEAVY_TRUCK": "structural_steel",
    "FORKLIFT": "structural_steel",
    "VAN": "structural_steel",
    # IT Infrastructure
    "SERVER": "server_per_unit",
    "NETWORK_SWITCH": "network_switch_per_unit",
    "STORAGE_ARRAY": "server_per_unit",
    "UPS": "ups_per_unit",
    "RACK_ENCLOSURE": "structural_steel",
    # Furniture & Fixtures
    "OFFICE_DESK": "timber_softwood",
    "OFFICE_CHAIR": "structural_steel",
    "SHELVING": "structural_steel",
    "PARTITION": "plasterboard",
    # Land improvements
    "PAVING": "asphalt",
    "LANDSCAPING": "timber_softwood",
    "FENCING": "structural_steel",
    "DRAINAGE": "hdpe_pipe",
    # Leasehold improvements
    "FITOUT_GENERAL": "plasterboard",
    "INTERIOR_PARTITION": "plasterboard",
    "FLOORING": "vinyl_flooring",
    "CEILING": "plasterboard",
    # Renewable / Electrical
    "SOLAR_PANEL": "solar_panel_per_kw",
    "WIND_TURBINE": "wind_turbine_per_mw",
    "BATTERY_STORAGE": "battery_li_ion_per_kwh",
    "ELECTRIC_MOTOR": "electric_motor_per_kw",
}

#: Maps top-level AssetCategory to default material for fallback.
ASSET_CATEGORY_DEFAULT_MATERIAL: Dict[str, str] = {
    "buildings": "concrete_25mpa",
    "machinery": "structural_steel",
    "equipment": "structural_steel",
    "vehicles": "structural_steel",
    "it_infrastructure": "server_per_unit",
    "furniture_fixtures": "timber_softwood",
    "land_improvements": "asphalt",
    "leasehold_improvements": "plasterboard",
}


# ============================================================================
# Embedded Data Tables - MATERIAL_DENSITY_KG_PER_L (~18 entries)
# ============================================================================

#: Material density in kg per litre for volume-to-mass conversions.
MATERIAL_DENSITY_KG_PER_L: Dict[str, Decimal] = {
    "structural_steel": Decimal("7.85"),
    "reinforcing_steel": Decimal("7.85"),
    "stainless_steel": Decimal("8.00"),
    "aluminum_sheet": Decimal("2.70"),
    "aluminum_extrusion": Decimal("2.70"),
    "copper_pipe": Decimal("8.96"),
    "copper_wire": Decimal("8.96"),
    "concrete_25mpa": Decimal("2.40"),
    "concrete_32mpa": Decimal("2.40"),
    "concrete_40mpa": Decimal("2.40"),
    "glass_float": Decimal("2.50"),
    "glass_tempered": Decimal("2.50"),
    "timber_softwood": Decimal("0.45"),
    "timber_hardwood": Decimal("0.70"),
    "brick": Decimal("1.90"),
    "pvc_pipe": Decimal("1.40"),
    "hdpe_pipe": Decimal("0.95"),
    "asphalt": Decimal("2.36"),
}


# ============================================================================
# Embedded Data Tables - Gas Breakdown Ratios by Material Family
# ============================================================================

#: Default gas breakdown ratios (CO2 share, CH4 share, N2O share) for
#: different material families.  Remaining balance after CO2+CH4 = N2O.
#: These are based on typical cradle-to-gate LCA profiles.
_GAS_BREAKDOWN: Dict[str, Tuple[Decimal, Decimal, Decimal]] = {
    # Metals: dominated by CO2 from smelting/refining energy
    "structural_steel": (Decimal("0.950"), Decimal("0.035"), Decimal("0.015")),
    "reinforcing_steel": (Decimal("0.945"), Decimal("0.038"), Decimal("0.017")),
    "stainless_steel": (Decimal("0.940"), Decimal("0.040"), Decimal("0.020")),
    "aluminum_sheet": (Decimal("0.930"), Decimal("0.045"), Decimal("0.025")),
    "aluminum_extrusion": (Decimal("0.935"), Decimal("0.042"), Decimal("0.023")),
    "copper_pipe": (Decimal("0.942"), Decimal("0.038"), Decimal("0.020")),
    "copper_wire": (Decimal("0.942"), Decimal("0.038"), Decimal("0.020")),
    # Concrete: process CO2 from calcination + energy CO2
    "concrete_25mpa": (Decimal("0.970"), Decimal("0.020"), Decimal("0.010")),
    "concrete_32mpa": (Decimal("0.970"), Decimal("0.020"), Decimal("0.010")),
    "concrete_40mpa": (Decimal("0.965"), Decimal("0.023"), Decimal("0.012")),
    "concrete_precast": (Decimal("0.968"), Decimal("0.022"), Decimal("0.010")),
    "brick": (Decimal("0.960"), Decimal("0.025"), Decimal("0.015")),
    # Glass: energy-intensive melting
    "glass_float": (Decimal("0.955"), Decimal("0.030"), Decimal("0.015")),
    "glass_tempered": (Decimal("0.955"), Decimal("0.030"), Decimal("0.015")),
    "glass_double_glazed": (Decimal("0.950"), Decimal("0.033"), Decimal("0.017")),
    # Timber: biogenic / low process emissions
    "timber_softwood": (Decimal("0.920"), Decimal("0.050"), Decimal("0.030")),
    "timber_hardwood": (Decimal("0.925"), Decimal("0.048"), Decimal("0.027")),
    "timber_glulam": (Decimal("0.918"), Decimal("0.052"), Decimal("0.030")),
    # Interior materials
    "plasterboard": (Decimal("0.955"), Decimal("0.030"), Decimal("0.015")),
    "ceramic_tiles": (Decimal("0.960"), Decimal("0.025"), Decimal("0.015")),
    "carpet": (Decimal("0.920"), Decimal("0.050"), Decimal("0.030")),
    "vinyl_flooring": (Decimal("0.915"), Decimal("0.055"), Decimal("0.030")),
    "paint_water_based": (Decimal("0.930"), Decimal("0.045"), Decimal("0.025")),
    "paint_solvent_based": (Decimal("0.910"), Decimal("0.058"), Decimal("0.032")),
    # Insulation
    "insulation_mineral_wool": (Decimal("0.945"), Decimal("0.035"), Decimal("0.020")),
    "insulation_eps": (Decimal("0.910"), Decimal("0.058"), Decimal("0.032")),
    "insulation_xps": (Decimal("0.905"), Decimal("0.060"), Decimal("0.035")),
    "insulation_pir": (Decimal("0.908"), Decimal("0.058"), Decimal("0.034")),
    # Piping & roofing
    "pvc_pipe": (Decimal("0.912"), Decimal("0.055"), Decimal("0.033")),
    "hdpe_pipe": (Decimal("0.915"), Decimal("0.052"), Decimal("0.033")),
    "roofing_membrane": (Decimal("0.920"), Decimal("0.050"), Decimal("0.030")),
    "asphalt": (Decimal("0.965"), Decimal("0.023"), Decimal("0.012")),
    # IT / Electronics (manufacturing-heavy, diverse supply chain)
    "server_per_unit": (Decimal("0.900"), Decimal("0.065"), Decimal("0.035")),
    "laptop_per_unit": (Decimal("0.895"), Decimal("0.068"), Decimal("0.037")),
    "desktop_per_unit": (Decimal("0.898"), Decimal("0.066"), Decimal("0.036")),
    "monitor_per_unit": (Decimal("0.900"), Decimal("0.065"), Decimal("0.035")),
    "network_switch_per_unit": (Decimal("0.905"), Decimal("0.062"), Decimal("0.033")),
    "ups_per_unit": (Decimal("0.910"), Decimal("0.058"), Decimal("0.032")),
    "led_panel_per_unit": (Decimal("0.908"), Decimal("0.060"), Decimal("0.032")),
    # Renewable energy & storage
    "solar_panel_per_kw": (Decimal("0.890"), Decimal("0.070"), Decimal("0.040")),
    "wind_turbine_per_mw": (Decimal("0.920"), Decimal("0.052"), Decimal("0.028")),
    "battery_li_ion_per_kwh": (Decimal("0.880"), Decimal("0.075"), Decimal("0.045")),
    # Electrical
    "transformer_per_unit": (Decimal("0.935"), Decimal("0.042"), Decimal("0.023")),
    "electric_motor_per_kw": (Decimal("0.940"), Decimal("0.040"), Decimal("0.020")),
}

#: Default gas breakdown for unknown materials.
_DEFAULT_GAS_BREAKDOWN: Tuple[Decimal, Decimal, Decimal] = (
    Decimal("0.940"),
    Decimal("0.040"),
    Decimal("0.020"),
)


# ============================================================================
# Embedded Data Tables - ICE Database v3.0 Emission Factors
# ============================================================================

#: ICE Database (Inventory of Carbon and Energy) v3.0, University of Bath.
#: Factors in kgCO2e per kg of material (cradle-to-gate).
ICE_DATABASE_FACTORS: Dict[str, Decimal] = {
    "structural_steel": Decimal("1.55"),
    "reinforcing_steel": Decimal("1.99"),
    "stainless_steel": Decimal("6.15"),
    "aluminum_sheet": Decimal("8.24"),
    "aluminum_extrusion": Decimal("6.67"),
    "copper_pipe": Decimal("3.81"),
    "copper_wire": Decimal("3.64"),
    "concrete_25mpa": Decimal("0.132"),
    "concrete_32mpa": Decimal("0.163"),
    "concrete_40mpa": Decimal("0.188"),
    "concrete_precast": Decimal("0.176"),
    "brick": Decimal("0.24"),
    "glass_float": Decimal("1.22"),
    "glass_tempered": Decimal("1.67"),
    "glass_double_glazed": Decimal("2.89"),
    "timber_softwood": Decimal("0.51"),
    "timber_hardwood": Decimal("0.86"),
    "timber_glulam": Decimal("0.45"),
    "plasterboard": Decimal("0.39"),
    "ceramic_tiles": Decimal("0.74"),
    "carpet": Decimal("5.88"),
    "vinyl_flooring": Decimal("4.21"),
    "paint_water_based": Decimal("2.41"),
    "paint_solvent_based": Decimal("3.56"),
    "insulation_mineral_wool": Decimal("1.28"),
    "insulation_eps": Decimal("3.29"),
    "insulation_xps": Decimal("3.48"),
    "insulation_pir": Decimal("3.44"),
    "pvc_pipe": Decimal("3.23"),
    "hdpe_pipe": Decimal("2.52"),
    "roofing_membrane": Decimal("4.12"),
    "asphalt": Decimal("0.043"),
}


# ============================================================================
# Embedded Data Tables - ecoinvent v3.11 Emission Factors
# ============================================================================

#: ecoinvent v3.11 cradle-to-gate emission factors in kgCO2e per kg.
#: Where available, factors use market activity for RoW (Rest of World).
ECOINVENT_FACTORS: Dict[str, Decimal] = {
    "structural_steel": Decimal("1.89"),
    "reinforcing_steel": Decimal("2.14"),
    "stainless_steel": Decimal("6.68"),
    "aluminum_sheet": Decimal("9.16"),
    "aluminum_extrusion": Decimal("7.42"),
    "copper_pipe": Decimal("4.22"),
    "copper_wire": Decimal("4.05"),
    "concrete_25mpa": Decimal("0.145"),
    "concrete_32mpa": Decimal("0.178"),
    "concrete_40mpa": Decimal("0.205"),
    "concrete_precast": Decimal("0.192"),
    "brick": Decimal("0.271"),
    "glass_float": Decimal("1.35"),
    "glass_tempered": Decimal("1.82"),
    "glass_double_glazed": Decimal("3.15"),
    "timber_softwood": Decimal("0.46"),
    "timber_hardwood": Decimal("0.79"),
    "timber_glulam": Decimal("0.42"),
    "plasterboard": Decimal("0.42"),
    "ceramic_tiles": Decimal("0.81"),
    "carpet": Decimal("6.12"),
    "vinyl_flooring": Decimal("4.55"),
    "paint_water_based": Decimal("2.58"),
    "paint_solvent_based": Decimal("3.78"),
    "insulation_mineral_wool": Decimal("1.38"),
    "insulation_eps": Decimal("3.52"),
    "insulation_xps": Decimal("3.71"),
    "insulation_pir": Decimal("3.68"),
    "pvc_pipe": Decimal("3.45"),
    "hdpe_pipe": Decimal("2.68"),
    "roofing_membrane": Decimal("4.38"),
    "asphalt": Decimal("0.048"),
    "server_per_unit": Decimal("530.0"),
    "laptop_per_unit": Decimal("370.0"),
    "desktop_per_unit": Decimal("295.0"),
    "monitor_per_unit": Decimal("195.0"),
    "network_switch_per_unit": Decimal("128.0"),
    "ups_per_unit": Decimal("475.0"),
    "solar_panel_per_kw": Decimal("1280.0"),
    "battery_li_ion_per_kwh": Decimal("80.0"),
}


# ============================================================================
# Embedded Data Tables - DEFRA 2023 Emission Factors
# ============================================================================

#: DEFRA/DESNZ 2023 emission conversion factors in kgCO2e per kg.
#: UK Department for Environment, Food & Rural Affairs.
DEFRA_FACTORS: Dict[str, Decimal] = {
    "structural_steel": Decimal("1.46"),
    "reinforcing_steel": Decimal("1.83"),
    "stainless_steel": Decimal("5.89"),
    "aluminum_sheet": Decimal("7.80"),
    "aluminum_extrusion": Decimal("6.33"),
    "copper_pipe": Decimal("3.55"),
    "copper_wire": Decimal("3.42"),
    "concrete_25mpa": Decimal("0.126"),
    "concrete_32mpa": Decimal("0.155"),
    "concrete_40mpa": Decimal("0.179"),
    "concrete_precast": Decimal("0.168"),
    "brick": Decimal("0.23"),
    "glass_float": Decimal("1.18"),
    "glass_tempered": Decimal("1.58"),
    "glass_double_glazed": Decimal("2.74"),
    "timber_softwood": Decimal("0.48"),
    "timber_hardwood": Decimal("0.82"),
    "timber_glulam": Decimal("0.43"),
    "plasterboard": Decimal("0.37"),
    "ceramic_tiles": Decimal("0.71"),
    "carpet": Decimal("5.62"),
    "vinyl_flooring": Decimal("4.05"),
    "paint_water_based": Decimal("2.32"),
    "paint_solvent_based": Decimal("3.41"),
    "insulation_mineral_wool": Decimal("1.22"),
    "insulation_eps": Decimal("3.14"),
    "insulation_xps": Decimal("3.32"),
    "insulation_pir": Decimal("3.29"),
    "pvc_pipe": Decimal("3.10"),
    "hdpe_pipe": Decimal("2.41"),
    "roofing_membrane": Decimal("3.95"),
    "asphalt": Decimal("0.040"),
    "server_per_unit": Decimal("480.0"),
    "laptop_per_unit": Decimal("330.0"),
    "desktop_per_unit": Decimal("265.0"),
    "monitor_per_unit": Decimal("170.0"),
    "network_switch_per_unit": Decimal("115.0"),
    "ups_per_unit": Decimal("430.0"),
    "solar_panel_per_kw": Decimal("1150.0"),
    "wind_turbine_per_mw": Decimal("330000.0"),
    "battery_li_ion_per_kwh": Decimal("72.0"),
    "transformer_per_unit": Decimal("760.0"),
    "electric_motor_per_kw": Decimal("14.2"),
}


# ============================================================================
# Embedded Data Tables - EF Source Priority
# ============================================================================

#: Priority ranking for EF source selection (lower number = higher priority).
#: ICE is preferred for construction materials, ecoinvent for complex products,
#: DEFRA as general-purpose UK/global fallback.
EF_SOURCE_PRIORITY: Dict[str, int] = {
    "ice_database": 1,
    "ecoinvent": 2,
    "defra": 3,
    "custom": 4,
}


# ============================================================================
# Embedded Data Tables - Regional Adjustment Factors
# ============================================================================

#: Regional grid carbon intensity adjustment factors relative to global average.
#: Used to adjust cradle-to-gate EFs for regional energy mix differences.
REGIONAL_ADJUSTMENT_FACTORS: Dict[str, Decimal] = {
    "GLOBAL": Decimal("1.00"),
    "EU": Decimal("0.85"),
    "EU27": Decimal("0.85"),
    "UK": Decimal("0.78"),
    "US": Decimal("1.02"),
    "USA": Decimal("1.02"),
    "CN": Decimal("1.25"),
    "CHINA": Decimal("1.25"),
    "IN": Decimal("1.30"),
    "INDIA": Decimal("1.30"),
    "JP": Decimal("0.95"),
    "JAPAN": Decimal("0.95"),
    "AU": Decimal("1.15"),
    "AUSTRALIA": Decimal("1.15"),
    "CA": Decimal("0.72"),
    "CANADA": Decimal("0.72"),
    "BR": Decimal("0.55"),
    "BRAZIL": Decimal("0.55"),
    "SE": Decimal("0.35"),
    "SWEDEN": Decimal("0.35"),
    "NO": Decimal("0.30"),
    "NORWAY": Decimal("0.30"),
    "DE": Decimal("0.88"),
    "GERMANY": Decimal("0.88"),
    "FR": Decimal("0.40"),
    "FRANCE": Decimal("0.40"),
    "ZA": Decimal("1.40"),
    "SOUTH_AFRICA": Decimal("1.40"),
    "KR": Decimal("1.10"),
    "SOUTH_KOREA": Decimal("1.10"),
    "NORDIC": Decimal("0.33"),
    "MIDDLE_EAST": Decimal("1.20"),
    "SOUTHEAST_ASIA": Decimal("1.15"),
    "LATIN_AMERICA": Decimal("0.65"),
    "AFRICA": Decimal("1.05"),
    "OCEANIA": Decimal("1.10"),
}


# ============================================================================
# Embedded Data Tables - Building EFs (per m2)
# ============================================================================

#: Building embodied carbon emission factors in kgCO2e per m2 of gross floor area.
#: Sources: RICS 2023, LETI Embodied Carbon Target, ICE Database v3.0.
BUILDING_EF_PER_M2: Dict[str, Decimal] = {
    "office_building": Decimal("500.0"),
    "warehouse": Decimal("350.0"),
    "manufacturing_facility": Decimal("600.0"),
    "retail_store": Decimal("450.0"),
    "hospital": Decimal("750.0"),
    "school": Decimal("400.0"),
    "residential_low_rise": Decimal("380.0"),
    "residential_high_rise": Decimal("550.0"),
    "data_centre": Decimal("700.0"),
    "logistics_hub": Decimal("320.0"),
    "laboratory": Decimal("680.0"),
    "cold_storage": Decimal("420.0"),
}


# ============================================================================
# Embedded Data Tables - Equipment / Vehicle / IT EFs
# ============================================================================

#: Equipment emission factors in kgCO2e per kg of equipment weight.
EQUIPMENT_EF_PER_KG: Dict[str, Decimal] = {
    "hvac": Decimal("2.80"),
    "electrical_panel": Decimal("2.50"),
    "generator": Decimal("3.20"),
    "compressor": Decimal("2.90"),
    "transformer": Decimal("2.60"),
    "cnc_machine": Decimal("2.40"),
    "press": Decimal("2.20"),
    "crane": Decimal("1.80"),
    "conveyor": Decimal("2.10"),
    "industrial_robot": Decimal("4.50"),
    "pump": Decimal("2.70"),
    "boiler": Decimal("2.30"),
}

#: Vehicle emission factors in kgCO2e per vehicle (cradle-to-gate).
VEHICLE_EF_PER_UNIT: Dict[str, Decimal] = {
    "passenger_car": Decimal("6000.0"),
    "passenger_car_electric": Decimal("8500.0"),
    "passenger_car_hybrid": Decimal("7200.0"),
    "light_truck": Decimal("8000.0"),
    "heavy_truck": Decimal("15000.0"),
    "van": Decimal("7500.0"),
    "forklift": Decimal("5000.0"),
    "forklift_electric": Decimal("6200.0"),
    "bus": Decimal("25000.0"),
    "bus_electric": Decimal("35000.0"),
    "motorcycle": Decimal("2500.0"),
    "tractor": Decimal("12000.0"),
}

#: IT equipment emission factors in kgCO2e per unit (cradle-to-gate).
IT_EQUIPMENT_EF_PER_UNIT: Dict[str, Decimal] = {
    "server": Decimal("500.0"),
    "server_blade": Decimal("350.0"),
    "laptop": Decimal("350.0"),
    "desktop": Decimal("280.0"),
    "monitor": Decimal("180.0"),
    "monitor_large": Decimal("250.0"),
    "network_switch": Decimal("120.0"),
    "network_router": Decimal("150.0"),
    "storage_array": Decimal("600.0"),
    "ups": Decimal("450.0"),
    "ups_large": Decimal("800.0"),
    "rack_enclosure": Decimal("200.0"),
    "printer": Decimal("100.0"),
    "printer_large": Decimal("350.0"),
    "phone_voip": Decimal("25.0"),
    "tablet": Decimal("75.0"),
}


# ============================================================================
# Embedded Data Tables - Uncertainty by EF Source
# ============================================================================

#: Uncertainty percentage range by EF source (min %, max %).
EF_SOURCE_UNCERTAINTY: Dict[str, Tuple[Decimal, Decimal]] = {
    "ice_database": (Decimal("25"), Decimal("40")),
    "ecoinvent": (Decimal("20"), Decimal("35")),
    "defra": (Decimal("30"), Decimal("50")),
    "custom": (Decimal("40"), Decimal("70")),
    "world_steel": (Decimal("15"), Decimal("25")),
    "iai": (Decimal("15"), Decimal("25")),
}


# ============================================================================
# Helper Functions
# ============================================================================


def _quantize(value: Decimal) -> Decimal:
    """Quantize a Decimal to 8 decimal places with ROUND_HALF_UP.

    Args:
        value: Decimal value to quantize.

    Returns:
        Quantized Decimal value.
    """
    return value.quantize(_PRECISION, rounding=ROUND_HALF_UP)


def _to_json_safe(obj: Any) -> str:
    """Serialize an object to JSON, handling Decimal and datetime.

    Args:
        obj: Object to serialize.

    Returns:
        JSON string representation.
    """
    def _default(o: Any) -> Any:
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, uuid.UUID):
            return str(o)
        if hasattr(o, "model_dump"):
            return o.model_dump()
        if hasattr(o, "value"):
            return o.value
        return str(o)

    return json.dumps(obj, default=_default, sort_keys=True)


# ============================================================================
# AverageDataCalculatorEngine
# ============================================================================


class AverageDataCalculatorEngine:
    """
    Physical quantity-based emission calculator for capital goods (Category 2).

    Implements the average-data calculation method per GHG Protocol Scope 3
    Technical Guidance, using emission factors from ICE Database v3.0,
    ecoinvent v3.11, and DEFRA 2023.  All calculations use deterministic
    Python Decimal arithmetic with 8 decimal places to ensure zero-hallucination
    reproducibility.

    Thread Safety:
        Singleton pattern with threading.RLock.  The instance is created lazily
        on first access via get_instance() or the constructor.  All mutable
        state (counters, caches) is guarded by the lock.

    Attributes:
        _calc_count: Running count of calculations performed.
        _total_emissions: Running total of emissions calculated (kgCO2e).
        _error_count: Running count of calculation errors.
        _batch_count: Running count of batch calculations performed.
        _last_calc_time_ms: Duration of last calculation in milliseconds.
        _ef_cache: Cached emission factor lookups for performance.

    Example:
        >>> engine = AverageDataCalculatorEngine()
        >>> record = PhysicalRecord(
        ...     asset_id="A-001",
        ...     material_type="structural_steel",
        ...     quantity=Decimal("5000"),
        ...     unit="kg",
        ...     weight_kg=Decimal("5000"),
        ... )
        >>> result = engine.calculate(record)
        >>> assert result.emissions_kg_co2e == Decimal("7750.00000000")
    """

    _instance: Optional[AverageDataCalculatorEngine] = None
    _lock: threading.RLock = threading.RLock()

    def __new__(cls, *args: Any, **kwargs: Any) -> AverageDataCalculatorEngine:
        """Create or return the singleton instance (thread-safe).

        Returns:
            The singleton AverageDataCalculatorEngine instance.
        """
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
            return cls._instance

    def __init__(self) -> None:
        """Initialize the engine (only on first instantiation).

        Sets up internal counters, caches, and emission factor tables.
        Subsequent calls to __init__ are no-ops due to the _initialized guard.
        """
        with self._lock:
            if self._initialized:
                return
            self._calc_count: int = 0
            self._total_emissions: Decimal = ZERO
            self._error_count: int = 0
            self._batch_count: int = 0
            self._last_calc_time_ms: Decimal = ZERO
            self._ef_cache: Dict[str, PhysicalEF] = {}
            self._initialized = True
            logger.info(
                "AverageDataCalculatorEngine initialized: "
                "ICE=%d factors, ecoinvent=%d factors, DEFRA=%d factors, "
                "units=%d, transport_modes=%d",
                len(ICE_DATABASE_FACTORS),
                len(ECOINVENT_FACTORS),
                len(DEFRA_FACTORS),
                len(UNIT_CONVERSION_TO_KG),
                len(TRANSPORT_EFS),
            )

    # ========================================================================
    # Public Method 1: calculate
    # ========================================================================

    def calculate(
        self,
        record: PhysicalRecord,
        config: Optional[Dict[str, Any]] = None,
    ) -> AverageDataResult:
        """Calculate emissions for a single physical record.

        Implements the average-data method: physical quantity multiplied by
        industry-average cradle-to-gate emission factor.  Optionally adds
        transport emissions if transport_distance_km and transport_mode are
        specified in config.

        Args:
            record: Physical record with material/quantity data.
            config: Optional configuration overrides:
                - ef_source: Preferred EF source ("ice_database", "ecoinvent", "defra").
                - transport_distance_km: Distance for transport emissions.
                - transport_mode: Transport mode key from TRANSPORT_EFS.
                - region: Region code for regional EF adjustment.
                - include_transport: Whether to include transport (default True).

        Returns:
            AverageDataResult with calculated emissions and metadata.

        Raises:
            ValueError: If record validation fails.

        Example:
            >>> result = engine.calculate(record)
            >>> assert result.emissions_kg_co2e > 0
        """
        start_time = time.monotonic()
        config = config or {}

        try:
            # Step 1: Validate the input record
            errors = self.validate_record(record)
            if errors:
                raise ValueError(
                    f"PhysicalRecord validation failed: {'; '.join(errors)}"
                )

            # Step 2: Resolve material type
            material_type = self._resolve_material_type(record)

            # Step 3: Select best emission factor
            preferred_source = config.get("ef_source", None)
            ef_value, ef_source_str = self._select_ef(
                material_type, preferred_source
            )

            # Step 4: Convert quantity to calculation basis
            weight_kg = self._resolve_weight_kg(record, material_type)
            area_m2 = record.area_m2

            # Step 5: Calculate material emissions
            material_emissions = self._compute_material_emissions(
                record=record,
                material_type=material_type,
                weight_kg=weight_kg,
                area_m2=area_m2,
                ef_value=ef_value,
            )

            # Step 6: Apply regional adjustment if specified
            region = config.get("region", None)
            if region:
                adjustment = self.apply_regional_adjustment(ONE, region)
                material_emissions = _quantize(material_emissions * adjustment)

            # Step 7: Calculate transport emissions
            transport_emissions = ZERO
            include_transport = config.get("include_transport", True)
            if include_transport:
                transport_distance_km = config.get("transport_distance_km", None)
                transport_mode = config.get("transport_mode", None)
                if transport_distance_km is not None and transport_mode is not None:
                    transport_distance_km = Decimal(str(transport_distance_km))
                    transport_emissions = self.calculate_transport_emissions(
                        weight_kg=weight_kg if weight_kg else ZERO,
                        distance_km=transport_distance_km,
                        mode=transport_mode,
                    )

            # Step 8: Compute total emissions
            total_emissions = _quantize(material_emissions + transport_emissions)

            # Step 9: Gas breakdown
            gas_breakdown = self.split_gas_breakdown(total_emissions, material_type)

            # Step 10: Determine EF source enum
            ef_source_enum = self._str_to_ef_source(ef_source_str)

            # Step 11: Build result
            calc_time_ms = Decimal(str(
                (time.monotonic() - start_time) * 1000
            ))
            self._last_calc_time_ms = _quantize(calc_time_ms)

            result = AverageDataResult(
                record_id=str(uuid.uuid4()),
                asset_id=record.asset_id,
                quantity=record.quantity,
                unit=record.unit,
                ef_value=ef_value,
                ef_source=ef_source_enum,
                emissions_kg_co2e=total_emissions,
                co2=gas_breakdown.get("CO2", ZERO),
                ch4=gas_breakdown.get("CH4", ZERO),
                n2o=gas_breakdown.get("N2O", ZERO),
                transport_emissions=transport_emissions,
                dqi_score=Decimal("3.0"),
                uncertainty_pct=Decimal("45.0"),
                method=CalculationMethod.AVERAGE_DATA,
                provenance_hash="",
            )

            # Step 12: Compute provenance hash
            prov_hash = self.compute_provenance_hash(record, result)

            # Step 13: Re-create result with hash (frozen model)
            result = AverageDataResult(
                record_id=result.record_id,
                asset_id=result.asset_id,
                quantity=result.quantity,
                unit=result.unit,
                ef_value=result.ef_value,
                ef_source=result.ef_source,
                emissions_kg_co2e=result.emissions_kg_co2e,
                co2=result.co2,
                ch4=result.ch4,
                n2o=result.n2o,
                transport_emissions=result.transport_emissions,
                dqi_score=result.dqi_score,
                uncertainty_pct=result.uncertainty_pct,
                method=result.method,
                provenance_hash=prov_hash,
            )

            # Step 14: Update running counters
            with self._lock:
                self._calc_count += 1
                self._total_emissions += total_emissions

            logger.info(
                "Average-data calculation complete: asset_id=%s, "
                "material=%s, emissions=%.4f kgCO2e, ef_source=%s, "
                "transport=%.4f kgCO2e, time=%.2f ms",
                record.asset_id,
                material_type,
                total_emissions,
                ef_source_str,
                transport_emissions,
                calc_time_ms,
            )
            return result

        except ValueError:
            with self._lock:
                self._error_count += 1
            raise

        except Exception as exc:
            with self._lock:
                self._error_count += 1
            logger.error(
                "Average-data calculation failed: asset_id=%s, error=%s",
                record.asset_id,
                str(exc),
                exc_info=True,
            )
            raise ValueError(
                f"Average-data calculation failed for asset {record.asset_id}: "
                f"{str(exc)}"
            ) from exc

    # ========================================================================
    # Public Method 2: calculate_batch
    # ========================================================================

    def calculate_batch(
        self,
        records: List[PhysicalRecord],
        config: Optional[Dict[str, Any]] = None,
    ) -> List[AverageDataResult]:
        """Calculate emissions for a batch of physical records.

        Processes each record independently.  Errors on individual records
        are logged but do not halt the batch; failed records are skipped
        and a warning is emitted.

        Args:
            records: List of PhysicalRecord instances.
            config: Optional configuration overrides applied to all records.

        Returns:
            List of AverageDataResult instances (one per successful record).

        Example:
            >>> results = engine.calculate_batch([record1, record2, record3])
            >>> assert len(results) <= 3
        """
        start_time = time.monotonic()
        config = config or {}
        results: List[AverageDataResult] = []
        error_count = 0

        logger.info(
            "Starting batch calculation: %d records", len(records)
        )

        for idx, record in enumerate(records):
            try:
                result = self.calculate(record, config)
                results.append(result)
            except (ValueError, Exception) as exc:
                error_count += 1
                logger.warning(
                    "Batch record %d/%d failed (asset_id=%s): %s",
                    idx + 1,
                    len(records),
                    record.asset_id,
                    str(exc),
                )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        with self._lock:
            self._batch_count += 1

        logger.info(
            "Batch calculation complete: %d/%d succeeded, %d errors, "
            "%.2f ms total",
            len(results),
            len(records),
            error_count,
            elapsed_ms,
        )
        return results

    # ========================================================================
    # Public Method 3: convert_to_kg
    # ========================================================================

    def convert_to_kg(self, quantity: Decimal, unit: str) -> Decimal:
        """Convert a physical quantity to kilograms using UNIT_CONVERSION_TO_KG.

        Args:
            quantity: Numeric quantity.
            unit: Unit string (case-insensitive).

        Returns:
            Quantity converted to kilograms.

        Raises:
            ValueError: If the unit is not recognized.

        Example:
            >>> engine.convert_to_kg(Decimal("5"), "tonne")
            Decimal('5000.00000000')
        """
        unit_lower = unit.lower().strip()
        factor = UNIT_CONVERSION_TO_KG.get(unit_lower)
        if factor is None:
            raise ValueError(
                f"Unknown unit '{unit}'. Supported units: "
                f"{sorted(UNIT_CONVERSION_TO_KG.keys())}"
            )
        return _quantize(quantity * factor)

    # ========================================================================
    # Public Method 4: lookup_physical_ef
    # ========================================================================

    def lookup_physical_ef(
        self, material_type: str, source: str = "ice_database"
    ) -> PhysicalEF:
        """Look up a physical emission factor for a material from a named source.

        Args:
            material_type: Material key (e.g., "structural_steel").
            source: EF source ("ice_database", "ecoinvent", "defra").

        Returns:
            PhysicalEF instance with the factor and metadata.

        Raises:
            ValueError: If material not found in the specified source.

        Example:
            >>> ef = engine.lookup_physical_ef("structural_steel", "ice_database")
            >>> assert ef.factor_kg_co2e_per_unit == Decimal("1.55")
        """
        cache_key = f"{material_type}|{source}"
        with self._lock:
            if cache_key in self._ef_cache:
                return self._ef_cache[cache_key]

        source_lower = source.lower().strip()
        factor = self._get_factor_from_source(material_type, source_lower)
        if factor is None:
            raise ValueError(
                f"Material '{material_type}' not found in source '{source}'"
            )

        ef_source_enum = self._str_to_ef_source(source_lower)
        unit = self._determine_ef_unit(material_type)

        ef = PhysicalEF(
            material_type=material_type,
            factor_kg_co2e_per_unit=factor,
            unit=unit,
            source=ef_source_enum,
            region="GLOBAL",
        )

        with self._lock:
            self._ef_cache[cache_key] = ef

        return ef

    # ========================================================================
    # Public Method 5: calculate_material_emissions
    # ========================================================================

    def calculate_material_emissions(
        self, weight_kg: Decimal, ef: Decimal
    ) -> Decimal:
        """Calculate mass-based emissions: weight_kg x ef_kgco2e_per_kg.

        This is the fundamental formula for materials measured by weight.

        Args:
            weight_kg: Weight of material in kilograms.
            ef: Emission factor in kgCO2e per kg.

        Returns:
            Emissions in kgCO2e.

        Example:
            >>> engine.calculate_material_emissions(Decimal("5000"), Decimal("1.55"))
            Decimal('7750.00000000')
        """
        if weight_kg < ZERO:
            raise ValueError(f"weight_kg must be >= 0, got {weight_kg}")
        if ef < ZERO:
            raise ValueError(f"Emission factor must be >= 0, got {ef}")
        return _quantize(weight_kg * ef)

    # ========================================================================
    # Public Method 6: calculate_area_emissions
    # ========================================================================

    def calculate_area_emissions(
        self, area_m2: Decimal, ef_per_m2: Decimal
    ) -> Decimal:
        """Calculate area-based emissions: area_m2 x ef_kgco2e_per_m2.

        Used for building embodied carbon calculations.

        Args:
            area_m2: Area in square metres.
            ef_per_m2: Emission factor in kgCO2e per m2.

        Returns:
            Emissions in kgCO2e.

        Example:
            >>> engine.calculate_area_emissions(Decimal("2000"), Decimal("500"))
            Decimal('1000000.00000000')
        """
        if area_m2 < ZERO:
            raise ValueError(f"area_m2 must be >= 0, got {area_m2}")
        if ef_per_m2 < ZERO:
            raise ValueError(f"ef_per_m2 must be >= 0, got {ef_per_m2}")
        return _quantize(area_m2 * ef_per_m2)

    # ========================================================================
    # Public Method 7: calculate_unit_emissions
    # ========================================================================

    def calculate_unit_emissions(
        self, quantity: Decimal, ef_per_unit: Decimal
    ) -> Decimal:
        """Calculate unit-based emissions: quantity x ef_kgco2e_per_unit.

        Used for IT equipment, vehicles, and other discrete units.

        Args:
            quantity: Number of units.
            ef_per_unit: Emission factor in kgCO2e per unit.

        Returns:
            Emissions in kgCO2e.

        Example:
            >>> engine.calculate_unit_emissions(Decimal("10"), Decimal("500"))
            Decimal('5000.00000000')
        """
        if quantity < ZERO:
            raise ValueError(f"quantity must be >= 0, got {quantity}")
        if ef_per_unit < ZERO:
            raise ValueError(f"ef_per_unit must be >= 0, got {ef_per_unit}")
        return _quantize(quantity * ef_per_unit)

    # ========================================================================
    # Public Method 8: calculate_transport_emissions
    # ========================================================================

    def calculate_transport_emissions(
        self,
        weight_kg: Decimal,
        distance_km: Decimal,
        mode: str,
    ) -> Decimal:
        """Calculate transport emissions for delivering capital goods.

        Formula: (weight_kg / 1000) x distance_km x transport_ef_per_tonne_km

        Args:
            weight_kg: Weight of goods in kilograms.
            distance_km: Transport distance in kilometres.
            mode: Transport mode key from TRANSPORT_EFS.

        Returns:
            Transport emissions in kgCO2e.

        Raises:
            ValueError: If transport mode is not recognized.

        Example:
            >>> engine.calculate_transport_emissions(
            ...     Decimal("5000"), Decimal("500"), "road_truck"
            ... )
            Decimal('155.00000000')
        """
        mode_lower = mode.lower().strip()
        ef = TRANSPORT_EFS.get(mode_lower)
        if ef is None:
            raise ValueError(
                f"Unknown transport mode '{mode}'. "
                f"Supported modes: {sorted(TRANSPORT_EFS.keys())}"
            )
        if weight_kg < ZERO:
            raise ValueError(f"weight_kg must be >= 0, got {weight_kg}")
        if distance_km < ZERO:
            raise ValueError(f"distance_km must be >= 0, got {distance_km}")

        weight_tonnes = _quantize(weight_kg / ONE_THOUSAND)
        return _quantize(weight_tonnes * distance_km * ef)

    # ========================================================================
    # Public Method 9: split_gas_breakdown
    # ========================================================================

    def split_gas_breakdown(
        self, total_co2e: Decimal, material: str
    ) -> Dict[str, Decimal]:
        """Split total CO2e into individual gas components (CO2, CH4, N2O).

        Uses material-specific gas breakdown ratios from _GAS_BREAKDOWN.
        Falls back to _DEFAULT_GAS_BREAKDOWN for unknown materials.

        Args:
            total_co2e: Total emissions in kgCO2e.
            material: Material type key.

        Returns:
            Dictionary with keys "CO2", "CH4", "N2O" in kgCO2e.

        Example:
            >>> breakdown = engine.split_gas_breakdown(Decimal("1000"), "structural_steel")
            >>> assert breakdown["CO2"] == Decimal("950.00000000")
        """
        ratios = _GAS_BREAKDOWN.get(material, _DEFAULT_GAS_BREAKDOWN)
        co2_share, ch4_share, n2o_share = ratios

        co2 = _quantize(total_co2e * co2_share)
        ch4 = _quantize(total_co2e * ch4_share)
        n2o = _quantize(total_co2e * n2o_share)

        # Ensure components sum exactly to total (adjust CO2 for rounding)
        residual = total_co2e - co2 - ch4 - n2o
        co2 = _quantize(co2 + residual)

        return {
            "CO2": co2,
            "CH4": ch4,
            "N2O": n2o,
        }

    # ========================================================================
    # Public Method 10: score_dqi
    # ========================================================================

    def score_dqi(
        self, record: PhysicalRecord, ef_source: str
    ) -> DQIAssessment:
        """Score data quality across 5 GHG Protocol dimensions.

        The average-data method yields base scores that vary by EF source:
        - ICE Database: reliability 2.5, temporal 2.5, technological 3.0
        - ecoinvent: reliability 2.0, temporal 2.0, technological 2.5
        - DEFRA: reliability 3.0, temporal 2.0, technological 3.0

        Completeness is scored based on the presence of physical data fields
        (weight_kg, area_m2, material_type).  Geographical score depends on
        whether the EF is region-specific.

        Args:
            record: Physical record being assessed.
            ef_source: EF source used ("ice_database", "ecoinvent", "defra").

        Returns:
            DQIAssessment with 5 dimension scores and composite.

        Example:
            >>> dqi = engine.score_dqi(record, "ice_database")
            >>> assert Decimal("1.0") <= dqi.composite_score <= Decimal("5.0")
        """
        findings: List[str] = []

        # Reliability score (based on EF source hierarchy)
        reliability_scores = {
            "ecoinvent": Decimal("2.0"),
            "ice_database": Decimal("2.5"),
            "defra": Decimal("3.0"),
            "world_steel": Decimal("2.0"),
            "iai": Decimal("2.0"),
            "custom": Decimal("4.0"),
        }
        reliability = reliability_scores.get(
            ef_source.lower(), Decimal("3.5")
        )

        # Temporal score (average-data is typically 1-3 years old)
        temporal_scores = {
            "ecoinvent": Decimal("2.0"),
            "ice_database": Decimal("2.5"),
            "defra": Decimal("2.0"),
            "world_steel": Decimal("2.0"),
            "iai": Decimal("2.0"),
            "custom": Decimal("3.5"),
        }
        temporal = temporal_scores.get(ef_source.lower(), Decimal("3.0"))

        # Geographical score (mostly global averages for average-data)
        geographical_scores = {
            "ecoinvent": Decimal("2.5"),
            "ice_database": Decimal("3.0"),
            "defra": Decimal("2.5"),
            "world_steel": Decimal("2.0"),
            "iai": Decimal("2.0"),
            "custom": Decimal("3.5"),
        }
        geographical = geographical_scores.get(
            ef_source.lower(), Decimal("3.0")
        )

        # Technological score
        technological_scores = {
            "ecoinvent": Decimal("2.5"),
            "ice_database": Decimal("3.0"),
            "defra": Decimal("3.0"),
            "world_steel": Decimal("2.0"),
            "iai": Decimal("2.0"),
            "custom": Decimal("3.5"),
        }
        technological = technological_scores.get(
            ef_source.lower(), Decimal("3.0")
        )

        # Completeness score (based on data field presence)
        completeness = Decimal("3.0")
        completeness_points = Decimal("0")
        if record.weight_kg is not None:
            completeness_points += Decimal("0.5")
            findings.append("Weight data available - improves completeness")
        if record.area_m2 is not None:
            completeness_points += Decimal("0.5")
            findings.append("Area data available - improves completeness")
        if record.material_type is not None:
            completeness_points += Decimal("0.5")
            findings.append("Material type specified - improves completeness")
        if record.asset_category is not None:
            completeness_points += Decimal("0.3")
            findings.append("Asset category specified")

        completeness = max(
            Decimal("1.0"),
            completeness - completeness_points,
        )

        # Compute composite score (arithmetic mean of 5 dimensions)
        composite = _quantize(
            (reliability + temporal + geographical + technological + completeness)
            / Decimal("5")
        )
        composite = min(Decimal("5.0"), max(Decimal("1.0"), composite))

        # Quality tier
        quality_tier = "Very Poor"
        for tier_name, (tier_min, tier_max) in DQI_QUALITY_TIERS.items():
            if tier_min <= composite < tier_max:
                quality_tier = tier_name
                break

        # EF hierarchy level for average-data
        hierarchy_map = {
            "ecoinvent": 4,
            "ice_database": 5,
            "defra": 5,
            "world_steel": 5,
            "iai": 5,
            "custom": 6,
        }
        ef_hierarchy_level = hierarchy_map.get(ef_source.lower(), 6)

        # Uncertainty factor from pedigree matrix
        uncertainty_factor = Decimal("1.10")
        if composite <= Decimal("1.6"):
            uncertainty_factor = Decimal("1.00")
        elif composite <= Decimal("2.6"):
            uncertainty_factor = Decimal("1.05")
        elif composite <= Decimal("3.6"):
            uncertainty_factor = Decimal("1.10")
        elif composite <= Decimal("4.6"):
            uncertainty_factor = Decimal("1.20")
        else:
            uncertainty_factor = Decimal("1.50")

        findings.append(f"EF source: {ef_source}")
        findings.append(f"Quality tier: {quality_tier}")
        findings.append(f"Method: average_data (medium accuracy +/- 30-60%)")

        return DQIAssessment(
            asset_id=record.asset_id,
            calculation_method=CalculationMethod.AVERAGE_DATA,
            temporal_score=temporal,
            geographical_score=geographical,
            technological_score=technological,
            completeness_score=completeness,
            reliability_score=reliability,
            composite_score=composite,
            quality_tier=quality_tier,
            uncertainty_factor=uncertainty_factor,
            findings=findings,
            ef_hierarchy_level=ef_hierarchy_level,
        )

    # ========================================================================
    # Public Method 11: aggregate_by_material
    # ========================================================================

    def aggregate_by_material(
        self, results: List[AverageDataResult]
    ) -> Dict[str, Decimal]:
        """Aggregate emissions by material type (from EF source tag).

        Groups results by the material type inferred from the EF and sums
        emissions_kg_co2e within each group.

        Args:
            results: List of AverageDataResult instances.

        Returns:
            Dictionary mapping material type to total kgCO2e.

        Example:
            >>> agg = engine.aggregate_by_material(results)
            >>> assert "structural_steel" in agg
        """
        aggregation: Dict[str, Decimal] = {}
        for r in results:
            # Use ef_source + unit to infer material grouping
            key = r.ef_source.value if r.ef_source else "unknown"
            material_key = f"{key}_{r.unit}"
            current = aggregation.get(material_key, ZERO)
            aggregation[material_key] = _quantize(current + r.emissions_kg_co2e)
        return aggregation

    # ========================================================================
    # Public Method 12: aggregate_by_category
    # ========================================================================

    def aggregate_by_category(
        self, results: List[AverageDataResult]
    ) -> Dict[str, Decimal]:
        """Aggregate emissions by asset category.

        Groups results by the unit field (approximation of asset category)
        and sums emissions_kg_co2e.

        Args:
            results: List of AverageDataResult instances.

        Returns:
            Dictionary mapping category to total kgCO2e.

        Example:
            >>> agg = engine.aggregate_by_category(results)
        """
        aggregation: Dict[str, Decimal] = {}
        for r in results:
            key = r.unit if r.unit else "unknown"
            current = aggregation.get(key, ZERO)
            aggregation[key] = _quantize(current + r.emissions_kg_co2e)
        return aggregation

    # ========================================================================
    # Public Method 13: aggregate_by_source
    # ========================================================================

    def aggregate_by_source(
        self, results: List[AverageDataResult]
    ) -> Dict[str, Decimal]:
        """Aggregate emissions by emission factor source.

        Groups results by ef_source and sums emissions_kg_co2e.

        Args:
            results: List of AverageDataResult instances.

        Returns:
            Dictionary mapping EF source name to total kgCO2e.

        Example:
            >>> agg = engine.aggregate_by_source(results)
            >>> assert "ice_database" in agg or "ecoinvent" in agg
        """
        aggregation: Dict[str, Decimal] = {}
        for r in results:
            key = r.ef_source.value if r.ef_source else "unknown"
            current = aggregation.get(key, ZERO)
            aggregation[key] = _quantize(current + r.emissions_kg_co2e)
        return aggregation

    # ========================================================================
    # Public Method 14: get_top_materials
    # ========================================================================

    def get_top_materials(
        self,
        results: List[AverageDataResult],
        n: int = 10,
    ) -> List[Tuple[str, Decimal]]:
        """Get the top N materials by emissions contribution.

        Args:
            results: List of AverageDataResult instances.
            n: Number of top materials to return (default 10).

        Returns:
            List of (material_key, total_kgCO2e) tuples sorted descending.

        Example:
            >>> top = engine.get_top_materials(results, n=5)
            >>> assert len(top) <= 5
        """
        aggregation: Dict[str, Decimal] = {}
        for r in results:
            key = f"{r.ef_source.value}_{r.unit}" if r.ef_source else r.unit
            current = aggregation.get(key, ZERO)
            aggregation[key] = _quantize(current + r.emissions_kg_co2e)

        sorted_materials = sorted(
            aggregation.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_materials[:n]

    # ========================================================================
    # Public Method 15: get_coverage_report
    # ========================================================================

    def get_coverage_report(
        self, results: List[AverageDataResult]
    ) -> CoverageReport:
        """Generate a coverage report for average-data calculations.

        Args:
            results: List of AverageDataResult instances.

        Returns:
            CoverageReport summarizing method coverage.

        Example:
            >>> report = engine.get_coverage_report(results)
            >>> assert report.coverage_pct >= Decimal("0")
        """
        total_assets = len(results)
        covered_assets = sum(
            1 for r in results if r.emissions_kg_co2e > ZERO
        )
        coverage_pct = ZERO
        if total_assets > 0:
            coverage_pct = _quantize(
                Decimal(str(covered_assets))
                / Decimal(str(total_assets))
                * Decimal("100")
            )

        total_emissions = _quantize(
            sum((r.emissions_kg_co2e for r in results), ZERO)
        )

        by_method: Dict[str, Dict[str, Decimal]] = {
            "average_data": {
                "count": Decimal(str(covered_assets)),
                "capex_usd": ZERO,
                "emissions_kgco2e": total_emissions,
            }
        }

        # Identify gap categories (units with zero emissions)
        unit_emissions: Dict[str, Decimal] = {}
        for r in results:
            key = r.unit
            current = unit_emissions.get(key, ZERO)
            unit_emissions[key] = current + r.emissions_kg_co2e
        gap_categories = [
            k for k, v in unit_emissions.items() if v == ZERO
        ]

        return CoverageReport(
            total_assets=total_assets,
            covered_assets=covered_assets,
            coverage_pct=coverage_pct,
            by_method=by_method,
            uncovered_capex_usd=ZERO,
            gap_categories=gap_categories,
        )

    # ========================================================================
    # Public Method 16: validate_record
    # ========================================================================

    def validate_record(self, record: PhysicalRecord) -> List[str]:
        """Validate a PhysicalRecord for average-data calculation.

        Checks that the record has sufficient data for emission calculation:
        material type, quantity, unit, and at least one physical dimension
        (weight or area).

        Args:
            record: PhysicalRecord to validate.

        Returns:
            List of validation error messages (empty if valid).

        Example:
            >>> errors = engine.validate_record(record)
            >>> assert errors == []
        """
        errors: List[str] = []

        if not record.asset_id:
            errors.append("asset_id is required")

        if record.quantity is None or record.quantity <= ZERO:
            errors.append("quantity must be > 0")

        if not record.unit:
            errors.append("unit is required")

        # Check that at least material_type or asset_category is provided
        if (
            record.material_type is None
            and record.asset_category is None
        ):
            errors.append(
                "Either material_type or asset_category must be provided "
                "for EF resolution"
            )

        # Validate unit is recognized
        if record.unit:
            unit_lower = record.unit.lower().strip()
            is_mass_unit = unit_lower in UNIT_CONVERSION_TO_KG
            is_area_unit = unit_lower in ("m2", "ft2", "sq_m", "sq_ft")
            is_count_unit = unit_lower in (
                "unit", "piece", "each", "pcs", "item", "set",
            )
            is_power_unit = unit_lower in (
                "kw", "kwh", "mw", "mwh", "hp",
            )
            if not (is_mass_unit or is_area_unit or is_count_unit or is_power_unit):
                errors.append(
                    f"Unrecognized unit '{record.unit}'. "
                    f"Use a mass, area, count, or power unit."
                )

        # Validate material_type if provided
        if record.material_type is not None:
            material_lower = record.material_type.lower().strip()
            if (
                material_lower not in CAPITAL_PHYSICAL_EMISSION_FACTORS
                and material_lower not in ICE_DATABASE_FACTORS
                and material_lower not in ECOINVENT_FACTORS
                and material_lower not in DEFRA_FACTORS
            ):
                errors.append(
                    f"Unknown material_type '{record.material_type}'. "
                    f"Available materials: {sorted(CAPITAL_PHYSICAL_EMISSION_FACTORS.keys())}"
                )

        # Validate weight_kg if provided
        if record.weight_kg is not None and record.weight_kg < ZERO:
            errors.append("weight_kg must be >= 0")

        # Validate area_m2 if provided
        if record.area_m2 is not None and record.area_m2 < ZERO:
            errors.append("area_m2 must be >= 0")

        return errors

    # ========================================================================
    # Public Method 17: get_supported_units
    # ========================================================================

    def get_supported_units(self) -> List[str]:
        """Return list of all supported unit strings.

        Returns:
            Sorted list of unit names recognized by convert_to_kg.

        Example:
            >>> units = engine.get_supported_units()
            >>> assert "kg" in units
        """
        return sorted(UNIT_CONVERSION_TO_KG.keys())

    # ========================================================================
    # Public Method 18: get_supported_materials
    # ========================================================================

    def get_supported_materials(self) -> List[str]:
        """Return list of all supported material type keys.

        Includes materials from all three EF sources (ICE, ecoinvent, DEFRA)
        plus the shared CAPITAL_PHYSICAL_EMISSION_FACTORS table.

        Returns:
            Sorted deduplicated list of material type keys.

        Example:
            >>> materials = engine.get_supported_materials()
            >>> assert "structural_steel" in materials
        """
        all_materials = set(CAPITAL_PHYSICAL_EMISSION_FACTORS.keys())
        all_materials.update(ICE_DATABASE_FACTORS.keys())
        all_materials.update(ECOINVENT_FACTORS.keys())
        all_materials.update(DEFRA_FACTORS.keys())
        return sorted(all_materials)

    # ========================================================================
    # Public Method 19: get_transport_modes
    # ========================================================================

    def get_transport_modes(self) -> List[str]:
        """Return list of all supported transport mode keys.

        Returns:
            Sorted list of transport mode strings.

        Example:
            >>> modes = engine.get_transport_modes()
            >>> assert "road_truck" in modes
        """
        return sorted(TRANSPORT_EFS.keys())

    # ========================================================================
    # Public Method 20: get_ef_sources
    # ========================================================================

    def get_ef_sources(self) -> List[str]:
        """Return list of all supported emission factor source names.

        Returns:
            List of EF source identifiers.

        Example:
            >>> sources = engine.get_ef_sources()
            >>> assert "ice_database" in sources
        """
        return sorted(EF_SOURCE_PRIORITY.keys())

    # ========================================================================
    # Public Method 21: estimate_uncertainty
    # ========================================================================

    def estimate_uncertainty(
        self, result: AverageDataResult
    ) -> Dict[str, Any]:
        """Estimate uncertainty for an average-data calculation result.

        Uses the GHG Protocol Scope 3 uncertainty ranges for the average-data
        method (+/- 30-60 %) combined with EF source-specific uncertainty.

        Args:
            result: AverageDataResult to assess.

        Returns:
            Dictionary with uncertainty metrics:
                - method_uncertainty_min_pct: Minimum uncertainty from method.
                - method_uncertainty_max_pct: Maximum uncertainty from method.
                - ef_source_uncertainty_min_pct: From EF source.
                - ef_source_uncertainty_max_pct: From EF source.
                - combined_uncertainty_pct: Root-sum-of-squares combination.
                - lower_bound_kgco2e: Emissions - uncertainty.
                - upper_bound_kgco2e: Emissions + uncertainty.
                - confidence_level: Default 95%.

        Example:
            >>> unc = engine.estimate_uncertainty(result)
            >>> assert unc["combined_uncertainty_pct"] > Decimal("0")
        """
        # Method uncertainty
        method_range = UNCERTAINTY_RANGES.get(
            CalculationMethod.AVERAGE_DATA,
            (Decimal("30"), Decimal("60")),
        )
        method_min = method_range[0]
        method_max = method_range[1]
        method_mid = _quantize((method_min + method_max) / Decimal("2"))

        # EF source uncertainty
        ef_source_key = result.ef_source.value if result.ef_source else "custom"
        ef_range = EF_SOURCE_UNCERTAINTY.get(
            ef_source_key, (Decimal("30"), Decimal("50"))
        )
        ef_min = ef_range[0]
        ef_max = ef_range[1]
        ef_mid = _quantize((ef_min + ef_max) / Decimal("2"))

        # Combined uncertainty (root-sum-of-squares)
        combined_squared = method_mid * method_mid + ef_mid * ef_mid
        combined = self._decimal_sqrt(combined_squared)
        combined = _quantize(combined)

        # Bounds
        fraction = _quantize(combined / Decimal("100"))
        lower = _quantize(
            result.emissions_kg_co2e * (ONE - fraction)
        )
        lower = max(ZERO, lower)
        upper = _quantize(
            result.emissions_kg_co2e * (ONE + fraction)
        )

        return {
            "method_uncertainty_min_pct": method_min,
            "method_uncertainty_max_pct": method_max,
            "method_uncertainty_mid_pct": method_mid,
            "ef_source": ef_source_key,
            "ef_source_uncertainty_min_pct": ef_min,
            "ef_source_uncertainty_max_pct": ef_max,
            "ef_source_uncertainty_mid_pct": ef_mid,
            "combined_uncertainty_pct": combined,
            "lower_bound_kgco2e": lower,
            "upper_bound_kgco2e": upper,
            "emissions_kgco2e": result.emissions_kg_co2e,
            "confidence_level": Decimal("95.0"),
        }

    # ========================================================================
    # Public Method 22: get_calculation_stats
    # ========================================================================

    def get_calculation_stats(self) -> Dict[str, Any]:
        """Return engine calculation statistics.

        Returns:
            Dictionary with running counts, totals, and cache info.

        Example:
            >>> stats = engine.get_calculation_stats()
            >>> assert stats["calc_count"] >= 0
        """
        with self._lock:
            return {
                "calc_count": self._calc_count,
                "batch_count": self._batch_count,
                "error_count": self._error_count,
                "total_emissions_kgco2e": str(self._total_emissions),
                "last_calc_time_ms": str(self._last_calc_time_ms),
                "ef_cache_size": len(self._ef_cache),
                "ice_factors_count": len(ICE_DATABASE_FACTORS),
                "ecoinvent_factors_count": len(ECOINVENT_FACTORS),
                "defra_factors_count": len(DEFRA_FACTORS),
                "supported_units_count": len(UNIT_CONVERSION_TO_KG),
                "supported_materials_count": len(
                    self.get_supported_materials()
                ),
                "transport_modes_count": len(TRANSPORT_EFS),
            }

    # ========================================================================
    # Public Method 23: compute_provenance_hash
    # ========================================================================

    def compute_provenance_hash(
        self,
        record: PhysicalRecord,
        result: AverageDataResult,
    ) -> str:
        """Compute SHA-256 provenance hash for audit trail.

        Combines the input record and output result into a deterministic
        hash that proves the calculation was performed on the stated inputs
        to produce the stated outputs.

        Args:
            record: Input PhysicalRecord.
            result: Output AverageDataResult.

        Returns:
            SHA-256 hex digest string (64 characters).

        Example:
            >>> h = engine.compute_provenance_hash(record, result)
            >>> assert len(h) == 64
        """
        input_str = _to_json_safe(record.model_dump())
        output_str = _to_json_safe({
            "emissions_kg_co2e": str(result.emissions_kg_co2e),
            "co2": str(result.co2),
            "ch4": str(result.ch4),
            "n2o": str(result.n2o),
            "ef_value": str(result.ef_value),
            "ef_source": result.ef_source.value if result.ef_source else "",
            "transport_emissions": str(result.transport_emissions),
            "method": result.method.value if result.method else "",
        })
        combined = f"AVGDATA|{input_str}|{output_str}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    # ========================================================================
    # Public Method 24: _decimal_sqrt (Newton's method)
    # ========================================================================

    def _decimal_sqrt(self, value: Decimal) -> Decimal:
        """Compute square root of a Decimal using Newton's method.

        Formula: x_{n+1} = (x_n + S / x_n) / 2

        Converges to sqrt(S) with quadratic convergence rate.

        Args:
            value: Non-negative Decimal value.

        Returns:
            Square root of the value.

        Raises:
            ValueError: If value is negative.

        Example:
            >>> engine._decimal_sqrt(Decimal("4.0"))
            Decimal('2.00000000')
        """
        if value < ZERO:
            raise ValueError(f"Cannot compute square root of negative: {value}")
        if value == ZERO:
            return ZERO

        # Initial guess: half the value or 1 if value < 1
        x = value / Decimal("2")
        if x == ZERO:
            x = Decimal("1")

        two = Decimal("2")
        for _ in range(_SQRT_MAX_ITERATIONS):
            x_next = (x + value / x) / two
            if abs(x_next - x) < _SQRT_TOLERANCE:
                return _quantize(x_next)
            x = x_next

        return _quantize(x)

    # ========================================================================
    # Public Method 25: select_best_ef
    # ========================================================================

    def select_best_ef(
        self,
        material_type: str,
        available_sources: Optional[List[str]] = None,
    ) -> Tuple[Decimal, str]:
        """Select the best available emission factor for a material.

        Searches EF sources in priority order (ICE > ecoinvent > DEFRA)
        and returns the first match.  If available_sources is provided,
        only those sources are searched.

        Args:
            material_type: Material key (e.g., "structural_steel").
            available_sources: Optional list of source names to search.

        Returns:
            Tuple of (ef_value, source_name).

        Raises:
            ValueError: If no factor found in any source.

        Example:
            >>> ef, source = engine.select_best_ef("structural_steel")
            >>> assert ef > Decimal("0")
        """
        return self._select_ef(material_type, None, available_sources)

    # ========================================================================
    # Public Method 26: get_building_ef
    # ========================================================================

    def get_building_ef(
        self, building_type: str, area_m2: Decimal
    ) -> Decimal:
        """Get total embodied carbon for a building type.

        Multiplies area by the per-m2 emission factor for the building type.

        Args:
            building_type: Building type key (e.g., "office_building").
            area_m2: Gross floor area in square metres.

        Returns:
            Total embodied carbon in kgCO2e.

        Raises:
            ValueError: If building type is unknown.

        Example:
            >>> ef = engine.get_building_ef("office_building", Decimal("2000"))
            >>> assert ef == Decimal("1000000.00000000")
        """
        bt_lower = building_type.lower().strip()
        ef_per_m2 = BUILDING_EF_PER_M2.get(bt_lower)
        if ef_per_m2 is None:
            raise ValueError(
                f"Unknown building type '{building_type}'. "
                f"Supported: {sorted(BUILDING_EF_PER_M2.keys())}"
            )
        if area_m2 < ZERO:
            raise ValueError(f"area_m2 must be >= 0, got {area_m2}")
        return _quantize(area_m2 * ef_per_m2)

    # ========================================================================
    # Public Method 27: get_equipment_ef
    # ========================================================================

    def get_equipment_ef(
        self, equipment_type: str, weight_kg: Decimal
    ) -> Decimal:
        """Get embodied carbon for equipment by type and weight.

        Multiplies weight by the per-kg emission factor for equipment type.

        Args:
            equipment_type: Equipment type key (e.g., "hvac", "generator").
            weight_kg: Equipment weight in kilograms.

        Returns:
            Total embodied carbon in kgCO2e.

        Raises:
            ValueError: If equipment type is unknown.

        Example:
            >>> ef = engine.get_equipment_ef("generator", Decimal("500"))
            >>> assert ef == Decimal("1600.00000000")
        """
        et_lower = equipment_type.lower().strip()
        ef_per_kg = EQUIPMENT_EF_PER_KG.get(et_lower)
        if ef_per_kg is None:
            raise ValueError(
                f"Unknown equipment type '{equipment_type}'. "
                f"Supported: {sorted(EQUIPMENT_EF_PER_KG.keys())}"
            )
        if weight_kg < ZERO:
            raise ValueError(f"weight_kg must be >= 0, got {weight_kg}")
        return _quantize(weight_kg * ef_per_kg)

    # ========================================================================
    # Public Method 28: get_vehicle_ef
    # ========================================================================

    def get_vehicle_ef(self, vehicle_type: str) -> Decimal:
        """Get cradle-to-gate emission factor for a vehicle type.

        Args:
            vehicle_type: Vehicle type key (e.g., "passenger_car").

        Returns:
            Emission factor in kgCO2e per vehicle.

        Raises:
            ValueError: If vehicle type is unknown.

        Example:
            >>> ef = engine.get_vehicle_ef("passenger_car")
            >>> assert ef == Decimal("6000.0")
        """
        vt_lower = vehicle_type.lower().strip()
        ef = VEHICLE_EF_PER_UNIT.get(vt_lower)
        if ef is None:
            raise ValueError(
                f"Unknown vehicle type '{vehicle_type}'. "
                f"Supported: {sorted(VEHICLE_EF_PER_UNIT.keys())}"
            )
        return ef

    # ========================================================================
    # Public Method 29: get_it_equipment_ef
    # ========================================================================

    def get_it_equipment_ef(
        self, it_type: str, quantity: int = 1
    ) -> Decimal:
        """Get total embodied carbon for IT equipment.

        Multiplies the per-unit EF by the quantity.

        Args:
            it_type: IT equipment type key (e.g., "server", "laptop").
            quantity: Number of units.

        Returns:
            Total embodied carbon in kgCO2e.

        Raises:
            ValueError: If IT equipment type is unknown.

        Example:
            >>> ef = engine.get_it_equipment_ef("server", 10)
            >>> assert ef == Decimal("5000.00000000")
        """
        it_lower = it_type.lower().strip()
        ef_per_unit = IT_EQUIPMENT_EF_PER_UNIT.get(it_lower)
        if ef_per_unit is None:
            raise ValueError(
                f"Unknown IT equipment type '{it_type}'. "
                f"Supported: {sorted(IT_EQUIPMENT_EF_PER_UNIT.keys())}"
            )
        if quantity < 0:
            raise ValueError(f"quantity must be >= 0, got {quantity}")
        return _quantize(ef_per_unit * Decimal(str(quantity)))

    # ========================================================================
    # Public Method 30: calculate_embodied_carbon
    # ========================================================================

    def calculate_embodied_carbon(
        self,
        material_type: str,
        quantity: Decimal,
        unit: str,
    ) -> Decimal:
        """Calculate embodied carbon for a material type and quantity.

        Resolves the best EF for the material, converts the quantity to
        the appropriate basis (kg, m2, or unit), and returns total kgCO2e.

        Args:
            material_type: Material key (e.g., "structural_steel").
            quantity: Physical quantity.
            unit: Unit of the quantity.

        Returns:
            Total embodied carbon in kgCO2e.

        Raises:
            ValueError: If material or unit is unknown.

        Example:
            >>> ec = engine.calculate_embodied_carbon(
            ...     "structural_steel", Decimal("5000"), "kg"
            ... )
            >>> assert ec > Decimal("0")
        """
        ef_value, ef_source = self._select_ef(material_type, None)
        ef_unit = self._determine_ef_unit(material_type)

        if ef_unit == "kg":
            weight_kg = self.convert_to_kg(quantity, unit)
            return self.calculate_material_emissions(weight_kg, ef_value)
        elif ef_unit == "m2":
            return self.calculate_area_emissions(quantity, ef_value)
        else:
            return self.calculate_unit_emissions(quantity, ef_value)

    # ========================================================================
    # Public Method 31: apply_regional_adjustment
    # ========================================================================

    def apply_regional_adjustment(
        self, ef: Decimal, region: str
    ) -> Decimal:
        """Apply regional carbon intensity adjustment to an emission factor.

        Adjusts EF based on the regional electricity grid carbon intensity
        relative to the global average.

        Args:
            ef: Base emission factor.
            region: Region code (e.g., "US", "EU", "CN", "GLOBAL").

        Returns:
            Adjusted emission factor.

        Example:
            >>> adjusted = engine.apply_regional_adjustment(Decimal("1.55"), "EU")
            >>> assert adjusted < Decimal("1.55")  # EU has lower grid intensity
        """
        region_upper = region.upper().strip()
        adjustment = REGIONAL_ADJUSTMENT_FACTORS.get(
            region_upper, Decimal("1.00")
        )
        return _quantize(ef * adjustment)

    # ========================================================================
    # Public Method 32: get_ice_database_factor
    # ========================================================================

    def get_ice_database_factor(self, material: str) -> Optional[Decimal]:
        """Look up a material's emission factor in ICE Database v3.0.

        Args:
            material: Material key (e.g., "structural_steel").

        Returns:
            Emission factor in kgCO2e per kg/unit, or None if not found.

        Example:
            >>> ef = engine.get_ice_database_factor("structural_steel")
            >>> assert ef == Decimal("1.55")
        """
        return ICE_DATABASE_FACTORS.get(material.lower().strip())

    # ========================================================================
    # Public Method 33: get_ecoinvent_factor
    # ========================================================================

    def get_ecoinvent_factor(self, material: str) -> Optional[Decimal]:
        """Look up a material's emission factor in ecoinvent v3.11.

        Args:
            material: Material key (e.g., "structural_steel").

        Returns:
            Emission factor in kgCO2e per kg/unit, or None if not found.

        Example:
            >>> ef = engine.get_ecoinvent_factor("structural_steel")
            >>> assert ef == Decimal("1.89")
        """
        return ECOINVENT_FACTORS.get(material.lower().strip())

    # ========================================================================
    # Public Method 34: get_defra_factor
    # ========================================================================

    def get_defra_factor(self, material: str) -> Optional[Decimal]:
        """Look up a material's emission factor in DEFRA 2023.

        Args:
            material: Material key (e.g., "structural_steel").

        Returns:
            Emission factor in kgCO2e per kg/unit, or None if not found.

        Example:
            >>> ef = engine.get_defra_factor("structural_steel")
            >>> assert ef == Decimal("1.46")
        """
        return DEFRA_FACTORS.get(material.lower().strip())

    # ========================================================================
    # Public Method 35: compare_ef_sources
    # ========================================================================

    def compare_ef_sources(self, material: str) -> Dict[str, Optional[Decimal]]:
        """Compare emission factors across all sources for a material.

        Args:
            material: Material key (e.g., "structural_steel").

        Returns:
            Dictionary mapping source name to EF value (None if unavailable).

        Example:
            >>> cmp = engine.compare_ef_sources("structural_steel")
            >>> assert cmp["ice_database"] == Decimal("1.55")
        """
        material_lower = material.lower().strip()
        return {
            "ice_database": ICE_DATABASE_FACTORS.get(material_lower),
            "ecoinvent": ECOINVENT_FACTORS.get(material_lower),
            "defra": DEFRA_FACTORS.get(material_lower),
            "capital_physical": CAPITAL_PHYSICAL_EMISSION_FACTORS.get(
                material_lower
            ),
        }

    # ========================================================================
    # Public Method 36: reset (classmethod)
    # ========================================================================

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance for testing or configuration reload.

        Thread-safe reset of the engine singleton.  After reset, the next
        instantiation will create a fresh engine.

        Example:
            >>> AverageDataCalculatorEngine.reset()
            >>> engine = AverageDataCalculatorEngine()
        """
        with cls._lock:
            cls._instance = None
            logger.info("AverageDataCalculatorEngine singleton reset")

    # ========================================================================
    # Private Methods
    # ========================================================================

    def _resolve_material_type(self, record: PhysicalRecord) -> str:
        """Resolve the material type key from the record.

        Resolution order:
        1. record.material_type (if present and valid)
        2. CATEGORY_DEFAULT_MATERIAL[subcategory] (if subcategory available)
        3. ASSET_CATEGORY_DEFAULT_MATERIAL[category] (if category available)
        4. "structural_steel" (ultimate fallback)

        Args:
            record: PhysicalRecord to resolve material from.

        Returns:
            Material type key string.
        """
        # Priority 1: Explicit material type
        if record.material_type:
            mt = record.material_type.lower().strip()
            if (
                mt in CAPITAL_PHYSICAL_EMISSION_FACTORS
                or mt in ICE_DATABASE_FACTORS
                or mt in ECOINVENT_FACTORS
                or mt in DEFRA_FACTORS
            ):
                return mt
            logger.warning(
                "Unknown material_type '%s' for asset %s, attempting fallback",
                record.material_type,
                record.asset_id,
            )

        # Priority 2: Subcategory default
        if record.asset_category is not None:
            # Try subcategory names from AssetSubCategory
            cat_value = record.asset_category.value
            # Check if there is a subcategory default (using uppercase enum name)
            for subcat_name, default_mat in CATEGORY_DEFAULT_MATERIAL.items():
                if subcat_name.lower() == cat_value.lower():
                    logger.info(
                        "Resolved material_type='%s' from category '%s' "
                        "for asset %s",
                        default_mat,
                        cat_value,
                        record.asset_id,
                    )
                    return default_mat

        # Priority 3: Top-level category default
        if record.asset_category is not None:
            cat_value = record.asset_category.value
            default_mat = ASSET_CATEGORY_DEFAULT_MATERIAL.get(cat_value)
            if default_mat:
                logger.info(
                    "Resolved material_type='%s' from asset_category '%s' "
                    "for asset %s",
                    default_mat,
                    cat_value,
                    record.asset_id,
                )
                return default_mat

        # Priority 4: Ultimate fallback
        logger.warning(
            "Using fallback material_type='structural_steel' for asset %s",
            record.asset_id,
        )
        return "structural_steel"

    def _select_ef(
        self,
        material_type: str,
        preferred_source: Optional[str] = None,
        available_sources: Optional[List[str]] = None,
    ) -> Tuple[Decimal, str]:
        """Select the best emission factor for a material.

        If preferred_source is specified and the material exists in that
        source, it is returned directly.  Otherwise, sources are searched
        in priority order.

        Args:
            material_type: Material key.
            preferred_source: Optional preferred EF source name.
            available_sources: Optional list of sources to search.

        Returns:
            Tuple of (ef_value, source_name).

        Raises:
            ValueError: If no factor found in any source.
        """
        mt_lower = material_type.lower().strip()

        # If preferred source specified, try it first
        if preferred_source:
            ps = preferred_source.lower().strip()
            factor = self._get_factor_from_source(mt_lower, ps)
            if factor is not None:
                return (factor, ps)
            logger.warning(
                "Material '%s' not in preferred source '%s', trying others",
                material_type,
                preferred_source,
            )

        # Search in priority order
        sources_to_search: List[Tuple[str, Dict[str, Decimal]]] = [
            ("ice_database", ICE_DATABASE_FACTORS),
            ("ecoinvent", ECOINVENT_FACTORS),
            ("defra", DEFRA_FACTORS),
        ]

        if available_sources:
            allowed = {s.lower().strip() for s in available_sources}
            sources_to_search = [
                (name, db)
                for name, db in sources_to_search
                if name in allowed
            ]

        for source_name, source_db in sources_to_search:
            factor = source_db.get(mt_lower)
            if factor is not None:
                return (factor, source_name)

        # Final fallback: CAPITAL_PHYSICAL_EMISSION_FACTORS
        factor = CAPITAL_PHYSICAL_EMISSION_FACTORS.get(mt_lower)
        if factor is not None:
            return (factor, "capital_physical")

        raise ValueError(
            f"No emission factor found for material '{material_type}' "
            f"in any source. Searched: ICE, ecoinvent, DEFRA, "
            f"CAPITAL_PHYSICAL_EMISSION_FACTORS."
        )

    def _get_factor_from_source(
        self, material_type: str, source: str
    ) -> Optional[Decimal]:
        """Get an emission factor from a specific named source.

        Args:
            material_type: Material key (lowercase).
            source: Source name (lowercase).

        Returns:
            Emission factor Decimal or None if not found.
        """
        source_map: Dict[str, Dict[str, Decimal]] = {
            "ice_database": ICE_DATABASE_FACTORS,
            "ice": ICE_DATABASE_FACTORS,
            "ecoinvent": ECOINVENT_FACTORS,
            "defra": DEFRA_FACTORS,
            "capital_physical": CAPITAL_PHYSICAL_EMISSION_FACTORS,
            "custom": {},
        }
        db = source_map.get(source.lower().strip(), {})
        return db.get(material_type.lower().strip())

    def _determine_ef_unit(self, material_type: str) -> str:
        """Determine the denominator unit for a material's emission factor.

        Per-unit materials (IT, vehicles, renewables) have "unit" denominator.
        Per-kW/kWh/MW materials have "kw", "kwh", or "mw" denominator.
        Area-based materials (buildings) use "m2" if detected.
        All others default to "kg".

        Args:
            material_type: Material key.

        Returns:
            Unit string ("kg", "unit", "kw", "kwh", "mw", "m2").
        """
        mt = material_type.lower().strip()
        if mt.endswith("_per_unit"):
            return "unit"
        if mt.endswith("_per_kw"):
            return "kw"
        if mt.endswith("_per_kwh"):
            return "kwh"
        if mt.endswith("_per_mw"):
            return "mw"
        if mt.endswith("_per_m2"):
            return "m2"
        return "kg"

    def _str_to_ef_source(self, source: str) -> PhysicalEFSource:
        """Convert a source string to PhysicalEFSource enum.

        Args:
            source: Source name string.

        Returns:
            PhysicalEFSource enum value.
        """
        source_mapping: Dict[str, PhysicalEFSource] = {
            "ice_database": PhysicalEFSource.ICE_DATABASE,
            "ice": PhysicalEFSource.ICE_DATABASE,
            "ecoinvent": PhysicalEFSource.ECOINVENT,
            "defra": PhysicalEFSource.DEFRA,
            "world_steel": PhysicalEFSource.WORLD_STEEL,
            "iai": PhysicalEFSource.IAI,
            "custom": PhysicalEFSource.CUSTOM,
            "capital_physical": PhysicalEFSource.ICE_DATABASE,
        }
        return source_mapping.get(
            source.lower().strip(),
            PhysicalEFSource.CUSTOM,
        )

    def _resolve_weight_kg(
        self, record: PhysicalRecord, material_type: str
    ) -> Optional[Decimal]:
        """Resolve weight in kg from a PhysicalRecord.

        Resolution order:
        1. record.weight_kg (if explicitly provided)
        2. Convert record.quantity using record.unit via UNIT_CONVERSION_TO_KG
        3. None (for non-mass-based materials)

        Args:
            record: PhysicalRecord to resolve weight from.
            material_type: Resolved material type.

        Returns:
            Weight in kilograms or None if not applicable.
        """
        # Priority 1: Explicit weight
        if record.weight_kg is not None and record.weight_kg > ZERO:
            return record.weight_kg

        # Priority 2: Convert from quantity + unit
        unit_lower = record.unit.lower().strip()
        if unit_lower in UNIT_CONVERSION_TO_KG:
            try:
                return self.convert_to_kg(record.quantity, unit_lower)
            except ValueError:
                pass

        return None

    def _compute_material_emissions(
        self,
        record: PhysicalRecord,
        material_type: str,
        weight_kg: Optional[Decimal],
        area_m2: Optional[Decimal],
        ef_value: Decimal,
    ) -> Decimal:
        """Compute material emissions based on available data.

        Selects the appropriate calculation formula based on the EF unit
        type and available physical data:
        - Per-kg materials: use weight_kg x ef
        - Per-m2 materials: use area_m2 x ef
        - Per-unit materials: use quantity x ef

        Args:
            record: Source PhysicalRecord.
            material_type: Resolved material type key.
            weight_kg: Weight in kg (may be None).
            area_m2: Area in m2 (may be None).
            ef_value: Emission factor value.

        Returns:
            Material emissions in kgCO2e.
        """
        ef_unit = self._determine_ef_unit(material_type)

        if ef_unit == "m2" and area_m2 is not None and area_m2 > ZERO:
            return self.calculate_area_emissions(area_m2, ef_value)

        if ef_unit in ("unit", "kw", "kwh", "mw"):
            return self.calculate_unit_emissions(record.quantity, ef_value)

        if weight_kg is not None and weight_kg > ZERO:
            return self.calculate_material_emissions(weight_kg, ef_value)

        # Fallback: use quantity directly with ef
        return self.calculate_unit_emissions(record.quantity, ef_value)

    def _is_per_unit_material(self, material_type: str) -> bool:
        """Check if a material type uses per-unit emission factors.

        Args:
            material_type: Material key.

        Returns:
            True if the material uses per-unit EFs.
        """
        mt = material_type.lower().strip()
        return (
            mt.endswith("_per_unit")
            or mt.endswith("_per_kw")
            or mt.endswith("_per_kwh")
            or mt.endswith("_per_mw")
            or mt.endswith("_per_m2")
        )


# ============================================================================
# Module-Level Convenience Functions
# ============================================================================


def get_engine() -> AverageDataCalculatorEngine:
    """Get or create the AverageDataCalculatorEngine singleton.

    Returns:
        AverageDataCalculatorEngine singleton instance.

    Example:
        >>> engine = get_engine()
        >>> result = engine.calculate(record)
    """
    return AverageDataCalculatorEngine()


def calculate_average_data(
    record: PhysicalRecord,
    config: Optional[Dict[str, Any]] = None,
) -> AverageDataResult:
    """Convenience function: calculate emissions for a single record.

    Args:
        record: PhysicalRecord input.
        config: Optional configuration overrides.

    Returns:
        AverageDataResult with calculated emissions.

    Example:
        >>> result = calculate_average_data(record)
    """
    return get_engine().calculate(record, config)


def calculate_average_data_batch(
    records: List[PhysicalRecord],
    config: Optional[Dict[str, Any]] = None,
) -> List[AverageDataResult]:
    """Convenience function: calculate emissions for a batch of records.

    Args:
        records: List of PhysicalRecord inputs.
        config: Optional configuration overrides.

    Returns:
        List of AverageDataResult instances.

    Example:
        >>> results = calculate_average_data_batch(records)
    """
    return get_engine().calculate_batch(records, config)


def lookup_ef(
    material_type: str,
    source: str = "ice_database",
) -> PhysicalEF:
    """Convenience function: look up an emission factor.

    Args:
        material_type: Material key.
        source: EF source name.

    Returns:
        PhysicalEF instance.

    Example:
        >>> ef = lookup_ef("structural_steel")
    """
    return get_engine().lookup_physical_ef(material_type, source)


def compare_sources(material: str) -> Dict[str, Optional[Decimal]]:
    """Convenience function: compare EFs across sources.

    Args:
        material: Material key.

    Returns:
        Dictionary mapping source name to EF value.

    Example:
        >>> cmp = compare_sources("structural_steel")
    """
    return get_engine().compare_ef_sources(material)


# ============================================================================
# Extended Data Tables - Material Categories for Reporting
# ============================================================================

#: Mapping of material types to high-level reporting categories.
#: Used by aggregation methods to group materials into analyst-friendly buckets.
MATERIAL_REPORTING_CATEGORY: Dict[str, str] = {
    # Metals
    "structural_steel": "metals",
    "reinforcing_steel": "metals",
    "stainless_steel": "metals",
    "aluminum_sheet": "metals",
    "aluminum_extrusion": "metals",
    "copper_pipe": "metals",
    "copper_wire": "metals",
    # Concrete & Masonry
    "concrete_25mpa": "concrete_masonry",
    "concrete_32mpa": "concrete_masonry",
    "concrete_40mpa": "concrete_masonry",
    "concrete_precast": "concrete_masonry",
    "brick": "concrete_masonry",
    # Glass
    "glass_float": "glass",
    "glass_tempered": "glass",
    "glass_double_glazed": "glass",
    # Timber
    "timber_softwood": "timber",
    "timber_hardwood": "timber",
    "timber_glulam": "timber",
    # Interior Materials
    "plasterboard": "interior",
    "ceramic_tiles": "interior",
    "carpet": "interior",
    "vinyl_flooring": "interior",
    "paint_water_based": "interior",
    "paint_solvent_based": "interior",
    # Insulation
    "insulation_mineral_wool": "insulation",
    "insulation_eps": "insulation",
    "insulation_xps": "insulation",
    "insulation_pir": "insulation",
    # Piping & Roofing
    "pvc_pipe": "piping_roofing",
    "hdpe_pipe": "piping_roofing",
    "roofing_membrane": "piping_roofing",
    "asphalt": "piping_roofing",
    # IT Equipment
    "server_per_unit": "it_equipment",
    "laptop_per_unit": "it_equipment",
    "desktop_per_unit": "it_equipment",
    "monitor_per_unit": "it_equipment",
    "network_switch_per_unit": "it_equipment",
    "ups_per_unit": "it_equipment",
    "led_panel_per_unit": "it_equipment",
    # Renewable Energy
    "solar_panel_per_kw": "renewable_energy",
    "wind_turbine_per_mw": "renewable_energy",
    "battery_li_ion_per_kwh": "renewable_energy",
    # Electrical
    "transformer_per_unit": "electrical",
    "electric_motor_per_kw": "electrical",
}


# ============================================================================
# Extended Data Tables - Material Carbon Intensity Benchmarks
# ============================================================================

#: Industry benchmark carbon intensity tiers for materials.
#: Used for comparative analysis and best-practice identification.
#: Tiers: "low" (< 25th percentile), "medium" (25th-75th),
#:         "high" (> 75th percentile) relative to material class.
MATERIAL_INTENSITY_BENCHMARK: Dict[str, str] = {
    "structural_steel": "medium",
    "reinforcing_steel": "medium",
    "stainless_steel": "high",
    "aluminum_sheet": "high",
    "aluminum_extrusion": "high",
    "copper_pipe": "high",
    "copper_wire": "high",
    "concrete_25mpa": "low",
    "concrete_32mpa": "low",
    "concrete_40mpa": "medium",
    "concrete_precast": "low",
    "brick": "low",
    "glass_float": "medium",
    "glass_tempered": "medium",
    "glass_double_glazed": "high",
    "timber_softwood": "low",
    "timber_hardwood": "low",
    "timber_glulam": "low",
    "plasterboard": "low",
    "ceramic_tiles": "medium",
    "carpet": "high",
    "vinyl_flooring": "high",
    "insulation_mineral_wool": "medium",
    "insulation_eps": "high",
    "insulation_xps": "high",
    "insulation_pir": "high",
    "pvc_pipe": "high",
    "hdpe_pipe": "medium",
    "asphalt": "low",
}


def get_material_category(material: str) -> str:
    """Get the reporting category for a material type.

    Args:
        material: Material type key.

    Returns:
        Reporting category string (e.g., "metals", "concrete_masonry").

    Example:
        >>> get_material_category("structural_steel")
        'metals'
    """
    return MATERIAL_REPORTING_CATEGORY.get(
        material.lower().strip(), "other"
    )


def get_material_benchmark(material: str) -> str:
    """Get the carbon intensity benchmark tier for a material.

    Args:
        material: Material type key.

    Returns:
        Benchmark tier ("low", "medium", "high", or "unknown").

    Example:
        >>> get_material_benchmark("structural_steel")
        'medium'
    """
    return MATERIAL_INTENSITY_BENCHMARK.get(
        material.lower().strip(), "unknown"
    )


def get_all_building_types() -> List[str]:
    """Return all supported building types for area-based calculations.

    Returns:
        Sorted list of building type keys.

    Example:
        >>> types = get_all_building_types()
        >>> assert "office_building" in types
    """
    return sorted(BUILDING_EF_PER_M2.keys())


def get_all_equipment_types() -> List[str]:
    """Return all supported equipment types for weight-based calculations.

    Returns:
        Sorted list of equipment type keys.

    Example:
        >>> types = get_all_equipment_types()
        >>> assert "generator" in types
    """
    return sorted(EQUIPMENT_EF_PER_KG.keys())


def get_all_vehicle_types() -> List[str]:
    """Return all supported vehicle types.

    Returns:
        Sorted list of vehicle type keys.

    Example:
        >>> types = get_all_vehicle_types()
        >>> assert "passenger_car" in types
    """
    return sorted(VEHICLE_EF_PER_UNIT.keys())


def get_all_it_types() -> List[str]:
    """Return all supported IT equipment types.

    Returns:
        Sorted list of IT equipment type keys.

    Example:
        >>> types = get_all_it_types()
        >>> assert "server" in types
    """
    return sorted(IT_EQUIPMENT_EF_PER_UNIT.keys())
