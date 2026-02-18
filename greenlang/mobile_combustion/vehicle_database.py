# -*- coding: utf-8 -*-
"""
VehicleDatabaseEngine - Engine 1: Mobile Combustion Agent (AGENT-MRV-003)

Comprehensive vehicle and emission factor database for GHG Protocol Scope 1
mobile combustion calculations. Manages 18 vehicle types across 5 categories,
15 fuel types with physical properties, vehicle-specific CH4/N2O emission
factors by model year and control technology, and distance-based emission
factors.

All factor values use ``Decimal`` for bit-perfect precision. The engine is
thread-safe via ``threading.Lock()`` and tracks every lookup through SHA-256
provenance hashing.

Data Sources:
    - EPA GHG Emission Factors Hub 2025 (Mobile Combustion Tables)
    - IPCC 2006 Guidelines Vol 2 Ch 3 (Mobile Combustion)
    - UK DEFRA 2025 Conversion Factors (Vehicle-specific)
    - EPA AP-42 Chapter 13 (Off-Road Equipment)

Vehicle Categories (5):
    ON_ROAD: Passenger cars, trucks, buses, motorcycles, vans
    OFF_ROAD: Construction, agriculture, industrial, mining, forklifts
    MARINE: Inland, coastal, ocean vessels
    AVIATION: Corporate jets, helicopters, turboprops
    RAIL: Diesel locomotives

Fuel Types (15):
    GASOLINE, DIESEL, BIODIESEL_B5, BIODIESEL_B20, BIODIESEL_B100,
    ETHANOL_E10, ETHANOL_E85, CNG, LNG, LPG_PROPANE,
    JET_FUEL_A, AVGAS, MARINE_DIESEL_OIL, HFO, SAF

Example:
    >>> from greenlang.mobile_combustion.vehicle_database import VehicleDatabaseEngine
    >>> db = VehicleDatabaseEngine()
    >>> fuel = db.get_fuel_type("DIESEL")
    >>> print(fuel["co2_ef"])  # Decimal('2.68')
    >>> factors = db.get_ch4_n2o_factors("PASSENGER_CAR_GASOLINE", 2010, "THREE_WAY_CATALYST")
    >>> print(factors["ch4_g_per_km"])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-003 Mobile Combustion (GL-MRV-SCOPE1-003)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["VehicleDatabaseEngine"]

# ---------------------------------------------------------------------------
# Decimal precision constant
# ---------------------------------------------------------------------------
_PRECISION = Decimal("0.00000001")  # 8 decimal places

# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------
from datetime import datetime, timezone


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ===========================================================================
# Vehicle Category Constants
# ===========================================================================

VEHICLE_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "ON_ROAD": {
        "display_name": "On-Road Vehicles",
        "description": "Highway and urban road vehicles including cars, trucks, buses, motorcycles, and vans",
        "source": "EPA Mobile Combustion Tables; IPCC 2006 Vol 2 Ch 3",
    },
    "OFF_ROAD": {
        "display_name": "Off-Road Equipment",
        "description": "Non-road mobile machinery for construction, agriculture, industrial, and mining",
        "source": "EPA NONROAD Model; IPCC 2006 Vol 2 Ch 3",
    },
    "MARINE": {
        "display_name": "Marine Vessels",
        "description": "Inland waterway, coastal, and ocean-going vessels",
        "source": "IMO GHG Study; IPCC 2006 Vol 2 Ch 3",
    },
    "AVIATION": {
        "display_name": "Aviation",
        "description": "Corporate jets, helicopters, and turboprop aircraft",
        "source": "ICAO Engine Emissions Databank; EPA AP-42 Ch 12",
    },
    "RAIL": {
        "display_name": "Rail",
        "description": "Diesel-powered locomotives and rail equipment",
        "source": "EPA Locomotive Emission Standards; IPCC 2006 Vol 2 Ch 3",
    },
}


# ===========================================================================
# Vehicle Type Definitions (18 types across 5 categories)
# ===========================================================================

VEHICLE_TYPES: Dict[str, Dict[str, Any]] = {
    # -----------------------------------------------------------------------
    # ON_ROAD (11 types)
    # -----------------------------------------------------------------------
    "PASSENGER_CAR_GASOLINE": {
        "category": "ON_ROAD",
        "display_name": "Passenger Car (Gasoline)",
        "description": "Light-duty gasoline passenger vehicles up to 3,856 kg GVWR",
        "default_fuel": "GASOLINE",
        "default_fuel_economy_km_per_l": Decimal("12.5"),
        "typical_annual_km": Decimal("19000"),
        "typical_load_factor": Decimal("1.0"),
        "weight_class_kg": (Decimal("0"), Decimal("3856")),
        "source": "EPA Mobile Combustion Table A-1; FHWA",
    },
    "PASSENGER_CAR_DIESEL": {
        "category": "ON_ROAD",
        "display_name": "Passenger Car (Diesel)",
        "description": "Light-duty diesel passenger vehicles up to 3,856 kg GVWR",
        "default_fuel": "DIESEL",
        "default_fuel_economy_km_per_l": Decimal("16.0"),
        "typical_annual_km": Decimal("19000"),
        "typical_load_factor": Decimal("1.0"),
        "weight_class_kg": (Decimal("0"), Decimal("3856")),
        "source": "EPA Mobile Combustion Table A-1; FHWA",
    },
    "PASSENGER_CAR_HYBRID": {
        "category": "ON_ROAD",
        "display_name": "Passenger Car (Hybrid)",
        "description": "Gasoline-electric hybrid passenger vehicles",
        "default_fuel": "GASOLINE",
        "default_fuel_economy_km_per_l": Decimal("21.0"),
        "typical_annual_km": Decimal("19000"),
        "typical_load_factor": Decimal("1.0"),
        "weight_class_kg": (Decimal("0"), Decimal("3856")),
        "source": "EPA Fuel Economy Guide; FHWA",
    },
    "PASSENGER_CAR_PHEV": {
        "category": "ON_ROAD",
        "display_name": "Passenger Car (PHEV)",
        "description": "Plug-in hybrid electric passenger vehicles (fuel combustion only)",
        "default_fuel": "GASOLINE",
        "default_fuel_economy_km_per_l": Decimal("30.0"),
        "typical_annual_km": Decimal("19000"),
        "typical_load_factor": Decimal("1.0"),
        "weight_class_kg": (Decimal("0"), Decimal("3856")),
        "source": "EPA Fuel Economy Guide; DOE AFDC",
    },
    "LIGHT_DUTY_TRUCK": {
        "category": "ON_ROAD",
        "display_name": "Light-Duty Truck",
        "description": "Light-duty trucks and SUVs up to 3,856 kg GVWR (Class 1-2a)",
        "default_fuel": "GASOLINE",
        "default_fuel_economy_km_per_l": Decimal("9.5"),
        "typical_annual_km": Decimal("22000"),
        "typical_load_factor": Decimal("0.5"),
        "weight_class_kg": (Decimal("0"), Decimal("3856")),
        "source": "EPA Mobile Combustion Table A-1; FHWA",
    },
    "MEDIUM_DUTY_TRUCK": {
        "category": "ON_ROAD",
        "display_name": "Medium-Duty Truck",
        "description": "Medium-duty trucks 3,856-11,793 kg GVWR (Class 2b-6)",
        "default_fuel": "DIESEL",
        "default_fuel_economy_km_per_l": Decimal("5.1"),
        "typical_annual_km": Decimal("32000"),
        "typical_load_factor": Decimal("0.6"),
        "weight_class_kg": (Decimal("3856"), Decimal("11793")),
        "source": "EPA Mobile Combustion Table A-1; VIUS",
    },
    "HEAVY_DUTY_TRUCK": {
        "category": "ON_ROAD",
        "display_name": "Heavy-Duty Truck",
        "description": "Heavy-duty trucks over 11,793 kg GVWR (Class 7-8)",
        "default_fuel": "DIESEL",
        "default_fuel_economy_km_per_l": Decimal("2.8"),
        "typical_annual_km": Decimal("80000"),
        "typical_load_factor": Decimal("0.65"),
        "weight_class_kg": (Decimal("11793"), Decimal("36287")),
        "source": "EPA Mobile Combustion Table A-1; VIUS",
    },
    "BUS_DIESEL": {
        "category": "ON_ROAD",
        "display_name": "Bus (Diesel)",
        "description": "Diesel transit and intercity buses",
        "default_fuel": "DIESEL",
        "default_fuel_economy_km_per_l": Decimal("2.6"),
        "typical_annual_km": Decimal("60000"),
        "typical_load_factor": Decimal("0.4"),
        "weight_class_kg": (Decimal("11793"), Decimal("18144")),
        "source": "EPA Mobile Combustion Table A-1; NTD",
    },
    "BUS_CNG": {
        "category": "ON_ROAD",
        "display_name": "Bus (CNG)",
        "description": "Compressed natural gas transit buses",
        "default_fuel": "CNG",
        "default_fuel_economy_km_per_l": Decimal("1.8"),
        "typical_annual_km": Decimal("60000"),
        "typical_load_factor": Decimal("0.4"),
        "weight_class_kg": (Decimal("11793"), Decimal("18144")),
        "source": "EPA AP-42; APTA",
    },
    "MOTORCYCLE": {
        "category": "ON_ROAD",
        "display_name": "Motorcycle",
        "description": "Two- and three-wheeled motor vehicles",
        "default_fuel": "GASOLINE",
        "default_fuel_economy_km_per_l": Decimal("25.0"),
        "typical_annual_km": Decimal("5000"),
        "typical_load_factor": Decimal("1.0"),
        "weight_class_kg": (Decimal("0"), Decimal("500")),
        "source": "EPA Mobile Combustion Table A-1; FHWA",
    },
    "VAN_LCV": {
        "category": "ON_ROAD",
        "display_name": "Van / Light Commercial Vehicle",
        "description": "Vans and light commercial vehicles up to 3,500 kg GVWR",
        "default_fuel": "DIESEL",
        "default_fuel_economy_km_per_l": Decimal("8.0"),
        "typical_annual_km": Decimal("25000"),
        "typical_load_factor": Decimal("0.5"),
        "weight_class_kg": (Decimal("0"), Decimal("3500")),
        "source": "DEFRA 2025; SMMT",
    },
    # -----------------------------------------------------------------------
    # OFF_ROAD (5 types)
    # -----------------------------------------------------------------------
    "CONSTRUCTION_EQUIPMENT": {
        "category": "OFF_ROAD",
        "display_name": "Construction Equipment",
        "description": "Excavators, loaders, bulldozers, graders, compactors, and cranes",
        "default_fuel": "DIESEL",
        "default_fuel_economy_km_per_l": None,
        "typical_annual_hours": Decimal("1500"),
        "fuel_consumption_l_per_hr": Decimal("18.0"),
        "typical_load_factor": Decimal("0.59"),
        "source": "EPA NONROAD Model; IPCC 2006 Vol 2 Ch 3 Table 3.3.1",
    },
    "AGRICULTURAL_EQUIPMENT": {
        "category": "OFF_ROAD",
        "display_name": "Agricultural Equipment",
        "description": "Tractors, harvesters, sprayers, and other farm machinery",
        "default_fuel": "DIESEL",
        "default_fuel_economy_km_per_l": None,
        "typical_annual_hours": Decimal("800"),
        "fuel_consumption_l_per_hr": Decimal("12.0"),
        "typical_load_factor": Decimal("0.48"),
        "source": "EPA NONROAD Model; IPCC 2006 Vol 2 Ch 3 Table 3.3.1",
    },
    "INDUSTRIAL_EQUIPMENT": {
        "category": "OFF_ROAD",
        "display_name": "Industrial Equipment",
        "description": "Industrial engines, generators, compressors, and pumps",
        "default_fuel": "DIESEL",
        "default_fuel_economy_km_per_l": None,
        "typical_annual_hours": Decimal("2000"),
        "fuel_consumption_l_per_hr": Decimal("15.0"),
        "typical_load_factor": Decimal("0.43"),
        "source": "EPA NONROAD Model; IPCC 2006 Vol 2 Ch 3",
    },
    "MINING_EQUIPMENT": {
        "category": "OFF_ROAD",
        "display_name": "Mining Equipment",
        "description": "Dump trucks, haul trucks, drilling equipment, and loaders",
        "default_fuel": "DIESEL",
        "default_fuel_economy_km_per_l": None,
        "typical_annual_hours": Decimal("4000"),
        "fuel_consumption_l_per_hr": Decimal("45.0"),
        "typical_load_factor": Decimal("0.59"),
        "source": "EPA NONROAD Model; IPCC 2006 Vol 2 Ch 3",
    },
    "FORKLIFT": {
        "category": "OFF_ROAD",
        "display_name": "Forklift",
        "description": "Industrial forklifts (diesel, LPG, or gasoline powered)",
        "default_fuel": "LPG_PROPANE",
        "default_fuel_economy_km_per_l": None,
        "typical_annual_hours": Decimal("1800"),
        "fuel_consumption_l_per_hr": Decimal("4.5"),
        "typical_load_factor": Decimal("0.30"),
        "source": "EPA NONROAD Model; ITA",
    },
    # -----------------------------------------------------------------------
    # MARINE (3 types)
    # -----------------------------------------------------------------------
    "MARINE_INLAND": {
        "category": "MARINE",
        "display_name": "Inland Vessel",
        "description": "Barges, river boats, and inland waterway vessels",
        "default_fuel": "MARINE_DIESEL_OIL",
        "default_fuel_economy_km_per_l": None,
        "typical_annual_hours": Decimal("3000"),
        "fuel_consumption_l_per_hr": Decimal("80.0"),
        "typical_load_factor": Decimal("0.50"),
        "source": "IMO GHG Study 2020; IPCC 2006 Vol 2 Ch 3 Table 3.5.3",
    },
    "MARINE_COASTAL": {
        "category": "MARINE",
        "display_name": "Coastal Vessel",
        "description": "Coastal shipping, ferries, and short-sea vessels",
        "default_fuel": "MARINE_DIESEL_OIL",
        "default_fuel_economy_km_per_l": None,
        "typical_annual_hours": Decimal("5000"),
        "fuel_consumption_l_per_hr": Decimal("250.0"),
        "typical_load_factor": Decimal("0.60"),
        "source": "IMO GHG Study 2020; IPCC 2006 Vol 2 Ch 3 Table 3.5.3",
    },
    "MARINE_OCEAN": {
        "category": "MARINE",
        "display_name": "Ocean Vessel",
        "description": "Deep-sea cargo, tanker, and container vessels",
        "default_fuel": "HFO",
        "default_fuel_economy_km_per_l": None,
        "typical_annual_hours": Decimal("6000"),
        "fuel_consumption_l_per_hr": Decimal("1500.0"),
        "typical_load_factor": Decimal("0.70"),
        "source": "IMO GHG Study 2020; IPCC 2006 Vol 2 Ch 3 Table 3.5.3",
    },
    # -----------------------------------------------------------------------
    # AVIATION (3 types)
    # -----------------------------------------------------------------------
    "CORPORATE_JET": {
        "category": "AVIATION",
        "display_name": "Corporate Jet",
        "description": "Business/corporate turbofan jet aircraft",
        "default_fuel": "JET_FUEL_A",
        "default_fuel_economy_km_per_l": Decimal("0.45"),
        "typical_annual_hours": Decimal("400"),
        "fuel_consumption_l_per_hr": Decimal("950.0"),
        "typical_load_factor": Decimal("0.60"),
        "source": "ICAO Engine Emissions Databank; EPA AP-42 Section 12",
    },
    "HELICOPTER": {
        "category": "AVIATION",
        "display_name": "Helicopter",
        "description": "Turboshaft-powered rotorcraft",
        "default_fuel": "JET_FUEL_A",
        "default_fuel_economy_km_per_l": Decimal("0.55"),
        "typical_annual_hours": Decimal("300"),
        "fuel_consumption_l_per_hr": Decimal("350.0"),
        "typical_load_factor": Decimal("0.50"),
        "source": "ICAO Engine Emissions Databank; FAA AEDT",
    },
    "TURBOPROP": {
        "category": "AVIATION",
        "display_name": "Turboprop",
        "description": "Turboprop-powered fixed-wing aircraft",
        "default_fuel": "JET_FUEL_A",
        "default_fuel_economy_km_per_l": Decimal("0.80"),
        "typical_annual_hours": Decimal("500"),
        "fuel_consumption_l_per_hr": Decimal("450.0"),
        "typical_load_factor": Decimal("0.55"),
        "source": "ICAO Engine Emissions Databank; EPA AP-42 Section 12",
    },
    # -----------------------------------------------------------------------
    # RAIL (1 type)
    # -----------------------------------------------------------------------
    "DIESEL_LOCOMOTIVE": {
        "category": "RAIL",
        "display_name": "Diesel Locomotive",
        "description": "Diesel-electric freight and passenger locomotives",
        "default_fuel": "DIESEL",
        "default_fuel_economy_km_per_l": Decimal("0.04"),
        "typical_annual_hours": Decimal("4000"),
        "fuel_consumption_l_per_hr": Decimal("400.0"),
        "typical_load_factor": Decimal("0.65"),
        "source": "EPA Locomotive Emission Standards; AAR; IPCC 2006 Vol 2 Ch 3",
    },
}


# ===========================================================================
# Fuel Type Definitions (15 types)
# ===========================================================================

FUEL_TYPES: Dict[str, Dict[str, Any]] = {
    "GASOLINE": {
        "display_name": "Motor Gasoline",
        "density_kg_per_l": Decimal("0.745"),
        "hhv_mj_per_l": Decimal("34.2"),
        "ncv_mj_per_l": Decimal("32.2"),
        "co2_ef_kg_per_l": Decimal("2.31"),
        "co2_ef_kg_per_kg": Decimal("3.10"),
        "co2_ef_kg_per_gj": Decimal("67.5"),
        "ch4_ef_kg_per_tj": Decimal("3.0"),
        "n2o_ef_kg_per_tj": Decimal("0.6"),
        "biofuel_fraction": Decimal("0.0"),
        "is_biofuel": False,
        "unit": "liters",
        "source": "EPA GHG Emission Factors Hub 2025; IPCC 2006 Vol 2 Ch 3",
    },
    "DIESEL": {
        "display_name": "Diesel / Gas Oil",
        "density_kg_per_l": Decimal("0.832"),
        "hhv_mj_per_l": Decimal("38.6"),
        "ncv_mj_per_l": Decimal("36.4"),
        "co2_ef_kg_per_l": Decimal("2.68"),
        "co2_ef_kg_per_kg": Decimal("3.22"),
        "co2_ef_kg_per_gj": Decimal("74.1"),
        "ch4_ef_kg_per_tj": Decimal("3.9"),
        "n2o_ef_kg_per_tj": Decimal("3.9"),
        "biofuel_fraction": Decimal("0.0"),
        "is_biofuel": False,
        "unit": "liters",
        "source": "EPA GHG Emission Factors Hub 2025; IPCC 2006 Vol 2 Ch 3",
    },
    "BIODIESEL_B5": {
        "display_name": "Biodiesel B5 (5% blend)",
        "density_kg_per_l": Decimal("0.834"),
        "hhv_mj_per_l": Decimal("38.4"),
        "ncv_mj_per_l": Decimal("36.2"),
        "co2_ef_kg_per_l": Decimal("2.68"),
        "co2_ef_kg_per_kg": Decimal("3.21"),
        "co2_ef_kg_per_gj": Decimal("74.1"),
        "ch4_ef_kg_per_tj": Decimal("3.9"),
        "n2o_ef_kg_per_tj": Decimal("3.9"),
        "biofuel_fraction": Decimal("0.05"),
        "is_biofuel": True,
        "unit": "liters",
        "source": "EPA GHG Emission Factors Hub 2025; ASTM D7467",
    },
    "BIODIESEL_B20": {
        "display_name": "Biodiesel B20 (20% blend)",
        "density_kg_per_l": Decimal("0.840"),
        "hhv_mj_per_l": Decimal("37.4"),
        "ncv_mj_per_l": Decimal("35.3"),
        "co2_ef_kg_per_l": Decimal("2.68"),
        "co2_ef_kg_per_kg": Decimal("3.19"),
        "co2_ef_kg_per_gj": Decimal("74.1"),
        "ch4_ef_kg_per_tj": Decimal("3.9"),
        "n2o_ef_kg_per_tj": Decimal("3.9"),
        "biofuel_fraction": Decimal("0.20"),
        "is_biofuel": True,
        "unit": "liters",
        "source": "EPA GHG Emission Factors Hub 2025; ASTM D7467",
    },
    "BIODIESEL_B100": {
        "display_name": "Biodiesel B100 (Pure)",
        "density_kg_per_l": Decimal("0.880"),
        "hhv_mj_per_l": Decimal("33.3"),
        "ncv_mj_per_l": Decimal("31.4"),
        "co2_ef_kg_per_l": Decimal("2.50"),
        "co2_ef_kg_per_kg": Decimal("2.84"),
        "co2_ef_kg_per_gj": Decimal("70.8"),
        "ch4_ef_kg_per_tj": Decimal("3.9"),
        "n2o_ef_kg_per_tj": Decimal("3.9"),
        "biofuel_fraction": Decimal("1.00"),
        "is_biofuel": True,
        "unit": "liters",
        "source": "EPA GHG Emission Factors Hub 2025; ASTM D6751",
    },
    "ETHANOL_E10": {
        "display_name": "Ethanol E10 (10% blend)",
        "density_kg_per_l": Decimal("0.748"),
        "hhv_mj_per_l": Decimal("33.0"),
        "ncv_mj_per_l": Decimal("31.0"),
        "co2_ef_kg_per_l": Decimal("2.21"),
        "co2_ef_kg_per_kg": Decimal("2.96"),
        "co2_ef_kg_per_gj": Decimal("66.0"),
        "ch4_ef_kg_per_tj": Decimal("3.0"),
        "n2o_ef_kg_per_tj": Decimal("0.6"),
        "biofuel_fraction": Decimal("0.10"),
        "is_biofuel": True,
        "unit": "liters",
        "source": "EPA GHG Emission Factors Hub 2025; ASTM D4806",
    },
    "ETHANOL_E85": {
        "display_name": "Ethanol E85 (85% blend)",
        "density_kg_per_l": Decimal("0.781"),
        "hhv_mj_per_l": Decimal("23.4"),
        "ncv_mj_per_l": Decimal("22.0"),
        "co2_ef_kg_per_l": Decimal("1.61"),
        "co2_ef_kg_per_kg": Decimal("2.06"),
        "co2_ef_kg_per_gj": Decimal("66.0"),
        "ch4_ef_kg_per_tj": Decimal("3.0"),
        "n2o_ef_kg_per_tj": Decimal("0.6"),
        "biofuel_fraction": Decimal("0.85"),
        "is_biofuel": True,
        "unit": "liters",
        "source": "EPA GHG Emission Factors Hub 2025; ASTM D5798",
    },
    "CNG": {
        "display_name": "Compressed Natural Gas",
        "density_kg_per_m3": Decimal("0.72"),
        "density_kg_per_l": Decimal("0.18"),
        "hhv_mj_per_m3": Decimal("39.0"),
        "ncv_mj_per_m3": Decimal("35.2"),
        "hhv_mj_per_l": None,
        "ncv_mj_per_l": None,
        "co2_ef_kg_per_m3": Decimal("2.02"),
        "co2_ef_kg_per_l": None,
        "co2_ef_kg_per_kg": Decimal("2.74"),
        "co2_ef_kg_per_gj": Decimal("56.1"),
        "ch4_ef_kg_per_tj": Decimal("92.0"),
        "n2o_ef_kg_per_tj": Decimal("3.0"),
        "biofuel_fraction": Decimal("0.0"),
        "is_biofuel": False,
        "unit": "m3",
        "source": "EPA GHG Emission Factors Hub 2025; IPCC 2006 Vol 2 Ch 3",
    },
    "LNG": {
        "display_name": "Liquefied Natural Gas",
        "density_kg_per_l": Decimal("0.450"),
        "hhv_mj_per_l": Decimal("25.3"),
        "ncv_mj_per_l": Decimal("22.8"),
        "co2_ef_kg_per_l": Decimal("2.75"),
        "co2_ef_kg_per_kg": Decimal("2.74"),
        "co2_ef_kg_per_gj": Decimal("56.1"),
        "ch4_ef_kg_per_tj": Decimal("92.0"),
        "n2o_ef_kg_per_tj": Decimal("3.0"),
        "biofuel_fraction": Decimal("0.0"),
        "is_biofuel": False,
        "unit": "liters",
        "source": "EPA GHG Emission Factors Hub 2025; IPCC 2006 Vol 2 Ch 3",
    },
    "LPG_PROPANE": {
        "display_name": "LPG / Propane",
        "density_kg_per_l": Decimal("0.510"),
        "hhv_mj_per_l": Decimal("25.3"),
        "ncv_mj_per_l": Decimal("23.4"),
        "co2_ef_kg_per_l": Decimal("1.51"),
        "co2_ef_kg_per_kg": Decimal("2.96"),
        "co2_ef_kg_per_gj": Decimal("63.1"),
        "ch4_ef_kg_per_tj": Decimal("1.0"),
        "n2o_ef_kg_per_tj": Decimal("0.1"),
        "biofuel_fraction": Decimal("0.0"),
        "is_biofuel": False,
        "unit": "liters",
        "source": "EPA GHG Emission Factors Hub 2025; IPCC 2006 Vol 2 Ch 3",
    },
    "JET_FUEL_A": {
        "display_name": "Jet Fuel A / Kerosene Type",
        "density_kg_per_l": Decimal("0.804"),
        "hhv_mj_per_l": Decimal("37.4"),
        "ncv_mj_per_l": Decimal("35.3"),
        "co2_ef_kg_per_l": Decimal("2.52"),
        "co2_ef_kg_per_kg": Decimal("3.13"),
        "co2_ef_kg_per_gj": Decimal("71.5"),
        "ch4_ef_kg_per_tj": Decimal("0.5"),
        "n2o_ef_kg_per_tj": Decimal("2.0"),
        "biofuel_fraction": Decimal("0.0"),
        "is_biofuel": False,
        "unit": "liters",
        "source": "ICAO Carbon Calculator; IPCC 2006 Vol 2 Ch 3",
    },
    "AVGAS": {
        "display_name": "Aviation Gasoline",
        "density_kg_per_l": Decimal("0.721"),
        "hhv_mj_per_l": Decimal("33.5"),
        "ncv_mj_per_l": Decimal("31.5"),
        "co2_ef_kg_per_l": Decimal("2.22"),
        "co2_ef_kg_per_kg": Decimal("3.08"),
        "co2_ef_kg_per_gj": Decimal("69.3"),
        "ch4_ef_kg_per_tj": Decimal("0.5"),
        "n2o_ef_kg_per_tj": Decimal("2.0"),
        "biofuel_fraction": Decimal("0.0"),
        "is_biofuel": False,
        "unit": "liters",
        "source": "FAA; IPCC 2006 Vol 2 Ch 3",
    },
    "MARINE_DIESEL_OIL": {
        "display_name": "Marine Diesel Oil (MDO)",
        "density_kg_per_l": Decimal("0.890"),
        "hhv_mj_per_l": Decimal("40.2"),
        "ncv_mj_per_l": Decimal("37.9"),
        "co2_ef_kg_per_l": Decimal("2.68"),
        "co2_ef_kg_per_kg": Decimal("3.21"),
        "co2_ef_kg_per_gj": Decimal("74.1"),
        "ch4_ef_kg_per_tj": Decimal("7.0"),
        "n2o_ef_kg_per_tj": Decimal("2.0"),
        "biofuel_fraction": Decimal("0.0"),
        "is_biofuel": False,
        "unit": "liters",
        "source": "IMO GHG Study 2020; IPCC 2006 Vol 2 Ch 3",
    },
    "HFO": {
        "display_name": "Heavy Fuel Oil",
        "density_kg_per_l": Decimal("0.980"),
        "hhv_mj_per_l": Decimal("41.7"),
        "ncv_mj_per_l": Decimal("39.3"),
        "co2_ef_kg_per_l": Decimal("3.11"),
        "co2_ef_kg_per_kg": Decimal("3.17"),
        "co2_ef_kg_per_gj": Decimal("77.4"),
        "ch4_ef_kg_per_tj": Decimal("7.0"),
        "n2o_ef_kg_per_tj": Decimal("2.0"),
        "biofuel_fraction": Decimal("0.0"),
        "is_biofuel": False,
        "unit": "liters",
        "source": "IMO GHG Study 2020; IPCC 2006 Vol 2 Ch 3",
    },
    "SAF": {
        "display_name": "Sustainable Aviation Fuel",
        "density_kg_per_l": Decimal("0.790"),
        "hhv_mj_per_l": Decimal("36.8"),
        "ncv_mj_per_l": Decimal("34.7"),
        "co2_ef_kg_per_l": Decimal("2.52"),
        "co2_ef_kg_per_kg": Decimal("3.19"),
        "co2_ef_kg_per_gj": Decimal("71.5"),
        "ch4_ef_kg_per_tj": Decimal("0.5"),
        "n2o_ef_kg_per_tj": Decimal("2.0"),
        "biofuel_fraction": Decimal("0.50"),
        "is_biofuel": True,
        "unit": "liters",
        "source": "ICAO CORSIA; ASTM D7566",
    },
}


# ===========================================================================
# CH4 and N2O Emission Factors by Vehicle Type, Model Year, Fuel
# Units: g per km (grams per kilometer)
# Source: EPA Mobile Combustion Tables A-1 through A-7; IPCC 2006 Vol 2 Ch 3
# ===========================================================================

# Passenger cars and light trucks - gasoline (g/km)
_CH4_N2O_ONROAD_GASOLINE: Dict[str, Dict[str, Dict[str, Decimal]]] = {
    # {model_year_range: {vehicle_type: {gas: value}}}
    "PRE_1985": {
        "PASSENGER_CAR_GASOLINE": {"CH4": Decimal("0.0602"), "N2O": Decimal("0.0063")},
        "PASSENGER_CAR_HYBRID": {"CH4": Decimal("0.0301"), "N2O": Decimal("0.0032")},
        "PASSENGER_CAR_PHEV": {"CH4": Decimal("0.0301"), "N2O": Decimal("0.0032")},
        "LIGHT_DUTY_TRUCK": {"CH4": Decimal("0.0751"), "N2O": Decimal("0.0079")},
        "MOTORCYCLE": {"CH4": Decimal("0.0800"), "N2O": Decimal("0.0032")},
        "VAN_LCV": {"CH4": Decimal("0.0650"), "N2O": Decimal("0.0070")},
    },
    "1985_1995": {
        "PASSENGER_CAR_GASOLINE": {"CH4": Decimal("0.0394"), "N2O": Decimal("0.0561")},
        "PASSENGER_CAR_HYBRID": {"CH4": Decimal("0.0197"), "N2O": Decimal("0.0281")},
        "PASSENGER_CAR_PHEV": {"CH4": Decimal("0.0197"), "N2O": Decimal("0.0281")},
        "LIGHT_DUTY_TRUCK": {"CH4": Decimal("0.0512"), "N2O": Decimal("0.0730")},
        "MOTORCYCLE": {"CH4": Decimal("0.0500"), "N2O": Decimal("0.0100")},
        "VAN_LCV": {"CH4": Decimal("0.0450"), "N2O": Decimal("0.0650")},
    },
    "1996_2005": {
        "PASSENGER_CAR_GASOLINE": {"CH4": Decimal("0.0226"), "N2O": Decimal("0.0376")},
        "PASSENGER_CAR_HYBRID": {"CH4": Decimal("0.0113"), "N2O": Decimal("0.0188")},
        "PASSENGER_CAR_PHEV": {"CH4": Decimal("0.0113"), "N2O": Decimal("0.0188")},
        "LIGHT_DUTY_TRUCK": {"CH4": Decimal("0.0312"), "N2O": Decimal("0.0514")},
        "MOTORCYCLE": {"CH4": Decimal("0.0300"), "N2O": Decimal("0.0080")},
        "VAN_LCV": {"CH4": Decimal("0.0280"), "N2O": Decimal("0.0450")},
    },
    "2006_PLUS": {
        "PASSENGER_CAR_GASOLINE": {"CH4": Decimal("0.0113"), "N2O": Decimal("0.0132")},
        "PASSENGER_CAR_HYBRID": {"CH4": Decimal("0.0057"), "N2O": Decimal("0.0066")},
        "PASSENGER_CAR_PHEV": {"CH4": Decimal("0.0057"), "N2O": Decimal("0.0066")},
        "LIGHT_DUTY_TRUCK": {"CH4": Decimal("0.0156"), "N2O": Decimal("0.0178")},
        "MOTORCYCLE": {"CH4": Decimal("0.0200"), "N2O": Decimal("0.0050")},
        "VAN_LCV": {"CH4": Decimal("0.0140"), "N2O": Decimal("0.0155")},
    },
}

# Passenger cars and light trucks - diesel (g/km)
_CH4_N2O_ONROAD_DIESEL: Dict[str, Dict[str, Dict[str, Decimal]]] = {
    "PRE_1985": {
        "PASSENGER_CAR_DIESEL": {"CH4": Decimal("0.0051"), "N2O": Decimal("0.0014")},
        "LIGHT_DUTY_TRUCK": {"CH4": Decimal("0.0063"), "N2O": Decimal("0.0018")},
        "VAN_LCV": {"CH4": Decimal("0.0057"), "N2O": Decimal("0.0016")},
    },
    "1985_1995": {
        "PASSENGER_CAR_DIESEL": {"CH4": Decimal("0.0034"), "N2O": Decimal("0.0014")},
        "LIGHT_DUTY_TRUCK": {"CH4": Decimal("0.0044"), "N2O": Decimal("0.0018")},
        "VAN_LCV": {"CH4": Decimal("0.0039"), "N2O": Decimal("0.0016")},
    },
    "1996_2005": {
        "PASSENGER_CAR_DIESEL": {"CH4": Decimal("0.0017"), "N2O": Decimal("0.0014")},
        "LIGHT_DUTY_TRUCK": {"CH4": Decimal("0.0022"), "N2O": Decimal("0.0018")},
        "VAN_LCV": {"CH4": Decimal("0.0020"), "N2O": Decimal("0.0016")},
    },
    "2006_PLUS": {
        "PASSENGER_CAR_DIESEL": {"CH4": Decimal("0.0005"), "N2O": Decimal("0.0014")},
        "LIGHT_DUTY_TRUCK": {"CH4": Decimal("0.0007"), "N2O": Decimal("0.0018")},
        "VAN_LCV": {"CH4": Decimal("0.0006"), "N2O": Decimal("0.0016")},
    },
}

# Heavy-duty on-road vehicles (g/km)
_CH4_N2O_HEAVY_DUTY: Dict[str, Dict[str, Decimal]] = {
    "MEDIUM_DUTY_TRUCK": {"CH4": Decimal("0.0176"), "N2O": Decimal("0.0150")},
    "HEAVY_DUTY_TRUCK": {"CH4": Decimal("0.0251"), "N2O": Decimal("0.0200")},
    "BUS_DIESEL": {"CH4": Decimal("0.0251"), "N2O": Decimal("0.0200")},
    "BUS_CNG": {"CH4": Decimal("1.9660"), "N2O": Decimal("0.0314")},
}

# Off-road equipment CH4/N2O factors (g/kg-fuel)
_CH4_N2O_OFFROAD: Dict[str, Dict[str, Decimal]] = {
    "CONSTRUCTION_EQUIPMENT": {"CH4": Decimal("0.17"), "N2O": Decimal("0.12")},
    "AGRICULTURAL_EQUIPMENT": {"CH4": Decimal("0.17"), "N2O": Decimal("0.12")},
    "INDUSTRIAL_EQUIPMENT": {"CH4": Decimal("0.17"), "N2O": Decimal("0.12")},
    "MINING_EQUIPMENT": {"CH4": Decimal("0.20"), "N2O": Decimal("0.15")},
    "FORKLIFT": {"CH4": Decimal("0.25"), "N2O": Decimal("0.08")},
}

# Marine vessel CH4/N2O factors (g/kg-fuel)
_CH4_N2O_MARINE: Dict[str, Dict[str, Decimal]] = {
    "MARINE_INLAND": {"CH4": Decimal("0.30"), "N2O": Decimal("0.08")},
    "MARINE_COASTAL": {"CH4": Decimal("0.30"), "N2O": Decimal("0.08")},
    "MARINE_OCEAN": {"CH4": Decimal("0.30"), "N2O": Decimal("0.08")},
}

# Aviation CH4/N2O factors (g/kg-fuel)
_CH4_N2O_AVIATION: Dict[str, Dict[str, Decimal]] = {
    "CORPORATE_JET": {"CH4": Decimal("0.02"), "N2O": Decimal("0.10")},
    "HELICOPTER": {"CH4": Decimal("0.02"), "N2O": Decimal("0.10")},
    "TURBOPROP": {"CH4": Decimal("0.02"), "N2O": Decimal("0.10")},
}

# Rail CH4/N2O factors (g/kg-fuel)
_CH4_N2O_RAIL: Dict[str, Dict[str, Decimal]] = {
    "DIESEL_LOCOMOTIVE": {"CH4": Decimal("0.25"), "N2O": Decimal("0.30")},
}


# ===========================================================================
# Distance-Based Emission Factors (g CO2e per km)
# Combines CO2, CH4, N2O into a single g CO2e/km for each vehicle type
# Source: DEFRA 2025, EPA SmartWay
# ===========================================================================

_DISTANCE_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    # {vehicle_type: {fuel_type: g_CO2e_per_km}}
    "PASSENGER_CAR_GASOLINE": {"GASOLINE": Decimal("192.0"), "ETHANOL_E10": Decimal("183.0"), "ETHANOL_E85": Decimal("135.0")},
    "PASSENGER_CAR_DIESEL": {"DIESEL": Decimal("167.0"), "BIODIESEL_B5": Decimal("163.0"), "BIODIESEL_B20": Decimal("147.0")},
    "PASSENGER_CAR_HYBRID": {"GASOLINE": Decimal("110.0"), "ETHANOL_E10": Decimal("105.0")},
    "PASSENGER_CAR_PHEV": {"GASOLINE": Decimal("77.0")},
    "LIGHT_DUTY_TRUCK": {"GASOLINE": Decimal("243.0"), "DIESEL": Decimal("209.0")},
    "MEDIUM_DUTY_TRUCK": {"DIESEL": Decimal("523.0")},
    "HEAVY_DUTY_TRUCK": {"DIESEL": Decimal("951.0")},
    "BUS_DIESEL": {"DIESEL": Decimal("1020.0")},
    "BUS_CNG": {"CNG": Decimal("1120.0")},
    "MOTORCYCLE": {"GASOLINE": Decimal("92.0")},
    "VAN_LCV": {"DIESEL": Decimal("250.0"), "GASOLINE": Decimal("280.0")},
}


# ===========================================================================
# Emission Control Technology Adjustments
# ===========================================================================

CONTROL_TECHNOLOGIES: Dict[str, Dict[str, Decimal]] = {
    "UNCONTROLLED": {
        "display_name": "Uncontrolled",
        "ch4_multiplier": Decimal("1.0"),
        "n2o_multiplier": Decimal("1.0"),
        "description": "No emission control system present",
        "source": "EPA AP-42; IPCC 2006 Vol 2 Ch 3",
    },
    "OXIDATION_CATALYST": {
        "display_name": "Oxidation Catalyst",
        "ch4_multiplier": Decimal("0.8"),
        "n2o_multiplier": Decimal("0.9"),
        "description": "Two-way catalytic converter (oxidation only, no NOx)",
        "source": "EPA AP-42; IPCC 2006 Vol 2 Ch 3 Table 3.2.5",
    },
    "THREE_WAY_CATALYST": {
        "display_name": "Three-Way Catalyst",
        "ch4_multiplier": Decimal("0.3"),
        "n2o_multiplier": Decimal("1.5"),
        "description": "Three-way catalytic converter with lambda sensor (generates N2O)",
        "source": "EPA AP-42; IPCC 2006 Vol 2 Ch 3 Table 3.2.5",
    },
    "ADVANCED_CATALYST": {
        "display_name": "Advanced Three-Way Catalyst",
        "ch4_multiplier": Decimal("0.1"),
        "n2o_multiplier": Decimal("0.5"),
        "description": "Advanced TWC with close-coupled and underfloor configuration",
        "source": "EPA Tier 2/3 Standards; Euro 5/6",
    },
    "EURO_1": {
        "display_name": "Euro 1 (1992)",
        "ch4_multiplier": Decimal("0.60"),
        "n2o_multiplier": Decimal("1.30"),
        "description": "European emission standard Euro 1 (ECE R83-02)",
        "source": "EEA EMEP/CORINAIR; Directive 91/441/EEC",
    },
    "EURO_2": {
        "display_name": "Euro 2 (1996)",
        "ch4_multiplier": Decimal("0.45"),
        "n2o_multiplier": Decimal("1.20"),
        "description": "European emission standard Euro 2 (ECE R83-03)",
        "source": "EEA EMEP/CORINAIR; Directive 94/12/EC",
    },
    "EURO_3": {
        "display_name": "Euro 3 (2000)",
        "ch4_multiplier": Decimal("0.35"),
        "n2o_multiplier": Decimal("1.00"),
        "description": "European emission standard Euro 3 with OBD I",
        "source": "EEA EMEP/CORINAIR; Directive 98/69/EC",
    },
    "EURO_4": {
        "display_name": "Euro 4 (2005)",
        "ch4_multiplier": Decimal("0.25"),
        "n2o_multiplier": Decimal("0.80"),
        "description": "European emission standard Euro 4 with OBD II",
        "source": "EEA EMEP/CORINAIR; Directive 98/69/EC",
    },
    "EURO_5": {
        "display_name": "Euro 5 (2009)",
        "ch4_multiplier": Decimal("0.15"),
        "n2o_multiplier": Decimal("0.60"),
        "description": "European emission standard Euro 5 with DPF (diesel)",
        "source": "EEA EMEP/CORINAIR; Regulation (EC) No 715/2007",
    },
    "EURO_6": {
        "display_name": "Euro 6 (2014)",
        "ch4_multiplier": Decimal("0.10"),
        "n2o_multiplier": Decimal("0.40"),
        "description": "European emission standard Euro 6 with SCR/DPF",
        "source": "EEA EMEP/CORINAIR; Regulation (EC) No 715/2007",
    },
}


# ===========================================================================
# GWP Values by Assessment Report
# ===========================================================================

_GWP_VALUES: Dict[str, Dict[str, Decimal]] = {
    "AR4": {"CO2": Decimal("1"), "CH4": Decimal("25"), "N2O": Decimal("298")},
    "AR5": {"CO2": Decimal("1"), "CH4": Decimal("28"), "N2O": Decimal("265")},
    "AR6": {"CO2": Decimal("1"), "CH4": Decimal("29.8"), "N2O": Decimal("273")},
    "AR6_20YR": {"CO2": Decimal("1"), "CH4": Decimal("82.5"), "N2O": Decimal("273")},
}


# ===========================================================================
# VehicleDatabaseEngine
# ===========================================================================


class VehicleDatabaseEngine:
    """Manages vehicle types, fuel types, and emission factors for mobile combustion.

    This engine is the authoritative source for all mobile combustion emission
    factor data. It supports 18 vehicle types across 5 categories, 15 fuel types
    with complete physical properties, and vehicle-specific CH4/N2O emission
    factors by model year range and emission control technology.

    All data is hardcoded as authoritative constants from EPA, IPCC, DEFRA,
    and IMO sources for deterministic, zero-hallucination lookups.

    Thread-safe: all mutable state is guarded by ``threading.Lock()``.

    Attributes:
        _config: Optional configuration dictionary.
        _custom_factors: Registry of user-defined emission factors.
        _lock: Thread lock for custom factor mutations.

    Example:
        >>> db = VehicleDatabaseEngine()
        >>> fuel = db.get_fuel_type("DIESEL")
        >>> print(fuel["co2_ef_kg_per_l"])  # Decimal('2.68')
        >>> veh = db.get_vehicle_type("HEAVY_DUTY_TRUCK")
        >>> print(veh["category"])  # 'ON_ROAD'
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize VehicleDatabaseEngine with optional configuration.

        Loads all built-in vehicle types, fuel types, and emission factors.
        No external database calls are made; all data is held in-memory for
        deterministic, zero-latency lookups.

        Args:
            config: Optional configuration dict. Currently supports:
                - ``enable_provenance`` (bool): Enable provenance tracking.
                  Defaults to True.
        """
        self._config: Dict[str, Any] = config or {}
        self._custom_factors: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._enable_provenance: bool = self._config.get("enable_provenance", True)

        logger.info(
            "VehicleDatabaseEngine initialized with %d vehicle types, "
            "%d fuel types, %d control technologies",
            len(VEHICLE_TYPES),
            len(FUEL_TYPES),
            len(CONTROL_TECHNOLOGIES),
        )

    # ------------------------------------------------------------------
    # Public API: Vehicle Type Lookups
    # ------------------------------------------------------------------

    def get_vehicle_type(self, vehicle_type: str) -> Dict[str, Any]:
        """Return full definition of a vehicle type.

        Args:
            vehicle_type: Vehicle type identifier (e.g. ``"HEAVY_DUTY_TRUCK"``).

        Returns:
            Dictionary with category, display_name, default_fuel,
            fuel_economy, load_factor, and source reference.

        Raises:
            ValueError: If vehicle_type is not recognized.
        """
        key = vehicle_type.upper().strip()
        if key not in VEHICLE_TYPES:
            raise ValueError(
                f"Unknown vehicle type: '{vehicle_type}'. "
                f"Valid types: {sorted(VEHICLE_TYPES.keys())}"
            )
        result = dict(VEHICLE_TYPES[key])
        result["vehicle_type"] = key

        if self._enable_provenance:
            result["_provenance_hash"] = self._compute_hash(
                "vehicle_type_lookup", {"vehicle_type": key}
            )

        logger.debug("Vehicle type lookup: %s -> %s", key, result.get("display_name"))
        return result

    def list_vehicle_types(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all vehicle types, optionally filtered by category.

        Args:
            category: Optional category filter (e.g. ``"ON_ROAD"``, ``"OFF_ROAD"``).
                If None, all vehicle types are returned.

        Returns:
            List of vehicle type dictionaries, each including the type key.

        Raises:
            ValueError: If category is specified but not recognized.
        """
        if category is not None:
            cat_key = category.upper().strip()
            if cat_key not in VEHICLE_CATEGORIES:
                raise ValueError(
                    f"Unknown vehicle category: '{category}'. "
                    f"Valid categories: {sorted(VEHICLE_CATEGORIES.keys())}"
                )
        else:
            cat_key = None

        results: List[Dict[str, Any]] = []
        for vtype_key, vtype_data in sorted(VEHICLE_TYPES.items()):
            if cat_key is not None and vtype_data["category"] != cat_key:
                continue
            entry = dict(vtype_data)
            entry["vehicle_type"] = vtype_key
            results.append(entry)

        logger.debug(
            "Listed %d vehicle types (category=%s)", len(results), cat_key or "ALL"
        )
        return results

    # ------------------------------------------------------------------
    # Public API: Fuel Type Lookups
    # ------------------------------------------------------------------

    def get_fuel_type(self, fuel_type: str) -> Dict[str, Any]:
        """Return full definition and properties of a fuel type.

        Args:
            fuel_type: Fuel type identifier (e.g. ``"DIESEL"``, ``"CNG"``).

        Returns:
            Dictionary with density, heating values, CO2 emission factor,
            CH4/N2O factors, biofuel fraction, and source reference.

        Raises:
            ValueError: If fuel_type is not recognized.
        """
        key = fuel_type.upper().strip()
        if key not in FUEL_TYPES:
            raise ValueError(
                f"Unknown fuel type: '{fuel_type}'. "
                f"Valid types: {sorted(FUEL_TYPES.keys())}"
            )
        result = dict(FUEL_TYPES[key])
        result["fuel_type"] = key

        if self._enable_provenance:
            result["_provenance_hash"] = self._compute_hash(
                "fuel_type_lookup", {"fuel_type": key}
            )

        logger.debug("Fuel type lookup: %s -> %s", key, result.get("display_name"))
        return result

    def list_fuel_types(self) -> List[Dict[str, Any]]:
        """List all available fuel types with their properties.

        Returns:
            List of fuel type dictionaries, each including the type key.
        """
        results: List[Dict[str, Any]] = []
        for ftype_key, ftype_data in sorted(FUEL_TYPES.items()):
            entry = dict(ftype_data)
            entry["fuel_type"] = ftype_key
            results.append(entry)

        logger.debug("Listed %d fuel types", len(results))
        return results

    # ------------------------------------------------------------------
    # Public API: Emission Factor Lookups
    # ------------------------------------------------------------------

    def get_emission_factor(
        self,
        vehicle_type: str,
        fuel_type: str,
        gas: str,
        source: str = "EPA",
    ) -> Dict[str, Any]:
        """Retrieve a specific emission factor for a vehicle/fuel/gas combination.

        For CO2, returns the fuel-based CO2 factor (kg per liter or kg per m3).
        For CH4 and N2O, returns vehicle-specific factors that may vary by
        model year and control technology.

        Args:
            vehicle_type: Vehicle type identifier.
            fuel_type: Fuel type identifier.
            gas: Greenhouse gas (``"CO2"``, ``"CH4"``, ``"N2O"``).
            source: Factor source authority (``"EPA"``, ``"IPCC"``, ``"DEFRA"``).

        Returns:
            Dictionary with value, unit, gas, source, and provenance hash.

        Raises:
            ValueError: If any identifier is not recognized.
        """
        vtype = vehicle_type.upper().strip()
        ftype = fuel_type.upper().strip()
        gas_key = gas.upper().strip()

        if vtype not in VEHICLE_TYPES:
            raise ValueError(f"Unknown vehicle type: '{vehicle_type}'")
        if ftype not in FUEL_TYPES:
            raise ValueError(f"Unknown fuel type: '{fuel_type}'")
        if gas_key not in ("CO2", "CH4", "N2O"):
            raise ValueError(f"Unknown gas: '{gas}'. Must be CO2, CH4, or N2O")

        fuel_data = FUEL_TYPES[ftype]

        if gas_key == "CO2":
            value = fuel_data.get("co2_ef_kg_per_l")
            unit = "kg CO2/L"
            if value is None:
                value = fuel_data.get("co2_ef_kg_per_m3")
                unit = "kg CO2/m3"
            if value is None:
                value = fuel_data.get("co2_ef_kg_per_kg")
                unit = "kg CO2/kg"
        elif gas_key == "CH4":
            value = fuel_data.get("ch4_ef_kg_per_tj")
            unit = "kg CH4/TJ"
        else:
            value = fuel_data.get("n2o_ef_kg_per_tj")
            unit = "kg N2O/TJ"

        result: Dict[str, Any] = {
            "vehicle_type": vtype,
            "fuel_type": ftype,
            "gas": gas_key,
            "value": value,
            "unit": unit,
            "source": source.upper().strip(),
        }

        if self._enable_provenance:
            result["_provenance_hash"] = self._compute_hash(
                "emission_factor_lookup",
                {"vehicle_type": vtype, "fuel_type": ftype, "gas": gas_key, "source": source},
            )

        logger.debug(
            "Emission factor lookup: %s/%s/%s -> %s %s",
            vtype, ftype, gas_key, value, unit,
        )
        return result

    def get_ch4_n2o_factors(
        self,
        vehicle_type: str,
        model_year: Optional[int] = None,
        control_technology: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve CH4 and N2O emission factors for a specific vehicle.

        For on-road gasoline/diesel light-duty vehicles, factors vary by model
        year range. For heavy-duty, off-road, marine, aviation, and rail,
        factors are fixed per vehicle type. Control technology adjustments
        are applied as multipliers.

        Args:
            vehicle_type: Vehicle type identifier.
            model_year: Model year of the vehicle (e.g. 2010). Required for
                on-road light-duty vehicles. If None, uses 2006+ defaults.
            control_technology: Emission control technology identifier
                (e.g. ``"THREE_WAY_CATALYST"``). If None, no adjustment.

        Returns:
            Dictionary with ch4_g_per_km, n2o_g_per_km (or g/kg-fuel for
            non-road), model_year_range, control_technology applied, and
            source reference.

        Raises:
            ValueError: If vehicle_type is not recognized.
        """
        vtype = vehicle_type.upper().strip()
        if vtype not in VEHICLE_TYPES:
            raise ValueError(f"Unknown vehicle type: '{vehicle_type}'")

        veh_data = VEHICLE_TYPES[vtype]
        category = veh_data["category"]

        # Determine base CH4/N2O factors
        ch4_value: Decimal
        n2o_value: Decimal
        unit: str
        year_range: str = "ALL_YEARS"

        if category == "ON_ROAD" and vtype in (
            "MEDIUM_DUTY_TRUCK", "HEAVY_DUTY_TRUCK", "BUS_DIESEL", "BUS_CNG",
        ):
            # Heavy-duty: fixed factors
            hd_factors = _CH4_N2O_HEAVY_DUTY.get(vtype)
            if hd_factors is None:
                raise ValueError(f"No CH4/N2O factors for heavy-duty type: {vtype}")
            ch4_value = hd_factors["CH4"]
            n2o_value = hd_factors["N2O"]
            unit = "g/km"
            year_range = "ALL_YEARS"

        elif category == "ON_ROAD":
            # Light-duty on-road: factors vary by model year
            year_range = self._resolve_model_year_range(model_year)
            default_fuel = veh_data.get("default_fuel", "GASOLINE")

            if default_fuel == "DIESEL":
                year_data = _CH4_N2O_ONROAD_DIESEL.get(year_range, {})
            else:
                year_data = _CH4_N2O_ONROAD_GASOLINE.get(year_range, {})

            vtype_factors = year_data.get(vtype)
            if vtype_factors is None:
                # Fallback: use latest year range
                fallback_range = "2006_PLUS"
                if default_fuel == "DIESEL":
                    fallback_data = _CH4_N2O_ONROAD_DIESEL.get(fallback_range, {})
                else:
                    fallback_data = _CH4_N2O_ONROAD_GASOLINE.get(fallback_range, {})
                vtype_factors = fallback_data.get(vtype, {"CH4": Decimal("0.0100"), "N2O": Decimal("0.0100")})
                year_range = fallback_range

            ch4_value = vtype_factors["CH4"]
            n2o_value = vtype_factors["N2O"]
            unit = "g/km"

        elif category == "OFF_ROAD":
            offroad_factors = _CH4_N2O_OFFROAD.get(vtype)
            if offroad_factors is None:
                raise ValueError(f"No CH4/N2O factors for off-road type: {vtype}")
            ch4_value = offroad_factors["CH4"]
            n2o_value = offroad_factors["N2O"]
            unit = "g/kg-fuel"

        elif category == "MARINE":
            marine_factors = _CH4_N2O_MARINE.get(vtype)
            if marine_factors is None:
                raise ValueError(f"No CH4/N2O factors for marine type: {vtype}")
            ch4_value = marine_factors["CH4"]
            n2o_value = marine_factors["N2O"]
            unit = "g/kg-fuel"

        elif category == "AVIATION":
            aviation_factors = _CH4_N2O_AVIATION.get(vtype)
            if aviation_factors is None:
                raise ValueError(f"No CH4/N2O factors for aviation type: {vtype}")
            ch4_value = aviation_factors["CH4"]
            n2o_value = aviation_factors["N2O"]
            unit = "g/kg-fuel"

        elif category == "RAIL":
            rail_factors = _CH4_N2O_RAIL.get(vtype)
            if rail_factors is None:
                raise ValueError(f"No CH4/N2O factors for rail type: {vtype}")
            ch4_value = rail_factors["CH4"]
            n2o_value = rail_factors["N2O"]
            unit = "g/kg-fuel"

        else:
            raise ValueError(f"Unsupported vehicle category: {category}")

        # Apply control technology adjustment
        tech_applied = "NONE"
        if control_technology is not None:
            tech_key = control_technology.upper().strip()
            if tech_key not in CONTROL_TECHNOLOGIES:
                raise ValueError(
                    f"Unknown control technology: '{control_technology}'. "
                    f"Valid: {sorted(CONTROL_TECHNOLOGIES.keys())}"
                )
            tech_data = CONTROL_TECHNOLOGIES[tech_key]
            ch4_value = (ch4_value * tech_data["ch4_multiplier"]).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )
            n2o_value = (n2o_value * tech_data["n2o_multiplier"]).quantize(
                _PRECISION, rounding=ROUND_HALF_UP
            )
            tech_applied = tech_key

        result: Dict[str, Any] = {
            "vehicle_type": vtype,
            "ch4_value": ch4_value,
            "n2o_value": n2o_value,
            "unit": unit,
            "model_year_range": year_range,
            "control_technology": tech_applied,
            "source": "EPA Mobile Combustion Tables; IPCC 2006 Vol 2 Ch 3",
        }

        if self._enable_provenance:
            result["_provenance_hash"] = self._compute_hash(
                "ch4_n2o_factor_lookup",
                {
                    "vehicle_type": vtype,
                    "model_year": str(model_year),
                    "control_technology": tech_applied,
                    "ch4": str(ch4_value),
                    "n2o": str(n2o_value),
                },
            )

        logger.debug(
            "CH4/N2O factor lookup: %s (year=%s, tech=%s) -> CH4=%s, N2O=%s %s",
            vtype, model_year, tech_applied, ch4_value, n2o_value, unit,
        )
        return result

    def get_fuel_density(self, fuel_type: str) -> Decimal:
        """Return fuel density in kg/L (or kg/m3 for CNG).

        Args:
            fuel_type: Fuel type identifier.

        Returns:
            Density as Decimal.

        Raises:
            ValueError: If fuel_type is not recognized or density unavailable.
        """
        key = fuel_type.upper().strip()
        if key not in FUEL_TYPES:
            raise ValueError(f"Unknown fuel type: '{fuel_type}'")

        fuel_data = FUEL_TYPES[key]
        density = fuel_data.get("density_kg_per_l")
        if density is None:
            density = fuel_data.get("density_kg_per_m3")
        if density is None:
            raise ValueError(
                f"No density data available for fuel type: '{fuel_type}'"
            )

        logger.debug("Fuel density lookup: %s -> %s", key, density)
        return density

    def get_heating_value(
        self, fuel_type: str, basis: str = "HHV"
    ) -> Decimal:
        """Return heating value of a fuel type.

        Args:
            fuel_type: Fuel type identifier.
            basis: ``"HHV"`` (higher/gross) or ``"NCV"`` (net/lower).

        Returns:
            Heating value as Decimal in MJ/L (or MJ/m3 for CNG).

        Raises:
            ValueError: If fuel_type or basis is invalid, or value unavailable.
        """
        key = fuel_type.upper().strip()
        basis_key = basis.upper().strip()

        if key not in FUEL_TYPES:
            raise ValueError(f"Unknown fuel type: '{fuel_type}'")
        if basis_key not in ("HHV", "NCV"):
            raise ValueError(f"Invalid heating value basis: '{basis}'. Must be HHV or NCV")

        fuel_data = FUEL_TYPES[key]

        if basis_key == "HHV":
            value = fuel_data.get("hhv_mj_per_l")
            if value is None:
                value = fuel_data.get("hhv_mj_per_m3")
        else:
            value = fuel_data.get("ncv_mj_per_l")
            if value is None:
                value = fuel_data.get("ncv_mj_per_m3")

        if value is None:
            raise ValueError(
                f"No {basis_key} heating value available for fuel type: '{fuel_type}'"
            )

        logger.debug("Heating value lookup: %s (%s) -> %s", key, basis_key, value)
        return value

    def get_biofuel_fraction(self, fuel_type: str) -> Decimal:
        """Return biofuel fraction of a fuel type.

        Args:
            fuel_type: Fuel type identifier.

        Returns:
            Biofuel fraction as Decimal (0.0 to 1.0).

        Raises:
            ValueError: If fuel_type is not recognized.
        """
        key = fuel_type.upper().strip()
        if key not in FUEL_TYPES:
            raise ValueError(f"Unknown fuel type: '{fuel_type}'")

        fraction = FUEL_TYPES[key]["biofuel_fraction"]
        logger.debug("Biofuel fraction lookup: %s -> %s", key, fraction)
        return fraction

    def get_distance_emission_factor(
        self, vehicle_type: str, fuel_type: str
    ) -> Decimal:
        """Return distance-based emission factor in g CO2e per km.

        Args:
            vehicle_type: Vehicle type identifier.
            fuel_type: Fuel type identifier.

        Returns:
            Emission factor in g CO2e/km as Decimal.

        Raises:
            ValueError: If no distance-based factor is available for this
                vehicle/fuel combination.
        """
        vtype = vehicle_type.upper().strip()
        ftype = fuel_type.upper().strip()

        if vtype not in VEHICLE_TYPES:
            raise ValueError(f"Unknown vehicle type: '{vehicle_type}'")
        if ftype not in FUEL_TYPES:
            raise ValueError(f"Unknown fuel type: '{fuel_type}'")

        veh_factors = _DISTANCE_EMISSION_FACTORS.get(vtype)
        if veh_factors is None:
            raise ValueError(
                f"No distance-based emission factors available for vehicle type: '{vtype}'. "
                f"Distance factors only available for on-road vehicles."
            )

        factor = veh_factors.get(ftype)
        if factor is None:
            raise ValueError(
                f"No distance-based emission factor for vehicle '{vtype}' with fuel '{ftype}'. "
                f"Available fuels: {sorted(veh_factors.keys())}"
            )

        logger.debug(
            "Distance emission factor lookup: %s/%s -> %s g CO2e/km",
            vtype, ftype, factor,
        )
        return factor

    def get_gwp(self, gas: str, source: str = "AR6") -> Decimal:
        """Return GWP value for a greenhouse gas from a specific assessment report.

        Args:
            gas: Greenhouse gas identifier (``"CO2"``, ``"CH4"``, ``"N2O"``).
            source: Assessment report (``"AR4"``, ``"AR5"``, ``"AR6"``, ``"AR6_20YR"``).

        Returns:
            GWP value as Decimal.

        Raises:
            ValueError: If gas or source is not recognized.
        """
        gas_key = gas.upper().strip()
        src_key = source.upper().strip()

        if src_key not in _GWP_VALUES:
            raise ValueError(
                f"Unknown GWP source: '{source}'. "
                f"Valid: {sorted(_GWP_VALUES.keys())}"
            )
        if gas_key not in _GWP_VALUES[src_key]:
            raise ValueError(
                f"Unknown gas: '{gas}'. "
                f"Valid: {sorted(_GWP_VALUES[src_key].keys())}"
            )

        value = _GWP_VALUES[src_key][gas_key]
        logger.debug("GWP lookup: %s (%s) -> %s", gas_key, src_key, value)
        return value

    def get_control_technology(self, tech_id: str) -> Dict[str, Any]:
        """Return details of an emission control technology.

        Args:
            tech_id: Control technology identifier
                (e.g. ``"THREE_WAY_CATALYST"``, ``"EURO_6"``).

        Returns:
            Dictionary with display_name, ch4_multiplier, n2o_multiplier,
            description, and source.

        Raises:
            ValueError: If tech_id is not recognized.
        """
        key = tech_id.upper().strip()
        if key not in CONTROL_TECHNOLOGIES:
            raise ValueError(
                f"Unknown control technology: '{tech_id}'. "
                f"Valid: {sorted(CONTROL_TECHNOLOGIES.keys())}"
            )

        result = dict(CONTROL_TECHNOLOGIES[key])
        result["technology_id"] = key
        logger.debug("Control technology lookup: %s", key)
        return result

    def list_control_technologies(self) -> List[Dict[str, Any]]:
        """List all available emission control technologies.

        Returns:
            List of control technology dictionaries.
        """
        results: List[Dict[str, Any]] = []
        for tech_key, tech_data in sorted(CONTROL_TECHNOLOGIES.items()):
            entry = dict(tech_data)
            entry["technology_id"] = tech_key
            results.append(entry)

        logger.debug("Listed %d control technologies", len(results))
        return results

    # ------------------------------------------------------------------
    # Public API: Search / Aggregate
    # ------------------------------------------------------------------

    def search_factors(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search emission factors with optional filters.

        Args:
            filters: Optional filter dictionary. Supported keys:
                - ``vehicle_type`` (str): Filter by vehicle type.
                - ``fuel_type`` (str): Filter by fuel type.
                - ``category`` (str): Filter by vehicle category.
                - ``gas`` (str): Filter by gas (CO2, CH4, N2O).
                - ``is_biofuel`` (bool): Filter biofuel/fossil fuels.
                - ``min_co2_ef`` (Decimal): Minimum CO2 EF threshold.
                - ``max_co2_ef`` (Decimal): Maximum CO2 EF threshold.

        Returns:
            List of matching emission factor records.
        """
        filters = filters or {}
        results: List[Dict[str, Any]] = []

        filter_vtype = filters.get("vehicle_type")
        filter_ftype = filters.get("fuel_type")
        filter_category = filters.get("category")
        filter_gas = filters.get("gas")
        filter_is_biofuel = filters.get("is_biofuel")
        filter_min_co2 = filters.get("min_co2_ef")
        filter_max_co2 = filters.get("max_co2_ef")

        # Normalize filters
        if filter_vtype:
            filter_vtype = filter_vtype.upper().strip()
        if filter_ftype:
            filter_ftype = filter_ftype.upper().strip()
        if filter_category:
            filter_category = filter_category.upper().strip()
        if filter_gas:
            filter_gas = filter_gas.upper().strip()

        for vtype_key, vtype_data in sorted(VEHICLE_TYPES.items()):
            if filter_vtype and vtype_key != filter_vtype:
                continue
            if filter_category and vtype_data["category"] != filter_category:
                continue

            default_fuel = vtype_data.get("default_fuel")
            fuels_to_check: List[str] = []

            if filter_ftype:
                fuels_to_check = [filter_ftype]
            elif default_fuel:
                fuels_to_check = [default_fuel]
            else:
                fuels_to_check = list(FUEL_TYPES.keys())

            for ftype_key in fuels_to_check:
                if ftype_key not in FUEL_TYPES:
                    continue
                fuel_data = FUEL_TYPES[ftype_key]

                if filter_is_biofuel is not None:
                    if fuel_data["is_biofuel"] != filter_is_biofuel:
                        continue

                co2_ef = fuel_data.get("co2_ef_kg_per_l") or fuel_data.get("co2_ef_kg_per_m3") or Decimal("0")
                if filter_min_co2 is not None and co2_ef < Decimal(str(filter_min_co2)):
                    continue
                if filter_max_co2 is not None and co2_ef > Decimal(str(filter_max_co2)):
                    continue

                gases_to_report = ["CO2", "CH4", "N2O"]
                if filter_gas:
                    gases_to_report = [filter_gas]

                for gas in gases_to_report:
                    if gas == "CO2":
                        value = co2_ef
                        unit = "kg CO2/L"
                    elif gas == "CH4":
                        value = fuel_data.get("ch4_ef_kg_per_tj", Decimal("0"))
                        unit = "kg CH4/TJ"
                    else:
                        value = fuel_data.get("n2o_ef_kg_per_tj", Decimal("0"))
                        unit = "kg N2O/TJ"

                    results.append({
                        "vehicle_type": vtype_key,
                        "fuel_type": ftype_key,
                        "gas": gas,
                        "value": value,
                        "unit": unit,
                        "category": vtype_data["category"],
                        "is_biofuel": fuel_data["is_biofuel"],
                    })

        logger.debug(
            "Search factors returned %d results (filters=%s)", len(results), filters
        )
        return results

    def get_factor_count(self) -> int:
        """Return total number of emission factors in the database.

        Counts all vehicle_type x fuel_type x gas combinations plus
        custom factors.

        Returns:
            Total factor count as integer.
        """
        builtin_count = len(VEHICLE_TYPES) * len(FUEL_TYPES) * 3  # CO2, CH4, N2O
        custom_count = len(self._custom_factors)
        return builtin_count + custom_count

    # ------------------------------------------------------------------
    # Public API: Custom Factor Registration
    # ------------------------------------------------------------------

    def register_custom_factor(
        self,
        factor_id: str,
        vehicle_type: str,
        fuel_type: str,
        gas: str,
        value: Decimal,
        unit: str,
        source: str = "CUSTOM",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Register a custom emission factor.

        Args:
            factor_id: Unique identifier for the custom factor.
            vehicle_type: Vehicle type this factor applies to.
            fuel_type: Fuel type this factor applies to.
            gas: Greenhouse gas (CO2, CH4, N2O).
            value: Factor value as Decimal.
            unit: Unit string (e.g. ``"kg CO2/L"``).
            source: Source authority string.
            metadata: Optional additional metadata.

        Returns:
            The factor_id as confirmation.

        Raises:
            ValueError: If vehicle_type, fuel_type, or gas is invalid.
        """
        vtype = vehicle_type.upper().strip()
        ftype = fuel_type.upper().strip()
        gas_key = gas.upper().strip()

        if vtype not in VEHICLE_TYPES:
            raise ValueError(f"Unknown vehicle type: '{vehicle_type}'")
        if ftype not in FUEL_TYPES:
            raise ValueError(f"Unknown fuel type: '{fuel_type}'")
        if gas_key not in ("CO2", "CH4", "N2O"):
            raise ValueError(f"Unknown gas: '{gas}'")

        with self._lock:
            self._custom_factors[factor_id] = {
                "vehicle_type": vtype,
                "fuel_type": ftype,
                "gas": gas_key,
                "value": value,
                "unit": unit,
                "source": source,
                "metadata": metadata or {},
                "registered_at": _utcnow().isoformat(),
            }

        logger.info(
            "Custom factor registered: %s (%s/%s/%s = %s %s)",
            factor_id, vtype, ftype, gas_key, value, unit,
        )
        return factor_id

    def get_custom_factor(self, factor_id: str) -> Dict[str, Any]:
        """Retrieve a registered custom factor.

        Args:
            factor_id: The custom factor identifier.

        Returns:
            Custom factor dictionary.

        Raises:
            ValueError: If factor_id is not found.
        """
        with self._lock:
            if factor_id not in self._custom_factors:
                raise ValueError(f"Custom factor not found: '{factor_id}'")
            return dict(self._custom_factors[factor_id])

    def list_custom_factors(self) -> List[Dict[str, Any]]:
        """List all registered custom factors.

        Returns:
            List of custom factor dictionaries.
        """
        with self._lock:
            results = []
            for fid, fdata in sorted(self._custom_factors.items()):
                entry = dict(fdata)
                entry["factor_id"] = fid
                results.append(entry)
            return results

    def delete_custom_factor(self, factor_id: str) -> bool:
        """Delete a custom emission factor.

        Args:
            factor_id: The custom factor identifier.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            if factor_id in self._custom_factors:
                del self._custom_factors[factor_id]
                logger.info("Custom factor deleted: %s", factor_id)
                return True
            return False

    # ------------------------------------------------------------------
    # Public API: Vehicle Category Info
    # ------------------------------------------------------------------

    def get_vehicle_category(self, category: str) -> Dict[str, Any]:
        """Return information about a vehicle category.

        Args:
            category: Category identifier (e.g. ``"ON_ROAD"``).

        Returns:
            Category dictionary with display_name, description, source.

        Raises:
            ValueError: If category is not recognized.
        """
        key = category.upper().strip()
        if key not in VEHICLE_CATEGORIES:
            raise ValueError(
                f"Unknown vehicle category: '{category}'. "
                f"Valid: {sorted(VEHICLE_CATEGORIES.keys())}"
            )
        result = dict(VEHICLE_CATEGORIES[key])
        result["category"] = key
        return result

    def list_vehicle_categories(self) -> List[Dict[str, Any]]:
        """List all vehicle categories.

        Returns:
            List of category dictionaries.
        """
        results: List[Dict[str, Any]] = []
        for cat_key, cat_data in sorted(VEHICLE_CATEGORIES.items()):
            entry = dict(cat_data)
            entry["category"] = cat_key
            entry["vehicle_count"] = sum(
                1 for v in VEHICLE_TYPES.values() if v["category"] == cat_key
            )
            results.append(entry)
        return results

    # ------------------------------------------------------------------
    # Internal: Model Year Resolution
    # ------------------------------------------------------------------

    def _resolve_model_year_range(self, model_year: Optional[int]) -> str:
        """Map a model year to the EPA model year range key.

        Args:
            model_year: Numeric model year. If None, defaults to 2006+.

        Returns:
            Year range key string.
        """
        if model_year is None:
            return "2006_PLUS"
        if model_year < 1985:
            return "PRE_1985"
        if model_year <= 1995:
            return "1985_1995"
        if model_year <= 2005:
            return "1996_2005"
        return "2006_PLUS"

    # ------------------------------------------------------------------
    # Internal: Provenance Hash
    # ------------------------------------------------------------------

    def _compute_hash(self, operation: str, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for provenance tracking.

        Args:
            operation: Operation name for the hash context.
            data: Data dictionary to include in the hash.

        Returns:
            Hexadecimal SHA-256 hash string.
        """
        hash_input = json.dumps(
            {"operation": operation, "data": data, "timestamp": _utcnow().isoformat()},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Internal: Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return database statistics summary.

        Returns:
            Dictionary with counts of vehicle types, fuel types, control
            technologies, custom factors, and total emission factors.
        """
        return {
            "vehicle_types": len(VEHICLE_TYPES),
            "vehicle_categories": len(VEHICLE_CATEGORIES),
            "fuel_types": len(FUEL_TYPES),
            "control_technologies": len(CONTROL_TECHNOLOGIES),
            "custom_factors": len(self._custom_factors),
            "total_emission_factors": self.get_factor_count(),
            "gwp_sources": sorted(_GWP_VALUES.keys()),
        }
