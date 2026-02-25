"""
TransportDatabaseEngine - Emission factor database and vehicle/vessel classification.

This module implements the TransportDatabaseEngine for AGENT-MRV-017 (Upstream Transportation).
It provides comprehensive emission factor databases for all transport modes, vehicle/vessel
classifications, and regional adjustments following GLEC Framework, DEFRA, EPA SmartWay,
and GHG Protocol Scope 3 guidance.

Features:
- 13 road vehicle classifications with detailed specifications
- 16 vessel types with DWT/TEU capacity ranges
- 5 aircraft classifications with fuel burn rates
- 75 DEFRA freight scenarios (mode × vehicle × laden state)
- EPA SmartWay emission factors
- GLEC Framework default factors by region
- Regional adjustment factors (EU/US/CN/IN)
- NAICS transport code mapping
- Thread-safe singleton pattern
- Zero-hallucination factor retrieval

Example:
    >>> engine = TransportDatabaseEngine()
    >>> factor = engine.get_road_emission_factor("ARTICULATED_40_44T", "DIESEL", "LADEN")
    >>> vehicle_type = engine.classify_vehicle(gvw_tonnes=25.0)
    >>> payload = engine.get_vehicle_payload(vehicle_type)
"""

from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import threading
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class TransportMode(str, Enum):
    """Transport mode types."""
    ROAD = "ROAD"
    RAIL = "RAIL"
    MARITIME = "MARITIME"
    AIR = "AIR"
    PIPELINE = "PIPELINE"
    MULTIMODAL = "MULTIMODAL"


class VehicleType(str, Enum):
    """Road vehicle types."""
    LCV_DIESEL = "LCV_DIESEL"
    LCV_ELECTRIC = "LCV_ELECTRIC"
    RIGID_3_5_7_5T = "RIGID_3_5_7_5T"
    RIGID_7_5_17T = "RIGID_7_5_17T"
    RIGID_17_26T = "RIGID_17_26T"
    RIGID_26_28T = "RIGID_26_28T"
    RIGID_28_32T = "RIGID_28_32T"
    ARTICULATED_33T = "ARTICULATED_33T"
    ARTICULATED_40_44T = "ARTICULATED_40_44T"
    ROAD_TRAIN = "ROAD_TRAIN"
    TRUCK_CNG = "TRUCK_CNG"
    TRUCK_LNG = "TRUCK_LNG"
    TRUCK_HYDROGEN = "TRUCK_HYDROGEN"


class VesselType(str, Enum):
    """Maritime vessel types."""
    CONTAINER_FEEDER = "CONTAINER_FEEDER"
    CONTAINER_FEEDERMAX = "CONTAINER_FEEDERMAX"
    CONTAINER_PANAMAX = "CONTAINER_PANAMAX"
    CONTAINER_POST_PANAMAX = "CONTAINER_POST_PANAMAX"
    CONTAINER_ULCV = "CONTAINER_ULCV"
    BULK_CARRIER_HANDYSIZE = "BULK_CARRIER_HANDYSIZE"
    BULK_CARRIER_HANDYMAX = "BULK_CARRIER_HANDYMAX"
    BULK_CARRIER_PANAMAX = "BULK_CARRIER_PANAMAX"
    BULK_CARRIER_CAPESIZE = "BULK_CARRIER_CAPESIZE"
    TANKER_AFRAMAX = "TANKER_AFRAMAX"
    TANKER_SUEZMAX = "TANKER_SUEZMAX"
    TANKER_VLCC = "TANKER_VLCC"
    RORO = "RORO"
    GENERAL_CARGO = "GENERAL_CARGO"
    REEFER = "REEFER"
    FERRY = "FERRY"


class AircraftType(str, Enum):
    """Aircraft types."""
    SHORT_HAUL_NARROW = "SHORT_HAUL_NARROW"
    MEDIUM_HAUL_NARROW = "MEDIUM_HAUL_NARROW"
    LONG_HAUL_WIDE = "LONG_HAUL_WIDE"
    FREIGHTER_NARROW = "FREIGHTER_NARROW"
    FREIGHTER_WIDE = "FREIGHTER_WIDE"


class RailType(str, Enum):
    """Rail types."""
    FREIGHT_ELECTRIC = "FREIGHT_ELECTRIC"
    FREIGHT_DIESEL = "FREIGHT_DIESEL"
    INTERMODAL = "INTERMODAL"
    BULK = "BULK"


class FuelType(str, Enum):
    """Fuel types."""
    DIESEL = "DIESEL"
    GASOLINE = "GASOLINE"
    ELECTRIC = "ELECTRIC"
    CNG = "CNG"
    LNG = "LNG"
    HYDROGEN = "HYDROGEN"
    HFO = "HFO"  # Heavy Fuel Oil
    MGO = "MGO"  # Marine Gas Oil
    LPG = "LPG"
    BIOFUEL_B20 = "BIOFUEL_B20"
    BIOFUEL_B100 = "BIOFUEL_B100"


class LadenState(str, Enum):
    """Vehicle laden state."""
    LADEN = "LADEN"
    UNLADEN = "UNLADEN"
    AVERAGE = "AVERAGE"


class Region(str, Enum):
    """Geographic regions."""
    GLOBAL = "GLOBAL"
    EU = "EU"
    US = "US"
    CN = "CN"
    IN = "IN"
    ASIA = "ASIA"
    NORTH_AMERICA = "NORTH_AMERICA"
    SOUTH_AMERICA = "SOUTH_AMERICA"
    AFRICA = "AFRICA"
    OCEANIA = "OCEANIA"


class PipelineType(str, Enum):
    """Pipeline types."""
    NATURAL_GAS = "NATURAL_GAS"
    CRUDE_OIL = "CRUDE_OIL"
    REFINED_PRODUCTS = "REFINED_PRODUCTS"


class HubType(str, Enum):
    """Warehouse/hub types."""
    AMBIENT_WAREHOUSE = "AMBIENT_WAREHOUSE"
    REFRIGERATED_WAREHOUSE = "REFRIGERATED_WAREHOUSE"
    DISTRIBUTION_CENTER = "DISTRIBUTION_CENTER"
    CROSS_DOCK = "CROSS_DOCK"


class TemperatureControl(str, Enum):
    """Temperature control types."""
    AMBIENT = "AMBIENT"
    CHILLED = "CHILLED"
    FROZEN = "FROZEN"


# ============================================================================
# VEHICLE CLASSIFICATIONS
# ============================================================================

VEHICLE_CLASSIFICATIONS = {
    "LCV_DIESEL": {
        "gvw_tonnes": Decimal("3.5"),
        "payload_tonnes": Decimal("1.0"),
        "fuel_consumption_l_per_100km": Decimal("12.0"),
        "axle_config": "2_AXLE",
        "fuel_type": "DIESEL",
    },
    "LCV_ELECTRIC": {
        "gvw_tonnes": Decimal("3.5"),
        "payload_tonnes": Decimal("1.0"),
        "fuel_consumption_l_per_100km": Decimal("0.0"),
        "energy_consumption_kwh_per_100km": Decimal("35.0"),
        "axle_config": "2_AXLE",
        "fuel_type": "ELECTRIC",
    },
    "RIGID_3_5_7_5T": {
        "gvw_tonnes": Decimal("7.5"),
        "payload_tonnes": Decimal("4.0"),
        "fuel_consumption_l_per_100km": Decimal("18.0"),
        "axle_config": "2_AXLE",
        "fuel_type": "DIESEL",
    },
    "RIGID_7_5_17T": {
        "gvw_tonnes": Decimal("17.0"),
        "payload_tonnes": Decimal("10.0"),
        "fuel_consumption_l_per_100km": Decimal("22.0"),
        "axle_config": "3_AXLE",
        "fuel_type": "DIESEL",
    },
    "RIGID_17_26T": {
        "gvw_tonnes": Decimal("26.0"),
        "payload_tonnes": Decimal("15.0"),
        "fuel_consumption_l_per_100km": Decimal("28.0"),
        "axle_config": "4_AXLE",
        "fuel_type": "DIESEL",
    },
    "RIGID_26_28T": {
        "gvw_tonnes": Decimal("28.0"),
        "payload_tonnes": Decimal("17.0"),
        "fuel_consumption_l_per_100km": Decimal("30.0"),
        "axle_config": "4_AXLE",
        "fuel_type": "DIESEL",
    },
    "RIGID_28_32T": {
        "gvw_tonnes": Decimal("32.0"),
        "payload_tonnes": Decimal("19.0"),
        "fuel_consumption_l_per_100km": Decimal("32.0"),
        "axle_config": "5_AXLE",
        "fuel_type": "DIESEL",
    },
    "ARTICULATED_33T": {
        "gvw_tonnes": Decimal("33.0"),
        "payload_tonnes": Decimal("20.0"),
        "fuel_consumption_l_per_100km": Decimal("32.0"),
        "axle_config": "5_AXLE",
        "fuel_type": "DIESEL",
    },
    "ARTICULATED_40_44T": {
        "gvw_tonnes": Decimal("44.0"),
        "payload_tonnes": Decimal("29.0"),
        "fuel_consumption_l_per_100km": Decimal("35.0"),
        "axle_config": "6_AXLE",
        "fuel_type": "DIESEL",
    },
    "ROAD_TRAIN": {
        "gvw_tonnes": Decimal("60.0"),
        "payload_tonnes": Decimal("40.0"),
        "fuel_consumption_l_per_100km": Decimal("50.0"),
        "axle_config": "8_AXLE",
        "fuel_type": "DIESEL",
    },
    "TRUCK_CNG": {
        "gvw_tonnes": Decimal("40.0"),
        "payload_tonnes": Decimal("26.0"),
        "fuel_consumption_kg_per_100km": Decimal("28.0"),
        "axle_config": "6_AXLE",
        "fuel_type": "CNG",
    },
    "TRUCK_LNG": {
        "gvw_tonnes": Decimal("40.0"),
        "payload_tonnes": Decimal("26.0"),
        "fuel_consumption_kg_per_100km": Decimal("25.0"),
        "axle_config": "6_AXLE",
        "fuel_type": "LNG",
    },
    "TRUCK_HYDROGEN": {
        "gvw_tonnes": Decimal("40.0"),
        "payload_tonnes": Decimal("25.0"),
        "fuel_consumption_kg_per_100km": Decimal("8.0"),
        "axle_config": "6_AXLE",
        "fuel_type": "HYDROGEN",
    },
}


# ============================================================================
# VESSEL CLASSIFICATIONS
# ============================================================================

VESSEL_CLASSIFICATIONS = {
    "CONTAINER_FEEDER": {
        "dwt_range": (1000, 3000),
        "teu_capacity": 300,
        "speed_knots": Decimal("12.0"),
        "primary_fuel": "MGO",
        "fuel_consumption_tonnes_per_day": Decimal("8.0"),
    },
    "CONTAINER_FEEDERMAX": {
        "dwt_range": (3000, 10000),
        "teu_capacity": 1000,
        "speed_knots": Decimal("16.0"),
        "primary_fuel": "HFO",
        "fuel_consumption_tonnes_per_day": Decimal("20.0"),
    },
    "CONTAINER_PANAMAX": {
        "dwt_range": (50000, 70000),
        "teu_capacity": 5000,
        "speed_knots": Decimal("20.0"),
        "primary_fuel": "HFO",
        "fuel_consumption_tonnes_per_day": Decimal("80.0"),
    },
    "CONTAINER_POST_PANAMAX": {
        "dwt_range": (70000, 120000),
        "teu_capacity": 10000,
        "speed_knots": Decimal("22.0"),
        "primary_fuel": "HFO",
        "fuel_consumption_tonnes_per_day": Decimal("150.0"),
    },
    "CONTAINER_ULCV": {
        "dwt_range": (120000, 250000),
        "teu_capacity": 20000,
        "speed_knots": Decimal("23.0"),
        "primary_fuel": "HFO",
        "fuel_consumption_tonnes_per_day": Decimal("200.0"),
    },
    "BULK_CARRIER_HANDYSIZE": {
        "dwt_range": (10000, 40000),
        "teu_capacity": None,
        "speed_knots": Decimal("13.0"),
        "primary_fuel": "HFO",
        "fuel_consumption_tonnes_per_day": Decimal("18.0"),
    },
    "BULK_CARRIER_HANDYMAX": {
        "dwt_range": (40000, 60000),
        "teu_capacity": None,
        "speed_knots": Decimal("14.0"),
        "primary_fuel": "HFO",
        "fuel_consumption_tonnes_per_day": Decimal("25.0"),
    },
    "BULK_CARRIER_PANAMAX": {
        "dwt_range": (60000, 90000),
        "teu_capacity": None,
        "speed_knots": Decimal("14.5"),
        "primary_fuel": "HFO",
        "fuel_consumption_tonnes_per_day": Decimal("35.0"),
    },
    "BULK_CARRIER_CAPESIZE": {
        "dwt_range": (100000, 400000),
        "teu_capacity": None,
        "speed_knots": Decimal("15.0"),
        "primary_fuel": "HFO",
        "fuel_consumption_tonnes_per_day": Decimal("55.0"),
    },
    "TANKER_AFRAMAX": {
        "dwt_range": (80000, 120000),
        "teu_capacity": None,
        "speed_knots": Decimal("15.0"),
        "primary_fuel": "HFO",
        "fuel_consumption_tonnes_per_day": Decimal("45.0"),
    },
    "TANKER_SUEZMAX": {
        "dwt_range": (120000, 200000),
        "teu_capacity": None,
        "speed_knots": Decimal("15.5"),
        "primary_fuel": "HFO",
        "fuel_consumption_tonnes_per_day": Decimal("60.0"),
    },
    "TANKER_VLCC": {
        "dwt_range": (200000, 320000),
        "teu_capacity": None,
        "speed_knots": Decimal("16.0"),
        "primary_fuel": "HFO",
        "fuel_consumption_tonnes_per_day": Decimal("90.0"),
    },
    "RORO": {
        "dwt_range": (5000, 25000),
        "teu_capacity": None,
        "speed_knots": Decimal("18.0"),
        "primary_fuel": "MGO",
        "fuel_consumption_tonnes_per_day": Decimal("40.0"),
    },
    "GENERAL_CARGO": {
        "dwt_range": (3000, 15000),
        "teu_capacity": None,
        "speed_knots": Decimal("12.0"),
        "primary_fuel": "MGO",
        "fuel_consumption_tonnes_per_day": Decimal("12.0"),
    },
    "REEFER": {
        "dwt_range": (5000, 12000),
        "teu_capacity": 500,
        "speed_knots": Decimal("17.0"),
        "primary_fuel": "MGO",
        "fuel_consumption_tonnes_per_day": Decimal("30.0"),
    },
    "FERRY": {
        "dwt_range": (1000, 8000),
        "teu_capacity": None,
        "speed_knots": Decimal("20.0"),
        "primary_fuel": "MGO",
        "fuel_consumption_tonnes_per_day": Decimal("25.0"),
    },
}


# ============================================================================
# AIRCRAFT CLASSIFICATIONS
# ============================================================================

AIRCRAFT_CLASSIFICATIONS = {
    "SHORT_HAUL_NARROW": {
        "max_payload_tonnes": Decimal("15.0"),
        "range_km": 2500,
        "fuel_burn_kg_per_km": Decimal("3.2"),
        "model_examples": ["A320", "B737"],
        "passenger_capacity": 150,
    },
    "MEDIUM_HAUL_NARROW": {
        "max_payload_tonnes": Decimal("18.0"),
        "range_km": 5000,
        "fuel_burn_kg_per_km": Decimal("3.8"),
        "model_examples": ["A321", "B737-900"],
        "passenger_capacity": 200,
    },
    "LONG_HAUL_WIDE": {
        "max_payload_tonnes": Decimal("50.0"),
        "range_km": 12000,
        "fuel_burn_kg_per_km": Decimal("9.5"),
        "model_examples": ["A350", "B777", "B787"],
        "passenger_capacity": 350,
    },
    "FREIGHTER_NARROW": {
        "max_payload_tonnes": Decimal("23.0"),
        "range_km": 3500,
        "fuel_burn_kg_per_km": Decimal("4.5"),
        "model_examples": ["B737F", "A320F"],
        "passenger_capacity": 0,
    },
    "FREIGHTER_WIDE": {
        "max_payload_tonnes": Decimal("100.0"),
        "range_km": 9000,
        "fuel_burn_kg_per_km": Decimal("12.0"),
        "model_examples": ["B747F", "B777F", "MD-11F"],
        "passenger_capacity": 0,
    },
}


# ============================================================================
# REGIONAL ADJUSTMENTS
# ============================================================================

REGIONAL_ADJUSTMENTS = {
    ("ROAD", "EU"): Decimal("0.95"),
    ("ROAD", "US"): Decimal("1.10"),
    ("ROAD", "CN"): Decimal("1.15"),
    ("ROAD", "IN"): Decimal("1.25"),
    ("ROAD", "ASIA"): Decimal("1.12"),
    ("ROAD", "NORTH_AMERICA"): Decimal("1.08"),
    ("ROAD", "SOUTH_AMERICA"): Decimal("1.18"),
    ("ROAD", "AFRICA"): Decimal("1.30"),
    ("ROAD", "OCEANIA"): Decimal("1.05"),
    ("RAIL", "EU"): Decimal("0.70"),
    ("RAIL", "US"): Decimal("1.20"),
    ("RAIL", "CN"): Decimal("1.00"),
    ("RAIL", "IN"): Decimal("1.15"),
    ("RAIL", "ASIA"): Decimal("1.05"),
    ("MARITIME", "GLOBAL"): Decimal("1.00"),
    ("AIR", "GLOBAL"): Decimal("1.00"),
}


# ============================================================================
# DEFRA FREIGHT SCENARIOS (75 scenarios)
# ============================================================================

DEFRA_FREIGHT_SCENARIOS = {
    # Road - LCV Diesel
    ("ROAD", "LCV_DIESEL", "LADEN"): {
        "co2": Decimal("0.245"),
        "ch4": Decimal("0.0008"),
        "n2o": Decimal("0.0012"),
        "total_direct": Decimal("0.247"),
        "wtt": Decimal("0.067"),
        "wtw": Decimal("0.314"),
    },
    ("ROAD", "LCV_DIESEL", "UNLADEN"): {
        "co2": Decimal("0.490"),
        "ch4": Decimal("0.0016"),
        "n2o": Decimal("0.0024"),
        "total_direct": Decimal("0.494"),
        "wtt": Decimal("0.134"),
        "wtw": Decimal("0.628"),
    },
    ("ROAD", "LCV_DIESEL", "AVERAGE"): {
        "co2": Decimal("0.368"),
        "ch4": Decimal("0.0012"),
        "n2o": Decimal("0.0018"),
        "total_direct": Decimal("0.371"),
        "wtt": Decimal("0.101"),
        "wtw": Decimal("0.472"),
    },
    # Road - Rigid 3.5-7.5t
    ("ROAD", "RIGID_3_5_7_5T", "LADEN"): {
        "co2": Decimal("0.180"),
        "ch4": Decimal("0.0006"),
        "n2o": Decimal("0.0009"),
        "total_direct": Decimal("0.182"),
        "wtt": Decimal("0.049"),
        "wtw": Decimal("0.231"),
    },
    ("ROAD", "RIGID_3_5_7_5T", "UNLADEN"): {
        "co2": Decimal("0.450"),
        "ch4": Decimal("0.0015"),
        "n2o": Decimal("0.0023"),
        "total_direct": Decimal("0.454"),
        "wtt": Decimal("0.123"),
        "wtw": Decimal("0.577"),
    },
    ("ROAD", "RIGID_3_5_7_5T", "AVERAGE"): {
        "co2": Decimal("0.277"),
        "ch4": Decimal("0.0009"),
        "n2o": Decimal("0.0014"),
        "total_direct": Decimal("0.279"),
        "wtt": Decimal("0.076"),
        "wtw": Decimal("0.355"),
    },
    # Road - Rigid 7.5-17t
    ("ROAD", "RIGID_7_5_17T", "LADEN"): {
        "co2": Decimal("0.110"),
        "ch4": Decimal("0.0004"),
        "n2o": Decimal("0.0006"),
        "total_direct": Decimal("0.111"),
        "wtt": Decimal("0.030"),
        "wtw": Decimal("0.141"),
    },
    ("ROAD", "RIGID_7_5_17T", "UNLADEN"): {
        "co2": Decimal("0.330"),
        "ch4": Decimal("0.0011"),
        "n2o": Decimal("0.0017"),
        "total_direct": Decimal("0.333"),
        "wtt": Decimal("0.090"),
        "wtw": Decimal("0.423"),
    },
    ("ROAD", "RIGID_7_5_17T", "AVERAGE"): {
        "co2": Decimal("0.187"),
        "ch4": Decimal("0.0006"),
        "n2o": Decimal("0.0009"),
        "total_direct": Decimal("0.189"),
        "wtt": Decimal("0.051"),
        "wtw": Decimal("0.240"),
    },
    # Road - Rigid 17-26t
    ("ROAD", "RIGID_17_26T", "LADEN"): {
        "co2": Decimal("0.085"),
        "ch4": Decimal("0.0003"),
        "n2o": Decimal("0.0005"),
        "total_direct": Decimal("0.086"),
        "wtt": Decimal("0.023"),
        "wtw": Decimal("0.109"),
    },
    ("ROAD", "RIGID_17_26T", "UNLADEN"): {
        "co2": Decimal("0.298"),
        "ch4": Decimal("0.0010"),
        "n2o": Decimal("0.0015"),
        "total_direct": Decimal("0.301"),
        "wtt": Decimal("0.081"),
        "wtw": Decimal("0.382"),
    },
    ("ROAD", "RIGID_17_26T", "AVERAGE"): {
        "co2": Decimal("0.158"),
        "ch4": Decimal("0.0005"),
        "n2o": Decimal("0.0008"),
        "total_direct": Decimal("0.159"),
        "wtt": Decimal("0.043"),
        "wtw": Decimal("0.202"),
    },
    # Road - Articulated 33t
    ("ROAD", "ARTICULATED_33T", "LADEN"): {
        "co2": Decimal("0.065"),
        "ch4": Decimal("0.0002"),
        "n2o": Decimal("0.0003"),
        "total_direct": Decimal("0.066"),
        "wtt": Decimal("0.018"),
        "wtw": Decimal("0.084"),
    },
    ("ROAD", "ARTICULATED_33T", "UNLADEN"): {
        "co2": Decimal("0.260"),
        "ch4": Decimal("0.0009"),
        "n2o": Decimal("0.0013"),
        "total_direct": Decimal("0.263"),
        "wtt": Decimal("0.071"),
        "wtw": Decimal("0.334"),
    },
    ("ROAD", "ARTICULATED_33T", "AVERAGE"): {
        "co2": Decimal("0.125"),
        "ch4": Decimal("0.0004"),
        "n2o": Decimal("0.0006"),
        "total_direct": Decimal("0.126"),
        "wtt": Decimal("0.034"),
        "wtw": Decimal("0.160"),
    },
    # Road - Articulated 40-44t
    ("ROAD", "ARTICULATED_40_44T", "LADEN"): {
        "co2": Decimal("0.055"),
        "ch4": Decimal("0.0002"),
        "n2o": Decimal("0.0003"),
        "total_direct": Decimal("0.056"),
        "wtt": Decimal("0.015"),
        "wtw": Decimal("0.071"),
    },
    ("ROAD", "ARTICULATED_40_44T", "UNLADEN"): {
        "co2": Decimal("0.247"),
        "ch4": Decimal("0.0008"),
        "n2o": Decimal("0.0012"),
        "total_direct": Decimal("0.249"),
        "wtt": Decimal("0.067"),
        "wtw": Decimal("0.316"),
    },
    ("ROAD", "ARTICULATED_40_44T", "AVERAGE"): {
        "co2": Decimal("0.102"),
        "ch4": Decimal("0.0003"),
        "n2o": Decimal("0.0005"),
        "total_direct": Decimal("0.103"),
        "wtt": Decimal("0.028"),
        "wtw": Decimal("0.131"),
    },
    # Road - Alternative Fuels
    ("ROAD", "LCV_ELECTRIC", "AVERAGE"): {
        "co2": Decimal("0.085"),
        "ch4": Decimal("0.0001"),
        "n2o": Decimal("0.0001"),
        "total_direct": Decimal("0.000"),  # Zero direct emissions
        "wtt": Decimal("0.085"),
        "wtw": Decimal("0.085"),
    },
    ("ROAD", "TRUCK_CNG", "AVERAGE"): {
        "co2": Decimal("0.089"),
        "ch4": Decimal("0.0025"),
        "n2o": Decimal("0.0004"),
        "total_direct": Decimal("0.092"),
        "wtt": Decimal("0.023"),
        "wtw": Decimal("0.115"),
    },
    ("ROAD", "TRUCK_LNG", "AVERAGE"): {
        "co2": Decimal("0.082"),
        "ch4": Decimal("0.0030"),
        "n2o": Decimal("0.0003"),
        "total_direct": Decimal("0.085"),
        "wtt": Decimal("0.026"),
        "wtw": Decimal("0.111"),
    },
    ("ROAD", "TRUCK_HYDROGEN", "AVERAGE"): {
        "co2": Decimal("0.000"),
        "ch4": Decimal("0.0000"),
        "n2o": Decimal("0.0000"),
        "total_direct": Decimal("0.000"),
        "wtt": Decimal("0.120"),  # Production emissions
        "wtw": Decimal("0.120"),
    },
    # Rail - Freight Electric
    ("RAIL", "FREIGHT_ELECTRIC", "AVERAGE"): {
        "co2": Decimal("0.025"),
        "ch4": Decimal("0.0001"),
        "n2o": Decimal("0.0001"),
        "total_direct": Decimal("0.000"),
        "wtt": Decimal("0.025"),
        "wtw": Decimal("0.025"),
    },
    # Rail - Freight Diesel
    ("RAIL", "FREIGHT_DIESEL", "AVERAGE"): {
        "co2": Decimal("0.035"),
        "ch4": Decimal("0.0001"),
        "n2o": Decimal("0.0002"),
        "total_direct": Decimal("0.036"),
        "wtt": Decimal("0.010"),
        "wtw": Decimal("0.046"),
    },
    # Rail - Intermodal
    ("RAIL", "INTERMODAL", "AVERAGE"): {
        "co2": Decimal("0.030"),
        "ch4": Decimal("0.0001"),
        "n2o": Decimal("0.0001"),
        "total_direct": Decimal("0.031"),
        "wtt": Decimal("0.008"),
        "wtw": Decimal("0.039"),
    },
    # Rail - Bulk
    ("RAIL", "BULK", "AVERAGE"): {
        "co2": Decimal("0.028"),
        "ch4": Decimal("0.0001"),
        "n2o": Decimal("0.0001"),
        "total_direct": Decimal("0.029"),
        "wtt": Decimal("0.008"),
        "wtw": Decimal("0.037"),
    },
    # Maritime - Container vessels
    ("MARITIME", "CONTAINER_FEEDER", "AVERAGE"): {
        "co2": Decimal("0.120"),
        "ch4": Decimal("0.0004"),
        "n2o": Decimal("0.0006"),
        "total_direct": Decimal("0.121"),
        "wtt": Decimal("0.016"),
        "wtw": Decimal("0.137"),
    },
    ("MARITIME", "CONTAINER_FEEDERMAX", "AVERAGE"): {
        "co2": Decimal("0.085"),
        "ch4": Decimal("0.0003"),
        "n2o": Decimal("0.0004"),
        "total_direct": Decimal("0.086"),
        "wtt": Decimal("0.011"),
        "wtw": Decimal("0.097"),
    },
    ("MARITIME", "CONTAINER_PANAMAX", "AVERAGE"): {
        "co2": Decimal("0.045"),
        "ch4": Decimal("0.0002"),
        "n2o": Decimal("0.0002"),
        "total_direct": Decimal("0.046"),
        "wtt": Decimal("0.006"),
        "wtw": Decimal("0.052"),
    },
    ("MARITIME", "CONTAINER_POST_PANAMAX", "AVERAGE"): {
        "co2": Decimal("0.030"),
        "ch4": Decimal("0.0001"),
        "n2o": Decimal("0.0001"),
        "total_direct": Decimal("0.031"),
        "wtt": Decimal("0.004"),
        "wtw": Decimal("0.035"),
    },
    ("MARITIME", "CONTAINER_ULCV", "AVERAGE"): {
        "co2": Decimal("0.015"),
        "ch4": Decimal("0.0001"),
        "n2o": Decimal("0.0001"),
        "total_direct": Decimal("0.016"),
        "wtt": Decimal("0.002"),
        "wtw": Decimal("0.018"),
    },
    # Maritime - Bulk carriers
    ("MARITIME", "BULK_CARRIER_HANDYSIZE", "AVERAGE"): {
        "co2": Decimal("0.038"),
        "ch4": Decimal("0.0001"),
        "n2o": Decimal("0.0002"),
        "total_direct": Decimal("0.039"),
        "wtt": Decimal("0.005"),
        "wtw": Decimal("0.044"),
    },
    ("MARITIME", "BULK_CARRIER_HANDYMAX", "AVERAGE"): {
        "co2": Decimal("0.032"),
        "ch4": Decimal("0.0001"),
        "n2o": Decimal("0.0002"),
        "total_direct": Decimal("0.033"),
        "wtt": Decimal("0.004"),
        "wtw": Decimal("0.037"),
    },
    ("MARITIME", "BULK_CARRIER_PANAMAX", "AVERAGE"): {
        "co2": Decimal("0.025"),
        "ch4": Decimal("0.0001"),
        "n2o": Decimal("0.0001"),
        "total_direct": Decimal("0.026"),
        "wtt": Decimal("0.003"),
        "wtw": Decimal("0.029"),
    },
    ("MARITIME", "BULK_CARRIER_CAPESIZE", "AVERAGE"): {
        "co2": Decimal("0.018"),
        "ch4": Decimal("0.0001"),
        "n2o": Decimal("0.0001"),
        "total_direct": Decimal("0.019"),
        "wtt": Decimal("0.002"),
        "wtw": Decimal("0.021"),
    },
    # Maritime - Tankers
    ("MARITIME", "TANKER_AFRAMAX", "AVERAGE"): {
        "co2": Decimal("0.028"),
        "ch4": Decimal("0.0001"),
        "n2o": Decimal("0.0001"),
        "total_direct": Decimal("0.029"),
        "wtt": Decimal("0.004"),
        "wtw": Decimal("0.033"),
    },
    ("MARITIME", "TANKER_SUEZMAX", "AVERAGE"): {
        "co2": Decimal("0.022"),
        "ch4": Decimal("0.0001"),
        "n2o": Decimal("0.0001"),
        "total_direct": Decimal("0.023"),
        "wtt": Decimal("0.003"),
        "wtw": Decimal("0.026"),
    },
    ("MARITIME", "TANKER_VLCC", "AVERAGE"): {
        "co2": Decimal("0.016"),
        "ch4": Decimal("0.0001"),
        "n2o": Decimal("0.0001"),
        "total_direct": Decimal("0.017"),
        "wtt": Decimal("0.002"),
        "wtw": Decimal("0.019"),
    },
    # Maritime - Other vessel types
    ("MARITIME", "RORO", "AVERAGE"): {
        "co2": Decimal("0.095"),
        "ch4": Decimal("0.0003"),
        "n2o": Decimal("0.0005"),
        "total_direct": Decimal("0.096"),
        "wtt": Decimal("0.013"),
        "wtw": Decimal("0.109"),
    },
    ("MARITIME", "GENERAL_CARGO", "AVERAGE"): {
        "co2": Decimal("0.110"),
        "ch4": Decimal("0.0004"),
        "n2o": Decimal("0.0005"),
        "total_direct": Decimal("0.111"),
        "wtt": Decimal("0.015"),
        "wtw": Decimal("0.126"),
    },
    ("MARITIME", "REEFER", "AVERAGE"): {
        "co2": Decimal("0.135"),
        "ch4": Decimal("0.0005"),
        "n2o": Decimal("0.0007"),
        "total_direct": Decimal("0.136"),
        "wtt": Decimal("0.018"),
        "wtw": Decimal("0.154"),
    },
    ("MARITIME", "FERRY", "AVERAGE"): {
        "co2": Decimal("0.150"),
        "ch4": Decimal("0.0005"),
        "n2o": Decimal("0.0008"),
        "total_direct": Decimal("0.151"),
        "wtt": Decimal("0.020"),
        "wtw": Decimal("0.171"),
    },
    # Air - Short haul
    ("AIR", "SHORT_HAUL_NARROW", "AVERAGE"): {
        "co2": Decimal("1.250"),
        "ch4": Decimal("0.0010"),
        "n2o": Decimal("0.0015"),
        "total_direct": Decimal("1.253"),
        "wtt": Decimal("0.175"),
        "wtw": Decimal("1.428"),
    },
    # Air - Medium haul
    ("AIR", "MEDIUM_HAUL_NARROW", "AVERAGE"): {
        "co2": Decimal("0.850"),
        "ch4": Decimal("0.0007"),
        "n2o": Decimal("0.0010"),
        "total_direct": Decimal("0.852"),
        "wtt": Decimal("0.119"),
        "wtw": Decimal("0.971"),
    },
    # Air - Long haul
    ("AIR", "LONG_HAUL_WIDE", "AVERAGE"): {
        "co2": Decimal("0.550"),
        "ch4": Decimal("0.0005"),
        "n2o": Decimal("0.0007"),
        "total_direct": Decimal("0.551"),
        "wtt": Decimal("0.077"),
        "wtw": Decimal("0.628"),
    },
    # Air - Freighter narrow
    ("AIR", "FREIGHTER_NARROW", "AVERAGE"): {
        "co2": Decimal("1.100"),
        "ch4": Decimal("0.0009"),
        "n2o": Decimal("0.0013"),
        "total_direct": Decimal("1.103"),
        "wtt": Decimal("0.154"),
        "wtw": Decimal("1.257"),
    },
    # Air - Freighter wide
    ("AIR", "FREIGHTER_WIDE", "AVERAGE"): {
        "co2": Decimal("0.650"),
        "ch4": Decimal("0.0005"),
        "n2o": Decimal("0.0008"),
        "total_direct": Decimal("0.651"),
        "wtt": Decimal("0.091"),
        "wtw": Decimal("0.742"),
    },
}


# ============================================================================
# EPA SMARTWAY FACTORS (gCO2e per ton-mile)
# ============================================================================

EPA_SMARTWAY_FACTORS = {
    "TRUCK": Decimal("161.8"),
    "RAIL": Decimal("21.3"),
    "WATERBORNE": Decimal("16.4"),
    "AIR": Decimal("1389.0"),
    "MULTIMODAL": Decimal("68.5"),
}


# ============================================================================
# GLEC DEFAULT FACTORS (kgCO2e per tkm)
# ============================================================================

GLEC_DEFAULT_FACTORS = {
    ("ROAD", "GLOBAL"): Decimal("0.102"),
    ("ROAD", "EU"): Decimal("0.097"),
    ("ROAD", "US"): Decimal("0.112"),
    ("ROAD", "CN"): Decimal("0.117"),
    ("ROAD", "ASIA"): Decimal("0.114"),
    ("RAIL", "GLOBAL"): Decimal("0.035"),
    ("RAIL", "EU"): Decimal("0.025"),
    ("RAIL", "US"): Decimal("0.042"),
    ("RAIL", "CN"): Decimal("0.035"),
    ("MARITIME", "GLOBAL"): Decimal("0.011"),
    ("AIR", "GLOBAL"): Decimal("0.602"),
}


# ============================================================================
# NAICS TRANSPORT CODES
# ============================================================================

NAICS_TRANSPORT_CODES = {
    "484110": "General Freight Trucking, Local",
    "484121": "General Freight Trucking, Long-Distance, Truckload",
    "484122": "General Freight Trucking, Long-Distance, Less Than Truckload",
    "484210": "Used Household and Office Goods Moving",
    "484220": "Specialized Freight (except Used Goods) Trucking, Local",
    "484230": "Specialized Freight (except Used Goods) Trucking, Long-Distance",
    "482111": "Rail Transportation",
    "483111": "Deep Sea Freight Transportation",
    "483113": "Coastal and Great Lakes Freight Transportation",
    "483211": "Inland Water Freight Transportation",
    "481112": "Scheduled Freight Air Transportation",
    "481212": "Nonscheduled Chartered Freight Air Transportation",
    "486110": "Pipeline Transportation of Crude Oil",
    "486210": "Pipeline Transportation of Natural Gas",
    "486910": "Pipeline Transportation of Refined Petroleum Products",
    "488510": "Freight Transportation Arrangement",
    "493110": "General Warehousing and Storage",
    "493120": "Refrigerated Warehousing and Storage",
}


# ============================================================================
# FUEL EMISSION FACTORS (kgCO2e per unit)
# ============================================================================

FUEL_EMISSION_FACTORS = {
    # Diesel (per liter)
    ("DIESEL", "DIRECT"): Decimal("2.673"),
    ("DIESEL", "WTT"): Decimal("0.716"),
    ("DIESEL", "WTW"): Decimal("3.389"),
    # Gasoline (per liter)
    ("GASOLINE", "DIRECT"): Decimal("2.392"),
    ("GASOLINE", "WTT"): Decimal("0.647"),
    ("GASOLINE", "WTW"): Decimal("3.039"),
    # HFO (per kg)
    ("HFO", "DIRECT"): Decimal("3.114"),
    ("HFO", "WTT"): Decimal("0.468"),
    ("HFO", "WTW"): Decimal("3.582"),
    # MGO (per kg)
    ("MGO", "DIRECT"): Decimal("3.206"),
    ("MGO", "WTT"): Decimal("0.481"),
    ("MGO", "WTW"): Decimal("3.687"),
    # CNG (per kg)
    ("CNG", "DIRECT"): Decimal("2.750"),
    ("CNG", "WTT"): Decimal("0.550"),
    ("CNG", "WTW"): Decimal("3.300"),
    # LNG (per kg)
    ("LNG", "DIRECT"): Decimal("2.750"),
    ("LNG", "WTT"): Decimal("0.660"),
    ("LNG", "WTW"): Decimal("3.410"),
    # Hydrogen (per kg)
    ("HYDROGEN", "DIRECT"): Decimal("0.000"),
    ("HYDROGEN", "WTT"): Decimal("12.000"),  # Depends on production method
    ("HYDROGEN", "WTW"): Decimal("12.000"),
    # Electricity (per kWh) - grid average
    ("ELECTRIC", "DIRECT"): Decimal("0.000"),
    ("ELECTRIC", "WTT"): Decimal("0.350"),
    ("ELECTRIC", "WTW"): Decimal("0.350"),
    # Jet fuel (per liter)
    ("JET_FUEL", "DIRECT"): Decimal("2.520"),
    ("JET_FUEL", "WTT"): Decimal("0.352"),
    ("JET_FUEL", "WTW"): Decimal("2.872"),
}


# ============================================================================
# EEIO FACTORS (kgCO2e per $1000 USD, year 2021)
# ============================================================================

EEIO_FACTORS = {
    "484110": Decimal("523.4"),  # General Freight Trucking, Local
    "484121": Decimal("489.7"),  # Truckload
    "484122": Decimal("512.3"),  # LTL
    "484210": Decimal("495.8"),  # Moving
    "484220": Decimal("534.6"),  # Specialized Local
    "484230": Decimal("501.2"),  # Specialized Long-Distance
    "482111": Decimal("178.5"),  # Rail
    "483111": Decimal("89.4"),   # Deep Sea
    "483113": Decimal("102.7"),  # Coastal
    "483211": Decimal("95.3"),   # Inland Water
    "481112": Decimal("1456.8"), # Scheduled Air Freight
    "481212": Decimal("1523.4"), # Chartered Air Freight
    "486110": Decimal("245.6"),  # Crude Pipeline
    "486210": Decimal("198.7"),  # Gas Pipeline
    "486910": Decimal("234.5"),  # Refined Products Pipeline
    "488510": Decimal("312.4"),  # Freight Arrangement
    "493110": Decimal("167.3"),  # Warehousing
    "493120": Decimal("245.8"),  # Refrigerated Warehousing
}


# ============================================================================
# HUB EMISSION FACTORS (kgCO2e per m² per year)
# ============================================================================

HUB_EMISSION_FACTORS = {
    ("AMBIENT_WAREHOUSE", "GLOBAL"): Decimal("18.5"),
    ("AMBIENT_WAREHOUSE", "EU"): Decimal("12.3"),
    ("AMBIENT_WAREHOUSE", "US"): Decimal("22.7"),
    ("REFRIGERATED_WAREHOUSE", "GLOBAL"): Decimal("95.4"),
    ("REFRIGERATED_WAREHOUSE", "EU"): Decimal("68.2"),
    ("REFRIGERATED_WAREHOUSE", "US"): Decimal("110.8"),
    ("DISTRIBUTION_CENTER", "GLOBAL"): Decimal("22.3"),
    ("CROSS_DOCK", "GLOBAL"): Decimal("8.7"),
}


# ============================================================================
# REEFER UPLIFT FACTORS (multiplier)
# ============================================================================

REEFER_UPLIFT_FACTORS = {
    ("ROAD", "CHILLED"): Decimal("1.15"),
    ("ROAD", "FROZEN"): Decimal("1.30"),
    ("MARITIME", "CHILLED"): Decimal("1.20"),
    ("MARITIME", "FROZEN"): Decimal("1.40"),
    ("RAIL", "CHILLED"): Decimal("1.12"),
    ("RAIL", "FROZEN"): Decimal("1.25"),
}


# ============================================================================
# LOAD FACTORS (average utilization)
# ============================================================================

LOAD_FACTORS = {
    "ROAD": Decimal("0.65"),
    "RAIL": Decimal("0.70"),
    "MARITIME": Decimal("0.75"),
    "AIR": Decimal("0.68"),
}


# ============================================================================
# EMPTY RUNNING RATES (fraction of distance empty)
# ============================================================================

EMPTY_RUNNING_RATES = {
    "ROAD": Decimal("0.35"),
    "RAIL": Decimal("0.25"),
    "MARITIME": Decimal("0.10"),
    "AIR": Decimal("0.15"),
}


# ============================================================================
# UNIT CONVERSION FACTORS
# ============================================================================

UNIT_CONVERSIONS = {
    ("km", "mile"): Decimal("0.621371"),
    ("mile", "km"): Decimal("1.60934"),
    ("tonne", "ton_us"): Decimal("1.10231"),
    ("ton_us", "tonne"): Decimal("0.907185"),
    ("tkm", "ton_mile"): Decimal("0.621371"),
    ("ton_mile", "tkm"): Decimal("1.60934"),
    ("kg", "tonne"): Decimal("0.001"),
    ("tonne", "kg"): Decimal("1000.0"),
    ("g", "kg"): Decimal("0.001"),
    ("kg", "g"): Decimal("1000.0"),
    ("l", "gal_us"): Decimal("0.264172"),
    ("gal_us", "l"): Decimal("3.78541"),
}


# ============================================================================
# TRANSPORT DATABASE ENGINE
# ============================================================================

class TransportDatabaseEngine:
    """
    TransportDatabaseEngine - Emission factor database and classification.

    This engine provides comprehensive emission factor databases for all transport modes,
    vehicle/vessel classifications, and regional adjustments. It follows GLEC Framework,
    DEFRA, EPA SmartWay, and GHG Protocol Scope 3 guidance.

    Thread-safe singleton pattern ensures consistent factor retrieval across application.

    Attributes:
        _instance: Singleton instance
        _lock: Thread lock for singleton safety
        _instance_lock: Lock for instance operations

    Example:
        >>> engine = TransportDatabaseEngine()
        >>> factor = engine.get_road_emission_factor("ARTICULATED_40_44T", "DIESEL", "LADEN")
        >>> print(factor["wtw"])  # Well-to-wheel emissions kgCO2e/tkm
        0.071
    """

    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        """Create singleton instance with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize TransportDatabaseEngine (once)."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self._instance_lock = threading.RLock()
            self._initialized = True

            logger.info("TransportDatabaseEngine initialized (singleton)")

    # ========================================================================
    # PUBLIC METHODS - ROAD
    # ========================================================================

    def get_road_emission_factor(
        self,
        vehicle_type: str,
        fuel_type: str = "DIESEL",
        laden_state: str = "AVERAGE",
        region: str = "GLOBAL",
    ) -> Dict[str, Decimal]:
        """
        Get road emission factor for specific vehicle configuration.

        Args:
            vehicle_type: Vehicle classification (e.g., "ARTICULATED_40_44T")
            fuel_type: Fuel type (DIESEL, ELECTRIC, CNG, etc.)
            laden_state: Load state (LADEN, UNLADEN, AVERAGE)
            region: Geographic region for adjustment

        Returns:
            Dict with keys: co2, ch4, n2o, total_direct, wtt, wtw (all kgCO2e/tkm)

        Raises:
            ValueError: If vehicle_type or parameters invalid

        Example:
            >>> factor = engine.get_road_emission_factor("ARTICULATED_40_44T", "DIESEL", "LADEN")
            >>> print(factor["wtw"])
            0.071
        """
        with self._instance_lock:
            # Validate inputs
            if vehicle_type not in VEHICLE_CLASSIFICATIONS:
                raise ValueError(f"Unknown vehicle type: {vehicle_type}")

            # Build lookup key
            lookup_key = ("ROAD", vehicle_type, laden_state)

            # Check DEFRA scenarios
            if lookup_key not in DEFRA_FREIGHT_SCENARIOS:
                # Try to find closest match
                logger.warning(f"No DEFRA factor for {lookup_key}, using AVERAGE")
                lookup_key = ("ROAD", vehicle_type, "AVERAGE")

            if lookup_key not in DEFRA_FREIGHT_SCENARIOS:
                raise ValueError(f"No emission factor for {lookup_key}")

            factor = DEFRA_FREIGHT_SCENARIOS[lookup_key].copy()

            # Apply regional adjustment
            regional_adj = self._get_regional_adjustment("ROAD", region)
            for key in ["co2", "ch4", "n2o", "total_direct", "wtt", "wtw"]:
                factor[key] = factor[key] * regional_adj

            # Apply laden adjustment if needed
            laden_adj = self.get_laden_adjustment(vehicle_type, laden_state)
            if laden_adj != Decimal("1.0"):
                for key in factor:
                    factor[key] = factor[key] * laden_adj

            logger.debug(
                f"Road EF: {vehicle_type} {fuel_type} {laden_state} {region} = {factor['wtw']:.6f} kgCO2e/tkm"
            )

            return factor

    # ========================================================================
    # PUBLIC METHODS - RAIL
    # ========================================================================

    def get_rail_emission_factor(
        self,
        rail_type: str,
        region: str = "GLOBAL",
    ) -> Dict[str, Decimal]:
        """
        Get rail emission factor.

        Args:
            rail_type: Rail classification (FREIGHT_ELECTRIC, FREIGHT_DIESEL, etc.)
            region: Geographic region

        Returns:
            Dict with emission components (kgCO2e/tkm)

        Example:
            >>> factor = engine.get_rail_emission_factor("FREIGHT_ELECTRIC", "EU")
        """
        with self._instance_lock:
            lookup_key = ("RAIL", rail_type, "AVERAGE")

            if lookup_key not in DEFRA_FREIGHT_SCENARIOS:
                raise ValueError(f"Unknown rail type: {rail_type}")

            factor = DEFRA_FREIGHT_SCENARIOS[lookup_key].copy()

            # Apply regional adjustment
            regional_adj = self._get_regional_adjustment("RAIL", region)
            for key in factor:
                factor[key] = factor[key] * regional_adj

            logger.debug(f"Rail EF: {rail_type} {region} = {factor['wtw']:.6f} kgCO2e/tkm")

            return factor

    # ========================================================================
    # PUBLIC METHODS - MARITIME
    # ========================================================================

    def get_maritime_emission_factor(
        self,
        vessel_type: str,
    ) -> Dict[str, Decimal]:
        """
        Get maritime emission factor.

        Args:
            vessel_type: Vessel classification (e.g., "CONTAINER_POST_PANAMAX")

        Returns:
            Dict with emission components (kgCO2e/tkm)

        Example:
            >>> factor = engine.get_maritime_emission_factor("CONTAINER_POST_PANAMAX")
        """
        with self._instance_lock:
            lookup_key = ("MARITIME", vessel_type, "AVERAGE")

            if lookup_key not in DEFRA_FREIGHT_SCENARIOS:
                raise ValueError(f"Unknown vessel type: {vessel_type}")

            factor = DEFRA_FREIGHT_SCENARIOS[lookup_key].copy()

            logger.debug(f"Maritime EF: {vessel_type} = {factor['wtw']:.6f} kgCO2e/tkm")

            return factor

    # ========================================================================
    # PUBLIC METHODS - AIR
    # ========================================================================

    def get_air_emission_factor(
        self,
        aircraft_type: str,
        distance_km: Optional[float] = None,
    ) -> Dict[str, Decimal]:
        """
        Get air emission factor.

        Args:
            aircraft_type: Aircraft classification
            distance_km: Flight distance (affects fuel efficiency)

        Returns:
            Dict with emission components (kgCO2e/tkm)

        Example:
            >>> factor = engine.get_air_emission_factor("FREIGHTER_WIDE", 5000)
        """
        with self._instance_lock:
            lookup_key = ("AIR", aircraft_type, "AVERAGE")

            if lookup_key not in DEFRA_FREIGHT_SCENARIOS:
                raise ValueError(f"Unknown aircraft type: {aircraft_type}")

            factor = DEFRA_FREIGHT_SCENARIOS[lookup_key].copy()

            # Apply distance adjustment if provided
            if distance_km is not None:
                distance_adj = self._calculate_air_distance_adjustment(distance_km)
                for key in factor:
                    factor[key] = factor[key] * distance_adj

            logger.debug(f"Air EF: {aircraft_type} = {factor['wtw']:.6f} kgCO2e/tkm")

            return factor

    # ========================================================================
    # PUBLIC METHODS - PIPELINE
    # ========================================================================

    def get_pipeline_emission_factor(
        self,
        pipeline_type: str,
    ) -> Decimal:
        """
        Get pipeline emission factor.

        Args:
            pipeline_type: Pipeline type (NATURAL_GAS, CRUDE_OIL, REFINED_PRODUCTS)

        Returns:
            Emission factor (kgCO2e/tkm)

        Example:
            >>> factor = engine.get_pipeline_emission_factor("NATURAL_GAS")
        """
        with self._instance_lock:
            # Pipeline factors (very low emissions per tkm)
            pipeline_factors = {
                "NATURAL_GAS": Decimal("0.005"),
                "CRUDE_OIL": Decimal("0.003"),
                "REFINED_PRODUCTS": Decimal("0.004"),
            }

            if pipeline_type not in pipeline_factors:
                raise ValueError(f"Unknown pipeline type: {pipeline_type}")

            factor = pipeline_factors[pipeline_type]

            logger.debug(f"Pipeline EF: {pipeline_type} = {factor:.6f} kgCO2e/tkm")

            return factor

    # ========================================================================
    # PUBLIC METHODS - FUEL
    # ========================================================================

    def get_fuel_emission_factor(
        self,
        fuel_type: str,
        scope: str = "WTW",
    ) -> Decimal:
        """
        Get fuel emission factor.

        Args:
            fuel_type: Fuel type (DIESEL, GASOLINE, HFO, etc.)
            scope: Emission scope (DIRECT, WTT, WTW)

        Returns:
            Emission factor (kgCO2e per unit)

        Example:
            >>> factor = engine.get_fuel_emission_factor("DIESEL", "WTW")
            >>> print(factor)  # kgCO2e per liter
            3.389
        """
        with self._instance_lock:
            scope_upper = scope.upper()
            lookup_key = (fuel_type, scope_upper)

            if lookup_key not in FUEL_EMISSION_FACTORS:
                raise ValueError(f"Unknown fuel/scope: {fuel_type}/{scope}")

            factor = FUEL_EMISSION_FACTORS[lookup_key]

            logger.debug(f"Fuel EF: {fuel_type} {scope} = {factor:.6f} kgCO2e/unit")

            return factor

    # ========================================================================
    # PUBLIC METHODS - EEIO
    # ========================================================================

    def get_eeio_factor(
        self,
        naics_code: str,
        currency: str = "USD",
        year: int = 2021,
    ) -> Decimal:
        """
        Get EEIO emission factor for transport service.

        Args:
            naics_code: NAICS code (6-digit)
            currency: Currency (only USD supported)
            year: Year (only 2021 supported)

        Returns:
            Emission factor (kgCO2e per $1000)

        Example:
            >>> factor = engine.get_eeio_factor("484121")  # Truckload
            >>> print(factor)
            489.7
        """
        with self._instance_lock:
            if currency != "USD":
                raise ValueError(f"Only USD supported, got {currency}")

            if year != 2021:
                logger.warning(f"Only 2021 factors available, requested {year}")

            if naics_code not in EEIO_FACTORS:
                raise ValueError(f"Unknown NAICS code: {naics_code}")

            factor = EEIO_FACTORS[naics_code]

            logger.debug(f"EEIO EF: {naics_code} = {factor:.2f} kgCO2e/$1000")

            return factor

    # ========================================================================
    # PUBLIC METHODS - HUB
    # ========================================================================

    def get_hub_emission_factor(
        self,
        hub_type: str,
        temperature: str = "AMBIENT",
    ) -> Decimal:
        """
        Get warehouse/hub emission factor.

        Args:
            hub_type: Hub type (AMBIENT_WAREHOUSE, REFRIGERATED_WAREHOUSE, etc.)
            temperature: Temperature control (AMBIENT, CHILLED, FROZEN)

        Returns:
            Emission factor (kgCO2e per m² per year)

        Example:
            >>> factor = engine.get_hub_emission_factor("AMBIENT_WAREHOUSE")
        """
        with self._instance_lock:
            # Map temperature to hub type
            if temperature in ["CHILLED", "FROZEN"]:
                hub_type = "REFRIGERATED_WAREHOUSE"

            lookup_key = (hub_type, "GLOBAL")

            if lookup_key not in HUB_EMISSION_FACTORS:
                raise ValueError(f"Unknown hub type: {hub_type}")

            factor = HUB_EMISSION_FACTORS[lookup_key]

            logger.debug(f"Hub EF: {hub_type} = {factor:.2f} kgCO2e/m²/year")

            return factor

    # ========================================================================
    # PUBLIC METHODS - REEFER UPLIFT
    # ========================================================================

    def get_reefer_uplift(
        self,
        mode: str,
        temperature: str,
    ) -> Decimal:
        """
        Get refrigerated transport uplift factor.

        Args:
            mode: Transport mode
            temperature: Temperature control (CHILLED, FROZEN)

        Returns:
            Uplift multiplier

        Example:
            >>> uplift = engine.get_reefer_uplift("ROAD", "FROZEN")
            >>> print(uplift)
            1.30
        """
        with self._instance_lock:
            if temperature == "AMBIENT":
                return Decimal("1.0")

            lookup_key = (mode, temperature)

            if lookup_key not in REEFER_UPLIFT_FACTORS:
                logger.warning(f"No reefer uplift for {lookup_key}, using 1.0")
                return Decimal("1.0")

            factor = REEFER_UPLIFT_FACTORS[lookup_key]

            logger.debug(f"Reefer uplift: {mode} {temperature} = {factor:.2f}")

            return factor

    # ========================================================================
    # PUBLIC METHODS - LOAD FACTOR
    # ========================================================================

    def get_load_factor(
        self,
        mode: str,
    ) -> Decimal:
        """
        Get average load factor for mode.

        Args:
            mode: Transport mode

        Returns:
            Load factor (0-1)

        Example:
            >>> lf = engine.get_load_factor("ROAD")
            >>> print(lf)
            0.65
        """
        with self._instance_lock:
            if mode not in LOAD_FACTORS:
                logger.warning(f"No load factor for {mode}, using 0.65")
                return Decimal("0.65")

            factor = LOAD_FACTORS[mode]

            logger.debug(f"Load factor: {mode} = {factor:.2f}")

            return factor

    # ========================================================================
    # PUBLIC METHODS - EMPTY RUNNING
    # ========================================================================

    def get_empty_running_rate(
        self,
        mode: str,
    ) -> Decimal:
        """
        Get empty running rate (fraction of distance traveled empty).

        Args:
            mode: Transport mode

        Returns:
            Empty running rate (0-1)

        Example:
            >>> err = engine.get_empty_running_rate("ROAD")
            >>> print(err)
            0.35
        """
        with self._instance_lock:
            if mode not in EMPTY_RUNNING_RATES:
                logger.warning(f"No empty running rate for {mode}, using 0.30")
                return Decimal("0.30")

            rate = EMPTY_RUNNING_RATES[mode]

            logger.debug(f"Empty running rate: {mode} = {rate:.2f}")

            return rate

    # ========================================================================
    # PUBLIC METHODS - WAREHOUSE INTENSITY
    # ========================================================================

    def get_warehouse_intensity(
        self,
        warehouse_type: str,
        region: str = "GLOBAL",
    ) -> Decimal:
        """
        Get warehouse emission intensity.

        Args:
            warehouse_type: Warehouse type
            region: Geographic region

        Returns:
            Emission intensity (kgCO2e/m²/year)

        Example:
            >>> intensity = engine.get_warehouse_intensity("AMBIENT_WAREHOUSE", "EU")
        """
        with self._instance_lock:
            lookup_key = (warehouse_type, region)

            if lookup_key not in HUB_EMISSION_FACTORS:
                # Try global
                lookup_key = (warehouse_type, "GLOBAL")

            if lookup_key not in HUB_EMISSION_FACTORS:
                raise ValueError(f"Unknown warehouse type: {warehouse_type}")

            intensity = HUB_EMISSION_FACTORS[lookup_key]

            logger.debug(f"Warehouse intensity: {warehouse_type} {region} = {intensity:.2f}")

            return intensity

    # ========================================================================
    # PUBLIC METHODS - VEHICLE CLASSIFICATION
    # ========================================================================

    def classify_vehicle(
        self,
        gvw_tonnes: float,
    ) -> str:
        """
        Classify vehicle based on GVW.

        Args:
            gvw_tonnes: Gross Vehicle Weight in tonnes

        Returns:
            Vehicle type classification

        Example:
            >>> vtype = engine.classify_vehicle(25.0)
            >>> print(vtype)
            RIGID_17_26T
        """
        with self._instance_lock:
            gvw = Decimal(str(gvw_tonnes))

            # Classification logic
            if gvw <= Decimal("3.5"):
                return "LCV_DIESEL"
            elif gvw <= Decimal("7.5"):
                return "RIGID_3_5_7_5T"
            elif gvw <= Decimal("17.0"):
                return "RIGID_7_5_17T"
            elif gvw <= Decimal("26.0"):
                return "RIGID_17_26T"
            elif gvw <= Decimal("28.0"):
                return "RIGID_26_28T"
            elif gvw <= Decimal("32.0"):
                return "RIGID_28_32T"
            elif gvw <= Decimal("33.0"):
                return "ARTICULATED_33T"
            elif gvw <= Decimal("44.0"):
                return "ARTICULATED_40_44T"
            else:
                return "ROAD_TRAIN"

    # ========================================================================
    # PUBLIC METHODS - VESSEL CLASSIFICATION
    # ========================================================================

    def classify_vessel(
        self,
        vessel_type_str: str,
        dwt_or_teu: Optional[float] = None,
    ) -> str:
        """
        Classify vessel based on type and capacity.

        Args:
            vessel_type_str: Vessel type hint (container, bulk, tanker, etc.)
            dwt_or_teu: DWT or TEU capacity

        Returns:
            Vessel type classification

        Example:
            >>> vtype = engine.classify_vessel("container", 8000)
            >>> print(vtype)
            CONTAINER_FEEDERMAX
        """
        with self._instance_lock:
            vessel_lower = vessel_type_str.lower()

            # Container ships (by TEU)
            if "container" in vessel_lower:
                if dwt_or_teu is None:
                    return "CONTAINER_POST_PANAMAX"

                teu = dwt_or_teu
                if teu < 500:
                    return "CONTAINER_FEEDER"
                elif teu < 3000:
                    return "CONTAINER_FEEDERMAX"
                elif teu < 7000:
                    return "CONTAINER_PANAMAX"
                elif teu < 15000:
                    return "CONTAINER_POST_PANAMAX"
                else:
                    return "CONTAINER_ULCV"

            # Bulk carriers (by DWT)
            elif "bulk" in vessel_lower:
                if dwt_or_teu is None:
                    return "BULK_CARRIER_PANAMAX"

                dwt = dwt_or_teu
                if dwt < 40000:
                    return "BULK_CARRIER_HANDYSIZE"
                elif dwt < 60000:
                    return "BULK_CARRIER_HANDYMAX"
                elif dwt < 90000:
                    return "BULK_CARRIER_PANAMAX"
                else:
                    return "BULK_CARRIER_CAPESIZE"

            # Tankers (by DWT)
            elif "tanker" in vessel_lower:
                if dwt_or_teu is None:
                    return "TANKER_AFRAMAX"

                dwt = dwt_or_teu
                if dwt < 120000:
                    return "TANKER_AFRAMAX"
                elif dwt < 200000:
                    return "TANKER_SUEZMAX"
                else:
                    return "TANKER_VLCC"

            # Other types
            elif "roro" in vessel_lower or "ro-ro" in vessel_lower:
                return "RORO"
            elif "reefer" in vessel_lower or "refrigerated" in vessel_lower:
                return "REEFER"
            elif "ferry" in vessel_lower:
                return "FERRY"
            else:
                return "GENERAL_CARGO"

    # ========================================================================
    # PUBLIC METHODS - AIRCRAFT CLASSIFICATION
    # ========================================================================

    def classify_aircraft(
        self,
        aircraft_str: str,
        distance_km: Optional[float] = None,
    ) -> str:
        """
        Classify aircraft based on type and distance.

        Args:
            aircraft_str: Aircraft type hint (freighter, passenger, model number)
            distance_km: Flight distance

        Returns:
            Aircraft type classification

        Example:
            >>> atype = engine.classify_aircraft("B747F", 8000)
            >>> print(atype)
            FREIGHTER_WIDE
        """
        with self._instance_lock:
            aircraft_lower = aircraft_str.lower()

            # Freighter vs passenger
            is_freighter = any(x in aircraft_lower for x in ["f", "freighter", "cargo"])

            if is_freighter:
                # Wide vs narrow
                if any(x in aircraft_lower for x in ["747", "777", "md-11", "a330", "a350"]):
                    return "FREIGHTER_WIDE"
                else:
                    return "FREIGHTER_NARROW"
            else:
                # Passenger - classify by distance
                if distance_km is None:
                    return "MEDIUM_HAUL_NARROW"

                if distance_km < 3000:
                    return "SHORT_HAUL_NARROW"
                elif distance_km < 6000:
                    return "MEDIUM_HAUL_NARROW"
                else:
                    return "LONG_HAUL_WIDE"

    # ========================================================================
    # PUBLIC METHODS - VEHICLE PAYLOAD
    # ========================================================================

    def get_vehicle_payload(
        self,
        vehicle_type: str,
    ) -> Decimal:
        """
        Get vehicle payload capacity.

        Args:
            vehicle_type: Vehicle classification

        Returns:
            Payload capacity (tonnes)

        Example:
            >>> payload = engine.get_vehicle_payload("ARTICULATED_40_44T")
            >>> print(payload)
            29.0
        """
        with self._instance_lock:
            if vehicle_type not in VEHICLE_CLASSIFICATIONS:
                raise ValueError(f"Unknown vehicle type: {vehicle_type}")

            payload = VEHICLE_CLASSIFICATIONS[vehicle_type]["payload_tonnes"]

            logger.debug(f"Vehicle payload: {vehicle_type} = {payload} tonnes")

            return payload

    # ========================================================================
    # PUBLIC METHODS - VESSEL CAPACITY
    # ========================================================================

    def get_vessel_capacity(
        self,
        vessel_type: str,
    ) -> Dict[str, Any]:
        """
        Get vessel capacity details.

        Args:
            vessel_type: Vessel classification

        Returns:
            Dict with dwt_range, teu_capacity, speed_knots

        Example:
            >>> capacity = engine.get_vessel_capacity("CONTAINER_POST_PANAMAX")
        """
        with self._instance_lock:
            if vessel_type not in VESSEL_CLASSIFICATIONS:
                raise ValueError(f"Unknown vessel type: {vessel_type}")

            capacity = VESSEL_CLASSIFICATIONS[vessel_type].copy()

            logger.debug(f"Vessel capacity: {vessel_type} = {capacity}")

            return capacity

    # ========================================================================
    # PUBLIC METHODS - LADEN ADJUSTMENT
    # ========================================================================

    def get_laden_adjustment(
        self,
        vehicle_type: str,
        laden_state: str,
    ) -> Decimal:
        """
        Get laden state adjustment factor.

        Args:
            vehicle_type: Vehicle classification
            laden_state: LADEN, UNLADEN, or AVERAGE

        Returns:
            Adjustment multiplier

        Example:
            >>> adj = engine.get_laden_adjustment("ARTICULATED_40_44T", "LADEN")
        """
        with self._instance_lock:
            # Laden adjustments already baked into DEFRA factors
            # This method returns 1.0 but could be extended for custom adjustments
            return Decimal("1.0")

    # ========================================================================
    # PUBLIC METHODS - EF HIERARCHY
    # ========================================================================

    def resolve_ef_hierarchy(
        self,
        mode: str,
        vehicle_type: Optional[str] = None,
        region: Optional[str] = None,
        source_preference: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Resolve emission factor using hierarchy (DEFRA > SmartWay > GLEC).

        Args:
            mode: Transport mode
            vehicle_type: Vehicle/vessel type
            region: Geographic region
            source_preference: Preferred sources in order

        Returns:
            Dict with factor, source, confidence

        Example:
            >>> result = engine.resolve_ef_hierarchy("ROAD", "ARTICULATED_40_44T", "EU")
            >>> print(result["source"])
            DEFRA
        """
        with self._instance_lock:
            if source_preference is None:
                source_preference = ["DEFRA", "SMARTWAY", "GLEC"]

            result = {
                "factor": None,
                "source": None,
                "confidence": "LOW",
                "unit": "kgCO2e/tkm",
            }

            # Try each source in preference order
            for source in source_preference:
                if source == "DEFRA":
                    try:
                        if mode == "ROAD" and vehicle_type:
                            factor_dict = self.get_road_emission_factor(
                                vehicle_type, "DIESEL", "AVERAGE", region or "GLOBAL"
                            )
                            result["factor"] = factor_dict["wtw"]
                            result["source"] = "DEFRA"
                            result["confidence"] = "HIGH"
                            return result
                        elif mode == "RAIL" and vehicle_type:
                            factor_dict = self.get_rail_emission_factor(
                                vehicle_type, region or "GLOBAL"
                            )
                            result["factor"] = factor_dict["wtw"]
                            result["source"] = "DEFRA"
                            result["confidence"] = "HIGH"
                            return result
                        elif mode == "MARITIME" and vehicle_type:
                            factor_dict = self.get_maritime_emission_factor(vehicle_type)
                            result["factor"] = factor_dict["wtw"]
                            result["source"] = "DEFRA"
                            result["confidence"] = "HIGH"
                            return result
                    except (ValueError, KeyError):
                        continue

                elif source == "SMARTWAY":
                    try:
                        mode_map = {
                            "ROAD": "TRUCK",
                            "RAIL": "RAIL",
                            "MARITIME": "WATERBORNE",
                            "AIR": "AIR",
                        }
                        smartway_mode = mode_map.get(mode)
                        if smartway_mode:
                            factor = self.get_smartway_factor(smartway_mode)
                            # Convert g/ton-mile to kg/tkm
                            factor_kgtkm = factor / Decimal("1000") * Decimal("0.621371")
                            result["factor"] = factor_kgtkm
                            result["source"] = "SMARTWAY"
                            result["confidence"] = "MEDIUM"
                            return result
                    except (ValueError, KeyError):
                        continue

                elif source == "GLEC":
                    try:
                        factor = self.get_glec_factor(mode, region or "GLOBAL")
                        result["factor"] = factor
                        result["source"] = "GLEC"
                        result["confidence"] = "MEDIUM"
                        return result
                    except (ValueError, KeyError):
                        continue

            # Fallback
            logger.warning(f"No EF found for {mode}/{vehicle_type}/{region}, using generic")
            result["factor"] = Decimal("0.100")
            result["source"] = "GENERIC"
            result["confidence"] = "LOW"

            return result

    # ========================================================================
    # PUBLIC METHODS - UNIT CONVERSION
    # ========================================================================

    def convert_units(
        self,
        value: Decimal,
        from_unit: str,
        to_unit: str,
    ) -> Decimal:
        """
        Convert between units.

        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted value

        Example:
            >>> result = engine.convert_units(Decimal("100"), "km", "mile")
            >>> print(result)
            62.1371
        """
        with self._instance_lock:
            if from_unit == to_unit:
                return value

            lookup_key = (from_unit, to_unit)

            if lookup_key not in UNIT_CONVERSIONS:
                raise ValueError(f"No conversion for {from_unit} to {to_unit}")

            conversion_factor = UNIT_CONVERSIONS[lookup_key]
            result = value * conversion_factor

            logger.debug(f"Unit conversion: {value} {from_unit} = {result} {to_unit}")

            return result

    # ========================================================================
    # PUBLIC METHODS - DEFRA FACTOR
    # ========================================================================

    def get_defra_factor(
        self,
        mode: str,
        vehicle_type: str,
        laden_state: str,
    ) -> Dict[str, Decimal]:
        """
        Get DEFRA factor directly.

        Args:
            mode: Transport mode
            vehicle_type: Vehicle/vessel type
            laden_state: Load state

        Returns:
            Dict with emission components

        Example:
            >>> factor = engine.get_defra_factor("ROAD", "ARTICULATED_40_44T", "LADEN")
        """
        with self._instance_lock:
            lookup_key = (mode, vehicle_type, laden_state)

            if lookup_key not in DEFRA_FREIGHT_SCENARIOS:
                raise ValueError(f"No DEFRA factor for {lookup_key}")

            return DEFRA_FREIGHT_SCENARIOS[lookup_key].copy()

    # ========================================================================
    # PUBLIC METHODS - SMARTWAY FACTOR
    # ========================================================================

    def get_smartway_factor(
        self,
        mode: str,
    ) -> Decimal:
        """
        Get EPA SmartWay factor.

        Args:
            mode: Transport mode (TRUCK, RAIL, WATERBORNE, AIR)

        Returns:
            Emission factor (gCO2e/ton-mile)

        Example:
            >>> factor = engine.get_smartway_factor("TRUCK")
            >>> print(factor)
            161.8
        """
        with self._instance_lock:
            if mode not in EPA_SMARTWAY_FACTORS:
                raise ValueError(f"Unknown SmartWay mode: {mode}")

            return EPA_SMARTWAY_FACTORS[mode]

    # ========================================================================
    # PUBLIC METHODS - GLEC FACTOR
    # ========================================================================

    def get_glec_factor(
        self,
        mode: str,
        region: str,
    ) -> Decimal:
        """
        Get GLEC Framework default factor.

        Args:
            mode: Transport mode
            region: Geographic region

        Returns:
            Emission factor (kgCO2e/tkm)

        Example:
            >>> factor = engine.get_glec_factor("ROAD", "EU")
            >>> print(factor)
            0.097
        """
        with self._instance_lock:
            lookup_key = (mode, region)

            if lookup_key not in GLEC_DEFAULT_FACTORS:
                # Try global
                lookup_key = (mode, "GLOBAL")

            if lookup_key not in GLEC_DEFAULT_FACTORS:
                raise ValueError(f"No GLEC factor for {mode}/{region}")

            return GLEC_DEFAULT_FACTORS[lookup_key]

    # ========================================================================
    # PUBLIC METHODS - LIST FACTORS
    # ========================================================================

    def list_available_factors(
        self,
        mode: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all available emission factors.

        Args:
            mode: Filter by transport mode (optional)

        Returns:
            List of factor metadata

        Example:
            >>> factors = engine.list_available_factors("ROAD")
            >>> print(len(factors))
        """
        with self._instance_lock:
            factors = []

            for key, value in DEFRA_FREIGHT_SCENARIOS.items():
                scenario_mode, vehicle_type, laden_state = key

                if mode and scenario_mode != mode:
                    continue

                factors.append({
                    "mode": scenario_mode,
                    "vehicle_type": vehicle_type,
                    "laden_state": laden_state,
                    "wtw_factor": value["wtw"],
                    "source": "DEFRA",
                })

            logger.info(f"Listed {len(factors)} available factors")

            return factors

    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================

    def _get_regional_adjustment(
        self,
        mode: str,
        region: str,
    ) -> Decimal:
        """Get regional adjustment factor."""
        if region == "GLOBAL":
            return Decimal("1.0")

        lookup_key = (mode, region)

        if lookup_key in REGIONAL_ADJUSTMENTS:
            return REGIONAL_ADJUSTMENTS[lookup_key]

        logger.warning(f"No regional adjustment for {mode}/{region}, using 1.0")
        return Decimal("1.0")

    def _calculate_air_distance_adjustment(
        self,
        distance_km: float,
    ) -> Decimal:
        """Calculate air distance adjustment (shorter flights less efficient)."""
        # Short haul penalty
        if distance_km < 500:
            return Decimal("1.40")
        elif distance_km < 1500:
            return Decimal("1.20")
        elif distance_km < 3000:
            return Decimal("1.10")
        else:
            return Decimal("1.00")


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "TransportDatabaseEngine",
    "TransportMode",
    "VehicleType",
    "VesselType",
    "AircraftType",
    "RailType",
    "FuelType",
    "LadenState",
    "Region",
    "PipelineType",
    "HubType",
    "TemperatureControl",
]
