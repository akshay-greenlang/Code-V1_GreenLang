# -*- coding: utf-8 -*-
"""
DistanceEstimatorEngine - Distance-Based Emission Estimation (Engine 4 of 7)

AGENT-MRV-003: Mobile Combustion Agent

Provides distance-based emission factor lookups, fuel economy estimation,
and fuel consumption derivation for mobile combustion sources. Supports
on-road vehicles, off-road equipment, marine vessels, and aviation
sources with multi-factor adjustments for vehicle age and load factor.

Distance-Based Methodology:
    The distance-based approach estimates emissions when actual fuel
    consumption data is unavailable. It uses the relationship:

        fuel_consumed = distance_km * fuel_economy / 100
        emissions = fuel_consumed * emission_factor

    Or directly:

        emissions = distance_km * distance_emission_factor

    The engine supports both approaches, selecting the most appropriate
    based on available data and vehicle type.

Supported Vehicle Categories:
    - On-road: Passenger cars (gasoline/diesel/hybrid/PHEV),
      light/medium/heavy-duty trucks, buses, motorcycles, vans
    - Off-road: Construction equipment, agricultural equipment, forklifts
    - Marine: Inland, coastal, ocean vessels
    - Aviation: Corporate jets, helicopters, turboprops

Adjustment Factors:
    - Vehicle age degradation: 5 tiers (0-3yr, 3-5yr, 5-8yr, 8-12yr, 12+yr)
    - Load factor: 6 levels (empty through overloaded)

Unit Conversions:
    - Distance: km, mi, nm (nautical miles)
    - Fuel economy: L/100km, mpg (US), mpg (UK), km/L

Zero-Hallucination Guarantees:
    - All factors are deterministic lookups from coded tables.
    - No LLM involvement in any numeric path.
    - Decimal arithmetic for bit-perfect reproducibility.
    - Every result carries a SHA-256 provenance hash.

Thread Safety:
    All mutable state is protected by a reentrant lock.

Example:
    >>> from greenlang.mobile_combustion.distance_estimator import (
    ...     DistanceEstimatorEngine,
    ... )
    >>> engine = DistanceEstimatorEngine()
    >>> result = engine.estimate_fuel_from_distance(
    ...     vehicle_type="PASSENGER_CAR_GASOLINE",
    ...     distance_km=Decimal("15000"),
    ...     fuel_type="GASOLINE",
    ...     vehicle_age_years=6,
    ...     load_factor="HALF_LOAD",
    ... )
    >>> print(result.fuel_consumed_litres)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-003 Mobile Combustion (GL-MRV-SCOPE1-003)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["DistanceEstimatorEngine"]

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from greenlang.mobile_combustion.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.mobile_combustion.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.mobile_combustion.metrics import (
        PROMETHEUS_AVAILABLE as _METRICS_AVAILABLE,
        record_distance_estimation as _record_distance_estimation,
        observe_calculation_duration as _observe_calculation_duration,
    )
except ImportError:
    _METRICS_AVAILABLE = False
    _record_distance_estimation = None  # type: ignore[assignment]
    _observe_calculation_duration = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _to_decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value (int, float, str, or Decimal).

    Returns:
        Decimal representation of the value.

    Raises:
        ValueError: If the value cannot be converted.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"Cannot convert {value!r} to Decimal") from exc


# ===========================================================================
# Enumerations
# ===========================================================================


class VehicleType(str, Enum):
    """Vehicle type classification for distance-based estimation.

    Covers all on-road, off-road, marine, and aviation categories
    tracked by GHG Protocol Chapter 7 and EPA MOVES model categories.

    On-Road Vehicles:
        PASSENGER_CAR_GASOLINE: Conventional gasoline passenger car.
        PASSENGER_CAR_DIESEL: Diesel passenger car.
        PASSENGER_CAR_HYBRID: Full hybrid (HEV) passenger car.
        PASSENGER_CAR_PHEV: Plug-in hybrid (PHEV), fuel portion only.
        LIGHT_DUTY_TRUCK_GASOLINE: Light-duty truck, gasoline (SUV/pickup).
        LIGHT_DUTY_TRUCK_DIESEL: Light-duty truck, diesel.
        MEDIUM_DUTY_TRUCK_GASOLINE: Medium-duty truck, gasoline.
        MEDIUM_DUTY_TRUCK_DIESEL: Medium-duty truck, diesel.
        HEAVY_DUTY_TRUCK_DIESEL: Heavy-duty truck (class 7-8), diesel.
        BUS_DIESEL: Transit/intercity bus, diesel.
        BUS_CNG: Bus, compressed natural gas.
        MOTORCYCLE: Motorcycle/scooter.
        VAN_LCV: Van / light commercial vehicle.

    Off-Road Equipment (operating hours):
        CONSTRUCTION_EQUIPMENT: Excavators, bulldozers, cranes.
        AGRICULTURAL_EQUIPMENT: Tractors, harvesters, sprayers.
        FORKLIFT: Industrial forklifts (diesel/LPG).

    Marine Vessels:
        MARINE_INLAND: Inland waterway vessels (barges, river boats).
        MARINE_COASTAL: Coastal/short-sea shipping vessels.
        MARINE_OCEAN: Deep-sea ocean-going vessels.

    Aviation:
        AVIATION_CORPORATE_JET: Corporate/business jets.
        AVIATION_HELICOPTER: Helicopters.
        AVIATION_TURBOPROP: Turboprop regional aircraft.
    """

    # On-road
    PASSENGER_CAR_GASOLINE = "PASSENGER_CAR_GASOLINE"
    PASSENGER_CAR_DIESEL = "PASSENGER_CAR_DIESEL"
    PASSENGER_CAR_HYBRID = "PASSENGER_CAR_HYBRID"
    PASSENGER_CAR_PHEV = "PASSENGER_CAR_PHEV"
    LIGHT_DUTY_TRUCK_GASOLINE = "LIGHT_DUTY_TRUCK_GASOLINE"
    LIGHT_DUTY_TRUCK_DIESEL = "LIGHT_DUTY_TRUCK_DIESEL"
    MEDIUM_DUTY_TRUCK_GASOLINE = "MEDIUM_DUTY_TRUCK_GASOLINE"
    MEDIUM_DUTY_TRUCK_DIESEL = "MEDIUM_DUTY_TRUCK_DIESEL"
    HEAVY_DUTY_TRUCK_DIESEL = "HEAVY_DUTY_TRUCK_DIESEL"
    BUS_DIESEL = "BUS_DIESEL"
    BUS_CNG = "BUS_CNG"
    MOTORCYCLE = "MOTORCYCLE"
    VAN_LCV = "VAN_LCV"

    # Off-road (operating hours)
    CONSTRUCTION_EQUIPMENT = "CONSTRUCTION_EQUIPMENT"
    AGRICULTURAL_EQUIPMENT = "AGRICULTURAL_EQUIPMENT"
    FORKLIFT = "FORKLIFT"

    # Marine
    MARINE_INLAND = "MARINE_INLAND"
    MARINE_COASTAL = "MARINE_COASTAL"
    MARINE_OCEAN = "MARINE_OCEAN"

    # Aviation
    AVIATION_CORPORATE_JET = "AVIATION_CORPORATE_JET"
    AVIATION_HELICOPTER = "AVIATION_HELICOPTER"
    AVIATION_TURBOPROP = "AVIATION_TURBOPROP"


class FuelType(str, Enum):
    """Fuel type for mobile combustion sources.

    GASOLINE: Motor gasoline (petrol).
    DIESEL: Diesel fuel (DERV / #2 diesel).
    CNG: Compressed natural gas.
    LPG: Liquefied petroleum gas (autogas).
    BIODIESEL: Biodiesel (B100 or blends).
    ETHANOL: Ethanol (E85 or blends).
    JET_FUEL: Aviation turbine fuel (Jet A / Jet A-1).
    MARINE_DIESEL: Marine diesel oil (MDO).
    MARINE_HFO: Heavy fuel oil (HFO/IFO).
    AVGAS: Aviation gasoline (100LL).
    """

    GASOLINE = "GASOLINE"
    DIESEL = "DIESEL"
    CNG = "CNG"
    LPG = "LPG"
    BIODIESEL = "BIODIESEL"
    ETHANOL = "ETHANOL"
    JET_FUEL = "JET_FUEL"
    MARINE_DIESEL = "MARINE_DIESEL"
    MARINE_HFO = "MARINE_HFO"
    AVGAS = "AVGAS"


class LoadFactor(str, Enum):
    """Load factor classification for fuel economy adjustment.

    EMPTY: Empty vehicle (no payload). Factor 0.70x.
    QUARTER_LOAD: Approximately 25% of rated payload. Factor 0.85x.
    HALF_LOAD: Approximately 50% of rated payload. Factor 1.00x (baseline).
    THREE_QUARTER_LOAD: Approximately 75% of rated payload. Factor 1.10x.
    FULL_LOAD: At rated payload capacity. Factor 1.20x.
    OVERLOADED: Exceeding rated payload capacity. Factor 1.35x.
    """

    EMPTY = "EMPTY"
    QUARTER_LOAD = "QUARTER_LOAD"
    HALF_LOAD = "HALF_LOAD"
    THREE_QUARTER_LOAD = "THREE_QUARTER_LOAD"
    FULL_LOAD = "FULL_LOAD"
    OVERLOADED = "OVERLOADED"


class DistanceUnit(str, Enum):
    """Distance measurement units.

    KM: Kilometres.
    MI: Statute miles.
    NM: Nautical miles.
    """

    KM = "KM"
    MI = "MI"
    NM = "NM"


class FuelEconomyUnit(str, Enum):
    """Fuel economy measurement units.

    L_PER_100KM: Litres per 100 kilometres (European standard).
    MPG_US: Miles per US gallon.
    MPG_UK: Miles per Imperial gallon.
    KM_PER_L: Kilometres per litre.
    """

    L_PER_100KM = "L_PER_100KM"
    MPG_US = "MPG_US"
    MPG_UK = "MPG_UK"
    KM_PER_L = "KM_PER_L"


class EquipmentOperatingType(str, Enum):
    """Off-road equipment type for operating-hour-based estimation.

    CONSTRUCTION_EXCAVATOR: Hydraulic excavator (20-45t class).
    CONSTRUCTION_BULLDOZER: Track-type bulldozer.
    CONSTRUCTION_CRANE: Mobile crane.
    CONSTRUCTION_LOADER: Wheel loader.
    AGRICULTURAL_TRACTOR: Farm tractor (50-200 HP).
    AGRICULTURAL_HARVESTER: Combine harvester.
    FORKLIFT_DIESEL: Diesel-powered forklift.
    FORKLIFT_LPG: LPG-powered forklift.
    GENERATOR_SMALL: Portable generator (<50 kW).
    GENERATOR_MEDIUM: Stationary generator (50-500 kW).
    GENERATOR_LARGE: Large generator (>500 kW).
    """

    CONSTRUCTION_EXCAVATOR = "CONSTRUCTION_EXCAVATOR"
    CONSTRUCTION_BULLDOZER = "CONSTRUCTION_BULLDOZER"
    CONSTRUCTION_CRANE = "CONSTRUCTION_CRANE"
    CONSTRUCTION_LOADER = "CONSTRUCTION_LOADER"
    AGRICULTURAL_TRACTOR = "AGRICULTURAL_TRACTOR"
    AGRICULTURAL_HARVESTER = "AGRICULTURAL_HARVESTER"
    FORKLIFT_DIESEL = "FORKLIFT_DIESEL"
    FORKLIFT_LPG = "FORKLIFT_LPG"
    GENERATOR_SMALL = "GENERATOR_SMALL"
    GENERATOR_MEDIUM = "GENERATOR_MEDIUM"
    GENERATOR_LARGE = "GENERATOR_LARGE"


class MarineVesselType(str, Enum):
    """Marine vessel type for tonne-km emission factors.

    BARGE: Inland waterway barge.
    RIVER_BOAT: Inland river vessel.
    COASTAL_TANKER: Coastal tanker vessel.
    COASTAL_CARGO: Coastal general cargo vessel.
    CONTAINER_FEEDER: Short-sea container feeder.
    OCEAN_CONTAINER: Deep-sea container ship.
    OCEAN_BULK: Ocean bulk carrier.
    OCEAN_TANKER: Ocean oil/chemical tanker.
    """

    BARGE = "BARGE"
    RIVER_BOAT = "RIVER_BOAT"
    COASTAL_TANKER = "COASTAL_TANKER"
    COASTAL_CARGO = "COASTAL_CARGO"
    CONTAINER_FEEDER = "CONTAINER_FEEDER"
    OCEAN_CONTAINER = "OCEAN_CONTAINER"
    OCEAN_BULK = "OCEAN_BULK"
    OCEAN_TANKER = "OCEAN_TANKER"


class AircraftType(str, Enum):
    """Aircraft type for aviation emission factors.

    LIGHT_JET: Light business jet (e.g. Citation CJ4, Phenom 300).
    MIDSIZE_JET: Midsize business jet (e.g. Citation Latitude, Challenger 350).
    HEAVY_JET: Heavy/long-range business jet (e.g. Global 7500, Gulfstream G700).
    HELICOPTER_LIGHT: Light helicopter (e.g. R44, EC130).
    HELICOPTER_MEDIUM: Medium helicopter (e.g. AW139, S-76).
    HELICOPTER_HEAVY: Heavy helicopter (e.g. S-92, AW101).
    TURBOPROP_SMALL: Small turboprop (e.g. King Air 250).
    TURBOPROP_LARGE: Large turboprop (e.g. ATR 72, Dash 8).
    """

    LIGHT_JET = "LIGHT_JET"
    MIDSIZE_JET = "MIDSIZE_JET"
    HEAVY_JET = "HEAVY_JET"
    HELICOPTER_LIGHT = "HELICOPTER_LIGHT"
    HELICOPTER_MEDIUM = "HELICOPTER_MEDIUM"
    HELICOPTER_HEAVY = "HELICOPTER_HEAVY"
    TURBOPROP_SMALL = "TURBOPROP_SMALL"
    TURBOPROP_LARGE = "TURBOPROP_LARGE"


# ===========================================================================
# Default Fuel Economy Database (L/100km)
# Sources: EPA MOVES, ICCT, manufacturer data
# ===========================================================================

_DEFAULT_FUEL_ECONOMY: Dict[str, Decimal] = {
    VehicleType.PASSENGER_CAR_GASOLINE.value: Decimal("8.5"),
    VehicleType.PASSENGER_CAR_DIESEL.value: Decimal("6.5"),
    VehicleType.PASSENGER_CAR_HYBRID.value: Decimal("5.0"),
    VehicleType.PASSENGER_CAR_PHEV.value: Decimal("2.5"),
    VehicleType.LIGHT_DUTY_TRUCK_GASOLINE.value: Decimal("11.5"),
    VehicleType.LIGHT_DUTY_TRUCK_DIESEL.value: Decimal("9.5"),
    VehicleType.MEDIUM_DUTY_TRUCK_GASOLINE.value: Decimal("16.0"),
    VehicleType.MEDIUM_DUTY_TRUCK_DIESEL.value: Decimal("14.0"),
    VehicleType.HEAVY_DUTY_TRUCK_DIESEL.value: Decimal("32.0"),
    VehicleType.BUS_DIESEL.value: Decimal("30.0"),
    VehicleType.BUS_CNG.value: Decimal("40.0"),
    VehicleType.MOTORCYCLE.value: Decimal("4.5"),
    VehicleType.VAN_LCV.value: Decimal("9.0"),
}

# Off-road equipment fuel consumption rates (L/hour)
_EQUIPMENT_FUEL_RATES: Dict[str, Decimal] = {
    VehicleType.CONSTRUCTION_EQUIPMENT.value: Decimal("25.0"),
    VehicleType.AGRICULTURAL_EQUIPMENT.value: Decimal("18.0"),
    VehicleType.FORKLIFT.value: Decimal("4.0"),
}


# ===========================================================================
# Distance Emission Factors (g CO2e/km)
# Sources: GHG Protocol, DEFRA, EPA
# ===========================================================================

_DISTANCE_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    VehicleType.PASSENGER_CAR_GASOLINE.value: {
        "CO2e": Decimal("192"),
        "CO2": Decimal("186"),
        "CH4": Decimal("0.8"),
        "N2O": Decimal("5.2"),
        "source": "GHG Protocol Mobile Guidance; DEFRA 2025",
    },
    VehicleType.PASSENGER_CAR_DIESEL.value: {
        "CO2e": Decimal("171"),
        "CO2": Decimal("168"),
        "CH4": Decimal("0.1"),
        "N2O": Decimal("2.9"),
        "source": "GHG Protocol Mobile Guidance; DEFRA 2025",
    },
    VehicleType.PASSENGER_CAR_HYBRID.value: {
        "CO2e": Decimal("115"),
        "CO2": Decimal("112"),
        "CH4": Decimal("0.3"),
        "N2O": Decimal("2.7"),
        "source": "GHG Protocol Mobile Guidance; DEFRA 2025",
    },
    VehicleType.PASSENGER_CAR_PHEV.value: {
        "CO2e": Decimal("58"),
        "CO2": Decimal("56"),
        "CH4": Decimal("0.2"),
        "N2O": Decimal("1.8"),
        "source": "GHG Protocol Mobile Guidance; ICCT 2024",
    },
    VehicleType.LIGHT_DUTY_TRUCK_GASOLINE.value: {
        "CO2e": Decimal("265"),
        "CO2": Decimal("257"),
        "CH4": Decimal("1.1"),
        "N2O": Decimal("6.9"),
        "source": "GHG Protocol Mobile Guidance; EPA MOVES",
    },
    VehicleType.LIGHT_DUTY_TRUCK_DIESEL.value: {
        "CO2e": Decimal("230"),
        "CO2": Decimal("226"),
        "CH4": Decimal("0.2"),
        "N2O": Decimal("3.8"),
        "source": "GHG Protocol Mobile Guidance; EPA MOVES",
    },
    VehicleType.MEDIUM_DUTY_TRUCK_GASOLINE.value: {
        "CO2e": Decimal("370"),
        "CO2": Decimal("359"),
        "CH4": Decimal("1.5"),
        "N2O": Decimal("9.5"),
        "source": "GHG Protocol Mobile Guidance; EPA MOVES",
    },
    VehicleType.MEDIUM_DUTY_TRUCK_DIESEL.value: {
        "CO2e": Decimal("340"),
        "CO2": Decimal("334"),
        "CH4": Decimal("0.3"),
        "N2O": Decimal("5.7"),
        "source": "GHG Protocol Mobile Guidance; EPA MOVES",
    },
    VehicleType.HEAVY_DUTY_TRUCK_DIESEL.value: {
        "CO2e": Decimal("850"),
        "CO2": Decimal("836"),
        "CH4": Decimal("0.5"),
        "N2O": Decimal("13.5"),
        "source": "GHG Protocol Mobile Guidance; EPA MOVES",
    },
    VehicleType.BUS_DIESEL.value: {
        "CO2e": Decimal("800"),
        "CO2": Decimal("787"),
        "CH4": Decimal("0.5"),
        "N2O": Decimal("12.5"),
        "source": "GHG Protocol Mobile Guidance; DEFRA 2025",
    },
    VehicleType.BUS_CNG.value: {
        "CO2e": Decimal("950"),
        "CO2": Decimal("840"),
        "CH4": Decimal("105"),
        "N2O": Decimal("5.0"),
        "source": "GHG Protocol Mobile Guidance; EPA AP-42",
    },
    VehicleType.MOTORCYCLE.value: {
        "CO2e": Decimal("103"),
        "CO2": Decimal("100"),
        "CH4": Decimal("0.6"),
        "N2O": Decimal("2.4"),
        "source": "GHG Protocol Mobile Guidance; DEFRA 2025",
    },
    VehicleType.VAN_LCV.value: {
        "CO2e": Decimal("210"),
        "CO2": Decimal("205"),
        "CH4": Decimal("0.5"),
        "N2O": Decimal("4.5"),
        "source": "GHG Protocol Mobile Guidance; DEFRA 2025",
    },
}


# ===========================================================================
# Vehicle Age Fuel Economy Degradation Factors
# Source: EPA MOVES, ICCT fleet studies
# ===========================================================================

# List of (max_age_exclusive, factor) pairs, searched in order.
_AGE_DEGRADATION_TABLE: List[Tuple[int, Decimal]] = [
    (3, Decimal("1.00")),    # 0-2 years: baseline
    (5, Decimal("1.02")),    # 3-4 years: +2%
    (8, Decimal("1.05")),    # 5-7 years: +5%
    (12, Decimal("1.10")),   # 8-11 years: +10%
]
_AGE_DEGRADATION_MAX: Decimal = Decimal("1.15")  # 12+ years: +15%


# ===========================================================================
# Load Factor Adjustment Multipliers
# Source: GHG Protocol, ICCT heavy-duty fuel consumption studies
# ===========================================================================

_LOAD_FACTOR_ADJUSTMENTS: Dict[str, Decimal] = {
    LoadFactor.EMPTY.value: Decimal("0.70"),
    LoadFactor.QUARTER_LOAD.value: Decimal("0.85"),
    LoadFactor.HALF_LOAD.value: Decimal("1.00"),
    LoadFactor.THREE_QUARTER_LOAD.value: Decimal("1.10"),
    LoadFactor.FULL_LOAD.value: Decimal("1.20"),
    LoadFactor.OVERLOADED.value: Decimal("1.35"),
}


# ===========================================================================
# Operating Hour Emission Factors (kg CO2/hour)
# Sources: EPA NONROAD, IPCC 2006 Vol 2 Ch 3
# ===========================================================================

_OPERATING_HOUR_FACTORS: Dict[str, Dict[str, Any]] = {
    EquipmentOperatingType.CONSTRUCTION_EXCAVATOR.value: {
        "co2_kg_per_hour": Decimal("45"),
        "ch4_g_per_hour": Decimal("3.2"),
        "n2o_g_per_hour": Decimal("1.8"),
        "fuel_l_per_hour": Decimal("17.0"),
        "typical_power_kw": Decimal("150"),
        "source": "EPA NONROAD; IPCC 2006 Vol 2 Ch 3",
    },
    EquipmentOperatingType.CONSTRUCTION_BULLDOZER.value: {
        "co2_kg_per_hour": Decimal("55"),
        "ch4_g_per_hour": Decimal("3.8"),
        "n2o_g_per_hour": Decimal("2.2"),
        "fuel_l_per_hour": Decimal("21.0"),
        "typical_power_kw": Decimal("185"),
        "source": "EPA NONROAD; IPCC 2006 Vol 2 Ch 3",
    },
    EquipmentOperatingType.CONSTRUCTION_CRANE.value: {
        "co2_kg_per_hour": Decimal("38"),
        "ch4_g_per_hour": Decimal("2.6"),
        "n2o_g_per_hour": Decimal("1.5"),
        "fuel_l_per_hour": Decimal("14.5"),
        "typical_power_kw": Decimal("130"),
        "source": "EPA NONROAD; IPCC 2006 Vol 2 Ch 3",
    },
    EquipmentOperatingType.CONSTRUCTION_LOADER.value: {
        "co2_kg_per_hour": Decimal("40"),
        "ch4_g_per_hour": Decimal("2.8"),
        "n2o_g_per_hour": Decimal("1.6"),
        "fuel_l_per_hour": Decimal("15.0"),
        "typical_power_kw": Decimal("140"),
        "source": "EPA NONROAD; IPCC 2006 Vol 2 Ch 3",
    },
    EquipmentOperatingType.AGRICULTURAL_TRACTOR.value: {
        "co2_kg_per_hour": Decimal("35"),
        "ch4_g_per_hour": Decimal("2.4"),
        "n2o_g_per_hour": Decimal("1.4"),
        "fuel_l_per_hour": Decimal("13.5"),
        "typical_power_kw": Decimal("120"),
        "source": "EPA NONROAD; IPCC 2006 Vol 2 Ch 3",
    },
    EquipmentOperatingType.AGRICULTURAL_HARVESTER.value: {
        "co2_kg_per_hour": Decimal("60"),
        "ch4_g_per_hour": Decimal("4.2"),
        "n2o_g_per_hour": Decimal("2.4"),
        "fuel_l_per_hour": Decimal("23.0"),
        "typical_power_kw": Decimal("200"),
        "source": "EPA NONROAD; IPCC 2006 Vol 2 Ch 3",
    },
    EquipmentOperatingType.FORKLIFT_DIESEL.value: {
        "co2_kg_per_hour": Decimal("8"),
        "ch4_g_per_hour": Decimal("0.6"),
        "n2o_g_per_hour": Decimal("0.3"),
        "fuel_l_per_hour": Decimal("3.0"),
        "typical_power_kw": Decimal("35"),
        "source": "EPA NONROAD; manufacturer data",
    },
    EquipmentOperatingType.FORKLIFT_LPG.value: {
        "co2_kg_per_hour": Decimal("7"),
        "ch4_g_per_hour": Decimal("1.0"),
        "n2o_g_per_hour": Decimal("0.2"),
        "fuel_l_per_hour": Decimal("3.5"),
        "typical_power_kw": Decimal("35"),
        "source": "EPA NONROAD; manufacturer data",
    },
    EquipmentOperatingType.GENERATOR_SMALL.value: {
        "co2_kg_per_hour": Decimal("12"),
        "ch4_g_per_hour": Decimal("0.8"),
        "n2o_g_per_hour": Decimal("0.4"),
        "fuel_l_per_hour": Decimal("4.5"),
        "typical_power_kw": Decimal("25"),
        "source": "EPA NONROAD; manufacturer data",
    },
    EquipmentOperatingType.GENERATOR_MEDIUM.value: {
        "co2_kg_per_hour": Decimal("65"),
        "ch4_g_per_hour": Decimal("4.5"),
        "n2o_g_per_hour": Decimal("2.5"),
        "fuel_l_per_hour": Decimal("25.0"),
        "typical_power_kw": Decimal("250"),
        "source": "EPA NONROAD; manufacturer data",
    },
    EquipmentOperatingType.GENERATOR_LARGE.value: {
        "co2_kg_per_hour": Decimal("200"),
        "ch4_g_per_hour": Decimal("14.0"),
        "n2o_g_per_hour": Decimal("7.0"),
        "fuel_l_per_hour": Decimal("75.0"),
        "typical_power_kw": Decimal("750"),
        "source": "EPA NONROAD; manufacturer data",
    },
}


# ===========================================================================
# Marine Emission Factors (g CO2/tonne-km)
# Sources: IMO Fourth GHG Study 2020, GLEC Framework, DEFRA
# ===========================================================================

_MARINE_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    MarineVesselType.BARGE.value: {
        "co2_g_per_tonne_km": Decimal("33"),
        "ch4_g_per_tonne_km": Decimal("0.02"),
        "n2o_g_per_tonne_km": Decimal("0.01"),
        "co2e_g_per_tonne_km": Decimal("34"),
        "source": "IMO Fourth GHG Study 2020; GLEC Framework",
    },
    MarineVesselType.RIVER_BOAT.value: {
        "co2_g_per_tonne_km": Decimal("35"),
        "ch4_g_per_tonne_km": Decimal("0.02"),
        "n2o_g_per_tonne_km": Decimal("0.01"),
        "co2e_g_per_tonne_km": Decimal("36"),
        "source": "IMO Fourth GHG Study 2020; GLEC Framework",
    },
    MarineVesselType.COASTAL_TANKER.value: {
        "co2_g_per_tonne_km": Decimal("16"),
        "ch4_g_per_tonne_km": Decimal("0.01"),
        "n2o_g_per_tonne_km": Decimal("0.005"),
        "co2e_g_per_tonne_km": Decimal("17"),
        "source": "IMO Fourth GHG Study 2020; GLEC Framework",
    },
    MarineVesselType.COASTAL_CARGO.value: {
        "co2_g_per_tonne_km": Decimal("18"),
        "ch4_g_per_tonne_km": Decimal("0.01"),
        "n2o_g_per_tonne_km": Decimal("0.005"),
        "co2e_g_per_tonne_km": Decimal("19"),
        "source": "IMO Fourth GHG Study 2020; GLEC Framework",
    },
    MarineVesselType.CONTAINER_FEEDER.value: {
        "co2_g_per_tonne_km": Decimal("14"),
        "ch4_g_per_tonne_km": Decimal("0.01"),
        "n2o_g_per_tonne_km": Decimal("0.005"),
        "co2e_g_per_tonne_km": Decimal("15"),
        "source": "IMO Fourth GHG Study 2020; GLEC Framework",
    },
    MarineVesselType.OCEAN_CONTAINER.value: {
        "co2_g_per_tonne_km": Decimal("8"),
        "ch4_g_per_tonne_km": Decimal("0.005"),
        "n2o_g_per_tonne_km": Decimal("0.003"),
        "co2e_g_per_tonne_km": Decimal("9"),
        "source": "IMO Fourth GHG Study 2020; GLEC Framework",
    },
    MarineVesselType.OCEAN_BULK.value: {
        "co2_g_per_tonne_km": Decimal("5"),
        "ch4_g_per_tonne_km": Decimal("0.003"),
        "n2o_g_per_tonne_km": Decimal("0.002"),
        "co2e_g_per_tonne_km": Decimal("6"),
        "source": "IMO Fourth GHG Study 2020; GLEC Framework",
    },
    MarineVesselType.OCEAN_TANKER.value: {
        "co2_g_per_tonne_km": Decimal("6"),
        "ch4_g_per_tonne_km": Decimal("0.004"),
        "n2o_g_per_tonne_km": Decimal("0.002"),
        "co2e_g_per_tonne_km": Decimal("7"),
        "source": "IMO Fourth GHG Study 2020; GLEC Framework",
    },
}


# ===========================================================================
# Aviation Emission Factors (g CO2/km)
# Sources: ICAO Carbon Calculator, DEFRA, EEA EMEP/EEA Guidebook
# ===========================================================================

_AVIATION_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    AircraftType.LIGHT_JET.value: {
        "co2_g_per_km": Decimal("3500"),
        "ch4_g_per_km": Decimal("0.1"),
        "n2o_g_per_km": Decimal("0.1"),
        "co2e_g_per_km": Decimal("3530"),
        "typical_passengers": Decimal("6"),
        "co2e_g_per_pax_km": Decimal("588"),
        "source": "ICAO Carbon Calculator; EEA EMEP",
    },
    AircraftType.MIDSIZE_JET.value: {
        "co2_g_per_km": Decimal("5000"),
        "ch4_g_per_km": Decimal("0.15"),
        "n2o_g_per_km": Decimal("0.15"),
        "co2e_g_per_km": Decimal("5045"),
        "typical_passengers": Decimal("9"),
        "co2e_g_per_pax_km": Decimal("561"),
        "source": "ICAO Carbon Calculator; EEA EMEP",
    },
    AircraftType.HEAVY_JET.value: {
        "co2_g_per_km": Decimal("7000"),
        "ch4_g_per_km": Decimal("0.2"),
        "n2o_g_per_km": Decimal("0.2"),
        "co2e_g_per_km": Decimal("7060"),
        "typical_passengers": Decimal("14"),
        "co2e_g_per_pax_km": Decimal("504"),
        "source": "ICAO Carbon Calculator; EEA EMEP",
    },
    AircraftType.HELICOPTER_LIGHT.value: {
        "co2_g_per_km": Decimal("500"),
        "ch4_g_per_km": Decimal("0.03"),
        "n2o_g_per_km": Decimal("0.02"),
        "co2e_g_per_km": Decimal("507"),
        "typical_passengers": Decimal("3"),
        "co2e_g_per_pax_km": Decimal("169"),
        "source": "EEA EMEP/EEA Guidebook; manufacturer data",
    },
    AircraftType.HELICOPTER_MEDIUM.value: {
        "co2_g_per_km": Decimal("850"),
        "ch4_g_per_km": Decimal("0.05"),
        "n2o_g_per_km": Decimal("0.04"),
        "co2e_g_per_km": Decimal("863"),
        "typical_passengers": Decimal("8"),
        "co2e_g_per_pax_km": Decimal("108"),
        "source": "EEA EMEP/EEA Guidebook; manufacturer data",
    },
    AircraftType.HELICOPTER_HEAVY.value: {
        "co2_g_per_km": Decimal("1200"),
        "ch4_g_per_km": Decimal("0.07"),
        "n2o_g_per_km": Decimal("0.05"),
        "co2e_g_per_km": Decimal("1216"),
        "typical_passengers": Decimal("16"),
        "co2e_g_per_pax_km": Decimal("76"),
        "source": "EEA EMEP/EEA Guidebook; manufacturer data",
    },
    AircraftType.TURBOPROP_SMALL.value: {
        "co2_g_per_km": Decimal("200"),
        "ch4_g_per_km": Decimal("0.01"),
        "n2o_g_per_km": Decimal("0.01"),
        "co2e_g_per_km": Decimal("206"),
        "typical_passengers": Decimal("7"),
        "co2e_g_per_pax_km": Decimal("29"),
        "source": "ICAO Carbon Calculator; EEA EMEP",
    },
    AircraftType.TURBOPROP_LARGE.value: {
        "co2_g_per_km": Decimal("400"),
        "ch4_g_per_km": Decimal("0.02"),
        "n2o_g_per_km": Decimal("0.02"),
        "co2e_g_per_km": Decimal("412"),
        "typical_passengers": Decimal("50"),
        "co2e_g_per_pax_km": Decimal("8"),
        "source": "ICAO Carbon Calculator; EEA EMEP",
    },
}


# ===========================================================================
# Unit Conversion Constants
# ===========================================================================

# Distance conversion factors to km
_DISTANCE_TO_KM: Dict[str, Decimal] = {
    DistanceUnit.KM.value: Decimal("1.0"),
    DistanceUnit.MI.value: Decimal("1.60934"),
    DistanceUnit.NM.value: Decimal("1.852"),
}

# Fuel economy conversion constants
_US_GALLON_TO_LITRES: Decimal = Decimal("3.78541")
_UK_GALLON_TO_LITRES: Decimal = Decimal("4.54609")
_KM_PER_MILE: Decimal = Decimal("1.60934")

# Fuel density (kg/L) for fuel consumption to energy conversions
_FUEL_DENSITIES: Dict[str, Decimal] = {
    FuelType.GASOLINE.value: Decimal("0.745"),
    FuelType.DIESEL.value: Decimal("0.832"),
    FuelType.CNG.value: Decimal("0.128"),
    FuelType.LPG.value: Decimal("0.510"),
    FuelType.BIODIESEL.value: Decimal("0.880"),
    FuelType.ETHANOL.value: Decimal("0.789"),
    FuelType.JET_FUEL.value: Decimal("0.804"),
    FuelType.MARINE_DIESEL.value: Decimal("0.850"),
    FuelType.MARINE_HFO.value: Decimal("0.960"),
    FuelType.AVGAS.value: Decimal("0.721"),
}

# Default fuel type for each vehicle type
_DEFAULT_FUEL_TYPE: Dict[str, str] = {
    VehicleType.PASSENGER_CAR_GASOLINE.value: FuelType.GASOLINE.value,
    VehicleType.PASSENGER_CAR_DIESEL.value: FuelType.DIESEL.value,
    VehicleType.PASSENGER_CAR_HYBRID.value: FuelType.GASOLINE.value,
    VehicleType.PASSENGER_CAR_PHEV.value: FuelType.GASOLINE.value,
    VehicleType.LIGHT_DUTY_TRUCK_GASOLINE.value: FuelType.GASOLINE.value,
    VehicleType.LIGHT_DUTY_TRUCK_DIESEL.value: FuelType.DIESEL.value,
    VehicleType.MEDIUM_DUTY_TRUCK_GASOLINE.value: FuelType.GASOLINE.value,
    VehicleType.MEDIUM_DUTY_TRUCK_DIESEL.value: FuelType.DIESEL.value,
    VehicleType.HEAVY_DUTY_TRUCK_DIESEL.value: FuelType.DIESEL.value,
    VehicleType.BUS_DIESEL.value: FuelType.DIESEL.value,
    VehicleType.BUS_CNG.value: FuelType.CNG.value,
    VehicleType.MOTORCYCLE.value: FuelType.GASOLINE.value,
    VehicleType.VAN_LCV.value: FuelType.DIESEL.value,
    VehicleType.CONSTRUCTION_EQUIPMENT.value: FuelType.DIESEL.value,
    VehicleType.AGRICULTURAL_EQUIPMENT.value: FuelType.DIESEL.value,
    VehicleType.FORKLIFT.value: FuelType.DIESEL.value,
    VehicleType.MARINE_INLAND.value: FuelType.MARINE_DIESEL.value,
    VehicleType.MARINE_COASTAL.value: FuelType.MARINE_DIESEL.value,
    VehicleType.MARINE_OCEAN.value: FuelType.MARINE_HFO.value,
    VehicleType.AVIATION_CORPORATE_JET.value: FuelType.JET_FUEL.value,
    VehicleType.AVIATION_HELICOPTER.value: FuelType.JET_FUEL.value,
    VehicleType.AVIATION_TURBOPROP.value: FuelType.JET_FUEL.value,
}


# ===========================================================================
# Dataclasses for results
# ===========================================================================


@dataclass
class FuelEstimationResult:
    """Result of distance-to-fuel estimation with provenance.

    Attributes:
        result_id: Unique identifier for this estimation.
        vehicle_type: Vehicle type used for estimation.
        distance_km: Distance in kilometres.
        fuel_type: Fuel type applied.
        base_fuel_economy_l_per_100km: Base fuel economy before adjustments.
        adjusted_fuel_economy_l_per_100km: Fuel economy after age and load
            adjustments.
        vehicle_age_years: Vehicle age in years.
        age_degradation_factor: Multiplier applied for vehicle age.
        load_factor: Load factor classification.
        load_factor_adjustment: Multiplier applied for load factor.
        fuel_consumed_litres: Estimated fuel consumption in litres.
        fuel_consumed_kg: Estimated fuel consumption in kilograms.
        estimated_co2e_g: Estimated CO2e emissions in grams (distance-based).
        estimated_co2e_kg: Estimated CO2e emissions in kilograms.
        estimation_method: Method used ("DISTANCE_BASED" or "FUEL_ECONOMY").
        provenance_hash: SHA-256 hash of the estimation.
        timestamp: UTC ISO-formatted timestamp.
        metadata: Additional metadata dictionary.
    """

    result_id: str
    vehicle_type: str
    distance_km: Decimal
    fuel_type: str
    base_fuel_economy_l_per_100km: Decimal
    adjusted_fuel_economy_l_per_100km: Decimal
    vehicle_age_years: int
    age_degradation_factor: Decimal
    load_factor: str
    load_factor_adjustment: Decimal
    fuel_consumed_litres: Decimal
    fuel_consumed_kg: Decimal
    estimated_co2e_g: Decimal
    estimated_co2e_kg: Decimal
    estimation_method: str
    provenance_hash: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the result to a plain dictionary."""
        return {
            "result_id": self.result_id,
            "vehicle_type": self.vehicle_type,
            "distance_km": str(self.distance_km),
            "fuel_type": self.fuel_type,
            "base_fuel_economy_l_per_100km": str(self.base_fuel_economy_l_per_100km),
            "adjusted_fuel_economy_l_per_100km": str(self.adjusted_fuel_economy_l_per_100km),
            "vehicle_age_years": self.vehicle_age_years,
            "age_degradation_factor": str(self.age_degradation_factor),
            "load_factor": self.load_factor,
            "load_factor_adjustment": str(self.load_factor_adjustment),
            "fuel_consumed_litres": str(self.fuel_consumed_litres),
            "fuel_consumed_kg": str(self.fuel_consumed_kg),
            "estimated_co2e_g": str(self.estimated_co2e_g),
            "estimated_co2e_kg": str(self.estimated_co2e_kg),
            "estimation_method": self.estimation_method,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class DistanceEmissionResult:
    """Result of distance-based emission factor lookup.

    Attributes:
        result_id: Unique identifier.
        vehicle_type: Vehicle type.
        fuel_type: Fuel type.
        co2e_g_per_km: Total CO2e emission factor in g/km.
        co2_g_per_km: CO2 component in g/km.
        ch4_g_per_km: CH4 component in g/km.
        n2o_g_per_km: N2O component in g/km.
        source: Data source reference.
        provenance_hash: SHA-256 hash.
        timestamp: UTC ISO-formatted timestamp.
    """

    result_id: str
    vehicle_type: str
    fuel_type: str
    co2e_g_per_km: Decimal
    co2_g_per_km: Decimal
    ch4_g_per_km: Decimal
    n2o_g_per_km: Decimal
    source: str
    provenance_hash: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "result_id": self.result_id,
            "vehicle_type": self.vehicle_type,
            "fuel_type": self.fuel_type,
            "co2e_g_per_km": str(self.co2e_g_per_km),
            "co2_g_per_km": str(self.co2_g_per_km),
            "ch4_g_per_km": str(self.ch4_g_per_km),
            "n2o_g_per_km": str(self.n2o_g_per_km),
            "source": self.source,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp,
        }


@dataclass
class OperatingHoursResult:
    """Result of operating-hours-based emission estimation.

    Attributes:
        result_id: Unique identifier.
        equipment_type: Equipment type.
        operating_hours: Number of operating hours.
        fuel_type: Fuel type used.
        co2_kg: CO2 emissions in kg.
        ch4_g: CH4 emissions in grams.
        n2o_g: N2O emissions in grams.
        co2e_kg: Total CO2e emissions in kg.
        fuel_consumed_litres: Estimated fuel consumption in litres.
        source: Data source reference.
        provenance_hash: SHA-256 hash.
        timestamp: UTC ISO-formatted timestamp.
    """

    result_id: str
    equipment_type: str
    operating_hours: Decimal
    fuel_type: str
    co2_kg: Decimal
    ch4_g: Decimal
    n2o_g: Decimal
    co2e_kg: Decimal
    fuel_consumed_litres: Decimal
    source: str
    provenance_hash: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "result_id": self.result_id,
            "equipment_type": self.equipment_type,
            "operating_hours": str(self.operating_hours),
            "fuel_type": self.fuel_type,
            "co2_kg": str(self.co2_kg),
            "ch4_g": str(self.ch4_g),
            "n2o_g": str(self.n2o_g),
            "co2e_kg": str(self.co2e_kg),
            "fuel_consumed_litres": str(self.fuel_consumed_litres),
            "source": self.source,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp,
        }


@dataclass
class MarineEmissionResult:
    """Result of marine emission factor lookup.

    Attributes:
        result_id: Unique identifier.
        vessel_type: Marine vessel type.
        cargo_tonnes: Cargo weight in tonnes.
        distance_km: Voyage distance in kilometres.
        co2_g: CO2 emissions in grams.
        ch4_g: CH4 emissions in grams.
        n2o_g: N2O emissions in grams.
        co2e_g: Total CO2e emissions in grams.
        co2e_kg: Total CO2e emissions in kilograms.
        emission_factor_g_per_tonne_km: The applied emission factor.
        source: Data source reference.
        provenance_hash: SHA-256 hash.
        timestamp: UTC ISO-formatted timestamp.
    """

    result_id: str
    vessel_type: str
    cargo_tonnes: Decimal
    distance_km: Decimal
    co2_g: Decimal
    ch4_g: Decimal
    n2o_g: Decimal
    co2e_g: Decimal
    co2e_kg: Decimal
    emission_factor_g_per_tonne_km: Decimal
    source: str
    provenance_hash: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "result_id": self.result_id,
            "vessel_type": self.vessel_type,
            "cargo_tonnes": str(self.cargo_tonnes),
            "distance_km": str(self.distance_km),
            "co2_g": str(self.co2_g),
            "ch4_g": str(self.ch4_g),
            "n2o_g": str(self.n2o_g),
            "co2e_g": str(self.co2e_g),
            "co2e_kg": str(self.co2e_kg),
            "emission_factor_g_per_tonne_km": str(self.emission_factor_g_per_tonne_km),
            "source": self.source,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp,
        }


@dataclass
class AviationEmissionResult:
    """Result of aviation emission factor lookup.

    Attributes:
        result_id: Unique identifier.
        aircraft_type: Aircraft type.
        passengers: Number of passengers.
        distance_km: Flight distance in kilometres.
        co2_g: CO2 emissions in grams.
        ch4_g: CH4 emissions in grams.
        n2o_g: N2O emissions in grams.
        co2e_g: Total CO2e emissions in grams.
        co2e_kg: Total CO2e emissions in kilograms.
        co2e_g_per_pax_km: Per-passenger-km emission factor.
        source: Data source reference.
        provenance_hash: SHA-256 hash.
        timestamp: UTC ISO-formatted timestamp.
    """

    result_id: str
    aircraft_type: str
    passengers: int
    distance_km: Decimal
    co2_g: Decimal
    ch4_g: Decimal
    n2o_g: Decimal
    co2e_g: Decimal
    co2e_kg: Decimal
    co2e_g_per_pax_km: Decimal
    source: str
    provenance_hash: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "result_id": self.result_id,
            "aircraft_type": self.aircraft_type,
            "passengers": self.passengers,
            "distance_km": str(self.distance_km),
            "co2_g": str(self.co2_g),
            "ch4_g": str(self.ch4_g),
            "n2o_g": str(self.n2o_g),
            "co2e_g": str(self.co2e_g),
            "co2e_kg": str(self.co2e_kg),
            "co2e_g_per_pax_km": str(self.co2e_g_per_pax_km),
            "source": self.source,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp,
        }


# ===========================================================================
# DistanceEstimatorEngine
# ===========================================================================


class DistanceEstimatorEngine:
    """Distance-based emission estimation engine for mobile combustion.

    Provides deterministic, zero-hallucination distance-to-fuel and
    distance-to-emissions estimation for on-road vehicles, off-road
    equipment, marine vessels, and aviation sources. All arithmetic uses
    Python Decimal for bit-perfect reproducibility.

    The engine supports:
        - 13 on-road vehicle types with default fuel economy tables
        - 3 off-road equipment categories with operating-hour factors
        - 8 marine vessel types with tonne-km emission factors
        - 8 aircraft types with per-km and per-passenger-km factors
        - 5-tier vehicle age fuel economy degradation
        - 6-level load factor adjustment
        - Distance unit conversions (km, mi, nm)
        - Fuel economy unit conversions (L/100km, mpg US, mpg UK, km/L)
        - SHA-256 provenance hash for every estimation

    Thread Safety:
        All mutable state (_estimation_history, _custom_fuel_economies)
        is protected by a reentrant lock.

    Example:
        >>> engine = DistanceEstimatorEngine()
        >>> result = engine.estimate_fuel_from_distance(
        ...     vehicle_type="PASSENGER_CAR_GASOLINE",
        ...     distance_km=Decimal("15000"),
        ... )
        >>> print(result.fuel_consumed_litres)
    """

    def __init__(self) -> None:
        """Initialize the DistanceEstimatorEngine.

        Loads default fuel economy tables and initialises internal state
        for custom rate registration and estimation history tracking.
        """
        self._custom_fuel_economies: Dict[str, Decimal] = {}
        self._estimation_history: List[FuelEstimationResult] = []
        self._lock: threading.RLock = threading.RLock()

        logger.info(
            "DistanceEstimatorEngine initialized: "
            "%d on-road vehicle types, "
            "%d equipment types, "
            "%d marine vessel types, "
            "%d aircraft types",
            len(_DEFAULT_FUEL_ECONOMY),
            len(_OPERATING_HOUR_FACTORS),
            len(_MARINE_EMISSION_FACTORS),
            len(_AVIATION_EMISSION_FACTORS),
        )

    # ------------------------------------------------------------------
    # Public API: Fuel Estimation from Distance
    # ------------------------------------------------------------------

    def estimate_fuel_from_distance(
        self,
        vehicle_type: str,
        distance_km: Decimal,
        fuel_type: Optional[str] = None,
        vehicle_age_years: int = 0,
        load_factor: str = "HALF_LOAD",
    ) -> FuelEstimationResult:
        """Estimate fuel consumption from distance travelled.

        Computes estimated fuel consumption by applying the base fuel
        economy for the vehicle type, adjusted for vehicle age degradation
        and load factor.

        Formula:
            adjusted_economy = base_economy * age_factor * load_factor
            fuel_litres = distance_km * adjusted_economy / 100

        Args:
            vehicle_type: Vehicle type string matching a VehicleType
                enum value (on-road types only).
            distance_km: Distance travelled in kilometres. Must be >= 0.
            fuel_type: Optional fuel type. Defaults to the standard fuel
                for the vehicle type.
            vehicle_age_years: Vehicle age in years. Must be >= 0.
                Defaults to 0.
            load_factor: Load factor string matching a LoadFactor enum
                value. Defaults to "HALF_LOAD".

        Returns:
            FuelEstimationResult with fuel consumption and emission estimates.

        Raises:
            ValueError: If vehicle_type is not a recognized on-road type,
                distance_km < 0, vehicle_age_years < 0, or load_factor
                is not recognized.
        """
        t_start = time.monotonic()

        # Validate and coerce inputs
        distance_km = _to_decimal(distance_km)
        if distance_km < Decimal("0"):
            raise ValueError(f"distance_km must be >= 0, got {distance_km}")
        if vehicle_age_years < 0:
            raise ValueError(
                f"vehicle_age_years must be >= 0, got {vehicle_age_years}"
            )

        self._validate_on_road_vehicle_type(vehicle_type)
        self._validate_load_factor(load_factor)

        if fuel_type is None:
            fuel_type = _DEFAULT_FUEL_TYPE.get(vehicle_type, FuelType.GASOLINE.value)
        self._validate_fuel_type(fuel_type)

        # Lookup base fuel economy
        base_economy = self.get_fuel_economy(vehicle_type)

        # Apply adjustments
        age_factor = self._get_age_degradation_factor(vehicle_age_years)
        load_adj = self._get_load_factor_adjustment(load_factor)

        adjusted_economy = (
            base_economy * age_factor * load_adj
        ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        # Compute fuel consumed
        fuel_litres = (
            distance_km * adjusted_economy / Decimal("100")
        ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        # Fuel mass
        fuel_density = _FUEL_DENSITIES.get(fuel_type, Decimal("0.745"))
        fuel_kg = (fuel_litres * fuel_density).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        # Distance-based emission estimate
        co2e_g = Decimal("0")
        if vehicle_type in _DISTANCE_EMISSION_FACTORS:
            ef = _DISTANCE_EMISSION_FACTORS[vehicle_type]["CO2e"]
            co2e_g = (distance_km * ef * age_factor * load_adj).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
        co2e_kg = (co2e_g / Decimal("1000")).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        # Provenance hash
        provenance_data = {
            "vehicle_type": vehicle_type,
            "distance_km": str(distance_km),
            "fuel_type": fuel_type,
            "vehicle_age_years": vehicle_age_years,
            "load_factor": load_factor,
            "base_economy": str(base_economy),
            "age_factor": str(age_factor),
            "load_adj": str(load_adj),
            "fuel_litres": str(fuel_litres),
            "co2e_g": str(co2e_g),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode("utf-8")
        ).hexdigest()

        timestamp = _utcnow().isoformat()

        result = FuelEstimationResult(
            result_id=f"de_{uuid4().hex[:12]}",
            vehicle_type=vehicle_type,
            distance_km=distance_km,
            fuel_type=fuel_type,
            base_fuel_economy_l_per_100km=base_economy,
            adjusted_fuel_economy_l_per_100km=adjusted_economy,
            vehicle_age_years=vehicle_age_years,
            age_degradation_factor=age_factor,
            load_factor=load_factor,
            load_factor_adjustment=load_adj,
            fuel_consumed_litres=fuel_litres,
            fuel_consumed_kg=fuel_kg,
            estimated_co2e_g=co2e_g,
            estimated_co2e_kg=co2e_kg,
            estimation_method="DISTANCE_BASED",
            provenance_hash=provenance_hash,
            timestamp=timestamp,
        )

        # Record in history
        with self._lock:
            self._estimation_history.append(result)

        # Provenance tracker integration
        self._record_provenance("fuel_estimation", result.result_id, provenance_data)

        # Metrics
        elapsed = time.monotonic() - t_start
        self._record_metrics(vehicle_type, elapsed)

        logger.debug(
            "Fuel estimated: vehicle=%s distance=%.1fkm "
            "fuel=%.1fL co2e=%.1fg age=%dyr load=%s in %.1fms",
            vehicle_type, distance_km, fuel_litres,
            co2e_g, vehicle_age_years, load_factor,
            elapsed * 1000,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: Distance Emission Factor Lookup
    # ------------------------------------------------------------------

    def get_distance_emission_factor(
        self,
        vehicle_type: str,
        fuel_type: Optional[str] = None,
    ) -> DistanceEmissionResult:
        """Get the distance-based emission factor for a vehicle type.

        Returns the g CO2e/km emission factor with per-gas breakdown
        for the specified vehicle type.

        Args:
            vehicle_type: Vehicle type string matching a VehicleType enum.
            fuel_type: Optional fuel type override. Defaults to standard
                fuel for the vehicle type.

        Returns:
            DistanceEmissionResult with per-gas emission factors.

        Raises:
            ValueError: If vehicle_type does not have a distance emission
                factor entry.
        """
        if vehicle_type not in _DISTANCE_EMISSION_FACTORS:
            raise ValueError(
                f"No distance emission factor for vehicle type '{vehicle_type}'. "
                f"Supported: {sorted(_DISTANCE_EMISSION_FACTORS.keys())}"
            )

        if fuel_type is None:
            fuel_type = _DEFAULT_FUEL_TYPE.get(vehicle_type, FuelType.GASOLINE.value)

        ef = _DISTANCE_EMISSION_FACTORS[vehicle_type]

        provenance_data = {
            "vehicle_type": vehicle_type,
            "fuel_type": fuel_type,
            "co2e_g_per_km": str(ef["CO2e"]),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return DistanceEmissionResult(
            result_id=f"def_{uuid4().hex[:12]}",
            vehicle_type=vehicle_type,
            fuel_type=fuel_type,
            co2e_g_per_km=ef["CO2e"],
            co2_g_per_km=ef["CO2"],
            ch4_g_per_km=ef["CH4"],
            n2o_g_per_km=ef["N2O"],
            source=ef["source"],
            provenance_hash=provenance_hash,
            timestamp=_utcnow().isoformat(),
        )

    # ------------------------------------------------------------------
    # Public API: Fuel Economy Lookup
    # ------------------------------------------------------------------

    def get_fuel_economy(self, vehicle_type: str) -> Decimal:
        """Get the base fuel economy for a vehicle type in L/100km.

        Checks custom-registered economies first, then falls back to
        the default database.

        Args:
            vehicle_type: Vehicle type string.

        Returns:
            Base fuel economy in L/100km as a Decimal.

        Raises:
            ValueError: If vehicle_type has no fuel economy entry.
        """
        with self._lock:
            if vehicle_type in self._custom_fuel_economies:
                return self._custom_fuel_economies[vehicle_type]

        if vehicle_type in _DEFAULT_FUEL_ECONOMY:
            return _DEFAULT_FUEL_ECONOMY[vehicle_type]

        raise ValueError(
            f"No fuel economy data for vehicle type '{vehicle_type}'. "
            f"Supported on-road types: {sorted(_DEFAULT_FUEL_ECONOMY.keys())}. "
            f"Use register_custom_fuel_economy() for custom types."
        )

    def register_custom_fuel_economy(
        self,
        vehicle_type: str,
        fuel_economy_l_per_100km: Decimal,
    ) -> None:
        """Register a custom fuel economy for a vehicle type.

        Overrides the default fuel economy for subsequent estimations
        using this vehicle type.

        Args:
            vehicle_type: Vehicle type identifier (may be custom).
            fuel_economy_l_per_100km: Fuel economy in L/100km.
                Must be > 0.

        Raises:
            ValueError: If fuel_economy_l_per_100km <= 0.
        """
        fuel_economy_l_per_100km = _to_decimal(fuel_economy_l_per_100km)
        if fuel_economy_l_per_100km <= Decimal("0"):
            raise ValueError(
                f"fuel_economy_l_per_100km must be > 0, "
                f"got {fuel_economy_l_per_100km}"
            )

        with self._lock:
            self._custom_fuel_economies[vehicle_type] = fuel_economy_l_per_100km

        logger.info(
            "Custom fuel economy registered: %s = %s L/100km",
            vehicle_type, fuel_economy_l_per_100km,
        )

    # ------------------------------------------------------------------
    # Public API: Fuel Economy Adjustments
    # ------------------------------------------------------------------

    def adjust_fuel_economy_for_age(
        self,
        base_economy: Decimal,
        vehicle_age_years: int,
    ) -> Decimal:
        """Adjust fuel economy for vehicle age degradation.

        Older vehicles have higher fuel consumption due to engine wear,
        drivetrain losses, and reduced efficiency of emissions controls.

        Age degradation factors:
            - 0-2 years: 1.00 (baseline)
            - 3-4 years: 1.02 (+2%)
            - 5-7 years: 1.05 (+5%)
            - 8-11 years: 1.10 (+10%)
            - 12+ years: 1.15 (+15%)

        Args:
            base_economy: Base fuel economy in L/100km. Must be > 0.
            vehicle_age_years: Vehicle age in years. Must be >= 0.

        Returns:
            Age-adjusted fuel economy in L/100km.

        Raises:
            ValueError: If base_economy <= 0 or vehicle_age_years < 0.
        """
        base_economy = _to_decimal(base_economy)
        if base_economy <= Decimal("0"):
            raise ValueError(
                f"base_economy must be > 0, got {base_economy}"
            )
        if vehicle_age_years < 0:
            raise ValueError(
                f"vehicle_age_years must be >= 0, got {vehicle_age_years}"
            )

        factor = self._get_age_degradation_factor(vehicle_age_years)

        return (base_economy * factor).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

    def adjust_for_load_factor(
        self,
        base_economy: Decimal,
        load_factor: str,
    ) -> Decimal:
        """Adjust fuel economy for vehicle load factor.

        Heavier loads increase fuel consumption; empty vehicles consume
        less fuel than at half-load baseline.

        Load factor adjustments:
            - EMPTY: 0.70 (-30%)
            - QUARTER_LOAD: 0.85 (-15%)
            - HALF_LOAD: 1.00 (baseline)
            - THREE_QUARTER_LOAD: 1.10 (+10%)
            - FULL_LOAD: 1.20 (+20%)
            - OVERLOADED: 1.35 (+35%)

        Args:
            base_economy: Base fuel economy in L/100km. Must be > 0.
            load_factor: Load factor string matching a LoadFactor enum.

        Returns:
            Load-adjusted fuel economy in L/100km.

        Raises:
            ValueError: If base_economy <= 0 or load_factor is not
                recognized.
        """
        base_economy = _to_decimal(base_economy)
        if base_economy <= Decimal("0"):
            raise ValueError(
                f"base_economy must be > 0, got {base_economy}"
            )
        self._validate_load_factor(load_factor)

        adjustment = self._get_load_factor_adjustment(load_factor)

        return (base_economy * adjustment).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

    # ------------------------------------------------------------------
    # Public API: Unit Conversions
    # ------------------------------------------------------------------

    def convert_distance(
        self,
        value: Decimal,
        from_unit: str,
        to_unit: str,
    ) -> Decimal:
        """Convert a distance value between units.

        Supported units: KM, MI, NM (nautical miles).

        Conversion factors:
            - 1 mile = 1.60934 km
            - 1 nautical mile = 1.852 km

        Args:
            value: Distance value to convert. Must be >= 0.
            from_unit: Source unit (KM, MI, NM).
            to_unit: Target unit (KM, MI, NM).

        Returns:
            Converted distance value.

        Raises:
            ValueError: If value < 0 or units are not recognized.
        """
        value = _to_decimal(value)
        if value < Decimal("0"):
            raise ValueError(f"Distance value must be >= 0, got {value}")

        self._validate_distance_unit(from_unit)
        self._validate_distance_unit(to_unit)

        if from_unit == to_unit:
            return value

        # Convert to km first, then to target unit
        km_value = value * _DISTANCE_TO_KM[from_unit]
        result = km_value / _DISTANCE_TO_KM[to_unit]

        return result.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def convert_fuel_economy(
        self,
        value: Decimal,
        from_unit: str,
        to_unit: str,
    ) -> Decimal:
        """Convert a fuel economy value between units.

        Supported units: L_PER_100KM, MPG_US, MPG_UK, KM_PER_L.

        Conversion logic:
            - L/100km to km/L: km_per_l = 100 / l_per_100km
            - L/100km to MPG US: mpg_us = 235.215 / l_per_100km
            - L/100km to MPG UK: mpg_uk = 282.481 / l_per_100km
            - MPG US to L/100km: l_per_100km = 235.215 / mpg_us
            - MPG UK to L/100km: l_per_100km = 282.481 / mpg_uk
            - km/L to L/100km: l_per_100km = 100 / km_per_l

        Args:
            value: Fuel economy value to convert. Must be > 0.
            from_unit: Source unit string matching FuelEconomyUnit.
            to_unit: Target unit string matching FuelEconomyUnit.

        Returns:
            Converted fuel economy value.

        Raises:
            ValueError: If value <= 0 or units are not recognized.
        """
        value = _to_decimal(value)
        if value <= Decimal("0"):
            raise ValueError(
                f"Fuel economy value must be > 0, got {value}"
            )

        self._validate_fuel_economy_unit(from_unit)
        self._validate_fuel_economy_unit(to_unit)

        if from_unit == to_unit:
            return value

        # Conversion constants
        mpg_us_factor = Decimal("235.215")
        mpg_uk_factor = Decimal("282.481")

        # First convert to L/100km as intermediate
        l_per_100km: Decimal
        if from_unit == FuelEconomyUnit.L_PER_100KM.value:
            l_per_100km = value
        elif from_unit == FuelEconomyUnit.MPG_US.value:
            l_per_100km = mpg_us_factor / value
        elif from_unit == FuelEconomyUnit.MPG_UK.value:
            l_per_100km = mpg_uk_factor / value
        elif from_unit == FuelEconomyUnit.KM_PER_L.value:
            l_per_100km = Decimal("100") / value
        else:
            raise ValueError(f"Unsupported from_unit: {from_unit}")

        # Then convert from L/100km to target unit
        result: Decimal
        if to_unit == FuelEconomyUnit.L_PER_100KM.value:
            result = l_per_100km
        elif to_unit == FuelEconomyUnit.MPG_US.value:
            result = mpg_us_factor / l_per_100km
        elif to_unit == FuelEconomyUnit.MPG_UK.value:
            result = mpg_uk_factor / l_per_100km
        elif to_unit == FuelEconomyUnit.KM_PER_L.value:
            result = Decimal("100") / l_per_100km
        else:
            raise ValueError(f"Unsupported to_unit: {to_unit}")

        return result.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    # ------------------------------------------------------------------
    # Public API: Operating Hours Estimation (Off-Road)
    # ------------------------------------------------------------------

    def estimate_operating_hours_emissions(
        self,
        equipment_type: str,
        hours: Decimal,
        fuel_type: Optional[str] = None,
    ) -> OperatingHoursResult:
        """Estimate emissions from off-road equipment operating hours.

        Uses equipment-specific emission factors (kg CO2/hour, g CH4/hour,
        g N2O/hour) derived from EPA NONROAD and IPCC 2006 Vol 2 Ch 3.

        Formula:
            co2_kg = hours * co2_kg_per_hour
            ch4_g = hours * ch4_g_per_hour
            n2o_g = hours * n2o_g_per_hour
            co2e_kg = co2_kg + ch4_g * gwp_ch4 / 1000 + n2o_g * gwp_n2o / 1000

        Supported equipment types:
            CONSTRUCTION_EXCAVATOR, CONSTRUCTION_BULLDOZER,
            CONSTRUCTION_CRANE, CONSTRUCTION_LOADER,
            AGRICULTURAL_TRACTOR, AGRICULTURAL_HARVESTER,
            FORKLIFT_DIESEL, FORKLIFT_LPG,
            GENERATOR_SMALL, GENERATOR_MEDIUM, GENERATOR_LARGE

        Args:
            equipment_type: Equipment type string matching an
                EquipmentOperatingType enum value.
            hours: Number of operating hours. Must be >= 0.
            fuel_type: Optional fuel type override.

        Returns:
            OperatingHoursResult with emission estimates.

        Raises:
            ValueError: If equipment_type is not recognized or hours < 0.
        """
        t_start = time.monotonic()

        hours = _to_decimal(hours)
        if hours < Decimal("0"):
            raise ValueError(f"hours must be >= 0, got {hours}")

        self._validate_equipment_type(equipment_type)

        if fuel_type is None:
            # Determine default fuel type from equipment type
            if "LPG" in equipment_type:
                fuel_type = FuelType.LPG.value
            else:
                fuel_type = FuelType.DIESEL.value

        ef = _OPERATING_HOUR_FACTORS[equipment_type]

        # GWP values (IPCC AR5 used for consistency with GHG Protocol)
        gwp_ch4 = Decimal("28")
        gwp_n2o = Decimal("265")

        co2_kg = (hours * ef["co2_kg_per_hour"]).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        ch4_g = (hours * ef["ch4_g_per_hour"]).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        n2o_g = (hours * ef["n2o_g_per_hour"]).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        fuel_l = (hours * ef["fuel_l_per_hour"]).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        # co2e = co2 + ch4 * gwp / 1000 + n2o * gwp / 1000
        co2e_kg = (
            co2_kg
            + ch4_g * gwp_ch4 / Decimal("1000")
            + n2o_g * gwp_n2o / Decimal("1000")
        ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        provenance_data = {
            "equipment_type": equipment_type,
            "hours": str(hours),
            "fuel_type": fuel_type,
            "co2_kg": str(co2_kg),
            "ch4_g": str(ch4_g),
            "n2o_g": str(n2o_g),
            "co2e_kg": str(co2e_kg),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode("utf-8")
        ).hexdigest()

        result = OperatingHoursResult(
            result_id=f"oh_{uuid4().hex[:12]}",
            equipment_type=equipment_type,
            operating_hours=hours,
            fuel_type=fuel_type,
            co2_kg=co2_kg,
            ch4_g=ch4_g,
            n2o_g=n2o_g,
            co2e_kg=co2e_kg,
            fuel_consumed_litres=fuel_l,
            source=ef["source"],
            provenance_hash=provenance_hash,
            timestamp=_utcnow().isoformat(),
        )

        elapsed = time.monotonic() - t_start
        logger.debug(
            "Operating hours estimated: equipment=%s hours=%.1f "
            "co2e=%.3fkg in %.1fms",
            equipment_type, hours, co2e_kg, elapsed * 1000,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: Marine Emission Factor
    # ------------------------------------------------------------------

    def get_marine_emission_factor(
        self,
        vessel_type: str,
        cargo_tonnes: Decimal,
        distance_km: Optional[Decimal] = None,
    ) -> MarineEmissionResult:
        """Get marine emission factor and optionally compute emissions.

        Uses IMO Fourth GHG Study 2020 and GLEC Framework tonne-km
        emission factors for marine vessel types.

        Formula:
            emissions_g = cargo_tonnes * distance_km * ef_g_per_tonne_km

        Supported vessel types:
            BARGE, RIVER_BOAT (inland: ~33 g CO2/tonne-km),
            COASTAL_TANKER, COASTAL_CARGO, CONTAINER_FEEDER (coastal: ~16),
            OCEAN_CONTAINER, OCEAN_BULK, OCEAN_TANKER (ocean: ~5-8)

        Args:
            vessel_type: Marine vessel type string matching a
                MarineVesselType enum value.
            cargo_tonnes: Cargo weight in metric tonnes. Must be >= 0.
            distance_km: Optional voyage distance in km. If provided,
                emissions are computed. If None, only the factor is
                returned with zero emissions.

        Returns:
            MarineEmissionResult with emission factor and computed emissions.

        Raises:
            ValueError: If vessel_type is not recognized, cargo_tonnes < 0,
                or distance_km < 0.
        """
        cargo_tonnes = _to_decimal(cargo_tonnes)
        if cargo_tonnes < Decimal("0"):
            raise ValueError(f"cargo_tonnes must be >= 0, got {cargo_tonnes}")

        self._validate_marine_vessel_type(vessel_type)

        if distance_km is not None:
            distance_km = _to_decimal(distance_km)
            if distance_km < Decimal("0"):
                raise ValueError(f"distance_km must be >= 0, got {distance_km}")
        else:
            distance_km = Decimal("0")

        ef = _MARINE_EMISSION_FACTORS[vessel_type]

        # GWP values (AR5)
        gwp_ch4 = Decimal("28")
        gwp_n2o = Decimal("265")

        tonne_km = cargo_tonnes * distance_km

        co2_g = (tonne_km * ef["co2_g_per_tonne_km"]).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        ch4_g = (tonne_km * ef["ch4_g_per_tonne_km"]).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        n2o_g = (tonne_km * ef["n2o_g_per_tonne_km"]).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        co2e_g = (
            co2_g
            + ch4_g * gwp_ch4
            + n2o_g * gwp_n2o
        ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        co2e_kg = (co2e_g / Decimal("1000")).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        provenance_data = {
            "vessel_type": vessel_type,
            "cargo_tonnes": str(cargo_tonnes),
            "distance_km": str(distance_km),
            "co2e_g": str(co2e_g),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return MarineEmissionResult(
            result_id=f"me_{uuid4().hex[:12]}",
            vessel_type=vessel_type,
            cargo_tonnes=cargo_tonnes,
            distance_km=distance_km,
            co2_g=co2_g,
            ch4_g=ch4_g,
            n2o_g=n2o_g,
            co2e_g=co2e_g,
            co2e_kg=co2e_kg,
            emission_factor_g_per_tonne_km=ef["co2e_g_per_tonne_km"],
            source=ef["source"],
            provenance_hash=provenance_hash,
            timestamp=_utcnow().isoformat(),
        )

    # ------------------------------------------------------------------
    # Public API: Aviation Emission Factor
    # ------------------------------------------------------------------

    def get_aviation_emission_factor(
        self,
        aircraft_type: str,
        passengers: int = 0,
        distance_km: Optional[Decimal] = None,
    ) -> AviationEmissionResult:
        """Get aviation emission factor and optionally compute emissions.

        Uses ICAO Carbon Calculator and EEA EMEP/EEA Guidebook emission
        factors for corporate/business aviation.

        Formula (total aircraft emissions):
            emissions_g = distance_km * co2e_g_per_km

        Per-passenger-km factor:
            co2e_g_per_pax_km = co2e_g_per_km / passengers

        Supported aircraft types:
            LIGHT_JET (3500 g CO2/km), MIDSIZE_JET (5000),
            HEAVY_JET (7000), HELICOPTER_LIGHT (500),
            HELICOPTER_MEDIUM (850), HELICOPTER_HEAVY (1200),
            TURBOPROP_SMALL (200), TURBOPROP_LARGE (400)

        Args:
            aircraft_type: Aircraft type string matching an AircraftType
                enum value.
            passengers: Number of passengers. If 0, the typical passenger
                count for the aircraft type is used for per-pax-km
                calculation.
            distance_km: Optional flight distance in km. If provided,
                total emissions are computed.

        Returns:
            AviationEmissionResult with emission factors and computed
            emissions.

        Raises:
            ValueError: If aircraft_type is not recognized, passengers < 0,
                or distance_km < 0.
        """
        if passengers < 0:
            raise ValueError(f"passengers must be >= 0, got {passengers}")

        self._validate_aircraft_type(aircraft_type)

        if distance_km is not None:
            distance_km = _to_decimal(distance_km)
            if distance_km < Decimal("0"):
                raise ValueError(
                    f"distance_km must be >= 0, got {distance_km}"
                )
        else:
            distance_km = Decimal("0")

        ef = _AVIATION_EMISSION_FACTORS[aircraft_type]

        # Determine passenger count for per-pax calculation
        pax = passengers if passengers > 0 else int(ef["typical_passengers"])

        co2_g = (distance_km * ef["co2_g_per_km"]).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        ch4_g = (distance_km * ef["ch4_g_per_km"]).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        n2o_g = (distance_km * ef["n2o_g_per_km"]).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        co2e_g = (distance_km * ef["co2e_g_per_km"]).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        co2e_kg = (co2e_g / Decimal("1000")).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        # Per-passenger-km factor
        if pax > 0 and distance_km > Decimal("0"):
            co2e_g_per_pax_km = (
                co2e_g / _to_decimal(pax) / distance_km
            ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        elif pax > 0:
            co2e_g_per_pax_km = (
                ef["co2e_g_per_km"] / _to_decimal(pax)
            ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        else:
            co2e_g_per_pax_km = ef.get(
                "co2e_g_per_pax_km", ef["co2e_g_per_km"]
            )

        provenance_data = {
            "aircraft_type": aircraft_type,
            "passengers": pax,
            "distance_km": str(distance_km),
            "co2e_g": str(co2e_g),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return AviationEmissionResult(
            result_id=f"av_{uuid4().hex[:12]}",
            aircraft_type=aircraft_type,
            passengers=pax,
            distance_km=distance_km,
            co2_g=co2_g,
            ch4_g=ch4_g,
            n2o_g=n2o_g,
            co2e_g=co2e_g,
            co2e_kg=co2e_kg,
            co2e_g_per_pax_km=co2e_g_per_pax_km,
            source=ef["source"],
            provenance_hash=provenance_hash,
            timestamp=_utcnow().isoformat(),
        )

    # ------------------------------------------------------------------
    # Public API: Batch Estimation
    # ------------------------------------------------------------------

    def estimate_fleet_emissions(
        self,
        vehicles: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Estimate total emissions for a fleet of vehicles.

        Processes a list of vehicle records, each specifying vehicle_type,
        distance_km, and optional adjustments, returning individual results
        and fleet-level aggregates.

        Each vehicle dict should contain:
            - vehicle_type (str): Required.
            - distance_km (Decimal/float/int): Required.
            - fuel_type (str): Optional.
            - vehicle_age_years (int): Optional, defaults to 0.
            - load_factor (str): Optional, defaults to "HALF_LOAD".

        Args:
            vehicles: List of vehicle dictionaries.

        Returns:
            Dictionary with:
                - results: List of FuelEstimationResult.to_dict()
                - total_fuel_litres: Fleet total fuel in litres.
                - total_co2e_kg: Fleet total CO2e in kg.
                - total_co2e_tonnes: Fleet total CO2e in tonnes.
                - vehicle_count: Number of vehicles processed.
                - errors: List of error messages for failed records.
                - provenance_hash: SHA-256 hash of the fleet estimation.
        """
        results: List[Dict[str, Any]] = []
        errors: List[str] = []
        total_fuel = Decimal("0")
        total_co2e_kg = Decimal("0")

        for idx, vehicle in enumerate(vehicles):
            try:
                v_type = vehicle.get("vehicle_type", "")
                dist = _to_decimal(vehicle.get("distance_km", 0))
                f_type = vehicle.get("fuel_type")
                age = int(vehicle.get("vehicle_age_years", 0))
                load = vehicle.get("load_factor", "HALF_LOAD")

                result = self.estimate_fuel_from_distance(
                    vehicle_type=v_type,
                    distance_km=dist,
                    fuel_type=f_type,
                    vehicle_age_years=age,
                    load_factor=load,
                )
                results.append(result.to_dict())
                total_fuel += result.fuel_consumed_litres
                total_co2e_kg += result.estimated_co2e_kg

            except (ValueError, KeyError, TypeError) as exc:
                error_msg = f"Vehicle #{idx}: {str(exc)}"
                errors.append(error_msg)
                logger.warning("Fleet estimation error: %s", error_msg)

        total_co2e_tonnes = (total_co2e_kg / Decimal("1000")).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        fleet_provenance = {
            "vehicle_count": len(results),
            "total_fuel_litres": str(total_fuel),
            "total_co2e_kg": str(total_co2e_kg),
            "errors": len(errors),
        }
        fleet_hash = hashlib.sha256(
            json.dumps(fleet_provenance, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return {
            "results": results,
            "total_fuel_litres": str(total_fuel.quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )),
            "total_co2e_kg": str(total_co2e_kg.quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )),
            "total_co2e_tonnes": str(total_co2e_tonnes),
            "vehicle_count": len(results),
            "errors": errors,
            "provenance_hash": fleet_hash,
            "timestamp": _utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Public API: Listing Supported Types
    # ------------------------------------------------------------------

    def list_vehicle_types(self) -> List[str]:
        """List all supported on-road vehicle types.

        Returns:
            Sorted list of vehicle type identifiers with fuel economy data.
        """
        return sorted(_DEFAULT_FUEL_ECONOMY.keys())

    def list_equipment_types(self) -> List[str]:
        """List all supported off-road equipment types.

        Returns:
            Sorted list of equipment type identifiers.
        """
        return sorted(_OPERATING_HOUR_FACTORS.keys())

    def list_marine_vessel_types(self) -> List[str]:
        """List all supported marine vessel types.

        Returns:
            Sorted list of marine vessel type identifiers.
        """
        return sorted(_MARINE_EMISSION_FACTORS.keys())

    def list_aircraft_types(self) -> List[str]:
        """List all supported aircraft types.

        Returns:
            Sorted list of aircraft type identifiers.
        """
        return sorted(_AVIATION_EMISSION_FACTORS.keys())

    def get_estimation_history(self) -> List[FuelEstimationResult]:
        """Return a copy of the estimation history.

        Returns:
            List of all FuelEstimationResult objects produced by this
            engine instance.
        """
        with self._lock:
            return list(self._estimation_history)

    def clear_history(self) -> int:
        """Clear the estimation history.

        Returns:
            Number of records cleared.
        """
        with self._lock:
            count = len(self._estimation_history)
            self._estimation_history.clear()
        logger.info("Estimation history cleared: %d records removed", count)
        return count

    # ------------------------------------------------------------------
    # Internal: Validation Methods
    # ------------------------------------------------------------------

    def _validate_on_road_vehicle_type(self, vehicle_type: str) -> None:
        """Validate that vehicle_type is a recognized on-road type.

        Args:
            vehicle_type: Vehicle type string to validate.

        Raises:
            ValueError: If not recognized or no fuel economy data.
        """
        all_types = set(_DEFAULT_FUEL_ECONOMY.keys())
        with self._lock:
            all_types.update(self._custom_fuel_economies.keys())

        if vehicle_type not in all_types:
            raise ValueError(
                f"Unrecognized on-road vehicle type '{vehicle_type}'. "
                f"Supported: {sorted(_DEFAULT_FUEL_ECONOMY.keys())}. "
                f"Use register_custom_fuel_economy() for custom types."
            )

    def _validate_fuel_type(self, fuel_type: str) -> None:
        """Validate that fuel_type is a recognized fuel type.

        Args:
            fuel_type: Fuel type string to validate.

        Raises:
            ValueError: If not recognized.
        """
        valid = {ft.value for ft in FuelType}
        if fuel_type not in valid:
            raise ValueError(
                f"Unrecognized fuel type '{fuel_type}'. "
                f"Supported: {sorted(valid)}"
            )

    def _validate_load_factor(self, load_factor: str) -> None:
        """Validate that load_factor is a recognized load factor.

        Args:
            load_factor: Load factor string to validate.

        Raises:
            ValueError: If not recognized.
        """
        if load_factor not in _LOAD_FACTOR_ADJUSTMENTS:
            raise ValueError(
                f"Unrecognized load factor '{load_factor}'. "
                f"Supported: {sorted(_LOAD_FACTOR_ADJUSTMENTS.keys())}"
            )

    def _validate_distance_unit(self, unit: str) -> None:
        """Validate that unit is a recognized distance unit.

        Args:
            unit: Distance unit string to validate.

        Raises:
            ValueError: If not recognized.
        """
        if unit not in _DISTANCE_TO_KM:
            raise ValueError(
                f"Unrecognized distance unit '{unit}'. "
                f"Supported: {sorted(_DISTANCE_TO_KM.keys())}"
            )

    def _validate_fuel_economy_unit(self, unit: str) -> None:
        """Validate that unit is a recognized fuel economy unit.

        Args:
            unit: Fuel economy unit string to validate.

        Raises:
            ValueError: If not recognized.
        """
        valid = {feu.value for feu in FuelEconomyUnit}
        if unit not in valid:
            raise ValueError(
                f"Unrecognized fuel economy unit '{unit}'. "
                f"Supported: {sorted(valid)}"
            )

    def _validate_equipment_type(self, equipment_type: str) -> None:
        """Validate that equipment_type is recognized.

        Args:
            equipment_type: Equipment type string to validate.

        Raises:
            ValueError: If not recognized.
        """
        if equipment_type not in _OPERATING_HOUR_FACTORS:
            raise ValueError(
                f"Unrecognized equipment type '{equipment_type}'. "
                f"Supported: {sorted(_OPERATING_HOUR_FACTORS.keys())}"
            )

    def _validate_marine_vessel_type(self, vessel_type: str) -> None:
        """Validate that vessel_type is recognized.

        Args:
            vessel_type: Marine vessel type string to validate.

        Raises:
            ValueError: If not recognized.
        """
        if vessel_type not in _MARINE_EMISSION_FACTORS:
            raise ValueError(
                f"Unrecognized marine vessel type '{vessel_type}'. "
                f"Supported: {sorted(_MARINE_EMISSION_FACTORS.keys())}"
            )

    def _validate_aircraft_type(self, aircraft_type: str) -> None:
        """Validate that aircraft_type is recognized.

        Args:
            aircraft_type: Aircraft type string to validate.

        Raises:
            ValueError: If not recognized.
        """
        if aircraft_type not in _AVIATION_EMISSION_FACTORS:
            raise ValueError(
                f"Unrecognized aircraft type '{aircraft_type}'. "
                f"Supported: {sorted(_AVIATION_EMISSION_FACTORS.keys())}"
            )

    # ------------------------------------------------------------------
    # Internal: Factor Lookups
    # ------------------------------------------------------------------

    def _get_age_degradation_factor(self, vehicle_age_years: int) -> Decimal:
        """Look up the age degradation factor from the coded table.

        Args:
            vehicle_age_years: Vehicle age in years.

        Returns:
            Age degradation multiplier (>= 1.0).
        """
        for max_age, factor in _AGE_DEGRADATION_TABLE:
            if vehicle_age_years < max_age:
                return factor
        return _AGE_DEGRADATION_MAX

    def _get_load_factor_adjustment(self, load_factor: str) -> Decimal:
        """Look up the load factor adjustment multiplier.

        Args:
            load_factor: Load factor classification string.

        Returns:
            Load factor multiplier.
        """
        return _LOAD_FACTOR_ADJUSTMENTS[load_factor]

    # ------------------------------------------------------------------
    # Internal: Provenance and Metrics
    # ------------------------------------------------------------------

    def _record_provenance(
        self,
        action: str,
        entity_id: str,
        data: Dict[str, Any],
    ) -> None:
        """Record provenance tracking event if available.

        Args:
            action: Action description.
            entity_id: Entity identifier.
            data: Provenance data dictionary.
        """
        if _PROVENANCE_AVAILABLE and _get_provenance_tracker is not None:
            try:
                tracker = _get_provenance_tracker()
                tracker.record(
                    entity_type="distance_estimation",
                    action=action,
                    entity_id=entity_id,
                    data=data,
                    metadata={"engine": "DistanceEstimatorEngine"},
                )
            except Exception:
                logger.debug("Provenance recording skipped", exc_info=True)

    def _record_metrics(self, vehicle_type: str, elapsed: float) -> None:
        """Record Prometheus metrics if available.

        Args:
            vehicle_type: Vehicle type for labelling.
            elapsed: Elapsed time in seconds.
        """
        if _METRICS_AVAILABLE and _record_distance_estimation is not None:
            try:
                _record_distance_estimation(vehicle_type, "complete")
            except Exception:
                logger.debug("Metrics recording skipped", exc_info=True)
        if _METRICS_AVAILABLE and _observe_calculation_duration is not None:
            try:
                _observe_calculation_duration(elapsed)
            except Exception:
                logger.debug("Metrics recording skipped", exc_info=True)
