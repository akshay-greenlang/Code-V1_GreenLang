"""
Shared pytest fixtures for GL-004 BURNMASTER test suite.

Provides combustion-specific test data, mock DCS/historian connections,
and utility functions for testing the combustion optimization system.
"""

import pytest
import asyncio
import numpy as np
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from unittest.mock import Mock, AsyncMock, MagicMock
import random


# ============================================================================
# ENUMS
# ============================================================================

class FuelType(Enum):
    """Supported fuel types for combustion calculations."""
    NATURAL_GAS = "natural_gas"
    DIESEL = "diesel"
    PROPANE = "propane"
    HYDROGEN = "hydrogen"
    COAL = "coal"
    BIOMASS = "biomass"
    FUEL_OIL_2 = "fuel_oil_2"
    FUEL_OIL_6 = "fuel_oil_6"


class OperatingMode(Enum):
    """Burner operating modes."""
    OFF = "off"
    PURGE = "purge"
    IGNITION = "ignition"
    LOW_FIRE = "low_fire"
    MODULATING = "modulating"
    HIGH_FIRE = "high_fire"
    SHUTDOWN = "shutdown"


class BurnerState(Enum):
    """Burner safety states."""
    SAFE = "safe"
    WARNING = "warning"
    ALARM = "alarm"
    TRIP = "trip"
    LOCKOUT = "lockout"


class ControlMode(Enum):
    """Control system operating modes."""
    OBSERVE = "observe"
    ADVISORY = "advisory"
    CLOSED_LOOP = "closed_loop"
    MANUAL = "manual"
    FALLBACK = "fallback"


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class FuelProperties:
    """Physical and chemical properties of a fuel."""
    fuel_type: FuelType
    lower_heating_value: float  # MJ/kg or MJ/Nm3
    higher_heating_value: float  # MJ/kg or MJ/Nm3
    stoichiometric_air_ratio: float  # kg air / kg fuel
    density: float  # kg/m3 at STP
    carbon_content: float  # mass fraction
    hydrogen_content: float  # mass fraction
    oxygen_content: float  # mass fraction
    nitrogen_content: float  # mass fraction
    sulfur_content: float  # mass fraction
    moisture_content: float  # mass fraction
    adiabatic_flame_temp: float  # K


@dataclass
class CombustionMeasurement:
    """A single combustion measurement from sensors."""
    timestamp: datetime
    fuel_flow_rate: float  # kg/s or Nm3/s
    air_flow_rate: float  # kg/s or Nm3/s
    flue_gas_temp: float  # K
    ambient_temp: float  # K
    o2_percentage: float  # % dry basis
    co_ppm: float  # ppm
    co2_percentage: float  # %
    nox_ppm: float  # ppm
    lambda_value: float  # excess air ratio
    efficiency: float  # %
    heat_output: float  # MW


@dataclass
class SafetyEnvelope:
    """Safety limits for burner operation."""
    min_lambda: float = 1.05
    max_lambda: float = 1.50
    min_fuel_flow: float = 0.0
    max_fuel_flow: float = 100.0
    min_air_flow: float = 0.0
    max_air_flow: float = 1000.0
    max_flue_gas_temp: float = 1500.0  # K
    max_co_ppm: float = 100.0
    max_nox_ppm: float = 50.0
    min_o2_percentage: float = 1.0
    max_o2_percentage: float = 10.0


@dataclass
class OptimizationResult:
    """Result of a combustion optimization cycle."""
    success: bool
    optimal_lambda: float
    optimal_air_flow: float
    predicted_efficiency: float
    predicted_co_ppm: float
    predicted_nox_ppm: float
    iterations: int
    convergence_time_ms: float
    constraints_satisfied: bool
    recommendations: List[str] = field(default_factory=list)


# ============================================================================
# REFERENCE DATA
# ============================================================================

STOICHIOMETRIC_AIR_RATIOS: Dict[FuelType, float] = {
    FuelType.NATURAL_GAS: 17.2,
    FuelType.DIESEL: 14.5,
    FuelType.PROPANE: 15.7,
    FuelType.HYDROGEN: 34.3,
    FuelType.COAL: 11.5,
    FuelType.BIOMASS: 6.5,
    FuelType.FUEL_OIL_2: 14.4,
    FuelType.FUEL_OIL_6: 13.8,
}

LOWER_HEATING_VALUES: Dict[FuelType, float] = {
    FuelType.NATURAL_GAS: 50.0,
    FuelType.DIESEL: 42.5,
    FuelType.PROPANE: 46.4,
    FuelType.HYDROGEN: 120.0,
    FuelType.COAL: 25.0,
    FuelType.BIOMASS: 18.0,
    FuelType.FUEL_OIL_2: 42.6,
    FuelType.FUEL_OIL_6: 40.5,
}

HIGHER_HEATING_VALUES: Dict[FuelType, float] = {
    FuelType.NATURAL_GAS: 55.5,
    FuelType.DIESEL: 45.4,
    FuelType.PROPANE: 50.3,
    FuelType.HYDROGEN: 141.8,
    FuelType.COAL: 27.0,
    FuelType.BIOMASS: 20.0,
    FuelType.FUEL_OIL_2: 45.5,
    FuelType.FUEL_OIL_6: 43.0,
}

ADIABATIC_FLAME_TEMPS: Dict[FuelType, float] = {
    FuelType.NATURAL_GAS: 2223.0,
    FuelType.DIESEL: 2327.0,
    FuelType.PROPANE: 2267.0,
    FuelType.HYDROGEN: 2483.0,
    FuelType.COAL: 2150.0,
    FuelType.BIOMASS: 1900.0,
    FuelType.FUEL_OIL_2: 2300.0,
    FuelType.FUEL_OIL_6: 2250.0,
}

CO2_EMISSION_FACTORS: Dict[FuelType, float] = {
    FuelType.NATURAL_GAS: 56.1,
    FuelType.DIESEL: 74.1,
    FuelType.PROPANE: 63.1,
    FuelType.HYDROGEN: 0.0,
    FuelType.COAL: 94.6,
    FuelType.BIOMASS: 0.0,
    FuelType.FUEL_OIL_2: 73.3,
    FuelType.FUEL_OIL_6: 77.4,
}
