"""
GL-033 Burner Balancer - Pydantic Models

This module defines data models for multi-burner load balancing
and air-fuel optimization.

Standards Reference:
- NFPA 85: Boiler and Combustion Systems Hazards Code
- NFPA 86: Standard for Ovens and Furnaces
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class BurnerType(str, Enum):
    """Types of industrial burners."""
    PREMIX = "PREMIX"
    NOZZLE_MIX = "NOZZLE_MIX"
    RAW_GAS = "RAW_GAS"
    STAGED_AIR = "STAGED_AIR"
    STAGED_FUEL = "STAGED_FUEL"
    LOW_NOX = "LOW_NOX"
    ULTRA_LOW_NOX = "ULTRA_LOW_NOX"


class FuelType(str, Enum):
    """Fuel types for combustion."""
    NATURAL_GAS = "NATURAL_GAS"
    PROPANE = "PROPANE"
    FUEL_OIL = "FUEL_OIL"
    HYDROGEN = "HYDROGEN"
    SYNGAS = "SYNGAS"
    BIOGAS = "BIOGAS"


class BurnerStatus(str, Enum):
    """Burner operational status."""
    OFF = "OFF"
    PILOT = "PILOT"
    LOW_FIRE = "LOW_FIRE"
    MODULATING = "MODULATING"
    HIGH_FIRE = "HIGH_FIRE"
    FAULT = "FAULT"


class BalancingObjective(str, Enum):
    """Optimization objectives for balancing."""
    EFFICIENCY = "EFFICIENCY"
    LOW_EMISSIONS = "LOW_EMISSIONS"
    UNIFORM_HEATING = "UNIFORM_HEATING"
    FUEL_COST = "FUEL_COST"
    BALANCED = "BALANCED"
