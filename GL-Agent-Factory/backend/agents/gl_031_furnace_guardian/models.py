"""
GL-031 Furnace Guardian - Pydantic Models

This module defines the data models for furnace safety monitoring,
including interlock validation, flame supervision, and compliance tracking.

Standards Reference:
- NFPA 86: Standard for Ovens and Furnaces
- API 560: Fired Heaters for General Refinery Service
- EN 746: Industrial Thermoprocessing Equipment
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator


class RiskLevel(str, Enum):
    """Risk level classifications per NFPA 86."""
    NONE = "NONE"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class InterlockType(str, Enum):
    """Types of furnace safety interlocks per NFPA 86."""
    FLAME_FAILURE = "FLAME_FAILURE"
    HIGH_TEMPERATURE = "HIGH_TEMPERATURE"
    LOW_TEMPERATURE = "LOW_TEMPERATURE"
    HIGH_PRESSURE = "HIGH_PRESSURE"
    LOW_PRESSURE = "LOW_PRESSURE"
    LOW_COMBUSTION_AIR = "LOW_COMBUSTION_AIR"
    HIGH_FUEL_PRESSURE = "HIGH_FUEL_PRESSURE"
    LOW_FUEL_PRESSURE = "LOW_FUEL_PRESSURE"
    PURGE_INCOMPLETE = "PURGE_INCOMPLETE"
    PILOT_FAILURE = "PILOT_FAILURE"
    MAIN_FUEL_VALVE = "MAIN_FUEL_VALVE"
    SAFETY_SHUTOFF_VALVE = "SAFETY_SHUTOFF_VALVE"
    COMBUSTION_AIR_FAN = "COMBUSTION_AIR_FAN"
    EXHAUST_FAN = "EXHAUST_FAN"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    DOOR_OPEN = "DOOR_OPEN"
    ATMOSPHERE_CONTROL = "ATMOSPHERE_CONTROL"


class ViolationSeverity(str, Enum):
    """Severity levels for safety violations."""
    INFO = "INFO"
    WARNING = "WARNING"
    ALARM = "ALARM"
    TRIP = "TRIP"
    EMERGENCY = "EMERGENCY"


class FlameDetectorType(str, Enum):
    """Types of flame detection sensors."""
    UV_SCANNER = "UV_SCANNER"
    IR_SCANNER = "IR_SCANNER"
    UV_IR_COMBINED = "UV_IR_COMBINED"
    FLAME_ROD = "FLAME_ROD"
    PHOTOCELL = "PHOTOCELL"


class FurnaceType(str, Enum):
    """Types of industrial furnaces."""
    FIRED_HEATER = "FIRED_HEATER"
    PROCESS_FURNACE = "PROCESS_FURNACE"
    HEAT_TREATMENT = "HEAT_TREATMENT"
    MELTING_FURNACE = "MELTING_FURNACE"
    ANNEALING_FURNACE = "ANNEALING_FURNACE"
    DRYING_OVEN = "DRYING_OVEN"
    CALCINER = "CALCINER"
    REFORMER = "REFORMER"


class ComplianceStandard(str, Enum):
    """Compliance standards for furnace safety."""
    NFPA_86 = "NFPA_86"
    API_560 = "API_560"
    EN_746 = "EN_746"
    IEC_61511 = "IEC_61511"
    ASME_CSD_1 = "ASME_CSD_1"


class PurgeStatus(str, Enum):
    """Status of furnace purge cycle."""
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    BYPASSED = "BYPASSED"
