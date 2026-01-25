"""
GL-032 Refractory Monitor - Pydantic Models

This module defines data models for refractory health assessment,
including thermal imaging analysis and remaining life prediction.

Standards Reference:
- API 560: Fired Heaters for General Refinery Service
- ASTM C155: Standard Classification of Insulating Firebrick
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from pydantic import BaseModel, Field, validator


class RefractoryMaterial(str, Enum):
    """Types of refractory materials."""
    DENSE_FIREBRICK = "DENSE_FIREBRICK"
    INSULATING_FIREBRICK = "INSULATING_FIREBRICK"
    CASTABLE = "CASTABLE"
    CERAMIC_FIBER = "CERAMIC_FIBER"
    SILICA = "SILICA"
    HIGH_ALUMINA = "HIGH_ALUMINA"
    BASIC = "BASIC"
    CHROME_MAGNESITE = "CHROME_MAGNESITE"


class RefractoryZone(str, Enum):
    """Zones in furnace/heater where refractory is used."""
    RADIANT_SECTION = "RADIANT_SECTION"
    CONVECTION_SECTION = "CONVECTION_SECTION"
    FLOOR = "FLOOR"
    ROOF = "ROOF"
    SIDEWALL = "SIDEWALL"
    BURNER_BLOCK = "BURNER_BLOCK"
    TARGET_WALL = "TARGET_WALL"
    TRANSITION = "TRANSITION"


class MaintenancePriority(str, Enum):
    """Maintenance priority levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    SCHEDULED = "SCHEDULED"


class HealthStatus(str, Enum):
    """Refractory health status classifications."""
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"
    CRITICAL = "CRITICAL"
    FAILED = "FAILED"


class DegradationMode(str, Enum):
    """Refractory degradation mechanisms."""
    THERMAL_SHOCK = "THERMAL_SHOCK"
    SPALLING = "SPALLING"
    EROSION = "EROSION"
    CHEMICAL_ATTACK = "CHEMICAL_ATTACK"
    MECHANICAL_DAMAGE = "MECHANICAL_DAMAGE"
    THERMAL_AGING = "THERMAL_AGING"
    ANCHOR_FAILURE = "ANCHOR_FAILURE"
    PINCH_SPALLING = "PINCH_SPALLING"
