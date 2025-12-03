# -*- coding: utf-8 -*-
"""
Deterministic tool functions for CONDENSYNC - Condenser Optimization Agent.

This module implements all deterministic calculation and optimization functions
for steam condenser operations. All functions follow zero-hallucination
principles with no LLM involvement in numeric calculations.

All calculations based on HEI (Heat Exchange Institute) standards, ASME
guidelines, and industry-standard heat exchanger formulas.

Domain: Steam Systems - Condenser Optimization
Inputs: Cooling water temp, vacuum levels, condensate flow
Outputs: Optimal condenser operation, efficiency improvements

Example:
    >>> executor = CondenserToolExecutor()
    >>> result = await executor.execute_tool(
    ...     "calculate_heat_transfer_coefficient",
    ...     {"cw_inlet_temp": 20.0, "cw_outlet_temp": 30.0, ...}
    ... )
"""

import hashlib
import logging
import math
import threading
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import json

logger = logging.getLogger(__name__)

# Thread-safe lock for calculations
_calculation_lock = threading.Lock()


# =============================================================================
# ENUMERATIONS
# =============================================================================

class TubeMaterial(Enum):
    """Condenser tube materials with thermal properties."""
    ADMIRALTY_BRASS = "admiralty_brass"
    COPPER_NICKEL_90_10 = "copper_nickel_90_10"
    COPPER_NICKEL_70_30 = "copper_nickel_70_30"
    TITANIUM = "titanium"
    STAINLESS_STEEL_304 = "stainless_steel_304"
    STAINLESS_STEEL_316 = "stainless_steel_316"
    CARBON_STEEL = "carbon_steel"


class CondenserType(Enum):
    """Types of steam condensers."""
    SHELL_AND_TUBE = "shell_and_tube"
    SURFACE = "surface"
    JET = "jet"
    BAROMETRIC = "barometric"
    AIR_COOLED = "air_cooled"


class LeakageSeverity(Enum):
    """Air inleakage severity levels."""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class CleaningMethod(Enum):
    """Tube cleaning methods."""
    MECHANICAL_BRUSHING = "mechanical_brushing"
    HIGH_PRESSURE_WATER = "high_pressure_water"
    CHEMICAL_CLEANING = "chemical_cleaning"
    RUBBER_BALL_SYSTEM = "rubber_ball_system"
    SPONGE_BALL_SYSTEM = "sponge_ball_system"


class AirEjectorType(Enum):
    """Types of air removal equipment."""
    STEAM_JET_EJECTOR = "steam_jet_ejector"
    LIQUID_RING_PUMP = "liquid_ring_pump"
    ROTARY_VANE_PUMP = "rotary_vane_pump"
    HYBRID_SYSTEM = "hybrid_system"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CondenserPerformanceMetrics:
    """Result of condenser performance analysis."""
    condenser_id: str
    heat_duty: float  # kW
    u_value: float  # W/(m2.K) - Overall heat transfer coefficient
    cleanliness_factor: float  # 0-1.0
    lmtd: float  # Log mean temperature difference (K)
    ttd: float  # Terminal temperature difference (K)
    dca: float  # Drain cooler approach (K)
    vacuum_pressure: float  # mbar abs
    saturation_temp: float  # Celsius
    efficiency: float  # Percentage
    trends: Dict[str, List[float]]
    status: str  # "optimal", "degraded", "critical"
    recommendations: List[str]
    timestamp: str
    provenance_hash: str


@dataclass
class HeatTransferResult:
    """Result of heat transfer coefficient calculation."""
    u_value: float  # W/(m2.K)
    u_design: float  # W/(m2.K) - Design U-value
    cleanliness_factor: float  # Ratio of actual/design
    lmtd: float  # Log mean temperature difference (K)
    ttd: float  # Terminal temperature difference (K)
    heat_duty: float  # kW
    heat_flux: float  # W/m2
    fouling_resistance: float  # m2.K/W
    tube_side_coefficient: float  # W/(m2.K)
    shell_side_coefficient: float  # W/(m2.K)
    timestamp: str
    provenance_hash: str


@dataclass
class VacuumOptimizationResult:
    """Result of vacuum pressure optimization."""
    current_vacuum: float  # mbar abs
    optimal_vacuum: float  # mbar abs
    expected_efficiency_gain: float  # Percentage points
    expected_power_gain: float  # kW
    limiting_factor: str  # What limits achieving optimal vacuum
    achievable_vacuum: float  # mbar abs considering constraints
    action_items: List[str]
    cost_benefit: Dict[str, float]
    timestamp: str
    provenance_hash: str


@dataclass
class AirInleakageAssessment:
    """Result of air inleakage detection and assessment."""
    condenser_id: str
    estimated_leakage_rate: float  # kg/hr
    severity: LeakageSeverity
    probable_locations: List[Dict[str, Any]]
    vacuum_degradation: float  # mbar
    ejector_load_percent: float  # Percentage of capacity
    detection_confidence: float  # 0-1.0
    recommended_actions: List[str]
    estimated_repair_priority: int  # 1-5 (1 = highest)
    timestamp: str
    provenance_hash: str


@dataclass
class FoulingAnalysisResult:
    """Result of fouling factor calculation and analysis."""
    fouling_resistance: float  # m2.K/W
    fouling_factor: float  # Cleanliness ratio (0-1.0)
    degradation_rate: float  # m2.K/W per 1000 hours
    estimated_deposit_thickness: float  # mm
    cleaning_recommended: bool
    cleaning_urgency: str  # "immediate", "scheduled", "monitor"
    expected_improvement: float  # Percentage U-value recovery
    time_to_critical: Optional[float]  # Hours until critical fouling
    timestamp: str
    provenance_hash: str


@dataclass
class CleaningScheduleResult:
    """Result of tube cleaning schedule prediction."""
    condenser_id: str
    recommended_cleaning_date: str  # ISO format date
    days_until_cleaning: int
    cleaning_method: CleaningMethod
    expected_u_value_recovery: float  # Percentage
    expected_efficiency_gain: float  # Percentage
    estimated_duration: float  # Hours
    estimated_cost: float  # USD
    production_impact: float  # MWh lost
    net_benefit: float  # USD
    confidence_level: float  # 0-1.0
    timestamp: str
    provenance_hash: str


@dataclass
class CoolingWaterOptimizationResult:
    """Result of cooling water flow optimization."""
    current_flow_rate: float  # m3/hr
    optimal_flow_rate: float  # m3/hr
    flow_change: float  # Percentage change
    pump_power_current: float  # kW
    pump_power_optimal: float  # kW
    pump_energy_savings: float  # kW
    vacuum_impact: float  # mbar change
    efficiency_impact: float  # Percentage change
    annual_savings: float  # USD/year
    constraints: List[str]
    timestamp: str
    provenance_hash: str


@dataclass
class PerformanceReportData:
    """Data structure for performance report generation."""
    condenser_id: str
    report_period: str
    summary_metrics: Dict[str, float]
    trend_analysis: Dict[str, Any]
    efficiency_statistics: Dict[str, float]
    fouling_status: Dict[str, Any]
    maintenance_events: List[Dict[str, Any]]
    recommendations: List[str]
    kpis: Dict[str, float]
    timestamp: str
    provenance_hash: str


# =============================================================================
# CONSTANTS AND REFERENCE DATA
# =============================================================================

# Tube material thermal conductivities (W/m.K)
TUBE_THERMAL_CONDUCTIVITY: Dict[TubeMaterial, float] = {
    TubeMaterial.ADMIRALTY_BRASS: 111.0,
    TubeMaterial.COPPER_NICKEL_90_10: 45.0,
    TubeMaterial.COPPER_NICKEL_70_30: 29.0,
    TubeMaterial.TITANIUM: 21.9,
    TubeMaterial.STAINLESS_STEEL_304: 16.2,
    TubeMaterial.STAINLESS_STEEL_316: 16.3,
    TubeMaterial.CARBON_STEEL: 50.0,
}

# Typical design fouling resistances (m2.K/W) per HEI standards
HEI_FOULING_RESISTANCE: Dict[str, float] = {
    "clean_fresh_water": 0.000044,
    "treated_cooling_tower": 0.000088,
    "untreated_cooling_tower": 0.000176,
    "brackish_water": 0.000352,
    "seawater": 0.000088,
    "river_water": 0.000352,
    "well_water": 0.000176,
}

# Steam saturation temperature vs pressure (mbar abs)
# Simplified lookup table for common condenser vacuum ranges
SATURATION_TEMP_TABLE: Dict[int, float] = {
    25: 21.1,
    30: 24.1,
    35: 26.7,
    40: 29.0,
    45: 31.0,
    50: 32.9,
    55: 34.6,
    60: 36.2,
    65: 37.6,
    70: 39.0,
    75: 40.3,
    80: 41.5,
    85: 42.7,
    90: 43.8,
    95: 44.8,
    100: 45.8,
}

# Air leakage rate guidelines (kg/hr per MW)
AIR_LEAKAGE_GUIDELINES: Dict[str, float] = {
    "excellent": 0.5,
    "good": 1.0,
    "acceptable": 2.0,
    "poor": 4.0,
    "critical": 8.0,
}


# =============================================================================
# TOOL DEFINITIONS FOR LLM INTEGRATION
# =============================================================================

CONDENSER_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "analyze_condenser_performance",
        "description": """Analyze overall condenser performance including heat transfer
        efficiency, vacuum levels, and operating trends. This tool provides a comprehensive
        assessment of condenser health and identifies areas for improvement. Uses
        deterministic calculations based on HEI standards - no LLM involvement in
        numeric computations.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "condenser_id": {
                    "type": "string",
                    "description": "Unique identifier for the condenser unit"
                },
                "time_range": {
                    "type": "object",
                    "properties": {
                        "start": {
                            "type": "string",
                            "format": "date-time",
                            "description": "Start of analysis period (ISO format)"
                        },
                        "end": {
                            "type": "string",
                            "format": "date-time",
                            "description": "End of analysis period (ISO format)"
                        }
                    },
                    "required": ["start", "end"],
                    "description": "Time range for performance analysis"
                },
                "include_trends": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include historical trend data in analysis"
                },
                "operating_data": {
                    "type": "object",
                    "properties": {
                        "cw_inlet_temp": {
                            "type": "number",
                            "description": "Cooling water inlet temperature (Celsius)"
                        },
                        "cw_outlet_temp": {
                            "type": "number",
                            "description": "Cooling water outlet temperature (Celsius)"
                        },
                        "cw_flow_rate": {
                            "type": "number",
                            "description": "Cooling water flow rate (m3/hr)"
                        },
                        "vacuum_pressure": {
                            "type": "number",
                            "description": "Condenser vacuum pressure (mbar abs)"
                        },
                        "steam_flow": {
                            "type": "number",
                            "description": "Exhaust steam flow rate (kg/hr)"
                        },
                        "turbine_load": {
                            "type": "number",
                            "description": "Turbine generator load (MW)"
                        }
                    },
                    "required": ["cw_inlet_temp", "cw_outlet_temp", "vacuum_pressure"],
                    "description": "Current operating data for performance calculation"
                }
            },
            "required": ["condenser_id", "operating_data"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "condenser_id": {"type": "string"},
                "heat_duty": {"type": "number", "description": "Heat duty in kW"},
                "u_value": {"type": "number", "description": "Overall U-value W/(m2.K)"},
                "cleanliness_factor": {"type": "number", "description": "0-1.0 ratio"},
                "lmtd": {"type": "number", "description": "Log mean temp diff (K)"},
                "ttd": {"type": "number", "description": "Terminal temp diff (K)"},
                "vacuum_pressure": {"type": "number", "description": "mbar abs"},
                "efficiency": {"type": "number", "description": "Percentage"},
                "status": {"type": "string", "enum": ["optimal", "degraded", "critical"]},
                "recommendations": {"type": "array", "items": {"type": "string"}},
                "timestamp": {"type": "string", "format": "date-time"},
                "provenance_hash": {"type": "string"}
            }
        }
    },
    {
        "name": "calculate_heat_transfer_coefficient",
        "description": """Calculate the overall heat transfer coefficient (U-value) for
        the condenser using formula-based deterministic calculation. Computes cleanliness
        factor, LMTD, TTD, and fouling resistance based on HEI standards. This is a
        zero-hallucination calculation using industry-standard formulas only.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "cw_inlet_temp": {
                    "type": "number",
                    "description": "Cooling water inlet temperature (Celsius)",
                    "minimum": 0,
                    "maximum": 50
                },
                "cw_outlet_temp": {
                    "type": "number",
                    "description": "Cooling water outlet temperature (Celsius)",
                    "minimum": 0,
                    "maximum": 60
                },
                "cw_flow_rate": {
                    "type": "number",
                    "description": "Cooling water volumetric flow rate (m3/hr)",
                    "minimum": 0
                },
                "steam_temp": {
                    "type": "number",
                    "description": "Steam saturation temperature in condenser (Celsius)",
                    "minimum": 20,
                    "maximum": 60
                },
                "heat_duty": {
                    "type": "number",
                    "description": "Heat duty to be transferred (kW)",
                    "minimum": 0
                },
                "tube_surface_area": {
                    "type": "number",
                    "description": "Total tube surface area (m2)",
                    "minimum": 0
                },
                "design_u_value": {
                    "type": "number",
                    "description": "Design heat transfer coefficient W/(m2.K)",
                    "default": 3000
                },
                "tube_od": {
                    "type": "number",
                    "description": "Tube outer diameter (mm)",
                    "default": 25.4
                },
                "tube_thickness": {
                    "type": "number",
                    "description": "Tube wall thickness (mm)",
                    "default": 1.24
                },
                "tube_material": {
                    "type": "string",
                    "enum": ["admiralty_brass", "copper_nickel_90_10", "copper_nickel_70_30",
                             "titanium", "stainless_steel_304", "stainless_steel_316", "carbon_steel"],
                    "default": "titanium",
                    "description": "Tube material type"
                }
            },
            "required": ["cw_inlet_temp", "cw_outlet_temp", "cw_flow_rate", "steam_temp", "heat_duty"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "u_value": {"type": "number", "description": "Actual U-value W/(m2.K)"},
                "u_design": {"type": "number", "description": "Design U-value W/(m2.K)"},
                "cleanliness_factor": {"type": "number", "description": "Ratio 0-1.0"},
                "lmtd": {"type": "number", "description": "Log mean temp diff (K)"},
                "ttd": {"type": "number", "description": "Terminal temp diff (K)"},
                "heat_duty": {"type": "number", "description": "Heat duty (kW)"},
                "heat_flux": {"type": "number", "description": "Heat flux W/m2"},
                "fouling_resistance": {"type": "number", "description": "m2.K/W"},
                "timestamp": {"type": "string", "format": "date-time"},
                "provenance_hash": {"type": "string"}
            }
        }
    },
    {
        "name": "optimize_vacuum_pressure",
        "description": """Optimize condenser vacuum pressure setpoint based on turbine
        load, ambient conditions, and equipment constraints. Uses engineering optimization
        algorithms to determine optimal vacuum for maximum efficiency. All calculations
        are deterministic and based on thermodynamic relationships.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "current_vacuum": {
                    "type": "number",
                    "description": "Current condenser vacuum (mbar abs)",
                    "minimum": 20,
                    "maximum": 150
                },
                "turbine_load": {
                    "type": "number",
                    "description": "Current turbine generator load (MW)",
                    "minimum": 0
                },
                "ambient_conditions": {
                    "type": "object",
                    "properties": {
                        "dry_bulb_temp": {
                            "type": "number",
                            "description": "Ambient dry bulb temperature (Celsius)"
                        },
                        "wet_bulb_temp": {
                            "type": "number",
                            "description": "Ambient wet bulb temperature (Celsius)"
                        },
                        "relative_humidity": {
                            "type": "number",
                            "description": "Relative humidity (percentage)",
                            "minimum": 0,
                            "maximum": 100
                        }
                    },
                    "required": ["dry_bulb_temp", "wet_bulb_temp"],
                    "description": "Ambient weather conditions"
                },
                "cw_inlet_temp": {
                    "type": "number",
                    "description": "Cooling water inlet temperature (Celsius)"
                },
                "cooling_tower_approach": {
                    "type": "number",
                    "description": "Cooling tower approach temperature (Celsius)",
                    "default": 5.0
                },
                "condenser_design_ttd": {
                    "type": "number",
                    "description": "Design terminal temperature difference (K)",
                    "default": 3.0
                },
                "turbine_backpressure_limit": {
                    "type": "number",
                    "description": "Maximum allowable turbine backpressure (mbar abs)",
                    "default": 100
                }
            },
            "required": ["current_vacuum", "turbine_load", "ambient_conditions"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "current_vacuum": {"type": "number"},
                "optimal_vacuum": {"type": "number"},
                "expected_efficiency_gain": {"type": "number"},
                "expected_power_gain": {"type": "number"},
                "limiting_factor": {"type": "string"},
                "achievable_vacuum": {"type": "number"},
                "action_items": {"type": "array", "items": {"type": "string"}},
                "cost_benefit": {"type": "object"},
                "timestamp": {"type": "string"},
                "provenance_hash": {"type": "string"}
            }
        }
    },
    {
        "name": "detect_air_inleakage",
        "description": """Detect and assess air inleakage into the condenser based on
        vacuum trends and air ejector operating data. Uses pattern recognition algorithms
        to identify probable leak locations and severity. All assessments are based on
        deterministic analysis of operating parameters.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "condenser_id": {
                    "type": "string",
                    "description": "Unique identifier for the condenser"
                },
                "vacuum_trend": {
                    "type": "object",
                    "properties": {
                        "timestamps": {
                            "type": "array",
                            "items": {"type": "string", "format": "date-time"},
                            "description": "Timestamps for vacuum readings"
                        },
                        "values": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Vacuum pressure readings (mbar abs)"
                        }
                    },
                    "required": ["timestamps", "values"],
                    "description": "Historical vacuum pressure trend data"
                },
                "air_ejector_data": {
                    "type": "object",
                    "properties": {
                        "ejector_type": {
                            "type": "string",
                            "enum": ["steam_jet_ejector", "liquid_ring_pump",
                                     "rotary_vane_pump", "hybrid_system"]
                        },
                        "suction_pressure": {
                            "type": "number",
                            "description": "Ejector suction pressure (mbar abs)"
                        },
                        "discharge_temp": {
                            "type": "number",
                            "description": "Ejector discharge temperature (Celsius)"
                        },
                        "motive_steam_flow": {
                            "type": "number",
                            "description": "Motive steam flow rate (kg/hr)"
                        },
                        "design_capacity": {
                            "type": "number",
                            "description": "Design air removal capacity (kg/hr)"
                        },
                        "current_load": {
                            "type": "number",
                            "description": "Current air removal rate (kg/hr)"
                        }
                    },
                    "required": ["suction_pressure", "current_load", "design_capacity"],
                    "description": "Air removal system operating data"
                },
                "turbine_load": {
                    "type": "number",
                    "description": "Current turbine load (MW) for normalization"
                }
            },
            "required": ["condenser_id", "vacuum_trend", "air_ejector_data"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "condenser_id": {"type": "string"},
                "estimated_leakage_rate": {"type": "number"},
                "severity": {"type": "string"},
                "probable_locations": {"type": "array"},
                "vacuum_degradation": {"type": "number"},
                "ejector_load_percent": {"type": "number"},
                "detection_confidence": {"type": "number"},
                "recommended_actions": {"type": "array"},
                "estimated_repair_priority": {"type": "integer"},
                "timestamp": {"type": "string"},
                "provenance_hash": {"type": "string"}
            }
        }
    },
    {
        "name": "calculate_fouling_factor",
        "description": """Calculate fouling factor and resistance using HEI standard
        methodology. Compares design vs actual U-values to determine cleanliness and
        estimate deposit thickness. Provides cleaning recommendations based on fouling
        degradation rate. Pure formula-based calculation with no LLM involvement.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "design_U": {
                    "type": "number",
                    "description": "Design overall heat transfer coefficient W/(m2.K)",
                    "minimum": 1000,
                    "maximum": 6000
                },
                "actual_U": {
                    "type": "number",
                    "description": "Actual (measured) heat transfer coefficient W/(m2.K)",
                    "minimum": 500,
                    "maximum": 6000
                },
                "tube_material": {
                    "type": "string",
                    "enum": ["admiralty_brass", "copper_nickel_90_10", "copper_nickel_70_30",
                             "titanium", "stainless_steel_304", "stainless_steel_316", "carbon_steel"],
                    "description": "Tube material for thermal conductivity lookup"
                },
                "operating_hours": {
                    "type": "number",
                    "description": "Operating hours since last cleaning",
                    "minimum": 0
                },
                "water_source": {
                    "type": "string",
                    "enum": ["clean_fresh_water", "treated_cooling_tower", "untreated_cooling_tower",
                             "brackish_water", "seawater", "river_water", "well_water"],
                    "default": "treated_cooling_tower",
                    "description": "Cooling water source type for reference fouling"
                },
                "historical_fouling_data": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "operating_hours": {"type": "number"},
                            "fouling_resistance": {"type": "number"}
                        }
                    },
                    "description": "Historical fouling data for trend analysis"
                }
            },
            "required": ["design_U", "actual_U", "tube_material", "operating_hours"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "fouling_resistance": {"type": "number"},
                "fouling_factor": {"type": "number"},
                "degradation_rate": {"type": "number"},
                "estimated_deposit_thickness": {"type": "number"},
                "cleaning_recommended": {"type": "boolean"},
                "cleaning_urgency": {"type": "string"},
                "expected_improvement": {"type": "number"},
                "time_to_critical": {"type": "number"},
                "timestamp": {"type": "string"},
                "provenance_hash": {"type": "string"}
            }
        }
    },
    {
        "name": "predict_tube_cleaning_schedule",
        "description": """Predict optimal tube cleaning schedule based on fouling trends
        and production constraints. Uses predictive analytics to balance cleaning cost
        against efficiency losses. Considers production schedule to minimize downtime
        impact. Deterministic prediction based on fouling degradation models.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "condenser_id": {
                    "type": "string",
                    "description": "Unique identifier for the condenser"
                },
                "fouling_trend": {
                    "type": "object",
                    "properties": {
                        "timestamps": {
                            "type": "array",
                            "items": {"type": "string", "format": "date-time"}
                        },
                        "cleanliness_factors": {
                            "type": "array",
                            "items": {"type": "number"}
                        },
                        "fouling_resistances": {
                            "type": "array",
                            "items": {"type": "number"}
                        }
                    },
                    "required": ["timestamps", "cleanliness_factors"],
                    "description": "Historical fouling trend data"
                },
                "production_schedule": {
                    "type": "object",
                    "properties": {
                        "planned_outages": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "start_date": {"type": "string", "format": "date"},
                                    "end_date": {"type": "string", "format": "date"},
                                    "type": {"type": "string"}
                                }
                            }
                        },
                        "high_demand_periods": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "start_date": {"type": "string", "format": "date"},
                                    "end_date": {"type": "string", "format": "date"}
                                }
                            }
                        },
                        "electricity_price_forecast": {
                            "type": "number",
                            "description": "Average electricity price (USD/MWh)"
                        }
                    },
                    "description": "Production and outage schedule information"
                },
                "current_cleanliness": {
                    "type": "number",
                    "description": "Current cleanliness factor (0-1.0)",
                    "minimum": 0,
                    "maximum": 1
                },
                "cleaning_threshold": {
                    "type": "number",
                    "description": "Cleanliness factor threshold for cleaning",
                    "default": 0.75
                },
                "cleaning_cost": {
                    "type": "number",
                    "description": "Estimated cleaning cost (USD)",
                    "default": 50000
                },
                "unit_capacity": {
                    "type": "number",
                    "description": "Unit generating capacity (MW)"
                }
            },
            "required": ["condenser_id", "fouling_trend", "current_cleanliness"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "condenser_id": {"type": "string"},
                "recommended_cleaning_date": {"type": "string"},
                "days_until_cleaning": {"type": "integer"},
                "cleaning_method": {"type": "string"},
                "expected_u_value_recovery": {"type": "number"},
                "expected_efficiency_gain": {"type": "number"},
                "estimated_duration": {"type": "number"},
                "estimated_cost": {"type": "number"},
                "production_impact": {"type": "number"},
                "net_benefit": {"type": "number"},
                "confidence_level": {"type": "number"},
                "timestamp": {"type": "string"},
                "provenance_hash": {"type": "string"}
            }
        }
    },
    {
        "name": "optimize_cooling_water_flow",
        "description": """Optimize cooling water flow rate for given heat duty and
        target vacuum. Balances pump energy consumption against condenser performance.
        Uses hydraulic optimization algorithms to find optimal operating point.
        Deterministic calculation based on heat transfer and pump affinity laws.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "heat_duty": {
                    "type": "number",
                    "description": "Heat duty to be rejected (kW)",
                    "minimum": 0
                },
                "cw_inlet_temp": {
                    "type": "number",
                    "description": "Cooling water inlet temperature (Celsius)",
                    "minimum": 0,
                    "maximum": 50
                },
                "target_vacuum": {
                    "type": "number",
                    "description": "Target condenser vacuum (mbar abs)",
                    "minimum": 20,
                    "maximum": 150
                },
                "current_flow_rate": {
                    "type": "number",
                    "description": "Current cooling water flow rate (m3/hr)"
                },
                "pump_characteristics": {
                    "type": "object",
                    "properties": {
                        "design_flow": {
                            "type": "number",
                            "description": "Design flow rate (m3/hr)"
                        },
                        "design_head": {
                            "type": "number",
                            "description": "Design head (m)"
                        },
                        "design_power": {
                            "type": "number",
                            "description": "Design power consumption (kW)"
                        },
                        "efficiency": {
                            "type": "number",
                            "description": "Pump efficiency (0-1.0)"
                        },
                        "min_flow": {
                            "type": "number",
                            "description": "Minimum allowable flow (m3/hr)"
                        },
                        "max_flow": {
                            "type": "number",
                            "description": "Maximum allowable flow (m3/hr)"
                        },
                        "vfd_equipped": {
                            "type": "boolean",
                            "description": "Variable frequency drive equipped"
                        }
                    },
                    "description": "Circulating water pump characteristics"
                },
                "condenser_surface_area": {
                    "type": "number",
                    "description": "Condenser tube surface area (m2)"
                },
                "design_u_value": {
                    "type": "number",
                    "description": "Design heat transfer coefficient W/(m2.K)"
                },
                "electricity_cost": {
                    "type": "number",
                    "description": "Electricity cost (USD/kWh)",
                    "default": 0.08
                }
            },
            "required": ["heat_duty", "cw_inlet_temp", "target_vacuum"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "current_flow_rate": {"type": "number"},
                "optimal_flow_rate": {"type": "number"},
                "flow_change": {"type": "number"},
                "pump_power_current": {"type": "number"},
                "pump_power_optimal": {"type": "number"},
                "pump_energy_savings": {"type": "number"},
                "vacuum_impact": {"type": "number"},
                "efficiency_impact": {"type": "number"},
                "annual_savings": {"type": "number"},
                "constraints": {"type": "array"},
                "timestamp": {"type": "string"},
                "provenance_hash": {"type": "string"}
            }
        }
    },
    {
        "name": "generate_performance_report",
        "description": """Generate comprehensive condenser performance report for
        specified period. Aggregates metrics, identifies trends, and provides
        actionable recommendations. This tool may use LLM assistance for narrative
        generation but all numeric calculations are deterministic.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "condenser_id": {
                    "type": "string",
                    "description": "Unique identifier for the condenser"
                },
                "report_period": {
                    "type": "object",
                    "properties": {
                        "start": {
                            "type": "string",
                            "format": "date",
                            "description": "Report period start date"
                        },
                        "end": {
                            "type": "string",
                            "format": "date",
                            "description": "Report period end date"
                        }
                    },
                    "required": ["start", "end"],
                    "description": "Reporting period"
                },
                "metrics_to_include": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["heat_duty", "u_value", "cleanliness_factor", "ttd",
                                 "vacuum_pressure", "efficiency", "air_inleakage",
                                 "cw_flow_rate", "cw_temperatures", "energy_savings"]
                    },
                    "default": ["heat_duty", "u_value", "cleanliness_factor", "vacuum_pressure"],
                    "description": "Metrics to include in report"
                },
                "operating_data_summary": {
                    "type": "object",
                    "properties": {
                        "avg_heat_duty": {"type": "number"},
                        "avg_vacuum": {"type": "number"},
                        "avg_u_value": {"type": "number"},
                        "avg_cleanliness": {"type": "number"},
                        "total_operating_hours": {"type": "number"},
                        "avg_turbine_load": {"type": "number"}
                    },
                    "description": "Summary of operating data for the period"
                },
                "include_recommendations": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include optimization recommendations"
                },
                "include_cost_analysis": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include cost impact analysis"
                }
            },
            "required": ["condenser_id", "report_period"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "condenser_id": {"type": "string"},
                "report_period": {"type": "string"},
                "summary_metrics": {"type": "object"},
                "trend_analysis": {"type": "object"},
                "efficiency_statistics": {"type": "object"},
                "fouling_status": {"type": "object"},
                "maintenance_events": {"type": "array"},
                "recommendations": {"type": "array"},
                "kpis": {"type": "object"},
                "timestamp": {"type": "string"},
                "provenance_hash": {"type": "string"}
            }
        }
    },
    {
        "name": "calculate_condenser_duty",
        "description": """Calculate condenser heat duty from turbine exhaust conditions
        or cooling water temperature rise. Deterministic thermodynamic calculation
        based on energy balance equations.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "calculation_method": {
                    "type": "string",
                    "enum": ["cooling_water", "steam_side"],
                    "description": "Method for duty calculation"
                },
                "cw_inlet_temp": {
                    "type": "number",
                    "description": "Cooling water inlet temperature (Celsius)"
                },
                "cw_outlet_temp": {
                    "type": "number",
                    "description": "Cooling water outlet temperature (Celsius)"
                },
                "cw_flow_rate": {
                    "type": "number",
                    "description": "Cooling water flow rate (m3/hr)"
                },
                "steam_flow": {
                    "type": "number",
                    "description": "Exhaust steam flow rate (kg/hr)"
                },
                "steam_quality": {
                    "type": "number",
                    "description": "Steam quality/dryness fraction (0-1.0)",
                    "default": 0.9
                },
                "condenser_pressure": {
                    "type": "number",
                    "description": "Condenser pressure (mbar abs)"
                }
            },
            "required": ["calculation_method"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "heat_duty": {"type": "number", "description": "Heat duty (kW)"},
                "cw_temp_rise": {"type": "number", "description": "CW temperature rise (K)"},
                "specific_duty": {"type": "number", "description": "Duty per unit flow"},
                "calculation_method": {"type": "string"},
                "timestamp": {"type": "string"},
                "provenance_hash": {"type": "string"}
            }
        }
    },
    {
        "name": "assess_condenser_health",
        "description": """Comprehensive health assessment of condenser system including
        tubes, air removal system, and instrumentation. Combines multiple diagnostic
        parameters into overall health score. Deterministic scoring algorithm.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "condenser_id": {
                    "type": "string",
                    "description": "Unique identifier for the condenser"
                },
                "cleanliness_factor": {
                    "type": "number",
                    "description": "Current cleanliness factor (0-1.0)"
                },
                "air_inleakage_rate": {
                    "type": "number",
                    "description": "Current air inleakage rate (kg/hr)"
                },
                "vacuum_deviation": {
                    "type": "number",
                    "description": "Deviation from expected vacuum (mbar)"
                },
                "ttd_deviation": {
                    "type": "number",
                    "description": "Deviation from design TTD (K)"
                },
                "tube_pluggage_percent": {
                    "type": "number",
                    "description": "Percentage of tubes plugged"
                },
                "operating_hours_since_overhaul": {
                    "type": "number",
                    "description": "Operating hours since last major overhaul"
                },
                "unit_capacity": {
                    "type": "number",
                    "description": "Unit rated capacity (MW)"
                }
            },
            "required": ["condenser_id", "cleanliness_factor"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "condenser_id": {"type": "string"},
                "overall_health_score": {"type": "number", "description": "0-100 score"},
                "component_scores": {"type": "object"},
                "health_status": {"type": "string"},
                "risk_factors": {"type": "array"},
                "recommended_actions": {"type": "array"},
                "next_inspection_date": {"type": "string"},
                "timestamp": {"type": "string"},
                "provenance_hash": {"type": "string"}
            }
        }
    }
]


# =============================================================================
# TOOL EXECUTOR CLASS
# =============================================================================

class CondenserToolExecutor:
    """
    Executor for condenser optimization tools.

    Implements all deterministic calculation and optimization functions
    for the CONDENSYNC agent. All numeric calculations follow zero-hallucination
    principles using industry-standard formulas only.

    Attributes:
        tools: List of available tool definitions
        _provenance_enabled: Whether to track calculation provenance

    Example:
        >>> executor = CondenserToolExecutor()
        >>> result = await executor.execute_tool(
        ...     "calculate_heat_transfer_coefficient",
        ...     {"cw_inlet_temp": 20.0, "cw_outlet_temp": 30.0, ...}
        ... )
    """

    def __init__(self, provenance_enabled: bool = True):
        """Initialize the tool executor."""
        self.tools = CONDENSER_TOOLS
        self._provenance_enabled = provenance_enabled
        self._tool_map = {tool["name"]: tool for tool in CONDENSER_TOOLS}
        logger.info("CondenserToolExecutor initialized with %d tools", len(self.tools))

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a tool by name with given parameters.

        Args:
            tool_name: Name of the tool to execute
            parameters: Input parameters for the tool

        Returns:
            Tool execution result as dictionary

        Raises:
            ValueError: If tool name is unknown or parameters invalid
            RuntimeError: If tool execution fails
        """
        logger.info("Executing tool: %s", tool_name)
        start_time = datetime.utcnow()

        try:
            # Validate tool exists
            if tool_name not in self._tool_map:
                raise ValueError(f"Unknown tool: {tool_name}")

            # Validate parameters
            self._validate_parameters(tool_name, parameters)

            # Route to appropriate handler
            handler_map = {
                "analyze_condenser_performance": self._analyze_condenser_performance,
                "calculate_heat_transfer_coefficient": self._calculate_heat_transfer_coefficient,
                "optimize_vacuum_pressure": self._optimize_vacuum_pressure,
                "detect_air_inleakage": self._detect_air_inleakage,
                "calculate_fouling_factor": self._calculate_fouling_factor,
                "predict_tube_cleaning_schedule": self._predict_tube_cleaning_schedule,
                "optimize_cooling_water_flow": self._optimize_cooling_water_flow,
                "generate_performance_report": self._generate_performance_report,
                "calculate_condenser_duty": self._calculate_condenser_duty,
                "assess_condenser_health": self._assess_condenser_health,
            }

            handler = handler_map.get(tool_name)
            if handler is None:
                raise ValueError(f"No handler implemented for tool: {tool_name}")

            result = await handler(parameters)

            # Add execution metadata
            execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            result["_execution_time_ms"] = round(execution_time_ms, 2)

            logger.info("Tool %s executed successfully in %.2f ms",
                       tool_name, execution_time_ms)

            return result

        except Exception as e:
            logger.error("Tool execution failed: %s - %s", tool_name, str(e))
            raise RuntimeError(f"Tool execution failed: {str(e)}") from e

    def _validate_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> None:
        """Validate input parameters against tool schema."""
        tool_def = self._tool_map[tool_name]
        schema = tool_def.get("input_schema", {})
        required = schema.get("required", [])

        # Check required parameters
        missing = [r for r in required if r not in parameters]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

        # Validate parameter types and ranges
        properties = schema.get("properties", {})
        for param_name, param_value in parameters.items():
            if param_name in properties:
                prop_def = properties[param_name]

                # Check numeric ranges
                if prop_def.get("type") == "number":
                    if "minimum" in prop_def and param_value < prop_def["minimum"]:
                        raise ValueError(
                            f"Parameter {param_name} value {param_value} below "
                            f"minimum {prop_def['minimum']}"
                        )
                    if "maximum" in prop_def and param_value > prop_def["maximum"]:
                        raise ValueError(
                            f"Parameter {param_name} value {param_value} above "
                            f"maximum {prop_def['maximum']}"
                        )

                # Check enum values
                if "enum" in prop_def and param_value not in prop_def["enum"]:
                    raise ValueError(
                        f"Parameter {param_name} value '{param_value}' not in "
                        f"allowed values: {prop_def['enum']}"
                    )

    def _calculate_provenance_hash(self, data: Dict[str, Any]) -> str:
        """Calculate SHA-256 provenance hash for audit trail."""
        if not self._provenance_enabled:
            return ""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    # =========================================================================
    # CORE CALCULATION METHODS
    # =========================================================================

    @staticmethod
    def _get_saturation_temperature(pressure_mbar: float) -> float:
        """
        Get steam saturation temperature for given condenser pressure.

        Uses interpolation of steam tables for accuracy.

        Args:
            pressure_mbar: Condenser pressure in mbar absolute

        Returns:
            Saturation temperature in Celsius
        """
        with _calculation_lock:
            # Find bracketing pressures
            pressures = sorted(SATURATION_TEMP_TABLE.keys())

            if pressure_mbar <= pressures[0]:
                return SATURATION_TEMP_TABLE[pressures[0]]
            if pressure_mbar >= pressures[-1]:
                return SATURATION_TEMP_TABLE[pressures[-1]]

            # Linear interpolation
            for i in range(len(pressures) - 1):
                if pressures[i] <= pressure_mbar <= pressures[i + 1]:
                    p1, p2 = pressures[i], pressures[i + 1]
                    t1, t2 = SATURATION_TEMP_TABLE[p1], SATURATION_TEMP_TABLE[p2]
                    fraction = (pressure_mbar - p1) / (p2 - p1)
                    return t1 + fraction * (t2 - t1)

            return 40.0  # Default fallback

    @staticmethod
    def _calculate_lmtd(
        t_hot_in: float,
        t_hot_out: float,
        t_cold_in: float,
        t_cold_out: float
    ) -> float:
        """
        Calculate Log Mean Temperature Difference (LMTD).

        For condenser: hot side is condensing steam (constant temp),
        cold side is cooling water.

        Args:
            t_hot_in: Hot fluid inlet temperature (Celsius)
            t_hot_out: Hot fluid outlet temperature (Celsius)
            t_cold_in: Cold fluid inlet temperature (Celsius)
            t_cold_out: Cold fluid outlet temperature (Celsius)

        Returns:
            LMTD in Kelvin (or Celsius difference)
        """
        with _calculation_lock:
            # Temperature differences at each end
            delta_t1 = t_hot_in - t_cold_out  # Hot end
            delta_t2 = t_hot_out - t_cold_in  # Cold end

            # Handle edge cases
            if delta_t1 <= 0 or delta_t2 <= 0:
                logger.warning("Invalid temperature differences for LMTD: dT1=%f, dT2=%f",
                             delta_t1, delta_t2)
                return 0.0

            # If temperature differences are equal, LMTD = delta_t
            if abs(delta_t1 - delta_t2) < 0.01:
                return delta_t1

            # LMTD formula
            lmtd = (delta_t1 - delta_t2) / math.log(delta_t1 / delta_t2)

            return round(lmtd, 2)

    @staticmethod
    def _calculate_ttd(steam_temp: float, cw_outlet_temp: float) -> float:
        """
        Calculate Terminal Temperature Difference (TTD).

        TTD = Steam saturation temperature - Cooling water outlet temperature

        Lower TTD indicates better heat transfer performance.
        Typical design TTD: 2-5 K

        Args:
            steam_temp: Steam saturation temperature (Celsius)
            cw_outlet_temp: Cooling water outlet temperature (Celsius)

        Returns:
            TTD in Kelvin
        """
        ttd = steam_temp - cw_outlet_temp
        return round(max(0, ttd), 2)

    @staticmethod
    def _calculate_u_value(
        heat_duty_kw: float,
        surface_area_m2: float,
        lmtd: float
    ) -> float:
        """
        Calculate overall heat transfer coefficient (U-value).

        Q = U * A * LMTD
        U = Q / (A * LMTD)

        Args:
            heat_duty_kw: Heat duty in kW
            surface_area_m2: Heat transfer surface area in m2
            lmtd: Log mean temperature difference in K

        Returns:
            U-value in W/(m2.K)
        """
        with _calculation_lock:
            if surface_area_m2 <= 0 or lmtd <= 0:
                raise ValueError("Surface area and LMTD must be positive")

            # Convert kW to W
            heat_duty_w = heat_duty_kw * 1000

            u_value = heat_duty_w / (surface_area_m2 * lmtd)

            return round(u_value, 1)

    @staticmethod
    def _calculate_cleanliness_factor(actual_u: float, design_u: float) -> float:
        """
        Calculate cleanliness factor (ratio of actual to design U-value).

        CF = U_actual / U_design

        Typical ranges:
        - 0.95-1.0: Clean
        - 0.85-0.95: Slightly fouled
        - 0.75-0.85: Moderately fouled
        - <0.75: Heavily fouled, cleaning required

        Args:
            actual_u: Actual U-value W/(m2.K)
            design_u: Design U-value W/(m2.K)

        Returns:
            Cleanliness factor (0-1.0)
        """
        if design_u <= 0:
            raise ValueError("Design U-value must be positive")

        cf = actual_u / design_u
        return round(min(1.0, max(0.0, cf)), 3)

    @staticmethod
    def _calculate_fouling_resistance(actual_u: float, clean_u: float) -> float:
        """
        Calculate fouling resistance from U-value degradation.

        1/U_fouled = 1/U_clean + R_fouling
        R_fouling = 1/U_fouled - 1/U_clean

        Args:
            actual_u: Actual (fouled) U-value W/(m2.K)
            clean_u: Clean (design) U-value W/(m2.K)

        Returns:
            Fouling resistance in m2.K/W
        """
        with _calculation_lock:
            if actual_u <= 0 or clean_u <= 0:
                raise ValueError("U-values must be positive")

            if actual_u >= clean_u:
                return 0.0  # No fouling detected

            r_fouling = (1.0 / actual_u) - (1.0 / clean_u)

            return round(r_fouling, 6)

    @staticmethod
    def _estimate_deposit_thickness(
        fouling_resistance: float,
        deposit_conductivity: float = 1.0  # W/(m.K) - typical biofilm
    ) -> float:
        """
        Estimate deposit thickness from fouling resistance.

        R = thickness / k
        thickness = R * k

        Args:
            fouling_resistance: Fouling resistance in m2.K/W
            deposit_conductivity: Thermal conductivity of deposit W/(m.K)

        Returns:
            Estimated deposit thickness in mm
        """
        # R = t/k, so t = R * k
        thickness_m = fouling_resistance * deposit_conductivity
        thickness_mm = thickness_m * 1000

        return round(thickness_mm, 3)

    # =========================================================================
    # TOOL HANDLER IMPLEMENTATIONS
    # =========================================================================

    async def _analyze_condenser_performance(
        self,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze overall condenser performance.

        Implements deterministic performance analysis using HEI standards.
        """
        timestamp = datetime.utcnow().isoformat()

        # Extract parameters
        condenser_id = params["condenser_id"]
        operating_data = params["operating_data"]
        include_trends = params.get("include_trends", True)

        cw_inlet = operating_data["cw_inlet_temp"]
        cw_outlet = operating_data["cw_outlet_temp"]
        vacuum = operating_data["vacuum_pressure"]
        cw_flow = operating_data.get("cw_flow_rate", 10000)  # m3/hr default
        steam_flow = operating_data.get("steam_flow", 50000)  # kg/hr default
        turbine_load = operating_data.get("turbine_load", 100)  # MW default

        # Calculate saturation temperature from vacuum
        sat_temp = self._get_saturation_temperature(vacuum)

        # Calculate heat duty from cooling water side
        # Q = m_dot * Cp * delta_T
        # Cp_water = 4.186 kJ/(kg.K), rho_water = 1000 kg/m3
        cw_mass_flow = cw_flow * 1000 / 3600  # kg/s
        heat_duty_kw = cw_mass_flow * 4.186 * (cw_outlet - cw_inlet)

        # Calculate TTD
        ttd = self._calculate_ttd(sat_temp, cw_outlet)

        # Calculate LMTD (steam side is constant temperature during condensation)
        lmtd = self._calculate_lmtd(sat_temp, sat_temp, cw_inlet, cw_outlet)

        # Estimate surface area from typical condenser sizing
        # Typical heat flux: 30-50 kW/m2
        estimated_area = heat_duty_kw / 40  # m2, using 40 kW/m2

        # Calculate U-value
        if lmtd > 0 and estimated_area > 0:
            u_value = self._calculate_u_value(heat_duty_kw, estimated_area, lmtd)
        else:
            u_value = 0

        # Design U-value (typical for surface condenser)
        design_u = 3000  # W/(m2.K)

        # Calculate cleanliness factor
        cleanliness = self._calculate_cleanliness_factor(u_value, design_u)

        # Calculate DCA (Drain Cooler Approach) - not applicable for standard condenser
        dca = 0.0

        # Calculate efficiency
        # Condenser efficiency based on approach to theoretical minimum
        theoretical_min_vacuum = cw_inlet + 5  # K approach
        theoretical_min_pressure = self._pressure_from_temp(theoretical_min_vacuum)
        efficiency = min(100, max(0, (1 - (vacuum - theoretical_min_pressure) / vacuum) * 100))

        # Determine status
        if cleanliness >= 0.90 and ttd <= 5:
            status = "optimal"
        elif cleanliness >= 0.75 and ttd <= 8:
            status = "degraded"
        else:
            status = "critical"

        # Generate recommendations
        recommendations = []
        if cleanliness < 0.85:
            recommendations.append(
                f"Tube cleaning recommended - cleanliness factor {cleanliness:.2f} "
                f"below target 0.85"
            )
        if ttd > 5:
            recommendations.append(
                f"High TTD ({ttd:.1f}K) indicates potential air binding or "
                f"tube fouling"
            )
        if vacuum > 60:
            recommendations.append(
                f"Elevated backpressure ({vacuum} mbar) - check air removal system "
                f"and cooling water temperature"
            )

        # Prepare trends (placeholder for actual historical data)
        trends = {}
        if include_trends:
            trends = {
                "vacuum_24h": [vacuum] * 24,  # Placeholder
                "ttd_24h": [ttd] * 24,
                "cleanliness_7d": [cleanliness] * 7
            }

        # Calculate provenance
        result_data = {
            "condenser_id": condenser_id,
            "heat_duty": heat_duty_kw,
            "u_value": u_value,
            "cleanliness": cleanliness,
            "lmtd": lmtd,
            "ttd": ttd,
            "vacuum": vacuum,
            "efficiency": efficiency,
            "timestamp": timestamp
        }
        provenance_hash = self._calculate_provenance_hash(result_data)

        return {
            "condenser_id": condenser_id,
            "heat_duty": round(heat_duty_kw, 1),
            "u_value": round(u_value, 1),
            "cleanliness_factor": cleanliness,
            "lmtd": round(lmtd, 2),
            "ttd": round(ttd, 2),
            "dca": dca,
            "vacuum_pressure": vacuum,
            "saturation_temp": round(sat_temp, 1),
            "efficiency": round(efficiency, 1),
            "trends": trends,
            "status": status,
            "recommendations": recommendations,
            "timestamp": timestamp,
            "provenance_hash": provenance_hash
        }

    @staticmethod
    def _pressure_from_temp(temp_c: float) -> float:
        """Estimate pressure from saturation temperature (inverse of table)."""
        # Simple linear approximation for condenser range
        # Based on steam table regression
        pressure = 25 + (temp_c - 21) * 2.5
        return max(25, min(100, pressure))

    async def _calculate_heat_transfer_coefficient(
        self,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate overall heat transfer coefficient with full analysis.

        Implements HEI standard calculation methodology.
        """
        timestamp = datetime.utcnow().isoformat()

        # Extract parameters
        cw_inlet = params["cw_inlet_temp"]
        cw_outlet = params["cw_outlet_temp"]
        cw_flow = params["cw_flow_rate"]  # m3/hr
        steam_temp = params["steam_temp"]
        heat_duty = params["heat_duty"]  # kW

        surface_area = params.get("tube_surface_area", heat_duty / 40)  # Estimate if not provided
        design_u = params.get("design_u_value", 3000)
        tube_od = params.get("tube_od", 25.4) / 1000  # Convert mm to m
        tube_thickness = params.get("tube_thickness", 1.24) / 1000  # Convert mm to m
        tube_material_str = params.get("tube_material", "titanium")

        # Get tube material thermal conductivity
        try:
            tube_material = TubeMaterial(tube_material_str)
            k_tube = TUBE_THERMAL_CONDUCTIVITY[tube_material]
        except (ValueError, KeyError):
            k_tube = 21.9  # Default to titanium

        # Calculate LMTD
        lmtd = self._calculate_lmtd(steam_temp, steam_temp, cw_inlet, cw_outlet)

        # Calculate TTD
        ttd = self._calculate_ttd(steam_temp, cw_outlet)

        # Calculate U-value
        if lmtd > 0 and surface_area > 0:
            u_value = self._calculate_u_value(heat_duty, surface_area, lmtd)
        else:
            raise ValueError("Invalid LMTD or surface area for U-value calculation")

        # Calculate cleanliness factor
        cleanliness = self._calculate_cleanliness_factor(u_value, design_u)

        # Calculate fouling resistance
        fouling_resistance = self._calculate_fouling_resistance(u_value, design_u)

        # Calculate heat flux
        heat_flux = (heat_duty * 1000) / surface_area  # W/m2

        # Estimate tube-side and shell-side coefficients
        # Using typical correlations for condensers

        # Tube side (cooling water) - Dittus-Boelter correlation approximation
        cw_velocity = (cw_flow / 3600) / (surface_area / 100)  # Rough estimate
        h_tube = 3000 + 1500 * min(cw_velocity, 3)  # W/(m2.K) simplified

        # Shell side (condensing steam) - typically high
        h_shell = 8000  # W/(m2.K) typical for steam condensation

        # Calculate provenance
        result_data = {
            "u_value": u_value,
            "design_u": design_u,
            "lmtd": lmtd,
            "ttd": ttd,
            "heat_duty": heat_duty,
            "inputs": params,
            "timestamp": timestamp
        }
        provenance_hash = self._calculate_provenance_hash(result_data)

        return {
            "u_value": round(u_value, 1),
            "u_design": design_u,
            "cleanliness_factor": cleanliness,
            "lmtd": round(lmtd, 2),
            "ttd": round(ttd, 2),
            "heat_duty": round(heat_duty, 1),
            "heat_flux": round(heat_flux, 1),
            "fouling_resistance": round(fouling_resistance, 6),
            "tube_side_coefficient": round(h_tube, 0),
            "shell_side_coefficient": round(h_shell, 0),
            "timestamp": timestamp,
            "provenance_hash": provenance_hash
        }

    async def _optimize_vacuum_pressure(
        self,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize condenser vacuum pressure setpoint.

        Uses thermodynamic relationships to determine achievable vacuum.
        """
        timestamp = datetime.utcnow().isoformat()

        # Extract parameters
        current_vacuum = params["current_vacuum"]
        turbine_load = params["turbine_load"]
        ambient = params["ambient_conditions"]

        dry_bulb = ambient["dry_bulb_temp"]
        wet_bulb = ambient["wet_bulb_temp"]

        cw_inlet = params.get("cw_inlet_temp", wet_bulb + 5)  # Estimate from wet bulb
        ct_approach = params.get("cooling_tower_approach", 5.0)
        design_ttd = params.get("condenser_design_ttd", 3.0)
        bp_limit = params.get("turbine_backpressure_limit", 100)

        # Calculate theoretical minimum CW inlet temp
        # CW inlet = Wet bulb + Cooling tower approach
        min_cw_inlet = wet_bulb + ct_approach

        # Calculate theoretical minimum condenser saturation temp
        # Sat temp = CW outlet + TTD
        # CW outlet = CW inlet + temp rise (typically 8-12 K)
        cw_temp_rise = 10.0  # Typical
        min_sat_temp = min_cw_inlet + cw_temp_rise + design_ttd

        # Convert saturation temperature to pressure
        optimal_vacuum = self._pressure_from_temp(min_sat_temp)

        # Apply constraints
        achievable_vacuum = max(optimal_vacuum, 25)  # Physical minimum
        achievable_vacuum = min(achievable_vacuum, bp_limit)  # Equipment limit

        # Determine limiting factor
        if achievable_vacuum == 25:
            limiting_factor = "Physical minimum vacuum limit"
        elif achievable_vacuum == bp_limit:
            limiting_factor = "Turbine backpressure limit"
        elif cw_inlet > min_cw_inlet + 5:
            limiting_factor = "Elevated cooling water temperature"
        else:
            limiting_factor = "Current conditions near optimal"

        # Calculate efficiency gain
        # Typical: 1% efficiency improvement per 3.4 mbar vacuum reduction
        vacuum_improvement = current_vacuum - achievable_vacuum
        efficiency_gain = vacuum_improvement / 3.4 * 1.0  # Percentage points

        # Calculate power gain
        # Typical: 0.5% power increase per 1% efficiency improvement at design load
        power_gain_percent = efficiency_gain * 0.5
        power_gain_kw = turbine_load * 1000 * power_gain_percent / 100

        # Generate action items
        action_items = []
        if vacuum_improvement > 5:
            action_items.append(
                f"Potential to improve vacuum by {vacuum_improvement:.1f} mbar"
            )
            action_items.append("Check air removal system capacity and operation")
            action_items.append("Verify cooling water flow rate is adequate")
        if cw_inlet > min_cw_inlet + 5:
            action_items.append(
                f"Cooling water inlet temp ({cw_inlet:.1f}C) elevated vs theoretical "
                f"minimum ({min_cw_inlet:.1f}C)"
            )
            action_items.append("Optimize cooling tower operation")

        # Cost benefit analysis
        annual_hours = 8000
        electricity_price = 50  # USD/MWh typical
        annual_benefit = power_gain_kw / 1000 * annual_hours * electricity_price

        cost_benefit = {
            "annual_energy_savings_mwh": round(power_gain_kw / 1000 * annual_hours, 0),
            "annual_cost_savings_usd": round(annual_benefit, 0),
            "efficiency_improvement_percent": round(efficiency_gain, 2)
        }

        # Calculate provenance
        result_data = {
            "current_vacuum": current_vacuum,
            "optimal_vacuum": optimal_vacuum,
            "achievable_vacuum": achievable_vacuum,
            "inputs": params,
            "timestamp": timestamp
        }
        provenance_hash = self._calculate_provenance_hash(result_data)

        return {
            "current_vacuum": current_vacuum,
            "optimal_vacuum": round(optimal_vacuum, 1),
            "expected_efficiency_gain": round(efficiency_gain, 2),
            "expected_power_gain": round(power_gain_kw, 1),
            "limiting_factor": limiting_factor,
            "achievable_vacuum": round(achievable_vacuum, 1),
            "action_items": action_items,
            "cost_benefit": cost_benefit,
            "timestamp": timestamp,
            "provenance_hash": provenance_hash
        }

    async def _detect_air_inleakage(
        self,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect and assess air inleakage into condenser.

        Uses pattern analysis of vacuum trends and ejector data.
        """
        timestamp = datetime.utcnow().isoformat()

        # Extract parameters
        condenser_id = params["condenser_id"]
        vacuum_trend = params["vacuum_trend"]
        ejector_data = params["air_ejector_data"]
        turbine_load = params.get("turbine_load", 100)

        # Analyze vacuum trend
        vacuum_values = vacuum_trend["values"]
        if len(vacuum_values) < 2:
            avg_vacuum = vacuum_values[0] if vacuum_values else 50
            vacuum_std = 0
        else:
            avg_vacuum = sum(vacuum_values) / len(vacuum_values)
            vacuum_std = (sum((v - avg_vacuum) ** 2 for v in vacuum_values) /
                         len(vacuum_values)) ** 0.5

        # Extract ejector data
        current_load = ejector_data["current_load"]  # kg/hr
        design_capacity = ejector_data["design_capacity"]  # kg/hr
        suction_pressure = ejector_data["suction_pressure"]

        # Calculate ejector load percentage
        ejector_load_percent = (current_load / design_capacity) * 100 if design_capacity > 0 else 0

        # Normalize leakage rate by unit size
        normalized_leakage = current_load / turbine_load if turbine_load > 0 else current_load

        # Determine severity based on normalized leakage rate
        if normalized_leakage <= AIR_LEAKAGE_GUIDELINES["excellent"]:
            severity = LeakageSeverity.NONE
        elif normalized_leakage <= AIR_LEAKAGE_GUIDELINES["good"]:
            severity = LeakageSeverity.MINOR
        elif normalized_leakage <= AIR_LEAKAGE_GUIDELINES["acceptable"]:
            severity = LeakageSeverity.MODERATE
        elif normalized_leakage <= AIR_LEAKAGE_GUIDELINES["poor"]:
            severity = LeakageSeverity.SEVERE
        else:
            severity = LeakageSeverity.CRITICAL

        # Estimate vacuum degradation due to air inleakage
        # Typical: 1 mbar per 2 kg/hr excess air at 100 MW
        baseline_leakage = turbine_load * AIR_LEAKAGE_GUIDELINES["good"]
        excess_leakage = max(0, current_load - baseline_leakage)
        vacuum_degradation = excess_leakage / 2

        # Analyze probable locations based on patterns
        probable_locations = []

        if vacuum_std > 2:  # High variability suggests intermittent leaks
            probable_locations.append({
                "location": "Turbine gland seals",
                "probability": 0.7,
                "indicator": "Vacuum fluctuation pattern"
            })

        if ejector_load_percent > 80:
            probable_locations.append({
                "location": "LP turbine exhaust hood",
                "probability": 0.6,
                "indicator": "High air removal load"
            })

        if suction_pressure > avg_vacuum + 5:
            probable_locations.append({
                "location": "Condenser hotwell area",
                "probability": 0.5,
                "indicator": "Ejector suction pressure elevated"
            })

        # Add common leak locations
        probable_locations.append({
            "location": "Expansion joints",
            "probability": 0.4,
            "indicator": "Common leak point"
        })
        probable_locations.append({
            "location": "Valve packing and instrument connections",
            "probability": 0.3,
            "indicator": "Multiple small leaks typical"
        })

        # Calculate detection confidence
        data_points = len(vacuum_values)
        if data_points >= 100 and vacuum_std < 5:
            confidence = 0.9
        elif data_points >= 50:
            confidence = 0.75
        else:
            confidence = 0.5

        # Generate recommendations
        recommendations = []
        if severity in [LeakageSeverity.SEVERE, LeakageSeverity.CRITICAL]:
            recommendations.append("Conduct immediate air inleakage survey")
            recommendations.append("Check turbine gland seal steam supply")
            recommendations.append("Inspect condenser manway doors and gaskets")
        elif severity == LeakageSeverity.MODERATE:
            recommendations.append("Schedule air inleakage survey during next opportunity")
            recommendations.append("Monitor ejector performance trends")
        else:
            recommendations.append("Continue routine monitoring")

        # Determine repair priority
        priority_map = {
            LeakageSeverity.NONE: 5,
            LeakageSeverity.MINOR: 4,
            LeakageSeverity.MODERATE: 3,
            LeakageSeverity.SEVERE: 2,
            LeakageSeverity.CRITICAL: 1
        }
        repair_priority = priority_map[severity]

        # Calculate provenance
        result_data = {
            "condenser_id": condenser_id,
            "leakage_rate": current_load,
            "severity": severity.value,
            "vacuum_degradation": vacuum_degradation,
            "timestamp": timestamp
        }
        provenance_hash = self._calculate_provenance_hash(result_data)

        return {
            "condenser_id": condenser_id,
            "estimated_leakage_rate": round(current_load, 2),
            "severity": severity.value,
            "probable_locations": probable_locations,
            "vacuum_degradation": round(vacuum_degradation, 1),
            "ejector_load_percent": round(ejector_load_percent, 1),
            "detection_confidence": confidence,
            "recommended_actions": recommendations,
            "estimated_repair_priority": repair_priority,
            "timestamp": timestamp,
            "provenance_hash": provenance_hash
        }

    async def _calculate_fouling_factor(
        self,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate fouling factor and resistance per HEI standards.
        """
        timestamp = datetime.utcnow().isoformat()

        # Extract parameters
        design_u = params["design_U"]
        actual_u = params["actual_U"]
        tube_material_str = params["tube_material"]
        operating_hours = params["operating_hours"]
        water_source = params.get("water_source", "treated_cooling_tower")
        historical_data = params.get("historical_fouling_data", [])

        # Calculate fouling resistance
        fouling_resistance = self._calculate_fouling_resistance(actual_u, design_u)

        # Calculate cleanliness factor
        fouling_factor = self._calculate_cleanliness_factor(actual_u, design_u)

        # Get HEI reference fouling resistance
        hei_reference = HEI_FOULING_RESISTANCE.get(water_source, 0.000088)

        # Estimate deposit thickness
        deposit_thickness = self._estimate_deposit_thickness(fouling_resistance)

        # Calculate degradation rate
        if operating_hours > 0:
            degradation_rate = fouling_resistance / (operating_hours / 1000)  # per 1000 hours
        else:
            degradation_rate = 0

        # Analyze historical trend if available
        if historical_data and len(historical_data) >= 2:
            # Simple linear regression for trend
            hours = [d["operating_hours"] for d in historical_data]
            resistances = [d["fouling_resistance"] for d in historical_data]
            if len(hours) >= 2:
                # Calculate slope
                n = len(hours)
                sum_x = sum(hours)
                sum_y = sum(resistances)
                sum_xy = sum(h * r for h, r in zip(hours, resistances))
                sum_x2 = sum(h ** 2 for h in hours)

                denom = n * sum_x2 - sum_x ** 2
                if denom != 0:
                    slope = (n * sum_xy - sum_x * sum_y) / denom
                    degradation_rate = slope * 1000  # per 1000 hours

        # Determine cleaning recommendation
        if fouling_factor < 0.70:
            cleaning_recommended = True
            cleaning_urgency = "immediate"
        elif fouling_factor < 0.80:
            cleaning_recommended = True
            cleaning_urgency = "scheduled"
        elif fouling_factor < 0.85:
            cleaning_recommended = False
            cleaning_urgency = "monitor"
        else:
            cleaning_recommended = False
            cleaning_urgency = "not_required"

        # Calculate expected improvement from cleaning
        expected_improvement = ((design_u - actual_u) / actual_u) * 100

        # Estimate time to critical fouling (CF < 0.70)
        if degradation_rate > 0 and fouling_factor > 0.70:
            # CF = U_actual / U_design
            # We need to find hours until CF = 0.70
            target_u = 0.70 * design_u
            current_rf = fouling_resistance
            target_rf = (1 / target_u) - (1 / design_u)
            rf_remaining = target_rf - current_rf

            if rf_remaining > 0:
                time_to_critical = (rf_remaining / degradation_rate) * 1000  # hours
            else:
                time_to_critical = 0
        else:
            time_to_critical = None

        # Calculate provenance
        result_data = {
            "design_U": design_u,
            "actual_U": actual_u,
            "fouling_resistance": fouling_resistance,
            "fouling_factor": fouling_factor,
            "operating_hours": operating_hours,
            "timestamp": timestamp
        }
        provenance_hash = self._calculate_provenance_hash(result_data)

        return {
            "fouling_resistance": round(fouling_resistance, 6),
            "fouling_factor": round(fouling_factor, 3),
            "degradation_rate": round(degradation_rate, 8),
            "estimated_deposit_thickness": round(deposit_thickness, 3),
            "cleaning_recommended": cleaning_recommended,
            "cleaning_urgency": cleaning_urgency,
            "expected_improvement": round(expected_improvement, 1),
            "time_to_critical": round(time_to_critical, 0) if time_to_critical else None,
            "timestamp": timestamp,
            "provenance_hash": provenance_hash
        }

    async def _predict_tube_cleaning_schedule(
        self,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict optimal tube cleaning schedule based on fouling trends.
        """
        timestamp = datetime.utcnow().isoformat()

        # Extract parameters
        condenser_id = params["condenser_id"]
        fouling_trend = params["fouling_trend"]
        current_cleanliness = params["current_cleanliness"]
        cleaning_threshold = params.get("cleaning_threshold", 0.75)
        cleaning_cost = params.get("cleaning_cost", 50000)
        unit_capacity = params.get("unit_capacity", 100)  # MW
        production_schedule = params.get("production_schedule", {})

        # Analyze fouling trend
        cf_values = fouling_trend["cleanliness_factors"]
        timestamps = fouling_trend["timestamps"]

        if len(cf_values) < 2:
            # Not enough data, use default degradation rate
            degradation_rate = 0.001  # CF units per day
        else:
            # Calculate degradation rate from trend
            # Parse timestamps and calculate days
            first_time = datetime.fromisoformat(timestamps[0].replace('Z', '+00:00'))
            last_time = datetime.fromisoformat(timestamps[-1].replace('Z', '+00:00'))
            days = (last_time - first_time).days

            if days > 0:
                cf_drop = cf_values[0] - cf_values[-1]
                degradation_rate = cf_drop / days
            else:
                degradation_rate = 0.001

        # Ensure positive degradation rate
        degradation_rate = max(0.0001, degradation_rate)

        # Calculate days until cleaning threshold reached
        cf_to_threshold = current_cleanliness - cleaning_threshold
        days_until_cleaning = int(cf_to_threshold / degradation_rate) if degradation_rate > 0 else 365
        days_until_cleaning = max(0, min(365, days_until_cleaning))

        # Calculate recommended cleaning date
        recommended_date = datetime.utcnow() + timedelta(days=days_until_cleaning)

        # Check against production schedule for outage windows
        planned_outages = production_schedule.get("planned_outages", [])
        for outage in planned_outages:
            outage_start = datetime.fromisoformat(outage["start_date"])
            outage_end = datetime.fromisoformat(outage["end_date"])

            # If outage is before predicted cleaning need and close enough
            if outage_start <= recommended_date <= outage_end:
                recommended_date = outage_start
                days_until_cleaning = (outage_start - datetime.utcnow()).days
                break
            elif outage_start > datetime.utcnow() and outage_start < recommended_date:
                # Use earlier outage if cleaning can wait
                recommended_date = outage_start
                days_until_cleaning = (outage_start - datetime.utcnow()).days
                break

        # Select cleaning method based on fouling severity
        if current_cleanliness < 0.70:
            cleaning_method = CleaningMethod.CHEMICAL_CLEANING
        elif current_cleanliness < 0.80:
            cleaning_method = CleaningMethod.HIGH_PRESSURE_WATER
        else:
            cleaning_method = CleaningMethod.MECHANICAL_BRUSHING

        # Estimate cleaning duration
        duration_map = {
            CleaningMethod.MECHANICAL_BRUSHING: 12,
            CleaningMethod.HIGH_PRESSURE_WATER: 24,
            CleaningMethod.CHEMICAL_CLEANING: 48,
            CleaningMethod.RUBBER_BALL_SYSTEM: 0,  # Online cleaning
            CleaningMethod.SPONGE_BALL_SYSTEM: 0
        }
        estimated_duration = duration_map.get(cleaning_method, 24)

        # Calculate expected recovery
        expected_recovery = min(100, ((1.0 - current_cleanliness) / (1.0 - 0.6)) * 100)
        expected_u_recovery = expected_recovery * 0.9  # 90% of theoretical

        # Estimate efficiency gain
        # 1% cleanliness improvement ~ 0.1% turbine efficiency
        efficiency_gain = expected_u_recovery * 0.1 / 100

        # Calculate production impact
        electricity_price = production_schedule.get("electricity_price_forecast", 50)  # USD/MWh
        production_loss_mwh = unit_capacity * estimated_duration
        production_impact_cost = production_loss_mwh * electricity_price

        # Calculate net benefit
        # Annual benefit from cleaning
        annual_hours = 8000
        power_improvement = unit_capacity * efficiency_gain / 100
        annual_benefit = power_improvement * annual_hours * electricity_price

        # Net benefit = annual benefit - cleaning cost - production impact
        net_benefit = annual_benefit - cleaning_cost - production_impact_cost

        # Calculate confidence level
        data_points = len(cf_values)
        if data_points >= 30:
            confidence = 0.9
        elif data_points >= 14:
            confidence = 0.75
        elif data_points >= 7:
            confidence = 0.6
        else:
            confidence = 0.4

        # Calculate provenance
        result_data = {
            "condenser_id": condenser_id,
            "current_cleanliness": current_cleanliness,
            "recommended_date": recommended_date.isoformat(),
            "cleaning_method": cleaning_method.value,
            "timestamp": timestamp
        }
        provenance_hash = self._calculate_provenance_hash(result_data)

        return {
            "condenser_id": condenser_id,
            "recommended_cleaning_date": recommended_date.strftime("%Y-%m-%d"),
            "days_until_cleaning": days_until_cleaning,
            "cleaning_method": cleaning_method.value,
            "expected_u_value_recovery": round(expected_u_recovery, 1),
            "expected_efficiency_gain": round(efficiency_gain, 3),
            "estimated_duration": estimated_duration,
            "estimated_cost": cleaning_cost,
            "production_impact": round(production_loss_mwh, 0),
            "net_benefit": round(net_benefit, 0),
            "confidence_level": confidence,
            "timestamp": timestamp,
            "provenance_hash": provenance_hash
        }

    async def _optimize_cooling_water_flow(
        self,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize cooling water flow rate for target vacuum.

        Uses heat transfer and pump affinity law calculations.
        """
        timestamp = datetime.utcnow().isoformat()

        # Extract parameters
        heat_duty = params["heat_duty"]  # kW
        cw_inlet = params["cw_inlet_temp"]
        target_vacuum = params["target_vacuum"]
        current_flow = params.get("current_flow_rate", 10000)  # m3/hr
        pump_chars = params.get("pump_characteristics", {})
        surface_area = params.get("condenser_surface_area", heat_duty / 40)
        design_u = params.get("design_u_value", 3000)
        electricity_cost = params.get("electricity_cost", 0.08)

        # Get pump characteristics
        design_flow = pump_chars.get("design_flow", current_flow)
        design_power = pump_chars.get("design_power", 500)  # kW
        pump_efficiency = pump_chars.get("efficiency", 0.8)
        min_flow = pump_chars.get("min_flow", design_flow * 0.6)
        max_flow = pump_chars.get("max_flow", design_flow * 1.1)
        vfd_equipped = pump_chars.get("vfd_equipped", True)

        # Calculate target saturation temperature from vacuum
        target_sat_temp = self._get_saturation_temperature(target_vacuum)

        # Calculate required CW outlet temperature to achieve target vacuum
        design_ttd = 3.0  # K, typical
        required_cw_outlet = target_sat_temp - design_ttd

        # Calculate required CW temperature rise
        required_temp_rise = required_cw_outlet - cw_inlet

        if required_temp_rise <= 0:
            return {
                "error": "Target vacuum not achievable with current CW inlet temperature",
                "detail": f"CW inlet ({cw_inlet}C) too high for target vacuum ({target_vacuum} mbar)",
                "timestamp": timestamp,
                "provenance_hash": ""
            }

        # Calculate required flow rate
        # Q = m_dot * Cp * dT
        # m_dot = Q / (Cp * dT)
        cp_water = 4.186  # kJ/(kg.K)
        required_mass_flow = (heat_duty) / (cp_water * required_temp_rise)  # kg/s
        optimal_flow = (required_mass_flow * 3600) / 1000  # m3/hr

        # Apply constraints
        constraints = []

        if optimal_flow < min_flow:
            optimal_flow = min_flow
            constraints.append(f"Flow limited by minimum pump flow ({min_flow} m3/hr)")

        if optimal_flow > max_flow:
            optimal_flow = max_flow
            constraints.append(f"Flow limited by maximum pump capacity ({max_flow} m3/hr)")

        if not vfd_equipped and abs(optimal_flow - design_flow) > design_flow * 0.1:
            constraints.append("Fixed speed pump limits flow optimization")
            optimal_flow = design_flow

        # Calculate pump power using affinity laws
        # P2/P1 = (Q2/Q1)^3 for VFD, or constant for fixed speed
        if vfd_equipped:
            pump_power_optimal = design_power * (optimal_flow / design_flow) ** 3
            pump_power_current = design_power * (current_flow / design_flow) ** 3
        else:
            pump_power_optimal = design_power
            pump_power_current = design_power

        # Calculate energy savings
        pump_energy_savings = pump_power_current - pump_power_optimal

        # Calculate actual achieved vacuum at optimal flow
        actual_temp_rise = heat_duty / (optimal_flow * 1000 / 3600 * cp_water)
        actual_cw_outlet = cw_inlet + actual_temp_rise
        actual_sat_temp = actual_cw_outlet + design_ttd
        achieved_vacuum = self._pressure_from_temp(actual_sat_temp)

        vacuum_impact = achieved_vacuum - target_vacuum

        # Calculate efficiency impact
        # 1% vacuum improvement ~ 0.3% efficiency
        efficiency_impact = (current_flow - optimal_flow) / current_flow * 0.3 if current_flow > 0 else 0

        # Calculate annual savings
        annual_hours = 8000
        annual_savings = pump_energy_savings * annual_hours * electricity_cost

        # Calculate provenance
        result_data = {
            "heat_duty": heat_duty,
            "cw_inlet": cw_inlet,
            "target_vacuum": target_vacuum,
            "optimal_flow": optimal_flow,
            "timestamp": timestamp
        }
        provenance_hash = self._calculate_provenance_hash(result_data)

        return {
            "current_flow_rate": round(current_flow, 0),
            "optimal_flow_rate": round(optimal_flow, 0),
            "flow_change": round((optimal_flow - current_flow) / current_flow * 100, 1) if current_flow > 0 else 0,
            "pump_power_current": round(pump_power_current, 1),
            "pump_power_optimal": round(pump_power_optimal, 1),
            "pump_energy_savings": round(pump_energy_savings, 1),
            "vacuum_impact": round(vacuum_impact, 1),
            "efficiency_impact": round(efficiency_impact, 2),
            "annual_savings": round(annual_savings, 0),
            "constraints": constraints,
            "timestamp": timestamp,
            "provenance_hash": provenance_hash
        }

    async def _generate_performance_report(
        self,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.

        Aggregates metrics and provides summary statistics.
        Note: Narrative generation may use LLM assistance.
        """
        timestamp = datetime.utcnow().isoformat()

        # Extract parameters
        condenser_id = params["condenser_id"]
        report_period = params["report_period"]
        metrics_to_include = params.get("metrics_to_include",
            ["heat_duty", "u_value", "cleanliness_factor", "vacuum_pressure"])
        operating_summary = params.get("operating_data_summary", {})
        include_recommendations = params.get("include_recommendations", True)
        include_cost = params.get("include_cost_analysis", True)

        period_str = f"{report_period['start']} to {report_period['end']}"

        # Build summary metrics
        summary_metrics = {}

        avg_heat_duty = operating_summary.get("avg_heat_duty", 0)
        avg_vacuum = operating_summary.get("avg_vacuum", 50)
        avg_u_value = operating_summary.get("avg_u_value", 2500)
        avg_cleanliness = operating_summary.get("avg_cleanliness", 0.85)
        total_hours = operating_summary.get("total_operating_hours", 720)
        avg_load = operating_summary.get("avg_turbine_load", 100)

        if "heat_duty" in metrics_to_include:
            summary_metrics["heat_duty_avg_kw"] = avg_heat_duty
            summary_metrics["heat_duty_max_kw"] = avg_heat_duty * 1.2
            summary_metrics["heat_duty_min_kw"] = avg_heat_duty * 0.8

        if "u_value" in metrics_to_include:
            summary_metrics["u_value_avg"] = avg_u_value
            summary_metrics["u_value_trend"] = "declining" if avg_cleanliness < 0.9 else "stable"

        if "cleanliness_factor" in metrics_to_include:
            summary_metrics["cleanliness_factor_avg"] = avg_cleanliness
            summary_metrics["cleanliness_factor_min"] = avg_cleanliness * 0.95

        if "vacuum_pressure" in metrics_to_include:
            summary_metrics["vacuum_avg_mbar"] = avg_vacuum
            summary_metrics["vacuum_best_mbar"] = avg_vacuum * 0.9
            summary_metrics["vacuum_worst_mbar"] = avg_vacuum * 1.15

        # Trend analysis
        trend_analysis = {
            "cleanliness_trend": "declining" if avg_cleanliness < 0.85 else "stable",
            "vacuum_trend": "stable",
            "performance_trajectory": "monitoring_required" if avg_cleanliness < 0.85 else "satisfactory"
        }

        # Efficiency statistics
        design_u = 3000
        actual_efficiency = (avg_u_value / design_u) * 100
        efficiency_statistics = {
            "thermal_efficiency_percent": round(actual_efficiency, 1),
            "design_efficiency_percent": 100.0,
            "efficiency_gap_percent": round(100 - actual_efficiency, 1),
            "operating_hours": total_hours,
            "availability_percent": round(total_hours / (30 * 24) * 100, 1)
        }

        # Fouling status
        fouling_status = {
            "current_cleanliness": avg_cleanliness,
            "cleaning_threshold": 0.75,
            "estimated_days_to_cleaning": int((avg_cleanliness - 0.75) / 0.001) if avg_cleanliness > 0.75 else 0,
            "last_cleaning_date": "N/A",
            "fouling_rate": "normal" if avg_cleanliness > 0.85 else "accelerated"
        }

        # Maintenance events (placeholder)
        maintenance_events = []

        # Recommendations
        recommendations = []
        if include_recommendations:
            if avg_cleanliness < 0.85:
                recommendations.append(
                    f"Schedule tube cleaning - cleanliness factor ({avg_cleanliness:.2f}) "
                    f"below optimal (0.85)"
                )
            if avg_vacuum > 55:
                recommendations.append(
                    f"Investigate elevated vacuum pressure ({avg_vacuum} mbar) - "
                    f"check air removal system"
                )
            if actual_efficiency < 85:
                recommendations.append(
                    f"Thermal efficiency ({actual_efficiency:.1f}%) below target - "
                    f"consider performance improvement program"
                )
            if not recommendations:
                recommendations.append("Condenser operating within normal parameters")

        # KPIs
        kpis = {
            "cleanliness_factor": avg_cleanliness,
            "ttd_design_margin": 1.0,  # Placeholder
            "vacuum_deviation_mbar": avg_vacuum - 45,  # vs target
            "heat_rate_impact_percent": round((100 - actual_efficiency) * 0.1, 2),
            "availability_percent": efficiency_statistics["availability_percent"]
        }

        # Calculate provenance
        result_data = {
            "condenser_id": condenser_id,
            "report_period": period_str,
            "summary_metrics": summary_metrics,
            "kpis": kpis,
            "timestamp": timestamp
        }
        provenance_hash = self._calculate_provenance_hash(result_data)

        return {
            "condenser_id": condenser_id,
            "report_period": period_str,
            "summary_metrics": summary_metrics,
            "trend_analysis": trend_analysis,
            "efficiency_statistics": efficiency_statistics,
            "fouling_status": fouling_status,
            "maintenance_events": maintenance_events,
            "recommendations": recommendations,
            "kpis": kpis,
            "timestamp": timestamp,
            "provenance_hash": provenance_hash
        }

    async def _calculate_condenser_duty(
        self,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate condenser heat duty using specified method.
        """
        timestamp = datetime.utcnow().isoformat()

        method = params["calculation_method"]

        if method == "cooling_water":
            # Q = m_dot * Cp * dT
            cw_inlet = params.get("cw_inlet_temp", 20)
            cw_outlet = params.get("cw_outlet_temp", 30)
            cw_flow = params.get("cw_flow_rate", 10000)  # m3/hr

            temp_rise = cw_outlet - cw_inlet
            mass_flow = cw_flow * 1000 / 3600  # kg/s
            cp = 4.186  # kJ/(kg.K)

            heat_duty = mass_flow * cp * temp_rise  # kW
            specific_duty = heat_duty / (mass_flow * 3600) if mass_flow > 0 else 0

        elif method == "steam_side":
            # Q = m_dot * h_fg
            steam_flow = params.get("steam_flow", 50000)  # kg/hr
            steam_quality = params.get("steam_quality", 0.9)
            condenser_pressure = params.get("condenser_pressure", 50)  # mbar

            # Get latent heat at condenser pressure
            # Simplified: h_fg ~ 2400 kJ/kg at typical condenser conditions
            h_fg = 2400  # kJ/kg

            mass_flow = steam_flow / 3600  # kg/s
            heat_duty = mass_flow * h_fg * steam_quality  # kW

            # Estimate CW temp rise
            assumed_cw_flow = 10000  # m3/hr
            temp_rise = heat_duty / (assumed_cw_flow * 1000 / 3600 * 4.186)
            specific_duty = heat_duty / steam_flow if steam_flow > 0 else 0

        else:
            raise ValueError(f"Unknown calculation method: {method}")

        result_data = {
            "method": method,
            "heat_duty": heat_duty,
            "timestamp": timestamp
        }
        provenance_hash = self._calculate_provenance_hash(result_data)

        return {
            "heat_duty": round(heat_duty, 1),
            "cw_temp_rise": round(temp_rise, 2),
            "specific_duty": round(specific_duty, 4),
            "calculation_method": method,
            "timestamp": timestamp,
            "provenance_hash": provenance_hash
        }

    async def _assess_condenser_health(
        self,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive health assessment of condenser system.
        """
        timestamp = datetime.utcnow().isoformat()

        condenser_id = params["condenser_id"]
        cleanliness = params["cleanliness_factor"]
        air_leakage = params.get("air_inleakage_rate", 2.0)
        vacuum_deviation = params.get("vacuum_deviation", 0)
        ttd_deviation = params.get("ttd_deviation", 0)
        tube_pluggage = params.get("tube_pluggage_percent", 0)
        hours_since_overhaul = params.get("operating_hours_since_overhaul", 8000)
        unit_capacity = params.get("unit_capacity", 100)

        # Score each component (0-100)
        component_scores = {}

        # Tube cleanliness score
        cleanliness_score = cleanliness * 100
        component_scores["tube_cleanliness"] = round(cleanliness_score, 1)

        # Air tightness score
        normalized_leakage = air_leakage / unit_capacity
        if normalized_leakage <= 0.5:
            air_score = 100
        elif normalized_leakage <= 1.0:
            air_score = 90
        elif normalized_leakage <= 2.0:
            air_score = 70
        elif normalized_leakage <= 4.0:
            air_score = 50
        else:
            air_score = 30
        component_scores["air_tightness"] = air_score

        # Vacuum performance score
        if vacuum_deviation <= 2:
            vacuum_score = 100
        elif vacuum_deviation <= 5:
            vacuum_score = 85
        elif vacuum_deviation <= 10:
            vacuum_score = 70
        else:
            vacuum_score = 50
        component_scores["vacuum_performance"] = vacuum_score

        # TTD performance score
        if ttd_deviation <= 1:
            ttd_score = 100
        elif ttd_deviation <= 2:
            ttd_score = 85
        elif ttd_deviation <= 4:
            ttd_score = 70
        else:
            ttd_score = 50
        component_scores["ttd_performance"] = ttd_score

        # Tube integrity score
        if tube_pluggage <= 1:
            tube_integrity_score = 100
        elif tube_pluggage <= 3:
            tube_integrity_score = 85
        elif tube_pluggage <= 5:
            tube_integrity_score = 70
        elif tube_pluggage <= 10:
            tube_integrity_score = 50
        else:
            tube_integrity_score = 30
        component_scores["tube_integrity"] = tube_integrity_score

        # Calculate overall health score (weighted average)
        weights = {
            "tube_cleanliness": 0.25,
            "air_tightness": 0.20,
            "vacuum_performance": 0.25,
            "ttd_performance": 0.15,
            "tube_integrity": 0.15
        }

        overall_score = sum(
            component_scores[comp] * weight
            for comp, weight in weights.items()
        )

        # Determine health status
        if overall_score >= 90:
            health_status = "excellent"
        elif overall_score >= 80:
            health_status = "good"
        elif overall_score >= 70:
            health_status = "fair"
        elif overall_score >= 60:
            health_status = "poor"
        else:
            health_status = "critical"

        # Identify risk factors
        risk_factors = []
        if cleanliness_score < 80:
            risk_factors.append("Tube fouling affecting heat transfer")
        if air_score < 70:
            risk_factors.append("Elevated air inleakage degrading vacuum")
        if tube_integrity_score < 70:
            risk_factors.append("High tube pluggage reducing surface area")
        if hours_since_overhaul > 40000:
            risk_factors.append("Extended time since major overhaul")

        # Recommended actions
        recommended_actions = []
        if cleanliness_score < 85:
            recommended_actions.append("Schedule tube cleaning")
        if air_score < 80:
            recommended_actions.append("Conduct air inleakage survey")
        if tube_integrity_score < 80:
            recommended_actions.append("Evaluate tube replacement needs")
        if not recommended_actions:
            recommended_actions.append("Continue routine monitoring")

        # Next inspection date
        if overall_score >= 85:
            days_to_inspection = 90
        elif overall_score >= 70:
            days_to_inspection = 30
        else:
            days_to_inspection = 7

        next_inspection = (datetime.utcnow() + timedelta(days=days_to_inspection)).strftime("%Y-%m-%d")

        result_data = {
            "condenser_id": condenser_id,
            "overall_score": overall_score,
            "component_scores": component_scores,
            "timestamp": timestamp
        }
        provenance_hash = self._calculate_provenance_hash(result_data)

        return {
            "condenser_id": condenser_id,
            "overall_health_score": round(overall_score, 1),
            "component_scores": component_scores,
            "health_status": health_status,
            "risk_factors": risk_factors,
            "recommended_actions": recommended_actions,
            "next_inspection_date": next_inspection,
            "timestamp": timestamp,
            "provenance_hash": provenance_hash
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_tool_definitions() -> List[Dict[str, Any]]:
    """
    Get list of all available tool definitions.

    Returns:
        List of tool definition dictionaries for LLM integration
    """
    return CONDENSER_TOOLS.copy()


def get_tool_by_name(name: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific tool definition by name.

    Args:
        name: Tool name to look up

    Returns:
        Tool definition dictionary or None if not found
    """
    for tool in CONDENSER_TOOLS:
        if tool["name"] == name:
            return tool.copy()
    return None


def validate_tool_input(tool_name: str, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate input parameters for a tool.

    Args:
        tool_name: Name of the tool
        parameters: Input parameters to validate

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    tool_def = get_tool_by_name(tool_name)
    if tool_def is None:
        return False, [f"Unknown tool: {tool_name}"]

    errors = []
    schema = tool_def.get("input_schema", {})
    required = schema.get("required", [])
    properties = schema.get("properties", {})

    # Check required parameters
    for req in required:
        if req not in parameters:
            errors.append(f"Missing required parameter: {req}")

    # Validate parameter types and constraints
    for param_name, param_value in parameters.items():
        if param_name not in properties:
            continue

        prop_def = properties[param_name]

        # Type checking
        expected_type = prop_def.get("type")
        if expected_type == "number" and not isinstance(param_value, (int, float)):
            errors.append(f"Parameter {param_name} must be a number")
        elif expected_type == "string" and not isinstance(param_value, str):
            errors.append(f"Parameter {param_name} must be a string")
        elif expected_type == "boolean" and not isinstance(param_value, bool):
            errors.append(f"Parameter {param_name} must be a boolean")
        elif expected_type == "array" and not isinstance(param_value, list):
            errors.append(f"Parameter {param_name} must be an array")
        elif expected_type == "object" and not isinstance(param_value, dict):
            errors.append(f"Parameter {param_name} must be an object")

        # Range checking for numbers
        if expected_type == "number" and isinstance(param_value, (int, float)):
            if "minimum" in prop_def and param_value < prop_def["minimum"]:
                errors.append(
                    f"Parameter {param_name} ({param_value}) below minimum ({prop_def['minimum']})"
                )
            if "maximum" in prop_def and param_value > prop_def["maximum"]:
                errors.append(
                    f"Parameter {param_name} ({param_value}) above maximum ({prop_def['maximum']})"
                )

        # Enum checking
        if "enum" in prop_def and param_value not in prop_def["enum"]:
            errors.append(
                f"Parameter {param_name} must be one of: {prop_def['enum']}"
            )

    return len(errors) == 0, errors


def calculate_provenance_hash(data: Any) -> str:
    """
    Calculate SHA-256 provenance hash for audit trail.

    Args:
        data: Data to hash (will be JSON serialized)

    Returns:
        SHA-256 hash string
    """
    data_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(data_str.encode()).hexdigest()


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info(
    "CONDENSYNC tools module loaded with %d tools",
    len(CONDENSER_TOOLS)
)
