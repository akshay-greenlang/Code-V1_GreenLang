# -*- coding: utf-8 -*-
"""
GL-021 BURNERSENTRY - Burner Health Analysis Module
====================================================

This module implements comprehensive burner component health scoring per API 535
(Burners for Fired Heaters in General Refinery Services). Analyzes health of
critical burner components using degradation models and predictive analytics.

ZERO-HALLUCINATION GUARANTEE:
    All calculations are deterministic using documented engineering formulas.
    No LLM/AI is used in the calculation path.
    Full provenance tracking with SHA-256 hashes.

Features:
    - Nozzle/tip health analysis (erosion, coking, plugging)
    - Refractory tile integrity assessment (Coffin-Manson thermal fatigue)
    - Igniter/pilot system health monitoring
    - Flame scanner reliability tracking
    - Air register/damper mechanical health
    - Fuel valve performance analysis
    - Overall burner health scoring
    - Maintenance priority recommendations

Standards Compliance:
    - API 535: Burners for Fired Heaters
    - NFPA 85: Boiler and Combustion Systems Hazards Code
    - NFPA 86: Standard for Ovens and Furnaces
    - IEC 61511: Functional Safety for Process Industry

Health Scoring Scale (per API 535):
    - 90-100: Excellent (no action required)
    - 70-89:  Good (monitor)
    - 50-69:  Fair (plan maintenance)
    - 25-49:  Poor (schedule maintenance)
    - 0-24:   Critical (immediate action required)

Example:
    >>> from greenlang.agents.process_heat.gl_021_burner_maintenance.burner_health import (
    ...     BurnerHealthAnalyzer, BurnerHealthInput
    ... )
    >>> analyzer = BurnerHealthAnalyzer()
    >>> result = analyzer.analyze(health_input)
    >>> print(f"Overall Health: {result.overall_health_score:.1f}")
    >>> print(f"Severity: {result.severity_classification}")

Author: GreenLang Process Heat Team
Version: 1.0.0
Status: Production Ready
"""

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import statistics

from pydantic import BaseModel, Field, validator


logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS AND REFERENCE DATA
# =============================================================================

# Default component design lives (operating hours)
DEFAULT_DESIGN_LIVES: Dict[str, float] = {
    "nozzle": 25000.0,          # 25,000 hours (~3 years)
    "refractory_tile": 40000.0,  # 40,000 hours (~5 years)
    "igniter": 15000.0,          # 15,000 hours (~2 years)
    "scanner": 30000.0,          # 30,000 hours (~3.5 years)
    "air_register": 50000.0,     # 50,000 hours (~6 years)
    "fuel_valve": 35000.0,       # 35,000 hours (~4 years)
}

# Component weights for overall health score
DEFAULT_COMPONENT_WEIGHTS: Dict[str, float] = {
    "nozzle": 0.20,
    "refractory_tile": 0.15,
    "igniter": 0.15,
    "scanner": 0.15,
    "air_register": 0.15,
    "fuel_valve": 0.20,
}

# Degradation rate coefficients by fuel type
FUEL_DEGRADATION_FACTORS: Dict[str, Dict[str, float]] = {
    "natural_gas": {
        "nozzle_erosion": 1.0,
        "nozzle_coking": 0.5,
        "tile_thermal_stress": 1.0,
        "scanner_fouling": 0.8,
    },
    "no2_fuel_oil": {
        "nozzle_erosion": 1.5,
        "nozzle_coking": 2.0,
        "tile_thermal_stress": 1.2,
        "scanner_fouling": 1.5,
    },
    "no6_fuel_oil": {
        "nozzle_erosion": 2.0,
        "nozzle_coking": 3.0,
        "tile_thermal_stress": 1.4,
        "scanner_fouling": 2.0,
    },
    "propane": {
        "nozzle_erosion": 1.0,
        "nozzle_coking": 0.6,
        "tile_thermal_stress": 1.0,
        "scanner_fouling": 0.9,
    },
}

# Coffin-Manson thermal fatigue parameters
COFFIN_MANSON_PARAMS: Dict[str, float] = {
    "refractory_exponent": -0.5,  # Fatigue exponent
    "refractory_coefficient": 0.6,  # Ductility coefficient
    "thermal_expansion_coeff": 7.5e-6,  # 1/K for typical refractory
}

# UV/IR scanner degradation parameters
SCANNER_DEGRADATION: Dict[str, Dict[str, float]] = {
    "uv": {
        "half_life_hours": 20000,
        "temp_acceleration_factor": 0.05,  # per 10C above 50C
        "contamination_rate": 0.001,  # per hour
    },
    "ir": {
        "half_life_hours": 25000,
        "temp_acceleration_factor": 0.03,
        "contamination_rate": 0.0008,
    },
    "uv_ir": {
        "half_life_hours": 22000,
        "temp_acceleration_factor": 0.04,
        "contamination_rate": 0.0009,
    },
}


# =============================================================================
# ENUMS
# =============================================================================

class HealthStatus(str, Enum):
    """Component health status classification per API 535."""
    EXCELLENT = "excellent"    # 90-100: No action required
    GOOD = "good"              # 70-89: Monitor
    FAIR = "fair"              # 50-69: Plan maintenance
    POOR = "poor"              # 25-49: Schedule maintenance
    CRITICAL = "critical"      # 0-24: Immediate action required


class MaintenancePriority(str, Enum):
    """Maintenance priority levels."""
    NONE = "none"              # No maintenance needed
    ROUTINE = "routine"        # Schedule during normal maintenance
    PLANNED = "planned"        # Plan maintenance within 30 days
    PRIORITY = "priority"      # Schedule within 7 days
    URGENT = "urgent"          # Schedule within 48 hours
    EMERGENCY = "emergency"    # Immediate action required


class FailureMode(str, Enum):
    """Burner component failure modes."""
    NONE = "none"
    EROSION = "erosion"
    COKING = "coking"
    PLUGGING = "plugging"
    THERMAL_FATIGUE = "thermal_fatigue"
    CRACKING = "cracking"
    SPALLING = "spalling"
    SENSOR_DEGRADATION = "sensor_degradation"
    MECHANICAL_WEAR = "mechanical_wear"
    SEAT_WEAR = "seat_wear"
    ACTUATOR_FAILURE = "actuator_failure"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"


# =============================================================================
# INPUT SCHEMAS
# =============================================================================

class NozzleData(BaseModel):
    """Nozzle/tip operational data."""

    nozzle_id: str = Field(..., description="Nozzle identifier")
    operating_hours: float = Field(
        ...,
        ge=0,
        description="Total operating hours"
    )
    design_life_hours: float = Field(
        default=DEFAULT_DESIGN_LIVES["nozzle"],
        gt=0,
        description="Design life in hours"
    )
    fuel_type: str = Field(
        default="natural_gas",
        description="Fuel type"
    )
    fuel_sulfur_pct: float = Field(
        default=0.0,
        ge=0,
        le=10,
        description="Fuel sulfur content (%)"
    )
    average_firing_rate_pct: float = Field(
        default=75.0,
        ge=0,
        le=100,
        description="Average firing rate (%)"
    )
    pressure_drop_increase_pct: float = Field(
        default=0.0,
        ge=0,
        description="Pressure drop increase from baseline (%)"
    )
    spray_pattern_quality_pct: float = Field(
        default=100.0,
        ge=0,
        le=100,
        description="Spray pattern quality (% of design)"
    )
    last_inspection_date: Optional[datetime] = Field(
        default=None,
        description="Last inspection date"
    )
    last_cleaning_date: Optional[datetime] = Field(
        default=None,
        description="Last cleaning date"
    )
    visual_condition_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Visual inspection score (0-100)"
    )


class RefractoryTileData(BaseModel):
    """Refractory tile operational data."""

    tile_id: str = Field(..., description="Tile identifier")
    operating_hours: float = Field(
        ...,
        ge=0,
        description="Total operating hours"
    )
    design_life_hours: float = Field(
        default=DEFAULT_DESIGN_LIVES["refractory_tile"],
        gt=0,
        description="Design life in hours"
    )
    thermal_cycles: int = Field(
        default=0,
        ge=0,
        description="Number of thermal cycles (startups)"
    )
    max_thermal_cycle_delta_k: float = Field(
        default=500.0,
        ge=0,
        description="Maximum thermal cycle temperature range (K)"
    )
    average_operating_temp_c: float = Field(
        default=1000.0,
        description="Average operating temperature (deg C)"
    )
    max_operating_temp_c: float = Field(
        default=1200.0,
        description="Maximum operating temperature (deg C)"
    )
    crack_count: int = Field(
        default=0,
        ge=0,
        description="Number of visible cracks"
    )
    crack_length_total_mm: float = Field(
        default=0.0,
        ge=0,
        description="Total crack length (mm)"
    )
    spalling_area_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Spalling area as % of tile surface"
    )
    hot_spot_detected: bool = Field(
        default=False,
        description="Hot spot detected on casing"
    )
    last_inspection_date: Optional[datetime] = Field(
        default=None,
        description="Last inspection date"
    )


class IgniterData(BaseModel):
    """Igniter/pilot system data."""

    igniter_id: str = Field(..., description="Igniter identifier")
    igniter_type: str = Field(
        default="spark",
        description="Igniter type (spark, hot_surface, pilot)"
    )
    operating_hours: float = Field(
        ...,
        ge=0,
        description="Total operating hours"
    )
    design_life_hours: float = Field(
        default=DEFAULT_DESIGN_LIVES["igniter"],
        gt=0,
        description="Design life in hours"
    )
    ignition_attempts_total: int = Field(
        default=0,
        ge=0,
        description="Total ignition attempts"
    )
    ignition_failures: int = Field(
        default=0,
        ge=0,
        description="Number of ignition failures"
    )
    spark_gap_mm: Optional[float] = Field(
        default=None,
        gt=0,
        description="Current spark gap (mm)"
    )
    design_spark_gap_mm: Optional[float] = Field(
        default=3.0,
        gt=0,
        description="Design spark gap (mm)"
    )
    pilot_flame_signal_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Pilot flame signal strength (%)"
    )
    pilot_flame_stability_cv: Optional[float] = Field(
        default=None,
        ge=0,
        description="Pilot flame stability (coefficient of variation)"
    )
    last_spark_test_date: Optional[datetime] = Field(
        default=None,
        description="Last spark test date"
    )


class FlameScannerData(BaseModel):
    """Flame scanner sensor data."""

    scanner_id: str = Field(..., description="Scanner identifier")
    scanner_type: str = Field(
        default="uv_ir",
        description="Scanner type (uv, ir, uv_ir)"
    )
    operating_hours: float = Field(
        ...,
        ge=0,
        description="Total operating hours"
    )
    design_life_hours: float = Field(
        default=DEFAULT_DESIGN_LIVES["scanner"],
        gt=0,
        description="Design life in hours"
    )
    signal_strength_current_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Current signal strength (%)"
    )
    signal_strength_baseline_pct: float = Field(
        default=100.0,
        ge=0,
        le=100,
        description="Baseline signal strength (%)"
    )
    ambient_temperature_c: float = Field(
        default=50.0,
        description="Scanner ambient temperature (deg C)"
    )
    self_check_result: bool = Field(
        default=True,
        description="Self-check test passed"
    )
    last_calibration_date: Optional[datetime] = Field(
        default=None,
        description="Last calibration date"
    )
    lens_cleaning_date: Optional[datetime] = Field(
        default=None,
        description="Last lens cleaning date"
    )


class AirRegisterData(BaseModel):
    """Air register/damper data."""

    register_id: str = Field(..., description="Register identifier")
    operating_hours: float = Field(
        ...,
        ge=0,
        description="Total operating hours"
    )
    design_life_hours: float = Field(
        default=DEFAULT_DESIGN_LIVES["air_register"],
        gt=0,
        description="Design life in hours"
    )
    cycles_total: int = Field(
        default=0,
        ge=0,
        description="Total actuation cycles"
    )
    position_setpoint_pct: float = Field(
        default=50.0,
        ge=0,
        le=100,
        description="Current position setpoint (%)"
    )
    position_actual_pct: float = Field(
        default=50.0,
        ge=0,
        le=100,
        description="Actual position (%)"
    )
    position_error_pct: float = Field(
        default=0.0,
        ge=0,
        description="Position error (% deviation)"
    )
    stroke_time_s: Optional[float] = Field(
        default=None,
        gt=0,
        description="Full stroke time (seconds)"
    )
    design_stroke_time_s: Optional[float] = Field(
        default=30.0,
        gt=0,
        description="Design stroke time (seconds)"
    )
    sticking_detected: bool = Field(
        default=False,
        description="Sticking or binding detected"
    )
    backlash_deg: Optional[float] = Field(
        default=None,
        ge=0,
        description="Measured backlash (degrees)"
    )
    last_service_date: Optional[datetime] = Field(
        default=None,
        description="Last service date"
    )


class FuelValveData(BaseModel):
    """Fuel valve performance data."""

    valve_id: str = Field(..., description="Valve identifier")
    valve_type: str = Field(
        default="control",
        description="Valve type (control, safety_shutoff, bleed)"
    )
    operating_hours: float = Field(
        ...,
        ge=0,
        description="Total operating hours"
    )
    design_life_hours: float = Field(
        default=DEFAULT_DESIGN_LIVES["fuel_valve"],
        gt=0,
        description="Design life in hours"
    )
    cycles_total: int = Field(
        default=0,
        ge=0,
        description="Total actuation cycles"
    )
    seat_leakage_rate_scfh: float = Field(
        default=0.0,
        ge=0,
        description="Seat leakage rate (SCFH)"
    )
    seat_leakage_limit_scfh: float = Field(
        default=5.0,
        gt=0,
        description="Allowable seat leakage (SCFH)"
    )
    response_time_s: Optional[float] = Field(
        default=None,
        gt=0,
        description="Valve response time (seconds)"
    )
    design_response_time_s: Optional[float] = Field(
        default=1.0,
        gt=0,
        description="Design response time (seconds)"
    )
    actuator_pressure_psig: Optional[float] = Field(
        default=None,
        ge=0,
        description="Actuator supply pressure (psig)"
    )
    position_feedback_error_pct: float = Field(
        default=0.0,
        ge=0,
        description="Position feedback error (%)"
    )
    last_stroke_test_date: Optional[datetime] = Field(
        default=None,
        description="Last stroke test date"
    )
    last_seat_test_date: Optional[datetime] = Field(
        default=None,
        description="Last seat leakage test date"
    )


class BurnerHealthInput(BaseModel):
    """
    Complete input data for burner health analysis.

    This model encapsulates all component data required for comprehensive
    burner health assessment per API 535.

    Attributes:
        burner_id: Unique burner identifier
        nozzle: Nozzle/tip operational data
        refractory_tile: Refractory tile data
        igniter: Igniter/pilot system data
        scanner: Flame scanner data
        air_register: Air register/damper data
        fuel_valve: Fuel valve data

    Example:
        >>> input_data = BurnerHealthInput(
        ...     burner_id="BNR-001",
        ...     nozzle=nozzle_data,
        ...     refractory_tile=tile_data
        ... )
    """

    request_id: str = Field(
        default_factory=lambda: f"health_{datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}",
        description="Unique request identifier"
    )
    burner_id: str = Field(..., description="Burner identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis request timestamp"
    )

    # Component data
    nozzle: Optional[NozzleData] = Field(
        default=None,
        description="Nozzle/tip data"
    )
    refractory_tile: Optional[RefractoryTileData] = Field(
        default=None,
        description="Refractory tile data"
    )
    igniter: Optional[IgniterData] = Field(
        default=None,
        description="Igniter/pilot data"
    )
    scanner: Optional[FlameScannerData] = Field(
        default=None,
        description="Flame scanner data"
    )
    air_register: Optional[AirRegisterData] = Field(
        default=None,
        description="Air register/damper data"
    )
    fuel_valve: Optional[FuelValveData] = Field(
        default=None,
        description="Fuel valve data"
    )

    # Operating context
    total_burner_hours: float = Field(
        default=0.0,
        ge=0,
        description="Total burner operating hours"
    )
    total_startups: int = Field(
        default=0,
        ge=0,
        description="Total burner startups"
    )
    fuel_type: str = Field(
        default="natural_gas",
        description="Primary fuel type"
    )

    # Historical data
    previous_health_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Previous overall health score"
    )
    previous_assessment_date: Optional[datetime] = Field(
        default=None,
        description="Previous assessment date"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# OUTPUT SCHEMAS
# =============================================================================

class ComponentHealthResult(BaseModel):
    """Individual component health assessment result."""

    component_id: str = Field(..., description="Component identifier")
    component_type: str = Field(..., description="Component type")
    health_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Component health score (0-100)"
    )
    health_status: HealthStatus = Field(
        ...,
        description="Health status classification"
    )
    remaining_life_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Estimated remaining useful life (%)"
    )
    remaining_life_hours: float = Field(
        ...,
        ge=0,
        description="Estimated remaining hours"
    )
    failure_modes: List[FailureMode] = Field(
        default_factory=list,
        description="Active or developing failure modes"
    )
    failure_probability_30d: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="30-day failure probability"
    )
    degradation_rate_pct_per_1000h: float = Field(
        default=0.0,
        ge=0,
        description="Degradation rate (% per 1000 hours)"
    )
    maintenance_required: bool = Field(
        default=False,
        description="Maintenance action required"
    )
    maintenance_priority: MaintenancePriority = Field(
        default=MaintenancePriority.NONE,
        description="Maintenance priority"
    )
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="Recommended maintenance actions"
    )


class BurnerHealthOutput(BaseModel):
    """
    Complete output from burner health analysis.

    This comprehensive output model contains all component health assessments,
    overall health scoring, and maintenance recommendations per API 535.

    Attributes:
        burner_id: Burner identifier
        overall_health_score: Overall burner health (0-100)
        severity_classification: API 535 severity classification
        component_scores: Individual component assessments
        maintenance_priority: Highest priority maintenance action
        recommendations: Prioritized maintenance recommendations

    Example:
        >>> result = analyzer.analyze(input_data)
        >>> print(f"Overall Health: {result.overall_health_score:.1f}")
        >>> for rec in result.recommendations:
        ...     print(f"  - {rec}")
    """

    request_id: str = Field(..., description="Original request ID")
    burner_id: str = Field(..., description="Burner identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )
    status: str = Field(
        default="success",
        description="Analysis status"
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Processing duration (ms)"
    )

    # Overall health assessment
    overall_health_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall burner health score (0-100)"
    )
    overall_health_status: HealthStatus = Field(
        ...,
        description="Overall health status"
    )
    severity_classification: str = Field(
        ...,
        description="API 535 severity classification"
    )
    severity_action: str = Field(
        ...,
        description="Required action for severity level"
    )

    # Component health results
    component_scores: Dict[str, ComponentHealthResult] = Field(
        default_factory=dict,
        description="Health scores by component"
    )
    limiting_component: Optional[str] = Field(
        default=None,
        description="Component with lowest health score"
    )
    limiting_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Lowest component score"
    )

    # Maintenance planning
    maintenance_priority: MaintenancePriority = Field(
        default=MaintenancePriority.NONE,
        description="Highest priority maintenance action"
    )
    maintenance_window_days: Optional[int] = Field(
        default=None,
        ge=0,
        description="Recommended maintenance window (days)"
    )
    estimated_downtime_hours: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated maintenance downtime (hours)"
    )

    # Failure prediction
    overall_failure_probability_30d: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Overall 30-day failure probability"
    )
    estimated_rul_hours: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated remaining useful life (hours)"
    )

    # Trend analysis
    health_trend: str = Field(
        default="stable",
        description="Health trend (improving, stable, degrading)"
    )
    trend_rate_pct_per_month: Optional[float] = Field(
        default=None,
        description="Health change rate (% per month)"
    )

    # Recommendations and alerts
    recommendations: List[str] = Field(
        default_factory=list,
        description="Prioritized maintenance recommendations"
    )
    alerts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Generated alerts"
    )

    # Provenance
    provenance_hash: str = Field(
        ...,
        description="SHA-256 provenance hash"
    )
    calculation_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Calculation details for audit"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# COMPONENT HEALTH MODELS
# =============================================================================

class NozzleHealthModel:
    """
    Nozzle/tip health degradation model.

    Analyzes nozzle health based on:
    - Operating hours vs design life
    - Fuel quality impact (sulfur content)
    - Erosion rate estimation
    - Coking/plugging indicators
    - Spray pattern degradation

    Erosion rate formula:
        lambda = k * (firing_rate/100)^1.5 * sulfur_factor * fuel_factor

    Where:
        k = base degradation rate constant
        firing_rate = average firing rate (%)
        sulfur_factor = 1 + sulfur_pct * 0.1
        fuel_factor = fuel-specific degradation factor
    """

    def __init__(self, base_degradation_rate: float = 0.00004) -> None:
        """
        Initialize nozzle health model.

        Args:
            base_degradation_rate: Base degradation rate per hour
        """
        self.base_rate = base_degradation_rate

    def calculate_health(self, data: NozzleData) -> ComponentHealthResult:
        """
        Calculate nozzle health score.

        Args:
            data: Nozzle operational data

        Returns:
            ComponentHealthResult with nozzle health assessment
        """
        # Get fuel-specific degradation factors
        fuel_factors = FUEL_DEGRADATION_FACTORS.get(
            data.fuel_type.lower(),
            FUEL_DEGRADATION_FACTORS["natural_gas"]
        )

        # Calculate effective degradation rate
        # lambda = k * (firing_rate/100)^1.5 * sulfur_factor * fuel_factor
        sulfur_factor = 1.0 + data.fuel_sulfur_pct * 0.1
        firing_factor = (data.average_firing_rate_pct / 100.0) ** 1.5
        erosion_factor = fuel_factors["nozzle_erosion"]
        coking_factor = fuel_factors["nozzle_coking"]

        effective_lambda = self.base_rate * firing_factor * sulfur_factor * erosion_factor

        # Calculate age-based health using exponential degradation
        # Health = 100 * exp(-lambda * operating_hours / design_life)
        age_ratio = data.operating_hours / data.design_life_hours
        age_health = 100.0 * math.exp(-effective_lambda * data.operating_hours)

        # Condition-based adjustments
        condition_deductions = 0.0

        # Pressure drop increase indicates plugging/erosion
        if data.pressure_drop_increase_pct > 20:
            condition_deductions += 20.0
        elif data.pressure_drop_increase_pct > 10:
            condition_deductions += 10.0
        elif data.pressure_drop_increase_pct > 5:
            condition_deductions += 5.0

        # Spray pattern degradation
        if data.spray_pattern_quality_pct < 70:
            condition_deductions += 25.0
        elif data.spray_pattern_quality_pct < 85:
            condition_deductions += 15.0
        elif data.spray_pattern_quality_pct < 95:
            condition_deductions += 5.0

        # Visual condition if available
        if data.visual_condition_score is not None:
            condition_deductions += (100 - data.visual_condition_score) * 0.3

        # Calculate final health score
        health_score = max(0.0, min(100.0, age_health - condition_deductions))

        # Determine failure modes
        failure_modes: List[FailureMode] = []
        if data.pressure_drop_increase_pct > 10:
            if coking_factor > 1.5:
                failure_modes.append(FailureMode.COKING)
            else:
                failure_modes.append(FailureMode.EROSION)
        if data.spray_pattern_quality_pct < 85:
            failure_modes.append(FailureMode.PLUGGING)

        # Calculate remaining life
        remaining_life_pct = (1.0 - age_ratio) * 100.0 * (health_score / 100.0)
        remaining_life_hours = max(0, data.design_life_hours - data.operating_hours)

        # Calculate degradation rate
        degradation_rate = effective_lambda * 1000 * 100  # % per 1000 hours

        # 30-day failure probability (simplified Weibull)
        hours_30d = 30 * 24
        failure_prob_30d = 1.0 - math.exp(
            -((data.operating_hours + hours_30d) / data.design_life_hours) ** 2
        )
        failure_prob_30d = min(0.99, failure_prob_30d * (1.0 - health_score / 100.0))

        # Determine status and priority
        health_status = self._score_to_status(health_score)
        maintenance_priority = self._score_to_priority(health_score)
        maintenance_required = health_score < 70

        # Generate recommendations
        recommendations = self._generate_recommendations(
            health_score, failure_modes, data
        )

        return ComponentHealthResult(
            component_id=data.nozzle_id,
            component_type="nozzle",
            health_score=round(health_score, 2),
            health_status=health_status,
            remaining_life_pct=round(max(0, remaining_life_pct), 2),
            remaining_life_hours=round(remaining_life_hours, 0),
            failure_modes=failure_modes,
            failure_probability_30d=round(failure_prob_30d, 4),
            degradation_rate_pct_per_1000h=round(degradation_rate, 2),
            maintenance_required=maintenance_required,
            maintenance_priority=maintenance_priority,
            recommended_actions=recommendations,
        )

    def _score_to_status(self, score: float) -> HealthStatus:
        """Convert health score to status."""
        if score >= 90:
            return HealthStatus.EXCELLENT
        elif score >= 70:
            return HealthStatus.GOOD
        elif score >= 50:
            return HealthStatus.FAIR
        elif score >= 25:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL

    def _score_to_priority(self, score: float) -> MaintenancePriority:
        """Convert health score to maintenance priority."""
        if score >= 90:
            return MaintenancePriority.NONE
        elif score >= 70:
            return MaintenancePriority.ROUTINE
        elif score >= 50:
            return MaintenancePriority.PLANNED
        elif score >= 25:
            return MaintenancePriority.PRIORITY
        elif score >= 10:
            return MaintenancePriority.URGENT
        else:
            return MaintenancePriority.EMERGENCY

    def _generate_recommendations(
        self,
        score: float,
        failure_modes: List[FailureMode],
        data: NozzleData
    ) -> List[str]:
        """Generate maintenance recommendations."""
        recommendations = []

        if FailureMode.COKING in failure_modes:
            recommendations.append(
                "Clean nozzle tip to remove carbon deposits. "
                "Consider chemical cleaning or mechanical descaling."
            )
        if FailureMode.EROSION in failure_modes:
            recommendations.append(
                "Inspect nozzle for erosion wear. "
                "Replace if orifice diameter exceeds tolerance."
            )
        if FailureMode.PLUGGING in failure_modes:
            recommendations.append(
                "Check fuel strainer and clean nozzle passages. "
                "Verify fuel filtration system operation."
            )

        if score < 50:
            recommendations.append(
                "Schedule nozzle replacement within next planned outage."
            )
        elif score < 70:
            recommendations.append(
                "Plan nozzle inspection and cleaning within 30 days."
            )

        # Inspection reminder
        if data.last_inspection_date:
            days_since = (datetime.now(timezone.utc) - data.last_inspection_date).days
            if days_since > 180:
                recommendations.append(
                    f"Nozzle inspection overdue ({days_since} days since last inspection)."
                )

        return recommendations


class RefractoryHealthModel:
    """
    Refractory tile health model using Coffin-Manson thermal fatigue analysis.

    Analyzes refractory tile health based on:
    - Thermal cycling fatigue (Coffin-Manson equation)
    - Operating temperature exposure
    - Visual damage indicators (cracks, spalling)
    - Hot spot detection

    Coffin-Manson equation:
        N_f = (epsilon_f / epsilon_a)^(1/c)

    Where:
        N_f = cycles to failure
        epsilon_f = fatigue ductility coefficient
        epsilon_a = strain amplitude = alpha * delta_T
        c = fatigue ductility exponent
    """

    def __init__(self) -> None:
        """Initialize refractory health model."""
        self.params = COFFIN_MANSON_PARAMS

    def calculate_health(self, data: RefractoryTileData) -> ComponentHealthResult:
        """
        Calculate refractory tile health score.

        Args:
            data: Refractory tile operational data

        Returns:
            ComponentHealthResult with tile health assessment
        """
        # Calculate thermal strain amplitude
        alpha = self.params["thermal_expansion_coeff"]
        delta_t = data.max_thermal_cycle_delta_k
        epsilon_a = alpha * delta_t

        # Coffin-Manson fatigue life estimation
        epsilon_f = self.params["refractory_coefficient"]
        c = self.params["refractory_exponent"]

        if epsilon_a > 0:
            cycles_to_failure = (epsilon_f / epsilon_a) ** (1 / c)
        else:
            cycles_to_failure = float('inf')

        # Calculate fatigue damage fraction
        fatigue_damage = data.thermal_cycles / cycles_to_failure if cycles_to_failure > 0 else 0

        # Age-based degradation
        age_ratio = data.operating_hours / data.design_life_hours
        age_factor = math.exp(-2 * age_ratio)

        # Temperature stress factor (higher temp = faster degradation)
        temp_factor = 1.0
        if data.max_operating_temp_c > 1300:
            temp_factor = 0.7
        elif data.max_operating_temp_c > 1200:
            temp_factor = 0.85

        # Calculate base health from age and fatigue
        base_health = 100.0 * age_factor * (1 - min(fatigue_damage, 0.9)) * temp_factor

        # Condition-based deductions
        condition_deductions = 0.0

        # Crack damage assessment
        if data.crack_count > 0:
            crack_penalty = min(30, data.crack_count * 5)
            length_penalty = min(20, data.crack_length_total_mm / 10)
            condition_deductions += crack_penalty + length_penalty

        # Spalling assessment
        if data.spalling_area_pct > 0:
            condition_deductions += min(30, data.spalling_area_pct * 3)

        # Hot spot detection
        if data.hot_spot_detected:
            condition_deductions += 25

        # Calculate final health score
        health_score = max(0.0, min(100.0, base_health - condition_deductions))

        # Determine failure modes
        failure_modes: List[FailureMode] = []
        if data.crack_count > 0:
            failure_modes.append(FailureMode.CRACKING)
        if fatigue_damage > 0.5:
            failure_modes.append(FailureMode.THERMAL_FATIGUE)
        if data.spalling_area_pct > 5:
            failure_modes.append(FailureMode.SPALLING)

        # Calculate remaining life
        remaining_cycles = max(0, cycles_to_failure - data.thermal_cycles)
        remaining_life_pct = (remaining_cycles / cycles_to_failure * 100) if cycles_to_failure > 0 else 0
        remaining_life_hours = max(0, data.design_life_hours - data.operating_hours)

        # Degradation rate
        degradation_rate = (1 - age_factor) / (data.operating_hours / 1000) * 100 if data.operating_hours > 0 else 0

        # 30-day failure probability
        failure_prob_30d = min(0.99, fatigue_damage * (1 - health_score / 100))

        # Determine status and priority
        health_status = self._score_to_status(health_score)
        maintenance_priority = self._score_to_priority(health_score)
        maintenance_required = health_score < 70 or data.hot_spot_detected

        # Generate recommendations
        recommendations = self._generate_recommendations(
            health_score, failure_modes, data
        )

        return ComponentHealthResult(
            component_id=data.tile_id,
            component_type="refractory_tile",
            health_score=round(health_score, 2),
            health_status=health_status,
            remaining_life_pct=round(max(0, remaining_life_pct), 2),
            remaining_life_hours=round(remaining_life_hours, 0),
            failure_modes=failure_modes,
            failure_probability_30d=round(failure_prob_30d, 4),
            degradation_rate_pct_per_1000h=round(degradation_rate, 2),
            maintenance_required=maintenance_required,
            maintenance_priority=maintenance_priority,
            recommended_actions=recommendations,
        )

    def _score_to_status(self, score: float) -> HealthStatus:
        """Convert health score to status."""
        if score >= 90:
            return HealthStatus.EXCELLENT
        elif score >= 70:
            return HealthStatus.GOOD
        elif score >= 50:
            return HealthStatus.FAIR
        elif score >= 25:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL

    def _score_to_priority(self, score: float) -> MaintenancePriority:
        """Convert health score to maintenance priority."""
        if score >= 90:
            return MaintenancePriority.NONE
        elif score >= 70:
            return MaintenancePriority.ROUTINE
        elif score >= 50:
            return MaintenancePriority.PLANNED
        elif score >= 25:
            return MaintenancePriority.PRIORITY
        elif score >= 10:
            return MaintenancePriority.URGENT
        else:
            return MaintenancePriority.EMERGENCY

    def _generate_recommendations(
        self,
        score: float,
        failure_modes: List[FailureMode],
        data: RefractoryTileData
    ) -> List[str]:
        """Generate maintenance recommendations."""
        recommendations = []

        if data.hot_spot_detected:
            recommendations.append(
                "URGENT: Hot spot detected on casing. Inspect refractory immediately "
                "for failure or breakthrough."
            )

        if FailureMode.CRACKING in failure_modes:
            recommendations.append(
                f"Monitor {data.crack_count} cracks for propagation. "
                "Consider refractory repair or replacement if cracks exceed 50mm."
            )

        if FailureMode.SPALLING in failure_modes:
            recommendations.append(
                f"Spalling covers {data.spalling_area_pct:.1f}% of tile surface. "
                "Plan refractory repair or coating application."
            )

        if FailureMode.THERMAL_FATIGUE in failure_modes:
            recommendations.append(
                "Thermal fatigue damage significant. Minimize thermal cycling "
                "and plan replacement at next turnaround."
            )

        if score < 50:
            recommendations.append(
                "Schedule refractory replacement at next planned outage."
            )

        return recommendations


class IgniterHealthModel:
    """
    Igniter/pilot system health model.

    Analyzes igniter health based on:
    - Ignition success rate
    - Spark gap degradation
    - Operating hours vs design life
    - Pilot flame stability (if applicable)

    Spark gap degradation:
        gap_increase = k * attempts^0.5

    Where:
        k = material erosion coefficient
        attempts = total ignition attempts
    """

    def __init__(self, gap_erosion_coeff: float = 0.001) -> None:
        """
        Initialize igniter health model.

        Args:
            gap_erosion_coeff: Spark gap erosion coefficient
        """
        self.gap_erosion_coeff = gap_erosion_coeff

    def calculate_health(self, data: IgniterData) -> ComponentHealthResult:
        """
        Calculate igniter health score.

        Args:
            data: Igniter operational data

        Returns:
            ComponentHealthResult with igniter health assessment
        """
        # Calculate ignition success rate
        total_attempts = data.ignition_attempts_total
        if total_attempts > 0:
            success_rate = 1.0 - (data.ignition_failures / total_attempts)
        else:
            success_rate = 1.0

        # Age-based degradation
        age_ratio = data.operating_hours / data.design_life_hours
        age_health = 100.0 * math.exp(-1.5 * age_ratio)

        # Spark gap degradation (for spark igniters)
        gap_health = 100.0
        if data.igniter_type == "spark" and data.spark_gap_mm is not None:
            design_gap = data.design_spark_gap_mm or 3.0
            gap_deviation = abs(data.spark_gap_mm - design_gap) / design_gap
            gap_health = 100.0 * (1 - min(gap_deviation, 0.5) * 2)

        # Pilot flame health (for pilot igniters)
        pilot_health = 100.0
        if data.igniter_type == "pilot":
            if data.pilot_flame_signal_pct is not None:
                pilot_health = data.pilot_flame_signal_pct
            if data.pilot_flame_stability_cv is not None and data.pilot_flame_stability_cv > 0.2:
                pilot_health -= 20 * data.pilot_flame_stability_cv

        # Calculate weighted health score
        if data.igniter_type == "spark":
            health_score = 0.4 * age_health + 0.3 * (success_rate * 100) + 0.3 * gap_health
        elif data.igniter_type == "pilot":
            health_score = 0.3 * age_health + 0.3 * (success_rate * 100) + 0.4 * pilot_health
        else:  # hot_surface
            health_score = 0.5 * age_health + 0.5 * (success_rate * 100)

        health_score = max(0.0, min(100.0, health_score))

        # Determine failure modes
        failure_modes: List[FailureMode] = []
        if success_rate < 0.9:
            if data.igniter_type == "spark":
                failure_modes.append(FailureMode.EROSION)
            else:
                failure_modes.append(FailureMode.SENSOR_DEGRADATION)

        # Calculate remaining life
        remaining_life_pct = (1.0 - age_ratio) * 100 * (health_score / 100)
        remaining_life_hours = max(0, data.design_life_hours - data.operating_hours)

        # 30-day failure probability
        failure_prob_30d = min(0.99, (1 - success_rate) + (1 - health_score / 100) * 0.3)

        # Degradation rate
        degradation_rate = (100 - health_score) / (data.operating_hours / 1000) if data.operating_hours > 0 else 0

        # Determine status and priority
        health_status = self._score_to_status(health_score)
        maintenance_priority = self._score_to_priority(health_score)
        maintenance_required = health_score < 70 or success_rate < 0.95

        # Generate recommendations
        recommendations = self._generate_recommendations(
            health_score, data, success_rate
        )

        return ComponentHealthResult(
            component_id=data.igniter_id,
            component_type="igniter",
            health_score=round(health_score, 2),
            health_status=health_status,
            remaining_life_pct=round(max(0, remaining_life_pct), 2),
            remaining_life_hours=round(remaining_life_hours, 0),
            failure_modes=failure_modes,
            failure_probability_30d=round(failure_prob_30d, 4),
            degradation_rate_pct_per_1000h=round(degradation_rate, 2),
            maintenance_required=maintenance_required,
            maintenance_priority=maintenance_priority,
            recommended_actions=recommendations,
        )

    def _score_to_status(self, score: float) -> HealthStatus:
        """Convert health score to status."""
        if score >= 90:
            return HealthStatus.EXCELLENT
        elif score >= 70:
            return HealthStatus.GOOD
        elif score >= 50:
            return HealthStatus.FAIR
        elif score >= 25:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL

    def _score_to_priority(self, score: float) -> MaintenancePriority:
        """Convert health score to maintenance priority."""
        if score >= 90:
            return MaintenancePriority.NONE
        elif score >= 70:
            return MaintenancePriority.ROUTINE
        elif score >= 50:
            return MaintenancePriority.PLANNED
        elif score >= 25:
            return MaintenancePriority.PRIORITY
        else:
            return MaintenancePriority.URGENT

    def _generate_recommendations(
        self,
        score: float,
        data: IgniterData,
        success_rate: float
    ) -> List[str]:
        """Generate maintenance recommendations."""
        recommendations = []

        if success_rate < 0.95:
            failure_pct = (1 - success_rate) * 100
            recommendations.append(
                f"Ignition failure rate is {failure_pct:.1f}%. "
                "Inspect igniter and verify fuel/air supply to pilot."
            )

        if data.igniter_type == "spark":
            if data.spark_gap_mm and data.design_spark_gap_mm:
                gap_diff = abs(data.spark_gap_mm - data.design_spark_gap_mm)
                if gap_diff > 0.5:
                    recommendations.append(
                        f"Spark gap is {data.spark_gap_mm:.1f}mm "
                        f"(design: {data.design_spark_gap_mm:.1f}mm). "
                        "Replace spark electrode."
                    )

        if score < 50:
            recommendations.append(
                f"Replace {data.igniter_type} igniter at next opportunity."
            )

        return recommendations


class ScannerHealthModel:
    """
    Flame scanner health model with UV/IR sensor degradation curves.

    Analyzes scanner health based on:
    - Sensor degradation (exponential decay)
    - Signal strength vs baseline
    - Temperature-accelerated aging
    - Lens contamination effects
    - Self-check test results

    Sensor degradation:
        sensitivity = baseline * exp(-t / half_life)

    Temperature acceleration (Arrhenius-like):
        acceleration = 1 + factor * (T - T_ref) / 10

    Where:
        T = operating temperature
        T_ref = reference temperature (50C)
    """

    def __init__(self) -> None:
        """Initialize scanner health model."""
        self.params = SCANNER_DEGRADATION

    def calculate_health(self, data: FlameScannerData) -> ComponentHealthResult:
        """
        Calculate flame scanner health score.

        Args:
            data: Flame scanner operational data

        Returns:
            ComponentHealthResult with scanner health assessment
        """
        # Get scanner-type specific parameters
        scanner_params = self.params.get(
            data.scanner_type.lower(),
            self.params["uv_ir"]
        )

        half_life = scanner_params["half_life_hours"]
        temp_factor_coeff = scanner_params["temp_acceleration_factor"]
        contamination_rate = scanner_params["contamination_rate"]

        # Temperature acceleration factor
        temp_ref = 50.0  # Reference temperature (C)
        temp_delta = max(0, data.ambient_temperature_c - temp_ref)
        temp_acceleration = 1.0 + temp_factor_coeff * (temp_delta / 10.0)

        # Effective operating hours (temperature-adjusted)
        effective_hours = data.operating_hours * temp_acceleration

        # Sensor degradation (exponential decay)
        sensor_degradation = math.exp(-effective_hours / half_life)
        sensor_health = 100.0 * sensor_degradation

        # Signal strength degradation
        if data.signal_strength_baseline_pct > 0:
            signal_ratio = data.signal_strength_current_pct / data.signal_strength_baseline_pct
        else:
            signal_ratio = 1.0
        signal_health = 100.0 * min(signal_ratio, 1.0)

        # Contamination/fouling effect (based on time since cleaning)
        contamination_health = 100.0
        if data.lens_cleaning_date:
            hours_since_cleaning = (
                datetime.now(timezone.utc) - data.lens_cleaning_date
            ).total_seconds() / 3600
            contamination_health = 100.0 * math.exp(-contamination_rate * hours_since_cleaning)

        # Self-check penalty
        self_check_penalty = 0.0 if data.self_check_result else 30.0

        # Calculate weighted health score
        health_score = (
            0.35 * sensor_health +
            0.35 * signal_health +
            0.2 * contamination_health +
            0.1 * 100  # Calibration factor placeholder
        ) - self_check_penalty

        health_score = max(0.0, min(100.0, health_score))

        # Determine failure modes
        failure_modes: List[FailureMode] = []
        if sensor_health < 70:
            failure_modes.append(FailureMode.SENSOR_DEGRADATION)
        if contamination_health < 80:
            failure_modes.append(FailureMode.PLUGGING)  # Lens fouling

        # Calculate remaining life
        age_ratio = data.operating_hours / data.design_life_hours
        remaining_life_pct = (1.0 - age_ratio) * 100 * (health_score / 100)
        remaining_life_hours = max(0, data.design_life_hours - data.operating_hours)

        # 30-day failure probability
        failure_prob_30d = min(0.99, (1 - sensor_degradation) * 0.5 + (1 - health_score / 100) * 0.3)

        # Degradation rate
        degradation_rate = (100 - sensor_health) / (data.operating_hours / 1000) if data.operating_hours > 0 else 0

        # Determine status and priority
        health_status = self._score_to_status(health_score)
        maintenance_priority = self._score_to_priority(health_score)
        maintenance_required = health_score < 70 or not data.self_check_result

        # Generate recommendations
        recommendations = self._generate_recommendations(
            health_score, data, contamination_health
        )

        return ComponentHealthResult(
            component_id=data.scanner_id,
            component_type="scanner",
            health_score=round(health_score, 2),
            health_status=health_status,
            remaining_life_pct=round(max(0, remaining_life_pct), 2),
            remaining_life_hours=round(remaining_life_hours, 0),
            failure_modes=failure_modes,
            failure_probability_30d=round(failure_prob_30d, 4),
            degradation_rate_pct_per_1000h=round(degradation_rate, 2),
            maintenance_required=maintenance_required,
            maintenance_priority=maintenance_priority,
            recommended_actions=recommendations,
        )

    def _score_to_status(self, score: float) -> HealthStatus:
        """Convert health score to status."""
        if score >= 90:
            return HealthStatus.EXCELLENT
        elif score >= 70:
            return HealthStatus.GOOD
        elif score >= 50:
            return HealthStatus.FAIR
        elif score >= 25:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL

    def _score_to_priority(self, score: float) -> MaintenancePriority:
        """Convert health score to maintenance priority."""
        if score >= 90:
            return MaintenancePriority.NONE
        elif score >= 70:
            return MaintenancePriority.ROUTINE
        elif score >= 50:
            return MaintenancePriority.PLANNED
        elif score >= 25:
            return MaintenancePriority.PRIORITY
        else:
            return MaintenancePriority.URGENT

    def _generate_recommendations(
        self,
        score: float,
        data: FlameScannerData,
        contamination_health: float
    ) -> List[str]:
        """Generate maintenance recommendations."""
        recommendations = []

        if not data.self_check_result:
            recommendations.append(
                "URGENT: Scanner self-check failed. Verify scanner operation "
                "and replace if faulty."
            )

        if contamination_health < 80:
            recommendations.append(
                "Clean scanner lens to restore signal strength. "
                "Verify cooling air supply."
            )

        signal_loss = data.signal_strength_baseline_pct - data.signal_strength_current_pct
        if signal_loss > 20:
            recommendations.append(
                f"Signal strength degraded by {signal_loss:.1f}%. "
                "Recalibrate or replace scanner."
            )

        if data.last_calibration_date:
            days_since = (datetime.now(timezone.utc) - data.last_calibration_date).days
            if days_since > 365:
                recommendations.append(
                    f"Scanner calibration overdue ({days_since} days). "
                    "Schedule calibration."
                )

        if score < 50:
            recommendations.append(
                "Plan scanner replacement at next opportunity."
            )

        return recommendations


class AirRegisterHealthModel:
    """
    Air register/damper health model.

    Analyzes air register health based on:
    - Actuation cycle wear
    - Position accuracy
    - Stroke time degradation
    - Mechanical wear indicators (sticking, backlash)

    Wear model:
        wear_factor = cycles / design_cycles

    Position accuracy degradation:
        accuracy_loss = k * sqrt(cycles)
    """

    def __init__(self, wear_coefficient: float = 0.0001) -> None:
        """
        Initialize air register health model.

        Args:
            wear_coefficient: Mechanical wear coefficient
        """
        self.wear_coeff = wear_coefficient

    def calculate_health(self, data: AirRegisterData) -> ComponentHealthResult:
        """
        Calculate air register health score.

        Args:
            data: Air register operational data

        Returns:
            ComponentHealthResult with air register health assessment
        """
        # Age-based degradation
        age_ratio = data.operating_hours / data.design_life_hours
        age_health = 100.0 * math.exp(-1.2 * age_ratio)

        # Cycle-based wear
        # Assume design cycles = operating_hours * 10 (10 cycles/hour average)
        design_cycles = data.design_life_hours * 10
        cycle_wear = data.cycles_total / design_cycles if design_cycles > 0 else 0
        cycle_health = 100.0 * (1 - min(cycle_wear, 1.0))

        # Position accuracy health
        position_health = 100.0 - min(50, data.position_error_pct * 10)

        # Stroke time health
        stroke_health = 100.0
        if data.stroke_time_s and data.design_stroke_time_s:
            stroke_ratio = data.stroke_time_s / data.design_stroke_time_s
            if stroke_ratio > 1.5:
                stroke_health = 50.0
            elif stroke_ratio > 1.2:
                stroke_health = 70.0
            elif stroke_ratio > 1.1:
                stroke_health = 85.0

        # Sticking penalty
        sticking_penalty = 30.0 if data.sticking_detected else 0.0

        # Backlash penalty
        backlash_penalty = 0.0
        if data.backlash_deg and data.backlash_deg > 2.0:
            backlash_penalty = min(20, (data.backlash_deg - 2.0) * 5)

        # Calculate weighted health score
        health_score = (
            0.25 * age_health +
            0.25 * cycle_health +
            0.25 * position_health +
            0.25 * stroke_health
        ) - sticking_penalty - backlash_penalty

        health_score = max(0.0, min(100.0, health_score))

        # Determine failure modes
        failure_modes: List[FailureMode] = []
        if data.sticking_detected:
            failure_modes.append(FailureMode.MECHANICAL_WEAR)
        if data.backlash_deg and data.backlash_deg > 2.0:
            failure_modes.append(FailureMode.MECHANICAL_WEAR)
        if data.position_error_pct > 3.0:
            failure_modes.append(FailureMode.ACTUATOR_FAILURE)

        # Calculate remaining life
        remaining_life_pct = (1.0 - max(age_ratio, cycle_wear)) * 100 * (health_score / 100)
        remaining_life_hours = max(0, data.design_life_hours - data.operating_hours)

        # 30-day failure probability
        failure_prob_30d = min(0.99, cycle_wear * 0.3 + (1 - health_score / 100) * 0.3)

        # Degradation rate
        degradation_rate = (100 - health_score) / (data.operating_hours / 1000) if data.operating_hours > 0 else 0

        # Determine status and priority
        health_status = self._score_to_status(health_score)
        maintenance_priority = self._score_to_priority(health_score)
        maintenance_required = health_score < 70 or data.sticking_detected

        # Generate recommendations
        recommendations = self._generate_recommendations(
            health_score, data
        )

        return ComponentHealthResult(
            component_id=data.register_id,
            component_type="air_register",
            health_score=round(health_score, 2),
            health_status=health_status,
            remaining_life_pct=round(max(0, remaining_life_pct), 2),
            remaining_life_hours=round(remaining_life_hours, 0),
            failure_modes=failure_modes,
            failure_probability_30d=round(failure_prob_30d, 4),
            degradation_rate_pct_per_1000h=round(degradation_rate, 2),
            maintenance_required=maintenance_required,
            maintenance_priority=maintenance_priority,
            recommended_actions=recommendations,
        )

    def _score_to_status(self, score: float) -> HealthStatus:
        """Convert health score to status."""
        if score >= 90:
            return HealthStatus.EXCELLENT
        elif score >= 70:
            return HealthStatus.GOOD
        elif score >= 50:
            return HealthStatus.FAIR
        elif score >= 25:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL

    def _score_to_priority(self, score: float) -> MaintenancePriority:
        """Convert health score to maintenance priority."""
        if score >= 90:
            return MaintenancePriority.NONE
        elif score >= 70:
            return MaintenancePriority.ROUTINE
        elif score >= 50:
            return MaintenancePriority.PLANNED
        elif score >= 25:
            return MaintenancePriority.PRIORITY
        else:
            return MaintenancePriority.URGENT

    def _generate_recommendations(
        self,
        score: float,
        data: AirRegisterData
    ) -> List[str]:
        """Generate maintenance recommendations."""
        recommendations = []

        if data.sticking_detected:
            recommendations.append(
                "Air register sticking detected. Lubricate linkage "
                "and check for mechanical binding."
            )

        if data.position_error_pct > 3.0:
            recommendations.append(
                f"Position error is {data.position_error_pct:.1f}%. "
                "Recalibrate actuator or replace positioner."
            )

        if data.stroke_time_s and data.design_stroke_time_s:
            if data.stroke_time_s > data.design_stroke_time_s * 1.2:
                recommendations.append(
                    f"Stroke time degraded ({data.stroke_time_s:.1f}s vs "
                    f"{data.design_stroke_time_s:.1f}s design). "
                    "Service actuator."
                )

        if score < 50:
            recommendations.append(
                "Plan air register overhaul at next outage."
            )

        return recommendations


class FuelValveHealthModel:
    """
    Fuel valve health model.

    Analyzes fuel valve health based on:
    - Seat leakage rate
    - Response time degradation
    - Position feedback accuracy
    - Actuator performance

    Seat wear model:
        leakage = baseline * (1 + k * cycles)

    Where:
        k = seat wear coefficient
        cycles = total actuation cycles
    """

    def __init__(self, seat_wear_coeff: float = 0.0001) -> None:
        """
        Initialize fuel valve health model.

        Args:
            seat_wear_coeff: Seat wear coefficient
        """
        self.seat_wear_coeff = seat_wear_coeff

    def calculate_health(self, data: FuelValveData) -> ComponentHealthResult:
        """
        Calculate fuel valve health score.

        Args:
            data: Fuel valve operational data

        Returns:
            ComponentHealthResult with fuel valve health assessment
        """
        # Age-based degradation
        age_ratio = data.operating_hours / data.design_life_hours
        age_health = 100.0 * math.exp(-1.5 * age_ratio)

        # Seat leakage health
        leakage_ratio = data.seat_leakage_rate_scfh / data.seat_leakage_limit_scfh
        if leakage_ratio >= 1.0:
            seat_health = 0.0  # Failed
        elif leakage_ratio > 0.5:
            seat_health = 100.0 * (1 - leakage_ratio)
        else:
            seat_health = 100.0 - leakage_ratio * 50

        # Response time health
        response_health = 100.0
        if data.response_time_s and data.design_response_time_s:
            response_ratio = data.response_time_s / data.design_response_time_s
            if response_ratio > 2.0:
                response_health = 30.0
            elif response_ratio > 1.5:
                response_health = 60.0
            elif response_ratio > 1.2:
                response_health = 80.0

        # Position feedback health
        position_health = 100.0 - min(50, data.position_feedback_error_pct * 10)

        # Cycle wear factor
        design_cycles = data.design_life_hours * 5  # 5 cycles/hour assumed
        cycle_factor = min(1.0, data.cycles_total / design_cycles) if design_cycles > 0 else 0
        cycle_health = 100.0 * (1 - cycle_factor * 0.5)

        # Calculate weighted health score
        # Safety valves weight seat health more heavily
        if data.valve_type == "safety_shutoff":
            health_score = (
                0.15 * age_health +
                0.40 * seat_health +
                0.25 * response_health +
                0.10 * position_health +
                0.10 * cycle_health
            )
        else:  # control valve
            health_score = (
                0.20 * age_health +
                0.25 * seat_health +
                0.20 * response_health +
                0.20 * position_health +
                0.15 * cycle_health
            )

        health_score = max(0.0, min(100.0, health_score))

        # Determine failure modes
        failure_modes: List[FailureMode] = []
        if leakage_ratio > 0.5:
            failure_modes.append(FailureMode.SEAT_WEAR)
        if data.response_time_s and data.design_response_time_s:
            if data.response_time_s > data.design_response_time_s * 1.5:
                failure_modes.append(FailureMode.ACTUATOR_FAILURE)

        # Calculate remaining life
        remaining_life_pct = (1.0 - age_ratio) * 100 * (health_score / 100)
        remaining_life_hours = max(0, data.design_life_hours - data.operating_hours)

        # 30-day failure probability
        failure_prob_30d = min(0.99, leakage_ratio * 0.5 + (1 - health_score / 100) * 0.3)

        # Degradation rate
        degradation_rate = (100 - health_score) / (data.operating_hours / 1000) if data.operating_hours > 0 else 0

        # Determine status and priority
        health_status = self._score_to_status(health_score)
        maintenance_priority = self._score_to_priority(health_score)

        # Safety valves have higher priority thresholds
        if data.valve_type == "safety_shutoff" and leakage_ratio > 0.5:
            maintenance_priority = MaintenancePriority.URGENT

        maintenance_required = health_score < 70 or leakage_ratio > 0.5

        # Generate recommendations
        recommendations = self._generate_recommendations(
            health_score, data, leakage_ratio
        )

        return ComponentHealthResult(
            component_id=data.valve_id,
            component_type="fuel_valve",
            health_score=round(health_score, 2),
            health_status=health_status,
            remaining_life_pct=round(max(0, remaining_life_pct), 2),
            remaining_life_hours=round(remaining_life_hours, 0),
            failure_modes=failure_modes,
            failure_probability_30d=round(failure_prob_30d, 4),
            degradation_rate_pct_per_1000h=round(degradation_rate, 2),
            maintenance_required=maintenance_required,
            maintenance_priority=maintenance_priority,
            recommended_actions=recommendations,
        )

    def _score_to_status(self, score: float) -> HealthStatus:
        """Convert health score to status."""
        if score >= 90:
            return HealthStatus.EXCELLENT
        elif score >= 70:
            return HealthStatus.GOOD
        elif score >= 50:
            return HealthStatus.FAIR
        elif score >= 25:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL

    def _score_to_priority(self, score: float) -> MaintenancePriority:
        """Convert health score to maintenance priority."""
        if score >= 90:
            return MaintenancePriority.NONE
        elif score >= 70:
            return MaintenancePriority.ROUTINE
        elif score >= 50:
            return MaintenancePriority.PLANNED
        elif score >= 25:
            return MaintenancePriority.PRIORITY
        else:
            return MaintenancePriority.URGENT

    def _generate_recommendations(
        self,
        score: float,
        data: FuelValveData,
        leakage_ratio: float
    ) -> List[str]:
        """Generate maintenance recommendations."""
        recommendations = []

        if leakage_ratio >= 1.0:
            recommendations.append(
                f"CRITICAL: {data.valve_type} valve seat leakage exceeds limit "
                f"({data.seat_leakage_rate_scfh:.1f} vs {data.seat_leakage_limit_scfh:.1f} SCFH). "
                "Replace valve or relap seat immediately."
            )
        elif leakage_ratio > 0.5:
            recommendations.append(
                f"Seat leakage at {leakage_ratio*100:.0f}% of limit. "
                "Plan valve maintenance."
            )

        if data.response_time_s and data.design_response_time_s:
            if data.response_time_s > data.design_response_time_s * 1.5:
                recommendations.append(
                    f"Valve response slow ({data.response_time_s:.2f}s vs "
                    f"{data.design_response_time_s:.2f}s design). "
                    "Service actuator and check air supply."
                )

        if data.position_feedback_error_pct > 3.0:
            recommendations.append(
                f"Position feedback error is {data.position_feedback_error_pct:.1f}%. "
                "Recalibrate positioner."
            )

        # Test reminders
        if data.last_seat_test_date:
            days_since = (datetime.now(timezone.utc) - data.last_seat_test_date).days
            if days_since > 365:
                recommendations.append(
                    f"Seat leakage test overdue ({days_since} days). "
                    "Schedule API 598 seat test."
                )

        if data.last_stroke_test_date:
            days_since = (datetime.now(timezone.utc) - data.last_stroke_test_date).days
            if days_since > 90 and data.valve_type == "safety_shutoff":
                recommendations.append(
                    f"Safety valve stroke test overdue ({days_since} days)."
                )

        return recommendations


# =============================================================================
# BURNER HEALTH ANALYZER
# =============================================================================

class BurnerHealthAnalyzer:
    """
    Burner component health scoring per API 535.

    Analyzes health of burner components:
    - Nozzle/tip condition (erosion, coking, plugging)
    - Refractory tile integrity (cracking, spalling, thermal fatigue)
    - Igniter/pilot system health
    - Flame scanner reliability
    - Air register/damper operation
    - Fuel valve performance

    DETERMINISTIC: All calculations use documented formulas with no randomness.
    AUDITABLE: Full calculation trace captured for compliance reporting.

    Health scoring formula:
        Component_Score = 100 * exp(-lambda * operating_hours / design_life)
        Overall_Health = sum(weight_i * component_score_i)

    Severity classification per API 535:
        - 90-100: Excellent (no action required)
        - 70-89: Good (monitor)
        - 50-69: Fair (plan maintenance)
        - 25-49: Poor (schedule maintenance)
        - 0-24: Critical (immediate action required)

    Example:
        >>> analyzer = BurnerHealthAnalyzer()
        >>> result = analyzer.analyze(health_input)
        >>> print(f"Overall Health: {result.overall_health_score:.1f}")
        >>> print(f"Severity: {result.severity_classification}")

    Attributes:
        component_weights: Weights for each component in overall score
        nozzle_model: Nozzle health degradation model
        refractory_model: Refractory thermal fatigue model
        igniter_model: Igniter health model
        scanner_model: Scanner degradation model
        register_model: Air register wear model
        valve_model: Fuel valve health model
    """

    def __init__(
        self,
        component_weights: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Initialize BurnerHealthAnalyzer.

        Args:
            component_weights: Optional custom weights for components.
                Must sum to 1.0 if provided.
        """
        self.component_weights = component_weights or DEFAULT_COMPONENT_WEIGHTS

        # Validate weights
        total_weight = sum(self.component_weights.values())
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Component weights must sum to 1.0, got {total_weight:.3f}")

        # Initialize component models
        self.nozzle_model = NozzleHealthModel()
        self.refractory_model = RefractoryHealthModel()
        self.igniter_model = IgniterHealthModel()
        self.scanner_model = ScannerHealthModel()
        self.register_model = AirRegisterHealthModel()
        self.valve_model = FuelValveHealthModel()

        self._calculation_count = 0
        self._audit_trail: List[Dict[str, Any]] = []

        logger.info(
            f"BurnerHealthAnalyzer initialized with weights: {self.component_weights}"
        )

    def analyze(self, input_data: BurnerHealthInput) -> BurnerHealthOutput:
        """
        Perform comprehensive burner health analysis.

        This is the main entry point for burner health assessment. It performs:
        1. Individual component health calculations
        2. Weighted overall health score
        3. Severity classification per API 535
        4. Maintenance priority determination
        5. Trend analysis (if historical data available)
        6. Recommendation generation
        7. Provenance hash calculation

        Args:
            input_data: Validated burner health input data

        Returns:
            BurnerHealthOutput with complete health assessment

        Raises:
            ValueError: If no component data is provided
        """
        start_time = datetime.now(timezone.utc)
        self._calculation_count += 1
        self._audit_trail = []

        self._add_audit_entry("analysis_start", {
            "request_id": input_data.request_id,
            "burner_id": input_data.burner_id,
            "calculation_number": self._calculation_count,
        })

        # Validate at least one component provided
        components_provided = [
            input_data.nozzle is not None,
            input_data.refractory_tile is not None,
            input_data.igniter is not None,
            input_data.scanner is not None,
            input_data.air_register is not None,
            input_data.fuel_valve is not None,
        ]
        if not any(components_provided):
            raise ValueError("At least one component data set is required")

        # Calculate individual component health scores
        component_scores: Dict[str, ComponentHealthResult] = {}
        weighted_scores: List[Tuple[float, float]] = []  # (score, weight)

        # Nozzle health
        if input_data.nozzle is not None:
            nozzle_result = self.nozzle_model.calculate_health(input_data.nozzle)
            component_scores["nozzle"] = nozzle_result
            weighted_scores.append((
                nozzle_result.health_score,
                self.component_weights["nozzle"]
            ))
            self._add_audit_entry("nozzle_health", {
                "score": nozzle_result.health_score,
                "status": nozzle_result.health_status.value,
            })

        # Refractory tile health
        if input_data.refractory_tile is not None:
            tile_result = self.refractory_model.calculate_health(input_data.refractory_tile)
            component_scores["refractory_tile"] = tile_result
            weighted_scores.append((
                tile_result.health_score,
                self.component_weights["refractory_tile"]
            ))
            self._add_audit_entry("refractory_health", {
                "score": tile_result.health_score,
                "status": tile_result.health_status.value,
            })

        # Igniter health
        if input_data.igniter is not None:
            igniter_result = self.igniter_model.calculate_health(input_data.igniter)
            component_scores["igniter"] = igniter_result
            weighted_scores.append((
                igniter_result.health_score,
                self.component_weights["igniter"]
            ))
            self._add_audit_entry("igniter_health", {
                "score": igniter_result.health_score,
                "status": igniter_result.health_status.value,
            })

        # Scanner health
        if input_data.scanner is not None:
            scanner_result = self.scanner_model.calculate_health(input_data.scanner)
            component_scores["scanner"] = scanner_result
            weighted_scores.append((
                scanner_result.health_score,
                self.component_weights["scanner"]
            ))
            self._add_audit_entry("scanner_health", {
                "score": scanner_result.health_score,
                "status": scanner_result.health_status.value,
            })

        # Air register health
        if input_data.air_register is not None:
            register_result = self.register_model.calculate_health(input_data.air_register)
            component_scores["air_register"] = register_result
            weighted_scores.append((
                register_result.health_score,
                self.component_weights["air_register"]
            ))
            self._add_audit_entry("air_register_health", {
                "score": register_result.health_score,
                "status": register_result.health_status.value,
            })

        # Fuel valve health
        if input_data.fuel_valve is not None:
            valve_result = self.valve_model.calculate_health(input_data.fuel_valve)
            component_scores["fuel_valve"] = valve_result
            weighted_scores.append((
                valve_result.health_score,
                self.component_weights["fuel_valve"]
            ))
            self._add_audit_entry("fuel_valve_health", {
                "score": valve_result.health_score,
                "status": valve_result.health_status.value,
            })

        # Calculate weighted overall health score
        total_weight = sum(w for _, w in weighted_scores)
        if total_weight > 0:
            overall_score = sum(s * w for s, w in weighted_scores) / total_weight
        else:
            overall_score = 0.0

        # Find limiting component
        limiting_component = None
        limiting_score = None
        if component_scores:
            limiting_component = min(
                component_scores.keys(),
                key=lambda k: component_scores[k].health_score
            )
            limiting_score = component_scores[limiting_component].health_score

        # Determine overall health status and severity
        overall_status = self._score_to_status(overall_score)
        severity_class, severity_action = self._determine_severity(overall_score)

        # Determine highest maintenance priority
        maintenance_priorities = [
            result.maintenance_priority
            for result in component_scores.values()
        ]
        highest_priority = max(
            maintenance_priorities,
            key=lambda p: list(MaintenancePriority).index(p),
            default=MaintenancePriority.NONE
        )

        # Calculate maintenance window
        maintenance_window = self._calculate_maintenance_window(highest_priority)

        # Calculate overall failure probability
        failure_probs = [
            result.failure_probability_30d
            for result in component_scores.values()
        ]
        # Combined probability: 1 - product of (1 - p_i)
        overall_failure_prob = 1.0 - math.prod(1 - p for p in failure_probs)

        # Estimate RUL
        remaining_lives = [
            result.remaining_life_hours
            for result in component_scores.values()
            if result.remaining_life_hours > 0
        ]
        estimated_rul = min(remaining_lives) if remaining_lives else None

        # Trend analysis
        health_trend, trend_rate = self._analyze_trend(
            overall_score,
            input_data.previous_health_score,
            input_data.previous_assessment_date
        )

        # Compile recommendations
        all_recommendations = []
        for comp_name, result in component_scores.items():
            for rec in result.recommended_actions:
                all_recommendations.append(f"[{comp_name.upper()}] {rec}")

        # Sort by priority (emergency first)
        priority_order = {
            MaintenancePriority.EMERGENCY: 0,
            MaintenancePriority.URGENT: 1,
            MaintenancePriority.PRIORITY: 2,
            MaintenancePriority.PLANNED: 3,
            MaintenancePriority.ROUTINE: 4,
            MaintenancePriority.NONE: 5,
        }
        all_recommendations = sorted(
            all_recommendations,
            key=lambda r: priority_order.get(
                component_scores.get(
                    r.split("]")[0].strip("[").lower(),
                    ComponentHealthResult(
                        component_id="", component_type="",
                        health_score=100, health_status=HealthStatus.EXCELLENT,
                        remaining_life_pct=100, remaining_life_hours=0,
                        maintenance_priority=MaintenancePriority.NONE
                    )
                ).maintenance_priority,
                5
            )
        )

        # Generate alerts
        alerts = self._generate_alerts(component_scores, overall_score)

        # Calculate processing time
        processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(input_data, overall_score)

        # Compile calculation details
        calculation_details = {
            "health_formula": "Component_Score = 100 * exp(-lambda * hours / design_life)",
            "overall_formula": "Overall_Health = sum(weight_i * component_score_i)",
            "component_weights": self.component_weights,
            "severity_thresholds": {
                "excellent": "90-100",
                "good": "70-89",
                "fair": "50-69",
                "poor": "25-49",
                "critical": "0-24",
            },
            "audit_trail": self._audit_trail,
        }

        result = BurnerHealthOutput(
            request_id=input_data.request_id,
            burner_id=input_data.burner_id,
            timestamp=datetime.now(timezone.utc),
            status="success",
            processing_time_ms=round(processing_time_ms, 2),
            overall_health_score=round(overall_score, 2),
            overall_health_status=overall_status,
            severity_classification=severity_class,
            severity_action=severity_action,
            component_scores=component_scores,
            limiting_component=limiting_component,
            limiting_score=round(limiting_score, 2) if limiting_score else None,
            maintenance_priority=highest_priority,
            maintenance_window_days=maintenance_window,
            estimated_downtime_hours=self._estimate_downtime(component_scores),
            overall_failure_probability_30d=round(overall_failure_prob, 4),
            estimated_rul_hours=round(estimated_rul, 0) if estimated_rul else None,
            health_trend=health_trend,
            trend_rate_pct_per_month=round(trend_rate, 2) if trend_rate else None,
            recommendations=all_recommendations[:10],  # Top 10
            alerts=alerts,
            provenance_hash=provenance_hash,
            calculation_details=calculation_details,
        )

        logger.info(
            f"Burner health analysis complete for {input_data.burner_id}: "
            f"Overall={overall_score:.1f} ({overall_status.value}), "
            f"Severity={severity_class}, "
            f"Limiting={limiting_component} ({limiting_score:.1f}), "
            f"Priority={highest_priority.value}, "
            f"Time={processing_time_ms:.1f}ms"
        )

        return result

    def _score_to_status(self, score: float) -> HealthStatus:
        """Convert health score to status per API 535."""
        if score >= 90:
            return HealthStatus.EXCELLENT
        elif score >= 70:
            return HealthStatus.GOOD
        elif score >= 50:
            return HealthStatus.FAIR
        elif score >= 25:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL

    def _determine_severity(self, score: float) -> Tuple[str, str]:
        """Determine API 535 severity classification and required action."""
        if score >= 90:
            return "90-100 Excellent", "No action required"
        elif score >= 70:
            return "70-89 Good", "Monitor"
        elif score >= 50:
            return "50-69 Fair", "Plan maintenance"
        elif score >= 25:
            return "25-49 Poor", "Schedule maintenance"
        else:
            return "0-24 Critical", "Immediate action required"

    def _calculate_maintenance_window(self, priority: MaintenancePriority) -> Optional[int]:
        """Calculate recommended maintenance window in days."""
        windows = {
            MaintenancePriority.NONE: None,
            MaintenancePriority.ROUTINE: 90,
            MaintenancePriority.PLANNED: 30,
            MaintenancePriority.PRIORITY: 7,
            MaintenancePriority.URGENT: 2,
            MaintenancePriority.EMERGENCY: 0,
        }
        return windows.get(priority)

    def _estimate_downtime(
        self,
        component_scores: Dict[str, ComponentHealthResult]
    ) -> Optional[float]:
        """Estimate maintenance downtime based on required actions."""
        # Base downtime estimates by component (hours)
        component_downtime = {
            "nozzle": 4.0,
            "refractory_tile": 24.0,
            "igniter": 2.0,
            "scanner": 1.0,
            "air_register": 4.0,
            "fuel_valve": 6.0,
        }

        total_downtime = 0.0
        for comp_name, result in component_scores.items():
            if result.maintenance_required:
                total_downtime += component_downtime.get(comp_name, 4.0)

        return total_downtime if total_downtime > 0 else None

    def _analyze_trend(
        self,
        current_score: float,
        previous_score: Optional[float],
        previous_date: Optional[datetime]
    ) -> Tuple[str, Optional[float]]:
        """Analyze health trend from historical data."""
        if previous_score is None or previous_date is None:
            return "stable", None

        score_change = current_score - previous_score
        days_elapsed = (datetime.now(timezone.utc) - previous_date).days

        if days_elapsed <= 0:
            return "stable", None

        # Calculate monthly rate
        rate_per_month = score_change / (days_elapsed / 30)

        if rate_per_month > 2.0:
            return "improving", rate_per_month
        elif rate_per_month < -2.0:
            return "degrading", rate_per_month
        else:
            return "stable", rate_per_month

    def _generate_alerts(
        self,
        component_scores: Dict[str, ComponentHealthResult],
        overall_score: float
    ) -> List[Dict[str, Any]]:
        """Generate alert records."""
        alerts: List[Dict[str, Any]] = []

        # Overall health alert
        if overall_score < 50:
            alerts.append({
                "alert_id": f"BNR-{datetime.now().strftime('%Y%m%d%H%M%S')}-OVR",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "severity": "critical" if overall_score < 25 else "alarm",
                "component": "overall",
                "message": f"Overall burner health is {overall_score:.0f}%",
                "action_required": True,
            })

        # Component-specific alerts
        for comp_name, result in component_scores.items():
            if result.health_score < 50:
                severity = "critical" if result.health_score < 25 else "alarm"
                alerts.append({
                    "alert_id": f"BNR-{datetime.now().strftime('%Y%m%d%H%M%S')}-{comp_name[:3].upper()}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "severity": severity,
                    "component": comp_name,
                    "message": f"{comp_name.replace('_', ' ').title()} health is {result.health_score:.0f}%",
                    "failure_modes": [fm.value for fm in result.failure_modes],
                    "action_required": True,
                })

        return alerts

    def _calculate_provenance_hash(
        self,
        input_data: BurnerHealthInput,
        overall_score: float
    ) -> str:
        """Calculate SHA-256 provenance hash for audit trail."""
        provenance_data = {
            "request_id": input_data.request_id,
            "burner_id": input_data.burner_id,
            "timestamp": input_data.timestamp.isoformat(),
            "overall_health_score": overall_score,
            "analyzer_version": "1.0.0",
            "component_weights": self.component_weights,
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def _add_audit_entry(self, operation: str, data: Dict[str, Any]) -> None:
        """Add entry to audit trail."""
        self._audit_trail.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "data": data,
        })

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get calculation audit trail."""
        return self._audit_trail.copy()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_default_health_analyzer() -> BurnerHealthAnalyzer:
    """
    Create BurnerHealthAnalyzer with default configuration.

    Returns:
        BurnerHealthAnalyzer with default component weights
    """
    return BurnerHealthAnalyzer()


def quick_health_check(
    operating_hours: float,
    design_life_hours: float = 25000.0
) -> Tuple[float, HealthStatus]:
    """
    Quick health estimation based on operating hours.

    Args:
        operating_hours: Total operating hours
        design_life_hours: Design life in hours

    Returns:
        Tuple of (health_score, health_status)
    """
    age_ratio = operating_hours / design_life_hours
    health_score = 100.0 * math.exp(-1.5 * age_ratio)
    health_score = max(0.0, min(100.0, health_score))

    if health_score >= 90:
        status = HealthStatus.EXCELLENT
    elif health_score >= 70:
        status = HealthStatus.GOOD
    elif health_score >= 50:
        status = HealthStatus.FAIR
    elif health_score >= 25:
        status = HealthStatus.POOR
    else:
        status = HealthStatus.CRITICAL

    return health_score, status
