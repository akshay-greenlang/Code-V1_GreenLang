# -*- coding: utf-8 -*-
"""
Combustion Diagnostics Calculator for GL-005 CombustionControlAgent

Real-time combustion fault detection, flame pattern analysis, and diagnostic trending.
Zero-hallucination design using deterministic signal processing and combustion physics.

Reference Standards:
- NFPA 85: Boiler and Combustion Systems Hazards Code
- NFPA 86: Standard for Ovens and Furnaces
- ISA-77.44.01: Fossil Fuel Power Plant - Drum-Type Boiler Control
- API 556: Instrumentation, Control, and Protective Systems for Gas Fired Heaters

Mathematical Formulas:
- Sensor Drift: drift_rate = (current - baseline) / time_elapsed
- Cross-Limiting: O2_trim = clamp(O2_setpoint + bias, O2_min, O2_max)
- Instability Index: II = sqrt(sum((x_i - mean)^2) / N) / mean
- Diagnostic Score: DS = product(1 - fault_probability_i) for all i

Author: GreenLang GL-005 Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================


class FaultType(str, Enum):
    """Types of combustion faults"""
    FLAME_INSTABILITY = "flame_instability"
    SENSOR_DRIFT = "sensor_drift"
    FUEL_QUALITY_DEGRADATION = "fuel_quality_degradation"
    AIR_FUEL_IMBALANCE = "air_fuel_imbalance"
    BURNER_FOULING = "burner_fouling"
    IGNITION_FAILURE = "ignition_failure"
    FLAME_ROLLOVER = "flame_rollover"
    INCOMPLETE_COMBUSTION = "incomplete_combustion"
    CROSS_LIMIT_VIOLATION = "cross_limit_violation"
    TRIM_CONTROL_SATURATION = "trim_control_saturation"


class FaultSeverity(str, Enum):
    """Severity levels for detected faults"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FlamePattern(str, Enum):
    """Flame pattern classifications"""
    STABLE = "stable"
    PULSATING = "pulsating"
    LIFTING = "lifting"
    IMPINGING = "impinging"
    DETACHED = "detached"
    ASYMMETRIC = "asymmetric"


class SensorType(str, Enum):
    """Sensor types for drift compensation"""
    O2_ANALYZER = "o2_analyzer"
    CO_ANALYZER = "co_analyzer"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    FLOW = "flow"
    FLAME_SCANNER = "flame_scanner"


class TrendDirection(str, Enum):
    """Trend analysis directions"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    OSCILLATING = "oscillating"


# =============================================================================
# FROZEN DATACLASSES
# =============================================================================


@dataclass(frozen=True)
class FaultDetectionResult:
    """Immutable result of fault detection analysis"""
    fault_type: FaultType
    severity: FaultSeverity
    confidence: float  # 0.0 to 1.0
    detected_at: datetime
    description: str
    recommended_action: str
    affected_parameters: Tuple[str, ...]
    provenance_hash: str


@dataclass(frozen=True)
class FlamePatternMetrics:
    """Immutable flame pattern analysis metrics"""
    pattern_type: FlamePattern
    stability_index: float  # 0.0 to 1.0
    pulsation_frequency_hz: float
    pulsation_amplitude_percent: float
    lift_distance_mm: float
    asymmetry_index: float  # 0.0 to 1.0
    luminosity_variance: float
    provenance_hash: str


@dataclass(frozen=True)
class SensorDriftCompensation:
    """Immutable sensor drift compensation result"""
    sensor_type: SensorType
    sensor_id: str
    baseline_value: float
    current_value: float
    drift_amount: float
    drift_rate_per_hour: float
    compensation_factor: float
    calibration_recommended: bool
    time_since_calibration_hours: float
    provenance_hash: str


@dataclass(frozen=True)
class CrossLimitParameters:
    """Immutable cross-limiting control parameters"""
    fuel_demand_percent: float
    air_demand_percent: float
    fuel_actual_percent: float
    air_actual_percent: float
    fuel_lead_lag_seconds: float
    air_lead_lag_seconds: float
    is_fuel_limited: bool
    is_air_limited: bool
    cross_limit_active: bool
    provenance_hash: str


@dataclass(frozen=True)
class TrimControlParameters:
    """Immutable trim control tuning parameters"""
    o2_setpoint_percent: float
    o2_actual_percent: float
    o2_trim_output_percent: float
    co_trim_output_percent: float
    combined_trim_percent: float
    trim_rate_limit_per_minute: float
    is_saturated_high: bool
    is_saturated_low: bool
    provenance_hash: str


@dataclass(frozen=True)
class FuelQualityMetrics:
    """Immutable fuel quality assessment metrics"""
    heating_value_mj_kg: float
    heating_value_deviation_percent: float
    wobbe_index: float
    wobbe_deviation_percent: float
    specific_gravity: float
    estimated_composition_change: bool
    quality_score: float  # 0.0 to 1.0
    provenance_hash: str


@dataclass(frozen=True)
class DiagnosticTrend:
    """Immutable diagnostic trend analysis result"""
    parameter_name: str
    direction: TrendDirection
    slope: float
    r_squared: float  # Correlation coefficient
    forecast_value: float
    forecast_horizon_minutes: int
    alert_threshold_eta_minutes: Optional[int]
    provenance_hash: str


@dataclass(frozen=True)
class CombustionInstabilityIndicators:
    """Immutable combustion instability indicators"""
    pressure_oscillation_amplitude_pa: float
    pressure_oscillation_frequency_hz: float
    temperature_variance_c: float
    flame_flicker_index: float
    combustion_noise_db: float
    instability_score: float  # 0.0 to 1.0
    is_thermoacoustic: bool
    provenance_hash: str


@dataclass(frozen=True)
class DiagnosticSummary:
    """Immutable comprehensive diagnostic summary"""
    timestamp: datetime
    overall_health_score: float  # 0.0 to 1.0
    active_faults: Tuple[FaultDetectionResult, ...]
    flame_pattern: FlamePatternMetrics
    instability_indicators: CombustionInstabilityIndicators
    sensor_drift_status: Tuple[SensorDriftCompensation, ...]
    cross_limit_status: CrossLimitParameters
    trim_control_status: TrimControlParameters
    fuel_quality: FuelQualityMetrics
    trends: Tuple[DiagnosticTrend, ...]
    requires_immediate_action: bool
    recommended_actions: Tuple[str, ...]
    provenance_hash: str


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================


class DiagnosticInput(BaseModel):
    """Input parameters for combustion diagnostics"""

    # Time series data
    temperature_readings_c: List[float] = Field(
        ...,
        description="Time-series temperature readings in Celsius",
        min_length=10,
        max_length=10000
    )
    pressure_readings_pa: List[float] = Field(
        ...,
        description="Time-series pressure readings in Pascal",
        min_length=10,
        max_length=10000
    )
    flame_intensity_readings: List[float] = Field(
        ...,
        description="Time-series flame intensity readings (0-100%)",
        min_length=10,
        max_length=10000
    )
    sampling_rate_hz: float = Field(
        default=10.0,
        ge=0.1,
        le=1000.0,
        description="Data sampling rate in Hz"
    )

    # Current operating parameters
    o2_actual_percent: float = Field(..., ge=0, le=21)
    co_actual_ppm: float = Field(..., ge=0)
    fuel_flow_kg_hr: float = Field(..., ge=0)
    air_flow_kg_hr: float = Field(..., ge=0)
    combustion_temperature_c: float = Field(..., ge=0, le=2000)
    furnace_pressure_pa: float = Field(...)

    # Setpoints
    o2_setpoint_percent: float = Field(..., ge=0, le=21)
    temperature_setpoint_c: float = Field(..., ge=0, le=2000)

    # Control outputs
    fuel_demand_percent: float = Field(..., ge=0, le=100)
    air_demand_percent: float = Field(..., ge=0, le=100)
    fuel_actual_percent: float = Field(..., ge=0, le=100)
    air_actual_percent: float = Field(..., ge=0, le=100)
    o2_trim_output_percent: float = Field(default=0, ge=-100, le=100)
    co_trim_output_percent: float = Field(default=0, ge=-100, le=100)

    # Fuel properties
    fuel_heating_value_mj_kg: float = Field(..., gt=0)
    fuel_specific_gravity: float = Field(default=0.6, gt=0)
    reference_heating_value_mj_kg: float = Field(default=50.0, gt=0)
    reference_wobbe_index: float = Field(default=50.0, gt=0)

    # Sensor calibration data
    sensor_baselines: Dict[str, float] = Field(
        default_factory=dict,
        description="Baseline values for sensor drift detection"
    )
    time_since_calibration_hours: Dict[str, float] = Field(
        default_factory=dict,
        description="Hours since last calibration per sensor"
    )

    # Cross-limiting parameters
    fuel_lead_lag_seconds: float = Field(default=0.5, ge=0)
    air_lead_lag_seconds: float = Field(default=0.5, ge=0)
    cross_limit_enabled: bool = Field(default=True)

    # Trim control limits
    trim_rate_limit_per_minute: float = Field(default=2.0, ge=0)
    trim_high_limit_percent: float = Field(default=15.0, ge=0)
    trim_low_limit_percent: float = Field(default=-15.0, le=0)

    # Historical data for trending
    historical_o2_readings: List[float] = Field(
        default_factory=list,
        description="Historical O2 readings for trend analysis"
    )
    historical_co_readings: List[float] = Field(
        default_factory=list,
        description="Historical CO readings for trend analysis"
    )
    historical_timestamps_minutes: List[float] = Field(
        default_factory=list,
        description="Timestamps for historical data in minutes"
    )

    @field_validator('temperature_readings_c', 'pressure_readings_pa', 'flame_intensity_readings')
    @classmethod
    def validate_readings_not_empty(cls, v: List[float]) -> List[float]:
        """Ensure readings are not empty and have valid values"""
        if not v:
            raise ValueError("Readings list cannot be empty")
        return v


class DiagnosticOutput(BaseModel):
    """Output from combustion diagnostics analysis"""

    summary: DiagnosticSummary = Field(..., description="Comprehensive diagnostic summary")
    processing_time_ms: float = Field(..., description="Processing duration in milliseconds")
    calculation_timestamp: datetime = Field(..., description="Timestamp of calculation")

    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# THREAD-SAFE CACHE
# =============================================================================


class ThreadSafeCache:
    """Thread-safe LRU cache for expensive calculations"""

    def __init__(self, maxsize: int = 1000):
        self._cache: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._maxsize = maxsize

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        with self._lock:
            if len(self._cache) >= self._maxsize:
                # Remove oldest entry
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[key] = value

    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()


# =============================================================================
# MAIN CALCULATOR CLASS
# =============================================================================


class CombustionDiagnosticsCalculator:
    """
    Real-time combustion diagnostics calculator.

    Zero-hallucination design using deterministic signal processing and
    combustion physics. All calculations are reproducible with SHA-256
    provenance tracking.

    Features:
    - Real-time combustion fault detection
    - Flame pattern analysis metrics
    - Combustion instability indicators
    - Sensor drift compensation
    - Cross-limiting logic validation
    - Fuel quality impact assessment
    - Trim control tuning parameters
    - Diagnostic trend analysis

    Thread-safe with LRU caching for performance optimization.
    """

    # Fault detection thresholds
    FLAME_INSTABILITY_THRESHOLD = 0.15  # CV > 15% = unstable
    SENSOR_DRIFT_THRESHOLD = 0.05  # 5% drift from baseline
    FUEL_QUALITY_DEVIATION_THRESHOLD = 0.10  # 10% heating value deviation
    CO_HIGH_THRESHOLD_PPM = 100  # High CO indicates incomplete combustion
    O2_LOW_THRESHOLD = 1.0  # Low O2 indicates rich combustion
    O2_HIGH_THRESHOLD = 8.0  # High O2 indicates lean combustion

    # Instability thresholds
    THERMOACOUSTIC_FREQUENCY_MIN_HZ = 50.0
    THERMOACOUSTIC_FREQUENCY_MAX_HZ = 500.0
    PRESSURE_OSCILLATION_THRESHOLD_PA = 500.0

    # Cross-limiting parameters
    CROSS_LIMIT_MARGIN_PERCENT = 2.0

    def __init__(self):
        """Initialize combustion diagnostics calculator"""
        self._logger = logging.getLogger(__name__)
        self._cache = ThreadSafeCache(maxsize=1000)

    def calculate_diagnostics(
        self,
        diagnostic_input: DiagnosticInput
    ) -> DiagnosticOutput:
        """
        Perform comprehensive combustion diagnostics analysis.

        Args:
            diagnostic_input: Input parameters for diagnostics

        Returns:
            DiagnosticOutput with complete diagnostic summary
        """
        start_time = datetime.now(timezone.utc)
        self._logger.info("Starting combustion diagnostics calculation")

        try:
            # Step 1: Detect faults
            faults = self._detect_all_faults(diagnostic_input)

            # Step 2: Analyze flame pattern
            flame_pattern = self._analyze_flame_pattern(
                diagnostic_input.flame_intensity_readings,
                diagnostic_input.sampling_rate_hz
            )

            # Step 3: Calculate instability indicators
            instability = self._calculate_instability_indicators(
                diagnostic_input.pressure_readings_pa,
                diagnostic_input.temperature_readings_c,
                diagnostic_input.flame_intensity_readings,
                diagnostic_input.sampling_rate_hz
            )

            # Step 4: Calculate sensor drift compensation
            sensor_drift = self._calculate_sensor_drift(
                diagnostic_input
            )

            # Step 5: Validate cross-limiting logic
            cross_limit = self._validate_cross_limiting(
                diagnostic_input
            )

            # Step 6: Calculate trim control parameters
            trim_control = self._calculate_trim_parameters(
                diagnostic_input
            )

            # Step 7: Assess fuel quality
            fuel_quality = self._assess_fuel_quality(
                diagnostic_input
            )

            # Step 8: Analyze trends
            trends = self._analyze_trends(
                diagnostic_input
            )

            # Step 9: Calculate overall health score
            health_score = self._calculate_health_score(
                faults,
                flame_pattern,
                instability,
                sensor_drift,
                fuel_quality
            )

            # Step 10: Generate recommendations
            requires_action, recommendations = self._generate_recommendations(
                faults,
                flame_pattern,
                instability,
                sensor_drift,
                cross_limit,
                trim_control,
                fuel_quality
            )

            # Create summary
            summary_data = {
                "timestamp": start_time.isoformat(),
                "health_score": health_score,
                "faults": [f.fault_type.value for f in faults],
                "flame_pattern": flame_pattern.pattern_type.value,
                "instability_score": instability.instability_score
            }
            summary_hash = self._compute_hash(summary_data)

            summary = DiagnosticSummary(
                timestamp=start_time,
                overall_health_score=health_score,
                active_faults=tuple(faults),
                flame_pattern=flame_pattern,
                instability_indicators=instability,
                sensor_drift_status=tuple(sensor_drift),
                cross_limit_status=cross_limit,
                trim_control_status=trim_control,
                fuel_quality=fuel_quality,
                trends=tuple(trends),
                requires_immediate_action=requires_action,
                recommended_actions=tuple(recommendations),
                provenance_hash=summary_hash
            )

            end_time = datetime.now(timezone.utc)
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            return DiagnosticOutput(
                summary=summary,
                processing_time_ms=processing_time_ms,
                calculation_timestamp=start_time
            )

        except Exception as e:
            self._logger.error(f"Diagnostics calculation failed: {e}", exc_info=True)
            raise

    def detect_fault(
        self,
        fault_type: FaultType,
        diagnostic_input: DiagnosticInput
    ) -> Optional[FaultDetectionResult]:
        """
        Detect a specific type of combustion fault.

        Args:
            fault_type: Type of fault to detect
            diagnostic_input: Input parameters

        Returns:
            FaultDetectionResult if fault detected, None otherwise
        """
        detection_methods = {
            FaultType.FLAME_INSTABILITY: self._detect_flame_instability,
            FaultType.SENSOR_DRIFT: self._detect_sensor_drift_fault,
            FaultType.FUEL_QUALITY_DEGRADATION: self._detect_fuel_quality_fault,
            FaultType.AIR_FUEL_IMBALANCE: self._detect_air_fuel_imbalance,
            FaultType.INCOMPLETE_COMBUSTION: self._detect_incomplete_combustion,
            FaultType.CROSS_LIMIT_VIOLATION: self._detect_cross_limit_violation,
            FaultType.TRIM_CONTROL_SATURATION: self._detect_trim_saturation,
        }

        detector = detection_methods.get(fault_type)
        if detector:
            return detector(diagnostic_input)
        return None

    def analyze_flame_pattern(
        self,
        intensity_readings: List[float],
        sampling_rate_hz: float
    ) -> FlamePatternMetrics:
        """
        Analyze flame pattern from intensity readings.

        Args:
            intensity_readings: Time-series flame intensity data
            sampling_rate_hz: Sampling rate in Hz

        Returns:
            FlamePatternMetrics with pattern analysis
        """
        return self._analyze_flame_pattern(intensity_readings, sampling_rate_hz)

    def calculate_sensor_drift(
        self,
        sensor_type: SensorType,
        sensor_id: str,
        current_value: float,
        baseline_value: float,
        time_since_calibration_hours: float
    ) -> SensorDriftCompensation:
        """
        Calculate sensor drift and compensation factor.

        Args:
            sensor_type: Type of sensor
            sensor_id: Unique sensor identifier
            current_value: Current sensor reading
            baseline_value: Baseline calibration value
            time_since_calibration_hours: Hours since last calibration

        Returns:
            SensorDriftCompensation with drift analysis
        """
        drift_amount = current_value - baseline_value
        drift_rate = drift_amount / time_since_calibration_hours if time_since_calibration_hours > 0 else 0

        # Compensation factor (inverse of drift)
        if baseline_value != 0:
            drift_percent = abs(drift_amount) / abs(baseline_value)
            compensation_factor = baseline_value / current_value if current_value != 0 else 1.0
        else:
            drift_percent = 0
            compensation_factor = 1.0

        # Recommend calibration if drift exceeds threshold
        calibration_recommended = drift_percent > self.SENSOR_DRIFT_THRESHOLD

        drift_data = {
            "sensor_type": sensor_type.value,
            "sensor_id": sensor_id,
            "drift_amount": drift_amount,
            "drift_rate": drift_rate
        }
        drift_hash = self._compute_hash(drift_data)

        return SensorDriftCompensation(
            sensor_type=sensor_type,
            sensor_id=sensor_id,
            baseline_value=baseline_value,
            current_value=current_value,
            drift_amount=self._round_decimal(drift_amount, 4),
            drift_rate_per_hour=self._round_decimal(drift_rate, 6),
            compensation_factor=self._round_decimal(compensation_factor, 4),
            calibration_recommended=calibration_recommended,
            time_since_calibration_hours=time_since_calibration_hours,
            provenance_hash=drift_hash
        )

    def validate_cross_limiting(
        self,
        fuel_demand: float,
        air_demand: float,
        fuel_actual: float,
        air_actual: float,
        fuel_lead_lag: float,
        air_lead_lag: float
    ) -> CrossLimitParameters:
        """
        Validate cross-limiting logic parameters.

        Cross-limiting ensures:
        - On load increase: Air leads fuel
        - On load decrease: Fuel leads air
        - Prevents fuel-rich conditions

        Args:
            fuel_demand: Fuel demand percentage
            air_demand: Air demand percentage
            fuel_actual: Actual fuel percentage
            air_actual: Actual air percentage
            fuel_lead_lag: Fuel lead/lag time in seconds
            air_lead_lag: Air lead/lag time in seconds

        Returns:
            CrossLimitParameters with validation results
        """
        # Check if cross-limiting is active
        is_fuel_limited = fuel_actual < fuel_demand - self.CROSS_LIMIT_MARGIN_PERCENT
        is_air_limited = air_actual < air_demand - self.CROSS_LIMIT_MARGIN_PERCENT
        cross_limit_active = is_fuel_limited or is_air_limited

        cross_limit_data = {
            "fuel_demand": fuel_demand,
            "air_demand": air_demand,
            "fuel_actual": fuel_actual,
            "air_actual": air_actual
        }
        cross_limit_hash = self._compute_hash(cross_limit_data)

        return CrossLimitParameters(
            fuel_demand_percent=fuel_demand,
            air_demand_percent=air_demand,
            fuel_actual_percent=fuel_actual,
            air_actual_percent=air_actual,
            fuel_lead_lag_seconds=fuel_lead_lag,
            air_lead_lag_seconds=air_lead_lag,
            is_fuel_limited=is_fuel_limited,
            is_air_limited=is_air_limited,
            cross_limit_active=cross_limit_active,
            provenance_hash=cross_limit_hash
        )

    def assess_fuel_quality(
        self,
        heating_value_mj_kg: float,
        specific_gravity: float,
        reference_heating_value: float,
        reference_wobbe_index: float
    ) -> FuelQualityMetrics:
        """
        Assess fuel quality impact on combustion.

        Args:
            heating_value_mj_kg: Measured heating value
            specific_gravity: Fuel specific gravity
            reference_heating_value: Reference heating value
            reference_wobbe_index: Reference Wobbe index

        Returns:
            FuelQualityMetrics with quality assessment
        """
        # Calculate Wobbe index: W = HV / sqrt(SG)
        wobbe_index = heating_value_mj_kg / math.sqrt(specific_gravity) if specific_gravity > 0 else 0

        # Calculate deviations
        hv_deviation = abs(heating_value_mj_kg - reference_heating_value) / reference_heating_value
        wobbe_deviation = abs(wobbe_index - reference_wobbe_index) / reference_wobbe_index

        # Estimate if composition has changed significantly
        composition_change = hv_deviation > 0.05 or wobbe_deviation > 0.05

        # Quality score (1.0 = perfect match to reference)
        quality_score = 1.0 - (hv_deviation + wobbe_deviation) / 2
        quality_score = max(0.0, min(1.0, quality_score))

        quality_data = {
            "heating_value": heating_value_mj_kg,
            "wobbe_index": wobbe_index,
            "quality_score": quality_score
        }
        quality_hash = self._compute_hash(quality_data)

        return FuelQualityMetrics(
            heating_value_mj_kg=self._round_decimal(heating_value_mj_kg, 2),
            heating_value_deviation_percent=self._round_decimal(hv_deviation * 100, 2),
            wobbe_index=self._round_decimal(wobbe_index, 2),
            wobbe_deviation_percent=self._round_decimal(wobbe_deviation * 100, 2),
            specific_gravity=self._round_decimal(specific_gravity, 4),
            estimated_composition_change=composition_change,
            quality_score=self._round_decimal(quality_score, 4),
            provenance_hash=quality_hash
        )

    def calculate_trim_parameters(
        self,
        o2_setpoint: float,
        o2_actual: float,
        o2_trim_output: float,
        co_trim_output: float,
        trim_rate_limit: float,
        trim_high_limit: float,
        trim_low_limit: float
    ) -> TrimControlParameters:
        """
        Calculate trim control tuning parameters.

        Args:
            o2_setpoint: O2 setpoint percentage
            o2_actual: Actual O2 percentage
            o2_trim_output: Current O2 trim output
            co_trim_output: Current CO trim output
            trim_rate_limit: Maximum trim rate per minute
            trim_high_limit: High trim limit
            trim_low_limit: Low trim limit

        Returns:
            TrimControlParameters with trim status
        """
        # Combined trim (O2 trim + CO trim with priority to CO)
        combined_trim = o2_trim_output + co_trim_output

        # Check for saturation
        is_saturated_high = combined_trim >= trim_high_limit
        is_saturated_low = combined_trim <= trim_low_limit

        trim_data = {
            "o2_setpoint": o2_setpoint,
            "o2_actual": o2_actual,
            "combined_trim": combined_trim
        }
        trim_hash = self._compute_hash(trim_data)

        return TrimControlParameters(
            o2_setpoint_percent=o2_setpoint,
            o2_actual_percent=o2_actual,
            o2_trim_output_percent=o2_trim_output,
            co_trim_output_percent=co_trim_output,
            combined_trim_percent=self._round_decimal(combined_trim, 2),
            trim_rate_limit_per_minute=trim_rate_limit,
            is_saturated_high=is_saturated_high,
            is_saturated_low=is_saturated_low,
            provenance_hash=trim_hash
        )

    def analyze_diagnostic_trend(
        self,
        parameter_name: str,
        values: List[float],
        timestamps_minutes: List[float],
        alert_threshold: Optional[float] = None,
        forecast_horizon_minutes: int = 60
    ) -> DiagnosticTrend:
        """
        Analyze diagnostic parameter trend using linear regression.

        Args:
            parameter_name: Name of the parameter
            values: Historical values
            timestamps_minutes: Timestamps in minutes
            alert_threshold: Optional threshold for ETA calculation
            forecast_horizon_minutes: Forecast horizon in minutes

        Returns:
            DiagnosticTrend with trend analysis
        """
        if len(values) < 2 or len(values) != len(timestamps_minutes):
            # Not enough data for trend analysis
            trend_hash = self._compute_hash({"parameter": parameter_name, "values": values})
            return DiagnosticTrend(
                parameter_name=parameter_name,
                direction=TrendDirection.STABLE,
                slope=0.0,
                r_squared=0.0,
                forecast_value=values[-1] if values else 0.0,
                forecast_horizon_minutes=forecast_horizon_minutes,
                alert_threshold_eta_minutes=None,
                provenance_hash=trend_hash
            )

        # Linear regression: y = mx + b
        n = len(values)
        sum_x = sum(timestamps_minutes)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(timestamps_minutes, values))
        sum_x2 = sum(x * x for x in timestamps_minutes)

        # Calculate slope (m) and intercept (b)
        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            slope = 0.0
            intercept = sum_y / n
        else:
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n

        # Calculate R-squared
        mean_y = sum_y / n
        ss_tot = sum((y - mean_y) ** 2 for y in values)
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(timestamps_minutes, values))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r_squared = max(0.0, min(1.0, r_squared))

        # Determine direction
        if abs(slope) < 1e-6:
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING

        # Check for oscillating pattern
        if r_squared < 0.3 and abs(slope) < 0.1:
            direction = TrendDirection.OSCILLATING

        # Forecast value
        last_time = timestamps_minutes[-1]
        forecast_time = last_time + forecast_horizon_minutes
        forecast_value = slope * forecast_time + intercept

        # Calculate ETA to alert threshold
        alert_eta = None
        if alert_threshold is not None and abs(slope) > 1e-6:
            time_to_threshold = (alert_threshold - values[-1]) / slope
            if time_to_threshold > 0:
                alert_eta = int(time_to_threshold)

        trend_data = {
            "parameter": parameter_name,
            "slope": slope,
            "r_squared": r_squared,
            "forecast": forecast_value
        }
        trend_hash = self._compute_hash(trend_data)

        return DiagnosticTrend(
            parameter_name=parameter_name,
            direction=direction,
            slope=self._round_decimal(slope, 6),
            r_squared=self._round_decimal(r_squared, 4),
            forecast_value=self._round_decimal(forecast_value, 4),
            forecast_horizon_minutes=forecast_horizon_minutes,
            alert_threshold_eta_minutes=alert_eta,
            provenance_hash=trend_hash
        )

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _detect_all_faults(
        self,
        diagnostic_input: DiagnosticInput
    ) -> List[FaultDetectionResult]:
        """Detect all applicable faults"""
        faults = []

        fault_types = [
            FaultType.FLAME_INSTABILITY,
            FaultType.AIR_FUEL_IMBALANCE,
            FaultType.INCOMPLETE_COMBUSTION,
            FaultType.CROSS_LIMIT_VIOLATION,
            FaultType.TRIM_CONTROL_SATURATION,
            FaultType.FUEL_QUALITY_DEGRADATION,
        ]

        for fault_type in fault_types:
            result = self.detect_fault(fault_type, diagnostic_input)
            if result is not None:
                faults.append(result)

        return faults

    def _detect_flame_instability(
        self,
        diagnostic_input: DiagnosticInput
    ) -> Optional[FaultDetectionResult]:
        """Detect flame instability fault"""
        readings = diagnostic_input.flame_intensity_readings
        mean_val = sum(readings) / len(readings)
        if mean_val == 0:
            return None

        variance = sum((x - mean_val) ** 2 for x in readings) / len(readings)
        cv = math.sqrt(variance) / mean_val

        if cv > self.FLAME_INSTABILITY_THRESHOLD:
            severity = FaultSeverity.HIGH if cv > 0.3 else FaultSeverity.MEDIUM
            confidence = min(cv / 0.5, 1.0)

            fault_data = {
                "type": FaultType.FLAME_INSTABILITY.value,
                "cv": cv,
                "severity": severity.value
            }

            return FaultDetectionResult(
                fault_type=FaultType.FLAME_INSTABILITY,
                severity=severity,
                confidence=self._round_decimal(confidence, 4),
                detected_at=datetime.now(timezone.utc),
                description=f"Flame coefficient of variation {cv:.2%} exceeds threshold",
                recommended_action="Check burner alignment and fuel/air ratio",
                affected_parameters=("flame_intensity", "combustion_stability"),
                provenance_hash=self._compute_hash(fault_data)
            )
        return None

    def _detect_sensor_drift_fault(
        self,
        diagnostic_input: DiagnosticInput
    ) -> Optional[FaultDetectionResult]:
        """Detect sensor drift fault"""
        # Check O2 sensor drift as primary indicator
        if "o2_analyzer" in diagnostic_input.sensor_baselines:
            baseline = diagnostic_input.sensor_baselines["o2_analyzer"]
            current = diagnostic_input.o2_actual_percent
            drift_percent = abs(current - baseline) / baseline if baseline > 0 else 0

            if drift_percent > self.SENSOR_DRIFT_THRESHOLD:
                fault_data = {
                    "type": FaultType.SENSOR_DRIFT.value,
                    "drift_percent": drift_percent
                }

                return FaultDetectionResult(
                    fault_type=FaultType.SENSOR_DRIFT,
                    severity=FaultSeverity.MEDIUM,
                    confidence=min(drift_percent / 0.1, 1.0),
                    detected_at=datetime.now(timezone.utc),
                    description=f"O2 sensor drift {drift_percent:.2%} detected",
                    recommended_action="Schedule O2 analyzer calibration",
                    affected_parameters=("o2_actual", "o2_trim"),
                    provenance_hash=self._compute_hash(fault_data)
                )
        return None

    def _detect_fuel_quality_fault(
        self,
        diagnostic_input: DiagnosticInput
    ) -> Optional[FaultDetectionResult]:
        """Detect fuel quality degradation fault"""
        hv_deviation = abs(
            diagnostic_input.fuel_heating_value_mj_kg -
            diagnostic_input.reference_heating_value_mj_kg
        ) / diagnostic_input.reference_heating_value_mj_kg

        if hv_deviation > self.FUEL_QUALITY_DEVIATION_THRESHOLD:
            fault_data = {
                "type": FaultType.FUEL_QUALITY_DEGRADATION.value,
                "hv_deviation": hv_deviation
            }

            return FaultDetectionResult(
                fault_type=FaultType.FUEL_QUALITY_DEGRADATION,
                severity=FaultSeverity.MEDIUM,
                confidence=min(hv_deviation / 0.2, 1.0),
                detected_at=datetime.now(timezone.utc),
                description=f"Fuel heating value deviation {hv_deviation:.2%}",
                recommended_action="Verify fuel supply and adjust combustion controls",
                affected_parameters=("fuel_heating_value", "combustion_efficiency"),
                provenance_hash=self._compute_hash(fault_data)
            )
        return None

    def _detect_air_fuel_imbalance(
        self,
        diagnostic_input: DiagnosticInput
    ) -> Optional[FaultDetectionResult]:
        """Detect air/fuel imbalance fault"""
        o2 = diagnostic_input.o2_actual_percent

        if o2 < self.O2_LOW_THRESHOLD:
            fault_data = {"type": FaultType.AIR_FUEL_IMBALANCE.value, "o2": o2, "condition": "rich"}
            return FaultDetectionResult(
                fault_type=FaultType.AIR_FUEL_IMBALANCE,
                severity=FaultSeverity.HIGH,
                confidence=1.0 - o2 / self.O2_LOW_THRESHOLD,
                detected_at=datetime.now(timezone.utc),
                description=f"Fuel-rich condition: O2 at {o2:.1f}%",
                recommended_action="Increase combustion air or reduce fuel",
                affected_parameters=("o2_actual", "fuel_flow", "air_flow"),
                provenance_hash=self._compute_hash(fault_data)
            )
        elif o2 > self.O2_HIGH_THRESHOLD:
            fault_data = {"type": FaultType.AIR_FUEL_IMBALANCE.value, "o2": o2, "condition": "lean"}
            return FaultDetectionResult(
                fault_type=FaultType.AIR_FUEL_IMBALANCE,
                severity=FaultSeverity.LOW,
                confidence=(o2 - self.O2_HIGH_THRESHOLD) / 5.0,
                detected_at=datetime.now(timezone.utc),
                description=f"Fuel-lean condition: O2 at {o2:.1f}%",
                recommended_action="Reduce excess air for better efficiency",
                affected_parameters=("o2_actual", "air_flow"),
                provenance_hash=self._compute_hash(fault_data)
            )
        return None

    def _detect_incomplete_combustion(
        self,
        diagnostic_input: DiagnosticInput
    ) -> Optional[FaultDetectionResult]:
        """Detect incomplete combustion fault"""
        if diagnostic_input.co_actual_ppm > self.CO_HIGH_THRESHOLD_PPM:
            severity = FaultSeverity.CRITICAL if diagnostic_input.co_actual_ppm > 500 else FaultSeverity.HIGH
            fault_data = {
                "type": FaultType.INCOMPLETE_COMBUSTION.value,
                "co_ppm": diagnostic_input.co_actual_ppm
            }

            return FaultDetectionResult(
                fault_type=FaultType.INCOMPLETE_COMBUSTION,
                severity=severity,
                confidence=min(diagnostic_input.co_actual_ppm / 1000, 1.0),
                detected_at=datetime.now(timezone.utc),
                description=f"High CO detected: {diagnostic_input.co_actual_ppm:.0f} ppm",
                recommended_action="Check burner flame pattern and increase excess air",
                affected_parameters=("co_actual", "combustion_efficiency"),
                provenance_hash=self._compute_hash(fault_data)
            )
        return None

    def _detect_cross_limit_violation(
        self,
        diagnostic_input: DiagnosticInput
    ) -> Optional[FaultDetectionResult]:
        """Detect cross-limiting violation"""
        if not diagnostic_input.cross_limit_enabled:
            return None

        fuel_demand = diagnostic_input.fuel_demand_percent
        air_demand = diagnostic_input.air_demand_percent
        fuel_actual = diagnostic_input.fuel_actual_percent
        air_actual = diagnostic_input.air_actual_percent

        # Cross-limiting should prevent fuel > air on increasing demand
        if fuel_actual > air_actual + self.CROSS_LIMIT_MARGIN_PERCENT:
            fault_data = {
                "type": FaultType.CROSS_LIMIT_VIOLATION.value,
                "fuel_actual": fuel_actual,
                "air_actual": air_actual
            }

            return FaultDetectionResult(
                fault_type=FaultType.CROSS_LIMIT_VIOLATION,
                severity=FaultSeverity.HIGH,
                confidence=0.9,
                detected_at=datetime.now(timezone.utc),
                description="Cross-limiting violation: fuel exceeds air",
                recommended_action="Check cross-limiting logic and air damper position",
                affected_parameters=("fuel_actual", "air_actual", "cross_limit"),
                provenance_hash=self._compute_hash(fault_data)
            )
        return None

    def _detect_trim_saturation(
        self,
        diagnostic_input: DiagnosticInput
    ) -> Optional[FaultDetectionResult]:
        """Detect trim control saturation"""
        combined_trim = (
            diagnostic_input.o2_trim_output_percent +
            diagnostic_input.co_trim_output_percent
        )

        if combined_trim >= diagnostic_input.trim_high_limit_percent:
            fault_data = {"type": FaultType.TRIM_CONTROL_SATURATION.value, "trim": combined_trim}
            return FaultDetectionResult(
                fault_type=FaultType.TRIM_CONTROL_SATURATION,
                severity=FaultSeverity.MEDIUM,
                confidence=0.8,
                detected_at=datetime.now(timezone.utc),
                description="O2 trim at high limit - insufficient air authority",
                recommended_action="Check damper positioning and O2 setpoint",
                affected_parameters=("o2_trim", "co_trim", "air_flow"),
                provenance_hash=self._compute_hash(fault_data)
            )
        elif combined_trim <= diagnostic_input.trim_low_limit_percent:
            fault_data = {"type": FaultType.TRIM_CONTROL_SATURATION.value, "trim": combined_trim}
            return FaultDetectionResult(
                fault_type=FaultType.TRIM_CONTROL_SATURATION,
                severity=FaultSeverity.MEDIUM,
                confidence=0.8,
                detected_at=datetime.now(timezone.utc),
                description="O2 trim at low limit - excess air authority",
                recommended_action="Review O2 setpoint curve",
                affected_parameters=("o2_trim", "co_trim", "air_flow"),
                provenance_hash=self._compute_hash(fault_data)
            )
        return None

    def _analyze_flame_pattern(
        self,
        intensity_readings: List[float],
        sampling_rate_hz: float
    ) -> FlamePatternMetrics:
        """Analyze flame pattern from intensity readings"""
        n = len(intensity_readings)
        mean_intensity = sum(intensity_readings) / n
        variance = sum((x - mean_intensity) ** 2 for x in intensity_readings) / n
        std_dev = math.sqrt(variance)

        # Stability index
        cv = std_dev / mean_intensity if mean_intensity > 0 else 0
        stability_index = 1.0 / (1.0 + cv)

        # Detect pulsation via zero-crossing analysis
        crossings = 0
        for i in range(1, n):
            if (intensity_readings[i-1] < mean_intensity <= intensity_readings[i] or
                intensity_readings[i-1] >= mean_intensity > intensity_readings[i]):
                crossings += 1

        duration_seconds = n / sampling_rate_hz
        pulsation_freq = (crossings / 2.0) / duration_seconds if duration_seconds > 0 else 0

        # Pulsation amplitude
        pulsation_amplitude = (max(intensity_readings) - min(intensity_readings)) / mean_intensity * 100 if mean_intensity > 0 else 0

        # Asymmetry index (compare first and second half variance)
        half = n // 2
        first_half_var = sum((x - mean_intensity) ** 2 for x in intensity_readings[:half]) / half if half > 0 else 0
        second_half_var = sum((x - mean_intensity) ** 2 for x in intensity_readings[half:]) / (n - half) if n > half else 0
        asymmetry = abs(first_half_var - second_half_var) / (first_half_var + second_half_var + 1e-10)

        # Classify pattern
        if stability_index > 0.9:
            pattern = FlamePattern.STABLE
        elif pulsation_freq > 5.0:
            pattern = FlamePattern.PULSATING
        elif mean_intensity < 30:
            pattern = FlamePattern.LIFTING
        elif asymmetry > 0.5:
            pattern = FlamePattern.ASYMMETRIC
        else:
            pattern = FlamePattern.STABLE

        pattern_data = {
            "pattern": pattern.value,
            "stability": stability_index,
            "pulsation_freq": pulsation_freq
        }

        return FlamePatternMetrics(
            pattern_type=pattern,
            stability_index=self._round_decimal(stability_index, 4),
            pulsation_frequency_hz=self._round_decimal(pulsation_freq, 2),
            pulsation_amplitude_percent=self._round_decimal(pulsation_amplitude, 2),
            lift_distance_mm=0.0,  # Would require additional sensor
            asymmetry_index=self._round_decimal(asymmetry, 4),
            luminosity_variance=self._round_decimal(variance, 4),
            provenance_hash=self._compute_hash(pattern_data)
        )

    def _calculate_instability_indicators(
        self,
        pressure_readings: List[float],
        temperature_readings: List[float],
        flame_readings: List[float],
        sampling_rate_hz: float
    ) -> CombustionInstabilityIndicators:
        """Calculate combustion instability indicators"""
        n = len(pressure_readings)

        # Pressure oscillation
        pressure_mean = sum(pressure_readings) / n
        pressure_variance = sum((x - pressure_mean) ** 2 for x in pressure_readings) / n
        pressure_amplitude = max(pressure_readings) - min(pressure_readings)

        # Pressure oscillation frequency
        crossings = 0
        for i in range(1, n):
            if (pressure_readings[i-1] < pressure_mean <= pressure_readings[i] or
                pressure_readings[i-1] >= pressure_mean > pressure_readings[i]):
                crossings += 1
        duration = n / sampling_rate_hz
        pressure_freq = (crossings / 2.0) / duration if duration > 0 else 0

        # Temperature variance
        temp_mean = sum(temperature_readings) / n
        temp_variance = sum((x - temp_mean) ** 2 for x in temperature_readings) / n

        # Flame flicker index
        flame_mean = sum(flame_readings) / n
        flame_variance = sum((x - flame_mean) ** 2 for x in flame_readings) / n
        flame_flicker = math.sqrt(flame_variance) / flame_mean if flame_mean > 0 else 0

        # Estimate combustion noise (simplified)
        noise_db = 20 * math.log10(pressure_amplitude + 1)

        # Instability score (composite)
        instability_score = (
            0.4 * min(pressure_amplitude / 1000, 1.0) +
            0.3 * min(math.sqrt(temp_variance) / 100, 1.0) +
            0.3 * min(flame_flicker / 0.5, 1.0)
        )
        instability_score = min(1.0, max(0.0, instability_score))

        # Check for thermoacoustic instability
        is_thermoacoustic = (
            self.THERMOACOUSTIC_FREQUENCY_MIN_HZ <= pressure_freq <= self.THERMOACOUSTIC_FREQUENCY_MAX_HZ and
            pressure_amplitude > self.PRESSURE_OSCILLATION_THRESHOLD_PA
        )

        instability_data = {
            "pressure_amplitude": pressure_amplitude,
            "pressure_freq": pressure_freq,
            "instability_score": instability_score
        }

        return CombustionInstabilityIndicators(
            pressure_oscillation_amplitude_pa=self._round_decimal(pressure_amplitude, 2),
            pressure_oscillation_frequency_hz=self._round_decimal(pressure_freq, 2),
            temperature_variance_c=self._round_decimal(temp_variance, 4),
            flame_flicker_index=self._round_decimal(flame_flicker, 4),
            combustion_noise_db=self._round_decimal(noise_db, 2),
            instability_score=self._round_decimal(instability_score, 4),
            is_thermoacoustic=is_thermoacoustic,
            provenance_hash=self._compute_hash(instability_data)
        )

    def _calculate_sensor_drift(
        self,
        diagnostic_input: DiagnosticInput
    ) -> List[SensorDriftCompensation]:
        """Calculate drift for all sensors"""
        results = []

        sensor_mappings = [
            (SensorType.O2_ANALYZER, "o2_analyzer", diagnostic_input.o2_actual_percent),
            (SensorType.TEMPERATURE, "temperature", diagnostic_input.combustion_temperature_c),
            (SensorType.PRESSURE, "pressure", diagnostic_input.furnace_pressure_pa),
        ]

        for sensor_type, sensor_id, current_value in sensor_mappings:
            if sensor_id in diagnostic_input.sensor_baselines:
                baseline = diagnostic_input.sensor_baselines[sensor_id]
                time_since_cal = diagnostic_input.time_since_calibration_hours.get(sensor_id, 0)

                result = self.calculate_sensor_drift(
                    sensor_type,
                    sensor_id,
                    current_value,
                    baseline,
                    time_since_cal
                )
                results.append(result)

        return results

    def _validate_cross_limiting(
        self,
        diagnostic_input: DiagnosticInput
    ) -> CrossLimitParameters:
        """Validate cross-limiting from input"""
        return self.validate_cross_limiting(
            diagnostic_input.fuel_demand_percent,
            diagnostic_input.air_demand_percent,
            diagnostic_input.fuel_actual_percent,
            diagnostic_input.air_actual_percent,
            diagnostic_input.fuel_lead_lag_seconds,
            diagnostic_input.air_lead_lag_seconds
        )

    def _calculate_trim_parameters(
        self,
        diagnostic_input: DiagnosticInput
    ) -> TrimControlParameters:
        """Calculate trim parameters from input"""
        return self.calculate_trim_parameters(
            diagnostic_input.o2_setpoint_percent,
            diagnostic_input.o2_actual_percent,
            diagnostic_input.o2_trim_output_percent,
            diagnostic_input.co_trim_output_percent,
            diagnostic_input.trim_rate_limit_per_minute,
            diagnostic_input.trim_high_limit_percent,
            diagnostic_input.trim_low_limit_percent
        )

    def _assess_fuel_quality(
        self,
        diagnostic_input: DiagnosticInput
    ) -> FuelQualityMetrics:
        """Assess fuel quality from input"""
        return self.assess_fuel_quality(
            diagnostic_input.fuel_heating_value_mj_kg,
            diagnostic_input.fuel_specific_gravity,
            diagnostic_input.reference_heating_value_mj_kg,
            diagnostic_input.reference_wobbe_index
        )

    def _analyze_trends(
        self,
        diagnostic_input: DiagnosticInput
    ) -> List[DiagnosticTrend]:
        """Analyze trends from historical data"""
        trends = []

        if diagnostic_input.historical_o2_readings and diagnostic_input.historical_timestamps_minutes:
            o2_trend = self.analyze_diagnostic_trend(
                "o2_percent",
                diagnostic_input.historical_o2_readings,
                diagnostic_input.historical_timestamps_minutes,
                alert_threshold=self.O2_HIGH_THRESHOLD
            )
            trends.append(o2_trend)

        if diagnostic_input.historical_co_readings and diagnostic_input.historical_timestamps_minutes:
            co_trend = self.analyze_diagnostic_trend(
                "co_ppm",
                diagnostic_input.historical_co_readings,
                diagnostic_input.historical_timestamps_minutes,
                alert_threshold=self.CO_HIGH_THRESHOLD_PPM
            )
            trends.append(co_trend)

        return trends

    def _calculate_health_score(
        self,
        faults: List[FaultDetectionResult],
        flame_pattern: FlamePatternMetrics,
        instability: CombustionInstabilityIndicators,
        sensor_drift: List[SensorDriftCompensation],
        fuel_quality: FuelQualityMetrics
    ) -> float:
        """Calculate overall health score"""
        # Start with perfect score
        score = 1.0

        # Deduct for faults
        for fault in faults:
            if fault.severity == FaultSeverity.CRITICAL:
                score -= 0.3
            elif fault.severity == FaultSeverity.HIGH:
                score -= 0.2
            elif fault.severity == FaultSeverity.MEDIUM:
                score -= 0.1
            elif fault.severity == FaultSeverity.LOW:
                score -= 0.05

        # Factor in flame stability
        score *= flame_pattern.stability_index

        # Factor in instability
        score *= (1.0 - instability.instability_score * 0.5)

        # Factor in fuel quality
        score *= fuel_quality.quality_score

        # Factor in sensor drift
        for drift in sensor_drift:
            if drift.calibration_recommended:
                score *= 0.95

        return self._round_decimal(max(0.0, min(1.0, score)), 4)

    def _generate_recommendations(
        self,
        faults: List[FaultDetectionResult],
        flame_pattern: FlamePatternMetrics,
        instability: CombustionInstabilityIndicators,
        sensor_drift: List[SensorDriftCompensation],
        cross_limit: CrossLimitParameters,
        trim_control: TrimControlParameters,
        fuel_quality: FuelQualityMetrics
    ) -> Tuple[bool, List[str]]:
        """Generate recommendations based on diagnostics"""
        recommendations = []
        requires_action = False

        # Add fault recommendations
        for fault in faults:
            recommendations.append(f"[{fault.severity.value.upper()}] {fault.recommended_action}")
            if fault.severity in (FaultSeverity.HIGH, FaultSeverity.CRITICAL):
                requires_action = True

        # Flame pattern recommendations
        if flame_pattern.stability_index < 0.7:
            recommendations.append("Low flame stability - inspect burner and adjust fuel/air ratio")
            requires_action = True

        # Instability recommendations
        if instability.is_thermoacoustic:
            recommendations.append("CRITICAL: Thermoacoustic instability detected - reduce firing rate")
            requires_action = True

        # Sensor drift recommendations
        for drift in sensor_drift:
            if drift.calibration_recommended:
                recommendations.append(f"Schedule {drift.sensor_type.value} calibration (drift: {drift.drift_amount:.4f})")

        # Trim control recommendations
        if trim_control.is_saturated_high or trim_control.is_saturated_low:
            recommendations.append("Trim control saturated - review O2 setpoint curve")

        # Fuel quality recommendations
        if fuel_quality.estimated_composition_change:
            recommendations.append("Fuel composition change detected - verify fuel supply")

        if not recommendations:
            recommendations.append("System operating normally - no action required")

        return requires_action, recommendations

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for provenance tracking"""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _round_decimal(self, value: float, places: int) -> float:
        """Round to specified decimal places using ROUND_HALF_UP"""
        if value is None:
            return 0.0
        decimal_value = Decimal(str(value))
        quantize_string = '0.' + '0' * places if places > 0 else '1'
        rounded = decimal_value.quantize(Decimal(quantize_string), rounding=ROUND_HALF_UP)
        return float(rounded)
