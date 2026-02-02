# -*- coding: utf-8 -*-
"""
Advanced Combustion Diagnostics Calculator for GL-005 CombustionControlAgent

Comprehensive real-time combustion fault detection, flame pattern analysis, burner health
scoring, and predictive maintenance diagnostics. Zero-hallucination design using
deterministic signal processing and combustion physics.

Reference Standards:
- ASME PTC 4.1: Fired Steam Generators Performance Test Codes
- NFPA 85: Boiler and Combustion Systems Hazards Code
- NFPA 86: Standard for Ovens and Furnaces
- ISA-77.44.01: Fossil Fuel Power Plant - Drum-Type Boiler Control
- API 556: Instrumentation, Control, and Protective Systems for Gas Fired Heaters
- EN 746-2: Industrial Thermoprocessing Equipment - Safety Requirements for Combustion

Key Features:
- Flame pattern analysis (stoichiometric, lean, rich diagnosis)
- Incomplete combustion detection (CO formation analysis)
- Combustion efficiency degradation trending
- Burner health scoring (0-100)
- Fuel quality variation detection
- Air distribution analysis (multi-zone)
- Soot formation prediction
- Flashback and blowoff risk assessment
- Maintenance recommendation generation with priority scoring

Mathematical Formulas:
- Equivalence Ratio: phi = (F/A)_actual / (F/A)_stoichiometric
- CO Formation Index: CFI = [CO] / ([CO] + [CO2]) * 1e6
- Soot Formation Index: SFI = f(phi, T, residence_time)
- Damkohler Number: Da = tau_flow / tau_chem (for blowoff prediction)
- Burner Health Score: BHS = w1*flame_stability + w2*emission_quality + w3*efficiency

Author: GreenLang GL-005 Team
Version: 2.0.0
Performance Target: <10ms per calculation
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - COMBUSTION PHYSICS
# =============================================================================

# Stoichiometric air-fuel ratios (kg air / kg fuel)
STOICHIOMETRIC_AFR = {
    "natural_gas": 17.2,
    "methane": 17.2,
    "propane": 15.7,
    "diesel": 14.5,
    "fuel_oil": 13.8,
    "coal": 10.8,
    "biomass": 6.5,
    "hydrogen": 34.3,
}

# Adiabatic flame temperatures (K) at stoichiometric conditions
ADIABATIC_FLAME_TEMP = {
    "natural_gas": 2223,
    "methane": 2223,
    "propane": 2253,
    "diesel": 2300,
    "fuel_oil": 2300,
    "coal": 2100,
    "hydrogen": 2400,
}

# Laminar flame speeds (m/s) at stoichiometric conditions
LAMINAR_FLAME_SPEED = {
    "natural_gas": 0.40,
    "methane": 0.40,
    "propane": 0.43,
    "diesel": 0.45,
    "hydrogen": 3.10,
    "coal": 0.20,
}

# CO equilibrium constants (for CO formation prediction)
CO_EQUILIBRIUM_CONSTANTS = {
    1000: 1.23e-4,
    1200: 2.45e-3,
    1400: 1.89e-2,
    1600: 8.12e-2,
    1800: 2.34e-1,
    2000: 5.67e-1,
}


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
    SOOT_FORMATION = "soot_formation"
    AIR_DISTRIBUTION_IMBALANCE = "air_distribution_imbalance"
    FLASHBACK_RISK = "flashback_risk"
    BLOWOFF_RISK = "blowoff_risk"
    THERMAL_NOX_EXCESSIVE = "thermal_nox_excessive"


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
    FLICKERING = "flickering"


class CombustionMode(str, Enum):
    """Combustion mode based on equivalence ratio"""
    STOICHIOMETRIC = "stoichiometric"  # phi = 0.95 - 1.05
    LEAN = "lean"                       # phi < 0.95
    RICH = "rich"                       # phi > 1.05
    ULTRA_LEAN = "ultra_lean"           # phi < 0.7
    VERY_RICH = "very_rich"             # phi > 1.3


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


class MaintenancePriority(str, Enum):
    """Maintenance recommendation priority levels"""
    IMMEDIATE = "immediate"
    URGENT = "urgent"
    SCHEDULED = "scheduled"
    ROUTINE = "routine"
    INFORMATIONAL = "informational"


class BurnerHealthCategory(str, Enum):
    """Burner health categories"""
    EXCELLENT = "excellent"    # 90-100
    GOOD = "good"              # 75-89
    FAIR = "fair"              # 60-74
    POOR = "poor"              # 40-59
    CRITICAL = "critical"      # 0-39


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
    combustion_mode: CombustionMode
    equivalence_ratio: float
    stability_index: float  # 0.0 to 1.0
    pulsation_frequency_hz: float
    pulsation_amplitude_percent: float
    lift_distance_mm: float
    asymmetry_index: float  # 0.0 to 1.0
    luminosity_variance: float
    flame_temperature_k: float
    provenance_hash: str


@dataclass(frozen=True)
class IncompleteCombustionMetrics:
    """Immutable incomplete combustion analysis"""
    co_formation_index: float  # CO / (CO + CO2) * 1e6
    co_concentration_ppm: float
    co_mass_rate_kg_hr: float
    combustion_efficiency_percent: float
    unburned_carbon_percent: float
    carbon_in_ash_percent: float
    chemical_efficiency_percent: float
    is_incomplete: bool
    severity: FaultSeverity
    root_cause: str
    provenance_hash: str


@dataclass(frozen=True)
class EfficiencyDegradationTrend:
    """Immutable efficiency degradation analysis"""
    current_efficiency_percent: float
    baseline_efficiency_percent: float
    degradation_percent: float
    degradation_rate_per_day: float
    days_to_maintenance_threshold: Optional[int]
    trend_confidence: float  # R-squared
    trend_direction: TrendDirection
    forecast_30_day_efficiency: float
    provenance_hash: str


@dataclass(frozen=True)
class BurnerHealthScore:
    """Immutable burner health scoring (0-100)"""
    overall_score: float  # 0-100
    category: BurnerHealthCategory
    flame_quality_score: float  # 0-100
    emission_quality_score: float  # 0-100
    efficiency_score: float  # 0-100
    stability_score: float  # 0-100
    fouling_score: float  # 0-100
    component_scores: Tuple[Tuple[str, float], ...]
    degradation_from_baseline: float
    estimated_remaining_life_days: Optional[int]
    provenance_hash: str


@dataclass(frozen=True)
class FuelQualityVariation:
    """Immutable fuel quality variation detection"""
    heating_value_mj_kg: float
    heating_value_deviation_percent: float
    wobbe_index: float
    wobbe_deviation_percent: float
    specific_gravity: float
    methane_number: Optional[float]
    hydrogen_content_variation: bool
    quality_score: float  # 0.0 to 1.0
    is_significant_variation: bool
    compensation_factor: float
    provenance_hash: str


@dataclass(frozen=True)
class ZoneAirDistribution:
    """Immutable single zone air distribution"""
    zone_id: str
    air_flow_percent: float
    target_flow_percent: float
    deviation_percent: float
    is_starved: bool
    is_over_supplied: bool
    damper_position_percent: float


@dataclass(frozen=True)
class AirDistributionAnalysis:
    """Immutable multi-zone air distribution analysis"""
    zones: Tuple[ZoneAirDistribution, ...]
    overall_balance_score: float  # 0-100
    total_air_flow_kg_hr: float
    primary_air_percent: float
    secondary_air_percent: float
    tertiary_air_percent: float
    air_staging_active: bool
    distribution_uniformity_index: float  # 0-1
    worst_zone_id: Optional[str]
    imbalance_severity: FaultSeverity
    provenance_hash: str


@dataclass(frozen=True)
class SootFormationPrediction:
    """Immutable soot formation prediction"""
    soot_formation_index: float  # 0-100
    soot_risk_level: FaultSeverity
    predicted_soot_rate_mg_nm3: float
    smoke_number: float  # 0-9 scale (Bacharach)
    is_sooting: bool
    contributing_factors: Tuple[str, ...]
    critical_equivalence_ratio: float
    current_equivalence_ratio: float
    margin_to_sooting: float
    provenance_hash: str


@dataclass(frozen=True)
class FlashbackBlowoffRisk:
    """Immutable flashback and blowoff risk assessment"""
    flashback_risk_score: float  # 0-100
    blowoff_risk_score: float  # 0-100
    flashback_severity: FaultSeverity
    blowoff_severity: FaultSeverity
    current_flame_velocity_m_s: float
    critical_flashback_velocity_m_s: float
    critical_blowoff_velocity_m_s: float
    damkohler_number: float  # Da > 1 stable, Da < 1 blowoff risk
    stability_margin_percent: float
    operating_regime: str  # "stable", "near_blowoff", "near_flashback"
    provenance_hash: str


@dataclass(frozen=True)
class MaintenanceRecommendation:
    """Immutable maintenance recommendation"""
    recommendation_id: str
    priority: MaintenancePriority
    category: str
    description: str
    detailed_action: str
    estimated_impact: str
    estimated_downtime_hours: float
    cost_category: str  # "low", "medium", "high"
    deadline_days: Optional[int]
    related_faults: Tuple[FaultType, ...]
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
class AdvancedDiagnosticSummary:
    """Immutable comprehensive advanced diagnostic summary"""
    timestamp: datetime
    overall_health_score: float  # 0.0 to 100.0
    burner_health: BurnerHealthScore
    active_faults: Tuple[FaultDetectionResult, ...]
    flame_pattern: FlamePatternMetrics
    incomplete_combustion: IncompleteCombustionMetrics
    efficiency_degradation: EfficiencyDegradationTrend
    fuel_quality: FuelQualityVariation
    air_distribution: AirDistributionAnalysis
    soot_prediction: SootFormationPrediction
    flashback_blowoff_risk: FlashbackBlowoffRisk
    instability_indicators: CombustionInstabilityIndicators
    sensor_drift_status: Tuple[SensorDriftCompensation, ...]
    cross_limit_status: CrossLimitParameters
    trim_control_status: TrimControlParameters
    trends: Tuple[DiagnosticTrend, ...]
    maintenance_recommendations: Tuple[MaintenanceRecommendation, ...]
    requires_immediate_action: bool
    provenance_hash: str


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================


class ZoneInput(BaseModel):
    """Input for a single combustion zone"""
    zone_id: str = Field(..., description="Zone identifier")
    air_flow_kg_hr: float = Field(..., ge=0, description="Zone air flow rate")
    target_flow_kg_hr: float = Field(..., ge=0, description="Target zone flow rate")
    damper_position_percent: float = Field(default=50.0, ge=0, le=100)
    temperature_c: Optional[float] = Field(None, ge=0, le=2000)


class AdvancedDiagnosticInput(BaseModel):
    """Input parameters for advanced combustion diagnostics"""

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
    co2_actual_percent: float = Field(default=10.0, ge=0, le=20)
    fuel_flow_kg_hr: float = Field(..., ge=0)
    air_flow_kg_hr: float = Field(..., ge=0)
    combustion_temperature_c: float = Field(..., ge=0, le=2000)
    furnace_pressure_pa: float = Field(...)

    # Fuel type and properties
    fuel_type: str = Field(default="natural_gas")
    fuel_heating_value_mj_kg: float = Field(..., gt=0)
    fuel_specific_gravity: float = Field(default=0.6, gt=0)
    reference_heating_value_mj_kg: float = Field(default=50.0, gt=0)
    reference_wobbe_index: float = Field(default=50.0, gt=0)
    fuel_hydrogen_content_percent: float = Field(default=23.0, ge=0, le=100)

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

    # Multi-zone air distribution
    zone_data: List[ZoneInput] = Field(
        default_factory=list,
        description="Multi-zone air distribution data"
    )

    # Burner parameters
    burner_diameter_mm: float = Field(default=100.0, ge=10, le=2000)
    burner_velocity_m_s: float = Field(default=20.0, ge=0, le=200)
    flame_length_mm: Optional[float] = Field(None, ge=0, le=10000)

    # Baseline and historical data
    baseline_efficiency_percent: float = Field(default=85.0, ge=50, le=100)
    efficiency_history: List[float] = Field(
        default_factory=list,
        description="Historical efficiency readings"
    )
    efficiency_timestamps_days: List[float] = Field(
        default_factory=list,
        description="Days since baseline for efficiency readings"
    )

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
    historical_o2_readings: List[float] = Field(default_factory=list)
    historical_co_readings: List[float] = Field(default_factory=list)
    historical_timestamps_minutes: List[float] = Field(default_factory=list)

    @field_validator('temperature_readings_c', 'pressure_readings_pa', 'flame_intensity_readings')
    @classmethod
    def validate_readings_not_empty(cls, v: List[float]) -> List[float]:
        """Ensure readings are not empty and have valid values"""
        if not v:
            raise ValueError("Readings list cannot be empty")
        return v

    @model_validator(mode='after')
    def validate_list_lengths(self) -> 'AdvancedDiagnosticInput':
        """Validate that time series data have consistent lengths"""
        temp_len = len(self.temperature_readings_c)
        pressure_len = len(self.pressure_readings_pa)
        flame_len = len(self.flame_intensity_readings)

        if not (temp_len == pressure_len == flame_len):
            raise ValueError(
                f"Time series data must have same length: "
                f"temp={temp_len}, pressure={pressure_len}, flame={flame_len}"
            )
        return self


class AdvancedDiagnosticOutput(BaseModel):
    """Output from advanced combustion diagnostics analysis"""

    summary: AdvancedDiagnosticSummary = Field(
        ...,
        description="Comprehensive advanced diagnostic summary"
    )
    processing_time_ms: float = Field(
        ...,
        description="Processing duration in milliseconds"
    )
    calculation_timestamp: datetime = Field(
        ...,
        description="Timestamp of calculation"
    )
    performance_target_met: bool = Field(
        ...,
        description="Whether <10ms performance target was met"
    )

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


class AdvancedCombustionDiagnosticsCalculator:
    """
    Advanced real-time combustion diagnostics calculator.

    Zero-hallucination design using deterministic signal processing and
    combustion physics. All calculations are reproducible with SHA-256
    provenance tracking.

    Features:
    - Flame pattern analysis (stoichiometric, lean, rich diagnosis)
    - Incomplete combustion detection (CO formation analysis)
    - Combustion efficiency degradation trending
    - Burner health scoring (0-100)
    - Fuel quality variation detection
    - Air distribution analysis (multi-zone)
    - Soot formation prediction
    - Flashback and blowoff risk assessment
    - Maintenance recommendation generation

    Thread-safe with LRU caching for performance optimization.
    Performance target: <10ms per calculation.

    Reference Standards:
    - ASME PTC 4.1: Fired Steam Generators Performance Test Codes
    - NFPA 85: Boiler and Combustion Systems Hazards Code
    """

    # Fault detection thresholds
    FLAME_INSTABILITY_THRESHOLD = 0.15  # CV > 15% = unstable
    SENSOR_DRIFT_THRESHOLD = 0.05  # 5% drift from baseline
    FUEL_QUALITY_DEVIATION_THRESHOLD = 0.10  # 10% heating value deviation
    CO_HIGH_THRESHOLD_PPM = 100  # High CO indicates incomplete combustion
    CO_CRITICAL_THRESHOLD_PPM = 500  # Critical CO level
    O2_LOW_THRESHOLD = 1.0  # Low O2 indicates rich combustion
    O2_HIGH_THRESHOLD = 8.0  # High O2 indicates lean combustion

    # Equivalence ratio thresholds
    STOICHIOMETRIC_LOW = 0.95
    STOICHIOMETRIC_HIGH = 1.05
    LEAN_LIMIT = 0.7
    RICH_LIMIT = 1.3

    # Instability thresholds
    THERMOACOUSTIC_FREQUENCY_MIN_HZ = 50.0
    THERMOACOUSTIC_FREQUENCY_MAX_HZ = 500.0
    PRESSURE_OSCILLATION_THRESHOLD_PA = 500.0

    # Cross-limiting parameters
    CROSS_LIMIT_MARGIN_PERCENT = 2.0

    # Burner health scoring weights
    HEALTH_WEIGHT_FLAME = 0.25
    HEALTH_WEIGHT_EMISSION = 0.25
    HEALTH_WEIGHT_EFFICIENCY = 0.30
    HEALTH_WEIGHT_STABILITY = 0.20

    # Maintenance thresholds
    EFFICIENCY_DEGRADATION_THRESHOLD = 5.0  # % below baseline
    MAINTENANCE_SCHEDULE_DAYS = 30

    def __init__(self):
        """Initialize advanced combustion diagnostics calculator"""
        self._logger = logging.getLogger(__name__)
        self._cache = ThreadSafeCache(maxsize=1000)
        self._recommendation_counter = 0

    def calculate_diagnostics(
        self,
        diagnostic_input: AdvancedDiagnosticInput
    ) -> AdvancedDiagnosticOutput:
        """
        Perform comprehensive advanced combustion diagnostics analysis.

        Args:
            diagnostic_input: Input parameters for diagnostics

        Returns:
            AdvancedDiagnosticOutput with complete diagnostic summary

        Performance:
            Target <10ms per calculation
        """
        start_time = time.perf_counter()
        start_datetime = datetime.now(timezone.utc)
        self._logger.info("Starting advanced combustion diagnostics calculation")

        try:
            # Step 1: Calculate equivalence ratio and combustion mode
            equivalence_ratio = self._calculate_equivalence_ratio(
                diagnostic_input.fuel_flow_kg_hr,
                diagnostic_input.air_flow_kg_hr,
                diagnostic_input.fuel_type
            )
            combustion_mode = self._classify_combustion_mode(equivalence_ratio)

            # Step 2: Analyze flame pattern with combustion mode
            flame_pattern = self._analyze_flame_pattern(
                diagnostic_input.flame_intensity_readings,
                diagnostic_input.sampling_rate_hz,
                equivalence_ratio,
                combustion_mode,
                diagnostic_input.combustion_temperature_c
            )

            # Step 3: Analyze incomplete combustion
            incomplete_combustion = self._analyze_incomplete_combustion(
                diagnostic_input.co_actual_ppm,
                diagnostic_input.co2_actual_percent,
                diagnostic_input.o2_actual_percent,
                diagnostic_input.fuel_flow_kg_hr,
                diagnostic_input.combustion_temperature_c,
                equivalence_ratio
            )

            # Step 4: Calculate efficiency degradation trend
            efficiency_degradation = self._calculate_efficiency_degradation(
                diagnostic_input.efficiency_history,
                diagnostic_input.efficiency_timestamps_days,
                diagnostic_input.baseline_efficiency_percent,
                incomplete_combustion.combustion_efficiency_percent
            )

            # Step 5: Calculate burner health score
            burner_health = self._calculate_burner_health(
                flame_pattern,
                incomplete_combustion,
                efficiency_degradation,
                diagnostic_input
            )

            # Step 6: Assess fuel quality variation
            fuel_quality = self._assess_fuel_quality(
                diagnostic_input.fuel_heating_value_mj_kg,
                diagnostic_input.fuel_specific_gravity,
                diagnostic_input.reference_heating_value_mj_kg,
                diagnostic_input.reference_wobbe_index,
                diagnostic_input.fuel_hydrogen_content_percent
            )

            # Step 7: Analyze multi-zone air distribution
            air_distribution = self._analyze_air_distribution(
                diagnostic_input.zone_data,
                diagnostic_input.air_flow_kg_hr
            )

            # Step 8: Predict soot formation
            soot_prediction = self._predict_soot_formation(
                equivalence_ratio,
                diagnostic_input.combustion_temperature_c,
                diagnostic_input.fuel_type,
                diagnostic_input.fuel_hydrogen_content_percent
            )

            # Step 9: Assess flashback and blowoff risk
            flashback_blowoff = self._assess_flashback_blowoff_risk(
                diagnostic_input.burner_velocity_m_s,
                diagnostic_input.fuel_type,
                equivalence_ratio,
                diagnostic_input.combustion_temperature_c,
                diagnostic_input.burner_diameter_mm
            )

            # Step 10: Calculate instability indicators
            instability = self._calculate_instability_indicators(
                diagnostic_input.pressure_readings_pa,
                diagnostic_input.temperature_readings_c,
                diagnostic_input.flame_intensity_readings,
                diagnostic_input.sampling_rate_hz
            )

            # Step 11: Detect all faults
            faults = self._detect_all_faults(
                diagnostic_input,
                flame_pattern,
                incomplete_combustion,
                fuel_quality,
                air_distribution,
                soot_prediction,
                flashback_blowoff
            )

            # Step 12: Calculate sensor drift
            sensor_drift = self._calculate_sensor_drift(diagnostic_input)

            # Step 13: Validate cross-limiting
            cross_limit = self._validate_cross_limiting(diagnostic_input)

            # Step 14: Calculate trim parameters
            trim_control = self._calculate_trim_parameters(diagnostic_input)

            # Step 15: Analyze trends
            trends = self._analyze_trends(diagnostic_input)

            # Step 16: Generate maintenance recommendations
            maintenance_recommendations = self._generate_maintenance_recommendations(
                faults,
                burner_health,
                efficiency_degradation,
                sensor_drift,
                soot_prediction,
                flashback_blowoff
            )

            # Step 17: Calculate overall health score
            overall_health = self._calculate_overall_health(
                burner_health,
                faults,
                instability,
                air_distribution
            )

            # Step 18: Determine if immediate action required
            requires_action = self._requires_immediate_action(
                faults,
                flashback_blowoff,
                soot_prediction,
                overall_health
            )

            # Create summary hash
            summary_data = {
                "timestamp": start_datetime.isoformat(),
                "health_score": overall_health,
                "burner_health": burner_health.overall_score,
                "faults": [f.fault_type.value for f in faults],
                "combustion_mode": combustion_mode.value,
                "equivalence_ratio": equivalence_ratio
            }
            summary_hash = self._compute_hash(summary_data)

            # Build summary
            summary = AdvancedDiagnosticSummary(
                timestamp=start_datetime,
                overall_health_score=overall_health,
                burner_health=burner_health,
                active_faults=tuple(faults),
                flame_pattern=flame_pattern,
                incomplete_combustion=incomplete_combustion,
                efficiency_degradation=efficiency_degradation,
                fuel_quality=fuel_quality,
                air_distribution=air_distribution,
                soot_prediction=soot_prediction,
                flashback_blowoff_risk=flashback_blowoff,
                instability_indicators=instability,
                sensor_drift_status=tuple(sensor_drift),
                cross_limit_status=cross_limit,
                trim_control_status=trim_control,
                trends=tuple(trends),
                maintenance_recommendations=tuple(maintenance_recommendations),
                requires_immediate_action=requires_action,
                provenance_hash=summary_hash
            )

            end_time = time.perf_counter()
            processing_time_ms = (end_time - start_time) * 1000
            performance_target_met = processing_time_ms < 10.0

            if not performance_target_met:
                self._logger.warning(
                    f"Performance target missed: {processing_time_ms:.2f}ms > 10ms"
                )

            return AdvancedDiagnosticOutput(
                summary=summary,
                processing_time_ms=self._round_decimal(processing_time_ms, 3),
                calculation_timestamp=start_datetime,
                performance_target_met=performance_target_met
            )

        except Exception as e:
            self._logger.error(f"Diagnostics calculation failed: {e}", exc_info=True)
            raise

    # =========================================================================
    # FLAME PATTERN ANALYSIS
    # =========================================================================

    def _calculate_equivalence_ratio(
        self,
        fuel_flow_kg_hr: float,
        air_flow_kg_hr: float,
        fuel_type: str
    ) -> float:
        """
        Calculate equivalence ratio (phi) for combustion diagnosis.

        Formula:
            phi = (F/A)_actual / (F/A)_stoichiometric
            phi = 1: Stoichiometric
            phi < 1: Lean (fuel-limited)
            phi > 1: Rich (air-limited)

        Args:
            fuel_flow_kg_hr: Fuel flow rate
            air_flow_kg_hr: Air flow rate
            fuel_type: Type of fuel

        Returns:
            Equivalence ratio (dimensionless)
        """
        if air_flow_kg_hr <= 0:
            return 0.0

        stoich_afr = STOICHIOMETRIC_AFR.get(fuel_type.lower(), 17.2)
        actual_afr = air_flow_kg_hr / fuel_flow_kg_hr if fuel_flow_kg_hr > 0 else float('inf')

        # phi = stoich_AFR / actual_AFR
        equivalence_ratio = stoich_afr / actual_afr if actual_afr > 0 else 0.0

        return self._round_decimal(equivalence_ratio, 4)

    def _classify_combustion_mode(self, equivalence_ratio: float) -> CombustionMode:
        """
        Classify combustion mode based on equivalence ratio.

        Classification:
            - Ultra-lean: phi < 0.7 (high excess air, low NOx but efficiency loss)
            - Lean: 0.7 <= phi < 0.95 (excess air, lower flame temp)
            - Stoichiometric: 0.95 <= phi <= 1.05 (optimal)
            - Rich: 1.05 < phi <= 1.3 (insufficient air, CO formation)
            - Very Rich: phi > 1.3 (dangerous, soot/CO formation)

        Args:
            equivalence_ratio: Calculated phi value

        Returns:
            CombustionMode classification
        """
        if equivalence_ratio < self.LEAN_LIMIT:
            return CombustionMode.ULTRA_LEAN
        elif equivalence_ratio < self.STOICHIOMETRIC_LOW:
            return CombustionMode.LEAN
        elif equivalence_ratio <= self.STOICHIOMETRIC_HIGH:
            return CombustionMode.STOICHIOMETRIC
        elif equivalence_ratio <= self.RICH_LIMIT:
            return CombustionMode.RICH
        else:
            return CombustionMode.VERY_RICH

    def _analyze_flame_pattern(
        self,
        intensity_readings: List[float],
        sampling_rate_hz: float,
        equivalence_ratio: float,
        combustion_mode: CombustionMode,
        temperature_c: float
    ) -> FlamePatternMetrics:
        """
        Analyze flame pattern with stoichiometric/lean/rich diagnosis.

        Args:
            intensity_readings: Time-series flame intensity data
            sampling_rate_hz: Sampling rate in Hz
            equivalence_ratio: Calculated equivalence ratio
            combustion_mode: Classified combustion mode
            temperature_c: Combustion temperature

        Returns:
            FlamePatternMetrics with comprehensive pattern analysis
        """
        n = len(intensity_readings)
        mean_intensity = sum(intensity_readings) / n
        variance = sum((x - mean_intensity) ** 2 for x in intensity_readings) / n
        std_dev = math.sqrt(variance)

        # Stability index (coefficient of variation based)
        cv = std_dev / mean_intensity if mean_intensity > 0 else 0
        stability_index = 1.0 / (1.0 + cv * 5)  # Scale CV impact

        # Pulsation detection via zero-crossing
        crossings = 0
        for i in range(1, n):
            if (intensity_readings[i-1] < mean_intensity <= intensity_readings[i] or
                intensity_readings[i-1] >= mean_intensity > intensity_readings[i]):
                crossings += 1

        duration_seconds = n / sampling_rate_hz
        pulsation_freq = (crossings / 2.0) / duration_seconds if duration_seconds > 0 else 0

        # Pulsation amplitude
        pulsation_amplitude = (
            (max(intensity_readings) - min(intensity_readings)) / mean_intensity * 100
            if mean_intensity > 0 else 0
        )

        # Asymmetry index
        half = n // 2
        first_half_mean = sum(intensity_readings[:half]) / half if half > 0 else 0
        second_half_mean = sum(intensity_readings[half:]) / (n - half) if n > half else 0
        asymmetry = (
            abs(first_half_mean - second_half_mean) / mean_intensity
            if mean_intensity > 0 else 0
        )

        # Estimate flame temperature (simplified)
        flame_temp_k = temperature_c + 273.15

        # Lift distance estimation (based on stability and mode)
        if combustion_mode == CombustionMode.ULTRA_LEAN:
            lift_distance = 50.0  # High lift in ultra-lean
        elif combustion_mode == CombustionMode.LEAN:
            lift_distance = 20.0
        elif stability_index < 0.7:
            lift_distance = 30.0
        else:
            lift_distance = 0.0

        # Classify pattern based on all metrics
        if stability_index > 0.9 and pulsation_amplitude < 10:
            pattern = FlamePattern.STABLE
        elif pulsation_freq > 5.0 and pulsation_amplitude > 20:
            pattern = FlamePattern.PULSATING
        elif lift_distance > 30:
            pattern = FlamePattern.LIFTING
        elif asymmetry > 0.3:
            pattern = FlamePattern.ASYMMETRIC
        elif cv > 0.3:
            pattern = FlamePattern.FLICKERING
        elif mean_intensity < 30:
            pattern = FlamePattern.DETACHED
        else:
            pattern = FlamePattern.STABLE

        pattern_data = {
            "pattern": pattern.value,
            "mode": combustion_mode.value,
            "phi": equivalence_ratio,
            "stability": stability_index
        }

        return FlamePatternMetrics(
            pattern_type=pattern,
            combustion_mode=combustion_mode,
            equivalence_ratio=equivalence_ratio,
            stability_index=self._round_decimal(stability_index, 4),
            pulsation_frequency_hz=self._round_decimal(pulsation_freq, 2),
            pulsation_amplitude_percent=self._round_decimal(pulsation_amplitude, 2),
            lift_distance_mm=self._round_decimal(lift_distance, 1),
            asymmetry_index=self._round_decimal(asymmetry, 4),
            luminosity_variance=self._round_decimal(variance, 4),
            flame_temperature_k=self._round_decimal(flame_temp_k, 1),
            provenance_hash=self._compute_hash(pattern_data)
        )

    # =========================================================================
    # INCOMPLETE COMBUSTION ANALYSIS
    # =========================================================================

    def _analyze_incomplete_combustion(
        self,
        co_ppm: float,
        co2_percent: float,
        o2_percent: float,
        fuel_flow_kg_hr: float,
        temperature_c: float,
        equivalence_ratio: float
    ) -> IncompleteCombustionMetrics:
        """
        Analyze incomplete combustion with CO formation analysis.

        CO Formation Index (CFI):
            CFI = [CO] / ([CO] + [CO2]) * 1e6

        The CFI indicates combustion completeness:
            - CFI < 100: Excellent combustion
            - CFI 100-500: Good combustion
            - CFI 500-2000: Fair combustion
            - CFI > 2000: Poor combustion (significant CO)

        Args:
            co_ppm: CO concentration in ppm
            co2_percent: CO2 concentration in percent
            o2_percent: O2 concentration in percent
            fuel_flow_kg_hr: Fuel flow rate
            temperature_c: Combustion temperature
            equivalence_ratio: Equivalence ratio

        Returns:
            IncompleteCombustionMetrics with detailed analysis
        """
        # CO formation index
        co2_ppm = co2_percent * 10000  # Convert to ppm
        co_formation_index = (
            co_ppm / (co_ppm + co2_ppm) * 1e6
            if (co_ppm + co2_ppm) > 0 else 0
        )

        # CO mass rate (kg/hr) - simplified
        # Assuming flue gas volume ~ 10 Nm3/kg fuel for natural gas
        flue_gas_volume = fuel_flow_kg_hr * 10  # Nm3/hr
        co_mg_per_nm3 = co_ppm * 28.01 / 22.4  # Convert ppm to mg/Nm3
        co_mass_rate = co_mg_per_nm3 * flue_gas_volume / 1e6  # kg/hr

        # Combustion efficiency (simplified Siegert formula approach)
        # Loss due to CO
        co_loss_percent = co_ppm * 0.001  # Approximate
        # Loss due to dry flue gas (simplified)
        excess_air = o2_percent / (21 - o2_percent) * 100 if o2_percent < 21 else 0
        stack_loss_percent = 0.5 * (1 + excess_air / 100) * 10  # Simplified

        combustion_efficiency = 100 - co_loss_percent - stack_loss_percent
        combustion_efficiency = max(0, min(100, combustion_efficiency))

        # Unburned carbon estimation
        unburned_carbon = co_ppm / 10000  # Simplified approximation
        carbon_in_ash = 0.0  # Would need ash analysis

        # Chemical efficiency
        chemical_efficiency = 100 - (co_formation_index / 100)
        chemical_efficiency = max(0, min(100, chemical_efficiency))

        # Determine if incomplete combustion is occurring
        is_incomplete = co_ppm > self.CO_HIGH_THRESHOLD_PPM

        # Severity classification
        if co_ppm > self.CO_CRITICAL_THRESHOLD_PPM:
            severity = FaultSeverity.CRITICAL
        elif co_ppm > 200:
            severity = FaultSeverity.HIGH
        elif co_ppm > self.CO_HIGH_THRESHOLD_PPM:
            severity = FaultSeverity.MEDIUM
        elif co_ppm > 50:
            severity = FaultSeverity.LOW
        else:
            severity = FaultSeverity.NONE

        # Root cause analysis
        if equivalence_ratio > 1.1:
            root_cause = "Fuel-rich combustion (insufficient air)"
        elif temperature_c < 800:
            root_cause = "Low combustion temperature (quenching)"
        elif o2_percent < 1.5:
            root_cause = "Oxygen deficiency in combustion zone"
        elif co_ppm > 200 and equivalence_ratio < 1.0:
            root_cause = "Poor fuel-air mixing or flame impingement"
        else:
            root_cause = "Within normal operating range"

        ic_data = {
            "cfi": co_formation_index,
            "co_ppm": co_ppm,
            "efficiency": combustion_efficiency,
            "severity": severity.value
        }

        return IncompleteCombustionMetrics(
            co_formation_index=self._round_decimal(co_formation_index, 2),
            co_concentration_ppm=self._round_decimal(co_ppm, 1),
            co_mass_rate_kg_hr=self._round_decimal(co_mass_rate, 4),
            combustion_efficiency_percent=self._round_decimal(combustion_efficiency, 2),
            unburned_carbon_percent=self._round_decimal(unburned_carbon, 4),
            carbon_in_ash_percent=self._round_decimal(carbon_in_ash, 4),
            chemical_efficiency_percent=self._round_decimal(chemical_efficiency, 2),
            is_incomplete=is_incomplete,
            severity=severity,
            root_cause=root_cause,
            provenance_hash=self._compute_hash(ic_data)
        )

    # =========================================================================
    # EFFICIENCY DEGRADATION TRENDING
    # =========================================================================

    def _calculate_efficiency_degradation(
        self,
        efficiency_history: List[float],
        timestamps_days: List[float],
        baseline_efficiency: float,
        current_efficiency: float
    ) -> EfficiencyDegradationTrend:
        """
        Calculate combustion efficiency degradation trend.

        Uses linear regression to determine degradation rate and
        forecast time to maintenance threshold.

        Args:
            efficiency_history: Historical efficiency readings
            timestamps_days: Days since baseline for readings
            baseline_efficiency: Baseline efficiency at commissioning
            current_efficiency: Current efficiency

        Returns:
            EfficiencyDegradationTrend with detailed analysis
        """
        degradation_percent = baseline_efficiency - current_efficiency

        # If no history, return basic metrics
        if len(efficiency_history) < 2:
            deg_data = {
                "current": current_efficiency,
                "baseline": baseline_efficiency,
                "degradation": degradation_percent
            }
            return EfficiencyDegradationTrend(
                current_efficiency_percent=current_efficiency,
                baseline_efficiency_percent=baseline_efficiency,
                degradation_percent=self._round_decimal(degradation_percent, 2),
                degradation_rate_per_day=0.0,
                days_to_maintenance_threshold=None,
                trend_confidence=0.0,
                trend_direction=TrendDirection.STABLE,
                forecast_30_day_efficiency=current_efficiency,
                provenance_hash=self._compute_hash(deg_data)
            )

        # Linear regression
        n = len(efficiency_history)
        if len(timestamps_days) != n:
            timestamps_days = list(range(n))

        x = timestamps_days
        y = efficiency_history

        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if abs(denominator) > 1e-10:
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean
        else:
            slope = 0.0
            intercept = y_mean

        # R-squared (trend confidence)
        ss_tot = sum((y[i] - y_mean) ** 2 for i in range(n))
        ss_res = sum((y[i] - (slope * x[i] + intercept)) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r_squared = max(0.0, min(1.0, r_squared))

        # Degradation rate per day
        degradation_rate = -slope  # Negative slope = degradation

        # Trend direction
        if abs(slope) < 0.01:
            trend_direction = TrendDirection.STABLE
        elif slope < 0:
            trend_direction = TrendDirection.DECREASING
        else:
            trend_direction = TrendDirection.INCREASING

        # Days to maintenance threshold
        maintenance_threshold = baseline_efficiency - self.EFFICIENCY_DEGRADATION_THRESHOLD
        if slope < 0 and current_efficiency > maintenance_threshold:
            days_to_threshold = int((current_efficiency - maintenance_threshold) / (-slope))
        else:
            days_to_threshold = None

        # 30-day forecast
        last_day = x[-1] if x else 0
        forecast_30_day = slope * (last_day + 30) + intercept

        deg_data = {
            "current": current_efficiency,
            "baseline": baseline_efficiency,
            "slope": slope,
            "r_squared": r_squared
        }

        return EfficiencyDegradationTrend(
            current_efficiency_percent=self._round_decimal(current_efficiency, 2),
            baseline_efficiency_percent=self._round_decimal(baseline_efficiency, 2),
            degradation_percent=self._round_decimal(degradation_percent, 2),
            degradation_rate_per_day=self._round_decimal(degradation_rate, 4),
            days_to_maintenance_threshold=days_to_threshold,
            trend_confidence=self._round_decimal(r_squared, 4),
            trend_direction=trend_direction,
            forecast_30_day_efficiency=self._round_decimal(forecast_30_day, 2),
            provenance_hash=self._compute_hash(deg_data)
        )

    # =========================================================================
    # BURNER HEALTH SCORING
    # =========================================================================

    def _calculate_burner_health(
        self,
        flame_pattern: FlamePatternMetrics,
        incomplete_combustion: IncompleteCombustionMetrics,
        efficiency_degradation: EfficiencyDegradationTrend,
        diagnostic_input: AdvancedDiagnosticInput
    ) -> BurnerHealthScore:
        """
        Calculate comprehensive burner health score (0-100).

        Scoring Components:
        - Flame quality score (25%): Pattern stability, symmetry
        - Emission quality score (25%): CO, NOx levels
        - Efficiency score (30%): Current vs baseline efficiency
        - Stability score (20%): Oscillation amplitude, frequency

        Args:
            flame_pattern: Flame pattern analysis results
            incomplete_combustion: CO analysis results
            efficiency_degradation: Efficiency trend results
            diagnostic_input: Original input parameters

        Returns:
            BurnerHealthScore with detailed component breakdown
        """
        # Flame quality score (0-100)
        flame_quality = flame_pattern.stability_index * 100
        if flame_pattern.pattern_type != FlamePattern.STABLE:
            flame_quality *= 0.8
        if flame_pattern.asymmetry_index > 0.3:
            flame_quality *= 0.9

        # Emission quality score (0-100)
        co_ppm = incomplete_combustion.co_concentration_ppm
        if co_ppm <= 50:
            emission_quality = 100.0
        elif co_ppm <= 100:
            emission_quality = 90 - (co_ppm - 50)
        elif co_ppm <= 200:
            emission_quality = 40 - (co_ppm - 100) / 5
        else:
            emission_quality = max(0, 20 - (co_ppm - 200) / 30)

        # Efficiency score (0-100)
        eff_ratio = (
            efficiency_degradation.current_efficiency_percent /
            efficiency_degradation.baseline_efficiency_percent
        )
        efficiency_score = min(100, eff_ratio * 100)

        # Stability score (0-100)
        stability_score = 100.0
        if flame_pattern.pulsation_amplitude_percent > 10:
            stability_score -= min(30, flame_pattern.pulsation_amplitude_percent)
        if flame_pattern.pulsation_frequency_hz > 2:
            stability_score -= min(20, flame_pattern.pulsation_frequency_hz * 5)
        stability_score = max(0, stability_score)

        # Fouling score (based on efficiency degradation)
        if efficiency_degradation.degradation_percent < 2:
            fouling_score = 100.0
        elif efficiency_degradation.degradation_percent < 5:
            fouling_score = 80.0
        elif efficiency_degradation.degradation_percent < 10:
            fouling_score = 50.0
        else:
            fouling_score = max(0, 50 - efficiency_degradation.degradation_percent * 2)

        # Overall weighted score
        overall_score = (
            self.HEALTH_WEIGHT_FLAME * flame_quality +
            self.HEALTH_WEIGHT_EMISSION * emission_quality +
            self.HEALTH_WEIGHT_EFFICIENCY * efficiency_score +
            self.HEALTH_WEIGHT_STABILITY * stability_score
        )

        # Classify category
        if overall_score >= 90:
            category = BurnerHealthCategory.EXCELLENT
        elif overall_score >= 75:
            category = BurnerHealthCategory.GOOD
        elif overall_score >= 60:
            category = BurnerHealthCategory.FAIR
        elif overall_score >= 40:
            category = BurnerHealthCategory.POOR
        else:
            category = BurnerHealthCategory.CRITICAL

        # Component scores tuple
        component_scores = (
            ("flame_quality", flame_quality),
            ("emission_quality", emission_quality),
            ("efficiency", efficiency_score),
            ("stability", stability_score),
            ("fouling", fouling_score),
        )

        # Estimated remaining life (simplified)
        if efficiency_degradation.degradation_rate_per_day > 0.1:
            remaining_life = int(
                (100 - overall_score) / efficiency_degradation.degradation_rate_per_day / 10
            )
        else:
            remaining_life = None

        health_data = {
            "overall": overall_score,
            "category": category.value,
            "components": dict(component_scores)
        }

        return BurnerHealthScore(
            overall_score=self._round_decimal(overall_score, 2),
            category=category,
            flame_quality_score=self._round_decimal(flame_quality, 2),
            emission_quality_score=self._round_decimal(emission_quality, 2),
            efficiency_score=self._round_decimal(efficiency_score, 2),
            stability_score=self._round_decimal(stability_score, 2),
            fouling_score=self._round_decimal(fouling_score, 2),
            component_scores=component_scores,
            degradation_from_baseline=self._round_decimal(
                efficiency_degradation.degradation_percent, 2
            ),
            estimated_remaining_life_days=remaining_life,
            provenance_hash=self._compute_hash(health_data)
        )

    # =========================================================================
    # FUEL QUALITY VARIATION DETECTION
    # =========================================================================

    def _assess_fuel_quality(
        self,
        heating_value_mj_kg: float,
        specific_gravity: float,
        reference_heating_value: float,
        reference_wobbe_index: float,
        hydrogen_content_percent: float
    ) -> FuelQualityVariation:
        """
        Detect fuel quality variations that affect combustion.

        Wobbe Index:
            W = HV / sqrt(SG)

        The Wobbe Index is the key interchangeability parameter
        for gaseous fuels.

        Args:
            heating_value_mj_kg: Measured heating value
            specific_gravity: Fuel specific gravity
            reference_heating_value: Reference/design heating value
            reference_wobbe_index: Reference Wobbe index
            hydrogen_content_percent: Hydrogen content for H2 blending detection

        Returns:
            FuelQualityVariation with detailed assessment
        """
        # Calculate Wobbe index
        wobbe_index = (
            heating_value_mj_kg / math.sqrt(specific_gravity)
            if specific_gravity > 0 else 0
        )

        # Calculate deviations
        hv_deviation = (
            abs(heating_value_mj_kg - reference_heating_value) / reference_heating_value
            if reference_heating_value > 0 else 0
        )
        wobbe_deviation = (
            abs(wobbe_index - reference_wobbe_index) / reference_wobbe_index
            if reference_wobbe_index > 0 else 0
        )

        # Detect hydrogen content variation (important for H2 blending)
        # Natural gas typically has ~23% H by mass, H2 blend increases this
        hydrogen_variation = hydrogen_content_percent > 25

        # Estimate methane number (simplified)
        # Higher H2 content reduces methane number
        methane_number = 100 - (hydrogen_content_percent - 23) * 2 if hydrogen_content_percent > 23 else 100

        # Quality score
        quality_score = 1.0 - (hv_deviation + wobbe_deviation) / 2
        quality_score = max(0.0, min(1.0, quality_score))

        # Significant variation flag
        is_significant = (
            hv_deviation > self.FUEL_QUALITY_DEVIATION_THRESHOLD or
            wobbe_deviation > 0.05
        )

        # Compensation factor for controls
        compensation_factor = reference_heating_value / heating_value_mj_kg if heating_value_mj_kg > 0 else 1.0

        fq_data = {
            "hv": heating_value_mj_kg,
            "wobbe": wobbe_index,
            "deviation": hv_deviation,
            "quality": quality_score
        }

        return FuelQualityVariation(
            heating_value_mj_kg=self._round_decimal(heating_value_mj_kg, 2),
            heating_value_deviation_percent=self._round_decimal(hv_deviation * 100, 2),
            wobbe_index=self._round_decimal(wobbe_index, 2),
            wobbe_deviation_percent=self._round_decimal(wobbe_deviation * 100, 2),
            specific_gravity=self._round_decimal(specific_gravity, 4),
            methane_number=self._round_decimal(methane_number, 1) if methane_number else None,
            hydrogen_content_variation=hydrogen_variation,
            quality_score=self._round_decimal(quality_score, 4),
            is_significant_variation=is_significant,
            compensation_factor=self._round_decimal(compensation_factor, 4),
            provenance_hash=self._compute_hash(fq_data)
        )

    # =========================================================================
    # MULTI-ZONE AIR DISTRIBUTION ANALYSIS
    # =========================================================================

    def _analyze_air_distribution(
        self,
        zone_data: List[ZoneInput],
        total_air_flow_kg_hr: float
    ) -> AirDistributionAnalysis:
        """
        Analyze multi-zone air distribution for combustion optimization.

        Proper air staging:
        - Primary air: 60-70% (for initial combustion)
        - Secondary air: 20-30% (for burnout)
        - Tertiary air: 5-10% (for NOx control)

        Args:
            zone_data: List of zone air flow data
            total_air_flow_kg_hr: Total combustion air flow

        Returns:
            AirDistributionAnalysis with detailed zone assessment
        """
        if not zone_data:
            # Default single-zone analysis
            air_data = {"total": total_air_flow_kg_hr, "zones": 0}
            return AirDistributionAnalysis(
                zones=(),
                overall_balance_score=100.0,
                total_air_flow_kg_hr=total_air_flow_kg_hr,
                primary_air_percent=100.0,
                secondary_air_percent=0.0,
                tertiary_air_percent=0.0,
                air_staging_active=False,
                distribution_uniformity_index=1.0,
                worst_zone_id=None,
                imbalance_severity=FaultSeverity.NONE,
                provenance_hash=self._compute_hash(air_data)
            )

        # Analyze each zone
        zones = []
        deviations = []
        worst_zone = None
        worst_deviation = 0.0

        for zone in zone_data:
            zone_percent = (
                zone.air_flow_kg_hr / total_air_flow_kg_hr * 100
                if total_air_flow_kg_hr > 0 else 0
            )
            target_percent = (
                zone.target_flow_kg_hr / total_air_flow_kg_hr * 100
                if total_air_flow_kg_hr > 0 else 0
            )
            deviation = zone_percent - target_percent

            is_starved = deviation < -10  # More than 10% below target
            is_over = deviation > 10  # More than 10% above target

            zones.append(ZoneAirDistribution(
                zone_id=zone.zone_id,
                air_flow_percent=self._round_decimal(zone_percent, 2),
                target_flow_percent=self._round_decimal(target_percent, 2),
                deviation_percent=self._round_decimal(deviation, 2),
                is_starved=is_starved,
                is_over_supplied=is_over,
                damper_position_percent=zone.damper_position_percent
            ))

            deviations.append(abs(deviation))
            if abs(deviation) > abs(worst_deviation):
                worst_deviation = deviation
                worst_zone = zone.zone_id

        # Calculate overall balance score
        max_deviation = max(deviations) if deviations else 0
        balance_score = max(0, 100 - max_deviation * 5)

        # Calculate uniformity index (0-1, 1 = perfect uniform)
        mean_flow = sum(z.air_flow_percent for z in zones) / len(zones) if zones else 0
        variance = (
            sum((z.air_flow_percent - mean_flow) ** 2 for z in zones) / len(zones)
            if zones else 0
        )
        uniformity = 1.0 / (1.0 + variance / 100)

        # Determine primary/secondary/tertiary split (simplified)
        primary = zones[0].air_flow_percent if len(zones) > 0 else 100
        secondary = zones[1].air_flow_percent if len(zones) > 1 else 0
        tertiary = zones[2].air_flow_percent if len(zones) > 2 else 0

        # Air staging detection
        air_staging = len(zones) >= 2 and secondary > 5

        # Severity classification
        if max_deviation > 30:
            severity = FaultSeverity.HIGH
        elif max_deviation > 20:
            severity = FaultSeverity.MEDIUM
        elif max_deviation > 10:
            severity = FaultSeverity.LOW
        else:
            severity = FaultSeverity.NONE

        air_data = {
            "zones": len(zones),
            "balance": balance_score,
            "uniformity": uniformity
        }

        return AirDistributionAnalysis(
            zones=tuple(zones),
            overall_balance_score=self._round_decimal(balance_score, 2),
            total_air_flow_kg_hr=self._round_decimal(total_air_flow_kg_hr, 2),
            primary_air_percent=self._round_decimal(primary, 2),
            secondary_air_percent=self._round_decimal(secondary, 2),
            tertiary_air_percent=self._round_decimal(tertiary, 2),
            air_staging_active=air_staging,
            distribution_uniformity_index=self._round_decimal(uniformity, 4),
            worst_zone_id=worst_zone,
            imbalance_severity=severity,
            provenance_hash=self._compute_hash(air_data)
        )

    # =========================================================================
    # SOOT FORMATION PREDICTION
    # =========================================================================

    def _predict_soot_formation(
        self,
        equivalence_ratio: float,
        temperature_c: float,
        fuel_type: str,
        hydrogen_content_percent: float
    ) -> SootFormationPrediction:
        """
        Predict soot formation based on combustion conditions.

        Soot formation is influenced by:
        - Equivalence ratio > 1 (rich conditions)
        - Temperature (peak soot at ~1500-1700K)
        - Fuel type (aromatics increase soot)
        - Hydrogen content (H2 reduces soot)

        Soot Formation Index (SFI):
            SFI = f(phi, T, fuel_type, H_content)

        Args:
            equivalence_ratio: Current equivalence ratio
            temperature_c: Combustion temperature
            fuel_type: Type of fuel
            hydrogen_content_percent: Hydrogen content

        Returns:
            SootFormationPrediction with risk assessment
        """
        temp_k = temperature_c + 273.15

        # Critical equivalence ratio for sooting (fuel dependent)
        critical_phi = {
            "natural_gas": 1.4,
            "methane": 1.4,
            "propane": 1.3,
            "diesel": 1.2,
            "fuel_oil": 1.15,
            "coal": 1.1,
        }
        critical_er = critical_phi.get(fuel_type.lower(), 1.3)

        # Margin to sooting
        margin = critical_er - equivalence_ratio

        # Soot formation index (0-100 scale)
        if equivalence_ratio <= 1.0:
            sfi = 0.0
        elif equivalence_ratio < critical_er:
            sfi = (equivalence_ratio - 1.0) / (critical_er - 1.0) * 50
        else:
            sfi = 50 + (equivalence_ratio - critical_er) * 100

        # Temperature factor (peak soot formation ~1600K)
        temp_factor = 1.0
        if 1500 < temp_k < 1800:
            temp_factor = 1.2
        elif temp_k < 1200:
            temp_factor = 0.5  # Too cool for significant soot

        sfi *= temp_factor

        # Hydrogen content reduces soot
        h2_factor = 1.0 - (hydrogen_content_percent - 20) * 0.02 if hydrogen_content_percent > 20 else 1.0
        sfi *= max(0.5, h2_factor)

        sfi = min(100, max(0, sfi))

        # Smoke number (Bacharach scale 0-9)
        smoke_number = sfi / 11  # Approximate mapping

        # Predicted soot rate (mg/Nm3) - empirical
        soot_rate = sfi * 0.5  # Simplified

        # Risk level
        if sfi > 70:
            risk_level = FaultSeverity.CRITICAL
        elif sfi > 50:
            risk_level = FaultSeverity.HIGH
        elif sfi > 30:
            risk_level = FaultSeverity.MEDIUM
        elif sfi > 10:
            risk_level = FaultSeverity.LOW
        else:
            risk_level = FaultSeverity.NONE

        is_sooting = sfi > 30

        # Contributing factors
        factors = []
        if equivalence_ratio > 1.1:
            factors.append("Rich combustion (phi > 1.1)")
        if 1500 < temp_k < 1800:
            factors.append("Temperature in peak soot formation range")
        if fuel_type.lower() in ["diesel", "fuel_oil", "coal"]:
            factors.append("High sooting tendency fuel")

        soot_data = {
            "sfi": sfi,
            "phi": equivalence_ratio,
            "critical_phi": critical_er
        }

        return SootFormationPrediction(
            soot_formation_index=self._round_decimal(sfi, 2),
            soot_risk_level=risk_level,
            predicted_soot_rate_mg_nm3=self._round_decimal(soot_rate, 2),
            smoke_number=self._round_decimal(smoke_number, 1),
            is_sooting=is_sooting,
            contributing_factors=tuple(factors),
            critical_equivalence_ratio=critical_er,
            current_equivalence_ratio=equivalence_ratio,
            margin_to_sooting=self._round_decimal(margin, 4),
            provenance_hash=self._compute_hash(soot_data)
        )

    # =========================================================================
    # FLASHBACK AND BLOWOFF RISK ASSESSMENT
    # =========================================================================

    def _assess_flashback_blowoff_risk(
        self,
        burner_velocity_m_s: float,
        fuel_type: str,
        equivalence_ratio: float,
        temperature_c: float,
        burner_diameter_mm: float
    ) -> FlashbackBlowoffRisk:
        """
        Assess flashback and blowoff (flame stability) risk.

        Flashback occurs when:
            Flow velocity < Flame speed (flame propagates upstream)

        Blowoff occurs when:
            Flow velocity >> Flame speed (flame is blown off)

        Damkohler Number:
            Da = tau_flow / tau_chem
            Da > 1: Stable combustion
            Da << 1: Blowoff risk
            Da >> 1 with low velocity: Flashback risk

        Args:
            burner_velocity_m_s: Flow velocity at burner
            fuel_type: Type of fuel
            equivalence_ratio: Equivalence ratio
            temperature_c: Combustion temperature
            burner_diameter_mm: Burner diameter

        Returns:
            FlashbackBlowoffRisk with stability assessment
        """
        # Laminar flame speed (adjusted for temperature and phi)
        base_flame_speed = LAMINAR_FLAME_SPEED.get(fuel_type.lower(), 0.4)

        # Flame speed peaks near stoichiometric
        phi_factor = 1.0 - abs(equivalence_ratio - 1.0) * 0.5
        phi_factor = max(0.3, phi_factor)

        # Temperature correction (flame speed increases with preheat)
        temp_k = temperature_c + 273.15
        temp_factor = (temp_k / 300) ** 1.5  # Simplified

        flame_speed = base_flame_speed * phi_factor * temp_factor

        # Turbulent flame speed (typically 5-10x laminar)
        turbulent_factor = 5.0
        turbulent_flame_speed = flame_speed * turbulent_factor

        # Critical velocities
        critical_flashback_velocity = flame_speed * 2  # Safety factor
        critical_blowoff_velocity = turbulent_flame_speed * 3

        # Damkohler number (simplified)
        residence_time = burner_diameter_mm / 1000 / burner_velocity_m_s if burner_velocity_m_s > 0 else 1
        chemical_time = 0.001  # Approximate for hydrocarbon fuels
        damkohler = residence_time / chemical_time

        # Flashback risk score (0-100)
        if burner_velocity_m_s <= critical_flashback_velocity:
            flashback_risk = min(100, (1 - burner_velocity_m_s / critical_flashback_velocity) * 100 + 50)
        else:
            flashback_risk = max(0, 50 - (burner_velocity_m_s - critical_flashback_velocity) * 10)

        # Blowoff risk score (0-100)
        if burner_velocity_m_s >= critical_blowoff_velocity:
            blowoff_risk = min(100, (burner_velocity_m_s / critical_blowoff_velocity - 1) * 100 + 50)
        else:
            blowoff_risk = max(0, 50 - (critical_blowoff_velocity - burner_velocity_m_s) * 2)

        # Severity classification
        flashback_severity = self._classify_risk_severity(flashback_risk)
        blowoff_severity = self._classify_risk_severity(blowoff_risk)

        # Stability margin
        stability_margin = min(
            (burner_velocity_m_s - critical_flashback_velocity) / burner_velocity_m_s * 100,
            (critical_blowoff_velocity - burner_velocity_m_s) / critical_blowoff_velocity * 100
        ) if burner_velocity_m_s > 0 else 0

        # Operating regime
        if flashback_risk > 50:
            regime = "near_flashback"
        elif blowoff_risk > 50:
            regime = "near_blowoff"
        else:
            regime = "stable"

        fb_data = {
            "flashback": flashback_risk,
            "blowoff": blowoff_risk,
            "damkohler": damkohler,
            "regime": regime
        }

        return FlashbackBlowoffRisk(
            flashback_risk_score=self._round_decimal(flashback_risk, 2),
            blowoff_risk_score=self._round_decimal(blowoff_risk, 2),
            flashback_severity=flashback_severity,
            blowoff_severity=blowoff_severity,
            current_flame_velocity_m_s=self._round_decimal(burner_velocity_m_s, 2),
            critical_flashback_velocity_m_s=self._round_decimal(critical_flashback_velocity, 2),
            critical_blowoff_velocity_m_s=self._round_decimal(critical_blowoff_velocity, 2),
            damkohler_number=self._round_decimal(damkohler, 2),
            stability_margin_percent=self._round_decimal(max(0, stability_margin), 2),
            operating_regime=regime,
            provenance_hash=self._compute_hash(fb_data)
        )

    def _classify_risk_severity(self, risk_score: float) -> FaultSeverity:
        """Classify risk score into severity level"""
        if risk_score > 80:
            return FaultSeverity.CRITICAL
        elif risk_score > 60:
            return FaultSeverity.HIGH
        elif risk_score > 40:
            return FaultSeverity.MEDIUM
        elif risk_score > 20:
            return FaultSeverity.LOW
        else:
            return FaultSeverity.NONE

    # =========================================================================
    # MAINTENANCE RECOMMENDATIONS
    # =========================================================================

    def _generate_maintenance_recommendations(
        self,
        faults: List[FaultDetectionResult],
        burner_health: BurnerHealthScore,
        efficiency_degradation: EfficiencyDegradationTrend,
        sensor_drift: List[SensorDriftCompensation],
        soot_prediction: SootFormationPrediction,
        flashback_blowoff: FlashbackBlowoffRisk
    ) -> List[MaintenanceRecommendation]:
        """
        Generate prioritized maintenance recommendations.

        Priority Levels:
        - IMMEDIATE: Safety-critical, within 24 hours
        - URGENT: Within 1 week
        - SCHEDULED: Within maintenance window
        - ROUTINE: Next scheduled maintenance
        - INFORMATIONAL: For awareness

        Args:
            faults: Detected faults
            burner_health: Burner health assessment
            efficiency_degradation: Efficiency trend
            sensor_drift: Sensor drift status
            soot_prediction: Soot formation prediction
            flashback_blowoff: Stability risk assessment

        Returns:
            List of prioritized maintenance recommendations
        """
        recommendations = []

        # Safety-critical recommendations
        if flashback_blowoff.flashback_severity == FaultSeverity.CRITICAL:
            recommendations.append(self._create_recommendation(
                priority=MaintenancePriority.IMMEDIATE,
                category="Safety",
                description="Critical flashback risk detected",
                detailed_action="Reduce fuel flow immediately. Inspect burner tip for damage. "
                               "Check fuel supply pressure and composition.",
                estimated_impact="Prevent burner damage and potential fire hazard",
                downtime_hours=2.0,
                cost_category="high",
                deadline_days=0,
                related_faults=(FaultType.FLASHBACK_RISK,)
            ))

        if flashback_blowoff.blowoff_severity == FaultSeverity.CRITICAL:
            recommendations.append(self._create_recommendation(
                priority=MaintenancePriority.IMMEDIATE,
                category="Safety",
                description="Critical blowoff risk detected",
                detailed_action="Reduce air flow. Check flame detector status. "
                               "Verify ignition system ready for relight.",
                estimated_impact="Prevent flame loss and potential explosion hazard",
                downtime_hours=1.0,
                cost_category="medium",
                deadline_days=0,
                related_faults=(FaultType.BLOWOFF_RISK,)
            ))

        # Soot formation recommendations
        if soot_prediction.is_sooting:
            recommendations.append(self._create_recommendation(
                priority=MaintenancePriority.URGENT,
                category="Combustion Quality",
                description="Soot formation detected - heat exchanger fouling risk",
                detailed_action="Increase excess air by 5-10%. Check fuel quality. "
                               "Inspect and clean burner nozzles. Schedule tube cleaning.",
                estimated_impact="Prevent efficiency loss and tube damage",
                downtime_hours=4.0,
                cost_category="medium",
                deadline_days=7,
                related_faults=(FaultType.SOOT_FORMATION, FaultType.INCOMPLETE_COMBUSTION)
            ))

        # Efficiency degradation recommendations
        if efficiency_degradation.degradation_percent > 5:
            recommendations.append(self._create_recommendation(
                priority=MaintenancePriority.SCHEDULED,
                category="Performance",
                description=f"Efficiency degraded {efficiency_degradation.degradation_percent:.1f}% from baseline",
                detailed_action="Perform combustion tuning. Clean heat transfer surfaces. "
                               "Check air/fuel ratio controls. Inspect for air leaks.",
                estimated_impact=f"Potential fuel savings of {efficiency_degradation.degradation_percent * 0.5:.1f}%",
                downtime_hours=8.0,
                cost_category="medium",
                deadline_days=efficiency_degradation.days_to_maintenance_threshold or 30,
                related_faults=(FaultType.BURNER_FOULING,)
            ))

        # Sensor calibration recommendations
        for drift in sensor_drift:
            if drift.calibration_recommended:
                recommendations.append(self._create_recommendation(
                    priority=MaintenancePriority.SCHEDULED,
                    category="Instrumentation",
                    description=f"{drift.sensor_type.value} calibration required",
                    detailed_action=f"Calibrate {drift.sensor_id}. Current drift: {drift.drift_amount:.4f}. "
                                   f"Apply compensation factor: {drift.compensation_factor:.4f}",
                    estimated_impact="Maintain accurate combustion control",
                    downtime_hours=1.0,
                    cost_category="low",
                    deadline_days=14,
                    related_faults=(FaultType.SENSOR_DRIFT,)
                ))

        # Burner health recommendations
        if burner_health.category == BurnerHealthCategory.POOR:
            recommendations.append(self._create_recommendation(
                priority=MaintenancePriority.URGENT,
                category="Equipment",
                description=f"Burner health score: {burner_health.overall_score:.1f} (Poor)",
                detailed_action="Comprehensive burner inspection required. Check nozzle wear, "
                               "flame pattern, and air register settings.",
                estimated_impact="Prevent further degradation and potential failure",
                downtime_hours=6.0,
                cost_category="high",
                deadline_days=7,
                related_faults=(FaultType.BURNER_FOULING, FaultType.FLAME_INSTABILITY)
            ))
        elif burner_health.category == BurnerHealthCategory.CRITICAL:
            recommendations.append(self._create_recommendation(
                priority=MaintenancePriority.IMMEDIATE,
                category="Equipment",
                description=f"CRITICAL: Burner health score: {burner_health.overall_score:.1f}",
                detailed_action="Emergency burner service required. Consider backup burner "
                               "or load reduction until repair completed.",
                estimated_impact="Prevent burner failure and unplanned outage",
                downtime_hours=12.0,
                cost_category="high",
                deadline_days=1,
                related_faults=(FaultType.BURNER_FOULING, FaultType.FLAME_INSTABILITY)
            ))

        # Fault-specific recommendations
        for fault in faults:
            if fault.severity in (FaultSeverity.HIGH, FaultSeverity.CRITICAL):
                recommendations.append(self._create_recommendation(
                    priority=MaintenancePriority.URGENT if fault.severity == FaultSeverity.HIGH else MaintenancePriority.IMMEDIATE,
                    category="Fault Response",
                    description=fault.description,
                    detailed_action=fault.recommended_action,
                    estimated_impact="Address detected fault condition",
                    downtime_hours=2.0,
                    cost_category="medium",
                    deadline_days=7 if fault.severity == FaultSeverity.HIGH else 1,
                    related_faults=(fault.fault_type,)
                ))

        # If no issues, add informational recommendation
        if not recommendations:
            recommendations.append(self._create_recommendation(
                priority=MaintenancePriority.INFORMATIONAL,
                category="Status",
                description="System operating within normal parameters",
                detailed_action="Continue routine monitoring and scheduled maintenance",
                estimated_impact="N/A",
                downtime_hours=0.0,
                cost_category="low",
                deadline_days=None,
                related_faults=()
            ))

        # Sort by priority
        priority_order = {
            MaintenancePriority.IMMEDIATE: 0,
            MaintenancePriority.URGENT: 1,
            MaintenancePriority.SCHEDULED: 2,
            MaintenancePriority.ROUTINE: 3,
            MaintenancePriority.INFORMATIONAL: 4
        }
        recommendations.sort(key=lambda r: priority_order.get(r.priority, 5))

        return recommendations

    def _create_recommendation(
        self,
        priority: MaintenancePriority,
        category: str,
        description: str,
        detailed_action: str,
        estimated_impact: str,
        downtime_hours: float,
        cost_category: str,
        deadline_days: Optional[int],
        related_faults: Tuple[FaultType, ...]
    ) -> MaintenanceRecommendation:
        """Create a maintenance recommendation with provenance"""
        self._recommendation_counter += 1
        rec_id = f"REC-{self._recommendation_counter:04d}"

        rec_data = {
            "id": rec_id,
            "priority": priority.value,
            "category": category,
            "description": description[:50]
        }

        return MaintenanceRecommendation(
            recommendation_id=rec_id,
            priority=priority,
            category=category,
            description=description,
            detailed_action=detailed_action,
            estimated_impact=estimated_impact,
            estimated_downtime_hours=downtime_hours,
            cost_category=cost_category,
            deadline_days=deadline_days,
            related_faults=related_faults,
            provenance_hash=self._compute_hash(rec_data)
        )

    # =========================================================================
    # FAULT DETECTION
    # =========================================================================

    def _detect_all_faults(
        self,
        diagnostic_input: AdvancedDiagnosticInput,
        flame_pattern: FlamePatternMetrics,
        incomplete_combustion: IncompleteCombustionMetrics,
        fuel_quality: FuelQualityVariation,
        air_distribution: AirDistributionAnalysis,
        soot_prediction: SootFormationPrediction,
        flashback_blowoff: FlashbackBlowoffRisk
    ) -> List[FaultDetectionResult]:
        """Detect all applicable faults based on analysis results"""
        faults = []
        now = datetime.now(timezone.utc)

        # Flame instability
        if flame_pattern.stability_index < 0.7:
            severity = FaultSeverity.HIGH if flame_pattern.stability_index < 0.5 else FaultSeverity.MEDIUM
            faults.append(FaultDetectionResult(
                fault_type=FaultType.FLAME_INSTABILITY,
                severity=severity,
                confidence=1.0 - flame_pattern.stability_index,
                detected_at=now,
                description=f"Flame stability index {flame_pattern.stability_index:.2f} below threshold",
                recommended_action="Check burner alignment, fuel/air ratio, and ignition system",
                affected_parameters=("flame_intensity", "combustion_stability"),
                provenance_hash=self._compute_hash({"fault": "flame_instability", "index": flame_pattern.stability_index})
            ))

        # Incomplete combustion
        if incomplete_combustion.is_incomplete:
            faults.append(FaultDetectionResult(
                fault_type=FaultType.INCOMPLETE_COMBUSTION,
                severity=incomplete_combustion.severity,
                confidence=min(1.0, incomplete_combustion.co_concentration_ppm / 500),
                detected_at=now,
                description=f"High CO: {incomplete_combustion.co_concentration_ppm:.0f} ppm. {incomplete_combustion.root_cause}",
                recommended_action="Increase excess air, check burner pattern, inspect for flame impingement",
                affected_parameters=("co_actual", "combustion_efficiency"),
                provenance_hash=self._compute_hash({"fault": "incomplete_combustion", "co": incomplete_combustion.co_concentration_ppm})
            ))

        # Fuel quality variation
        if fuel_quality.is_significant_variation:
            faults.append(FaultDetectionResult(
                fault_type=FaultType.FUEL_QUALITY_DEGRADATION,
                severity=FaultSeverity.MEDIUM,
                confidence=fuel_quality.heating_value_deviation_percent / 20,
                detected_at=now,
                description=f"Fuel heating value deviation {fuel_quality.heating_value_deviation_percent:.1f}%",
                recommended_action="Verify fuel supply, adjust combustion controls for new fuel quality",
                affected_parameters=("fuel_heating_value", "combustion_efficiency"),
                provenance_hash=self._compute_hash({"fault": "fuel_quality", "deviation": fuel_quality.heating_value_deviation_percent})
            ))

        # Air distribution imbalance
        if air_distribution.imbalance_severity in (FaultSeverity.MEDIUM, FaultSeverity.HIGH):
            faults.append(FaultDetectionResult(
                fault_type=FaultType.AIR_DISTRIBUTION_IMBALANCE,
                severity=air_distribution.imbalance_severity,
                confidence=0.8,
                detected_at=now,
                description=f"Air distribution imbalance in zone {air_distribution.worst_zone_id}",
                recommended_action="Check damper positions, inspect for blockages, rebalance air flow",
                affected_parameters=("air_flow", "zone_temperatures"),
                provenance_hash=self._compute_hash({"fault": "air_imbalance", "zone": air_distribution.worst_zone_id})
            ))

        # Soot formation
        if soot_prediction.is_sooting:
            faults.append(FaultDetectionResult(
                fault_type=FaultType.SOOT_FORMATION,
                severity=soot_prediction.soot_risk_level,
                confidence=soot_prediction.soot_formation_index / 100,
                detected_at=now,
                description=f"Soot formation index {soot_prediction.soot_formation_index:.1f}",
                recommended_action="Increase excess air, reduce equivalence ratio below sooting limit",
                affected_parameters=("equivalence_ratio", "co_actual", "smoke"),
                provenance_hash=self._compute_hash({"fault": "soot", "sfi": soot_prediction.soot_formation_index})
            ))

        # Flashback risk
        if flashback_blowoff.flashback_severity in (FaultSeverity.HIGH, FaultSeverity.CRITICAL):
            faults.append(FaultDetectionResult(
                fault_type=FaultType.FLASHBACK_RISK,
                severity=flashback_blowoff.flashback_severity,
                confidence=flashback_blowoff.flashback_risk_score / 100,
                detected_at=now,
                description=f"Flashback risk score {flashback_blowoff.flashback_risk_score:.1f}%",
                recommended_action="Increase flow velocity, check for burner damage, verify fuel composition",
                affected_parameters=("burner_velocity", "flame_position"),
                provenance_hash=self._compute_hash({"fault": "flashback", "risk": flashback_blowoff.flashback_risk_score})
            ))

        # Blowoff risk
        if flashback_blowoff.blowoff_severity in (FaultSeverity.HIGH, FaultSeverity.CRITICAL):
            faults.append(FaultDetectionResult(
                fault_type=FaultType.BLOWOFF_RISK,
                severity=flashback_blowoff.blowoff_severity,
                confidence=flashback_blowoff.blowoff_risk_score / 100,
                detected_at=now,
                description=f"Blowoff risk score {flashback_blowoff.blowoff_risk_score:.1f}%",
                recommended_action="Reduce flow velocity, check flame detector, verify ignition ready",
                affected_parameters=("burner_velocity", "flame_status"),
                provenance_hash=self._compute_hash({"fault": "blowoff", "risk": flashback_blowoff.blowoff_risk_score})
            ))

        # Air-fuel imbalance (from O2 readings)
        o2 = diagnostic_input.o2_actual_percent
        if o2 < self.O2_LOW_THRESHOLD:
            faults.append(FaultDetectionResult(
                fault_type=FaultType.AIR_FUEL_IMBALANCE,
                severity=FaultSeverity.HIGH,
                confidence=1.0 - o2 / self.O2_LOW_THRESHOLD,
                detected_at=now,
                description=f"Fuel-rich condition: O2 at {o2:.1f}%",
                recommended_action="Increase combustion air or reduce fuel",
                affected_parameters=("o2_actual", "fuel_flow", "air_flow"),
                provenance_hash=self._compute_hash({"fault": "rich", "o2": o2})
            ))
        elif o2 > self.O2_HIGH_THRESHOLD:
            faults.append(FaultDetectionResult(
                fault_type=FaultType.AIR_FUEL_IMBALANCE,
                severity=FaultSeverity.LOW,
                confidence=(o2 - self.O2_HIGH_THRESHOLD) / 5.0,
                detected_at=now,
                description=f"Fuel-lean condition: O2 at {o2:.1f}%",
                recommended_action="Reduce excess air for better efficiency",
                affected_parameters=("o2_actual", "air_flow"),
                provenance_hash=self._compute_hash({"fault": "lean", "o2": o2})
            ))

        return faults

    # =========================================================================
    # SUPPORTING CALCULATIONS
    # =========================================================================

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

        # Combustion noise estimate
        noise_db = 20 * math.log10(pressure_amplitude + 1)

        # Instability score
        instability_score = (
            0.4 * min(pressure_amplitude / 1000, 1.0) +
            0.3 * min(math.sqrt(temp_variance) / 100, 1.0) +
            0.3 * min(flame_flicker / 0.5, 1.0)
        )
        instability_score = min(1.0, max(0.0, instability_score))

        # Thermoacoustic detection
        is_thermoacoustic = (
            self.THERMOACOUSTIC_FREQUENCY_MIN_HZ <= pressure_freq <= self.THERMOACOUSTIC_FREQUENCY_MAX_HZ and
            pressure_amplitude > self.PRESSURE_OSCILLATION_THRESHOLD_PA
        )

        inst_data = {
            "pressure_amp": pressure_amplitude,
            "pressure_freq": pressure_freq,
            "score": instability_score
        }

        return CombustionInstabilityIndicators(
            pressure_oscillation_amplitude_pa=self._round_decimal(pressure_amplitude, 2),
            pressure_oscillation_frequency_hz=self._round_decimal(pressure_freq, 2),
            temperature_variance_c=self._round_decimal(temp_variance, 4),
            flame_flicker_index=self._round_decimal(flame_flicker, 4),
            combustion_noise_db=self._round_decimal(noise_db, 2),
            instability_score=self._round_decimal(instability_score, 4),
            is_thermoacoustic=is_thermoacoustic,
            provenance_hash=self._compute_hash(inst_data)
        )

    def _calculate_sensor_drift(
        self,
        diagnostic_input: AdvancedDiagnosticInput
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

                drift_amount = current_value - baseline
                drift_rate = drift_amount / time_since_cal if time_since_cal > 0 else 0

                if baseline != 0:
                    drift_percent = abs(drift_amount) / abs(baseline)
                    compensation_factor = baseline / current_value if current_value != 0 else 1.0
                else:
                    drift_percent = 0
                    compensation_factor = 1.0

                calibration_recommended = drift_percent > self.SENSOR_DRIFT_THRESHOLD

                drift_data = {
                    "sensor": sensor_id,
                    "drift": drift_amount,
                    "rate": drift_rate
                }

                results.append(SensorDriftCompensation(
                    sensor_type=sensor_type,
                    sensor_id=sensor_id,
                    baseline_value=baseline,
                    current_value=current_value,
                    drift_amount=self._round_decimal(drift_amount, 4),
                    drift_rate_per_hour=self._round_decimal(drift_rate, 6),
                    compensation_factor=self._round_decimal(compensation_factor, 4),
                    calibration_recommended=calibration_recommended,
                    time_since_calibration_hours=time_since_cal,
                    provenance_hash=self._compute_hash(drift_data)
                ))

        return results

    def _validate_cross_limiting(
        self,
        diagnostic_input: AdvancedDiagnosticInput
    ) -> CrossLimitParameters:
        """Validate cross-limiting from input"""
        is_fuel_limited = (
            diagnostic_input.fuel_actual_percent <
            diagnostic_input.fuel_demand_percent - self.CROSS_LIMIT_MARGIN_PERCENT
        )
        is_air_limited = (
            diagnostic_input.air_actual_percent <
            diagnostic_input.air_demand_percent - self.CROSS_LIMIT_MARGIN_PERCENT
        )
        cross_limit_active = is_fuel_limited or is_air_limited

        cl_data = {
            "fuel_demand": diagnostic_input.fuel_demand_percent,
            "air_demand": diagnostic_input.air_demand_percent,
            "active": cross_limit_active
        }

        return CrossLimitParameters(
            fuel_demand_percent=diagnostic_input.fuel_demand_percent,
            air_demand_percent=diagnostic_input.air_demand_percent,
            fuel_actual_percent=diagnostic_input.fuel_actual_percent,
            air_actual_percent=diagnostic_input.air_actual_percent,
            fuel_lead_lag_seconds=diagnostic_input.fuel_lead_lag_seconds,
            air_lead_lag_seconds=diagnostic_input.air_lead_lag_seconds,
            is_fuel_limited=is_fuel_limited,
            is_air_limited=is_air_limited,
            cross_limit_active=cross_limit_active,
            provenance_hash=self._compute_hash(cl_data)
        )

    def _calculate_trim_parameters(
        self,
        diagnostic_input: AdvancedDiagnosticInput
    ) -> TrimControlParameters:
        """Calculate trim parameters from input"""
        combined_trim = (
            diagnostic_input.o2_trim_output_percent +
            diagnostic_input.co_trim_output_percent
        )

        is_saturated_high = combined_trim >= diagnostic_input.trim_high_limit_percent
        is_saturated_low = combined_trim <= diagnostic_input.trim_low_limit_percent

        trim_data = {
            "o2_setpoint": diagnostic_input.o2_setpoint_percent,
            "combined_trim": combined_trim
        }

        return TrimControlParameters(
            o2_setpoint_percent=diagnostic_input.o2_setpoint_percent,
            o2_actual_percent=diagnostic_input.o2_actual_percent,
            o2_trim_output_percent=diagnostic_input.o2_trim_output_percent,
            co_trim_output_percent=diagnostic_input.co_trim_output_percent,
            combined_trim_percent=self._round_decimal(combined_trim, 2),
            trim_rate_limit_per_minute=diagnostic_input.trim_rate_limit_per_minute,
            is_saturated_high=is_saturated_high,
            is_saturated_low=is_saturated_low,
            provenance_hash=self._compute_hash(trim_data)
        )

    def _analyze_trends(
        self,
        diagnostic_input: AdvancedDiagnosticInput
    ) -> List[DiagnosticTrend]:
        """Analyze trends from historical data"""
        trends = []

        if diagnostic_input.historical_o2_readings and diagnostic_input.historical_timestamps_minutes:
            o2_trend = self._analyze_single_trend(
                "o2_percent",
                diagnostic_input.historical_o2_readings,
                diagnostic_input.historical_timestamps_minutes,
                self.O2_HIGH_THRESHOLD
            )
            trends.append(o2_trend)

        if diagnostic_input.historical_co_readings and diagnostic_input.historical_timestamps_minutes:
            co_trend = self._analyze_single_trend(
                "co_ppm",
                diagnostic_input.historical_co_readings,
                diagnostic_input.historical_timestamps_minutes,
                self.CO_HIGH_THRESHOLD_PPM
            )
            trends.append(co_trend)

        return trends

    def _analyze_single_trend(
        self,
        parameter_name: str,
        values: List[float],
        timestamps: List[float],
        alert_threshold: Optional[float] = None,
        forecast_horizon: int = 60
    ) -> DiagnosticTrend:
        """Analyze trend for a single parameter"""
        if len(values) < 2 or len(values) != len(timestamps):
            trend_hash = self._compute_hash({"param": parameter_name, "n": len(values)})
            return DiagnosticTrend(
                parameter_name=parameter_name,
                direction=TrendDirection.STABLE,
                slope=0.0,
                r_squared=0.0,
                forecast_value=values[-1] if values else 0.0,
                forecast_horizon_minutes=forecast_horizon,
                alert_threshold_eta_minutes=None,
                provenance_hash=trend_hash
            )

        n = len(values)
        x_mean = sum(timestamps) / n
        y_mean = sum(values) / n

        numerator = sum((timestamps[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((timestamps[i] - x_mean) ** 2 for i in range(n))

        if abs(denominator) < 1e-10:
            slope = 0.0
            intercept = y_mean
        else:
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean

        # R-squared
        ss_tot = sum((values[i] - y_mean) ** 2 for i in range(n))
        ss_res = sum((values[i] - (slope * timestamps[i] + intercept)) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r_squared = max(0.0, min(1.0, r_squared))

        # Direction
        if abs(slope) < 1e-6:
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING

        if r_squared < 0.3 and abs(slope) < 0.1:
            direction = TrendDirection.OSCILLATING

        # Forecast
        last_time = timestamps[-1]
        forecast_value = slope * (last_time + forecast_horizon) + intercept

        # Alert ETA
        alert_eta = None
        if alert_threshold is not None and abs(slope) > 1e-6:
            time_to_threshold = (alert_threshold - values[-1]) / slope
            if time_to_threshold > 0:
                alert_eta = int(time_to_threshold)

        trend_hash = self._compute_hash({
            "param": parameter_name,
            "slope": slope,
            "r2": r_squared
        })

        return DiagnosticTrend(
            parameter_name=parameter_name,
            direction=direction,
            slope=self._round_decimal(slope, 6),
            r_squared=self._round_decimal(r_squared, 4),
            forecast_value=self._round_decimal(forecast_value, 4),
            forecast_horizon_minutes=forecast_horizon,
            alert_threshold_eta_minutes=alert_eta,
            provenance_hash=trend_hash
        )

    def _calculate_overall_health(
        self,
        burner_health: BurnerHealthScore,
        faults: List[FaultDetectionResult],
        instability: CombustionInstabilityIndicators,
        air_distribution: AirDistributionAnalysis
    ) -> float:
        """Calculate overall system health score (0-100)"""
        # Start with burner health
        score = burner_health.overall_score

        # Deduct for faults
        for fault in faults:
            if fault.severity == FaultSeverity.CRITICAL:
                score -= 20
            elif fault.severity == FaultSeverity.HIGH:
                score -= 10
            elif fault.severity == FaultSeverity.MEDIUM:
                score -= 5
            elif fault.severity == FaultSeverity.LOW:
                score -= 2

        # Factor in instability
        score *= (1.0 - instability.instability_score * 0.3)

        # Factor in air distribution
        score *= (air_distribution.overall_balance_score / 100) ** 0.5

        return self._round_decimal(max(0.0, min(100.0, score)), 2)

    def _requires_immediate_action(
        self,
        faults: List[FaultDetectionResult],
        flashback_blowoff: FlashbackBlowoffRisk,
        soot_prediction: SootFormationPrediction,
        overall_health: float
    ) -> bool:
        """Determine if immediate action is required"""
        # Critical faults
        for fault in faults:
            if fault.severity == FaultSeverity.CRITICAL:
                return True

        # Critical stability risks
        if flashback_blowoff.flashback_severity == FaultSeverity.CRITICAL:
            return True
        if flashback_blowoff.blowoff_severity == FaultSeverity.CRITICAL:
            return True

        # Critical soot formation
        if soot_prediction.soot_risk_level == FaultSeverity.CRITICAL:
            return True

        # Very low health score
        if overall_health < 30:
            return True

        return False

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

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


# =============================================================================
# BACKWARD COMPATIBILITY - EXPOSE ORIGINAL CLASSES
# =============================================================================

# Alias for backward compatibility
CombustionDiagnosticsCalculator = AdvancedCombustionDiagnosticsCalculator
DiagnosticInput = AdvancedDiagnosticInput
DiagnosticOutput = AdvancedDiagnosticOutput
DiagnosticSummary = AdvancedDiagnosticSummary
