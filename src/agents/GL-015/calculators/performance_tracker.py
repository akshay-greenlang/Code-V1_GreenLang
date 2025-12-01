# -*- coding: utf-8 -*-
"""
Insulation Performance Tracker Module for GL-015 INSULSCAN

Historical tracking and trend analysis for insulation performance monitoring.
Implements degradation rate analysis, remaining useful life estimation,
fleet benchmarking, and predictive analytics.

Author: GL-CalculatorEngineer
Agent: GL-015 INSULSCAN
Version: 1.0.0
Standards: ASTM C1055, ISO 12241, CINI Manual, ASHRAE Guidelines

Zero-Hallucination Guarantee:
    - All calculations use deterministic engineering formulas
    - No LLM inference for numeric values
    - Bit-perfect reproducibility with SHA-256 provenance
    - Complete audit trails for regulatory compliance
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


# =============================================================================
# ENUMERATIONS
# =============================================================================

class DegradationModel(Enum):
    """Degradation model types for R-value decay analysis."""
    LINEAR = "linear"                    # R(t) = R_0 - k*t
    EXPONENTIAL = "exponential"          # R(t) = R_0 * exp(-lambda*t)
    WEIBULL = "weibull"                  # Reliability-based degradation
    LOGARITHMIC = "logarithmic"          # R(t) = R_0 - k*ln(t+1)
    POWER_LAW = "power_law"              # R(t) = R_0 * t^(-alpha)


class ConditionCategory(Enum):
    """Insulation condition categories per ASTM C1055."""
    EXCELLENT = "excellent"              # >90% of design R-value
    GOOD = "good"                        # 75-90% of design R-value
    FAIR = "fair"                        # 50-75% of design R-value
    POOR = "poor"                        # 25-50% of design R-value
    FAILED = "failed"                    # <25% of design R-value


class AlertSeverity(Enum):
    """Alert severity levels for condition changes."""
    INFO = "info"                        # Normal degradation
    WARNING = "warning"                  # Accelerated degradation
    CRITICAL = "critical"                # Immediate attention needed
    EMERGENCY = "emergency"              # Safety concern


class TrendDirection(Enum):
    """Trend direction for performance metrics."""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    ACCELERATING_DEGRADATION = "accelerating_degradation"


class InspectionType(Enum):
    """Types of insulation inspections."""
    THERMAL_IMAGING = "thermal_imaging"
    VISUAL = "visual"
    THICKNESS_MEASUREMENT = "thickness_measurement"
    COMPREHENSIVE = "comprehensive"
    SPOT_CHECK = "spot_check"


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class PerformanceDataPoint:
    """Single performance measurement data point."""
    timestamp: datetime
    surface_temperature_c: float
    ambient_temperature_c: float
    process_temperature_c: float
    r_value_m2k_w: float
    heat_loss_rate_w: float
    efficiency_percent: float
    inspector_id: Optional[str] = None
    inspection_type: InspectionType = InspectionType.THERMAL_IMAGING
    notes: Optional[str] = None
    data_quality_score: float = 1.0  # 0-1 quality indicator

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "surface_temperature_c": self.surface_temperature_c,
            "ambient_temperature_c": self.ambient_temperature_c,
            "process_temperature_c": self.process_temperature_c,
            "r_value_m2k_w": self.r_value_m2k_w,
            "heat_loss_rate_w": self.heat_loss_rate_w,
            "efficiency_percent": self.efficiency_percent,
            "inspection_type": self.inspection_type.value,
            "data_quality_score": self.data_quality_score
        }


@dataclass
class DegradationAnalysis:
    """Results of degradation rate analysis."""
    model_type: DegradationModel
    initial_r_value: float              # R_0
    degradation_rate: float             # k or lambda
    r_squared: float                    # Goodness of fit
    current_r_value: float              # Current estimated R-value
    predicted_r_value_1y: float         # 1 year prediction
    predicted_r_value_5y: float         # 5 year prediction
    annual_degradation_percent: float   # Annual % decrease
    is_accelerating: bool               # Acceleration detection
    acceleration_factor: float          # Rate of acceleration
    confidence_level: float             # 0-1 confidence
    calculation_steps: List[Dict]       # Audit trail
    provenance_hash: str


@dataclass
class RemainingUsefulLife:
    """Remaining useful life estimation results."""
    equipment_id: str
    current_r_value: float
    minimum_acceptable_r_value: float
    safety_limit_temperature_c: float
    time_to_minimum_r_years: float
    time_to_safety_limit_years: float
    confidence_interval_lower_years: float
    confidence_interval_upper_years: float
    weibull_shape: float                # Weibull beta parameter
    weibull_scale: float                # Weibull eta parameter
    reliability_at_1y: float            # P(survival) at 1 year
    reliability_at_5y: float            # P(survival) at 5 years
    recommended_inspection_interval_days: int
    recommended_replacement_date: Optional[datetime]
    provenance_hash: str


@dataclass
class InspectionRecord:
    """Complete inspection record with findings."""
    inspection_id: str
    equipment_id: str
    inspection_date: datetime
    inspection_type: InspectionType
    inspector_id: str
    performance_data: PerformanceDataPoint
    previous_inspection_id: Optional[str]
    change_from_previous: Optional[Dict[str, float]]
    condition_category: ConditionCategory
    alerts: List[Dict[str, Any]]
    recommendations: List[str]
    images_captured: int
    areas_inspected: int
    defects_found: int
    provenance_hash: str


@dataclass
class TrendAnalysisResult:
    """Results of trend analysis."""
    metric_name: str
    data_points_analyzed: int
    time_span_days: int
    moving_average_7d: float
    moving_average_30d: float
    moving_average_90d: float
    trend_direction: TrendDirection
    rate_of_change_per_day: float
    seasonal_pattern_detected: bool
    seasonal_amplitude: Optional[float]
    seasonal_period_days: Optional[int]
    anomalies_detected: List[Dict[str, Any]]
    forecast_30d: float
    forecast_90d: float
    provenance_hash: str


@dataclass
class FleetBenchmark:
    """Fleet/facility benchmarking results."""
    fleet_size: int
    equipment_ids: List[str]
    average_r_value: float
    median_r_value: float
    std_deviation: float
    percentile_25: float
    percentile_75: float
    best_performer_id: str
    worst_performer_id: str
    average_age_years: float
    age_correlation: float              # Correlation between age and R-value
    type_comparison: Dict[str, Dict[str, float]]
    location_comparison: Dict[str, Dict[str, float]]
    recommendations: List[str]
    provenance_hash: str


@dataclass
class FutureConditionForecast:
    """Predictive analytics forecast results."""
    equipment_id: str
    forecast_date: datetime
    forecast_horizon_days: int
    predicted_r_value: float
    predicted_heat_loss_w: float
    predicted_efficiency_percent: float
    confidence_interval_r_lower: float
    confidence_interval_r_upper: float
    maintenance_trigger_date: Optional[datetime]
    replacement_trigger_date: Optional[datetime]
    estimated_cost_to_date: float
    budget_forecast_1y: float
    budget_forecast_5y: float
    risk_score: float                   # 0-100
    provenance_hash: str


@dataclass
class KPIDashboard:
    """KPI dashboard metrics."""
    dashboard_id: str
    generated_at: datetime
    reporting_period_start: datetime
    reporting_period_end: datetime

    # Overall health metrics
    overall_health_index: float         # 0-100 composite score
    equipment_count: int

    # Condition distribution
    condition_distribution: Dict[str, int]
    condition_distribution_percent: Dict[str, float]

    # Heat loss metrics
    total_heat_loss_w: float
    total_heat_loss_change_percent: float
    heat_loss_by_area: Dict[str, float]

    # R-value metrics
    fleet_average_r_value: float
    r_value_trend_30d: float

    # Inspection metrics
    inspections_completed: int
    inspections_scheduled: int
    inspection_completion_rate: float

    # Repair metrics
    repairs_completed: int
    repairs_pending: int
    repair_completion_rate: float
    average_repair_time_days: float

    # Cost metrics
    energy_loss_cost_period: float
    repair_cost_period: float
    total_cost_period: float
    cost_trend_percent: float

    # Alerts summary
    active_alerts_count: int
    alerts_by_severity: Dict[str, int]

    # Predictions
    equipment_requiring_attention_30d: int
    estimated_cost_next_quarter: float

    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "dashboard_id": self.dashboard_id,
            "generated_at": self.generated_at.isoformat(),
            "overall_health_index": self.overall_health_index,
            "equipment_count": self.equipment_count,
            "condition_distribution_percent": self.condition_distribution_percent,
            "total_heat_loss_w": self.total_heat_loss_w,
            "inspection_completion_rate": self.inspection_completion_rate,
            "repair_completion_rate": self.repair_completion_rate,
            "active_alerts_count": self.active_alerts_count,
            "provenance_hash": self.provenance_hash
        }


# =============================================================================
# CALCULATION STEP TRACKING
# =============================================================================

@dataclass
class CalculationStep:
    """Records a single calculation step for audit trail."""
    step_number: int
    description: str
    operation: str
    inputs: Dict[str, Any]
    output_value: Any
    output_name: str
    formula: Optional[str] = None
    units: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        inputs_serialized = {}
        for k, v in self.inputs.items():
            if isinstance(v, Decimal):
                inputs_serialized[k] = str(v)
            elif isinstance(v, datetime):
                inputs_serialized[k] = v.isoformat()
            else:
                inputs_serialized[k] = v

        output_serialized = str(self.output_value) if isinstance(self.output_value, Decimal) else self.output_value

        return {
            "step_number": self.step_number,
            "description": self.description,
            "operation": self.operation,
            "inputs": inputs_serialized,
            "output_value": output_serialized,
            "output_name": self.output_name,
            "formula": self.formula,
            "units": self.units,
            "timestamp": self.timestamp
        }


# =============================================================================
# INSULATION PERFORMANCE TRACKER
# =============================================================================

class InsulationPerformanceTracker:
    """
    Insulation Performance Tracker for GL-015 INSULSCAN.

    Provides historical tracking, trend analysis, degradation modeling,
    remaining useful life estimation, fleet benchmarking, and predictive
    analytics for industrial insulation systems.

    Zero-Hallucination Guarantee:
        - All calculations use deterministic engineering formulas
        - No LLM inference for numeric values
        - Bit-perfect reproducibility with SHA-256 provenance
        - Complete audit trails for regulatory compliance

    Example:
        >>> tracker = InsulationPerformanceTracker()
        >>> tracker.add_performance_data("EQ-001", data_point)
        >>> degradation = tracker.calculate_degradation_rate("EQ-001")
        >>> rul = tracker.estimate_remaining_life("EQ-001")
        >>> dashboard = tracker.generate_kpi_dashboard()
    """

    VERSION: str = "1.0.0"
    PRECISION: int = 4

    # Default thresholds
    MINIMUM_R_VALUE_RATIO: float = 0.50      # 50% of design value
    SAFETY_TEMPERATURE_LIMIT_C: float = 60.0  # OSHA personnel protection
    ANOMALY_ZSCORE_THRESHOLD: float = 2.5     # Standard deviations
    ACCELERATION_THRESHOLD: float = 1.5       # 50% faster than baseline

    # Weibull default parameters for insulation systems
    DEFAULT_WEIBULL_SHAPE: float = 2.5        # Beta (wear-out behavior)
    DEFAULT_WEIBULL_SCALE_YEARS: float = 15.0  # Eta (characteristic life)

    # Energy cost default
    DEFAULT_ENERGY_COST_PER_KWH: float = 0.08  # $/kWh

    def __init__(
        self,
        precision: int = 4,
        energy_cost_per_kwh: float = 0.08,
        safety_temperature_limit_c: float = 60.0
    ) -> None:
        """
        Initialize the Insulation Performance Tracker.

        Args:
            precision: Decimal places for rounding
            energy_cost_per_kwh: Energy cost in $/kWh
            safety_temperature_limit_c: Personnel safety temperature limit
        """
        self.precision = precision
        self.energy_cost = Decimal(str(energy_cost_per_kwh))
        self.safety_limit_c = Decimal(str(safety_temperature_limit_c))

        # Data storage (in production, this would be a database)
        self._performance_history: Dict[str, List[PerformanceDataPoint]] = {}
        self._inspection_records: Dict[str, List[InspectionRecord]] = {}
        self._equipment_metadata: Dict[str, Dict[str, Any]] = {}

        # Calculation tracking
        self._calculation_steps: List[CalculationStep] = []
        self._step_counter: int = 0

    # =========================================================================
    # 1. PERFORMANCE METRICS TRACKING
    # =========================================================================

    def track_performance_history(
        self,
        equipment_id: str,
        data_points: List[PerformanceDataPoint]
    ) -> Dict[str, Any]:
        """
        Track performance history for equipment.

        Records performance data points and calculates summary statistics.

        Args:
            equipment_id: Unique equipment identifier
            data_points: List of performance measurements

        Returns:
            Summary of tracked performance data
        """
        self._reset_calculation_state()

        # Initialize storage if needed
        if equipment_id not in self._performance_history:
            self._performance_history[equipment_id] = []

        # Add data points
        for dp in data_points:
            self._performance_history[equipment_id].append(dp)

        # Sort by timestamp
        self._performance_history[equipment_id].sort(key=lambda x: x.timestamp)

        history = self._performance_history[equipment_id]

        if len(history) < 2:
            return {
                "equipment_id": equipment_id,
                "data_points_count": len(history),
                "status": "insufficient_data",
                "message": "At least 2 data points required for analysis"
            }

        # Calculate summary statistics
        r_values = [dp.r_value_m2k_w for dp in history]
        temperatures = [dp.surface_temperature_c for dp in history]
        heat_losses = [dp.heat_loss_rate_w for dp in history]
        efficiencies = [dp.efficiency_percent for dp in history]

        time_span = (history[-1].timestamp - history[0].timestamp).days

        self._add_calculation_step(
            description="Calculate performance history statistics",
            operation="summary_statistics",
            inputs={
                "equipment_id": equipment_id,
                "data_points_count": len(history),
                "time_span_days": time_span
            },
            output_value=statistics.mean(r_values),
            output_name="mean_r_value",
            formula="mean(R_values)"
        )

        summary = {
            "equipment_id": equipment_id,
            "data_points_count": len(history),
            "time_span_days": time_span,
            "first_measurement": history[0].timestamp.isoformat(),
            "last_measurement": history[-1].timestamp.isoformat(),
            "r_value_statistics": {
                "current": self._round(r_values[-1]),
                "mean": self._round(statistics.mean(r_values)),
                "min": self._round(min(r_values)),
                "max": self._round(max(r_values)),
                "std_dev": self._round(statistics.stdev(r_values)) if len(r_values) > 1 else 0.0
            },
            "surface_temperature_statistics": {
                "current": self._round(temperatures[-1]),
                "mean": self._round(statistics.mean(temperatures)),
                "min": self._round(min(temperatures)),
                "max": self._round(max(temperatures))
            },
            "heat_loss_statistics": {
                "current": self._round(heat_losses[-1]),
                "mean": self._round(statistics.mean(heat_losses)),
                "total_energy_lost_kwh": self._round(sum(heat_losses) * time_span * 24 / 1000)
            },
            "efficiency_statistics": {
                "current": self._round(efficiencies[-1]),
                "mean": self._round(statistics.mean(efficiencies))
            },
            "provenance_hash": self._generate_provenance_hash({
                "equipment_id": equipment_id,
                "data_points_count": len(history)
            })
        }

        return summary

    def add_performance_data(
        self,
        equipment_id: str,
        data_point: PerformanceDataPoint
    ) -> None:
        """
        Add a single performance data point.

        Args:
            equipment_id: Equipment identifier
            data_point: Performance measurement
        """
        if equipment_id not in self._performance_history:
            self._performance_history[equipment_id] = []

        self._performance_history[equipment_id].append(data_point)
        self._performance_history[equipment_id].sort(key=lambda x: x.timestamp)

    # =========================================================================
    # 2. DEGRADATION RATE ANALYSIS
    # =========================================================================

    def calculate_degradation_rate(
        self,
        equipment_id: str,
        model_type: DegradationModel = DegradationModel.EXPONENTIAL,
        design_r_value: Optional[float] = None
    ) -> DegradationAnalysis:
        """
        Calculate degradation rate using curve fitting.

        Supports multiple degradation models:
        - Linear: R(t) = R_0 - k*t
        - Exponential: R(t) = R_0 * exp(-lambda*t)
        - Weibull: Based on reliability analysis

        Args:
            equipment_id: Equipment identifier
            model_type: Degradation model to use
            design_r_value: Original design R-value (optional)

        Returns:
            DegradationAnalysis with rate and predictions
        """
        self._reset_calculation_state()

        history = self._performance_history.get(equipment_id, [])

        if len(history) < 3:
            raise ValueError(f"Insufficient data for degradation analysis. Need at least 3 points, have {len(history)}")

        # Extract R-values and time (in years)
        r_values = [dp.r_value_m2k_w for dp in history]
        base_time = history[0].timestamp
        times_years = [(dp.timestamp - base_time).days / 365.25 for dp in history]

        # Initial R-value estimation
        r_0 = Decimal(str(design_r_value)) if design_r_value else Decimal(str(r_values[0]))

        self._add_calculation_step(
            description="Extract degradation analysis inputs",
            operation="data_extraction",
            inputs={
                "equipment_id": equipment_id,
                "data_points": len(history),
                "time_span_years": times_years[-1]
            },
            output_value=float(r_0),
            output_name="initial_r_value",
            units="m2K/W"
        )

        # Fit degradation model
        if model_type == DegradationModel.LINEAR:
            k, r_squared = self._fit_linear_degradation(times_years, r_values, float(r_0))
            degradation_rate = Decimal(str(k))

            # Predictions
            current_r = r_0 - degradation_rate * Decimal(str(times_years[-1]))
            r_1y = r_0 - degradation_rate * Decimal(str(times_years[-1] + 1))
            r_5y = r_0 - degradation_rate * Decimal(str(times_years[-1] + 5))

            self._add_calculation_step(
                description="Linear degradation model fitting",
                operation="linear_regression",
                inputs={"R_0": float(r_0), "data_points": len(r_values)},
                output_value=k,
                output_name="degradation_rate_k",
                formula="R(t) = R_0 - k*t",
                units="m2K/W/year"
            )

        elif model_type == DegradationModel.EXPONENTIAL:
            lambda_val, r_squared = self._fit_exponential_degradation(times_years, r_values, float(r_0))
            degradation_rate = Decimal(str(lambda_val))

            # Predictions: R(t) = R_0 * exp(-lambda*t)
            import math
            current_r = r_0 * Decimal(str(math.exp(-lambda_val * times_years[-1])))
            r_1y = r_0 * Decimal(str(math.exp(-lambda_val * (times_years[-1] + 1))))
            r_5y = r_0 * Decimal(str(math.exp(-lambda_val * (times_years[-1] + 5))))

            self._add_calculation_step(
                description="Exponential degradation model fitting",
                operation="exponential_regression",
                inputs={"R_0": float(r_0), "data_points": len(r_values)},
                output_value=lambda_val,
                output_name="decay_constant_lambda",
                formula="R(t) = R_0 * exp(-lambda*t)",
                units="1/year"
            )

        elif model_type == DegradationModel.LOGARITHMIC:
            k, r_squared = self._fit_logarithmic_degradation(times_years, r_values, float(r_0))
            degradation_rate = Decimal(str(k))

            # Predictions: R(t) = R_0 - k*ln(t+1)
            current_r = r_0 - degradation_rate * Decimal(str(math.log(times_years[-1] + 1)))
            r_1y = r_0 - degradation_rate * Decimal(str(math.log(times_years[-1] + 2)))
            r_5y = r_0 - degradation_rate * Decimal(str(math.log(times_years[-1] + 6)))

            self._add_calculation_step(
                description="Logarithmic degradation model fitting",
                operation="logarithmic_regression",
                inputs={"R_0": float(r_0), "data_points": len(r_values)},
                output_value=k,
                output_name="degradation_rate_k",
                formula="R(t) = R_0 - k*ln(t+1)",
                units="m2K/W"
            )
        else:
            # Default to exponential
            lambda_val, r_squared = self._fit_exponential_degradation(times_years, r_values, float(r_0))
            degradation_rate = Decimal(str(lambda_val))
            current_r = r_0 * Decimal(str(math.exp(-lambda_val * times_years[-1])))
            r_1y = r_0 * Decimal(str(math.exp(-lambda_val * (times_years[-1] + 1))))
            r_5y = r_0 * Decimal(str(math.exp(-lambda_val * (times_years[-1] + 5))))

        # Calculate annual degradation percentage
        if model_type == DegradationModel.EXPONENTIAL:
            annual_deg_pct = (1 - math.exp(-float(degradation_rate))) * 100
        else:
            annual_deg_pct = (float(degradation_rate) / float(r_0)) * 100 if r_0 > 0 else 0

        # Detect acceleration
        is_accelerating, accel_factor = self._detect_acceleration(times_years, r_values)

        self._add_calculation_step(
            description="Calculate annual degradation percentage",
            operation="percentage_calculation",
            inputs={
                "degradation_rate": float(degradation_rate),
                "initial_r_value": float(r_0)
            },
            output_value=annual_deg_pct,
            output_name="annual_degradation_percent",
            units="%"
        )

        # Confidence based on R-squared and data points
        confidence = min(r_squared, len(history) / 20)  # Cap at 1.0

        provenance_hash = self._generate_provenance_hash({
            "equipment_id": equipment_id,
            "model_type": model_type.value,
            "degradation_rate": float(degradation_rate),
            "r_squared": r_squared
        })

        return DegradationAnalysis(
            model_type=model_type,
            initial_r_value=self._round(float(r_0)),
            degradation_rate=self._round(float(degradation_rate)),
            r_squared=self._round(r_squared),
            current_r_value=self._round(float(current_r)),
            predicted_r_value_1y=self._round(float(max(Decimal('0'), r_1y))),
            predicted_r_value_5y=self._round(float(max(Decimal('0'), r_5y))),
            annual_degradation_percent=self._round(annual_deg_pct),
            is_accelerating=is_accelerating,
            acceleration_factor=self._round(accel_factor),
            confidence_level=self._round(confidence),
            calculation_steps=[s.to_dict() for s in self._calculation_steps],
            provenance_hash=provenance_hash
        )

    def _fit_linear_degradation(
        self,
        times: List[float],
        r_values: List[float],
        r_0: float
    ) -> Tuple[float, float]:
        """
        Fit linear degradation model using least squares.

        R(t) = R_0 - k*t

        Returns:
            Tuple of (k, R_squared)
        """
        n = len(times)

        # Calculate slope k using least squares
        # Minimize sum((R_0 - k*t - R_actual)^2)
        sum_t = sum(times)
        sum_r = sum(r_values)
        sum_t_sq = sum(t * t for t in times)
        sum_tr = sum(t * r for t, r in zip(times, r_values))

        # k = (n*sum(t*R) - sum(t)*sum(R)) / (n*sum(t^2) - sum(t)^2)
        # But we're fitting R = R_0 - k*t, so we need to adjust
        denominator = n * sum_t_sq - sum_t * sum_t

        if abs(denominator) < 1e-10:
            return 0.0, 0.0

        # Slope (negative because degradation)
        k = -(n * sum_tr - sum_t * sum_r) / denominator

        # Ensure k is positive (degradation)
        k = max(0, k)

        # Calculate R-squared
        r_mean = sum_r / n
        ss_tot = sum((r - r_mean) ** 2 for r in r_values)
        ss_res = sum((r - (r_0 - k * t)) ** 2 for t, r in zip(times, r_values))

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        r_squared = max(0, min(1, r_squared))

        return k, r_squared

    def _fit_exponential_degradation(
        self,
        times: List[float],
        r_values: List[float],
        r_0: float
    ) -> Tuple[float, float]:
        """
        Fit exponential degradation model.

        R(t) = R_0 * exp(-lambda*t)
        ln(R/R_0) = -lambda*t

        Returns:
            Tuple of (lambda, R_squared)
        """
        # Transform to linear: ln(R/R_0) = -lambda*t
        log_ratios = []
        valid_times = []

        for t, r in zip(times, r_values):
            if r > 0 and r_0 > 0:
                log_ratios.append(math.log(r / r_0))
                valid_times.append(t)

        if len(valid_times) < 2:
            return 0.01, 0.0  # Default small decay rate

        n = len(valid_times)
        sum_t = sum(valid_times)
        sum_y = sum(log_ratios)
        sum_t_sq = sum(t * t for t in valid_times)
        sum_ty = sum(t * y for t, y in zip(valid_times, log_ratios))

        denominator = n * sum_t_sq - sum_t * sum_t

        if abs(denominator) < 1e-10:
            return 0.01, 0.0

        # Slope is -lambda
        slope = (n * sum_ty - sum_t * sum_y) / denominator
        lambda_val = -slope

        # Ensure lambda is positive
        lambda_val = max(0.001, lambda_val)

        # Calculate R-squared
        y_mean = sum_y / n
        ss_tot = sum((y - y_mean) ** 2 for y in log_ratios)
        ss_res = sum((y - (-lambda_val * t)) ** 2 for t, y in zip(valid_times, log_ratios))

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        r_squared = max(0, min(1, r_squared))

        return lambda_val, r_squared

    def _fit_logarithmic_degradation(
        self,
        times: List[float],
        r_values: List[float],
        r_0: float
    ) -> Tuple[float, float]:
        """
        Fit logarithmic degradation model.

        R(t) = R_0 - k*ln(t+1)

        Returns:
            Tuple of (k, R_squared)
        """
        # Transform time: x = ln(t+1)
        log_times = [math.log(t + 1) for t in times]

        n = len(times)
        sum_x = sum(log_times)
        sum_r = sum(r_values)
        sum_x_sq = sum(x * x for x in log_times)
        sum_xr = sum(x * r for x, r in zip(log_times, r_values))

        denominator = n * sum_x_sq - sum_x * sum_x

        if abs(denominator) < 1e-10:
            return 0.0, 0.0

        # Slope (negative for degradation)
        k = -(n * sum_xr - sum_x * sum_r) / denominator
        k = max(0, k)

        # Calculate R-squared
        r_mean = sum_r / n
        ss_tot = sum((r - r_mean) ** 2 for r in r_values)
        ss_res = sum((r - (r_0 - k * x)) ** 2 for x, r in zip(log_times, r_values))

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        r_squared = max(0, min(1, r_squared))

        return k, r_squared

    def _detect_acceleration(
        self,
        times: List[float],
        r_values: List[float]
    ) -> Tuple[bool, float]:
        """
        Detect if degradation is accelerating.

        Compares recent degradation rate to historical rate.

        Returns:
            Tuple of (is_accelerating, acceleration_factor)
        """
        if len(times) < 6:
            return False, 1.0

        # Split data into two halves
        mid = len(times) // 2

        # Calculate degradation rate for each half
        early_times = times[:mid]
        early_r = r_values[:mid]
        late_times = times[mid:]
        late_r = r_values[mid:]

        # Simple rate calculation (R change / time change)
        if len(early_times) > 1 and len(late_times) > 1:
            early_rate = abs(early_r[-1] - early_r[0]) / (early_times[-1] - early_times[0]) if early_times[-1] != early_times[0] else 0
            late_rate = abs(late_r[-1] - late_r[0]) / (late_times[-1] - late_times[0]) if late_times[-1] != late_times[0] else 0

            if early_rate > 0:
                accel_factor = late_rate / early_rate
                is_accelerating = accel_factor > self.ACCELERATION_THRESHOLD
                return is_accelerating, accel_factor

        return False, 1.0

    # =========================================================================
    # 3. REMAINING USEFUL LIFE (RUL) ESTIMATION
    # =========================================================================

    def estimate_remaining_life(
        self,
        equipment_id: str,
        minimum_r_value_ratio: float = 0.50,
        design_r_value: Optional[float] = None,
        installation_date: Optional[datetime] = None
    ) -> RemainingUsefulLife:
        """
        Estimate remaining useful life of insulation.

        Uses degradation analysis and Weibull reliability modeling
        to predict time to failure thresholds.

        Args:
            equipment_id: Equipment identifier
            minimum_r_value_ratio: Minimum acceptable R-value as ratio of design
            design_r_value: Original design R-value
            installation_date: Date of installation

        Returns:
            RemainingUsefulLife with predictions and confidence intervals
        """
        self._reset_calculation_state()

        history = self._performance_history.get(equipment_id, [])

        if len(history) < 3:
            raise ValueError(f"Insufficient data for RUL estimation. Need at least 3 points, have {len(history)}")

        # Get degradation analysis
        degradation = self.calculate_degradation_rate(
            equipment_id,
            DegradationModel.EXPONENTIAL,
            design_r_value
        )

        # Current state
        current_r = Decimal(str(history[-1].r_value_m2k_w))
        r_0 = Decimal(str(degradation.initial_r_value))
        min_r = r_0 * Decimal(str(minimum_r_value_ratio))

        self._add_calculation_step(
            description="Define RUL thresholds",
            operation="threshold_calculation",
            inputs={
                "design_r_value": float(r_0),
                "minimum_ratio": minimum_r_value_ratio
            },
            output_value=float(min_r),
            output_name="minimum_acceptable_r_value",
            units="m2K/W"
        )

        # Calculate time to minimum R-value
        # For exponential: R_min = R_0 * exp(-lambda*t)
        # t = -ln(R_min/R_0) / lambda
        lambda_val = degradation.degradation_rate

        if lambda_val > 0 and min_r > 0 and r_0 > 0:
            ratio = float(min_r / r_0)
            if ratio > 0:
                time_to_min_years = -math.log(ratio) / lambda_val
            else:
                time_to_min_years = float('inf')
        else:
            time_to_min_years = float('inf')

        # Adjust for current age
        if installation_date:
            current_age_years = (history[-1].timestamp - installation_date).days / 365.25
        else:
            current_age_years = (history[-1].timestamp - history[0].timestamp).days / 365.25

        remaining_time_years = max(0, time_to_min_years - current_age_years)

        self._add_calculation_step(
            description="Calculate time to minimum R-value threshold",
            operation="rul_calculation",
            inputs={
                "lambda": lambda_val,
                "current_age_years": current_age_years,
                "time_to_threshold_years": time_to_min_years
            },
            output_value=remaining_time_years,
            output_name="remaining_useful_life_years",
            formula="RUL = -ln(R_min/R_0)/lambda - current_age",
            units="years"
        )

        # Calculate time to safety temperature limit
        # This requires process temperature and heat transfer analysis
        current_surface_temp = Decimal(str(history[-1].surface_temperature_c))
        process_temp = Decimal(str(history[-1].process_temperature_c))

        if current_r > 0:
            # Simplified calculation assuming linear relationship
            # As R-value decreases, surface temperature increases
            temp_ratio = (float(self.safety_limit_c) - float(history[-1].ambient_temperature_c)) / \
                        (float(process_temp) - float(history[-1].ambient_temperature_c))
            r_at_safety_limit = float(current_r) * (1 - temp_ratio) if temp_ratio < 1 else 0

            if r_at_safety_limit > 0 and lambda_val > 0:
                time_to_safety_years = -math.log(r_at_safety_limit / float(r_0)) / lambda_val - current_age_years
            else:
                time_to_safety_years = remaining_time_years * 0.8  # Conservative estimate
        else:
            time_to_safety_years = 0

        time_to_safety_years = max(0, time_to_safety_years)

        # Weibull reliability analysis
        weibull_shape = self.DEFAULT_WEIBULL_SHAPE
        weibull_scale = self.DEFAULT_WEIBULL_SCALE_YEARS

        # Adjust Weibull parameters based on observed degradation
        if degradation.is_accelerating:
            weibull_shape *= 1.2  # Increase shape for accelerated wear-out

        # Reliability at 1 and 5 years: R(t) = exp(-(t/eta)^beta)
        reliability_1y = math.exp(-((current_age_years + 1) / weibull_scale) ** weibull_shape)
        reliability_5y = math.exp(-((current_age_years + 5) / weibull_scale) ** weibull_shape)

        self._add_calculation_step(
            description="Weibull reliability calculation",
            operation="weibull_analysis",
            inputs={
                "shape_beta": weibull_shape,
                "scale_eta": weibull_scale,
                "current_age_years": current_age_years
            },
            output_value=reliability_1y,
            output_name="reliability_1_year",
            formula="R(t) = exp(-(t/eta)^beta)"
        )

        # Confidence intervals (using degradation uncertainty)
        uncertainty_factor = 1 - degradation.r_squared
        ci_lower = remaining_time_years * (1 - uncertainty_factor)
        ci_upper = remaining_time_years * (1 + uncertainty_factor)

        # Recommended inspection interval
        if remaining_time_years < 1:
            inspection_interval = 30  # Monthly
        elif remaining_time_years < 3:
            inspection_interval = 90  # Quarterly
        elif remaining_time_years < 5:
            inspection_interval = 180  # Semi-annual
        else:
            inspection_interval = 365  # Annual

        # Recommended replacement date
        if remaining_time_years < 20:
            replacement_date = history[-1].timestamp + timedelta(days=remaining_time_years * 365.25)
        else:
            replacement_date = None

        provenance_hash = self._generate_provenance_hash({
            "equipment_id": equipment_id,
            "current_r_value": float(current_r),
            "remaining_life_years": remaining_time_years
        })

        return RemainingUsefulLife(
            equipment_id=equipment_id,
            current_r_value=self._round(float(current_r)),
            minimum_acceptable_r_value=self._round(float(min_r)),
            safety_limit_temperature_c=float(self.safety_limit_c),
            time_to_minimum_r_years=self._round(remaining_time_years),
            time_to_safety_limit_years=self._round(time_to_safety_years),
            confidence_interval_lower_years=self._round(ci_lower),
            confidence_interval_upper_years=self._round(ci_upper),
            weibull_shape=self._round(weibull_shape),
            weibull_scale=self._round(weibull_scale),
            reliability_at_1y=self._round(reliability_1y),
            reliability_at_5y=self._round(reliability_5y),
            recommended_inspection_interval_days=inspection_interval,
            recommended_replacement_date=replacement_date,
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # 4. INSPECTION HISTORY MANAGEMENT
    # =========================================================================

    def manage_inspection_history(
        self,
        equipment_id: str,
        inspection_data: Optional[Dict[str, Any]] = None,
        action: str = "get"
    ) -> Union[InspectionRecord, List[InspectionRecord], Dict[str, Any]]:
        """
        Manage inspection history for equipment.

        Actions:
        - "add": Add new inspection record
        - "get": Get all inspection records
        - "compare": Compare current vs previous inspection
        - "recommend_interval": Get recommended inspection interval

        Args:
            equipment_id: Equipment identifier
            inspection_data: Inspection data for "add" action
            action: Action to perform

        Returns:
            Inspection records or comparison results
        """
        self._reset_calculation_state()

        if action == "add" and inspection_data:
            return self._add_inspection_record(equipment_id, inspection_data)

        elif action == "get":
            return self._get_inspection_history(equipment_id)

        elif action == "compare":
            return self._compare_inspections(equipment_id)

        elif action == "recommend_interval":
            return self._recommend_inspection_interval(equipment_id)

        else:
            raise ValueError(f"Unknown action: {action}")

    def _add_inspection_record(
        self,
        equipment_id: str,
        inspection_data: Dict[str, Any]
    ) -> InspectionRecord:
        """Add a new inspection record."""
        # Initialize storage if needed
        if equipment_id not in self._inspection_records:
            self._inspection_records[equipment_id] = []

        # Get previous inspection
        previous = self._inspection_records[equipment_id][-1] if self._inspection_records[equipment_id] else None

        # Create performance data point
        perf_data = PerformanceDataPoint(
            timestamp=inspection_data.get("timestamp", datetime.utcnow()),
            surface_temperature_c=inspection_data["surface_temperature_c"],
            ambient_temperature_c=inspection_data["ambient_temperature_c"],
            process_temperature_c=inspection_data["process_temperature_c"],
            r_value_m2k_w=inspection_data["r_value_m2k_w"],
            heat_loss_rate_w=inspection_data["heat_loss_rate_w"],
            efficiency_percent=inspection_data["efficiency_percent"],
            inspector_id=inspection_data.get("inspector_id"),
            inspection_type=InspectionType(inspection_data.get("inspection_type", "thermal_imaging"))
        )

        # Calculate change from previous
        change_from_previous = None
        if previous:
            prev_perf = previous.performance_data
            change_from_previous = {
                "r_value_change": perf_data.r_value_m2k_w - prev_perf.r_value_m2k_w,
                "r_value_change_percent": ((perf_data.r_value_m2k_w - prev_perf.r_value_m2k_w) / prev_perf.r_value_m2k_w * 100) if prev_perf.r_value_m2k_w > 0 else 0,
                "heat_loss_change": perf_data.heat_loss_rate_w - prev_perf.heat_loss_rate_w,
                "surface_temp_change": perf_data.surface_temperature_c - prev_perf.surface_temperature_c,
                "days_since_previous": (perf_data.timestamp - prev_perf.timestamp).days
            }

        # Determine condition category
        design_r = self._equipment_metadata.get(equipment_id, {}).get("design_r_value", perf_data.r_value_m2k_w * 1.5)
        r_ratio = perf_data.r_value_m2k_w / design_r if design_r > 0 else 1.0

        if r_ratio > 0.90:
            condition = ConditionCategory.EXCELLENT
        elif r_ratio > 0.75:
            condition = ConditionCategory.GOOD
        elif r_ratio > 0.50:
            condition = ConditionCategory.FAIR
        elif r_ratio > 0.25:
            condition = ConditionCategory.POOR
        else:
            condition = ConditionCategory.FAILED

        # Generate alerts
        alerts = self._generate_alerts(perf_data, change_from_previous, condition)

        # Generate recommendations
        recommendations = self._generate_recommendations(condition, change_from_previous)

        # Create inspection record
        inspection_id = f"INS-{equipment_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        record = InspectionRecord(
            inspection_id=inspection_id,
            equipment_id=equipment_id,
            inspection_date=perf_data.timestamp,
            inspection_type=perf_data.inspection_type,
            inspector_id=perf_data.inspector_id or "system",
            performance_data=perf_data,
            previous_inspection_id=previous.inspection_id if previous else None,
            change_from_previous=change_from_previous,
            condition_category=condition,
            alerts=alerts,
            recommendations=recommendations,
            images_captured=inspection_data.get("images_captured", 0),
            areas_inspected=inspection_data.get("areas_inspected", 1),
            defects_found=inspection_data.get("defects_found", 0),
            provenance_hash=self._generate_provenance_hash({
                "inspection_id": inspection_id,
                "equipment_id": equipment_id,
                "r_value": perf_data.r_value_m2k_w
            })
        )

        self._inspection_records[equipment_id].append(record)

        # Also add to performance history
        self.add_performance_data(equipment_id, perf_data)

        return record

    def _get_inspection_history(self, equipment_id: str) -> List[InspectionRecord]:
        """Get all inspection records for equipment."""
        return self._inspection_records.get(equipment_id, [])

    def _compare_inspections(self, equipment_id: str) -> Dict[str, Any]:
        """Compare current vs previous inspection."""
        records = self._inspection_records.get(equipment_id, [])

        if len(records) < 2:
            return {
                "equipment_id": equipment_id,
                "status": "insufficient_data",
                "message": "Need at least 2 inspections to compare"
            }

        current = records[-1]
        previous = records[-2]

        comparison = {
            "equipment_id": equipment_id,
            "current_inspection": current.inspection_id,
            "previous_inspection": previous.inspection_id,
            "time_between_days": (current.inspection_date - previous.inspection_date).days,
            "changes": current.change_from_previous,
            "condition_change": {
                "previous": previous.condition_category.value,
                "current": current.condition_category.value,
                "improved": list(ConditionCategory).index(current.condition_category) < list(ConditionCategory).index(previous.condition_category)
            },
            "alerts": current.alerts,
            "recommendations": current.recommendations
        }

        return comparison

    def _recommend_inspection_interval(self, equipment_id: str) -> Dict[str, Any]:
        """Recommend inspection interval based on condition and trends."""
        records = self._inspection_records.get(equipment_id, [])

        if not records:
            return {
                "equipment_id": equipment_id,
                "recommended_interval_days": 365,
                "reason": "No inspection history - annual inspection recommended"
            }

        current = records[-1]
        condition = current.condition_category

        # Base interval on condition
        base_intervals = {
            ConditionCategory.EXCELLENT: 365,
            ConditionCategory.GOOD: 180,
            ConditionCategory.FAIR: 90,
            ConditionCategory.POOR: 30,
            ConditionCategory.FAILED: 7
        }

        interval = base_intervals[condition]
        reason = f"Based on {condition.value} condition"

        # Adjust based on degradation rate if available
        if len(records) >= 3:
            try:
                degradation = self.calculate_degradation_rate(equipment_id)
                if degradation.is_accelerating:
                    interval = int(interval * 0.5)
                    reason += " with accelerated degradation detected"
            except (ValueError, Exception):
                pass

        return {
            "equipment_id": equipment_id,
            "recommended_interval_days": interval,
            "next_inspection_date": (current.inspection_date + timedelta(days=interval)).isoformat(),
            "reason": reason
        }

    def _generate_alerts(
        self,
        perf_data: PerformanceDataPoint,
        change: Optional[Dict[str, float]],
        condition: ConditionCategory
    ) -> List[Dict[str, Any]]:
        """Generate alerts based on inspection results."""
        alerts = []

        # Safety temperature alert
        if perf_data.surface_temperature_c > float(self.safety_limit_c):
            alerts.append({
                "severity": AlertSeverity.EMERGENCY.value,
                "type": "safety_temperature_exceeded",
                "message": f"Surface temperature {perf_data.surface_temperature_c}C exceeds safety limit {self.safety_limit_c}C",
                "action_required": "Immediate repair or access restriction"
            })
        elif perf_data.surface_temperature_c > float(self.safety_limit_c) * 0.9:
            alerts.append({
                "severity": AlertSeverity.CRITICAL.value,
                "type": "safety_temperature_approaching",
                "message": f"Surface temperature {perf_data.surface_temperature_c}C approaching safety limit",
                "action_required": "Schedule repair within 7 days"
            })

        # Condition alerts
        if condition == ConditionCategory.FAILED:
            alerts.append({
                "severity": AlertSeverity.CRITICAL.value,
                "type": "insulation_failed",
                "message": "Insulation has failed - immediate replacement required",
                "action_required": "Replace insulation"
            })
        elif condition == ConditionCategory.POOR:
            alerts.append({
                "severity": AlertSeverity.WARNING.value,
                "type": "insulation_poor",
                "message": "Insulation in poor condition - schedule repair",
                "action_required": "Plan repair within 30 days"
            })

        # Rapid change alerts
        if change and change.get("r_value_change_percent", 0) < -10:
            alerts.append({
                "severity": AlertSeverity.WARNING.value,
                "type": "rapid_degradation",
                "message": f"R-value decreased {abs(change['r_value_change_percent']):.1f}% since last inspection",
                "action_required": "Investigate cause of rapid degradation"
            })

        return alerts

    def _generate_recommendations(
        self,
        condition: ConditionCategory,
        change: Optional[Dict[str, float]]
    ) -> List[str]:
        """Generate recommendations based on condition."""
        recommendations = []

        if condition == ConditionCategory.FAILED:
            recommendations.extend([
                "Replace insulation immediately",
                "Conduct root cause analysis",
                "Review installation procedures"
            ])
        elif condition == ConditionCategory.POOR:
            recommendations.extend([
                "Schedule insulation repair or replacement",
                "Evaluate repair vs replace economics",
                "Check for moisture intrusion"
            ])
        elif condition == ConditionCategory.FAIR:
            recommendations.extend([
                "Monitor degradation trend closely",
                "Budget for future replacement",
                "Inspect jacket for damage"
            ])
        elif condition == ConditionCategory.GOOD:
            recommendations.append("Continue routine monitoring")
        else:
            recommendations.append("Maintain current inspection schedule")

        if change and change.get("r_value_change_percent", 0) < -5:
            recommendations.insert(0, "Investigate cause of accelerated degradation")

        return recommendations

    # =========================================================================
    # 5. TREND ANALYSIS
    # =========================================================================

    def analyze_trends(
        self,
        equipment_id: str,
        metric: str = "r_value",
        lookback_days: int = 365
    ) -> TrendAnalysisResult:
        """
        Analyze trends in performance metrics.

        Features:
        - Moving average calculation (7d, 30d, 90d)
        - Seasonal pattern detection
        - Anomaly detection
        - Rate of change trending
        - Forecasting

        Args:
            equipment_id: Equipment identifier
            metric: Metric to analyze ("r_value", "heat_loss", "efficiency", "temperature")
            lookback_days: Number of days to analyze

        Returns:
            TrendAnalysisResult with trend metrics
        """
        self._reset_calculation_state()

        history = self._performance_history.get(equipment_id, [])

        if len(history) < 7:
            raise ValueError(f"Insufficient data for trend analysis. Need at least 7 points, have {len(history)}")

        # Filter to lookback period
        cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)
        filtered = [dp for dp in history if dp.timestamp >= cutoff_date]

        if len(filtered) < 7:
            filtered = history[-30:]  # Use last 30 points if not enough in lookback

        # Extract metric values
        if metric == "r_value":
            values = [dp.r_value_m2k_w for dp in filtered]
        elif metric == "heat_loss":
            values = [dp.heat_loss_rate_w for dp in filtered]
        elif metric == "efficiency":
            values = [dp.efficiency_percent for dp in filtered]
        elif metric == "temperature":
            values = [dp.surface_temperature_c for dp in filtered]
        else:
            raise ValueError(f"Unknown metric: {metric}")

        timestamps = [dp.timestamp for dp in filtered]
        time_span = (timestamps[-1] - timestamps[0]).days

        # Calculate moving averages
        ma_7d = self._calculate_moving_average(values, timestamps, 7)
        ma_30d = self._calculate_moving_average(values, timestamps, 30)
        ma_90d = self._calculate_moving_average(values, timestamps, 90)

        self._add_calculation_step(
            description="Calculate moving averages",
            operation="moving_average",
            inputs={"data_points": len(values), "time_span_days": time_span},
            output_value=ma_30d,
            output_name="moving_average_30d"
        )

        # Determine trend direction
        trend_direction = self._determine_trend_direction(values, metric)

        # Calculate rate of change
        rate_of_change = self._calculate_rate_of_change(values, timestamps)

        self._add_calculation_step(
            description="Calculate rate of change",
            operation="rate_calculation",
            inputs={"start_value": values[0], "end_value": values[-1], "days": time_span},
            output_value=rate_of_change,
            output_name="rate_of_change_per_day"
        )

        # Seasonal pattern detection
        seasonal_detected, amplitude, period = self._detect_seasonal_pattern(values, timestamps)

        # Anomaly detection
        anomalies = self._detect_anomalies(values, timestamps)

        # Forecasting (simple linear extrapolation)
        forecast_30d = self._forecast_value(values, timestamps, 30)
        forecast_90d = self._forecast_value(values, timestamps, 90)

        self._add_calculation_step(
            description="Generate forecasts",
            operation="linear_extrapolation",
            inputs={"current_value": values[-1], "rate_of_change": rate_of_change},
            output_value=forecast_30d,
            output_name="forecast_30d"
        )

        provenance_hash = self._generate_provenance_hash({
            "equipment_id": equipment_id,
            "metric": metric,
            "data_points": len(values),
            "ma_30d": ma_30d
        })

        return TrendAnalysisResult(
            metric_name=metric,
            data_points_analyzed=len(values),
            time_span_days=time_span,
            moving_average_7d=self._round(ma_7d),
            moving_average_30d=self._round(ma_30d),
            moving_average_90d=self._round(ma_90d),
            trend_direction=trend_direction,
            rate_of_change_per_day=self._round(rate_of_change),
            seasonal_pattern_detected=seasonal_detected,
            seasonal_amplitude=self._round(amplitude) if amplitude else None,
            seasonal_period_days=period,
            anomalies_detected=anomalies,
            forecast_30d=self._round(forecast_30d),
            forecast_90d=self._round(forecast_90d),
            provenance_hash=provenance_hash
        )

    def _calculate_moving_average(
        self,
        values: List[float],
        timestamps: List[datetime],
        window_days: int
    ) -> float:
        """Calculate moving average over specified window."""
        if not values:
            return 0.0

        cutoff = timestamps[-1] - timedelta(days=window_days)
        window_values = [v for v, t in zip(values, timestamps) if t >= cutoff]

        if not window_values:
            return values[-1]

        return statistics.mean(window_values)

    def _determine_trend_direction(
        self,
        values: List[float],
        metric: str
    ) -> TrendDirection:
        """Determine overall trend direction."""
        if len(values) < 3:
            return TrendDirection.STABLE

        # Compare recent values to older values
        recent_avg = statistics.mean(values[-len(values)//3:])
        older_avg = statistics.mean(values[:len(values)//3])

        change_percent = ((recent_avg - older_avg) / older_avg * 100) if older_avg != 0 else 0

        # For R-value and efficiency, decreasing is degrading
        # For heat loss and temperature, increasing is degrading
        if metric in ["r_value", "efficiency"]:
            if change_percent > 5:
                return TrendDirection.IMPROVING
            elif change_percent < -10:
                return TrendDirection.ACCELERATING_DEGRADATION
            elif change_percent < -2:
                return TrendDirection.DEGRADING
            else:
                return TrendDirection.STABLE
        else:
            if change_percent < -5:
                return TrendDirection.IMPROVING
            elif change_percent > 10:
                return TrendDirection.ACCELERATING_DEGRADATION
            elif change_percent > 2:
                return TrendDirection.DEGRADING
            else:
                return TrendDirection.STABLE

    def _calculate_rate_of_change(
        self,
        values: List[float],
        timestamps: List[datetime]
    ) -> float:
        """Calculate rate of change per day."""
        if len(values) < 2:
            return 0.0

        days = (timestamps[-1] - timestamps[0]).days
        if days == 0:
            return 0.0

        return (values[-1] - values[0]) / days

    def _detect_seasonal_pattern(
        self,
        values: List[float],
        timestamps: List[datetime]
    ) -> Tuple[bool, Optional[float], Optional[int]]:
        """
        Detect seasonal patterns in data.

        Returns:
            Tuple of (pattern_detected, amplitude, period_days)
        """
        if len(values) < 30:
            return False, None, None

        # Simple approach: check for quarterly patterns
        # Group by month and look for repeating patterns
        monthly_values: Dict[int, List[float]] = {}
        for v, t in zip(values, timestamps):
            month = t.month
            if month not in monthly_values:
                monthly_values[month] = []
            monthly_values[month].append(v)

        if len(monthly_values) < 4:
            return False, None, None

        # Calculate monthly averages
        monthly_avgs = {m: statistics.mean(vals) for m, vals in monthly_values.items()}

        # Check variance between months
        avg_values = list(monthly_avgs.values())
        if len(avg_values) < 2:
            return False, None, None

        std_dev = statistics.stdev(avg_values)
        mean_val = statistics.mean(avg_values)

        # If coefficient of variation > 10%, consider it seasonal
        cv = (std_dev / mean_val * 100) if mean_val != 0 else 0

        if cv > 10:
            amplitude = max(avg_values) - min(avg_values)
            return True, amplitude, 365  # Annual pattern assumed

        return False, None, None

    def _detect_anomalies(
        self,
        values: List[float],
        timestamps: List[datetime]
    ) -> List[Dict[str, Any]]:
        """Detect anomalies using z-score method."""
        if len(values) < 10:
            return []

        mean_val = statistics.mean(values)
        std_dev = statistics.stdev(values)

        if std_dev == 0:
            return []

        anomalies = []
        for i, (v, t) in enumerate(zip(values, timestamps)):
            z_score = abs(v - mean_val) / std_dev
            if z_score > self.ANOMALY_ZSCORE_THRESHOLD:
                anomalies.append({
                    "index": i,
                    "timestamp": t.isoformat(),
                    "value": self._round(v),
                    "z_score": self._round(z_score),
                    "deviation_percent": self._round((v - mean_val) / mean_val * 100)
                })

        return anomalies

    def _forecast_value(
        self,
        values: List[float],
        timestamps: List[datetime],
        days_ahead: int
    ) -> float:
        """Forecast future value using linear extrapolation."""
        if len(values) < 2:
            return values[-1] if values else 0.0

        rate = self._calculate_rate_of_change(values, timestamps)
        return values[-1] + rate * days_ahead

    # =========================================================================
    # 6. FLEET/FACILITY BENCHMARKING
    # =========================================================================

    def benchmark_fleet_performance(
        self,
        equipment_ids: Optional[List[str]] = None,
        group_by: str = "type"
    ) -> FleetBenchmark:
        """
        Benchmark performance across fleet/facility.

        Features:
        - Statistical comparison across equipment
        - Age cohort analysis
        - Insulation type comparison
        - Location-based patterns
        - Best/worst performer identification

        Args:
            equipment_ids: List of equipment to include (None = all)
            group_by: Grouping for comparison ("type", "location", "age")

        Returns:
            FleetBenchmark with comparison metrics
        """
        self._reset_calculation_state()

        # Get equipment list
        if equipment_ids is None:
            equipment_ids = list(self._performance_history.keys())

        if len(equipment_ids) < 2:
            raise ValueError("Need at least 2 equipment items for benchmarking")

        # Collect current R-values for each equipment
        r_values: Dict[str, float] = {}
        ages: Dict[str, float] = {}

        for eq_id in equipment_ids:
            history = self._performance_history.get(eq_id, [])
            if history:
                r_values[eq_id] = history[-1].r_value_m2k_w
                # Calculate age from first measurement
                ages[eq_id] = (history[-1].timestamp - history[0].timestamp).days / 365.25

        if len(r_values) < 2:
            raise ValueError("Insufficient performance data for benchmarking")

        r_list = list(r_values.values())

        # Calculate statistics
        avg_r = statistics.mean(r_list)
        median_r = statistics.median(r_list)
        std_dev = statistics.stdev(r_list) if len(r_list) > 1 else 0.0

        sorted_r = sorted(r_list)
        n = len(sorted_r)
        p25 = sorted_r[int(n * 0.25)] if n >= 4 else sorted_r[0]
        p75 = sorted_r[int(n * 0.75)] if n >= 4 else sorted_r[-1]

        self._add_calculation_step(
            description="Calculate fleet statistics",
            operation="statistical_analysis",
            inputs={"equipment_count": len(r_values)},
            output_value=avg_r,
            output_name="fleet_average_r_value"
        )

        # Find best and worst performers
        best_id = max(r_values, key=r_values.get)
        worst_id = min(r_values, key=r_values.get)

        # Calculate average age
        avg_age = statistics.mean(list(ages.values())) if ages else 0.0

        # Calculate age-R correlation
        if len(ages) >= 3 and len(r_values) >= 3:
            age_correlation = self._calculate_correlation(
                [ages[eq_id] for eq_id in equipment_ids if eq_id in ages],
                [r_values[eq_id] for eq_id in equipment_ids if eq_id in r_values]
            )
        else:
            age_correlation = 0.0

        self._add_calculation_step(
            description="Calculate age-R correlation",
            operation="correlation_analysis",
            inputs={"data_pairs": min(len(ages), len(r_values))},
            output_value=age_correlation,
            output_name="age_r_correlation"
        )

        # Group comparisons
        type_comparison = self._compare_by_group(equipment_ids, r_values, "type")
        location_comparison = self._compare_by_group(equipment_ids, r_values, "location")

        # Generate recommendations
        recommendations = self._generate_fleet_recommendations(
            r_values, avg_r, std_dev, worst_id, age_correlation
        )

        provenance_hash = self._generate_provenance_hash({
            "fleet_size": len(equipment_ids),
            "average_r_value": avg_r,
            "best_performer": best_id
        })

        return FleetBenchmark(
            fleet_size=len(equipment_ids),
            equipment_ids=list(equipment_ids),
            average_r_value=self._round(avg_r),
            median_r_value=self._round(median_r),
            std_deviation=self._round(std_dev),
            percentile_25=self._round(p25),
            percentile_75=self._round(p75),
            best_performer_id=best_id,
            worst_performer_id=worst_id,
            average_age_years=self._round(avg_age),
            age_correlation=self._round(age_correlation),
            type_comparison=type_comparison,
            location_comparison=location_comparison,
            recommendations=recommendations,
            provenance_hash=provenance_hash
        )

    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 3:
            return 0.0

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)

        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _compare_by_group(
        self,
        equipment_ids: List[str],
        r_values: Dict[str, float],
        group_by: str
    ) -> Dict[str, Dict[str, float]]:
        """Compare R-values grouped by type or location."""
        groups: Dict[str, List[float]] = {}

        for eq_id in equipment_ids:
            if eq_id not in r_values:
                continue

            metadata = self._equipment_metadata.get(eq_id, {})
            group_key = metadata.get(group_by, "unknown")

            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(r_values[eq_id])

        comparison = {}
        for group_key, values in groups.items():
            if values:
                comparison[group_key] = {
                    "count": len(values),
                    "average": self._round(statistics.mean(values)),
                    "min": self._round(min(values)),
                    "max": self._round(max(values))
                }

        return comparison

    def _generate_fleet_recommendations(
        self,
        r_values: Dict[str, float],
        avg_r: float,
        std_dev: float,
        worst_id: str,
        age_correlation: float
    ) -> List[str]:
        """Generate fleet-wide recommendations."""
        recommendations = []

        # High variability
        cv = (std_dev / avg_r * 100) if avg_r > 0 else 0
        if cv > 25:
            recommendations.append(
                f"High variability in fleet performance (CV={cv:.1f}%). "
                "Standardize insulation specifications and installation procedures."
            )

        # Worst performer
        worst_r = r_values[worst_id]
        if worst_r < avg_r * 0.7:
            recommendations.append(
                f"Equipment {worst_id} significantly underperforming. "
                f"R-value {worst_r:.2f} is {((avg_r - worst_r) / avg_r * 100):.1f}% below fleet average."
            )

        # Age correlation
        if age_correlation < -0.7:
            recommendations.append(
                "Strong negative correlation between age and R-value detected. "
                "Consider proactive replacement program for aging insulation."
            )

        # General recommendations based on fleet average
        if avg_r < 2.0:  # Arbitrary threshold
            recommendations.append(
                "Fleet average R-value below industry benchmark. "
                "Evaluate insulation upgrade program."
            )

        if not recommendations:
            recommendations.append("Fleet performance within acceptable parameters. Continue routine monitoring.")

        return recommendations

    # =========================================================================
    # 7. PREDICTIVE ANALYTICS
    # =========================================================================

    def forecast_future_condition(
        self,
        equipment_id: str,
        forecast_horizon_days: int = 365,
        energy_cost_per_kwh: Optional[float] = None
    ) -> FutureConditionForecast:
        """
        Forecast future condition with predictive analytics.

        Features:
        - R-value prediction
        - Heat loss prediction
        - Maintenance trigger prediction
        - Budget forecasting
        - Risk scoring

        Args:
            equipment_id: Equipment identifier
            forecast_horizon_days: Days to forecast
            energy_cost_per_kwh: Energy cost for budget calculations

        Returns:
            FutureConditionForecast with predictions
        """
        self._reset_calculation_state()

        history = self._performance_history.get(equipment_id, [])

        if len(history) < 5:
            raise ValueError(f"Insufficient data for forecasting. Need at least 5 points, have {len(history)}")

        energy_cost = Decimal(str(energy_cost_per_kwh)) if energy_cost_per_kwh else self.energy_cost

        # Get degradation analysis
        degradation = self.calculate_degradation_rate(equipment_id, DegradationModel.EXPONENTIAL)

        # Get current values
        current = history[-1]
        current_r = Decimal(str(current.r_value_m2k_w))
        current_heat_loss = Decimal(str(current.heat_loss_rate_w))
        current_efficiency = Decimal(str(current.efficiency_percent))

        # Calculate forecast date
        forecast_date = current.timestamp + timedelta(days=forecast_horizon_days)

        # Predict R-value using degradation model
        r_0 = Decimal(str(degradation.initial_r_value))
        lambda_val = Decimal(str(degradation.degradation_rate))

        # Current age in years
        current_age_years = (current.timestamp - history[0].timestamp).days / 365.25
        future_age_years = current_age_years + (forecast_horizon_days / 365.25)

        # R(t) = R_0 * exp(-lambda * t)
        predicted_r = r_0 * Decimal(str(math.exp(-float(lambda_val) * future_age_years)))

        self._add_calculation_step(
            description="Predict future R-value",
            operation="exponential_decay",
            inputs={
                "R_0": float(r_0),
                "lambda": float(lambda_val),
                "future_age_years": future_age_years
            },
            output_value=float(predicted_r),
            output_name="predicted_r_value",
            formula="R(t) = R_0 * exp(-lambda*t)"
        )

        # Predict heat loss (inversely proportional to R-value)
        if predicted_r > 0:
            r_ratio = current_r / predicted_r
            predicted_heat_loss = current_heat_loss * r_ratio
        else:
            predicted_heat_loss = current_heat_loss * Decimal('2')

        # Predict efficiency
        predicted_efficiency = max(
            Decimal('0'),
            current_efficiency - (current_efficiency - Decimal('50')) * (Decimal('1') - predicted_r / current_r)
        ) if current_r > 0 else Decimal('50')

        self._add_calculation_step(
            description="Predict heat loss and efficiency",
            operation="proportional_scaling",
            inputs={
                "current_heat_loss": float(current_heat_loss),
                "r_value_ratio": float(current_r / predicted_r) if predicted_r > 0 else 2.0
            },
            output_value=float(predicted_heat_loss),
            output_name="predicted_heat_loss"
        )

        # Confidence intervals based on R-squared
        uncertainty = Decimal(str(1 - degradation.r_squared))
        ci_lower = float(predicted_r * (Decimal('1') - uncertainty))
        ci_upper = float(predicted_r * (Decimal('1') + uncertainty))

        # Maintenance trigger prediction
        # Trigger when R-value drops below 60% of design
        maintenance_threshold = r_0 * Decimal('0.6')
        if predicted_r < maintenance_threshold and current_r >= maintenance_threshold:
            # Calculate when it crosses threshold
            if lambda_val > 0:
                time_to_threshold = -math.log(float(maintenance_threshold / r_0)) / float(lambda_val) - current_age_years
                maintenance_date = current.timestamp + timedelta(days=time_to_threshold * 365.25)
            else:
                maintenance_date = None
        elif predicted_r < maintenance_threshold:
            maintenance_date = current.timestamp  # Already needs maintenance
        else:
            maintenance_date = None

        # Replacement trigger prediction (R < 40% of design)
        replacement_threshold = r_0 * Decimal('0.4')
        if lambda_val > 0:
            time_to_replacement = -math.log(float(replacement_threshold / r_0)) / float(lambda_val) - current_age_years
            if time_to_replacement > 0:
                replacement_date = current.timestamp + timedelta(days=time_to_replacement * 365.25)
            else:
                replacement_date = current.timestamp
        else:
            replacement_date = None

        # Cost calculations
        hours_in_period = Decimal(str(forecast_horizon_days * 24))
        avg_heat_loss = (current_heat_loss + predicted_heat_loss) / Decimal('2')
        energy_lost_kwh = avg_heat_loss * hours_in_period / Decimal('1000')
        cost_to_date = energy_lost_kwh * energy_cost

        # Budget forecasting (1 year and 5 year)
        budget_1y = float(energy_lost_kwh * Decimal('365.25') / Decimal(str(forecast_horizon_days)) * energy_cost)
        budget_5y = budget_1y * 5 * 1.1  # 10% inflation factor

        self._add_calculation_step(
            description="Calculate energy loss costs",
            operation="cost_calculation",
            inputs={
                "avg_heat_loss_w": float(avg_heat_loss),
                "hours": float(hours_in_period),
                "energy_cost": float(energy_cost)
            },
            output_value=float(cost_to_date),
            output_name="estimated_cost_to_date",
            units="$"
        )

        # Risk scoring (0-100)
        risk_factors = []

        # R-value risk (0-40)
        r_pct_of_design = float(predicted_r / r_0) * 100 if r_0 > 0 else 0
        r_risk = max(0, 40 - r_pct_of_design * 0.4)
        risk_factors.append(r_risk)

        # Degradation rate risk (0-30)
        annual_deg = degradation.annual_degradation_percent
        deg_risk = min(30, annual_deg * 3)
        risk_factors.append(deg_risk)

        # Acceleration risk (0-30)
        if degradation.is_accelerating:
            accel_risk = min(30, degradation.acceleration_factor * 15)
        else:
            accel_risk = 0
        risk_factors.append(accel_risk)

        risk_score = sum(risk_factors)

        self._add_calculation_step(
            description="Calculate risk score",
            operation="risk_assessment",
            inputs={
                "r_value_risk": r_risk,
                "degradation_risk": deg_risk,
                "acceleration_risk": accel_risk
            },
            output_value=risk_score,
            output_name="risk_score"
        )

        provenance_hash = self._generate_provenance_hash({
            "equipment_id": equipment_id,
            "forecast_horizon_days": forecast_horizon_days,
            "predicted_r_value": float(predicted_r),
            "risk_score": risk_score
        })

        return FutureConditionForecast(
            equipment_id=equipment_id,
            forecast_date=forecast_date,
            forecast_horizon_days=forecast_horizon_days,
            predicted_r_value=self._round(float(predicted_r)),
            predicted_heat_loss_w=self._round(float(predicted_heat_loss)),
            predicted_efficiency_percent=self._round(float(predicted_efficiency)),
            confidence_interval_r_lower=self._round(ci_lower),
            confidence_interval_r_upper=self._round(ci_upper),
            maintenance_trigger_date=maintenance_date,
            replacement_trigger_date=replacement_date,
            estimated_cost_to_date=self._round(float(cost_to_date)),
            budget_forecast_1y=self._round(budget_1y),
            budget_forecast_5y=self._round(budget_5y),
            risk_score=self._round(risk_score),
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # 8. KPI DASHBOARD
    # =========================================================================

    def generate_kpi_dashboard(
        self,
        reporting_period_days: int = 30,
        equipment_ids: Optional[List[str]] = None
    ) -> KPIDashboard:
        """
        Generate KPI dashboard metrics.

        Features:
        - Overall insulation health index
        - Condition distribution
        - Total heat loss trend
        - Inspection completion rate
        - Repair completion rate
        - Active alerts summary
        - Cost metrics

        Args:
            reporting_period_days: Days to include in reporting period
            equipment_ids: Equipment to include (None = all)

        Returns:
            KPIDashboard with comprehensive metrics
        """
        self._reset_calculation_state()

        if equipment_ids is None:
            equipment_ids = list(self._performance_history.keys())

        if not equipment_ids:
            raise ValueError("No equipment data available for dashboard")

        generated_at = datetime.utcnow()
        period_end = generated_at
        period_start = generated_at - timedelta(days=reporting_period_days)

        # Initialize counters
        condition_dist: Dict[str, int] = {c.value: 0 for c in ConditionCategory}
        total_heat_loss = Decimal('0')
        total_heat_loss_prev = Decimal('0')
        r_values: List[float] = []
        health_scores: List[float] = []

        inspections_completed = 0
        inspections_scheduled = len(equipment_ids)  # Assume all should be inspected
        repairs_completed = 0
        repairs_pending = 0
        repair_times: List[int] = []

        active_alerts: List[Dict] = []
        alerts_by_severity: Dict[str, int] = {s.value: 0 for s in AlertSeverity}

        energy_loss_cost = Decimal('0')
        repair_cost = Decimal('0')

        equipment_needing_attention = 0

        for eq_id in equipment_ids:
            history = self._performance_history.get(eq_id, [])
            inspections = self._inspection_records.get(eq_id, [])

            if not history:
                continue

            current = history[-1]
            r_values.append(current.r_value_m2k_w)

            # Calculate condition
            metadata = self._equipment_metadata.get(eq_id, {})
            design_r = metadata.get("design_r_value", current.r_value_m2k_w * 1.5)
            r_ratio = current.r_value_m2k_w / design_r if design_r > 0 else 1.0

            if r_ratio > 0.90:
                condition = ConditionCategory.EXCELLENT
                health_score = 100
            elif r_ratio > 0.75:
                condition = ConditionCategory.GOOD
                health_score = 80
            elif r_ratio > 0.50:
                condition = ConditionCategory.FAIR
                health_score = 60
            elif r_ratio > 0.25:
                condition = ConditionCategory.POOR
                health_score = 30
            else:
                condition = ConditionCategory.FAILED
                health_score = 0

            condition_dist[condition.value] += 1
            health_scores.append(health_score)

            # Heat loss
            total_heat_loss += Decimal(str(current.heat_loss_rate_w))

            # Previous period heat loss
            prev_period_data = [dp for dp in history if period_start - timedelta(days=reporting_period_days) <= dp.timestamp < period_start]
            if prev_period_data:
                total_heat_loss_prev += Decimal(str(prev_period_data[-1].heat_loss_rate_w))

            # Inspections in period
            period_inspections = [i for i in inspections if period_start <= i.inspection_date <= period_end]
            inspections_completed += len(period_inspections)

            # Collect alerts
            if inspections:
                latest = inspections[-1]
                for alert in latest.alerts:
                    severity = alert.get("severity", "info")
                    alerts_by_severity[severity] = alerts_by_severity.get(severity, 0) + 1
                    if severity in ["critical", "emergency"]:
                        active_alerts.append({
                            "equipment_id": eq_id,
                            "alert": alert
                        })

            # Cost calculations
            hours_in_period = reporting_period_days * 24
            energy_kwh = Decimal(str(current.heat_loss_rate_w)) * Decimal(str(hours_in_period)) / Decimal('1000')
            energy_loss_cost += energy_kwh * self.energy_cost

            # Equipment needing attention (forecast)
            if condition in [ConditionCategory.POOR, ConditionCategory.FAILED]:
                equipment_needing_attention += 1
            elif len(history) >= 3:
                try:
                    forecast = self.forecast_future_condition(eq_id, 30)
                    if forecast.risk_score > 70:
                        equipment_needing_attention += 1
                except (ValueError, Exception):
                    pass

        # Calculate overall health index
        overall_health = statistics.mean(health_scores) if health_scores else 0.0

        # Condition distribution percentages
        total_equipment = len(equipment_ids)
        condition_dist_pct = {
            k: (v / total_equipment * 100) if total_equipment > 0 else 0.0
            for k, v in condition_dist.items()
        }

        # Heat loss trend
        heat_loss_change_pct = 0.0
        if total_heat_loss_prev > 0:
            heat_loss_change_pct = float(
                (total_heat_loss - total_heat_loss_prev) / total_heat_loss_prev * 100
            )

        # R-value trend (30-day)
        fleet_avg_r = statistics.mean(r_values) if r_values else 0.0
        r_value_trend = 0.0  # Would need historical fleet data

        # Inspection and repair rates
        inspection_rate = (inspections_completed / inspections_scheduled * 100) if inspections_scheduled > 0 else 0.0
        repair_rate = (repairs_completed / (repairs_completed + repairs_pending) * 100) if (repairs_completed + repairs_pending) > 0 else 100.0
        avg_repair_time = statistics.mean(repair_times) if repair_times else 0.0

        # Total cost
        total_cost = energy_loss_cost + repair_cost

        # Cost trend (placeholder - would need historical data)
        cost_trend = 0.0

        # Estimated next quarter cost
        estimated_next_quarter = float(energy_loss_cost) * 3  # Simple projection

        self._add_calculation_step(
            description="Generate KPI dashboard",
            operation="dashboard_generation",
            inputs={
                "equipment_count": len(equipment_ids),
                "reporting_period_days": reporting_period_days
            },
            output_value=overall_health,
            output_name="overall_health_index"
        )

        provenance_hash = self._generate_provenance_hash({
            "dashboard_type": "kpi",
            "equipment_count": len(equipment_ids),
            "period_days": reporting_period_days,
            "overall_health": overall_health
        })

        return KPIDashboard(
            dashboard_id=f"KPI-{generated_at.strftime('%Y%m%d%H%M%S')}",
            generated_at=generated_at,
            reporting_period_start=period_start,
            reporting_period_end=period_end,
            overall_health_index=self._round(overall_health),
            equipment_count=len(equipment_ids),
            condition_distribution=condition_dist,
            condition_distribution_percent={k: self._round(v) for k, v in condition_dist_pct.items()},
            total_heat_loss_w=self._round(float(total_heat_loss)),
            total_heat_loss_change_percent=self._round(heat_loss_change_pct),
            heat_loss_by_area={},  # Would need area grouping
            fleet_average_r_value=self._round(fleet_avg_r),
            r_value_trend_30d=self._round(r_value_trend),
            inspections_completed=inspections_completed,
            inspections_scheduled=inspections_scheduled,
            inspection_completion_rate=self._round(inspection_rate),
            repairs_completed=repairs_completed,
            repairs_pending=repairs_pending,
            repair_completion_rate=self._round(repair_rate),
            average_repair_time_days=self._round(avg_repair_time),
            energy_loss_cost_period=self._round(float(energy_loss_cost)),
            repair_cost_period=self._round(float(repair_cost)),
            total_cost_period=self._round(float(total_cost)),
            cost_trend_percent=self._round(cost_trend),
            active_alerts_count=len(active_alerts),
            alerts_by_severity=alerts_by_severity,
            equipment_requiring_attention_30d=equipment_needing_attention,
            estimated_cost_next_quarter=self._round(estimated_next_quarter),
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def set_equipment_metadata(
        self,
        equipment_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Set metadata for equipment.

        Args:
            equipment_id: Equipment identifier
            metadata: Metadata dictionary with keys like:
                - design_r_value
                - type (insulation type)
                - location
                - installation_date
                - manufacturer
        """
        self._equipment_metadata[equipment_id] = metadata

    def _reset_calculation_state(self) -> None:
        """Reset calculation tracking state."""
        self._calculation_steps = []
        self._step_counter = 0

    def _add_calculation_step(
        self,
        description: str,
        operation: str,
        inputs: Dict[str, Any],
        output_value: Any,
        output_name: str,
        formula: Optional[str] = None,
        units: Optional[str] = None
    ) -> None:
        """Record a calculation step for audit trail."""
        self._step_counter += 1
        step = CalculationStep(
            step_number=self._step_counter,
            description=description,
            operation=operation,
            inputs=inputs,
            output_value=output_value,
            output_name=output_name,
            formula=formula,
            units=units
        )
        self._calculation_steps.append(step)

    def _generate_provenance_hash(self, data: Dict[str, Any]) -> str:
        """Generate SHA-256 provenance hash."""
        data_with_version = {
            "calculator": "InsulationPerformanceTracker",
            "version": self.VERSION,
            **data
        }
        canonical_json = json.dumps(data_with_version, sort_keys=True, separators=(',', ':'), default=str)
        return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()

    def _round(self, value: float) -> float:
        """Round value to configured precision."""
        if value is None or math.isinf(value) or math.isnan(value):
            return 0.0
        decimal_value = Decimal(str(value))
        quantize_str = '0.' + '0' * self.precision
        rounded = decimal_value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)
        return float(rounded)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Main class
    "InsulationPerformanceTracker",

    # Enumerations
    "DegradationModel",
    "ConditionCategory",
    "AlertSeverity",
    "TrendDirection",
    "InspectionType",

    # Data models
    "PerformanceDataPoint",
    "DegradationAnalysis",
    "RemainingUsefulLife",
    "InspectionRecord",
    "TrendAnalysisResult",
    "FleetBenchmark",
    "FutureConditionForecast",
    "KPIDashboard",
    "CalculationStep",
]
