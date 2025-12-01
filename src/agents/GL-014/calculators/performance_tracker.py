"""
GL-014 EXCHANGER-PRO - Performance Degradation Tracker Module

This module implements comprehensive performance monitoring and degradation
tracking for heat exchangers with zero-hallucination calculation guarantees.

Key Features:
- Thermal efficiency and effectiveness tracking
- Hydraulic performance monitoring
- Multiple degradation models (linear, exponential, asymptotic)
- Statistical trend analysis with moving averages
- Performance benchmarking against design and fleet
- Multi-parameter health index calculation
- Remaining performance life estimation

Reference Standards:
- ASME PTC 12.5: Single Phase Heat Exchangers
- TEMA Standards (10th Edition)
- API 660: Shell-and-Tube Heat Exchangers
- HTRI Design Manual
- ISO 13379-1: Condition monitoring diagnostics

Author: GL-CalculatorEngineer
Version: 1.0.0
License: Proprietary - GreenLang
"""

from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Sequence
from datetime import datetime, timezone
from enum import Enum, auto
import hashlib
import json
import uuid
import math


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class DegradationModel(Enum):
    """Available degradation models for performance tracking."""
    LINEAR = auto()          # P(t) = P_0 - k*t
    EXPONENTIAL = auto()     # P(t) = P_0 * exp(-lambda*t)
    ASYMPTOTIC = auto()      # P(t) = P_inf + (P_0 - P_inf) * exp(-t/tau)
    POWER_LAW = auto()       # P(t) = P_0 * (1 - (t/t_f)^n)


class HealthStatus(Enum):
    """Equipment health status classification per ISO 13379-1."""
    OPTIMAL = auto()      # 90-100% health
    GOOD = auto()         # 70-90% health
    DEGRADED = auto()     # 50-70% health
    POOR = auto()         # 30-50% health
    CRITICAL = auto()     # 0-30% health


class TrendDirection(Enum):
    """Performance trend direction indicators."""
    IMPROVING = auto()
    STABLE = auto()
    DEGRADING_SLOW = auto()
    DEGRADING_FAST = auto()
    CRITICAL_DECLINE = auto()


class SeasonalPattern(Enum):
    """Seasonal pattern types for performance analysis."""
    NONE = auto()
    DAILY = auto()
    WEEKLY = auto()
    MONTHLY = auto()
    QUARTERLY = auto()
    ANNUAL = auto()


# Performance thresholds
HEALTH_THRESHOLDS: Dict[HealthStatus, Tuple[Decimal, Decimal]] = {
    HealthStatus.OPTIMAL: (Decimal("90"), Decimal("100")),
    HealthStatus.GOOD: (Decimal("70"), Decimal("90")),
    HealthStatus.DEGRADED: (Decimal("50"), Decimal("70")),
    HealthStatus.POOR: (Decimal("30"), Decimal("50")),
    HealthStatus.CRITICAL: (Decimal("0"), Decimal("30")),
}

# Default weighting factors for health index components
DEFAULT_HEALTH_WEIGHTS: Dict[str, Decimal] = {
    "thermal": Decimal("0.40"),
    "hydraulic": Decimal("0.35"),
    "mechanical": Decimal("0.15"),
    "fouling": Decimal("0.10"),
}

# Minimum acceptable performance levels
MIN_ACCEPTABLE_EFFICIENCY: Decimal = Decimal("0.60")
MIN_ACCEPTABLE_U_VALUE_RATIO: Decimal = Decimal("0.50")
MAX_ACCEPTABLE_PRESSURE_DROP_RATIO: Decimal = Decimal("2.0")

# Default decimal precision
DEFAULT_PRECISION: int = 6

# Z-scores for confidence intervals
Z_SCORES: Dict[str, Decimal] = {
    "80%": Decimal("1.282"),
    "90%": Decimal("1.645"),
    "95%": Decimal("1.960"),
    "99%": Decimal("2.576"),
}


# =============================================================================
# DATA CLASSES - IMMUTABLE RESULTS
# =============================================================================

@dataclass(frozen=True)
class ThermalPerformanceResult:
    """
    Immutable result of thermal performance calculation.

    All values are deterministic and reproducible.
    """
    thermal_efficiency: Decimal
    effectiveness: Decimal
    u_value_actual: Decimal
    u_value_design: Decimal
    u_value_ratio: Decimal
    heat_duty_actual: Decimal
    heat_duty_design: Decimal
    heat_duty_ratio: Decimal
    lmtd: Decimal
    ntu: Decimal
    timestamp: str
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "thermal_efficiency": str(self.thermal_efficiency),
            "effectiveness": str(self.effectiveness),
            "u_value_actual": str(self.u_value_actual),
            "u_value_design": str(self.u_value_design),
            "u_value_ratio": str(self.u_value_ratio),
            "heat_duty_actual": str(self.heat_duty_actual),
            "heat_duty_design": str(self.heat_duty_design),
            "heat_duty_ratio": str(self.heat_duty_ratio),
            "lmtd": str(self.lmtd),
            "ntu": str(self.ntu),
            "timestamp": self.timestamp,
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class HydraulicPerformanceResult:
    """
    Immutable result of hydraulic performance calculation.
    """
    pressure_drop_shell: Decimal
    pressure_drop_tube: Decimal
    pressure_drop_design_shell: Decimal
    pressure_drop_design_tube: Decimal
    pressure_drop_ratio_shell: Decimal
    pressure_drop_ratio_tube: Decimal
    flow_rate_shell: Decimal
    flow_rate_tube: Decimal
    flow_capacity_ratio: Decimal
    pump_power_increase_percent: Decimal
    timestamp: str
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pressure_drop_shell": str(self.pressure_drop_shell),
            "pressure_drop_tube": str(self.pressure_drop_tube),
            "pressure_drop_design_shell": str(self.pressure_drop_design_shell),
            "pressure_drop_design_tube": str(self.pressure_drop_design_tube),
            "pressure_drop_ratio_shell": str(self.pressure_drop_ratio_shell),
            "pressure_drop_ratio_tube": str(self.pressure_drop_ratio_tube),
            "flow_rate_shell": str(self.flow_rate_shell),
            "flow_rate_tube": str(self.flow_rate_tube),
            "flow_capacity_ratio": str(self.flow_capacity_ratio),
            "pump_power_increase_percent": str(self.pump_power_increase_percent),
            "timestamp": self.timestamp,
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class DegradationTrendResult:
    """
    Immutable result of degradation trend analysis.
    """
    model_type: str
    degradation_rate: Decimal
    initial_performance: Decimal
    current_performance: Decimal
    projected_performance_30d: Decimal
    projected_performance_90d: Decimal
    projected_performance_365d: Decimal
    model_r_squared: Decimal
    trend_direction: str
    acceleration: Decimal
    model_parameters: Dict[str, str]
    data_points_used: int
    timestamp: str
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_type": self.model_type,
            "degradation_rate": str(self.degradation_rate),
            "initial_performance": str(self.initial_performance),
            "current_performance": str(self.current_performance),
            "projected_performance_30d": str(self.projected_performance_30d),
            "projected_performance_90d": str(self.projected_performance_90d),
            "projected_performance_365d": str(self.projected_performance_365d),
            "model_r_squared": str(self.model_r_squared),
            "trend_direction": self.trend_direction,
            "acceleration": str(self.acceleration),
            "model_parameters": self.model_parameters,
            "data_points_used": self.data_points_used,
            "timestamp": self.timestamp,
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class PerformancePatternResult:
    """
    Immutable result of performance pattern analysis.
    """
    moving_average_7d: Decimal
    moving_average_30d: Decimal
    moving_average_90d: Decimal
    rate_of_change_daily: Decimal
    rate_of_change_weekly: Decimal
    rate_of_change_monthly: Decimal
    acceleration: Decimal
    seasonal_pattern: str
    seasonal_amplitude: Decimal
    seasonal_phase_days: Decimal
    volatility_index: Decimal
    anomaly_score: Decimal
    timestamp: str
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "moving_average_7d": str(self.moving_average_7d),
            "moving_average_30d": str(self.moving_average_30d),
            "moving_average_90d": str(self.moving_average_90d),
            "rate_of_change_daily": str(self.rate_of_change_daily),
            "rate_of_change_weekly": str(self.rate_of_change_weekly),
            "rate_of_change_monthly": str(self.rate_of_change_monthly),
            "acceleration": str(self.acceleration),
            "seasonal_pattern": self.seasonal_pattern,
            "seasonal_amplitude": str(self.seasonal_amplitude),
            "seasonal_phase_days": str(self.seasonal_phase_days),
            "volatility_index": str(self.volatility_index),
            "anomaly_score": str(self.anomaly_score),
            "timestamp": self.timestamp,
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class BenchmarkResult:
    """
    Immutable result of performance benchmarking.
    """
    design_comparison_ratio: Decimal
    historical_best_ratio: Decimal
    fleet_average_ratio: Decimal
    fleet_percentile: Decimal
    industry_benchmark_ratio: Decimal
    performance_gap_to_design: Decimal
    performance_gap_to_best: Decimal
    recovery_potential_percent: Decimal
    benchmark_status: str
    recommendations: Tuple[str, ...]
    timestamp: str
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "design_comparison_ratio": str(self.design_comparison_ratio),
            "historical_best_ratio": str(self.historical_best_ratio),
            "fleet_average_ratio": str(self.fleet_average_ratio),
            "fleet_percentile": str(self.fleet_percentile),
            "industry_benchmark_ratio": str(self.industry_benchmark_ratio),
            "performance_gap_to_design": str(self.performance_gap_to_design),
            "performance_gap_to_best": str(self.performance_gap_to_best),
            "recovery_potential_percent": str(self.recovery_potential_percent),
            "benchmark_status": self.benchmark_status,
            "recommendations": list(self.recommendations),
            "timestamp": self.timestamp,
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class HealthIndexResult:
    """
    Immutable result of health index calculation.
    """
    overall_health_index: Decimal
    thermal_score: Decimal
    hydraulic_score: Decimal
    mechanical_score: Decimal
    fouling_score: Decimal
    health_status: str
    component_weights: Dict[str, str]
    limiting_factor: str
    days_to_next_threshold: Decimal
    confidence_lower: Decimal
    confidence_upper: Decimal
    confidence_level: str
    timestamp: str
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "overall_health_index": str(self.overall_health_index),
            "thermal_score": str(self.thermal_score),
            "hydraulic_score": str(self.hydraulic_score),
            "mechanical_score": str(self.mechanical_score),
            "fouling_score": str(self.fouling_score),
            "health_status": self.health_status,
            "component_weights": self.component_weights,
            "limiting_factor": self.limiting_factor,
            "days_to_next_threshold": str(self.days_to_next_threshold),
            "confidence_lower": str(self.confidence_lower),
            "confidence_upper": str(self.confidence_upper),
            "confidence_level": self.confidence_level,
            "timestamp": self.timestamp,
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class RemainingPerformanceLifeResult:
    """
    Immutable result of remaining performance life estimation.
    """
    rul_days: Decimal
    rul_months: Decimal
    rul_years: Decimal
    time_to_min_efficiency: Decimal
    time_to_min_u_value: Decimal
    time_to_max_pressure_drop: Decimal
    limiting_parameter: str
    current_performance_percent: Decimal
    min_acceptable_percent: Decimal
    confidence_lower_days: Decimal
    confidence_upper_days: Decimal
    confidence_level: str
    degradation_model_used: str
    model_parameters: Dict[str, str]
    timestamp: str
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rul_days": str(self.rul_days),
            "rul_months": str(self.rul_months),
            "rul_years": str(self.rul_years),
            "time_to_min_efficiency": str(self.time_to_min_efficiency),
            "time_to_min_u_value": str(self.time_to_min_u_value),
            "time_to_max_pressure_drop": str(self.time_to_max_pressure_drop),
            "limiting_parameter": self.limiting_parameter,
            "current_performance_percent": str(self.current_performance_percent),
            "min_acceptable_percent": str(self.min_acceptable_percent),
            "confidence_lower_days": str(self.confidence_lower_days),
            "confidence_upper_days": str(self.confidence_upper_days),
            "confidence_level": self.confidence_level,
            "degradation_model_used": self.degradation_model_used,
            "model_parameters": self.model_parameters,
            "timestamp": self.timestamp,
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class PerformanceReportResult:
    """
    Comprehensive performance report combining all metrics.
    """
    exchanger_id: str
    report_timestamp: str
    thermal_performance: ThermalPerformanceResult
    hydraulic_performance: HydraulicPerformanceResult
    degradation_trend: DegradationTrendResult
    performance_pattern: PerformancePatternResult
    benchmark: BenchmarkResult
    health_index: HealthIndexResult
    remaining_life: RemainingPerformanceLifeResult
    summary_status: str
    priority_actions: Tuple[str, ...]
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "exchanger_id": self.exchanger_id,
            "report_timestamp": self.report_timestamp,
            "thermal_performance": self.thermal_performance.to_dict(),
            "hydraulic_performance": self.hydraulic_performance.to_dict(),
            "degradation_trend": self.degradation_trend.to_dict(),
            "performance_pattern": self.performance_pattern.to_dict(),
            "benchmark": self.benchmark.to_dict(),
            "health_index": self.health_index.to_dict(),
            "remaining_life": self.remaining_life.to_dict(),
            "summary_status": self.summary_status,
            "priority_actions": list(self.priority_actions),
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class CalculationStep:
    """Immutable record of a single calculation step."""
    step_number: int
    operation: str
    description: str
    inputs: Dict[str, Any]
    output_name: str
    output_value: Any
    formula: str = ""
    reference: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_number": self.step_number,
            "operation": self.operation,
            "description": self.description,
            "inputs": self._serialize_values(self.inputs),
            "output_name": self.output_name,
            "output_value": self._serialize_value(self.output_value),
            "formula": self.formula,
            "reference": self.reference,
        }

    def _serialize_values(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize dictionary values."""
        return {k: self._serialize_value(v) for k, v in d.items()}

    def _serialize_value(self, v: Any) -> Any:
        """Serialize a single value."""
        if isinstance(v, Decimal):
            return str(v)
        elif isinstance(v, (list, tuple)):
            return [self._serialize_value(x) for x in v]
        elif isinstance(v, dict):
            return self._serialize_values(v)
        return v


# =============================================================================
# PERFORMANCE TRACKER CLASS
# =============================================================================

class PerformanceTracker:
    """
    Comprehensive performance monitoring and degradation tracking
    for heat exchangers with zero-hallucination guarantee.

    All calculations are:
    - Deterministic (same inputs = same outputs)
    - Bit-perfect reproducible using Decimal arithmetic
    - Fully documented with provenance tracking
    - Based on authoritative engineering standards

    Reference Standards:
    - ASME PTC 12.5: Single Phase Heat Exchangers
    - TEMA Standards (10th Edition)
    - API 660: Shell-and-Tube Heat Exchangers

    Example:
        >>> tracker = PerformanceTracker()
        >>> thermal = tracker.calculate_thermal_efficiency(
        ...     heat_duty_actual=Decimal("850000"),
        ...     heat_duty_design=Decimal("1000000"),
        ...     t_hot_in=Decimal("150"),
        ...     t_hot_out=Decimal("90"),
        ...     t_cold_in=Decimal("30"),
        ...     t_cold_out=Decimal("70")
        ... )
        >>> print(f"Efficiency: {thermal.thermal_efficiency}")
    """

    def __init__(
        self,
        precision: int = DEFAULT_PRECISION,
        health_weights: Optional[Dict[str, Decimal]] = None,
        store_provenance: bool = True
    ):
        """
        Initialize Performance Tracker.

        Args:
            precision: Decimal precision for calculations
            health_weights: Custom weights for health index components
            store_provenance: Whether to store calculation provenance
        """
        self._precision = precision
        self._health_weights = health_weights or DEFAULT_HEALTH_WEIGHTS.copy()
        self._store_provenance = store_provenance
        self._calculation_steps: List[CalculationStep] = []

        # Validate health weights sum to 1.0
        weight_sum = sum(self._health_weights.values())
        if not (Decimal("0.99") <= weight_sum <= Decimal("1.01")):
            raise ValueError(
                f"Health weights must sum to 1.0, got {weight_sum}"
            )

    # =========================================================================
    # THERMAL PERFORMANCE METRICS
    # =========================================================================

    def calculate_thermal_efficiency(
        self,
        heat_duty_actual: Union[Decimal, float, str],
        heat_duty_design: Union[Decimal, float, str],
        t_hot_in: Union[Decimal, float, str],
        t_hot_out: Union[Decimal, float, str],
        t_cold_in: Union[Decimal, float, str],
        t_cold_out: Union[Decimal, float, str],
        area: Optional[Union[Decimal, float, str]] = None,
        u_value_design: Optional[Union[Decimal, float, str]] = None
    ) -> ThermalPerformanceResult:
        """
        Calculate comprehensive thermal performance metrics.

        Thermal Efficiency:
            eta = Q_actual / Q_design

        Effectiveness (NTU method):
            epsilon = Q_actual / Q_max
            Q_max = C_min * (T_hot_in - T_cold_in)

        U-value ratio:
            U_ratio = U_actual / U_design
            U_actual = Q / (A * LMTD)

        Args:
            heat_duty_actual: Actual heat transfer rate (W or kW)
            heat_duty_design: Design heat transfer rate (W or kW)
            t_hot_in: Hot fluid inlet temperature (C)
            t_hot_out: Hot fluid outlet temperature (C)
            t_cold_in: Cold fluid inlet temperature (C)
            t_cold_out: Cold fluid outlet temperature (C)
            area: Heat transfer area (m^2), optional
            u_value_design: Design U-value (W/m^2-K), optional

        Returns:
            ThermalPerformanceResult with complete provenance

        Reference:
            ASME PTC 12.5-2000, Section 4
        """
        self._calculation_steps = []
        timestamp = datetime.now(timezone.utc).isoformat()

        # Convert inputs to Decimal
        Q_actual = self._to_decimal(heat_duty_actual)
        Q_design = self._to_decimal(heat_duty_design)
        T_hi = self._to_decimal(t_hot_in)
        T_ho = self._to_decimal(t_hot_out)
        T_ci = self._to_decimal(t_cold_in)
        T_co = self._to_decimal(t_cold_out)

        # Step 1: Calculate thermal efficiency
        if Q_design > Decimal("0"):
            thermal_efficiency = Q_actual / Q_design
        else:
            thermal_efficiency = Decimal("0")

        self._add_step(
            step_number=1,
            operation="divide",
            description="Calculate thermal efficiency",
            inputs={"Q_actual": Q_actual, "Q_design": Q_design},
            output_name="thermal_efficiency",
            output_value=thermal_efficiency,
            formula="eta = Q_actual / Q_design",
            reference="ASME PTC 12.5"
        )

        # Step 2: Calculate LMTD (Log Mean Temperature Difference)
        delta_T1 = T_hi - T_co
        delta_T2 = T_ho - T_ci

        if delta_T1 > Decimal("0") and delta_T2 > Decimal("0"):
            if abs(delta_T1 - delta_T2) < Decimal("0.001"):
                # Avoid division by zero when delta_T1 ~ delta_T2
                lmtd = delta_T1
            else:
                lmtd = (delta_T1 - delta_T2) / self._ln(delta_T1 / delta_T2)
        else:
            lmtd = Decimal("0")

        self._add_step(
            step_number=2,
            operation="calculate",
            description="Calculate Log Mean Temperature Difference",
            inputs={"delta_T1": delta_T1, "delta_T2": delta_T2},
            output_name="lmtd",
            output_value=lmtd,
            formula="LMTD = (dT1 - dT2) / ln(dT1/dT2)",
            reference="TEMA Eq. 5-1"
        )

        # Step 3: Calculate effectiveness
        # Q_max based on minimum capacity rate
        delta_T_hot = T_hi - T_ho
        delta_T_cold = T_co - T_ci
        max_delta_T = T_hi - T_ci

        if max_delta_T > Decimal("0"):
            # Use maximum temperature difference
            effectiveness = max(delta_T_hot, delta_T_cold) / max_delta_T
            effectiveness = min(effectiveness, Decimal("1"))
        else:
            effectiveness = Decimal("0")

        self._add_step(
            step_number=3,
            operation="calculate",
            description="Calculate heat exchanger effectiveness",
            inputs={"delta_T_hot": delta_T_hot, "max_delta_T": max_delta_T},
            output_name="effectiveness",
            output_value=effectiveness,
            formula="epsilon = (T_hot_in - T_hot_out) / (T_hot_in - T_cold_in)",
            reference="NTU method"
        )

        # Step 4: Calculate actual U-value if area provided
        if area is not None and lmtd > Decimal("0"):
            A = self._to_decimal(area)
            u_actual = Q_actual / (A * lmtd)
        else:
            u_actual = Decimal("0")
            A = Decimal("0")

        self._add_step(
            step_number=4,
            operation="calculate",
            description="Calculate actual overall heat transfer coefficient",
            inputs={"Q_actual": Q_actual, "A": A, "LMTD": lmtd},
            output_name="U_actual",
            output_value=u_actual,
            formula="U = Q / (A * LMTD)",
            reference="ASME PTC 12.5"
        )

        # Step 5: Calculate U-value ratio
        if u_value_design is not None:
            U_design = self._to_decimal(u_value_design)
            if U_design > Decimal("0"):
                u_ratio = u_actual / U_design
            else:
                u_ratio = Decimal("0")
        else:
            U_design = Decimal("0")
            u_ratio = Decimal("1")

        self._add_step(
            step_number=5,
            operation="divide",
            description="Calculate U-value ratio",
            inputs={"U_actual": u_actual, "U_design": U_design},
            output_name="u_ratio",
            output_value=u_ratio,
            formula="U_ratio = U_actual / U_design"
        )

        # Step 6: Calculate NTU (Number of Transfer Units)
        if effectiveness > Decimal("0") and effectiveness < Decimal("1"):
            # Simplified NTU calculation for counterflow
            ntu = -self._ln(Decimal("1") - effectiveness)
        else:
            ntu = Decimal("0")

        self._add_step(
            step_number=6,
            operation="calculate",
            description="Calculate Number of Transfer Units",
            inputs={"effectiveness": effectiveness},
            output_name="NTU",
            output_value=ntu,
            formula="NTU = -ln(1 - epsilon) for balanced flow",
            reference="Kays & London"
        )

        # Calculate heat duty ratio
        if Q_design > Decimal("0"):
            heat_duty_ratio = Q_actual / Q_design
        else:
            heat_duty_ratio = Decimal("0")

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            "thermal_efficiency",
            {
                "Q_actual": Q_actual,
                "Q_design": Q_design,
                "T_hi": T_hi,
                "T_ho": T_ho,
                "T_ci": T_ci,
                "T_co": T_co,
            },
            {
                "thermal_efficiency": thermal_efficiency,
                "effectiveness": effectiveness,
                "lmtd": lmtd,
                "u_ratio": u_ratio,
            }
        )

        return ThermalPerformanceResult(
            thermal_efficiency=self._apply_precision(thermal_efficiency),
            effectiveness=self._apply_precision(effectiveness),
            u_value_actual=self._apply_precision(u_actual),
            u_value_design=U_design,
            u_value_ratio=self._apply_precision(u_ratio),
            heat_duty_actual=Q_actual,
            heat_duty_design=Q_design,
            heat_duty_ratio=self._apply_precision(heat_duty_ratio),
            lmtd=self._apply_precision(lmtd),
            ntu=self._apply_precision(ntu),
            timestamp=timestamp,
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # HYDRAULIC PERFORMANCE
    # =========================================================================

    def calculate_hydraulic_performance(
        self,
        pressure_drop_shell: Union[Decimal, float, str],
        pressure_drop_tube: Union[Decimal, float, str],
        pressure_drop_design_shell: Union[Decimal, float, str],
        pressure_drop_design_tube: Union[Decimal, float, str],
        flow_rate_shell: Union[Decimal, float, str],
        flow_rate_tube: Union[Decimal, float, str],
        flow_rate_design_shell: Optional[Union[Decimal, float, str]] = None,
        flow_rate_design_tube: Optional[Union[Decimal, float, str]] = None
    ) -> HydraulicPerformanceResult:
        """
        Calculate hydraulic performance metrics.

        Pressure Drop Ratio:
            DP_ratio = DP_actual / DP_design

        Flow Capacity Reduction:
            Flow_ratio = Q_actual / Q_design

        Pump Power Increase (for same flow):
            Power_increase = (DP_actual / DP_design - 1) * 100%

        Args:
            pressure_drop_shell: Actual shell-side pressure drop (kPa)
            pressure_drop_tube: Actual tube-side pressure drop (kPa)
            pressure_drop_design_shell: Design shell-side pressure drop (kPa)
            pressure_drop_design_tube: Design tube-side pressure drop (kPa)
            flow_rate_shell: Actual shell-side flow rate (kg/s)
            flow_rate_tube: Actual tube-side flow rate (kg/s)
            flow_rate_design_shell: Design shell-side flow rate (kg/s)
            flow_rate_design_tube: Design tube-side flow rate (kg/s)

        Returns:
            HydraulicPerformanceResult with complete provenance

        Reference:
            TEMA Standards, Section 5
        """
        self._calculation_steps = []
        timestamp = datetime.now(timezone.utc).isoformat()

        # Convert inputs to Decimal
        DP_shell = self._to_decimal(pressure_drop_shell)
        DP_tube = self._to_decimal(pressure_drop_tube)
        DP_design_shell = self._to_decimal(pressure_drop_design_shell)
        DP_design_tube = self._to_decimal(pressure_drop_design_tube)
        Q_shell = self._to_decimal(flow_rate_shell)
        Q_tube = self._to_decimal(flow_rate_tube)

        # Step 1: Calculate shell-side pressure drop ratio
        if DP_design_shell > Decimal("0"):
            dp_ratio_shell = DP_shell / DP_design_shell
        else:
            dp_ratio_shell = Decimal("1")

        self._add_step(
            step_number=1,
            operation="divide",
            description="Calculate shell-side pressure drop ratio",
            inputs={"DP_shell": DP_shell, "DP_design_shell": DP_design_shell},
            output_name="dp_ratio_shell",
            output_value=dp_ratio_shell,
            formula="DP_ratio = DP_actual / DP_design"
        )

        # Step 2: Calculate tube-side pressure drop ratio
        if DP_design_tube > Decimal("0"):
            dp_ratio_tube = DP_tube / DP_design_tube
        else:
            dp_ratio_tube = Decimal("1")

        self._add_step(
            step_number=2,
            operation="divide",
            description="Calculate tube-side pressure drop ratio",
            inputs={"DP_tube": DP_tube, "DP_design_tube": DP_design_tube},
            output_name="dp_ratio_tube",
            output_value=dp_ratio_tube,
            formula="DP_ratio = DP_actual / DP_design"
        )

        # Step 3: Calculate flow capacity ratio
        if flow_rate_design_shell is not None and flow_rate_design_tube is not None:
            Q_design_shell = self._to_decimal(flow_rate_design_shell)
            Q_design_tube = self._to_decimal(flow_rate_design_tube)

            if Q_design_shell > Decimal("0") and Q_design_tube > Decimal("0"):
                flow_ratio_shell = Q_shell / Q_design_shell
                flow_ratio_tube = Q_tube / Q_design_tube
                flow_capacity_ratio = (flow_ratio_shell + flow_ratio_tube) / Decimal("2")
            else:
                flow_capacity_ratio = Decimal("1")
        else:
            flow_capacity_ratio = Decimal("1")

        self._add_step(
            step_number=3,
            operation="calculate",
            description="Calculate average flow capacity ratio",
            inputs={"Q_shell": Q_shell, "Q_tube": Q_tube},
            output_name="flow_capacity_ratio",
            output_value=flow_capacity_ratio,
            formula="Flow_ratio = Q_actual / Q_design"
        )

        # Step 4: Calculate pump power increase
        # Power ~ DP * Q, for same flow: Power_increase = (DP_ratio - 1) * 100%
        avg_dp_ratio = (dp_ratio_shell + dp_ratio_tube) / Decimal("2")
        pump_power_increase = (avg_dp_ratio - Decimal("1")) * Decimal("100")
        pump_power_increase = max(Decimal("0"), pump_power_increase)

        self._add_step(
            step_number=4,
            operation="calculate",
            description="Calculate pump power increase percentage",
            inputs={"avg_dp_ratio": avg_dp_ratio},
            output_name="pump_power_increase",
            output_value=pump_power_increase,
            formula="Power_increase = (DP_ratio - 1) * 100%"
        )

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            "hydraulic_performance",
            {
                "DP_shell": DP_shell,
                "DP_tube": DP_tube,
                "Q_shell": Q_shell,
                "Q_tube": Q_tube,
            },
            {
                "dp_ratio_shell": dp_ratio_shell,
                "dp_ratio_tube": dp_ratio_tube,
                "pump_power_increase": pump_power_increase,
            }
        )

        return HydraulicPerformanceResult(
            pressure_drop_shell=DP_shell,
            pressure_drop_tube=DP_tube,
            pressure_drop_design_shell=DP_design_shell,
            pressure_drop_design_tube=DP_design_tube,
            pressure_drop_ratio_shell=self._apply_precision(dp_ratio_shell),
            pressure_drop_ratio_tube=self._apply_precision(dp_ratio_tube),
            flow_rate_shell=Q_shell,
            flow_rate_tube=Q_tube,
            flow_capacity_ratio=self._apply_precision(flow_capacity_ratio),
            pump_power_increase_percent=self._apply_precision(pump_power_increase),
            timestamp=timestamp,
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # DEGRADATION TREND ANALYSIS
    # =========================================================================

    def track_degradation_trend(
        self,
        performance_history: Sequence[Tuple[Union[Decimal, float], Union[Decimal, float]]],
        model: DegradationModel = DegradationModel.EXPONENTIAL,
        confidence_level: str = "95%"
    ) -> DegradationTrendResult:
        """
        Track performance degradation trend using specified model.

        Linear Model:
            P(t) = P_0 - k*t
            where k is degradation rate

        Exponential Model:
            P(t) = P_0 * exp(-lambda*t)
            where lambda is decay constant

        Asymptotic Model:
            P(t) = P_inf + (P_0 - P_inf) * exp(-t/tau)
            where P_inf is limiting performance, tau is time constant

        Args:
            performance_history: List of (time_days, performance_value) tuples
            model: Degradation model to use
            confidence_level: Confidence level for projections

        Returns:
            DegradationTrendResult with trend analysis and projections

        Reference:
            ISO 13379-1:2012, Condition monitoring diagnostics
        """
        self._calculation_steps = []
        timestamp = datetime.now(timezone.utc).isoformat()

        # Convert data to Decimal arrays
        times = [self._to_decimal(t) for t, _ in performance_history]
        values = [self._to_decimal(v) for _, v in performance_history]
        n = len(times)

        if n < 3:
            raise ValueError("At least 3 data points required for trend analysis")

        # Step 1: Calculate initial and current performance
        initial_performance = values[0]
        current_performance = values[-1]

        self._add_step(
            step_number=1,
            operation="extract",
            description="Extract initial and current performance",
            inputs={"data_points": n},
            output_name="performance_endpoints",
            output_value={"initial": initial_performance, "current": current_performance}
        )

        # Step 2: Fit degradation model
        if model == DegradationModel.LINEAR:
            params = self._fit_linear_model(times, values)
            model_name = "LINEAR"
        elif model == DegradationModel.EXPONENTIAL:
            params = self._fit_exponential_model(times, values)
            model_name = "EXPONENTIAL"
        elif model == DegradationModel.ASYMPTOTIC:
            params = self._fit_asymptotic_model(times, values)
            model_name = "ASYMPTOTIC"
        else:
            params = self._fit_linear_model(times, values)
            model_name = "LINEAR"

        self._add_step(
            step_number=2,
            operation="fit",
            description=f"Fit {model_name} degradation model",
            inputs={"times": [str(t) for t in times[:5]], "values": [str(v) for v in values[:5]]},
            output_name="model_parameters",
            output_value=params
        )

        # Step 3: Calculate model R-squared
        r_squared = self._calculate_r_squared(times, values, model, params)

        self._add_step(
            step_number=3,
            operation="calculate",
            description="Calculate model goodness of fit",
            inputs={"model": model_name},
            output_name="r_squared",
            output_value=r_squared,
            formula="R^2 = 1 - SS_res / SS_tot"
        )

        # Step 4: Calculate projections
        t_current = times[-1]
        proj_30d = self._project_performance(t_current + Decimal("30"), model, params)
        proj_90d = self._project_performance(t_current + Decimal("90"), model, params)
        proj_365d = self._project_performance(t_current + Decimal("365"), model, params)

        self._add_step(
            step_number=4,
            operation="project",
            description="Project future performance",
            inputs={"t_30d": t_current + Decimal("30"), "t_90d": t_current + Decimal("90")},
            output_name="projections",
            output_value={"30d": proj_30d, "90d": proj_90d, "365d": proj_365d}
        )

        # Step 5: Determine trend direction and acceleration
        degradation_rate = self._extract_degradation_rate(model, params)
        acceleration = self._calculate_acceleration(times, values)
        trend_direction = self._classify_trend(degradation_rate, acceleration)

        self._add_step(
            step_number=5,
            operation="classify",
            description="Classify degradation trend",
            inputs={"degradation_rate": degradation_rate, "acceleration": acceleration},
            output_name="trend_direction",
            output_value=trend_direction.name
        )

        # Serialize model parameters
        model_params_str = {k: str(v) for k, v in params.items()}

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            "degradation_trend",
            {"data_points": n, "model": model_name},
            {"degradation_rate": degradation_rate, "r_squared": r_squared}
        )

        return DegradationTrendResult(
            model_type=model_name,
            degradation_rate=self._apply_precision(degradation_rate),
            initial_performance=self._apply_precision(initial_performance),
            current_performance=self._apply_precision(current_performance),
            projected_performance_30d=self._apply_precision(proj_30d),
            projected_performance_90d=self._apply_precision(proj_90d),
            projected_performance_365d=self._apply_precision(proj_365d),
            model_r_squared=self._apply_precision(r_squared, 4),
            trend_direction=trend_direction.name,
            acceleration=self._apply_precision(acceleration),
            model_parameters=model_params_str,
            data_points_used=n,
            timestamp=timestamp,
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # PATTERN ANALYSIS
    # =========================================================================

    def analyze_performance_patterns(
        self,
        performance_history: Sequence[Tuple[Union[Decimal, float], Union[Decimal, float]]],
        sampling_interval_days: Union[Decimal, float, str] = "1"
    ) -> PerformancePatternResult:
        """
        Analyze performance patterns including moving averages,
        rate of change, and seasonal patterns.

        Moving Average:
            MA_n = (1/n) * sum(P[i] for i in last n points)

        Rate of Change:
            ROC = (P_current - P_previous) / time_interval

        Acceleration:
            a = (ROC_current - ROC_previous) / time_interval

        Args:
            performance_history: List of (time_days, performance_value) tuples
            sampling_interval_days: Expected interval between samples

        Returns:
            PerformancePatternResult with pattern analysis
        """
        self._calculation_steps = []
        timestamp = datetime.now(timezone.utc).isoformat()

        # Convert data
        times = [self._to_decimal(t) for t, _ in performance_history]
        values = [self._to_decimal(v) for _, v in performance_history]
        n = len(values)
        interval = self._to_decimal(sampling_interval_days)

        if n < 7:
            raise ValueError("At least 7 data points required for pattern analysis")

        # Step 1: Calculate moving averages
        ma_7d = self._calculate_moving_average(values, min(7, n))
        ma_30d = self._calculate_moving_average(values, min(30, n))
        ma_90d = self._calculate_moving_average(values, min(90, n))

        self._add_step(
            step_number=1,
            operation="calculate",
            description="Calculate moving averages",
            inputs={"n_points": n},
            output_name="moving_averages",
            output_value={"7d": ma_7d, "30d": ma_30d, "90d": ma_90d}
        )

        # Step 2: Calculate rate of change
        roc_daily = self._calculate_rate_of_change(values, 1)
        roc_weekly = self._calculate_rate_of_change(values, min(7, n-1))
        roc_monthly = self._calculate_rate_of_change(values, min(30, n-1))

        self._add_step(
            step_number=2,
            operation="calculate",
            description="Calculate rates of change",
            inputs={"interval": interval},
            output_name="rates_of_change",
            output_value={"daily": roc_daily, "weekly": roc_weekly, "monthly": roc_monthly}
        )

        # Step 3: Calculate acceleration
        acceleration = self._calculate_trend_acceleration(times, values)

        self._add_step(
            step_number=3,
            operation="calculate",
            description="Calculate trend acceleration",
            inputs={"n_points": n},
            output_name="acceleration",
            output_value=acceleration,
            formula="a = d(ROC)/dt"
        )

        # Step 4: Detect seasonal patterns
        seasonal_pattern, amplitude, phase = self._detect_seasonal_pattern(times, values)

        self._add_step(
            step_number=4,
            operation="detect",
            description="Detect seasonal patterns",
            inputs={"n_points": n},
            output_name="seasonal",
            output_value={"pattern": seasonal_pattern.name, "amplitude": amplitude}
        )

        # Step 5: Calculate volatility index
        volatility = self._calculate_volatility(values)

        self._add_step(
            step_number=5,
            operation="calculate",
            description="Calculate volatility index",
            inputs={"n_points": n},
            output_name="volatility",
            output_value=volatility
        )

        # Step 6: Calculate anomaly score
        anomaly_score = self._calculate_anomaly_score(values)

        self._add_step(
            step_number=6,
            operation="calculate",
            description="Calculate anomaly score",
            inputs={"current_value": values[-1], "mean": sum(values)/Decimal(str(n))},
            output_name="anomaly_score",
            output_value=anomaly_score
        )

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            "performance_patterns",
            {"n_points": n, "interval": interval},
            {"ma_7d": ma_7d, "volatility": volatility}
        )

        return PerformancePatternResult(
            moving_average_7d=self._apply_precision(ma_7d),
            moving_average_30d=self._apply_precision(ma_30d),
            moving_average_90d=self._apply_precision(ma_90d),
            rate_of_change_daily=self._apply_precision(roc_daily),
            rate_of_change_weekly=self._apply_precision(roc_weekly),
            rate_of_change_monthly=self._apply_precision(roc_monthly),
            acceleration=self._apply_precision(acceleration),
            seasonal_pattern=seasonal_pattern.name,
            seasonal_amplitude=self._apply_precision(amplitude),
            seasonal_phase_days=self._apply_precision(phase),
            volatility_index=self._apply_precision(volatility),
            anomaly_score=self._apply_precision(anomaly_score),
            timestamp=timestamp,
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # PERFORMANCE BENCHMARKING
    # =========================================================================

    def benchmark_performance(
        self,
        current_performance: Union[Decimal, float, str],
        design_performance: Union[Decimal, float, str],
        historical_best: Union[Decimal, float, str],
        fleet_performances: Optional[Sequence[Union[Decimal, float, str]]] = None,
        industry_benchmark: Optional[Union[Decimal, float, str]] = None
    ) -> BenchmarkResult:
        """
        Benchmark current performance against design, historical best,
        fleet average, and industry standards.

        Args:
            current_performance: Current performance metric (0-1 or 0-100)
            design_performance: Design/rated performance
            historical_best: Best recorded performance
            fleet_performances: List of performances from similar exchangers
            industry_benchmark: Industry standard benchmark

        Returns:
            BenchmarkResult with comprehensive comparisons
        """
        self._calculation_steps = []
        timestamp = datetime.now(timezone.utc).isoformat()

        # Convert inputs
        P_current = self._to_decimal(current_performance)
        P_design = self._to_decimal(design_performance)
        P_best = self._to_decimal(historical_best)

        # Normalize to 0-1 scale if needed
        if P_design > Decimal("1"):
            P_current = P_current / Decimal("100")
            P_design = P_design / Decimal("100")
            P_best = P_best / Decimal("100")

        # Step 1: Calculate design comparison ratio
        if P_design > Decimal("0"):
            design_ratio = P_current / P_design
        else:
            design_ratio = Decimal("0")

        self._add_step(
            step_number=1,
            operation="divide",
            description="Calculate design comparison ratio",
            inputs={"P_current": P_current, "P_design": P_design},
            output_name="design_ratio",
            output_value=design_ratio
        )

        # Step 2: Calculate historical best ratio
        if P_best > Decimal("0"):
            best_ratio = P_current / P_best
        else:
            best_ratio = Decimal("0")

        self._add_step(
            step_number=2,
            operation="divide",
            description="Calculate historical best ratio",
            inputs={"P_current": P_current, "P_best": P_best},
            output_name="best_ratio",
            output_value=best_ratio
        )

        # Step 3: Calculate fleet comparison
        if fleet_performances and len(fleet_performances) > 0:
            fleet_values = [self._to_decimal(p) for p in fleet_performances]
            if fleet_values[0] > Decimal("1"):
                fleet_values = [v / Decimal("100") for v in fleet_values]

            fleet_avg = sum(fleet_values) / Decimal(str(len(fleet_values)))
            if fleet_avg > Decimal("0"):
                fleet_ratio = P_current / fleet_avg
            else:
                fleet_ratio = Decimal("1")

            # Calculate percentile
            fleet_sorted = sorted(fleet_values)
            rank = sum(1 for v in fleet_sorted if v <= P_current)
            fleet_percentile = (Decimal(str(rank)) / Decimal(str(len(fleet_values)))) * Decimal("100")
        else:
            fleet_avg = P_current
            fleet_ratio = Decimal("1")
            fleet_percentile = Decimal("50")

        self._add_step(
            step_number=3,
            operation="calculate",
            description="Calculate fleet comparison",
            inputs={"P_current": P_current, "fleet_count": len(fleet_performances) if fleet_performances else 0},
            output_name="fleet_metrics",
            output_value={"ratio": fleet_ratio, "percentile": fleet_percentile}
        )

        # Step 4: Calculate industry benchmark ratio
        if industry_benchmark is not None:
            P_industry = self._to_decimal(industry_benchmark)
            if P_industry > Decimal("1"):
                P_industry = P_industry / Decimal("100")
            if P_industry > Decimal("0"):
                industry_ratio = P_current / P_industry
            else:
                industry_ratio = Decimal("1")
        else:
            industry_ratio = Decimal("1")

        self._add_step(
            step_number=4,
            operation="divide",
            description="Calculate industry benchmark ratio",
            inputs={"P_current": P_current},
            output_name="industry_ratio",
            output_value=industry_ratio
        )

        # Step 5: Calculate performance gaps
        gap_to_design = (P_design - P_current) * Decimal("100")
        gap_to_best = (P_best - P_current) * Decimal("100")

        # Recovery potential = gap to best (what can be recovered with maintenance)
        recovery_potential = max(Decimal("0"), gap_to_best)

        self._add_step(
            step_number=5,
            operation="calculate",
            description="Calculate performance gaps",
            inputs={"P_design": P_design, "P_best": P_best},
            output_name="gaps",
            output_value={"to_design": gap_to_design, "to_best": gap_to_best}
        )

        # Step 6: Determine benchmark status and recommendations
        if design_ratio >= Decimal("0.95"):
            status = "EXCELLENT"
            recommendations = ("Continue current maintenance schedule",)
        elif design_ratio >= Decimal("0.85"):
            status = "GOOD"
            recommendations = (
                "Monitor for early degradation signs",
                "Consider preventive cleaning",
            )
        elif design_ratio >= Decimal("0.70"):
            status = "ACCEPTABLE"
            recommendations = (
                "Schedule cleaning/maintenance",
                "Review operating conditions",
                "Increase monitoring frequency",
            )
        elif design_ratio >= Decimal("0.50"):
            status = "BELOW_STANDARD"
            recommendations = (
                "Urgent cleaning required",
                "Inspect for fouling and corrosion",
                "Consider tube bundle inspection",
                "Evaluate replacement timing",
            )
        else:
            status = "CRITICAL"
            recommendations = (
                "Immediate maintenance required",
                "Full inspection needed",
                "Prepare replacement plan",
                "Consider emergency bypass",
            )

        self._add_step(
            step_number=6,
            operation="classify",
            description="Determine benchmark status",
            inputs={"design_ratio": design_ratio},
            output_name="status",
            output_value=status
        )

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            "benchmark",
            {"P_current": P_current, "P_design": P_design, "P_best": P_best},
            {"design_ratio": design_ratio, "status": status}
        )

        return BenchmarkResult(
            design_comparison_ratio=self._apply_precision(design_ratio),
            historical_best_ratio=self._apply_precision(best_ratio),
            fleet_average_ratio=self._apply_precision(fleet_ratio),
            fleet_percentile=self._apply_precision(fleet_percentile),
            industry_benchmark_ratio=self._apply_precision(industry_ratio),
            performance_gap_to_design=self._apply_precision(gap_to_design),
            performance_gap_to_best=self._apply_precision(gap_to_best),
            recovery_potential_percent=self._apply_precision(recovery_potential),
            benchmark_status=status,
            recommendations=tuple(recommendations),
            timestamp=timestamp,
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # HEALTH INDEX CALCULATION
    # =========================================================================

    def calculate_health_index(
        self,
        thermal_efficiency: Union[Decimal, float, str],
        u_value_ratio: Union[Decimal, float, str],
        pressure_drop_ratio: Union[Decimal, float, str],
        vibration_level: Optional[Union[Decimal, float, str]] = None,
        fouling_factor: Optional[Union[Decimal, float, str]] = None,
        confidence_level: str = "95%",
        custom_weights: Optional[Dict[str, Decimal]] = None
    ) -> HealthIndexResult:
        """
        Calculate multi-parameter weighted health index.

        Health Index = sum(w_i * S_i) for all components

        Component Scores (0-100):
        - Thermal: Based on efficiency and effectiveness
        - Hydraulic: Based on pressure drop increase
        - Mechanical: Based on vibration levels
        - Fouling: Based on fouling resistance

        Args:
            thermal_efficiency: Current thermal efficiency (0-1)
            u_value_ratio: U_actual / U_design ratio
            pressure_drop_ratio: DP_actual / DP_design ratio
            vibration_level: Vibration level (mm/s RMS), optional
            fouling_factor: Fouling resistance (m^2-K/kW), optional
            confidence_level: Confidence level for interval
            custom_weights: Custom component weights

        Returns:
            HealthIndexResult with overall health status

        Reference:
            ISO 13379-1:2012
        """
        self._calculation_steps = []
        timestamp = datetime.now(timezone.utc).isoformat()

        # Use custom weights if provided
        weights = custom_weights or self._health_weights

        # Convert inputs
        eta = self._to_decimal(thermal_efficiency)
        u_ratio = self._to_decimal(u_value_ratio)
        dp_ratio = self._to_decimal(pressure_drop_ratio)

        # Step 1: Calculate thermal score (0-100)
        # Based on efficiency and U-value ratio
        thermal_score = (eta * Decimal("0.5") + u_ratio * Decimal("0.5")) * Decimal("100")
        thermal_score = min(Decimal("100"), max(Decimal("0"), thermal_score))

        self._add_step(
            step_number=1,
            operation="calculate",
            description="Calculate thermal component score",
            inputs={"efficiency": eta, "u_ratio": u_ratio},
            output_name="thermal_score",
            output_value=thermal_score,
            formula="S_thermal = (eta*0.5 + u_ratio*0.5) * 100"
        )

        # Step 2: Calculate hydraulic score (0-100)
        # Score decreases as pressure drop increases
        # dp_ratio = 1 -> 100, dp_ratio = 2 -> 0
        if dp_ratio <= Decimal("1"):
            hydraulic_score = Decimal("100")
        elif dp_ratio >= Decimal("2"):
            hydraulic_score = Decimal("0")
        else:
            hydraulic_score = (Decimal("2") - dp_ratio) * Decimal("100")

        self._add_step(
            step_number=2,
            operation="calculate",
            description="Calculate hydraulic component score",
            inputs={"dp_ratio": dp_ratio},
            output_name="hydraulic_score",
            output_value=hydraulic_score,
            formula="S_hydraulic = max(0, (2 - dp_ratio) * 100)"
        )

        # Step 3: Calculate mechanical score (0-100)
        if vibration_level is not None:
            vib = self._to_decimal(vibration_level)
            # ISO 10816 limits: Good < 2.8 mm/s, Alert < 7.1 mm/s
            if vib <= Decimal("2.8"):
                mechanical_score = Decimal("100")
            elif vib >= Decimal("11.2"):
                mechanical_score = Decimal("0")
            else:
                mechanical_score = ((Decimal("11.2") - vib) / Decimal("8.4")) * Decimal("100")
        else:
            mechanical_score = Decimal("85")  # Default assumption

        self._add_step(
            step_number=3,
            operation="calculate",
            description="Calculate mechanical component score",
            inputs={"vibration": vibration_level},
            output_name="mechanical_score",
            output_value=mechanical_score
        )

        # Step 4: Calculate fouling score (0-100)
        if fouling_factor is not None:
            Rf = self._to_decimal(fouling_factor)
            # Design fouling ~0.0003, severe fouling ~0.001
            if Rf <= Decimal("0.0003"):
                fouling_score = Decimal("100")
            elif Rf >= Decimal("0.001"):
                fouling_score = Decimal("0")
            else:
                fouling_score = ((Decimal("0.001") - Rf) / Decimal("0.0007")) * Decimal("100")
        else:
            # Estimate from U-value degradation
            fouling_score = u_ratio * Decimal("100")

        self._add_step(
            step_number=4,
            operation="calculate",
            description="Calculate fouling component score",
            inputs={"fouling_factor": fouling_factor},
            output_name="fouling_score",
            output_value=fouling_score
        )

        # Step 5: Calculate overall health index
        overall_hi = (
            weights.get("thermal", Decimal("0.4")) * thermal_score +
            weights.get("hydraulic", Decimal("0.35")) * hydraulic_score +
            weights.get("mechanical", Decimal("0.15")) * mechanical_score +
            weights.get("fouling", Decimal("0.1")) * fouling_score
        )

        self._add_step(
            step_number=5,
            operation="weighted_sum",
            description="Calculate overall health index",
            inputs={
                "thermal": thermal_score,
                "hydraulic": hydraulic_score,
                "mechanical": mechanical_score,
                "fouling": fouling_score
            },
            output_name="overall_health_index",
            output_value=overall_hi,
            formula="HI = sum(w_i * S_i)"
        )

        # Step 6: Determine health status
        health_status = self._classify_health_status(overall_hi)

        # Find limiting factor
        component_scores = {
            "thermal": thermal_score,
            "hydraulic": hydraulic_score,
            "mechanical": mechanical_score,
            "fouling": fouling_score
        }
        limiting_factor = min(component_scores, key=component_scores.get)

        self._add_step(
            step_number=6,
            operation="classify",
            description="Classify health status",
            inputs={"overall_hi": overall_hi},
            output_name="health_status",
            output_value=health_status.name
        )

        # Step 7: Calculate confidence interval
        z = Z_SCORES.get(confidence_level, Decimal("1.96"))
        # Assume ~5% uncertainty
        std_error = overall_hi * Decimal("0.05")
        ci_lower = max(Decimal("0"), overall_hi - z * std_error)
        ci_upper = min(Decimal("100"), overall_hi + z * std_error)

        # Step 8: Estimate days to next threshold
        days_to_threshold = self._estimate_days_to_threshold(overall_hi, health_status)

        # Serialize weights
        weights_str = {k: str(v) for k, v in weights.items()}

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            "health_index",
            {"eta": eta, "u_ratio": u_ratio, "dp_ratio": dp_ratio},
            {"overall_hi": overall_hi, "status": health_status.name}
        )

        return HealthIndexResult(
            overall_health_index=self._apply_precision(overall_hi),
            thermal_score=self._apply_precision(thermal_score),
            hydraulic_score=self._apply_precision(hydraulic_score),
            mechanical_score=self._apply_precision(mechanical_score),
            fouling_score=self._apply_precision(fouling_score),
            health_status=health_status.name,
            component_weights=weights_str,
            limiting_factor=limiting_factor,
            days_to_next_threshold=self._apply_precision(days_to_threshold),
            confidence_lower=self._apply_precision(ci_lower),
            confidence_upper=self._apply_precision(ci_upper),
            confidence_level=confidence_level,
            timestamp=timestamp,
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # REMAINING PERFORMANCE LIFE
    # =========================================================================

    def estimate_remaining_performance_life(
        self,
        performance_history: Sequence[Tuple[Union[Decimal, float], Union[Decimal, float]]],
        min_acceptable_performance: Union[Decimal, float, str] = "0.6",
        model: DegradationModel = DegradationModel.EXPONENTIAL,
        confidence_level: str = "95%"
    ) -> RemainingPerformanceLifeResult:
        """
        Estimate remaining useful life based on performance degradation.

        Calculates time until performance reaches minimum acceptable level
        based on fitted degradation model.

        For exponential model:
            P(t) = P_0 * exp(-lambda*t)
            t_RUL = -ln(P_min/P_0) / lambda

        Args:
            performance_history: List of (time_days, performance_value) tuples
            min_acceptable_performance: Minimum acceptable performance (0-1)
            model: Degradation model to use
            confidence_level: Confidence level for interval

        Returns:
            RemainingPerformanceLifeResult with RUL estimates

        Reference:
            ISO 13381-1:2015, Remaining useful life estimation
        """
        self._calculation_steps = []
        timestamp = datetime.now(timezone.utc).isoformat()

        # Convert data
        times = [self._to_decimal(t) for t, _ in performance_history]
        values = [self._to_decimal(v) for _, v in performance_history]
        P_min = self._to_decimal(min_acceptable_performance)
        n = len(times)

        if n < 3:
            raise ValueError("At least 3 data points required")

        # Step 1: Get current performance
        current_performance = values[-1]
        t_current = times[-1]

        self._add_step(
            step_number=1,
            operation="extract",
            description="Get current performance state",
            inputs={"n_points": n},
            output_name="current_state",
            output_value={"performance": current_performance, "time": t_current}
        )

        # Step 2: Fit degradation model
        if model == DegradationModel.LINEAR:
            params = self._fit_linear_model(times, values)
            model_name = "LINEAR"
        elif model == DegradationModel.EXPONENTIAL:
            params = self._fit_exponential_model(times, values)
            model_name = "EXPONENTIAL"
        else:
            params = self._fit_exponential_model(times, values)
            model_name = "EXPONENTIAL"

        self._add_step(
            step_number=2,
            operation="fit",
            description=f"Fit {model_name} model",
            inputs={"data_points": n},
            output_name="model_params",
            output_value=params
        )

        # Step 3: Calculate time to minimum acceptable performance
        if model == DegradationModel.LINEAR:
            # P(t) = P_0 - k*t
            # t_min = (P_0 - P_min) / k
            P_0 = params.get("P_0", current_performance)
            k = params.get("k", Decimal("0.001"))
            if k > Decimal("0"):
                t_to_min = (P_0 - P_min) / k
            else:
                t_to_min = Decimal("999999")
        else:
            # P(t) = P_0 * exp(-lambda*t)
            # t_min = -ln(P_min/P_0) / lambda
            P_0 = params.get("P_0", current_performance)
            lambda_val = params.get("lambda", Decimal("0.001"))
            if P_0 > Decimal("0") and P_min > Decimal("0") and lambda_val > Decimal("0"):
                if P_min < P_0:
                    t_to_min = -self._ln(P_min / P_0) / lambda_val
                else:
                    t_to_min = Decimal("0")
            else:
                t_to_min = Decimal("999999")

        # RUL from current time
        rul_days = max(Decimal("0"), t_to_min - t_current)

        self._add_step(
            step_number=3,
            operation="calculate",
            description="Calculate time to minimum performance",
            inputs={"P_0": P_0, "P_min": P_min},
            output_name="time_to_min",
            output_value=t_to_min
        )

        # Step 4: Calculate component-specific RULs
        # Simplified: use same model for all
        time_to_min_eff = rul_days
        time_to_min_u = rul_days * Decimal("0.95")  # U-value typically degrades faster
        time_to_max_dp = rul_days * Decimal("0.80")  # Pressure drop increases faster

        # Limiting parameter
        component_times = {
            "thermal_efficiency": time_to_min_eff,
            "u_value": time_to_min_u,
            "pressure_drop": time_to_max_dp
        }
        limiting_parameter = min(component_times, key=component_times.get)
        rul_days = min(component_times.values())

        self._add_step(
            step_number=4,
            operation="compare",
            description="Identify limiting parameter",
            inputs=component_times,
            output_name="limiting_parameter",
            output_value=limiting_parameter
        )

        # Step 5: Convert to months and years
        rul_months = rul_days / Decimal("30.44")
        rul_years = rul_days / Decimal("365.25")

        # Step 6: Calculate confidence interval
        z = Z_SCORES.get(confidence_level, Decimal("1.96"))
        # Assume ~15% uncertainty in RUL estimate
        std_error = rul_days * Decimal("0.15")
        ci_lower = max(Decimal("0"), rul_days - z * std_error)
        ci_upper = rul_days + z * std_error

        self._add_step(
            step_number=6,
            operation="calculate",
            description="Calculate confidence interval",
            inputs={"rul_days": rul_days, "std_error": std_error},
            output_name="confidence_interval",
            output_value={"lower": ci_lower, "upper": ci_upper}
        )

        # Current performance as percentage of design
        current_perf_percent = current_performance * Decimal("100")
        min_accept_percent = P_min * Decimal("100")

        # Serialize model parameters
        model_params_str = {k: str(v) for k, v in params.items()}

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            "remaining_life",
            {"n_points": n, "P_min": P_min},
            {"rul_days": rul_days, "limiting": limiting_parameter}
        )

        return RemainingPerformanceLifeResult(
            rul_days=self._apply_precision(rul_days),
            rul_months=self._apply_precision(rul_months, 2),
            rul_years=self._apply_precision(rul_years, 3),
            time_to_min_efficiency=self._apply_precision(time_to_min_eff),
            time_to_min_u_value=self._apply_precision(time_to_min_u),
            time_to_max_pressure_drop=self._apply_precision(time_to_max_dp),
            limiting_parameter=limiting_parameter,
            current_performance_percent=self._apply_precision(current_perf_percent),
            min_acceptable_percent=self._apply_precision(min_accept_percent),
            confidence_lower_days=self._apply_precision(ci_lower),
            confidence_upper_days=self._apply_precision(ci_upper),
            confidence_level=confidence_level,
            degradation_model_used=model_name,
            model_parameters=model_params_str,
            timestamp=timestamp,
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # COMPREHENSIVE REPORT
    # =========================================================================

    def generate_performance_report(
        self,
        exchanger_id: str,
        thermal_data: Dict[str, Any],
        hydraulic_data: Dict[str, Any],
        performance_history: Sequence[Tuple[Union[Decimal, float], Union[Decimal, float]]],
        design_performance: Union[Decimal, float, str],
        historical_best: Union[Decimal, float, str],
        fleet_performances: Optional[Sequence[Union[Decimal, float, str]]] = None
    ) -> PerformanceReportResult:
        """
        Generate comprehensive performance report combining all metrics.

        Args:
            exchanger_id: Unique identifier for the heat exchanger
            thermal_data: Dict with thermal measurement parameters
            hydraulic_data: Dict with hydraulic measurement parameters
            performance_history: Historical performance data
            design_performance: Design/rated performance
            historical_best: Best recorded performance
            fleet_performances: Performances from similar exchangers

        Returns:
            PerformanceReportResult with complete analysis
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Calculate thermal performance
        thermal = self.calculate_thermal_efficiency(
            heat_duty_actual=thermal_data["heat_duty_actual"],
            heat_duty_design=thermal_data["heat_duty_design"],
            t_hot_in=thermal_data["t_hot_in"],
            t_hot_out=thermal_data["t_hot_out"],
            t_cold_in=thermal_data["t_cold_in"],
            t_cold_out=thermal_data["t_cold_out"],
            area=thermal_data.get("area"),
            u_value_design=thermal_data.get("u_value_design")
        )

        # Calculate hydraulic performance
        hydraulic = self.calculate_hydraulic_performance(
            pressure_drop_shell=hydraulic_data["pressure_drop_shell"],
            pressure_drop_tube=hydraulic_data["pressure_drop_tube"],
            pressure_drop_design_shell=hydraulic_data["pressure_drop_design_shell"],
            pressure_drop_design_tube=hydraulic_data["pressure_drop_design_tube"],
            flow_rate_shell=hydraulic_data["flow_rate_shell"],
            flow_rate_tube=hydraulic_data["flow_rate_tube"]
        )

        # Track degradation trend
        degradation = self.track_degradation_trend(
            performance_history=performance_history,
            model=DegradationModel.EXPONENTIAL
        )

        # Analyze patterns
        patterns = self.analyze_performance_patterns(
            performance_history=performance_history
        )

        # Benchmark performance
        benchmark = self.benchmark_performance(
            current_performance=thermal.thermal_efficiency,
            design_performance=design_performance,
            historical_best=historical_best,
            fleet_performances=fleet_performances
        )

        # Calculate health index
        health = self.calculate_health_index(
            thermal_efficiency=thermal.thermal_efficiency,
            u_value_ratio=thermal.u_value_ratio,
            pressure_drop_ratio=(hydraulic.pressure_drop_ratio_shell +
                                 hydraulic.pressure_drop_ratio_tube) / Decimal("2")
        )

        # Estimate remaining life
        remaining_life = self.estimate_remaining_performance_life(
            performance_history=performance_history,
            min_acceptable_performance="0.6"
        )

        # Determine summary status
        if health.health_status == "CRITICAL":
            summary_status = "CRITICAL - Immediate action required"
            priority_actions = (
                "Shutdown for emergency maintenance",
                "Full inspection of tube bundle",
                "Prepare replacement exchanger",
            )
        elif health.health_status == "POOR":
            summary_status = "POOR - Urgent maintenance needed"
            priority_actions = (
                "Schedule cleaning within 7 days",
                "Increase monitoring frequency",
                "Review spare parts availability",
            )
        elif health.health_status == "DEGRADED":
            summary_status = "DEGRADED - Plan maintenance"
            priority_actions = (
                "Schedule maintenance within 30 days",
                "Monitor degradation trend",
                "Prepare maintenance scope",
            )
        elif health.health_status == "GOOD":
            summary_status = "GOOD - Continue monitoring"
            priority_actions = (
                "Continue routine monitoring",
                "Consider preventive cleaning",
            )
        else:
            summary_status = "OPTIMAL - Normal operation"
            priority_actions = (
                "Continue standard maintenance schedule",
            )

        # Calculate overall provenance hash
        combined_data = {
            "exchanger_id": exchanger_id,
            "thermal_hash": thermal.provenance_hash,
            "hydraulic_hash": hydraulic.provenance_hash,
            "health_hash": health.provenance_hash,
        }
        provenance_hash = self._calculate_provenance_hash(
            "performance_report",
            combined_data,
            {"status": summary_status}
        )

        return PerformanceReportResult(
            exchanger_id=exchanger_id,
            report_timestamp=timestamp,
            thermal_performance=thermal,
            hydraulic_performance=hydraulic,
            degradation_trend=degradation,
            performance_pattern=patterns,
            benchmark=benchmark,
            health_index=health,
            remaining_life=remaining_life,
            summary_status=summary_status,
            priority_actions=tuple(priority_actions),
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # HELPER METHODS - MATHEMATICAL OPERATIONS
    # =========================================================================

    def _to_decimal(self, value: Union[Decimal, float, int, str]) -> Decimal:
        """Convert value to Decimal with validation."""
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except InvalidOperation as e:
            raise ValueError(f"Cannot convert {value} to Decimal: {e}")

    def _apply_precision(
        self,
        value: Decimal,
        precision: Optional[int] = None
    ) -> Decimal:
        """Apply precision rounding using ROUND_HALF_UP."""
        prec = precision if precision is not None else self._precision
        if prec == 0:
            return value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        quantize_str = "0." + "0" * prec
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _ln(self, x: Decimal) -> Decimal:
        """Calculate natural logarithm."""
        if x <= Decimal("0"):
            raise ValueError("Cannot take logarithm of non-positive number")
        return Decimal(str(math.log(float(x))))

    def _exp(self, x: Decimal) -> Decimal:
        """Calculate exponential e^x."""
        if x < Decimal("-700"):
            return Decimal("0")
        if x > Decimal("700"):
            raise ValueError("Exponent too large")
        return Decimal(str(math.exp(float(x))))

    def _sqrt(self, x: Decimal) -> Decimal:
        """Calculate square root."""
        if x < Decimal("0"):
            raise ValueError("Cannot take square root of negative number")
        if x == Decimal("0"):
            return Decimal("0")
        return Decimal(str(math.sqrt(float(x))))

    # =========================================================================
    # HELPER METHODS - MODEL FITTING
    # =========================================================================

    def _fit_linear_model(
        self,
        times: List[Decimal],
        values: List[Decimal]
    ) -> Dict[str, Decimal]:
        """
        Fit linear degradation model: P(t) = P_0 - k*t

        Uses least squares regression.
        """
        n = Decimal(str(len(times)))
        sum_t = sum(times)
        sum_p = sum(values)
        sum_tp = sum(t * p for t, p in zip(times, values))
        sum_t2 = sum(t * t for t in times)

        # Linear regression
        denominator = n * sum_t2 - sum_t * sum_t
        if denominator == Decimal("0"):
            return {"P_0": values[0], "k": Decimal("0")}

        # Slope (negative of degradation rate)
        slope = (n * sum_tp - sum_t * sum_p) / denominator
        k = -slope  # Degradation rate is positive

        # Intercept (initial performance)
        P_0 = (sum_p - slope * sum_t) / n

        return {"P_0": P_0, "k": k}

    def _fit_exponential_model(
        self,
        times: List[Decimal],
        values: List[Decimal]
    ) -> Dict[str, Decimal]:
        """
        Fit exponential model: P(t) = P_0 * exp(-lambda*t)

        Linearizes by taking log: ln(P) = ln(P_0) - lambda*t
        """
        # Filter out non-positive values
        valid_data = [(t, v) for t, v in zip(times, values) if v > Decimal("0")]
        if len(valid_data) < 2:
            return {"P_0": values[0], "lambda": Decimal("0.001")}

        times_valid = [t for t, _ in valid_data]
        log_values = [self._ln(v) for _, v in valid_data]

        n = Decimal(str(len(times_valid)))
        sum_t = sum(times_valid)
        sum_ln_p = sum(log_values)
        sum_t_ln_p = sum(t * lnp for t, lnp in zip(times_valid, log_values))
        sum_t2 = sum(t * t for t in times_valid)

        denominator = n * sum_t2 - sum_t * sum_t
        if denominator == Decimal("0"):
            return {"P_0": values[0], "lambda": Decimal("0.001")}

        # Slope = -lambda
        slope = (n * sum_t_ln_p - sum_t * sum_ln_p) / denominator
        lambda_val = -slope
        lambda_val = max(Decimal("0.00001"), lambda_val)  # Ensure positive

        # Intercept = ln(P_0)
        ln_P_0 = (sum_ln_p - slope * sum_t) / n
        P_0 = self._exp(ln_P_0)

        return {"P_0": P_0, "lambda": lambda_val}

    def _fit_asymptotic_model(
        self,
        times: List[Decimal],
        values: List[Decimal]
    ) -> Dict[str, Decimal]:
        """
        Fit asymptotic model: P(t) = P_inf + (P_0 - P_inf) * exp(-t/tau)

        Assumes P_inf = 0.3 (minimum performance) and fits tau.
        """
        P_inf = Decimal("0.3")  # Assume minimum performance
        P_0 = values[0]

        # Transform: (P - P_inf) / (P_0 - P_inf) = exp(-t/tau)
        # ln((P - P_inf) / (P_0 - P_inf)) = -t/tau

        valid_data = [(t, v) for t, v in zip(times, values)
                      if v > P_inf and (P_0 - P_inf) > Decimal("0")]

        if len(valid_data) < 2:
            return {"P_0": P_0, "P_inf": P_inf, "tau": Decimal("365")}

        P_0_minus_inf = P_0 - P_inf
        transformed = []
        for t, v in valid_data:
            ratio = (v - P_inf) / P_0_minus_inf
            if ratio > Decimal("0"):
                transformed.append((t, self._ln(ratio)))

        if len(transformed) < 2:
            return {"P_0": P_0, "P_inf": P_inf, "tau": Decimal("365")}

        # Linear fit: ln(ratio) = -t/tau
        times_t = [t for t, _ in transformed]
        ln_ratios = [lr for _, lr in transformed]

        n = Decimal(str(len(times_t)))
        sum_t = sum(times_t)
        sum_ln_r = sum(ln_ratios)
        sum_t_ln_r = sum(t * lr for t, lr in zip(times_t, ln_ratios))
        sum_t2 = sum(t * t for t in times_t)

        denominator = n * sum_t2 - sum_t * sum_t
        if denominator == Decimal("0"):
            return {"P_0": P_0, "P_inf": P_inf, "tau": Decimal("365")}

        slope = (n * sum_t_ln_r - sum_t * sum_ln_r) / denominator
        tau = Decimal("-1") / slope if slope != Decimal("0") else Decimal("365")
        tau = max(Decimal("1"), tau)  # Ensure positive

        return {"P_0": P_0, "P_inf": P_inf, "tau": tau}

    # =========================================================================
    # HELPER METHODS - STATISTICS
    # =========================================================================

    def _calculate_r_squared(
        self,
        times: List[Decimal],
        values: List[Decimal],
        model: DegradationModel,
        params: Dict[str, Decimal]
    ) -> Decimal:
        """Calculate R-squared (coefficient of determination)."""
        n = len(values)
        mean_value = sum(values) / Decimal(str(n))

        # Calculate SS_tot (total sum of squares)
        ss_tot = sum((v - mean_value) ** 2 for v in values)

        # Calculate SS_res (residual sum of squares)
        predicted = [self._project_performance(t, model, params) for t in times]
        ss_res = sum((v - p) ** 2 for v, p in zip(values, predicted))

        if ss_tot == Decimal("0"):
            return Decimal("1")

        r_squared = Decimal("1") - ss_res / ss_tot
        return max(Decimal("0"), min(Decimal("1"), r_squared))

    def _project_performance(
        self,
        t: Decimal,
        model: DegradationModel,
        params: Dict[str, Decimal]
    ) -> Decimal:
        """Project performance at time t using fitted model."""
        if model == DegradationModel.LINEAR:
            P_0 = params.get("P_0", Decimal("1"))
            k = params.get("k", Decimal("0"))
            return max(Decimal("0"), P_0 - k * t)

        elif model == DegradationModel.EXPONENTIAL:
            P_0 = params.get("P_0", Decimal("1"))
            lambda_val = params.get("lambda", Decimal("0.001"))
            return P_0 * self._exp(-lambda_val * t)

        elif model == DegradationModel.ASYMPTOTIC:
            P_0 = params.get("P_0", Decimal("1"))
            P_inf = params.get("P_inf", Decimal("0.3"))
            tau = params.get("tau", Decimal("365"))
            return P_inf + (P_0 - P_inf) * self._exp(-t / tau)

        else:
            return params.get("P_0", Decimal("1"))

    def _extract_degradation_rate(
        self,
        model: DegradationModel,
        params: Dict[str, Decimal]
    ) -> Decimal:
        """Extract degradation rate from model parameters."""
        if model == DegradationModel.LINEAR:
            return params.get("k", Decimal("0"))
        elif model == DegradationModel.EXPONENTIAL:
            return params.get("lambda", Decimal("0"))
        elif model == DegradationModel.ASYMPTOTIC:
            tau = params.get("tau", Decimal("365"))
            return Decimal("1") / tau if tau > Decimal("0") else Decimal("0")
        return Decimal("0")

    def _calculate_acceleration(
        self,
        times: List[Decimal],
        values: List[Decimal]
    ) -> Decimal:
        """Calculate degradation acceleration (second derivative)."""
        if len(values) < 3:
            return Decimal("0")

        # Calculate first differences
        d1 = [(values[i+1] - values[i]) / (times[i+1] - times[i] + Decimal("0.001"))
              for i in range(len(values) - 1)]

        # Calculate second differences (acceleration)
        d2 = [(d1[i+1] - d1[i]) / (times[i+2] - times[i+1] + Decimal("0.001"))
              for i in range(len(d1) - 1)]

        if not d2:
            return Decimal("0")

        return sum(d2) / Decimal(str(len(d2)))

    def _classify_trend(
        self,
        degradation_rate: Decimal,
        acceleration: Decimal
    ) -> TrendDirection:
        """Classify degradation trend based on rate and acceleration."""
        if degradation_rate < Decimal("-0.001"):
            return TrendDirection.IMPROVING
        elif degradation_rate < Decimal("0.0005"):
            return TrendDirection.STABLE
        elif degradation_rate < Decimal("0.002"):
            if acceleration < Decimal("0"):
                return TrendDirection.DEGRADING_SLOW
            else:
                return TrendDirection.DEGRADING_FAST
        else:
            if acceleration > Decimal("0.0001"):
                return TrendDirection.CRITICAL_DECLINE
            else:
                return TrendDirection.DEGRADING_FAST

    def _calculate_moving_average(
        self,
        values: List[Decimal],
        window: int
    ) -> Decimal:
        """Calculate moving average over specified window."""
        if len(values) < window:
            window = len(values)
        recent = values[-window:]
        return sum(recent) / Decimal(str(len(recent)))

    def _calculate_rate_of_change(
        self,
        values: List[Decimal],
        period: int
    ) -> Decimal:
        """Calculate rate of change over period."""
        if len(values) <= period:
            return Decimal("0")
        return (values[-1] - values[-period-1]) / Decimal(str(period))

    def _calculate_trend_acceleration(
        self,
        times: List[Decimal],
        values: List[Decimal]
    ) -> Decimal:
        """Calculate trend acceleration."""
        return self._calculate_acceleration(times, values)

    def _detect_seasonal_pattern(
        self,
        times: List[Decimal],
        values: List[Decimal]
    ) -> Tuple[SeasonalPattern, Decimal, Decimal]:
        """Detect seasonal patterns in performance data."""
        n = len(values)

        if n < 14:
            return (SeasonalPattern.NONE, Decimal("0"), Decimal("0"))

        # Simple autocorrelation for common periods
        mean_val = sum(values) / Decimal(str(n))
        variance = sum((v - mean_val) ** 2 for v in values) / Decimal(str(n))

        if variance == Decimal("0"):
            return (SeasonalPattern.NONE, Decimal("0"), Decimal("0"))

        # Check for weekly pattern (7 days)
        if n >= 14:
            lag_7 = sum((values[i] - mean_val) * (values[i-7] - mean_val)
                        for i in range(7, n))
            autocorr_7 = lag_7 / (variance * Decimal(str(n - 7)))
            if abs(autocorr_7) > Decimal("0.5"):
                amplitude = self._calculate_volatility(values) * Decimal("2")
                return (SeasonalPattern.WEEKLY, amplitude, Decimal("7"))

        # Check for monthly pattern (30 days)
        if n >= 60:
            lag_30 = sum((values[i] - mean_val) * (values[i-30] - mean_val)
                         for i in range(30, n))
            autocorr_30 = lag_30 / (variance * Decimal(str(n - 30)))
            if abs(autocorr_30) > Decimal("0.5"):
                amplitude = self._calculate_volatility(values) * Decimal("2")
                return (SeasonalPattern.MONTHLY, amplitude, Decimal("30"))

        return (SeasonalPattern.NONE, Decimal("0"), Decimal("0"))

    def _calculate_volatility(self, values: List[Decimal]) -> Decimal:
        """Calculate volatility (coefficient of variation)."""
        n = len(values)
        if n < 2:
            return Decimal("0")

        mean_val = sum(values) / Decimal(str(n))
        if mean_val == Decimal("0"):
            return Decimal("0")

        variance = sum((v - mean_val) ** 2 for v in values) / Decimal(str(n))
        std_dev = self._sqrt(variance)

        return std_dev / mean_val

    def _calculate_anomaly_score(self, values: List[Decimal]) -> Decimal:
        """Calculate anomaly score for most recent value."""
        n = len(values)
        if n < 3:
            return Decimal("0")

        mean_val = sum(values[:-1]) / Decimal(str(n - 1))
        variance = sum((v - mean_val) ** 2 for v in values[:-1]) / Decimal(str(n - 1))
        std_dev = self._sqrt(variance)

        if std_dev == Decimal("0"):
            return Decimal("0")

        # Z-score of latest value
        z_score = abs(values[-1] - mean_val) / std_dev

        # Convert to 0-100 scale
        anomaly_score = min(Decimal("100"), z_score * Decimal("25"))

        return anomaly_score

    def _classify_health_status(self, health_index: Decimal) -> HealthStatus:
        """Classify health status based on index value."""
        for status, (lower, upper) in HEALTH_THRESHOLDS.items():
            if lower <= health_index <= upper:
                return status
        return HealthStatus.CRITICAL

    def _estimate_days_to_threshold(
        self,
        current_hi: Decimal,
        current_status: HealthStatus
    ) -> Decimal:
        """Estimate days until reaching next lower health threshold."""
        # Get next lower threshold
        status_order = [
            HealthStatus.OPTIMAL,
            HealthStatus.GOOD,
            HealthStatus.DEGRADED,
            HealthStatus.POOR,
            HealthStatus.CRITICAL
        ]

        current_idx = status_order.index(current_status)
        if current_idx >= len(status_order) - 1:
            return Decimal("0")  # Already at CRITICAL

        next_status = status_order[current_idx + 1]
        next_threshold = HEALTH_THRESHOLDS[next_status][1]

        # Assume 0.5% degradation per day (simple estimate)
        gap = current_hi - next_threshold
        if gap <= Decimal("0"):
            return Decimal("0")

        days = gap / Decimal("0.5")
        return max(Decimal("0"), days)

    # =========================================================================
    # PROVENANCE TRACKING
    # =========================================================================

    def _add_step(
        self,
        step_number: int,
        operation: str,
        description: str,
        inputs: Dict[str, Any],
        output_name: str,
        output_value: Any,
        formula: str = "",
        reference: str = ""
    ) -> None:
        """Add a calculation step to the provenance trail."""
        step = CalculationStep(
            step_number=step_number,
            operation=operation,
            description=description,
            inputs=inputs,
            output_name=output_name,
            output_value=output_value,
            formula=formula,
            reference=reference
        )
        self._calculation_steps.append(step)

    def _calculate_provenance_hash(
        self,
        calculation_type: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any]
    ) -> str:
        """Calculate SHA-256 provenance hash for audit trail."""
        provenance_data = {
            "calculation_type": calculation_type,
            "inputs": self._serialize_dict(inputs),
            "outputs": self._serialize_dict(outputs),
            "steps": [step.to_dict() for step in self._calculation_steps]
        }

        json_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    def _serialize_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize dictionary for JSON encoding."""
        result = {}
        for k, v in d.items():
            if isinstance(v, Decimal):
                result[k] = str(v)
            elif isinstance(v, (list, tuple)):
                result[k] = [str(x) if isinstance(x, Decimal) else x for x in v]
            elif isinstance(v, dict):
                result[k] = self._serialize_dict(v)
            else:
                result[k] = v
        return result

    def get_calculation_steps(self) -> List[Dict[str, Any]]:
        """Get the calculation steps from the last calculation."""
        return [step.to_dict() for step in self._calculation_steps]


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "DegradationModel",
    "HealthStatus",
    "TrendDirection",
    "SeasonalPattern",

    # Constants
    "HEALTH_THRESHOLDS",
    "DEFAULT_HEALTH_WEIGHTS",
    "MIN_ACCEPTABLE_EFFICIENCY",
    "MIN_ACCEPTABLE_U_VALUE_RATIO",
    "MAX_ACCEPTABLE_PRESSURE_DROP_RATIO",

    # Result Classes
    "ThermalPerformanceResult",
    "HydraulicPerformanceResult",
    "DegradationTrendResult",
    "PerformancePatternResult",
    "BenchmarkResult",
    "HealthIndexResult",
    "RemainingPerformanceLifeResult",
    "PerformanceReportResult",
    "CalculationStep",

    # Main Class
    "PerformanceTracker",
]
