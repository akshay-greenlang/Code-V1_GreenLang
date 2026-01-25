"""
GL-007 FurnacePulse - Draft Analyzer

Deterministic calculator for furnace draft pressure stability analysis,
variance calculations, control effort metrics, and damper/fan performance
indicators.

Key Calculations:
    - Draft pressure stability metrics (variance, std dev)
    - Control effort calculations (damper movements, PID output)
    - Damper position tracking and performance
    - Fan performance indicators (speed, current, efficiency)
    - Stability scoring and alert generation

Example:
    >>> analyzer = DraftAnalyzer(agent_id="GL-007")
    >>> inputs = DraftInputs(
    ...     timestamp=datetime.now(timezone.utc),
    ...     draft_readings_pa=[-25, -24, -26, -25, -23, -27],
    ...     setpoint_pa=-25.0,
    ...     damper_position_pct=45.0
    ... )
    >>> result = analyzer.calculate(inputs)
    >>> print(f"Stability score: {result.result.stability_metrics.stability_score:.1f}")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import math
import sys
import os

# Add framework path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Framework_GreenLang', 'shared'))

from calculator_base import DeterministicCalculator, CalculationResult


class DraftType(str, Enum):
    """Types of furnace draft configurations."""
    NATURAL = "natural"          # Stack-driven draft
    FORCED = "forced"            # Fan pushing air in
    INDUCED = "induced"          # Fan pulling flue gas out
    BALANCED = "balanced"        # Both forced and induced


class ControlMode(str, Enum):
    """Draft control modes."""
    MANUAL = "manual"
    AUTO_PID = "auto_pid"
    CASCADE = "cascade"
    RATIO = "ratio"


@dataclass
class DraftThresholds:
    """
    Thresholds for draft stability monitoring.

    Attributes:
        max_variance_pa2: Maximum acceptable variance
        max_deviation_pa: Maximum deviation from setpoint
        min_draft_pa: Minimum safe draft (most negative)
        max_draft_pa: Maximum safe draft (least negative)
        oscillation_threshold: Threshold for oscillation detection
    """
    max_variance_pa2: float = 4.0       # 4 Pa^2 max variance
    max_deviation_pa: float = 3.0       # +/- 3 Pa from setpoint
    min_draft_pa: float = -50.0         # Strong negative draft limit
    max_draft_pa: float = 0.0           # Positive pressure limit
    oscillation_threshold: float = 2.0  # Pa oscillation amplitude


@dataclass
class DraftInputs:
    """
    Input data for draft analysis.

    Attributes:
        timestamp: Reading timestamp
        draft_readings_pa: Time series of draft pressure readings (Pa)
        setpoint_pa: Draft setpoint (Pa, typically negative)
        damper_position_pct: Current damper position (0-100%)
        sample_interval_s: Time between samples (seconds)
        draft_type: Type of draft configuration
        control_mode: Current control mode
        fan_speed_rpm: Optional fan speed (RPM)
        fan_current_a: Optional fan motor current (Amps)
        fan_rated_power_kw: Optional fan rated power (kW)
        previous_damper_positions: Historical damper positions
        pid_output_pct: Optional PID controller output (%)
    """
    timestamp: datetime
    draft_readings_pa: List[float]
    setpoint_pa: float
    damper_position_pct: float
    sample_interval_s: float = 1.0
    draft_type: DraftType = DraftType.INDUCED
    control_mode: ControlMode = ControlMode.AUTO_PID
    fan_speed_rpm: Optional[float] = None
    fan_current_a: Optional[float] = None
    fan_rated_power_kw: Optional[float] = None
    fan_rated_speed_rpm: Optional[float] = None
    previous_damper_positions: List[float] = field(default_factory=list)
    pid_output_pct: Optional[float] = None


@dataclass
class DraftStabilityMetrics:
    """
    Draft pressure stability metrics.

    Attributes:
        mean_draft_pa: Mean draft pressure
        variance_pa2: Variance of draft readings
        std_deviation_pa: Standard deviation
        max_deviation_from_setpoint_pa: Maximum deviation from setpoint
        integral_absolute_error: IAE metric
        oscillation_count: Number of zero-crossings (oscillations)
        oscillation_frequency_hz: Estimated oscillation frequency
        stability_score: Overall stability score (0-100)
        stability_status: Status classification
    """
    mean_draft_pa: float
    variance_pa2: float
    std_deviation_pa: float
    max_deviation_from_setpoint_pa: float
    integral_absolute_error: float
    oscillation_count: int
    oscillation_frequency_hz: float
    stability_score: float
    stability_status: str


@dataclass
class ControlEffortMetrics:
    """
    Control effort metrics.

    Attributes:
        damper_movement_total_pct: Total damper movement
        damper_movement_rate_pct_s: Damper movement rate
        damper_reversals: Number of direction changes
        pid_output_variance: Variance of PID output
        control_effort_score: Overall control effort score (0-100)
    """
    damper_movement_total_pct: float
    damper_movement_rate_pct_s: float
    damper_reversals: int
    pid_output_variance: Optional[float]
    control_effort_score: float


@dataclass
class DamperPerformance:
    """
    Damper performance indicators.

    Attributes:
        current_position_pct: Current position
        avg_position_pct: Average position over period
        position_range_pct: Range of positions (max - min)
        saturation_time_pct: Time at limits (0 or 100%)
        response_lag_s: Estimated response lag
        hysteresis_detected: Whether hysteresis is detected
        performance_score: Overall damper performance (0-100)
    """
    current_position_pct: float
    avg_position_pct: float
    position_range_pct: float
    saturation_time_pct: float
    response_lag_s: Optional[float]
    hysteresis_detected: bool
    performance_score: float


@dataclass
class FanPerformance:
    """
    Fan performance indicators.

    Attributes:
        current_speed_rpm: Current speed
        speed_ratio_pct: Speed as percentage of rated
        power_consumption_kw: Estimated power consumption
        efficiency_pct: Fan efficiency estimate
        operating_point: Operating point description
        performance_score: Overall fan performance (0-100)
    """
    current_speed_rpm: Optional[float]
    speed_ratio_pct: Optional[float]
    power_consumption_kw: Optional[float]
    efficiency_pct: Optional[float]
    operating_point: str
    performance_score: float


@dataclass
class DraftOutputs:
    """
    Complete draft analysis output.

    Attributes:
        timestamp: Analysis timestamp
        draft_type: Draft configuration type
        control_mode: Control mode
        stability_metrics: Draft stability metrics
        control_effort: Control effort metrics
        damper_performance: Damper performance indicators
        fan_performance: Fan performance indicators (if applicable)
        overall_status: Overall system status
        alerts: Generated alerts
        recommendations: Improvement recommendations
    """
    timestamp: datetime
    draft_type: DraftType
    control_mode: ControlMode
    stability_metrics: DraftStabilityMetrics
    control_effort: ControlEffortMetrics
    damper_performance: DamperPerformance
    fan_performance: Optional[FanPerformance]
    overall_status: str
    alerts: List[str]
    recommendations: List[str]


class DraftAnalyzer(DeterministicCalculator[DraftInputs, DraftOutputs]):
    """
    Deterministic calculator for furnace draft analysis.

    Analyzes draft pressure stability, control effort, damper performance,
    and fan performance using deterministic statistical calculations.
    All calculations are reproducible with SHA-256 provenance tracking.

    Metrics Calculated:
        - Variance and standard deviation of draft
        - Integral Absolute Error (IAE)
        - Oscillation frequency and count
        - Damper movement and reversals
        - Fan efficiency (when data available)

    Example:
        >>> analyzer = DraftAnalyzer(agent_id="GL-007")
        >>> inputs = DraftInputs(
        ...     timestamp=datetime.now(timezone.utc),
        ...     draft_readings_pa=[-25, -24, -26, -25, -23, -27],
        ...     setpoint_pa=-25.0,
        ...     damper_position_pct=45.0
        ... )
        >>> result = analyzer.calculate(inputs)
        >>> print(f"Stability: {result.result.stability_metrics.stability_score:.1f}")
    """

    NAME = "FurnaceDraftAnalyzer"
    VERSION = "1.0.0"

    def __init__(
        self,
        agent_id: str = "GL-007",
        track_provenance: bool = True,
        thresholds: Optional[DraftThresholds] = None,
    ):
        """
        Initialize draft analyzer.

        Args:
            agent_id: Agent identifier for provenance
            track_provenance: Whether to track calculation provenance
            thresholds: Custom draft thresholds
        """
        super().__init__(agent_id, track_provenance)
        self.thresholds = thresholds or DraftThresholds()

    def _validate_inputs(self, inputs: DraftInputs) -> List[str]:
        """
        Validate draft analysis inputs.

        Args:
            inputs: Draft inputs to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate draft readings
        if not inputs.draft_readings_pa:
            errors.append("draft_readings_pa cannot be empty")
        elif len(inputs.draft_readings_pa) < 3:
            errors.append("Need at least 3 draft readings for analysis")

        # Validate setpoint range
        if inputs.setpoint_pa > 0:
            errors.append(
                "setpoint_pa should typically be negative for furnace draft"
            )
        if inputs.setpoint_pa < -100:
            errors.append(
                "setpoint_pa below -100 Pa is unusually high negative draft"
            )

        # Validate damper position
        if not 0 <= inputs.damper_position_pct <= 100:
            errors.append("damper_position_pct must be between 0 and 100")

        # Validate sample interval
        if inputs.sample_interval_s <= 0:
            errors.append("sample_interval_s must be positive")

        # Validate fan data if provided
        if inputs.fan_speed_rpm is not None and inputs.fan_speed_rpm < 0:
            errors.append("fan_speed_rpm cannot be negative")
        if inputs.fan_current_a is not None and inputs.fan_current_a < 0:
            errors.append("fan_current_a cannot be negative")

        # Validate previous damper positions
        for i, pos in enumerate(inputs.previous_damper_positions):
            if not 0 <= pos <= 100:
                errors.append(
                    f"previous_damper_positions[{i}] out of range: {pos}"
                )

        return errors

    def _calculate(self, inputs: DraftInputs, **kwargs: Any) -> DraftOutputs:
        """
        Perform draft analysis.

        This is a DETERMINISTIC calculation using standard statistics.

        Args:
            inputs: Validated draft inputs

        Returns:
            DraftOutputs with all analysis results
        """
        # Calculate stability metrics
        stability_metrics = self._calculate_stability_metrics(inputs)

        # Calculate control effort metrics
        control_effort = self._calculate_control_effort(inputs)

        # Calculate damper performance
        damper_performance = self._calculate_damper_performance(inputs)

        # Calculate fan performance (if data available)
        fan_performance = None
        if inputs.fan_speed_rpm is not None:
            fan_performance = self._calculate_fan_performance(inputs)

        # Generate alerts
        alerts = self._generate_alerts(
            inputs, stability_metrics, control_effort, damper_performance
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            stability_metrics, control_effort, damper_performance, fan_performance
        )

        # Determine overall status
        overall_status = self._determine_overall_status(
            stability_metrics, control_effort, damper_performance
        )

        return DraftOutputs(
            timestamp=inputs.timestamp,
            draft_type=inputs.draft_type,
            control_mode=inputs.control_mode,
            stability_metrics=stability_metrics,
            control_effort=control_effort,
            damper_performance=damper_performance,
            fan_performance=fan_performance,
            overall_status=overall_status,
            alerts=alerts,
            recommendations=recommendations,
        )

    def _calculate_stability_metrics(
        self,
        inputs: DraftInputs,
    ) -> DraftStabilityMetrics:
        """Calculate draft stability metrics."""
        readings = inputs.draft_readings_pa
        n = len(readings)

        # Mean
        mean_draft = sum(readings) / n

        # Variance and standard deviation
        variance = sum((x - mean_draft) ** 2 for x in readings) / n
        std_dev = math.sqrt(variance)

        # Deviation from setpoint
        deviations = [abs(x - inputs.setpoint_pa) for x in readings]
        max_deviation = max(deviations)

        # Integral Absolute Error (IAE)
        iae = sum(deviations) * inputs.sample_interval_s

        # Oscillation detection (zero-crossings around setpoint)
        oscillation_count = 0
        for i in range(1, n):
            prev_sign = readings[i-1] - inputs.setpoint_pa
            curr_sign = readings[i] - inputs.setpoint_pa
            if prev_sign * curr_sign < 0:  # Sign change
                oscillation_count += 1

        # Oscillation frequency
        total_time = (n - 1) * inputs.sample_interval_s
        if total_time > 0:
            # Each full oscillation has 2 zero-crossings
            oscillation_freq = oscillation_count / (2 * total_time)
        else:
            oscillation_freq = 0.0

        # Stability score (0-100)
        # Based on variance, deviation, and oscillations
        variance_score = max(0, 40 * (1 - variance / self.thresholds.max_variance_pa2))
        deviation_score = max(0, 30 * (1 - max_deviation / self.thresholds.max_deviation_pa))
        oscillation_score = max(0, 30 * (1 - oscillation_count / 20))

        stability_score = variance_score + deviation_score + oscillation_score

        # Status classification
        if stability_score >= 80:
            status = "STABLE"
        elif stability_score >= 60:
            status = "ACCEPTABLE"
        elif stability_score >= 40:
            status = "MARGINAL"
        else:
            status = "UNSTABLE"

        return DraftStabilityMetrics(
            mean_draft_pa=round(mean_draft, 3),
            variance_pa2=round(variance, 4),
            std_deviation_pa=round(std_dev, 3),
            max_deviation_from_setpoint_pa=round(max_deviation, 3),
            integral_absolute_error=round(iae, 3),
            oscillation_count=oscillation_count,
            oscillation_frequency_hz=round(oscillation_freq, 4),
            stability_score=round(stability_score, 1),
            stability_status=status,
        )

    def _calculate_control_effort(
        self,
        inputs: DraftInputs,
    ) -> ControlEffortMetrics:
        """Calculate control effort metrics."""
        positions = inputs.previous_damper_positions
        if not positions:
            positions = [inputs.damper_position_pct]

        # Add current position
        all_positions = positions + [inputs.damper_position_pct]
        n = len(all_positions)

        # Total movement
        total_movement = 0.0
        reversals = 0
        prev_direction = 0

        for i in range(1, n):
            movement = all_positions[i] - all_positions[i-1]
            total_movement += abs(movement)

            # Track reversals
            if movement != 0:
                current_direction = 1 if movement > 0 else -1
                if prev_direction != 0 and current_direction != prev_direction:
                    reversals += 1
                prev_direction = current_direction

        # Movement rate
        total_time = (n - 1) * inputs.sample_interval_s
        movement_rate = total_movement / total_time if total_time > 0 else 0.0

        # PID output variance
        pid_variance = None
        if inputs.pid_output_pct is not None:
            # Single value, so variance = 0 for this sample
            # In practice, would track history
            pid_variance = 0.0

        # Control effort score (0-100)
        # Lower movement and fewer reversals = better
        movement_score = max(0, 50 * (1 - total_movement / 100))
        reversal_score = max(0, 50 * (1 - reversals / 10))
        effort_score = movement_score + reversal_score

        return ControlEffortMetrics(
            damper_movement_total_pct=round(total_movement, 2),
            damper_movement_rate_pct_s=round(movement_rate, 3),
            damper_reversals=reversals,
            pid_output_variance=pid_variance,
            control_effort_score=round(effort_score, 1),
        )

    def _calculate_damper_performance(
        self,
        inputs: DraftInputs,
    ) -> DamperPerformance:
        """Calculate damper performance indicators."""
        positions = inputs.previous_damper_positions + [inputs.damper_position_pct]

        # Statistics
        avg_position = sum(positions) / len(positions)
        min_pos = min(positions)
        max_pos = max(positions)
        position_range = max_pos - min_pos

        # Saturation time (at 0% or 100%)
        saturation_count = sum(1 for p in positions if p <= 1 or p >= 99)
        saturation_time_pct = (saturation_count / len(positions)) * 100

        # Hysteresis detection (significant position changes with little effect)
        # Simplified: detect if damper moves significantly but draft doesn't
        hysteresis_detected = False
        if len(positions) > 5 and position_range > 20:
            # Check if variance is still high despite large damper range
            draft_readings = inputs.draft_readings_pa
            draft_variance = sum(
                (x - sum(draft_readings)/len(draft_readings))**2
                for x in draft_readings
            ) / len(draft_readings)
            if draft_variance > self.thresholds.max_variance_pa2:
                hysteresis_detected = True

        # Response lag (would need more data in practice)
        response_lag = None

        # Performance score
        saturation_penalty = min(30, saturation_time_pct * 0.5)
        range_score = max(0, 40 - position_range * 0.4)
        hysteresis_penalty = 30 if hysteresis_detected else 0
        performance_score = 100 - saturation_penalty - (40 - range_score) - hysteresis_penalty

        return DamperPerformance(
            current_position_pct=inputs.damper_position_pct,
            avg_position_pct=round(avg_position, 2),
            position_range_pct=round(position_range, 2),
            saturation_time_pct=round(saturation_time_pct, 1),
            response_lag_s=response_lag,
            hysteresis_detected=hysteresis_detected,
            performance_score=round(max(0, performance_score), 1),
        )

    def _calculate_fan_performance(
        self,
        inputs: DraftInputs,
    ) -> FanPerformance:
        """Calculate fan performance indicators."""
        speed = inputs.fan_speed_rpm
        rated_speed = inputs.fan_rated_speed_rpm or 1800  # Default
        rated_power = inputs.fan_rated_power_kw or 10.0   # Default

        # Speed ratio
        speed_ratio = (speed / rated_speed) * 100 if rated_speed > 0 else None

        # Power consumption estimate (affinity laws: P ~ n^3)
        if speed_ratio is not None:
            power_ratio = (speed / rated_speed) ** 3
            power_consumption = rated_power * power_ratio
        else:
            power_consumption = None

        # Efficiency estimate (simplified)
        # Real calculation would need flow rate and pressure rise
        efficiency = None
        if inputs.fan_current_a is not None and rated_power > 0:
            # Estimate power from current (assuming voltage ~ 460V, 3-phase, PF 0.85)
            measured_power = inputs.fan_current_a * 460 * math.sqrt(3) * 0.85 / 1000
            if measured_power > 0 and power_consumption is not None:
                # Compare to theoretical
                efficiency = min(100, (power_consumption / measured_power) * 100)

        # Operating point
        if speed_ratio is None:
            operating_point = "UNKNOWN"
        elif speed_ratio < 50:
            operating_point = "LOW_LOAD"
        elif speed_ratio < 80:
            operating_point = "NORMAL"
        elif speed_ratio < 95:
            operating_point = "HIGH_LOAD"
        else:
            operating_point = "NEAR_MAXIMUM"

        # Performance score
        if speed_ratio is None:
            performance_score = 50.0  # Unknown
        elif 40 <= speed_ratio <= 90:
            performance_score = 100.0  # Optimal range
        elif speed_ratio < 40:
            performance_score = 70.0  # Low speed (may be oversized)
        else:
            performance_score = 80.0 - (speed_ratio - 90) * 2  # High speed penalty

        return FanPerformance(
            current_speed_rpm=speed,
            speed_ratio_pct=round(speed_ratio, 1) if speed_ratio else None,
            power_consumption_kw=round(power_consumption, 2) if power_consumption else None,
            efficiency_pct=round(efficiency, 1) if efficiency else None,
            operating_point=operating_point,
            performance_score=round(max(0, performance_score), 1),
        )

    def _generate_alerts(
        self,
        inputs: DraftInputs,
        stability: DraftStabilityMetrics,
        effort: ControlEffortMetrics,
        damper: DamperPerformance,
    ) -> List[str]:
        """Generate alerts based on analysis results."""
        alerts = []

        # Stability alerts
        if stability.stability_status == "UNSTABLE":
            alerts.append(
                f"CRITICAL: Draft unstable - variance {stability.variance_pa2:.2f} Pa^2, "
                f"score {stability.stability_score:.0f}"
            )
        elif stability.stability_status == "MARGINAL":
            alerts.append(
                f"WARNING: Marginal draft stability - check control tuning"
            )

        # Deviation alerts
        if stability.max_deviation_from_setpoint_pa > self.thresholds.max_deviation_pa:
            alerts.append(
                f"WARNING: Draft deviation {stability.max_deviation_from_setpoint_pa:.1f} Pa "
                f"exceeds threshold {self.thresholds.max_deviation_pa:.1f} Pa"
            )

        # Safety limits
        for reading in inputs.draft_readings_pa:
            if reading > self.thresholds.max_draft_pa:
                alerts.append(
                    f"DANGER: Positive furnace pressure detected ({reading:.1f} Pa)"
                )
                break
            if reading < self.thresholds.min_draft_pa:
                alerts.append(
                    f"WARNING: Excessive draft ({reading:.1f} Pa) - "
                    f"check for air leaks or over-firing"
                )
                break

        # Control effort alerts
        if effort.damper_reversals > 10:
            alerts.append(
                f"WARNING: High damper reversal count ({effort.damper_reversals}) - "
                f"check PID tuning"
            )

        # Damper alerts
        if damper.saturation_time_pct > 20:
            alerts.append(
                f"WARNING: Damper saturated {damper.saturation_time_pct:.0f}% of time - "
                f"check system sizing"
            )

        if damper.hysteresis_detected:
            alerts.append(
                "WARNING: Damper hysteresis detected - inspect actuator and linkage"
            )

        return alerts

    def _generate_recommendations(
        self,
        stability: DraftStabilityMetrics,
        effort: ControlEffortMetrics,
        damper: DamperPerformance,
        fan: Optional[FanPerformance],
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []

        # Oscillation recommendations
        if stability.oscillation_frequency_hz > 0.1:
            recommendations.append(
                f"Oscillation detected at {stability.oscillation_frequency_hz:.3f} Hz. "
                f"Consider reducing PID gain or increasing derivative term."
            )

        # Control effort recommendations
        if effort.control_effort_score < 50:
            recommendations.append(
                "High control effort detected. Review PID tuning - "
                "reduce proportional gain or add filtering."
            )

        # Damper recommendations
        if damper.avg_position_pct > 80:
            recommendations.append(
                "Damper operating near maximum. Consider increasing fan speed "
                "or checking for obstructions."
            )
        elif damper.avg_position_pct < 20:
            recommendations.append(
                "Damper operating near minimum. Fan may be oversized. "
                "Consider VFD speed reduction."
            )

        # Fan recommendations
        if fan is not None:
            if fan.operating_point == "NEAR_MAXIMUM":
                recommendations.append(
                    "Fan operating near maximum speed. Monitor motor temperature "
                    "and consider capacity upgrade."
                )
            elif fan.operating_point == "LOW_LOAD":
                recommendations.append(
                    "Fan at low speed - good efficiency opportunity. "
                    "Consider trimming impeller or reducing speed setpoint."
                )

        # Stability recommendations
        if stability.stability_status in ["MARGINAL", "UNSTABLE"]:
            recommendations.append(
                "Consider installing draft transmitter closer to measurement point "
                "or adding process damping."
            )

        return recommendations

    def _determine_overall_status(
        self,
        stability: DraftStabilityMetrics,
        effort: ControlEffortMetrics,
        damper: DamperPerformance,
    ) -> str:
        """Determine overall system status."""
        # Weight the scores
        weighted_score = (
            stability.stability_score * 0.5 +
            effort.control_effort_score * 0.3 +
            damper.performance_score * 0.2
        )

        if weighted_score >= 80:
            return "OPTIMAL"
        elif weighted_score >= 60:
            return "ACCEPTABLE"
        elif weighted_score >= 40:
            return "DEGRADED"
        else:
            return "CRITICAL"

    def analyze_trend(
        self,
        historical_readings: List[List[float]],
        timestamps: List[datetime],
        setpoint_pa: float,
    ) -> CalculationResult[Dict[str, Any]]:
        """
        Analyze draft trend over multiple periods.

        Args:
            historical_readings: List of reading arrays
            timestamps: Timestamps for each period
            setpoint_pa: Draft setpoint

        Returns:
            Trend analysis results
        """
        if len(historical_readings) < 2:
            return CalculationResult(
                result=None,
                computation_hash="",
                inputs_hash=self._compute_hash(historical_readings),
                calculator_name=self.NAME,
                calculator_version=self.VERSION,
                is_valid=False,
                warnings=["Need at least 2 periods for trend analysis"],
            )

        # Calculate variance for each period
        variances = []
        means = []
        for readings in historical_readings:
            n = len(readings)
            mean = sum(readings) / n
            variance = sum((x - mean) ** 2 for x in readings) / n
            variances.append(variance)
            means.append(mean)

        # Trend detection (simple linear regression on variance)
        n_periods = len(variances)
        x_mean = (n_periods - 1) / 2
        y_mean = sum(variances) / n_periods

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(variances))
        denominator = sum((i - x_mean) ** 2 for i in range(n_periods))

        if denominator > 0:
            slope = numerator / denominator
        else:
            slope = 0

        # Trend classification
        if slope > 0.5:
            trend = "DEGRADING"
            trend_message = "Draft stability is worsening over time"
        elif slope < -0.5:
            trend = "IMPROVING"
            trend_message = "Draft stability is improving"
        else:
            trend = "STABLE"
            trend_message = "Draft stability is consistent"

        result = {
            "period_count": n_periods,
            "variance_values": [round(v, 4) for v in variances],
            "mean_values": [round(m, 3) for m in means],
            "variance_trend_slope": round(slope, 6),
            "trend_direction": trend,
            "trend_message": trend_message,
            "latest_variance": round(variances[-1], 4),
            "average_variance": round(sum(variances) / len(variances), 4),
        }

        # Compute provenance
        inputs_hash = self._compute_hash({
            "readings": historical_readings,
            "timestamps": [t.isoformat() for t in timestamps],
            "setpoint": setpoint_pa,
        })
        outputs_hash = self._compute_hash(result)
        computation_hash = self._compute_combined_hash(inputs_hash, outputs_hash, {})

        return CalculationResult(
            result=result,
            computation_hash=computation_hash,
            inputs_hash=inputs_hash,
            calculator_name=self.NAME,
            calculator_version=self.VERSION,
            is_valid=True,
        )
