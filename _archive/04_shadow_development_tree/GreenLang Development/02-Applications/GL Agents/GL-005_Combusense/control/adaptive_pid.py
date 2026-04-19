# -*- coding: utf-8 -*-
"""
Adaptive PID Controller for GL-005 CombustionControlAgent

Implements adaptive PID control with online learning capabilities for combustion
systems. This module extends the base PID controller with auto-tuning, gain
scheduling, and Model Reference Adaptive Control (MRAC).

SAFETY NOTICE:
    Adaptive tuning is ONLY for setpoint optimization loops (non-safety-critical).
    Safety-critical control loops MUST use fixed, validated gains as per IEC 61508.

Reference Standards:
- ISA-5.1: Instrumentation Symbols and Identification
- ANSI/ISA-51.1: Process Instrumentation Terminology
- IEC 61508: Functional Safety of E/E/PE Systems
- Astrom & Hagglund: PID Controllers - Theory, Design, and Tuning
- ANSI/ISA-5.9: Controller Algorithm and Performance Test Standard

Mathematical Formulas:
- Relay Feedback (Astrom-Hagglund):
    Ku = 4*d / (pi*a)  where d=relay amplitude, a=oscillation amplitude
    Tu = period of sustained oscillation
- MRAC Update: theta_dot = -gamma * e * phi
- RLS Update: theta = theta + K * (y - phi.T * theta)
    K = P * phi / (lambda + phi.T * P * phi)
    P = (P - K * phi.T * P) / lambda

Example:
    >>> config = AdaptivePIDConfig(kp=1.5, ki=0.3, kd=0.1)
    >>> controller = AdaptivePIDController(config)
    >>> result = controller.calculate(input_data)
    >>> if result.tuning_recommended:
    ...     report = controller.generate_tuning_report()
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from decimal import Decimal, ROUND_HALF_UP
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from enum import Enum
import math
import logging
import hashlib
import statistics
from dataclasses import dataclass, field as dataclass_field
from collections import deque
from datetime import datetime
import copy

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class AdaptiveTuningMethod(str, Enum):
    """Adaptive tuning methods available"""
    RELAY_FEEDBACK = "relay_feedback"  # Astrom-Hagglund relay feedback
    MRAC = "mrac"  # Model Reference Adaptive Control
    RLS = "rls"  # Recursive Least Squares
    GAIN_SCHEDULING = "gain_scheduling"  # Gain scheduling based on operating point
    DISABLED = "disabled"  # No adaptive tuning (fixed gains)


class ControlMode(str, Enum):
    """Control modes"""
    MANUAL = "manual"
    AUTO = "auto"
    CASCADE = "cascade"
    ADAPTIVE = "adaptive"


class TuningState(str, Enum):
    """Current tuning state"""
    IDLE = "idle"  # Normal operation, not tuning
    RELAY_TEST = "relay_test"  # Running relay feedback test
    ANALYZING = "analyzing"  # Analyzing tuning data
    APPLYING = "applying"  # Applying new gains
    WAITING_APPROVAL = "waiting_approval"  # Waiting for operator approval
    FAILED = "failed"  # Tuning failed


class SafetyLevel(str, Enum):
    """Safety criticality level of control loop"""
    NON_CRITICAL = "non_critical"  # Adaptive tuning allowed
    LOW_CRITICAL = "low_critical"  # Adaptive with limits
    HIGH_CRITICAL = "high_critical"  # Fixed gains only
    SAFETY_CRITICAL = "safety_critical"  # SIS level, no adaptation


# =============================================================================
# DATA CLASSES AND MODELS
# =============================================================================

@dataclass
class GainScheduleEntry:
    """Entry in the gain schedule table"""
    operating_point_min: float  # Minimum operating point (e.g., % load)
    operating_point_max: float  # Maximum operating point
    kp: float  # Proportional gain
    ki: float  # Integral gain
    kd: float  # Derivative gain
    description: str = ""  # Description of operating region

    def contains(self, operating_point: float) -> bool:
        """Check if operating point is within this entry's range"""
        return self.operating_point_min <= operating_point <= self.operating_point_max


@dataclass
class RelayFeedbackResult:
    """Results from relay feedback auto-tuning test"""
    ultimate_gain_ku: float
    ultimate_period_tu: float
    oscillation_amplitude: float
    oscillation_count: int
    test_duration_sec: float
    success: bool
    error_message: Optional[str] = None

    # Derived gains using Ziegler-Nichols formulas
    kp_zn: Optional[float] = None  # Kp = 0.6 * Ku
    ki_zn: Optional[float] = None  # Ki = 1.2 * Ku / Tu
    kd_zn: Optional[float] = None  # Kd = 0.075 * Ku * Tu

    # Derived gains using Tyreus-Luyben (more conservative)
    kp_tl: Optional[float] = None  # Kp = 0.45 * Ku
    ki_tl: Optional[float] = None  # Ki = 0.54 * Ku / Tu
    kd_tl: Optional[float] = None  # Kd = 0.15 * Ku * Tu


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics for PID controller"""
    # Error integrals
    ise: float = 0.0  # Integral Squared Error: integral(e^2 dt)
    iae: float = 0.0  # Integral Absolute Error: integral(|e| dt)
    itae: float = 0.0  # Integral Time-weighted Absolute Error: integral(t*|e| dt)

    # Step response metrics
    overshoot_percent: float = 0.0  # Maximum overshoot percentage
    settling_time_sec: float = 0.0  # Time to settle within 2% of setpoint
    rise_time_sec: float = 0.0  # Time to rise from 10% to 90%
    peak_time_sec: float = 0.0  # Time to first peak

    # Steady-state metrics
    steady_state_error: float = 0.0  # Error at steady state
    control_effort: float = 0.0  # Integral of control signal squared

    # Variability metrics
    error_variance: float = 0.0
    output_variance: float = 0.0

    # Time tracking
    measurement_start_time: float = 0.0
    measurement_duration_sec: float = 0.0
    sample_count: int = 0


class SafetyConstraints(BaseModel):
    """Safety constraints for adaptive tuning"""

    # Gain bounds (absolute limits)
    kp_min: float = Field(default=0.01, ge=0, description="Minimum Kp")
    kp_max: float = Field(default=100.0, gt=0, description="Maximum Kp")
    ki_min: float = Field(default=0.0, ge=0, description="Minimum Ki")
    ki_max: float = Field(default=50.0, ge=0, description="Maximum Ki")
    kd_min: float = Field(default=0.0, ge=0, description="Minimum Kd")
    kd_max: float = Field(default=25.0, ge=0, description="Maximum Kd")

    # Rate of change limits (per adaptation cycle)
    max_kp_change_rate: float = Field(
        default=0.1,
        ge=0,
        le=1.0,
        description="Maximum Kp change per cycle (fraction of current value)"
    )
    max_ki_change_rate: float = Field(
        default=0.1,
        ge=0,
        le=1.0,
        description="Maximum Ki change per cycle (fraction of current value)"
    )
    max_kd_change_rate: float = Field(
        default=0.1,
        ge=0,
        le=1.0,
        description="Maximum Kd change per cycle (fraction of current value)"
    )

    # Stability thresholds
    max_oscillation_amplitude_percent: float = Field(
        default=20.0,
        ge=0,
        description="Maximum oscillation amplitude before fallback"
    )
    max_overshoot_percent: float = Field(
        default=30.0,
        ge=0,
        description="Maximum overshoot before fallback"
    )

    # Operator approval thresholds
    require_approval_kp_change_percent: float = Field(
        default=50.0,
        ge=0,
        description="Require approval if Kp changes more than this %"
    )
    require_approval_ki_change_percent: float = Field(
        default=50.0,
        ge=0,
        description="Require approval if Ki changes more than this %"
    )
    require_approval_kd_change_percent: float = Field(
        default=50.0,
        ge=0,
        description="Require approval if Kd changes more than this %"
    )

    # Safety level
    safety_level: SafetyLevel = Field(
        default=SafetyLevel.NON_CRITICAL,
        description="Safety criticality level"
    )

    @field_validator('kp_max')
    @classmethod
    def validate_kp_bounds(cls, v: float, info: ValidationInfo):
        """Ensure max > min"""
        if info.data.get('kp_min') is not None and v <= info.data['kp_min']:
            raise ValueError("kp_max must be greater than kp_min")
        return v


class AdaptivePIDConfig(BaseModel):
    """Configuration for adaptive PID controller"""

    # Initial PID gains
    kp: float = Field(default=1.0, ge=0, description="Initial proportional gain")
    ki: float = Field(default=0.0, ge=0, description="Initial integral gain")
    kd: float = Field(default=0.0, ge=0, description="Initial derivative gain")

    # Output limits
    output_min: float = Field(default=0.0, description="Minimum output")
    output_max: float = Field(default=100.0, description="Maximum output")

    # Adaptive tuning configuration
    tuning_method: AdaptiveTuningMethod = Field(
        default=AdaptiveTuningMethod.DISABLED,
        description="Adaptive tuning method"
    )
    enable_online_learning: bool = Field(
        default=False,
        description="Enable continuous online parameter adjustment"
    )

    # Safety constraints
    safety_constraints: SafetyConstraints = Field(
        default_factory=SafetyConstraints,
        description="Safety constraints for adaptation"
    )

    # Relay feedback parameters
    relay_amplitude: float = Field(
        default=10.0,
        gt=0,
        description="Relay amplitude for auto-tuning"
    )
    relay_hysteresis: float = Field(
        default=0.5,
        ge=0,
        description="Relay hysteresis for noise rejection"
    )
    relay_min_cycles: int = Field(
        default=3,
        ge=2,
        description="Minimum oscillation cycles for valid tuning"
    )
    relay_max_duration_sec: float = Field(
        default=300.0,
        gt=0,
        description="Maximum relay test duration"
    )

    # MRAC parameters
    mrac_gamma: float = Field(
        default=0.01,
        gt=0,
        le=1.0,
        description="MRAC adaptation gain"
    )
    mrac_reference_model_wn: float = Field(
        default=1.0,
        gt=0,
        description="Reference model natural frequency (rad/s)"
    )
    mrac_reference_model_zeta: float = Field(
        default=0.7,
        ge=0,
        le=1.0,
        description="Reference model damping ratio"
    )

    # RLS parameters
    rls_forgetting_factor: float = Field(
        default=0.99,
        gt=0,
        le=1.0,
        description="RLS forgetting factor (0.95-0.99 for tracking)"
    )
    rls_initial_covariance: float = Field(
        default=1000.0,
        gt=0,
        description="Initial covariance matrix diagonal value"
    )

    # Gain scheduling
    gain_schedule: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Gain schedule table entries"
    )

    # Performance monitoring
    performance_window_sec: float = Field(
        default=60.0,
        gt=0,
        description="Window for performance metrics calculation"
    )

    # Fallback configuration
    fallback_gains: Optional[Dict[str, float]] = Field(
        default=None,
        description="Fallback gains for stability issues"
    )

    @field_validator('output_max')
    @classmethod
    def validate_output_limits(cls, v: float, info: ValidationInfo):
        """Ensure max > min"""
        if info.data.get('output_min') is not None and v <= info.data['output_min']:
            raise ValueError("output_max must be greater than output_min")
        return v


class AdaptivePIDInput(BaseModel):
    """Input for adaptive PID controller"""

    # Process variables
    setpoint: float = Field(..., description="Target setpoint")
    process_variable: float = Field(..., description="Current measured value")
    timestamp: float = Field(..., ge=0, description="Current timestamp (seconds)")

    # Control mode
    control_mode: ControlMode = Field(
        default=ControlMode.ADAPTIVE,
        description="Control mode"
    )
    manual_output: Optional[float] = Field(
        None,
        description="Manual output (if in manual mode)"
    )

    # Operating point for gain scheduling
    operating_point: float = Field(
        default=100.0,
        ge=0,
        description="Current operating point (% of rated capacity)"
    )

    # Adaptive tuning control
    start_relay_test: bool = Field(
        default=False,
        description="Start relay feedback test"
    )
    abort_relay_test: bool = Field(
        default=False,
        description="Abort current relay test"
    )
    approve_tuning: bool = Field(
        default=False,
        description="Approve pending tuning changes"
    )
    reject_tuning: bool = Field(
        default=False,
        description="Reject pending tuning changes"
    )

    # Output limits override
    output_min: Optional[float] = Field(None, description="Override output min")
    output_max: Optional[float] = Field(None, description="Override output max")


class AdaptivePIDOutput(BaseModel):
    """Output from adaptive PID controller"""

    # Control output
    control_output: float = Field(..., description="Controller output")

    # Error terms
    error: float = Field(..., description="Current error")
    error_integral: float = Field(..., description="Integral error")
    error_derivative: float = Field(..., description="Derivative error")

    # PID contributions
    p_term: float = Field(..., description="Proportional contribution")
    i_term: float = Field(..., description="Integral contribution")
    d_term: float = Field(..., description="Derivative contribution")

    # Current gains
    current_kp: float = Field(..., description="Current Kp")
    current_ki: float = Field(..., description="Current Ki")
    current_kd: float = Field(..., description="Current Kd")

    # Tuning state
    tuning_state: TuningState = Field(..., description="Current tuning state")
    tuning_method: AdaptiveTuningMethod = Field(..., description="Active tuning method")

    # Gain scheduling
    gain_scheduled: bool = Field(
        default=False,
        description="Whether gains were scheduled"
    )
    active_schedule_region: Optional[str] = Field(
        None,
        description="Active gain schedule region"
    )

    # Performance metrics
    performance_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Current performance metrics"
    )

    # Safety status
    output_saturated: bool = Field(..., description="Output hit limits")
    anti_windup_active: bool = Field(..., description="Anti-windup active")
    stability_warning: bool = Field(
        default=False,
        description="Stability issue detected"
    )
    fallback_active: bool = Field(
        default=False,
        description="Using fallback gains"
    )

    # Tuning recommendations
    tuning_recommended: bool = Field(
        default=False,
        description="Tuning is recommended"
    )
    tuning_recommendation: Optional[str] = Field(
        None,
        description="Tuning recommendation message"
    )
    requires_operator_approval: bool = Field(
        default=False,
        description="Pending changes need approval"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")


class TuningReport(BaseModel):
    """Comprehensive tuning report"""

    # Report metadata
    report_timestamp: str = Field(..., description="Report generation timestamp")
    report_id: str = Field(..., description="Unique report identifier")

    # Current configuration
    current_gains: Dict[str, float] = Field(..., description="Current PID gains")
    tuning_method: AdaptiveTuningMethod = Field(..., description="Tuning method used")

    # Recommended gains
    recommended_gains: Optional[Dict[str, float]] = Field(
        None,
        description="Recommended PID gains"
    )
    gains_change_percent: Optional[Dict[str, float]] = Field(
        None,
        description="Percentage change in gains"
    )

    # Performance analysis
    current_performance: Dict[str, float] = Field(
        ...,
        description="Current performance metrics"
    )
    predicted_performance: Optional[Dict[str, float]] = Field(
        None,
        description="Predicted performance with new gains"
    )

    # Process characteristics
    process_characteristics: Dict[str, float] = Field(
        default_factory=dict,
        description="Identified process characteristics"
    )

    # Recommendations
    tuning_quality_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall tuning quality score"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Tuning recommendations"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Tuning warnings"
    )

    # Safety assessment
    safety_assessment: str = Field(
        ...,
        description="Safety assessment of proposed changes"
    )
    requires_approval: bool = Field(
        ...,
        description="Whether operator approval is required"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")


# =============================================================================
# RECURSIVE LEAST SQUARES ESTIMATOR
# =============================================================================

class RLSEstimator:
    """
    Recursive Least Squares Parameter Estimator for online system identification.

    Implements RLS with exponential forgetting for tracking time-varying dynamics.

    Mathematical Formulation:
        theta_hat(k) = theta_hat(k-1) + K(k) * [y(k) - phi(k)^T * theta_hat(k-1)]
        K(k) = P(k-1) * phi(k) / [lambda + phi(k)^T * P(k-1) * phi(k)]
        P(k) = [P(k-1) - K(k) * phi(k)^T * P(k-1)] / lambda

    Where:
        theta_hat = parameter estimates
        phi = regressor vector
        y = system output
        K = Kalman gain
        P = covariance matrix
        lambda = forgetting factor (0.95-0.99)
    """

    def __init__(
        self,
        num_parameters: int = 3,
        forgetting_factor: float = 0.99,
        initial_covariance: float = 1000.0,
        parameter_bounds: Optional[List[Tuple[float, float]]] = None
    ):
        """
        Initialize RLS estimator.

        Args:
            num_parameters: Number of parameters to estimate
            forgetting_factor: Forgetting factor lambda (0.95-0.99)
            initial_covariance: Initial diagonal value of covariance matrix P
            parameter_bounds: Optional bounds for each parameter [(min, max), ...]
        """
        self.logger = logging.getLogger(__name__)

        self.n = num_parameters
        self.lambda_factor = forgetting_factor

        # Initialize parameter estimates to zeros
        self.theta = [0.0] * self.n

        # Initialize covariance matrix as diagonal
        self.P = [[0.0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            self.P[i][i] = initial_covariance

        # Parameter bounds
        self.bounds = parameter_bounds or [(0.0, 100.0)] * self.n

        # History for analysis
        self.theta_history = deque(maxlen=1000)
        self.residual_history = deque(maxlen=1000)

        self.logger.info(
            f"RLS estimator initialized: n={num_parameters}, "
            f"lambda={forgetting_factor}, P0={initial_covariance}"
        )

    def update(
        self,
        y: float,
        phi: List[float]
    ) -> Tuple[List[float], float]:
        """
        Update parameter estimates with new measurement.

        Args:
            y: System output measurement
            phi: Regressor vector [phi_1, phi_2, ..., phi_n]

        Returns:
            Tuple of (updated parameters, prediction error)
        """
        if len(phi) != self.n:
            raise ValueError(f"Regressor dimension {len(phi)} != {self.n}")

        # Prediction error (innovation)
        y_hat = sum(phi[i] * self.theta[i] for i in range(self.n))
        error = y - y_hat

        # Calculate P * phi
        P_phi = [sum(self.P[i][j] * phi[j] for j in range(self.n)) for i in range(self.n)]

        # Calculate phi^T * P * phi (scalar)
        phi_P_phi = sum(phi[i] * P_phi[i] for i in range(self.n))

        # Kalman gain K = P * phi / (lambda + phi^T * P * phi)
        denominator = self.lambda_factor + phi_P_phi
        if abs(denominator) < 1e-10:
            denominator = 1e-10  # Prevent division by zero

        K = [P_phi[i] / denominator for i in range(self.n)]

        # Update parameter estimates
        theta_new = [self.theta[i] + K[i] * error for i in range(self.n)]

        # Apply bounds
        for i in range(self.n):
            theta_new[i] = max(self.bounds[i][0], min(self.bounds[i][1], theta_new[i]))

        self.theta = theta_new

        # Update covariance matrix P = (P - K * phi^T * P) / lambda
        # First calculate K * phi^T * P
        for i in range(self.n):
            for j in range(self.n):
                self.P[i][j] = (self.P[i][j] - K[i] * P_phi[j]) / self.lambda_factor

        # Ensure positive definiteness
        for i in range(self.n):
            if self.P[i][i] < 1e-6:
                self.P[i][i] = 1e-6

        # Store history
        self.theta_history.append(list(self.theta))
        self.residual_history.append(error)

        return list(self.theta), error

    def get_parameters(self) -> List[float]:
        """Get current parameter estimates"""
        return list(self.theta)

    def get_covariance_trace(self) -> float:
        """Get trace of covariance matrix (uncertainty indicator)"""
        return sum(self.P[i][i] for i in range(self.n))

    def reset(self, initial_covariance: float = 1000.0) -> None:
        """Reset estimator to initial state"""
        self.theta = [0.0] * self.n
        for i in range(self.n):
            for j in range(self.n):
                self.P[i][j] = initial_covariance if i == j else 0.0
        self.theta_history.clear()
        self.residual_history.clear()


# =============================================================================
# MODEL REFERENCE ADAPTIVE CONTROL
# =============================================================================

class MRACController:
    """
    Model Reference Adaptive Control (MRAC) for PID gain adjustment.

    The controller adjusts PID gains to make the closed-loop system
    behavior match a desired reference model.

    Reference Model (second-order):
        G_m(s) = wn^2 / (s^2 + 2*zeta*wn*s + wn^2)

    Adaptation Law (MIT rule):
        d(theta)/dt = -gamma * e * phi

    Where:
        theta = [Kp, Ki, Kd] = adaptive parameters
        e = y - y_m = tracking error (actual - reference model output)
        phi = regressor vector
        gamma = adaptation gain
    """

    def __init__(
        self,
        gamma: float = 0.01,
        reference_wn: float = 1.0,
        reference_zeta: float = 0.7,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        """
        Initialize MRAC controller.

        Args:
            gamma: Adaptation gain (smaller = slower, more stable)
            reference_wn: Reference model natural frequency (rad/s)
            reference_zeta: Reference model damping ratio
            bounds: Bounds for gains {'kp': (min, max), 'ki': ..., 'kd': ...}
        """
        self.logger = logging.getLogger(__name__)

        self.gamma = gamma
        self.wn = reference_wn
        self.zeta = reference_zeta

        # Default bounds
        self.bounds = bounds or {
            'kp': (0.01, 100.0),
            'ki': (0.0, 50.0),
            'kd': (0.0, 25.0)
        }

        # Reference model state
        self.x_m1 = 0.0  # Reference model state 1
        self.x_m2 = 0.0  # Reference model state 2
        self.y_m = 0.0   # Reference model output

        # Previous values for adaptation
        self.prev_error = 0.0
        self.integral_error = 0.0

        # Adaptation history
        self.adaptation_history = deque(maxlen=1000)

        self.logger.info(
            f"MRAC initialized: gamma={gamma}, wn={reference_wn}, zeta={reference_zeta}"
        )

    def update_reference_model(self, r: float, dt: float) -> float:
        """
        Update reference model state with setpoint input.

        Uses discretized second-order reference model.

        Args:
            r: Setpoint (reference input)
            dt: Time step

        Returns:
            Reference model output y_m
        """
        if dt <= 0:
            return self.y_m

        # Discretized state-space update
        # dx1/dt = x2
        # dx2/dt = wn^2 * (r - x1) - 2*zeta*wn*x2

        wn2 = self.wn * self.wn

        # Euler integration
        dx1 = self.x_m2
        dx2 = wn2 * (r - self.x_m1) - 2 * self.zeta * self.wn * self.x_m2

        self.x_m1 += dx1 * dt
        self.x_m2 += dx2 * dt

        self.y_m = self.x_m1

        return self.y_m

    def adapt_gains(
        self,
        kp: float,
        ki: float,
        kd: float,
        error: float,
        tracking_error: float,
        dt: float
    ) -> Tuple[float, float, float]:
        """
        Adapt PID gains using MRAC update law.

        Args:
            kp, ki, kd: Current PID gains
            error: Control error (setpoint - process_variable)
            tracking_error: Reference model tracking error (y - y_m)
            dt: Time step

        Returns:
            Tuple of adapted (kp, ki, kd)
        """
        if dt <= 0:
            return kp, ki, kd

        # Calculate regressors (simplified)
        phi_kp = error
        phi_ki = self.integral_error
        phi_kd = (error - self.prev_error) / dt if dt > 0 else 0.0

        # MIT rule adaptation: d(theta)/dt = -gamma * e * phi
        delta_kp = -self.gamma * tracking_error * phi_kp * dt
        delta_ki = -self.gamma * tracking_error * phi_ki * dt * 0.1  # Slower integral adaptation
        delta_kd = -self.gamma * tracking_error * phi_kd * dt * 0.1  # Slower derivative adaptation

        # Update gains
        new_kp = kp + delta_kp
        new_ki = ki + delta_ki
        new_kd = kd + delta_kd

        # Apply bounds
        new_kp = max(self.bounds['kp'][0], min(self.bounds['kp'][1], new_kp))
        new_ki = max(self.bounds['ki'][0], min(self.bounds['ki'][1], new_ki))
        new_kd = max(self.bounds['kd'][0], min(self.bounds['kd'][1], new_kd))

        # Update internal state
        self.integral_error += error * dt
        self.prev_error = error

        # Store history
        self.adaptation_history.append({
            'tracking_error': tracking_error,
            'delta_kp': delta_kp,
            'delta_ki': delta_ki,
            'delta_kd': delta_kd
        })

        return new_kp, new_ki, new_kd

    def reset(self) -> None:
        """Reset MRAC state"""
        self.x_m1 = 0.0
        self.x_m2 = 0.0
        self.y_m = 0.0
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.adaptation_history.clear()


# =============================================================================
# TUNING ASSISTANT
# =============================================================================

class TuningAssistant:
    """
    Tuning assistant that provides recommendations and generates reports.

    Analyzes process characteristics and performance metrics to suggest
    optimal PID gains and tuning strategies.
    """

    def __init__(self):
        """Initialize tuning assistant"""
        self.logger = logging.getLogger(__name__)

    def suggest_initial_gains(
        self,
        process_gain: float,
        time_constant: float,
        dead_time: float,
        response_type: str = "moderate"
    ) -> Dict[str, float]:
        """
        Suggest initial PID gains based on process characteristics.

        Uses first-order plus dead-time (FOPDT) model rules.

        Args:
            process_gain: Process steady-state gain K
            time_constant: Process time constant tau (seconds)
            dead_time: Process dead time L (seconds)
            response_type: "aggressive", "moderate", or "conservative"

        Returns:
            Dictionary with suggested kp, ki, kd
        """
        if time_constant <= 0:
            time_constant = 1.0
        if dead_time <= 0:
            dead_time = 0.1

        # Controllability ratio
        controllability = dead_time / time_constant

        # SIMC (Simple Internal Model Control) tuning rules
        # Kc = tau / (K * (tc + L))  where tc is closed-loop time constant

        # Response type determines closed-loop time constant
        tc_multipliers = {
            "aggressive": 0.5,
            "moderate": 1.0,
            "conservative": 2.0
        }
        tc_mult = tc_multipliers.get(response_type, 1.0)

        tc = max(dead_time * tc_mult, time_constant * 0.2)  # Closed-loop time constant

        # Calculate gains
        if process_gain != 0:
            kp = time_constant / (process_gain * (tc + dead_time))
        else:
            kp = 1.0

        ki = kp / time_constant
        kd = kp * dead_time * 0.5

        # Adjust based on controllability
        if controllability > 0.5:
            # Difficult process - reduce gains
            kp *= 0.7
            ki *= 0.5
            kd *= 0.5

        self.logger.info(
            f"Suggested gains for K={process_gain}, tau={time_constant}, L={dead_time}: "
            f"Kp={kp:.4f}, Ki={ki:.4f}, Kd={kd:.4f}"
        )

        return {
            'kp': round(kp, 4),
            'ki': round(ki, 4),
            'kd': round(kd, 4)
        }

    def recommend_tuning(
        self,
        current_metrics: PerformanceMetrics,
        current_gains: Dict[str, float],
        oscillation_detected: bool = False,
        slow_response: bool = False
    ) -> Tuple[Optional[Dict[str, float]], List[str]]:
        """
        Recommend tuning adjustments based on performance metrics.

        Args:
            current_metrics: Current performance metrics
            current_gains: Current PID gains
            oscillation_detected: Whether oscillations detected
            slow_response: Whether response is too slow

        Returns:
            Tuple of (recommended gains or None, list of recommendations)
        """
        recommendations = []
        new_gains = None

        kp = current_gains.get('kp', 1.0)
        ki = current_gains.get('ki', 0.0)
        kd = current_gains.get('kd', 0.0)

        # Check for oscillation issues
        if oscillation_detected or current_metrics.overshoot_percent > 30:
            recommendations.append(
                "Oscillation detected - reduce Kp by 20% and increase Kd by 10%"
            )
            new_gains = {
                'kp': round(kp * 0.8, 4),
                'ki': round(ki * 0.9, 4),
                'kd': round(kd * 1.1, 4)
            }

        # Check for slow response
        elif slow_response or current_metrics.rise_time_sec > 60:
            recommendations.append(
                "Slow response detected - increase Kp by 20%"
            )
            new_gains = {
                'kp': round(kp * 1.2, 4),
                'ki': ki,
                'kd': kd
            }

        # Check for large steady-state error (integral action needed)
        elif abs(current_metrics.steady_state_error) > 1.0 and ki < 0.01:
            recommendations.append(
                "Large steady-state error - add integral action"
            )
            new_gains = {
                'kp': kp,
                'ki': round(kp * 0.1, 4),  # Initial Ki based on Kp
                'kd': kd
            }

        # Check for excessive noise amplification
        elif current_metrics.output_variance > 100:
            recommendations.append(
                "High output variance - reduce Kd and add derivative filtering"
            )
            new_gains = {
                'kp': kp,
                'ki': ki,
                'kd': round(kd * 0.5, 4)
            }

        # System performing well
        else:
            recommendations.append(
                "System performance acceptable - no tuning changes recommended"
            )

        return new_gains, recommendations

    def generate_tuning_report(
        self,
        current_gains: Dict[str, float],
        recommended_gains: Optional[Dict[str, float]],
        metrics: PerformanceMetrics,
        process_info: Dict[str, float],
        tuning_method: AdaptiveTuningMethod,
        safety_constraints: SafetyConstraints
    ) -> TuningReport:
        """
        Generate comprehensive tuning report.

        Args:
            current_gains: Current PID gains
            recommended_gains: Recommended PID gains
            metrics: Current performance metrics
            process_info: Process characteristics (Ku, Tu, etc.)
            tuning_method: Tuning method used
            safety_constraints: Safety constraints

        Returns:
            TuningReport with complete analysis
        """
        report_time = datetime.utcnow().isoformat() + "Z"
        report_id = hashlib.sha256(f"{report_time}_{current_gains}".encode()).hexdigest()[:16]

        # Calculate gains change percentage
        gains_change = None
        requires_approval = False

        if recommended_gains:
            gains_change = {}
            for gain_name in ['kp', 'ki', 'kd']:
                current = current_gains.get(gain_name, 0.0)
                recommended = recommended_gains.get(gain_name, 0.0)
                if current != 0:
                    change_percent = abs(recommended - current) / current * 100
                else:
                    change_percent = 100.0 if recommended != 0 else 0.0
                gains_change[gain_name] = round(change_percent, 1)

            # Check if approval is needed
            if gains_change.get('kp', 0) > safety_constraints.require_approval_kp_change_percent:
                requires_approval = True
            if gains_change.get('ki', 0) > safety_constraints.require_approval_ki_change_percent:
                requires_approval = True
            if gains_change.get('kd', 0) > safety_constraints.require_approval_kd_change_percent:
                requires_approval = True

        # Generate recommendations
        recommendations = []
        warnings = []

        if metrics.overshoot_percent > 25:
            recommendations.append("Consider reducing Kp to decrease overshoot")
        if metrics.settling_time_sec > 120:
            recommendations.append("Consider increasing gains for faster response")
        if metrics.steady_state_error > 1.0:
            recommendations.append("Increase Ki to reduce steady-state error")

        if safety_constraints.safety_level == SafetyLevel.SAFETY_CRITICAL:
            warnings.append("WARNING: This is a safety-critical loop - manual gain changes only")
        if requires_approval:
            warnings.append("Large gain changes proposed - operator approval required")

        # Calculate tuning quality score
        quality_score = self._calculate_tuning_quality_score(metrics)

        # Safety assessment
        if safety_constraints.safety_level == SafetyLevel.SAFETY_CRITICAL:
            safety_assessment = "SAFETY CRITICAL: No automatic tuning permitted"
        elif requires_approval:
            safety_assessment = "APPROVAL REQUIRED: Significant gain changes proposed"
        else:
            safety_assessment = "SAFE: Changes within acceptable limits"

        # Current performance dict
        current_performance = {
            'ise': metrics.ise,
            'iae': metrics.iae,
            'overshoot_percent': metrics.overshoot_percent,
            'settling_time_sec': metrics.settling_time_sec,
            'rise_time_sec': metrics.rise_time_sec,
            'steady_state_error': metrics.steady_state_error
        }

        # Provenance hash
        provenance_data = {
            'report_id': report_id,
            'current_gains': current_gains,
            'recommended_gains': recommended_gains,
            'current_performance': current_performance
        }
        provenance_hash = hashlib.sha256(str(provenance_data).encode()).hexdigest()

        return TuningReport(
            report_timestamp=report_time,
            report_id=report_id,
            current_gains=current_gains,
            tuning_method=tuning_method,
            recommended_gains=recommended_gains,
            gains_change_percent=gains_change,
            current_performance=current_performance,
            predicted_performance=None,  # Would require simulation
            process_characteristics=process_info,
            tuning_quality_score=quality_score,
            recommendations=recommendations,
            warnings=warnings,
            safety_assessment=safety_assessment,
            requires_approval=requires_approval,
            provenance_hash=provenance_hash
        )

    def _calculate_tuning_quality_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate tuning quality score from 0-100"""
        score = 100.0

        # Penalize overshoot
        if metrics.overshoot_percent > 25:
            score -= min(30, (metrics.overshoot_percent - 25) * 1.5)
        elif metrics.overshoot_percent > 10:
            score -= (metrics.overshoot_percent - 10)

        # Penalize slow settling
        if metrics.settling_time_sec > 120:
            score -= min(20, (metrics.settling_time_sec - 120) / 10)

        # Penalize steady-state error
        if abs(metrics.steady_state_error) > 1:
            score -= min(20, abs(metrics.steady_state_error) * 5)

        # Penalize high variability
        if metrics.error_variance > 10:
            score -= min(15, metrics.error_variance / 2)

        return max(0, min(100, score))


# =============================================================================
# ADAPTIVE PID CONTROLLER
# =============================================================================

class AdaptivePIDController:
    """
    Adaptive PID Controller with Online Learning.

    This controller extends the base PID with adaptive tuning capabilities:
    - Relay feedback auto-tuning (Astrom-Hagglund method)
    - Model Reference Adaptive Control (MRAC)
    - Recursive Least Squares (RLS) parameter estimation
    - Gain scheduling based on operating point

    SAFETY NOTICE:
        Adaptive tuning is ONLY for non-safety-critical control loops.
        For safety-critical loops (SIL-rated), use fixed, validated gains.

    Example:
        >>> config = AdaptivePIDConfig(kp=1.5, ki=0.3, kd=0.1)
        >>> controller = AdaptivePIDController(config)
        >>>
        >>> # Start relay feedback test
        >>> input_data = AdaptivePIDInput(
        ...     setpoint=1200.0,
        ...     process_variable=1150.0,
        ...     timestamp=0.0,
        ...     start_relay_test=True
        ... )
        >>> result = controller.calculate(input_data)
        >>>
        >>> # After test completes, get tuning report
        >>> if result.tuning_state == TuningState.WAITING_APPROVAL:
        ...     report = controller.get_tuning_assistant().generate_tuning_report(...)
    """

    def __init__(self, config: AdaptivePIDConfig):
        """
        Initialize Adaptive PID Controller.

        Args:
            config: Controller configuration
        """
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.config = config

        # Current gains
        self.kp = config.kp
        self.ki = config.ki
        self.kd = config.kd

        # Fallback gains (for stability issues)
        self.fallback_gains = config.fallback_gains or {
            'kp': config.kp,
            'ki': config.ki,
            'kd': config.kd
        }

        # Output limits
        self.output_min = config.output_min
        self.output_max = config.output_max

        # PID internal state
        self.error_previous = 0.0
        self.error_integral = 0.0
        self.derivative_filtered = 0.0
        self.output_previous = 0.0
        self.time_previous: Optional[float] = None

        # Anti-windup state
        self.windup_active = False

        # Tuning state
        self.tuning_state = TuningState.IDLE
        self.tuning_method = config.tuning_method

        # Relay feedback test state
        self.relay_active = False
        self.relay_data: List[Tuple[float, float]] = []  # (timestamp, output)
        self.relay_peaks: List[Tuple[float, float]] = []  # (timestamp, value)
        self.relay_start_time: Optional[float] = None
        self.relay_last_switch_time: Optional[float] = None
        self.relay_switch_count = 0
        self.relay_current_direction = 1  # +1 or -1

        # Relay feedback results
        self.relay_result: Optional[RelayFeedbackResult] = None

        # Pending gains (awaiting approval)
        self.pending_gains: Optional[Dict[str, float]] = None

        # Gain schedule
        self.gain_schedule: List[GainScheduleEntry] = []
        for entry_dict in config.gain_schedule:
            self.gain_schedule.append(GainScheduleEntry(
                operating_point_min=entry_dict.get('operating_point_min', 0),
                operating_point_max=entry_dict.get('operating_point_max', 100),
                kp=entry_dict.get('kp', config.kp),
                ki=entry_dict.get('ki', config.ki),
                kd=entry_dict.get('kd', config.kd),
                description=entry_dict.get('description', '')
            ))

        # MRAC controller (if enabled)
        self.mrac: Optional[MRACController] = None
        if config.tuning_method == AdaptiveTuningMethod.MRAC:
            self.mrac = MRACController(
                gamma=config.mrac_gamma,
                reference_wn=config.mrac_reference_model_wn,
                reference_zeta=config.mrac_reference_model_zeta,
                bounds={
                    'kp': (config.safety_constraints.kp_min, config.safety_constraints.kp_max),
                    'ki': (config.safety_constraints.ki_min, config.safety_constraints.ki_max),
                    'kd': (config.safety_constraints.kd_min, config.safety_constraints.kd_max)
                }
            )

        # RLS estimator (if enabled)
        self.rls: Optional[RLSEstimator] = None
        if config.tuning_method == AdaptiveTuningMethod.RLS:
            self.rls = RLSEstimator(
                num_parameters=3,  # Kp, Ki, Kd
                forgetting_factor=config.rls_forgetting_factor,
                initial_covariance=config.rls_initial_covariance,
                parameter_bounds=[
                    (config.safety_constraints.kp_min, config.safety_constraints.kp_max),
                    (config.safety_constraints.ki_min, config.safety_constraints.ki_max),
                    (config.safety_constraints.kd_min, config.safety_constraints.kd_max)
                ]
            )

        # Performance metrics tracking
        self.metrics = PerformanceMetrics()
        self.metrics_window: deque = deque(maxlen=1000)
        self.setpoint_change_time: Optional[float] = None
        self.setpoint_previous: Optional[float] = None
        self.peak_value: Optional[float] = None

        # Error and output history
        self.error_history: deque = deque(maxlen=1000)
        self.output_history: deque = deque(maxlen=1000)

        # Stability detection
        self.stability_warning = False
        self.fallback_active = False
        self.oscillation_count = 0

        # Tuning assistant
        self.tuning_assistant = TuningAssistant()

        self.logger.info(
            f"AdaptivePIDController initialized: Kp={self.kp}, Ki={self.ki}, Kd={self.kd}, "
            f"method={config.tuning_method.value}"
        )

    def calculate(self, pid_input: AdaptivePIDInput) -> AdaptivePIDOutput:
        """
        Calculate adaptive PID control output.

        This is the main method that processes input and returns control output
        with adaptive tuning applied based on configuration.

        Args:
            pid_input: Control input parameters

        Returns:
            AdaptivePIDOutput with control signal and diagnostics
        """
        # Handle adaptive tuning commands
        if pid_input.start_relay_test:
            self._start_relay_test(pid_input.timestamp)
        elif pid_input.abort_relay_test:
            self._abort_relay_test()
        elif pid_input.approve_tuning:
            self._approve_tuning()
        elif pid_input.reject_tuning:
            self._reject_tuning()

        # Override output limits if provided
        output_min = pid_input.output_min if pid_input.output_min is not None else self.output_min
        output_max = pid_input.output_max if pid_input.output_max is not None else self.output_max

        # Apply gain scheduling if enabled
        gain_scheduled = False
        active_region = None

        if self.tuning_method == AdaptiveTuningMethod.GAIN_SCHEDULING and self.gain_schedule:
            scheduled = self._apply_gain_schedule(pid_input.operating_point)
            if scheduled:
                gain_scheduled = True
                active_region = scheduled

        # Calculate control output based on state
        if self.relay_active and self.tuning_state == TuningState.RELAY_TEST:
            # Relay feedback test mode
            control_output = self._relay_feedback_step(
                pid_input.setpoint,
                pid_input.process_variable,
                pid_input.timestamp,
                output_min,
                output_max
            )

            # Use PID for tracking during relay
            error = pid_input.setpoint - pid_input.process_variable
            p_term = self.kp * error
            i_term = self.ki * self.error_integral
            d_term = self.kd * self.derivative_filtered
        else:
            # Normal PID control (with possible MRAC adaptation)
            control_output, error, p_term, i_term, d_term = self._calculate_pid(
                pid_input.setpoint,
                pid_input.process_variable,
                pid_input.timestamp,
                output_min,
                output_max
            )

            # Apply MRAC adaptation if enabled
            if (self.tuning_method == AdaptiveTuningMethod.MRAC and
                self.config.enable_online_learning and
                self.mrac is not None):
                self._apply_mrac_adaptation(
                    pid_input.setpoint,
                    pid_input.process_variable,
                    error,
                    pid_input.timestamp
                )

            # Apply RLS adaptation if enabled
            if (self.tuning_method == AdaptiveTuningMethod.RLS and
                self.config.enable_online_learning and
                self.rls is not None):
                self._apply_rls_adaptation(
                    pid_input.setpoint,
                    pid_input.process_variable,
                    control_output,
                    pid_input.timestamp
                )

        # Update performance metrics
        self._update_performance_metrics(
            pid_input.setpoint,
            pid_input.process_variable,
            control_output,
            pid_input.timestamp
        )

        # Check stability
        self._check_stability()

        # Generate tuning recommendations
        tuning_recommended, tuning_recommendation = self._check_tuning_recommendation()

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance(
            pid_input, control_output, p_term, i_term, d_term
        )

        # Prepare performance metrics dict
        perf_dict = {
            'ise': self.metrics.ise,
            'iae': self.metrics.iae,
            'overshoot_percent': self.metrics.overshoot_percent,
            'settling_time_sec': self.metrics.settling_time_sec,
            'rise_time_sec': self.metrics.rise_time_sec,
            'steady_state_error': self.metrics.steady_state_error,
            'control_effort': self.metrics.control_effort
        }

        return AdaptivePIDOutput(
            control_output=self._round_decimal(control_output, 4),
            error=self._round_decimal(error, 4),
            error_integral=self._round_decimal(self.error_integral, 4),
            error_derivative=self._round_decimal(self.derivative_filtered, 4),
            p_term=self._round_decimal(p_term, 4),
            i_term=self._round_decimal(i_term, 4),
            d_term=self._round_decimal(d_term, 4),
            current_kp=self._round_decimal(self.kp, 4),
            current_ki=self._round_decimal(self.ki, 4),
            current_kd=self._round_decimal(self.kd, 4),
            tuning_state=self.tuning_state,
            tuning_method=self.tuning_method,
            gain_scheduled=gain_scheduled,
            active_schedule_region=active_region,
            performance_metrics=perf_dict,
            output_saturated=(control_output == output_min or control_output == output_max),
            anti_windup_active=self.windup_active,
            stability_warning=self.stability_warning,
            fallback_active=self.fallback_active,
            tuning_recommended=tuning_recommended,
            tuning_recommendation=tuning_recommendation,
            requires_operator_approval=(self.tuning_state == TuningState.WAITING_APPROVAL),
            provenance_hash=provenance_hash
        )

    def _calculate_pid(
        self,
        setpoint: float,
        process_variable: float,
        timestamp: float,
        output_min: float,
        output_max: float
    ) -> Tuple[float, float, float, float, float]:
        """
        Calculate standard PID output with anti-windup.

        Returns:
            Tuple of (control_output, error, p_term, i_term, d_term)
        """
        # Calculate time step
        if self.time_previous is None:
            dt = 0.1  # Default 100ms
            self.time_previous = timestamp
        else:
            dt = timestamp - self.time_previous
            if dt <= 0:
                dt = 0.1

        # Calculate error
        error = setpoint - process_variable

        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup
        if not self.windup_active:
            self.error_integral += error * dt

        i_term = self.ki * self.error_integral

        # Derivative term with filtering
        if dt > 0:
            derivative_raw = (error - self.error_previous) / dt
        else:
            derivative_raw = 0.0

        # Low-pass filter on derivative
        alpha = 0.1  # Filter coefficient
        self.derivative_filtered = alpha * self.derivative_filtered + (1 - alpha) * derivative_raw

        d_term = self.kd * self.derivative_filtered

        # Total output
        output_raw = p_term + i_term + d_term

        # Apply limits
        output_limited = self._clamp(output_raw, output_min, output_max)

        # Anti-windup check
        output_saturated = (output_limited != output_raw)

        if output_saturated:
            self.windup_active = True
            # Back-calculation anti-windup
            if self.ki > 0:
                max_integral = (output_max - p_term - d_term) / self.ki
                min_integral = (output_min - p_term - d_term) / self.ki
                self.error_integral = self._clamp(self.error_integral, min_integral, max_integral)
        else:
            self.windup_active = False

        # Update state
        self.error_previous = error
        self.output_previous = output_limited
        self.time_previous = timestamp

        # Store history
        self.error_history.append(error)
        self.output_history.append(output_limited)

        return output_limited, error, p_term, i_term, d_term

    def _start_relay_test(self, timestamp: float) -> None:
        """Start relay feedback auto-tuning test"""
        if self.config.safety_constraints.safety_level in [
            SafetyLevel.HIGH_CRITICAL, SafetyLevel.SAFETY_CRITICAL
        ]:
            self.logger.warning(
                "Cannot start relay test on safety-critical loop"
            )
            return

        self.relay_active = True
        self.relay_data = []
        self.relay_peaks = []
        self.relay_start_time = timestamp
        self.relay_last_switch_time = timestamp
        self.relay_switch_count = 0
        self.relay_current_direction = 1
        self.tuning_state = TuningState.RELAY_TEST
        self.relay_result = None

        self.logger.info("Relay feedback test started")

    def _abort_relay_test(self) -> None:
        """Abort current relay feedback test"""
        self.relay_active = False
        self.tuning_state = TuningState.IDLE
        self.logger.info("Relay feedback test aborted")

    def _relay_feedback_step(
        self,
        setpoint: float,
        process_variable: float,
        timestamp: float,
        output_min: float,
        output_max: float
    ) -> float:
        """
        Execute one step of relay feedback test.

        Implements Astrom-Hagglund relay feedback method:
        - Apply relay output (+d or -d) based on error sign
        - Detect sustained oscillation
        - Extract Ku and Tu from oscillation characteristics
        """
        error = setpoint - process_variable

        # Store data point
        self.relay_data.append((timestamp, process_variable))

        # Check relay switching with hysteresis
        hysteresis = self.config.relay_hysteresis

        should_switch = False
        if self.relay_current_direction > 0 and error < -hysteresis:
            should_switch = True
        elif self.relay_current_direction < 0 and error > hysteresis:
            should_switch = True

        if should_switch:
            # Record peak at switch point
            if len(self.relay_data) > 1:
                # Find local peak in recent data
                peak_value = process_variable
                peak_time = timestamp

                # Look back for actual peak
                window_size = min(20, len(self.relay_data))
                recent_data = list(self.relay_data)[-window_size:]

                if self.relay_current_direction > 0:
                    # Was going up, find max
                    peak_value = max(pv for _, pv in recent_data)
                else:
                    # Was going down, find min
                    peak_value = min(pv for _, pv in recent_data)

                self.relay_peaks.append((timestamp, peak_value))

            self.relay_current_direction *= -1
            self.relay_switch_count += 1
            self.relay_last_switch_time = timestamp

        # Calculate relay output
        d = self.config.relay_amplitude
        center_output = (output_min + output_max) / 2

        if self.relay_current_direction > 0:
            relay_output = center_output + d
        else:
            relay_output = center_output - d

        relay_output = self._clamp(relay_output, output_min, output_max)

        # Check if test complete
        test_duration = timestamp - self.relay_start_time

        if (self.relay_switch_count >= self.config.relay_min_cycles * 2 and
            len(self.relay_peaks) >= self.config.relay_min_cycles * 2):
            # Analyze results
            self._analyze_relay_results(timestamp)
        elif test_duration > self.config.relay_max_duration_sec:
            # Test timeout
            self.relay_active = False
            self.tuning_state = TuningState.FAILED
            self.relay_result = RelayFeedbackResult(
                ultimate_gain_ku=0.0,
                ultimate_period_tu=0.0,
                oscillation_amplitude=0.0,
                oscillation_count=self.relay_switch_count // 2,
                test_duration_sec=test_duration,
                success=False,
                error_message="Test timeout - insufficient oscillation"
            )
            self.logger.warning("Relay test timeout")

        return relay_output

    def _analyze_relay_results(self, timestamp: float) -> None:
        """Analyze relay feedback test results and calculate tuning parameters"""
        test_duration = timestamp - self.relay_start_time

        if len(self.relay_peaks) < 4:
            self.relay_active = False
            self.tuning_state = TuningState.FAILED
            self.relay_result = RelayFeedbackResult(
                ultimate_gain_ku=0.0,
                ultimate_period_tu=0.0,
                oscillation_amplitude=0.0,
                oscillation_count=len(self.relay_peaks) // 2,
                test_duration_sec=test_duration,
                success=False,
                error_message="Insufficient peaks detected"
            )
            return

        # Calculate oscillation amplitude (peak-to-peak / 2)
        peaks = [v for _, v in self.relay_peaks]
        amplitude = (max(peaks) - min(peaks)) / 2

        # Calculate oscillation period from peak times
        peak_times = [t for t, _ in self.relay_peaks]
        periods = []
        for i in range(2, len(peak_times)):
            # Period is time between alternate peaks (max to max or min to min)
            period = peak_times[i] - peak_times[i-2]
            periods.append(period)

        if not periods:
            self.relay_active = False
            self.tuning_state = TuningState.FAILED
            return

        Tu = statistics.mean(periods)  # Ultimate period

        # Calculate ultimate gain using Astrom-Hagglund formula
        # Ku = 4 * d / (pi * a)
        d = self.config.relay_amplitude
        if amplitude > 0:
            Ku = (4 * d) / (math.pi * amplitude)
        else:
            Ku = 1.0

        # Calculate recommended gains using different methods
        # Ziegler-Nichols
        kp_zn = 0.6 * Ku
        ki_zn = 1.2 * Ku / Tu if Tu > 0 else 0.0
        kd_zn = 0.075 * Ku * Tu

        # Tyreus-Luyben (more conservative)
        kp_tl = 0.45 * Ku
        ki_tl = 0.54 * Ku / Tu if Tu > 0 else 0.0
        kd_tl = 0.15 * Ku * Tu

        self.relay_result = RelayFeedbackResult(
            ultimate_gain_ku=round(Ku, 4),
            ultimate_period_tu=round(Tu, 4),
            oscillation_amplitude=round(amplitude, 4),
            oscillation_count=len(self.relay_peaks) // 2,
            test_duration_sec=round(test_duration, 2),
            success=True,
            kp_zn=round(kp_zn, 4),
            ki_zn=round(ki_zn, 4),
            kd_zn=round(kd_zn, 4),
            kp_tl=round(kp_tl, 4),
            ki_tl=round(ki_tl, 4),
            kd_tl=round(kd_tl, 4)
        )

        # Set pending gains (using Tyreus-Luyben by default, safer)
        self.pending_gains = {
            'kp': kp_tl,
            'ki': ki_tl,
            'kd': kd_tl
        }

        # Check if approval is required
        needs_approval = self._check_requires_approval(self.pending_gains)

        self.relay_active = False

        if needs_approval:
            self.tuning_state = TuningState.WAITING_APPROVAL
            self.logger.info(
                f"Relay test complete. Ku={Ku:.4f}, Tu={Tu:.4f}. "
                f"Proposed gains: Kp={kp_tl:.4f}, Ki={ki_tl:.4f}, Kd={kd_tl:.4f}. "
                f"Awaiting operator approval."
            )
        else:
            # Auto-apply gains
            self._apply_pending_gains()

    def _check_requires_approval(self, new_gains: Dict[str, float]) -> bool:
        """Check if proposed gains require operator approval"""
        constraints = self.config.safety_constraints

        # Safety-critical loops always require approval
        if constraints.safety_level in [SafetyLevel.HIGH_CRITICAL, SafetyLevel.SAFETY_CRITICAL]:
            return True

        # Check percentage change thresholds
        current = {'kp': self.kp, 'ki': self.ki, 'kd': self.kd}
        thresholds = {
            'kp': constraints.require_approval_kp_change_percent,
            'ki': constraints.require_approval_ki_change_percent,
            'kd': constraints.require_approval_kd_change_percent
        }

        for gain_name, threshold in thresholds.items():
            current_val = current[gain_name]
            new_val = new_gains.get(gain_name, current_val)

            if current_val != 0:
                change_percent = abs(new_val - current_val) / current_val * 100
            else:
                change_percent = 100.0 if new_val != 0 else 0.0

            if change_percent > threshold:
                return True

        return False

    def _approve_tuning(self) -> None:
        """Approve pending tuning changes"""
        if self.tuning_state == TuningState.WAITING_APPROVAL and self.pending_gains:
            self._apply_pending_gains()
            self.logger.info("Tuning approved and applied")

    def _reject_tuning(self) -> None:
        """Reject pending tuning changes"""
        if self.tuning_state == TuningState.WAITING_APPROVAL:
            self.pending_gains = None
            self.tuning_state = TuningState.IDLE
            self.logger.info("Tuning rejected")

    def _apply_pending_gains(self) -> None:
        """Apply pending gains with rate limiting"""
        if not self.pending_gains:
            return

        constraints = self.config.safety_constraints

        # Apply with rate limiting
        new_kp = self._rate_limit_gain(
            self.kp,
            self.pending_gains.get('kp', self.kp),
            constraints.max_kp_change_rate
        )
        new_ki = self._rate_limit_gain(
            self.ki,
            self.pending_gains.get('ki', self.ki),
            constraints.max_ki_change_rate
        )
        new_kd = self._rate_limit_gain(
            self.kd,
            self.pending_gains.get('kd', self.kd),
            constraints.max_kd_change_rate
        )

        # Apply bounds
        new_kp = self._clamp(new_kp, constraints.kp_min, constraints.kp_max)
        new_ki = self._clamp(new_ki, constraints.ki_min, constraints.ki_max)
        new_kd = self._clamp(new_kd, constraints.kd_min, constraints.kd_max)

        self.kp = new_kp
        self.ki = new_ki
        self.kd = new_kd

        self.pending_gains = None
        self.tuning_state = TuningState.IDLE

        self.logger.info(f"Gains applied: Kp={self.kp}, Ki={self.ki}, Kd={self.kd}")

    def _rate_limit_gain(
        self,
        current: float,
        target: float,
        max_rate: float
    ) -> float:
        """Apply rate limiting to gain change"""
        if current == 0:
            return target  # Can't rate limit from zero

        max_change = current * max_rate
        change = target - current

        if abs(change) > max_change:
            return current + max_change * (1 if change > 0 else -1)

        return target

    def _apply_gain_schedule(self, operating_point: float) -> Optional[str]:
        """Apply gain scheduling based on operating point"""
        for entry in self.gain_schedule:
            if entry.contains(operating_point):
                self.kp = entry.kp
                self.ki = entry.ki
                self.kd = entry.kd
                return entry.description or f"{entry.operating_point_min}-{entry.operating_point_max}%"

        return None

    def _apply_mrac_adaptation(
        self,
        setpoint: float,
        process_variable: float,
        error: float,
        timestamp: float
    ) -> None:
        """Apply MRAC adaptation to gains"""
        if self.mrac is None or self.time_previous is None:
            return

        dt = timestamp - self.time_previous
        if dt <= 0:
            return

        # Update reference model
        y_m = self.mrac.update_reference_model(setpoint, dt)

        # Tracking error (actual - reference model)
        tracking_error = process_variable - y_m

        # Adapt gains
        new_kp, new_ki, new_kd = self.mrac.adapt_gains(
            self.kp, self.ki, self.kd,
            error, tracking_error, dt
        )

        # Apply with rate limiting
        constraints = self.config.safety_constraints

        self.kp = self._rate_limit_gain(self.kp, new_kp, constraints.max_kp_change_rate * 0.1)
        self.ki = self._rate_limit_gain(self.ki, new_ki, constraints.max_ki_change_rate * 0.1)
        self.kd = self._rate_limit_gain(self.kd, new_kd, constraints.max_kd_change_rate * 0.1)

    def _apply_rls_adaptation(
        self,
        setpoint: float,
        process_variable: float,
        control_output: float,
        timestamp: float
    ) -> None:
        """Apply RLS parameter estimation for adaptation"""
        if self.rls is None or self.time_previous is None:
            return

        dt = timestamp - self.time_previous
        if dt <= 0:
            return

        # RLS estimates process model parameters, not PID gains directly
        # We use a simplified approach where phi contains error, integral, derivative
        error = setpoint - process_variable

        phi = [
            error,
            self.error_integral,
            self.derivative_filtered
        ]

        # Estimate parameters (Kp, Ki, Kd that would produce current output)
        _, residual = self.rls.update(control_output, phi)

        # Get updated parameter estimates
        theta = self.rls.get_parameters()

        # Only update if residual is reasonable (not unstable)
        if abs(residual) < self.config.safety_constraints.max_oscillation_amplitude_percent:
            constraints = self.config.safety_constraints

            new_kp = self._clamp(theta[0], constraints.kp_min, constraints.kp_max)
            new_ki = self._clamp(theta[1], constraints.ki_min, constraints.ki_max)
            new_kd = self._clamp(theta[2], constraints.kd_min, constraints.kd_max)

            # Apply with very slow rate limiting
            self.kp = self._rate_limit_gain(self.kp, new_kp, constraints.max_kp_change_rate * 0.01)
            self.ki = self._rate_limit_gain(self.ki, new_ki, constraints.max_ki_change_rate * 0.01)
            self.kd = self._rate_limit_gain(self.kd, new_kd, constraints.max_kd_change_rate * 0.01)

    def _update_performance_metrics(
        self,
        setpoint: float,
        process_variable: float,
        control_output: float,
        timestamp: float
    ) -> None:
        """Update real-time performance metrics"""
        # Initialize measurement start time on first call
        if self.metrics.measurement_start_time == 0.0:
            self.metrics.measurement_start_time = timestamp

        # Update sample count
        self.metrics.sample_count += 1

        # Calculate dt from previous call
        if len(self.metrics_window) > 0:
            prev_timestamp = self.metrics_window[-1].get('timestamp', timestamp)
            dt = timestamp - prev_timestamp
        else:
            dt = 0.1  # Default dt for first sample

        if dt <= 0:
            dt = 0.1

        error = setpoint - process_variable

        # Check for setpoint change or first setpoint establishment
        if self.setpoint_previous is None:
            # First call - establish setpoint and setpoint_change_time
            self.setpoint_change_time = timestamp
            self.peak_value = None
        elif setpoint != self.setpoint_previous:
            # Setpoint changed - reset step response metrics
            self.setpoint_change_time = timestamp
            self.peak_value = None
            self.metrics.overshoot_percent = 0.0
            self.metrics.rise_time_sec = 0.0
            self.metrics.settling_time_sec = 0.0

        self.setpoint_previous = setpoint

        # Update error integrals
        self.metrics.ise += error * error * dt
        self.metrics.iae += abs(error) * dt
        elapsed = timestamp - self.metrics.measurement_start_time
        self.metrics.itae += elapsed * abs(error) * dt

        # Update control effort
        self.metrics.control_effort += control_output * control_output * dt

        # Track overshoot
        if self.setpoint_change_time is not None:
            step_elapsed = timestamp - self.setpoint_change_time

            # Calculate overshoot if past initial response
            if step_elapsed > 1.0:  # Wait 1 second minimum
                if error < 0:  # Overshoot (PV > SP)
                    overshoot = -error
                    if self.peak_value is None or overshoot > self.peak_value:
                        self.peak_value = overshoot
                        self.metrics.overshoot_percent = (overshoot / setpoint * 100) if setpoint != 0 else 0
                        self.metrics.peak_time_sec = step_elapsed

        # Update variance estimates
        self.metrics_window.append({
            'error': error,
            'output': control_output,
            'timestamp': timestamp
        })

        if len(self.metrics_window) > 10:
            errors = [m['error'] for m in self.metrics_window]
            outputs = [m['output'] for m in self.metrics_window]
            self.metrics.error_variance = statistics.variance(errors) if len(errors) > 1 else 0
            self.metrics.output_variance = statistics.variance(outputs) if len(outputs) > 1 else 0

        # Update steady-state error (use recent average)
        if len(self.metrics_window) >= 10:
            recent_errors = [m['error'] for m in list(self.metrics_window)[-10:]]
            self.metrics.steady_state_error = statistics.mean(recent_errors)

        # Update tracking
        self.metrics.measurement_duration_sec = elapsed

    def _check_stability(self) -> None:
        """Check for stability issues and activate fallback if needed"""
        constraints = self.config.safety_constraints

        # Check output variance for oscillation detection
        # Use a lower threshold for more sensitive detection
        oscillation_threshold = constraints.max_oscillation_amplitude_percent ** 2
        if self.metrics.output_variance > oscillation_threshold:
            self.oscillation_count += 1
        else:
            self.oscillation_count = max(0, self.oscillation_count - 1)

        # Check error variance as well for oscillation
        if self.metrics.error_variance > constraints.max_oscillation_amplitude_percent ** 2:
            self.oscillation_count += 1

        # Check overshoot
        excessive_overshoot = self.metrics.overshoot_percent > constraints.max_overshoot_percent

        # Stability warning if persistent oscillation or excessive overshoot
        self.stability_warning = (self.oscillation_count > 5 or excessive_overshoot)

        # Activate fallback if stability issues persist
        if self.stability_warning and self.oscillation_count > 10:
            if not self.fallback_active:
                self.logger.warning(
                    "Stability issues detected - activating fallback gains"
                )
                self.kp = self.fallback_gains.get('kp', self.config.kp)
                self.ki = self.fallback_gains.get('ki', self.config.ki)
                self.kd = self.fallback_gains.get('kd', self.config.kd)
                self.fallback_active = True

    def _check_tuning_recommendation(self) -> Tuple[bool, Optional[str]]:
        """Check if tuning is recommended based on performance"""
        # Don't recommend during active tuning
        if self.tuning_state != TuningState.IDLE:
            return False, None

        # Check performance thresholds
        if self.metrics.overshoot_percent > 25:
            return True, "High overshoot detected - consider reducing Kp"

        if self.metrics.settling_time_sec > 120:
            return True, "Slow response - consider increasing gains"

        if abs(self.metrics.steady_state_error) > 5:
            return True, "Large steady-state error - consider increasing Ki"

        if self.stability_warning:
            return True, "Stability issues detected - run auto-tuning"

        return False, None

    def _calculate_provenance(
        self,
        pid_input: AdaptivePIDInput,
        control_output: float,
        p_term: float,
        i_term: float,
        d_term: float
    ) -> str:
        """Calculate SHA-256 provenance hash"""
        provenance_data = {
            'setpoint': pid_input.setpoint,
            'process_variable': pid_input.process_variable,
            'timestamp': pid_input.timestamp,
            'kp': self.kp,
            'ki': self.ki,
            'kd': self.kd,
            'control_output': control_output,
            'p_term': p_term,
            'i_term': i_term,
            'd_term': d_term,
            'tuning_state': self.tuning_state.value,
            'tuning_method': self.tuning_method.value
        }

        provenance_str = str(provenance_data)
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def _clamp(self, value: float, min_val: float, max_val: float) -> float:
        """Clamp value between min and max"""
        return max(min_val, min(max_val, value))

    def _round_decimal(self, value: float, places: int) -> float:
        """Round to specified decimal places"""
        decimal_value = Decimal(str(value))
        quantize_string = '0.' + '0' * places if places > 0 else '1'
        rounded = decimal_value.quantize(Decimal(quantize_string), rounding=ROUND_HALF_UP)
        return float(rounded)

    # =========================================================================
    # PUBLIC API METHODS
    # =========================================================================

    def get_current_gains(self) -> Dict[str, float]:
        """Get current PID gains"""
        return {
            'kp': self.kp,
            'ki': self.ki,
            'kd': self.kd
        }

    def set_gains(self, kp: float, ki: float, kd: float) -> None:
        """
        Manually set PID gains.

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
        """
        constraints = self.config.safety_constraints

        self.kp = self._clamp(kp, constraints.kp_min, constraints.kp_max)
        self.ki = self._clamp(ki, constraints.ki_min, constraints.ki_max)
        self.kd = self._clamp(kd, constraints.kd_min, constraints.kd_max)

        self.logger.info(f"Gains manually set: Kp={self.kp}, Ki={self.ki}, Kd={self.kd}")

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        return copy.copy(self.metrics)

    def get_relay_result(self) -> Optional[RelayFeedbackResult]:
        """Get relay feedback test result"""
        return self.relay_result

    def get_tuning_assistant(self) -> TuningAssistant:
        """Get tuning assistant instance"""
        return self.tuning_assistant

    def generate_tuning_report(self) -> TuningReport:
        """Generate comprehensive tuning report"""
        return self.tuning_assistant.generate_tuning_report(
            current_gains=self.get_current_gains(),
            recommended_gains=self.pending_gains,
            metrics=self.metrics,
            process_info={
                'ultimate_gain_ku': self.relay_result.ultimate_gain_ku if self.relay_result else 0,
                'ultimate_period_tu': self.relay_result.ultimate_period_tu if self.relay_result else 0
            },
            tuning_method=self.tuning_method,
            safety_constraints=self.config.safety_constraints
        )

    def reset_integral(self) -> None:
        """Reset integral term to zero"""
        self.error_integral = 0.0
        self.logger.info("Integral term reset")

    def reset_metrics(self) -> None:
        """Reset performance metrics"""
        self.metrics = PerformanceMetrics()
        self.metrics_window.clear()
        self.logger.info("Performance metrics reset")

    def enable_fallback(self) -> None:
        """Manually enable fallback gains"""
        self.kp = self.fallback_gains.get('kp', self.config.kp)
        self.ki = self.fallback_gains.get('ki', self.config.ki)
        self.kd = self.fallback_gains.get('kd', self.config.kd)
        self.fallback_active = True
        self.logger.info("Fallback gains enabled")

    def disable_fallback(self) -> None:
        """Disable fallback mode (return to adapted gains)"""
        self.fallback_active = False
        self.oscillation_count = 0
        self.stability_warning = False
        self.logger.info("Fallback mode disabled")
