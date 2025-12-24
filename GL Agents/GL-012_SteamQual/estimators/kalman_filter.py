# -*- coding: utf-8 -*-
"""
Extended Kalman Filter for Steam Quality State Estimation

This module implements an Extended Kalman Filter (EKF) for estimating
steam quality states with physics-based process models and uncertainty
propagation.

State Vector:
- x[0]: Dryness fraction (0-1)
- x[1]: Rate of change of dryness (dx/dt)
- x[2]: Separator efficiency (0-1)

Zero-Hallucination Guarantee:
- All predictions use deterministic Kalman equations
- Physics constraints enforced at every update step
- Complete provenance tracking for audit trails

Author: GL-BackendDeveloper
Date: December 2024
Version: 1.0.0
"""

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS AND BOUNDS
# ============================================================================

# Physical bounds for state variables
DRYNESS_MIN = 0.0
DRYNESS_MAX = 1.0
DRYNESS_RATE_MIN = -0.1  # Max rate of dryness decrease (per second)
DRYNESS_RATE_MAX = 0.1   # Max rate of dryness increase (per second)
SEPARATOR_EFF_MIN = 0.5  # Minimum reasonable separator efficiency
SEPARATOR_EFF_MAX = 0.999  # Maximum separator efficiency

# Default process noise covariance
DEFAULT_PROCESS_NOISE_DRYNESS = 1e-4
DEFAULT_PROCESS_NOISE_RATE = 1e-5
DEFAULT_PROCESS_NOISE_EFFICIENCY = 1e-6

# Default measurement noise
DEFAULT_MEASUREMENT_NOISE_TEMP = 0.5  # Celsius
DEFAULT_MEASUREMENT_NOISE_PRESSURE = 5.0  # kPa


class FilterStatus(Enum):
    """Kalman filter operational status."""
    NOMINAL = "nominal"
    DEGRADED = "degraded"
    POOR_DATA = "poor_data"
    DIVERGING = "diverging"
    RESET_REQUIRED = "reset_required"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class EKFConfig:
    """Configuration for Extended Kalman Filter."""

    # Process noise covariance diagonal elements
    process_noise_dryness: float = DEFAULT_PROCESS_NOISE_DRYNESS
    process_noise_rate: float = DEFAULT_PROCESS_NOISE_RATE
    process_noise_efficiency: float = DEFAULT_PROCESS_NOISE_EFFICIENCY

    # Measurement noise
    measurement_noise_temperature: float = DEFAULT_MEASUREMENT_NOISE_TEMP
    measurement_noise_pressure: float = DEFAULT_MEASUREMENT_NOISE_PRESSURE

    # Filter tuning
    adaptive_gain: bool = True
    innovation_threshold: float = 3.0  # Chi-squared threshold for outlier rejection
    min_update_interval_s: float = 0.1  # Minimum time between updates
    max_update_interval_s: float = 60.0  # Maximum time gap before reset

    # Convergence monitoring
    max_covariance_trace: float = 10.0  # Reset if covariance exceeds
    divergence_detection: bool = True

    # Physics constraints
    enforce_bounds: bool = True
    bound_method: str = "projection"  # projection, clipping, or reflection

    # State estimation mode
    use_square_root_filter: bool = False  # More numerically stable

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "process_noise_dryness": self.process_noise_dryness,
            "process_noise_rate": self.process_noise_rate,
            "process_noise_efficiency": self.process_noise_efficiency,
            "measurement_noise_temperature": self.measurement_noise_temperature,
            "measurement_noise_pressure": self.measurement_noise_pressure,
            "adaptive_gain": self.adaptive_gain,
            "innovation_threshold": self.innovation_threshold,
            "enforce_bounds": self.enforce_bounds,
        }


@dataclass
class StateVector:
    """
    State vector for EKF.

    State: [dryness_fraction, dryness_rate, separator_efficiency]
    """
    dryness_fraction: float
    dryness_rate: float
    separator_efficiency: float

    def to_array(self) -> NDArray[np.float64]:
        """Convert to numpy array."""
        return np.array([
            self.dryness_fraction,
            self.dryness_rate,
            self.separator_efficiency
        ], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> "StateVector":
        """Create from numpy array."""
        return cls(
            dryness_fraction=float(arr[0]),
            dryness_rate=float(arr[1]),
            separator_efficiency=float(arr[2])
        )

    def is_valid(self) -> bool:
        """Check if state is within physical bounds."""
        return (
            DRYNESS_MIN <= self.dryness_fraction <= DRYNESS_MAX and
            DRYNESS_RATE_MIN <= self.dryness_rate <= DRYNESS_RATE_MAX and
            SEPARATOR_EFF_MIN <= self.separator_efficiency <= SEPARATOR_EFF_MAX
        )


@dataclass
class MeasurementVector:
    """
    Measurement vector for EKF.

    Measurements: [temperature_C, pressure_kPa, optional: enthalpy_inferred]
    """
    temperature_c: float
    pressure_kpa: float
    enthalpy_inferred: Optional[float] = None
    measurement_timestamp: datetime = field(default_factory=datetime.utcnow)

    # Measurement quality indicators
    temperature_quality: float = 1.0  # 0-1, 1 = perfect
    pressure_quality: float = 1.0

    def to_array(self) -> NDArray[np.float64]:
        """Convert to numpy array (excluding optional)."""
        if self.enthalpy_inferred is not None:
            return np.array([
                self.temperature_c,
                self.pressure_kpa,
                self.enthalpy_inferred
            ], dtype=np.float64)
        return np.array([self.temperature_c, self.pressure_kpa], dtype=np.float64)

    def get_quality_weights(self) -> NDArray[np.float64]:
        """Get measurement quality weights for adaptive R matrix."""
        weights = np.array([self.temperature_quality, self.pressure_quality])
        if self.enthalpy_inferred is not None:
            weights = np.append(weights, 0.8)  # Inferred has lower confidence
        return weights


@dataclass
class ProcessModel:
    """
    Physics-based process model for state prediction.

    State transition: x(k+1) = f(x(k), u(k), dt)
    """

    def predict(
        self,
        state: StateVector,
        dt: float,
        load_fraction: float = 1.0,
        control_action: Optional[float] = None
    ) -> StateVector:
        """
        Predict next state using physics model.

        Args:
            state: Current state
            dt: Time step (seconds)
            load_fraction: Current load as fraction of nominal
            control_action: Control input (optional)

        Returns:
            Predicted state
        """
        # Dryness dynamics: first-order with rate
        x_new = state.dryness_fraction + state.dryness_rate * dt

        # Rate dynamics: decay towards equilibrium with process disturbance
        # Higher loads tend to reduce dryness rate
        rate_damping = 0.1  # Rate decay constant
        rate_new = state.dryness_rate * math.exp(-rate_damping * dt)

        # Separator efficiency: slow degradation (very small drift)
        efficiency_drift = -1e-7 * dt  # Slow degradation
        eta_new = state.separator_efficiency + efficiency_drift

        return StateVector(
            dryness_fraction=x_new,
            dryness_rate=rate_new,
            separator_efficiency=eta_new
        )

    def jacobian(
        self,
        state: StateVector,
        dt: float
    ) -> NDArray[np.float64]:
        """
        Compute Jacobian of process model (F matrix).

        Returns:
            3x3 Jacobian matrix df/dx
        """
        rate_damping = 0.1

        # F = df/dx
        F = np.array([
            [1.0, dt, 0.0],  # dx/dx, dx/dx_dot, dx/deta
            [0.0, math.exp(-rate_damping * dt), 0.0],  # dx_dot dynamics
            [0.0, 0.0, 1.0]  # eta slow drift
        ], dtype=np.float64)

        return F


@dataclass
class MeasurementModel:
    """
    Measurement model relating state to observations.

    Observation: z = h(x) + v
    """

    # Saturation temperature lookup (simplified - use thermodynamics module in production)
    _pressure_to_tsat: Dict[float, float] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize saturation temperature lookup."""
        # Approximate saturation temperatures for common pressures (kPa, C)
        self._pressure_to_tsat = {
            100.0: 99.6,
            200.0: 120.2,
            300.0: 133.5,
            400.0: 143.6,
            500.0: 151.8,
            600.0: 158.8,
            700.0: 164.9,
            800.0: 170.4,
            900.0: 175.4,
            1000.0: 179.9,
            1500.0: 198.3,
            2000.0: 212.4,
        }

    def get_saturation_temp(self, pressure_kpa: float) -> float:
        """
        Get saturation temperature for given pressure.

        Uses interpolation of lookup table.
        """
        pressures = sorted(self._pressure_to_tsat.keys())

        if pressure_kpa <= pressures[0]:
            return self._pressure_to_tsat[pressures[0]]
        if pressure_kpa >= pressures[-1]:
            return self._pressure_to_tsat[pressures[-1]]

        # Linear interpolation
        for i in range(len(pressures) - 1):
            if pressures[i] <= pressure_kpa <= pressures[i + 1]:
                p1, p2 = pressures[i], pressures[i + 1]
                t1, t2 = self._pressure_to_tsat[p1], self._pressure_to_tsat[p2]
                return t1 + (t2 - t1) * (pressure_kpa - p1) / (p2 - p1)

        return self._pressure_to_tsat[pressures[-1]]

    def predict_measurement(
        self,
        state: StateVector,
        pressure_kpa: float
    ) -> NDArray[np.float64]:
        """
        Predict measurement from state.

        For wet steam: T = T_sat(P)
        For superheated: T > T_sat based on dryness approaching/exceeding 1

        Args:
            state: Current state
            pressure_kpa: Operating pressure

        Returns:
            Predicted measurement [T, P]
        """
        t_sat = self.get_saturation_temp(pressure_kpa)

        # Temperature prediction
        if state.dryness_fraction >= 0.99:
            # Near or at saturated vapor - slight superheat
            superheat = (state.dryness_fraction - 0.99) * 100.0  # Up to 1C superheat
            t_predicted = t_sat + superheat
        else:
            # Wet steam - at saturation
            t_predicted = t_sat

        return np.array([t_predicted, pressure_kpa], dtype=np.float64)

    def jacobian(
        self,
        state: StateVector,
        pressure_kpa: float
    ) -> NDArray[np.float64]:
        """
        Compute Jacobian of measurement model (H matrix).

        Returns:
            2x3 Jacobian matrix dh/dx
        """
        # dT/dx: small effect near saturation
        if state.dryness_fraction >= 0.99:
            dT_dx = 100.0  # Temperature sensitivity to dryness
        else:
            dT_dx = 0.0  # At saturation, T independent of x

        # H = dh/dx
        H = np.array([
            [dT_dx, 0.0, 0.0],  # dT/dx, dT/dx_dot, dT/deta
            [0.0, 0.0, 0.0]     # dP/dx (pressure is input, not state)
        ], dtype=np.float64)

        return H


@dataclass
class EKFState:
    """
    Complete EKF state including covariance.
    """
    state: StateVector
    covariance: NDArray[np.float64]  # 3x3 covariance matrix P
    timestamp: datetime
    filter_status: FilterStatus = FilterStatus.NOMINAL

    # Innovation statistics for monitoring
    innovation_sequence: List[float] = field(default_factory=list)
    normalized_innovation_squared: float = 0.0

    def get_uncertainty(self) -> Dict[str, float]:
        """Get 1-sigma uncertainties for each state."""
        return {
            "dryness_fraction": math.sqrt(max(0, self.covariance[0, 0])),
            "dryness_rate": math.sqrt(max(0, self.covariance[1, 1])),
            "separator_efficiency": math.sqrt(max(0, self.covariance[2, 2]))
        }

    def get_confidence_interval(
        self,
        confidence: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        """Get confidence intervals for states."""
        # Z-score for confidence level
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_scores.get(confidence, 1.96)

        uncertainties = self.get_uncertainty()

        return {
            "dryness_fraction": (
                max(DRYNESS_MIN, self.state.dryness_fraction - z * uncertainties["dryness_fraction"]),
                min(DRYNESS_MAX, self.state.dryness_fraction + z * uncertainties["dryness_fraction"])
            ),
            "dryness_rate": (
                self.state.dryness_rate - z * uncertainties["dryness_rate"],
                self.state.dryness_rate + z * uncertainties["dryness_rate"]
            ),
            "separator_efficiency": (
                max(SEPARATOR_EFF_MIN, self.state.separator_efficiency - z * uncertainties["separator_efficiency"]),
                min(SEPARATOR_EFF_MAX, self.state.separator_efficiency + z * uncertainties["separator_efficiency"])
            )
        }


@dataclass
class KalmanFilterResult:
    """Result from a Kalman filter update cycle."""

    state: StateVector
    covariance: NDArray[np.float64]
    timestamp: datetime
    filter_status: FilterStatus

    # Update information
    innovation: NDArray[np.float64]
    kalman_gain: NDArray[np.float64]
    measurement_rejected: bool = False

    # Uncertainty metrics
    uncertainty_1sigma: Dict[str, float] = field(default_factory=dict)
    confidence_interval_95: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Diagnostics
    normalized_innovation_squared: float = 0.0
    covariance_trace: float = 0.0
    condition_number: float = 0.0

    # Provenance
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "state": {
                "dryness_fraction": round(self.state.dryness_fraction, 6),
                "dryness_rate": round(self.state.dryness_rate, 8),
                "separator_efficiency": round(self.state.separator_efficiency, 6),
            },
            "uncertainty_1sigma": {
                k: round(v, 6) for k, v in self.uncertainty_1sigma.items()
            },
            "confidence_interval_95": {
                k: (round(v[0], 4), round(v[1], 4))
                for k, v in self.confidence_interval_95.items()
            },
            "filter_status": self.filter_status.value,
            "measurement_rejected": self.measurement_rejected,
            "normalized_innovation_squared": round(self.normalized_innovation_squared, 4),
            "covariance_trace": round(self.covariance_trace, 6),
            "provenance_hash": self.provenance_hash,
        }


# ============================================================================
# MAIN EKF CLASS
# ============================================================================

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for steam quality state estimation.

    Implements physics-constrained EKF with:
    - State vector: [dryness, dryness_rate, separator_efficiency]
    - Physics-based process model
    - Measurement model linking state to observables
    - Adaptive measurement noise based on data quality
    - Outlier rejection using chi-squared test
    - Bounded state updates enforcing physical constraints

    Zero-Hallucination Guarantee:
    - All calculations use deterministic Kalman filter equations
    - Physics bounds enforced at every step
    - No ML predictions - pure state estimation

    Example:
        >>> config = EKFConfig()
        >>> ekf = ExtendedKalmanFilter(config)
        >>> ekf.initialize(dryness=0.95, separator_eff=0.98)
        >>>
        >>> measurement = MeasurementVector(temperature_c=180.0, pressure_kpa=1000.0)
        >>> result = ekf.update(measurement)
        >>> print(f"Estimated dryness: {result.state.dryness_fraction:.3f}")
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[EKFConfig] = None):
        """
        Initialize Extended Kalman Filter.

        Args:
            config: Filter configuration (optional)
        """
        self.config = config or EKFConfig()

        # Initialize models
        self.process_model = ProcessModel()
        self.measurement_model = MeasurementModel()

        # State and covariance
        self._state: Optional[EKFState] = None
        self._initialized = False

        # Process noise covariance Q
        self._Q = np.diag([
            self.config.process_noise_dryness,
            self.config.process_noise_rate,
            self.config.process_noise_efficiency
        ])

        # Base measurement noise covariance R
        self._R_base = np.diag([
            self.config.measurement_noise_temperature ** 2,
            self.config.measurement_noise_pressure ** 2
        ])

        # Innovation sequence for monitoring
        self._innovation_history: List[float] = []
        self._max_history = 100

        logger.info("ExtendedKalmanFilter initialized")

    def initialize(
        self,
        dryness: float = 0.95,
        dryness_rate: float = 0.0,
        separator_eff: float = 0.98,
        initial_covariance: Optional[NDArray[np.float64]] = None,
        timestamp: Optional[datetime] = None
    ) -> EKFState:
        """
        Initialize filter with initial state estimate.

        Args:
            dryness: Initial dryness fraction estimate
            dryness_rate: Initial rate of change
            separator_eff: Initial separator efficiency
            initial_covariance: Initial P matrix (optional)
            timestamp: Initial timestamp (optional)

        Returns:
            Initial EKF state

        Raises:
            ValueError: If initial values are out of bounds
        """
        # Validate inputs
        if not (DRYNESS_MIN <= dryness <= DRYNESS_MAX):
            raise ValueError(
                f"Initial dryness {dryness} out of bounds [{DRYNESS_MIN}, {DRYNESS_MAX}]"
            )
        if not (SEPARATOR_EFF_MIN <= separator_eff <= SEPARATOR_EFF_MAX):
            raise ValueError(
                f"Initial separator efficiency {separator_eff} out of bounds "
                f"[{SEPARATOR_EFF_MIN}, {SEPARATOR_EFF_MAX}]"
            )

        # Create initial state
        state = StateVector(
            dryness_fraction=dryness,
            dryness_rate=dryness_rate,
            separator_efficiency=separator_eff
        )

        # Initial covariance - relatively large to allow convergence
        if initial_covariance is None:
            initial_covariance = np.diag([0.01, 0.001, 0.001])  # ~10% initial uncertainty

        self._state = EKFState(
            state=state,
            covariance=initial_covariance,
            timestamp=timestamp or datetime.utcnow(),
            filter_status=FilterStatus.NOMINAL
        )

        self._initialized = True
        self._innovation_history = []

        logger.info(
            f"EKF initialized: dryness={dryness:.3f}, "
            f"separator_eff={separator_eff:.3f}"
        )

        return self._state

    def predict(
        self,
        dt: float,
        load_fraction: float = 1.0
    ) -> EKFState:
        """
        Perform prediction step (time update).

        Args:
            dt: Time step in seconds
            load_fraction: Current load as fraction of nominal

        Returns:
            Predicted state (prior)

        Raises:
            RuntimeError: If filter not initialized
        """
        if not self._initialized or self._state is None:
            raise RuntimeError("Filter not initialized. Call initialize() first.")

        # Check time step validity
        if dt <= 0:
            logger.warning(f"Invalid time step dt={dt}, skipping prediction")
            return self._state

        if dt > self.config.max_update_interval_s:
            logger.warning(
                f"Time gap {dt:.1f}s exceeds maximum {self.config.max_update_interval_s}s. "
                "Consider reinitializing."
            )
            self._state.filter_status = FilterStatus.DEGRADED

        # State prediction: x_pred = f(x, dt)
        predicted_state = self.process_model.predict(
            self._state.state,
            dt,
            load_fraction
        )

        # Jacobian of process model
        F = self.process_model.jacobian(self._state.state, dt)

        # Covariance prediction: P_pred = F * P * F' + Q
        P_pred = F @ self._state.covariance @ F.T + self._Q * dt

        # Ensure symmetry
        P_pred = (P_pred + P_pred.T) / 2.0

        # Update state
        self._state = EKFState(
            state=predicted_state,
            covariance=P_pred,
            timestamp=self._state.timestamp,
            filter_status=self._state.filter_status,
            innovation_sequence=self._state.innovation_sequence.copy()
        )

        # Enforce bounds on predicted state
        if self.config.enforce_bounds:
            self._enforce_state_bounds()

        return self._state

    def update(
        self,
        measurement: MeasurementVector,
        dt: Optional[float] = None
    ) -> KalmanFilterResult:
        """
        Perform full predict-update cycle.

        Args:
            measurement: New measurement
            dt: Time step (computed from timestamps if not provided)

        Returns:
            Filter result with updated state and diagnostics

        Raises:
            RuntimeError: If filter not initialized
        """
        if not self._initialized or self._state is None:
            raise RuntimeError("Filter not initialized. Call initialize() first.")

        # Compute time step
        if dt is None:
            dt = (measurement.measurement_timestamp - self._state.timestamp).total_seconds()

        if dt < self.config.min_update_interval_s:
            logger.debug(f"Time step {dt:.3f}s below minimum, skipping")
            return self._create_result(
                measurement_rejected=True,
                innovation=np.zeros(2)
            )

        # Prediction step
        self.predict(dt)

        # Get predicted measurement
        z_pred = self.measurement_model.predict_measurement(
            self._state.state,
            measurement.pressure_kpa
        )

        # Actual measurement
        z = measurement.to_array()[:2]  # [T, P]

        # Innovation (measurement residual)
        innovation = z - z_pred

        # Measurement Jacobian
        H = self.measurement_model.jacobian(self._state.state, measurement.pressure_kpa)

        # Adaptive R based on measurement quality
        R = self._get_adaptive_R(measurement)

        # Innovation covariance: S = H * P * H' + R
        S = H @ self._state.covariance @ H.T + R

        # Ensure S is invertible
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            logger.warning("Innovation covariance singular, using pseudo-inverse")
            S_inv = np.linalg.pinv(S)

        # Normalized innovation squared (chi-squared test)
        nis = float(innovation.T @ S_inv @ innovation)

        # Outlier rejection
        measurement_rejected = False
        if self.config.divergence_detection:
            chi2_threshold = self.config.innovation_threshold ** 2 * len(innovation)
            if nis > chi2_threshold:
                logger.warning(
                    f"Measurement rejected: NIS={nis:.2f} > threshold={chi2_threshold:.2f}"
                )
                measurement_rejected = True
                self._state.filter_status = FilterStatus.POOR_DATA

        if not measurement_rejected:
            # Kalman gain: K = P * H' * S^-1
            K = self._state.covariance @ H.T @ S_inv

            # State update: x = x + K * innovation
            x_arr = self._state.state.to_array()
            x_updated = x_arr + K @ innovation

            # Covariance update: P = (I - K*H) * P
            # Joseph form for numerical stability
            I_KH = np.eye(3) - K @ H
            P_updated = I_KH @ self._state.covariance @ I_KH.T + K @ R @ K.T

            # Ensure symmetry
            P_updated = (P_updated + P_updated.T) / 2.0

            # Update state
            self._state = EKFState(
                state=StateVector.from_array(x_updated),
                covariance=P_updated,
                timestamp=measurement.measurement_timestamp,
                filter_status=FilterStatus.NOMINAL,
                normalized_innovation_squared=nis
            )

            # Enforce bounds
            if self.config.enforce_bounds:
                self._enforce_state_bounds()

            # Store Kalman gain for result
            kalman_gain = K
        else:
            # Skip update, keep prediction
            kalman_gain = np.zeros((3, 2))
            self._state = EKFState(
                state=self._state.state,
                covariance=self._state.covariance,
                timestamp=measurement.measurement_timestamp,
                filter_status=self._state.filter_status,
                normalized_innovation_squared=nis
            )

        # Update innovation history
        self._innovation_history.append(nis)
        if len(self._innovation_history) > self._max_history:
            self._innovation_history.pop(0)

        # Monitor filter health
        self._check_filter_health()

        return self._create_result(
            measurement_rejected=measurement_rejected,
            innovation=innovation,
            kalman_gain=kalman_gain,
            nis=nis
        )

    def _get_adaptive_R(self, measurement: MeasurementVector) -> NDArray[np.float64]:
        """
        Get adaptive measurement noise covariance based on data quality.

        Args:
            measurement: Measurement with quality indicators

        Returns:
            Adjusted R matrix
        """
        if not self.config.adaptive_gain:
            return self._R_base

        # Scale R inversely with quality (lower quality = higher noise)
        weights = measurement.get_quality_weights()
        quality_scale = np.clip(weights, 0.1, 1.0)  # Bound quality factors

        R = self._R_base.copy()
        R[0, 0] /= quality_scale[0] ** 2  # Temperature
        R[1, 1] /= quality_scale[1] ** 2  # Pressure

        return R

    def _enforce_state_bounds(self) -> None:
        """Enforce physical bounds on state variables."""
        if self._state is None:
            return

        state = self._state.state

        # Apply bounds using projection method
        bounded_state = StateVector(
            dryness_fraction=np.clip(state.dryness_fraction, DRYNESS_MIN, DRYNESS_MAX),
            dryness_rate=np.clip(state.dryness_rate, DRYNESS_RATE_MIN, DRYNESS_RATE_MAX),
            separator_efficiency=np.clip(state.separator_efficiency, SEPARATOR_EFF_MIN, SEPARATOR_EFF_MAX)
        )

        self._state.state = bounded_state

    def _check_filter_health(self) -> None:
        """Monitor filter convergence and health."""
        if self._state is None:
            return

        # Check covariance trace
        trace = np.trace(self._state.covariance)
        if trace > self.config.max_covariance_trace:
            logger.warning(f"Covariance trace {trace:.4f} exceeds limit. Filter may be diverging.")
            self._state.filter_status = FilterStatus.DIVERGING

        # Check innovation consistency
        if len(self._innovation_history) >= 10:
            mean_nis = np.mean(self._innovation_history[-10:])
            if mean_nis > self.config.innovation_threshold ** 2 * 2:
                logger.warning(f"Mean NIS {mean_nis:.2f} indicates model mismatch")
                self._state.filter_status = FilterStatus.DEGRADED

    def _create_result(
        self,
        measurement_rejected: bool,
        innovation: NDArray[np.float64],
        kalman_gain: Optional[NDArray[np.float64]] = None,
        nis: float = 0.0
    ) -> KalmanFilterResult:
        """Create KalmanFilterResult from current state."""
        if self._state is None:
            raise RuntimeError("Cannot create result: filter not initialized")

        if kalman_gain is None:
            kalman_gain = np.zeros((3, 2))

        # Compute diagnostics
        ekf_state = EKFState(
            state=self._state.state,
            covariance=self._state.covariance,
            timestamp=self._state.timestamp,
            filter_status=self._state.filter_status
        )

        uncertainty = ekf_state.get_uncertainty()
        confidence_95 = ekf_state.get_confidence_interval(0.95)

        # Compute condition number
        try:
            eigvals = np.linalg.eigvalsh(self._state.covariance)
            condition = float(max(eigvals) / max(min(eigvals), 1e-10))
        except Exception:
            condition = float('inf')

        # Compute provenance hash
        provenance_data = {
            "version": self.VERSION,
            "timestamp": self._state.timestamp.isoformat(),
            "dryness": round(self._state.state.dryness_fraction, 6),
            "nis": round(nis, 4),
            "config": self.config.to_dict()
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()[:16]

        return KalmanFilterResult(
            state=self._state.state,
            covariance=self._state.covariance.copy(),
            timestamp=self._state.timestamp,
            filter_status=self._state.filter_status,
            innovation=innovation,
            kalman_gain=kalman_gain,
            measurement_rejected=measurement_rejected,
            uncertainty_1sigma=uncertainty,
            confidence_interval_95=confidence_95,
            normalized_innovation_squared=nis,
            covariance_trace=float(np.trace(self._state.covariance)),
            condition_number=condition,
            provenance_hash=provenance_hash
        )

    def reset(self) -> None:
        """Reset filter to uninitialized state."""
        self._state = None
        self._initialized = False
        self._innovation_history = []
        logger.info("EKF reset")

    @property
    def is_initialized(self) -> bool:
        """Check if filter is initialized."""
        return self._initialized

    @property
    def current_state(self) -> Optional[EKFState]:
        """Get current filter state."""
        return self._state

    def get_state_covariance(self) -> Optional[NDArray[np.float64]]:
        """Get current state covariance matrix."""
        if self._state is None:
            return None
        return self._state.covariance.copy()

    def get_innovation_statistics(self) -> Dict[str, float]:
        """Get innovation sequence statistics for diagnostics."""
        if not self._innovation_history:
            return {"mean": 0.0, "std": 0.0, "count": 0}

        return {
            "mean": float(np.mean(self._innovation_history)),
            "std": float(np.std(self._innovation_history)),
            "count": len(self._innovation_history),
            "last": self._innovation_history[-1] if self._innovation_history else 0.0
        }
