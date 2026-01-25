"""
Steam Quality Estimation Module for GL-012_SteamQual

This module provides advanced steam quality estimation capabilities combining:
1. Physics-based estimation from P/T/flow measurements
2. State estimation using Kalman/Extended Kalman Filters (EKF)
3. Soft sensor combining physics + data-driven corrections
4. Confidence interval computation with uncertainty propagation

All calculations are DETERMINISTIC (given fixed random seeds where applicable)
with complete provenance tracking for zero-hallucination compliance.

Key Components:
- PhysicsBasedEstimator: Direct thermodynamic calculations
- KalmanQualityEstimator: Standard Kalman filter for linear systems
- ExtendedKalmanEstimator: EKF for nonlinear steam dynamics
- SoftSensorEstimator: Hybrid physics + ML corrections
- ConfidenceIntervalCalculator: Uncertainty quantification

Author: GL-BackendDeveloper
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import hashlib
import json
import math
import logging
from datetime import datetime

import numpy as np

from .steam_properties import (
    get_saturation_properties,
    get_saturation_temperature,
    compute_superheat_margin,
    determine_steam_state,
    SteamState,
)
from .iapws_wrapper import (
    kpa_to_mpa,
    compute_provenance_hash,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class EstimationMethod(Enum):
    """Quality estimation method identifier."""
    PHYSICS_ENTHALPY = "physics_enthalpy"
    PHYSICS_ENTROPY = "physics_entropy"
    PHYSICS_TEMPERATURE = "physics_temperature"
    KALMAN_FILTER = "kalman_filter"
    EXTENDED_KALMAN = "extended_kalman"
    SOFT_SENSOR = "soft_sensor"
    ENSEMBLE = "ensemble"


class ConfidenceLevel(Enum):
    """Confidence level for interval computation."""
    LEVEL_68 = 0.68    # 1 sigma
    LEVEL_90 = 0.90    # 1.645 sigma
    LEVEL_95 = 0.95    # 1.96 sigma
    LEVEL_99 = 0.99    # 2.576 sigma


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class QualityEstimate:
    """
    Result of steam quality estimation.

    Attributes:
        quality_x: Estimated steam quality (dryness fraction)
        uncertainty: Standard uncertainty in quality estimate
        confidence_interval: (lower, upper) bounds at specified confidence
        confidence_level: Confidence level for interval
        method: Estimation method used
        is_valid: Whether estimate is within physical bounds [0, 1]
        evidence: List of supporting evidence/reasoning
        provenance_hash: SHA-256 hash for audit trail
        timestamp: Estimation timestamp
    """
    quality_x: float
    uncertainty: float
    confidence_interval: Tuple[float, float]
    confidence_level: float
    method: EstimationMethod
    is_valid: bool
    evidence: List[str] = field(default_factory=list)
    provenance_hash: str = ""
    timestamp: str = ""


@dataclass
class FilterState:
    """
    State of Kalman/EKF filter.

    Attributes:
        x: State vector [quality, quality_rate]
        P: Covariance matrix
        timestamp: Last update timestamp
        innovation: Last measurement innovation
        innovation_covariance: Innovation covariance (S)
    """
    x: np.ndarray
    P: np.ndarray
    timestamp: str = ""
    innovation: float = 0.0
    innovation_covariance: float = 1.0


@dataclass
class SoftSensorOutput:
    """
    Output from soft sensor estimation.

    Attributes:
        quality_x: Fused quality estimate
        physics_contribution: Contribution from physics model
        correction_contribution: Contribution from data-driven correction
        fusion_weight: Weight given to physics model (0-1)
        model_confidence: Confidence in physics model
        data_confidence: Confidence in data-driven correction
    """
    quality_x: float
    physics_contribution: float
    correction_contribution: float
    fusion_weight: float
    model_confidence: float
    data_confidence: float


@dataclass
class MeasurementInput:
    """
    Input measurements for quality estimation.

    Attributes:
        pressure_kpa: Steam pressure in kPa
        temperature_c: Steam temperature in Celsius
        flow_rate_kg_s: Mass flow rate in kg/s (optional)
        enthalpy_kj_kg: Specific enthalpy if available (optional)
        entropy_kj_kgk: Specific entropy if available (optional)
        differential_pressure_kpa: Orifice/venturi dP (optional)
        timestamp: Measurement timestamp
    """
    pressure_kpa: float
    temperature_c: float
    flow_rate_kg_s: Optional[float] = None
    enthalpy_kj_kg: Optional[float] = None
    entropy_kj_kgk: Optional[float] = None
    differential_pressure_kpa: Optional[float] = None
    timestamp: str = ""


# =============================================================================
# PHYSICS-BASED ESTIMATOR
# =============================================================================

class PhysicsBasedEstimator:
    """
    Physics-based steam quality estimation using thermodynamic relationships.

    This estimator uses fundamental thermodynamic equations to estimate
    steam quality from available measurements. All calculations are
    DETERMINISTIC.

    Methods:
    - Enthalpy-based: x = (h - hf) / hfg
    - Entropy-based: x = (s - sf) / sfg
    - Temperature-based: Infer from T vs Tsat
    """

    def __init__(
        self,
        enthalpy_uncertainty: float = 5.0,
        entropy_uncertainty: float = 0.01,
        temperature_uncertainty: float = 1.0,
    ):
        """
        Initialize PhysicsBasedEstimator.

        Args:
            enthalpy_uncertainty: Uncertainty in enthalpy measurement [kJ/kg]
            entropy_uncertainty: Uncertainty in entropy measurement [kJ/(kg*K)]
            temperature_uncertainty: Uncertainty in temperature measurement [C]
        """
        self.enthalpy_uncertainty = enthalpy_uncertainty
        self.entropy_uncertainty = entropy_uncertainty
        self.temperature_uncertainty = temperature_uncertainty

    def estimate_from_enthalpy(
        self,
        pressure_kpa: float,
        enthalpy_kj_kg: float,
        confidence_level: float = 0.95,
    ) -> QualityEstimate:
        """
        Estimate quality from pressure and enthalpy.

        DETERMINISTIC: Same inputs always produce same output.

        Uses: x = (h - hf) / hfg

        Args:
            pressure_kpa: Pressure in kPa
            enthalpy_kj_kg: Specific enthalpy in kJ/kg
            confidence_level: Confidence level for interval

        Returns:
            QualityEstimate with computed quality and uncertainty
        """
        hf, hg, hfg, _, _, _, _, _ = get_saturation_properties(pressure_kpa)

        # Compute quality
        if abs(hfg) < 1e-10:
            raise ValueError("Near critical point - hfg approaches zero")

        x = (enthalpy_kj_kg - hf) / hfg

        # Propagate uncertainty: sigma_x = sigma_h / hfg
        sigma_x = self.enthalpy_uncertainty / abs(hfg)

        # Compute confidence interval
        z_score = self._get_z_score(confidence_level)
        lower = x - z_score * sigma_x
        upper = x + z_score * sigma_x

        # Clamp to physical bounds for display
        is_valid = 0 <= x <= 1

        evidence = [
            f"Enthalpy method: x = (h - hf) / hfg",
            f"h = {enthalpy_kj_kg:.2f} kJ/kg, hf = {hf:.2f} kJ/kg, hfg = {hfg:.2f} kJ/kg",
            f"x = ({enthalpy_kj_kg:.2f} - {hf:.2f}) / {hfg:.2f} = {x:.4f}",
        ]

        provenance = compute_provenance_hash(
            {"pressure_kpa": pressure_kpa, "enthalpy_kj_kg": enthalpy_kj_kg},
            {"quality_x": x, "uncertainty": sigma_x},
            "physics_enthalpy",
        )

        return QualityEstimate(
            quality_x=x,
            uncertainty=sigma_x,
            confidence_interval=(lower, upper),
            confidence_level=confidence_level,
            method=EstimationMethod.PHYSICS_ENTHALPY,
            is_valid=is_valid,
            evidence=evidence,
            provenance_hash=provenance,
            timestamp=datetime.utcnow().isoformat(),
        )

    def estimate_from_entropy(
        self,
        pressure_kpa: float,
        entropy_kj_kgk: float,
        confidence_level: float = 0.95,
    ) -> QualityEstimate:
        """
        Estimate quality from pressure and entropy.

        DETERMINISTIC: Same inputs always produce same output.

        Uses: x = (s - sf) / sfg

        Args:
            pressure_kpa: Pressure in kPa
            entropy_kj_kgk: Specific entropy in kJ/(kg*K)
            confidence_level: Confidence level for interval

        Returns:
            QualityEstimate with computed quality and uncertainty
        """
        _, _, _, sf, sg, sfg, _, _ = get_saturation_properties(pressure_kpa)

        if abs(sfg) < 1e-10:
            raise ValueError("Near critical point - sfg approaches zero")

        x = (entropy_kj_kgk - sf) / sfg

        # Propagate uncertainty
        sigma_x = self.entropy_uncertainty / abs(sfg)

        z_score = self._get_z_score(confidence_level)
        lower = x - z_score * sigma_x
        upper = x + z_score * sigma_x

        is_valid = 0 <= x <= 1

        evidence = [
            f"Entropy method: x = (s - sf) / sfg",
            f"s = {entropy_kj_kgk:.4f} kJ/(kg*K), sf = {sf:.4f}, sfg = {sfg:.4f}",
            f"x = ({entropy_kj_kgk:.4f} - {sf:.4f}) / {sfg:.4f} = {x:.4f}",
        ]

        provenance = compute_provenance_hash(
            {"pressure_kpa": pressure_kpa, "entropy_kj_kgk": entropy_kj_kgk},
            {"quality_x": x, "uncertainty": sigma_x},
            "physics_entropy",
        )

        return QualityEstimate(
            quality_x=x,
            uncertainty=sigma_x,
            confidence_interval=(lower, upper),
            confidence_level=confidence_level,
            method=EstimationMethod.PHYSICS_ENTROPY,
            is_valid=is_valid,
            evidence=evidence,
            provenance_hash=provenance,
            timestamp=datetime.utcnow().isoformat(),
        )

    def estimate_from_temperature(
        self,
        pressure_kpa: float,
        temperature_c: float,
        confidence_level: float = 0.95,
    ) -> QualityEstimate:
        """
        Estimate quality from temperature relative to saturation.

        DETERMINISTIC: Same inputs always produce same output.

        Temperature-based estimation has limitations:
        - T < Tsat: Subcooled liquid (x = 0)
        - T = Tsat: Cannot determine x from T alone
        - T > Tsat: Superheated vapor (x = 1)

        Args:
            pressure_kpa: Pressure in kPa
            temperature_c: Temperature in Celsius
            confidence_level: Confidence level for interval

        Returns:
            QualityEstimate with computed quality and uncertainty
        """
        superheat = compute_superheat_margin(pressure_kpa, temperature_c)
        T_sat = superheat.saturation_temperature_c

        # Temperature-based inference
        margin = superheat.margin_c
        tolerance = 2.0 * self.temperature_uncertainty

        if margin < -tolerance:
            # Definitely subcooled
            x = 0.0
            sigma_x = 0.01  # Very low uncertainty
            evidence = [
                f"Temperature {temperature_c:.2f} C is {abs(margin):.1f} C below Tsat",
                "State: Subcooled liquid (x = 0)",
            ]

        elif margin > tolerance:
            # Definitely superheated
            x = 1.0
            sigma_x = 0.01
            evidence = [
                f"Temperature {temperature_c:.2f} C is {margin:.1f} C above Tsat",
                "State: Superheated vapor (x = 1)",
            ]

        else:
            # Near saturation - high uncertainty
            # Use linear interpolation as rough estimate
            x = 0.5 + 0.5 * (margin / tolerance) if tolerance > 0 else 0.5
            x = max(0, min(1, x))
            sigma_x = 0.3  # High uncertainty
            evidence = [
                f"Temperature {temperature_c:.2f} C is within {abs(margin):.1f} C of Tsat",
                "State: Near saturation - high uncertainty in quality",
                f"Rough estimate: x = {x:.2f} +/- {sigma_x:.2f}",
            ]

        z_score = self._get_z_score(confidence_level)
        lower = max(0, x - z_score * sigma_x)
        upper = min(1, x + z_score * sigma_x)

        is_valid = 0 <= x <= 1

        provenance = compute_provenance_hash(
            {"pressure_kpa": pressure_kpa, "temperature_c": temperature_c},
            {"quality_x": x, "uncertainty": sigma_x, "margin_c": margin},
            "physics_temperature",
        )

        return QualityEstimate(
            quality_x=x,
            uncertainty=sigma_x,
            confidence_interval=(lower, upper),
            confidence_level=confidence_level,
            method=EstimationMethod.PHYSICS_TEMPERATURE,
            is_valid=is_valid,
            evidence=evidence,
            provenance_hash=provenance,
            timestamp=datetime.utcnow().isoformat(),
        )

    def _get_z_score(self, confidence_level: float) -> float:
        """Get z-score for given confidence level."""
        z_scores = {
            0.68: 1.0,
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576,
        }
        # Find closest match or interpolate
        if confidence_level in z_scores:
            return z_scores[confidence_level]
        # Use inverse normal approximation
        # For simplicity, use linear interpolation
        levels = sorted(z_scores.keys())
        for i in range(len(levels) - 1):
            if levels[i] <= confidence_level <= levels[i + 1]:
                t = (confidence_level - levels[i]) / (levels[i + 1] - levels[i])
                return z_scores[levels[i]] + t * (z_scores[levels[i + 1]] - z_scores[levels[i]])
        return 1.96  # Default to 95%


# =============================================================================
# KALMAN FILTER ESTIMATOR
# =============================================================================

class KalmanQualityEstimator:
    """
    Kalman filter for steam quality state estimation.

    Implements a standard Kalman filter for tracking steam quality
    over time. The state vector is [quality, quality_rate].

    The filter is DETERMINISTIC given fixed initial conditions
    and measurement sequence.
    """

    def __init__(
        self,
        process_noise_q: float = 0.001,
        measurement_noise_r: float = 0.01,
        initial_quality: float = 0.9,
        initial_variance: float = 0.1,
        dt: float = 1.0,
    ):
        """
        Initialize Kalman filter.

        Args:
            process_noise_q: Process noise variance
            measurement_noise_r: Measurement noise variance
            initial_quality: Initial quality estimate
            initial_variance: Initial estimate variance
            dt: Time step in seconds
        """
        self.Q = np.array([[process_noise_q, 0], [0, process_noise_q * 0.1]])
        self.R = measurement_noise_r
        self.dt = dt

        # State transition matrix [x, dx/dt] -> [x + dx/dt*dt, dx/dt]
        self.F = np.array([[1, dt], [0, 1]])

        # Observation matrix (we observe quality directly)
        self.H = np.array([[1, 0]])

        # Initialize state
        self.state = FilterState(
            x=np.array([initial_quality, 0.0]),  # [quality, rate]
            P=np.array([[initial_variance, 0], [0, initial_variance * 0.1]]),
            timestamp=datetime.utcnow().isoformat(),
        )

    def predict(self) -> FilterState:
        """
        Predict next state (time update).

        DETERMINISTIC: Same state always produces same prediction.

        Returns:
            Predicted filter state
        """
        # Predicted state
        x_pred = self.F @ self.state.x

        # Predicted covariance
        P_pred = self.F @ self.state.P @ self.F.T + self.Q

        # Clamp quality to [0, 1]
        x_pred[0] = max(0, min(1, x_pred[0]))

        self.state.x = x_pred
        self.state.P = P_pred
        self.state.timestamp = datetime.utcnow().isoformat()

        return self.state

    def update(self, measurement: float) -> FilterState:
        """
        Update state with measurement (measurement update).

        DETERMINISTIC: Same state and measurement produce same update.

        Args:
            measurement: Observed quality value

        Returns:
            Updated filter state
        """
        x_pred = self.state.x
        P_pred = self.state.P

        # Innovation (measurement residual)
        y = measurement - (self.H @ x_pred)[0]

        # Innovation covariance
        S = (self.H @ P_pred @ self.H.T)[0, 0] + self.R

        # Kalman gain
        K = (P_pred @ self.H.T) / S

        # Updated state
        x_new = x_pred + K.flatten() * y

        # Updated covariance
        I = np.eye(2)
        P_new = (I - K @ self.H) @ P_pred

        # Clamp quality to [0, 1]
        x_new[0] = max(0, min(1, x_new[0]))

        self.state = FilterState(
            x=x_new,
            P=P_new,
            timestamp=datetime.utcnow().isoformat(),
            innovation=y,
            innovation_covariance=S,
        )

        return self.state

    def estimate(
        self,
        measurement: float,
        confidence_level: float = 0.95,
    ) -> QualityEstimate:
        """
        Perform predict-update cycle and return estimate.

        DETERMINISTIC: Same sequence produces same estimates.

        Args:
            measurement: Observed quality value
            confidence_level: Confidence level for interval

        Returns:
            QualityEstimate with filtered quality
        """
        # Predict
        self.predict()

        # Update
        self.update(measurement)

        # Extract estimate
        x = self.state.x[0]
        variance = self.state.P[0, 0]
        sigma = math.sqrt(variance)

        # Confidence interval
        z = 1.96 if confidence_level == 0.95 else 1.645
        lower = max(0, x - z * sigma)
        upper = min(1, x + z * sigma)

        evidence = [
            f"Kalman filter estimate: x = {x:.4f}",
            f"State variance: {variance:.6f}",
            f"Innovation: {self.state.innovation:.4f}",
            f"Quality rate: {self.state.x[1]:.4f}/s",
        ]

        provenance = compute_provenance_hash(
            {"measurement": measurement},
            {"quality_x": x, "variance": variance},
            "kalman_filter",
        )

        return QualityEstimate(
            quality_x=x,
            uncertainty=sigma,
            confidence_interval=(lower, upper),
            confidence_level=confidence_level,
            method=EstimationMethod.KALMAN_FILTER,
            is_valid=0 <= x <= 1,
            evidence=evidence,
            provenance_hash=provenance,
            timestamp=self.state.timestamp,
        )

    def reset(self, initial_quality: float = 0.9):
        """Reset filter to initial state."""
        self.state = FilterState(
            x=np.array([initial_quality, 0.0]),
            P=np.array([[0.1, 0], [0, 0.01]]),
            timestamp=datetime.utcnow().isoformat(),
        )


# =============================================================================
# EXTENDED KALMAN FILTER ESTIMATOR
# =============================================================================

class ExtendedKalmanEstimator:
    """
    Extended Kalman Filter for nonlinear steam quality dynamics.

    The EKF handles the nonlinear relationship between steam properties
    and quality, particularly near phase boundaries.

    State vector: [quality, enthalpy, rate_of_change]
    """

    def __init__(
        self,
        process_noise: float = 0.001,
        measurement_noise: float = 0.01,
        initial_quality: float = 0.9,
        reference_pressure_kpa: float = 1000.0,
        dt: float = 1.0,
    ):
        """
        Initialize Extended Kalman Filter.

        Args:
            process_noise: Process noise variance
            measurement_noise: Measurement noise variance
            initial_quality: Initial quality estimate
            reference_pressure_kpa: Reference pressure for linearization
            dt: Time step in seconds
        """
        self.Q = np.diag([process_noise, process_noise * 100, process_noise * 0.1])
        self.R = measurement_noise
        self.dt = dt
        self.reference_pressure_kpa = reference_pressure_kpa

        # Get saturation properties for reference
        self._update_saturation_props()

        # Initial state [quality, enthalpy, rate]
        initial_h = self.hf + initial_quality * self.hfg

        self.state = FilterState(
            x=np.array([initial_quality, initial_h, 0.0]),
            P=np.diag([0.1, 100.0, 0.01]),
            timestamp=datetime.utcnow().isoformat(),
        )

    def _update_saturation_props(self):
        """Update saturation properties at reference pressure."""
        hf, hg, hfg, sf, sg, sfg, vf, vg = get_saturation_properties(
            self.reference_pressure_kpa
        )
        self.hf = hf
        self.hg = hg
        self.hfg = hfg

    def _state_transition(self, x: np.ndarray) -> np.ndarray:
        """
        Nonlinear state transition function.

        DETERMINISTIC: Same input produces same output.
        """
        quality = x[0]
        enthalpy = x[1]
        rate = x[2]

        # Simple dynamics: quality changes with rate
        new_quality = quality + rate * self.dt
        new_quality = max(0, min(1, new_quality))

        # Enthalpy consistent with quality
        new_enthalpy = self.hf + new_quality * self.hfg

        # Rate decays toward zero
        new_rate = rate * 0.95

        return np.array([new_quality, new_enthalpy, new_rate])

    def _jacobian_F(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of state transition function.

        DETERMINISTIC: Same input produces same output.
        """
        # Linearized around current state
        return np.array([
            [1, 0, self.dt],       # d(new_quality)/d(state)
            [self.hfg, 0, 0],      # d(new_enthalpy)/d(state) - simplified
            [0, 0, 0.95],          # d(new_rate)/d(state)
        ])

    def _measurement_function(self, x: np.ndarray) -> float:
        """
        Measurement function (quality from enthalpy).

        DETERMINISTIC: Same input produces same output.
        """
        quality = x[0]
        return quality

    def _jacobian_H(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of measurement function."""
        return np.array([[1, 0, 0]])

    def predict(self) -> FilterState:
        """
        EKF prediction step.

        DETERMINISTIC: Same state produces same prediction.
        """
        x_pred = self._state_transition(self.state.x)
        F = self._jacobian_F(self.state.x)
        P_pred = F @ self.state.P @ F.T + self.Q

        self.state.x = x_pred
        self.state.P = P_pred

        return self.state

    def update(self, measurement: float) -> FilterState:
        """
        EKF update step.

        DETERMINISTIC: Same state and measurement produce same update.

        Args:
            measurement: Observed quality value
        """
        x_pred = self.state.x
        P_pred = self.state.P

        # Predicted measurement
        z_pred = self._measurement_function(x_pred)

        # Innovation
        y = measurement - z_pred

        # Jacobian
        H = self._jacobian_H(x_pred)

        # Innovation covariance
        S = H @ P_pred @ H.T + self.R

        # Kalman gain
        K = P_pred @ H.T / S

        # Update
        x_new = x_pred + K.flatten() * y
        P_new = (np.eye(3) - K @ H) @ P_pred

        # Clamp quality
        x_new[0] = max(0, min(1, x_new[0]))

        self.state = FilterState(
            x=x_new,
            P=P_new,
            timestamp=datetime.utcnow().isoformat(),
            innovation=y,
            innovation_covariance=float(S),
        )

        return self.state

    def estimate(
        self,
        measurement: float,
        pressure_kpa: Optional[float] = None,
        confidence_level: float = 0.95,
    ) -> QualityEstimate:
        """
        Perform EKF predict-update and return estimate.

        Args:
            measurement: Observed quality or enthalpy
            pressure_kpa: Current pressure (updates reference if provided)
            confidence_level: Confidence level for interval

        Returns:
            QualityEstimate with EKF-filtered quality
        """
        if pressure_kpa is not None:
            self.reference_pressure_kpa = pressure_kpa
            self._update_saturation_props()

        self.predict()
        self.update(measurement)

        x = self.state.x[0]
        sigma = math.sqrt(self.state.P[0, 0])

        z = 1.96 if confidence_level == 0.95 else 1.645
        lower = max(0, x - z * sigma)
        upper = min(1, x + z * sigma)

        evidence = [
            f"EKF estimate: x = {x:.4f}",
            f"Enthalpy state: {self.state.x[1]:.2f} kJ/kg",
            f"Rate of change: {self.state.x[2]:.4f}/s",
            f"Innovation: {self.state.innovation:.4f}",
        ]

        provenance = compute_provenance_hash(
            {"measurement": measurement, "pressure_kpa": pressure_kpa},
            {"quality_x": x, "enthalpy": self.state.x[1]},
            "extended_kalman",
        )

        return QualityEstimate(
            quality_x=x,
            uncertainty=sigma,
            confidence_interval=(lower, upper),
            confidence_level=confidence_level,
            method=EstimationMethod.EXTENDED_KALMAN,
            is_valid=0 <= x <= 1,
            evidence=evidence,
            provenance_hash=provenance,
            timestamp=self.state.timestamp,
        )


# =============================================================================
# SOFT SENSOR ESTIMATOR
# =============================================================================

class SoftSensorEstimator:
    """
    Soft sensor combining physics-based model with data-driven corrections.

    The soft sensor fuses:
    1. Physics model (thermodynamic equations)
    2. Data-driven correction (bias adjustment based on historical data)

    Fusion weight adapts based on model confidence and data availability.
    All calculations are DETERMINISTIC.
    """

    def __init__(
        self,
        physics_estimator: PhysicsBasedEstimator,
        correction_buffer_size: int = 100,
        initial_physics_weight: float = 0.8,
        adaptation_rate: float = 0.01,
    ):
        """
        Initialize SoftSensorEstimator.

        Args:
            physics_estimator: Physics-based estimator instance
            correction_buffer_size: Size of correction history buffer
            initial_physics_weight: Initial weight for physics model
            adaptation_rate: Rate of weight adaptation
        """
        self.physics = physics_estimator
        self.buffer_size = correction_buffer_size
        self.physics_weight = initial_physics_weight
        self.adaptation_rate = adaptation_rate

        # Correction history
        self.corrections: List[float] = []
        self.current_bias: float = 0.0
        self.correction_variance: float = 0.01

    def update_correction(self, true_quality: float, physics_estimate: float):
        """
        Update correction model with true measurement.

        DETERMINISTIC: Given same sequence, produces same corrections.

        Args:
            true_quality: True/reference quality value
            physics_estimate: Physics model estimate
        """
        error = true_quality - physics_estimate

        self.corrections.append(error)
        if len(self.corrections) > self.buffer_size:
            self.corrections.pop(0)

        # Update bias estimate (exponential moving average)
        self.current_bias = (
            (1 - self.adaptation_rate) * self.current_bias +
            self.adaptation_rate * error
        )

        # Update correction variance
        if len(self.corrections) > 1:
            self.correction_variance = sum(
                (c - self.current_bias) ** 2 for c in self.corrections
            ) / len(self.corrections)

    def estimate(
        self,
        measurement: MeasurementInput,
        confidence_level: float = 0.95,
    ) -> QualityEstimate:
        """
        Estimate quality using soft sensor fusion.

        DETERMINISTIC: Same input produces same output.

        Args:
            measurement: Input measurements
            confidence_level: Confidence level for interval

        Returns:
            QualityEstimate with fused quality
        """
        # Get physics estimate
        if measurement.enthalpy_kj_kg is not None:
            physics_result = self.physics.estimate_from_enthalpy(
                measurement.pressure_kpa,
                measurement.enthalpy_kj_kg,
                confidence_level,
            )
        elif measurement.entropy_kj_kgk is not None:
            physics_result = self.physics.estimate_from_entropy(
                measurement.pressure_kpa,
                measurement.entropy_kj_kgk,
                confidence_level,
            )
        else:
            physics_result = self.physics.estimate_from_temperature(
                measurement.pressure_kpa,
                measurement.temperature_c,
                confidence_level,
            )

        physics_x = physics_result.quality_x
        physics_var = physics_result.uncertainty ** 2

        # Apply data-driven correction
        corrected_x = physics_x + self.current_bias * (1 - self.physics_weight)

        # Fused variance
        fused_var = (
            self.physics_weight ** 2 * physics_var +
            (1 - self.physics_weight) ** 2 * self.correction_variance
        )
        fused_sigma = math.sqrt(fused_var)

        # Clamp to valid range
        corrected_x = max(0, min(1, corrected_x))

        z = 1.96 if confidence_level == 0.95 else 1.645
        lower = max(0, corrected_x - z * fused_sigma)
        upper = min(1, corrected_x + z * fused_sigma)

        evidence = [
            f"Physics estimate: {physics_x:.4f}",
            f"Bias correction: {self.current_bias:.4f}",
            f"Physics weight: {self.physics_weight:.2f}",
            f"Fused estimate: {corrected_x:.4f}",
            *physics_result.evidence,
        ]

        provenance = compute_provenance_hash(
            {
                "pressure_kpa": measurement.pressure_kpa,
                "temperature_c": measurement.temperature_c,
                "physics_x": physics_x,
            },
            {"quality_x": corrected_x, "bias": self.current_bias},
            "soft_sensor",
        )

        return QualityEstimate(
            quality_x=corrected_x,
            uncertainty=fused_sigma,
            confidence_interval=(lower, upper),
            confidence_level=confidence_level,
            method=EstimationMethod.SOFT_SENSOR,
            is_valid=0 <= corrected_x <= 1,
            evidence=evidence,
            provenance_hash=provenance,
            timestamp=datetime.utcnow().isoformat(),
        )


# =============================================================================
# CONFIDENCE INTERVAL CALCULATOR
# =============================================================================

class ConfidenceIntervalCalculator:
    """
    Calculate confidence intervals for quality estimates.

    Supports multiple methods:
    - Gaussian: Assumes normal distribution
    - Bootstrap: Non-parametric bootstrap (requires sample data)
    - Wilson: For proportions near 0 or 1
    """

    @staticmethod
    def gaussian_interval(
        estimate: float,
        std_error: float,
        confidence_level: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Calculate Gaussian confidence interval.

        DETERMINISTIC: Same inputs produce same output.

        Args:
            estimate: Point estimate
            std_error: Standard error of estimate
            confidence_level: Confidence level

        Returns:
            (lower, upper) confidence bounds
        """
        z_scores = {0.68: 1.0, 0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_scores.get(confidence_level, 1.96)

        lower = estimate - z * std_error
        upper = estimate + z * std_error

        return (lower, upper)

    @staticmethod
    def wilson_interval(
        estimate: float,
        n_samples: int,
        confidence_level: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Calculate Wilson score interval (good for proportions near 0 or 1).

        DETERMINISTIC: Same inputs produce same output.

        Args:
            estimate: Proportion estimate (0-1)
            n_samples: Effective sample size
            confidence_level: Confidence level

        Returns:
            (lower, upper) confidence bounds
        """
        z_scores = {0.68: 1.0, 0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_scores.get(confidence_level, 1.96)

        p = estimate
        n = n_samples

        denominator = 1 + z ** 2 / n
        center = (p + z ** 2 / (2 * n)) / denominator
        spread = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denominator

        lower = max(0, center - spread)
        upper = min(1, center + spread)

        return (lower, upper)

    @staticmethod
    def monte_carlo_interval(
        samples: List[float],
        confidence_level: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Calculate interval from Monte Carlo samples.

        DETERMINISTIC: Same samples produce same output.

        Args:
            samples: List of Monte Carlo samples
            confidence_level: Confidence level

        Returns:
            (lower, upper) percentile bounds
        """
        if not samples:
            raise ValueError("Empty sample list")

        sorted_samples = sorted(samples)
        n = len(sorted_samples)

        alpha = 1 - confidence_level
        lower_idx = int(math.floor(alpha / 2 * n))
        upper_idx = int(math.ceil((1 - alpha / 2) * n)) - 1

        lower_idx = max(0, min(lower_idx, n - 1))
        upper_idx = max(0, min(upper_idx, n - 1))

        return (sorted_samples[lower_idx], sorted_samples[upper_idx])


# =============================================================================
# ENSEMBLE ESTIMATOR
# =============================================================================

def estimate_quality_ensemble(
    measurement: MeasurementInput,
    methods: Optional[List[EstimationMethod]] = None,
    confidence_level: float = 0.95,
) -> QualityEstimate:
    """
    Estimate quality using ensemble of methods.

    DETERMINISTIC: Same inputs produce same output.

    Combines multiple estimation methods using weighted averaging,
    where weights are inversely proportional to uncertainty.

    Args:
        measurement: Input measurements
        methods: List of methods to use (default: all applicable)
        confidence_level: Confidence level for interval

    Returns:
        QualityEstimate with ensemble quality
    """
    physics = PhysicsBasedEstimator()
    estimates: List[Tuple[float, float, str]] = []  # (quality, weight, method)

    # Enthalpy-based if available
    if measurement.enthalpy_kj_kg is not None:
        try:
            result = physics.estimate_from_enthalpy(
                measurement.pressure_kpa,
                measurement.enthalpy_kj_kg,
                confidence_level,
            )
            weight = 1.0 / (result.uncertainty ** 2 + 1e-10)
            estimates.append((result.quality_x, weight, "enthalpy"))
        except Exception as e:
            logger.warning(f"Enthalpy estimation failed: {e}")

    # Entropy-based if available
    if measurement.entropy_kj_kgk is not None:
        try:
            result = physics.estimate_from_entropy(
                measurement.pressure_kpa,
                measurement.entropy_kj_kgk,
                confidence_level,
            )
            weight = 1.0 / (result.uncertainty ** 2 + 1e-10)
            estimates.append((result.quality_x, weight, "entropy"))
        except Exception as e:
            logger.warning(f"Entropy estimation failed: {e}")

    # Temperature-based (always available)
    try:
        result = physics.estimate_from_temperature(
            measurement.pressure_kpa,
            measurement.temperature_c,
            confidence_level,
        )
        weight = 1.0 / (result.uncertainty ** 2 + 1e-10)
        estimates.append((result.quality_x, weight, "temperature"))
    except Exception as e:
        logger.warning(f"Temperature estimation failed: {e}")

    if not estimates:
        raise ValueError("No estimation methods succeeded")

    # Weighted average
    total_weight = sum(w for _, w, _ in estimates)
    ensemble_x = sum(x * w for x, w, _ in estimates) / total_weight

    # Ensemble uncertainty (inverse of total weight)
    ensemble_var = 1.0 / total_weight
    ensemble_sigma = math.sqrt(ensemble_var)

    # Confidence interval
    z = 1.96 if confidence_level == 0.95 else 1.645
    lower = max(0, ensemble_x - z * ensemble_sigma)
    upper = min(1, ensemble_x + z * ensemble_sigma)

    evidence = [
        f"Ensemble of {len(estimates)} methods",
        *[f"{m}: x={x:.4f}, weight={w:.2f}" for x, w, m in estimates],
        f"Weighted average: {ensemble_x:.4f}",
    ]

    provenance = compute_provenance_hash(
        {
            "pressure_kpa": measurement.pressure_kpa,
            "temperature_c": measurement.temperature_c,
            "methods": [m for _, _, m in estimates],
        },
        {"quality_x": ensemble_x, "n_methods": len(estimates)},
        "ensemble",
    )

    return QualityEstimate(
        quality_x=ensemble_x,
        uncertainty=ensemble_sigma,
        confidence_interval=(lower, upper),
        confidence_level=confidence_level,
        method=EstimationMethod.ENSEMBLE,
        is_valid=0 <= ensemble_x <= 1,
        evidence=evidence,
        provenance_hash=provenance,
        timestamp=datetime.utcnow().isoformat(),
    )
