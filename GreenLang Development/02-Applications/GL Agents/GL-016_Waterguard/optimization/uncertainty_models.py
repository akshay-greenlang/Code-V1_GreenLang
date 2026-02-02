"""
GL-016 Waterguard Uncertainty Models - Uncertainty Quantification

Uncertainty quantification for sensor accuracy, model assumptions,
and prediction confidence in water treatment optimization.

Key Features:
    - Sensor uncertainty characterization
    - Confidence and prediction interval calculation
    - Uncertainty propagation through calculations
    - Conservative action recommendations under high uncertainty

Reference Standards:
    - ISO/IEC Guide 98-3:2008 (Uncertainty of measurement)
    - GUM (Guide to expression of uncertainty in measurement)
    - JCGM 100:2008

Author: GreenLang Water Treatment Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class UncertaintyType(str, Enum):
    """Types of uncertainty."""
    SENSOR_ACCURACY = "sensor_accuracy"
    SENSOR_PRECISION = "sensor_precision"
    CALIBRATION_DRIFT = "calibration_drift"
    MODEL_STRUCTURAL = "model_structural"
    MODEL_PARAMETRIC = "model_parametric"
    MEASUREMENT_NOISE = "measurement_noise"
    ENVIRONMENTAL = "environmental"


class DistributionType(str, Enum):
    """Distribution types for uncertainty."""
    NORMAL = "normal"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"
    LOGNORMAL = "lognormal"
    BETA = "beta"


class ConfidenceLevel(str, Enum):
    """Standard confidence levels."""
    LEVEL_68 = "68%"  # 1 sigma
    LEVEL_90 = "90%"
    LEVEL_95 = "95%"  # 2 sigma approximately
    LEVEL_99 = "99%"


# =============================================================================
# DATA MODELS
# =============================================================================

class SensorUncertainty(BaseModel):
    """Uncertainty characteristics of a sensor."""
    sensor_id: str = Field(..., description="Sensor identifier")
    sensor_type: str = Field(..., description="Type of sensor (pH, conductivity, etc.)")
    measurement_unit: str = Field(..., description="Unit of measurement")

    # Accuracy specifications
    accuracy: float = Field(..., description="Accuracy (same units as measurement)")
    accuracy_percent: float = Field(default=0.0, description="Accuracy as % of reading")
    resolution: float = Field(default=0.1, description="Sensor resolution")

    # Precision specifications
    repeatability: float = Field(default=0.0, description="Repeatability std dev")
    reproducibility: float = Field(default=0.0, description="Reproducibility std dev")

    # Calibration
    calibration_uncertainty: float = Field(default=0.0, description="Calibration uncertainty")
    drift_rate_per_day: float = Field(default=0.0, description="Drift rate per day")
    days_since_calibration: int = Field(default=0, ge=0)

    # Environmental sensitivity
    temp_coefficient: float = Field(default=0.0, description="% change per degC")
    reference_temp_c: float = Field(default=25.0)

    @property
    def combined_standard_uncertainty(self) -> float:
        """
        Calculate combined standard uncertainty using GUM method.

        Combines accuracy, precision, calibration, and drift uncertainties
        using root-sum-square (assuming independent sources).
        """
        # Convert accuracy to standard uncertainty (assume uniform distribution)
        u_accuracy = self.accuracy / math.sqrt(3)

        # Repeatability is already a standard deviation
        u_repeat = self.repeatability

        # Calibration uncertainty (assume normal, k=2)
        u_cal = self.calibration_uncertainty / 2.0

        # Drift accumulation
        u_drift = self.drift_rate_per_day * self.days_since_calibration / math.sqrt(3)

        # Root-sum-square
        return math.sqrt(
            u_accuracy ** 2 +
            u_repeat ** 2 +
            u_cal ** 2 +
            u_drift ** 2
        )

    @property
    def expanded_uncertainty_95(self) -> float:
        """Expanded uncertainty at 95% confidence (k=2)."""
        return 2.0 * self.combined_standard_uncertainty


class ConfidenceInterval(BaseModel):
    """Confidence interval for a parameter."""
    parameter_name: str
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float = Field(default=0.95, ge=0, le=1.0)
    unit: str = ""

    @property
    def interval_width(self) -> float:
        """Width of the confidence interval."""
        return self.upper_bound - self.lower_bound

    @property
    def relative_uncertainty(self) -> float:
        """Relative uncertainty (half-width / point estimate)."""
        if self.point_estimate == 0:
            return float('inf')
        return (self.interval_width / 2) / abs(self.point_estimate)


class PredictionInterval(BaseModel):
    """Prediction interval for a future observation."""
    variable_name: str
    point_prediction: float
    lower_bound: float
    upper_bound: float
    confidence_level: float = Field(default=0.95)
    horizon_minutes: int = Field(default=30)
    model_name: str = ""
    unit: str = ""

    # Provenance
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            provenance_str = (
                f"{self.variable_name}|{self.point_prediction:.4f}|"
                f"{self.lower_bound:.4f}|{self.upper_bound:.4f}"
            )
            self.provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

    @property
    def interval_width(self) -> float:
        """Width of prediction interval."""
        return self.upper_bound - self.lower_bound

    def contains(self, value: float) -> bool:
        """Check if value falls within interval."""
        return self.lower_bound <= value <= self.upper_bound


class UQResult(BaseModel):
    """Complete uncertainty quantification result."""
    result_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Point estimates with uncertainty
    confidence_intervals: List[ConfidenceInterval] = Field(default_factory=list)
    prediction_intervals: List[PredictionInterval] = Field(default_factory=list)

    # Overall uncertainty assessment
    total_uncertainty: float = Field(default=0.0, ge=0)
    uncertainty_level: str = "low"  # low, medium, high
    is_reliable: bool = True

    # Conservative action recommendation
    recommend_conservative: bool = False
    conservative_reason: str = ""

    # Computation info
    computation_time_ms: float = 0.0
    provenance_hash: str = ""

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance and assess uncertainty level."""
        # Assess uncertainty level
        if self.total_uncertainty < 0.05:
            self.uncertainty_level = "low"
        elif self.total_uncertainty < 0.15:
            self.uncertainty_level = "medium"
        else:
            self.uncertainty_level = "high"
            self.recommend_conservative = True
            self.conservative_reason = f"High uncertainty: {self.total_uncertainty:.2%}"

        # Provenance
        if not self.provenance_hash:
            provenance_str = (
                f"{self.result_id}|{self.timestamp.isoformat()}|"
                f"{self.total_uncertainty:.4f}"
            )
            self.provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()


# =============================================================================
# UNCERTAINTY MODEL
# =============================================================================

class UncertaintyModel:
    """
    Uncertainty quantification engine - ZERO HALLUCINATION.

    Provides deterministic uncertainty quantification for:
    - Sensor measurement uncertainty
    - Confidence interval calculation
    - Prediction interval calculation
    - Uncertainty propagation through calculations

    All calculations follow GUM (Guide to Uncertainty in Measurement).

    Example:
        >>> model = UncertaintyModel()
        >>> sensor = SensorUncertainty(
        ...     sensor_id="CT-COND-01",
        ...     sensor_type="conductivity",
        ...     measurement_unit="uS/cm",
        ...     accuracy=50,
        ...     repeatability=10
        ... )
        >>> model.register_sensor(sensor)
        >>> bounds = model.calculate_confidence_bounds(
        ...     measurement=1500.0,
        ...     sensor_id="CT-COND-01"
        ... )
    """

    # Z-scores for confidence levels
    Z_SCORES = {
        0.68: 1.0,
        0.80: 1.282,
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576
    }

    def __init__(self):
        """Initialize uncertainty model."""
        self._sensors: Dict[str, SensorUncertainty] = {}
        self._model_uncertainties: Dict[str, float] = {}

    def register_sensor(self, sensor: SensorUncertainty) -> None:
        """Register a sensor for uncertainty calculations."""
        self._sensors[sensor.sensor_id] = sensor
        logger.debug("Registered sensor %s with U95=%.3f",
                    sensor.sensor_id, sensor.expanded_uncertainty_95)

    def register_model_uncertainty(
        self,
        model_name: str,
        relative_uncertainty: float
    ) -> None:
        """Register uncertainty for a calculation model."""
        self._model_uncertainties[model_name] = relative_uncertainty

    def get_sensor(self, sensor_id: str) -> Optional[SensorUncertainty]:
        """Get registered sensor by ID."""
        return self._sensors.get(sensor_id)

    def calculate_confidence_bounds(
        self,
        measurement: float,
        sensor_id: str,
        confidence_level: float = 0.95,
        additional_uncertainty: float = 0.0
    ) -> ConfidenceInterval:
        """
        Calculate confidence interval for a measurement - DETERMINISTIC.

        Args:
            measurement: Measured value
            sensor_id: ID of the sensor
            confidence_level: Desired confidence level (0.68, 0.90, 0.95, 0.99)
            additional_uncertainty: Additional uncertainty to include

        Returns:
            ConfidenceInterval with bounds
        """
        sensor = self._sensors.get(sensor_id)

        if sensor is None:
            # Use default 5% uncertainty if sensor not registered
            std_uncertainty = abs(measurement) * 0.05 / 2.0
            logger.warning("Sensor %s not registered, using default uncertainty", sensor_id)
        else:
            std_uncertainty = sensor.combined_standard_uncertainty

            # Add percentage-based accuracy if specified
            if sensor.accuracy_percent > 0:
                pct_unc = abs(measurement) * sensor.accuracy_percent / 100 / math.sqrt(3)
                std_uncertainty = math.sqrt(std_uncertainty ** 2 + pct_unc ** 2)

        # Add any additional uncertainty
        if additional_uncertainty > 0:
            std_uncertainty = math.sqrt(
                std_uncertainty ** 2 + additional_uncertainty ** 2
            )

        # Get coverage factor (z-score)
        k = self.Z_SCORES.get(confidence_level, 1.96)

        # Calculate bounds
        expanded_unc = k * std_uncertainty
        lower = measurement - expanded_unc
        upper = measurement + expanded_unc

        return ConfidenceInterval(
            parameter_name=sensor.sensor_type if sensor else "unknown",
            point_estimate=measurement,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=confidence_level,
            unit=sensor.measurement_unit if sensor else ""
        )

    def propagate_uncertainty(
        self,
        inputs: Dict[str, Tuple[float, float]],
        calculation_function: Callable,
        method: str = "linear"
    ) -> Tuple[float, float]:
        """
        Propagate uncertainty through a calculation - DETERMINISTIC.

        Uses linear error propagation (GUM method) or Monte Carlo.

        Args:
            inputs: Dict of {name: (value, std_uncertainty)}
            calculation_function: Function taking dict of values
            method: "linear" or "monte_carlo"

        Returns:
            Tuple of (result_value, result_std_uncertainty)
        """
        if method == "monte_carlo":
            return self._propagate_monte_carlo(inputs, calculation_function)
        else:
            return self._propagate_linear(inputs, calculation_function)

    def _propagate_linear(
        self,
        inputs: Dict[str, Tuple[float, float]],
        calculation_function: Callable
    ) -> Tuple[float, float]:
        """
        Linear uncertainty propagation (first-order Taylor) - DETERMINISTIC.

        u_y^2 = sum_i (df/dx_i)^2 * u_xi^2
        """
        # Calculate central value
        central_values = {k: v[0] for k, v in inputs.items()}
        result = calculation_function(central_values)

        # Calculate partial derivatives numerically
        variance_sum = 0.0

        for name, (value, uncertainty) in inputs.items():
            if uncertainty == 0:
                continue

            # Perturb this input
            delta = max(uncertainty * 0.01, abs(value) * 1e-6)

            perturbed_high = central_values.copy()
            perturbed_high[name] = value + delta
            result_high = calculation_function(perturbed_high)

            perturbed_low = central_values.copy()
            perturbed_low[name] = value - delta
            result_low = calculation_function(perturbed_low)

            # Partial derivative
            partial = (result_high - result_low) / (2 * delta)

            # Contribution to variance
            variance_sum += (partial * uncertainty) ** 2

        result_uncertainty = math.sqrt(variance_sum)
        return result, result_uncertainty

    def _propagate_monte_carlo(
        self,
        inputs: Dict[str, Tuple[float, float]],
        calculation_function: Callable,
        n_samples: int = 1000
    ) -> Tuple[float, float]:
        """
        Monte Carlo uncertainty propagation - DETERMINISTIC.

        Uses seeded RNG for reproducibility.
        """
        np.random.seed(42)  # Fixed seed for reproducibility

        results = []

        for _ in range(n_samples):
            # Sample from input distributions
            sampled_values = {}
            for name, (value, uncertainty) in inputs.items():
                sampled_values[name] = np.random.normal(value, uncertainty)

            # Calculate result
            try:
                result = calculation_function(sampled_values)
                results.append(result)
            except Exception:
                continue

        if not results:
            # Fallback to linear
            return self._propagate_linear(inputs, calculation_function)

        mean_result = np.mean(results)
        std_result = np.std(results)

        return mean_result, std_result

    def calculate_prediction_interval(
        self,
        point_prediction: float,
        model_uncertainty: float,
        measurement_uncertainties: List[float],
        horizon_minutes: int = 30,
        confidence_level: float = 0.95,
        variable_name: str = "prediction",
        model_name: str = ""
    ) -> PredictionInterval:
        """
        Calculate prediction interval for a forecast - DETERMINISTIC.

        Prediction intervals are wider than confidence intervals because
        they account for both model uncertainty and future observation variance.

        Args:
            point_prediction: Central prediction value
            model_uncertainty: Model standard error
            measurement_uncertainties: Uncertainties from input measurements
            horizon_minutes: Forecast horizon
            confidence_level: Desired confidence level
            variable_name: Name of predicted variable
            model_name: Name of prediction model

        Returns:
            PredictionInterval with bounds
        """
        # Combine uncertainties (root-sum-square)
        measurement_variance = sum(u ** 2 for u in measurement_uncertainties)

        # Scale model uncertainty with horizon (random walk assumption)
        horizon_factor = math.sqrt(horizon_minutes / 30.0)
        scaled_model_var = (model_uncertainty * horizon_factor) ** 2

        # Total prediction variance
        total_variance = measurement_variance + scaled_model_var
        total_std = math.sqrt(total_variance)

        # Coverage factor
        k = self.Z_SCORES.get(confidence_level, 1.96)

        # For prediction intervals, add factor for future observation
        # Using simplified approach: multiply by sqrt(2) for prediction vs confidence
        prediction_factor = 1.2  # Slightly larger than confidence interval

        expanded_unc = k * total_std * prediction_factor
        lower = point_prediction - expanded_unc
        upper = point_prediction + expanded_unc

        return PredictionInterval(
            variable_name=variable_name,
            point_prediction=point_prediction,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=confidence_level,
            horizon_minutes=horizon_minutes,
            model_name=model_name
        )

    def assess_overall_uncertainty(
        self,
        confidence_intervals: List[ConfidenceInterval],
        prediction_intervals: List[PredictionInterval],
        high_threshold: float = 0.15,
        medium_threshold: float = 0.05
    ) -> UQResult:
        """
        Assess overall uncertainty and recommend action - DETERMINISTIC.

        Args:
            confidence_intervals: List of confidence intervals
            prediction_intervals: List of prediction intervals
            high_threshold: Threshold for high uncertainty
            medium_threshold: Threshold for medium uncertainty

        Returns:
            UQResult with assessment and recommendations
        """
        start_time = time.time()

        # Calculate relative uncertainties
        relative_uncertainties = []

        for ci in confidence_intervals:
            rel_unc = ci.relative_uncertainty
            if rel_unc < float('inf'):
                relative_uncertainties.append(rel_unc)

        for pi in prediction_intervals:
            if pi.point_prediction != 0:
                rel_unc = pi.interval_width / (2 * abs(pi.point_prediction))
                relative_uncertainties.append(rel_unc)

        # Aggregate uncertainty (RMS of relative uncertainties)
        if relative_uncertainties:
            total_uncertainty = math.sqrt(
                sum(u ** 2 for u in relative_uncertainties) / len(relative_uncertainties)
            )
        else:
            total_uncertainty = 0.0

        # Determine level and recommendation
        if total_uncertainty >= high_threshold:
            uncertainty_level = "high"
            recommend_conservative = True
            conservative_reason = (
                f"Overall uncertainty {total_uncertainty:.1%} exceeds "
                f"high threshold {high_threshold:.1%}"
            )
            is_reliable = False
        elif total_uncertainty >= medium_threshold:
            uncertainty_level = "medium"
            recommend_conservative = False
            conservative_reason = ""
            is_reliable = True
        else:
            uncertainty_level = "low"
            recommend_conservative = False
            conservative_reason = ""
            is_reliable = True

        computation_time_ms = (time.time() - start_time) * 1000

        return UQResult(
            confidence_intervals=confidence_intervals,
            prediction_intervals=prediction_intervals,
            total_uncertainty=total_uncertainty,
            uncertainty_level=uncertainty_level,
            is_reliable=is_reliable,
            recommend_conservative=recommend_conservative,
            conservative_reason=conservative_reason,
            computation_time_ms=computation_time_ms
        )

    def recommend_conservative_action(
        self,
        uq_result: UQResult,
        normal_setpoint: float,
        conservative_setpoint: float,
        action_name: str = "setpoint"
    ) -> Tuple[float, str]:
        """
        Recommend action based on uncertainty - DETERMINISTIC.

        When uncertainty is high, recommends more conservative action.

        Args:
            uq_result: Uncertainty quantification result
            normal_setpoint: Normal recommended setpoint
            conservative_setpoint: Conservative setpoint for high uncertainty
            action_name: Name of the action for logging

        Returns:
            Tuple of (recommended_setpoint, reason)
        """
        if uq_result.recommend_conservative:
            reason = (
                f"Using conservative {action_name} due to {uq_result.uncertainty_level} "
                f"uncertainty ({uq_result.total_uncertainty:.1%})"
            )
            logger.warning(reason)
            return conservative_setpoint, reason
        else:
            reason = f"Using optimal {action_name} (uncertainty: {uq_result.total_uncertainty:.1%})"
            return normal_setpoint, reason


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_conductivity_sensor(
    sensor_id: str,
    days_since_cal: int = 0
) -> SensorUncertainty:
    """Create typical conductivity sensor uncertainty model."""
    return SensorUncertainty(
        sensor_id=sensor_id,
        sensor_type="conductivity",
        measurement_unit="uS/cm",
        accuracy=50,  # +/- 50 uS/cm
        accuracy_percent=2.0,  # +/- 2% of reading
        resolution=1.0,
        repeatability=10.0,
        calibration_uncertainty=25.0,
        drift_rate_per_day=1.0,
        days_since_calibration=days_since_cal,
        temp_coefficient=2.0,
        reference_temp_c=25.0
    )


def create_ph_sensor(
    sensor_id: str,
    days_since_cal: int = 0
) -> SensorUncertainty:
    """Create typical pH sensor uncertainty model."""
    return SensorUncertainty(
        sensor_id=sensor_id,
        sensor_type="pH",
        measurement_unit="pH",
        accuracy=0.05,  # +/- 0.05 pH
        resolution=0.01,
        repeatability=0.02,
        calibration_uncertainty=0.03,
        drift_rate_per_day=0.005,
        days_since_calibration=days_since_cal,
        temp_coefficient=0.003,  # 0.003 pH per degC
        reference_temp_c=25.0
    )


def create_temperature_sensor(
    sensor_id: str,
    days_since_cal: int = 0
) -> SensorUncertainty:
    """Create typical temperature sensor uncertainty model."""
    return SensorUncertainty(
        sensor_id=sensor_id,
        sensor_type="temperature",
        measurement_unit="degC",
        accuracy=0.5,  # +/- 0.5 degC
        resolution=0.1,
        repeatability=0.1,
        calibration_uncertainty=0.2,
        drift_rate_per_day=0.01,
        days_since_calibration=days_since_cal
    )


def create_flow_sensor(
    sensor_id: str,
    days_since_cal: int = 0
) -> SensorUncertainty:
    """Create typical flow sensor uncertainty model."""
    return SensorUncertainty(
        sensor_id=sensor_id,
        sensor_type="flow",
        measurement_unit="gpm",
        accuracy=0,  # Use percent only
        accuracy_percent=2.0,  # +/- 2% of reading
        resolution=0.1,
        repeatability=0.5,
        calibration_uncertainty=1.0,
        drift_rate_per_day=0.1,
        days_since_calibration=days_since_cal
    )


def create_standard_uncertainty_model() -> UncertaintyModel:
    """
    Create uncertainty model with standard water treatment sensors.

    Pre-registers common sensor types for cooling tower monitoring.
    """
    model = UncertaintyModel()

    # Register standard sensors
    model.register_sensor(create_conductivity_sensor("CT-COND-01"))
    model.register_sensor(create_ph_sensor("CT-PH-01"))
    model.register_sensor(create_temperature_sensor("CT-TEMP-01"))
    model.register_sensor(create_flow_sensor("CT-FLOW-BLOWDOWN"))
    model.register_sensor(create_flow_sensor("CT-FLOW-MAKEUP"))

    # Register model uncertainties
    model.register_model_uncertainty("lsi_calculation", 0.1)
    model.register_model_uncertainty("coc_prediction", 0.08)
    model.register_model_uncertainty("scaling_risk_model", 0.15)
    model.register_model_uncertainty("corrosion_risk_model", 0.15)

    logger.info("Created standard uncertainty model with %d sensors",
               len(model._sensors))

    return model
