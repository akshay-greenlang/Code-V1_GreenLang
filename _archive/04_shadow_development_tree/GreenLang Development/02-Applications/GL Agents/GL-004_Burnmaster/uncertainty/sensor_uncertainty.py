# -*- coding: utf-8 -*-
"""
GL-004 Burnmaster - Sensor Uncertainty Module

Manages uncertainty quantification for combustion measurement sensors.
Implements GUM (Guide to Expression of Uncertainty in Measurement) principles
for Type A (statistical) and Type B (systematic) uncertainty analysis.

Typical combustion sensors:
    - O2 analyzers: +/-0.1-0.5% accuracy
    - CO analyzers: +/-1-5 ppm or 2-5% of reading
    - Flow meters: +/-0.5-2% accuracy
    - Thermocouples: +/-0.75% or +/-2.2C (whichever is greater)
    - Pressure transmitters: +/-0.1-0.25% of span

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
from datetime import datetime
import numpy as np
import hashlib
import json


class UncertaintyType(str, Enum):
    """GUM uncertainty classification."""
    TYPE_A = "type_a"  # Statistical (repeated measurements)
    TYPE_B = "type_b"  # Systematic (manufacturer specs, calibration)
    COMBINED = "combined"


class DistributionType(str, Enum):
    """Probability distribution for uncertainty estimation."""
    NORMAL = "normal"
    RECTANGULAR = "rectangular"  # Uniform
    TRIANGULAR = "triangular"
    U_SHAPED = "u_shaped"


class SensorType(str, Enum):
    """Common combustion sensor types."""
    O2_ANALYZER = "o2_analyzer"
    CO_ANALYZER = "co_analyzer"
    NOX_ANALYZER = "nox_analyzer"
    CO2_ANALYZER = "co2_analyzer"
    FLOW_METER = "flow_meter"
    THERMOCOUPLE = "thermocouple"
    RTD = "rtd"
    PRESSURE_TRANSMITTER = "pressure_transmitter"
    FUEL_METER = "fuel_meter"


@dataclass
class SensorSpecs:
    """
    Manufacturer specifications for sensor uncertainty.

    Attributes:
        accuracy_percent: Accuracy as percentage of reading (e.g., 0.5 for +/-0.5%)
        accuracy_absolute: Fixed accuracy component (e.g., 0.1 for +/-0.1 units)
        resolution: Measurement resolution
        range_min: Minimum measurable value
        range_max: Maximum measurable value
        response_time_s: Sensor response time in seconds
        drift_per_day: Expected drift per day (units/day)
        temperature_coefficient: Temperature sensitivity (%/degC)
        calibration_interval_days: Recommended calibration interval
        distribution: Assumed probability distribution for uncertainty
    """
    accuracy_percent: float = 0.0
    accuracy_absolute: float = 0.0
    resolution: float = 0.01
    range_min: float = 0.0
    range_max: float = 100.0
    response_time_s: float = 1.0
    drift_per_day: float = 0.0
    temperature_coefficient: float = 0.0
    calibration_interval_days: int = 90
    distribution: DistributionType = DistributionType.NORMAL
    manufacturer: str = ""
    model: str = ""
    serial_number: str = ""


@dataclass
class SensorUncertainty:
    """
    Complete uncertainty characterization for a sensor.

    Follows GUM principles for uncertainty budgeting.
    """
    sensor_id: str
    sensor_type: SensorType
    specs: SensorSpecs

    # Type B uncertainty components (from specs)
    accuracy_uncertainty: float = 0.0
    resolution_uncertainty: float = 0.0
    drift_uncertainty: float = 0.0
    temperature_uncertainty: float = 0.0

    # Type A uncertainty (from calibration/measurements)
    repeatability_uncertainty: float = 0.0
    reproducibility_uncertainty: float = 0.0

    # Combined standard uncertainty
    combined_standard_uncertainty: float = 0.0

    # Coverage factor (k=2 for 95% confidence)
    coverage_factor: float = 2.0
    expanded_uncertainty: float = 0.0

    # Metadata
    last_calibration: Optional[datetime] = None
    days_since_calibration: int = 0
    provenance_hash: str = ""

    def __post_init__(self):
        """Calculate combined uncertainty after initialization."""
        self._calculate_combined_uncertainty()
        self._compute_provenance_hash()

    def _calculate_combined_uncertainty(self) -> None:
        """
        Calculate combined standard uncertainty per GUM.

        Uses root-sum-of-squares for uncorrelated components.
        """
        components = [
            self.accuracy_uncertainty,
            self.resolution_uncertainty,
            self.drift_uncertainty,
            self.temperature_uncertainty,
            self.repeatability_uncertainty,
            self.reproducibility_uncertainty,
        ]

        # RSS combination (assumes uncorrelated components)
        self.combined_standard_uncertainty = np.sqrt(sum(c**2 for c in components))
        self.expanded_uncertainty = self.coverage_factor * self.combined_standard_uncertainty

    def _compute_provenance_hash(self) -> None:
        """Compute SHA-256 hash for audit trail."""
        data = {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type.value,
            "accuracy_uncertainty": self.accuracy_uncertainty,
            "resolution_uncertainty": self.resolution_uncertainty,
            "drift_uncertainty": self.drift_uncertainty,
            "combined_standard_uncertainty": self.combined_standard_uncertainty,
            "coverage_factor": self.coverage_factor,
            "expanded_uncertainty": self.expanded_uncertainty,
        }
        self.provenance_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()


@dataclass
class MeasurementUncertainty:
    """
    Uncertainty for a specific measurement value.

    Attributes:
        value: Measured value
        standard_uncertainty: Standard uncertainty (1-sigma)
        expanded_uncertainty: Expanded uncertainty (k*sigma)
        coverage_factor: Coverage factor k
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        lower_bound: Lower confidence bound
        upper_bound: Upper confidence bound
        relative_uncertainty_percent: Relative uncertainty as percentage
    """
    value: float
    standard_uncertainty: float
    expanded_uncertainty: float
    coverage_factor: float = 2.0
    confidence_level: float = 0.95

    lower_bound: float = field(init=False)
    upper_bound: float = field(init=False)
    relative_uncertainty_percent: float = field(init=False)
    provenance_hash: str = field(init=False, default="")

    def __post_init__(self):
        self.lower_bound = self.value - self.expanded_uncertainty
        self.upper_bound = self.value + self.expanded_uncertainty
        self.relative_uncertainty_percent = (
            (self.expanded_uncertainty / abs(self.value)) * 100
            if self.value != 0 else float('inf')
        )
        self._compute_provenance_hash()

    def _compute_provenance_hash(self) -> None:
        """Compute SHA-256 hash for audit trail."""
        data = {
            "value": self.value,
            "standard_uncertainty": self.standard_uncertainty,
            "expanded_uncertainty": self.expanded_uncertainty,
            "coverage_factor": self.coverage_factor,
        }
        self.provenance_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()


@dataclass
class PropagatedUncertainty:
    """
    Result of uncertainty propagation through a calculation.

    Attributes:
        output_value: Calculated output value
        output_uncertainty: Propagated standard uncertainty
        input_contributions: Contribution of each input to total uncertainty
        sensitivity_coefficients: Partial derivatives (sensitivity)
        correlation_matrix: Input correlations (if any)
        method: Propagation method used (linear, monte_carlo)
    """
    output_value: float
    output_uncertainty: float
    expanded_uncertainty: float
    coverage_factor: float = 2.0
    input_contributions: Dict[str, float] = field(default_factory=dict)
    sensitivity_coefficients: Dict[str, float] = field(default_factory=dict)
    correlation_matrix: Optional[np.ndarray] = None
    method: str = "linear"
    provenance_hash: str = ""

    def __post_init__(self):
        self._compute_provenance_hash()

    def _compute_provenance_hash(self) -> None:
        """Compute SHA-256 hash for audit trail."""
        data = {
            "output_value": self.output_value,
            "output_uncertainty": self.output_uncertainty,
            "expanded_uncertainty": self.expanded_uncertainty,
            "method": self.method,
            "input_contributions": self.input_contributions,
        }
        self.provenance_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()


@dataclass
class CalibrationResult:
    """
    Result of sensor calibration for uncertainty adjustment.

    Attributes:
        sensor_id: Sensor identifier
        reference_value: True/reference value
        measured_value: Sensor reading
        bias: Systematic error (measured - reference)
        random_error_std: Standard deviation of random errors
        correction_factor: Multiplicative correction
        correction_offset: Additive correction
        residual_uncertainty: Uncertainty after correction
        calibration_timestamp: When calibration was performed
        is_within_tolerance: Whether sensor is within acceptable limits
    """
    sensor_id: str
    reference_value: float
    measured_value: float
    bias: float = field(init=False)
    random_error_std: float = 0.0
    correction_factor: float = 1.0
    correction_offset: float = 0.0
    residual_uncertainty: float = 0.0
    calibration_timestamp: datetime = field(default_factory=datetime.utcnow)
    is_within_tolerance: bool = True
    tolerance_limit: float = 0.0
    provenance_hash: str = ""

    def __post_init__(self):
        self.bias = self.measured_value - self.reference_value
        self._compute_provenance_hash()

    def _compute_provenance_hash(self) -> None:
        """Compute SHA-256 hash for audit trail."""
        data = {
            "sensor_id": self.sensor_id,
            "reference_value": self.reference_value,
            "measured_value": self.measured_value,
            "bias": self.bias,
            "correction_factor": self.correction_factor,
            "correction_offset": self.correction_offset,
            "residual_uncertainty": self.residual_uncertainty,
            "calibration_timestamp": self.calibration_timestamp.isoformat(),
        }
        self.provenance_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()


class SensorUncertaintyManager:
    """
    Manages uncertainty quantification for combustion sensors.

    Implements GUM (Guide to Expression of Uncertainty in Measurement)
    principles for both Type A and Type B uncertainty analysis.

    ZERO HALLUCINATION: All calculations are deterministic.
    Same inputs -> Same outputs (guaranteed).

    Example Usage:
        >>> manager = SensorUncertaintyManager()
        >>> specs = SensorSpecs(accuracy_percent=0.5, accuracy_absolute=0.1)
        >>> sensor_unc = manager.define_sensor_uncertainty("O2_001", specs, SensorType.O2_ANALYZER)
        >>> meas_unc = manager.get_measurement_uncertainty("O2_001", 3.5)
        >>> print(f"O2 = {meas_unc.value} +/- {meas_unc.expanded_uncertainty}")
    """

    # Default sensor specifications for common combustion sensors
    DEFAULT_SPECS: Dict[SensorType, SensorSpecs] = {
        SensorType.O2_ANALYZER: SensorSpecs(
            accuracy_percent=0.25,
            accuracy_absolute=0.1,
            resolution=0.01,
            range_min=0.0,
            range_max=25.0,
            response_time_s=10.0,
            drift_per_day=0.01,
            calibration_interval_days=30,
            distribution=DistributionType.NORMAL,
        ),
        SensorType.CO_ANALYZER: SensorSpecs(
            accuracy_percent=2.0,
            accuracy_absolute=5.0,
            resolution=1.0,
            range_min=0.0,
            range_max=5000.0,
            response_time_s=30.0,
            drift_per_day=0.5,
            calibration_interval_days=30,
            distribution=DistributionType.NORMAL,
        ),
        SensorType.NOX_ANALYZER: SensorSpecs(
            accuracy_percent=2.0,
            accuracy_absolute=2.0,
            resolution=0.5,
            range_min=0.0,
            range_max=1000.0,
            response_time_s=60.0,
            drift_per_day=0.2,
            calibration_interval_days=30,
            distribution=DistributionType.NORMAL,
        ),
        SensorType.FLOW_METER: SensorSpecs(
            accuracy_percent=0.75,
            accuracy_absolute=0.0,
            resolution=0.1,
            range_min=0.0,
            range_max=10000.0,
            response_time_s=0.5,
            drift_per_day=0.01,
            calibration_interval_days=365,
            distribution=DistributionType.NORMAL,
        ),
        SensorType.THERMOCOUPLE: SensorSpecs(
            accuracy_percent=0.75,
            accuracy_absolute=2.2,
            resolution=0.1,
            range_min=-200.0,
            range_max=1800.0,
            response_time_s=1.0,
            drift_per_day=0.01,
            temperature_coefficient=0.0,
            calibration_interval_days=365,
            distribution=DistributionType.NORMAL,
        ),
        SensorType.PRESSURE_TRANSMITTER: SensorSpecs(
            accuracy_percent=0.1,
            accuracy_absolute=0.0,
            resolution=0.01,
            range_min=0.0,
            range_max=100.0,
            response_time_s=0.1,
            drift_per_day=0.001,
            temperature_coefficient=0.02,
            calibration_interval_days=365,
            distribution=DistributionType.NORMAL,
        ),
        SensorType.FUEL_METER: SensorSpecs(
            accuracy_percent=0.5,
            accuracy_absolute=0.0,
            resolution=0.01,
            range_min=0.0,
            range_max=1000.0,
            response_time_s=1.0,
            drift_per_day=0.005,
            calibration_interval_days=180,
            distribution=DistributionType.NORMAL,
        ),
    }

    def __init__(self):
        """Initialize the sensor uncertainty manager."""
        self._sensors: Dict[str, SensorUncertainty] = {}
        self._calibration_history: Dict[str, List[CalibrationResult]] = {}
        self._measurement_history: Dict[str, List[MeasurementUncertainty]] = {}

    def define_sensor_uncertainty(
        self,
        sensor_id: str,
        specs: SensorSpecs,
        sensor_type: Optional[SensorType] = None,
        last_calibration: Optional[datetime] = None,
        type_a_data: Optional[np.ndarray] = None,
    ) -> SensorUncertainty:
        """
        Define uncertainty characteristics for a sensor.

        Combines Type A (statistical) and Type B (systematic) uncertainties
        following GUM principles.
        """
        if sensor_type is None:
            sensor_type = SensorType.O2_ANALYZER

        days_since_cal = 0
        if last_calibration:
            days_since_cal = (datetime.utcnow() - last_calibration).days

        accuracy_uncertainty = self._calculate_accuracy_uncertainty(specs)
        resolution_uncertainty = self._calculate_resolution_uncertainty(specs)
        drift_uncertainty = self._calculate_drift_uncertainty(specs, days_since_cal)
        temperature_uncertainty = self._calculate_temperature_uncertainty(specs)

        repeatability_uncertainty = 0.0
        reproducibility_uncertainty = 0.0
        if type_a_data is not None and len(type_a_data) > 1:
            repeatability_uncertainty = np.std(type_a_data, ddof=1) / np.sqrt(len(type_a_data))
            reproducibility_uncertainty = repeatability_uncertainty * 2.0

        sensor_uncertainty = SensorUncertainty(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            specs=specs,
            accuracy_uncertainty=accuracy_uncertainty,
            resolution_uncertainty=resolution_uncertainty,
            drift_uncertainty=drift_uncertainty,
            temperature_uncertainty=temperature_uncertainty,
            repeatability_uncertainty=repeatability_uncertainty,
            reproducibility_uncertainty=reproducibility_uncertainty,
            last_calibration=last_calibration,
            days_since_calibration=days_since_cal,
        )

        self._sensors[sensor_id] = sensor_uncertainty
        return sensor_uncertainty

    def get_measurement_uncertainty(
        self,
        sensor_id: str,
        value: float,
        coverage_factor: float = 2.0,
    ) -> MeasurementUncertainty:
        """Get measurement uncertainty for a specific value from a sensor."""
        if sensor_id not in self._sensors:
            raise ValueError(f"Sensor {sensor_id} not defined.")

        sensor_unc = self._sensors[sensor_id]
        specs = sensor_unc.specs

        percent_uncertainty = abs(value) * (specs.accuracy_percent / 100.0)
        absolute_uncertainty = specs.accuracy_absolute
        value_accuracy_uncertainty = max(percent_uncertainty, absolute_uncertainty)

        components = [
            value_accuracy_uncertainty,
            sensor_unc.resolution_uncertainty,
            sensor_unc.drift_uncertainty,
            sensor_unc.temperature_uncertainty,
            sensor_unc.repeatability_uncertainty,
        ]

        standard_uncertainty = np.sqrt(sum(c**2 for c in components))
        expanded_uncertainty = coverage_factor * standard_uncertainty

        measurement = MeasurementUncertainty(
            value=value,
            standard_uncertainty=standard_uncertainty,
            expanded_uncertainty=expanded_uncertainty,
            coverage_factor=coverage_factor,
            confidence_level=0.95 if coverage_factor == 2.0 else self._get_confidence_level(coverage_factor),
        )

        if sensor_id not in self._measurement_history:
            self._measurement_history[sensor_id] = []
        self._measurement_history[sensor_id].append(measurement)

        return measurement

    def propagate_sensor_uncertainty(
        self,
        sensors: List[Tuple[str, float]],
        calculation: Callable[..., float],
        sensitivity_coefficients: Optional[Dict[str, float]] = None,
    ) -> PropagatedUncertainty:
        """Propagate uncertainties from multiple sensors through a calculation."""
        values = {}
        uncertainties = {}

        for sensor_id, value in sensors:
            values[sensor_id] = value
            meas_unc = self.get_measurement_uncertainty(sensor_id, value)
            uncertainties[sensor_id] = meas_unc.standard_uncertainty

        output_value = calculation(**values)

        if sensitivity_coefficients is None:
            sensitivity_coefficients = self._estimate_sensitivity_coefficients(
                calculation, values, uncertainties
            )

        variance = 0.0
        input_contributions = {}

        for sensor_id in values:
            c_i = sensitivity_coefficients.get(sensor_id, 1.0)
            u_i = uncertainties[sensor_id]
            contribution = (c_i * u_i) ** 2
            variance += contribution
            input_contributions[sensor_id] = np.sqrt(contribution)

        output_uncertainty = np.sqrt(variance)

        return PropagatedUncertainty(
            output_value=output_value,
            output_uncertainty=output_uncertainty,
            expanded_uncertainty=2.0 * output_uncertainty,
            coverage_factor=2.0,
            input_contributions=input_contributions,
            sensitivity_coefficients=sensitivity_coefficients,
            method="linear",
        )

    def calibrate_uncertainty(
        self,
        sensor_id: str,
        reference: float,
        measured: float,
        tolerance: Optional[float] = None,
        update_sensor: bool = True,
    ) -> CalibrationResult:
        """Perform calibration and update sensor uncertainty."""
        if sensor_id not in self._sensors:
            raise ValueError(f"Sensor {sensor_id} not defined.")

        sensor_unc = self._sensors[sensor_id]
        bias = measured - reference

        if reference != 0:
            correction_factor = reference / measured if measured != 0 else 1.0
        else:
            correction_factor = 1.0
        correction_offset = -bias

        residual_uncertainty = sensor_unc.combined_standard_uncertainty * 0.5

        if tolerance is None:
            tolerance = sensor_unc.expanded_uncertainty
        is_within_tolerance = abs(bias) <= tolerance

        result = CalibrationResult(
            sensor_id=sensor_id,
            reference_value=reference,
            measured_value=measured,
            random_error_std=sensor_unc.repeatability_uncertainty,
            correction_factor=correction_factor,
            correction_offset=correction_offset,
            residual_uncertainty=residual_uncertainty,
            is_within_tolerance=is_within_tolerance,
            tolerance_limit=tolerance,
        )

        if sensor_id not in self._calibration_history:
            self._calibration_history[sensor_id] = []
        self._calibration_history[sensor_id].append(result)

        if update_sensor:
            sensor_unc.last_calibration = result.calibration_timestamp
            sensor_unc.days_since_calibration = 0
            sensor_unc.drift_uncertainty = 0.0
            sensor_unc._calculate_combined_uncertainty()
            sensor_unc._compute_provenance_hash()

        return result

    def get_sensor(self, sensor_id: str) -> Optional[SensorUncertainty]:
        """Get stored sensor uncertainty by ID."""
        return self._sensors.get(sensor_id)

    def list_sensors(self) -> List[str]:
        """List all registered sensor IDs."""
        return list(self._sensors.keys())

    def get_calibration_history(self, sensor_id: str) -> List[CalibrationResult]:
        """Get calibration history for a sensor."""
        return self._calibration_history.get(sensor_id, [])

    def register_default_sensor(
        self,
        sensor_id: str,
        sensor_type: SensorType,
        last_calibration: Optional[datetime] = None,
    ) -> SensorUncertainty:
        """Register a sensor with default specifications."""
        if sensor_type not in self.DEFAULT_SPECS:
            raise ValueError(f"No default specs for sensor type: {sensor_type}")

        specs = self.DEFAULT_SPECS[sensor_type]
        return self.define_sensor_uncertainty(
            sensor_id=sensor_id,
            specs=specs,
            sensor_type=sensor_type,
            last_calibration=last_calibration,
        )

    def _calculate_accuracy_uncertainty(self, specs: SensorSpecs) -> float:
        """Calculate accuracy uncertainty component."""
        midrange_value = (specs.range_max + specs.range_min) / 2
        percent_accuracy = midrange_value * (specs.accuracy_percent / 100.0)
        accuracy = max(percent_accuracy, specs.accuracy_absolute)

        if specs.distribution == DistributionType.NORMAL:
            return accuracy / 2.0
        elif specs.distribution == DistributionType.RECTANGULAR:
            return accuracy / np.sqrt(3)
        elif specs.distribution == DistributionType.TRIANGULAR:
            return accuracy / np.sqrt(6)
        else:
            return accuracy / 2.0

    def _calculate_resolution_uncertainty(self, specs: SensorSpecs) -> float:
        """Calculate resolution uncertainty component."""
        return specs.resolution / (2.0 * np.sqrt(3))

    def _calculate_drift_uncertainty(self, specs: SensorSpecs, days_since_cal: int) -> float:
        """Calculate drift uncertainty based on time since calibration."""
        drift = specs.drift_per_day * days_since_cal
        return drift / np.sqrt(3)

    def _calculate_temperature_uncertainty(
        self,
        specs: SensorSpecs,
        ambient_variation: float = 10.0,
    ) -> float:
        """Calculate temperature-induced uncertainty."""
        if specs.temperature_coefficient == 0:
            return 0.0

        midrange_value = (specs.range_max + specs.range_min) / 2
        temp_effect = midrange_value * (specs.temperature_coefficient / 100.0) * ambient_variation
        return temp_effect / np.sqrt(3)

    def _estimate_sensitivity_coefficients(
        self,
        calculation: Callable,
        values: Dict[str, float],
        uncertainties: Dict[str, float],
        delta: float = 1e-6,
    ) -> Dict[str, float]:
        """Estimate sensitivity coefficients numerically."""
        coefficients = {}

        for var_name in values:
            values_plus = values.copy()
            values_minus = values.copy()

            step = max(delta * abs(values[var_name]), delta)
            values_plus[var_name] = values[var_name] + step
            values_minus[var_name] = values[var_name] - step

            f_plus = calculation(**values_plus)
            f_minus = calculation(**values_minus)

            coefficients[var_name] = (f_plus - f_minus) / (2 * step)

        return coefficients

    def _get_confidence_level(self, coverage_factor: float) -> float:
        """Get confidence level for a given coverage factor."""
        from scipy.stats import norm
        return 2 * norm.cdf(coverage_factor) - 1
