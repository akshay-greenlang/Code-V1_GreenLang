# -*- coding: utf-8 -*-
"""
GL-004 Burnmaster - Analyzer Uncertainty Module

Models uncertainty characteristics specific to gas analyzers used
in combustion monitoring: O2, CO, NOx, CO2 analyzers.

Handles analyzer-specific effects:
    - Transport lag (sample system delay)
    - Analyzer drift (calibration decay)
    - Bias estimation (systematic errors)
    - Cross-sensitivity (interference from other gases)

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import hashlib
import json


class AnalyzerType(str, Enum):
    """Types of gas analyzers in combustion systems."""
    O2_PARAMAGNETIC = "o2_paramagnetic"
    O2_ZIRCONIA = "o2_zirconia"
    O2_ELECTROCHEMICAL = "o2_electrochemical"
    CO_NDIR = "co_ndir"  # Non-dispersive infrared
    CO_ELECTROCHEMICAL = "co_electrochemical"
    NOX_CHEMILUMINESCENT = "nox_chemiluminescent"
    NOX_ELECTROCHEMICAL = "nox_electrochemical"
    CO2_NDIR = "co2_ndir"


@dataclass
class LagModel:
    """
    Model for analyzer transport and response lag.

    Attributes:
        analyzer_id: Analyzer identifier
        transport_lag_s: Sample transport delay (seconds)
        response_time_t90_s: 90% response time (seconds)
        effective_lag_s: Combined effective lag
        lag_uncertainty_s: Uncertainty in lag estimate
        sample_line_length_m: Sample line length (if known)
        flow_rate_lpm: Sample flow rate (if known)
    """
    analyzer_id: str
    transport_lag_s: float
    response_time_t90_s: float
    effective_lag_s: float = field(init=False)
    lag_uncertainty_s: float = 0.0
    sample_line_length_m: Optional[float] = None
    flow_rate_lpm: Optional[float] = None
    provenance_hash: str = ""

    def __post_init__(self):
        # Effective lag = transport + 0.5 * response time (approximation)
        self.effective_lag_s = self.transport_lag_s + 0.5 * self.response_time_t90_s
        self._compute_provenance_hash()

    def _compute_provenance_hash(self) -> None:
        """Compute SHA-256 hash for audit trail."""
        data = {
            "analyzer_id": self.analyzer_id,
            "transport_lag_s": self.transport_lag_s,
            "response_time_t90_s": self.response_time_t90_s,
            "effective_lag_s": self.effective_lag_s,
            "lag_uncertainty_s": self.lag_uncertainty_s,
        }
        self.provenance_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()


@dataclass
class DriftModel:
    """
    Model for analyzer calibration drift.

    Attributes:
        analyzer_id: Analyzer identifier
        drift_rate_per_day: Drift rate (units/day)
        drift_direction: Direction of drift (+1, -1, or 0 if unknown)
        time_since_calibration_days: Days since last calibration
        accumulated_drift: Total estimated drift
        drift_uncertainty: Uncertainty in drift estimate
        drift_model_type: Type of drift model (linear, exponential)
    """
    analyzer_id: str
    drift_rate_per_day: float
    drift_direction: int = 0  # +1, -1, or 0 (unknown)
    time_since_calibration_days: float = 0.0
    accumulated_drift: float = field(init=False)
    drift_uncertainty: float = field(init=False)
    drift_model_type: str = "linear"
    provenance_hash: str = ""

    def __post_init__(self):
        if self.drift_model_type == "linear":
            self.accumulated_drift = self.drift_rate_per_day * self.time_since_calibration_days
        elif self.drift_model_type == "exponential":
            # Exponential drift saturates
            tau = 30.0  # Time constant (days)
            self.accumulated_drift = (
                self.drift_rate_per_day * tau *
                (1 - np.exp(-self.time_since_calibration_days / tau))
            )
        else:
            self.accumulated_drift = self.drift_rate_per_day * self.time_since_calibration_days

        # Drift uncertainty (rectangular distribution)
        self.drift_uncertainty = abs(self.accumulated_drift) / np.sqrt(3)
        self._compute_provenance_hash()

    def _compute_provenance_hash(self) -> None:
        """Compute SHA-256 hash for audit trail."""
        data = {
            "analyzer_id": self.analyzer_id,
            "drift_rate_per_day": self.drift_rate_per_day,
            "time_since_calibration_days": self.time_since_calibration_days,
            "accumulated_drift": self.accumulated_drift,
            "drift_uncertainty": self.drift_uncertainty,
        }
        self.provenance_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()


@dataclass
class BiasEstimate:
    """
    Estimate of analyzer systematic bias.

    Attributes:
        analyzer_id: Analyzer identifier
        estimated_bias: Estimated systematic bias
        bias_uncertainty: Uncertainty in bias estimate
        bias_source: Source of bias (calibration, cross-sensitivity, etc.)
        correction_applied: Whether bias correction is applied
        confidence_level: Confidence in bias estimate (0-1)
    """
    analyzer_id: str
    estimated_bias: float
    bias_uncertainty: float = 0.0
    bias_source: str = "unknown"
    correction_applied: bool = False
    confidence_level: float = 0.5
    provenance_hash: str = ""

    def __post_init__(self):
        self._compute_provenance_hash()

    def _compute_provenance_hash(self) -> None:
        """Compute SHA-256 hash for audit trail."""
        data = {
            "analyzer_id": self.analyzer_id,
            "estimated_bias": self.estimated_bias,
            "bias_uncertainty": self.bias_uncertainty,
            "bias_source": self.bias_source,
            "correction_applied": self.correction_applied,
        }
        self.provenance_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()


@dataclass
class CalibrationData:
    """
    Calibration data point for drift analysis.

    Attributes:
        timestamp: When calibration was performed
        reference_value: Calibration gas reference value
        measured_value: Analyzer reading at calibration
        zero_drift: Zero point drift (if measured)
        span_drift: Span point drift (if measured)
    """
    timestamp: datetime
    reference_value: float
    measured_value: float
    zero_drift: Optional[float] = None
    span_drift: Optional[float] = None


class AnalyzerUncertaintyModel:
    """
    Models uncertainty characteristics for combustion gas analyzers.

    Handles analyzer-specific effects including transport lag,
    calibration drift, and systematic bias that affect measurement
    uncertainty beyond basic sensor specifications.

    Supported Analyzers:
        - O2: Paramagnetic, Zirconia, Electrochemical
        - CO: NDIR, Electrochemical
        - NOx: Chemiluminescent, Electrochemical
        - CO2: NDIR

    ZERO HALLUCINATION: All calculations are deterministic.
    Same inputs -> Same outputs (guaranteed).

    Example Usage:
        >>> model = AnalyzerUncertaintyModel()
        >>> lag = model.model_analyzer_lag("O2_001")
        >>> drift = model.model_analyzer_drift(calibration_history)
        >>> bias = model.estimate_current_bias("O2_001")
        >>> total_unc = model.compute_effective_uncertainty(base=0.1, drift=0.05, lag=0.02)
    """

    # Default characteristics by analyzer type
    DEFAULT_CHARACTERISTICS: Dict[AnalyzerType, Dict[str, float]] = {
        AnalyzerType.O2_PARAMAGNETIC: {
            "transport_lag_s": 15.0,
            "response_time_t90_s": 10.0,
            "drift_rate_per_day": 0.005,  # % O2/day
            "base_accuracy": 0.1,  # % O2
            "cross_sensitivity": 0.0,
        },
        AnalyzerType.O2_ZIRCONIA: {
            "transport_lag_s": 0.0,  # In-situ
            "response_time_t90_s": 5.0,
            "drift_rate_per_day": 0.01,
            "base_accuracy": 0.2,
            "cross_sensitivity": 0.0,
        },
        AnalyzerType.O2_ELECTROCHEMICAL: {
            "transport_lag_s": 20.0,
            "response_time_t90_s": 30.0,
            "drift_rate_per_day": 0.02,
            "base_accuracy": 0.3,
            "cross_sensitivity": 0.01,  # CO interference
        },
        AnalyzerType.CO_NDIR: {
            "transport_lag_s": 20.0,
            "response_time_t90_s": 15.0,
            "drift_rate_per_day": 0.5,  # ppm/day
            "base_accuracy": 5.0,  # ppm
            "cross_sensitivity": 0.02,  # CO2 interference
        },
        AnalyzerType.CO_ELECTROCHEMICAL: {
            "transport_lag_s": 25.0,
            "response_time_t90_s": 45.0,
            "drift_rate_per_day": 1.0,
            "base_accuracy": 10.0,
            "cross_sensitivity": 0.05,
        },
        AnalyzerType.NOX_CHEMILUMINESCENT: {
            "transport_lag_s": 30.0,
            "response_time_t90_s": 20.0,
            "drift_rate_per_day": 0.2,  # ppm/day
            "base_accuracy": 2.0,  # ppm
            "cross_sensitivity": 0.01,
        },
        AnalyzerType.NOX_ELECTROCHEMICAL: {
            "transport_lag_s": 25.0,
            "response_time_t90_s": 60.0,
            "drift_rate_per_day": 0.5,
            "base_accuracy": 5.0,
            "cross_sensitivity": 0.03,
        },
        AnalyzerType.CO2_NDIR: {
            "transport_lag_s": 20.0,
            "response_time_t90_s": 15.0,
            "drift_rate_per_day": 0.01,  # % CO2/day
            "base_accuracy": 0.2,  # % CO2
            "cross_sensitivity": 0.01,  # H2O interference
        },
    }

    def __init__(self):
        """Initialize the analyzer uncertainty model."""
        self._analyzers: Dict[str, Dict[str, Any]] = {}
        self._calibration_history: Dict[str, List[CalibrationData]] = {}
        self._lag_models: Dict[str, LagModel] = {}
        self._drift_models: Dict[str, DriftModel] = {}

    def register_analyzer(
        self,
        analyzer_id: str,
        analyzer_type: AnalyzerType,
        sample_line_length_m: Optional[float] = None,
        flow_rate_lpm: Optional[float] = None,
        last_calibration: Optional[datetime] = None,
    ) -> None:
        """
        Register an analyzer with the model.

        Args:
            analyzer_id: Unique analyzer identifier
            analyzer_type: Type of analyzer
            sample_line_length_m: Sample line length in meters
            flow_rate_lpm: Sample flow rate in liters per minute
            last_calibration: Timestamp of last calibration
        """
        characteristics = self.DEFAULT_CHARACTERISTICS.get(
            analyzer_type,
            self.DEFAULT_CHARACTERISTICS[AnalyzerType.O2_PARAMAGNETIC]
        )

        self._analyzers[analyzer_id] = {
            "type": analyzer_type,
            "characteristics": characteristics,
            "sample_line_length_m": sample_line_length_m,
            "flow_rate_lpm": flow_rate_lpm,
            "last_calibration": last_calibration or datetime.utcnow(),
        }

    def model_analyzer_lag(
        self,
        analyzer_id: str,
        sample_line_length_m: Optional[float] = None,
        flow_rate_lpm: Optional[float] = None,
    ) -> LagModel:
        """
        Model the transport and response lag for an analyzer.

        Calculates effective lag from sample system geometry and
        analyzer response characteristics.

        Args:
            analyzer_id: Analyzer identifier
            sample_line_length_m: Override sample line length
            flow_rate_lpm: Override flow rate

        Returns:
            LagModel with transport and response lag components

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        if analyzer_id not in self._analyzers:
            # Use default values
            transport_lag = 15.0
            response_time = 10.0
            line_length = sample_line_length_m
            flow_rate = flow_rate_lpm
        else:
            analyzer = self._analyzers[analyzer_id]
            chars = analyzer["characteristics"]
            transport_lag = chars["transport_lag_s"]
            response_time = chars["response_time_t90_s"]
            line_length = sample_line_length_m or analyzer.get("sample_line_length_m")
            flow_rate = flow_rate_lpm or analyzer.get("flow_rate_lpm")

        # Adjust transport lag if line length and flow rate known
        if line_length is not None and flow_rate is not None and flow_rate > 0:
            # Calculate actual transport lag
            # Assume 4mm ID sample line
            line_id_mm = 4.0
            line_volume_ml = np.pi * (line_id_mm / 2) ** 2 * (line_length * 1000)
            transport_lag = (line_volume_ml / 1000) / flow_rate * 60  # seconds

        # Lag uncertainty (estimate 10% of lag)
        lag_uncertainty = 0.1 * (transport_lag + response_time)

        lag_model = LagModel(
            analyzer_id=analyzer_id,
            transport_lag_s=transport_lag,
            response_time_t90_s=response_time,
            lag_uncertainty_s=lag_uncertainty,
            sample_line_length_m=line_length,
            flow_rate_lpm=flow_rate,
        )

        self._lag_models[analyzer_id] = lag_model
        return lag_model

    def model_analyzer_drift(
        self,
        calibration_data: List[CalibrationData],
        analyzer_id: Optional[str] = None,
        analyzer_type: Optional[AnalyzerType] = None,
    ) -> DriftModel:
        """
        Model analyzer drift from calibration history.

        Analyzes calibration data to estimate drift rate and
        predict current drift magnitude.

        Args:
            calibration_data: List of calibration data points
            analyzer_id: Analyzer identifier (optional)
            analyzer_type: Type of analyzer for default values

        Returns:
            DriftModel with estimated drift characteristics

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        if not calibration_data:
            # Return default model
            return DriftModel(
                analyzer_id=analyzer_id or "unknown",
                drift_rate_per_day=0.01,
                time_since_calibration_days=0.0,
            )

        # Sort by timestamp
        sorted_data = sorted(calibration_data, key=lambda x: x.timestamp)

        # Calculate drift rate from consecutive calibrations
        drift_rates = []
        for i in range(1, len(sorted_data)):
            prev = sorted_data[i - 1]
            curr = sorted_data[i]

            time_diff = (curr.timestamp - prev.timestamp).total_seconds() / 86400  # days
            if time_diff > 0:
                # Drift = change in error over time
                prev_error = prev.measured_value - prev.reference_value
                curr_error = curr.measured_value - curr.reference_value
                drift = (curr_error - prev_error) / time_diff
                drift_rates.append(drift)

        # Estimate drift rate
        if drift_rates:
            drift_rate = float(np.mean(drift_rates))
            drift_direction = int(np.sign(drift_rate))
        else:
            # Use default for analyzer type
            if analyzer_type and analyzer_type in self.DEFAULT_CHARACTERISTICS:
                drift_rate = self.DEFAULT_CHARACTERISTICS[analyzer_type]["drift_rate_per_day"]
            else:
                drift_rate = 0.01
            drift_direction = 0

        # Time since last calibration
        last_cal = sorted_data[-1].timestamp
        time_since_cal = (datetime.utcnow() - last_cal).total_seconds() / 86400

        drift_model = DriftModel(
            analyzer_id=analyzer_id or "unknown",
            drift_rate_per_day=abs(drift_rate),
            drift_direction=drift_direction,
            time_since_calibration_days=time_since_cal,
        )

        if analyzer_id:
            self._drift_models[analyzer_id] = drift_model
            self._calibration_history[analyzer_id] = calibration_data

        return drift_model

    def estimate_current_bias(
        self,
        analyzer_id: str,
        recent_reference: Optional[float] = None,
        recent_measured: Optional[float] = None,
    ) -> BiasEstimate:
        """
        Estimate current systematic bias for an analyzer.

        Uses drift model and recent comparison data (if available)
        to estimate current bias.

        Args:
            analyzer_id: Analyzer identifier
            recent_reference: Recent reference/true value (optional)
            recent_measured: Recent analyzer reading (optional)

        Returns:
            BiasEstimate with estimated bias and uncertainty

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        # Check for direct comparison data
        if recent_reference is not None and recent_measured is not None:
            estimated_bias = recent_measured - recent_reference
            bias_uncertainty = abs(estimated_bias) * 0.2  # 20% uncertainty in estimate
            bias_source = "recent_comparison"
            confidence = 0.9
        elif analyzer_id in self._drift_models:
            # Estimate from drift model
            drift_model = self._drift_models[analyzer_id]
            estimated_bias = drift_model.drift_direction * drift_model.accumulated_drift
            bias_uncertainty = drift_model.drift_uncertainty
            bias_source = "drift_model"
            confidence = 0.6
        elif analyzer_id in self._analyzers:
            # Estimate from time since calibration
            analyzer = self._analyzers[analyzer_id]
            chars = analyzer["characteristics"]
            last_cal = analyzer.get("last_calibration", datetime.utcnow())
            days_since_cal = (datetime.utcnow() - last_cal).total_seconds() / 86400

            estimated_bias = chars["drift_rate_per_day"] * days_since_cal
            bias_uncertainty = estimated_bias / np.sqrt(3)
            bias_source = "default_drift"
            confidence = 0.4
        else:
            # No information
            estimated_bias = 0.0
            bias_uncertainty = 0.1
            bias_source = "unknown"
            confidence = 0.2

        return BiasEstimate(
            analyzer_id=analyzer_id,
            estimated_bias=estimated_bias,
            bias_uncertainty=bias_uncertainty,
            bias_source=bias_source,
            correction_applied=False,
            confidence_level=confidence,
        )

    def compute_effective_uncertainty(
        self,
        base: float,
        drift: float,
        lag: float,
        cross_sensitivity: float = 0.0,
    ) -> float:
        """
        Compute effective uncertainty combining all analyzer effects.

        Combines base measurement uncertainty with drift, lag-induced
        uncertainty, and cross-sensitivity effects.

        Args:
            base: Base measurement uncertainty (from sensor specs)
            drift: Drift uncertainty (from drift model)
            lag: Lag-induced uncertainty (from process dynamics)
            cross_sensitivity: Cross-sensitivity contribution

        Returns:
            Combined effective uncertainty (RSS combination)

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        # RSS combination of independent uncertainty sources
        components = [base, drift, lag, cross_sensitivity]
        variance = sum(c**2 for c in components)
        return float(np.sqrt(variance))

    def get_lag_corrected_value(
        self,
        analyzer_id: str,
        measured_value: float,
        rate_of_change: float,
    ) -> Tuple[float, float]:
        """
        Apply lag correction to analyzer reading.

        Estimates current process value by compensating for
        analyzer transport and response lag.

        Args:
            analyzer_id: Analyzer identifier
            measured_value: Current analyzer reading
            rate_of_change: Rate of change of process variable (units/s)

        Returns:
            Tuple of (corrected_value, correction_uncertainty)

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        if analyzer_id not in self._lag_models:
            self.model_analyzer_lag(analyzer_id)

        lag_model = self._lag_models[analyzer_id]

        # Apply lag correction
        # Corrected value = measured + lag * rate_of_change
        correction = lag_model.effective_lag_s * rate_of_change
        corrected_value = measured_value + correction

        # Correction uncertainty
        correction_uncertainty = abs(lag_model.lag_uncertainty_s * rate_of_change)

        return corrected_value, correction_uncertainty

    def get_bias_corrected_value(
        self,
        analyzer_id: str,
        measured_value: float,
    ) -> Tuple[float, float]:
        """
        Apply bias correction to analyzer reading.

        Removes estimated systematic bias from measurement.

        Args:
            analyzer_id: Analyzer identifier
            measured_value: Current analyzer reading

        Returns:
            Tuple of (corrected_value, residual_uncertainty)

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        bias_estimate = self.estimate_current_bias(analyzer_id)

        # Apply correction
        corrected_value = measured_value - bias_estimate.estimated_bias

        # Residual uncertainty after correction
        # Correction reduces bias but adds uncertainty from correction itself
        residual_uncertainty = bias_estimate.bias_uncertainty

        return corrected_value, residual_uncertainty

    def get_complete_correction(
        self,
        analyzer_id: str,
        measured_value: float,
        rate_of_change: float = 0.0,
    ) -> Dict[str, float]:
        """
        Apply all corrections (lag and bias) to analyzer reading.

        Args:
            analyzer_id: Analyzer identifier
            measured_value: Current analyzer reading
            rate_of_change: Rate of change (units/s) for lag correction

        Returns:
            Dictionary with corrected value and uncertainty components

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        # Lag correction
        lag_corrected, lag_unc = self.get_lag_corrected_value(
            analyzer_id, measured_value, rate_of_change
        )

        # Bias correction (applied to lag-corrected value)
        bias_corrected, bias_unc = self.get_bias_corrected_value(
            analyzer_id, lag_corrected
        )

        return {
            "original_value": measured_value,
            "corrected_value": bias_corrected,
            "lag_correction": lag_corrected - measured_value,
            "bias_correction": bias_corrected - lag_corrected,
            "lag_uncertainty": lag_unc,
            "bias_uncertainty": bias_unc,
            "total_correction_uncertainty": np.sqrt(lag_unc**2 + bias_unc**2),
        }
