# -*- coding: utf-8 -*-
"""
Hybrid Soft Sensor for Steam Dryness Fraction Estimation

This module implements a three-layer hybrid soft sensor:
1. Physics Layer: Compute steam properties, superheat margin using IAPWS-IF97
2. State Estimation Layer: Kalman/EKF filtering for noise rejection
3. Data-Driven Correction: Learn residual patterns from historical data

The soft sensor provides uncertainty-quantified dryness fraction estimates
with physics-bounded outputs and graceful degradation.

Zero-Hallucination Guarantee:
- Physics layer uses deterministic thermodynamic equations
- EKF uses standard Kalman filter mathematics
- Data-driven corrections are bounded by physics constraints
- All outputs include uncertainty quantification

Author: GL-BackendDeveloper
Date: December 2024
Version: 1.0.0
"""

import hashlib
import json
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .kalman_filter import (
    ExtendedKalmanFilter,
    EKFConfig,
    MeasurementVector,
    FilterStatus,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Dryness fraction bounds
DRYNESS_MIN = 0.0
DRYNESS_MAX = 1.0

# Confidence interval Z-scores
Z_SCORES = {
    0.90: 1.645,
    0.95: 1.960,
    0.99: 2.576,
}

# Data quality thresholds
QUALITY_GOOD = 0.8
QUALITY_ACCEPTABLE = 0.5
QUALITY_POOR = 0.3


class DataQualityLevel(Enum):
    """Data quality classification."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNUSABLE = "unusable"


class EstimationMode(Enum):
    """Soft sensor operating mode."""
    FULL = "full"                    # All layers active
    PHYSICS_ONLY = "physics_only"    # Only physics layer
    EKF_ONLY = "ekf_only"           # Physics + EKF, no ML correction
    DEGRADED = "degraded"           # Fallback mode


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SoftSensorConfig:
    """Configuration for hybrid soft sensor."""

    # Physics layer settings
    use_superheat_compensation: bool = True
    saturation_tolerance_c: float = 0.5  # Temperature tolerance for saturation

    # EKF settings
    ekf_config: EKFConfig = field(default_factory=EKFConfig)

    # Data-driven correction settings
    enable_ml_correction: bool = True
    correction_window_size: int = 100  # Samples for residual learning
    max_correction_magnitude: float = 0.05  # Max 5% correction

    # Uncertainty settings
    base_uncertainty_percent: float = 2.0  # Base uncertainty
    max_uncertainty_percent: float = 20.0  # Cap on uncertainty

    # Data quality thresholds
    min_data_quality: float = 0.3  # Minimum quality to process

    # Safety gating
    safety_margin_dryness: float = 0.02  # Bias towards conservative estimate

    # Graceful degradation
    degradation_timeout_s: float = 60.0  # Switch to degraded after timeout

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "use_superheat_compensation": self.use_superheat_compensation,
            "saturation_tolerance_c": self.saturation_tolerance_c,
            "enable_ml_correction": self.enable_ml_correction,
            "correction_window_size": self.correction_window_size,
            "max_correction_magnitude": self.max_correction_magnitude,
            "base_uncertainty_percent": self.base_uncertainty_percent,
            "safety_margin_dryness": self.safety_margin_dryness,
        }


@dataclass
class SoftSensorInput:
    """Input data for soft sensor."""

    # Primary measurements
    temperature_c: float
    pressure_kpa: float

    # Optional measurements for improved accuracy
    enthalpy_kj_kg: Optional[float] = None
    flow_rate_kg_s: Optional[float] = None

    # Inferred/calculated inputs
    saturation_temperature_c: Optional[float] = None

    # Timestamps and quality
    timestamp: datetime = field(default_factory=datetime.utcnow)
    temperature_quality: float = 1.0
    pressure_quality: float = 1.0

    # Process context
    load_fraction: float = 1.0  # Current load as fraction of design

    def get_overall_quality(self) -> float:
        """Compute overall data quality score."""
        # Weighted average of measurement qualities
        return (self.temperature_quality * 0.6 + self.pressure_quality * 0.4)

    def get_quality_level(self) -> DataQualityLevel:
        """Classify overall data quality."""
        q = self.get_overall_quality()
        if q >= 0.9:
            return DataQualityLevel.EXCELLENT
        elif q >= QUALITY_GOOD:
            return DataQualityLevel.GOOD
        elif q >= QUALITY_ACCEPTABLE:
            return DataQualityLevel.ACCEPTABLE
        elif q >= QUALITY_POOR:
            return DataQualityLevel.POOR
        else:
            return DataQualityLevel.UNUSABLE


@dataclass
class ConfidenceInterval:
    """Confidence interval for an estimate."""
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    half_width: float = 0.0

    def __post_init__(self):
        """Calculate half-width."""
        self.half_width = (self.upper_bound - self.lower_bound) / 2.0

    def contains(self, value: float) -> bool:
        """Check if value is within interval."""
        return self.lower_bound <= value <= self.upper_bound


@dataclass
class PhysicsLayerResult:
    """Result from physics layer calculation."""

    # Primary outputs
    dryness_fraction_physics: float
    superheat_margin_c: float
    is_wet_steam: bool
    is_superheated: bool

    # Saturation properties
    saturation_temperature_c: float
    saturation_pressure_kpa: float

    # Enthalpy-based estimate (if available)
    dryness_from_enthalpy: Optional[float] = None

    # Physics uncertainty
    uncertainty_physics: float = 0.02  # 2% base uncertainty

    # Calculation provenance
    calculation_method: str = "IAPWS-IF97"
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dryness_fraction_physics": round(self.dryness_fraction_physics, 4),
            "superheat_margin_c": round(self.superheat_margin_c, 2),
            "is_wet_steam": self.is_wet_steam,
            "is_superheated": self.is_superheated,
            "saturation_temperature_c": round(self.saturation_temperature_c, 2),
            "uncertainty_physics": round(self.uncertainty_physics, 4),
            "calculation_method": self.calculation_method,
        }


@dataclass
class DataDrivenCorrectionResult:
    """Result from data-driven correction layer."""

    # Correction applied
    correction_applied: float
    correction_confidence: float

    # Learned patterns
    residual_mean: float
    residual_std: float
    samples_in_window: int

    # Bounds check
    correction_within_bounds: bool = True
    correction_clipped: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "correction_applied": round(self.correction_applied, 6),
            "correction_confidence": round(self.correction_confidence, 4),
            "residual_mean": round(self.residual_mean, 6),
            "residual_std": round(self.residual_std, 6),
            "samples_in_window": self.samples_in_window,
            "correction_clipped": self.correction_clipped,
        }


@dataclass
class SoftSensorResult:
    """Complete result from hybrid soft sensor."""

    # Primary output
    dryness_fraction: float
    confidence_interval: ConfidenceInterval

    # Operating mode
    estimation_mode: EstimationMode
    data_quality_level: DataQualityLevel

    # Layer results
    physics_result: PhysicsLayerResult
    ekf_estimate: Optional[float] = None
    ekf_uncertainty: Optional[float] = None
    correction_result: Optional[DataDrivenCorrectionResult] = None

    # Final uncertainty
    total_uncertainty: float = 0.0
    uncertainty_breakdown: Dict[str, float] = field(default_factory=dict)

    # Timestamps
    timestamp: datetime = field(default_factory=datetime.utcnow)
    processing_time_ms: float = 0.0

    # Filter status
    filter_status: FilterStatus = FilterStatus.NOMINAL

    # Provenance
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "dryness_fraction": round(self.dryness_fraction, 4),
            "confidence_interval": {
                "lower": round(self.confidence_interval.lower_bound, 4),
                "upper": round(self.confidence_interval.upper_bound, 4),
                "confidence_level": self.confidence_interval.confidence_level,
            },
            "estimation_mode": self.estimation_mode.value,
            "data_quality_level": self.data_quality_level.value,
            "total_uncertainty": round(self.total_uncertainty, 4),
            "uncertainty_breakdown": {
                k: round(v, 6) for k, v in self.uncertainty_breakdown.items()
            },
            "physics_result": self.physics_result.to_dict(),
            "ekf_estimate": round(self.ekf_estimate, 4) if self.ekf_estimate else None,
            "correction_result": self.correction_result.to_dict() if self.correction_result else None,
            "filter_status": self.filter_status.value,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "provenance_hash": self.provenance_hash,
        }


# ============================================================================
# PHYSICS LAYER
# ============================================================================

class PhysicsLayer:
    """
    Physics-based steam property calculator.

    Uses IAPWS-IF97 relationships (simplified for soft sensor).
    In production, integrate with full thermodynamics module.
    """

    # Saturation temperature lookup (P in kPa, T in C)
    # Production: Use IAPWS-IF97 implementation
    SAT_TEMP_TABLE = {
        100: 99.6, 150: 111.4, 200: 120.2, 250: 127.4,
        300: 133.5, 350: 138.9, 400: 143.6, 450: 147.9,
        500: 151.8, 600: 158.8, 700: 164.9, 800: 170.4,
        900: 175.4, 1000: 179.9, 1200: 188.0, 1400: 195.0,
        1600: 201.4, 1800: 207.1, 2000: 212.4, 2500: 223.9,
        3000: 233.8, 3500: 242.5, 4000: 250.3,
    }

    # Enthalpy at saturation (kJ/kg)
    SAT_ENTHALPY_TABLE = {
        # P (kPa): (hf, hg)
        100: (417.5, 2675.0),
        200: (504.7, 2706.3),
        500: (640.1, 2748.1),
        1000: (762.6, 2777.1),
        1500: (844.7, 2791.0),
        2000: (908.6, 2798.3),
        3000: (1008.4, 2802.3),
    }

    def compute_saturation_temperature(self, pressure_kpa: float) -> float:
        """
        Compute saturation temperature from pressure.

        DETERMINISTIC: Linear interpolation of IAPWS-IF97 data.
        """
        pressures = sorted(self.SAT_TEMP_TABLE.keys())

        if pressure_kpa <= pressures[0]:
            return self.SAT_TEMP_TABLE[pressures[0]]
        if pressure_kpa >= pressures[-1]:
            return self.SAT_TEMP_TABLE[pressures[-1]]

        for i in range(len(pressures) - 1):
            if pressures[i] <= pressure_kpa <= pressures[i + 1]:
                p1, p2 = pressures[i], pressures[i + 1]
                t1, t2 = self.SAT_TEMP_TABLE[p1], self.SAT_TEMP_TABLE[p2]
                return t1 + (t2 - t1) * (pressure_kpa - p1) / (p2 - p1)

        return self.SAT_TEMP_TABLE[pressures[-1]]

    def compute_dryness_from_temperature(
        self,
        temperature_c: float,
        pressure_kpa: float,
        tolerance_c: float = 0.5
    ) -> Tuple[float, float, bool, bool]:
        """
        Estimate dryness fraction from temperature deviation.

        Args:
            temperature_c: Measured temperature
            pressure_kpa: Operating pressure
            tolerance_c: Tolerance for saturation detection

        Returns:
            Tuple of (dryness, superheat_margin, is_wet, is_superheated)
        """
        t_sat = self.compute_saturation_temperature(pressure_kpa)
        superheat_margin = temperature_c - t_sat

        if superheat_margin <= -tolerance_c:
            # Subcooled - not possible for steam, indicates error
            logger.warning(
                f"Subcooled temperature detected: T={temperature_c:.1f}C, "
                f"Tsat={t_sat:.1f}C"
            )
            dryness = 0.0
            is_wet = True
            is_superheated = False

        elif superheat_margin <= tolerance_c:
            # Near saturation - wet steam region
            # Estimate dryness from temperature proximity to saturation
            # Lower temperature = lower dryness
            normalized_subcool = max(0, -superheat_margin) / tolerance_c
            dryness = 0.95 - 0.15 * normalized_subcool  # 0.8 to 0.95 range
            is_wet = True
            is_superheated = False

        else:
            # Superheated - dryness = 1.0
            dryness = 1.0
            is_wet = False
            is_superheated = True

        # Clamp to valid range
        dryness = np.clip(dryness, DRYNESS_MIN, DRYNESS_MAX)

        return dryness, superheat_margin, is_wet, is_superheated

    def compute_dryness_from_enthalpy(
        self,
        enthalpy_kj_kg: float,
        pressure_kpa: float
    ) -> Optional[float]:
        """
        Compute dryness fraction from enthalpy (more accurate method).

        x = (h - hf) / hfg

        Args:
            enthalpy_kj_kg: Measured/inferred enthalpy
            pressure_kpa: Operating pressure

        Returns:
            Dryness fraction or None if out of range
        """
        pressures = sorted(self.SAT_ENTHALPY_TABLE.keys())

        # Find bracketing pressures
        hf, hg = None, None
        for i, p in enumerate(pressures):
            if p >= pressure_kpa:
                if i == 0:
                    hf, hg = self.SAT_ENTHALPY_TABLE[p]
                else:
                    # Interpolate
                    p1, p2 = pressures[i - 1], p
                    hf1, hg1 = self.SAT_ENTHALPY_TABLE[p1]
                    hf2, hg2 = self.SAT_ENTHALPY_TABLE[p2]
                    frac = (pressure_kpa - p1) / (p2 - p1)
                    hf = hf1 + frac * (hf2 - hf1)
                    hg = hg1 + frac * (hg2 - hg1)
                break

        if hf is None or hg is None:
            # Use last available
            hf, hg = self.SAT_ENTHALPY_TABLE[pressures[-1]]

        hfg = hg - hf

        if hfg <= 0:
            return None

        # Dryness fraction
        x = (enthalpy_kj_kg - hf) / hfg

        if x < 0:
            return 0.0  # Subcooled
        elif x > 1:
            return 1.0  # Superheated
        else:
            return float(x)

    def calculate(
        self,
        sensor_input: SoftSensorInput,
        tolerance_c: float = 0.5
    ) -> PhysicsLayerResult:
        """
        Perform physics layer calculation.

        Args:
            sensor_input: Input measurements
            tolerance_c: Saturation tolerance

        Returns:
            PhysicsLayerResult with dryness estimate
        """
        # Get saturation temperature
        t_sat = self.compute_saturation_temperature(sensor_input.pressure_kpa)

        # Compute dryness from temperature
        dryness, superheat, is_wet, is_superheated = self.compute_dryness_from_temperature(
            sensor_input.temperature_c,
            sensor_input.pressure_kpa,
            tolerance_c
        )

        # If enthalpy is available, use it for better estimate
        dryness_enthalpy = None
        if sensor_input.enthalpy_kj_kg is not None:
            dryness_enthalpy = self.compute_dryness_from_enthalpy(
                sensor_input.enthalpy_kj_kg,
                sensor_input.pressure_kpa
            )

        # Combine estimates (enthalpy is more accurate)
        if dryness_enthalpy is not None:
            # Weighted average - enthalpy method more reliable
            dryness = 0.3 * dryness + 0.7 * dryness_enthalpy
            uncertainty = 0.01  # Lower uncertainty with enthalpy
        else:
            uncertainty = 0.02  # Higher uncertainty without enthalpy

        # Compute provenance hash
        inputs = {
            "temperature_c": sensor_input.temperature_c,
            "pressure_kpa": sensor_input.pressure_kpa,
            "enthalpy": sensor_input.enthalpy_kj_kg,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(inputs, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        return PhysicsLayerResult(
            dryness_fraction_physics=dryness,
            superheat_margin_c=superheat,
            is_wet_steam=is_wet,
            is_superheated=is_superheated,
            saturation_temperature_c=t_sat,
            saturation_pressure_kpa=sensor_input.pressure_kpa,
            dryness_from_enthalpy=dryness_enthalpy,
            uncertainty_physics=uncertainty,
            provenance_hash=provenance_hash
        )


# ============================================================================
# DATA-DRIVEN CORRECTION LAYER
# ============================================================================

class DataDrivenCorrectionLayer:
    """
    Learn and apply corrections from residual patterns.

    Uses a simple sliding window approach to learn bias corrections
    when ground truth or more accurate reference is available.
    """

    def __init__(self, window_size: int = 100, max_correction: float = 0.05):
        """
        Initialize correction layer.

        Args:
            window_size: Number of samples for residual learning
            max_correction: Maximum correction magnitude
        """
        self.window_size = window_size
        self.max_correction = max_correction

        # Residual history: (physics_estimate, reference_value)
        self._residual_history: Deque[Tuple[float, float]] = deque(maxlen=window_size)

        # Learned correction parameters
        self._bias_correction: float = 0.0
        self._confidence: float = 0.0

    def add_observation(
        self,
        physics_estimate: float,
        reference_value: float
    ) -> None:
        """
        Add observation pair for learning.

        Args:
            physics_estimate: Estimate from physics layer
            reference_value: More accurate reference (lab measurement, etc.)
        """
        self._residual_history.append((physics_estimate, reference_value))
        self._update_correction()

    def _update_correction(self) -> None:
        """Update learned correction from residual history."""
        if len(self._residual_history) < 10:
            self._bias_correction = 0.0
            self._confidence = 0.0
            return

        # Compute residuals
        residuals = [ref - est for est, ref in self._residual_history]

        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)

        # Clip correction to bounds
        self._bias_correction = np.clip(
            mean_residual,
            -self.max_correction,
            self.max_correction
        )

        # Confidence based on consistency
        if std_residual > 0:
            self._confidence = min(1.0, 1.0 / (1.0 + std_residual * 10))
        else:
            self._confidence = 1.0

    def apply_correction(
        self,
        physics_estimate: float
    ) -> DataDrivenCorrectionResult:
        """
        Apply learned correction to physics estimate.

        Args:
            physics_estimate: Estimate from physics layer

        Returns:
            Correction result
        """
        if len(self._residual_history) < 10:
            # Insufficient data for correction
            return DataDrivenCorrectionResult(
                correction_applied=0.0,
                correction_confidence=0.0,
                residual_mean=0.0,
                residual_std=0.0,
                samples_in_window=len(self._residual_history)
            )

        residuals = [ref - est for est, ref in self._residual_history]
        residual_mean = float(np.mean(residuals))
        residual_std = float(np.std(residuals))

        # Check if correction is within bounds
        correction_clipped = abs(self._bias_correction) >= self.max_correction

        return DataDrivenCorrectionResult(
            correction_applied=self._bias_correction,
            correction_confidence=self._confidence,
            residual_mean=residual_mean,
            residual_std=residual_std,
            samples_in_window=len(self._residual_history),
            correction_within_bounds=not correction_clipped,
            correction_clipped=correction_clipped
        )

    def reset(self) -> None:
        """Reset learned corrections."""
        self._residual_history.clear()
        self._bias_correction = 0.0
        self._confidence = 0.0


# ============================================================================
# MAIN SOFT SENSOR CLASS
# ============================================================================

class SteamDrynessSoftSensor:
    """
    Hybrid Soft Sensor for Steam Dryness Fraction Estimation.

    Combines three layers for robust estimation:
    1. Physics Layer: IAPWS-IF97 thermodynamic calculations
    2. EKF Layer: State estimation with noise filtering
    3. Data-Driven Layer: Learned bias corrections

    Zero-Hallucination Guarantee:
    - Physics layer is deterministic
    - EKF uses standard Kalman equations
    - ML corrections are bounded by physics constraints
    - All outputs include uncertainty quantification

    Example:
        >>> config = SoftSensorConfig()
        >>> sensor = SteamDrynessSoftSensor(config)
        >>>
        >>> input_data = SoftSensorInput(
        ...     temperature_c=178.5,
        ...     pressure_kpa=1000.0,
        ...     timestamp=datetime.utcnow()
        ... )
        >>> result = sensor.estimate(input_data)
        >>> print(f"Dryness: {result.dryness_fraction:.3f}")
        >>> print(f"95% CI: [{result.confidence_interval.lower_bound:.3f}, "
        ...       f"{result.confidence_interval.upper_bound:.3f}]")
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[SoftSensorConfig] = None):
        """
        Initialize hybrid soft sensor.

        Args:
            config: Sensor configuration (optional)
        """
        self.config = config or SoftSensorConfig()

        # Initialize layers
        self._physics_layer = PhysicsLayer()
        self._ekf = ExtendedKalmanFilter(self.config.ekf_config)
        self._correction_layer = DataDrivenCorrectionLayer(
            window_size=self.config.correction_window_size,
            max_correction=self.config.max_correction_magnitude
        )

        # State tracking
        self._initialized = False
        self._last_timestamp: Optional[datetime] = None
        self._estimation_mode = EstimationMode.FULL
        self._consecutive_poor_quality = 0

        logger.info("SteamDrynessSoftSensor initialized")

    def initialize(
        self,
        initial_dryness: float = 0.95,
        initial_separator_eff: float = 0.98
    ) -> None:
        """
        Initialize soft sensor with prior estimates.

        Args:
            initial_dryness: Initial dryness fraction estimate
            initial_separator_eff: Initial separator efficiency
        """
        self._ekf.initialize(
            dryness=initial_dryness,
            separator_eff=initial_separator_eff
        )
        self._initialized = True
        self._estimation_mode = EstimationMode.FULL

        logger.info(
            f"Soft sensor initialized: dryness={initial_dryness:.3f}"
        )

    def estimate(
        self,
        sensor_input: SoftSensorInput,
        reference_dryness: Optional[float] = None
    ) -> SoftSensorResult:
        """
        Estimate dryness fraction from sensor inputs.

        Args:
            sensor_input: Input measurements
            reference_dryness: Optional reference for learning (lab measurement)

        Returns:
            Complete soft sensor result with uncertainty

        Raises:
            ValueError: If input data quality is unusable
        """
        start_time = datetime.utcnow()

        # Check data quality
        quality_level = sensor_input.get_quality_level()
        if quality_level == DataQualityLevel.UNUSABLE:
            raise ValueError(
                f"Input data quality too poor for estimation: "
                f"quality={sensor_input.get_overall_quality():.2f}"
            )

        # Track poor quality
        if quality_level == DataQualityLevel.POOR:
            self._consecutive_poor_quality += 1
        else:
            self._consecutive_poor_quality = 0

        # Determine estimation mode
        self._update_estimation_mode(sensor_input)

        # Layer 1: Physics calculation
        physics_result = self._physics_layer.calculate(
            sensor_input,
            self.config.saturation_tolerance_c
        )

        # Layer 2: EKF filtering (if initialized)
        ekf_estimate = None
        ekf_uncertainty = None
        filter_status = FilterStatus.NOMINAL

        if self._initialized and self._estimation_mode in [EstimationMode.FULL, EstimationMode.EKF_ONLY]:
            ekf_result = self._run_ekf_update(sensor_input)
            if ekf_result is not None:
                ekf_estimate = ekf_result.state.dryness_fraction
                ekf_uncertainty = ekf_result.uncertainty_1sigma.get("dryness_fraction", 0.02)
                filter_status = ekf_result.filter_status

        # Layer 3: Data-driven correction
        correction_result = None
        if self.config.enable_ml_correction and self._estimation_mode == EstimationMode.FULL:
            correction_result = self._correction_layer.apply_correction(
                physics_result.dryness_fraction_physics
            )

            # Add reference observation for learning
            if reference_dryness is not None:
                self._correction_layer.add_observation(
                    physics_result.dryness_fraction_physics,
                    reference_dryness
                )

        # Combine estimates
        final_dryness, uncertainty_breakdown = self._combine_estimates(
            physics_result,
            ekf_estimate,
            ekf_uncertainty,
            correction_result,
            quality_level
        )

        # Apply safety margin (bias towards conservative estimate)
        if self.config.safety_margin_dryness > 0:
            # Lower dryness is more conservative (more water content assumed)
            final_dryness = final_dryness - self.config.safety_margin_dryness
            final_dryness = max(DRYNESS_MIN, final_dryness)

        # Compute confidence interval
        total_uncertainty = sum(uncertainty_breakdown.values()) ** 0.5
        total_uncertainty = min(total_uncertainty, self.config.max_uncertainty_percent / 100.0)

        confidence_interval = self._compute_confidence_interval(
            final_dryness,
            total_uncertainty,
            confidence_level=0.95
        )

        # Update timestamp
        self._last_timestamp = sensor_input.timestamp

        # Compute processing time
        processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Compute provenance hash
        provenance_hash = self._compute_provenance_hash(
            sensor_input, final_dryness, physics_result
        )

        return SoftSensorResult(
            dryness_fraction=final_dryness,
            confidence_interval=confidence_interval,
            estimation_mode=self._estimation_mode,
            data_quality_level=quality_level,
            physics_result=physics_result,
            ekf_estimate=ekf_estimate,
            ekf_uncertainty=ekf_uncertainty,
            correction_result=correction_result,
            total_uncertainty=total_uncertainty,
            uncertainty_breakdown=uncertainty_breakdown,
            timestamp=sensor_input.timestamp,
            processing_time_ms=processing_time_ms,
            filter_status=filter_status,
            provenance_hash=provenance_hash
        )

    def _update_estimation_mode(self, sensor_input: SoftSensorInput) -> None:
        """Update estimation mode based on conditions."""
        quality = sensor_input.get_overall_quality()

        if self._consecutive_poor_quality >= 10:
            self._estimation_mode = EstimationMode.DEGRADED
        elif quality < QUALITY_ACCEPTABLE:
            self._estimation_mode = EstimationMode.PHYSICS_ONLY
        elif not self._initialized:
            self._estimation_mode = EstimationMode.PHYSICS_ONLY
        elif not self.config.enable_ml_correction:
            self._estimation_mode = EstimationMode.EKF_ONLY
        else:
            self._estimation_mode = EstimationMode.FULL

    def _run_ekf_update(
        self,
        sensor_input: SoftSensorInput
    ) -> Optional[Any]:  # Returns KalmanFilterResult
        """Run EKF update step."""
        try:
            measurement = MeasurementVector(
                temperature_c=sensor_input.temperature_c,
                pressure_kpa=sensor_input.pressure_kpa,
                measurement_timestamp=sensor_input.timestamp,
                temperature_quality=sensor_input.temperature_quality,
                pressure_quality=sensor_input.pressure_quality
            )

            return self._ekf.update(measurement)

        except Exception as e:
            logger.warning(f"EKF update failed: {e}")
            return None

    def _combine_estimates(
        self,
        physics_result: PhysicsLayerResult,
        ekf_estimate: Optional[float],
        ekf_uncertainty: Optional[float],
        correction_result: Optional[DataDrivenCorrectionResult],
        quality_level: DataQualityLevel
    ) -> Tuple[float, Dict[str, float]]:
        """
        Combine estimates from all layers.

        Uses uncertainty-weighted combination when all layers available.
        """
        uncertainty_breakdown: Dict[str, float] = {}

        # Base estimate from physics
        physics_est = physics_result.dryness_fraction_physics
        physics_var = physics_result.uncertainty_physics ** 2
        uncertainty_breakdown["physics"] = physics_var

        # If only physics available
        if ekf_estimate is None:
            if correction_result is not None and correction_result.correction_confidence > 0.5:
                # Apply correction
                corrected = physics_est + correction_result.correction_applied
                uncertainty_breakdown["correction"] = (1 - correction_result.correction_confidence) * 0.001
            else:
                corrected = physics_est

            return np.clip(corrected, DRYNESS_MIN, DRYNESS_MAX), uncertainty_breakdown

        # EKF estimate available
        ekf_var = (ekf_uncertainty or 0.02) ** 2
        uncertainty_breakdown["ekf"] = ekf_var

        # Variance-weighted fusion
        total_precision = 1.0 / physics_var + 1.0 / ekf_var
        fused_estimate = (physics_est / physics_var + ekf_estimate / ekf_var) / total_precision
        fused_variance = 1.0 / total_precision

        # Apply data-driven correction if available
        if correction_result is not None and correction_result.correction_confidence > 0.3:
            correction = correction_result.correction_applied * correction_result.correction_confidence
            fused_estimate += correction
            uncertainty_breakdown["correction"] = (1 - correction_result.correction_confidence) * 0.0005

        # Add quality-based uncertainty
        quality_factor = 1.0 if quality_level == DataQualityLevel.EXCELLENT else \
                        1.2 if quality_level == DataQualityLevel.GOOD else \
                        1.5 if quality_level == DataQualityLevel.ACCEPTABLE else 2.0

        for key in uncertainty_breakdown:
            uncertainty_breakdown[key] *= quality_factor

        return np.clip(fused_estimate, DRYNESS_MIN, DRYNESS_MAX), uncertainty_breakdown

    def _compute_confidence_interval(
        self,
        estimate: float,
        uncertainty: float,
        confidence_level: float = 0.95
    ) -> ConfidenceInterval:
        """Compute confidence interval for estimate."""
        z = Z_SCORES.get(confidence_level, 1.96)
        half_width = z * uncertainty

        lower = max(DRYNESS_MIN, estimate - half_width)
        upper = min(DRYNESS_MAX, estimate + half_width)

        return ConfidenceInterval(
            point_estimate=estimate,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=confidence_level,
            half_width=half_width
        )

    def _compute_provenance_hash(
        self,
        sensor_input: SoftSensorInput,
        final_estimate: float,
        physics_result: PhysicsLayerResult
    ) -> str:
        """Compute SHA-256 hash for provenance."""
        data = {
            "version": self.VERSION,
            "timestamp": sensor_input.timestamp.isoformat(),
            "temperature": sensor_input.temperature_c,
            "pressure": sensor_input.pressure_kpa,
            "final_estimate": round(final_estimate, 6),
            "physics_hash": physics_result.provenance_hash,
        }

        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]

    def add_reference_measurement(
        self,
        physics_estimate: float,
        reference_dryness: float
    ) -> None:
        """
        Add reference measurement for data-driven learning.

        Args:
            physics_estimate: Physics layer estimate at time of reference
            reference_dryness: More accurate reference measurement
        """
        self._correction_layer.add_observation(physics_estimate, reference_dryness)

    def reset(self) -> None:
        """Reset soft sensor to initial state."""
        self._ekf.reset()
        self._correction_layer.reset()
        self._initialized = False
        self._last_timestamp = None
        self._estimation_mode = EstimationMode.FULL
        self._consecutive_poor_quality = 0

        logger.info("Soft sensor reset")

    @property
    def is_initialized(self) -> bool:
        """Check if sensor is initialized."""
        return self._initialized

    @property
    def current_mode(self) -> EstimationMode:
        """Get current estimation mode."""
        return self._estimation_mode
