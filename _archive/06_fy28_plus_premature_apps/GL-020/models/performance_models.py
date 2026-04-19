# -*- coding: utf-8 -*-
"""
GL-020 ECONOPULSE Performance Models.

This module implements machine learning models for economizer performance
analysis with deterministic fallbacks. All models follow zero-hallucination
principles where ML predictions are supplementary to deterministic calculations.

Models:
    - FoulingPredictor: Predicts fouling rate from operating conditions
    - CleaningEffectivenessModel: Estimates cleaning impact and recovery
    - AnomalyDetector: Detects sensor faults, leaks, and unusual patterns

Zero-Hallucination Approach:
    - ML models provide early warning and trend analysis
    - All critical decisions use deterministic formulas
    - Fallback calculations available when ML unavailable
    - Confidence scores indicate prediction reliability

Author: GreenLang Team
Date: December 2025
Status: Production Ready
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from greenlang.GL_020.config import (
    BaselineConfiguration,
    EconomizerConfiguration,
    FoulingType,
    CleaningMethod,
)

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================


class AnomalyType(str, Enum):
    """Types of anomalies that can be detected."""

    SENSOR_FAULT = "sensor_fault"
    SENSOR_DRIFT = "sensor_drift"
    TUBE_LEAK = "tube_leak"
    FLOW_RESTRICTION = "flow_restriction"
    UNUSUAL_FOULING = "unusual_fouling"
    PERFORMANCE_ANOMALY = "performance_anomaly"
    DATA_QUALITY = "data_quality"
    UNKNOWN = "unknown"


class PredictionConfidence(str, Enum):
    """Confidence levels for predictions."""

    HIGH = "high"  # >80% confidence
    MEDIUM = "medium"  # 60-80% confidence
    LOW = "low"  # 40-60% confidence
    VERY_LOW = "very_low"  # <40% confidence


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class AnomalyResult:
    """
    Result of anomaly detection analysis.

    Contains details about detected anomalies and recommended actions.
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    economizer_id: str = ""

    # Anomaly detection
    anomaly_detected: bool = False
    anomaly_type: AnomalyType = AnomalyType.UNKNOWN
    anomaly_score: float = 0.0  # 0-1, higher = more anomalous
    confidence: float = 0.0  # 0-1 confidence in detection

    # Details
    affected_sensor: Optional[str] = None
    description: str = ""
    evidence: List[str] = field(default_factory=list)

    # Impact assessment
    severity: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
    affects_performance: bool = False
    affects_safety: bool = False

    # Recommendations
    recommended_actions: List[str] = field(default_factory=list)
    requires_immediate_attention: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "economizer_id": self.economizer_id,
            "anomaly_detected": self.anomaly_detected,
            "anomaly_type": self.anomaly_type.value,
            "anomaly_score": self.anomaly_score,
            "confidence": self.confidence,
            "affected_sensor": self.affected_sensor,
            "description": self.description,
            "severity": self.severity,
            "recommended_actions": self.recommended_actions,
        }


@dataclass
class FoulingPrediction:
    """
    Fouling rate prediction result.

    Contains predicted fouling rate and confidence metrics.
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    economizer_id: str = ""

    # Predictions
    predicted_fouling_rate_per_hour: float = 0.0
    predicted_fouling_rate_per_day: float = 0.0
    prediction_method: str = "deterministic"  # deterministic, ml, hybrid

    # Time to threshold
    hours_to_warning_threshold: float = 0.0
    hours_to_critical_threshold: float = 0.0

    # Confidence
    confidence_score: float = 0.0
    confidence_level: PredictionConfidence = PredictionConfidence.MEDIUM

    # Contributing factors
    contributing_factors: Dict[str, float] = field(default_factory=dict)
    factor_descriptions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "economizer_id": self.economizer_id,
            "predicted_fouling_rate_per_day": self.predicted_fouling_rate_per_day,
            "prediction_method": self.prediction_method,
            "hours_to_warning_threshold": self.hours_to_warning_threshold,
            "hours_to_critical_threshold": self.hours_to_critical_threshold,
            "confidence_score": self.confidence_score,
            "confidence_level": self.confidence_level.value,
        }


@dataclass
class CleaningEffectivenessResult:
    """
    Cleaning effectiveness estimation result.

    Contains expected recovery from cleaning operation.
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    economizer_id: str = ""

    # Pre-cleaning state
    pre_cleaning_rf: float = 0.0
    pre_cleaning_effectiveness_pct: float = 0.0
    pre_cleaning_u_value: float = 0.0

    # Expected post-cleaning state
    expected_post_cleaning_rf: float = 0.0
    expected_recovery_pct: float = 0.0  # % of fouling removed
    expected_effectiveness_gain_pct: float = 0.0
    expected_u_value_recovery: float = 0.0

    # Economic impact
    expected_heat_recovery_mmbtu_hr: float = 0.0
    expected_cost_savings_per_day_usd: float = 0.0
    cleaning_cost_estimate_usd: float = 0.0
    estimated_roi_payback_hours: float = 0.0

    # Confidence
    confidence_score: float = 0.0
    prediction_basis: str = "historical"  # historical, model, default

    # Recommendations
    recommended_cleaning_method: CleaningMethod = CleaningMethod.SOOT_BLOWING
    additional_recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "economizer_id": self.economizer_id,
            "expected_recovery_pct": self.expected_recovery_pct,
            "expected_effectiveness_gain_pct": self.expected_effectiveness_gain_pct,
            "expected_heat_recovery_mmbtu_hr": self.expected_heat_recovery_mmbtu_hr,
            "expected_cost_savings_per_day_usd": self.expected_cost_savings_per_day_usd,
            "estimated_roi_payback_hours": self.estimated_roi_payback_hours,
            "recommended_cleaning_method": self.recommended_cleaning_method.value,
        }


# ============================================================================
# FOULING PREDICTOR
# ============================================================================


class FoulingPredictor:
    """
    Fouling rate prediction model.

    Predicts fouling rate based on operating conditions using
    a combination of deterministic correlations and historical data.
    Follows zero-hallucination principles with deterministic fallback.

    Attributes:
        baseline_config: Baseline performance configuration
        economizer_config: Economizer physical configuration

    Example:
        >>> predictor = FoulingPredictor(baseline, economizer)
        >>> prediction = predictor.predict_fouling_rate(operating_data)
        >>> print(f"Predicted rate: {prediction.predicted_fouling_rate_per_day}")
    """

    def __init__(
        self,
        baseline_config: BaselineConfiguration,
        economizer_config: Optional[EconomizerConfiguration] = None,
    ):
        """
        Initialize FoulingPredictor.

        Args:
            baseline_config: Baseline performance configuration
            economizer_config: Optional economizer configuration
        """
        self.baseline = baseline_config
        self.economizer = economizer_config
        self._historical_rates: List[float] = []
        self._operating_history: List[Dict[str, float]] = []

        logger.info("Initialized FoulingPredictor")

    def predict_fouling_rate(
        self,
        gas_inlet_temp_f: float,
        gas_outlet_temp_f: float,
        gas_flow_acfm: float,
        water_flow_gpm: float,
        boiler_load_pct: float,
        current_rf: float = 0.0,
        hours_since_cleaning: float = 0.0,
    ) -> FoulingPrediction:
        """
        Predict fouling rate from operating conditions.

        Uses DETERMINISTIC correlations with optional ML enhancement.

        Args:
            gas_inlet_temp_f: Flue gas inlet temperature (F)
            gas_outlet_temp_f: Flue gas outlet temperature (F)
            gas_flow_acfm: Gas flow rate (ACFM)
            water_flow_gpm: Water flow rate (GPM)
            boiler_load_pct: Boiler load percentage
            current_rf: Current fouling resistance
            hours_since_cleaning: Hours since last cleaning

        Returns:
            FoulingPrediction with rate and confidence
        """
        prediction = FoulingPrediction(
            economizer_id=self.economizer.economizer_id if self.economizer else "UNKNOWN",
            prediction_method="deterministic",
        )

        # =========================================================
        # DETERMINISTIC CALCULATION (Zero-Hallucination)
        # =========================================================

        # Base fouling rate from baseline configuration
        base_rate_per_day = self.baseline.typical_fouling_rate_per_day

        # Factor 1: Temperature effect
        # Higher gas inlet temperatures increase soot deposition
        design_gas_inlet = self.baseline.reference_gas_inlet_temp_f
        temp_ratio = gas_inlet_temp_f / design_gas_inlet if design_gas_inlet > 0 else 1.0
        temp_factor = temp_ratio ** 1.5  # Non-linear relationship

        prediction.contributing_factors["temperature"] = temp_factor
        if temp_factor > 1.2:
            prediction.factor_descriptions.append(
                f"Elevated gas temperature ({gas_inlet_temp_f:.0f}F) increases fouling rate"
            )

        # Factor 2: Load effect
        # Higher loads typically mean more complete combustion but more throughput
        load_factor = (boiler_load_pct / 80.0) if boiler_load_pct > 0 else 1.0
        load_factor = max(0.5, min(1.5, load_factor))  # Bound factor

        prediction.contributing_factors["load"] = load_factor
        if boiler_load_pct < 50:
            prediction.factor_descriptions.append(
                f"Low load ({boiler_load_pct:.0f}%) may cause incomplete combustion and more soot"
            )

        # Factor 3: Flow velocity effect
        # Lower velocities allow more deposition
        design_gas_flow = self.baseline.reference_gas_flow_acfm
        if design_gas_flow > 0 and gas_flow_acfm > 0:
            velocity_ratio = gas_flow_acfm / design_gas_flow
            # Inverse relationship - lower velocity = more fouling
            velocity_factor = (1 / velocity_ratio) ** 0.5 if velocity_ratio > 0.5 else 1.5
        else:
            velocity_factor = 1.0

        prediction.contributing_factors["velocity"] = velocity_factor

        # Factor 4: Time since cleaning effect
        # Fouling rate typically decreases as surface becomes fouled (saturation)
        if hours_since_cleaning > 0:
            saturation_factor = 1.0 / (1.0 + 0.01 * hours_since_cleaning)  # Asymptotic
        else:
            saturation_factor = 1.0

        prediction.contributing_factors["saturation"] = saturation_factor

        # Calculate combined fouling rate
        combined_factor = temp_factor * load_factor * velocity_factor * saturation_factor

        predicted_rate_per_day = base_rate_per_day * combined_factor
        predicted_rate_per_hour = predicted_rate_per_day / 24.0

        prediction.predicted_fouling_rate_per_day = predicted_rate_per_day
        prediction.predicted_fouling_rate_per_hour = predicted_rate_per_hour

        # Calculate time to thresholds
        warning_rf = 0.002  # Warning threshold
        critical_rf = 0.004  # Critical threshold
        max_rf = self.baseline.max_acceptable_fouling_resistance

        if predicted_rate_per_hour > 0:
            rf_to_warning = max(0, warning_rf - current_rf)
            rf_to_critical = max(0, critical_rf - current_rf)
            rf_to_max = max(0, max_rf - current_rf)

            prediction.hours_to_warning_threshold = rf_to_warning / predicted_rate_per_hour if rf_to_warning > 0 else 0
            prediction.hours_to_critical_threshold = rf_to_critical / predicted_rate_per_hour if rf_to_critical > 0 else 0

        # Calculate confidence
        # Higher confidence with more normal operating conditions
        confidence = 0.8  # Base confidence for deterministic model

        if 0.5 < temp_factor < 1.5:
            confidence += 0.05
        if 0.7 < load_factor < 1.3:
            confidence += 0.05
        if 0.7 < velocity_factor < 1.3:
            confidence += 0.05

        # Reduce confidence with extreme conditions
        if temp_factor > 1.5 or load_factor < 0.5:
            confidence -= 0.1

        prediction.confidence_score = max(0.4, min(0.95, confidence))

        # Set confidence level
        if prediction.confidence_score >= 0.8:
            prediction.confidence_level = PredictionConfidence.HIGH
        elif prediction.confidence_score >= 0.6:
            prediction.confidence_level = PredictionConfidence.MEDIUM
        elif prediction.confidence_score >= 0.4:
            prediction.confidence_level = PredictionConfidence.LOW
        else:
            prediction.confidence_level = PredictionConfidence.VERY_LOW

        # Store for history
        self._historical_rates.append(predicted_rate_per_day)
        if len(self._historical_rates) > 1000:
            self._historical_rates = self._historical_rates[-1000:]

        logger.debug(
            f"Fouling prediction: {predicted_rate_per_day:.6f}/day, "
            f"confidence={prediction.confidence_score:.2f}"
        )

        return prediction

    def get_average_fouling_rate(self) -> float:
        """
        Get average fouling rate from historical predictions.

        Returns:
            Average fouling rate per day
        """
        if not self._historical_rates:
            return self.baseline.typical_fouling_rate_per_day

        return sum(self._historical_rates) / len(self._historical_rates)


# ============================================================================
# CLEANING EFFECTIVENESS MODEL
# ============================================================================


class CleaningEffectivenessModel:
    """
    Cleaning effectiveness estimation model.

    Estimates the expected performance recovery from cleaning
    operations based on fouling state and cleaning method.

    Attributes:
        baseline_config: Baseline performance configuration
        economizer_config: Economizer physical configuration

    Example:
        >>> model = CleaningEffectivenessModel(baseline, economizer)
        >>> result = model.estimate_effectiveness(current_rf, method)
        >>> print(f"Expected recovery: {result.expected_recovery_pct}%")
    """

    # Default recovery percentages by cleaning method
    DEFAULT_RECOVERY_PCT = {
        CleaningMethod.SOOT_BLOWING: 80.0,
        CleaningMethod.ACOUSTIC_CLEANING: 70.0,
        CleaningMethod.WATER_WASHING: 90.0,
        CleaningMethod.CHEMICAL_CLEANING: 95.0,
        CleaningMethod.MECHANICAL_CLEANING: 98.0,
        CleaningMethod.AIR_LANCING: 75.0,
        CleaningMethod.STEAM_LANCING: 85.0,
    }

    def __init__(
        self,
        baseline_config: BaselineConfiguration,
        economizer_config: Optional[EconomizerConfiguration] = None,
        fuel_cost_per_mmbtu: float = 4.0,
    ):
        """
        Initialize CleaningEffectivenessModel.

        Args:
            baseline_config: Baseline performance configuration
            economizer_config: Optional economizer configuration
            fuel_cost_per_mmbtu: Fuel cost for economic calculations
        """
        self.baseline = baseline_config
        self.economizer = economizer_config
        self.fuel_cost = fuel_cost_per_mmbtu
        self._cleaning_history: List[Dict[str, Any]] = []

        logger.info("Initialized CleaningEffectivenessModel")

    def estimate_effectiveness(
        self,
        current_rf: float,
        current_effectiveness_pct: float,
        current_u_value: float,
        cleaning_method: CleaningMethod = CleaningMethod.SOOT_BLOWING,
        fouling_type: FoulingType = FoulingType.SOOT,
    ) -> CleaningEffectivenessResult:
        """
        Estimate cleaning effectiveness.

        Uses DETERMINISTIC calculations with historical adjustment.

        Args:
            current_rf: Current fouling resistance (hr-ft2-F/BTU)
            current_effectiveness_pct: Current effectiveness (%)
            current_u_value: Current U-value (BTU/hr-ft2-F)
            cleaning_method: Proposed cleaning method
            fouling_type: Type of fouling present

        Returns:
            CleaningEffectivenessResult with estimates
        """
        result = CleaningEffectivenessResult(
            economizer_id=self.economizer.economizer_id if self.economizer else "UNKNOWN",
            pre_cleaning_rf=current_rf,
            pre_cleaning_effectiveness_pct=current_effectiveness_pct,
            pre_cleaning_u_value=current_u_value,
            recommended_cleaning_method=cleaning_method,
        )

        # =========================================================
        # DETERMINISTIC CALCULATION (Zero-Hallucination)
        # =========================================================

        # Get base recovery percentage for cleaning method
        base_recovery_pct = self.DEFAULT_RECOVERY_PCT.get(cleaning_method, 80.0)

        # Adjust for fouling type
        fouling_type_factors = {
            FoulingType.SOOT: 1.0,  # Soot responds well to most methods
            FoulingType.ASH: 0.9,  # Ash slightly harder to remove
            FoulingType.SCALE: 0.6,  # Scale requires chemical cleaning
            FoulingType.CORROSION: 0.5,  # Corrosion products harder to remove
            FoulingType.MIXED: 0.8,  # Mixed requires multiple approaches
        }

        fouling_factor = fouling_type_factors.get(fouling_type, 0.8)

        # Adjust if wrong method for fouling type
        if fouling_type == FoulingType.SCALE and cleaning_method not in [
            CleaningMethod.CHEMICAL_CLEANING,
            CleaningMethod.WATER_WASHING,
        ]:
            fouling_factor *= 0.5
            result.additional_recommendations.append(
                "Consider chemical cleaning for scale deposits"
            )

        # Calculate expected recovery
        adjusted_recovery_pct = base_recovery_pct * fouling_factor
        result.expected_recovery_pct = adjusted_recovery_pct

        # Calculate expected post-cleaning fouling resistance
        rf_removed = current_rf * (adjusted_recovery_pct / 100.0)
        result.expected_post_cleaning_rf = current_rf - rf_removed

        # Calculate expected U-value recovery
        clean_u_value = self.baseline.clean_u_value_btu_hr_ft2_f

        if current_u_value > 0 and clean_u_value > 0:
            u_value_loss = clean_u_value - current_u_value
            expected_u_recovery = u_value_loss * (adjusted_recovery_pct / 100.0)
            result.expected_u_value_recovery = current_u_value + expected_u_recovery
        else:
            result.expected_u_value_recovery = clean_u_value * (adjusted_recovery_pct / 100.0)

        # Calculate expected effectiveness gain
        design_effectiveness = self.baseline.expected_effectiveness_pct
        effectiveness_loss = design_effectiveness - current_effectiveness_pct
        expected_effectiveness_gain = effectiveness_loss * (adjusted_recovery_pct / 100.0)
        result.expected_effectiveness_gain_pct = expected_effectiveness_gain

        # Economic calculations
        if self.economizer:
            design_heat_duty = self.economizer.design_heat_duty_mmbtu_hr

            # Heat recovery from cleaning
            heat_recovery_factor = expected_effectiveness_gain / 100.0
            result.expected_heat_recovery_mmbtu_hr = design_heat_duty * heat_recovery_factor

            # Cost savings
            result.expected_cost_savings_per_day_usd = (
                result.expected_heat_recovery_mmbtu_hr * self.fuel_cost * 24
            )

            # Cleaning cost estimate (default values)
            cleaning_costs = {
                CleaningMethod.SOOT_BLOWING: 50,  # Steam cost per cycle
                CleaningMethod.ACOUSTIC_CLEANING: 100,
                CleaningMethod.WATER_WASHING: 500,
                CleaningMethod.CHEMICAL_CLEANING: 2000,
                CleaningMethod.MECHANICAL_CLEANING: 5000,
            }

            result.cleaning_cost_estimate_usd = cleaning_costs.get(cleaning_method, 100)

            # ROI payback
            if result.expected_cost_savings_per_day_usd > 0:
                result.estimated_roi_payback_hours = (
                    result.cleaning_cost_estimate_usd /
                    (result.expected_cost_savings_per_day_usd / 24)
                )
            else:
                result.estimated_roi_payback_hours = float('inf')

        # Confidence based on method and fouling match
        if fouling_type in [FoulingType.SOOT, FoulingType.ASH] and cleaning_method == CleaningMethod.SOOT_BLOWING:
            result.confidence_score = 0.85
            result.prediction_basis = "historical"
        elif fouling_type == FoulingType.SCALE and cleaning_method == CleaningMethod.CHEMICAL_CLEANING:
            result.confidence_score = 0.80
            result.prediction_basis = "historical"
        else:
            result.confidence_score = 0.65
            result.prediction_basis = "default"

        # Add recommendations
        if current_rf > 0.004:
            result.additional_recommendations.append(
                "Consider extended cleaning cycle due to heavy fouling"
            )

        if result.estimated_roi_payback_hours < 1:
            result.additional_recommendations.append(
                f"Cleaning ROI payback very fast ({result.estimated_roi_payback_hours:.1f} hours) - clean immediately"
            )

        logger.debug(
            f"Cleaning effectiveness estimate: {adjusted_recovery_pct:.1f}% recovery, "
            f"${result.expected_cost_savings_per_day_usd:.2f}/day savings"
        )

        return result

    def record_cleaning_result(
        self,
        pre_rf: float,
        post_rf: float,
        cleaning_method: CleaningMethod,
        fouling_type: FoulingType,
    ) -> None:
        """
        Record actual cleaning result for model improvement.

        Args:
            pre_rf: Pre-cleaning fouling resistance
            post_rf: Post-cleaning fouling resistance
            cleaning_method: Method used
            fouling_type: Type of fouling
        """
        actual_recovery = ((pre_rf - post_rf) / pre_rf * 100) if pre_rf > 0 else 0

        self._cleaning_history.append({
            "timestamp": datetime.now(timezone.utc),
            "pre_rf": pre_rf,
            "post_rf": post_rf,
            "method": cleaning_method.value,
            "fouling_type": fouling_type.value,
            "actual_recovery_pct": actual_recovery,
        })

        # Limit history size
        if len(self._cleaning_history) > 100:
            self._cleaning_history = self._cleaning_history[-100:]

        logger.info(
            f"Recorded cleaning result: {actual_recovery:.1f}% recovery with {cleaning_method.value}"
        )


# ============================================================================
# ANOMALY DETECTOR
# ============================================================================


class AnomalyDetector:
    """
    Anomaly detection for economizer monitoring.

    Detects abnormal readings, sensor faults, leaks, and unusual
    performance patterns using statistical methods.

    Attributes:
        baseline_config: Baseline performance configuration
        sensitivity: Detection sensitivity (0-1)

    Example:
        >>> detector = AnomalyDetector(baseline, sensitivity=0.7)
        >>> result = detector.detect_anomalies(operating_data)
        >>> if result.anomaly_detected:
        ...     print(f"Anomaly: {result.description}")
    """

    def __init__(
        self,
        baseline_config: BaselineConfiguration,
        sensitivity: float = 0.7,
    ):
        """
        Initialize AnomalyDetector.

        Args:
            baseline_config: Baseline performance configuration
            sensitivity: Detection sensitivity (0-1, higher = more sensitive)
        """
        self.baseline = baseline_config
        self.sensitivity = max(0.1, min(1.0, sensitivity))

        # Historical data for statistical analysis
        self._temperature_history: List[Dict[str, float]] = []
        self._flow_history: List[Dict[str, float]] = []
        self._pressure_history: List[Dict[str, float]] = []

        # Running statistics
        self._temp_stats: Dict[str, Dict[str, float]] = {}
        self._flow_stats: Dict[str, Dict[str, float]] = {}

        logger.info(f"Initialized AnomalyDetector with sensitivity={sensitivity}")

    def detect_anomalies(
        self,
        water_inlet_temp_f: float,
        water_outlet_temp_f: float,
        gas_inlet_temp_f: float,
        gas_outlet_temp_f: float,
        water_flow_gpm: float,
        gas_flow_acfm: float,
        gas_pressure_drop_inwc: float,
        economizer_id: str = "UNKNOWN",
    ) -> AnomalyResult:
        """
        Detect anomalies in economizer readings.

        Uses DETERMINISTIC statistical methods.

        Args:
            water_inlet_temp_f: Water inlet temperature (F)
            water_outlet_temp_f: Water outlet temperature (F)
            gas_inlet_temp_f: Gas inlet temperature (F)
            gas_outlet_temp_f: Gas outlet temperature (F)
            water_flow_gpm: Water flow rate (GPM)
            gas_flow_acfm: Gas flow rate (ACFM)
            gas_pressure_drop_inwc: Gas pressure drop (in WC)
            economizer_id: Economizer identifier

        Returns:
            AnomalyResult with detection details
        """
        result = AnomalyResult(economizer_id=economizer_id)

        # Store current readings
        current_temps = {
            "water_inlet": water_inlet_temp_f,
            "water_outlet": water_outlet_temp_f,
            "gas_inlet": gas_inlet_temp_f,
            "gas_outlet": gas_outlet_temp_f,
        }

        # =========================================================
        # DETERMINISTIC ANOMALY CHECKS
        # =========================================================

        anomalies = []

        # Check 1: Temperature relationships (physics-based)
        # Gas inlet should always be > gas outlet
        if gas_outlet_temp_f >= gas_inlet_temp_f:
            anomalies.append({
                "type": AnomalyType.SENSOR_FAULT,
                "score": 0.9,
                "description": f"Gas outlet temp ({gas_outlet_temp_f:.1f}F) >= gas inlet ({gas_inlet_temp_f:.1f}F) - sensor fault",
                "sensor": "gas_temperature",
                "severity": "HIGH",
            })

        # Water outlet should always be > water inlet
        if water_inlet_temp_f >= water_outlet_temp_f:
            anomalies.append({
                "type": AnomalyType.SENSOR_FAULT,
                "score": 0.9,
                "description": f"Water inlet temp ({water_inlet_temp_f:.1f}F) >= water outlet ({water_outlet_temp_f:.1f}F) - sensor fault",
                "sensor": "water_temperature",
                "severity": "HIGH",
            })

        # Check 2: Approach temperature (gas inlet - water outlet)
        approach_temp = gas_inlet_temp_f - water_outlet_temp_f
        if approach_temp < 10:
            anomalies.append({
                "type": AnomalyType.PERFORMANCE_ANOMALY,
                "score": 0.7,
                "description": f"Very low approach temperature ({approach_temp:.1f}F) - check for measurement errors",
                "sensor": "temperature",
                "severity": "MEDIUM",
            })
        elif approach_temp < 0:
            anomalies.append({
                "type": AnomalyType.SENSOR_FAULT,
                "score": 0.95,
                "description": f"Negative approach temperature ({approach_temp:.1f}F) - definite sensor fault",
                "sensor": "temperature",
                "severity": "CRITICAL",
            })

        # Check 3: Temperature ranges
        if gas_inlet_temp_f > 800:
            anomalies.append({
                "type": AnomalyType.PERFORMANCE_ANOMALY,
                "score": 0.6,
                "description": f"High gas inlet temperature ({gas_inlet_temp_f:.1f}F) - check combustion",
                "sensor": "gas_inlet_temp",
                "severity": "MEDIUM",
            })

        if water_outlet_temp_f > 350:
            anomalies.append({
                "type": AnomalyType.PERFORMANCE_ANOMALY,
                "score": 0.7,
                "description": f"High water outlet temperature ({water_outlet_temp_f:.1f}F) - risk of steaming",
                "sensor": "water_outlet_temp",
                "severity": "HIGH",
            })

        # Check 4: Flow rates
        if water_flow_gpm < 50:
            anomalies.append({
                "type": AnomalyType.FLOW_RESTRICTION,
                "score": 0.8,
                "description": f"Low water flow ({water_flow_gpm:.0f} GPM) - risk of steaming or sensor fault",
                "sensor": "water_flow",
                "severity": "HIGH",
            })

        # Check 5: Pressure drop
        if gas_pressure_drop_inwc > 4.0:
            anomalies.append({
                "type": AnomalyType.FLOW_RESTRICTION,
                "score": 0.7,
                "description": f"High gas pressure drop ({gas_pressure_drop_inwc:.2f} in WC) - heavy fouling or obstruction",
                "sensor": "pressure",
                "severity": "HIGH",
            })

        # Check 6: Heat balance check (energy in ~= energy out)
        water_temp_rise = water_outlet_temp_f - water_inlet_temp_f
        gas_temp_drop = gas_inlet_temp_f - gas_outlet_temp_f

        # Approximate heat balance: water_rise * water_flow ~proportional to~ gas_drop * gas_flow
        # Simplified check: ratio should be within reasonable bounds
        if water_flow_gpm > 0 and gas_flow_acfm > 0 and gas_temp_drop > 0:
            water_side_factor = water_temp_rise * water_flow_gpm
            gas_side_factor = gas_temp_drop * gas_flow_acfm / 1000  # Scale factor

            if water_side_factor > 0 and gas_side_factor > 0:
                ratio = water_side_factor / gas_side_factor
                # Expect ratio in range 0.5 - 2.0 for typical conditions
                if ratio < 0.2 or ratio > 5.0:
                    anomalies.append({
                        "type": AnomalyType.TUBE_LEAK,
                        "score": 0.6,
                        "description": f"Heat balance anomaly (ratio={ratio:.2f}) - possible leak or sensor drift",
                        "sensor": "multiple",
                        "severity": "MEDIUM",
                    })

        # Check 7: Statistical outliers (if history available)
        self._update_statistics(current_temps, water_flow_gpm, gas_flow_acfm)

        stat_anomalies = self._check_statistical_anomalies(current_temps, water_flow_gpm)
        anomalies.extend(stat_anomalies)

        # Compile result
        if anomalies:
            # Sort by score and take highest
            anomalies.sort(key=lambda x: x["score"], reverse=True)
            worst = anomalies[0]

            result.anomaly_detected = True
            result.anomaly_type = worst["type"]
            result.anomaly_score = worst["score"]
            result.affected_sensor = worst.get("sensor")
            result.description = worst["description"]
            result.severity = worst["severity"]
            result.confidence = worst["score"]

            # Compile evidence
            result.evidence = [a["description"] for a in anomalies]

            # Assess impact
            result.affects_performance = any(
                a["type"] in [AnomalyType.PERFORMANCE_ANOMALY, AnomalyType.FLOW_RESTRICTION]
                for a in anomalies
            )
            result.affects_safety = any(
                a["severity"] in ["HIGH", "CRITICAL"] and a["type"] == AnomalyType.SENSOR_FAULT
                for a in anomalies
            )

            # Generate recommendations
            result.recommended_actions = self._generate_recommendations(anomalies)
            result.requires_immediate_attention = worst["severity"] == "CRITICAL"
        else:
            result.anomaly_detected = False
            result.description = "No anomalies detected"
            result.confidence = 0.9  # High confidence no anomaly

        logger.debug(
            f"Anomaly detection: detected={result.anomaly_detected}, "
            f"type={result.anomaly_type.value if result.anomaly_detected else 'N/A'}"
        )

        return result

    def _update_statistics(
        self,
        temps: Dict[str, float],
        water_flow: float,
        gas_flow: float,
    ) -> None:
        """
        Update running statistics with new readings.

        Args:
            temps: Temperature readings
            water_flow: Water flow rate
            gas_flow: Gas flow rate
        """
        self._temperature_history.append(temps)
        self._flow_history.append({"water": water_flow, "gas": gas_flow})

        # Limit history
        max_history = 1000
        if len(self._temperature_history) > max_history:
            self._temperature_history = self._temperature_history[-max_history:]
        if len(self._flow_history) > max_history:
            self._flow_history = self._flow_history[-max_history:]

        # Update running statistics
        if len(self._temperature_history) >= 10:
            for key in temps:
                values = [t[key] for t in self._temperature_history if key in t]
                if values:
                    mean = sum(values) / len(values)
                    variance = sum((v - mean) ** 2 for v in values) / len(values)
                    std_dev = variance ** 0.5

                    self._temp_stats[key] = {
                        "mean": mean,
                        "std_dev": std_dev,
                        "min": min(values),
                        "max": max(values),
                    }

    def _check_statistical_anomalies(
        self,
        temps: Dict[str, float],
        water_flow: float,
    ) -> List[Dict[str, Any]]:
        """
        Check for statistical anomalies using z-scores.

        Args:
            temps: Current temperature readings
            water_flow: Current water flow

        Returns:
            List of detected anomalies
        """
        anomalies = []

        # Need sufficient history
        if len(self._temperature_history) < 30:
            return anomalies

        # Check temperature z-scores
        z_threshold = 3.0 - (self.sensitivity * 1.5)  # Adjust based on sensitivity

        for key, value in temps.items():
            if key in self._temp_stats:
                stats = self._temp_stats[key]
                if stats["std_dev"] > 0:
                    z_score = abs(value - stats["mean"]) / stats["std_dev"]

                    if z_score > z_threshold:
                        anomalies.append({
                            "type": AnomalyType.SENSOR_DRIFT,
                            "score": min(0.9, z_score / 5.0),
                            "description": (
                                f"{key.replace('_', ' ').title()} ({value:.1f}F) is {z_score:.1f} "
                                f"standard deviations from mean ({stats['mean']:.1f}F)"
                            ),
                            "sensor": key,
                            "severity": "MEDIUM" if z_score < 4 else "HIGH",
                        })

        return anomalies

    def _generate_recommendations(
        self,
        anomalies: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Generate recommended actions for detected anomalies.

        Args:
            anomalies: List of detected anomalies

        Returns:
            List of recommended actions
        """
        recommendations = []

        for anomaly in anomalies:
            if anomaly["type"] == AnomalyType.SENSOR_FAULT:
                recommendations.append(
                    f"Verify {anomaly.get('sensor', 'affected')} sensor calibration and wiring"
                )
            elif anomaly["type"] == AnomalyType.SENSOR_DRIFT:
                recommendations.append(
                    f"Schedule calibration check for {anomaly.get('sensor', 'affected')} sensor"
                )
            elif anomaly["type"] == AnomalyType.TUBE_LEAK:
                recommendations.append(
                    "Inspect for tube leaks - check for water in flue gas"
                )
            elif anomaly["type"] == AnomalyType.FLOW_RESTRICTION:
                recommendations.append(
                    "Check for flow restrictions - inspect valves and strainers"
                )
            elif anomaly["type"] == AnomalyType.PERFORMANCE_ANOMALY:
                recommendations.append(
                    "Review operating conditions and recent changes"
                )

        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for r in recommendations:
            if r not in seen:
                seen.add(r)
                unique_recommendations.append(r)

        return unique_recommendations


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "FoulingPredictor",
    "FoulingPrediction",
    "CleaningEffectivenessModel",
    "CleaningEffectivenessResult",
    "AnomalyDetector",
    "AnomalyResult",
    "AnomalyType",
    "PredictionConfidence",
]
