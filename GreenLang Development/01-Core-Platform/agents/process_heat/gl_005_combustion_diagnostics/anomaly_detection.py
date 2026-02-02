# -*- coding: utf-8 -*-
"""
GL-005 Anomaly Detection Module
===============================

This module implements combustion anomaly detection using a combination of:
1. Statistical Process Control (SPC) - Western Electric rules
2. Machine Learning (ML) - Isolation Forest pattern recognition
3. Rule-Based Detection - Domain-specific combustion rules

The module is designed for DIAGNOSTICS ONLY - it identifies anomalies and
generates alerts but does NOT execute any control actions.

ZERO-HALLUCINATION GUARANTEE:
    All detection algorithms are deterministic.
    SPC uses documented statistical methods.
    ML uses trained models with fixed parameters.
    Full provenance tracking for all detections.

Author: GreenLang Process Heat Team
Version: 1.0.0
Status: Production Ready
"""

import hashlib
import json
import logging
import math
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

from greenlang.agents.process_heat.gl_005_combustion_diagnostics.config import (
    AnomalyDetectionConfig,
    AnomalyType,
    MLAnomalyConfig,
    SPCConfig,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.schemas import (
    AnalysisStatus,
    AnomalyDetectionResult,
    AnomalyEvent,
    AnomalySeverity,
    FlueGasReading,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SPCStatistics:
    """Statistical Process Control statistics for a parameter."""

    parameter: str
    mean: float
    std_dev: float
    ucl: float  # Upper Control Limit
    lcl: float  # Lower Control Limit
    uwl: float  # Upper Warning Limit
    lwl: float  # Lower Warning Limit
    sample_count: int
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SPCPoint:
    """Single data point for SPC analysis."""

    timestamp: datetime
    parameter: str
    value: float
    zscore: float = 0.0
    in_control: bool = True
    violations: List[str] = field(default_factory=list)


# =============================================================================
# STATISTICAL PROCESS CONTROL
# =============================================================================

class SPCAnalyzer:
    """
    Statistical Process Control (SPC) Analyzer.

    Implements Western Electric rules for detecting out-of-control conditions:
    - Rule 1: Point beyond 3 sigma
    - Rule 2: 2 of 3 points beyond 2 sigma on same side
    - Rule 3: 4 of 5 points beyond 1 sigma on same side
    - Rule 4: 7+ consecutive points on same side of centerline
    - Rule 5: 6+ consecutive points trending in same direction

    This is a DETERMINISTIC algorithm using documented statistical methods.
    """

    def __init__(self, config: SPCConfig) -> None:
        """
        Initialize SPC analyzer.

        Args:
            config: SPC configuration
        """
        self.config = config
        self._statistics: Dict[str, SPCStatistics] = {}
        self._history: Dict[str, Deque[SPCPoint]] = {}
        self._baseline_data: Dict[str, List[float]] = {}

        logger.info(
            f"SPC Analyzer initialized (warning={config.sigma_warning}sigma, "
            f"control={config.sigma_control}sigma)"
        )

    def update_baseline(self, parameter: str, values: List[float]) -> SPCStatistics:
        """
        Update baseline statistics for a parameter.

        Calculates mean, standard deviation, and control limits from
        baseline data.

        Args:
            parameter: Parameter name
            values: List of baseline values

        Returns:
            Updated SPC statistics
        """
        if len(values) < 2:
            raise ValueError(f"Need at least 2 values for baseline, got {len(values)}")

        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 0.001

        stats = SPCStatistics(
            parameter=parameter,
            mean=mean,
            std_dev=std_dev,
            ucl=mean + self.config.sigma_control * std_dev,
            lcl=mean - self.config.sigma_control * std_dev,
            uwl=mean + self.config.sigma_warning * std_dev,
            lwl=mean - self.config.sigma_warning * std_dev,
            sample_count=n,
        )

        self._statistics[parameter] = stats
        self._baseline_data[parameter] = values.copy()

        # Initialize history deque
        if parameter not in self._history:
            self._history[parameter] = deque(maxlen=self.config.moving_window_size)

        logger.debug(
            f"SPC baseline updated for {parameter}: "
            f"mean={mean:.3f}, std={std_dev:.3f}, UCL={stats.ucl:.3f}, LCL={stats.lcl:.3f}"
        )

        return stats

    def analyze_point(
        self,
        parameter: str,
        value: float,
        timestamp: datetime,
    ) -> Tuple[SPCPoint, List[str]]:
        """
        Analyze a single data point against SPC rules.

        Args:
            parameter: Parameter name
            value: Current value
            timestamp: Value timestamp

        Returns:
            Tuple of (SPCPoint, list of violations)
        """
        violations = []

        # Get or create statistics
        if parameter not in self._statistics:
            # No baseline - can't perform SPC
            point = SPCPoint(
                timestamp=timestamp,
                parameter=parameter,
                value=value,
                in_control=True,
            )
            return point, []

        stats = self._statistics[parameter]

        # Calculate z-score
        zscore = (value - stats.mean) / stats.std_dev if stats.std_dev > 0 else 0

        # Initialize point
        point = SPCPoint(
            timestamp=timestamp,
            parameter=parameter,
            value=value,
            zscore=zscore,
            in_control=True,
        )

        # Add to history
        if parameter not in self._history:
            self._history[parameter] = deque(maxlen=self.config.moving_window_size)
        self._history[parameter].append(point)

        # Apply Western Electric rules
        violations = self._check_western_electric_rules(parameter, point, stats)

        if violations:
            point.in_control = False
            point.violations = violations

        return point, violations

    def _check_western_electric_rules(
        self,
        parameter: str,
        current_point: SPCPoint,
        stats: SPCStatistics,
    ) -> List[str]:
        """
        Check Western Electric rules for SPC violation detection.

        Args:
            parameter: Parameter name
            current_point: Current data point
            stats: SPC statistics

        Returns:
            List of rule violations
        """
        violations = []
        history = list(self._history[parameter])

        # Rule 1: Point beyond 3 sigma (control limits)
        if abs(current_point.zscore) > self.config.sigma_control:
            violations.append(
                f"Rule 1: Point beyond {self.config.sigma_control}sigma "
                f"(value={current_point.value:.3f}, zscore={current_point.zscore:.2f})"
            )

        if not self.config.enable_run_rules:
            return violations

        # Need enough history for run rules
        if len(history) < 2:
            return violations

        # Rule 2: 2 of 3 points beyond 2 sigma on same side
        if len(history) >= 3:
            recent_3 = history[-3:]
            above_2sig = sum(1 for p in recent_3 if p.zscore > self.config.sigma_warning)
            below_2sig = sum(1 for p in recent_3 if p.zscore < -self.config.sigma_warning)
            if above_2sig >= 2 or below_2sig >= 2:
                violations.append(
                    f"Rule 2: 2 of 3 points beyond {self.config.sigma_warning}sigma on same side"
                )

        # Rule 3: 4 of 5 points beyond 1 sigma on same side
        if len(history) >= 5:
            recent_5 = history[-5:]
            above_1sig = sum(1 for p in recent_5 if p.zscore > 1.0)
            below_1sig = sum(1 for p in recent_5 if p.zscore < -1.0)
            if above_1sig >= 4 or below_1sig >= 4:
                violations.append("Rule 3: 4 of 5 points beyond 1sigma on same side")

        # Rule 4: 7+ consecutive points on same side of centerline
        consecutive_same_side = self.config.consecutive_one_side
        if len(history) >= consecutive_same_side:
            recent = history[-consecutive_same_side:]
            all_above = all(p.zscore > 0 for p in recent)
            all_below = all(p.zscore < 0 for p in recent)
            if all_above or all_below:
                violations.append(
                    f"Rule 4: {consecutive_same_side} consecutive points on same side of centerline"
                )

        # Rule 5: 6+ consecutive points trending
        consecutive_trending = self.config.consecutive_trending
        if len(history) >= consecutive_trending:
            recent = history[-consecutive_trending:]
            values = [p.value for p in recent]

            # Check increasing trend
            increasing = all(values[i] < values[i + 1] for i in range(len(values) - 1))
            # Check decreasing trend
            decreasing = all(values[i] > values[i + 1] for i in range(len(values) - 1))

            if increasing or decreasing:
                direction = "increasing" if increasing else "decreasing"
                violations.append(
                    f"Rule 5: {consecutive_trending} consecutive points {direction}"
                )

        return violations

    def get_statistics(self, parameter: str) -> Optional[SPCStatistics]:
        """Get current SPC statistics for parameter."""
        return self._statistics.get(parameter)

    def reset_history(self, parameter: Optional[str] = None) -> None:
        """Reset history for parameter(s)."""
        if parameter:
            if parameter in self._history:
                self._history[parameter].clear()
        else:
            for key in self._history:
                self._history[key].clear()


# =============================================================================
# MACHINE LEARNING ANOMALY DETECTION
# =============================================================================

class MLAnomalyDetector:
    """
    Machine Learning Anomaly Detector using Isolation Forest.

    Isolation Forest is an unsupervised learning algorithm that isolates
    anomalies by randomly selecting features and split values. Anomalies
    are isolated closer to the root of the tree, requiring fewer splits.

    This implementation uses a simplified version suitable for real-time
    combustion monitoring without external ML dependencies.

    DETERMINISTIC: Uses fixed random seed for reproducibility.
    """

    def __init__(self, config: MLAnomalyConfig) -> None:
        """
        Initialize ML anomaly detector.

        Args:
            config: ML anomaly detection configuration
        """
        self.config = config
        self._feature_names: List[str] = []
        self._feature_ranges: Dict[str, Tuple[float, float]] = {}
        self._is_fitted = False
        self._baseline_data: List[Dict[str, float]] = []

        # Simplified isolation forest state
        self._n_trees = config.n_estimators
        self._trees: List[Dict] = []
        self._random_seed = 42

        logger.info(
            f"ML Anomaly Detector initialized "
            f"(estimators={config.n_estimators}, contamination={config.contamination})"
        )

    def fit(self, training_data: List[Dict[str, float]]) -> None:
        """
        Fit the anomaly detector on training data.

        Args:
            training_data: List of feature dictionaries
        """
        if not training_data:
            raise ValueError("Training data cannot be empty")

        self._baseline_data = training_data.copy()
        self._feature_names = list(training_data[0].keys())

        # Calculate feature ranges for normalization
        for feature in self._feature_names:
            values = [d[feature] for d in training_data]
            self._feature_ranges[feature] = (min(values), max(values))

        # Build simplified isolation forest
        self._build_forest(training_data)
        self._is_fitted = True

        logger.info(
            f"ML detector fitted on {len(training_data)} samples "
            f"with {len(self._feature_names)} features"
        )

    def _build_forest(self, data: List[Dict[str, float]]) -> None:
        """Build simplified isolation forest."""
        import random

        random.seed(self._random_seed)
        self._trees = []

        for tree_idx in range(self._n_trees):
            # Subsample data
            sample_size = min(256, len(data))
            sample = random.sample(data, sample_size)

            # Build tree (simplified - store split points)
            tree = self._build_tree(sample, depth=0, max_depth=int(math.log2(sample_size)))
            self._trees.append(tree)

    def _build_tree(
        self,
        data: List[Dict[str, float]],
        depth: int,
        max_depth: int,
    ) -> Dict:
        """Build a single isolation tree."""
        import random

        if depth >= max_depth or len(data) <= 1:
            return {"type": "leaf", "size": len(data)}

        # Random feature and split point
        feature = random.choice(self._feature_names)
        values = [d[feature] for d in data]
        min_val, max_val = min(values), max(values)

        if min_val == max_val:
            return {"type": "leaf", "size": len(data)}

        split_value = random.uniform(min_val, max_val)

        left_data = [d for d in data if d[feature] < split_value]
        right_data = [d for d in data if d[feature] >= split_value]

        return {
            "type": "split",
            "feature": feature,
            "split_value": split_value,
            "left": self._build_tree(left_data, depth + 1, max_depth),
            "right": self._build_tree(right_data, depth + 1, max_depth),
        }

    def predict_anomaly_score(self, features: Dict[str, float]) -> float:
        """
        Calculate anomaly score for a data point.

        Args:
            features: Feature dictionary

        Returns:
            Anomaly score (0-1, higher = more anomalous)
        """
        if not self._is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        # Calculate average path length across all trees
        path_lengths = []
        for tree in self._trees:
            path_length = self._get_path_length(tree, features, 0)
            path_lengths.append(path_length)

        avg_path_length = sum(path_lengths) / len(path_lengths)

        # Calculate anomaly score using isolation forest formula
        # s(x, n) = 2^(-E(h(x))/c(n))
        # where c(n) is average path length of unsuccessful search in BST
        n = len(self._baseline_data)
        c_n = self._c_factor(n)

        if c_n == 0:
            return 0.5

        anomaly_score = 2 ** (-avg_path_length / c_n)
        return min(1.0, max(0.0, anomaly_score))

    def _get_path_length(self, node: Dict, features: Dict[str, float], depth: int) -> float:
        """Calculate path length to isolate a point."""
        if node["type"] == "leaf":
            # Add adjustment for external nodes
            return depth + self._c_factor(node["size"])

        feature = node["feature"]
        value = features.get(feature, 0)

        if value < node["split_value"]:
            return self._get_path_length(node["left"], features, depth + 1)
        else:
            return self._get_path_length(node["right"], features, depth + 1)

    def _c_factor(self, n: int) -> float:
        """Calculate average path length adjustment factor."""
        if n <= 1:
            return 0
        elif n == 2:
            return 1
        else:
            # Harmonic number approximation
            return 2 * (math.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n)

    def detect(self, features: Dict[str, float]) -> Tuple[bool, float, Dict[str, float]]:
        """
        Detect if a data point is anomalous.

        Args:
            features: Feature dictionary

        Returns:
            Tuple of (is_anomaly, anomaly_score, feature_contributions)
        """
        anomaly_score = self.predict_anomaly_score(features)
        is_anomaly = anomaly_score >= self.config.anomaly_threshold

        # Calculate feature contributions (simplified deviation-based)
        contributions = {}
        if self.config.track_feature_importance:
            for feature in self._feature_names:
                if feature in features and feature in self._feature_ranges:
                    min_val, max_val = self._feature_ranges[feature]
                    if max_val > min_val:
                        # Normalized deviation from range center
                        center = (max_val + min_val) / 2
                        deviation = abs(features[feature] - center) / (max_val - min_val)
                        contributions[feature] = round(deviation, 4)

        return is_anomaly, anomaly_score, contributions


# =============================================================================
# RULE-BASED DETECTION
# =============================================================================

class RuleBasedDetector:
    """
    Rule-based combustion anomaly detector.

    Implements domain-specific rules for combustion anomaly detection based on
    engineering knowledge and safety requirements.

    These rules are DETERMINISTIC and based on documented combustion science.
    """

    def __init__(self) -> None:
        """Initialize rule-based detector."""
        self._rules: List[Dict[str, Any]] = self._define_rules()
        logger.info(f"Rule-based detector initialized with {len(self._rules)} rules")

    def _define_rules(self) -> List[Dict[str, Any]]:
        """Define combustion anomaly detection rules."""
        return [
            {
                "id": "R001",
                "name": "Low Oxygen - Incomplete Combustion Risk",
                "type": AnomalyType.LOW_OXYGEN,
                "condition": lambda fg: fg.oxygen_pct < 1.5,
                "severity": AnomalySeverity.CRITICAL,
                "causes": ["Insufficient combustion air", "Air damper malfunction", "Fan failure"],
                "actions": ["Check combustion air supply", "Verify damper position", "Inspect fan operation"],
            },
            {
                "id": "R002",
                "name": "Excess Oxygen - Efficiency Loss",
                "type": AnomalyType.EXCESS_OXYGEN,
                "condition": lambda fg: fg.oxygen_pct > 8.0,
                "severity": AnomalySeverity.WARNING,
                "causes": ["Air-fuel ratio too lean", "Air leakage", "Control system issue"],
                "actions": ["Adjust air-fuel ratio", "Check for air leaks", "Verify control settings"],
            },
            {
                "id": "R003",
                "name": "High CO - Incomplete Combustion",
                "type": AnomalyType.HIGH_CO,
                "condition": lambda fg: fg.co_ppm > 400,
                "severity": AnomalySeverity.ALARM,
                "causes": ["Insufficient air", "Burner fouling", "Fuel quality issue", "Flame impingement"],
                "actions": ["Increase combustion air", "Clean burner", "Check fuel quality", "Inspect flame pattern"],
            },
            {
                "id": "R004",
                "name": "Critical CO Level",
                "type": AnomalyType.HIGH_CO,
                "condition": lambda fg: fg.co_ppm > 1000,
                "severity": AnomalySeverity.CRITICAL,
                "causes": ["Severe incomplete combustion", "Major burner issue", "Safety hazard"],
                "actions": ["Immediate investigation required", "Consider reducing load", "Check safety interlocks"],
            },
            {
                "id": "R005",
                "name": "High NOx Emissions",
                "type": AnomalyType.HIGH_NOX,
                "condition": lambda fg: fg.nox_ppm > 200,
                "severity": AnomalySeverity.WARNING,
                "causes": ["High flame temperature", "Excess air", "Hot spots in combustion zone"],
                "actions": ["Optimize air-fuel ratio", "Check for FGR operation", "Adjust burner settings"],
            },
            {
                "id": "R006",
                "name": "High Combustibles",
                "type": AnomalyType.HIGH_COMBUSTIBLES,
                "condition": lambda fg: (fg.combustibles_pct or 0) > 0.5,
                "severity": AnomalySeverity.ALARM,
                "causes": ["Incomplete combustion", "Burner issue", "Fuel atomization problem"],
                "actions": ["Increase excess air", "Check burner condition", "Verify fuel atomization"],
            },
            {
                "id": "R007",
                "name": "Air-Fuel Imbalance Detected",
                "type": AnomalyType.AIR_FUEL_IMBALANCE,
                "condition": lambda fg: self._check_air_fuel_imbalance(fg),
                "severity": AnomalySeverity.WARNING,
                "causes": ["Control system drift", "Sensor calibration issue", "Fuel property change"],
                "actions": ["Recalibrate sensors", "Check control loop", "Verify fuel composition"],
            },
            {
                "id": "R008",
                "name": "Stack Temperature High - Possible Fouling",
                "type": AnomalyType.FOULING_DETECTED,
                "condition": lambda fg: fg.flue_gas_temp_c > 300,
                "severity": AnomalySeverity.INFO,
                "causes": ["Heat transfer surface fouling", "Scale buildup", "Soot accumulation"],
                "actions": ["Schedule inspection", "Monitor efficiency trend", "Plan cleaning"],
            },
        ]

    def _check_air_fuel_imbalance(self, flue_gas: FlueGasReading) -> bool:
        """Check for air-fuel ratio imbalance."""
        # Expected CO2 for natural gas at given O2
        expected_co2 = 11.8 * (20.95 - flue_gas.oxygen_pct) / 20.95
        if expected_co2 <= 0:
            return False
        deviation = abs(flue_gas.co2_pct - expected_co2) / expected_co2
        return deviation > 0.15  # 15% deviation threshold

    def detect(self, flue_gas: FlueGasReading) -> List[Dict[str, Any]]:
        """
        Apply all rules to detect anomalies.

        Args:
            flue_gas: Flue gas reading

        Returns:
            List of triggered rules
        """
        triggered = []

        for rule in self._rules:
            try:
                if rule["condition"](flue_gas):
                    triggered.append({
                        "rule_id": rule["id"],
                        "name": rule["name"],
                        "type": rule["type"],
                        "severity": rule["severity"],
                        "causes": rule["causes"],
                        "actions": rule["actions"],
                    })
            except Exception as e:
                logger.warning(f"Rule {rule['id']} evaluation failed: {e}")

        return triggered


# =============================================================================
# INTEGRATED ANOMALY DETECTOR
# =============================================================================

class CombustionAnomalyDetector:
    """
    Integrated Combustion Anomaly Detector.

    Combines SPC, ML, and rule-based detection methods for comprehensive
    anomaly detection. Each method provides different perspectives:

    - SPC: Statistical deviations from baseline
    - ML: Pattern-based anomaly recognition
    - Rules: Domain-specific combustion knowledge

    This is a DIAGNOSTIC-ONLY component. It identifies anomalies but does
    NOT execute any control actions.

    Example:
        >>> config = AnomalyDetectionConfig()
        >>> detector = CombustionAnomalyDetector(config)
        >>> detector.initialize_baseline(historical_readings)
        >>> result = detector.detect(current_reading)
        >>> if result.anomaly_detected:
        ...     print(f"Found {result.total_anomalies} anomalies")
    """

    def __init__(self, config: AnomalyDetectionConfig) -> None:
        """
        Initialize integrated anomaly detector.

        Args:
            config: Anomaly detection configuration
        """
        self.config = config
        self.spc = SPCAnalyzer(config.spc)
        self.ml = MLAnomalyDetector(config.ml) if config.ml.enabled else None
        self.rules = RuleBasedDetector()

        self._last_alerts: Dict[str, datetime] = {}  # For cooldown tracking
        self._audit_trail: List[Dict[str, Any]] = []

        logger.info(
            f"Combustion Anomaly Detector initialized "
            f"(modes: {config.detection_modes})"
        )

    def initialize_baseline(self, readings: List[FlueGasReading]) -> None:
        """
        Initialize baseline statistics from historical data.

        Args:
            readings: Historical flue gas readings
        """
        if not readings:
            logger.warning("No baseline data provided")
            return

        # Extract parameter lists
        o2_values = [r.oxygen_pct for r in readings]
        co2_values = [r.co2_pct for r in readings]
        co_values = [r.co_ppm for r in readings]
        nox_values = [r.nox_ppm for r in readings]
        temp_values = [r.flue_gas_temp_c for r in readings]

        # Initialize SPC baselines
        self.spc.update_baseline("oxygen", o2_values)
        self.spc.update_baseline("co2", co2_values)
        self.spc.update_baseline("co", co_values)
        self.spc.update_baseline("nox", nox_values)
        self.spc.update_baseline("stack_temp", temp_values)

        # Initialize ML baseline
        if self.ml:
            training_data = [
                {
                    "oxygen": r.oxygen_pct,
                    "co2": r.co2_pct,
                    "co": r.co_ppm,
                    "nox": r.nox_ppm,
                    "stack_temp": r.flue_gas_temp_c,
                }
                for r in readings
            ]
            self.ml.fit(training_data)

        logger.info(f"Baseline initialized from {len(readings)} readings")

    def detect(self, reading: FlueGasReading) -> AnomalyDetectionResult:
        """
        Perform anomaly detection on a flue gas reading.

        Applies all enabled detection methods and aggregates results.

        Args:
            reading: Current flue gas reading

        Returns:
            Comprehensive anomaly detection result
        """
        start_time = datetime.now(timezone.utc)
        self._audit_trail = []

        anomalies: List[AnomalyEvent] = []
        spc_violations: List[str] = []
        spc_in_control = True
        ml_health_score = None

        # 1. SPC Analysis
        if "spc" in self.config.detection_modes:
            spc_anomalies, violations, in_control = self._run_spc_detection(reading)
            anomalies.extend(spc_anomalies)
            spc_violations = violations
            spc_in_control = in_control

        # 2. ML Analysis
        if "ml" in self.config.detection_modes and self.ml and self.ml._is_fitted:
            ml_anomalies, health_score = self._run_ml_detection(reading)
            anomalies.extend(ml_anomalies)
            ml_health_score = health_score

        # 3. Rule-Based Analysis
        if "rule_based" in self.config.detection_modes:
            rule_anomalies = self._run_rule_detection(reading)
            anomalies.extend(rule_anomalies)

        # Apply alert cooldown
        anomalies = self._apply_cooldown(anomalies)

        # Calculate severity counts
        critical_count = sum(1 for a in anomalies if a.severity == AnomalySeverity.CRITICAL)
        alarm_count = sum(1 for a in anomalies if a.severity == AnomalySeverity.ALARM)
        warning_count = sum(1 for a in anomalies if a.severity == AnomalySeverity.WARNING)
        info_count = sum(1 for a in anomalies if a.severity == AnomalySeverity.INFO)

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(reading, anomalies)

        result = AnomalyDetectionResult(
            status=AnalysisStatus.SUCCESS,
            anomaly_detected=len(anomalies) > 0,
            total_anomalies=len(anomalies),
            anomalies=anomalies,
            critical_count=critical_count,
            alarm_count=alarm_count,
            warning_count=warning_count,
            info_count=info_count,
            spc_in_control=spc_in_control,
            spc_violations=spc_violations,
            ml_health_score=ml_health_score,
            analysis_timestamp=start_time,
            samples_analyzed=1,
            provenance_hash=provenance_hash,
        )

        if anomalies:
            logger.info(
                f"Anomaly detection: {len(anomalies)} anomalies found "
                f"(critical={critical_count}, alarm={alarm_count}, warning={warning_count})"
            )

        return result

    def _run_spc_detection(
        self,
        reading: FlueGasReading,
    ) -> Tuple[List[AnomalyEvent], List[str], bool]:
        """Run SPC-based anomaly detection."""
        anomalies = []
        all_violations = []
        all_in_control = True

        parameters = [
            ("oxygen", reading.oxygen_pct, AnomalyType.EXCESS_OXYGEN),
            ("co2", reading.co2_pct, AnomalyType.AIR_FUEL_IMBALANCE),
            ("co", reading.co_ppm, AnomalyType.HIGH_CO),
            ("nox", reading.nox_ppm, AnomalyType.HIGH_NOX),
            ("stack_temp", reading.flue_gas_temp_c, AnomalyType.HEAT_TRANSFER_DEGRADATION),
        ]

        for param_name, value, anomaly_type in parameters:
            point, violations = self.spc.analyze_point(
                param_name, value, reading.timestamp
            )

            if violations:
                all_in_control = False
                all_violations.extend(violations)

                stats = self.spc.get_statistics(param_name)
                expected = stats.mean if stats else value

                anomaly = AnomalyEvent(
                    anomaly_id=str(uuid.uuid4()),
                    timestamp=reading.timestamp,
                    anomaly_type=anomaly_type,
                    severity=self._spc_severity(point.zscore),
                    detection_method="spc",
                    confidence=min(0.99, abs(point.zscore) / 5),
                    observed_value=value,
                    expected_value=expected,
                    deviation_pct=((value - expected) / expected * 100) if expected != 0 else 0,
                    affected_parameter=param_name,
                    potential_causes=[f"SPC violation: {v}" for v in violations],
                    recommended_actions=["Review SPC trend", "Investigate root cause"],
                    control_limit_upper=stats.ucl if stats else None,
                    control_limit_lower=stats.lcl if stats else None,
                    sigma_deviation=point.zscore,
                )
                anomalies.append(anomaly)

        self._add_audit_entry("spc_detection", {
            "parameters_checked": len(parameters),
            "violations_found": len(all_violations),
        })

        return anomalies, all_violations, all_in_control

    def _run_ml_detection(
        self,
        reading: FlueGasReading,
    ) -> Tuple[List[AnomalyEvent], float]:
        """Run ML-based anomaly detection."""
        anomalies = []

        features = {
            "oxygen": reading.oxygen_pct,
            "co2": reading.co2_pct,
            "co": reading.co_ppm,
            "nox": reading.nox_ppm,
            "stack_temp": reading.flue_gas_temp_c,
        }

        is_anomaly, score, contributions = self.ml.detect(features)
        health_score = 1.0 - score  # Invert: high anomaly score = low health

        if is_anomaly:
            # Find most contributing feature
            top_feature = max(contributions, key=contributions.get) if contributions else "unknown"

            anomaly = AnomalyEvent(
                anomaly_id=str(uuid.uuid4()),
                timestamp=reading.timestamp,
                anomaly_type=self._infer_anomaly_type_from_features(contributions),
                severity=self._ml_severity(score),
                detection_method="ml",
                confidence=score,
                observed_value=features.get(top_feature, 0),
                expected_value=0,  # ML doesn't provide expected value
                deviation_pct=contributions.get(top_feature, 0) * 100,
                affected_parameter=top_feature,
                potential_causes=["Pattern deviation detected by ML model"],
                recommended_actions=["Investigate unusual operating pattern"],
                anomaly_score=score,
                feature_contributions=contributions,
            )
            anomalies.append(anomaly)

        self._add_audit_entry("ml_detection", {
            "anomaly_score": score,
            "is_anomaly": is_anomaly,
            "health_score": health_score,
        })

        return anomalies, health_score

    def _run_rule_detection(self, reading: FlueGasReading) -> List[AnomalyEvent]:
        """Run rule-based anomaly detection."""
        anomalies = []
        triggered_rules = self.rules.detect(reading)

        for rule in triggered_rules:
            anomaly = AnomalyEvent(
                anomaly_id=str(uuid.uuid4()),
                timestamp=reading.timestamp,
                anomaly_type=rule["type"],
                severity=rule["severity"],
                detection_method="rule_based",
                confidence=0.95,  # Rules have high confidence
                observed_value=self._get_rule_value(reading, rule["type"]),
                expected_value=0,
                deviation_pct=0,
                affected_parameter=rule["type"].value,
                potential_causes=rule["causes"],
                recommended_actions=rule["actions"],
            )
            anomalies.append(anomaly)

        self._add_audit_entry("rule_detection", {
            "rules_checked": len(self.rules._rules),
            "rules_triggered": len(triggered_rules),
        })

        return anomalies

    def _get_rule_value(self, reading: FlueGasReading, anomaly_type: AnomalyType) -> float:
        """Get the value associated with an anomaly type."""
        mapping = {
            AnomalyType.EXCESS_OXYGEN: reading.oxygen_pct,
            AnomalyType.LOW_OXYGEN: reading.oxygen_pct,
            AnomalyType.HIGH_CO: reading.co_ppm,
            AnomalyType.HIGH_NOX: reading.nox_ppm,
            AnomalyType.HIGH_COMBUSTIBLES: reading.combustibles_pct or 0,
            AnomalyType.FOULING_DETECTED: reading.flue_gas_temp_c,
        }
        return mapping.get(anomaly_type, 0)

    def _infer_anomaly_type_from_features(self, contributions: Dict[str, float]) -> AnomalyType:
        """Infer anomaly type from ML feature contributions."""
        if not contributions:
            return AnomalyType.AIR_FUEL_IMBALANCE

        top_feature = max(contributions, key=contributions.get)
        mapping = {
            "oxygen": AnomalyType.EXCESS_OXYGEN,
            "co": AnomalyType.HIGH_CO,
            "nox": AnomalyType.HIGH_NOX,
            "co2": AnomalyType.AIR_FUEL_IMBALANCE,
            "stack_temp": AnomalyType.HEAT_TRANSFER_DEGRADATION,
        }
        return mapping.get(top_feature, AnomalyType.AIR_FUEL_IMBALANCE)

    def _spc_severity(self, zscore: float) -> AnomalySeverity:
        """Determine severity from z-score."""
        zscore_abs = abs(zscore)
        if zscore_abs >= 4:
            return AnomalySeverity.CRITICAL
        elif zscore_abs >= 3:
            return AnomalySeverity.ALARM
        elif zscore_abs >= 2:
            return AnomalySeverity.WARNING
        else:
            return AnomalySeverity.INFO

    def _ml_severity(self, score: float) -> AnomalySeverity:
        """Determine severity from ML anomaly score."""
        if score >= 0.9:
            return AnomalySeverity.CRITICAL
        elif score >= 0.8:
            return AnomalySeverity.ALARM
        elif score >= 0.7:
            return AnomalySeverity.WARNING
        else:
            return AnomalySeverity.INFO

    def _apply_cooldown(self, anomalies: List[AnomalyEvent]) -> List[AnomalyEvent]:
        """Apply cooldown to prevent alert flooding."""
        now = datetime.now(timezone.utc)
        filtered = []

        for anomaly in anomalies:
            key = f"{anomaly.anomaly_type.value}_{anomaly.detection_method}"
            last_alert = self._last_alerts.get(key)

            if last_alert is None or (now - last_alert).total_seconds() > self.config.alert_cooldown_s:
                self._last_alerts[key] = now
                filtered.append(anomaly)

        return filtered

    def _calculate_provenance_hash(
        self,
        reading: FlueGasReading,
        anomalies: List[AnomalyEvent],
    ) -> str:
        """Calculate provenance hash for audit trail."""
        data = {
            "input": {
                "timestamp": reading.timestamp.isoformat(),
                "o2": reading.oxygen_pct,
                "co2": reading.co2_pct,
                "co": reading.co_ppm,
            },
            "output": {
                "anomaly_count": len(anomalies),
                "anomaly_types": [a.anomaly_type.value for a in anomalies],
            },
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def _add_audit_entry(self, operation: str, data: Dict[str, Any]) -> None:
        """Add entry to audit trail."""
        self._audit_trail.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "data": data,
        })

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get detection audit trail."""
        return self._audit_trail.copy()
