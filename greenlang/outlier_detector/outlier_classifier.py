# -*- coding: utf-8 -*-
"""
Outlier Classifier Engine - AGENT-DATA-013

Classifies detected outliers into five root-cause categories:
ERROR, GENUINE_EXTREME, DATA_ENTRY, REGIME_CHANGE, and SENSOR_FAULT.
Uses deterministic heuristic rules to assign confidence-scored
classifications based on data patterns.

Zero-Hallucination: All classification logic uses deterministic Python
rules and pattern matching. No LLM calls for classification decisions.

Example:
    >>> from greenlang.outlier_detector.outlier_classifier import OutlierClassifierEngine
    >>> engine = OutlierClassifierEngine()
    >>> classifications = engine.classify_outliers(detections, records)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection (GL-DATA-X-016)
Status: Production Ready
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

from greenlang.outlier_detector.config import get_config
from greenlang.outlier_detector.models import (
    OutlierClass,
    OutlierClassification,
    OutlierScore,
    SeverityLevel,
    TreatmentStrategy,
)
from greenlang.outlier_detector.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)


def _safe_mean(values: List[float]) -> float:
    """Compute arithmetic mean, returning 0.0 for empty lists."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_std(values: List[float], mean: Optional[float] = None) -> float:
    """Compute population standard deviation."""
    if len(values) < 2:
        return 0.0
    m = mean if mean is not None else _safe_mean(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def _severity_from_score(score: float) -> SeverityLevel:
    """Map normalised score to severity level."""
    if score >= 0.95:
        return SeverityLevel.CRITICAL
    if score >= 0.80:
        return SeverityLevel.HIGH
    if score >= 0.60:
        return SeverityLevel.MEDIUM
    if score >= 0.40:
        return SeverityLevel.LOW
    return SeverityLevel.INFO


# Treatment recommendations by outlier class
_CLASS_TREATMENTS: Dict[OutlierClass, TreatmentStrategy] = {
    OutlierClass.ERROR: TreatmentStrategy.REMOVE,
    OutlierClass.GENUINE_EXTREME: TreatmentStrategy.FLAG,
    OutlierClass.DATA_ENTRY: TreatmentStrategy.REPLACE,
    OutlierClass.REGIME_CHANGE: TreatmentStrategy.INVESTIGATE,
    OutlierClass.SENSOR_FAULT: TreatmentStrategy.REMOVE,
}


class OutlierClassifierEngine:
    """Outlier root-cause classification engine.

    Classifies detected outliers into one of five categories using
    deterministic pattern-matching heuristics on the data values,
    their context, and their detection characteristics.

    Attributes:
        _config: Outlier detector configuration.
        _provenance: SHA-256 provenance tracker.

    Example:
        >>> engine = OutlierClassifierEngine()
        >>> result = engine.classify_single(detection, record)
        >>> print(result.outlier_class, result.confidence)
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize OutlierClassifierEngine.

        Args:
            config: Optional OutlierDetectorConfig override.
        """
        self._config = config or get_config()
        self._provenance = ProvenanceTracker()
        logger.info("OutlierClassifierEngine initialized")

    # ------------------------------------------------------------------
    # Batch classification
    # ------------------------------------------------------------------

    def classify_outliers(
        self,
        detections: List[OutlierScore],
        records: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[OutlierClassification]:
        """Classify all detected outliers.

        Iterates over the detection scores that are flagged as outliers
        and assigns a root-cause classification to each one.

        Args:
            detections: List of OutlierScore from detection stage.
            records: Original record dictionaries.
            context: Optional context information (time series data,
                neighboring values, domain info).

        Returns:
            List of OutlierClassification for flagged outliers.
        """
        start = time.time()
        ctx = context or {}
        classifications: List[OutlierClassification] = []

        for detection in detections:
            if not detection.is_outlier:
                continue

            record = {}
            if detection.record_index < len(records):
                record = records[detection.record_index]

            classification = self.classify_single(detection, record, ctx)
            classifications.append(classification)

        elapsed = time.time() - start
        logger.debug(
            "Classified %d outliers in %.3fs",
            len(classifications), elapsed,
        )
        return classifications

    # ------------------------------------------------------------------
    # Single-point classification
    # ------------------------------------------------------------------

    def classify_single(
        self,
        detection: OutlierScore,
        record: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> OutlierClassification:
        """Classify a single detected outlier.

        Evaluates all five classification heuristics and assigns the
        class with the highest confidence score.

        Args:
            detection: OutlierScore for this point.
            record: Original record dictionary.
            context: Optional context (time_series, neighbors, domain).

        Returns:
            OutlierClassification with class, confidence, evidence.
        """
        ctx = context or {}
        scores: Dict[OutlierClass, Tuple[float, List[str]]] = {}

        # Evaluate each classifier
        scores[OutlierClass.ERROR] = self._classify_error(detection, record)
        scores[OutlierClass.DATA_ENTRY] = self._classify_data_entry(detection, record)
        scores[OutlierClass.REGIME_CHANGE] = self._classify_regime_change(
            detection, record, ctx,
        )
        scores[OutlierClass.SENSOR_FAULT] = self._classify_sensor_fault(
            detection, record,
        )
        scores[OutlierClass.GENUINE_EXTREME] = self._classify_genuine_extreme(
            detection, record,
        )

        # Select highest-confidence class
        best_class = OutlierClass.GENUINE_EXTREME
        best_score = 0.0
        best_evidence: List[str] = []

        for oc, (sc, ev) in scores.items():
            if sc > best_score:
                best_class = oc
                best_score = sc
                best_evidence = ev

        # Build class_scores dict
        class_scores = {oc.value: sc for oc, (sc, _) in scores.items()}

        overall_confidence = self.compute_classification_confidence(class_scores)

        provenance_hash = self._provenance.build_hash({
            "index": detection.record_index,
            "value": detection.value,
            "class": best_class.value,
            "confidence": overall_confidence,
            "class_scores": class_scores,
        })

        severity = _severity_from_score(detection.score)
        treatment = _CLASS_TREATMENTS.get(best_class, TreatmentStrategy.FLAG)

        return OutlierClassification(
            record_index=detection.record_index,
            column_name=detection.column_name,
            value=detection.value,
            outlier_class=best_class,
            confidence=overall_confidence,
            class_scores=class_scores,
            evidence=best_evidence,
            severity=severity,
            recommended_treatment=treatment,
            provenance_hash=provenance_hash,
        )

    # ------------------------------------------------------------------
    # Error classification
    # ------------------------------------------------------------------

    def _classify_error(
        self,
        detection: OutlierScore,
        record: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        """Classify as data processing or transmission error.

        Patterns: negative values where not expected, NaN-like strings,
        impossibly large/small values, values that are exact powers of 2
        (binary corruption).

        Args:
            detection: Detection score.
            record: Record data.

        Returns:
            Tuple of (confidence_score, evidence_list).
        """
        score = 0.0
        evidence: List[str] = []
        value = detection.value

        if value is None:
            return 0.0, []

        try:
            v = float(value)
        except (ValueError, TypeError):
            evidence.append("Non-numeric value in numeric field")
            return 0.7, evidence

        # Check for extremely large magnitude (possible overflow)
        if abs(v) > 1e15:
            score += 0.4
            evidence.append(f"Extreme magnitude: {v:.2e}")

        # Check for exact zero where others are non-zero
        if v == 0.0 and detection.score > 0.5:
            score += 0.2
            evidence.append("Exact zero in non-zero context")

        # Check for negative values (if column is typically positive)
        if v < 0 and detection.details.get("mean", 0) > 0:
            score += 0.3
            evidence.append("Negative value in typically positive column")

        # Check for binary corruption pattern (exact powers of 2)
        if v != 0 and abs(v) > 1:
            log2_val = math.log2(abs(v))
            if abs(log2_val - round(log2_val)) < 0.001:
                score += 0.15
                evidence.append(f"Value is exact power of 2: {v}")

        return min(1.0, score), evidence

    # ------------------------------------------------------------------
    # Data entry classification
    # ------------------------------------------------------------------

    def _classify_data_entry(
        self,
        detection: OutlierScore,
        record: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        """Classify as human data entry mistake.

        Patterns: digit transposition (1234 vs 1324), decimal point
        shift (10x or 0.1x expected), unit confusion (km vs m),
        round number (exactly 1000, 10000, etc.).

        Args:
            detection: Detection score.
            record: Record data.

        Returns:
            Tuple of (confidence_score, evidence_list).
        """
        score = 0.0
        evidence: List[str] = []
        value = detection.value

        if value is None:
            return 0.0, []

        try:
            v = float(value)
        except (ValueError, TypeError):
            return 0.0, []

        mean = detection.details.get("mean", 0)
        std = detection.details.get("std", 1)

        # Decimal point shift (value is ~10x or ~100x the mean)
        if mean != 0 and abs(mean) > 1e-10:
            ratio = abs(v / mean)
            if 9.0 < ratio < 11.0:
                score += 0.35
                evidence.append(f"Value is ~10x the mean ({ratio:.1f}x)")
            elif 99.0 < ratio < 101.0:
                score += 0.4
                evidence.append(f"Value is ~100x the mean ({ratio:.1f}x)")
            elif 0.009 < ratio < 0.011:
                score += 0.35
                evidence.append(f"Value is ~0.01x the mean ({ratio:.4f}x)")
            elif 0.09 < ratio < 0.11:
                score += 0.35
                evidence.append(f"Value is ~0.1x the mean ({ratio:.3f}x)")

        # Round number check
        if v != 0:
            str_v = str(abs(v))
            if "." not in str_v or str_v.endswith(".0"):
                # Count trailing zeros
                int_str = str(int(abs(v)))
                zeros = len(int_str) - len(int_str.rstrip("0"))
                if zeros >= 3 and len(int_str) >= 4:
                    score += 0.2
                    evidence.append(f"Round number with {zeros} trailing zeros")

        # Digit transposition check (compare digit sorted vs original)
        if mean != 0 and abs(v) > 1:
            v_digits = sorted(str(int(abs(v))))
            m_digits = sorted(str(int(abs(mean))))
            if v_digits == m_digits and v != mean:
                score += 0.4
                evidence.append("Possible digit transposition")

        # Unit confusion (value is 1000x or 1/1000 of expected)
        if mean != 0 and abs(mean) > 1e-10:
            ratio = abs(v / mean)
            if 999.0 < ratio < 1001.0:
                score += 0.3
                evidence.append("Possible unit confusion (1000x)")
            elif 0.0009 < ratio < 0.0011:
                score += 0.3
                evidence.append("Possible unit confusion (1/1000x)")

        return min(1.0, score), evidence

    # ------------------------------------------------------------------
    # Regime change classification
    # ------------------------------------------------------------------

    def _classify_regime_change(
        self,
        detection: OutlierScore,
        record: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        """Classify as regime change or structural shift.

        Patterns: value is close to a new baseline that persists in
        subsequent observations, step change followed by stability,
        change correlates with known events.

        Args:
            detection: Detection score.
            record: Record data.
            context: Context with optional time_series and events.

        Returns:
            Tuple of (confidence_score, evidence_list).
        """
        score = 0.0
        evidence: List[str] = []
        value = detection.value

        if value is None:
            return 0.0, []

        try:
            v = float(value)
        except (ValueError, TypeError):
            return 0.0, []

        # Check if there are subsequent values at the same level
        time_series = context.get("time_series", [])
        idx = detection.record_index

        if time_series and idx < len(time_series) - 2:
            # Check if the next few values are similar to this one
            subsequent = time_series[idx + 1:idx + 4]
            if subsequent:
                try:
                    sub_values = [float(x) for x in subsequent]
                    sub_mean = _safe_mean(sub_values)
                    sub_std = _safe_std(sub_values, sub_mean)

                    # If subsequent values cluster around this value
                    if sub_std > 0:
                        closeness = abs(v - sub_mean) / sub_std
                    else:
                        closeness = abs(v - sub_mean) if sub_mean != 0 else 0.0

                    if closeness < 1.5:
                        score += 0.4
                        evidence.append(
                            "Subsequent values cluster near this level"
                        )
                except (ValueError, TypeError):
                    pass

        # Check for step change pattern
        if time_series and idx >= 3:
            try:
                before = [float(x) for x in time_series[max(0, idx - 3):idx]]
                if before:
                    before_mean = _safe_mean(before)
                    if before_mean != 0:
                        step_size = abs(v - before_mean) / abs(before_mean)
                        if step_size > 0.3:
                            score += 0.25
                            evidence.append(
                                f"Step change of {step_size:.1%} from prior mean"
                            )
            except (ValueError, TypeError):
                pass

        # Check for known events
        events = context.get("events", [])
        if events:
            score += 0.2
            evidence.append("Coincides with known event(s)")

        # If detection is from temporal method, higher probability
        if detection.method.value == "temporal":
            score += 0.15
            evidence.append("Detected by temporal method")

        return min(1.0, score), evidence

    # ------------------------------------------------------------------
    # Sensor fault classification
    # ------------------------------------------------------------------

    def _classify_sensor_fault(
        self,
        detection: OutlierScore,
        record: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        """Classify as sensor malfunction.

        Patterns: stuck at constant value (zero variance), sudden spike
        followed by return to normal, drift pattern (gradual offset),
        value at sensor min/max bounds.

        Args:
            detection: Detection score.
            record: Record data.

        Returns:
            Tuple of (confidence_score, evidence_list).
        """
        score = 0.0
        evidence: List[str] = []
        value = detection.value

        if value is None:
            return 0.0, []

        try:
            v = float(value)
        except (ValueError, TypeError):
            return 0.0, []

        # Spike pattern: very high score but isolated
        if detection.score > 0.9:
            score += 0.2
            evidence.append("Very high outlier score (possible spike)")

        # Check for common sensor bound values
        sensor_bounds = [
            0.0, -1.0, 9999.0, -9999.0, 65535.0, -32768.0,
            32767.0, 99999.0, -99999.0,
        ]
        if v in sensor_bounds:
            score += 0.45
            evidence.append(f"Value matches common sensor bound: {v}")

        # Check for exact min/max of data type ranges
        if v == 0.0 and detection.details.get("std", 1) > 10:
            score += 0.3
            evidence.append("Stuck at zero (possible sensor fault)")

        # Repeated exact value (stuck sensor)
        details = detection.details
        if details.get("repeated_count", 0) > 3:
            score += 0.35
            evidence.append(
                f"Value repeated {details['repeated_count']} times (stuck sensor)"
            )

        # Check for very rapid change (spike)
        prev_value = details.get("previous_value")
        if prev_value is not None:
            try:
                pv = float(prev_value)
                if pv != 0:
                    change_pct = abs(v - pv) / abs(pv)
                    if change_pct > 10.0:
                        score += 0.3
                        evidence.append(
                            f"Rapid change: {change_pct:.0%} from previous value"
                        )
            except (ValueError, TypeError):
                pass

        return min(1.0, score), evidence

    # ------------------------------------------------------------------
    # Genuine extreme classification
    # ------------------------------------------------------------------

    def _classify_genuine_extreme(
        self,
        detection: OutlierScore,
        record: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        """Classify as legitimate extreme event.

        This is the "default" classification. Points receive higher
        genuine-extreme scores when other classifications score low
        and the value is within physically plausible ranges.

        Args:
            detection: Detection score.
            record: Record data.

        Returns:
            Tuple of (confidence_score, evidence_list).
        """
        score = 0.3  # Base score: genuine extreme is the default
        evidence: List[str] = []
        value = detection.value

        if value is None:
            return score, evidence

        try:
            v = float(value)
        except (ValueError, TypeError):
            return 0.0, []

        # Moderate outlier scores suggest genuine extremes
        if 0.5 < detection.score < 0.85:
            score += 0.2
            evidence.append("Moderate outlier score suggests genuine extreme")

        # Value is on the same side of distribution as the tail
        mean = detection.details.get("mean", 0)
        std = detection.details.get("std", 1)
        if std > 0 and mean != 0:
            z = (v - mean) / std
            if 3.0 < abs(z) < 6.0:
                score += 0.15
                evidence.append(f"Z-score {z:.1f} is in extreme but plausible range")

        # Not a round number (genuine values tend to be irregular)
        if v != 0:
            str_v = str(abs(v))
            if "." in str_v:
                decimal_part = str_v.split(".")[1]
                if len(decimal_part) >= 2 and decimal_part != "0" * len(decimal_part):
                    score += 0.1
                    evidence.append("Non-round value suggests natural measurement")

        # Physical plausibility (value is positive for typically positive data)
        if v > 0 and mean > 0:
            score += 0.05
            evidence.append("Value is positive and in expected direction")

        return min(1.0, score), evidence

    # ------------------------------------------------------------------
    # Confidence computation
    # ------------------------------------------------------------------

    def compute_classification_confidence(
        self,
        class_scores: Dict[str, float],
    ) -> float:
        """Compute overall classification confidence.

        Higher confidence when one class dominates and others are low.
        Uses the margin between the highest and second-highest scores.

        Args:
            class_scores: Per-class confidence scores.

        Returns:
            Overall confidence (0.0-1.0).
        """
        if not class_scores:
            return 0.0

        sorted_scores = sorted(class_scores.values(), reverse=True)
        top = sorted_scores[0]
        second = sorted_scores[1] if len(sorted_scores) > 1 else 0.0

        # Confidence is higher when margin is large
        margin = top - second
        # Also factor in the absolute strength of the top class
        confidence = min(1.0, 0.3 * top + 0.7 * margin)

        return max(0.0, min(1.0, confidence))


__all__ = [
    "OutlierClassifierEngine",
]
