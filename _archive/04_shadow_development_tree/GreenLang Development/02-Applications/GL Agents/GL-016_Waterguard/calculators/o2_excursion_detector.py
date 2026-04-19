# -*- coding: utf-8 -*-
"""
GL-016 Waterguard O2 Excursion Detector - Dissolved O2 Monitoring

This module provides dissolved oxygen excursion detection for power plant
water chemistry. Detects deaerator issues, air in-leakage, and oxygen
scavenger failures.

ALL CALCULATIONS ARE DETERMINISTIC - NO GENERATIVE AI FOR NUMERIC DECISIONS.

Example:
    >>> monitor = DissolvedO2Monitor(config)
    >>> event = monitor.detect_o2_excursion(o2_values, limit=7.0)
    >>> if event.detected:
    ...     print(f"O2 Excursion: {event.cause_hypothesis}, Severity: {event.severity}")

Author: GreenLang Waterguard Team
Version: 1.0.0
Agent: GL-016_Waterguard
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class O2CauseHypothesis(str, Enum):
    """Hypothesized causes for O2 excursions."""
    DEAERATOR_MALFUNCTION = "DEAERATOR_MALFUNCTION"
    AIR_INLEAKAGE = "AIR_INLEAKAGE"
    SCAVENGER_UNDERFEED = "SCAVENGER_UNDERFEED"
    SCAVENGER_PUMP_FAILURE = "SCAVENGER_PUMP_FAILURE"
    CONDENSER_VACUUM_LOSS = "CONDENSER_VACUUM_LOSS"
    HOTWELL_ISSUE = "HOTWELL_ISSUE"
    LOW_TEMPERATURE = "LOW_TEMPERATURE"
    STARTUP_TRANSIENT = "STARTUP_TRANSIENT"
    UNKNOWN = "UNKNOWN"


class O2Severity(str, Enum):
    """Severity levels for O2 excursions."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class O2Location(str, Enum):
    """Location of O2 measurement."""
    DEAERATOR_OUTLET = "DEAERATOR_OUTLET"
    ECONOMIZER_INLET = "ECONOMIZER_INLET"
    BOILER_FEEDWATER = "BOILER_FEEDWATER"
    CONDENSATE = "CONDENSATE"
    HOTWELL = "HOTWELL"
    UNKNOWN = "UNKNOWN"


@dataclass
class O2MonitorConfig:
    """Configuration for DissolvedO2Monitor."""

    # O2 thresholds (ppb)
    o2_target_ppb: float = 5.0
    o2_warning_ppb: float = 7.0
    o2_alarm_ppb: float = 10.0
    o2_critical_ppb: float = 20.0

    # Scavenger residual thresholds (ppb)
    scavenger_min_residual_ppb: float = 10.0
    scavenger_max_residual_ppb: float = 50.0

    # Detection parameters
    spike_min_duration: int = 3
    sustained_min_duration: int = 10
    rate_threshold_ppb_per_min: float = 2.0

    # Temperature context
    min_deaerator_temp_f: float = 220.0

    # Confidence threshold
    min_confidence: float = 0.7

    def __post_init__(self):
        """Validate configuration."""
        if self.o2_target_ppb <= 0:
            raise ValueError("o2_target_ppb must be positive")
        if self.o2_warning_ppb <= self.o2_target_ppb:
            raise ValueError("o2_warning_ppb must be greater than o2_target_ppb")


@dataclass
class O2ExcursionEvent:
    """Result from O2 excursion detection."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    detected: bool = False
    cause_hypothesis: O2CauseHypothesis = O2CauseHypothesis.UNKNOWN
    severity: O2Severity = O2Severity.LOW
    confidence: float = 0.0
    location: O2Location = O2Location.UNKNOWN
    recommended_action: str = ""
    detection_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_start_time: Optional[datetime] = None
    description: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    affected_tags: List[str] = field(default_factory=list)
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "detected": self.detected,
            "cause_hypothesis": self.cause_hypothesis.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "location": self.location.value,
            "recommended_action": self.recommended_action,
            "detection_time": self.detection_time.isoformat(),
            "event_start_time": self.event_start_time.isoformat() if self.event_start_time else None,
            "description": self.description,
            "metrics": self.metrics,
            "affected_tags": self.affected_tags,
            "provenance_hash": self.provenance_hash,
        }


class DissolvedO2Monitor:
    """
    Dissolved oxygen monitoring for power plant water chemistry.

    This monitor uses ONLY statistical methods for O2 excursion detection,
    ensuring zero-hallucination compliance per GreenLang standards.

    Detection Methods:
        - Threshold analysis: O2 level excursions
        - Rate of change: Rapid O2 increase detection
        - Duration analysis: Sustained elevation detection
        - Context analysis: Deaerator temp, scavenger residual

    Attributes:
        config: Monitor configuration

    Example:
        >>> config = O2MonitorConfig(o2_alarm_ppb=10.0)
        >>> monitor = DissolvedO2Monitor(config)
        >>> event = monitor.detect_o2_excursion(o2_values, limit=7.0)
    """

    def __init__(self, config: Optional[O2MonitorConfig] = None):
        """
        Initialize DissolvedO2Monitor.

        Args:
            config: Monitor configuration. Uses defaults if None.
        """
        self.config = config or O2MonitorConfig()
        logger.info(
            f"DissolvedO2Monitor initialized with alarm={self.config.o2_alarm_ppb}ppb"
        )

    def detect_o2_excursion(
        self,
        o2_values: List[float],
        limit: Optional[float] = None,
        timestamps: Optional[List[datetime]] = None,
        tag_id: str = "",
        location: O2Location = O2Location.UNKNOWN
    ) -> O2ExcursionEvent:
        """
        Detect O2 excursion events.

        Uses threshold and statistical analysis to detect elevated O2.
        Zero-hallucination: Uses deterministic threshold analysis.

        Args:
            o2_values: Dissolved O2 concentration values (ppb)
            limit: Optional limit override (ppb)
            timestamps: Optional timestamps for readings
            tag_id: O2 sensor tag identifier
            location: Measurement location

        Returns:
            O2ExcursionEvent with detection details
        """
        start_time = datetime.now()
        threshold = limit or self.config.o2_warning_ppb

        if len(o2_values) < 3:
            return O2ExcursionEvent(
                detected=False,
                description="Insufficient data for O2 excursion detection",
                affected_tags=[tag_id] if tag_id else [],
                location=location,
            )

        values = np.array(o2_values)
        current_value = values[-1]
        mean_value = np.mean(values)
        max_value = np.max(values)
        std_value = np.std(values)

        # Determine severity based on thresholds
        severity = O2Severity.LOW
        detected = False

        if max_value >= self.config.o2_critical_ppb:
            severity = O2Severity.CRITICAL
            detected = True
        elif max_value >= self.config.o2_alarm_ppb:
            severity = O2Severity.HIGH
            detected = True
        elif max_value >= self.config.o2_warning_ppb:
            severity = O2Severity.MEDIUM
            detected = True
        elif max_value >= threshold:
            severity = O2Severity.LOW
            detected = True

        # Calculate rate of change
        rate_of_change = 0.0
        if len(values) >= 2:
            rate_of_change = float(np.mean(np.diff(values)))

        # Calculate z-score
        z_score = 0.0
        if std_value > 0:
            z_score = (max_value - mean_value) / std_value

        # Infer cause hypothesis
        cause = self._infer_cause(
            max_value, rate_of_change, severity, location
        )

        # Generate recommended action
        action = self._generate_action(severity, cause)

        # Calculate confidence
        confidence = self._calculate_confidence(
            current_value, threshold, self.config.o2_alarm_ppb, z_score
        )

        result = O2ExcursionEvent(
            detected=detected,
            cause_hypothesis=cause,
            severity=severity,
            confidence=confidence,
            location=location,
            recommended_action=action,
            detection_time=datetime.now(timezone.utc),
            event_start_time=timestamps[0] if timestamps else None,
            description=f"O2 {'excursion detected' if detected else 'within limits'}: max={max_value:.1f}ppb",
            metrics={
                "current_value_ppb": float(current_value),
                "mean_value_ppb": float(mean_value),
                "max_value_ppb": float(max_value),
                "threshold_ppb": float(threshold),
                "rate_of_change": float(rate_of_change),
                "z_score": float(z_score),
                "exceedance_ratio": float(max_value / threshold) if threshold > 0 else 0,
            },
            affected_tags=[tag_id] if tag_id else [],
        )

        result.provenance_hash = self._calculate_provenance(result, o2_values)
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"O2 excursion detection completed in {duration_ms:.2f}ms")

        return result

    def detect_with_context(
        self,
        o2_values: List[float],
        deaerator_temp: Optional[float] = None,
        scavenger_residual: Optional[float] = None,
        condenser_vacuum: Optional[float] = None,
        timestamps: Optional[List[datetime]] = None,
        tag_id: str = "",
        location: O2Location = O2Location.DEAERATOR_OUTLET
    ) -> O2ExcursionEvent:
        """
        Detect O2 excursion with additional context data.

        Uses context from related parameters to improve diagnosis.
        Zero-hallucination: Uses deterministic context analysis.

        Args:
            o2_values: Dissolved O2 concentration values (ppb)
            deaerator_temp: Deaerator temperature (F)
            scavenger_residual: Oxygen scavenger residual (ppb)
            condenser_vacuum: Condenser vacuum (in Hg)
            timestamps: Optional timestamps for readings
            tag_id: O2 sensor tag identifier
            location: Measurement location

        Returns:
            O2ExcursionEvent with detailed diagnosis
        """
        # First, get basic detection
        event = self.detect_o2_excursion(
            o2_values, timestamps=timestamps, tag_id=tag_id, location=location
        )

        if not event.detected:
            return event

        # Enhance diagnosis with context
        cause = event.cause_hypothesis
        confidence_boost = 0.0

        # Check deaerator temperature
        if deaerator_temp is not None:
            if deaerator_temp < self.config.min_deaerator_temp_f:
                cause = O2CauseHypothesis.DEAERATOR_MALFUNCTION
                confidence_boost += 0.15
                event.metrics["deaerator_temp_f"] = float(deaerator_temp)
                event.metrics["deaerator_temp_low"] = True

        # Check scavenger residual
        if scavenger_residual is not None:
            event.metrics["scavenger_residual_ppb"] = float(scavenger_residual)
            if scavenger_residual < self.config.scavenger_min_residual_ppb:
                if cause == O2CauseHypothesis.UNKNOWN:
                    cause = O2CauseHypothesis.SCAVENGER_UNDERFEED
                confidence_boost += 0.1
                event.metrics["scavenger_low"] = True

        # Check condenser vacuum
        if condenser_vacuum is not None:
            event.metrics["condenser_vacuum_inhg"] = float(condenser_vacuum)
            if condenser_vacuum < 25.0:  # Low vacuum indicates air in-leakage
                if cause == O2CauseHypothesis.UNKNOWN:
                    cause = O2CauseHypothesis.CONDENSER_VACUUM_LOSS
                confidence_boost += 0.15
                event.metrics["vacuum_loss"] = True

        # Update event with enhanced diagnosis
        event.cause_hypothesis = cause
        event.confidence = min(1.0, event.confidence + confidence_boost)
        event.recommended_action = self._generate_action(event.severity, cause)
        event.description = f"O2 excursion: {cause.value}, max={event.metrics['max_value_ppb']:.1f}ppb"

        # Recalculate provenance
        event.provenance_hash = self._calculate_provenance(event, o2_values)

        return event

    def detect_sustained_elevation(
        self,
        o2_values: List[float],
        timestamps: Optional[List[datetime]] = None,
        tag_id: str = ""
    ) -> O2ExcursionEvent:
        """
        Detect sustained O2 elevation.

        Identifies chronic O2 issues vs acute spikes.
        Zero-hallucination: Uses deterministic duration analysis.

        Args:
            o2_values: O2 concentration values (ppb)
            timestamps: Optional timestamps for readings
            tag_id: Sensor tag identifier

        Returns:
            O2ExcursionEvent with detection details
        """
        threshold = self.config.o2_warning_ppb

        if len(o2_values) < self.config.sustained_min_duration:
            return O2ExcursionEvent(
                detected=False,
                description="Insufficient data for sustained O2 detection",
                affected_tags=[tag_id] if tag_id else [],
            )

        arr = np.array(o2_values)
        above_threshold = arr > threshold

        # Count consecutive samples above threshold
        max_consecutive = 0
        current_consecutive = 0
        start_idx = None
        sustained_start = None

        for i, is_above in enumerate(above_threshold):
            if is_above:
                if current_consecutive == 0:
                    start_idx = i
                current_consecutive += 1
                if current_consecutive > max_consecutive:
                    max_consecutive = current_consecutive
                    sustained_start = start_idx
            else:
                current_consecutive = 0

        detected = max_consecutive >= self.config.sustained_min_duration

        if detected:
            mean_elevated = np.mean(arr[arr > threshold])
            max_value = np.max(arr)

            severity = O2Severity.MEDIUM
            if max_consecutive >= self.config.sustained_min_duration * 2:
                severity = O2Severity.HIGH
            if max_consecutive >= self.config.sustained_min_duration * 3:
                severity = O2Severity.CRITICAL

            confidence = min(1.0, max_consecutive / (self.config.sustained_min_duration * 2))

            result = O2ExcursionEvent(
                detected=True,
                cause_hypothesis=O2CauseHypothesis.DEAERATOR_MALFUNCTION,
                severity=severity,
                confidence=confidence,
                recommended_action="Investigate deaerator operation; check vent condenser",
                detection_time=datetime.now(timezone.utc),
                event_start_time=timestamps[sustained_start] if timestamps and sustained_start else None,
                description=f"Sustained O2 elevation: {max_consecutive} samples above {threshold}ppb",
                metrics={
                    "max_consecutive": max_consecutive,
                    "threshold_ppb": float(threshold),
                    "mean_elevated_ppb": float(mean_elevated),
                    "max_value_ppb": float(max_value),
                },
                affected_tags=[tag_id] if tag_id else [],
            )

            result.provenance_hash = self._calculate_provenance(result, o2_values)
            return result

        return O2ExcursionEvent(
            detected=False,
            description="No sustained O2 elevation detected",
            affected_tags=[tag_id] if tag_id else [],
        )

    def detect_rapid_increase(
        self,
        o2_values: List[float],
        sample_interval_min: float = 1.0,
        timestamps: Optional[List[datetime]] = None,
        tag_id: str = ""
    ) -> O2ExcursionEvent:
        """
        Detect rapid O2 increase indicating air in-leakage.

        Uses rate of change analysis to detect sudden O2 increases.
        Zero-hallucination: Uses deterministic rate calculation.

        Args:
            o2_values: O2 concentration values (ppb)
            sample_interval_min: Sampling interval in minutes
            timestamps: Optional timestamps for readings
            tag_id: Sensor tag identifier

        Returns:
            O2ExcursionEvent with detection details
        """
        if len(o2_values) < 5:
            return O2ExcursionEvent(
                detected=False,
                description="Insufficient data for rapid increase detection",
                affected_tags=[tag_id] if tag_id else [],
            )

        arr = np.array(o2_values)
        diffs = np.diff(arr)
        rate_per_min = diffs / sample_interval_min

        max_rate = float(np.max(rate_per_min))
        max_rate_idx = int(np.argmax(rate_per_min))

        detected = max_rate >= self.config.rate_threshold_ppb_per_min

        if detected:
            severity = O2Severity.MEDIUM
            if max_rate >= self.config.rate_threshold_ppb_per_min * 2:
                severity = O2Severity.HIGH
            if max_rate >= self.config.rate_threshold_ppb_per_min * 3:
                severity = O2Severity.CRITICAL

            confidence = min(1.0, max_rate / (self.config.rate_threshold_ppb_per_min * 2))

            result = O2ExcursionEvent(
                detected=True,
                cause_hypothesis=O2CauseHypothesis.AIR_INLEAKAGE,
                severity=severity,
                confidence=confidence,
                recommended_action="Check for air in-leakage; inspect condensate pumps and seals",
                detection_time=datetime.now(timezone.utc),
                event_start_time=timestamps[max_rate_idx] if timestamps else None,
                description=f"Rapid O2 increase detected: {max_rate:.2f} ppb/min",
                metrics={
                    "max_rate_ppb_per_min": float(max_rate),
                    "rate_threshold": self.config.rate_threshold_ppb_per_min,
                    "sample_interval_min": float(sample_interval_min),
                    "exceedance_index": max_rate_idx,
                },
                affected_tags=[tag_id] if tag_id else [],
            )

            result.provenance_hash = self._calculate_provenance(result, o2_values)
            return result

        return O2ExcursionEvent(
            detected=False,
            description="No rapid O2 increase detected",
            affected_tags=[tag_id] if tag_id else [],
        )

    def _infer_cause(
        self,
        max_value: float,
        rate_of_change: float,
        severity: O2Severity,
        location: O2Location
    ) -> O2CauseHypothesis:
        """Infer cause based on O2 characteristics."""
        # High rate of change suggests air in-leakage
        if rate_of_change > self.config.rate_threshold_ppb_per_min:
            return O2CauseHypothesis.AIR_INLEAKAGE

        # Location-based inference
        if location == O2Location.DEAERATOR_OUTLET:
            return O2CauseHypothesis.DEAERATOR_MALFUNCTION
        elif location == O2Location.CONDENSATE:
            return O2CauseHypothesis.CONDENSER_VACUUM_LOSS
        elif location == O2Location.HOTWELL:
            return O2CauseHypothesis.HOTWELL_ISSUE

        # Severity-based inference
        if severity == O2Severity.CRITICAL:
            return O2CauseHypothesis.AIR_INLEAKAGE
        elif severity == O2Severity.HIGH:
            return O2CauseHypothesis.DEAERATOR_MALFUNCTION

        return O2CauseHypothesis.SCAVENGER_UNDERFEED

    def _generate_action(
        self,
        severity: O2Severity,
        cause: O2CauseHypothesis
    ) -> str:
        """Generate recommended action based on severity and cause."""
        actions = {
            O2CauseHypothesis.DEAERATOR_MALFUNCTION: {
                O2Severity.CRITICAL: "IMMEDIATE: Check DA vent, verify temperature, inspect spray valves",
                O2Severity.HIGH: "URGENT: Increase DA temperature, check vent condenser operation",
                O2Severity.MEDIUM: "Verify DA operating temperature; check steam supply",
                O2Severity.LOW: "Monitor DA performance; check vent operation",
            },
            O2CauseHypothesis.AIR_INLEAKAGE: {
                O2Severity.CRITICAL: "IMMEDIATE: Perform air in-leakage survey; check vacuum breakers",
                O2Severity.HIGH: "URGENT: Inspect condenser seals, check LP turbine glands",
                O2Severity.MEDIUM: "Schedule air in-leakage test; check condensate pumps",
                O2Severity.LOW: "Monitor for trends; schedule routine inspection",
            },
            O2CauseHypothesis.SCAVENGER_UNDERFEED: {
                O2Severity.CRITICAL: "IMMEDIATE: Check scavenger pump, verify chemical inventory",
                O2Severity.HIGH: "URGENT: Increase scavenger feed rate; check pump operation",
                O2Severity.MEDIUM: "Verify scavenger residual; adjust feed rate",
                O2Severity.LOW: "Monitor scavenger residual; optimize feed rate",
            },
        }

        cause_actions = actions.get(cause, {})
        return cause_actions.get(severity, "Continue monitoring; investigate if persists")

    def _calculate_confidence(
        self,
        current: float,
        threshold: float,
        alarm: float,
        z_score: float
    ) -> float:
        """Calculate detection confidence."""
        exceedance = current / threshold if threshold > 0 else 0
        alarm_ratio = current / alarm if alarm > 0 else 0

        base_confidence = min(1.0, exceedance / 2)
        z_confidence = min(1.0, abs(z_score) / 4)

        confidence = (base_confidence + z_confidence) / 2

        if alarm_ratio >= 1.0:
            confidence = max(confidence, 0.8)

        return max(0.0, min(1.0, confidence))

    def _calculate_provenance(
        self,
        result: O2ExcursionEvent,
        values: List[float]
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_str = (
            f"{result.event_id}"
            f"{result.cause_hypothesis}"
            f"{result.detection_time.isoformat()}"
            f"{len(values)}"
            f"{sum(values):.10f}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()
