# -*- coding: utf-8 -*-
"""
GL-016 Waterguard Corrosion Detector - Corrosion Event Detection

This module provides corrosion event detection through iron and copper spike
monitoring. Detects condenser leaks, tube failures, and corrosion events.

ALL CALCULATIONS ARE DETERMINISTIC - NO GENERATIVE AI FOR NUMERIC DECISIONS.

Example:
    >>> detector = CorrosionDetector(config)
    >>> event = detector.detect_iron_spike(iron_values, baseline=0.05)
    >>> if event.detected:
    ...     print(f"Corrosion: {event.affected_system}, Action: {event.recommended_action}")

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


class CorrosionSeverity(str, Enum):
    """Severity levels for corrosion events."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AffectedSystem(str, Enum):
    """Systems that may be affected by corrosion."""
    CONDENSER = "CONDENSER"
    FEEDWATER_HEATER = "FEEDWATER_HEATER"
    ECONOMIZER = "ECONOMIZER"
    BOILER_TUBES = "BOILER_TUBES"
    STEAM_DRUM = "STEAM_DRUM"
    DEAERATOR = "DEAERATOR"
    COOLING_TOWER = "COOLING_TOWER"
    PIPING = "PIPING"
    UNKNOWN = "UNKNOWN"


class CorrosionType(str, Enum):
    """Types of corrosion detected."""
    IRON_SPIKE = "IRON_SPIKE"
    COPPER_SPIKE = "COPPER_SPIKE"
    COMBINED_SPIKE = "COMBINED_SPIKE"
    SUSTAINED_ELEVATED = "SUSTAINED_ELEVATED"
    FLOW_ACCELERATED = "FLOW_ACCELERATED"
    OXYGEN_INDUCED = "OXYGEN_INDUCED"
    CHEMICAL_ATTACK = "CHEMICAL_ATTACK"


@dataclass
class CorrosionDetectorConfig:
    """Configuration for CorrosionDetector."""

    # Iron thresholds (ppb)
    iron_baseline_ppb: float = 10.0
    iron_warning_ppb: float = 50.0
    iron_alarm_ppb: float = 100.0
    iron_critical_ppb: float = 500.0

    # Copper thresholds (ppb)
    copper_baseline_ppb: float = 2.0
    copper_warning_ppb: float = 10.0
    copper_alarm_ppb: float = 20.0
    copper_critical_ppb: float = 50.0

    # Spike detection parameters
    spike_sigma_threshold: float = 3.0
    spike_min_duration: int = 3
    sustained_min_duration: int = 10

    # Rate of change thresholds
    rate_of_change_threshold: float = 2.0

    # Confidence thresholds
    min_confidence: float = 0.7

    def __post_init__(self):
        """Validate configuration."""
        if self.iron_baseline_ppb <= 0:
            raise ValueError("iron_baseline_ppb must be positive")
        if self.copper_baseline_ppb <= 0:
            raise ValueError("copper_baseline_ppb must be positive")


@dataclass
class CorrosionEvent:
    """Result from corrosion detection."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    detected: bool = False
    corrosion_type: Optional[CorrosionType] = None
    affected_system: AffectedSystem = AffectedSystem.UNKNOWN
    severity: CorrosionSeverity = CorrosionSeverity.LOW
    confidence: float = 0.0
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
            "corrosion_type": self.corrosion_type.value if self.corrosion_type else None,
            "affected_system": self.affected_system.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "recommended_action": self.recommended_action,
            "detection_time": self.detection_time.isoformat(),
            "event_start_time": self.event_start_time.isoformat() if self.event_start_time else None,
            "description": self.description,
            "metrics": self.metrics,
            "affected_tags": self.affected_tags,
            "provenance_hash": self.provenance_hash,
        }


class CorrosionDetector:
    """
    Corrosion event detection for water chemistry monitoring.

    This detector uses ONLY statistical methods for corrosion detection,
    ensuring zero-hallucination compliance per GreenLang standards.

    Detection Methods:
        - Threshold analysis: Iron/copper spike detection
        - Z-score: Deviation from baseline
        - Rate of change: Rapid increase detection
        - Duration analysis: Sustained elevation detection

    Attributes:
        config: Detector configuration

    Example:
        >>> config = CorrosionDetectorConfig(iron_alarm_ppb=100.0)
        >>> detector = CorrosionDetector(config)
        >>> event = detector.detect_iron_spike(values, baseline=10.0)
    """

    def __init__(self, config: Optional[CorrosionDetectorConfig] = None):
        """
        Initialize CorrosionDetector.

        Args:
            config: Detector configuration. Uses defaults if None.
        """
        self.config = config or CorrosionDetectorConfig()
        logger.info(
            f"CorrosionDetector initialized with iron_alarm={self.config.iron_alarm_ppb}ppb"
        )

    def detect_iron_spike(
        self,
        iron_values: List[float],
        baseline: Optional[float] = None,
        timestamps: Optional[List[datetime]] = None,
        tag_id: str = ""
    ) -> CorrosionEvent:
        """
        Detect iron spikes indicating corrosion or tube failure.

        Uses threshold and z-score analysis to detect elevated iron levels.
        Zero-hallucination: Uses deterministic statistical thresholds.

        Args:
            iron_values: Iron concentration values (ppb)
            baseline: Optional baseline override
            timestamps: Optional timestamps for readings
            tag_id: Iron sensor tag identifier

        Returns:
            CorrosionEvent with detection details
        """
        start_time = datetime.now()
        base = baseline or self.config.iron_baseline_ppb

        if len(iron_values) < 3:
            return CorrosionEvent(
                detected=False,
                description="Insufficient data for iron spike detection",
                affected_tags=[tag_id] if tag_id else [],
            )

        values = np.array(iron_values)
        current_value = values[-1]
        mean_value = np.mean(values)
        max_value = np.max(values)
        std_value = np.std(values)

        # Determine severity based on thresholds
        severity = CorrosionSeverity.LOW
        detected = False

        if max_value >= self.config.iron_critical_ppb:
            severity = CorrosionSeverity.CRITICAL
            detected = True
        elif max_value >= self.config.iron_alarm_ppb:
            severity = CorrosionSeverity.HIGH
            detected = True
        elif max_value >= self.config.iron_warning_ppb:
            severity = CorrosionSeverity.MEDIUM
            detected = True
        elif max_value >= base * 3:
            severity = CorrosionSeverity.LOW
            detected = True

        # Calculate z-score if we have enough data
        z_score = 0.0
        if std_value > 0:
            z_score = (max_value - mean_value) / std_value

        # Determine affected system based on iron levels
        affected_system = self._infer_affected_system_iron(max_value, severity)

        # Generate recommended action
        action = self._generate_iron_action(severity, max_value)

        # Calculate confidence
        confidence = self._calculate_confidence(
            current_value, base, self.config.iron_alarm_ppb, z_score
        )

        result = CorrosionEvent(
            detected=detected,
            corrosion_type=CorrosionType.IRON_SPIKE if detected else None,
            affected_system=affected_system,
            severity=severity,
            confidence=confidence,
            recommended_action=action,
            detection_time=datetime.now(timezone.utc),
            event_start_time=timestamps[0] if timestamps else None,
            description=f"Iron {'spike detected' if detected else 'within limits'}: max={max_value:.1f}ppb",
            metrics={
                "current_value_ppb": float(current_value),
                "mean_value_ppb": float(mean_value),
                "max_value_ppb": float(max_value),
                "baseline_ppb": float(base),
                "z_score": float(z_score),
                "exceedance_ratio": float(max_value / base),
            },
            affected_tags=[tag_id] if tag_id else [],
        )

        result.provenance_hash = self._calculate_provenance(result, iron_values)
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Iron spike detection completed in {duration_ms:.2f}ms")

        return result

    def detect_copper_spike(
        self,
        copper_values: List[float],
        baseline: Optional[float] = None,
        timestamps: Optional[List[datetime]] = None,
        tag_id: str = ""
    ) -> CorrosionEvent:
        """
        Detect copper spikes indicating condenser tube leaks.

        Uses threshold and z-score analysis to detect elevated copper levels.
        Zero-hallucination: Uses deterministic statistical thresholds.

        Args:
            copper_values: Copper concentration values (ppb)
            baseline: Optional baseline override
            timestamps: Optional timestamps for readings
            tag_id: Copper sensor tag identifier

        Returns:
            CorrosionEvent with detection details
        """
        start_time = datetime.now()
        base = baseline or self.config.copper_baseline_ppb

        if len(copper_values) < 3:
            return CorrosionEvent(
                detected=False,
                description="Insufficient data for copper spike detection",
                affected_tags=[tag_id] if tag_id else [],
            )

        values = np.array(copper_values)
        current_value = values[-1]
        mean_value = np.mean(values)
        max_value = np.max(values)
        std_value = np.std(values)

        # Determine severity based on thresholds
        severity = CorrosionSeverity.LOW
        detected = False

        if max_value >= self.config.copper_critical_ppb:
            severity = CorrosionSeverity.CRITICAL
            detected = True
        elif max_value >= self.config.copper_alarm_ppb:
            severity = CorrosionSeverity.HIGH
            detected = True
        elif max_value >= self.config.copper_warning_ppb:
            severity = CorrosionSeverity.MEDIUM
            detected = True
        elif max_value >= base * 3:
            severity = CorrosionSeverity.LOW
            detected = True

        # Calculate z-score
        z_score = 0.0
        if std_value > 0:
            z_score = (max_value - mean_value) / std_value

        # Copper usually indicates condenser issues
        affected_system = AffectedSystem.CONDENSER if detected else AffectedSystem.UNKNOWN

        # Generate recommended action
        action = self._generate_copper_action(severity, max_value)

        # Calculate confidence
        confidence = self._calculate_confidence(
            current_value, base, self.config.copper_alarm_ppb, z_score
        )

        result = CorrosionEvent(
            detected=detected,
            corrosion_type=CorrosionType.COPPER_SPIKE if detected else None,
            affected_system=affected_system,
            severity=severity,
            confidence=confidence,
            recommended_action=action,
            detection_time=datetime.now(timezone.utc),
            event_start_time=timestamps[0] if timestamps else None,
            description=f"Copper {'spike detected' if detected else 'within limits'}: max={max_value:.1f}ppb",
            metrics={
                "current_value_ppb": float(current_value),
                "mean_value_ppb": float(mean_value),
                "max_value_ppb": float(max_value),
                "baseline_ppb": float(base),
                "z_score": float(z_score),
                "exceedance_ratio": float(max_value / base),
            },
            affected_tags=[tag_id] if tag_id else [],
        )

        result.provenance_hash = self._calculate_provenance(result, copper_values)
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Copper spike detection completed in {duration_ms:.2f}ms")

        return result

    def detect_combined_corrosion(
        self,
        iron_values: List[float],
        copper_values: List[float],
        timestamps: Optional[List[datetime]] = None,
        iron_tag: str = "",
        copper_tag: str = ""
    ) -> CorrosionEvent:
        """
        Detect combined iron and copper corrosion events.

        Analyzes correlation between iron and copper to identify
        systemic corrosion issues.
        Zero-hallucination: Uses deterministic correlation analysis.

        Args:
            iron_values: Iron concentration values (ppb)
            copper_values: Copper concentration values (ppb)
            timestamps: Optional timestamps for readings
            iron_tag: Iron sensor tag identifier
            copper_tag: Copper sensor tag identifier

        Returns:
            CorrosionEvent with detection details
        """
        if len(iron_values) != len(copper_values):
            raise ValueError("Iron and copper arrays must have equal length")

        if len(iron_values) < 5:
            return CorrosionEvent(
                detected=False,
                description="Insufficient data for combined corrosion detection",
                affected_tags=[iron_tag, copper_tag],
            )

        iron_arr = np.array(iron_values)
        copper_arr = np.array(copper_values)

        # Check individual spikes first
        iron_event = self.detect_iron_spike(iron_values, tag_id=iron_tag)
        copper_event = self.detect_copper_spike(copper_values, tag_id=copper_tag)

        # Calculate correlation
        correlation, p_value = stats.pearsonr(iron_arr, copper_arr)

        # Combined detection logic
        both_detected = iron_event.detected and copper_event.detected
        correlated = correlation > 0.7 and p_value < 0.05

        if both_detected and correlated:
            # Significant combined event - likely systemic issue
            max_severity = max(
                iron_event.severity.value,
                copper_event.severity.value,
                key=lambda x: ["LOW", "MEDIUM", "HIGH", "CRITICAL"].index(x)
            )
            severity = CorrosionSeverity(max_severity)

            affected_system = self._infer_combined_system(iron_event, copper_event)

            confidence = (iron_event.confidence + copper_event.confidence) / 2 * (1 + correlation) / 2

            result = CorrosionEvent(
                detected=True,
                corrosion_type=CorrosionType.COMBINED_SPIKE,
                affected_system=affected_system,
                severity=severity,
                confidence=min(1.0, confidence),
                recommended_action=self._generate_combined_action(severity),
                detection_time=datetime.now(timezone.utc),
                event_start_time=timestamps[0] if timestamps else None,
                description=f"Combined Fe/Cu corrosion event: correlation={correlation:.2f}",
                metrics={
                    "iron_max_ppb": float(np.max(iron_arr)),
                    "copper_max_ppb": float(np.max(copper_arr)),
                    "correlation": float(correlation),
                    "p_value": float(p_value),
                    "iron_severity": iron_event.severity.value,
                    "copper_severity": copper_event.severity.value,
                },
                affected_tags=[iron_tag, copper_tag],
            )

            result.provenance_hash = self._calculate_provenance(
                result, iron_values + copper_values
            )
            return result

        elif iron_event.detected or copper_event.detected:
            # Return the more significant single event
            if iron_event.severity.value > copper_event.severity.value:
                return iron_event
            return copper_event

        return CorrosionEvent(
            detected=False,
            description="No significant corrosion detected",
            affected_tags=[iron_tag, copper_tag],
        )

    def detect_sustained_elevation(
        self,
        values: List[float],
        threshold: float,
        element: str = "iron",
        timestamps: Optional[List[datetime]] = None,
        tag_id: str = ""
    ) -> CorrosionEvent:
        """
        Detect sustained elevation above threshold.

        Identifies chronic corrosion conditions vs acute spikes.
        Zero-hallucination: Uses deterministic duration analysis.

        Args:
            values: Concentration values (ppb)
            threshold: Threshold for elevated condition
            element: Element being monitored ("iron" or "copper")
            timestamps: Optional timestamps for readings
            tag_id: Sensor tag identifier

        Returns:
            CorrosionEvent with detection details
        """
        if len(values) < self.config.sustained_min_duration:
            return CorrosionEvent(
                detected=False,
                description="Insufficient data for sustained elevation detection",
                affected_tags=[tag_id] if tag_id else [],
            )

        arr = np.array(values)

        # Count consecutive samples above threshold
        above_threshold = arr > threshold
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

            severity = CorrosionSeverity.MEDIUM
            if max_consecutive >= self.config.sustained_min_duration * 3:
                severity = CorrosionSeverity.HIGH
            if max_consecutive >= self.config.sustained_min_duration * 5:
                severity = CorrosionSeverity.CRITICAL

            confidence = min(1.0, max_consecutive / (self.config.sustained_min_duration * 2))

            result = CorrosionEvent(
                detected=True,
                corrosion_type=CorrosionType.SUSTAINED_ELEVATED,
                affected_system=AffectedSystem.UNKNOWN,
                severity=severity,
                confidence=confidence,
                recommended_action=f"Investigate chronic {element} source; check water treatment",
                detection_time=datetime.now(timezone.utc),
                event_start_time=timestamps[sustained_start] if timestamps and sustained_start else None,
                description=f"Sustained {element} elevation: {max_consecutive} samples above {threshold}ppb",
                metrics={
                    "max_consecutive": max_consecutive,
                    "threshold_ppb": float(threshold),
                    "mean_elevated_ppb": float(mean_elevated),
                    "max_value_ppb": float(max_value),
                },
                affected_tags=[tag_id] if tag_id else [],
            )

            result.provenance_hash = self._calculate_provenance(result, values)
            return result

        return CorrosionEvent(
            detected=False,
            description=f"No sustained {element} elevation detected",
            affected_tags=[tag_id] if tag_id else [],
        )

    def _infer_affected_system_iron(
        self,
        iron_ppb: float,
        severity: CorrosionSeverity
    ) -> AffectedSystem:
        """Infer affected system based on iron levels and patterns."""
        if severity == CorrosionSeverity.CRITICAL:
            return AffectedSystem.BOILER_TUBES
        elif severity == CorrosionSeverity.HIGH:
            return AffectedSystem.FEEDWATER_HEATER
        elif severity == CorrosionSeverity.MEDIUM:
            return AffectedSystem.ECONOMIZER
        return AffectedSystem.PIPING

    def _generate_iron_action(self, severity: CorrosionSeverity, max_ppb: float) -> str:
        """Generate recommended action for iron spike."""
        if severity == CorrosionSeverity.CRITICAL:
            return "IMMEDIATE: Reduce load, perform chemical clean, inspect boiler tubes"
        elif severity == CorrosionSeverity.HIGH:
            return "URGENT: Increase blowdown, check feedwater heaters, schedule inspection"
        elif severity == CorrosionSeverity.MEDIUM:
            return "Monitor closely, verify pH and oxygen scavenger levels"
        return "Continue monitoring, verify water treatment program"

    def _generate_copper_action(self, severity: CorrosionSeverity, max_ppb: float) -> str:
        """Generate recommended action for copper spike."""
        if severity == CorrosionSeverity.CRITICAL:
            return "IMMEDIATE: Check condenser for tube leak, consider load reduction"
        elif severity == CorrosionSeverity.HIGH:
            return "URGENT: Inspect condenser tubes, check cooling water treatment"
        elif severity == CorrosionSeverity.MEDIUM:
            return "Schedule condenser inspection, verify cooling water chemistry"
        return "Monitor copper trend, check admiralty brass tube condition"

    def _generate_combined_action(self, severity: CorrosionSeverity) -> str:
        """Generate recommended action for combined corrosion."""
        if severity == CorrosionSeverity.CRITICAL:
            return "IMMEDIATE: Systemic corrosion event - reduce load, full chemistry review"
        elif severity == CorrosionSeverity.HIGH:
            return "URGENT: Multiple systems affected - comprehensive inspection required"
        return "Investigate corrosion source, review water treatment across systems"

    def _infer_combined_system(
        self,
        iron_event: CorrosionEvent,
        copper_event: CorrosionEvent
    ) -> AffectedSystem:
        """Infer affected system from combined analysis."""
        if copper_event.severity in [CorrosionSeverity.HIGH, CorrosionSeverity.CRITICAL]:
            return AffectedSystem.CONDENSER
        return iron_event.affected_system

    def _calculate_confidence(
        self,
        current: float,
        baseline: float,
        alarm: float,
        z_score: float
    ) -> float:
        """Calculate detection confidence."""
        exceedance = current / baseline
        alarm_ratio = current / alarm

        base_confidence = min(1.0, exceedance / 3)
        z_confidence = min(1.0, abs(z_score) / 5)

        confidence = (base_confidence + z_confidence) / 2

        if alarm_ratio >= 1.0:
            confidence = max(confidence, 0.8)

        return max(0.0, min(1.0, confidence))

    def _calculate_provenance(
        self,
        result: CorrosionEvent,
        values: List[float]
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_str = (
            f"{result.event_id}"
            f"{result.corrosion_type}"
            f"{result.detection_time.isoformat()}"
            f"{len(values)}"
            f"{sum(values):.10f}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()
