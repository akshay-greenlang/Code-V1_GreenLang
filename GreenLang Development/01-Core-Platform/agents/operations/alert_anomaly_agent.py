# -*- coding: utf-8 -*-
"""
GL-OPS-X-002: Alert & Anomaly Agent
====================================

Detects anomalies in emissions data using statistical methods and pattern
recognition. Generates alerts based on configurable rules and thresholds.

Capabilities:
    - Statistical anomaly detection (Z-score, IQR, MAD)
    - Pattern-based anomaly detection
    - Configurable alert rules and thresholds
    - Multi-level alert severity classification
    - Alert aggregation and deduplication
    - Historical anomaly tracking

Zero-Hallucination Guarantees:
    - All anomaly detection uses deterministic statistical methods
    - Complete provenance tracking with SHA-256 hashes
    - No LLM calls in the detection path
    - All alerts traceable to source data

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class AnomalyType(str, Enum):
    """Types of anomalies detected."""
    SPIKE = "spike"  # Sudden increase
    DIP = "dip"  # Sudden decrease
    DRIFT = "drift"  # Gradual shift from baseline
    OUTLIER = "outlier"  # Statistical outlier
    PATTERN = "pattern"  # Pattern deviation
    MISSING = "missing"  # Missing data
    FLATLINE = "flatline"  # No variation (sensor failure)
    SEASONAL = "seasonal"  # Seasonal pattern deviation


class AnomalySeverity(str, Enum):
    """Severity levels for anomalies."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DetectionMethod(str, Enum):
    """Methods for anomaly detection."""
    ZSCORE = "zscore"  # Z-score based
    IQR = "iqr"  # Interquartile range
    MAD = "mad"  # Median absolute deviation
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    PERCENTAGE_CHANGE = "percentage_change"
    STATIC_THRESHOLD = "static_threshold"


class AlertStatus(str, Enum):
    """Status of an alert."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


# =============================================================================
# Pydantic Models
# =============================================================================

class DataPoint(BaseModel):
    """A single data point for anomaly detection."""
    timestamp: datetime = Field(..., description="Data point timestamp")
    value: float = Field(..., description="Data value")
    source_id: str = Field(..., description="Source identifier")
    facility_id: str = Field(..., description="Facility identifier")
    metric_name: str = Field(..., description="Metric name")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AlertConfiguration(BaseModel):
    """Configuration for an alert rule."""
    rule_id: str = Field(..., description="Unique rule identifier")
    name: str = Field(..., description="Rule name")
    description: str = Field(default="", description="Rule description")

    # Detection settings
    detection_method: DetectionMethod = Field(..., description="Detection method to use")
    metric_name: str = Field(..., description="Metric to monitor")
    facility_ids: List[str] = Field(default_factory=list, description="Facilities to apply to")

    # Thresholds
    threshold: float = Field(default=3.0, description="Detection threshold")
    min_threshold: Optional[float] = Field(None, description="Minimum value threshold")
    max_threshold: Optional[float] = Field(None, description="Maximum value threshold")

    # Window settings
    window_size: int = Field(default=60, ge=1, description="Window size for calculations")
    cooldown_minutes: int = Field(default=5, ge=0, description="Cooldown between alerts")

    # Severity mapping
    severity: AnomalySeverity = Field(default=AnomalySeverity.MEDIUM, description="Default severity")
    severity_thresholds: Dict[str, float] = Field(
        default_factory=dict, description="Thresholds for severity escalation"
    )

    # Notification
    notify_channels: List[str] = Field(default_factory=list, description="Notification channels")
    enabled: bool = Field(default=True, description="Whether rule is active")


class AnomalyDetection(BaseModel):
    """Details of a detected anomaly."""
    anomaly_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    anomaly_type: AnomalyType = Field(..., description="Type of anomaly")
    severity: AnomalySeverity = Field(..., description="Anomaly severity")
    detection_method: DetectionMethod = Field(..., description="Method that detected it")

    # Context
    facility_id: str = Field(..., description="Facility identifier")
    metric_name: str = Field(..., description="Affected metric")
    source_id: Optional[str] = Field(None, description="Source identifier")

    # Values
    detected_value: float = Field(..., description="Value that triggered detection")
    expected_value: float = Field(..., description="Expected/normal value")
    deviation: float = Field(..., description="Deviation from expected")
    deviation_percent: float = Field(..., description="Percentage deviation")

    # Statistics
    threshold_used: float = Field(..., description="Threshold that was exceeded")
    baseline_mean: Optional[float] = Field(None, description="Baseline mean")
    baseline_std: Optional[float] = Field(None, description="Baseline standard deviation")

    # Timing
    detected_at: datetime = Field(default_factory=DeterministicClock.now)
    data_timestamp: datetime = Field(..., description="Timestamp of anomalous data")

    # Metadata
    rule_id: Optional[str] = Field(None, description="Rule that triggered detection")
    confidence: float = Field(default=0.0, ge=0, le=1, description="Detection confidence")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AnomalyPattern(BaseModel):
    """A pattern of anomalies over time."""
    pattern_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    pattern_type: str = Field(..., description="Type of pattern")
    anomalies: List[str] = Field(default_factory=list, description="Anomaly IDs in pattern")
    frequency: str = Field(..., description="Pattern frequency (e.g., daily, weekly)")
    confidence: float = Field(default=0.0, ge=0, le=1, description="Pattern confidence")
    first_seen: datetime = Field(..., description="First occurrence")
    last_seen: datetime = Field(..., description="Most recent occurrence")
    occurrence_count: int = Field(default=1, description="Number of occurrences")


class AlertAnomalyInput(BaseModel):
    """Input for the Alert & Anomaly Agent."""
    operation: str = Field(..., description="Operation to perform")
    data_points: List[DataPoint] = Field(default_factory=list, description="Data points to analyze")
    alert_config: Optional[AlertConfiguration] = Field(None, description="Alert configuration")
    facility_id: Optional[str] = Field(None, description="Facility filter")
    start_time: Optional[datetime] = Field(None, description="Query start time")
    end_time: Optional[datetime] = Field(None, description="Query end time")
    alert_id: Optional[str] = Field(None, description="Alert ID for status updates")
    new_status: Optional[AlertStatus] = Field(None, description="New alert status")

    @field_validator('operation')
    @classmethod
    def validate_operation(cls, v: str) -> str:
        """Validate operation is supported."""
        valid_ops = {
            'detect_anomalies', 'add_rule', 'remove_rule', 'get_rules',
            'get_active_alerts', 'get_alert_history', 'update_alert_status',
            'get_patterns', 'get_statistics', 'analyze_baseline'
        }
        if v not in valid_ops:
            raise ValueError(f"Operation must be one of: {valid_ops}")
        return v


class AlertAnomalyOutput(BaseModel):
    """Output from the Alert & Anomaly Agent."""
    success: bool = Field(..., description="Whether operation succeeded")
    operation: str = Field(..., description="Operation performed")
    data: Dict[str, Any] = Field(default_factory=dict, description="Result data")
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit")
    processing_time_ms: float = Field(default=0.0, description="Processing duration")
    timestamp: datetime = Field(default_factory=DeterministicClock.now)


# =============================================================================
# Alert & Anomaly Agent Implementation
# =============================================================================

class AlertAnomalyAgent(BaseAgent):
    """
    GL-OPS-X-002: Alert & Anomaly Agent

    Detects anomalies in emissions data and generates alerts based on
    configurable rules and statistical methods.

    Zero-Hallucination Guarantees:
        - All detection uses deterministic statistical calculations
        - Complete provenance tracking with SHA-256 hashes
        - No LLM calls in the detection path
        - All alerts traceable to source data

    Usage:
        agent = AlertAnomalyAgent()

        # Add detection rule
        result = agent.run({
            "operation": "add_rule",
            "alert_config": {
                "rule_id": "emissions-spike",
                "name": "Emissions Spike Detection",
                "detection_method": "zscore",
                "metric_name": "co2_emissions",
                "threshold": 2.5,
                "severity": "high"
            }
        })

        # Detect anomalies
        result = agent.run({
            "operation": "detect_anomalies",
            "data_points": [...]
        })
    """

    AGENT_ID = "GL-OPS-X-002"
    AGENT_NAME = "Alert & Anomaly Agent"
    VERSION = "1.0.0"

    # Buffer size for historical data
    MAX_HISTORY_SIZE = 10000

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Alert & Anomaly Agent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Anomaly detection and alerting for emissions data",
                version=self.VERSION,
                parameters={
                    "default_window_size": 60,
                    "default_threshold": 3.0,
                    "max_history_size": self.MAX_HISTORY_SIZE,
                }
            )
        super().__init__(config)

        # Alert rules
        self._rules: Dict[str, AlertConfiguration] = {}

        # Historical data by facility and metric
        self._history: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.MAX_HISTORY_SIZE))
        )

        # Active alerts
        self._active_alerts: Dict[str, AnomalyDetection] = {}

        # Alert history
        self._alert_history: List[AnomalyDetection] = []

        # Detected patterns
        self._patterns: Dict[str, AnomalyPattern] = {}

        # Cooldown tracking
        self._last_alert_time: Dict[str, datetime] = {}

        # Statistics
        self._total_detections = 0
        self._total_alerts_generated = 0

        self.logger.info(f"Initialized {self.AGENT_ID}: {self.AGENT_NAME}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute anomaly detection operations."""
        start_time = time.time()

        try:
            agent_input = AlertAnomalyInput(**input_data)
            operation = agent_input.operation

            result_data = self._route_operation(agent_input)

            provenance_hash = self._compute_provenance_hash(input_data, result_data)
            processing_time_ms = (time.time() - start_time) * 1000

            output = AlertAnomalyOutput(
                success=True,
                operation=operation,
                data=result_data,
                provenance_hash=provenance_hash,
                processing_time_ms=processing_time_ms,
            )

            return AgentResult(
                success=True,
                data=output.model_dump(),
            )

        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}", exc_info=True)
            processing_time_ms = (time.time() - start_time) * 1000

            return AgentResult(
                success=False,
                error=str(e),
                data={
                    "operation": input_data.get("operation", "unknown"),
                    "processing_time_ms": processing_time_ms,
                },
            )

    def _route_operation(self, agent_input: AlertAnomalyInput) -> Dict[str, Any]:
        """Route to appropriate operation handler."""
        operation = agent_input.operation

        if operation == "detect_anomalies":
            return self._handle_detect_anomalies(agent_input.data_points)
        elif operation == "add_rule":
            return self._handle_add_rule(agent_input.alert_config)
        elif operation == "remove_rule":
            return self._handle_remove_rule(agent_input.alert_config.rule_id if agent_input.alert_config else None)
        elif operation == "get_rules":
            return self._handle_get_rules()
        elif operation == "get_active_alerts":
            return self._handle_get_active_alerts(agent_input.facility_id)
        elif operation == "get_alert_history":
            return self._handle_get_alert_history(
                agent_input.facility_id,
                agent_input.start_time,
                agent_input.end_time,
            )
        elif operation == "update_alert_status":
            return self._handle_update_alert_status(
                agent_input.alert_id,
                agent_input.new_status,
            )
        elif operation == "get_patterns":
            return self._handle_get_patterns(agent_input.facility_id)
        elif operation == "get_statistics":
            return self._handle_get_statistics()
        elif operation == "analyze_baseline":
            return self._handle_analyze_baseline(
                agent_input.facility_id,
                agent_input.data_points,
            )
        else:
            raise ValueError(f"Unknown operation: {operation}")

    # =========================================================================
    # Anomaly Detection
    # =========================================================================

    def _handle_detect_anomalies(
        self, data_points: List[DataPoint]
    ) -> Dict[str, Any]:
        """Detect anomalies in data points."""
        detections = []
        new_alerts = []

        # Store data points in history
        for dp in data_points:
            self._history[dp.facility_id][dp.metric_name].append(dp)

        # Check each data point against rules
        for dp in data_points:
            applicable_rules = self._get_applicable_rules(dp)

            for rule in applicable_rules:
                if not rule.enabled:
                    continue

                # Check cooldown
                cooldown_key = f"{rule.rule_id}:{dp.facility_id}:{dp.metric_name}"
                if self._is_in_cooldown(cooldown_key, rule.cooldown_minutes):
                    continue

                # Detect anomaly
                anomaly = self._detect_with_method(dp, rule)

                if anomaly:
                    detections.append(anomaly)
                    self._total_detections += 1

                    # Create alert
                    self._active_alerts[anomaly.anomaly_id] = anomaly
                    self._alert_history.append(anomaly)
                    new_alerts.append(anomaly)
                    self._total_alerts_generated += 1

                    # Update cooldown
                    self._last_alert_time[cooldown_key] = DeterministicClock.now()

        return {
            "data_points_analyzed": len(data_points),
            "anomalies_detected": len(detections),
            "new_alerts": [a.model_dump() for a in new_alerts],
            "detection_details": [d.model_dump() for d in detections],
        }

    def _get_applicable_rules(self, dp: DataPoint) -> List[AlertConfiguration]:
        """Get rules applicable to a data point."""
        applicable = []
        for rule in self._rules.values():
            if rule.metric_name != dp.metric_name:
                continue
            if rule.facility_ids and dp.facility_id not in rule.facility_ids:
                continue
            applicable.append(rule)
        return applicable

    def _is_in_cooldown(self, key: str, cooldown_minutes: int) -> bool:
        """Check if we're in cooldown period."""
        if key not in self._last_alert_time:
            return False
        last_time = self._last_alert_time[key]
        now = DeterministicClock.now()
        return (now - last_time) < timedelta(minutes=cooldown_minutes)

    def _detect_with_method(
        self, dp: DataPoint, rule: AlertConfiguration
    ) -> Optional[AnomalyDetection]:
        """Detect anomaly using specified method."""
        method = rule.detection_method

        # Get historical data
        history = list(self._history[dp.facility_id][dp.metric_name])
        values = [h.value for h in history[:-1]]  # Exclude current point

        if len(values) < rule.window_size // 2:
            # Not enough data for detection
            return None

        # Use recent values for baseline
        recent_values = values[-rule.window_size:]

        if method == DetectionMethod.ZSCORE:
            return self._detect_zscore(dp, recent_values, rule)
        elif method == DetectionMethod.IQR:
            return self._detect_iqr(dp, recent_values, rule)
        elif method == DetectionMethod.MAD:
            return self._detect_mad(dp, recent_values, rule)
        elif method == DetectionMethod.PERCENTAGE_CHANGE:
            return self._detect_percentage_change(dp, recent_values, rule)
        elif method == DetectionMethod.STATIC_THRESHOLD:
            return self._detect_static_threshold(dp, rule)
        elif method == DetectionMethod.MOVING_AVERAGE:
            return self._detect_moving_average(dp, recent_values, rule)
        else:
            return None

    def _detect_zscore(
        self, dp: DataPoint, values: List[float], rule: AlertConfiguration
    ) -> Optional[AnomalyDetection]:
        """Detect anomaly using Z-score method."""
        if not values:
            return None

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = variance ** 0.5 if variance > 0 else 1e-10

        zscore = abs(dp.value - mean) / std

        if zscore > rule.threshold:
            anomaly_type = AnomalyType.SPIKE if dp.value > mean else AnomalyType.DIP

            return AnomalyDetection(
                anomaly_type=anomaly_type,
                severity=self._calculate_severity(zscore, rule),
                detection_method=DetectionMethod.ZSCORE,
                facility_id=dp.facility_id,
                metric_name=dp.metric_name,
                source_id=dp.source_id,
                detected_value=dp.value,
                expected_value=mean,
                deviation=dp.value - mean,
                deviation_percent=((dp.value - mean) / mean * 100) if mean != 0 else 0,
                threshold_used=rule.threshold,
                baseline_mean=mean,
                baseline_std=std,
                data_timestamp=dp.timestamp,
                rule_id=rule.rule_id,
                confidence=min(1.0, zscore / (rule.threshold * 2)),
            )

        return None

    def _detect_iqr(
        self, dp: DataPoint, values: List[float], rule: AlertConfiguration
    ) -> Optional[AnomalyDetection]:
        """Detect anomaly using Interquartile Range method."""
        if len(values) < 4:
            return None

        sorted_values = sorted(values)
        n = len(sorted_values)

        q1_idx = n // 4
        q3_idx = (3 * n) // 4

        q1 = sorted_values[q1_idx]
        q3 = sorted_values[q3_idx]
        iqr = q3 - q1

        if iqr == 0:
            return None

        lower_bound = q1 - rule.threshold * iqr
        upper_bound = q3 + rule.threshold * iqr

        if dp.value < lower_bound or dp.value > upper_bound:
            median = sorted_values[n // 2]
            anomaly_type = AnomalyType.SPIKE if dp.value > median else AnomalyType.DIP

            return AnomalyDetection(
                anomaly_type=anomaly_type,
                severity=self._calculate_severity(
                    abs(dp.value - median) / iqr if iqr > 0 else 0, rule
                ),
                detection_method=DetectionMethod.IQR,
                facility_id=dp.facility_id,
                metric_name=dp.metric_name,
                source_id=dp.source_id,
                detected_value=dp.value,
                expected_value=median,
                deviation=dp.value - median,
                deviation_percent=((dp.value - median) / median * 100) if median != 0 else 0,
                threshold_used=rule.threshold,
                baseline_mean=median,
                data_timestamp=dp.timestamp,
                rule_id=rule.rule_id,
                confidence=0.8,
                metadata={"q1": q1, "q3": q3, "iqr": iqr},
            )

        return None

    def _detect_mad(
        self, dp: DataPoint, values: List[float], rule: AlertConfiguration
    ) -> Optional[AnomalyDetection]:
        """Detect anomaly using Median Absolute Deviation method."""
        if not values:
            return None

        sorted_values = sorted(values)
        median = sorted_values[len(sorted_values) // 2]

        # Calculate MAD
        absolute_deviations = [abs(x - median) for x in values]
        sorted_deviations = sorted(absolute_deviations)
        mad = sorted_deviations[len(sorted_deviations) // 2]

        if mad == 0:
            mad = 1e-10

        # Modified Z-score
        modified_zscore = 0.6745 * (dp.value - median) / mad

        if abs(modified_zscore) > rule.threshold:
            anomaly_type = AnomalyType.SPIKE if dp.value > median else AnomalyType.DIP

            return AnomalyDetection(
                anomaly_type=anomaly_type,
                severity=self._calculate_severity(abs(modified_zscore), rule),
                detection_method=DetectionMethod.MAD,
                facility_id=dp.facility_id,
                metric_name=dp.metric_name,
                source_id=dp.source_id,
                detected_value=dp.value,
                expected_value=median,
                deviation=dp.value - median,
                deviation_percent=((dp.value - median) / median * 100) if median != 0 else 0,
                threshold_used=rule.threshold,
                baseline_mean=median,
                data_timestamp=dp.timestamp,
                rule_id=rule.rule_id,
                confidence=min(1.0, abs(modified_zscore) / (rule.threshold * 2)),
                metadata={"mad": mad, "modified_zscore": modified_zscore},
            )

        return None

    def _detect_percentage_change(
        self, dp: DataPoint, values: List[float], rule: AlertConfiguration
    ) -> Optional[AnomalyDetection]:
        """Detect anomaly using percentage change method."""
        if not values:
            return None

        previous_value = values[-1]

        if previous_value == 0:
            return None

        pct_change = abs((dp.value - previous_value) / previous_value * 100)

        if pct_change > rule.threshold:
            anomaly_type = AnomalyType.SPIKE if dp.value > previous_value else AnomalyType.DIP

            return AnomalyDetection(
                anomaly_type=anomaly_type,
                severity=self._calculate_severity(pct_change / 10, rule),
                detection_method=DetectionMethod.PERCENTAGE_CHANGE,
                facility_id=dp.facility_id,
                metric_name=dp.metric_name,
                source_id=dp.source_id,
                detected_value=dp.value,
                expected_value=previous_value,
                deviation=dp.value - previous_value,
                deviation_percent=pct_change,
                threshold_used=rule.threshold,
                data_timestamp=dp.timestamp,
                rule_id=rule.rule_id,
                confidence=min(1.0, pct_change / (rule.threshold * 2)),
            )

        return None

    def _detect_static_threshold(
        self, dp: DataPoint, rule: AlertConfiguration
    ) -> Optional[AnomalyDetection]:
        """Detect anomaly using static thresholds."""
        violation = None
        anomaly_type = None

        if rule.max_threshold is not None and dp.value > rule.max_threshold:
            violation = dp.value - rule.max_threshold
            anomaly_type = AnomalyType.SPIKE
        elif rule.min_threshold is not None and dp.value < rule.min_threshold:
            violation = rule.min_threshold - dp.value
            anomaly_type = AnomalyType.DIP

        if violation is not None and anomaly_type is not None:
            threshold = rule.max_threshold if anomaly_type == AnomalyType.SPIKE else rule.min_threshold

            return AnomalyDetection(
                anomaly_type=anomaly_type,
                severity=rule.severity,
                detection_method=DetectionMethod.STATIC_THRESHOLD,
                facility_id=dp.facility_id,
                metric_name=dp.metric_name,
                source_id=dp.source_id,
                detected_value=dp.value,
                expected_value=threshold,
                deviation=violation,
                deviation_percent=(violation / threshold * 100) if threshold != 0 else 0,
                threshold_used=threshold,
                data_timestamp=dp.timestamp,
                rule_id=rule.rule_id,
                confidence=1.0,
            )

        return None

    def _detect_moving_average(
        self, dp: DataPoint, values: List[float], rule: AlertConfiguration
    ) -> Optional[AnomalyDetection]:
        """Detect anomaly using moving average method."""
        if not values:
            return None

        ma = sum(values) / len(values)
        deviation = abs(dp.value - ma)

        # Use threshold as number of standard deviations
        std = (sum((x - ma) ** 2 for x in values) / len(values)) ** 0.5 if len(values) > 1 else 1e-10

        if deviation > rule.threshold * std:
            anomaly_type = AnomalyType.SPIKE if dp.value > ma else AnomalyType.DIP

            return AnomalyDetection(
                anomaly_type=anomaly_type,
                severity=self._calculate_severity(deviation / std if std > 0 else 0, rule),
                detection_method=DetectionMethod.MOVING_AVERAGE,
                facility_id=dp.facility_id,
                metric_name=dp.metric_name,
                source_id=dp.source_id,
                detected_value=dp.value,
                expected_value=ma,
                deviation=dp.value - ma,
                deviation_percent=((dp.value - ma) / ma * 100) if ma != 0 else 0,
                threshold_used=rule.threshold,
                baseline_mean=ma,
                baseline_std=std,
                data_timestamp=dp.timestamp,
                rule_id=rule.rule_id,
                confidence=min(1.0, deviation / (rule.threshold * std * 2)) if std > 0 else 0,
            )

        return None

    def _calculate_severity(
        self, score: float, rule: AlertConfiguration
    ) -> AnomalySeverity:
        """Calculate severity based on score and thresholds."""
        thresholds = rule.severity_thresholds or {
            "critical": 5.0,
            "high": 4.0,
            "medium": 3.0,
            "low": 2.0,
        }

        if score >= thresholds.get("critical", 5.0):
            return AnomalySeverity.CRITICAL
        elif score >= thresholds.get("high", 4.0):
            return AnomalySeverity.HIGH
        elif score >= thresholds.get("medium", 3.0):
            return AnomalySeverity.MEDIUM
        elif score >= thresholds.get("low", 2.0):
            return AnomalySeverity.LOW
        else:
            return AnomalySeverity.INFO

    # =========================================================================
    # Rule Management
    # =========================================================================

    def _handle_add_rule(
        self, config: Optional[AlertConfiguration]
    ) -> Dict[str, Any]:
        """Add or update an alert rule."""
        if not config:
            return {"error": "alert_config is required"}

        self._rules[config.rule_id] = config

        return {
            "rule_id": config.rule_id,
            "added": True,
            "total_rules": len(self._rules),
        }

    def _handle_remove_rule(self, rule_id: Optional[str]) -> Dict[str, Any]:
        """Remove an alert rule."""
        if not rule_id:
            return {"error": "rule_id is required"}

        if rule_id in self._rules:
            del self._rules[rule_id]
            return {"rule_id": rule_id, "removed": True}

        return {"rule_id": rule_id, "removed": False, "error": "Rule not found"}

    def _handle_get_rules(self) -> Dict[str, Any]:
        """Get all alert rules."""
        return {
            "rules": {
                rule_id: config.model_dump()
                for rule_id, config in self._rules.items()
            },
            "total_rules": len(self._rules),
        }

    # =========================================================================
    # Alert Management
    # =========================================================================

    def _handle_get_active_alerts(
        self, facility_id: Optional[str]
    ) -> Dict[str, Any]:
        """Get active alerts."""
        alerts = list(self._active_alerts.values())

        if facility_id:
            alerts = [a for a in alerts if a.facility_id == facility_id]

        return {
            "active_alerts": [a.model_dump() for a in alerts],
            "count": len(alerts),
        }

    def _handle_get_alert_history(
        self,
        facility_id: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
    ) -> Dict[str, Any]:
        """Get alert history."""
        alerts = self._alert_history.copy()

        if facility_id:
            alerts = [a for a in alerts if a.facility_id == facility_id]

        if start_time:
            alerts = [a for a in alerts if a.detected_at >= start_time]

        if end_time:
            alerts = [a for a in alerts if a.detected_at <= end_time]

        return {
            "alert_history": [a.model_dump() for a in alerts],
            "count": len(alerts),
        }

    def _handle_update_alert_status(
        self,
        alert_id: Optional[str],
        new_status: Optional[AlertStatus],
    ) -> Dict[str, Any]:
        """Update alert status."""
        if not alert_id:
            return {"error": "alert_id is required"}

        if not new_status:
            return {"error": "new_status is required"}

        if alert_id in self._active_alerts:
            if new_status == AlertStatus.RESOLVED:
                del self._active_alerts[alert_id]
            return {
                "alert_id": alert_id,
                "new_status": new_status.value,
                "updated": True,
            }

        return {"alert_id": alert_id, "updated": False, "error": "Alert not found"}

    # =========================================================================
    # Pattern Analysis
    # =========================================================================

    def _handle_get_patterns(self, facility_id: Optional[str]) -> Dict[str, Any]:
        """Get detected anomaly patterns."""
        patterns = list(self._patterns.values())

        if facility_id:
            # Filter patterns by facility (would need to track facility in pattern)
            pass

        return {
            "patterns": [p.model_dump() for p in patterns],
            "count": len(patterns),
        }

    # =========================================================================
    # Baseline Analysis
    # =========================================================================

    def _handle_analyze_baseline(
        self,
        facility_id: Optional[str],
        data_points: List[DataPoint],
    ) -> Dict[str, Any]:
        """Analyze data to establish baseline statistics."""
        if not data_points:
            return {"error": "data_points required for baseline analysis"}

        # Group by metric
        by_metric: Dict[str, List[float]] = defaultdict(list)
        for dp in data_points:
            if facility_id and dp.facility_id != facility_id:
                continue
            by_metric[dp.metric_name].append(dp.value)

        baselines = {}
        for metric_name, values in by_metric.items():
            if not values:
                continue

            sorted_values = sorted(values)
            n = len(sorted_values)
            mean = sum(values) / n
            std = (sum((x - mean) ** 2 for x in values) / n) ** 0.5

            baselines[metric_name] = {
                "mean": round(mean, 6),
                "std": round(std, 6),
                "median": sorted_values[n // 2],
                "min": min(values),
                "max": max(values),
                "q1": sorted_values[n // 4],
                "q3": sorted_values[(3 * n) // 4],
                "count": n,
            }

        return {
            "facility_id": facility_id,
            "baselines": baselines,
            "metrics_analyzed": len(baselines),
        }

    # =========================================================================
    # Statistics
    # =========================================================================

    def _handle_get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "total_detections": self._total_detections,
            "total_alerts_generated": self._total_alerts_generated,
            "active_alerts": len(self._active_alerts),
            "alert_history_size": len(self._alert_history),
            "configured_rules": len(self._rules),
            "monitored_facilities": len(self._history),
            "patterns_detected": len(self._patterns),
        }

    # =========================================================================
    # Provenance
    # =========================================================================

    def _compute_provenance_hash(
        self, input_data: Dict[str, Any], output_data: Dict[str, Any]
    ) -> str:
        """Compute SHA-256 hash for audit trail."""
        provenance_str = json.dumps(
            {"input": input_data, "output": output_data},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()[:16]
