"""
AlertOrchestrator - Centralized Alert Management for FurnacePulse

This module implements the AlertOrchestrator for the FurnacePulse furnace
monitoring system. It handles alert creation, taxonomy classification,
confidence scoring, owner assignment, escalation workflows, and de-duplication.

Alert Taxonomy (from playbook):
    - A-001: Hotspot Advisory - TMT approaching threshold
    - A-002: Hotspot Warning - Sustained TMT exceedance or accelerating rate-of-rise
    - A-003: Hotspot Urgent - High-confidence hotspot with risk to tube integrity
    - A-010: Efficiency Degradation - Efficiency/SFC deviation from baseline
    - A-020: Draft Instability - Draft variance/effort indicates control issue
    - A-030: Sensor Drift/Stuck - Signal fails drift/stuck tests

Example:
    >>> config = AlertOrchestratorConfig(...)
    >>> orchestrator = AlertOrchestrator(config)
    >>> alert = orchestrator.create_alert(
    ...     code=AlertCode.HOTSPOT_ADVISORY,
    ...     source_sensor="TMT-101",
    ...     current_value=875.0,
    ...     threshold=900.0
    ... )
    >>> orchestrator.process_alert(alert)
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class AlertCode(str, Enum):
    """Alert taxonomy codes as defined in the FurnacePulse playbook."""

    HOTSPOT_ADVISORY = "A-001"
    HOTSPOT_WARNING = "A-002"
    HOTSPOT_URGENT = "A-003"
    EFFICIENCY_DEGRADATION = "A-010"
    DRAFT_INSTABILITY = "A-020"
    SENSOR_DRIFT_STUCK = "A-030"


class AlertSeverity(str, Enum):
    """Alert severity levels for prioritization."""

    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class OwnerRole(str, Enum):
    """Owner roles responsible for alert response."""

    OPERATOR = "Operator"
    PROCESS_ENGINEER = "Process Engineer"
    RELIABILITY = "Reliability"
    SAFETY = "Safety"
    OT_CONTROLS = "OT/Controls"


class AlertStatus(str, Enum):
    """Alert lifecycle status."""

    NEW = "NEW"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    IN_PROGRESS = "IN_PROGRESS"
    ESCALATED = "ESCALATED"
    RESOLVED = "RESOLVED"
    SUPPRESSED = "SUPPRESSED"
    CLOSED = "CLOSED"


@dataclass
class AlertTaxonomyEntry:
    """Defines the taxonomy configuration for an alert code."""

    code: AlertCode
    name: str
    description: str
    default_severity: AlertSeverity
    primary_owner: OwnerRole
    secondary_owners: List[OwnerRole]
    escalation_timeout_minutes: int
    suppression_window_seconds: int
    requires_acknowledgement: bool
    auto_escalate: bool


# Alert taxonomy definitions from playbook
ALERT_TAXONOMY: Dict[AlertCode, AlertTaxonomyEntry] = {
    AlertCode.HOTSPOT_ADVISORY: AlertTaxonomyEntry(
        code=AlertCode.HOTSPOT_ADVISORY,
        name="Hotspot Advisory",
        description="TMT approaching threshold - early warning for potential hotspot",
        default_severity=AlertSeverity.LOW,
        primary_owner=OwnerRole.OPERATOR,
        secondary_owners=[OwnerRole.PROCESS_ENGINEER],
        escalation_timeout_minutes=30,
        suppression_window_seconds=300,
        requires_acknowledgement=False,
        auto_escalate=False,
    ),
    AlertCode.HOTSPOT_WARNING: AlertTaxonomyEntry(
        code=AlertCode.HOTSPOT_WARNING,
        name="Hotspot Warning",
        description="Sustained TMT exceedance or accelerating rate-of-rise",
        default_severity=AlertSeverity.MEDIUM,
        primary_owner=OwnerRole.OPERATOR,
        secondary_owners=[OwnerRole.PROCESS_ENGINEER, OwnerRole.RELIABILITY],
        escalation_timeout_minutes=15,
        suppression_window_seconds=180,
        requires_acknowledgement=True,
        auto_escalate=True,
    ),
    AlertCode.HOTSPOT_URGENT: AlertTaxonomyEntry(
        code=AlertCode.HOTSPOT_URGENT,
        name="Hotspot Urgent",
        description="High-confidence hotspot with risk to tube integrity",
        default_severity=AlertSeverity.CRITICAL,
        primary_owner=OwnerRole.OPERATOR,
        secondary_owners=[OwnerRole.PROCESS_ENGINEER, OwnerRole.RELIABILITY, OwnerRole.SAFETY],
        escalation_timeout_minutes=5,
        suppression_window_seconds=60,
        requires_acknowledgement=True,
        auto_escalate=True,
    ),
    AlertCode.EFFICIENCY_DEGRADATION: AlertTaxonomyEntry(
        code=AlertCode.EFFICIENCY_DEGRADATION,
        name="Efficiency Degradation",
        description="Efficiency/SFC deviation from baseline indicates performance issue",
        default_severity=AlertSeverity.MEDIUM,
        primary_owner=OwnerRole.PROCESS_ENGINEER,
        secondary_owners=[OwnerRole.OPERATOR, OwnerRole.RELIABILITY],
        escalation_timeout_minutes=60,
        suppression_window_seconds=600,
        requires_acknowledgement=True,
        auto_escalate=False,
    ),
    AlertCode.DRAFT_INSTABILITY: AlertTaxonomyEntry(
        code=AlertCode.DRAFT_INSTABILITY,
        name="Draft Instability",
        description="Draft variance/effort indicates control issue",
        default_severity=AlertSeverity.MEDIUM,
        primary_owner=OwnerRole.OT_CONTROLS,
        secondary_owners=[OwnerRole.OPERATOR, OwnerRole.PROCESS_ENGINEER],
        escalation_timeout_minutes=30,
        suppression_window_seconds=300,
        requires_acknowledgement=True,
        auto_escalate=True,
    ),
    AlertCode.SENSOR_DRIFT_STUCK: AlertTaxonomyEntry(
        code=AlertCode.SENSOR_DRIFT_STUCK,
        name="Sensor Drift/Stuck",
        description="Signal fails drift/stuck tests - data quality issue",
        default_severity=AlertSeverity.HIGH,
        primary_owner=OwnerRole.OT_CONTROLS,
        secondary_owners=[OwnerRole.RELIABILITY, OwnerRole.OPERATOR],
        escalation_timeout_minutes=15,
        suppression_window_seconds=120,
        requires_acknowledgement=True,
        auto_escalate=True,
    ),
}


class AlertContext(BaseModel):
    """Contextual information for an alert."""

    source_sensor: str = Field(..., description="Sensor tag that triggered the alert")
    current_value: float = Field(..., description="Current reading from the sensor")
    threshold: Optional[float] = Field(None, description="Threshold that was exceeded")
    baseline_value: Optional[float] = Field(None, description="Normal baseline value")
    rate_of_change: Optional[float] = Field(None, description="Rate of change (units/min)")
    duration_seconds: Optional[int] = Field(None, description="Duration of anomaly in seconds")
    affected_zone: Optional[str] = Field(None, description="Furnace zone affected")
    tube_id: Optional[str] = Field(None, description="Specific tube identifier if applicable")
    related_sensors: List[str] = Field(default_factory=list, description="Related sensor tags")
    additional_data: Dict[str, Any] = Field(default_factory=dict, description="Extra context")

    class Config:
        extra = "allow"


class ConfidenceScore(BaseModel):
    """Confidence scoring for alert validity."""

    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence 0-1")
    data_quality_score: float = Field(..., ge=0.0, le=1.0, description="Input data quality")
    pattern_match_score: float = Field(..., ge=0.0, le=1.0, description="Pattern detection confidence")
    model_agreement_score: float = Field(..., ge=0.0, le=1.0, description="Multi-model agreement")
    historical_correlation: float = Field(..., ge=0.0, le=1.0, description="Historical pattern match")
    explanation: str = Field(..., description="Human-readable confidence explanation")

    @validator("overall_score", pre=True, always=True)
    def compute_overall(cls, v, values):
        """Compute overall score as weighted average if not provided."""
        if v is not None:
            return v
        weights = {"data_quality_score": 0.25, "pattern_match_score": 0.35,
                   "model_agreement_score": 0.25, "historical_correlation": 0.15}
        total = sum(values.get(k, 0.5) * w for k, w in weights.items())
        return round(total, 3)


class Alert(BaseModel):
    """Core alert data model with full lifecycle tracking."""

    alert_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique alert ID")
    code: AlertCode = Field(..., description="Alert taxonomy code")
    severity: AlertSeverity = Field(..., description="Alert severity level")
    status: AlertStatus = Field(default=AlertStatus.NEW, description="Current lifecycle status")
    context: AlertContext = Field(..., description="Alert context and sensor data")
    confidence: ConfidenceScore = Field(..., description="Confidence scoring")
    primary_owner: OwnerRole = Field(..., description="Primary responsible role")
    secondary_owners: List[OwnerRole] = Field(default_factory=list, description="Backup owners")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = Field(None, description="When acknowledged")
    acknowledged_by: Optional[str] = Field(None, description="User who acknowledged")
    resolved_at: Optional[datetime] = Field(None, description="When resolved")
    resolved_by: Optional[str] = Field(None, description="User who resolved")
    escalation_level: int = Field(default=0, description="Current escalation level")
    escalated_at: Optional[datetime] = Field(None, description="When last escalated")
    suppression_key: str = Field(default="", description="Key for de-duplication")
    provenance_hash: str = Field(default="", description="SHA-256 for audit trail")
    notes: List[str] = Field(default_factory=list, description="Audit trail notes")

    class Config:
        use_enum_values = True

    def add_note(self, note: str, user: Optional[str] = None) -> None:
        """Add an audit note to the alert."""
        timestamp = datetime.utcnow().isoformat()
        user_str = f" by {user}" if user else ""
        self.notes.append(f"[{timestamp}]{user_str}: {note}")
        self.updated_at = datetime.utcnow()


class AlertOrchestratorConfig(BaseModel):
    """Configuration for the AlertOrchestrator."""

    enable_suppression: bool = Field(default=True, description="Enable alert de-duplication")
    enable_auto_escalation: bool = Field(default=True, description="Enable automatic escalation")
    suppression_window_multiplier: float = Field(default=1.0, ge=0.1, le=10.0)
    escalation_timeout_multiplier: float = Field(default=1.0, ge=0.1, le=10.0)
    min_confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    high_confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    max_active_alerts: int = Field(default=1000, ge=100, le=100000)
    retention_hours: int = Field(default=168, ge=1, description="Alert retention period")


@dataclass
class SuppressionEntry:
    """Tracks suppression state for de-duplication."""

    suppression_key: str
    first_seen: datetime
    last_seen: datetime
    count: int
    active_alert_id: str


class AlertOrchestrator:
    """
    Centralized alert management for FurnacePulse.

    This class handles the complete alert lifecycle including creation,
    classification, confidence scoring, owner assignment, escalation,
    and de-duplication.

    Attributes:
        config: Orchestrator configuration
        active_alerts: Dictionary of currently active alerts
        suppression_cache: Cache for de-duplication logic
        taxonomy: Alert taxonomy definitions

    Example:
        >>> config = AlertOrchestratorConfig()
        >>> orchestrator = AlertOrchestrator(config)
        >>> context = AlertContext(source_sensor="TMT-101", current_value=875.0)
        >>> confidence = orchestrator.calculate_confidence(AlertCode.HOTSPOT_ADVISORY, context)
        >>> alert = orchestrator.create_alert(AlertCode.HOTSPOT_ADVISORY, context, confidence)
    """

    def __init__(self, config: AlertOrchestratorConfig):
        """
        Initialize AlertOrchestrator.

        Args:
            config: Orchestrator configuration settings
        """
        self.config = config
        self.taxonomy = ALERT_TAXONOMY
        self.active_alerts: Dict[str, Alert] = {}
        self.suppression_cache: Dict[str, SuppressionEntry] = {}
        self._escalation_timers: Dict[str, datetime] = {}
        logger.info("AlertOrchestrator initialized with config: %s", config.json())

    def create_alert(
        self,
        code: AlertCode,
        context: AlertContext,
        confidence: Optional[ConfidenceScore] = None,
        override_severity: Optional[AlertSeverity] = None,
    ) -> Optional[Alert]:
        """
        Create a new alert with full taxonomy classification.

        Args:
            code: Alert taxonomy code
            context: Alert context with sensor data
            confidence: Pre-calculated confidence score (optional)
            override_severity: Override default severity (optional)

        Returns:
            Created Alert object, or None if suppressed

        Raises:
            ValueError: If alert code is invalid
        """
        start_time = datetime.utcnow()

        if code not in self.taxonomy:
            raise ValueError(f"Invalid alert code: {code}")

        taxonomy_entry = self.taxonomy[code]

        # Calculate confidence if not provided
        if confidence is None:
            confidence = self.calculate_confidence(code, context)

        # Check minimum confidence threshold
        if confidence.overall_score < self.config.min_confidence_threshold:
            logger.debug(
                "Alert %s suppressed due to low confidence: %.3f < %.3f",
                code.value, confidence.overall_score, self.config.min_confidence_threshold
            )
            return None

        # Generate suppression key for de-duplication
        suppression_key = self._generate_suppression_key(code, context)

        # Check for duplicate/suppression
        if self.config.enable_suppression:
            if self._should_suppress(suppression_key, taxonomy_entry):
                logger.debug("Alert %s suppressed (duplicate): %s", code.value, suppression_key)
                return None

        # Determine severity
        severity = override_severity or self._determine_severity(code, context, confidence)

        # Create the alert
        alert = Alert(
            code=code,
            severity=severity,
            context=context,
            confidence=confidence,
            primary_owner=taxonomy_entry.primary_owner,
            secondary_owners=taxonomy_entry.secondary_owners,
            suppression_key=suppression_key,
        )

        # Calculate provenance hash
        alert.provenance_hash = self._calculate_provenance_hash(alert)

        # Add to active alerts
        self.active_alerts[alert.alert_id] = alert

        # Update suppression cache
        self._update_suppression_cache(suppression_key, alert.alert_id)

        # Set escalation timer if needed
        if self.config.enable_auto_escalation and taxonomy_entry.auto_escalate:
            timeout = int(
                taxonomy_entry.escalation_timeout_minutes
                * self.config.escalation_timeout_multiplier
            )
            self._escalation_timers[alert.alert_id] = start_time + timedelta(minutes=timeout)

        processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info(
            "Alert created: %s [%s] severity=%s confidence=%.3f processing_time=%.2fms",
            alert.alert_id, code.value, severity.value, confidence.overall_score, processing_time_ms
        )

        return alert

    def calculate_confidence(self, code: AlertCode, context: AlertContext) -> ConfidenceScore:
        """
        Calculate confidence score for an alert.

        Uses deterministic scoring based on data quality, pattern matching,
        and historical correlation. No LLM involvement for numeric calculations.

        Args:
            code: Alert taxonomy code
            context: Alert context with sensor data

        Returns:
            ConfidenceScore with component scores and explanation
        """
        # Data quality score - based on sensor data validity
        data_quality = self._score_data_quality(context)

        # Pattern match score - how well the pattern matches the alert type
        pattern_match = self._score_pattern_match(code, context)

        # Model agreement - placeholder for multi-model consensus
        model_agreement = self._score_model_agreement(code, context)

        # Historical correlation - match to known historical patterns
        historical = self._score_historical_correlation(code, context)

        # Calculate overall score (weighted average)
        overall = (
            data_quality * 0.25
            + pattern_match * 0.35
            + model_agreement * 0.25
            + historical * 0.15
        )

        # Generate explanation
        explanation = self._generate_confidence_explanation(
            code, overall, data_quality, pattern_match, model_agreement, historical
        )

        return ConfidenceScore(
            overall_score=round(overall, 3),
            data_quality_score=round(data_quality, 3),
            pattern_match_score=round(pattern_match, 3),
            model_agreement_score=round(model_agreement, 3),
            historical_correlation=round(historical, 3),
            explanation=explanation,
        )

    def _score_data_quality(self, context: AlertContext) -> float:
        """Score data quality based on sensor reading validity."""
        score = 1.0

        # Penalize if value is at sensor limits (possibly stuck)
        if context.current_value <= 0 or context.current_value > 2000:
            score -= 0.3

        # Penalize if rate of change is extreme (possible spike)
        if context.rate_of_change is not None and abs(context.rate_of_change) > 50:
            score -= 0.2

        # Boost if we have related sensors confirming
        if len(context.related_sensors) > 0:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _score_pattern_match(self, code: AlertCode, context: AlertContext) -> float:
        """Score how well the pattern matches expected alert signature."""
        score = 0.5  # Default neutral score

        if code in (AlertCode.HOTSPOT_ADVISORY, AlertCode.HOTSPOT_WARNING, AlertCode.HOTSPOT_URGENT):
            # Hotspot alerts - score based on threshold proximity
            if context.threshold is not None and context.threshold > 0:
                ratio = context.current_value / context.threshold
                if ratio >= 1.0:
                    score = 0.9  # Threshold exceeded
                elif ratio >= 0.95:
                    score = 0.8  # Very close
                elif ratio >= 0.90:
                    score = 0.7  # Approaching
                else:
                    score = 0.5

            # Rate of rise increases confidence for warnings
            if code == AlertCode.HOTSPOT_WARNING and context.rate_of_change is not None:
                if context.rate_of_change > 5:  # Accelerating rise
                    score = min(1.0, score + 0.15)

        elif code == AlertCode.EFFICIENCY_DEGRADATION:
            # Efficiency - score based on deviation from baseline
            if context.baseline_value is not None and context.baseline_value > 0:
                deviation = abs(context.current_value - context.baseline_value) / context.baseline_value
                if deviation > 0.10:
                    score = 0.85
                elif deviation > 0.05:
                    score = 0.7
                else:
                    score = 0.5

        elif code == AlertCode.DRAFT_INSTABILITY:
            # Draft - variance-based scoring
            if context.duration_seconds is not None and context.duration_seconds > 60:
                score = 0.8  # Sustained instability
            else:
                score = 0.6

        elif code == AlertCode.SENSOR_DRIFT_STUCK:
            # Sensor issues - duration increases confidence
            if context.duration_seconds is not None:
                if context.duration_seconds > 300:
                    score = 0.9
                elif context.duration_seconds > 120:
                    score = 0.75
                else:
                    score = 0.6

        return score

    def _score_model_agreement(self, code: AlertCode, context: AlertContext) -> float:
        """Score based on multi-model agreement (placeholder for ML ensemble)."""
        # In production, this would aggregate votes from multiple models
        # For now, return a baseline score
        return 0.7

    def _score_historical_correlation(self, code: AlertCode, context: AlertContext) -> float:
        """Score based on historical pattern matching."""
        # In production, this would compare to historical alert patterns
        # For now, return a baseline score
        return 0.6

    def _generate_confidence_explanation(
        self,
        code: AlertCode,
        overall: float,
        data_quality: float,
        pattern_match: float,
        model_agreement: float,
        historical: float,
    ) -> str:
        """Generate human-readable confidence explanation."""
        level = "High" if overall >= 0.8 else "Medium" if overall >= 0.5 else "Low"
        taxonomy_name = self.taxonomy[code].name

        components = []
        if data_quality >= 0.8:
            components.append("strong data quality")
        elif data_quality < 0.5:
            components.append("data quality concerns")

        if pattern_match >= 0.8:
            components.append("clear pattern match")
        elif pattern_match < 0.5:
            components.append("weak pattern match")

        component_str = ", ".join(components) if components else "standard confidence"

        return f"{level} confidence ({overall:.1%}) for {taxonomy_name} based on {component_str}."

    def _determine_severity(
        self, code: AlertCode, context: AlertContext, confidence: ConfidenceScore
    ) -> AlertSeverity:
        """Determine alert severity based on context and confidence."""
        base_severity = self.taxonomy[code].default_severity

        # Upgrade severity for high confidence critical conditions
        if confidence.overall_score >= self.config.high_confidence_threshold:
            if code == AlertCode.HOTSPOT_WARNING:
                # Check for rapid escalation conditions
                if context.rate_of_change is not None and context.rate_of_change > 10:
                    return AlertSeverity.HIGH

        # Downgrade severity for low confidence
        if confidence.overall_score < 0.5:
            severity_order = [AlertSeverity.INFO, AlertSeverity.LOW, AlertSeverity.MEDIUM,
                             AlertSeverity.HIGH, AlertSeverity.CRITICAL]
            current_idx = severity_order.index(base_severity)
            if current_idx > 0:
                return severity_order[current_idx - 1]

        return base_severity

    def _generate_suppression_key(self, code: AlertCode, context: AlertContext) -> str:
        """Generate a unique key for alert de-duplication."""
        # Key based on alert type and source sensor
        key_parts = [
            code.value,
            context.source_sensor,
            context.affected_zone or "unknown_zone",
        ]
        return ":".join(key_parts)

    def _should_suppress(self, suppression_key: str, taxonomy_entry: AlertTaxonomyEntry) -> bool:
        """Check if an alert should be suppressed as a duplicate."""
        if suppression_key not in self.suppression_cache:
            return False

        entry = self.suppression_cache[suppression_key]
        window_seconds = int(
            taxonomy_entry.suppression_window_seconds
            * self.config.suppression_window_multiplier
        )
        window = timedelta(seconds=window_seconds)

        if datetime.utcnow() - entry.last_seen < window:
            # Update the existing entry
            entry.last_seen = datetime.utcnow()
            entry.count += 1
            logger.debug(
                "Suppressing duplicate alert (count=%d): %s",
                entry.count, suppression_key
            )
            return True

        return False

    def _update_suppression_cache(self, suppression_key: str, alert_id: str) -> None:
        """Update the suppression cache with a new alert."""
        now = datetime.utcnow()
        self.suppression_cache[suppression_key] = SuppressionEntry(
            suppression_key=suppression_key,
            first_seen=now,
            last_seen=now,
            count=1,
            active_alert_id=alert_id,
        )

    def _calculate_provenance_hash(self, alert: Alert) -> str:
        """Calculate SHA-256 hash for audit trail."""
        # Create a deterministic string representation
        provenance_str = (
            f"{alert.code.value}|{alert.severity.value}|"
            f"{alert.context.source_sensor}|{alert.context.current_value}|"
            f"{alert.created_at.isoformat()}|{alert.confidence.overall_score}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def acknowledge_alert(
        self, alert_id: str, user_id: str, notes: Optional[str] = None
    ) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert identifier
            user_id: User acknowledging the alert
            notes: Optional notes

        Returns:
            True if acknowledged successfully
        """
        if alert_id not in self.active_alerts:
            logger.warning("Cannot acknowledge unknown alert: %s", alert_id)
            return False

        alert = self.active_alerts[alert_id]
        if alert.status not in (AlertStatus.NEW, AlertStatus.ESCALATED):
            logger.warning("Cannot acknowledge alert in status: %s", alert.status)
            return False

        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.utcnow()
        alert.acknowledged_by = user_id
        alert.add_note(f"Acknowledged: {notes}" if notes else "Acknowledged", user_id)

        # Cancel escalation timer
        if alert_id in self._escalation_timers:
            del self._escalation_timers[alert_id]

        logger.info("Alert acknowledged: %s by %s", alert_id, user_id)
        return True

    def escalate_alert(self, alert_id: str, reason: Optional[str] = None) -> bool:
        """
        Escalate an alert to the next level.

        Args:
            alert_id: Alert identifier
            reason: Escalation reason

        Returns:
            True if escalated successfully
        """
        if alert_id not in self.active_alerts:
            logger.warning("Cannot escalate unknown alert: %s", alert_id)
            return False

        alert = self.active_alerts[alert_id]
        if alert.status in (AlertStatus.RESOLVED, AlertStatus.CLOSED, AlertStatus.SUPPRESSED):
            logger.warning("Cannot escalate alert in terminal status: %s", alert.status)
            return False

        alert.escalation_level += 1
        alert.escalated_at = datetime.utcnow()
        alert.status = AlertStatus.ESCALATED

        # Upgrade severity if not already critical
        severity_order = [AlertSeverity.INFO, AlertSeverity.LOW, AlertSeverity.MEDIUM,
                         AlertSeverity.HIGH, AlertSeverity.CRITICAL]
        current_idx = severity_order.index(AlertSeverity(alert.severity))
        if current_idx < len(severity_order) - 1:
            alert.severity = severity_order[current_idx + 1].value

        reason_str = f": {reason}" if reason else ""
        alert.add_note(f"Escalated to level {alert.escalation_level}{reason_str}")

        logger.warning(
            "Alert escalated: %s to level %d, severity=%s",
            alert_id, alert.escalation_level, alert.severity
        )
        return True

    def resolve_alert(
        self, alert_id: str, user_id: str, resolution_notes: str
    ) -> bool:
        """
        Resolve an alert.

        Args:
            alert_id: Alert identifier
            user_id: User resolving the alert
            resolution_notes: Notes on how the issue was resolved

        Returns:
            True if resolved successfully
        """
        if alert_id not in self.active_alerts:
            logger.warning("Cannot resolve unknown alert: %s", alert_id)
            return False

        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow()
        alert.resolved_by = user_id
        alert.add_note(f"Resolved: {resolution_notes}", user_id)

        # Clean up escalation timer
        if alert_id in self._escalation_timers:
            del self._escalation_timers[alert_id]

        logger.info("Alert resolved: %s by %s", alert_id, user_id)
        return True

    def process_escalation_timers(self) -> List[str]:
        """
        Process escalation timers and escalate overdue alerts.

        Returns:
            List of alert IDs that were escalated
        """
        now = datetime.utcnow()
        escalated_ids = []

        for alert_id, deadline in list(self._escalation_timers.items()):
            if now >= deadline:
                if self.escalate_alert(alert_id, "Auto-escalated due to timeout"):
                    escalated_ids.append(alert_id)

                    # Reset timer for next escalation level
                    alert = self.active_alerts.get(alert_id)
                    if alert:
                        taxonomy = self.taxonomy.get(AlertCode(alert.code))
                        if taxonomy:
                            timeout = int(
                                taxonomy.escalation_timeout_minutes
                                * self.config.escalation_timeout_multiplier
                            )
                            self._escalation_timers[alert_id] = now + timedelta(minutes=timeout)

        return escalated_ids

    def get_active_alerts(
        self,
        severity_filter: Optional[AlertSeverity] = None,
        owner_filter: Optional[OwnerRole] = None,
        status_filter: Optional[AlertStatus] = None,
    ) -> List[Alert]:
        """
        Get active alerts with optional filtering.

        Args:
            severity_filter: Filter by severity level
            owner_filter: Filter by owner role
            status_filter: Filter by status

        Returns:
            List of matching alerts
        """
        alerts = list(self.active_alerts.values())

        if severity_filter:
            alerts = [a for a in alerts if AlertSeverity(a.severity) == severity_filter]

        if owner_filter:
            alerts = [a for a in alerts if OwnerRole(a.primary_owner) == owner_filter]

        if status_filter:
            alerts = [a for a in alerts if AlertStatus(a.status) == status_filter]

        # Sort by severity (critical first) then by creation time
        severity_priority = {
            AlertSeverity.CRITICAL.value: 0,
            AlertSeverity.HIGH.value: 1,
            AlertSeverity.MEDIUM.value: 2,
            AlertSeverity.LOW.value: 3,
            AlertSeverity.INFO.value: 4,
        }
        alerts.sort(key=lambda a: (severity_priority.get(a.severity, 5), a.created_at))

        return alerts

    def cleanup_stale_alerts(self) -> int:
        """
        Clean up stale alerts beyond retention period.

        Returns:
            Number of alerts cleaned up
        """
        cutoff = datetime.utcnow() - timedelta(hours=self.config.retention_hours)
        stale_ids = [
            aid for aid, alert in self.active_alerts.items()
            if alert.status in (AlertStatus.RESOLVED, AlertStatus.CLOSED)
            and alert.updated_at < cutoff
        ]

        for alert_id in stale_ids:
            del self.active_alerts[alert_id]
            if alert_id in self._escalation_timers:
                del self._escalation_timers[alert_id]

        if stale_ids:
            logger.info("Cleaned up %d stale alerts", len(stale_ids))

        return len(stale_ids)

    def get_alert_statistics(self) -> Dict[str, Any]:
        """
        Get alert statistics for monitoring.

        Returns:
            Dictionary with alert statistics
        """
        stats = {
            "total_active": len(self.active_alerts),
            "by_severity": defaultdict(int),
            "by_status": defaultdict(int),
            "by_code": defaultdict(int),
            "by_owner": defaultdict(int),
            "suppression_cache_size": len(self.suppression_cache),
            "pending_escalations": len(self._escalation_timers),
        }

        for alert in self.active_alerts.values():
            stats["by_severity"][alert.severity] += 1
            stats["by_status"][alert.status] += 1
            stats["by_code"][alert.code] += 1
            stats["by_owner"][alert.primary_owner] += 1

        # Convert defaultdicts to regular dicts for serialization
        stats["by_severity"] = dict(stats["by_severity"])
        stats["by_status"] = dict(stats["by_status"])
        stats["by_code"] = dict(stats["by_code"])
        stats["by_owner"] = dict(stats["by_owner"])

        return stats
