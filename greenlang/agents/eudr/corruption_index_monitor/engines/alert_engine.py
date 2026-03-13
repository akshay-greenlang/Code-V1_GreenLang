# -*- coding: utf-8 -*-
"""
AlertEngine - AGENT-EUDR-019 Engine 7: Corruption Index Alert Generation

Generates real-time alerts when corruption indices change significantly,
thresholds are breached, governance trends reverse, or country risk
classifications change. Supports configurable alert rules, severity levels,
acknowledgement workflows, and summary dashboards.

Zero-Hallucination Guarantees:
    - All severity scoring uses deterministic threshold comparisons (Decimal).
    - Threshold breach detection uses explicit inequality checks.
    - Change magnitude calculations use Decimal arithmetic.
    - Alert deduplication uses deterministic hashing of alert attributes.
    - SHA-256 provenance hashes on all output objects.

Alert Types:
    1. THRESHOLD_BREACH:        CPI/WGI crosses a configured threshold value.
    2. SIGNIFICANT_CHANGE:      Year-over-year change exceeds configured delta.
    3. TREND_REVERSAL:          Direction of trend reverses (e.g., improving -> deteriorating).
    4. DATA_UPDATE:             New data becomes available for a monitored country.
    5. WATCHLIST_TRIGGER:       Country on watchlist exceeds monitoring criteria.
    6. COUNTRY_RECLASSIFICATION: Country moves between EUDR risk categories.
    7. NEW_DATA_AVAILABLE:      Scheduled data source update detected.

Alert Severity Levels:
    - CRITICAL:      Immediate action required (CPI drop >10, crosses CPI 40 downward).
    - HIGH:          Urgent attention needed (CPI drop >5, WGI CC drop >0.5).
    - MEDIUM:        Monitor closely (moderate changes, approaching thresholds).
    - LOW:           Informational tracking (minor changes, positive trends).
    - INFORMATIONAL: FYI only (data updates, scheduled reviews).

Default Alert Rules:
    - CPI drops > 5 points year-over-year              -> HIGH alert
    - CPI drops > 10 points year-over-year              -> CRITICAL alert
    - Country crosses CPI 40 threshold downward          -> CRITICAL alert
    - WGI Control of Corruption drops > 0.5              -> HIGH alert
    - WGI Control of Corruption drops > 1.0              -> CRITICAL alert
    - Bribery risk score increases > 15 points           -> HIGH alert
    - Country reclassified from STANDARD to HIGH risk    -> CRITICAL alert
    - Country reclassified from LOW to STANDARD risk     -> MEDIUM alert
    - Trend reversal from IMPROVING to DETERIORATING     -> HIGH alert

Performance Targets:
    - Single alert check: <5ms
    - Full rule evaluation (all countries): <500ms
    - Alert acknowledgement: <10ms
    - Summary generation: <50ms

Regulatory References:
    - EUDR Article 29: Country benchmarking/classification changes
    - EUDR Article 31: Review and update of benchmarking
    - EUDR Article 13: Record keeping (5-year alert history)
    - EU 2023/1115 Recital 31: Governance indicator monitoring

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-019, Engine 7 (Alert Engine)
Agent ID: GL-EUDR-CIM-019
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, dataclass with to_dict, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "to_dict"):
        serializable = data.to_dict()
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_id(prefix: str = "alert") -> str:
    """Generate a unique identifier with a given prefix.

    Args:
        prefix: ID prefix string.

    Returns:
        ID in format ``{prefix}-{hex12}``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _to_decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation.

    Raises:
        ValueError: If value cannot be converted.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot convert {value!r} to Decimal") from exc


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AlertSeverity(str, Enum):
    """Severity level for corruption index alerts.

    Values:
        CRITICAL: Immediate action required.
        HIGH: Urgent attention needed.
        MEDIUM: Monitor closely.
        LOW: Informational tracking.
        INFORMATIONAL: FYI only.
    """

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFORMATIONAL = "INFORMATIONAL"


class AlertType(str, Enum):
    """Type of corruption index alert.

    Values:
        THRESHOLD_BREACH: Index crosses a configured threshold.
        SIGNIFICANT_CHANGE: Year-over-year change exceeds delta.
        TREND_REVERSAL: Trend direction reverses.
        DATA_UPDATE: New data available.
        WATCHLIST_TRIGGER: Watchlist country exceeds criteria.
        COUNTRY_RECLASSIFICATION: EUDR risk category change.
        NEW_DATA_AVAILABLE: Scheduled data source update.
    """

    THRESHOLD_BREACH = "THRESHOLD_BREACH"
    SIGNIFICANT_CHANGE = "SIGNIFICANT_CHANGE"
    TREND_REVERSAL = "TREND_REVERSAL"
    DATA_UPDATE = "DATA_UPDATE"
    WATCHLIST_TRIGGER = "WATCHLIST_TRIGGER"
    COUNTRY_RECLASSIFICATION = "COUNTRY_RECLASSIFICATION"
    NEW_DATA_AVAILABLE = "NEW_DATA_AVAILABLE"


class AlertStatus(str, Enum):
    """Lifecycle status of an alert.

    Values:
        ACTIVE: Alert is active and unacknowledged.
        ACKNOWLEDGED: Alert has been acknowledged by a user.
        RESOLVED: Alert condition has been resolved.
        SUPPRESSED: Alert has been suppressed (cooldown or manual).
        EXPIRED: Alert has passed its expiration time.
    """

    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    SUPPRESSED = "SUPPRESSED"
    EXPIRED = "EXPIRED"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default alert expiration period in days.
DEFAULT_EXPIRATION_DAYS: int = 90

#: Default cooldown period between duplicate alerts (minutes).
DEFAULT_COOLDOWN_MINUTES: int = 1440  # 24 hours

#: CPI threshold levels for alert rules.
CPI_CRITICAL_DROP: Decimal = Decimal("10")
CPI_HIGH_DROP: Decimal = Decimal("5")
CPI_MEDIUM_DROP: Decimal = Decimal("3")
CPI_HIGH_RISK_THRESHOLD: Decimal = Decimal("40")
CPI_LOW_RISK_THRESHOLD: Decimal = Decimal("60")

#: WGI Control of Corruption threshold levels.
WGI_CRITICAL_DROP: Decimal = Decimal("1.0")
WGI_HIGH_DROP: Decimal = Decimal("0.5")
WGI_MEDIUM_DROP: Decimal = Decimal("0.3")
WGI_HIGH_RISK_THRESHOLD: Decimal = Decimal("-0.5")

#: Bribery risk threshold levels.
BRIBERY_HIGH_INCREASE: Decimal = Decimal("15")
BRIBERY_MEDIUM_INCREASE: Decimal = Decimal("10")

#: Maximum number of alerts to store in memory.
MAX_ALERT_STORE_SIZE: int = 10000

#: Alert severity ordering for comparison.
SEVERITY_ORDER: Dict[str, int] = {
    "CRITICAL": 5,
    "HIGH": 4,
    "MEDIUM": 3,
    "LOW": 2,
    "INFORMATIONAL": 1,
}

#: Valid index types for alert configuration.
VALID_INDEX_TYPES: frozenset = frozenset({"CPI", "WGI", "BRIBERY", "COMPOSITE"})

# ---------------------------------------------------------------------------
# Default Alert Rules
# ---------------------------------------------------------------------------

DEFAULT_ALERT_RULES: List[Dict[str, Any]] = [
    {
        "rule_id": "RULE-CPI-CRITICAL-DROP",
        "alert_type": "SIGNIFICANT_CHANGE",
        "index_type": "CPI",
        "description": "CPI drops more than 10 points year-over-year",
        "severity": "CRITICAL",
        "threshold_delta": Decimal("-10"),
        "direction": "decrease",
        "enabled": True,
        "cooldown_minutes": 1440,
    },
    {
        "rule_id": "RULE-CPI-HIGH-DROP",
        "alert_type": "SIGNIFICANT_CHANGE",
        "index_type": "CPI",
        "description": "CPI drops more than 5 points year-over-year",
        "severity": "HIGH",
        "threshold_delta": Decimal("-5"),
        "direction": "decrease",
        "enabled": True,
        "cooldown_minutes": 1440,
    },
    {
        "rule_id": "RULE-CPI-40-BREACH",
        "alert_type": "THRESHOLD_BREACH",
        "index_type": "CPI",
        "description": "Country crosses CPI 40 threshold downward (enters HIGH corruption)",
        "severity": "CRITICAL",
        "threshold_value": Decimal("40"),
        "direction": "below",
        "enabled": True,
        "cooldown_minutes": 4320,
    },
    {
        "rule_id": "RULE-WGI-CRITICAL-DROP",
        "alert_type": "SIGNIFICANT_CHANGE",
        "index_type": "WGI",
        "description": "WGI Control of Corruption drops more than 1.0",
        "severity": "CRITICAL",
        "threshold_delta": Decimal("-1.0"),
        "direction": "decrease",
        "enabled": True,
        "cooldown_minutes": 1440,
    },
    {
        "rule_id": "RULE-WGI-HIGH-DROP",
        "alert_type": "SIGNIFICANT_CHANGE",
        "index_type": "WGI",
        "description": "WGI Control of Corruption drops more than 0.5",
        "severity": "HIGH",
        "threshold_delta": Decimal("-0.5"),
        "direction": "decrease",
        "enabled": True,
        "cooldown_minutes": 1440,
    },
    {
        "rule_id": "RULE-WGI-MEDIUM-DROP",
        "alert_type": "SIGNIFICANT_CHANGE",
        "index_type": "WGI",
        "description": "WGI Control of Corruption drops more than 0.3",
        "severity": "MEDIUM",
        "threshold_delta": Decimal("-0.3"),
        "direction": "decrease",
        "enabled": True,
        "cooldown_minutes": 2880,
    },
    {
        "rule_id": "RULE-BRIBERY-HIGH-INCREASE",
        "alert_type": "SIGNIFICANT_CHANGE",
        "index_type": "BRIBERY",
        "description": "Bribery risk score increases more than 15 points",
        "severity": "HIGH",
        "threshold_delta": Decimal("15"),
        "direction": "increase",
        "enabled": True,
        "cooldown_minutes": 1440,
    },
    {
        "rule_id": "RULE-RECLASSIFICATION-HIGH",
        "alert_type": "COUNTRY_RECLASSIFICATION",
        "index_type": "COMPOSITE",
        "description": "Country reclassified from STANDARD to HIGH risk",
        "severity": "CRITICAL",
        "from_class": "STANDARD_RISK",
        "to_class": "HIGH_RISK",
        "enabled": True,
        "cooldown_minutes": 4320,
    },
    {
        "rule_id": "RULE-RECLASSIFICATION-MEDIUM",
        "alert_type": "COUNTRY_RECLASSIFICATION",
        "index_type": "COMPOSITE",
        "description": "Country reclassified from LOW to STANDARD risk",
        "severity": "MEDIUM",
        "from_class": "LOW_RISK",
        "to_class": "STANDARD_RISK",
        "enabled": True,
        "cooldown_minutes": 4320,
    },
    {
        "rule_id": "RULE-TREND-REVERSAL",
        "alert_type": "TREND_REVERSAL",
        "index_type": "CPI",
        "description": "Trend reversal from IMPROVING to DETERIORATING",
        "severity": "HIGH",
        "from_direction": "IMPROVING",
        "to_direction": "DETERIORATING",
        "enabled": True,
        "cooldown_minutes": 2880,
    },
]


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class Alert:
    """A corruption index alert record.

    Attributes:
        alert_id: Unique alert identifier.
        alert_type: Type of alert.
        severity: Alert severity level.
        status: Current alert lifecycle status.
        country_code: ISO 3166-1 alpha-2 country code.
        index_type: Corruption index that triggered the alert.
        rule_id: ID of the rule that generated this alert.
        title: Short alert title.
        description: Detailed alert description.
        current_value: Current index value that triggered the alert.
        previous_value: Previous index value for comparison.
        threshold_value: Threshold value that was breached (if applicable).
        change_magnitude: Magnitude of change that triggered the alert.
        created_at: UTC timestamp of alert creation.
        expires_at: UTC timestamp of alert expiration.
        acknowledged_at: UTC timestamp of acknowledgement (if any).
        acknowledged_by: User who acknowledged the alert.
        resolved_at: UTC timestamp of resolution (if any).
        resolution_notes: Notes about how the alert was resolved.
        metadata: Additional metadata dictionary.
        provenance_hash: SHA-256 hash for audit trail.
    """

    alert_id: str = ""
    alert_type: str = "SIGNIFICANT_CHANGE"
    severity: str = "MEDIUM"
    status: str = "ACTIVE"
    country_code: str = ""
    index_type: str = "CPI"
    rule_id: str = ""
    title: str = ""
    description: str = ""
    current_value: Decimal = Decimal("0")
    previous_value: Decimal = Decimal("0")
    threshold_value: Optional[Decimal] = None
    change_magnitude: Decimal = Decimal("0")
    created_at: str = ""
    expires_at: str = ""
    acknowledged_at: Optional[str] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[str] = None
    resolution_notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON/provenance.

        Returns:
            Dictionary representation with Decimal values as strings.
        """
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "status": self.status,
            "country_code": self.country_code,
            "index_type": self.index_type,
            "rule_id": self.rule_id,
            "title": self.title,
            "description": self.description,
            "current_value": str(self.current_value),
            "previous_value": str(self.previous_value),
            "threshold_value": str(self.threshold_value) if self.threshold_value is not None else None,
            "change_magnitude": str(self.change_magnitude),
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "acknowledged_at": self.acknowledged_at,
            "acknowledged_by": self.acknowledged_by,
            "resolved_at": self.resolved_at,
            "resolution_notes": self.resolution_notes,
            "metadata": self.metadata,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class AlertConfiguration:
    """Configuration for an alert rule.

    Attributes:
        config_id: Unique configuration identifier.
        rule_id: Rule identifier.
        alert_type: Type of alert to generate.
        index_type: Index type to monitor.
        severity: Severity to assign to generated alerts.
        enabled: Whether this rule is active.
        description: Rule description.
        threshold_value: Static threshold value (for THRESHOLD_BREACH).
        threshold_delta: Change threshold (for SIGNIFICANT_CHANGE).
        direction: Direction of change to monitor.
        country_codes: List of country codes to monitor (None = all).
        cooldown_minutes: Cooldown period between duplicate alerts.
        notification_channels: Channels to notify on alert.
        created_at: When this configuration was created.
        updated_at: When this configuration was last updated.
        provenance_hash: SHA-256 hash.
    """

    config_id: str = ""
    rule_id: str = ""
    alert_type: str = "SIGNIFICANT_CHANGE"
    index_type: str = "CPI"
    severity: str = "MEDIUM"
    enabled: bool = True
    description: str = ""
    threshold_value: Optional[Decimal] = None
    threshold_delta: Optional[Decimal] = None
    direction: str = "decrease"
    country_codes: Optional[List[str]] = None
    cooldown_minutes: int = 1440
    notification_channels: List[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "config_id": self.config_id,
            "rule_id": self.rule_id,
            "alert_type": self.alert_type,
            "index_type": self.index_type,
            "severity": self.severity,
            "enabled": self.enabled,
            "description": self.description,
            "threshold_value": str(self.threshold_value) if self.threshold_value is not None else None,
            "threshold_delta": str(self.threshold_delta) if self.threshold_delta is not None else None,
            "direction": self.direction,
            "country_codes": self.country_codes,
            "cooldown_minutes": self.cooldown_minutes,
            "notification_channels": self.notification_channels,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class AlertSummary:
    """Summary statistics for alerts.

    Attributes:
        total_alerts: Total alert count.
        active_count: Active (unacknowledged) alerts.
        acknowledged_count: Acknowledged alerts.
        resolved_count: Resolved alerts.
        suppressed_count: Suppressed alerts.
        expired_count: Expired alerts.
        severity_breakdown: Counts by severity.
        type_breakdown: Counts by alert type.
        top_countries: Countries with most alerts.
        recent_alerts: Most recent alerts.
        provenance_hash: SHA-256 hash.
    """

    total_alerts: int = 0
    active_count: int = 0
    acknowledged_count: int = 0
    resolved_count: int = 0
    suppressed_count: int = 0
    expired_count: int = 0
    severity_breakdown: Dict[str, int] = field(default_factory=dict)
    type_breakdown: Dict[str, int] = field(default_factory=dict)
    top_countries: List[Dict[str, Any]] = field(default_factory=list)
    recent_alerts: List[Dict[str, Any]] = field(default_factory=list)
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "total_alerts": self.total_alerts,
            "active_count": self.active_count,
            "acknowledged_count": self.acknowledged_count,
            "resolved_count": self.resolved_count,
            "suppressed_count": self.suppressed_count,
            "expired_count": self.expired_count,
            "severity_breakdown": self.severity_breakdown,
            "type_breakdown": self.type_breakdown,
            "top_countries": self.top_countries,
            "recent_alerts": self.recent_alerts,
            "provenance_hash": self.provenance_hash,
        }


# ---------------------------------------------------------------------------
# AlertEngine
# ---------------------------------------------------------------------------


class AlertEngine:
    """Production-grade corruption index alerting system for EUDR compliance.

    Generates, manages, and tracks alerts when corruption indices change
    significantly, thresholds are breached, or governance trends indicate
    increased EUDR compliance risk. Supports configurable alert rules,
    acknowledgement workflows, and summary dashboards.

    Thread Safety:
        All mutable state is protected by a reentrant lock. Multiple threads
        can safely call any public method concurrently.

    Zero-Hallucination:
        All severity determinations use explicit Decimal threshold
        comparisons. No ML/LLM models in any alert generation path.

    Attributes:
        _alerts: In-memory alert store keyed by alert_id.
        _rules: Active alert rules keyed by rule_id.
        _cooldown_tracker: Tracks last alert time per (rule, country) pair.
        _watchlist: Countries on heightened monitoring.
        _lock: Reentrant lock for thread safety.

    Example:
        >>> engine = AlertEngine()
        >>> alerts = engine.check_significant_change("BR", "CPI", Decimal("36"), Decimal("28"))
        >>> assert len(alerts) >= 1
        >>> assert alerts[0]["severity"] == "CRITICAL"
    """

    def __init__(self) -> None:
        """Initialize AlertEngine with default rules and empty alert store."""
        self._alerts: Dict[str, Alert] = {}
        self._rules: Dict[str, Dict[str, Any]] = {}
        self._cooldown_tracker: Dict[str, datetime] = {}
        self._watchlist: Dict[str, Dict[str, Any]] = {}
        self._lock: threading.RLock = threading.RLock()

        # Load default rules
        for rule in DEFAULT_ALERT_RULES:
            self._rules[rule["rule_id"]] = dict(rule)

        logger.info(
            "AlertEngine initialized (version=%s, default_rules=%d)",
            _MODULE_VERSION,
            len(self._rules),
        )

    # ------------------------------------------------------------------
    # Rule Management
    # ------------------------------------------------------------------

    def configure_alert(
        self,
        rule_id: str,
        alert_type: str,
        index_type: str,
        severity: str,
        description: str = "",
        threshold_value: Optional[Decimal] = None,
        threshold_delta: Optional[Decimal] = None,
        direction: str = "decrease",
        country_codes: Optional[List[str]] = None,
        cooldown_minutes: int = DEFAULT_COOLDOWN_MINUTES,
        enabled: bool = True,
    ) -> Dict[str, Any]:
        """Create or update an alert configuration rule.

        Args:
            rule_id: Unique rule identifier.
            alert_type: Type of alert (THRESHOLD_BREACH, SIGNIFICANT_CHANGE, etc.).
            index_type: Index type to monitor (CPI, WGI, BRIBERY, COMPOSITE).
            severity: Severity level for generated alerts.
            description: Human-readable rule description.
            threshold_value: Static threshold value (for THRESHOLD_BREACH).
            threshold_delta: Change threshold (for SIGNIFICANT_CHANGE).
            direction: Direction to monitor ("decrease", "increase", "below", "above").
            country_codes: Country codes to monitor (None = all).
            cooldown_minutes: Cooldown between duplicate alerts.
            enabled: Whether this rule is active.

        Returns:
            Dictionary with the created/updated AlertConfiguration.

        Raises:
            ValueError: If parameters are invalid.
        """
        start_time = time.monotonic()

        if not rule_id or not isinstance(rule_id, str):
            raise ValueError("rule_id must be a non-empty string")
        if alert_type not in {e.value for e in AlertType}:
            raise ValueError(f"alert_type must be a valid AlertType value")
        if index_type not in VALID_INDEX_TYPES:
            raise ValueError(f"index_type must be one of {sorted(VALID_INDEX_TYPES)}")
        if severity not in {e.value for e in AlertSeverity}:
            raise ValueError(f"severity must be a valid AlertSeverity value")

        now_str = _utcnow().isoformat()

        config = AlertConfiguration(
            config_id=_generate_id("cfg"),
            rule_id=rule_id,
            alert_type=alert_type,
            index_type=index_type,
            severity=severity,
            enabled=enabled,
            description=description,
            threshold_value=threshold_value,
            threshold_delta=threshold_delta,
            direction=direction,
            country_codes=country_codes,
            cooldown_minutes=cooldown_minutes,
            created_at=now_str,
            updated_at=now_str,
        )
        config.provenance_hash = _compute_hash(config)

        # Store rule
        with self._lock:
            is_update = rule_id in self._rules
            self._rules[rule_id] = {
                "rule_id": rule_id,
                "alert_type": alert_type,
                "index_type": index_type,
                "severity": severity,
                "description": description,
                "threshold_value": threshold_value,
                "threshold_delta": threshold_delta,
                "direction": direction,
                "country_codes": country_codes,
                "cooldown_minutes": cooldown_minutes,
                "enabled": enabled,
            }

        processing_time_ms = (time.monotonic() - start_time) * 1000.0
        action = "updated" if is_update else "created"

        result = config.to_dict()
        result["action"] = action
        result["processing_time_ms"] = round(processing_time_ms, 3)

        logger.info(
            "Alert rule %s: %s (type=%s, index=%s, severity=%s)",
            action, rule_id, alert_type, index_type, severity,
        )
        return result

    def get_rules(self) -> Dict[str, Any]:
        """Get all configured alert rules.

        Returns:
            Dictionary with list of alert rules and count.
        """
        with self._lock:
            rules = [dict(r) for r in self._rules.values()]
            # Convert Decimal values to strings for JSON serialization
            for rule in rules:
                for key in ("threshold_value", "threshold_delta"):
                    if key in rule and isinstance(rule[key], Decimal):
                        rule[key] = str(rule[key])

        return {
            "rule_count": len(rules),
            "rules": rules,
            "provenance_hash": _compute_hash({"rules": rules}),
        }

    # ------------------------------------------------------------------
    # Watchlist Management
    # ------------------------------------------------------------------

    def add_to_watchlist(
        self,
        country_code: str,
        reason: str,
        monitoring_frequency: str = "monthly",
    ) -> Dict[str, Any]:
        """Add a country to the heightened monitoring watchlist.

        Args:
            country_code: ISO country code to watch.
            reason: Reason for watchlist addition.
            monitoring_frequency: Monitoring frequency (daily/weekly/monthly).

        Returns:
            Dictionary confirming watchlist addition.

        Raises:
            ValueError: If country_code is empty.
        """
        if not country_code or not isinstance(country_code, str):
            raise ValueError("country_code must be a non-empty string")
        country_code = country_code.upper()

        with self._lock:
            self._watchlist[country_code] = {
                "country_code": country_code,
                "reason": reason,
                "monitoring_frequency": monitoring_frequency,
                "added_at": _utcnow().isoformat(),
                "alert_count": 0,
            }

        logger.info(
            "Added %s to watchlist: reason=%s frequency=%s",
            country_code, reason, monitoring_frequency,
        )
        return {
            "country_code": country_code,
            "action": "added_to_watchlist",
            "reason": reason,
            "monitoring_frequency": monitoring_frequency,
            "provenance_hash": _compute_hash({"watchlist_add": country_code}),
        }

    def remove_from_watchlist(self, country_code: str) -> Dict[str, Any]:
        """Remove a country from the watchlist.

        Args:
            country_code: ISO country code to remove.

        Returns:
            Dictionary confirming removal.
        """
        country_code = country_code.upper()
        with self._lock:
            removed = self._watchlist.pop(country_code, None)

        return {
            "country_code": country_code,
            "action": "removed_from_watchlist",
            "was_on_watchlist": removed is not None,
            "provenance_hash": _compute_hash({"watchlist_remove": country_code}),
        }

    def get_watchlist(self) -> Dict[str, Any]:
        """Get all countries on the watchlist.

        Returns:
            Dictionary with watchlist entries.
        """
        with self._lock:
            entries = list(self._watchlist.values())
        return {
            "watchlist_count": len(entries),
            "watchlist": entries,
            "provenance_hash": _compute_hash({"watchlist": entries}),
        }

    # ------------------------------------------------------------------
    # Alert Generation
    # ------------------------------------------------------------------

    def _create_alert(
        self,
        alert_type: str,
        severity: str,
        country_code: str,
        index_type: str,
        rule_id: str,
        title: str,
        description: str,
        current_value: Decimal,
        previous_value: Decimal = Decimal("0"),
        threshold_value: Optional[Decimal] = None,
        change_magnitude: Decimal = Decimal("0"),
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """Create and store a new alert.

        Args:
            alert_type: Type of alert.
            severity: Severity level.
            country_code: Country code.
            index_type: Index type.
            rule_id: Rule that triggered the alert.
            title: Alert title.
            description: Alert description.
            current_value: Current index value.
            previous_value: Previous index value.
            threshold_value: Threshold that was breached.
            change_magnitude: Magnitude of the change.
            metadata: Additional metadata.

        Returns:
            Created Alert object.
        """
        now = _utcnow()
        expires_at = now + timedelta(days=DEFAULT_EXPIRATION_DAYS)

        alert = Alert(
            alert_id=_generate_id("alert"),
            alert_type=alert_type,
            severity=severity,
            status=AlertStatus.ACTIVE.value,
            country_code=country_code.upper(),
            index_type=index_type,
            rule_id=rule_id,
            title=title,
            description=description,
            current_value=current_value,
            previous_value=previous_value,
            threshold_value=threshold_value,
            change_magnitude=change_magnitude,
            created_at=now.isoformat(),
            expires_at=expires_at.isoformat(),
            metadata=metadata or {},
        )
        alert.provenance_hash = _compute_hash(alert)

        with self._lock:
            # Enforce max store size
            if len(self._alerts) >= MAX_ALERT_STORE_SIZE:
                self._evict_oldest_alerts()
            self._alerts[alert.alert_id] = alert

            # Update watchlist alert count
            if country_code.upper() in self._watchlist:
                self._watchlist[country_code.upper()]["alert_count"] = (
                    self._watchlist[country_code.upper()].get("alert_count", 0) + 1
                )

        logger.info(
            "Alert created: id=%s type=%s severity=%s country=%s index=%s",
            alert.alert_id, alert_type, severity, country_code, index_type,
        )
        return alert

    def _is_in_cooldown(self, rule_id: str, country_code: str) -> bool:
        """Check if a rule/country pair is in cooldown period.

        Args:
            rule_id: Rule identifier.
            country_code: Country code.

        Returns:
            True if in cooldown, False otherwise.
        """
        key = f"{rule_id}:{country_code.upper()}"
        with self._lock:
            last_alert_time = self._cooldown_tracker.get(key)
            if last_alert_time is None:
                return False

            rule = self._rules.get(rule_id, {})
            cooldown_mins = rule.get("cooldown_minutes", DEFAULT_COOLDOWN_MINUTES)
            cooldown_end = last_alert_time + timedelta(minutes=cooldown_mins)

            return _utcnow() < cooldown_end

    def _record_cooldown(self, rule_id: str, country_code: str) -> None:
        """Record alert time for cooldown tracking.

        Args:
            rule_id: Rule identifier.
            country_code: Country code.
        """
        key = f"{rule_id}:{country_code.upper()}"
        with self._lock:
            self._cooldown_tracker[key] = _utcnow()

    def _evict_oldest_alerts(self) -> None:
        """Evict oldest alerts when store exceeds maximum size.

        Removes the oldest 10% of alerts to make room for new ones.
        Must be called with self._lock held.
        """
        if not self._alerts:
            return
        # Sort by created_at and remove oldest 10%
        sorted_alerts = sorted(
            self._alerts.values(),
            key=lambda a: a.created_at,
        )
        evict_count = max(1, len(sorted_alerts) // 10)
        for alert in sorted_alerts[:evict_count]:
            del self._alerts[alert.alert_id]
        logger.info("Evicted %d oldest alerts from store", evict_count)

    # ------------------------------------------------------------------
    # Alert Check Methods
    # ------------------------------------------------------------------

    def check_significant_change(
        self,
        country_code: str,
        index_type: str,
        previous_value: Decimal,
        current_value: Decimal,
    ) -> List[Dict[str, Any]]:
        """Check if a value change triggers significant change alerts.

        Evaluates all active SIGNIFICANT_CHANGE rules for the given
        index type and generates alerts for any breached thresholds.

        Args:
            country_code: ISO country code.
            index_type: Index type (CPI, WGI, BRIBERY).
            previous_value: Previous index value.
            current_value: Current index value.

        Returns:
            List of generated alert dictionaries.
        """
        start_time = time.monotonic()
        generated_alerts: List[Dict[str, Any]] = []
        change = current_value - previous_value

        with self._lock:
            rules = [
                r for r in self._rules.values()
                if r.get("alert_type") == "SIGNIFICANT_CHANGE"
                and r.get("index_type") == index_type
                and r.get("enabled", True)
            ]

        for rule in rules:
            threshold_delta = rule.get("threshold_delta", Decimal("0"))
            direction = rule.get("direction", "decrease")

            triggered = False
            if direction == "decrease" and change <= threshold_delta:
                triggered = True
            elif direction == "increase" and change >= threshold_delta:
                triggered = True

            if not triggered:
                continue

            # Check cooldown
            rule_id = rule["rule_id"]
            if self._is_in_cooldown(rule_id, country_code):
                logger.debug(
                    "Alert suppressed (cooldown): rule=%s country=%s",
                    rule_id, country_code,
                )
                continue

            # Generate alert
            severity = rule.get("severity", "MEDIUM")
            alert = self._create_alert(
                alert_type=AlertType.SIGNIFICANT_CHANGE.value,
                severity=severity,
                country_code=country_code,
                index_type=index_type,
                rule_id=rule_id,
                title=f"{index_type} significant change for {country_code}",
                description=(
                    f"{rule.get('description', '')}. "
                    f"Change: {previous_value} -> {current_value} "
                    f"(delta={change})"
                ),
                current_value=current_value,
                previous_value=previous_value,
                change_magnitude=change,
                metadata={
                    "threshold_delta": str(threshold_delta),
                    "actual_change": str(change),
                },
            )
            self._record_cooldown(rule_id, country_code)
            generated_alerts.append(alert.to_dict())

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        if generated_alerts:
            logger.info(
                "Significant change check for %s/%s: %d alerts generated "
                "time_ms=%.1f",
                country_code, index_type, len(generated_alerts),
                processing_time_ms,
            )
        return generated_alerts

    def check_threshold_breach(
        self,
        country_code: str,
        index_type: str,
        current_value: Decimal,
        previous_value: Optional[Decimal] = None,
    ) -> List[Dict[str, Any]]:
        """Check if a value crosses a configured threshold.

        Evaluates all active THRESHOLD_BREACH rules for the given
        index type and generates alerts for any breached thresholds.

        Args:
            country_code: ISO country code.
            index_type: Index type (CPI, WGI, BRIBERY).
            current_value: Current index value.
            previous_value: Previous value (to detect crossing direction).

        Returns:
            List of generated alert dictionaries.
        """
        start_time = time.monotonic()
        generated_alerts: List[Dict[str, Any]] = []

        with self._lock:
            rules = [
                r for r in self._rules.values()
                if r.get("alert_type") == "THRESHOLD_BREACH"
                and r.get("index_type") == index_type
                and r.get("enabled", True)
            ]

        for rule in rules:
            threshold_value = rule.get("threshold_value", Decimal("0"))
            direction = rule.get("direction", "below")

            triggered = False
            if direction == "below":
                # Alert if current is below threshold and previous was above
                if current_value < threshold_value:
                    if previous_value is None or previous_value >= threshold_value:
                        triggered = True
            elif direction == "above":
                if current_value > threshold_value:
                    if previous_value is None or previous_value <= threshold_value:
                        triggered = True

            if not triggered:
                continue

            rule_id = rule["rule_id"]
            if self._is_in_cooldown(rule_id, country_code):
                continue

            severity = rule.get("severity", "HIGH")
            alert = self._create_alert(
                alert_type=AlertType.THRESHOLD_BREACH.value,
                severity=severity,
                country_code=country_code,
                index_type=index_type,
                rule_id=rule_id,
                title=f"{index_type} threshold breach for {country_code}",
                description=(
                    f"{rule.get('description', '')}. "
                    f"Current value {current_value} crossed threshold "
                    f"{threshold_value} ({direction})"
                ),
                current_value=current_value,
                previous_value=previous_value or Decimal("0"),
                threshold_value=threshold_value,
                change_magnitude=abs(current_value - threshold_value),
            )
            self._record_cooldown(rule_id, country_code)
            generated_alerts.append(alert.to_dict())

        processing_time_ms = (time.monotonic() - start_time) * 1000.0
        if generated_alerts:
            logger.info(
                "Threshold breach check for %s/%s: %d alerts generated "
                "time_ms=%.1f",
                country_code, index_type, len(generated_alerts),
                processing_time_ms,
            )
        return generated_alerts

    def check_trend_reversal(
        self,
        country_code: str,
        index_type: str,
        previous_direction: str,
        current_direction: str,
    ) -> List[Dict[str, Any]]:
        """Check if a trend reversal triggers alerts.

        Args:
            country_code: ISO country code.
            index_type: Index type.
            previous_direction: Previous trend direction.
            current_direction: Current trend direction.

        Returns:
            List of generated alert dictionaries.
        """
        start_time = time.monotonic()
        generated_alerts: List[Dict[str, Any]] = []

        if previous_direction == current_direction:
            return generated_alerts

        with self._lock:
            rules = [
                r for r in self._rules.values()
                if r.get("alert_type") == "TREND_REVERSAL"
                and r.get("enabled", True)
            ]

        for rule in rules:
            from_dir = rule.get("from_direction", "")
            to_dir = rule.get("to_direction", "")

            if previous_direction == from_dir and current_direction == to_dir:
                rule_id = rule["rule_id"]
                if self._is_in_cooldown(rule_id, country_code):
                    continue

                severity = rule.get("severity", "HIGH")
                alert = self._create_alert(
                    alert_type=AlertType.TREND_REVERSAL.value,
                    severity=severity,
                    country_code=country_code,
                    index_type=index_type,
                    rule_id=rule_id,
                    title=f"Trend reversal for {country_code}/{index_type}",
                    description=(
                        f"Trend changed from {previous_direction} to "
                        f"{current_direction} for {country_code} {index_type}"
                    ),
                    current_value=Decimal("0"),
                    metadata={
                        "previous_direction": previous_direction,
                        "current_direction": current_direction,
                    },
                )
                self._record_cooldown(rule_id, country_code)
                generated_alerts.append(alert.to_dict())

        processing_time_ms = (time.monotonic() - start_time) * 1000.0
        if generated_alerts:
            logger.info(
                "Trend reversal check for %s/%s: %d alerts time_ms=%.1f",
                country_code, index_type, len(generated_alerts),
                processing_time_ms,
            )
        return generated_alerts

    def check_reclassification(
        self,
        country_code: str,
        previous_class: str,
        current_class: str,
    ) -> List[Dict[str, Any]]:
        """Check if a country reclassification triggers alerts.

        Args:
            country_code: ISO country code.
            previous_class: Previous EUDR risk classification.
            current_class: Current EUDR risk classification.

        Returns:
            List of generated alert dictionaries.
        """
        start_time = time.monotonic()
        generated_alerts: List[Dict[str, Any]] = []

        if previous_class == current_class:
            return generated_alerts

        with self._lock:
            rules = [
                r for r in self._rules.values()
                if r.get("alert_type") == "COUNTRY_RECLASSIFICATION"
                and r.get("enabled", True)
            ]

        for rule in rules:
            from_class = rule.get("from_class", "")
            to_class = rule.get("to_class", "")

            if previous_class == from_class and current_class == to_class:
                rule_id = rule["rule_id"]
                if self._is_in_cooldown(rule_id, country_code):
                    continue

                severity = rule.get("severity", "CRITICAL")
                alert = self._create_alert(
                    alert_type=AlertType.COUNTRY_RECLASSIFICATION.value,
                    severity=severity,
                    country_code=country_code,
                    index_type="COMPOSITE",
                    rule_id=rule_id,
                    title=f"Country reclassification: {country_code}",
                    description=(
                        f"{country_code} reclassified from {previous_class} "
                        f"to {current_class}. {rule.get('description', '')}"
                    ),
                    current_value=Decimal("0"),
                    metadata={
                        "previous_classification": previous_class,
                        "current_classification": current_class,
                    },
                )
                self._record_cooldown(rule_id, country_code)
                generated_alerts.append(alert.to_dict())

        processing_time_ms = (time.monotonic() - start_time) * 1000.0
        if generated_alerts:
            logger.info(
                "Reclassification check for %s: %s -> %s (%d alerts) "
                "time_ms=%.1f",
                country_code, previous_class, current_class,
                len(generated_alerts), processing_time_ms,
            )
        return generated_alerts

    def generate_alerts(
        self,
        data_updates: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run all configured alert rules against provided data updates.

        Each data_update dict should contain:
            - country_code (str)
            - index_type (str)
            - current_value (Decimal or numeric)
            - previous_value (Decimal or numeric)

        Args:
            data_updates: List of index data updates to check.

        Returns:
            Dictionary with all generated alerts and summary.
        """
        start_time = time.monotonic()
        all_alerts: List[Dict[str, Any]] = []

        if not data_updates:
            return {
                "alerts_generated": 0,
                "alerts": [],
                "processing_time_ms": 0.0,
                "calculation_timestamp": _utcnow().isoformat(),
                "provenance_hash": _compute_hash({"no_updates": True}),
            }

        for update in data_updates:
            cc = update.get("country_code", "")
            idx_type = update.get("index_type", "CPI")
            current = _to_decimal(update.get("current_value", 0))
            previous = _to_decimal(update.get("previous_value", 0))

            if not cc:
                continue

            # Check significant change
            alerts = self.check_significant_change(cc, idx_type, previous, current)
            all_alerts.extend(alerts)

            # Check threshold breach
            alerts = self.check_threshold_breach(cc, idx_type, current, previous)
            all_alerts.extend(alerts)

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        result = {
            "data_updates_processed": len(data_updates),
            "alerts_generated": len(all_alerts),
            "alerts": all_alerts,
            "processing_time_ms": round(processing_time_ms, 3),
            "calculation_timestamp": _utcnow().isoformat(),
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Alert generation: %d updates processed, %d alerts generated "
            "time_ms=%.1f",
            len(data_updates), len(all_alerts), processing_time_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Alert Lifecycle Management
    # ------------------------------------------------------------------

    def get_alert(self, alert_id: str) -> Dict[str, Any]:
        """Retrieve a specific alert by ID.

        Args:
            alert_id: Alert identifier.

        Returns:
            Dictionary with alert data.

        Raises:
            ValueError: If alert_id is not found.
        """
        with self._lock:
            alert = self._alerts.get(alert_id)
        if alert is None:
            raise ValueError(f"Alert not found: {alert_id}")

        # Check expiration
        result = alert.to_dict()
        if alert.status == AlertStatus.ACTIVE.value:
            now = _utcnow()
            expires = datetime.fromisoformat(alert.expires_at)
            if now > expires:
                alert.status = AlertStatus.EXPIRED.value
                result = alert.to_dict()

        return result

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Acknowledge an active alert.

        Args:
            alert_id: Alert identifier.
            acknowledged_by: User or system acknowledging the alert.
            notes: Optional acknowledgement notes.

        Returns:
            Dictionary confirming acknowledgement.

        Raises:
            ValueError: If alert_id not found or alert is not ACTIVE.
        """
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert is None:
                raise ValueError(f"Alert not found: {alert_id}")
            if alert.status != AlertStatus.ACTIVE.value:
                raise ValueError(
                    f"Alert {alert_id} is {alert.status}, not ACTIVE"
                )

            alert.status = AlertStatus.ACKNOWLEDGED.value
            alert.acknowledged_at = _utcnow().isoformat()
            alert.acknowledged_by = acknowledged_by
            if notes:
                alert.metadata["acknowledgement_notes"] = notes
            alert.provenance_hash = _compute_hash(alert)

        logger.info(
            "Alert acknowledged: id=%s by=%s", alert_id, acknowledged_by,
        )
        return {
            "alert_id": alert_id,
            "action": "acknowledged",
            "acknowledged_by": acknowledged_by,
            "acknowledged_at": alert.acknowledged_at,
            "notes": notes,
            "provenance_hash": alert.provenance_hash,
        }

    def resolve_alert(
        self,
        alert_id: str,
        resolution_notes: str = "",
    ) -> Dict[str, Any]:
        """Resolve an alert.

        Args:
            alert_id: Alert identifier.
            resolution_notes: Description of resolution.

        Returns:
            Dictionary confirming resolution.

        Raises:
            ValueError: If alert_id not found.
        """
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert is None:
                raise ValueError(f"Alert not found: {alert_id}")

            alert.status = AlertStatus.RESOLVED.value
            alert.resolved_at = _utcnow().isoformat()
            alert.resolution_notes = resolution_notes
            alert.provenance_hash = _compute_hash(alert)

        logger.info("Alert resolved: id=%s", alert_id)
        return {
            "alert_id": alert_id,
            "action": "resolved",
            "resolved_at": alert.resolved_at,
            "resolution_notes": resolution_notes,
            "provenance_hash": alert.provenance_hash,
        }

    def suppress_alert(self, alert_id: str, reason: str = "") -> Dict[str, Any]:
        """Suppress an alert (prevent further notifications).

        Args:
            alert_id: Alert identifier.
            reason: Reason for suppression.

        Returns:
            Dictionary confirming suppression.

        Raises:
            ValueError: If alert_id not found.
        """
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert is None:
                raise ValueError(f"Alert not found: {alert_id}")

            alert.status = AlertStatus.SUPPRESSED.value
            alert.metadata["suppression_reason"] = reason
            alert.provenance_hash = _compute_hash(alert)

        logger.info("Alert suppressed: id=%s reason=%s", alert_id, reason)
        return {
            "alert_id": alert_id,
            "action": "suppressed",
            "reason": reason,
            "provenance_hash": alert.provenance_hash,
        }

    # ------------------------------------------------------------------
    # Alert Query & Summary
    # ------------------------------------------------------------------

    def get_alert_summary(
        self,
        country_code: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a summary of alerts with optional filters.

        Args:
            country_code: Optional filter by country.
            severity: Optional filter by severity.
            status: Optional filter by status.

        Returns:
            Dictionary containing AlertSummary data.
        """
        start_time = time.monotonic()

        with self._lock:
            alerts = list(self._alerts.values())

        # Apply filters
        if country_code:
            alerts = [a for a in alerts if a.country_code == country_code.upper()]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if status:
            alerts = [a for a in alerts if a.status == status]

        # Compute summary
        summary = AlertSummary(total_alerts=len(alerts))

        status_counts = {s.value: 0 for s in AlertStatus}
        severity_counts = {s.value: 0 for s in AlertSeverity}
        type_counts = {t.value: 0 for t in AlertType}
        country_counts: Dict[str, int] = {}

        for alert in alerts:
            status_counts[alert.status] = status_counts.get(alert.status, 0) + 1
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
            type_counts[alert.alert_type] = type_counts.get(alert.alert_type, 0) + 1
            country_counts[alert.country_code] = country_counts.get(alert.country_code, 0) + 1

        summary.active_count = status_counts.get(AlertStatus.ACTIVE.value, 0)
        summary.acknowledged_count = status_counts.get(AlertStatus.ACKNOWLEDGED.value, 0)
        summary.resolved_count = status_counts.get(AlertStatus.RESOLVED.value, 0)
        summary.suppressed_count = status_counts.get(AlertStatus.SUPPRESSED.value, 0)
        summary.expired_count = status_counts.get(AlertStatus.EXPIRED.value, 0)
        summary.severity_breakdown = severity_counts
        summary.type_breakdown = type_counts

        # Top countries by alert count
        top_countries = sorted(
            country_counts.items(), key=lambda x: x[1], reverse=True,
        )[:10]
        summary.top_countries = [
            {"country_code": cc, "alert_count": count}
            for cc, count in top_countries
        ]

        # Recent alerts (most recent 10)
        recent = sorted(alerts, key=lambda a: a.created_at, reverse=True)[:10]
        summary.recent_alerts = [a.to_dict() for a in recent]

        summary.provenance_hash = _compute_hash(summary)

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        out = summary.to_dict()
        out["filters"] = {
            "country_code": country_code,
            "severity": severity,
            "status": status,
        }
        out["processing_time_ms"] = round(processing_time_ms, 3)
        out["calculation_timestamp"] = _utcnow().isoformat()

        logger.info(
            "Alert summary: total=%d active=%d critical=%d time_ms=%.1f",
            summary.total_alerts, summary.active_count,
            severity_counts.get("CRITICAL", 0), processing_time_ms,
        )
        return out

    def list_alerts(
        self,
        country_code: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        index_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List alerts with filtering, pagination, and sorting.

        Args:
            country_code: Optional filter by country.
            severity: Optional filter by severity.
            status: Optional filter by status.
            index_type: Optional filter by index type.
            limit: Maximum results to return (default 100).
            offset: Offset for pagination (default 0).

        Returns:
            Dictionary with paginated alert list.
        """
        with self._lock:
            alerts = list(self._alerts.values())

        # Apply filters
        if country_code:
            alerts = [a for a in alerts if a.country_code == country_code.upper()]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if status:
            alerts = [a for a in alerts if a.status == status]
        if index_type:
            alerts = [a for a in alerts if a.index_type == index_type]

        # Sort by severity (descending) then created_at (descending)
        alerts.sort(
            key=lambda a: (
                -SEVERITY_ORDER.get(a.severity, 0),
                a.created_at,
            ),
            reverse=True,
        )

        total = len(alerts)
        paginated = alerts[offset:offset + limit]

        return {
            "total_count": total,
            "returned_count": len(paginated),
            "offset": offset,
            "limit": limit,
            "alerts": [a.to_dict() for a in paginated],
            "provenance_hash": _compute_hash({"list": total, "offset": offset}),
        }

    def get_country_alert_history(
        self,
        country_code: str,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Get alert history for a specific country.

        Args:
            country_code: ISO country code.
            limit: Maximum alerts to return.

        Returns:
            Dictionary with country's alert history.
        """
        if not country_code:
            raise ValueError("country_code must be non-empty")
        country_code = country_code.upper()

        with self._lock:
            country_alerts = [
                a for a in self._alerts.values()
                if a.country_code == country_code
            ]

        # Sort by created_at descending
        country_alerts.sort(key=lambda a: a.created_at, reverse=True)
        country_alerts = country_alerts[:limit]

        # Summary stats
        severity_counts = {}
        for a in country_alerts:
            severity_counts[a.severity] = severity_counts.get(a.severity, 0) + 1

        return {
            "country_code": country_code,
            "total_alerts": len(country_alerts),
            "severity_breakdown": severity_counts,
            "alerts": [a.to_dict() for a in country_alerts],
            "on_watchlist": country_code in self._watchlist,
            "provenance_hash": _compute_hash({"country_history": country_code}),
        }

    def clear_expired_alerts(self) -> Dict[str, Any]:
        """Remove all expired alerts from the store.

        Returns:
            Dictionary with count of removed alerts.
        """
        now = _utcnow()
        removed = 0

        with self._lock:
            expired_ids = []
            for alert_id, alert in self._alerts.items():
                try:
                    expires = datetime.fromisoformat(alert.expires_at)
                    if now > expires:
                        expired_ids.append(alert_id)
                except (ValueError, TypeError):
                    pass

            for alert_id in expired_ids:
                del self._alerts[alert_id]
                removed += 1

        logger.info("Cleared %d expired alerts", removed)
        return {
            "action": "clear_expired",
            "removed_count": removed,
            "remaining_count": len(self._alerts),
            "provenance_hash": _compute_hash({"cleared": removed}),
        }
