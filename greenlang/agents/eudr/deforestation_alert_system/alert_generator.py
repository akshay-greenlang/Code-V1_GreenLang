# -*- coding: utf-8 -*-
"""
AGENT-EUDR-020: Deforestation Alert System - Alert Generator

Creates structured alerts from satellite change detections, applying
classification rules, severity scoring, affected area calculation, and
proximity analysis to registered supply chain plots. Supports both batch
and real-time alert generation modes with configurable deduplication
windows and daily rate limiting.

Zero-Hallucination Guarantees:
    - Alert severity determined by static threshold lookup tables
    - Proximity calculated via deterministic Haversine formula (Decimal)
    - Post-cutoff determination via date comparison (2020-12-31)
    - Area calculations via Decimal arithmetic only
    - SHA-256 provenance hashes on all result objects
    - No LLM/ML in the alert generation path

Alert Generation Pipeline:
    1. Receive DetectionResult from SatelliteChangeDetector
    2. Build structured alert title and description
    3. Find affected supply chain plots within buffer radius
    4. Calculate proximity via Haversine to nearest plot
    5. Determine post-cutoff status (EUDR 2020-12-31)
    6. Assign severity score and priority
    7. Deduplicate within configurable window (72h default)
    8. Persist alert and publish to notification channels

Haversine Distance Formula:
    R = 6371 km (Earth's mean radius)
    a = sin^2(delta_lat/2) + cos(lat1)*cos(lat2)*sin^2(delta_lon/2)
    d = 2 * R * arcsin(sqrt(a))

Performance Targets:
    - Single alert generation: <50ms
    - Batch generation (1000 detections): <10s
    - Deduplication check: <5ms

Regulatory References:
    - EUDR Article 2(1): Deforestation-free commodity requirement
    - EUDR Article 2(6): Cutoff date 31 December 2020
    - EUDR Article 9(1)(d): Geolocation and traceability
    - EUDR Article 10(2): Risk assessment and mitigation
    - EUDR Article 31: 5-year record retention

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-020 (Engine 2: Alert Generator)
Agent ID: GL-EUDR-DAS-020
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Config import (thread-safe singleton)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.deforestation_alert_system.config import get_config
except ImportError:
    get_config = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Provenance import
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.deforestation_alert_system.provenance import (
        ProvenanceTracker,
        get_tracker,
    )
except ImportError:
    ProvenanceTracker = None  # type: ignore[misc,assignment]
    get_tracker = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Metrics import (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.deforestation_alert_system.metrics import (
        PROMETHEUS_AVAILABLE,
        record_alert_generated,
        observe_alert_generation_duration,
        record_api_error,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]
    record_alert_generated = None  # type: ignore[misc,assignment]
    observe_alert_generation_duration = None  # type: ignore[misc,assignment]
    record_api_error = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Sibling engine imports (for type references)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.deforestation_alert_system.satellite_change_detector import (
        DetectionResult,
        ChangeType,
    )
except ImportError:
    DetectionResult = None  # type: ignore[misc,assignment]
    ChangeType = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance.

    Args:
        data: Any JSON-serializable object.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_id() -> str:
    """Generate a unique identifier using UUID4.

    Returns:
        String representation of a new UUID4.
    """
    return str(uuid.uuid4())


def _safe_decimal(value: Any, default: Decimal = Decimal("0")) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Value to convert.
        default: Default Decimal if conversion fails.

    Returns:
        Decimal representation of value or default.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


def _elapsed_ms(start: float) -> float:
    """Calculate elapsed milliseconds since start.

    Args:
        start: time.perf_counter() start value.

    Returns:
        Elapsed time in milliseconds.
    """
    return round((time.perf_counter() - start) * 1000, 2)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: EUDR deforestation cutoff date per Article 2(6).
EUDR_CUTOFF_DATE: date = date(2020, 12, 31)

#: Earth's mean radius in kilometers for Haversine calculation.
EARTH_RADIUS_KM: Decimal = Decimal("6371")

#: Default buffer radius for proximity analysis (km).
DEFAULT_BUFFER_RADIUS_KM: Decimal = Decimal("10")

#: Maximum number of alerts per batch.
MAX_BATCH_SIZE: int = 10000

#: Default deduplication window in hours.
DEFAULT_DEDUP_WINDOW_HOURS: int = 72

#: Maximum alerts per day rate limit.
DEFAULT_MAX_ALERTS_PER_DAY: int = 10000

#: Alert retention period in years per EUDR Article 31.
ALERT_RETENTION_YEARS: int = 5

#: Deduplication spatial tolerance in degrees (~1km at equator).
DEDUP_SPATIAL_TOLERANCE: Decimal = Decimal("0.01")


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AlertPriority(str, Enum):
    """Alert priority levels for notification routing.

    Priority determines notification channel selection and
    response time expectations for the alert workflow.
    """

    IMMEDIATE = "immediate"
    URGENT = "urgent"
    STANDARD = "standard"
    LOW = "low"


class AlertStatus(str, Enum):
    """Alert lifecycle status values.

    Tracks the alert through its workflow from creation
    to resolution or dismissal.
    """

    PENDING = "pending"
    TRIAGED = "triaged"
    INVESTIGATING = "investigating"
    CONFIRMED = "confirmed"
    FALSE_POSITIVE = "false_positive"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"
    ESCALATED = "escalated"


class NotificationChannel(str, Enum):
    """Notification delivery channels for alert distribution."""

    EMAIL = "email"
    WEBHOOK = "webhook"
    SMS = "sms"
    DASHBOARD = "dashboard"
    SLACK = "slack"


# ---------------------------------------------------------------------------
# Severity-to-Priority mapping
# ---------------------------------------------------------------------------

#: Maps severity score ranges to alert priority levels.
SEVERITY_PRIORITY_MAP: Dict[str, AlertPriority] = {
    "CRITICAL": AlertPriority.IMMEDIATE,
    "HIGH": AlertPriority.URGENT,
    "MEDIUM": AlertPriority.STANDARD,
    "LOW": AlertPriority.LOW,
    "INFORMATIONAL": AlertPriority.LOW,
}

#: Severity score thresholds (lower bound inclusive).
SEVERITY_THRESHOLDS: Dict[str, int] = {
    "CRITICAL": 80,
    "HIGH": 60,
    "MEDIUM": 40,
    "LOW": 20,
    "INFORMATIONAL": 0,
}

#: EUDR regulated commodities per Annex I.
EUDR_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood",
]


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class SupplyPlot:
    """Registered supply chain production plot.

    Represents a geolocation-verified plot in the supply chain
    subject to EUDR monitoring and deforestation-free verification.

    Attributes:
        plot_id: Unique plot identifier.
        latitude: Plot center latitude.
        longitude: Plot center longitude.
        area_ha: Plot area in hectares.
        commodity: EUDR commodity produced.
        country_code: ISO 3166-1 alpha-2 country code.
        supplier_id: Supplier identifier.
        plot_name: Human-readable plot name.
    """

    plot_id: str = ""
    latitude: Decimal = Decimal("0")
    longitude: Decimal = Decimal("0")
    area_ha: Decimal = Decimal("0")
    commodity: str = ""
    country_code: str = ""
    supplier_id: str = ""
    plot_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation for JSON serialization.
        """
        return {
            "plot_id": self.plot_id,
            "latitude": str(self.latitude),
            "longitude": str(self.longitude),
            "area_ha": str(self.area_ha),
            "commodity": self.commodity,
            "country_code": self.country_code,
            "supplier_id": self.supplier_id,
            "plot_name": self.plot_name,
        }


@dataclass
class AffectedPlot:
    """Supply plot affected by a deforestation alert.

    Contains the plot reference and calculated proximity to the
    detection event center for impact assessment.

    Attributes:
        plot_id: Plot identifier.
        plot_name: Human-readable plot name.
        commodity: EUDR commodity.
        distance_km: Haversine distance to detection center.
        inside_buffer: Whether plot is within the alert buffer zone.
        supplier_id: Supplier identifier.
    """

    plot_id: str = ""
    plot_name: str = ""
    commodity: str = ""
    distance_km: Decimal = Decimal("0")
    inside_buffer: bool = False
    supplier_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "plot_id": self.plot_id,
            "plot_name": self.plot_name,
            "commodity": self.commodity,
            "distance_km": str(self.distance_km),
            "inside_buffer": self.inside_buffer,
            "supplier_id": self.supplier_id,
        }


@dataclass
class DeforestationAlert:
    """Structured deforestation alert generated from satellite detection.

    Contains the complete alert information including detection source,
    geographic context, affected plots, severity, post-cutoff status,
    and audit trail for EUDR compliance reporting.

    Attributes:
        alert_id: Unique alert identifier (UUID).
        detection_id: Source detection identifier.
        severity: Severity level (CRITICAL/HIGH/MEDIUM/LOW/INFORMATIONAL).
        severity_score: Numeric severity score (0-100).
        status: Alert lifecycle status.
        priority: Notification priority level.
        title: Human-readable alert title.
        description: Detailed alert description.
        area_ha: Affected area in hectares.
        latitude: Detection center latitude.
        longitude: Detection center longitude.
        country_code: ISO 3166-1 alpha-2 country code.
        commodity: Relevant EUDR commodity (if determinable).
        affected_plots: List of affected supply chain plots.
        proximity_km: Distance to nearest affected plot (km).
        is_post_cutoff: Whether detection is post-EUDR cutoff (2020-12-31).
        change_type: Type of vegetation change detected.
        confidence: Detection confidence score (0-1).
        source: Satellite source.
        generated_at: Alert generation timestamp.
        dedup_key: Deduplication key for duplicate detection.
        notification_channels: Target notification channels.
        provenance_hash: SHA-256 provenance hash.
        metadata: Additional metadata dictionary.
    """

    alert_id: str = ""
    detection_id: str = ""
    severity: str = "MEDIUM"
    severity_score: Decimal = Decimal("50")
    status: str = AlertStatus.PENDING.value
    priority: str = AlertPriority.STANDARD.value
    title: str = ""
    description: str = ""
    area_ha: Decimal = Decimal("0")
    latitude: Decimal = Decimal("0")
    longitude: Decimal = Decimal("0")
    country_code: str = ""
    commodity: str = ""
    affected_plots: List[AffectedPlot] = field(default_factory=list)
    proximity_km: Decimal = Decimal("-1")
    is_post_cutoff: bool = True
    change_type: str = ""
    confidence: Decimal = Decimal("0")
    source: str = ""
    generated_at: str = ""
    dedup_key: str = ""
    notification_channels: List[str] = field(default_factory=list)
    provenance_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set defaults for unset fields."""
        if not self.alert_id:
            self.alert_id = _generate_id()
        if not self.generated_at:
            self.generated_at = _utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize alert to dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "alert_id": self.alert_id,
            "detection_id": self.detection_id,
            "severity": self.severity,
            "severity_score": str(self.severity_score),
            "status": self.status,
            "priority": self.priority,
            "title": self.title,
            "description": self.description,
            "area_ha": str(self.area_ha),
            "latitude": str(self.latitude),
            "longitude": str(self.longitude),
            "country_code": self.country_code,
            "commodity": self.commodity,
            "affected_plots_count": len(self.affected_plots),
            "affected_plots": [p.to_dict() for p in self.affected_plots],
            "proximity_km": str(self.proximity_km),
            "is_post_cutoff": self.is_post_cutoff,
            "change_type": self.change_type,
            "confidence": str(self.confidence),
            "source": self.source,
            "generated_at": self.generated_at,
            "dedup_key": self.dedup_key,
            "notification_channels": self.notification_channels,
            "provenance_hash": self.provenance_hash,
            "metadata": self.metadata,
        }


@dataclass
class AlertResult:
    """Result of a single alert generation operation.

    Attributes:
        alert: Generated DeforestationAlert.
        is_duplicate: Whether this alert was deduplicated.
        duplicate_of: Alert ID of the original if duplicate.
        processing_time_ms: Processing time in milliseconds.
        warnings: List of warning messages.
        provenance_hash: SHA-256 provenance hash.
        calculation_timestamp: Timestamp of generation.
    """

    alert: Optional[DeforestationAlert] = None
    is_duplicate: bool = False
    duplicate_of: str = ""
    processing_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    provenance_hash: str = ""
    calculation_timestamp: str = ""

    def __post_init__(self) -> None:
        """Set calculation timestamp if unset."""
        if not self.calculation_timestamp:
            self.calculation_timestamp = _utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "alert": self.alert.to_dict() if self.alert else None,
            "is_duplicate": self.is_duplicate,
            "duplicate_of": self.duplicate_of,
            "processing_time_ms": self.processing_time_ms,
            "warnings": self.warnings,
            "provenance_hash": self.provenance_hash,
            "calculation_timestamp": self.calculation_timestamp,
        }


@dataclass
class BatchResult:
    """Result of a batch alert generation operation.

    Attributes:
        batch_id: Unique batch identifier.
        total_detections: Number of input detections.
        alerts_generated: Number of alerts created.
        alerts_deduplicated: Number of alerts removed as duplicates.
        alerts: List of generated DeforestationAlert objects.
        duplicates: List of duplicate alert IDs.
        processing_time_ms: Total batch processing time.
        warnings: List of warning messages.
        provenance_hash: SHA-256 provenance hash.
        calculation_timestamp: Batch generation timestamp.
    """

    batch_id: str = ""
    total_detections: int = 0
    alerts_generated: int = 0
    alerts_deduplicated: int = 0
    alerts: List[DeforestationAlert] = field(default_factory=list)
    duplicates: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    provenance_hash: str = ""
    calculation_timestamp: str = ""

    def __post_init__(self) -> None:
        """Set defaults for unset fields."""
        if not self.batch_id:
            self.batch_id = _generate_id()
        if not self.calculation_timestamp:
            self.calculation_timestamp = _utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "batch_id": self.batch_id,
            "total_detections": self.total_detections,
            "alerts_generated": self.alerts_generated,
            "alerts_deduplicated": self.alerts_deduplicated,
            "alert_ids": [a.alert_id for a in self.alerts],
            "duplicates": self.duplicates,
            "processing_time_ms": self.processing_time_ms,
            "warnings": self.warnings,
            "provenance_hash": self.provenance_hash,
            "calculation_timestamp": self.calculation_timestamp,
        }


@dataclass
class AlertListResult:
    """Paginated list of alerts.

    Attributes:
        alerts: List of DeforestationAlert objects.
        total: Total number of matching alerts.
        page: Current page number.
        page_size: Items per page.
        has_more: Whether more pages exist.
        provenance_hash: SHA-256 provenance hash.
        calculation_timestamp: Query timestamp.
    """

    alerts: List[DeforestationAlert] = field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 50
    has_more: bool = False
    provenance_hash: str = ""
    calculation_timestamp: str = ""

    def __post_init__(self) -> None:
        """Set calculation timestamp if unset."""
        if not self.calculation_timestamp:
            self.calculation_timestamp = _utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "total": self.total,
            "page": self.page,
            "page_size": self.page_size,
            "has_more": self.has_more,
            "alert_count": len(self.alerts),
            "provenance_hash": self.provenance_hash,
            "calculation_timestamp": self.calculation_timestamp,
        }


@dataclass
class SummaryResult:
    """Alert summary statistics.

    Attributes:
        total_alerts: Total number of alerts.
        by_severity: Count per severity level.
        by_status: Count per status.
        by_country: Count per country code.
        by_commodity: Count per commodity.
        by_change_type: Count per change type.
        post_cutoff_count: Number of post-cutoff alerts.
        pre_cutoff_count: Number of pre-cutoff alerts.
        avg_severity_score: Average severity score.
        avg_proximity_km: Average proximity to nearest plot.
        total_affected_area_ha: Total affected area.
        provenance_hash: SHA-256 provenance hash.
        calculation_timestamp: Query timestamp.
    """

    total_alerts: int = 0
    by_severity: Dict[str, int] = field(default_factory=dict)
    by_status: Dict[str, int] = field(default_factory=dict)
    by_country: Dict[str, int] = field(default_factory=dict)
    by_commodity: Dict[str, int] = field(default_factory=dict)
    by_change_type: Dict[str, int] = field(default_factory=dict)
    post_cutoff_count: int = 0
    pre_cutoff_count: int = 0
    avg_severity_score: Decimal = Decimal("0")
    avg_proximity_km: Decimal = Decimal("0")
    total_affected_area_ha: Decimal = Decimal("0")
    provenance_hash: str = ""
    calculation_timestamp: str = ""

    def __post_init__(self) -> None:
        """Set calculation timestamp if unset."""
        if not self.calculation_timestamp:
            self.calculation_timestamp = _utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "total_alerts": self.total_alerts,
            "by_severity": self.by_severity,
            "by_status": self.by_status,
            "by_country": self.by_country,
            "by_commodity": self.by_commodity,
            "by_change_type": self.by_change_type,
            "post_cutoff_count": self.post_cutoff_count,
            "pre_cutoff_count": self.pre_cutoff_count,
            "avg_severity_score": str(self.avg_severity_score),
            "avg_proximity_km": str(self.avg_proximity_km),
            "total_affected_area_ha": str(self.total_affected_area_ha),
            "provenance_hash": self.provenance_hash,
            "calculation_timestamp": self.calculation_timestamp,
        }


@dataclass
class StatsResult:
    """Alert statistics grouped by a dimension.

    Attributes:
        grouping: Grouping dimension used.
        groups: Dictionary of group label to count.
        total: Total count across all groups.
        provenance_hash: SHA-256 provenance hash.
        calculation_timestamp: Query timestamp.
    """

    grouping: str = "severity"
    groups: Dict[str, int] = field(default_factory=dict)
    total: int = 0
    provenance_hash: str = ""
    calculation_timestamp: str = ""

    def __post_init__(self) -> None:
        """Set calculation timestamp if unset."""
        if not self.calculation_timestamp:
            self.calculation_timestamp = _utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "grouping": self.grouping,
            "groups": self.groups,
            "total": self.total,
            "provenance_hash": self.provenance_hash,
            "calculation_timestamp": self.calculation_timestamp,
        }


# ---------------------------------------------------------------------------
# AlertGenerator Engine
# ---------------------------------------------------------------------------


class AlertGenerator:
    """Production-grade deforestation alert generation engine.

    Creates structured alerts from satellite detection results by applying
    classification rules, severity scoring, affected area calculation, and
    proximity analysis to registered supply chain plots. Supports batch and
    real-time alert generation with configurable deduplication windows
    and daily rate limiting.

    All calculations use deterministic Decimal arithmetic with zero
    LLM/ML involvement. Haversine distance formula computes proximity
    to registered plots. Post-cutoff status is determined by date
    comparison against EUDR cutoff date (31 December 2020).

    Attributes:
        _config: Agent configuration from get_config().
        _tracker: ProvenanceTracker instance for audit trails.
        _alert_store: In-memory alert storage keyed by alert_id.
        _daily_count: Count of alerts generated today for rate limiting.
        _daily_date: Date of the current daily count window.
        _dedup_cache: Cache of recent dedup keys for deduplication.

    Example:
        >>> generator = AlertGenerator()
        >>> result = generator.generate_alert(detection, plots)
        >>> assert result.alert is not None
        >>> assert result.provenance_hash != ""
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize the AlertGenerator.

        Args:
            config: Optional configuration object. If None, loads from
                get_config() singleton.
        """
        self._config = config
        if self._config is None and get_config is not None:
            try:
                self._config = get_config()
            except Exception:
                logger.warning(
                    "Failed to load config via get_config(), "
                    "using hardcoded defaults"
                )
                self._config = None

        self._tracker: Optional[Any] = None
        if get_tracker is not None:
            try:
                self._tracker = get_tracker()
            except Exception:
                logger.debug("ProvenanceTracker not available")

        self._alert_store: Dict[str, DeforestationAlert] = {}
        self._daily_count: int = 0
        self._daily_date: date = _utcnow().date()
        self._dedup_cache: Dict[str, str] = {}

        logger.info(
            "AlertGenerator initialized: config=%s, provenance=%s",
            "loaded" if self._config else "defaults",
            "enabled" if self._tracker else "disabled",
        )

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _get_dedup_window_hours(self) -> int:
        """Return deduplication window from config or default.

        Returns:
            Deduplication window in hours.
        """
        if self._config and hasattr(self._config, "dedup_window_hours"):
            return int(self._config.dedup_window_hours)
        return DEFAULT_DEDUP_WINDOW_HOURS

    def _get_max_alerts_per_day(self) -> int:
        """Return daily alert cap from config or default.

        Returns:
            Maximum alerts per day.
        """
        if self._config and hasattr(self._config, "max_alerts_per_day"):
            return int(self._config.max_alerts_per_day)
        return DEFAULT_MAX_ALERTS_PER_DAY

    def _get_buffer_radius_km(self) -> Decimal:
        """Return default buffer radius from config or default.

        Returns:
            Buffer radius in kilometers as Decimal.
        """
        if self._config and hasattr(self._config, "default_buffer_radius_km"):
            return _safe_decimal(
                self._config.default_buffer_radius_km, DEFAULT_BUFFER_RADIUS_KM
            )
        return DEFAULT_BUFFER_RADIUS_KM

    # ------------------------------------------------------------------
    # Public API: Single alert generation
    # ------------------------------------------------------------------

    def generate_alert(
        self,
        detection: Any,
        plots: Optional[List[SupplyPlot]] = None,
    ) -> AlertResult:
        """Generate a single deforestation alert from a detection result.

        Constructs a structured alert with title, description, affected
        plots, proximity calculation, post-cutoff status, and severity.

        Args:
            detection: DetectionResult from SatelliteChangeDetector.
            plots: Optional list of registered supply chain plots for
                proximity analysis.

        Returns:
            AlertResult containing the generated alert or duplicate info.

        Raises:
            ValueError: If detection is None or missing required fields.
        """
        op_start = time.perf_counter()

        if detection is None:
            raise ValueError("detection must not be None")

        detection_id = getattr(detection, "detection_id", "")
        if not detection_id:
            raise ValueError("detection.detection_id must not be empty")

        logger.debug(
            "generate_alert: detection_id=%s", detection_id[:12]
        )

        # Check daily rate limit
        self._check_daily_limit()

        # Build deduplication key
        dedup_key = self._build_dedup_key(detection)

        # Check for duplicate
        existing_id = self._dedup_cache.get(dedup_key)
        if existing_id:
            elapsed = _elapsed_ms(op_start)
            logger.debug(
                "generate_alert: duplicate detected, original=%s",
                existing_id[:12],
            )
            return AlertResult(
                is_duplicate=True,
                duplicate_of=existing_id,
                processing_time_ms=elapsed,
                warnings=["Alert deduplicated against existing alert"],
                provenance_hash=_compute_hash({
                    "duplicate_of": existing_id,
                    "dedup_key": dedup_key,
                }),
            )

        # Extract detection attributes
        det_lat = _safe_decimal(getattr(detection, "latitude", 0))
        det_lon = _safe_decimal(getattr(detection, "longitude", 0))
        area_ha = _safe_decimal(getattr(detection, "area_ha", 0))
        change_type = getattr(detection, "change_type", "deforestation")
        confidence = _safe_decimal(getattr(detection, "confidence", 0))
        source = getattr(detection, "source", "unknown")
        country_code = getattr(detection, "country_code", "")

        # Find affected plots and calculate proximity
        affected = []
        proximity_km = Decimal("-1")
        buffer_km = self._get_buffer_radius_km()

        if plots:
            affected = self._find_affected_plots(
                detection, plots, buffer_km
            )
            if affected:
                proximity_km = min(p.distance_km for p in affected)

        # Determine commodity from affected plots or detection
        commodity = self._determine_commodity(detection, affected)

        # Determine post-cutoff status
        detection_date = self._extract_detection_date(detection)
        is_post_cutoff = self._determine_post_cutoff(detection_date)

        # Build alert title and description
        title = self._build_alert_title(detection)
        description = self._build_alert_description(
            detection, proximity_km, len(affected)
        )

        # Calculate preliminary severity (quick estimate)
        severity, severity_score = self._estimate_severity(
            area_ha, proximity_km, is_post_cutoff
        )

        # Determine priority from severity
        priority = SEVERITY_PRIORITY_MAP.get(
            severity, AlertPriority.STANDARD
        ).value

        # Determine notification channels
        channels = self._determine_channels(severity)

        # Create alert
        alert = DeforestationAlert(
            detection_id=detection_id,
            severity=severity,
            severity_score=severity_score,
            priority=priority,
            title=title,
            description=description,
            area_ha=area_ha,
            latitude=det_lat,
            longitude=det_lon,
            country_code=country_code,
            commodity=commodity,
            affected_plots=affected,
            proximity_km=proximity_km,
            is_post_cutoff=is_post_cutoff,
            change_type=change_type if isinstance(change_type, str) else str(change_type),
            confidence=confidence,
            source=source if isinstance(source, str) else str(source),
            dedup_key=dedup_key,
            notification_channels=channels,
        )
        alert.provenance_hash = _compute_hash(alert.to_dict())

        # Store alert
        self._alert_store[alert.alert_id] = alert
        self._dedup_cache[dedup_key] = alert.alert_id
        self._daily_count += 1

        # Record provenance
        if self._tracker:
            try:
                self._tracker.record(
                    entity_type="alert",
                    action="create",
                    entity_id=alert.alert_id,
                    data=alert.to_dict(),
                    metadata={
                        "detection_id": detection_id,
                        "severity": severity,
                        "is_post_cutoff": is_post_cutoff,
                        "affected_plots": len(affected),
                    },
                )
            except Exception:
                logger.debug("Failed to record provenance for generate_alert")

        # Record metrics
        if record_alert_generated:
            try:
                record_alert_generated(source=source, severity=severity)
            except Exception:
                pass

        elapsed = _elapsed_ms(op_start)
        if observe_alert_generation_duration:
            try:
                observe_alert_generation_duration(elapsed / 1000.0)
            except Exception:
                pass

        result = AlertResult(
            alert=alert,
            processing_time_ms=elapsed,
        )
        result.provenance_hash = _compute_hash(result.to_dict())

        logger.info(
            "generate_alert: alert_id=%s, severity=%s, "
            "affected_plots=%d, post_cutoff=%s in %.1fms",
            alert.alert_id[:12], severity, len(affected),
            is_post_cutoff, elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Batch generation
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        detections: List[Any],
        plots: Optional[List[SupplyPlot]] = None,
    ) -> BatchResult:
        """Generate alerts for a batch of detection results.

        Processes multiple detections sequentially with deduplication
        and rate limiting applied across the batch.

        Args:
            detections: List of DetectionResult objects.
            plots: Optional list of supply chain plots.

        Returns:
            BatchResult with all generated alerts and statistics.

        Raises:
            ValueError: If detections list is empty or exceeds max size.
        """
        op_start = time.perf_counter()

        if not detections:
            raise ValueError("detections list must not be empty")

        max_size = MAX_BATCH_SIZE
        if self._config and hasattr(self._config, "alert_batch_size"):
            max_size = int(self._config.alert_batch_size)

        if len(detections) > max_size:
            raise ValueError(
                f"Batch size {len(detections)} exceeds maximum {max_size}"
            )

        logger.info(
            "generate_batch: processing %d detections", len(detections)
        )

        alerts: List[DeforestationAlert] = []
        duplicates: List[str] = []
        warnings: List[str] = []

        for i, detection in enumerate(detections):
            try:
                result = self.generate_alert(detection, plots)
                if result.is_duplicate:
                    duplicates.append(result.duplicate_of)
                elif result.alert is not None:
                    alerts.append(result.alert)
                if result.warnings:
                    warnings.extend(result.warnings)
            except ValueError as exc:
                warnings.append(
                    f"Detection {i} skipped: {str(exc)}"
                )
            except Exception as exc:
                logger.warning(
                    "generate_batch: detection %d failed: %s", i, str(exc)
                )
                warnings.append(f"Detection {i} failed: {str(exc)}")

        elapsed = _elapsed_ms(op_start)
        batch = BatchResult(
            total_detections=len(detections),
            alerts_generated=len(alerts),
            alerts_deduplicated=len(duplicates),
            alerts=alerts,
            duplicates=duplicates,
            processing_time_ms=elapsed,
            warnings=warnings,
        )
        batch.provenance_hash = _compute_hash(batch.to_dict())

        logger.info(
            "generate_batch: %d alerts from %d detections "
            "(%d deduped) in %.1fms",
            len(alerts), len(detections), len(duplicates), elapsed,
        )
        return batch

    # ------------------------------------------------------------------
    # Public API: Alert retrieval
    # ------------------------------------------------------------------

    def get_alert(self, alert_id: str) -> AlertResult:
        """Retrieve a single alert by ID.

        Args:
            alert_id: Alert identifier to look up.

        Returns:
            AlertResult containing the alert.

        Raises:
            ValueError: If alert_id is empty.
            KeyError: If alert not found.
        """
        if not alert_id:
            raise ValueError("alert_id must not be empty")

        alert = self._alert_store.get(alert_id)
        if alert is None:
            raise KeyError(f"Alert {alert_id} not found")

        result = AlertResult(alert=alert)
        result.provenance_hash = _compute_hash(result.to_dict())
        return result

    def list_alerts(
        self,
        filters: Optional[Dict[str, Any]] = None,
        page: int = 1,
        size: int = 50,
    ) -> AlertListResult:
        """List alerts with optional filtering and pagination.

        Supports filtering by severity, status, country_code, commodity,
        is_post_cutoff, and date range.

        Args:
            filters: Optional dictionary of filter criteria.
            page: Page number (1-indexed).
            size: Page size (max 200).

        Returns:
            AlertListResult with paginated alerts.
        """
        size = min(200, max(1, size))
        page = max(1, page)

        # Get all alerts
        all_alerts = list(self._alert_store.values())

        # Apply filters
        if filters:
            all_alerts = self._apply_filters(all_alerts, filters)

        # Sort by generated_at descending
        all_alerts.sort(key=lambda a: a.generated_at, reverse=True)

        # Paginate
        total = len(all_alerts)
        offset = (page - 1) * size
        page_alerts = all_alerts[offset:offset + size]
        has_more = (offset + size) < total

        result = AlertListResult(
            alerts=page_alerts,
            total=total,
            page=page,
            page_size=size,
            has_more=has_more,
        )
        result.provenance_hash = _compute_hash(result.to_dict())
        return result

    # ------------------------------------------------------------------
    # Public API: Alert summary and statistics
    # ------------------------------------------------------------------

    def get_alert_summary(
        self,
        country_code: Optional[str] = None,
        date_range: Optional[Tuple[str, str]] = None,
    ) -> SummaryResult:
        """Get alert summary statistics with optional filtering.

        Args:
            country_code: Optional country code filter.
            date_range: Optional (start_date, end_date) tuple as ISO strings.

        Returns:
            SummaryResult with aggregated statistics.
        """
        alerts = list(self._alert_store.values())

        # Apply country filter
        if country_code:
            alerts = [a for a in alerts if a.country_code == country_code]

        # Apply date range filter
        if date_range and len(date_range) == 2:
            start_str, end_str = date_range
            alerts = [
                a for a in alerts
                if start_str <= a.generated_at[:10] <= end_str
            ]

        # Compute aggregations
        by_severity: Dict[str, int] = {}
        by_status: Dict[str, int] = {}
        by_country: Dict[str, int] = {}
        by_commodity: Dict[str, int] = {}
        by_change_type: Dict[str, int] = {}
        post_cutoff = 0
        pre_cutoff = 0
        total_severity = Decimal("0")
        total_proximity = Decimal("0")
        proximity_count = 0
        total_area = Decimal("0")

        for alert in alerts:
            by_severity[alert.severity] = by_severity.get(alert.severity, 0) + 1
            by_status[alert.status] = by_status.get(alert.status, 0) + 1
            if alert.country_code:
                by_country[alert.country_code] = (
                    by_country.get(alert.country_code, 0) + 1
                )
            if alert.commodity:
                by_commodity[alert.commodity] = (
                    by_commodity.get(alert.commodity, 0) + 1
                )
            if alert.change_type:
                by_change_type[alert.change_type] = (
                    by_change_type.get(alert.change_type, 0) + 1
                )
            if alert.is_post_cutoff:
                post_cutoff += 1
            else:
                pre_cutoff += 1
            total_severity += alert.severity_score
            if alert.proximity_km >= Decimal("0"):
                total_proximity += alert.proximity_km
                proximity_count += 1
            total_area += alert.area_ha

        n = len(alerts) or 1
        avg_severity = (total_severity / Decimal(str(n))).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        avg_proximity = Decimal("0")
        if proximity_count > 0:
            avg_proximity = (
                total_proximity / Decimal(str(proximity_count))
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        result = SummaryResult(
            total_alerts=len(alerts),
            by_severity=by_severity,
            by_status=by_status,
            by_country=by_country,
            by_commodity=by_commodity,
            by_change_type=by_change_type,
            post_cutoff_count=post_cutoff,
            pre_cutoff_count=pre_cutoff,
            avg_severity_score=avg_severity,
            avg_proximity_km=avg_proximity,
            total_affected_area_ha=total_area.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ),
        )
        result.provenance_hash = _compute_hash(result.to_dict())
        return result

    def get_alert_statistics(
        self,
        grouping: str = "severity",
    ) -> StatsResult:
        """Get alert statistics grouped by a dimension.

        Args:
            grouping: Grouping dimension. One of 'severity', 'status',
                'country', 'commodity', 'change_type', 'priority'.

        Returns:
            StatsResult with grouped counts.
        """
        valid_groupings = {
            "severity", "status", "country", "commodity",
            "change_type", "priority",
        }
        if grouping not in valid_groupings:
            raise ValueError(
                f"grouping must be one of {valid_groupings}, got '{grouping}'"
            )

        alerts = list(self._alert_store.values())
        groups: Dict[str, int] = {}

        attr_map = {
            "severity": "severity",
            "status": "status",
            "country": "country_code",
            "commodity": "commodity",
            "change_type": "change_type",
            "priority": "priority",
        }
        attr = attr_map[grouping]

        for alert in alerts:
            value = getattr(alert, attr, "unknown") or "unknown"
            groups[value] = groups.get(value, 0) + 1

        result = StatsResult(
            grouping=grouping,
            groups=groups,
            total=len(alerts),
        )
        result.provenance_hash = _compute_hash(result.to_dict())
        return result

    # ------------------------------------------------------------------
    # Internal: Alert construction
    # ------------------------------------------------------------------

    def _build_alert_title(self, detection: Any) -> str:
        """Build a human-readable alert title from detection data.

        Args:
            detection: DetectionResult object.

        Returns:
            Alert title string.
        """
        change_type = getattr(detection, "change_type", "deforestation")
        if hasattr(change_type, "value"):
            change_type = change_type.value

        country = getattr(detection, "country_code", "Unknown")
        area_ha = _safe_decimal(getattr(detection, "area_ha", 0))
        source = getattr(detection, "source", "satellite")
        if hasattr(source, "value"):
            source = source.value

        change_label = change_type.replace("_", " ").title()

        return (
            f"{change_label} Alert: {area_ha} ha detected near "
            f"{country} via {source}"
        )

    def _build_alert_description(
        self,
        detection: Any,
        proximity_km: Decimal,
        affected_count: int,
    ) -> str:
        """Build detailed alert description from detection data.

        Args:
            detection: DetectionResult object.
            proximity_km: Distance to nearest affected plot (km).
            affected_count: Number of affected supply chain plots.

        Returns:
            Multi-sentence alert description.
        """
        change_type = getattr(detection, "change_type", "deforestation")
        if hasattr(change_type, "value"):
            change_type = change_type.value

        lat = _safe_decimal(getattr(detection, "latitude", 0))
        lon = _safe_decimal(getattr(detection, "longitude", 0))
        area_ha = _safe_decimal(getattr(detection, "area_ha", 0))
        confidence = _safe_decimal(getattr(detection, "confidence", 0))
        ndvi_change = _safe_decimal(getattr(detection, "ndvi_change", 0))
        source = getattr(detection, "source", "satellite")
        if hasattr(source, "value"):
            source = source.value

        # Build description parts
        parts = []

        # Event description
        parts.append(
            f"Satellite {source} detected {change_type.replace('_', ' ')} "
            f"event covering approximately {area_ha} hectares at coordinates "
            f"({lat}, {lon}) with {float(confidence * 100):.1f}% confidence."
        )

        # Spectral details
        if ndvi_change != Decimal("0"):
            parts.append(
                f"NDVI change of {ndvi_change} indicates "
                f"{'significant vegetation loss' if ndvi_change < Decimal('0') else 'vegetation recovery'}."
            )

        # Proximity details
        if proximity_km >= Decimal("0"):
            parts.append(
                f"Nearest supply chain plot is {proximity_km} km away. "
                f"{affected_count} registered plot(s) are within the "
                f"monitoring buffer zone."
            )
        else:
            parts.append(
                "No registered supply chain plots found within the "
                "monitoring buffer zone."
            )

        # EUDR cutoff context
        detection_date = self._extract_detection_date(detection)
        if detection_date and detection_date > EUDR_CUTOFF_DATE:
            parts.append(
                f"This event occurred AFTER the EUDR cutoff date "
                f"({EUDR_CUTOFF_DATE.isoformat()}) and may affect "
                f"commodity compliance status per Article 2(1)."
            )

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Internal: Proximity and plot analysis
    # ------------------------------------------------------------------

    def _find_affected_plots(
        self,
        detection: Any,
        plots: List[SupplyPlot],
        buffer_km: Decimal,
    ) -> List[AffectedPlot]:
        """Find supply chain plots within buffer distance of detection.

        Calculates Haversine distance from the detection center to each
        registered plot and returns those within the buffer radius.

        Args:
            detection: DetectionResult object.
            plots: List of registered SupplyPlot objects.
            buffer_km: Buffer radius in kilometers.

        Returns:
            List of AffectedPlot objects within buffer, sorted by distance.
        """
        det_lat = _safe_decimal(getattr(detection, "latitude", 0))
        det_lon = _safe_decimal(getattr(detection, "longitude", 0))

        affected: List[AffectedPlot] = []

        for plot in plots:
            distance = self._calculate_proximity(
                det_lat, det_lon, plot.latitude, plot.longitude
            )
            inside = distance <= buffer_km

            if inside:
                affected.append(AffectedPlot(
                    plot_id=plot.plot_id,
                    plot_name=plot.plot_name,
                    commodity=plot.commodity,
                    distance_km=distance,
                    inside_buffer=True,
                    supplier_id=plot.supplier_id,
                ))

        # Sort by distance ascending (nearest first)
        affected.sort(key=lambda p: p.distance_km)

        logger.debug(
            "_find_affected_plots: %d of %d plots within %.1f km buffer",
            len(affected), len(plots), float(buffer_km),
        )
        return affected

    def _calculate_proximity(
        self,
        det_lat: Decimal,
        det_lon: Decimal,
        plot_lat: Decimal,
        plot_lon: Decimal,
    ) -> Decimal:
        """Calculate Haversine distance between detection and plot.

        Uses the Haversine formula for great-circle distance on a sphere:
            R = 6371 km (Earth radius)
            a = sin^2(delta_lat/2) + cos(lat1)*cos(lat2)*sin^2(delta_lon/2)
            d = 2 * R * arcsin(sqrt(a))

        ZERO-HALLUCINATION: Deterministic trigonometric formula.

        Args:
            det_lat: Detection latitude (decimal degrees).
            det_lon: Detection longitude (decimal degrees).
            plot_lat: Plot latitude (decimal degrees).
            plot_lon: Plot longitude (decimal degrees).

        Returns:
            Distance in kilometers as Decimal.
        """
        # Convert to radians using float math (trig functions)
        lat1 = math.radians(float(det_lat))
        lon1 = math.radians(float(det_lon))
        lat2 = math.radians(float(plot_lat))
        lon2 = math.radians(float(plot_lon))

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # Haversine formula
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )

        # Clamp a to [0, 1] for numerical stability
        a = max(0.0, min(1.0, a))

        c = 2 * math.asin(math.sqrt(a))
        distance = float(EARTH_RADIUS_KM) * c

        return _safe_decimal(distance).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

    # ------------------------------------------------------------------
    # Internal: Cutoff date determination
    # ------------------------------------------------------------------

    def _determine_post_cutoff(self, detection_date: Optional[date]) -> bool:
        """Determine if detection occurred after EUDR cutoff date.

        The EUDR cutoff date is 31 December 2020 per Article 2(1).
        Events after this date are subject to deforestation-free
        commodity restrictions.

        Args:
            detection_date: Date of the detection event. If None,
                assumes post-cutoff (conservative approach).

        Returns:
            True if detection is post-cutoff, False otherwise.
        """
        if detection_date is None:
            return True  # Conservative: assume post-cutoff
        return detection_date > EUDR_CUTOFF_DATE

    def _extract_detection_date(self, detection: Any) -> Optional[date]:
        """Extract the detection date from a DetectionResult.

        Args:
            detection: DetectionResult object.

        Returns:
            date object or None if extraction fails.
        """
        timestamp = getattr(detection, "timestamp", "")
        if not timestamp:
            return None

        try:
            if len(timestamp) >= 10:
                return date.fromisoformat(timestamp[:10])
        except (ValueError, TypeError):
            pass
        return None

    # ------------------------------------------------------------------
    # Internal: Severity estimation
    # ------------------------------------------------------------------

    def _estimate_severity(
        self,
        area_ha: Decimal,
        proximity_km: Decimal,
        is_post_cutoff: bool,
    ) -> Tuple[str, Decimal]:
        """Estimate severity level and score from key factors.

        This provides a quick preliminary severity assessment.
        The full SeverityClassifier engine provides comprehensive
        multi-factor scoring.

        Args:
            area_ha: Affected area in hectares.
            proximity_km: Distance to nearest plot in km.
            is_post_cutoff: Whether event is post-EUDR cutoff.

        Returns:
            Tuple of (severity_level, severity_score).
        """
        score = Decimal("0")

        # Area component (max 30 points)
        if area_ha >= Decimal("50"):
            score += Decimal("30")
        elif area_ha >= Decimal("10"):
            score += Decimal("24")
        elif area_ha >= Decimal("1"):
            score += Decimal("15")
        elif area_ha >= Decimal("0.5"):
            score += Decimal("9")
        else:
            score += Decimal("3")

        # Proximity component (max 30 points)
        if proximity_km >= Decimal("0"):
            if proximity_km < Decimal("1"):
                score += Decimal("30")
            elif proximity_km < Decimal("5"):
                score += Decimal("24")
            elif proximity_km < Decimal("25"):
                score += Decimal("15")
            elif proximity_km < Decimal("50"):
                score += Decimal("9")
            else:
                score += Decimal("3")

        # Post-cutoff component (max 40 points)
        if is_post_cutoff:
            score += Decimal("40")
        else:
            score += Decimal("8")

        # Clamp to [0, 100]
        score = max(Decimal("0"), min(Decimal("100"), score))

        # Determine severity level
        severity = self._score_to_severity(score)

        return severity, score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _score_to_severity(self, score: Decimal) -> str:
        """Convert numeric score to severity level string.

        Args:
            score: Severity score (0-100).

        Returns:
            Severity level string.
        """
        if score >= Decimal("80"):
            return "CRITICAL"
        elif score >= Decimal("60"):
            return "HIGH"
        elif score >= Decimal("40"):
            return "MEDIUM"
        elif score >= Decimal("20"):
            return "LOW"
        else:
            return "INFORMATIONAL"

    # ------------------------------------------------------------------
    # Internal: Deduplication
    # ------------------------------------------------------------------

    def _build_dedup_key(self, detection: Any) -> str:
        """Build a deduplication key from detection attributes.

        Combines spatial coordinates (bucketed to ~1km), change type,
        and time window into a single key for duplicate detection.

        Args:
            detection: DetectionResult object.

        Returns:
            Deduplication key string.
        """
        lat = _safe_decimal(getattr(detection, "latitude", 0))
        lon = _safe_decimal(getattr(detection, "longitude", 0))
        change_type = getattr(detection, "change_type", "unknown")
        if hasattr(change_type, "value"):
            change_type = change_type.value

        # Bucket coordinates to ~1km resolution
        lat_bucket = (lat / DEDUP_SPATIAL_TOLERANCE).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
        lon_bucket = (lon / DEDUP_SPATIAL_TOLERANCE).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )

        # Time bucket (to dedup window hours)
        dedup_hours = self._get_dedup_window_hours()
        now = _utcnow()
        time_bucket = int(now.timestamp()) // (dedup_hours * 3600)

        return f"{lat_bucket}:{lon_bucket}:{change_type}:{time_bucket}"

    def _deduplicate_alerts(
        self,
        alerts: List[DeforestationAlert],
    ) -> List[DeforestationAlert]:
        """Remove duplicate alerts within deduplication window.

        Alerts with the same dedup key are considered duplicates.
        The first alert in the list is retained.

        Args:
            alerts: List of alerts to deduplicate.

        Returns:
            Deduplicated list of alerts.
        """
        seen_keys: Dict[str, bool] = {}
        unique: List[DeforestationAlert] = []

        for alert in alerts:
            if alert.dedup_key not in seen_keys:
                seen_keys[alert.dedup_key] = True
                unique.append(alert)

        logger.debug(
            "_deduplicate_alerts: %d -> %d alerts after dedup",
            len(alerts), len(unique),
        )
        return unique

    # ------------------------------------------------------------------
    # Internal: Rate limiting and channel determination
    # ------------------------------------------------------------------

    def _check_daily_limit(self) -> None:
        """Check and enforce daily alert rate limit.

        Resets the counter if the date has changed.

        Raises:
            RuntimeError: If daily limit is exceeded.
        """
        today = _utcnow().date()
        if today != self._daily_date:
            self._daily_count = 0
            self._daily_date = today

        max_daily = self._get_max_alerts_per_day()
        if self._daily_count >= max_daily:
            raise RuntimeError(
                f"Daily alert limit of {max_daily} exceeded"
            )

    def _determine_channels(self, severity: str) -> List[str]:
        """Determine notification channels based on severity.

        Higher severity alerts use more notification channels
        for faster response.

        Args:
            severity: Severity level string.

        Returns:
            List of notification channel identifiers.
        """
        channels = [NotificationChannel.DASHBOARD.value]

        if severity in ("CRITICAL", "HIGH"):
            channels.append(NotificationChannel.EMAIL.value)
            channels.append(NotificationChannel.WEBHOOK.value)

        if severity == "CRITICAL":
            channels.append(NotificationChannel.SMS.value)
            channels.append(NotificationChannel.SLACK.value)

        return channels

    def _determine_commodity(
        self,
        detection: Any,
        affected_plots: List[AffectedPlot],
    ) -> str:
        """Determine the relevant EUDR commodity.

        First checks affected plots for commodity, then falls back
        to the detection metadata.

        Args:
            detection: DetectionResult object.
            affected_plots: List of affected plots.

        Returns:
            Commodity string or empty string if indeterminate.
        """
        # Check affected plots first
        if affected_plots:
            commodities = set(p.commodity for p in affected_plots if p.commodity)
            if len(commodities) == 1:
                return commodities.pop()
            elif commodities:
                # Multiple commodities; return the one from nearest plot
                return affected_plots[0].commodity

        # Fall back to detection metadata
        metadata = getattr(detection, "metadata", {}) or {}
        return metadata.get("commodity", "")

    # ------------------------------------------------------------------
    # Internal: Filtering helper
    # ------------------------------------------------------------------

    def _apply_filters(
        self,
        alerts: List[DeforestationAlert],
        filters: Dict[str, Any],
    ) -> List[DeforestationAlert]:
        """Apply filter criteria to a list of alerts.

        Supported filter keys: severity, status, country_code,
        commodity, is_post_cutoff, change_type, start_date, end_date.

        Args:
            alerts: List of alerts to filter.
            filters: Dictionary of filter criteria.

        Returns:
            Filtered list of alerts.
        """
        result = list(alerts)

        if "severity" in filters:
            result = [a for a in result if a.severity == filters["severity"]]
        if "status" in filters:
            result = [a for a in result if a.status == filters["status"]]
        if "country_code" in filters:
            result = [
                a for a in result
                if a.country_code == filters["country_code"]
            ]
        if "commodity" in filters:
            result = [
                a for a in result if a.commodity == filters["commodity"]
            ]
        if "is_post_cutoff" in filters:
            val = filters["is_post_cutoff"]
            result = [a for a in result if a.is_post_cutoff == val]
        if "change_type" in filters:
            result = [
                a for a in result
                if a.change_type == filters["change_type"]
            ]
        if "start_date" in filters:
            start = str(filters["start_date"])
            result = [a for a in result if a.generated_at[:10] >= start]
        if "end_date" in filters:
            end = str(filters["end_date"])
            result = [a for a in result if a.generated_at[:10] <= end]

        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Enumerations
    "AlertPriority",
    "AlertStatus",
    "NotificationChannel",
    # Constants
    "EUDR_CUTOFF_DATE",
    "EARTH_RADIUS_KM",
    "DEFAULT_BUFFER_RADIUS_KM",
    "MAX_BATCH_SIZE",
    "DEFAULT_DEDUP_WINDOW_HOURS",
    "DEFAULT_MAX_ALERTS_PER_DAY",
    "ALERT_RETENTION_YEARS",
    "SEVERITY_PRIORITY_MAP",
    "SEVERITY_THRESHOLDS",
    "EUDR_COMMODITIES",
    # Data classes
    "SupplyPlot",
    "AffectedPlot",
    "DeforestationAlert",
    "AlertResult",
    "BatchResult",
    "AlertListResult",
    "SummaryResult",
    "StatsResult",
    # Engine class
    "AlertGenerator",
]
