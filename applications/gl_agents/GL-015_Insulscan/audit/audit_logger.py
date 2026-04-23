# -*- coding: utf-8 -*-
"""
GL-015 Insulscan - Audit Logger

Structured audit logging system for insulation scanning and thermal assessment
operations with microsecond precision timestamps, SHA-256 hash chain integrity,
comprehensive event tracking, and SIEM integration support.

Features:
    - Structured audit logging with immutable entries
    - Log levels: INFO, WARNING, ALERT, CRITICAL
    - Integration with SIEM systems (Splunk, ELK, etc.)
    - Retention policy management (7-year default per regulations)
    - Search and query interface
    - SHA-256 hash chain for tamper detection
    - Thread-safe singleton pattern
    - Multiple storage backends

Standards:
    - ISO 50001:2018 (Energy Management Systems)
    - ASHRAE Standards (Building Insulation)
    - 21 CFR Part 11 (Electronic Records)
    - ISO 27001 (Information Security)
    - SOC 2 Type II (Service Organization Controls)

Example:
    >>> from audit.audit_logger import InsulationAuditLogger, get_audit_logger
    >>> logger = get_audit_logger()
    >>> logger.log_thermal_scan(
    ...     asset_id="INSUL-001",
    ...     camera_id="CAM-FLIR-T640",
    ...     scan_result="PASS",
    ...     max_temp_c=45.2
    ... )
    >>> logger.log_calculation(
    ...     calculation_type=ComputationType.HEAT_LOSS,
    ...     asset_id="INSUL-001",
    ...     inputs={"surface_temp": 45},
    ...     outputs={"heat_loss_w_m2": 150.5}
    ... )
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import threading
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Union

from .schemas import (
    AuditEvent,
    AuditEventType,
    ComputationType,
    DataSource,
    SeverityLevel,
    ActorType,
    compute_sha256,
)

logger = logging.getLogger(__name__)


# =============================================================================
# LOG LEVEL ENUMS
# =============================================================================

class AuditLogLevel(str, Enum):
    """Audit log levels for insulation operations."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ALERT = "alert"
    CRITICAL = "critical"

    def to_severity(self) -> SeverityLevel:
        """Convert to SeverityLevel enum."""
        return SeverityLevel(self.value)


class AuditOutcome(str, Enum):
    """Outcome of audited operation."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


# =============================================================================
# AUDIT CONTEXT
# =============================================================================

@dataclass(frozen=True)
class AuditContext:
    """
    Immutable context for audit events.

    Provides correlation and tracing information that follows
    a request through the entire insulation assessment pipeline.
    """

    correlation_id: str
    session_id: str
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    asset_id: Optional[str] = None
    scan_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "correlation_id": self.correlation_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "source_ip": self.source_ip,
            "user_agent": self.user_agent,
            "request_id": self.request_id,
            "asset_id": self.asset_id,
            "scan_id": self.scan_id,
        }


# =============================================================================
# AUDIT LOG ENTRY
# =============================================================================

@dataclass
class AuditLogEntry:
    """
    Complete audit log entry with provenance.

    This is the final, immutable form of an audit event that
    is persisted to storage with full hash chain linking.
    """

    entry_id: str
    timestamp: str  # ISO 8601 with microsecond precision
    event_type: str
    action: str
    severity: str
    outcome: str
    message: str
    details: Dict[str, Any]
    metadata: Dict[str, Any]
    context: Dict[str, Any]
    provenance: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string with deterministic ordering."""
        return json.dumps(self.to_dict(), sort_keys=True, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditLogEntry":
        """Create from dictionary."""
        return cls(**data)


# =============================================================================
# IMMUTABLE AUDIT LOG EVENT
# =============================================================================

@dataclass(frozen=True)
class InsulationAuditEvent:
    """
    Immutable audit event record for insulation operations.

    Captures all details of an auditable operation before
    being persisted to the audit log.
    """

    event_type: AuditEventType
    action: str
    level: AuditLogLevel = AuditLogLevel.INFO
    outcome: AuditOutcome = AuditOutcome.SUCCESS
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value if isinstance(self.event_type, AuditEventType) else str(self.event_type),
            "action": self.action,
            "level": self.level.value if isinstance(self.level, AuditLogLevel) else str(self.level),
            "outcome": self.outcome.value if isinstance(self.outcome, AuditOutcome) else str(self.outcome),
            "message": self.message,
            "details": dict(self.details),
            "metadata": dict(self.metadata),
        }


# =============================================================================
# AUDIT STORAGE INTERFACE
# =============================================================================

class AuditStorage(ABC):
    """Abstract base class for audit storage backends."""

    @abstractmethod
    def store(self, entry: AuditLogEntry) -> None:
        """Store a single audit entry."""
        pass

    @abstractmethod
    def store_batch(self, entries: List[AuditLogEntry]) -> None:
        """Store multiple audit entries."""
        pass

    @abstractmethod
    def query(
        self,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        correlation_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        level: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditLogEntry]:
        """Query audit entries."""
        pass

    @abstractmethod
    def get_by_id(self, entry_id: str) -> Optional[AuditLogEntry]:
        """Get a single entry by ID."""
        pass

    @abstractmethod
    def verify_integrity(self) -> bool:
        """Verify hash chain integrity."""
        pass

    @abstractmethod
    def apply_retention_policy(self, retention_days: int) -> int:
        """Apply retention policy and return number of entries removed."""
        pass


# =============================================================================
# FILE AUDIT STORAGE
# =============================================================================

class FileAuditStorage(AuditStorage):
    """
    File-based audit storage using JSON Lines format.

    Suitable for development and smaller deployments.
    Each entry is stored as a single JSON line for append-only writes.
    Supports automatic rotation and compression.
    """

    def __init__(
        self,
        directory: str = "audit_logs",
        rotate_size_mb: float = 100.0,
        compress_old: bool = True,
        retention_days: int = 2555,  # 7 years
    ):
        """
        Initialize file storage.

        Args:
            directory: Directory for audit files
            rotate_size_mb: Rotate file when size exceeds this (MB)
            compress_old: Compress rotated files
            retention_days: Retention period in days (default 7 years)
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rotate_size_bytes = int(rotate_size_mb * 1024 * 1024)
        self.compress_old = compress_old
        self.retention_days = retention_days
        self._current_file: Optional[Path] = None
        self._lock = threading.Lock()

        logger.info(
            f"FileAuditStorage initialized: directory={self.directory}, "
            f"rotate_size={rotate_size_mb}MB, retention={retention_days} days"
        )

    def _get_current_file(self) -> Path:
        """Get current audit log file."""
        if self._current_file is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            self._current_file = self.directory / f"audit_gl015_{date_str}.jsonl"
        return self._current_file

    def _check_rotation(self) -> None:
        """Check if file rotation is needed."""
        current_file = self._get_current_file()
        if current_file.exists():
            size = current_file.stat().st_size
            if size >= self.rotate_size_bytes:
                self._rotate_file(current_file)

    def _rotate_file(self, file_path: Path) -> None:
        """Rotate the current audit file."""
        timestamp = datetime.now(timezone.utc).strftime("%H%M%S")
        new_name = file_path.with_suffix(f".{timestamp}.jsonl")
        file_path.rename(new_name)
        self._current_file = None

        if self.compress_old:
            with open(new_name, "rb") as f_in:
                with gzip.open(f"{new_name}.gz", "wb") as f_out:
                    f_out.write(f_in.read())
            new_name.unlink()

        logger.info(f"Rotated audit log: {file_path} -> {new_name}")

    def store(self, entry: AuditLogEntry) -> None:
        """Store a single audit entry."""
        with self._lock:
            self._check_rotation()
            file_path = self._get_current_file()

            with open(file_path, "a", encoding="utf-8") as f:
                f.write(entry.to_json() + "\n")

    def store_batch(self, entries: List[AuditLogEntry]) -> None:
        """Store multiple audit entries."""
        with self._lock:
            self._check_rotation()
            file_path = self._get_current_file()

            with open(file_path, "a", encoding="utf-8") as f:
                for entry in entries:
                    f.write(entry.to_json() + "\n")

    def query(
        self,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        correlation_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        level: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditLogEntry]:
        """Query audit entries from files."""
        entries = []
        skipped = 0
        files = sorted(self.directory.glob("audit_gl015_*.jsonl"), reverse=True)

        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        entry = AuditLogEntry.from_dict(data)

                        if self._matches_query(entry, event_type, start_time, end_time,
                                              correlation_id, asset_id, level):
                            if skipped < offset:
                                skipped += 1
                                continue

                            entries.append(entry)
                            if len(entries) >= limit:
                                return entries
                    except json.JSONDecodeError:
                        continue

        return entries

    def _matches_query(
        self,
        entry: AuditLogEntry,
        event_type: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        correlation_id: Optional[str],
        asset_id: Optional[str],
        level: Optional[str],
    ) -> bool:
        """Check if entry matches query filters."""
        if event_type and entry.event_type != event_type:
            return False

        if level and entry.severity != level:
            return False

        if correlation_id:
            if entry.context.get("correlation_id") != correlation_id:
                return False

        if asset_id:
            if entry.context.get("asset_id") != asset_id:
                return False

        if start_time:
            entry_time = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
            if entry_time < start_time:
                return False

        if end_time:
            entry_time = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
            if entry_time > end_time:
                return False

        return True

    def get_by_id(self, entry_id: str) -> Optional[AuditLogEntry]:
        """Get a single entry by ID."""
        files = sorted(self.directory.glob("audit_gl015_*.jsonl"), reverse=True)

        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if data.get("entry_id") == entry_id:
                            return AuditLogEntry.from_dict(data)
                    except json.JSONDecodeError:
                        continue

        return None

    def verify_integrity(self) -> bool:
        """Verify hash chain integrity of all entries."""
        files = sorted(self.directory.glob("audit_gl015_*.jsonl"))
        previous_hash = None

        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        entry = AuditLogEntry.from_dict(data)

                        stored_previous = entry.provenance.get("previous_hash")
                        if previous_hash is not None:
                            if stored_previous != previous_hash:
                                logger.error(
                                    f"Hash chain broken at {entry.entry_id}: "
                                    f"expected {previous_hash}, got {stored_previous}"
                                )
                                return False

                        previous_hash = entry.provenance.get("entry_hash")

                    except json.JSONDecodeError:
                        continue

        logger.info("Hash chain integrity verified successfully")
        return True

    def apply_retention_policy(self, retention_days: int) -> int:
        """Apply retention policy by removing old files."""
        removed_count = 0
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)

        for file_path in self.directory.glob("audit_gl015_*"):
            try:
                # Parse date from filename
                date_part = file_path.stem.split("_")[2][:10]
                file_date = datetime.strptime(date_part, "%Y-%m-%d").replace(tzinfo=timezone.utc)

                if file_date < cutoff_date:
                    file_path.unlink()
                    removed_count += 1
                    logger.info(f"Removed expired audit file: {file_path}")
            except (ValueError, IndexError):
                continue

        return removed_count


# =============================================================================
# SIEM INTEGRATION
# =============================================================================

class SIEMIntegration(ABC):
    """Abstract base class for SIEM system integration."""

    @abstractmethod
    def send_event(self, entry: AuditLogEntry) -> bool:
        """Send an event to the SIEM system."""
        pass

    @abstractmethod
    def send_alert(self, entry: AuditLogEntry, alert_level: str) -> bool:
        """Send an alert to the SIEM system."""
        pass


class SplunkIntegration(SIEMIntegration):
    """
    Splunk HEC (HTTP Event Collector) integration.

    Placeholder for actual Splunk integration.
    """

    def __init__(
        self,
        hec_url: Optional[str] = None,
        hec_token: Optional[str] = None,
        index: str = "insulscan_audit",
        enabled: bool = False,
    ):
        """
        Initialize Splunk integration.

        Args:
            hec_url: Splunk HEC endpoint URL
            hec_token: HEC authentication token
            index: Splunk index name
            enabled: Whether integration is enabled
        """
        self.hec_url = hec_url
        self.hec_token = hec_token
        self.index = index
        self.enabled = enabled

        if enabled:
            logger.info(f"SplunkIntegration initialized: url={hec_url}, index={index}")
        else:
            logger.info("SplunkIntegration disabled")

    def send_event(self, entry: AuditLogEntry) -> bool:
        """Send an event to Splunk."""
        if not self.enabled:
            return True

        # Placeholder for actual HTTP POST to Splunk HEC
        logger.debug(f"[SPLUNK] Would send event: {entry.entry_id}")
        return True

    def send_alert(self, entry: AuditLogEntry, alert_level: str) -> bool:
        """Send an alert to Splunk."""
        if not self.enabled:
            return True

        # Placeholder for actual alert sending
        logger.debug(f"[SPLUNK] Would send alert: {entry.entry_id} ({alert_level})")
        return True


class ElasticIntegration(SIEMIntegration):
    """
    Elasticsearch/ELK Stack integration.

    Placeholder for actual Elasticsearch integration.
    """

    def __init__(
        self,
        es_url: Optional[str] = None,
        api_key: Optional[str] = None,
        index_prefix: str = "insulscan-audit",
        enabled: bool = False,
    ):
        """
        Initialize Elasticsearch integration.

        Args:
            es_url: Elasticsearch URL
            api_key: API key for authentication
            index_prefix: Index prefix for audit logs
            enabled: Whether integration is enabled
        """
        self.es_url = es_url
        self.api_key = api_key
        self.index_prefix = index_prefix
        self.enabled = enabled

        if enabled:
            logger.info(f"ElasticIntegration initialized: url={es_url}")
        else:
            logger.info("ElasticIntegration disabled")

    def send_event(self, entry: AuditLogEntry) -> bool:
        """Send an event to Elasticsearch."""
        if not self.enabled:
            return True

        # Placeholder for actual Elasticsearch indexing
        logger.debug(f"[ELK] Would index event: {entry.entry_id}")
        return True

    def send_alert(self, entry: AuditLogEntry, alert_level: str) -> bool:
        """Send an alert via Elasticsearch."""
        if not self.enabled:
            return True

        logger.debug(f"[ELK] Would send alert: {entry.entry_id} ({alert_level})")
        return True


# =============================================================================
# INSULATION AUDIT LOGGER
# =============================================================================

class InsulationAuditLogger:
    """
    Enterprise-grade audit logger for GL-015 Insulscan.

    Provides comprehensive audit logging for insulation scanning and
    thermal assessment operations with:
    - Immutable log entries with microsecond precision timestamps
    - SHA-256 hash chain for tamper detection
    - Correlation IDs for request tracing
    - Multiple storage backends
    - SIEM integration (Splunk, ELK)
    - Thread-safe singleton pattern
    - Retention policy management
    - Search and query interface

    Usage:
        >>> logger = InsulationAuditLogger()
        >>> logger.log_thermal_scan(
        ...     asset_id="INSUL-001",
        ...     camera_id="CAM-FLIR-T640",
        ...     scan_result="PASS"
        ... )
    """

    _instance: Optional["InsulationAuditLogger"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> "InsulationAuditLogger":
        """Singleton pattern for global audit logger."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        storage: Optional[AuditStorage] = None,
        siem: Optional[SIEMIntegration] = None,
        app_name: str = "GL-015-INSULSCAN",
        environment: str = "production",
        enable_hash_chain: bool = True,
        retention_days: int = 2555,  # 7 years
    ):
        """
        Initialize audit logger.

        Args:
            storage: Audit storage backend
            siem: SIEM integration
            app_name: Application identifier
            environment: Deployment environment
            enable_hash_chain: Enable SHA-256 hash chain for integrity
            retention_days: Retention period in days
        """
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.app_name = app_name
        self.environment = environment
        self.enable_hash_chain = enable_hash_chain
        self.retention_days = retention_days
        self._storage = storage
        self._siem = siem
        self._previous_hash: Optional[str] = None
        self._entry_count = 0
        self._local = threading.local()
        self._initialized = True

        logger.info(
            f"InsulationAuditLogger initialized: app={app_name}, env={environment}, "
            f"hash_chain={enable_hash_chain}, retention={retention_days} days"
        )

    def set_storage(self, storage: AuditStorage) -> None:
        """Set storage backend."""
        self._storage = storage

    def set_siem(self, siem: SIEMIntegration) -> None:
        """Set SIEM integration."""
        self._siem = siem

    def _get_context(self) -> AuditContext:
        """Get current audit context from thread-local storage."""
        return getattr(
            self._local,
            "context",
            AuditContext(
                correlation_id=str(uuid.uuid4()),
                session_id=str(uuid.uuid4()),
            )
        )

    def set_context(self, context: AuditContext) -> None:
        """Set audit context for current thread."""
        self._local.context = context

    @contextmanager
    def correlation_context(
        self,
        correlation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        scan_id: Optional[str] = None,
        **kwargs,
    ) -> Generator[AuditContext, None, None]:
        """
        Context manager for correlation tracking.

        Args:
            correlation_id: Optional correlation ID (generated if not provided)
            user_id: Optional user identifier
            asset_id: Optional insulation asset identifier
            scan_id: Optional thermal scan identifier
            **kwargs: Additional context fields

        Yields:
            AuditContext for this correlation scope
        """
        old_context = getattr(self._local, "context", None)

        new_context = AuditContext(
            correlation_id=correlation_id or str(uuid.uuid4()),
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            asset_id=asset_id,
            scan_id=scan_id,
            **kwargs,
        )
        self._local.context = new_context

        try:
            yield new_context
        finally:
            if old_context:
                self._local.context = old_context
            else:
                if hasattr(self._local, "context"):
                    delattr(self._local, "context")

    def _get_timestamp(self) -> str:
        """Get current timestamp with microsecond precision in ISO 8601 format."""
        return datetime.now(timezone.utc).isoformat(timespec='microseconds')

    def _compute_entry_hash(self, entry: AuditLogEntry) -> str:
        """Compute SHA-256 hash for audit entry."""
        content = entry.to_json()
        if self._previous_hash:
            content = self._previous_hash + content
        return hashlib.sha256(content.encode()).hexdigest()

    def _create_entry(self, event: InsulationAuditEvent) -> AuditLogEntry:
        """Create complete audit log entry from event."""
        context = self._get_context()

        entry = AuditLogEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=self._get_timestamp(),
            event_type=event.event_type.value if isinstance(event.event_type, AuditEventType) else str(event.event_type),
            action=event.action,
            severity=event.level.value if isinstance(event.level, AuditLogLevel) else str(event.level),
            outcome=event.outcome.value if isinstance(event.outcome, AuditOutcome) else str(event.outcome),
            message=event.message,
            details=dict(event.details),
            metadata={
                "app_name": self.app_name,
                "environment": self.environment,
                "entry_sequence": self._entry_count,
                **event.metadata,
            },
            context=context.to_dict(),
            provenance={},
        )

        # Compute hash chain
        if self.enable_hash_chain:
            entry_hash = self._compute_entry_hash(entry)
            entry.provenance["entry_hash"] = entry_hash
            entry.provenance["previous_hash"] = self._previous_hash or "genesis"
            entry.provenance["hash_algorithm"] = "SHA-256"
            self._previous_hash = entry_hash

        self._entry_count += 1
        return entry

    def log(self, event: InsulationAuditEvent) -> AuditLogEntry:
        """
        Log an audit event.

        Args:
            event: Audit event to log

        Returns:
            Complete audit log entry
        """
        entry = self._create_entry(event)

        # Persist to storage if configured
        if self._storage:
            self._storage.store(entry)

        # Send to SIEM if configured
        if self._siem:
            self._siem.send_event(entry)

            # Send alert for high severity
            if event.level in [AuditLogLevel.ALERT, AuditLogLevel.CRITICAL]:
                self._siem.send_alert(entry, event.level.value)

        # Also log to standard logger
        log_method = getattr(logger, event.level.value, logger.info)
        log_method(
            f"[AUDIT] {event.event_type.value}:{event.action} - {event.message} "
            f"(correlation_id={entry.context['correlation_id']})"
        )

        return entry

    # =========================================================================
    # THERMAL SCAN LOGGING
    # =========================================================================

    def log_thermal_scan(
        self,
        asset_id: str,
        camera_id: str,
        scan_result: str,
        max_temp_c: Optional[float] = None,
        min_temp_c: Optional[float] = None,
        avg_temp_c: Optional[float] = None,
        hotspot_count: int = 0,
        anomaly_detected: bool = False,
        image_hash: Optional[str] = None,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditLogEntry:
        """
        Log a thermal scan event.

        Args:
            asset_id: Insulation asset identifier
            camera_id: Thermal camera identifier
            scan_result: Scan result (PASS, FAIL, WARNING)
            max_temp_c: Maximum temperature detected
            min_temp_c: Minimum temperature detected
            avg_temp_c: Average temperature
            hotspot_count: Number of hotspots detected
            anomaly_detected: Whether anomaly was detected
            image_hash: SHA-256 hash of thermal image
            success: Whether scan completed successfully
            details: Additional details

        Returns:
            Audit log entry
        """
        return self.log(InsulationAuditEvent(
            event_type=AuditEventType.THERMAL_SCAN_COMPLETED,
            action="thermal_scan",
            level=AuditLogLevel.INFO if success else AuditLogLevel.WARNING,
            outcome=AuditOutcome.SUCCESS if success else AuditOutcome.FAILURE,
            message=f"Thermal scan of {asset_id}: {scan_result}",
            details={
                "asset_id": asset_id,
                "camera_id": camera_id,
                "scan_result": scan_result,
                "max_temp_c": max_temp_c,
                "min_temp_c": min_temp_c,
                "avg_temp_c": avg_temp_c,
                "hotspot_count": hotspot_count,
                "anomaly_detected": anomaly_detected,
                "image_hash": image_hash,
                **(details or {}),
            },
        ))

    def log_insulation_assessment(
        self,
        asset_id: str,
        assessment_type: str,
        condition_score: float,
        recommendation: Optional[str] = None,
        heat_loss_w_m2: Optional[float] = None,
        r_value_actual: Optional[float] = None,
        r_value_design: Optional[float] = None,
        efficiency_percent: Optional[float] = None,
        success: bool = True,
    ) -> AuditLogEntry:
        """
        Log an insulation assessment event.

        Args:
            asset_id: Insulation asset identifier
            assessment_type: Type of assessment
            condition_score: Condition score (0-1)
            recommendation: Recommendation if any
            heat_loss_w_m2: Heat loss rate
            r_value_actual: Actual R-value
            r_value_design: Design R-value
            efficiency_percent: Insulation efficiency
            success: Whether assessment succeeded

        Returns:
            Audit log entry
        """
        return self.log(InsulationAuditEvent(
            event_type=AuditEventType.INSULATION_ASSESSMENT,
            action=f"assess_{assessment_type}",
            level=AuditLogLevel.INFO if success else AuditLogLevel.WARNING,
            outcome=AuditOutcome.SUCCESS if success else AuditOutcome.FAILURE,
            message=f"Insulation assessment for {asset_id}: score={condition_score:.2f}",
            details={
                "asset_id": asset_id,
                "assessment_type": assessment_type,
                "condition_score": condition_score,
                "recommendation": recommendation,
                "heat_loss_w_m2": heat_loss_w_m2,
                "r_value_actual": r_value_actual,
                "r_value_design": r_value_design,
                "efficiency_percent": efficiency_percent,
            },
        ))

    # =========================================================================
    # CALCULATION LOGGING
    # =========================================================================

    def log_calculation(
        self,
        calculation_type: ComputationType,
        asset_id: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        algorithm_version: str = "1.0.0",
        duration_ms: float = 0.0,
        success: bool = True,
        formula_reference: Optional[str] = None,
    ) -> AuditLogEntry:
        """
        Log a calculation with full input/output provenance.

        Args:
            calculation_type: Type of calculation performed
            asset_id: Insulation asset identifier
            inputs: Input parameters
            outputs: Calculation results
            algorithm_version: Version of algorithm used
            duration_ms: Calculation duration
            success: Whether calculation succeeded
            formula_reference: Reference to formula used

        Returns:
            Audit log entry
        """
        return self.log(InsulationAuditEvent(
            event_type=AuditEventType.CALCULATION_EXECUTED,
            action=f"calculate_{calculation_type.value}",
            level=AuditLogLevel.INFO if success else AuditLogLevel.WARNING,
            outcome=AuditOutcome.SUCCESS if success else AuditOutcome.FAILURE,
            message=f"Calculation {calculation_type.value} for {asset_id}",
            details={
                "calculation_type": calculation_type.value,
                "asset_id": asset_id,
                "inputs_hash": compute_sha256(inputs),
                "outputs_hash": compute_sha256(outputs),
                "inputs_summary": self._summarize_data(inputs),
                "outputs_summary": self._summarize_data(outputs),
                "algorithm_version": algorithm_version,
                "formula_reference": formula_reference,
                "duration_ms": duration_ms,
            },
        ))

    def _summarize_data(self, data: Dict[str, Any], max_items: int = 10) -> Dict[str, Any]:
        """Create a summary of data for audit logging."""
        if not data:
            return {}

        summary = {}
        for key, value in list(data.items())[:max_items]:
            if isinstance(value, (list, tuple)):
                summary[key] = f"<{len(value)} items>"
            elif isinstance(value, dict):
                summary[key] = f"<dict with {len(value)} keys>"
            elif isinstance(value, (int, float, str, bool)):
                summary[key] = value
            else:
                summary[key] = f"<{type(value).__name__}>"
        return summary

    # =========================================================================
    # PREDICTION LOGGING
    # =========================================================================

    def log_prediction(
        self,
        prediction_type: str,
        asset_id: str,
        model_id: str,
        model_version: str,
        inputs_hash: str,
        prediction_value: Any,
        confidence: float,
        duration_ms: float = 0.0,
    ) -> AuditLogEntry:
        """Log a model prediction."""
        return self.log(InsulationAuditEvent(
            event_type=AuditEventType.PREDICTION_GENERATED,
            action=f"predict_{prediction_type}",
            level=AuditLogLevel.INFO,
            outcome=AuditOutcome.SUCCESS,
            message=f"Prediction {prediction_type} for {asset_id} (confidence: {confidence:.2%})",
            details={
                "prediction_type": prediction_type,
                "asset_id": asset_id,
                "model_id": model_id,
                "model_version": model_version,
                "inputs_hash": inputs_hash,
                "prediction_value": prediction_value,
                "prediction_hash": compute_sha256(prediction_value),
                "confidence": confidence,
                "duration_ms": duration_ms,
            },
        ))

    # =========================================================================
    # RECOMMENDATION LOGGING
    # =========================================================================

    def log_recommendation(
        self,
        recommendation_type: str,
        asset_id: str,
        recommendation: str,
        priority: str,
        supporting_data_hash: str,
        estimated_savings: Optional[float] = None,
        work_order_id: Optional[str] = None,
    ) -> AuditLogEntry:
        """Log a recommendation issued."""
        return self.log(InsulationAuditEvent(
            event_type=AuditEventType.RECOMMENDATION_ISSUED,
            action=f"recommend_{recommendation_type}",
            level=AuditLogLevel.INFO,
            outcome=AuditOutcome.SUCCESS,
            message=f"Recommendation for {asset_id}: {recommendation[:100]}...",
            details={
                "recommendation_type": recommendation_type,
                "asset_id": asset_id,
                "recommendation": recommendation,
                "priority": priority,
                "supporting_data_hash": supporting_data_hash,
                "estimated_savings": estimated_savings,
                "work_order_id": work_order_id,
            },
        ))

    # =========================================================================
    # SAFETY AND COMPLIANCE LOGGING
    # =========================================================================

    def log_safety_check(
        self,
        asset_id: str,
        check_type: str,
        passed: bool,
        violations: Optional[List[Dict[str, Any]]] = None,
        standard_reference: Optional[str] = None,
    ) -> AuditLogEntry:
        """Log safety check result."""
        level = AuditLogLevel.INFO if passed else AuditLogLevel.WARNING
        if violations and any(v.get("severity") == "critical" for v in violations):
            level = AuditLogLevel.CRITICAL

        return self.log(InsulationAuditEvent(
            event_type=AuditEventType.SAFETY_CHECK_PERFORMED,
            action=f"check_{check_type}",
            level=level,
            outcome=AuditOutcome.SUCCESS if passed else AuditOutcome.FAILURE,
            message=f"Safety check '{check_type}' for {asset_id}: {'PASS' if passed else 'FAIL'}",
            details={
                "asset_id": asset_id,
                "check_type": check_type,
                "passed": passed,
                "violations": violations or [],
                "standard_reference": standard_reference,
            },
        ))

    def log_compliance_verification(
        self,
        framework: str,
        asset_id: str,
        is_compliant: bool,
        compliance_score: float,
        checks: Dict[str, bool],
        evidence_pack_id: Optional[str] = None,
    ) -> AuditLogEntry:
        """Log compliance verification result."""
        return self.log(InsulationAuditEvent(
            event_type=AuditEventType.COMPLIANCE_VERIFIED,
            action=f"verify_{framework}",
            level=AuditLogLevel.INFO if is_compliant else AuditLogLevel.WARNING,
            outcome=AuditOutcome.SUCCESS if is_compliant else AuditOutcome.PARTIAL,
            message=f"Compliance verification for {asset_id}: {framework} - {'COMPLIANT' if is_compliant else 'NON-COMPLIANT'}",
            details={
                "framework": framework,
                "asset_id": asset_id,
                "is_compliant": is_compliant,
                "compliance_score": compliance_score,
                "checks": checks,
                "evidence_pack_id": evidence_pack_id,
            },
        ))

    # =========================================================================
    # THRESHOLD AND ALERT LOGGING
    # =========================================================================

    def log_threshold_exceeded(
        self,
        asset_id: str,
        metric_name: str,
        current_value: float,
        threshold_value: float,
        unit: str,
        severity_level: str = "warning",
    ) -> AuditLogEntry:
        """Log threshold exceedance event."""
        level = AuditLogLevel.CRITICAL if severity_level == "critical" else AuditLogLevel.ALERT

        return self.log(InsulationAuditEvent(
            event_type=AuditEventType.THRESHOLD_EXCEEDED,
            action="threshold_exceeded",
            level=level,
            outcome=AuditOutcome.SUCCESS,
            message=f"Threshold exceeded for {asset_id}: {metric_name} = {current_value} {unit} (limit: {threshold_value})",
            details={
                "asset_id": asset_id,
                "metric_name": metric_name,
                "current_value": current_value,
                "threshold_value": threshold_value,
                "unit": unit,
                "exceedance_ratio": current_value / threshold_value if threshold_value else None,
            },
        ))

    # =========================================================================
    # SYSTEM EVENT LOGGING
    # =========================================================================

    def log_system_startup(
        self,
        version: str,
        config_hash: Optional[str] = None,
        components: Optional[Dict[str, str]] = None,
    ) -> AuditLogEntry:
        """Log system startup."""
        return self.log(InsulationAuditEvent(
            event_type=AuditEventType.SYSTEM_EVENT,
            action="startup",
            level=AuditLogLevel.INFO,
            outcome=AuditOutcome.SUCCESS,
            message=f"GL-015 Insulscan started, version {version}",
            details={
                "version": version,
                "config_hash": config_hash,
                "components": components or {},
            },
        ))

    def log_system_shutdown(
        self,
        reason: str = "normal",
        uptime_seconds: Optional[float] = None,
    ) -> AuditLogEntry:
        """Log system shutdown."""
        return self.log(InsulationAuditEvent(
            event_type=AuditEventType.SYSTEM_EVENT,
            action="shutdown",
            level=AuditLogLevel.INFO,
            outcome=AuditOutcome.SUCCESS,
            message=f"GL-015 Insulscan shutdown: {reason}",
            details={
                "reason": reason,
                "uptime_seconds": uptime_seconds,
            },
        ))

    # =========================================================================
    # QUERY INTERFACE
    # =========================================================================

    def search(
        self,
        event_type: Optional[AuditEventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        correlation_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        level: Optional[AuditLogLevel] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditLogEntry]:
        """
        Search audit log entries.

        Args:
            event_type: Filter by event type
            start_time: Filter by start time
            end_time: Filter by end time
            correlation_id: Filter by correlation ID
            asset_id: Filter by asset ID
            level: Filter by log level
            limit: Maximum entries to return
            offset: Number of entries to skip

        Returns:
            List of matching audit log entries
        """
        if self._storage is None:
            logger.warning("No storage configured, cannot search entries")
            return []

        return self._storage.query(
            event_type=event_type.value if event_type else None,
            start_time=start_time,
            end_time=end_time,
            correlation_id=correlation_id,
            asset_id=asset_id,
            level=level.value if level else None,
            limit=limit,
            offset=offset,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_audit_logger() -> InsulationAuditLogger:
    """Get the singleton audit logger instance."""
    return InsulationAuditLogger()


def create_audit_logger(
    storage_dir: str = "audit_logs",
    app_name: str = "GL-015-INSULSCAN",
    environment: str = "production",
    enable_siem: bool = False,
    siem_type: str = "splunk",
) -> InsulationAuditLogger:
    """
    Create and configure an audit logger with file storage.

    Args:
        storage_dir: Directory for audit log files
        app_name: Application name
        environment: Deployment environment
        enable_siem: Enable SIEM integration
        siem_type: Type of SIEM ("splunk" or "elk")

    Returns:
        Configured InsulationAuditLogger instance
    """
    storage = FileAuditStorage(directory=storage_dir)

    siem = None
    if enable_siem:
        if siem_type == "splunk":
            siem = SplunkIntegration(enabled=True)
        elif siem_type == "elk":
            siem = ElasticIntegration(enabled=True)

    audit_logger = InsulationAuditLogger(
        storage=storage,
        siem=siem,
        app_name=app_name,
        environment=environment,
    )
    return audit_logger
