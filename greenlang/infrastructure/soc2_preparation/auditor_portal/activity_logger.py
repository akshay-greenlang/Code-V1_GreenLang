# -*- coding: utf-8 -*-
"""
Auditor Activity Logger - SEC-009 Phase 5

Comprehensive activity logging for auditor portal actions. Tracks all access,
downloads, and requests for security audit and compliance purposes. Includes
anomaly detection for suspicious behavior patterns.

Logged Activities:
    - Portal access (login, logout, session events)
    - Evidence access and downloads
    - Request creation and updates
    - Report access and downloads
    - Search queries

Example:
    >>> logger = AuditorActivityLogger()
    >>> await logger.log_access(
    ...     auditor_id=auditor_uuid,
    ...     resource="evidence/cc6-mfa-config.pdf",
    ...     action="view",
    ...     ip="192.168.1.100",
    ... )
    >>> anomalies = await logger.detect_anomalies(auditor_uuid)
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Anomaly detection thresholds
MAX_DOWNLOADS_PER_HOUR = 50
MAX_ACCESS_PER_MINUTE = 30
MAX_FAILED_ACTIONS_PER_HOUR = 10
UNUSUAL_TIME_HOURS = (0, 5)  # 12am - 5am considered unusual


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ActivityType(str, Enum):
    """Types of auditor activities."""

    # Session activities
    LOGIN = "login"
    LOGOUT = "logout"
    SESSION_EXPIRED = "session_expired"
    MFA_CHALLENGE = "mfa_challenge"
    MFA_VERIFIED = "mfa_verified"
    MFA_FAILED = "mfa_failed"

    # Evidence activities
    VIEW_EVIDENCE = "view_evidence"
    DOWNLOAD_EVIDENCE = "download_evidence"
    SEARCH_EVIDENCE = "search_evidence"

    # Request activities
    CREATE_REQUEST = "create_request"
    VIEW_REQUEST = "view_request"
    UPDATE_REQUEST = "update_request"
    ADD_COMMENT = "add_comment"

    # Report activities
    VIEW_REPORT = "view_report"
    DOWNLOAD_REPORT = "download_report"
    EXPORT_REPORT = "export_report"

    # Control testing
    VIEW_CONTROL_TEST = "view_control_test"
    VIEW_TEST_RESULTS = "view_test_results"

    # Dashboard
    VIEW_DASHBOARD = "view_dashboard"

    # Other
    SEARCH = "search"
    API_CALL = "api_call"


class AnomalyType(str, Enum):
    """Types of detected anomalies."""

    EXCESSIVE_DOWNLOADS = "excessive_downloads"
    """Too many downloads in a short time period."""

    EXCESSIVE_ACCESS = "excessive_access"
    """Too many access attempts in a short period."""

    UNUSUAL_TIME_ACCESS = "unusual_time_access"
    """Access during unusual hours (midnight - 5am)."""

    GEOGRAPHIC_ANOMALY = "geographic_anomaly"
    """Access from unexpected geographic location."""

    FAILED_ACTION_SPIKE = "failed_action_spike"
    """Multiple failed actions in short period."""

    NEW_IP_ADDRESS = "new_ip_address"
    """Access from previously unseen IP address."""

    LARGE_DOWNLOAD = "large_download"
    """Download of unusually large files."""

    SENSITIVE_ACCESS = "sensitive_access"
    """Access to highly sensitive resources."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class ActivityLog(BaseModel):
    """Record of an auditor activity.

    Attributes:
        log_id: Unique log identifier.
        auditor_id: ID of the auditor.
        activity_type: Type of activity.
        resource: Resource accessed (path or ID).
        action: Specific action taken.
        ip_address: Client IP address.
        user_agent: Client user agent.
        session_id: Associated session ID.
        success: Whether the action succeeded.
        error_message: Error message if action failed.
        details: Additional activity details.
        timestamp: Activity timestamp.
    """

    model_config = ConfigDict(extra="forbid")

    log_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique log identifier.",
    )
    auditor_id: str = Field(
        ...,
        description="ID of the auditor.",
    )
    activity_type: ActivityType = Field(
        ...,
        description="Type of activity.",
    )
    resource: str = Field(
        default="",
        max_length=512,
        description="Resource accessed.",
    )
    action: str = Field(
        default="",
        max_length=256,
        description="Specific action taken.",
    )
    ip_address: str = Field(
        default="",
        max_length=45,
        description="Client IP address.",
    )
    user_agent: str = Field(
        default="",
        max_length=512,
        description="Client user agent.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Associated session ID.",
    )
    success: bool = Field(
        default=True,
        description="Whether action succeeded.",
    )
    error_message: str = Field(
        default="",
        max_length=1024,
        description="Error message if failed.",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional details.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Activity timestamp.",
    )


class Anomaly(BaseModel):
    """Detected anomaly in auditor behavior.

    Attributes:
        anomaly_id: Unique anomaly identifier.
        auditor_id: Associated auditor ID.
        anomaly_type: Type of anomaly detected.
        description: Human-readable description.
        severity: Severity level (low, medium, high, critical).
        detected_at: Detection timestamp.
        related_logs: Related activity log IDs.
        resolved: Whether anomaly has been reviewed.
        resolution_notes: Notes from resolution.
    """

    model_config = ConfigDict(extra="forbid")

    anomaly_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique anomaly identifier.",
    )
    auditor_id: str = Field(
        ...,
        description="Associated auditor ID.",
    )
    anomaly_type: AnomalyType = Field(
        ...,
        description="Type of anomaly.",
    )
    description: str = Field(
        default="",
        max_length=1024,
        description="Human-readable description.",
    )
    severity: str = Field(
        default="medium",
        description="Severity: low, medium, high, critical.",
    )
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Detection timestamp.",
    )
    related_logs: List[str] = Field(
        default_factory=list,
        description="Related activity log IDs.",
    )
    resolved: bool = Field(
        default=False,
        description="Whether reviewed/resolved.",
    )
    resolution_notes: str = Field(
        default="",
        max_length=2048,
        description="Resolution notes.",
    )


class DateRange(BaseModel):
    """Date range for activity queries.

    Attributes:
        start: Start date (inclusive).
        end: End date (inclusive).
    """

    start: datetime = Field(..., description="Start date (inclusive).")
    end: datetime = Field(..., description="End date (inclusive).")


# ---------------------------------------------------------------------------
# Auditor Activity Logger
# ---------------------------------------------------------------------------


class AuditorActivityLogger:
    """Log and analyze auditor portal activities.

    Provides comprehensive activity logging with anomaly detection for
    security monitoring and compliance audit trails.

    Attributes:
        _logs: Activity logs by auditor_id.
        _all_logs: All logs in chronological order.
        _anomalies: Detected anomalies.
        _known_ips: Known IP addresses by auditor.
    """

    def __init__(self) -> None:
        """Initialize the activity logger."""
        self._logs: Dict[str, List[ActivityLog]] = defaultdict(list)
        self._all_logs: List[ActivityLog] = []
        self._anomalies: List[Anomaly] = []
        self._known_ips: Dict[str, set] = defaultdict(set)
        logger.info("AuditorActivityLogger initialized")

    # ------------------------------------------------------------------
    # Activity Logging
    # ------------------------------------------------------------------

    async def log_access(
        self,
        auditor_id: uuid.UUID,
        resource: str,
        action: str,
        ip: str,
        session_id: Optional[str] = None,
        user_agent: str = "",
        success: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ) -> ActivityLog:
        """Log an access activity.

        Args:
            auditor_id: Auditor identifier.
            resource: Resource being accessed.
            action: Action taken (view, download, etc.).
            ip: Client IP address.
            session_id: Session identifier.
            user_agent: Client user agent.
            success: Whether access was successful.
            details: Additional details.

        Returns:
            Created ActivityLog.
        """
        auditor_id_str = str(auditor_id)

        # Determine activity type from action
        activity_type = self._action_to_type(action)

        log_entry = ActivityLog(
            auditor_id=auditor_id_str,
            activity_type=activity_type,
            resource=resource,
            action=action,
            ip_address=ip,
            user_agent=user_agent,
            session_id=session_id,
            success=success,
            details=details or {},
        )

        self._logs[auditor_id_str].append(log_entry)
        self._all_logs.append(log_entry)

        # Track IP addresses
        if ip:
            is_new_ip = ip not in self._known_ips[auditor_id_str]
            self._known_ips[auditor_id_str].add(ip)

            # Check for new IP anomaly
            if is_new_ip and len(self._known_ips[auditor_id_str]) > 1:
                await self._create_anomaly(
                    auditor_id_str,
                    AnomalyType.NEW_IP_ADDRESS,
                    f"Access from new IP address: {ip}",
                    severity="low",
                    related_logs=[log_entry.log_id],
                )

        logger.debug(
            "Logged access: auditor=%s resource=%s action=%s",
            auditor_id_str[:8],
            resource[:50],
            action,
        )

        return log_entry

    async def log_download(
        self,
        auditor_id: uuid.UUID,
        file_id: uuid.UUID,
        filename: str,
        ip: str = "",
        file_size_bytes: int = 0,
        session_id: Optional[str] = None,
    ) -> ActivityLog:
        """Log a file download activity.

        Args:
            auditor_id: Auditor identifier.
            file_id: Downloaded file ID.
            filename: Downloaded filename.
            ip: Client IP address.
            file_size_bytes: File size in bytes.
            session_id: Session identifier.

        Returns:
            Created ActivityLog.
        """
        auditor_id_str = str(auditor_id)

        log_entry = ActivityLog(
            auditor_id=auditor_id_str,
            activity_type=ActivityType.DOWNLOAD_EVIDENCE,
            resource=str(file_id),
            action="download",
            ip_address=ip,
            session_id=session_id,
            details={
                "filename": filename,
                "file_size_bytes": file_size_bytes,
            },
        )

        self._logs[auditor_id_str].append(log_entry)
        self._all_logs.append(log_entry)

        # Check for large download anomaly (> 100MB)
        if file_size_bytes > 100 * 1024 * 1024:
            await self._create_anomaly(
                auditor_id_str,
                AnomalyType.LARGE_DOWNLOAD,
                f"Large file download: {filename} ({file_size_bytes / 1024 / 1024:.1f} MB)",
                severity="low",
                related_logs=[log_entry.log_id],
            )

        logger.info(
            "Logged download: auditor=%s file=%s size=%d",
            auditor_id_str[:8],
            filename,
            file_size_bytes,
        )

        return log_entry

    async def log_request(
        self,
        auditor_id: uuid.UUID,
        request_id: uuid.UUID,
        action: str,
        ip: str = "",
        session_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> ActivityLog:
        """Log a request-related activity.

        Args:
            auditor_id: Auditor identifier.
            request_id: Request identifier.
            action: Action taken (create, update, comment).
            ip: Client IP address.
            session_id: Session identifier.
            details: Additional details.

        Returns:
            Created ActivityLog.
        """
        auditor_id_str = str(auditor_id)

        activity_type = {
            "create": ActivityType.CREATE_REQUEST,
            "view": ActivityType.VIEW_REQUEST,
            "update": ActivityType.UPDATE_REQUEST,
            "comment": ActivityType.ADD_COMMENT,
        }.get(action, ActivityType.VIEW_REQUEST)

        log_entry = ActivityLog(
            auditor_id=auditor_id_str,
            activity_type=activity_type,
            resource=str(request_id),
            action=action,
            ip_address=ip,
            session_id=session_id,
            details=details or {},
        )

        self._logs[auditor_id_str].append(log_entry)
        self._all_logs.append(log_entry)

        logger.debug(
            "Logged request activity: auditor=%s request=%s action=%s",
            auditor_id_str[:8],
            str(request_id)[:8],
            action,
        )

        return log_entry

    def _action_to_type(self, action: str) -> ActivityType:
        """Map action string to ActivityType.

        Args:
            action: Action string.

        Returns:
            Corresponding ActivityType.
        """
        action_map = {
            "view": ActivityType.VIEW_EVIDENCE,
            "download": ActivityType.DOWNLOAD_EVIDENCE,
            "search": ActivityType.SEARCH,
            "login": ActivityType.LOGIN,
            "logout": ActivityType.LOGOUT,
            "view_report": ActivityType.VIEW_REPORT,
            "download_report": ActivityType.DOWNLOAD_REPORT,
            "view_dashboard": ActivityType.VIEW_DASHBOARD,
        }
        return action_map.get(action.lower(), ActivityType.API_CALL)

    # ------------------------------------------------------------------
    # Activity Retrieval
    # ------------------------------------------------------------------

    async def get_activity_report(
        self,
        auditor_id: uuid.UUID,
        period: DateRange,
    ) -> List[ActivityLog]:
        """Get activity logs for an auditor within a date range.

        Args:
            auditor_id: Auditor identifier.
            period: Date range to query.

        Returns:
            List of activity logs within the period.
        """
        auditor_id_str = str(auditor_id)
        logs = self._logs.get(auditor_id_str, [])

        return [
            log for log in logs
            if period.start <= log.timestamp <= period.end
        ]

    async def get_all_activity(
        self,
        period: Optional[DateRange] = None,
        activity_type: Optional[ActivityType] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ActivityLog]:
        """Get all activity logs with optional filtering.

        Args:
            period: Optional date range filter.
            activity_type: Optional activity type filter.
            limit: Maximum results.
            offset: Results offset.

        Returns:
            List of activity logs.
        """
        logs = self._all_logs

        if period:
            logs = [
                log for log in logs
                if period.start <= log.timestamp <= period.end
            ]

        if activity_type:
            logs = [log for log in logs if log.activity_type == activity_type]

        # Sort by timestamp descending
        logs = sorted(logs, key=lambda l: l.timestamp, reverse=True)

        return logs[offset : offset + limit]

    # ------------------------------------------------------------------
    # Anomaly Detection
    # ------------------------------------------------------------------

    async def detect_anomalies(
        self,
        auditor_id: uuid.UUID,
    ) -> List[Anomaly]:
        """Detect anomalies in auditor behavior.

        Analyzes recent activity for:
        - Excessive downloads
        - Excessive access attempts
        - Unusual time access
        - Failed action spikes

        Args:
            auditor_id: Auditor identifier.

        Returns:
            List of detected anomalies.
        """
        auditor_id_str = str(auditor_id)
        logs = self._logs.get(auditor_id_str, [])

        if not logs:
            return []

        anomalies: List[Anomaly] = []
        now = datetime.now(timezone.utc)
        one_hour_ago = now - timedelta(hours=1)
        one_minute_ago = now - timedelta(minutes=1)

        # Get recent logs
        recent_hour = [l for l in logs if l.timestamp >= one_hour_ago]
        recent_minute = [l for l in logs if l.timestamp >= one_minute_ago]

        # Check excessive downloads
        downloads = [l for l in recent_hour if l.activity_type == ActivityType.DOWNLOAD_EVIDENCE]
        if len(downloads) > MAX_DOWNLOADS_PER_HOUR:
            anomaly = await self._create_anomaly(
                auditor_id_str,
                AnomalyType.EXCESSIVE_DOWNLOADS,
                f"Excessive downloads: {len(downloads)} in the last hour (threshold: {MAX_DOWNLOADS_PER_HOUR})",
                severity="high",
                related_logs=[l.log_id for l in downloads],
            )
            anomalies.append(anomaly)

        # Check excessive access
        if len(recent_minute) > MAX_ACCESS_PER_MINUTE:
            anomaly = await self._create_anomaly(
                auditor_id_str,
                AnomalyType.EXCESSIVE_ACCESS,
                f"Excessive access: {len(recent_minute)} actions in the last minute",
                severity="high",
                related_logs=[l.log_id for l in recent_minute],
            )
            anomalies.append(anomaly)

        # Check unusual time access
        current_hour = now.hour
        if UNUSUAL_TIME_HOURS[0] <= current_hour < UNUSUAL_TIME_HOURS[1]:
            unusual_logs = [l for l in recent_hour if UNUSUAL_TIME_HOURS[0] <= l.timestamp.hour < UNUSUAL_TIME_HOURS[1]]
            if unusual_logs:
                anomaly = await self._create_anomaly(
                    auditor_id_str,
                    AnomalyType.UNUSUAL_TIME_ACCESS,
                    f"Access during unusual hours ({UNUSUAL_TIME_HOURS[0]}:00 - {UNUSUAL_TIME_HOURS[1]}:00 UTC)",
                    severity="low",
                    related_logs=[l.log_id for l in unusual_logs],
                )
                anomalies.append(anomaly)

        # Check failed action spikes
        failed = [l for l in recent_hour if not l.success]
        if len(failed) > MAX_FAILED_ACTIONS_PER_HOUR:
            anomaly = await self._create_anomaly(
                auditor_id_str,
                AnomalyType.FAILED_ACTION_SPIKE,
                f"Multiple failed actions: {len(failed)} failures in the last hour",
                severity="medium",
                related_logs=[l.log_id for l in failed],
            )
            anomalies.append(anomaly)

        return anomalies

    async def _create_anomaly(
        self,
        auditor_id: str,
        anomaly_type: AnomalyType,
        description: str,
        severity: str = "medium",
        related_logs: Optional[List[str]] = None,
    ) -> Anomaly:
        """Create and store an anomaly.

        Args:
            auditor_id: Auditor identifier.
            anomaly_type: Type of anomaly.
            description: Description.
            severity: Severity level.
            related_logs: Related log IDs.

        Returns:
            Created Anomaly.
        """
        anomaly = Anomaly(
            auditor_id=auditor_id,
            anomaly_type=anomaly_type,
            description=description,
            severity=severity,
            related_logs=related_logs or [],
        )

        self._anomalies.append(anomaly)

        logger.warning(
            "Anomaly detected: auditor=%s type=%s severity=%s",
            auditor_id[:8],
            anomaly_type.value,
            severity,
        )

        return anomaly

    async def get_anomalies(
        self,
        auditor_id: Optional[uuid.UUID] = None,
        unresolved_only: bool = False,
        severity: Optional[str] = None,
    ) -> List[Anomaly]:
        """Get detected anomalies with optional filtering.

        Args:
            auditor_id: Filter by auditor.
            unresolved_only: Only return unresolved anomalies.
            severity: Filter by severity.

        Returns:
            List of matching anomalies.
        """
        anomalies = self._anomalies

        if auditor_id:
            anomalies = [a for a in anomalies if a.auditor_id == str(auditor_id)]

        if unresolved_only:
            anomalies = [a for a in anomalies if not a.resolved]

        if severity:
            anomalies = [a for a in anomalies if a.severity == severity]

        return sorted(anomalies, key=lambda a: a.detected_at, reverse=True)

    async def resolve_anomaly(
        self,
        anomaly_id: str,
        notes: str,
    ) -> Optional[Anomaly]:
        """Mark an anomaly as resolved.

        Args:
            anomaly_id: Anomaly identifier.
            notes: Resolution notes.

        Returns:
            Updated Anomaly if found.
        """
        for anomaly in self._anomalies:
            if anomaly.anomaly_id == anomaly_id:
                anomaly.resolved = True
                anomaly.resolution_notes = notes
                logger.info("Resolved anomaly %s: %s", anomaly_id[:8], notes[:50])
                return anomaly

        return None

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    async def get_statistics(
        self,
        auditor_id: Optional[uuid.UUID] = None,
        period: Optional[DateRange] = None,
    ) -> Dict[str, Any]:
        """Get activity statistics.

        Args:
            auditor_id: Optional auditor filter.
            period: Optional date range.

        Returns:
            Dictionary with activity metrics.
        """
        if auditor_id:
            logs = self._logs.get(str(auditor_id), [])
        else:
            logs = self._all_logs

        if period:
            logs = [l for l in logs if period.start <= l.timestamp <= period.end]

        by_type: Dict[str, int] = {}
        by_hour: Dict[int, int] = {}
        success_count = 0
        failure_count = 0
        unique_resources: set = set()

        for log in logs:
            by_type[log.activity_type.value] = by_type.get(log.activity_type.value, 0) + 1
            by_hour[log.timestamp.hour] = by_hour.get(log.timestamp.hour, 0) + 1

            if log.success:
                success_count += 1
            else:
                failure_count += 1

            if log.resource:
                unique_resources.add(log.resource)

        return {
            "total_activities": len(logs),
            "by_type": by_type,
            "by_hour": dict(sorted(by_hour.items())),
            "success_count": success_count,
            "failure_count": failure_count,
            "unique_resources": len(unique_resources),
            "anomalies_detected": len([a for a in self._anomalies if not a.resolved]),
        }


__all__ = [
    "AuditorActivityLogger",
    "ActivityLog",
    "ActivityType",
    "Anomaly",
    "AnomalyType",
    "DateRange",
]
