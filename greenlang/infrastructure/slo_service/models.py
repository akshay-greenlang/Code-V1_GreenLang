# -*- coding: utf-8 -*-
"""
SLO Service Models - OBS-005: SLO/SLI Definitions & Error Budget Management

Core data models for the SLO/SLI service including SLI, SLO, ErrorBudget,
BurnRateAlert, SLOReport, and associated enumerations. All models use
dataclasses for configuration types and provide full serialization support.

Example:
    >>> from greenlang.infrastructure.slo_service.models import (
    ...     SLO, SLI, SLIType, SLOWindow, BudgetStatus,
    ... )
    >>> sli = SLI(
    ...     name="api_availability",
    ...     sli_type=SLIType.AVAILABILITY,
    ...     good_query='sum(rate(http_requests_total{code!~"5.."}[5m]))',
    ...     total_query='sum(rate(http_requests_total[5m]))',
    ... )
    >>> slo = SLO(
    ...     slo_id="api-availability-99.9",
    ...     name="API Availability",
    ...     service="api-gateway",
    ...     sli=sli,
    ...     target=99.9,
    ... )

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-005 SLO/SLI Definitions & Error Budget Management
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SLIType(str, Enum):
    """Service Level Indicator measurement types."""

    AVAILABILITY = "availability"
    LATENCY = "latency"
    CORRECTNESS = "correctness"
    THROUGHPUT = "throughput"
    FRESHNESS = "freshness"


class SLOWindow(str, Enum):
    """SLO compliance measurement windows."""

    SEVEN_DAYS = "7d"
    TWENTY_EIGHT_DAYS = "28d"
    THIRTY_DAYS = "30d"
    NINETY_DAYS = "90d"
    CALENDAR_MONTH = "calendar_month"
    CALENDAR_QUARTER = "calendar_quarter"

    @property
    def minutes(self) -> int:
        """Return the window size in minutes.

        For calendar windows, return 30d or 90d equivalent.

        Returns:
            Window duration in minutes.
        """
        mapping = {
            "7d": 7 * 24 * 60,
            "28d": 28 * 24 * 60,
            "30d": 30 * 24 * 60,
            "90d": 90 * 24 * 60,
            "calendar_month": 30 * 24 * 60,
            "calendar_quarter": 90 * 24 * 60,
        }
        return mapping[self.value]

    @property
    def prometheus_duration(self) -> str:
        """Return the PromQL-compatible duration string.

        Returns:
            Duration string (e.g. ``7d``, ``30d``).
        """
        mapping = {
            "7d": "7d",
            "28d": "28d",
            "30d": "30d",
            "90d": "90d",
            "calendar_month": "30d",
            "calendar_quarter": "90d",
        }
        return mapping[self.value]


class BurnRateWindow(str, Enum):
    """Burn rate alerting window sizes (Google SRE Book)."""

    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"

    @property
    def long_window(self) -> str:
        """Return the long observation window for this burn rate tier.

        Returns:
            PromQL duration string.
        """
        mapping = {"fast": "1h", "medium": "6h", "slow": "3d"}
        return mapping[self.value]

    @property
    def short_window(self) -> str:
        """Return the short confirmation window for this burn rate tier.

        Returns:
            PromQL duration string.
        """
        mapping = {"fast": "5m", "medium": "30m", "slow": "6h"}
        return mapping[self.value]

    @property
    def threshold(self) -> float:
        """Return the burn rate threshold for this tier.

        Returns:
            Burn rate multiplier threshold.
        """
        mapping = {"fast": 14.4, "medium": 6.0, "slow": 1.0}
        return mapping[self.value]

    @property
    def long_window_minutes(self) -> int:
        """Return the long window duration in minutes.

        Returns:
            Duration in minutes.
        """
        mapping = {"fast": 60, "medium": 360, "slow": 4320}
        return mapping[self.value]

    @property
    def short_window_minutes(self) -> int:
        """Return the short window duration in minutes.

        Returns:
            Duration in minutes.
        """
        mapping = {"fast": 5, "medium": 30, "slow": 360}
        return mapping[self.value]

    @property
    def severity(self) -> str:
        """Return the alert severity for this burn rate tier.

        Returns:
            Severity string (critical, warning, info).
        """
        mapping = {"fast": "critical", "medium": "warning", "slow": "info"}
        return mapping[self.value]


class BudgetStatus(str, Enum):
    """Error budget health status."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    EXHAUSTED = "exhausted"


# ---------------------------------------------------------------------------
# SLI
# ---------------------------------------------------------------------------


@dataclass
class SLI:
    """Service Level Indicator definition.

    An SLI measures the ratio of good events to total events. The queries
    are PromQL expressions that return scalar values when evaluated over
    a time range.

    Attributes:
        name: Human-readable SLI name.
        sli_type: Type of SLI measurement.
        good_query: PromQL query returning good event count/rate.
        total_query: PromQL query returning total event count/rate.
        threshold_ms: Latency threshold in ms (for latency SLIs).
        unit: Measurement unit (e.g. ``ms``, ``requests``, ``bytes``).
        description: Human-readable description of what this SLI measures.
    """

    name: str
    sli_type: SLIType
    good_query: str
    total_query: str
    threshold_ms: Optional[float] = None
    unit: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the SLI to a plain dictionary.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        return {
            "name": self.name,
            "sli_type": self.sli_type.value,
            "good_query": self.good_query,
            "total_query": self.total_query,
            "threshold_ms": self.threshold_ms,
            "unit": self.unit,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SLI:
        """Deserialize an SLI from a dictionary.

        Args:
            data: Dictionary (e.g. from YAML or JSON).

        Returns:
            SLI instance.
        """
        sli_type = data.get("sli_type", "availability")
        if isinstance(sli_type, str):
            sli_type = SLIType(sli_type)

        return cls(
            name=data.get("name", ""),
            sli_type=sli_type,
            good_query=data.get("good_query", ""),
            total_query=data.get("total_query", ""),
            threshold_ms=data.get("threshold_ms"),
            unit=data.get("unit", ""),
            description=data.get("description", ""),
        )


# ---------------------------------------------------------------------------
# ErrorBudget
# ---------------------------------------------------------------------------


@dataclass
class ErrorBudget:
    """Real-time error budget state for an SLO.

    Attributes:
        slo_id: SLO identifier this budget belongs to.
        total_minutes: Total budget in minutes over the SLO window.
        consumed_minutes: Budget consumed so far.
        remaining_minutes: Budget remaining.
        remaining_percent: Percentage of budget remaining (0-100).
        consumed_percent: Percentage of budget consumed (0-100).
        status: Current budget health status.
        sli_value: Current SLI value (0-100 as a percentage).
        window: SLO window this budget was calculated for.
        calculated_at: Timestamp when this budget was calculated.
    """

    slo_id: str
    total_minutes: float
    consumed_minutes: float
    remaining_minutes: float
    remaining_percent: float
    consumed_percent: float
    status: BudgetStatus
    sli_value: float = 0.0
    window: str = "30d"
    calculated_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Set calculated_at if not provided."""
        if self.calculated_at is None:
            self.calculated_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the ErrorBudget to a plain dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "slo_id": self.slo_id,
            "total_minutes": round(self.total_minutes, 4),
            "consumed_minutes": round(self.consumed_minutes, 4),
            "remaining_minutes": round(self.remaining_minutes, 4),
            "remaining_percent": round(self.remaining_percent, 4),
            "consumed_percent": round(self.consumed_percent, 4),
            "status": self.status.value,
            "sli_value": round(self.sli_value, 6),
            "window": self.window,
            "calculated_at": self.calculated_at.isoformat() if self.calculated_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ErrorBudget:
        """Deserialize an ErrorBudget from a dictionary.

        Args:
            data: Dictionary.

        Returns:
            ErrorBudget instance.
        """
        calculated_at = data.get("calculated_at")
        if isinstance(calculated_at, str):
            calculated_at = datetime.fromisoformat(calculated_at)

        status = data.get("status", "healthy")
        if isinstance(status, str):
            status = BudgetStatus(status)

        return cls(
            slo_id=data.get("slo_id", ""),
            total_minutes=float(data.get("total_minutes", 0)),
            consumed_minutes=float(data.get("consumed_minutes", 0)),
            remaining_minutes=float(data.get("remaining_minutes", 0)),
            remaining_percent=float(data.get("remaining_percent", 100)),
            consumed_percent=float(data.get("consumed_percent", 0)),
            status=status,
            sli_value=float(data.get("sli_value", 0)),
            window=data.get("window", "30d"),
            calculated_at=calculated_at,
        )


# ---------------------------------------------------------------------------
# BurnRateAlert
# ---------------------------------------------------------------------------


@dataclass
class BurnRateAlert:
    """A burn rate alert firing for a specific SLO and window.

    Attributes:
        alert_id: Unique alert identifier.
        slo_id: SLO that triggered the alert.
        slo_name: Human-readable SLO name.
        burn_window: Which burn rate window triggered (fast/medium/slow).
        burn_rate_long: Burn rate over the long window.
        burn_rate_short: Burn rate over the short window.
        threshold: Burn rate threshold for this window.
        severity: Alert severity (critical/warning/info).
        service: Affected service name.
        message: Human-readable alert description.
        fired_at: Timestamp when the alert fired.
    """

    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    slo_id: str = ""
    slo_name: str = ""
    burn_window: str = ""
    burn_rate_long: float = 0.0
    burn_rate_short: float = 0.0
    threshold: float = 0.0
    severity: str = "warning"
    service: str = ""
    message: str = ""
    fired_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Set fired_at if not provided."""
        if self.fired_at is None:
            self.fired_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "alert_id": self.alert_id,
            "slo_id": self.slo_id,
            "slo_name": self.slo_name,
            "burn_window": self.burn_window,
            "burn_rate_long": round(self.burn_rate_long, 4),
            "burn_rate_short": round(self.burn_rate_short, 4),
            "threshold": self.threshold,
            "severity": self.severity,
            "service": self.service,
            "message": self.message,
            "fired_at": self.fired_at.isoformat() if self.fired_at else None,
        }


# ---------------------------------------------------------------------------
# SLO
# ---------------------------------------------------------------------------


@dataclass
class SLO:
    """Service Level Objective definition.

    An SLO combines an SLI with a target percentage over a measurement
    window, with optional labels and alerting configuration.

    Attributes:
        slo_id: Unique SLO identifier (slug-style).
        name: Human-readable SLO name.
        service: Service this SLO applies to.
        sli: Service Level Indicator definition.
        target: Target percentage (e.g. 99.9).
        window: Measurement window.
        description: Human-readable description.
        team: Owning team.
        labels: Additional labels for grouping/filtering.
        enabled: Whether this SLO is actively evaluated.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
        version: Configuration version (incremented on update).
        deleted: Soft-delete flag.
    """

    slo_id: str
    name: str
    service: str
    sli: SLI
    target: float = 99.9
    window: SLOWindow = SLOWindow.THIRTY_DAYS
    description: str = ""
    team: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    version: int = 1
    deleted: bool = False

    def __post_init__(self) -> None:
        """Set timestamps if not provided and parse enums."""
        now = datetime.now(timezone.utc)
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now
        if isinstance(self.window, str):
            self.window = SLOWindow(self.window)

    @property
    def error_budget_fraction(self) -> float:
        """Return the fraction of time allowed for errors.

        For a 99.9% target, this returns 0.001.

        Returns:
            Error budget as a decimal fraction.
        """
        return 1.0 - (self.target / 100.0)

    @property
    def window_minutes(self) -> int:
        """Return the SLO window duration in minutes.

        Returns:
            Window duration in minutes.
        """
        return self.window.minutes

    @property
    def safe_name(self) -> str:
        """Return a Prometheus-safe metric name.

        Replaces non-alphanumeric characters with underscores.

        Returns:
            Sanitized name string.
        """
        import re
        return re.sub(r"[^a-zA-Z0-9_]", "_", self.name.lower())

    def fingerprint(self) -> str:
        """Generate a unique fingerprint for deduplication.

        Returns:
            MD5 hex digest of slo_id + service + target.
        """
        raw = f"{self.slo_id}|{self.service}|{self.target}|{self.window.value}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the SLO to a plain dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "slo_id": self.slo_id,
            "name": self.name,
            "service": self.service,
            "sli": self.sli.to_dict(),
            "target": self.target,
            "window": self.window.value,
            "description": self.description,
            "team": self.team,
            "labels": dict(self.labels),
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "version": self.version,
            "deleted": self.deleted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SLO:
        """Deserialize an SLO from a dictionary.

        Args:
            data: Dictionary (e.g. from YAML or JSON).

        Returns:
            SLO instance.
        """
        sli_data = data.get("sli", {})
        sli = SLI.from_dict(sli_data) if isinstance(sli_data, dict) else sli_data

        window = data.get("window", "30d")
        if isinstance(window, str):
            window = SLOWindow(window)

        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)

        return cls(
            slo_id=data.get("slo_id", str(uuid.uuid4())),
            name=data.get("name", ""),
            service=data.get("service", ""),
            sli=sli,
            target=float(data.get("target", 99.9)),
            window=window,
            description=data.get("description", ""),
            team=data.get("team", ""),
            labels=data.get("labels", {}),
            enabled=data.get("enabled", True),
            created_at=created_at,
            updated_at=updated_at,
            version=int(data.get("version", 1)),
            deleted=data.get("deleted", False),
        )


# ---------------------------------------------------------------------------
# SLOReportEntry
# ---------------------------------------------------------------------------


@dataclass
class SLOReportEntry:
    """A single SLO entry within a compliance report.

    Attributes:
        slo_id: SLO identifier.
        slo_name: Human-readable SLO name.
        service: Service name.
        target: SLO target percentage.
        current_sli: Current SLI value.
        met: Whether the SLO is currently being met.
        budget_remaining_percent: Error budget remaining percentage.
        budget_status: Budget health status.
        trend: Performance trend (improving/stable/degrading).
        violations_count: Number of violations in the reporting period.
    """

    slo_id: str
    slo_name: str
    service: str
    target: float
    current_sli: float
    met: bool
    budget_remaining_percent: float
    budget_status: str
    trend: str = "stable"
    violations_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "slo_id": self.slo_id,
            "slo_name": self.slo_name,
            "service": self.service,
            "target": self.target,
            "current_sli": round(self.current_sli, 6),
            "met": self.met,
            "budget_remaining_percent": round(self.budget_remaining_percent, 4),
            "budget_status": self.budget_status,
            "trend": self.trend,
            "violations_count": self.violations_count,
        }


# ---------------------------------------------------------------------------
# SLOReport
# ---------------------------------------------------------------------------


@dataclass
class SLOReport:
    """A compliance report covering a time period.

    Attributes:
        report_id: Unique report identifier.
        report_type: Type of report (weekly, monthly, quarterly).
        period_start: Start of the reporting period.
        period_end: End of the reporting period.
        entries: Individual SLO report entries.
        overall_compliance_percent: Percentage of SLOs meeting target.
        total_slos: Total number of SLOs evaluated.
        slos_met: Number of SLOs meeting target.
        slos_not_met: Number of SLOs not meeting target.
        generated_at: Timestamp when report was generated.
    """

    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    report_type: str = "weekly"
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    entries: List[SLOReportEntry] = field(default_factory=list)
    overall_compliance_percent: float = 0.0
    total_slos: int = 0
    slos_met: int = 0
    slos_not_met: int = 0
    generated_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Set generated_at if not provided."""
        if self.generated_at is None:
            self.generated_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "report_id": self.report_id,
            "report_type": self.report_type,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "entries": [e.to_dict() for e in self.entries],
            "overall_compliance_percent": round(self.overall_compliance_percent, 4),
            "total_slos": self.total_slos,
            "slos_met": self.slos_met,
            "slos_not_met": self.slos_not_met,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SLOReport:
        """Deserialize from a dictionary.

        Args:
            data: Dictionary.

        Returns:
            SLOReport instance.
        """
        period_start = data.get("period_start")
        if isinstance(period_start, str):
            period_start = datetime.fromisoformat(period_start)

        period_end = data.get("period_end")
        if isinstance(period_end, str):
            period_end = datetime.fromisoformat(period_end)

        generated_at = data.get("generated_at")
        if isinstance(generated_at, str):
            generated_at = datetime.fromisoformat(generated_at)

        entries_data = data.get("entries", [])
        entries = []
        for entry_data in entries_data:
            entries.append(SLOReportEntry(
                slo_id=entry_data.get("slo_id", ""),
                slo_name=entry_data.get("slo_name", ""),
                service=entry_data.get("service", ""),
                target=float(entry_data.get("target", 0)),
                current_sli=float(entry_data.get("current_sli", 0)),
                met=entry_data.get("met", False),
                budget_remaining_percent=float(entry_data.get("budget_remaining_percent", 0)),
                budget_status=entry_data.get("budget_status", "healthy"),
                trend=entry_data.get("trend", "stable"),
                violations_count=int(entry_data.get("violations_count", 0)),
            ))

        return cls(
            report_id=data.get("report_id", str(uuid.uuid4())),
            report_type=data.get("report_type", "weekly"),
            period_start=period_start,
            period_end=period_end,
            entries=entries,
            overall_compliance_percent=float(data.get("overall_compliance_percent", 0)),
            total_slos=int(data.get("total_slos", 0)),
            slos_met=int(data.get("slos_met", 0)),
            slos_not_met=int(data.get("slos_not_met", 0)),
            generated_at=generated_at,
        )
