# -*- coding: utf-8 -*-
"""
GL-CBAM-APP Deadline Tracker Engine v1.1

Proactive deadline monitoring for CBAM quarterly report submissions.
Generates escalating alerts at configurable thresholds (T-30, T-14, T-7,
T-3, T-1 days) and tracks overdue reports.

Per EU CBAM Implementing Regulation 2023/1773 Article 6(1):
  - Quarterly reports must be submitted within 30 days after quarter end
  - Non-compliance may result in penalties per Article 16

All date calculations are deterministic Python arithmetic (ZERO HALLUCINATION).

Version: 1.1.0
Author: GreenLang CBAM Team
License: Proprietary
"""

import logging
import threading
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .models import (
    ALERT_THRESHOLDS,
    AMENDMENT_WINDOW_DAYS,
    FINAL_WARNING_DAYS,
    SUBMISSION_DEADLINE_DAYS,
    AlertLevel,
    DeadlineAlert,
    NotificationType,
    QuarterlyPeriod,
    QuarterlyReportPeriod,
    ReportStatus,
)
from .quarterly_scheduler import QuarterlySchedulerEngine

logger = logging.getLogger(__name__)


class DeadlineTrackerEngine:
    """
    Engine for monitoring CBAM report submission deadlines.

    Generates proactive alerts at configured thresholds before the
    submission deadline, tracks overdue reports, and maintains an
    alert history for compliance audit trails.

    Thread Safety:
        Uses threading.RLock for all mutable state access.

    Example:
        >>> tracker = DeadlineTrackerEngine()
        >>> alerts = tracker.check_upcoming_deadlines("NL123456789012")
        >>> for alert in alerts:
        ...     print(f"{alert.alert_level.value}: {alert.message}")
    """

    def __init__(self) -> None:
        """Initialize the deadline tracker engine."""
        self._lock = threading.RLock()
        self._scheduler = QuarterlySchedulerEngine()
        # importer_id -> {alert_id -> DeadlineAlert}
        self._alert_store: Dict[str, Dict[str, DeadlineAlert]] = {}
        # Track which (importer, period, threshold) combos have been alerted
        self._alerted_thresholds: Dict[str, set] = {}
        # importer_id -> report_status for tracking submissions
        self._submission_status: Dict[str, Dict[str, ReportStatus]] = {}
        logger.info("DeadlineTrackerEngine initialized")

    # ========================================================================
    # DEADLINE CHECKING
    # ========================================================================

    def check_upcoming_deadlines(
        self,
        importer_id: str,
        reference_date: Optional[date] = None,
        recipients: Optional[List[str]] = None,
    ) -> List[DeadlineAlert]:
        """
        Check all upcoming deadlines for an importer and generate alerts.

        Scans the current and next quarter for approaching deadlines.
        Generates alerts at configured thresholds (30, 14, 7, 3, 1 days).
        Deduplicates alerts to avoid repeated notifications for the same
        threshold.

        Args:
            importer_id: Importer EORI or internal identifier.
            reference_date: Date to check against (defaults to today).
            recipients: List of notification recipients (email/webhook).

        Returns:
            List of newly generated DeadlineAlert objects.
        """
        if reference_date is None:
            reference_date = date.today()

        if recipients is None:
            recipients = [f"{importer_id}@notifications.cbam"]

        new_alerts: List[DeadlineAlert] = []

        # Check current quarter and next quarter
        periods_to_check = self._get_relevant_periods(reference_date)

        for period in periods_to_check:
            # Skip if already submitted for this period
            period_key = f"{importer_id}:{period.period_label}"
            current_status = self._get_report_status(importer_id, period.period_label)
            if current_status in (ReportStatus.SUBMITTED, ReportStatus.ACCEPTED):
                continue

            days_remaining = self._scheduler.get_days_until_deadline(
                period, reference_date
            )

            # Check if overdue
            if days_remaining < 0:
                alert = self._maybe_create_alert(
                    importer_id=importer_id,
                    period=period,
                    alert_level=AlertLevel.CRITICAL,
                    notification_type=NotificationType.DEADLINE_OVERDUE,
                    days_remaining=days_remaining,
                    recipients=recipients,
                    threshold_key=f"overdue_{abs(days_remaining) // 7}",
                )
                if alert:
                    new_alerts.append(alert)
                continue

            # Check each threshold
            all_thresholds = list(ALERT_THRESHOLDS.items()) + [
                (AlertLevel.CRITICAL, FINAL_WARNING_DAYS)
            ]

            for alert_level, threshold_days in all_thresholds:
                if days_remaining <= threshold_days:
                    alert = self._maybe_create_alert(
                        importer_id=importer_id,
                        period=period,
                        alert_level=alert_level,
                        notification_type=NotificationType.DEADLINE_APPROACHING,
                        days_remaining=days_remaining,
                        recipients=recipients,
                        threshold_key=f"t_minus_{threshold_days}",
                    )
                    if alert:
                        new_alerts.append(alert)

        if new_alerts:
            logger.info(
                "Generated %d new deadline alerts for importer %s",
                len(new_alerts), importer_id,
            )

        return new_alerts

    def get_overdue_reports(
        self,
        importer_id: str,
        reference_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all overdue report submissions for an importer.

        Scans all quarters from CBAM start (Q4 2023) to the current quarter
        and identifies any that are past the submission deadline without a
        submitted/accepted status.

        Args:
            importer_id: Importer EORI or internal identifier.
            reference_date: Date to check against (defaults to today).

        Returns:
            List of dicts describing overdue reports:
            [
                {
                    "period_label": "2025Q4",
                    "submission_deadline": "2026-01-30",
                    "days_overdue": 5,
                    "amendment_deadline": "2026-03-01",
                    "amendment_window_open": True,
                    "status": "draft",
                },
            ]
        """
        if reference_date is None:
            reference_date = date.today()

        overdue: List[Dict[str, Any]] = []

        # Check last 8 quarters (2 years back)
        current_year = reference_date.year
        try:
            all_periods = self._scheduler.get_all_quarters(
                max(2023, current_year - 2), current_year
            )
        except ValueError:
            all_periods = []

        for period in all_periods:
            days_remaining = self._scheduler.get_days_until_deadline(
                period, reference_date
            )

            if days_remaining >= 0:
                continue  # Not yet past deadline

            # Check if submitted
            status = self._get_report_status(importer_id, period.period_label)
            if status in (ReportStatus.SUBMITTED, ReportStatus.ACCEPTED):
                continue

            amendment_days = self._scheduler.get_days_until_amendment_deadline(
                period, reference_date
            )

            overdue.append({
                "period_label": period.period_label,
                "submission_deadline": period.submission_deadline.isoformat(),
                "days_overdue": abs(days_remaining),
                "amendment_deadline": period.amendment_deadline.isoformat(),
                "amendment_window_open": amendment_days >= 0,
                "status": status.value if status else "not_started",
            })

        if overdue:
            logger.warning(
                "Importer %s has %d overdue reports", importer_id, len(overdue)
            )

        return overdue

    # ========================================================================
    # ALERT MANAGEMENT
    # ========================================================================

    def create_alert(
        self,
        period: QuarterlyReportPeriod,
        alert_level: AlertLevel,
        message: str,
        recipients: List[str],
        importer_id: str = "system",
        notification_type: NotificationType = NotificationType.DEADLINE_APPROACHING,
        days_remaining: int = 0,
    ) -> DeadlineAlert:
        """
        Create and store a deadline alert.

        Args:
            period: The quarterly period this alert relates to.
            alert_level: Severity level.
            message: Human-readable alert message.
            recipients: Notification recipient addresses.
            importer_id: Importer identifier for storage.
            notification_type: Type of notification.
            days_remaining: Days until/past deadline.

        Returns:
            Created DeadlineAlert.
        """
        alert_id = f"ALERT-{period.period_label}-{uuid.uuid4().hex[:8].upper()}"

        alert = DeadlineAlert(
            alert_id=alert_id,
            report_period=period,
            alert_level=alert_level,
            notification_type=notification_type,
            days_remaining=days_remaining,
            message=message,
            recipients=recipients,
            sent_at=None,
            acknowledged=False,
        )

        with self._lock:
            if importer_id not in self._alert_store:
                self._alert_store[importer_id] = {}
            self._alert_store[importer_id][alert_id] = alert

        logger.info(
            "Alert created: id=%s, level=%s, period=%s, days_remaining=%d",
            alert_id, alert_level.value, period.period_label, days_remaining,
        )

        return alert

    def acknowledge_alert(
        self,
        alert_id: str,
        importer_id: Optional[str] = None,
    ) -> DeadlineAlert:
        """
        Mark an alert as acknowledged by the user.

        Args:
            alert_id: The alert identifier to acknowledge.
            importer_id: Importer ID (for lookup optimization).

        Returns:
            Updated DeadlineAlert with acknowledged=True.

        Raises:
            ValueError: If alert not found.
        """
        with self._lock:
            alert = self._find_alert(alert_id, importer_id)
            if not alert:
                raise ValueError(f"Alert {alert_id} not found")

            updated = alert.model_copy(update={"acknowledged": True})

            # Update in store
            for imp_id, alerts in self._alert_store.items():
                if alert_id in alerts:
                    alerts[alert_id] = updated
                    break

            logger.info("Alert %s acknowledged", alert_id)
            return updated

    def get_alert_history(
        self,
        importer_id: str,
        limit: int = 100,
    ) -> List[DeadlineAlert]:
        """
        Get the alert history for an importer.

        Returns alerts sorted by creation time (most recent first).

        Args:
            importer_id: Importer identifier.
            limit: Maximum number of alerts to return.

        Returns:
            List of DeadlineAlert objects.
        """
        with self._lock:
            alerts = self._alert_store.get(importer_id, {})
            sorted_alerts = sorted(
                alerts.values(),
                key=lambda a: a.sent_at or datetime.min.replace(tzinfo=timezone.utc),
                reverse=True,
            )
            return sorted_alerts[:limit]

    # ========================================================================
    # COMPLIANCE CALENDAR
    # ========================================================================

    def get_compliance_calendar(
        self,
        importer_id: str,
        year: int,
        reference_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate a full compliance calendar for an importer.

        Includes all quarterly deadlines, current status, alert counts,
        and recommended actions for the specified year.

        Args:
            importer_id: Importer identifier.
            year: Calendar year.
            reference_date: Reference date for lifecycle calculation.

        Returns:
            List of quarterly calendar entries with compliance info.
        """
        if reference_date is None:
            reference_date = date.today()

        calendar_entries: List[Dict[str, Any]] = []

        try:
            periods = self._scheduler.get_all_quarters(year, year)
        except ValueError:
            return calendar_entries

        for period in periods:
            days_until_sub = self._scheduler.get_days_until_deadline(
                period, reference_date
            )
            days_until_amend = self._scheduler.get_days_until_amendment_deadline(
                period, reference_date
            )
            lifecycle = self._scheduler.get_report_lifecycle(period, reference_date)

            # Get report status
            status = self._get_report_status(importer_id, period.period_label)

            # Count alerts for this period
            alert_count = self._count_alerts_for_period(
                importer_id, period.period_label
            )

            # Determine compliance status
            if status in (ReportStatus.SUBMITTED, ReportStatus.ACCEPTED):
                compliance_status = "compliant"
            elif days_until_sub < 0:
                compliance_status = "overdue"
            elif days_until_sub <= 7:
                compliance_status = "at_risk"
            elif reference_date <= period.end_date:
                compliance_status = "in_progress"
            else:
                compliance_status = "pending"

            entry = {
                "period_label": period.period_label,
                "quarter": period.quarter.value,
                "start_date": period.start_date.isoformat(),
                "end_date": period.end_date.isoformat(),
                "submission_deadline": period.submission_deadline.isoformat(),
                "amendment_deadline": period.amendment_deadline.isoformat(),
                "is_transitional": period.is_transitional,
                "days_until_submission": days_until_sub,
                "days_until_amendment": days_until_amend,
                "report_status": status.value if status else "not_started",
                "compliance_status": compliance_status,
                "alert_count": alert_count,
                "lifecycle_stage": lifecycle["stage"],
                "recommended_action": lifecycle["recommended_action"],
            }
            calendar_entries.append(entry)

        logger.info(
            "Generated compliance calendar for %s, year %d (%d quarters)",
            importer_id, year, len(calendar_entries),
        )

        return calendar_entries

    # ========================================================================
    # STATUS TRACKING
    # ========================================================================

    def update_report_status(
        self,
        importer_id: str,
        period_label: str,
        status: ReportStatus,
    ) -> None:
        """
        Update the tracked submission status for a report.

        Args:
            importer_id: Importer identifier.
            period_label: Period label (e.g., "2025Q4").
            status: New report status.
        """
        with self._lock:
            if importer_id not in self._submission_status:
                self._submission_status[importer_id] = {}
            self._submission_status[importer_id][period_label] = status

            logger.info(
                "Report status updated: importer=%s, period=%s, status=%s",
                importer_id, period_label, status.value,
            )

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    def _get_relevant_periods(
        self,
        reference_date: date
    ) -> List[QuarterlyReportPeriod]:
        """
        Get the quarterly periods relevant for deadline checking.

        Checks the current quarter and the previous quarter (in case
        the previous quarter's deadline is still upcoming or recently passed).

        Args:
            reference_date: The reference date.

        Returns:
            List of relevant QuarterlyReportPeriod objects.
        """
        periods: List[QuarterlyReportPeriod] = []

        # Current quarter
        try:
            current = self._scheduler.get_quarter_for_date(reference_date)
            periods.append(current)
        except ValueError:
            pass

        # Previous quarter (may still have an open deadline)
        prev_date = reference_date - timedelta(days=90)
        try:
            previous = self._scheduler.get_quarter_for_date(prev_date)
            # Only include if deadline is within range
            days_to_amend = self._scheduler.get_days_until_amendment_deadline(
                previous, reference_date
            )
            if days_to_amend >= -30:  # Include up to 30 days past amendment deadline
                periods.append(previous)
        except ValueError:
            pass

        return periods

    def _maybe_create_alert(
        self,
        importer_id: str,
        period: QuarterlyReportPeriod,
        alert_level: AlertLevel,
        notification_type: NotificationType,
        days_remaining: int,
        recipients: List[str],
        threshold_key: str,
    ) -> Optional[DeadlineAlert]:
        """
        Create an alert if this threshold hasn't been alerted already.

        Deduplication key: (importer_id, period_label, threshold_key).

        Args:
            importer_id: Importer identifier.
            period: Quarterly period.
            alert_level: Alert severity.
            notification_type: Notification type.
            days_remaining: Days until/past deadline.
            recipients: Notification recipients.
            threshold_key: Deduplication key for this threshold.

        Returns:
            DeadlineAlert if newly created, None if already alerted.
        """
        with self._lock:
            dedup_key = f"{importer_id}:{period.period_label}:{threshold_key}"
            alerted = self._alerted_thresholds.get(importer_id, set())

            if dedup_key in alerted:
                return None

            # Generate message
            if days_remaining < 0:
                message = (
                    f"CBAM quarterly report for {period.period_label} "
                    f"({period.quarter.label}) is OVERDUE by "
                    f"{abs(days_remaining)} day(s). Submission deadline was "
                    f"{period.submission_deadline}. Submit immediately to "
                    f"avoid penalties per Regulation 2023/956 Article 16."
                )
            elif days_remaining == 0:
                message = (
                    f"CBAM quarterly report for {period.period_label} "
                    f"({period.quarter.label}) is DUE TODAY "
                    f"({period.submission_deadline}). Submit before end of day."
                )
            else:
                message = (
                    f"CBAM quarterly report for {period.period_label} "
                    f"({period.quarter.label}) is due in {days_remaining} day(s) "
                    f"on {period.submission_deadline}."
                )

            alert = self.create_alert(
                period=period,
                alert_level=alert_level,
                message=message,
                recipients=recipients,
                importer_id=importer_id,
                notification_type=notification_type,
                days_remaining=days_remaining,
            )

            # Mark as alerted
            if importer_id not in self._alerted_thresholds:
                self._alerted_thresholds[importer_id] = set()
            self._alerted_thresholds[importer_id].add(dedup_key)

            return alert

    def _find_alert(
        self,
        alert_id: str,
        importer_id: Optional[str] = None,
    ) -> Optional[DeadlineAlert]:
        """
        Find an alert by ID, optionally scoped to an importer.

        Args:
            alert_id: The alert identifier.
            importer_id: Optional importer scope for faster lookup.

        Returns:
            DeadlineAlert if found, None otherwise.
        """
        if importer_id:
            alerts = self._alert_store.get(importer_id, {})
            return alerts.get(alert_id)

        # Search all importers
        for alerts in self._alert_store.values():
            if alert_id in alerts:
                return alerts[alert_id]
        return None

    def _get_report_status(
        self,
        importer_id: str,
        period_label: str,
    ) -> Optional[ReportStatus]:
        """
        Get the tracked report status for a period.

        Args:
            importer_id: Importer identifier.
            period_label: Period label.

        Returns:
            ReportStatus if tracked, None otherwise.
        """
        with self._lock:
            statuses = self._submission_status.get(importer_id, {})
            return statuses.get(period_label)

    def _count_alerts_for_period(
        self,
        importer_id: str,
        period_label: str,
    ) -> int:
        """
        Count the number of alerts for a specific period.

        Args:
            importer_id: Importer identifier.
            period_label: Period label.

        Returns:
            Number of alerts.
        """
        with self._lock:
            alerts = self._alert_store.get(importer_id, {})
            count = 0
            for alert in alerts.values():
                if alert.report_period.period_label == period_label:
                    count += 1
            return count
