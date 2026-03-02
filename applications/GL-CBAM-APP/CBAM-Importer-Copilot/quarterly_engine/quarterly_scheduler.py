# -*- coding: utf-8 -*-
"""
GL-CBAM-APP Quarterly Scheduler Engine v1.1

Thread-safe singleton engine for CBAM quarterly period calculation,
deadline management, and report lifecycle tracking.

Per EU CBAM Regulation 2023/956 Article 6:
  - Importers submit quarterly reports within 30 days after quarter end
  - Transitional period: Oct 2023 - Dec 2025 (simplified reporting)
  - Definitive period: Jan 2026+ (full reporting with certificates)

Per Implementing Regulation 2023/1773 Article 9:
  - Amendments allowed within 60 days after quarter end

All date calculations are deterministic Python arithmetic (ZERO HALLUCINATION).

Version: 1.1.0
Author: GreenLang CBAM Team
License: Proprietary
"""

import calendar
import logging
import threading
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .models import (
    AMENDMENT_WINDOW_DAYS,
    CBAM_REPORTING_START,
    DEFINITIVE_PERIOD_START,
    QUARTER_MONTHS,
    REPORT_ID_PREFIX,
    SUBMISSION_DEADLINE_DAYS,
    TRANSITIONAL_PERIOD_END,
    QuarterlyPeriod,
    QuarterlyReportPeriod,
    ReportStatus,
    VALID_STATUS_TRANSITIONS,
)

logger = logging.getLogger(__name__)


class QuarterlySchedulerEngine:
    """
    Thread-safe singleton engine for CBAM quarterly scheduling.

    Manages period calculation, deadline tracking, report lifecycle stages,
    and automated report generation triggers.

    This engine uses deterministic date arithmetic only -- no LLM calls,
    no external API lookups for dates. All calculations follow the EU CBAM
    Regulation 2023/956 and Implementing Regulation 2023/1773.

    Thread Safety:
        Uses threading.RLock to protect singleton creation and all mutable
        state. Safe for concurrent access from multiple API request handlers.

    Example:
        >>> engine = QuarterlySchedulerEngine()
        >>> period = engine.get_current_quarter()
        >>> print(f"Deadline: {period.submission_deadline}")
        >>> days_left = engine.get_days_until_deadline(period)
        >>> print(f"Days remaining: {days_left}")
    """

    _instance: Optional["QuarterlySchedulerEngine"] = None
    _lock: threading.RLock = threading.RLock()

    def __new__(cls) -> "QuarterlySchedulerEngine":
        """Thread-safe singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        """Initialize the scheduler engine (runs once due to singleton)."""
        with self._lock:
            if self._initialized:
                return
            self._initialized = True
            self._report_registry: Dict[str, Dict[str, Any]] = {}
            logger.info("QuarterlySchedulerEngine initialized (singleton)")

    # ========================================================================
    # PERIOD CALCULATION (DETERMINISTIC)
    # ========================================================================

    def get_current_quarter(self) -> QuarterlyReportPeriod:
        """
        Determine the current quarterly reporting period.

        Returns the period for the current calendar quarter based on today's date.

        Returns:
            QuarterlyReportPeriod for the current quarter.

        Example:
            >>> engine = QuarterlySchedulerEngine()
            >>> period = engine.get_current_quarter()
            >>> print(period.period_label)  # e.g., "2026Q1"
        """
        today = date.today()
        return self.get_quarter_for_date(today)

    def get_quarter_for_date(self, target_date: date) -> QuarterlyReportPeriod:
        """
        Get the quarterly reporting period that contains the given date.

        Args:
            target_date: The date to find the quarter for.

        Returns:
            QuarterlyReportPeriod containing the target date.

        Raises:
            ValueError: If the date is before CBAM reporting started (Oct 2023).

        Example:
            >>> engine = QuarterlySchedulerEngine()
            >>> period = engine.get_quarter_for_date(date(2025, 11, 15))
            >>> assert period.quarter == QuarterlyPeriod.Q4
        """
        if target_date < CBAM_REPORTING_START:
            raise ValueError(
                f"Date {target_date} is before CBAM reporting start ({CBAM_REPORTING_START})"
            )

        year = target_date.year
        month = target_date.month

        quarter = self._month_to_quarter(month)
        return self._build_period(year, quarter)

    def get_all_quarters(
        self,
        start_year: int,
        end_year: int
    ) -> List[QuarterlyReportPeriod]:
        """
        Generate all quarterly reporting periods for a range of years.

        Includes only periods that fall within the CBAM reporting timeline
        (starting Q4 2023).

        Args:
            start_year: First year to include (>= 2023).
            end_year: Last year to include.

        Returns:
            List of QuarterlyReportPeriod objects in chronological order.

        Raises:
            ValueError: If start_year > end_year or out of range.

        Example:
            >>> engine = QuarterlySchedulerEngine()
            >>> quarters = engine.get_all_quarters(2024, 2025)
            >>> assert len(quarters) == 8  # 4 quarters x 2 years
        """
        if start_year > end_year:
            raise ValueError(
                f"start_year ({start_year}) must be <= end_year ({end_year})"
            )
        if start_year < 2023:
            raise ValueError("CBAM reporting started in 2023 (Q4)")

        periods: List[QuarterlyReportPeriod] = []
        for year in range(start_year, end_year + 1):
            for quarter in QuarterlyPeriod:
                period = self._build_period(year, quarter)
                # Skip periods before CBAM started
                if period.end_date < CBAM_REPORTING_START:
                    continue
                periods.append(period)

        logger.info(
            "Generated %d quarterly periods from %d to %d",
            len(periods), start_year, end_year,
        )
        return periods

    def is_transitional_period(self, period: QuarterlyReportPeriod) -> bool:
        """
        Check if the given period falls within the CBAM transitional phase.

        The transitional period runs from Oct 2023 through Dec 2025.
        During this phase, reporting is simplified (no certificate purchases).

        Args:
            period: The quarterly period to check.

        Returns:
            True if the period is within the transitional phase.
        """
        return period.end_date <= TRANSITIONAL_PERIOD_END

    def is_definitive_period(self, period: QuarterlyReportPeriod) -> bool:
        """
        Check if the given period falls within the CBAM definitive phase.

        The definitive period begins Jan 2026. Full reporting requirements
        apply, including CBAM certificate purchases.

        Args:
            period: The quarterly period to check.

        Returns:
            True if the period is within the definitive phase.
        """
        return period.start_date >= DEFINITIVE_PERIOD_START

    # ========================================================================
    # DEADLINE CALCULATION (DETERMINISTIC)
    # ========================================================================

    def get_submission_deadline(self, period: QuarterlyReportPeriod) -> date:
        """
        Get the submission deadline for a quarterly period.

        Deadline is T+30 calendar days after the last day of the quarter,
        per Implementing Regulation 2023/1773 Article 6(1).

        Args:
            period: The quarterly reporting period.

        Returns:
            Submission deadline date.
        """
        return period.submission_deadline

    def get_amendment_deadline(self, period: QuarterlyReportPeriod) -> date:
        """
        Get the amendment window deadline for a quarterly period.

        Amendment window is T+60 calendar days after the last day of the quarter,
        per Implementing Regulation 2023/1773 Article 9.

        Args:
            period: The quarterly reporting period.

        Returns:
            Amendment deadline date.
        """
        return period.amendment_deadline

    def get_days_until_deadline(
        self,
        period: QuarterlyReportPeriod,
        reference_date: Optional[date] = None
    ) -> int:
        """
        Calculate the number of days remaining until the submission deadline.

        A positive result means the deadline is in the future.
        A negative result means the deadline has passed (overdue).
        Zero means the deadline is today.

        Args:
            period: The quarterly reporting period.
            reference_date: Date to calculate from (defaults to today).

        Returns:
            Number of days until deadline (negative if overdue).

        Example:
            >>> engine = QuarterlySchedulerEngine()
            >>> period = engine.get_current_quarter()
            >>> days = engine.get_days_until_deadline(period)
            >>> if days < 0:
            ...     print(f"OVERDUE by {abs(days)} days!")
        """
        if reference_date is None:
            reference_date = date.today()
        delta = period.submission_deadline - reference_date
        return delta.days

    def get_days_until_amendment_deadline(
        self,
        period: QuarterlyReportPeriod,
        reference_date: Optional[date] = None
    ) -> int:
        """
        Calculate the number of days remaining until the amendment deadline.

        Args:
            period: The quarterly reporting period.
            reference_date: Date to calculate from (defaults to today).

        Returns:
            Number of days until amendment deadline (negative if expired).
        """
        if reference_date is None:
            reference_date = date.today()
        delta = period.amendment_deadline - reference_date
        return delta.days

    # ========================================================================
    # REPORT GENERATION TRIGGERS
    # ========================================================================

    def should_generate_report(
        self,
        period: QuarterlyReportPeriod,
        reference_date: Optional[date] = None
    ) -> bool:
        """
        Determine if a quarterly report should be auto-generated.

        Reports are triggered T+15 days after the end of the quarter to allow
        sufficient time for data collection while leaving 15 days before
        the submission deadline.

        Args:
            period: The quarterly reporting period.
            reference_date: Date to check against (defaults to today).

        Returns:
            True if report generation should be triggered.
        """
        if reference_date is None:
            reference_date = date.today()

        trigger_date = period.end_date + timedelta(days=15)
        # Trigger on or after T+15, but only if not past submission deadline
        return trigger_date <= reference_date <= period.submission_deadline

    def get_report_lifecycle(
        self,
        period: QuarterlyReportPeriod,
        reference_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Get the current lifecycle stage for a quarterly period.

        Returns a comprehensive status dict including current stage,
        days remaining, and recommended actions.

        Args:
            period: The quarterly reporting period.
            reference_date: Reference date (defaults to today).

        Returns:
            Dict with lifecycle information:
            {
                "period_label": "2025Q4",
                "stage": "data_collection"|"report_generation"|"review"|"overdue"|"closed",
                "days_until_submission": int,
                "days_until_amendment": int,
                "submission_deadline": "2026-01-30",
                "amendment_deadline": "2026-03-01",
                "is_transitional": bool,
                "recommended_action": str,
            }
        """
        if reference_date is None:
            reference_date = date.today()

        days_until_sub = self.get_days_until_deadline(period, reference_date)
        days_until_amend = self.get_days_until_amendment_deadline(
            period, reference_date
        )

        # Determine stage
        if reference_date <= period.end_date:
            stage = "data_collection"
            action = "Continue collecting shipment data for this quarter"
        elif reference_date <= period.end_date + timedelta(days=15):
            stage = "data_collection"
            action = "Finalize data collection; report generation imminent"
        elif days_until_sub > 0:
            stage = "report_generation"
            action = f"Generate and review report; {days_until_sub} days until deadline"
        elif days_until_sub == 0:
            stage = "review"
            action = "DEADLINE TODAY - submit report immediately"
        elif days_until_amend > 0:
            stage = "overdue"
            action = f"OVERDUE by {abs(days_until_sub)} days; amendment window still open"
        else:
            stage = "closed"
            action = "Amendment window closed; contact competent authority for late submission"

        return {
            "period_label": period.period_label,
            "stage": stage,
            "days_until_submission": days_until_sub,
            "days_until_amendment": days_until_amend,
            "submission_deadline": period.submission_deadline.isoformat(),
            "amendment_deadline": period.amendment_deadline.isoformat(),
            "is_transitional": period.is_transitional,
            "recommended_action": action,
        }

    # ========================================================================
    # REPORT REGISTRATION & TRIGGERING
    # ========================================================================

    def trigger_report_generation(
        self,
        period: QuarterlyReportPeriod,
        importer_id: str
    ) -> str:
        """
        Register and trigger a quarterly report generation job.

        Creates a report registration entry and returns a unique job ID
        that can be used to track the generation progress.

        Args:
            period: The quarterly reporting period.
            importer_id: Importer EORI or internal identifier.

        Returns:
            Unique job identifier for tracking.

        Raises:
            ValueError: If report already exists for this period and importer.
        """
        with self._lock:
            registry_key = f"{period.period_label}:{importer_id}"

            if registry_key in self._report_registry:
                existing = self._report_registry[registry_key]
                if existing["status"] not in ("failed", "cancelled"):
                    raise ValueError(
                        f"Report generation already registered for "
                        f"{period.period_label} / {importer_id} "
                        f"(status: {existing['status']})"
                    )

            job_id = f"JOB-{period.period_label}-{uuid.uuid4().hex[:8].upper()}"
            timestamp = datetime.now(timezone.utc)

            report_id = (
                f"{REPORT_ID_PREFIX}-{period.period_label}-"
                f"{importer_id[:16]}-{timestamp.strftime('%Y%m%d%H%M%S')}"
            )

            self._report_registry[registry_key] = {
                "job_id": job_id,
                "report_id": report_id,
                "period_label": period.period_label,
                "importer_id": importer_id,
                "status": "pending",
                "triggered_at": timestamp.isoformat(),
                "completed_at": None,
            }

            logger.info(
                "Report generation triggered: job_id=%s, period=%s, importer=%s",
                job_id, period.period_label, importer_id,
            )
            return job_id

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a report generation job.

        Args:
            job_id: The job identifier returned by trigger_report_generation.

        Returns:
            Job status dict or None if not found.
        """
        with self._lock:
            for entry in self._report_registry.values():
                if entry["job_id"] == job_id:
                    return dict(entry)
            return None

    def update_job_status(
        self,
        job_id: str,
        status: str,
        report_id: Optional[str] = None
    ) -> None:
        """
        Update the status of a report generation job.

        Args:
            job_id: The job identifier.
            status: New status (pending, running, completed, failed, cancelled).
            report_id: Updated report ID if available.
        """
        with self._lock:
            for entry in self._report_registry.values():
                if entry["job_id"] == job_id:
                    entry["status"] = status
                    if report_id:
                        entry["report_id"] = report_id
                    if status in ("completed", "failed", "cancelled"):
                        entry["completed_at"] = datetime.now(timezone.utc).isoformat()
                    logger.info(
                        "Job %s status updated to %s", job_id, status
                    )
                    return
            logger.warning("Job %s not found in registry", job_id)

    # ========================================================================
    # REPORTING CALENDAR
    # ========================================================================

    def get_reporting_calendar(self, year: int) -> List[Dict[str, Any]]:
        """
        Generate a full year reporting calendar with all deadlines and milestones.

        Useful for compliance planning and calendar integration.

        Args:
            year: The calendar year to generate for.

        Returns:
            List of dicts, one per quarter, with dates and status:
            [
                {
                    "period_label": "2025Q1",
                    "quarter": "Q1",
                    "start_date": "2025-01-01",
                    "end_date": "2025-03-31",
                    "data_collection_end": "2025-04-15",
                    "submission_deadline": "2025-04-30",
                    "amendment_deadline": "2025-05-30",
                    "is_transitional": True,
                    "lifecycle": {...},
                },
                ...
            ]
        """
        quarters = self.get_all_quarters(year, year)
        calendar_entries: List[Dict[str, Any]] = []

        for period in quarters:
            lifecycle = self.get_report_lifecycle(period)
            data_collection_end = period.end_date + timedelta(days=15)

            entry = {
                "period_label": period.period_label,
                "quarter": period.quarter.value,
                "start_date": period.start_date.isoformat(),
                "end_date": period.end_date.isoformat(),
                "data_collection_end": data_collection_end.isoformat(),
                "submission_deadline": period.submission_deadline.isoformat(),
                "amendment_deadline": period.amendment_deadline.isoformat(),
                "is_transitional": period.is_transitional,
                "is_definitive": period.is_definitive,
                "lifecycle": lifecycle,
            }
            calendar_entries.append(entry)

        logger.info("Generated reporting calendar for year %d (%d quarters)", year, len(calendar_entries))
        return calendar_entries

    def get_next_deadline(
        self,
        reference_date: Optional[date] = None
    ) -> Optional[QuarterlyReportPeriod]:
        """
        Get the period with the next upcoming submission deadline.

        Args:
            reference_date: Reference date (defaults to today).

        Returns:
            QuarterlyReportPeriod with the next deadline, or None if none upcoming.
        """
        if reference_date is None:
            reference_date = date.today()

        # Check current quarter and up to 2 quarters ahead
        for offset_months in range(0, 9, 3):
            check_date = reference_date + timedelta(days=offset_months * 30)
            try:
                period = self.get_quarter_for_date(check_date)
                if period.submission_deadline >= reference_date:
                    return period
            except ValueError:
                continue
        return None

    # ========================================================================
    # STATUS TRANSITION VALIDATION
    # ========================================================================

    def validate_status_transition(
        self,
        current_status: ReportStatus,
        target_status: ReportStatus
    ) -> bool:
        """
        Validate that a report status transition is allowed.

        Args:
            current_status: Current report status.
            target_status: Desired new status.

        Returns:
            True if transition is valid.

        Raises:
            ValueError: If transition is not allowed.
        """
        allowed = VALID_STATUS_TRANSITIONS.get(current_status, [])
        if target_status not in allowed:
            raise ValueError(
                f"Invalid status transition: {current_status.value} -> "
                f"{target_status.value}. Allowed from {current_status.value}: "
                f"{[s.value for s in allowed]}"
            )
        return True

    # ========================================================================
    # INTERNAL HELPERS (DETERMINISTIC)
    # ========================================================================

    def _month_to_quarter(self, month: int) -> QuarterlyPeriod:
        """
        Convert a month number (1-12) to a QuarterlyPeriod.

        Args:
            month: Month number (1 = January, 12 = December).

        Returns:
            Corresponding QuarterlyPeriod.
        """
        if 1 <= month <= 3:
            return QuarterlyPeriod.Q1
        elif 4 <= month <= 6:
            return QuarterlyPeriod.Q2
        elif 7 <= month <= 9:
            return QuarterlyPeriod.Q3
        else:
            return QuarterlyPeriod.Q4

    def _build_period(
        self,
        year: int,
        quarter: QuarterlyPeriod
    ) -> QuarterlyReportPeriod:
        """
        Build a QuarterlyReportPeriod with computed dates and deadlines.

        All date calculations are deterministic Python arithmetic.

        Args:
            year: Reporting year.
            quarter: Quarter identifier.

        Returns:
            Fully populated QuarterlyReportPeriod.
        """
        months = QUARTER_MONTHS[quarter]
        start_month = months[0]
        end_month = months[2]

        start_date = date(year, start_month, 1)
        last_day = calendar.monthrange(year, end_month)[1]
        end_date = date(year, end_month, last_day)

        submission_deadline = end_date + timedelta(days=SUBMISSION_DEADLINE_DAYS)
        amendment_deadline = end_date + timedelta(days=AMENDMENT_WINDOW_DAYS)

        is_transitional = end_date <= TRANSITIONAL_PERIOD_END

        return QuarterlyReportPeriod(
            year=year,
            quarter=quarter,
            start_date=start_date,
            end_date=end_date,
            submission_deadline=submission_deadline,
            amendment_deadline=amendment_deadline,
            is_transitional=is_transitional,
        )

    # ========================================================================
    # SINGLETON RESET (TESTING ONLY)
    # ========================================================================

    @classmethod
    def _reset_singleton(cls) -> None:
        """
        Reset the singleton instance. FOR TESTING ONLY.

        This method is not thread-safe against concurrent production use.
        It exists solely to support test isolation.
        """
        with cls._lock:
            cls._instance = None
            logger.debug("QuarterlySchedulerEngine singleton reset (testing)")
