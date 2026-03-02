# -*- coding: utf-8 -*-
"""
Unit tests for GL-CBAM-APP v1.1 Quarterly Scheduler

Tests quarterly scheduling:
- get_current_quarter (all 4 quarters)
- get_quarter_for_date (boundary dates)
- is_transitional_period (before/after Dec 2025)
- is_definitive_period (Jan 2026+)
- get_submission_deadline (30 days after quarter end)
- get_amendment_deadline (60 days)
- should_generate_report (15 days trigger)
- get_reporting_calendar (full year)
- Edge cases: leap years, year boundaries

Target: 50+ tests
"""

import pytest
from datetime import date, datetime, timedelta


# ---------------------------------------------------------------------------
# Inline quarterly scheduler for self-contained tests
# ---------------------------------------------------------------------------

class QuarterlyScheduler:
    """CBAM quarterly scheduling engine."""

    TRANSITIONAL_END = date(2025, 12, 31)
    DEFINITIVE_START = date(2026, 1, 1)
    SUBMISSION_DEADLINE_DAYS = 30
    AMENDMENT_DEADLINE_DAYS = 60
    REPORT_TRIGGER_DAYS = 15

    QUARTER_RANGES = {
        1: {"start_month": 1, "start_day": 1, "end_month": 3, "end_day": 31},
        2: {"start_month": 4, "start_day": 1, "end_month": 6, "end_day": 30},
        3: {"start_month": 7, "start_day": 1, "end_month": 9, "end_day": 30},
        4: {"start_month": 10, "start_day": 1, "end_month": 12, "end_day": 31},
    }

    def get_current_quarter(self, ref_date: date = None) -> str:
        ref_date = ref_date or date.today()
        q = self._quarter_number(ref_date)
        return f"{ref_date.year}Q{q}"

    def _quarter_number(self, d: date) -> int:
        return (d.month - 1) // 3 + 1

    def get_quarter_for_date(self, d: date) -> str:
        q = self._quarter_number(d)
        return f"{d.year}Q{q}"

    def get_quarter_start(self, quarter: str) -> date:
        year, q = self._parse_quarter(quarter)
        r = self.QUARTER_RANGES[q]
        return date(year, r["start_month"], r["start_day"])

    def get_quarter_end(self, quarter: str) -> date:
        year, q = self._parse_quarter(quarter)
        r = self.QUARTER_RANGES[q]
        return date(year, r["end_month"], r["end_day"])

    def _parse_quarter(self, quarter: str):
        year = int(quarter[:4])
        q = int(quarter[5])
        return year, q

    def is_transitional_period(self, d: date) -> bool:
        return d <= self.TRANSITIONAL_END

    def is_definitive_period(self, d: date) -> bool:
        return d >= self.DEFINITIVE_START

    def get_submission_deadline(self, quarter: str) -> date:
        end = self.get_quarter_end(quarter)
        return end + timedelta(days=self.SUBMISSION_DEADLINE_DAYS)

    def get_amendment_deadline(self, quarter: str) -> date:
        end = self.get_quarter_end(quarter)
        return end + timedelta(days=self.AMENDMENT_DEADLINE_DAYS)

    def should_generate_report(self, quarter: str, ref_date: date = None) -> bool:
        ref_date = ref_date or date.today()
        end = self.get_quarter_end(quarter)
        days_after = (ref_date - end).days
        return days_after >= self.REPORT_TRIGGER_DAYS

    def is_within_submission_window(self, quarter: str,
                                    ref_date: date = None) -> bool:
        ref_date = ref_date or date.today()
        end = self.get_quarter_end(quarter)
        deadline = self.get_submission_deadline(quarter)
        return end < ref_date <= deadline

    def is_within_amendment_window(self, quarter: str,
                                   ref_date: date = None) -> bool:
        ref_date = ref_date or date.today()
        end = self.get_quarter_end(quarter)
        deadline = self.get_amendment_deadline(quarter)
        return end < ref_date <= deadline

    def get_reporting_calendar(self, year: int) -> list:
        calendar = []
        for q in range(1, 5):
            quarter = f"{year}Q{q}"
            calendar.append({
                "quarter": quarter,
                "period_start": self.get_quarter_start(quarter).isoformat(),
                "period_end": self.get_quarter_end(quarter).isoformat(),
                "submission_deadline": self.get_submission_deadline(quarter).isoformat(),
                "amendment_deadline": self.get_amendment_deadline(quarter).isoformat(),
                "is_transitional": self.is_transitional_period(
                    self.get_quarter_end(quarter)
                ),
                "is_definitive": self.is_definitive_period(
                    self.get_quarter_start(quarter)
                ),
            })
        return calendar

    def days_until_deadline(self, quarter: str,
                            ref_date: date = None) -> int:
        ref_date = ref_date or date.today()
        deadline = self.get_submission_deadline(quarter)
        return (deadline - ref_date).days

    def get_next_quarter(self, quarter: str) -> str:
        year, q = self._parse_quarter(quarter)
        if q == 4:
            return f"{year + 1}Q1"
        return f"{year}Q{q + 1}"

    def get_previous_quarter(self, quarter: str) -> str:
        year, q = self._parse_quarter(quarter)
        if q == 1:
            return f"{year - 1}Q4"
        return f"{year}Q{q - 1}"


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def scheduler():
    return QuarterlyScheduler()


# ===========================================================================
# TEST CLASS -- get_current_quarter
# ===========================================================================

class TestGetCurrentQuarter:
    """Tests for get_current_quarter."""

    @pytest.mark.parametrize("month,expected_q", [
        (1, 1), (2, 1), (3, 1),
        (4, 2), (5, 2), (6, 2),
        (7, 3), (8, 3), (9, 3),
        (10, 4), (11, 4), (12, 4),
    ])
    def test_all_months(self, scheduler, month, expected_q):
        d = date(2026, month, 15)
        assert scheduler.get_current_quarter(d) == f"2026Q{expected_q}"

    def test_january_first(self, scheduler):
        assert scheduler.get_current_quarter(date(2026, 1, 1)) == "2026Q1"

    def test_december_31(self, scheduler):
        assert scheduler.get_current_quarter(date(2026, 12, 31)) == "2026Q4"


# ===========================================================================
# TEST CLASS -- get_quarter_for_date
# ===========================================================================

class TestGetQuarterForDate:
    """Tests for get_quarter_for_date with boundary dates."""

    @pytest.mark.parametrize("d,expected", [
        (date(2026, 1, 1), "2026Q1"),
        (date(2026, 3, 31), "2026Q1"),
        (date(2026, 4, 1), "2026Q2"),
        (date(2026, 6, 30), "2026Q2"),
        (date(2026, 7, 1), "2026Q3"),
        (date(2026, 9, 30), "2026Q3"),
        (date(2026, 10, 1), "2026Q4"),
        (date(2026, 12, 31), "2026Q4"),
    ])
    def test_boundary_dates(self, scheduler, d, expected):
        assert scheduler.get_quarter_for_date(d) == expected

    def test_leap_year_feb_29(self, scheduler):
        assert scheduler.get_quarter_for_date(date(2028, 2, 29)) == "2028Q1"


# ===========================================================================
# TEST CLASS -- is_transitional_period
# ===========================================================================

class TestIsTransitionalPeriod:
    """Tests for is_transitional_period."""

    @pytest.mark.parametrize("d,expected", [
        (date(2023, 10, 1), True),
        (date(2024, 6, 15), True),
        (date(2025, 12, 31), True),
        (date(2026, 1, 1), False),
        (date(2027, 6, 1), False),
    ])
    def test_transitional_boundary(self, scheduler, d, expected):
        assert scheduler.is_transitional_period(d) is expected


# ===========================================================================
# TEST CLASS -- is_definitive_period
# ===========================================================================

class TestIsDefinitivePeriod:
    """Tests for is_definitive_period."""

    @pytest.mark.parametrize("d,expected", [
        (date(2025, 12, 31), False),
        (date(2026, 1, 1), True),
        (date(2026, 6, 15), True),
        (date(2030, 1, 1), True),
    ])
    def test_definitive_boundary(self, scheduler, d, expected):
        assert scheduler.is_definitive_period(d) is expected


# ===========================================================================
# TEST CLASS -- get_submission_deadline
# ===========================================================================

class TestGetSubmissionDeadline:
    """Tests for get_submission_deadline."""

    @pytest.mark.parametrize("quarter,expected_end,expected_deadline", [
        ("2026Q1", date(2026, 3, 31), date(2026, 4, 30)),
        ("2026Q2", date(2026, 6, 30), date(2026, 7, 30)),
        ("2026Q3", date(2026, 9, 30), date(2026, 10, 30)),
        ("2026Q4", date(2026, 12, 31), date(2027, 1, 30)),
    ])
    def test_all_quarters(self, scheduler, quarter, expected_end,
                          expected_deadline):
        assert scheduler.get_quarter_end(quarter) == expected_end
        assert scheduler.get_submission_deadline(quarter) == expected_deadline

    def test_cross_year_q4(self, scheduler):
        deadline = scheduler.get_submission_deadline("2025Q4")
        assert deadline.year == 2026
        assert deadline.month == 1


# ===========================================================================
# TEST CLASS -- get_amendment_deadline
# ===========================================================================

class TestGetAmendmentDeadline:
    """Tests for get_amendment_deadline (60 days)."""

    def test_q1_amendment_deadline(self, scheduler):
        deadline = scheduler.get_amendment_deadline("2026Q1")
        expected = date(2026, 3, 31) + timedelta(days=60)
        assert deadline == expected

    def test_amendment_after_submission(self, scheduler):
        sub_deadline = scheduler.get_submission_deadline("2026Q1")
        amend_deadline = scheduler.get_amendment_deadline("2026Q1")
        assert amend_deadline > sub_deadline

    def test_amendment_window_duration(self, scheduler):
        q_end = scheduler.get_quarter_end("2026Q2")
        amend = scheduler.get_amendment_deadline("2026Q2")
        assert (amend - q_end).days == 60


# ===========================================================================
# TEST CLASS -- should_generate_report
# ===========================================================================

class TestShouldGenerateReport:
    """Tests for should_generate_report (15-day trigger)."""

    def test_before_trigger(self, scheduler):
        q_end = scheduler.get_quarter_end("2026Q1")
        ref = q_end + timedelta(days=14)
        assert scheduler.should_generate_report("2026Q1", ref) is False

    def test_at_trigger(self, scheduler):
        q_end = scheduler.get_quarter_end("2026Q1")
        ref = q_end + timedelta(days=15)
        assert scheduler.should_generate_report("2026Q1", ref) is True

    def test_after_trigger(self, scheduler):
        q_end = scheduler.get_quarter_end("2026Q1")
        ref = q_end + timedelta(days=30)
        assert scheduler.should_generate_report("2026Q1", ref) is True

    def test_during_quarter(self, scheduler):
        ref = date(2026, 2, 15)  # still in Q1
        assert scheduler.should_generate_report("2026Q1", ref) is False


# ===========================================================================
# TEST CLASS -- get_reporting_calendar
# ===========================================================================

class TestGetReportingCalendar:
    """Tests for get_reporting_calendar."""

    def test_calendar_has_four_entries(self, scheduler):
        cal = scheduler.get_reporting_calendar(2026)
        assert len(cal) == 4

    def test_calendar_quarters_sequential(self, scheduler):
        cal = scheduler.get_reporting_calendar(2026)
        quarters = [e["quarter"] for e in cal]
        assert quarters == ["2026Q1", "2026Q2", "2026Q3", "2026Q4"]

    def test_2025_transitional(self, scheduler):
        cal = scheduler.get_reporting_calendar(2025)
        assert all(e["is_transitional"] for e in cal)

    def test_2026_definitive(self, scheduler):
        cal = scheduler.get_reporting_calendar(2026)
        assert all(e["is_definitive"] for e in cal)

    def test_calendar_contains_all_fields(self, scheduler):
        cal = scheduler.get_reporting_calendar(2026)
        required = {"quarter", "period_start", "period_end",
                     "submission_deadline", "amendment_deadline",
                     "is_transitional", "is_definitive"}
        for entry in cal:
            assert required.issubset(entry.keys())


# ===========================================================================
# TEST CLASS -- Edge cases
# ===========================================================================

class TestEdgeCases:
    """Tests for edge cases: leap years, year boundaries, navigation."""

    def test_leap_year_q1_end(self, scheduler):
        end = scheduler.get_quarter_end("2028Q1")
        assert end == date(2028, 3, 31)

    def test_non_leap_year_q1_end(self, scheduler):
        end = scheduler.get_quarter_end("2027Q1")
        assert end == date(2027, 3, 31)

    def test_year_boundary_next_quarter(self, scheduler):
        assert scheduler.get_next_quarter("2026Q4") == "2027Q1"

    def test_year_boundary_previous_quarter(self, scheduler):
        assert scheduler.get_previous_quarter("2027Q1") == "2026Q4"

    def test_next_quarter_sequence(self, scheduler):
        assert scheduler.get_next_quarter("2026Q1") == "2026Q2"
        assert scheduler.get_next_quarter("2026Q2") == "2026Q3"
        assert scheduler.get_next_quarter("2026Q3") == "2026Q4"
        assert scheduler.get_next_quarter("2026Q4") == "2027Q1"

    def test_previous_quarter_sequence(self, scheduler):
        assert scheduler.get_previous_quarter("2026Q4") == "2026Q3"
        assert scheduler.get_previous_quarter("2026Q3") == "2026Q2"
        assert scheduler.get_previous_quarter("2026Q2") == "2026Q1"
        assert scheduler.get_previous_quarter("2026Q1") == "2025Q4"

    def test_days_until_deadline_positive(self, scheduler):
        ref = date(2026, 4, 1)
        days = scheduler.days_until_deadline("2026Q1", ref)
        assert days > 0

    def test_days_until_deadline_negative(self, scheduler):
        ref = date(2026, 6, 1)
        days = scheduler.days_until_deadline("2026Q1", ref)
        assert days < 0

    def test_within_submission_window(self, scheduler):
        q_end = scheduler.get_quarter_end("2026Q1")
        ref = q_end + timedelta(days=10)
        assert scheduler.is_within_submission_window("2026Q1", ref) is True

    def test_outside_submission_window(self, scheduler):
        q_end = scheduler.get_quarter_end("2026Q1")
        ref = q_end + timedelta(days=31)
        assert scheduler.is_within_submission_window("2026Q1", ref) is False

    def test_within_amendment_window(self, scheduler):
        q_end = scheduler.get_quarter_end("2026Q1")
        ref = q_end + timedelta(days=45)
        assert scheduler.is_within_amendment_window("2026Q1", ref) is True

    def test_outside_amendment_window(self, scheduler):
        q_end = scheduler.get_quarter_end("2026Q1")
        ref = q_end + timedelta(days=61)
        assert scheduler.is_within_amendment_window("2026Q1", ref) is False
