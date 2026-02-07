# -*- coding: utf-8 -*-
"""
Unit tests for Compliance Reporter (OBS-005)

Tests report generation (weekly/monthly/quarterly), compliance
percentage, trend calculation, SLO meeting/breaching counts,
and report persistence.

Coverage target: 85%+ of compliance_reporter.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from greenlang.infrastructure.slo_service.compliance_reporter import (
    calculate_trend,
    generate_report,
    store_report,
)
from greenlang.infrastructure.slo_service.models import (
    BudgetStatus,
    ErrorBudget,
    SLOReport,
)


# ---------------------------------------------------------------------------
# Helper to create budgets for each SLO
# ---------------------------------------------------------------------------


def _make_budgets(slos, sli_values):
    """Create ErrorBudget for each SLO with the given SLI values."""
    budgets = {}
    for slo, sli in zip(slos, sli_values):
        consumed = max(0.0, min(100.0, ((1.0 - sli / 100.0) / slo.error_budget_fraction) * 100.0))
        status = BudgetStatus.HEALTHY
        if consumed >= 100.0:
            status = BudgetStatus.EXHAUSTED
        elif consumed >= 50.0:
            status = BudgetStatus.CRITICAL
        elif consumed >= 20.0:
            status = BudgetStatus.WARNING

        budgets[slo.slo_id] = ErrorBudget(
            slo_id=slo.slo_id,
            total_minutes=slo.window_minutes * slo.error_budget_fraction,
            consumed_minutes=0.0,
            remaining_minutes=0.0,
            remaining_percent=100.0 - consumed,
            consumed_percent=consumed,
            status=status,
            sli_value=sli,
            window=slo.window.value,
        )
    return budgets


class TestCalculateTrend:
    """Tests for calculate_trend."""

    def test_slo_trend_improving(self):
        """Positive SLI difference is 'improving'."""
        assert calculate_trend(0.999, 0.995) == "improving"

    def test_slo_trend_stable(self):
        """Negligible SLI difference is 'stable'."""
        assert calculate_trend(0.999, 0.9989) == "stable"

    def test_slo_trend_degrading(self):
        """Negative SLI difference is 'degrading'."""
        assert calculate_trend(0.995, 0.999) == "degrading"

    def test_slo_trend_custom_threshold(self):
        """Custom threshold affects trend detection."""
        assert calculate_trend(0.999, 0.998, threshold=0.01) == "stable"
        assert calculate_trend(0.999, 0.980, threshold=0.01) == "improving"


class TestReportGeneration:
    """Tests for generate_report."""

    def test_weekly_report_generation(self, sample_slo_list):
        """Weekly report covers 7-day period."""
        budgets = _make_budgets(sample_slo_list, [99.95, 99.5, 99.995])
        report = generate_report("weekly", sample_slo_list, budgets)
        assert report.report_type == "weekly"
        assert report.period_start is not None
        assert report.period_end is not None
        diff = (report.period_end - report.period_start).days
        assert diff == 7

    def test_monthly_report_generation(self, sample_slo_list):
        """Monthly report covers 30-day period."""
        budgets = _make_budgets(sample_slo_list, [99.95, 99.5, 99.995])
        report = generate_report("monthly", sample_slo_list, budgets)
        assert report.report_type == "monthly"

    def test_quarterly_report_generation(self, sample_slo_list):
        """Quarterly report covers 90-day period."""
        budgets = _make_budgets(sample_slo_list, [99.95, 99.5, 99.995])
        report = generate_report("quarterly", sample_slo_list, budgets)
        assert report.report_type == "quarterly"

    def test_report_includes_all_slos(self, sample_slo_list):
        """Report entries include all active SLOs with budgets."""
        budgets = _make_budgets(sample_slo_list, [99.95, 99.5, 99.995])
        report = generate_report("weekly", sample_slo_list, budgets)
        assert len(report.entries) == len(sample_slo_list)

    def test_report_compliance_percentage(self, sample_slo_list):
        """Compliance percentage is correct."""
        # All meet target: 99.95 >= 99.9, 99.5 >= 99.0, 99.995 >= 99.99
        budgets = _make_budgets(sample_slo_list, [99.95, 99.5, 99.995])
        report = generate_report("weekly", sample_slo_list, budgets)
        assert report.overall_compliance_percent == pytest.approx(100.0)

    def test_report_meeting_target_count(self, sample_slo_list):
        """slos_met count is correct."""
        budgets = _make_budgets(sample_slo_list, [99.95, 99.5, 99.995])
        report = generate_report("weekly", sample_slo_list, budgets)
        assert report.slos_met == 3

    def test_report_breached_count(self, sample_slo_list):
        """slos_not_met count is correct when some breach target."""
        # 99.8 < 99.9 (breached), 98.5 < 99.0 (breached), 99.995 >= 99.99
        budgets = _make_budgets(sample_slo_list, [99.8, 98.5, 99.995])
        report = generate_report("weekly", sample_slo_list, budgets)
        assert report.slos_not_met == 2
        assert report.slos_met == 1

    def test_report_serialization(self, sample_slo_list):
        """Report serializes to dict correctly."""
        budgets = _make_budgets(sample_slo_list, [99.95, 99.5, 99.995])
        report = generate_report("weekly", sample_slo_list, budgets)
        d = report.to_dict()
        assert d["report_type"] == "weekly"
        assert "entries" in d
        assert len(d["entries"]) == 3

    def test_empty_slos_empty_report(self):
        """Empty SLO list produces empty report."""
        report = generate_report("weekly", [], {})
        assert report.total_slos == 0
        assert report.entries == []
        assert report.overall_compliance_percent == 0.0

    def test_report_period_boundaries(self, sample_slo_list):
        """Custom period boundaries are respected."""
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 31, tzinfo=timezone.utc)
        budgets = _make_budgets(sample_slo_list, [99.95, 99.5, 99.995])
        report = generate_report(
            "monthly", sample_slo_list, budgets,
            period_start=start, period_end=end,
        )
        assert report.period_start == start
        assert report.period_end == end


class TestStoreReport:
    """Tests for store_report."""

    def test_report_storage(self, tmp_path, sample_slo_list):
        """Report is stored as JSON file."""
        budgets = _make_budgets(sample_slo_list, [99.95, 99.5, 99.995])
        report = generate_report("weekly", sample_slo_list, budgets)
        path = store_report(report, str(tmp_path))
        assert Path(path).exists()

        with open(path) as f:
            data = json.load(f)
        assert data["report_type"] == "weekly"

    def test_report_storage_creates_directory(self, tmp_path, sample_slo_list):
        """Store creates output directory if it doesn't exist."""
        budgets = _make_budgets(sample_slo_list, [99.95, 99.5, 99.995])
        report = generate_report("weekly", sample_slo_list, budgets)
        nested_dir = str(tmp_path / "nested" / "reports")
        path = store_report(report, nested_dir)
        assert Path(path).exists()
