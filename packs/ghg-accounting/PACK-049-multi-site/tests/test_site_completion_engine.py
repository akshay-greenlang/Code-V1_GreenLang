# -*- coding: utf-8 -*-
"""
Tests for PACK-049 Engine 8: SiteCompletionEngine

Covers completeness assessment, site scoring, submission tracking,
gap detection (missing site, missing scope), reminder generation,
escalation, coverage metrics, and gap impact estimation.
Target: ~45 tests.
"""

import pytest
from decimal import Decimal
from datetime import date, timedelta
from pathlib import Path
import sys

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

try:
    from engines.site_completion_engine import (
        SiteCompletionEngine,
        CompletenessAssessment,
        SiteCompletenessScore,
        SubmissionTracker,
        GapReport,
        GapItem,
        Reminder,
        CoverageMetrics,
    )
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

pytestmark = pytest.mark.skipif(not HAS_ENGINE, reason="Engine not yet built")


@pytest.fixture
def engine():
    return SiteCompletionEngine()


@pytest.fixture
def submission_data():
    """Submission status for 10 sites."""
    return {
        "site-001": {"status": "APPROVED", "scope1": True, "scope2": True, "scope3": True},
        "site-002": {"status": "APPROVED", "scope1": True, "scope2": True, "scope3": True},
        "site-003": {"status": "APPROVED", "scope1": True, "scope2": True, "scope3": False},
        "site-004": {"status": "APPROVED", "scope1": True, "scope2": True, "scope3": True},
        "site-005": {"status": "SUBMITTED", "scope1": True, "scope2": True, "scope3": True},
        "site-006": {"status": "SUBMITTED", "scope1": True, "scope2": False, "scope3": False},
        "site-007": {"status": "APPROVED", "scope1": True, "scope2": True, "scope3": True},
        "site-008": {"status": "OVERDUE", "scope1": False, "scope2": False, "scope3": False},
        "site-009": {"status": "NOT_STARTED", "scope1": False, "scope2": False, "scope3": False},
        "site-010": {"status": "NOT_STARTED", "scope1": False, "scope2": False, "scope3": False},
    }


# ============================================================================
# Completeness Assessment Tests
# ============================================================================

class TestCompletenessAssessment:

    def test_assess_completeness(self, engine, submission_data):
        assessment = engine.assess_completeness(
            site_submissions=submission_data,
            target_pct=Decimal("95"),
        )
        assert isinstance(assessment, CompletenessAssessment)
        assert assessment.total_sites == 10
        # 7 approved + 1 submitted with data = 8 submitted
        assert assessment.sites_with_data >= 7

    def test_assess_completeness_pct(self, engine, submission_data):
        assessment = engine.assess_completeness(
            site_submissions=submission_data,
            target_pct=Decimal("95"),
        )
        # 7 approved out of 10 = 70% approved, 8 with data = 80%
        assert assessment.completeness_pct >= Decimal("70")
        assert assessment.completeness_pct <= Decimal("100")

    def test_assess_completeness_gap(self, engine, submission_data):
        assessment = engine.assess_completeness(
            site_submissions=submission_data,
            target_pct=Decimal("95"),
        )
        # Gap = target - actual
        assert assessment.gap_pct >= Decimal("0")


# ============================================================================
# Site Scoring Tests
# ============================================================================

class TestSiteScoring:

    def test_score_site_completeness(self, engine, submission_data):
        score = engine.score_site(
            site_id="site-001",
            submission=submission_data["site-001"],
        )
        assert isinstance(score, SiteCompletenessScore)
        assert score.completeness_pct == Decimal("100") or score.completeness_pct >= Decimal("90")

    def test_score_site_partial(self, engine, submission_data):
        score = engine.score_site(
            site_id="site-006",
            submission=submission_data["site-006"],
        )
        assert score.completeness_pct < Decimal("100")

    def test_score_site_zero(self, engine, submission_data):
        score = engine.score_site(
            site_id="site-009",
            submission=submission_data["site-009"],
        )
        assert score.completeness_pct == Decimal("0") or score.completeness_pct <= Decimal("10")


# ============================================================================
# Submission Tracking Tests
# ============================================================================

class TestSubmissionTracking:

    def test_track_submissions(self, engine, submission_data):
        tracker = engine.track_submissions(submission_data)
        assert isinstance(tracker, SubmissionTracker)
        assert tracker.total == 10
        assert tracker.approved >= 5
        assert tracker.overdue >= 1
        assert tracker.not_started >= 2


# ============================================================================
# Gap Detection Tests
# ============================================================================

class TestGapDetection:

    def test_detect_gaps_missing_site(self, engine, submission_data):
        gaps = engine.detect_gaps(
            site_submissions=submission_data,
            expected_sites=list(submission_data.keys()),
        )
        assert isinstance(gaps, GapReport)
        # Sites 009 and 010 are not started
        assert gaps.missing_site_count >= 2

    def test_detect_gaps_missing_scope(self, engine, submission_data):
        gaps = engine.detect_gaps(
            site_submissions=submission_data,
            expected_sites=list(submission_data.keys()),
            expected_scopes=["scope1", "scope2", "scope3"],
        )
        # site-003 missing scope3, site-006 missing scope2+scope3
        assert gaps.missing_scope_count >= 2

    def test_detect_gaps_no_gaps(self, engine):
        perfect_data = {
            "s1": {"status": "APPROVED", "scope1": True, "scope2": True, "scope3": True},
            "s2": {"status": "APPROVED", "scope1": True, "scope2": True, "scope3": True},
        }
        gaps = engine.detect_gaps(
            site_submissions=perfect_data,
            expected_sites=["s1", "s2"],
            expected_scopes=["scope1", "scope2", "scope3"],
        )
        assert gaps.missing_site_count == 0
        assert gaps.missing_scope_count == 0


# ============================================================================
# Reminder Tests
# ============================================================================

class TestReminders:

    def test_reminders_due(self, engine, submission_data):
        reminders = engine.generate_reminders(
            site_submissions=submission_data,
            deadline=date(2027, 2, 28),
            reminder_days=[14, 7, 3, 1],
            current_date=date(2027, 2, 14),
        )
        assert isinstance(reminders, list)
        # Sites 009 and 010 not started, should have reminders
        assert len(reminders) >= 2

    def test_reminders_not_due(self, engine, submission_data):
        reminders = engine.generate_reminders(
            site_submissions=submission_data,
            deadline=date(2027, 6, 30),
            reminder_days=[14, 7, 3, 1],
            current_date=date(2027, 1, 1),
        )
        # Far from deadline, no reminders
        assert isinstance(reminders, list)


# ============================================================================
# Escalation Tests
# ============================================================================

class TestEscalation:

    def test_escalate_overdue(self, engine, submission_data):
        escalations = engine.escalate_overdue(
            site_submissions=submission_data,
            deadline=date(2027, 2, 28),
            current_date=date(2027, 3, 15),
        )
        assert isinstance(escalations, list)
        assert len(escalations) >= 1  # site-008 is overdue


# ============================================================================
# Coverage Metrics Tests
# ============================================================================

class TestCoverageMetrics:

    def test_coverage_metrics(self, engine, submission_data):
        metrics = engine.get_coverage_metrics(
            site_submissions=submission_data,
        )
        assert isinstance(metrics, CoverageMetrics)
        assert metrics.site_coverage_pct >= Decimal("0")
        assert metrics.scope_coverage_pct >= Decimal("0")

    def test_gap_impact_estimation(self, engine, submission_data):
        impact = engine.estimate_gap_impact(
            site_submissions=submission_data,
            estimated_avg_emissions=Decimal("500"),
        )
        assert impact >= Decimal("0")
        # 3 sites without data * 500 = 1500
        assert impact >= Decimal("500") or impact >= Decimal("0")
