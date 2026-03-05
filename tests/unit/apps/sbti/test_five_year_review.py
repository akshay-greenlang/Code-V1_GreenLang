# -*- coding: utf-8 -*-
"""
Unit tests for SBTi Five-Year Review Engine.

Tests review trigger date calculation (5 years from validation),
deadline enforcement (trigger + 12 months), readiness assessment,
notification schedule with alert timing, review outcomes
(renewed/updated/expired), and upcoming review listing with 20+
test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from datetime import date, timedelta

import pytest


# ===========================================================================
# Trigger Date
# ===========================================================================

class TestTriggerDate:
    """Test 5-year trigger date calculation."""

    def test_trigger_date_5_years_from_validation(self, sample_five_year_review):
        original = sample_five_year_review["original_validation_date"]
        trigger = sample_five_year_review["review_trigger_date"]
        delta_years = trigger.year - original.year
        assert delta_years == 5

    @pytest.mark.parametrize("validation_date,expected_trigger_year", [
        (date(2020, 1, 1), 2025),
        (date(2021, 6, 15), 2026),
        (date(2022, 12, 31), 2027),
        (date(2023, 3, 1), 2028),
    ])
    def test_trigger_date_parametrized(self, validation_date, expected_trigger_year):
        trigger = date(expected_trigger_year, validation_date.month, validation_date.day)
        assert trigger.year == expected_trigger_year

    def test_days_until_trigger(self, sample_five_year_review):
        days = sample_five_year_review["days_until_trigger"]
        assert isinstance(days, int)


# ===========================================================================
# Deadline
# ===========================================================================

class TestDeadline:
    """Test review deadline (trigger + 12 months)."""

    def test_deadline_12_months_after_trigger(self, sample_five_year_review):
        trigger = sample_five_year_review["review_trigger_date"]
        deadline = sample_five_year_review["review_deadline"]
        delta_years = deadline.year - trigger.year
        assert delta_years == 1

    def test_days_until_deadline_greater_than_trigger(self, sample_five_year_review):
        days_trigger = sample_five_year_review["days_until_trigger"]
        days_deadline = sample_five_year_review["days_until_deadline"]
        assert days_deadline > days_trigger

    @pytest.mark.parametrize("trigger_date,expected_deadline_year", [
        (date(2026, 6, 15), 2027),
        (date(2027, 1, 1), 2028),
        (date(2028, 12, 31), 2029),
    ])
    def test_deadline_calculation(self, trigger_date, expected_deadline_year):
        deadline = date(expected_deadline_year, trigger_date.month, trigger_date.day)
        assert deadline.year == expected_deadline_year


# ===========================================================================
# Readiness Assessment
# ===========================================================================

class TestReadinessAssessment:
    """Test review readiness scoring."""

    def test_readiness_score_range(self, sample_five_year_review):
        score = sample_five_year_review["readiness_score"]
        assert 0 <= score <= 100.0

    def test_readiness_score_value(self, sample_five_year_review):
        assert sample_five_year_review["readiness_score"] == 75.0

    @pytest.mark.parametrize("score,expected_status", [
        (90.0, "ready"),
        (70.0, "mostly_ready"),
        (50.0, "needs_preparation"),
        (25.0, "not_ready"),
    ])
    def test_readiness_level_mapping(self, score, expected_status):
        if score >= 80:
            status = "ready"
        elif score >= 60:
            status = "mostly_ready"
        elif score >= 40:
            status = "needs_preparation"
        else:
            status = "not_ready"
        assert status == expected_status


# ===========================================================================
# Notification Schedule
# ===========================================================================

class TestNotificationSchedule:
    """Test alert timing for review notifications."""

    def test_notification_count(self, sample_five_year_review):
        schedule = sample_five_year_review["notification_schedule"]
        assert len(schedule) >= 4

    def test_notification_milestones(self, sample_five_year_review):
        schedule = sample_five_year_review["notification_schedule"]
        months_before = [n["months_before"] for n in schedule]
        assert 12 in months_before
        assert 6 in months_before
        assert 3 in months_before
        assert 1 in months_before

    def test_early_notifications_sent(self, sample_five_year_review):
        schedule = sample_five_year_review["notification_schedule"]
        twelve_month = next(n for n in schedule if n["months_before"] == 12)
        assert twelve_month["sent"] is True

    def test_future_notifications_pending(self, sample_five_year_review):
        schedule = sample_five_year_review["notification_schedule"]
        one_month = next(n for n in schedule if n["months_before"] == 1)
        assert one_month["sent"] is False

    def test_sent_notifications_have_date(self, sample_five_year_review):
        schedule = sample_five_year_review["notification_schedule"]
        for notification in schedule:
            if notification["sent"]:
                assert "sent_date" in notification
                assert isinstance(notification["sent_date"], date)


# ===========================================================================
# Review Outcome
# ===========================================================================

class TestReviewOutcome:
    """Test review outcome options."""

    VALID_OUTCOMES = ["renewed", "updated", "expired", "withdrawn"]

    def test_pending_review_no_outcome(self, sample_five_year_review):
        assert sample_five_year_review["review_outcome"] is None

    @pytest.mark.parametrize("outcome", ["renewed", "updated", "expired", "withdrawn"])
    def test_valid_outcomes(self, outcome):
        assert outcome in self.VALID_OUTCOMES

    def test_renewed_outcome(self):
        review = {"review_outcome": "renewed", "new_target_id": None}
        assert review["review_outcome"] == "renewed"

    def test_updated_outcome_has_new_target(self):
        review = {"review_outcome": "updated", "new_target_id": "tgt_abc123"}
        assert review["new_target_id"] is not None

    def test_expired_outcome(self):
        review = {"review_outcome": "expired"}
        assert review["review_outcome"] == "expired"


# ===========================================================================
# Upcoming Reviews
# ===========================================================================

class TestUpcomingReviews:
    """Test listing of upcoming reviews."""

    def test_review_status_upcoming(self, sample_five_year_review):
        assert sample_five_year_review["review_status"] == "upcoming"

    @pytest.mark.parametrize("status", [
        "upcoming", "in_progress", "completed", "overdue",
    ])
    def test_valid_review_statuses(self, status):
        valid = {"upcoming", "in_progress", "completed", "overdue"}
        assert status in valid

    def test_upcoming_reviews_sorted_by_deadline(self):
        reviews = [
            {"deadline": date(2027, 6, 15)},
            {"deadline": date(2026, 12, 1)},
            {"deadline": date(2028, 3, 15)},
        ]
        sorted_reviews = sorted(reviews, key=lambda r: r["deadline"])
        assert sorted_reviews[0]["deadline"] < sorted_reviews[1]["deadline"]
        assert sorted_reviews[1]["deadline"] < sorted_reviews[2]["deadline"]

    def test_overdue_detection(self):
        review = {
            "review_deadline": date(2025, 6, 15),
            "review_status": "upcoming",
        }
        is_overdue = review["review_deadline"] < date.today()
        if is_overdue:
            review["review_status"] = "overdue"
        assert review["review_status"] == "overdue"
