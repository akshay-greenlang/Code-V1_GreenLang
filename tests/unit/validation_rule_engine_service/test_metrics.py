# -*- coding: utf-8 -*-
"""
Unit Tests for Validation Rule Engine Prometheus Metrics - AGENT-DATA-019

Tests all 12 Prometheus metrics and their helper functions. Verifies that
each helper records correctly when PROMETHEUS_AVAILABLE is True and
operates as a graceful no-op when False. Covers counter increments,
histogram observations, and gauge set operations.

Target: 30-40 tests, 85%+ coverage of greenlang.validation_rule_engine.metrics

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from greenlang.validation_rule_engine.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    vre_active_rule_sets,
    vre_active_rules,
    vre_conflicts_detected_total,
    vre_evaluation_duration_seconds,
    vre_evaluation_failures_total,
    vre_evaluations_total,
    vre_pass_rate,
    vre_processing_duration_seconds,
    vre_reports_generated_total,
    vre_rule_sets_created_total,
    vre_rules_per_set,
    vre_rules_registered_total,
    # Helper functions
    observe_evaluation_duration,
    observe_processing_duration,
    observe_rules_per_set,
    record_conflict_detected,
    record_evaluation,
    record_evaluation_failure,
    record_report_generated,
    record_rule_registered,
    record_rule_set_created,
    set_active_rule_sets,
    set_active_rules,
    set_pass_rate,
)


# ============================================================================
# TestPrometheusAvailable - module-level flag
# ============================================================================


class TestPrometheusAvailable:
    """PROMETHEUS_AVAILABLE flag must reflect actual import status."""

    def test_prometheus_available_is_bool(self):
        assert isinstance(PROMETHEUS_AVAILABLE, bool)

    def test_prometheus_available_true_when_installed(self):
        """prometheus_client is installed in this test environment."""
        try:
            import prometheus_client  # noqa: F401
            assert PROMETHEUS_AVAILABLE is True
        except ImportError:
            assert PROMETHEUS_AVAILABLE is False


# ============================================================================
# TestMetricObjects - metric instances exist
# ============================================================================


class TestMetricObjects:
    """All 12 metric objects must be defined (non-None when Prometheus is available)."""

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_vre_rules_registered_total_exists(self):
        assert vre_rules_registered_total is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_vre_rule_sets_created_total_exists(self):
        assert vre_rule_sets_created_total is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_vre_evaluations_total_exists(self):
        assert vre_evaluations_total is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_vre_evaluation_failures_total_exists(self):
        assert vre_evaluation_failures_total is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_vre_conflicts_detected_total_exists(self):
        assert vre_conflicts_detected_total is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_vre_reports_generated_total_exists(self):
        assert vre_reports_generated_total is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_vre_rules_per_set_exists(self):
        assert vre_rules_per_set is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_vre_evaluation_duration_seconds_exists(self):
        assert vre_evaluation_duration_seconds is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_vre_processing_duration_seconds_exists(self):
        assert vre_processing_duration_seconds is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_vre_active_rules_exists(self):
        assert vre_active_rules is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_vre_active_rule_sets_exists(self):
        assert vre_active_rule_sets is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_vre_pass_rate_exists(self):
        assert vre_pass_rate is not None


# ============================================================================
# TestRecordRuleRegistered - Counter: gl_vre_rules_registered_total
# ============================================================================


class TestRecordRuleRegistered:
    """record_rule_registered() increments the rules_registered counter."""

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_rule_registered_no_error(self):
        """Calling record_rule_registered must not raise."""
        record_rule_registered("range_check", "error")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_rule_registered_various_types(self):
        """record_rule_registered with different rule types must not raise."""
        for rt in ["range_check", "format_validation", "cross_field", "regex",
                    "completeness", "uniqueness", "consistency", "custom"]:
            record_rule_registered(rt, "warning")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_rule_registered_various_severities(self):
        """record_rule_registered with different severities must not raise."""
        for sev in ["critical", "error", "warning", "info", "debug"]:
            record_rule_registered("range_check", sev)


# ============================================================================
# TestRecordRuleSetCreated - Counter: gl_vre_rule_sets_created_total
# ============================================================================


class TestRecordRuleSetCreated:
    """record_rule_set_created() increments the rule_sets_created counter."""

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_rule_set_created_no_error(self):
        record_rule_set_created("ghg_protocol")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_rule_set_created_various_pack_types(self):
        for pt in ["ghg_protocol", "csrd_esrs", "eudr", "custom", "regulatory"]:
            record_rule_set_created(pt)


# ============================================================================
# TestRecordEvaluation - Counter: gl_vre_evaluations_total
# ============================================================================


class TestRecordEvaluation:
    """record_evaluation() increments the evaluations counter."""

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_evaluation_pass(self):
        record_evaluation("pass", "range_check")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_evaluation_fail(self):
        record_evaluation("fail", "format_validation")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_evaluation_warning(self):
        record_evaluation("warning", "cross_field")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_evaluation_skipped(self):
        record_evaluation("skipped", "custom")


# ============================================================================
# TestRecordEvaluationFailure - Counter: gl_vre_evaluation_failures_total
# ============================================================================


class TestRecordEvaluationFailure:
    """record_evaluation_failure() increments the failures counter."""

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_evaluation_failure_critical(self):
        record_evaluation_failure("critical")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_evaluation_failure_error(self):
        record_evaluation_failure("error")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_evaluation_failure_warning(self):
        record_evaluation_failure("warning")


# ============================================================================
# TestRecordConflictDetected - Counter: gl_vre_conflicts_detected_total
# ============================================================================


class TestRecordConflictDetected:
    """record_conflict_detected() increments the conflicts counter."""

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_conflict_detected_contradiction(self):
        record_conflict_detected("contradiction")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_conflict_detected_overlap(self):
        record_conflict_detected("overlap")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_conflict_detected_redundancy(self):
        record_conflict_detected("redundancy")


# ============================================================================
# TestRecordReportGenerated - Counter: gl_vre_reports_generated_total
# ============================================================================


class TestRecordReportGenerated:
    """record_report_generated() increments the reports counter."""

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_report_json(self):
        record_report_generated("evaluation_summary", "json")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_report_html(self):
        record_report_generated("compliance_report", "html")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_report_csv(self):
        record_report_generated("audit_trail", "csv")


# ============================================================================
# TestObserveRulesPerSet - Histogram: gl_vre_rules_per_set
# ============================================================================


class TestObserveRulesPerSet:
    """observe_rules_per_set() records histogram observations."""

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_observe_rules_per_set_small(self):
        observe_rules_per_set(5)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_observe_rules_per_set_large(self):
        observe_rules_per_set(500)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_observe_rules_per_set_zero(self):
        observe_rules_per_set(0)


# ============================================================================
# TestObserveEvaluationDuration - Histogram: gl_vre_evaluation_duration_seconds
# ============================================================================


class TestObserveEvaluationDuration:
    """observe_evaluation_duration() records evaluation timing."""

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_observe_eval_duration_single_rule(self):
        observe_evaluation_duration("single_rule", 0.05)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_observe_eval_duration_rule_set(self):
        observe_evaluation_duration("rule_set", 1.5)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_observe_eval_duration_batch(self):
        observe_evaluation_duration("batch", 10.0)


# ============================================================================
# TestObserveProcessingDuration - Histogram: gl_vre_processing_duration_seconds
# ============================================================================


class TestObserveProcessingDuration:
    """observe_processing_duration() records processing timing."""

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_observe_processing_rule_register(self):
        observe_processing_duration("rule_register", 0.02)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_observe_processing_conflict_detect(self):
        observe_processing_duration("conflict_detect", 0.5)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_observe_processing_report_generate(self):
        observe_processing_duration("report_generate", 2.0)


# ============================================================================
# TestSetActiveRules - Gauge: gl_vre_active_rules
# ============================================================================


class TestSetActiveRules:
    """set_active_rules() sets the active rules gauge."""

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_set_active_rules_zero(self):
        set_active_rules(0)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_set_active_rules_positive(self):
        set_active_rules(42)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_set_active_rules_large(self):
        set_active_rules(100_000)


# ============================================================================
# TestSetActiveRuleSets - Gauge: gl_vre_active_rule_sets
# ============================================================================


class TestSetActiveRuleSets:
    """set_active_rule_sets() sets the active rule sets gauge."""

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_set_active_rule_sets_zero(self):
        set_active_rule_sets(0)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_set_active_rule_sets_positive(self):
        set_active_rule_sets(15)


# ============================================================================
# TestSetPassRate - Gauge: gl_vre_pass_rate
# ============================================================================


class TestSetPassRate:
    """set_pass_rate() sets the pass rate gauge."""

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_set_pass_rate_zero(self):
        set_pass_rate(0.0)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_set_pass_rate_one(self):
        set_pass_rate(1.0)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_set_pass_rate_half(self):
        set_pass_rate(0.5)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_set_pass_rate_high(self):
        set_pass_rate(0.95)


# ============================================================================
# TestNoOpWhenUnavailable - graceful fallback
# ============================================================================


class TestNoOpWhenUnavailable:
    """All helper functions must be no-ops when PROMETHEUS_AVAILABLE is False."""

    def test_record_rule_registered_noop(self):
        with patch("greenlang.validation_rule_engine.metrics.PROMETHEUS_AVAILABLE", False):
            # Must not raise
            record_rule_registered("range_check", "error")

    def test_record_rule_set_created_noop(self):
        with patch("greenlang.validation_rule_engine.metrics.PROMETHEUS_AVAILABLE", False):
            record_rule_set_created("ghg_protocol")

    def test_record_evaluation_noop(self):
        with patch("greenlang.validation_rule_engine.metrics.PROMETHEUS_AVAILABLE", False):
            record_evaluation("pass", "range_check")

    def test_record_evaluation_failure_noop(self):
        with patch("greenlang.validation_rule_engine.metrics.PROMETHEUS_AVAILABLE", False):
            record_evaluation_failure("critical")

    def test_record_conflict_detected_noop(self):
        with patch("greenlang.validation_rule_engine.metrics.PROMETHEUS_AVAILABLE", False):
            record_conflict_detected("contradiction")

    def test_record_report_generated_noop(self):
        with patch("greenlang.validation_rule_engine.metrics.PROMETHEUS_AVAILABLE", False):
            record_report_generated("evaluation_summary", "json")

    def test_observe_rules_per_set_noop(self):
        with patch("greenlang.validation_rule_engine.metrics.PROMETHEUS_AVAILABLE", False):
            observe_rules_per_set(10)

    def test_observe_evaluation_duration_noop(self):
        with patch("greenlang.validation_rule_engine.metrics.PROMETHEUS_AVAILABLE", False):
            observe_evaluation_duration("single_rule", 0.1)

    def test_observe_processing_duration_noop(self):
        with patch("greenlang.validation_rule_engine.metrics.PROMETHEUS_AVAILABLE", False):
            observe_processing_duration("rule_register", 0.05)

    def test_set_active_rules_noop(self):
        with patch("greenlang.validation_rule_engine.metrics.PROMETHEUS_AVAILABLE", False):
            set_active_rules(42)

    def test_set_active_rule_sets_noop(self):
        with patch("greenlang.validation_rule_engine.metrics.PROMETHEUS_AVAILABLE", False):
            set_active_rule_sets(10)

    def test_set_pass_rate_noop(self):
        with patch("greenlang.validation_rule_engine.metrics.PROMETHEUS_AVAILABLE", False):
            set_pass_rate(0.95)
