# -*- coding: utf-8 -*-
"""
Tests for Factors SLO/SLI definitions and error budget helpers.

Covers:
  - SLI definition evaluation logic
  - SLO definition properties
  - check_sli and check_all_slis
  - Error budget calculation
  - Budget health classification
  - Compliance summary generation
"""

from __future__ import annotations

import pytest

from greenlang.factors.observability.sla import (
    FACTORS_SLO,
    FACTORS_SLOS,
    SLI_AVAILABILITY,
    SLI_ERROR_RATE,
    SLI_MATCH_CONFIDENCE_MEDIAN,
    SLI_REQUEST_LATENCY_P99,
    SLI_SEARCH_LATENCY_P95,
    BudgetHealth,
    ComplianceStatus,
    ErrorBudgetResult,
    SLICategory,
    SLICheckResult,
    SLIDefinition,
    SLODefinition,
    _classify_budget_health,
    calculate_error_budget,
    check_all_slis,
    check_sli,
    compliance_summary,
)


# ---------------------------------------------------------------------------
# SLI Definition
# ---------------------------------------------------------------------------


class TestSLIDefinition:
    """Tests for SLIDefinition.evaluate()."""

    def test_lt_comparison_met(self):
        sli = SLIDefinition(
            name="test",
            sli_id="test.lt",
            category=SLICategory.LATENCY,
            description="test",
            target=0.5,
            unit="seconds",
            comparison="lt",
            promql="",
        )
        assert sli.evaluate(0.3) is True

    def test_lt_comparison_not_met(self):
        sli = SLIDefinition(
            name="test",
            sli_id="test.lt",
            category=SLICategory.LATENCY,
            description="test",
            target=0.5,
            unit="seconds",
            comparison="lt",
            promql="",
        )
        assert sli.evaluate(0.5) is False
        assert sli.evaluate(0.6) is False

    def test_lte_comparison(self):
        sli = SLIDefinition(
            name="test",
            sli_id="test.lte",
            category=SLICategory.LATENCY,
            description="test",
            target=0.5,
            unit="seconds",
            comparison="lte",
            promql="",
        )
        assert sli.evaluate(0.5) is True
        assert sli.evaluate(0.4) is True
        assert sli.evaluate(0.6) is False

    def test_gt_comparison(self):
        sli = SLIDefinition(
            name="test",
            sli_id="test.gt",
            category=SLICategory.QUALITY,
            description="test",
            target=0.5,
            unit="ratio",
            comparison="gt",
            promql="",
        )
        assert sli.evaluate(0.6) is True
        assert sli.evaluate(0.5) is False
        assert sli.evaluate(0.4) is False

    def test_gte_comparison(self):
        sli = SLIDefinition(
            name="test",
            sli_id="test.gte",
            category=SLICategory.AVAILABILITY,
            description="test",
            target=99.9,
            unit="percent",
            comparison="gte",
            promql="",
        )
        assert sli.evaluate(99.9) is True
        assert sli.evaluate(100.0) is True
        assert sli.evaluate(99.8) is False

    def test_unknown_comparison_returns_false(self):
        sli = SLIDefinition(
            name="test",
            sli_id="test.bad",
            category=SLICategory.LATENCY,
            description="test",
            target=0.5,
            unit="seconds",
            comparison="neq",
            promql="",
        )
        assert sli.evaluate(0.3) is False


# ---------------------------------------------------------------------------
# SLO Definition
# ---------------------------------------------------------------------------


class TestSLODefinition:
    """Tests for SLODefinition properties."""

    def test_window_minutes(self):
        assert FACTORS_SLO.window_minutes == 30 * 24 * 60

    def test_error_budget_fraction(self):
        # 99.9% availability -> 0.001 error budget fraction
        assert FACTORS_SLO.error_budget_fraction == pytest.approx(0.001)

    def test_error_budget_minutes(self):
        expected = 30 * 24 * 60 * 0.001  # 43.2 minutes
        assert FACTORS_SLO.error_budget_minutes == pytest.approx(expected)

    def test_has_five_slis(self):
        assert len(FACTORS_SLO.slis) == 5

    def test_slos_list_contains_factors_slo(self):
        assert len(FACTORS_SLOS) == 1
        assert FACTORS_SLOS[0] is FACTORS_SLO


# ---------------------------------------------------------------------------
# Predefined SLIs
# ---------------------------------------------------------------------------


class TestPredefinedSLIs:
    """Test that predefined SLI constants have correct properties."""

    def test_request_latency_p99(self):
        assert SLI_REQUEST_LATENCY_P99.target == 0.5
        assert SLI_REQUEST_LATENCY_P99.comparison == "lt"
        assert SLI_REQUEST_LATENCY_P99.category == SLICategory.LATENCY

    def test_error_rate(self):
        assert SLI_ERROR_RATE.target == 0.001
        assert SLI_ERROR_RATE.comparison == "lt"
        assert SLI_ERROR_RATE.category == SLICategory.ERROR_RATE

    def test_search_latency_p95(self):
        assert SLI_SEARCH_LATENCY_P95.target == 0.2
        assert SLI_SEARCH_LATENCY_P95.comparison == "lt"
        assert SLI_SEARCH_LATENCY_P95.category == SLICategory.LATENCY

    def test_match_confidence_median(self):
        assert SLI_MATCH_CONFIDENCE_MEDIAN.target == 0.5
        assert SLI_MATCH_CONFIDENCE_MEDIAN.comparison == "gt"
        assert SLI_MATCH_CONFIDENCE_MEDIAN.category == SLICategory.QUALITY

    def test_availability(self):
        assert SLI_AVAILABILITY.target == 99.9
        assert SLI_AVAILABILITY.comparison == "gte"
        assert SLI_AVAILABILITY.category == SLICategory.AVAILABILITY

    def test_all_have_promql(self):
        for sli in FACTORS_SLO.slis:
            assert sli.promql, f"SLI {sli.sli_id} has empty promql"


# ---------------------------------------------------------------------------
# check_sli
# ---------------------------------------------------------------------------


class TestCheckSli:
    """Tests for check_sli function."""

    def test_sli_met(self):
        result = check_sli(SLI_REQUEST_LATENCY_P99, 0.35)
        assert result.met is True
        assert result.status == ComplianceStatus.MET
        assert result.actual == 0.35
        assert result.target == 0.5

    def test_sli_violated(self):
        result = check_sli(SLI_REQUEST_LATENCY_P99, 0.8)
        assert result.met is False
        assert result.status == ComplianceStatus.VIOLATED

    def test_result_has_timestamp(self):
        result = check_sli(SLI_ERROR_RATE, 0.0001)
        assert result.checked_at is not None
        assert len(result.checked_at) > 0

    def test_result_to_dict(self):
        result = check_sli(SLI_ERROR_RATE, 0.0001)
        d = result.to_dict()
        assert d["sli_id"] == "factors.errors.5xx"
        assert d["met"] is True
        assert d["status"] == "met"
        assert "checked_at" in d


# ---------------------------------------------------------------------------
# check_all_slis
# ---------------------------------------------------------------------------


class TestCheckAllSlis:
    """Tests for check_all_slis function."""

    def _good_values(self):
        return {
            "factors.latency.p99": 0.35,
            "factors.errors.5xx": 0.0005,
            "factors.search.latency.p95": 0.12,
            "factors.match.confidence.p50": 0.72,
            "factors.availability": 99.95,
        }

    def test_all_met(self):
        results = check_all_slis(self._good_values())
        assert len(results) == 5
        assert all(r.met for r in results)

    def test_one_violated(self):
        values = self._good_values()
        values["factors.latency.p99"] = 0.8  # Exceeds 500ms target
        results = check_all_slis(values)
        violated = [r for r in results if not r.met]
        assert len(violated) == 1
        assert violated[0].sli_id == "factors.latency.p99"

    def test_missing_value_marked_unknown(self):
        values = {"factors.latency.p99": 0.35}  # Only one value provided
        results = check_all_slis(values)
        unknown = [r for r in results if r.status == ComplianceStatus.UNKNOWN]
        assert len(unknown) == 4  # 4 out of 5 missing

    def test_custom_slo(self):
        custom_slo = SLODefinition(
            name="Custom",
            slo_id="custom-slo",
            service="test",
            description="test",
            target_availability=99.5,
            window_days=7,
            slis=[SLI_REQUEST_LATENCY_P99],
        )
        results = check_all_slis({"factors.latency.p99": 0.3}, slo=custom_slo)
        assert len(results) == 1
        assert results[0].met is True


# ---------------------------------------------------------------------------
# Error budget calculation
# ---------------------------------------------------------------------------


class TestCalculateErrorBudget:
    """Tests for calculate_error_budget function."""

    def test_perfect_availability(self):
        budget = calculate_error_budget(
            target_availability=99.9,
            actual_availability=100.0,
        )
        assert budget.consumed_percent == pytest.approx(0.0)
        assert budget.remaining_percent == pytest.approx(100.0)
        assert budget.health == BudgetHealth.HEALTHY

    def test_exactly_at_target(self):
        budget = calculate_error_budget(
            target_availability=99.9,
            actual_availability=99.9,
        )
        assert budget.consumed_percent == pytest.approx(100.0)
        assert budget.remaining_percent == pytest.approx(0.0)
        assert budget.health == BudgetHealth.EXHAUSTED

    def test_half_budget_consumed(self):
        # 99.9% target -> 0.1% budget. If actual is 99.95%, error = 0.05% which is 50% of 0.1%
        budget = calculate_error_budget(
            target_availability=99.9,
            actual_availability=99.95,
        )
        assert budget.consumed_percent == pytest.approx(50.0)
        assert budget.remaining_percent == pytest.approx(50.0)
        assert budget.health == BudgetHealth.WARNING

    def test_total_budget_minutes_30_day(self):
        budget = calculate_error_budget(
            target_availability=99.9,
            actual_availability=100.0,
            window_days=30,
        )
        expected_minutes = 30 * 24 * 60 * 0.001  # 43.2 minutes
        assert budget.total_budget_minutes == pytest.approx(expected_minutes)

    def test_custom_window_minutes(self):
        budget = calculate_error_budget(
            target_availability=99.9,
            actual_availability=100.0,
            window_minutes=10080,  # 7 days
        )
        expected = 10080 * 0.001
        assert budget.total_budget_minutes == pytest.approx(expected)

    def test_worse_than_target(self):
        budget = calculate_error_budget(
            target_availability=99.9,
            actual_availability=99.8,
        )
        assert budget.consumed_percent == pytest.approx(100.0)
        assert budget.health == BudgetHealth.EXHAUSTED

    def test_exhaustion_forecast(self):
        budget = calculate_error_budget(
            target_availability=99.9,
            actual_availability=99.95,
            elapsed_minutes=100,
        )
        # consumed 50% in 100 minutes, so 100 more minutes for the other 50%
        assert budget.exhaustion_forecast_hours is not None
        assert budget.exhaustion_forecast_hours > 0

    def test_no_forecast_when_no_consumption(self):
        budget = calculate_error_budget(
            target_availability=99.9,
            actual_availability=100.0,
        )
        assert budget.exhaustion_forecast_hours is None

    def test_to_dict(self):
        budget = calculate_error_budget(
            target_availability=99.9,
            actual_availability=99.95,
        )
        d = budget.to_dict()
        assert "slo_id" in d
        assert "total_budget_minutes" in d
        assert "health" in d
        assert d["health"] == "warning"


# ---------------------------------------------------------------------------
# Budget health classification
# ---------------------------------------------------------------------------


class TestClassifyBudgetHealth:
    """Tests for _classify_budget_health."""

    def test_healthy(self):
        assert _classify_budget_health(0.0) == BudgetHealth.HEALTHY
        assert _classify_budget_health(49.9) == BudgetHealth.HEALTHY

    def test_warning(self):
        assert _classify_budget_health(50.0) == BudgetHealth.WARNING
        assert _classify_budget_health(79.9) == BudgetHealth.WARNING

    def test_critical(self):
        assert _classify_budget_health(80.0) == BudgetHealth.CRITICAL
        assert _classify_budget_health(99.9) == BudgetHealth.CRITICAL

    def test_exhausted(self):
        assert _classify_budget_health(100.0) == BudgetHealth.EXHAUSTED
        assert _classify_budget_health(150.0) == BudgetHealth.EXHAUSTED


# ---------------------------------------------------------------------------
# Compliance summary
# ---------------------------------------------------------------------------


class TestComplianceSummary:
    """Tests for compliance_summary function."""

    def _good_values(self):
        return {
            "factors.latency.p99": 0.35,
            "factors.errors.5xx": 0.0005,
            "factors.search.latency.p95": 0.12,
            "factors.match.confidence.p50": 0.72,
            "factors.availability": 99.95,
        }

    def test_compliant_summary(self):
        summary = compliance_summary(
            current_values=self._good_values(),
            actual_availability=99.95,
        )
        assert summary["overall_status"] == "compliant"
        assert summary["slo_id"] == "factors-api-slo-99.9"
        assert summary["service"] == "factors"
        assert len(summary["sli_results"]) == 5
        assert "error_budget" in summary

    def test_at_risk_summary(self):
        values = self._good_values()
        values["factors.latency.p99"] = 0.8  # violates
        summary = compliance_summary(
            current_values=values,
            actual_availability=99.95,
        )
        # one SLI violated but budget warning -> at_risk
        assert summary["overall_status"] == "at_risk"

    def test_critical_summary(self):
        values = self._good_values()
        values["factors.latency.p99"] = 0.8  # violates
        summary = compliance_summary(
            current_values=values,
            actual_availability=99.85,  # budget nearly exhausted
        )
        assert summary["overall_status"] == "critical"

    def test_has_checked_at(self):
        summary = compliance_summary(
            current_values=self._good_values(),
            actual_availability=100.0,
        )
        assert summary["checked_at"] is not None


# ---------------------------------------------------------------------------
# SLICheckResult
# ---------------------------------------------------------------------------


class TestSLICheckResult:
    """Tests for SLICheckResult dataclass."""

    def test_auto_timestamp(self):
        result = SLICheckResult(
            sli_id="test",
            name="Test",
            target=1.0,
            actual=0.5,
            unit="ratio",
            met=True,
            status=ComplianceStatus.MET,
        )
        assert result.checked_at != ""

    def test_custom_timestamp(self):
        result = SLICheckResult(
            sli_id="test",
            name="Test",
            target=1.0,
            actual=0.5,
            unit="ratio",
            met=True,
            status=ComplianceStatus.MET,
            checked_at="2026-04-19T00:00:00+00:00",
        )
        assert result.checked_at == "2026-04-19T00:00:00+00:00"

    def test_to_dict_keys(self):
        result = SLICheckResult(
            sli_id="test",
            name="Test",
            target=1.0,
            actual=0.5,
            unit="ratio",
            met=True,
            status=ComplianceStatus.MET,
        )
        d = result.to_dict()
        expected_keys = {"sli_id", "name", "target", "actual", "unit", "met", "status", "checked_at"}
        assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# ErrorBudgetResult
# ---------------------------------------------------------------------------


class TestErrorBudgetResult:
    """Tests for ErrorBudgetResult dataclass."""

    def test_to_dict_rounding(self):
        result = ErrorBudgetResult(
            slo_id="test",
            target_availability=99.9,
            actual_availability=99.95,
            window_days=30,
            total_budget_minutes=43.2,
            consumed_minutes=21.6,
            remaining_minutes=21.6,
            consumed_percent=50.0,
            remaining_percent=50.0,
            health=BudgetHealth.WARNING,
            exhaustion_forecast_hours=1.5555,
        )
        d = result.to_dict()
        assert d["total_budget_minutes"] == 43.2
        assert d["exhaustion_forecast_hours"] == 1.6  # rounded to 1 decimal

    def test_none_forecast_in_dict(self):
        result = ErrorBudgetResult(
            slo_id="test",
            target_availability=99.9,
            actual_availability=100.0,
            window_days=30,
            total_budget_minutes=43.2,
            consumed_minutes=0.0,
            remaining_minutes=43.2,
            consumed_percent=0.0,
            remaining_percent=100.0,
            health=BudgetHealth.HEALTHY,
        )
        d = result.to_dict()
        assert d["exhaustion_forecast_hours"] is None
