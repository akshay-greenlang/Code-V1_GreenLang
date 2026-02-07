# -*- coding: utf-8 -*-
"""
Unit tests for Burn Rate Calculator (OBS-005)

Tests burn rate calculation formula, multi-window threshold evaluation,
alert firing logic, exhaustion time estimates, and PromQL rule generation.

Coverage target: 85%+ of burn_rate.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import pytest

from greenlang.infrastructure.slo_service.burn_rate import (
    build_burn_rate_promql,
    calculate_burn_rate,
    evaluate_burn_rate_windows,
    generate_all_burn_rate_rules,
    generate_burn_rate_alert_rule,
    should_alert,
    time_to_exhaustion_hours,
)
from greenlang.infrastructure.slo_service.models import BurnRateWindow


class TestBurnRateCalculation:
    """Tests for the calculate_burn_rate function."""

    def test_burn_rate_calculation_formula(self):
        """Burn rate = actual error rate / allowed error rate."""
        rate = calculate_burn_rate(0.01, 0.001)
        assert rate == pytest.approx(10.0)

    def test_burn_rate_fast_threshold_14_4(self):
        """Burn rate of 14.4x exhausts 30d budget in ~2 hours."""
        rate = calculate_burn_rate(0.0144, 0.001)
        assert rate == pytest.approx(14.4)

    def test_burn_rate_medium_threshold_6_0(self):
        """Burn rate of 6x exhausts 30d budget in ~5 days."""
        rate = calculate_burn_rate(0.006, 0.001)
        assert rate == pytest.approx(6.0)

    def test_burn_rate_slow_threshold_1_0(self):
        """Burn rate of 1x means budget consumed at the expected rate."""
        rate = calculate_burn_rate(0.001, 0.001)
        assert rate == pytest.approx(1.0)

    def test_burn_rate_with_zero_error_rate(self):
        """Zero error rate produces zero burn rate."""
        rate = calculate_burn_rate(0.0, 0.001)
        assert rate == 0.0

    def test_burn_rate_with_100_percent_errors(self):
        """100% error rate produces very high burn rate."""
        rate = calculate_burn_rate(1.0, 0.001)
        assert rate == pytest.approx(1000.0)

    def test_burn_rate_zero_budget(self):
        """Zero budget fraction returns inf for any error."""
        rate = calculate_burn_rate(0.01, 0.0)
        assert rate == float("inf")

    def test_burn_rate_zero_budget_no_errors(self):
        """Zero budget fraction with zero errors returns 0."""
        rate = calculate_burn_rate(0.0, 0.0)
        assert rate == 0.0


class TestShouldAlert:
    """Tests for the should_alert function."""

    def test_burn_rate_alert_fires_when_both_windows_exceed(self):
        """Alert fires when both long and short exceed threshold."""
        assert should_alert(15.0, 16.0, 14.4) is True

    def test_burn_rate_alert_no_fire_long_only(self):
        """Alert does not fire when only long window exceeds."""
        assert should_alert(15.0, 10.0, 14.4) is False

    def test_burn_rate_alert_no_fire_short_only(self):
        """Alert does not fire when only short window exceeds."""
        assert should_alert(10.0, 15.0, 14.4) is False

    def test_burn_rate_alert_no_fire_below_threshold(self):
        """Alert does not fire when both are below threshold."""
        assert should_alert(5.0, 3.0, 14.4) is False

    def test_burn_rate_alert_at_exact_threshold(self):
        """Alert does not fire at exact threshold (must exceed)."""
        assert should_alert(14.4, 14.4, 14.4) is False

    @pytest.mark.parametrize("threshold", [14.4, 6.0, 1.0])
    def test_custom_thresholds(self, threshold):
        """Alert fires at each threshold level."""
        assert should_alert(threshold + 1, threshold + 1, threshold) is True
        assert should_alert(threshold - 0.1, threshold + 1, threshold) is False


class TestEvaluateBurnRateWindows:
    """Tests for evaluate_burn_rate_windows."""

    def test_fast_burn_alert(self, sample_slo):
        """Fast burn triggers critical alert."""
        burn_rates = {
            "fast": {"long": 16.0, "short": 17.0},
            "medium": {"long": 3.0, "short": 2.0},
            "slow": {"long": 0.5, "short": 0.3},
        }
        alerts = evaluate_burn_rate_windows(sample_slo, burn_rates)
        assert len(alerts) == 1
        assert alerts[0].burn_window == "fast"
        assert alerts[0].severity == "critical"

    def test_medium_burn_alert(self, sample_slo):
        """Medium burn triggers warning alert."""
        burn_rates = {
            "fast": {"long": 5.0, "short": 4.0},
            "medium": {"long": 7.0, "short": 8.0},
            "slow": {"long": 0.5, "short": 0.3},
        }
        alerts = evaluate_burn_rate_windows(sample_slo, burn_rates)
        assert len(alerts) == 1
        assert alerts[0].burn_window == "medium"
        assert alerts[0].severity == "warning"

    def test_slow_burn_alert(self, sample_slo):
        """Slow burn triggers info alert."""
        burn_rates = {
            "fast": {"long": 5.0, "short": 4.0},
            "medium": {"long": 3.0, "short": 2.0},
            "slow": {"long": 1.5, "short": 1.2},
        }
        alerts = evaluate_burn_rate_windows(sample_slo, burn_rates)
        assert len(alerts) == 1
        assert alerts[0].burn_window == "slow"
        assert alerts[0].severity == "info"

    def test_no_alerts_when_all_below(self, sample_slo):
        """No alerts when all burn rates are below thresholds."""
        burn_rates = {
            "fast": {"long": 5.0, "short": 4.0},
            "medium": {"long": 3.0, "short": 2.0},
            "slow": {"long": 0.5, "short": 0.3},
        }
        alerts = evaluate_burn_rate_windows(sample_slo, burn_rates)
        assert len(alerts) == 0

    def test_multiple_alerts(self, sample_slo):
        """Multiple windows can fire simultaneously."""
        burn_rates = {
            "fast": {"long": 20.0, "short": 18.0},
            "medium": {"long": 8.0, "short": 7.0},
            "slow": {"long": 1.5, "short": 1.2},
        }
        alerts = evaluate_burn_rate_windows(sample_slo, burn_rates)
        assert len(alerts) == 3


class TestExhaustionTime:
    """Tests for time_to_exhaustion_hours."""

    def test_fast_burn_exhausts_in_2_hours(self):
        """Burn rate 14.4x on a 30d window exhausts in ~50 hours."""
        # total_hours = 30*24 = 720, 720/14.4 = 50
        hours = time_to_exhaustion_hours(14.4, 30)
        assert hours == pytest.approx(50.0)

    def test_medium_burn_exhausts_in_1_day(self):
        """Burn rate 6x on a 30d window exhausts in 5 days (120h)."""
        hours = time_to_exhaustion_hours(6.0, 30)
        assert hours == pytest.approx(120.0)

    def test_slow_burn_exhausts_in_30_days(self):
        """Burn rate 1x on a 30d window exhausts in exactly 30 days."""
        hours = time_to_exhaustion_hours(1.0, 30)
        assert hours == pytest.approx(720.0)

    def test_zero_burn_rate(self):
        """Zero burn rate returns infinity."""
        assert time_to_exhaustion_hours(0.0) == float("inf")


class TestBurnRatePromQL:
    """Tests for PromQL generation."""

    def test_burn_rate_promql_generation(self, sample_slo):
        """Generated PromQL contains the error budget fraction."""
        promql = build_burn_rate_promql(sample_slo, "1h")
        assert "1h" in promql
        assert str(sample_slo.error_budget_fraction) in promql
        assert sample_slo.sli.good_query in promql

    @pytest.mark.parametrize("window", ["5m", "30m", "1h", "6h", "3d"])
    def test_burn_rate_window_configuration(self, sample_slo, window):
        """PromQL correctly references the given window."""
        promql = build_burn_rate_promql(sample_slo, window)
        assert f"[{window}]" in promql


class TestBurnRateAlertRuleGeneration:
    """Tests for alert rule generation."""

    def test_generate_burn_rate_alert_rule_format(self, sample_slo):
        """Generated alert rule has required fields."""
        rule = generate_burn_rate_alert_rule(sample_slo, BurnRateWindow.FAST)
        assert "alert" in rule
        assert "expr" in rule
        assert "for" in rule
        assert "labels" in rule
        assert "annotations" in rule

    @pytest.mark.parametrize("window,severity", [
        (BurnRateWindow.FAST, "critical"),
        (BurnRateWindow.MEDIUM, "warning"),
        (BurnRateWindow.SLOW, "info"),
    ])
    def test_burn_rate_alert_severity_mapping(self, sample_slo, window, severity):
        """Alert severity matches the window tier."""
        rule = generate_burn_rate_alert_rule(sample_slo, window)
        assert rule["labels"]["severity"] == severity

    def test_burn_rate_alert_annotations(self, sample_slo):
        """Alert annotations include runbook URL."""
        rule = generate_burn_rate_alert_rule(sample_slo, BurnRateWindow.FAST)
        assert "runbook_url" in rule["annotations"]
        assert sample_slo.slo_id in rule["annotations"]["runbook_url"]

    def test_generate_all_burn_rate_rules(self, sample_slo_list):
        """All SLOs get 3 burn rate rules each."""
        result = generate_all_burn_rate_rules(sample_slo_list)
        rules = result["groups"][0]["rules"]
        assert len(rules) == len(sample_slo_list) * 3

    def test_generate_all_burn_rate_rules_disabled_excluded(self, slo_factory):
        """Disabled SLOs are excluded from rule generation."""
        slos = [
            slo_factory(slo_id="on", name="Enabled", enabled=True),
            slo_factory(slo_id="off", name="Disabled", enabled=False),
        ]
        result = generate_all_burn_rate_rules(slos)
        rules = result["groups"][0]["rules"]
        assert len(rules) == 3  # Only the enabled SLO's 3 windows
