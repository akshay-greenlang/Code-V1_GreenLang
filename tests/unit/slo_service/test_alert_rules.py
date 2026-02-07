# -*- coding: utf-8 -*-
"""
Unit tests for Alert Rules Generator (OBS-005)

Tests burn rate alerts, budget threshold alerts, self-monitoring alerts,
severity mapping, annotations, and YAML file writing.

Coverage target: 85%+ of alert_rules.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from pathlib import Path

import pytest

from greenlang.infrastructure.slo_service.alert_rules import (
    generate_all_alert_rules,
    generate_budget_critical_alert,
    generate_budget_exhausted_alert,
    generate_budget_warning_alert,
    generate_self_monitoring_alerts,
    write_alert_rules_file,
)
from greenlang.infrastructure.slo_service.models import BurnRateWindow


class TestBurnRateAlertRules:
    """Tests for burn rate alert rule generation."""

    def test_generate_burn_rate_fast_alert(self, sample_slo):
        """Fast burn rate alert has critical severity."""
        from greenlang.infrastructure.slo_service.burn_rate import generate_burn_rate_alert_rule
        rule = generate_burn_rate_alert_rule(sample_slo, BurnRateWindow.FAST)
        assert rule["labels"]["severity"] == "critical"
        assert "for" in rule
        assert "2m" == rule["for"]

    def test_generate_burn_rate_medium_alert(self, sample_slo):
        """Medium burn rate alert has warning severity."""
        from greenlang.infrastructure.slo_service.burn_rate import generate_burn_rate_alert_rule
        rule = generate_burn_rate_alert_rule(sample_slo, BurnRateWindow.MEDIUM)
        assert rule["labels"]["severity"] == "warning"
        assert rule["for"] == "5m"

    def test_generate_burn_rate_slow_alert(self, sample_slo):
        """Slow burn rate alert has info severity."""
        from greenlang.infrastructure.slo_service.burn_rate import generate_burn_rate_alert_rule
        rule = generate_burn_rate_alert_rule(sample_slo, BurnRateWindow.SLOW)
        assert rule["labels"]["severity"] == "info"
        assert rule["for"] == "30m"


class TestBudgetAlertRules:
    """Tests for budget threshold alert generation."""

    def test_generate_budget_exhausted_alert(self, sample_slo):
        """Budget exhausted alert has critical severity."""
        rule = generate_budget_exhausted_alert(sample_slo)
        assert "Exhausted" in rule["alert"]
        assert rule["labels"]["severity"] == "critical"
        assert "runbook_url" in rule["annotations"]

    def test_generate_budget_critical_alert(self, sample_slo):
        """Budget critical alert has warning severity."""
        rule = generate_budget_critical_alert(sample_slo)
        assert "Critical" in rule["alert"]
        assert rule["labels"]["severity"] == "warning"

    def test_generate_budget_warning_alert(self, sample_slo):
        """Budget warning alert has info severity."""
        rule = generate_budget_warning_alert(sample_slo)
        assert "Warning" in rule["alert"]
        assert rule["labels"]["severity"] == "info"

    @pytest.mark.parametrize("generator,expected_severity", [
        (generate_budget_exhausted_alert, "critical"),
        (generate_budget_critical_alert, "warning"),
        (generate_budget_warning_alert, "info"),
    ])
    def test_alert_severity_mapping(self, sample_slo, generator, expected_severity):
        """Budget alerts have correct severity levels."""
        rule = generator(sample_slo)
        assert rule["labels"]["severity"] == expected_severity

    def test_alert_annotations_include_runbook(self, sample_slo):
        """All budget alerts include runbook URLs."""
        for gen in [generate_budget_exhausted_alert, generate_budget_critical_alert, generate_budget_warning_alert]:
            rule = gen(sample_slo)
            assert "runbook_url" in rule["annotations"]
            assert sample_slo.slo_id in rule["annotations"]["runbook_url"]

    def test_alert_labels_include_service(self, sample_slo):
        """All budget alerts include service label."""
        for gen in [generate_budget_exhausted_alert, generate_budget_critical_alert, generate_budget_warning_alert]:
            rule = gen(sample_slo)
            assert rule["labels"]["service"] == sample_slo.service

    def test_alert_expression_correctness(self, sample_slo):
        """Alert expressions reference the correct recording rule metric."""
        rule = generate_budget_exhausted_alert(sample_slo)
        assert sample_slo.safe_name in rule["expr"]
        assert "error_budget_remaining" in rule["expr"]

    @pytest.mark.parametrize("generator,for_duration", [
        (generate_budget_exhausted_alert, "5m"),
        (generate_budget_critical_alert, "15m"),
        (generate_budget_warning_alert, "30m"),
    ])
    def test_alert_for_duration(self, sample_slo, generator, for_duration):
        """Alert 'for' durations are correct."""
        rule = generator(sample_slo)
        assert rule["for"] == for_duration


class TestSelfMonitoringAlerts:
    """Tests for self-monitoring alert generation."""

    def test_self_monitoring_alerts(self):
        """Self-monitoring alerts are generated."""
        alerts = generate_self_monitoring_alerts()
        assert len(alerts) == 2
        names = [a["alert"] for a in alerts]
        assert "SLOServiceEvaluationStale" in names
        assert "SLOServiceEvaluationErrors" in names


class TestGenerateAllAlertRules:
    """Tests for generate_all_alert_rules."""

    def test_generate_all_alerts_yaml(self, sample_slo_list):
        """Full alert rules structure has 3 groups."""
        result = generate_all_alert_rules(sample_slo_list)
        assert len(result["groups"]) == 3
        assert result["groups"][0]["name"] == "slo_burn_rate_alerts"
        assert result["groups"][1]["name"] == "slo_budget_alerts"
        assert result["groups"][2]["name"] == "slo_self_monitoring"

    def test_burn_rate_rule_count(self, sample_slo_list):
        """3 burn rate rules per SLO."""
        result = generate_all_alert_rules(sample_slo_list)
        burn_rules = result["groups"][0]["rules"]
        assert len(burn_rules) == len(sample_slo_list) * 3

    def test_budget_rule_count(self, sample_slo_list):
        """3 budget rules per SLO (exhausted, critical, warning)."""
        result = generate_all_alert_rules(sample_slo_list)
        budget_rules = result["groups"][1]["rules"]
        assert len(budget_rules) == len(sample_slo_list) * 3

    def test_empty_slos_no_alerts(self):
        """Empty SLO list produces no rules except self-monitoring."""
        result = generate_all_alert_rules([])
        assert len(result["groups"][0]["rules"]) == 0  # burn rate
        assert len(result["groups"][1]["rules"]) == 0  # budget
        assert len(result["groups"][2]["rules"]) == 2  # self-monitoring


class TestWriteAlertRulesFile:
    """Tests for write_alert_rules_file."""

    def test_write_alerts_file(self, tmp_path, sample_slo_list):
        """Writing alert rules creates a YAML file."""
        output = str(tmp_path / "alerts.yaml")
        result_path = write_alert_rules_file(sample_slo_list, output)
        assert Path(result_path).exists()

    def test_written_yaml_parses(self, tmp_path, sample_slo_list):
        """Written YAML parses correctly."""
        import yaml
        output = str(tmp_path / "alerts.yaml")
        write_alert_rules_file(sample_slo_list, output)
        with open(output) as f:
            data = yaml.safe_load(f)
        assert "groups" in data
