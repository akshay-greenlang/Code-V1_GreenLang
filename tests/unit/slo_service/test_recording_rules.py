# -*- coding: utf-8 -*-
"""
Unit tests for Recording Rules Generator (OBS-005)

Tests SLI, error budget, and burn rate recording rule generation,
naming conventions, YAML output, and file writing.

Coverage target: 85%+ of recording_rules.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from pathlib import Path

import pytest

from greenlang.infrastructure.slo_service.models import BurnRateWindow
from greenlang.infrastructure.slo_service.recording_rules import (
    generate_all_recording_rules,
    generate_burn_rate_recording_rule,
    generate_error_budget_recording_rule,
    generate_sli_recording_rule,
    write_recording_rules_file,
)


class TestSLIRecordingRule:
    """Tests for SLI recording rule generation."""

    def test_generate_sli_recording_rule_availability(self, sample_slo):
        """Availability SLI recording rule has correct structure."""
        rule = generate_sli_recording_rule(sample_slo)
        assert rule["record"] == f"slo:{sample_slo.safe_name}:sli_ratio"
        assert "expr" in rule
        assert rule["labels"]["sli_type"] == "availability"

    def test_generate_sli_recording_rule_latency(self, sample_slo_latency):
        """Latency SLI recording rule references correct metric."""
        rule = generate_sli_recording_rule(sample_slo_latency)
        assert "sli_ratio" in rule["record"]
        assert rule["labels"]["sli_type"] == "latency"

    def test_recording_rule_naming_convention(self, sample_slo):
        """Rule names follow slo:<safe_name>:<metric> convention."""
        rule = generate_sli_recording_rule(sample_slo)
        assert rule["record"].startswith("slo:")
        assert rule["record"].endswith(":sli_ratio")

    def test_rule_labels_preserved(self, sample_slo):
        """Labels include slo_id and service."""
        rule = generate_sli_recording_rule(sample_slo)
        assert rule["labels"]["slo_id"] == sample_slo.slo_id
        assert rule["labels"]["service"] == sample_slo.service


class TestErrorBudgetRecordingRule:
    """Tests for error budget recording rule generation."""

    def test_generate_error_budget_recording_rule(self, sample_slo):
        """Error budget recording rule has correct metric name."""
        rule = generate_error_budget_recording_rule(sample_slo)
        expected = f"slo:{sample_slo.safe_name}:error_budget_remaining"
        assert rule["record"] == expected
        assert "clamp_min" in rule["expr"]

    def test_error_budget_rule_labels(self, sample_slo):
        """Error budget rule has slo_id and service labels."""
        rule = generate_error_budget_recording_rule(sample_slo)
        assert rule["labels"]["slo_id"] == sample_slo.slo_id
        assert rule["labels"]["service"] == sample_slo.service


class TestBurnRateRecordingRule:
    """Tests for burn rate recording rule generation."""

    @pytest.mark.parametrize("window", [
        BurnRateWindow.FAST,
        BurnRateWindow.MEDIUM,
        BurnRateWindow.SLOW,
    ])
    def test_generate_burn_rate_recording_rule(self, sample_slo, window):
        """Burn rate recording rule for each window tier."""
        rule = generate_burn_rate_recording_rule(sample_slo, window)
        expected = f"slo:{sample_slo.safe_name}:burn_rate_{window.value}"
        assert rule["record"] == expected
        assert rule["labels"]["burn_window"] == window.value


class TestGenerateAllRecordingRules:
    """Tests for generate_all_recording_rules."""

    def test_generate_all_rules_includes_all_sections(self, sample_slo_list):
        """Output includes SLI, budget, and burn rate groups."""
        result = generate_all_recording_rules(sample_slo_list)
        groups = result["groups"]
        assert len(groups) == 3
        assert groups[0]["name"] == "slo_sli_ratio_rules"
        assert groups[1]["name"] == "slo_error_budget_rules"
        assert groups[2]["name"] == "slo_burn_rate_rules"

    def test_rule_group_structure(self, sample_slo_list):
        """Each group has name, interval, and rules."""
        result = generate_all_recording_rules(sample_slo_list)
        for group in result["groups"]:
            assert "name" in group
            assert "interval" in group
            assert "rules" in group

    def test_evaluation_interval(self, sample_slo_list):
        """SLI and budget groups have 60s interval, burn rate 30s."""
        result = generate_all_recording_rules(sample_slo_list)
        assert result["groups"][0]["interval"] == "60s"
        assert result["groups"][1]["interval"] == "60s"
        assert result["groups"][2]["interval"] == "30s"

    def test_rule_count_per_slo(self, sample_slo_list):
        """Each SLO produces 1 SLI + 1 budget + 3 burn rate = 5 rules."""
        result = generate_all_recording_rules(sample_slo_list)
        n = len(sample_slo_list)
        assert len(result["groups"][0]["rules"]) == n      # SLI
        assert len(result["groups"][1]["rules"]) == n      # Budget
        assert len(result["groups"][2]["rules"]) == n * 3  # Burn rate

    def test_empty_slos_empty_rules(self):
        """Empty SLO list produces empty rule groups."""
        result = generate_all_recording_rules([])
        for group in result["groups"]:
            assert group["rules"] == []

    def test_multiple_slos_multiple_rules(self, sample_slo_list):
        """Multiple SLOs generate proportional rules."""
        result = generate_all_recording_rules(sample_slo_list)
        total_rules = sum(len(g["rules"]) for g in result["groups"])
        expected = len(sample_slo_list) * 5  # 1 SLI + 1 budget + 3 burn
        assert total_rules == expected


class TestWriteRecordingRulesFile:
    """Tests for write_recording_rules_file."""

    def test_write_rules_file_creates_file(self, tmp_path, sample_slo_list):
        """Writing recording rules creates a YAML file."""
        output = str(tmp_path / "rules.yaml")
        result_path = write_recording_rules_file(sample_slo_list, output)
        assert Path(result_path).exists()

    def test_recording_rule_yaml_output(self, tmp_path, sample_slo_list):
        """Written YAML file parses correctly."""
        import yaml

        output = str(tmp_path / "rules.yaml")
        write_recording_rules_file(sample_slo_list, output)

        with open(output) as f:
            data = yaml.safe_load(f)
        assert "groups" in data

    def test_write_creates_parent_dirs(self, tmp_path, sample_slo_list):
        """Parent directories are created automatically."""
        output = str(tmp_path / "nested" / "dir" / "rules.yaml")
        write_recording_rules_file(sample_slo_list, output)
        assert Path(output).exists()
