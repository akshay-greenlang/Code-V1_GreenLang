# -*- coding: utf-8 -*-
"""
Unit tests for Dashboard Generator (OBS-005)

Tests Grafana overview and error budget dashboard generation, panel
types, grid positions, templating variables, and file writing.

Coverage target: 85%+ of dashboard_generator.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from greenlang.infrastructure.slo_service.dashboard_generator import (
    generate_error_budget_dashboard,
    generate_overview_dashboard,
    write_dashboards,
)


class TestOverviewDashboard:
    """Tests for SLO overview dashboard generation."""

    def test_generate_overview_dashboard_structure(self, sample_slo_list):
        """Overview dashboard has required top-level keys."""
        dashboard = generate_overview_dashboard(sample_slo_list)
        assert "uid" in dashboard
        assert "title" in dashboard
        assert "panels" in dashboard
        assert "templating" in dashboard

    def test_overview_dashboard_has_24_panels(self, sample_slo_list):
        """Overview dashboard generates exactly 24 panels."""
        dashboard = generate_overview_dashboard(sample_slo_list)
        assert len(dashboard["panels"]) == 24

    def test_dashboard_uid_uniqueness(self, sample_slo_list):
        """Overview and budget dashboards have different UIDs."""
        overview = generate_overview_dashboard(sample_slo_list)
        budget = generate_error_budget_dashboard(sample_slo_list)
        assert overview["uid"] != budget["uid"]

    def test_dashboard_title(self, sample_slo_list):
        """Overview dashboard has the expected title."""
        dashboard = generate_overview_dashboard(sample_slo_list)
        assert dashboard["title"] == "SLO Overview"

    def test_dashboard_tags(self, sample_slo_list):
        """Dashboard has expected tags."""
        dashboard = generate_overview_dashboard(sample_slo_list)
        assert "slo" in dashboard["tags"]
        assert "obs-005" in dashboard["tags"]

    def test_panel_types_correct(self, sample_slo_list):
        """Dashboard uses expected panel types."""
        dashboard = generate_overview_dashboard(sample_slo_list)
        panel_types = {p["type"] for p in dashboard["panels"]}
        expected_types = {"stat", "gauge", "timeseries", "table", "piechart", "barchart"}
        assert expected_types.issubset(panel_types)

    def test_panel_grid_positions(self, sample_slo_list):
        """All panels have gridPos with x, y, w, h."""
        dashboard = generate_overview_dashboard(sample_slo_list)
        for panel in dashboard["panels"]:
            pos = panel["gridPos"]
            assert "x" in pos
            assert "y" in pos
            assert "w" in pos
            assert "h" in pos
            assert pos["w"] > 0
            assert pos["h"] > 0

    def test_dashboard_templating_variables(self, sample_slo_list):
        """Templating includes service and slo_id variables."""
        dashboard = generate_overview_dashboard(sample_slo_list)
        var_names = [v["name"] for v in dashboard["templating"]["list"]]
        assert "service" in var_names
        assert "slo_id" in var_names

    def test_dashboard_datasource_references(self, sample_slo_list):
        """Panels reference the Prometheus datasource."""
        dashboard = generate_overview_dashboard(sample_slo_list)
        for panel in dashboard["panels"]:
            ds = panel.get("datasource", {})
            assert ds.get("type") == "prometheus"

    def test_color_thresholds_for_budget(self, sample_slo_list):
        """Budget gauge panels have red/yellow/green thresholds."""
        dashboard = generate_overview_dashboard(sample_slo_list)
        budget_panels = [
            p for p in dashboard["panels"]
            if p["type"] == "gauge" and "Budget" in p["title"]
        ]
        for panel in budget_panels:
            steps = panel["fieldConfig"]["defaults"]["thresholds"]["steps"]
            colors = [s["color"] for s in steps]
            assert "red" in colors
            assert "green" in colors


class TestErrorBudgetDashboard:
    """Tests for error budget detail dashboard."""

    def test_error_budget_dashboard_has_12_panels(self, sample_slo_list):
        """Error budget dashboard generates exactly 12 panels."""
        dashboard = generate_error_budget_dashboard(sample_slo_list)
        assert len(dashboard["panels"]) == 12

    def test_error_budget_dashboard_structure(self, sample_slo_list):
        """Error budget dashboard has required structure."""
        dashboard = generate_error_budget_dashboard(sample_slo_list)
        assert dashboard["uid"] == "slo-error-budget"
        assert "error-budget" in dashboard["tags"]


class TestWriteDashboards:
    """Tests for write_dashboards."""

    def test_write_dashboards_creates_files(self, tmp_path, sample_slo_list):
        """Writing dashboards creates 2 JSON files."""
        paths = write_dashboards(sample_slo_list, str(tmp_path))
        assert len(paths) == 2
        for path in paths:
            assert Path(path).exists()
            assert path.endswith(".json")

    def test_written_dashboards_are_valid_json(self, tmp_path, sample_slo_list):
        """Written dashboard files contain valid JSON."""
        paths = write_dashboards(sample_slo_list, str(tmp_path))
        for path in paths:
            with open(path) as f:
                data = json.load(f)
            assert "panels" in data
