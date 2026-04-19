"""
Unit tests for PACK-046 Templates (Executive Dashboard + Detailed Report).

Tests all 2 implemented templates with 50+ tests covering:
  - IntensityExecutiveDashboard: Markdown, HTML, JSON rendering
  - IntensityDetailedReport: Markdown, HTML, JSON rendering
  - Provenance hash generation
  - All dashboard sections (metrics, benchmark, target, decomposition, actions)
  - All report sections (methodology, scopes, denominators, time-series, entities)
  - Empty data handling
  - Config override support
  - Processing time tracking
  - HTML structure validation
  - JSON chart data building

Author: GreenLang QA Team
"""

import json
import sys
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from templates.intensity_executive_dashboard import (
    ActionItem,
    BenchmarkResult,
    ChangeDirection,
    DashboardInput,
    DecompositionSummary,
    IntensityExecutiveDashboard,
    IntensityMetricItem,
    OutputFormat,
    SparklinePoint,
    TargetStatus,
    TrafficLight,
    _arrow,
    _html_arrow,
    _tl_color,
    _tl_css,
    _tl_label,
)
from templates.intensity_detailed_report import (
    DataSourceEntry,
    DenominatorDetail,
    DetailedReportInput,
    EntityBreakdown,
    IntensityByDenominator,
    IntensityByScope,
    IntensityDetailedReport,
    TimeSeriesRow,
)


# ---------------------------------------------------------------------------
# Helper Function Tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Tests for template helper functions."""

    def test_arrow_up(self):
        assert _arrow(ChangeDirection.UP) == "^"

    def test_arrow_down(self):
        assert _arrow(ChangeDirection.DOWN) == "v"

    def test_arrow_flat(self):
        assert _arrow(ChangeDirection.FLAT) == "-"

    def test_html_arrow_up(self):
        assert _html_arrow(ChangeDirection.UP) == "&#9650;"

    def test_html_arrow_down(self):
        assert _html_arrow(ChangeDirection.DOWN) == "&#9660;"

    def test_tl_label_green(self):
        assert _tl_label(TrafficLight.GREEN) == "GREEN"

    def test_tl_label_red(self):
        assert _tl_label(TrafficLight.RED) == "RED"

    def test_tl_css_green(self):
        assert _tl_css(TrafficLight.GREEN) == "tl-green"

    def test_tl_css_amber(self):
        assert _tl_css(TrafficLight.AMBER) == "tl-amber"

    def test_tl_color_green(self):
        assert _tl_color(TrafficLight.GREEN) == "#2a9d8f"

    def test_tl_color_red(self):
        assert _tl_color(TrafficLight.RED) == "#e76f51"


# ---------------------------------------------------------------------------
# Pydantic Model Tests
# ---------------------------------------------------------------------------


class TestDashboardPydanticModels:
    """Tests for dashboard Pydantic models."""

    def test_intensity_metric_item_yoy_auto_computed(self):
        item = IntensityMetricItem(
            metric_name="Revenue Intensity",
            current_value=25.0,
            prior_value=30.0,
        )
        assert item.yoy_change_pct is not None
        expected = ((25.0 - 30.0) / 30.0) * 100.0
        assert item.yoy_change_pct == pytest.approx(expected, abs=0.01)

    def test_intensity_metric_item_yoy_provided(self):
        item = IntensityMetricItem(
            metric_name="Revenue Intensity",
            current_value=25.0,
            prior_value=30.0,
            yoy_change_pct=-10.0,
        )
        assert item.yoy_change_pct == -10.0

    def test_intensity_metric_item_yoy_none_without_prior(self):
        item = IntensityMetricItem(
            metric_name="Revenue Intensity",
            current_value=25.0,
        )
        assert item.yoy_change_pct is None

    def test_benchmark_result_valid(self):
        b = BenchmarkResult(
            metric_name="Revenue Intensity",
            percentile_rank=35.0,
            peer_group="Manufacturing EU",
        )
        assert b.percentile_rank == 35.0

    def test_target_status_valid(self):
        t = TargetStatus(
            target_name="SBTi 2030",
            target_year=2030,
            target_value=15.0,
            current_value=25.5,
            base_value=40.0,
            pct_achieved=58.0,
            on_track=True,
            status=TrafficLight.GREEN,
        )
        assert t.pct_achieved == 58.0

    def test_dashboard_input_defaults(self):
        d = DashboardInput()
        assert d.company_name == "Organization"
        assert d.intensity_metrics == []


# ---------------------------------------------------------------------------
# IntensityExecutiveDashboard Tests
# ---------------------------------------------------------------------------


class TestExecutiveDashboardInit:
    """Tests for IntensityExecutiveDashboard initialisation."""

    def test_init_default(self):
        template = IntensityExecutiveDashboard()
        assert template is not None
        assert template.config == {}

    def test_init_with_config(self):
        template = IntensityExecutiveDashboard(config={"company_name": "Override"})
        assert template.config["company_name"] == "Override"


class TestExecutiveDashboardMarkdown:
    """Tests for Markdown rendering."""

    def test_render_markdown_returns_string(self, sample_dashboard_data):
        template = IntensityExecutiveDashboard()
        result = template.render_markdown(sample_dashboard_data)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_render_markdown_contains_header(self, sample_dashboard_data):
        template = IntensityExecutiveDashboard()
        result = template.render_markdown(sample_dashboard_data)
        assert "# Intensity Metrics Dashboard" in result
        assert "ACME Corp" in result

    def test_render_markdown_key_metrics_section(self, sample_dashboard_data):
        template = IntensityExecutiveDashboard()
        result = template.render_markdown(sample_dashboard_data)
        assert "## 1. Key Intensity Metrics" in result
        assert "Revenue Intensity" in result

    def test_render_markdown_benchmark_section(self, sample_dashboard_data):
        template = IntensityExecutiveDashboard()
        result = template.render_markdown(sample_dashboard_data)
        assert "## 2. Benchmark Position" in result

    def test_render_markdown_target_section(self, sample_dashboard_data):
        template = IntensityExecutiveDashboard()
        result = template.render_markdown(sample_dashboard_data)
        assert "## 3. Target Progress" in result
        assert "SBTi 2030" in result

    def test_render_markdown_decomposition_section(self, sample_dashboard_data):
        template = IntensityExecutiveDashboard()
        result = template.render_markdown(sample_dashboard_data)
        assert "## 4. Decomposition Highlights" in result
        assert "Activity Effect" in result

    def test_render_markdown_action_items_section(self, sample_dashboard_data):
        template = IntensityExecutiveDashboard()
        result = template.render_markdown(sample_dashboard_data)
        assert "## 5. Action Items" in result
        assert "heat pump" in result

    def test_render_markdown_provenance_hash(self, sample_dashboard_data):
        template = IntensityExecutiveDashboard()
        result = template.render_markdown(sample_dashboard_data)
        assert "Provenance Hash:" in result

    def test_render_markdown_empty_data(self):
        template = IntensityExecutiveDashboard()
        result = template.render_markdown({})
        assert "# Intensity Metrics Dashboard" in result

    def test_render_markdown_processing_time(self, sample_dashboard_data):
        template = IntensityExecutiveDashboard()
        template.render_markdown(sample_dashboard_data)
        assert template.processing_time_ms >= 0.0


class TestExecutiveDashboardHTML:
    """Tests for HTML rendering."""

    def test_render_html_returns_string(self, sample_dashboard_data):
        template = IntensityExecutiveDashboard()
        result = template.render_html(sample_dashboard_data)
        assert isinstance(result, str)

    def test_render_html_contains_doctype(self, sample_dashboard_data):
        template = IntensityExecutiveDashboard()
        result = template.render_html(sample_dashboard_data)
        assert "<!DOCTYPE html>" in result

    def test_render_html_contains_title(self, sample_dashboard_data):
        template = IntensityExecutiveDashboard()
        result = template.render_html(sample_dashboard_data)
        assert "<title>Intensity Dashboard - ACME Corp</title>" in result

    def test_render_html_contains_css(self, sample_dashboard_data):
        template = IntensityExecutiveDashboard()
        result = template.render_html(sample_dashboard_data)
        assert "<style>" in result
        assert ".tl-green" in result

    def test_render_html_metric_cards(self, sample_dashboard_data):
        template = IntensityExecutiveDashboard()
        result = template.render_html(sample_dashboard_data)
        assert "metric-card" in result

    def test_render_html_provenance_footer(self, sample_dashboard_data):
        template = IntensityExecutiveDashboard()
        result = template.render_html(sample_dashboard_data)
        assert "Provenance Hash:" in result


class TestExecutiveDashboardJSON:
    """Tests for JSON rendering."""

    def test_render_json_returns_dict(self, sample_dashboard_data):
        template = IntensityExecutiveDashboard()
        result = template.render_json(sample_dashboard_data)
        assert isinstance(result, dict)

    def test_render_json_has_template_name(self, sample_dashboard_data):
        template = IntensityExecutiveDashboard()
        result = template.render_json(sample_dashboard_data)
        assert result["template"] == "intensity_executive_dashboard"

    def test_render_json_has_provenance_hash(self, sample_dashboard_data):
        template = IntensityExecutiveDashboard()
        result = template.render_json(sample_dashboard_data)
        assert len(result["provenance_hash"]) == 64

    def test_render_json_sparkline_data(self, sample_dashboard_data):
        template = IntensityExecutiveDashboard()
        result = template.render_json(sample_dashboard_data)
        sparklines = result["chart_data"]["sparklines"]
        assert len(sparklines) > 0
        assert sparklines[0]["metric_name"] == "Revenue Intensity"
        assert len(sparklines[0]["points"]) == 4

    def test_render_json_radar_data(self, sample_dashboard_data):
        template = IntensityExecutiveDashboard()
        result = template.render_json(sample_dashboard_data)
        radar = result["chart_data"]["benchmark_radar"]
        assert "labels" in radar
        assert "org_values" in radar

    def test_render_json_deterministic_provenance(self, sample_dashboard_data):
        template = IntensityExecutiveDashboard()
        r1 = template.render_json(sample_dashboard_data)
        r2 = template.render_json(sample_dashboard_data)
        assert r1["provenance_hash"] == r2["provenance_hash"]


# ---------------------------------------------------------------------------
# IntensityDetailedReport Tests
# ---------------------------------------------------------------------------


class TestDetailedReportInit:
    """Tests for IntensityDetailedReport initialisation."""

    def test_init_default(self):
        template = IntensityDetailedReport()
        assert template is not None

    def test_init_with_config(self):
        template = IntensityDetailedReport(config={"company_name": "Override"})
        assert template.config["company_name"] == "Override"


class TestDetailedReportMarkdown:
    """Tests for Markdown rendering."""

    def test_render_markdown_returns_string(self, sample_detailed_report_data):
        template = IntensityDetailedReport()
        result = template.render_markdown(sample_detailed_report_data)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_render_markdown_methodology_section(self, sample_detailed_report_data):
        template = IntensityDetailedReport()
        result = template.render_markdown(sample_detailed_report_data)
        assert "## 1. Methodology" in result
        assert "GHG Protocol" in result

    def test_render_markdown_scope_config_section(self, sample_detailed_report_data):
        template = IntensityDetailedReport()
        result = template.render_markdown(sample_detailed_report_data)
        assert "## 2. Scope Configuration" in result

    def test_render_markdown_denominator_details(self, sample_detailed_report_data):
        template = IntensityDetailedReport()
        result = template.render_markdown(sample_detailed_report_data)
        assert "## 3. Denominator Details" in result
        assert "Revenue" in result

    def test_render_markdown_intensity_by_scope(self, sample_detailed_report_data):
        template = IntensityDetailedReport()
        result = template.render_markdown(sample_detailed_report_data)
        assert "## 4. Intensity by Scope" in result
        assert "Scope 1" in result

    def test_render_markdown_intensity_by_denominator(self, sample_detailed_report_data):
        template = IntensityDetailedReport()
        result = template.render_markdown(sample_detailed_report_data)
        assert "## 5. Intensity by Denominator" in result

    def test_render_markdown_time_series(self, sample_detailed_report_data):
        template = IntensityDetailedReport()
        result = template.render_markdown(sample_detailed_report_data)
        assert "## 6. Time Series" in result
        assert "2022" in result

    def test_render_markdown_entity_breakdown(self, sample_detailed_report_data):
        template = IntensityDetailedReport()
        result = template.render_markdown(sample_detailed_report_data)
        assert "## 7. Entity Breakdown" in result
        assert "Plant A" in result

    def test_render_markdown_data_sources(self, sample_detailed_report_data):
        template = IntensityDetailedReport()
        result = template.render_markdown(sample_detailed_report_data)
        assert "## 8. Data Sources" in result
        assert "SAP ERP" in result

    def test_render_markdown_limitations(self, sample_detailed_report_data):
        template = IntensityDetailedReport()
        result = template.render_markdown(sample_detailed_report_data)
        assert "## 9. Limitations" in result
        assert "Scope 3" in result

    def test_render_markdown_provenance_hash(self, sample_detailed_report_data):
        template = IntensityDetailedReport()
        result = template.render_markdown(sample_detailed_report_data)
        assert "Provenance Hash:" in result


class TestDetailedReportHTML:
    """Tests for HTML rendering."""

    def test_render_html_contains_doctype(self, sample_detailed_report_data):
        template = IntensityDetailedReport()
        result = template.render_html(sample_detailed_report_data)
        assert "<!DOCTYPE html>" in result

    def test_render_html_title(self, sample_detailed_report_data):
        template = IntensityDetailedReport()
        result = template.render_html(sample_detailed_report_data)
        assert "Intensity Detailed Report - ACME Corp" in result

    def test_render_html_methodology_box(self, sample_detailed_report_data):
        template = IntensityDetailedReport()
        result = template.render_html(sample_detailed_report_data)
        assert "methodology-box" in result

    def test_render_html_tables(self, sample_detailed_report_data):
        template = IntensityDetailedReport()
        result = template.render_html(sample_detailed_report_data)
        assert "<table>" in result

    def test_render_html_provenance_footer(self, sample_detailed_report_data):
        template = IntensityDetailedReport()
        result = template.render_html(sample_detailed_report_data)
        assert "Provenance Hash:" in result


class TestDetailedReportJSON:
    """Tests for JSON rendering."""

    def test_render_json_returns_dict(self, sample_detailed_report_data):
        template = IntensityDetailedReport()
        result = template.render_json(sample_detailed_report_data)
        assert isinstance(result, dict)

    def test_render_json_has_template_name(self, sample_detailed_report_data):
        template = IntensityDetailedReport()
        result = template.render_json(sample_detailed_report_data)
        assert result["template"] == "intensity_detailed_report"

    def test_render_json_has_provenance_hash(self, sample_detailed_report_data):
        template = IntensityDetailedReport()
        result = template.render_json(sample_detailed_report_data)
        assert len(result["provenance_hash"]) == 64

    def test_render_json_all_sections_present(self, sample_detailed_report_data):
        template = IntensityDetailedReport()
        result = template.render_json(sample_detailed_report_data)
        expected_keys = [
            "methodology_description", "calculation_approach",
            "scope_configuration", "denominator_details",
            "intensity_by_scope", "intensity_by_denominator",
            "time_series", "entity_breakdown",
            "data_sources", "limitations",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_render_json_deterministic(self, sample_detailed_report_data):
        template = IntensityDetailedReport()
        r1 = template.render_json(sample_detailed_report_data)
        r2 = template.render_json(sample_detailed_report_data)
        assert r1["provenance_hash"] == r2["provenance_hash"]


class TestDetailedReportEmptyData:
    """Tests for empty data handling."""

    def test_render_markdown_empty_data(self):
        template = IntensityDetailedReport()
        result = template.render_markdown({})
        assert "# Intensity Metrics Detailed Report" in result
        assert "No methodology description provided" in result

    def test_render_html_empty_data(self):
        template = IntensityDetailedReport()
        result = template.render_html({})
        assert "<!DOCTYPE html>" in result

    def test_render_json_empty_data(self):
        template = IntensityDetailedReport()
        result = template.render_json({})
        assert isinstance(result, dict)
        assert result["intensity_by_scope"] == []
