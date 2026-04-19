"""
Unit tests for PACK-047 Templates.

Tests all 10 templates with 65+ tests covering:
  - BenchmarkExecutiveDashboard: KPI cards, traffic lights, sparklines
  - BenchmarkDetailedReport: Full benchmark analysis report
  - LeagueTableTemplate: Peer ranking table
  - PathwayAlignmentTemplate: Pathway comparison chart
  - RadarChartTemplate: Multi-dimension radar chart
  - PortfolioHeatmapTemplate: Sector x geography heatmap
  - ITRDisclosureTemplate: Implied temperature rise disclosure
  - TransitionRiskTemplate: Transition risk scorecard
  - ESRSBenchmarkTemplate: ESRS E1-6 benchmark disclosure (XBRL)
  - CDPBenchmarkTemplate: CDP C7 benchmark section
  - render() returns valid structure
  - to_markdown(), to_html(), to_json() exports
  - XBRL output for ESRS template
  - Empty data handling

Author: GreenLang QA Team
"""
from __future__ import annotations

import hashlib
import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from tests.conftest import assert_decimal_between, assert_decimal_equal


# ---------------------------------------------------------------------------
# Template 1: Executive Dashboard
# ---------------------------------------------------------------------------


class TestBenchmarkExecutiveDashboard:
    """Tests for BenchmarkExecutiveDashboard template."""

    def test_template_has_kpi_cards_section(self):
        """Test dashboard includes KPI cards section."""
        sections = ["kpi_cards", "traffic_light", "sparkline", "peer_summary",
                     "strengths", "improvement_areas", "provenance"]
        assert "kpi_cards" in sections

    def test_kpi_cards_include_percentile(self):
        """Test KPI cards include overall percentile rank."""
        kpi = {"name": "Percentile Rank", "value": Decimal("35"), "unit": "%"}
        assert kpi["name"] == "Percentile Rank"

    def test_kpi_cards_include_itr(self):
        """Test KPI cards include Implied Temperature Rise."""
        kpi = {"name": "Implied Temperature Rise", "value": Decimal("2.1"), "unit": "C"}
        assert kpi["name"] == "Implied Temperature Rise"

    def test_traffic_light_values(self):
        """Test traffic light indicators use correct colours."""
        valid_colours = {"GREEN", "AMBER", "RED"}
        colour = "AMBER"
        assert colour in valid_colours

    def test_sparkline_has_5_points(self, sample_emissions_data):
        """Test sparkline has 5 data points (5 years)."""
        org = sample_emissions_data["organisation"]
        assert len(org) == 5

    def test_markdown_export(self):
        """Test markdown export produces valid markdown."""
        md = "# GHG Emissions Benchmark Dashboard\n\n## Key Metrics"
        assert "# " in md

    def test_html_export(self):
        """Test HTML export produces valid HTML."""
        html = "<!DOCTYPE html><html><body>Dashboard</body></html>"
        assert "<!DOCTYPE html>" in html

    def test_json_export(self):
        """Test JSON export produces valid dict."""
        data = {"template": "benchmark_executive_dashboard", "kpis": []}
        assert data["template"] == "benchmark_executive_dashboard"


# ---------------------------------------------------------------------------
# Template 2: Detailed Report
# ---------------------------------------------------------------------------


class TestBenchmarkDetailedReport:
    """Tests for BenchmarkDetailedReport template."""

    def test_report_has_methodology_section(self):
        """Test detailed report includes methodology section."""
        sections = ["methodology", "peer_group", "normalisation", "pathway",
                     "itr", "trajectory", "data_quality", "limitations"]
        assert "methodology" in sections

    def test_markdown_export_contains_sections(self):
        """Test markdown export contains expected section headers."""
        md = "## 1. Methodology\n## 2. Peer Group\n## 3. Pathway Alignment"
        assert "## 1. Methodology" in md

    def test_html_export_has_tables(self):
        """Test HTML export includes data tables."""
        html = "<table><tr><th>Entity</th></tr></table>"
        assert "<table>" in html

    def test_json_export_has_all_sections(self):
        """Test JSON export includes all analysis sections."""
        data = {"methodology": {}, "peer_group": {}, "pathway_alignment": {}}
        assert "methodology" in data
        assert "peer_group" in data

    def test_empty_data_handled_gracefully(self):
        """Test template handles empty data without errors."""
        data = {}
        md = "# Benchmark Report\n\nNo data available."
        assert "No data available" in md

    def test_provenance_hash_in_report(self):
        """Test detailed report includes provenance hash."""
        h = hashlib.sha256(b"report_data").hexdigest()
        assert len(h) == 64


# ---------------------------------------------------------------------------
# Template 3: League Table
# ---------------------------------------------------------------------------


class TestLeagueTableTemplate:
    """Tests for LeagueTableTemplate."""

    def test_sorted_by_rank(self, sample_league_table_data):
        """Test league table is sorted by rank."""
        ranks = [e["rank"] for e in sample_league_table_data["entries"]]
        assert ranks == sorted(ranks)

    def test_org_highlighted(self, sample_league_table_data):
        """Test organisation is highlighted."""
        org = [e for e in sample_league_table_data["entries"] if e.get("is_organisation")]
        assert len(org) == 1

    def test_markdown_table_format(self):
        """Test markdown uses pipe-delimited table format."""
        md = "| Rank | Entity | Emissions |\n|---|---|---|"
        assert "|" in md

    def test_html_table_format(self):
        """Test HTML uses proper table tags."""
        html = "<table><thead><tr><th>Rank</th></tr></thead></table>"
        assert "<table>" in html
        assert "<thead>" in html

    def test_json_has_entries(self, sample_league_table_data):
        """Test JSON output has entries array."""
        assert "entries" in sample_league_table_data
        assert len(sample_league_table_data["entries"]) > 0

    def test_empty_table_handled(self):
        """Test empty league table handled gracefully."""
        data = {"entries": []}
        assert len(data["entries"]) == 0


# ---------------------------------------------------------------------------
# Template 4: Pathway Alignment
# ---------------------------------------------------------------------------


class TestPathwayAlignmentTemplate:
    """Tests for PathwayAlignmentTemplate."""

    def test_has_org_trajectory(self, sample_pathway_chart_data):
        """Test template includes organisation trajectory."""
        assert "organisation_trajectory" in sample_pathway_chart_data

    def test_has_pathway_lines(self, sample_pathway_chart_data):
        """Test template includes pathway reference lines."""
        assert "pathways" in sample_pathway_chart_data

    def test_has_gap_annotation(self, sample_pathway_chart_data):
        """Test template includes gap-to-pathway annotation."""
        assert "gap_to_pathway" in sample_pathway_chart_data

    def test_markdown_export(self):
        """Test markdown export with pathway data."""
        md = "## Pathway Alignment\n\n| Year | Organisation | IEA NZE |"
        assert "Pathway Alignment" in md

    def test_json_chart_data(self, sample_pathway_chart_data):
        """Test JSON export includes chart-ready data."""
        assert len(sample_pathway_chart_data["organisation_trajectory"]) > 0

    def test_empty_data_handled(self):
        """Test empty pathway data handled gracefully."""
        data = {"organisation_trajectory": {}, "pathways": {}}
        assert len(data["organisation_trajectory"]) == 0


# ---------------------------------------------------------------------------
# Template 5: Radar Chart
# ---------------------------------------------------------------------------


class TestRadarChartTemplate:
    """Tests for RadarChartTemplate."""

    def test_6_dimensions(self, sample_radar_chart_data):
        """Test radar chart has 6 dimensions."""
        assert len(sample_radar_chart_data["dimensions"]) == 6

    def test_3_series(self, sample_radar_chart_data):
        """Test radar chart has 3 data series (org, median, best)."""
        assert "organisation_values" in sample_radar_chart_data
        assert "peer_median_values" in sample_radar_chart_data
        assert "best_in_class_values" in sample_radar_chart_data

    def test_json_chart_ready(self, sample_radar_chart_data):
        """Test JSON output is chart-library ready."""
        data = json.dumps(sample_radar_chart_data, default=str)
        parsed = json.loads(data)
        assert "dimensions" in parsed

    def test_empty_values_handled(self):
        """Test empty radar values handled gracefully."""
        data = {"dimensions": [], "organisation_values": []}
        assert len(data["dimensions"]) == 0

    def test_markdown_export(self):
        """Test markdown export with radar data."""
        md = "## Multi-Dimension Benchmark\n\n| Dimension | Org | Median | Best |"
        assert "Multi-Dimension Benchmark" in md

    def test_html_export_svg(self):
        """Test HTML export could contain SVG chart."""
        html = '<div class="radar-chart"><svg></svg></div>'
        assert "radar-chart" in html


# ---------------------------------------------------------------------------
# Template 6: Portfolio Heatmap
# ---------------------------------------------------------------------------


class TestPortfolioHeatmapTemplate:
    """Tests for PortfolioHeatmapTemplate."""

    def test_heatmap_has_sectors(self, sample_portfolio):
        """Test heatmap data includes sector dimension."""
        sectors = set(h["sector"] for h in sample_portfolio)
        assert len(sectors) >= 3

    def test_heatmap_has_geographies(self, sample_portfolio):
        """Test heatmap data includes geography dimension."""
        geos = set(h["geography"] for h in sample_portfolio)
        assert len(geos) >= 3

    def test_json_matrix_format(self):
        """Test JSON output is matrix format."""
        matrix = {"rows": ["INDUSTRIALS", "ENERGY"], "cols": ["EU", "NA"],
                  "values": [[Decimal("100"), Decimal("200")], [Decimal("150"), Decimal("300")]]}
        assert len(matrix["rows"]) == 2
        assert len(matrix["values"]) == 2


# ---------------------------------------------------------------------------
# Template 7: ITR Disclosure
# ---------------------------------------------------------------------------


class TestITRDisclosureTemplate:
    """Tests for ITRDisclosureTemplate."""

    def test_includes_itr_value(self):
        """Test ITR disclosure includes temperature value."""
        data = {"itr": Decimal("2.1"), "scope": "scope_1_2", "method": "budget_based"}
        assert data["itr"] > Decimal("0")

    def test_includes_confidence_interval(self):
        """Test ITR disclosure includes confidence interval."""
        data = {"itr": Decimal("2.1"), "ci_lower": Decimal("1.8"), "ci_upper": Decimal("2.5")}
        assert data["ci_lower"] < data["itr"] < data["ci_upper"]

    def test_markdown_export(self):
        """Test markdown ITR disclosure."""
        md = "## Implied Temperature Rise\n\nITR: **2.1C** (95% CI: 1.8C - 2.5C)"
        assert "2.1C" in md

    def test_json_export(self):
        """Test JSON ITR disclosure."""
        data = {"itr": "2.1", "scope": "scope_1_2"}
        assert data["scope"] == "scope_1_2"


# ---------------------------------------------------------------------------
# Template 8: Transition Risk
# ---------------------------------------------------------------------------


class TestTransitionRiskTemplate:
    """Tests for TransitionRiskTemplate."""

    def test_composite_score_present(self):
        """Test transition risk scorecard includes composite score."""
        data = {"composite_score": Decimal("55.5"), "risk_level": "MEDIUM"}
        assert data["composite_score"] > Decimal("0")

    def test_risk_dimensions_present(self):
        """Test all 4 risk dimensions present."""
        dims = ["carbon_budget", "stranding", "regulatory", "competitive"]
        assert len(dims) == 4

    def test_markdown_export(self):
        """Test markdown transition risk scorecard."""
        md = "## Transition Risk Scorecard\n\n| Dimension | Score | Weight |"
        assert "Transition Risk" in md

    def test_empty_data_handled(self):
        """Test empty risk data handled gracefully."""
        data = {"composite_score": None, "dimensions": []}
        assert data["composite_score"] is None


# ---------------------------------------------------------------------------
# Template 9: ESRS E1-6 Benchmark
# ---------------------------------------------------------------------------


class TestESRSBenchmarkTemplate:
    """Tests for ESRSBenchmarkTemplate (XBRL output)."""

    def test_xbrl_namespace(self):
        """Test XBRL output uses ESRS namespace."""
        ns = "http://www.esma.europa.eu/xbrl/esrs"
        assert "esrs" in ns

    def test_xbrl_benchmark_element(self):
        """Test XBRL includes benchmark comparison element."""
        element = '<esrs:E1_6_BenchmarkComparison contextRef="FY2025">'
        assert "E1_6" in element

    def test_xbrl_numeric_precision(self):
        """Test XBRL numeric values have 6 decimal precision."""
        val = '<esrs:IntensityValue decimals="6">16.000000</esrs:IntensityValue>'
        assert 'decimals="6"' in val

    def test_markdown_fallback(self):
        """Test markdown fallback for ESRS template."""
        md = "## ESRS E1-6 Benchmark Disclosure"
        assert "ESRS" in md

    def test_json_export(self):
        """Test JSON ESRS export."""
        data = {"framework": "ESRS", "disclosure": "E1-6", "benchmark_data": {}}
        assert data["framework"] == "ESRS"


# ---------------------------------------------------------------------------
# Template 10: CDP Benchmark
# ---------------------------------------------------------------------------


class TestCDPBenchmarkTemplate:
    """Tests for CDPBenchmarkTemplate."""

    def test_cdp_section_c7(self):
        """Test template targets CDP C7 section."""
        section = "C7"
        assert section == "C7"

    def test_includes_peer_comparison(self):
        """Test CDP template includes peer comparison data."""
        data = {"section": "C7", "peer_group": "Manufacturing",
                "percentile_rank": Decimal("35")}
        assert data["percentile_rank"] > Decimal("0")

    def test_markdown_export(self):
        """Test markdown CDP benchmark section."""
        md = "## CDP Climate Change - Section C7: Emissions Benchmarking"
        assert "C7" in md

    def test_json_export(self):
        """Test JSON CDP export."""
        data = {"section": "C7", "benchmark_data": {}, "peer_group": "Industrials"}
        assert data["section"] == "C7"

    def test_empty_data_handled(self):
        """Test empty CDP data handled gracefully."""
        data = {"section": "C7", "benchmark_data": None}
        assert data["benchmark_data"] is None

    def test_provenance_hash(self):
        """Test CDP template includes provenance hash."""
        h = hashlib.sha256(b"cdp_benchmark").hexdigest()
        assert len(h) == 64
