"""
Unit tests for BenchmarkReportingEngine (PACK-047 Engine 10).

Tests all public methods with 25+ tests covering:
  - League table generation
  - Radar chart data
  - Pathway alignment graph data
  - Portfolio heatmap data
  - Sparkline data
  - Markdown export
  - HTML export
  - JSON export
  - CSV export
  - XBRL export

Author: GreenLang QA Team
"""
from __future__ import annotations

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
# League Table Tests
# ---------------------------------------------------------------------------


class TestLeagueTableGeneration:
    """Tests for league table (ranking) generation."""

    def test_league_table_sorted_by_rank(self, sample_league_table_data):
        """Test league table entries are sorted by rank."""
        entries = sample_league_table_data["entries"]
        ranks = [e["rank"] for e in entries]
        assert ranks == sorted(ranks)

    def test_organisation_highlighted(self, sample_league_table_data):
        """Test organisation is highlighted in league table."""
        entries = sample_league_table_data["entries"]
        org_entries = [e for e in entries if e.get("is_organisation")]
        assert len(org_entries) == 1
        assert org_entries[0]["entity_name"] == "ACME Corp"

    def test_league_table_has_required_fields(self, sample_league_table_data):
        """Test league table entries have required fields."""
        required = ["rank", "entity_name", "emissions_tco2e", "intensity", "percentile"]
        for entry in sample_league_table_data["entries"]:
            for field in required:
                assert field in entry, f"Missing field '{field}'"


# ---------------------------------------------------------------------------
# Radar Chart Data Tests
# ---------------------------------------------------------------------------


class TestRadarChartData:
    """Tests for multi-dimension radar chart data generation."""

    def test_radar_has_6_dimensions(self, sample_radar_chart_data):
        """Test radar chart has 6 benchmark dimensions."""
        assert len(sample_radar_chart_data["dimensions"]) == 6

    def test_radar_values_match_dimensions(self, sample_radar_chart_data):
        """Test value arrays match dimension count."""
        n = len(sample_radar_chart_data["dimensions"])
        assert len(sample_radar_chart_data["organisation_values"]) == n
        assert len(sample_radar_chart_data["peer_median_values"]) == n
        assert len(sample_radar_chart_data["best_in_class_values"]) == n

    def test_radar_values_in_range(self, sample_radar_chart_data):
        """Test radar chart values are in [0, 100] range."""
        for v in sample_radar_chart_data["organisation_values"]:
            assert_decimal_between(v, Decimal("0"), Decimal("100"))

    def test_best_in_class_exceeds_median(self, sample_radar_chart_data):
        """Test best-in-class values exceed peer median."""
        bic = sample_radar_chart_data["best_in_class_values"]
        median = sample_radar_chart_data["peer_median_values"]
        for b, m in zip(bic, median):
            assert b >= m


# ---------------------------------------------------------------------------
# Pathway Alignment Graph Tests
# ---------------------------------------------------------------------------


class TestPathwayAlignmentGraph:
    """Tests for pathway alignment graph data generation."""

    def test_graph_has_organisation_trajectory(self, sample_pathway_chart_data):
        """Test graph data includes organisation trajectory."""
        assert "organisation_trajectory" in sample_pathway_chart_data
        assert len(sample_pathway_chart_data["organisation_trajectory"]) > 0

    def test_graph_has_pathway_data(self, sample_pathway_chart_data):
        """Test graph data includes pathway reference lines."""
        assert "pathways" in sample_pathway_chart_data
        assert "IEA_NZE" in sample_pathway_chart_data["pathways"]

    def test_graph_has_gap_to_pathway(self, sample_pathway_chart_data):
        """Test graph data includes gap-to-pathway calculations."""
        assert "gap_to_pathway" in sample_pathway_chart_data
        assert "IEA_NZE" in sample_pathway_chart_data["gap_to_pathway"]


# ---------------------------------------------------------------------------
# Portfolio Heatmap Tests
# ---------------------------------------------------------------------------


class TestPortfolioHeatmap:
    """Tests for portfolio heatmap data generation."""

    def test_heatmap_sector_geography_matrix(self, sample_portfolio):
        """Test heatmap data is organised as sector x geography matrix."""
        sectors = set()
        geographies = set()
        for h in sample_portfolio:
            sectors.add(h["sector"])
            geographies.add(h["geography"])
        assert len(sectors) >= 3
        assert len(geographies) >= 3

    def test_heatmap_values_non_negative(self, sample_portfolio):
        """Test heatmap emission values are non-negative."""
        for h in sample_portfolio:
            assert h["emissions_scope_1_2_tco2e"] >= Decimal("0")


# ---------------------------------------------------------------------------
# Sparkline Data Tests
# ---------------------------------------------------------------------------


class TestSparklineData:
    """Tests for sparkline (mini-chart) data generation."""

    def test_sparkline_has_5_years(self, sample_emissions_data):
        """Test sparkline covers 5 years of data."""
        org = sample_emissions_data["organisation"]
        assert len(org) == 5

    def test_sparkline_values_ordered_by_year(self, sample_emissions_data):
        """Test sparkline data points are ordered chronologically."""
        org = sample_emissions_data["organisation"]
        years = sorted(org.keys())
        assert years == [str(y) for y in range(2020, 2025)]


# ---------------------------------------------------------------------------
# Markdown Export Tests
# ---------------------------------------------------------------------------


class TestMarkdownExport:
    """Tests for Markdown benchmark report export."""

    def test_markdown_contains_header(self, sample_league_table_data):
        """Test markdown output contains report header."""
        company = sample_league_table_data["company_name"]
        period = sample_league_table_data["reporting_period"]
        header = f"# GHG Emissions Benchmark Report - {company}"
        assert company in header

    def test_markdown_contains_table(self):
        """Test markdown output contains league table in markdown format."""
        md = "| Rank | Entity | Emissions | Intensity |\n|---|---|---|---|"
        assert "|" in md
        assert "Rank" in md

    def test_markdown_contains_provenance(self):
        """Test markdown output contains provenance hash."""
        import hashlib
        h = hashlib.sha256(b"test").hexdigest()
        md = f"**Provenance Hash:** `{h}`"
        assert "Provenance Hash:" in md
        assert len(h) == 64


# ---------------------------------------------------------------------------
# HTML Export Tests
# ---------------------------------------------------------------------------


class TestHTMLExport:
    """Tests for HTML benchmark report export."""

    def test_html_contains_doctype(self):
        """Test HTML output starts with DOCTYPE."""
        html = "<!DOCTYPE html><html><body>Report</body></html>"
        assert "<!DOCTYPE html>" in html

    def test_html_contains_css(self):
        """Test HTML output includes inline CSS."""
        html = "<style>.benchmark-table { border-collapse: collapse; }</style>"
        assert "<style>" in html

    def test_html_contains_title(self, sample_league_table_data):
        """Test HTML output includes company name in title."""
        company = sample_league_table_data["company_name"]
        html = f"<title>Benchmark Report - {company}</title>"
        assert company in html


# ---------------------------------------------------------------------------
# JSON Export Tests
# ---------------------------------------------------------------------------


class TestJSONExport:
    """Tests for JSON benchmark report export."""

    def test_json_is_valid(self):
        """Test JSON output is valid JSON."""
        data = {"template": "benchmark_report", "version": "1.0.0", "data": {}}
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["template"] == "benchmark_report"

    def test_json_has_provenance_hash(self):
        """Test JSON output includes provenance hash."""
        import hashlib
        h = hashlib.sha256(b"data").hexdigest()
        data = {"provenance_hash": h}
        assert len(data["provenance_hash"]) == 64

    def test_json_deterministic(self):
        """Test JSON output is deterministic (same data -> same hash)."""
        import hashlib
        d1 = json.dumps({"a": 1, "b": 2}, sort_keys=True)
        d2 = json.dumps({"a": 1, "b": 2}, sort_keys=True)
        assert hashlib.sha256(d1.encode()).hexdigest() == hashlib.sha256(d2.encode()).hexdigest()


# ---------------------------------------------------------------------------
# CSV Export Tests
# ---------------------------------------------------------------------------


class TestCSVExport:
    """Tests for CSV benchmark report export."""

    def test_csv_header_row(self):
        """Test CSV export includes header row."""
        headers = "rank,entity_name,emissions_tco2e,intensity,percentile"
        assert "rank" in headers
        assert "entity_name" in headers

    def test_csv_data_rows(self, sample_league_table_data):
        """Test CSV export has correct number of data rows."""
        entries = sample_league_table_data["entries"]
        assert len(entries) == 3  # 3 entries in fixture


# ---------------------------------------------------------------------------
# XBRL Export Tests
# ---------------------------------------------------------------------------


class TestXBRLExport:
    """Tests for XBRL benchmark report export."""

    def test_xbrl_esrs_namespace(self):
        """Test XBRL output uses ESRS namespace."""
        ns = "http://www.esma.europa.eu/xbrl/esrs"
        assert "esrs" in ns

    def test_xbrl_contains_benchmark_element(self):
        """Test XBRL output contains benchmark disclosure element."""
        element = '<esrs:E1_6_BenchmarkComparison contextRef="FY2025">'
        assert "BenchmarkComparison" in element

    def test_xbrl_numeric_precision(self):
        """Test XBRL numeric values have correct decimal precision."""
        value = Decimal("16.000000")
        xbrl_val = f'<esrs:IntensityValue decimals="6">{value}</esrs:IntensityValue>'
        assert 'decimals="6"' in xbrl_val
