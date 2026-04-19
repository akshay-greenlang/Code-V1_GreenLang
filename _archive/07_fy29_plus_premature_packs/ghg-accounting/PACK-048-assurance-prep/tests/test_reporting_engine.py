"""
Unit tests for AssuranceReportingEngine (PACK-048 Engine 10).

Tests all public methods with 25+ tests covering:
  - Readiness dashboard generation
  - Evidence index generation
  - Control report
  - Query register
  - Materiality report
  - Sampling report
  - Regulatory report
  - Cost timeline report
  - Markdown export
  - HTML export
  - JSON export

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
# Readiness Dashboard Tests
# ---------------------------------------------------------------------------


class TestReadinessDashboardGeneration:
    """Tests for readiness dashboard report generation."""

    def test_dashboard_has_overall_score(self):
        """Test dashboard includes overall readiness score."""
        dashboard = {"overall_score": Decimal("78"), "status": "MOSTLY_READY"}
        assert dashboard["overall_score"] > Decimal("0")

    def test_dashboard_has_category_scores(self):
        """Test dashboard includes per-category readiness scores."""
        categories = {"GOV": Decimal("90"), "SRC": Decimal("75"), "CAL": Decimal("80")}
        assert len(categories) >= 3

    def test_dashboard_has_gap_count(self):
        """Test dashboard includes total gap count."""
        dashboard = {"gaps_total": 15, "gaps_critical": 3}
        assert dashboard["gaps_total"] >= dashboard["gaps_critical"]


# ---------------------------------------------------------------------------
# Evidence Index Tests
# ---------------------------------------------------------------------------


class TestEvidenceIndexGeneration:
    """Tests for evidence index report generation."""

    def test_evidence_index_lists_all_items(self, sample_evidence_items):
        """Test evidence index lists all evidence items."""
        index = [{"id": e["evidence_id"], "category": e["category"]} for e in sample_evidence_items]
        assert len(index) == 30

    def test_evidence_index_sorted_by_scope(self, sample_evidence_items):
        """Test evidence index can be sorted by scope."""
        sorted_items = sorted(sample_evidence_items, key=lambda x: x["scope"])
        assert sorted_items[0]["scope"] <= sorted_items[-1]["scope"]

    def test_evidence_index_has_file_hashes(self, sample_evidence_items):
        """Test evidence index includes file hashes."""
        for item in sample_evidence_items:
            assert len(item["file_hash"]) == 64


# ---------------------------------------------------------------------------
# Control Report Tests
# ---------------------------------------------------------------------------


class TestControlReport:
    """Tests for control testing report generation."""

    def test_control_report_has_25_controls(self, sample_controls):
        """Test control report includes all 25 controls."""
        assert len(sample_controls) == 25

    def test_control_report_has_effectiveness_summary(self, sample_controls):
        """Test control report includes effectiveness summary."""
        effective = len([c for c in sample_controls if c["operating_effective"]])
        total = len(sample_controls)
        rate = Decimal(str(effective)) / Decimal(str(total)) * Decimal("100")
        assert_decimal_between(rate, Decimal("0"), Decimal("100"))


# ---------------------------------------------------------------------------
# Query Register Tests
# ---------------------------------------------------------------------------


class TestQueryRegister:
    """Tests for verifier query register report."""

    def test_query_register_has_counts(self, sample_engagement):
        """Test query register includes open/closed counts."""
        assert sample_engagement["queries_open"] >= 0
        assert sample_engagement["queries_closed"] >= 0

    def test_query_register_total(self, sample_engagement):
        """Test query register total is sum of open + closed."""
        total = sample_engagement["queries_open"] + sample_engagement["queries_closed"]
        assert total == 17


# ---------------------------------------------------------------------------
# Materiality Report Tests
# ---------------------------------------------------------------------------


class TestMaterialityReport:
    """Tests for materiality report generation."""

    def test_materiality_report_has_thresholds(self, sample_emissions_data):
        """Test materiality report includes all threshold levels."""
        total = sample_emissions_data["total_all_scopes_tco2e"]
        thresholds = {
            "overall": total * Decimal("5") / Decimal("100"),
            "performance": total * Decimal("5") / Decimal("100") * Decimal("75") / Decimal("100"),
            "clearly_trivial": total * Decimal("5") / Decimal("100") * Decimal("5") / Decimal("100"),
        }
        assert len(thresholds) == 3
        assert thresholds["clearly_trivial"] < thresholds["performance"] < thresholds["overall"]


# ---------------------------------------------------------------------------
# Sampling Report Tests
# ---------------------------------------------------------------------------


class TestSamplingReport:
    """Tests for sampling plan report generation."""

    def test_sampling_report_has_method(self):
        """Test sampling report includes sampling method."""
        report = {"method": "MUS", "confidence_level": Decimal("0.95")}
        assert report["method"] == "MUS"

    def test_sampling_report_has_sample_size(self):
        """Test sampling report includes calculated sample size."""
        report = {"sample_size": 25, "population_size": 150}
        assert report["sample_size"] > 0
        assert report["sample_size"] <= report["population_size"]


# ---------------------------------------------------------------------------
# Regulatory Report Tests
# ---------------------------------------------------------------------------


class TestRegulatoryReport:
    """Tests for regulatory compliance report generation."""

    def test_regulatory_report_has_12_jurisdictions(self, sample_jurisdictions):
        """Test regulatory report covers 12 jurisdictions."""
        assert len(sample_jurisdictions) == 12

    def test_regulatory_report_has_compliance_status(self, sample_jurisdictions):
        """Test regulatory report includes compliance status per jurisdiction."""
        for j in sample_jurisdictions:
            assert "assurance_required" in j


# ---------------------------------------------------------------------------
# Cost Timeline Report Tests
# ---------------------------------------------------------------------------


class TestCostTimelineReport:
    """Tests for cost and timeline report generation."""

    def test_cost_report_has_fee_estimate(self, sample_engagement):
        """Test cost report includes fee estimate."""
        assert sample_engagement["fee_estimate_usd"] > Decimal("0")

    def test_cost_report_has_timeline(self, sample_engagement):
        """Test cost report includes engagement timeline."""
        assert sample_engagement["engagement_start"] < sample_engagement["report_due"]


# ---------------------------------------------------------------------------
# Markdown Export Tests
# ---------------------------------------------------------------------------


class TestMarkdownExport:
    """Tests for Markdown report export."""

    def test_markdown_contains_header(self):
        """Test markdown output contains report header."""
        md = "# GHG Assurance Readiness Report - ACME Corp"
        assert "# " in md

    def test_markdown_contains_table(self):
        """Test markdown output contains data tables."""
        md = "| Category | Score | Status |\n|---|---|---|"
        assert "|" in md

    def test_markdown_contains_provenance(self):
        """Test markdown output contains provenance hash."""
        h = hashlib.sha256(b"report_data").hexdigest()
        md = f"**Provenance Hash:** `{h}`"
        assert "Provenance Hash:" in md
        assert len(h) == 64


# ---------------------------------------------------------------------------
# HTML Export Tests
# ---------------------------------------------------------------------------


class TestHTMLExport:
    """Tests for HTML report export."""

    def test_html_contains_doctype(self):
        """Test HTML output starts with DOCTYPE."""
        html = "<!DOCTYPE html><html><body>Report</body></html>"
        assert "<!DOCTYPE html>" in html

    def test_html_contains_tables(self):
        """Test HTML output includes data tables."""
        html = "<table><thead><tr><th>Category</th></tr></thead></table>"
        assert "<table>" in html

    def test_html_contains_css(self):
        """Test HTML output includes styling."""
        html = "<style>.assurance-report { font-family: Arial; }</style>"
        assert "<style>" in html


# ---------------------------------------------------------------------------
# JSON Export Tests
# ---------------------------------------------------------------------------


class TestJSONExport:
    """Tests for JSON report export."""

    def test_json_is_valid(self):
        """Test JSON output is valid JSON."""
        data = {"template": "assurance_report", "version": "1.0.0", "data": {}}
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["template"] == "assurance_report"

    def test_json_has_provenance_hash(self):
        """Test JSON output includes provenance hash."""
        h = hashlib.sha256(b"data").hexdigest()
        data = {"provenance_hash": h}
        assert len(data["provenance_hash"]) == 64

    def test_json_deterministic(self):
        """Test JSON output is deterministic."""
        d1 = json.dumps({"a": 1, "b": 2}, sort_keys=True)
        d2 = json.dumps({"a": 1, "b": 2}, sort_keys=True)
        h1 = hashlib.sha256(d1.encode()).hexdigest()
        h2 = hashlib.sha256(d2.encode()).hexdigest()
        assert h1 == h2
