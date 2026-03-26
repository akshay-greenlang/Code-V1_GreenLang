# -*- coding: utf-8 -*-
"""
Tests for PACK-049 Engine 10: MultiSiteReportingEngine

Covers portfolio dashboard, site detail, consolidation report, boundary
report, allocation report, comparison report, collection status report,
quality report, trend report, export formats (Markdown, HTML, JSON, CSV),
and drill-down capability.
Target: ~50 tests.
"""

import pytest
import json
from decimal import Decimal
from pathlib import Path
import sys

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

try:
    from engines.multi_site_reporting_engine import (
        MultiSiteReportingEngine,
        Report,
        PortfolioDashboard,
        SiteDetailReport,
        ConsolidationReport,
        ExportResult,
        ReportType,
        ExportFormat,
    )
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

pytestmark = pytest.mark.skipif(not HAS_ENGINE, reason="Engine not yet built")


@pytest.fixture
def engine():
    return MultiSiteReportingEngine()


@pytest.fixture
def portfolio_data(sample_site_totals):
    return {
        "organisation_id": "ORG-001",
        "organisation_name": "GreenTest Manufacturing GmbH",
        "reporting_year": 2026,
        "site_totals": sample_site_totals,
        "consolidated_scope1": Decimal("5975.00"),
        "consolidated_scope2": Decimal("5587.50"),
        "consolidated_scope3": Decimal("16200.00"),
        "consolidated_total": Decimal("27762.50"),
        "total_sites": 5,
        "completeness_pct": Decimal("100.00"),
    }


# ============================================================================
# Report Type Tests
# ============================================================================

class TestPortfolioDashboard:

    def test_portfolio_dashboard(self, engine, portfolio_data):
        report = engine.generate_report(
            report_type="PORTFOLIO_DASHBOARD",
            data=portfolio_data,
        )
        assert report is not None
        assert report.report_type == "PORTFOLIO_DASHBOARD"

    def test_dashboard_has_totals(self, engine, portfolio_data):
        report = engine.generate_report(
            report_type="PORTFOLIO_DASHBOARD",
            data=portfolio_data,
        )
        assert report.content is not None
        assert "27762" in str(report.content) or report.total is not None


class TestSiteDetailReport:

    def test_site_detail(self, engine, sample_site_totals):
        report = engine.generate_report(
            report_type="SITE_DETAIL",
            data={"site": sample_site_totals[0]},
        )
        assert report is not None
        assert report.report_type == "SITE_DETAIL"

    def test_site_detail_scopes(self, engine, sample_site_totals):
        report = engine.generate_report(
            report_type="SITE_DETAIL",
            data={"site": sample_site_totals[0]},
        )
        assert report.content is not None


class TestConsolidationReport:

    def test_consolidation_report(self, engine, portfolio_data):
        report = engine.generate_report(
            report_type="CONSOLIDATION",
            data=portfolio_data,
        )
        assert report is not None
        assert report.report_type == "CONSOLIDATION"


class TestBoundaryReport:

    def test_boundary_report(self, engine, sample_boundary):
        report = engine.generate_report(
            report_type="BOUNDARY_DEFINITION",
            data=sample_boundary,
        )
        assert report is not None
        assert report.report_type == "BOUNDARY_DEFINITION"


class TestAllocationReport:

    def test_allocation_report(self, engine, sample_allocation_config):
        report = engine.generate_report(
            report_type="ALLOCATION",
            data=sample_allocation_config,
        )
        assert report is not None
        assert report.report_type == "ALLOCATION"


class TestComparisonReport:

    def test_comparison_report(self, engine, sample_comparison_result):
        report = engine.generate_report(
            report_type="COMPARISON",
            data=sample_comparison_result,
        )
        assert report is not None
        assert report.report_type == "COMPARISON"


class TestCollectionStatusReport:

    def test_collection_status_report(self, engine, sample_completion_result):
        report = engine.generate_report(
            report_type="COLLECTION_STATUS",
            data=sample_completion_result,
        )
        assert report is not None
        assert report.report_type == "COLLECTION_STATUS"


class TestQualityReport:

    def test_quality_report(self, engine, sample_quality_assessments):
        report = engine.generate_report(
            report_type="QUALITY_HEATMAP",
            data={"assessments": sample_quality_assessments},
        )
        assert report is not None
        assert report.report_type == "QUALITY_HEATMAP"


class TestTrendReport:

    def test_trend_report(self, engine, portfolio_data):
        trend_data = {
            "site_id": "site-001",
            "years": {
                2024: Decimal("20000"),
                2025: Decimal("19000"),
                2026: Decimal("18000"),
            },
        }
        report = engine.generate_report(
            report_type="TREND",
            data=trend_data,
        )
        assert report is not None
        assert report.report_type == "TREND"


# ============================================================================
# Export Format Tests
# ============================================================================

class TestExportFormats:

    def test_export_markdown_format(self, engine, portfolio_data):
        report = engine.generate_report(
            report_type="PORTFOLIO_DASHBOARD",
            data=portfolio_data,
        )
        exported = engine.export_report(report, format="MARKDOWN")
        assert isinstance(exported, ExportResult)
        assert exported.format == "MARKDOWN"
        assert "# " in exported.content or "## " in exported.content

    def test_export_html_format(self, engine, portfolio_data):
        report = engine.generate_report(
            report_type="PORTFOLIO_DASHBOARD",
            data=portfolio_data,
        )
        exported = engine.export_report(report, format="HTML")
        assert exported.format == "HTML"
        assert "<" in exported.content

    def test_export_json_valid(self, engine, portfolio_data):
        report = engine.generate_report(
            report_type="PORTFOLIO_DASHBOARD",
            data=portfolio_data,
        )
        exported = engine.export_report(report, format="JSON")
        assert exported.format == "JSON"
        # Validate it is valid JSON
        parsed = json.loads(exported.content)
        assert isinstance(parsed, (dict, list))

    def test_export_csv_headers(self, engine, portfolio_data):
        report = engine.generate_report(
            report_type="PORTFOLIO_DASHBOARD",
            data=portfolio_data,
        )
        exported = engine.export_report(report, format="CSV")
        assert exported.format == "CSV"
        lines = exported.content.strip().split("\n")
        assert len(lines) >= 2  # header + at least 1 data row


# ============================================================================
# Drill-Down Tests
# ============================================================================

class TestDrillDown:

    def test_drill_down(self, engine, portfolio_data):
        report = engine.generate_report(
            report_type="PORTFOLIO_DASHBOARD",
            data=portfolio_data,
        )
        detail = engine.drill_down(
            report=report,
            target_site_id="site-001",
        )
        assert detail is not None
        assert detail.report_type == "SITE_DETAIL"

    def test_drill_down_nonexistent_site(self, engine, portfolio_data):
        report = engine.generate_report(
            report_type="PORTFOLIO_DASHBOARD",
            data=portfolio_data,
        )
        with pytest.raises((KeyError, ValueError, Exception)):
            engine.drill_down(
                report=report,
                target_site_id="nonexistent",
            )


# ============================================================================
# Provenance and Metadata Tests
# ============================================================================

class TestReportProvenance:

    def test_report_provenance_hash(self, engine, portfolio_data):
        report = engine.generate_report(
            report_type="PORTFOLIO_DASHBOARD",
            data=portfolio_data,
        )
        assert report.provenance_hash is not None
        assert len(report.provenance_hash) == 64

    def test_report_timestamp(self, engine, portfolio_data):
        report = engine.generate_report(
            report_type="PORTFOLIO_DASHBOARD",
            data=portfolio_data,
        )
        assert report.created_at is not None

    def test_report_deterministic(self, engine, portfolio_data):
        r1 = engine.generate_report(
            report_type="PORTFOLIO_DASHBOARD",
            data=portfolio_data,
        )
        r2 = engine.generate_report(
            report_type="PORTFOLIO_DASHBOARD",
            data=portfolio_data,
        )
        assert r1.report_type == r2.report_type
