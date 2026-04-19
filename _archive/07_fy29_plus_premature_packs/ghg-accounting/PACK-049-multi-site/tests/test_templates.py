# -*- coding: utf-8 -*-
"""
Tests for all 10 PACK-049 report templates.

Each template: test_render, test_render_markdown, test_render_html,
test_render_json, test_export_csv, test_provenance_hash.
Plus additional template-specific assertions.
Target: ~80 tests.
"""

import json
import pytest
from decimal import Decimal
from pathlib import Path
import sys

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

# ---------------------------------------------------------------------------
# Import all 10 templates directly
# ---------------------------------------------------------------------------

try:
    from templates.site_portfolio_dashboard import SitePortfolioDashboard
    HAS_PORTFOLIO = True
except ImportError:
    HAS_PORTFOLIO = False

try:
    from templates.site_detail_report import SiteDetailReport
    HAS_DETAIL = True
except ImportError:
    HAS_DETAIL = False

try:
    from templates.consolidation_report import ConsolidationReport
    HAS_CONSOLIDATION = True
except ImportError:
    HAS_CONSOLIDATION = False

try:
    from templates.boundary_definition_report import BoundaryDefinitionReport
    HAS_BOUNDARY = True
except ImportError:
    HAS_BOUNDARY = False

try:
    from templates.regional_factor_report import RegionalFactorReport
    HAS_FACTOR = True
except ImportError:
    HAS_FACTOR = False

try:
    from templates.allocation_report import AllocationReport
    HAS_ALLOCATION = True
except ImportError:
    HAS_ALLOCATION = False

try:
    from templates.site_comparison_report import SiteComparisonReport
    HAS_COMPARISON = True
except ImportError:
    HAS_COMPARISON = False

try:
    from templates.data_collection_status_report import DataCollectionStatusReport
    HAS_COLLECTION = True
except ImportError:
    HAS_COLLECTION = False

try:
    from templates.data_quality_report import DataQualityReport
    HAS_QUALITY = True
except ImportError:
    HAS_QUALITY = False

try:
    from templates.multi_site_trend_report import MultiSiteTrendReport
    HAS_TREND = True
except ImportError:
    HAS_TREND = False


# ============================================================================
# Shared Fixtures
# ============================================================================

@pytest.fixture
def portfolio_data():
    """Data for SitePortfolioDashboard."""
    return {
        "company_name": "GreenTest GmbH",
        "reporting_period": "FY2026",
        "sites": [
            {"site_id": "s1", "site_name": "Plant A", "country": "US",
             "facility_type": "manufacturing", "scope_1_tco2e": "5000",
             "scope_2_tco2e": "3000", "scope_3_tco2e": "10000",
             "total_tco2e": "18000", "status": "approved"},
            {"site_id": "s2", "site_name": "Office B", "country": "GB",
             "facility_type": "office", "scope_1_tco2e": "100",
             "scope_2_tco2e": "350", "scope_3_tco2e": "800",
             "total_tco2e": "1250", "status": "approved"},
        ],
    }


@pytest.fixture
def detail_data():
    """Data for SiteDetailReport."""
    return {
        "company_name": "GreenTest GmbH",
        "reporting_period": "FY2026",
        "site_id": "s1",
        "site_name": "Chicago Manufacturing Plant",
        "facility_type": "manufacturing",
        "country": "US",
        "emission_sources": [
            {"source": "Natural Gas", "scope": "Scope 1",
             "tco2e": "5000", "pct_of_total": "27.8"},
            {"source": "Electricity", "scope": "Scope 2",
             "tco2e": "3000", "pct_of_total": "16.7"},
        ],
        "intensity_kpis": [
            {"kpi_name": "tCO2e per m2", "value": "0.72", "unit": "tCO2e/m2"},
        ],
        "yoy_trend": [
            {"year": 2025, "total_tco2e": "20000"},
            {"year": 2026, "total_tco2e": "18000"},
        ],
    }


@pytest.fixture
def consolidation_data():
    """Data for ConsolidationReport."""
    return {
        "company_name": "GreenTest GmbH",
        "reporting_period": "FY2026",
        "consolidation_approach": "operational_control",
        "entities": [
            {"entity_name": "Parent Corp", "scope_1": "5100",
             "scope_2": "3350", "scope_3": "10800", "total": "19250",
             "ownership_pct": "100"},
        ],
        "consolidated_scope_1": "5100",
        "consolidated_scope_2": "3350",
        "consolidated_scope_3": "10800",
        "consolidated_total": "19250",
    }


@pytest.fixture
def boundary_data():
    """Data for BoundaryDefinitionReport."""
    return {
        "company_name": "GreenTest GmbH",
        "reporting_period": "FY2026",
        "consolidation_approach": "equity_share",
        "entities": [
            {"entity_name": "Parent Corp", "ownership_pct": "100",
             "is_included": True, "entity_type": "holding"},
            {"entity_name": "Sub A GmbH", "ownership_pct": "75",
             "is_included": True, "entity_type": "subsidiary"},
        ],
        "exclusions": [],
        "materiality_threshold": "0.05",
    }


@pytest.fixture
def factor_data():
    """Data for RegionalFactorReport."""
    return {
        "company_name": "GreenTest GmbH",
        "reporting_period": "FY2026",
        "assignments": [
            {"site_name": "Plant A", "country": "US", "source_type": "electricity",
             "factor_value": "0.000417", "factor_unit": "tCO2e/kWh",
             "factor_source": "EPA eGRID", "tier": "tier_1_regional",
             "grid_region": "RFC_WEST"},
            {"site_name": "Office B", "country": "GB", "source_type": "electricity",
             "factor_value": "0.000207", "factor_unit": "tCO2e/kWh",
             "factor_source": "DEFRA", "tier": "tier_2_national",
             "grid_region": "UK_GRID"},
        ],
    }


@pytest.fixture
def allocation_data():
    """Data for AllocationReport."""
    return {
        "company_name": "GreenTest GmbH",
        "reporting_period": "FY2026",
        "allocations": [
            {"service_name": "HQ Utilities", "method": "floor_area",
             "source_emissions": "500", "site_name": "Plant A",
             "allocated_tco2e": "295", "allocation_pct": "59"},
            {"service_name": "HQ Utilities", "method": "floor_area",
             "source_emissions": "500", "site_name": "Office B",
             "allocated_tco2e": "205", "allocation_pct": "41"},
        ],
        "total_allocated": "500",
    }


@pytest.fixture
def comparison_data():
    """Data for SiteComparisonReport."""
    return {
        "company_name": "GreenTest GmbH",
        "reporting_period": "FY2026",
        "league_table": [
            {"rank": 1, "site_name": "Portland Plant", "peer_group": "Manufacturing",
             "kpi_name": "tCO2e/m2", "kpi_value": "0.44", "kpi_unit": "tCO2e/m2",
             "performance_band": "top_quartile", "percentile": "95"},
            {"rank": 2, "site_name": "Dallas Plant", "peer_group": "Manufacturing",
             "kpi_name": "tCO2e/m2", "kpi_value": "0.60", "kpi_unit": "tCO2e/m2",
             "performance_band": "second_quartile", "percentile": "70"},
        ],
        "total_reduction_potential_tco2e": "1500",
    }


@pytest.fixture
def collection_data():
    """Data for DataCollectionStatusReport."""
    return {
        "company_name": "GreenTest GmbH",
        "reporting_period": "FY2026",
        "deadline": "2027-02-28",
        "site_statuses": [
            {"site_id": "s1", "site_name": "Plant A", "status": "approved",
             "scope_1_complete": True, "scope_2_complete": True,
             "scope_3_complete": False, "entries_count": 50,
             "completeness_pct": "85", "errors": 0, "warnings": 2},
            {"site_id": "s2", "site_name": "Office B", "status": "submitted",
             "scope_1_complete": True, "scope_2_complete": True,
             "scope_3_complete": True, "entries_count": 20,
             "completeness_pct": "100", "errors": 0, "warnings": 0},
        ],
    }


@pytest.fixture
def quality_data():
    """Data for DataQualityReport."""
    return {
        "company_name": "GreenTest GmbH",
        "reporting_period": "FY2026",
        "site_quality": [
            {"site_id": "s1", "site_name": "Plant A",
             "completeness": "85", "accuracy": "78", "consistency": "80",
             "transparency": "70", "timeliness": "90", "relevance": "88"},
            {"site_id": "s2", "site_name": "Office B",
             "completeness": "95", "accuracy": "92", "consistency": "88",
             "transparency": "85", "timeliness": "95", "relevance": "90"},
        ],
    }


@pytest.fixture
def trend_data():
    """Data for MultiSiteTrendReport."""
    return {
        "company_name": "GreenTest GmbH",
        "reporting_period": "FY2020-FY2026",
        "corporate_trend": [
            {"year": 2020, "scope_1_tco2e": "8000", "scope_2_tco2e": "5000",
             "scope_3_tco2e": "15000", "total_tco2e": "28000", "site_count": 5},
            {"year": 2023, "scope_1_tco2e": "6500", "scope_2_tco2e": "4000",
             "scope_3_tco2e": "12000", "total_tco2e": "22500", "site_count": 5},
            {"year": 2026, "scope_1_tco2e": "5100", "scope_2_tco2e": "3350",
             "scope_3_tco2e": "10800", "total_tco2e": "19250", "site_count": 5},
        ],
        "improvement_leaders": [
            {"rank": 1, "site_name": "Plant A", "reduction_tco2e": "3000",
             "reduction_pct": "14.3", "primary_driver": "Energy efficiency"},
        ],
    }


# ============================================================================
# 1. SitePortfolioDashboard (7 tests)
# ============================================================================

@pytest.mark.skipif(not HAS_PORTFOLIO, reason="SitePortfolioDashboard not built")
class TestSitePortfolioDashboard:

    def test_instantiate(self):
        t = SitePortfolioDashboard()
        assert t is not None

    def test_render(self, portfolio_data):
        t = SitePortfolioDashboard()
        result = t.render(portfolio_data)
        assert result is not None
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64

    def test_render_markdown(self, portfolio_data):
        t = SitePortfolioDashboard()
        md = t.render_markdown(portfolio_data)
        assert isinstance(md, str)
        assert "GreenTest" in md

    def test_render_html(self, portfolio_data):
        t = SitePortfolioDashboard()
        html = t.render_html(portfolio_data)
        assert "<" in html
        assert "html" in html.lower()

    def test_render_json(self, portfolio_data):
        t = SitePortfolioDashboard()
        j = t.render_json(portfolio_data)
        assert isinstance(j, dict)

    def test_export_csv(self, portfolio_data):
        t = SitePortfolioDashboard()
        result = t.render(portfolio_data)
        csv = t.export_csv(result)
        assert isinstance(csv, str)
        assert "," in csv

    def test_provenance_deterministic(self, portfolio_data):
        t = SitePortfolioDashboard()
        r1 = t.render(portfolio_data)
        r2 = t.render(portfolio_data)
        assert r1.provenance_hash == r2.provenance_hash


# ============================================================================
# 2. SiteDetailReport (7 tests)
# ============================================================================

@pytest.mark.skipif(not HAS_DETAIL, reason="SiteDetailReport not built")
class TestSiteDetailReport:

    def test_instantiate(self):
        assert SiteDetailReport() is not None

    def test_render(self, detail_data):
        result = SiteDetailReport().render(detail_data)
        assert result is not None
        assert len(result.provenance_hash) == 64

    def test_render_markdown(self, detail_data):
        md = SiteDetailReport().render_markdown(detail_data)
        assert "Chicago" in md

    def test_render_html(self, detail_data):
        html = SiteDetailReport().render_html(detail_data)
        assert "<" in html

    def test_render_json(self, detail_data):
        j = SiteDetailReport().render_json(detail_data)
        assert isinstance(j, dict)

    def test_export_csv(self, detail_data):
        t = SiteDetailReport()
        result = t.render(detail_data)
        csv = t.export_csv(result)
        assert isinstance(csv, str)

    def test_provenance_hash(self, detail_data):
        r = SiteDetailReport().render(detail_data)
        assert len(r.provenance_hash) == 64


# ============================================================================
# 3. ConsolidationReport (7 tests)
# ============================================================================

@pytest.mark.skipif(not HAS_CONSOLIDATION, reason="ConsolidationReport not built")
class TestConsolidationReport:

    def test_instantiate(self):
        assert ConsolidationReport() is not None

    def test_render(self, consolidation_data):
        r = ConsolidationReport().render(consolidation_data)
        assert r is not None
        assert len(r.provenance_hash) == 64

    def test_render_markdown(self, consolidation_data):
        md = ConsolidationReport().render_markdown(consolidation_data)
        assert "Consolidation" in md or "GreenTest" in md

    def test_render_html(self, consolidation_data):
        html = ConsolidationReport().render_html(consolidation_data)
        assert "<" in html

    def test_render_json(self, consolidation_data):
        j = ConsolidationReport().render_json(consolidation_data)
        assert isinstance(j, dict)

    def test_export_csv(self, consolidation_data):
        t = ConsolidationReport()
        r = t.render(consolidation_data)
        csv = t.export_csv(r)
        assert isinstance(csv, str)

    def test_provenance_deterministic(self, consolidation_data):
        t = ConsolidationReport()
        r1 = t.render(consolidation_data)
        r2 = t.render(consolidation_data)
        assert r1.provenance_hash == r2.provenance_hash


# ============================================================================
# 4. BoundaryDefinitionReport (6 tests)
# ============================================================================

@pytest.mark.skipif(not HAS_BOUNDARY, reason="BoundaryDefinitionReport not built")
class TestBoundaryDefinitionReport:

    def test_instantiate(self):
        assert BoundaryDefinitionReport() is not None

    def test_render(self, boundary_data):
        r = BoundaryDefinitionReport().render(boundary_data)
        assert len(r.provenance_hash) == 64

    def test_render_markdown(self, boundary_data):
        md = BoundaryDefinitionReport().render_markdown(boundary_data)
        assert "Boundary" in md or "GreenTest" in md

    def test_render_html(self, boundary_data):
        html = BoundaryDefinitionReport().render_html(boundary_data)
        assert "<" in html

    def test_render_json(self, boundary_data):
        j = BoundaryDefinitionReport().render_json(boundary_data)
        assert isinstance(j, dict)

    def test_export_csv(self, boundary_data):
        t = BoundaryDefinitionReport()
        r = t.render(boundary_data)
        csv = t.export_csv(r)
        assert isinstance(csv, str)


# ============================================================================
# 5. RegionalFactorReport (6 tests)
# ============================================================================

@pytest.mark.skipif(not HAS_FACTOR, reason="RegionalFactorReport not built")
class TestRegionalFactorReport:

    def test_instantiate(self):
        assert RegionalFactorReport() is not None

    def test_render(self, factor_data):
        r = RegionalFactorReport().render(factor_data)
        assert len(r.provenance_hash) == 64

    def test_render_markdown(self, factor_data):
        md = RegionalFactorReport().render_markdown(factor_data)
        assert "Factor" in md or "GreenTest" in md

    def test_render_html(self, factor_data):
        html = RegionalFactorReport().render_html(factor_data)
        assert "<" in html

    def test_render_json(self, factor_data):
        j = RegionalFactorReport().render_json(factor_data)
        assert isinstance(j, dict)

    def test_export_csv(self, factor_data):
        t = RegionalFactorReport()
        r = t.render(factor_data)
        csv = t.export_csv(r)
        assert isinstance(csv, str)


# ============================================================================
# 6. AllocationReport (6 tests)
# ============================================================================

@pytest.mark.skipif(not HAS_ALLOCATION, reason="AllocationReport not built")
class TestAllocationReport:

    def test_instantiate(self):
        assert AllocationReport() is not None

    def test_render(self, allocation_data):
        r = AllocationReport().render(allocation_data)
        assert len(r.provenance_hash) == 64

    def test_render_markdown(self, allocation_data):
        md = AllocationReport().render_markdown(allocation_data)
        assert "Allocation" in md or "GreenTest" in md

    def test_render_html(self, allocation_data):
        html = AllocationReport().render_html(allocation_data)
        assert "<" in html

    def test_render_json(self, allocation_data):
        j = AllocationReport().render_json(allocation_data)
        assert isinstance(j, dict)

    def test_export_csv(self, allocation_data):
        t = AllocationReport()
        r = t.render(allocation_data)
        csv = t.export_csv(r)
        assert isinstance(csv, str)


# ============================================================================
# 7. SiteComparisonReport (7 tests)
# ============================================================================

@pytest.mark.skipif(not HAS_COMPARISON, reason="SiteComparisonReport not built")
class TestSiteComparisonReport:

    def test_instantiate(self):
        assert SiteComparisonReport() is not None

    def test_render(self, comparison_data):
        r = SiteComparisonReport().render(comparison_data)
        assert len(r.provenance_hash) == 64

    def test_render_markdown(self, comparison_data):
        md = SiteComparisonReport().render_markdown(comparison_data)
        assert "Comparison" in md or "GreenTest" in md

    def test_render_html(self, comparison_data):
        html = SiteComparisonReport().render_html(comparison_data)
        assert "<" in html

    def test_render_json(self, comparison_data):
        j = SiteComparisonReport().render_json(comparison_data)
        assert isinstance(j, dict)

    def test_export_csv(self, comparison_data):
        t = SiteComparisonReport()
        r = t.render(comparison_data)
        csv = t.export_csv(r)
        assert isinstance(csv, str)
        assert "Portland" in csv or "rank" in csv

    def test_quartile_distribution(self, comparison_data):
        r = SiteComparisonReport().render(comparison_data)
        assert isinstance(r.distribution_quartiles, dict)


# ============================================================================
# 8. DataCollectionStatusReport (7 tests)
# ============================================================================

@pytest.mark.skipif(not HAS_COLLECTION, reason="DataCollectionStatusReport not built")
class TestDataCollectionStatusReport:

    def test_instantiate(self):
        assert DataCollectionStatusReport() is not None

    def test_render(self, collection_data):
        r = DataCollectionStatusReport().render(collection_data)
        assert len(r.provenance_hash) == 64

    def test_total_sites(self, collection_data):
        r = DataCollectionStatusReport().render(collection_data)
        assert r.total_sites == 2

    def test_approved_count(self, collection_data):
        r = DataCollectionStatusReport().render(collection_data)
        assert r.approved_count == 1

    def test_render_markdown(self, collection_data):
        md = DataCollectionStatusReport().render_markdown(collection_data)
        assert "Collection" in md or "Status" in md

    def test_render_html(self, collection_data):
        html = DataCollectionStatusReport().render_html(collection_data)
        assert "<" in html

    def test_export_csv(self, collection_data):
        t = DataCollectionStatusReport()
        r = t.render(collection_data)
        csv = t.export_csv(r)
        assert "Plant A" in csv


# ============================================================================
# 9. DataQualityReport (8 tests)
# ============================================================================

@pytest.mark.skipif(not HAS_QUALITY, reason="DataQualityReport not built")
class TestDataQualityReport:

    def test_instantiate(self):
        assert DataQualityReport() is not None

    def test_render(self, quality_data):
        r = DataQualityReport().render(quality_data)
        assert len(r.provenance_hash) == 64

    def test_heatmap_generated(self, quality_data):
        r = DataQualityReport().render(quality_data)
        assert len(r.heatmap) > 0
        # 2 sites * 6 dimensions = 12 cells
        assert len(r.heatmap) == 12

    def test_corporate_score_calculated(self, quality_data):
        r = DataQualityReport().render(quality_data)
        assert r.corporate_score > Decimal("0")

    def test_dimension_summaries(self, quality_data):
        r = DataQualityReport().render(quality_data)
        assert len(r.dimension_summaries) == 6

    def test_render_markdown(self, quality_data):
        md = DataQualityReport().render_markdown(quality_data)
        assert "Quality" in md

    def test_render_html(self, quality_data):
        html = DataQualityReport().render_html(quality_data)
        assert "<" in html

    def test_export_csv(self, quality_data):
        t = DataQualityReport()
        r = t.render(quality_data)
        csv = t.export_csv(r)
        assert "Plant A" in csv


# ============================================================================
# 10. MultiSiteTrendReport (8 tests)
# ============================================================================

@pytest.mark.skipif(not HAS_TREND, reason="MultiSiteTrendReport not built")
class TestMultiSiteTrendReport:

    def test_instantiate(self):
        assert MultiSiteTrendReport() is not None

    def test_render(self, trend_data):
        r = MultiSiteTrendReport().render(trend_data)
        assert len(r.provenance_hash) == 64

    def test_yoy_calculated(self, trend_data):
        r = MultiSiteTrendReport().render(trend_data)
        assert len(r.corporate_trend) == 3
        # First year has no YoY, second and third should
        assert r.corporate_trend[0].yoy_change_pct is None
        assert r.corporate_trend[1].yoy_change_pct is not None

    def test_cagr_calculated(self, trend_data):
        r = MultiSiteTrendReport().render(trend_data)
        assert r.cagr_pct is not None
        # Emissions decreased from 28000 to 19250 over 6 years = negative CAGR
        assert r.cagr_pct < Decimal("0")

    def test_latest_total(self, trend_data):
        r = MultiSiteTrendReport().render(trend_data)
        assert r.latest_total_tco2e == Decimal("19250")

    def test_render_markdown(self, trend_data):
        md = MultiSiteTrendReport().render_markdown(trend_data)
        assert "Trend" in md

    def test_render_html(self, trend_data):
        html = MultiSiteTrendReport().render_html(trend_data)
        assert "<" in html

    def test_export_csv(self, trend_data):
        t = MultiSiteTrendReport()
        r = t.render(trend_data)
        csv = t.export_csv(r)
        assert "2020" in csv
        assert "2026" in csv


# ============================================================================
# Template Registry Tests (5 tests)
# ============================================================================

class TestTemplateRegistry:

    def test_registry_init(self):
        from templates import TemplateRegistry
        registry = TemplateRegistry()
        assert registry is not None

    def test_list_template_names(self):
        from templates import TemplateRegistry
        registry = TemplateRegistry()
        names = registry.list_template_names()
        assert isinstance(names, list)
        assert len(names) == 10

    def test_template_count(self):
        from templates import TemplateRegistry
        registry = TemplateRegistry()
        assert registry.template_count == 10

    def test_get_template_by_name(self):
        from templates import TemplateRegistry
        registry = TemplateRegistry()
        if registry.has_template("site_portfolio_dashboard"):
            t = registry.get("site_portfolio_dashboard")
            assert t is not None

    def test_get_by_category(self):
        from templates import TemplateRegistry
        registry = TemplateRegistry()
        exec_templates = registry.get_by_category("executive")
        assert isinstance(exec_templates, list)
