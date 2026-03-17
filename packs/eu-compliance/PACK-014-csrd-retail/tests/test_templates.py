# -*- coding: utf-8 -*-
"""
PACK-014 CSRD Retail Pack - Template Tests
=============================================

Tests for the TemplateRegistry and all 8 retail report templates:
store emissions, supply chain, packaging compliance, product sustainability,
food waste, circular economy, retail ESG scorecard, and ESRS retail disclosure.

29 tests across 9 test classes.
"""

import importlib
import importlib.util
import os
import sys
import types

import pytest

# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------

PACK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(PACK_ROOT, "..", "..", "..", ".."))

# Ensure project root is on sys.path for package imports
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _load_module(name: str, subdir: str = "templates"):
    """Load a module from PACK-014 via importlib."""
    path = os.path.join(PACK_ROOT, subdir, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_template_package():
    """Load the templates package properly to support relative imports."""
    # Create the package hierarchy in sys.modules so relative imports work
    pack_parts = [
        "packs",
        "packs.eu_compliance",
        "packs.eu_compliance.PACK_014_csrd_retail",
        "packs.eu_compliance.PACK_014_csrd_retail.templates",
    ]
    dirs = [
        os.path.join(PROJECT_ROOT, "packs"),
        os.path.join(PROJECT_ROOT, "packs", "eu-compliance"),
        PACK_ROOT,
        os.path.join(PACK_ROOT, "templates"),
    ]
    for pkg_name, pkg_dir in zip(pack_parts, dirs):
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [pkg_dir]
            pkg.__package__ = pkg_name
            sys.modules[pkg_name] = pkg

    # Now import the templates __init__ which uses relative imports
    tmpl_init_path = os.path.join(PACK_ROOT, "templates", "__init__.py")
    spec = importlib.util.spec_from_file_location(
        "packs.eu_compliance.PACK_014_csrd_retail.templates",
        tmpl_init_path,
        submodule_search_locations=[os.path.join(PACK_ROOT, "templates")],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "packs.eu_compliance.PACK_014_csrd_retail.templates"
    sys.modules["packs.eu_compliance.PACK_014_csrd_retail.templates"] = mod
    spec.loader.exec_module(mod)
    return mod


# Load the templates package (handles relative imports)
templates_pkg = _load_template_package()

TemplateRegistry = templates_pkg.TemplateRegistry
TEMPLATE_CATALOG = templates_pkg.TEMPLATE_CATALOG

# Get individual template classes from the loaded package
store_report_cls = templates_pkg.StoreEmissionsReportTemplate
supply_report_cls = templates_pkg.SupplyChainReportTemplate
packaging_report_cls = templates_pkg.PackagingComplianceReportTemplate
product_report_cls = templates_pkg.ProductSustainabilityReportTemplate
food_waste_report_cls = templates_pkg.FoodWasteReportTemplate
circular_report_cls = templates_pkg.CircularEconomyReportTemplate
esg_scorecard_cls = templates_pkg.RetailESGScorecardTemplate
esrs_disclosure_cls = templates_pkg.ESRSRetailDisclosureTemplate


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

def _store_emissions_data():
    """Sample store emissions report data."""
    return {
        "report_title": "Store Emissions Report 2025",
        "reporting_year": 2025,
        "company_name": "Test Retail GmbH",
        "total_scope1_tco2e": 1250.5,
        "total_scope2_location_tco2e": 3400.2,
        "total_scope2_market_tco2e": 2100.8,
        "per_store_results": [
            {
                "store_id": "S001",
                "store_name": "Munich Main",
                "scope1_total_tco2e": 650.0,
                "scope2_location_tco2e": 1800.0,
                "total_tco2e": 2450.0,
                "emissions_per_sqm": 2.45,
            }
        ],
        "consolidated": {
            "total_emissions_tco2e": 4650.7,
            "avg_emissions_per_sqm": 2.45,
        },
        "fgas_status": {
            "R-404A": {"gwp": 4728, "charge_kg": 200, "leakage_kg": 30},
            "R-134a": {"gwp": 1530, "charge_kg": 100, "leakage_kg": 5},
        },
    }


def _supply_chain_data():
    """Sample supply chain report data."""
    return {
        "report_title": "Supply Chain Report 2025",
        "scope3_total_tco2e": 15000.0,
        "category_breakdown": [
            {"category": "cat_1", "emissions_tco2e": 8000.0},
            {"category": "cat_4", "emissions_tco2e": 3000.0},
        ],
        "hotspots": [{"supplier_name": "Supplier A", "emissions_tco2e": 2000.0}],
        "data_quality_score": 72.5,
    }


def _packaging_data():
    """Sample packaging compliance data."""
    return {
        "report_title": "Packaging Compliance 2025",
        "total_items": 150,
        "recycled_content_gaps": [
            {"material": "PET", "current_pct": 20.0, "target_pct": 30.0}
        ],
        "epr_fees_total_eur": 45000.0,
        "labeling_compliant_pct": 85.0,
    }


def _product_data():
    """Sample product sustainability data."""
    return {
        "report_title": "Product Sustainability 2025",
        "total_products": 500,
        "dpp_coverage_pct": 45.0,
        "green_claims_audit": {"total": 20, "substantiated": 15, "prohibited": 2},
        "pef_results": [{"product_id": "P001", "carbon_footprint_kgco2e": 3.5}],
    }


def _food_waste_data():
    """Sample food waste data."""
    return {
        "report_title": "Food Waste Report 2025",
        "baseline_year": 2020,
        "baseline_waste_tonnes": 500.0,
        "current_waste_tonnes": 380.0,
        "reduction_pct": 24.0,
        "eu_target_pct": 30.0,
        "category_breakdown": [
            {"category": "bakery", "waste_tonnes": 80.0},
            {"category": "produce", "waste_tonnes": 120.0},
        ],
    }


def _circular_data():
    """Sample circular economy data."""
    return {
        "report_title": "Circular Economy Report 2025",
        "mci_score": 0.42,
        "take_back_programs": [
            {"name": "Bottles", "recovery_rate_pct": 78.0}
        ],
        "epr_compliance": {"packaging": True, "weee": True},
        "waste_diversion_pct": 72.0,
    }


def _esg_scorecard_data():
    """Sample ESG scorecard data."""
    return {
        "entity_name": "Test Retail GmbH",
        "reporting_year": 2025,
        "kpis": {
            "total_emissions_tco2e": 4650.7,
            "scope1_tco2e": 1250.5,
            "scope2_tco2e": 3400.2,
            "scope3_tco2e": 15000.0,
            "emissions_intensity_per_sqm": 2.45,
            "renewable_electricity_pct": 45.0,
            "food_waste_reduction_pct": 24.0,
            "recycled_content_pct": 32.0,
            "mci_score": 0.42,
            "esrs_completeness_pct": 78.0,
        },
        "sbti_status": {"status": "Committed", "target_year": 2030, "pathway": "1.5C"},
        "percentile_rankings": {"emissions": 65, "renewable": 72},
        "regulatory_summary": {"csrd": "compliant", "ppwr": "partial"},
    }


def _esrs_disclosure_data():
    """Sample ESRS disclosure data."""
    return {
        "entity_name": "Test Retail GmbH",
        "reporting_year": 2025,
        "completeness_pct": 78.0,
        "disclosure_chapters": [
            {
                "topic": "E1",
                "chapter_title": "Climate Change",
                "completeness_pct": 85.0,
                "datapoints_used": 42,
                "content": "GHG emissions analysis...",
            },
            {
                "topic": "E5",
                "chapter_title": "Resource Use and Circular Economy",
                "completeness_pct": 70.0,
                "datapoints_used": 28,
                "content": "Circular economy metrics...",
            },
        ],
        "material_topics": ["E1", "E5", "S2", "S4"],
        "total_datapoints": 120,
        "collected_datapoints": 94,
        "audit_trail": [{"hash": "a" * 64, "step": "workflow_output"}],
    }


# ======================================================================
# 1. TestTemplateRegistry (5 tests)
# ======================================================================


class TestTemplateRegistry:
    """Tests for TemplateRegistry."""

    def test_init(self):
        registry = TemplateRegistry()
        assert registry is not None
        assert registry.template_count >= 8

    def test_list_all_10(self):
        registry = TemplateRegistry()
        templates = registry.list_templates()
        assert len(templates) == 10

    def test_get_instance(self):
        registry = TemplateRegistry()
        template = registry.get("store_emissions_report")
        assert template is not None
        assert hasattr(template, "render_markdown")
        assert hasattr(template, "render_html")
        assert hasattr(template, "render_json")

    def test_invalid_key(self):
        registry = TemplateRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent_template")

    def test_category_filter(self):
        registry = TemplateRegistry()
        emissions = registry.get_by_category("emissions")
        assert len(emissions) >= 1
        assert all(t["category"] == "emissions" for t in emissions)


# ======================================================================
# 2. TestStoreEmissionsReport (3 tests)
# ======================================================================


class TestStoreEmissionsReport:
    """Tests for StoreEmissionsReportTemplate."""

    def test_markdown(self):
        tmpl = store_report_cls()
        md = tmpl.render_markdown(_store_emissions_data())
        assert isinstance(md, str)
        assert len(md) > 100
        assert "#" in md  # Has headings

    def test_sections(self):
        tmpl = store_report_cls()
        md = tmpl.render_markdown(_store_emissions_data())
        # Should contain key section markers
        lower = md.lower()
        assert "scope" in lower or "emission" in lower or "store" in lower

    def test_html(self):
        tmpl = store_report_cls()
        html = tmpl.render_html(_store_emissions_data())
        assert isinstance(html, str)
        assert "<" in html  # Contains HTML tags


# ======================================================================
# 3. TestSupplyChainReport (3 tests)
# ======================================================================


class TestSupplyChainReport:
    """Tests for SupplyChainReportTemplate."""

    def test_markdown(self):
        tmpl = supply_report_cls()
        md = tmpl.render_markdown(_supply_chain_data())
        assert isinstance(md, str)
        assert len(md) > 50

    def test_sections(self):
        tmpl = supply_report_cls()
        md = tmpl.render_markdown(_supply_chain_data())
        lower = md.lower()
        assert "scope" in lower or "supply" in lower or "chain" in lower

    def test_json(self):
        tmpl = supply_report_cls()
        result = tmpl.render_json(_supply_chain_data())
        assert isinstance(result, dict)
        assert "template" in result or "version" in result or len(result) > 0


# ======================================================================
# 4. TestPackagingComplianceReport (3 tests)
# ======================================================================


class TestPackagingComplianceReport:
    """Tests for PackagingComplianceReportTemplate."""

    def test_markdown(self):
        tmpl = packaging_report_cls()
        md = tmpl.render_markdown(_packaging_data())
        assert isinstance(md, str)
        assert len(md) > 50

    def test_sections(self):
        tmpl = packaging_report_cls()
        md = tmpl.render_markdown(_packaging_data())
        lower = md.lower()
        assert "packaging" in lower or "compliance" in lower or "recycl" in lower

    def test_html(self):
        tmpl = packaging_report_cls()
        html = tmpl.render_html(_packaging_data())
        assert isinstance(html, str)
        assert "<" in html


# ======================================================================
# 5. TestProductSustainabilityReport (3 tests)
# ======================================================================


class TestProductSustainabilityReport:
    """Tests for ProductSustainabilityReportTemplate."""

    def test_markdown(self):
        tmpl = product_report_cls()
        md = tmpl.render_markdown(_product_data())
        assert isinstance(md, str)
        assert len(md) > 50

    def test_sections(self):
        tmpl = product_report_cls()
        md = tmpl.render_markdown(_product_data())
        lower = md.lower()
        assert "product" in lower or "dpp" in lower or "sustain" in lower

    def test_json(self):
        tmpl = product_report_cls()
        result = tmpl.render_json(_product_data())
        # render_json may return str (JSON) or dict
        assert isinstance(result, (dict, str))
        if isinstance(result, str):
            assert len(result) > 10


# ======================================================================
# 6. TestFoodWasteReport (3 tests)
# ======================================================================


class TestFoodWasteReport:
    """Tests for FoodWasteReportTemplate."""

    def test_markdown(self):
        tmpl = food_waste_report_cls()
        md = tmpl.render_markdown(_food_waste_data())
        assert isinstance(md, str)
        assert len(md) > 50

    def test_sections(self):
        tmpl = food_waste_report_cls()
        md = tmpl.render_markdown(_food_waste_data())
        lower = md.lower()
        assert "waste" in lower or "food" in lower or "reduction" in lower

    def test_html(self):
        tmpl = food_waste_report_cls()
        html = tmpl.render_html(_food_waste_data())
        assert isinstance(html, str)
        assert "<" in html


# ======================================================================
# 7. TestCircularEconomyReport (3 tests)
# ======================================================================


class TestCircularEconomyReport:
    """Tests for CircularEconomyReportTemplate."""

    def test_markdown(self):
        tmpl = circular_report_cls()
        md = tmpl.render_markdown(_circular_data())
        assert isinstance(md, str)
        assert len(md) > 50

    def test_sections(self):
        tmpl = circular_report_cls()
        md = tmpl.render_markdown(_circular_data())
        lower = md.lower()
        assert "circular" in lower or "mci" in lower or "epr" in lower

    def test_json(self):
        tmpl = circular_report_cls()
        result = tmpl.render_json(_circular_data())
        # render_json may return str (JSON) or dict
        assert isinstance(result, (dict, str))
        if isinstance(result, str):
            assert len(result) > 10


# ======================================================================
# 8. TestRetailESGScorecard (3 tests)
# ======================================================================


class TestRetailESGScorecard:
    """Tests for RetailESGScorecardTemplate."""

    def test_markdown(self):
        tmpl = esg_scorecard_cls()
        md = tmpl.render_markdown(_esg_scorecard_data())
        assert isinstance(md, str)
        assert len(md) > 50

    def test_sections(self):
        tmpl = esg_scorecard_cls()
        md = tmpl.render_markdown(_esg_scorecard_data())
        lower = md.lower()
        assert "esg" in lower or "scorecard" in lower or "kpi" in lower

    def test_html(self):
        tmpl = esg_scorecard_cls()
        html = tmpl.render_html(_esg_scorecard_data())
        assert isinstance(html, str)
        assert "<" in html


# ======================================================================
# 9. TestESRSRetailDisclosure (3 tests)
# ======================================================================


class TestESRSRetailDisclosure:
    """Tests for ESRSRetailDisclosureTemplate."""

    def test_markdown(self):
        tmpl = esrs_disclosure_cls()
        md = tmpl.render_markdown(_esrs_disclosure_data())
        assert isinstance(md, str)
        assert len(md) > 50

    def test_sections(self):
        tmpl = esrs_disclosure_cls()
        md = tmpl.render_markdown(_esrs_disclosure_data())
        lower = md.lower()
        assert "esrs" in lower or "disclosure" in lower or "e1" in lower

    def test_json(self):
        tmpl = esrs_disclosure_cls()
        result = tmpl.render_json(_esrs_disclosure_data())
        assert isinstance(result, dict)
