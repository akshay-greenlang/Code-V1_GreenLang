# -*- coding: utf-8 -*-
"""
Test suite for PACK-028 Sector Pathway Pack - Templates.

Tests 8 report templates, TemplateRegistry, rendering, and TEMPLATE_CATALOG.

Author:  GreenLang Test Engineering
Pack:    PACK-028 Sector Pathway Pack
"""

import sys
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from templates import (
    SectorPathwayReportTemplate,
    IntensityConvergenceReportTemplate,
    TechnologyRoadmapReportTemplate,
    AbatementWaterfallReportTemplate,
    SectorBenchmarkReportTemplate,
    ScenarioComparisonReportTemplate,
    SBTiValidationReportTemplate,
    SectorStrategyReportTemplate,
    TemplateRegistry,
    TEMPLATE_CATALOG,
)


# ===========================================================================
# Template Registry
# ===========================================================================


class TestTemplateRegistry:
    """Test TemplateRegistry."""

    def test_registry_instantiates(self):
        reg = TemplateRegistry()
        assert reg is not None

    def test_registry_has_8_templates(self):
        reg = TemplateRegistry()
        assert reg.template_count == 8

    def test_registry_list_names(self):
        reg = TemplateRegistry()
        names = reg.list_template_names()
        assert len(names) == 8
        assert "sector_pathway_report" in names
        assert "intensity_convergence_report" in names

    def test_registry_categories(self):
        reg = TemplateRegistry()
        cats = reg.categories
        assert len(cats) >= 5

    def test_registry_has_template(self):
        reg = TemplateRegistry()
        assert reg.has_template("sector_pathway_report") is True
        assert reg.has_template("nonexistent_template") is False

    def test_registry_get_template(self):
        reg = TemplateRegistry()
        t = reg.get_template("sector_pathway_report")
        assert t is not None
        assert isinstance(t, SectorPathwayReportTemplate)

    def test_registry_get_info(self):
        reg = TemplateRegistry()
        info = reg.get_info("sector_pathway_report")
        assert info is not None
        assert "name" in info
        assert "description" in info
        assert "category" in info

    def test_registry_search(self):
        reg = TemplateRegistry()
        results = reg.search("benchmark")
        assert len(results) > 0
        assert results[0]["name"] == "sector_benchmark_report"

    def test_registry_get_by_category(self):
        reg = TemplateRegistry()
        pathway_templates = reg.get_by_category("pathway")
        assert len(pathway_templates) > 0


# ===========================================================================
# Template Rendering
# ===========================================================================


class TestTemplateRendering:
    """Test template rendering."""

    TEMPLATE_NAMES = [
        "sector_pathway_report",
        "intensity_convergence_report",
        "technology_roadmap_report",
        "abatement_waterfall_report",
        "sector_benchmark_report",
        "scenario_comparison_report",
        "sbti_validation_report",
        "sector_strategy_report",
    ]

    @pytest.mark.parametrize("name", TEMPLATE_NAMES)
    def test_template_renders_markdown(self, name):
        reg = TemplateRegistry()
        result = reg.render(name, {"entity_name": "TestCo", "sector": "steel"})
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.parametrize("name", TEMPLATE_NAMES)
    def test_template_has_render_methods(self, name):
        reg = TemplateRegistry()
        t = reg.get_template(name)
        assert hasattr(t, "render_markdown")
        assert hasattr(t, "render_html")
        assert hasattr(t, "render_json")

    def test_render_with_data(self):
        reg = TemplateRegistry()
        result = reg.render("sector_pathway_report", {
            "entity_name": "SteelCo",
            "sector": "steel",
            "base_year": 2020,
            "target_year": 2050,
        })
        assert isinstance(result, str)
        assert len(result) > 100

    def test_render_nonexistent_template(self):
        reg = TemplateRegistry()
        with pytest.raises(KeyError):
            reg.render("nonexistent", {})


# ===========================================================================
# Individual Template Classes
# ===========================================================================


class TestIndividualTemplates:
    """Test individual template class instantiation."""

    def test_sector_pathway_template(self):
        t = SectorPathwayReportTemplate()
        assert t is not None

    def test_intensity_convergence_template(self):
        t = IntensityConvergenceReportTemplate()
        assert t is not None

    def test_technology_roadmap_template(self):
        t = TechnologyRoadmapReportTemplate()
        assert t is not None

    def test_abatement_waterfall_template(self):
        t = AbatementWaterfallReportTemplate()
        assert t is not None

    def test_sector_benchmark_template(self):
        t = SectorBenchmarkReportTemplate()
        assert t is not None

    def test_scenario_comparison_template(self):
        t = ScenarioComparisonReportTemplate()
        assert t is not None

    def test_sbti_validation_template(self):
        t = SBTiValidationReportTemplate()
        assert t is not None

    def test_sector_strategy_template(self):
        t = SectorStrategyReportTemplate()
        assert t is not None


# ===========================================================================
# Template Catalog
# ===========================================================================


class TestTemplateCatalog:
    """Test TEMPLATE_CATALOG constant."""

    def test_catalog_is_list(self):
        assert isinstance(TEMPLATE_CATALOG, list)

    def test_catalog_has_8_entries(self):
        assert len(TEMPLATE_CATALOG) == 8

    def test_catalog_entries_have_info(self):
        for entry in TEMPLATE_CATALOG:
            assert isinstance(entry, dict)
            assert "name" in entry
            assert "description" in entry
            assert "category" in entry

    def test_catalog_names_unique(self):
        names = [e["name"] for e in TEMPLATE_CATALOG]
        assert len(names) == len(set(names))

    def test_catalog_all_names_present(self):
        names = {e["name"] for e in TEMPLATE_CATALOG}
        expected = {
            "sector_pathway_report",
            "intensity_convergence_report",
            "technology_roadmap_report",
            "abatement_waterfall_report",
            "sector_benchmark_report",
            "scenario_comparison_report",
            "sbti_validation_report",
            "sector_strategy_report",
        }
        assert expected == names


# ===========================================================================
# Template Content Validation
# ===========================================================================


class TestTemplateContentValidation:
    """Validate template output content."""

    def test_pathway_report_has_sections(self):
        reg = TemplateRegistry()
        md = reg.render("sector_pathway_report", {
            "entity_name": "TestCo", "sector": "power_generation"})
        assert len(md) > 100

    def test_benchmark_report_has_sections(self):
        reg = TemplateRegistry()
        md = reg.render("sector_benchmark_report", {
            "entity_name": "TestCo", "sector": "steel"})
        assert len(md) > 100

    @pytest.mark.parametrize("name", [
        "sector_pathway_report", "intensity_convergence_report",
        "technology_roadmap_report", "sector_benchmark_report",
    ])
    def test_template_render_non_empty(self, name):
        reg = TemplateRegistry()
        md = reg.render(name, {"entity_name": "Co", "sector": "cement"})
        assert md is not None
        assert len(md) > 50
