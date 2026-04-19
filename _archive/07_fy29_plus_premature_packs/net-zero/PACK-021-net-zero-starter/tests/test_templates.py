# -*- coding: utf-8 -*-
"""
Unit tests for PACK-021 Net Zero Starter Pack Templates.

Tests the TemplateRegistry, all 8 templates' markdown/html/json rendering,
template categories, discovery, and convenience methods.

Author:  GL-TestEngineer
Pack:    PACK-021 Net Zero Starter
"""

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from templates import (
    BenchmarkComparisonReportTemplate,
    GHGBaselineReportTemplate,
    NetZeroScorecardReportTemplate,
    NetZeroStrategyReportTemplate,
    OffsetPortfolioReportTemplate,
    ProgressDashboardReportTemplate,
    ReductionRoadmapReportTemplate,
    TargetValidationReportTemplate,
    TEMPLATE_CATALOG,
    TemplateRegistry,
)


# ========================================================================
# Expected template names
# ========================================================================

EXPECTED_TEMPLATE_NAMES = [
    "net_zero_strategy_report",
    "ghg_baseline_report",
    "target_validation_report",
    "reduction_roadmap_report",
    "offset_portfolio_report",
    "net_zero_scorecard_report",
    "progress_dashboard_report",
    "benchmark_comparison_report",
]

EXPECTED_CATEGORIES = [
    "strategy",
    "baseline",
    "targets",
    "reduction",
    "offsets",
    "scorecard",
    "progress",
    "benchmark",
]


# ========================================================================
# Minimal sample data for rendering
# ========================================================================


def _sample_data() -> Dict[str, Any]:
    """Generate minimal sample data that templates can render.

    Each template expects different data keys, but they should
    degrade gracefully with missing keys.
    """
    return {
        "entity_name": "TestCo",
        "report_date": "2026-03-18",
        "assessment_year": 2026,
        "base_year": 2020,
        "target_year": 2050,
        "sector": "manufacturing",
        "total_emissions_tco2e": 20000,
        "scope1_tco2e": 5000,
        "scope2_tco2e": 3000,
        "scope3_tco2e": 12000,
        "reduction_target_pct": 90,
        "overall_score": 55,
        "maturity_level": "Developing",
        "dimensions": [],
        "recommendations": [],
        "credits": [],
        "kpis": {"items": []},
        "actions": [],
        "milestones": [],
        "phases": [],
        "provenance_hash": "a" * 64,
    }


# ========================================================================
# Fixtures
# ========================================================================


@pytest.fixture
def registry() -> TemplateRegistry:
    """Fresh TemplateRegistry."""
    return TemplateRegistry()


@pytest.fixture
def sample_data() -> Dict[str, Any]:
    """Sample rendering data."""
    return _sample_data()


# ========================================================================
# Template Registry
# ========================================================================


class TestTemplateRegistryLoads:
    """Tests for TemplateRegistry initialization."""

    def test_template_registry_loads(self):
        """Registry creates without error."""
        reg = TemplateRegistry()
        assert reg is not None

    def test_template_count_8(self, registry):
        """Registry has exactly 8 templates."""
        assert registry.template_count == 8

    def test_list_template_names(self, registry):
        """list_template_names returns all 8 names."""
        names = registry.list_template_names()
        assert len(names) == 8
        for name in EXPECTED_TEMPLATE_NAMES:
            assert name in names

    def test_list_templates_returns_metadata(self, registry):
        """list_templates returns dicts with name, description, category."""
        templates = registry.list_templates()
        assert len(templates) == 8
        for t in templates:
            assert "name" in t
            assert "description" in t
            assert "category" in t
            assert "formats" in t
            assert "version" in t


# ========================================================================
# All Templates Registered
# ========================================================================


class TestAllTemplatesRegistered:
    """Validate every expected template is in the registry."""

    @pytest.mark.parametrize("name", EXPECTED_TEMPLATE_NAMES)
    def test_template_registered(self, registry, name):
        """Template '{name}' is registered."""
        assert registry.has_template(name)

    @pytest.mark.parametrize("name", EXPECTED_TEMPLATE_NAMES)
    def test_template_get_succeeds(self, registry, name):
        """Template '{name}' can be retrieved."""
        template = registry.get(name)
        assert template is not None

    def test_get_nonexistent_raises_key_error(self, registry):
        """Getting a nonexistent template raises KeyError."""
        with pytest.raises(KeyError):
            registry.get("nonexistent_template")


# ========================================================================
# Template Class Imports
# ========================================================================


class TestTemplateClassImports:
    """Validate that all 8 template classes imported successfully."""

    @pytest.mark.parametrize(
        "cls",
        [
            NetZeroStrategyReportTemplate,
            GHGBaselineReportTemplate,
            TargetValidationReportTemplate,
            ReductionRoadmapReportTemplate,
            OffsetPortfolioReportTemplate,
            NetZeroScorecardReportTemplate,
            ProgressDashboardReportTemplate,
            BenchmarkComparisonReportTemplate,
        ],
        ids=[
            "strategy",
            "baseline",
            "target_validation",
            "reduction_roadmap",
            "offset_portfolio",
            "scorecard",
            "progress_dashboard",
            "benchmark_comparison",
        ],
    )
    def test_template_class_not_none(self, cls):
        """Template class imported successfully (not None)."""
        assert cls is not None


# ========================================================================
# Markdown Rendering
# ========================================================================


class TestMarkdownRendering:
    """Tests for render_markdown on all 8 templates."""

    @pytest.mark.parametrize("name", EXPECTED_TEMPLATE_NAMES)
    def test_markdown_render(self, registry, sample_data, name):
        """Template '{name}' renders markdown as a string."""
        template = registry.get(name)
        md = template.render_markdown(sample_data)
        assert isinstance(md, str)
        assert len(md) > 0


# ========================================================================
# HTML Rendering
# ========================================================================


class TestHTMLRendering:
    """Tests for render_html on all 8 templates."""

    @pytest.mark.parametrize("name", EXPECTED_TEMPLATE_NAMES)
    def test_html_render(self, registry, sample_data, name):
        """Template '{name}' renders HTML as a string."""
        template = registry.get(name)
        html = template.render_html(sample_data)
        assert isinstance(html, str)
        assert len(html) > 0


# ========================================================================
# JSON Rendering
# ========================================================================


class TestJSONRendering:
    """Tests for render_json on all 8 templates."""

    @pytest.mark.parametrize("name", EXPECTED_TEMPLATE_NAMES)
    def test_json_render(self, registry, sample_data, name):
        """Template '{name}' renders JSON as a dict or string."""
        template = registry.get(name)
        json_out = template.render_json(sample_data)
        assert json_out is not None
        # JSON output is typically a dict or a JSON string
        assert isinstance(json_out, (dict, str))


# ========================================================================
# Template Categories
# ========================================================================


class TestTemplateCategories:
    """Tests for template category filtering."""

    @pytest.mark.parametrize("category", EXPECTED_CATEGORIES)
    def test_category_has_templates(self, registry, category):
        """Category '{category}' returns at least 1 template."""
        templates = registry.get_by_category(category)
        assert len(templates) >= 1

    def test_all_categories_covered(self, registry):
        """All 8 expected categories have templates."""
        for cat in EXPECTED_CATEGORIES:
            result = registry.get_by_category(cat)
            assert len(result) >= 1

    def test_category_filter_returns_correct_category(self, registry):
        """Filtered templates have matching category."""
        for cat in EXPECTED_CATEGORIES:
            for t in registry.get_by_category(cat):
                assert t["category"] == cat


# ========================================================================
# has_template Method
# ========================================================================


class TestHasTemplateMethod:
    """Tests for has_template check."""

    def test_has_template_returns_true(self, registry):
        """has_template returns True for known templates."""
        assert registry.has_template("net_zero_strategy_report") is True

    def test_has_template_returns_false(self, registry):
        """has_template returns False for unknown templates."""
        assert registry.has_template("nonexistent") is False


# ========================================================================
# Render Convenience Method
# ========================================================================


class TestRenderConvenienceMethod:
    """Tests for the registry.render() shortcut."""

    @pytest.mark.parametrize(
        "format_str",
        ["markdown", "html", "json"],
    )
    def test_render_method_format(self, registry, sample_data, format_str):
        """registry.render() supports markdown/html/json formats."""
        result = registry.render(
            "net_zero_strategy_report", sample_data, format=format_str
        )
        assert result is not None

    def test_render_invalid_format_raises(self, registry, sample_data):
        """registry.render() raises ValueError for unsupported format."""
        with pytest.raises(ValueError):
            registry.render("net_zero_strategy_report", sample_data, format="xml")

    def test_render_invalid_template_raises(self, registry, sample_data):
        """registry.render() raises KeyError for unknown template."""
        with pytest.raises(KeyError):
            registry.render("nonexistent", sample_data)


# ========================================================================
# Template Info
# ========================================================================


class TestTemplateInfo:
    """Tests for get_info metadata retrieval."""

    @pytest.mark.parametrize("name", EXPECTED_TEMPLATE_NAMES)
    def test_get_info_returns_metadata(self, registry, name):
        """get_info returns dict with name, description, category, class_name."""
        info = registry.get_info(name)
        assert info["name"] == name
        assert info["description"]
        assert info["category"]
        assert info["class_name"]
        assert info["version"] == "21.0.0"

    def test_get_info_nonexistent_raises(self, registry):
        """get_info raises KeyError for nonexistent template."""
        with pytest.raises(KeyError):
            registry.get_info("nonexistent")


# ========================================================================
# TEMPLATE_CATALOG
# ========================================================================


class TestTemplateCatalog:
    """Tests for the TEMPLATE_CATALOG constant."""

    def test_catalog_has_8_entries(self):
        """TEMPLATE_CATALOG has exactly 8 entries."""
        assert len(TEMPLATE_CATALOG) == 8

    def test_catalog_entries_have_required_keys(self):
        """Each catalog entry has name, class, description, category, formats."""
        for entry in TEMPLATE_CATALOG:
            assert "name" in entry
            assert "class" in entry
            assert "description" in entry
            assert "category" in entry
            assert "formats" in entry
            assert "version" in entry

    def test_catalog_all_classes_non_none(self):
        """All template classes in catalog imported successfully."""
        for entry in TEMPLATE_CATALOG:
            assert entry["class"] is not None, f"Template {entry['name']} class is None"

    def test_catalog_formats_include_all_three(self):
        """Each catalog entry supports markdown, html, json."""
        for entry in TEMPLATE_CATALOG:
            assert "markdown" in entry["formats"]
            assert "html" in entry["formats"]
            assert "json" in entry["formats"]


# ========================================================================
# Registry Repr
# ========================================================================


class TestRegistryRepr:
    """Tests for string representation."""

    def test_repr_includes_count(self, registry):
        """repr shows template count."""
        rep = repr(registry)
        assert "8" in rep
        assert "TemplateRegistry" in rep
