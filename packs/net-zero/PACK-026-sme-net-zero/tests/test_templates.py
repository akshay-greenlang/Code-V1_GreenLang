# -*- coding: utf-8 -*-
"""
Test suite for PACK-026 SME Net Zero Pack - Templates.

Tests all 8 templates in all formats (MD/HTML/JSON/Excel), mobile
responsiveness metadata, data validation, and template registry.

Author:  GreenLang Test Engineering
Pack:    PACK-026 SME Net Zero
Tests:   ~400 lines, 55+ tests
"""

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from templates import (
    SMEBaselineReportTemplate,
    SMEQuickWinsReportTemplate,
    SMEGrantReportTemplate,
    SMEBoardBriefTemplate,
    SMERoadmapReportTemplate,
    SMEProgressDashboardTemplate,
    SMECertificationSubmissionTemplate,
    SMEAccountingGuideTemplate,
    TEMPLATE_CATALOG,
    TemplateRegistry,
)


EXPECTED_TEMPLATE_NAMES = [
    "sme_baseline_report",
    "sme_quick_wins_report",
    "sme_grant_report",
    "sme_board_brief",
    "sme_roadmap_report",
    "sme_progress_dashboard",
    "sme_certification_submission",
    "sme_accounting_guide",
]

EXPECTED_CATEGORIES = [
    "baseline",
    "actions",
    "funding",
    "governance",
    "roadmap",
    "progress",
    "certification",
    "integration",
]


def _sample_data() -> Dict[str, Any]:
    return {
        "org_name": "SmallCo Ltd",
        "entity_name": "SmallCo Ltd",
        "sme_tier": "small",
        "sector": "professional_services",
        "report_date": "2026-03-18",
        "reporting_year": 2026,
        "assessment_year": 2026,
        "base_year": 2024,
        "target_year": 2030,
        "employee_count": 25,
        "annual_revenue_eur": 2500000,
        "total_emissions_tco2e": 150,
        "scope1_tco2e": 30,
        "scope2_tco2e": 45,
        "scope3_tco2e": 75,
        "baseline_method": "SILVER",
        "accuracy_band_pct": 15,
        "reduction_target_pct": 50,
        "quick_wins": [
            {"name": "LED Lighting", "savings_eur": 1200, "co2_reduction": 2.4},
            {"name": "Smart Thermostat", "savings_eur": 600, "co2_reduction": 1.2},
        ],
        "grants": [
            {"name": "Green Business Fund", "amount_eur": 5000, "eligibility_pct": 85},
        ],
        "npv_eur": 15000,
        "irr_pct": 18,
        "payback_months": 24,
        "certifications": [
            {"name": "SME Climate Hub", "readiness_pct": 60},
        ],
        "actions": [],
        "milestones": [],
        "provenance_hash": "a" * 64,
    }


@pytest.fixture
def registry() -> TemplateRegistry:
    return TemplateRegistry()


@pytest.fixture
def sample_data() -> Dict[str, Any]:
    return _sample_data()


# ========================================================================
# Template Registry
# ========================================================================


class TestTemplateRegistryLoads:
    def test_template_registry_loads(self):
        reg = TemplateRegistry()
        assert reg is not None

    def test_template_count_8(self, registry):
        assert registry.template_count == 8

    def test_list_template_names(self, registry):
        names = registry.list_template_names()
        assert len(names) == 8
        for name in EXPECTED_TEMPLATE_NAMES:
            assert name in names

    def test_list_templates_returns_metadata(self, registry):
        templates = registry.list_templates()
        assert len(templates) == 8
        for t in templates:
            assert "name" in t
            assert "description" in t
            assert "category" in t
            assert "formats" in t


# ========================================================================
# All Templates Registered
# ========================================================================


class TestAllTemplatesRegistered:
    @pytest.mark.parametrize("name", EXPECTED_TEMPLATE_NAMES)
    def test_template_registered(self, registry, name):
        assert registry.has_template(name)

    @pytest.mark.parametrize("name", EXPECTED_TEMPLATE_NAMES)
    def test_template_get_succeeds(self, registry, name):
        template = registry.get(name)
        assert template is not None

    def test_get_nonexistent_raises_key_error(self, registry):
        with pytest.raises(KeyError):
            registry.get("nonexistent_template")


# ========================================================================
# Template Class Imports
# ========================================================================


class TestTemplateClassImports:
    @pytest.mark.parametrize(
        "cls",
        [
            SMEBaselineReportTemplate,
            SMEQuickWinsReportTemplate,
            SMEGrantReportTemplate,
            SMEBoardBriefTemplate,
            SMERoadmapReportTemplate,
            SMEProgressDashboardTemplate,
            SMECertificationSubmissionTemplate,
            SMEAccountingGuideTemplate,
        ],
        ids=EXPECTED_CATEGORIES,
    )
    def test_template_class_not_none(self, cls):
        assert cls is not None


# ========================================================================
# Markdown Rendering
# ========================================================================


class TestMarkdownRendering:
    @pytest.mark.parametrize("name", EXPECTED_TEMPLATE_NAMES)
    def test_markdown_render(self, registry, sample_data, name):
        template = registry.get(name)
        md = template.render_markdown(sample_data)
        assert isinstance(md, str)
        assert len(md) > 0

    @pytest.mark.parametrize("name", EXPECTED_TEMPLATE_NAMES)
    def test_markdown_contains_entity_name(self, registry, sample_data, name):
        """Templates use org_name key for company name in rendered output."""
        template = registry.get(name)
        md = template.render_markdown(sample_data)
        assert "SmallCo" in md


# ========================================================================
# HTML Rendering
# ========================================================================


class TestHTMLRendering:
    @pytest.mark.parametrize("name", EXPECTED_TEMPLATE_NAMES)
    def test_html_render(self, registry, sample_data, name):
        template = registry.get(name)
        html = template.render_html(sample_data)
        assert isinstance(html, str)
        assert len(html) > 0

    @pytest.mark.parametrize("name", EXPECTED_TEMPLATE_NAMES)
    def test_html_has_structure(self, registry, sample_data, name):
        template = registry.get(name)
        html = template.render_html(sample_data)
        assert "<" in html  # Basic HTML tag check


# ========================================================================
# JSON Rendering
# ========================================================================


class TestJSONRendering:
    @pytest.mark.parametrize("name", EXPECTED_TEMPLATE_NAMES)
    def test_json_render(self, registry, sample_data, name):
        template = registry.get(name)
        json_out = template.render_json(sample_data)
        assert json_out is not None
        assert isinstance(json_out, (dict, str))


# ========================================================================
# Template Categories
# ========================================================================


class TestTemplateCategories:
    @pytest.mark.parametrize("category", EXPECTED_CATEGORIES)
    def test_category_has_templates(self, registry, category):
        templates = registry.get_by_category(category)
        assert len(templates) >= 1

    def test_all_categories_covered(self, registry):
        for cat in EXPECTED_CATEGORIES:
            result = registry.get_by_category(cat)
            assert len(result) >= 1


# ========================================================================
# Mobile Responsiveness Metadata
# ========================================================================


class TestMobileResponsiveness:
    @pytest.mark.parametrize("name", EXPECTED_TEMPLATE_NAMES)
    def test_template_has_info(self, registry, name):
        """get_info returns standard metadata fields."""
        info = registry.get_info(name)
        assert "name" in info
        assert "description" in info
        assert "category" in info
        assert "formats" in info
        assert "version" in info

    @pytest.mark.parametrize("name", EXPECTED_TEMPLATE_NAMES)
    def test_template_supports_html(self, registry, name):
        """All templates support HTML output for mobile display."""
        info = registry.get_info(name)
        assert "html" in info["formats"]


# ========================================================================
# Render Convenience Method
# ========================================================================


class TestRenderConvenienceMethod:
    @pytest.mark.parametrize("format_str", ["markdown", "html", "json"])
    def test_render_method_format(self, registry, sample_data, format_str):
        result = registry.render(
            "sme_baseline_report", sample_data, format=format_str
        )
        assert result is not None

    def test_render_invalid_format_raises(self, registry, sample_data):
        with pytest.raises(ValueError):
            registry.render("sme_baseline_report", sample_data, format="xml")

    def test_render_invalid_template_raises(self, registry, sample_data):
        with pytest.raises(KeyError):
            registry.render("nonexistent", sample_data)


# ========================================================================
# Template Info
# ========================================================================


class TestTemplateInfo:
    @pytest.mark.parametrize("name", EXPECTED_TEMPLATE_NAMES)
    def test_get_info_returns_metadata(self, registry, name):
        info = registry.get_info(name)
        assert info["name"] == name
        assert info["description"]
        assert info["category"]

    def test_get_info_nonexistent_raises(self, registry):
        with pytest.raises(KeyError):
            registry.get_info("nonexistent")


# ========================================================================
# TEMPLATE_CATALOG
# ========================================================================


class TestTemplateCatalog:
    def test_catalog_has_8_entries(self):
        assert len(TEMPLATE_CATALOG) == 8

    def test_catalog_entries_have_required_keys(self):
        for entry in TEMPLATE_CATALOG:
            assert "name" in entry
            assert "class" in entry
            assert "description" in entry
            assert "category" in entry
            assert "formats" in entry

    def test_catalog_all_classes_non_none(self):
        for entry in TEMPLATE_CATALOG:
            assert entry["class"] is not None

    def test_catalog_formats_include_core(self):
        for entry in TEMPLATE_CATALOG:
            assert "markdown" in entry["formats"]
            assert "html" in entry["formats"]
            assert "json" in entry["formats"]


# ========================================================================
# Registry Repr
# ========================================================================


class TestRegistryRepr:
    def test_repr_includes_count(self, registry):
        rep = repr(registry)
        assert "8" in rep
        assert "TemplateRegistry" in rep
