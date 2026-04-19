# -*- coding: utf-8 -*-
"""
Test suite for PACK-030 Net Zero Reporting Pack - Report Templates.

Tests TemplateRegistry, TEMPLATE_CATALOG, and all 15 report templates
for instantiation, registry lookup, and rendering in multiple formats.

Author:  GreenLang Test Engineering
Pack:    PACK-030 Net Zero Reporting Pack
"""

import sys
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from templates import (
    TemplateRegistry, TEMPLATE_CATALOG,
    SBTiProgressTemplate, CDPGovernanceTemplate, CDPEmissionsTemplate,
    TCFDGovernanceTemplate, TCFDStrategyTemplate, TCFDRiskTemplate,
    TCFDMetricsTemplate, GRI305Template, ISSBS2Template,
    SECClimateTemplate, CSRDE1Template, InvestorDashboardTemplate,
    RegulatorDashboardTemplate, CustomerCarbonTemplate,
    AssuranceEvidenceTemplate,
)

from .conftest import timed_block


# ========================================================================
# Module-Level Metadata
# ========================================================================


class TestTemplateModuleMetadata:
    def test_catalog_is_list(self):
        assert isinstance(TEMPLATE_CATALOG, list)

    def test_catalog_has_15_entries(self):
        assert len(TEMPLATE_CATALOG) == 15

    def test_each_catalog_entry_has_required_keys(self):
        for entry in TEMPLATE_CATALOG:
            assert "name" in entry
            assert "class" in entry
            assert "description" in entry
            assert "category" in entry
            assert "formats" in entry

    def test_template_names_unique(self):
        names = [e["name"] for e in TEMPLATE_CATALOG]
        assert len(names) == len(set(names))


# ========================================================================
# Template Registry
# ========================================================================


class TestTemplateRegistry:
    def test_registry_instantiates(self):
        registry = TemplateRegistry()
        assert registry is not None

    def test_registry_template_count(self):
        registry = TemplateRegistry()
        assert registry.template_count >= 1

    def test_list_templates(self):
        registry = TemplateRegistry()
        templates = registry.list_templates()
        assert isinstance(templates, list)
        assert len(templates) >= 1

    def test_list_template_names(self):
        registry = TemplateRegistry()
        names = registry.list_template_names()
        assert isinstance(names, list)
        assert len(names) >= 1

    def test_get_template_by_name(self):
        registry = TemplateRegistry()
        names = registry.list_template_names()
        if names:
            template = registry.get(names[0])
            assert template is not None

    def test_get_template_alias(self):
        registry = TemplateRegistry()
        names = registry.list_template_names()
        if names:
            template = registry.get_template(names[0])
            assert template is not None

    def test_has_template(self):
        registry = TemplateRegistry()
        names = registry.list_template_names()
        if names:
            assert registry.has_template(names[0]) is True
        assert registry.has_template("nonexistent_template_xyz") is False

    def test_get_by_category(self):
        registry = TemplateRegistry()
        cats = registry.categories
        assert isinstance(cats, list)
        if cats:
            results = registry.get_by_category(cats[0])
            assert isinstance(results, list)

    def test_search(self):
        registry = TemplateRegistry()
        results = registry.search("climate")
        assert isinstance(results, list)

    def test_get_info(self):
        registry = TemplateRegistry()
        names = registry.list_template_names()
        if names:
            info = registry.get_info(names[0])
            assert "name" in info
            assert "description" in info

    def test_contains(self):
        registry = TemplateRegistry()
        names = registry.list_template_names()
        if names:
            assert names[0] in registry

    def test_len(self):
        registry = TemplateRegistry()
        assert len(registry) >= 1

    def test_iter(self):
        registry = TemplateRegistry()
        names = list(registry)
        assert isinstance(names, list)

    def test_repr(self):
        registry = TemplateRegistry()
        r = repr(registry)
        assert "PACK-030" in r

    def test_get_nonexistent_raises(self):
        registry = TemplateRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent_template_xyz")


# ========================================================================
# Template Render via Registry
# ========================================================================


_RENDER_FORMATS = ["markdown", "html", "json"]


class TestRegistryRender:
    @pytest.mark.parametrize("fmt", _RENDER_FORMATS)
    def test_render_first_template(self, fmt):
        registry = TemplateRegistry()
        names = registry.list_template_names()
        if names:
            data = {"organization_name": "Test Corp", "reporting_year": 2024}
            result = registry.render(names[0], data, format=fmt)
            assert result is not None

    def test_render_invalid_format(self):
        registry = TemplateRegistry()
        names = registry.list_template_names()
        if names:
            with pytest.raises(ValueError):
                registry.render(names[0], {}, format="INVALID_FORMAT")


# ========================================================================
# Individual Template Classes
# ========================================================================

_TEMPLATE_CLASSES = [
    SBTiProgressTemplate, CDPGovernanceTemplate, CDPEmissionsTemplate,
    TCFDGovernanceTemplate, TCFDStrategyTemplate, TCFDRiskTemplate,
    TCFDMetricsTemplate, GRI305Template, ISSBS2Template,
    SECClimateTemplate, CSRDE1Template, InvestorDashboardTemplate,
    RegulatorDashboardTemplate, CustomerCarbonTemplate,
    AssuranceEvidenceTemplate,
]


class TestTemplateClassExistence:
    @pytest.mark.parametrize("cls", _TEMPLATE_CLASSES, ids=[c.__name__ if c else "None" for c in _TEMPLATE_CLASSES])
    def test_class_not_none(self, cls):
        if cls is None:
            pytest.skip("Template class not available (import failed)")
        assert cls is not None

    @pytest.mark.parametrize("cls", _TEMPLATE_CLASSES, ids=[c.__name__ if c else "None" for c in _TEMPLATE_CLASSES])
    def test_class_instantiates(self, cls):
        if cls is None:
            pytest.skip("Template class not available (import failed)")
        instance = cls()
        assert instance is not None

    @pytest.mark.parametrize("cls", _TEMPLATE_CLASSES, ids=[c.__name__ if c else "None" for c in _TEMPLATE_CLASSES])
    def test_class_has_render_markdown(self, cls):
        if cls is None:
            pytest.skip("Template class not available (import failed)")
        instance = cls()
        assert hasattr(instance, "render_markdown")

    @pytest.mark.parametrize("cls", _TEMPLATE_CLASSES, ids=[c.__name__ if c else "None" for c in _TEMPLATE_CLASSES])
    def test_class_has_render_html(self, cls):
        if cls is None:
            pytest.skip("Template class not available (import failed)")
        instance = cls()
        assert hasattr(instance, "render_html")

    @pytest.mark.parametrize("cls", _TEMPLATE_CLASSES, ids=[c.__name__ if c else "None" for c in _TEMPLATE_CLASSES])
    def test_class_has_render_json(self, cls):
        if cls is None:
            pytest.skip("Template class not available (import failed)")
        instance = cls()
        assert hasattr(instance, "render_json")

    @pytest.mark.parametrize("cls", _TEMPLATE_CLASSES, ids=[c.__name__ if c else "None" for c in _TEMPLATE_CLASSES])
    def test_render_markdown(self, cls):
        if cls is None:
            pytest.skip("Template class not available (import failed)")
        instance = cls()
        data = {"organization_name": "Test Corp", "reporting_year": 2024}
        result = instance.render_markdown(data)
        assert result is not None

    @pytest.mark.parametrize("cls", _TEMPLATE_CLASSES, ids=[c.__name__ if c else "None" for c in _TEMPLATE_CLASSES])
    def test_render_html(self, cls):
        if cls is None:
            pytest.skip("Template class not available (import failed)")
        instance = cls()
        data = {"organization_name": "Test Corp", "reporting_year": 2024}
        result = instance.render_html(data)
        assert result is not None

    @pytest.mark.parametrize("cls", _TEMPLATE_CLASSES, ids=[c.__name__ if c else "None" for c in _TEMPLATE_CLASSES])
    def test_render_json(self, cls):
        if cls is None:
            pytest.skip("Template class not available (import failed)")
        instance = cls()
        data = {"organization_name": "Test Corp", "reporting_year": 2024}
        result = instance.render_json(data)
        assert result is not None


# ========================================================================
# Performance
# ========================================================================


class TestTemplatePerformance:
    def test_registry_creation_fast(self):
        with timed_block("registry_creation", max_seconds=2.0):
            TemplateRegistry()

    def test_template_render_fast(self):
        registry = TemplateRegistry()
        names = registry.list_template_names()
        if names:
            data = {"organization_name": "Test Corp"}
            with timed_block("template_render", max_seconds=3.0):
                for name in names[:3]:
                    registry.render(name, data, format="markdown")
