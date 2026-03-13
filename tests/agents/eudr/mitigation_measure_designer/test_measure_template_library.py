# -*- coding: utf-8 -*-
"""
Unit tests for MeasureTemplateLibrary - AGENT-EUDR-029

Tests template loading, retrieval, search, and coverage of Article 11
categories, risk dimensions, and commodities.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.agents.eudr.mitigation_measure_designer.config import (
    MitigationMeasureDesignerConfig,
)
from greenlang.agents.eudr.mitigation_measure_designer.measure_template_library import (
    MeasureTemplateLibrary,
)
from greenlang.agents.eudr.mitigation_measure_designer.models import (
    Article11Category,
    EUDRCommodity,
    MeasureTemplate,
    RiskDimension,
)


@pytest.fixture
def library():
    config = MitigationMeasureDesignerConfig()
    lib = MeasureTemplateLibrary(config=config)
    lib.load_templates()
    return lib


class TestLoadTemplates:
    """Test template loading from built-in definitions."""

    def test_load_returns_count_ge_50(self, library):
        assert library.template_count >= 50

    def test_load_is_idempotent(self, library):
        count1 = library.template_count
        library.load_templates()
        count2 = library.template_count
        assert count1 == count2

    def test_is_loaded_flag(self):
        lib = MeasureTemplateLibrary()
        assert lib.is_loaded is False
        lib.load_templates()
        assert lib.is_loaded is True

    def test_auto_load_on_get_template(self):
        lib = MeasureTemplateLibrary()
        assert lib.is_loaded is False
        t = lib.get_template("MMD-TPL-001")
        assert lib.is_loaded is True
        assert t is not None


class TestGetTemplate:
    """Test template retrieval by ID."""

    def test_get_existing_template(self, library):
        t = library.get_template("MMD-TPL-001")
        assert t is not None
        assert isinstance(t, MeasureTemplate)
        assert t.template_id == "MMD-TPL-001"

    def test_get_nonexistent_template_returns_none(self, library):
        assert library.get_template("NONEXISTENT") is None

    def test_get_template_has_required_fields(self, library):
        t = library.get_template("MMD-TPL-001")
        assert t.title != ""
        assert t.description != ""
        assert len(t.applicable_dimensions) > 0
        assert t.base_effectiveness > Decimal("0")
        assert t.regulatory_reference != ""


class TestSearchTemplates:
    """Test template search by dimension, category, commodity."""

    def test_search_by_country_dimension(self, library):
        results = library.search_templates(dimension=RiskDimension.COUNTRY)
        assert len(results) > 0
        for t in results:
            assert RiskDimension.COUNTRY in t.applicable_dimensions

    def test_search_by_supplier_dimension(self, library):
        results = library.search_templates(dimension=RiskDimension.SUPPLIER)
        assert len(results) > 0

    def test_search_by_independent_audit_category(self, library):
        results = library.search_templates(category=Article11Category.INDEPENDENT_AUDIT)
        assert len(results) > 0
        for t in results:
            assert t.article11_category == Article11Category.INDEPENDENT_AUDIT

    def test_search_by_commodity_oil_palm(self, library):
        results = library.search_templates(commodity=EUDRCommodity.OIL_PALM)
        assert len(results) > 0

    def test_search_combined_filters(self, library):
        results = library.search_templates(
            dimension=RiskDimension.COMMODITY,
            category=Article11Category.INDEPENDENT_AUDIT,
        )
        for t in results:
            assert RiskDimension.COMMODITY in t.applicable_dimensions
            assert t.article11_category == Article11Category.INDEPENDENT_AUDIT


class TestGetTemplatesForDimension:
    """Test get_templates_for_dimension convenience method."""

    def test_returns_sorted_by_effectiveness(self, library):
        results = library.get_templates_for_dimension(
            dimension=RiskDimension.DEFORESTATION,
            commodity=EUDRCommodity.COFFEE,
        )
        assert len(results) > 0
        for i in range(len(results) - 1):
            assert results[i].base_effectiveness >= results[i + 1].base_effectiveness

    def test_returns_results_for_all_dimensions(self, library):
        for dim in RiskDimension:
            results = library.get_templates_for_dimension(
                dimension=dim,
                commodity=EUDRCommodity.COFFEE,
            )
            # Every dimension should have at least one template
            assert len(results) > 0, f"No templates for dimension {dim.value}"


class TestCoverageCompleteness:
    """Test that the template library covers all required categories."""

    def test_all_article11_categories_covered(self, library):
        all_templates = library.get_all_templates()
        categories_used = {t.article11_category for t in all_templates}
        for cat in Article11Category:
            assert cat in categories_used, f"Category {cat.value} not covered"

    def test_all_risk_dimensions_covered(self, library):
        all_templates = library.get_all_templates()
        dims_covered = set()
        for t in all_templates:
            for d in t.applicable_dimensions:
                dims_covered.add(d)
        for dim in RiskDimension:
            assert dim in dims_covered, f"Dimension {dim.value} not covered"

    def test_commodity_specific_templates_exist(self, library):
        """At least some templates should be commodity-specific."""
        all_templates = library.get_all_templates()
        commodity_specific = [t for t in all_templates if len(t.applicable_commodities) > 0]
        assert len(commodity_specific) > 0

    def test_template_count_property(self, library):
        count = library.template_count
        all_templates = library.get_all_templates()
        assert count == len(all_templates)
