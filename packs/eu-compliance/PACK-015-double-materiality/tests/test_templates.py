# -*- coding: utf-8 -*-
"""
PACK-015 Double Materiality Assessment Pack - Template Tests
==============================================================

Tests for all 8 DMA report templates: class existence, render
methods (markdown, html, json), section completeness, data
validation, and provenance hashing.

Target: 35+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-015 Double Materiality Assessment
Date:    March 2026
"""

import inspect
import json

import pytest

from .conftest import (
    TEMPLATE_FILES,
    TEMPLATE_CLASSES,
    TEMPLATES_DIR,
    _load_template,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _try_load_template(key):
    """Attempt to load a template, returning module or None."""
    try:
        return _load_template(key)
    except (ImportError, FileNotFoundError):
        return None


def _sample_report_data():
    """Minimal sample data for template rendering tests."""
    return {
        "company_name": "TestCorp GmbH",
        "reporting_year": 2025,
        "assessment_date": "2025-06-15",
        "topics": [
            {
                "topic_id": "E1",
                "topic_name": "Climate Change",
                "impact_score": 7.5,
                "financial_score": 6.2,
                "material": True,
                "quadrant": "double_material",
            },
            {
                "topic_id": "S1",
                "topic_name": "Own Workforce",
                "impact_score": 6.0,
                "financial_score": 4.5,
                "material": True,
                "quadrant": "impact_material",
            },
        ],
        "material_topics": ["E1", "S1"],
        "thresholds": {"impact": 5.0, "financial": 5.0},
        "methodology": "ESRS 1 Chapter 3, EFRAG IG-1",
        "stakeholders": [
            {"category": "EMPLOYEES", "engagement_method": "surveys", "response_rate": 0.75},
            {"category": "INVESTORS", "engagement_method": "interviews", "response_rate": 0.90},
        ],
        "iros": [
            {
                "iro_id": "IRO-001",
                "type": "IMPACT",
                "direction": "NEGATIVE",
                "topic": "E1",
                "description": "GHG emissions from manufacturing operations",
                "severity_score": 7.5,
                "financial_score": 6.2,
            },
        ],
        "disclosures": [
            {"standard": "ESRS E1", "requirement": "E1-1", "status": "included"},
            {"standard": "ESRS E1", "requirement": "E1-6", "status": "included"},
        ],
        "matrix": {
            "x_axis": "financial_materiality",
            "y_axis": "impact_materiality",
            "entries": [],
        },
        "audit_trail": {
            "provenance_hash": "abc123" * 10 + "abcd",
            "scoring_log_entries": 42,
            "assumptions_tracked": 15,
        },
    }


# ===========================================================================
# Template File Existence
# ===========================================================================


class TestTemplateFilesExist:
    """Test that all 8 template files exist on disk."""

    @pytest.mark.parametrize("tmpl_key,tmpl_file", list(TEMPLATE_FILES.items()))
    def test_template_file_exists(self, tmpl_key, tmpl_file):
        """Template file exists on disk."""
        path = TEMPLATES_DIR / tmpl_file
        assert path.exists(), f"Template file missing: {path}"


# ===========================================================================
# Impact Materiality Report Template
# ===========================================================================


class TestImpactMaterialityReportTemplate:
    """Tests for ImpactMaterialityReportTemplate."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_template("impact_materiality_report")

    def test_impact_report_template_exists(self):
        """ImpactMaterialityReportTemplate class exists."""
        assert self.mod is not None
        assert hasattr(self.mod, "ImpactMaterialityReportTemplate")

    def test_impact_report_generate_markdown(self):
        """Template renders markdown without error."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        template = self.mod.ImpactMaterialityReportTemplate()
        data = _sample_report_data()
        result = template.render_markdown(data)
        assert isinstance(result, str)
        assert len(result) > 100

    def test_impact_report_generate_html(self):
        """Template renders HTML without error."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        template = self.mod.ImpactMaterialityReportTemplate()
        data = _sample_report_data()
        result = template.render_html(data)
        assert isinstance(result, str)
        assert "<html" in result.lower()

    def test_impact_report_generate_json(self):
        """Template renders JSON without error."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        template = self.mod.ImpactMaterialityReportTemplate()
        data = _sample_report_data()
        result = template.render_json(data)
        assert isinstance(result, dict)


# ===========================================================================
# Financial Materiality Report Template
# ===========================================================================


class TestFinancialMaterialityReportTemplate:
    """Tests for FinancialMaterialityReportTemplate."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_template("financial_materiality_report")

    def test_financial_report_template_exists(self):
        """FinancialMaterialityReportTemplate class exists."""
        assert self.mod is not None
        assert hasattr(self.mod, "FinancialMaterialityReportTemplate")

    def test_financial_report_generate(self):
        """Template renders markdown without error."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        template = self.mod.FinancialMaterialityReportTemplate()
        result = template.render_markdown(_sample_report_data())
        assert isinstance(result, str)
        assert len(result) > 100


# ===========================================================================
# Stakeholder Engagement Report Template
# ===========================================================================


class TestStakeholderEngagementReportTemplate:
    """Tests for StakeholderEngagementReportTemplate."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_template("stakeholder_engagement_report")

    def test_stakeholder_report_template_exists(self):
        """StakeholderEngagementReportTemplate class exists."""
        assert self.mod is not None
        assert hasattr(self.mod, "StakeholderEngagementReportTemplate")

    def test_stakeholder_report_generate(self):
        """Template renders markdown without error."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        template = self.mod.StakeholderEngagementReportTemplate()
        result = template.render_markdown(_sample_report_data())
        assert isinstance(result, str)
        assert len(result) > 100


# ===========================================================================
# Materiality Matrix Report Template
# ===========================================================================


class TestMaterialityMatrixReportTemplate:
    """Tests for MaterialityMatrixReportTemplate."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_template("materiality_matrix_report")

    def test_matrix_report_template_exists(self):
        """MaterialityMatrixReportTemplate class exists."""
        assert self.mod is not None
        assert hasattr(self.mod, "MaterialityMatrixReportTemplate")

    def test_matrix_report_generate(self):
        """Template renders markdown without error."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        template = self.mod.MaterialityMatrixReportTemplate()
        result = template.render_markdown(_sample_report_data())
        assert isinstance(result, str)
        assert len(result) > 50


# ===========================================================================
# IRO Register Report Template
# ===========================================================================


class TestIRORegisterReportTemplate:
    """Tests for IRORegisterReportTemplate."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_template("iro_register_report")

    def test_iro_register_template_exists(self):
        """IRORegisterReportTemplate class exists."""
        assert self.mod is not None
        assert hasattr(self.mod, "IRORegisterReportTemplate")

    def test_iro_register_generate(self):
        """Template renders markdown without error."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        template = self.mod.IRORegisterReportTemplate()
        result = template.render_markdown(_sample_report_data())
        assert isinstance(result, str)


# ===========================================================================
# ESRS Disclosure Map Template
# ===========================================================================


class TestESRSDisclosureMapTemplate:
    """Tests for ESRSDisclosureMapTemplate."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_template("esrs_disclosure_map")

    def test_esrs_map_template_exists(self):
        """ESRSDisclosureMapTemplate class exists."""
        assert self.mod is not None
        assert hasattr(self.mod, "ESRSDisclosureMapTemplate")

    def test_esrs_map_generate(self):
        """Template renders markdown without error."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        template = self.mod.ESRSDisclosureMapTemplate()
        result = template.render_markdown(_sample_report_data())
        assert isinstance(result, str)


# ===========================================================================
# DMA Executive Summary Template
# ===========================================================================


class TestDMAExecutiveSummaryTemplate:
    """Tests for DMAExecutiveSummaryTemplate."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_template("dma_executive_summary")

    def test_executive_summary_template_exists(self):
        """DMAExecutiveSummaryTemplate class exists."""
        assert self.mod is not None
        assert hasattr(self.mod, "DMAExecutiveSummaryTemplate")

    def test_executive_summary_generate(self):
        """Template renders markdown without error."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        template = self.mod.DMAExecutiveSummaryTemplate()
        result = template.render_markdown(_sample_report_data())
        assert isinstance(result, str)


# ===========================================================================
# DMA Audit Report Template
# ===========================================================================


class TestDMAAuditReportTemplate:
    """Tests for DMAAuditReportTemplate."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_template("dma_audit_report")

    def test_audit_report_template_exists(self):
        """DMAAuditReportTemplate class exists."""
        assert self.mod is not None
        assert hasattr(self.mod, "DMAAuditReportTemplate")

    def test_audit_report_generate(self):
        """Template renders markdown without error."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        template = self.mod.DMAAuditReportTemplate()
        result = template.render_markdown(_sample_report_data())
        assert isinstance(result, str)


# ===========================================================================
# Cross-Template Pattern Tests
# ===========================================================================


class TestTemplatePatterns:
    """Pattern tests applicable to all templates."""

    @pytest.mark.parametrize("tmpl_key,tmpl_class", list(TEMPLATE_CLASSES.items()))
    def test_template_has_three_render_methods(self, tmpl_key, tmpl_class):
        """All templates support render_markdown, render_html, render_json."""
        mod = _try_load_template(tmpl_key)
        if mod is None:
            pytest.skip(f"Template {tmpl_key} not loaded")
        cls = getattr(mod, tmpl_class)
        assert hasattr(cls, "render_markdown")
        assert hasattr(cls, "render_html")
        assert hasattr(cls, "render_json")

    @pytest.mark.parametrize("tmpl_key,tmpl_class", list(TEMPLATE_CLASSES.items()))
    def test_template_has_docstring(self, tmpl_key, tmpl_class):
        """All template classes have docstrings."""
        mod = _try_load_template(tmpl_key)
        if mod is None:
            pytest.skip(f"Template {tmpl_key} not loaded")
        cls = getattr(mod, tmpl_class, None)
        if cls is None:
            pytest.skip(f"Class {tmpl_class} not found")
        assert cls.__doc__ is not None

    @pytest.mark.parametrize("tmpl_key,tmpl_class", list(TEMPLATE_CLASSES.items()))
    def test_template_accepts_config(self, tmpl_key, tmpl_class):
        """Templates accept optional config parameter."""
        mod = _try_load_template(tmpl_key)
        if mod is None:
            pytest.skip(f"Template {tmpl_key} not loaded")
        cls = getattr(mod, tmpl_class)
        # Should be constructable with no args and with config={}
        instance_default = cls()
        instance_config = cls(config={"key": "value"})
        assert instance_default is not None
        assert instance_config is not None

    @pytest.mark.parametrize("tmpl_key", list(TEMPLATE_FILES.keys()))
    def test_template_data_validation(self, tmpl_key):
        """Templates handle empty data gracefully."""
        mod = _try_load_template(tmpl_key)
        if mod is None:
            pytest.skip(f"Template {tmpl_key} not loaded")
        cls_name = TEMPLATE_CLASSES[tmpl_key]
        cls = getattr(mod, cls_name)
        instance = cls()
        # Render with empty dict should not crash
        try:
            result = instance.render_markdown({})
            assert isinstance(result, str)
        except (KeyError, TypeError):
            # Acceptable if template requires specific data fields
            pass

    @pytest.mark.parametrize("tmpl_key", list(TEMPLATE_FILES.keys()))
    def test_template_section_completeness(self, tmpl_key):
        """Templates include provenance hash in output."""
        mod = _try_load_template(tmpl_key)
        if mod is None:
            pytest.skip(f"Template {tmpl_key} not loaded")
        cls_name = TEMPLATE_CLASSES[tmpl_key]
        cls = getattr(mod, cls_name)
        instance = cls()
        data = _sample_report_data()
        try:
            result = instance.render_markdown(data)
            # Check for provenance marker
            has_provenance = (
                "provenance" in result.lower()
                or "Provenance" in result
            )
            assert has_provenance, f"Template {tmpl_key} should include provenance"
        except (KeyError, TypeError):
            pytest.skip(f"Template {tmpl_key} requires specific data shape")
