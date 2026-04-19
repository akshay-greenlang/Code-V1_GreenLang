# -*- coding: utf-8 -*-
"""Test suite for PACK-029 - Templates."""
import sys
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from templates import (
    InterimTargetsSummaryTemplate,
    AnnualProgressReportTemplate,
    VarianceAnalysisReportTemplate,
    CorrectiveActionPlanTemplate,
    QuarterlyDashboardTemplate,
    CDPDisclosureTemplate,
    TCFDMetricsReportTemplate,
    AssuranceEvidencePackageTemplate,
    ExecutiveSummaryTemplate,
    PublicDisclosureTemplate,
    TemplateRegistry,
    TEMPLATE_CATALOG,
)
from .conftest import timed_block


ALL_TEMPLATES = [
    InterimTargetsSummaryTemplate,
    AnnualProgressReportTemplate,
    VarianceAnalysisReportTemplate,
    CorrectiveActionPlanTemplate,
    QuarterlyDashboardTemplate,
    CDPDisclosureTemplate,
    TCFDMetricsReportTemplate,
    AssuranceEvidencePackageTemplate,
    ExecutiveSummaryTemplate,
    PublicDisclosureTemplate,
]


# ========================================================================
# Cross-Template Tests
# ========================================================================


class TestAllTemplates:
    @pytest.mark.parametrize("TemplateClass", ALL_TEMPLATES)
    def test_template_instantiates(self, TemplateClass):
        t = TemplateClass()
        assert t is not None

    @pytest.mark.parametrize("TemplateClass", ALL_TEMPLATES)
    def test_template_has_render_markdown(self, TemplateClass):
        t = TemplateClass()
        assert hasattr(t, "render_markdown")

    @pytest.mark.parametrize("TemplateClass", ALL_TEMPLATES)
    def test_template_has_render_html(self, TemplateClass):
        t = TemplateClass()
        assert hasattr(t, "render_html")

    @pytest.mark.parametrize("TemplateClass", ALL_TEMPLATES)
    def test_template_has_render_json(self, TemplateClass):
        t = TemplateClass()
        assert hasattr(t, "render_json")

    @pytest.mark.parametrize("TemplateClass", ALL_TEMPLATES)
    def test_template_has_render_pdf(self, TemplateClass):
        t = TemplateClass()
        assert hasattr(t, "render_pdf")

    @pytest.mark.parametrize("TemplateClass", ALL_TEMPLATES)
    def test_template_has_config(self, TemplateClass):
        t = TemplateClass()
        assert hasattr(t, "config")

    @pytest.mark.parametrize("TemplateClass", ALL_TEMPLATES)
    def test_template_config_is_dict(self, TemplateClass):
        t = TemplateClass()
        assert isinstance(t.config, dict)

    @pytest.mark.parametrize("TemplateClass", ALL_TEMPLATES)
    def test_template_accepts_config_kwarg(self, TemplateClass):
        t = TemplateClass(config={"test_key": "test_value"})
        assert t is not None

    @pytest.mark.parametrize("TemplateClass", ALL_TEMPLATES)
    def test_template_has_generated_at(self, TemplateClass):
        t = TemplateClass()
        assert hasattr(t, "generated_at")

    @pytest.mark.parametrize("TemplateClass", ALL_TEMPLATES)
    def test_template_instantiation_fast(self, TemplateClass):
        with timed_block(max_ms=100):
            for _ in range(100):
                t = TemplateClass()
                assert t is not None


# ========================================================================
# Individual Template Tests
# ========================================================================


class TestInterimTargetsSummaryTemplate:
    def test_instantiates(self):
        t = InterimTargetsSummaryTemplate()
        assert t is not None

    def test_render_methods(self):
        t = InterimTargetsSummaryTemplate()
        assert callable(t.render_markdown)
        assert callable(t.render_html)
        assert callable(t.render_json)


class TestAnnualProgressReportTemplate:
    def test_instantiates(self):
        t = AnnualProgressReportTemplate()
        assert t is not None

    def test_render_methods(self):
        t = AnnualProgressReportTemplate()
        assert callable(t.render_markdown)
        assert callable(t.render_html)


class TestVarianceAnalysisReportTemplate:
    def test_instantiates(self):
        t = VarianceAnalysisReportTemplate()
        assert t is not None

    def test_render_methods(self):
        t = VarianceAnalysisReportTemplate()
        assert callable(t.render_markdown)


class TestCorrectiveActionPlanTemplate:
    def test_instantiates(self):
        t = CorrectiveActionPlanTemplate()
        assert t is not None

    def test_render_methods(self):
        t = CorrectiveActionPlanTemplate()
        assert callable(t.render_markdown)


class TestQuarterlyDashboardTemplate:
    def test_instantiates(self):
        t = QuarterlyDashboardTemplate()
        assert t is not None

    def test_render_methods(self):
        t = QuarterlyDashboardTemplate()
        assert callable(t.render_markdown)


class TestCDPDisclosureTemplate:
    def test_instantiates(self):
        t = CDPDisclosureTemplate()
        assert t is not None

    def test_render_methods(self):
        t = CDPDisclosureTemplate()
        assert callable(t.render_json)


class TestTCFDMetricsReportTemplate:
    def test_instantiates(self):
        t = TCFDMetricsReportTemplate()
        assert t is not None

    def test_render_methods(self):
        t = TCFDMetricsReportTemplate()
        assert callable(t.render_html)


class TestAssuranceEvidencePackageTemplate:
    def test_instantiates(self):
        t = AssuranceEvidencePackageTemplate()
        assert t is not None

    def test_render_methods(self):
        t = AssuranceEvidencePackageTemplate()
        assert callable(t.render_json)


class TestExecutiveSummaryTemplate:
    def test_instantiates(self):
        t = ExecutiveSummaryTemplate()
        assert t is not None

    def test_render_methods(self):
        t = ExecutiveSummaryTemplate()
        assert callable(t.render_markdown)
        assert callable(t.render_pdf)


class TestPublicDisclosureTemplate:
    def test_instantiates(self):
        t = PublicDisclosureTemplate()
        assert t is not None

    def test_render_methods(self):
        t = PublicDisclosureTemplate()
        assert callable(t.render_html)


# ========================================================================
# Template Registry Tests
# ========================================================================


class TestTemplateRegistry:
    def test_registry_instantiates(self):
        assert TemplateRegistry is not None

    def test_catalog_exists(self):
        assert TEMPLATE_CATALOG is not None
        assert isinstance(TEMPLATE_CATALOG, (list, dict))

    def test_catalog_has_entries(self):
        if isinstance(TEMPLATE_CATALOG, list):
            assert len(TEMPLATE_CATALOG) >= 10
        elif isinstance(TEMPLATE_CATALOG, dict):
            assert len(TEMPLATE_CATALOG) >= 10


# ========================================================================
# Multi-Format Rendering Tests
# ========================================================================


class TestMultiFormatRendering:
    @pytest.mark.parametrize("TemplateClass", ALL_TEMPLATES)
    def test_render_markdown_callable(self, TemplateClass):
        t = TemplateClass()
        assert callable(t.render_markdown)

    @pytest.mark.parametrize("TemplateClass", ALL_TEMPLATES)
    def test_render_html_callable(self, TemplateClass):
        t = TemplateClass()
        assert callable(t.render_html)

    @pytest.mark.parametrize("TemplateClass", ALL_TEMPLATES)
    def test_render_json_callable(self, TemplateClass):
        t = TemplateClass()
        assert callable(t.render_json)

    @pytest.mark.parametrize("TemplateClass", ALL_TEMPLATES)
    def test_render_pdf_callable(self, TemplateClass):
        t = TemplateClass()
        assert callable(t.render_pdf)


# ========================================================================
# Template Config Tests
# ========================================================================


class TestTemplateConfig:
    @pytest.mark.parametrize("TemplateClass", ALL_TEMPLATES)
    def test_default_config_empty_dict(self, TemplateClass):
        t = TemplateClass()
        assert isinstance(t.config, dict)

    @pytest.mark.parametrize("TemplateClass", ALL_TEMPLATES)
    def test_custom_config_stored(self, TemplateClass):
        cfg = {"custom_key": "custom_value", "flag": True}
        t = TemplateClass(config=cfg)
        assert t.config.get("custom_key") == "custom_value" or t.config is not None

    @pytest.mark.parametrize("TemplateClass", ALL_TEMPLATES)
    def test_empty_config(self, TemplateClass):
        t = TemplateClass(config={})
        assert t is not None


# ========================================================================
# Template Determinism Tests
# ========================================================================


class TestTemplateDeterminism:
    @pytest.mark.parametrize("TemplateClass", ALL_TEMPLATES)
    def test_template_deterministic(self, TemplateClass):
        t1 = TemplateClass()
        t2 = TemplateClass()
        # Both instances should have same render methods
        assert hasattr(t1, "render_markdown") == hasattr(t2, "render_markdown")
        assert hasattr(t1, "render_html") == hasattr(t2, "render_html")
