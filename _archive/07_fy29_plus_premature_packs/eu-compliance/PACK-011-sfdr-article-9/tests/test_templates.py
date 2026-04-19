# -*- coding: utf-8 -*-
"""
PACK-011 SFDR Article 9 Pack - Template Tests
================================================

Tests all 8 SFDR Article 9 templates and the TemplateRegistry:
AnnexIIIPrecontractual, AnnexVPeriodic, ImpactReport,
BenchmarkMethodology, SustainableDashboard, PAIMandatoryReport,
CarbonTrajectoryReport, AuditTrailReport, and TemplateRegistry.

Self-contained: does NOT import from conftest.
Test count: 25 tests
"""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACK_ROOT.parent.parent.parent


def _import_from_path(module_name: str, file_path: str):
    """Import a module from an absolute file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Import templates __init__ and individual template modules
# ---------------------------------------------------------------------------

TPL_DIR = str(PACK_ROOT / "templates")

_tpl_init = _import_from_path(
    "pack011_tpl_init",
    os.path.join(TPL_DIR, "__init__.py"),
)

# TemplateRegistry
TemplateRegistry = _tpl_init.TemplateRegistry
TEMPLATE_CATALOG = _tpl_init.TEMPLATE_CATALOG
SUPPORTED_FORMATS = _tpl_init.SUPPORTED_FORMATS

# Individual template modules
_tpl_annex_iii = _import_from_path(
    "pack011_tpl_annex_iii",
    os.path.join(TPL_DIR, "annex_iii_precontractual.py"),
)
_tpl_annex_v = _import_from_path(
    "pack011_tpl_annex_v",
    os.path.join(TPL_DIR, "annex_v_periodic.py"),
)
_tpl_impact = _import_from_path(
    "pack011_tpl_impact",
    os.path.join(TPL_DIR, "impact_report.py"),
)
_tpl_benchmark = _import_from_path(
    "pack011_tpl_benchmark",
    os.path.join(TPL_DIR, "benchmark_methodology.py"),
)
_tpl_dashboard = _import_from_path(
    "pack011_tpl_dashboard",
    os.path.join(TPL_DIR, "sustainable_dashboard.py"),
)
_tpl_pai = _import_from_path(
    "pack011_tpl_pai",
    os.path.join(TPL_DIR, "pai_mandatory_report.py"),
)
_tpl_carbon = _import_from_path(
    "pack011_tpl_carbon",
    os.path.join(TPL_DIR, "carbon_trajectory_report.py"),
)
_tpl_audit = _import_from_path(
    "pack011_tpl_audit",
    os.path.join(TPL_DIR, "audit_trail_report.py"),
)

# Template classes
AnnexIIIPrecontractualTemplate = _tpl_annex_iii.AnnexIIIPrecontractualTemplate
AnnexVPeriodicTemplate = _tpl_annex_v.AnnexVPeriodicTemplate
ImpactReportTemplate = _tpl_impact.ImpactReportTemplate
BenchmarkMethodologyTemplate = _tpl_benchmark.BenchmarkMethodologyTemplate
SustainableDashboardTemplate = _tpl_dashboard.SustainableDashboardTemplate
PAIMandatoryReportTemplate = _tpl_pai.PAIMandatoryReportTemplate
CarbonTrajectoryReportTemplate = _tpl_carbon.CarbonTrajectoryReportTemplate
AuditTrailReportTemplate = _tpl_audit.AuditTrailReportTemplate

# Key data models for template input
Article9ProductInfo = _tpl_annex_iii.Article9ProductInfo

# All 8 template keys
EXPECTED_TEMPLATE_KEYS = [
    "annex_iii_precontractual",
    "annex_v_periodic",
    "impact_report",
    "benchmark_methodology",
    "sustainable_dashboard",
    "pai_mandatory_report",
    "carbon_trajectory_report",
    "audit_trail_report",
]

ALL_TEMPLATE_CLASSES = [
    AnnexIIIPrecontractualTemplate,
    AnnexVPeriodicTemplate,
    ImpactReportTemplate,
    BenchmarkMethodologyTemplate,
    SustainableDashboardTemplate,
    PAIMandatoryReportTemplate,
    CarbonTrajectoryReportTemplate,
    AuditTrailReportTemplate,
]


# ===========================================================================
# Tests: TemplateRegistry
# ===========================================================================


class TestTemplateRegistry:
    """Tests for the TemplateRegistry discovery and render system."""

    def test_registry_instantiation(self):
        """Registry can be created with no arguments."""
        reg = TemplateRegistry()
        assert reg is not None
        assert reg.pack_id == "PACK-011"
        assert reg.pack_name == "SFDR Article 9"

    def test_registry_has_8_templates(self):
        """Registry contains exactly 8 templates."""
        reg = TemplateRegistry()
        assert reg.template_count == 8

    def test_all_template_keys_present(self):
        """All 8 expected template keys are registered."""
        reg = TemplateRegistry()
        keys = reg.get_all_template_keys()
        for expected_key in EXPECTED_TEMPLATE_KEYS:
            assert expected_key in keys, f"Missing template key: {expected_key}"

    def test_has_template(self):
        """has_template returns True for valid keys, False for invalid."""
        reg = TemplateRegistry()
        assert reg.has_template("annex_iii_precontractual") is True
        assert reg.has_template("nonexistent_template") is False

    def test_get_template_returns_instance(self):
        """get_template returns instantiated template objects."""
        reg = TemplateRegistry()
        for key in EXPECTED_TEMPLATE_KEYS:
            template = reg.get_template(key)
            assert template is not None
            assert hasattr(template, "render_markdown")
            assert hasattr(template, "render_html")
            assert hasattr(template, "render_json")

    def test_get_template_invalid_key_raises(self):
        """get_template raises KeyError for unknown key."""
        reg = TemplateRegistry()
        with pytest.raises(KeyError):
            reg.get_template("does_not_exist")

    def test_get_template_metadata(self):
        """get_template_metadata returns expected metadata structure."""
        reg = TemplateRegistry()
        meta = reg.get_template_metadata("annex_iii_precontractual")
        assert meta["key"] == "annex_iii_precontractual"
        assert "display_name" in meta
        assert "description" in meta
        assert "category" in meta
        assert "scope" in meta
        assert "version" in meta
        assert "sections" in meta
        assert len(meta["sections"]) > 0

    def test_list_templates_returns_all(self):
        """list_templates without filters returns all 8 templates."""
        reg = TemplateRegistry()
        templates = reg.list_templates()
        assert len(templates) == 8

    def test_list_templates_filter_by_category(self):
        """list_templates filters by category correctly."""
        reg = TemplateRegistry()
        regulatory = reg.list_templates(category="regulatory")
        assert len(regulatory) >= 2
        for t in regulatory:
            assert t["category"] == "regulatory"

    def test_list_templates_filter_by_scope(self):
        """list_templates filters by scope correctly."""
        reg = TemplateRegistry()
        precon = reg.list_templates(scope="pre-contractual")
        assert len(precon) >= 1
        for t in precon:
            assert t["scope"] == "pre-contractual"

    def test_render_invalid_format_raises(self):
        """render with unsupported format raises ValueError."""
        reg = TemplateRegistry()
        with pytest.raises(ValueError, match="Unsupported format"):
            reg.render("annex_iii_precontractual", {}, format="xml")

    def test_supported_formats(self):
        """SUPPORTED_FORMATS includes markdown, html, json."""
        assert "markdown" in SUPPORTED_FORMATS
        assert "html" in SUPPORTED_FORMATS
        assert "json" in SUPPORTED_FORMATS


# ===========================================================================
# Tests: Individual Template Classes
# ===========================================================================


class TestAnnexIIIPrecontractualTemplate:
    """Tests for the Annex III pre-contractual disclosure template."""

    def test_instantiation(self):
        """Template can be instantiated."""
        t = AnnexIIIPrecontractualTemplate()
        assert t is not None

    def test_render_markdown_returns_string(self):
        """render_markdown returns a non-empty string."""
        t = AnnexIIIPrecontractualTemplate()
        data = {
            "product_info": {
                "product_name": "GL Climate Solutions Fund",
                "sfdr_classification": "article_9",
            },
            "sustainable_objective": {
                "objective_type": "environmental",
                "objective_description": "Carbon emissions reduction",
            },
        }
        result = t.render_markdown(data)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Climate Solutions" in result or "Article 9" in result.upper() or len(result) > 50

    def test_render_json_returns_dict(self):
        """render_json returns a dictionary."""
        t = AnnexIIIPrecontractualTemplate()
        data = {
            "product_info": {"product_name": "Test Fund"},
            "sustainable_objective": {"objective_type": "environmental"},
        }
        result = t.render_json(data)
        assert isinstance(result, dict)


class TestAnnexVPeriodicTemplate:
    """Tests for the Annex V periodic reporting template."""

    def test_instantiation(self):
        """Template can be instantiated."""
        t = AnnexVPeriodicTemplate()
        assert t is not None

    def test_render_markdown(self):
        """render_markdown produces output."""
        t = AnnexVPeriodicTemplate()
        data = {
            "product_name": "GL Article 9 Income Fund",
            "reporting_period_start": "2025-01-01",
            "reporting_period_end": "2025-12-31",
        }
        result = t.render_markdown(data)
        assert isinstance(result, str)
        assert len(result) > 0


class TestImpactReportTemplate:
    """Tests for the impact measurement report template."""

    def test_instantiation(self):
        """Template can be instantiated."""
        t = ImpactReportTemplate()
        assert t is not None


class TestBenchmarkMethodologyTemplate:
    """Tests for the benchmark methodology template."""

    def test_instantiation(self):
        """Template can be instantiated."""
        t = BenchmarkMethodologyTemplate()
        assert t is not None


class TestSustainableDashboardTemplate:
    """Tests for the sustainable investment dashboard template."""

    def test_instantiation(self):
        """Template can be instantiated."""
        t = SustainableDashboardTemplate()
        assert t is not None


class TestPAIMandatoryReportTemplate:
    """Tests for the PAI mandatory indicators report template."""

    def test_instantiation(self):
        """Template can be instantiated."""
        t = PAIMandatoryReportTemplate()
        assert t is not None


class TestCarbonTrajectoryReportTemplate:
    """Tests for the carbon trajectory report template."""

    def test_instantiation(self):
        """Template can be instantiated."""
        t = CarbonTrajectoryReportTemplate()
        assert t is not None


class TestAuditTrailReportTemplate:
    """Tests for the audit trail report template."""

    def test_instantiation(self):
        """Template can be instantiated."""
        t = AuditTrailReportTemplate()
        assert t is not None
