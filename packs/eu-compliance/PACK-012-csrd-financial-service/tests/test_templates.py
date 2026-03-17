# -*- coding: utf-8 -*-
"""
PACK-012 CSRD Financial Service Pack - Template Tests
========================================================

Tests all 8 templates and the TemplateRegistry:
PCAFReportTemplate, GARBTARReportTemplate, Pillar3ESGTemplate,
ClimateRiskReportTemplate, FSESRSChapterTemplate,
FinancedEmissionsDashboard, InsuranceESGTemplate, SBTiFIReportTemplate.

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
    "pack012_tpl_init",
    os.path.join(TPL_DIR, "__init__.py"),
)

# TemplateRegistry
TemplateRegistry = _tpl_init.TemplateRegistry
TEMPLATE_CATALOG = _tpl_init.TEMPLATE_CATALOG
SUPPORTED_FORMATS = _tpl_init.SUPPORTED_FORMATS

# Individual template modules
_tpl_pcaf = _import_from_path(
    "pack012_tpl_pcaf",
    os.path.join(TPL_DIR, "pcaf_report.py"),
)
_tpl_gar = _import_from_path(
    "pack012_tpl_gar",
    os.path.join(TPL_DIR, "gar_btar_report.py"),
)
_tpl_pillar3 = _import_from_path(
    "pack012_tpl_pillar3",
    os.path.join(TPL_DIR, "pillar3_esg_template.py"),
)
_tpl_climate = _import_from_path(
    "pack012_tpl_climate",
    os.path.join(TPL_DIR, "climate_risk_report.py"),
)
_tpl_esrs = _import_from_path(
    "pack012_tpl_esrs",
    os.path.join(TPL_DIR, "fs_esrs_chapter.py"),
)
_tpl_dashboard = _import_from_path(
    "pack012_tpl_dashboard",
    os.path.join(TPL_DIR, "financed_emissions_dashboard.py"),
)
_tpl_insurance = _import_from_path(
    "pack012_tpl_insurance",
    os.path.join(TPL_DIR, "insurance_esg_template.py"),
)
_tpl_sbti = _import_from_path(
    "pack012_tpl_sbti",
    os.path.join(TPL_DIR, "sbti_fi_report.py"),
)

# Template classes
PCAFReportTemplate = _tpl_pcaf.PCAFReportTemplate
GARBTARReportTemplate = _tpl_gar.GARBTARReportTemplate
Pillar3ESGTemplate = _tpl_pillar3.Pillar3ESGTemplate
ClimateRiskReportTemplate = _tpl_climate.ClimateRiskReportTemplate
FSESRSChapterTemplate = _tpl_esrs.FSESRSChapterTemplate
FinancedEmissionsDashboard = _tpl_dashboard.FinancedEmissionsDashboard
InsuranceESGTemplate = _tpl_insurance.InsuranceESGTemplate
SBTiFIReportTemplate = _tpl_sbti.SBTiFIReportTemplate

# All 8 template keys
EXPECTED_TEMPLATE_KEYS = [
    "pcaf_report",
    "gar_btar_report",
    "pillar3_esg_template",
    "climate_risk_report",
    "fs_esrs_chapter",
    "financed_emissions_dashboard",
    "insurance_esg_template",
    "sbti_fi_report",
]

ALL_TEMPLATE_CLASSES = [
    PCAFReportTemplate,
    GARBTARReportTemplate,
    Pillar3ESGTemplate,
    ClimateRiskReportTemplate,
    FSESRSChapterTemplate,
    FinancedEmissionsDashboard,
    InsuranceESGTemplate,
    SBTiFIReportTemplate,
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
        assert reg.pack_id == "PACK-012"
        assert reg.pack_name == "CSRD Financial Service"

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
        assert reg.has_template("pcaf_report") is True
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
        meta = reg.get_template_metadata("pcaf_report")
        assert meta["key"] == "pcaf_report"
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
        financed = reg.list_templates(scope="financed_emissions")
        assert len(financed) >= 1
        for t in financed:
            assert t["scope"] == "financed_emissions"

    def test_render_invalid_format_raises(self):
        """render with unsupported format raises ValueError."""
        reg = TemplateRegistry()
        with pytest.raises(ValueError, match="Unsupported format"):
            reg.render("pcaf_report", {}, format="xml")

    def test_supported_formats(self):
        """SUPPORTED_FORMATS includes markdown, html, json."""
        assert "markdown" in SUPPORTED_FORMATS
        assert "html" in SUPPORTED_FORMATS
        assert "json" in SUPPORTED_FORMATS


# ===========================================================================
# Tests: Individual Template Classes
# ===========================================================================


class TestPCAFReportTemplate:
    """Tests for the PCAF financed emissions disclosure template."""

    def test_instantiation(self):
        """Template can be instantiated."""
        t = PCAFReportTemplate()
        assert t is not None

    def test_render_markdown_returns_string(self):
        """render_markdown returns a non-empty string."""
        t = PCAFReportTemplate()
        data = {
            "institution_name": "GL Test Bank",
            "reporting_period": "2025",
            "total_financed_emissions": 12500.0,
            "waci": 145.3,
            "data_quality_score": 2.8,
        }
        result = t.render_markdown(data)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_render_json_returns_dict(self):
        """render_json returns a dictionary."""
        t = PCAFReportTemplate()
        data = {"institution_name": "GL Test Bank", "reporting_period": "2025"}
        result = t.render_json(data)
        assert isinstance(result, dict)


class TestGARBTARReportTemplate:
    """Tests for the GAR/BTAR disclosure template."""

    def test_instantiation(self):
        """Template can be instantiated."""
        t = GARBTARReportTemplate()
        assert t is not None

    def test_render_markdown(self):
        """render_markdown produces output."""
        t = GARBTARReportTemplate()
        data = {"gar_pct": 15.2, "btar_pct": 22.1, "reporting_date": "2025-12-31"}
        result = t.render_markdown(data)
        assert isinstance(result, str)
        assert len(result) > 0


class TestPillar3ESGTemplate:
    """Tests for the Pillar 3 ESG ITS template."""

    def test_instantiation(self):
        """Template can be instantiated."""
        t = Pillar3ESGTemplate()
        assert t is not None

    def test_render_json(self):
        """render_json returns structured data."""
        t = Pillar3ESGTemplate()
        data = {"templates_populated": 8, "institution_name": "GL Bank AG"}
        result = t.render_json(data)
        assert isinstance(result, dict)


class TestClimateRiskReportTemplate:
    """Tests for the climate risk report template."""

    def test_instantiation(self):
        """Template can be instantiated."""
        t = ClimateRiskReportTemplate()
        assert t is not None

    def test_render_markdown(self):
        """render_markdown produces output."""
        t = ClimateRiskReportTemplate()
        data = {"scenarios_tested": 3, "max_credit_loss_pct": 4.2}
        result = t.render_markdown(data)
        assert isinstance(result, str)
        assert len(result) > 0


class TestFSESRSChapterTemplate:
    """Tests for the FI-specific ESRS chapters template."""

    def test_instantiation(self):
        """Template can be instantiated."""
        t = FSESRSChapterTemplate()
        assert t is not None

    def test_render_json(self):
        """render_json returns structured data."""
        t = FSESRSChapterTemplate()
        data = {"material_topics": ["E1", "S1", "G1"]}
        result = t.render_json(data)
        assert isinstance(result, dict)


class TestFinancedEmissionsDashboard:
    """Tests for the financed emissions dashboard template."""

    def test_instantiation(self):
        """Template can be instantiated."""
        t = FinancedEmissionsDashboard()
        assert t is not None

    def test_render_markdown(self):
        """render_markdown produces output."""
        t = FinancedEmissionsDashboard()
        data = {"total_emissions": 15000.0, "waci": 120.5}
        result = t.render_markdown(data)
        assert isinstance(result, str)
        assert len(result) > 0


class TestInsuranceESGTemplate:
    """Tests for the insurance ESG disclosure template."""

    def test_instantiation(self):
        """Template can be instantiated."""
        t = InsuranceESGTemplate()
        assert t is not None

    def test_render_json(self):
        """render_json returns structured data."""
        t = InsuranceESGTemplate()
        data = {"gross_emissions": 5000.0, "lines_of_business": 6}
        result = t.render_json(data)
        assert isinstance(result, dict)


class TestSBTiFIReportTemplate:
    """Tests for the SBTi-FI progress report template."""

    def test_instantiation(self):
        """Template can be instantiated."""
        t = SBTiFIReportTemplate()
        assert t is not None

    def test_render_markdown(self):
        """render_markdown produces output."""
        t = SBTiFIReportTemplate()
        data = {"net_zero_year": 2050, "interim_target_year": 2030}
        result = t.render_markdown(data)
        assert isinstance(result, str)
        assert len(result) > 0
