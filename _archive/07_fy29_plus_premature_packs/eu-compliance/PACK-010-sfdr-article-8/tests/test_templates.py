# -*- coding: utf-8 -*-
"""
PACK-010 SFDR Article 8 Pack - Template Tests
================================================

Tests all 8 SFDR templates and the TemplateRegistry:
AnnexIIPrecontractual, AnnexIVPeriodic, AnnexIIIWebsite,
PAIStatement, PortfolioESGDashboard, TaxonomyAlignmentReport,
ExecutiveSummary, AuditTrailReport, and TemplateRegistry.

Self-contained: does NOT import from conftest.
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
# Import the templates __init__ (which re-exports everything)
# ---------------------------------------------------------------------------

TPL_DIR = str(PACK_ROOT / "templates")

_tpl_init = _import_from_path(
    "pack010_tpl_init",
    os.path.join(TPL_DIR, "__init__.py"),
)

# TemplateRegistry
TemplateRegistry = _tpl_init.TemplateRegistry

# Individual template files
_tpl_annex_ii = _import_from_path(
    "pack010_tpl_annex_ii",
    os.path.join(TPL_DIR, "annex_ii_precontractual.py"),
)
_tpl_annex_iv = _import_from_path(
    "pack010_tpl_annex_iv",
    os.path.join(TPL_DIR, "annex_iv_periodic.py"),
)
_tpl_annex_iii = _import_from_path(
    "pack010_tpl_annex_iii",
    os.path.join(TPL_DIR, "annex_iii_website.py"),
)
_tpl_pai = _import_from_path(
    "pack010_tpl_pai",
    os.path.join(TPL_DIR, "pai_statement_template.py"),
)
_tpl_dashboard = _import_from_path(
    "pack010_tpl_dashboard",
    os.path.join(TPL_DIR, "portfolio_esg_dashboard.py"),
)
_tpl_taxonomy = _import_from_path(
    "pack010_tpl_taxonomy",
    os.path.join(TPL_DIR, "taxonomy_alignment_report.py"),
)
_tpl_exec = _import_from_path(
    "pack010_tpl_exec",
    os.path.join(TPL_DIR, "executive_summary.py"),
)
_tpl_audit = _import_from_path(
    "pack010_tpl_audit",
    os.path.join(TPL_DIR, "audit_trail_report.py"),
)

# Template classes
AnnexIIPrecontractualTemplate = _tpl_annex_ii.AnnexIIPrecontractualTemplate
AnnexIVPeriodicTemplate = _tpl_annex_iv.AnnexIVPeriodicTemplate
AnnexIIIWebsiteTemplate = _tpl_annex_iii.AnnexIIIWebsiteTemplate
PAIStatementTemplate = _tpl_pai.PAIStatementTemplate
PortfolioESGDashboardTemplate = _tpl_dashboard.PortfolioESGDashboardTemplate
TaxonomyAlignmentReportTemplate = _tpl_taxonomy.TaxonomyAlignmentReportTemplate
ExecutiveSummaryTemplate = _tpl_exec.ExecutiveSummaryTemplate
AuditTrailReportTemplate = _tpl_audit.AuditTrailReportTemplate

# Data models for templates
PrecontractualData = _tpl_annex_ii.PrecontractualData
ProductInfo = _tpl_annex_ii.ProductInfo

PeriodicData = _tpl_annex_iv.PeriodicData
ReportingPeriod = _tpl_annex_iv.ReportingPeriod

WebsiteDisclosureData = _tpl_annex_iii.WebsiteDisclosureData
PAIStatementData = _tpl_pai.PAIStatementData
DashboardData = _tpl_dashboard.DashboardData
AlignmentReportData = _tpl_taxonomy.AlignmentReportData
ExecutiveSummaryData = _tpl_exec.ExecutiveSummaryData
AuditTrailData = _tpl_audit.AuditTrailData


# ---------------------------------------------------------------------------
# TemplateRegistry Tests
# ---------------------------------------------------------------------------


class TestTemplateRegistry:
    """Tests for the TemplateRegistry that manages all 8 templates."""

    def test_registry_instantiation(self):
        """TemplateRegistry can be instantiated."""
        registry = TemplateRegistry()
        assert registry is not None

    def test_template_count(self):
        """Registry has exactly 8 templates."""
        registry = TemplateRegistry()
        assert registry.template_count == 8

    def test_list_templates_returns_all_8(self):
        """list_templates returns all 8 template keys."""
        registry = TemplateRegistry()
        templates = registry.list_templates()
        assert len(templates) == 8

    def test_all_template_keys_present(self):
        """All expected template keys exist in the registry."""
        registry = TemplateRegistry()
        expected_keys = [
            "annex_ii_precontractual",
            "annex_iv_periodic",
            "annex_iii_website",
            "pai_statement",
            "portfolio_esg_dashboard",
            "taxonomy_alignment_report",
            "executive_summary",
            "audit_trail_report",
        ]
        all_keys = registry.get_all_template_keys()
        for key in expected_keys:
            assert key in all_keys, f"Missing template key: {key}"

    def test_has_template(self):
        """has_template returns True for known templates."""
        registry = TemplateRegistry()
        assert registry.has_template("annex_ii_precontractual") is True
        assert registry.has_template("nonexistent_template") is False

    def test_pack_metadata(self):
        """Registry exposes pack_id and pack_name."""
        registry = TemplateRegistry()
        assert registry.pack_id == "PACK-010"
        assert registry.pack_name == "SFDR Article 8"


# ---------------------------------------------------------------------------
# Individual Template Tests
# ---------------------------------------------------------------------------


class TestAnnexIIPrecontractualTemplate:
    """Tests for AnnexIIPrecontractualTemplate."""

    def test_instantiation(self):
        """Template can be instantiated with default config."""
        tpl = AnnexIIPrecontractualTemplate(config={})
        assert tpl is not None

    def test_render_markdown(self):
        """render_markdown produces non-empty string output."""
        tpl = AnnexIIPrecontractualTemplate(config={})
        data = PrecontractualData(
            product_info=ProductInfo(product_name="GL Green Bond Fund"),
        ).model_dump()
        output = tpl.render_markdown(data)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_render_json(self):
        """render_json produces dict output."""
        tpl = AnnexIIPrecontractualTemplate(config={})
        data = PrecontractualData(
            product_info=ProductInfo(product_name="GL Green Bond Fund"),
        ).model_dump()
        output = tpl.render_json(data)
        assert isinstance(output, dict)
        assert len(output) > 0


class TestAnnexIVPeriodicTemplate:
    """Tests for AnnexIVPeriodicTemplate."""

    def test_instantiation(self):
        """Template can be instantiated with default config."""
        tpl = AnnexIVPeriodicTemplate(config={})
        assert tpl is not None

    def test_render_markdown(self):
        """render_markdown produces non-empty string output."""
        tpl = AnnexIVPeriodicTemplate(config={})
        data = PeriodicData(
            reporting_period=ReportingPeriod(
                start_date="2025-01-01",
                end_date="2025-12-31",
                fund_name="GL ESG Fund",
            ),
        ).model_dump()
        output = tpl.render_markdown(data)
        assert isinstance(output, str)
        assert len(output) > 0


class TestAnnexIIIWebsiteTemplate:
    """Tests for AnnexIIIWebsiteTemplate."""

    def test_instantiation_and_render(self):
        """Template instantiates and renders markdown output."""
        tpl = AnnexIIIWebsiteTemplate(config={})
        data = WebsiteDisclosureData(
            product_info=_tpl_annex_iii.WebProductInfo(product_name="GL ESG Fund"),
        ).model_dump()
        output = tpl.render_markdown(data)
        assert isinstance(output, str)
        assert len(output) > 0


class TestPAIStatementTemplate:
    """Tests for PAIStatementTemplate."""

    def test_instantiation_and_render(self):
        """Template instantiates and renders markdown output."""
        tpl = PAIStatementTemplate(config={})
        data = PAIStatementData(
            entity_name="GL Asset Management",
            reporting_period_start="2025-01-01",
            reporting_period_end="2025-12-31",
        ).model_dump()
        output = tpl.render_markdown(data)
        assert isinstance(output, str)
        assert len(output) > 0


class TestPortfolioESGDashboardTemplate:
    """Tests for PortfolioESGDashboardTemplate."""

    def test_instantiation_and_render_html(self):
        """Template instantiates and renders HTML output."""
        tpl = PortfolioESGDashboardTemplate(config={})
        data = DashboardData(
            fund_overview=_tpl_dashboard.FundOverview(fund_name="GL ESG Fund"),
        ).model_dump()
        output = tpl.render_html(data)
        assert isinstance(output, str)
        assert len(output) > 0


class TestTaxonomyAlignmentReportTemplate:
    """Tests for TaxonomyAlignmentReportTemplate."""

    def test_instantiation_and_render(self):
        """Template instantiates and renders markdown output."""
        tpl = TaxonomyAlignmentReportTemplate(config={})
        data = AlignmentReportData(
            fund_name="GL Taxonomy Aligned Fund",
        ).model_dump()
        output = tpl.render_markdown(data)
        assert isinstance(output, str)
        assert len(output) > 0


class TestExecutiveSummaryTemplate:
    """Tests for ExecutiveSummaryTemplate."""

    def test_instantiation_and_render(self):
        """Template instantiates and renders markdown output."""
        tpl = ExecutiveSummaryTemplate(config={})
        data = ExecutiveSummaryData(
            fund_info=_tpl_exec.FundInfo(fund_name="GL ESG Fund"),
        ).model_dump()
        output = tpl.render_markdown(data)
        assert isinstance(output, str)
        assert len(output) > 0


class TestAuditTrailReportTemplate:
    """Tests for AuditTrailReportTemplate."""

    def test_instantiation_and_render(self):
        """Template instantiates and renders markdown output."""
        tpl = AuditTrailReportTemplate(config={})
        data = AuditTrailData(
            fund_name="GL Audit Fund",
            reporting_period="2025-01-01 to 2025-12-31",
        ).model_dump()
        output = tpl.render_markdown(data)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_render_json(self):
        """render_json produces dict output."""
        tpl = AuditTrailReportTemplate(config={})
        data = AuditTrailData(
            fund_name="GL Audit Fund",
            reporting_period="2025-01-01 to 2025-12-31",
        ).model_dump()
        output = tpl.render_json(data)
        assert isinstance(output, dict)
        assert len(output) > 0
