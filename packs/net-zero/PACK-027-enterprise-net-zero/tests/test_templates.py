# -*- coding: utf-8 -*-
"""
Test suite for PACK-027 Enterprise Net Zero Pack - Templates.

Tests all enterprise report templates: GHG inventory, SBTi submission,
CDP response, TCFD/ISSB, executive dashboard, supply chain heatmap,
scenario comparison, assurance statement, board climate report, regulatory filings.

Author:  GreenLang Test Engineering
Pack:    PACK-027 Enterprise Net Zero
"""

import sys
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from templates import (
    GHGInventoryReportTemplate,
    SBTiTargetSubmissionTemplate,
    CDPClimateResponseTemplate,
    TCFDReportTemplate,
    ExecutiveDashboardTemplate,
    SupplyChainHeatmapTemplate,
    ScenarioComparisonTemplate,
    AssuranceStatementTemplate,
    BoardClimateReportTemplate,
    SECClimateFilingTemplate,
    CSRDESRSReportTemplate,
    MaterialityAssessmentTemplate,
    TemplateRegistry,
)

# Convenience aliases
GHGInventoryReport = GHGInventoryReportTemplate
SBTiTargetSubmission = SBTiTargetSubmissionTemplate
CDPClimateResponse = CDPClimateResponseTemplate
TCFDReport = TCFDReportTemplate
ExecutiveDashboard = ExecutiveDashboardTemplate
SupplyChainHeatmap = SupplyChainHeatmapTemplate
ScenarioComparison = ScenarioComparisonTemplate
AssuranceStatement = AssuranceStatementTemplate
BoardClimateReport = BoardClimateReportTemplate
RegulatoryFilings = SECClimateFilingTemplate


# ========================================================================
# GHG Inventory Report
# ========================================================================


class TestGHGInventoryReport:
    def test_template_instantiates(self):
        t = GHGInventoryReport()
        assert t is not None

    def test_supported_formats(self):
        t = GHGInventoryReport()
        formats = t.FORMATS
        assert "markdown" in formats
        assert "html" in formats
        assert "json" in formats
        assert "excel" in formats

    def test_has_render_methods(self):
        t = GHGInventoryReport()
        assert hasattr(t, "render_markdown")
        assert hasattr(t, "render_html")
        assert hasattr(t, "render_json")

    def test_config_is_dict(self):
        t = GHGInventoryReport()
        assert isinstance(t.config, dict)

    def test_template_id(self):
        t = GHGInventoryReport()
        assert t.TEMPLATE_ID is not None

    def test_version(self):
        t = GHGInventoryReport()
        assert t.VERSION is not None

    def test_pack_id(self):
        t = GHGInventoryReport()
        assert t.PACK_ID is not None


# ========================================================================
# SBTi Target Submission
# ========================================================================


class TestSBTiTargetSubmission:
    def test_template_instantiates(self):
        t = SBTiTargetSubmission()
        assert t is not None

    def test_supported_formats(self):
        t = SBTiTargetSubmission()
        formats = t.FORMATS
        assert "markdown" in formats
        assert "html" in formats
        assert "json" in formats
        assert "pdf" in formats

    def test_has_render_methods(self):
        t = SBTiTargetSubmission()
        assert hasattr(t, "render_markdown")
        assert hasattr(t, "render_html")

    def test_config_is_dict(self):
        t = SBTiTargetSubmission()
        assert isinstance(t.config, dict)

    def test_template_id(self):
        t = SBTiTargetSubmission()
        assert t.TEMPLATE_ID is not None

    def test_version(self):
        t = SBTiTargetSubmission()
        assert t.VERSION is not None


# ========================================================================
# CDP Climate Response
# ========================================================================


class TestCDPClimateResponse:
    def test_template_instantiates(self):
        t = CDPClimateResponse()
        assert t is not None

    def test_supported_formats(self):
        t = CDPClimateResponse()
        formats = t.FORMATS
        assert "markdown" in formats
        assert "html" in formats
        assert "json" in formats

    def test_has_render_methods(self):
        t = CDPClimateResponse()
        assert hasattr(t, "render_markdown")
        assert hasattr(t, "render_html")

    def test_config_is_dict(self):
        t = CDPClimateResponse()
        assert isinstance(t.config, dict)


# ========================================================================
# TCFD / ISSB S2 Report
# ========================================================================


class TestTCFDReport:
    def test_template_instantiates(self):
        t = TCFDReport()
        assert t is not None

    def test_supported_formats(self):
        t = TCFDReport()
        formats = t.FORMATS
        assert "markdown" in formats
        assert "html" in formats
        assert "json" in formats
        assert "pdf" in formats

    def test_has_render_methods(self):
        t = TCFDReport()
        assert hasattr(t, "render_markdown")
        assert hasattr(t, "render_html")

    def test_config_is_dict(self):
        t = TCFDReport()
        assert isinstance(t.config, dict)


# ========================================================================
# Executive Dashboard
# ========================================================================


class TestExecutiveDashboard:
    def test_template_instantiates(self):
        t = ExecutiveDashboard()
        assert t is not None

    def test_supported_formats(self):
        t = ExecutiveDashboard()
        formats = t.FORMATS
        assert "markdown" in formats
        assert "html" in formats
        assert "json" in formats

    def test_has_render_methods(self):
        t = ExecutiveDashboard()
        assert hasattr(t, "render_markdown")
        assert hasattr(t, "render_html")

    def test_config_is_dict(self):
        t = ExecutiveDashboard()
        assert isinstance(t.config, dict)


# ========================================================================
# Supply Chain Heatmap
# ========================================================================


class TestSupplyChainHeatmap:
    def test_template_instantiates(self):
        t = SupplyChainHeatmap()
        assert t is not None

    def test_supported_formats(self):
        t = SupplyChainHeatmap()
        formats = t.FORMATS
        assert "markdown" in formats
        assert "html" in formats
        assert "json" in formats

    def test_has_render_methods(self):
        t = SupplyChainHeatmap()
        assert hasattr(t, "render_markdown")
        assert hasattr(t, "render_html")

    def test_config_is_dict(self):
        t = SupplyChainHeatmap()
        assert isinstance(t.config, dict)


# ========================================================================
# Scenario Comparison
# ========================================================================


class TestScenarioComparison:
    def test_template_instantiates(self):
        t = ScenarioComparison()
        assert t is not None

    def test_supported_formats(self):
        t = ScenarioComparison()
        formats = t.FORMATS
        assert "markdown" in formats
        assert "html" in formats
        assert "json" in formats

    def test_has_render_methods(self):
        t = ScenarioComparison()
        assert hasattr(t, "render_markdown")
        assert hasattr(t, "render_html")

    def test_config_is_dict(self):
        t = ScenarioComparison()
        assert isinstance(t.config, dict)


# ========================================================================
# Assurance Statement
# ========================================================================


class TestAssuranceStatement:
    def test_template_instantiates(self):
        t = AssuranceStatement()
        assert t is not None

    def test_supported_formats(self):
        t = AssuranceStatement()
        formats = t.FORMATS
        assert "markdown" in formats
        assert "html" in formats
        assert "json" in formats
        assert "pdf" in formats

    def test_has_render_methods(self):
        t = AssuranceStatement()
        assert hasattr(t, "render_markdown")
        assert hasattr(t, "render_html")

    def test_config_is_dict(self):
        t = AssuranceStatement()
        assert isinstance(t.config, dict)


# ========================================================================
# Board Climate Report
# ========================================================================


class TestBoardClimateReport:
    def test_template_instantiates(self):
        t = BoardClimateReport()
        assert t is not None

    def test_supported_formats(self):
        t = BoardClimateReport()
        formats = t.FORMATS
        assert "markdown" in formats
        assert "html" in formats
        assert "json" in formats
        assert "pdf" in formats

    def test_has_render_methods(self):
        t = BoardClimateReport()
        assert hasattr(t, "render_markdown")
        assert hasattr(t, "render_html")

    def test_config_is_dict(self):
        t = BoardClimateReport()
        assert isinstance(t.config, dict)


# ========================================================================
# Regulatory Filings (SEC Climate Filing)
# ========================================================================


class TestRegulatoryFilings:
    def test_template_instantiates(self):
        t = RegulatoryFilings()
        assert t is not None

    def test_supported_formats(self):
        t = RegulatoryFilings()
        formats = t.FORMATS
        assert "markdown" in formats
        assert "html" in formats
        assert "json" in formats

    def test_has_render_methods(self):
        t = RegulatoryFilings()
        assert hasattr(t, "render_markdown")
        assert hasattr(t, "render_html")

    def test_config_is_dict(self):
        t = RegulatoryFilings()
        assert isinstance(t.config, dict)


# ========================================================================
# Additional Templates (CSRD ESRS, Materiality Assessment)
# ========================================================================


class TestCSRDESRSReport:
    def test_template_instantiates(self):
        t = CSRDESRSReportTemplate()
        assert t is not None

    def test_supported_formats(self):
        t = CSRDESRSReportTemplate()
        formats = t.FORMATS
        assert "markdown" in formats
        assert "html" in formats

    def test_has_render_methods(self):
        t = CSRDESRSReportTemplate()
        assert hasattr(t, "render_markdown")
        assert hasattr(t, "render_html")


class TestMaterialityAssessment:
    def test_template_instantiates(self):
        t = MaterialityAssessmentTemplate()
        assert t is not None

    def test_supported_formats(self):
        t = MaterialityAssessmentTemplate()
        formats = t.FORMATS
        assert "markdown" in formats
        assert "html" in formats

    def test_has_render_methods(self):
        t = MaterialityAssessmentTemplate()
        assert hasattr(t, "render_markdown")
        assert hasattr(t, "render_html")


# ========================================================================
# Cross-Template Tests
# ========================================================================


ALL_TEMPLATE_CLASSES = [
    GHGInventoryReportTemplate,
    SBTiTargetSubmissionTemplate,
    CDPClimateResponseTemplate,
    TCFDReportTemplate,
    ExecutiveDashboardTemplate,
    SupplyChainHeatmapTemplate,
    ScenarioComparisonTemplate,
    AssuranceStatementTemplate,
    BoardClimateReportTemplate,
    SECClimateFilingTemplate,
    CSRDESRSReportTemplate,
    MaterialityAssessmentTemplate,
]


class TestCrossTemplateConsistency:
    def test_all_templates_instantiate(self):
        for tmpl_cls in ALL_TEMPLATE_CLASSES:
            t = tmpl_cls()
            assert t is not None

    def test_all_templates_have_render_markdown(self):
        for tmpl_cls in ALL_TEMPLATE_CLASSES:
            t = tmpl_cls()
            assert hasattr(t, "render_markdown")

    def test_all_templates_have_render_html(self):
        for tmpl_cls in ALL_TEMPLATE_CLASSES:
            t = tmpl_cls()
            assert hasattr(t, "render_html")

    def test_all_templates_have_formats(self):
        for tmpl_cls in ALL_TEMPLATE_CLASSES:
            t = tmpl_cls()
            assert hasattr(t, "FORMATS")
            assert len(t.FORMATS) >= 2

    def test_all_templates_have_config(self):
        for tmpl_cls in ALL_TEMPLATE_CLASSES:
            t = tmpl_cls()
            assert hasattr(t, "config")

    def test_template_registry_count(self):
        registry = TemplateRegistry()
        templates = registry.list_templates()
        assert len(templates) >= 10

    def test_template_registry_get(self):
        registry = TemplateRegistry()
        # Should be able to get a template by name
        t = registry.get_template("ghg_inventory_report")
        assert t is not None

    def test_template_registry_categories(self):
        registry = TemplateRegistry()
        cats = registry.categories
        assert len(cats) >= 3
