# -*- coding: utf-8 -*-
"""
PACK-005 CBAM Complete Pack - Templates Tests (25 tests)

Tests all 6 PACK-005 templates in markdown, HTML, and JSON formats,
plus provenance hash verification, template registry listing, and
render dispatch.

Author: GreenLang QA Team
"""

import json
from typing import Any, Dict, List

import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from conftest import (
    PACK005_TEMPLATE_IDS,
    _compute_hash,
    render_template_stub,
)


# ---------------------------------------------------------------------------
# Template test data per template ID
# ---------------------------------------------------------------------------

TEMPLATE_TEST_DATA = {
    "certificate_portfolio_report": {
        "portfolio_id": "PF-EUROSTEEL-2026",
        "total_certificates": 50,
        "active_certificates": 30,
        "total_value_eur": 2350.00,
        "valuation_method": "FIFO",
        "year": 2026,
    },
    "group_consolidation_report": {
        "group_id": "GRP-EUROSTEEL-001",
        "entities": 3,
        "total_emissions_tco2e": 22500.0,
        "total_certificates_required": 563,
        "consolidation_method": "volume",
        "year": 2026,
    },
    "sourcing_scenario_analysis": {
        "analysis_id": "SSA-2026-001",
        "scenarios_count": 5,
        "best_scenario": "optimized_mix",
        "best_cost_eur": 42000.0,
        "worst_cost_eur": 58000.0,
        "confidence_level": 0.95,
    },
    "cross_regulation_mapping_report": {
        "cbam_emissions_tco2e": 22500.0,
        "frameworks_mapped": 6,
        "data_fields_reused": 12,
        "consistency_issues": 0,
        "year": 2026,
    },
    "customs_integration_report": {
        "declarations_processed": 15,
        "cbam_items_identified": 42,
        "non_cbam_items": 8,
        "anti_circumvention_alerts": 2,
        "year": 2026,
    },
    "audit_readiness_scorecard": {
        "overall_score": 95.0,
        "evidence_complete": True,
        "verification_obtained": True,
        "nca_package_ready": True,
        "open_findings": 0,
        "year": 2026,
    },
}


# ---------------------------------------------------------------------------
# Per-template format tests (18 tests: 6 templates x 3 formats)
# ---------------------------------------------------------------------------

class TestCertificatePortfolioReportTemplate:
    """Test certificate portfolio report template."""

    def test_render_markdown(self, template_renderer):
        data = TEMPLATE_TEST_DATA["certificate_portfolio_report"]
        result = template_renderer("certificate_portfolio_report", data, "markdown")
        assert "# Certificate Portfolio Report" in result["content"]
        assert "portfolio_id" in result["content"]

    def test_render_html(self, template_renderer):
        data = TEMPLATE_TEST_DATA["certificate_portfolio_report"]
        result = template_renderer("certificate_portfolio_report", data, "html")
        assert "<html>" in result["content"]
        assert "Certificate Portfolio Report" in result["content"]

    def test_render_json(self, template_renderer):
        data = TEMPLATE_TEST_DATA["certificate_portfolio_report"]
        result = template_renderer("certificate_portfolio_report", data, "json")
        parsed = json.loads(result["content"])
        assert parsed["template_id"] == "certificate_portfolio_report"


class TestGroupConsolidationReportTemplate:
    """Test group consolidation report template."""

    def test_render_markdown(self, template_renderer):
        data = TEMPLATE_TEST_DATA["group_consolidation_report"]
        result = template_renderer("group_consolidation_report", data, "markdown")
        assert "Group Consolidation Report" in result["content"]

    def test_render_html(self, template_renderer):
        data = TEMPLATE_TEST_DATA["group_consolidation_report"]
        result = template_renderer("group_consolidation_report", data, "html")
        assert "<html>" in result["content"]

    def test_render_json(self, template_renderer):
        data = TEMPLATE_TEST_DATA["group_consolidation_report"]
        result = template_renderer("group_consolidation_report", data, "json")
        parsed = json.loads(result["content"])
        assert parsed["template_id"] == "group_consolidation_report"


class TestSourcingScenarioAnalysisTemplate:
    """Test sourcing scenario analysis template."""

    def test_render_markdown(self, template_renderer):
        data = TEMPLATE_TEST_DATA["sourcing_scenario_analysis"]
        result = template_renderer("sourcing_scenario_analysis", data, "markdown")
        assert "Sourcing Scenario Analysis" in result["content"]

    def test_render_html(self, template_renderer):
        data = TEMPLATE_TEST_DATA["sourcing_scenario_analysis"]
        result = template_renderer("sourcing_scenario_analysis", data, "html")
        assert "<html>" in result["content"]

    def test_render_json(self, template_renderer):
        data = TEMPLATE_TEST_DATA["sourcing_scenario_analysis"]
        result = template_renderer("sourcing_scenario_analysis", data, "json")
        parsed = json.loads(result["content"])
        assert parsed["template_id"] == "sourcing_scenario_analysis"


class TestCrossRegulationMappingTemplate:
    """Test cross-regulation mapping report template."""

    def test_render_markdown(self, template_renderer):
        data = TEMPLATE_TEST_DATA["cross_regulation_mapping_report"]
        result = template_renderer("cross_regulation_mapping_report", data, "markdown")
        assert "Cross Regulation Mapping Report" in result["content"]

    def test_render_html(self, template_renderer):
        data = TEMPLATE_TEST_DATA["cross_regulation_mapping_report"]
        result = template_renderer("cross_regulation_mapping_report", data, "html")
        assert "<html>" in result["content"]

    def test_render_json(self, template_renderer):
        data = TEMPLATE_TEST_DATA["cross_regulation_mapping_report"]
        result = template_renderer("cross_regulation_mapping_report", data, "json")
        parsed = json.loads(result["content"])
        assert parsed["template_id"] == "cross_regulation_mapping_report"


class TestCustomsIntegrationReportTemplate:
    """Test customs integration report template."""

    def test_render_markdown(self, template_renderer):
        data = TEMPLATE_TEST_DATA["customs_integration_report"]
        result = template_renderer("customs_integration_report", data, "markdown")
        assert "Customs Integration Report" in result["content"]

    def test_render_html(self, template_renderer):
        data = TEMPLATE_TEST_DATA["customs_integration_report"]
        result = template_renderer("customs_integration_report", data, "html")
        assert "<html>" in result["content"]

    def test_render_json(self, template_renderer):
        data = TEMPLATE_TEST_DATA["customs_integration_report"]
        result = template_renderer("customs_integration_report", data, "json")
        parsed = json.loads(result["content"])
        assert parsed["template_id"] == "customs_integration_report"


class TestAuditReadinessScorecardTemplate:
    """Test audit readiness scorecard template."""

    def test_render_markdown(self, template_renderer):
        data = TEMPLATE_TEST_DATA["audit_readiness_scorecard"]
        result = template_renderer("audit_readiness_scorecard", data, "markdown")
        assert "Audit Readiness Scorecard" in result["content"]

    def test_render_html(self, template_renderer):
        data = TEMPLATE_TEST_DATA["audit_readiness_scorecard"]
        result = template_renderer("audit_readiness_scorecard", data, "html")
        assert "<html>" in result["content"]

    def test_render_json(self, template_renderer):
        data = TEMPLATE_TEST_DATA["audit_readiness_scorecard"]
        result = template_renderer("audit_readiness_scorecard", data, "json")
        parsed = json.loads(result["content"])
        assert parsed["template_id"] == "audit_readiness_scorecard"


# ---------------------------------------------------------------------------
# Provenance Hash Tests (6 tests)
# ---------------------------------------------------------------------------

class TestTemplateProvenance:
    """Test provenance hashes on all templates."""

    @pytest.mark.parametrize("template_id", PACK005_TEMPLATE_IDS)
    def test_provenance_hash(self, template_renderer, template_id):
        """Test each template produces a provenance hash."""
        data = TEMPLATE_TEST_DATA.get(template_id, {"test": True})
        result = template_renderer(template_id, data, "markdown")
        assert "provenance_hash" in result
        h = result["provenance_hash"]
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# Registry Tests (1 test)
# ---------------------------------------------------------------------------

class TestTemplateRegistry:
    """Test template registry operations."""

    def test_template_registry_lists_all(self, template_ids):
        """Test all 6 templates are in the registry."""
        assert len(template_ids) == 6
        for tid in PACK005_TEMPLATE_IDS:
            assert tid in template_ids
