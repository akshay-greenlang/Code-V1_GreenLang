# -*- coding: utf-8 -*-
"""
PACK-004 CBAM Readiness Pack - Templates Tests (25 tests)

Tests all 8 CBAM templates in markdown, HTML, and JSON formats,
plus provenance hash verification and template registry tests.

Author: GreenLang QA Team
"""

import json
from typing import Any, Dict, List

import pytest

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from conftest import CBAM_TEMPLATE_IDS, _compute_hash, render_template_stub


# ---------------------------------------------------------------------------
# Template test data
# ---------------------------------------------------------------------------

TEMPLATE_TEST_DATA = {
    "quarterly_report": {
        "report_id": "QR-2026-Q1-001",
        "quarter": "Q1",
        "year": 2026,
        "importer_name": "EuroSteel Imports GmbH",
        "total_imports": 10,
        "total_emissions_tco2e": 5847.5,
        "categories": ["steel", "aluminium", "cement"],
    },
    "annual_declaration": {
        "declaration_id": "AD-2026-001",
        "year": 2026,
        "importer_name": "EuroSteel Imports GmbH",
        "total_annual_emissions_tco2e": 22500.0,
        "certificates_surrendered": 562,
        "cost_eur": 44133.0,
    },
    "certificate_summary": {
        "obligation_id": "OBL-2026-001",
        "year": 2026,
        "gross_obligation_tco2e": 22500.0,
        "free_allocation_pct": 97.5,
        "net_obligation_tco2e": 562.5,
        "estimated_cost_eur": 44133.0,
        "ets_price_eur": 78.50,
    },
    "supplier_data_request": {
        "request_id": "DR-2026-Q1-001",
        "supplier_name": "Eregli Demir ve Celik",
        "reporting_period": "Q1-2026",
        "goods_categories": ["steel"],
        "deadline": "2026-03-15",
        "cn_codes_required": ["7207 11 14", "7208 51 20"],
    },
    "verification_statement": {
        "statement_id": "VS-2026-001",
        "verifier_name": "TUV Rheinland Energy GmbH",
        "opinion": "unqualified",
        "scope": ["steel", "aluminium"],
        "material_findings": 0,
        "observations": 2,
    },
    "deminimis_assessment": {
        "assessment_id": "DMA-2026-001",
        "year": 2026,
        "total_weight_kg": 145000,
        "threshold_kg": 150000,
        "utilization_pct": 96.7,
        "exempt": True,
    },
    "compliance_dashboard": {
        "compliance_score": 95.0,
        "total_rules": 50,
        "rules_passed": 48,
        "rules_failed": 2,
        "categories_assessed": 3,
        "last_updated": "2026-03-14",
    },
    "emission_calculation_detail": {
        "input_id": "EI-001",
        "cn_code": "7207 11 14",
        "goods_category": "steel",
        "origin_country": "TR",
        "weight_tonnes": 500.0,
        "specific_emission": 1.85,
        "total_emissions_tco2e": 925.0,
        "methodology": "actual",
    },
}


# ---------------------------------------------------------------------------
# Per-template markdown, HTML, JSON tests (24 tests: 8 templates x 3 formats)
# ---------------------------------------------------------------------------

class TestTemplateRendering:
    """Test all 8 CBAM templates in 3 output formats."""

    @pytest.mark.parametrize("template_id", CBAM_TEMPLATE_IDS)
    def test_render_markdown(self, template_id):
        """Test template renders to valid markdown."""
        data = TEMPLATE_TEST_DATA[template_id]
        result = render_template_stub(template_id, data, "markdown")
        assert result["format"] == "markdown"
        assert result["template_id"] == template_id
        assert "# " in result["content"]
        assert "Provenance:" in result["content"]
        assert len(result["provenance_hash"]) == 64

    @pytest.mark.parametrize("template_id", CBAM_TEMPLATE_IDS)
    def test_render_html(self, template_id):
        """Test template renders to valid HTML."""
        data = TEMPLATE_TEST_DATA[template_id]
        result = render_template_stub(template_id, data, "html")
        assert result["format"] == "html"
        assert result["template_id"] == template_id
        assert "<html>" in result["content"]
        assert "<h1>" in result["content"]
        assert "Provenance:" in result["content"]
        assert len(result["provenance_hash"]) == 64

    @pytest.mark.parametrize("template_id", CBAM_TEMPLATE_IDS)
    def test_render_json(self, template_id):
        """Test template renders to valid JSON."""
        data = TEMPLATE_TEST_DATA[template_id]
        result = render_template_stub(template_id, data, "json")
        assert result["format"] == "json"
        assert result["template_id"] == template_id
        parsed = json.loads(result["content"])
        assert "template_id" in parsed
        assert "provenance_hash" in parsed
        assert parsed["template_id"] == template_id


# ---------------------------------------------------------------------------
# Provenance and registry tests
# ---------------------------------------------------------------------------

class TestTemplateProvenance:
    """Test provenance hashing across all templates."""

    def test_provenance_hash_on_all_templates(self):
        """Test every template produces a valid SHA-256 provenance hash."""
        for template_id in CBAM_TEMPLATE_IDS:
            data = TEMPLATE_TEST_DATA[template_id]
            result = render_template_stub(template_id, data, "json")
            assert len(result["provenance_hash"]) == 64
            assert all(c in "0123456789abcdef" for c in result["provenance_hash"])


class TestTemplateRegistry:
    """Test template registry."""

    def test_template_registry_list(self):
        """Test template registry lists all 8 CBAM templates."""
        registry = list(CBAM_TEMPLATE_IDS)
        assert len(registry) == 8
        expected_ids = {
            "quarterly_report",
            "annual_declaration",
            "certificate_summary",
            "supplier_data_request",
            "verification_statement",
            "deminimis_assessment",
            "compliance_dashboard",
            "emission_calculation_detail",
        }
        assert set(registry) == expected_ids
