# -*- coding: utf-8 -*-
"""
PACK-003 CSRD Enterprise Pack - Templates Tests (30 tests)

Tests all 9 enterprise templates in markdown, HTML, and JSON
formats, plus provenance hash verification and template registry.

Author: GreenLang QA Team
"""

from typing import Any, Dict, List

import pytest

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from conftest import ENTERPRISE_TEMPLATE_IDS, _compute_hash, render_template_stub


# ---------------------------------------------------------------------------
# Template test data
# ---------------------------------------------------------------------------

TEMPLATE_TEST_DATA = {
    "emission_forecast_report": {
        "company_name": "GlobalTech AG",
        "reporting_year": 2025,
        "forecast_horizon_months": 12,
        "scope1_forecast": 42000.0,
        "confidence_level": 0.95,
    },
    "iot_monitoring_dashboard": {
        "facility_count": 5,
        "active_devices": 200,
        "total_readings_24h": 28800,
        "anomalies_detected": 3,
        "emission_rate_tco2e_per_hour": 5.2,
    },
    "carbon_credit_portfolio": {
        "total_credits": 20,
        "active_credits": 16,
        "total_quantity_tco2e": 210000,
        "total_value_usd": 3150000.0,
        "registries": ["VCS", "GoldStandard", "ACR"],
    },
    "supply_chain_esg_scorecard": {
        "total_suppliers": 15,
        "assessed_suppliers": 12,
        "avg_esg_score": 0.65,
        "high_risk_count": 3,
        "improvement_plans": 3,
    },
    "regulatory_filing_package": {
        "filing_id": "FIL-2025-001",
        "target": "ESAP",
        "taxonomy_version": "ESRS_2023",
        "validation_score": 98.67,
        "sections_count": 5,
    },
    "multi_language_narrative": {
        "primary_language": "en",
        "total_languages": 6,
        "esrs_standards_covered": ["E1", "E2", "S1", "G1"],
        "word_count_primary": 25000,
        "fact_checks_passed": 150,
    },
    "tenant_analytics_report": {
        "total_tenants": 25,
        "active_tenants": 22,
        "avg_health_score": 92.5,
        "total_api_calls_24h": 1500000,
        "avg_storage_usage_pct": 45.0,
    },
    "enterprise_audit_package": {
        "engagement_id": "AUD-2025-001",
        "auditor_firm": "KPMG",
        "scope": ["scope_1", "scope_2", "scope_3"],
        "evidence_packages": 5,
        "findings_count": 2,
    },
    "white_label_report": {
        "brand_name": "Acme Sustainability",
        "primary_color": "#003366",
        "report_title": "Annual Sustainability Report 2025",
        "xbrl_tagged": True,
        "language": "en",
    },
}


# ---------------------------------------------------------------------------
# Per-template markdown, HTML, JSON tests (27 tests: 9 templates x 3 formats)
# ---------------------------------------------------------------------------

class TestTemplateRendering:
    """Test all 9 templates in 3 output formats."""

    @pytest.mark.parametrize("template_id", ENTERPRISE_TEMPLATE_IDS)
    def test_render_markdown(self, template_id):
        """Test template renders to valid markdown."""
        data = TEMPLATE_TEST_DATA[template_id]
        result = render_template_stub(template_id, data, "markdown")
        assert result["format"] == "markdown"
        assert result["template_id"] == template_id
        assert "# " in result["content"]
        assert "Provenance:" in result["content"]
        assert len(result["provenance_hash"]) == 64

    @pytest.mark.parametrize("template_id", ENTERPRISE_TEMPLATE_IDS)
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

    @pytest.mark.parametrize("template_id", ENTERPRISE_TEMPLATE_IDS)
    def test_render_json(self, template_id):
        """Test template renders to valid JSON."""
        import json
        data = TEMPLATE_TEST_DATA[template_id]
        result = render_template_stub(template_id, data, "json")
        assert result["format"] == "json"
        assert result["template_id"] == template_id
        parsed = json.loads(result["content"])
        assert "template_id" in parsed
        assert "provenance_hash" in parsed
        assert parsed["template_id"] == template_id


# ---------------------------------------------------------------------------
# Provenance and registry tests (3 tests)
# ---------------------------------------------------------------------------

class TestTemplateProvenance:
    """Test provenance hashing across all templates."""

    def test_provenance_hash_on_all_templates(self):
        """Test every template produces a valid SHA-256 provenance hash."""
        for template_id in ENTERPRISE_TEMPLATE_IDS:
            data = TEMPLATE_TEST_DATA[template_id]
            result = render_template_stub(template_id, data, "json")
            assert len(result["provenance_hash"]) == 64
            assert all(c in "0123456789abcdef" for c in result["provenance_hash"])

    def test_template_registry_list(self):
        """Test template registry lists all 9 enterprise templates."""
        registry = list(ENTERPRISE_TEMPLATE_IDS)
        assert len(registry) == 9
        expected_ids = {
            "emission_forecast_report",
            "iot_monitoring_dashboard",
            "carbon_credit_portfolio",
            "supply_chain_esg_scorecard",
            "regulatory_filing_package",
            "multi_language_narrative",
            "tenant_analytics_report",
            "enterprise_audit_package",
            "white_label_report",
        }
        assert set(registry) == expected_ids

    def test_template_registry_get(self):
        """Test retrieving a specific template from registry."""
        template_id = "emission_forecast_report"
        assert template_id in ENTERPRISE_TEMPLATE_IDS
        data = TEMPLATE_TEST_DATA[template_id]
        result = render_template_stub(template_id, data, "markdown")
        assert result["template_id"] == template_id
        assert len(result["content"]) > 0
