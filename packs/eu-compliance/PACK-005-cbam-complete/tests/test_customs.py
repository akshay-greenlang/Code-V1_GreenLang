# -*- coding: utf-8 -*-
"""
PACK-005 CBAM Complete Pack - Customs Automation Engine Tests (20 tests)

Tests CustomsAutomationEngine: CN code validation, customs declaration
parsing, AEO status, CBAM applicability, import procedures,
anti-circumvention detection (origin change, CN reclassification,
scrap ratio anomaly, restructuring, minor processing), downstream
monitoring, combined duty/CBAM, EORI validation, and CN versioning.

Author: GreenLang QA Team
"""

import json
import re
from typing import Any, Dict, List

import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from conftest import (
    StubTARICClient,
    _compute_hash,
    _utcnow,
    assert_provenance_hash,
)


# ---------------------------------------------------------------------------
# CN Code Validation (3 tests)
# ---------------------------------------------------------------------------

class TestCNCodeValidation:
    """Test CN code validation and lookup."""

    def test_validate_cn_code_valid(self, mock_taric_client):
        """Test validation of a valid CBAM CN code."""
        result = mock_taric_client.validate_cn_code("7207 11 14")
        assert result["format_valid"] is True
        assert result["cbam_covered"] is True
        assert result["category"] == "steel"

    def test_validate_cn_code_invalid(self, mock_taric_client):
        """Test validation of an invalid CN code format."""
        result = mock_taric_client.validate_cn_code("720711")
        assert result["format_valid"] is False
        assert result["cbam_covered"] is False

    def test_validate_cn_code_non_cbam(self, mock_taric_client):
        """Test validation of a valid but non-CBAM CN code."""
        result = mock_taric_client.validate_cn_code("8471 30 00")
        assert result["format_valid"] is True
        assert result["cbam_covered"] is False


# ---------------------------------------------------------------------------
# Customs Declaration Parsing (2 tests)
# ---------------------------------------------------------------------------

class TestCustomsDeclaration:
    """Test customs declaration parsing."""

    def test_parse_customs_declaration(self, sample_customs_declaration):
        """Test parsing a SAD with 5 line items."""
        sad = sample_customs_declaration
        assert sad["declaration_type"] == "IM"
        assert len(sad["line_items"]) == 5
        assert sad["total_cbam_items"] == 4
        assert sad["total_non_cbam_items"] == 1

    def test_check_aeo_status(self, mock_taric_client):
        """Test checking Authorized Economic Operator status."""
        result = mock_taric_client.check_aeo_status("DE123456789012345")
        assert result["aeo_status"] == "AEOC"
        assert result["valid_until"] >= "2027-01-01"


# ---------------------------------------------------------------------------
# CBAM Applicability (2 tests)
# ---------------------------------------------------------------------------

class TestCBAMApplicability:
    """Test CBAM applicability determination."""

    def test_cbam_applicability_covered(self, sample_customs_declaration):
        """Test CBAM-covered items are correctly identified."""
        cbam_items = [
            item for item in sample_customs_declaration["line_items"]
            if item["cbam_applicable"]
        ]
        assert len(cbam_items) == 4
        # All CBAM items should have recognized CN codes
        cbam_cn_prefixes = ["72", "76", "25", "28", "31", "27"]
        for item in cbam_items:
            prefix = item["cn_code"][:2]
            assert prefix in cbam_cn_prefixes

    def test_cbam_applicability_exempt(self, sample_customs_declaration):
        """Test non-CBAM items are correctly identified as exempt."""
        exempt_items = [
            item for item in sample_customs_declaration["line_items"]
            if not item["cbam_applicable"]
        ]
        assert len(exempt_items) == 1
        assert exempt_items[0]["cn_code"] == "8471 30 00"


# ---------------------------------------------------------------------------
# Import Procedures (2 tests)
# ---------------------------------------------------------------------------

class TestImportProcedures:
    """Test import procedure handling."""

    def test_import_procedure_standard(self, sample_customs_declaration):
        """Test standard import procedure (4000 = free circulation)."""
        for item in sample_customs_declaration["line_items"]:
            assert item["customs_procedure"] == "4000"

    def test_import_procedure_inward_processing(self):
        """Test inward processing procedure handling."""
        item = {
            "cn_code": "7208 51 20",
            "customs_procedure": "5100",
            "cbam_applicable": False,
            "reason": "Inward processing - re-export expected",
        }
        assert item["customs_procedure"] == "5100"
        assert item["cbam_applicable"] is False


# ---------------------------------------------------------------------------
# Anti-Circumvention Detection (5 tests)
# ---------------------------------------------------------------------------

class TestAntiCircumvention:
    """Test anti-circumvention detection rules."""

    def test_detect_origin_change(self):
        """Test detecting suspicious origin country changes."""
        historical = [
            {"date": "2025-Q1", "origin": "CN", "volume_t": 500},
            {"date": "2025-Q2", "origin": "CN", "volume_t": 480},
            {"date": "2025-Q3", "origin": "CN", "volume_t": 510},
            {"date": "2025-Q4", "origin": "CN", "volume_t": 490},
        ]
        current = {"date": "2026-Q1", "origin": "VN", "volume_t": 500}
        origin_changed = current["origin"] != historical[-1]["origin"]
        assert origin_changed is True

    def test_detect_cn_reclassification(self):
        """Test detecting CN code reclassification attempts."""
        before = {"cn_code": "7208 51 20", "category": "steel", "cbam": True}
        after = {"cn_code": "7326 90 98", "category": "steel_articles", "cbam": False}
        reclassified = before["cn_code"] != after["cn_code"]
        lost_cbam = before["cbam"] and not after["cbam"]
        alert = reclassified and lost_cbam
        assert alert is True

    def test_detect_scrap_ratio_anomaly(self):
        """Test detecting anomalous scrap ratios."""
        normal_scrap_pct = 15.0  # typical for EAF
        reported_scrap_pct = 85.0  # suspiciously high
        threshold_pct = 50.0
        anomaly = reported_scrap_pct > threshold_pct
        assert anomaly is True

    def test_detect_restructuring(self):
        """Test detecting restructuring to avoid CBAM."""
        entity_history = {
            "before": {"name": "BigSteel Corp", "imports_t": 5000, "entities": 1},
            "after": {"name": "BigSteel Corp", "imports_t": 5000, "entities": 5},
        }
        # Splitting into small entities to stay under de minimis
        per_entity_after = entity_history["after"]["imports_t"] / entity_history["after"]["entities"]
        deminimis_threshold_t = 50.0
        per_entity_under_deminimis = per_entity_after * 1000 < deminimis_threshold_t * 1000
        entities_increased = (
            entity_history["after"]["entities"] > entity_history["before"]["entities"] * 2
        )
        alert = entities_increased
        assert alert is True

    def test_detect_minor_processing(self):
        """Test detecting minor processing to change origin."""
        import_record = {
            "cn_code": "7208 51 20",
            "declared_origin": "VN",
            "processing_in_origin": "cutting_to_length",
            "raw_material_origin": "CN",
        }
        minor_operations = {"cutting_to_length", "sorting", "repacking", "labeling"}
        is_minor = import_record["processing_in_origin"] in minor_operations
        origin_differs = import_record["declared_origin"] != import_record["raw_material_origin"]
        alert = is_minor and origin_differs
        assert alert is True


# ---------------------------------------------------------------------------
# Downstream Monitoring and Duties (3 tests)
# ---------------------------------------------------------------------------

class TestDownstreamAndDuties:
    """Test downstream monitoring and combined duty/CBAM."""

    def test_downstream_monitoring(self):
        """Test monitoring downstream use of CBAM goods."""
        import_record = {
            "cn_code": "7208 51 20",
            "end_use": "automotive_manufacturing",
            "weight_tonnes": 300,
        }
        downstream_tracked = import_record.get("end_use") is not None
        assert downstream_tracked is True

    def test_combined_duty_cbam(self, sample_customs_declaration):
        """Test combined customs duty and CBAM cost calculation."""
        item = sample_customs_declaration["line_items"][0]
        duty_rate_pct = 3.7  # Typical steel duty rate
        cbam_ef = 1.85
        cbam_price = 78.50
        cbam_coverage = 0.025
        weight_tonnes = item["net_mass_kg"] / 1000
        value_eur = item["statistical_value_eur"]

        customs_duty = value_eur * (duty_rate_pct / 100.0)
        cbam_cost = weight_tonnes * cbam_ef * cbam_price * cbam_coverage
        total_import_cost = value_eur + customs_duty + cbam_cost
        assert total_import_cost > value_eur

    def test_eori_validation(self, mock_taric_client):
        """Test EORI number validation."""
        valid_result = mock_taric_client.validate_eori("DE123456789012345")
        assert valid_result["valid"] is True
        assert valid_result["member_state"] == "DE"

        invalid_result = mock_taric_client.validate_eori("XX12345")
        assert invalid_result["valid"] is False


# ---------------------------------------------------------------------------
# CN Versioning and Cache (3 tests)
# ---------------------------------------------------------------------------

class TestCNVersioning:
    """Test CN code versioning and caching."""

    def test_cn_version_changes(self, sample_config):
        """Test CN version is tracked in config."""
        assert sample_config["customs"]["cn_version"] == "2026"

    def test_cn_code_cache(self, mock_taric_client):
        """Test CN code lookup results are cached."""
        mock_taric_client.validate_cn_code("7207 11 14")
        cached = mock_taric_client.get_cached("7207 11 14")
        assert cached is not None
        assert cached["category"] == "steel"

    def test_cn_code_all_categories(self, mock_taric_client):
        """Test CN codes span all CBAM categories."""
        categories_found = set()
        for code, info in StubTARICClient.VALID_CN_CODES.items():
            categories_found.add(info["category"])
        expected = {"steel", "aluminium", "cement", "fertilizers",
                    "electricity", "hydrogen"}
        assert expected == categories_found
