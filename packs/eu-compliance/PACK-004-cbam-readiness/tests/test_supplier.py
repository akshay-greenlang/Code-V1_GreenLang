# -*- coding: utf-8 -*-
"""
PACK-004 CBAM Readiness Pack - Supplier Management Tests (15 tests)

Tests supplier registration, EORI validation, installation management,
emission data submission, review process, quality scoring, and
supplier portal functionality.

Author: GreenLang QA Team
"""

import re
from typing import Any, Dict, List

import pytest

import sys, os
from greenlang.schemas import utcnow
sys.path.insert(0, os.path.dirname(__file__))

from conftest import (

    StubSupplierPortal,
    _compute_hash,
    _new_uuid,
    _utcnow,
)


@pytest.fixture
def portal():
    """Create a StubSupplierPortal instance."""
    return StubSupplierPortal()


class TestSupplierRegistration:
    """Test supplier registration and validation."""

    def test_register_supplier(self, portal):
        """Test supplier registration with valid data."""
        result = portal.register_supplier({
            "supplier_id": "SUP-REG-001",
            "company_name": "Eregli Steel TR",
            "country": "TR",
        })
        assert result["status"] == "active"
        assert result["supplier_id"] == "SUP-REG-001"
        assert result["company_name"] == "Eregli Steel TR"

    def test_validate_eori(self, sample_suppliers):
        """Test EORI number validation for suppliers."""
        for supplier in sample_suppliers:
            eori = supplier.get("eori_number")
            if eori:
                valid = bool(re.match(r"^[A-Z]{2}\d{13,17}$", eori))
                assert valid, f"Invalid EORI for {supplier['supplier_id']}: {eori}"


class TestInstallationManagement:
    """Test installation management."""

    def test_add_installation(self, portal):
        """Test adding an installation to a supplier."""
        portal.register_supplier({
            "supplier_id": "SUP-INST-001",
            "company_name": "Installation Corp",
            "country": "CN",
        })
        result = portal.add_installation("SUP-INST-001", {
            "installation_id": "INST-ADD-001",
            "name": "Shanghai Steelworks",
        })
        assert result["status"] == "registered"
        assert result["installation_id"] == "INST-ADD-001"

    def test_multi_installation(self, sample_suppliers):
        """Test supplier with multiple installations."""
        multi_install_supplier = next(
            (s for s in sample_suppliers if len(s.get("installations", [])) > 1), None,
        )
        assert multi_install_supplier is not None, (
            "Should have at least one supplier with multiple installations"
        )
        installations = multi_install_supplier["installations"]
        assert len(installations) >= 2
        ids = {inst["installation_id"] for inst in installations}
        assert len(ids) == len(installations), "Installation IDs must be unique"


class TestEmissionDataFlow:
    """Test emission data submission and review."""

    def test_request_emission_data(self, sample_suppliers):
        """Test creating a data request for supplier."""
        supplier = sample_suppliers[0]
        request = {
            "request_id": f"DR-{_new_uuid()[:8]}",
            "supplier_id": supplier["supplier_id"],
            "reporting_period": "Q1-2026",
            "goods_categories": ["steel"],
            "deadline": "2026-03-15",
            "status": "sent",
            "created_at": utcnow().isoformat(),
        }
        assert request["status"] == "sent"
        assert request["supplier_id"] == "SUP-TR-001"

    def test_submit_emission_data(self, portal):
        """Test supplier emission data submission."""
        portal.register_supplier({
            "supplier_id": "SUP-DATA-001",
            "company_name": "Data Corp",
            "country": "TR",
        })
        result = portal.submit_emission_data("SUP-DATA-001", {
            "cn_code": "7207 11 14",
            "goods_category": "steel",
            "weight_tonnes": 500.0,
            "specific_emission_tco2e_per_tonne": 1.85,
            "methodology": "actual",
        })
        assert result["status"] == "submitted"
        assert result["weight_tonnes"] == 500.0

    def test_review_accept(self, portal):
        """Test accepting a supplier submission."""
        result = portal.review_submission(
            "SUB-001", "accepted", "Data verified against documentation",
        )
        assert result["decision"] == "accepted"

    def test_review_reject(self, portal):
        """Test rejecting a supplier submission."""
        result = portal.review_submission(
            "SUB-002", "rejected", "Emission factor not substantiated",
        )
        assert result["decision"] == "rejected"
        assert "reviewed_at" in result


class TestQualityScoring:
    """Test supplier quality scoring."""

    def test_quality_score_excellent(self, portal):
        """Test excellent quality score (>= 85)."""
        portal.register_supplier({
            "supplier_id": "SUP-EXC-001",
            "company_name": "Excellent Corp",
            "country": "TR",
        })
        portal.suppliers["SUP-EXC-001"]["quality_score"] = 92.0
        score = portal.get_quality_score("SUP-EXC-001")
        assert score["quality_score"] == 92.0
        assert score["rating"] == "excellent"

    def test_quality_score_poor(self, portal):
        """Test poor quality score (< 50)."""
        portal.register_supplier({
            "supplier_id": "SUP-POOR-001",
            "company_name": "Poor Data Corp",
            "country": "CN",
        })
        portal.suppliers["SUP-POOR-001"]["quality_score"] = 35.0
        score = portal.get_quality_score("SUP-POOR-001")
        assert score["quality_score"] == 35.0
        assert score["rating"] == "poor"


class TestSupplierDashboard:
    """Test supplier dashboard and history."""

    def test_supplier_dashboard(self, sample_suppliers):
        """Test supplier dashboard summary data."""
        total_suppliers = len(sample_suppliers)
        active = sum(1 for s in sample_suppliers if s["status"] == "active")
        avg_quality = sum(s["quality_score"] for s in sample_suppliers) / total_suppliers
        dashboard = {
            "total_suppliers": total_suppliers,
            "active_suppliers": active,
            "avg_quality_score": round(avg_quality, 1),
            "countries": list({s["country"] for s in sample_suppliers}),
        }
        assert dashboard["total_suppliers"] == 3
        assert dashboard["active_suppliers"] == 3
        assert dashboard["avg_quality_score"] > 0

    def test_submission_history(self, sample_emission_submissions):
        """Test emission submission history tracking."""
        accepted = [
            s for s in sample_emission_submissions if s["review_status"] == "accepted"
        ]
        pending = [
            s for s in sample_emission_submissions if s["review_status"] == "pending"
        ]
        rejected = [
            s for s in sample_emission_submissions if s["review_status"] == "rejected"
        ]
        assert len(accepted) == 3
        assert len(pending) == 1
        assert len(rejected) == 1

    def test_data_exchange(self, sample_emission_submissions):
        """Test supplier data exchange format validation."""
        for sub in sample_emission_submissions:
            assert "cn_code" in sub
            assert "weight_tonnes" in sub
            assert sub["weight_tonnes"] > 0
            assert "specific_emission_tco2e_per_tonne" in sub
            assert sub["specific_emission_tco2e_per_tonne"] > 0

    def test_supplier_filter(self, sample_suppliers):
        """Test filtering suppliers by country and sector."""
        tr_suppliers = [s for s in sample_suppliers if s["country"] == "TR"]
        assert len(tr_suppliers) >= 1
        steel_suppliers = [s for s in sample_suppliers if s["sector"] == "steel"]
        assert len(steel_suppliers) >= 1

    def test_supplier_update(self, portal):
        """Test supplier data update."""
        portal.register_supplier({
            "supplier_id": "SUP-UPD-001",
            "company_name": "Update Corp",
            "country": "IN",
        })
        portal.suppliers["SUP-UPD-001"]["quality_score"] = 80.0
        portal.suppliers["SUP-UPD-001"]["contact_email"] = "new@update.com"
        supplier = portal.suppliers["SUP-UPD-001"]
        assert supplier["quality_score"] == 80.0
        assert supplier["contact_email"] == "new@update.com"
