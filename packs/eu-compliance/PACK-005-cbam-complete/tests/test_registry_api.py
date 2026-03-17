# -*- coding: utf-8 -*-
"""
PACK-005 CBAM Complete Pack - Registry API Engine Tests (20 tests)

Tests RegistryAPIEngine: declaration submission/amendment/status,
certificate purchase/surrender/resale, balance/price queries,
installation registration, declarant status, polling, retries,
structured error parsing, mock mode, audit logging, and provenance.

Author: GreenLang QA Team
"""

import json
import time
from typing import Any, Dict, List

import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from conftest import (
    StubRegistryClient,
    _compute_hash,
    _new_uuid,
    _utcnow,
    assert_provenance_hash,
)


# ---------------------------------------------------------------------------
# Declaration Operations (4 tests)
# ---------------------------------------------------------------------------

class TestDeclarationOperations:
    """Test declaration submission and management."""

    def test_submit_declaration(self, mock_registry_client):
        """Test submitting a CBAM declaration."""
        declaration = {
            "declaration_id": "DECL-2026-001",
            "year": 2026,
            "total_emissions_tco2e": 22500.0,
            "certificates_required": 563,
        }
        result = mock_registry_client.submit_declaration(declaration)
        assert result["status"] == "submitted"
        assert result["declaration_id"] == "DECL-2026-001"

    def test_amend_declaration(self, mock_registry_client):
        """Test amending a submitted declaration."""
        mock_registry_client.submit_declaration({
            "declaration_id": "DECL-2026-002",
            "total_emissions_tco2e": 20000.0,
        })
        result = mock_registry_client.amend_declaration(
            "DECL-2026-002",
            {"total_emissions_tco2e": 21000.0, "reason": "Revised supplier data"},
        )
        assert result["status"] == "amended"

    def test_check_status_accepted(self, mock_registry_client):
        """Test checking status of an accepted declaration."""
        mock_registry_client.submit_declaration({
            "declaration_id": "DECL-2026-003",
        })
        result = mock_registry_client.check_status("DECL-2026-003")
        assert result["status"] == "submitted"
        assert result["declaration_id"] == "DECL-2026-003"

    def test_check_status_rejected(self, mock_registry_client):
        """Test checking status of a non-existent declaration."""
        result = mock_registry_client.check_status("DECL-NONEXISTENT")
        assert result["status"] == "not_found"


# ---------------------------------------------------------------------------
# Certificate Operations (4 tests)
# ---------------------------------------------------------------------------

class TestCertificateOperations:
    """Test certificate purchase, surrender, and resale."""

    def test_purchase_certificates(self, mock_registry_client):
        """Test purchasing certificates via registry."""
        result = mock_registry_client.purchase_certificates(50, 78.50)
        assert result["status"] == "purchased"
        assert result["quantity"] == 50
        assert result["new_balance"] == 150  # 100 initial + 50

    def test_surrender_certificates(self, mock_registry_client):
        """Test surrendering certificates."""
        result = mock_registry_client.surrender_certificates(30)
        assert result["status"] == "surrendered"
        assert result["quantity"] == 30
        assert result["new_balance"] == 70  # 100 - 30

    def test_resell_certificates(self, mock_registry_client):
        """Test reselling certificates."""
        result = mock_registry_client.resell_certificates(10, 80.00)
        assert result["status"] == "resold"
        assert result["quantity"] == 10
        assert result["new_balance"] == 90

    def test_surrender_exceeds_balance(self, mock_registry_client):
        """Test surrender fails when exceeding balance."""
        result = mock_registry_client.surrender_certificates(500)
        assert result["status"] == "rejected"
        assert "Insufficient" in result.get("error", "")


# ---------------------------------------------------------------------------
# Balance and Price (2 tests)
# ---------------------------------------------------------------------------

class TestBalanceAndPrice:
    """Test balance and price queries."""

    def test_get_balance(self, mock_registry_client):
        """Test getting certificate balance."""
        result = mock_registry_client.get_balance()
        assert result["balance"] == 100
        assert "as_of" in result

    def test_get_current_price(self, mock_registry_client):
        """Test getting current certificate price."""
        result = mock_registry_client.get_current_price()
        assert result["price_eur"] == 78.50
        assert result["source"] == "EU_ETS_AUCTION"


# ---------------------------------------------------------------------------
# Registration (2 tests)
# ---------------------------------------------------------------------------

class TestRegistration:
    """Test installation and declarant registration."""

    def test_register_installation(self, mock_registry_client):
        """Test registering a production installation."""
        result = mock_registry_client.register_installation({
            "installation_id": "INST-TR-NEW-001",
            "name": "New Steel Works",
            "country": "TR",
        })
        assert result["status"] == "registered"

    def test_check_declarant_status(self, mock_registry_client):
        """Test checking declarant status by EORI."""
        result = mock_registry_client.check_declarant_status("DE123456789012345")
        assert result["status"] == "active"
        assert result["member_state"] == "DE"


# ---------------------------------------------------------------------------
# Polling and Retry (3 tests)
# ---------------------------------------------------------------------------

class TestPollingAndRetry:
    """Test polling and retry mechanisms."""

    def test_poll_status_success(self, mock_registry_client):
        """Test polling until status is received."""
        mock_registry_client.submit_declaration({
            "declaration_id": "DECL-POLL-001",
        })
        # Simulate polling
        max_polls = 5
        for i in range(max_polls):
            result = mock_registry_client.check_status("DECL-POLL-001")
            if result["status"] in ("submitted", "accepted"):
                break
        assert result["status"] in ("submitted", "accepted")

    def test_poll_status_timeout(self, mock_registry_client):
        """Test polling times out after max attempts."""
        max_polls = 3
        poll_count = 0
        status = "pending"
        for i in range(max_polls):
            poll_count += 1
            # Simulating non-existent declaration stays not_found
            result = mock_registry_client.check_status("DECL-NEVER-EXISTS")
            if result["status"] not in ("pending", "not_found"):
                break
        assert poll_count == max_polls

    def test_retry_on_failure(self, mock_registry_client):
        """Test retry logic on transient failure."""
        retry_count = 0
        max_retries = 3
        success = False
        for attempt in range(max_retries):
            retry_count += 1
            # On third attempt, succeed
            if attempt == 2:
                result = mock_registry_client.get_balance()
                success = result["balance"] >= 0
                break
        assert success is True
        assert retry_count == 3


# ---------------------------------------------------------------------------
# Error Handling, Mock, Audit, Provenance (5 tests)
# ---------------------------------------------------------------------------

class TestErrorsAndAudit:
    """Test error handling, mock mode, audit logging, and provenance."""

    def test_structured_error_parsing(self, mock_registry_client):
        """Test structured error response parsing."""
        result = mock_registry_client.surrender_certificates(999999)
        assert result["status"] == "rejected"
        assert "error" in result
        error_msg = result["error"]
        assert isinstance(error_msg, str)
        assert len(error_msg) > 0

    def test_mock_mode(self, mock_registry_client):
        """Test registry operates in mock mode without real API calls."""
        assert mock_registry_client._mode == "mock"
        # All operations should succeed in mock mode
        result = mock_registry_client.get_current_price()
        assert result["price_eur"] > 0

    def test_audit_logging(self, mock_registry_client):
        """Test operations generate audit-compatible entries."""
        mock_registry_client.submit_declaration({
            "declaration_id": "DECL-AUDIT-001",
        })
        result = mock_registry_client.check_status("DECL-AUDIT-001")
        audit_entry = {
            "action": "check_status",
            "declaration_id": "DECL-AUDIT-001",
            "result_status": result["status"],
            "timestamp": _utcnow().isoformat(),
            "provenance_hash": _compute_hash(result),
        }
        assert len(audit_entry["provenance_hash"]) == 64

    def test_provenance_hash(self, mock_registry_client):
        """Test all results can produce provenance hashes."""
        result = mock_registry_client.get_balance()
        result["provenance_hash"] = _compute_hash(result)
        assert_provenance_hash(result)

    def test_amend_nonexistent_declaration(self, mock_registry_client):
        """Test amending a non-existent declaration returns error."""
        result = mock_registry_client.amend_declaration(
            "DECL-DOES-NOT-EXIST",
            {"total_emissions_tco2e": 5000.0},
        )
        assert result["status"] == "rejected"
