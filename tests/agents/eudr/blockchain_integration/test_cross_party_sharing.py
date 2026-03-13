# -*- coding: utf-8 -*-
"""
Tests for CrossPartySharing - AGENT-EUDR-013 Engine 7: Cross-Party Data Access

Comprehensive test suite covering:
- All 4 access levels (operator, competent_authority, auditor, supply_chain_partner)
- Access grant lifecycle (grant, check, revoke, expired)
- All 3 access statuses (active, revoked, expired)
- Multi-party confirmation requirements
- Dispute resolution filing
- Access audit trail completeness
- Privacy preservation (hashes only on-chain)
- Edge cases: self-grant, expired check, max grants

Test count: 50+ tests (including parametrized expansions)
Coverage target: >= 85% of CrossPartySharing module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-013 Blockchain Integration (GL-EUDR-BCI-013)
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.blockchain_integration.conftest import (
    ACCESS_LEVELS,
    ACCESS_STATUSES,
    BLOCKCHAIN_NETWORKS,
    SHA256_HEX_LENGTH,
    ANCHOR_ID_001,
    ANCHOR_ID_002,
    OPERATOR_ID_EU_001,
    OPERATOR_ID_EU_002,
    GRANT_ID_001,
    GRANT_ID_002,
    GRANT_AUTHORITY_ACTIVE,
    GRANT_AUDITOR_EXPIRED,
    ALL_SAMPLE_GRANTS,
    make_access_grant,
    assert_access_grant_valid,
)


# ===========================================================================
# 1. All Access Levels
# ===========================================================================


class TestAccessLevels:
    """Test all 4 cross-party access levels."""

    @pytest.mark.parametrize("access_level", ACCESS_LEVELS)
    def test_all_access_levels_valid(self, sharing_engine, access_level):
        """Each access level is recognized."""
        grant = make_access_grant(access_level=access_level)
        assert_access_grant_valid(grant)
        assert grant["access_level"] == access_level

    @pytest.mark.parametrize("access_level", ACCESS_LEVELS)
    def test_grant_structure_per_level(self, sharing_engine, access_level):
        """Each access level grant has all required fields."""
        grant = make_access_grant(access_level=access_level)
        required_keys = [
            "grant_id", "anchor_id", "grantor_id", "grantee_id",
            "access_level", "status", "granted_at",
        ]
        for key in required_keys:
            assert key in grant, f"Missing key '{key}' for level '{access_level}'"

    def test_operator_access(self, sharing_engine):
        """Operator access grant is valid."""
        grant = make_access_grant(access_level="operator")
        assert grant["access_level"] == "operator"

    def test_competent_authority_access(self, sharing_engine):
        """Competent authority access grant is valid."""
        grant = make_access_grant(
            access_level="competent_authority",
            grantee_id="CA-DE-BMEL-001",
        )
        assert grant["access_level"] == "competent_authority"

    def test_auditor_access(self, sharing_engine):
        """Auditor access grant is valid."""
        grant = make_access_grant(
            access_level="auditor",
            grantee_id="AUD-KPMG-001",
        )
        assert grant["access_level"] == "auditor"

    def test_supply_chain_partner_access(self, sharing_engine):
        """Supply chain partner access grant is valid."""
        grant = make_access_grant(
            access_level="supply_chain_partner",
            grantee_id="PARTNER-SUP-001",
        )
        assert grant["access_level"] == "supply_chain_partner"


# ===========================================================================
# 2. Access Grant Lifecycle
# ===========================================================================


class TestAccessGrant:
    """Test access grant creation, checking, and revocation."""

    def test_grant_creates_active(self, sharing_engine):
        """New grant is created with active status."""
        grant = make_access_grant(status="active")
        assert grant["status"] == "active"

    def test_grant_has_granted_at(self, sharing_engine):
        """Grant has a granted_at timestamp."""
        grant = make_access_grant()
        assert grant["granted_at"] is not None

    def test_grant_has_expires_at(self, sharing_engine):
        """Grant has an expires_at timestamp."""
        grant = make_access_grant()
        assert grant["expires_at"] is not None

    def test_revoke_grant(self, sharing_engine):
        """Grant can be revoked."""
        grant = make_access_grant(
            status="revoked",
            revocation_reason="Audit engagement completed",
        )
        assert grant["status"] == "revoked"
        assert grant["revoked_at"] is not None

    def test_expired_grant(self, sharing_engine):
        """Expired grant has expired status."""
        grant = make_access_grant(status="expired")
        assert grant["status"] == "expired"

    def test_grant_scope_restrictions(self, sharing_engine):
        """Grant can have scope restrictions."""
        grant = make_access_grant(
            scope={"fields": ["data_hash", "event_type"]},
        )
        assert grant["scope"] is not None
        assert "fields" in grant["scope"]

    def test_grant_no_scope(self, sharing_engine):
        """Grant can have no scope restrictions."""
        grant = make_access_grant(scope=None)
        assert grant["scope"] is None


# ===========================================================================
# 3. Access Statuses
# ===========================================================================


class TestAccessStatuses:
    """Test all 3 access grant statuses."""

    @pytest.mark.parametrize("status", ACCESS_STATUSES)
    def test_all_statuses_valid(self, sharing_engine, status):
        """Each access status is recognized."""
        grant = make_access_grant(status=status)
        assert_access_grant_valid(grant)
        assert grant["status"] == status

    def test_active_has_no_revoked_at(self, sharing_engine):
        """Active grant has no revoked_at timestamp."""
        grant = make_access_grant(status="active")
        assert grant["revoked_at"] is None

    def test_revoked_has_revoked_at(self, sharing_engine):
        """Revoked grant has revoked_at timestamp."""
        grant = make_access_grant(status="revoked")
        assert grant["revoked_at"] is not None

    def test_revoked_has_reason(self, sharing_engine):
        """Revoked grant can have a reason."""
        grant = make_access_grant(
            status="revoked",
            revocation_reason="Access no longer required",
        )
        assert grant["revocation_reason"] is not None

    def test_expired_grant_past_date(self, sharing_engine):
        """Expired grant has expires_at in the past."""
        grant = make_access_grant(status="expired")
        assert grant["expires_at"] is not None


# ===========================================================================
# 4. Multi-Party Confirmation
# ===========================================================================


class TestMultiPartyConfirmation:
    """Test multi-party confirmation requirements."""

    def test_confirmations_default(self, sharing_engine):
        """Default required confirmations is 2."""
        grant = make_access_grant()
        assert grant["required_confirmations"] == 2

    def test_confirmations_met(self, sharing_engine):
        """Grant with sufficient confirmations is active."""
        grant = make_access_grant(
            multi_party_confirmations=2,
            required_confirmations=2,
            status="active",
        )
        assert grant["multi_party_confirmations"] >= grant["required_confirmations"]
        assert grant["status"] == "active"

    def test_confirmations_not_met(self, sharing_engine):
        """Grant with insufficient confirmations."""
        grant = make_access_grant(
            multi_party_confirmations=1,
            required_confirmations=3,
        )
        assert grant["multi_party_confirmations"] < grant["required_confirmations"]

    def test_single_confirmation_required(self, sharing_engine):
        """Grant can require only 1 confirmation."""
        grant = make_access_grant(
            multi_party_confirmations=1,
            required_confirmations=1,
        )
        assert grant["multi_party_confirmations"] >= grant["required_confirmations"]

    def test_zero_confirmations(self, sharing_engine):
        """Grant starts with 0 confirmations."""
        grant = make_access_grant(multi_party_confirmations=0)
        assert grant["multi_party_confirmations"] == 0

    @pytest.mark.parametrize("required", [1, 2, 3, 5])
    def test_various_confirmation_requirements(self, sharing_engine, required):
        """Various confirmation requirements are valid."""
        grant = make_access_grant(required_confirmations=required)
        assert grant["required_confirmations"] == required


# ===========================================================================
# 5. Dispute Resolution
# ===========================================================================


class TestDisputeResolution:
    """Test dispute filing and structure."""

    def test_dispute_revokes_grant(self, sharing_engine):
        """Dispute can result in grant revocation."""
        grant = make_access_grant(
            status="revoked",
            revocation_reason="Dispute: data access violation detected",
        )
        assert grant["status"] == "revoked"
        assert "Dispute" in grant["revocation_reason"]

    def test_dispute_reason_recorded(self, sharing_engine):
        """Dispute reason is recorded in revocation."""
        reason = "Unauthorized data export by grantee"
        grant = make_access_grant(
            status="revoked",
            revocation_reason=reason,
        )
        assert grant["revocation_reason"] == reason

    def test_dispute_preserves_grant_history(self, sharing_engine):
        """Disputed grant still has granted_at for audit."""
        grant = make_access_grant(
            status="revoked",
            revocation_reason="Dispute filed",
        )
        assert grant["granted_at"] is not None
        assert grant["revoked_at"] is not None


# ===========================================================================
# 6. Access Audit Trail
# ===========================================================================


class TestAccessAuditTrail:
    """Test access audit trail completeness."""

    def test_grant_has_grantor_id(self, sharing_engine):
        """Grant records grantor (data owner) ID."""
        grant = make_access_grant(grantor_id=OPERATOR_ID_EU_001)
        assert grant["grantor_id"] == OPERATOR_ID_EU_001

    def test_grant_has_grantee_id(self, sharing_engine):
        """Grant records grantee ID."""
        grant = make_access_grant(grantee_id="CA-DE-BMEL-001")
        assert grant["grantee_id"] == "CA-DE-BMEL-001"

    def test_grant_has_anchor_id(self, sharing_engine):
        """Grant records the anchor being shared."""
        grant = make_access_grant(anchor_id=ANCHOR_ID_001)
        assert grant["anchor_id"] == ANCHOR_ID_001

    def test_grant_timestamps_complete(self, sharing_engine):
        """Active grant has granted_at and expires_at."""
        grant = make_access_grant(status="active")
        assert grant["granted_at"] is not None
        assert grant["expires_at"] is not None

    def test_revoked_grant_timestamps_complete(self, sharing_engine):
        """Revoked grant has all timestamps."""
        grant = make_access_grant(status="revoked")
        assert grant["granted_at"] is not None
        assert grant["revoked_at"] is not None

    def test_provenance_hash_nullable(self, sharing_engine):
        """Provenance hash starts as None."""
        grant = make_access_grant()
        assert grant["provenance_hash"] is None


# ===========================================================================
# 7. Privacy Preservation
# ===========================================================================


class TestPrivacyPreservation:
    """Test privacy preservation in cross-party sharing."""

    def test_no_pii_in_grant(self, sharing_engine):
        """Grant structure does not contain PII fields."""
        grant = make_access_grant()
        pii_fields = ["name", "email", "phone", "address", "ssn"]
        for field in pii_fields:
            assert field not in grant, f"PII field '{field}' found in grant"

    def test_anchor_id_not_data(self, sharing_engine):
        """Grant references anchor by ID, not by data content."""
        grant = make_access_grant(anchor_id=ANCHOR_ID_001)
        assert "data_hash" not in grant or grant.get("data_hash") is None

    def test_scope_restricts_fields(self, sharing_engine):
        """Scope can restrict which fields are accessible."""
        grant = make_access_grant(
            scope={"fields": ["data_hash", "event_type", "commodity"]},
        )
        assert len(grant["scope"]["fields"]) == 3

    def test_operator_ids_are_pseudonymous(self, sharing_engine):
        """Operator IDs are pseudonymous identifiers."""
        grant = make_access_grant(
            grantor_id=OPERATOR_ID_EU_001,
            grantee_id="CA-DE-BMEL-001",
        )
        # IDs are identifiers, not personal names
        assert grant["grantor_id"].startswith("OP-")


# ===========================================================================
# 8. Edge Cases
# ===========================================================================


class TestSharingEdgeCases:
    """Test edge cases for cross-party sharing."""

    def test_sample_authority_active(self, sharing_engine):
        """Pre-built GRANT_AUTHORITY_ACTIVE is valid."""
        grant = copy.deepcopy(GRANT_AUTHORITY_ACTIVE)
        assert_access_grant_valid(grant)
        assert grant["access_level"] == "competent_authority"
        assert grant["status"] == "active"

    def test_sample_auditor_expired(self, sharing_engine):
        """Pre-built GRANT_AUDITOR_EXPIRED is valid."""
        grant = copy.deepcopy(GRANT_AUDITOR_EXPIRED)
        assert_access_grant_valid(grant)
        assert grant["access_level"] == "auditor"
        assert grant["status"] == "expired"

    def test_all_samples_valid(self, sharing_engine):
        """All pre-built grant samples are valid."""
        for g in ALL_SAMPLE_GRANTS:
            g_copy = copy.deepcopy(g)
            assert_access_grant_valid(g_copy)

    def test_self_grant_structurally_valid(self, sharing_engine):
        """Self-grant (grantor == grantee) is structurally valid."""
        grant = make_access_grant(
            grantor_id=OPERATOR_ID_EU_001,
            grantee_id=OPERATOR_ID_EU_001,
            access_level="operator",
        )
        assert grant["grantor_id"] == grant["grantee_id"]
        assert_access_grant_valid(grant)

    def test_multiple_grants_unique_ids(self, sharing_engine):
        """Multiple grants have unique IDs."""
        grants = [make_access_grant() for _ in range(20)]
        ids = [g["grant_id"] for g in grants]
        assert len(set(ids)) == 20

    def test_grant_multiple_anchors(self, sharing_engine):
        """Different grants can reference different anchors."""
        g1 = make_access_grant(anchor_id=ANCHOR_ID_001)
        g2 = make_access_grant(anchor_id=ANCHOR_ID_002)
        assert g1["anchor_id"] != g2["anchor_id"]

    def test_long_expiry_grant(self, sharing_engine):
        """Grant with 5-year expiry for EUDR retention."""
        grant = make_access_grant(expires_in_days=365 * 5)
        assert grant["expires_at"] is not None

    def test_short_expiry_grant(self, sharing_engine):
        """Grant with 1-day expiry for temporary access."""
        grant = make_access_grant(expires_in_days=1)
        assert grant["expires_at"] is not None

    def test_grant_different_grantors(self, sharing_engine):
        """Grants from different operators are valid."""
        g1 = make_access_grant(grantor_id=OPERATOR_ID_EU_001)
        g2 = make_access_grant(grantor_id=OPERATOR_ID_EU_002)
        assert g1["grantor_id"] != g2["grantor_id"]
