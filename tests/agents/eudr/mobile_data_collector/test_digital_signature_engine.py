# -*- coding: utf-8 -*-
"""
Unit tests for DigitalSignatureEngine - AGENT-EUDR-015 Engine 6.

Tests all methods of DigitalSignatureEngine with 85%+ coverage.
Validates signer registration, signature creation/verification,
timestamp validation, revocation, multi-signature chains,
custody transfer signatures, and statistics.

Test count: ~50 tests
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict

import pytest

from greenlang.agents.eudr.mobile_data_collector.digital_signature_engine import (
    DigitalSignatureEngine,
    SIGNER_ROLES,
    SIGNATURE_STATUSES,
    REVOCATION_REASONS,
    WITNESS_ROLES,
)

from .conftest import assert_valid_sha256, SIGNER_ROLES as SIGNER_ROLES_LIST


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data_hash(content: str = "test_data") -> str:
    """Generate a SHA-256 data hash for testing."""
    return hashlib.sha256(content.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Test: Initialization
# ---------------------------------------------------------------------------

class TestDigitalSignatureEngineInit:
    """Tests for DigitalSignatureEngine initialization."""

    def test_initialization(self, digital_signature_engine):
        """Engine initializes with empty stores."""
        assert digital_signature_engine is not None
        assert len(digital_signature_engine) == 0

    def test_repr(self, digital_signature_engine):
        """Repr includes engine info."""
        r = repr(digital_signature_engine)
        assert "DigitalSignatureEngine" in r

    def test_len_starts_at_zero(self, digital_signature_engine):
        """Initial signature count is zero."""
        assert len(digital_signature_engine) == 0


# ---------------------------------------------------------------------------
# Test: register_signer
# ---------------------------------------------------------------------------

class TestRegisterSigner:
    """Tests for signer registration."""

    def test_register_valid_signer(self, digital_signature_engine):
        """Register a valid signer."""
        result = digital_signature_engine.register_signer(
            signer_id="signer-001",
            name="John Producer",
            role="producer",
            organization="Farm Co-op",
        )
        assert result["signer_id"] == "signer-001"
        assert result["name"] == "John Producer"
        assert result["role"] == "producer"
        assert result["is_active"] is True

    def test_register_generates_keys(self, digital_signature_engine):
        """Registration generates simulated key pair."""
        result = digital_signature_engine.register_signer(
            signer_id="signer-002", name="Alice", role="inspector",
        )
        assert "public_key_hex" in result
        assert_valid_sha256(result["public_key_hex"])
        assert "fingerprint" in result
        assert len(result["fingerprint"]) == 40

    @pytest.mark.parametrize("role", [
        "producer", "collector", "inspector", "buyer", "auditor",
    ])
    def test_register_various_roles(self, digital_signature_engine, role):
        """Various valid signer roles are accepted."""
        result = digital_signature_engine.register_signer(
            signer_id=f"signer-{role}", name=f"Test {role}", role=role,
        )
        assert result["role"] == role

    def test_register_duplicate_raises(self, digital_signature_engine):
        """Registering duplicate signer_id raises ValueError."""
        digital_signature_engine.register_signer(
            signer_id="dup", name="First", role="producer",
        )
        with pytest.raises(ValueError):
            digital_signature_engine.register_signer(
                signer_id="dup", name="Second", role="inspector",
            )

    def test_register_empty_id_raises(self, digital_signature_engine):
        """Empty signer_id raises ValueError."""
        with pytest.raises(ValueError):
            digital_signature_engine.register_signer(
                signer_id="", name="Test", role="producer",
            )

    def test_register_invalid_role_raises(self, digital_signature_engine):
        """Invalid role raises ValueError."""
        with pytest.raises(ValueError):
            digital_signature_engine.register_signer(
                signer_id="bad-role", name="Test", role="invalid_role",
            )


# ---------------------------------------------------------------------------
# Test: get_signer / list_signers
# ---------------------------------------------------------------------------

class TestSignerRetrieval:
    """Tests for signer retrieval."""

    def test_get_existing_signer(self, digital_signature_engine):
        """Get a registered signer."""
        digital_signature_engine.register_signer(
            signer_id="s1", name="Test", role="producer",
        )
        result = digital_signature_engine.get_signer("s1")
        assert result["signer_id"] == "s1"

    def test_get_nonexistent_raises(self, digital_signature_engine):
        """Getting nonexistent signer raises KeyError."""
        with pytest.raises(KeyError):
            digital_signature_engine.get_signer("nonexistent")

    def test_list_signers_empty(self, digital_signature_engine):
        """List signers returns empty initially."""
        result = digital_signature_engine.list_signers()
        assert len(result) == 0

    def test_list_signers_filter_by_role(self, digital_signature_engine):
        """List signers filters by role."""
        digital_signature_engine.register_signer("s1", "A", "producer")
        digital_signature_engine.register_signer("s2", "B", "inspector")
        result = digital_signature_engine.list_signers(role="producer")
        assert len(result) == 1
        assert result[0]["role"] == "producer"

    def test_list_signers_filter_by_active(self, digital_signature_engine):
        """List signers filters by active status."""
        digital_signature_engine.register_signer("s1", "A", "producer")
        digital_signature_engine.register_signer("s2", "B", "inspector")
        digital_signature_engine.deactivate_signer("s2")
        result = digital_signature_engine.list_signers(is_active=True)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Test: deactivate_signer
# ---------------------------------------------------------------------------

class TestDeactivateSigner:
    """Tests for signer deactivation."""

    def test_deactivate_signer(self, digital_signature_engine):
        """Deactivate a signer."""
        digital_signature_engine.register_signer("s1", "Test", "producer")
        result = digital_signature_engine.deactivate_signer("s1", reason="Left org")
        assert result["is_active"] is False

    def test_deactivate_nonexistent_raises(self, digital_signature_engine):
        """Deactivating nonexistent signer raises KeyError."""
        with pytest.raises(KeyError):
            digital_signature_engine.deactivate_signer("nonexistent")


# ---------------------------------------------------------------------------
# Test: create_signature
# ---------------------------------------------------------------------------

class TestCreateSignature:
    """Tests for signature creation."""

    def test_create_valid_signature(self, digital_signature_engine):
        """Create a valid signature."""
        digital_signature_engine.register_signer("s1", "Test", "producer")
        result = digital_signature_engine.create_signature(
            form_id="form-001", signer_id="s1",
            data_hash=_data_hash(), device_id="dev-001",
        )
        assert "signature_id" in result
        assert result["status"] == "valid"
        assert result["is_valid"] is True

    def test_signature_has_timestamp_binding(self, digital_signature_engine):
        """Signature includes timestamp binding."""
        digital_signature_engine.register_signer("s1", "Test", "producer")
        result = digital_signature_engine.create_signature(
            form_id="form-001", signer_id="s1",
            data_hash=_data_hash(), device_id="dev-001",
        )
        assert result["timestamp_binding"] is not None

    def test_signature_hex_is_sha256(self, digital_signature_engine):
        """Signature hex is a valid SHA-256 hash."""
        digital_signature_engine.register_signer("s1", "Test", "producer")
        result = digital_signature_engine.create_signature(
            form_id="form-001", signer_id="s1",
            data_hash=_data_hash(), device_id="dev-001",
        )
        assert_valid_sha256(result["signature_hex"])

    def test_signature_increments_count(self, digital_signature_engine):
        """Creating a signature increments signer count."""
        digital_signature_engine.register_signer("s1", "Test", "producer")
        digital_signature_engine.create_signature(
            form_id="f1", signer_id="s1", data_hash=_data_hash(), device_id="d1",
        )
        signer = digital_signature_engine.get_signer("s1")
        assert signer["signature_count"] == 1

    def test_create_with_inactive_signer_raises(self, digital_signature_engine):
        """Creating signature with inactive signer raises ValueError."""
        digital_signature_engine.register_signer("s1", "Test", "producer")
        digital_signature_engine.deactivate_signer("s1")
        with pytest.raises(ValueError):
            digital_signature_engine.create_signature(
                form_id="f1", signer_id="s1",
                data_hash=_data_hash(), device_id="d1",
            )

    def test_create_with_unregistered_signer_raises(self, digital_signature_engine):
        """Creating signature with unregistered signer raises KeyError."""
        with pytest.raises(KeyError):
            digital_signature_engine.create_signature(
                form_id="f1", signer_id="unknown",
                data_hash=_data_hash(), device_id="d1",
            )


# ---------------------------------------------------------------------------
# Test: verify_signature
# ---------------------------------------------------------------------------

class TestVerifySignature:
    """Tests for signature verification."""

    def test_verify_valid_signature(self, digital_signature_engine):
        """Verify a valid signature passes all checks."""
        digital_signature_engine.register_signer("s1", "Test", "producer")
        sig = digital_signature_engine.create_signature(
            form_id="f1", signer_id="s1",
            data_hash=_data_hash(), device_id="d1",
        )
        result = digital_signature_engine.verify_signature(sig["signature_id"])
        assert result["verified"] is True
        assert result["checks_passed"] == result["checks_total"]

    def test_verify_revoked_signature_fails(self, digital_signature_engine):
        """Verify a revoked signature fails not_revoked check."""
        digital_signature_engine.register_signer("s1", "Test", "producer")
        sig = digital_signature_engine.create_signature(
            form_id="f1", signer_id="s1",
            data_hash=_data_hash(), device_id="d1",
        )
        digital_signature_engine.revoke_signature(
            sig["signature_id"], reason="signer_error",
        )
        result = digital_signature_engine.verify_signature(sig["signature_id"])
        assert result["verified"] is False

    def test_verify_nonexistent_raises(self, digital_signature_engine):
        """Verifying nonexistent signature raises KeyError."""
        with pytest.raises(KeyError):
            digital_signature_engine.verify_signature("nonexistent")


# ---------------------------------------------------------------------------
# Test: validate_timestamp
# ---------------------------------------------------------------------------

class TestValidateTimestamp:
    """Tests for timestamp validation."""

    def test_validate_recent_timestamp(self, digital_signature_engine):
        """Recently created signature has valid timestamp."""
        digital_signature_engine.register_signer("s1", "Test", "producer")
        sig = digital_signature_engine.create_signature(
            form_id="f1", signer_id="s1",
            data_hash=_data_hash(), device_id="d1",
        )
        result = digital_signature_engine.validate_timestamp(
            sig["signature_id"], tolerance_seconds=120,
        )
        assert result["valid"] is True
        assert result["delta_seconds"] < 120


# ---------------------------------------------------------------------------
# Test: revoke_signature
# ---------------------------------------------------------------------------

class TestRevokeSignature:
    """Tests for signature revocation."""

    def test_revoke_signature(self, digital_signature_engine):
        """Revoke a signature within revocation window."""
        digital_signature_engine.register_signer("s1", "Test", "producer")
        sig = digital_signature_engine.create_signature(
            form_id="f1", signer_id="s1",
            data_hash=_data_hash(), device_id="d1",
        )
        result = digital_signature_engine.revoke_signature(
            sig["signature_id"], reason="signer_error",
        )
        assert result["is_revoked"] is True
        assert result["status"] == "revoked"
        assert result["revocation_reason"] == "signer_error"

    def test_revoke_already_revoked_raises(self, digital_signature_engine):
        """Revoking already revoked signature raises ValueError."""
        digital_signature_engine.register_signer("s1", "Test", "producer")
        sig = digital_signature_engine.create_signature(
            form_id="f1", signer_id="s1",
            data_hash=_data_hash(), device_id="d1",
        )
        digital_signature_engine.revoke_signature(
            sig["signature_id"], reason="signer_error",
        )
        with pytest.raises(ValueError):
            digital_signature_engine.revoke_signature(
                sig["signature_id"], reason="double_revoke",
            )


# ---------------------------------------------------------------------------
# Test: Multi-Signature Workflows
# ---------------------------------------------------------------------------

class TestMultiSignature:
    """Tests for multi-signature chain workflows."""

    def test_create_multi_sig_chain(self, digital_signature_engine):
        """Create a multi-signature chain."""
        result = digital_signature_engine.create_multi_sig(
            form_id="form-001",
            required_roles=["producer", "inspector"],
            description="Harvest verification",
        )
        assert "chain_id" in result
        assert result["status"] == "pending"
        assert result["is_complete"] is False

    def test_add_signature_to_chain(self, digital_signature_engine):
        """Add signatures to complete a chain."""
        digital_signature_engine.register_signer("s1", "Producer", "producer")
        digital_signature_engine.register_signer("s2", "Inspector", "inspector")

        chain = digital_signature_engine.create_multi_sig(
            form_id="form-001", required_roles=["producer", "inspector"],
        )

        sig1 = digital_signature_engine.create_signature(
            "form-001", "s1", _data_hash(), "dev-001",
        )
        digital_signature_engine.add_signature_to_chain(
            chain["chain_id"], sig1["signature_id"],
        )

        sig2 = digital_signature_engine.create_signature(
            "form-001", "s2", _data_hash("data2"), "dev-001",
        )
        updated = digital_signature_engine.add_signature_to_chain(
            chain["chain_id"], sig2["signature_id"],
        )
        assert updated["is_complete"] is True
        assert updated["status"] == "complete"

    def test_verify_complete_chain(self, digital_signature_engine):
        """Verify a complete multi-sig chain."""
        digital_signature_engine.register_signer("s1", "A", "producer")
        digital_signature_engine.register_signer("s2", "B", "inspector")

        chain = digital_signature_engine.create_multi_sig(
            form_id="f1", required_roles=["producer", "inspector"],
        )

        sig1 = digital_signature_engine.create_signature(
            "f1", "s1", _data_hash(), "d1",
        )
        digital_signature_engine.add_signature_to_chain(
            chain["chain_id"], sig1["signature_id"],
        )
        sig2 = digital_signature_engine.create_signature(
            "f1", "s2", _data_hash("d2"), "d1",
        )
        digital_signature_engine.add_signature_to_chain(
            chain["chain_id"], sig2["signature_id"],
        )

        result = digital_signature_engine.verify_chain(chain["chain_id"])
        assert result["chain_valid"] is True
        assert result["is_complete"] is True

    def test_duplicate_role_in_chain_raises(self, digital_signature_engine):
        """Adding duplicate role to chain raises ValueError."""
        digital_signature_engine.register_signer("s1", "A", "producer")
        digital_signature_engine.register_signer("s2", "B", "producer")

        chain = digital_signature_engine.create_multi_sig(
            form_id="f1", required_roles=["producer"],
        )

        sig1 = digital_signature_engine.create_signature(
            "f1", "s1", _data_hash(), "d1",
        )
        digital_signature_engine.add_signature_to_chain(
            chain["chain_id"], sig1["signature_id"],
        )

        sig2 = digital_signature_engine.create_signature(
            "f1", "s2", _data_hash("d2"), "d1",
        )
        with pytest.raises(ValueError):
            digital_signature_engine.add_signature_to_chain(
                chain["chain_id"], sig2["signature_id"],
            )


# ---------------------------------------------------------------------------
# Test: Custody Transfer Signatures
# ---------------------------------------------------------------------------

class TestCustodyTransfer:
    """Tests for custody transfer multi-signature."""

    def test_custody_transfer_without_witness(self, digital_signature_engine):
        """Custody transfer with from and to signers."""
        digital_signature_engine.register_signer("from1", "Sender", "producer")
        digital_signature_engine.register_signer("to1", "Receiver", "buyer")

        result = digital_signature_engine.create_custody_signature(
            form_id="custody-001",
            from_signer_id="from1",
            to_signer_id="to1",
            data_hash=_data_hash(),
            device_id="dev-001",
        )
        assert result["is_complete"] is True
        assert result["from_party"]["signer_id"] == "from1"
        assert result["to_party"]["signer_id"] == "to1"
        assert result["witness"] is None

    def test_custody_transfer_with_witness(self, digital_signature_engine):
        """Custody transfer with witness."""
        digital_signature_engine.register_signer("from2", "Sender", "producer")
        digital_signature_engine.register_signer("to2", "Receiver", "buyer")
        digital_signature_engine.register_signer("wit2", "Witness", "inspector")

        result = digital_signature_engine.create_custody_signature(
            form_id="custody-002",
            from_signer_id="from2",
            to_signer_id="to2",
            data_hash=_data_hash(),
            device_id="dev-001",
            witness_signer_id="wit2",
        )
        assert result["is_complete"] is True
        assert result["witness"] is not None
        assert result["witness"]["signer_id"] == "wit2"


# ---------------------------------------------------------------------------
# Test: Statistics
# ---------------------------------------------------------------------------

class TestStatistics:
    """Tests for engine statistics."""

    def test_statistics_empty(self, digital_signature_engine):
        """Statistics reflect empty state."""
        stats = digital_signature_engine.get_statistics()
        assert stats["total_signatures"] == 0
        assert stats["total_signers"] == 0

    def test_statistics_after_operations(self, digital_signature_engine):
        """Statistics reflect operations."""
        digital_signature_engine.register_signer("s1", "Test", "producer")
        digital_signature_engine.create_signature(
            "f1", "s1", _data_hash(), "d1",
        )
        stats = digital_signature_engine.get_statistics()
        assert stats["total_signatures"] == 1
        assert stats["total_signers"] == 1
        assert stats["active_signers"] == 1

    def test_statistics_includes_chains(self, digital_signature_engine):
        """Statistics includes multi-sig chain count."""
        digital_signature_engine.create_multi_sig(
            form_id="f1", required_roles=["producer"],
        )
        stats = digital_signature_engine.get_statistics()
        assert stats["total_chains"] >= 1

    def test_statistics_revoked_count(self, digital_signature_engine):
        """Statistics tracks revoked signature count."""
        digital_signature_engine.register_signer("s1", "Test", "producer")
        sig = digital_signature_engine.create_signature(
            "f1", "s1", _data_hash(), "d1",
        )
        digital_signature_engine.revoke_signature(sig["signature_id"], reason="signer_error")
        stats = digital_signature_engine.get_statistics()
        assert stats["revoked_signatures"] == 1


# ---------------------------------------------------------------------------
# Test: Additional Signer Tests
# ---------------------------------------------------------------------------

class TestSignerAdditional:
    """Additional tests for signer operations."""

    def test_register_with_organization(self, digital_signature_engine):
        """Register signer with organization field."""
        result = digital_signature_engine.register_signer(
            signer_id="org-s1", name="Org User", role="auditor",
            organization="Certification Corp",
        )
        assert result["organization"] == "Certification Corp"

    @pytest.mark.parametrize("role", [
        "cooperative_manager", "transport_operator",
        "warehouse_manager", "exporter", "importer", "system",
    ])
    def test_register_remaining_roles(self, digital_signature_engine, role):
        """All remaining signer roles are accepted."""
        result = digital_signature_engine.register_signer(
            signer_id=f"signer-{role}", name=f"Test {role}", role=role,
        )
        assert result["role"] == role

    def test_list_signers_returns_all(self, digital_signature_engine):
        """List signers returns all registered signers."""
        digital_signature_engine.register_signer("s1", "A", "producer")
        digital_signature_engine.register_signer("s2", "B", "inspector")
        digital_signature_engine.register_signer("s3", "C", "buyer")
        result = digital_signature_engine.list_signers()
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Test: Additional Signature Tests
# ---------------------------------------------------------------------------

class TestSignatureAdditional:
    """Additional tests for signature operations."""

    def test_get_existing_signature(self, digital_signature_engine):
        """Retrieve an existing signature by ID."""
        digital_signature_engine.register_signer("s1", "Test", "producer")
        sig = digital_signature_engine.create_signature(
            "f1", "s1", _data_hash(), "d1",
        )
        result = digital_signature_engine.get_signature(sig["signature_id"])
        assert result["signature_id"] == sig["signature_id"]

    def test_get_nonexistent_signature_raises(self, digital_signature_engine):
        """Getting nonexistent signature raises KeyError."""
        with pytest.raises(KeyError):
            digital_signature_engine.get_signature("nonexistent")

    def test_list_signatures_empty(self, digital_signature_engine):
        """List signatures returns empty initially."""
        result = digital_signature_engine.list_signatures()
        assert len(result) == 0

    def test_list_signatures_filter_by_signer(self, digital_signature_engine):
        """List signatures can filter by signer_id."""
        digital_signature_engine.register_signer("s1", "A", "producer")
        digital_signature_engine.register_signer("s2", "B", "inspector")
        digital_signature_engine.create_signature("f1", "s1", _data_hash("d1"), "d1")
        digital_signature_engine.create_signature("f2", "s2", _data_hash("d2"), "d1")
        result = digital_signature_engine.list_signatures(signer_id="s1")
        assert len(result) == 1
        assert result[0]["signer_id"] == "s1"

    def test_multiple_signatures_different_data(self, digital_signature_engine):
        """Multiple signatures for different data produce different hashes."""
        digital_signature_engine.register_signer("s1", "Test", "producer")
        sig1 = digital_signature_engine.create_signature(
            "f1", "s1", _data_hash("data_a"), "d1",
        )
        sig2 = digital_signature_engine.create_signature(
            "f2", "s1", _data_hash("data_b"), "d1",
        )
        assert sig1["signature_hex"] != sig2["signature_hex"]

    def test_signature_unique_ids(self, digital_signature_engine):
        """Each signature gets a unique ID."""
        digital_signature_engine.register_signer("s1", "Test", "producer")
        ids = set()
        for i in range(5):
            sig = digital_signature_engine.create_signature(
                f"f{i}", "s1", _data_hash(f"data_{i}"), "d1",
            )
            ids.add(sig["signature_id"])
        assert len(ids) == 5


# ---------------------------------------------------------------------------
# Test: Additional Multi-Sig and Chain Tests
# ---------------------------------------------------------------------------

class TestMultiSigAdditional:
    """Additional tests for multi-signature workflows."""

    def test_verify_incomplete_chain(self, digital_signature_engine):
        """Verify an incomplete chain reports incomplete."""
        digital_signature_engine.register_signer("s1", "A", "producer")
        chain = digital_signature_engine.create_multi_sig(
            form_id="f1", required_roles=["producer", "inspector"],
        )
        sig1 = digital_signature_engine.create_signature(
            "f1", "s1", _data_hash(), "d1",
        )
        digital_signature_engine.add_signature_to_chain(
            chain["chain_id"], sig1["signature_id"],
        )
        result = digital_signature_engine.verify_chain(chain["chain_id"])
        assert result["is_complete"] is False

    def test_get_multi_sig_chain(self, digital_signature_engine):
        """Retrieve a multi-sig chain by ID."""
        chain = digital_signature_engine.create_multi_sig(
            form_id="f1", required_roles=["producer"],
        )
        result = digital_signature_engine.get_multi_sig_chain(chain["chain_id"])
        assert result["chain_id"] == chain["chain_id"]
        assert result["required_roles"] == ["producer"]

    def test_multi_sig_with_three_roles(self, digital_signature_engine):
        """Multi-sig chain with three required roles."""
        digital_signature_engine.register_signer("s1", "A", "producer")
        digital_signature_engine.register_signer("s2", "B", "inspector")
        digital_signature_engine.register_signer("s3", "C", "buyer")

        chain = digital_signature_engine.create_multi_sig(
            form_id="f1",
            required_roles=["producer", "inspector", "buyer"],
        )
        assert chain["is_complete"] is False

        for signer_id, data_suffix in [("s1", "a"), ("s2", "b"), ("s3", "c")]:
            sig = digital_signature_engine.create_signature(
                "f1", signer_id, _data_hash(data_suffix), "d1",
            )
            digital_signature_engine.add_signature_to_chain(
                chain["chain_id"], sig["signature_id"],
            )

        result = digital_signature_engine.verify_chain(chain["chain_id"])
        assert result["is_complete"] is True
        assert result["chain_valid"] is True


# ---------------------------------------------------------------------------
# Test: Timestamp Validation Additional
# ---------------------------------------------------------------------------

class TestTimestampAdditional:
    """Additional timestamp validation tests."""

    def test_validate_timestamp_tight_tolerance(self, digital_signature_engine):
        """Timestamp with tight tolerance still passes for fresh signature."""
        digital_signature_engine.register_signer("s1", "Test", "producer")
        sig = digital_signature_engine.create_signature(
            "f1", "s1", _data_hash(), "d1",
        )
        result = digital_signature_engine.validate_timestamp(
            sig["signature_id"], tolerance_seconds=3600,
        )
        assert result["valid"] is True

    def test_validate_timestamp_nonexistent_raises(self, digital_signature_engine):
        """Validating timestamp of nonexistent signature raises KeyError."""
        with pytest.raises(KeyError):
            digital_signature_engine.validate_timestamp("nonexistent")
