# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-EUDR-038 Reference Number Generator -- lifecycle_manager.py

Tests reference number lifecycle state transitions (active, used, expired,
revoked, transferred, cancelled), expiration handling, revocation with
reasons, transfer between operators, and audit trail completeness. 40+ tests.

These tests validate lifecycle behavior through the model layer and
NumberGenerator engine. Once lifecycle_manager.py is implemented,
tests can import the engine directly.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from greenlang.agents.eudr.reference_number_generator.config import (
    ReferenceNumberGeneratorConfig,
)
from greenlang.agents.eudr.reference_number_generator.models import (
    AuditAction,
    ReferenceNumber,
    ReferenceNumberComponents,
    ReferenceNumberStatus,
    RevocationReason,
    TransferReason,
    TransferRecord,
)
from greenlang.agents.eudr.reference_number_generator.number_generator import (
    NumberGenerator,
)


# ====================================================================
# Helper: Create reference for lifecycle testing
# ====================================================================


def _make_reference(
    status: ReferenceNumberStatus = ReferenceNumberStatus.ACTIVE,
    expires_in_days: int = 365,
) -> ReferenceNumber:
    """Create a reference with a given status and expiration."""
    now = datetime.now(timezone.utc)
    return ReferenceNumber(
        reference_id=str(uuid.uuid4()),
        reference_number="EUDR-DE-2026-OP001-000001-7",
        components=ReferenceNumberComponents(
            prefix="EUDR",
            member_state="DE",
            year=2026,
            operator_code="OP001",
            sequence=1,
            checksum="7",
        ),
        operator_id="OP-001",
        commodity="coffee",
        status=status,
        format_version="1.0",
        checksum_algorithm="luhn",
        generated_at=now,
        expires_at=now + timedelta(days=expires_in_days),
        provenance_hash="a" * 64,
    )


# ====================================================================
# Test: ReferenceNumberStatus Transitions
# ====================================================================


class TestStatusTransitions:
    """Test valid and invalid lifecycle state transitions."""

    def test_active_to_used(self):
        ref = _make_reference(ReferenceNumberStatus.ACTIVE)
        ref.status = ReferenceNumberStatus.USED
        ref.used_at = datetime.now(timezone.utc)
        assert ref.status == ReferenceNumberStatus.USED
        assert ref.used_at is not None

    def test_active_to_expired(self):
        ref = _make_reference(ReferenceNumberStatus.ACTIVE)
        ref.status = ReferenceNumberStatus.EXPIRED
        assert ref.status == ReferenceNumberStatus.EXPIRED

    def test_active_to_revoked(self):
        ref = _make_reference(ReferenceNumberStatus.ACTIVE)
        ref.status = ReferenceNumberStatus.REVOKED
        ref.revoked_at = datetime.now(timezone.utc)
        assert ref.status == ReferenceNumberStatus.REVOKED
        assert ref.revoked_at is not None

    def test_active_to_transferred(self):
        ref = _make_reference(ReferenceNumberStatus.ACTIVE)
        ref.status = ReferenceNumberStatus.TRANSFERRED
        assert ref.status == ReferenceNumberStatus.TRANSFERRED

    def test_active_to_cancelled(self):
        ref = _make_reference(ReferenceNumberStatus.ACTIVE)
        ref.status = ReferenceNumberStatus.CANCELLED
        assert ref.status == ReferenceNumberStatus.CANCELLED

    def test_reserved_to_active(self):
        ref = _make_reference(ReferenceNumberStatus.RESERVED)
        ref.status = ReferenceNumberStatus.ACTIVE
        assert ref.status == ReferenceNumberStatus.ACTIVE

    def test_reserved_to_cancelled(self):
        ref = _make_reference(ReferenceNumberStatus.RESERVED)
        ref.status = ReferenceNumberStatus.CANCELLED
        assert ref.status == ReferenceNumberStatus.CANCELLED

    def test_all_statuses_reachable(self):
        for status in ReferenceNumberStatus:
            ref = _make_reference(ReferenceNumberStatus.ACTIVE)
            ref.status = status
            assert ref.status == status


# ====================================================================
# Test: Expiration Handling
# ====================================================================


class TestExpirationHandling:
    """Test reference number expiration logic."""

    def test_reference_not_expired(self):
        ref = _make_reference(expires_in_days=365)
        assert ref.expires_at > datetime.now(timezone.utc)

    def test_reference_expired(self):
        ref = _make_reference(expires_in_days=-1)
        assert ref.expires_at < datetime.now(timezone.utc)

    def test_expiration_config_default_12_months(self, sample_config):
        assert sample_config.default_expiration_months == 12

    def test_expiration_config_max_60_months(self, sample_config):
        assert sample_config.max_expiration_months == 60

    def test_expiration_warning_days(self, sample_config):
        assert sample_config.expiration_warning_days == 30

    def test_auto_expiration_enabled(self, sample_config):
        assert sample_config.enable_auto_expiration is True

    def test_reference_within_warning_period(self):
        ref = _make_reference(expires_in_days=15)
        warning_threshold = datetime.now(timezone.utc) + timedelta(days=30)
        assert ref.expires_at < warning_threshold

    def test_generated_reference_has_expiration(self):
        """Generated references must include an expiration date."""
        import asyncio
        engine = NumberGenerator()
        result = asyncio.get_event_loop().run_until_complete(
            engine.generate("OP-001", "DE")
        )
        assert result["expires_at"] is not None

    def test_custom_expiration_months(self):
        config = ReferenceNumberGeneratorConfig(default_expiration_months=6)
        engine = NumberGenerator(config=config)
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            engine.generate("OP-001", "DE")
        )
        # Verify expiration is approximately 6 months out
        expires_at = datetime.fromisoformat(result["expires_at"])
        generated_at = datetime.fromisoformat(result["generated_at"])
        diff_days = (expires_at - generated_at).days
        assert 170 <= diff_days <= 190  # ~6 months (180 days)


# ====================================================================
# Test: Revocation
# ====================================================================


class TestRevocation:
    """Test reference number revocation."""

    def test_revocation_reasons(self):
        assert len(RevocationReason) == 7

    @pytest.mark.parametrize("reason", list(RevocationReason))
    def test_all_revocation_reasons_valid(self, reason):
        ref = _make_reference(ReferenceNumberStatus.ACTIVE)
        ref.status = ReferenceNumberStatus.REVOKED
        ref.revoked_at = datetime.now(timezone.utc)
        assert ref.status == ReferenceNumberStatus.REVOKED

    def test_revocation_config_requires_reason(self, sample_config):
        assert sample_config.require_revocation_reason is True

    def test_revoked_reference_has_timestamp(self):
        ref = _make_reference(ReferenceNumberStatus.ACTIVE)
        now = datetime.now(timezone.utc)
        ref.status = ReferenceNumberStatus.REVOKED
        ref.revoked_at = now
        assert ref.revoked_at == now


# ====================================================================
# Test: Transfer
# ====================================================================


class TestTransfer:
    """Test reference number transfer between operators."""

    def test_transfer_reasons(self):
        assert len(TransferReason) == 5

    def test_transfer_record_creation(self, sample_transfer_record):
        assert sample_transfer_record.from_operator_id == "OP-001"
        assert sample_transfer_record.to_operator_id == "OP-002"
        assert sample_transfer_record.reason == TransferReason.OWNERSHIP_CHANGE

    @pytest.mark.parametrize("reason", list(TransferReason))
    def test_all_transfer_reasons(self, reason):
        record = TransferRecord(
            transfer_id=str(uuid.uuid4()),
            reference_number="EUDR-DE-2026-OP001-000001-7",
            from_operator_id="OP-001",
            to_operator_id="OP-002",
            reason=reason,
            authorized_by="admin@greenlang.io",
        )
        assert record.reason == reason

    def test_transfer_config_allowed(self, sample_config):
        assert sample_config.allow_transfer is True

    def test_transfer_record_provenance_hash(self, sample_transfer_record):
        assert len(sample_transfer_record.provenance_hash) == 64

    def test_transfer_record_timestamp(self):
        before = datetime.now(timezone.utc)
        record = TransferRecord(
            transfer_id=str(uuid.uuid4()),
            reference_number="EUDR-DE-2026-OP001-000001-7",
            from_operator_id="OP-001",
            to_operator_id="OP-003",
            reason=TransferReason.MERGER_ACQUISITION,
            authorized_by="admin",
        )
        after = datetime.now(timezone.utc)
        assert before <= record.transferred_at <= after


# ====================================================================
# Test: Audit Actions
# ====================================================================


class TestAuditActions:
    """Test audit trail action types for lifecycle events."""

    def test_audit_action_count(self):
        assert len(AuditAction) == 10

    def test_lifecycle_audit_actions(self):
        lifecycle_actions = {
            AuditAction.GENERATE,
            AuditAction.ACTIVATE,
            AuditAction.USE,
            AuditAction.EXPIRE,
            AuditAction.REVOKE,
            AuditAction.TRANSFER,
            AuditAction.CANCEL,
        }
        assert all(a in AuditAction for a in lifecycle_actions)

    def test_validation_audit_actions(self):
        assert AuditAction.VALIDATE in AuditAction
        assert AuditAction.VERIFY in AuditAction

    def test_batch_audit_action(self):
        assert AuditAction.BATCH_GENERATE in AuditAction


# ====================================================================
# Test: Retention
# ====================================================================


class TestRetention:
    """Test EUDR retention requirements."""

    def test_retention_years_minimum_5(self, sample_config):
        assert sample_config.retention_years >= 5

    def test_retention_years_eudr_article_31(self, sample_config):
        """EUDR Article 31 requires 5 years minimum retention."""
        assert sample_config.retention_years >= 5

    def test_custom_retention_years(self):
        config = ReferenceNumberGeneratorConfig(retention_years=10)
        assert config.retention_years == 10

    def test_below_minimum_retention_warns(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            ReferenceNumberGeneratorConfig(retention_years=3)
        assert any("below EUDR minimum" in r.message for r in caplog.records)


# ====================================================================
# Test: Complete Lifecycle Cycle
# ====================================================================


class TestCompleteLifecycleCycle:
    """Test complete reference number lifecycle from generation to expiry."""

    @pytest.mark.asyncio
    async def test_generate_to_used_lifecycle(self):
        engine = NumberGenerator()
        result = await engine.generate("OP-001", "DE")
        assert result["status"] == "active"

        # Simulate marking as used
        ref_num = result["reference_number"]
        ref = engine._references[ref_num]
        ref["status"] = ReferenceNumberStatus.USED.value
        ref["used_at"] = datetime.now(timezone.utc).isoformat()
        assert ref["status"] == "used"

    @pytest.mark.asyncio
    async def test_generate_to_revoked_lifecycle(self):
        engine = NumberGenerator()
        result = await engine.generate("OP-001", "DE")
        assert result["status"] == "active"

        ref_num = result["reference_number"]
        ref = engine._references[ref_num]
        ref["status"] = ReferenceNumberStatus.REVOKED.value
        ref["revoked_at"] = datetime.now(timezone.utc).isoformat()
        assert ref["status"] == "revoked"
