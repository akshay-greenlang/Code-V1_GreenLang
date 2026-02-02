# -*- coding: utf-8 -*-
"""
FR-043: Signed Approvals/Attestations - Unit Tests
===================================================

Comprehensive unit tests for the approval workflow implementation.

Tests cover:
- ApprovalAttestation model creation and hashing
- ApprovalRequest model creation and provenance
- InMemoryApprovalStore CRUD operations
- SignatureUtils Ed25519 key generation, signing, and verification
- ApprovalWorkflow end-to-end workflow

Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import hashlib
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from greenlang.orchestrator.governance.approvals import (
    ApprovalStatus,
    ApprovalDecision,
    ApprovalAttestation,
    ApprovalRequest,
    ApprovalStore,
    InMemoryApprovalStore,
    SignatureUtils,
    ApprovalWorkflow,
    ApprovalError,
    ApprovalNotFoundError,
    ApprovalExpiredError,
    ApprovalAlreadyDecidedError,
    SignatureVerificationError,
    CRYPTO_AVAILABLE,
)

# Import ApprovalRequirement and ApprovalType from approvals.py fallback
try:
    from greenlang.orchestrator.governance.policy_engine import ApprovalRequirement, ApprovalType
except ImportError:
    from greenlang.orchestrator.governance.approvals import ApprovalType
    from pydantic import BaseModel
    from typing import Optional

    class ApprovalRequirement(BaseModel):
        approval_type: ApprovalType
        approver_id: Optional[str] = None
        approver_role: Optional[str] = None
        reason: str
        deadline_hours: int = 24
        auto_deny_on_timeout: bool = True


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_timestamp():
    """Return a fixed UTC timestamp for deterministic testing."""
    return datetime(2026, 1, 28, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_attestation(sample_timestamp):
    """Create a sample attestation for testing."""
    return ApprovalAttestation(
        approver_id="user-123",
        approver_name="John Doe",
        approver_role="manager",
        decision=ApprovalDecision.APPROVED,
        reason="Approved for production deployment",
        timestamp=sample_timestamp,
        signature="dGVzdC1zaWduYXR1cmU=",
        public_key="dGVzdC1wdWJsaWMta2V5",
    )


@pytest.fixture
def sample_requirement():
    """Create a sample approval requirement."""
    return ApprovalRequirement(
        approval_type=ApprovalType.MANAGER,
        reason="Manager approval required for production deployment",
        deadline_hours=24,
    )


@pytest.fixture
def sample_request(sample_timestamp, sample_requirement):
    """Create a sample approval request for testing."""
    deadline = sample_timestamp + timedelta(hours=24)
    return ApprovalRequest(
        request_id="apr-test123456",
        run_id="run-abc123",
        step_id="step-deploy",
        approval_type=ApprovalType.MANAGER,
        reason="Manager approval required for production deployment",
        requested_by="system",
        requested_at=sample_timestamp,
        deadline=deadline,
        requirement=sample_requirement,
    )


@pytest.fixture
def memory_store():
    """Create an InMemoryApprovalStore for testing."""
    return InMemoryApprovalStore()


@pytest.fixture
def approval_workflow(memory_store):
    """Create an ApprovalWorkflow with in-memory store."""
    return ApprovalWorkflow(store=memory_store)


@pytest.fixture
def keypair():
    """Generate a test keypair if cryptography is available."""
    if not CRYPTO_AVAILABLE:
        pytest.skip("cryptography library not available")
    return SignatureUtils.generate_keypair()


# =============================================================================
# APPROVAL ATTESTATION TESTS
# =============================================================================

class TestApprovalAttestation:
    """Tests for ApprovalAttestation model."""

    def test_create_attestation(self, sample_attestation):
        """Test creating an attestation."""
        assert sample_attestation.approver_id == "user-123"
        assert sample_attestation.decision == ApprovalDecision.APPROVED
        assert sample_attestation.reason == "Approved for production deployment"

    def test_compute_content_hash(self, sample_attestation):
        """Test content hash computation is deterministic."""
        hash1 = sample_attestation.compute_content_hash()
        hash2 = sample_attestation.compute_content_hash()
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest

    def test_get_signable_content(self, sample_attestation):
        """Test signable content generation."""
        content = sample_attestation.get_signable_content()
        assert isinstance(content, bytes)
        # Should be canonical JSON
        assert b"approver_id" in content
        assert b"decision" in content

    def test_attestation_hash_stored(self, sample_attestation):
        """Test attestation hash can be computed and stored."""
        sample_attestation.attestation_hash = sample_attestation.compute_content_hash()
        assert sample_attestation.attestation_hash != ""
        assert len(sample_attestation.attestation_hash) == 64

    def test_ensure_utc_timestamp(self):
        """Test timestamp validation ensures UTC."""
        # Naive datetime should get UTC added
        attestation = ApprovalAttestation(
            approver_id="user-123",
            decision=ApprovalDecision.APPROVED,
            timestamp=datetime(2026, 1, 28, 12, 0, 0),  # No timezone
            signature="test",
            public_key="test",
        )
        assert attestation.timestamp.tzinfo == timezone.utc

    def test_ensure_utc_from_string(self):
        """Test timestamp parsing from ISO string."""
        attestation = ApprovalAttestation(
            approver_id="user-123",
            decision=ApprovalDecision.REJECTED,
            timestamp="2026-01-28T12:00:00+00:00",
            signature="test",
            public_key="test",
            reason="Rejected due to policy",
        )
        assert attestation.timestamp.tzinfo == timezone.utc


# =============================================================================
# APPROVAL REQUEST TESTS
# =============================================================================

class TestApprovalRequest:
    """Tests for ApprovalRequest model."""

    def test_create_request(self, sample_request):
        """Test creating an approval request."""
        assert sample_request.request_id == "apr-test123456"
        assert sample_request.run_id == "run-abc123"
        assert sample_request.step_id == "step-deploy"
        assert sample_request.status == ApprovalStatus.PENDING

    def test_compute_provenance_hash(self, sample_request):
        """Test provenance hash computation."""
        hash1 = sample_request.compute_provenance_hash()
        hash2 = sample_request.compute_provenance_hash()
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_is_expired_false(self, sample_request):
        """Test is_expired returns false for future deadline."""
        # Deadline is 24 hours in the future from sample_timestamp
        # Mock DeterministicClock to return sample_timestamp
        with patch("greenlang.orchestrator.governance.approvals.DeterministicClock") as mock_clock:
            mock_clock.now.return_value = sample_request.requested_at
            assert not sample_request.is_expired()

    def test_is_expired_true(self, sample_request):
        """Test is_expired returns true for past deadline."""
        with patch("greenlang.orchestrator.governance.approvals.DeterministicClock") as mock_clock:
            mock_clock.now.return_value = sample_request.deadline + timedelta(hours=1)
            assert sample_request.is_expired()

    def test_provenance_hash_includes_attestation(self, sample_request, sample_attestation):
        """Test provenance hash changes when attestation is added."""
        hash_before = sample_request.compute_provenance_hash()

        sample_attestation.attestation_hash = sample_attestation.compute_content_hash()
        sample_request.attestation = sample_attestation
        sample_request.status = ApprovalStatus.APPROVED

        hash_after = sample_request.compute_provenance_hash()
        assert hash_before != hash_after


# =============================================================================
# IN-MEMORY APPROVAL STORE TESTS
# =============================================================================

class TestInMemoryApprovalStore:
    """Tests for InMemoryApprovalStore."""

    @pytest.mark.asyncio
    async def test_save_and_get(self, memory_store, sample_request):
        """Test saving and retrieving a request."""
        request_id = await memory_store.save(sample_request)
        assert request_id == sample_request.request_id

        retrieved = await memory_store.get(request_id)
        assert retrieved is not None
        assert retrieved.request_id == sample_request.request_id

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, memory_store):
        """Test getting a nonexistent request returns None."""
        result = await memory_store.get("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_run(self, memory_store, sample_request):
        """Test retrieving requests by run_id."""
        await memory_store.save(sample_request)

        requests = await memory_store.get_by_run(sample_request.run_id)
        assert len(requests) == 1
        assert requests[0].request_id == sample_request.request_id

    @pytest.mark.asyncio
    async def test_get_by_step(self, memory_store, sample_request):
        """Test retrieving request by run_id and step_id."""
        await memory_store.save(sample_request)

        request = await memory_store.get_by_step(
            sample_request.run_id,
            sample_request.step_id
        )
        assert request is not None
        assert request.request_id == sample_request.request_id

    @pytest.mark.asyncio
    async def test_get_pending(self, memory_store, sample_request):
        """Test retrieving pending requests."""
        await memory_store.save(sample_request)

        pending = await memory_store.get_pending()
        assert len(pending) == 1

        # Mark as approved and check again
        sample_request.status = ApprovalStatus.APPROVED
        await memory_store.update(sample_request)

        pending = await memory_store.get_pending()
        assert len(pending) == 0

    @pytest.mark.asyncio
    async def test_update(self, memory_store, sample_request):
        """Test updating a request."""
        await memory_store.save(sample_request)

        sample_request.status = ApprovalStatus.REJECTED
        result = await memory_store.update(sample_request)
        assert result is True

        retrieved = await memory_store.get(sample_request.request_id)
        assert retrieved.status == ApprovalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_update_nonexistent(self, memory_store, sample_request):
        """Test updating a nonexistent request returns False."""
        result = await memory_store.update(sample_request)
        assert result is False

    @pytest.mark.asyncio
    async def test_expire_stale(self, memory_store, sample_request):
        """Test expiring stale requests."""
        await memory_store.save(sample_request)

        # Mock time to be after deadline
        with patch("greenlang.orchestrator.governance.approvals.DeterministicClock") as mock_clock:
            mock_clock.now.return_value = sample_request.deadline + timedelta(hours=1)
            count = await memory_store.expire_stale()

        assert count == 1

        retrieved = await memory_store.get(sample_request.request_id)
        assert retrieved.status == ApprovalStatus.EXPIRED

    @pytest.mark.asyncio
    async def test_clear(self, memory_store, sample_request):
        """Test clearing all requests."""
        await memory_store.save(sample_request)
        await memory_store.clear()

        result = await memory_store.get(sample_request.request_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_provenance_hash_computed_on_save(self, memory_store, sample_request):
        """Test that provenance hash is computed on save."""
        sample_request.provenance_hash = ""
        await memory_store.save(sample_request)

        retrieved = await memory_store.get(sample_request.request_id)
        assert retrieved.provenance_hash != ""
        assert len(retrieved.provenance_hash) == 64


# =============================================================================
# SIGNATURE UTILS TESTS
# =============================================================================

@pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography library not available")
class TestSignatureUtils:
    """Tests for SignatureUtils cryptographic operations."""

    def test_generate_keypair(self):
        """Test key pair generation."""
        private_key, public_key = SignatureUtils.generate_keypair()
        assert private_key is not None
        assert public_key is not None
        assert b"BEGIN PRIVATE KEY" in private_key
        assert b"BEGIN PUBLIC KEY" in public_key

    def test_sign_and_verify(self):
        """Test signing and verification."""
        private_key, public_key = SignatureUtils.generate_keypair()
        content = b"Test message to sign"

        signature = SignatureUtils.sign(content, private_key)
        assert signature is not None

        is_valid = SignatureUtils.verify(content, signature, public_key)
        assert is_valid is True

    def test_verify_invalid_signature(self):
        """Test verification fails with wrong signature."""
        private_key, public_key = SignatureUtils.generate_keypair()
        content = b"Test message"

        # Sign with different content
        signature = SignatureUtils.sign(b"Different content", private_key)

        is_valid = SignatureUtils.verify(content, signature, public_key)
        assert is_valid is False

    def test_verify_wrong_key(self):
        """Test verification fails with wrong public key."""
        private_key1, public_key1 = SignatureUtils.generate_keypair()
        private_key2, public_key2 = SignatureUtils.generate_keypair()

        content = b"Test message"
        signature = SignatureUtils.sign(content, private_key1)

        # Verify with wrong key
        is_valid = SignatureUtils.verify(content, signature, public_key2)
        assert is_valid is False

    def test_base64_round_trip(self):
        """Test base64 encoding/decoding."""
        original = b"Test binary data \x00\x01\x02"
        encoded = SignatureUtils.bytes_to_base64(original)
        decoded = SignatureUtils.base64_to_bytes(encoded)
        assert decoded == original


# =============================================================================
# APPROVAL WORKFLOW TESTS
# =============================================================================

class TestApprovalWorkflow:
    """Tests for ApprovalWorkflow."""

    @pytest.mark.asyncio
    async def test_request_approval(self, approval_workflow, sample_requirement):
        """Test requesting approval."""
        request_id = await approval_workflow.request_approval(
            run_id="run-123",
            step_id="step-deploy",
            requirement=sample_requirement,
            requested_by="system",
        )

        assert request_id is not None
        assert request_id.startswith("apr-")

    @pytest.mark.asyncio
    async def test_request_approval_idempotent(self, approval_workflow, sample_requirement):
        """Test requesting approval is idempotent for same step."""
        request_id1 = await approval_workflow.request_approval(
            run_id="run-123",
            step_id="step-deploy",
            requirement=sample_requirement,
        )

        request_id2 = await approval_workflow.request_approval(
            run_id="run-123",
            step_id="step-deploy",
            requirement=sample_requirement,
        )

        # Should return same ID for pending request
        assert request_id1 == request_id2

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography library not available")
    async def test_submit_approval(self, approval_workflow, sample_requirement):
        """Test submitting an approval decision."""
        private_key, public_key = SignatureUtils.generate_keypair()

        request_id = await approval_workflow.request_approval(
            run_id="run-123",
            step_id="step-deploy",
            requirement=sample_requirement,
        )

        attestation = await approval_workflow.submit_approval(
            approval_id=request_id,
            approver_id="user-456",
            decision=ApprovalDecision.APPROVED,
            private_key=private_key,
            public_key=public_key,
            reason="LGTM",
            approver_name="Jane Doe",
            approver_role="manager",
        )

        assert attestation is not None
        assert attestation.approver_id == "user-456"
        assert attestation.decision == ApprovalDecision.APPROVED

    @pytest.mark.asyncio
    async def test_submit_approval_not_found(self, approval_workflow):
        """Test submitting approval for nonexistent request."""
        if not CRYPTO_AVAILABLE:
            pytest.skip("cryptography library not available")

        private_key, public_key = SignatureUtils.generate_keypair()

        with pytest.raises(ValueError, match="not found"):
            await approval_workflow.submit_approval(
                approval_id="nonexistent",
                approver_id="user-123",
                decision=ApprovalDecision.APPROVED,
                private_key=private_key,
                public_key=public_key,
            )

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography library not available")
    async def test_submit_approval_already_decided(self, approval_workflow, sample_requirement):
        """Test submitting approval for already decided request."""
        private_key, public_key = SignatureUtils.generate_keypair()

        request_id = await approval_workflow.request_approval(
            run_id="run-123",
            step_id="step-deploy",
            requirement=sample_requirement,
        )

        # First approval
        await approval_workflow.submit_approval(
            approval_id=request_id,
            approver_id="user-456",
            decision=ApprovalDecision.APPROVED,
            private_key=private_key,
            public_key=public_key,
        )

        # Second approval should fail
        with pytest.raises(ValueError, match="Already decided"):
            await approval_workflow.submit_approval(
                approval_id=request_id,
                approver_id="user-789",
                decision=ApprovalDecision.REJECTED,
                private_key=private_key,
                public_key=public_key,
            )

    @pytest.mark.asyncio
    async def test_check_approval_status(self, approval_workflow, sample_requirement):
        """Test checking approval status."""
        request_id = await approval_workflow.request_approval(
            run_id="run-123",
            step_id="step-deploy",
            requirement=sample_requirement,
        )

        status = await approval_workflow.check_approval_status(request_id)
        assert status == ApprovalStatus.PENDING

    @pytest.mark.asyncio
    async def test_check_approval_status_not_found(self, approval_workflow):
        """Test checking status for nonexistent request."""
        with pytest.raises(ValueError, match="Not found"):
            await approval_workflow.check_approval_status("nonexistent")

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography library not available")
    async def test_verify_attestation(self, approval_workflow, sample_requirement):
        """Test verifying attestation signature."""
        private_key, public_key = SignatureUtils.generate_keypair()

        request_id = await approval_workflow.request_approval(
            run_id="run-123",
            step_id="step-deploy",
            requirement=sample_requirement,
        )

        await approval_workflow.submit_approval(
            approval_id=request_id,
            approver_id="user-456",
            decision=ApprovalDecision.APPROVED,
            private_key=private_key,
            public_key=public_key,
        )

        is_valid = await approval_workflow.verify_attestation(request_id)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_get_pending_approvals(self, approval_workflow, sample_requirement):
        """Test getting pending approvals."""
        await approval_workflow.request_approval(
            run_id="run-123",
            step_id="step-1",
            requirement=sample_requirement,
        )
        await approval_workflow.request_approval(
            run_id="run-123",
            step_id="step-2",
            requirement=sample_requirement,
        )

        pending = await approval_workflow.get_pending_approvals("run-123")
        assert len(pending) == 2

    @pytest.mark.asyncio
    async def test_get_approval(self, approval_workflow, sample_requirement):
        """Test getting a specific approval."""
        request_id = await approval_workflow.request_approval(
            run_id="run-123",
            step_id="step-deploy",
            requirement=sample_requirement,
        )

        approval = await approval_workflow.get_approval(request_id)
        assert approval is not None
        assert approval.request_id == request_id

    @pytest.mark.asyncio
    async def test_get_step_approval(self, approval_workflow, sample_requirement):
        """Test getting approval by step."""
        await approval_workflow.request_approval(
            run_id="run-123",
            step_id="step-deploy",
            requirement=sample_requirement,
        )

        approval = await approval_workflow.get_step_approval("run-123", "step-deploy")
        assert approval is not None
        assert approval.step_id == "step-deploy"


# =============================================================================
# EXCEPTION TESTS
# =============================================================================

class TestExceptions:
    """Tests for approval exception hierarchy."""

    def test_approval_error_base(self):
        """Test ApprovalError is base exception."""
        err = ApprovalError("Base error")
        assert str(err) == "Base error"
        assert isinstance(err, Exception)

    def test_approval_not_found_error(self):
        """Test ApprovalNotFoundError."""
        err = ApprovalNotFoundError("Not found")
        assert isinstance(err, ApprovalError)

    def test_approval_expired_error(self):
        """Test ApprovalExpiredError."""
        err = ApprovalExpiredError("Expired")
        assert isinstance(err, ApprovalError)

    def test_approval_already_decided_error(self):
        """Test ApprovalAlreadyDecidedError."""
        err = ApprovalAlreadyDecidedError("Already decided")
        assert isinstance(err, ApprovalError)

    def test_signature_verification_error(self):
        """Test SignatureVerificationError."""
        err = SignatureVerificationError("Invalid signature")
        assert isinstance(err, ApprovalError)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography library not available")
class TestApprovalWorkflowIntegration:
    """Integration tests for complete approval workflow."""

    @pytest.mark.asyncio
    async def test_full_approval_workflow(self):
        """Test complete approval workflow from request to verification."""
        # Setup
        store = InMemoryApprovalStore()
        workflow = ApprovalWorkflow(store=store)
        private_key, public_key = SignatureUtils.generate_keypair()

        requirement = ApprovalRequirement(
            approval_type=ApprovalType.MANAGER,
            reason="Production deployment requires manager approval",
            deadline_hours=24,
        )

        # Step 1: Request approval
        request_id = await workflow.request_approval(
            run_id="run-prod-001",
            step_id="deploy-prod",
            requirement=requirement,
            requested_by="ci-pipeline",
            metadata={"environment": "production", "version": "1.2.3"},
        )

        assert request_id is not None

        # Step 2: Check pending
        pending = await workflow.get_pending_approvals("run-prod-001")
        assert len(pending) == 1
        assert pending[0].request_id == request_id

        # Step 3: Verify status is pending
        status = await workflow.check_approval_status(request_id)
        assert status == ApprovalStatus.PENDING

        # Step 4: Submit approval
        attestation = await workflow.submit_approval(
            approval_id=request_id,
            approver_id="manager-john",
            decision=ApprovalDecision.APPROVED,
            private_key=private_key,
            public_key=public_key,
            reason="Approved after code review",
            approver_name="John Smith",
            approver_role="Engineering Manager",
        )

        assert attestation.approver_id == "manager-john"
        assert attestation.decision == ApprovalDecision.APPROVED

        # Step 5: Verify status changed
        status = await workflow.check_approval_status(request_id)
        assert status == ApprovalStatus.APPROVED

        # Step 6: Verify signature
        is_valid = await workflow.verify_attestation(request_id)
        assert is_valid is True

        # Step 7: Verify no more pending
        pending = await workflow.get_pending_approvals("run-prod-001")
        assert len(pending) == 0

        # Step 8: Verify provenance hash
        approval = await workflow.get_approval(request_id)
        assert approval.provenance_hash != ""
        assert approval.attestation.attestation_hash != ""

    @pytest.mark.asyncio
    async def test_rejection_workflow(self):
        """Test rejection workflow."""
        store = InMemoryApprovalStore()
        workflow = ApprovalWorkflow(store=store)
        private_key, public_key = SignatureUtils.generate_keypair()

        requirement = ApprovalRequirement(
            approval_type=ApprovalType.SECURITY,
            reason="Security review required",
            deadline_hours=48,
        )

        request_id = await workflow.request_approval(
            run_id="run-sec-001",
            step_id="security-scan",
            requirement=requirement,
        )

        attestation = await workflow.submit_approval(
            approval_id=request_id,
            approver_id="security-alice",
            decision=ApprovalDecision.REJECTED,
            private_key=private_key,
            public_key=public_key,
            reason="Failed security scan - critical vulnerabilities found",
        )

        assert attestation.decision == ApprovalDecision.REJECTED

        status = await workflow.check_approval_status(request_id)
        assert status == ApprovalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_multiple_steps_approval(self):
        """Test approving multiple steps in a run."""
        store = InMemoryApprovalStore()
        workflow = ApprovalWorkflow(store=store)
        private_key, public_key = SignatureUtils.generate_keypair()

        manager_req = ApprovalRequirement(
            approval_type=ApprovalType.MANAGER,
            reason="Manager approval",
            deadline_hours=24,
        )

        security_req = ApprovalRequirement(
            approval_type=ApprovalType.SECURITY,
            reason="Security approval",
            deadline_hours=24,
        )

        # Request both approvals
        req1_id = await workflow.request_approval(
            run_id="run-multi-001",
            step_id="manager-gate",
            requirement=manager_req,
        )

        req2_id = await workflow.request_approval(
            run_id="run-multi-001",
            step_id="security-gate",
            requirement=security_req,
        )

        # Both should be pending
        pending = await workflow.get_pending_approvals("run-multi-001")
        assert len(pending) == 2

        # Approve first
        await workflow.submit_approval(
            approval_id=req1_id,
            approver_id="manager-1",
            decision=ApprovalDecision.APPROVED,
            private_key=private_key,
            public_key=public_key,
        )

        # One pending
        pending = await workflow.get_pending_approvals("run-multi-001")
        assert len(pending) == 1

        # Approve second
        await workflow.submit_approval(
            approval_id=req2_id,
            approver_id="security-1",
            decision=ApprovalDecision.APPROVED,
            private_key=private_key,
            public_key=public_key,
        )

        # None pending
        pending = await workflow.get_pending_approvals("run-multi-001")
        assert len(pending) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
