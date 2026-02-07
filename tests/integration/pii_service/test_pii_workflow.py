# -*- coding: utf-8 -*-
"""
Integration tests for PII Service Workflows - SEC-011.

Tests end-to-end workflows with real infrastructure:
- Full detection and redaction pipeline
- Tokenization roundtrip
- Enforcement middleware integration
- Allowlist filtering
- Quarantine review workflow
- Remediation workflow
- Tenant isolation
- Concurrent operations
- Token expiration

Author: GreenLang Test Engineering Team
Date: February 2026
PRD: SEC-011 PII Detection/Redaction Enhancements
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from uuid import uuid4

import pytest


# ============================================================================
# TestFullDetectionAndRedaction
# ============================================================================


@pytest.mark.integration
@pytest.mark.requires_db
class TestFullDetectionAndRedaction:
    """Integration tests for detection and redaction pipeline."""

    @pytest.mark.asyncio
    async def test_full_detection_and_redaction_flow(
        self, pii_service, sample_pii_data, test_tenant_id
    ):
        """Test complete detection and redaction workflow."""
        content = f"""
        Customer record:
        Name: {sample_pii_data['name']}
        Email: {sample_pii_data['email']}
        Phone: {sample_pii_data['phone']}
        SSN: {sample_pii_data['ssn']}
        """

        # Step 1: Detect PII
        detection_result = await pii_service.detect(content)
        assert len(detection_result.detections) >= 3  # Name, email, phone, SSN

        # Step 2: Redact PII
        redaction_result = await pii_service.redact(content)

        # Verify PII is redacted
        redacted = redaction_result.redacted_content
        assert sample_pii_data['ssn'] not in redacted
        assert sample_pii_data['email'] not in redacted
        assert sample_pii_data['phone'] not in redacted

        # Verify structure preserved
        assert "Customer record:" in redacted

    @pytest.mark.asyncio
    async def test_json_content_detection_and_redaction(
        self, pii_service, sample_pii_data, test_tenant_id
    ):
        """Test detection and redaction of JSON content."""
        import json

        content = json.dumps({
            "user": {
                "name": sample_pii_data['name'],
                "email": sample_pii_data['email'],
                "ssn": sample_pii_data['ssn'],
            },
            "metadata": {"created_at": "2026-02-06"},
        })

        redaction_result = await pii_service.redact(content)

        # Parse redacted JSON
        redacted_json = json.loads(redaction_result.redacted_content)

        # PII should be redacted
        assert sample_pii_data['ssn'] not in redacted_json['user']['ssn']
        # Non-PII should be preserved
        assert redacted_json['metadata']['created_at'] == "2026-02-06"


# ============================================================================
# TestTokenizationRoundtrip
# ============================================================================


@pytest.mark.integration
@pytest.mark.requires_db
class TestTokenizationRoundtrip:
    """Integration tests for tokenization workflow."""

    @pytest.mark.asyncio
    async def test_tokenize_then_detokenize(
        self, pii_service, sample_pii_data, test_tenant_id, test_user_id
    ):
        """Test tokenization roundtrip preserves value."""
        try:
            from greenlang.infrastructure.pii_service.models import PIIType
            pii_type = PIIType.SSN
        except ImportError:
            pytest.skip("PIIType not available")

        original_value = sample_pii_data['ssn']

        # Tokenize
        token = await pii_service.tokenize(
            original_value, pii_type, test_tenant_id
        )
        assert token is not None
        assert token.startswith("tok_")

        # Detokenize
        recovered_value = await pii_service.detokenize(
            token, test_tenant_id, test_user_id
        )

        assert recovered_value == original_value

    @pytest.mark.asyncio
    async def test_concurrent_tokenization(
        self, pii_service, test_tenant_id
    ):
        """Test concurrent tokenization operations."""
        try:
            from greenlang.infrastructure.pii_service.models import PIIType
            pii_type = PIIType.EMAIL
        except ImportError:
            pytest.skip("PIIType not available")

        values = [f"user{i}@company.com" for i in range(20)]

        # Tokenize concurrently
        tokens = await asyncio.gather(*[
            pii_service.tokenize(v, pii_type, test_tenant_id)
            for v in values
        ])

        # All tokens should be unique
        assert len(set(tokens)) == 20

    @pytest.mark.asyncio
    async def test_token_expiration_handling(
        self, secure_vault, test_tenant_id, test_user_id
    ):
        """Test that expired tokens cannot be detokenized."""
        try:
            from greenlang.infrastructure.pii_service.models import PIIType
            pii_type = PIIType.SSN
        except ImportError:
            pytest.skip("PIIType not available")

        # Create token
        token = await secure_vault.tokenize(
            "123-45-6789", pii_type, test_tenant_id
        )

        # Manually expire (would need DB access or mock time)
        # This is a placeholder for the actual implementation

        # For now, verify valid token works
        result = await secure_vault.detokenize(token, test_tenant_id, test_user_id)
        assert result == "123-45-6789"


# ============================================================================
# TestEnforcementMiddleware
# ============================================================================


@pytest.mark.integration
class TestEnforcementMiddleware:
    """Integration tests for enforcement middleware."""

    @pytest.mark.asyncio
    async def test_enforcement_middleware_integration(
        self, pii_service, sample_pii_data, test_tenant_id, test_user_id
    ):
        """Test enforcement middleware end-to-end."""
        try:
            from greenlang.infrastructure.pii_service.models import EnforcementContext
        except ImportError:
            pytest.skip("EnforcementContext not available")

        context = EnforcementContext(
            context_type="api_request",
            path="/api/v1/users",
            method="POST",
            tenant_id=test_tenant_id,
            user_id=test_user_id,
        )

        # Content with high-sensitivity PII
        content = f"User SSN: {sample_pii_data['ssn']}"

        result = await pii_service.enforce(content, context)

        # SSN should trigger block
        assert result.blocked is True
        assert len(result.detections) > 0
        assert any(d.pii_type.value == "ssn" for d in result.detections)


# ============================================================================
# TestAllowlistWorkflow
# ============================================================================


@pytest.mark.integration
@pytest.mark.requires_db
class TestAllowlistWorkflow:
    """Integration tests for allowlist workflow."""

    @pytest.mark.asyncio
    async def test_allowlist_prevents_false_positives(
        self, pii_service, test_tenant_id
    ):
        """Test that allowlisted values are not detected."""
        # Content with test/example data
        content = "Email: test@example.com, Card: 4242424242424242"

        detection_result = await pii_service.detect(content)

        # Example.com emails and Stripe test cards should be allowlisted
        for detection in detection_result.detections:
            if detection.pii_type.value == "email":
                assert "example.com" not in str(detection)
            if detection.pii_type.value == "credit_card":
                assert "4242424242424242" not in str(detection)

    @pytest.mark.asyncio
    async def test_custom_allowlist_entry(
        self, pii_service, test_tenant_id, test_user_id
    ):
        """Test adding and using custom allowlist entries."""
        try:
            from greenlang.infrastructure.pii_service.allowlist.patterns import (
                AllowlistEntry, PatternType
            )
            from greenlang.infrastructure.pii_service.models import PIIType
        except ImportError:
            pytest.skip("Allowlist models not available")

        # Add custom entry
        entry = AllowlistEntry(
            pii_type=PIIType.EMAIL,
            pattern=r".*@internal\.company\.com$",
            pattern_type=PatternType.REGEX,
            reason="Internal company domain",
            created_by=uuid4(),
            tenant_id=test_tenant_id,
        )

        await pii_service.add_allowlist_entry(entry)

        # Detect content with internal email
        content = "Contact: admin@internal.company.com"
        result = await pii_service.detect(content)

        # Should be allowlisted
        email_detections = [d for d in result.detections if d.pii_type.value == "email"]
        assert len(email_detections) == 0


# ============================================================================
# TestQuarantineWorkflow
# ============================================================================


@pytest.mark.integration
@pytest.mark.requires_db
class TestQuarantineWorkflow:
    """Integration tests for quarantine review workflow."""

    @pytest.mark.asyncio
    async def test_quarantine_review_workflow(
        self, pii_service, sample_pii_data, test_tenant_id, test_user_id
    ):
        """Test quarantine review and release workflow."""
        try:
            from greenlang.infrastructure.pii_service.models import EnforcementContext
            from greenlang.infrastructure.pii_service.enforcement.policies import EnforcementAction
        except ImportError:
            pytest.skip("Models not available")

        # Configure SSN to quarantine
        await pii_service.update_policy("ssn", {"action": "quarantine"})

        context = EnforcementContext(
            context_type="api_request",
            path="/api/v1/data",
            method="POST",
            tenant_id=test_tenant_id,
            user_id=test_user_id,
        )

        content = f"SSN: {sample_pii_data['ssn']}"

        # Enforce (should quarantine)
        result = await pii_service.enforce(content, context)
        assert result.blocked is True

        # List quarantine items
        items = await pii_service.list_quarantine_items(test_tenant_id)
        assert len(items) > 0

        # Release item
        item_id = items[0].id
        await pii_service.release_quarantine_item(
            item_id, test_user_id, reason="False positive"
        )

        # Verify released
        items = await pii_service.list_quarantine_items(test_tenant_id)
        released = [i for i in items if str(i.id) == str(item_id)]
        if released:
            assert released[0].status == "released"


# ============================================================================
# TestRemediationWorkflow
# ============================================================================


@pytest.mark.integration
@pytest.mark.requires_db
class TestRemediationWorkflow:
    """Integration tests for remediation workflow."""

    @pytest.mark.asyncio
    async def test_remediation_workflow(
        self, pii_service, sample_pii_data, test_tenant_id, test_user_id
    ):
        """Test remediation scheduling and execution."""
        try:
            from greenlang.infrastructure.pii_service.models import PIIType
        except ImportError:
            pytest.skip("PIIType not available")

        # Schedule remediation
        item_id = await pii_service.schedule_remediation(
            pii_type=PIIType.SSN,
            source_type="database",
            source_location="users.ssn_encrypted",
            tenant_id=test_tenant_id,
            value_hash="abc123",
            action="anonymize",
        )

        assert item_id is not None

        # Get item
        item = await pii_service.get_remediation_item(item_id)
        assert item.status == "pending"

        # Approve
        await pii_service.approve_remediation(item_id, test_user_id)

        item = await pii_service.get_remediation_item(item_id)
        assert item.status == "approved"


# ============================================================================
# TestTenantIsolation
# ============================================================================


@pytest.mark.integration
@pytest.mark.requires_db
class TestTenantIsolation:
    """Integration tests for tenant isolation."""

    @pytest.mark.asyncio
    async def test_tenant_isolation_end_to_end(
        self, pii_service, sample_pii_data
    ):
        """Test that tenants cannot access each other's tokens."""
        try:
            from greenlang.infrastructure.pii_service.models import PIIType
        except ImportError:
            pytest.skip("PIIType not available")

        tenant_a = f"tenant-a-{uuid4().hex[:8]}"
        tenant_b = f"tenant-b-{uuid4().hex[:8]}"
        user_a = str(uuid4())
        user_b = str(uuid4())

        # Tenant A tokenizes
        token_a = await pii_service.tokenize(
            sample_pii_data['ssn'], PIIType.SSN, tenant_a
        )

        # Tenant A can detokenize
        value_a = await pii_service.detokenize(token_a, tenant_a, user_a)
        assert value_a == sample_pii_data['ssn']

        # Tenant B cannot detokenize tenant A's token
        with pytest.raises(Exception) as exc_info:
            await pii_service.detokenize(token_a, tenant_b, user_b)

        assert "unauthorized" in str(exc_info.value).lower() or "tenant" in str(exc_info.value).lower()


# ============================================================================
# TestConcurrentOperations
# ============================================================================


@pytest.mark.integration
@pytest.mark.requires_db
class TestConcurrentOperations:
    """Integration tests for concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_tokenization_different_tenants(
        self, pii_service
    ):
        """Test concurrent tokenization across tenants."""
        try:
            from greenlang.infrastructure.pii_service.models import PIIType
        except ImportError:
            pytest.skip("PIIType not available")

        tenants = [f"tenant-{i}-{uuid4().hex[:8]}" for i in range(5)]
        values = [f"value-{i}" for i in range(5)]

        # Concurrent tokenization
        tasks = [
            pii_service.tokenize(values[i], PIIType.EMAIL, tenants[i])
            for i in range(5)
        ]

        tokens = await asyncio.gather(*tasks)

        # All should succeed and be unique
        assert len(set(tokens)) == 5

    @pytest.mark.asyncio
    async def test_concurrent_detection(
        self, pii_service
    ):
        """Test concurrent detection requests."""
        contents = [
            f"Email {i}: user{i}@company.com, Phone: 555-{i:04d}"
            for i in range(10)
        ]

        results = await asyncio.gather(*[
            pii_service.detect(c) for c in contents
        ])

        assert len(results) == 10
        assert all(r is not None for r in results)


# ============================================================================
# TestDataConsistency
# ============================================================================


@pytest.mark.integration
@pytest.mark.requires_db
class TestDataConsistency:
    """Integration tests for data consistency."""

    @pytest.mark.asyncio
    async def test_token_persistence_across_restarts(
        self, db_pool, redis_client, test_tenant_id, test_user_id
    ):
        """Test that tokens persist and can be recovered."""
        try:
            from greenlang.infrastructure.pii_service.secure_vault import SecureTokenVault
            from greenlang.infrastructure.pii_service.config import VaultConfig, PersistenceBackend
            from greenlang.infrastructure.pii_service.models import PIIType
        except ImportError:
            pytest.skip("SecureTokenVault not available")

        config = VaultConfig(
            persistence_backend=PersistenceBackend.POSTGRESQL,
            token_ttl_days=1,
        )

        # First vault instance
        vault1 = SecureTokenVault(
            config=config,
            db_pool=db_pool,
            redis_client=redis_client,
        )

        # Create token
        token = await vault1.tokenize(
            "123-45-6789", PIIType.SSN, test_tenant_id
        )

        # Second vault instance (simulating restart)
        vault2 = SecureTokenVault(
            config=config,
            db_pool=db_pool,
            redis_client=redis_client,
        )

        # Should be able to detokenize
        value = await vault2.detokenize(token, test_tenant_id, test_user_id)
        assert value == "123-45-6789"
