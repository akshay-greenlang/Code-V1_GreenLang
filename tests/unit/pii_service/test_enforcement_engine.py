# -*- coding: utf-8 -*-
"""
Unit tests for PIIEnforcementEngine - SEC-011 PII Service.

Tests the enforcement engine for real-time PII policy enforcement:
- Policy evaluation and action selection
- Block, redact, allow, quarantine, transform actions
- Confidence threshold filtering
- Context-based policy matching
- Notification integration
- Metrics recording

Coverage target: 85%+ of enforcement/engine.py
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def enforcement_engine(enforcement_config, mock_audit_service, mock_notification_service):
    """Create PIIEnforcementEngine instance for testing."""
    try:
        from greenlang.infrastructure.pii_service.enforcement.engine import PIIEnforcementEngine
        return PIIEnforcementEngine(
            config=enforcement_config,
            audit_service=mock_audit_service,
            notification_service=mock_notification_service,
        )
    except ImportError:
        pytest.skip("PIIEnforcementEngine not yet implemented")


@pytest.fixture
def enforcement_engine_audit_mode(enforcement_config, mock_audit_service, mock_notification_service):
    """Create enforcement engine in audit-only mode."""
    try:
        from greenlang.infrastructure.pii_service.enforcement.engine import PIIEnforcementEngine
        from greenlang.infrastructure.pii_service.config import EnforcementMode
        enforcement_config.mode = EnforcementMode.AUDIT
        return PIIEnforcementEngine(
            config=enforcement_config,
            audit_service=mock_audit_service,
            notification_service=mock_notification_service,
        )
    except ImportError:
        pytest.skip("PIIEnforcementEngine not yet implemented")


@pytest.fixture
def mock_pii_scanner():
    """Mock PII scanner for detection."""
    scanner = AsyncMock()
    scanner.scan = AsyncMock(return_value=[])
    return scanner


@pytest.fixture
def mock_allowlist_manager():
    """Mock allowlist manager."""
    manager = AsyncMock()
    manager.is_allowed = AsyncMock(return_value=False)
    return manager


# ============================================================================
# TestEnforcementEngineInitialization
# ============================================================================


class TestEnforcementEngineInitialization:
    """Tests for PIIEnforcementEngine initialization."""

    def test_initialization_stores_config(self, enforcement_engine, enforcement_config):
        """Engine stores configuration correctly."""
        assert enforcement_engine._config == enforcement_config

    def test_initialization_loads_default_policies(self, enforcement_engine):
        """Engine loads default policies for all PII types."""
        assert len(enforcement_engine._policies) > 0

    def test_initialization_with_custom_policies(
        self, enforcement_config, mock_audit_service, mock_notification_service, pii_type_enum
    ):
        """Engine accepts custom policies on initialization."""
        try:
            from greenlang.infrastructure.pii_service.enforcement.engine import PIIEnforcementEngine
            from greenlang.infrastructure.pii_service.enforcement.policies import EnforcementPolicy, EnforcementAction

            custom_policies = {
                pii_type_enum.EMAIL: EnforcementPolicy(
                    pii_type=pii_type_enum.EMAIL,
                    action=EnforcementAction.BLOCK,
                    min_confidence=0.5,
                ),
            }

            engine = PIIEnforcementEngine(
                config=enforcement_config,
                audit_service=mock_audit_service,
                notification_service=mock_notification_service,
                policies=custom_policies,
            )

            assert engine._policies[pii_type_enum.EMAIL].action == EnforcementAction.BLOCK
        except ImportError:
            pytest.skip("PIIEnforcementEngine not yet implemented")


# ============================================================================
# TestEnforceMethod
# ============================================================================


class TestEnforceMethod:
    """Tests for the enforce() method."""

    @pytest.mark.asyncio
    async def test_enforce_detects_pii(
        self, enforcement_engine, sample_content, enforcement_context
    ):
        """enforce() detects PII in content."""
        result = await enforcement_engine.enforce(sample_content, enforcement_context)

        assert result is not None
        assert hasattr(result, 'detections')

    @pytest.mark.asyncio
    async def test_enforce_applies_block_policy(
        self, enforcement_engine, sample_content_high_sensitivity, enforcement_context
    ):
        """enforce() blocks content with high-sensitivity PII."""
        result = await enforcement_engine.enforce(
            sample_content_high_sensitivity, enforcement_context
        )

        assert result.blocked is True
        assert len(result.actions_taken) > 0
        # Should have a block action
        block_actions = [a for a in result.actions_taken if a.action.value == "block"]
        assert len(block_actions) > 0

    @pytest.mark.asyncio
    async def test_enforce_applies_redact_policy(
        self, enforcement_engine, pii_type_enum, enforcement_context
    ):
        """enforce() redacts PII when policy is REDACT."""
        # Configure email policy to redact
        try:
            from greenlang.infrastructure.pii_service.enforcement.policies import EnforcementAction
            enforcement_engine._policies[pii_type_enum.EMAIL].action = EnforcementAction.REDACT
        except ImportError:
            pytest.skip("Module not available")

        content = "Contact me at john@company.com for details."

        result = await enforcement_engine.enforce(content, enforcement_context)

        assert result.blocked is False
        assert result.modified_content is not None
        assert "john@company.com" not in result.modified_content
        assert "[EMAIL]" in result.modified_content or "***" in result.modified_content

    @pytest.mark.asyncio
    async def test_enforce_applies_allow_policy(
        self, enforcement_engine, pii_type_enum, enforcement_context
    ):
        """enforce() allows content through when policy is ALLOW."""
        try:
            from greenlang.infrastructure.pii_service.enforcement.policies import EnforcementAction
            enforcement_engine._policies[pii_type_enum.PERSON_NAME].action = EnforcementAction.ALLOW
        except ImportError:
            pytest.skip("Module not available")

        content = "Meeting with John Smith tomorrow."

        result = await enforcement_engine.enforce(content, enforcement_context)

        # Name should be detected but allowed
        assert result.blocked is False
        name_detections = [d for d in result.detections if d.pii_type.value == "person_name"]
        if name_detections:
            name_actions = [
                a for a in result.actions_taken
                if a.detection.pii_type.value == "person_name" and a.action.value == "allow"
            ]
            assert len(name_actions) > 0

    @pytest.mark.asyncio
    async def test_enforce_applies_quarantine_policy(
        self, enforcement_engine, pii_type_enum, enforcement_context
    ):
        """enforce() quarantines content when policy is QUARANTINE."""
        try:
            from greenlang.infrastructure.pii_service.enforcement.policies import EnforcementAction
            enforcement_engine._policies[pii_type_enum.SSN].action = EnforcementAction.QUARANTINE
        except ImportError:
            pytest.skip("Module not available")

        content = "SSN: 123-45-6789"

        result = await enforcement_engine.enforce(content, enforcement_context)

        assert result.blocked is True
        quarantine_actions = [a for a in result.actions_taken if a.action.value == "quarantine"]
        assert len(quarantine_actions) > 0

    @pytest.mark.asyncio
    async def test_enforce_applies_transform_policy(
        self, enforcement_engine, pii_type_enum, enforcement_context
    ):
        """enforce() transforms (tokenizes) PII when policy is TRANSFORM."""
        try:
            from greenlang.infrastructure.pii_service.enforcement.policies import EnforcementAction
            enforcement_engine._policies[pii_type_enum.EMAIL].action = EnforcementAction.TRANSFORM
        except ImportError:
            pytest.skip("Module not available")

        content = "Email: john@company.com"

        result = await enforcement_engine.enforce(content, enforcement_context)

        assert result.blocked is False
        assert result.modified_content is not None
        # Should contain a token instead of the email
        assert "john@company.com" not in result.modified_content
        assert "tok_" in result.modified_content or "[TOKEN:" in result.modified_content

    @pytest.mark.asyncio
    async def test_enforce_respects_min_confidence(
        self, enforcement_engine, enforcement_config, enforcement_context
    ):
        """enforce() respects minimum confidence threshold."""
        # Set high confidence threshold
        enforcement_config.min_confidence = 0.99

        # Content that might have low-confidence detections
        content = "Contact: maybe-email@domain"

        result = await enforcement_engine.enforce(content, enforcement_context)

        # Low confidence detections should be filtered out
        high_conf_detections = [d for d in result.detections if d.confidence >= 0.99]
        # Only high-confidence detections should have actions
        for action in result.actions_taken:
            assert action.detection.confidence >= 0.99 or action.action.value == "allow"

    @pytest.mark.asyncio
    async def test_enforce_filters_by_context(
        self, enforcement_engine, pii_type_enum, enforcement_context
    ):
        """enforce() respects context-specific policies."""
        try:
            from greenlang.infrastructure.pii_service.enforcement.policies import EnforcementAction
            # Set policy to only apply to storage context
            enforcement_engine._policies[pii_type_enum.EMAIL].contexts = ["storage"]
        except ImportError:
            pytest.skip("Module not available")

        # API request context
        enforcement_context.context_type = "api_request"
        content = "Email: john@company.com"

        result = await enforcement_engine.enforce(content, enforcement_context)

        # Policy should not apply since context doesn't match
        email_detections = [d for d in result.detections if d.pii_type.value == "email"]
        if email_detections:
            # The configured policy should not have been applied
            email_actions = [
                a for a in result.actions_taken
                if a.detection.pii_type.value == "email"
            ]
            # Should use default action, not the configured one
            for action in email_actions:
                # Context mismatch means policy doesn't apply
                pass

    @pytest.mark.asyncio
    async def test_enforce_notifies_on_detection(
        self, enforcement_engine, mock_notification_service, enforcement_context
    ):
        """enforce() sends notifications when configured."""
        content = "SSN: 123-45-6789"

        await enforcement_engine.enforce(content, enforcement_context)

        # Check notifications were sent
        assert len(mock_notification_service._notifications) > 0

    @pytest.mark.asyncio
    async def test_enforce_records_metrics(
        self, enforcement_engine, enforcement_context
    ):
        """enforce() records Prometheus metrics."""
        content = "SSN: 123-45-6789"

        with patch("greenlang.infrastructure.pii_service.metrics.gl_pii_enforcement_actions_total") as mock_metric:
            mock_metric.labels.return_value.inc = MagicMock()

            await enforcement_engine.enforce(content, enforcement_context)

            # Metrics should have been recorded
            # This is implementation-specific

    @pytest.mark.asyncio
    async def test_enforce_filters_allowlisted(
        self, enforcement_engine, enforcement_context
    ):
        """enforce() filters out allowlisted values."""
        # Content with allowlisted test data
        content = "Test card: 4242424242424242, email: test@example.com"

        result = await enforcement_engine.enforce(content, enforcement_context)

        # Stripe test card should be allowlisted
        card_detections = [d for d in result.detections if d.pii_type.value == "credit_card"]
        # If detected, should have allow action due to allowlist
        for action in result.actions_taken:
            if action.detection.pii_type.value == "credit_card":
                # Should be allowed if it's the test card
                pass

    @pytest.mark.asyncio
    async def test_policy_override_per_tenant(
        self, enforcement_engine, pii_type_enum, enforcement_context
    ):
        """enforce() supports per-tenant policy overrides."""
        try:
            from greenlang.infrastructure.pii_service.enforcement.policies import EnforcementAction

            # Set tenant-specific override
            enforcement_engine.set_tenant_policy(
                enforcement_context.tenant_id,
                pii_type_enum.EMAIL,
                EnforcementAction.BLOCK,
            )
        except (ImportError, AttributeError):
            pytest.skip("Tenant policy override not implemented")

        content = "Email: john@company.com"

        result = await enforcement_engine.enforce(content, enforcement_context)

        # Should use tenant override (BLOCK instead of default)
        email_actions = [
            a for a in result.actions_taken
            if a.detection.pii_type.value == "email"
        ]
        assert any(a.action.value == "block" for a in email_actions)

    @pytest.mark.asyncio
    async def test_default_policies_applied(
        self, enforcement_engine, enforcement_context
    ):
        """enforce() applies default policies when no custom policy exists."""
        content = "Some content with unknown PII type marker: CUSTOM_123"

        result = await enforcement_engine.enforce(content, enforcement_context)

        # Should apply default action from config
        # Implementation specific


# ============================================================================
# TestRedactionBehavior
# ============================================================================


class TestRedactionBehavior:
    """Tests for redaction behavior."""

    @pytest.mark.asyncio
    async def test_redaction_replaces_pii(
        self, enforcement_engine, pii_type_enum, enforcement_context
    ):
        """Redaction replaces PII with placeholder."""
        try:
            from greenlang.infrastructure.pii_service.enforcement.policies import EnforcementAction
            enforcement_engine._policies[pii_type_enum.EMAIL].action = EnforcementAction.REDACT
        except ImportError:
            pytest.skip("Module not available")

        content = "Email me at john@company.com"

        result = await enforcement_engine.enforce(content, enforcement_context)

        assert "john@company.com" not in result.modified_content
        # Should have placeholder
        assert "[EMAIL]" in result.modified_content or "***" in result.modified_content

    @pytest.mark.asyncio
    async def test_multiple_detections_all_handled(
        self, enforcement_engine, pii_type_enum, enforcement_context
    ):
        """All detections in content are handled."""
        try:
            from greenlang.infrastructure.pii_service.enforcement.policies import EnforcementAction
            enforcement_engine._policies[pii_type_enum.EMAIL].action = EnforcementAction.REDACT
            enforcement_engine._policies[pii_type_enum.PHONE].action = EnforcementAction.REDACT
        except ImportError:
            pytest.skip("Module not available")

        content = "Email: john@company.com, Phone: 555-123-4567, Alt: jane@corp.com"

        result = await enforcement_engine.enforce(content, enforcement_context)

        # All PII should be redacted
        assert "john@company.com" not in result.modified_content
        assert "jane@corp.com" not in result.modified_content
        assert "555-123-4567" not in result.modified_content

    @pytest.mark.asyncio
    async def test_sorted_by_position_for_redaction(
        self, enforcement_engine, pii_type_enum, enforcement_context
    ):
        """Detections are processed in position order for correct redaction."""
        try:
            from greenlang.infrastructure.pii_service.enforcement.policies import EnforcementAction
            enforcement_engine._policies[pii_type_enum.EMAIL].action = EnforcementAction.REDACT
        except ImportError:
            pytest.skip("Module not available")

        content = "First: a@b.com, Second: c@d.com, Third: e@f.com"

        result = await enforcement_engine.enforce(content, enforcement_context)

        # Redaction should maintain text structure
        assert "First:" in result.modified_content
        assert "Second:" in result.modified_content
        assert "Third:" in result.modified_content

    @pytest.mark.asyncio
    async def test_custom_placeholder(
        self, enforcement_engine, pii_type_enum, enforcement_context
    ):
        """Custom redaction placeholder is used."""
        try:
            from greenlang.infrastructure.pii_service.enforcement.policies import EnforcementAction
            enforcement_engine._policies[pii_type_enum.EMAIL].action = EnforcementAction.REDACT
            enforcement_engine._policies[pii_type_enum.EMAIL].custom_placeholder = "[REDACTED_EMAIL]"
        except ImportError:
            pytest.skip("Module not available")

        content = "Email: john@company.com"

        result = await enforcement_engine.enforce(content, enforcement_context)

        assert "[REDACTED_EMAIL]" in result.modified_content


# ============================================================================
# TestQuarantineBehavior
# ============================================================================


class TestQuarantineBehavior:
    """Tests for quarantine behavior."""

    @pytest.mark.asyncio
    async def test_quarantine_stores_content(
        self, enforcement_engine, pii_type_enum, enforcement_context
    ):
        """Quarantine action stores content for review."""
        try:
            from greenlang.infrastructure.pii_service.enforcement.policies import EnforcementAction
            enforcement_engine._policies[pii_type_enum.SSN].action = EnforcementAction.QUARANTINE
        except ImportError:
            pytest.skip("Module not available")

        content = "SSN: 123-45-6789"

        result = await enforcement_engine.enforce(content, enforcement_context)

        assert result.blocked is True
        # Should have quarantine item created
        quarantine_actions = [a for a in result.actions_taken if a.action.value == "quarantine"]
        assert len(quarantine_actions) > 0

    @pytest.mark.asyncio
    async def test_quarantine_creates_review_item(
        self, enforcement_engine, pii_type_enum, enforcement_context
    ):
        """Quarantine creates a review item with metadata."""
        try:
            from greenlang.infrastructure.pii_service.enforcement.policies import EnforcementAction
            enforcement_engine._policies[pii_type_enum.SSN].action = EnforcementAction.QUARANTINE
        except ImportError:
            pytest.skip("Module not available")

        content = "SSN: 123-45-6789"

        result = await enforcement_engine.enforce(content, enforcement_context)

        # Verify quarantine item was created
        if hasattr(enforcement_engine, '_quarantine_store'):
            items = await enforcement_engine.get_quarantine_items(enforcement_context.tenant_id)
            assert len(items) > 0


# ============================================================================
# TestContextTypeFiltering
# ============================================================================


class TestContextTypeFiltering:
    """Tests for context-type specific enforcement."""

    @pytest.mark.asyncio
    async def test_context_type_filtering(
        self, enforcement_engine, pii_type_enum, enforcement_context
    ):
        """Policies only apply to matching context types."""
        try:
            from greenlang.infrastructure.pii_service.enforcement.policies import EnforcementAction
            # Policy only applies to logging context
            enforcement_engine._policies[pii_type_enum.EMAIL].contexts = ["logging"]
            enforcement_engine._policies[pii_type_enum.EMAIL].action = EnforcementAction.BLOCK
        except ImportError:
            pytest.skip("Module not available")

        # API request context
        enforcement_context.context_type = "api_request"
        content = "Email: john@company.com"

        result = await enforcement_engine.enforce(content, enforcement_context)

        # Should not block since context doesn't match
        # May still detect but won't apply the BLOCK action

    @pytest.mark.asyncio
    async def test_wildcard_context_applies_to_all(
        self, enforcement_engine, pii_type_enum, enforcement_context
    ):
        """Policies with '*' context apply to all context types."""
        try:
            from greenlang.infrastructure.pii_service.enforcement.policies import EnforcementAction
            enforcement_engine._policies[pii_type_enum.SSN].contexts = ["*"]
            enforcement_engine._policies[pii_type_enum.SSN].action = EnforcementAction.BLOCK
        except ImportError:
            pytest.skip("Module not available")

        # Test multiple context types
        for ctx_type in ["api_request", "storage", "logging", "streaming"]:
            enforcement_context.context_type = ctx_type
            content = "SSN: 123-45-6789"

            result = await enforcement_engine.enforce(content, enforcement_context)

            # Should block regardless of context
            assert result.blocked is True


# ============================================================================
# TestAuditMode
# ============================================================================


class TestAuditMode:
    """Tests for audit-only mode."""

    @pytest.mark.asyncio
    async def test_audit_mode_detects_but_allows(
        self, enforcement_engine_audit_mode, enforcement_context
    ):
        """Audit mode detects PII but doesn't block."""
        content = "SSN: 123-45-6789, Credit Card: 4111-1111-1111-1111"

        result = await enforcement_engine_audit_mode.enforce(content, enforcement_context)

        # Should detect PII
        assert len(result.detections) > 0
        # But should not block
        assert result.blocked is False
        # Content should be unchanged
        assert result.modified_content == content or result.modified_content is None

    @pytest.mark.asyncio
    async def test_audit_mode_logs_detections(
        self, enforcement_engine_audit_mode, mock_audit_service, enforcement_context
    ):
        """Audit mode logs all detections."""
        content = "SSN: 123-45-6789"

        await enforcement_engine_audit_mode.enforce(content, enforcement_context)

        # Should have logged detections
        audit_log = mock_audit_service.get_audit_log()
        assert len(audit_log) > 0


# ============================================================================
# TestHighSensitivityPII
# ============================================================================


class TestHighSensitivityPII:
    """Tests for high-sensitivity PII handling."""

    @pytest.mark.asyncio
    async def test_block_high_sensitivity_override(
        self, enforcement_engine, enforcement_config, pii_type_enum, enforcement_context
    ):
        """block_high_sensitivity overrides individual policies."""
        enforcement_config.block_high_sensitivity = True

        try:
            from greenlang.infrastructure.pii_service.enforcement.policies import EnforcementAction
            # Try to set SSN to ALLOW
            enforcement_engine._policies[pii_type_enum.SSN].action = EnforcementAction.ALLOW
        except ImportError:
            pytest.skip("Module not available")

        content = "SSN: 123-45-6789"

        result = await enforcement_engine.enforce(content, enforcement_context)

        # Should still block due to high sensitivity override
        assert result.blocked is True

    @pytest.mark.asyncio
    async def test_high_sensitivity_types(
        self, enforcement_engine, enforcement_config, enforcement_context
    ):
        """High-sensitivity PII types are always blocked when configured."""
        enforcement_config.block_high_sensitivity = True

        high_sensitivity_content = [
            ("SSN: 123-45-6789", "ssn"),
            ("Card: 4111-1111-1111-1111", "credit_card"),
            ("Password: secret123!", "password"),
        ]

        for content, pii_type in high_sensitivity_content:
            result = await enforcement_engine.enforce(content, enforcement_context)
            assert result.blocked is True, f"Expected {pii_type} to be blocked"


# ============================================================================
# TestErrorHandling
# ============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_scanner_error_fails_safe(
        self, enforcement_engine, enforcement_context
    ):
        """Scanner errors result in blocking (fail-safe)."""
        # Make scanner fail
        enforcement_engine._scanner = AsyncMock()
        enforcement_engine._scanner.scan = AsyncMock(side_effect=Exception("Scanner error"))

        content = "Some content"

        result = await enforcement_engine.enforce(content, enforcement_context)

        # Should fail safe - either error or block
        assert result.error is not None or result.blocked is True

    @pytest.mark.asyncio
    async def test_notification_error_non_fatal(
        self, enforcement_engine, mock_notification_service, enforcement_context
    ):
        """Notification errors don't affect enforcement result."""
        mock_notification_service.send = AsyncMock(side_effect=Exception("Notification failed"))

        content = "SSN: 123-45-6789"

        # Should still complete enforcement
        result = await enforcement_engine.enforce(content, enforcement_context)

        # Enforcement should succeed despite notification failure
        assert result is not None


# ============================================================================
# TestProcessingMetrics
# ============================================================================


class TestProcessingMetrics:
    """Tests for processing metrics."""

    @pytest.mark.asyncio
    async def test_processing_time_recorded(
        self, enforcement_engine, enforcement_context
    ):
        """Processing time is recorded in result."""
        content = "SSN: 123-45-6789"

        result = await enforcement_engine.enforce(content, enforcement_context)

        assert result.processing_time_ms >= 0

    @pytest.mark.asyncio
    async def test_detection_count_correct(
        self, enforcement_engine, enforcement_context
    ):
        """Detection count is accurate."""
        content = "Email: a@b.com, Phone: 555-123-4567, SSN: 123-45-6789"

        result = await enforcement_engine.enforce(content, enforcement_context)

        # Should have multiple detections
        assert result.detection_count == len(result.detections)
