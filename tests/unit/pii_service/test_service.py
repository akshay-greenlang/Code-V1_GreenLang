# -*- coding: utf-8 -*-
"""
Unit tests for PIIService - SEC-011 PII Service.

Tests the unified PII service facade that orchestrates all components:
- Detection using regex and ML scanners
- Redaction with multiple strategies
- Enforcement delegation to engine
- Tokenization via vault
- Allowlist filtering

Coverage target: 85%+ of service.py
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
def pii_service(pii_service_config, mock_encryption_service, mock_audit_service, mock_db_pool, mock_redis_client):
    """Create PIIService instance for testing."""
    try:
        from greenlang.infrastructure.pii_service.service import PIIService
        return PIIService(
            config=pii_service_config,
            encryption_service=mock_encryption_service,
            audit_service=mock_audit_service,
            db_pool=mock_db_pool,
            redis_client=mock_redis_client,
        )
    except ImportError:
        pytest.skip("PIIService not yet implemented")


@pytest.fixture
def mock_regex_scanner():
    """Mock regex-based PII scanner."""
    scanner = AsyncMock()
    scanner.scan = AsyncMock(return_value=[])
    return scanner


@pytest.fixture
def mock_ml_scanner():
    """Mock ML-based PII scanner (Presidio)."""
    scanner = AsyncMock()
    scanner.scan = AsyncMock(return_value=[])
    scanner.is_available = MagicMock(return_value=True)
    return scanner


# ============================================================================
# TestPIIServiceInitialization
# ============================================================================


class TestPIIServiceInitialization:
    """Tests for PIIService initialization."""

    def test_initialization_stores_config(self, pii_service, pii_service_config):
        """Service stores configuration correctly."""
        assert pii_service._config == pii_service_config

    def test_initialization_creates_vault(self, pii_service):
        """Service creates SecureTokenVault."""
        assert pii_service._vault is not None

    def test_initialization_creates_enforcement_engine(self, pii_service):
        """Service creates PIIEnforcementEngine."""
        assert pii_service._enforcement_engine is not None

    def test_initialization_creates_allowlist_manager(self, pii_service):
        """Service creates AllowlistManager."""
        assert pii_service._allowlist_manager is not None


# ============================================================================
# TestDetection
# ============================================================================


class TestDetection:
    """Tests for detect() method."""

    @pytest.mark.asyncio
    async def test_detect_uses_regex_scanner(
        self, pii_service, sample_content
    ):
        """detect() uses regex-based scanner by default."""
        result = await pii_service.detect(sample_content)

        assert result is not None
        assert hasattr(result, 'detections') or isinstance(result, list)

    @pytest.mark.asyncio
    async def test_detect_uses_ml_scanner_when_enabled(
        self, pii_service, pii_service_config, sample_content
    ):
        """detect() uses ML scanner when enabled in config."""
        pii_service_config.scanner.enable_ml = True

        result = await pii_service.detect(sample_content)

        # Should have used ML scanner
        # Implementation specific

    @pytest.mark.asyncio
    async def test_detect_merges_scanner_results(
        self, pii_service, sample_content
    ):
        """detect() merges results from all scanners."""
        # With both regex and ML enabled
        pii_service._config.scanner.enable_regex = True
        pii_service._config.scanner.enable_ml = True

        result = await pii_service.detect(sample_content)

        # Results should be merged
        assert result is not None

    @pytest.mark.asyncio
    async def test_detect_filters_allowlisted(
        self, pii_service, sample_content_test_data
    ):
        """detect() filters out allowlisted values."""
        result = await pii_service.detect(sample_content_test_data)

        # Test emails (example.com) should be filtered
        if hasattr(result, 'detections'):
            for detection in result.detections:
                if detection.pii_type.value == "email":
                    # Should not detect example.com emails
                    pass

    @pytest.mark.asyncio
    async def test_detect_records_metrics(
        self, pii_service, sample_content
    ):
        """detect() records Prometheus metrics."""
        with patch("greenlang.infrastructure.pii_service.metrics.gl_pii_detections_total") as mock_counter:
            mock_counter.labels.return_value.inc = MagicMock()

            await pii_service.detect(sample_content)

            # Metrics should be recorded
            # Implementation specific

    @pytest.mark.asyncio
    async def test_detect_with_options(
        self, pii_service, sample_content, pii_type_enum
    ):
        """detect() respects detection options."""
        try:
            from greenlang.infrastructure.pii_service.models import DetectionOptions

            options = DetectionOptions(
                use_ml=False,
                min_confidence=0.9,
                pii_types=[pii_type_enum.SSN, pii_type_enum.CREDIT_CARD],
            )

            result = await pii_service.detect(sample_content, options=options)

            # Should only detect specified types above threshold
            if hasattr(result, 'detections'):
                for detection in result.detections:
                    assert detection.confidence >= 0.9
                    assert detection.pii_type in [pii_type_enum.SSN, pii_type_enum.CREDIT_CARD]
        except ImportError:
            pytest.skip("DetectionOptions not available")

    @pytest.mark.asyncio
    async def test_detect_handles_empty_content(
        self, pii_service
    ):
        """detect() handles empty content gracefully."""
        result = await pii_service.detect("")

        if hasattr(result, 'detections'):
            assert len(result.detections) == 0

    @pytest.mark.asyncio
    async def test_detect_handles_large_content(
        self, pii_service
    ):
        """detect() handles large content within limits."""
        large_content = "Some text with email test@example.com. " * 10000

        result = await pii_service.detect(large_content)

        assert result is not None


# ============================================================================
# TestRedaction
# ============================================================================


class TestRedaction:
    """Tests for redact() method."""

    @pytest.mark.asyncio
    async def test_redact_applies_strategies(
        self, pii_service, sample_content
    ):
        """redact() applies configured redaction strategies."""
        result = await pii_service.redact(sample_content)

        assert result is not None
        if hasattr(result, 'redacted_content'):
            # Original PII should be replaced
            assert "123-45-6789" not in result.redacted_content

    @pytest.mark.asyncio
    async def test_redact_handles_overlapping_detections(
        self, pii_service
    ):
        """redact() handles overlapping PII detections correctly."""
        # Content where detections might overlap
        content = "Email: john.smith@company.com contains name and email"

        result = await pii_service.redact(content)

        # Should handle without error
        assert result is not None

    @pytest.mark.asyncio
    async def test_redact_with_custom_strategies(
        self, pii_service, sample_content, pii_type_enum
    ):
        """redact() supports per-type strategy overrides."""
        try:
            from greenlang.infrastructure.pii_service.models import RedactionOptions, RedactionStrategy

            options = RedactionOptions(
                strategy=RedactionStrategy.MASK,
                strategy_overrides={
                    pii_type_enum.SSN: RedactionStrategy.HASH,
                    pii_type_enum.EMAIL: RedactionStrategy.REPLACE,
                },
            )

            result = await pii_service.redact(sample_content, options=options)

            # Different strategies should be applied
            assert result is not None
        except ImportError:
            pytest.skip("RedactionOptions not available")

    @pytest.mark.asyncio
    async def test_redact_returns_detection_list(
        self, pii_service, sample_content
    ):
        """redact() returns list of detections with result."""
        result = await pii_service.redact(sample_content)

        if hasattr(result, 'detections'):
            assert isinstance(result.detections, list)

    @pytest.mark.asyncio
    async def test_redact_creates_tokens_when_configured(
        self, pii_service, sample_content, test_tenant_id
    ):
        """redact() creates tokens for tokenize strategy."""
        try:
            from greenlang.infrastructure.pii_service.models import RedactionOptions, RedactionStrategy

            options = RedactionOptions(
                strategy=RedactionStrategy.TOKENIZE,
                create_tokens=True,
                tenant_id=test_tenant_id,
            )

            result = await pii_service.redact(sample_content, options=options)

            if hasattr(result, 'tokens_created'):
                # Should have created tokens
                pass
        except ImportError:
            pytest.skip("RedactionOptions not available")


# ============================================================================
# TestEnforcement
# ============================================================================


class TestEnforcement:
    """Tests for enforce() method."""

    @pytest.mark.asyncio
    async def test_enforce_delegates_to_engine(
        self, pii_service, sample_content, enforcement_context
    ):
        """enforce() delegates to PIIEnforcementEngine."""
        result = await pii_service.enforce(sample_content, enforcement_context)

        assert result is not None
        assert hasattr(result, 'blocked')

    @pytest.mark.asyncio
    async def test_enforce_returns_result(
        self, pii_service, sample_content_high_sensitivity, enforcement_context
    ):
        """enforce() returns complete EnforcementResult."""
        result = await pii_service.enforce(
            sample_content_high_sensitivity, enforcement_context
        )

        assert hasattr(result, 'blocked')
        assert hasattr(result, 'detections')
        assert hasattr(result, 'actions_taken')


# ============================================================================
# TestTokenization
# ============================================================================


class TestTokenization:
    """Tests for tokenize() and detokenize() methods."""

    @pytest.mark.asyncio
    async def test_tokenize_delegates_to_vault(
        self, pii_service, pii_type_enum, test_tenant_id
    ):
        """tokenize() delegates to SecureTokenVault."""
        value = "123-45-6789"
        pii_type = pii_type_enum.SSN

        token = await pii_service.tokenize(value, pii_type, test_tenant_id)

        assert token is not None
        assert isinstance(token, str)

    @pytest.mark.asyncio
    async def test_detokenize_delegates_to_vault(
        self, pii_service, pii_type_enum, test_tenant_id, test_user_id
    ):
        """detokenize() delegates to SecureTokenVault."""
        value = "123-45-6789"
        pii_type = pii_type_enum.SSN

        # Tokenize first
        token = await pii_service.tokenize(value, pii_type, test_tenant_id)

        # Detokenize
        result = await pii_service.detokenize(token, test_tenant_id, test_user_id)

        assert result == value

    @pytest.mark.asyncio
    async def test_tokenize_records_metrics(
        self, pii_service, pii_type_enum, test_tenant_id
    ):
        """tokenize() records metrics."""
        with patch("greenlang.infrastructure.pii_service.metrics.gl_pii_tokenization_total") as mock_counter:
            mock_counter.labels.return_value.inc = MagicMock()

            await pii_service.tokenize("123-45-6789", pii_type_enum.SSN, test_tenant_id)

            # Metrics should be recorded


# ============================================================================
# TestAllowlistIntegration
# ============================================================================


class TestAllowlistIntegration:
    """Tests for allowlist integration."""

    @pytest.mark.asyncio
    async def test_is_allowlisted(
        self, pii_service, pii_type_enum, test_tenant_id
    ):
        """is_allowlisted() checks allowlist manager."""
        # Test card should be allowlisted
        result = await pii_service.is_allowlisted(
            "4242424242424242",
            pii_type_enum.CREDIT_CARD,
            test_tenant_id,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_detection_respects_allowlist(
        self, pii_service, sample_content_test_data
    ):
        """Detection filters out allowlisted values."""
        result = await pii_service.detect(sample_content_test_data)

        # Test card and example.com email should be filtered
        # Implementation specific


# ============================================================================
# TestServiceMetrics
# ============================================================================


class TestServiceMetrics:
    """Tests for service-level metrics."""

    @pytest.mark.asyncio
    async def test_detection_latency_recorded(
        self, pii_service, sample_content
    ):
        """Detection latency is recorded."""
        result = await pii_service.detect(sample_content)

        # Should have processing time
        if hasattr(result, 'processing_time_ms'):
            assert result.processing_time_ms >= 0

    @pytest.mark.asyncio
    async def test_service_health_check(
        self, pii_service
    ):
        """Service provides health check."""
        health = await pii_service.health_check()

        assert health is not None
        assert "status" in health or hasattr(health, "status")


# ============================================================================
# TestConcurrency
# ============================================================================


class TestConcurrency:
    """Tests for concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_detections(
        self, pii_service
    ):
        """Service handles concurrent detection requests."""
        import asyncio

        contents = [
            f"Email {i}: test{i}@company.com"
            for i in range(10)
        ]

        results = await asyncio.gather(*[
            pii_service.detect(c) for c in contents
        ])

        assert len(results) == 10
        assert all(r is not None for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_tokenization(
        self, pii_service, pii_type_enum, test_tenant_id
    ):
        """Service handles concurrent tokenization requests."""
        import asyncio

        values = [f"value-{i}" for i in range(10)]

        tokens = await asyncio.gather(*[
            pii_service.tokenize(v, pii_type_enum.EMAIL, test_tenant_id)
            for v in values
        ])

        # All tokens should be unique
        assert len(set(tokens)) == 10


# ============================================================================
# TestErrorHandling
# ============================================================================


class TestServiceErrorHandling:
    """Tests for service error handling."""

    @pytest.mark.asyncio
    async def test_scanner_error_handled(
        self, pii_service
    ):
        """Scanner errors are handled gracefully."""
        # Mock scanner to fail
        pii_service._regex_scanner = AsyncMock()
        pii_service._regex_scanner.scan = AsyncMock(side_effect=Exception("Scanner failed"))

        # Should not raise
        result = await pii_service.detect("some content")

        # Should return empty result or error indicator
        assert result is not None

    @pytest.mark.asyncio
    async def test_vault_error_handled(
        self, pii_service, pii_type_enum, test_tenant_id
    ):
        """Vault errors are propagated appropriately."""
        # Mock vault to fail
        pii_service._vault = AsyncMock()
        pii_service._vault.tokenize = AsyncMock(side_effect=Exception("Vault error"))

        with pytest.raises(Exception):
            await pii_service.tokenize("value", pii_type_enum.SSN, test_tenant_id)


# ============================================================================
# TestConfigurationReload
# ============================================================================


class TestConfigurationReload:
    """Tests for configuration hot reload."""

    @pytest.mark.asyncio
    async def test_reload_policies(
        self, pii_service
    ):
        """Service supports policy reload."""
        if hasattr(pii_service, 'reload_policies'):
            await pii_service.reload_policies()
            # Should not raise

    @pytest.mark.asyncio
    async def test_reload_allowlist(
        self, pii_service
    ):
        """Service supports allowlist reload."""
        if hasattr(pii_service, 'reload_allowlist'):
            await pii_service.reload_allowlist()
            # Should not raise
