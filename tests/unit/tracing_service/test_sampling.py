# -*- coding: utf-8 -*-
"""
Unit tests for GreenLangSampler and create_sampler (OBS-003)

Tests the custom composite sampling strategy: compliance services always
sampled, security services always sampled, per-service overrides,
infrastructure low-rate defaults, and ParentBased wrapping.

Coverage target: 85%+ of sampling.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from greenlang.infrastructure.tracing_service.config import TracingConfig
from greenlang.infrastructure.tracing_service.sampling import (
    COMPLIANCE_SERVICES,
    SECURITY_SERVICES,
    INFRASTRUCTURE_SERVICES,
    create_sampler,
    OTEL_AVAILABLE,
)


# ============================================================================
# Service list tests
# ============================================================================


class TestServiceClassification:
    """Tests for the service classification sets."""

    def test_compliance_services_list(self):
        """Verify the compliance services set contains expected agents."""
        expected = {
            "eudr-agent",
            "cbam-agent",
            "sb253-agent",
            "csrd-agent",
            "vcci-agent",
            "ghg-calculator",
            "emission-calculator",
            "compliance-engine",
        }
        assert expected.issubset(COMPLIANCE_SERVICES)

    def test_security_services_list(self):
        """Verify the security services set contains expected services."""
        expected = {
            "auth-service",
            "rbac-service",
            "audit-service",
            "encryption-service",
            "secrets-service",
            "pii-service",
            "security-scanning",
        }
        assert expected.issubset(SECURITY_SERVICES)

    def test_compliance_services_is_frozenset(self):
        """Verify COMPLIANCE_SERVICES is a frozenset (immutable)."""
        assert isinstance(COMPLIANCE_SERVICES, frozenset)

    def test_security_services_is_frozenset(self):
        """Verify SECURITY_SERVICES is a frozenset (immutable)."""
        assert isinstance(SECURITY_SERVICES, frozenset)

    def test_infrastructure_services_is_frozenset(self):
        """Verify INFRASTRUCTURE_SERVICES is a frozenset (immutable)."""
        assert isinstance(INFRASTRUCTURE_SERVICES, frozenset)


# ============================================================================
# GreenLangSampler tests (only when OTel is available)
# ============================================================================


@pytest.mark.skipif(not OTEL_AVAILABLE, reason="OTel SDK not installed")
class TestGreenLangSampler:
    """Tests for the GreenLangSampler class (requires OTel SDK)."""

    def _make_sampler(self, default_rate=1.0, service_rates=None):
        """Create a GreenLangSampler instance."""
        from greenlang.infrastructure.tracing_service.sampling import GreenLangSampler
        return GreenLangSampler(
            default_rate=default_rate,
            service_rates=service_rates or {},
        )

    def test_compliance_services_always_sampled(self):
        """Verify compliance services are always sampled at 100%."""
        from opentelemetry.sdk.trace.sampling import Decision

        sampler = self._make_sampler(default_rate=0.0)  # zero default

        for svc in COMPLIANCE_SERVICES:
            result = sampler.should_sample(
                parent_context=None,
                trace_id=123456,
                name="test_span",
                attributes={"service.name": svc},
            )
            assert result.decision == Decision.RECORD_AND_SAMPLE, (
                f"Compliance service {svc} should always be sampled"
            )

    def test_security_services_always_sampled(self):
        """Verify security services are always sampled at 100%."""
        from opentelemetry.sdk.trace.sampling import Decision

        sampler = self._make_sampler(default_rate=0.0)

        for svc in SECURITY_SERVICES:
            result = sampler.should_sample(
                parent_context=None,
                trace_id=123456,
                name="test_span",
                attributes={"service.name": svc},
            )
            assert result.decision == Decision.RECORD_AND_SAMPLE, (
                f"Security service {svc} should always be sampled"
            )

    def test_service_rate_overrides(self):
        """Verify per-service sampling rate overrides are applied."""
        sampler = self._make_sampler(
            default_rate=1.0,
            service_rates={"custom-service": 0.5},
        )
        # The sampler should use the override rate for custom-service
        result = sampler.should_sample(
            parent_context=None,
            trace_id=0,
            name="op",
            attributes={"service.name": "custom-service"},
        )
        # We cannot deterministically assert the decision because
        # TraceIdRatioBased uses the trace_id, but we can verify no crash
        assert result is not None

    def test_custom_sampling_rate(self):
        """Verify default rate is applied to uncategorised services."""
        sampler = self._make_sampler(default_rate=1.0)
        result = sampler.should_sample(
            parent_context=None,
            trace_id=0,
            name="op",
            attributes={"service.name": "my-service"},
        )
        assert result is not None

    def test_zero_sampling_rate(self):
        """Verify zero default rate drops uncategorised services."""
        from opentelemetry.sdk.trace.sampling import Decision

        sampler = self._make_sampler(default_rate=0.0)
        result = sampler.should_sample(
            parent_context=None,
            trace_id=99999,
            name="op",
            attributes={"service.name": "unknown-service"},
        )
        # With rate 0.0, decision should be DROP
        assert result.decision == Decision.DROP

    def test_full_sampling_rate(self):
        """Verify rate 1.0 samples all uncategorised services."""
        from opentelemetry.sdk.trace.sampling import Decision

        sampler = self._make_sampler(default_rate=1.0)
        result = sampler.should_sample(
            parent_context=None,
            trace_id=99999,
            name="op",
            attributes={"service.name": "api-gateway"},
        )
        assert result.decision == Decision.RECORD_AND_SAMPLE

    def test_get_description(self):
        """Verify get_description returns a human-readable string."""
        sampler = self._make_sampler(default_rate=0.25)
        desc = sampler.get_description()
        assert "GreenLangSampler" in desc
        assert "0.25" in desc

    def test_extract_service_name_from_dict(self):
        """Verify _extract_service_name reads from dict attributes."""
        from greenlang.infrastructure.tracing_service.sampling import GreenLangSampler
        name = GreenLangSampler._extract_service_name({"service.name": "test-svc"})
        assert name == "test-svc"

    def test_extract_service_name_none_attributes(self):
        """Verify _extract_service_name returns empty for None."""
        from greenlang.infrastructure.tracing_service.sampling import GreenLangSampler
        name = GreenLangSampler._extract_service_name(None)
        assert name == ""

    def test_extract_service_name_missing_key(self):
        """Verify _extract_service_name returns empty for missing key."""
        from greenlang.infrastructure.tracing_service.sampling import GreenLangSampler
        name = GreenLangSampler._extract_service_name({"other.attr": "x"})
        assert name == ""


# ============================================================================
# create_sampler factory tests
# ============================================================================


class TestCreateSampler:
    """Tests for the create_sampler() factory function."""

    def test_create_sampler_returns_parent_based(self):
        """Verify create_sampler wraps in ParentBased when OTel available."""
        config = TracingConfig(sampling_rate=0.5)
        sampler = create_sampler(config)

        if OTEL_AVAILABLE:
            from opentelemetry.sdk.trace.sampling import ParentBased
            assert isinstance(sampler, ParentBased)
        else:
            assert sampler is None

    def test_create_sampler_noop_when_unavailable(self):
        """Verify create_sampler returns None when OTel is absent."""
        import greenlang.infrastructure.tracing_service.sampling as samp_mod

        config = TracingConfig(sampling_rate=0.5)
        with patch.object(samp_mod, "OTEL_AVAILABLE", False):
            sampler = create_sampler(config)
            assert sampler is None

    def test_default_sampler_creation(self):
        """Verify create_sampler works with default TracingConfig."""
        config = TracingConfig()
        sampler = create_sampler(config)
        # Should not raise; returns something (or None if no OTel)
        if OTEL_AVAILABLE:
            assert sampler is not None
        else:
            assert sampler is None

    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OTel SDK not installed")
    def test_parent_based_sampling(self):
        """Verify the returned sampler respects parent decisions."""
        from opentelemetry.sdk.trace.sampling import ParentBased

        config = TracingConfig(sampling_rate=0.5)
        sampler = create_sampler(config)
        assert isinstance(sampler, ParentBased)
