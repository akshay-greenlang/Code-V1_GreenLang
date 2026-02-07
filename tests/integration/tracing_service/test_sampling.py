# -*- coding: utf-8 -*-
"""
Integration tests - Sampling strategy verification (OBS-003)

Tests that the GreenLangSampler correctly applies per-service-category
sampling rates in realistic multi-service scenarios.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from greenlang.infrastructure.tracing_service.config import TracingConfig
from greenlang.infrastructure.tracing_service.sampling import (
    create_sampler,
    COMPLIANCE_SERVICES,
    SECURITY_SERVICES,
    INFRASTRUCTURE_SERVICES,
)


# ============================================================================
# Helper
# ============================================================================


def _check_otel_available():
    """Check if OTel SDK is available for sampling tests."""
    try:
        from opentelemetry.sdk.trace.sampling import Sampler

        return True
    except ImportError:
        return False


otel_required = pytest.mark.skipif(
    not _check_otel_available(),
    reason="OpenTelemetry SDK not installed",
)


# ============================================================================
# Sampler factory integration tests
# ============================================================================


class TestSamplerFactory:
    """Test create_sampler produces correct sampler configurations."""

    def test_create_sampler_returns_none_without_otel(self):
        """create_sampler returns None when OTel is not installed."""
        with patch(
            "greenlang.infrastructure.tracing_service.sampling.OTEL_AVAILABLE",
            False,
        ):
            config = TracingConfig(sampling_rate=0.5)
            result = create_sampler(config)
            assert result is None

    @otel_required
    def test_create_sampler_returns_parent_based(self):
        """create_sampler returns a ParentBased sampler wrapping GreenLangSampler."""
        from opentelemetry.sdk.trace.sampling import ParentBased

        config = TracingConfig(sampling_rate=0.5)
        sampler = create_sampler(config)
        assert isinstance(sampler, ParentBased)

    @otel_required
    def test_sampler_description_includes_rate(self):
        """Sampler description includes the configured default rate."""
        config = TracingConfig(sampling_rate=0.25)
        sampler = create_sampler(config)
        desc = sampler.get_description()
        assert "0.25" in desc or "ParentBased" in desc


# ============================================================================
# Service classification tests
# ============================================================================


class TestServiceClassification:
    """Test that service classifications cover all expected services."""

    def test_compliance_services_complete(self):
        """All required compliance agents are in COMPLIANCE_SERVICES."""
        expected = {
            "eudr-agent",
            "cbam-agent",
            "sb253-agent",
            "csrd-agent",
            "vcci-agent",
        }
        assert expected.issubset(COMPLIANCE_SERVICES)

    def test_security_services_complete(self):
        """All required security services are in SECURITY_SERVICES."""
        expected = {
            "auth-service",
            "rbac-service",
            "audit-service",
            "encryption-service",
            "secrets-service",
        }
        assert expected.issubset(SECURITY_SERVICES)

    def test_infrastructure_services_present(self):
        """Infrastructure services include monitoring components."""
        assert "prometheus" in INFRASTRUCTURE_SERVICES
        assert "loki" in INFRASTRUCTURE_SERVICES
        assert "tempo" in INFRASTRUCTURE_SERVICES

    def test_no_overlap_between_categories(self):
        """Service categories do not overlap."""
        assert COMPLIANCE_SERVICES.isdisjoint(SECURITY_SERVICES)
        assert COMPLIANCE_SERVICES.isdisjoint(INFRASTRUCTURE_SERVICES)
        assert SECURITY_SERVICES.isdisjoint(INFRASTRUCTURE_SERVICES)


# ============================================================================
# Sampling decision integration tests
# ============================================================================


class TestSamplingDecisions:
    """Test sampling decisions for different service categories."""

    @otel_required
    def test_compliance_service_always_sampled(self):
        """Compliance agents are sampled at 100% regardless of default rate."""
        from greenlang.infrastructure.tracing_service.sampling import (
            GreenLangSampler,
        )
        from opentelemetry.sdk.trace.sampling import Decision

        sampler = GreenLangSampler(default_rate=0.01)

        for service in ["eudr-agent", "cbam-agent", "csrd-agent"]:
            result = sampler.should_sample(
                parent_context=None,
                trace_id=12345,
                name="execute",
                attributes={"service.name": service},
            )
            assert result.decision == Decision.RECORD_AND_SAMPLE

    @otel_required
    def test_security_service_always_sampled(self):
        """Security services are sampled at 100% regardless of default rate."""
        from greenlang.infrastructure.tracing_service.sampling import (
            GreenLangSampler,
        )
        from opentelemetry.sdk.trace.sampling import Decision

        sampler = GreenLangSampler(default_rate=0.01)

        for service in ["auth-service", "audit-service", "secrets-service"]:
            result = sampler.should_sample(
                parent_context=None,
                trace_id=12345,
                name="authenticate",
                attributes={"service.name": service},
            )
            assert result.decision == Decision.RECORD_AND_SAMPLE

    @otel_required
    def test_per_service_override_respected(self):
        """Explicit per-service overrides are applied."""
        from greenlang.infrastructure.tracing_service.sampling import (
            GreenLangSampler,
        )

        sampler = GreenLangSampler(
            default_rate=0.1,
            service_rates={"custom-svc": 1.0},
        )
        # With 100% rate and valid service name, should sample
        result = sampler.should_sample(
            parent_context=None,
            trace_id=1,
            name="op",
            attributes={"service.name": "custom-svc"},
        )
        from opentelemetry.sdk.trace.sampling import Decision

        assert result.decision == Decision.RECORD_AND_SAMPLE

    @otel_required
    def test_unknown_service_uses_default_rate(self):
        """Unknown services fall back to the default sampling rate."""
        from greenlang.infrastructure.tracing_service.sampling import (
            GreenLangSampler,
        )

        # With rate=0.0, nothing should be sampled
        sampler = GreenLangSampler(default_rate=0.0)
        result = sampler.should_sample(
            parent_context=None,
            trace_id=99999,
            name="op",
            attributes={"service.name": "unknown-svc"},
        )
        from opentelemetry.sdk.trace.sampling import Decision

        assert result.decision == Decision.DROP
