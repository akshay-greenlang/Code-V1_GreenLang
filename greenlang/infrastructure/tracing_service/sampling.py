# -*- coding: utf-8 -*-
"""
GreenLangSampler - Custom sampling strategy for GreenLang Climate OS (OBS-003)

Implements a composite sampling strategy that:
1. Respects parent-based sampling decisions (W3C propagation).
2. Always samples compliance and security agent traces at 100%.
3. Applies per-service rate overrides.
4. Falls back to the configured default sampling rate.

The sampler is integrated at TracerProvider creation time and requires
no per-span intervention from application code.

Example:
    >>> from greenlang.infrastructure.tracing_service.sampling import create_sampler
    >>> from greenlang.infrastructure.tracing_service.config import TracingConfig
    >>> sampler = create_sampler(TracingConfig(sampling_rate=0.1))

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional OTel imports
# ---------------------------------------------------------------------------

try:
    from opentelemetry.sdk.trace.sampling import (
        Sampler,
        SamplingResult,
        Decision,
        ParentBased,
        TraceIdRatioBased,
        ALWAYS_ON,
        ALWAYS_OFF,
    )
    from opentelemetry.context import Context
    from opentelemetry.trace import Link, SpanKind
    from opentelemetry.util.types import Attributes

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    # Placeholder types so the module can be imported without OTel
    Sampler = object  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Service classification
# ---------------------------------------------------------------------------

# Compliance agents -- always sampled at 100 % for audit completeness.
COMPLIANCE_SERVICES: frozenset[str] = frozenset({
    "eudr-agent",
    "cbam-agent",
    "sb253-agent",
    "csrd-agent",
    "vcci-agent",
    "ghg-calculator",
    "emission-calculator",
    "compliance-engine",
})

# Security services -- always sampled at 100 %.
SECURITY_SERVICES: frozenset[str] = frozenset({
    "auth-service",
    "rbac-service",
    "audit-service",
    "encryption-service",
    "secrets-service",
    "pii-service",
    "security-scanning",
})

# Infrastructure services -- sampled at a lower rate by default.
INFRASTRUCTURE_SERVICES: frozenset[str] = frozenset({
    "otel-collector",
    "prometheus",
    "loki",
    "tempo",
})


# ---------------------------------------------------------------------------
# GreenLangSampler
# ---------------------------------------------------------------------------

if OTEL_AVAILABLE:

    class GreenLangSampler(Sampler):
        """Custom sampler with per-service-category rate overrides.

        Attributes:
            default_rate: Base sampling rate applied to uncategorised services.
            service_rates: Explicit per-service sampling rate overrides.
        """

        def __init__(
            self,
            default_rate: float = 1.0,
            service_rates: Optional[Dict[str, float]] = None,
        ) -> None:
            """Initialise the sampler.

            Args:
                default_rate: Sampling rate for services without an override.
                service_rates: Optional dict mapping service name to rate.
            """
            self._default_rate = max(0.0, min(1.0, default_rate))
            self._service_rates: Dict[str, float] = dict(service_rates or {})
            self._default_sampler = TraceIdRatioBased(self._default_rate)
            # Pre-build per-service samplers for efficiency.
            self._service_samplers: Dict[str, Sampler] = {
                svc: TraceIdRatioBased(max(0.0, min(1.0, rate)))
                for svc, rate in self._service_rates.items()
            }
            logger.debug(
                "GreenLangSampler: default_rate=%.2f overrides=%d",
                self._default_rate,
                len(self._service_rates),
            )

        # ---- Sampler interface -----------------------------------------------

        def should_sample(
            self,
            parent_context: Optional["Context"],
            trace_id: int,
            name: str,
            kind: Optional["SpanKind"] = None,
            attributes: "Attributes" = None,
            links: Optional[Sequence["Link"]] = None,
        ) -> "SamplingResult":
            """Decide whether to sample a trace.

            Decision logic:
            1. Compliance services -> ALWAYS sample.
            2. Security services  -> ALWAYS sample.
            3. Explicit per-service override -> use override rate.
            4. Infrastructure services -> sample at 10 % (or override).
            5. Everything else -> default rate.

            Args:
                parent_context: Parent span context (may be None).
                trace_id: 128-bit trace identifier.
                name: Span name.
                kind: Span kind (SERVER, CLIENT, etc.).
                attributes: Initial span attributes.
                links: Span links.

            Returns:
                SamplingResult with the sampling decision.
            """
            service_name = self._extract_service_name(attributes)

            # 1. Compliance -- always sample
            if service_name in COMPLIANCE_SERVICES:
                return SamplingResult(
                    Decision.RECORD_AND_SAMPLE,
                    attributes,
                    _trace_state_from_context(parent_context),
                )

            # 2. Security -- always sample
            if service_name in SECURITY_SERVICES:
                return SamplingResult(
                    Decision.RECORD_AND_SAMPLE,
                    attributes,
                    _trace_state_from_context(parent_context),
                )

            # 3. Explicit per-service override
            if service_name in self._service_samplers:
                return self._service_samplers[service_name].should_sample(
                    parent_context, trace_id, name, kind, attributes, links,
                )

            # 4. Infrastructure -- low default rate
            if service_name in INFRASTRUCTURE_SERVICES:
                infra_sampler = self._service_samplers.get(
                    service_name, TraceIdRatioBased(0.1),
                )
                return infra_sampler.should_sample(
                    parent_context, trace_id, name, kind, attributes, links,
                )

            # 5. Default rate
            return self._default_sampler.should_sample(
                parent_context, trace_id, name, kind, attributes, links,
            )

        def get_description(self) -> str:
            """Return a human-readable sampler description."""
            return (
                f"GreenLangSampler(default_rate={self._default_rate}, "
                f"overrides={len(self._service_rates)})"
            )

        # ---- Helpers ---------------------------------------------------------

        @staticmethod
        def _extract_service_name(attributes: "Attributes") -> str:
            """Extract service.name from span attributes if present."""
            if attributes is None:
                return ""
            if isinstance(attributes, dict):
                return str(attributes.get("service.name", ""))
            return ""

    def _trace_state_from_context(parent_context: Optional["Context"]) -> Any:
        """Extract TraceState from parent context, or return empty."""
        try:
            from opentelemetry.trace import get_current_span

            if parent_context is not None:
                parent_span = get_current_span(parent_context)  # type: ignore[arg-type]
                if parent_span is not None:
                    return parent_span.get_span_context().trace_state
        except Exception:
            pass
        from opentelemetry.trace import TraceState

        return TraceState()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_sampler(config: Any) -> Any:
    """Create the appropriate sampler based on TracingConfig.

    If OTel is not available, returns ``None``.  Otherwise wraps the
    ``GreenLangSampler`` inside a ``ParentBased`` sampler so that
    incoming propagated sampling decisions are respected.

    Args:
        config: A TracingConfig instance.

    Returns:
        A ParentBased(GreenLangSampler) or None.
    """
    if not OTEL_AVAILABLE:
        logger.debug("OTel not available; returning None sampler")
        return None

    root_sampler = GreenLangSampler(
        default_rate=config.sampling_rate,
        service_rates=getattr(config, "service_overrides", None) or {},
    )
    sampler = ParentBased(root=root_sampler)
    logger.info(
        "Created ParentBased(GreenLangSampler) sampler: default_rate=%.2f",
        config.sampling_rate,
    )
    return sampler


__all__ = [
    "GreenLangSampler",
    "create_sampler",
    "COMPLIANCE_SERVICES",
    "SECURITY_SERVICES",
    "INFRASTRUCTURE_SERVICES",
]
