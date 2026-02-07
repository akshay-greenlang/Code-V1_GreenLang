# -*- coding: utf-8 -*-
"""
SpanEnricher - GreenLang semantic attribute enrichment for spans (OBS-003)

Defines the ``gl.*`` semantic convention namespace used across all GreenLang
services and provides a ``SpanEnricher`` helper class that injects standard
attributes into spans at creation time.

Semantic conventions follow the OpenTelemetry naming pattern but use the
``gl.`` prefix to avoid collisions with upstream conventions.

Example:
    >>> from greenlang.infrastructure.tracing_service.span_enrichment import (
    ...     SpanEnricher, GL_TENANT_ID,
    ... )
    >>> enricher = SpanEnricher(environment="prod")
    >>> enricher.enrich_agent_span(span, "carbon-calc", "a-123", tenant_id="t-corp")

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GreenLang Semantic Conventions (gl.* namespace)
# ---------------------------------------------------------------------------

# Identity & multi-tenancy
GL_TENANT_ID = "gl.tenant_id"
GL_REQUEST_ID = "gl.request_id"
GL_USER_ID = "gl.user_id"
GL_CORRELATION_ID = "gl.correlation_id"

# Agent attributes
GL_AGENT_TYPE = "gl.agent_type"
GL_AGENT_ID = "gl.agent_id"
GL_AGENT_KEY = "gl.agent_key"
GL_AGENT_VERSION = "gl.agent_version"
GL_AGENT_OPERATION = "gl.agent_operation"

# Pipeline attributes
GL_PIPELINE_ID = "gl.pipeline_id"
GL_PIPELINE_NAME = "gl.pipeline_name"
GL_PIPELINE_STAGE = "gl.pipeline_stage"
GL_PIPELINE_STEP = "gl.pipeline_step"

# Emission / climate attributes
GL_EMISSION_SCOPE = "gl.emission_scope"
GL_EMISSION_CATEGORY = "gl.emission_category"
GL_REGULATION = "gl.regulation"
GL_FRAMEWORK = "gl.framework"
GL_DATA_SOURCE = "gl.data_source"
GL_CALCULATION_TYPE = "gl.calculation_type"
GL_CALCULATION_METHOD = "gl.calculation_method"
GL_REPORTING_PERIOD = "gl.reporting_period"

# Environment
GL_ENVIRONMENT = "gl.environment"
GL_DEPLOYMENT_ID = "gl.deployment_id"

# Data quality
GL_DATA_QUALITY_SCORE = "gl.data_quality_score"
GL_PROVENANCE_HASH = "gl.provenance_hash"


# ---------------------------------------------------------------------------
# SpanEnricher
# ---------------------------------------------------------------------------

class SpanEnricher:
    """Enrich spans with GreenLang-specific semantic attributes.

    The enricher holds shared context (e.g. environment) and exposes
    domain-specific helper methods so call sites do not need to remember
    the attribute key strings.

    Attributes:
        environment: Current deployment environment.
    """

    def __init__(self, environment: str = "dev") -> None:
        """Initialise the enricher.

        Args:
            environment: Deployment environment name.
        """
        self.environment = environment

    # ---- Generic enrichment -------------------------------------------------

    def enrich_span(self, span: Any, attributes: Dict[str, Any]) -> None:
        """Add a dictionary of attributes to *span*.

        Automatically injects ``gl.environment``.  ``None`` values are
        silently skipped.

        Args:
            span: An OTel Span (or NoOpSpan).
            attributes: Key/value pairs to set on the span.
        """
        _safe_set(span, GL_ENVIRONMENT, self.environment)
        for key, value in attributes.items():
            if value is not None:
                _safe_set(span, key, _coerce(value))

    # ---- Agent enrichment ---------------------------------------------------

    def enrich_agent_span(
        self,
        span: Any,
        agent_type: str,
        agent_id: str,
        *,
        agent_key: str = "",
        agent_version: str = "",
        tenant_id: Optional[str] = None,
        operation: str = "",
    ) -> None:
        """Enrich *span* with agent execution attributes.

        Args:
            span: Span to enrich.
            agent_type: Agent class name or logical type.
            agent_id: Unique agent instance identifier.
            agent_key: Agent registry key (optional).
            agent_version: Agent SemVer version (optional).
            tenant_id: Tenant the agent is running for (optional).
            operation: Current operation (e.g. "execute", "validate").
        """
        _safe_set(span, GL_ENVIRONMENT, self.environment)
        _safe_set(span, GL_AGENT_TYPE, agent_type)
        _safe_set(span, GL_AGENT_ID, agent_id)
        if agent_key:
            _safe_set(span, GL_AGENT_KEY, agent_key)
        if agent_version:
            _safe_set(span, GL_AGENT_VERSION, agent_version)
        if tenant_id:
            _safe_set(span, GL_TENANT_ID, tenant_id)
        if operation:
            _safe_set(span, GL_AGENT_OPERATION, operation)

    # ---- Emission enrichment ------------------------------------------------

    def enrich_emission_span(
        self,
        span: Any,
        scope: str,
        regulation: str,
        data_source: str,
        *,
        category: str = "",
        calculation_type: str = "",
        calculation_method: str = "",
        framework: str = "",
        reporting_period: str = "",
    ) -> None:
        """Enrich *span* with emission-calculation attributes.

        Args:
            span: Span to enrich.
            scope: GHG scope (1, 2, 3).
            regulation: Applicable regulation (EUDR, CBAM, SB253, ...).
            data_source: Origin of activity data.
            category: Emission category within the scope.
            calculation_type: e.g. "spend-based", "activity-based".
            calculation_method: Specific method used (e.g. "GHG Protocol").
            framework: Reporting framework (GRI, SASB, TCFD, ...).
            reporting_period: ISO-8601 period string.
        """
        _safe_set(span, GL_ENVIRONMENT, self.environment)
        _safe_set(span, GL_EMISSION_SCOPE, scope)
        _safe_set(span, GL_REGULATION, regulation)
        _safe_set(span, GL_DATA_SOURCE, data_source)
        if category:
            _safe_set(span, GL_EMISSION_CATEGORY, category)
        if calculation_type:
            _safe_set(span, GL_CALCULATION_TYPE, calculation_type)
        if calculation_method:
            _safe_set(span, GL_CALCULATION_METHOD, calculation_method)
        if framework:
            _safe_set(span, GL_FRAMEWORK, framework)
        if reporting_period:
            _safe_set(span, GL_REPORTING_PERIOD, reporting_period)

    # ---- Pipeline enrichment ------------------------------------------------

    def enrich_pipeline_span(
        self,
        span: Any,
        pipeline_id: str,
        stage: str,
        *,
        pipeline_name: str = "",
        step: str = "",
    ) -> None:
        """Enrich *span* with pipeline execution attributes.

        Args:
            span: Span to enrich.
            pipeline_id: Unique pipeline run identifier.
            stage: Current pipeline stage name.
            pipeline_name: Human-readable pipeline name.
            step: Sub-step within the stage.
        """
        _safe_set(span, GL_ENVIRONMENT, self.environment)
        _safe_set(span, GL_PIPELINE_ID, pipeline_id)
        _safe_set(span, GL_PIPELINE_STAGE, stage)
        if pipeline_name:
            _safe_set(span, GL_PIPELINE_NAME, pipeline_name)
        if step:
            _safe_set(span, GL_PIPELINE_STEP, step)

    # ---- Data-quality enrichment --------------------------------------------

    def enrich_data_quality(
        self,
        span: Any,
        quality_score: float,
        provenance_hash: str = "",
    ) -> None:
        """Enrich *span* with data-quality metadata.

        Args:
            span: Span to enrich.
            quality_score: Normalised quality score (0.0-1.0).
            provenance_hash: SHA-256 provenance hash.
        """
        _safe_set(span, GL_DATA_QUALITY_SCORE, quality_score)
        if provenance_hash:
            _safe_set(span, GL_PROVENANCE_HASH, provenance_hash)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _safe_set(span: Any, key: str, value: Any) -> None:
    """Set an attribute on *span*, swallowing any error."""
    try:
        span.set_attribute(key, value)
    except Exception:
        pass


def _coerce(value: Any) -> Any:
    """Coerce a value to an OTel-compatible attribute type.

    OTel accepts str, bool, int, float, and sequences thereof.
    Everything else is stringified.
    """
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_coerce(v) for v in value]
    return str(value)


__all__ = [
    "SpanEnricher",
    # Semantic convention constants
    "GL_TENANT_ID",
    "GL_REQUEST_ID",
    "GL_USER_ID",
    "GL_CORRELATION_ID",
    "GL_AGENT_TYPE",
    "GL_AGENT_ID",
    "GL_AGENT_KEY",
    "GL_AGENT_VERSION",
    "GL_AGENT_OPERATION",
    "GL_PIPELINE_ID",
    "GL_PIPELINE_NAME",
    "GL_PIPELINE_STAGE",
    "GL_PIPELINE_STEP",
    "GL_EMISSION_SCOPE",
    "GL_EMISSION_CATEGORY",
    "GL_REGULATION",
    "GL_FRAMEWORK",
    "GL_DATA_SOURCE",
    "GL_CALCULATION_TYPE",
    "GL_CALCULATION_METHOD",
    "GL_REPORTING_PERIOD",
    "GL_ENVIRONMENT",
    "GL_DEPLOYMENT_ID",
    "GL_DATA_QUALITY_SCORE",
    "GL_PROVENANCE_HASH",
]
