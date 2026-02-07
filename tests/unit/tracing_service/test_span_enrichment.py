# -*- coding: utf-8 -*-
"""
Unit tests for span enrichment with GreenLang-specific attributes (OBS-003)

Tests the SpanEnricher class and GL_* constants used to decorate spans
with agent, emission, pipeline, tenant, and regulatory metadata.

Coverage target: 85%+ of span_enrichment.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Helpers
# ============================================================================


def _import_enrichment():
    """Import the span_enrichment module, skipping if not yet created."""
    try:
        from greenlang.infrastructure.tracing_service import span_enrichment
        return span_enrichment
    except ImportError:
        pytest.skip("span_enrichment module not yet built")


# ============================================================================
# GL_* constant tests
# ============================================================================


class TestGLConstants:
    """Tests for GL_* span attribute constants."""

    def test_gl_constants_defined(self):
        """Verify all GL_* constants are defined and are strings."""
        enrich = _import_enrichment()
        required = [
            "GL_TENANT_ID",
            "GL_AGENT_NAME",
            "GL_AGENT_TYPE",
            "GL_AGENT_VERSION",
            "GL_PIPELINE_NAME",
            "GL_PIPELINE_STEP",
            "GL_EMISSION_SCOPE",
            "GL_EMISSION_CATEGORY",
            "GL_REGULATION",
            "GL_ENVIRONMENT",
        ]
        for name in required:
            assert hasattr(enrich, name), f"Missing constant: {name}"
            val = getattr(enrich, name)
            assert isinstance(val, str), f"{name} should be a string, got {type(val)}"

    def test_all_gl_constants_start_with_gl(self):
        """Verify every GL_* constant value starts with 'gl.'."""
        enrich = _import_enrichment()
        for attr_name in dir(enrich):
            if attr_name.startswith("GL_") and attr_name.isupper():
                val = getattr(enrich, attr_name)
                if isinstance(val, str):
                    assert val.startswith("gl."), (
                        f"{attr_name} = {val!r} does not start with 'gl.'"
                    )

    def test_gl_tenant_id_constant(self):
        """Verify GL_TENANT_ID equals 'gl.tenant.id'."""
        enrich = _import_enrichment()
        assert enrich.GL_TENANT_ID == "gl.tenant.id"

    def test_gl_agent_type_constant(self):
        """Verify GL_AGENT_TYPE equals 'gl.agent.type'."""
        enrich = _import_enrichment()
        assert enrich.GL_AGENT_TYPE == "gl.agent.type"

    def test_gl_emission_scope_constant(self):
        """Verify GL_EMISSION_SCOPE equals 'gl.emission.scope'."""
        enrich = _import_enrichment()
        assert enrich.GL_EMISSION_SCOPE == "gl.emission.scope"

    def test_gl_regulation_constant(self):
        """Verify GL_REGULATION equals 'gl.regulation'."""
        enrich = _import_enrichment()
        assert enrich.GL_REGULATION == "gl.regulation"


# ============================================================================
# SpanEnricher tests
# ============================================================================


class TestSpanEnricher:
    """Tests for the SpanEnricher class."""

    def test_span_enricher_initialization(self):
        """Verify SpanEnricher can be created with a config."""
        enrich = _import_enrichment()
        enricher = enrich.SpanEnricher(environment="test")
        assert enricher is not None

    def test_enrich_span_sets_environment(self, mock_span):
        """Verify enrich sets the environment attribute on the span."""
        enrich = _import_enrichment()
        enricher = enrich.SpanEnricher(environment="staging")
        enricher.enrich(mock_span)

        mock_span.set_attribute.assert_any_call(enrich.GL_ENVIRONMENT, "staging")

    def test_enrich_span_sets_attributes(self, mock_span):
        """Verify enrich sets additional attributes from kwargs."""
        enrich = _import_enrichment()
        enricher = enrich.SpanEnricher(environment="test")
        enricher.enrich(mock_span, **{enrich.GL_TENANT_ID: "t-acme"})

        mock_span.set_attribute.assert_any_call(enrich.GL_TENANT_ID, "t-acme")

    def test_enrich_span_skips_none_values(self, mock_span):
        """Verify enrich does not set attributes with None values."""
        enrich = _import_enrichment()
        enricher = enrich.SpanEnricher(environment="test")
        enricher.enrich(mock_span, **{enrich.GL_TENANT_ID: None})

        # Should NOT have called set_attribute with None value
        for call in mock_span.set_attribute.call_args_list:
            args, kwargs = call
            if len(args) >= 2:
                assert args[1] is not None, f"set_attribute called with None: {args}"

    def test_enrich_agent_span(self, mock_span):
        """Verify enrich_agent_span sets agent-specific attributes."""
        enrich = _import_enrichment()
        enricher = enrich.SpanEnricher(environment="test")
        enricher.enrich_agent_span(
            mock_span,
            agent_name="cbam-agent",
            agent_type="compliance",
            agent_version="1.2.0",
        )

        mock_span.set_attribute.assert_any_call(enrich.GL_AGENT_NAME, "cbam-agent")
        mock_span.set_attribute.assert_any_call(enrich.GL_AGENT_TYPE, "compliance")
        mock_span.set_attribute.assert_any_call(enrich.GL_AGENT_VERSION, "1.2.0")

    def test_enrich_agent_span_with_tenant(self, mock_span):
        """Verify enrich_agent_span includes tenant when provided."""
        enrich = _import_enrichment()
        enricher = enrich.SpanEnricher(environment="test")
        enricher.enrich_agent_span(
            mock_span,
            agent_name="ghg-calc",
            tenant_id="t-globex",
        )

        mock_span.set_attribute.assert_any_call(enrich.GL_TENANT_ID, "t-globex")

    def test_enrich_emission_span(self, mock_span):
        """Verify enrich_emission_span sets emission-specific attributes."""
        enrich = _import_enrichment()
        enricher = enrich.SpanEnricher(environment="test")
        enricher.enrich_emission_span(
            mock_span,
            scope="scope_1",
            category="stationary_combustion",
            regulation="cbam",
        )

        mock_span.set_attribute.assert_any_call(enrich.GL_EMISSION_SCOPE, "scope_1")
        mock_span.set_attribute.assert_any_call(
            enrich.GL_EMISSION_CATEGORY, "stationary_combustion"
        )
        mock_span.set_attribute.assert_any_call(enrich.GL_REGULATION, "cbam")

    def test_enrich_pipeline_span(self, mock_span):
        """Verify enrich_pipeline_span sets pipeline-specific attributes."""
        enrich = _import_enrichment()
        enricher = enrich.SpanEnricher(environment="test")
        enricher.enrich_pipeline_span(
            mock_span,
            pipeline_name="emission-pipeline",
            step="calculate",
        )

        mock_span.set_attribute.assert_any_call(
            enrich.GL_PIPELINE_NAME, "emission-pipeline"
        )
        mock_span.set_attribute.assert_any_call(enrich.GL_PIPELINE_STEP, "calculate")

    def test_enricher_with_mock_span(self, mock_span):
        """Verify enricher works correctly with a fully mocked span."""
        enrich = _import_enrichment()
        enricher = enrich.SpanEnricher(environment="production")

        enricher.enrich(mock_span)
        enricher.enrich_agent_span(mock_span, agent_name="test-agent")
        enricher.enrich_emission_span(mock_span, scope="scope_2")
        enricher.enrich_pipeline_span(mock_span, pipeline_name="pipe-1")

        # At minimum environment should be set
        assert mock_span.set_attribute.call_count >= 4
