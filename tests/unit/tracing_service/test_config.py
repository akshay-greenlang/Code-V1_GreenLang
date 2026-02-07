# -*- coding: utf-8 -*-
"""
Unit tests for TracingConfig (OBS-003)

Tests configuration defaults, environment variable parsing, validation
bounds, and immutability guarantees.

Coverage target: 85%+ of config.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from greenlang.infrastructure.tracing_service.config import (
    TracingConfig,
    _env_bool,
    _env_float,
    _env_int,
)


# ============================================================================
# Helper function tests
# ============================================================================


class TestEnvHelpers:
    """Tests for the _env_bool, _env_float, _env_int helper functions."""

    def test_env_bool_true_values(self, monkeypatch):
        """Verify _env_bool recognises 'true', '1', 'yes' as True."""
        for val in ("true", "True", "TRUE", "1", "yes", "YES"):
            monkeypatch.setenv("TEST_BOOL", val)
            assert _env_bool("TEST_BOOL") is True

    def test_env_bool_false_values(self, monkeypatch):
        """Verify _env_bool returns False for other values."""
        for val in ("false", "0", "no", "anything", ""):
            monkeypatch.setenv("TEST_BOOL", val)
            assert _env_bool("TEST_BOOL") is False

    def test_env_bool_missing_uses_default(self, monkeypatch):
        """Verify _env_bool uses default when env var is absent."""
        monkeypatch.delenv("TEST_BOOL_MISSING", raising=False)
        assert _env_bool("TEST_BOOL_MISSING", "false") is False
        assert _env_bool("TEST_BOOL_MISSING", "true") is True

    def test_env_float_valid(self, monkeypatch):
        """Verify _env_float parses a valid float."""
        monkeypatch.setenv("TEST_FLOAT", "0.42")
        assert _env_float("TEST_FLOAT") == pytest.approx(0.42)

    def test_env_float_invalid_uses_default(self, monkeypatch):
        """Verify _env_float falls back to default on invalid input."""
        monkeypatch.setenv("TEST_FLOAT", "not_a_number")
        assert _env_float("TEST_FLOAT", "1.0") == pytest.approx(1.0)

    def test_env_int_valid(self, monkeypatch):
        """Verify _env_int parses a valid integer."""
        monkeypatch.setenv("TEST_INT", "42")
        assert _env_int("TEST_INT") == 42

    def test_env_int_invalid_uses_default(self, monkeypatch):
        """Verify _env_int falls back to default on invalid input."""
        monkeypatch.setenv("TEST_INT", "xyz")
        assert _env_int("TEST_INT", "10") == 10


# ============================================================================
# TracingConfig tests
# ============================================================================


class TestTracingConfig:
    """Test suite for TracingConfig dataclass."""

    def test_default_config_values(self, monkeypatch):
        """Test that default configuration has production-safe values."""
        # Clear all OTEL env vars to get true defaults
        for key in list(os.environ):
            if key.startswith("OTEL_") or key.startswith("GL_"):
                monkeypatch.delenv(key, raising=False)

        config = TracingConfig()

        assert config.service_name == "greenlang"
        assert config.service_version == "1.0.0"
        assert config.environment == "dev"
        assert config.otlp_endpoint == "http://otel-collector:4317"
        assert config.otlp_insecure is True
        assert config.otlp_timeout == 0  # _env_int default "0"
        assert config.sampling_rate == pytest.approx(1.0)
        assert config.always_sample_errors is True
        assert config.batch_max_queue_size == 2048
        assert config.batch_max_export_batch_size == 512
        assert config.batch_schedule_delay_ms == 5000
        assert config.batch_export_timeout_ms == 30000
        assert config.tenant_header == "X-Tenant-ID"
        assert config.propagate_baggage is True
        assert config.enrich_spans is True
        assert config.console_exporter is False
        assert config.enabled is True

    def test_config_from_env_vars(self, env_otel_vars):
        """Test that TracingConfig reads from environment variables."""
        config = TracingConfig()

        assert config.service_name == "env-service"
        assert config.service_version == "2.0.0"
        assert config.environment == "staging"
        assert config.otlp_endpoint == "http://otel:4317"
        assert config.otlp_insecure is False
        assert config.otlp_timeout == 30
        assert config.sampling_rate == pytest.approx(0.25)
        assert config.console_exporter is True
        assert config.enabled is True

    def test_otlp_endpoint_from_env(self, monkeypatch):
        """Test OTLP endpoint is read from OTEL_EXPORTER_OTLP_ENDPOINT."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://custom:4317")
        config = TracingConfig()
        assert config.otlp_endpoint == "http://custom:4317"

    def test_sampling_rate_from_env(self, monkeypatch):
        """Test sampling rate is read from OTEL_TRACES_SAMPLER_ARG."""
        monkeypatch.setenv("OTEL_TRACES_SAMPLER_ARG", "0.5")
        config = TracingConfig()
        assert config.sampling_rate == pytest.approx(0.5)

    def test_enabled_from_env(self, monkeypatch):
        """Test enabled flag is read from OTEL_TRACES_ENABLED."""
        monkeypatch.setenv("OTEL_TRACES_ENABLED", "false")
        config = TracingConfig()
        assert config.enabled is False

    def test_console_exporter_from_env(self, monkeypatch):
        """Test console exporter flag is read from OTEL_TRACES_CONSOLE."""
        monkeypatch.setenv("OTEL_TRACES_CONSOLE", "true")
        config = TracingConfig()
        assert config.console_exporter is True

    def test_instrument_flags_default_true(self):
        """Test all auto-instrumentation flags default to True."""
        config = TracingConfig()

        assert config.instrument_fastapi is True
        assert config.instrument_httpx is True
        assert config.instrument_psycopg is True
        assert config.instrument_redis is True
        assert config.instrument_celery is True
        assert config.instrument_requests is True

    def test_custom_config_override(self):
        """Test that explicit constructor values override defaults."""
        config = TracingConfig(
            service_name="my-agent",
            sampling_rate=0.1,
            instrument_redis=False,
            batch_max_queue_size=4096,
        )

        assert config.service_name == "my-agent"
        assert config.sampling_rate == pytest.approx(0.1)
        assert config.instrument_redis is False
        assert config.batch_max_queue_size == 4096

    def test_environment_detection(self, monkeypatch):
        """Test environment field reads GL_ENVIRONMENT."""
        monkeypatch.setenv("GL_ENVIRONMENT", "production")
        config = TracingConfig()
        assert config.environment == "production"

    def test_batch_processor_defaults(self):
        """Test batch span processor defaults are sensible."""
        config = TracingConfig()

        assert config.batch_max_queue_size == 2048
        assert config.batch_max_export_batch_size == 512
        assert config.batch_schedule_delay_ms == 5000
        assert config.batch_export_timeout_ms == 30000
        # export batch size <= queue size
        assert config.batch_max_export_batch_size <= config.batch_max_queue_size

    def test_tenant_header_default(self):
        """Test tenant header defaults to X-Tenant-ID."""
        config = TracingConfig()
        assert config.tenant_header == "X-Tenant-ID"

    def test_sampling_rate_bounds_clamped_high(self):
        """Test sampling rate >1.0 is clamped to 1.0."""
        config = TracingConfig(sampling_rate=5.0)
        assert config.sampling_rate == pytest.approx(1.0)

    def test_sampling_rate_bounds_clamped_low(self):
        """Test sampling rate <0.0 is clamped to 0.0."""
        config = TracingConfig(sampling_rate=-0.5)
        assert config.sampling_rate == pytest.approx(0.0)

    def test_sampling_rate_zero(self):
        """Test zero sampling rate is accepted."""
        config = TracingConfig(sampling_rate=0.0)
        assert config.sampling_rate == pytest.approx(0.0)

    def test_sampling_rate_one(self):
        """Test full sampling rate is accepted."""
        config = TracingConfig(sampling_rate=1.0)
        assert config.sampling_rate == pytest.approx(1.0)

    def test_disabled_config(self):
        """Test creating a fully disabled configuration."""
        config = TracingConfig(enabled=False)
        assert config.enabled is False

    def test_config_service_name_explicit(self):
        """Test service name can be explicitly set."""
        config = TracingConfig(service_name="cbam-agent")
        assert config.service_name == "cbam-agent"

    def test_from_env_factory(self, monkeypatch):
        """Test the from_env() classmethod factory."""
        monkeypatch.setenv("OTEL_SERVICE_NAME", "factory-service")
        config = TracingConfig.from_env()
        assert config.service_name == "factory-service"

    def test_batch_export_size_capped_by_queue(self):
        """Test batch export size is capped at queue size in __post_init__."""
        config = TracingConfig(
            batch_max_queue_size=100,
            batch_max_export_batch_size=500,
        )
        assert config.batch_max_export_batch_size <= config.batch_max_queue_size

    def test_empty_otlp_endpoint_gets_default(self):
        """Test empty OTLP endpoint is replaced with the default."""
        config = TracingConfig(otlp_endpoint="")
        assert config.otlp_endpoint == "http://otel-collector:4317"

    def test_service_overrides_default_empty(self):
        """Test service_overrides defaults to empty dict."""
        config = TracingConfig()
        assert config.service_overrides == {}

    def test_service_overrides_set(self):
        """Test service_overrides can be populated."""
        overrides = {"api-service": 0.5, "worker": 0.1}
        config = TracingConfig(service_overrides=overrides)
        assert config.service_overrides == overrides

    def test_otlp_headers_default_empty(self):
        """Test otlp_headers defaults to empty dict."""
        config = TracingConfig()
        assert config.otlp_headers == {}

    def test_otlp_headers_set(self):
        """Test otlp_headers can be populated."""
        headers = {"Authorization": "Bearer tok"}
        config = TracingConfig(otlp_headers=headers)
        assert config.otlp_headers == headers
