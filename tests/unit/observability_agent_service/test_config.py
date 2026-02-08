# -*- coding: utf-8 -*-
"""
Unit Tests for ObservabilityAgentConfig (AGENT-FOUND-010)

Tests configuration creation, env var overrides with GL_OBSERVABILITY_AGENT_
prefix, type coercion (bool/int/float), and singleton get_config/set_config/
reset_config lifecycle.

Coverage target: 85%+ of config.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import os
import threading

import pytest

from greenlang.observability_agent.config import (
    ObservabilityAgentConfig,
    get_config,
    reset_config,
    set_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_env_and_singleton(monkeypatch):
    """Reset singleton and strip GL_OBSERVABILITY_AGENT_ env vars between tests."""
    reset_config()
    prefix = "GL_OBSERVABILITY_AGENT_"
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            monkeypatch.delenv(key, raising=False)
    yield
    reset_config()


# ---------------------------------------------------------------------------
# Default Values
# ---------------------------------------------------------------------------

class TestObservabilityAgentConfigDefaults:
    """Tests that all default values are correct."""

    def test_default_database_url(self):
        cfg = ObservabilityAgentConfig()
        assert cfg.database_url == "postgresql://localhost:5432/greenlang"

    def test_default_redis_url(self):
        cfg = ObservabilityAgentConfig()
        assert cfg.redis_url == "redis://localhost:6379/0"

    def test_default_prometheus_url(self):
        cfg = ObservabilityAgentConfig()
        assert cfg.prometheus_url == "http://localhost:9090"

    def test_default_grafana_url(self):
        cfg = ObservabilityAgentConfig()
        assert cfg.grafana_url == "http://localhost:3000"

    def test_default_tempo_url(self):
        cfg = ObservabilityAgentConfig()
        assert cfg.tempo_url == "http://localhost:3200"

    def test_default_loki_url(self):
        cfg = ObservabilityAgentConfig()
        assert cfg.loki_url == "http://localhost:3100"

    def test_default_alertmanager_url(self):
        cfg = ObservabilityAgentConfig()
        assert cfg.alertmanager_url == "http://localhost:9093"

    def test_default_metrics_retention_hours(self):
        cfg = ObservabilityAgentConfig()
        assert cfg.metrics_retention_hours == 24

    def test_default_max_active_spans(self):
        cfg = ObservabilityAgentConfig()
        assert cfg.max_active_spans == 10000

    def test_default_log_buffer_size(self):
        cfg = ObservabilityAgentConfig()
        assert cfg.log_buffer_size == 10000

    def test_default_alert_evaluation_interval(self):
        cfg = ObservabilityAgentConfig()
        assert cfg.alert_evaluation_interval_seconds == 60

    def test_default_health_check_interval(self):
        cfg = ObservabilityAgentConfig()
        assert cfg.health_check_interval_seconds == 30

    def test_default_slo_evaluation_interval(self):
        cfg = ObservabilityAgentConfig()
        assert cfg.slo_evaluation_interval_seconds == 300

    def test_default_enable_prometheus_export(self):
        cfg = ObservabilityAgentConfig()
        assert cfg.enable_prometheus_export is True

    def test_default_enable_opentelemetry(self):
        cfg = ObservabilityAgentConfig()
        assert cfg.enable_opentelemetry is True

    def test_default_enable_structured_logging(self):
        cfg = ObservabilityAgentConfig()
        assert cfg.enable_structured_logging is True

    def test_default_slo_target(self):
        cfg = ObservabilityAgentConfig()
        assert cfg.default_slo_target == pytest.approx(0.999)

    def test_default_burn_rate_windows(self):
        cfg = ObservabilityAgentConfig()
        assert cfg.burn_rate_short_window_minutes == 5
        assert cfg.burn_rate_long_window_minutes == 60

    def test_default_pool_sizes(self):
        cfg = ObservabilityAgentConfig()
        assert cfg.pool_min_size == 2
        assert cfg.pool_max_size == 10

    def test_default_log_level(self):
        cfg = ObservabilityAgentConfig()
        assert cfg.log_level == "INFO"


# ---------------------------------------------------------------------------
# Environment Variable Overrides
# ---------------------------------------------------------------------------

class TestObservabilityAgentConfigEnvOverrides:
    """Tests each env var override via from_env()."""

    def test_env_override_database_url(self, monkeypatch):
        monkeypatch.setenv("GL_OBSERVABILITY_AGENT_DATABASE_URL", "postgresql://prod:5432/obs")
        cfg = ObservabilityAgentConfig.from_env()
        assert cfg.database_url == "postgresql://prod:5432/obs"

    def test_env_override_redis_url(self, monkeypatch):
        monkeypatch.setenv("GL_OBSERVABILITY_AGENT_REDIS_URL", "redis://prod:6380/1")
        cfg = ObservabilityAgentConfig.from_env()
        assert cfg.redis_url == "redis://prod:6380/1"

    def test_env_override_prometheus_url(self, monkeypatch):
        monkeypatch.setenv("GL_OBSERVABILITY_AGENT_PROMETHEUS_URL", "http://prom:9090")
        cfg = ObservabilityAgentConfig.from_env()
        assert cfg.prometheus_url == "http://prom:9090"

    def test_env_override_grafana_url(self, monkeypatch):
        monkeypatch.setenv("GL_OBSERVABILITY_AGENT_GRAFANA_URL", "http://grafana:3000")
        cfg = ObservabilityAgentConfig.from_env()
        assert cfg.grafana_url == "http://grafana:3000"

    def test_env_override_tempo_url(self, monkeypatch):
        monkeypatch.setenv("GL_OBSERVABILITY_AGENT_TEMPO_URL", "http://tempo:3200")
        cfg = ObservabilityAgentConfig.from_env()
        assert cfg.tempo_url == "http://tempo:3200"

    def test_env_override_loki_url(self, monkeypatch):
        monkeypatch.setenv("GL_OBSERVABILITY_AGENT_LOKI_URL", "http://loki:3100")
        cfg = ObservabilityAgentConfig.from_env()
        assert cfg.loki_url == "http://loki:3100"

    def test_env_override_alertmanager_url(self, monkeypatch):
        monkeypatch.setenv("GL_OBSERVABILITY_AGENT_ALERTMANAGER_URL", "http://am:9093")
        cfg = ObservabilityAgentConfig.from_env()
        assert cfg.alertmanager_url == "http://am:9093"

    def test_env_override_log_level(self, monkeypatch):
        monkeypatch.setenv("GL_OBSERVABILITY_AGENT_LOG_LEVEL", "DEBUG")
        cfg = ObservabilityAgentConfig.from_env()
        assert cfg.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# Type Coercion
# ---------------------------------------------------------------------------

class TestObservabilityAgentConfigTypeParsing:
    """Tests boolean, integer, and float env var parsing."""

    @pytest.mark.parametrize("raw,expected", [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
    ])
    def test_boolean_env_parsing(self, monkeypatch, raw, expected):
        monkeypatch.setenv("GL_OBSERVABILITY_AGENT_ENABLE_PROMETHEUS_EXPORT", raw)
        cfg = ObservabilityAgentConfig.from_env()
        assert cfg.enable_prometheus_export is expected

    def test_integer_env_parsing(self, monkeypatch):
        monkeypatch.setenv("GL_OBSERVABILITY_AGENT_METRICS_RETENTION_HOURS", "72")
        cfg = ObservabilityAgentConfig.from_env()
        assert cfg.metrics_retention_hours == 72

    def test_integer_env_invalid_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("GL_OBSERVABILITY_AGENT_METRICS_RETENTION_HOURS", "abc")
        cfg = ObservabilityAgentConfig.from_env()
        assert cfg.metrics_retention_hours == 24  # default

    def test_float_env_parsing(self, monkeypatch):
        monkeypatch.setenv("GL_OBSERVABILITY_AGENT_DEFAULT_SLO_TARGET", "0.995")
        cfg = ObservabilityAgentConfig.from_env()
        assert cfg.default_slo_target == pytest.approx(0.995)

    def test_float_env_invalid_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("GL_OBSERVABILITY_AGENT_DEFAULT_SLO_TARGET", "not-a-float")
        cfg = ObservabilityAgentConfig.from_env()
        assert cfg.default_slo_target == pytest.approx(0.999)

    def test_pool_min_size_env(self, monkeypatch):
        monkeypatch.setenv("GL_OBSERVABILITY_AGENT_POOL_MIN_SIZE", "5")
        cfg = ObservabilityAgentConfig.from_env()
        assert cfg.pool_min_size == 5

    def test_pool_max_size_env(self, monkeypatch):
        monkeypatch.setenv("GL_OBSERVABILITY_AGENT_POOL_MAX_SIZE", "50")
        cfg = ObservabilityAgentConfig.from_env()
        assert cfg.pool_max_size == 50


# ---------------------------------------------------------------------------
# Singleton Lifecycle
# ---------------------------------------------------------------------------

class TestObservabilityAgentConfigSingleton:
    """Tests get_config / set_config / reset_config lifecycle."""

    def test_get_config_singleton_returns_same_instance(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_set_config_replaces_singleton(self):
        custom = ObservabilityAgentConfig(prometheus_url="http://custom:9090")
        set_config(custom)
        assert get_config() is custom
        assert get_config().prometheus_url == "http://custom:9090"

    def test_reset_config_clears_singleton(self):
        _ = get_config()
        reset_config()
        # After reset a new instance should be created on next call
        cfg = get_config()
        assert cfg is not None
        assert cfg.prometheus_url == "http://localhost:9090"

    def test_env_prefix_is_gl_observability_agent(self, monkeypatch):
        monkeypatch.setenv("GL_OBSERVABILITY_AGENT_LOG_LEVEL", "WARNING")
        cfg = ObservabilityAgentConfig.from_env()
        assert cfg.log_level == "WARNING"

    def test_config_is_dataclass(self):
        cfg = ObservabilityAgentConfig()
        assert hasattr(cfg, "__dataclass_fields__")
