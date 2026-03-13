# -*- coding: utf-8 -*-
"""
Unit tests for InformationGatheringConfig - AGENT-EUDR-027

Tests configuration defaults, environment variable overrides, singleton
behavior, frozen dataclass constraints, and element weight validation.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-027 (GL-EUDR-IGA-027)
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.agents.eudr.information_gathering.config import (
    CertificationBodyConfig,
    ExternalSourceConfig,
    InformationGatheringConfig,
    get_config,
    reset_config,
)


class TestInformationGatheringConfigDefaults:
    """Test default values of InformationGatheringConfig."""

    def test_default_database_url(self, config):
        assert config.database_url == "postgresql+asyncpg://gl:gl@localhost:5432/greenlang"

    def test_default_redis_url(self, config):
        assert config.redis_url == "redis://localhost:6379/0"

    def test_default_pool_size(self, config):
        assert config.pool_size == 10

    def test_default_pool_timeout(self, config):
        assert config.pool_timeout == 30

    def test_default_redis_ttl(self, config):
        assert config.redis_ttl_seconds == 86400

    def test_default_log_level(self, config):
        assert config.log_level == "INFO"

    def test_default_harvest_interval(self, config):
        assert config.harvest_interval_hours == 24

    def test_default_freshness_threshold(self, config):
        assert config.freshness_threshold_hours == 48

    def test_default_incremental_updates(self, config):
        assert config.incremental_updates_enabled is True

    def test_default_fuzzy_match_threshold(self, config):
        assert config.fuzzy_match_threshold == 0.85

    def test_default_dedup_enabled(self, config):
        assert config.dedup_enabled is True

    def test_default_provenance_algorithm(self, config):
        assert config.provenance_algorithm == "sha256"

    def test_default_metrics_prefix(self, config):
        assert config.metrics_prefix == "gl_eudr_iga_"

    def test_default_retention_days(self, config):
        assert config.retention_days == 1825

    def test_default_max_package_size_mb(self, config):
        assert config.max_package_size_mb == 500

    def test_default_circuit_breaker_failure_threshold(self, config):
        assert config.circuit_breaker_failure_threshold == 5

    def test_default_circuit_breaker_reset_timeout(self, config):
        assert config.circuit_breaker_reset_timeout == 60

    def test_default_insufficient_threshold(self, config):
        assert config.insufficient_threshold == Decimal("60")

    def test_default_partial_threshold(self, config):
        assert config.partial_threshold == Decimal("90")


class TestConfigEnvironmentOverrides:
    """Test environment variable overrides for configuration."""

    def test_env_override_database_url(self, monkeypatch):
        monkeypatch.setenv("GL_EUDR_IGA_DATABASE_URL", "postgresql+asyncpg://test:test@db:5432/test")
        cfg = InformationGatheringConfig()
        assert cfg.database_url == "postgresql+asyncpg://test:test@db:5432/test"

    def test_env_override_redis_url(self, monkeypatch):
        monkeypatch.setenv("GL_EUDR_IGA_REDIS_URL", "redis://redis:6379/1")
        cfg = InformationGatheringConfig()
        assert cfg.redis_url == "redis://redis:6379/1"

    def test_env_override_int_values(self, monkeypatch):
        monkeypatch.setenv("GL_EUDR_IGA_POOL_SIZE", "20")
        monkeypatch.setenv("GL_EUDR_IGA_POOL_TIMEOUT", "60")
        monkeypatch.setenv("GL_EUDR_IGA_REDIS_TTL", "3600")
        cfg = InformationGatheringConfig()
        assert cfg.pool_size == 20
        assert cfg.pool_timeout == 60
        assert cfg.redis_ttl_seconds == 3600

    def test_env_override_bool_values(self, monkeypatch):
        monkeypatch.setenv("GL_EUDR_IGA_DEDUP_ENABLED", "false")
        monkeypatch.setenv("GL_EUDR_IGA_INCREMENTAL_UPDATES", "false")
        monkeypatch.setenv("GL_EUDR_IGA_PROVENANCE_ENABLED", "false")
        cfg = InformationGatheringConfig()
        assert cfg.dedup_enabled is False
        assert cfg.incremental_updates_enabled is False
        assert cfg.provenance_enabled is False

    def test_env_override_decimal_values(self, monkeypatch):
        monkeypatch.setenv("GL_EUDR_IGA_INSUFFICIENT_THRESHOLD", "50")
        monkeypatch.setenv("GL_EUDR_IGA_PARTIAL_THRESHOLD", "85")
        cfg = InformationGatheringConfig()
        assert cfg.insufficient_threshold == Decimal("50")
        assert cfg.partial_threshold == Decimal("85")


class TestConfigExternalSourcesAndBodies:
    """Test external sources and certification bodies configuration."""

    def test_external_sources_default_count(self, config):
        assert len(config.external_sources) == 11

    def test_certification_bodies_default_count(self, config):
        assert len(config.certification_bodies) == 6

    def test_element_weights_sum_to_one(self, config):
        total = sum(config.element_weights.values())
        assert total == Decimal("1.00")

    def test_external_sources_contain_eu_traces(self, config):
        assert "eu_traces" in config.external_sources

    def test_certification_bodies_contain_fsc(self, config):
        assert "fsc" in config.certification_bodies

    def test_element_weights_has_10_elements(self, config):
        assert len(config.element_weights) == 10


class TestConfigSingleton:
    """Test singleton behavior of get_config and reset_config."""

    def test_singleton_get_config(self):
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_reset_config(self):
        config1 = get_config()
        reset_config()
        config2 = get_config()
        assert config1 is not config2


class TestFrozenDataclasses:
    """Test frozen dataclass behavior for ExternalSourceConfig and CertificationBodyConfig."""

    def test_external_source_config_frozen(self):
        src = ExternalSourceConfig(name="Test", base_url="https://example.com")
        with pytest.raises(AttributeError):
            src.name = "Changed"

    def test_certification_body_config_frozen(self):
        body = CertificationBodyConfig(name="Test", api_url="https://example.com")
        with pytest.raises(AttributeError):
            body.name = "Changed"

    def test_external_source_config_defaults(self):
        src = ExternalSourceConfig(name="Test", base_url="https://example.com")
        assert src.rate_limit_rps == 10
        assert src.timeout_seconds == 30
        assert src.retry_max == 3
        assert src.cache_ttl_hours == 24
        assert src.enabled is True

    def test_certification_body_config_defaults(self):
        body = CertificationBodyConfig(name="Test", api_url="https://example.com")
        assert body.verification_cache_ttl_hours == 24
        assert body.batch_size_max == 100
        assert body.enabled is True
