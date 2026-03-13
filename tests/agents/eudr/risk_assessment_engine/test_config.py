# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-EUDR-028 RiskAssessmentEngineConfig.

Validates default values, environment variable overrides, weight sums,
threshold ordering, singleton behaviour, and all configuration sections.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.agents.eudr.risk_assessment_engine.config import (
    RiskAssessmentEngineConfig,
    get_config,
    reset_config,
)


# ---------------------------------------------------------------------------
# Default value tests
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    """Verify default configuration values for all sections."""

    def test_default_database_url(self, config: RiskAssessmentEngineConfig):
        assert "postgresql" in config.database_url

    def test_default_redis_url(self, config: RiskAssessmentEngineConfig):
        assert config.redis_url.startswith("redis://")

    def test_default_pool_size(self, config: RiskAssessmentEngineConfig):
        assert config.pool_size == 10

    def test_default_pool_timeout(self, config: RiskAssessmentEngineConfig):
        assert config.pool_timeout == 30

    def test_default_redis_ttl(self, config: RiskAssessmentEngineConfig):
        assert config.redis_ttl_seconds == 3600

    def test_default_log_level(self, config: RiskAssessmentEngineConfig):
        assert config.log_level == "INFO"


# ---------------------------------------------------------------------------
# Weight tests
# ---------------------------------------------------------------------------


class TestConfigWeights:
    """Verify risk dimension weights."""

    def test_risk_dimension_weights_sum_to_one(self, config: RiskAssessmentEngineConfig):
        """CRITICAL: All 8 weights must sum to exactly Decimal('1.00')."""
        total = (
            config.country_weight
            + config.commodity_weight
            + config.supplier_weight
            + config.deforestation_weight
            + config.corruption_weight
            + config.supply_chain_complexity_weight
            + config.mixing_risk_weight
            + config.circumvention_risk_weight
        )
        assert total == Decimal("1.00"), f"Weights sum to {total}, expected 1.00"

    def test_country_weight_default(self, config: RiskAssessmentEngineConfig):
        assert config.country_weight == Decimal("0.20")

    def test_commodity_weight_default(self, config: RiskAssessmentEngineConfig):
        assert config.commodity_weight == Decimal("0.15")

    def test_supplier_weight_default(self, config: RiskAssessmentEngineConfig):
        assert config.supplier_weight == Decimal("0.20")

    def test_deforestation_weight_default(self, config: RiskAssessmentEngineConfig):
        assert config.deforestation_weight == Decimal("0.20")

    def test_corruption_weight_default(self, config: RiskAssessmentEngineConfig):
        assert config.corruption_weight == Decimal("0.10")

    def test_supply_chain_complexity_weight_default(self, config: RiskAssessmentEngineConfig):
        assert config.supply_chain_complexity_weight == Decimal("0.05")

    def test_mixing_risk_weight_default(self, config: RiskAssessmentEngineConfig):
        assert config.mixing_risk_weight == Decimal("0.05")

    def test_circumvention_risk_weight_default(self, config: RiskAssessmentEngineConfig):
        assert config.circumvention_risk_weight == Decimal("0.05")


# ---------------------------------------------------------------------------
# Threshold tests
# ---------------------------------------------------------------------------


class TestConfigThresholds:
    """Verify risk classification thresholds."""

    def test_threshold_ordering(self, config: RiskAssessmentEngineConfig):
        """Thresholds must be in strictly ascending order."""
        assert (
            config.negligible_threshold
            < config.low_threshold
            < config.standard_threshold
            < config.high_threshold
            <= config.critical_threshold
        )

    def test_negligible_threshold_default(self, config: RiskAssessmentEngineConfig):
        assert config.negligible_threshold == Decimal("15")

    def test_low_threshold_default(self, config: RiskAssessmentEngineConfig):
        assert config.low_threshold == Decimal("30")

    def test_standard_threshold_default(self, config: RiskAssessmentEngineConfig):
        assert config.standard_threshold == Decimal("60")

    def test_high_threshold_default(self, config: RiskAssessmentEngineConfig):
        assert config.high_threshold == Decimal("80")

    def test_critical_threshold_default(self, config: RiskAssessmentEngineConfig):
        assert config.critical_threshold == Decimal("100")

    def test_hysteresis_buffer_value(self, config: RiskAssessmentEngineConfig):
        assert config.hysteresis_buffer == Decimal("3")


# ---------------------------------------------------------------------------
# Country benchmark multipliers
# ---------------------------------------------------------------------------


class TestConfigBenchmarkMultipliers:
    """Verify country benchmark multiplier defaults."""

    def test_benchmark_low_multiplier(self, config: RiskAssessmentEngineConfig):
        assert config.benchmark_low_multiplier == Decimal("0.70")

    def test_benchmark_standard_multiplier(self, config: RiskAssessmentEngineConfig):
        assert config.benchmark_standard_multiplier == Decimal("1.00")

    def test_benchmark_high_multiplier(self, config: RiskAssessmentEngineConfig):
        assert config.benchmark_high_multiplier == Decimal("1.50")


# ---------------------------------------------------------------------------
# Simplified DD settings
# ---------------------------------------------------------------------------


class TestConfigSimplifiedDD:
    """Verify simplified due diligence defaults."""

    def test_simplified_dd_enabled(self, config: RiskAssessmentEngineConfig):
        assert config.simplified_dd_enabled is True

    def test_simplified_dd_max_score(self, config: RiskAssessmentEngineConfig):
        assert config.simplified_dd_max_score == Decimal("30")

    def test_simplified_dd_require_all_low(self, config: RiskAssessmentEngineConfig):
        assert config.simplified_dd_require_all_low is True


# ---------------------------------------------------------------------------
# Upstream agent URLs
# ---------------------------------------------------------------------------


class TestConfigUpstreamURLs:
    """Verify upstream agent URL defaults."""

    def test_country_risk_url(self, config: RiskAssessmentEngineConfig):
        assert "eudr-country-risk" in config.country_risk_url

    def test_supplier_risk_url(self, config: RiskAssessmentEngineConfig):
        assert "eudr-supplier-risk" in config.supplier_risk_url

    def test_commodity_risk_url(self, config: RiskAssessmentEngineConfig):
        assert "eudr-commodity-risk" in config.commodity_risk_url

    def test_corruption_index_url(self, config: RiskAssessmentEngineConfig):
        assert "eudr-corruption-index" in config.corruption_index_url

    def test_deforestation_alert_url(self, config: RiskAssessmentEngineConfig):
        assert "eudr-deforestation-alert" in config.deforestation_alert_url


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------


class TestConfigEnvOverrides:
    """Verify environment variable overrides work correctly."""

    def test_env_override_database_url(self, monkeypatch):
        monkeypatch.setenv("GL_EUDR_RAE_DATABASE_URL", "postgresql://test:5432/test")
        reset_config()
        cfg = RiskAssessmentEngineConfig()
        assert cfg.database_url == "postgresql://test:5432/test"

    def test_env_override_redis_url(self, monkeypatch):
        monkeypatch.setenv("GL_EUDR_RAE_REDIS_URL", "redis://test:6379/1")
        reset_config()
        cfg = RiskAssessmentEngineConfig()
        assert cfg.redis_url == "redis://test:6379/1"

    def test_env_override_decimal_weights(self, monkeypatch):
        monkeypatch.setenv("GL_EUDR_RAE_COUNTRY_WEIGHT", "0.25")
        monkeypatch.setenv("GL_EUDR_RAE_COMMODITY_WEIGHT", "0.10")
        reset_config()
        cfg = RiskAssessmentEngineConfig()
        assert cfg.country_weight == Decimal("0.25")
        assert cfg.commodity_weight == Decimal("0.10")

    def test_env_override_hysteresis(self, monkeypatch):
        monkeypatch.setenv("GL_EUDR_RAE_HYSTERESIS_BUFFER", "5")
        reset_config()
        cfg = RiskAssessmentEngineConfig()
        assert cfg.hysteresis_buffer == Decimal("5")


# ---------------------------------------------------------------------------
# Rate limiting and circuit breaker
# ---------------------------------------------------------------------------


class TestConfigRateLimits:
    """Verify rate limiting tier defaults."""

    def test_rate_limit_anonymous(self, config: RiskAssessmentEngineConfig):
        assert config.rate_limit_anonymous == 10

    def test_rate_limit_basic(self, config: RiskAssessmentEngineConfig):
        assert config.rate_limit_basic == 50

    def test_rate_limit_standard(self, config: RiskAssessmentEngineConfig):
        assert config.rate_limit_standard == 200

    def test_rate_limit_premium(self, config: RiskAssessmentEngineConfig):
        assert config.rate_limit_premium == 1000

    def test_rate_limit_admin(self, config: RiskAssessmentEngineConfig):
        assert config.rate_limit_admin == 5000

    def test_circuit_breaker_failure_threshold(self, config: RiskAssessmentEngineConfig):
        assert config.circuit_breaker_failure_threshold == 5

    def test_circuit_breaker_reset_timeout(self, config: RiskAssessmentEngineConfig):
        assert config.circuit_breaker_reset_timeout == 60

    def test_circuit_breaker_half_open_max(self, config: RiskAssessmentEngineConfig):
        assert config.circuit_breaker_half_open_max == 3


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestConfigSingleton:
    """Verify singleton pattern for get_config / reset_config."""

    def test_singleton_get_config(self):
        reset_config()
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_reset_config(self):
        reset_config()
        cfg1 = get_config()
        reset_config()
        cfg2 = get_config()
        assert cfg1 is not cfg2
