# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-010 Scope 2 Market-Based Emissions Agent Configuration.

Tests Scope2MarketConfig singleton pattern, environment variable loading
(GL_S2M_* prefix), default values, validation rules, serialization,
connection URL builders, instrument hierarchy accessor, framework checks,
residual mix config, dual reporting, certificate config, and module-level
convenience functions.

Target: 80+ tests, 85%+ coverage of config.py.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import json
import os
import threading
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Conditional import guard
# ---------------------------------------------------------------------------

try:
    from greenlang.scope2_market.config import (
        Scope2MarketConfig,
        get_config,
        set_config,
        reset_config,
        validate_config,
        DEFAULT_ENABLED_FRAMEWORKS,
        DEFAULT_INSTRUMENT_HIERARCHY,
        VALID_GWP_SOURCES,
        VALID_ALLOCATION_METHODS,
        VALID_ROUNDING_MODES,
        VALID_SSL_MODES,
        VALID_RESIDUAL_SOURCES,
        VALID_INSTRUMENT_TYPES,
        VALID_TRACKING_SYSTEMS,
        VALID_FRAMEWORKS,
        ENV_PREFIX,
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

_SKIP = pytest.mark.skipif(not CONFIG_AVAILABLE, reason="config not available")


# ===========================================================================
# Singleton Pattern Tests
# ===========================================================================


@_SKIP
class TestSingleton:
    """Tests for the Scope2MarketConfig singleton pattern."""

    def test_singleton_returns_same_instance(self):
        """Two instantiations return the same object."""
        cfg1 = Scope2MarketConfig()
        cfg2 = Scope2MarketConfig()
        assert cfg1 is cfg2

    def test_reset_creates_new_instance(self):
        """After reset(), a new instance is created."""
        cfg1 = Scope2MarketConfig()
        Scope2MarketConfig.reset()
        cfg2 = Scope2MarketConfig()
        assert cfg1 is not cfg2

    def test_reset_clears_initialized_flag(self):
        """reset() clears the _initialized flag."""
        _ = Scope2MarketConfig()
        assert Scope2MarketConfig._initialized is True
        Scope2MarketConfig.reset()
        assert Scope2MarketConfig._initialized is False

    def test_reset_clears_instance(self):
        """reset() sets _instance to None."""
        _ = Scope2MarketConfig()
        Scope2MarketConfig.reset()
        assert Scope2MarketConfig._instance is None

    def test_thread_safe_construction(self):
        """Singleton is thread-safe under concurrent access."""
        instances = []

        def create_config():
            cfg = Scope2MarketConfig()
            instances.append(id(cfg))

        threads = [threading.Thread(target=create_config) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same instance
        assert len(set(instances)) == 1


# ===========================================================================
# Default Value Tests
# ===========================================================================


@_SKIP
class TestDefaultValues:
    """Tests for Scope2MarketConfig default values (no env vars)."""

    def test_enabled_default_true(self):
        """Default enabled is True."""
        cfg = Scope2MarketConfig()
        assert cfg.enabled is True

    def test_default_gwp_source(self):
        """Default GWP source is AR5."""
        cfg = Scope2MarketConfig()
        assert cfg.default_gwp_source == "AR5"

    def test_default_allocation_method(self):
        """Default allocation method is priority_based."""
        cfg = Scope2MarketConfig()
        assert cfg.default_allocation_method == "priority_based"

    def test_default_decimal_precision(self):
        """Default decimal precision is 10."""
        cfg = Scope2MarketConfig()
        assert cfg.decimal_precision == 10

    def test_default_rounding_mode(self):
        """Default rounding mode is ROUND_HALF_UP."""
        cfg = Scope2MarketConfig()
        assert cfg.rounding_mode == "ROUND_HALF_UP"

    def test_default_max_batch_size(self):
        """Default max batch size is 1000."""
        cfg = Scope2MarketConfig()
        assert cfg.max_batch_size == 1000

    def test_default_vintage_window(self):
        """Default vintage window is 1 year."""
        cfg = Scope2MarketConfig()
        assert cfg.default_vintage_window == 1

    def test_default_db_host(self):
        """Default database host is localhost."""
        cfg = Scope2MarketConfig()
        assert cfg.db_host == "localhost"

    def test_default_db_port(self):
        """Default database port is 5432."""
        cfg = Scope2MarketConfig()
        assert cfg.db_port == 5432

    def test_default_db_name(self):
        """Default database name is greenlang."""
        cfg = Scope2MarketConfig()
        assert cfg.db_name == "greenlang"

    def test_default_redis_host(self):
        """Default Redis host is localhost."""
        cfg = Scope2MarketConfig()
        assert cfg.redis_host == "localhost"

    def test_default_redis_port(self):
        """Default Redis port is 6379."""
        cfg = Scope2MarketConfig()
        assert cfg.redis_port == 6379

    def test_default_redis_ttl(self):
        """Default Redis TTL is 3600 seconds."""
        cfg = Scope2MarketConfig()
        assert cfg.redis_ttl == 3600

    def test_default_mc_iterations(self):
        """Default Monte Carlo iterations is 5000."""
        cfg = Scope2MarketConfig()
        assert cfg.default_mc_iterations == 5000

    def test_default_confidence_level(self):
        """Default confidence level is 0.95."""
        cfg = Scope2MarketConfig()
        assert cfg.default_confidence_level == 0.95

    def test_default_api_prefix(self):
        """Default API prefix is /api/v1/scope2-market."""
        cfg = Scope2MarketConfig()
        assert cfg.api_prefix == "/api/v1/scope2-market"

    def test_default_api_rate_limit(self):
        """Default API rate limit is 100."""
        cfg = Scope2MarketConfig()
        assert cfg.api_rate_limit == 100

    def test_default_log_level(self):
        """Default log level is INFO."""
        cfg = Scope2MarketConfig()
        assert cfg.log_level == "INFO"

    def test_default_enable_metrics(self):
        """Default enable_metrics is True."""
        cfg = Scope2MarketConfig()
        assert cfg.enable_metrics is True

    def test_default_metrics_prefix(self):
        """Default metrics prefix is gl_s2m_."""
        cfg = Scope2MarketConfig()
        assert cfg.metrics_prefix == "gl_s2m_"

    def test_default_service_name(self):
        """Default service name is scope2-market-service."""
        cfg = Scope2MarketConfig()
        assert cfg.service_name == "scope2-market-service"

    def test_default_genesis_hash(self):
        """Default genesis hash is the GL-MRV-X-010 anchor string."""
        cfg = Scope2MarketConfig()
        assert cfg.genesis_hash == "GL-MRV-X-010-SCOPE2-MARKET-GENESIS"

    def test_default_worker_threads(self):
        """Default worker threads is 4."""
        cfg = Scope2MarketConfig()
        assert cfg.worker_threads == 4

    def test_default_health_check_interval(self):
        """Default health check interval is 30 seconds."""
        cfg = Scope2MarketConfig()
        assert cfg.health_check_interval == 30

    def test_default_enabled_frameworks_count(self):
        """Default enabled frameworks has 7 entries."""
        cfg = Scope2MarketConfig()
        assert len(cfg.enabled_frameworks) == 7

    def test_default_residual_source(self):
        """Default residual mix source is aib."""
        cfg = Scope2MarketConfig()
        assert cfg.default_residual_source == "aib"

    def test_default_auto_retire_on_use(self):
        """Default auto_retire_on_use is True."""
        cfg = Scope2MarketConfig()
        assert cfg.auto_retire_on_use is True

    def test_default_require_tracking_system(self):
        """Default require_tracking_system is True."""
        cfg = Scope2MarketConfig()
        assert cfg.require_tracking_system is True

    def test_default_allow_unbundled_certs(self):
        """Default allow_unbundled_certs is True."""
        cfg = Scope2MarketConfig()
        assert cfg.allow_unbundled_certs is True

    def test_default_dual_reporting_required(self):
        """Default dual_reporting_required is True."""
        cfg = Scope2MarketConfig()
        assert cfg.dual_reporting_required is True

    def test_default_instrument_hierarchy(self):
        """Default instrument hierarchy has 11 entries with direct_line first."""
        cfg = Scope2MarketConfig()
        assert len(cfg.default_instrument_hierarchy) == 11
        assert cfg.default_instrument_hierarchy[0] == "direct_line"


# ===========================================================================
# Environment Variable Override Tests
# ===========================================================================


@_SKIP
class TestEnvVarOverrides:
    """Tests for GL_S2M_* environment variable overrides."""

    def test_env_default_gwp(self, monkeypatch):
        """GL_S2M_DEFAULT_GWP_SOURCE overrides default GWP source."""
        monkeypatch.setenv("GL_S2M_DEFAULT_GWP_SOURCE", "AR6")
        cfg = Scope2MarketConfig()
        assert cfg.default_gwp_source == "AR6"

    def test_env_db_host(self, monkeypatch):
        """GL_S2M_DB_HOST overrides database host."""
        monkeypatch.setenv("GL_S2M_DB_HOST", "db.example.com")
        cfg = Scope2MarketConfig()
        assert cfg.db_host == "db.example.com"

    def test_env_db_port(self, monkeypatch):
        """GL_S2M_DB_PORT overrides database port."""
        monkeypatch.setenv("GL_S2M_DB_PORT", "5433")
        cfg = Scope2MarketConfig()
        assert cfg.db_port == 5433

    def test_env_redis_host(self, monkeypatch):
        """GL_S2M_REDIS_HOST overrides Redis host."""
        monkeypatch.setenv("GL_S2M_REDIS_HOST", "redis.example.com")
        cfg = Scope2MarketConfig()
        assert cfg.redis_host == "redis.example.com"

    def test_env_decimal_precision(self, monkeypatch):
        """GL_S2M_DECIMAL_PRECISION overrides precision."""
        monkeypatch.setenv("GL_S2M_DECIMAL_PRECISION", "12")
        cfg = Scope2MarketConfig()
        assert cfg.decimal_precision == 12

    def test_env_max_batch_size(self, monkeypatch):
        """GL_S2M_MAX_BATCH_SIZE overrides batch size."""
        monkeypatch.setenv("GL_S2M_MAX_BATCH_SIZE", "5000")
        cfg = Scope2MarketConfig()
        assert cfg.max_batch_size == 5000

    def test_env_allocation_method(self, monkeypatch):
        """GL_S2M_DEFAULT_ALLOCATION_METHOD overrides allocation method."""
        monkeypatch.setenv("GL_S2M_DEFAULT_ALLOCATION_METHOD", "pro_rata")
        cfg = Scope2MarketConfig()
        assert cfg.default_allocation_method == "pro_rata"

    def test_env_vintage_window(self, monkeypatch):
        """GL_S2M_DEFAULT_VINTAGE_WINDOW overrides vintage window."""
        monkeypatch.setenv("GL_S2M_DEFAULT_VINTAGE_WINDOW", "3")
        cfg = Scope2MarketConfig()
        assert cfg.default_vintage_window == 3

    def test_env_bool_true_variants(self, monkeypatch):
        """Boolean env vars accept true/1/yes (case-insensitive)."""
        for truthy in ("true", "True", "TRUE", "1", "yes", "YES"):
            Scope2MarketConfig.reset()
            monkeypatch.setenv("GL_S2M_ENABLE_SUPPLIER_FACTORS", truthy)
            cfg = Scope2MarketConfig()
            assert cfg.enable_supplier_factors is True, f"'{truthy}' not parsed as True"

    def test_env_bool_false_variants(self, monkeypatch):
        """Non-truthy boolean values are treated as False."""
        for falsy in ("false", "0", "no", "anything"):
            Scope2MarketConfig.reset()
            monkeypatch.setenv("GL_S2M_ENABLE_SUPPLIER_FACTORS", falsy)
            cfg = Scope2MarketConfig()
            assert cfg.enable_supplier_factors is False, f"'{falsy}' not parsed as False"

    def test_env_invalid_int_uses_default(self, monkeypatch):
        """Invalid integer env var falls back to default."""
        monkeypatch.setenv("GL_S2M_DB_PORT", "not_a_number")
        cfg = Scope2MarketConfig()
        assert cfg.db_port == 5432  # default

    def test_env_invalid_float_uses_default(self, monkeypatch):
        """Invalid float env var falls back to default."""
        monkeypatch.setenv("GL_S2M_DEFAULT_CONFIDENCE_LEVEL", "bad")
        cfg = Scope2MarketConfig()
        assert cfg.default_confidence_level == 0.95  # default

    def test_env_list_parsing(self, monkeypatch):
        """Comma-separated list env var is parsed correctly."""
        monkeypatch.setenv("GL_S2M_CORS_ORIGINS", "http://a.com,http://b.com")
        cfg = Scope2MarketConfig()
        assert cfg.cors_origins == ["http://a.com", "http://b.com"]

    def test_env_list_empty_uses_default(self, monkeypatch):
        """Empty list env var falls back to default."""
        monkeypatch.setenv("GL_S2M_CORS_ORIGINS", "")
        cfg = Scope2MarketConfig()
        assert cfg.cors_origins == ["*"]

    def test_env_enabled_frameworks(self, monkeypatch):
        """GL_S2M_ENABLED_FRAMEWORKS overrides framework list."""
        monkeypatch.setenv("GL_S2M_ENABLED_FRAMEWORKS", "cdp,csrd_esrs")
        cfg = Scope2MarketConfig()
        assert cfg.enabled_frameworks == ["cdp", "csrd_esrs"]

    def test_env_enabled_false(self, monkeypatch):
        """GL_S2M_ENABLED=false disables the agent."""
        monkeypatch.setenv("GL_S2M_ENABLED", "false")
        cfg = Scope2MarketConfig()
        assert cfg.enabled is False

    def test_env_residual_source(self, monkeypatch):
        """GL_S2M_DEFAULT_RESIDUAL_SOURCE overrides residual source."""
        monkeypatch.setenv("GL_S2M_DEFAULT_RESIDUAL_SOURCE", "green_e")
        cfg = Scope2MarketConfig()
        assert cfg.default_residual_source == "green_e"


# ===========================================================================
# Validation Tests
# ===========================================================================


@_SKIP
class TestValidation:
    """Tests for the validate() method."""

    def test_default_config_is_valid(self):
        """Default configuration passes all validation checks."""
        cfg = Scope2MarketConfig()
        errors = cfg.validate()
        assert errors == []

    def test_invalid_gwp_source(self):
        """Invalid GWP source produces a validation error."""
        cfg = Scope2MarketConfig()
        cfg.default_gwp_source = "AR99"
        errors = cfg.validate()
        assert any("default_gwp_source" in e for e in errors)

    def test_invalid_allocation_method(self):
        """Invalid allocation method produces a validation error."""
        cfg = Scope2MarketConfig()
        cfg.default_allocation_method = "random_allocation"
        errors = cfg.validate()
        assert any("default_allocation_method" in e for e in errors)

    def test_negative_decimal_precision(self):
        """Negative decimal precision produces a validation error."""
        cfg = Scope2MarketConfig()
        cfg.decimal_precision = -1
        errors = cfg.validate()
        assert any("decimal_precision" in e for e in errors)

    def test_excessive_decimal_precision(self):
        """Decimal precision > 28 produces a validation error."""
        cfg = Scope2MarketConfig()
        cfg.decimal_precision = 30
        errors = cfg.validate()
        assert any("decimal_precision" in e for e in errors)

    def test_invalid_rounding_mode(self):
        """Invalid rounding mode produces a validation error."""
        cfg = Scope2MarketConfig()
        cfg.rounding_mode = "ROUND_INVALID"
        errors = cfg.validate()
        assert any("rounding_mode" in e for e in errors)

    def test_empty_db_host(self):
        """Empty db_host produces a validation error."""
        cfg = Scope2MarketConfig()
        cfg.db_host = ""
        errors = cfg.validate()
        assert any("db_host" in e for e in errors)

    def test_negative_db_port(self):
        """Negative db_port produces a validation error."""
        cfg = Scope2MarketConfig()
        cfg.db_port = -1
        errors = cfg.validate()
        assert any("db_port" in e for e in errors)

    def test_db_port_exceeds_max(self):
        """db_port > 65535 produces a validation error."""
        cfg = Scope2MarketConfig()
        cfg.db_port = 70000
        errors = cfg.validate()
        assert any("db_port" in e for e in errors)

    def test_pool_min_exceeds_max(self):
        """db_pool_min > db_pool_max produces a validation error."""
        cfg = Scope2MarketConfig()
        cfg.db_pool_min = 20
        cfg.db_pool_max = 5
        errors = cfg.validate()
        assert any("db_pool_min" in e for e in errors)

    def test_invalid_ssl_mode(self):
        """Invalid SSL mode produces a validation error."""
        cfg = Scope2MarketConfig()
        cfg.db_ssl_mode = "invalid_ssl"
        errors = cfg.validate()
        assert any("db_ssl_mode" in e for e in errors)

    def test_negative_redis_ttl(self):
        """Negative redis_ttl produces a validation error."""
        cfg = Scope2MarketConfig()
        cfg.redis_ttl = -100
        errors = cfg.validate()
        assert any("redis_ttl" in e for e in errors)

    def test_invalid_residual_source(self):
        """Invalid residual source produces a validation error."""
        cfg = Scope2MarketConfig()
        cfg.default_residual_source = "unknown_source"
        errors = cfg.validate()
        assert any("default_residual_source" in e for e in errors)

    def test_invalid_framework_in_list(self):
        """Unknown framework in enabled_frameworks produces an error."""
        cfg = Scope2MarketConfig()
        cfg.enabled_frameworks = ["ghg_protocol_scope2", "unknown_fw"]
        errors = cfg.validate()
        assert any("unknown_fw" in e for e in errors)

    def test_empty_genesis_hash_with_provenance_enabled(self):
        """Empty genesis_hash with provenance enabled is invalid."""
        cfg = Scope2MarketConfig()
        cfg.enable_provenance = True
        cfg.genesis_hash = ""
        errors = cfg.validate()
        assert any("genesis_hash" in e for e in errors)

    def test_zero_worker_threads(self):
        """Zero worker_threads produces a validation error."""
        cfg = Scope2MarketConfig()
        cfg.worker_threads = 0
        errors = cfg.validate()
        assert any("worker_threads" in e for e in errors)

    def test_invalid_log_level(self):
        """Invalid log level produces a validation error."""
        cfg = Scope2MarketConfig()
        cfg.log_level = "VERBOSE"
        errors = cfg.validate()
        assert any("log_level" in e for e in errors)

    def test_negative_mc_iterations(self):
        """Negative MC iterations produces a validation error."""
        cfg = Scope2MarketConfig()
        cfg.default_mc_iterations = -1
        errors = cfg.validate()
        assert any("default_mc_iterations" in e for e in errors)

    def test_confidence_level_out_of_range(self):
        """Confidence level >= 1.0 produces a validation error."""
        cfg = Scope2MarketConfig()
        cfg.default_confidence_level = 1.0
        errors = cfg.validate()
        assert any("default_confidence_level" in e for e in errors)

    def test_certificate_retirement_without_auto_retire(self):
        """enable_certificate_retirement without auto_retire_on_use is invalid."""
        cfg = Scope2MarketConfig()
        cfg.enable_certificate_retirement = True
        cfg.auto_retire_on_use = False
        errors = cfg.validate()
        assert any("enable_certificate_retirement" in e for e in errors)

    def test_re100_without_sbti_or_green_e(self):
        """RE100 without sbti or green_e produces a validation error."""
        cfg = Scope2MarketConfig()
        cfg.enabled_frameworks = ["re100", "ghg_protocol_scope2"]
        errors = cfg.validate()
        assert any("RE100" in e for e in errors)

    def test_sbti_without_ghg_protocol(self):
        """SBTi without ghg_protocol_scope2 produces a validation error."""
        cfg = Scope2MarketConfig()
        cfg.enabled_frameworks = ["sbti"]
        errors = cfg.validate()
        assert any("SBTi" in e for e in errors)

    def test_dual_reporting_without_ghg_protocol(self):
        """dual_reporting_required without ghg_protocol_scope2 is invalid."""
        cfg = Scope2MarketConfig()
        cfg.dual_reporting_required = True
        cfg.enabled_frameworks = ["cdp"]
        errors = cfg.validate()
        assert any("dual_reporting_required" in e for e in errors)

    def test_geographic_match_without_tracking_system(self):
        """validate_geographic_match without require_tracking_system is invalid."""
        cfg = Scope2MarketConfig()
        cfg.validate_geographic_match = True
        cfg.require_tracking_system = False
        errors = cfg.validate()
        assert any("validate_geographic_match" in e for e in errors)

    def test_residual_uncertainty_less_than_instrument(self):
        """residual_uncertainty_pct < instrument_uncertainty_pct is invalid."""
        cfg = Scope2MarketConfig()
        cfg.residual_uncertainty_pct = 0.01
        cfg.instrument_uncertainty_pct = 0.10
        errors = cfg.validate()
        assert any("residual_uncertainty_pct" in e for e in errors)


# ===========================================================================
# Serialization Tests
# ===========================================================================


@_SKIP
class TestSerialization:
    """Tests for to_dict(), from_dict(), to_json(), and from_json()."""

    def test_to_dict_returns_dict(self):
        """to_dict() returns a dictionary."""
        cfg = Scope2MarketConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_redacts_passwords(self):
        """to_dict() redacts db_password and redis_password."""
        cfg = Scope2MarketConfig()
        cfg.db_password = "secret123"
        cfg.redis_password = "redis_secret"
        d = cfg.to_dict()
        assert d["db_password"] == "***"
        assert d["redis_password"] == "***"

    def test_to_dict_empty_password_not_redacted(self):
        """to_dict() does not redact empty passwords."""
        cfg = Scope2MarketConfig()
        cfg.db_password = ""
        d = cfg.to_dict()
        assert d["db_password"] == ""

    def test_to_dict_contains_all_keys(self):
        """to_dict() contains all expected configuration keys."""
        cfg = Scope2MarketConfig()
        d = cfg.to_dict()
        expected_keys = [
            "enabled", "db_host", "db_port", "db_name", "db_user",
            "db_password", "db_pool_min", "db_pool_max", "db_ssl_mode",
            "redis_host", "redis_port", "redis_db", "redis_password",
            "redis_ttl", "default_gwp_source", "default_allocation_method",
            "decimal_precision", "rounding_mode", "max_batch_size",
            "default_vintage_window", "auto_retire_on_use",
            "require_tracking_system", "allow_unbundled_certs",
            "max_instruments_per_purchase", "validate_geographic_match",
            "default_instrument_hierarchy", "allowed_instrument_types",
            "allowed_tracking_systems", "default_residual_source",
            "residual_cache_ttl", "auto_update_factors",
            "residual_data_year", "fallback_to_location_based",
            "default_mc_iterations", "default_confidence_level",
            "residual_uncertainty_pct", "instrument_uncertainty_pct",
            "api_prefix", "api_rate_limit", "cors_origins", "enable_docs",
            "enabled_frameworks", "auto_compliance_check",
            "dual_reporting_required", "log_level", "enable_metrics",
            "metrics_prefix", "enable_tracing", "service_name",
            "enable_dual_reporting", "enable_certificate_retirement",
            "enable_supplier_factors", "enable_uncertainty",
            "enable_provenance", "genesis_hash", "enable_auth",
            "worker_threads", "enable_background_tasks",
            "health_check_interval",
        ]
        for key in expected_keys:
            assert key in d, f"Key '{key}' missing from to_dict()"

    def test_from_dict_applies_overrides(self):
        """from_dict() applies dictionary values to a new instance."""
        cfg = Scope2MarketConfig.from_dict({
            "default_gwp_source": "AR6",
            "decimal_precision": 12,
        })
        assert cfg.default_gwp_source == "AR6"
        assert cfg.decimal_precision == 12

    def test_from_dict_skips_redacted_password(self):
        """from_dict() skips redacted password '***' values."""
        cfg1 = Scope2MarketConfig()
        cfg1.db_password = "real_secret"
        d = cfg1.to_dict()  # db_password will be '***'
        cfg2 = Scope2MarketConfig.from_dict(d)
        # Password should NOT be overwritten with '***'
        assert cfg2.db_password != "***"

    def test_to_json_returns_valid_json(self):
        """to_json() returns a valid JSON string."""
        cfg = Scope2MarketConfig()
        json_str = cfg.to_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "default_gwp_source" in parsed

    def test_from_json_creates_config(self):
        """from_json() creates a valid configuration."""
        json_str = '{"default_gwp_source": "AR6", "decimal_precision": 15}'
        cfg = Scope2MarketConfig.from_json(json_str)
        assert cfg.default_gwp_source == "AR6"
        assert cfg.decimal_precision == 15

    def test_copy_includes_passwords(self):
        """copy() returns full values including passwords."""
        cfg = Scope2MarketConfig()
        cfg.db_password = "secret123"
        d = cfg.copy()
        assert d["db_password"] == "secret123"


# ===========================================================================
# Connection URL Builder Tests
# ===========================================================================


@_SKIP
class TestConnectionURLs:
    """Tests for get_db_url(), get_async_db_url(), and get_redis_url()."""

    def test_get_db_url_without_password(self):
        """get_db_url() builds correct URL without password."""
        cfg = Scope2MarketConfig()
        url = cfg.get_db_url()
        assert url.startswith("postgresql://")
        assert "greenlang@localhost:5432/greenlang" in url

    def test_get_db_url_with_password(self):
        """get_db_url() includes URL-encoded password."""
        cfg = Scope2MarketConfig()
        cfg.db_password = "p@ss w0rd!"
        url = cfg.get_db_url()
        assert "p%40ss+w0rd%21" in url or "p%40ss%20w0rd%21" in url

    def test_get_db_url_includes_ssl_mode(self):
        """get_db_url() appends sslmode query parameter."""
        cfg = Scope2MarketConfig()
        url = cfg.get_db_url()
        assert "sslmode=prefer" in url

    def test_get_async_db_url(self):
        """get_async_db_url() uses asyncpg scheme."""
        cfg = Scope2MarketConfig()
        url = cfg.get_async_db_url()
        assert url.startswith("postgresql+asyncpg://")

    def test_get_redis_url_without_password(self):
        """get_redis_url() builds correct URL without password."""
        cfg = Scope2MarketConfig()
        url = cfg.get_redis_url()
        assert url == "redis://localhost:6379/0"

    def test_get_redis_url_with_password(self):
        """get_redis_url() includes password in auth section."""
        cfg = Scope2MarketConfig()
        cfg.redis_password = "redis_pass"
        url = cfg.get_redis_url()
        assert ":redis_pass@" in url


# ===========================================================================
# Accessor Method Tests
# ===========================================================================


@_SKIP
class TestAccessorMethods:
    """Tests for instrument hierarchy, framework checks, and other accessors."""

    def test_get_instrument_hierarchy_returns_copy(self):
        """get_instrument_hierarchy() returns a copy, not a reference."""
        cfg = Scope2MarketConfig()
        h1 = cfg.get_instrument_hierarchy()
        h2 = cfg.get_instrument_hierarchy()
        assert h1 == h2
        assert h1 is not h2

    def test_get_instrument_hierarchy_content(self):
        """get_instrument_hierarchy() returns correct ordered list."""
        cfg = Scope2MarketConfig()
        h = cfg.get_instrument_hierarchy()
        assert h[0] == "direct_line"
        assert "rec" in h
        assert "ppa" in h

    def test_is_framework_enabled_true(self):
        """is_framework_enabled() returns True for enabled framework."""
        cfg = Scope2MarketConfig()
        assert cfg.is_framework_enabled("cdp") is True
        assert cfg.is_framework_enabled("CDP") is True  # case-insensitive

    def test_is_framework_enabled_false(self):
        """is_framework_enabled() returns False for unknown framework."""
        cfg = Scope2MarketConfig()
        assert cfg.is_framework_enabled("unknown_framework") is False

    def test_get_enabled_frameworks_returns_copy(self):
        """get_enabled_frameworks() returns a copy."""
        cfg = Scope2MarketConfig()
        fws = cfg.get_enabled_frameworks()
        assert isinstance(fws, list)
        assert len(fws) == 7

    def test_get_rounding_mode(self):
        """get_rounding_mode() maps string to decimal module constant."""
        cfg = Scope2MarketConfig()
        mode = cfg.get_rounding_mode()
        from decimal import ROUND_HALF_UP
        assert mode == ROUND_HALF_UP

    def test_get_rounding_mode_invalid_raises(self):
        """get_rounding_mode() raises ValueError for invalid mode."""
        cfg = Scope2MarketConfig()
        cfg.rounding_mode = "ROUND_INVALID"
        with pytest.raises(ValueError):
            cfg.get_rounding_mode()

    def test_get_residual_mix_config(self):
        """get_residual_mix_config() returns correct parameters."""
        cfg = Scope2MarketConfig()
        rmx = cfg.get_residual_mix_config()
        assert rmx["default_source"] == "aib"
        assert rmx["cache_ttl"] == 3600
        assert rmx["auto_update"] is False
        assert rmx["data_year"] == 2024

    def test_get_instrument_config(self):
        """get_instrument_config() returns correct parameters."""
        cfg = Scope2MarketConfig()
        inst = cfg.get_instrument_config()
        assert inst["auto_retire_on_use"] is True
        assert inst["require_tracking_system"] is True
        assert inst["allow_unbundled_certs"] is True
        assert inst["vintage_window"] == 1

    def test_get_uncertainty_params(self):
        """get_uncertainty_params() returns correct parameters."""
        cfg = Scope2MarketConfig()
        params = cfg.get_uncertainty_params()
        assert params["mc_iterations"] == 5000
        assert params["confidence_level"] == 0.95
        assert params["enabled"] is True

    def test_get_db_pool_params(self):
        """get_db_pool_params() returns pool configuration."""
        cfg = Scope2MarketConfig()
        params = cfg.get_db_pool_params()
        assert params["min_size"] == 2
        assert params["max_size"] == 10
        assert "conninfo" in params

    def test_get_api_config(self):
        """get_api_config() returns API configuration."""
        cfg = Scope2MarketConfig()
        api_cfg = cfg.get_api_config()
        assert api_cfg["prefix"] == "/api/v1/scope2-market"
        assert api_cfg["rate_limit"] == 100

    def test_get_observability_config(self):
        """get_observability_config() returns observability settings."""
        cfg = Scope2MarketConfig()
        obs_cfg = cfg.get_observability_config()
        assert obs_cfg["metrics_prefix"] == "gl_s2m_"
        assert obs_cfg["service_name"] == "scope2-market-service"

    def test_get_feature_flags(self):
        """get_feature_flags() returns all feature flags."""
        cfg = Scope2MarketConfig()
        flags = cfg.get_feature_flags()
        assert isinstance(flags, dict)
        assert "enable_dual_reporting" in flags
        assert "enable_uncertainty" in flags
        assert "enable_certificate_retirement" in flags

    def test_get_calculation_config(self):
        """get_calculation_config() returns calculation settings."""
        cfg = Scope2MarketConfig()
        calc = cfg.get_calculation_config()
        assert calc["gwp_source"] == "AR5"
        assert calc["allocation_method"] == "priority_based"
        assert calc["decimal_precision"] == 10
        assert calc["vintage_window"] == 1

    def test_get_allocation_config(self):
        """get_allocation_config() returns allocation settings."""
        cfg = Scope2MarketConfig()
        alloc = cfg.get_allocation_config()
        assert alloc["method"] == "priority_based"
        assert alloc["vintage_window"] == 1
        assert alloc["auto_retire_on_use"] is True

    def test_get_dual_reporting_config(self):
        """get_dual_reporting_config() returns dual reporting settings."""
        cfg = Scope2MarketConfig()
        dr = cfg.get_dual_reporting_config()
        assert dr["enabled"] is True
        assert dr["required"] is True

    def test_get_certificate_config(self):
        """get_certificate_config() returns certificate management settings."""
        cfg = Scope2MarketConfig()
        cert = cfg.get_certificate_config()
        assert cert["retirement_enabled"] is True
        assert cert["auto_retire_on_use"] is True
        assert cert["vintage_window"] == 1

    def test_health_summary(self):
        """health_summary() returns health-check data."""
        cfg = Scope2MarketConfig()
        summary = cfg.health_summary()
        assert summary["agent_id"] == "AGENT-MRV-010"
        assert summary["validation_status"] == "PASS"
        assert summary["gwp_source"] == "AR5"
        assert summary["allocation_method"] == "priority_based"

    def test_normalise(self):
        """normalise() converts values to canonical forms."""
        cfg = Scope2MarketConfig()
        cfg.default_gwp_source = "ar5"
        cfg.log_level = "debug"
        cfg.default_allocation_method = "PRIORITY_BASED"
        cfg.normalise()
        assert cfg.default_gwp_source == "AR5"
        assert cfg.log_level == "DEBUG"
        assert cfg.default_allocation_method == "priority_based"

    def test_merge(self):
        """merge() applies override values."""
        cfg = Scope2MarketConfig()
        cfg.merge({"decimal_precision": 14, "max_batch_size": 2000})
        assert cfg.decimal_precision == 14
        assert cfg.max_batch_size == 2000


# ===========================================================================
# String Representation Tests
# ===========================================================================


@_SKIP
class TestStringRepresentation:
    """Tests for __repr__ and __str__."""

    def test_repr_contains_class_name(self):
        """__repr__ contains the class name."""
        cfg = Scope2MarketConfig()
        assert "Scope2MarketConfig" in repr(cfg)

    def test_str_contains_key_settings(self):
        """__str__ contains key configuration summary."""
        cfg = Scope2MarketConfig()
        s = str(cfg)
        assert "gwp=" in s
        assert "allocation=" in s

    def test_eq_same_config(self):
        """Two configs with same settings are equal."""
        cfg1 = Scope2MarketConfig()
        # Create a second config via from_dict to compare
        d = cfg1.to_dict()
        cfg2 = Scope2MarketConfig.from_dict(d)
        assert cfg1 == cfg2

    def test_eq_different_type(self):
        """Config is not equal to a non-config object."""
        cfg = Scope2MarketConfig()
        assert cfg != "not a config"


# ===========================================================================
# Module-Level Convenience Function Tests
# ===========================================================================


@_SKIP
class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_config_returns_singleton(self):
        """get_config() returns the singleton instance."""
        cfg = get_config()
        assert isinstance(cfg, Scope2MarketConfig)
        assert cfg is Scope2MarketConfig()

    def test_set_config_with_overrides(self):
        """set_config() applies keyword overrides."""
        cfg = set_config(default_gwp_source="AR6")
        assert cfg.default_gwp_source == "AR6"

    def test_set_config_with_dict_overrides(self):
        """set_config() applies dictionary overrides."""
        cfg = set_config(overrides={"decimal_precision": 15})
        assert cfg.decimal_precision == 15

    def test_reset_config_clears_singleton(self):
        """reset_config() clears the singleton."""
        _ = get_config()
        reset_config()
        assert Scope2MarketConfig._instance is None

    def test_validate_config_returns_list(self):
        """validate_config() returns a list of errors."""
        errors = validate_config()
        assert isinstance(errors, list)
        assert len(errors) == 0  # defaults are valid


# ===========================================================================
# Module-Level Constants Tests
# ===========================================================================


@_SKIP
class TestModuleLevelConstants:
    """Tests for exported module-level constants."""

    def test_env_prefix(self):
        """ENV_PREFIX is GL_S2M_."""
        assert ENV_PREFIX == "GL_S2M_"

    def test_default_frameworks(self):
        """DEFAULT_ENABLED_FRAMEWORKS has 7 entries."""
        assert len(DEFAULT_ENABLED_FRAMEWORKS) == 7

    def test_default_instrument_hierarchy(self):
        """DEFAULT_INSTRUMENT_HIERARCHY has 11 entries."""
        assert len(DEFAULT_INSTRUMENT_HIERARCHY) == 11

    def test_valid_gwp_sources(self):
        """VALID_GWP_SOURCES is a frozenset with AR4, AR5, AR6."""
        assert isinstance(VALID_GWP_SOURCES, frozenset)
        assert "AR4" in VALID_GWP_SOURCES
        assert "AR5" in VALID_GWP_SOURCES
        assert "AR6" in VALID_GWP_SOURCES

    def test_valid_allocation_methods(self):
        """VALID_ALLOCATION_METHODS contains 5 methods."""
        assert isinstance(VALID_ALLOCATION_METHODS, frozenset)
        assert "priority_based" in VALID_ALLOCATION_METHODS
        assert "pro_rata" in VALID_ALLOCATION_METHODS

    def test_valid_rounding_modes(self):
        """VALID_ROUNDING_MODES contains standard decimal modes."""
        assert "ROUND_HALF_UP" in VALID_ROUNDING_MODES
        assert "ROUND_HALF_DOWN" in VALID_ROUNDING_MODES

    def test_valid_ssl_modes(self):
        """VALID_SSL_MODES contains prefer and require."""
        assert "prefer" in VALID_SSL_MODES
        assert "require" in VALID_SSL_MODES

    def test_valid_residual_sources(self):
        """VALID_RESIDUAL_SOURCES contains 6 sources."""
        assert isinstance(VALID_RESIDUAL_SOURCES, frozenset)
        assert "aib" in VALID_RESIDUAL_SOURCES
        assert "green_e" in VALID_RESIDUAL_SOURCES

    def test_valid_instrument_types(self):
        """VALID_INSTRUMENT_TYPES contains 12 types."""
        assert isinstance(VALID_INSTRUMENT_TYPES, frozenset)
        assert "rec" in VALID_INSTRUMENT_TYPES
        assert "ppa" in VALID_INSTRUMENT_TYPES

    def test_valid_tracking_systems(self):
        """VALID_TRACKING_SYSTEMS contains 10 systems."""
        assert isinstance(VALID_TRACKING_SYSTEMS, frozenset)
        assert "m_rets" in VALID_TRACKING_SYSTEMS
        assert "aib_eecs" in VALID_TRACKING_SYSTEMS

    def test_valid_frameworks(self):
        """VALID_FRAMEWORKS contains ghg_protocol_scope2 and cdp."""
        assert "ghg_protocol_scope2" in VALID_FRAMEWORKS
        assert "cdp" in VALID_FRAMEWORKS
        assert "re100" in VALID_FRAMEWORKS
