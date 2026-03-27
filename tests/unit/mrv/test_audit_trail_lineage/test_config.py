# -*- coding: utf-8 -*-
"""
Test suite for audit_trail_lineage.config - AGENT-MRV-030.

Tests configuration management for the Audit Trail & Lineage Agent
(GL-MRV-X-042) including default values, environment variable loading,
singleton pattern, thread safety, validation, serialization, and
cross-validation logic.

Coverage:
- Default config values for GeneralConfig, DatabaseConfig, RedisConfig,
  AuditConfig, EvidenceConfig, ComplianceConfig
- GL_ATL_ environment variable loading (monkeypatch)
- Singleton pattern (get_config returns same instance)
- Thread safety (concurrent get_config calls)
- Validation (invalid values raise errors)
- to_dict / from_dict round-trip
- Frozen immutability on section dataclasses
- Master AuditTrailLineageConfig validate_all and cross-validation
- reset_config / set_config / validate_config / print_config

Target: ~40 tests, 85%+ coverage.

Author: GL-TestEngineer
Date: March 2026
"""

import os
import threading
from decimal import Decimal
from typing import Any, Dict, List

import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.audit_trail_lineage.config import (
        GeneralConfig,
        DatabaseConfig,
        RedisConfig,
        AuditConfig,
        EvidenceConfig,
        ComplianceConfig,
        AuditTrailLineageConfig,
        get_config,
        set_config,
        reset_config,
        validate_config,
        print_config,
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not CONFIG_AVAILABLE,
    reason="audit_trail_lineage.config not available",
)


# ==============================================================================
# GENERAL CONFIGURATION TESTS
# ==============================================================================


@_SKIP
class TestGeneralConfig:
    """Test GeneralConfig dataclass."""

    def test_default_enabled(self):
        """Test default enabled is True."""
        cfg = GeneralConfig()
        assert cfg.enabled is True

    def test_default_debug(self):
        """Test default debug is False."""
        cfg = GeneralConfig()
        assert cfg.debug is False

    def test_default_log_level(self):
        """Test default log_level is INFO."""
        cfg = GeneralConfig()
        assert cfg.log_level == "INFO"

    def test_default_agent_id(self):
        """Test default agent_id is GL-MRV-X-042."""
        cfg = GeneralConfig()
        assert cfg.agent_id == "GL-MRV-X-042"

    def test_default_agent_component(self):
        """Test default agent_component is AGENT-MRV-030."""
        cfg = GeneralConfig()
        assert cfg.agent_component == "AGENT-MRV-030"

    def test_default_version(self):
        """Test default version is 1.0.0."""
        cfg = GeneralConfig()
        assert cfg.version == "1.0.0"

    def test_default_api_prefix(self):
        """Test default api_prefix."""
        cfg = GeneralConfig()
        assert cfg.api_prefix == "/api/v1/audit-trail-lineage"

    def test_default_max_batch_size(self):
        """Test default max_batch_size is 10000."""
        cfg = GeneralConfig()
        assert cfg.max_batch_size == 10000

    def test_default_chain_hash_algorithm(self):
        """Test default chain_hash_algorithm is sha256."""
        cfg = GeneralConfig()
        assert cfg.chain_hash_algorithm == "sha256"

    def test_default_genesis_hash(self):
        """Test default genesis_hash."""
        cfg = GeneralConfig()
        assert cfg.genesis_hash == "greenlang-atl-genesis-v1"

    def test_default_max_chain_length(self):
        """Test default max_chain_length is 10000000."""
        cfg = GeneralConfig()
        assert cfg.max_chain_length == 10000000

    def test_default_enable_signatures(self):
        """Test default enable_signatures is True."""
        cfg = GeneralConfig()
        assert cfg.enable_signatures is True

    def test_frozen_immutability(self):
        """Test GeneralConfig is frozen (cannot modify attributes)."""
        cfg = GeneralConfig()
        with pytest.raises(AttributeError):
            cfg.agent_id = "modified"  # type: ignore[misc]

    def test_to_dict(self):
        """Test to_dict returns all fields."""
        cfg = GeneralConfig()
        d = cfg.to_dict()
        assert d["agent_id"] == "GL-MRV-X-042"
        assert d["enabled"] is True
        assert d["chain_hash_algorithm"] == "sha256"

    def test_from_dict_roundtrip(self):
        """Test from_dict recreates equivalent config."""
        cfg = GeneralConfig()
        d = cfg.to_dict()
        cfg2 = GeneralConfig.from_dict(d)
        assert cfg2.agent_id == cfg.agent_id
        assert cfg2.version == cfg.version

    def test_from_env_defaults(self):
        """Test from_env with no env vars returns defaults."""
        cfg = GeneralConfig.from_env()
        assert cfg.agent_id == "GL-MRV-X-042"
        assert cfg.enabled is True

    def test_from_env_custom_values(self, monkeypatch):
        """Test from_env reads GL_ATL_ environment variables."""
        monkeypatch.setenv("GL_ATL_ENABLED", "false")
        monkeypatch.setenv("GL_ATL_DEBUG", "true")
        monkeypatch.setenv("GL_ATL_LOG_LEVEL", "DEBUG")
        cfg = GeneralConfig.from_env()
        assert cfg.enabled is False
        assert cfg.debug is True
        assert cfg.log_level == "DEBUG"

    def test_validate_valid_config(self):
        """Test validate() passes for default config."""
        cfg = GeneralConfig()
        cfg.validate()  # Should not raise

    def test_validate_invalid_log_level(self):
        """Test validate() fails for invalid log_level."""
        cfg = GeneralConfig(log_level="INVALID")
        with pytest.raises(ValueError, match="log_level"):
            cfg.validate()

    def test_validate_empty_agent_id(self):
        """Test validate() fails for empty agent_id."""
        cfg = GeneralConfig(agent_id="")
        with pytest.raises(ValueError, match="agent_id"):
            cfg.validate()

    def test_validate_invalid_version(self):
        """Test validate() fails for non-semver version."""
        cfg = GeneralConfig(version="1.0")
        with pytest.raises(ValueError, match="version"):
            cfg.validate()

    def test_validate_invalid_api_prefix(self):
        """Test validate() fails for api_prefix not starting with /."""
        cfg = GeneralConfig(api_prefix="api/v1")
        with pytest.raises(ValueError, match="api_prefix"):
            cfg.validate()

    def test_validate_invalid_max_batch_size(self):
        """Test validate() fails for out-of-range max_batch_size."""
        cfg = GeneralConfig(max_batch_size=0)
        with pytest.raises(ValueError, match="max_batch_size"):
            cfg.validate()

    def test_validate_invalid_hash_algorithm(self):
        """Test validate() fails for invalid hash algorithm."""
        cfg = GeneralConfig(chain_hash_algorithm="md5")
        with pytest.raises(ValueError, match="chain_hash_algorithm"):
            cfg.validate()

    def test_validate_short_genesis_hash(self):
        """Test validate() fails for genesis_hash < 5 chars."""
        cfg = GeneralConfig(genesis_hash="abc")
        with pytest.raises(ValueError, match="genesis_hash"):
            cfg.validate()


# ==============================================================================
# DATABASE CONFIGURATION TESTS
# ==============================================================================


@_SKIP
class TestDatabaseConfig:
    """Test DatabaseConfig dataclass."""

    def test_default_values(self):
        """Test DatabaseConfig default values."""
        cfg = DatabaseConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 5432
        assert cfg.database == "greenlang"
        assert cfg.schema == "audit_trail_lineage_service"
        assert cfg.table_prefix == "gl_atl_"
        assert cfg.pool_min == 2
        assert cfg.pool_max == 10

    def test_frozen(self):
        """Test DatabaseConfig is frozen."""
        cfg = DatabaseConfig()
        with pytest.raises(AttributeError):
            cfg.host = "changed"  # type: ignore[misc]

    def test_get_connection_url(self):
        """Test get_connection_url builds proper URL."""
        cfg = DatabaseConfig(host="db.example.com", port=5433, password="secret")
        url = cfg.get_connection_url()
        assert "db.example.com" in url
        assert "5433" in url
        assert "secret" in url

    def test_get_connection_url_no_password(self):
        """Test get_connection_url without password."""
        cfg = DatabaseConfig()
        url = cfg.get_connection_url()
        assert "greenlang@localhost:5432/greenlang" in url

    def test_validate_valid(self):
        """Test validate passes for defaults."""
        cfg = DatabaseConfig()
        cfg.validate()

    def test_validate_invalid_port(self):
        """Test validate fails for port out of range."""
        cfg = DatabaseConfig(port=99999)
        with pytest.raises(ValueError, match="port"):
            cfg.validate()

    def test_validate_table_prefix_no_underscore(self):
        """Test validate fails if table_prefix does not end with _."""
        cfg = DatabaseConfig(table_prefix="gl_atl")
        with pytest.raises(ValueError, match="table_prefix"):
            cfg.validate()

    def test_validate_pool_min_exceeds_max(self):
        """Test validate fails if pool_min > pool_max."""
        cfg = DatabaseConfig(pool_min=20, pool_max=5)
        with pytest.raises(ValueError, match="pool_min"):
            cfg.validate()

    def test_validate_invalid_ssl_mode(self):
        """Test validate fails for invalid ssl_mode."""
        cfg = DatabaseConfig(ssl_mode="none")
        with pytest.raises(ValueError, match="ssl_mode"):
            cfg.validate()

    def test_from_env_custom(self, monkeypatch):
        """Test from_env reads GL_ATL_DB_ environment variables."""
        monkeypatch.setenv("GL_ATL_DB_HOST", "prod-db.example.com")
        monkeypatch.setenv("GL_ATL_DB_PORT", "5433")
        cfg = DatabaseConfig.from_env()
        assert cfg.host == "prod-db.example.com"
        assert cfg.port == 5433


# ==============================================================================
# REDIS CONFIGURATION TESTS
# ==============================================================================


@_SKIP
class TestRedisConfig:
    """Test RedisConfig dataclass."""

    def test_default_values(self):
        """Test RedisConfig default values."""
        cfg = RedisConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 6379
        assert cfg.db == 0
        assert cfg.prefix == "gl_atl:"
        assert cfg.ttl_seconds == 3600

    def test_frozen(self):
        """Test RedisConfig is frozen."""
        cfg = RedisConfig()
        with pytest.raises(AttributeError):
            cfg.host = "changed"  # type: ignore[misc]

    def test_get_connection_url_no_ssl(self):
        """Test get_connection_url without SSL."""
        cfg = RedisConfig()
        url = cfg.get_connection_url()
        assert url.startswith("redis://")

    def test_get_connection_url_with_ssl(self):
        """Test get_connection_url with SSL."""
        cfg = RedisConfig(ssl=True)
        url = cfg.get_connection_url()
        assert url.startswith("rediss://")

    def test_get_connection_url_with_password(self):
        """Test get_connection_url includes password."""
        cfg = RedisConfig(password="secret123")
        url = cfg.get_connection_url()
        assert "secret123" in url

    def test_validate_valid(self):
        """Test validate passes for defaults."""
        cfg = RedisConfig()
        cfg.validate()

    def test_validate_invalid_db(self):
        """Test validate fails for db > 15."""
        cfg = RedisConfig(db=16)
        with pytest.raises(ValueError, match="db"):
            cfg.validate()

    def test_validate_prefix_no_colon(self):
        """Test validate fails if prefix does not end with ':'."""
        cfg = RedisConfig(prefix="gl_atl")
        with pytest.raises(ValueError, match="prefix"):
            cfg.validate()

    def test_validate_ttl_too_large(self):
        """Test validate fails for ttl > 604800."""
        cfg = RedisConfig(ttl_seconds=700000)
        with pytest.raises(ValueError, match="ttl_seconds"):
            cfg.validate()


# ==============================================================================
# AUDIT CONFIGURATION TESTS
# ==============================================================================


@_SKIP
class TestAuditConfig:
    """Test AuditConfig dataclass."""

    def test_default_values(self):
        """Test AuditConfig default values."""
        cfg = AuditConfig()
        assert cfg.max_event_payload_bytes == 65536
        assert cfg.enable_chain_verification is True
        assert cfg.chain_verification_interval_events == 1000
        assert cfg.max_lineage_depth == 50
        assert cfg.materiality_threshold_pct == Decimal("5.0")

    def test_frozen(self):
        """Test AuditConfig is frozen."""
        cfg = AuditConfig()
        with pytest.raises(AttributeError):
            cfg.max_lineage_depth = 100  # type: ignore[misc]

    def test_validate_valid(self):
        """Test validate passes for defaults."""
        cfg = AuditConfig()
        cfg.validate()

    def test_validate_payload_too_small(self):
        """Test validate fails for payload < 1024."""
        cfg = AuditConfig(max_event_payload_bytes=512)
        with pytest.raises(ValueError, match="max_event_payload_bytes"):
            cfg.validate()

    def test_validate_lineage_depth_too_large(self):
        """Test validate fails for max_lineage_depth > 500."""
        cfg = AuditConfig(max_lineage_depth=501)
        with pytest.raises(ValueError, match="max_lineage_depth"):
            cfg.validate()

    def test_validate_materiality_over_100(self):
        """Test validate fails for materiality > 100."""
        cfg = AuditConfig(materiality_threshold_pct=Decimal("101"))
        with pytest.raises(ValueError, match="materiality_threshold_pct"):
            cfg.validate()

    def test_from_env(self, monkeypatch):
        """Test from_env reads GL_ATL_ audit env vars."""
        monkeypatch.setenv("GL_ATL_MAX_LINEAGE_DEPTH", "100")
        monkeypatch.setenv("GL_ATL_MATERIALITY_THRESHOLD_PCT", "10.0")
        cfg = AuditConfig.from_env()
        assert cfg.max_lineage_depth == 100
        assert cfg.materiality_threshold_pct == Decimal("10.0")

    def test_from_dict_decimal_coercion(self):
        """Test from_dict coerces materiality_threshold_pct to Decimal."""
        d = {"materiality_threshold_pct": "7.5"}
        cfg = AuditConfig.from_dict(d)
        assert cfg.materiality_threshold_pct == Decimal("7.5")


# ==============================================================================
# EVIDENCE CONFIGURATION TESTS
# ==============================================================================


@_SKIP
class TestEvidenceConfig:
    """Test EvidenceConfig dataclass."""

    def test_default_values(self):
        """Test EvidenceConfig default values."""
        cfg = EvidenceConfig()
        assert cfg.default_assurance_level == "limited"
        assert cfg.max_package_size_mb == 100
        assert cfg.enable_digital_signatures is True
        assert cfg.default_signature_algorithm == "ed25519"
        assert cfg.package_retention_years == 10
        assert cfg.enable_compression is True

    def test_frozen(self):
        """Test EvidenceConfig is frozen."""
        cfg = EvidenceConfig()
        with pytest.raises(AttributeError):
            cfg.default_assurance_level = "reasonable"  # type: ignore[misc]

    def test_validate_valid(self):
        """Test validate passes for defaults."""
        cfg = EvidenceConfig()
        cfg.validate()

    def test_validate_invalid_assurance_level(self):
        """Test validate fails for invalid assurance level."""
        cfg = EvidenceConfig(default_assurance_level="extreme")
        with pytest.raises(ValueError, match="default_assurance_level"):
            cfg.validate()

    def test_validate_invalid_signature_algorithm(self):
        """Test validate fails for invalid signature algorithm."""
        cfg = EvidenceConfig(default_signature_algorithm="md5")
        with pytest.raises(ValueError, match="default_signature_algorithm"):
            cfg.validate()

    def test_validate_retention_out_of_range(self):
        """Test validate fails for retention > 100 years."""
        cfg = EvidenceConfig(package_retention_years=101)
        with pytest.raises(ValueError, match="package_retention_years"):
            cfg.validate()


# ==============================================================================
# COMPLIANCE CONFIGURATION TESTS
# ==============================================================================


@_SKIP
class TestComplianceConfig:
    """Test ComplianceConfig dataclass."""

    def test_default_frameworks(self):
        """Test default supported frameworks contain 9 frameworks."""
        cfg = ComplianceConfig()
        assert len(cfg.supported_frameworks) == 9

    def test_default_frameworks_has_ghg(self):
        """Test GHG_PROTOCOL is in default frameworks."""
        cfg = ComplianceConfig()
        assert "GHG_PROTOCOL" in cfg.supported_frameworks

    def test_default_frameworks_has_iso(self):
        """Test ISO_14064 is in default frameworks."""
        cfg = ComplianceConfig()
        assert "ISO_14064" in cfg.supported_frameworks

    def test_default_frameworks_has_csrd(self):
        """Test CSRD_ESRS is in default frameworks."""
        cfg = ComplianceConfig()
        assert "CSRD_ESRS" in cfg.supported_frameworks

    def test_default_coverage_thresholds(self):
        """Test default coverage thresholds."""
        cfg = ComplianceConfig()
        assert cfg.coverage_warn_threshold == Decimal("0.80")
        assert cfg.coverage_fail_threshold == Decimal("0.50")

    def test_frozen(self):
        """Test ComplianceConfig is frozen."""
        cfg = ComplianceConfig()
        with pytest.raises(AttributeError):
            cfg.enable_gap_analysis = False  # type: ignore[misc]

    def test_validate_valid(self):
        """Test validate passes for defaults."""
        cfg = ComplianceConfig()
        cfg.validate()

    def test_validate_invalid_framework(self):
        """Test validate fails for unknown framework."""
        cfg = ComplianceConfig(supported_frameworks=("INVALID_FW",))
        with pytest.raises(ValueError, match="Invalid framework"):
            cfg.validate()

    def test_validate_duplicate_frameworks(self):
        """Test validate fails for duplicate frameworks."""
        cfg = ComplianceConfig(supported_frameworks=("GHG_PROTOCOL", "GHG_PROTOCOL"))
        with pytest.raises(ValueError, match="duplicate"):
            cfg.validate()

    def test_validate_fail_threshold_exceeds_warn(self):
        """Test validate fails when fail >= warn threshold."""
        cfg = ComplianceConfig(
            coverage_warn_threshold=Decimal("0.50"),
            coverage_fail_threshold=Decimal("0.60"),
        )
        with pytest.raises(ValueError, match="coverage_fail_threshold"):
            cfg.validate()

    def test_from_env(self, monkeypatch):
        """Test from_env reads GL_ATL_ compliance env vars."""
        monkeypatch.setenv("GL_ATL_SUPPORTED_FRAMEWORKS", "GHG_PROTOCOL,CDP")
        cfg = ComplianceConfig.from_env()
        assert len(cfg.supported_frameworks) == 2
        assert "GHG_PROTOCOL" in cfg.supported_frameworks
        assert "CDP" in cfg.supported_frameworks

    def test_from_dict_decimal_coercion(self):
        """Test from_dict coerces Decimal fields."""
        d = {
            "coverage_warn_threshold": "0.90",
            "coverage_fail_threshold": "0.60",
            "supported_frameworks": ["GHG_PROTOCOL"],
        }
        cfg = ComplianceConfig.from_dict(d)
        assert cfg.coverage_warn_threshold == Decimal("0.90")


# ==============================================================================
# MASTER CONFIG TESTS
# ==============================================================================


@_SKIP
class TestAuditTrailLineageConfig:
    """Test AuditTrailLineageConfig master configuration."""

    def test_default_construction(self):
        """Test master config can be constructed with all defaults."""
        cfg = AuditTrailLineageConfig()
        assert cfg.general.agent_id == "GL-MRV-X-042"
        assert cfg.database.table_prefix == "gl_atl_"
        assert cfg.redis.prefix == "gl_atl:"

    def test_validate_all_no_errors(self):
        """Test validate_all returns empty list for valid defaults."""
        cfg = AuditTrailLineageConfig()
        errors = cfg.validate_all()
        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_validate_raises_on_errors(self):
        """Test validate() raises ValueError if errors exist."""
        cfg = AuditTrailLineageConfig(
            general=GeneralConfig(log_level="INVALID"),
        )
        with pytest.raises(ValueError, match="validation failed"):
            cfg.validate()

    def test_validate_all_collects_multiple_errors(self):
        """Test validate_all collects errors from multiple sections."""
        cfg = AuditTrailLineageConfig(
            general=GeneralConfig(log_level="INVALID"),
            database=DatabaseConfig(port=99999),
        )
        errors = cfg.validate_all()
        assert len(errors) >= 2

    def test_cross_validation_api_prefix(self):
        """Test cross-validation warns about non-standard api_prefix."""
        cfg = AuditTrailLineageConfig(
            general=GeneralConfig(api_prefix="/v1/custom"),
        )
        errors = cfg.validate_all()
        cross_errors = [e for e in errors if "cross-validation" in e]
        assert len(cross_errors) >= 1

    def test_cross_validation_coverage_thresholds(self):
        """Test cross-validation catches fail >= warn thresholds."""
        cfg = AuditTrailLineageConfig(
            compliance=ComplianceConfig(
                coverage_warn_threshold=Decimal("0.50"),
                coverage_fail_threshold=Decimal("0.50"),
            ),
        )
        errors = cfg.validate_all()
        cross_errors = [e for e in errors if "coverage_fail_threshold" in e]
        assert len(cross_errors) >= 1

    def test_to_dict_all_sections(self):
        """Test to_dict contains all 6 configuration sections."""
        cfg = AuditTrailLineageConfig()
        d = cfg.to_dict()
        assert "general" in d
        assert "database" in d
        assert "redis" in d
        assert "audit" in d
        assert "evidence" in d
        assert "compliance" in d

    def test_from_dict_roundtrip(self):
        """Test from_dict recreates equivalent configuration."""
        cfg = AuditTrailLineageConfig()
        d = cfg.to_dict()
        cfg2 = AuditTrailLineageConfig.from_dict(d)
        assert cfg2.general.agent_id == cfg.general.agent_id
        assert cfg2.database.table_prefix == cfg.database.table_prefix

    def test_from_env(self):
        """Test from_env creates config from environment."""
        cfg = AuditTrailLineageConfig.from_env()
        assert cfg.general.agent_id == "GL-MRV-X-042"


# ==============================================================================
# SINGLETON PATTERN TESTS
# ==============================================================================


@_SKIP
class TestConfigSingleton:
    """Test thread-safe singleton pattern for get_config/set_config/reset_config."""

    def test_get_config_returns_instance(self):
        """Test get_config returns an AuditTrailLineageConfig instance."""
        cfg = get_config()
        assert isinstance(cfg, AuditTrailLineageConfig)

    def test_get_config_returns_same_instance(self):
        """Test get_config returns the same singleton instance."""
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_reset_config_clears_singleton(self):
        """Test reset_config clears the cached instance."""
        cfg1 = get_config()
        reset_config()
        cfg2 = get_config()
        # After reset, a new instance is created
        assert cfg2 is not cfg1

    def test_set_config_replaces_singleton(self):
        """Test set_config replaces the cached instance."""
        custom = AuditTrailLineageConfig(
            general=GeneralConfig(debug=True),
        )
        set_config(custom)
        cfg = get_config()
        assert cfg.general.debug is True

    def test_set_config_type_error(self):
        """Test set_config raises TypeError for non-config objects."""
        with pytest.raises(TypeError, match="AuditTrailLineageConfig"):
            set_config("not a config")  # type: ignore[arg-type]

    def test_validate_config_helper(self):
        """Test validate_config function returns errors list."""
        cfg = AuditTrailLineageConfig()
        errors = validate_config(cfg)
        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_thread_safety(self):
        """Test get_config is thread-safe under concurrent access."""
        results = []
        errors = []

        def _get():
            try:
                cfg = get_config()
                results.append(id(cfg))
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=_get) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All threads should get the same instance
        assert len(set(results)) == 1

    def test_print_config_no_error(self, capsys):
        """Test print_config runs without errors."""
        cfg = AuditTrailLineageConfig()
        print_config(cfg)
        captured = capsys.readouterr()
        assert "GL-MRV-X-042" in captured.out
        assert "AGENT-MRV-030" in captured.out

    def test_print_config_redacts_password(self, capsys):
        """Test print_config redacts password fields."""
        cfg = AuditTrailLineageConfig(
            database=DatabaseConfig(password="supersecret"),
        )
        print_config(cfg)
        captured = capsys.readouterr()
        assert "supersecret" not in captured.out
        assert "[REDACTED]" in captured.out
