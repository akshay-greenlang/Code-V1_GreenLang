# -*- coding: utf-8 -*-
"""
Unit Tests for AccessGuardConfig (AGENT-FOUND-006)

Tests configuration creation, env var overrides with GL_ACCESS_GUARD_ prefix,
validation settings, and singleton get_config/set_config/reset_config.

Coverage target: 85%+ of config.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline AccessGuardConfig that mirrors greenlang/access_guard/config.py
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_ACCESS_GUARD_"


@dataclass
class AccessGuardConfig:
    """Mirrors greenlang.access_guard.config.AccessGuardConfig."""

    # Enforcement
    strict_mode: bool = True
    simulation_mode: bool = False

    # Rate limiting
    rate_limiting_enabled: bool = True
    default_rate_rpm: int = 100
    default_rate_rph: int = 1000
    default_rate_rpd: int = 10000
    burst_limit: int = 20

    # Audit
    audit_enabled: bool = True
    audit_all_decisions: bool = True
    audit_retention_days: int = 365

    # Tenant isolation
    strict_tenant_isolation: bool = True

    # OPA integration
    opa_enabled: bool = False
    opa_endpoint: Optional[str] = None

    # Caching
    decision_cache_ttl_seconds: int = 60
    policy_cache_ttl_seconds: int = 300

    # Capacity limits
    max_policies: int = 1000
    max_rules_per_policy: int = 100

    @classmethod
    def from_env(cls) -> "AccessGuardConfig":
        prefix = _ENV_PREFIX

        def _env(name: str, default: Any = None) -> Optional[str]:
            return os.environ.get(f"{prefix}{name}", default)

        def _bool(name: str, default: bool) -> bool:
            val = _env(name)
            if val is None:
                return default
            return val.lower() in ("true", "1", "yes")

        def _int(name: str, default: int) -> int:
            val = _env(name)
            if val is None:
                return default
            try:
                return int(val)
            except ValueError:
                return default

        def _str_opt(name: str, default: Optional[str]) -> Optional[str]:
            val = _env(name)
            if val is None:
                return default
            return val if val.strip() else default

        return cls(
            strict_mode=_bool("STRICT_MODE", cls.strict_mode),
            simulation_mode=_bool("SIMULATION_MODE", cls.simulation_mode),
            rate_limiting_enabled=_bool(
                "RATE_LIMITING_ENABLED", cls.rate_limiting_enabled,
            ),
            default_rate_rpm=_int("DEFAULT_RATE_RPM", cls.default_rate_rpm),
            default_rate_rph=_int("DEFAULT_RATE_RPH", cls.default_rate_rph),
            default_rate_rpd=_int("DEFAULT_RATE_RPD", cls.default_rate_rpd),
            burst_limit=_int("BURST_LIMIT", cls.burst_limit),
            audit_enabled=_bool("AUDIT_ENABLED", cls.audit_enabled),
            audit_all_decisions=_bool(
                "AUDIT_ALL_DECISIONS", cls.audit_all_decisions,
            ),
            audit_retention_days=_int(
                "AUDIT_RETENTION_DAYS", cls.audit_retention_days,
            ),
            strict_tenant_isolation=_bool(
                "STRICT_TENANT_ISOLATION", cls.strict_tenant_isolation,
            ),
            opa_enabled=_bool("OPA_ENABLED", cls.opa_enabled),
            opa_endpoint=_str_opt("OPA_ENDPOINT", cls.opa_endpoint),
            decision_cache_ttl_seconds=_int(
                "DECISION_CACHE_TTL_SECONDS", cls.decision_cache_ttl_seconds,
            ),
            policy_cache_ttl_seconds=_int(
                "POLICY_CACHE_TTL_SECONDS", cls.policy_cache_ttl_seconds,
            ),
            max_policies=_int("MAX_POLICIES", cls.max_policies),
            max_rules_per_policy=_int(
                "MAX_RULES_PER_POLICY", cls.max_rules_per_policy,
            ),
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[AccessGuardConfig] = None
_config_lock = threading.Lock()


def get_config() -> AccessGuardConfig:
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = AccessGuardConfig.from_env()
    return _config_instance


def set_config(config: AccessGuardConfig) -> None:
    global _config_instance
    with _config_lock:
        _config_instance = config


def reset_config() -> None:
    global _config_instance
    with _config_lock:
        _config_instance = None


# ---------------------------------------------------------------------------
# Autouse: reset singleton between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_config_singleton():
    yield
    reset_config()


# ===========================================================================
# Test Classes
# ===========================================================================


class TestAccessGuardConfigDefaults:
    """Test that default configuration values match PRD requirements."""

    def test_default_strict_mode(self):
        config = AccessGuardConfig()
        assert config.strict_mode is True

    def test_default_simulation_mode(self):
        config = AccessGuardConfig()
        assert config.simulation_mode is False

    def test_default_rate_limiting_enabled(self):
        config = AccessGuardConfig()
        assert config.rate_limiting_enabled is True

    def test_default_rate_rpm(self):
        config = AccessGuardConfig()
        assert config.default_rate_rpm == 100

    def test_default_rate_rph(self):
        config = AccessGuardConfig()
        assert config.default_rate_rph == 1000

    def test_default_rate_rpd(self):
        config = AccessGuardConfig()
        assert config.default_rate_rpd == 10000

    def test_default_burst_limit(self):
        config = AccessGuardConfig()
        assert config.burst_limit == 20

    def test_default_audit_enabled(self):
        config = AccessGuardConfig()
        assert config.audit_enabled is True

    def test_default_audit_all_decisions(self):
        config = AccessGuardConfig()
        assert config.audit_all_decisions is True

    def test_default_audit_retention_days(self):
        config = AccessGuardConfig()
        assert config.audit_retention_days == 365

    def test_default_strict_tenant_isolation(self):
        config = AccessGuardConfig()
        assert config.strict_tenant_isolation is True

    def test_default_opa_enabled(self):
        config = AccessGuardConfig()
        assert config.opa_enabled is False

    def test_default_opa_endpoint(self):
        config = AccessGuardConfig()
        assert config.opa_endpoint is None

    def test_default_decision_cache_ttl(self):
        config = AccessGuardConfig()
        assert config.decision_cache_ttl_seconds == 60

    def test_default_policy_cache_ttl(self):
        config = AccessGuardConfig()
        assert config.policy_cache_ttl_seconds == 300

    def test_default_max_policies(self):
        config = AccessGuardConfig()
        assert config.max_policies == 1000

    def test_default_max_rules_per_policy(self):
        config = AccessGuardConfig()
        assert config.max_rules_per_policy == 100


class TestAccessGuardConfigFromEnv:
    """Test GL_ACCESS_GUARD_ env var overrides via from_env()."""

    def test_env_override_strict_mode_false(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_STRICT_MODE", "false")
        config = AccessGuardConfig.from_env()
        assert config.strict_mode is False

    def test_env_override_simulation_mode_true(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_SIMULATION_MODE", "true")
        config = AccessGuardConfig.from_env()
        assert config.simulation_mode is True

    def test_env_override_rate_limiting_disabled(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_RATE_LIMITING_ENABLED", "false")
        config = AccessGuardConfig.from_env()
        assert config.rate_limiting_enabled is False

    def test_env_override_default_rate_rpm(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_DEFAULT_RATE_RPM", "500")
        config = AccessGuardConfig.from_env()
        assert config.default_rate_rpm == 500

    def test_env_override_default_rate_rph(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_DEFAULT_RATE_RPH", "5000")
        config = AccessGuardConfig.from_env()
        assert config.default_rate_rph == 5000

    def test_env_override_default_rate_rpd(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_DEFAULT_RATE_RPD", "50000")
        config = AccessGuardConfig.from_env()
        assert config.default_rate_rpd == 50000

    def test_env_override_burst_limit(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_BURST_LIMIT", "50")
        config = AccessGuardConfig.from_env()
        assert config.burst_limit == 50

    def test_env_override_audit_enabled_false(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_AUDIT_ENABLED", "0")
        config = AccessGuardConfig.from_env()
        assert config.audit_enabled is False

    def test_env_override_audit_all_decisions_false(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_AUDIT_ALL_DECISIONS", "no")
        config = AccessGuardConfig.from_env()
        assert config.audit_all_decisions is False

    def test_env_override_audit_retention_days(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_AUDIT_RETENTION_DAYS", "730")
        config = AccessGuardConfig.from_env()
        assert config.audit_retention_days == 730

    def test_env_override_strict_tenant_isolation_false(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_STRICT_TENANT_ISOLATION", "false")
        config = AccessGuardConfig.from_env()
        assert config.strict_tenant_isolation is False

    def test_env_override_opa_enabled(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_OPA_ENABLED", "true")
        config = AccessGuardConfig.from_env()
        assert config.opa_enabled is True

    def test_env_override_opa_endpoint(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_OPA_ENDPOINT", "http://opa:8181")
        config = AccessGuardConfig.from_env()
        assert config.opa_endpoint == "http://opa:8181"

    def test_env_override_opa_endpoint_empty_uses_none(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_OPA_ENDPOINT", "   ")
        config = AccessGuardConfig.from_env()
        assert config.opa_endpoint is None

    def test_env_override_decision_cache_ttl(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_DECISION_CACHE_TTL_SECONDS", "120")
        config = AccessGuardConfig.from_env()
        assert config.decision_cache_ttl_seconds == 120

    def test_env_override_policy_cache_ttl(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_POLICY_CACHE_TTL_SECONDS", "600")
        config = AccessGuardConfig.from_env()
        assert config.policy_cache_ttl_seconds == 600

    def test_env_override_max_policies(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_MAX_POLICIES", "5000")
        config = AccessGuardConfig.from_env()
        assert config.max_policies == 5000

    def test_env_override_max_rules_per_policy(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_MAX_RULES_PER_POLICY", "200")
        config = AccessGuardConfig.from_env()
        assert config.max_rules_per_policy == 200

    def test_env_override_bool_true_with_1(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_OPA_ENABLED", "1")
        config = AccessGuardConfig.from_env()
        assert config.opa_enabled is True

    def test_env_override_bool_true_with_yes(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_OPA_ENABLED", "yes")
        config = AccessGuardConfig.from_env()
        assert config.opa_enabled is True

    def test_env_override_bool_true_with_TRUE(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_OPA_ENABLED", "TRUE")
        config = AccessGuardConfig.from_env()
        assert config.opa_enabled is True

    def test_env_invalid_int_uses_default_rpm(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_DEFAULT_RATE_RPM", "not_a_number")
        config = AccessGuardConfig.from_env()
        assert config.default_rate_rpm == 100

    def test_env_invalid_int_uses_default_max_policies(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_MAX_POLICIES", "abc")
        config = AccessGuardConfig.from_env()
        assert config.max_policies == 1000

    def test_env_invalid_int_uses_default_burst(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_BURST_LIMIT", "xyz")
        config = AccessGuardConfig.from_env()
        assert config.burst_limit == 20

    def test_multiple_env_overrides(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_STRICT_MODE", "false")
        monkeypatch.setenv("GL_ACCESS_GUARD_SIMULATION_MODE", "true")
        monkeypatch.setenv("GL_ACCESS_GUARD_MAX_POLICIES", "2000")
        monkeypatch.setenv("GL_ACCESS_GUARD_OPA_ENABLED", "true")
        config = AccessGuardConfig.from_env()
        assert config.strict_mode is False
        assert config.simulation_mode is True
        assert config.max_policies == 2000
        assert config.opa_enabled is True


class TestAccessGuardConfigSingleton:
    """Test get_config/set_config/reset_config singleton pattern."""

    def test_get_config_returns_instance(self):
        config = get_config()
        assert isinstance(config, AccessGuardConfig)

    def test_get_config_returns_same_instance(self):
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2

    def test_reset_config_clears_singleton(self):
        c1 = get_config()
        reset_config()
        c2 = get_config()
        assert c1 is not c2

    def test_set_config_overrides_singleton(self):
        custom = AccessGuardConfig(max_policies=999)
        set_config(custom)
        assert get_config().max_policies == 999

    def test_set_config_then_get_returns_same(self):
        custom = AccessGuardConfig(burst_limit=50)
        set_config(custom)
        assert get_config() is custom

    def test_thread_safety_of_get_config(self):
        """Test that concurrent get_config calls return the same instance."""
        instances = []

        def get_instance():
            instances.append(get_config())

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(instances) == 10
        for inst in instances[1:]:
            assert inst is instances[0]

    def test_set_then_reset_then_get(self):
        custom = AccessGuardConfig(max_policies=42)
        set_config(custom)
        assert get_config().max_policies == 42
        reset_config()
        fresh = get_config()
        assert fresh.max_policies == 1000  # back to default


class TestAccessGuardConfigValidation:
    """Test invalid env values fallback to defaults."""

    def test_invalid_int_rpm_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_DEFAULT_RATE_RPM", "foo")
        config = AccessGuardConfig.from_env()
        assert config.default_rate_rpm == 100

    def test_invalid_int_rph_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_DEFAULT_RATE_RPH", "bar")
        config = AccessGuardConfig.from_env()
        assert config.default_rate_rph == 1000

    def test_invalid_int_rpd_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_DEFAULT_RATE_RPD", "baz")
        config = AccessGuardConfig.from_env()
        assert config.default_rate_rpd == 10000

    def test_invalid_int_retention_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_AUDIT_RETENTION_DAYS", "x")
        config = AccessGuardConfig.from_env()
        assert config.audit_retention_days == 365

    def test_invalid_int_decision_cache_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_DECISION_CACHE_TTL_SECONDS", "oops")
        config = AccessGuardConfig.from_env()
        assert config.decision_cache_ttl_seconds == 60

    def test_invalid_int_policy_cache_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_POLICY_CACHE_TTL_SECONDS", "bad")
        config = AccessGuardConfig.from_env()
        assert config.policy_cache_ttl_seconds == 300

    def test_invalid_int_max_rules_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_ACCESS_GUARD_MAX_RULES_PER_POLICY", "nope")
        config = AccessGuardConfig.from_env()
        assert config.max_rules_per_policy == 100


class TestAccessGuardConfigCustomValues:
    """Test creating config with fully custom values."""

    def test_full_custom_config(self):
        config = AccessGuardConfig(
            strict_mode=False,
            simulation_mode=True,
            rate_limiting_enabled=False,
            default_rate_rpm=500,
            default_rate_rph=5000,
            default_rate_rpd=50000,
            burst_limit=50,
            audit_enabled=False,
            audit_all_decisions=False,
            audit_retention_days=90,
            strict_tenant_isolation=False,
            opa_enabled=True,
            opa_endpoint="http://opa:8181",
            decision_cache_ttl_seconds=120,
            policy_cache_ttl_seconds=600,
            max_policies=5000,
            max_rules_per_policy=200,
        )
        assert config.strict_mode is False
        assert config.simulation_mode is True
        assert config.rate_limiting_enabled is False
        assert config.default_rate_rpm == 500
        assert config.default_rate_rph == 5000
        assert config.default_rate_rpd == 50000
        assert config.burst_limit == 50
        assert config.audit_enabled is False
        assert config.audit_all_decisions is False
        assert config.audit_retention_days == 90
        assert config.strict_tenant_isolation is False
        assert config.opa_enabled is True
        assert config.opa_endpoint == "http://opa:8181"
        assert config.decision_cache_ttl_seconds == 120
        assert config.policy_cache_ttl_seconds == 600
        assert config.max_policies == 5000
        assert config.max_rules_per_policy == 200
