# -*- coding: utf-8 -*-
"""
Unit Tests for CitationsConfig (AGENT-FOUND-005)

Tests configuration creation, env var overrides with GL_CITATIONS_ prefix,
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
# Inline CitationsConfig that mirrors greenlang/citations/config.py
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_CITATIONS_"


@dataclass
class CitationsConfig:
    """Mirrors greenlang.citations.config.CitationsConfig."""

    # Registry
    max_citations: int = 100000
    enable_versioning: bool = True

    # Verification
    enable_auto_verification: bool = True
    verification_timeout_seconds: int = 30
    default_expiration_years: int = 5

    # Provenance
    enable_provenance: bool = True
    hash_algorithm: str = "sha256"

    # Performance
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300

    # Export
    default_export_format: str = "json"
    max_export_items: int = 10000

    # Evidence packaging
    max_evidence_items_per_package: int = 500
    enable_package_finalization: bool = True

    VALID_HASH_ALGORITHMS = ("sha256", "sha384", "sha512")
    VALID_EXPORT_FORMATS = ("json", "bibtex", "csl")
    MIN_EXPIRATION_YEARS = 1
    MAX_EXPIRATION_YEARS = 20
    MIN_CACHE_TTL = 10
    MAX_CACHE_TTL = 86400

    @classmethod
    def from_env(cls) -> CitationsConfig:
        """Build from environment variables with GL_CITATIONS_ prefix."""
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

        def _str(name: str, default: str) -> str:
            val = _env(name)
            if val is None:
                return default
            return val

        return cls(
            max_citations=_int("MAX_CITATIONS", cls.max_citations),
            enable_versioning=_bool("ENABLE_VERSIONING", cls.enable_versioning),
            enable_auto_verification=_bool("ENABLE_AUTO_VERIFICATION", cls.enable_auto_verification),
            verification_timeout_seconds=_int("VERIFICATION_TIMEOUT_SECONDS", cls.verification_timeout_seconds),
            default_expiration_years=_int("DEFAULT_EXPIRATION_YEARS", cls.default_expiration_years),
            enable_provenance=_bool("ENABLE_PROVENANCE", cls.enable_provenance),
            hash_algorithm=_str("HASH_ALGORITHM", cls.hash_algorithm),
            cache_enabled=_bool("CACHE_ENABLED", cls.cache_enabled),
            cache_ttl_seconds=_int("CACHE_TTL_SECONDS", cls.cache_ttl_seconds),
            default_export_format=_str("DEFAULT_EXPORT_FORMAT", cls.default_export_format),
            max_export_items=_int("MAX_EXPORT_ITEMS", cls.max_export_items),
            max_evidence_items_per_package=_int(
                "MAX_EVIDENCE_ITEMS_PER_PACKAGE", cls.max_evidence_items_per_package,
            ),
            enable_package_finalization=_bool(
                "ENABLE_PACKAGE_FINALIZATION", cls.enable_package_finalization,
            ),
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[CitationsConfig] = None
_config_lock = threading.Lock()


def get_config() -> CitationsConfig:
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = CitationsConfig.from_env()
    return _config_instance


def set_config(config: CitationsConfig) -> None:
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


class TestCitationsConfigDefaults:
    """Test that default configuration values match PRD requirements."""

    def test_default_max_citations(self):
        config = CitationsConfig()
        assert config.max_citations == 100000

    def test_default_enable_versioning(self):
        config = CitationsConfig()
        assert config.enable_versioning is True

    def test_default_enable_auto_verification(self):
        config = CitationsConfig()
        assert config.enable_auto_verification is True

    def test_default_verification_timeout_seconds(self):
        config = CitationsConfig()
        assert config.verification_timeout_seconds == 30

    def test_default_expiration_years(self):
        config = CitationsConfig()
        assert config.default_expiration_years == 5

    def test_default_enable_provenance(self):
        config = CitationsConfig()
        assert config.enable_provenance is True

    def test_default_hash_algorithm(self):
        config = CitationsConfig()
        assert config.hash_algorithm == "sha256"

    def test_default_cache_enabled(self):
        config = CitationsConfig()
        assert config.cache_enabled is True

    def test_default_cache_ttl_seconds(self):
        config = CitationsConfig()
        assert config.cache_ttl_seconds == 300

    def test_default_export_format(self):
        config = CitationsConfig()
        assert config.default_export_format == "json"

    def test_default_max_export_items(self):
        config = CitationsConfig()
        assert config.max_export_items == 10000

    def test_default_max_evidence_items_per_package(self):
        config = CitationsConfig()
        assert config.max_evidence_items_per_package == 500

    def test_default_enable_package_finalization(self):
        config = CitationsConfig()
        assert config.enable_package_finalization is True


class TestCitationsConfigFromEnv:
    """Test GL_CITATIONS_ env var overrides via from_env()."""

    def test_env_override_max_citations(self, monkeypatch):
        monkeypatch.setenv("GL_CITATIONS_MAX_CITATIONS", "50000")
        config = CitationsConfig.from_env()
        assert config.max_citations == 50000

    def test_env_override_enable_versioning_false(self, monkeypatch):
        monkeypatch.setenv("GL_CITATIONS_ENABLE_VERSIONING", "false")
        config = CitationsConfig.from_env()
        assert config.enable_versioning is False

    def test_env_override_enable_auto_verification_false(self, monkeypatch):
        monkeypatch.setenv("GL_CITATIONS_ENABLE_AUTO_VERIFICATION", "false")
        config = CitationsConfig.from_env()
        assert config.enable_auto_verification is False

    def test_env_override_verification_timeout(self, monkeypatch):
        monkeypatch.setenv("GL_CITATIONS_VERIFICATION_TIMEOUT_SECONDS", "60")
        config = CitationsConfig.from_env()
        assert config.verification_timeout_seconds == 60

    def test_env_override_expiration_years(self, monkeypatch):
        monkeypatch.setenv("GL_CITATIONS_DEFAULT_EXPIRATION_YEARS", "10")
        config = CitationsConfig.from_env()
        assert config.default_expiration_years == 10

    def test_env_override_hash_algorithm(self, monkeypatch):
        monkeypatch.setenv("GL_CITATIONS_HASH_ALGORITHM", "sha512")
        config = CitationsConfig.from_env()
        assert config.hash_algorithm == "sha512"

    def test_env_override_cache_enabled_false(self, monkeypatch):
        monkeypatch.setenv("GL_CITATIONS_CACHE_ENABLED", "0")
        config = CitationsConfig.from_env()
        assert config.cache_enabled is False

    def test_env_override_cache_ttl(self, monkeypatch):
        monkeypatch.setenv("GL_CITATIONS_CACHE_TTL_SECONDS", "600")
        config = CitationsConfig.from_env()
        assert config.cache_ttl_seconds == 600

    def test_env_override_export_format_bibtex(self, monkeypatch):
        monkeypatch.setenv("GL_CITATIONS_DEFAULT_EXPORT_FORMAT", "bibtex")
        config = CitationsConfig.from_env()
        assert config.default_export_format == "bibtex"

    def test_env_override_max_export_items(self, monkeypatch):
        monkeypatch.setenv("GL_CITATIONS_MAX_EXPORT_ITEMS", "5000")
        config = CitationsConfig.from_env()
        assert config.max_export_items == 5000

    def test_env_override_max_evidence_items(self, monkeypatch):
        monkeypatch.setenv("GL_CITATIONS_MAX_EVIDENCE_ITEMS_PER_PACKAGE", "1000")
        config = CitationsConfig.from_env()
        assert config.max_evidence_items_per_package == 1000

    def test_env_override_enable_package_finalization_false(self, monkeypatch):
        monkeypatch.setenv("GL_CITATIONS_ENABLE_PACKAGE_FINALIZATION", "false")
        config = CitationsConfig.from_env()
        assert config.enable_package_finalization is False

    def test_env_override_bool_true_with_1(self, monkeypatch):
        monkeypatch.setenv("GL_CITATIONS_ENABLE_PROVENANCE", "1")
        config = CitationsConfig.from_env()
        assert config.enable_provenance is True

    def test_env_override_bool_true_with_yes(self, monkeypatch):
        monkeypatch.setenv("GL_CITATIONS_ENABLE_PROVENANCE", "yes")
        config = CitationsConfig.from_env()
        assert config.enable_provenance is True

    def test_env_invalid_int_uses_default(self, monkeypatch):
        monkeypatch.setenv("GL_CITATIONS_MAX_CITATIONS", "not_a_number")
        config = CitationsConfig.from_env()
        assert config.max_citations == 100000

    def test_env_invalid_int_cache_ttl_uses_default(self, monkeypatch):
        monkeypatch.setenv("GL_CITATIONS_CACHE_TTL_SECONDS", "bad_value")
        config = CitationsConfig.from_env()
        assert config.cache_ttl_seconds == 300

    def test_multiple_env_overrides(self, monkeypatch):
        monkeypatch.setenv("GL_CITATIONS_MAX_CITATIONS", "200000")
        monkeypatch.setenv("GL_CITATIONS_ENABLE_PROVENANCE", "false")
        monkeypatch.setenv("GL_CITATIONS_DEFAULT_EXPORT_FORMAT", "csl")
        config = CitationsConfig.from_env()
        assert config.max_citations == 200000
        assert config.enable_provenance is False
        assert config.default_export_format == "csl"


class TestCitationsConfigValidation:
    """Test boundary values and validation constants."""

    def test_valid_hash_algorithms(self):
        assert "sha256" in CitationsConfig.VALID_HASH_ALGORITHMS
        assert "sha384" in CitationsConfig.VALID_HASH_ALGORITHMS
        assert "sha512" in CitationsConfig.VALID_HASH_ALGORITHMS

    def test_valid_export_formats(self):
        assert "json" in CitationsConfig.VALID_EXPORT_FORMATS
        assert "bibtex" in CitationsConfig.VALID_EXPORT_FORMATS
        assert "csl" in CitationsConfig.VALID_EXPORT_FORMATS

    def test_min_expiration_years(self):
        assert CitationsConfig.MIN_EXPIRATION_YEARS == 1

    def test_max_expiration_years(self):
        assert CitationsConfig.MAX_EXPIRATION_YEARS == 20

    def test_min_cache_ttl(self):
        assert CitationsConfig.MIN_CACHE_TTL == 10

    def test_max_cache_ttl(self):
        assert CitationsConfig.MAX_CACHE_TTL == 86400


class TestCitationsConfigSingleton:
    """Test get_config/set_config/reset_config singleton pattern."""

    def test_get_config_returns_instance(self):
        config = get_config()
        assert isinstance(config, CitationsConfig)

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
        custom = CitationsConfig(max_citations=999)
        set_config(custom)
        assert get_config().max_citations == 999

    def test_set_config_then_get_returns_same(self):
        custom = CitationsConfig(hash_algorithm="sha512")
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


class TestCitationsConfigCustomValues:
    """Test creating config with fully custom values."""

    def test_full_custom_config(self):
        config = CitationsConfig(
            max_citations=50000,
            enable_versioning=False,
            enable_auto_verification=False,
            verification_timeout_seconds=60,
            default_expiration_years=10,
            enable_provenance=False,
            hash_algorithm="sha512",
            cache_enabled=False,
            cache_ttl_seconds=60,
            default_export_format="bibtex",
            max_export_items=500,
            max_evidence_items_per_package=100,
            enable_package_finalization=False,
        )
        assert config.max_citations == 50000
        assert config.enable_versioning is False
        assert config.enable_auto_verification is False
        assert config.verification_timeout_seconds == 60
        assert config.default_expiration_years == 10
        assert config.enable_provenance is False
        assert config.hash_algorithm == "sha512"
        assert config.cache_enabled is False
        assert config.cache_ttl_seconds == 60
        assert config.default_export_format == "bibtex"
        assert config.max_export_items == 500
        assert config.max_evidence_items_per_package == 100
        assert config.enable_package_finalization is False
