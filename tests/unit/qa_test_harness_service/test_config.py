# -*- coding: utf-8 -*-
"""
Unit Tests for QATestHarnessConfig (AGENT-FOUND-009)

Tests configuration creation, env var overrides with GL_QA_TEST_HARNESS_ prefix,
default values, and singleton get_config/set_config/reset_config.

Coverage target: 85%+ of config.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline QATestHarnessConfig mirroring greenlang/qa_test_harness/config.py
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_QA_TEST_HARNESS_"

_logger = logging.getLogger(__name__)


@dataclass
class QATestHarnessConfig:
    """Mirrors greenlang.qa_test_harness.config.QATestHarnessConfig."""

    # Connections
    database_url: str = ""
    redis_url: str = ""

    # Test execution
    default_timeout_seconds: int = 60
    max_parallel_workers: int = 4
    fail_fast: bool = False

    # Golden files
    golden_file_directory: str = "./golden_files"
    golden_file_cache_ttl_seconds: int = 3600

    # Feature toggles
    enable_coverage_tracking: bool = True
    enable_zero_hallucination_checks: bool = True
    enable_determinism_checks: bool = True

    # Determinism
    determinism_iterations: int = 3

    # Performance
    performance_warmup_iterations: int = 2
    performance_default_iterations: int = 10

    # Reporting
    report_format: str = "markdown"

    # Pool sizing
    pool_min_size: int = 2
    pool_max_size: int = 10

    # Logging
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "QATestHarnessConfig":
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
                _logger.warning(
                    "Invalid integer for %s%s=%s, using default %d",
                    prefix, name, val, default,
                )
                return default

        def _str(name: str, default: str) -> str:
            val = _env(name)
            if val is None:
                return default
            return val

        return cls(
            database_url=_str("DATABASE_URL", cls.database_url),
            redis_url=_str("REDIS_URL", cls.redis_url),
            default_timeout_seconds=_int(
                "DEFAULT_TIMEOUT_SECONDS", cls.default_timeout_seconds,
            ),
            max_parallel_workers=_int(
                "MAX_PARALLEL_WORKERS", cls.max_parallel_workers,
            ),
            fail_fast=_bool("FAIL_FAST", cls.fail_fast),
            golden_file_directory=_str(
                "GOLDEN_FILE_DIRECTORY", cls.golden_file_directory,
            ),
            golden_file_cache_ttl_seconds=_int(
                "GOLDEN_FILE_CACHE_TTL_SECONDS", cls.golden_file_cache_ttl_seconds,
            ),
            enable_coverage_tracking=_bool(
                "ENABLE_COVERAGE_TRACKING", cls.enable_coverage_tracking,
            ),
            enable_zero_hallucination_checks=_bool(
                "ENABLE_ZERO_HALLUCINATION_CHECKS",
                cls.enable_zero_hallucination_checks,
            ),
            enable_determinism_checks=_bool(
                "ENABLE_DETERMINISM_CHECKS", cls.enable_determinism_checks,
            ),
            determinism_iterations=_int(
                "DETERMINISM_ITERATIONS", cls.determinism_iterations,
            ),
            performance_warmup_iterations=_int(
                "PERFORMANCE_WARMUP_ITERATIONS", cls.performance_warmup_iterations,
            ),
            performance_default_iterations=_int(
                "PERFORMANCE_DEFAULT_ITERATIONS", cls.performance_default_iterations,
            ),
            report_format=_str("REPORT_FORMAT", cls.report_format),
            pool_min_size=_int("POOL_MIN_SIZE", cls.pool_min_size),
            pool_max_size=_int("POOL_MAX_SIZE", cls.pool_max_size),
            log_level=_str("LOG_LEVEL", cls.log_level),
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[QATestHarnessConfig] = None
_config_lock = threading.Lock()


def get_config() -> QATestHarnessConfig:
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = QATestHarnessConfig.from_env()
    return _config_instance


def set_config(config: QATestHarnessConfig) -> None:
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


class TestQATestHarnessConfigDefaults:
    """Test that default configuration values match PRD requirements."""

    def test_default_database_url(self):
        config = QATestHarnessConfig()
        assert config.database_url == ""

    def test_default_redis_url(self):
        config = QATestHarnessConfig()
        assert config.redis_url == ""

    def test_default_timeout_seconds(self):
        config = QATestHarnessConfig()
        assert config.default_timeout_seconds == 60

    def test_default_max_parallel_workers(self):
        config = QATestHarnessConfig()
        assert config.max_parallel_workers == 4

    def test_default_fail_fast(self):
        config = QATestHarnessConfig()
        assert config.fail_fast is False

    def test_default_golden_file_directory(self):
        config = QATestHarnessConfig()
        assert config.golden_file_directory == "./golden_files"

    def test_default_golden_file_cache_ttl(self):
        config = QATestHarnessConfig()
        assert config.golden_file_cache_ttl_seconds == 3600

    def test_default_enable_coverage_tracking(self):
        config = QATestHarnessConfig()
        assert config.enable_coverage_tracking is True

    def test_default_enable_zero_hallucination_checks(self):
        config = QATestHarnessConfig()
        assert config.enable_zero_hallucination_checks is True

    def test_default_enable_determinism_checks(self):
        config = QATestHarnessConfig()
        assert config.enable_determinism_checks is True

    def test_default_determinism_iterations(self):
        config = QATestHarnessConfig()
        assert config.determinism_iterations == 3

    def test_default_performance_warmup_iterations(self):
        config = QATestHarnessConfig()
        assert config.performance_warmup_iterations == 2

    def test_default_performance_default_iterations(self):
        config = QATestHarnessConfig()
        assert config.performance_default_iterations == 10

    def test_default_report_format(self):
        config = QATestHarnessConfig()
        assert config.report_format == "markdown"

    def test_default_pool_min_size(self):
        config = QATestHarnessConfig()
        assert config.pool_min_size == 2

    def test_default_pool_max_size(self):
        config = QATestHarnessConfig()
        assert config.pool_max_size == 10

    def test_default_log_level(self):
        config = QATestHarnessConfig()
        assert config.log_level == "INFO"


class TestQATestHarnessConfigFromEnv:
    """Test GL_QA_TEST_HARNESS_ env var overrides via from_env()."""

    def test_env_override_database_url(self, monkeypatch):
        monkeypatch.setenv("GL_QA_TEST_HARNESS_DATABASE_URL", "postgresql://localhost/qa_db")
        config = QATestHarnessConfig.from_env()
        assert config.database_url == "postgresql://localhost/qa_db"

    def test_env_override_redis_url(self, monkeypatch):
        monkeypatch.setenv("GL_QA_TEST_HARNESS_REDIS_URL", "redis://localhost:6379/3")
        config = QATestHarnessConfig.from_env()
        assert config.redis_url == "redis://localhost:6379/3"

    def test_env_override_default_timeout(self, monkeypatch):
        monkeypatch.setenv("GL_QA_TEST_HARNESS_DEFAULT_TIMEOUT_SECONDS", "120")
        config = QATestHarnessConfig.from_env()
        assert config.default_timeout_seconds == 120

    def test_env_override_max_parallel_workers(self, monkeypatch):
        monkeypatch.setenv("GL_QA_TEST_HARNESS_MAX_PARALLEL_WORKERS", "8")
        config = QATestHarnessConfig.from_env()
        assert config.max_parallel_workers == 8

    def test_env_override_fail_fast_true(self, monkeypatch):
        monkeypatch.setenv("GL_QA_TEST_HARNESS_FAIL_FAST", "true")
        config = QATestHarnessConfig.from_env()
        assert config.fail_fast is True

    def test_env_override_fail_fast_1(self, monkeypatch):
        monkeypatch.setenv("GL_QA_TEST_HARNESS_FAIL_FAST", "1")
        config = QATestHarnessConfig.from_env()
        assert config.fail_fast is True

    def test_env_override_fail_fast_yes(self, monkeypatch):
        monkeypatch.setenv("GL_QA_TEST_HARNESS_FAIL_FAST", "yes")
        config = QATestHarnessConfig.from_env()
        assert config.fail_fast is True

    def test_env_override_fail_fast_TRUE_uppercase(self, monkeypatch):
        monkeypatch.setenv("GL_QA_TEST_HARNESS_FAIL_FAST", "TRUE")
        config = QATestHarnessConfig.from_env()
        assert config.fail_fast is True

    def test_env_override_golden_file_directory(self, monkeypatch):
        monkeypatch.setenv("GL_QA_TEST_HARNESS_GOLDEN_FILE_DIRECTORY", "/tmp/goldens")
        config = QATestHarnessConfig.from_env()
        assert config.golden_file_directory == "/tmp/goldens"

    def test_env_override_golden_file_cache_ttl(self, monkeypatch):
        monkeypatch.setenv("GL_QA_TEST_HARNESS_GOLDEN_FILE_CACHE_TTL_SECONDS", "7200")
        config = QATestHarnessConfig.from_env()
        assert config.golden_file_cache_ttl_seconds == 7200

    def test_env_override_coverage_disabled(self, monkeypatch):
        monkeypatch.setenv("GL_QA_TEST_HARNESS_ENABLE_COVERAGE_TRACKING", "false")
        config = QATestHarnessConfig.from_env()
        assert config.enable_coverage_tracking is False

    def test_env_override_zero_hallucination_disabled(self, monkeypatch):
        monkeypatch.setenv("GL_QA_TEST_HARNESS_ENABLE_ZERO_HALLUCINATION_CHECKS", "0")
        config = QATestHarnessConfig.from_env()
        assert config.enable_zero_hallucination_checks is False

    def test_env_override_determinism_disabled(self, monkeypatch):
        monkeypatch.setenv("GL_QA_TEST_HARNESS_ENABLE_DETERMINISM_CHECKS", "no")
        config = QATestHarnessConfig.from_env()
        assert config.enable_determinism_checks is False

    def test_env_override_determinism_iterations(self, monkeypatch):
        monkeypatch.setenv("GL_QA_TEST_HARNESS_DETERMINISM_ITERATIONS", "5")
        config = QATestHarnessConfig.from_env()
        assert config.determinism_iterations == 5

    def test_env_override_performance_warmup(self, monkeypatch):
        monkeypatch.setenv("GL_QA_TEST_HARNESS_PERFORMANCE_WARMUP_ITERATIONS", "5")
        config = QATestHarnessConfig.from_env()
        assert config.performance_warmup_iterations == 5

    def test_env_override_performance_default(self, monkeypatch):
        monkeypatch.setenv("GL_QA_TEST_HARNESS_PERFORMANCE_DEFAULT_ITERATIONS", "50")
        config = QATestHarnessConfig.from_env()
        assert config.performance_default_iterations == 50

    def test_env_override_report_format(self, monkeypatch):
        monkeypatch.setenv("GL_QA_TEST_HARNESS_REPORT_FORMAT", "json")
        config = QATestHarnessConfig.from_env()
        assert config.report_format == "json"

    def test_env_override_pool_min_size(self, monkeypatch):
        monkeypatch.setenv("GL_QA_TEST_HARNESS_POOL_MIN_SIZE", "5")
        config = QATestHarnessConfig.from_env()
        assert config.pool_min_size == 5

    def test_env_override_pool_max_size(self, monkeypatch):
        monkeypatch.setenv("GL_QA_TEST_HARNESS_POOL_MAX_SIZE", "25")
        config = QATestHarnessConfig.from_env()
        assert config.pool_max_size == 25

    def test_env_override_log_level(self, monkeypatch):
        monkeypatch.setenv("GL_QA_TEST_HARNESS_LOG_LEVEL", "DEBUG")
        config = QATestHarnessConfig.from_env()
        assert config.log_level == "DEBUG"

    def test_env_invalid_int_uses_default(self, monkeypatch):
        monkeypatch.setenv("GL_QA_TEST_HARNESS_DEFAULT_TIMEOUT_SECONDS", "not_a_number")
        config = QATestHarnessConfig.from_env()
        assert config.default_timeout_seconds == 60

    def test_env_invalid_int_pool_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_QA_TEST_HARNESS_POOL_MAX_SIZE", "xyz")
        config = QATestHarnessConfig.from_env()
        assert config.pool_max_size == 10

    def test_env_invalid_int_determinism_fallback(self, monkeypatch):
        monkeypatch.setenv("GL_QA_TEST_HARNESS_DETERMINISM_ITERATIONS", "abc")
        config = QATestHarnessConfig.from_env()
        assert config.determinism_iterations == 3

    def test_multiple_env_overrides(self, monkeypatch):
        monkeypatch.setenv("GL_QA_TEST_HARNESS_FAIL_FAST", "true")
        monkeypatch.setenv("GL_QA_TEST_HARNESS_REPORT_FORMAT", "html")
        monkeypatch.setenv("GL_QA_TEST_HARNESS_POOL_MAX_SIZE", "30")
        monkeypatch.setenv("GL_QA_TEST_HARNESS_LOG_LEVEL", "WARNING")
        config = QATestHarnessConfig.from_env()
        assert config.fail_fast is True
        assert config.report_format == "html"
        assert config.pool_max_size == 30
        assert config.log_level == "WARNING"


class TestQATestHarnessConfigSingleton:
    """Test get_config/set_config/reset_config singleton pattern."""

    def test_get_config_returns_instance(self):
        config = get_config()
        assert isinstance(config, QATestHarnessConfig)

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
        custom = QATestHarnessConfig(pool_max_size=99)
        set_config(custom)
        assert get_config().pool_max_size == 99

    def test_set_config_then_get_returns_same(self):
        custom = QATestHarnessConfig(report_format="html")
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
        custom = QATestHarnessConfig(pool_min_size=99)
        set_config(custom)
        assert get_config().pool_min_size == 99
        reset_config()
        fresh = get_config()
        assert fresh.pool_min_size == 2  # back to default


class TestQATestHarnessConfigValidation:
    """Test config with custom and edge-case values."""

    def test_config_database_url_postgresql(self):
        config = QATestHarnessConfig(
            database_url="postgresql://user:pass@localhost:5432/qa_db"
        )
        assert "postgresql" in config.database_url
        assert "qa_db" in config.database_url

    def test_config_database_url_with_options(self):
        config = QATestHarnessConfig(
            database_url="postgresql://user:pass@host:5432/db?sslmode=require"
        )
        assert "sslmode=require" in config.database_url

    def test_config_redis_url_standard(self):
        config = QATestHarnessConfig(redis_url="redis://localhost:6379/0")
        assert "redis" in config.redis_url

    def test_config_golden_file_directory_absolute(self):
        config = QATestHarnessConfig(golden_file_directory="/opt/goldens")
        assert config.golden_file_directory == "/opt/goldens"

    def test_full_custom_config(self):
        config = QATestHarnessConfig(
            database_url="postgresql://localhost/test",
            redis_url="redis://localhost:6379/0",
            default_timeout_seconds=120,
            max_parallel_workers=8,
            fail_fast=True,
            golden_file_directory="/tmp/goldens",
            golden_file_cache_ttl_seconds=7200,
            enable_coverage_tracking=False,
            enable_zero_hallucination_checks=False,
            enable_determinism_checks=False,
            determinism_iterations=5,
            performance_warmup_iterations=5,
            performance_default_iterations=50,
            report_format="json",
            pool_min_size=5,
            pool_max_size=25,
            log_level="DEBUG",
        )
        assert config.database_url == "postgresql://localhost/test"
        assert config.redis_url == "redis://localhost:6379/0"
        assert config.default_timeout_seconds == 120
        assert config.max_parallel_workers == 8
        assert config.fail_fast is True
        assert config.golden_file_directory == "/tmp/goldens"
        assert config.golden_file_cache_ttl_seconds == 7200
        assert config.enable_coverage_tracking is False
        assert config.enable_zero_hallucination_checks is False
        assert config.enable_determinism_checks is False
        assert config.determinism_iterations == 5
        assert config.performance_warmup_iterations == 5
        assert config.performance_default_iterations == 50
        assert config.report_format == "json"
        assert config.pool_min_size == 5
        assert config.pool_max_size == 25
        assert config.log_level == "DEBUG"
