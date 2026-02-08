# -*- coding: utf-8 -*-
"""
QA Test Harness Service Configuration - AGENT-FOUND-009: QA Test Harness

Centralized configuration for the QA Test Harness SDK covering:
- Test execution defaults (timeout, parallelism, fail-fast)
- Golden file management (directory, cache TTL)
- Zero-hallucination and determinism toggles
- Performance benchmarking defaults (warmup, iterations)
- Coverage tracking toggle
- Report format defaults
- Connection pool sizing

All settings can be overridden via environment variables with the
``GL_QA_TEST_HARNESS_`` prefix (e.g. ``GL_QA_TEST_HARNESS_DEFAULT_TIMEOUT_SECONDS``).

Example:
    >>> from greenlang.qa_test_harness.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_timeout_seconds, cfg.max_parallel_workers)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-009 QA Test Harness
Status: Production Ready
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_QA_TEST_HARNESS_"


# ---------------------------------------------------------------------------
# QATestHarnessConfig
# ---------------------------------------------------------------------------


@dataclass
class QATestHarnessConfig:
    """Complete configuration for the GreenLang QA Test Harness SDK.

    Attributes are grouped by concern: connections, test execution,
    golden files, feature toggles, determinism, performance, reporting,
    pool sizing, and logging.

    All attributes can be overridden via environment variables using the
    ``GL_QA_TEST_HARNESS_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage.
        redis_url: Redis connection URL for caching layer.
        default_timeout_seconds: Default timeout for individual test execution.
        max_parallel_workers: Maximum number of parallel test workers.
        golden_file_directory: Directory path for golden file storage.
        enable_coverage_tracking: Whether to track test coverage metrics.
        enable_zero_hallucination_checks: Whether to run zero-hallucination checks.
        enable_determinism_checks: Whether to run determinism checks.
        determinism_iterations: Number of iterations for determinism verification.
        performance_warmup_iterations: Number of warmup iterations before benchmarking.
        performance_default_iterations: Default number of benchmark iterations.
        fail_fast: Whether to stop test suite on first failure.
        golden_file_cache_ttl_seconds: TTL for golden file content caching.
        report_format: Default report format (text, json, markdown, html).
        pool_min_size: Minimum connection pool size.
        pool_max_size: Maximum connection pool size.
        log_level: Logging level for the QA test harness service.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = ""
    redis_url: str = ""

    # -- Test execution ------------------------------------------------------
    default_timeout_seconds: int = 60
    max_parallel_workers: int = 4
    fail_fast: bool = False

    # -- Golden files --------------------------------------------------------
    golden_file_directory: str = "./golden_files"
    golden_file_cache_ttl_seconds: int = 3600

    # -- Feature toggles -----------------------------------------------------
    enable_coverage_tracking: bool = True
    enable_zero_hallucination_checks: bool = True
    enable_determinism_checks: bool = True

    # -- Determinism ---------------------------------------------------------
    determinism_iterations: int = 3

    # -- Performance ---------------------------------------------------------
    performance_warmup_iterations: int = 2
    performance_default_iterations: int = 10

    # -- Reporting -----------------------------------------------------------
    report_format: str = "markdown"

    # -- Pool sizing ---------------------------------------------------------
    pool_min_size: int = 2
    pool_max_size: int = 10

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> QATestHarnessConfig:
        """Build a QATestHarnessConfig from environment variables.

        Every field can be overridden via ``GL_QA_TEST_HARNESS_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.

        Returns:
            Populated QATestHarnessConfig instance.
        """
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
                logger.warning(
                    "Invalid integer for %s%s=%s, using default %d",
                    prefix, name, val, default,
                )
                return default

        def _str(name: str, default: str) -> str:
            val = _env(name)
            if val is None:
                return default
            return val

        config = cls(
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

        logger.info(
            "QATestHarnessConfig loaded: timeout=%ds, workers=%d, "
            "fail_fast=%s, golden_dir=%s, determinism_iters=%d, "
            "zero_hallucination=%s, coverage=%s, report_format=%s",
            config.default_timeout_seconds,
            config.max_parallel_workers,
            config.fail_fast,
            config.golden_file_directory,
            config.determinism_iterations,
            config.enable_zero_hallucination_checks,
            config.enable_coverage_tracking,
            config.report_format,
        )
        return config


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[QATestHarnessConfig] = None
_config_lock = threading.Lock()


def get_config() -> QATestHarnessConfig:
    """Return the singleton QATestHarnessConfig, creating from env if needed.

    Returns:
        QATestHarnessConfig singleton instance.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = QATestHarnessConfig.from_env()
    return _config_instance


def set_config(config: QATestHarnessConfig) -> None:
    """Replace the singleton QATestHarnessConfig (useful for testing).

    Args:
        config: New configuration to install.
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info("QATestHarnessConfig replaced programmatically")


def reset_config() -> None:
    """Reset the singleton (primarily for test teardown)."""
    global _config_instance
    with _config_lock:
        _config_instance = None


__all__ = [
    "QATestHarnessConfig",
    "get_config",
    "set_config",
    "reset_config",
]
