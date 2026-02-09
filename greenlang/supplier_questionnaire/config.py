# -*- coding: utf-8 -*-
"""
Supplier Questionnaire Processor Service Configuration - AGENT-DATA-008

Centralized configuration for the Supplier Questionnaire Processor SDK covering:
- Database and cache connection URLs
- Questionnaire framework defaults
- Deadline and reminder scheduling
- Scoring thresholds (leader, advanced, developing, lagging)
- Batch processing and worker concurrency
- Connection pool sizing
- Portal and email integration
- Localization defaults
- Data retention

All settings can be overridden via environment variables with the
``GL_SUPPLIER_QUEST_`` prefix (e.g. ``GL_SUPPLIER_QUEST_DEFAULT_FRAMEWORK``).

Example:
    >>> from greenlang.supplier_questionnaire.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_framework, cfg.default_deadline_days)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-008 Supplier Questionnaire Processor
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

_ENV_PREFIX = "GL_SUPPLIER_QUEST_"


# ---------------------------------------------------------------------------
# SupplierQuestionnaireConfig
# ---------------------------------------------------------------------------


@dataclass
class SupplierQuestionnaireConfig:
    """Complete configuration for the GreenLang Supplier Questionnaire SDK.

    Attributes are grouped by concern: connections, questionnaire defaults,
    deadline and reminder scheduling, scoring thresholds, batch processing,
    pool sizing, portal integration, email settings, localization, retention,
    and logging.

    All attributes can be overridden via environment variables using the
    ``GL_SUPPLIER_QUEST_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage.
        redis_url: Redis connection URL for caching layer.
        log_level: Logging level for the supplier questionnaire service.
        default_framework: Default questionnaire framework for new templates.
        default_deadline_days: Default number of days for questionnaire deadlines.
        max_reminders: Maximum number of reminders per distribution.
        reminder_gentle_days: Days before deadline for gentle reminder.
        reminder_firm_days: Days before deadline for firm reminder.
        reminder_urgent_days: Days before deadline for urgent reminder.
        min_completion_pct: Minimum completion percentage to accept responses.
        score_leader_threshold: Minimum score for Leader performance tier.
        score_advanced_threshold: Minimum score for Advanced performance tier.
        score_developing_threshold: Minimum score for Developing performance tier.
        score_lagging_threshold: Minimum score for Lagging performance tier.
        batch_size: Number of records per processing batch.
        worker_count: Number of parallel workers for batch processing.
        cache_ttl_seconds: Cache time-to-live in seconds for template lookups.
        pool_min_size: Minimum connection pool size.
        pool_max_size: Maximum connection pool size.
        retention_days: Data retention period in days (default 3 years).
        portal_base_url: Base URL for the supplier self-service portal.
        smtp_host: SMTP host for outbound email notifications.
        default_language: Default language code (ISO 639-1) for templates.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = ""
    redis_url: str = ""

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Questionnaire defaults ----------------------------------------------
    default_framework: str = "custom"
    default_deadline_days: int = 60

    # -- Reminder scheduling -------------------------------------------------
    max_reminders: int = 4
    reminder_gentle_days: int = 7
    reminder_firm_days: int = 3
    reminder_urgent_days: int = 1

    # -- Completion and scoring thresholds -----------------------------------
    min_completion_pct: float = 80.0
    score_leader_threshold: int = 80
    score_advanced_threshold: int = 60
    score_developing_threshold: int = 40
    score_lagging_threshold: int = 20

    # -- Batch processing ----------------------------------------------------
    batch_size: int = 100
    worker_count: int = 4

    # -- Caching -------------------------------------------------------------
    cache_ttl_seconds: int = 1800

    # -- Pool sizing ---------------------------------------------------------
    pool_min_size: int = 2
    pool_max_size: int = 10

    # -- Retention -----------------------------------------------------------
    retention_days: int = 1095

    # -- Portal integration --------------------------------------------------
    portal_base_url: str = ""

    # -- Email integration ---------------------------------------------------
    smtp_host: str = ""

    # -- Localization --------------------------------------------------------
    default_language: str = "en"

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> SupplierQuestionnaireConfig:
        """Build a SupplierQuestionnaireConfig from environment variables.

        Every field can be overridden via ``GL_SUPPLIER_QUEST_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.

        Returns:
            Populated SupplierQuestionnaireConfig instance.
        """
        prefix = _ENV_PREFIX

        def _env(name: str, default: Any = None) -> Optional[str]:
            return os.environ.get(f"{prefix}{name}", default)

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

        def _float(name: str, default: float) -> float:
            val = _env(name)
            if val is None:
                return default
            try:
                return float(val)
            except ValueError:
                logger.warning(
                    "Invalid float for %s%s=%s, using default %s",
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
            log_level=_str("LOG_LEVEL", cls.log_level),
            default_framework=_str(
                "DEFAULT_FRAMEWORK", cls.default_framework,
            ),
            default_deadline_days=_int(
                "DEFAULT_DEADLINE_DAYS", cls.default_deadline_days,
            ),
            max_reminders=_int(
                "MAX_REMINDERS", cls.max_reminders,
            ),
            reminder_gentle_days=_int(
                "REMINDER_GENTLE_DAYS", cls.reminder_gentle_days,
            ),
            reminder_firm_days=_int(
                "REMINDER_FIRM_DAYS", cls.reminder_firm_days,
            ),
            reminder_urgent_days=_int(
                "REMINDER_URGENT_DAYS", cls.reminder_urgent_days,
            ),
            min_completion_pct=_float(
                "MIN_COMPLETION_PCT", cls.min_completion_pct,
            ),
            score_leader_threshold=_int(
                "SCORE_LEADER_THRESHOLD", cls.score_leader_threshold,
            ),
            score_advanced_threshold=_int(
                "SCORE_ADVANCED_THRESHOLD", cls.score_advanced_threshold,
            ),
            score_developing_threshold=_int(
                "SCORE_DEVELOPING_THRESHOLD", cls.score_developing_threshold,
            ),
            score_lagging_threshold=_int(
                "SCORE_LAGGING_THRESHOLD", cls.score_lagging_threshold,
            ),
            batch_size=_int("BATCH_SIZE", cls.batch_size),
            worker_count=_int("WORKER_COUNT", cls.worker_count),
            cache_ttl_seconds=_int(
                "CACHE_TTL_SECONDS", cls.cache_ttl_seconds,
            ),
            pool_min_size=_int("POOL_MIN_SIZE", cls.pool_min_size),
            pool_max_size=_int("POOL_MAX_SIZE", cls.pool_max_size),
            retention_days=_int("RETENTION_DAYS", cls.retention_days),
            portal_base_url=_str(
                "PORTAL_BASE_URL", cls.portal_base_url,
            ),
            smtp_host=_str("SMTP_HOST", cls.smtp_host),
            default_language=_str(
                "DEFAULT_LANGUAGE", cls.default_language,
            ),
        )

        logger.info(
            "SupplierQuestionnaireConfig loaded: framework=%s, "
            "deadline_days=%d, max_reminders=%d, "
            "min_completion=%.1f%%, leader_threshold=%d, "
            "batch_size=%d, workers=%d, cache_ttl=%ds, "
            "retention=%dd, language=%s",
            config.default_framework,
            config.default_deadline_days,
            config.max_reminders,
            config.min_completion_pct,
            config.score_leader_threshold,
            config.batch_size,
            config.worker_count,
            config.cache_ttl_seconds,
            config.retention_days,
            config.default_language,
        )
        return config


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[SupplierQuestionnaireConfig] = None
_config_lock = threading.Lock()


def get_config() -> SupplierQuestionnaireConfig:
    """Return the singleton SupplierQuestionnaireConfig, creating from env if needed.

    Returns:
        SupplierQuestionnaireConfig singleton instance.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = SupplierQuestionnaireConfig.from_env()
    return _config_instance


def set_config(config: SupplierQuestionnaireConfig) -> None:
    """Replace the singleton SupplierQuestionnaireConfig (useful for testing).

    Args:
        config: New configuration to install.
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info("SupplierQuestionnaireConfig replaced programmatically")


def reset_config() -> None:
    """Reset the singleton (primarily for test teardown)."""
    global _config_instance
    with _config_lock:
        _config_instance = None


__all__ = [
    "SupplierQuestionnaireConfig",
    "get_config",
    "set_config",
    "reset_config",
]
