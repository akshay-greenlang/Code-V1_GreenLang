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
    >>> from greenlang.agents.data.supplier_questionnaire.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_framework, cfg.default_deadline_days)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-008 Supplier Questionnaire Processor
Status: Production Ready
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from greenlang.data_commons.config_base import (
    BaseDataConfig,
    EnvReader,
    create_config_singleton,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_SUPPLIER_QUEST_"


# ---------------------------------------------------------------------------
# SupplierQuestionnaireConfig
# ---------------------------------------------------------------------------


@dataclass
class SupplierQuestionnaireConfig(BaseDataConfig):
    """Configuration for the GreenLang Supplier Questionnaire SDK.

    Inherits shared connection, pool, batch, and logging fields from
    ``BaseDataConfig``.  Only questionnaire-specific fields are declared here.

    All attributes can be overridden via environment variables using the
    ``GL_SUPPLIER_QUEST_`` prefix.

    Attributes:
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
        retention_days: Data retention period in days (default 3 years).
        portal_base_url: Base URL for the supplier self-service portal.
        smtp_host: SMTP host for outbound email notifications.
        default_language: Default language code (ISO 639-1) for templates.
    """

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

    # -- Batch processing (questionnaire-specific) ---------------------------
    batch_size: int = 100
    worker_count: int = 4

    # -- Caching -------------------------------------------------------------
    cache_ttl_seconds: int = 1800

    # -- Retention -----------------------------------------------------------
    retention_days: int = 1095

    # -- Portal integration --------------------------------------------------
    portal_base_url: str = ""

    # -- Email integration ---------------------------------------------------
    smtp_host: str = ""

    # -- Localization --------------------------------------------------------
    default_language: str = "en"

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> SupplierQuestionnaireConfig:
        """Build a SupplierQuestionnaireConfig from environment variables.

        Every field can be overridden via ``GL_SUPPLIER_QUEST_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).

        Returns:
            Populated SupplierQuestionnaireConfig instance.
        """
        env = EnvReader(_ENV_PREFIX)
        base_kwargs = cls._base_kwargs_from_env(env)

        config = cls(
            **base_kwargs,
            default_framework=env.str(
                "DEFAULT_FRAMEWORK", cls.default_framework,
            ),
            default_deadline_days=env.int(
                "DEFAULT_DEADLINE_DAYS", cls.default_deadline_days,
            ),
            max_reminders=env.int(
                "MAX_REMINDERS", cls.max_reminders,
            ),
            reminder_gentle_days=env.int(
                "REMINDER_GENTLE_DAYS", cls.reminder_gentle_days,
            ),
            reminder_firm_days=env.int(
                "REMINDER_FIRM_DAYS", cls.reminder_firm_days,
            ),
            reminder_urgent_days=env.int(
                "REMINDER_URGENT_DAYS", cls.reminder_urgent_days,
            ),
            min_completion_pct=env.float(
                "MIN_COMPLETION_PCT", cls.min_completion_pct,
            ),
            score_leader_threshold=env.int(
                "SCORE_LEADER_THRESHOLD", cls.score_leader_threshold,
            ),
            score_advanced_threshold=env.int(
                "SCORE_ADVANCED_THRESHOLD", cls.score_advanced_threshold,
            ),
            score_developing_threshold=env.int(
                "SCORE_DEVELOPING_THRESHOLD", cls.score_developing_threshold,
            ),
            score_lagging_threshold=env.int(
                "SCORE_LAGGING_THRESHOLD", cls.score_lagging_threshold,
            ),
            batch_size=env.int("BATCH_SIZE", cls.batch_size),
            worker_count=env.int("WORKER_COUNT", cls.worker_count),
            cache_ttl_seconds=env.int(
                "CACHE_TTL_SECONDS", cls.cache_ttl_seconds,
            ),
            retention_days=env.int("RETENTION_DAYS", cls.retention_days),
            portal_base_url=env.str(
                "PORTAL_BASE_URL", cls.portal_base_url,
            ),
            smtp_host=env.str("SMTP_HOST", cls.smtp_host),
            default_language=env.str(
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

get_config, set_config, reset_config = create_config_singleton(
    SupplierQuestionnaireConfig, _ENV_PREFIX,
)

__all__ = [
    "SupplierQuestionnaireConfig",
    "get_config",
    "set_config",
    "reset_config",
]
