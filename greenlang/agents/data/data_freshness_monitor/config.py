# -*- coding: utf-8 -*-
"""
Data Freshness Monitor Agent Service Configuration - AGENT-DATA-016

Centralized configuration for the Data Freshness Monitor SDK covering:
- Database, cache connection defaults
- Dataset processing limits (max datasets, batch size)
- SLA thresholds (warning and critical hours)
- Freshness tier boundaries (excellent, good, fair, poor hours)
- Check interval and alert throttling settings
- Prediction engine parameters (history window, minimum samples)
- Staleness pattern detection window
- Worker, pool, cache, rate limiting, and provenance settings
- Alert and escalation feature toggles

All settings can be overridden via environment variables with the
``GL_DFM_`` prefix (e.g. ``GL_DFM_MAX_DATASETS``).

Example:
    >>> from greenlang.agents.data.data_freshness_monitor.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_sla_warning_hours, cfg.default_sla_critical_hours)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor (GL-DATA-X-019)
Status: Production Ready
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

from greenlang.data_commons.config_base import (
    BaseDataConfig,
    EnvReader,
    create_config_singleton,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_DFM_"


# ---------------------------------------------------------------------------
# DataFreshnessMonitorConfig
# ---------------------------------------------------------------------------


@dataclass
class DataFreshnessMonitorConfig(BaseDataConfig):
    """Configuration for the GreenLang Data Freshness Monitor SDK.

    Inherits shared connection, pool, batch, and logging fields from
    ``BaseDataConfig``.  Only freshness-monitor-specific fields are declared here.

    All attributes can be overridden via environment variables using the
    ``GL_DFM_`` prefix.

    Attributes:
        batch_size: Default batch size for dataset processing.
        max_datasets: Maximum datasets allowed to be monitored concurrently.
        default_sla_warning_hours: Hours before a warning-level SLA alert.
        default_sla_critical_hours: Hours before a critical-level SLA alert.
        freshness_excellent_hours: Upper bound for excellent freshness.
        freshness_good_hours: Upper bound for good freshness.
        freshness_fair_hours: Upper bound for fair freshness.
        freshness_poor_hours: Upper bound for poor freshness.
        check_interval_minutes: Interval between scheduled freshness checks.
        alert_throttle_minutes: Minimum minutes between consecutive alerts.
        alert_dedup_window_hours: Hours for duplicate alert suppression.
        prediction_history_days: Historical days for prediction engine.
        prediction_min_samples: Minimum observations for predictions.
        staleness_pattern_window_days: Sliding window for staleness patterns.
        max_workers: Number of parallel workers for batch processing.
        pool_size: Connection pool size for database connections.
        cache_ttl: Cache time-to-live in seconds.
        rate_limit: Rate limit in requests per minute.
        enable_provenance: Whether SHA-256 provenance tracking is enabled.
        enable_predictions: Whether predictive staleness engine is enabled.
        enable_alerts: Whether alert generation is enabled.
        escalation_enabled: Whether automatic alert escalation is enabled.
        genesis_hash: Seed string for the SHA-256 provenance chain.
    """

    # -- Dataset processing --------------------------------------------------
    batch_size: int = 1000
    max_datasets: int = 50_000

    # -- SLA thresholds ------------------------------------------------------
    default_sla_warning_hours: float = 24.0
    default_sla_critical_hours: float = 72.0

    # -- Freshness tiers -----------------------------------------------------
    freshness_excellent_hours: float = 1.0
    freshness_good_hours: float = 6.0
    freshness_fair_hours: float = 24.0
    freshness_poor_hours: float = 72.0

    # -- Check scheduling ----------------------------------------------------
    check_interval_minutes: int = 15
    alert_throttle_minutes: int = 60
    alert_dedup_window_hours: int = 24

    # -- Prediction engine ---------------------------------------------------
    prediction_history_days: int = 90
    prediction_min_samples: int = 5
    staleness_pattern_window_days: int = 30

    # -- Worker pool ---------------------------------------------------------
    max_workers: int = 4
    pool_size: int = 5

    # -- Cache ---------------------------------------------------------------
    cache_ttl: int = 300

    # -- Rate limiting -------------------------------------------------------
    rate_limit: int = 100

    # -- Provenance ----------------------------------------------------------
    enable_provenance: bool = True

    # -- Feature toggles -----------------------------------------------------
    enable_predictions: bool = True
    enable_alerts: bool = True
    escalation_enabled: bool = True

    # -- Genesis hash --------------------------------------------------------
    genesis_hash: str = "greenlang-data-freshness-monitor-genesis"

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> DataFreshnessMonitorConfig:
        """Build a DataFreshnessMonitorConfig from environment variables.

        Every field can be overridden via ``GL_DFM_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).

        Returns:
            Populated DataFreshnessMonitorConfig instance.
        """
        env = EnvReader(_ENV_PREFIX)
        base_kwargs = cls._base_kwargs_from_env(env)

        config = cls(
            **base_kwargs,
            # Dataset processing
            batch_size=env.int("BATCH_SIZE", cls.batch_size),
            max_datasets=env.int("MAX_DATASETS", cls.max_datasets),
            # SLA thresholds
            default_sla_warning_hours=env.float(
                "DEFAULT_SLA_WARNING_HOURS",
                cls.default_sla_warning_hours,
            ),
            default_sla_critical_hours=env.float(
                "DEFAULT_SLA_CRITICAL_HOURS",
                cls.default_sla_critical_hours,
            ),
            # Freshness tiers
            freshness_excellent_hours=env.float(
                "FRESHNESS_EXCELLENT_HOURS",
                cls.freshness_excellent_hours,
            ),
            freshness_good_hours=env.float(
                "FRESHNESS_GOOD_HOURS",
                cls.freshness_good_hours,
            ),
            freshness_fair_hours=env.float(
                "FRESHNESS_FAIR_HOURS",
                cls.freshness_fair_hours,
            ),
            freshness_poor_hours=env.float(
                "FRESHNESS_POOR_HOURS",
                cls.freshness_poor_hours,
            ),
            # Check scheduling
            check_interval_minutes=env.int(
                "CHECK_INTERVAL_MINUTES",
                cls.check_interval_minutes,
            ),
            alert_throttle_minutes=env.int(
                "ALERT_THROTTLE_MINUTES",
                cls.alert_throttle_minutes,
            ),
            alert_dedup_window_hours=env.int(
                "ALERT_DEDUP_WINDOW_HOURS",
                cls.alert_dedup_window_hours,
            ),
            # Prediction engine
            prediction_history_days=env.int(
                "PREDICTION_HISTORY_DAYS",
                cls.prediction_history_days,
            ),
            prediction_min_samples=env.int(
                "PREDICTION_MIN_SAMPLES",
                cls.prediction_min_samples,
            ),
            staleness_pattern_window_days=env.int(
                "STALENESS_PATTERN_WINDOW_DAYS",
                cls.staleness_pattern_window_days,
            ),
            # Worker pool
            max_workers=env.int("MAX_WORKERS", cls.max_workers),
            pool_size=env.int("POOL_SIZE", cls.pool_size),
            # Cache
            cache_ttl=env.int("CACHE_TTL", cls.cache_ttl),
            # Rate limiting
            rate_limit=env.int("RATE_LIMIT", cls.rate_limit),
            # Provenance
            enable_provenance=env.bool(
                "ENABLE_PROVENANCE", cls.enable_provenance,
            ),
            # Feature toggles
            enable_predictions=env.bool(
                "ENABLE_PREDICTIONS", cls.enable_predictions,
            ),
            enable_alerts=env.bool(
                "ENABLE_ALERTS", cls.enable_alerts,
            ),
            escalation_enabled=env.bool(
                "ESCALATION_ENABLED", cls.escalation_enabled,
            ),
            # Genesis hash
            genesis_hash=env.str("GENESIS_HASH", cls.genesis_hash),
        )

        logger.info(
            "DataFreshnessMonitorConfig loaded: batch=%d, "
            "max_datasets=%d, "
            "sla_warning=%.1fh, sla_critical=%.1fh, "
            "freshness_tiers=[%.1f/%.1f/%.1f/%.1f]h, "
            "check_interval=%dmin, alert_throttle=%dmin, "
            "dedup_window=%dh, "
            "prediction_history=%dd, min_samples=%d, "
            "staleness_window=%dd, "
            "workers=%d, pool=%d, cache_ttl=%ds, "
            "rate=%d rpm, provenance=%s, "
            "predictions=%s, alerts=%s, escalation=%s",
            config.batch_size,
            config.max_datasets,
            config.default_sla_warning_hours,
            config.default_sla_critical_hours,
            config.freshness_excellent_hours,
            config.freshness_good_hours,
            config.freshness_fair_hours,
            config.freshness_poor_hours,
            config.check_interval_minutes,
            config.alert_throttle_minutes,
            config.alert_dedup_window_hours,
            config.prediction_history_days,
            config.prediction_min_samples,
            config.staleness_pattern_window_days,
            config.max_workers,
            config.pool_size,
            config.cache_ttl,
            config.rate_limit,
            config.enable_provenance,
            config.enable_predictions,
            config.enable_alerts,
            config.escalation_enabled,
        )
        return config

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate all configuration constraints after initialization.

        Raises:
            ValueError: If any constraint is violated.
        """
        errors: list[str] = []

        # Dataset processing
        if self.batch_size < 1:
            errors.append("batch_size must be >= 1")
        if self.max_datasets < 1:
            errors.append("max_datasets must be >= 1")
        if self.batch_size > self.max_datasets:
            errors.append("batch_size must be <= max_datasets")

        # SLA thresholds
        if self.default_sla_warning_hours <= 0.0:
            errors.append("default_sla_warning_hours must be > 0.0")
        if self.default_sla_critical_hours <= 0.0:
            errors.append("default_sla_critical_hours must be > 0.0")
        if self.default_sla_warning_hours >= self.default_sla_critical_hours:
            errors.append(
                "default_sla_warning_hours must be < "
                "default_sla_critical_hours"
            )

        # Freshness tiers (must be strictly ascending)
        if self.freshness_excellent_hours <= 0.0:
            errors.append("freshness_excellent_hours must be > 0.0")
        if self.freshness_good_hours <= 0.0:
            errors.append("freshness_good_hours must be > 0.0")
        if self.freshness_fair_hours <= 0.0:
            errors.append("freshness_fair_hours must be > 0.0")
        if self.freshness_poor_hours <= 0.0:
            errors.append("freshness_poor_hours must be > 0.0")
        if self.freshness_excellent_hours >= self.freshness_good_hours:
            errors.append(
                "freshness_excellent_hours must be < freshness_good_hours"
            )
        if self.freshness_good_hours >= self.freshness_fair_hours:
            errors.append(
                "freshness_good_hours must be < freshness_fair_hours"
            )
        if self.freshness_fair_hours >= self.freshness_poor_hours:
            errors.append(
                "freshness_fair_hours must be < freshness_poor_hours"
            )

        # Check scheduling
        if self.check_interval_minutes < 1:
            errors.append("check_interval_minutes must be >= 1")
        if self.alert_throttle_minutes < 1:
            errors.append("alert_throttle_minutes must be >= 1")
        if self.alert_dedup_window_hours < 1:
            errors.append("alert_dedup_window_hours must be >= 1")

        # Prediction engine
        if self.prediction_history_days < 1:
            errors.append("prediction_history_days must be >= 1")
        if self.prediction_min_samples < 1:
            errors.append("prediction_min_samples must be >= 1")
        if self.staleness_pattern_window_days < 1:
            errors.append("staleness_pattern_window_days must be >= 1")
        if self.staleness_pattern_window_days > self.prediction_history_days:
            errors.append(
                "staleness_pattern_window_days must be <= "
                "prediction_history_days"
            )

        # Worker pool
        if self.max_workers < 1:
            errors.append("max_workers must be >= 1")
        if self.pool_size < 1:
            errors.append("pool_size must be >= 1")

        # Cache
        if self.cache_ttl < 0:
            errors.append("cache_ttl must be >= 0")

        # Rate limiting
        if self.rate_limit < 1:
            errors.append("rate_limit must be >= 1")

        # Log level
        valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        if self.log_level.upper() not in valid_levels:
            errors.append(
                f"log_level must be one of {valid_levels}, "
                f"got '{self.log_level}'"
            )

        # Genesis hash
        if not self.genesis_hash:
            errors.append("genesis_hash must not be empty")

        if errors:
            msg = "; ".join(errors)
            logger.error(
                "DataFreshnessMonitorConfig validation failed: %s", msg,
            )
            raise ValueError(
                f"DataFreshnessMonitorConfig validation failed: {msg}"
            )

        logger.debug("DataFreshnessMonitorConfig validated successfully")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the configuration to a plain dictionary.

        Returns:
            Dictionary with all configuration fields and their current
            values. Useful for logging, debugging, and exporting
            configuration state to JSON or YAML.
        """
        return {
            "database_url": self.database_url,
            "redis_url": self.redis_url,
            "log_level": self.log_level,
            "batch_size": self.batch_size,
            "max_datasets": self.max_datasets,
            "default_sla_warning_hours": self.default_sla_warning_hours,
            "default_sla_critical_hours": self.default_sla_critical_hours,
            "freshness_excellent_hours": self.freshness_excellent_hours,
            "freshness_good_hours": self.freshness_good_hours,
            "freshness_fair_hours": self.freshness_fair_hours,
            "freshness_poor_hours": self.freshness_poor_hours,
            "check_interval_minutes": self.check_interval_minutes,
            "alert_throttle_minutes": self.alert_throttle_minutes,
            "alert_dedup_window_hours": self.alert_dedup_window_hours,
            "prediction_history_days": self.prediction_history_days,
            "prediction_min_samples": self.prediction_min_samples,
            "staleness_pattern_window_days": self.staleness_pattern_window_days,
            "max_workers": self.max_workers,
            "pool_size": self.pool_size,
            "cache_ttl": self.cache_ttl,
            "rate_limit": self.rate_limit,
            "enable_provenance": self.enable_provenance,
            "enable_predictions": self.enable_predictions,
            "enable_alerts": self.enable_alerts,
            "escalation_enabled": self.escalation_enabled,
            "genesis_hash": self.genesis_hash,
        }


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

get_config, set_config, reset_config = create_config_singleton(
    DataFreshnessMonitorConfig, _ENV_PREFIX,
)

__all__ = [
    "DataFreshnessMonitorConfig",
    "get_config",
    "set_config",
    "reset_config",
]
