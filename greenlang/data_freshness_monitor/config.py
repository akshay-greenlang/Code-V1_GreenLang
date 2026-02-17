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
    >>> from greenlang.data_freshness_monitor.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_sla_warning_hours, cfg.default_sla_critical_hours)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor (GL-DATA-X-019)
Status: Production Ready
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_DFM_"


# ---------------------------------------------------------------------------
# DataFreshnessMonitorConfig
# ---------------------------------------------------------------------------


@dataclass
class DataFreshnessMonitorConfig:
    """Complete configuration for the GreenLang Data Freshness Monitor SDK.

    Attributes are grouped by concern: connections, logging, dataset
    processing, SLA thresholds, freshness tiers, check scheduling,
    alert throttling, prediction engine, staleness detection, worker
    pool, cache, rate limiting, provenance, feature toggles, and
    genesis hash.

    All attributes can be overridden via environment variables using the
    ``GL_DFM_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage.
            Holds dataset registrations, freshness check results, SLA
            violation records, prediction models, and provenance chains.
        redis_url: Redis connection URL for caching layer.
            Used for freshness check result caching, rate limiting
            counters, and alert deduplication state.
        log_level: Logging level for the data freshness monitor service.
            Accepts standard Python logging levels: DEBUG, INFO,
            WARNING, ERROR, CRITICAL.
        batch_size: Default batch size for dataset processing.
            Controls how many datasets are checked in a single
            chunk during batch freshness check operations.
        max_datasets: Maximum datasets allowed to be monitored
            concurrently. Prevents runaway resource consumption
            from excessively large monitoring scopes.
        default_sla_warning_hours: Hours after last update before a
            dataset triggers a warning-level SLA alert. Indicates
            the dataset is approaching staleness but is not yet
            critically overdue.
        default_sla_critical_hours: Hours after last update before a
            dataset triggers a critical-level SLA alert. Indicates
            the dataset has exceeded its acceptable freshness window
            and requires immediate attention. Must be greater than
            default_sla_warning_hours.
        freshness_excellent_hours: Upper bound (exclusive) in hours
            for a dataset to be classified as having excellent
            freshness. Datasets updated within this window receive
            the highest freshness tier score.
        freshness_good_hours: Upper bound (exclusive) in hours for
            a dataset to be classified as having good freshness.
            Must be greater than freshness_excellent_hours.
        freshness_fair_hours: Upper bound (exclusive) in hours for
            a dataset to be classified as having fair freshness.
            Must be greater than freshness_good_hours.
        freshness_poor_hours: Upper bound (exclusive) in hours for
            a dataset to be classified as having poor freshness.
            Datasets exceeding this window are classified as stale.
            Must be greater than freshness_fair_hours.
        check_interval_minutes: Interval in minutes between scheduled
            freshness checks for monitored datasets. Controls the
            polling frequency of the monitoring loop.
        alert_throttle_minutes: Minimum minutes between consecutive
            alerts for the same dataset. Prevents alert fatigue by
            suppressing duplicate notifications within this window.
        alert_dedup_window_hours: Hours during which duplicate alerts
            for the same dataset and severity are suppressed. Provides
            a broader deduplication window than alert_throttle_minutes.
        prediction_history_days: Number of historical days of freshness
            data used by the prediction engine to forecast future
            staleness events.
        prediction_min_samples: Minimum number of historical freshness
            observations required before the prediction engine will
            generate a forecast. Prevents unreliable predictions from
            insufficient data.
        staleness_pattern_window_days: Number of days in the sliding
            window used for staleness pattern detection. The engine
            analyzes update frequency patterns within this window to
            identify recurring staleness issues.
        max_workers: Number of parallel workers for batch processing.
            Controls concurrency for multi-dataset freshness checks.
        pool_size: Connection pool size for database connections.
            Controls the number of concurrent database connections
            available for freshness monitoring operations.
        cache_ttl: Cache time-to-live in seconds for freshness check
            results. Prevents redundant re-computation when the same
            dataset is queried within the TTL window.
        rate_limit: Rate limit in requests per minute for the
            freshness monitoring API. Protects backend resources from
            excessive concurrent monitoring requests.
        enable_provenance: Whether SHA-256 provenance tracking is
            enabled. When True, every freshness check operation records
            a provenance chain including input hashes, check results,
            SLA evaluations, and output hashes for full auditability.
        enable_predictions: Whether the predictive staleness engine
            is enabled. When True, the monitor uses historical update
            patterns to forecast when datasets are likely to become
            stale and generates proactive alerts.
        enable_alerts: Whether alert generation is enabled. When True,
            SLA violations and staleness events trigger notifications
            through configured alert channels.
        escalation_enabled: Whether automatic alert escalation is
            enabled. When True, unresolved alerts are escalated to
            higher-priority channels after configurable time windows.
        genesis_hash: Seed string used to initialize the SHA-256
            provenance chain for this agent. Provides a deterministic
            starting point for the audit trail.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = ""
    redis_url: str = ""

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

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
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> DataFreshnessMonitorConfig:
        """Build a DataFreshnessMonitorConfig from environment variables.

        Every field can be overridden via ``GL_DFM_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.

        Returns:
            Populated DataFreshnessMonitorConfig instance.
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

        def _float(name: str, default: float) -> float:
            val = _env(name)
            if val is None:
                return default
            try:
                return float(val)
            except ValueError:
                logger.warning(
                    "Invalid float for %s%s=%s, using default %f",
                    prefix, name, val, default,
                )
                return default

        def _str(name: str, default: str) -> str:
            val = _env(name)
            if val is None:
                return default
            return val

        config = cls(
            # Connections
            database_url=_str("DATABASE_URL", cls.database_url),
            redis_url=_str("REDIS_URL", cls.redis_url),
            # Logging
            log_level=_str("LOG_LEVEL", cls.log_level),
            # Dataset processing
            batch_size=_int("BATCH_SIZE", cls.batch_size),
            max_datasets=_int("MAX_DATASETS", cls.max_datasets),
            # SLA thresholds
            default_sla_warning_hours=_float(
                "DEFAULT_SLA_WARNING_HOURS",
                cls.default_sla_warning_hours,
            ),
            default_sla_critical_hours=_float(
                "DEFAULT_SLA_CRITICAL_HOURS",
                cls.default_sla_critical_hours,
            ),
            # Freshness tiers
            freshness_excellent_hours=_float(
                "FRESHNESS_EXCELLENT_HOURS",
                cls.freshness_excellent_hours,
            ),
            freshness_good_hours=_float(
                "FRESHNESS_GOOD_HOURS",
                cls.freshness_good_hours,
            ),
            freshness_fair_hours=_float(
                "FRESHNESS_FAIR_HOURS",
                cls.freshness_fair_hours,
            ),
            freshness_poor_hours=_float(
                "FRESHNESS_POOR_HOURS",
                cls.freshness_poor_hours,
            ),
            # Check scheduling
            check_interval_minutes=_int(
                "CHECK_INTERVAL_MINUTES",
                cls.check_interval_minutes,
            ),
            alert_throttle_minutes=_int(
                "ALERT_THROTTLE_MINUTES",
                cls.alert_throttle_minutes,
            ),
            alert_dedup_window_hours=_int(
                "ALERT_DEDUP_WINDOW_HOURS",
                cls.alert_dedup_window_hours,
            ),
            # Prediction engine
            prediction_history_days=_int(
                "PREDICTION_HISTORY_DAYS",
                cls.prediction_history_days,
            ),
            prediction_min_samples=_int(
                "PREDICTION_MIN_SAMPLES",
                cls.prediction_min_samples,
            ),
            staleness_pattern_window_days=_int(
                "STALENESS_PATTERN_WINDOW_DAYS",
                cls.staleness_pattern_window_days,
            ),
            # Worker pool
            max_workers=_int("MAX_WORKERS", cls.max_workers),
            pool_size=_int("POOL_SIZE", cls.pool_size),
            # Cache
            cache_ttl=_int("CACHE_TTL", cls.cache_ttl),
            # Rate limiting
            rate_limit=_int("RATE_LIMIT", cls.rate_limit),
            # Provenance
            enable_provenance=_bool(
                "ENABLE_PROVENANCE", cls.enable_provenance,
            ),
            # Feature toggles
            enable_predictions=_bool(
                "ENABLE_PREDICTIONS", cls.enable_predictions,
            ),
            enable_alerts=_bool(
                "ENABLE_ALERTS", cls.enable_alerts,
            ),
            escalation_enabled=_bool(
                "ESCALATION_ENABLED", cls.escalation_enabled,
            ),
            # Genesis hash
            genesis_hash=_str("GENESIS_HASH", cls.genesis_hash),
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

_config_instance: Optional[DataFreshnessMonitorConfig] = None
_config_lock = threading.Lock()


def get_config() -> DataFreshnessMonitorConfig:
    """Return the singleton DataFreshnessMonitorConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path.

    Returns:
        DataFreshnessMonitorConfig singleton instance.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = DataFreshnessMonitorConfig.from_env()
    return _config_instance


def set_config(config: DataFreshnessMonitorConfig) -> None:
    """Replace the singleton DataFreshnessMonitorConfig (useful for testing).

    Args:
        config: New configuration to install.
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info("DataFreshnessMonitorConfig replaced programmatically")


def reset_config() -> None:
    """Reset the singleton (primarily for test teardown)."""
    global _config_instance
    with _config_lock:
        _config_instance = None


__all__ = [
    "DataFreshnessMonitorConfig",
    "get_config",
    "set_config",
    "reset_config",
]
