# -*- coding: utf-8 -*-
"""
Schema Migration Agent Service Configuration - AGENT-DATA-017

Centralized configuration for the Schema Migration Agent SDK covering:
- Database, cache, and connection defaults
- Schema capacity limits (max schemas, max versions per schema)
- Migration execution settings (batch size, timeout, dry-run, auto-rollback)
- Schema compatibility enforcement (backward, forward, full, none)
- Drift detection parameters (check interval, sample size)
- Provenance tracking (genesis hash, SHA-256 chain anchoring)
- Field mapping confidence thresholds and deprecation policy
- Impact analysis depth limits and change traversal
- Connection pool sizing, rate limiting, and cache TTL
- Retry and checkpoint strategies for long-running migrations
- Processing limits and logging level

All settings can be overridden via environment variables with the
``GL_SM_`` prefix (e.g. ``GL_SM_MAX_SCHEMAS``, ``GL_SM_MIGRATION_TIMEOUT_SECONDS``).

Environment Variable Reference (GL_SM_ prefix):
    GL_SM_DATABASE_URL               - PostgreSQL connection URL
    GL_SM_REDIS_URL                  - Redis connection URL
    GL_SM_LOG_LEVEL                  - Logging level (DEBUG/INFO/WARNING/ERROR)
    GL_SM_MAX_SCHEMAS                - Maximum schemas managed by the service
    GL_SM_MAX_VERSIONS_PER_SCHEMA    - Maximum versions tracked per schema
    GL_SM_MAX_MIGRATION_BATCH_SIZE   - Maximum records per migration batch
    GL_SM_MIGRATION_TIMEOUT_SECONDS  - Timeout (s) for a single migration run
    GL_SM_ENABLE_DRY_RUN             - Enable dry-run mode (true/false)
    GL_SM_ENABLE_AUTO_ROLLBACK       - Enable automatic rollback on failure
    GL_SM_COMPATIBILITY_DEFAULT_LEVEL- Default schema compatibility level
    GL_SM_DRIFT_CHECK_INTERVAL_MINUTES - Interval (min) between drift checks
    GL_SM_DRIFT_SAMPLE_SIZE          - Sample size for drift detection analysis
    GL_SM_ENABLE_PROVENANCE          - Enable SHA-256 provenance chain tracking
    GL_SM_GENESIS_HASH               - Genesis anchor string for provenance chain
    GL_SM_MAX_WORKERS                - Thread/worker pool size for migrations
    GL_SM_POOL_SIZE                  - Database connection pool size
    GL_SM_CACHE_TTL                  - Cache time-to-live in seconds
    GL_SM_RATE_LIMIT                 - Max API requests per minute
    GL_SM_CHECKPOINT_INTERVAL        - Records between migration checkpoints
    GL_SM_RETRY_MAX_ATTEMPTS         - Maximum retry attempts on transient failure
    GL_SM_RETRY_BACKOFF_BASE         - Exponential backoff base multiplier (s)
    GL_SM_FIELD_MAPPING_MIN_CONFIDENCE - Min confidence for field mapping (0-1)
    GL_SM_DEPRECATION_WARNING_DAYS   - Days before removal to warn on deprecated
    GL_SM_MAX_CHANGE_DEPTH           - Maximum dependency traversal depth
    GL_SM_ENABLE_IMPACT_ANALYSIS     - Enable downstream impact analysis

Example:
    >>> from greenlang.schema_migration.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.max_schemas, cfg.compatibility_default_level)
    50000 backward

    >>> # Override for testing
    >>> from greenlang.schema_migration.config import set_config, reset_config
    >>> from greenlang.schema_migration.config import SchemaMigrationConfig
    >>> set_config(SchemaMigrationConfig(max_schemas=100, enable_dry_run=False))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-017 Schema Migration Agent (GL-DATA-X-020)
Status: Production Ready
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_SM_"

# ---------------------------------------------------------------------------
# Valid compatibility levels accepted by the service
# ---------------------------------------------------------------------------

_VALID_COMPATIBILITY_LEVELS = frozenset(
    {"backward", "forward", "full", "none", "backward_transitive",
     "forward_transitive", "full_transitive"}
)

# ---------------------------------------------------------------------------
# Valid log levels
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)


# ---------------------------------------------------------------------------
# SchemaMigrationConfig
# ---------------------------------------------------------------------------


@dataclass
class SchemaMigrationConfig:
    """Complete configuration for the GreenLang Schema Migration Agent SDK.

    Attributes are grouped by concern: connections, schema capacity,
    migration execution, compatibility enforcement, drift detection,
    provenance tracking, field mapping, impact analysis, performance
    tuning, retry/checkpoint strategy, and logging.

    All attributes can be overridden via environment variables using the
    ``GL_SM_`` prefix (e.g. ``GL_SM_MAX_SCHEMAS=100000``).

    Compatibility levels follow Apache Avro/Confluent Schema Registry
    semantics:
    - ``backward``            : New schema can read data written with old schema.
    - ``forward``             : Old schema can read data written with new schema.
    - ``full``                : Both backward and forward compatible.
    - ``none``                : No compatibility check enforced.
    - ``backward_transitive`` : Backward-compatible with ALL prior versions.
    - ``forward_transitive``  : Forward-compatible with ALL prior versions.
    - ``full_transitive``     : Full compatibility with ALL prior versions.

    Attributes:
        database_url: PostgreSQL connection URL for persistent schema storage.
        redis_url: Redis connection URL for caching schema lookups and locks.
        log_level: Logging verbosity level for the schema migration service.
        max_schemas: Maximum number of distinct schema subjects managed.
        max_versions_per_schema: Maximum version history retained per schema.
        max_migration_batch_size: Maximum number of records per migration batch.
        migration_timeout_seconds: Wall-clock timeout (s) for a migration run.
        enable_dry_run: When True, migrations are validated but not committed.
        enable_auto_rollback: When True, failed migrations auto-revert to prior.
        compatibility_default_level: Default compatibility rule for new subjects.
        drift_check_interval_minutes: Interval between scheduled drift checks.
        drift_sample_size: Row sample size used during schema drift analysis.
        enable_provenance: Whether to compute and store SHA-256 provenance hashes.
        genesis_hash: Anchor string used as the root of every provenance chain.
        max_workers: Parallel worker threads for concurrent migration tasks.
        pool_size: PostgreSQL connection pool size for the migration service.
        cache_ttl: Time-to-live (s) for cached schema and migration records.
        rate_limit: Maximum inbound API requests allowed per minute.
        checkpoint_interval: Records processed between durable checkpoints.
        retry_max_attempts: Maximum retry attempts on transient failures.
        retry_backoff_base: Base multiplier (s) for exponential retry backoff.
        field_mapping_min_confidence: Minimum acceptable confidence for field
            mapping suggestions (0.0 to 1.0).
        deprecation_warning_days: Days before scheduled removal to emit
            deprecation warnings for schema fields.
        max_change_depth: Maximum graph traversal depth for dependency analysis.
        enable_impact_analysis: Whether to compute downstream impact of changes.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = ""
    redis_url: str = ""

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Schema capacity -----------------------------------------------------
    max_schemas: int = 50_000
    max_versions_per_schema: int = 1_000

    # -- Migration execution -------------------------------------------------
    max_migration_batch_size: int = 10_000
    migration_timeout_seconds: int = 3_600
    enable_dry_run: bool = True
    enable_auto_rollback: bool = True

    # -- Compatibility enforcement -------------------------------------------
    compatibility_default_level: str = "backward"

    # -- Drift detection -----------------------------------------------------
    drift_check_interval_minutes: int = 60
    drift_sample_size: int = 1_000

    # -- Provenance tracking -------------------------------------------------
    enable_provenance: bool = True
    genesis_hash: str = "greenlang-schema-migration-genesis"

    # -- Performance tuning --------------------------------------------------
    max_workers: int = 4
    pool_size: int = 5
    cache_ttl: int = 300
    rate_limit: int = 100

    # -- Retry / checkpoint strategy -----------------------------------------
    checkpoint_interval: int = 100
    retry_max_attempts: int = 3
    retry_backoff_base: float = 2.0

    # -- Field mapping -------------------------------------------------------
    field_mapping_min_confidence: float = 0.8

    # -- Deprecation policy --------------------------------------------------
    deprecation_warning_days: int = 30

    # -- Impact analysis -----------------------------------------------------
    max_change_depth: int = 10
    enable_impact_analysis: bool = True

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate configuration constraints after initialisation.

        Raises:
            ValueError: If any configuration value is outside its valid range
                or uses an unsupported enumerated value.
        """
        errors: list[str] = []

        # -- Connections -----------------------------------------------------
        # database_url and redis_url are allowed to be empty at construction
        # time (they may be injected at runtime by the service mesh), so we
        # only emit a WARNING rather than raising.
        if not self.database_url:
            logger.warning(
                "SchemaMigrationConfig: database_url is empty; "
                "the service will fail to connect until GL_SM_DATABASE_URL is set."
            )
        if not self.redis_url:
            logger.warning(
                "SchemaMigrationConfig: redis_url is empty; "
                "caching and distributed locks are disabled."
            )

        # -- Logging ---------------------------------------------------------
        normalised_log = self.log_level.upper()
        if normalised_log not in _VALID_LOG_LEVELS:
            errors.append(
                f"log_level must be one of {sorted(_VALID_LOG_LEVELS)}, "
                f"got '{self.log_level}'"
            )
        else:
            self.log_level = normalised_log

        # -- Schema capacity -------------------------------------------------
        if self.max_schemas <= 0:
            errors.append(
                f"max_schemas must be > 0, got {self.max_schemas}"
            )
        if self.max_versions_per_schema <= 0:
            errors.append(
                f"max_versions_per_schema must be > 0, "
                f"got {self.max_versions_per_schema}"
            )

        # -- Migration execution ---------------------------------------------
        if self.max_migration_batch_size <= 0:
            errors.append(
                f"max_migration_batch_size must be > 0, "
                f"got {self.max_migration_batch_size}"
            )
        if self.migration_timeout_seconds <= 0:
            errors.append(
                f"migration_timeout_seconds must be > 0, "
                f"got {self.migration_timeout_seconds}"
            )

        # -- Compatibility ---------------------------------------------------
        if self.compatibility_default_level not in _VALID_COMPATIBILITY_LEVELS:
            errors.append(
                f"compatibility_default_level must be one of "
                f"{sorted(_VALID_COMPATIBILITY_LEVELS)}, "
                f"got '{self.compatibility_default_level}'"
            )

        # -- Drift detection -------------------------------------------------
        if self.drift_check_interval_minutes <= 0:
            errors.append(
                f"drift_check_interval_minutes must be > 0, "
                f"got {self.drift_check_interval_minutes}"
            )
        if self.drift_sample_size <= 0:
            errors.append(
                f"drift_sample_size must be > 0, "
                f"got {self.drift_sample_size}"
            )

        # -- Provenance ------------------------------------------------------
        if not self.genesis_hash:
            errors.append("genesis_hash must not be empty")

        # -- Performance -----------------------------------------------------
        if self.max_workers <= 0:
            errors.append(
                f"max_workers must be > 0, got {self.max_workers}"
            )
        if self.pool_size <= 0:
            errors.append(
                f"pool_size must be > 0, got {self.pool_size}"
            )
        if self.cache_ttl <= 0:
            errors.append(
                f"cache_ttl must be > 0, got {self.cache_ttl}"
            )
        if self.rate_limit <= 0:
            errors.append(
                f"rate_limit must be > 0, got {self.rate_limit}"
            )

        # -- Retry / checkpoint ----------------------------------------------
        if self.checkpoint_interval <= 0:
            errors.append(
                f"checkpoint_interval must be > 0, "
                f"got {self.checkpoint_interval}"
            )
        if self.retry_max_attempts <= 0:
            errors.append(
                f"retry_max_attempts must be > 0, "
                f"got {self.retry_max_attempts}"
            )
        if self.retry_backoff_base <= 0.0:
            errors.append(
                f"retry_backoff_base must be > 0.0, "
                f"got {self.retry_backoff_base}"
            )

        # -- Field mapping ---------------------------------------------------
        if not (0.0 <= self.field_mapping_min_confidence <= 1.0):
            errors.append(
                f"field_mapping_min_confidence must be in [0.0, 1.0], "
                f"got {self.field_mapping_min_confidence}"
            )

        # -- Deprecation policy ----------------------------------------------
        if self.deprecation_warning_days < 0:
            errors.append(
                f"deprecation_warning_days must be >= 0, "
                f"got {self.deprecation_warning_days}"
            )

        # -- Impact analysis -------------------------------------------------
        if self.max_change_depth <= 0:
            errors.append(
                f"max_change_depth must be > 0, got {self.max_change_depth}"
            )

        if errors:
            raise ValueError(
                "SchemaMigrationConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "SchemaMigrationConfig validated successfully: "
            "max_schemas=%d, compatibility=%s, dry_run=%s, "
            "auto_rollback=%s, provenance=%s",
            self.max_schemas,
            self.compatibility_default_level,
            self.enable_dry_run,
            self.enable_auto_rollback,
            self.enable_provenance,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> SchemaMigrationConfig:
        """Build a SchemaMigrationConfig from environment variables.

        Every field can be overridden via ``GL_SM_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.
        Unknown / malformed values fall back to the class-level default and
        emit a WARNING log so the issue is visible in deployment logs.

        Returns:
            Populated SchemaMigrationConfig instance, validated via
            ``__post_init__``.

        Example:
            >>> import os
            >>> os.environ["GL_SM_MAX_SCHEMAS"] = "100000"
            >>> cfg = SchemaMigrationConfig.from_env()
            >>> cfg.max_schemas
            100000
        """
        prefix = _ENV_PREFIX

        def _env(name: str, default: Any = None) -> Optional[str]:
            return os.environ.get(f"{prefix}{name}", default)

        def _bool(name: str, default: bool) -> bool:
            val = _env(name)
            if val is None:
                return default
            return val.strip().lower() in ("true", "1", "yes")

        def _int(name: str, default: int) -> int:
            val = _env(name)
            if val is None:
                return default
            try:
                return int(val.strip())
            except ValueError:
                logger.warning(
                    "Invalid integer for %s%s=%r, using default %d",
                    prefix,
                    name,
                    val,
                    default,
                )
                return default

        def _float(name: str, default: float) -> float:
            val = _env(name)
            if val is None:
                return default
            try:
                return float(val.strip())
            except ValueError:
                logger.warning(
                    "Invalid float for %s%s=%r, using default %f",
                    prefix,
                    name,
                    val,
                    default,
                )
                return default

        def _str(name: str, default: str) -> str:
            val = _env(name)
            if val is None:
                return default
            return val.strip()

        config = cls(
            # Connections
            database_url=_str("DATABASE_URL", cls.database_url),
            redis_url=_str("REDIS_URL", cls.redis_url),
            # Logging
            log_level=_str("LOG_LEVEL", cls.log_level),
            # Schema capacity
            max_schemas=_int("MAX_SCHEMAS", cls.max_schemas),
            max_versions_per_schema=_int(
                "MAX_VERSIONS_PER_SCHEMA",
                cls.max_versions_per_schema,
            ),
            # Migration execution
            max_migration_batch_size=_int(
                "MAX_MIGRATION_BATCH_SIZE",
                cls.max_migration_batch_size,
            ),
            migration_timeout_seconds=_int(
                "MIGRATION_TIMEOUT_SECONDS",
                cls.migration_timeout_seconds,
            ),
            enable_dry_run=_bool("ENABLE_DRY_RUN", cls.enable_dry_run),
            enable_auto_rollback=_bool(
                "ENABLE_AUTO_ROLLBACK",
                cls.enable_auto_rollback,
            ),
            # Compatibility enforcement
            compatibility_default_level=_str(
                "COMPATIBILITY_DEFAULT_LEVEL",
                cls.compatibility_default_level,
            ),
            # Drift detection
            drift_check_interval_minutes=_int(
                "DRIFT_CHECK_INTERVAL_MINUTES",
                cls.drift_check_interval_minutes,
            ),
            drift_sample_size=_int(
                "DRIFT_SAMPLE_SIZE",
                cls.drift_sample_size,
            ),
            # Provenance tracking
            enable_provenance=_bool("ENABLE_PROVENANCE", cls.enable_provenance),
            genesis_hash=_str("GENESIS_HASH", cls.genesis_hash),
            # Performance tuning
            max_workers=_int("MAX_WORKERS", cls.max_workers),
            pool_size=_int("POOL_SIZE", cls.pool_size),
            cache_ttl=_int("CACHE_TTL", cls.cache_ttl),
            rate_limit=_int("RATE_LIMIT", cls.rate_limit),
            # Retry / checkpoint strategy
            checkpoint_interval=_int(
                "CHECKPOINT_INTERVAL",
                cls.checkpoint_interval,
            ),
            retry_max_attempts=_int(
                "RETRY_MAX_ATTEMPTS",
                cls.retry_max_attempts,
            ),
            retry_backoff_base=_float(
                "RETRY_BACKOFF_BASE",
                cls.retry_backoff_base,
            ),
            # Field mapping
            field_mapping_min_confidence=_float(
                "FIELD_MAPPING_MIN_CONFIDENCE",
                cls.field_mapping_min_confidence,
            ),
            # Deprecation policy
            deprecation_warning_days=_int(
                "DEPRECATION_WARNING_DAYS",
                cls.deprecation_warning_days,
            ),
            # Impact analysis
            max_change_depth=_int("MAX_CHANGE_DEPTH", cls.max_change_depth),
            enable_impact_analysis=_bool(
                "ENABLE_IMPACT_ANALYSIS",
                cls.enable_impact_analysis,
            ),
        )

        logger.info(
            "SchemaMigrationConfig loaded: "
            "max_schemas=%d, max_versions=%d, "
            "max_batch=%d, timeout=%ds, "
            "dry_run=%s, auto_rollback=%s, "
            "compatibility=%s, "
            "drift_interval=%dmin, drift_sample=%d, "
            "provenance=%s, "
            "workers=%d, pool=%d, cache_ttl=%ds, rate_limit=%d/min, "
            "checkpoint=%d, retry=%d, backoff_base=%.1f, "
            "field_confidence=%.2f, deprecation_days=%d, "
            "max_depth=%d, impact_analysis=%s",
            config.max_schemas,
            config.max_versions_per_schema,
            config.max_migration_batch_size,
            config.migration_timeout_seconds,
            config.enable_dry_run,
            config.enable_auto_rollback,
            config.compatibility_default_level,
            config.drift_check_interval_minutes,
            config.drift_sample_size,
            config.enable_provenance,
            config.max_workers,
            config.pool_size,
            config.cache_ttl,
            config.rate_limit,
            config.checkpoint_interval,
            config.retry_max_attempts,
            config.retry_backoff_base,
            config.field_mapping_min_confidence,
            config.deprecation_warning_days,
            config.max_change_depth,
            config.enable_impact_analysis,
        )
        return config

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the configuration to a plain Python dictionary.

        The returned dictionary is safe to pass to ``json.dumps``,
        ``yaml.dump``, or any structured logging framework.  All values
        are JSON-serialisable primitives (str, int, float, bool).

        Sensitive connection strings (``database_url``, ``redis_url``) are
        redacted to prevent accidental credential leakage in logs.

        Returns:
            Dictionary representation of the configuration with sensitive
            fields redacted.

        Example:
            >>> cfg = SchemaMigrationConfig()
            >>> d = cfg.to_dict()
            >>> d["compatibility_default_level"]
            'backward'
            >>> d["database_url"]  # redacted
            '***'
        """
        return {
            # -- Connections (redacted) --------------------------------------
            "database_url": "***" if self.database_url else "",
            "redis_url": "***" if self.redis_url else "",
            # -- Logging -----------------------------------------------------
            "log_level": self.log_level,
            # -- Schema capacity ---------------------------------------------
            "max_schemas": self.max_schemas,
            "max_versions_per_schema": self.max_versions_per_schema,
            # -- Migration execution -----------------------------------------
            "max_migration_batch_size": self.max_migration_batch_size,
            "migration_timeout_seconds": self.migration_timeout_seconds,
            "enable_dry_run": self.enable_dry_run,
            "enable_auto_rollback": self.enable_auto_rollback,
            # -- Compatibility enforcement ------------------------------------
            "compatibility_default_level": self.compatibility_default_level,
            # -- Drift detection ---------------------------------------------
            "drift_check_interval_minutes": self.drift_check_interval_minutes,
            "drift_sample_size": self.drift_sample_size,
            # -- Provenance tracking -----------------------------------------
            "enable_provenance": self.enable_provenance,
            "genesis_hash": self.genesis_hash,
            # -- Performance tuning ------------------------------------------
            "max_workers": self.max_workers,
            "pool_size": self.pool_size,
            "cache_ttl": self.cache_ttl,
            "rate_limit": self.rate_limit,
            # -- Retry / checkpoint strategy ---------------------------------
            "checkpoint_interval": self.checkpoint_interval,
            "retry_max_attempts": self.retry_max_attempts,
            "retry_backoff_base": self.retry_backoff_base,
            # -- Field mapping -----------------------------------------------
            "field_mapping_min_confidence": self.field_mapping_min_confidence,
            # -- Deprecation policy ------------------------------------------
            "deprecation_warning_days": self.deprecation_warning_days,
            # -- Impact analysis ---------------------------------------------
            "max_change_depth": self.max_change_depth,
            "enable_impact_analysis": self.enable_impact_analysis,
        }

    def __repr__(self) -> str:
        """Return a developer-friendly, credential-safe representation.

        Sensitive fields (database_url, redis_url) are replaced with
        ``'***'`` so that repr output is safe to include in log messages
        and exception tracebacks.

        Returns:
            String representation of the configuration.
        """
        d = self.to_dict()
        pairs = ", ".join(f"{k}={v!r}" for k, v in d.items())
        return f"SchemaMigrationConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[SchemaMigrationConfig] = None
_config_lock = threading.Lock()


def get_config() -> SchemaMigrationConfig:
    """Return the singleton SchemaMigrationConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path.  The instance is created on first call
    by reading all ``GL_SM_*`` environment variables via
    :meth:`SchemaMigrationConfig.from_env`.

    Returns:
        SchemaMigrationConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.max_schemas
        50000
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = SchemaMigrationConfig.from_env()
    return _config_instance


def set_config(config: SchemaMigrationConfig) -> None:
    """Replace the singleton SchemaMigrationConfig.

    Primarily intended for testing and dependency injection scenarios
    where a custom configuration must be supplied without relying on
    environment variables.

    Args:
        config: New :class:`SchemaMigrationConfig` to install as the singleton.

    Example:
        >>> cfg = SchemaMigrationConfig(max_schemas=100, enable_dry_run=False)
        >>> set_config(cfg)
        >>> assert get_config().max_schemas == 100
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info(
        "SchemaMigrationConfig replaced programmatically: "
        "max_schemas=%d, compatibility=%s",
        config.max_schemas,
        config.compatibility_default_level,
    )


def reset_config() -> None:
    """Reset the singleton SchemaMigrationConfig to ``None``.

    The next call to :func:`get_config` will re-read environment variables
    and construct a fresh instance.  Intended for test teardown to prevent
    state leakage between test cases.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads GL_SM_* env vars
    """
    global _config_instance
    with _config_lock:
        _config_instance = None
    logger.debug("SchemaMigrationConfig singleton reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "SchemaMigrationConfig",
    "get_config",
    "set_config",
    "reset_config",
]
