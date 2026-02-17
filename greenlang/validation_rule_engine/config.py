# -*- coding: utf-8 -*-
"""
Validation Rule Engine Service Configuration - AGENT-DATA-019

Centralized configuration for the Validation Rule Engine Agent SDK covering:
- Database, cache, and connection defaults
- Rule and rule-set capacity limits (max rules, rule sets, rules per set)
- Compound rule nesting depth (maximum depth for nested compound rules)
- Evaluation thresholds (pass and warn thresholds for validation scoring)
- Evaluation timeout (maximum seconds per evaluation run)
- Batch processing (batch size and max datasets per batch evaluation)
- Provenance tracking (genesis hash, SHA-256 chain anchoring)
- Connection pool sizing, rate limiting, and cache TTL
- Conflict detection (automatic detection of contradictory rules)
- Short-circuit evaluation (early termination on rule-set failure)
- Processing limits (maximum rows per evaluation, report retention)
- Prometheus metrics export toggle

All settings can be overridden via environment variables with the
``GL_VRE_`` prefix (e.g. ``GL_VRE_MAX_RULES``, ``GL_VRE_BATCH_SIZE``).

Environment Variable Reference (GL_VRE_ prefix):
    GL_VRE_DATABASE_URL              - PostgreSQL connection URL
    GL_VRE_REDIS_URL                 - Redis connection URL
    GL_VRE_LOG_LEVEL                 - Logging level (DEBUG/INFO/WARNING/ERROR)
    GL_VRE_MAX_RULES                 - Maximum validation rules in the registry
    GL_VRE_MAX_RULE_SETS             - Maximum rule sets in the registry
    GL_VRE_MAX_RULES_PER_SET         - Maximum rules within a single rule set
    GL_VRE_MAX_COMPOUND_DEPTH        - Maximum nesting depth for compound rules
    GL_VRE_DEFAULT_PASS_THRESHOLD    - Default pass threshold for evaluations (0-1)
    GL_VRE_DEFAULT_WARN_THRESHOLD    - Default warn threshold for evaluations (0-1)
    GL_VRE_EVALUATION_TIMEOUT        - Maximum seconds per evaluation run
    GL_VRE_BATCH_SIZE                - Maximum records per batch evaluation chunk
    GL_VRE_MAX_BATCH_DATASETS        - Maximum datasets per batch evaluation run
    GL_VRE_ENABLE_PROVENANCE         - Enable SHA-256 provenance chain tracking
    GL_VRE_GENESIS_HASH              - Genesis anchor string for provenance chain
    GL_VRE_ENABLE_METRICS            - Enable Prometheus metrics export (true/false)
    GL_VRE_POOL_SIZE                 - Database connection pool size
    GL_VRE_CACHE_TTL                 - Cache time-to-live in seconds
    GL_VRE_RATE_LIMIT                - Max API requests per minute
    GL_VRE_ENABLE_CONFLICT_DETECTION - Enable rule conflict detection (true/false)
    GL_VRE_ENABLE_SHORT_CIRCUIT      - Enable short-circuit evaluation (true/false)
    GL_VRE_MAX_EVALUATION_ROWS       - Maximum rows per single evaluation run
    GL_VRE_REPORT_RETENTION_DAYS     - Days to retain validation reports

Example:
    >>> from greenlang.validation_rule_engine.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.max_rules, cfg.default_pass_threshold)
    100000 0.95

    >>> # Override for testing
    >>> from greenlang.validation_rule_engine.config import set_config, reset_config
    >>> from greenlang.validation_rule_engine.config import ValidationRuleEngineConfig
    >>> set_config(ValidationRuleEngineConfig(max_rules=500, enable_short_circuit=False))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-019 Validation Rule Engine (GL-DATA-X-022)
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

_ENV_PREFIX = "GL_VRE_"

# ---------------------------------------------------------------------------
# Valid log levels
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)


# ---------------------------------------------------------------------------
# ValidationRuleEngineConfig
# ---------------------------------------------------------------------------


@dataclass
class ValidationRuleEngineConfig:
    """Complete configuration for the GreenLang Validation Rule Engine Agent SDK.

    Attributes are grouped by concern: connections, logging, rule/rule-set
    capacity, compound rule nesting, evaluation thresholds, evaluation
    timeout, batch processing, provenance tracking, metrics export,
    performance tuning, conflict detection, short-circuit evaluation,
    processing limits, and report retention.

    All attributes can be overridden via environment variables using the
    ``GL_VRE_`` prefix (e.g. ``GL_VRE_MAX_RULES=200000``).

    Attributes:
        database_url: PostgreSQL connection URL for persistent rule storage
            and evaluation result persistence.
        redis_url: Redis connection URL for caching compiled rule sets,
            evaluation results, and distributed locks.
        log_level: Logging verbosity level. Accepts DEBUG, INFO, WARNING,
            ERROR, or CRITICAL.
        max_rules: Maximum individual validation rules in the registry.
            Each rule is an atomic check (range, regex, cross-field).
        max_rule_sets: Maximum rule sets (named rule collections) in the
            registry (e.g. "CSRD-E1-completeness").
        max_rules_per_set: Maximum rules within a single rule set.
        max_compound_depth: Maximum nesting depth for compound rules
            (AND/OR/NOT). Prevents stack overflow during evaluation.
        default_pass_threshold: Pass threshold (0.0-1.0) for evaluation
            scoring. Individual rule sets may override this value.
        default_warn_threshold: Warning threshold (0.0-1.0). Must be
            strictly less than default_pass_threshold.
        evaluation_timeout: Maximum seconds per evaluation run before
            forcible termination.
        batch_size: Maximum records per batch evaluation chunk. Controls
            memory consumption and checkpoint granularity.
        max_batch_datasets: Maximum datasets per batch evaluation request.
        enable_provenance: Compute and store SHA-256 provenance hashes for
            rule changes, evaluation runs, and result records.
        genesis_hash: Anchor string used as the root of every provenance
            chain.
        enable_metrics: When True, Prometheus metrics are exported under
            the ``gl_vre_`` prefix.
        pool_size: PostgreSQL connection pool size for the rule engine.
        cache_ttl: TTL (seconds) for cached compiled rule sets and recent
            evaluation results in Redis.
        rate_limit: Maximum inbound API requests per minute.
        enable_conflict_detection: When True, detects contradictory or
            overlapping rules within a rule set and surfaces conflicts
            in the rule set health report.
        enable_short_circuit: When True, evaluation terminates early when
            remaining rules cannot change the pass/fail outcome.
        max_evaluation_rows: Maximum data rows per evaluation run.
            Datasets exceeding this limit must be split.
        report_retention_days: Days to retain completed validation reports
            before automatic purging.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = ""
    redis_url: str = ""

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Rule / rule-set capacity --------------------------------------------
    max_rules: int = 100_000
    max_rule_sets: int = 10_000
    max_rules_per_set: int = 500

    # -- Compound rule nesting -----------------------------------------------
    max_compound_depth: int = 10

    # -- Evaluation thresholds -----------------------------------------------
    default_pass_threshold: float = 0.95
    default_warn_threshold: float = 0.80

    # -- Evaluation timeout --------------------------------------------------
    evaluation_timeout: int = 300

    # -- Batch processing ----------------------------------------------------
    batch_size: int = 1000
    max_batch_datasets: int = 100

    # -- Provenance tracking -------------------------------------------------
    enable_provenance: bool = True
    genesis_hash: str = "greenlang-validation-rule-genesis"

    # -- Metrics export ------------------------------------------------------
    enable_metrics: bool = True

    # -- Performance tuning --------------------------------------------------
    pool_size: int = 5
    cache_ttl: int = 300
    rate_limit: int = 200

    # -- Conflict detection --------------------------------------------------
    enable_conflict_detection: bool = True

    # -- Short-circuit evaluation --------------------------------------------
    enable_short_circuit: bool = True

    # -- Processing limits ---------------------------------------------------
    max_evaluation_rows: int = 1_000_000

    # -- Report retention ----------------------------------------------------
    report_retention_days: int = 90

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate configuration constraints after initialisation.

        Performs range checks on all numeric fields, relational checks
        between interdependent fields (e.g. warn threshold must be less
        than pass threshold), and normalisation of enumerated values
        (e.g. log_level to uppercase).

        Raises:
            ValueError: If any configuration value is outside its valid
                range or violates a relational constraint. The exception
                message lists all detected errors, not just the first one.
        """
        errors: list[str] = []

        # -- Connections -----------------------------------------------------
        # database_url and redis_url are allowed to be empty at construction
        # time (they may be injected at runtime by the service mesh), so we
        # only emit a WARNING rather than raising.
        if not self.database_url:
            logger.warning(
                "ValidationRuleEngineConfig: database_url is empty; "
                "the service will fail to connect until "
                "GL_VRE_DATABASE_URL is set."
            )
        if not self.redis_url:
            logger.warning(
                "ValidationRuleEngineConfig: redis_url is empty; "
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

        # -- Rule / rule-set capacity ----------------------------------------
        if self.max_rules <= 0:
            errors.append(
                f"max_rules must be > 0, got {self.max_rules}"
            )
        if self.max_rule_sets <= 0:
            errors.append(
                f"max_rule_sets must be > 0, got {self.max_rule_sets}"
            )
        if self.max_rules_per_set <= 0:
            errors.append(
                f"max_rules_per_set must be > 0, "
                f"got {self.max_rules_per_set}"
            )
        if (
            self.max_rules > 0
            and self.max_rule_sets > 0
            and self.max_rules_per_set > 0
            and self.max_rules_per_set > self.max_rules
        ):
            errors.append(
                f"max_rules_per_set ({self.max_rules_per_set}) must not "
                f"exceed max_rules ({self.max_rules})"
            )

        # -- Compound rule nesting -------------------------------------------
        if self.max_compound_depth <= 0:
            errors.append(
                f"max_compound_depth must be > 0, "
                f"got {self.max_compound_depth}"
            )
        if self.max_compound_depth > 100:
            errors.append(
                f"max_compound_depth must be <= 100 to prevent stack "
                f"overflow, got {self.max_compound_depth}"
            )

        # -- Evaluation thresholds -------------------------------------------
        if not (0.0 <= self.default_pass_threshold <= 1.0):
            errors.append(
                f"default_pass_threshold must be in [0.0, 1.0], "
                f"got {self.default_pass_threshold}"
            )
        if not (0.0 <= self.default_warn_threshold <= 1.0):
            errors.append(
                f"default_warn_threshold must be in [0.0, 1.0], "
                f"got {self.default_warn_threshold}"
            )
        if (
            0.0 <= self.default_warn_threshold <= 1.0
            and 0.0 <= self.default_pass_threshold <= 1.0
            and self.default_warn_threshold >= self.default_pass_threshold
        ):
            errors.append(
                f"default_warn_threshold ({self.default_warn_threshold}) "
                f"must be strictly less than default_pass_threshold "
                f"({self.default_pass_threshold})"
            )

        # -- Evaluation timeout ----------------------------------------------
        if self.evaluation_timeout <= 0:
            errors.append(
                f"evaluation_timeout must be > 0, "
                f"got {self.evaluation_timeout}"
            )
        if self.evaluation_timeout > 3600:
            errors.append(
                f"evaluation_timeout must be <= 3600 seconds (1 hour), "
                f"got {self.evaluation_timeout}"
            )

        # -- Batch processing ------------------------------------------------
        if self.batch_size <= 0:
            errors.append(
                f"batch_size must be > 0, got {self.batch_size}"
            )
        if self.max_batch_datasets <= 0:
            errors.append(
                f"max_batch_datasets must be > 0, "
                f"got {self.max_batch_datasets}"
            )

        # -- Provenance ------------------------------------------------------
        if not self.genesis_hash:
            errors.append("genesis_hash must not be empty")

        # -- Performance -----------------------------------------------------
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

        # -- Processing limits -----------------------------------------------
        if self.max_evaluation_rows <= 0:
            errors.append(
                f"max_evaluation_rows must be > 0, "
                f"got {self.max_evaluation_rows}"
            )

        # -- Report retention ------------------------------------------------
        if self.report_retention_days <= 0:
            errors.append(
                f"report_retention_days must be > 0, "
                f"got {self.report_retention_days}"
            )
        if self.report_retention_days > 3650:
            errors.append(
                f"report_retention_days must be <= 3650 (10 years), "
                f"got {self.report_retention_days}"
            )

        if errors:
            raise ValueError(
                "ValidationRuleEngineConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "ValidationRuleEngineConfig validated successfully: "
            "max_rules=%d, max_rule_sets=%d, max_rules_per_set=%d, "
            "max_compound_depth=%d, pass_threshold=%.2f, "
            "warn_threshold=%.2f, evaluation_timeout=%ds, "
            "batch_size=%d, max_batch_datasets=%d, "
            "conflict_detection=%s, short_circuit=%s, "
            "max_evaluation_rows=%d, report_retention_days=%d, "
            "provenance=%s, metrics=%s",
            self.max_rules,
            self.max_rule_sets,
            self.max_rules_per_set,
            self.max_compound_depth,
            self.default_pass_threshold,
            self.default_warn_threshold,
            self.evaluation_timeout,
            self.batch_size,
            self.max_batch_datasets,
            self.enable_conflict_detection,
            self.enable_short_circuit,
            self.max_evaluation_rows,
            self.report_retention_days,
            self.enable_provenance,
            self.enable_metrics,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> ValidationRuleEngineConfig:
        """Build a ValidationRuleEngineConfig from environment variables.

        Every field can be overridden via ``GL_VRE_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.
        Unknown or malformed values fall back to the class-level default
        and emit a WARNING log so the issue is visible in deployment logs.

        Returns:
            Populated ValidationRuleEngineConfig instance, validated via
            ``__post_init__``.

        Example:
            >>> import os
            >>> os.environ["GL_VRE_MAX_RULES"] = "200000"
            >>> cfg = ValidationRuleEngineConfig.from_env()
            >>> cfg.max_rules
            200000
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
            # Rule / rule-set capacity
            max_rules=_int("MAX_RULES", cls.max_rules),
            max_rule_sets=_int("MAX_RULE_SETS", cls.max_rule_sets),
            max_rules_per_set=_int(
                "MAX_RULES_PER_SET",
                cls.max_rules_per_set,
            ),
            # Compound rule nesting
            max_compound_depth=_int(
                "MAX_COMPOUND_DEPTH",
                cls.max_compound_depth,
            ),
            # Evaluation thresholds
            default_pass_threshold=_float(
                "DEFAULT_PASS_THRESHOLD",
                cls.default_pass_threshold,
            ),
            default_warn_threshold=_float(
                "DEFAULT_WARN_THRESHOLD",
                cls.default_warn_threshold,
            ),
            # Evaluation timeout
            evaluation_timeout=_int(
                "EVALUATION_TIMEOUT",
                cls.evaluation_timeout,
            ),
            # Batch processing
            batch_size=_int("BATCH_SIZE", cls.batch_size),
            max_batch_datasets=_int(
                "MAX_BATCH_DATASETS",
                cls.max_batch_datasets,
            ),
            # Provenance tracking
            enable_provenance=_bool(
                "ENABLE_PROVENANCE",
                cls.enable_provenance,
            ),
            genesis_hash=_str("GENESIS_HASH", cls.genesis_hash),
            # Metrics export
            enable_metrics=_bool("ENABLE_METRICS", cls.enable_metrics),
            # Performance tuning
            pool_size=_int("POOL_SIZE", cls.pool_size),
            cache_ttl=_int("CACHE_TTL", cls.cache_ttl),
            rate_limit=_int("RATE_LIMIT", cls.rate_limit),
            # Conflict detection
            enable_conflict_detection=_bool(
                "ENABLE_CONFLICT_DETECTION",
                cls.enable_conflict_detection,
            ),
            # Short-circuit evaluation
            enable_short_circuit=_bool(
                "ENABLE_SHORT_CIRCUIT",
                cls.enable_short_circuit,
            ),
            # Processing limits
            max_evaluation_rows=_int(
                "MAX_EVALUATION_ROWS",
                cls.max_evaluation_rows,
            ),
            # Report retention
            report_retention_days=_int(
                "REPORT_RETENTION_DAYS",
                cls.report_retention_days,
            ),
        )

        logger.info(
            "ValidationRuleEngineConfig loaded: "
            "max_rules=%d, max_rule_sets=%d, max_rules_per_set=%d, "
            "max_compound_depth=%d, "
            "pass_threshold=%.2f, warn_threshold=%.2f, "
            "evaluation_timeout=%ds, "
            "batch_size=%d, max_batch_datasets=%d, "
            "provenance=%s, "
            "pool=%d, cache_ttl=%ds, rate_limit=%d/min, "
            "conflict_detection=%s, short_circuit=%s, "
            "max_evaluation_rows=%d, report_retention_days=%d, "
            "metrics=%s",
            config.max_rules,
            config.max_rule_sets,
            config.max_rules_per_set,
            config.max_compound_depth,
            config.default_pass_threshold,
            config.default_warn_threshold,
            config.evaluation_timeout,
            config.batch_size,
            config.max_batch_datasets,
            config.enable_provenance,
            config.pool_size,
            config.cache_ttl,
            config.rate_limit,
            config.enable_conflict_detection,
            config.enable_short_circuit,
            config.max_evaluation_rows,
            config.report_retention_days,
            config.enable_metrics,
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
        redacted to prevent accidental credential leakage in logs,
        exception tracebacks, and monitoring dashboards.

        Returns:
            Dictionary representation of the configuration with sensitive
            fields redacted.

        Example:
            >>> cfg = ValidationRuleEngineConfig()
            >>> d = cfg.to_dict()
            >>> d["max_rules"]
            100000
            >>> d["database_url"]  # redacted
            '***'
        """
        return {
            # -- Connections (redacted) --------------------------------------
            "database_url": "***" if self.database_url else "",
            "redis_url": "***" if self.redis_url else "",
            # -- Logging -----------------------------------------------------
            "log_level": self.log_level,
            # -- Rule / rule-set capacity ------------------------------------
            "max_rules": self.max_rules,
            "max_rule_sets": self.max_rule_sets,
            "max_rules_per_set": self.max_rules_per_set,
            # -- Compound rule nesting ---------------------------------------
            "max_compound_depth": self.max_compound_depth,
            # -- Evaluation thresholds ---------------------------------------
            "default_pass_threshold": self.default_pass_threshold,
            "default_warn_threshold": self.default_warn_threshold,
            # -- Evaluation timeout ------------------------------------------
            "evaluation_timeout": self.evaluation_timeout,
            # -- Batch processing --------------------------------------------
            "batch_size": self.batch_size,
            "max_batch_datasets": self.max_batch_datasets,
            # -- Provenance tracking -----------------------------------------
            "enable_provenance": self.enable_provenance,
            "genesis_hash": self.genesis_hash,
            # -- Metrics export ----------------------------------------------
            "enable_metrics": self.enable_metrics,
            # -- Performance tuning ------------------------------------------
            "pool_size": self.pool_size,
            "cache_ttl": self.cache_ttl,
            "rate_limit": self.rate_limit,
            # -- Conflict detection ------------------------------------------
            "enable_conflict_detection": self.enable_conflict_detection,
            # -- Short-circuit evaluation ------------------------------------
            "enable_short_circuit": self.enable_short_circuit,
            # -- Processing limits -------------------------------------------
            "max_evaluation_rows": self.max_evaluation_rows,
            # -- Report retention --------------------------------------------
            "report_retention_days": self.report_retention_days,
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
        return f"ValidationRuleEngineConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[ValidationRuleEngineConfig] = None
_config_lock = threading.Lock()


def get_config() -> ValidationRuleEngineConfig:
    """Return the singleton ValidationRuleEngineConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path.  The instance is created on first call
    by reading all ``GL_VRE_*`` environment variables via
    :meth:`ValidationRuleEngineConfig.from_env`.

    Returns:
        ValidationRuleEngineConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.max_rules
        100000
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = ValidationRuleEngineConfig.from_env()
    return _config_instance


def set_config(config: ValidationRuleEngineConfig) -> None:
    """Replace the singleton ValidationRuleEngineConfig.

    Primarily intended for testing and dependency injection scenarios
    where a custom configuration must be supplied without relying on
    environment variables.

    Args:
        config: New :class:`ValidationRuleEngineConfig` to install as the
            singleton.

    Example:
        >>> cfg = ValidationRuleEngineConfig(max_rules=500, enable_short_circuit=False)
        >>> set_config(cfg)
        >>> assert get_config().max_rules == 500
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info(
        "ValidationRuleEngineConfig replaced programmatically: "
        "max_rules=%d, max_rule_sets=%d, max_compound_depth=%d, "
        "pass_threshold=%.2f, warn_threshold=%.2f",
        config.max_rules,
        config.max_rule_sets,
        config.max_compound_depth,
        config.default_pass_threshold,
        config.default_warn_threshold,
    )


def reset_config() -> None:
    """Reset the singleton ValidationRuleEngineConfig to ``None``.

    The next call to :func:`get_config` will re-read environment variables
    and construct a fresh instance.  Intended for test teardown to prevent
    state leakage between test cases.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads GL_VRE_* env vars
    """
    global _config_instance
    with _config_lock:
        _config_instance = None
    logger.debug("ValidationRuleEngineConfig singleton reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "ValidationRuleEngineConfig",
    "get_config",
    "set_config",
    "reset_config",
]
