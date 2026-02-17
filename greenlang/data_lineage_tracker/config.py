# -*- coding: utf-8 -*-
"""
Data Lineage Tracker Service Configuration - AGENT-DATA-018

Centralized configuration for the Data Lineage Tracker Agent SDK covering:
- Database, cache, and connection defaults
- Asset and graph capacity limits (max assets, transformations, edges, depth)
- Traversal settings (default traversal depth for upstream/downstream queries)
- Snapshot scheduling (interval for lineage graph snapshots)
- Column-level lineage toggle (field-level tracking granularity)
- Change detection (automatic lineage change event capture)
- Provenance tracking (genesis hash, SHA-256 chain anchoring)
- Connection pool sizing, rate limiting, and cache TTL
- Batch processing limits for bulk lineage ingestion
- Lineage coverage quality thresholds (warn and fail levels)
- Data freshness constraints (maximum acceptable age for lineage data)
- Quality score weight distribution across scoring dimensions
- Prometheus metrics export toggle
- Processing limits and logging level

All settings can be overridden via environment variables with the
``GL_DLT_`` prefix (e.g. ``GL_DLT_MAX_ASSETS``, ``GL_DLT_MAX_GRAPH_DEPTH``).

Environment Variable Reference (GL_DLT_ prefix):
    GL_DLT_DATABASE_URL              - PostgreSQL connection URL
    GL_DLT_REDIS_URL                 - Redis connection URL
    GL_DLT_LOG_LEVEL                 - Logging level (DEBUG/INFO/WARNING/ERROR)
    GL_DLT_MAX_ASSETS                - Maximum data assets tracked
    GL_DLT_MAX_TRANSFORMATIONS       - Maximum transformations tracked
    GL_DLT_MAX_EDGES                 - Maximum lineage graph edges
    GL_DLT_MAX_GRAPH_DEPTH           - Maximum graph traversal depth limit
    GL_DLT_DEFAULT_TRAVERSAL_DEPTH   - Default upstream/downstream traversal depth
    GL_DLT_SNAPSHOT_INTERVAL_MINUTES - Interval (min) between lineage snapshots
    GL_DLT_ENABLE_COLUMN_LINEAGE     - Enable column-level lineage (true/false)
    GL_DLT_ENABLE_CHANGE_DETECTION   - Enable lineage change detection (true/false)
    GL_DLT_ENABLE_PROVENANCE         - Enable SHA-256 provenance chain tracking
    GL_DLT_GENESIS_HASH              - Genesis anchor string for provenance chain
    GL_DLT_POOL_SIZE                 - Database connection pool size
    GL_DLT_CACHE_TTL                 - Cache time-to-live in seconds
    GL_DLT_RATE_LIMIT                - Max API requests per minute
    GL_DLT_BATCH_SIZE                - Maximum records per bulk ingestion batch
    GL_DLT_COVERAGE_WARN_THRESHOLD   - Lineage coverage warning threshold (0-1)
    GL_DLT_COVERAGE_FAIL_THRESHOLD   - Lineage coverage failure threshold (0-1)
    GL_DLT_FRESHNESS_MAX_AGE_HOURS   - Maximum acceptable lineage data age (hours)
    GL_DLT_QUALITY_SCORE_WEIGHTS     - JSON string of quality score dimension weights
    GL_DLT_ENABLE_METRICS            - Enable Prometheus metrics export (true/false)

Example:
    >>> from greenlang.data_lineage_tracker.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.max_assets, cfg.default_traversal_depth)
    100000 10

    >>> # Override for testing
    >>> from greenlang.data_lineage_tracker.config import set_config, reset_config
    >>> from greenlang.data_lineage_tracker.config import DataLineageTrackerConfig
    >>> set_config(DataLineageTrackerConfig(max_assets=500, enable_column_lineage=False))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-018 Data Lineage Tracker (GL-DATA-X-021)
Status: Production Ready
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_DLT_"

# ---------------------------------------------------------------------------
# Valid log levels
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

# ---------------------------------------------------------------------------
# Default quality score weights
# ---------------------------------------------------------------------------

_DEFAULT_QUALITY_SCORE_WEIGHTS = (
    '{"source_credibility":0.3,"transformation_depth":0.2,'
    '"freshness":0.2,"documentation":0.15,"manual_interventions":0.15}'
)

# ---------------------------------------------------------------------------
# Required quality score weight keys
# ---------------------------------------------------------------------------

_REQUIRED_WEIGHT_KEYS = frozenset(
    {"source_credibility", "transformation_depth", "freshness",
     "documentation", "manual_interventions"}
)


# ---------------------------------------------------------------------------
# DataLineageTrackerConfig
# ---------------------------------------------------------------------------


@dataclass
class DataLineageTrackerConfig:
    """Complete configuration for the GreenLang Data Lineage Tracker Agent SDK.

    Attributes are grouped by concern: connections, logging, asset/graph
    capacity, traversal settings, snapshot scheduling, column lineage,
    change detection, provenance tracking, performance tuning, batch
    processing, lineage coverage quality, freshness constraints, quality
    scoring, and metrics export.

    All attributes can be overridden via environment variables using the
    ``GL_DLT_`` prefix (e.g. ``GL_DLT_MAX_ASSETS=200000``).

    Attributes:
        database_url: PostgreSQL connection URL for persistent lineage storage.
        redis_url: Redis connection URL for caching lineage queries and locks.
        log_level: Logging verbosity level for the data lineage tracker service.
        max_assets: Maximum number of data assets tracked in the lineage graph.
        max_transformations: Maximum number of transformation records managed.
        max_edges: Maximum number of directed edges in the lineage graph.
        max_graph_depth: Hard ceiling on graph traversal depth to prevent
            runaway queries on deeply nested lineage chains.
        default_traversal_depth: Default depth used for upstream/downstream
            lineage queries when no explicit depth is specified by the caller.
        snapshot_interval_minutes: Interval between automatic lineage graph
            snapshots for point-in-time recovery and historical analysis.
        enable_column_lineage: When True, the tracker records field-level
            (column-level) lineage in addition to asset-level lineage.
        enable_change_detection: When True, the tracker automatically detects
            and records changes to lineage relationships over time.
        enable_provenance: Whether to compute and store SHA-256 provenance
            hashes for every lineage event and transformation record.
        genesis_hash: Anchor string used as the root of every provenance chain.
        pool_size: PostgreSQL connection pool size for the lineage service.
        cache_ttl: Time-to-live (s) for cached lineage query results.
        rate_limit: Maximum inbound API requests allowed per minute.
        batch_size: Maximum records per bulk lineage ingestion batch.
        coverage_warn_threshold: Lineage coverage ratio below which a warning
            is emitted (0.0 to 1.0). Indicates incomplete lineage documentation.
        coverage_fail_threshold: Lineage coverage ratio below which a failure
            is reported (0.0 to 1.0). Must be less than coverage_warn_threshold.
        freshness_max_age_hours: Maximum acceptable age (hours) for lineage
            data before it is considered stale and triggers a freshness alert.
        quality_score_weights: JSON string defining the weight distribution
            across quality scoring dimensions. Must contain exactly the keys:
            source_credibility, transformation_depth, freshness, documentation,
            manual_interventions. Weights must sum to 1.0.
        enable_metrics: When True, Prometheus metrics are exported for the
            data lineage tracker service under the ``gl_dlt_`` prefix.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = ""
    redis_url: str = ""

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Asset / graph capacity ----------------------------------------------
    max_assets: int = 100_000
    max_transformations: int = 500_000
    max_edges: int = 1_000_000
    max_graph_depth: int = 50

    # -- Traversal settings --------------------------------------------------
    default_traversal_depth: int = 10

    # -- Snapshot scheduling -------------------------------------------------
    snapshot_interval_minutes: int = 60

    # -- Column-level lineage ------------------------------------------------
    enable_column_lineage: bool = True

    # -- Change detection ----------------------------------------------------
    enable_change_detection: bool = True

    # -- Provenance tracking -------------------------------------------------
    enable_provenance: bool = True
    genesis_hash: str = "greenlang-data-lineage-genesis"

    # -- Performance tuning --------------------------------------------------
    pool_size: int = 5
    cache_ttl: int = 300
    rate_limit: int = 200

    # -- Batch processing ----------------------------------------------------
    batch_size: int = 1000

    # -- Lineage coverage quality --------------------------------------------
    coverage_warn_threshold: float = 0.8
    coverage_fail_threshold: float = 0.5

    # -- Freshness constraints -----------------------------------------------
    freshness_max_age_hours: int = 24

    # -- Quality scoring -----------------------------------------------------
    quality_score_weights: str = _DEFAULT_QUALITY_SCORE_WEIGHTS

    # -- Metrics export ------------------------------------------------------
    enable_metrics: bool = True

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
                "DataLineageTrackerConfig: database_url is empty; "
                "the service will fail to connect until GL_DLT_DATABASE_URL is set."
            )
        if not self.redis_url:
            logger.warning(
                "DataLineageTrackerConfig: redis_url is empty; "
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

        # -- Asset / graph capacity ------------------------------------------
        if self.max_assets <= 0:
            errors.append(
                f"max_assets must be > 0, got {self.max_assets}"
            )
        if self.max_transformations <= 0:
            errors.append(
                f"max_transformations must be > 0, "
                f"got {self.max_transformations}"
            )
        if self.max_edges <= 0:
            errors.append(
                f"max_edges must be > 0, got {self.max_edges}"
            )
        if self.max_graph_depth <= 0:
            errors.append(
                f"max_graph_depth must be > 0, got {self.max_graph_depth}"
            )

        # -- Traversal settings ----------------------------------------------
        if self.default_traversal_depth <= 0:
            errors.append(
                f"default_traversal_depth must be > 0, "
                f"got {self.default_traversal_depth}"
            )
        if self.default_traversal_depth > self.max_graph_depth:
            errors.append(
                f"default_traversal_depth ({self.default_traversal_depth}) "
                f"must not exceed max_graph_depth ({self.max_graph_depth})"
            )

        # -- Snapshot scheduling ---------------------------------------------
        if self.snapshot_interval_minutes <= 0:
            errors.append(
                f"snapshot_interval_minutes must be > 0, "
                f"got {self.snapshot_interval_minutes}"
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

        # -- Batch processing ------------------------------------------------
        if self.batch_size <= 0:
            errors.append(
                f"batch_size must be > 0, got {self.batch_size}"
            )

        # -- Lineage coverage quality ----------------------------------------
        if not (0.0 <= self.coverage_warn_threshold <= 1.0):
            errors.append(
                f"coverage_warn_threshold must be in [0.0, 1.0], "
                f"got {self.coverage_warn_threshold}"
            )
        if not (0.0 <= self.coverage_fail_threshold <= 1.0):
            errors.append(
                f"coverage_fail_threshold must be in [0.0, 1.0], "
                f"got {self.coverage_fail_threshold}"
            )
        if self.coverage_fail_threshold > self.coverage_warn_threshold:
            errors.append(
                f"coverage_fail_threshold ({self.coverage_fail_threshold}) "
                f"must not exceed coverage_warn_threshold "
                f"({self.coverage_warn_threshold})"
            )

        # -- Freshness constraints -------------------------------------------
        if self.freshness_max_age_hours <= 0:
            errors.append(
                f"freshness_max_age_hours must be > 0, "
                f"got {self.freshness_max_age_hours}"
            )

        # -- Quality scoring -------------------------------------------------
        try:
            parsed_weights = json.loads(self.quality_score_weights)
            if not isinstance(parsed_weights, dict):
                errors.append(
                    "quality_score_weights must be a JSON object, "
                    f"got {type(parsed_weights).__name__}"
                )
            else:
                missing_keys = _REQUIRED_WEIGHT_KEYS - set(parsed_weights.keys())
                if missing_keys:
                    errors.append(
                        f"quality_score_weights missing required keys: "
                        f"{sorted(missing_keys)}"
                    )
                extra_keys = set(parsed_weights.keys()) - _REQUIRED_WEIGHT_KEYS
                if extra_keys:
                    errors.append(
                        f"quality_score_weights has unknown keys: "
                        f"{sorted(extra_keys)}"
                    )
                # Validate all values are numeric
                for key, val in parsed_weights.items():
                    if not isinstance(val, (int, float)):
                        errors.append(
                            f"quality_score_weights['{key}'] must be numeric, "
                            f"got {type(val).__name__}"
                        )
                # Validate weights sum to 1.0 (with tolerance)
                if not missing_keys and not extra_keys:
                    weight_sum = sum(
                        v for v in parsed_weights.values()
                        if isinstance(v, (int, float))
                    )
                    if abs(weight_sum - 1.0) > 0.001:
                        errors.append(
                            f"quality_score_weights must sum to 1.0, "
                            f"got {weight_sum:.6f}"
                        )
        except json.JSONDecodeError as exc:
            errors.append(
                f"quality_score_weights is not valid JSON: {exc}"
            )

        if errors:
            raise ValueError(
                "DataLineageTrackerConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "DataLineageTrackerConfig validated successfully: "
            "max_assets=%d, max_transformations=%d, max_edges=%d, "
            "max_graph_depth=%d, default_traversal_depth=%d, "
            "column_lineage=%s, change_detection=%s, provenance=%s, "
            "metrics=%s",
            self.max_assets,
            self.max_transformations,
            self.max_edges,
            self.max_graph_depth,
            self.default_traversal_depth,
            self.enable_column_lineage,
            self.enable_change_detection,
            self.enable_provenance,
            self.enable_metrics,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> DataLineageTrackerConfig:
        """Build a DataLineageTrackerConfig from environment variables.

        Every field can be overridden via ``GL_DLT_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.
        Unknown / malformed values fall back to the class-level default and
        emit a WARNING log so the issue is visible in deployment logs.

        Returns:
            Populated DataLineageTrackerConfig instance, validated via
            ``__post_init__``.

        Example:
            >>> import os
            >>> os.environ["GL_DLT_MAX_ASSETS"] = "200000"
            >>> cfg = DataLineageTrackerConfig.from_env()
            >>> cfg.max_assets
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
            # Asset / graph capacity
            max_assets=_int("MAX_ASSETS", cls.max_assets),
            max_transformations=_int(
                "MAX_TRANSFORMATIONS",
                cls.max_transformations,
            ),
            max_edges=_int("MAX_EDGES", cls.max_edges),
            max_graph_depth=_int(
                "MAX_GRAPH_DEPTH",
                cls.max_graph_depth,
            ),
            # Traversal settings
            default_traversal_depth=_int(
                "DEFAULT_TRAVERSAL_DEPTH",
                cls.default_traversal_depth,
            ),
            # Snapshot scheduling
            snapshot_interval_minutes=_int(
                "SNAPSHOT_INTERVAL_MINUTES",
                cls.snapshot_interval_minutes,
            ),
            # Column-level lineage
            enable_column_lineage=_bool(
                "ENABLE_COLUMN_LINEAGE",
                cls.enable_column_lineage,
            ),
            # Change detection
            enable_change_detection=_bool(
                "ENABLE_CHANGE_DETECTION",
                cls.enable_change_detection,
            ),
            # Provenance tracking
            enable_provenance=_bool("ENABLE_PROVENANCE", cls.enable_provenance),
            genesis_hash=_str("GENESIS_HASH", cls.genesis_hash),
            # Performance tuning
            pool_size=_int("POOL_SIZE", cls.pool_size),
            cache_ttl=_int("CACHE_TTL", cls.cache_ttl),
            rate_limit=_int("RATE_LIMIT", cls.rate_limit),
            # Batch processing
            batch_size=_int("BATCH_SIZE", cls.batch_size),
            # Lineage coverage quality
            coverage_warn_threshold=_float(
                "COVERAGE_WARN_THRESHOLD",
                cls.coverage_warn_threshold,
            ),
            coverage_fail_threshold=_float(
                "COVERAGE_FAIL_THRESHOLD",
                cls.coverage_fail_threshold,
            ),
            # Freshness constraints
            freshness_max_age_hours=_int(
                "FRESHNESS_MAX_AGE_HOURS",
                cls.freshness_max_age_hours,
            ),
            # Quality scoring
            quality_score_weights=_str(
                "QUALITY_SCORE_WEIGHTS",
                cls.quality_score_weights,
            ),
            # Metrics export
            enable_metrics=_bool("ENABLE_METRICS", cls.enable_metrics),
        )

        logger.info(
            "DataLineageTrackerConfig loaded: "
            "max_assets=%d, max_transformations=%d, max_edges=%d, "
            "max_graph_depth=%d, default_traversal=%d, "
            "snapshot_interval=%dmin, "
            "column_lineage=%s, change_detection=%s, "
            "provenance=%s, "
            "pool=%d, cache_ttl=%ds, rate_limit=%d/min, "
            "batch_size=%d, "
            "coverage_warn=%.2f, coverage_fail=%.2f, "
            "freshness_max_age=%dh, "
            "metrics=%s",
            config.max_assets,
            config.max_transformations,
            config.max_edges,
            config.max_graph_depth,
            config.default_traversal_depth,
            config.snapshot_interval_minutes,
            config.enable_column_lineage,
            config.enable_change_detection,
            config.enable_provenance,
            config.pool_size,
            config.cache_ttl,
            config.rate_limit,
            config.batch_size,
            config.coverage_warn_threshold,
            config.coverage_fail_threshold,
            config.freshness_max_age_hours,
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
        redacted to prevent accidental credential leakage in logs.

        Returns:
            Dictionary representation of the configuration with sensitive
            fields redacted.

        Example:
            >>> cfg = DataLineageTrackerConfig()
            >>> d = cfg.to_dict()
            >>> d["max_assets"]
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
            # -- Asset / graph capacity --------------------------------------
            "max_assets": self.max_assets,
            "max_transformations": self.max_transformations,
            "max_edges": self.max_edges,
            "max_graph_depth": self.max_graph_depth,
            # -- Traversal settings ------------------------------------------
            "default_traversal_depth": self.default_traversal_depth,
            # -- Snapshot scheduling -----------------------------------------
            "snapshot_interval_minutes": self.snapshot_interval_minutes,
            # -- Column-level lineage ----------------------------------------
            "enable_column_lineage": self.enable_column_lineage,
            # -- Change detection --------------------------------------------
            "enable_change_detection": self.enable_change_detection,
            # -- Provenance tracking -----------------------------------------
            "enable_provenance": self.enable_provenance,
            "genesis_hash": self.genesis_hash,
            # -- Performance tuning ------------------------------------------
            "pool_size": self.pool_size,
            "cache_ttl": self.cache_ttl,
            "rate_limit": self.rate_limit,
            # -- Batch processing --------------------------------------------
            "batch_size": self.batch_size,
            # -- Lineage coverage quality ------------------------------------
            "coverage_warn_threshold": self.coverage_warn_threshold,
            "coverage_fail_threshold": self.coverage_fail_threshold,
            # -- Freshness constraints ---------------------------------------
            "freshness_max_age_hours": self.freshness_max_age_hours,
            # -- Quality scoring ---------------------------------------------
            "quality_score_weights": self.quality_score_weights,
            # -- Metrics export ----------------------------------------------
            "enable_metrics": self.enable_metrics,
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
        return f"DataLineageTrackerConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[DataLineageTrackerConfig] = None
_config_lock = threading.Lock()


def get_config() -> DataLineageTrackerConfig:
    """Return the singleton DataLineageTrackerConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path.  The instance is created on first call
    by reading all ``GL_DLT_*`` environment variables via
    :meth:`DataLineageTrackerConfig.from_env`.

    Returns:
        DataLineageTrackerConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.max_assets
        100000
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = DataLineageTrackerConfig.from_env()
    return _config_instance


def set_config(config: DataLineageTrackerConfig) -> None:
    """Replace the singleton DataLineageTrackerConfig.

    Primarily intended for testing and dependency injection scenarios
    where a custom configuration must be supplied without relying on
    environment variables.

    Args:
        config: New :class:`DataLineageTrackerConfig` to install as the
            singleton.

    Example:
        >>> cfg = DataLineageTrackerConfig(max_assets=500, enable_column_lineage=False)
        >>> set_config(cfg)
        >>> assert get_config().max_assets == 500
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info(
        "DataLineageTrackerConfig replaced programmatically: "
        "max_assets=%d, max_graph_depth=%d, column_lineage=%s",
        config.max_assets,
        config.max_graph_depth,
        config.enable_column_lineage,
    )


def reset_config() -> None:
    """Reset the singleton DataLineageTrackerConfig to ``None``.

    The next call to :func:`get_config` will re-read environment variables
    and construct a fresh instance.  Intended for test teardown to prevent
    state leakage between test cases.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads GL_DLT_* env vars
    """
    global _config_instance
    with _config_lock:
        _config_instance = None
    logger.debug("DataLineageTrackerConfig singleton reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "DataLineageTrackerConfig",
    "get_config",
    "set_config",
    "reset_config",
]
