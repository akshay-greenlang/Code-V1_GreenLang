# -*- coding: utf-8 -*-
"""
Supply Chain Mapper Configuration - AGENT-EUDR-001

Centralized configuration for the Supply Chain Mapping Master Agent covering:
- Database and cache connection settings
- Risk propagation weight defaults (country, commodity, supplier, deforestation)
- Performance thresholds (graph query latency, batch throughput, memory limits)
- Graph capacity limits (max nodes, max edges, max tier depth)
- Gap analysis sensitivity settings
- Provenance tracking (genesis hash, SHA-256 chain anchoring)
- Prometheus metrics export toggle
- Connection pool sizing, rate limiting, and cache TTL

All settings can be overridden via environment variables with the
``GL_EUDR_SCM_`` prefix (e.g. ``GL_EUDR_SCM_DATABASE_URL``,
``GL_EUDR_SCM_RISK_WEIGHT_COUNTRY``).

Environment Variable Reference (GL_EUDR_SCM_ prefix):
    GL_EUDR_SCM_DATABASE_URL                - PostgreSQL connection URL
    GL_EUDR_SCM_REDIS_URL                   - Redis connection URL
    GL_EUDR_SCM_LOG_LEVEL                   - Logging level (DEBUG/INFO/WARNING/ERROR)
    GL_EUDR_SCM_RISK_WEIGHT_COUNTRY         - Country risk weight (0.0-1.0)
    GL_EUDR_SCM_RISK_WEIGHT_COMMODITY       - Commodity risk weight (0.0-1.0)
    GL_EUDR_SCM_RISK_WEIGHT_SUPPLIER        - Supplier risk weight (0.0-1.0)
    GL_EUDR_SCM_RISK_WEIGHT_DEFORESTATION   - Deforestation risk weight (0.0-1.0)
    GL_EUDR_SCM_MAX_NODES_PER_GRAPH         - Maximum nodes per graph
    GL_EUDR_SCM_MAX_EDGES_PER_GRAPH         - Maximum edges per graph
    GL_EUDR_SCM_MAX_TIER_DEPTH              - Maximum tier depth for recursive mapping
    GL_EUDR_SCM_GRAPH_QUERY_TIMEOUT_MS      - Graph query timeout in milliseconds
    GL_EUDR_SCM_BATCH_THROUGHPUT_TARGET     - Target custody transfers per minute
    GL_EUDR_SCM_MEMORY_LIMIT_MB            - Memory limit for in-memory graph (MB)
    GL_EUDR_SCM_RISK_HIGH_THRESHOLD         - Score threshold for HIGH risk (0-100)
    GL_EUDR_SCM_RISK_LOW_THRESHOLD          - Score threshold for LOW risk (0-100)
    GL_EUDR_SCM_MASS_BALANCE_TOLERANCE      - Tolerance % for mass balance checks
    GL_EUDR_SCM_STALE_DATA_DAYS             - Days before data is flagged stale
    GL_EUDR_SCM_ENABLE_PROVENANCE           - Enable SHA-256 provenance chain
    GL_EUDR_SCM_GENESIS_HASH                - Genesis anchor string for provenance
    GL_EUDR_SCM_ENABLE_METRICS              - Enable Prometheus metrics export
    GL_EUDR_SCM_POOL_SIZE                   - Database connection pool size
    GL_EUDR_SCM_CACHE_TTL                   - Cache time-to-live in seconds
    GL_EUDR_SCM_RATE_LIMIT                  - Max API requests per minute

Example:
    >>> from greenlang.agents.eudr.supply_chain_mapper.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.risk_weight_country, cfg.risk_high_threshold)
    0.3 70.0

    >>> # Override for testing
    >>> from greenlang.agents.eudr.supply_chain_mapper.config import (
    ...     set_config, reset_config, SupplyChainMapperConfig,
    ... )
    >>> set_config(SupplyChainMapperConfig(risk_weight_country=0.4))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-001 Supply Chain Mapping Master (GL-EUDR-SCM-001)
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

_ENV_PREFIX = "GL_EUDR_SCM_"

# ---------------------------------------------------------------------------
# Valid log levels
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)


# ---------------------------------------------------------------------------
# SupplyChainMapperConfig
# ---------------------------------------------------------------------------


@dataclass
class SupplyChainMapperConfig:
    """Complete configuration for the EUDR Supply Chain Mapping Master Agent.

    Attributes are grouped by concern: connections, logging, risk
    propagation weights, graph capacity limits, performance targets,
    gap analysis settings, provenance tracking, metrics export, and
    connection pool tuning.

    All attributes can be overridden via environment variables using
    the ``GL_EUDR_SCM_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage
            of supply chain graphs, nodes, edges, and gap analysis results.
        redis_url: Redis connection URL for graph query caching,
            node lookup caching, and distributed locks.
        log_level: Logging verbosity level. Accepts DEBUG, INFO,
            WARNING, ERROR, or CRITICAL.
        risk_weight_country: Weight for country-level risk in the
            composite risk formula (0.0-1.0, default 0.30).
        risk_weight_commodity: Weight for commodity-level risk in the
            composite risk formula (0.0-1.0, default 0.20).
        risk_weight_supplier: Weight for supplier-level risk in the
            composite risk formula (0.0-1.0, default 0.25).
        risk_weight_deforestation: Weight for deforestation risk in
            the composite risk formula (0.0-1.0, default 0.25).
        max_nodes_per_graph: Maximum number of nodes allowed in a
            single supply chain graph before sharding is recommended.
        max_edges_per_graph: Maximum number of edges allowed in a
            single supply chain graph.
        max_tier_depth: Maximum tier depth for recursive supply chain
            mapping (from importer back to producer).
        graph_query_timeout_ms: Timeout in milliseconds for graph
            traversal queries. Target: < 500ms p99 for 10K nodes.
        batch_throughput_target: Target custody transfers per minute
            for batch processing operations.
        memory_limit_mb: Maximum memory in MB for in-memory graph
            representation. Target: < 2048 MB for 100K nodes.
        risk_high_threshold: Risk score at or above which a node is
            classified as HIGH risk (0-100).
        risk_low_threshold: Risk score at or below which a node is
            classified as LOW risk (0-100).
        mass_balance_tolerance: Tolerance percentage for mass balance
            verification (output vs input quantity). Values above
            this tolerance trigger a gap alert.
        stale_data_days: Number of days after which data without a
            refresh is flagged as stale per EUDR Article 31.
        enable_provenance: Enable SHA-256 provenance chain tracking
            for all graph mutations and operations.
        genesis_hash: Anchor string for the provenance chain, unique
            to the Supply Chain Mapper agent.
        enable_metrics: Enable Prometheus metrics export under the
            ``gl_eudr_scm_`` prefix.
        pool_size: PostgreSQL connection pool size.
        cache_ttl: Cache TTL in seconds for graph query results
            and node lookups in Redis.
        rate_limit: Maximum inbound API requests per minute.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = "postgresql://localhost:5432/greenlang"
    redis_url: str = "redis://localhost:6379/0"

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Risk propagation weights (must sum to 1.0) --------------------------
    risk_weight_country: float = 0.30
    risk_weight_commodity: float = 0.20
    risk_weight_supplier: float = 0.25
    risk_weight_deforestation: float = 0.25

    # -- Graph capacity limits -----------------------------------------------
    max_nodes_per_graph: int = 100_000
    max_edges_per_graph: int = 500_000
    max_tier_depth: int = 50

    # -- Performance targets -------------------------------------------------
    graph_query_timeout_ms: int = 500
    batch_throughput_target: int = 50_000
    memory_limit_mb: int = 2048

    # -- Risk classification thresholds --------------------------------------
    risk_high_threshold: float = 70.0
    risk_low_threshold: float = 30.0

    # -- Gap analysis settings -----------------------------------------------
    mass_balance_tolerance: float = 2.0
    stale_data_days: int = 365

    # -- Provenance tracking -------------------------------------------------
    enable_provenance: bool = True
    genesis_hash: str = "GL-EUDR-SCM-001-SUPPLY-CHAIN-MAPPER-GENESIS"

    # -- Metrics export ------------------------------------------------------
    enable_metrics: bool = True

    # -- Performance tuning --------------------------------------------------
    pool_size: int = 10
    cache_ttl: int = 3600
    rate_limit: int = 1000

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate configuration constraints after initialization.

        Performs range checks on all numeric fields, enumeration checks
        on string fields, and normalization. Collects all errors before
        raising a single ValueError with all violations listed.

        Raises:
            ValueError: If any configuration value is outside its valid
                range or violates a constraint.
        """
        errors: list[str] = []

        # -- Logging ---------------------------------------------------------
        normalised_log = self.log_level.upper()
        if normalised_log not in _VALID_LOG_LEVELS:
            errors.append(
                f"log_level must be one of {sorted(_VALID_LOG_LEVELS)}, "
                f"got '{self.log_level}'"
            )
        else:
            self.log_level = normalised_log

        # -- Risk weights ----------------------------------------------------
        weights = [
            ("risk_weight_country", self.risk_weight_country),
            ("risk_weight_commodity", self.risk_weight_commodity),
            ("risk_weight_supplier", self.risk_weight_supplier),
            ("risk_weight_deforestation", self.risk_weight_deforestation),
        ]
        for name, value in weights:
            if not (0.0 <= value <= 1.0):
                errors.append(
                    f"{name} must be in [0.0, 1.0], got {value}"
                )

        weight_sum = sum(v for _, v in weights)
        if abs(weight_sum - 1.0) > 0.001:
            errors.append(
                f"Risk weights must sum to 1.0, got {weight_sum:.4f}"
            )

        # -- Risk thresholds -------------------------------------------------
        if not (0.0 <= self.risk_low_threshold <= 100.0):
            errors.append(
                f"risk_low_threshold must be in [0, 100], "
                f"got {self.risk_low_threshold}"
            )
        if not (0.0 <= self.risk_high_threshold <= 100.0):
            errors.append(
                f"risk_high_threshold must be in [0, 100], "
                f"got {self.risk_high_threshold}"
            )
        if self.risk_low_threshold >= self.risk_high_threshold:
            errors.append(
                f"risk_low_threshold ({self.risk_low_threshold}) must be "
                f"less than risk_high_threshold ({self.risk_high_threshold})"
            )

        # -- Capacity limits -------------------------------------------------
        for field_name, value, upper in [
            ("max_nodes_per_graph", self.max_nodes_per_graph, 10_000_000),
            ("max_edges_per_graph", self.max_edges_per_graph, 50_000_000),
            ("max_tier_depth", self.max_tier_depth, 1000),
        ]:
            if value <= 0:
                errors.append(f"{field_name} must be > 0, got {value}")
            if value > upper:
                errors.append(f"{field_name} must be <= {upper}, got {value}")

        # -- Performance targets ---------------------------------------------
        if self.graph_query_timeout_ms <= 0:
            errors.append(
                f"graph_query_timeout_ms must be > 0, "
                f"got {self.graph_query_timeout_ms}"
            )
        if self.batch_throughput_target <= 0:
            errors.append(
                f"batch_throughput_target must be > 0, "
                f"got {self.batch_throughput_target}"
            )
        if self.memory_limit_mb <= 0:
            errors.append(
                f"memory_limit_mb must be > 0, got {self.memory_limit_mb}"
            )

        # -- Gap analysis settings -------------------------------------------
        if not (0.0 <= self.mass_balance_tolerance <= 100.0):
            errors.append(
                f"mass_balance_tolerance must be in [0, 100], "
                f"got {self.mass_balance_tolerance}"
            )
        if self.stale_data_days <= 0:
            errors.append(
                f"stale_data_days must be > 0, got {self.stale_data_days}"
            )

        # -- Provenance ------------------------------------------------------
        if not self.genesis_hash:
            errors.append("genesis_hash must not be empty")

        # -- Performance tuning ----------------------------------------------
        if self.pool_size <= 0:
            errors.append(f"pool_size must be > 0, got {self.pool_size}")
        if self.cache_ttl <= 0:
            errors.append(f"cache_ttl must be > 0, got {self.cache_ttl}")
        if self.rate_limit <= 0:
            errors.append(f"rate_limit must be > 0, got {self.rate_limit}")

        if errors:
            raise ValueError(
                "SupplyChainMapperConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "SupplyChainMapperConfig validated successfully: "
            "risk_weights=[%.2f, %.2f, %.2f, %.2f], "
            "max_nodes=%d, max_edges=%d, max_tier_depth=%d, "
            "query_timeout=%dms, provenance=%s, metrics=%s",
            self.risk_weight_country,
            self.risk_weight_commodity,
            self.risk_weight_supplier,
            self.risk_weight_deforestation,
            self.max_nodes_per_graph,
            self.max_edges_per_graph,
            self.max_tier_depth,
            self.graph_query_timeout_ms,
            self.enable_provenance,
            self.enable_metrics,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> SupplyChainMapperConfig:
        """Build a SupplyChainMapperConfig from environment variables.

        Every field can be overridden via ``GL_EUDR_SCM_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.
        Unknown or malformed values fall back to the class-level default
        and emit a WARNING log so the issue is visible in deployment logs.

        Returns:
            Populated SupplyChainMapperConfig instance, validated via
            ``__post_init__``.
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
                    prefix, name, val, default,
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
                    prefix, name, val, default,
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
            # Risk weights
            risk_weight_country=_float(
                "RISK_WEIGHT_COUNTRY", cls.risk_weight_country,
            ),
            risk_weight_commodity=_float(
                "RISK_WEIGHT_COMMODITY", cls.risk_weight_commodity,
            ),
            risk_weight_supplier=_float(
                "RISK_WEIGHT_SUPPLIER", cls.risk_weight_supplier,
            ),
            risk_weight_deforestation=_float(
                "RISK_WEIGHT_DEFORESTATION", cls.risk_weight_deforestation,
            ),
            # Graph capacity
            max_nodes_per_graph=_int(
                "MAX_NODES_PER_GRAPH", cls.max_nodes_per_graph,
            ),
            max_edges_per_graph=_int(
                "MAX_EDGES_PER_GRAPH", cls.max_edges_per_graph,
            ),
            max_tier_depth=_int("MAX_TIER_DEPTH", cls.max_tier_depth),
            # Performance
            graph_query_timeout_ms=_int(
                "GRAPH_QUERY_TIMEOUT_MS", cls.graph_query_timeout_ms,
            ),
            batch_throughput_target=_int(
                "BATCH_THROUGHPUT_TARGET", cls.batch_throughput_target,
            ),
            memory_limit_mb=_int("MEMORY_LIMIT_MB", cls.memory_limit_mb),
            # Risk thresholds
            risk_high_threshold=_float(
                "RISK_HIGH_THRESHOLD", cls.risk_high_threshold,
            ),
            risk_low_threshold=_float(
                "RISK_LOW_THRESHOLD", cls.risk_low_threshold,
            ),
            # Gap analysis
            mass_balance_tolerance=_float(
                "MASS_BALANCE_TOLERANCE", cls.mass_balance_tolerance,
            ),
            stale_data_days=_int(
                "STALE_DATA_DAYS", cls.stale_data_days,
            ),
            # Provenance
            enable_provenance=_bool(
                "ENABLE_PROVENANCE", cls.enable_provenance,
            ),
            genesis_hash=_str("GENESIS_HASH", cls.genesis_hash),
            # Metrics
            enable_metrics=_bool("ENABLE_METRICS", cls.enable_metrics),
            # Performance tuning
            pool_size=_int("POOL_SIZE", cls.pool_size),
            cache_ttl=_int("CACHE_TTL", cls.cache_ttl),
            rate_limit=_int("RATE_LIMIT", cls.rate_limit),
        )

        logger.info(
            "SupplyChainMapperConfig loaded: "
            "risk_weights=[%.2f, %.2f, %.2f, %.2f], "
            "max_nodes=%d, max_edges=%d, max_tier_depth=%d, "
            "query_timeout=%dms, risk_thresholds=[%.1f, %.1f], "
            "mass_balance_tolerance=%.1f%%, stale_data_days=%d, "
            "provenance=%s, pool=%d, cache_ttl=%ds, "
            "rate_limit=%d/min, metrics=%s",
            config.risk_weight_country,
            config.risk_weight_commodity,
            config.risk_weight_supplier,
            config.risk_weight_deforestation,
            config.max_nodes_per_graph,
            config.max_edges_per_graph,
            config.max_tier_depth,
            config.graph_query_timeout_ms,
            config.risk_low_threshold,
            config.risk_high_threshold,
            config.mass_balance_tolerance,
            config.stale_data_days,
            config.enable_provenance,
            config.pool_size,
            config.cache_ttl,
            config.rate_limit,
            config.enable_metrics,
        )
        return config

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def risk_weights(self) -> Dict[str, float]:
        """Return risk weights as a dictionary for the risk propagation engine.

        Returns:
            Dictionary with keys: country, commodity, supplier, deforestation.
        """
        return {
            "country": self.risk_weight_country,
            "commodity": self.risk_weight_commodity,
            "supplier": self.risk_weight_supplier,
            "deforestation": self.risk_weight_deforestation,
        }

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the configuration to a plain Python dictionary.

        Sensitive connection strings (database_url, redis_url) are
        redacted to prevent accidental credential leakage in logs,
        exception tracebacks, and monitoring dashboards.

        Returns:
            Dictionary representation with sensitive fields redacted.
        """
        return {
            # Connections (redacted)
            "database_url": "***" if self.database_url else "",
            "redis_url": "***" if self.redis_url else "",
            # Logging
            "log_level": self.log_level,
            # Risk weights
            "risk_weight_country": self.risk_weight_country,
            "risk_weight_commodity": self.risk_weight_commodity,
            "risk_weight_supplier": self.risk_weight_supplier,
            "risk_weight_deforestation": self.risk_weight_deforestation,
            # Graph capacity
            "max_nodes_per_graph": self.max_nodes_per_graph,
            "max_edges_per_graph": self.max_edges_per_graph,
            "max_tier_depth": self.max_tier_depth,
            # Performance
            "graph_query_timeout_ms": self.graph_query_timeout_ms,
            "batch_throughput_target": self.batch_throughput_target,
            "memory_limit_mb": self.memory_limit_mb,
            # Risk thresholds
            "risk_high_threshold": self.risk_high_threshold,
            "risk_low_threshold": self.risk_low_threshold,
            # Gap analysis
            "mass_balance_tolerance": self.mass_balance_tolerance,
            "stale_data_days": self.stale_data_days,
            # Provenance
            "enable_provenance": self.enable_provenance,
            "genesis_hash": self.genesis_hash,
            # Metrics
            "enable_metrics": self.enable_metrics,
            # Performance tuning
            "pool_size": self.pool_size,
            "cache_ttl": self.cache_ttl,
            "rate_limit": self.rate_limit,
        }

    def __repr__(self) -> str:
        """Return a developer-friendly, credential-safe representation.

        Returns:
            String representation with sensitive fields redacted.
        """
        d = self.to_dict()
        pairs = ", ".join(f"{k}={v!r}" for k, v in d.items())
        return f"SupplyChainMapperConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[SupplyChainMapperConfig] = None
_config_lock = threading.Lock()


def get_config() -> SupplyChainMapperConfig:
    """Return the singleton SupplyChainMapperConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path. The instance is created on first call
    by reading all ``GL_EUDR_SCM_*`` environment variables.

    Returns:
        SupplyChainMapperConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.risk_weight_country
        0.3
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = SupplyChainMapperConfig.from_env()
    return _config_instance


def set_config(config: SupplyChainMapperConfig) -> None:
    """Replace the singleton SupplyChainMapperConfig.

    Primarily intended for testing and dependency injection.

    Args:
        config: New SupplyChainMapperConfig to install.

    Example:
        >>> cfg = SupplyChainMapperConfig(risk_weight_country=0.4,
        ...     risk_weight_commodity=0.2, risk_weight_supplier=0.2,
        ...     risk_weight_deforestation=0.2)
        >>> set_config(cfg)
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info(
        "SupplyChainMapperConfig replaced programmatically: "
        "risk_weights=[%.2f, %.2f, %.2f, %.2f], "
        "max_nodes=%d, max_tier_depth=%d",
        config.risk_weight_country,
        config.risk_weight_commodity,
        config.risk_weight_supplier,
        config.risk_weight_deforestation,
        config.max_nodes_per_graph,
        config.max_tier_depth,
    )


def reset_config() -> None:
    """Reset the singleton SupplyChainMapperConfig to None.

    The next call to get_config() will re-read GL_EUDR_SCM_* env vars
    and construct a fresh instance. Intended for test teardown.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads env vars
    """
    global _config_instance
    with _config_lock:
        _config_instance = None
    logger.debug("SupplyChainMapperConfig singleton reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "SupplyChainMapperConfig",
    "get_config",
    "set_config",
    "reset_config",
]
