# -*- coding: utf-8 -*-
"""
Orchestrator Configuration - AGENT-FOUND-001: GreenLang DAG Orchestrator

Centralized configuration for the DAG execution engine covering:
- Max parallel node execution
- Default retry and timeout policies
- Checkpoint strategy and storage
- Provenance and determinism toggles
- Prometheus metrics toggle
- Logging level

All settings can be overridden via environment variables with the
``GL_ORCHESTRATOR_`` prefix (e.g. ``GL_ORCHESTRATOR_MAX_PARALLEL_NODES``).

Example:
    >>> from greenlang.orchestrator.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.max_parallel_nodes, cfg.default_retry_max)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-001 GreenLang Orchestrator
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

_ENV_PREFIX = "GL_ORCHESTRATOR_"


# ---------------------------------------------------------------------------
# OrchestratorConfig
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorConfig:
    """Complete configuration for the GreenLang DAG Orchestrator.

    Attributes are grouped by concern: execution limits, retry defaults,
    timeout defaults, checkpointing, provenance, determinism, metrics,
    and logging.

    All attributes can be overridden via environment variables using the
    ``GL_ORCHESTRATOR_`` prefix.

    Attributes:
        max_parallel_nodes: Maximum nodes to execute concurrently per level.
        default_retry_max: Default maximum retry attempts per node.
        default_retry_delay: Default base delay in seconds between retries.
        default_retry_max_delay: Default maximum delay between retries.
        default_timeout: Default timeout per node in seconds.
        checkpoint_strategy: Storage backend for checkpoints
            (memory, file, database).
        checkpoint_dir: Directory for file-based checkpoints.
        db_connection_string: Connection string for database checkpoints.
        enable_provenance: Enable SHA-256 provenance chain tracking.
        enable_determinism: Enable deterministic scheduling and timestamps.
        enable_metrics: Enable Prometheus metric recording.
        log_level: Logging level for orchestrator components.
    """

    # -- Execution limits ---------------------------------------------------
    max_parallel_nodes: int = 10

    # -- Default retry policy -----------------------------------------------
    default_retry_max: int = 3
    default_retry_delay: float = 1.0
    default_retry_max_delay: float = 60.0

    # -- Default timeout policy ---------------------------------------------
    default_timeout: float = 60.0

    # -- Checkpoint storage -------------------------------------------------
    checkpoint_strategy: str = "memory"
    checkpoint_dir: str = "/tmp/greenlang/orchestrator/checkpoints"
    db_connection_string: str = ""

    # -- Provenance ---------------------------------------------------------
    enable_provenance: bool = True

    # -- Determinism --------------------------------------------------------
    enable_determinism: bool = True

    # -- Metrics ------------------------------------------------------------
    enable_metrics: bool = True

    # -- Logging ------------------------------------------------------------
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> OrchestratorConfig:
        """Build an OrchestratorConfig from environment variables.

        Every field can be overridden via ``GL_ORCHESTRATOR_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.

        Returns:
            Populated OrchestratorConfig instance.
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
                    "Invalid float for %s%s=%s, using default %.1f",
                    prefix, name, val, default,
                )
                return default

        config = cls(
            max_parallel_nodes=_int("MAX_PARALLEL_NODES", cls.max_parallel_nodes),
            default_retry_max=_int("DEFAULT_RETRY_MAX", cls.default_retry_max),
            default_retry_delay=_float("DEFAULT_RETRY_DELAY", cls.default_retry_delay),
            default_retry_max_delay=_float(
                "DEFAULT_RETRY_MAX_DELAY", cls.default_retry_max_delay,
            ),
            default_timeout=_float("DEFAULT_TIMEOUT", cls.default_timeout),
            checkpoint_strategy=(
                _env("CHECKPOINT_STRATEGY", cls.checkpoint_strategy)
                or cls.checkpoint_strategy
            ),
            checkpoint_dir=(
                _env("CHECKPOINT_DIR", cls.checkpoint_dir) or cls.checkpoint_dir
            ),
            db_connection_string=(
                _env("DB_CONNECTION_STRING", cls.db_connection_string)
                or cls.db_connection_string
            ),
            enable_provenance=_bool("ENABLE_PROVENANCE", cls.enable_provenance),
            enable_determinism=_bool("ENABLE_DETERMINISM", cls.enable_determinism),
            enable_metrics=_bool("ENABLE_METRICS", cls.enable_metrics),
            log_level=(
                _env("LOG_LEVEL", cls.log_level) or cls.log_level
            ),
        )

        logger.info(
            "OrchestratorConfig loaded: max_parallel=%d, checkpoint=%s, "
            "provenance=%s, determinism=%s, metrics=%s",
            config.max_parallel_nodes,
            config.checkpoint_strategy,
            config.enable_provenance,
            config.enable_determinism,
            config.enable_metrics,
        )
        return config


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[OrchestratorConfig] = None
_config_lock = threading.Lock()


def get_config() -> OrchestratorConfig:
    """Return the singleton OrchestratorConfig, creating from env if needed.

    Returns:
        OrchestratorConfig singleton instance.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = OrchestratorConfig.from_env()
    return _config_instance


def set_config(config: OrchestratorConfig) -> None:
    """Replace the singleton OrchestratorConfig (useful for testing).

    Args:
        config: New configuration to install.
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info("OrchestratorConfig replaced programmatically")


def reset_config() -> None:
    """Reset the singleton (primarily for test teardown)."""
    global _config_instance
    with _config_lock:
        _config_instance = None


__all__ = [
    "OrchestratorConfig",
    "get_config",
    "set_config",
    "reset_config",
]
