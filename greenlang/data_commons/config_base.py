# -*- coding: utf-8 -*-
"""
Base Data Configuration - Shared config infrastructure for GreenLang data agents.

This module eliminates ~200 lines of identical boilerplate from each of the 20
data-layer agent ``config.py`` files by providing:

1.  **BaseDataConfig** -- a ``@dataclass`` with the 8 fields that appear in
    every agent config (connections, pool sizing, batch defaults, logging).
2.  **EnvReader** -- a small helper that wraps the ``_env / _bool / _int /
    _float / _str`` closures that were copy-pasted identically in every
    ``from_env()`` classmethod.
3.  **create_config_singleton** -- a factory that returns the thread-safe
    ``(get_config, set_config, reset_config)`` triple that was duplicated
    verbatim across every config module.

Typical usage in an agent's ``config.py``::

    from dataclasses import dataclass
    from greenlang.data_commons.config_base import (
        BaseDataConfig, EnvReader, create_config_singleton,
    )

    _ENV_PREFIX = "GL_MY_AGENT_"

    @dataclass
    class MyAgentConfig(BaseDataConfig):
        # Agent-specific fields only
        my_custom_field: str = "default"

        @classmethod
        def from_env(cls) -> "MyAgentConfig":
            env = EnvReader(_ENV_PREFIX)
            base_kwargs = cls._base_kwargs_from_env(env)
            return cls(
                **base_kwargs,
                my_custom_field=env.str("MY_CUSTOM_FIELD", cls.my_custom_field),
            )

    get_config, set_config, reset_config = create_config_singleton(
        MyAgentConfig, _ENV_PREFIX,
    )

Author: GreenLang Platform Team
Date: April 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, fields as dc_fields
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

logger = logging.getLogger(__name__)

# Generic type variable bound to BaseDataConfig (or any subclass).
_C = TypeVar("_C", bound="BaseDataConfig")


# ---------------------------------------------------------------------------
# EnvReader -- typed environment variable reader
# ---------------------------------------------------------------------------


class EnvReader:
    """Read environment variables with a fixed prefix and typed conversions.

    This replaces the identical ``_env / _bool / _int / _float / _str``
    closures that were duplicated in every agent ``from_env()`` method.

    Args:
        prefix: The environment variable prefix, e.g. ``"GL_PDF_EXTRACTOR_"``.

    Example:
        >>> env = EnvReader("GL_PDF_EXTRACTOR_")
        >>> env.str("DATABASE_URL", "")
        ''
        >>> env.int("POOL_MIN_SIZE", 2)
        2
    """

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix

    # -- raw lookup --------------------------------------------------------

    def raw(self, name: str, default: Any = None) -> Optional[str]:
        """Return the raw string value of ``{prefix}{name}``, or *default*.

        Args:
            name: Suffix after the prefix (e.g. ``"DATABASE_URL"``).
            default: Fallback when the variable is not set.

        Returns:
            The environment variable value, or *default*.
        """
        return os.environ.get(f"{self._prefix}{name}", default)

    # -- typed helpers -----------------------------------------------------

    def str(self, name: str, default: str) -> str:
        """Read a string variable, falling back to *default*.

        Args:
            name: Variable suffix.
            default: Fallback value.

        Returns:
            The string value.
        """
        val = self.raw(name)
        if val is None:
            return default
        return val

    def bool(self, name: str, default: bool) -> bool:
        """Read a boolean variable (accepts ``true/1/yes``, case-insensitive).

        Args:
            name: Variable suffix.
            default: Fallback value.

        Returns:
            The parsed boolean.
        """
        val = self.raw(name)
        if val is None:
            return default
        return val.lower() in ("true", "1", "yes")

    def int(self, name: str, default: int) -> int:
        """Read an integer variable, logging a warning on parse failure.

        Args:
            name: Variable suffix.
            default: Fallback value.

        Returns:
            The parsed integer, or *default* on failure.
        """
        val = self.raw(name)
        if val is None:
            return default
        try:
            return int(val)
        except ValueError:
            logger.warning(
                "Invalid integer for %s%s=%s, using default %d",
                self._prefix,
                name,
                val,
                default,
            )
            return default

    def float(self, name: str, default: float) -> float:
        """Read a float variable, logging a warning on parse failure.

        Args:
            name: Variable suffix.
            default: Fallback value.

        Returns:
            The parsed float, or *default* on failure.
        """
        val = self.raw(name)
        if val is None:
            return default
        try:
            return float(val)
        except ValueError:
            logger.warning(
                "Invalid float for %s%s=%s, using default %f",
                self._prefix,
                name,
                val,
                default,
            )
            return default


# ---------------------------------------------------------------------------
# BaseDataConfig -- shared fields across all 20 data agents
# ---------------------------------------------------------------------------


@dataclass
class BaseDataConfig:
    """Base configuration shared by every GreenLang data-layer agent.

    These 8 fields appear -- with identical names, types, and defaults -- in
    all 20 ``config.py`` files under ``greenlang/agents/data/``.  Subclasses
    only need to declare agent-specific fields.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage.
        redis_url: Redis connection URL for caching layer.
        s3_bucket_url: S3 bucket URL for object/document storage.
        pool_min_size: Minimum connection pool size.
        pool_max_size: Maximum connection pool size.
        batch_max_items: Maximum items in a single batch job.
        batch_worker_count: Number of parallel workers for batch processing.
        log_level: Logging level for the agent service.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = ""
    redis_url: str = ""
    s3_bucket_url: str = ""

    # -- Pool sizing ---------------------------------------------------------
    pool_min_size: int = 2
    pool_max_size: int = 10

    # -- Batch processing ----------------------------------------------------
    batch_max_items: int = 100
    batch_worker_count: int = 4

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Helpers for subclass ``from_env()`` methods
    # ------------------------------------------------------------------

    @classmethod
    def _base_kwargs_from_env(cls, env: EnvReader) -> Dict[str, Any]:
        """Return a dict of all base-class field values read from the env.

        Subclasses call this inside their ``from_env()`` classmethod and
        unpack the result into the constructor together with their own
        agent-specific kwargs.

        Args:
            env: An EnvReader bound to the agent's prefix.

        Returns:
            Dict mapping base field names to their env-resolved values.

        Example:
            >>> env = EnvReader("GL_MY_AGENT_")
            >>> base = MyConfig._base_kwargs_from_env(env)
            >>> config = MyConfig(**base, my_field=env.str("MY_FIELD", "x"))
        """
        return {
            "database_url": env.str("DATABASE_URL", cls.database_url),
            "redis_url": env.str("REDIS_URL", cls.redis_url),
            "s3_bucket_url": env.str("S3_BUCKET_URL", cls.s3_bucket_url),
            "pool_min_size": env.int("POOL_MIN_SIZE", cls.pool_min_size),
            "pool_max_size": env.int("POOL_MAX_SIZE", cls.pool_max_size),
            "batch_max_items": env.int("BATCH_MAX_ITEMS", cls.batch_max_items),
            "batch_worker_count": env.int(
                "BATCH_WORKER_COUNT", cls.batch_worker_count,
            ),
            "log_level": env.str("LOG_LEVEL", cls.log_level),
        }


# ---------------------------------------------------------------------------
# Thread-safe singleton factory
# ---------------------------------------------------------------------------


def create_config_singleton(
    config_cls: Type[_C],
    env_prefix: str,
) -> Tuple[Callable[[], _C], Callable[[_C], None], Callable[[], None]]:
    """Create a thread-safe get/set/reset singleton triple for *config_cls*.

    This replaces the ~35 lines of boilerplate (``_config_instance``,
    ``_config_lock``, ``get_config()``, ``set_config()``, ``reset_config()``)
    duplicated verbatim in every data agent ``config.py``.

    Args:
        config_cls: The ``@dataclass`` config class (must have ``from_env``).
        env_prefix: The ``GL_...`` prefix (passed for logging only; the
            class's own ``from_env`` already knows its prefix).

    Returns:
        A ``(get_config, set_config, reset_config)`` tuple of callables.

    Example:
        >>> get_config, set_config, reset_config = create_config_singleton(
        ...     PDFExtractorConfig, "GL_PDF_EXTRACTOR_",
        ... )
        >>> cfg = get_config()
    """
    lock = threading.Lock()
    # Use a mutable container so the nested functions can rebind the value.
    holder: Dict[str, Optional[_C]] = {"instance": None}
    class_name = config_cls.__name__

    def get_config() -> _C:
        """Return the singleton config, creating from env if needed.

        Returns:
            The singleton config instance.
        """
        if holder["instance"] is None:
            with lock:
                if holder["instance"] is None:
                    holder["instance"] = config_cls.from_env()
        return holder["instance"]  # type: ignore[return-value]

    def set_config(config: _C) -> None:
        """Replace the singleton config (useful for testing).

        Args:
            config: New configuration to install.
        """
        with lock:
            holder["instance"] = config
        logger.info("%s replaced programmatically", class_name)

    def reset_config() -> None:
        """Reset the singleton (primarily for test teardown)."""
        with lock:
            holder["instance"] = None

    # Attach useful metadata for introspection / debugging.
    get_config.__qualname__ = f"get_config[{class_name}]"
    set_config.__qualname__ = f"set_config[{class_name}]"
    reset_config.__qualname__ = f"reset_config[{class_name}]"

    return get_config, set_config, reset_config


__all__ = [
    "BaseDataConfig",
    "EnvReader",
    "create_config_singleton",
]
