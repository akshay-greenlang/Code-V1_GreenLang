"""
Legacy Agent Registrar - Agent Factory (INFRA-010)

Registers discovered legacy agents in the Agent Factory's infrastructure
agent registry.  Each agent receives a synthetic registration record with
``version="0.1.0"``, ``status="running"``, and metadata marking it as
legacy.  Registration is idempotent -- already-registered agents are
silently skipped.

Classes:
    - RegistrationReport: Summary of a batch registration operation.
    - LegacyRegistrar: Registers discovered agents in the factory registry.

Example:
    >>> registrar = LegacyRegistrar(db_pool, redis_client)
    >>> report = await registrar.register_all(discovered_agents)
    >>> print(report.newly_registered, report.already_registered)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.agent_factory.legacy.discovery import (
    DiscoveredAgent,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_VERSION = "0.1.0"
_DEFAULT_STATUS = "running"

_UPSERT_SQL = """\
INSERT INTO infrastructure.agent_registry (
    agent_key, version, status, module_path, class_name,
    base_class, metadata, registered_at
) VALUES (
    $1, $2, $3, $4, $5, $6, $7::jsonb, $8
)
ON CONFLICT (agent_key) DO UPDATE SET
    module_path  = EXCLUDED.module_path,
    class_name   = EXCLUDED.class_name,
    base_class   = EXCLUDED.base_class,
    metadata     = infrastructure.agent_registry.metadata || EXCLUDED.metadata,
    registered_at = EXCLUDED.registered_at
WHERE infrastructure.agent_registry.metadata->>'legacy' = 'true'
"""

_CHECK_EXISTS_SQL = """\
SELECT 1 FROM infrastructure.agent_registry WHERE agent_key = $1 LIMIT 1
"""

_COUNT_ALL_SQL = """\
SELECT COUNT(*) FROM infrastructure.agent_registry
WHERE metadata->>'legacy' = 'true'
"""

# Redis key for a lightweight lookup cache
_REDIS_REGISTRY_SET = "gl:agent_factory:registered_keys"


# ---------------------------------------------------------------------------
# RegistrationReport
# ---------------------------------------------------------------------------


@dataclass
class RegistrationReport:
    """Summary of a batch legacy agent registration operation.

    Attributes:
        total_discovered: Total agents passed for registration.
        newly_registered: Number of agents successfully registered for
            the first time.
        already_registered: Number of agents that were already present
            in the registry and skipped.
        failed: Number of agents whose registration raised an error.
        errors: Human-readable error messages for failed registrations.
        duration_ms: Wall-clock time of the batch operation.
    """

    total_discovered: int = 0
    newly_registered: int = 0
    already_registered: int = 0
    failed: int = 0
    errors: List[str] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        """Percentage of successful registrations (0.0 - 100.0)."""
        if self.total_discovered == 0:
            return 100.0
        successful = self.newly_registered + self.already_registered
        return (successful / self.total_discovered) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the report to a plain dictionary."""
        return {
            "total_discovered": self.total_discovered,
            "newly_registered": self.newly_registered,
            "already_registered": self.already_registered,
            "failed": self.failed,
            "errors": self.errors,
            "duration_ms": round(self.duration_ms, 2),
            "success_rate": round(self.success_rate, 2),
        }


# ---------------------------------------------------------------------------
# LegacyRegistrar
# ---------------------------------------------------------------------------


class LegacyRegistrar:
    """Registers legacy agents in the infrastructure.agent_registry table.

    Provides idempotent registration: agents already present in the
    registry are silently skipped.  A lightweight Redis set is maintained
    to avoid repeated database round-trips for existence checks.

    Attributes:
        _pool: Async PostgreSQL connection pool.
        _redis: Async Redis client (optional; disables cache if None).
    """

    def __init__(
        self,
        db_pool: Any,
        redis_client: Optional[Any] = None,
    ) -> None:
        """Initialize the legacy registrar.

        Args:
            db_pool: Async PostgreSQL connection pool.  Must support
                ``async with pool.connection() as conn`` (psycopg) or
                ``async with pool.acquire() as conn`` (asyncpg).
            redis_client: Optional async Redis client for caching the
                set of registered agent keys.
        """
        self._pool = db_pool
        self._redis = redis_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def register_all(
        self,
        agents: List[DiscoveredAgent],
    ) -> RegistrationReport:
        """Register all discovered agents.  Idempotent -- skips existing.

        Args:
            agents: List of DiscoveredAgent from the discovery scanner.

        Returns:
            RegistrationReport summarising the operation.
        """
        report = RegistrationReport(total_discovered=len(agents))
        start = time.perf_counter()

        logger.info(
            "LegacyRegistrar: registering %d discovered agents", len(agents),
        )

        for agent in agents:
            try:
                already = await self.is_registered(agent.agent_key)
                if already:
                    report.already_registered += 1
                    continue

                success = await self.register_one(agent)
                if success:
                    report.newly_registered += 1
                else:
                    report.failed += 1
                    report.errors.append(
                        f"{agent.agent_key}: registration returned False"
                    )
            except Exception as exc:
                report.failed += 1
                msg = f"{agent.agent_key}: {exc}"
                report.errors.append(msg)
                logger.error("LegacyRegistrar: failed to register %s: %s",
                             agent.agent_key, exc)

        report.duration_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "LegacyRegistrar: completed (new=%d, existing=%d, failed=%d, "
            "%.1fms)",
            report.newly_registered,
            report.already_registered,
            report.failed,
            report.duration_ms,
        )

        return report

    async def register_one(self, agent: DiscoveredAgent) -> bool:
        """Register a single legacy agent with synthetic metadata.

        Sets ``version="0.1.0"``, ``status="running"``, and attaches
        legacy-specific metadata.

        Args:
            agent: The DiscoveredAgent to register.

        Returns:
            True if the agent was successfully registered.
        """
        metadata = {
            "legacy": True,
            "migrated": False,
            "base_class": agent.base_class,
            "has_pack_yaml": agent.has_pack_yaml,
            "has_tests": agent.has_tests,
            "source_file": str(agent.file_path),
            "discovered_at": datetime.now(timezone.utc).isoformat(),
        }

        if agent.input_type_hint:
            metadata["input_type_hint"] = agent.input_type_hint
        if agent.output_type_hint:
            metadata["output_type_hint"] = agent.output_type_hint

        now = datetime.now(timezone.utc)
        metadata_json = json.dumps(metadata, default=str)

        try:
            conn = await self._acquire_connection()
            try:
                await conn.execute(
                    _UPSERT_SQL,
                    agent.agent_key,
                    _DEFAULT_VERSION,
                    _DEFAULT_STATUS,
                    agent.module_path,
                    agent.class_name,
                    agent.base_class,
                    metadata_json,
                    now,
                )
            finally:
                await self._release_connection(conn)

            # Update the Redis cache
            await self._cache_add(agent.agent_key)

            logger.debug(
                "LegacyRegistrar: registered %s (%s.%s)",
                agent.agent_key,
                agent.module_path,
                agent.class_name,
            )
            return True

        except Exception as exc:
            logger.error(
                "LegacyRegistrar: DB write failed for %s: %s",
                agent.agent_key,
                exc,
            )
            raise

    async def is_registered(self, agent_key: str) -> bool:
        """Check if an agent is already registered in the factory.

        Uses the Redis cache first; falls back to a database query.

        Args:
            agent_key: The agent key to check.

        Returns:
            True if the agent is already registered.
        """
        # Check Redis cache first
        if self._redis is not None:
            try:
                cached = await self._redis.sismember(
                    _REDIS_REGISTRY_SET, agent_key,
                )
                if cached:
                    return True
            except Exception as exc:
                logger.debug(
                    "LegacyRegistrar: Redis cache miss for %s: %s",
                    agent_key, exc,
                )

        # Fall back to database
        try:
            conn = await self._acquire_connection()
            try:
                row = await conn.fetchrow(_CHECK_EXISTS_SQL, agent_key)
            finally:
                await self._release_connection(conn)

            exists = row is not None
            if exists:
                await self._cache_add(agent_key)
            return exists

        except Exception as exc:
            logger.error(
                "LegacyRegistrar: DB check failed for %s: %s",
                agent_key, exc,
            )
            return False

    async def count_legacy_agents(self) -> int:
        """Return the count of registered legacy agents.

        Returns:
            Number of agents with ``metadata.legacy = true``.
        """
        try:
            conn = await self._acquire_connection()
            try:
                row = await conn.fetchrow(_COUNT_ALL_SQL)
            finally:
                await self._release_connection(conn)
            return row[0] if row else 0
        except Exception as exc:
            logger.error(
                "LegacyRegistrar: count query failed: %s", exc,
            )
            return 0

    # ------------------------------------------------------------------
    # Internal: database helpers
    # ------------------------------------------------------------------

    async def _acquire_connection(self) -> Any:
        """Acquire a connection from the pool.

        Supports both psycopg (``connection()``) and asyncpg
        (``acquire()``) patterns.

        Returns:
            An async database connection.

        Raises:
            TypeError: If the pool type is not recognised.
        """
        if hasattr(self._pool, "connection"):
            ctx = self._pool.connection()
            conn = await ctx.__aenter__()
            conn._ctx = ctx  # type: ignore[attr-defined]
            return conn
        elif hasattr(self._pool, "acquire"):
            return await self._pool.acquire()
        else:
            raise TypeError(
                f"Unsupported pool type: {type(self._pool).__name__}"
            )

    async def _release_connection(self, conn: Any) -> None:
        """Release a connection back to the pool."""
        if hasattr(conn, "_ctx"):
            ctx = conn._ctx
            await ctx.__aexit__(None, None, None)
        elif hasattr(self._pool, "release"):
            await self._pool.release(conn)

    # ------------------------------------------------------------------
    # Internal: Redis cache helpers
    # ------------------------------------------------------------------

    async def _cache_add(self, agent_key: str) -> None:
        """Add an agent key to the Redis registered-keys set."""
        if self._redis is None:
            return
        try:
            await self._redis.sadd(_REDIS_REGISTRY_SET, agent_key)
        except Exception as exc:
            logger.debug(
                "LegacyRegistrar: Redis SADD failed for %s: %s",
                agent_key, exc,
            )


__all__ = [
    "LegacyRegistrar",
    "RegistrationReport",
]
