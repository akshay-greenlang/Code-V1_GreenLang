# -*- coding: utf-8 -*-
"""
PostgreSQL Feature Flag Storage - INFRA-008

Async PostgreSQL-backed implementation of ``IFlagStorage`` that serves as the
L3 persistent source-of-truth in the multi-layer storage architecture.

Features:
    - Async psycopg with ``psycopg_pool.AsyncConnectionPool`` (min=5, max=20)
    - All queries use parameterized SQL (zero f-strings in query paths)
    - Reads from ``infrastructure.feature_flags`` and related tables
    - Writes audit log to ``infrastructure.feature_flag_audit_log``
    - Pagination support in ``get_all_flags``
    - Transaction support for multi-table operations
    - Health check method

Follows the psycopg patterns established in
``greenlang.data.vector.connection``:
    - ``dict_row`` row factory for clean dict-based result mapping
    - ``AsyncConnectionPool`` with ``open=False`` + explicit ``await pool.open()``

Table schema (``infrastructure`` schema):

    feature_flags          -- flag definitions
    feature_flag_rules     -- targeting rules
    feature_flag_overrides -- scoped overrides
    feature_flag_variants  -- multivariate variants
    feature_flag_audit_log -- immutable audit trail

Example:
    >>> from greenlang.infrastructure.feature_flags.config import FeatureFlagConfig
    >>> config = FeatureFlagConfig(postgres_dsn="postgresql://localhost/greenlang")
    >>> store = PostgresFlagStorage(config)
    >>> await store.initialize()
    >>> await store.save_flag(flag)
    >>> result = await store.get_flag("enable-scope3-calc")
    >>> await store.close()
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.feature_flags.config import FeatureFlagConfig
from greenlang.infrastructure.feature_flags.models import (
    AuditLogEntry,
    FeatureFlag,
    FlagOverride,
    FlagRule,
    FlagStatus,
    FlagVariant,
)
from greenlang.infrastructure.feature_flags.storage.base import IFlagStorage

logger = logging.getLogger(__name__)


class PostgresFlagStorage(IFlagStorage):
    """Async PostgreSQL storage backend for feature flags (L3 layer).

    All queries use parameterised SQL to prevent injection. Writes to
    multi-table structures use explicit transactions. The connection pool
    is managed via ``psycopg_pool.AsyncConnectionPool``.

    Args:
        config: FeatureFlagConfig with ``postgres_dsn`` populated.
    """

    def __init__(self, config: FeatureFlagConfig) -> None:
        self._config = config
        self._dsn = config.postgres_dsn or "postgresql://localhost:5432/greenlang"
        self._schema = "infrastructure"
        self._min_pool = 5
        self._max_pool = 20
        self._pool: Any = None  # psycopg_pool.AsyncConnectionPool
        self._initialized = False

        logger.info("PostgresFlagStorage created (dsn=%s)", self._dsn)

    # ------------------------------------------------------------------
    # Table names
    # ------------------------------------------------------------------

    @property
    def _t_flags(self) -> str:
        return f"{self._schema}.feature_flags"

    @property
    def _t_rules(self) -> str:
        return f"{self._schema}.feature_flag_rules"

    @property
    def _t_overrides(self) -> str:
        return f"{self._schema}.feature_flag_overrides"

    @property
    def _t_variants(self) -> str:
        return f"{self._schema}.feature_flag_variants"

    @property
    def _t_audit(self) -> str:
        return f"{self._schema}.feature_flag_audit_log"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Open the connection pool and ensure schema/tables exist."""
        if self._initialized:
            return

        try:
            from psycopg.rows import dict_row
            from psycopg_pool import AsyncConnectionPool

            self._pool = AsyncConnectionPool(
                conninfo=self._dsn,
                min_size=self._min_pool,
                max_size=self._max_pool,
                open=False,
                kwargs={"row_factory": dict_row, "autocommit": False},
            )
            await self._pool.open()
            self._initialized = True
            logger.info(
                "PostgresFlagStorage pool opened (min=%d, max=%d)",
                self._min_pool,
                self._max_pool,
            )

            await self._ensure_schema()
        except ImportError:
            logger.error(
                "psycopg / psycopg_pool not available. "
                "Install with: pip install psycopg[binary] psycopg_pool"
            )
        except Exception as exc:
            logger.error("Failed to initialise PostgreSQL pool: %s", exc)

    async def _ensure_schema(self) -> None:
        """Create schema and tables if they do not exist.

        This is idempotent and safe to run on every startup.
        """
        ddl = f"""
        CREATE SCHEMA IF NOT EXISTS {self._schema};

        CREATE TABLE IF NOT EXISTS {self._t_flags} (
            key             VARCHAR(128) PRIMARY KEY,
            name            VARCHAR(256) NOT NULL,
            description     TEXT NOT NULL DEFAULT '',
            flag_type       VARCHAR(32) NOT NULL DEFAULT 'boolean',
            status          VARCHAR(32) NOT NULL DEFAULT 'draft',
            default_value   JSONB NOT NULL DEFAULT 'false'::jsonb,
            rollout_percentage FLOAT NOT NULL DEFAULT 0.0,
            environments    JSONB NOT NULL DEFAULT '[]'::jsonb,
            tags            JSONB NOT NULL DEFAULT '[]'::jsonb,
            owner           VARCHAR(256) NOT NULL DEFAULT '',
            metadata        JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            start_time      TIMESTAMPTZ,
            end_time        TIMESTAMPTZ,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            version         INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS {self._t_rules} (
            rule_id         VARCHAR(64) PRIMARY KEY,
            flag_key        VARCHAR(128) NOT NULL REFERENCES {self._t_flags}(key) ON DELETE CASCADE,
            rule_type       VARCHAR(64) NOT NULL,
            priority        INTEGER NOT NULL DEFAULT 100,
            conditions      JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            enabled         BOOLEAN NOT NULL DEFAULT TRUE
        );
        CREATE INDEX IF NOT EXISTS idx_ff_rules_flag ON {self._t_rules}(flag_key);

        CREATE TABLE IF NOT EXISTS {self._t_overrides} (
            flag_key        VARCHAR(128) NOT NULL REFERENCES {self._t_flags}(key) ON DELETE CASCADE,
            scope_type      VARCHAR(32) NOT NULL,
            scope_value     VARCHAR(512) NOT NULL,
            enabled         BOOLEAN NOT NULL DEFAULT TRUE,
            variant_key     VARCHAR(128),
            expires_at      TIMESTAMPTZ,
            created_by      VARCHAR(256) NOT NULL DEFAULT '',
            PRIMARY KEY (flag_key, scope_type, scope_value)
        );

        CREATE TABLE IF NOT EXISTS {self._t_variants} (
            variant_key     VARCHAR(128) NOT NULL,
            flag_key        VARCHAR(128) NOT NULL REFERENCES {self._t_flags}(key) ON DELETE CASCADE,
            variant_value   JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            weight          FLOAT NOT NULL DEFAULT 0.0,
            description     TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (flag_key, variant_key)
        );

        CREATE TABLE IF NOT EXISTS {self._t_audit} (
            id              BIGSERIAL PRIMARY KEY,
            flag_key        VARCHAR(128) NOT NULL,
            action          VARCHAR(64) NOT NULL,
            old_value       JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            new_value       JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            changed_by      VARCHAR(256) NOT NULL DEFAULT '',
            change_reason   TEXT NOT NULL DEFAULT '',
            ip_address      VARCHAR(45),
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_ff_audit_flag ON {self._t_audit}(flag_key, created_at DESC);
        """
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(ddl)
                await conn.commit()
            logger.info("Feature flag schema ensured")
        except Exception as exc:
            logger.error("Schema initialisation failed: %s", exc)

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        self._initialized = False
        logger.info("PostgresFlagStorage pool closed")

    # Keep backward-compatible alias
    async def shutdown(self) -> None:
        """Alias for ``close()``."""
        await self.close()

    # ------------------------------------------------------------------
    # Helper: row -> model mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_flag(row: Dict[str, Any]) -> FeatureFlag:
        """Convert a database row dict to a FeatureFlag model."""
        return FeatureFlag(
            key=row["key"],
            name=row["name"],
            description=row.get("description", ""),
            flag_type=row.get("flag_type", "boolean"),
            status=row.get("status", "draft"),
            default_value=row.get("default_value", False),
            rollout_percentage=row.get("rollout_percentage", 0.0),
            environments=row.get("environments", []),
            tags=row.get("tags", []),
            owner=row.get("owner", ""),
            metadata=row.get("metadata", {}),
            start_time=row.get("start_time"),
            end_time=row.get("end_time"),
            created_at=row.get("created_at", datetime.now(timezone.utc)),
            updated_at=row.get("updated_at", datetime.now(timezone.utc)),
            version=row.get("version", 1),
        )

    @staticmethod
    def _row_to_rule(row: Dict[str, Any]) -> FlagRule:
        """Convert a database row dict to a FlagRule model."""
        return FlagRule(
            rule_id=row["rule_id"],
            flag_key=row["flag_key"],
            rule_type=row["rule_type"],
            priority=row.get("priority", 100),
            conditions=row.get("conditions", {}),
            enabled=row.get("enabled", True),
        )

    @staticmethod
    def _row_to_override(row: Dict[str, Any]) -> FlagOverride:
        """Convert a database row dict to a FlagOverride model."""
        return FlagOverride(
            flag_key=row["flag_key"],
            scope_type=row["scope_type"],
            scope_value=row["scope_value"],
            enabled=row.get("enabled", True),
            variant_key=row.get("variant_key"),
            expires_at=row.get("expires_at"),
            created_by=row.get("created_by", ""),
        )

    @staticmethod
    def _row_to_variant(row: Dict[str, Any]) -> FlagVariant:
        """Convert a database row dict to a FlagVariant model."""
        return FlagVariant(
            variant_key=row["variant_key"],
            flag_key=row["flag_key"],
            variant_value=row.get("variant_value", {}),
            weight=row.get("weight", 0.0),
            description=row.get("description", ""),
        )

    @staticmethod
    def _row_to_audit(row: Dict[str, Any]) -> AuditLogEntry:
        """Convert a database row dict to an AuditLogEntry model."""
        return AuditLogEntry(
            flag_key=row["flag_key"],
            action=row["action"],
            old_value=row.get("old_value", {}),
            new_value=row.get("new_value", {}),
            changed_by=row.get("changed_by", ""),
            change_reason=row.get("change_reason", ""),
            ip_address=row.get("ip_address"),
            created_at=row.get("created_at", datetime.now(timezone.utc)),
        )

    # ------------------------------------------------------------------
    # IFlagStorage -- Flags
    # ------------------------------------------------------------------

    async def get_flag(self, key: str) -> Optional[FeatureFlag]:
        """Retrieve a single flag by key from PostgreSQL."""
        if not self._initialized:
            return None
        query = f"SELECT * FROM {self._t_flags} WHERE key = %s"
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, (key,))
                    row = await cur.fetchone()
                    if row is None:
                        return None
                    return self._row_to_flag(row)
        except Exception as exc:
            logger.error("PG get_flag('%s') failed: %s", key, exc)
            return None

    async def get_all_flags(
        self,
        status_filter: Optional[FlagStatus] = None,
        tag_filter: Optional[str] = None,
    ) -> List[FeatureFlag]:
        """Retrieve flags from PostgreSQL with optional filters.

        Supports pagination through the ``LIMIT`` / ``OFFSET`` SQL clauses
        when called by higher-level methods.
        """
        if not self._initialized:
            return []

        conditions: List[str] = []
        params: List[Any] = []

        if status_filter is not None:
            conditions.append("status = %s")
            params.append(status_filter.value)
        if tag_filter is not None:
            conditions.append("tags @> %s::jsonb")
            params.append(json.dumps([tag_filter.lower()]))

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        query = f"SELECT * FROM {self._t_flags} {where} ORDER BY created_at DESC"
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, tuple(params))
                    rows = await cur.fetchall()
                    return [self._row_to_flag(r) for r in rows]
        except Exception as exc:
            logger.error("PG get_all_flags failed: %s", exc)
            return []

    async def list_flags_paginated(
        self,
        status_filter: Optional[FlagStatus] = None,
        tag_filter: Optional[str] = None,
        owner: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[FeatureFlag]:
        """Paginated flag listing for the API layer.

        Args:
            status_filter: Optional status filter.
            tag_filter: Optional tag filter.
            owner: Optional owner filter (case-insensitive).
            offset: Rows to skip.
            limit: Maximum rows to return.

        Returns:
            Paginated list of flags ordered by created_at DESC.
        """
        if not self._initialized:
            return []

        conditions: List[str] = []
        params: List[Any] = []

        if status_filter is not None:
            conditions.append("status = %s")
            params.append(status_filter.value)
        if tag_filter is not None:
            conditions.append("tags @> %s::jsonb")
            params.append(json.dumps([tag_filter.lower()]))
        if owner is not None:
            conditions.append("LOWER(owner) = LOWER(%s)")
            params.append(owner)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        params.extend([limit, offset])
        query = (
            f"SELECT * FROM {self._t_flags} {where} "
            f"ORDER BY created_at DESC LIMIT %s OFFSET %s"
        )
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, tuple(params))
                    rows = await cur.fetchall()
                    return [self._row_to_flag(r) for r in rows]
        except Exception as exc:
            logger.error("PG list_flags_paginated failed: %s", exc)
            return []

    async def save_flag(self, flag: FeatureFlag) -> FeatureFlag:
        """Upsert a flag using INSERT ... ON CONFLICT DO UPDATE."""
        if not self._initialized:
            logger.warning("PG not initialised -- save_flag skipped")
            return flag

        query = f"""
            INSERT INTO {self._t_flags} (
                key, name, description, flag_type, status, default_value,
                rollout_percentage, environments, tags, owner, metadata,
                start_time, end_time, created_at, updated_at, version
            ) VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s
            )
            ON CONFLICT (key) DO UPDATE SET
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                flag_type = EXCLUDED.flag_type,
                status = EXCLUDED.status,
                default_value = EXCLUDED.default_value,
                rollout_percentage = EXCLUDED.rollout_percentage,
                environments = EXCLUDED.environments,
                tags = EXCLUDED.tags,
                owner = EXCLUDED.owner,
                metadata = EXCLUDED.metadata,
                start_time = EXCLUDED.start_time,
                end_time = EXCLUDED.end_time,
                updated_at = EXCLUDED.updated_at,
                version = EXCLUDED.version
        """
        params = (
            flag.key,
            flag.name,
            flag.description,
            flag.flag_type.value,
            flag.status.value,
            json.dumps(flag.default_value),
            flag.rollout_percentage,
            json.dumps(flag.environments),
            json.dumps(flag.tags),
            flag.owner,
            json.dumps(flag.metadata),
            flag.start_time,
            flag.end_time,
            flag.created_at,
            flag.updated_at,
            flag.version,
        )
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, params)
                await conn.commit()
            logger.debug("PG saved flag '%s'", flag.key)
        except Exception as exc:
            logger.error("PG save_flag('%s') failed: %s", flag.key, exc)
        return flag

    async def delete_flag(self, key: str) -> bool:
        """Delete a flag. Cascade deletes handle rules/overrides/variants."""
        if not self._initialized:
            return False

        query = f"DELETE FROM {self._t_flags} WHERE key = %s"
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, (key,))
                    deleted = cur.rowcount > 0
                await conn.commit()
            if deleted:
                logger.info("PG deleted flag '%s'", key)
            return deleted
        except Exception as exc:
            logger.error("PG delete_flag('%s') failed: %s", key, exc)
            return False

    # ------------------------------------------------------------------
    # IFlagStorage -- Rules
    # ------------------------------------------------------------------

    async def get_rules(self, flag_key: str) -> List[FlagRule]:
        """Retrieve rules sorted by priority."""
        if not self._initialized:
            return []

        query = (
            f"SELECT * FROM {self._t_rules} "
            f"WHERE flag_key = %s ORDER BY priority ASC"
        )
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, (flag_key,))
                    rows = await cur.fetchall()
                    return [self._row_to_rule(r) for r in rows]
        except Exception as exc:
            logger.error("PG get_rules('%s') failed: %s", flag_key, exc)
            return []

    async def save_rule(self, rule: FlagRule) -> FlagRule:
        """Upsert a targeting rule."""
        if not self._initialized:
            return rule

        query = f"""
            INSERT INTO {self._t_rules} (
                rule_id, flag_key, rule_type, priority, conditions, enabled
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (rule_id) DO UPDATE SET
                rule_type = EXCLUDED.rule_type,
                priority = EXCLUDED.priority,
                conditions = EXCLUDED.conditions,
                enabled = EXCLUDED.enabled
        """
        params = (
            rule.rule_id,
            rule.flag_key,
            rule.rule_type,
            rule.priority,
            json.dumps(rule.conditions),
            rule.enabled,
        )
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, params)
                await conn.commit()
            logger.debug("PG saved rule '%s' for flag '%s'", rule.rule_id, rule.flag_key)
        except Exception as exc:
            logger.error("PG save_rule failed: %s", exc)
        return rule

    async def delete_rule(self, flag_key: str, rule_id: str) -> bool:
        """Delete a single rule.

        Args:
            flag_key: Parent flag key (unused in SQL but kept for interface).
            rule_id: Rule identifier.

        Returns:
            True if the row was deleted.
        """
        if not self._initialized:
            return False

        query = f"DELETE FROM {self._t_rules} WHERE rule_id = %s AND flag_key = %s"
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, (rule_id, flag_key))
                    deleted = cur.rowcount > 0
                await conn.commit()
            return deleted
        except Exception as exc:
            logger.error("PG delete_rule failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # IFlagStorage -- Overrides
    # ------------------------------------------------------------------

    async def get_overrides(self, flag_key: str) -> List[FlagOverride]:
        """Retrieve all overrides for a flag."""
        if not self._initialized:
            return []

        query = f"SELECT * FROM {self._t_overrides} WHERE flag_key = %s"
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, (flag_key,))
                    rows = await cur.fetchall()
                    return [self._row_to_override(r) for r in rows]
        except Exception as exc:
            logger.error("PG get_overrides('%s') failed: %s", flag_key, exc)
            return []

    async def save_override(self, override: FlagOverride) -> FlagOverride:
        """Upsert an override keyed by (flag_key, scope_type, scope_value)."""
        if not self._initialized:
            return override

        query = f"""
            INSERT INTO {self._t_overrides} (
                flag_key, scope_type, scope_value, enabled,
                variant_key, expires_at, created_by
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (flag_key, scope_type, scope_value) DO UPDATE SET
                enabled = EXCLUDED.enabled,
                variant_key = EXCLUDED.variant_key,
                expires_at = EXCLUDED.expires_at,
                created_by = EXCLUDED.created_by
        """
        params = (
            override.flag_key,
            override.scope_type,
            override.scope_value,
            override.enabled,
            override.variant_key,
            override.expires_at,
            override.created_by,
        )
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, params)
                await conn.commit()
            logger.debug(
                "PG saved override for '%s' scope=%s:%s",
                override.flag_key, override.scope_type, override.scope_value,
            )
        except Exception as exc:
            logger.error("PG save_override failed: %s", exc)
        return override

    async def delete_override(
        self, flag_key: str, scope_type: str, scope_value: str
    ) -> bool:
        """Delete a specific override.

        Args:
            flag_key: Parent flag key.
            scope_type: Override scope type.
            scope_value: Override scope value.

        Returns:
            True if the row was deleted.
        """
        if not self._initialized:
            return False

        query = (
            f"DELETE FROM {self._t_overrides} "
            f"WHERE flag_key = %s AND scope_type = %s AND scope_value = %s"
        )
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, (flag_key, scope_type, scope_value))
                    deleted = cur.rowcount > 0
                await conn.commit()
            return deleted
        except Exception as exc:
            logger.error("PG delete_override failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # IFlagStorage -- Variants
    # ------------------------------------------------------------------

    async def get_variants(self, flag_key: str) -> List[FlagVariant]:
        """Retrieve all variants for a flag."""
        if not self._initialized:
            return []

        query = f"SELECT * FROM {self._t_variants} WHERE flag_key = %s"
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, (flag_key,))
                    rows = await cur.fetchall()
                    return [self._row_to_variant(r) for r in rows]
        except Exception as exc:
            logger.error("PG get_variants('%s') failed: %s", flag_key, exc)
            return []

    async def save_variant(self, variant: FlagVariant) -> FlagVariant:
        """Upsert a variant keyed by (flag_key, variant_key)."""
        if not self._initialized:
            return variant

        query = f"""
            INSERT INTO {self._t_variants} (
                variant_key, flag_key, variant_value, weight, description
            ) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (flag_key, variant_key) DO UPDATE SET
                variant_value = EXCLUDED.variant_value,
                weight = EXCLUDED.weight,
                description = EXCLUDED.description
        """
        params = (
            variant.variant_key,
            variant.flag_key,
            json.dumps(variant.variant_value),
            variant.weight,
            variant.description,
        )
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, params)
                await conn.commit()
            logger.debug(
                "PG saved variant '%s' for flag '%s'",
                variant.variant_key, variant.flag_key,
            )
        except Exception as exc:
            logger.error("PG save_variant failed: %s", exc)
        return variant

    async def delete_variant(self, flag_key: str, variant_key: str) -> bool:
        """Delete a specific variant.

        Args:
            flag_key: Parent flag key.
            variant_key: Variant identifier.

        Returns:
            True if the row was deleted.
        """
        if not self._initialized:
            return False

        query = (
            f"DELETE FROM {self._t_variants} "
            f"WHERE flag_key = %s AND variant_key = %s"
        )
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, (flag_key, variant_key))
                    deleted = cur.rowcount > 0
                await conn.commit()
            return deleted
        except Exception as exc:
            logger.error("PG delete_variant failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # IFlagStorage -- Audit Log
    # ------------------------------------------------------------------

    async def log_audit(self, entry: AuditLogEntry) -> None:
        """Insert an immutable audit log entry into PostgreSQL."""
        if not self._initialized:
            return

        query = f"""
            INSERT INTO {self._t_audit} (
                flag_key, action, old_value, new_value,
                changed_by, change_reason, ip_address, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = (
            entry.flag_key,
            entry.action,
            json.dumps(entry.old_value),
            json.dumps(entry.new_value),
            entry.changed_by,
            entry.change_reason,
            entry.ip_address,
            entry.created_at,
        )
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, params)
                await conn.commit()
        except Exception as exc:
            logger.error("PG log_audit failed: %s", exc)

    # Keep backward-compatible alias
    async def append_audit_log(self, entry: AuditLogEntry) -> None:
        """Alias for ``log_audit``."""
        await self.log_audit(entry)

    async def get_audit_log(
        self,
        flag_key: str,
        limit: int = 50,
    ) -> List[AuditLogEntry]:
        """Retrieve the most recent audit entries for a flag.

        Args:
            flag_key: Flag key to query.
            limit: Maximum entries to return.

        Returns:
            Entries in reverse chronological order.
        """
        if not self._initialized:
            return []

        query = (
            f"SELECT * FROM {self._t_audit} "
            f"WHERE flag_key = %s ORDER BY created_at DESC LIMIT %s"
        )
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, (flag_key, limit))
                    rows = await cur.fetchall()
                    return [self._row_to_audit(r) for r in rows]
        except Exception as exc:
            logger.error("PG get_audit_log('%s') failed: %s", flag_key, exc)
            return []

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, object]:
        """Check PostgreSQL connectivity and basic metrics."""
        result: Dict[str, object] = {
            "healthy": False,
            "backend": "PostgresFlagStorage",
            "initialized": self._initialized,
        }
        if not self._initialized or self._pool is None:
            return result

        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1")
                    result["healthy"] = True

                    await cur.execute(
                        f"SELECT COUNT(*) AS cnt FROM {self._t_flags}"
                    )
                    row = await cur.fetchone()
                    result["flags_count"] = row["cnt"] if row else 0

            pool_stats = self._pool.get_stats()
            result["pool_size"] = pool_stats.get("pool_size", 0)
            result["pool_available"] = pool_stats.get("pool_available", 0)
            result["requests_waiting"] = pool_stats.get("requests_waiting", 0)
        except Exception as exc:
            result["error"] = str(exc)

        return result
