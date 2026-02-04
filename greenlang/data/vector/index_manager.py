"""
HNSW and IVFFlat index lifecycle manager for pgvector.

Manages index creation, rebuilds, tuning, and health monitoring
with environment-specific parameter configurations.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from greenlang.data.vector.config import (
    DISTANCE_OPS,
    DistanceMetric,
    Environment,
    IndexConfig,
    IndexType,
)
from greenlang.data.vector.connection import VectorDBConnection

logger = logging.getLogger(__name__)


class IndexManager:
    """
    pgvector index lifecycle manager.

    Handles:
    - HNSW index creation with configurable m/ef_construction
    - IVFFlat index creation for bulk operations
    - Partial indexes per namespace for filtered queries
    - Index rebuild (REINDEX CONCURRENTLY)
    - Index health monitoring (size, usage stats)
    - Environment-specific tuning (dev/staging/prod)
    """

    def __init__(
        self,
        db: VectorDBConnection,
        config: IndexConfig,
    ):
        self.db = db
        self.config = config

    async def create_hnsw_index(
        self,
        table: str = "vector_embeddings",
        column: str = "embedding",
        index_name: Optional[str] = None,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        where_clause: Optional[str] = None,
        m: Optional[int] = None,
        ef_construction: Optional[int] = None,
    ) -> str:
        """
        Create an HNSW index on a vector column.

        Uses CONCURRENTLY to avoid blocking writes.
        """
        m = m or self.config.hnsw_m
        ef = ef_construction or self.config.hnsw_ef_construction
        ops = DISTANCE_OPS[distance_metric]
        name = index_name or f"idx_{table}_hnsw_{distance_metric.value}"

        sql = f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS {name}
            ON {table}
            USING hnsw ({column} {ops})
            WITH (m = {m}, ef_construction = {ef})
        """
        if where_clause:
            sql += f" WHERE {where_clause}"

        logger.info("Creating HNSW index '%s' (m=%d, ef=%d)", name, m, ef)

        async with self.db.acquire_writer() as conn:
            # Set maintenance work mem for index build
            await conn.execute(
                f"SET maintenance_work_mem = '{self.config.maintenance_work_mem}'"
            )
            await conn.execute(sql)
            await conn.commit()

        logger.info("HNSW index '%s' created successfully", name)
        return name

    async def create_ivfflat_index(
        self,
        table: str = "vector_embeddings",
        column: str = "embedding",
        index_name: Optional[str] = None,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        lists: Optional[int] = None,
        where_clause: Optional[str] = None,
    ) -> str:
        """
        Create an IVFFlat index on a vector column.

        IVFFlat is faster to build than HNSW and suitable for
        bulk operations and large datasets.
        """
        lists = lists or self.config.ivfflat_lists
        ops = DISTANCE_OPS[distance_metric]
        name = index_name or f"idx_{table}_ivfflat_{distance_metric.value}"

        sql = f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS {name}
            ON {table}
            USING ivfflat ({column} {ops})
            WITH (lists = {lists})
        """
        if where_clause:
            sql += f" WHERE {where_clause}"

        logger.info("Creating IVFFlat index '%s' (lists=%d)", name, lists)

        async with self.db.acquire_writer() as conn:
            await conn.execute(
                f"SET maintenance_work_mem = '{self.config.maintenance_work_mem}'"
            )
            await conn.execute(sql)
            await conn.commit()

        logger.info("IVFFlat index '%s' created successfully", name)
        return name

    async def create_namespace_indexes(
        self,
        namespaces: Optional[List[str]] = None,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
    ) -> List[str]:
        """
        Create partial HNSW indexes for each namespace.

        Partial indexes are more efficient for namespace-filtered queries
        as PostgreSQL can use a smaller, focused index.
        """
        if namespaces is None:
            namespaces = ["csrd", "cbam", "eudr", "vcci", "sb253", "taxonomy", "csddd"]

        created = []
        for ns in namespaces:
            name = f"idx_embeddings_{ns}_hnsw"
            await self.create_hnsw_index(
                index_name=name,
                where_clause=f"namespace = '{ns}'",
                distance_metric=distance_metric,
            )
            created.append(name)

        return created

    async def rebuild_index(self, index_name: str) -> None:
        """
        Rebuild an index using REINDEX CONCURRENTLY.

        Should be run periodically after large data changes
        to maintain index quality.
        """
        logger.info("Rebuilding index '%s'", index_name)
        async with self.db.acquire_writer() as conn:
            await conn.execute(
                f"SET maintenance_work_mem = '{self.config.maintenance_work_mem}'"
            )
            await conn.execute(f"REINDEX INDEX CONCURRENTLY {index_name}")
            await conn.commit()
        logger.info("Index '%s' rebuilt successfully", index_name)

    async def drop_index(self, index_name: str) -> None:
        """Drop an index."""
        async with self.db.acquire_writer() as conn:
            await conn.execute(f"DROP INDEX IF EXISTS {index_name}")
            await conn.commit()
        logger.info("Index '%s' dropped", index_name)

    async def get_index_info(self) -> List[Dict]:
        """Get information about all vector-related indexes."""
        rows = await self.db.execute(
            """
            SELECT
                i.indexname AS name,
                i.tablename AS table,
                pg_size_pretty(pg_relation_size(i.indexname::regclass)) AS size,
                pg_relation_size(i.indexname::regclass) AS size_bytes,
                s.idx_scan AS scans,
                s.idx_tup_read AS tuples_read,
                s.idx_tup_fetch AS tuples_fetched,
                i.indexdef AS definition
            FROM pg_indexes i
            LEFT JOIN pg_stat_user_indexes s
                ON s.indexrelname = i.indexname
            WHERE i.tablename IN ('vector_embeddings', 'vector_embeddings_partitioned')
            ORDER BY pg_relation_size(i.indexname::regclass) DESC
            """,
            use_reader=True,
        )
        return [dict(row) for row in rows]

    async def get_index_health(self) -> Dict:
        """
        Assess overall index health.

        Returns metrics for monitoring:
        - Total index size
        - Index-to-table size ratio
        - Unused indexes
        - Index scan statistics
        """
        info = await self.get_index_info()

        total_size = sum(i.get("size_bytes", 0) for i in info)
        total_scans = sum(i.get("scans", 0) for i in info)
        unused = [i["name"] for i in info if i.get("scans", 0) == 0]

        # Get table size for ratio
        table_size_row = await self.db.execute_one(
            """
            SELECT pg_total_relation_size('vector_embeddings') AS size
            """,
            use_reader=True,
        )
        table_size = table_size_row["size"] if table_size_row else 0

        return {
            "total_indexes": len(info),
            "total_index_size_bytes": total_size,
            "total_index_size": _format_bytes(total_size),
            "table_size_bytes": table_size,
            "index_to_table_ratio": round(total_size / max(1, table_size), 2),
            "total_scans": total_scans,
            "unused_indexes": unused,
            "indexes": info,
        }

    async def tune_for_environment(self, env: Environment) -> Dict:
        """
        Apply environment-specific tuning parameters.

        Returns the parameters that were set.
        """
        config = IndexConfig.for_environment(env)
        ef_search_map = {
            Environment.DEVELOPMENT: 40,
            Environment.STAGING: 100,
            Environment.PRODUCTION: 200,
        }
        ef_search = ef_search_map.get(env, 100)

        async with self.db.acquire_writer() as conn:
            await conn.execute(f"SET hnsw.ef_search = {ef_search}")
            await conn.commit()

        params = {
            "environment": env.value,
            "hnsw_m": config.hnsw_m,
            "hnsw_ef_construction": config.hnsw_ef_construction,
            "ef_search": ef_search,
            "ivfflat_lists": config.ivfflat_lists,
            "maintenance_work_mem": config.maintenance_work_mem,
        }
        logger.info("Tuned for %s: %s", env.value, params)
        return params

    async def vacuum_analyze(self, table: str = "vector_embeddings") -> None:
        """Run VACUUM ANALYZE on the vector table."""
        logger.info("Running VACUUM ANALYZE on '%s'", table)
        async with self.db.acquire_writer() as conn:
            # VACUUM requires autocommit
            await conn.set_autocommit(True)
            await conn.execute(f"VACUUM ANALYZE {table}")
        logger.info("VACUUM ANALYZE completed on '%s'", table)


def _format_bytes(size: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"
