"""
Vector search engine with similarity, filtered, and hybrid search.

Implements pgvector-based search operations including:
- Basic cosine similarity search
- Filtered search with namespace, source_type, and JSONB metadata
- Hybrid search combining vector + full-text with Reciprocal Rank Fusion (RRF)
- Configurable ef_search per query for recall/latency tradeoff
- Query logging for analytics
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from greenlang.data.vector.config import (
    DISTANCE_OPERATORS,
    DistanceMetric,
    SearchConfig,
)
from greenlang.data.vector.connection import VectorDBConnection
from greenlang.data.vector.models import (
    HybridSearchRequest,
    SearchMatch,
    SearchRequest,
    SearchResult,
)

logger = logging.getLogger(__name__)


class SearchEngine:
    """
    pgvector search engine with multiple search strategies.

    Supports:
    - similarity_search: Basic top-K cosine similarity
    - filtered_search: Search with namespace/source_type/metadata filters
    - hybrid_search: Combined vector + full-text with RRF fusion
    """

    def __init__(
        self,
        db: VectorDBConnection,
        config: SearchConfig,
        embedding_fn=None,
    ):
        self.db = db
        self.config = config
        self._embed = embedding_fn  # async callable: str -> np.ndarray

    def set_embedding_fn(self, fn) -> None:
        """Set the embedding function for query vectorization."""
        self._embed = fn

    async def _get_query_embedding(self, query: str) -> np.ndarray:
        """Convert query text to embedding vector."""
        if self._embed is None:
            raise RuntimeError(
                "No embedding function configured. "
                "Set via SearchEngine(embedding_fn=...) or set_embedding_fn()"
            )
        return await self._embed(query)

    async def similarity_search(
        self, request: SearchRequest
    ) -> SearchResult:
        """
        Basic top-K similarity search using cosine distance.

        Uses HNSW index for fast approximate nearest neighbor search.
        """
        start_time = time.monotonic()
        query_embedding = await self._get_query_embedding(request.query)

        ef_search = request.ef_search or self.config.ef_search
        operator = DISTANCE_OPERATORS[self.config.distance_metric]

        async with self.db.acquire_reader() as conn:
            await self.db.set_search_params(conn, ef_search=ef_search)
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    SELECT
                        id,
                        source_type,
                        source_id,
                        chunk_index,
                        content_preview,
                        metadata,
                        1 - (embedding {operator} %s::vector) AS similarity
                    FROM vector_embeddings
                    WHERE
                        namespace = %s
                        AND 1 - (embedding {operator} %s::vector) >= %s
                    ORDER BY embedding {operator} %s::vector
                    LIMIT %s
                    """,
                    (
                        query_embedding,
                        request.namespace,
                        query_embedding,
                        request.threshold,
                        query_embedding,
                        request.top_k,
                    ),
                )
                rows = await cur.fetchall()

        matches = [self._row_to_match(row) for row in rows]
        latency_ms = int((time.monotonic() - start_time) * 1000)

        result = SearchResult(
            matches=matches,
            query_text=request.query,
            total_results=len(matches),
            latency_ms=latency_ms,
            search_type="similarity",
            namespace=request.namespace,
        )

        if self.config.log_queries:
            await self._log_search(request, query_embedding, result)

        return result

    async def filtered_search(
        self, request: SearchRequest
    ) -> SearchResult:
        """
        Filtered similarity search with namespace, source_type, and metadata conditions.

        Leverages partial HNSW indexes for namespace-filtered queries.
        """
        start_time = time.monotonic()
        query_embedding = await self._get_query_embedding(request.query)

        ef_search = request.ef_search or self.config.ef_search
        operator = DISTANCE_OPERATORS[self.config.distance_metric]

        conditions = ["namespace = %s"]
        params: list = [query_embedding, request.namespace]

        if request.source_type:
            conditions.append("source_type = %s")
            params.append(request.source_type)

        if request.metadata_filter:
            conditions.append("metadata @> %s::jsonb")
            params.append(json.dumps(request.metadata_filter))

        conditions.append(
            f"1 - (embedding {operator} %s::vector) >= %s"
        )
        params.extend([query_embedding, request.threshold])

        where_clause = " AND ".join(conditions)
        params.extend([query_embedding, request.top_k])

        async with self.db.acquire_reader() as conn:
            await self.db.set_search_params(conn, ef_search=ef_search)
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    SELECT
                        id,
                        source_type,
                        source_id,
                        chunk_index,
                        content_preview,
                        metadata,
                        1 - (embedding {operator} %s::vector) AS similarity
                    FROM vector_embeddings
                    WHERE {where_clause}
                    ORDER BY embedding {operator} %s::vector
                    LIMIT %s
                    """,
                    tuple(params),
                )
                rows = await cur.fetchall()

        matches = [self._row_to_match(row) for row in rows]
        latency_ms = int((time.monotonic() - start_time) * 1000)

        result = SearchResult(
            matches=matches,
            query_text=request.query,
            total_results=len(matches),
            latency_ms=latency_ms,
            search_type="filtered",
            namespace=request.namespace,
        )

        if self.config.log_queries:
            await self._log_search(request, query_embedding, result)

        return result

    async def hybrid_search(
        self, request: HybridSearchRequest
    ) -> SearchResult:
        """
        Hybrid search combining vector similarity + full-text search
        using Reciprocal Rank Fusion (RRF).

        Executes both searches in parallel and merges results using
        configurable RRF constant (k=60 by default).
        """
        start_time = time.monotonic()
        query_embedding = await self._get_query_embedding(request.query)

        ef_search = self.config.ef_search
        rrf_k = request.rrf_k or self.config.rrf_k

        async with self.db.acquire_reader() as conn:
            await self.db.set_search_params(conn, ef_search=ef_search)
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    WITH vector_results AS (
                        SELECT
                            id,
                            ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS vector_rank
                        FROM vector_embeddings
                        WHERE namespace = %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT 100
                    ),
                    text_results AS (
                        SELECT
                            id,
                            ROW_NUMBER() OVER (
                                ORDER BY ts_rank(
                                    to_tsvector('english', content_preview),
                                    plainto_tsquery('english', %s)
                                ) DESC
                            ) AS text_rank
                        FROM vector_embeddings
                        WHERE
                            namespace = %s
                            AND content_preview IS NOT NULL
                            AND to_tsvector('english', content_preview)
                                @@ plainto_tsquery('english', %s)
                        LIMIT 100
                    ),
                    rrf_scores AS (
                        SELECT
                            COALESCE(v.id, t.id) AS id,
                            COALESCE(1.0 / (%s + v.vector_rank), 0) +
                            COALESCE(1.0 / (%s + t.text_rank), 0) AS rrf_score,
                            v.vector_rank,
                            t.text_rank
                        FROM vector_results v
                        FULL OUTER JOIN text_results t ON v.id = t.id
                    )
                    SELECT
                        e.id,
                        e.source_type,
                        e.source_id,
                        e.chunk_index,
                        e.content_preview,
                        e.metadata,
                        r.rrf_score,
                        r.vector_rank,
                        r.text_rank
                    FROM rrf_scores r
                    JOIN vector_embeddings e ON e.id = r.id
                    ORDER BY r.rrf_score DESC
                    LIMIT %s
                    """,
                    (
                        query_embedding,
                        request.namespace,
                        query_embedding,
                        request.query,
                        request.namespace,
                        request.query,
                        rrf_k,
                        rrf_k,
                        request.top_k,
                    ),
                )
                rows = await cur.fetchall()

        matches = []
        for row in rows:
            matches.append(
                SearchMatch(
                    id=str(row["id"]),
                    source_type=row["source_type"],
                    source_id=str(row["source_id"]),
                    chunk_index=row.get("chunk_index", 0),
                    content_preview=row.get("content_preview"),
                    metadata=row.get("metadata", {}),
                    similarity=0.0,  # Not directly available in hybrid
                    vector_rank=row.get("vector_rank"),
                    text_rank=row.get("text_rank"),
                    rrf_score=float(row["rrf_score"]),
                )
            )

        latency_ms = int((time.monotonic() - start_time) * 1000)

        result = SearchResult(
            matches=matches,
            query_text=request.query,
            total_results=len(matches),
            latency_ms=latency_ms,
            search_type="hybrid",
            namespace=request.namespace,
        )

        if self.config.log_queries:
            await self._log_search(
                SearchRequest(
                    query=request.query,
                    namespace=request.namespace,
                    top_k=request.top_k,
                ),
                query_embedding,
                result,
            )

        return result

    async def _log_search(
        self,
        request: SearchRequest,
        query_embedding: np.ndarray,
        result: SearchResult,
    ) -> None:
        """Log search query for analytics."""
        try:
            async with self.db.acquire_writer() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        INSERT INTO vector_search_logs (
                            query_embedding, query_text, search_type,
                            namespace, top_k, threshold,
                            result_count, latency_ms, ef_search_used
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            query_embedding,
                            request.query,
                            result.search_type,
                            request.namespace,
                            request.top_k,
                            request.threshold,
                            result.total_results,
                            result.latency_ms,
                            request.ef_search or self.config.ef_search,
                        ),
                    )
                await conn.commit()
        except Exception as e:
            logger.warning("Failed to log search query: %s", e)

    @staticmethod
    def _row_to_match(row: dict) -> SearchMatch:
        """Convert a database row to a SearchMatch."""
        return SearchMatch(
            id=str(row["id"]),
            source_type=row["source_type"],
            source_id=str(row["source_id"]),
            chunk_index=row.get("chunk_index", 0),
            content_preview=row.get("content_preview"),
            metadata=row.get("metadata", {}),
            similarity=float(row.get("similarity", 0)),
        )

    async def get_stats(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Get vector database statistics."""
        async with self.db.acquire_reader() as conn:
            async with conn.cursor() as cur:
                # Total embeddings
                if namespace:
                    await cur.execute(
                        "SELECT COUNT(*) as count FROM vector_embeddings WHERE namespace = %s",
                        (namespace,),
                    )
                else:
                    await cur.execute(
                        "SELECT COUNT(*) as count FROM vector_embeddings"
                    )
                total = (await cur.fetchone())["count"]

                # Per-namespace counts
                await cur.execute(
                    """
                    SELECT namespace, COUNT(*) as count
                    FROM vector_embeddings
                    GROUP BY namespace
                    ORDER BY count DESC
                    """
                )
                namespace_counts = await cur.fetchall()

                # Index sizes
                await cur.execute(
                    """
                    SELECT
                        indexname,
                        pg_size_pretty(pg_relation_size(indexname::regclass)) as size
                    FROM pg_indexes
                    WHERE tablename = 'vector_embeddings'
                    """
                )
                indexes = await cur.fetchall()

                # Recent search stats
                await cur.execute(
                    """
                    SELECT
                        COUNT(*) as total_searches,
                        AVG(latency_ms) as avg_latency_ms,
                        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) as p99_latency_ms
                    FROM vector_search_logs
                    WHERE created_at > NOW() - INTERVAL '1 hour'
                    """
                )
                search_stats = await cur.fetchone()

        return {
            "total_embeddings": total,
            "namespace_counts": {r["namespace"]: r["count"] for r in namespace_counts},
            "indexes": [{"name": r["indexname"], "size": r["size"]} for r in indexes],
            "search_stats_1h": {
                "total_searches": search_stats["total_searches"],
                "avg_latency_ms": round(search_stats["avg_latency_ms"] or 0, 2),
                "p99_latency_ms": round(search_stats["p99_latency_ms"] or 0, 2),
            },
        }
