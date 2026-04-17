# -*- coding: utf-8 -*-
"""
PgVector-backed semantic index for emission factor matching (F040).

Uses pgvector for cosine similarity search on pre-computed embeddings.
Implements the SemanticIndex protocol from semantic_index.py.

Schema: factors_catalog.factor_embeddings(
    edition_id, factor_id, embedding vector(384), search_text, content_hash
)

Requires: psycopg[binary] and pgvector extension in Postgres.
Gracefully degrades to NoopSemanticIndex when unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Dimension constants for supported embedding models
EMBEDDING_DIM_MINILM = 384
EMBEDDING_DIM_MPNET = 768
EMBEDDING_DIM_OPENAI = 1536

DEFAULT_EMBEDDING_DIM = EMBEDDING_DIM_MINILM


@dataclass
class PgVectorConfig:
    """Configuration for pgvector semantic index."""

    dsn: str
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    schema_name: str = "factors_catalog"
    table_name: str = "factor_embeddings"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    hnsw_ef_search: int = 40
    distance_metric: str = "cosine"  # cosine, l2, inner_product
    max_results: int = 50


@dataclass
class EmbeddingRecord:
    """A single factor embedding record."""

    edition_id: str
    factor_id: str
    embedding: List[float]
    search_text: str
    content_hash: str


class PgVectorSemanticIndex:
    """
    Semantic index backed by PostgreSQL + pgvector.

    Stores pre-computed embeddings and performs cosine similarity search.
    Implements the SemanticIndex protocol.

    Usage:
        config = PgVectorConfig(dsn="postgresql://...")
        index = PgVectorSemanticIndex(config)
        await index.initialize()

        # Store embeddings
        await index.upsert_embeddings(edition_id, records)

        # Search
        results = await index.search(edition_id, query_vector, k=20)
    """

    def __init__(self, config: PgVectorConfig):
        self._config = config
        self._conn = None
        self._initialized = False

    @property
    def full_table_name(self) -> str:
        return f"{self._config.schema_name}.{self._config.table_name}"

    def initialize(self) -> None:
        """Initialize connection and ensure pgvector extension + table exist."""
        try:
            import psycopg

            self._conn = psycopg.connect(self._config.dsn)
            self._ensure_schema()
            self._initialized = True
            logger.info(
                "PgVectorSemanticIndex initialized: dim=%d table=%s",
                self._config.embedding_dim, self.full_table_name,
            )
        except ImportError:
            logger.warning("psycopg not available — PgVectorSemanticIndex disabled")
            self._initialized = False
        except Exception as exc:
            logger.warning(
                "PgVectorSemanticIndex initialization failed: %s", exc,
            )
            self._initialized = False

    def _ensure_schema(self) -> None:
        """Create extension, schema, table and indexes if needed."""
        if not self._conn:
            return
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                "CREATE SCHEMA IF NOT EXISTS %s" % self._config.schema_name
            )
            cur.execute("""
                CREATE TABLE IF NOT EXISTS %(table)s (
                    edition_id   TEXT NOT NULL,
                    factor_id    TEXT NOT NULL,
                    embedding    vector(%(dim)s) NOT NULL,
                    search_text  TEXT NOT NULL DEFAULT '',
                    content_hash TEXT NOT NULL DEFAULT '',
                    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (edition_id, factor_id)
                )
            """ % {"table": self.full_table_name, "dim": self._config.embedding_dim})
            # HNSW index for cosine similarity
            idx_name = f"idx_{self._config.table_name}_hnsw"
            cur.execute("""
                CREATE INDEX IF NOT EXISTS %(idx)s
                ON %(table)s USING hnsw (embedding vector_cosine_ops)
                WITH (m = %(m)s, ef_construction = %(ef)s)
            """ % {
                "idx": idx_name,
                "table": self.full_table_name,
                "m": self._config.hnsw_m,
                "ef": self._config.hnsw_ef_construction,
            })
        self._conn.commit()

    @property
    def is_available(self) -> bool:
        return self._initialized and self._conn is not None

    def embed_text(self, text: str) -> List[float]:
        """Protocol method — returns empty; use external embedding model."""
        return []

    def search(
        self,
        edition_id: str,
        vector: List[float],
        k: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Search for nearest factors by cosine similarity.

        Returns list of dicts with factor_id, distance, search_text.
        """
        if not self.is_available or not vector:
            return []

        try:
            with self._conn.cursor() as cur:
                # Set ef_search for this query
                cur.execute(
                    "SET LOCAL hnsw.ef_search = %s",
                    (self._config.hnsw_ef_search,),
                )
                cur.execute(
                    """
                    SELECT factor_id, search_text, content_hash,
                           embedding <=> %s::vector AS distance
                    FROM %(table)s
                    WHERE edition_id = %%(eid)s
                    ORDER BY embedding <=> %%(vec)s::vector
                    LIMIT %%(k)s
                    """ % {"table": self.full_table_name},
                    {"eid": edition_id, "vec": str(vector), "k": k},
                )
                rows = cur.fetchall()

            results = []
            for row in rows:
                results.append({
                    "factor_id": row[0],
                    "search_text": row[1],
                    "content_hash": row[2],
                    "distance": float(row[3]),
                    "similarity": 1.0 - float(row[3]),
                })
            logger.debug(
                "PgVector search: edition=%s k=%d results=%d",
                edition_id, k, len(results),
            )
            return results
        except Exception as exc:
            logger.warning("PgVector search failed: %s", exc)
            return []

    def upsert_embeddings(
        self,
        edition_id: str,
        records: Sequence[EmbeddingRecord],
        batch_size: int = 500,
    ) -> int:
        """
        Upsert embedding records into the index.

        Returns number of records upserted.
        """
        if not self.is_available:
            logger.warning("PgVector not available — skipping upsert")
            return 0

        total = 0
        for i in range(0, len(records), batch_size):
            batch = records[i: i + batch_size]
            try:
                with self._conn.cursor() as cur:
                    for rec in batch:
                        cur.execute(
                            """
                            INSERT INTO %(table)s
                                (edition_id, factor_id, embedding, search_text, content_hash)
                            VALUES
                                (%%(eid)s, %%(fid)s, %%(emb)s::vector, %%(st)s, %%(ch)s)
                            ON CONFLICT (edition_id, factor_id)
                            DO UPDATE SET
                                embedding = EXCLUDED.embedding,
                                search_text = EXCLUDED.search_text,
                                content_hash = EXCLUDED.content_hash,
                                created_at = now()
                            """ % {"table": self.full_table_name},
                            {
                                "eid": rec.edition_id,
                                "fid": rec.factor_id,
                                "emb": str(rec.embedding),
                                "st": rec.search_text,
                                "ch": rec.content_hash,
                            },
                        )
                self._conn.commit()
                total += len(batch)
            except Exception as exc:
                logger.error(
                    "PgVector upsert batch failed (offset=%d): %s", i, exc,
                )
                try:
                    self._conn.rollback()
                except Exception:
                    pass

        logger.info(
            "PgVector upsert: edition=%s total=%d/%d",
            edition_id, total, len(records),
        )
        return total

    def delete_edition(self, edition_id: str) -> int:
        """Delete all embeddings for an edition. Returns rows deleted."""
        if not self.is_available:
            return 0
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM %s WHERE edition_id = %%s" % self.full_table_name,
                    (edition_id,),
                )
                deleted = cur.rowcount
            self._conn.commit()
            logger.info("PgVector delete: edition=%s deleted=%d", edition_id, deleted)
            return deleted
        except Exception as exc:
            logger.error("PgVector delete failed: %s", exc)
            return 0

    def count_embeddings(self, edition_id: str) -> int:
        """Count embeddings for an edition."""
        if not self.is_available:
            return 0
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM %s WHERE edition_id = %%s" % self.full_table_name,
                    (edition_id,),
                )
                return cur.fetchone()[0]
        except Exception:
            return 0

    def get_stale_factors(
        self,
        edition_id: str,
        current_hashes: Dict[str, str],
    ) -> List[str]:
        """
        Find factors whose content_hash has changed since last embedding.

        Returns list of factor_ids that need re-embedding.
        """
        if not self.is_available or not current_hashes:
            return list(current_hashes.keys()) if current_hashes else []

        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT factor_id, content_hash FROM %s WHERE edition_id = %%s"
                    % self.full_table_name,
                    (edition_id,),
                )
                stored = {row[0]: row[1] for row in cur.fetchall()}

            stale = []
            for fid, h in current_hashes.items():
                if fid not in stored or stored[fid] != h:
                    stale.append(fid)
            return stale
        except Exception as exc:
            logger.warning("PgVector stale check failed: %s", exc)
            return list(current_hashes.keys())

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
            self._initialized = False


def create_semantic_index(
    dsn: Optional[str] = None,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> PgVectorSemanticIndex:
    """Factory function to create and initialize a PgVector index."""
    import os

    dsn = dsn or os.environ.get("GL_FACTORS_PG_DSN", "")
    if not dsn:
        logger.info("No GL_FACTORS_PG_DSN — semantic index disabled")
        from greenlang.factors.matching.semantic_index import NoopSemanticIndex
        return NoopSemanticIndex()  # type: ignore[return-value]

    config = PgVectorConfig(dsn=dsn, embedding_dim=embedding_dim)
    index = PgVectorSemanticIndex(config)
    index.initialize()
    return index
