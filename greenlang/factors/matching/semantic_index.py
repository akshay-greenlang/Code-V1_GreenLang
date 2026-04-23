# -*- coding: utf-8 -*-
"""
Semantic retrieval index — production implementation (M3).

This module replaces the FY26 no-op stub with a real two-tier index:

    1. **Postgres + pgvector** (preferred, when ``DATABASE_URL`` /
       ``GL_FACTORS_PG_DSN`` is set). Wraps
       :class:`PgVectorSemanticIndex` for hosted deployments — HNSW
       cosine similarity, 384/768/1536-d embeddings, edition-scoped.
    2. **In-memory NumPy fallback** (always available). A pure-Python
       brute-force cosine search over a dict ``{factor_id: vector}``.
       Used in tests/CI and any environment without Postgres so the
       matching pipeline can keep running with deterministic results.

The public class is :class:`SemanticIndex` with the following stable
methods (mirrors the M3 spec in the FY27 launch checklist):

* ``upsert(factor_id, text, metadata=None, edition_id=None)``
* ``search(query, k=10, edition_id=None, filters=None)``
* ``delete(factor_id, edition_id=None)``
* ``health()``

All methods are synchronous to match the existing Factors API shape
(``api_endpoints.build_resolution_explain`` is sync).  The pgvector
backend already wraps async psycopg internally where relevant.

Embedding model resolution
--------------------------

By default the index calls
:class:`greenlang.factors.matching.embedding.EmbeddingPipeline` to embed
text, which itself loads ``sentence-transformers`` when available and
transparently falls back to :class:`StubEmbeddingModel` otherwise.  Pass
``embedder=`` to inject a custom embedder (any object with an
``embed_text(str) -> list[float]`` method).
"""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default edition id used when callers do not pass one explicitly
# ---------------------------------------------------------------------------

_DEFAULT_EDITION_ID = "default"


# ---------------------------------------------------------------------------
# Embedder protocol + factory
# ---------------------------------------------------------------------------


class _Embedder(Protocol):
    """Anything that can turn text into a fixed-dimension vector."""

    def embed_text(self, text: str) -> List[float]:
        ...


def get_embedder(
    *,
    use_stub: Optional[bool] = None,
    model_name: Optional[str] = None,
) -> _Embedder:
    """Return the default embedder used by :class:`SemanticIndex`.

    Resolution order:
        1. If ``use_stub`` is True (or ``GL_FACTORS_EMBED_STUB=1``), the
           deterministic :class:`StubEmbeddingModel` is returned — fast,
           no extra deps.
        2. Otherwise build an :class:`EmbeddingPipeline` from
           :class:`EmbeddingConfig`. The pipeline itself falls back to
           the stub model when ``sentence-transformers`` is missing.
    """
    from greenlang.factors.matching.embedding import (
        EmbeddingConfig,
        EmbeddingPipeline,
    )

    if use_stub is None:
        use_stub = os.environ.get("GL_FACTORS_EMBED_STUB", "").strip() in (
            "1", "true", "yes", "on",
        )

    cfg_kwargs: Dict[str, Any] = {"use_stub": bool(use_stub)}
    if model_name:
        cfg_kwargs["model_name"] = model_name
    return EmbeddingPipeline.from_config(EmbeddingConfig(**cfg_kwargs))


# ---------------------------------------------------------------------------
# In-memory fallback
# ---------------------------------------------------------------------------


@dataclass
class _MemoryRecord:
    """One vector + metadata in the in-memory index."""

    factor_id: str
    edition_id: str
    vector: List[float]
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def _cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length vectors.

    Returns ``0.0`` when either vector has zero magnitude (no NaN).
    NumPy is preferred when available because it handles 100k+ records
    in milliseconds; we fall back to pure-Python so unit tests don't
    require numpy.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    try:
        import numpy as np

        va = np.asarray(a, dtype="float64")
        vb = np.asarray(b, dtype="float64")
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))
    except ImportError:  # pragma: no cover — numpy is in core deps
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)


class _InMemorySemanticIndex:
    """Pure-Python cosine-similarity index used as a fallback / for tests."""

    def __init__(self) -> None:
        # (edition_id, factor_id) → record
        self._records: Dict[Tuple[str, str], _MemoryRecord] = {}
        self._lock = threading.RLock()

    # -- writes --

    def upsert(self, record: _MemoryRecord) -> None:
        with self._lock:
            self._records[(record.edition_id, record.factor_id)] = record

    def delete(self, factor_id: str, edition_id: str) -> bool:
        with self._lock:
            return self._records.pop((edition_id, factor_id), None) is not None

    # -- reads --

    def search(
        self,
        query_vector: List[float],
        k: int,
        edition_id: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not query_vector:
            return []
        filters = filters or {}
        with self._lock:
            candidates = [
                r for (eid, _fid), r in self._records.items()
                if eid == edition_id and _matches_filters(r.metadata, filters)
            ]
        scored: List[Tuple[float, _MemoryRecord]] = []
        for rec in candidates:
            sim = _cosine(query_vector, rec.vector)
            scored.append((sim, rec))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = scored[: max(0, int(k))]
        return [
            {
                "factor_id": rec.factor_id,
                "edition_id": rec.edition_id,
                "similarity": float(sim),
                "distance": 1.0 - float(sim),
                "search_text": rec.text,
                "metadata": dict(rec.metadata),
            }
            for sim, rec in top
        ]

    # -- introspection --

    def size(self, edition_id: Optional[str] = None) -> int:
        with self._lock:
            if edition_id is None:
                return len(self._records)
            return sum(1 for (eid, _f) in self._records if eid == edition_id)


def _matches_filters(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """Simple equality-only filter match.  Empty filters always match."""
    if not filters:
        return True
    for k, v in filters.items():
        if metadata.get(k) != v:
            return False
    return True


# ---------------------------------------------------------------------------
# Public protocol (kept for backward compatibility with the old stub)
# ---------------------------------------------------------------------------


class SemanticIndexProtocol(Protocol):
    def embed_text(self, text: str) -> List[float]:
        ...

    def search(
        self, edition_id: str, vector: List[float], k: int
    ) -> List[Dict[str, Any]]:
        ...


class NoopSemanticIndex:
    """Deprecated placeholder retained for compat with the FY26 imports.

    New code should construct :class:`SemanticIndex` instead.  This
    type is still imported by ``pgvector_index.create_semantic_index``
    when no DSN is configured; we keep it returning empty results so
    nothing crashes during the deprecation window.
    """

    def embed_text(self, text: str) -> List[float]:  # pragma: no cover
        return []

    def search(  # pragma: no cover
        self, edition_id: str, vector: List[float], k: int
    ) -> List[Dict[str, Any]]:
        return []


# ---------------------------------------------------------------------------
# Production SemanticIndex
# ---------------------------------------------------------------------------


class SemanticIndex:
    """Production semantic index with pgvector + in-memory fallback.

    Construction is cheap and safe; the pgvector connection is opened
    lazily on first use.  In environments without a DSN, the in-memory
    backend is used immediately.

    Args:
        embedder: object with ``embed_text(str) -> list[float]``. If not
            supplied, :func:`get_embedder` is called.
        pg_dsn: Postgres DSN.  Defaults to ``DATABASE_URL`` then
            ``GL_FACTORS_PG_DSN``.  Pass ``""`` to force the in-memory
            backend.
        default_edition_id: edition used when callers omit ``edition_id=``.
        embedding_dim: dimension hint passed to the pgvector backend.
    """

    def __init__(
        self,
        *,
        embedder: Optional[_Embedder] = None,
        pg_dsn: Optional[str] = None,
        default_edition_id: str = _DEFAULT_EDITION_ID,
        embedding_dim: Optional[int] = None,
    ) -> None:
        self._embedder = embedder or get_embedder()
        self._default_edition_id = default_edition_id
        self._memory = _InMemorySemanticIndex()
        self._pg = None  # populated lazily on first use
        self._pg_init_attempted = False
        self._pg_dsn = (
            pg_dsn
            if pg_dsn is not None
            else (
                os.environ.get("DATABASE_URL", "").strip()
                or os.environ.get("GL_FACTORS_PG_DSN", "").strip()
            )
        )
        self._embedding_dim = embedding_dim

    # ------------------------------------------------------------------
    # Lazy pgvector init
    # ------------------------------------------------------------------

    def _ensure_pg(self) -> None:
        if self._pg_init_attempted:
            return
        self._pg_init_attempted = True
        if not self._pg_dsn:
            return
        try:
            from greenlang.factors.matching.pgvector_index import (
                DEFAULT_EMBEDDING_DIM,
                PgVectorConfig,
                PgVectorSemanticIndex,
            )

            dim = self._embedding_dim or DEFAULT_EMBEDDING_DIM
            cfg = PgVectorConfig(dsn=self._pg_dsn, embedding_dim=dim)
            pg = PgVectorSemanticIndex(cfg)
            pg.initialize()
            if pg.is_available:
                self._pg = pg
                logger.info(
                    "SemanticIndex: pgvector backend ready (dim=%d)", dim,
                )
            else:
                logger.warning(
                    "SemanticIndex: pgvector configured but not available; "
                    "using in-memory fallback only.",
                )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "SemanticIndex: pgvector init failed (%s); "
                "using in-memory fallback only.",
                exc,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def has_pg_backend(self) -> bool:
        """True when the pgvector backend is initialized and connected."""
        self._ensure_pg()
        return self._pg is not None

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string using the configured embedder."""
        if not text:
            return []
        return list(self._embedder.embed_text(text))

    def upsert(
        self,
        factor_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        edition_id: Optional[str] = None,
    ) -> bool:
        """Index ``text`` under ``factor_id``.

        Returns True on success.  Always writes to the in-memory layer
        (which functions as a write-through cache); also writes to
        pgvector when available.
        """
        if not factor_id:
            raise ValueError("factor_id must be non-empty")
        edition = edition_id or self._default_edition_id
        vec = self.embed_text(text)
        if not vec:
            logger.debug("SemanticIndex.upsert: empty embedding for %s", factor_id)
            return False

        # Always write the in-memory copy so search() works even when pg
        # is down.  Tests and stripped-down deployments rely on this.
        self._memory.upsert(_MemoryRecord(
            factor_id=factor_id,
            edition_id=edition,
            vector=vec,
            text=text,
            metadata=dict(metadata or {}),
        ))

        # Best-effort pgvector write.
        self._ensure_pg()
        if self._pg is not None:
            try:
                from greenlang.factors.matching.pgvector_index import EmbeddingRecord

                self._pg.upsert_embeddings(
                    edition,
                    [EmbeddingRecord(
                        edition_id=edition,
                        factor_id=factor_id,
                        embedding=vec,
                        search_text=text,
                        content_hash="",
                    )],
                )
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning(
                    "SemanticIndex.upsert: pgvector write failed (%s)", exc,
                )
        return True

    def search(
        self,
        query: str,
        k: int = 10,
        *,
        edition_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return up to ``k`` factor candidates ranked by cosine similarity.

        Each result dict contains ``factor_id``, ``similarity``,
        ``distance``, ``search_text``, ``metadata``.  The pgvector
        backend is preferred when available; the in-memory backend
        provides a deterministic top-k otherwise.
        """
        if not query or k <= 0:
            return []
        vec = self.embed_text(query)
        if not vec:
            return []
        edition = edition_id or self._default_edition_id

        self._ensure_pg()
        if self._pg is not None:
            try:
                pg_results = self._pg.search(edition, vec, k=k)
                if pg_results:
                    return [
                        {
                            "factor_id": r.get("factor_id"),
                            "edition_id": edition,
                            "similarity": float(r.get("similarity", 0.0)),
                            "distance": float(r.get("distance", 1.0)),
                            "search_text": r.get("search_text", ""),
                            "metadata": {},
                        }
                        for r in pg_results
                    ]
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning(
                    "SemanticIndex.search: pgvector failed (%s); "
                    "falling back to in-memory.", exc,
                )

        return self._memory.search(
            query_vector=vec, k=k, edition_id=edition, filters=filters,
        )

    def delete(
        self,
        factor_id: str,
        *,
        edition_id: Optional[str] = None,
    ) -> bool:
        """Remove ``factor_id`` from both backends.  Returns True if present."""
        edition = edition_id or self._default_edition_id
        removed = self._memory.delete(factor_id, edition)

        self._ensure_pg()
        if self._pg is not None:
            try:
                # PgVectorSemanticIndex exposes delete_edition() but not
                # per-factor delete; use direct SQL for a precise removal.
                conn = getattr(self._pg, "_conn", None)
                if conn is not None:
                    with conn.cursor() as cur:
                        cur.execute(
                            "DELETE FROM %s WHERE edition_id = %%s "
                            "AND factor_id = %%s" % self._pg.full_table_name,
                            (edition, factor_id),
                        )
                        if cur.rowcount:
                            removed = True
                    conn.commit()
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning(
                    "SemanticIndex.delete: pgvector delete failed (%s)", exc,
                )
        return removed

    def health(self) -> Dict[str, Any]:
        """Return a lightweight backend health report."""
        self._ensure_pg()
        return {
            "embedder": type(self._embedder).__name__,
            "memory_records": self._memory.size(),
            "default_edition_id": self._default_edition_id,
            "pg_backend": {
                "configured": bool(self._pg_dsn),
                "connected": self._pg is not None,
                "table": (
                    getattr(self._pg, "full_table_name", None)
                    if self._pg is not None
                    else None
                ),
            },
        }


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------


_default_index: Optional[SemanticIndex] = None
_default_index_lock = threading.RLock()


def get_default_semantic_index() -> SemanticIndex:
    """Return a process-wide default :class:`SemanticIndex` (lazy init)."""
    global _default_index
    with _default_index_lock:
        if _default_index is None:
            _default_index = SemanticIndex()
        return _default_index


__all__ = [
    "SemanticIndex",
    "SemanticIndexProtocol",
    "NoopSemanticIndex",
    "get_embedder",
    "get_default_semantic_index",
]
