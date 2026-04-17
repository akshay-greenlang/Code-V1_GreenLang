# -*- coding: utf-8 -*-
"""
Embedding pipeline for emission factor semantic search (F041).

Constructs search text for each factor, generates embeddings via
a pluggable model backend, and caches results by content_hash.

Supported backends:
- sentence-transformers (local, default: MiniLM-L6-v2, 384d)
- stub (for testing, returns deterministic pseudo-vectors)

Usage:
    pipeline = EmbeddingPipeline.from_config(EmbeddingConfig())
    records = pipeline.embed_factors(edition_id, factors)
    # Then: pgvector_index.upsert_embeddings(edition_id, records)
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from greenlang.factors.matching.pgvector_index import (
    DEFAULT_EMBEDDING_DIM,
    EmbeddingRecord,
)

logger = logging.getLogger(__name__)


# ============================================================
# Search text construction
# ============================================================


def build_search_text(factor: Any) -> str:
    """
    Construct search text from an emission factor record or dict.

    Template: "{fuel_type} {geography} {scope} {boundary} {tags} {notes}"
    """
    if isinstance(factor, dict):
        return _search_text_from_dict(factor)
    return _search_text_from_record(factor)


def _search_text_from_record(rec: Any) -> str:
    parts = [
        getattr(rec, "fuel_type", "") or "",
        getattr(rec, "geography", "") or "",
    ]
    scope = getattr(rec, "scope", None)
    if scope:
        parts.append(scope.value if hasattr(scope, "value") else str(scope))
    boundary = getattr(rec, "boundary", None)
    if boundary:
        parts.append(boundary.value if hasattr(boundary, "value") else str(boundary))
    tags = getattr(rec, "tags", []) or []
    parts.extend(tags)
    notes = getattr(rec, "notes", "") or ""
    if notes:
        parts.append(notes)
    unit = getattr(rec, "unit", "") or ""
    if unit:
        parts.append(unit)
    prov = getattr(rec, "provenance", None)
    if prov:
        org = getattr(prov, "source_org", "") or ""
        if org:
            parts.append(org)
    return " ".join(p for p in parts if p).strip()


def _search_text_from_dict(d: Dict[str, Any]) -> str:
    parts = [
        d.get("fuel_type", "") or "",
        d.get("geography", "") or "",
        d.get("scope", "") or "",
        d.get("boundary", "") or "",
    ]
    tags = d.get("tags", []) or []
    if isinstance(tags, list):
        parts.extend(tags)
    notes = d.get("notes", "") or ""
    if notes:
        parts.append(notes)
    unit = d.get("unit", "") or ""
    if unit:
        parts.append(unit)
    source_org = d.get("source_org", "") or ""
    if source_org:
        parts.append(source_org)
    return " ".join(p for p in parts if p).strip()


# ============================================================
# Embedding model protocol and implementations
# ============================================================


class EmbeddingModel(Protocol):
    """Protocol for embedding models."""

    @property
    def dimension(self) -> int: ...

    def encode(self, texts: List[str]) -> List[List[float]]: ...

    def encode_single(self, text: str) -> List[float]: ...


class StubEmbeddingModel:
    """
    Deterministic stub model for testing.

    Produces pseudo-embeddings based on text hash, ensuring:
    - Same text -> same vector (deterministic)
    - Different text -> different vector (high probability)
    - Unit-normalized vectors (cosine similarity meaningful)
    """

    def __init__(self, dim: int = DEFAULT_EMBEDDING_DIM):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def encode_single(self, text: str) -> List[float]:
        h = hashlib.sha256(text.lower().strip().encode("utf-8")).digest()
        raw = []
        for i in range(self._dim):
            byte_idx = i % len(h)
            # Convert byte to float in [-1, 1]
            raw.append((h[byte_idx] - 128) / 128.0)
        # L2-normalize
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    def encode(self, texts: List[str]) -> List[List[float]]:
        return [self.encode_single(t) for t in texts]


class SentenceTransformerModel:
    """
    Embedding model using sentence-transformers library.

    Default: all-MiniLM-L6-v2 (384d).
    Requires: pip install sentence-transformers
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
            logger.info(
                "SentenceTransformer loaded: model=%s dim=%d",
                model_name, self._dim,
            )
        except ImportError:
            raise ImportError(
                "sentence-transformers required: pip install sentence-transformers"
            )

    @property
    def dimension(self) -> int:
        return self._dim

    def encode_single(self, text: str) -> List[float]:
        result = self._model.encode([text], normalize_embeddings=True)
        return result[0].tolist()

    def encode(self, texts: List[str]) -> List[List[float]]:
        result = self._model.encode(texts, normalize_embeddings=True, batch_size=64)
        return [v.tolist() for v in result]


# ============================================================
# Embedding pipeline
# ============================================================


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding pipeline."""

    model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    batch_size: int = 1000
    use_stub: bool = False


@dataclass
class EmbeddingStats:
    """Statistics from an embedding run."""

    total_factors: int = 0
    embedded: int = 0
    skipped_cached: int = 0
    errors: int = 0


class EmbeddingPipeline:
    """
    End-to-end pipeline: factors -> search text -> embeddings -> records.

    Supports incremental embedding by checking content_hash cache.
    """

    def __init__(self, model: EmbeddingModel, config: EmbeddingConfig):
        self._model = model
        self._config = config
        self._cache: Dict[str, List[float]] = {}

    @classmethod
    def from_config(cls, config: Optional[EmbeddingConfig] = None) -> EmbeddingPipeline:
        config = config or EmbeddingConfig()
        if config.use_stub:
            model = StubEmbeddingModel(dim=config.embedding_dim)
        else:
            try:
                model = SentenceTransformerModel(config.model_name)
            except ImportError:
                logger.warning(
                    "sentence-transformers not available, falling back to stub model"
                )
                model = StubEmbeddingModel(dim=config.embedding_dim)
        return cls(model, config)

    @property
    def dimension(self) -> int:
        return self._model.dimension

    def embed_text(self, text: str) -> List[float]:
        """Embed a single query text."""
        return self._model.encode_single(text)

    def embed_factors(
        self,
        edition_id: str,
        factors: Sequence[Any],
        *,
        cached_hashes: Optional[Dict[str, str]] = None,
    ) -> tuple[List[EmbeddingRecord], EmbeddingStats]:
        """
        Embed a list of factors, skipping those already cached.

        Args:
            edition_id: Edition ID for the records
            factors: Sequence of factor records or dicts
            cached_hashes: Dict of factor_id -> content_hash for already-embedded factors

        Returns:
            Tuple of (list of EmbeddingRecord, EmbeddingStats)
        """
        cached_hashes = cached_hashes or {}
        stats = EmbeddingStats(total_factors=len(factors))

        # Build (factor_id, search_text, content_hash) tuples for factors that need embedding
        to_embed: List[tuple[str, str, str]] = []
        for f in factors:
            if isinstance(f, dict):
                fid = f.get("factor_id", "")
                ch = f.get("content_hash", "")
            else:
                fid = getattr(f, "factor_id", "")
                ch = getattr(f, "content_hash", "")

            # Skip if content_hash unchanged
            if fid in cached_hashes and cached_hashes[fid] == ch:
                stats.skipped_cached += 1
                continue

            search_text = build_search_text(f)
            to_embed.append((fid, search_text, ch))

        # Batch embed
        records: List[EmbeddingRecord] = []
        for i in range(0, len(to_embed), self._config.batch_size):
            batch = to_embed[i: i + self._config.batch_size]
            texts = [t[1] for t in batch]
            try:
                vectors = self._model.encode(texts)
                for (fid, st, ch), vec in zip(batch, vectors):
                    records.append(EmbeddingRecord(
                        edition_id=edition_id,
                        factor_id=fid,
                        embedding=vec,
                        search_text=st,
                        content_hash=ch,
                    ))
                    stats.embedded += 1
            except Exception as exc:
                logger.error(
                    "Embedding batch failed (offset=%d, size=%d): %s",
                    i, len(batch), exc,
                )
                stats.errors += len(batch)

        logger.info(
            "Embedding pipeline: edition=%s total=%d embedded=%d cached=%d errors=%d",
            edition_id, stats.total_factors, stats.embedded,
            stats.skipped_cached, stats.errors,
        )
        return records, stats

    def embed_query(self, query: str) -> List[float]:
        """Embed a search query for vector similarity lookup."""
        return self._model.encode_single(query)
