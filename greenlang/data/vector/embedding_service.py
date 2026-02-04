"""
Embedding generation service with batch processing, caching, and pgvector storage.

Supports multiple embedding models (MiniLM, MPNet, OpenAI) with automatic
batching, retry logic, and optional Redis caching for repeated texts.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Dict, List, Optional

import numpy as np

from greenlang.data.vector.config import EmbeddingConfig, EMBEDDING_MODELS
from greenlang.data.vector.connection import VectorDBConnection
from greenlang.data.vector.models import (
    BatchInsertResult,
    EmbeddingRequest,
    EmbeddingResult,
    VectorRecord,
)

logger = logging.getLogger(__name__)


class EmbeddingModelLoader:
    """Lazy-loads and caches embedding models."""

    def __init__(self):
        self._models: Dict[str, object] = {}

    def get_model(self, model_name: str, device: str = "cpu"):
        """Load or retrieve cached model."""
        cache_key = f"{model_name}:{device}"
        if cache_key not in self._models:
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(model_name, device=device)
                self._models[cache_key] = model
                logger.info("Loaded embedding model: %s on %s", model_name, device)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
        return self._models[cache_key]


# Global model loader (singleton)
_model_loader = EmbeddingModelLoader()


class EmbeddingCache:
    """Optional Redis-based cache for embedding results."""

    def __init__(self, redis_url: Optional[str] = None, ttl: int = 3600):
        self._client = None
        self._ttl = ttl
        if redis_url:
            try:
                import redis
                self._client = redis.from_url(redis_url)
                logger.info("Embedding cache connected to Redis")
            except ImportError:
                logger.warning("redis package not installed, caching disabled")

    def _cache_key(self, text: str, model: str) -> str:
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"emb:{model}:{text_hash}"

    def get(self, text: str, model: str) -> Optional[np.ndarray]:
        if not self._client:
            return None
        try:
            key = self._cache_key(text, model)
            data = self._client.get(key)
            if data:
                return np.frombuffer(data, dtype=np.float32)
        except Exception as e:
            logger.debug("Cache get failed: %s", e)
        return None

    def put(self, text: str, model: str, embedding: np.ndarray) -> None:
        if not self._client:
            return
        try:
            key = self._cache_key(text, model)
            self._client.setex(key, self._ttl, embedding.tobytes())
        except Exception as e:
            logger.debug("Cache put failed: %s", e)

    def get_batch(
        self, texts: List[str], model: str
    ) -> Dict[int, np.ndarray]:
        """Get cached embeddings for a batch. Returns {index: embedding}."""
        result = {}
        if not self._client:
            return result
        for i, text in enumerate(texts):
            cached = self.get(text, model)
            if cached is not None:
                result[i] = cached
        return result


class EmbeddingService:
    """
    Production embedding service with:
    - Batch processing (configurable batch size, default 1000)
    - GPU acceleration (optional)
    - Redis caching for repeated texts
    - Automatic retries with exponential backoff
    - pgvector storage integration
    """

    def __init__(
        self,
        config: EmbeddingConfig,
        db: Optional[VectorDBConnection] = None,
        redis_url: Optional[str] = None,
    ):
        self.config = config
        self.db = db
        self._cache = EmbeddingCache(
            redis_url=redis_url, ttl=config.cache_ttl
        ) if config.cache_enabled else EmbeddingCache()

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Generate embeddings for a list of texts."""
        model_name = request.model_name or self.config.model_name
        start_time = time.monotonic()

        model = _model_loader.get_model(model_name, self.config.device)
        spec = EMBEDDING_MODELS.get(model_name)
        dimensions = spec.dimensions if spec else self.config.dimensions

        # Check cache for already-computed embeddings
        cached = self._cache.get_batch(request.texts, model_name)
        uncached_indices = [
            i for i in range(len(request.texts)) if i not in cached
        ]
        uncached_texts = [request.texts[i] for i in uncached_indices]

        # Generate embeddings for uncached texts in batches
        all_embeddings = np.zeros(
            (len(request.texts), dimensions), dtype=np.float32
        )

        # Fill in cached embeddings
        for idx, emb in cached.items():
            all_embeddings[idx] = emb

        # Process uncached texts in batches
        if uncached_texts:
            for batch_start in range(0, len(uncached_texts), self.config.batch_size):
                batch_end = min(
                    batch_start + self.config.batch_size, len(uncached_texts)
                )
                batch = uncached_texts[batch_start:batch_end]

                batch_embeddings = self._encode_with_retry(model, batch)

                for i, emb in enumerate(batch_embeddings):
                    original_idx = uncached_indices[batch_start + i]
                    all_embeddings[original_idx] = emb

                    # Cache the result
                    self._cache.put(
                        request.texts[original_idx], model_name, emb
                    )

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        return EmbeddingResult(
            embeddings=all_embeddings,
            model=model_name,
            dimensions=dimensions,
            processing_time_ms=elapsed_ms,
        )

    def _encode_with_retry(
        self, model, texts: List[str], attempt: int = 0
    ) -> np.ndarray:
        """Encode texts with exponential backoff retry."""
        try:
            embeddings = model.encode(
                texts,
                normalize_embeddings=self.config.normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embeddings.astype(np.float32)
        except Exception as e:
            if attempt < self.config.max_retries:
                delay = self.config.retry_delay * (2 ** attempt)
                logger.warning(
                    "Embedding encode failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    self.config.max_retries,
                    delay,
                    e,
                )
                import asyncio
                # Use synchronous sleep since encode is synchronous
                time.sleep(delay)
                return self._encode_with_retry(model, texts, attempt + 1)
            raise

    async def embed_and_store(self, request: EmbeddingRequest) -> BatchInsertResult:
        """Generate embeddings and store in pgvector."""
        if not self.db:
            raise RuntimeError("Database connection not configured")

        # Generate embeddings
        result = await self.embed(request)

        # Build records
        source_id = request.source_id or str(__import__("uuid").uuid4())
        records = []
        for i, text in enumerate(request.texts):
            records.append(
                VectorRecord(
                    source_type=request.source_type,
                    source_id=source_id,
                    content=text,
                    embedding=result.embeddings[i],
                    namespace=request.namespace,
                    chunk_index=i,
                    metadata=request.metadata or {},
                    embedding_model=result.model,
                    content_hash=hashlib.sha256(text.encode()).hexdigest(),
                )
            )

        # Store in pgvector
        return await self._batch_insert(records)

    async def _batch_insert(self, records: List[VectorRecord]) -> BatchInsertResult:
        """Insert vector records into pgvector in batches."""
        start_time = time.monotonic()
        inserted = 0
        failed = 0
        duplicates = 0
        errors = []

        for batch_start in range(0, len(records), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(records))
            batch = records[batch_start:batch_end]

            try:
                async with self.db.acquire_writer() as conn:
                    async with conn.cursor() as cur:
                        for record in batch:
                            try:
                                await cur.execute(
                                    """
                                    INSERT INTO vector_embeddings (
                                        id, source_type, source_id, chunk_index,
                                        content_hash, content_preview, embedding,
                                        embedding_model, metadata, namespace,
                                        collection_id
                                    ) VALUES (
                                        %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s
                                    )
                                    ON CONFLICT (source_type, source_id, chunk_index, embedding_model)
                                    DO UPDATE SET
                                        content_hash = EXCLUDED.content_hash,
                                        content_preview = EXCLUDED.content_preview,
                                        embedding = EXCLUDED.embedding,
                                        metadata = EXCLUDED.metadata,
                                        updated_at = NOW()
                                    """,
                                    (
                                        record.id,
                                        record.source_type,
                                        record.source_id,
                                        record.chunk_index,
                                        record.content_hash,
                                        record.content[:500] if record.content else None,
                                        record.embedding,
                                        record.embedding_model,
                                        __import__("json").dumps(record.metadata),
                                        record.namespace,
                                        record.collection_id,
                                    ),
                                )
                                inserted += 1
                            except Exception as e:
                                if "unique" in str(e).lower():
                                    duplicates += 1
                                else:
                                    failed += 1
                                    errors.append(f"Record {record.id}: {e}")
                                    logger.error("Insert failed for %s: %s", record.id, e)
                    await conn.commit()
            except Exception as e:
                failed += len(batch) - inserted
                errors.append(f"Batch error: {e}")
                logger.error("Batch insert failed: %s", e)

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        return BatchInsertResult(
            total_count=len(records),
            inserted_count=inserted,
            failed_count=failed,
            duplicate_count=duplicates,
            processing_time_ms=elapsed_ms,
            errors=errors,
        )

    async def get_embedding(self, text: str) -> np.ndarray:
        """Get a single text embedding (for query embedding)."""
        result = await self.embed(
            EmbeddingRequest(texts=[text])
        )
        return result.embeddings[0]
