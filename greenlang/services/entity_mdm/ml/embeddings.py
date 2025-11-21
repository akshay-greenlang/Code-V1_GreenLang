# -*- coding: utf-8 -*-
"""
Embedding generation pipeline for entity resolution.

This module implements sentence-transformer based embeddings with caching,
batch processing, and normalization for efficient similarity search.

Author: GreenLang AI
Phase: 5 - Entity Resolution ML
"""

from typing import List, Optional, Union, Dict, Any
import numpy as np
import hashlib
import json
import logging
from pathlib import Path
import redis

# Runtime check for optional ML dependencies
try:
    from greenlang.utils.ml_imports import check_ml_dependencies
    check_ml_dependencies("Entity Resolution Embedding Pipeline")
except ImportError as e:
    raise ImportError(
        "\n" + "=" * 80 + "\n"
        "Missing Optional Dependencies: ML Features\n"
        "=" * 80 + "\n\n"
        "Entity resolution requires PyTorch and sentence-transformers.\n\n"
        "To install ML dependencies, run:\n"
        "  pip install greenlang-cli[ml]\n\n"
        "Or install all AI capabilities:\n"
        "  pip install greenlang-cli[ai-full]\n"
        + "=" * 80 + "\n"
    ) from e

from sentence_transformers import SentenceTransformer
import torch

from entity_mdm.ml.config import MLConfig, ModelConfig, CacheConfig
from entity_mdm.ml.exceptions import EmbeddingException

logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    """
    Sentence-transformer based embedding pipeline with caching.

    This class handles:
    - Loading and managing sentence transformer models
    - Batch processing for efficient embedding generation
    - Redis caching to avoid recomputation
    - Text normalization and preprocessing
    """

    def __init__(self, config: Optional[MLConfig] = None) -> None:
        """
        Initialize the embedding pipeline.

        Args:
            config: ML configuration object. If None, uses defaults.
        """
        self.config = config or MLConfig()
        self.model_config: ModelConfig = self.config.model
        self.cache_config: CacheConfig = self.config.cache

        # Initialize model
        self._model: Optional[SentenceTransformer] = None
        self._device = self._get_device()

        # Initialize cache
        self._cache: Optional[redis.Redis] = None
        if self.cache_config.enabled:
            self._init_cache()

        # Statistics
        self._stats = {
            "embeddings_generated": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        logger.info(
            f"Initialized EmbeddingPipeline with model={self.model_config.embedding_model}, "
            f"device={self._device}, caching={'enabled' if self.cache_config.enabled else 'disabled'}"
        )

    def _get_device(self) -> str:
        """
        Determine the appropriate device for computation.

        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        if self.model_config.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif self.model_config.device == "mps" and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _init_cache(self) -> None:
        """Initialize Redis connection for caching."""
        try:
            self._cache = redis.Redis(
                host=self.cache_config.host,
                port=self.cache_config.port,
                db=self.cache_config.db,
                password=self.cache_config.password,
                max_connections=self.cache_config.max_connections,
                socket_timeout=self.cache_config.socket_timeout,
                decode_responses=False,  # We'll handle binary data
            )
            # Test connection
            self._cache.ping()
            logger.info("Redis cache connection established")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis cache: {e}. Caching disabled.")
            self._cache = None

    @property
    def model(self) -> SentenceTransformer:
        """
        Lazy-load the sentence transformer model.

        Returns:
            Loaded SentenceTransformer model

        Raises:
            EmbeddingException: If model loading fails
        """
        if self._model is None:
            try:
                logger.info(f"Loading embedding model: {self.model_config.embedding_model}")
                self._model = SentenceTransformer(
                    self.model_config.embedding_model,
                    cache_folder=str(self.model_config.model_cache_dir),
                    device=self._device,
                )
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                raise EmbeddingException(
                    message=f"Failed to load embedding model: {e}",
                    details={"model": self.model_config.embedding_model},
                )
        return self._model

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for embedding.

        Args:
            text: Raw input text

        Returns:
            Normalized text
        """
        # Strip whitespace
        text = text.strip()

        # Convert to lowercase for consistency
        text = text.lower()

        # Remove extra whitespace
        text = " ".join(text.split())

        return text

    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key for text.

        Args:
            text: Input text

        Returns:
            Cache key (hash of normalized text)
        """
        normalized = self._normalize_text(text)
        return f"emb:{hashlib.sha256(normalized.encode()).hexdigest()}"

    def _get_from_cache(self, text: str) -> Optional[np.ndarray]:
        """
        Retrieve embedding from cache.

        Args:
            text: Input text

        Returns:
            Cached embedding or None if not found
        """
        if not self._cache:
            return None

        try:
            cache_key = self._get_cache_key(text)
            cached = self._cache.get(cache_key)
            if cached:
                self._stats["cache_hits"] += 1
                return np.frombuffer(cached, dtype=np.float32)
            else:
                self._stats["cache_misses"] += 1
                return None
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None

    def _save_to_cache(self, text: str, embedding: np.ndarray) -> None:
        """
        Save embedding to cache.

        Args:
            text: Input text
            embedding: Generated embedding
        """
        if not self._cache:
            return

        try:
            cache_key = self._get_cache_key(text)
            # Convert to bytes for storage
            embedding_bytes = embedding.astype(np.float32).tobytes()
            self._cache.setex(
                cache_key,
                self.cache_config.embedding_ttl,
                embedding_bytes,
            )
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for input texts.

        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for processing. Uses config default if None.
            normalize: Whether to L2-normalize embeddings
            show_progress: Show progress bar for large batches

        Returns:
            Single embedding array or list of embedding arrays

        Raises:
            EmbeddingException: If embedding generation fails
        """
        # Handle single string input
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        # Check cache for each text
        embeddings: List[Optional[np.ndarray]] = []
        texts_to_embed: List[str] = []
        indices_to_embed: List[int] = []

        for i, text in enumerate(texts):
            if not text or not text.strip():
                # Empty text gets zero vector
                embeddings.append(np.zeros(self.model_config.embedding_dimension))
            else:
                cached = self._get_from_cache(text)
                if cached is not None:
                    embeddings.append(cached)
                else:
                    embeddings.append(None)
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)

        # Generate embeddings for cache misses
        if texts_to_embed:
            try:
                batch_size = batch_size or self.model_config.batch_size
                new_embeddings = self.model.encode(
                    texts_to_embed,
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    normalize_embeddings=normalize,
                    convert_to_numpy=True,
                )

                # Update embeddings list and cache
                for idx, embedding in zip(indices_to_embed, new_embeddings):
                    embeddings[idx] = embedding
                    self._save_to_cache(texts[idx], embedding)

                self._stats["embeddings_generated"] += len(texts_to_embed)

            except Exception as e:
                raise EmbeddingException(
                    message=f"Failed to generate embeddings: {e}",
                    details={
                        "num_texts": len(texts_to_embed),
                        "batch_size": batch_size,
                    },
                )

        # Return single embedding or list
        result = embeddings[0] if single_input else embeddings
        return result

    def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of texts (optimized for large batches).

        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            show_progress: Show progress bar

        Returns:
            2D numpy array of embeddings (num_texts x embedding_dim)

        Raises:
            EmbeddingException: If embedding generation fails
        """
        embeddings = self.embed(
            texts,
            batch_size=batch_size,
            normalize=True,
            show_progress=show_progress,
        )
        return np.array(embeddings)

    def similarity(
        self,
        text1: str,
        text2: str,
        metric: str = "cosine",
    ) -> float:
        """
        Calculate similarity between two texts.

        Args:
            text1: First text
            text2: Second text
            metric: Similarity metric ('cosine' or 'dot')

        Returns:
            Similarity score (0.0 to 1.0)

        Raises:
            EmbeddingException: If similarity calculation fails
        """
        try:
            emb1, emb2 = self.embed([text1, text2])

            if metric == "cosine":
                # Cosine similarity (embeddings are already normalized)
                return float(np.dot(emb1, emb2))
            elif metric == "dot":
                # Dot product
                return float(np.dot(emb1, emb2))
            else:
                raise ValueError(f"Unknown metric: {metric}")

        except Exception as e:
            raise EmbeddingException(
                message=f"Failed to calculate similarity: {e}",
                details={"metric": metric},
            )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.

        Returns:
            Dictionary with statistics
        """
        stats = self._stats.copy()
        if self._cache:
            cache_total = stats["cache_hits"] + stats["cache_misses"]
            stats["cache_hit_rate"] = (
                stats["cache_hits"] / cache_total if cache_total > 0 else 0.0
            )
        return stats

    def clear_cache(self) -> int:
        """
        Clear all cached embeddings.

        Returns:
            Number of keys cleared

        Raises:
            EmbeddingException: If cache clearing fails
        """
        if not self._cache:
            return 0

        try:
            # Find all embedding cache keys
            keys = self._cache.keys("emb:*")
            if keys:
                count = self._cache.delete(*keys)
                logger.info(f"Cleared {count} cached embeddings")
                return count
            return 0
        except Exception as e:
            raise EmbeddingException(
                message=f"Failed to clear cache: {e}",
            )

    def __del__(self) -> None:
        """Cleanup resources."""
        if self._cache:
            try:
                self._cache.close()
            except Exception:
                pass
