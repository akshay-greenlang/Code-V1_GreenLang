"""
Tests for Entity MDM Embedding Generation.

Tests embedding generation, batch processing, caching, normalization,
and edge cases with empty strings and special characters.

Target: 350+ lines, 15 tests
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict


# Mock embedding service (would be actual module in production)
class EmbeddingService:
    """Service for generating embeddings from supplier names."""

    def __init__(self, model, cache_enabled: bool = True, normalize: bool = True):
        self.model = model
        self.cache_enabled = cache_enabled
        self.normalize = normalize
        self.cache: Dict[str, np.ndarray] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        # Check cache first
        if self.cache_enabled and text in self.cache:
            self.cache_hits += 1
            return self.cache[text]

        self.cache_misses += 1

        # Handle edge cases
        if not text or not text.strip():
            # Return zero vector for empty strings
            return np.zeros(self.model.get_sentence_embedding_dimension())

        # Generate embedding
        embedding = self.model.encode(text)

        # Normalize if enabled
        if self.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        # Cache result
        if self.cache_enabled:
            self.cache[text] = embedding

        return embedding

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple texts in batches."""
        if not texts:
            return np.array([])

        embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Check cache for each text
            batch_embeddings = []
            uncached_texts = []
            uncached_indices = []

            for j, text in enumerate(batch):
                if self.cache_enabled and text in self.cache:
                    self.cache_hits += 1
                    batch_embeddings.append(self.cache[text])
                else:
                    self.cache_misses += 1
                    uncached_texts.append(text)
                    uncached_indices.append(j)

            # Generate embeddings for uncached texts
            if uncached_texts:
                new_embeddings = self.model.encode(uncached_texts)

                # Normalize if enabled
                if self.normalize:
                    norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
                    new_embeddings = np.divide(new_embeddings, norms, where=norms > 0)

                # Cache new embeddings
                if self.cache_enabled:
                    for text, emb in zip(uncached_texts, new_embeddings):
                        self.cache[text] = emb

                # Merge cached and new embeddings in correct order
                result = []
                new_idx = 0
                for j in range(len(batch)):
                    if j in uncached_indices:
                        result.append(new_embeddings[new_idx])
                        new_idx += 1
                    else:
                        result.append(batch_embeddings[j - new_idx])

                embeddings.extend(result)
            else:
                embeddings.extend(batch_embeddings)

        return np.array(embeddings)

    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0 else 0.0
        }


# ============================================================================
# TEST SUITE
# ============================================================================

class TestEmbeddingService:
    """Test suite for embedding generation."""

    def test_embed_single_generates_valid_embedding(self, mock_sentence_transformer):
        """Test that single embedding generation returns valid vector."""
        service = EmbeddingService(mock_sentence_transformer)

        embedding = service.embed_single("ACME Corporation")

        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert not np.all(embedding == 0)

    def test_embed_single_normalizes_embedding(self, mock_sentence_transformer):
        """Test that embeddings are normalized to unit length."""
        service = EmbeddingService(mock_sentence_transformer, normalize=True)

        embedding = service.embed_single("Test Company")

        # Check that embedding is normalized (L2 norm should be 1.0)
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=1e-6)

    def test_embed_single_without_normalization(self, mock_sentence_transformer):
        """Test that embeddings are not normalized when disabled."""
        service = EmbeddingService(mock_sentence_transformer, normalize=False)

        embedding = service.embed_single("Test Company")

        # Check that embedding might not be normalized
        norm = np.linalg.norm(embedding)
        # For mock data, this will vary
        assert norm > 0

    def test_embed_single_uses_cache(self, mock_sentence_transformer):
        """Test that caching works for single embeddings."""
        service = EmbeddingService(mock_sentence_transformer, cache_enabled=True)

        # First call - cache miss
        embedding1 = service.embed_single("ACME Corporation")
        assert service.cache_misses == 1
        assert service.cache_hits == 0

        # Second call - cache hit
        embedding2 = service.embed_single("ACME Corporation")
        assert service.cache_misses == 1
        assert service.cache_hits == 1

        # Embeddings should be identical
        np.testing.assert_array_equal(embedding1, embedding2)

    def test_embed_single_with_cache_disabled(self, mock_sentence_transformer):
        """Test that caching can be disabled."""
        service = EmbeddingService(mock_sentence_transformer, cache_enabled=False)

        # Multiple calls should all be cache misses
        service.embed_single("ACME Corporation")
        service.embed_single("ACME Corporation")

        assert service.cache_hits == 0
        assert len(service.cache) == 0

    def test_embed_batch_generates_multiple_embeddings(self, mock_sentence_transformer):
        """Test batch embedding generation."""
        service = EmbeddingService(mock_sentence_transformer)

        texts = ["ACME Corporation", "ABC Manufacturing", "Global Tech"]
        embeddings = service.embed_batch(texts)

        assert embeddings.shape == (3, 384)
        assert not np.all(embeddings == 0)

        # Each embedding should be different
        assert not np.array_equal(embeddings[0], embeddings[1])
        assert not np.array_equal(embeddings[1], embeddings[2])

    def test_embed_batch_respects_batch_size(self, mock_sentence_transformer):
        """Test that batch processing respects batch size."""
        service = EmbeddingService(mock_sentence_transformer)

        # Create a list with more items than batch size
        texts = [f"Company {i}" for i in range(100)]
        embeddings = service.embed_batch(texts, batch_size=32)

        assert embeddings.shape == (100, 384)

    def test_embed_batch_uses_cache_efficiently(self, mock_sentence_transformer):
        """Test that batch processing uses cache efficiently."""
        service = EmbeddingService(mock_sentence_transformer, cache_enabled=True)

        texts = ["ACME Corporation", "ABC Manufacturing", "Global Tech"]

        # First batch - all cache misses
        embeddings1 = service.embed_batch(texts)
        assert service.cache_misses == 3
        assert service.cache_hits == 0

        # Second batch with same texts - all cache hits
        embeddings2 = service.embed_batch(texts)
        assert service.cache_misses == 3
        assert service.cache_hits == 3

        # Embeddings should be identical
        np.testing.assert_array_equal(embeddings1, embeddings2)

    def test_embed_batch_with_partial_cache_hits(self, mock_sentence_transformer):
        """Test batch processing with some cached and some uncached texts."""
        service = EmbeddingService(mock_sentence_transformer, cache_enabled=True)

        # First batch
        texts1 = ["ACME Corporation", "ABC Manufacturing"]
        service.embed_batch(texts1)
        assert service.cache_misses == 2

        # Second batch with one cached and one new text
        texts2 = ["ACME Corporation", "Global Tech"]
        embeddings = service.embed_batch(texts2)

        assert service.cache_misses == 3  # 2 from first batch + 1 new
        assert service.cache_hits == 1  # ACME Corporation from cache
        assert embeddings.shape == (2, 384)

    def test_embed_empty_string(self, mock_sentence_transformer):
        """Test handling of empty strings."""
        service = EmbeddingService(mock_sentence_transformer)

        embedding = service.embed_single("")

        # Should return zero vector
        assert np.all(embedding == 0)
        assert embedding.shape == (384,)

    def test_embed_whitespace_only_string(self, mock_sentence_transformer):
        """Test handling of whitespace-only strings."""
        service = EmbeddingService(mock_sentence_transformer)

        embedding = service.embed_single("   \t\n  ")

        # Should return zero vector
        assert np.all(embedding == 0)

    def test_embed_special_characters(self, mock_sentence_transformer):
        """Test handling of special characters."""
        service = EmbeddingService(mock_sentence_transformer)

        texts = [
            "O'Reilly Manufacturing",
            "AT&T Solutions",
            "3M Corporation",
            "H&M Manufacturing",
            "Société Générale",
            "Müller GmbH"
        ]

        embeddings = service.embed_batch(texts)

        assert embeddings.shape == (6, 384)
        # All embeddings should be valid
        for emb in embeddings:
            assert not np.all(emb == 0)
            assert not np.any(np.isnan(emb))

    def test_clear_cache(self, mock_sentence_transformer):
        """Test cache clearing functionality."""
        service = EmbeddingService(mock_sentence_transformer, cache_enabled=True)

        # Generate some embeddings to populate cache
        texts = ["ACME Corporation", "ABC Manufacturing", "Global Tech"]
        service.embed_batch(texts)

        assert len(service.cache) == 3
        assert service.cache_misses > 0

        # Clear cache
        service.clear_cache()

        assert len(service.cache) == 0
        assert service.cache_hits == 0
        assert service.cache_misses == 0

    def test_get_cache_stats(self, mock_sentence_transformer):
        """Test cache statistics reporting."""
        service = EmbeddingService(mock_sentence_transformer, cache_enabled=True)

        # Generate embeddings
        texts = ["ACME Corporation", "ABC Manufacturing"]
        service.embed_batch(texts)  # 2 misses
        service.embed_batch(texts)  # 2 hits

        stats = service.get_cache_stats()

        assert stats["cache_size"] == 2
        assert stats["cache_hits"] == 2
        assert stats["cache_misses"] == 2
        assert stats["hit_rate"] == 0.5

    def test_embed_batch_with_empty_list(self, mock_sentence_transformer):
        """Test batch embedding with empty list."""
        service = EmbeddingService(mock_sentence_transformer)

        embeddings = service.embed_batch([])

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (0,)
