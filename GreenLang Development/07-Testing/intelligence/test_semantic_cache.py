# -*- coding: utf-8 -*-
"""
Tests for Semantic Cache

Tests:
- Embedding generation
- Similarity search
- Cache hit/miss
- TTL expiration
- Performance benchmarks
"""

import time
from datetime import datetime, timedelta

import numpy as np
import pytest

from greenlang.determinism import DeterministicClock
from greenlang.intelligence.semantic_cache import (
    CacheEntry,
    CacheMetrics,
    EmbeddingGenerator,
    FAISSIndex,
    InMemoryCache,
    SemanticCache,
    get_global_cache,
)


class TestEmbeddingGenerator:
    """Test embedding generation"""

    def test_init(self):
        """Test initialization"""
        embedder = EmbeddingGenerator()
        assert embedder.embedding_dim == 384  # all-MiniLM-L6-v2 dimension

    def test_encode_single(self):
        """Test encoding single text"""
        embedder = EmbeddingGenerator()

        text = "What is the carbon footprint of natural gas?"
        embedding = embedder.encode(text)

        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32

        # Check normalized (L2 norm should be close to 1)
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01

    def test_encode_batch(self):
        """Test encoding batch of texts"""
        embedder = EmbeddingGenerator()

        texts = [
            "What is the carbon footprint of natural gas?",
            "Calculate emissions for electricity",
            "Recommend solar panel options",
        ]

        embeddings = embedder.encode_batch(texts)

        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32

    def test_semantic_similarity(self):
        """Test semantic similarity"""
        embedder = EmbeddingGenerator()

        # Similar texts
        text1 = "What is the carbon footprint of natural gas?"
        text2 = "What's the CO2 emissions from natural gas?"

        # Different text
        text3 = "How to install solar panels?"

        emb1 = embedder.encode(text1)
        emb2 = embedder.encode(text2)
        emb3 = embedder.encode(text3)

        # Cosine similarity (dot product for normalized vectors)
        sim_12 = np.dot(emb1, emb2)
        sim_13 = np.dot(emb1, emb3)

        # Similar texts should have higher similarity
        assert sim_12 > 0.8
        assert sim_13 < sim_12


class TestFAISSIndex:
    """Test FAISS index"""

    @pytest.fixture
    def index(self):
        """Create FAISS index"""
        return FAISSIndex(dimension=384)

    @pytest.fixture
    def embedder(self):
        """Create embedder"""
        return EmbeddingGenerator()

    def test_init(self, index):
        """Test initialization"""
        assert index.dimension == 384
        assert index.size == 0

    def test_add(self, index, embedder):
        """Test adding vectors"""
        text = "What is the carbon footprint?"
        embedding = embedder.encode(text)

        index.add("key1", embedding)

        assert index.size == 1

    def test_search(self, index, embedder):
        """Test similarity search"""
        # Add vectors
        texts = [
            "What is the carbon footprint of natural gas?",
            "Calculate electricity emissions",
            "Recommend solar options",
        ]

        for i, text in enumerate(texts):
            embedding = embedder.encode(text)
            index.add(f"key{i}", embedding)

        # Search for similar text
        query = "What's the CO2 from natural gas?"
        query_embedding = embedder.encode(query)

        results = index.search(query_embedding, k=2)

        assert len(results) == 2

        # First result should be most similar (key0)
        key, similarity = results[0]
        assert key == "key0"
        assert similarity > 0.8


class TestInMemoryCache:
    """Test in-memory cache"""

    @pytest.fixture
    def cache(self):
        """Create in-memory cache"""
        return InMemoryCache()

    def test_set_get(self, cache):
        """Test set and get"""
        entry = CacheEntry(
            embedding=np.zeros(384, dtype=np.float32),
            prompt="test prompt",
            response="test response",
            metadata={},
            timestamp=DeterministicClock.now(),
        )

        cache.set("key1", entry)

        retrieved = cache.get("key1")
        assert retrieved is not None
        assert retrieved.prompt == "test prompt"
        assert retrieved.response == "test response"

    def test_get_nonexistent(self, cache):
        """Test getting non-existent key"""
        result = cache.get("nonexistent")
        assert result is None

    def test_delete(self, cache):
        """Test deletion"""
        entry = CacheEntry(
            embedding=np.zeros(384, dtype=np.float32),
            prompt="test",
            response="test",
            metadata={},
            timestamp=DeterministicClock.now(),
        )

        cache.set("key1", entry)
        cache.delete("key1")

        result = cache.get("key1")
        assert result is None

    def test_ttl_expiration(self, cache):
        """Test TTL expiration"""
        # Create entry with 1 second TTL
        entry = CacheEntry(
            embedding=np.zeros(384, dtype=np.float32),
            prompt="test",
            response="test",
            metadata={},
            timestamp=DeterministicClock.now() - timedelta(seconds=2),  # 2 seconds ago
            ttl_seconds=1,
        )

        cache.set("key1", entry)

        # Should be expired
        result = cache.get("key1")
        assert result is None


class TestSemanticCache:
    """Test semantic cache"""

    @pytest.fixture
    def cache(self):
        """Create semantic cache"""
        return SemanticCache(
            use_redis=False,  # Use in-memory for testing
            similarity_threshold=0.90,
        )

    def test_init(self, cache):
        """Test initialization"""
        assert cache.similarity_threshold == 0.90
        assert isinstance(cache.cache_storage, InMemoryCache)

    def test_set_and_get(self, cache):
        """Test setting and getting cache entries"""
        # Add to cache
        cache.set(
            prompt="What is the carbon footprint of natural gas?",
            response="Natural gas has a carbon footprint of ~0.185 kg CO2/kWh",
            metadata={"model": "gpt-4"},
            agent_id="carbon_calc",
        )

        # Get exact match
        result = cache.get(
            prompt="What is the carbon footprint of natural gas?",
            metadata={"model": "gpt-4"},
            agent_id="carbon_calc",
        )

        assert result is not None
        response, similarity, entry = result
        assert "0.185" in response
        assert similarity > 0.99  # Exact match

    def test_semantic_similarity(self, cache):
        """Test semantic similarity matching"""
        # Add to cache
        cache.set(
            prompt="What is the carbon footprint of natural gas?",
            response="Natural gas has a carbon footprint of ~0.185 kg CO2/kWh",
            metadata={"model": "gpt-4"},
        )

        # Query with similar but different phrasing
        result = cache.get(
            prompt="What's the CO2 emissions from natural gas?",
            similarity_threshold=0.85,
        )

        assert result is not None
        response, similarity, entry = result
        assert "0.185" in response
        assert similarity > 0.85

    def test_cache_miss(self, cache):
        """Test cache miss"""
        # Add to cache
        cache.set(
            prompt="What is the carbon footprint of natural gas?",
            response="Natural gas emissions...",
            metadata={"model": "gpt-4"},
        )

        # Query completely different topic
        result = cache.get(
            prompt="What is the weather today?",
        )

        assert result is None

    def test_hit_rate_tracking(self, cache):
        """Test cache hit rate tracking"""
        # Add entry
        cache.set(
            prompt="Test query",
            response="Test response",
            metadata={},
        )

        # Cache miss
        cache.get(prompt="Different query")

        # Cache hit
        cache.get(prompt="Test query")
        cache.get(prompt="Test query")

        metrics = cache.get_metrics()
        assert metrics.total_requests == 3
        assert metrics.cache_hits == 2
        assert metrics.cache_misses == 1
        assert metrics.hit_rate == 2 / 3

    def test_metadata_filtering(self, cache):
        """Test filtering by metadata"""
        # Add entries with different metadata
        cache.set(
            prompt="Test",
            response="Response 1",
            metadata={"model": "gpt-4"},
        )

        cache.set(
            prompt="Test",
            response="Response 2",
            metadata={"model": "gpt-3.5"},
        )

        # Query with metadata filter
        result = cache.get(
            prompt="Test",
            metadata={"model": "gpt-4"},
        )

        assert result is not None
        response, _, _ = result
        assert response == "Response 1"

    def test_agent_filtering(self, cache):
        """Test filtering by agent ID"""
        # Add entries for different agents
        cache.set(
            prompt="Test",
            response="Agent 1 response",
            agent_id="agent1",
        )

        cache.set(
            prompt="Test",
            response="Agent 2 response",
            agent_id="agent2",
        )

        # Query with agent filter
        result = cache.get(
            prompt="Test",
            agent_id="agent1",
        )

        assert result is not None
        response, _, _ = result
        assert response == "Agent 1 response"


class TestCacheMetrics:
    """Test cache metrics"""

    def test_init(self):
        """Test initialization"""
        metrics = CacheMetrics()
        assert metrics.total_requests == 0
        assert metrics.cache_hits == 0
        assert metrics.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation"""
        metrics = CacheMetrics()
        metrics.total_requests = 10
        metrics.cache_hits = 3

        assert metrics.hit_rate == 0.3


class TestCachePerformance:
    """Performance benchmarks"""

    @pytest.fixture
    def cache(self):
        """Create cache for benchmarking"""
        return SemanticCache(use_redis=False)

    def test_embedding_performance(self):
        """Test embedding generation performance"""
        embedder = EmbeddingGenerator()

        text = "What is the carbon footprint of natural gas?"

        # Measure time
        start = time.time()
        for _ in range(100):
            embedder.encode(text)
        elapsed = time.time() - start

        avg_time = (elapsed / 100) * 1000  # Convert to ms
        print(f"\nAvg embedding time: {avg_time:.2f}ms")

        # Should be fast (<50ms per embedding)
        assert avg_time < 50

    def test_cache_lookup_performance(self, cache):
        """Test cache lookup performance"""
        # Add 100 entries
        for i in range(100):
            cache.set(
                prompt=f"Query {i}",
                response=f"Response {i}",
                metadata={},
            )

        # Measure lookup time
        start = time.time()
        for _ in range(100):
            cache.get(prompt="Query 50")
        elapsed = time.time() - start

        avg_time = (elapsed / 100) * 1000
        print(f"\nAvg lookup time: {avg_time:.2f}ms")

        # Should be fast (<50ms per lookup)
        assert avg_time < 50


class TestGlobalCache:
    """Test global cache singleton"""

    def test_singleton(self):
        """Test global cache is singleton"""
        cache1 = get_global_cache()
        cache2 = get_global_cache()

        assert cache1 is cache2


if __name__ == "__main__":
    """Run tests"""
    pytest.main([__file__, "-v", "-s"])
