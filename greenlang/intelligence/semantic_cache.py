"""
Semantic Cache for LLM Responses

Implements semantic similarity-based caching to reduce LLM API costs by:
- Using vector embeddings (sentence-transformers) for semantic similarity
- FAISS for fast similarity search
- Redis for cache storage with TTL
- Cosine similarity matching (threshold > 0.95)
- Cache hit rate tracking per agent

Architecture:
    User Query -> Embed -> FAISS Search -> Similarity Check
                                |
                         > 0.95 threshold?
                            /        \
                          Yes         No
                           |           |
                    Return Cache    Call LLM
                                      |
                                 Cache Response

Performance Targets:
- Cache hit rate: >30%
- Lookup latency: <50ms
- Cost savings: >40%
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with: pip install faiss-cpu")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available. Install with: pip install redis")


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """
    Cache entry with embedding, prompt, response, and metadata

    Attributes:
        embedding: Vector embedding of the prompt (384-dimensional)
        prompt: Original prompt text
        response: Cached LLM response
        metadata: Additional metadata (model, temperature, etc.)
        timestamp: When entry was created
        hit_count: Number of times this entry was returned
        last_accessed: Last access timestamp
        agent_id: ID of the agent that created this entry
        ttl_seconds: Time-to-live in seconds
    """
    embedding: np.ndarray
    prompt: str
    response: str
    metadata: Dict[str, Any]
    timestamp: datetime
    hit_count: int = 0
    last_accessed: Optional[datetime] = None
    agent_id: Optional[str] = None
    ttl_seconds: int = 86400  # 24 hours default

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "embedding": self.embedding.tolist(),
            "prompt": self.prompt,
            "response": self.response,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "hit_count": self.hit_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "agent_id": self.agent_id,
            "ttl_seconds": self.ttl_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CacheEntry:
        """Create from dictionary"""
        return cls(
            embedding=np.array(data["embedding"], dtype=np.float32),
            prompt=data["prompt"],
            response=data["response"],
            metadata=data["metadata"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            hit_count=data.get("hit_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
            agent_id=data.get("agent_id"),
            ttl_seconds=data.get("ttl_seconds", 86400),
        )

    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL"""
        age = datetime.now() - self.timestamp
        return age.total_seconds() > self.ttl_seconds


@dataclass
class CacheMetrics:
    """
    Metrics for cache performance tracking

    Attributes:
        total_requests: Total number of cache lookups
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
        total_entries: Total number of cached entries
        avg_similarity: Average similarity score for hits
        cost_savings_usd: Estimated cost savings
        hit_rate_by_agent: Hit rate breakdown by agent
    """
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_entries: int = 0
    avg_similarity: float = 0.0
    cost_savings_usd: float = 0.0
    hit_rate_by_agent: Dict[str, float] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_entries": self.total_entries,
            "hit_rate": self.hit_rate,
            "avg_similarity": self.avg_similarity,
            "cost_savings_usd": self.cost_savings_usd,
            "hit_rate_by_agent": self.hit_rate_by_agent,
        }


class EmbeddingGenerator:
    """
    Generate embeddings using sentence-transformers

    Uses all-MiniLM-L6-v2 model:
    - 384-dimensional embeddings
    - Fast inference (20ms per sentence)
    - Good semantic understanding
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[str] = None):
        """
        Initialize embedding generator

        Args:
            model_name: Name of sentence-transformers model
            cache_dir: Directory to cache model files
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "greenlang" / "embeddings")

        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, cache_folder=self.cache_dir)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(f"Embedding model loaded. Dimension: {self.embedding_dim}")

    def encode(self, text: str) -> np.ndarray:
        """
        Generate embedding for text

        Args:
            text: Input text to embed

        Returns:
            Embedding vector (384-dimensional)
        """
        # Normalize text
        text = text.strip()

        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

        return embedding.astype(np.float32)

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for batch of texts

        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding

        Returns:
            Array of embeddings
        """
        # Normalize texts
        texts = [t.strip() for t in texts]

        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.astype(np.float32)


class FAISSIndex:
    """
    FAISS index for fast similarity search

    Uses IndexFlatIP (Inner Product) for cosine similarity:
    - Exact search (not approximate)
    - Fast for moderate cache sizes (<100k entries)
    - Returns k nearest neighbors
    """

    def __init__(self, dimension: int = 384):
        """
        Initialize FAISS index

        Args:
            dimension: Embedding dimension
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")

        self.dimension = dimension

        # Use IndexFlatIP for cosine similarity (assumes normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)

        # Mapping from FAISS index to cache keys
        self.index_to_key: List[str] = []

        logger.info(f"FAISS index initialized. Dimension: {dimension}")

    def add(self, key: str, embedding: np.ndarray):
        """
        Add embedding to index

        Args:
            key: Cache key
            embedding: Embedding vector
        """
        # Ensure embedding is 2D array
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Add to FAISS index
        self.index.add(embedding)

        # Store key mapping
        self.index_to_key.append(key)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for k nearest neighbors

        Args:
            query_embedding: Query embedding
            k: Number of neighbors to return

        Returns:
            List of (key, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            return []

        # Ensure query is 2D array
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search FAISS index
        k = min(k, self.index.ntotal)
        similarities, indices = self.index.search(query_embedding, k)

        # Convert to list of (key, similarity) tuples
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.index_to_key):
                key = self.index_to_key[idx]
                similarity = float(similarities[0][i])
                results.append((key, similarity))

        return results

    def save(self, filepath: str):
        """Save index to disk"""
        faiss.write_index(self.index, filepath)

        # Save key mapping
        mapping_path = filepath + ".mapping"
        with open(mapping_path, "wb") as f:
            pickle.dump(self.index_to_key, f)

        logger.info(f"FAISS index saved to {filepath}")

    def load(self, filepath: str):
        """Load index from disk"""
        self.index = faiss.read_index(filepath)

        # Load key mapping
        mapping_path = filepath + ".mapping"
        with open(mapping_path, "rb") as f:
            self.index_to_key = pickle.load(f)

        logger.info(f"FAISS index loaded from {filepath}")

    @property
    def size(self) -> int:
        """Get number of vectors in index"""
        return self.index.ntotal


class RedisCache:
    """
    Redis-based cache storage with TTL

    Uses Redis for:
    - Fast key-value lookup
    - Automatic TTL expiration
    - Persistence across restarts
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "greenlang:semantic_cache:",
    ):
        """
        Initialize Redis cache

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            prefix: Key prefix for namespacing
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install with: pip install redis")

        self.prefix = prefix

        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False,  # We'll handle encoding
            )
            # Test connection
            self.client.ping()
            logger.info(f"Redis cache connected: {host}:{port}")
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory fallback.")
            self.client = None

    def _make_key(self, key: str) -> str:
        """Create prefixed Redis key"""
        return f"{self.prefix}{key}"

    def set(self, key: str, entry: CacheEntry):
        """
        Set cache entry with TTL

        Args:
            key: Cache key
            entry: Cache entry to store
        """
        if self.client is None:
            return

        redis_key = self._make_key(key)

        # Serialize entry
        data = json.dumps(entry.to_dict())

        # Store with TTL
        self.client.setex(redis_key, entry.ttl_seconds, data)

    def get(self, key: str) -> Optional[CacheEntry]:
        """
        Get cache entry

        Args:
            key: Cache key

        Returns:
            Cache entry or None if not found
        """
        if self.client is None:
            return None

        redis_key = self._make_key(key)

        # Get from Redis
        data = self.client.get(redis_key)
        if data is None:
            return None

        # Deserialize
        entry_dict = json.loads(data)
        entry = CacheEntry.from_dict(entry_dict)

        # Check expiration
        if entry.is_expired():
            self.delete(key)
            return None

        return entry

    def delete(self, key: str):
        """Delete cache entry"""
        if self.client is None:
            return

        redis_key = self._make_key(key)
        self.client.delete(redis_key)

    def clear(self):
        """Clear all cache entries"""
        if self.client is None:
            return

        # Delete all keys with prefix
        pattern = f"{self.prefix}*"
        keys = self.client.keys(pattern)
        if keys:
            self.client.delete(*keys)

    def keys(self) -> List[str]:
        """Get all cache keys"""
        if self.client is None:
            return []

        pattern = f"{self.prefix}*"
        redis_keys = self.client.keys(pattern)

        # Remove prefix
        return [k.decode().replace(self.prefix, "") for k in redis_keys]


class InMemoryCache:
    """
    Fallback in-memory cache when Redis is not available

    Uses simple dictionary with manual TTL checking.
    Not recommended for production (no persistence, limited scalability).
    """

    def __init__(self):
        """Initialize in-memory cache"""
        self.cache: Dict[str, CacheEntry] = {}
        logger.info("Using in-memory cache (Redis not available)")

    def set(self, key: str, entry: CacheEntry):
        """Set cache entry"""
        self.cache[key] = entry

    def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry"""
        entry = self.cache.get(key)
        if entry is None:
            return None

        # Check expiration
        if entry.is_expired():
            self.delete(key)
            return None

        return entry

    def delete(self, key: str):
        """Delete cache entry"""
        self.cache.pop(key, None)

    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()

    def keys(self) -> List[str]:
        """Get all cache keys"""
        return list(self.cache.keys())


class SemanticCache:
    """
    Semantic cache for LLM responses

    Main interface for semantic caching with:
    - Vector embeddings for semantic similarity
    - FAISS for fast similarity search
    - Redis/in-memory storage
    - Automatic cache warming
    - Hit rate tracking

    Example:
        >>> cache = SemanticCache()
        >>>
        >>> # Add response to cache
        >>> cache.set(
        ...     prompt="What is the carbon footprint of natural gas?",
        ...     response="Natural gas has a carbon footprint of...",
        ...     metadata={"model": "gpt-4", "temperature": 0.0},
        ...     agent_id="carbon_calc",
        ... )
        >>>
        >>> # Lookup similar prompt
        >>> result = cache.get(
        ...     prompt="What's the CO2 emissions from natural gas?",
        ...     similarity_threshold=0.95,
        ... )
        >>> if result:
        ...     print(f"Cache hit! Similarity: {result[1]:.3f}")
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.95,
        use_redis: bool = True,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        default_ttl: int = 86400,
        index_path: Optional[str] = None,
    ):
        """
        Initialize semantic cache

        Args:
            embedding_model: Sentence-transformers model name
            similarity_threshold: Minimum similarity for cache hit (0-1)
            use_redis: Whether to use Redis (fallback to in-memory if False)
            redis_host: Redis host
            redis_port: Redis port
            default_ttl: Default TTL in seconds
            index_path: Path to save/load FAISS index
        """
        self.similarity_threshold = similarity_threshold
        self.default_ttl = default_ttl
        self.index_path = index_path

        # Initialize embedding generator
        self.embedder = EmbeddingGenerator(model_name=embedding_model)

        # Initialize FAISS index
        self.faiss_index = FAISSIndex(dimension=self.embedder.embedding_dim)

        # Initialize cache storage
        if use_redis and REDIS_AVAILABLE:
            try:
                self.cache_storage = RedisCache(host=redis_host, port=redis_port)
            except Exception as e:
                logger.warning(f"Redis initialization failed: {e}. Using in-memory cache.")
                self.cache_storage = InMemoryCache()
        else:
            self.cache_storage = InMemoryCache()

        # Metrics
        self.metrics = CacheMetrics()

        # Load index if exists
        if index_path and Path(index_path).exists():
            self.load_index(index_path)

        logger.info("SemanticCache initialized")

    def _generate_key(self, prompt: str, metadata: Dict[str, Any]) -> str:
        """
        Generate unique cache key from prompt and metadata

        Args:
            prompt: Prompt text
            metadata: Metadata (model, temperature, etc.)

        Returns:
            Cache key (hash)
        """
        # Create deterministic string from prompt + metadata
        data = {
            "prompt": prompt,
            "metadata": metadata,
        }
        data_str = json.dumps(data, sort_keys=True)

        # Hash to create key
        key = hashlib.sha256(data_str.encode()).hexdigest()

        return key

    def set(
        self,
        prompt: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ):
        """
        Add response to cache

        Args:
            prompt: Prompt text
            response: LLM response
            metadata: Metadata (model, temperature, etc.)
            agent_id: ID of agent that created this response
            ttl_seconds: TTL in seconds (default: 24 hours)
        """
        metadata = metadata or {}
        ttl_seconds = ttl_seconds or self.default_ttl

        # Generate embedding
        embedding = self.embedder.encode(prompt)

        # Create cache entry
        entry = CacheEntry(
            embedding=embedding,
            prompt=prompt,
            response=response,
            metadata=metadata,
            timestamp=datetime.now(),
            agent_id=agent_id,
            ttl_seconds=ttl_seconds,
        )

        # Generate key
        key = self._generate_key(prompt, metadata)

        # Store in cache
        self.cache_storage.set(key, entry)

        # Add to FAISS index
        self.faiss_index.add(key, embedding)

        # Update metrics
        self.metrics.total_entries = self.faiss_index.size

        logger.debug(f"Cached response for prompt: {prompt[:50]}...")

    def get(
        self,
        prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        agent_id: Optional[str] = None,
    ) -> Optional[Tuple[str, float, CacheEntry]]:
        """
        Lookup cached response by semantic similarity

        Args:
            prompt: Query prompt
            metadata: Metadata to match (optional)
            similarity_threshold: Minimum similarity (default: 0.95)
            agent_id: Filter by agent ID (optional)

        Returns:
            Tuple of (response, similarity_score, entry) or None if not found
        """
        self.metrics.total_requests += 1

        similarity_threshold = similarity_threshold or self.similarity_threshold

        # Generate query embedding
        query_embedding = self.embedder.encode(prompt)

        # Search FAISS index
        start_time = time.time()
        results = self.faiss_index.search(query_embedding, k=5)
        search_time = time.time() - start_time

        logger.debug(f"FAISS search completed in {search_time*1000:.2f}ms")

        # Check results
        for key, similarity in results:
            # Check similarity threshold
            if similarity < similarity_threshold:
                continue

            # Get cache entry
            entry = self.cache_storage.get(key)
            if entry is None:
                continue

            # Filter by agent ID if specified
            if agent_id and entry.agent_id != agent_id:
                continue

            # Filter by metadata if specified
            if metadata:
                if not all(entry.metadata.get(k) == v for k, v in metadata.items()):
                    continue

            # Cache hit!
            entry.hit_count += 1
            entry.last_accessed = datetime.now()
            self.cache_storage.set(key, entry)

            # Update metrics
            self.metrics.cache_hits += 1
            self._update_hit_rate(agent_id or "unknown")

            # Estimate cost savings (assume $0.002 per request saved)
            self.metrics.cost_savings_usd += 0.002

            logger.info(f"Cache hit! Similarity: {similarity:.3f}, Prompt: {prompt[:50]}...")

            return (entry.response, similarity, entry)

        # Cache miss
        self.metrics.cache_misses += 1
        logger.debug(f"Cache miss for prompt: {prompt[:50]}...")

        return None

    def _update_hit_rate(self, agent_id: str):
        """Update hit rate by agent"""
        if agent_id not in self.metrics.hit_rate_by_agent:
            self.metrics.hit_rate_by_agent[agent_id] = 0.0

        # Simple exponential moving average
        alpha = 0.1
        current_rate = self.metrics.hit_rate_by_agent[agent_id]
        self.metrics.hit_rate_by_agent[agent_id] = alpha * 1.0 + (1 - alpha) * current_rate

    def clear(self):
        """Clear all cache entries"""
        self.cache_storage.clear()
        self.faiss_index = FAISSIndex(dimension=self.embedder.embedding_dim)
        self.metrics = CacheMetrics()
        logger.info("Cache cleared")

    def save_index(self, filepath: Optional[str] = None):
        """Save FAISS index to disk"""
        filepath = filepath or self.index_path
        if filepath is None:
            raise ValueError("No index path specified")

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        self.faiss_index.save(filepath)

        logger.info(f"Index saved to {filepath}")

    def load_index(self, filepath: Optional[str] = None):
        """Load FAISS index from disk"""
        filepath = filepath or self.index_path
        if filepath is None:
            raise ValueError("No index path specified")

        # Load FAISS index
        self.faiss_index.load(filepath)

        # Update metrics
        self.metrics.total_entries = self.faiss_index.size

        logger.info(f"Index loaded from {filepath}")

    def get_metrics(self) -> CacheMetrics:
        """Get current cache metrics"""
        self.metrics.total_entries = self.faiss_index.size
        return self.metrics

    def get_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics"""
        metrics = self.get_metrics()

        return {
            "cache_size": self.faiss_index.size,
            "hit_rate": metrics.hit_rate,
            "total_requests": metrics.total_requests,
            "cache_hits": metrics.cache_hits,
            "cache_misses": metrics.cache_misses,
            "cost_savings_usd": metrics.cost_savings_usd,
            "hit_rate_by_agent": metrics.hit_rate_by_agent,
            "embedding_model": self.embedder.model_name,
            "similarity_threshold": self.similarity_threshold,
        }


# Singleton instance
_global_cache: Optional[SemanticCache] = None


def get_global_cache() -> SemanticCache:
    """
    Get global semantic cache instance

    Returns:
        Global SemanticCache instance
    """
    global _global_cache

    if _global_cache is None:
        _global_cache = SemanticCache()

    return _global_cache


if __name__ == "__main__":
    """
    Demo and testing
    """
    print("=" * 80)
    print("GreenLang Semantic Cache Demo")
    print("=" * 80)

    # Initialize cache
    cache = SemanticCache(similarity_threshold=0.90)

    # Add some entries
    print("\n1. Adding cache entries...")
    cache.set(
        prompt="What is the carbon footprint of natural gas?",
        response="Natural gas has a carbon footprint of approximately 0.185 kg CO2/kWh.",
        metadata={"model": "gpt-4", "temperature": 0.0},
        agent_id="carbon_calc",
    )

    cache.set(
        prompt="Calculate emissions for 1000 kWh electricity",
        response="1000 kWh of electricity produces approximately 400 kg CO2.",
        metadata={"model": "gpt-4", "temperature": 0.0},
        agent_id="emission_calc",
    )

    # Test similar queries
    print("\n2. Testing semantic similarity...")

    test_queries = [
        "What's the CO2 emissions from natural gas?",  # Similar to first
        "Tell me about natural gas carbon impact",     # Similar to first
        "How much CO2 for 1000 kWh of power?",        # Similar to second
        "What's the weather today?",                   # Different
    ]

    for query in test_queries:
        result = cache.get(query, agent_id="carbon_calc")
        if result:
            response, similarity, entry = result
            print(f"\n   Query: {query}")
            print(f"   Similarity: {similarity:.3f}")
            print(f"   Response: {response[:60]}...")
        else:
            print(f"\n   Query: {query}")
            print(f"   Result: Cache miss")

    # Show stats
    print("\n3. Cache statistics:")
    stats = cache.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 80)
