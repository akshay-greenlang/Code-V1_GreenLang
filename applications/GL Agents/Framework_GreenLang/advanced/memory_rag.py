"""
GreenLang Framework - RAG-Enabled Memory Management
Vector Store and Semantic Memory for AI Agents

Based on:
- LangGraph MongoDB Long-Term Memory
- Pinecone/Milvus Vector Store Patterns
- AWS AgentCore Memory Design
- Hindsight Agentic Memory (91% accuracy)

This module provides semantic memory capabilities for GreenLang agents
with multi-tier storage, memory consolidation, and retrieval-augmented
generation support.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union
import hashlib
import json
import logging
import uuid
from collections import defaultdict
import heapq


logger = logging.getLogger(__name__)

T = TypeVar('T')


class MemoryType(Enum):
    """Types of memory in the system."""
    EPISODIC = "episodic"        # Specific events/interactions
    SEMANTIC = "semantic"        # Facts and knowledge
    PROCEDURAL = "procedural"    # How to do things
    WORKING = "working"          # Short-term active memory


class MemoryStatus(Enum):
    """Status of a memory entry."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    CONSOLIDATED = "consolidated"
    INVALID = "invalid"  # For contradicted memories


@dataclass
class MemoryEntry:
    """Single memory entry in the system."""
    memory_id: str
    content: Any
    memory_type: MemoryType
    status: MemoryStatus = MemoryStatus.ACTIVE
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance_score: float = 0.5
    access_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    source_agent_id: str = ""
    provenance_hash: str = ""

    def __post_init__(self):
        if not self.memory_id:
            self.memory_id = str(uuid.uuid4())
        if not self.provenance_hash:
            data = f"{self.memory_id}:{json.dumps(self.content, sort_keys=True, default=str)}"
            self.provenance_hash = hashlib.sha256(data.encode()).hexdigest()

    def update_access(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)

    def is_expired(self) -> bool:
        """Check if memory has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "status": self.status.value,
            "importance_score": self.importance_score,
            "access_count": self.access_count,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "source_agent_id": self.source_agent_id,
            "provenance_hash": self.provenance_hash
        }


@dataclass
class RetrievalResult:
    """Result from memory retrieval."""
    memory: MemoryEntry
    similarity_score: float
    retrieval_method: str  # "vector", "keyword", "hybrid"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory": self.memory.to_dict(),
            "similarity_score": self.similarity_score,
            "retrieval_method": self.retrieval_method
        }


class EmbeddingProvider(ABC):
    """Abstract embedding provider for vector operations."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass


class SimpleEmbedding(EmbeddingProvider):
    """
    Simple TF-IDF-like embedding for demonstration.

    In production, replace with:
    - OpenAI text-embedding-ada-002/3
    - Sentence Transformers
    - Cohere embeddings
    """

    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self._vocabulary: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}

    def embed(self, text: str) -> List[float]:
        """Generate simple hash-based embedding."""
        # Simple deterministic embedding for industrial use
        text_lower = text.lower()
        words = text_lower.split()

        # Create sparse vector
        vector = [0.0] * self._dimension

        for i, word in enumerate(words):
            # Hash word to dimension
            word_hash = hash(word) % self._dimension
            # Add weighted value
            weight = 1.0 / (i + 1)  # Position-based weighting
            vector[word_hash] += weight

        # Normalize
        magnitude = sum(v ** 2 for v in vector) ** 0.5
        if magnitude > 0:
            vector = [v / magnitude for v in vector]

        return vector

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(text) for text in texts]

    @property
    def dimension(self) -> int:
        return self._dimension


class VectorIndex(ABC):
    """Abstract vector index for similarity search."""

    @abstractmethod
    def add(self, memory_id: str, embedding: List[float]) -> None:
        """Add embedding to index."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        k: int = 10
    ) -> List[tuple[str, float]]:
        """Search for similar embeddings."""
        pass

    @abstractmethod
    def remove(self, memory_id: str) -> bool:
        """Remove embedding from index."""
        pass


class SimpleVectorIndex(VectorIndex):
    """
    Simple in-memory vector index using brute-force search.

    In production, replace with:
    - FAISS
    - Pinecone
    - Milvus
    - Weaviate
    """

    def __init__(self):
        self._vectors: Dict[str, List[float]] = {}

    def add(self, memory_id: str, embedding: List[float]) -> None:
        self._vectors[memory_id] = embedding

    def search(
        self,
        query_embedding: List[float],
        k: int = 10
    ) -> List[tuple[str, float]]:
        """Brute-force cosine similarity search."""
        if not self._vectors:
            return []

        def cosine_similarity(a: List[float], b: List[float]) -> float:
            dot_product = sum(x * y for x, y in zip(a, b))
            magnitude_a = sum(x ** 2 for x in a) ** 0.5
            magnitude_b = sum(x ** 2 for x in b) ** 0.5
            if magnitude_a * magnitude_b == 0:
                return 0.0
            return dot_product / (magnitude_a * magnitude_b)

        scores = []
        for memory_id, embedding in self._vectors.items():
            score = cosine_similarity(query_embedding, embedding)
            scores.append((memory_id, score))

        # Return top k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def remove(self, memory_id: str) -> bool:
        if memory_id in self._vectors:
            del self._vectors[memory_id]
            return True
        return False


class MemoryStore:
    """
    Multi-tier memory store for GreenLang agents.

    Implements the L0-L1-L2 memory architecture:
    - L0 (Raw): Immediate interactions and observations
    - L1 (Working): Active context for current tasks
    - L2 (Long-term): Consolidated knowledge and patterns
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_index: Optional[VectorIndex] = None,
        max_working_memory: int = 100,
        consolidation_threshold: int = 50
    ):
        self._embedding_provider = embedding_provider or SimpleEmbedding()
        self._vector_index = vector_index or SimpleVectorIndex()
        self._max_working_memory = max_working_memory
        self._consolidation_threshold = consolidation_threshold

        # Memory tiers
        self._working_memory: Dict[str, MemoryEntry] = {}  # L1
        self._long_term_memory: Dict[str, MemoryEntry] = {}  # L2

        # Indexes
        self._type_index: Dict[MemoryType, set[str]] = defaultdict(set)
        self._agent_index: Dict[str, set[str]] = defaultdict(set)

    def add(
        self,
        content: Any,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: float = 0.5,
        source_agent_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None
    ) -> MemoryEntry:
        """Add a new memory entry."""
        # Create content string for embedding
        if isinstance(content, dict):
            content_str = json.dumps(content, sort_keys=True, default=str)
        else:
            content_str = str(content)

        # Generate embedding
        embedding = self._embedding_provider.embed(content_str)

        # Calculate expiration
        expires_at = None
        if ttl_seconds:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)

        # Create memory entry
        memory = MemoryEntry(
            memory_id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            embedding=embedding,
            importance_score=importance,
            source_agent_id=source_agent_id,
            metadata=metadata or {},
            expires_at=expires_at
        )

        # Add to working memory
        self._working_memory[memory.memory_id] = memory

        # Add to vector index
        self._vector_index.add(memory.memory_id, embedding)

        # Update indexes
        self._type_index[memory_type].add(memory.memory_id)
        if source_agent_id:
            self._agent_index[source_agent_id].add(memory.memory_id)

        # Check for consolidation
        if len(self._working_memory) > self._consolidation_threshold:
            self._consolidate()

        logger.debug(f"Added memory: {memory.memory_id}")
        return memory

    def retrieve(
        self,
        query: str,
        k: int = 5,
        memory_types: Optional[List[MemoryType]] = None,
        min_importance: float = 0.0,
        source_agent_id: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant memories using hybrid search.

        Combines vector similarity with metadata filtering.
        """
        # Generate query embedding
        query_embedding = self._embedding_provider.embed(query)

        # Vector search
        vector_results = self._vector_index.search(query_embedding, k * 2)

        # Filter and rank
        results = []
        for memory_id, similarity in vector_results:
            # Get memory from appropriate tier
            memory = self._working_memory.get(memory_id) or self._long_term_memory.get(memory_id)
            if not memory:
                continue

            # Apply filters
            if memory.status != MemoryStatus.ACTIVE:
                continue
            if memory.is_expired():
                continue
            if memory.importance_score < min_importance:
                continue
            if memory_types and memory.memory_type not in memory_types:
                continue
            if source_agent_id and memory.source_agent_id != source_agent_id:
                continue

            # Update access
            memory.update_access()

            results.append(RetrievalResult(
                memory=memory,
                similarity_score=similarity,
                retrieval_method="vector"
            ))

        # Sort by combined score (similarity + importance)
        results.sort(
            key=lambda r: r.similarity_score * 0.7 + r.memory.importance_score * 0.3,
            reverse=True
        )

        return results[:k]

    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory by ID."""
        memory = self._working_memory.get(memory_id) or self._long_term_memory.get(memory_id)
        if memory:
            memory.update_access()
        return memory

    def update(self, memory_id: str, content: Any, importance: Optional[float] = None) -> bool:
        """Update an existing memory."""
        memory = self.get(memory_id)
        if not memory:
            return False

        memory.content = content
        if importance is not None:
            memory.importance_score = importance

        # Update embedding
        if isinstance(content, dict):
            content_str = json.dumps(content, sort_keys=True, default=str)
        else:
            content_str = str(content)
        memory.embedding = self._embedding_provider.embed(content_str)
        memory.provenance_hash = hashlib.sha256(
            f"{memory.memory_id}:{content_str}".encode()
        ).hexdigest()

        # Update vector index
        self._vector_index.remove(memory_id)
        self._vector_index.add(memory_id, memory.embedding)

        logger.debug(f"Updated memory: {memory_id}")
        return True

    def invalidate(self, memory_id: str, reason: str = "") -> bool:
        """Mark a memory as invalid (for handling contradictions)."""
        memory = self.get(memory_id)
        if not memory:
            return False

        memory.status = MemoryStatus.INVALID
        memory.metadata["invalidation_reason"] = reason
        memory.metadata["invalidated_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(f"Invalidated memory: {memory_id}, reason: {reason}")
        return True

    def _consolidate(self) -> int:
        """
        Consolidate working memory into long-term memory.

        Uses importance and access patterns to determine what to keep.
        """
        if len(self._working_memory) <= self._max_working_memory:
            return 0

        # Calculate retention scores
        def retention_score(memory: MemoryEntry) -> float:
            recency = (datetime.now(timezone.utc) - memory.last_accessed).total_seconds()
            recency_factor = 1.0 / (1.0 + recency / 3600)  # Decay over hours
            access_factor = min(memory.access_count / 10, 1.0)
            return (
                memory.importance_score * 0.4 +
                recency_factor * 0.3 +
                access_factor * 0.3
            )

        # Sort by retention score
        scored_memories = [
            (retention_score(m), memory_id, m)
            for memory_id, m in self._working_memory.items()
        ]
        scored_memories.sort(key=lambda x: x[0])

        # Move lowest-scoring memories to long-term
        to_move = len(self._working_memory) - self._max_working_memory
        moved = 0

        for score, memory_id, memory in scored_memories[:to_move]:
            memory.status = MemoryStatus.CONSOLIDATED
            self._long_term_memory[memory_id] = memory
            del self._working_memory[memory_id]
            moved += 1

        logger.info(f"Consolidated {moved} memories to long-term storage")
        return moved

    def cleanup_expired(self) -> int:
        """Remove expired memories."""
        expired = []
        for memory_id, memory in list(self._working_memory.items()):
            if memory.is_expired():
                expired.append(memory_id)

        for memory_id in expired:
            del self._working_memory[memory_id]
            self._vector_index.remove(memory_id)

        logger.info(f"Cleaned up {len(expired)} expired memories")
        return len(expired)

    def get_context_window(
        self,
        query: str,
        max_tokens: int = 4000,
        include_types: Optional[List[MemoryType]] = None
    ) -> str:
        """
        Get a formatted context window for RAG.

        Returns relevant memories formatted for LLM context.
        """
        results = self.retrieve(
            query=query,
            k=20,
            memory_types=include_types
        )

        context_parts = []
        estimated_tokens = 0

        for result in results:
            # Estimate tokens (rough: 4 chars = 1 token)
            if isinstance(result.memory.content, dict):
                content_str = json.dumps(result.memory.content, indent=2)
            else:
                content_str = str(result.memory.content)

            tokens = len(content_str) // 4

            if estimated_tokens + tokens > max_tokens:
                break

            context_parts.append(f"[{result.memory.memory_type.value}] {content_str}")
            estimated_tokens += tokens

        return "\n---\n".join(context_parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        return {
            "working_memory_count": len(self._working_memory),
            "long_term_memory_count": len(self._long_term_memory),
            "total_memories": len(self._working_memory) + len(self._long_term_memory),
            "type_distribution": {
                t.value: len(ids) for t, ids in self._type_index.items()
            },
            "agent_count": len(self._agent_index)
        }


# ============================================================================
# GLOBAL MEMORY INSTANCE
# ============================================================================

GREENLANG_MEMORY = MemoryStore()


# ============================================================================
# MEMORY DECORATORS FOR EASY INTEGRATION
# ============================================================================

def remember(
    memory_type: MemoryType = MemoryType.EPISODIC,
    importance: float = 0.5
):
    """
    Decorator to automatically store function results in memory.

    Usage:
        @remember(memory_type=MemoryType.SEMANTIC, importance=0.8)
        def calculate_efficiency(fuel_flow, steam_flow):
            ...
    """
    def decorator(func):
        from functools import wraps

        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            # Store result in memory
            content = {
                "function": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs),
                "result": result
            }
            GREENLANG_MEMORY.add(
                content=content,
                memory_type=memory_type,
                importance=importance
            )

            return result
        return wrapper
    return decorator


def with_context(query_template: str, max_tokens: int = 2000):
    """
    Decorator to inject relevant context from memory.

    Usage:
        @with_context("boiler efficiency {fuel_type}")
        def optimize_combustion(fuel_type, context=None):
            ...
    """
    def decorator(func):
        from functools import wraps

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Build query from template and args
            query = query_template.format(**kwargs)

            # Retrieve context
            context = GREENLANG_MEMORY.get_context_window(query, max_tokens)
            kwargs['context'] = context

            return func(*args, **kwargs)
        return wrapper
    return decorator
