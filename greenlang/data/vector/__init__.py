"""
GreenLang Vector Database Infrastructure (INFRA-005)

pgvector-based vector embedding storage and similarity search
for the GreenLang Climate Operating System.

Components:
    - EmbeddingService: Generate and store embeddings
    - SearchEngine: Similarity, filtered, and hybrid search
    - ChunkingService: Document chunking with multiple strategies
    - BatchProcessor: High-throughput bulk vector operations
    - IndexManager: HNSW/IVFFlat index lifecycle management
    - VectorDBConnection: Async PostgreSQL + pgvector connection pool
"""

from greenlang.data.vector.config import (
    EmbeddingConfig,
    SearchConfig,
    VectorDBConfig,
    IndexConfig,
    EnvironmentConfig,
)
from greenlang.data.vector.models import (
    EmbeddingRequest,
    EmbeddingResult,
    SearchRequest,
    SearchResult,
    SearchMatch,
    HybridSearchRequest,
    BatchInsertResult,
    CollectionInfo,
    JobStatus,
)

__all__ = [
    "EmbeddingConfig",
    "SearchConfig",
    "VectorDBConfig",
    "IndexConfig",
    "EnvironmentConfig",
    "EmbeddingRequest",
    "EmbeddingResult",
    "SearchRequest",
    "SearchResult",
    "SearchMatch",
    "HybridSearchRequest",
    "BatchInsertResult",
    "CollectionInfo",
    "JobStatus",
]
