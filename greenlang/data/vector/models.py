"""
Data models for pgvector infrastructure.

Defines request/response types for embedding operations,
search queries, batch processing, and job management.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class SourceType(str, Enum):
    DOCUMENT = "document"
    REGULATION = "regulation"
    REPORT = "report"
    POLICY = "policy"
    EMISSION_FACTOR = "emission_factor"
    BENCHMARK = "benchmark"


class ChunkStrategy(str, Enum):
    SEMANTIC = "semantic"
    FIXED = "fixed"
    SLIDING_WINDOW = "sliding_window"
    HIERARCHICAL = "hierarchical"


@dataclass
class EmbeddingRequest:
    texts: List[str]
    namespace: str = "default"
    source_type: str = "document"
    source_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    collection_name: Optional[str] = None
    model_name: Optional[str] = None


@dataclass
class EmbeddingResult:
    embeddings: np.ndarray  # Shape: (n_texts, dimensions)
    model: str
    dimensions: int
    processing_time_ms: int
    count: int = 0

    def __post_init__(self):
        if self.count == 0:
            self.count = len(self.embeddings)


@dataclass
class SearchMatch:
    id: str
    source_type: str
    source_id: str
    chunk_index: int
    content_preview: Optional[str]
    metadata: Dict[str, Any]
    similarity: float
    vector_rank: Optional[int] = None
    text_rank: Optional[int] = None
    rrf_score: Optional[float] = None


@dataclass
class SearchRequest:
    query: str
    namespace: str = "default"
    top_k: int = 10
    threshold: float = 0.7
    source_type: Optional[str] = None
    metadata_filter: Optional[Dict[str, Any]] = None
    ef_search: Optional[int] = None


@dataclass
class SearchResult:
    matches: List[SearchMatch]
    query_text: str
    total_results: int
    latency_ms: int
    search_type: str = "similarity"
    namespace: str = "default"

    @property
    def top_match(self) -> Optional[SearchMatch]:
        return self.matches[0] if self.matches else None


@dataclass
class HybridSearchRequest:
    query: str
    namespace: str = "default"
    top_k: int = 10
    rrf_k: int = 60
    vector_weight: float = 0.7
    text_weight: float = 0.3


@dataclass
class VectorRecord:
    source_type: str
    source_id: str
    content: str
    embedding: np.ndarray
    namespace: str = "default"
    chunk_index: int = 0
    content_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding_model: str = "all-MiniLM-L6-v2"
    collection_id: Optional[str] = None
    id: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.content_hash:
            import hashlib
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()


@dataclass
class BatchInsertResult:
    total_count: int
    inserted_count: int
    failed_count: int
    duplicate_count: int
    processing_time_ms: int
    job_id: Optional[str] = None
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.inserted_count / self.total_count


@dataclass
class CollectionInfo:
    id: str
    name: str
    namespace: str
    embedding_model: str
    dimensions: int
    distance_metric: str
    vector_count: int
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass
class JobStatus:
    id: str
    status: str
    source_type: str
    source_count: int
    processed_count: int
    failed_count: int
    error_message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime

    @property
    def progress_pct(self) -> float:
        if self.source_count == 0:
            return 0.0
        return (self.processed_count + self.failed_count) / self.source_count * 100

    @property
    def is_complete(self) -> bool:
        return self.status in ("completed", "failed", "cancelled")


@dataclass
class ChunkResult:
    chunks: List[str]
    chunk_count: int
    strategy: str
    avg_chunk_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.chunk_count = len(self.chunks)
        if self.chunks:
            self.avg_chunk_size = sum(len(c) for c in self.chunks) // len(self.chunks)
