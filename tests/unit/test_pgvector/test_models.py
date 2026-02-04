"""Tests for pgvector data models."""

import numpy as np
import pytest

from greenlang.data.vector.models import (
    BatchInsertResult,
    ChunkResult,
    CollectionInfo,
    EmbeddingRequest,
    EmbeddingResult,
    HybridSearchRequest,
    JobStatus,
    SearchMatch,
    SearchRequest,
    SearchResult,
    SourceType,
    VectorRecord,
)


class TestEmbeddingRequest:
    def test_basic_request(self):
        req = EmbeddingRequest(texts=["hello world"])
        assert req.texts == ["hello world"]
        assert req.namespace == "default"
        assert req.source_type == "document"

    def test_full_request(self):
        req = EmbeddingRequest(
            texts=["a", "b"],
            namespace="csrd",
            source_type="regulation",
            source_id="abc-123",
            metadata={"key": "value"},
        )
        assert req.namespace == "csrd"
        assert req.source_type == "regulation"
        assert req.metadata == {"key": "value"}


class TestEmbeddingResult:
    def test_result_count(self):
        emb = np.random.rand(5, 384).astype(np.float32)
        result = EmbeddingResult(
            embeddings=emb,
            model="all-MiniLM-L6-v2",
            dimensions=384,
            processing_time_ms=100,
        )
        assert result.count == 5
        assert result.dimensions == 384

    def test_explicit_count(self):
        emb = np.random.rand(3, 384).astype(np.float32)
        result = EmbeddingResult(
            embeddings=emb,
            model="test",
            dimensions=384,
            processing_time_ms=50,
            count=3,
        )
        assert result.count == 3


class TestSearchMatch:
    def test_basic_match(self):
        match = SearchMatch(
            id="abc",
            source_type="document",
            source_id="doc-1",
            chunk_index=0,
            content_preview="preview",
            metadata={},
            similarity=0.95,
        )
        assert match.similarity == 0.95
        assert match.rrf_score is None


class TestSearchResult:
    def test_top_match(self):
        matches = [
            SearchMatch(id="1", source_type="doc", source_id="1",
                       chunk_index=0, content_preview="a", metadata={}, similarity=0.9),
            SearchMatch(id="2", source_type="doc", source_id="2",
                       chunk_index=0, content_preview="b", metadata={}, similarity=0.8),
        ]
        result = SearchResult(
            matches=matches,
            query_text="test",
            total_results=2,
            latency_ms=15,
        )
        assert result.top_match.id == "1"

    def test_empty_result(self):
        result = SearchResult(
            matches=[],
            query_text="test",
            total_results=0,
            latency_ms=5,
        )
        assert result.top_match is None


class TestVectorRecord:
    def test_auto_id_and_hash(self):
        record = VectorRecord(
            source_type="document",
            source_id="doc-1",
            content="test content",
            embedding=np.zeros(384, dtype=np.float32),
        )
        assert record.id is not None
        assert len(record.content_hash) == 64  # SHA-256 hex

    def test_provided_id(self):
        record = VectorRecord(
            source_type="document",
            source_id="doc-1",
            content="test",
            embedding=np.zeros(384, dtype=np.float32),
            id="custom-id",
        )
        assert record.id == "custom-id"


class TestBatchInsertResult:
    def test_success_rate(self):
        result = BatchInsertResult(
            total_count=100,
            inserted_count=95,
            failed_count=5,
            duplicate_count=0,
            processing_time_ms=500,
        )
        assert result.success_rate == 0.95

    def test_zero_total(self):
        result = BatchInsertResult(
            total_count=0,
            inserted_count=0,
            failed_count=0,
            duplicate_count=0,
            processing_time_ms=0,
        )
        assert result.success_rate == 0.0


class TestJobStatus:
    def test_progress(self):
        job = JobStatus(
            id="j1",
            status="running",
            source_type="document",
            source_count=100,
            processed_count=50,
            failed_count=10,
            error_message=None,
            started_at=None,
            completed_at=None,
            created_at=None,
        )
        assert job.progress_pct == 60.0
        assert not job.is_complete

    def test_completed(self):
        job = JobStatus(
            id="j2",
            status="completed",
            source_type="document",
            source_count=100,
            processed_count=100,
            failed_count=0,
            error_message=None,
            started_at=None,
            completed_at=None,
            created_at=None,
        )
        assert job.is_complete


class TestChunkResult:
    def test_auto_counts(self):
        result = ChunkResult(
            chunks=["chunk1", "chunk2 longer text"],
            chunk_count=0,
            strategy="semantic",
            avg_chunk_size=0,
        )
        assert result.chunk_count == 2
        assert result.avg_chunk_size > 0


class TestSourceType:
    def test_enum_values(self):
        assert SourceType.DOCUMENT == "document"
        assert SourceType.REGULATION == "regulation"
        assert SourceType.EMISSION_FACTOR == "emission_factor"
