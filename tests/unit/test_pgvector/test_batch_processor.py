"""Tests for pgvector batch processor."""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from greenlang.data.vector.batch_processor import BatchProcessor
from greenlang.data.vector.models import BatchInsertResult, JobStatus, VectorRecord


class TestBatchProcessor:
    def setup_method(self):
        self.db = MagicMock()
        self.processor = BatchProcessor(db=self.db, batch_size=100)

    def test_init(self):
        assert self.processor.batch_size == 100
        assert self.processor.db is self.db

    def test_default_batch_size(self):
        proc = BatchProcessor(db=self.db)
        assert proc.batch_size == 1000


class TestVectorRecord:
    def test_auto_hash(self):
        record = VectorRecord(
            source_type="document",
            source_id="doc-1",
            content="test content for hashing",
            embedding=np.zeros(384, dtype=np.float32),
        )
        assert len(record.content_hash) == 64

    def test_same_content_same_hash(self):
        content = "identical content"
        r1 = VectorRecord(
            source_type="doc", source_id="1",
            content=content,
            embedding=np.zeros(384, dtype=np.float32),
        )
        r2 = VectorRecord(
            source_type="doc", source_id="2",
            content=content,
            embedding=np.zeros(384, dtype=np.float32),
        )
        assert r1.content_hash == r2.content_hash

    def test_different_content_different_hash(self):
        r1 = VectorRecord(
            source_type="doc", source_id="1",
            content="content a",
            embedding=np.zeros(384, dtype=np.float32),
        )
        r2 = VectorRecord(
            source_type="doc", source_id="2",
            content="content b",
            embedding=np.zeros(384, dtype=np.float32),
        )
        assert r1.content_hash != r2.content_hash


class TestBatchInsertResult:
    def test_success_rate_full(self):
        result = BatchInsertResult(
            total_count=100,
            inserted_count=100,
            failed_count=0,
            duplicate_count=0,
            processing_time_ms=500,
        )
        assert result.success_rate == 1.0

    def test_success_rate_partial(self):
        result = BatchInsertResult(
            total_count=100,
            inserted_count=80,
            failed_count=20,
            duplicate_count=0,
            processing_time_ms=500,
        )
        assert result.success_rate == 0.8

    def test_success_rate_zero(self):
        result = BatchInsertResult(
            total_count=0,
            inserted_count=0,
            failed_count=0,
            duplicate_count=0,
            processing_time_ms=0,
        )
        assert result.success_rate == 0.0


class TestJobStatus:
    def test_progress_calculation(self):
        job = JobStatus(
            id="j1", status="running", source_type="doc",
            source_count=200, processed_count=100, failed_count=20,
            error_message=None, started_at=None, completed_at=None,
            created_at=None,
        )
        assert job.progress_pct == 60.0

    def test_zero_source_count(self):
        job = JobStatus(
            id="j1", status="pending", source_type="doc",
            source_count=0, processed_count=0, failed_count=0,
            error_message=None, started_at=None, completed_at=None,
            created_at=None,
        )
        assert job.progress_pct == 0.0

    def test_is_complete_states(self):
        for status in ("completed", "failed", "cancelled"):
            job = JobStatus(
                id="j1", status=status, source_type="doc",
                source_count=100, processed_count=100, failed_count=0,
                error_message=None, started_at=None, completed_at=None,
                created_at=None,
            )
            assert job.is_complete

    def test_not_complete_states(self):
        for status in ("pending", "running"):
            job = JobStatus(
                id="j1", status=status, source_type="doc",
                source_count=100, processed_count=50, failed_count=0,
                error_message=None, started_at=None, completed_at=None,
                created_at=None,
            )
            assert not job.is_complete
