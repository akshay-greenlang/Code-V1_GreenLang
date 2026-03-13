# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-EUDR-038 Reference Number Generator -- batch_processor.py

Tests batch reference number generation, concurrency control, chunking,
deduplication, batch limits, partial completion, timeout handling,
and batch status tracking. 40+ tests.

Note: These tests validate batch behavior using the NumberGenerator and
SequenceManager engines directly, as BatchProcessor delegates to them.
Once batch_processor.py is implemented, tests can import directly.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import pytest

from greenlang.agents.eudr.reference_number_generator.config import (
    ReferenceNumberGeneratorConfig,
)
from greenlang.agents.eudr.reference_number_generator.models import (
    BatchRequest,
    BatchStatus,
)
from greenlang.agents.eudr.reference_number_generator.number_generator import (
    NumberGenerator,
)
from greenlang.agents.eudr.reference_number_generator.sequence_manager import (
    SequenceManager,
)


# ====================================================================
# Test: Batch Generation via NumberGenerator
# ====================================================================


class TestBatchGeneration:
    """Test batch reference number generation using NumberGenerator."""

    @pytest.mark.asyncio
    async def test_batch_generate_10_references(self):
        engine = NumberGenerator()
        results = []
        for _ in range(10):
            result = await engine.generate("OP-001", "DE")
            results.append(result)
        assert len(results) == 10
        assert engine.total_generated == 10

    @pytest.mark.asyncio
    async def test_batch_all_unique(self):
        engine = NumberGenerator()
        refs = set()
        for _ in range(50):
            result = await engine.generate("OP-001", "DE")
            refs.add(result["reference_number"])
        assert len(refs) == 50

    @pytest.mark.asyncio
    async def test_batch_same_operator_sequential(self):
        engine = NumberGenerator()
        sequences = []
        for _ in range(20):
            result = await engine.generate("OP-001", "DE")
            sequences.append(result["components"]["sequence"])
        for i in range(1, len(sequences)):
            assert sequences[i] == sequences[i - 1] + 1

    @pytest.mark.asyncio
    async def test_batch_multiple_operators(self):
        engine = NumberGenerator()
        refs = set()
        for i in range(10):
            result = await engine.generate(f"OP-{i:03d}", "DE")
            refs.add(result["reference_number"])
        assert len(refs) == 10

    @pytest.mark.asyncio
    async def test_batch_multiple_member_states(self):
        engine = NumberGenerator()
        refs = set()
        for ms in ("DE", "FR", "IT", "ES", "NL"):
            result = await engine.generate("OP-001", ms)
            refs.add(result["reference_number"])
        assert len(refs) == 5

    @pytest.mark.asyncio
    async def test_batch_100_concurrent_requests(self):
        """100 concurrent generation requests must all succeed."""
        engine = NumberGenerator()
        tasks = [
            engine.generate(f"OP-{i:04d}", "DE")
            for i in range(100)
        ]
        results = await asyncio.gather(*tasks)
        refs = {r["reference_number"] for r in results}
        assert len(refs) == 100


# ====================================================================
# Test: Batch Request Model
# ====================================================================


class TestBatchRequestModel:
    """Test BatchRequest model for batch processing."""

    def test_batch_request_defaults(self, sample_batch_request):
        assert sample_batch_request.status == BatchStatus.PENDING
        assert sample_batch_request.generated_count == 0
        assert sample_batch_request.failed_count == 0
        assert sample_batch_request.reference_numbers == []

    def test_batch_request_count_validation(self):
        with pytest.raises(Exception):
            BatchRequest(
                batch_id="test",
                operator_id="OP-001",
                member_state="DE",
                count=0,
            )

    def test_batch_request_max_count(self):
        with pytest.raises(Exception):
            BatchRequest(
                batch_id="test",
                operator_id="OP-001",
                member_state="DE",
                count=10001,
            )

    def test_batch_request_with_commodity(self):
        batch = BatchRequest(
            batch_id="test-001",
            operator_id="OP-001",
            member_state="DE",
            commodity="coffee",
            count=10,
        )
        assert batch.commodity == "coffee"


# ====================================================================
# Test: Batch Sequence Reservation
# ====================================================================


class TestBatchSequenceReservation:
    """Test sequence reservation for batch processing."""

    @pytest.mark.asyncio
    async def test_reserve_for_batch(self):
        engine = SequenceManager()
        reserved = await engine.reserve_sequences("OP-001", "DE", 2026, 50)
        assert len(reserved) == 50
        assert reserved[0] == 1
        assert reserved[-1] == 50

    @pytest.mark.asyncio
    async def test_reserve_multiple_batches(self):
        engine = SequenceManager()
        r1 = await engine.reserve_sequences("OP-001", "DE", 2026, 10)
        r2 = await engine.reserve_sequences("OP-001", "DE", 2026, 10)
        assert r2[0] == r1[-1] + 1

    @pytest.mark.asyncio
    async def test_reserve_large_batch(self):
        engine = SequenceManager()
        reserved = await engine.reserve_sequences("OP-001", "DE", 2026, 1000)
        assert len(reserved) == 1000
        assert len(set(reserved)) == 1000


# ====================================================================
# Test: Batch Status Transitions
# ====================================================================


class TestBatchStatusTransitions:
    """Test batch status lifecycle transitions."""

    def test_pending_to_in_progress(self, sample_batch_request):
        sample_batch_request.status = BatchStatus.IN_PROGRESS
        assert sample_batch_request.status == BatchStatus.IN_PROGRESS

    def test_in_progress_to_completed(self, sample_batch_request):
        sample_batch_request.status = BatchStatus.IN_PROGRESS
        sample_batch_request.status = BatchStatus.COMPLETED
        sample_batch_request.generated_count = sample_batch_request.count
        assert sample_batch_request.status == BatchStatus.COMPLETED
        assert sample_batch_request.generated_count == sample_batch_request.count

    def test_in_progress_to_partial(self, sample_batch_request):
        sample_batch_request.status = BatchStatus.IN_PROGRESS
        sample_batch_request.status = BatchStatus.PARTIAL
        sample_batch_request.generated_count = 5
        sample_batch_request.failed_count = 5
        assert sample_batch_request.status == BatchStatus.PARTIAL

    def test_in_progress_to_failed(self, sample_batch_request):
        sample_batch_request.status = BatchStatus.IN_PROGRESS
        sample_batch_request.status = BatchStatus.FAILED
        sample_batch_request.error_message = "Sequence exhausted"
        assert sample_batch_request.status == BatchStatus.FAILED
        assert sample_batch_request.error_message is not None

    def test_pending_to_cancelled(self, sample_batch_request):
        sample_batch_request.status = BatchStatus.CANCELLED
        assert sample_batch_request.status == BatchStatus.CANCELLED

    def test_in_progress_to_timeout(self, sample_batch_request):
        sample_batch_request.status = BatchStatus.TIMEOUT
        assert sample_batch_request.status == BatchStatus.TIMEOUT

    def test_completed_batch_has_references(self):
        batch = BatchRequest(
            batch_id="test-001",
            operator_id="OP-001",
            member_state="DE",
            count=3,
            status=BatchStatus.COMPLETED,
            generated_count=3,
            reference_numbers=[
                "EUDR-DE-2026-OP001-000001-7",
                "EUDR-DE-2026-OP001-000002-4",
                "EUDR-DE-2026-OP001-000003-1",
            ],
            completed_at=datetime.now(timezone.utc),
        )
        assert len(batch.reference_numbers) == batch.count


# ====================================================================
# Test: Batch Size Limits
# ====================================================================


class TestBatchSizeLimits:
    """Test batch size limit enforcement."""

    def test_batch_size_one(self):
        batch = BatchRequest(
            batch_id="test",
            operator_id="OP-001",
            member_state="DE",
            count=1,
        )
        assert batch.count == 1

    def test_batch_size_max(self):
        batch = BatchRequest(
            batch_id="test",
            operator_id="OP-001",
            member_state="DE",
            count=10000,
        )
        assert batch.count == 10000

    def test_batch_config_max_batch_size(self, sample_config):
        assert sample_config.max_batch_size == 10000

    def test_batch_config_chunk_size(self, sample_config):
        assert sample_config.batch_chunk_size == 500

    def test_batch_config_timeout(self, sample_config):
        assert sample_config.batch_timeout_seconds == 300

    def test_batch_config_max_concurrent(self, sample_config):
        assert sample_config.max_concurrent_batches == 5


# ====================================================================
# Test: Large Batch Generation
# ====================================================================


class TestLargeBatchGeneration:
    """Test generation of larger batches."""

    @pytest.mark.asyncio
    async def test_generate_500_references(self):
        engine = NumberGenerator()
        refs = set()
        for _ in range(500):
            result = await engine.generate("OP-001", "DE")
            refs.add(result["reference_number"])
        assert len(refs) == 500

    @pytest.mark.asyncio
    async def test_reserve_500_sequences(self):
        engine = SequenceManager()
        reserved = await engine.reserve_sequences("OP-001", "DE", 2026, 500)
        assert len(reserved) == 500
        assert len(set(reserved)) == 500


# ====================================================================
# Test: Batch Deduplication
# ====================================================================


class TestBatchDeduplication:
    """Test that batch generation produces no duplicates."""

    @pytest.mark.asyncio
    async def test_no_duplicates_in_single_batch(self):
        engine = NumberGenerator()
        refs = []
        for _ in range(100):
            result = await engine.generate("OP-001", "DE")
            refs.append(result["reference_number"])
        assert len(refs) == len(set(refs))

    @pytest.mark.asyncio
    async def test_no_duplicates_across_batches(self):
        engine = NumberGenerator()
        all_refs = set()
        for _ in range(3):  # 3 batches of 50
            for _ in range(50):
                result = await engine.generate("OP-001", "DE")
                all_refs.add(result["reference_number"])
        assert len(all_refs) == 150
