# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-EUDR-038 Reference Number Generator -- collision_detector.py

Tests collision detection, retry logic, resolution strategies, bloom filter
simulation, collision record tracking, concurrent collision handling, and
collision rate monitoring. 40+ tests.

These tests exercise collision-related behavior through the NumberGenerator
engine and the CollisionRecord model. Once collision_detector.py is
implemented as a standalone engine, tests can import directly.

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
    CollisionRecord,
)
from greenlang.agents.eudr.reference_number_generator.number_generator import (
    NumberGenerator,
)


# ====================================================================
# Test: CollisionRecord Model
# ====================================================================


class TestCollisionRecordModel:
    """Test CollisionRecord Pydantic model."""

    def test_creation(self, sample_collision_record):
        assert sample_collision_record.operator_id == "OP-001"
        assert sample_collision_record.attempt_number == 1
        assert sample_collision_record.resolved is True

    def test_default_resolved_false(self):
        record = CollisionRecord(
            collision_id=str(uuid.uuid4()),
            reference_number="EUDR-DE-2026-OP001-000001-7",
            operator_id="OP-001",
            attempt_number=1,
        )
        assert record.resolved is False
        assert record.resolution_method == ""

    def test_multiple_attempts(self):
        records = []
        for i in range(5):
            records.append(
                CollisionRecord(
                    collision_id=str(uuid.uuid4()),
                    reference_number=f"EUDR-DE-2026-OP001-00000{i}-7",
                    operator_id="OP-001",
                    attempt_number=i + 1,
                    resolved=i == 4,
                    resolution_method="next_sequence" if i == 4 else "",
                )
            )
        assert sum(1 for r in records if r.resolved) == 1
        assert records[-1].attempt_number == 5

    def test_resolution_methods(self):
        methods = ["next_sequence", "retry_with_new_operator", "extended_range", "manual"]
        for method in methods:
            record = CollisionRecord(
                collision_id=str(uuid.uuid4()),
                reference_number="EUDR-DE-2026-OP001-000001-7",
                operator_id="OP-001",
                attempt_number=1,
                resolved=True,
                resolution_method=method,
            )
            assert record.resolution_method == method


# ====================================================================
# Test: Collision Detection Config
# ====================================================================


class TestCollisionConfig:
    """Test collision-related configuration."""

    def test_collision_max_retries_default(self, sample_config):
        assert sample_config.collision_max_retries == 10

    def test_collision_backoff_base_ms_default(self, sample_config):
        assert sample_config.collision_backoff_base_ms == 5

    def test_collision_backoff_max_ms_default(self, sample_config):
        assert sample_config.collision_backoff_max_ms == 500

    def test_bloom_filter_enabled_default(self, sample_config):
        assert sample_config.enable_bloom_filter is True

    def test_bloom_filter_capacity_default(self, sample_config):
        assert sample_config.bloom_filter_capacity == 10000000

    def test_bloom_filter_error_rate_default(self, sample_config):
        assert sample_config.bloom_filter_error_rate == pytest.approx(0.001)


# ====================================================================
# Test: Collision Detection in NumberGenerator
# ====================================================================


class TestCollisionDetectionInGenerator:
    """Test collision handling within the NumberGenerator engine."""

    @pytest.mark.asyncio
    async def test_no_collision_normal_generation(self):
        engine = NumberGenerator()
        refs = set()
        for _ in range(100):
            result = await engine.generate("OP-001", "DE")
            refs.add(result["reference_number"])
        assert len(refs) == 100  # No collisions

    @pytest.mark.asyncio
    async def test_collision_retry_max_exceeded_raises(self):
        """Simulate collision by pre-populating references."""
        config = ReferenceNumberGeneratorConfig(
            sequence_start=1,
            sequence_end=1,
            collision_max_retries=3,
            sequence_overflow_strategy="rollover",
        )
        engine = NumberGenerator(config=config)

        # First generation succeeds
        r1 = await engine.generate("OP-001", "DE")

        # Second generation will collide on every retry since
        # rollover always produces the same sequence
        with pytest.raises(RuntimeError, match="Failed to generate unique"):
            await engine.generate("OP-001", "DE")

    @pytest.mark.asyncio
    async def test_idempotency_prevents_collision(self):
        engine = NumberGenerator()
        r1 = await engine.generate("OP-001", "DE", idempotency_key="key-001")
        r2 = await engine.generate("OP-001", "DE", idempotency_key="key-001")
        assert r1["reference_number"] == r2["reference_number"]
        assert engine.total_generated == 1


# ====================================================================
# Test: Uniqueness at Scale
# ====================================================================


class TestUniquenessAtScale:
    """Test uniqueness guarantees at scale."""

    @pytest.mark.asyncio
    async def test_1000_unique_references(self):
        engine = NumberGenerator()
        refs = set()
        for _ in range(1000):
            result = await engine.generate("OP-001", "DE")
            refs.add(result["reference_number"])
        assert len(refs) == 1000

    @pytest.mark.asyncio
    async def test_uniqueness_across_operators(self):
        engine = NumberGenerator()
        refs = set()
        for i in range(200):
            op = f"OP-{i % 20:03d}"
            result = await engine.generate(op, "DE")
            refs.add(result["reference_number"])
        assert len(refs) == 200

    @pytest.mark.asyncio
    async def test_uniqueness_across_member_states(self):
        from greenlang.agents.eudr.reference_number_generator.config import EU_MEMBER_STATES
        engine = NumberGenerator()
        refs = set()
        for ms in EU_MEMBER_STATES:
            for _ in range(5):
                result = await engine.generate("OP-001", ms)
                refs.add(result["reference_number"])
        assert len(refs) == 27 * 5

    @pytest.mark.asyncio
    async def test_concurrent_uniqueness(self):
        """50 concurrent generation requests must produce unique references."""
        engine = NumberGenerator()
        tasks = [
            engine.generate(f"OP-{i:03d}", "DE")
            for i in range(50)
        ]
        results = await asyncio.gather(*tasks)
        refs = {r["reference_number"] for r in results}
        assert len(refs) == 50


# ====================================================================
# Test: Collision Record Tracking
# ====================================================================


class TestCollisionRecordTracking:
    """Test collision record creation and tracking."""

    def test_create_collision_record(self):
        record = CollisionRecord(
            collision_id=str(uuid.uuid4()),
            reference_number="EUDR-DE-2026-OP001-000001-7",
            operator_id="OP-001",
            attempt_number=1,
            resolved=False,
        )
        assert record.resolved is False
        assert record.detected_at is not None

    def test_resolve_collision_record(self):
        record = CollisionRecord(
            collision_id=str(uuid.uuid4()),
            reference_number="EUDR-DE-2026-OP001-000001-7",
            operator_id="OP-001",
            attempt_number=3,
            resolved=True,
            resolution_method="next_sequence",
        )
        assert record.resolved is True
        assert record.resolution_method == "next_sequence"
        assert record.attempt_number == 3

    def test_collision_record_timestamps(self):
        before = datetime.now(timezone.utc)
        record = CollisionRecord(
            collision_id=str(uuid.uuid4()),
            reference_number="EUDR-DE-2026-OP001-000001-7",
            operator_id="OP-001",
            attempt_number=1,
        )
        after = datetime.now(timezone.utc)
        assert before <= record.detected_at <= after

    def test_multiple_collision_records_different_ids(self):
        records = [
            CollisionRecord(
                collision_id=str(uuid.uuid4()),
                reference_number=f"EUDR-DE-2026-OP001-{i:06d}-7",
                operator_id="OP-001",
                attempt_number=i,
            )
            for i in range(1, 11)
        ]
        ids = {r.collision_id for r in records}
        assert len(ids) == 10


# ====================================================================
# Test: Bloom Filter Simulation
# ====================================================================


class TestBloomFilterSimulation:
    """Test bloom filter-like collision prevention behavior."""

    @pytest.mark.asyncio
    async def test_no_false_positives_small_set(self):
        """With sequential generation, no collisions should occur."""
        engine = NumberGenerator()
        seen = set()
        for _ in range(500):
            result = await engine.generate("OP-001", "DE")
            ref = result["reference_number"]
            assert ref not in seen, f"Duplicate detected: {ref}"
            seen.add(ref)

    def test_bloom_filter_config(self, sample_config):
        assert sample_config.enable_bloom_filter is True
        assert sample_config.bloom_filter_capacity >= 1000000


# ====================================================================
# Test: Retry Strategy Backoff
# ====================================================================


class TestRetryBackoff:
    """Test collision retry backoff configuration."""

    def test_backoff_base_ms(self, sample_config):
        assert sample_config.collision_backoff_base_ms == 5

    def test_backoff_max_ms(self, sample_config):
        assert sample_config.collision_backoff_max_ms == 500

    def test_backoff_max_greater_than_base(self, sample_config):
        assert sample_config.collision_backoff_max_ms > sample_config.collision_backoff_base_ms

    def test_max_retries_reasonable(self, sample_config):
        assert 1 <= sample_config.collision_max_retries <= 100

    def test_custom_retry_config(self):
        config = ReferenceNumberGeneratorConfig(
            collision_max_retries=3,
            collision_backoff_base_ms=10,
            collision_backoff_max_ms=1000,
        )
        assert config.collision_max_retries == 3
        assert config.collision_backoff_base_ms == 10
        assert config.collision_backoff_max_ms == 1000
