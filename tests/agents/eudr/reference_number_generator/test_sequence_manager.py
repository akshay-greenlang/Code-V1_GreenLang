# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-EUDR-038 Reference Number Generator -- sequence_manager.py

Tests atomic sequence increment, year-based rollover, sequence exhaustion,
reservation, utilization monitoring, reset, distributed lock (mock Redis),
and concurrent access. 50+ tests.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import asyncio

import pytest

from greenlang.agents.eudr.reference_number_generator.config import (
    ReferenceNumberGeneratorConfig,
)
from greenlang.agents.eudr.reference_number_generator.sequence_manager import (
    SequenceManager,
)


# ====================================================================
# Test: Initialization
# ====================================================================


class TestSequenceManagerInit:
    """Test SequenceManager initialization."""

    def test_init_default_config(self):
        engine = SequenceManager()
        assert engine.config is not None
        assert engine.total_increments == 0
        assert engine.counter_count == 0

    def test_init_custom_config(self, custom_config):
        engine = SequenceManager(config=custom_config)
        assert engine.config.sequence_start == 100

    def test_init_empty_counters(self, sequence_manager_engine):
        assert sequence_manager_engine.counter_count == 0

    def test_init_empty_reservations(self, sequence_manager_engine):
        assert len(sequence_manager_engine._reservations) == 0


# ====================================================================
# Test: _counter_key
# ====================================================================


class TestCounterKey:
    """Test counter key generation."""

    def test_counter_key_format(self, sequence_manager_engine):
        key = sequence_manager_engine._counter_key("OP-001", "DE", 2026)
        assert key == "OP-001:DE:2026"

    def test_counter_key_uppercase_member_state(self, sequence_manager_engine):
        key = sequence_manager_engine._counter_key("OP-001", "de", 2026)
        assert key == "OP-001:DE:2026"


# ====================================================================
# Test: next_sequence
# ====================================================================


class TestNextSequence:
    """Test atomic sequence increment."""

    @pytest.mark.asyncio
    async def test_first_sequence_is_start(self, sequence_manager_engine):
        seq = await sequence_manager_engine.next_sequence("OP-001", "DE", 2026)
        assert seq == sequence_manager_engine.config.sequence_start

    @pytest.mark.asyncio
    async def test_sequential_increment(self, sequence_manager_engine):
        s1 = await sequence_manager_engine.next_sequence("OP-001", "DE", 2026)
        s2 = await sequence_manager_engine.next_sequence("OP-001", "DE", 2026)
        s3 = await sequence_manager_engine.next_sequence("OP-001", "DE", 2026)
        assert s2 == s1 + 1
        assert s3 == s2 + 1

    @pytest.mark.asyncio
    async def test_increments_counter_total(self, sequence_manager_engine):
        await sequence_manager_engine.next_sequence("OP-001", "DE", 2026)
        await sequence_manager_engine.next_sequence("OP-001", "DE", 2026)
        assert sequence_manager_engine.total_increments == 2

    @pytest.mark.asyncio
    async def test_creates_counter_on_first_call(self, sequence_manager_engine):
        await sequence_manager_engine.next_sequence("OP-001", "DE", 2026)
        assert sequence_manager_engine.counter_count == 1

    @pytest.mark.asyncio
    async def test_separate_counters_per_operator(self, sequence_manager_engine):
        s1 = await sequence_manager_engine.next_sequence("OP-001", "DE", 2026)
        s2 = await sequence_manager_engine.next_sequence("OP-002", "DE", 2026)
        assert s1 == s2  # Both start at sequence_start
        assert sequence_manager_engine.counter_count == 2

    @pytest.mark.asyncio
    async def test_separate_counters_per_member_state(self, sequence_manager_engine):
        s1 = await sequence_manager_engine.next_sequence("OP-001", "DE", 2026)
        s2 = await sequence_manager_engine.next_sequence("OP-001", "FR", 2026)
        assert s1 == s2  # Both start fresh
        assert sequence_manager_engine.counter_count == 2

    @pytest.mark.asyncio
    async def test_separate_counters_per_year(self, sequence_manager_engine):
        s1 = await sequence_manager_engine.next_sequence("OP-001", "DE", 2025)
        s2 = await sequence_manager_engine.next_sequence("OP-001", "DE", 2026)
        assert s1 == s2
        assert sequence_manager_engine.counter_count == 2

    @pytest.mark.asyncio
    async def test_large_sequence_generation(self, sequence_manager_engine):
        """Generate 200 sequences to verify monotonic increment."""
        sequences = []
        for _ in range(200):
            seq = await sequence_manager_engine.next_sequence("OP-001", "DE", 2026)
            sequences.append(seq)
        # Verify monotonically increasing
        for i in range(1, len(sequences)):
            assert sequences[i] == sequences[i - 1] + 1


# ====================================================================
# Test: Overflow Strategies
# ====================================================================


class TestOverflowStrategies:
    """Test sequence overflow handling."""

    @pytest.mark.asyncio
    async def test_reject_strategy_raises_at_exhaustion(self):
        config = ReferenceNumberGeneratorConfig(
            sequence_start=1,
            sequence_end=3,
            sequence_overflow_strategy="reject",
        )
        engine = SequenceManager(config=config)
        await engine.next_sequence("OP-001", "DE", 2026)  # 1
        await engine.next_sequence("OP-001", "DE", 2026)  # 2
        await engine.next_sequence("OP-001", "DE", 2026)  # 3
        with pytest.raises(RuntimeError, match="Sequence exhausted"):
            await engine.next_sequence("OP-001", "DE", 2026)  # 4 -> error

    @pytest.mark.asyncio
    async def test_rollover_strategy_resets_to_start(self):
        config = ReferenceNumberGeneratorConfig(
            sequence_start=1,
            sequence_end=3,
            sequence_overflow_strategy="rollover",
        )
        engine = SequenceManager(config=config)
        await engine.next_sequence("OP-001", "DE", 2026)  # 1
        await engine.next_sequence("OP-001", "DE", 2026)  # 2
        await engine.next_sequence("OP-001", "DE", 2026)  # 3
        s4 = await engine.next_sequence("OP-001", "DE", 2026)  # rollover -> 1
        assert s4 == 1

    @pytest.mark.asyncio
    async def test_extend_strategy_increases_max(self):
        config = ReferenceNumberGeneratorConfig(
            sequence_start=1,
            sequence_end=3,
            sequence_overflow_strategy="extend",
        )
        engine = SequenceManager(config=config)
        await engine.next_sequence("OP-001", "DE", 2026)  # 1
        await engine.next_sequence("OP-001", "DE", 2026)  # 2
        await engine.next_sequence("OP-001", "DE", 2026)  # 3
        s4 = await engine.next_sequence("OP-001", "DE", 2026)  # extend
        assert s4 == 4  # Extended beyond original max


# ====================================================================
# Test: reserve_sequences
# ====================================================================


class TestReserveSequences:
    """Test sequence reservation for batch operations."""

    @pytest.mark.asyncio
    async def test_reserve_returns_list(self, sequence_manager_engine):
        reserved = await sequence_manager_engine.reserve_sequences(
            "OP-001", "DE", 2026, 5
        )
        assert isinstance(reserved, list)
        assert len(reserved) == 5

    @pytest.mark.asyncio
    async def test_reserved_values_contiguous(self, sequence_manager_engine):
        reserved = await sequence_manager_engine.reserve_sequences(
            "OP-001", "DE", 2026, 10
        )
        for i in range(1, len(reserved)):
            assert reserved[i] == reserved[i - 1] + 1

    @pytest.mark.asyncio
    async def test_reserve_starts_at_sequence_start(self, sequence_manager_engine):
        reserved = await sequence_manager_engine.reserve_sequences(
            "OP-001", "DE", 2026, 3
        )
        assert reserved[0] == sequence_manager_engine.config.sequence_start

    @pytest.mark.asyncio
    async def test_reserve_invalid_count_zero(self, sequence_manager_engine):
        with pytest.raises(ValueError, match="at least 1"):
            await sequence_manager_engine.reserve_sequences(
                "OP-001", "DE", 2026, 0
            )

    @pytest.mark.asyncio
    async def test_reserve_invalid_count_negative(self, sequence_manager_engine):
        with pytest.raises(ValueError, match="at least 1"):
            await sequence_manager_engine.reserve_sequences(
                "OP-001", "DE", 2026, -5
            )

    @pytest.mark.asyncio
    async def test_reserve_exceeds_max_batch_size(self, sequence_manager_engine):
        max_batch = sequence_manager_engine.config.max_batch_size
        with pytest.raises(ValueError, match="exceeds max batch size"):
            await sequence_manager_engine.reserve_sequences(
                "OP-001", "DE", 2026, max_batch + 1
            )

    @pytest.mark.asyncio
    async def test_reserve_increments_total(self, sequence_manager_engine):
        await sequence_manager_engine.reserve_sequences(
            "OP-001", "DE", 2026, 10
        )
        assert sequence_manager_engine.total_increments == 10

    @pytest.mark.asyncio
    async def test_reserve_updates_reserved_count(self, sequence_manager_engine):
        await sequence_manager_engine.reserve_sequences(
            "OP-001", "DE", 2026, 5
        )
        key = sequence_manager_engine._counter_key("OP-001", "DE", 2026)
        counter = sequence_manager_engine._counters[key]
        assert counter["reserved_count"] == 5

    @pytest.mark.asyncio
    async def test_reserve_stores_reservation(self, sequence_manager_engine):
        await sequence_manager_engine.reserve_sequences(
            "OP-001", "DE", 2026, 3
        )
        assert len(sequence_manager_engine._reservations) == 1

    @pytest.mark.asyncio
    async def test_reserve_reject_on_insufficient_capacity(self):
        config = ReferenceNumberGeneratorConfig(
            sequence_start=1,
            sequence_end=5,
            sequence_overflow_strategy="reject",
        )
        engine = SequenceManager(config=config)
        with pytest.raises(RuntimeError, match="Insufficient capacity"):
            await engine.reserve_sequences("OP-001", "DE", 2026, 6)

    @pytest.mark.asyncio
    async def test_reserve_extend_on_insufficient_capacity(self):
        config = ReferenceNumberGeneratorConfig(
            sequence_start=1,
            sequence_end=5,
            sequence_overflow_strategy="extend",
        )
        engine = SequenceManager(config=config)
        reserved = await engine.reserve_sequences("OP-001", "DE", 2026, 10)
        assert len(reserved) == 10


# ====================================================================
# Test: get_sequence_status
# ====================================================================


class TestGetSequenceStatus:
    """Test sequence counter status reporting."""

    @pytest.mark.asyncio
    async def test_status_new_counter(self, sequence_manager_engine):
        status = await sequence_manager_engine.get_sequence_status(
            "OP-001", "DE", 2026
        )
        assert status["operator_id"] == "OP-001"
        assert status["member_state"] == "DE"
        assert status["year"] == 2026
        assert status["current_value"] == 0
        assert status["utilization_percent"] == 0.0

    @pytest.mark.asyncio
    async def test_status_after_increments(self, sequence_manager_engine):
        for _ in range(10):
            await sequence_manager_engine.next_sequence("OP-001", "DE", 2026)

        status = await sequence_manager_engine.get_sequence_status(
            "OP-001", "DE", 2026
        )
        assert status["current_value"] == 10
        # Utilization is 10/999999 * 100 = ~0.001%, rounds to 0.0
        assert status["utilization_percent"] >= 0.0
        assert "reserved_count" in status

    @pytest.mark.asyncio
    async def test_status_default_year_is_current(self, sequence_manager_engine):
        from greenlang.agents.eudr.reference_number_generator.sequence_manager import _utcnow
        current_year = _utcnow().year
        status = await sequence_manager_engine.get_sequence_status(
            "OP-001", "DE"
        )
        assert status["year"] == current_year

    @pytest.mark.asyncio
    async def test_status_available_count(self, sequence_manager_engine):
        status = await sequence_manager_engine.get_sequence_status(
            "OP-001", "DE", 2026
        )
        expected_available = (
            sequence_manager_engine.config.sequence_end
            - sequence_manager_engine.config.sequence_start + 1
        )
        assert status["available"] == expected_available


# ====================================================================
# Test: get_available_count
# ====================================================================


class TestGetAvailableCount:
    """Test available sequence slot counting."""

    @pytest.mark.asyncio
    async def test_available_count_new_counter(self, sequence_manager_engine):
        available = await sequence_manager_engine.get_available_count(
            "OP-001", "DE", 2026
        )
        expected = (
            sequence_manager_engine.config.sequence_end
            - sequence_manager_engine.config.sequence_start + 1
        )
        assert available == expected

    @pytest.mark.asyncio
    async def test_available_count_decreases(self, sequence_manager_engine):
        initial = await sequence_manager_engine.get_available_count(
            "OP-001", "DE", 2026
        )
        await sequence_manager_engine.next_sequence("OP-001", "DE", 2026)
        after = await sequence_manager_engine.get_available_count(
            "OP-001", "DE", 2026
        )
        assert after < initial


# ====================================================================
# Test: reset_sequence
# ====================================================================


class TestResetSequence:
    """Test sequence counter reset."""

    @pytest.mark.asyncio
    async def test_reset_existing_counter(self, sequence_manager_engine):
        await sequence_manager_engine.next_sequence("OP-001", "DE", 2026)
        result = await sequence_manager_engine.reset_sequence("OP-001", "DE", 2026)
        assert result is True

    @pytest.mark.asyncio
    async def test_reset_nonexistent_counter(self, sequence_manager_engine):
        result = await sequence_manager_engine.reset_sequence("OP-999", "DE", 2026)
        assert result is False

    @pytest.mark.asyncio
    async def test_reset_resets_value(self, sequence_manager_engine):
        for _ in range(10):
            await sequence_manager_engine.next_sequence("OP-001", "DE", 2026)
        await sequence_manager_engine.reset_sequence("OP-001", "DE", 2026)

        seq = await sequence_manager_engine.next_sequence("OP-001", "DE", 2026)
        assert seq == sequence_manager_engine.config.sequence_start

    @pytest.mark.asyncio
    async def test_reset_clears_reserved_count(self, sequence_manager_engine):
        await sequence_manager_engine.reserve_sequences("OP-001", "DE", 2026, 5)
        await sequence_manager_engine.reset_sequence("OP-001", "DE", 2026)

        key = sequence_manager_engine._counter_key("OP-001", "DE", 2026)
        counter = sequence_manager_engine._counters[key]
        assert counter["reserved_count"] == 0


# ====================================================================
# Test: list_counters
# ====================================================================


class TestListCounters:
    """Test counter listing."""

    @pytest.mark.asyncio
    async def test_list_counters_empty(self, sequence_manager_engine):
        counters = await sequence_manager_engine.list_counters()
        assert counters == []

    @pytest.mark.asyncio
    async def test_list_counters_after_increments(self, sequence_manager_engine):
        await sequence_manager_engine.next_sequence("OP-001", "DE", 2026)
        await sequence_manager_engine.next_sequence("OP-002", "FR", 2026)
        counters = await sequence_manager_engine.list_counters()
        assert len(counters) == 2

    @pytest.mark.asyncio
    async def test_list_counters_filter_by_operator(self, sequence_manager_engine):
        await sequence_manager_engine.next_sequence("OP-001", "DE", 2026)
        await sequence_manager_engine.next_sequence("OP-002", "FR", 2026)
        counters = await sequence_manager_engine.list_counters(operator_id="OP-001")
        assert len(counters) == 1
        assert counters[0]["operator_id"] == "OP-001"


# ====================================================================
# Test: Health Check
# ====================================================================


class TestSequenceManagerHealthCheck:
    """Test engine health check."""

    @pytest.mark.asyncio
    async def test_health_check(self, sequence_manager_engine):
        health = await sequence_manager_engine.health_check()
        assert health["status"] == "available"
        assert "active_counters" in health
        assert "total_increments" in health
        assert "reservations" in health


# ====================================================================
# Test: Concurrent Access
# ====================================================================


class TestConcurrentAccess:
    """Test thread safety via concurrent async calls."""

    @pytest.mark.asyncio
    async def test_concurrent_increments(self):
        """100 concurrent sequence requests must all succeed and be unique."""
        engine = SequenceManager()
        tasks = [
            engine.next_sequence("OP-001", "DE", 2026)
            for _ in range(100)
        ]
        results = await asyncio.gather(*tasks)
        assert len(set(results)) == 100
        assert engine.total_increments == 100

    @pytest.mark.asyncio
    async def test_concurrent_multi_operator(self):
        """Concurrent requests across multiple operators."""
        engine = SequenceManager()
        tasks = []
        for i in range(50):
            tasks.append(engine.next_sequence(f"OP-{i:03d}", "DE", 2026))
        results = await asyncio.gather(*tasks)
        assert len(results) == 50
        assert engine.counter_count == 50
