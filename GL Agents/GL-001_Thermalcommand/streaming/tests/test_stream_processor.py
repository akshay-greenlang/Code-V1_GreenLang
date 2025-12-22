"""
Tests for Stream Processor Module - GL-001 ThermalCommand

Comprehensive test coverage for stream processing, windowing,
and aggregation operations.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from typing import AsyncGenerator

from ..stream_processor import (
    # Configuration
    WindowConfig,
    StreamProcessorConfig,
    WindowType,
    AggregationType,
    WatermarkStrategy,
    # Results
    AggregationResult,
    AggregationValue,
    WindowBounds,
    GroupKey,
    # State
    WindowState,
    InMemoryStateStore,
    # Engine
    AggregationEngine,
    # Assigners
    TumblingWindowAssigner,
    SlidingWindowAssigner,
    SessionWindowAssigner,
    get_window_assigner,
    # Processor
    StreamProcessor,
    StreamProcessorBuilder,
    ProcessorMetrics,
)
from ..event_envelope import EventEnvelope
from ..kafka_schemas import (
    TelemetryNormalizedEvent,
    TelemetryPoint,
    QualityCode,
    UnitOfMeasure,
)


class TestWindowConfig:
    """Tests for window configuration."""

    def test_tumbling_window_config(self) -> None:
        """Test tumbling window configuration."""
        config = WindowConfig(
            window_type=WindowType.TUMBLING,
            window_size_seconds=60,
            aggregations=[AggregationType.AVG, AggregationType.MAX],
        )

        assert config.window_type == WindowType.TUMBLING
        assert config.window_size_seconds == 60
        assert AggregationType.AVG in config.aggregations

    def test_sliding_window_config(self) -> None:
        """Test sliding window requires slide_seconds."""
        config = WindowConfig(
            window_type=WindowType.SLIDING,
            window_size_seconds=300,
            slide_seconds=60,
            aggregations=[AggregationType.AVG],
        )

        assert config.slide_seconds == 60

    def test_sliding_window_missing_slide_raises(self) -> None:
        """Test sliding window without slide_seconds raises error."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            WindowConfig(
                window_type=WindowType.SLIDING,
                window_size_seconds=300,
                # Missing slide_seconds
                aggregations=[AggregationType.AVG],
            )

    def test_session_window_config(self) -> None:
        """Test session window requires gap_seconds."""
        config = WindowConfig(
            window_type=WindowType.SESSION,
            window_size_seconds=300,
            gap_seconds=30,
            aggregations=[AggregationType.COUNT],
        )

        assert config.gap_seconds == 30

    def test_window_config_with_grouping(self) -> None:
        """Test window configuration with group by fields."""
        config = WindowConfig(
            window_type=WindowType.TUMBLING,
            window_size_seconds=60,
            aggregations=[AggregationType.AVG],
            group_by_fields=["tag_id", "equipment_id"],
        )

        assert "tag_id" in config.group_by_fields
        assert "equipment_id" in config.group_by_fields

    def test_watermark_configuration(self) -> None:
        """Test watermark configuration options."""
        config = WindowConfig(
            window_type=WindowType.TUMBLING,
            window_size_seconds=60,
            aggregations=[AggregationType.AVG],
            watermark_seconds=30,
            watermark_strategy=WatermarkStrategy.ALLOW_LATE,
            allowed_lateness_seconds=300,
        )

        assert config.watermark_seconds == 30
        assert config.watermark_strategy == WatermarkStrategy.ALLOW_LATE
        assert config.allowed_lateness_seconds == 300


class TestWindowBounds:
    """Tests for window bounds."""

    def test_window_bounds_duration(self) -> None:
        """Test window duration calculation."""
        now = datetime.now(timezone.utc)
        bounds = WindowBounds(
            start=now,
            end=now + timedelta(seconds=60),
        )

        assert bounds.duration_seconds == 60.0

    def test_window_bounds_contains(self) -> None:
        """Test timestamp containment check."""
        now = datetime.now(timezone.utc)
        bounds = WindowBounds(
            start=now,
            end=now + timedelta(seconds=60),
        )

        assert bounds.contains(now + timedelta(seconds=30))
        assert not bounds.contains(now - timedelta(seconds=1))
        assert not bounds.contains(now + timedelta(seconds=61))


class TestGroupKey:
    """Tests for group key."""

    def test_group_key_equality(self) -> None:
        """Test group key equality."""
        key1 = GroupKey(fields={"tag_id": "T-101", "equipment_id": "boiler-01"})
        key2 = GroupKey(fields={"tag_id": "T-101", "equipment_id": "boiler-01"})
        key3 = GroupKey(fields={"tag_id": "T-102", "equipment_id": "boiler-01"})

        assert key1 == key2
        assert key1 != key3

    def test_group_key_hashable(self) -> None:
        """Test group key is hashable."""
        key1 = GroupKey(fields={"tag_id": "T-101"})
        key2 = GroupKey(fields={"tag_id": "T-101"})

        assert hash(key1) == hash(key2)

        # Can use as dict key
        d = {key1: "value1"}
        assert d[key2] == "value1"


class TestWindowState:
    """Tests for window state."""

    def test_window_state_add_value(self) -> None:
        """Test adding values to window state."""
        now = datetime.now(timezone.utc)
        state = WindowState(
            window_id="win-001",
            bounds=WindowBounds(start=now, end=now + timedelta(seconds=60)),
            group_key=None,
        )

        added = state.add_value(450.5, now, QualityCode.GOOD, "evt-001")
        assert added is True
        assert state.count == 1

        # Duplicate should not be added
        added = state.add_value(451.0, now, QualityCode.GOOD, "evt-001")
        assert added is False
        assert state.count == 1

    def test_window_state_clear(self) -> None:
        """Test clearing window state."""
        now = datetime.now(timezone.utc)
        state = WindowState(
            window_id="win-001",
            bounds=WindowBounds(start=now, end=now + timedelta(seconds=60)),
            group_key=None,
        )

        state.add_value(450.5, now, QualityCode.GOOD, "evt-001")
        state.add_value(451.0, now, QualityCode.GOOD, "evt-002")

        assert state.count == 2

        state.clear()

        assert state.count == 0
        assert len(state.event_ids) == 0


class TestAggregationEngine:
    """Tests for aggregation engine."""

    def test_count_aggregation(self) -> None:
        """Test count aggregation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = AggregationEngine.compute(values, AggregationType.COUNT)

        assert result.value == 5.0
        assert result.aggregation_type == AggregationType.COUNT

    def test_sum_aggregation(self) -> None:
        """Test sum aggregation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = AggregationEngine.compute(values, AggregationType.SUM)

        assert result.value == 15.0

    def test_avg_aggregation(self) -> None:
        """Test average aggregation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = AggregationEngine.compute(values, AggregationType.AVG)

        assert result.value == 3.0

    def test_min_max_aggregation(self) -> None:
        """Test min and max aggregation."""
        values = [1.0, 5.0, 3.0, 2.0, 4.0]

        min_result = AggregationEngine.compute(values, AggregationType.MIN)
        max_result = AggregationEngine.compute(values, AggregationType.MAX)

        assert min_result.value == 1.0
        assert max_result.value == 5.0

    def test_first_last_aggregation(self) -> None:
        """Test first and last aggregation."""
        values = [10.0, 20.0, 30.0]

        first_result = AggregationEngine.compute(values, AggregationType.FIRST)
        last_result = AggregationEngine.compute(values, AggregationType.LAST)

        assert first_result.value == 10.0
        assert last_result.value == 30.0

    def test_stddev_variance_aggregation(self) -> None:
        """Test standard deviation and variance."""
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]

        stddev_result = AggregationEngine.compute(values, AggregationType.STDDEV)
        variance_result = AggregationEngine.compute(values, AggregationType.VARIANCE)

        assert stddev_result.value == pytest.approx(2.0, rel=0.01)
        assert variance_result.value == pytest.approx(4.0, rel=0.01)

    def test_median_aggregation(self) -> None:
        """Test median aggregation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = AggregationEngine.compute(values, AggregationType.MEDIAN)

        assert result.value == 3.0

    def test_percentile_aggregation(self) -> None:
        """Test percentile aggregations."""
        values = list(range(1, 101))  # 1 to 100

        p90_result = AggregationEngine.compute(values, AggregationType.PERCENTILE_90)
        p95_result = AggregationEngine.compute(values, AggregationType.PERCENTILE_95)
        p99_result = AggregationEngine.compute(values, AggregationType.PERCENTILE_99)

        assert p90_result.value == pytest.approx(90.0, rel=0.1)
        assert p95_result.value == pytest.approx(95.0, rel=0.1)
        assert p99_result.value == pytest.approx(99.0, rel=0.1)

    def test_rate_aggregation(self) -> None:
        """Test rate (events per second) aggregation."""
        values = [1.0] * 60  # 60 events
        result = AggregationEngine.compute(
            values,
            AggregationType.RATE,
            window_seconds=60.0,
        )

        assert result.value == 1.0  # 60 events / 60 seconds

    def test_distinct_count_aggregation(self) -> None:
        """Test distinct count aggregation."""
        values = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0]
        result = AggregationEngine.compute(values, AggregationType.DISTINCT_COUNT)

        assert result.value == 3.0

    def test_empty_values_aggregation(self) -> None:
        """Test aggregation with empty values."""
        result = AggregationEngine.compute([], AggregationType.AVG)

        assert result.value == 0.0
        assert result.count == 0

    def test_compute_all(self) -> None:
        """Test computing all aggregations."""
        now = datetime.now(timezone.utc)
        state = WindowState(
            window_id="win-001",
            bounds=WindowBounds(start=now, end=now + timedelta(seconds=60)),
            group_key=None,
        )

        for i, v in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
            state.add_value(v, now, QualityCode.GOOD, f"evt-{i}")

        results = AggregationEngine.compute_all(
            state,
            [AggregationType.AVG, AggregationType.MIN, AggregationType.MAX],
        )

        assert "avg" in results
        assert "min" in results
        assert "max" in results
        assert results["avg"].value == 3.0
        assert results["min"].value == 1.0
        assert results["max"].value == 5.0


class TestWindowAssigners:
    """Tests for window assigners."""

    def test_tumbling_window_assigner(self) -> None:
        """Test tumbling window assignment."""
        assigner = TumblingWindowAssigner()
        config = WindowConfig(
            window_type=WindowType.TUMBLING,
            window_size_seconds=60,
            aggregations=[AggregationType.AVG],
        )

        # Timestamp at 12:01:30
        timestamp = datetime(2024, 1, 15, 12, 1, 30, tzinfo=timezone.utc)
        windows = assigner.assign_windows(timestamp, config)

        assert len(windows) == 1
        # Should be assigned to 12:01:00-12:02:00 window
        assert windows[0].start.minute == 1
        assert windows[0].start.second == 0

    def test_sliding_window_assigner(self) -> None:
        """Test sliding window assignment."""
        assigner = SlidingWindowAssigner()
        config = WindowConfig(
            window_type=WindowType.SLIDING,
            window_size_seconds=60,
            slide_seconds=30,
            aggregations=[AggregationType.AVG],
        )

        timestamp = datetime(2024, 1, 15, 12, 1, 30, tzinfo=timezone.utc)
        windows = assigner.assign_windows(timestamp, config)

        # With 60s window and 30s slide, event should be in 2 windows
        assert len(windows) == 2

    def test_session_window_assigner(self) -> None:
        """Test session window assignment."""
        assigner = SessionWindowAssigner()
        config = WindowConfig(
            window_type=WindowType.SESSION,
            window_size_seconds=60,
            gap_seconds=30,
            aggregations=[AggregationType.COUNT],
        )

        timestamp = datetime(2024, 1, 15, 12, 1, 30, tzinfo=timezone.utc)
        windows = assigner.assign_windows(timestamp, config)

        assert len(windows) == 1
        assert windows[0].start == timestamp

    def test_get_window_assigner(self) -> None:
        """Test getting appropriate assigner by type."""
        tumbling = get_window_assigner(WindowType.TUMBLING)
        sliding = get_window_assigner(WindowType.SLIDING)
        session = get_window_assigner(WindowType.SESSION)

        assert isinstance(tumbling, TumblingWindowAssigner)
        assert isinstance(sliding, SlidingWindowAssigner)
        assert isinstance(session, SessionWindowAssigner)


class TestInMemoryStateStore:
    """Tests for in-memory state store."""

    @pytest.fixture
    def store(self) -> InMemoryStateStore:
        """Create a test state store."""
        return InMemoryStateStore()

    @pytest.mark.asyncio
    async def test_put_and_get_window(self, store: InMemoryStateStore) -> None:
        """Test storing and retrieving window state."""
        now = datetime.now(timezone.utc)
        state = WindowState(
            window_id="win-001",
            bounds=WindowBounds(start=now, end=now + timedelta(seconds=60)),
            group_key=None,
        )
        state.add_value(450.5, now, QualityCode.GOOD, "evt-001")

        await store.put_window(state)
        retrieved = await store.get_window("win-001", None)

        assert retrieved is not None
        assert retrieved.count == 1
        assert retrieved.values[0] == 450.5

    @pytest.mark.asyncio
    async def test_delete_window(self, store: InMemoryStateStore) -> None:
        """Test deleting window state."""
        now = datetime.now(timezone.utc)
        state = WindowState(
            window_id="win-001",
            bounds=WindowBounds(start=now, end=now + timedelta(seconds=60)),
            group_key=None,
        )

        await store.put_window(state)
        await store.delete_window("win-001", None)
        retrieved = await store.get_window("win-001", None)

        assert retrieved is None

    @pytest.mark.asyncio
    async def test_get_expired_windows(self, store: InMemoryStateStore) -> None:
        """Test getting expired windows."""
        now = datetime.now(timezone.utc)

        # Old window
        old_state = WindowState(
            window_id="win-old",
            bounds=WindowBounds(
                start=now - timedelta(hours=2),
                end=now - timedelta(hours=1),
            ),
            group_key=None,
        )

        # Current window
        current_state = WindowState(
            window_id="win-current",
            bounds=WindowBounds(
                start=now,
                end=now + timedelta(seconds=60),
            ),
            group_key=None,
        )

        await store.put_window(old_state)
        await store.put_window(current_state)

        expired = await store.get_expired_windows(now - timedelta(minutes=30))

        assert len(expired) == 1
        assert expired[0].window_id == "win-old"

    @pytest.mark.asyncio
    async def test_checkpoint(self, store: InMemoryStateStore) -> None:
        """Test checkpoint creation."""
        now = datetime.now(timezone.utc)
        state = WindowState(
            window_id="win-001",
            bounds=WindowBounds(start=now, end=now + timedelta(seconds=60)),
            group_key=None,
        )
        state.add_value(450.5, now, QualityCode.GOOD, "evt-001")

        await store.put_window(state)
        checkpoint_id = await store.checkpoint()

        assert checkpoint_id.startswith("ckpt-")

    @pytest.mark.asyncio
    async def test_restore_from_checkpoint(self, store: InMemoryStateStore) -> None:
        """Test restoring from checkpoint."""
        now = datetime.now(timezone.utc)
        state = WindowState(
            window_id="win-001",
            bounds=WindowBounds(start=now, end=now + timedelta(seconds=60)),
            group_key=None,
        )
        state.add_value(450.5, now, QualityCode.GOOD, "evt-001")

        await store.put_window(state)
        checkpoint_id = await store.checkpoint()

        # Clear the store
        await store.delete_window("win-001", None)
        assert await store.get_window("win-001", None) is None

        # Restore
        restored = await store.restore(checkpoint_id)
        assert restored is True

        retrieved = await store.get_window("win-001", None)
        assert retrieved is not None
        assert retrieved.count == 1


class TestAggregationResult:
    """Tests for aggregation result."""

    def test_aggregation_result_creation(self) -> None:
        """Test aggregation result creation."""
        now = datetime.now(timezone.utc)
        result = AggregationResult(
            window_id="win-001",
            window_bounds=WindowBounds(
                start=now,
                end=now + timedelta(seconds=60),
            ),
            aggregations={
                "avg": AggregationValue(
                    aggregation_type=AggregationType.AVG,
                    value=450.5,
                    count=100,
                )
            },
            event_count=100,
        )

        assert result.window_id == "win-001"
        assert result.event_count == 100
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_aggregation_result_with_group_key(self) -> None:
        """Test aggregation result with group key."""
        now = datetime.now(timezone.utc)
        result = AggregationResult(
            window_id="win-001",
            window_bounds=WindowBounds(
                start=now,
                end=now + timedelta(seconds=60),
            ),
            group_key=GroupKey(fields={"tag_id": "T-101"}),
            aggregations={},
            event_count=50,
        )

        assert result.group_key is not None
        assert result.group_key.fields["tag_id"] == "T-101"


class TestStreamProcessor:
    """Tests for stream processor."""

    @pytest.fixture
    def processor(self) -> StreamProcessor:
        """Create a test processor."""
        config = StreamProcessorConfig(
            checkpoint_interval_seconds=5,
            dedup_enabled=True,
        )
        return StreamProcessor(config)

    def test_add_window(self, processor: StreamProcessor) -> None:
        """Test adding window configuration."""
        config = WindowConfig(
            window_type=WindowType.TUMBLING,
            window_size_seconds=60,
            aggregations=[AggregationType.AVG],
        )

        processor.add_window(config)

        assert config.window_id in processor._window_configs

    def test_remove_window(self, processor: StreamProcessor) -> None:
        """Test removing window configuration."""
        config = WindowConfig(
            window_type=WindowType.TUMBLING,
            window_size_seconds=60,
            aggregations=[AggregationType.AVG],
        )

        processor.add_window(config)
        removed = processor.remove_window(config.window_id)

        assert removed is True
        assert config.window_id not in processor._window_configs

    @pytest.mark.asyncio
    async def test_processor_start_stop(self, processor: StreamProcessor) -> None:
        """Test processor lifecycle."""
        processor.add_window(
            WindowConfig(
                window_type=WindowType.TUMBLING,
                window_size_seconds=60,
                aggregations=[AggregationType.AVG],
            )
        )

        await processor.start()
        assert processor._running is True

        await processor.stop()
        assert processor._running is False

    @pytest.mark.asyncio
    async def test_process_telemetry(self, processor: StreamProcessor) -> None:
        """Test processing telemetry events."""
        processor.add_window(
            WindowConfig(
                window_id="test-window",
                window_type=WindowType.TUMBLING,
                window_size_seconds=60,
                aggregations=[AggregationType.AVG, AggregationType.COUNT],
            )
        )

        await processor.start()

        try:
            now = datetime.now(timezone.utc)
            telemetry = TelemetryNormalizedEvent(
                source_system="test",
                points=[
                    TelemetryPoint(
                        tag_id="T-101",
                        value=450.5,
                        unit=UnitOfMeasure.CELSIUS,
                        timestamp=now,
                    )
                ],
                collection_timestamp=now,
                batch_id="batch-001",
                sequence_number=1,
            )

            envelope = EventEnvelope.create(
                event_type="gl001.telemetry.normalized",
                source="test",
                payload=telemetry,
            )

            results = []
            async for result in processor.process_telemetry(envelope):
                results.append(result)

            # Check metrics
            assert processor.metrics.events_processed == 1
        finally:
            await processor.stop()

    @pytest.mark.asyncio
    async def test_deduplication(self, processor: StreamProcessor) -> None:
        """Test event deduplication."""
        processor.add_window(
            WindowConfig(
                window_type=WindowType.TUMBLING,
                window_size_seconds=60,
                aggregations=[AggregationType.COUNT],
            )
        )

        await processor.start()

        try:
            now = datetime.now(timezone.utc)
            telemetry = TelemetryNormalizedEvent(
                source_system="test",
                points=[
                    TelemetryPoint(
                        tag_id="T-101",
                        value=450.5,
                        unit=UnitOfMeasure.CELSIUS,
                        timestamp=now,
                    )
                ],
                collection_timestamp=now,
                batch_id="batch-001",
                sequence_number=1,
            )

            # Same envelope sent twice
            envelope = EventEnvelope.create(
                event_type="gl001.telemetry.normalized",
                source="test",
                payload=telemetry,
            )

            # Process first time
            async for _ in processor.process_telemetry(envelope):
                pass

            # Process same envelope again (should be deduplicated)
            async for _ in processor.process_telemetry(envelope):
                pass

            assert processor.metrics.events_deduplicated == 1
        finally:
            await processor.stop()

    def test_get_metrics(self, processor: StreamProcessor) -> None:
        """Test getting processor metrics."""
        metrics = processor.get_metrics()

        assert isinstance(metrics, ProcessorMetrics)
        assert metrics.events_processed == 0

    @pytest.mark.asyncio
    async def test_manual_checkpoint(self, processor: StreamProcessor) -> None:
        """Test manual checkpoint creation."""
        await processor.start()

        try:
            checkpoint_id = await processor.checkpoint()

            assert checkpoint_id.startswith("ckpt-")
            assert processor.metrics.checkpoints_created == 1
        finally:
            await processor.stop()


class TestStreamProcessorBuilder:
    """Tests for stream processor builder."""

    def test_builder_tumbling_window(self) -> None:
        """Test building processor with tumbling window."""
        processor = (
            StreamProcessorBuilder()
            .with_processor_id("test-processor")
            .with_tumbling_window(60, [AggregationType.AVG, AggregationType.MAX])
            .build()
        )

        assert processor.config.processor_id == "test-processor"
        assert len(processor._window_configs) == 1

    def test_builder_sliding_window(self) -> None:
        """Test building processor with sliding window."""
        processor = (
            StreamProcessorBuilder()
            .with_sliding_window(300, 60, [AggregationType.PERCENTILE_95])
            .build()
        )

        assert len(processor._window_configs) == 1
        config = list(processor._window_configs.values())[0]
        assert config.window_type == WindowType.SLIDING
        assert config.slide_seconds == 60

    def test_builder_session_window(self) -> None:
        """Test building processor with session window."""
        processor = (
            StreamProcessorBuilder()
            .with_session_window(30, [AggregationType.COUNT])
            .build()
        )

        assert len(processor._window_configs) == 1
        config = list(processor._window_configs.values())[0]
        assert config.window_type == WindowType.SESSION
        assert config.gap_seconds == 30

    def test_builder_multiple_windows(self) -> None:
        """Test building processor with multiple windows."""
        processor = (
            StreamProcessorBuilder()
            .with_tumbling_window(60, [AggregationType.AVG])
            .with_sliding_window(300, 60, [AggregationType.MAX])
            .with_session_window(30, [AggregationType.COUNT])
            .build()
        )

        assert len(processor._window_configs) == 3

    def test_builder_with_watermark(self) -> None:
        """Test building processor with watermark configuration."""
        processor = (
            StreamProcessorBuilder()
            .with_tumbling_window(60, [AggregationType.AVG])
            .with_watermark(30, WatermarkStrategy.STRICT)
            .build()
        )

        config = list(processor._window_configs.values())[0]
        assert config.watermark_seconds == 30
        assert config.watermark_strategy == WatermarkStrategy.STRICT

    def test_builder_with_deduplication(self) -> None:
        """Test building processor with deduplication configuration."""
        processor = (
            StreamProcessorBuilder()
            .with_tumbling_window(60, [AggregationType.AVG])
            .with_deduplication(enabled=True, window_seconds=600)
            .build()
        )

        assert processor.config.dedup_enabled is True
        assert processor.config.dedup_window_seconds == 600

    def test_builder_with_parallelism(self) -> None:
        """Test building processor with parallelism configuration."""
        processor = (
            StreamProcessorBuilder()
            .with_parallelism(8)
            .with_tumbling_window(60, [AggregationType.AVG])
            .build()
        )

        assert processor.config.parallelism == 8

    def test_builder_with_state_store(self) -> None:
        """Test building processor with custom state store."""
        custom_store = InMemoryStateStore()

        processor = (
            StreamProcessorBuilder()
            .with_tumbling_window(60, [AggregationType.AVG])
            .with_state_store(custom_store)
            .build()
        )

        assert processor.state_store is custom_store

    def test_builder_complete_example(self) -> None:
        """Test complete builder example."""
        processor = (
            StreamProcessorBuilder()
            .with_processor_id("telemetry-aggregator")
            .with_parallelism(4)
            .with_tumbling_window(
                60,
                [AggregationType.AVG, AggregationType.MIN, AggregationType.MAX],
                group_by=["tag_id"],
            )
            .with_sliding_window(
                300,
                60,
                [AggregationType.PERCENTILE_95],
                group_by=["equipment_id"],
            )
            .with_watermark(30, WatermarkStrategy.ALLOW_LATE)
            .with_deduplication(enabled=True, window_seconds=300)
            .build()
        )

        assert processor.config.processor_id == "telemetry-aggregator"
        assert processor.config.parallelism == 4
        assert len(processor._window_configs) == 2
        assert processor.config.dedup_enabled is True
