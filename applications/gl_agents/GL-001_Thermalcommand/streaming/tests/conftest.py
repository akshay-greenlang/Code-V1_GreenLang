"""
Pytest Configuration and Fixtures - GL-001 ThermalCommand Streaming Tests

Provides shared fixtures for streaming module tests.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, AsyncGenerator

from ..event_envelope import EventEnvelope
from ..kafka_schemas import (
    TelemetryNormalizedEvent,
    TelemetryPoint,
    DispatchPlanEvent,
    LoadAllocation,
    ExpectedImpact,
    SafetyEvent,
    SafetyLevel,
    BoundaryViolation,
    AuditLogEvent,
    AuditAction,
    QualityCode,
    UnitOfMeasure,
    SolverStatus,
)
from ..kafka_streaming import (
    KafkaConfig,
    ProducerConfig,
    ConsumerConfig,
    ThermalCommandProducer,
    ThermalCommandConsumer,
)
from ..stream_processor import (
    StreamProcessor,
    StreamProcessorConfig,
    WindowConfig,
    WindowType,
    AggregationType,
    InMemoryStateStore,
)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_telemetry_point() -> TelemetryPoint:
    """Create a sample telemetry point."""
    return TelemetryPoint(
        tag_id="T-101",
        value=450.5,
        unit=UnitOfMeasure.CELSIUS,
        quality=QualityCode.GOOD,
        timestamp=datetime.now(timezone.utc),
        sensor_id="sensor-001",
        equipment_id="boiler-01",
    )


@pytest.fixture
def sample_telemetry_points() -> List[TelemetryPoint]:
    """Create a list of sample telemetry points."""
    now = datetime.now(timezone.utc)
    return [
        TelemetryPoint(
            tag_id=f"T-{100 + i}",
            value=float(400 + i * 10),
            unit=UnitOfMeasure.CELSIUS,
            quality=QualityCode.GOOD,
            timestamp=now + timedelta(seconds=i),
            equipment_id="boiler-01",
        )
        for i in range(10)
    ]


@pytest.fixture
def sample_telemetry_event(
    sample_telemetry_points: List[TelemetryPoint],
) -> TelemetryNormalizedEvent:
    """Create a sample telemetry normalized event."""
    return TelemetryNormalizedEvent(
        source_system="opc-ua-collector-01",
        points=sample_telemetry_points,
        collection_timestamp=datetime.now(timezone.utc),
        batch_id="test-batch-001",
        sequence_number=1,
    )


@pytest.fixture
def sample_dispatch_plan() -> DispatchPlanEvent:
    """Create a sample dispatch plan event."""
    now = datetime.now(timezone.utc)
    return DispatchPlanEvent(
        plan_id="plan-test-001",
        horizon_start=now,
        horizon_end=now + timedelta(hours=24),
        allocations=[
            LoadAllocation(
                equipment_id=f"boiler-{i:02d}",
                load_mw=float(20 + i * 10),
                min_load_mw=10.0,
                max_load_mw=100.0,
                efficiency_percent=90.0 + i,
                emissions_rate_kgco2_mwh=180.0,
                fuel_type="natural_gas",
                ramp_rate_mw_min=5.0,
                marginal_cost_usd_mwh=45.0,
            )
            for i in range(3)
        ],
        solver_status=SolverStatus.OPTIMAL,
        solver_gap_percent=0.01,
        solver_time_seconds=2.5,
        expected_impact=ExpectedImpact(
            total_cost_usd=50000.0,
            total_emissions_tco2=250.0,
            average_efficiency_percent=92.0,
            cost_savings_usd=5000.0,
            emissions_reduction_tco2=25.0,
        ),
        demand_mw=120.0,
        created_by="test-optimizer",
    )


@pytest.fixture
def sample_safety_event() -> SafetyEvent:
    """Create a sample safety event."""
    now = datetime.now(timezone.utc)
    return SafetyEvent(
        event_id="safety-test-001",
        level=SafetyLevel.ALARM,
        event_timestamp=now,
        equipment_id="boiler-01",
        equipment_name="Main Process Boiler 01",
        area_id="area-A",
        boundary_violations=[
            BoundaryViolation(
                tag_id="T-101",
                boundary_type="high",
                limit_value=550.0,
                actual_value=565.0,
                deviation_percent=2.73,
                unit=UnitOfMeasure.CELSIUS,
                duration_seconds=15.5,
            )
        ],
        operator_action_required=True,
        action_deadline=now + timedelta(minutes=5),
        escalation_level=2,
    )


@pytest.fixture
def sample_audit_event() -> AuditLogEvent:
    """Create a sample audit log event."""
    now = datetime.now(timezone.utc)
    return AuditLogEvent(
        audit_id="audit-test-001",
        action=AuditAction.CREATE,
        action_timestamp=now,
        actor_id="test-service",
        actor_type="service",
        resource_type="dispatch_plan",
        resource_id="plan-001",
        correlation_id="corr-test-001",
        outcome="success",
        compliance_tags=["ISO50001", "test"],
    )


@pytest.fixture
def sample_envelope(
    sample_telemetry_event: TelemetryNormalizedEvent,
) -> EventEnvelope:
    """Create a sample event envelope with telemetry."""
    return EventEnvelope.create(
        event_type="gl001.telemetry.normalized",
        source="test-collector",
        payload=sample_telemetry_event,
        correlation_id="corr-test-001",
    )


@pytest.fixture
def kafka_config() -> KafkaConfig:
    """Create a test Kafka configuration."""
    return KafkaConfig(
        bootstrap_servers="localhost:9092",
        client_id="test-client",
    )


@pytest.fixture
def producer_config() -> ProducerConfig:
    """Create a test producer configuration."""
    return ProducerConfig(
        enable_idempotence=True,
        linger_ms=1,
    )


@pytest.fixture
def consumer_config() -> ConsumerConfig:
    """Create a test consumer configuration."""
    return ConsumerConfig(
        group_id="test-consumer-group",
        enable_auto_commit=False,
    )


@pytest.fixture
def stream_processor_config() -> StreamProcessorConfig:
    """Create a test stream processor configuration."""
    return StreamProcessorConfig(
        processor_id="test-processor",
        checkpoint_interval_seconds=5,
        dedup_enabled=True,
        dedup_window_seconds=60,
    )


@pytest.fixture
def tumbling_window_config() -> WindowConfig:
    """Create a tumbling window configuration."""
    return WindowConfig(
        window_id="tumbling-60s",
        window_type=WindowType.TUMBLING,
        window_size_seconds=60,
        aggregations=[
            AggregationType.AVG,
            AggregationType.MIN,
            AggregationType.MAX,
            AggregationType.COUNT,
        ],
    )


@pytest.fixture
def sliding_window_config() -> WindowConfig:
    """Create a sliding window configuration."""
    return WindowConfig(
        window_id="sliding-300s-60s",
        window_type=WindowType.SLIDING,
        window_size_seconds=300,
        slide_seconds=60,
        aggregations=[
            AggregationType.AVG,
            AggregationType.PERCENTILE_95,
        ],
    )


@pytest.fixture
def state_store() -> InMemoryStateStore:
    """Create an in-memory state store."""
    return InMemoryStateStore()


@pytest.fixture
async def stream_processor(
    stream_processor_config: StreamProcessorConfig,
    tumbling_window_config: WindowConfig,
    state_store: InMemoryStateStore,
) -> AsyncGenerator[StreamProcessor, None]:
    """Create and start a stream processor."""
    processor = StreamProcessor(
        config=stream_processor_config,
        state_store=state_store,
    )
    processor.add_window(tumbling_window_config)

    await processor.start()
    yield processor
    await processor.stop()


@pytest.fixture
async def producer(
    kafka_config: KafkaConfig,
    producer_config: ProducerConfig,
) -> AsyncGenerator[ThermalCommandProducer, None]:
    """Create and start a Kafka producer."""
    prod = ThermalCommandProducer(kafka_config, producer_config)
    await prod.start()
    yield prod
    await prod.close()


@pytest.fixture
async def consumer(
    kafka_config: KafkaConfig,
    consumer_config: ConsumerConfig,
) -> AsyncGenerator[ThermalCommandConsumer, None]:
    """Create and start a Kafka consumer."""
    cons = ThermalCommandConsumer(kafka_config, consumer_config)
    await cons.start()
    yield cons
    await cons.close()


def create_telemetry_envelope(
    tag_id: str = "T-101",
    value: float = 450.5,
    source: str = "test",
) -> EventEnvelope:
    """Helper to create telemetry envelopes for testing."""
    now = datetime.now(timezone.utc)
    event = TelemetryNormalizedEvent(
        source_system=source,
        points=[
            TelemetryPoint(
                tag_id=tag_id,
                value=value,
                unit=UnitOfMeasure.CELSIUS,
                timestamp=now,
            )
        ],
        collection_timestamp=now,
        batch_id=f"batch-{tag_id}",
        sequence_number=1,
    )
    return EventEnvelope.create(
        event_type="gl001.telemetry.normalized",
        source=source,
        payload=event,
    )


def create_telemetry_stream(
    count: int = 100,
    tag_id: str = "T-101",
    base_value: float = 400.0,
    value_range: float = 100.0,
) -> List[EventEnvelope]:
    """Helper to create a stream of telemetry envelopes."""
    import random

    envelopes = []
    now = datetime.now(timezone.utc)

    for i in range(count):
        value = base_value + random.uniform(0, value_range)
        timestamp = now + timedelta(seconds=i)

        event = TelemetryNormalizedEvent(
            source_system="test-stream",
            points=[
                TelemetryPoint(
                    tag_id=tag_id,
                    value=value,
                    unit=UnitOfMeasure.CELSIUS,
                    timestamp=timestamp,
                )
            ],
            collection_timestamp=timestamp,
            batch_id=f"stream-batch-{i}",
            sequence_number=i,
        )

        envelope = EventEnvelope.create(
            event_type="gl001.telemetry.normalized",
            source="test-stream",
            payload=event,
        )
        envelopes.append(envelope)

    return envelopes
