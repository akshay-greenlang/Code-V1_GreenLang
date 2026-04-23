"""
GreenLang Framework - Conftest Template

Standard pytest configuration template for GreenLang agents.
Copy and customize this file as conftest.py in your agent's tests/ directory.

Provides:
- Common fixtures for GreenLang agents
- Mock integrations for SCADA/OPC-UA
- Database fixtures
- Async test support
- Performance testing utilities
- Determinism testing fixtures

Target Coverage: 85%+
Author: GreenLang QA Team
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
import numpy as np

# Try to import optional dependencies
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import sqlalchemy
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker, Session
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


# =============================================================================
# Configuration Constants
# =============================================================================

# Reproducibility seed
TEST_SEED = 42

# Timeout settings
DEFAULT_TIMEOUT = 30  # seconds
ASYNC_TIMEOUT = 10    # seconds
PERFORMANCE_TIMEOUT = 60  # seconds

# Performance targets
OPTIMIZATION_CYCLE_TIME_TARGET = 5.0  # seconds
DATA_PROCESSING_RATE_TARGET = 10000  # points per second
MEMORY_USAGE_TARGET = 512  # MB
API_RESPONSE_TIME_TARGET = 0.200  # seconds

# Coverage target
COVERAGE_TARGET = 85.0  # percent


# =============================================================================
# Data Classes for Fixtures
# =============================================================================

@dataclass
class AgentTestConfig:
    """Configuration for agent testing."""
    agent_id: str
    agent_name: str
    version: str = "1.0.0"
    environment: str = "test"
    track_provenance: bool = True
    enable_caching: bool = False
    log_level: str = "DEBUG"
    timeout_seconds: int = DEFAULT_TIMEOUT


@dataclass
class OPCUATagValue:
    """Represents an OPC-UA tag value."""
    node_id: str
    value: Any
    quality: str = "Good"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_timestamp: datetime = None
    server_timestamp: datetime = None

    def __post_init__(self):
        if self.source_timestamp is None:
            self.source_timestamp = self.timestamp
        if self.server_timestamp is None:
            self.server_timestamp = self.timestamp


@dataclass
class SCADAPoint:
    """Represents a SCADA data point."""
    tag_name: str
    value: float
    unit: str
    quality: int = 192  # OPC Quality Good
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    alarm_state: str = "NORMAL"
    low_limit: Optional[float] = None
    high_limit: Optional[float] = None


@dataclass
class MockDatabaseRecord:
    """Represents a database record for testing."""
    id: int
    created_at: datetime
    updated_at: datetime
    data: Dict[str, Any]
    provenance_hash: Optional[str] = None


# =============================================================================
# Seed Management
# =============================================================================

@pytest.fixture(autouse=True)
def set_random_seeds():
    """Ensure reproducible randomness across all tests."""
    random.seed(TEST_SEED)
    np.random.seed(TEST_SEED)
    yield


@pytest.fixture(scope="session")
def test_seed() -> int:
    """Provide the test seed for reproducibility."""
    return TEST_SEED


# =============================================================================
# Agent Configuration Fixtures
# =============================================================================

@pytest.fixture
def agent_config() -> AgentTestConfig:
    """
    Provide default agent test configuration.

    Override in your conftest.py with specific agent values:

        @pytest.fixture
        def agent_config():
            return AgentTestConfig(
                agent_id="GL-001",
                agent_name="THERMALCOMMAND",
                version="1.0.0"
            )
    """
    return AgentTestConfig(
        agent_id="GL-XXX",
        agent_name="TEMPLATE",
        version="1.0.0",
    )


@pytest.fixture
def agent_settings(agent_config: AgentTestConfig) -> Dict[str, Any]:
    """Provide agent settings as dictionary."""
    return {
        "agent_id": agent_config.agent_id,
        "agent_name": agent_config.agent_name,
        "version": agent_config.version,
        "environment": agent_config.environment,
        "track_provenance": agent_config.track_provenance,
        "enable_caching": agent_config.enable_caching,
        "log_level": agent_config.log_level,
    }


# =============================================================================
# Provenance Testing Fixtures
# =============================================================================

@pytest.fixture
def provenance_calculator():
    """Provide provenance hash calculation utilities."""

    class ProvenanceCalculator:
        @staticmethod
        def compute_hash(data: Any) -> str:
            """Compute SHA-256 hash for provenance."""
            if isinstance(data, dict):
                data_str = json.dumps(data, sort_keys=True, default=str)
            else:
                data_str = str(data)
            return hashlib.sha256(data_str.encode()).hexdigest()

        @staticmethod
        def verify_hash(data: Any, expected_hash: str) -> bool:
            """Verify data matches expected hash."""
            actual = ProvenanceCalculator.compute_hash(data)
            return actual == expected_hash

        @staticmethod
        def create_provenance_record(
            inputs: Dict[str, Any],
            outputs: Dict[str, Any],
            agent_id: str,
            version: str,
        ) -> Dict[str, Any]:
            """Create a complete provenance record."""
            inputs_hash = ProvenanceCalculator.compute_hash(inputs)
            outputs_hash = ProvenanceCalculator.compute_hash(outputs)

            combined = {
                "inputs_hash": inputs_hash,
                "outputs_hash": outputs_hash,
                "agent_id": agent_id,
                "version": version,
            }

            return {
                "computation_hash": ProvenanceCalculator.compute_hash(combined),
                "inputs_hash": inputs_hash,
                "outputs_hash": outputs_hash,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent_id": agent_id,
                "version": version,
            }

    return ProvenanceCalculator()


@pytest.fixture
def determinism_checker():
    """Provide determinism checking utilities."""

    class DeterminismChecker:
        def __init__(self):
            self.results = []

        def run_multiple(
            self,
            func: Callable,
            args: tuple = (),
            kwargs: dict = None,
            iterations: int = 10,
        ) -> List[Any]:
            """Run function multiple times and collect results."""
            kwargs = kwargs or {}
            self.results = [func(*args, **kwargs) for _ in range(iterations)]
            return self.results

        def check_identical_results(self) -> Tuple[bool, str]:
            """Check if all results are identical."""
            if not self.results:
                return False, "No results to check"

            first = self.results[0]
            for i, result in enumerate(self.results[1:], 2):
                if result != first:
                    return False, f"Result {i} differs from result 1"

            return True, f"All {len(self.results)} results are identical"

        def check_identical_hashes(
            self,
            hash_func: Callable[[Any], str],
        ) -> Tuple[bool, str]:
            """Check if all result hashes are identical."""
            if not self.results:
                return False, "No results to check"

            hashes = [hash_func(r) for r in self.results]
            first_hash = hashes[0]

            for i, h in enumerate(hashes[1:], 2):
                if h != first_hash:
                    return False, f"Hash {i} ({h[:16]}...) differs from hash 1 ({first_hash[:16]}...)"

            return True, f"All {len(hashes)} hashes are identical: {first_hash[:16]}..."

    return DeterminismChecker()


# =============================================================================
# OPC-UA Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_opcua_client():
    """
    Mock OPC-UA client for integration testing.

    Simulates common OPC-UA operations:
    - Connect/disconnect
    - Read/write nodes
    - Subscriptions
    - Browse operations
    """
    client = MagicMock()

    # Connection methods
    client.connect = MagicMock(return_value=True)
    client.disconnect = MagicMock(return_value=True)
    client.is_connected = MagicMock(return_value=True)

    # Node reading
    def read_node(node_id: str) -> OPCUATagValue:
        """Simulate reading a node value."""
        base_values = {
            "ns=2;s=Temperature": 450.0,
            "ns=2;s=Pressure": 15.0,
            "ns=2;s=FlowRate": 500.0,
            "ns=2;s=Efficiency": 0.85,
        }
        value = base_values.get(node_id, random.uniform(0, 100))
        return OPCUATagValue(
            node_id=node_id,
            value=value + random.gauss(0, value * 0.01),  # 1% noise
            quality="Good",
        )

    client.read_node = MagicMock(side_effect=read_node)
    client.read_nodes = MagicMock(side_effect=lambda ids: [read_node(i) for i in ids])

    # Node writing
    client.write_node = MagicMock(return_value=True)
    client.write_nodes = MagicMock(return_value=[True] * 10)

    # Subscriptions
    subscription = MagicMock()
    subscription.subscription_id = "sub_123"
    subscription.subscribe_data_change = MagicMock(return_value="mon_456")
    subscription.unsubscribe = MagicMock(return_value=True)
    client.create_subscription = MagicMock(return_value=subscription)

    # Browsing
    client.browse = MagicMock(return_value=[
        {"node_id": "ns=2;s=Boiler1", "display_name": "Boiler 1"},
        {"node_id": "ns=2;s=Boiler2", "display_name": "Boiler 2"},
    ])

    # Endpoints
    client.get_endpoints = MagicMock(return_value=[
        {
            "url": "opc.tcp://localhost:4840",
            "security_mode": "SignAndEncrypt",
            "security_policy": "Basic256Sha256",
        }
    ])

    return client


@pytest.fixture
async def async_mock_opcua_client():
    """Async mock OPC-UA client for async integration testing."""
    client = AsyncMock()

    client.connect = AsyncMock(return_value=True)
    client.disconnect = AsyncMock(return_value=True)
    client.is_connected = AsyncMock(return_value=True)

    async def async_read_node(node_id: str) -> OPCUATagValue:
        await asyncio.sleep(0.001)  # Simulate network delay
        return OPCUATagValue(
            node_id=node_id,
            value=random.uniform(0, 100),
            quality="Good",
        )

    client.read_node = AsyncMock(side_effect=async_read_node)
    client.write_node = AsyncMock(return_value=True)

    return client


# =============================================================================
# SCADA Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_scada_interface():
    """
    Mock SCADA interface for testing.

    Provides simulated SCADA functionality:
    - Tag reading/writing
    - Alarm management
    - Historical data access
    """
    interface = MagicMock()

    # Tag database
    tag_db = {
        "TIC-101.PV": SCADAPoint("TIC-101.PV", 450.0, "degC", low_limit=0, high_limit=600),
        "TIC-101.SP": SCADAPoint("TIC-101.SP", 450.0, "degC"),
        "PIC-101.PV": SCADAPoint("PIC-101.PV", 15.0, "bar", low_limit=0, high_limit=50),
        "FIC-101.PV": SCADAPoint("FIC-101.PV", 500.0, "m3/h", low_limit=0, high_limit=1000),
        "AIC-101.PV": SCADAPoint("AIC-101.PV", 0.85, "fraction"),
    }

    def read_tag(tag_name: str) -> SCADAPoint:
        if tag_name in tag_db:
            point = tag_db[tag_name]
            # Add some noise
            point.value = point.value + random.gauss(0, abs(point.value) * 0.005)
            point.timestamp = datetime.now(timezone.utc)
            return point
        raise KeyError(f"Tag not found: {tag_name}")

    def write_tag(tag_name: str, value: float) -> bool:
        if tag_name in tag_db:
            tag_db[tag_name].value = value
            tag_db[tag_name].timestamp = datetime.now(timezone.utc)
            return True
        return False

    def get_historical_data(
        tag_name: str,
        start_time: datetime,
        end_time: datetime,
        interval_seconds: int = 60,
    ) -> List[SCADAPoint]:
        """Generate historical data for a tag."""
        if tag_name not in tag_db:
            return []

        base_point = tag_db[tag_name]
        points = []
        current = start_time

        while current <= end_time:
            noise = random.gauss(0, abs(base_point.value) * 0.02)
            trend = 0.001 * (current - start_time).total_seconds()

            points.append(SCADAPoint(
                tag_name=tag_name,
                value=base_point.value + noise + trend,
                unit=base_point.unit,
                timestamp=current,
            ))
            current += timedelta(seconds=interval_seconds)

        return points

    interface.read_tag = MagicMock(side_effect=read_tag)
    interface.read_tags = MagicMock(side_effect=lambda tags: [read_tag(t) for t in tags])
    interface.write_tag = MagicMock(side_effect=write_tag)
    interface.get_historical_data = MagicMock(side_effect=get_historical_data)

    # Alarm methods
    interface.get_active_alarms = MagicMock(return_value=[])
    interface.acknowledge_alarm = MagicMock(return_value=True)

    return interface


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest.fixture
def mock_database():
    """
    Mock database for testing.

    Provides in-memory database simulation:
    - CRUD operations
    - Query simulation
    - Transaction management
    """
    db = MagicMock()
    records: Dict[int, MockDatabaseRecord] = {}
    next_id = 1

    def insert(data: Dict[str, Any]) -> int:
        nonlocal next_id
        record = MockDatabaseRecord(
            id=next_id,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            data=data,
            provenance_hash=hashlib.sha256(
                json.dumps(data, sort_keys=True).encode()
            ).hexdigest(),
        )
        records[next_id] = record
        next_id += 1
        return record.id

    def get(record_id: int) -> Optional[MockDatabaseRecord]:
        return records.get(record_id)

    def update(record_id: int, data: Dict[str, Any]) -> bool:
        if record_id in records:
            records[record_id].data.update(data)
            records[record_id].updated_at = datetime.now(timezone.utc)
            records[record_id].provenance_hash = hashlib.sha256(
                json.dumps(records[record_id].data, sort_keys=True).encode()
            ).hexdigest()
            return True
        return False

    def delete(record_id: int) -> bool:
        if record_id in records:
            del records[record_id]
            return True
        return False

    def query(filters: Dict[str, Any] = None) -> List[MockDatabaseRecord]:
        if not filters:
            return list(records.values())

        result = []
        for record in records.values():
            match = True
            for key, value in filters.items():
                if record.data.get(key) != value:
                    match = False
                    break
            if match:
                result.append(record)
        return result

    db.insert = MagicMock(side_effect=insert)
    db.get = MagicMock(side_effect=get)
    db.update = MagicMock(side_effect=update)
    db.delete = MagicMock(side_effect=delete)
    db.query = MagicMock(side_effect=query)
    db.execute = MagicMock(return_value=[])
    db.commit = MagicMock()
    db.rollback = MagicMock()

    return db


@pytest.fixture
def sqlite_test_db(tmp_path):
    """
    SQLite test database fixture.

    Creates a temporary SQLite database for integration testing.
    Requires SQLAlchemy to be installed.
    """
    if not SQLALCHEMY_AVAILABLE:
        pytest.skip("SQLAlchemy not installed")

    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")

    # Create test tables
    from sqlalchemy import MetaData, Table, Column, Integer, String, DateTime, JSON

    metadata = MetaData()

    calculations = Table(
        'calculations',
        metadata,
        Column('id', Integer, primary_key=True),
        Column('agent_id', String(10)),
        Column('computation_hash', String(64)),
        Column('inputs_hash', String(64)),
        Column('outputs_hash', String(64)),
        Column('timestamp', DateTime),
        Column('data', JSON),
    )

    metadata.create_all(engine)

    SessionLocal = sessionmaker(bind=engine)

    @contextmanager
    def get_session() -> Generator[Session, None, None]:
        session = SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return {
        "engine": engine,
        "get_session": get_session,
        "metadata": metadata,
    }


# =============================================================================
# Kafka Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_kafka_producer():
    """Mock Kafka producer for streaming integration testing."""
    producer = MagicMock()

    messages_sent = []

    def send(topic: str, value: bytes, key: bytes = None):
        future = MagicMock()
        future.get = MagicMock(return_value=None)
        messages_sent.append({
            "topic": topic,
            "value": value,
            "key": key,
            "timestamp": datetime.now(timezone.utc),
        })
        return future

    producer.send = MagicMock(side_effect=send)
    producer.flush = MagicMock()
    producer.close = MagicMock()
    producer.messages_sent = messages_sent

    return producer


@pytest.fixture
def mock_kafka_consumer():
    """Mock Kafka consumer for streaming integration testing."""
    consumer = MagicMock()

    message_queue = []

    def poll(timeout_ms: int = 1000):
        if message_queue:
            return message_queue.pop(0)
        return None

    consumer.subscribe = MagicMock()
    consumer.poll = MagicMock(side_effect=poll)
    consumer.commit = MagicMock()
    consumer.close = MagicMock()
    consumer.add_message = lambda msg: message_queue.append(msg)

    return consumer


# =============================================================================
# HTTP Client Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_http_client():
    """Mock HTTP client for API integration testing."""
    client = MagicMock()

    def mock_request(method: str, url: str, **kwargs) -> MagicMock:
        response = MagicMock()
        response.status_code = 200
        response.ok = True
        response.headers = {"Content-Type": "application/json"}

        # Simulate different endpoints
        if "health" in url:
            response.json = MagicMock(return_value={"status": "healthy"})
        elif "calculate" in url:
            response.json = MagicMock(return_value={
                "result": 42.0,
                "provenance_hash": "abc123" * 10 + "abc1",
            })
        else:
            response.json = MagicMock(return_value={"message": "OK"})

        return response

    client.get = MagicMock(side_effect=lambda url, **kw: mock_request("GET", url, **kw))
    client.post = MagicMock(side_effect=lambda url, **kw: mock_request("POST", url, **kw))
    client.put = MagicMock(side_effect=lambda url, **kw: mock_request("PUT", url, **kw))
    client.delete = MagicMock(side_effect=lambda url, **kw: mock_request("DELETE", url, **kw))

    return client


@pytest.fixture
async def async_mock_http_client():
    """Async mock HTTP client for async API testing."""
    client = AsyncMock()

    async def mock_request(method: str, url: str, **kwargs):
        response = MagicMock()
        response.status = 200
        response.headers = {"Content-Type": "application/json"}

        async def json():
            return {"status": "ok", "method": method}

        response.json = json
        return response

    client.get = AsyncMock(side_effect=lambda url, **kw: mock_request("GET", url, **kw))
    client.post = AsyncMock(side_effect=lambda url, **kw: mock_request("POST", url, **kw))

    return client


# =============================================================================
# GraphQL and gRPC Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_graphql_client():
    """Mock GraphQL client for testing."""
    client = MagicMock()

    def execute(query: str, variables: Dict = None):
        # Parse query to determine response
        if "thermalStatus" in query:
            return {
                "data": {
                    "thermalStatus": {
                        "temperature": 450.0,
                        "pressure": 15.0,
                        "efficiency": 0.85,
                    }
                }
            }
        elif "optimizationResult" in query:
            return {
                "data": {
                    "optimizationResult": {
                        "objectiveValue": 1000.0,
                        "status": "optimal",
                        "provenanceHash": "abc123" * 10 + "abc1",
                    }
                }
            }
        return {"data": {}}

    client.execute = MagicMock(side_effect=execute)
    client.execute_async = AsyncMock(side_effect=execute)

    return client


@pytest.fixture
def mock_grpc_stub():
    """Mock gRPC stub for testing."""
    stub = MagicMock()

    # Mock response types
    class ThermalDataResponse:
        temperature = 450.0
        pressure = 15.0
        flow_rate = 500.0
        timestamp = datetime.now(timezone.utc).isoformat()
        quality = "Good"

    class SetpointResponse:
        success = True
        message = "Setpoint updated"
        new_value = 0.0

    stub.GetThermalData = MagicMock(return_value=ThermalDataResponse())
    stub.SetSetpoint = MagicMock(return_value=SetpointResponse())
    stub.StreamData = MagicMock(return_value=iter([ThermalDataResponse()] * 10))

    return stub


# =============================================================================
# CMMS Integration Mock
# =============================================================================

@pytest.fixture
def mock_cmms_client():
    """Mock CMMS (Computerized Maintenance Management System) client."""
    client = MagicMock()

    work_orders = []
    next_wo_id = 1

    def create_work_order(data: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal next_wo_id
        wo = {
            "work_order_id": f"WO-{next_wo_id:06d}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "OPEN",
            **data,
        }
        work_orders.append(wo)
        next_wo_id += 1
        return wo

    client.create_work_order = MagicMock(side_effect=create_work_order)
    client.get_work_order = MagicMock(side_effect=lambda wo_id: next(
        (wo for wo in work_orders if wo["work_order_id"] == wo_id), None
    ))
    client.update_work_order = MagicMock(return_value=True)
    client.get_maintenance_schedule = MagicMock(return_value=[
        {
            "asset_id": "BOILER_001",
            "scheduled_date": (datetime.now() + timedelta(days=30)).date().isoformat(),
            "type": "preventive",
            "description": "Annual inspection",
        }
    ])
    client.update_asset_status = MagicMock(return_value=True)
    client.get_asset_history = MagicMock(return_value=[])

    return client


# =============================================================================
# Async Test Support
# =============================================================================

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def async_timeout():
    """Provide async timeout context manager."""

    @contextmanager
    def timeout_context(seconds: float = ASYNC_TIMEOUT):
        async def run_with_timeout(coro):
            return await asyncio.wait_for(coro, timeout=seconds)
        yield run_with_timeout

    return timeout_context


# =============================================================================
# Performance Testing Fixtures
# =============================================================================

@pytest.fixture
def performance_timer():
    """Provide performance timing utilities."""

    class PerformanceTimer:
        def __init__(self):
            self.start_time: Optional[float] = None
            self.end_time: Optional[float] = None
            self.elapsed: Optional[float] = None
            self.measurements: List[float] = []

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.end_time = time.perf_counter()
            self.elapsed = self.end_time - self.start_time
            self.measurements.append(self.elapsed)

        def lap(self) -> float:
            """Record a lap time."""
            if self.start_time is None:
                raise RuntimeError("Timer not started")
            lap_time = time.perf_counter() - self.start_time
            self.measurements.append(lap_time)
            return lap_time

        @property
        def average(self) -> float:
            """Get average of all measurements."""
            return sum(self.measurements) / len(self.measurements) if self.measurements else 0

        @property
        def percentile_95(self) -> float:
            """Get 95th percentile of measurements."""
            if not self.measurements:
                return 0
            sorted_times = sorted(self.measurements)
            idx = int(0.95 * len(sorted_times))
            return sorted_times[min(idx, len(sorted_times) - 1)]

        def assert_within_target(self, target_seconds: float) -> None:
            """Assert elapsed time is within target."""
            if self.elapsed is None:
                raise RuntimeError("Timer not stopped")
            assert self.elapsed <= target_seconds, (
                f"Execution time {self.elapsed:.3f}s exceeds target {target_seconds}s"
            )

    return PerformanceTimer


@pytest.fixture
def memory_tracker():
    """Provide memory usage tracking."""
    import tracemalloc

    class MemoryTracker:
        def __init__(self):
            self.start_size: Optional[int] = None
            self.peak_size: Optional[int] = None
            self.end_size: Optional[int] = None

        def __enter__(self):
            tracemalloc.start()
            self.start_size, _ = tracemalloc.get_traced_memory()
            return self

        def __exit__(self, *args):
            self.end_size, self.peak_size = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        @property
        def memory_used_mb(self) -> float:
            """Memory used in MB."""
            if self.peak_size is None or self.start_size is None:
                return 0
            return (self.peak_size - self.start_size) / (1024 * 1024)

        @property
        def peak_memory_mb(self) -> float:
            """Peak memory in MB."""
            return (self.peak_size or 0) / (1024 * 1024)

        def assert_within_limit(self, limit_mb: float) -> None:
            """Assert memory usage is within limit."""
            assert self.memory_used_mb <= limit_mb, (
                f"Memory usage {self.memory_used_mb:.1f}MB exceeds limit {limit_mb}MB"
            )

    return MemoryTracker


@pytest.fixture
def throughput_measurer():
    """Provide throughput measurement utilities."""

    class ThroughputMeasurer:
        def __init__(self):
            self.item_count = 0
            self.start_time: Optional[float] = None
            self.end_time: Optional[float] = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.end_time = time.perf_counter()

        def add_items(self, count: int):
            """Add processed items to counter."""
            self.item_count += count

        @property
        def items_per_second(self) -> float:
            """Calculate throughput in items/second."""
            if self.start_time is None or self.end_time is None:
                return 0
            duration = self.end_time - self.start_time
            return self.item_count / duration if duration > 0 else 0

        def assert_meets_target(self, target_per_second: float) -> None:
            """Assert throughput meets target."""
            assert self.items_per_second >= target_per_second, (
                f"Throughput {self.items_per_second:.0f}/s below target {target_per_second}/s"
            )

    return ThroughputMeasurer


# =============================================================================
# Test Data Generators
# =============================================================================

@pytest.fixture
def test_data_generator(test_seed):
    """Provide test data generation utilities."""

    class TestDataGenerator:
        def __init__(self, seed: int):
            self.rng = np.random.default_rng(seed)
            random.seed(seed)
            self.base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)

        def generate_time_series(
            self,
            count: int,
            base_value: float,
            noise_std: float = 0.0,
            trend: float = 0.0,
            interval_seconds: int = 60,
        ) -> List[Tuple[datetime, float]]:
            """Generate time series data."""
            series = []
            for i in range(count):
                timestamp = self.base_time + timedelta(seconds=i * interval_seconds)
                value = base_value + trend * i + self.rng.normal(0, noise_std)
                series.append((timestamp, value))
            return series

        def generate_random_inputs(
            self,
            template: Dict[str, Tuple[float, float]],
        ) -> Dict[str, float]:
            """Generate random inputs based on template of (min, max) ranges."""
            return {
                key: float(self.rng.uniform(min_val, max_val))
                for key, (min_val, max_val) in template.items()
            }

    return TestDataGenerator(test_seed)


# =============================================================================
# Boundary Testing Fixtures
# =============================================================================

@pytest.fixture
def boundary_conditions():
    """Provide standard boundary test conditions."""
    return {
        "temperature_kelvin": {
            "min": 0.0,
            "max": 2000.0,
            "below_min": -1.0,
            "above_max": 3000.0,
            "at_min": 0.0,
            "at_max": 2000.0,
        },
        "pressure_pa": {
            "min": 0.0,
            "max": 100e6,
            "below_min": -1.0,
            "above_max": 200e6,
            "at_min": 0.0,
            "at_max": 100e6,
        },
        "efficiency": {
            "min": 0.0,
            "max": 1.0,
            "below_min": -0.1,
            "above_max": 1.5,
            "at_min": 0.0,
            "at_max": 1.0,
        },
        "flow_rate": {
            "min": 0.0,
            "max": 10000.0,
            "below_min": -1.0,
            "above_max": 50000.0,
            "at_min": 0.0,
            "at_max": 10000.0,
        },
    }


# =============================================================================
# SLA Testing Fixtures
# =============================================================================

@pytest.fixture
def sla_requirements():
    """Provide SLA requirements for acceptance testing."""
    return {
        "optimization_cycle_time": OPTIMIZATION_CYCLE_TIME_TARGET,
        "data_processing_rate": DATA_PROCESSING_RATE_TARGET,
        "memory_usage_mb": MEMORY_USAGE_TARGET,
        "api_response_time": API_RESPONSE_TIME_TARGET,
        "calculation_reproducibility": 1.0,  # 100% reproducible
        "data_availability": 0.999,  # 99.9%
        "coverage_target": COVERAGE_TARGET,
    }


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    markers = [
        "unit: Unit tests",
        "integration: Integration tests",
        "performance: Performance tests",
        "safety: Safety tests",
        "determinism: Determinism tests",
        "acceptance: Acceptance tests",
        "slow: Slow-running tests",
        "requires_opcua: Tests requiring OPC-UA connection",
        "requires_kafka: Tests requiring Kafka",
        "requires_database: Tests requiring database connection",
        "requires_network: Tests requiring network access",
    ]

    for marker in markers:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on path."""
    for item in items:
        path_str = str(item.fspath)

        # Auto-add markers based on directory
        if "test_unit" in path_str:
            item.add_marker(pytest.mark.unit)
        elif "test_integration" in path_str:
            item.add_marker(pytest.mark.integration)
        elif "test_performance" in path_str:
            item.add_marker(pytest.mark.performance)
        elif "test_safety" in path_str:
            item.add_marker(pytest.mark.safety)
        elif "test_determinism" in path_str:
            item.add_marker(pytest.mark.determinism)
        elif "test_acceptance" in path_str:
            item.add_marker(pytest.mark.acceptance)


# =============================================================================
# Utility Functions
# =============================================================================

def assert_within_tolerance(
    actual: float,
    expected: float,
    tolerance: float = 0.001,
    relative: bool = True,
) -> None:
    """Assert actual value is within tolerance of expected."""
    if relative:
        if expected == 0:
            diff = abs(actual)
        else:
            diff = abs((actual - expected) / expected)
    else:
        diff = abs(actual - expected)

    assert diff <= tolerance, (
        f"Expected {expected} +/- {tolerance * 100 if relative else tolerance}"
        f"{'%' if relative else ''}, got {actual} (diff={diff})"
    )


def assert_provenance_valid(data: Dict, hash_value: str) -> None:
    """Assert provenance hash is valid for given data."""
    calculated = hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()
    assert calculated == hash_value, f"Hash mismatch: {calculated} != {hash_value}"


def assert_within_sla(
    metric_name: str,
    actual: float,
    target: float,
    unit: str = "",
) -> None:
    """Assert metric meets SLA target."""
    assert actual <= target, (
        f"SLA violation: {metric_name} = {actual}{unit}, target = {target}{unit}"
    )
