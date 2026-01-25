"""
GL-013 PREDICTMAINT Integration Tests.

Comprehensive test suite for all integration connectors:
- Base Connector (connection management, circuit breaker, rate limiting)
- CMMS Connector (work orders, equipment, maintenance history)
- Condition Monitoring Connector (vibration, spectrum, alarms)
- IoT Sensor Connector (MQTT, sensor readings, batches)
- Agent Coordinator (message bus, task distribution, consensus)
- Data Transformers (unit conversion, schema mapping, quality scoring)

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
import json

# Base Connector imports
from .base_connector import (
    ConnectionState,
    CircuitState,
    HealthStatus,
    ConnectorType,
    DataQualityLevel,
    RateLimitStrategy,
    BaseConnectorConfig,
    ConnectionInfo,
    HealthCheckResult,
    DataQualityResult,
    MetricsSnapshot,
    AuditLogEntry,
    ConnectorError,
    ConnectionError,
    AuthenticationError,
    TimeoutError,
    ValidationError,
    CircuitOpenError,
    RetryExhaustedError,
    ConfigurationError,
    DataQualityError,
    RateLimitError,
    LRUCache,
    TokenBucketRateLimiter,
    SlidingWindowRateLimiter,
    CircuitBreaker,
    ConnectionPool,
    MetricsCollector,
    AuditLogger,
    DataValidator,
    BaseConnector,
    with_retry,
    create_connector_config,
)

# CMMS Connector imports
from .cmms_connector import (
    CMSProvider,
    WorkOrderStatus,
    WorkOrderPriority,
    WorkOrderType,
    EquipmentStatus,
    EquipmentCriticality,
    MaintenanceType,
    AuthenticationType,
    OAuth2Config,
    APIKeyConfig,
    BasicAuthConfig,
    CMSSConnectorConfig,
    Equipment,
    WorkOrder,
    MaintenanceHistory,
    SparePart,
    Notification,
    CostRecord,
    WorkOrderCreateRequest,
    WorkOrderUpdateRequest,
    EquipmentQueryParams,
    MaintenanceHistoryQueryParams,
    CMSSConnector,
    create_cmms_connector,
)

# Condition Monitoring Connector imports
from .condition_monitoring_connector import (
    ConditionMonitoringProvider,
    CommunicationProtocol,
    VibrationUnit,
    MeasurementType,
    AlarmSeverity,
    AlarmState,
    MeasurementAxis,
    MachineState,
    TrendDirection,
    OPCUAConfig,
    ModbusConfig,
    ConditionMonitoringConnectorConfig,
    MeasurementPoint,
    VibrationReading,
    SpectrumData,
    WaveformData,
    Alarm,
    TrendData,
    RouteData,
    ConditionMonitoringConnector,
    create_condition_monitoring_connector,
)

# IoT Sensor Connector imports
from .iot_sensor_connector import (
    SensorType,
    SensorProtocol,
    MQTTQoS,
    SensorStatus,
    DataFormat,
    TimeSyncMode,
    MQTTConfig,
    SensorGatewayConfig,
    IoTSensorConnectorConfig,
    IoTSensor,
    SensorReading,
    SensorBatch,
    GatewayStatus,
    IoTSensorConnector,
    create_iot_sensor_connector,
)

# Agent Coordinator imports
from .agent_coordinator import (
    AgentID,
    MessageType,
    MessagePriority,
    TaskStatus,
    ConsensusState,
    RoutingStrategy,
    LoadBalanceStrategy,
    MessageBusConfig,
    AgentCoordinatorConfig,
    AgentMessage,
    AgentResponse,
    AgentStatus,
    DistributedTask,
    ConsensusProposal,
    AgentCoordinator,
    create_agent_coordinator,
)

# Data Transformers imports
from .data_transformers import (
    UnitCategory,
    TimeZoneHandling,
    MissingDataStrategy,
    OutlierMethod,
    DataQualityDimension,
    UnitConverter,
    TimestampNormalizer,
    SchemaMapper,
    DataQualityScorer,
    MissingDataHandler,
    OutlierDetector,
    DataTransformer,
    UnitDefinition,
    FieldMapping,
    QualityMetric,
)


# =============================================================================
# BASE CONNECTOR TESTS
# =============================================================================

class TestLRUCache:
    """Tests for LRU cache implementation."""

    def test_cache_set_and_get(self):
        """Test basic cache set and get operations."""
        cache = LRUCache(max_size=10, ttl_seconds=300)
        cache.set("key1", "value1")

        result = cache.get("key1")
        assert result == "value1"

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = LRUCache(max_size=10, ttl_seconds=300)

        result = cache.get("nonexistent")
        assert result is None

    def test_cache_expiration(self):
        """Test cache TTL expiration."""
        cache = LRUCache(max_size=10, ttl_seconds=0)  # Immediate expiration
        cache.set("key1", "value1")

        # Allow time for expiration
        import time
        time.sleep(0.01)

        result = cache.get("key1")
        assert result is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = LRUCache(max_size=2, ttl_seconds=300)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_cache_clear(self):
        """Test cache clear operation."""
        cache = LRUCache(max_size=10, ttl_seconds=300)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cache_delete(self):
        """Test cache delete operation."""
        cache = LRUCache(max_size=10, ttl_seconds=300)
        cache.set("key1", "value1")
        cache.delete("key1")

        assert cache.get("key1") is None


class TestTokenBucketRateLimiter:
    """Tests for token bucket rate limiter."""

    @pytest.mark.asyncio
    async def test_acquire_within_limit(self):
        """Test acquiring tokens within rate limit."""
        limiter = TokenBucketRateLimiter(
            rate=10.0,
            capacity=10,
            initial_tokens=10
        )

        for _ in range(5):
            result = await limiter.acquire()
            assert result is True

    @pytest.mark.asyncio
    async def test_acquire_exceeds_limit(self):
        """Test behavior when rate limit is exceeded."""
        limiter = TokenBucketRateLimiter(
            rate=1.0,
            capacity=2,
            initial_tokens=2
        )

        # Consume all tokens
        await limiter.acquire()
        await limiter.acquire()

        # Third should fail or block
        result = await limiter.acquire(timeout=0.1)
        assert result is False

    @pytest.mark.asyncio
    async def test_token_refill(self):
        """Test token refill over time."""
        limiter = TokenBucketRateLimiter(
            rate=100.0,  # 100 tokens per second
            capacity=10,
            initial_tokens=0
        )

        # Wait for refill
        await asyncio.sleep(0.1)

        result = await limiter.acquire()
        assert result is True


class TestCircuitBreaker:
    """Tests for circuit breaker implementation."""

    def test_initial_state_closed(self):
        """Test circuit breaker starts in closed state."""
        cb = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=5.0,
            half_open_requests=1
        )

        assert cb.state == CircuitState.CLOSED

    def test_transition_to_open(self):
        """Test circuit opens after failure threshold."""
        cb = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=5.0,
            half_open_requests=1
        )

        # Record failures
        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN

    def test_allow_request_when_closed(self):
        """Test requests allowed when circuit is closed."""
        cb = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=5.0,
            half_open_requests=1
        )

        assert cb.allow_request() is True

    def test_block_request_when_open(self):
        """Test requests blocked when circuit is open."""
        cb = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=0.0,  # No recovery timeout for test
            half_open_requests=1
        )

        # Open the circuit
        for _ in range(3):
            cb.record_failure()

        # Force state to remain open for test
        cb._state = CircuitState.OPEN
        cb._last_failure_time = datetime.now(timezone.utc)

        assert cb.allow_request() is False

    def test_record_success_resets_failures(self):
        """Test success resets failure count."""
        cb = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=5.0,
            half_open_requests=1
        )

        cb.record_failure()
        cb.record_failure()
        cb.record_success()

        assert cb._failure_count == 0
        assert cb.state == CircuitState.CLOSED


class TestConnectionPool:
    """Tests for connection pool implementation."""

    @pytest.mark.asyncio
    async def test_pool_creation(self):
        """Test connection pool initialization."""
        async def create_connection():
            return MagicMock()

        pool = ConnectionPool(
            create_connection=create_connection,
            min_size=2,
            max_size=10,
            health_check_interval=30.0
        )

        await pool.initialize()

        assert pool.size >= pool._min_size

        await pool.close()

    @pytest.mark.asyncio
    async def test_acquire_and_release(self):
        """Test acquiring and releasing connections."""
        connection_mock = MagicMock()

        async def create_connection():
            return connection_mock

        pool = ConnectionPool(
            create_connection=create_connection,
            min_size=1,
            max_size=5,
            health_check_interval=30.0
        )

        await pool.initialize()

        conn = await pool.acquire()
        assert conn is not None

        await pool.release(conn)

        await pool.close()


class TestMetricsCollector:
    """Tests for metrics collector."""

    def test_increment_counter(self):
        """Test counter increment."""
        collector = MetricsCollector(connector_name="test")

        collector.increment("requests_total")
        collector.increment("requests_total")
        collector.increment("requests_total", 3)

        metrics = collector.get_snapshot()
        assert metrics.counters.get("requests_total") == 5

    def test_observe_histogram(self):
        """Test histogram observation."""
        collector = MetricsCollector(connector_name="test")

        collector.observe("request_duration", 0.1)
        collector.observe("request_duration", 0.2)
        collector.observe("request_duration", 0.3)

        metrics = collector.get_snapshot()
        histogram = metrics.histograms.get("request_duration")

        assert histogram is not None
        assert histogram["count"] == 3
        assert histogram["sum"] == pytest.approx(0.6, rel=0.01)

    def test_set_gauge(self):
        """Test gauge setting."""
        collector = MetricsCollector(connector_name="test")

        collector.set_gauge("active_connections", 5)
        collector.set_gauge("active_connections", 10)

        metrics = collector.get_snapshot()
        assert metrics.gauges.get("active_connections") == 10


class TestDataValidator:
    """Tests for data validator."""

    def test_validate_required_fields_success(self):
        """Test validation passes with all required fields."""
        validator = DataValidator()

        data = {"name": "Test", "value": 123}
        required = ["name", "value"]

        result = validator.validate_required_fields(data, required)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_required_fields_missing(self):
        """Test validation fails with missing fields."""
        validator = DataValidator()

        data = {"name": "Test"}
        required = ["name", "value", "status"]

        result = validator.validate_required_fields(data, required)
        assert result.is_valid is False
        assert "value" in str(result.errors)
        assert "status" in str(result.errors)

    def test_validate_data_types(self):
        """Test data type validation."""
        validator = DataValidator()

        data = {"name": "Test", "count": 42, "active": True}
        type_map = {"name": str, "count": int, "active": bool}

        result = validator.validate_data_types(data, type_map)
        assert result.is_valid is True

    def test_validate_data_types_mismatch(self):
        """Test data type validation with mismatched types."""
        validator = DataValidator()

        data = {"name": "Test", "count": "not_a_number"}
        type_map = {"name": str, "count": int}

        result = validator.validate_data_types(data, type_map)
        assert result.is_valid is False


class TestBaseConnectorConfig:
    """Tests for base connector configuration."""

    def test_valid_config(self):
        """Test valid configuration creation."""
        config = BaseConnectorConfig(
            connector_id="test-connector-001",
            connector_type=ConnectorType.API,
            name="Test Connector",
            host="api.example.com",
            port=443,
            use_ssl=True,
            timeout_seconds=30.0,
            max_retries=3
        )

        assert config.connector_id == "test-connector-001"
        assert config.port == 443
        assert config.use_ssl is True

    def test_config_validation_port_range(self):
        """Test port range validation."""
        with pytest.raises(ValueError):
            BaseConnectorConfig(
                connector_id="test",
                connector_type=ConnectorType.API,
                name="Test",
                host="localhost",
                port=70000  # Invalid port
            )

    def test_config_validation_timeout(self):
        """Test timeout validation."""
        with pytest.raises(ValueError):
            BaseConnectorConfig(
                connector_id="test",
                connector_type=ConnectorType.API,
                name="Test",
                host="localhost",
                timeout_seconds=-1  # Invalid timeout
            )


# =============================================================================
# CMMS CONNECTOR TESTS
# =============================================================================

class TestCMSSConnectorConfig:
    """Tests for CMMS connector configuration."""

    def test_valid_oauth2_config(self):
        """Test valid OAuth2 configuration."""
        oauth_config = OAuth2Config(
            client_id="client123",
            client_secret="secret456",
            token_url="https://auth.example.com/oauth/token",
            scope="read write"
        )

        config = CMSSConnectorConfig(
            connector_id="cmms-001",
            connector_type=ConnectorType.API,
            name="SAP PM Connector",
            host="sap.example.com",
            port=443,
            provider=CMSProvider.SAP_PM,
            auth_type=AuthenticationType.OAUTH2,
            oauth2_config=oauth_config,
            api_version="v1"
        )

        assert config.provider == CMSProvider.SAP_PM
        assert config.auth_type == AuthenticationType.OAUTH2

    def test_valid_api_key_config(self):
        """Test valid API key configuration."""
        api_key_config = APIKeyConfig(
            api_key="test-api-key-12345",
            header_name="X-API-Key"
        )

        config = CMSSConnectorConfig(
            connector_id="cmms-002",
            connector_type=ConnectorType.API,
            name="Maximo Connector",
            host="maximo.example.com",
            provider=CMSProvider.IBM_MAXIMO,
            auth_type=AuthenticationType.API_KEY,
            api_key_config=api_key_config
        )

        assert config.provider == CMSProvider.IBM_MAXIMO
        assert config.api_key_config.api_key == "test-api-key-12345"


class TestWorkOrderModels:
    """Tests for work order data models."""

    def test_work_order_creation(self):
        """Test work order model creation."""
        work_order = WorkOrder(
            work_order_id="WO-001",
            title="Pump Maintenance",
            description="Quarterly maintenance for cooling pump",
            equipment_id="EQ-PUMP-001",
            status=WorkOrderStatus.OPEN,
            priority=WorkOrderPriority.HIGH,
            work_order_type=WorkOrderType.PREVENTIVE,
            created_at=datetime.now(timezone.utc),
            scheduled_start=datetime.now(timezone.utc) + timedelta(days=1)
        )

        assert work_order.work_order_id == "WO-001"
        assert work_order.status == WorkOrderStatus.OPEN
        assert work_order.priority == WorkOrderPriority.HIGH

    def test_work_order_create_request(self):
        """Test work order create request validation."""
        request = WorkOrderCreateRequest(
            title="Motor Inspection",
            description="Annual motor inspection",
            equipment_id="EQ-MOTOR-001",
            priority=WorkOrderPriority.MEDIUM,
            work_order_type=WorkOrderType.INSPECTION,
            scheduled_start=datetime.now(timezone.utc) + timedelta(hours=4)
        )

        assert request.title == "Motor Inspection"
        assert request.work_order_type == WorkOrderType.INSPECTION


class TestEquipmentModels:
    """Tests for equipment data models."""

    def test_equipment_creation(self):
        """Test equipment model creation."""
        equipment = Equipment(
            equipment_id="EQ-001",
            name="Main Cooling Pump",
            description="Primary cooling system pump",
            equipment_type="Centrifugal Pump",
            manufacturer="Grundfos",
            model_number="CR-32-4",
            serial_number="SN-12345",
            status=EquipmentStatus.OPERATIONAL,
            criticality=EquipmentCriticality.CRITICAL,
            location="Building A - Level 1",
            installation_date=datetime(2020, 1, 15, tzinfo=timezone.utc)
        )

        assert equipment.equipment_id == "EQ-001"
        assert equipment.status == EquipmentStatus.OPERATIONAL
        assert equipment.criticality == EquipmentCriticality.CRITICAL


class TestCMSSConnector:
    """Tests for CMMS connector."""

    @pytest.fixture
    def cmms_config(self):
        """Create CMMS connector configuration fixture."""
        return CMSSConnectorConfig(
            connector_id="cmms-test",
            connector_type=ConnectorType.API,
            name="Test CMMS",
            host="cmms.test.com",
            port=443,
            provider=CMSProvider.SAP_PM,
            auth_type=AuthenticationType.API_KEY,
            api_key_config=APIKeyConfig(
                api_key="test-key",
                header_name="X-API-Key"
            )
        )

    @pytest.mark.asyncio
    async def test_connector_initialization(self, cmms_config):
        """Test CMMS connector initialization."""
        connector = CMSSConnector(cmms_config)

        assert connector.config.provider == CMSProvider.SAP_PM
        assert connector._connection_state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_connector_context_manager(self, cmms_config):
        """Test CMMS connector as context manager."""
        connector = CMSSConnector(cmms_config)

        with patch.object(connector, 'connect', new_callable=AsyncMock):
            with patch.object(connector, 'disconnect', new_callable=AsyncMock):
                async with connector:
                    pass

                connector.connect.assert_called_once()
                connector.disconnect.assert_called_once()


# =============================================================================
# CONDITION MONITORING CONNECTOR TESTS
# =============================================================================

class TestConditionMonitoringConfig:
    """Tests for condition monitoring connector configuration."""

    def test_opcua_config(self):
        """Test OPC-UA configuration."""
        opcua_config = OPCUAConfig(
            endpoint_url="opc.tcp://localhost:4840",
            security_policy="Basic256Sha256",
            security_mode="SignAndEncrypt",
            certificate_path="/path/to/cert.pem",
            private_key_path="/path/to/key.pem"
        )

        config = ConditionMonitoringConnectorConfig(
            connector_id="cm-001",
            connector_type=ConnectorType.PROTOCOL,
            name="SKF Connector",
            host="localhost",
            port=4840,
            provider=ConditionMonitoringProvider.SKF_APTITUDE,
            protocol=CommunicationProtocol.OPC_UA,
            opcua_config=opcua_config
        )

        assert config.provider == ConditionMonitoringProvider.SKF_APTITUDE
        assert config.protocol == CommunicationProtocol.OPC_UA

    def test_modbus_config(self):
        """Test Modbus configuration."""
        modbus_config = ModbusConfig(
            slave_id=1,
            baudrate=9600,
            parity="N",
            stopbits=1,
            bytesize=8
        )

        config = ConditionMonitoringConnectorConfig(
            connector_id="cm-002",
            connector_type=ConnectorType.PROTOCOL,
            name="Emerson Connector",
            host="192.168.1.100",
            port=502,
            provider=ConditionMonitoringProvider.EMERSON_AMS,
            protocol=CommunicationProtocol.MODBUS_TCP,
            modbus_config=modbus_config
        )

        assert config.provider == ConditionMonitoringProvider.EMERSON_AMS
        assert config.protocol == CommunicationProtocol.MODBUS_TCP


class TestVibrationModels:
    """Tests for vibration data models."""

    def test_vibration_reading_creation(self):
        """Test vibration reading model creation."""
        reading = VibrationReading(
            reading_id="VIB-001",
            measurement_point_id="MP-001",
            timestamp=datetime.now(timezone.utc),
            overall_velocity=2.5,
            overall_acceleration=0.8,
            overall_displacement=0.05,
            velocity_unit=VibrationUnit.MM_S,
            acceleration_unit=VibrationUnit.G,
            displacement_unit=VibrationUnit.MICRON,
            axis=MeasurementAxis.HORIZONTAL,
            machine_state=MachineState.RUNNING,
            rpm=1750.0
        )

        assert reading.overall_velocity == 2.5
        assert reading.velocity_unit == VibrationUnit.MM_S
        assert reading.axis == MeasurementAxis.HORIZONTAL

    def test_spectrum_data_creation(self):
        """Test spectrum data model creation."""
        spectrum = SpectrumData(
            spectrum_id="SPEC-001",
            reading_id="VIB-001",
            measurement_point_id="MP-001",
            timestamp=datetime.now(timezone.utc),
            frequency_values=[10.0, 20.0, 30.0, 40.0, 50.0],
            amplitude_values=[0.1, 0.5, 0.3, 0.2, 0.1],
            frequency_unit="Hz",
            amplitude_unit=VibrationUnit.MM_S,
            lines=400,
            fmax=1000.0,
            resolution=2.5
        )

        assert len(spectrum.frequency_values) == 5
        assert spectrum.fmax == 1000.0


class TestAlarmModels:
    """Tests for alarm data models."""

    def test_alarm_creation(self):
        """Test alarm model creation."""
        alarm = Alarm(
            alarm_id="ALM-001",
            measurement_point_id="MP-001",
            equipment_id="EQ-001",
            timestamp=datetime.now(timezone.utc),
            severity=AlarmSeverity.WARNING,
            state=AlarmState.ACTIVE,
            alarm_type="High Vibration",
            message="Vibration exceeded warning threshold",
            threshold_value=5.0,
            actual_value=6.2,
            unit=VibrationUnit.MM_S
        )

        assert alarm.severity == AlarmSeverity.WARNING
        assert alarm.state == AlarmState.ACTIVE
        assert alarm.actual_value > alarm.threshold_value


# =============================================================================
# IOT SENSOR CONNECTOR TESTS
# =============================================================================

class TestIoTSensorConfig:
    """Tests for IoT sensor connector configuration."""

    def test_mqtt_config(self):
        """Test MQTT configuration."""
        mqtt_config = MQTTConfig(
            broker_host="mqtt.example.com",
            broker_port=8883,
            use_tls=True,
            username="sensor_client",
            password="secret123",
            client_id="iot-sensor-001",
            qos=MQTTQoS.AT_LEAST_ONCE,
            keep_alive=60
        )

        config = IoTSensorConnectorConfig(
            connector_id="iot-001",
            connector_type=ConnectorType.MESSAGE_QUEUE,
            name="IoT Sensor Hub",
            host="mqtt.example.com",
            port=8883,
            protocol=SensorProtocol.MQTT,
            mqtt_config=mqtt_config,
            data_format=DataFormat.JSON
        )

        assert config.protocol == SensorProtocol.MQTT
        assert config.mqtt_config.use_tls is True


class TestSensorModels:
    """Tests for sensor data models."""

    def test_iot_sensor_creation(self):
        """Test IoT sensor model creation."""
        sensor = IoTSensor(
            sensor_id="SENS-001",
            name="Temperature Sensor 1",
            sensor_type=SensorType.TEMPERATURE,
            equipment_id="EQ-MOTOR-001",
            location="Motor Housing - Top",
            manufacturer="Sensirion",
            model="STS40",
            firmware_version="2.1.0",
            status=SensorStatus.ONLINE,
            last_seen=datetime.now(timezone.utc),
            calibration_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
            battery_level=85.0
        )

        assert sensor.sensor_type == SensorType.TEMPERATURE
        assert sensor.status == SensorStatus.ONLINE
        assert sensor.battery_level == 85.0

    def test_sensor_reading_creation(self):
        """Test sensor reading model creation."""
        reading = SensorReading(
            reading_id="READ-001",
            sensor_id="SENS-001",
            timestamp=datetime.now(timezone.utc),
            value=72.5,
            unit="celsius",
            quality=0.98,
            metadata={"raw_value": 725, "scaling_factor": 0.1}
        )

        assert reading.value == 72.5
        assert reading.quality == 0.98

    def test_sensor_batch_creation(self):
        """Test sensor batch model creation."""
        readings = [
            SensorReading(
                reading_id=f"READ-{i:03d}",
                sensor_id="SENS-001",
                timestamp=datetime.now(timezone.utc) + timedelta(seconds=i),
                value=70.0 + i * 0.5,
                unit="celsius",
                quality=0.95
            )
            for i in range(10)
        ]

        batch = SensorBatch(
            batch_id="BATCH-001",
            sensor_id="SENS-001",
            readings=readings,
            start_time=readings[0].timestamp,
            end_time=readings[-1].timestamp,
            record_count=len(readings)
        )

        assert batch.record_count == 10
        assert len(batch.readings) == 10


# =============================================================================
# AGENT COORDINATOR TESTS
# =============================================================================

class TestAgentCoordinatorConfig:
    """Tests for agent coordinator configuration."""

    def test_message_bus_config(self):
        """Test message bus configuration."""
        bus_config = MessageBusConfig(
            broker_url="redis://localhost:6379",
            channel_prefix="greenlang.agents",
            message_ttl_seconds=3600,
            max_message_size=1048576
        )

        config = AgentCoordinatorConfig(
            connector_id="coord-001",
            connector_type=ConnectorType.MESSAGE_QUEUE,
            name="Agent Coordinator",
            host="localhost",
            port=6379,
            agent_id=AgentID.GL_013,
            message_bus=bus_config,
            routing_strategy=RoutingStrategy.TOPIC_BASED,
            load_balance_strategy=LoadBalanceStrategy.ROUND_ROBIN
        )

        assert config.agent_id == AgentID.GL_013
        assert config.routing_strategy == RoutingStrategy.TOPIC_BASED


class TestAgentMessageModels:
    """Tests for agent message models."""

    def test_agent_message_creation(self):
        """Test agent message model creation."""
        message = AgentMessage(
            message_id="MSG-001",
            source_agent=AgentID.GL_013,
            target_agent=AgentID.GL_001,
            message_type=MessageType.REQUEST,
            priority=MessagePriority.HIGH,
            payload={"action": "get_emissions", "equipment_id": "EQ-001"},
            timestamp=datetime.now(timezone.utc),
            correlation_id="CORR-001",
            reply_to="greenlang.agents.gl013.responses"
        )

        assert message.source_agent == AgentID.GL_013
        assert message.target_agent == AgentID.GL_001
        assert message.priority == MessagePriority.HIGH

    def test_agent_response_creation(self):
        """Test agent response model creation."""
        response = AgentResponse(
            response_id="RESP-001",
            message_id="MSG-001",
            source_agent=AgentID.GL_001,
            target_agent=AgentID.GL_013,
            status=TaskStatus.COMPLETED,
            payload={"emissions": 125.5, "unit": "kg_co2e"},
            timestamp=datetime.now(timezone.utc),
            processing_time_ms=45.2
        )

        assert response.status == TaskStatus.COMPLETED
        assert response.processing_time_ms == 45.2


class TestDistributedTask:
    """Tests for distributed task model."""

    def test_distributed_task_creation(self):
        """Test distributed task model creation."""
        task = DistributedTask(
            task_id="TASK-001",
            name="Calculate Total Emissions",
            description="Aggregate emissions from all equipment",
            source_agent=AgentID.GL_013,
            assigned_agents=[AgentID.GL_001, AgentID.GL_002, AgentID.GL_003],
            status=TaskStatus.IN_PROGRESS,
            priority=MessagePriority.HIGH,
            payload={"equipment_ids": ["EQ-001", "EQ-002", "EQ-003"]},
            created_at=datetime.now(timezone.utc),
            deadline=datetime.now(timezone.utc) + timedelta(hours=1),
            progress=33.3
        )

        assert len(task.assigned_agents) == 3
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.progress == 33.3


class TestConsensusProposal:
    """Tests for consensus proposal model."""

    def test_consensus_proposal_creation(self):
        """Test consensus proposal model creation."""
        proposal = ConsensusProposal(
            proposal_id="PROP-001",
            proposer=AgentID.GL_013,
            participants=[AgentID.GL_001, AgentID.GL_002, AgentID.GL_003],
            proposal_type="maintenance_schedule",
            proposal_data={"equipment_id": "EQ-001", "action": "shutdown"},
            state=ConsensusState.VOTING,
            votes_for=[AgentID.GL_001],
            votes_against=[],
            abstentions=[],
            quorum_required=0.66,
            created_at=datetime.now(timezone.utc),
            voting_deadline=datetime.now(timezone.utc) + timedelta(minutes=5)
        )

        assert proposal.state == ConsensusState.VOTING
        assert proposal.quorum_required == 0.66
        assert len(proposal.votes_for) == 1


# =============================================================================
# DATA TRANSFORMERS TESTS
# =============================================================================

class TestUnitConverter:
    """Tests for unit converter."""

    def test_temperature_celsius_to_fahrenheit(self):
        """Test Celsius to Fahrenheit conversion."""
        converter = UnitConverter()

        result = converter.convert(100.0, "celsius", "fahrenheit")
        assert result == pytest.approx(212.0, rel=0.01)

    def test_temperature_fahrenheit_to_celsius(self):
        """Test Fahrenheit to Celsius conversion."""
        converter = UnitConverter()

        result = converter.convert(32.0, "fahrenheit", "celsius")
        assert result == pytest.approx(0.0, rel=0.01)

    def test_length_meters_to_feet(self):
        """Test meters to feet conversion."""
        converter = UnitConverter()

        result = converter.convert(1.0, "meter", "foot")
        assert result == pytest.approx(3.28084, rel=0.01)

    def test_pressure_bar_to_psi(self):
        """Test bar to PSI conversion."""
        converter = UnitConverter()

        result = converter.convert(1.0, "bar", "psi")
        assert result == pytest.approx(14.5038, rel=0.01)

    def test_velocity_mm_s_to_in_s(self):
        """Test mm/s to in/s conversion."""
        converter = UnitConverter()

        result = converter.convert(25.4, "mm_s", "in_s")
        assert result == pytest.approx(1.0, rel=0.01)

    def test_same_unit_conversion(self):
        """Test conversion between same units."""
        converter = UnitConverter()

        result = converter.convert(100.0, "celsius", "celsius")
        assert result == 100.0

    def test_invalid_conversion(self):
        """Test invalid unit conversion raises error."""
        converter = UnitConverter()

        with pytest.raises(ValueError):
            converter.convert(100.0, "invalid_unit", "celsius")


class TestTimestampNormalizer:
    """Tests for timestamp normalizer."""

    def test_iso_format_normalization(self):
        """Test ISO format timestamp normalization."""
        normalizer = TimestampNormalizer()

        result = normalizer.normalize("2024-06-15T10:30:00Z")

        assert result.year == 2024
        assert result.month == 6
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30
        assert result.tzinfo == timezone.utc

    def test_unix_timestamp_normalization(self):
        """Test Unix timestamp normalization."""
        normalizer = TimestampNormalizer()

        # Unix timestamp for 2024-01-01 00:00:00 UTC
        result = normalizer.normalize(1704067200)

        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1

    def test_custom_format_normalization(self):
        """Test custom format timestamp normalization."""
        normalizer = TimestampNormalizer()

        result = normalizer.normalize("15/06/2024 10:30:00", format="%d/%m/%Y %H:%M:%S")

        assert result.year == 2024
        assert result.month == 6
        assert result.day == 15

    def test_timezone_conversion(self):
        """Test timezone conversion."""
        normalizer = TimestampNormalizer(target_timezone="UTC")

        # Create datetime with offset
        dt = datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone(timedelta(hours=5)))
        result = normalizer.to_utc(dt)

        assert result.hour == 5  # 10:00 +05:00 = 05:00 UTC
        assert result.tzinfo == timezone.utc


class TestSchemaMapper:
    """Tests for schema mapper."""

    def test_simple_field_mapping(self):
        """Test simple field mapping."""
        mapper = SchemaMapper()

        mappings = [
            FieldMapping(source_field="temp", target_field="temperature"),
            FieldMapping(source_field="press", target_field="pressure"),
            FieldMapping(source_field="ts", target_field="timestamp")
        ]

        source_data = {
            "temp": 72.5,
            "press": 14.7,
            "ts": "2024-06-15T10:00:00Z"
        }

        result = mapper.map(source_data, mappings)

        assert result["temperature"] == 72.5
        assert result["pressure"] == 14.7
        assert result["timestamp"] == "2024-06-15T10:00:00Z"

    def test_field_mapping_with_transformer(self):
        """Test field mapping with value transformer."""
        mapper = SchemaMapper()

        mappings = [
            FieldMapping(
                source_field="temp_f",
                target_field="temperature_c",
                transformer=lambda x: (x - 32) * 5 / 9
            ),
        ]

        source_data = {"temp_f": 212.0}

        result = mapper.map(source_data, mappings)

        assert result["temperature_c"] == pytest.approx(100.0, rel=0.01)

    def test_nested_field_mapping(self):
        """Test nested field mapping."""
        mapper = SchemaMapper()

        mappings = [
            FieldMapping(source_field="sensor.reading.value", target_field="value"),
            FieldMapping(source_field="sensor.metadata.unit", target_field="unit")
        ]

        source_data = {
            "sensor": {
                "reading": {"value": 25.5},
                "metadata": {"unit": "celsius"}
            }
        }

        result = mapper.map(source_data, mappings)

        assert result["value"] == 25.5
        assert result["unit"] == "celsius"


class TestDataQualityScorer:
    """Tests for data quality scorer."""

    def test_completeness_scoring(self):
        """Test completeness scoring."""
        scorer = DataQualityScorer(required_fields=["name", "value", "timestamp"])

        # Complete record
        complete_record = {"name": "Test", "value": 100, "timestamp": "2024-06-15"}
        score = scorer.score_completeness(complete_record)
        assert score == 100.0

        # Partial record
        partial_record = {"name": "Test", "value": None}
        score = scorer.score_completeness(partial_record)
        assert score < 100.0

    def test_validity_scoring(self):
        """Test validity scoring."""
        scorer = DataQualityScorer(
            required_fields=["value"],
            validation_rules={
                "value": lambda x: isinstance(x, (int, float)) and 0 <= x <= 100
            }
        )

        valid_record = {"value": 50}
        score = scorer.score_validity(valid_record)
        assert score == 100.0

        invalid_record = {"value": 150}
        score = scorer.score_validity(invalid_record)
        assert score < 100.0

    def test_overall_quality_score(self):
        """Test overall quality score calculation."""
        scorer = DataQualityScorer(
            required_fields=["name", "value"],
            weights={
                DataQualityDimension.COMPLETENESS: 0.4,
                DataQualityDimension.VALIDITY: 0.3,
                DataQualityDimension.CONSISTENCY: 0.2,
                DataQualityDimension.UNIQUENESS: 0.1
            }
        )

        records = [
            {"name": "A", "value": 10},
            {"name": "B", "value": 20},
            {"name": "C", "value": 30}
        ]

        score = scorer.score_batch(records)

        assert 0 <= score <= 100


class TestMissingDataHandler:
    """Tests for missing data handler."""

    def test_drop_strategy(self):
        """Test drop missing records strategy."""
        handler = MissingDataHandler(strategy=MissingDataStrategy.DROP)

        records = [
            {"name": "A", "value": 10},
            {"name": "B", "value": None},
            {"name": "C", "value": 30}
        ]

        result = handler.handle(records, required_fields=["name", "value"])

        assert len(result) == 2
        assert all(r["value"] is not None for r in result)

    def test_fill_zero_strategy(self):
        """Test fill with zero strategy."""
        handler = MissingDataHandler(strategy=MissingDataStrategy.FILL_ZERO)

        records = [
            {"name": "A", "value": None},
        ]

        result = handler.handle(records, required_fields=["value"])

        assert result[0]["value"] == 0

    def test_fill_mean_strategy(self):
        """Test fill with mean strategy."""
        handler = MissingDataHandler(strategy=MissingDataStrategy.FILL_MEAN)

        records = [
            {"value": 10},
            {"value": 20},
            {"value": None},
            {"value": 30}
        ]

        result = handler.handle(records, required_fields=["value"])

        # Mean of [10, 20, 30] = 20
        assert result[2]["value"] == 20

    def test_forward_fill_strategy(self):
        """Test forward fill strategy."""
        handler = MissingDataHandler(strategy=MissingDataStrategy.FORWARD_FILL)

        records = [
            {"value": 10},
            {"value": None},
            {"value": None},
            {"value": 30}
        ]

        result = handler.handle(records, required_fields=["value"])

        assert result[1]["value"] == 10
        assert result[2]["value"] == 10


class TestOutlierDetector:
    """Tests for outlier detector."""

    def test_zscore_detection(self):
        """Test Z-score outlier detection."""
        detector = OutlierDetector(method=OutlierMethod.ZSCORE, threshold=2.0)

        # Normal values with one outlier
        values = [10, 11, 12, 10, 11, 100, 12, 11, 10]

        outliers = detector.detect(values)

        assert 100 in [values[i] for i in outliers]

    def test_iqr_detection(self):
        """Test IQR outlier detection."""
        detector = OutlierDetector(method=OutlierMethod.IQR, threshold=1.5)

        values = [10, 11, 12, 10, 11, 100, 12, 11, 10]

        outliers = detector.detect(values)

        assert len(outliers) > 0

    def test_modified_zscore_detection(self):
        """Test Modified Z-score outlier detection."""
        detector = OutlierDetector(method=OutlierMethod.MODIFIED_ZSCORE, threshold=3.5)

        values = [10, 11, 12, 10, 11, 100, 12, 11, 10]

        outliers = detector.detect(values)

        assert len(outliers) > 0

    def test_no_outliers(self):
        """Test detection with no outliers."""
        detector = OutlierDetector(method=OutlierMethod.ZSCORE, threshold=3.0)

        # Normal distribution-like values
        values = [10, 11, 12, 11, 10, 11, 12, 10, 11]

        outliers = detector.detect(values)

        assert len(outliers) == 0


class TestDataTransformer:
    """Tests for composite data transformer."""

    def test_full_transformation_pipeline(self):
        """Test complete transformation pipeline."""
        transformer = DataTransformer(
            unit_converter=UnitConverter(),
            timestamp_normalizer=TimestampNormalizer(target_timezone="UTC"),
            schema_mapper=SchemaMapper(),
            quality_scorer=DataQualityScorer(required_fields=["temperature", "timestamp"]),
            missing_data_handler=MissingDataHandler(strategy=MissingDataStrategy.DROP),
            outlier_detector=OutlierDetector(method=OutlierMethod.ZSCORE, threshold=3.0)
        )

        # Source data with different formats
        source_records = [
            {"temp_f": 212.0, "ts": "2024-06-15T10:00:00Z"},
            {"temp_f": 68.0, "ts": "2024-06-15T11:00:00Z"},
            {"temp_f": 72.0, "ts": "2024-06-15T12:00:00Z"}
        ]

        field_mappings = [
            FieldMapping(
                source_field="temp_f",
                target_field="temperature",
                transformer=lambda x: (x - 32) * 5 / 9  # F to C
            ),
            FieldMapping(source_field="ts", target_field="timestamp")
        ]

        result = transformer.transform(source_records, field_mappings)

        assert len(result.records) == 3
        assert result.records[0]["temperature"] == pytest.approx(100.0, rel=0.01)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestConnectorIntegration:
    """Integration tests for connector interactions."""

    @pytest.mark.asyncio
    async def test_cmms_to_coordinator_flow(self):
        """Test data flow from CMMS to agent coordinator."""
        # Mock CMMS connector
        cmms_config = CMSSConnectorConfig(
            connector_id="cmms-int-test",
            connector_type=ConnectorType.API,
            name="Test CMMS",
            host="cmms.test.com",
            provider=CMSProvider.SAP_PM,
            auth_type=AuthenticationType.API_KEY,
            api_key_config=APIKeyConfig(api_key="test", header_name="X-API-Key")
        )
        cmms = CMSSConnector(cmms_config)

        # Mock equipment data
        equipment = Equipment(
            equipment_id="EQ-001",
            name="Test Pump",
            description="Test equipment",
            equipment_type="Pump",
            status=EquipmentStatus.OPERATIONAL,
            criticality=EquipmentCriticality.HIGH
        )

        # Create message for coordinator
        message = AgentMessage(
            message_id="MSG-INT-001",
            source_agent=AgentID.GL_013,
            target_agent=AgentID.GL_001,
            message_type=MessageType.REQUEST,
            priority=MessagePriority.NORMAL,
            payload={
                "action": "get_equipment_emissions",
                "equipment_id": equipment.equipment_id,
                "equipment_type": equipment.equipment_type
            },
            timestamp=datetime.now(timezone.utc)
        )

        assert message.payload["equipment_id"] == "EQ-001"

    @pytest.mark.asyncio
    async def test_sensor_to_transformer_flow(self):
        """Test data flow from IoT sensors through transformers."""
        # Create sensor readings
        readings = [
            SensorReading(
                reading_id=f"READ-{i}",
                sensor_id="SENS-001",
                timestamp=datetime.now(timezone.utc) + timedelta(minutes=i),
                value=70.0 + i * 2.0,  # Fahrenheit
                unit="fahrenheit",
                quality=0.95
            )
            for i in range(5)
        ]

        # Transform to Celsius
        converter = UnitConverter()
        transformed = []
        for reading in readings:
            temp_c = converter.convert(reading.value, "fahrenheit", "celsius")
            transformed.append({
                "reading_id": reading.reading_id,
                "temperature_celsius": temp_c,
                "timestamp": reading.timestamp.isoformat()
            })

        assert len(transformed) == 5
        assert transformed[0]["temperature_celsius"] == pytest.approx(21.11, rel=0.1)

    @pytest.mark.asyncio
    async def test_condition_monitoring_alarm_flow(self):
        """Test alarm generation from condition monitoring data."""
        # Create vibration reading
        reading = VibrationReading(
            reading_id="VIB-INT-001",
            measurement_point_id="MP-001",
            timestamp=datetime.now(timezone.utc),
            overall_velocity=8.5,  # Above warning threshold
            velocity_unit=VibrationUnit.MM_S,
            axis=MeasurementAxis.HORIZONTAL,
            machine_state=MachineState.RUNNING,
            rpm=1750.0
        )

        # Define thresholds
        warning_threshold = 5.0
        alarm_threshold = 10.0

        # Generate alarm if needed
        alarm = None
        if reading.overall_velocity >= alarm_threshold:
            severity = AlarmSeverity.ALARM
        elif reading.overall_velocity >= warning_threshold:
            severity = AlarmSeverity.WARNING
            alarm = Alarm(
                alarm_id="ALM-INT-001",
                measurement_point_id=reading.measurement_point_id,
                equipment_id="EQ-001",
                timestamp=datetime.now(timezone.utc),
                severity=severity,
                state=AlarmState.ACTIVE,
                alarm_type="High Vibration",
                message=f"Vibration {reading.overall_velocity} exceeds warning threshold {warning_threshold}",
                threshold_value=warning_threshold,
                actual_value=reading.overall_velocity,
                unit=reading.velocity_unit
            )

        assert alarm is not None
        assert alarm.severity == AlarmSeverity.WARNING


# =============================================================================
# FIXTURES AND TEST UTILITIES
# =============================================================================

@pytest.fixture
def sample_equipment_list():
    """Generate sample equipment list for testing."""
    return [
        Equipment(
            equipment_id=f"EQ-{i:03d}",
            name=f"Equipment {i}",
            description=f"Test equipment {i}",
            equipment_type="Pump" if i % 2 == 0 else "Motor",
            manufacturer="TestMfg",
            status=EquipmentStatus.OPERATIONAL,
            criticality=EquipmentCriticality.HIGH if i < 3 else EquipmentCriticality.MEDIUM
        )
        for i in range(10)
    ]


@pytest.fixture
def sample_sensor_readings():
    """Generate sample sensor readings for testing."""
    return [
        SensorReading(
            reading_id=f"READ-{i:03d}",
            sensor_id="SENS-001",
            timestamp=datetime.now(timezone.utc) + timedelta(seconds=i * 10),
            value=70.0 + (i % 5) * 0.5,
            unit="celsius",
            quality=0.9 + (i % 10) * 0.01
        )
        for i in range(100)
    ]


@pytest.fixture
def sample_work_orders():
    """Generate sample work orders for testing."""
    statuses = [WorkOrderStatus.OPEN, WorkOrderStatus.IN_PROGRESS, WorkOrderStatus.COMPLETED]
    priorities = [WorkOrderPriority.HIGH, WorkOrderPriority.MEDIUM, WorkOrderPriority.LOW]

    return [
        WorkOrder(
            work_order_id=f"WO-{i:03d}",
            title=f"Work Order {i}",
            description=f"Test work order {i}",
            equipment_id=f"EQ-{i % 10:03d}",
            status=statuses[i % 3],
            priority=priorities[i % 3],
            work_order_type=WorkOrderType.PREVENTIVE,
            created_at=datetime.now(timezone.utc) - timedelta(days=i)
        )
        for i in range(20)
    ]


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance tests for integration components."""

    def test_cache_performance(self):
        """Test cache performance with many entries."""
        cache = LRUCache(max_size=10000, ttl_seconds=300)

        # Insert many entries
        import time
        start = time.time()
        for i in range(10000):
            cache.set(f"key_{i}", f"value_{i}")
        insert_time = time.time() - start

        # Retrieve entries
        start = time.time()
        for i in range(10000):
            cache.get(f"key_{i}")
        retrieve_time = time.time() - start

        # Should complete in reasonable time
        assert insert_time < 1.0  # Less than 1 second
        assert retrieve_time < 1.0  # Less than 1 second

    def test_data_transformer_batch_performance(self):
        """Test data transformer performance with large batches."""
        converter = UnitConverter()

        # Large batch of conversions
        import time
        start = time.time()
        for _ in range(10000):
            converter.convert(72.0, "fahrenheit", "celsius")
        conversion_time = time.time() - start

        # Should be fast
        assert conversion_time < 1.0  # Less than 1 second for 10k conversions

    def test_quality_scorer_batch_performance(self):
        """Test quality scorer performance with large batches."""
        scorer = DataQualityScorer(required_fields=["name", "value", "timestamp"])

        records = [
            {"name": f"Item {i}", "value": i * 10, "timestamp": "2024-06-15"}
            for i in range(1000)
        ]

        import time
        start = time.time()
        score = scorer.score_batch(records)
        scoring_time = time.time() - start

        assert scoring_time < 1.0  # Less than 1 second
        assert score > 0


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Tests for error handling across connectors."""

    def test_connection_error_handling(self):
        """Test connection error is properly raised."""
        with pytest.raises(ConnectionError):
            raise ConnectionError("Failed to connect to server")

    def test_authentication_error_handling(self):
        """Test authentication error is properly raised."""
        with pytest.raises(AuthenticationError):
            raise AuthenticationError("Invalid credentials")

    def test_timeout_error_handling(self):
        """Test timeout error is properly raised."""
        with pytest.raises(TimeoutError):
            raise TimeoutError("Request timed out after 30 seconds")

    def test_validation_error_handling(self):
        """Test validation error is properly raised."""
        with pytest.raises(ValidationError):
            raise ValidationError("Invalid data format")

    def test_circuit_open_error_handling(self):
        """Test circuit open error is properly raised."""
        with pytest.raises(CircuitOpenError):
            raise CircuitOpenError("Circuit breaker is open")

    def test_retry_exhausted_error_handling(self):
        """Test retry exhausted error is properly raised."""
        with pytest.raises(RetryExhaustedError):
            raise RetryExhaustedError("Max retries exceeded")

    def test_rate_limit_error_handling(self):
        """Test rate limit error is properly raised."""
        with pytest.raises(RateLimitError):
            raise RateLimitError("Rate limit exceeded")


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
        "--cov=.",
        "--cov-report=term-missing"
    ])
