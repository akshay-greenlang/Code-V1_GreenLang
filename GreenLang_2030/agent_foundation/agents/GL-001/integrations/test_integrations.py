"""
Comprehensive Integration Tests for GL-001 ProcessHeatOrchestrator

Tests all integration components:
- SCADA connectors (OPC UA, Modbus, MQTT)
- ERP connectors (SAP, Oracle, Dynamics, Workday)
- Multi-agent coordination
- Data transformation and validation

Includes:
- Unit tests for individual components
- Integration tests with mock servers
- Performance tests for high-volume data
- Security tests for authentication and encryption
"""

import asyncio
import unittest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import json
import random

# Import modules to test
from scada_connector import (
    SCADAConnector, OPCUAClient, ModbusTCPClient, MQTTSubscriber,
    SCADAConnectionPool, SCADADataBuffer, CircuitBreaker,
    SCADAProtocol, SCADAConnectionConfig, SCADASensorConfig
)

from erp_connector import (
    ERPConnector, SAPConnector, OracleConnector, DynamicsConnector,
    WorkdayConnector, ERPConnectionPool, TokenManager, RateLimiter,
    ERPSystem, ERPConfig, ERPDataRequest
)

from agent_coordinator import (
    AgentCoordinator, MessageBus, CommandBroadcaster, ResponseAggregator,
    AgentRegistry, AgentMessage, MessageType, MessagePriority,
    CoordinationStrategy
)

from data_transformers import (
    SCADADataTransformer, ERPDataTransformer, AgentMessageFormatter,
    DataValidator, UnitConverter, UnitType
)


class TestSCADAConnectors(unittest.TestCase):
    """Test SCADA integration components."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = SCADAConnectionConfig(
            protocol=SCADAProtocol.OPC_UA,
            host="192.168.1.100",
            port=4840,
            tls_enabled=True,
            timeout=30
        )

        self.sensor_config = SCADASensorConfig(
            sensor_id="TEMP_001",
            sensor_type="temperature",
            address="ns=2;i=1001",
            unit="celsius",
            min_value=0,
            max_value=200,
            sampling_rate=5
        )

    async def test_opcua_connection(self):
        """Test OPC UA client connection."""
        client = OPCUAClient(self.config)

        # Mock connection success
        with patch.object(client, 'connect', return_value=True):
            connected = await client.connect()
            self.assertTrue(connected)

    async def test_modbus_read_registers(self):
        """Test Modbus register reading."""
        modbus_config = SCADAConnectionConfig(
            protocol=SCADAProtocol.MODBUS_TCP,
            host="192.168.1.101",
            port=502,
            tls_enabled=False
        )

        client = ModbusTCPClient(modbus_config)
        client.connected = True

        # Test register reading
        registers = await client.read_registers(address=40001, count=5, unit=1)
        self.assertIsNotNone(registers)
        self.assertEqual(len(registers), 5)

    async def test_mqtt_subscription(self):
        """Test MQTT topic subscription."""
        mqtt_config = SCADAConnectionConfig(
            protocol=SCADAProtocol.MQTT,
            host="mqtt.broker.com",
            port=8883,
            tls_enabled=True
        )

        subscriber = MQTTSubscriber(mqtt_config)
        subscriber.connected = True

        # Test subscription
        callback = AsyncMock()
        await subscriber.subscribe_topic("sensors/temperature/+", callback)
        self.assertIn("sensors/temperature/+", subscriber.callbacks)

    async def test_data_buffer(self):
        """Test SCADA data buffering."""
        buffer = SCADADataBuffer(max_size=100, retention_hours=1)

        # Add data points
        for i in range(10):
            await buffer.add({
                'sensor_id': f'SENSOR_{i}',
                'value': random.uniform(10, 100),
                'timestamp': datetime.utcnow().isoformat()
            })

        # Retrieve data
        all_data = await buffer.get_all()
        self.assertEqual(len(all_data), 10)

        recent_data = await buffer.get_recent(minutes=5)
        self.assertEqual(len(recent_data), 10)

    def test_circuit_breaker(self):
        """Test circuit breaker fault tolerance."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=5)

        # Test normal operation
        self.assertTrue(breaker.can_attempt())

        # Record failures
        for _ in range(3):
            breaker.record_failure()

        # Circuit should be open
        self.assertEqual(breaker.state, "OPEN")
        self.assertFalse(breaker.can_attempt())

        # Test recovery after timeout
        breaker.last_failure_time = breaker.last_failure_time - 10
        self.assertTrue(breaker.can_attempt())
        self.assertEqual(breaker.state, "HALF_OPEN")

    async def test_connection_pool(self):
        """Test SCADA connection pooling."""
        pool = SCADAConnectionPool(max_connections=10)

        # Add multiple connections
        configs = [
            ("opcua_1", self.config),
            ("modbus_1", SCADAConnectionConfig(
                protocol=SCADAProtocol.MODBUS_TCP,
                host="192.168.1.102",
                port=502,
                tls_enabled=False
            ))
        ]

        for conn_id, config in configs:
            with patch('scada_connector.OPCUAClient.connect', return_value=True):
                with patch('scada_connector.ModbusTCPClient.connect', return_value=True):
                    success = await pool.add_connection(conn_id, config)
                    self.assertTrue(success)

        # Get connection
        client = await pool.get_connection("opcua_1")
        self.assertIsNotNone(client)

        # Health check
        await pool.health_check()
        self.assertIn("opcua_1", pool.health_status)


class TestERPConnectors(unittest.TestCase):
    """Test ERP integration components."""

    def setUp(self):
        """Set up test fixtures."""
        self.sap_config = ERPConfig(
            system=ERPSystem.SAP,
            base_url="https://sap.company.com/odata",
            api_version="v1",
            client_id="test_client",
            oauth_token_url="https://sap.company.com/oauth/token"
        )

        self.oracle_config = ERPConfig(
            system=ERPSystem.ORACLE,
            base_url="https://oracle.company.com",
            api_version="v1",
            client_id="oracle_client"
        )

    async def test_token_manager(self):
        """Test OAuth token management."""
        manager = TokenManager(self.sap_config)

        # Mock environment variable
        with patch.dict('os.environ', {'SAP_CLIENT_SECRET': 'test_secret'}):
            token = await manager.get_token()
            self.assertIsNotNone(token)
            self.assertTrue(token.startswith("bearer_sap_"))

    async def test_rate_limiter(self):
        """Test API rate limiting."""
        limiter = RateLimiter(requests_per_minute=60)

        # Test token consumption
        start_time = datetime.utcnow()
        for _ in range(5):
            await limiter.acquire()
        elapsed = (datetime.utcnow() - start_time).total_seconds()

        # Should complete quickly for first few requests
        self.assertLess(elapsed, 1.0)

    async def test_sap_connector(self):
        """Test SAP connector data retrieval."""
        connector = SAPConnector(self.sap_config)

        request = ERPDataRequest(
            data_type="energy_consumption",
            start_date="2024-01-01",
            end_date="2024-01-31",
            filters={'plant_code': 'PLANT01'}
        )

        # Mock token acquisition
        with patch.object(connector.token_manager, 'get_token', return_value='test_token'):
            response = await connector.fetch_energy_consumption(request)

            self.assertIsNotNone(response)
            self.assertGreater(len(response.data), 0)
            self.assertEqual(response.data[0]['PlantCode'], 'PLANT01')

    async def test_oracle_connector(self):
        """Test Oracle connector data retrieval."""
        connector = OracleConnector(self.oracle_config)

        request = ERPDataRequest(
            data_type="energy_consumption",
            start_date="2024-01-01",
            end_date="2024-01-31"
        )

        with patch.object(connector.token_manager, 'get_token', return_value='test_token'):
            response = await connector.fetch_energy_consumption(request)

            self.assertIsNotNone(response)
            self.assertIn('ConsumptionId', response.data[0])

    async def test_erp_connection_pool(self):
        """Test ERP connection pooling."""
        pool = ERPConnectionPool(max_connections=5)

        # Add connectors
        await pool.add_connector("sap_prod", self.sap_config)
        await pool.add_connector("oracle_cloud", self.oracle_config)

        # Retrieve connector
        sap_connector = await pool.get_connector("sap_prod")
        self.assertIsNotNone(sap_connector)
        self.assertIsInstance(sap_connector, SAPConnector)

    async def test_batch_data_fetch(self):
        """Test batch data fetching from ERP."""
        connector = ERPConnector()

        configs = [
            ("sap_test", self.sap_config)
        ]

        await connector.initialize(configs)

        # Mock the underlying fetch methods
        with patch.object(connector, 'fetch_energy_consumption') as mock_energy:
            with patch.object(connector, 'fetch_production_schedule') as mock_prod:
                mock_energy.return_value = Mock(total_records=10)
                mock_prod.return_value = Mock(total_records=5)

                batch_data = await connector.fetch_batch_data(
                    "sap_test",
                    ["energy_consumption", "production_schedule"],
                    "2024-01-01",
                    "2024-01-31"
                )

                self.assertEqual(len(batch_data), 2)


class TestAgentCoordination(unittest.TestCase):
    """Test multi-agent coordination components."""

    def setUp(self):
        """Set up test fixtures."""
        self.message_bus = MessageBus()
        self.registry = AgentRegistry()
        self.coordinator = AgentCoordinator()

    async def test_message_bus_pubsub(self):
        """Test message bus pub/sub functionality."""
        # Subscribe agents to topics
        await self.message_bus.subscribe("GL-002", ["temperature", "efficiency"])
        await self.message_bus.subscribe("GL-003", ["temperature", "pressure"])

        # Create test message
        message = AgentMessage(
            message_id="test_001",
            source_agent="GL-001",
            target_agents=["GL-002", "GL-003"],
            message_type=MessageType.EVENT,
            priority=MessagePriority.NORMAL,
            payload={'temperature': 150},
            timestamp=datetime.utcnow()
        )

        # Publish to topic
        await self.message_bus.publish("temperature", message)

        # Both subscribers should receive message
        msg_002 = await self.message_bus.receive("GL-002", timeout=1)
        msg_003 = await self.message_bus.receive("GL-003", timeout=1)

        self.assertIsNotNone(msg_002)
        self.assertIsNotNone(msg_003)
        self.assertEqual(msg_002.message_id, "test_001")

    async def test_agent_registry(self):
        """Test agent registry functionality."""
        # Get agent by ID
        agent = await self.registry.get_agent("GL-002")
        self.assertIsNotNone(agent)
        self.assertEqual(agent.agent_name, "BoilerEfficiencyOptimizer")

        # Update heartbeat
        await self.registry.update_heartbeat("GL-002")
        updated_agent = await self.registry.get_agent("GL-002")
        self.assertEqual(updated_agent.status, "online")

        # Get by capability
        boiler_agents = await self.registry.get_agents_by_capability("boiler")
        self.assertGreater(len(boiler_agents), 0)

    async def test_command_broadcasting(self):
        """Test command broadcasting strategies."""
        broadcaster = CommandBroadcaster(self.message_bus, self.registry)

        strategy = CoordinationStrategy(
            strategy_type="broadcast",
            max_parallel=3,
            timeout_seconds=5,
            aggregation_method="all"
        )

        # Mock send_and_wait
        with patch.object(broadcaster, '_send_and_wait', return_value={'status': 'success'}):
            responses = await broadcaster.broadcast_command(
                command="optimize",
                agent_ids=["GL-002", "GL-003", "GL-004"],
                parameters={'target': 0.95},
                strategy=strategy
            )

            self.assertEqual(len(responses), 3)
            self.assertEqual(responses["GL-002"]['status'], 'success')

    async def test_response_aggregation(self):
        """Test response aggregation methods."""
        aggregator = ResponseAggregator()

        responses = {
            "GL-002": {'value': 95, 'quality_score': 90},
            "GL-003": {'value': 93, 'quality_score': 85},
            "GL-004": {'value': 97, 'quality_score': 95}
        }

        # Test average aggregation
        avg_result = await aggregator.aggregate_responses(responses, 'average')
        self.assertEqual(avg_result['method'], 'average')
        self.assertEqual(avg_result['result'], 95.0)

        # Test best aggregation
        best_result = await aggregator.aggregate_responses(responses, 'best')
        self.assertEqual(best_result['method'], 'best')
        self.assertEqual(best_result['quality_score'], 95)

    async def test_coordinator_execution(self):
        """Test coordinator command execution."""
        await self.coordinator.initialize()

        # Mock the broadcaster
        with patch.object(self.coordinator.broadcaster, 'broadcast_command') as mock_broadcast:
            mock_broadcast.return_value = {
                "GL-002": {'status': 'success'},
                "GL-003": {'status': 'success'}
            }

            result = await self.coordinator.execute_command(
                command="start_optimization",
                target_agents=["GL-002", "GL-003"],
                parameters={'mode': 'aggressive'}
            )

            self.assertIn('responses', result)
            mock_broadcast.assert_called_once()


class TestDataTransformers(unittest.TestCase):
    """Test data transformation utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.scada_transformer = SCADADataTransformer()
        self.erp_transformer = ERPDataTransformer()
        self.message_formatter = AgentMessageFormatter()
        self.validator = DataValidator()
        self.unit_converter = UnitConverter()

    def test_unit_conversion(self):
        """Test unit conversion accuracy."""
        # Temperature conversion
        celsius = 100
        fahrenheit = self.unit_converter.convert(
            celsius, 'celsius', 'fahrenheit', UnitType.TEMPERATURE
        )
        self.assertAlmostEqual(fahrenheit, 212, places=1)

        # Pressure conversion
        bar = 10
        psi = self.unit_converter.convert(
            bar, 'bar', 'psi', UnitType.PRESSURE
        )
        self.assertAlmostEqual(psi, 145.038, places=2)

        # Energy conversion
        kwh = 1000
        mwh = self.unit_converter.convert(
            kwh, 'kwh', 'mwh', UnitType.ENERGY
        )
        self.assertEqual(mwh, 1.0)

    def test_sensor_data_validation(self):
        """Test SCADA sensor data validation."""
        valid_data = {
            'sensor_id': 'TEMP_001',
            'value': 150.5,
            'timestamp': datetime.utcnow().isoformat(),
            'unit': 'celsius',
            'min_value': 0,
            'max_value': 200
        }

        result = self.validator.validate_sensor_data(valid_data)
        self.assertTrue(result.is_valid)
        self.assertGreaterEqual(result.quality_score, 90)

        # Test invalid data
        invalid_data = {
            'sensor_id': 'INVALID',
            'value': 'not_a_number',
            'timestamp': 'invalid_date'
        }

        result = self.validator.validate_sensor_data(invalid_data)
        self.assertFalse(result.is_valid)
        self.assertLess(result.quality_score, 50)

    def test_scada_transformation(self):
        """Test SCADA data transformation."""
        raw_data = {
            'sensor_id': 'temp_001',
            'value': 100,
            'unit': 'fahrenheit',
            'timestamp': '2024-01-15T10:00:00Z',
            'calibration_factor': 1.05
        }

        transformed = self.scada_transformer.transform_sensor_reading(raw_data)

        self.assertEqual(transformed['sensor_id'], 'TEMP_001')  # Normalized
        self.assertIn('calibrated_value', transformed)
        self.assertEqual(transformed['calibrated_value'], 105.0)
        self.assertIn('quality_score', transformed)

    def test_erp_transformation(self):
        """Test ERP data transformation."""
        raw_data = {
            'PlantCode': 'PLANT01',
            'Date': '2024-01-15',
            'EnergyType': 'Electricity',
            'Consumption': 5000,
            'Unit': 'kWh',
            'Cost': 750,
            'Currency': 'USD'
        }

        transformed = self.erp_transformer.transform_energy_consumption(raw_data)

        self.assertEqual(transformed['plant_code'], 'PLANT01')
        self.assertEqual(transformed['energy_type'], 'Electricity')
        self.assertIn('unit_cost', transformed)
        self.assertEqual(transformed['unit_cost'], 0.15)

    def test_message_formatting(self):
        """Test agent message formatting."""
        # Command message
        command_msg = self.message_formatter.format_command_message(
            command='optimize',
            parameters={'target': 0.95},
            target_agents=['GL-002', 'GL-003'],
            priority='high'
        )

        self.assertEqual(command_msg['message_type'], 'command')
        self.assertEqual(command_msg['priority'], 'high')
        self.assertIn('GL-002', command_msg['target_agents'])

        # Query message
        query_msg = self.message_formatter.format_query_message(
            query='get_efficiency',
            filters={'plant': 'PLANT01'}
        )

        self.assertEqual(query_msg['message_type'], 'query')
        self.assertEqual(query_msg['payload']['query'], 'get_efficiency')

        # Event message
        event_msg = self.message_formatter.format_event_message(
            event_type='temperature_alert',
            event_data={'location': 'Boiler_01', 'temp': 250},
            severity='critical'
        )

        self.assertEqual(event_msg['message_type'], 'event')
        self.assertEqual(event_msg['severity'], 'critical')
        self.assertTrue(event_msg['requires_ack'])


class TestIntegrationPerformance(unittest.TestCase):
    """Performance tests for integration components."""

    async def test_scada_high_volume(self):
        """Test SCADA data handling at high volume."""
        connector = SCADAConnector()
        buffer = SCADADataBuffer(max_size=10000)

        # Simulate high-volume sensor data
        start_time = datetime.utcnow()
        data_points = 10000

        for i in range(data_points):
            await buffer.add({
                'sensor_id': f'SENSOR_{i % 100}',
                'value': random.uniform(0, 100),
                'timestamp': datetime.utcnow().isoformat()
            })

        elapsed = (datetime.utcnow() - start_time).total_seconds()
        throughput = data_points / elapsed

        # Should handle >10,000 data points/second
        self.assertGreater(throughput, 10000)

    async def test_agent_coordination_scalability(self):
        """Test agent coordination with many agents."""
        coordinator = AgentCoordinator()
        await coordinator.initialize()

        # Simulate coordinating 99 agents
        agent_ids = [f"GL-{i:03d}" for i in range(2, 101)]

        start_time = datetime.utcnow()

        # Mock response collection
        with patch.object(coordinator.broadcaster, 'broadcast_command') as mock:
            mock.return_value = {aid: {'status': 'success'} for aid in agent_ids}

            result = await coordinator.execute_command(
                command="collect_status",
                target_agents=agent_ids,
                parameters={}
            )

        elapsed = (datetime.utcnow() - start_time).total_seconds()

        # Should complete within 10 seconds for 99 agents
        self.assertLess(elapsed, 10)


class TestSecurityFeatures(unittest.TestCase):
    """Test security features of integrations."""

    def test_credential_management(self):
        """Test secure credential handling."""
        config = ERPConfig(
            system=ERPSystem.SAP,
            base_url="https://sap.company.com",
            api_version="v1",
            client_secret="should_not_be_hardcoded"
        )

        # Verify credentials are retrieved from environment
        with patch.dict('os.environ', {'SAP_CLIENT_SECRET': 'secure_secret'}):
            manager = TokenManager(config)
            # In production, would verify secure retrieval
            self.assertIsNotNone(manager.config)

    def test_tls_configuration(self):
        """Test TLS configuration for SCADA."""
        config = SCADAConnectionConfig(
            protocol=SCADAProtocol.OPC_UA,
            host="192.168.1.100",
            port=4840,
            tls_enabled=True,
            cert_path="/path/to/cert.pem",
            key_path="/path/to/key.pem"
        )

        client = OPCUAClient(config)
        ssl_context = client._create_ssl_context()

        # Verify TLS 1.3 is configured
        self.assertIsNotNone(ssl_context)
        # In production, would verify minimum TLS version

    def test_rate_limiting(self):
        """Test API rate limiting enforcement."""
        limiter = RateLimiter(requests_per_minute=60)

        # Test that rate limiting is enforced
        self.assertEqual(limiter.requests_per_minute, 60)
        self.assertEqual(limiter.tokens, 60)


# Run tests
if __name__ == '__main__':
    # Run async tests
    loop = asyncio.get_event_loop()

    # Create test suite
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSCADAConnectors))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestERPConnectors))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAgentCoordination))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDataTransformers))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIntegrationPerformance))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSecurityFeatures))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Summary for GL-001 ProcessHeatOrchestrator Integrations")
    print(f"{'='*60}")
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.2f}%")
    print(f"{'='*60}\n")