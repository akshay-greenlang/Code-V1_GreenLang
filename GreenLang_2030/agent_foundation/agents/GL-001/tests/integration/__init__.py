"""
GL-001 ProcessHeatOrchestrator Integration Tests

Comprehensive integration test suite for GL-001 master orchestrator including:
- End-to-end workflow testing
- SCADA integration (OPC UA, Modbus)
- ERP integration (SAP, Oracle)
- Sub-agent coordination (GL-002, GL-003, GL-004, GL-005)
- Multi-plant orchestration
- Performance and load testing
- Compliance validation

Test Structure:
    - conftest.py: Shared fixtures and test infrastructure
    - mock_servers.py: Mock external services
    - test_e2e_workflow.py: End-to-end orchestration tests
    - test_scada_integration.py: SCADA connectivity tests
    - test_erp_integration.py: ERP system integration tests
    - test_agent_coordination.py: Sub-agent orchestration tests
    - test_multi_plant_orchestration.py: Multi-plant scenarios
    - test_thermal_efficiency.py: Thermal efficiency integration
    - test_heat_distribution.py: Heat distribution optimization
    - test_emissions_compliance.py: Emissions compliance testing
    - test_performance_integration.py: Performance and load tests

Running Integration Tests:
    # All integration tests
    pytest tests/integration/ -v

    # Specific test file
    pytest tests/integration/test_e2e_workflow.py -v

    # With Docker infrastructure
    docker-compose -f tests/integration/docker-compose.test.yml up -d
    pytest tests/integration/ -v --docker
    docker-compose -f tests/integration/docker-compose.test.yml down

    # Performance tests only
    pytest tests/integration/test_performance_integration.py -v -m performance

    # End-to-end tests only
    pytest tests/integration/test_e2e_workflow.py -v -m e2e

Requirements:
    - Docker and docker-compose for test infrastructure
    - See requirements-test.txt for Python dependencies
    - Mock servers for SCADA, ERP, and sub-agents
    - PostgreSQL, Redis, MQTT for integration testing

Environment Variables:
    - TEST_POSTGRES_HOST: PostgreSQL host (default: localhost)
    - TEST_POSTGRES_PORT: PostgreSQL port (default: 5432)
    - TEST_REDIS_HOST: Redis host (default: localhost)
    - TEST_REDIS_PORT: Redis port (default: 6379)
    - TEST_MQTT_HOST: MQTT broker host (default: localhost)
    - TEST_MQTT_PORT: MQTT broker port (default: 1883)
    - TEST_SCADA_OPC_PORT: OPC UA port (default: 4840)
    - TEST_SCADA_MODBUS_PORT: Modbus TCP port (default: 502)
    - TEST_ERP_SAP_PORT: SAP RFC port (default: 3300)
    - TEST_ERP_ORACLE_PORT: Oracle API port (default: 8080)

Test Coverage Target: 85%+
Performance Target: <2s orchestration latency for 3 plants
"""

__version__ = "1.0.0"
__author__ = "GreenLang Test Engineering Team"

# Test markers
INTEGRATION_MARKERS = [
    "e2e",           # End-to-end workflow tests
    "scada",         # SCADA integration tests
    "erp",           # ERP integration tests
    "coordination",  # Agent coordination tests
    "multi_plant",   # Multi-plant orchestration tests
    "performance",   # Performance and load tests
    "compliance",    # Compliance validation tests
    "slow",          # Slow-running tests
]

# Test configuration
TEST_CONFIG = {
    "integration_timeout_seconds": 300,
    "performance_test_duration_seconds": 60,
    "multi_plant_count": 3,
    "load_test_concurrent_requests": 100,
    "scada_connection_timeout_seconds": 10,
    "erp_connection_timeout_seconds": 15,
}
