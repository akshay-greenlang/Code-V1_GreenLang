"""
Integration Tests for GL-003 SteamSystemAnalyzer

This package contains comprehensive integration tests covering:
- SCADA/DCS connectivity (OPC UA, Modbus)
- Steam meter integration
- Pressure sensor integration
- End-to-end workflows
- Parent agent coordination

Test Infrastructure:
- Docker containers for mock services
- PostgreSQL for data persistence
- Redis for caching
- MQTT broker for real-time messaging
- Mock SCADA/meters/sensors

Run tests with:
    docker-compose -f docker-compose.test.yml up --abort-on-container-exit

Author: GreenLang Test Engineering Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__all__ = [
    "conftest",
    "mock_servers",
    "test_scada_integration",
    "test_steam_meter_integration",
    "test_pressure_sensor_integration",
    "test_e2e_workflow",
    "test_parent_coordination"
]
