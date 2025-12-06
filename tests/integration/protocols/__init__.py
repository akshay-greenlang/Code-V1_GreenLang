# -*- coding: utf-8 -*-
"""
GreenLang Protocol Integration Tests
====================================

Comprehensive integration tests for industrial protocol implementations:
- OPC-UA: Industrial automation standard (ISA-95 compliant)
- Modbus TCP/RTU: PLC communication
- MQTT: IoT pub/sub messaging
- Kafka: Event streaming with exactly-once semantics

Test Coverage Targets:
- Unit tests: 85%+
- Integration tests: Connection, read/write, subscriptions, failover
- Performance tests: Throughput, latency benchmarks
- Resilience tests: Failover, reconnection, error handling

Test Categories:
- test_opcua_integration.py: OPC-UA client/server integration
- test_modbus_integration.py: Modbus TCP/RTU gateway tests
- test_mqtt_integration.py: MQTT pub/sub messaging tests
- test_kafka_integration.py: Kafka producer/consumer tests
- test_protocol_failover.py: Protocol failover and resilience

Author: GreenLang Test Engineering Team
Date: December 2025
"""

from .conftest import (
    # Mock servers
    mock_opcua_server,
    mock_modbus_server,
    mock_mqtt_broker,
    mock_kafka_cluster,
    # Running fixtures
    running_mock_opcua_server,
    connected_mock_modbus_server,
    connected_mock_mqtt_broker,
    mock_kafka_with_data,
    # Patchers for external libraries
    patch_asyncua,
    patch_pymodbus,
    patch_aiomqtt,
    patch_aiokafka,
    # Test data
    sample_process_heat_data,
    sample_emission_event,
)

__all__ = [
    # Mock servers
    "mock_opcua_server",
    "mock_modbus_server",
    "mock_mqtt_broker",
    "mock_kafka_cluster",
    # Running fixtures
    "running_mock_opcua_server",
    "connected_mock_modbus_server",
    "connected_mock_mqtt_broker",
    "mock_kafka_with_data",
    # Patchers
    "patch_asyncua",
    "patch_pymodbus",
    "patch_aiomqtt",
    "patch_aiokafka",
    # Test data
    "sample_process_heat_data",
    "sample_emission_event",
]
