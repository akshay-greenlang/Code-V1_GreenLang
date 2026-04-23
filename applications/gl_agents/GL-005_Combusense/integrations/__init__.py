# -*- coding: utf-8 -*-
"""
GL-005 CombustionControlAgent Integration Connectors

Industrial-grade integration connectors for combustion control systems.

Connectors:
- DCSConnector: Distributed Control System (OPC UA, Modbus TCP)
- PLCConnector: Programmable Logic Controller (Modbus TCP/RTU)
- CombustionAnalyzerConnector: Gas analyzers (MQTT, Modbus)
- FlameScannerConnector: Flame detection (Digital I/O, Modbus)
- TemperatureSensorArrayConnector: Multi-sensor temperature (Modbus RTU)
- SCADAIntegration: SCADA system integration (OPC UA, MQTT)

Author: GL-DataIntegrationEngineer
Date: 2025-11-18
Version: 1.0.0
"""

from .dcs_connector import (
    DCSConnector,
    DCSConfig,
    DCSProtocol,
    ProcessVariable,
    DCSAlarm,
    DataQuality,
    CircuitBreaker,
    CircuitBreakerState
)

from .plc_connector import (
    PLCConnector,
    PLCConfig,
    PLCProtocol,
    PLCCoil,
    PLCRegister,
    CoilType,
    RegisterType,
    DataType
)

from .combustion_analyzer_connector import (
    CombustionAnalyzerConnector,
    AnalyzerConfig,
    AnalyzerProtocol,
    GasType,
    GasMeasurement,
    CalibrationStatus,
    AnalyzerStatus
)

from .flame_scanner_connector import (
    FlameScannerConnector,
    FlameScannerConfig,
    ScannerType,
    FlameStatus,
    FlameDetectionEvent,
    FlameStabilityMetrics,
    ScannerHealth
)

from .temperature_sensor_array_connector import (
    TemperatureSensorArrayConnector,
    SensorArrayConfig,
    TemperatureSensor,
    SensorType,
    TemperatureZone,
    TemperatureReading,
    SensorHealth
)

from .scada_integration import (
    SCADAIntegration,
    SCADAConfig,
    SCADATag,
    SCADAAlarm,
    OperatorCommand,
    DataPriority,
    AlarmSeverity,
    CommandType
)

__all__ = [
    # DCS Connector
    'DCSConnector',
    'DCSConfig',
    'DCSProtocol',
    'ProcessVariable',
    'DCSAlarm',
    'DataQuality',
    'CircuitBreaker',
    'CircuitBreakerState',

    # PLC Connector
    'PLCConnector',
    'PLCConfig',
    'PLCProtocol',
    'PLCCoil',
    'PLCRegister',
    'CoilType',
    'RegisterType',
    'DataType',

    # Combustion Analyzer
    'CombustionAnalyzerConnector',
    'AnalyzerConfig',
    'AnalyzerProtocol',
    'GasType',
    'GasMeasurement',
    'CalibrationStatus',
    'AnalyzerStatus',

    # Flame Scanner
    'FlameScannerConnector',
    'FlameScannerConfig',
    'ScannerType',
    'FlameStatus',
    'FlameDetectionEvent',
    'FlameStabilityMetrics',
    'ScannerHealth',

    # Temperature Sensors
    'TemperatureSensorArrayConnector',
    'SensorArrayConfig',
    'TemperatureSensor',
    'SensorType',
    'TemperatureZone',
    'TemperatureReading',
    'SensorHealth',

    # SCADA Integration
    'SCADAIntegration',
    'SCADAConfig',
    'SCADATag',
    'SCADAAlarm',
    'OperatorCommand',
    'DataPriority',
    'AlarmSeverity',
    'CommandType'
]

__version__ = '1.0.0'
