# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER - Integration Module

Provides connectors and integrations for steam trap monitoring:
- Trap monitoring system connectors (Armstrong, Spirax Sarco, TLV, Flowserve)
- Acoustic sensor integration for ultrasonic inspection
- Thermal camera connectors for IR imaging
- CMMS integration (SAP PM, Maximo)
- Historian connectors (OSIsoft PI, Wonderware, InfluxDB, TimescaleDB)

All connectors implement:
- Async operations with connection pooling
- Circuit breaker pattern for fault tolerance
- Comprehensive error handling and retry logic
- Real-time subscriptions where applicable

Author: GL-DataIntegrationEngineer
Date: December 2025
Version: 1.0.0
"""

from .trap_monitor_connector import (
    TrapMonitorConnector,
    TrapMonitorConfig,
    TrapSensorData,
    TrapDataBundle,
    TagMapping,
    SensorType,
    DataQuality,
    ConnectionState,
    ProtocolType,
    MonitorSystemVendor,
    create_trap_monitor_connector,
)

from .acoustic_sensor_connector import (
    AcousticSensorConnector,
    AcousticSensorConfig,
    AcousticReading,
    WaveformData,
    FFTResult,
    AcousticSensorType,
    AcquisitionMode,
    create_acoustic_connector,
)

from .thermal_camera_connector import (
    ThermalCameraConnector,
    ThermalCameraConfig,
    ThermalImage,
    ThermalReading,
    HotSpot,
    TemperatureMap,
    CameraType,
    create_thermal_connector,
)

from .cmms_connector import (
    CMMSConnector,
    CMMSConfig,
    WorkOrder,
    Asset,
    MaintenanceHistory,
    CMMSType,
    WorkOrderPriority,
    WorkOrderStatus,
    create_cmms_connector,
)

from .historian_connector import (
    HistorianConnector,
    HistorianConfig,
    TimeSeriesData,
    TimeSeriesPoint,
    BackfillResult,
    HistorianType,
    InterpolationMode,
    BackfillStatus,
    create_historian_connector,
)


__all__ = [
    # Trap Monitor Connector
    "TrapMonitorConnector",
    "TrapMonitorConfig",
    "TrapSensorData",
    "TrapDataBundle",
    "TagMapping",
    "SensorType",
    "DataQuality",
    "ConnectionState",
    "ProtocolType",
    "MonitorSystemVendor",
    "create_trap_monitor_connector",
    # Acoustic Sensor Connector
    "AcousticSensorConnector",
    "AcousticSensorConfig",
    "AcousticReading",
    "WaveformData",
    "FFTResult",
    "AcousticSensorType",
    "AcquisitionMode",
    "create_acoustic_connector",
    # Thermal Camera Connector
    "ThermalCameraConnector",
    "ThermalCameraConfig",
    "ThermalImage",
    "ThermalReading",
    "HotSpot",
    "TemperatureMap",
    "CameraType",
    "create_thermal_connector",
    # CMMS Connector
    "CMMSConnector",
    "CMMSConfig",
    "WorkOrder",
    "Asset",
    "MaintenanceHistory",
    "CMMSType",
    "WorkOrderPriority",
    "WorkOrderStatus",
    "create_cmms_connector",
    # Historian Connector
    "HistorianConnector",
    "HistorianConfig",
    "TimeSeriesData",
    "TimeSeriesPoint",
    "BackfillResult",
    "HistorianType",
    "InterpolationMode",
    "BackfillStatus",
    "create_historian_connector",
]

__version__ = "1.0.0"
__author__ = "GL-DataIntegrationEngineer"
