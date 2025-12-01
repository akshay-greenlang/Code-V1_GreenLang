# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER Integrations Package

Provides connectors for steam trap monitoring and maintenance systems:
- TrapMonitorConnector: Real-time sensor data from trap monitoring systems
- MaintenanceSystemConnector: CMMS/EAM integration for work orders

Supported Protocols:
- OPC-UA for modern SCADA systems
- Modbus TCP/RTU for legacy systems
- REST API for cloud-based systems
- MQTT for IoT sensor networks

Author: GreenLang Industrial Optimization Team
Date: December 2025
Version: 1.0.0
"""

from .trap_monitor_connector import (
    TrapMonitorConnector,
    TrapMonitorConfig,
    TrapSensorData,
    SensorType,
    ConnectionState,
    create_trap_monitor_connector,
)

from .maintenance_system_connector import (
    MaintenanceSystemConnector,
    MaintenanceSystemConfig,
    WorkOrderData,
    WorkOrderStatus,
    WorkOrderPriority,
    CMMSType,
    create_maintenance_connector,
)

__all__ = [
    # Trap monitor connector
    "TrapMonitorConnector",
    "TrapMonitorConfig",
    "TrapSensorData",
    "SensorType",
    "ConnectionState",
    "create_trap_monitor_connector",
    # Maintenance system connector
    "MaintenanceSystemConnector",
    "MaintenanceSystemConfig",
    "WorkOrderData",
    "WorkOrderStatus",
    "WorkOrderPriority",
    "CMMSType",
    "create_maintenance_connector",
]

__version__ = "1.0.0"
