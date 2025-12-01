# -*- coding: utf-8 -*-
"""
GL-007 FURNACEPULSE Integrations Module

This module provides connectors for integrating the FurnacePerformanceOptimizer
agent with industrial control systems, sensors, and analyzers.

Connectors:
- FurnaceControllerConnector: SCADA/DCS integration for furnace control
- TemperatureSensorConnector: Thermocouple/RTD temperature sensor integration
- CombustionAnalyzerConnector: Flue gas analyzer integration for emissions monitoring

Protocols Supported:
- OPC UA (IEC 62541)
- Modbus TCP/RTU
- MQTT (IEC 62591)
- EtherNet/IP

Safety Standards:
- IEC 61508 (Functional Safety)
- IEC 61511 (Process Safety)
- NFPA 86 (Industrial Furnaces)
- API 556 (Fired Heaters)

Author: GL-DataIntegrationEngineer
Date: 2025-11-22
Version: 1.0.0
"""

from typing import TYPE_CHECKING

# Lazy imports for connectors
if TYPE_CHECKING:
    from .furnace_controller_connector import FurnaceControllerConnector
    from .temperature_sensor_connector import TemperatureSensorConnector
    from .combustion_analyzer_connector import CombustionAnalyzerConnector


__all__ = [
    "FurnaceControllerConnector",
    "TemperatureSensorConnector",
    "CombustionAnalyzerConnector",
]

__version__ = "1.0.0"
__author__ = "GL-DataIntegrationEngineer"
