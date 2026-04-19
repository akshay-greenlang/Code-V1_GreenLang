"""
GL-017 CONDENSYNC Integrations Module

Enterprise-grade integrations for CondenserOptimizationAgent.
Provides connectivity to SCADA, DCS, cooling systems, historians, and CMMS.

Integration Components:
- SCADA Integration: OPC-UA client for condenser instrumentation
- Cooling Tower Integration: PLC communication for cooling tower control
- DCS Integration: Distributed Control System interface
- Historian Integration: PI System / OSIsoft data retrieval
- CMMS Integration: Computerized Maintenance Management System
- Message Bus Integration: Inter-agent communication

Author: GreenLang AI Platform
Version: 1.0.0
"""

from typing import Dict, Any, List, Optional

# SCADA Integration exports
from .scada_integration import (
    SCADAIntegration,
    SCADAConfig,
    CondenserTagMapping,
    TagValue,
    SetpointCommand,
    SCADAConnectionError,
    SCADAReadError,
    SCADAWriteError,
)

# Cooling Tower Integration exports
from .cooling_tower_integration import (
    CoolingTowerIntegration,
    CoolingTowerConfig,
    CellStatus,
    FanSpeedCommand,
    BasinTemperature,
    BlowdownConfig,
    WeatherData,
    CoolingTowerError,
)

# DCS Integration exports
from .dcs_integration import (
    DCSIntegration,
    DCSConfig,
    ControlMode,
    InterlockStatus,
    ValvePosition,
    CascadeController,
    DCSConnectionError,
    DCSControlError,
)

# Historian Integration exports
from .historian_integration import (
    HistorianIntegration,
    HistorianConfig,
    TrendQuery,
    HistoricalDataPoint,
    PerformanceBaseline,
    BatchExportConfig,
    HistorianError,
)

# CMMS Integration exports
from .cmms_integration import (
    CMMSIntegration,
    CMMSConfig,
    WorkOrder,
    WorkOrderPriority,
    EquipmentHistory,
    MaintenanceSchedule,
    SparePartStatus,
    CMMSError,
)

# Message Bus Integration exports
from .message_bus_integration import (
    MessageBusIntegration,
    MessageBusConfig,
    CondenserEvent,
    EventType,
    AgentMessage,
    AlertLevel,
    MessageBusError,
)

__version__ = "1.0.0"
__author__ = "GreenLang AI Platform"
__agent__ = "GL-017 CONDENSYNC"

__all__ = [
    # SCADA
    "SCADAIntegration",
    "SCADAConfig",
    "CondenserTagMapping",
    "TagValue",
    "SetpointCommand",
    "SCADAConnectionError",
    "SCADAReadError",
    "SCADAWriteError",
    # Cooling Tower
    "CoolingTowerIntegration",
    "CoolingTowerConfig",
    "CellStatus",
    "FanSpeedCommand",
    "BasinTemperature",
    "BlowdownConfig",
    "WeatherData",
    "CoolingTowerError",
    # DCS
    "DCSIntegration",
    "DCSConfig",
    "ControlMode",
    "InterlockStatus",
    "ValvePosition",
    "CascadeController",
    "DCSConnectionError",
    "DCSControlError",
    # Historian
    "HistorianIntegration",
    "HistorianConfig",
    "TrendQuery",
    "HistoricalDataPoint",
    "PerformanceBaseline",
    "BatchExportConfig",
    "HistorianError",
    # CMMS
    "CMMSIntegration",
    "CMMSConfig",
    "WorkOrder",
    "WorkOrderPriority",
    "EquipmentHistory",
    "MaintenanceSchedule",
    "SparePartStatus",
    "CMMSError",
    # Message Bus
    "MessageBusIntegration",
    "MessageBusConfig",
    "CondenserEvent",
    "EventType",
    "AgentMessage",
    "AlertLevel",
    "MessageBusError",
]


def get_integration_info() -> Dict[str, Any]:
    """
    Get information about all available integrations.

    Returns:
        Dictionary containing integration metadata
    """
    return {
        "module": "GL-017 CONDENSYNC Integrations",
        "version": __version__,
        "agent": __agent__,
        "integrations": {
            "scada": {
                "description": "OPC-UA client for condenser instrumentation",
                "protocol": "OPC-UA",
                "features": [
                    "Real-time data acquisition",
                    "Setpoint writing",
                    "Connection management",
                    "Auto-reconnection",
                ],
            },
            "cooling_tower": {
                "description": "PLC communication for cooling tower control",
                "protocol": "Modbus TCP / EtherNet/IP",
                "features": [
                    "Fan speed control",
                    "Basin temperature monitoring",
                    "Multi-cell load balancing",
                    "Weather compensation",
                ],
            },
            "dcs": {
                "description": "Distributed Control System interface",
                "protocol": "OPC-UA / Proprietary",
                "features": [
                    "Cascade control coordination",
                    "Interlock monitoring",
                    "Mode selection",
                    "Valve position feedback",
                ],
            },
            "historian": {
                "description": "PI System / OSIsoft data retrieval",
                "protocol": "PI Web API",
                "features": [
                    "Historical data retrieval",
                    "Trend analysis",
                    "Performance baselines",
                    "Batch export",
                ],
            },
            "cmms": {
                "description": "Maintenance management integration",
                "protocol": "REST API",
                "features": [
                    "Work order creation",
                    "Equipment history",
                    "Maintenance scheduling",
                    "Spare parts inventory",
                ],
            },
            "message_bus": {
                "description": "Inter-agent communication",
                "protocol": "AMQP / Redis Streams",
                "features": [
                    "Event publishing",
                    "Agent coordination",
                    "Alert broadcasting",
                    "State synchronization",
                ],
            },
        },
    }


async def initialize_all_integrations(
    scada_config: Optional["SCADAConfig"] = None,
    cooling_tower_config: Optional["CoolingTowerConfig"] = None,
    dcs_config: Optional["DCSConfig"] = None,
    historian_config: Optional["HistorianConfig"] = None,
    cmms_config: Optional["CMMSConfig"] = None,
    message_bus_config: Optional["MessageBusConfig"] = None,
) -> Dict[str, Any]:
    """
    Initialize all integration clients.

    Args:
        scada_config: SCADA integration configuration
        cooling_tower_config: Cooling tower integration configuration
        dcs_config: DCS integration configuration
        historian_config: Historian integration configuration
        cmms_config: CMMS integration configuration
        message_bus_config: Message bus integration configuration

    Returns:
        Dictionary containing initialized integration clients
    """
    integrations = {}

    if scada_config:
        integrations["scada"] = SCADAIntegration(scada_config)
        await integrations["scada"].connect()

    if cooling_tower_config:
        integrations["cooling_tower"] = CoolingTowerIntegration(cooling_tower_config)
        await integrations["cooling_tower"].connect()

    if dcs_config:
        integrations["dcs"] = DCSIntegration(dcs_config)
        await integrations["dcs"].connect()

    if historian_config:
        integrations["historian"] = HistorianIntegration(historian_config)
        await integrations["historian"].connect()

    if cmms_config:
        integrations["cmms"] = CMMSIntegration(cmms_config)
        await integrations["cmms"].connect()

    if message_bus_config:
        integrations["message_bus"] = MessageBusIntegration(message_bus_config)
        await integrations["message_bus"].connect()

    return integrations


async def shutdown_all_integrations(integrations: Dict[str, Any]) -> None:
    """
    Gracefully shutdown all integration clients.

    Args:
        integrations: Dictionary of initialized integration clients
    """
    for name, client in integrations.items():
        try:
            if hasattr(client, "disconnect"):
                await client.disconnect()
            elif hasattr(client, "close"):
                await client.close()
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                f"Error shutting down {name} integration: {e}"
            )
