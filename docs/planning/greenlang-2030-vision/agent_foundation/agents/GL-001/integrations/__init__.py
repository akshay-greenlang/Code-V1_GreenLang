# -*- coding: utf-8 -*-
"""
GL-001 ProcessHeatOrchestrator Integration Package

This package provides secure, scalable integration connectors for:
- SCADA systems (OPC UA, Modbus TCP, MQTT)
- ERP systems (SAP, Oracle, Microsoft Dynamics, Workday)
- Multi-agent coordination (GL-002 through GL-100)
"""

from .scada_connector import (
    SCADAConnector,
    OPCUAClient,
    ModbusTCPClient,
    MQTTSubscriber,
    SCADAConnectionPool,
    SCADADataBuffer
)

from .erp_connector import (
    ERPConnector,
    SAPConnector,
    OracleConnector,
    DynamicsConnector,
    WorkdayConnector,
    ERPConnectionPool
)

from .agent_coordinator import (
    AgentCoordinator,
    MessageBus,
    CommandBroadcaster,
    ResponseAggregator,
    AgentRegistry
)

from .data_transformers import (
    SCADADataTransformer,
    ERPDataTransformer,
    AgentMessageFormatter,
    DataValidator,
    UnitConverter
)

__version__ = "1.0.0"
__author__ = "GreenLang Data Integration Team"

__all__ = [
    # SCADA
    "SCADAConnector",
    "OPCUAClient",
    "ModbusTCPClient",
    "MQTTSubscriber",
    "SCADAConnectionPool",
    "SCADADataBuffer",

    # ERP
    "ERPConnector",
    "SAPConnector",
    "OracleConnector",
    "DynamicsConnector",
    "WorkdayConnector",
    "ERPConnectionPool",

    # Agent Coordination
    "AgentCoordinator",
    "MessageBus",
    "CommandBroadcaster",
    "ResponseAggregator",
    "AgentRegistry",

    # Data Transformation
    "SCADADataTransformer",
    "ERPDataTransformer",
    "AgentMessageFormatter",
    "DataValidator",
    "UnitConverter"
]