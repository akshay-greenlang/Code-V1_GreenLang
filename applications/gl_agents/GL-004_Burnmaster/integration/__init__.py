"""
GL-004 BURNMASTER - Integration Module

This module provides OT/IT integrations for the BURNMASTER combustion optimization
system, including DCS connectivity, BMS interface, historian connections, and
OPC-UA client capabilities.

Components:
    - DCSConnector: Distributed Control System connectivity
    - BMSInterface: Burner Management System read-only interface
    - HistorianConnector: Multi-historian data retrieval (PI, PHD, Proficy, InfluxDB)
    - OPCUAClient: OPC-UA protocol client using asyncua
    - IntegrationDataValidator: Data quality and freshness validation
    - WriteController: Safe write operations with audit logging
    - ConnectionManager: Connection lifecycle and failover management

Safety Philosophy:
    - All writes require safety checks and audit logging
    - BMS interface is strictly read-only (NEVER writes to BMS)
    - Mode permissive checks before any control writes
    - Handshake protocol for critical operations

Integration Architecture:
    DCS <-> DCSConnector <-> WriteController <-> BURNMASTER Core
    BMS -> BMSInterface -> Safety Checks -> BURNMASTER Core
    Historian <-> HistorianConnector <-> BURNMASTER Analytics
    OPC-UA <-> OPCUAClient <-> BURNMASTER Core

Author: GreenLang Combustion Systems Team
Version: 1.0.0
"""

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == 'DCSConnector':
        from .dcs_connector import DCSConnector
        return DCSConnector
    elif name == 'BMSInterface':
        from .bms_interface import BMSInterface
        return BMSInterface
    elif name == 'HistorianConnector':
        from .historian_connector import HistorianConnector
        return HistorianConnector
    elif name == 'OPCUAClient':
        from .opcua_client import OPCUAClient
        return OPCUAClient
    elif name == 'IntegrationDataValidator':
        from .data_validator import IntegrationDataValidator
        return IntegrationDataValidator
    elif name == 'WriteController':
        from .write_controller import WriteController
        return WriteController
    elif name == 'ConnectionManager':
        from .connection_manager import ConnectionManager
        return ConnectionManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'DCSConnector',
    'BMSInterface', 
    'HistorianConnector',
    'OPCUAClient',
    'IntegrationDataValidator',
    'WriteController',
    'ConnectionManager',
]

__version__ = '1.0.0'
__author__ = 'GreenLang Combustion Systems Team'
