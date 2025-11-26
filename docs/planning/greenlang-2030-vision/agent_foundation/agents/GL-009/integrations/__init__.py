"""
GL-009 THERMALIQ Integration Connectors.

Industrial system connectors for real-time energy monitoring and optimization.

Supported Systems:
- Energy meters (Modbus TCP/RTU, OPC-UA)
- Process historians (OSIsoft PI, Wonderware, AspenTech)
- SCADA systems (OPC-UA)
- ERP systems (SAP, Oracle)
- Fuel flow meters
- Steam meters

All connectors implement:
- Automatic reconnection with exponential backoff
- Comprehensive error handling and logging
- Health monitoring
- Audit logging
- Async/sync operation modes
"""

from .base_connector import BaseConnector, ConnectorStatus, ConnectorHealth
from .energy_meter_connector import (
    EnergyMeterConnector,
    MeterProtocol,
    MeterConfig,
    EnergyReading,
)
from .historian_connector import (
    HistorianConnector,
    HistorianType,
    HistorianConfig,
    TimeSeriesData,
    AggregationType,
    InterpolationType,
)
from .scada_connector import (
    SCADAConnector,
    SCADAConfig,
    TagValue,
    AlarmEvent,
    TagSubscription,
)
from .erp_connector import (
    ERPConnector,
    ERPSystem,
    ERPConfig,
    EnergyCostData,
    ProductionData,
)
from .fuel_flow_connector import (
    FuelFlowConnector,
    FuelFlowConfig,
    FuelType,
    FlowReading,
    FlowMeterType,
)
from .steam_meter_connector import (
    SteamMeterConnector,
    SteamMeterConfig,
    SteamReading,
    SteamQuality,
)

__all__ = [
    # Base
    "BaseConnector",
    "ConnectorStatus",
    "ConnectorHealth",
    # Energy Meter
    "EnergyMeterConnector",
    "MeterProtocol",
    "MeterConfig",
    "EnergyReading",
    # Historian
    "HistorianConnector",
    "HistorianType",
    "HistorianConfig",
    "TimeSeriesData",
    "AggregationType",
    "InterpolationType",
    # SCADA
    "SCADAConnector",
    "SCADAConfig",
    "TagValue",
    "AlarmEvent",
    "TagSubscription",
    # ERP
    "ERPConnector",
    "ERPSystem",
    "ERPConfig",
    "EnergyCostData",
    "ProductionData",
    # Fuel Flow
    "FuelFlowConnector",
    "FuelFlowConfig",
    "FuelType",
    "FlowReading",
    "FlowMeterType",
    # Steam Meter
    "SteamMeterConnector",
    "SteamMeterConfig",
    "SteamReading",
    "SteamQuality",
]

__version__ = "1.0.0"
__author__ = "GreenLang GL-009 THERMALIQ Team"
