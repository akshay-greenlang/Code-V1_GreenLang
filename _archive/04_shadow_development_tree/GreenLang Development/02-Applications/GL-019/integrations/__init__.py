"""
GL-019 HEATSCHEDULER Integration Module

Enterprise data integration connectors for production planning systems and energy management.
Supports ERP/MES integration, energy management systems, SCADA/DCS, and tariff data providers.

Integration Points:
- ERP Systems: SAP (RFC/BAPI), Oracle ERP (REST API)
- MES Systems: OPC UA, REST API interfaces
- Energy Management: Real-time pricing, demand response, grid operator signals
- SCADA/DCS: Equipment monitoring, heating equipment control
- Tariff Providers: Utility APIs, wholesale market data

Protocols Supported:
- OPC UA (Unified Architecture)
- Modbus TCP/RTU
- REST/HTTP(S)
- RFC (SAP Remote Function Call)

Author: GreenLang Data Integration Engineering Team
Version: 1.0.0
"""

from typing import List

# Version information
__version__ = "1.0.0"
__author__ = "GreenLang Data Integration Engineering Team"

# ERP Connector exports
from .erp_connector import (
    # Base classes
    ERPConnectorBase,
    ERPConfig,
    ERPConnectionStatus,
    # SAP connector
    SAPERPConnector,
    SAPConfig,
    SAPAuthMethod,
    # Oracle connector
    OracleERPConnector,
    OracleConfig,
    # Data models
    ProductionSchedule,
    WorkOrder,
    MaintenanceSchedule,
    ScheduleItem,
    WorkOrderStatus,
    MaintenanceType,
    # Factory functions
    create_erp_connector,
)

# Energy Management Connector exports
from .energy_management_connector import (
    # Base classes
    EMSConnectorBase,
    EMSConfig,
    EMSConnectionStatus,
    # Real-time pricing
    RealTimePricingConnector,
    PricingConfig,
    PricePoint,
    PricingFeed,
    # Demand response
    DemandResponseConnector,
    DemandResponseConfig,
    DemandResponseSignal,
    DemandResponseEvent,
    DemandResponseLevel,
    # Grid operator
    GridOperatorConnector,
    GridOperatorConfig,
    ISOSignal,
    GridFrequency,
    # Energy meter
    EnergyMeterConnector,
    EnergyMeterConfig,
    EnergyMeterReading,
    MeterProtocol,
    # Factory functions
    create_pricing_connector,
    create_demand_response_connector,
    create_grid_operator_connector,
    create_energy_meter_connector,
)

# SCADA Integration exports
from .scada_integration import (
    # Configuration
    SCADAConfig,
    ConnectionProtocol,
    # Equipment types
    EquipmentType,
    EquipmentStatus,
    HeatingEquipment,
    # Data models
    EquipmentReading,
    TemperatureReading,
    PowerReading,
    ControlSetpoint,
    # SCADA client
    SCADAClient,
    # Factory functions
    create_scada_client,
    create_heating_equipment_tags,
)

# Tariff Provider exports
from .tariff_provider import (
    # Configuration
    TariffProviderConfig,
    TariffType,
    # Data models
    TariffRate,
    TariffSchedule,
    TimeOfUseRate,
    DemandCharge,
    RateChange,
    # Connectors
    UtilityTariffConnector,
    WholesaleMarketConnector,
    LMPPricingConnector,
    # Factory functions
    create_tariff_connector,
    create_wholesale_connector,
    create_lmp_connector,
)

# Module-level exports
__all__: List[str] = [
    # Version
    "__version__",
    "__author__",
    # ERP Connector
    "ERPConnectorBase",
    "ERPConfig",
    "ERPConnectionStatus",
    "SAPERPConnector",
    "SAPConfig",
    "SAPAuthMethod",
    "OracleERPConnector",
    "OracleConfig",
    "ProductionSchedule",
    "WorkOrder",
    "MaintenanceSchedule",
    "ScheduleItem",
    "WorkOrderStatus",
    "MaintenanceType",
    "create_erp_connector",
    # Energy Management
    "EMSConnectorBase",
    "EMSConfig",
    "EMSConnectionStatus",
    "RealTimePricingConnector",
    "PricingConfig",
    "PricePoint",
    "PricingFeed",
    "DemandResponseConnector",
    "DemandResponseConfig",
    "DemandResponseSignal",
    "DemandResponseEvent",
    "DemandResponseLevel",
    "GridOperatorConnector",
    "GridOperatorConfig",
    "ISOSignal",
    "GridFrequency",
    "EnergyMeterConnector",
    "EnergyMeterConfig",
    "EnergyMeterReading",
    "MeterProtocol",
    "create_pricing_connector",
    "create_demand_response_connector",
    "create_grid_operator_connector",
    "create_energy_meter_connector",
    # SCADA Integration
    "SCADAConfig",
    "ConnectionProtocol",
    "EquipmentType",
    "EquipmentStatus",
    "HeatingEquipment",
    "EquipmentReading",
    "TemperatureReading",
    "PowerReading",
    "ControlSetpoint",
    "SCADAClient",
    "create_scada_client",
    "create_heating_equipment_tags",
    # Tariff Provider
    "TariffProviderConfig",
    "TariffType",
    "TariffRate",
    "TariffSchedule",
    "TimeOfUseRate",
    "DemandCharge",
    "RateChange",
    "UtilityTariffConnector",
    "WholesaleMarketConnector",
    "LMPPricingConnector",
    "create_tariff_connector",
    "create_wholesale_connector",
    "create_lmp_connector",
]