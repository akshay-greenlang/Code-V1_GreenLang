"""
GL-011 FUELCRAFT - Integration Package

Enterprise integration connectors for fuel optimization data:
- Kafka for event streaming (inventory, prices, recommendations)
- OPC-UA for raw OT telemetry (tank levels, temperatures)
- ERP systems for procurement and contracts
- Carbon accounting systems for emissions data

All integrations include:
- Circuit breakers per IEC 61511
- Automatic reconnection with exponential backoff
- Schema validation and registry integration
- Dead letter queue handling for failed messages

Topics:
- fuel.inventory.v1 - Inventory telemetry
- fuel.prices.v1 - Price updates
- fuel.recommendations.v1 - Optimization results
- fuel.audit.v1 - Audit events

OPC-UA Tags:
- TankLevel - Fuel tank level sensors
- Temperature - Storage temperature sensors
- FlowRate - Flow meters
"""

from .kafka_producer import (
    FuelCraftKafkaProducer,
    KafkaProducerConfig,
    RecommendationPublishedEvent,
    AuditEventPublished,
)

from .kafka_consumer import (
    FuelCraftKafkaConsumer,
    KafkaConsumerConfig,
    InventoryUpdateHandler,
    PriceUpdateHandler,
)

from .opcua_connector import (
    FuelCraftOPCUAConnector,
    OPCUAConfig,
    TankLevelTag,
    TemperatureTag,
    FlowRateTag,
    TelemetryReading,
)

from .erp_connector import (
    ERPConnector,
    ERPConfig,
    ProcurementOrder,
    ContractData,
    DeliverySchedule,
)

from .carbon_accounting_connector import (
    CarbonAccountingConnector,
    CarbonAccountingConfig,
    EmissionFactorData,
    CarbonFootprintExport,
    ReconciliationReport,
)

__all__ = [
    # Kafka Producer
    "FuelCraftKafkaProducer",
    "KafkaProducerConfig",
    "RecommendationPublishedEvent",
    "AuditEventPublished",
    # Kafka Consumer
    "FuelCraftKafkaConsumer",
    "KafkaConsumerConfig",
    "InventoryUpdateHandler",
    "PriceUpdateHandler",
    # OPC-UA
    "FuelCraftOPCUAConnector",
    "OPCUAConfig",
    "TankLevelTag",
    "TemperatureTag",
    "FlowRateTag",
    "TelemetryReading",
    # ERP
    "ERPConnector",
    "ERPConfig",
    "ProcurementOrder",
    "ContractData",
    "DeliverySchedule",
    # Carbon Accounting
    "CarbonAccountingConnector",
    "CarbonAccountingConfig",
    "EmissionFactorData",
    "CarbonFootprintExport",
    "ReconciliationReport",
]

__version__ = "1.0.0"
