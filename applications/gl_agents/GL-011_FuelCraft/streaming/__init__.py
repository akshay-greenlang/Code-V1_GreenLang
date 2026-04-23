"""
GL-011 FUELCRAFT - Streaming Package

Event streaming definitions and schemas for FuelCraft.

Topics:
- fuel.inventory.v1 - Inventory telemetry updates
- fuel.prices.v1 - Fuel price updates
- fuel.recommendations.v1 - Optimization recommendations
- fuel.audit.v1 - Audit trail events

Schema Formats:
- Avro for high-throughput telemetry
- JSON Schema for API events
- All schemas registered in Schema Registry

Features:
- Backward compatible schema evolution
- Schema versioning
- Event correlation via trace IDs
- Provenance tracking via hashes
"""

from .event_schemas import (
    # Event Base
    EventHeader,
    EventMetadata,
    # Inventory Events
    InventoryUpdateEvent,
    InventoryLevelReading,
    TankStatus,
    # Price Events
    PriceUpdateEvent,
    FuelPrice,
    PriceSource,
    # Optimization Events
    OptimizationCompleteEvent,
    OptimizationResult,
    FuelBlend,
    # Audit Events
    AuditEvent,
    AuditAction,
    # Weather Events
    WeatherUpdateEvent,
    WeatherForecast,
    # Schema Registry
    SchemaDefinition,
    get_avro_schema,
    get_json_schema,
)

__all__ = [
    # Event Base
    "EventHeader",
    "EventMetadata",
    # Inventory Events
    "InventoryUpdateEvent",
    "InventoryLevelReading",
    "TankStatus",
    # Price Events
    "PriceUpdateEvent",
    "FuelPrice",
    "PriceSource",
    # Optimization Events
    "OptimizationCompleteEvent",
    "OptimizationResult",
    "FuelBlend",
    # Audit Events
    "AuditEvent",
    "AuditAction",
    # Weather Events
    "WeatherUpdateEvent",
    "WeatherForecast",
    # Schema Registry
    "SchemaDefinition",
    "get_avro_schema",
    "get_json_schema",
]

__version__ = "1.0.0"
