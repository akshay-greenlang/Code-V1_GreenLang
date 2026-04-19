"""
GL-011 FUELCRAFT - Event Schemas

Event schema definitions for Kafka streaming with:
- Avro schemas for high-throughput telemetry
- JSON schemas for API events
- Schema Registry integration
- Backward compatible evolution

Events:
- InventoryUpdateEvent - Tank level and temperature updates
- PriceUpdateEvent - Fuel price changes
- OptimizationCompleteEvent - Optimization results
- AuditEvent - Compliance audit trail

Schema Evolution Rules:
- Add optional fields only
- Never remove required fields
- Never change field types
- Use default values for new fields
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import hashlib
import json
import uuid

from pydantic import BaseModel, Field, validator


# =============================================================================
# Enumerations
# =============================================================================

class EventType(str, Enum):
    """Event types for FuelCraft streaming."""
    INVENTORY_UPDATE = "inventory_update"
    PRICE_UPDATE = "price_update"
    OPTIMIZATION_COMPLETE = "optimization_complete"
    AUDIT = "audit"
    WEATHER_UPDATE = "weather_update"
    ALERT = "alert"


class DataQuality(str, Enum):
    """Data quality indicators."""
    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"
    STALE = "stale"


class TankStatus(str, Enum):
    """Tank operational status."""
    OPERATIONAL = "operational"
    FILLING = "filling"
    DRAINING = "draining"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


class PriceSource(str, Enum):
    """Price data source."""
    PLATTS = "platts"
    ARGUS = "argus"
    ICE = "ice"
    NYMEX = "nymex"
    INTERNAL = "internal"
    MANUAL = "manual"


class AuditAction(str, Enum):
    """Audit action types."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    READ = "read"
    EXECUTE = "execute"
    APPROVE = "approve"
    REJECT = "reject"


# =============================================================================
# Base Event Models
# =============================================================================

class EventHeader(BaseModel):
    """
    Standard event header for all FuelCraft events.

    Contains metadata for routing, tracing, and schema management.
    """
    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique event identifier"
    )
    event_type: EventType = Field(..., description="Event type")
    schema_version: str = Field("1.0.0", description="Schema version")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Event timestamp (ISO 8601)"
    )
    source: str = Field(..., description="Source system/component")

    # Tracing
    correlation_id: Optional[str] = Field(
        None, description="Correlation ID for request tracing"
    )
    trace_id: Optional[str] = Field(
        None, description="Distributed trace ID"
    )
    span_id: Optional[str] = Field(
        None, description="Span ID for distributed tracing"
    )

    # Provenance
    producer_id: str = Field("fuelcraft", description="Producer identifier")
    partition_key: Optional[str] = Field(
        None, description="Key for Kafka partitioning"
    )


class EventMetadata(BaseModel):
    """Extended metadata for events."""
    site_id: str = Field(..., description="Site identifier")
    tenant_id: Optional[str] = Field(None, description="Multi-tenant ID")
    environment: str = Field("production", description="Environment")
    region: Optional[str] = Field(None, description="Geographic region")
    tags: Dict[str, str] = Field(default_factory=dict)


# =============================================================================
# Inventory Events
# =============================================================================

class InventoryLevelReading(BaseModel):
    """Individual tank level reading."""
    tank_id: str = Field(..., description="Tank identifier")
    fuel_type: str = Field(..., description="Fuel type in tank")

    # Level readings
    level_percent: float = Field(..., ge=0, le=100, description="Fill level %")
    level_mmbtu: float = Field(..., ge=0, description="Content in MMBtu")
    level_volume_gallons: Optional[float] = Field(None, ge=0)

    # Temperature
    temperature_c: Optional[float] = Field(None, description="Temperature in Celsius")
    temperature_f: Optional[float] = Field(None, description="Temperature in Fahrenheit")

    # Density
    density_kg_m3: Optional[float] = Field(None, gt=0)
    api_gravity: Optional[float] = Field(None)

    # Tank capacity
    tank_capacity_mmbtu: float = Field(..., gt=0)

    # Status
    tank_status: TankStatus = Field(TankStatus.OPERATIONAL)
    data_quality: DataQuality = Field(DataQuality.GOOD)

    # Timestamps
    reading_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    sensor_timestamp: Optional[str] = Field(None)

    @validator("temperature_f", always=True)
    def compute_fahrenheit(cls, v, values):
        if v is None and "temperature_c" in values and values["temperature_c"] is not None:
            return values["temperature_c"] * 9/5 + 32
        return v


class InventoryUpdateEvent(BaseModel):
    """
    Inventory update event.

    Topic: fuel.inventory.v1

    Published when tank levels or temperatures change.
    Supports batch updates from multiple tanks.
    """
    header: EventHeader
    metadata: EventMetadata

    # Readings
    readings: List[InventoryLevelReading] = Field(
        ..., description="Tank readings", min_items=1
    )

    # Aggregates
    total_inventory_mmbtu: float = Field(..., ge=0)
    total_capacity_mmbtu: float = Field(..., gt=0)
    overall_fill_percent: float = Field(..., ge=0, le=100)

    # Alerts
    low_level_tanks: List[str] = Field(
        default=[], description="Tanks below minimum threshold"
    )
    high_level_tanks: List[str] = Field(
        default=[], description="Tanks near capacity"
    )

    def to_avro_dict(self) -> Dict[str, Any]:
        """Convert to Avro-compatible dictionary."""
        return {
            "header": self.header.dict(),
            "metadata": self.metadata.dict(),
            "readings": [r.dict() for r in self.readings],
            "total_inventory_mmbtu": self.total_inventory_mmbtu,
            "total_capacity_mmbtu": self.total_capacity_mmbtu,
            "overall_fill_percent": self.overall_fill_percent,
            "low_level_tanks": self.low_level_tanks,
            "high_level_tanks": self.high_level_tanks,
        }

    def to_kafka_bytes(self) -> bytes:
        """Serialize for Kafka."""
        return json.dumps(self.dict(), default=str).encode("utf-8")


# =============================================================================
# Price Events
# =============================================================================

class FuelPrice(BaseModel):
    """Fuel price data."""
    fuel_type: str = Field(..., description="Fuel type")
    region: str = Field("US", description="Price region")

    # Prices
    spot_price_usd_mmbtu: float = Field(..., ge=0)
    forward_1m_usd_mmbtu: Optional[float] = Field(None, ge=0)
    forward_3m_usd_mmbtu: Optional[float] = Field(None, ge=0)
    forward_6m_usd_mmbtu: Optional[float] = Field(None, ge=0)
    forward_12m_usd_mmbtu: Optional[float] = Field(None, ge=0)

    # Basis
    basis_differential_usd_mmbtu: float = Field(0.0)
    transport_cost_usd_mmbtu: float = Field(0.0, ge=0)

    # Price metadata
    price_source: PriceSource = Field(PriceSource.PLATTS)
    quote_type: str = Field("mid", description="bid, ask, mid")
    currency: str = Field("USD")

    # Timestamps
    effective_date: str
    quote_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class PriceUpdateEvent(BaseModel):
    """
    Price update event.

    Topic: fuel.prices.v1

    Published when fuel prices change.
    """
    header: EventHeader
    metadata: EventMetadata

    # Prices
    prices: List[FuelPrice] = Field(..., min_items=1)

    # Market context
    market_session: str = Field("regular", description="regular, after_hours")
    is_settlement: bool = Field(False, description="Settlement price flag")

    # Change indicators
    significant_changes: List[Dict[str, Any]] = Field(
        default=[], description="Prices with significant movement"
    )

    def to_kafka_bytes(self) -> bytes:
        """Serialize for Kafka."""
        return json.dumps(self.dict(), default=str).encode("utf-8")


# =============================================================================
# Optimization Events
# =============================================================================

class FuelBlend(BaseModel):
    """Fuel blend component."""
    fuel_type: str
    percentage: float = Field(..., ge=0, le=100)
    quantity_mmbtu: float = Field(..., ge=0)
    cost_usd: float = Field(..., ge=0)
    emissions_mtco2e: float = Field(..., ge=0)


class OptimizationResult(BaseModel):
    """Optimization result summary."""
    run_id: str = Field(..., description="Optimization run ID")
    objective: str = Field(..., description="Optimization objective")
    solver_status: str = Field(..., description="Solver status")

    # Fuel mix
    fuel_blend: List[FuelBlend] = Field(...)

    # Totals
    total_quantity_mmbtu: float = Field(..., ge=0)
    total_cost_usd: float = Field(..., ge=0)
    total_emissions_mtco2e: float = Field(..., ge=0)

    # Savings
    cost_savings_usd: float = Field(0.0)
    cost_savings_percent: float = Field(0.0)
    emission_reduction_mtco2e: float = Field(0.0)
    emission_reduction_percent: float = Field(0.0)

    # Time window
    effective_start: str
    effective_end: str

    # Performance
    optimization_time_ms: float = Field(..., ge=0)

    # Provenance
    bundle_hash: str = Field(..., description="SHA-256 computation hash")
    input_snapshot_ids: Dict[str, str] = Field(default_factory=dict)


class OptimizationCompleteEvent(BaseModel):
    """
    Optimization complete event.

    Topic: fuel.recommendations.v1

    Published when optimization run completes.
    """
    header: EventHeader
    metadata: EventMetadata

    # Result
    result: OptimizationResult

    # Procurement actions
    procurement_recommendations: List[Dict[str, Any]] = Field(default=[])

    # Alerts
    constraint_violations: List[str] = Field(default=[])
    warnings: List[str] = Field(default=[])

    def to_kafka_bytes(self) -> bytes:
        """Serialize for Kafka."""
        return json.dumps(self.dict(), default=str).encode("utf-8")


# =============================================================================
# Audit Events
# =============================================================================

class AuditEvent(BaseModel):
    """
    Audit event for compliance tracking.

    Topic: fuel.audit.v1

    Published for all significant actions for ISO 14064 compliance.
    """
    header: EventHeader
    metadata: EventMetadata

    # Audit details
    action: AuditAction = Field(..., description="Action performed")
    resource_type: str = Field(..., description="Resource type")
    resource_id: str = Field(..., description="Resource identifier")

    # Actor
    user_id: str = Field(..., description="User or service performing action")
    user_type: str = Field("user", description="user, service, system")
    client_ip: Optional[str] = Field(None)
    user_agent: Optional[str] = Field(None)

    # Request context
    request_id: Optional[str] = Field(None)
    api_endpoint: Optional[str] = Field(None)
    http_method: Optional[str] = Field(None)

    # Change tracking
    before_state: Optional[Dict[str, Any]] = Field(None)
    after_state: Optional[Dict[str, Any]] = Field(None)
    changes: Optional[List[Dict[str, Any]]] = Field(None)
    change_summary: Optional[str] = Field(None)

    # Related resources
    related_run_id: Optional[str] = Field(None, description="Related optimization run")
    related_resources: List[str] = Field(default=[])

    # Provenance
    computation_hash: Optional[str] = Field(
        None, description="Hash for reproducibility"
    )

    # Result
    success: bool = Field(True)
    error_message: Optional[str] = Field(None)

    def compute_audit_hash(self) -> str:
        """Compute hash of audit event for integrity."""
        data = json.dumps({
            "action": self.action.value,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "user_id": self.user_id,
            "timestamp": self.header.timestamp,
            "changes": self.changes,
        }, sort_keys=True, default=str)
        return hashlib.sha256(data.encode()).hexdigest()

    def to_kafka_bytes(self) -> bytes:
        """Serialize for Kafka."""
        return json.dumps(self.dict(), default=str).encode("utf-8")


# =============================================================================
# Weather Events
# =============================================================================

class WeatherForecast(BaseModel):
    """Weather forecast data point."""
    forecast_time: str
    temperature_c: float
    temperature_f: Optional[float] = None
    humidity_percent: Optional[float] = Field(None, ge=0, le=100)
    wind_speed_kmh: Optional[float] = Field(None, ge=0)
    precipitation_mm: Optional[float] = Field(None, ge=0)

    # Degree days
    hdd: Optional[float] = Field(None, ge=0, description="Heating degree days")
    cdd: Optional[float] = Field(None, ge=0, description="Cooling degree days")

    @validator("temperature_f", always=True)
    def compute_fahrenheit(cls, v, values):
        if v is None and "temperature_c" in values:
            return values["temperature_c"] * 9/5 + 32
        return v


class WeatherUpdateEvent(BaseModel):
    """
    Weather update event.

    Topic: fuel.weather.v1

    Published when weather forecasts are updated.
    """
    header: EventHeader
    metadata: EventMetadata

    # Forecast source
    forecast_source: str = Field(..., description="Weather data provider")
    forecast_model: Optional[str] = Field(None, description="Forecast model used")

    # Location
    location_id: str = Field(..., description="Location identifier")
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)

    # Forecasts
    forecasts: List[WeatherForecast] = Field(..., min_items=1)
    forecast_horizon_hours: int = Field(..., gt=0)

    # Aggregates
    avg_temperature_c: float
    total_hdd: float = Field(0.0, ge=0)
    total_cdd: float = Field(0.0, ge=0)

    # Demand impact
    estimated_demand_impact_percent: Optional[float] = Field(
        None, description="Estimated impact on fuel demand"
    )

    def to_kafka_bytes(self) -> bytes:
        """Serialize for Kafka."""
        return json.dumps(self.dict(), default=str).encode("utf-8")


# =============================================================================
# Schema Registry Definitions
# =============================================================================

class SchemaDefinition(BaseModel):
    """Schema definition for Schema Registry."""
    schema_name: str
    schema_type: str = Field("AVRO", description="AVRO, JSON, PROTOBUF")
    version: str
    schema_content: Dict[str, Any]
    compatibility: str = Field("BACKWARD", description="BACKWARD, FORWARD, FULL, NONE")


def get_avro_schema(event_type: EventType) -> Dict[str, Any]:
    """
    Get Avro schema for event type.

    Args:
        event_type: Event type

    Returns:
        Avro schema definition
    """
    schemas = {
        EventType.INVENTORY_UPDATE: {
            "type": "record",
            "name": "InventoryUpdateEvent",
            "namespace": "io.greenlang.fuelcraft.events",
            "fields": [
                {"name": "event_id", "type": "string"},
                {"name": "event_type", "type": "string"},
                {"name": "schema_version", "type": "string", "default": "1.0.0"},
                {"name": "timestamp", "type": "string"},
                {"name": "source", "type": "string"},
                {"name": "site_id", "type": "string"},
                {"name": "readings", "type": {
                    "type": "array",
                    "items": {
                        "type": "record",
                        "name": "InventoryLevelReading",
                        "fields": [
                            {"name": "tank_id", "type": "string"},
                            {"name": "fuel_type", "type": "string"},
                            {"name": "level_percent", "type": "double"},
                            {"name": "level_mmbtu", "type": "double"},
                            {"name": "temperature_c", "type": ["null", "double"], "default": None},
                            {"name": "data_quality", "type": "string", "default": "good"},
                        ]
                    }
                }},
                {"name": "total_inventory_mmbtu", "type": "double"},
                {"name": "total_capacity_mmbtu", "type": "double"},
            ]
        },
        EventType.PRICE_UPDATE: {
            "type": "record",
            "name": "PriceUpdateEvent",
            "namespace": "io.greenlang.fuelcraft.events",
            "fields": [
                {"name": "event_id", "type": "string"},
                {"name": "event_type", "type": "string"},
                {"name": "schema_version", "type": "string", "default": "1.0.0"},
                {"name": "timestamp", "type": "string"},
                {"name": "source", "type": "string"},
                {"name": "prices", "type": {
                    "type": "array",
                    "items": {
                        "type": "record",
                        "name": "FuelPrice",
                        "fields": [
                            {"name": "fuel_type", "type": "string"},
                            {"name": "region", "type": "string"},
                            {"name": "spot_price_usd_mmbtu", "type": "double"},
                            {"name": "price_source", "type": "string"},
                            {"name": "effective_date", "type": "string"},
                        ]
                    }
                }},
            ]
        },
        EventType.OPTIMIZATION_COMPLETE: {
            "type": "record",
            "name": "OptimizationCompleteEvent",
            "namespace": "io.greenlang.fuelcraft.events",
            "fields": [
                {"name": "event_id", "type": "string"},
                {"name": "event_type", "type": "string"},
                {"name": "schema_version", "type": "string", "default": "1.0.0"},
                {"name": "timestamp", "type": "string"},
                {"name": "run_id", "type": "string"},
                {"name": "site_id", "type": "string"},
                {"name": "total_cost_usd", "type": "double"},
                {"name": "total_emissions_mtco2e", "type": "double"},
                {"name": "cost_savings_percent", "type": "double"},
                {"name": "bundle_hash", "type": "string"},
            ]
        },
        EventType.AUDIT: {
            "type": "record",
            "name": "AuditEvent",
            "namespace": "io.greenlang.fuelcraft.events",
            "fields": [
                {"name": "event_id", "type": "string"},
                {"name": "event_type", "type": "string"},
                {"name": "schema_version", "type": "string", "default": "1.0.0"},
                {"name": "timestamp", "type": "string"},
                {"name": "action", "type": "string"},
                {"name": "resource_type", "type": "string"},
                {"name": "resource_id", "type": "string"},
                {"name": "user_id", "type": "string"},
                {"name": "success", "type": "boolean"},
                {"name": "computation_hash", "type": ["null", "string"], "default": None},
            ]
        },
    }

    return schemas.get(event_type, {})


def get_json_schema(event_type: EventType) -> Dict[str, Any]:
    """
    Get JSON Schema for event type.

    Args:
        event_type: Event type

    Returns:
        JSON Schema definition
    """
    schemas = {
        EventType.INVENTORY_UPDATE: {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "InventoryUpdateEvent",
            "type": "object",
            "required": ["header", "metadata", "readings"],
            "properties": {
                "header": {
                    "type": "object",
                    "required": ["event_id", "event_type", "timestamp", "source"],
                    "properties": {
                        "event_id": {"type": "string", "format": "uuid"},
                        "event_type": {"type": "string", "enum": ["inventory_update"]},
                        "schema_version": {"type": "string", "default": "1.0.0"},
                        "timestamp": {"type": "string", "format": "date-time"},
                        "source": {"type": "string"},
                    }
                },
                "metadata": {
                    "type": "object",
                    "required": ["site_id"],
                    "properties": {
                        "site_id": {"type": "string"},
                        "tenant_id": {"type": "string"},
                    }
                },
                "readings": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["tank_id", "fuel_type", "level_percent", "level_mmbtu"],
                        "properties": {
                            "tank_id": {"type": "string"},
                            "fuel_type": {"type": "string"},
                            "level_percent": {"type": "number", "minimum": 0, "maximum": 100},
                            "level_mmbtu": {"type": "number", "minimum": 0},
                            "temperature_c": {"type": "number"},
                            "data_quality": {"type": "string", "enum": ["good", "uncertain", "bad", "stale"]},
                        }
                    }
                },
                "total_inventory_mmbtu": {"type": "number", "minimum": 0},
                "total_capacity_mmbtu": {"type": "number", "minimum": 0},
            }
        },
        EventType.PRICE_UPDATE: {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "PriceUpdateEvent",
            "type": "object",
            "required": ["header", "metadata", "prices"],
            "properties": {
                "header": {"type": "object"},
                "metadata": {"type": "object"},
                "prices": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["fuel_type", "spot_price_usd_mmbtu"],
                        "properties": {
                            "fuel_type": {"type": "string"},
                            "spot_price_usd_mmbtu": {"type": "number", "minimum": 0},
                            "price_source": {"type": "string"},
                        }
                    }
                }
            }
        },
        EventType.OPTIMIZATION_COMPLETE: {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "OptimizationCompleteEvent",
            "type": "object",
            "required": ["header", "metadata", "result"],
            "properties": {
                "header": {"type": "object"},
                "metadata": {"type": "object"},
                "result": {
                    "type": "object",
                    "required": ["run_id", "total_cost_usd", "bundle_hash"],
                    "properties": {
                        "run_id": {"type": "string"},
                        "total_cost_usd": {"type": "number"},
                        "total_emissions_mtco2e": {"type": "number"},
                        "bundle_hash": {"type": "string"},
                    }
                }
            }
        },
        EventType.AUDIT: {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "AuditEvent",
            "type": "object",
            "required": ["header", "metadata", "action", "resource_type", "resource_id", "user_id"],
            "properties": {
                "header": {"type": "object"},
                "metadata": {"type": "object"},
                "action": {"type": "string", "enum": ["create", "update", "delete", "read", "execute", "approve", "reject"]},
                "resource_type": {"type": "string"},
                "resource_id": {"type": "string"},
                "user_id": {"type": "string"},
                "success": {"type": "boolean"},
                "computation_hash": {"type": "string"},
            }
        },
    }

    return schemas.get(event_type, {})
