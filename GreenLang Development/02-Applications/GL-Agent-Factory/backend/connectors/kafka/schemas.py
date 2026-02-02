"""
Avro and JSON Schema Definitions for Kafka Events.

This module provides schema definitions for agent events, supporting both
Avro (for Schema Registry) and JSON Schema (for validation) formats.

Features:
- Avro schema definitions for all event types
- JSON Schema equivalents for validation
- Schema versioning and evolution support
- Schema Registry integration helpers
- Type-safe schema generation from Pydantic models

Schema Evolution Rules:
- Adding optional fields is backward compatible
- Removing optional fields is forward compatible
- Changing field types requires new schema version
- Renaming fields requires aliases

Usage:
    from connectors.kafka.schemas import (
        AgentEventSchemas,
        get_avro_schema,
        validate_json_schema,
    )

    # Get Avro schema for calculation events
    schema = AgentEventSchemas.CALCULATION_COMPLETED_AVRO

    # Validate event against JSON schema
    is_valid = validate_json_schema(event_data, "calculation_completed")
"""

import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# =============================================================================
# Schema Version Management
# =============================================================================


class SchemaVersion(BaseModel):
    """Schema version information."""

    major: int = 1
    minor: int = 0
    patch: int = 0

    @property
    def version_string(self) -> str:
        """Get version as string."""
        return f"{self.major}.{self.minor}.{self.patch}"

    def is_compatible_with(self, other: "SchemaVersion") -> bool:
        """Check if this version is compatible with another."""
        # Same major version means compatible
        return self.major == other.major


# =============================================================================
# Avro Schema Namespace
# =============================================================================

AVRO_NAMESPACE = "com.greenlang.agent.events"


# =============================================================================
# Common Avro Types
# =============================================================================

AVRO_TIMESTAMP_LOGICAL_TYPE = {
    "type": "long",
    "logicalType": "timestamp-millis",
}

AVRO_UUID_LOGICAL_TYPE = {
    "type": "string",
    "logicalType": "uuid",
}

AVRO_DECIMAL_LOGICAL_TYPE = {
    "type": "bytes",
    "logicalType": "decimal",
    "precision": 18,
    "scale": 6,
}


# =============================================================================
# Event Metadata Schema
# =============================================================================

EVENT_METADATA_AVRO_SCHEMA = {
    "type": "record",
    "name": "EventMetadata",
    "namespace": AVRO_NAMESPACE,
    "doc": "Metadata for event tracking and correlation",
    "fields": [
        {
            "name": "correlation_id",
            "type": ["null", "string"],
            "default": None,
            "doc": "ID for correlating related events",
        },
        {
            "name": "causation_id",
            "type": ["null", "string"],
            "default": None,
            "doc": "ID of the event that caused this event",
        },
        {
            "name": "source_system",
            "type": "string",
            "default": "gl-agent-factory",
            "doc": "System that generated the event",
        },
        {
            "name": "source_version",
            "type": "string",
            "default": "1.0.0",
            "doc": "Version of the source system",
        },
        {
            "name": "tenant_id",
            "type": ["null", "string"],
            "default": None,
            "doc": "Tenant identifier",
        },
        {
            "name": "user_id",
            "type": ["null", "string"],
            "default": None,
            "doc": "User who triggered the event",
        },
        {
            "name": "trace_id",
            "type": ["null", "string"],
            "default": None,
            "doc": "Distributed tracing ID",
        },
        {
            "name": "span_id",
            "type": ["null", "string"],
            "default": None,
            "doc": "Span ID within trace",
        },
        {
            "name": "tags",
            "type": {
                "type": "map",
                "values": "string",
            },
            "default": {},
            "doc": "Additional tags for filtering",
        },
    ],
}


# =============================================================================
# Base Agent Event Schema
# =============================================================================

AGENT_EVENT_AVRO_SCHEMA = {
    "type": "record",
    "name": "AgentEvent",
    "namespace": AVRO_NAMESPACE,
    "doc": "Base schema for all agent events",
    "fields": [
        {
            "name": "event_id",
            "type": "string",
            "doc": "Unique event identifier (UUID)",
        },
        {
            "name": "event_type",
            "type": "string",
            "doc": "Type of event",
        },
        {
            "name": "event_version",
            "type": "string",
            "default": "1.0",
            "doc": "Schema version for event evolution",
        },
        {
            "name": "agent_id",
            "type": "string",
            "doc": "Agent that generated the event",
        },
        {
            "name": "timestamp",
            "type": AVRO_TIMESTAMP_LOGICAL_TYPE,
            "doc": "Event timestamp (UTC milliseconds)",
        },
        {
            "name": "payload",
            "type": {
                "type": "map",
                "values": ["null", "boolean", "int", "long", "float", "double", "string"],
            },
            "default": {},
            "doc": "Event-specific payload data",
        },
        {
            "name": "provenance_hash",
            "type": ["null", "string"],
            "default": None,
            "doc": "SHA-256 hash of event data",
        },
        {
            "name": "metadata",
            "type": EVENT_METADATA_AVRO_SCHEMA,
            "doc": "Event metadata",
        },
    ],
}


# =============================================================================
# Calculation Event Schemas
# =============================================================================

CALCULATION_INPUT_AVRO_SCHEMA = {
    "type": "record",
    "name": "CalculationInput",
    "namespace": AVRO_NAMESPACE,
    "doc": "Input data for a calculation",
    "fields": [
        {
            "name": "parameters",
            "type": {
                "type": "map",
                "values": ["null", "boolean", "int", "long", "float", "double", "string"],
            },
            "doc": "Calculation input parameters",
        },
        {
            "name": "input_hash",
            "type": "string",
            "doc": "SHA-256 hash of input parameters",
        },
    ],
}

CALCULATION_OUTPUT_AVRO_SCHEMA = {
    "type": "record",
    "name": "CalculationOutput",
    "namespace": AVRO_NAMESPACE,
    "doc": "Output data from a calculation",
    "fields": [
        {
            "name": "result",
            "type": {
                "type": "map",
                "values": ["null", "boolean", "int", "long", "float", "double", "string"],
            },
            "doc": "Calculation result",
        },
        {
            "name": "output_hash",
            "type": "string",
            "doc": "SHA-256 hash of output",
        },
        {
            "name": "unit",
            "type": ["null", "string"],
            "default": None,
            "doc": "Unit of measurement (QUDT)",
        },
    ],
}

CALCULATION_COMPLETED_AVRO_SCHEMA = {
    "type": "record",
    "name": "AgentCalculationCompleted",
    "namespace": AVRO_NAMESPACE,
    "doc": "Event emitted when an agent completes a calculation",
    "fields": [
        {
            "name": "event_id",
            "type": "string",
            "doc": "Unique event identifier",
        },
        {
            "name": "event_type",
            "type": "string",
            "default": "agent.calculation.completed",
            "doc": "Event type",
        },
        {
            "name": "event_version",
            "type": "string",
            "default": "1.0",
            "doc": "Schema version",
        },
        {
            "name": "agent_id",
            "type": "string",
            "doc": "Agent identifier",
        },
        {
            "name": "timestamp",
            "type": AVRO_TIMESTAMP_LOGICAL_TYPE,
            "doc": "Event timestamp",
        },
        {
            "name": "provenance_hash",
            "type": ["null", "string"],
            "default": None,
            "doc": "Event provenance hash",
        },
        {
            "name": "metadata",
            "type": EVENT_METADATA_AVRO_SCHEMA,
            "doc": "Event metadata",
        },
        {
            "name": "calculation_type",
            "type": "string",
            "doc": "Type of calculation performed",
        },
        {
            "name": "formula_id",
            "type": "string",
            "doc": "Identifier of the formula used",
        },
        {
            "name": "formula_version",
            "type": "string",
            "default": "1.0",
            "doc": "Version of the formula",
        },
        {
            "name": "input_data",
            "type": CALCULATION_INPUT_AVRO_SCHEMA,
            "doc": "Calculation input with hash",
        },
        {
            "name": "output_data",
            "type": CALCULATION_OUTPUT_AVRO_SCHEMA,
            "doc": "Calculation output with hash",
        },
        {
            "name": "calculation_chain_hash",
            "type": "string",
            "doc": "Hash of complete calculation chain",
        },
        {
            "name": "processing_time_ms",
            "type": "double",
            "doc": "Processing time in milliseconds",
        },
        {
            "name": "emission_factor_source",
            "type": ["null", "string"],
            "default": None,
            "doc": "Source of emission factors",
        },
        {
            "name": "uncertainty_percent",
            "type": ["null", "double"],
            "default": None,
            "doc": "Uncertainty percentage",
        },
    ],
}


# =============================================================================
# Alert Event Schemas
# =============================================================================

ALERT_SEVERITY_ENUM = {
    "type": "enum",
    "name": "AlertSeverity",
    "namespace": AVRO_NAMESPACE,
    "symbols": ["info", "warning", "error", "critical"],
}

ALERT_CONTEXT_AVRO_SCHEMA = {
    "type": "record",
    "name": "AlertContext",
    "namespace": AVRO_NAMESPACE,
    "doc": "Context information for an alert",
    "fields": [
        {
            "name": "threshold",
            "type": ["null", "double"],
            "default": None,
            "doc": "Threshold that was exceeded",
        },
        {
            "name": "actual_value",
            "type": ["null", "double"],
            "default": None,
            "doc": "Actual value that triggered the alert",
        },
        {
            "name": "measurement_unit",
            "type": ["null", "string"],
            "default": None,
            "doc": "Unit of measurement",
        },
        {
            "name": "related_entity_id",
            "type": ["null", "string"],
            "default": None,
            "doc": "ID of related entity",
        },
        {
            "name": "location",
            "type": ["null", "string"],
            "default": None,
            "doc": "Physical or logical location",
        },
    ],
}

ALERT_RAISED_AVRO_SCHEMA = {
    "type": "record",
    "name": "AgentAlertRaised",
    "namespace": AVRO_NAMESPACE,
    "doc": "Event emitted when an agent raises an alert",
    "fields": [
        {
            "name": "event_id",
            "type": "string",
            "doc": "Unique event identifier",
        },
        {
            "name": "event_type",
            "type": "string",
            "default": "agent.alert.raised",
            "doc": "Event type",
        },
        {
            "name": "event_version",
            "type": "string",
            "default": "1.0",
            "doc": "Schema version",
        },
        {
            "name": "agent_id",
            "type": "string",
            "doc": "Agent identifier",
        },
        {
            "name": "timestamp",
            "type": AVRO_TIMESTAMP_LOGICAL_TYPE,
            "doc": "Event timestamp",
        },
        {
            "name": "provenance_hash",
            "type": ["null", "string"],
            "default": None,
            "doc": "Event provenance hash",
        },
        {
            "name": "metadata",
            "type": EVENT_METADATA_AVRO_SCHEMA,
            "doc": "Event metadata",
        },
        {
            "name": "severity",
            "type": ALERT_SEVERITY_ENUM,
            "doc": "Alert severity level",
        },
        {
            "name": "alert_code",
            "type": "string",
            "doc": "Unique alert code",
        },
        {
            "name": "title",
            "type": "string",
            "doc": "Short alert title",
        },
        {
            "name": "description",
            "type": "string",
            "doc": "Detailed alert description",
        },
        {
            "name": "context",
            "type": ALERT_CONTEXT_AVRO_SCHEMA,
            "doc": "Alert context information",
        },
        {
            "name": "recommended_actions",
            "type": {
                "type": "array",
                "items": "string",
            },
            "default": [],
            "doc": "List of recommended actions",
        },
        {
            "name": "auto_resolvable",
            "type": "boolean",
            "default": False,
            "doc": "Whether the alert can auto-resolve",
        },
        {
            "name": "expiry_time",
            "type": ["null", AVRO_TIMESTAMP_LOGICAL_TYPE],
            "default": None,
            "doc": "When the alert expires",
        },
        {
            "name": "related_alerts",
            "type": {
                "type": "array",
                "items": "string",
            },
            "default": [],
            "doc": "IDs of related alerts",
        },
    ],
}


# =============================================================================
# Recommendation Event Schemas
# =============================================================================

RECOMMENDATION_PRIORITY_ENUM = {
    "type": "enum",
    "name": "RecommendationPriority",
    "namespace": AVRO_NAMESPACE,
    "symbols": ["low", "medium", "high", "urgent"],
}

RECOMMENDATION_CATEGORY_ENUM = {
    "type": "enum",
    "name": "RecommendationCategory",
    "namespace": AVRO_NAMESPACE,
    "symbols": [
        "energy_efficiency",
        "emissions_reduction",
        "cost_optimization",
        "compliance",
        "safety",
        "maintenance",
        "process_improvement",
    ],
}

RECOMMENDATION_IMPACT_AVRO_SCHEMA = {
    "type": "record",
    "name": "RecommendationImpact",
    "namespace": AVRO_NAMESPACE,
    "doc": "Estimated impact of implementing a recommendation",
    "fields": [
        {
            "name": "energy_savings_percent",
            "type": ["null", "double"],
            "default": None,
            "doc": "Estimated energy savings (%)",
        },
        {
            "name": "emissions_reduction_kgco2e",
            "type": ["null", "double"],
            "default": None,
            "doc": "Estimated emissions reduction (kg CO2e)",
        },
        {
            "name": "cost_savings_usd",
            "type": ["null", "double"],
            "default": None,
            "doc": "Estimated cost savings (USD)",
        },
        {
            "name": "payback_period_months",
            "type": ["null", "int"],
            "default": None,
            "doc": "Estimated payback period (months)",
        },
        {
            "name": "implementation_cost_usd",
            "type": ["null", "double"],
            "default": None,
            "doc": "Estimated implementation cost (USD)",
        },
        {
            "name": "confidence_score",
            "type": "double",
            "default": 0.8,
            "doc": "Confidence in the estimate (0-1)",
        },
    ],
}

RECOMMENDATION_GENERATED_AVRO_SCHEMA = {
    "type": "record",
    "name": "AgentRecommendationGenerated",
    "namespace": AVRO_NAMESPACE,
    "doc": "Event emitted when an agent generates a recommendation",
    "fields": [
        {
            "name": "event_id",
            "type": "string",
            "doc": "Unique event identifier",
        },
        {
            "name": "event_type",
            "type": "string",
            "default": "agent.recommendation.generated",
            "doc": "Event type",
        },
        {
            "name": "event_version",
            "type": "string",
            "default": "1.0",
            "doc": "Schema version",
        },
        {
            "name": "agent_id",
            "type": "string",
            "doc": "Agent identifier",
        },
        {
            "name": "timestamp",
            "type": AVRO_TIMESTAMP_LOGICAL_TYPE,
            "doc": "Event timestamp",
        },
        {
            "name": "provenance_hash",
            "type": ["null", "string"],
            "default": None,
            "doc": "Event provenance hash",
        },
        {
            "name": "metadata",
            "type": EVENT_METADATA_AVRO_SCHEMA,
            "doc": "Event metadata",
        },
        {
            "name": "priority",
            "type": RECOMMENDATION_PRIORITY_ENUM,
            "doc": "Recommendation priority",
        },
        {
            "name": "category",
            "type": RECOMMENDATION_CATEGORY_ENUM,
            "doc": "Recommendation category",
        },
        {
            "name": "title",
            "type": "string",
            "doc": "Short recommendation title",
        },
        {
            "name": "description",
            "type": "string",
            "doc": "Detailed recommendation description",
        },
        {
            "name": "rationale",
            "type": "string",
            "doc": "Rationale for the recommendation",
        },
        {
            "name": "impact",
            "type": RECOMMENDATION_IMPACT_AVRO_SCHEMA,
            "doc": "Estimated impact",
        },
        {
            "name": "implementation_steps",
            "type": {
                "type": "array",
                "items": "string",
            },
            "default": [],
            "doc": "Implementation steps",
        },
        {
            "name": "prerequisites",
            "type": {
                "type": "array",
                "items": "string",
            },
            "default": [],
            "doc": "Prerequisites for implementation",
        },
        {
            "name": "related_equipment_ids",
            "type": {
                "type": "array",
                "items": "string",
            },
            "default": [],
            "doc": "IDs of related equipment",
        },
        {
            "name": "expiry_date",
            "type": ["null", AVRO_TIMESTAMP_LOGICAL_TYPE],
            "default": None,
            "doc": "When the recommendation expires",
        },
    ],
}


# =============================================================================
# Health Check Event Schemas
# =============================================================================

HEALTH_STATUS_ENUM = {
    "type": "enum",
    "name": "HealthStatus",
    "namespace": AVRO_NAMESPACE,
    "symbols": ["healthy", "degraded", "unhealthy", "unknown"],
}

HEALTH_METRICS_AVRO_SCHEMA = {
    "type": "record",
    "name": "HealthMetrics",
    "namespace": AVRO_NAMESPACE,
    "doc": "Health metrics for an agent",
    "fields": [
        {
            "name": "response_time_ms",
            "type": "double",
            "doc": "Response time in milliseconds",
        },
        {
            "name": "memory_usage_mb",
            "type": ["null", "double"],
            "default": None,
            "doc": "Memory usage in megabytes",
        },
        {
            "name": "cpu_usage_percent",
            "type": ["null", "double"],
            "default": None,
            "doc": "CPU usage percentage",
        },
        {
            "name": "error_rate_percent",
            "type": "double",
            "default": 0,
            "doc": "Error rate percentage",
        },
        {
            "name": "throughput_per_minute",
            "type": ["null", "double"],
            "default": None,
            "doc": "Requests per minute",
        },
        {
            "name": "queue_depth",
            "type": ["null", "int"],
            "default": None,
            "doc": "Current queue depth",
        },
        {
            "name": "last_successful_execution",
            "type": ["null", AVRO_TIMESTAMP_LOGICAL_TYPE],
            "default": None,
            "doc": "Last successful execution timestamp",
        },
    ],
}

DEPENDENCY_HEALTH_AVRO_SCHEMA = {
    "type": "record",
    "name": "DependencyHealth",
    "namespace": AVRO_NAMESPACE,
    "doc": "Health status of a dependency",
    "fields": [
        {
            "name": "dependency_name",
            "type": "string",
            "doc": "Name of the dependency",
        },
        {
            "name": "status",
            "type": HEALTH_STATUS_ENUM,
            "doc": "Health status",
        },
        {
            "name": "latency_ms",
            "type": ["null", "double"],
            "default": None,
            "doc": "Latency in milliseconds",
        },
        {
            "name": "error_message",
            "type": ["null", "string"],
            "default": None,
            "doc": "Error message if unhealthy",
        },
    ],
}

HEALTH_CHECK_AVRO_SCHEMA = {
    "type": "record",
    "name": "AgentHealthCheck",
    "namespace": AVRO_NAMESPACE,
    "doc": "Event emitted for agent health monitoring",
    "fields": [
        {
            "name": "event_id",
            "type": "string",
            "doc": "Unique event identifier",
        },
        {
            "name": "event_type",
            "type": "string",
            "default": "agent.health.check",
            "doc": "Event type",
        },
        {
            "name": "event_version",
            "type": "string",
            "default": "1.0",
            "doc": "Schema version",
        },
        {
            "name": "agent_id",
            "type": "string",
            "doc": "Agent identifier",
        },
        {
            "name": "timestamp",
            "type": AVRO_TIMESTAMP_LOGICAL_TYPE,
            "doc": "Event timestamp",
        },
        {
            "name": "provenance_hash",
            "type": ["null", "string"],
            "default": None,
            "doc": "Event provenance hash",
        },
        {
            "name": "metadata",
            "type": EVENT_METADATA_AVRO_SCHEMA,
            "doc": "Event metadata",
        },
        {
            "name": "status",
            "type": HEALTH_STATUS_ENUM,
            "doc": "Overall health status",
        },
        {
            "name": "metrics",
            "type": HEALTH_METRICS_AVRO_SCHEMA,
            "doc": "Health metrics",
        },
        {
            "name": "dependencies",
            "type": {
                "type": "array",
                "items": DEPENDENCY_HEALTH_AVRO_SCHEMA,
            },
            "default": [],
            "doc": "Health of dependencies",
        },
        {
            "name": "agent_version",
            "type": "string",
            "doc": "Agent version",
        },
        {
            "name": "config_valid",
            "type": "boolean",
            "default": True,
            "doc": "Whether configuration is valid",
        },
        {
            "name": "issues",
            "type": {
                "type": "array",
                "items": "string",
            },
            "default": [],
            "doc": "List of issues if unhealthy",
        },
        {
            "name": "last_config_update",
            "type": ["null", AVRO_TIMESTAMP_LOGICAL_TYPE],
            "default": None,
            "doc": "Last configuration update timestamp",
        },
    ],
}


# =============================================================================
# Configuration Event Schemas
# =============================================================================

CONFIG_CHANGE_AVRO_SCHEMA = {
    "type": "record",
    "name": "ConfigChange",
    "namespace": AVRO_NAMESPACE,
    "doc": "Details of a configuration change",
    "fields": [
        {
            "name": "field_path",
            "type": "string",
            "doc": "Path to the changed field",
        },
        {
            "name": "old_value",
            "type": ["null", "boolean", "int", "long", "float", "double", "string"],
            "default": None,
            "doc": "Previous value",
        },
        {
            "name": "new_value",
            "type": ["null", "boolean", "int", "long", "float", "double", "string"],
            "doc": "New value",
        },
        {
            "name": "change_type",
            "type": "string",
            "default": "update",
            "doc": "Type of change: create, update, delete",
        },
    ],
}

CONFIG_CHANGED_AVRO_SCHEMA = {
    "type": "record",
    "name": "AgentConfigurationChanged",
    "namespace": AVRO_NAMESPACE,
    "doc": "Event emitted when agent configuration changes",
    "fields": [
        {
            "name": "event_id",
            "type": "string",
            "doc": "Unique event identifier",
        },
        {
            "name": "event_type",
            "type": "string",
            "default": "agent.config.changed",
            "doc": "Event type",
        },
        {
            "name": "event_version",
            "type": "string",
            "default": "1.0",
            "doc": "Schema version",
        },
        {
            "name": "agent_id",
            "type": "string",
            "doc": "Agent identifier",
        },
        {
            "name": "timestamp",
            "type": AVRO_TIMESTAMP_LOGICAL_TYPE,
            "doc": "Event timestamp",
        },
        {
            "name": "provenance_hash",
            "type": ["null", "string"],
            "default": None,
            "doc": "Event provenance hash",
        },
        {
            "name": "metadata",
            "type": EVENT_METADATA_AVRO_SCHEMA,
            "doc": "Event metadata",
        },
        {
            "name": "changes",
            "type": {
                "type": "array",
                "items": CONFIG_CHANGE_AVRO_SCHEMA,
            },
            "doc": "List of configuration changes",
        },
        {
            "name": "config_version",
            "type": "string",
            "doc": "New configuration version",
        },
        {
            "name": "previous_config_version",
            "type": ["null", "string"],
            "default": None,
            "doc": "Previous configuration version",
        },
        {
            "name": "changed_by",
            "type": "string",
            "doc": "User or system that made the change",
        },
        {
            "name": "change_reason",
            "type": "string",
            "doc": "Reason for the change",
        },
        {
            "name": "validated",
            "type": "boolean",
            "default": True,
            "doc": "Whether the new config passed validation",
        },
        {
            "name": "validation_errors",
            "type": {
                "type": "array",
                "items": "string",
            },
            "default": [],
            "doc": "Validation errors if any",
        },
        {
            "name": "rollback_available",
            "type": "boolean",
            "default": True,
            "doc": "Whether rollback is available",
        },
        {
            "name": "effective_from",
            "type": AVRO_TIMESTAMP_LOGICAL_TYPE,
            "doc": "When the change becomes effective",
        },
    ],
}


# =============================================================================
# JSON Schema Definitions
# =============================================================================


AGENT_EVENT_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "https://greenlang.io/schemas/agent-event.json",
    "title": "AgentEvent",
    "description": "Base schema for all agent events",
    "type": "object",
    "properties": {
        "event_id": {
            "type": "string",
            "format": "uuid",
            "description": "Unique event identifier",
        },
        "event_type": {
            "type": "string",
            "description": "Type of event",
        },
        "event_version": {
            "type": "string",
            "default": "1.0",
            "description": "Schema version",
        },
        "agent_id": {
            "type": "string",
            "description": "Agent that generated the event",
        },
        "timestamp": {
            "type": "string",
            "format": "date-time",
            "description": "Event timestamp (ISO 8601)",
        },
        "payload": {
            "type": "object",
            "description": "Event-specific payload data",
        },
        "provenance_hash": {
            "type": ["string", "null"],
            "pattern": "^[a-f0-9]{64}$",
            "description": "SHA-256 hash of event data",
        },
        "metadata": {
            "$ref": "#/definitions/EventMetadata",
        },
    },
    "required": ["event_id", "event_type", "agent_id", "timestamp"],
    "definitions": {
        "EventMetadata": {
            "type": "object",
            "properties": {
                "correlation_id": {"type": ["string", "null"]},
                "causation_id": {"type": ["string", "null"]},
                "source_system": {"type": "string", "default": "gl-agent-factory"},
                "source_version": {"type": "string", "default": "1.0.0"},
                "tenant_id": {"type": ["string", "null"]},
                "user_id": {"type": ["string", "null"]},
                "trace_id": {"type": ["string", "null"]},
                "span_id": {"type": ["string", "null"]},
                "tags": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
            },
        },
    },
}


# =============================================================================
# Schema Registry Helpers
# =============================================================================


class AgentEventSchemas:
    """Container for all agent event schemas."""

    # Avro schemas
    EVENT_METADATA_AVRO = EVENT_METADATA_AVRO_SCHEMA
    AGENT_EVENT_AVRO = AGENT_EVENT_AVRO_SCHEMA
    CALCULATION_COMPLETED_AVRO = CALCULATION_COMPLETED_AVRO_SCHEMA
    ALERT_RAISED_AVRO = ALERT_RAISED_AVRO_SCHEMA
    RECOMMENDATION_GENERATED_AVRO = RECOMMENDATION_GENERATED_AVRO_SCHEMA
    HEALTH_CHECK_AVRO = HEALTH_CHECK_AVRO_SCHEMA
    CONFIG_CHANGED_AVRO = CONFIG_CHANGED_AVRO_SCHEMA

    # JSON schemas
    AGENT_EVENT_JSON = AGENT_EVENT_JSON_SCHEMA

    @classmethod
    def get_avro_schema(cls, event_type: str) -> Dict[str, Any]:
        """
        Get Avro schema for an event type.

        Args:
            event_type: Event type string

        Returns:
            Avro schema dictionary

        Raises:
            ValueError: If event type is unknown
        """
        schema_map = {
            "agent.calculation.completed": cls.CALCULATION_COMPLETED_AVRO,
            "agent.alert.raised": cls.ALERT_RAISED_AVRO,
            "agent.recommendation.generated": cls.RECOMMENDATION_GENERATED_AVRO,
            "agent.health.check": cls.HEALTH_CHECK_AVRO,
            "agent.config.changed": cls.CONFIG_CHANGED_AVRO,
        }

        if event_type not in schema_map:
            # Return base event schema for unknown types
            return cls.AGENT_EVENT_AVRO

        return schema_map[event_type]

    @classmethod
    def get_schema_json(cls, event_type: str) -> str:
        """
        Get Avro schema as JSON string.

        Args:
            event_type: Event type string

        Returns:
            JSON string of schema
        """
        schema = cls.get_avro_schema(event_type)
        return json.dumps(schema, indent=2)

    @classmethod
    def get_all_schemas(cls) -> Dict[str, Dict[str, Any]]:
        """Get all Avro schemas."""
        return {
            "agent.calculation.completed": cls.CALCULATION_COMPLETED_AVRO,
            "agent.alert.raised": cls.ALERT_RAISED_AVRO,
            "agent.recommendation.generated": cls.RECOMMENDATION_GENERATED_AVRO,
            "agent.health.check": cls.HEALTH_CHECK_AVRO,
            "agent.config.changed": cls.CONFIG_CHANGED_AVRO,
            "agent.event.base": cls.AGENT_EVENT_AVRO,
        }


def get_avro_schema(event_type: str) -> Dict[str, Any]:
    """
    Get Avro schema for an event type.

    Args:
        event_type: Event type string

    Returns:
        Avro schema dictionary
    """
    return AgentEventSchemas.get_avro_schema(event_type)


def validate_json_schema(data: Dict[str, Any], schema_name: str) -> bool:
    """
    Validate data against a JSON schema.

    Args:
        data: Data to validate
        schema_name: Name of the schema

    Returns:
        True if valid, False otherwise
    """
    try:
        import jsonschema

        schema = AGENT_EVENT_JSON_SCHEMA
        jsonschema.validate(instance=data, schema=schema)
        return True
    except ImportError:
        logger.warning("jsonschema not installed, skipping validation")
        return True
    except Exception as e:
        logger.error(f"JSON schema validation failed: {e}")
        return False


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Schema version
    "SchemaVersion",
    "AVRO_NAMESPACE",
    # Avro schemas
    "EVENT_METADATA_AVRO_SCHEMA",
    "AGENT_EVENT_AVRO_SCHEMA",
    "CALCULATION_COMPLETED_AVRO_SCHEMA",
    "ALERT_RAISED_AVRO_SCHEMA",
    "RECOMMENDATION_GENERATED_AVRO_SCHEMA",
    "HEALTH_CHECK_AVRO_SCHEMA",
    "CONFIG_CHANGED_AVRO_SCHEMA",
    # JSON schemas
    "AGENT_EVENT_JSON_SCHEMA",
    # Helper classes
    "AgentEventSchemas",
    # Functions
    "get_avro_schema",
    "validate_json_schema",
]
