"""
GL-003 UNIFIEDSTEAM - Kafka Schema Definitions

This module provides canonical Kafka schema definitions for steam system
data streaming, including raw signals, validated data, features, computed
properties, recommendations, and events.

Topic Naming Convention:
    gl003.<site>.<area>.<stream_type>

Stream Types:
    - raw: Raw OT signals with minimal processing
    - validated: Normalized data with quality flags
    - features: Engineered features for ML models
    - computed: Thermodynamic properties and KPIs
    - recommendations: Optimization recommendations
    - events: Maintenance events, alarms, setpoint changes

Author: GL-003 Data Engineering Team
Version: 1.0.0
"""

from .kafka_schemas import (
    # Raw signals
    RawSignalSchema,
    SensorQuality,
    SensorMetadata,
    # Validated signals
    ValidatedSignalSchema,
    ValidationStatus,
    QualityFlags,
    # Features
    FeatureSchema,
    TrapFeatureSchema,
    HeaderFeatureSchema,
    DesuperheaterFeatureSchema,
    # Computed
    ComputedPropertiesSchema,
    SteamPropertiesComputed,
    EnthalpyBalanceComputed,
    KPIComputed,
    # Recommendations
    RecommendationSchema,
    RecommendationType,
    RecommendationPriority,
    # Events
    EventSchema,
    EventType,
    AlarmSchema,
    MaintenanceEventSchema,
    SetpointChangeSchema,
)

from .avro_schemas import (
    get_raw_signal_avro,
    get_validated_signal_avro,
    get_feature_avro,
    get_computed_avro,
    get_recommendation_avro,
    get_event_avro,
)

from .schema_registry import (
    SchemaRegistry,
    SchemaVersion,
    SchemaCompatibility,
)

__all__ = [
    # Raw
    "RawSignalSchema",
    "SensorQuality",
    "SensorMetadata",
    # Validated
    "ValidatedSignalSchema",
    "ValidationStatus",
    "QualityFlags",
    # Features
    "FeatureSchema",
    "TrapFeatureSchema",
    "HeaderFeatureSchema",
    "DesuperheaterFeatureSchema",
    # Computed
    "ComputedPropertiesSchema",
    "SteamPropertiesComputed",
    "EnthalpyBalanceComputed",
    "KPIComputed",
    # Recommendations
    "RecommendationSchema",
    "RecommendationType",
    "RecommendationPriority",
    # Events
    "EventSchema",
    "EventType",
    "AlarmSchema",
    "MaintenanceEventSchema",
    "SetpointChangeSchema",
    # Avro
    "get_raw_signal_avro",
    "get_validated_signal_avro",
    "get_feature_avro",
    "get_computed_avro",
    "get_recommendation_avro",
    "get_event_avro",
    # Registry
    "SchemaRegistry",
    "SchemaVersion",
    "SchemaCompatibility",
]
