"""
GL-001 ThermalCommand: Data Contracts Package

This package provides canonical data contracts, schemas, and validation for the
ThermalCommand ProcessHeatOrchestrator system.

Package Contents:

1. domain_schemas.py - Canonical schemas for all data domains:
   - ProcessSensorData: SCADA sensor measurements
   - EnergyConsumptionData: Fuel, electricity, steam data
   - SafetySystemStatus: SIS permissives and trip statuses
   - ProductionSchedule: Campaigns, batches, and targets
   - WeatherForecast: Weather data with uncertainty
   - EnergyPrices: Day-ahead and real-time pricing
   - EquipmentHealth: Vibration, fouling, RUL predictions
   - AlarmState: Alarm management per ISA-18.2

2. tag_dictionary.py - Minimum data dictionary with:
   - Standardized tag naming (e.g., steam.headerA.pressure)
   - Engineering unit definitions and conversions
   - Tag validation rules and limits
   - SCADA/OPC UA mapping

3. data_quality.py - Data quality validation:
   - Time synchronization validation (NTP/PTP)
   - Completeness checking
   - Validity validation
   - Truth label handling
   - Lineage tracking with SHA-256 hashes

4. schema_registry.py - Schema version management:
   - Semantic versioning
   - Compatibility checking (forward/backward)
   - Schema migration support
   - Runtime validation

Usage Example:

    from data_contracts import (
        # Domain schemas
        ProcessSensorData,
        EnergyConsumptionData,
        SafetySystemStatus,

        # Tag dictionary
        get_tag_dictionary,
        TagDefinition,

        # Data quality
        get_quality_manager,
        DataQualityManager,

        # Schema registry
        get_schema_registry,
        SchemaRegistry,
    )

    # Create sensor data
    sensor_data = ProcessSensorData(
        facility_id="PLANT-001",
        timestamp=datetime.utcnow(),
        steam_headers={"headerA": SteamHeaderData(...)},
    )

    # Validate using tag dictionary
    tag_dict = get_tag_dictionary()
    is_valid, error = tag_dict.validate_value(
        "steam.headerA.pressure",
        sensor_data.steam_headers["headerA"].pressure_barg
    )

    # Check data quality
    quality_manager = get_quality_manager()
    validation_result = quality_manager.validate_record(
        sensor_data.model_dump(),
        "ProcessSensorData"
    )

    # Use schema registry
    registry = get_schema_registry()
    schema = registry.get_schema("ProcessSensorData")

Standards Compliance:
- ISO 50001 Energy Management
- IEC 62443 Industrial Cybersecurity
- ISA-95 Enterprise-Control Integration
- ISA-18.2 Alarm Management
- ASME PTC 4.1 Steam Generators
- JSON Schema Draft 2020-12

Author: GreenLang Data Integration Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "GreenLang Data Integration Team"

# =============================================================================
# Domain Schemas
# =============================================================================

from .domain_schemas import (
    # Enums
    DataQualityLevel,
    UnitSystem,
    AlarmSeverity,
    EquipmentStatus,
    TripStatus,
    ForecastConfidence,
    PriceMarket,
    FuelType,
    # Base models
    ProvenanceInfo,
    DataQualityMetrics,
    BaseDataContract,
    # Domain schemas
    ProcessSensorData,
    EnergyConsumptionData,
    SafetySystemStatus,
    ProductionSchedule,
    WeatherForecast,
    EnergyPrices,
    EquipmentHealth,
    AlarmState,
    # Sub-schemas
    SteamHeaderData,
    ValvePosition,
    FuelConsumption,
    BoilerPerformance,
    SISPermissive,
    TripPoint,
    BypassRecord,
    BatchPlan,
    UnitTarget,
    Campaign,
    HourlyForecast,
    ForecastUncertainty,
    ElectricityPrice,
    FuelPrice,
    TariffPeriod,
    VibrationData,
    LubeOilAnalysis,
    FoulingIndicator,
    RemainingUsefulLife,
    AlarmRecord,
    AlarmStatistics,
    # Registries
    DOMAIN_SCHEMAS,
    SUB_SCHEMAS,
)

# =============================================================================
# Tag Dictionary
# =============================================================================

from .tag_dictionary import (
    # Enums
    TagDataType,
    TagCategory,
    UnitCategory,
    QualityCode,
    # Models
    TagDefinition,
    TagValue,
    # Classes
    TagDictionary,
    UnitConversion,
    # Functions
    get_tag_dictionary,
)

# =============================================================================
# Data Quality
# =============================================================================

from .data_quality import (
    # Enums
    ValidationSeverity,
    TimeSource,
    DataLineageType,
    TruthLabelStatus,
    # Validators
    TimeSyncValidator,
    CompletenessValidator,
    ValidityValidator,
    # Lineage
    LineageNode,
    LineageTracker,
    # Truth labels
    TruthLabel,
    TruthLabelHandler,
    # Scoring
    DataQualityScorer,
    # Manager
    DataQualityManager,
    # Functions
    get_quality_manager,
)

# =============================================================================
# Schema Registry
# =============================================================================

from .schema_registry import (
    # Enums
    CompatibilityMode,
    SchemaStatus,
    ChangeType,
    # Models
    SemanticVersion,
    SchemaChange,
    SchemaDefinition,
    SchemaMigration,
    # Classes
    CompatibilityChecker,
    SchemaMigrator,
    SchemaRegistry,
    # Functions
    get_schema_registry,
)

# =============================================================================
# Package-Level Utilities
# =============================================================================


def validate_record(
    data: dict,
    schema_name: str,
    strict: bool = True
) -> dict:
    """
    Convenience function to validate a data record.

    Args:
        data: Data record to validate
        schema_name: Schema name to validate against
        strict: If True, raise exception on failure

    Returns:
        Validation result dictionary

    Raises:
        ValueError: If strict=True and validation fails
    """
    from datetime import datetime, timezone

    quality_mgr = get_quality_manager()

    timestamp = data.get("timestamp")
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    elif timestamp is None:
        timestamp = datetime.now(timezone.utc)

    result = quality_mgr.validate_record(data, schema_name, timestamp)

    if strict and result["validation_status"] == "failed":
        raise ValueError(
            f"Validation failed for {schema_name}: "
            f"score={result['quality_report']['total_score']}"
        )

    return result


def create_data_record(
    schema_class: type,
    source_system: str = "unknown",
    **kwargs
) -> BaseDataContract:
    """
    Create a data record with automatic provenance.

    Args:
        schema_class: Pydantic schema class
        source_system: Source system identifier
        **kwargs: Field values

    Returns:
        Instantiated schema with provenance
    """
    from datetime import datetime, timezone

    # Add provenance if not provided
    if "provenance" not in kwargs:
        kwargs["provenance"] = ProvenanceInfo(
            source_system=source_system,
            timestamp_collected=kwargs.get("timestamp", datetime.now(timezone.utc)),
        )

    # Ensure timestamp
    if "timestamp" not in kwargs:
        kwargs["timestamp"] = datetime.now(timezone.utc)

    return schema_class(**kwargs)


def get_schema_json(schema_name: str, version: str = None) -> dict:
    """
    Get JSON Schema for a domain schema.

    Args:
        schema_name: Schema name
        version: Optional version

    Returns:
        JSON Schema dictionary
    """
    registry = get_schema_registry()
    schema_def = registry.get_schema(schema_name, version)

    if schema_def:
        return schema_def.schema_content

    # Try to get from domain schemas
    if schema_name in DOMAIN_SCHEMAS:
        return DOMAIN_SCHEMAS[schema_name].model_json_schema()

    raise ValueError(f"Schema not found: {schema_name}")


# =============================================================================
# All Exports
# =============================================================================

__all__ = [
    # Version
    "__version__",
    "__author__",

    # Domain Schema Enums
    "DataQualityLevel",
    "UnitSystem",
    "AlarmSeverity",
    "EquipmentStatus",
    "TripStatus",
    "ForecastConfidence",
    "PriceMarket",
    "FuelType",

    # Base Models
    "ProvenanceInfo",
    "DataQualityMetrics",
    "BaseDataContract",

    # Domain Schemas
    "ProcessSensorData",
    "EnergyConsumptionData",
    "SafetySystemStatus",
    "ProductionSchedule",
    "WeatherForecast",
    "EnergyPrices",
    "EquipmentHealth",
    "AlarmState",

    # Sub-schemas
    "SteamHeaderData",
    "ValvePosition",
    "FuelConsumption",
    "BoilerPerformance",
    "SISPermissive",
    "TripPoint",
    "BypassRecord",
    "BatchPlan",
    "UnitTarget",
    "Campaign",
    "HourlyForecast",
    "ForecastUncertainty",
    "ElectricityPrice",
    "FuelPrice",
    "TariffPeriod",
    "VibrationData",
    "LubeOilAnalysis",
    "FoulingIndicator",
    "RemainingUsefulLife",
    "AlarmRecord",
    "AlarmStatistics",

    # Schema Registries
    "DOMAIN_SCHEMAS",
    "SUB_SCHEMAS",

    # Tag Dictionary Enums
    "TagDataType",
    "TagCategory",
    "UnitCategory",
    "QualityCode",

    # Tag Dictionary Models
    "TagDefinition",
    "TagValue",

    # Tag Dictionary Classes
    "TagDictionary",
    "UnitConversion",
    "get_tag_dictionary",

    # Data Quality Enums
    "ValidationSeverity",
    "TimeSource",
    "DataLineageType",
    "TruthLabelStatus",

    # Data Quality Classes
    "TimeSyncValidator",
    "CompletenessValidator",
    "ValidityValidator",
    "LineageNode",
    "LineageTracker",
    "TruthLabel",
    "TruthLabelHandler",
    "DataQualityScorer",
    "DataQualityManager",
    "get_quality_manager",

    # Schema Registry Enums
    "CompatibilityMode",
    "SchemaStatus",
    "ChangeType",

    # Schema Registry Models
    "SemanticVersion",
    "SchemaChange",
    "SchemaDefinition",
    "SchemaMigration",

    # Schema Registry Classes
    "CompatibilityChecker",
    "SchemaMigrator",
    "SchemaRegistry",
    "get_schema_registry",

    # Utility Functions
    "validate_record",
    "create_data_record",
    "get_schema_json",
]
