"""
GL-012 STEAMQUAL - Integration Module

OT data acquisition, sensor conditioning, tag mapping, historian connectivity,
and GL-003 UNIFIEDSTEAM interface for steam quality control.

Components:
- Sensor connector for P/T/flow, chemistry, separator DP, and drain valve data
- Tag mapper for OT-to-internal name translation with unit conversion
- Historian connector for time series retrieval and backfill operations
- GL-003 interface for exporting quality constraints, state, events, and costs

Playbook Requirements:
- Time synchronization across all data sources
- Unit normalization to SI units (kPa, degC, kg/s)
- Data quality flagging per OPC-UA standards
- Provenance tracking with SHA-256 hashing
"""

from .sensor_connector import (
    SensorConnector,
    SensorConnectorConfig,
    SensorType,
    SensorReading,
    PressureSensor,
    TemperatureSensor,
    FlowSensor,
    ChemistryAnalyzer,
    SeparatorDPSensor,
    DrainValveSensor,
    SignalConditioner,
    SignalConditionerConfig,
    FilterType,
    SensorSubscription,
    SensorBatch,
    create_steam_quality_sensors,
)

from .tag_mapper import (
    TagMapper,
    TagMapperConfig,
    TagMapping,
    TagQualityFlag,
    UnitConversion,
    TagMetadata,
    TagValidationResult,
    SteamQualityTagSet,
    create_default_tag_mapping,
)

from .historian_connector import (
    HistorianConnector,
    HistorianConfig,
    HistorianType,
    TimeSeriesQuery,
    TimeSeriesResult,
    TimeSeriesPoint,
    BackfillRequest,
    BackfillResult,
    BackfillStatus,
    OutageBuffer,
    BufferEntry,
    create_historian_connector,
)

from .gl003_interface import (
    GL003Interface,
    GL003InterfaceConfig,
    QualityConstraints,
    QualityState,
    QualityEvent,
    QualityEventType,
    EventSeverity,
    QualityCosts,
    GL003ExportResult,
    SteamQualitySnapshot,
    create_gl003_interface,
)

__all__ = [
    # Sensor Connector
    "SensorConnector",
    "SensorConnectorConfig",
    "SensorType",
    "SensorReading",
    "PressureSensor",
    "TemperatureSensor",
    "FlowSensor",
    "ChemistryAnalyzer",
    "SeparatorDPSensor",
    "DrainValveSensor",
    "SignalConditioner",
    "SignalConditionerConfig",
    "FilterType",
    "SensorSubscription",
    "SensorBatch",
    "create_steam_quality_sensors",
    # Tag Mapper
    "TagMapper",
    "TagMapperConfig",
    "TagMapping",
    "TagQualityFlag",
    "UnitConversion",
    "TagMetadata",
    "TagValidationResult",
    "SteamQualityTagSet",
    "create_default_tag_mapping",
    # Historian
    "HistorianConnector",
    "HistorianConfig",
    "HistorianType",
    "TimeSeriesQuery",
    "TimeSeriesResult",
    "TimeSeriesPoint",
    "BackfillRequest",
    "BackfillResult",
    "BackfillStatus",
    "OutageBuffer",
    "BufferEntry",
    "create_historian_connector",
    # GL-003 Interface
    "GL003Interface",
    "GL003InterfaceConfig",
    "QualityConstraints",
    "QualityState",
    "QualityEvent",
    "QualityEventType",
    "EventSeverity",
    "QualityCosts",
    "GL003ExportResult",
    "SteamQualitySnapshot",
    "create_gl003_interface",
]
