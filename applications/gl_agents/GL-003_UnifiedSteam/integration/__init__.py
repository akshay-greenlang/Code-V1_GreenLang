"""
GL-003 UNIFIEDSTEAM - Integration Module

OT data acquisition, sensor transformation, tag mapping, and historian connectivity
for steam system optimization.

Components:
- OPC-UA connector for PLC/DCS/historian data acquisition
- Sensor data transformation with unit conversion and quality checks
- Tag mapping service for canonical naming conventions
- Historian connector for batch backfill operations
- Acoustics connector for steam trap edge devices
"""

from .opcua_connector import (
    OPCUAConnector,
    OPCUAConfig,
    SecurityPolicy,
    MessageSecurityMode,
    Subscription,
    TagValue,
    Node,
    NodeClass,
    DataChangeCallback,
)
from .sensor_transformer import (
    SensorTransformer,
    ValidationResult,
    QualifiedValue,
    TransformedData,
    CalibrationParams,
    UnitConverter,
    QualityCode,
)
from .tag_mapper import (
    TagMapper,
    TagMapping,
    SensorMetadata,
    ValidationError,
    TagNamingConvention,
)
from .historian_connector import (
    HistorianConnector,
    HistorianConfig,
    HistorianType,
    BackfillResult,
    TimeSeriesData,
)
from .acoustics_connector import (
    AcousticsConnector,
    AcousticFeatures,
    EdgeDeviceConfig,
    TrapAcousticSubscription,
)

__all__ = [
    # OPC-UA
    "OPCUAConnector",
    "OPCUAConfig",
    "SecurityPolicy",
    "MessageSecurityMode",
    "Subscription",
    "TagValue",
    "Node",
    "NodeClass",
    "DataChangeCallback",
    # Sensor Transformer
    "SensorTransformer",
    "ValidationResult",
    "QualifiedValue",
    "TransformedData",
    "CalibrationParams",
    "UnitConverter",
    "QualityCode",
    # Tag Mapper
    "TagMapper",
    "TagMapping",
    "SensorMetadata",
    "ValidationError",
    "TagNamingConvention",
    # Historian
    "HistorianConnector",
    "HistorianConfig",
    "HistorianType",
    "BackfillResult",
    "TimeSeriesData",
    # Acoustics
    "AcousticsConnector",
    "AcousticFeatures",
    "EdgeDeviceConfig",
    "TrapAcousticSubscription",
]
