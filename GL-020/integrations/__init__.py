"""
GL-020 ECONOPULSE - Enterprise Data Integration Module

This module provides production-grade connectors for economizer instrumentation
and control systems, including:

- Sensor Integration: RTD, thermocouple, flow meter, pressure transducer
- SCADA/DCS Integration: OPC UA and Modbus TCP/RTU clients
- Soot Blower Control: Zone-based cleaning with safety interlocks
- Process Historian: OSIsoft PI, AspenTech InfoPlus.21, Wonderware
- Data Quality: Validation, redundancy management, quality scoring

All connectors implement:
- Circuit breaker pattern for fault tolerance
- Connection pooling and retry logic
- Thread-safe concurrent access
- Comprehensive error handling and logging
"""

__version__ = "1.0.0"
__agent_id__ = "GL-020"
__codename__ = "ECONOPULSE"

# Sensor Connectors
from .sensor_connector import (
    SensorConnectorBase,
    RTDTemperatureSensor,
    ThermocoupleConnector,
    FlowMeterConnector,
    PressureTransducerConnector,
    SensorProtocol,
    SensorReading,
    SensorStatus,
    CalibrationData,
)

# SCADA/DCS Integration
from .scada_integration import (
    SCADAClient,
    SCADAProtocol,
    TagSubscription,
    TagGroup,
    EconomizerTagGroups,
    HistoricalDataRequest,
    SetpointWriteRequest,
)

# Soot Blower Control
from .soot_blower_integration import (
    SootBlowerController,
    BlowerStatus,
    CleaningCycle,
    CleaningZone,
    MediaConsumption,
    CleaningEffectiveness,
    SafetyInterlock,
)

# Process Historian
from .historian_connector import (
    HistorianConnectorBase,
    OSIsoftPIConnector,
    AspenInfoPlusConnector,
    WonderwareHistorianConnector,
    TimeSeriesData,
    CompressedDataConfig,
    CalculatedTag,
)

# Data Quality
from .data_quality import (
    DataQualityValidator,
    QualityFlag,
    ConfidenceScore,
    RedundancyManager,
    BadDataSubstitution,
    RangeCheck,
    RateOfChangeLimit,
)

__all__ = [
    # Version info
    "__version__",
    "__agent_id__",
    "__codename__",
    # Sensor connectors
    "SensorConnectorBase",
    "RTDTemperatureSensor",
    "ThermocoupleConnector",
    "FlowMeterConnector",
    "PressureTransducerConnector",
    "SensorProtocol",
    "SensorReading",
    "SensorStatus",
    "CalibrationData",
    # SCADA
    "SCADAClient",
    "SCADAProtocol",
    "TagSubscription",
    "TagGroup",
    "EconomizerTagGroups",
    "HistoricalDataRequest",
    "SetpointWriteRequest",
    # Soot blower
    "SootBlowerController",
    "BlowerStatus",
    "CleaningCycle",
    "CleaningZone",
    "MediaConsumption",
    "CleaningEffectiveness",
    "SafetyInterlock",
    # Historian
    "HistorianConnectorBase",
    "OSIsoftPIConnector",
    "AspenInfoPlusConnector",
    "WonderwareHistorianConnector",
    "TimeSeriesData",
    "CompressedDataConfig",
    "CalculatedTag",
    # Data quality
    "DataQualityValidator",
    "QualityFlag",
    "ConfidenceScore",
    "RedundancyManager",
    "BadDataSubstitution",
    "RangeCheck",
    "RateOfChangeLimit",
]
