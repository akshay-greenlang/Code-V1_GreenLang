"""
GL-010 EMISSIONWATCH Integration Connectors Package.

This package provides enterprise-grade integration connectors for emissions
monitoring and regulatory reporting. All connectors implement common patterns
for connection pooling, retry logic, circuit breaker protection, health
monitoring, caching, and audit logging.

Connectors:
    - CEMSConnector: Continuous Emissions Monitoring System integration
    - EPACEDRIConnector: EPA CEDRI electronic reporting
    - EUETSConnector: EU Emissions Trading System integration
    - StackAnalyzerConnector: Stack gas analyzer integration
    - FuelFlowConnector: Fuel flow metering integration
    - WeatherConnector: Meteorological data integration
    - PermitDatabaseConnector: Permit and regulatory limits database
    - ReportingConnector: Multi-system regulatory reporting

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

from .base_connector import (
    # Base classes
    BaseConnector,
    BaseConnectorConfig,
    # Connection pooling
    ConnectionPool,
    # Circuit breaker
    CircuitBreaker,
    CircuitState,
    # Cache
    LRUCache,
    CacheEntry,
    # Retry
    with_retry,
    # Metrics
    MetricsCollector,
    MetricsSnapshot,
    # Audit
    AuditLogger,
    AuditLogEntry,
    # Enums
    ConnectionState,
    ConnectorType,
    HealthStatus,
    # Models
    ConnectionInfo,
    HealthCheckResult,
    # Exceptions
    ConnectorError,
    ConnectionError,
    AuthenticationError,
    TimeoutError,
    ValidationError,
    CircuitOpenError,
    RetryExhaustedError,
    ConfigurationError,
    DataQualityError,
    # Context manager
    ConnectorContextManager,
)

from .cems_connector import (
    # Main connector
    CEMSConnector,
    CEMSConnectorConfig,
    # Data models
    EmissionsData,
    AnalyzerReading,
    CalibrationData,
    QualityFlags,
    # Enums
    CEMSVendor,
    ProtocolType,
    AnalyzerType,
    MeasurementUnit,
    QualityAssuranceStatus,
    CalibrationStatus,
    SubstitutionMethod,
    # Supporting classes
    CEMSDataProcessor,
    CalibrationManager,
    ModbusHandler,
    OPCUAHandler,
    ModbusRegisterMap,
    OPCUANodeMap,
    # Factory
    create_cems_connector,
)

from .epa_cedri_connector import (
    # Main connector
    EPACEDRIConnector,
    EPACEDRIConnectorConfig,
    # Data models
    CEDRIReport,
    FacilityIdentification,
    EmissionUnit,
    EmissionsRecord,
    DeviationRecord,
    ExcessEmissionsRecord,
    SubmissionResult,
    CDXCredentials,
    # Enums
    ReportType,
    SubmissionStatus as CEDRISubmissionStatus,
    CDXEnvironment,
    PollutantCode,
    UnitMeasureCode,
    RegulatoryProgram,
    # Supporting classes
    CEDRIXMLGenerator,
    CDXAPIClient,
    DigitalSignatureHandler,
    # Factory
    create_cedri_connector,
)

from .eu_ets_connector import (
    # Main connector
    EUETSConnector,
    EUETSConnectorConfig,
    # Data models
    InstallationIdentification,
    AllowanceBalance,
    AllowanceTransaction,
    VerifiedEmissions,
    MonitoringPlan,
    AnnualReport,
    RegistryCredentials,
    # Enums
    ETSPhase,
    AllowanceType,
    TransactionType,
    VerificationStatus,
    InstallationType,
    MonitoringApproach,
    ActivityType,
    MemberState,
    # Supporting classes
    EmissionsCalculator,
    ETSRegistryClient,
    FreeAllocationCalculator,
    # Factory
    create_eu_ets_connector,
)

from .stack_analyzer_connector import (
    # Main connector
    StackAnalyzerConnector,
    StackAnalyzerConnectorConfig,
    # Data models
    AnalyzerReading as StackAnalyzerReading,
    AnalyzerChannelConfig,
    CalibrationResult,
    AnalyzerAlarm,
    AnalyzerDiagnostics,
    # Enums
    AnalyzerTechnology,
    AnalyzerVendor,
    GasComponent,
    MeasurementRange,
    AnalyzerStatus,
    CalibrationMode,
    AlarmType,
    # Supporting classes
    AnalyzerDataValidator,
    AnalyzerCalibrationManager,
    ModbusProtocolHandler,
    SerialProtocolHandler,
    # Factory
    create_stack_analyzer_connector,
)

from .fuel_flow_connector import (
    # Main connector
    FuelFlowConnector,
    FuelFlowConnectorConfig,
    # Data models
    FlowReading,
    GasComposition,
    FuelAnalysis,
    HeatInput,
    FuelBlend,
    MeterConfiguration,
    # Enums
    FuelType,
    MeterType,
    MeterVendor,
    FlowUnit,
    EnergyUnit,
    CompositionMethod,
    # Supporting classes
    HeatInputCalculator,
    FuelFlowProtocolHandler,
    GasChromatographHandler,
    # Factory
    create_fuel_flow_connector,
)

from .weather_connector import (
    # Main connector
    WeatherConnector,
    WeatherConnectorConfig,
    # Data models
    MeteorologicalObservation,
    WindData,
    TemperatureData,
    PressureData,
    SolarRadiationData,
    PrecipitationData,
    CloudData,
    AtmosphericStability,
    MixingHeightEstimate,
    # Enums
    DataSource,
    StabilityClass,
    CloudCoverCategory,
    PrecipitationType,
    WindDirection,
    MetStationVendor,
    # Supporting classes
    StabilityClassifier,
    NWSAPIClient,
    # Factory
    create_weather_connector,
)

from .permit_database_connector import (
    # Main connector
    PermitDatabaseConnector,
    PermitDatabaseConnectorConfig,
    # Data models
    Permit,
    EmissionLimit,
    PermitCondition,
    StartupShutdownProvision,
    MalfunctionProvision,
    ApplicabilityDetermination,
    ComplianceRecord,
    # Enums
    PermitType,
    PermitStatus,
    LimitType,
    AveragingPeriod,
    Pollutant,
    Jurisdiction,
    OperatingMode,
    ComplianceStatus,
    # Supporting classes
    PermitRepository,
    ComplianceEvaluator,
    # Factory
    create_permit_database_connector,
)

from .reporting_connector import (
    # Main connector
    ReportingConnector,
    ReportingConnectorConfig,
    # Data models
    ReportingAgency,
    ReportingRequirement,
    EmissionsDataRecord,
    ReportDocument,
    ReportSubmission,
    ReportingSchedule,
    # Enums
    ReportingSystem,
    ReportFormat,
    ReportFrequency,
    SubmissionMethod,
    SubmissionStatus as ReportingSubmissionStatus,
    PollutantCategory,
    # Supporting classes
    ReportGenerator,
    SubmissionHandler,
    ScheduleManager,
    # Factory
    create_reporting_connector,
)


__all__ = [
    # ==========================================================================
    # Base Connector
    # ==========================================================================
    "BaseConnector",
    "BaseConnectorConfig",
    "ConnectionPool",
    "CircuitBreaker",
    "CircuitState",
    "LRUCache",
    "CacheEntry",
    "with_retry",
    "MetricsCollector",
    "MetricsSnapshot",
    "AuditLogger",
    "AuditLogEntry",
    "ConnectionState",
    "ConnectorType",
    "HealthStatus",
    "ConnectionInfo",
    "HealthCheckResult",
    "ConnectorError",
    "ConnectionError",
    "AuthenticationError",
    "TimeoutError",
    "ValidationError",
    "CircuitOpenError",
    "RetryExhaustedError",
    "ConfigurationError",
    "DataQualityError",
    "ConnectorContextManager",

    # ==========================================================================
    # CEMS Connector
    # ==========================================================================
    "CEMSConnector",
    "CEMSConnectorConfig",
    "EmissionsData",
    "AnalyzerReading",
    "CalibrationData",
    "QualityFlags",
    "CEMSVendor",
    "ProtocolType",
    "AnalyzerType",
    "MeasurementUnit",
    "QualityAssuranceStatus",
    "CalibrationStatus",
    "SubstitutionMethod",
    "CEMSDataProcessor",
    "CalibrationManager",
    "ModbusHandler",
    "OPCUAHandler",
    "ModbusRegisterMap",
    "OPCUANodeMap",
    "create_cems_connector",

    # ==========================================================================
    # EPA CEDRI Connector
    # ==========================================================================
    "EPACEDRIConnector",
    "EPACEDRIConnectorConfig",
    "CEDRIReport",
    "FacilityIdentification",
    "EmissionUnit",
    "EmissionsRecord",
    "DeviationRecord",
    "ExcessEmissionsRecord",
    "SubmissionResult",
    "CDXCredentials",
    "ReportType",
    "CEDRISubmissionStatus",
    "CDXEnvironment",
    "PollutantCode",
    "UnitMeasureCode",
    "RegulatoryProgram",
    "CEDRIXMLGenerator",
    "CDXAPIClient",
    "DigitalSignatureHandler",
    "create_cedri_connector",

    # ==========================================================================
    # EU ETS Connector
    # ==========================================================================
    "EUETSConnector",
    "EUETSConnectorConfig",
    "InstallationIdentification",
    "AllowanceBalance",
    "AllowanceTransaction",
    "VerifiedEmissions",
    "MonitoringPlan",
    "AnnualReport",
    "RegistryCredentials",
    "ETSPhase",
    "AllowanceType",
    "TransactionType",
    "VerificationStatus",
    "InstallationType",
    "MonitoringApproach",
    "ActivityType",
    "MemberState",
    "EmissionsCalculator",
    "ETSRegistryClient",
    "FreeAllocationCalculator",
    "create_eu_ets_connector",

    # ==========================================================================
    # Stack Analyzer Connector
    # ==========================================================================
    "StackAnalyzerConnector",
    "StackAnalyzerConnectorConfig",
    "StackAnalyzerReading",
    "AnalyzerChannelConfig",
    "CalibrationResult",
    "AnalyzerAlarm",
    "AnalyzerDiagnostics",
    "AnalyzerTechnology",
    "AnalyzerVendor",
    "GasComponent",
    "MeasurementRange",
    "AnalyzerStatus",
    "CalibrationMode",
    "AlarmType",
    "AnalyzerDataValidator",
    "AnalyzerCalibrationManager",
    "ModbusProtocolHandler",
    "SerialProtocolHandler",
    "create_stack_analyzer_connector",

    # ==========================================================================
    # Fuel Flow Connector
    # ==========================================================================
    "FuelFlowConnector",
    "FuelFlowConnectorConfig",
    "FlowReading",
    "GasComposition",
    "FuelAnalysis",
    "HeatInput",
    "FuelBlend",
    "MeterConfiguration",
    "FuelType",
    "MeterType",
    "MeterVendor",
    "FlowUnit",
    "EnergyUnit",
    "CompositionMethod",
    "HeatInputCalculator",
    "FuelFlowProtocolHandler",
    "GasChromatographHandler",
    "create_fuel_flow_connector",

    # ==========================================================================
    # Weather Connector
    # ==========================================================================
    "WeatherConnector",
    "WeatherConnectorConfig",
    "MeteorologicalObservation",
    "WindData",
    "TemperatureData",
    "PressureData",
    "SolarRadiationData",
    "PrecipitationData",
    "CloudData",
    "AtmosphericStability",
    "MixingHeightEstimate",
    "DataSource",
    "StabilityClass",
    "CloudCoverCategory",
    "PrecipitationType",
    "WindDirection",
    "MetStationVendor",
    "StabilityClassifier",
    "NWSAPIClient",
    "create_weather_connector",

    # ==========================================================================
    # Permit Database Connector
    # ==========================================================================
    "PermitDatabaseConnector",
    "PermitDatabaseConnectorConfig",
    "Permit",
    "EmissionLimit",
    "PermitCondition",
    "StartupShutdownProvision",
    "MalfunctionProvision",
    "ApplicabilityDetermination",
    "ComplianceRecord",
    "PermitType",
    "PermitStatus",
    "LimitType",
    "AveragingPeriod",
    "Pollutant",
    "Jurisdiction",
    "OperatingMode",
    "ComplianceStatus",
    "PermitRepository",
    "ComplianceEvaluator",
    "create_permit_database_connector",

    # ==========================================================================
    # Reporting Connector
    # ==========================================================================
    "ReportingConnector",
    "ReportingConnectorConfig",
    "ReportingAgency",
    "ReportingRequirement",
    "EmissionsDataRecord",
    "ReportDocument",
    "ReportSubmission",
    "ReportingSchedule",
    "ReportingSystem",
    "ReportFormat",
    "ReportFrequency",
    "SubmissionMethod",
    "ReportingSubmissionStatus",
    "PollutantCategory",
    "ReportGenerator",
    "SubmissionHandler",
    "ScheduleManager",
    "create_reporting_connector",
]

__version__ = "1.0.0"
__author__ = "GL-DataIntegrationEngineer"
