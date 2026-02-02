# -*- coding: utf-8 -*-
"""
GreenLang Data Layer Agents
===========================

The Data Layer provides data ingestion, connectivity, and management agents
for integrating external data sources into the GreenLang Climate OS.

Agents:
    GL-DATA-X-001: Document Ingestion & OCR Agent - PDF/invoice processing
    GL-DATA-X-002: SCADA/Historians Connector - Industrial time-series data
    GL-DATA-X-003: BMS Connector Agent - Building management system data
    GL-DATA-X-004: ERP/Finance Connector - Spend and procurement data
    GL-DATA-X-005: Fleet Telematics Connector - Vehicle tracking and fuel
    GL-DATA-X-006: Ag Sensors & Farm IoT - Agricultural sensor data
    GL-DATA-X-007: Satellite & Remote Sensing - Land cover and NBS
    GL-DATA-X-008: Weather & Climate Data - Weather and projections
    GL-DATA-X-009: Utility Tariff & Grid Factor - Grid emission factors
    GL-DATA-X-010: Emission Factor Library - Emission factors database
    GL-DATA-X-011: Materials & LCI Database - Life cycle inventory
    GL-DATA-X-012: Supplier Data Exchange - Supplier PCF data
    GL-DATA-X-013: IoT Meter Management - Meter inventory and trust
"""

# GL-DATA-X-001: Document Ingestion & OCR Agent
from greenlang.agents.data.document_ingestion_agent import (
    DocumentIngestionAgent,
    DocumentIngestionInput,
    DocumentIngestionOutput,
    DocumentType,
    ExtractionStatus,
    OCREngine,
    ExtractedField,
    LineItem,
    InvoiceData,
    ManifestData,
    UtilityBillData,
    BoundingBox,
)

# GL-DATA-X-002: SCADA/Historians Connector
from greenlang.agents.data.scada_connector_agent import (
    SCADAConnectorAgent,
    SCADAQueryInput,
    SCADAQueryOutput,
    ConnectionConfig,
    TagMapping,
    DataPoint,
    TimeSeriesData,
    ProtocolType,
    DataQuality,
    AggregationType,
    TagDataType,
    GreenLangDataCategory,
)

# GL-DATA-X-003: BMS Connector Agent
from greenlang.agents.data.bms_connector_agent import (
    BMSConnectorAgent,
    BMSConnectionConfig,
    BMSQueryInput,
    BMSQueryOutput,
    EquipmentConfig,
    MeterConfig,
    BMSDataPoint,
    MeterReading as BMSMeterReading,
    WeatherData as BMSWeatherData,
    OccupancyData,
    BuildingPerformance,
    BMSProtocol,
    EquipmentType,
    MeterType as BMSMeterType,
    OccupancyState,
)

# GL-DATA-X-004: ERP/Finance Connector
from greenlang.agents.data.erp_connector_agent import (
    ERPConnectorAgent,
    ERPConnectionConfig,
    ERPQueryInput,
    ERPQueryOutput,
    VendorMapping,
    MaterialMapping,
    PurchaseOrder,
    PurchaseOrderLine,
    SpendRecord,
    InventoryItem,
    ERPSystem,
    Scope3Category,
    TransactionType,
    SpendCategory,
)

# GL-DATA-X-005: Fleet Telematics Connector
from greenlang.agents.data.fleet_telematics_agent import (
    FleetTelematicsAgent,
    TelematicsConnectionConfig,
    VehicleConfig,
    FleetQueryInput,
    FleetQueryOutput,
    GPSPoint,
    FuelEvent,
    TripSummary,
    IdleEvent,
    DriverMetrics,
    TelematicsProvider,
    VehicleType,
    FuelType,
    EventType,
)

# GL-DATA-X-006: Ag Sensors & Farm IoT
from greenlang.agents.data.ag_sensors_agent import (
    AgSensorsAgent,
    AgIoTConnectionConfig,
    FieldConfig,
    SensorConfig,
    AgQueryInput,
    AgQueryOutput,
    SensorReading,
    IrrigationEvent,
    FertilizerApplication,
    CropYield,
    AgIoTPlatform,
    SensorType,
    CropType,
    FertilizerType,
)

# GL-DATA-X-007: Satellite & Remote Sensing
from greenlang.agents.data.satellite_remote_sensing_agent import (
    SatelliteRemoteSensingAgent,
    SatelliteConnectionConfig,
    AreaOfInterest,
    SatelliteQueryInput,
    SatelliteQueryOutput,
    VegetationIndexValue,
    LandCoverObservation,
    LandUseChange,
    CarbonStockEstimate,
    SatelliteProvider,
    VegetationIndex,
    LandCoverClass,
    ForestType,
    ChangeType,
)

# GL-DATA-X-008: Weather & Climate Data
from greenlang.agents.data.weather_climate_agent import (
    WeatherClimateAgent,
    WeatherConnectionConfig,
    Location,
    WeatherQueryInput,
    WeatherQueryOutput,
    WeatherObservation,
    DailyWeatherSummary,
    ClimateProjection,
    WeatherNormalization,
    WeatherProvider,
    ClimateScenario,
    WeatherVariable,
)

# GL-DATA-X-009: Utility Tariff & Grid Factor
from greenlang.agents.data.utility_tariff_agent import (
    UtilityTariffAgent,
    UtilityTariff,
    TariffQueryInput,
    TariffQueryOutput,
    GridEmissionFactor,
    HourlyEmissionFactor,
    RatePeriod,
    RECertificate,
    EmissionsCalculation,
    GridRegion,
    EmissionFactorType,
    TariffType,
    PeriodType,
)

# GL-DATA-X-010: Emission Factor Library
from greenlang.agents.data.emission_factor_library_agent import (
    EmissionFactorLibraryAgent,
    EmissionFactor,
    EmissionFactorCitation,
    FactorLookupRequest,
    FactorApplication,
    FactorLibraryInput,
    FactorLibraryOutput,
    EmissionScope,
    FactorSource,
    ActivityCategory,
    UnitType,
    QualityTier,
)

# GL-DATA-X-011: Materials & LCI Database
from greenlang.agents.data.materials_lci_agent import (
    MaterialsLCIAgent,
    MaterialDataset,
    ProcessDataset,
    LCIDatasetMeta,
    ImpactValue,
    MaterialLookup,
    MaterialCalculation,
    LCIQueryInput,
    LCIQueryOutput,
    LCIDatabase,
    MaterialCategory,
    SystemBoundary,
    ImpactCategory,
)

# GL-DATA-X-012: Supplier Data Exchange
from greenlang.agents.data.supplier_data_exchange_agent import (
    SupplierDataExchangeAgent,
    SupplierInfo,
    ProductInfo,
    PCFDataPoint,
    PCFSubmission,
    SupplierMapping,
    SupplierEmissions,
    ValidationCheck,
    SupplierQueryInput,
    SupplierQueryOutput,
    PCFStandard,
    SubmissionStatus,
    DataQualityRating,
    ValidationResult,
)

# GL-DATA-X-013: IoT Meter Management
from greenlang.agents.data.iot_meter_management_agent import (
    IoTMeterManagementAgent,
    Meter,
    MeterLocation,
    MeterSpecification,
    CalibrationRecord,
    MeterReading,
    MeterAnomaly,
    MeterTrustScore,
    VirtualMeterConfig,
    MeterQueryInput,
    MeterQueryOutput,
    MeterType,
    MeterStatus,
    CommunicationType,
    CalibrationResult,
    TrustLevel,
    AnomalyType,
)

__all__ = [
    # Document Ingestion Agent (GL-DATA-X-001)
    "DocumentIngestionAgent",
    "DocumentIngestionInput",
    "DocumentIngestionOutput",
    "DocumentType",
    "ExtractionStatus",
    "OCREngine",
    "ExtractedField",
    "LineItem",
    "InvoiceData",
    "ManifestData",
    "UtilityBillData",
    "BoundingBox",
    # SCADA Connector Agent (GL-DATA-X-002)
    "SCADAConnectorAgent",
    "SCADAQueryInput",
    "SCADAQueryOutput",
    "ConnectionConfig",
    "TagMapping",
    "DataPoint",
    "TimeSeriesData",
    "ProtocolType",
    "DataQuality",
    "AggregationType",
    "TagDataType",
    "GreenLangDataCategory",
    # BMS Connector Agent (GL-DATA-X-003)
    "BMSConnectorAgent",
    "BMSConnectionConfig",
    "BMSQueryInput",
    "BMSQueryOutput",
    "EquipmentConfig",
    "MeterConfig",
    "BMSDataPoint",
    "BMSMeterReading",
    "BMSWeatherData",
    "OccupancyData",
    "BuildingPerformance",
    "BMSProtocol",
    "EquipmentType",
    "BMSMeterType",
    "OccupancyState",
    # ERP Connector Agent (GL-DATA-X-004)
    "ERPConnectorAgent",
    "ERPConnectionConfig",
    "ERPQueryInput",
    "ERPQueryOutput",
    "VendorMapping",
    "MaterialMapping",
    "PurchaseOrder",
    "PurchaseOrderLine",
    "SpendRecord",
    "InventoryItem",
    "ERPSystem",
    "Scope3Category",
    "TransactionType",
    "SpendCategory",
    # Fleet Telematics Agent (GL-DATA-X-005)
    "FleetTelematicsAgent",
    "TelematicsConnectionConfig",
    "VehicleConfig",
    "FleetQueryInput",
    "FleetQueryOutput",
    "GPSPoint",
    "FuelEvent",
    "TripSummary",
    "IdleEvent",
    "DriverMetrics",
    "TelematicsProvider",
    "VehicleType",
    "FuelType",
    "EventType",
    # Ag Sensors Agent (GL-DATA-X-006)
    "AgSensorsAgent",
    "AgIoTConnectionConfig",
    "FieldConfig",
    "SensorConfig",
    "AgQueryInput",
    "AgQueryOutput",
    "SensorReading",
    "IrrigationEvent",
    "FertilizerApplication",
    "CropYield",
    "AgIoTPlatform",
    "SensorType",
    "CropType",
    "FertilizerType",
    # Satellite Remote Sensing Agent (GL-DATA-X-007)
    "SatelliteRemoteSensingAgent",
    "SatelliteConnectionConfig",
    "AreaOfInterest",
    "SatelliteQueryInput",
    "SatelliteQueryOutput",
    "VegetationIndexValue",
    "LandCoverObservation",
    "LandUseChange",
    "CarbonStockEstimate",
    "SatelliteProvider",
    "VegetationIndex",
    "LandCoverClass",
    "ForestType",
    "ChangeType",
    # Weather Climate Agent (GL-DATA-X-008)
    "WeatherClimateAgent",
    "WeatherConnectionConfig",
    "Location",
    "WeatherQueryInput",
    "WeatherQueryOutput",
    "WeatherObservation",
    "DailyWeatherSummary",
    "ClimateProjection",
    "WeatherNormalization",
    "WeatherProvider",
    "ClimateScenario",
    "WeatherVariable",
    # Utility Tariff Agent (GL-DATA-X-009)
    "UtilityTariffAgent",
    "UtilityTariff",
    "TariffQueryInput",
    "TariffQueryOutput",
    "GridEmissionFactor",
    "HourlyEmissionFactor",
    "RatePeriod",
    "RECertificate",
    "EmissionsCalculation",
    "GridRegion",
    "EmissionFactorType",
    "TariffType",
    "PeriodType",
    # Emission Factor Library Agent (GL-DATA-X-010)
    "EmissionFactorLibraryAgent",
    "EmissionFactor",
    "EmissionFactorCitation",
    "FactorLookupRequest",
    "FactorApplication",
    "FactorLibraryInput",
    "FactorLibraryOutput",
    "EmissionScope",
    "FactorSource",
    "ActivityCategory",
    "UnitType",
    "QualityTier",
    # Materials LCI Agent (GL-DATA-X-011)
    "MaterialsLCIAgent",
    "MaterialDataset",
    "ProcessDataset",
    "LCIDatasetMeta",
    "ImpactValue",
    "MaterialLookup",
    "MaterialCalculation",
    "LCIQueryInput",
    "LCIQueryOutput",
    "LCIDatabase",
    "MaterialCategory",
    "SystemBoundary",
    "ImpactCategory",
    # Supplier Data Exchange Agent (GL-DATA-X-012)
    "SupplierDataExchangeAgent",
    "SupplierInfo",
    "ProductInfo",
    "PCFDataPoint",
    "PCFSubmission",
    "SupplierMapping",
    "SupplierEmissions",
    "ValidationCheck",
    "SupplierQueryInput",
    "SupplierQueryOutput",
    "PCFStandard",
    "SubmissionStatus",
    "DataQualityRating",
    "ValidationResult",
    # IoT Meter Management Agent (GL-DATA-X-013)
    "IoTMeterManagementAgent",
    "Meter",
    "MeterLocation",
    "MeterSpecification",
    "CalibrationRecord",
    "MeterReading",
    "MeterAnomaly",
    "MeterTrustScore",
    "VirtualMeterConfig",
    "MeterQueryInput",
    "MeterQueryOutput",
    "MeterType",
    "MeterStatus",
    "CommunicationType",
    "CalibrationResult",
    "TrustLevel",
    "AnomalyType",
]
