# -*- coding: utf-8 -*-
"""
Data Contracts Module for GL-014 Exchangerpro Agent.

Provides validated Pydantic v2 data schemas for heat exchanger performance
monitoring, fouling prediction, and cleaning optimization with
zero-hallucination guarantees.

This module exports schemas for:
- Asset master data (ExchangerAsset, ExchangerGeometry, FluidProperties)
- Process measurements (Temperature, Flow, Pressure, Timeseries)
- Maintenance history (CleaningEvent, InspectionRecord, WorkOrder)
- Computation records (ComputationRecord, ThermalComputationResult, PredictionRecord)
- Data quality (DataQualityReport, SensorValidation)

All schemas follow GreenLang conventions:
- Strict validation with Field constraints
- Units in field descriptions
- JSON serialization support
- Examples in json_schema_extra for documentation

Author: GreenLang AI Team
Version: 1.0.0
"""

# =============================================================================
# ASSET SCHEMAS
# =============================================================================
from .asset_schemas import (
    # Enumerations
    TEMAType,
    ShellType,
    FlowArrangement,
    BaffleType,
    TubePattern,
    MaterialClass,
    ExchangerStatus,
    ExchangerCriticality,
    # Supporting models
    MaterialSpecification,
    NozzleSpecification,
    DesignConditions,
    # Main schemas
    ExchangerGeometry,
    FluidProperties,
    ExchangerAsset,
    # Export dictionary
    ASSET_SCHEMAS,
)

# =============================================================================
# MEASUREMENT SCHEMAS
# =============================================================================
from .measurement_schemas import (
    # Enumerations
    MeasurementQuality,
    TemperatureUnit,
    PressureUnit,
    FlowUnit,
    MeasurementSource,
    # Models
    TimeseriesPoint,
    TemperatureMeasurement,
    FlowMeasurement,
    PressureMeasurement,
    ProcessMeasurementSet,
    ProcessMeasurementBatch,
    # Export dictionary
    MEASUREMENT_SCHEMAS,
)

# =============================================================================
# MAINTENANCE SCHEMAS
# =============================================================================
from .maintenance_schemas import (
    # Enumerations
    CleaningType,
    CleaningMethod,
    CleaningSide,
    DepositType,
    InspectionType,
    WorkOrderStatus,
    WorkOrderPriority,
    MaintenanceCategory,
    # Supporting models
    ChemicalUsage,
    InspectionFinding,
    WorkOrderCost,
    # Main schemas
    CleaningEvent,
    InspectionRecord,
    WorkOrder,
    # Export dictionary
    MAINTENANCE_SCHEMAS,
)

# =============================================================================
# COMPUTATION SCHEMAS
# =============================================================================
from .computation_schemas import (
    # Enumerations
    ComputationType,
    ComputationStatus,
    ValidityFlag,
    WarningCode,
    PredictionType,
    ModelType,
    # Supporting models
    ComputationWarning,
    LMTDResult,
    NTUResult,
    HeatDutyResult,
    FoulingResult,
    ConfidenceInterval,
    FeatureImportance,
    ExplanationPayload,
    # Main schemas
    ComputationRecord,
    ThermalComputationResult,
    PredictionRecord,
    # Export dictionary
    COMPUTATION_SCHEMAS,
)

# =============================================================================
# DATA QUALITY SCHEMAS
# =============================================================================
from .data_quality import (
    # Enumerations
    QualitySeverity,
    QualityDimension,
    SensorStatus,
    DriftType,
    IssueCategory,
    # Quality issue model
    QualityIssue,
    # Sensor validation
    CalibrationRecord,
    DriftAnalysis,
    SensorValidation,
    # Metrics models
    CompletenessMetrics,
    RangeViolationMetrics,
    StuckSensorMetrics,
    EnergyBalanceMetrics,
    # Main schemas
    DataQualityReport,
    QualityThresholds,
    # Export dictionary
    DATA_QUALITY_SCHEMAS,
)

# =============================================================================
# ALL SCHEMAS DICTIONARY
# =============================================================================
ALL_SCHEMAS = {
    **ASSET_SCHEMAS,
    **MEASUREMENT_SCHEMAS,
    **MAINTENANCE_SCHEMAS,
    **COMPUTATION_SCHEMAS,
    **DATA_QUALITY_SCHEMAS,
}

# =============================================================================
# MODULE EXPORTS
# =============================================================================
__all__ = [
    # =========================================================================
    # Asset Schemas
    # =========================================================================
    # Enumerations
    "TEMAType",
    "ShellType",
    "FlowArrangement",
    "BaffleType",
    "TubePattern",
    "MaterialClass",
    "ExchangerStatus",
    "ExchangerCriticality",
    # Supporting models
    "MaterialSpecification",
    "NozzleSpecification",
    "DesignConditions",
    # Main schemas
    "ExchangerGeometry",
    "FluidProperties",
    "ExchangerAsset",
    # Export dictionary
    "ASSET_SCHEMAS",
    # =========================================================================
    # Measurement Schemas
    # =========================================================================
    # Enumerations
    "MeasurementQuality",
    "TemperatureUnit",
    "PressureUnit",
    "FlowUnit",
    "MeasurementSource",
    # Models
    "TimeseriesPoint",
    "TemperatureMeasurement",
    "FlowMeasurement",
    "PressureMeasurement",
    "ProcessMeasurementSet",
    "ProcessMeasurementBatch",
    # Export dictionary
    "MEASUREMENT_SCHEMAS",
    # =========================================================================
    # Maintenance Schemas
    # =========================================================================
    # Enumerations
    "CleaningType",
    "CleaningMethod",
    "CleaningSide",
    "DepositType",
    "InspectionType",
    "WorkOrderStatus",
    "WorkOrderPriority",
    "MaintenanceCategory",
    # Supporting models
    "ChemicalUsage",
    "InspectionFinding",
    "WorkOrderCost",
    # Main schemas
    "CleaningEvent",
    "InspectionRecord",
    "WorkOrder",
    # Export dictionary
    "MAINTENANCE_SCHEMAS",
    # =========================================================================
    # Computation Schemas
    # =========================================================================
    # Enumerations
    "ComputationType",
    "ComputationStatus",
    "ValidityFlag",
    "WarningCode",
    "PredictionType",
    "ModelType",
    # Supporting models
    "ComputationWarning",
    "LMTDResult",
    "NTUResult",
    "HeatDutyResult",
    "FoulingResult",
    "ConfidenceInterval",
    "FeatureImportance",
    "ExplanationPayload",
    # Main schemas
    "ComputationRecord",
    "ThermalComputationResult",
    "PredictionRecord",
    # Export dictionary
    "COMPUTATION_SCHEMAS",
    # =========================================================================
    # Data Quality Schemas
    # =========================================================================
    # Enumerations
    "QualitySeverity",
    "QualityDimension",
    "SensorStatus",
    "DriftType",
    "IssueCategory",
    # Quality issue model
    "QualityIssue",
    # Sensor validation
    "CalibrationRecord",
    "DriftAnalysis",
    "SensorValidation",
    # Metrics models
    "CompletenessMetrics",
    "RangeViolationMetrics",
    "StuckSensorMetrics",
    "EnergyBalanceMetrics",
    # Main schemas
    "DataQualityReport",
    "QualityThresholds",
    # Export dictionary
    "DATA_QUALITY_SCHEMAS",
    # =========================================================================
    # Combined Export
    # =========================================================================
    "ALL_SCHEMAS",
]

# =============================================================================
# VERSION INFO
# =============================================================================
__version__ = "1.0.0"
__schema_version__ = "1.0"
