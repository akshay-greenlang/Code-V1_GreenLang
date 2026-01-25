# -*- coding: utf-8 -*-
"""
Data Contracts Module for GL-015 Insulscan Agent.

Provides validated Pydantic v2 data schemas for insulation scanning,
thermal assessment, condition monitoring, and maintenance planning with
zero-hallucination guarantees.

This module exports schemas for:
- Asset master data (InsulationAssetConfig, InsulationSpec, SurfaceGeometry, AssetMetadata)
- Thermal measurements (ThermalImageData, SurfaceTemperatureMeasurement, AmbientConditions, InspectionRecord)
- Heat loss computations (HeatLossInput, HeatLossOutput, ConditionAssessmentInput, ConditionAssessmentOutput)
- Maintenance records (RepairWorkOrder, MaterialSpec, MaintenanceSchedule)
- Data quality (DataQualityScore, ValidationRule, QualityReport)

All schemas follow GreenLang conventions:
- Strict validation with Field constraints
- Units in field descriptions
- JSON serialization support (OpenAPI 3.0 compatible)
- Examples in json_schema_extra for documentation
- SHA-256 provenance hashing for audit trails

Author: GreenLang AI Team
Version: 1.0.0
"""

# =============================================================================
# ASSET SCHEMAS
# =============================================================================
from .asset_schemas import (
    # Enumerations
    SurfaceType,
    GeometryShape,
    InsulationMaterialType,
    JacketMaterial,
    AssetStatus,
    AssetCriticality,
    ConditionRating,
    # Supporting models
    DimensionSpec,
    SurfaceGeometry,
    InsulationSpec,
    AssetMetadata,
    # Main schema
    InsulationAssetConfig,
    # Export dictionary
    ASSET_SCHEMAS,
)

# =============================================================================
# MEASUREMENT SCHEMAS
# =============================================================================
from .measurement_schemas import (
    # Enumerations
    MeasurementQuality,
    DataSource,
    CameraType,
    CalibrationStatus,
    InspectionMethod,
    FindingSeverity,
    DamageType,
    # Supporting models
    CalibrationData,
    InspectionFinding,
    # Main schemas
    ThermalImageData,
    SurfaceTemperatureMeasurement,
    AmbientConditions,
    InspectionRecord,
    # Export dictionary
    MEASUREMENT_SCHEMAS,
)

# =============================================================================
# COMPUTATION SCHEMAS
# =============================================================================
from .computation_schemas import (
    # Enumerations
    ComputationType,
    ComputationStatus,
    ValidityFlag,
    HeatTransferMechanism,
    ConditionCategory,
    SeverityLevel,
    WarningCode,
    # Supporting models
    ComputationWarning,
    OperatingConditions,
    HeatLossBreakdown,
    UncertaintyBounds,
    ContributingFactor,
    # Main schemas
    HeatLossInput,
    HeatLossOutput,
    ConditionAssessmentInput,
    ConditionAssessmentOutput,
    # Export dictionary
    COMPUTATION_SCHEMAS,
)

# =============================================================================
# MAINTENANCE SCHEMAS
# =============================================================================
from .maintenance_schemas import (
    # Enumerations
    RepairType,
    WorkOrderStatus,
    WorkOrderPriority,
    MaintenanceCategory,
    MaterialUnit,
    ScheduleStatus,
    CrewType,
    # Supporting models
    MaterialSpec,
    LaborSpec,
    # Main schemas
    RepairWorkOrder,
    MaintenanceSchedule,
    # Export dictionary
    MAINTENANCE_SCHEMAS,
)

# =============================================================================
# DATA QUALITY SCHEMAS
# =============================================================================
from .data_quality import (
    # Enumerations
    QualityDimension,
    RuleSeverity,
    RuleCategory,
    ValidationOutcome,
    QualityLevel,
    IssueStatus,
    # Supporting models
    DimensionScore,
    RuleParameter,
    ValidationResult,
    QualityIssue,
    # Main schemas
    DataQualityScore,
    ValidationRule,
    QualityReport,
    QualityConfiguration,
    # Export dictionary
    DATA_QUALITY_SCHEMAS,
)

# =============================================================================
# ALL SCHEMAS DICTIONARY
# =============================================================================
ALL_SCHEMAS = {
    **ASSET_SCHEMAS,
    **MEASUREMENT_SCHEMAS,
    **COMPUTATION_SCHEMAS,
    **MAINTENANCE_SCHEMAS,
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
    "SurfaceType",
    "GeometryShape",
    "InsulationMaterialType",
    "JacketMaterial",
    "AssetStatus",
    "AssetCriticality",
    "ConditionRating",
    # Supporting models
    "DimensionSpec",
    "SurfaceGeometry",
    "InsulationSpec",
    "AssetMetadata",
    # Main schema
    "InsulationAssetConfig",
    # Export dictionary
    "ASSET_SCHEMAS",
    # =========================================================================
    # Measurement Schemas
    # =========================================================================
    # Enumerations
    "MeasurementQuality",
    "DataSource",
    "CameraType",
    "CalibrationStatus",
    "InspectionMethod",
    "FindingSeverity",
    "DamageType",
    # Supporting models
    "CalibrationData",
    "InspectionFinding",
    # Main schemas
    "ThermalImageData",
    "SurfaceTemperatureMeasurement",
    "AmbientConditions",
    "InspectionRecord",
    # Export dictionary
    "MEASUREMENT_SCHEMAS",
    # =========================================================================
    # Computation Schemas
    # =========================================================================
    # Enumerations
    "ComputationType",
    "ComputationStatus",
    "ValidityFlag",
    "HeatTransferMechanism",
    "ConditionCategory",
    "SeverityLevel",
    "WarningCode",
    # Supporting models
    "ComputationWarning",
    "OperatingConditions",
    "HeatLossBreakdown",
    "UncertaintyBounds",
    "ContributingFactor",
    # Main schemas
    "HeatLossInput",
    "HeatLossOutput",
    "ConditionAssessmentInput",
    "ConditionAssessmentOutput",
    # Export dictionary
    "COMPUTATION_SCHEMAS",
    # =========================================================================
    # Maintenance Schemas
    # =========================================================================
    # Enumerations
    "RepairType",
    "WorkOrderStatus",
    "WorkOrderPriority",
    "MaintenanceCategory",
    "MaterialUnit",
    "ScheduleStatus",
    "CrewType",
    # Supporting models
    "MaterialSpec",
    "LaborSpec",
    # Main schemas
    "RepairWorkOrder",
    "MaintenanceSchedule",
    # Export dictionary
    "MAINTENANCE_SCHEMAS",
    # =========================================================================
    # Data Quality Schemas
    # =========================================================================
    # Enumerations
    "QualityDimension",
    "RuleSeverity",
    "RuleCategory",
    "ValidationOutcome",
    "QualityLevel",
    "IssueStatus",
    # Supporting models
    "DimensionScore",
    "RuleParameter",
    "ValidationResult",
    "QualityIssue",
    # Main schemas
    "DataQualityScore",
    "ValidationRule",
    "QualityReport",
    "QualityConfiguration",
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
