"""
GL-004 BURNMASTER Schemas Module

This module provides comprehensive Pydantic v2 schemas for all data models
used in the BURNMASTER combustion optimization agent. All schemas support:
- Strict validation with Pydantic v2
- JSON schema export capability
- Serialization/deserialization
- Comprehensive field descriptions
- Proper typing with Optional, Union, Literal

Submodules:
    combustion_schemas: Core combustion data models (CombustionData, FuelProperties, etc.)
    optimization_schemas: Optimization objectives, constraints, and results
    control_schemas: Operating modes, setpoint writes, mode transitions
    safety_schemas: Safety limits, envelopes, interlocks, and hazard levels
    monitoring_schemas: Alerts, KPI dashboards, and health status
    audit_schemas: Audit records, provenance tracking, and compliance reports
    integration_schemas: Tag values, write requests, and connection status

Example:
    >>> from schemas import CombustionData, FuelProperties, OptimizationResult
    >>> from schemas import OperatingMode, SafetyLimit, Alert

Usage:
    # Import specific schemas
    from schemas.combustion_schemas import CombustionData, BurnerState
    from schemas.optimization_schemas import OptimizationResult, Constraint
    from schemas.control_schemas import SetpointWrite, ControlCycleResult
    from schemas.safety_schemas import SafetyLimit, SafetyEnvelope
    from schemas.monitoring_schemas import Alert, KPIDashboard
    from schemas.audit_schemas import AuditRecord, EvidencePack
    from schemas.integration_schemas import TagValue, WriteRequest

    # Or import from main module
    from schemas import CombustionData, OptimizationResult, Alert
"""

# Combustion Schemas
from .combustion_schemas import (
    FuelType,
    CombustionPhase,
    TemperatureReading,
    PressureReading,
    CombustionData,
    FuelComposition,
    FuelProperties,
    FlueGasComposition,
    StabilityIndicator,
    BurnerState,
    ParameterLimit,
    OperatingEnvelope,
)

# Optimization Schemas
from .optimization_schemas import (
    ObjectiveType,
    ConstraintType,
    ConstraintOperator,
    OptimizationStatus,
    OptimizationObjective,
    Constraint,
    ConstraintSet,
    SetpointValue,
    BindingConstraint,
    OptimizationResult,
    ConfidenceLevel,
    RecommendationPriority,
    SetpointRecommendation,
    OptimizationScenario,
)

# Control Schemas
from .control_schemas import (
    OperatingMode,
    WriteStatus,
    ActionType,
    ControlAuthority,
    AuditContext,
    SetpointWrite,
    ModeTransitionTrigger,
    ModeTransition,
    ControlAction,
    StateSnapshot,
    ControlCycleResult,
    ControllerConfig,
)

# Safety Schemas
from .safety_schemas import (
    HazardLevel,
    SafetyCategory,
    InterlockType,
    InterlockState,
    SafetyCheckType,
    SafetyLimit,
    SafetyEnvelope,
    SafetyCheck,
    InterlockStatus,
    SafetyAssessment,
    EmergencyAction,
)

# Monitoring Schemas
from .monitoring_schemas import (
    AlertLevel,
    AlertCategory,
    AlertState,
    ComponentStatus,
    TrendDirection,
    Alert,
    AlertSummary,
    MetricValue,
    KPIDashboard,
    HealthCheck,
    HealthStatus,
    SystemHealthSummary,
    MonitoringConfig,
)

# Audit Schemas
from .audit_schemas import (
    AuditEventType,
    AuditSeverity,
    ComplianceStatus,
    CertificationStatus,
    AuditRecord,
    ProvenanceLink,
    DataSnapshot,
    CalculationRecord,
    EvidencePack,
    EmissionsRecord,
    Certification,
    ComplianceViolation,
    ComplianceReport,
)

# Integration Schemas
from .integration_schemas import (
    TagQuality,
    ConnectionState,
    WriteRequestStatus,
    DataType,
    ProtocolType,
    TagValue,
    TagValueBatch,
    WriteRequest,
    ConnectionStatus,
    TagConfiguration,
    IntegrationConfig,
    DataExchange,
)


# Version information
__version__ = "1.0.0"
__author__ = "GreenLang"


# All exports
__all__ = [
    # Combustion Schemas
    "FuelType",
    "CombustionPhase",
    "TemperatureReading",
    "PressureReading",
    "CombustionData",
    "FuelComposition",
    "FuelProperties",
    "FlueGasComposition",
    "StabilityIndicator",
    "BurnerState",
    "ParameterLimit",
    "OperatingEnvelope",

    # Optimization Schemas
    "ObjectiveType",
    "ConstraintType",
    "ConstraintOperator",
    "OptimizationStatus",
    "OptimizationObjective",
    "Constraint",
    "ConstraintSet",
    "SetpointValue",
    "BindingConstraint",
    "OptimizationResult",
    "ConfidenceLevel",
    "RecommendationPriority",
    "SetpointRecommendation",
    "OptimizationScenario",

    # Control Schemas
    "OperatingMode",
    "WriteStatus",
    "ActionType",
    "ControlAuthority",
    "AuditContext",
    "SetpointWrite",
    "ModeTransitionTrigger",
    "ModeTransition",
    "ControlAction",
    "StateSnapshot",
    "ControlCycleResult",
    "ControllerConfig",

    # Safety Schemas
    "HazardLevel",
    "SafetyCategory",
    "InterlockType",
    "InterlockState",
    "SafetyCheckType",
    "SafetyLimit",
    "SafetyEnvelope",
    "SafetyCheck",
    "InterlockStatus",
    "SafetyAssessment",
    "EmergencyAction",

    # Monitoring Schemas
    "AlertLevel",
    "AlertCategory",
    "AlertState",
    "ComponentStatus",
    "TrendDirection",
    "Alert",
    "AlertSummary",
    "MetricValue",
    "KPIDashboard",
    "HealthCheck",
    "HealthStatus",
    "SystemHealthSummary",
    "MonitoringConfig",

    # Audit Schemas
    "AuditEventType",
    "AuditSeverity",
    "ComplianceStatus",
    "CertificationStatus",
    "AuditRecord",
    "ProvenanceLink",
    "DataSnapshot",
    "CalculationRecord",
    "EvidencePack",
    "EmissionsRecord",
    "Certification",
    "ComplianceViolation",
    "ComplianceReport",

    # Integration Schemas
    "TagQuality",
    "ConnectionState",
    "WriteRequestStatus",
    "DataType",
    "ProtocolType",
    "TagValue",
    "TagValueBatch",
    "WriteRequest",
    "ConnectionStatus",
    "TagConfiguration",
    "IntegrationConfig",
    "DataExchange",
]


def get_json_schema(schema_class: type) -> dict:
    """
    Get JSON schema for a Pydantic model.

    Args:
        schema_class: Pydantic model class

    Returns:
        JSON schema dictionary

    Example:
        >>> from schemas import CombustionData
        >>> schema = get_json_schema(CombustionData)
        >>> print(schema['title'])
        'CombustionData'
    """
    return schema_class.model_json_schema()


def export_all_schemas() -> dict:
    """
    Export JSON schemas for all schema classes.

    Returns:
        Dictionary mapping class names to JSON schemas

    Example:
        >>> schemas = export_all_schemas()
        >>> print(list(schemas.keys())[:3])
        ['FuelType', 'CombustionPhase', 'TemperatureReading']
    """
    schemas = {}
    for name in __all__:
        cls = globals().get(name)
        if cls is not None and hasattr(cls, 'model_json_schema'):
            try:
                schemas[name] = cls.model_json_schema()
            except Exception:
                # Skip non-model classes (e.g., Enums)
                pass
    return schemas
