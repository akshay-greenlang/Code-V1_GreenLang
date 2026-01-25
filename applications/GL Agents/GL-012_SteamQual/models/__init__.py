"""
GL-012_SteamQual - Models Module

Pydantic v2 data models for the SteamQual Steam Quality Controller agent.

This module provides:
- Domain models: Enumerations for steam states, event types, severities, and consumer classes
- Input models: Validated input structures for measurements and process data
- Output models: Result structures with provenance tracking and GL-003 compatibility
- Configuration models: Site, header, and consumer configuration structures

All models support JSON serialization for API communication and audit trail storage.

Usage:
    >>> from gl_012_steamqual.models import (
    ...     SteamMeasurement,
    ...     QualityEstimate,
    ...     SteamState,
    ...     ConsumerClass,
    ...     SiteConfig,
    ... )
    >>>
    >>> # Create a measurement
    >>> measurement = SteamMeasurement(
    ...     pressure_kpa=1000.0,
    ...     temperature_c=180.0,
    ...     flow_kg_s=5.0,
    ...     quality_x=0.95
    ... )
    >>>
    >>> # Serialize to JSON
    >>> json_data = measurement.model_dump_json()

Standards Compliance:
- IAPWS-IF97: Steam properties and physical bounds
- ISA-18.2: Alarm severity levels
- GreenLang: Provenance tracking and validation patterns

Author: GL-BackendDeveloper
Version: 1.0.0
"""

from __future__ import annotations

# =============================================================================
# Domain Models (Enumerations)
# =============================================================================

from .domain import (
    # Steam state enumerations
    SteamState,
    SteamRegion,
    # Event types
    EventType,
    # Severity levels
    Severity,
    AlarmPriority,
    # Consumer classifications
    ConsumerClass,
    # Data quality
    DataQualityFlag,
    EstimationMethod,
    ConstraintType,
    # Header and separator types
    HeaderType,
    SeparatorType,
    # Recommendations
    RecommendationAction,
)

# =============================================================================
# Input Models
# =============================================================================

from .inputs import (
    # Core input models
    SteamMeasurement,
    HeaderData,
    SeparatorData,
    ProcessData,
    # Request model
    QualityEstimationRequest,
    # Constants for validation
    PRESSURE_MIN_KPA,
    PRESSURE_MAX_KPA,
    PRESSURE_CRITICAL_KPA,
    TEMPERATURE_MIN_C,
    TEMPERATURE_MAX_C,
    TEMPERATURE_CRITICAL_C,
    FLOW_MIN_KG_S,
    FLOW_MAX_KG_S,
    QUALITY_MIN,
    QUALITY_MAX,
)

# =============================================================================
# Output Models
# =============================================================================

from .outputs import (
    # Quality estimation outputs
    QualityEstimate,
    CarryoverRiskAssessment,
    ContributingFactor,
    # GL-003 interface models
    QualityState,
    QualityConstraint,
    QualityConstraints,
    # Event model
    QualityEvent,
    # Complete response
    QualityEstimationResponse,
)

# =============================================================================
# Configuration Models
# =============================================================================

from .config_models import (
    # Threshold configurations
    QualityThresholds,
    PressureThresholds,
    # Equipment configurations
    ConsumerConfig,
    SeparatorConfig,
    HeaderConfig,
    # Alarm configuration
    AlarmConfig,
    # Site configuration
    SiteConfig,
    # Agent settings
    AgentSettings,
)

# =============================================================================
# Module Metadata
# =============================================================================

__version__ = "1.0.0"
__agent_id__ = "GL-012"
__agent_name__ = "SteamQual"

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Version info
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # Domain enumerations
    "SteamState",
    "SteamRegion",
    "EventType",
    "Severity",
    "AlarmPriority",
    "ConsumerClass",
    "DataQualityFlag",
    "EstimationMethod",
    "ConstraintType",
    "HeaderType",
    "SeparatorType",
    "RecommendationAction",
    # Input models
    "SteamMeasurement",
    "HeaderData",
    "SeparatorData",
    "ProcessData",
    "QualityEstimationRequest",
    # Validation constants
    "PRESSURE_MIN_KPA",
    "PRESSURE_MAX_KPA",
    "PRESSURE_CRITICAL_KPA",
    "TEMPERATURE_MIN_C",
    "TEMPERATURE_MAX_C",
    "TEMPERATURE_CRITICAL_C",
    "FLOW_MIN_KG_S",
    "FLOW_MAX_KG_S",
    "QUALITY_MIN",
    "QUALITY_MAX",
    # Output models
    "QualityEstimate",
    "CarryoverRiskAssessment",
    "ContributingFactor",
    "QualityState",
    "QualityConstraint",
    "QualityConstraints",
    "QualityEvent",
    "QualityEstimationResponse",
    # Configuration models
    "QualityThresholds",
    "PressureThresholds",
    "ConsumerConfig",
    "SeparatorConfig",
    "HeaderConfig",
    "AlarmConfig",
    "SiteConfig",
    "AgentSettings",
]
