"""
GL-016 WATERGUARD Boiler Water Treatment Agent - Core Module

This module provides the core functionality for the WATERGUARD
Boiler Water Treatment Agent, including configuration, schemas, handlers,
and coordinators for comprehensive boiler water chemistry optimization.

The WATERGUARD agent implements:
    - Cycles of Concentration (CoC) calculation and optimization
    - Blowdown optimization (continuous and intermittent)
    - Chemical dosing control for corrosion/scale prevention
    - Water chemistry monitoring and compliance
    - Zero-hallucination deterministic calculations
    - Full audit trail with SHA-256 provenance tracking

Standards Compliance:
    - ASME Boiler and Pressure Vessel Code
    - ABMA (American Boiler Manufacturers Association) Guidelines
    - IEC 62443 (Industrial Cybersecurity)
    - ISO 50001 (Energy Management Systems)

Business Value: Optimized water treatment reduces blowdown losses,
prevents scale/corrosion, and extends equipment life.

Target: Q1 2026

Example:
    >>> from gl_016_waterguard.core import WaterguardConfig, ChemistryCoordinator
    >>> config = WaterguardConfig(agent_id="GL-016-001")
    >>> coordinator = ChemistryCoordinator(config)
    >>> result = await coordinator.calculate_cycles_of_concentration(input_data)
"""

from .config import (
    # Enums
    SafetyLevel,
    OperatingMode,
    ProtocolType,
    DeploymentMode,
    ChemicalType,
    BlowdownMode,
    ConstraintType,
    QualityFlag,
    ComplianceStatus,
    # Configuration Classes
    KafkaConfig,
    OPCUAConfig,
    CMMSConfig,
    ConstraintConfig,
    ConductivityLimitsConfig,
    SilicaLimitsConfig,
    pHLimitsConfig,
    AlkalinityLimitsConfig,
    DissolvedO2LimitsConfig,
    IronLimitsConfig,
    CopperLimitsConfig,
    ChemistryLimitsConfig,
    DosingConfig,
    BlowdownConfig,
    SafetyConfig,
    MetricsConfig,
    WaterguardConfig,
)

from .schemas import (
    # Input Models
    WaterChemistryInput,
    BoilerOperatingInput,
    FeedwaterChemistryInput,
    # Result Models
    CyclesOfConcentrationResult,
    BlowdownRecommendation,
    DosingRecommendation,
    ComplianceViolation,
    ComplianceWarning,
    ConstraintDistance,
    ComplianceStatusResult,
    ChemistryState,
    # Calculation Trace
    CalculationStep,
    ProvenanceRecord,
    # Events
    WaterguardEvent,
    ChemistryEvent,
    SafetyEvent,
    AnomalyEvent,
    # API Schemas
    APIResponse,
    HealthCheckResponse,
)

from .handlers import (
    # Base Handler
    ChemistryEventHandler,
    # Specific Handlers
    SafetyEventHandler,
    ComplianceEventHandler,
    AnomalyEventHandler,
    DosingEventHandler,
    BlowdownEventHandler,
    AuditEventHandler,
    MetricsEventHandler,
)

from .coordinators import (
    # Coordinators
    ChemistryCoordinator,
    OptimizationCoordinator,
    SafetyCoordinator,
)

__all__ = [
    # Config Enums
    "SafetyLevel",
    "OperatingMode",
    "ProtocolType",
    "DeploymentMode",
    "ChemicalType",
    "BlowdownMode",
    "ConstraintType",
    "QualityFlag",
    "ComplianceStatus",
    # Configuration Classes
    "KafkaConfig",
    "OPCUAConfig",
    "CMMSConfig",
    "ConstraintConfig",
    "ConductivityLimitsConfig",
    "SilicaLimitsConfig",
    "pHLimitsConfig",
    "AlkalinityLimitsConfig",
    "DissolvedO2LimitsConfig",
    "IronLimitsConfig",
    "CopperLimitsConfig",
    "ChemistryLimitsConfig",
    "DosingConfig",
    "BlowdownConfig",
    "SafetyConfig",
    "MetricsConfig",
    "WaterguardConfig",
    # Schema Classes - Inputs
    "WaterChemistryInput",
    "BoilerOperatingInput",
    "FeedwaterChemistryInput",
    # Schema Classes - Results
    "CyclesOfConcentrationResult",
    "BlowdownRecommendation",
    "DosingRecommendation",
    "ComplianceViolation",
    "ComplianceWarning",
    "ConstraintDistance",
    "ComplianceStatusResult",
    "ChemistryState",
    # Schema Classes - Calculation Trace
    "CalculationStep",
    "ProvenanceRecord",
    # Schema Classes - Events
    "WaterguardEvent",
    "ChemistryEvent",
    "SafetyEvent",
    "AnomalyEvent",
    # Schema Classes - API
    "APIResponse",
    "HealthCheckResponse",
    # Handlers
    "ChemistryEventHandler",
    "SafetyEventHandler",
    "ComplianceEventHandler",
    "AnomalyEventHandler",
    "DosingEventHandler",
    "BlowdownEventHandler",
    "AuditEventHandler",
    "MetricsEventHandler",
    # Coordinators
    "ChemistryCoordinator",
    "OptimizationCoordinator",
    "SafetyCoordinator",
]

# Agent metadata
__version__ = "1.0.0"
__author__ = "GreenLang Team"
__agent_id__ = "GL-016"
__codename__ = "WATERGUARD"
__capability__ = "WATERGUARD"

# Standards compliance
__standards__ = [
    "ASME Boiler and Pressure Vessel Code",
    "ABMA (American Boiler Manufacturers Association)",
    "IEC 62443 (Industrial Cybersecurity)",
]


def get_agent_info() -> dict:
    """
    Get agent identification information.

    Returns:
        Dictionary containing agent metadata for registration
        and health check endpoints.
    """
    return {
        "agent_id": __agent_id__,
        "codename": __codename__,
        "capability": __capability__,
        "version": __version__,
        "author": __author__,
        "standards": __standards__,
    }
