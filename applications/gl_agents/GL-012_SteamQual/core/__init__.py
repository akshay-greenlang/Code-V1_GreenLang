# -*- coding: utf-8 -*-
"""
GL-012 STEAMQUAL SteamQualityController - Core Module

This module provides the core functionality for the STEAMQUAL
Steam Quality Controller agent including:

- Configuration management with pydantic-settings
- Deterministic seed management for reproducibility
- Event handlers for quality events and alerts
- Main orchestrator for coordinating analysis workflows

All calculations follow GreenLang zero-hallucination principles:
- Deterministic thermodynamic calculations (IAPWS-IF97)
- SHA-256 provenance tracking for audit trails
- No LLM involvement in numeric computations
- Complete reproducibility via seed management

Standards Compliance:
    - ASME PTC 19.11 (Steam and Water Sampling)
    - IAPWS-IF97 (Industrial Formulation for Water and Steam)
    - IEC 61511 (Functional Safety)
    - EU AI Act (Transparency and Reproducibility)

Example:
    >>> from core import SteamQualOrchestrator, get_settings
    >>> config = get_settings()
    >>> orchestrator = SteamQualOrchestrator(config)
    >>> result = await orchestrator.analyze_quality(measurement)

Author: GL-BackendDeveloper
Version: 1.0.0
"""

__version__ = "1.0.0"
__agent_id__ = "GL-012"
__agent_name__ = "STEAMQUAL"

# =============================================================================
# CONFIGURATION EXPORTS
# =============================================================================

from .config import (
    # Main configuration
    SteamQualConfig,
    DEFAULT_CONFIG,
    get_settings,
    get_settings_for_site,
    create_turbine_config,

    # Sub-configurations
    QualityThresholdsConfig,
    CarryoverRiskConfig,
    SeparatorEfficiencyConfig,
    CalculationConfig,
    SafetyConfig,
    MonitoringConfig,
    IntegrationConfig,
    SiteConfig,
    ProvenanceConfig,

    # Enums
    SteamPhase,
    QualityControlMode,
    CarryoverRiskLevel,
    SeparatorType,
    AlertSeverity,
    CalculationMethod,
)

# =============================================================================
# SEED MANAGEMENT EXPORTS
# =============================================================================

from .seed_management import (
    # Main classes
    SeedManager,
    SeedConfig,
    SeedState,

    # Convenience functions
    set_global_seed,
    get_reproducibility_hash,
    verify_determinism,
    get_default_manager,
    initialize_default_seeds,
    reset_default_seeds,

    # Decorators
    deterministic,
    with_provenance,
)

# =============================================================================
# EVENT HANDLER EXPORTS
# =============================================================================

from .handlers import (
    # Base classes
    EventHandler,
    EventDispatcher,

    # Event types
    EventType,

    # Event models
    SteamQualityEvent,
    QualityAlertEvent,
    CarryoverAlertEvent,
    ControlActionEvent,
    CalculationEvent,

    # Specialized handlers
    QualityEventHandler,
    CarryoverEventHandler,
    SeparatorEventHandler,
    ControlActionEventHandler,
    SafetyEventHandler,
    AuditEventHandler,
)

# =============================================================================
# ORCHESTRATOR EXPORTS
# =============================================================================

from .orchestrator import (
    # Main orchestrator
    SteamQualOrchestrator,

    # Input/Output models
    SteamMeasurement,
    QualityResult,
    CarryoverAssessment,
    ControlRecommendation,
    OrchestratorStatus,

    # Factory functions
    create_orchestrator,
    quick_quality_check,
)

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Version info
    "__version__",
    "__agent_id__",
    "__agent_name__",

    # Configuration
    "SteamQualConfig",
    "DEFAULT_CONFIG",
    "get_settings",
    "get_settings_for_site",
    "create_turbine_config",
    "QualityThresholdsConfig",
    "CarryoverRiskConfig",
    "SeparatorEfficiencyConfig",
    "CalculationConfig",
    "SafetyConfig",
    "MonitoringConfig",
    "IntegrationConfig",
    "SiteConfig",
    "ProvenanceConfig",

    # Enums
    "SteamPhase",
    "QualityControlMode",
    "CarryoverRiskLevel",
    "SeparatorType",
    "AlertSeverity",
    "CalculationMethod",

    # Seed management
    "SeedManager",
    "SeedConfig",
    "SeedState",
    "set_global_seed",
    "get_reproducibility_hash",
    "verify_determinism",
    "get_default_manager",
    "initialize_default_seeds",
    "reset_default_seeds",
    "deterministic",
    "with_provenance",

    # Event handling
    "EventHandler",
    "EventDispatcher",
    "EventType",
    "SteamQualityEvent",
    "QualityAlertEvent",
    "CarryoverAlertEvent",
    "ControlActionEvent",
    "CalculationEvent",
    "QualityEventHandler",
    "CarryoverEventHandler",
    "SeparatorEventHandler",
    "ControlActionEventHandler",
    "SafetyEventHandler",
    "AuditEventHandler",

    # Orchestrator
    "SteamQualOrchestrator",
    "SteamMeasurement",
    "QualityResult",
    "CarryoverAssessment",
    "ControlRecommendation",
    "OrchestratorStatus",
    "create_orchestrator",
    "quick_quality_check",
]


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

def get_agent_info() -> dict:
    """
    Get agent information dictionary.

    Returns:
        Dictionary with agent metadata
    """
    return {
        "agent_id": __agent_id__,
        "agent_name": __agent_name__,
        "version": __version__,
        "description": "Steam Quality Controller - Real-time steam dryness monitoring and optimization",
        "category": "steam",
        "type": "controller",
        "standards": [
            "ASME PTC 19.11",
            "IAPWS-IF97",
            "IEC 61511",
        ],
        "primary_calculations": [
            "Steam quality (dryness fraction) estimation",
            "Carryover risk assessment",
            "Separator efficiency monitoring",
            "Superheat margin calculation",
        ],
        "zero_hallucination": True,
        "provenance_tracking": True,
        "deterministic": True,
    }


def validate_installation() -> dict:
    """
    Validate core module installation.

    Returns:
        Dictionary with validation results
    """
    results = {
        "status": "ok",
        "checks": {},
        "errors": [],
    }

    # Check config loading
    try:
        config = get_settings()
        results["checks"]["config"] = "ok"
    except Exception as e:
        results["checks"]["config"] = "error"
        results["errors"].append(f"Config: {str(e)}")

    # Check seed manager
    try:
        manager = get_default_manager()
        results["checks"]["seed_manager"] = "ok"
    except Exception as e:
        results["checks"]["seed_manager"] = "error"
        results["errors"].append(f"SeedManager: {str(e)}")

    # Check orchestrator creation
    try:
        orchestrator = create_orchestrator()
        status = orchestrator.get_status()
        results["checks"]["orchestrator"] = "ok"
        results["orchestrator_status"] = status.status
    except Exception as e:
        results["checks"]["orchestrator"] = "error"
        results["errors"].append(f"Orchestrator: {str(e)}")

    # Overall status
    if results["errors"]:
        results["status"] = "error"

    return results
