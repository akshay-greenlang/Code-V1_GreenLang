"""
GL-013 PREDICTMAINT - Predictive Maintenance Agent

This module implements the PredictiveMaintenanceAgent for GreenLang.
The agent predicts equipment failures before they occur using condition
monitoring, vibration analysis, and deterministic RUL calculations.

Key Features:
    - Remaining Useful Life (RUL) calculation using Weibull analysis
    - Failure probability estimation using survival analysis
    - Vibration analysis per ISO 10816 standards
    - Thermal degradation analysis
    - Maintenance schedule optimization
    - Spare parts forecasting
    - Anomaly detection
    - CMMS integration
    - Condition monitoring system integration

Standards Compliance:
    - ISO 10816: Mechanical vibration evaluation
    - ISO 13373: Condition monitoring and diagnostics
    - ISO 13381: Prognostics and health management
    - ISO 17359: General guidelines for condition monitoring
    - ISO 55000: Asset management
    - IEC 61511: Functional safety
    - ASME: American Society of Mechanical Engineers

Zero-Hallucination Policy:
    All numeric calculations (RUL, failure probability, health scores,
    thresholds) are performed using deterministic formulas. LLM is only
    used for classification, entity resolution, and narrative generation.

Example:
    >>> from gl_013 import PredictiveMaintenanceAgent, AgentConfig
    >>> config = AgentConfig(deterministic=True, seed=42)
    >>> agent = PredictiveMaintenanceAgent(config)
    >>> result = agent.process(vibration_data, temperature_data)
    >>> print(f"Health Score: {result.health_indices.overall_health_score}")
    >>> print(f"RUL: {result.remaining_useful_life.rul_days} days")

Author: GreenLang Team
Created: 2024-12-01
Version: 1.0.0
"""

from typing import TYPE_CHECKING

# Version information
__version__ = "1.0.0"
__agent_id__ = "GL-013"
__codename__ = "PREDICTMAINT"
__author__ = "GreenLang Team"
__email__ = "support@greenlang.io"
__license__ = "Proprietary"
__status__ = "Development"

# Package metadata
__all__ = [
    # Version info
    "__version__",
    "__agent_id__",
    "__codename__",
    # Core agent
    "PredictiveMaintenanceAgent",
    "AgentConfig",
    # Input models
    "VibrationData",
    "TemperatureData",
    "PressureData",
    "OperatingHours",
    "MaintenanceHistory",
    "EquipmentParameters",
    "CMSSData",
    "CMSData",
    # Output models
    "MaintenanceSchedule",
    "FailurePrediction",
    "PartsInventory",
    "HealthIndex",
    "RemainingUsefulLife",
    "AnomalyAlert",
    "CoordinationResult",
    # Tools
    "PredictiveMaintenanceTools",
    # Calculators
    "RULCalculator",
    "FailureProbabilityCalculator",
    "VibrationAnalyzer",
    "ThermalAnalyzer",
    "AnomalyDetector",
    "HealthIndexCalculator",
    # Integrations
    "CMSSIntegration",
    "ConditionMonitoringIntegration",
    # Validators
    "validate_config",
    "ConfigValidator",
    # Exceptions
    "PredictMaintenanceError",
    "ValidationError",
    "CalculationError",
    "IntegrationError",
]

# Lazy imports for better startup performance
if TYPE_CHECKING:
    from gl_013.agent import PredictiveMaintenanceAgent
    from gl_013.config import AgentConfig
    from gl_013.models.inputs import (
        VibrationData,
        TemperatureData,
        PressureData,
        OperatingHours,
        MaintenanceHistory,
        EquipmentParameters,
        CMSSData,
        CMSData,
    )
    from gl_013.models.outputs import (
        MaintenanceSchedule,
        FailurePrediction,
        PartsInventory,
        HealthIndex,
        RemainingUsefulLife,
        AnomalyAlert,
        CoordinationResult,
    )
    from gl_013.tools import PredictiveMaintenanceTools
    from gl_013.calculators import (
        RULCalculator,
        FailureProbabilityCalculator,
        VibrationAnalyzer,
        ThermalAnalyzer,
        AnomalyDetector,
        HealthIndexCalculator,
    )
    from gl_013.integrations import (
        CMSSIntegration,
        ConditionMonitoringIntegration,
    )
    from gl_013.validate_config import validate_config, ConfigValidator
    from gl_013.exceptions import (
        PredictMaintenanceError,
        ValidationError,
        CalculationError,
        IntegrationError,
    )


def __getattr__(name: str):
    """
    Lazy import implementation for better startup performance.

    This allows the package to be imported quickly while deferring
    the loading of heavy dependencies until they are actually needed.
    """
    if name == "PredictiveMaintenanceAgent":
        from gl_013.agent import PredictiveMaintenanceAgent
        return PredictiveMaintenanceAgent
    elif name == "AgentConfig":
        from gl_013.config import AgentConfig
        return AgentConfig
    elif name in ("VibrationData", "TemperatureData", "PressureData",
                  "OperatingHours", "MaintenanceHistory", "EquipmentParameters",
                  "CMSSData", "CMSData"):
        from gl_013.models import inputs
        return getattr(inputs, name)
    elif name in ("MaintenanceSchedule", "FailurePrediction", "PartsInventory",
                  "HealthIndex", "RemainingUsefulLife", "AnomalyAlert",
                  "CoordinationResult"):
        from gl_013.models import outputs
        return getattr(outputs, name)
    elif name == "PredictiveMaintenanceTools":
        from gl_013.tools import PredictiveMaintenanceTools
        return PredictiveMaintenanceTools
    elif name in ("RULCalculator", "FailureProbabilityCalculator",
                  "VibrationAnalyzer", "ThermalAnalyzer", "AnomalyDetector",
                  "HealthIndexCalculator"):
        from gl_013 import calculators
        return getattr(calculators, name)
    elif name in ("CMSSIntegration", "ConditionMonitoringIntegration"):
        from gl_013 import integrations
        return getattr(integrations, name)
    elif name in ("validate_config", "ConfigValidator"):
        from gl_013 import validate_config as vc
        return getattr(vc, name)
    elif name in ("PredictMaintenanceError", "ValidationError",
                  "CalculationError", "IntegrationError"):
        from gl_013 import exceptions
        return getattr(exceptions, name)
    raise AttributeError(f"module 'gl_013' has no attribute '{name}'")


def get_version() -> str:
    """Return the current version of GL-013 PREDICTMAINT."""
    return __version__


def get_agent_info() -> dict:
    """
    Return comprehensive agent information.

    Returns:
        dict: Agent metadata including version, ID, codename, and capabilities.
    """
    return {
        "agent_id": __agent_id__,
        "codename": __codename__,
        "version": __version__,
        "name": "PredictiveMaintenanceAgent",
        "category": "Maintenance",
        "type": "Predictor",
        "priority": "P1",
        "tam": "$10B",
        "target_release": "Q1 2026",
        "description": "Predicts equipment failures before they occur using ML models and deterministic calculations",
        "author": __author__,
        "email": __email__,
        "status": __status__,
        "standards": [
            "ISO_10816",
            "ISO_13373",
            "ISO_13381",
            "ISO_17359",
            "ISO_55000",
            "IEC_61511",
            "ASME",
        ],
        "capabilities": [
            "remaining_useful_life_calculation",
            "failure_probability_estimation",
            "vibration_analysis",
            "thermal_degradation_analysis",
            "maintenance_scheduling_optimization",
            "spare_parts_forecasting",
            "anomaly_detection",
            "condition_monitoring",
            "cmms_integration",
            "agent_coordination",
        ],
        "zero_hallucination": True,
        "deterministic": True,
    }
