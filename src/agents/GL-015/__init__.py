# -*- coding: utf-8 -*-
"""
GL-015 INSULSCAN Insulation Inspection Agent

Industrial insulation inspection agent for detecting degradation using thermal
imaging and advanced analytics for energy conservation optimization.

Agent ID: GL-015
Codename: INSULSCAN
Version: 1.0.0
Category: Energy Conservation
Type: Monitor

This module provides the core foundation for the Insulation Inspection agent,
which analyzes thermal images to detect insulation degradation, quantify heat
losses, and generate prioritized repair recommendations based on ROI.

Key Capabilities:
    - Thermal image analysis and hotspot detection
    - Heat loss quantification with energy cost impact
    - Insulation degradation severity assessment
    - Repair priority ranking with ROI calculations
    - Multi-framework compliance (ASTM, CINI, ISO, ASHRAE)
    - Integration with thermal cameras and CMMS systems

Zero-Hallucination Guarantee:
    All thermal calculations use deterministic engineering formulas.
    No LLM inference for numeric values - only validated heat transfer equations.

Example:
    >>> from agents.GL_015 import InsulationInspectionAgent
    >>> from agents.GL_015.config import InsulationConfig
    >>>
    >>> config = InsulationConfig()
    >>> agent = InsulationInspectionAgent(config)
    >>> result = agent.process(thermal_image_data, ambient_conditions, equipment_params)

Author: GreenLang AI Agent Factory
License: Apache-2.0
Repository: https://github.com/greenlang/agents/gl-015
Documentation: https://docs.greenlang.io/agents/gl-015
"""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING, Any, List

__version__ = "1.0.0"
__agent_id__ = "GL-015"
__agent_codename__ = "INSULSCAN"
__agent_name__ = "InsulationInspectionAgent"
__author__ = "GreenLang AI Agent Factory"
__license__ = "Apache-2.0"

# Public API exports - these are lazily loaded for performance
__all__: List[str] = [
    # Agent metadata
    "__version__",
    "__agent_id__",
    "__agent_codename__",
    "__agent_name__",
    # Core enumerations
    "InsulationType",
    "DegradationSeverity",
    "RepairPriority",
    "SurfaceType",
    "EquipmentType",
    "WeatherCondition",
    "CameraType",
    "EmissivityClass",
    "FailureMode",
    "InspectionMethod",
    "InsulationMaterial",
    "JacketMaterial",
    "DamageType",
    "MoistureState",
    "CorrosionRisk",
    # Input models
    "ThermalImageData",
    "TemperatureMatrix",
    "Hotspot",
    "RegionOfInterest",
    "AmbientConditions",
    "EquipmentParameters",
    "InsulationSpecifications",
    "HistoricalData",
    "InsulationInspectionInput",
    # Output models
    "HeatLossAnalysis",
    "DegradationAssessment",
    "RepairRecommendation",
    "RepairPriorities",
    "InspectionReport",
    "EconomicImpact",
    "InsulationInspectionOutput",
    # Configuration
    "InsulationConfig",
    "Settings",
    # Validation results
    "ValidationResult",
    "ValidationError",
]


# Lazy loading implementation for improved import performance
# This pattern defers module imports until they are actually needed
def __getattr__(name: str) -> Any:
    """
    Lazy loading of module components.

    This function implements PEP 562 lazy module loading to improve
    import performance by deferring the loading of submodules until
    they are actually accessed.

    Args:
        name: The name of the attribute being accessed.

    Returns:
        The requested attribute from the appropriate submodule.

    Raises:
        AttributeError: If the requested attribute does not exist.
    """
    # Mapping of public names to their source modules
    _lazy_imports = {
        # Enumerations from config module
        "InsulationType": "config",
        "DegradationSeverity": "config",
        "RepairPriority": "config",
        "SurfaceType": "config",
        "EquipmentType": "config",
        "WeatherCondition": "config",
        "CameraType": "config",
        "EmissivityClass": "config",
        "FailureMode": "config",
        "InspectionMethod": "config",
        "InsulationMaterial": "config",
        "JacketMaterial": "config",
        "DamageType": "config",
        "MoistureState": "config",
        "CorrosionRisk": "config",
        # Input models from config module
        "ThermalImageData": "config",
        "TemperatureMatrix": "config",
        "Hotspot": "config",
        "RegionOfInterest": "config",
        "AmbientConditions": "config",
        "EquipmentParameters": "config",
        "InsulationSpecifications": "config",
        "HistoricalData": "config",
        "InsulationInspectionInput": "config",
        # Output models from config module
        "HeatLossAnalysis": "config",
        "DegradationAssessment": "config",
        "RepairRecommendation": "config",
        "RepairPriorities": "config",
        "InspectionReport": "config",
        "EconomicImpact": "config",
        "InsulationInspectionOutput": "config",
        # Configuration from config module
        "InsulationConfig": "config",
        "Settings": "config",
        # Validation from config module
        "ValidationResult": "config",
        "ValidationError": "config",
    }

    if name in _lazy_imports:
        module_name = _lazy_imports[name]
        # Import the submodule relative to this package
        module = importlib.import_module(f".{module_name}", __package__)
        # Get the attribute from the submodule
        value = getattr(module, name)
        # Cache the attribute in the module namespace for future access
        setattr(sys.modules[__name__], name, value)
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    """
    Provide accurate attribute listing for the module.

    This enables proper auto-completion and introspection tools
    to discover all available public attributes.

    Returns:
        List of all public attribute names.
    """
    # Combine module-level attributes with lazy-loaded attributes
    module_attrs = list(globals().keys())
    return sorted(set(module_attrs + __all__))


# Type checking imports - these are only used for static analysis
# and do not affect runtime performance
if TYPE_CHECKING:
    from .config import (
        # Enumerations
        InsulationType,
        DegradationSeverity,
        RepairPriority,
        SurfaceType,
        EquipmentType,
        WeatherCondition,
        CameraType,
        EmissivityClass,
        FailureMode,
        InspectionMethod,
        InsulationMaterial,
        JacketMaterial,
        DamageType,
        MoistureState,
        CorrosionRisk,
        # Input models
        ThermalImageData,
        TemperatureMatrix,
        Hotspot,
        RegionOfInterest,
        AmbientConditions,
        EquipmentParameters,
        InsulationSpecifications,
        HistoricalData,
        InsulationInspectionInput,
        # Output models
        HeatLossAnalysis,
        DegradationAssessment,
        RepairRecommendation,
        RepairPriorities,
        InspectionReport,
        EconomicImpact,
        InsulationInspectionOutput,
        # Configuration
        InsulationConfig,
        Settings,
        # Validation
        ValidationResult,
        ValidationError,
    )
