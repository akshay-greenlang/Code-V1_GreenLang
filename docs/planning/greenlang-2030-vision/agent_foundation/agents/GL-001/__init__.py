# -*- coding: utf-8 -*-
"""
GL-001 ProcessHeatOrchestrator Package.

Master orchestrator for all process heat operations across industrial facilities.
This package implements the THERMOSYNC agent for coordinating process heat
operations, optimizing thermal efficiency, and ensuring compliance with
industrial standards (ISO 50001, ASME).

Example:
    >>> from GL001 import ProcessHeatOrchestrator, ProcessHeatConfig, ProcessData
    >>> config = ProcessHeatConfig(...)
    >>> orchestrator = ProcessHeatOrchestrator(config)
    >>> result = await orchestrator.execute(plant_data)

Modules:
    process_heat_orchestrator: Main orchestrator implementation
    config: Configuration models and settings
    tools: Deterministic calculation tools
    calculators: Specialized calculation engines
    integrations: SCADA/ERP integration modules
"""

from .process_heat_orchestrator import ProcessHeatOrchestrator, ProcessData
from .config import (
    ProcessHeatConfig,
    PlantConfiguration,
    SensorConfiguration,
    SCADAIntegration,
    ERPIntegration,
    OptimizationParameters,
    PlantType,
    SensorType,
    IntegrationProtocol
)
from .tools import (
    ProcessHeatTools,
    ThermalEfficiencyResult,
    HeatDistributionStrategy,
    EnergyBalance,
    ComplianceResult
)

__all__ = [
    # Main orchestrator
    "ProcessHeatOrchestrator",
    "ProcessData",
    # Configuration
    "ProcessHeatConfig",
    "PlantConfiguration",
    "SensorConfiguration",
    "SCADAIntegration",
    "ERPIntegration",
    "OptimizationParameters",
    # Enums
    "PlantType",
    "SensorType",
    "IntegrationProtocol",
    # Tools and results
    "ProcessHeatTools",
    "ThermalEfficiencyResult",
    "HeatDistributionStrategy",
    "EnergyBalance",
    "ComplianceResult"
]

__version__ = "1.0.0"
__agent_id__ = "GL-001"
__codename__ = "THERMOSYNC"
__author__ = "GreenLang Team"
__description__ = "Master orchestrator for all process heat operations across industrial facilities"