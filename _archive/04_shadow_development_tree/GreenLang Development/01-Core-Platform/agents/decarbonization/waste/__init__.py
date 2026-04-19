# -*- coding: utf-8 -*-
"""
GreenLang Waste Decarbonization Agents
======================================

This package provides decarbonization planning agents for the waste
and circularity sector.

Agents:
    GL-DECARB-WST-001: WasteReductionPlannerAgent - Waste minimization planning
    GL-DECARB-WST-002: CircularEconomyAgent - Circular economy strategies
    GL-DECARB-WST-003: LandfillGasCaptureAgent - Methane capture planning
    GL-DECARB-WST-004: WasteToEnergyOptimizerAgent - WtE optimization
    GL-DECARB-WST-005: EPRPlannerAgent - Extended Producer Responsibility
    GL-DECARB-WST-006: IndustrialSymbiosisAgent - Industrial symbiosis networks

All agents follow science-based target methodologies with zero-hallucination guarantee.

Reference Standards:
    - Science Based Targets Initiative (SBTi)
    - Ellen MacArthur Foundation Circular Economy
    - EU Circular Economy Action Plan
    - EPA WARM Model
    - Zero Waste International Alliance
"""

from greenlang.agents.decarbonization.waste.base import (
    # Base classes
    BaseWasteDecarbAgent,
    WasteDecarbInput,
    WasteDecarbOutput,
    # Enums
    DecarbonizationStrategy,
    ImplementationTimeline,
    CostCategory,
    ConfidenceLevel,
    # Models
    DecarbonizationIntervention,
    DecarbonizationPathway,
)

from greenlang.agents.decarbonization.waste.waste_reduction_planner import (
    WasteReductionPlannerAgent,
    WasteReductionInput,
    WasteReductionOutput,
    WasteReductionCategory,
)

from greenlang.agents.decarbonization.waste.circular_economy_agent import (
    CircularEconomyAgent,
    CircularEconomyInput,
    CircularEconomyOutput,
    CircularStrategy,
    CircularityMetrics,
)

from greenlang.agents.decarbonization.waste.landfill_gas_capture import (
    LandfillGasCaptureAgent,
    LandfillGasCaptureInput,
    LandfillGasCaptureOutput,
    LFGUtilizationType,
    LFGCaptureProjection,
)

from greenlang.agents.decarbonization.waste.waste_to_energy_optimizer import (
    WasteToEnergyOptimizerAgent,
    WasteToEnergyInput,
    WasteToEnergyOutput,
    WtETechnology,
    OptimizationStrategy,
    WtEOptimizationScenario,
)

from greenlang.agents.decarbonization.waste.epr_planner import (
    EPRPlannerAgent,
    EPRPlannerInput,
    EPRPlannerOutput,
    EPRProductCategory,
    EPRSchemeType,
    EPRFeeStructure,
    EPRTarget,
)

from greenlang.agents.decarbonization.waste.industrial_symbiosis_agent import (
    IndustrialSymbiosisAgent,
    IndustrialSymbiosisInput,
    IndustrialSymbiosisOutput,
    WasteStreamCategory,
    SymbiosisExchange,
    SymbiosisNetworkMetrics,
)

__all__ = [
    # Base classes
    "BaseWasteDecarbAgent",
    "WasteDecarbInput",
    "WasteDecarbOutput",
    # Common enums
    "DecarbonizationStrategy",
    "ImplementationTimeline",
    "CostCategory",
    "ConfidenceLevel",
    # Common models
    "DecarbonizationIntervention",
    "DecarbonizationPathway",
    # Waste Reduction Planner (GL-DECARB-WST-001)
    "WasteReductionPlannerAgent",
    "WasteReductionInput",
    "WasteReductionOutput",
    "WasteReductionCategory",
    # Circular Economy (GL-DECARB-WST-002)
    "CircularEconomyAgent",
    "CircularEconomyInput",
    "CircularEconomyOutput",
    "CircularStrategy",
    "CircularityMetrics",
    # Landfill Gas Capture (GL-DECARB-WST-003)
    "LandfillGasCaptureAgent",
    "LandfillGasCaptureInput",
    "LandfillGasCaptureOutput",
    "LFGUtilizationType",
    "LFGCaptureProjection",
    # Waste-to-Energy Optimizer (GL-DECARB-WST-004)
    "WasteToEnergyOptimizerAgent",
    "WasteToEnergyInput",
    "WasteToEnergyOutput",
    "WtETechnology",
    "OptimizationStrategy",
    "WtEOptimizationScenario",
    # EPR Planner (GL-DECARB-WST-005)
    "EPRPlannerAgent",
    "EPRPlannerInput",
    "EPRPlannerOutput",
    "EPRProductCategory",
    "EPRSchemeType",
    "EPRFeeStructure",
    "EPRTarget",
    # Industrial Symbiosis (GL-DECARB-WST-006)
    "IndustrialSymbiosisAgent",
    "IndustrialSymbiosisInput",
    "IndustrialSymbiosisOutput",
    "WasteStreamCategory",
    "SymbiosisExchange",
    "SymbiosisNetworkMetrics",
]
