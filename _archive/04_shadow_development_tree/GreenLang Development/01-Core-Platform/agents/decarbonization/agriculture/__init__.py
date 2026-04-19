# -*- coding: utf-8 -*-
"""
GreenLang Decarbonization Agriculture Sector Agents
====================================================

Agriculture decarbonization agents for emissions reduction,
sustainable farming, and land use optimization.

Agents:
    GL-DECARB-AGR-001 to GL-DECARB-AGR-012
"""

from greenlang.agents.decarbonization.agriculture.agents import (
    CropEmissionsReductionAgent,
    LivestockEmissionsReductionAgent,
    SoilCarbonSequestrationAgent,
    FertilizerOptimizationAgent,
    AgroforestryPlannerAgent,
    RiceEmissionsReductionAgent,
    AgriculturalMachineryElectrificationAgent,
    ManureManagementAgent,
    RegenerativeAgricultureAgent,
    AgriculturalSupplyChainDecarbAgent,
    CropRotationOptimizerAgent,
    PrecisionAgricultureAgent,
)

__all__ = [
    "CropEmissionsReductionAgent",
    "LivestockEmissionsReductionAgent",
    "SoilCarbonSequestrationAgent",
    "FertilizerOptimizationAgent",
    "AgroforestryPlannerAgent",
    "RiceEmissionsReductionAgent",
    "AgriculturalMachineryElectrificationAgent",
    "ManureManagementAgent",
    "RegenerativeAgricultureAgent",
    "AgriculturalSupplyChainDecarbAgent",
    "CropRotationOptimizerAgent",
    "PrecisionAgricultureAgent",
]

AGENT_REGISTRY = {
    "GL-DECARB-AGR-001": CropEmissionsReductionAgent,
    "GL-DECARB-AGR-002": LivestockEmissionsReductionAgent,
    "GL-DECARB-AGR-003": SoilCarbonSequestrationAgent,
    "GL-DECARB-AGR-004": FertilizerOptimizationAgent,
    "GL-DECARB-AGR-005": AgroforestryPlannerAgent,
    "GL-DECARB-AGR-006": RiceEmissionsReductionAgent,
    "GL-DECARB-AGR-007": AgriculturalMachineryElectrificationAgent,
    "GL-DECARB-AGR-008": ManureManagementAgent,
    "GL-DECARB-AGR-009": RegenerativeAgricultureAgent,
    "GL-DECARB-AGR-010": AgriculturalSupplyChainDecarbAgent,
    "GL-DECARB-AGR-011": CropRotationOptimizerAgent,
    "GL-DECARB-AGR-012": PrecisionAgricultureAgent,
}
