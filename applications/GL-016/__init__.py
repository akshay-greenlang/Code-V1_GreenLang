# -*- coding: utf-8 -*-
"""
GL-016 WATERGUARD - Boiler Water Treatment Agent Package.

This package provides comprehensive boiler water chemistry management to prevent
scale formation and corrosion. It integrates with SCADA systems for real-time
water quality monitoring and chemical dosing optimization.

Key Features:
    - Real-time water chemistry analysis
    - Scale and corrosion risk assessment
    - Blowdown optimization
    - Chemical dosing control
    - ASME/ABMA compliance monitoring
    - Integration with water analyzers and dosing systems

Author: GreenLang Team
Date: December 2025
Status: Production Ready
"""

__version__ = "1.0.0"
__agent_id__ = "GL-016"
__codename__ = "WATERGUARD"
__author__ = "GreenLang Team"
__description__ = "Manages boiler water chemistry to prevent scale and corrosion"

# Main orchestrator
from greenlang.GL_016.boiler_water_treatment_agent import (
    BoilerWaterTreatmentAgent,
    WaterChemistryData,
    BlowdownData,
    ChemicalDosingData,
    WaterTreatmentResult,
    ChemicalOptimizationResult,
    ScaleCorrosionRiskAssessment,
)

# Configuration models
from greenlang.GL_016.config import (
    AgentConfiguration,
    BoilerConfiguration,
    BoilerType,
    WaterSourceType,
    TreatmentProgramType,
    WaterQualityLimits,
    ChemicalInventory,
    SCADAIntegration,
    ERPIntegration,
    WaterAnalyzerConfiguration,
    ChemicalDosingSystemConfiguration,
)

__all__ = [
    # Version and metadata
    "__version__",
    "__agent_id__",
    "__codename__",
    "__author__",
    "__description__",
    # Main orchestrator
    "BoilerWaterTreatmentAgent",
    # Data models
    "WaterChemistryData",
    "BlowdownData",
    "ChemicalDosingData",
    "WaterTreatmentResult",
    "ChemicalOptimizationResult",
    "ScaleCorrosionRiskAssessment",
    # Configuration
    "AgentConfiguration",
    "BoilerConfiguration",
    "BoilerType",
    "WaterSourceType",
    "TreatmentProgramType",
    "WaterQualityLimits",
    "ChemicalInventory",
    "SCADAIntegration",
    "ERPIntegration",
    "WaterAnalyzerConfiguration",
    "ChemicalDosingSystemConfiguration",
]
