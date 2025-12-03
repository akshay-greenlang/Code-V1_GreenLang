# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC - Condenser Optimization Agent Package.

This package provides comprehensive condenser performance optimization for
steam turbine condensers. It monitors cooling water conditions, vacuum levels,
and condensate flow to maximize heat transfer efficiency and minimize
backpressure losses.

Key Features:
    - Real-time condenser performance monitoring
    - Vacuum pressure optimization
    - Cooling water flow optimization
    - Heat transfer efficiency calculations
    - Air inleakage detection
    - Fouling prediction and tube cleaning recommendations
    - Cooling tower coordination
    - Turbine backpressure coordination
    - HEI (Heat Exchange Institute) compliance monitoring

Author: GreenLang Team
Date: December 2025
Status: Production Ready
"""

__version__ = "1.0.0"
__agent_id__ = "GL-017"
__codename__ = "CONDENSYNC"
__author__ = "GreenLang Team"
__description__ = "Optimizes condenser performance through vacuum, cooling water, and heat transfer management"

# Main orchestrator
from greenlang.GL_017.condenser_optimization_agent import (
    CondenserOptimizationAgent,
    CoolingWaterData,
    VacuumData,
    CondensateData,
    HeatTransferData,
    FoulingAssessment,
    OptimizationResult,
    CondenserPerformanceResult,
)

# Configuration models
from greenlang.GL_017.config import (
    AgentConfiguration,
    CondenserConfiguration,
    CondenserType,
    CoolingSystemType,
    TubePattern,
    CleaningMethod,
    FoulingType,
    CoolingWaterConfig,
    VacuumSystemConfig,
    TubeConfiguration,
    WaterQualityLimits,
    PerformanceTargets,
    AlertThresholds,
    SCADAIntegration,
    CoolingTowerIntegration,
)

__all__ = [
    # Version and metadata
    "__version__",
    "__agent_id__",
    "__codename__",
    "__author__",
    "__description__",
    # Main orchestrator
    "CondenserOptimizationAgent",
    # Data models
    "CoolingWaterData",
    "VacuumData",
    "CondensateData",
    "HeatTransferData",
    "FoulingAssessment",
    "OptimizationResult",
    "CondenserPerformanceResult",
    # Configuration
    "AgentConfiguration",
    "CondenserConfiguration",
    "CondenserType",
    "CoolingSystemType",
    "TubePattern",
    "CleaningMethod",
    "FoulingType",
    "CoolingWaterConfig",
    "VacuumSystemConfig",
    "TubeConfiguration",
    "WaterQualityLimits",
    "PerformanceTargets",
    "AlertThresholds",
    "SCADAIntegration",
    "CoolingTowerIntegration",
]
