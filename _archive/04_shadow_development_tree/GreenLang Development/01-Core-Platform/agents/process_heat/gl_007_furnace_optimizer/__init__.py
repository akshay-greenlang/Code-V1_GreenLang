# -*- coding: utf-8 -*-
"""
GL-007 Furnace Optimizer and Cooling Tower Optimizer Agent

This module provides comprehensive industrial furnace and cooling tower
optimization with zero-hallucination calculations, NFPA 86 and ASHRAE
compliance, and full SHAP/LIME explainability.

Components:
    - FurnaceOptimizer: Industrial furnace optimization with combustion analysis
    - CoolingTowerOptimizer: Cooling tower performance optimization (Merkel method)
    - FurnaceHeatTransfer: Heat transfer calculations for furnaces
    - CombustionCalculator: Combustion efficiency and emissions calculations
    - NFPA86Compliance: Safety compliance validation
    - ASHRAESafetyLimits: ASHRAE guideline compliance
    - ProvenanceTracker: SHA-256 audit trail for all calculations
    - SHAPFurnaceAnalyzer: SHAP explainability for furnace optimization
    - LIMEExplainer: LIME explainability for local interpretations

Standards Compliance:
    - NFPA 86: Standard for Ovens and Furnaces
    - ASHRAE 90.1: Energy Standard for Buildings
    - ASHRAE Handbook: HVAC Systems and Equipment
    - API 560: Fired Heaters for General Refinery Service
    - ASME PTC 4: Fired Steam Generators

Example:
    >>> from greenlang.agents.process_heat.gl_007_furnace_optimizer import (
    ...     FurnaceOptimizer,
    ...     CoolingTowerOptimizer,
    ...     GL007Config,
    ... )
    >>> config = GL007Config.create_default()
    >>> optimizer = FurnaceOptimizer(config.furnace)
    >>> result = optimizer.optimize(reading)

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from greenlang.agents.process_heat.gl_007_furnace_optimizer.config import (
    GL007Config,
    FurnaceOptimizerConfig,
    CoolingTowerConfig,
    ASHRAEConfig,
    NFPA86Config,
    CombustionConfig,
    HeatTransferConfig,
    ExplainabilityConfig,
    ProvenanceConfig,
    IntegrationConfig,
    create_default_config,
    create_high_efficiency_config,
)

from greenlang.agents.process_heat.gl_007_furnace_optimizer.schemas import (
    FurnaceReading,
    CoolingTowerReading,
    OptimizationResult,
    FurnaceOptimizationResult,
    CoolingTowerOptimizationResult,
    CombustionAnalysis,
    HeatTransferAnalysis,
    SafetyStatus,
    ValidationStatus,
    OperatingMode,
)

from greenlang.agents.process_heat.gl_007_furnace_optimizer.optimizer import (
    FurnaceOptimizer,
)

from greenlang.agents.process_heat.gl_007_furnace_optimizer.cooling_tower import (
    CoolingTowerOptimizer,
)

from greenlang.agents.process_heat.gl_007_furnace_optimizer.heat_transfer import (
    FurnaceHeatTransfer,
)

from greenlang.agents.process_heat.gl_007_furnace_optimizer.combustion import (
    CombustionCalculator,
)

from greenlang.agents.process_heat.gl_007_furnace_optimizer.provenance import (
    ProvenanceTracker,
    generate_provenance_hash,
    verify_provenance_hash,
)

from greenlang.agents.process_heat.gl_007_furnace_optimizer.explainability import (
    SHAPFurnaceAnalyzer,
    LIMEExplainer,
)

from greenlang.agents.process_heat.gl_007_furnace_optimizer.safety import (
    NFPA86Compliance,
    ASHRAESafetyLimits,
    SafetyValidator,
)

from greenlang.agents.process_heat.gl_007_furnace_optimizer.integration import (
    OPCUAConnector,
    MQTTPublisher,
)


__version__ = "1.0.0"
__author__ = "GreenLang Process Heat Team"

__all__ = [
    # Configuration
    "GL007Config",
    "FurnaceOptimizerConfig",
    "CoolingTowerConfig",
    "ASHRAEConfig",
    "NFPA86Config",
    "CombustionConfig",
    "HeatTransferConfig",
    "ExplainabilityConfig",
    "ProvenanceConfig",
    "IntegrationConfig",
    "create_default_config",
    "create_high_efficiency_config",
    # Schemas
    "FurnaceReading",
    "CoolingTowerReading",
    "OptimizationResult",
    "FurnaceOptimizationResult",
    "CoolingTowerOptimizationResult",
    "CombustionAnalysis",
    "HeatTransferAnalysis",
    "SafetyStatus",
    "ValidationStatus",
    "OperatingMode",
    # Optimizers
    "FurnaceOptimizer",
    "CoolingTowerOptimizer",
    # Calculators
    "FurnaceHeatTransfer",
    "CombustionCalculator",
    # Provenance
    "ProvenanceTracker",
    "generate_provenance_hash",
    "verify_provenance_hash",
    # Explainability
    "SHAPFurnaceAnalyzer",
    "LIMEExplainer",
    # Safety
    "NFPA86Compliance",
    "ASHRAESafetyLimits",
    "SafetyValidator",
    # Integration
    "OPCUAConnector",
    "MQTTPublisher",
]
