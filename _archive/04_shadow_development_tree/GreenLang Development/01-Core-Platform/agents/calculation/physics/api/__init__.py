"""
GreenLang API Standards Library

Zero-Hallucination API Standards-Compliant Calculations

This module provides deterministic, API standards-compliant calculations
for petroleum refinery and petrochemical equipment.

Modules:
    - api_530: Heater tube thickness calculations
    - api_530_creep: Creep life assessment (Robinson's rule, Omega method)
    - api_560: Fired heater design
    - api_579: Fitness-for-service assessment
    - api_580: Risk-based inspection

All calculations provide:
    - Deterministic outputs (same input = same output)
    - Complete provenance tracking (SHA-256 hashes)
    - API standards compliance
    - Conservative design approach

Author: GreenLang Engineering Team
License: MIT
"""

# API 530 - Heater Tube Thickness
from .api_530 import (
    API530Calculator,
    API530Result,
    TubeMaterial,
    DesignMethod,
    heater_tube_thickness,
    tube_remaining_life,
)

# API 530 Enhanced - Creep Life Assessment
from .api_530_creep import (
    CreepLifeAssessor,
    CreepLifeResult,
    CreepAccumulationResult,
    OmegaMethodResult,
    CreepMaterial,
    OperatingCondition,
    CreepDataPoint,
    creep_rupture_time,
    creep_life_fraction,
    creep_remaining_life,
    assess_tube_creep,
)

# API 560 - Fired Heaters
from .api_560 import (
    API560FiredHeater,
    FiredHeaterResult,
    FiredHeaterInput,
    HeaterType,
    FlowArrangement,
    design_fired_heater,
    heater_efficiency,
)

# API 579 - Fitness-For-Service
from .api_579 import (
    API579FFS,
    FitnessForServiceResult,
    MetalLossData,
    DamageType,
    AssessmentLevel,
    assess_metal_loss,
    remaining_strength_factor,
)

# API 580 - Risk-Based Inspection
from .api_580 import (
    API580RBI,
    RBIResult,
    LikelihoodFactors,
    ConsequenceFactors,
    DamageMechanism,
    ConsequenceCategory,
    RiskLevel,
    quick_risk_assessment,
    get_inspection_interval,
)

__version__ = "1.1.0"

__all__ = [
    # API 530
    "API530Calculator",
    "API530Result",
    "TubeMaterial",
    "DesignMethod",
    "heater_tube_thickness",
    "tube_remaining_life",
    # API 530 Creep
    "CreepLifeAssessor",
    "CreepLifeResult",
    "CreepAccumulationResult",
    "OmegaMethodResult",
    "CreepMaterial",
    "OperatingCondition",
    "CreepDataPoint",
    "creep_rupture_time",
    "creep_life_fraction",
    "creep_remaining_life",
    "assess_tube_creep",
    # API 560
    "API560FiredHeater",
    "FiredHeaterResult",
    "FiredHeaterInput",
    "HeaterType",
    "FlowArrangement",
    "design_fired_heater",
    "heater_efficiency",
    # API 579
    "API579FFS",
    "FitnessForServiceResult",
    "MetalLossData",
    "DamageType",
    "AssessmentLevel",
    "assess_metal_loss",
    "remaining_strength_factor",
    # API 580
    "API580RBI",
    "RBIResult",
    "LikelihoodFactors",
    "ConsequenceFactors",
    "DamageMechanism",
    "ConsequenceCategory",
    "RiskLevel",
    "quick_risk_assessment",
    "get_inspection_interval",
]
