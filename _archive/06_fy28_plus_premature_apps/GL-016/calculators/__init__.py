# -*- coding: utf-8 -*-
"""
GL-016 WATERGUARD Calculator Package

Zero-Hallucination Water Treatment Calculators

This package provides deterministic, bit-perfect calculations for:
- Water chemistry analysis (LSI, RSI, PSI, ionic strength, etc.)
- Scale formation prediction and kinetics
- Corrosion rate estimation and remaining life
- Blowdown optimization and heat recovery
- Chemical dosing calculations
- Provenance tracking with SHA-256 hashing

All calculations use:
- Decimal arithmetic for precision
- Industry standard formulas (ASTM, NACE, ASME, ABMA, ASHRAE, CTI, EPRI)
- Complete provenance tracking
- NO LLM/AI in calculation path (zero hallucination guarantee)

Author: GL-016 WATERGUARD Engineering Team
Version: 1.1.0
"""

from .provenance import (
    ProvenanceTracker,
    ProvenanceRecord,
    ProvenanceValidator,
    CalculationStep,
    create_calculation_hash
)

from .water_chemistry_calculator import (
    WaterChemistryCalculator,
    WaterSample
)

from .scale_formation_calculator import (
    ScaleFormationCalculator,
    ScaleConditions
)

from .corrosion_rate_calculator import (
    CorrosionRateCalculator,
    CorrosionConditions
)

from .blowdown_optimizer import (
    BlowdownOptimizer,
    BlowdownConditions,
    BlowdownResult,
    CoolingTowerConditions
)

from .chemical_dosing_calculator import (
    ChemicalDosingCalculator,
    ChemicalDosingResult,
    WaterConditions,
    OxygenScavengerType,
    ScaleInhibitorType,
    BiocideType
)

__version__ = "1.1.0"
__all__ = [
    # Provenance
    "ProvenanceTracker",
    "ProvenanceRecord",
    "ProvenanceValidator",
    "CalculationStep",
    "create_calculation_hash",
    # Water Chemistry
    "WaterChemistryCalculator",
    "WaterSample",
    # Scale Formation
    "ScaleFormationCalculator",
    "ScaleConditions",
    # Corrosion
    "CorrosionRateCalculator",
    "CorrosionConditions",
    # Blowdown Optimization
    "BlowdownOptimizer",
    "BlowdownConditions",
    "BlowdownResult",
    "CoolingTowerConditions",
    # Chemical Dosing
    "ChemicalDosingCalculator",
    "ChemicalDosingResult",
    "WaterConditions",
    "OxygenScavengerType",
    "ScaleInhibitorType",
    "BiocideType",
]
