# -*- coding: utf-8 -*-
"""
PACK-020 Battery Passport Prep - Engine modules.

Provides four core engines for EU Battery Regulation (2023/1542) compliance:

1. CarbonFootprintEngine  - Art 7 / Annex II lifecycle carbon footprint
2. RecycledContentEngine  - Art 8 recycled content tracking
3. BatteryPassportEngine  - Art 77-78 / Annex XIII passport compilation
4. PerformanceDurabilityEngine - Art 10 / Annex IV performance assessment
"""

from .carbon_footprint_engine import (
    CarbonFootprintEngine,
    CarbonFootprintInput,
    CarbonFootprintResult,
    LifecycleEmissions,
    LifecycleStage,
    CarbonFootprintClass,
    BatteryCategory,
    BatteryChemistry,
)

from .recycled_content_engine import (
    RecycledContentEngine,
    RecycledContentInput,
    RecycledContentResult,
    MaterialInput,
    MaterialResult,
    CriticalRawMaterial,
    RecycledContentPhase,
)

from .battery_passport_engine import (
    BatteryPassportEngine,
    PassportData,
    PassportValidationResult,
    GeneralInfo,
    CarbonFootprintInfo,
    SupplyChainDD,
    MaterialComposition,
    PerformanceDurability,
    EndOfLifeInfo,
    PassportField,
    PassportStatus,
    DataQuality,
)

from .performance_durability_engine import (
    PerformanceDurabilityEngine,
    PerformanceInput,
    PerformanceResult,
    PerformanceMetric,
    DurabilityRating,
    MetricStatus,
)

__all__ = [
    # Engine 1: Carbon Footprint
    "CarbonFootprintEngine",
    "CarbonFootprintInput",
    "CarbonFootprintResult",
    "LifecycleEmissions",
    "LifecycleStage",
    "CarbonFootprintClass",
    "BatteryCategory",
    "BatteryChemistry",
    # Engine 2: Recycled Content
    "RecycledContentEngine",
    "RecycledContentInput",
    "RecycledContentResult",
    "MaterialInput",
    "MaterialResult",
    "CriticalRawMaterial",
    "RecycledContentPhase",
    # Engine 3: Battery Passport
    "BatteryPassportEngine",
    "PassportData",
    "PassportValidationResult",
    "GeneralInfo",
    "CarbonFootprintInfo",
    "SupplyChainDD",
    "MaterialComposition",
    "PerformanceDurability",
    "EndOfLifeInfo",
    "PassportField",
    "PassportStatus",
    "DataQuality",
    # Engine 4: Performance & Durability
    "PerformanceDurabilityEngine",
    "PerformanceInput",
    "PerformanceResult",
    "PerformanceMetric",
    "DurabilityRating",
    "MetricStatus",
]
