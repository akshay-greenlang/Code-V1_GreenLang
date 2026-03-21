# -*- coding: utf-8 -*-
"""
PACK-020 Battery Passport Prep - Engines Module
=====================================================

Eight deterministic, zero-hallucination engines for EU Battery Regulation
(2023/1542) compliance preparation.

Engines:
    1. CarbonFootprintEngine         - Art 7 / Annex II lifecycle carbon footprint
    2. RecycledContentEngine         - Art 8 recycled content tracking
    3. BatteryPassportEngine         - Art 77-78 / Annex XIII passport compilation
    4. PerformanceDurabilityEngine   - Art 10 / Annex IV performance assessment
    5. SupplyChainDDEngine           - Art 48 supply chain due diligence
    6. LabellingComplianceEngine     - Art 13-14 labelling compliance
    7. EndOfLifeEngine               - Art 59-62, 71 end-of-life management
    8. ConformityAssessmentEngine    - Art 17-18 conformity assessment

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-020 Battery Passport Prep Pack
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-020"
__pack_name__: str = "Battery Passport Prep Pack"
__engines_count__: int = 8

_loaded_engines: list[str] = []

# ---------------------------------------------------------------------------
# Engine 1: Carbon Footprint
# ---------------------------------------------------------------------------
_ENGINE_1_SYMBOLS: list[str] = [
    "CarbonFootprintEngine",
    "CarbonFootprintInput",
    "CarbonFootprintResult",
    "LifecycleEmissions",
    "LifecycleStage",
    "CarbonFootprintClass",
    "BatteryCategory",
    "BatteryChemistry",
]
try:
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
    _loaded_engines.append("CarbonFootprintEngine")
except ImportError as e:
    logger.debug("Engine 1 (CarbonFootprintEngine) not available: %s", e)
    _ENGINE_1_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 2: Recycled Content
# ---------------------------------------------------------------------------
_ENGINE_2_SYMBOLS: list[str] = [
    "RecycledContentEngine",
    "RecycledContentInput",
    "RecycledContentResult",
    "MaterialInput",
    "MaterialResult",
    "CriticalRawMaterial",
    "RecycledContentPhase",
]
try:
    from .recycled_content_engine import (
        RecycledContentEngine,
        RecycledContentInput,
        RecycledContentResult,
        MaterialInput,
        MaterialResult,
        CriticalRawMaterial,
        RecycledContentPhase,
    )
    _loaded_engines.append("RecycledContentEngine")
except ImportError as e:
    logger.debug("Engine 2 (RecycledContentEngine) not available: %s", e)
    _ENGINE_2_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 3: Battery Passport
# ---------------------------------------------------------------------------
_ENGINE_3_SYMBOLS: list[str] = [
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
]
try:
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
    _loaded_engines.append("BatteryPassportEngine")
except ImportError as e:
    logger.debug("Engine 3 (BatteryPassportEngine) not available: %s", e)
    _ENGINE_3_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 4: Performance & Durability
# ---------------------------------------------------------------------------
_ENGINE_4_SYMBOLS: list[str] = [
    "PerformanceDurabilityEngine",
    "PerformanceInput",
    "PerformanceResult",
    "PerformanceMetric",
    "DurabilityRating",
    "MetricStatus",
]
try:
    from .performance_durability_engine import (
        PerformanceDurabilityEngine,
        PerformanceInput,
        PerformanceResult,
        PerformanceMetric,
        DurabilityRating,
        MetricStatus,
    )
    _loaded_engines.append("PerformanceDurabilityEngine")
except ImportError as e:
    logger.debug("Engine 4 (PerformanceDurabilityEngine) not available: %s", e)
    _ENGINE_4_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 5: Supply Chain Due Diligence
# ---------------------------------------------------------------------------
_ENGINE_5_SYMBOLS: list[str] = [
    "SupplyChainDDEngine",
    "DDResult",
    "SupplierAssessment",
    "OECDStepAssessment",
    "RiskSummary",
    "CriticalRawMaterial",
    "DueDiligenceRisk",
    "OECDStep",
    "SupplierTier",
]
try:
    from .supply_chain_dd_engine import (
        SupplyChainDDEngine,
        DDResult,
        SupplierAssessment,
        OECDStepAssessment,
        RiskSummary,
        CriticalRawMaterial as SCDDCriticalRawMaterial,
        DueDiligenceRisk,
        OECDStep,
        SupplierTier,
    )
    _loaded_engines.append("SupplyChainDDEngine")
except ImportError as e:
    logger.debug("Engine 5 (SupplyChainDDEngine) not available: %s", e)
    _ENGINE_5_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 6: Labelling Compliance
# ---------------------------------------------------------------------------
_ENGINE_6_SYMBOLS: list[str] = [
    "LabellingComplianceEngine",
    "LabelCheckResult",
    "LabelElementCheck",
    "LabelRequirement",
    "LabelElement",
    "LabelStatus",
]
try:
    from .labelling_compliance_engine import (
        LabellingComplianceEngine,
        LabelCheckResult,
        LabelElementCheck,
        LabelRequirement,
        LabelElement,
        LabelStatus,
    )
    _loaded_engines.append("LabellingComplianceEngine")
except ImportError as e:
    logger.debug("Engine 6 (LabellingComplianceEngine) not available: %s", e)
    _ENGINE_6_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 7: End of Life
# ---------------------------------------------------------------------------
_ENGINE_7_SYMBOLS: list[str] = [
    "EndOfLifeEngine",
    "EOLResult",
    "CollectionData",
    "RecyclingData",
    "MaterialRecoveryData",
    "SecondLifeAssessment",
    "RecoveryMaterial",
    "EOLPhase",
]
try:
    from .end_of_life_engine import (
        EndOfLifeEngine,
        EOLResult,
        CollectionData,
        RecyclingData,
        MaterialRecoveryData,
        SecondLifeAssessment,
        RecoveryMaterial,
        EOLPhase,
    )
    _loaded_engines.append("EndOfLifeEngine")
except ImportError as e:
    logger.debug("Engine 7 (EndOfLifeEngine) not available: %s", e)
    _ENGINE_7_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 8: Conformity Assessment
# ---------------------------------------------------------------------------
_ENGINE_8_SYMBOLS: list[str] = [
    "ConformityAssessmentEngine",
    "ConformityResult",
    "ConformityInput",
    "DocumentationItem",
    "TestResult",
    "ConformityModule",
    "DocumentationType",
    "ConformityStatus",
]
try:
    from .conformity_assessment_engine import (
        ConformityAssessmentEngine,
        ConformityResult,
        ConformityInput,
        DocumentationItem,
        TestResult,
        ConformityModule,
        DocumentationType,
        ConformityStatus,
    )
    _loaded_engines.append("ConformityAssessmentEngine")
except ImportError as e:
    logger.debug("Engine 8 (ConformityAssessmentEngine) not available: %s", e)
    _ENGINE_8_SYMBOLS = []

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_METADATA_SYMBOLS: list[str] = [
    "__version__",
    "__pack__",
    "__pack_name__",
    "__engines_count__",
]

__all__: list[str] = [
    *_METADATA_SYMBOLS,
    *_ENGINE_1_SYMBOLS,
    *_ENGINE_2_SYMBOLS,
    *_ENGINE_3_SYMBOLS,
    *_ENGINE_4_SYMBOLS,
    *_ENGINE_5_SYMBOLS,
    *_ENGINE_6_SYMBOLS,
    *_ENGINE_7_SYMBOLS,
    *_ENGINE_8_SYMBOLS,
    "get_loaded_engines",
    "get_engine_count",
]


def get_loaded_engines() -> list[str]:
    """Return list of successfully loaded engine class names."""
    return list(_loaded_engines)


def get_engine_count() -> int:
    """Return count of loaded engines."""
    return len(_loaded_engines)


logger.info(
    "PACK-020 engines: %d/%d loaded",
    len(_loaded_engines),
    __engines_count__,
)
