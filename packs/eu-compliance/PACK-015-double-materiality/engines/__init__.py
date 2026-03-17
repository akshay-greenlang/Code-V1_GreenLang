# -*- coding: utf-8 -*-
"""
PACK-015 Double Materiality Pack - Engines Module
====================================================

Deterministic, zero-hallucination calculation engines for conducting
Double Materiality Assessments (DMA) per ESRS 1 Chapter 3 and ESRS 2
(IRO-1, IRO-2, SBM-3).

Every engine produces bit-perfect reproducible results with SHA-256
provenance hashing.  No LLM is used in any scoring, classification,
or calculation path.

Engines:
    1. ImpactMaterialityEngine         - Impact materiality scoring (ESRS 1 Para 43-48)
    2. FinancialMaterialityEngine      - Financial materiality scoring (ESRS 1 Para 49-51)
    3. StakeholderEngagementEngine     - Stakeholder identification and consultation (ESRS 1 Para 22-23)
    4. IROIdentificationEngine         - IRO identification and register (ESRS 1 Para 28-39)
    5. MaterialityMatrixEngine         - Double materiality matrix generation
    6. ESRSTopicMappingEngine          - ESRS disclosure requirement mapping
    7. ThresholdScoringEngine          - Configurable thresholds and scoring methodologies
    8. DMAReportEngine                 - DMA report assembly with methodology documentation

Regulatory Basis:
    EU Directive 2022/2464 (CSRD)
    EU Delegated Regulation 2023/2772 (ESRS Set 1)
    ESRS 1 Chapter 3: Double materiality methodology
    ESRS 2 General Disclosures (IRO-1, IRO-2, SBM-3)
    EFRAG Implementation Guidance IG 1 (Materiality Assessment, 2024)
    EU Directive 2026/470 (Omnibus I revised thresholds)

Pack Tier: Standalone Cross-Sector (PACK-015)
Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-015"
__pack_name__: str = "Double Materiality Pack"
__engines_count__: int = 8

# Track which engines loaded successfully
_loaded_engines: list[str] = []


# ===================================================================
# Engine 1: Impact Materiality
# ===================================================================
_ENGINE_1_SYMBOLS: list[str] = [
    "ImpactMaterialityEngine",
    "SustainabilityMatter",
    "ImpactAssessment",
    "ImpactMaterialityResult",
    "BatchImpactResult",
    "IM_TimeHorizon",
    "IM_ValueChainStage",
    "IM_ESRSTopic",
    "ImpactType",
    "ScaleLevel",
    "ESRS_SUSTAINABILITY_MATTERS",
    "SCALE_WEIGHTS",
    "SCOPE_WEIGHTS",
    "IRREMEDIABILITY_WEIGHTS",
]

try:
    from .impact_materiality_engine import (
        ESRS_SUSTAINABILITY_MATTERS,
        IRREMEDIABILITY_WEIGHTS,
        SCALE_WEIGHTS,
        SCOPE_WEIGHTS,
        BatchImpactResult,
        ESRSTopic as IM_ESRSTopic,
        ImpactAssessment,
        ImpactMaterialityEngine,
        ImpactMaterialityResult,
        ImpactType,
        ScaleLevel,
        SustainabilityMatter,
        TimeHorizon as IM_TimeHorizon,
        ValueChainStage as IM_ValueChainStage,
    )
    _loaded_engines.append("ImpactMaterialityEngine")
except ImportError as e:
    logger.debug("Engine 1 (ImpactMaterialityEngine) not available: %s", e)
    _ENGINE_1_SYMBOLS = []


# ===================================================================
# Engine 2: Financial Materiality
# ===================================================================
_ENGINE_2_SYMBOLS: list[str] = [
    "FinancialMaterialityEngine",
    "FinancialImpact",
    "FinancialMaterialityResult",
    "BatchFinancialResult",
    "FM_TimeHorizon",
    "FinancialMagnitude",
    "FinancialLikelihood",
    "AffectedResource",
    "RiskOrOpportunity",
    "FM_ESRSTopic",
    "TIME_HORIZON_WEIGHTS",
    "MAGNITUDE_DESCRIPTIONS",
    "LIKELIHOOD_DESCRIPTIONS",
    "FINANCIAL_KPI_MAP",
]

try:
    from .financial_materiality_engine import (
        FINANCIAL_KPI_MAP,
        LIKELIHOOD_DESCRIPTIONS,
        MAGNITUDE_DESCRIPTIONS,
        TIME_HORIZON_WEIGHTS,
        AffectedResource,
        BatchFinancialResult,
        ESRSTopic as FM_ESRSTopic,
        FinancialImpact,
        FinancialLikelihood,
        FinancialMagnitude,
        FinancialMaterialityEngine,
        FinancialMaterialityResult,
        RiskOrOpportunity,
        TimeHorizon as FM_TimeHorizon,
    )
    _loaded_engines.append("FinancialMaterialityEngine")
except ImportError as e:
    logger.debug("Engine 2 (FinancialMaterialityEngine) not available: %s", e)
    _ENGINE_2_SYMBOLS = []


# ===================================================================
# Engine 3: Stakeholder Engagement
# ===================================================================
_ENGINE_3_SYMBOLS: list[str] = [
    "StakeholderEngagementEngine",
    "Stakeholder",
    "ConsultationRecord",
    "StakeholderEngagementResult",
    "StakeholderCategory",
    "EngagementMethod",
    "ConsultationStatus",
    "SECTOR_STAKEHOLDER_MAP",
    "ENGAGEMENT_QUALITY_CRITERIA",
]

try:
    from .stakeholder_engagement_engine import (
        ENGAGEMENT_QUALITY_CRITERIA,
        SECTOR_STAKEHOLDER_MAP,
        ConsultationRecord,
        ConsultationStatus,
        EngagementMethod,
        Stakeholder,
        StakeholderCategory,
        StakeholderEngagementEngine,
        StakeholderEngagementResult,
    )
    _loaded_engines.append("StakeholderEngagementEngine")
except ImportError as e:
    logger.debug("Engine 3 (StakeholderEngagementEngine) not available: %s", e)
    _ENGINE_3_SYMBOLS = []


# ===================================================================
# Engine 4: IRO Identification
# ===================================================================
_ENGINE_4_SYMBOLS: list[str] = [
    "IROIdentificationEngine",
    "IRO",
    "IROAssessment",
    "IRORegister",
    "IROType",
    "PriorityLevel",
    "IRO_ESRSTopic",
    "IRO_ValueChainStage",
    "IRO_TimeHorizon",
    "ESRS_IRO_CATALOG",
    "SECTOR_IRO_PRIORITIES",
]

try:
    from .iro_identification_engine import (
        ESRS_IRO_CATALOG,
        SECTOR_IRO_PRIORITIES,
        ESRSTopic as IRO_ESRSTopic,
        IRO,
        IROAssessment,
        IROIdentificationEngine,
        IRORegister,
        IROType,
        PriorityLevel,
        TimeHorizon as IRO_TimeHorizon,
        ValueChainStage as IRO_ValueChainStage,
    )
    _loaded_engines.append("IROIdentificationEngine")
except ImportError as e:
    logger.debug("Engine 4 (IROIdentificationEngine) not available: %s", e)
    _ENGINE_4_SYMBOLS = []


# ===================================================================
# Engine 5: Materiality Matrix
# ===================================================================
_ENGINE_5_SYMBOLS: list[str] = [
    "MaterialityMatrixEngine",
    "MatrixEntry",
    "MaterialityMatrix",
    "MatrixVisualizationData",
    "MatrixDelta",
    "ScoreChange",
    "ImpactScoreInput",
    "FinancialScoreInput",
    "Quadrant",
    "MatrixLayout",
    "CombinedScoreMethod",
    "DEFAULT_IMPACT_THRESHOLD",
    "DEFAULT_FINANCIAL_THRESHOLD",
    "QUADRANT_DESCRIPTIONS",
    "COLOR_MAP",
]

try:
    from .materiality_matrix_engine import (
        COLOR_MAP,
        DEFAULT_FINANCIAL_THRESHOLD,
        DEFAULT_IMPACT_THRESHOLD,
        QUADRANT_DESCRIPTIONS,
        CombinedScoreMethod,
        FinancialScoreInput,
        ImpactScoreInput,
        MaterialityMatrix,
        MaterialityMatrixEngine,
        MatrixDelta,
        MatrixEntry,
        MatrixLayout,
        MatrixVisualizationData,
        Quadrant,
        ScoreChange,
    )
    _loaded_engines.append("MaterialityMatrixEngine")
except ImportError as e:
    logger.debug("Engine 5 (MaterialityMatrixEngine) not available: %s", e)
    _ENGINE_5_SYMBOLS = []


# ===================================================================
# Engine 6: ESRS Topic Mapping
# ===================================================================
_ENGINE_6_SYMBOLS: list[str] = [
    "ESRSTopicMappingEngine",
    "ESRSDisclosureRequirement",
    "TopicMapping",
    "ESRSMappingResult",
    "DisclosureGap",
    "MaterialTopicInput",
    "AvailableDataInput",
    "ESRSStandard",
    "DisclosureStatus",
    "CoverageLevel",
    "ESRS_DISCLOSURE_CATALOG",
    "EFFORT_ESTIMATES_PER_DISCLOSURE",
    "TOPIC_TO_STANDARD",
]

try:
    from .esrs_topic_mapping_engine import (
        EFFORT_ESTIMATES_PER_DISCLOSURE,
        ESRS_DISCLOSURE_CATALOG,
        TOPIC_TO_STANDARD,
        AvailableDataInput,
        CoverageLevel,
        DisclosureGap,
        DisclosureStatus,
        ESRSDisclosureRequirement,
        ESRSMappingResult,
        ESRSStandard,
        ESRSTopicMappingEngine,
        MaterialTopicInput,
        TopicMapping,
    )
    _loaded_engines.append("ESRSTopicMappingEngine")
except ImportError as e:
    logger.debug("Engine 6 (ESRSTopicMappingEngine) not available: %s", e)
    _ENGINE_6_SYMBOLS = []


# ===================================================================
# Engine 7: Threshold Scoring
# ===================================================================
_ENGINE_7_SYMBOLS: list[str] = [
    "ThresholdScoringEngine",
    "ScoringProfile",
    "ThresholdSet",
    "ScoringResult",
    "SensitivityAnalysis",
    "SensitivityPoint",
    "RawScoreInput",
    "ScoringMethodology",
    "NormalizationMethod",
    "ThresholdSource",
    "INDUSTRY_THRESHOLDS",
    "SECTOR_ADJUSTMENT_FACTORS",
    "METHODOLOGY_DESCRIPTIONS",
]

try:
    from .threshold_scoring_engine import (
        INDUSTRY_THRESHOLDS,
        METHODOLOGY_DESCRIPTIONS,
        SECTOR_ADJUSTMENT_FACTORS,
        NormalizationMethod,
        RawScoreInput,
        ScoringMethodology,
        ScoringProfile,
        ScoringResult,
        SensitivityAnalysis,
        SensitivityPoint,
        ThresholdScoringEngine,
        ThresholdSet,
        ThresholdSource,
    )
    _loaded_engines.append("ThresholdScoringEngine")
except ImportError as e:
    logger.debug("Engine 7 (ThresholdScoringEngine) not available: %s", e)
    _ENGINE_7_SYMBOLS = []


# ===================================================================
# Engine 8: DMA Report
# ===================================================================
_ENGINE_8_SYMBOLS: list[str] = [
    "DMAReportEngine",
    "DMAReport",
    "DMAMethodology",
    "DMASection",
    "DMADelta",
    "ReportScoreChange",
    "StakeholderSummary",
    "IROEntry",
    "ReportAssemblyInput",
    "ReportFormat",
    "SectionType",
    "REPORT_SECTIONS_ORDER",
    "SECTION_TEMPLATES",
    "ESRS_2_DISCLOSURE_REQUIREMENTS",
]

try:
    from .dma_report_engine import (
        ESRS_2_DISCLOSURE_REQUIREMENTS,
        REPORT_SECTIONS_ORDER,
        SECTION_TEMPLATES,
        DMADelta,
        DMAMethodology,
        DMAReport,
        DMAReportEngine,
        DMASection,
        IROEntry,
        ReportAssemblyInput,
        ReportFormat,
        ReportScoreChange,
        SectionType,
        StakeholderSummary,
    )
    _loaded_engines.append("DMAReportEngine")
except ImportError as e:
    logger.debug("Engine 8 (DMAReportEngine) not available: %s", e)
    _ENGINE_8_SYMBOLS = []


# ===================================================================
# Public API - dynamically collected from successfully loaded engines
# ===================================================================

_METADATA_SYMBOLS: list[str] = [
    "__version__",
    "__pack__",
    "__pack_name__",
    "__engines_count__",
]

__all__: list[str] = [
    *_METADATA_SYMBOLS,
    # Engine 1: Impact Materiality
    *_ENGINE_1_SYMBOLS,
    # Engine 2: Financial Materiality
    *_ENGINE_2_SYMBOLS,
    # Engine 3: Stakeholder Engagement
    *_ENGINE_3_SYMBOLS,
    # Engine 4: IRO Identification
    *_ENGINE_4_SYMBOLS,
    # Engine 5: Materiality Matrix
    *_ENGINE_5_SYMBOLS,
    # Engine 6: ESRS Topic Mapping
    *_ENGINE_6_SYMBOLS,
    # Engine 7: Threshold Scoring
    *_ENGINE_7_SYMBOLS,
    # Engine 8: DMA Report
    *_ENGINE_8_SYMBOLS,
]


def get_loaded_engines() -> list[str]:
    """Return list of engine class names that loaded successfully."""
    return list(_loaded_engines)


def get_engine_count() -> int:
    """Return count of engines that loaded successfully."""
    return len(_loaded_engines)


logger.info(
    "PACK-015 Double Materiality engines: %d/%d loaded",
    len(_loaded_engines),
    __engines_count__,
)
