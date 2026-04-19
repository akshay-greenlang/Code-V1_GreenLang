# -*- coding: utf-8 -*-
"""
PACK-023 SBTi Alignment Pack - Engines Module
===============================================

10 calculation engines for SBTi alignment lifecycle management, covering
target setting through submission readiness assessment.

Engines:
    TargetSettingEngine          -- ACA/SDA/FLAG target definition with 1.5C/WB2C/2C ambition
    CriteriaValidationEngine    -- 42-criterion automated SBTi compliance validation
    Scope3ScreeningEngine       -- 15-category Scope 3 materiality with 40% trigger
    RecalculationEngine         -- Base year recalculation with 5% significance threshold
    FLAGAssessmentEngine        -- FLAG 11-commodity assessment at 3.03%/yr linear reduction
    TemperatureRatingEngine     -- SBTi TR v2.0 with 6 portfolio aggregation methods
    ProgressTrackingEngine      -- Annual progress tracking with RAG status
    SDASectorEngine             -- SDA intensity convergence for 12 homogeneous sectors
    FIPortfolioEngine           -- FINZ V1.0 portfolio targets with PCAF scoring
    SubmissionReadinessEngine   -- 5-dimension submission readiness assessment

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-023 SBTi Alignment Pack
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-023"
__pack_name__: str = "SBTi Alignment Pack"
__engines_count__: int = 10

_loaded_engines: list[str] = []

# ---------------------------------------------------------------------------
# Engine 1: TargetSettingEngine
# ---------------------------------------------------------------------------
_TARGET_SETTING_SYMBOLS: list[str] = [
    "TargetSettingEngine",
    "TargetSettingInput",
    "TargetSettingResult",
    "TargetType",
    "AmbitionLevel",
    "PathwayMethod",
    "TargetScope",
    "BoundaryApproach",
    "EmissionsInventory",
    "PathwayMilestone",
    "ScopeTarget",
    "TargetDefinition",
    "AmbitionAssessment",
    "BaseYearValidation",
    "CoverageCheck",
]
try:
    from .target_setting_engine import (
        TargetSettingEngine,
        TargetSettingInput,
        TargetSettingResult,
        TargetType,
        AmbitionLevel,
        PathwayMethod,
        TargetScope,
        BoundaryApproach,
        EmissionsInventory,
        PathwayMilestone,
        ScopeTarget,
        TargetDefinition,
        AmbitionAssessment,
        BaseYearValidation,
        CoverageCheck,
    )
    _loaded_engines.append("TargetSettingEngine")
except ImportError as e:
    logger.debug("Engine 1 (TargetSettingEngine) not available: %s", e)
    _TARGET_SETTING_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 2: CriteriaValidationEngine
# ---------------------------------------------------------------------------
_CRITERIA_VALIDATION_SYMBOLS: list[str] = [
    "CriteriaValidationEngine",
    "ValidationInput",
    "ValidationResult",
    "CriterionStatus",
    "CriterionCategory",
    "TargetData",
    "InventoryData",
    "GovernanceData",
    "CriterionCheck",
    "GapItem",
    "ReadinessScore",
]
try:
    from .criteria_validation_engine import (
        CriteriaValidationEngine,
        ValidationInput,
        ValidationResult,
        CriterionStatus,
        CriterionCategory,
        TargetData,
        InventoryData,
        GovernanceData,
        CriterionCheck,
        GapItem,
        ReadinessScore,
    )
    _loaded_engines.append("CriteriaValidationEngine")
except ImportError as e:
    logger.debug("Engine 2 (CriteriaValidationEngine) not available: %s", e)
    _CRITERIA_VALIDATION_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 3: Scope3ScreeningEngine
# ---------------------------------------------------------------------------
_SCOPE3_SCREENING_SYMBOLS: list[str] = [
    "Scope3ScreeningEngine",
    "Scope3ScreeningInput",
    "Scope3ScreeningResult",
    "Scope3Category",
    "MaterialityLevel",
    "DataQualityTier",
    "TargetApproach",
    "ScreeningStatus",
    "ReductionPotential",
    "InfluenceLevel",
    "CategoryInput",
    "SupplierEngagementInput",
    "CategoryScreeningResult",
    "TriggerAssessment",
    "CoverageAssessment",
    "SupplierEngagementAssessment",
    "DataQualityAssessment",
    "MaterialitySummary",
    "PrioritisationResult",
    "ActionRecommendation",
]
try:
    from .scope3_screening_engine import (
        Scope3ScreeningEngine,
        Scope3ScreeningInput,
        Scope3ScreeningResult,
        Scope3Category,
        MaterialityLevel,
        DataQualityTier,
        TargetApproach,
        ScreeningStatus,
        ReductionPotential,
        InfluenceLevel,
        CategoryInput,
        SupplierEngagementInput,
        CategoryScreeningResult,
        TriggerAssessment,
        CoverageAssessment,
        SupplierEngagementAssessment,
        DataQualityAssessment,
        MaterialitySummary,
        PrioritisationResult,
        ActionRecommendation,
    )
    _loaded_engines.append("Scope3ScreeningEngine")
except ImportError as e:
    logger.debug("Engine 3 (Scope3ScreeningEngine) not available: %s", e)
    _SCOPE3_SCREENING_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 4: RecalculationEngine
# ---------------------------------------------------------------------------
_RECALCULATION_SYMBOLS: list[str] = [
    "RecalculationEngine",
    "RecalculationInput",
    "RecalculationResult",
    "RecalculationConfig",
    "RecalculationTrigger",
    "RecalculationStatus",
    "SignificanceLevel",
    "TargetAdjustment",
    "AuditEntry",
]
try:
    from .recalculation_engine import (
        RecalculationEngine,
        RecalculationInput,
        RecalculationResult,
        RecalculationConfig,
        RecalculationTrigger,
        RecalculationStatus,
        SignificanceLevel,
        TargetAdjustment,
        AuditEntry,
    )
    _loaded_engines.append("RecalculationEngine")
except ImportError as e:
    logger.debug("Engine 4 (RecalculationEngine) not available: %s", e)
    _RECALCULATION_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 5: FLAGAssessmentEngine
# ---------------------------------------------------------------------------
_FLAG_ASSESSMENT_SYMBOLS: list[str] = [
    "FLAGAssessmentEngine",
    "FLAGAssessmentInput",
    "FLAGAssessmentResult",
    "FLAGCommodity",
    "FLAGTargetType",
    "FLAGTriggerStatus",
    "DeforestationCommitmentStatus",
    "LandUseChangeType",
    "EmissionCategory",
    "AssessmentConfidence",
    "PathwayStatus",
    "LandUseChangeEntry",
    "CommodityEmissionBreakdown",
    "CommodityInput",
    "DeforestationCommitment",
    "CommodityAssessment",
    "FLAGTriggerAssessment",
    "FLAGPathwayMilestone",
    "FLAGTargetDefinition",
    "DeforestationValidation",
    "LandUseChangeQuantification",
    "FLAGProgressTracking",
]
try:
    from .flag_assessment_engine import (
        FLAGAssessmentEngine,
        FLAGAssessmentInput,
        FLAGAssessmentResult,
        FLAGCommodity,
        FLAGTargetType,
        FLAGTriggerStatus,
        DeforestationCommitmentStatus,
        LandUseChangeType,
        EmissionCategory,
        AssessmentConfidence,
        PathwayStatus,
        LandUseChangeEntry,
        CommodityEmissionBreakdown,
        CommodityInput,
        DeforestationCommitment,
        CommodityAssessment,
        FLAGTriggerAssessment,
        FLAGPathwayMilestone,
        FLAGTargetDefinition,
        DeforestationValidation,
        LandUseChangeQuantification,
        FLAGProgressTracking,
    )
    _loaded_engines.append("FLAGAssessmentEngine")
except ImportError as e:
    logger.debug("Engine 5 (FLAGAssessmentEngine) not available: %s", e)
    _FLAG_ASSESSMENT_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 6: TemperatureRatingEngine
# ---------------------------------------------------------------------------
_TEMPERATURE_RATING_SYMBOLS: list[str] = [
    "TemperatureRatingEngine",
    "TemperatureInput",
    "TemperatureResult",
    "TemperatureRatingConfig",
    "ScoreType",
    "TargetTimeframe",
    "TemperatureBand",
    "TargetValidityStatus",
    "AggregationMethod",
    "CompanyTarget",
    "CompanyScoreInput",
    "CompanyScore",
    "ContributionEntry",
    "PortfolioScore",
    "WhatIfScenario",
    "WhatIfResult",
]
try:
    from .temperature_rating_engine import (
        TemperatureRatingEngine,
        TemperatureInput,
        TemperatureResult,
        TemperatureRatingConfig,
        ScoreType,
        TargetTimeframe,
        TemperatureBand,
        TargetValidityStatus,
        AggregationMethod,
        CompanyTarget,
        CompanyScoreInput,
        CompanyScore,
        ContributionEntry,
        PortfolioScore,
        WhatIfScenario,
        WhatIfResult,
    )
    _loaded_engines.append("TemperatureRatingEngine")
except ImportError as e:
    logger.debug("Engine 6 (TemperatureRatingEngine) not available: %s", e)
    _TEMPERATURE_RATING_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 7: ProgressTrackingEngine
# ---------------------------------------------------------------------------
_PROGRESS_TRACKING_SYMBOLS: list[str] = [
    "ProgressTrackingEngine",
    "ProgressInput",
    "ProgressResult",
    "ProgressTrackingConfig",
    "TrackingStatus",
    "RAGStatus",
    "ActionPriority",
    "AnnualDataPoint",
    "TrajectoryPoint",
    "CorrectiveAction",
    "BudgetAnalysis",
]
try:
    from .progress_tracking_engine import (
        ProgressTrackingEngine,
        ProgressInput,
        ProgressResult,
        ProgressTrackingConfig,
        TrackingStatus,
        RAGStatus,
        ActionPriority,
        AnnualDataPoint,
        TrajectoryPoint,
        CorrectiveAction,
        BudgetAnalysis,
    )
    _loaded_engines.append("ProgressTrackingEngine")
except ImportError as e:
    logger.debug("Engine 7 (ProgressTrackingEngine) not available: %s", e)
    _PROGRESS_TRACKING_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 8: SDASectorEngine
# ---------------------------------------------------------------------------
_SDA_SECTOR_SYMBOLS: list[str] = [
    "SDASectorEngine",
    "SdaInput",
    "SdaResult",
    "SdaSector",
    "IntensityUnit",
    "ConvergenceStatus",
    "ValidationStatus",
    "ScopeInclusion",
    "CompanyIntensityInput",
    "CrossValidationInput",
    "AnnualMilestone",
    "ConvergenceAssessment",
    "CrossValidationResult",
    "CrossValidationSummary",
    "AbsolutePathwayPoint",
    "ProductionForecast",
    "SectorBenchmarkInfo",
    "SdaRecommendation",
]
try:
    from .sda_sector_engine import (
        SDASectorEngine,
        SdaInput,
        SdaResult,
        SdaSector,
        IntensityUnit,
        ConvergenceStatus,
        ValidationStatus,
        ScopeInclusion,
        CompanyIntensityInput,
        CrossValidationInput,
        AnnualMilestone,
        ConvergenceAssessment,
        CrossValidationResult,
        CrossValidationSummary,
        AbsolutePathwayPoint,
        ProductionForecast,
        SectorBenchmarkInfo,
        SdaRecommendation,
    )
    _loaded_engines.append("SDASectorEngine")
except ImportError as e:
    logger.debug("Engine 8 (SDASectorEngine) not available: %s", e)
    _SDA_SECTOR_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 9: FIPortfolioEngine
# ---------------------------------------------------------------------------
_FI_PORTFOLIO_SYMBOLS: list[str] = [
    "FIPortfolioEngine",
    "FIPortfolioInput",
    "FIPortfolioResult",
    "AssetClass",
    "PcafDataQuality",
    "TargetMethodology",
    "PortfolioCoverageStatus",
    "EngagementStatus",
    "TemperatureAlignment",
    "AssetClassRisk",
    "PortfolioEntityInput",
    "PortfolioTargetInput",
    "AssetClassSummary",
    "PortfolioCoverageResult",
    "EngagementTrackingResult",
    "PcafDataQualityAssessment",
    "TemperatureAlignmentResult",
    "FIRecommendation",
]
try:
    from .fi_portfolio_engine import (
        FIPortfolioEngine,
        FIPortfolioInput,
        FIPortfolioResult,
        AssetClass,
        PcafDataQuality,
        TargetMethodology,
        PortfolioCoverageStatus,
        EngagementStatus,
        TemperatureAlignment,
        AssetClassRisk,
        PortfolioEntityInput,
        PortfolioTargetInput,
        AssetClassSummary,
        PortfolioCoverageResult,
        EngagementTrackingResult,
        PcafDataQualityAssessment,
        TemperatureAlignmentResult,
        FIRecommendation,
    )
    _loaded_engines.append("FIPortfolioEngine")
except ImportError as e:
    logger.debug("Engine 9 (FIPortfolioEngine) not available: %s", e)
    _FI_PORTFOLIO_SYMBOLS = []

# ---------------------------------------------------------------------------
# Engine 10: SubmissionReadinessEngine
# ---------------------------------------------------------------------------
_SUBMISSION_READINESS_SYMBOLS: list[str] = [
    "SubmissionReadinessEngine",
    "SubmissionReadinessInput",
    "SubmissionReadinessResult",
    "ReadinessDimension",
    "OverallReadiness",
    "GapPriority",
    "EffortLevel",
    "CriterionInput",
    "CriterionAssessmentResult",
    "DimensionScoreResult",
    "TimelineEstimate",
    "GapClosureAction",
    "DocumentationChecklistItem",
    "ComplianceSummary",
]
try:
    from .submission_readiness_engine import (
        SubmissionReadinessEngine,
        SubmissionReadinessInput,
        SubmissionReadinessResult,
        ReadinessDimension,
        OverallReadiness,
        GapPriority,
        EffortLevel,
        CriterionInput,
        CriterionAssessmentResult,
        DimensionScoreResult,
        TimelineEstimate,
        GapClosureAction,
        DocumentationChecklistItem,
        ComplianceSummary,
    )
    _loaded_engines.append("SubmissionReadinessEngine")
except ImportError as e:
    logger.debug("Engine 10 (SubmissionReadinessEngine) not available: %s", e)
    _SUBMISSION_READINESS_SYMBOLS = []


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
    *_TARGET_SETTING_SYMBOLS,
    *_CRITERIA_VALIDATION_SYMBOLS,
    *_SCOPE3_SCREENING_SYMBOLS,
    *_RECALCULATION_SYMBOLS,
    *_FLAG_ASSESSMENT_SYMBOLS,
    *_TEMPERATURE_RATING_SYMBOLS,
    *_PROGRESS_TRACKING_SYMBOLS,
    *_SDA_SECTOR_SYMBOLS,
    *_FI_PORTFOLIO_SYMBOLS,
    *_SUBMISSION_READINESS_SYMBOLS,
]


def get_loaded_engines() -> list[str]:
    """Return names of successfully loaded engine classes."""
    return list(_loaded_engines)


def get_engine_count() -> int:
    """Return number of successfully loaded engines."""
    return len(_loaded_engines)


logger.info(
    "PACK-023 SBTi Alignment engines: %d/%d loaded",
    len(_loaded_engines),
    __engines_count__,
)
