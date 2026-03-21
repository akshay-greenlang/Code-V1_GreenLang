# -*- coding: utf-8 -*-
"""
PACK-026 SME Net Zero Pack - Engines Module
==============================================

Deterministic, zero-hallucination calculation engines optimized for
small and medium enterprises (SMEs).  Each engine is designed for
minimal data requirements, rapid setup, and actionable outputs that
SMEs can implement without dedicated sustainability teams.

Every engine produces bit-perfect reproducible results with SHA-256
provenance hashing.  No LLM is used in any scoring, classification,
or calculation path.

Engines:
    1. SMEBaselineEngine              - Three-tier GHG baseline (Bronze/Silver/Gold)
    2. SimplifiedTargetEngine         - Hard-coded 1.5C pathway (ACA, no SDA)
    3. QuickWinsEngine                - 50+ SME quick-win actions with scoring
    4. Scope3EstimatorEngine          - Spend-based Scope 3 with accounting integration
    5. ActionPrioritizationEngine     - MACC-lite NPV/IRR/payback prioritization
    6. CostBenefitEngine              - Financial CBA with grant adjustment
    7. GrantFinderEngine              - 50+ grant programs with matching algorithm
    8. CertificationReadinessEngine   - 6-pathway readiness with gap analysis

Design Principles:
    - Three-tier data approach (Bronze: 15 min, Silver: 1 hour, Gold: 2-3 hours)
    - Spend-based Scope 3 (DEFRA/EPA EEIO factors, no activity-based hybrid)
    - SME-specific industry averages by NACE code + company size
    - Accounting software integration ready (Xero/QuickBooks mappings)
    - Performance: <2 seconds for baseline, <30 seconds for full roadmap

Regulatory / Framework Basis:
    GHG Protocol Corporate Standard (2004, revised 2015)
    GHG Protocol Scope 3 Standard (2011)
    SBTi SME Target Setting Route (2023)
    SBTi Corporate Net-Zero Standard v1.2 (2024)
    IPCC AR6 WG1 (2021) - GWP-100 values
    DEFRA/BEIS 2024 UK GHG Conversion Factors
    US EPA EEIO v2.0 - Spend-based emission factors
    SME Climate Hub (UN Race to Zero)
    Paris Agreement (2015) - 1.5C temperature target

Pack Tier: SME (PACK-026)
Category: Net Zero Packs
Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-026"
__pack_name__: str = "SME Net Zero Pack"
__engines_count__: int = 8

# --- Engine 1: SME Baseline ---------------------------------------------------
_engine_1_symbols: list[str] = []
try:
    from .sme_baseline_engine import (  # noqa: F401
        DataTier,
        CompanySize,
        SMEFuelType,
        SMESector,
        Scope3SMECategory,
        DataQualityLevel,
        SMEFuelEntry,
        SMEElectricityEntry,
        SMERefrigerantEntry,
        SMESpendEntry,
        SMEVehicleEntry,
        SMEBaselineInput,
        ScopeBreakdown,
        AccuracyBand,
        IntensityMetrics,
        DataQualityAssessment,
        NextStepRecommendation,
        SMEBaselineResult,
        SMEBaselineEngine,
    )
    _engine_1_symbols = [
        "DataTier", "CompanySize", "SMEFuelType", "SMESector",
        "Scope3SMECategory", "DataQualityLevel",
        "SMEFuelEntry", "SMEElectricityEntry", "SMERefrigerantEntry",
        "SMESpendEntry", "SMEVehicleEntry", "SMEBaselineInput",
        "ScopeBreakdown", "AccuracyBand", "IntensityMetrics",
        "DataQualityAssessment", "NextStepRecommendation",
        "SMEBaselineResult", "SMEBaselineEngine",
    ]
    logger.debug("Engine 1 (SMEBaselineEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 1 (SMEBaselineEngine) not available: %s", exc)

# --- Engine 2: Simplified Target ----------------------------------------------
_engine_2_symbols: list[str] = []
try:
    from .simplified_target_engine import (  # noqa: F401
        TargetAmbition,
        ScopeInclusion,
        MilestoneStatus,
        TargetCommitment,
        TargetInput,
        MilestoneEntry,
        TargetDefinition,
        ProgressAssessment,
        Scope3Coverage,
        TargetStatement,
        SimplifiedTargetResult,
        SimplifiedTargetEngine,
    )
    _engine_2_symbols = [
        "TargetAmbition", "ScopeInclusion", "MilestoneStatus",
        "TargetCommitment", "TargetInput", "MilestoneEntry",
        "TargetDefinition", "ProgressAssessment", "Scope3Coverage",
        "TargetStatement", "SimplifiedTargetResult", "SimplifiedTargetEngine",
    ]
    logger.debug("Engine 2 (SimplifiedTargetEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 2 (SimplifiedTargetEngine) not available: %s", exc)

# --- Engine 3: Quick Wins ------------------------------------------------------
_engine_3_symbols: list[str] = []
try:
    from .quick_wins_engine import (  # noqa: F401
        ActionCategory,
        DifficultyLevel,
        ScopeImpact,
        TimelinePhase,
        SMESectorFilter,
        QuickWinsInput,
        QuickWinAction,
        QuickWinsSummary,
        QuickWinsResult,
        QuickWinsEngine,
    )
    _engine_3_symbols = [
        "ActionCategory", "DifficultyLevel", "ScopeImpact",
        "TimelinePhase", "SMESectorFilter",
        "QuickWinsInput", "QuickWinAction", "QuickWinsSummary",
        "QuickWinsResult", "QuickWinsEngine",
    ]
    logger.debug("Engine 3 (QuickWinsEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 3 (QuickWinsEngine) not available: %s", exc)

# --- Engine 4: Scope 3 Estimator -----------------------------------------------
_engine_4_symbols: list[str] = []
try:
    from .scope3_estimator_engine import (  # noqa: F401
        Scope3Category,
        SpendCurrency,
        DataSourceType,
        IndustryType,
        DataQualityDimension,
        SpendEntry,
        CommutingEstimateInput,
        Scope3EstimatorInput,
        CategoryEstimate,
        DataQualityScore,
        Scope3EstimatorResult,
        Scope3EstimatorEngine,
    )
    _engine_4_symbols = [
        "Scope3Category", "SpendCurrency", "DataSourceType",
        "IndustryType", "DataQualityDimension",
        "SpendEntry", "CommutingEstimateInput", "Scope3EstimatorInput",
        "CategoryEstimate", "DataQualityScore",
        "Scope3EstimatorResult", "Scope3EstimatorEngine",
    ]
    logger.debug("Engine 4 (Scope3EstimatorEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 4 (Scope3EstimatorEngine) not available: %s", exc)

# --- Engine 5: Action Prioritization -------------------------------------------
_engine_5_symbols: list[str] = []
try:
    from .action_prioritization_engine import (  # noqa: F401
        ActionScope,
        ActionEase,
        RoadmapPhase,
        SensitivityScenario,
        ActionInput,
        PrioritizationInput,
        FinancialMetrics,
        SensitivityResult,
        PrioritizedAction,
        RoadmapSummary,
        PrioritizationResult,
        ActionPrioritizationEngine,
    )
    _engine_5_symbols = [
        "ActionScope", "ActionEase", "RoadmapPhase", "SensitivityScenario",
        "ActionInput", "PrioritizationInput",
        "FinancialMetrics", "SensitivityResult", "PrioritizedAction",
        "RoadmapSummary", "PrioritizationResult",
        "ActionPrioritizationEngine",
    ]
    logger.debug("Engine 5 (ActionPrioritizationEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 5 (ActionPrioritizationEngine) not available: %s", exc)

# --- Engine 6: Cost-Benefit ----------------------------------------------------
_engine_6_symbols: list[str] = []
try:
    from .cost_benefit_engine import (  # noqa: F401
        CostCategory,
        ScenarioType,
        RiskLevel,
        CostBenefitItem,
        CostBenefitInput,
        YearCashFlow,
        ScenarioAnalysis,
        ItemAnalysis,
        PortfolioSummary,
        CostBenefitResult,
        CostBenefitEngine,
    )
    _engine_6_symbols = [
        "CostCategory", "ScenarioType", "RiskLevel",
        "CostBenefitItem", "CostBenefitInput",
        "YearCashFlow", "ScenarioAnalysis", "ItemAnalysis",
        "PortfolioSummary", "CostBenefitResult", "CostBenefitEngine",
    ]
    logger.debug("Engine 6 (CostBenefitEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 6 (CostBenefitEngine) not available: %s", exc)

# --- Engine 7: Grant Finder ----------------------------------------------------
_engine_7_symbols: list[str] = []
try:
    from .grant_finder_engine import (  # noqa: F401
        GrantRegion,
        ProjectType,
        IndustryCode,
        GrantStatus,
        GrantFinderInput,
        GrantMatch,
        GrantFinderResult,
        GrantFinderEngine,
    )
    _engine_7_symbols = [
        "GrantRegion", "ProjectType", "IndustryCode", "GrantStatus",
        "GrantFinderInput", "GrantMatch", "GrantFinderResult",
        "GrantFinderEngine",
    ]
    logger.debug("Engine 7 (GrantFinderEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 7 (GrantFinderEngine) not available: %s", exc)

# --- Engine 8: Certification Readiness -----------------------------------------
_engine_8_symbols: list[str] = []
try:
    from .certification_readiness_engine import (  # noqa: F401
        CertificationPathway,
        ReadinessDimension,
        ReadinessLevel,
        GapSeverity,
        DimensionInput,
        CertificationReadinessInput,
        DimensionScore,
        GapRemediationItem,
        CertificationAssessment,
        CertificationReadinessResult,
        CertificationReadinessEngine,
    )
    _engine_8_symbols = [
        "CertificationPathway", "ReadinessDimension", "ReadinessLevel",
        "GapSeverity", "DimensionInput", "CertificationReadinessInput",
        "DimensionScore", "GapRemediationItem", "CertificationAssessment",
        "CertificationReadinessResult", "CertificationReadinessEngine",
    ]
    logger.debug("Engine 8 (CertificationReadinessEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 8 (CertificationReadinessEngine) not available: %s", exc)

# --- Dynamic __all__ -----------------------------------------------------------

_loaded_engines: list[str] = []
if _engine_1_symbols:
    _loaded_engines.append("SMEBaselineEngine")
if _engine_2_symbols:
    _loaded_engines.append("SimplifiedTargetEngine")
if _engine_3_symbols:
    _loaded_engines.append("QuickWinsEngine")
if _engine_4_symbols:
    _loaded_engines.append("Scope3EstimatorEngine")
if _engine_5_symbols:
    _loaded_engines.append("ActionPrioritizationEngine")
if _engine_6_symbols:
    _loaded_engines.append("CostBenefitEngine")
if _engine_7_symbols:
    _loaded_engines.append("GrantFinderEngine")
if _engine_8_symbols:
    _loaded_engines.append("CertificationReadinessEngine")

__all__: list[str] = (
    _engine_1_symbols
    + _engine_2_symbols
    + _engine_3_symbols
    + _engine_4_symbols
    + _engine_5_symbols
    + _engine_6_symbols
    + _engine_7_symbols
    + _engine_8_symbols
)


def get_loaded_engines() -> list[str]:
    """Return names of successfully loaded engines."""
    return list(_loaded_engines)


def get_engine_count() -> int:
    """Return total number of expected engines."""
    return __engines_count__


def get_loaded_engine_count() -> int:
    """Return number of successfully loaded engines."""
    return len(_loaded_engines)


logger.info(
    "PACK-026 engines: %d / %d loaded",
    get_loaded_engine_count(),
    get_engine_count(),
)
