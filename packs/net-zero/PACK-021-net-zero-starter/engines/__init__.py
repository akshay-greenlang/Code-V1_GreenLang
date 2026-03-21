# -*- coding: utf-8 -*-
"""
PACK-021 Net Zero Starter Pack - Engines Module
==================================================

Deterministic, zero-hallucination calculation engines for the Net Zero
Starter Pack.  Each engine covers a specific aspect of the net-zero
journey -- from GHG baseline establishment through target setting,
gap analysis, reduction planning, residual emissions, offset management,
maturity scoring, and peer benchmarking.

Every engine produces bit-perfect reproducible results with SHA-256
provenance hashing.  No LLM is used in any scoring, classification,
or calculation path.

Engines:
    1. NetZeroBaselineEngine       - Unified GHG baseline (Scopes 1, 2, 3)
    2. NetZeroTargetEngine         - SBTi Net-Zero Standard target setting
    3. NetZeroGapEngine            - Gap-to-net-zero trajectory analysis
    4. ReductionPathwayEngine      - Quantified reduction pathway with MACC
    5. ResidualEmissionsEngine     - Residual emissions & neutralization
    6. OffsetPortfolioEngine       - Carbon credit portfolio management
    7. NetZeroScorecardEngine      - Net-zero readiness & maturity scorecard
    8. NetZeroBenchmarkEngine      - Peer benchmarking against sector

Regulatory / Framework Basis:
    SBTi Corporate Net-Zero Standard v1.2 (2024)
    GHG Protocol Corporate Standard (2004, revised 2015)
    GHG Protocol Scope 3 Standard (2011)
    IPCC AR6 WG1 (2021) -- GWP-100 values
    Paris Agreement (2015) -- 1.5 C temperature target
    ISO 14064-1:2018 -- Organization-level GHG quantification
    VCMI Claims Code (2023) -- Carbon credit quality
    ISO 14068-1:2023 -- Carbon neutrality quantification
    Oxford Principles for Net Zero Aligned Carbon Offsetting (2020)
    SBTi FLAG Guidance (2022)

Pack Tier: Starter (PACK-021)
Category: Net Zero Packs
Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-021"
__pack_name__: str = "Net Zero Starter Pack"
__engines_count__: int = 8

# ─── Engine 1: Net Zero Baseline ──────────────────────────────────────
_engine_1_symbols: list[str] = []
try:
    from .net_zero_baseline_engine import (  # noqa: F401
        BoundaryMethod,
        DataQualityScore,
        FuelType,
        GasType,
        NetZeroBaselineEngine,
        Scope,
        Scope3Category,
        Scope1SourceType,
        BaselineInput,
        BaselineResult,
        FuelConsumptionEntry,
        RefrigerantEntry,
        ElectricityEntry,
        Scope3SpendEntry,
        Scope1Detail,
        Scope2Detail,
        Scope3Detail,
        EmissionsByGas,
        IntensityMetrics,
        BaseYearAssessment,
        DataQualityAssessment,
    )
    _engine_1_symbols = [
        "BoundaryMethod", "DataQualityScore", "FuelType", "GasType",
        "NetZeroBaselineEngine", "Scope", "Scope3Category", "Scope1SourceType",
        "BaselineInput", "BaselineResult", "FuelConsumptionEntry",
        "RefrigerantEntry", "ElectricityEntry", "Scope3SpendEntry",
        "Scope1Detail", "Scope2Detail", "Scope3Detail",
        "EmissionsByGas", "IntensityMetrics", "BaseYearAssessment",
        "DataQualityAssessment",
    ]
    logger.debug("Engine 1 (NetZeroBaselineEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 1 (NetZeroBaselineEngine) not available: %s", exc)

# ─── Engine 2: Net Zero Target ────────────────────────────────────────
_engine_2_symbols: list[str] = []
try:
    from .net_zero_target_engine import (  # noqa: F401
        AmbitionLevel,
        PathwayType,
        SBTiSector,
        ScopeCategory,
        TargetTimeframe,
        TargetType,
        TemperatureAlignment,
        NetZeroTargetEngine,
        TargetInput,
        TargetResult,
        TargetDefinition,
        MilestoneEntry,
        ValidationCheck,
    )
    _engine_2_symbols = [
        "AmbitionLevel", "PathwayType", "SBTiSector", "ScopeCategory",
        "TargetTimeframe", "TargetType", "TemperatureAlignment",
        "NetZeroTargetEngine", "TargetInput", "TargetResult",
        "TargetDefinition", "MilestoneEntry", "ValidationCheck",
    ]
    logger.debug("Engine 2 (NetZeroTargetEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 2 (NetZeroTargetEngine) not available: %s", exc)

# ─── Engine 3: Net Zero Gap ───────────────────────────────────────────
_engine_3_symbols: list[str] = []
try:
    from .net_zero_gap_engine import (  # noqa: F401
        GapSeverity,
        ProjectionMethod,
        TrajectoryStatus,
        NetZeroGapEngine,
        GapInput,
        GapResult,
        YearlyGap,
        ScopeGap,
        ScopeEmissions,
        HistoricalDataPoint,
        BudgetAnalysis,
        RiskAssessment,
    )
    _engine_3_symbols = [
        "GapSeverity", "ProjectionMethod", "TrajectoryStatus",
        "NetZeroGapEngine", "GapInput", "GapResult",
        "YearlyGap", "ScopeGap", "ScopeEmissions", "HistoricalDataPoint",
        "BudgetAnalysis", "RiskAssessment",
    ]
    logger.debug("Engine 3 (NetZeroGapEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 3 (NetZeroGapEngine) not available: %s", exc)

# ─── Engine 4: Reduction Pathway ──────────────────────────────────────
_engine_4_symbols: list[str] = []
try:
    from .reduction_pathway_engine import (  # noqa: F401
        AbatementCategory,
        ImplementationPhase,
        TechnologyReadiness,
        TimeHorizon,
        ReductionPathwayEngine,
        PathwayInput,
        PathwayResult,
        AbatementOption,
        MACCPoint,
        PhasedAction,
    )
    _engine_4_symbols = [
        "AbatementCategory", "ImplementationPhase", "TechnologyReadiness",
        "TimeHorizon", "ReductionPathwayEngine", "PathwayInput",
        "PathwayResult", "AbatementOption", "MACCPoint",
        "PhasedAction",
    ]
    logger.debug("Engine 4 (ReductionPathwayEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 4 (ReductionPathwayEngine) not available: %s", exc)

# ─── Engine 5: Residual Emissions ─────────────────────────────────────
_engine_5_symbols: list[str] = []
try:
    from .residual_emissions_engine import (  # noqa: F401
        CDRType,
        PermanenceCategory,
        ResidualAllowanceLevel,
        CDRReadinessLevel,
        ResidualEmissionsEngine,
        ResidualInput,
        ResidualResult,
        CDROptionAssessment,
        NeutralizationTimeline,
    )
    _engine_5_symbols = [
        "CDRType", "PermanenceCategory", "ResidualAllowanceLevel",
        "CDRReadinessLevel", "ResidualEmissionsEngine", "ResidualInput",
        "ResidualResult", "CDROptionAssessment", "NeutralizationTimeline",
    ]
    logger.debug("Engine 5 (ResidualEmissionsEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 5 (ResidualEmissionsEngine) not available: %s", exc)

# ─── Engine 6: Offset Portfolio ───────────────────────────────────────
_engine_6_symbols: list[str] = []
try:
    from .offset_portfolio_engine import (  # noqa: F401
        CreditStandard,
        CreditType,
        CreditCategory,
        QualityDimension,
        RetirementStatus,
        VCMIClaim,
        SBTiCreditUse,
        OffsetPortfolioEngine,
        PortfolioResult,
        CreditEntry,
        CreditQualityScore,
        PortfolioSummary,
        SBTiComplianceResult,
        VCMIAlignmentResult,
        OxfordAlignmentResult,
    )
    _engine_6_symbols = [
        "CreditStandard", "CreditType", "CreditCategory",
        "QualityDimension", "RetirementStatus", "VCMIClaim",
        "SBTiCreditUse", "OffsetPortfolioEngine", "PortfolioResult",
        "CreditEntry", "CreditQualityScore", "PortfolioSummary",
        "SBTiComplianceResult", "VCMIAlignmentResult",
        "OxfordAlignmentResult",
    ]
    logger.debug("Engine 6 (OffsetPortfolioEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 6 (OffsetPortfolioEngine) not available: %s", exc)

# ─── Engine 7: Net Zero Scorecard ─────────────────────────────────────
_engine_7_symbols: list[str] = []
try:
    from .net_zero_scorecard_engine import (  # noqa: F401
        MaturityLevel,
        RecommendationPriority,
        ScorecardDimension,
        NetZeroScorecardEngine,
        ScorecardInput,
        ScorecardResult,
        DimensionScore,
        DimensionInput,
        ScorecardRecommendation,
    )
    _engine_7_symbols = [
        "MaturityLevel", "RecommendationPriority", "ScorecardDimension",
        "NetZeroScorecardEngine", "ScorecardInput", "ScorecardResult",
        "DimensionScore", "DimensionInput", "ScorecardRecommendation",
    ]
    logger.debug("Engine 7 (NetZeroScorecardEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 7 (NetZeroScorecardEngine) not available: %s", exc)

# ─── Engine 8: Net Zero Benchmark ─────────────────────────────────────
_engine_8_symbols: list[str] = []
try:
    from .net_zero_benchmark_engine import (  # noqa: F401
        BenchmarkSector,
        PerformanceIndicator,
        PerformanceTrend,
        Percentile,
        SBTiStatus,
        NetZeroBenchmarkEngine,
        BenchmarkInput,
        BenchmarkResult,
        PeerComparison,
        KPIBenchmark,
    )
    _engine_8_symbols = [
        "BenchmarkSector", "PerformanceIndicator", "PerformanceTrend",
        "Percentile", "SBTiStatus", "NetZeroBenchmarkEngine",
        "BenchmarkInput", "BenchmarkResult", "PeerComparison",
        "KPIBenchmark",
    ]
    logger.debug("Engine 8 (NetZeroBenchmarkEngine) loaded successfully")
except ImportError as exc:
    logger.debug("Engine 8 (NetZeroBenchmarkEngine) not available: %s", exc)

# ─── Dynamic __all__ ──────────────────────────────────────────────────

_loaded_engines: list[str] = []
if _engine_1_symbols:
    _loaded_engines.append("NetZeroBaselineEngine")
if _engine_2_symbols:
    _loaded_engines.append("NetZeroTargetEngine")
if _engine_3_symbols:
    _loaded_engines.append("NetZeroGapEngine")
if _engine_4_symbols:
    _loaded_engines.append("ReductionPathwayEngine")
if _engine_5_symbols:
    _loaded_engines.append("ResidualEmissionsEngine")
if _engine_6_symbols:
    _loaded_engines.append("OffsetPortfolioEngine")
if _engine_7_symbols:
    _loaded_engines.append("NetZeroScorecardEngine")
if _engine_8_symbols:
    _loaded_engines.append("NetZeroBenchmarkEngine")

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
    "PACK-021 engines: %d / %d loaded",
    get_loaded_engine_count(),
    get_engine_count(),
)
