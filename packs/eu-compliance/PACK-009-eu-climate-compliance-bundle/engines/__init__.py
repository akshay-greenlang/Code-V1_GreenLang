# -*- coding: utf-8 -*-
"""
PACK-009 EU Climate Compliance Bundle - Computation Engines

Eight specialized engines providing the computational backbone for
cross-regulation EU climate compliance management spanning CSRD, CBAM,
EU Taxonomy, and EUDR:

    1. CrossFrameworkDataMapperEngine  - Bidirectional field mapping across
                                         4 EU regulations (~100 mappings)
    2. DataDeduplicationEngine         - Duplicate detection and golden
                                         record generation (~200 fields)
    3. RegulatoryCalendarEngine        - Deadline orchestration across all
                                         4 regulatory calendars
    4. CrossRegulationGapAnalyzerEngine - Gap analysis spanning all 4
                                         regulation requirement sets
    5. ConsolidatedMetricsEngine       - Aggregated KPIs from all 4
                                         constituent packs (~60 metrics)
    6. MultiRegulationConsistencyEngine - Shared data point validation
                                          across packs (~60 fields)
    7. BundleComplianceScoringEngine   - Weighted compliance scoring with
                                          maturity assessment (5 levels)
    8. CrossRegulationEvidenceEngine   - Unified evidence repository with
                                          reuse tracking (~80 requirements)

Regulatory Basis:
    EU Directive 2022/2464 (CSRD)
    EU Regulation 2023/956 (CBAM)
    EU Regulation 2020/852 (Taxonomy)
    EU Regulation 2023/1115 (EUDR)

Pack Tier: Enterprise (PACK-009)
Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-009"
__pack_name__: str = "EU Climate Compliance Bundle"
__engines_count__: int = 8

_loaded_engines: list[str] = []

# ===================================================================
# Engine 1: Cross-Framework Data Mapper
# ===================================================================
_ENGINE_1_SYMBOLS: list[str] = [
    "CrossFrameworkDataMapperEngine",
    "CrossFrameworkDataMapperConfig",
    "FieldMapping",
    "MappingResult",
    "BatchMappingResult",
    "OverlapStatistics",
    "MappingPath",
    "MappingType",
    "Regulation",
    "MappingCategory",
]
try:
    from .cross_framework_data_mapper import (
        CrossFrameworkDataMapperEngine,
        CrossFrameworkDataMapperConfig,
        FieldMapping,
        MappingResult,
        BatchMappingResult,
        OverlapStatistics,
        MappingPath,
        MappingType,
        Regulation,
        MappingCategory,
    )
    _loaded_engines.append("CrossFrameworkDataMapperEngine")
except ImportError as e:
    logger.debug("Engine 1 (CrossFrameworkDataMapperEngine) not available: %s", e)
    _ENGINE_1_SYMBOLS = []

# ===================================================================
# Engine 2: Data Deduplication
# ===================================================================
_ENGINE_2_SYMBOLS: list[str] = [
    "DataDeduplicationEngine",
    "DataDeduplicationConfig",
    "DataRequirement",
    "DeduplicationGroup",
    "DeduplicationResult",
    "GoldenRecord",
    "DedupReport",
    "MergeConflict",
    "MergeStrategy",
    "MatchType",
    "ConflictSeverity",
]
try:
    from .data_deduplication_engine import (
        DataDeduplicationEngine,
        DataDeduplicationConfig,
        DataRequirement,
        DeduplicationGroup,
        DeduplicationResult,
        GoldenRecord,
        DedupReport,
        MergeConflict,
        MergeStrategy,
        MatchType,
        ConflictSeverity,
    )
    _loaded_engines.append("DataDeduplicationEngine")
except ImportError as e:
    logger.debug("Engine 2 (DataDeduplicationEngine) not available: %s", e)
    _ENGINE_2_SYMBOLS = []

# ===================================================================
# Engine 3: Regulatory Calendar
# ===================================================================
_ENGINE_3_SYMBOLS: list[str] = [
    "RegulatoryCalendarEngine",
    "CalendarConfig",
    "CalendarResult",
    "CalendarEvent",
    "CalendarEventType",
    "EventStatus",
    "AlertLevel",
    "DeadlineConflict",
    "DependencyChain",
    "CalendarAlert",
    "ICalExport",
    "REGULATORY_DEADLINES",
]
try:
    from .regulatory_calendar_engine import (
        RegulatoryCalendarEngine,
        CalendarConfig,
        CalendarResult,
        CalendarEvent,
        CalendarEventType,
        EventStatus,
        AlertLevel,
        DeadlineConflict,
        DependencyChain,
        CalendarAlert,
        ICalExport,
        REGULATORY_DEADLINES,
    )
    _loaded_engines.append("RegulatoryCalendarEngine")
except ImportError as e:
    logger.debug("Engine 3 (RegulatoryCalendarEngine) not available: %s", e)
    _ENGINE_3_SYMBOLS = []

# ===================================================================
# Engine 4: Cross-Regulation Gap Analyzer
# ===================================================================
_ENGINE_4_SYMBOLS: list[str] = [
    "CrossRegulationGapAnalyzerEngine",
    "GapAnalyzerConfig",
    "GapAnalysisResult",
    "Gap",
    "ComplianceRequirement",
    "RemediationRoadmapItem",
    "CrossImpactEntry",
    "GapSeverity",
    "ComplianceStatus",
    "RequirementCategory",
    "RemediationPriority",
    "COMPLIANCE_REQUIREMENTS",
]
try:
    from .cross_regulation_gap_analyzer import (
        CrossRegulationGapAnalyzerEngine,
        GapAnalyzerConfig,
        GapAnalysisResult,
        Gap,
        ComplianceRequirement,
        RemediationRoadmapItem,
        CrossImpactEntry,
        GapSeverity,
        ComplianceStatus,
        RequirementCategory,
        RemediationPriority,
        COMPLIANCE_REQUIREMENTS,
    )
    _loaded_engines.append("CrossRegulationGapAnalyzerEngine")
except ImportError as e:
    logger.debug("Engine 4 (CrossRegulationGapAnalyzerEngine) not available: %s", e)
    _ENGINE_4_SYMBOLS = []

# ===================================================================
# Engine 5: Consolidated Metrics
# ===================================================================
_ENGINE_5_SYMBOLS: list[str] = [
    "ConsolidatedMetricsEngine",
    "ConsolidatedMetricsConfig",
    "ConsolidatedMetricsResult",
    "RegulationMetrics",
    "TrendDataPoint",
    "PeriodSnapshot",
    "PeriodComparison",
    "ExecutiveSummary",
    "TrendDirection",
    "SummaryRating",
    "REGULATION_METRIC_DEFINITIONS",
]
try:
    from .consolidated_metrics_engine import (
        ConsolidatedMetricsEngine,
        ConsolidatedMetricsConfig,
        ConsolidatedMetricsResult,
        RegulationMetrics,
        TrendDataPoint,
        PeriodSnapshot,
        PeriodComparison,
        ExecutiveSummary,
        TrendDirection,
        SummaryRating,
        REGULATION_METRIC_DEFINITIONS,
    )
    _loaded_engines.append("ConsolidatedMetricsEngine")
except ImportError as e:
    logger.debug("Engine 5 (ConsolidatedMetricsEngine) not available: %s", e)
    _ENGINE_5_SYMBOLS = []

# ===================================================================
# Engine 6: Multi-Regulation Consistency
# ===================================================================
_ENGINE_6_SYMBOLS: list[str] = [
    "MultiRegulationConsistencyEngine",
    "ConsistencyConfig",
    "ConsistencyResult",
    "ConsistencyCheck",
    "ConsistencyLevel",
    "DataPoint",
    "ConflictResolution",
    "ReconciliationItem",
    "ComparisonMode",
    "FieldType",
    "FieldCategory",
    "ResolutionStrategy",
    "SHARED_DATA_FIELDS",
]
try:
    from .multi_regulation_consistency_engine import (
        MultiRegulationConsistencyEngine,
        ConsistencyConfig,
        ConsistencyResult,
        ConsistencyCheck,
        ConsistencyLevel,
        DataPoint,
        ConflictResolution,
        ReconciliationItem,
        ComparisonMode,
        FieldType,
        FieldCategory,
        ResolutionStrategy,
        SHARED_DATA_FIELDS,
    )
    _loaded_engines.append("MultiRegulationConsistencyEngine")
except ImportError as e:
    logger.debug("Engine 6 (MultiRegulationConsistencyEngine) not available: %s", e)
    _ENGINE_6_SYMBOLS = []

# ===================================================================
# Engine 7: Bundle Compliance Scoring
# ===================================================================
_ENGINE_7_SYMBOLS: list[str] = [
    "BundleComplianceScoringEngine",
    "ScoringConfig",
    "BundleScoringResult",
    "RegulationInput",
    "ComplianceScore",
    "MaturityAssessment",
    "MaturityLevel",
    "HeatmapCell",
    "HeatmapStatus",
    "Recommendation",
    "BenchmarkComparison",
    "RiskSeverity",
    "DEFAULT_WEIGHTS",
    "INDUSTRY_BENCHMARKS",
    "MATURITY_DEFINITIONS",
]
try:
    from .bundle_compliance_scoring_engine import (
        BundleComplianceScoringEngine,
        ScoringConfig,
        BundleScoringResult,
        RegulationInput,
        ComplianceScore,
        MaturityAssessment,
        MaturityLevel,
        HeatmapCell,
        HeatmapStatus,
        Recommendation,
        BenchmarkComparison,
        RiskSeverity,
        DEFAULT_WEIGHTS,
        INDUSTRY_BENCHMARKS,
        MATURITY_DEFINITIONS,
    )
    _loaded_engines.append("BundleComplianceScoringEngine")
except ImportError as e:
    logger.debug("Engine 7 (BundleComplianceScoringEngine) not available: %s", e)
    _ENGINE_7_SYMBOLS = []

# ===================================================================
# Engine 8: Cross-Regulation Evidence
# ===================================================================
_ENGINE_8_SYMBOLS: list[str] = [
    "CrossRegulationEvidenceEngine",
    "EvidenceConfig",
    "EvidenceResult",
    "EvidenceItem",
    "EvidenceMapping",
    "EvidenceGap",
    "ExpiringEvidence",
    "CoverageMatrix",
    "ReuseSavings",
    "EvidenceType",
    "EvidenceStatus",
    "EVIDENCE_REQUIREMENTS",
    "EVIDENCE_REUSE_MAP",
]
try:
    from .cross_regulation_evidence_engine import (
        CrossRegulationEvidenceEngine,
        EvidenceConfig,
        EvidenceResult,
        EvidenceItem,
        EvidenceMapping,
        EvidenceGap,
        ExpiringEvidence,
        CoverageMatrix,
        ReuseSavings,
        EvidenceType,
        EvidenceStatus,
        EVIDENCE_REQUIREMENTS,
        EVIDENCE_REUSE_MAP,
    )
    _loaded_engines.append("CrossRegulationEvidenceEngine")
except ImportError as e:
    logger.debug("Engine 8 (CrossRegulationEvidenceEngine) not available: %s", e)
    _ENGINE_8_SYMBOLS = []

# ===================================================================
# Public API
# ===================================================================

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
]


def get_loaded_engines() -> list[str]:
    """Return list of successfully loaded engine class names."""
    return list(_loaded_engines)


def get_engine_count() -> int:
    """Return count of successfully loaded engines."""
    return len(_loaded_engines)


logger.info(
    "PACK-009 engines: %d/%d loaded",
    len(_loaded_engines),
    __engines_count__,
)
