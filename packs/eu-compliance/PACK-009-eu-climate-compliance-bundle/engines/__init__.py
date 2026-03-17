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
    3. UnifiedTimelineEngine           - Deadline orchestration across all
                                         4 regulatory calendars (planned)
    4. CrossRegulationGapEngine        - Gap analysis spanning all 4
                                         regulation requirement sets (planned)
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

__version__: str = "1.0.0"
__pack__: str = "PACK-009"
__pack_name__: str = "EU Climate Compliance Bundle"
__engines_count__: int = 8

# ===================================================================
# Engine 1: Cross-Framework Data Mapper
# ===================================================================
from packs.eu_compliance.PACK_009_eu_climate_compliance_bundle.engines.cross_framework_data_mapper import (
    BatchMappingResult,
    CrossFrameworkDataMapperConfig,
    CrossFrameworkDataMapperEngine,
    FieldMapping,
    MappingCategory,
    MappingPath,
    MappingResult,
    MappingType,
    OverlapStatistics,
    Regulation,
)

# ===================================================================
# Engine 2: Data Deduplication
# ===================================================================
from packs.eu_compliance.PACK_009_eu_climate_compliance_bundle.engines.data_deduplication_engine import (
    ConflictSeverity,
    DataDeduplicationConfig,
    DataDeduplicationEngine,
    DataRequirement,
    DedupReport,
    DeduplicationGroup,
    DeduplicationResult,
    GoldenRecord,
    MatchType,
    MergeConflict,
    MergeStrategy,
)

# ===================================================================
# Engine 5: Consolidated Metrics
# ===================================================================
from packs.eu_compliance.PACK_009_eu_climate_compliance_bundle.engines.consolidated_metrics_engine import (
    ConsolidatedMetricsConfig,
    ConsolidatedMetricsEngine,
    ConsolidatedMetricsResult,
    ExecutiveSummary,
    PeriodComparison,
    PeriodSnapshot,
    REGULATION_METRIC_DEFINITIONS,
    RegulationMetrics,
    SummaryRating,
    TrendDataPoint,
    TrendDirection,
)

# ===================================================================
# Engine 6: Multi-Regulation Consistency
# ===================================================================
from packs.eu_compliance.PACK_009_eu_climate_compliance_bundle.engines.multi_regulation_consistency_engine import (
    ComparisonMode,
    ConflictResolution,
    ConsistencyCheck,
    ConsistencyConfig,
    ConsistencyLevel,
    ConsistencyResult,
    DataPoint,
    FieldCategory,
    FieldType,
    MultiRegulationConsistencyEngine,
    ReconciliationItem,
    ResolutionStrategy,
    SHARED_DATA_FIELDS,
)

# ===================================================================
# Engine 7: Bundle Compliance Scoring
# ===================================================================
from packs.eu_compliance.PACK_009_eu_climate_compliance_bundle.engines.bundle_compliance_scoring_engine import (
    BenchmarkComparison,
    BundleComplianceScoringEngine,
    BundleScoringResult,
    ComplianceScore,
    DEFAULT_WEIGHTS,
    HeatmapCell,
    HeatmapStatus,
    INDUSTRY_BENCHMARKS,
    MATURITY_DEFINITIONS,
    MaturityAssessment,
    MaturityLevel,
    Recommendation,
    RegulationInput,
    RiskSeverity,
    ScoringConfig,
)

# ===================================================================
# Engine 8: Cross-Regulation Evidence
# ===================================================================
from packs.eu_compliance.PACK_009_eu_climate_compliance_bundle.engines.cross_regulation_evidence_engine import (
    CoverageMatrix,
    CrossRegulationEvidenceEngine,
    EvidenceConfig,
    EvidenceGap,
    EvidenceItem,
    EvidenceMapping,
    EVIDENCE_REQUIREMENTS,
    EVIDENCE_REUSE_MAP,
    EvidenceResult,
    EvidenceStatus,
    EvidenceType,
    ExpiringEvidence,
    ReuseSavings,
)

# ===================================================================
# Public API
# ===================================================================

__all__: list[str] = [
    # Pack metadata
    "__version__",
    "__pack__",
    "__pack_name__",
    "__engines_count__",
    # Engine 1: Cross-Framework Data Mapper
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
    # Engine 2: Data Deduplication
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
    # Engine 5: Consolidated Metrics
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
    # Engine 6: Multi-Regulation Consistency
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
    # Engine 7: Bundle Compliance Scoring
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
    # Engine 8: Cross-Regulation Evidence
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
