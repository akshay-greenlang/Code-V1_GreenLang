# -*- coding: utf-8 -*-
"""
PACK-006 EUDR Starter - Computation Engines

Seven specialized engines providing the computational backbone for
EUDR (EU Deforestation Regulation) starter compliance:

    1. DDSAssemblyEngine               - Due Diligence Statement composition
                                          per EUDR Annex II (~20 fields)
    2. GeolocationEngine               - WGS84 coordinate and polygon validation
                                          per Article 9 requirements
    3. RiskScoringEngine               - Multi-source weighted risk aggregation
                                          per Articles 10-11
    4. CommodityClassificationEngine   - CN code mapping and Annex I commodity
                                          coverage (~7 commodity groups)
    5. SupplierComplianceEngine        - Supplier DD status tracking and
                                          completeness scoring
    6. CutoffDateEngine                - December 31, 2020 cutoff date
                                          verification per Article 3
    7. PolicyComplianceEngine          - 45 compliance rules across 7
                                          regulatory categories

Regulatory Basis:
    EU Regulation 2023/1115 (EUDR)

Pack Tier: Starter (PACK-006)
Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-006"
__pack_name__: str = "EUDR Starter Pack"
__engines_count__: int = 7

_loaded_engines: list[str] = []

# ===================================================================
# Engine 1: DDS Assembly
# ===================================================================
_ENGINE_1_SYMBOLS: list[str] = [
    "DDSAssemblyEngine",
    "DDSType",
    "DDSStatus",
    "RiskConclusion",
    "EvidenceType",
    "GeolocationFormat",
    "SupplierInfo",
    "ProductInfo",
    "GeolocationPoint",
    "GeolocationPolygon",
    "GeolocationInfo",
    "RiskSummary",
    "MitigationSummary",
    "OperatorDeclaration",
    "DDSEvidence",
    "FormattedGeolocation",
    "AnnexIIField",
    "AnnexIIValidation",
    "DDSDocument",
    "EUISSubmission",
    "FinalizedDDS",
]
try:
    from .dds_assembly_engine import (
        DDSAssemblyEngine,
        DDSType,
        DDSStatus,
        RiskConclusion,
        EvidenceType,
        GeolocationFormat,
        SupplierInfo,
        ProductInfo,
        GeolocationPoint,
        GeolocationPolygon,
        GeolocationInfo,
        RiskSummary,
        MitigationSummary,
        OperatorDeclaration,
        DDSEvidence,
        FormattedGeolocation,
        AnnexIIField,
        AnnexIIValidation,
        DDSDocument,
        EUISSubmission,
        FinalizedDDS,
    )
    _loaded_engines.append("DDSAssemblyEngine")
except ImportError as e:
    logger.debug("Engine 1 (DDSAssemblyEngine) not available: %s", e)
    _ENGINE_1_SYMBOLS = []

# ===================================================================
# Engine 2: Geolocation
# ===================================================================
_ENGINE_2_SYMBOLS: list[str] = [
    "GeolocationEngine",
    "CoordinateFormat",
    "ValidationSeverity",
    "PlotSizeCategory",
    "ValidationIssue",
    "CoordinateValidation",
    "PolygonValidation",
    "NormalizedCoordinate",
    "AreaResult",
    "OverlapResult",
    "CountryResult",
    "PlotSizeRule",
    "BatchValidationResult",
    "ParsedGeoJSON",
    "Article9Geolocation",
]
try:
    from .geolocation_engine import (
        GeolocationEngine,
        CoordinateFormat,
        ValidationSeverity,
        PlotSizeCategory,
        ValidationIssue,
        CoordinateValidation,
        PolygonValidation,
        NormalizedCoordinate,
        AreaResult,
        OverlapResult,
        CountryResult,
        PlotSizeRule,
        BatchValidationResult,
        ParsedGeoJSON,
        Article9Geolocation,
    )
    _loaded_engines.append("GeolocationEngine")
except ImportError as e:
    logger.debug("Engine 2 (GeolocationEngine) not available: %s", e)
    _ENGINE_2_SYMBOLS = []

# ===================================================================
# Engine 3: Risk Scoring
# ===================================================================
_ENGINE_3_SYMBOLS: list[str] = [
    "RiskScoringEngine",
    "RiskLevel",
    "Article29Benchmark",
    "EUDRCommodity",
    "RiskFactorCategory",
    "TrendDirection",
    "RiskFactor",
    "CountryRiskScore",
    "SupplierRiskScore",
    "CommodityRiskScore",
    "DocumentRiskScore",
    "CompositeRiskScore",
    "CountryBenchmark",
    "SimplifiedDDEligibility",
    "RiskTrendPoint",
    "RiskTrend",
]
try:
    from .risk_scoring_engine import (
        RiskScoringEngine,
        RiskLevel,
        Article29Benchmark,
        EUDRCommodity,
        RiskFactorCategory,
        TrendDirection,
        RiskFactor,
        CountryRiskScore,
        SupplierRiskScore,
        CommodityRiskScore,
        DocumentRiskScore,
        CompositeRiskScore,
        CountryBenchmark,
        SimplifiedDDEligibility,
        RiskTrendPoint,
        RiskTrend,
    )
    _loaded_engines.append("RiskScoringEngine")
except ImportError as e:
    logger.debug("Engine 3 (RiskScoringEngine) not available: %s", e)
    _ENGINE_3_SYMBOLS = []

# ===================================================================
# Engine 4: Commodity Classification
# ===================================================================
_ENGINE_4_SYMBOLS: list[str] = [
    "CommodityClassificationEngine",
    "EUDRCommodity",
    "ProductType",
    "CommodityClassification",
    "CNCode",
    "CNCodeValidation",
    "DerivedProduct",
    "CNCodeMatch",
]
try:
    from .commodity_classification_engine import (
        CommodityClassificationEngine,
        EUDRCommodity as CommodityEUDRCommodity,
        ProductType,
        CommodityClassification,
        CNCode,
        CNCodeValidation,
        DerivedProduct,
        CNCodeMatch,
    )
    _loaded_engines.append("CommodityClassificationEngine")
except ImportError as e:
    logger.debug("Engine 4 (CommodityClassificationEngine) not available: %s", e)
    _ENGINE_4_SYMBOLS = []

# ===================================================================
# Engine 5: Supplier Compliance
# ===================================================================
_ENGINE_5_SYMBOLS: list[str] = [
    "SupplierComplianceEngine",
    "SupplierDDStatus",
    "SupplierTier",
    "EngagementType",
    "PriorityLevel",
    "CertificationStatus",
    "SupplierProfile",
    "DDStatusUpdate",
    "CompletenessScore",
    "CertificationRecord",
    "CertValidation",
    "PrioritizedSupplier",
    "DataRequest",
    "EngagementRecord",
    "ComplianceCalendarEntry",
    "ComplianceCalendar",
    "SupplierDashboard",
]
try:
    from .supplier_compliance_engine import (
        SupplierComplianceEngine,
        SupplierDDStatus,
        SupplierTier,
        EngagementType,
        PriorityLevel,
        CertificationStatus,
        SupplierProfile,
        DDStatusUpdate,
        CompletenessScore,
        CertificationRecord,
        CertValidation,
        PrioritizedSupplier,
        DataRequest,
        EngagementRecord,
        ComplianceCalendarEntry,
        ComplianceCalendar,
        SupplierDashboard,
    )
    _loaded_engines.append("SupplierComplianceEngine")
except ImportError as e:
    logger.debug("Engine 5 (SupplierComplianceEngine) not available: %s", e)
    _ENGINE_5_SYMBOLS = []

# ===================================================================
# Engine 6: Cutoff Date
# ===================================================================
_ENGINE_6_SYMBOLS: list[str] = [
    "CutoffDateEngine",
    "CutoffComplianceStatus",
    "LandUseType",
    "EvidenceSource",
    "EvidenceStrength",
    "ExemptionReason",
    "TemporalEvidence",
    "CutoffVerification",
    "DeforestationFreeResult",
    "LandUseChange",
    "LandUseHistory",
    "CutoffDeclaration",
    "ExemptionResult",
    "BatchCutoffResult",
    "CutoffSummary",
    "CUTOFF_DATE",
    "CUTOFF_DATE_DATE",
]
try:
    from .cutoff_date_engine import (
        CutoffDateEngine,
        CutoffComplianceStatus,
        LandUseType,
        EvidenceSource,
        EvidenceStrength,
        ExemptionReason,
        TemporalEvidence,
        CutoffVerification,
        DeforestationFreeResult,
        LandUseChange,
        LandUseHistory,
        CutoffDeclaration,
        ExemptionResult,
        BatchCutoffResult,
        CutoffSummary,
        CUTOFF_DATE,
        CUTOFF_DATE_DATE,
    )
    _loaded_engines.append("CutoffDateEngine")
except ImportError as e:
    logger.debug("Engine 6 (CutoffDateEngine) not available: %s", e)
    _ENGINE_6_SYMBOLS = []

# ===================================================================
# Engine 7: Policy Compliance
# ===================================================================
_ENGINE_7_SYMBOLS: list[str] = [
    "PolicyComplianceEngine",
    "RuleSeverity",
    "RuleCategory",
    "RuleStatus",
    "RemediationPriority",
    "ComplianceRule",
    "RuleResult",
    "ComplianceResult",
    "SimplifiedDDCheck",
    "PenaltyExposure",
    "RemediationAction",
    "RemediationPlan",
    "ComplianceAuditEntry",
    "COMPLIANCE_RULES",
    "PENALTY_RANGES",
]
try:
    from .policy_compliance_engine import (
        PolicyComplianceEngine,
        RuleSeverity,
        RuleCategory,
        RuleStatus,
        RemediationPriority,
        ComplianceRule,
        RuleResult,
        ComplianceResult,
        SimplifiedDDCheck,
        PenaltyExposure,
        RemediationAction,
        RemediationPlan,
        ComplianceAuditEntry,
        COMPLIANCE_RULES,
        PENALTY_RANGES,
    )
    _loaded_engines.append("PolicyComplianceEngine")
except ImportError as e:
    logger.debug("Engine 7 (PolicyComplianceEngine) not available: %s", e)
    _ENGINE_7_SYMBOLS = []

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
]


def get_loaded_engines() -> list[str]:
    """Return list of successfully loaded engine class names."""
    return list(_loaded_engines)


def get_engine_count() -> int:
    """Return count of successfully loaded engines."""
    return len(_loaded_engines)


logger.info(
    "PACK-006 engines: %d/%d loaded",
    len(_loaded_engines),
    __engines_count__,
)
