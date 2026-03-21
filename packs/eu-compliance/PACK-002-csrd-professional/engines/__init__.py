# -*- coding: utf-8 -*-
"""
PACK-002 CSRD Professional - Engines
======================================

Seven specialized engines providing the computational backbone for
professional-tier CSRD reporting:

    1. ConsolidationEngine     - Multi-entity ESRS consolidation
    2. ApprovalWorkflowEngine  - 4-level approval chain
    3. QualityGateEngine       - 3-gate quality assurance
    4. BenchmarkingEngine      - Peer comparison & ESG rating alignment
    5. StakeholderEngine       - Stakeholder engagement management
    6. RegulatoryImpactEngine  - Regulatory change impact analysis
    7. DataGovernanceEngine    - Data lifecycle, classification & GDPR

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-002 CSRD Professional
Status: Production Ready
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-002"
__pack_name__: str = "CSRD Professional Pack"
__engines_count__: int = 7

_loaded_engines: list[str] = []

# ===================================================================
# Engine 1: Multi-Entity Consolidation
# ===================================================================
_ENGINE_1_SYMBOLS: list[str] = [
    "ConsolidationEngine",
    "ConsolidationConfig",
    "ConsolidationApproach",
    "ConsolidationMethod",
    "ConsolidationResult",
    "EntityDefinition",
    "EntityESRSData",
    "IntercompanyTransaction",
    "ReconciliationEntry",
    "TransactionType",
]

try:
    from .consolidation_engine import (  # noqa: F401
        ConsolidationApproach,
        ConsolidationConfig,
        ConsolidationEngine,
        ConsolidationMethod,
        ConsolidationResult,
        EntityDefinition,
        EntityESRSData,
        IntercompanyTransaction,
        ReconciliationEntry,
        TransactionType,
    )
    _loaded_engines.append("ConsolidationEngine")
except ImportError as e:
    logger.debug("Engine 1 (ConsolidationEngine) not available: %s", e)
    _ENGINE_1_SYMBOLS = []

# ===================================================================
# Engine 2: Approval Workflow
# ===================================================================
_ENGINE_2_SYMBOLS: list[str] = [
    "ApprovalWorkflowEngine",
    "ApprovalConfig",
    "ApprovalLevel",
    "ApprovalStatus",
    "ApprovalLevelConfig",
    "ApprovalComment",
    "ApprovalDecision",
    "ApprovalRequest",
    "ApprovalChainResult",
    "DecisionType",
    "DelegationEntry",
]

try:
    from .approval_workflow_engine import (  # noqa: F401
        ApprovalChainResult,
        ApprovalComment,
        ApprovalConfig,
        ApprovalDecision,
        ApprovalLevel,
        ApprovalLevelConfig,
        ApprovalRequest,
        ApprovalStatus,
        ApprovalWorkflowEngine,
        DecisionType,
        DelegationEntry,
    )
    _loaded_engines.append("ApprovalWorkflowEngine")
except ImportError as e:
    logger.debug("Engine 2 (ApprovalWorkflowEngine) not available: %s", e)
    _ENGINE_2_SYMBOLS = []

# ===================================================================
# Engine 3: Quality Gates
# ===================================================================
_ENGINE_3_SYMBOLS: list[str] = [
    "QualityGateEngine",
    "QualityGateConfig",
    "QualityGateId",
    "GateCheckDefinition",
    "GateCheckResult",
    "QualityGateResult",
    "GateOverride",
    "RemediationPriority",
    "RemediationSuggestion",
]

try:
    from .quality_gate_engine import (  # noqa: F401
        GateCheckDefinition,
        GateCheckResult,
        GateOverride,
        QualityGateConfig,
        QualityGateEngine,
        QualityGateId,
        QualityGateResult,
        RemediationPriority,
        RemediationSuggestion,
    )
    _loaded_engines.append("QualityGateEngine")
except ImportError as e:
    logger.debug("Engine 3 (QualityGateEngine) not available: %s", e)
    _ENGINE_3_SYMBOLS = []

# ===================================================================
# Engine 4: Benchmarking
# ===================================================================
_ENGINE_4_SYMBOLS: list[str] = [
    "BenchmarkingEngine",
    "BenchmarkingConfig",
    "BenchmarkDataset",
    "PeerComparison",
    "ESGRatingPrediction",
    "TrendAnalysis",
    "BenchmarkReport",
]

try:
    from .benchmarking_engine import (  # noqa: F401
        BenchmarkDataset,
        BenchmarkingConfig,
        BenchmarkingEngine,
        BenchmarkReport,
        ESGRatingPrediction,
        PeerComparison,
        TrendAnalysis,
    )
    _loaded_engines.append("BenchmarkingEngine")
except ImportError as e:
    logger.debug("Engine 4 (BenchmarkingEngine) not available: %s", e)
    _ENGINE_4_SYMBOLS = []

# ===================================================================
# Engine 5: Stakeholder Engagement
# ===================================================================
_ENGINE_5_SYMBOLS: list[str] = [
    "StakeholderEngine",
    "StakeholderConfig",
    "StakeholderCategory",
    "SalienceCategory",
    "EngagementType",
    "Stakeholder",
    "EngagementActivity",
    "MaterialityInput",
    "SalienceMap",
    "EngagementReport",
]

try:
    from .stakeholder_engine import (  # noqa: F401
        EngagementActivity,
        EngagementReport,
        EngagementType,
        MaterialityInput,
        SalienceCategory,
        SalienceMap,
        Stakeholder,
        StakeholderCategory,
        StakeholderConfig,
        StakeholderEngine,
    )
    _loaded_engines.append("StakeholderEngine")
except ImportError as e:
    logger.debug("Engine 5 (StakeholderEngine) not available: %s", e)
    _ENGINE_5_SYMBOLS = []

# ===================================================================
# Engine 6: Regulatory Impact
# ===================================================================
_ENGINE_6_SYMBOLS: list[str] = [
    "RegulatoryImpactEngine",
    "RegulatoryConfig",
    "RegulationSource",
    "ChangeSeverity",
    "DeadlineStatus",
    "GapStatus",
    "RegulatoryChange",
    "ImpactAssessment",
    "ComplianceGap",
    "RegulatoryDeadline",
    "RegulatoryCalendar",
]

try:
    from .regulatory_impact_engine import (  # noqa: F401
        ChangeSeverity,
        ComplianceGap,
        DeadlineStatus,
        GapStatus,
        ImpactAssessment,
        RegulationSource,
        RegulatoryCalendar,
        RegulatoryChange,
        RegulatoryConfig,
        RegulatoryDeadline,
        RegulatoryImpactEngine,
    )
    _loaded_engines.append("RegulatoryImpactEngine")
except ImportError as e:
    logger.debug("Engine 6 (RegulatoryImpactEngine) not available: %s", e)
    _ENGINE_6_SYMBOLS = []

# ===================================================================
# Engine 7: Data Governance
# ===================================================================
_ENGINE_7_SYMBOLS: list[str] = [
    "DataGovernanceEngine",
    "DataGovernanceConfig",
    "DataClassificationLevel",
    "DataSubjectRequestType",
    "RequestStatus",
    "DataClassification",
    "RetentionPolicy",
    "DataSubjectRequest",
    "GovernanceReport",
]

try:
    from .data_governance_engine import (  # noqa: F401
        DataClassification,
        DataClassificationLevel,
        DataGovernanceConfig,
        DataGovernanceEngine,
        DataSubjectRequest,
        DataSubjectRequestType,
        GovernanceReport,
        RequestStatus,
        RetentionPolicy,
    )
    _loaded_engines.append("DataGovernanceEngine")
except ImportError as e:
    logger.debug("Engine 7 (DataGovernanceEngine) not available: %s", e)
    _ENGINE_7_SYMBOLS = []

# ===================================================================
# Module exports
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
    """Return list of engine class names that loaded successfully."""
    return list(_loaded_engines)


def get_engine_count() -> int:
    """Return count of engines that loaded successfully."""
    return len(_loaded_engines)


logger.info(
    "PACK-002 engines: %d/%d loaded",
    len(_loaded_engines),
    __engines_count__,
)
