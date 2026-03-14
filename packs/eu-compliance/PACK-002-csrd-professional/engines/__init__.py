# -*- coding: utf-8 -*-
"""
PACK-002 CSRD Professional - Computation Engines

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

# Engine 1: Multi-Entity Consolidation
from packs.eu_compliance.PACK_002_csrd_professional.engines.consolidation_engine import (
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

# Engine 2: Approval Workflow
from packs.eu_compliance.PACK_002_csrd_professional.engines.approval_workflow_engine import (
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

# Engine 3: Quality Gates
from packs.eu_compliance.PACK_002_csrd_professional.engines.quality_gate_engine import (
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

# Engine 4: Benchmarking
from packs.eu_compliance.PACK_002_csrd_professional.engines.benchmarking_engine import (
    BenchmarkDataset,
    BenchmarkingConfig,
    BenchmarkingEngine,
    BenchmarkReport,
    ESGRatingPrediction,
    PeerComparison,
    TrendAnalysis,
)

# Engine 5: Stakeholder Engagement
from packs.eu_compliance.PACK_002_csrd_professional.engines.stakeholder_engine import (
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

# Engine 6: Regulatory Impact
from packs.eu_compliance.PACK_002_csrd_professional.engines.regulatory_impact_engine import (
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

# Engine 7: Data Governance
from packs.eu_compliance.PACK_002_csrd_professional.engines.data_governance_engine import (
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

__all__: list[str] = [
    # Engine 1: Consolidation
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
    # Engine 2: Approval Workflow
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
    # Engine 3: Quality Gates
    "QualityGateEngine",
    "QualityGateConfig",
    "QualityGateId",
    "GateCheckDefinition",
    "GateCheckResult",
    "QualityGateResult",
    "GateOverride",
    "RemediationPriority",
    "RemediationSuggestion",
    # Engine 4: Benchmarking
    "BenchmarkingEngine",
    "BenchmarkingConfig",
    "BenchmarkDataset",
    "PeerComparison",
    "ESGRatingPrediction",
    "TrendAnalysis",
    "BenchmarkReport",
    # Engine 5: Stakeholder
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
    # Engine 6: Regulatory Impact
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
    # Engine 7: Data Governance
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
