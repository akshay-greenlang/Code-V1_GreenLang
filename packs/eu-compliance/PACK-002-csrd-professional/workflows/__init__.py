# -*- coding: utf-8 -*-
"""
PACK-002 CSRD Professional Pack - Workflow Orchestration
=========================================================

Pre-built workflow orchestrators for professional CSRD compliance operations.
Each workflow coordinates GreenLang agents, data pipelines, and validation
engines into end-to-end compliance processes for multi-entity, multi-framework
enterprise deployments.

Workflows:
    - ConsolidatedReportingWorkflow: Multi-entity annual reporting cycle
    - CrossFrameworkAlignmentWorkflow: 7-framework alignment mapping
    - ScenarioAnalysisWorkflow: Climate scenario analysis (ESRS E1 / TCFD)
    - ContinuousComplianceWorkflow: Real-time compliance monitoring
    - StakeholderEngagementWorkflow: Stakeholder engagement per ESRS 1
    - RegulatoryChangeMgmtWorkflow: Regulatory change management
    - BoardGovernanceWorkflow: Board governance pack generation (ESRS 2 GOV)
    - ProfessionalAuditWorkflow: Enhanced audit preparation with assurance levels

Author: GreenLang Team
Version: 2.0.0
"""

from packs.eu_compliance.PACK_002_csrd_professional.workflows.consolidated_reporting import (
    ConsolidatedReportingWorkflow,
    ConsolidatedReportingInput,
    ConsolidatedReportingResult,
)
from packs.eu_compliance.PACK_002_csrd_professional.workflows.cross_framework_alignment import (
    CrossFrameworkAlignmentWorkflow,
    CrossFrameworkInput,
    CrossFrameworkResult,
)
from packs.eu_compliance.PACK_002_csrd_professional.workflows.scenario_analysis import (
    ScenarioAnalysisWorkflow,
    ScenarioAnalysisInput,
    ScenarioAnalysisResult,
)
from packs.eu_compliance.PACK_002_csrd_professional.workflows.continuous_compliance import (
    ContinuousComplianceWorkflow,
    ContinuousComplianceInput,
    ComplianceMonitoringResult,
)
from packs.eu_compliance.PACK_002_csrd_professional.workflows.stakeholder_engagement import (
    StakeholderEngagementWorkflow,
    StakeholderEngagementInput,
    StakeholderEngagementResult,
)
from packs.eu_compliance.PACK_002_csrd_professional.workflows.regulatory_change_mgmt import (
    RegulatoryChangeMgmtWorkflow,
    RegulatoryChangeMgmtInput,
    RegulatoryChangeMgmtResult,
)
from packs.eu_compliance.PACK_002_csrd_professional.workflows.board_governance import (
    BoardGovernanceWorkflow,
    BoardGovernanceInput,
    BoardGovernanceResult,
)
from packs.eu_compliance.PACK_002_csrd_professional.workflows.professional_audit import (
    ProfessionalAuditWorkflow,
    ProfessionalAuditInput,
    ProfessionalAuditResult,
)

__all__ = [
    "ConsolidatedReportingWorkflow",
    "ConsolidatedReportingInput",
    "ConsolidatedReportingResult",
    "CrossFrameworkAlignmentWorkflow",
    "CrossFrameworkInput",
    "CrossFrameworkResult",
    "ScenarioAnalysisWorkflow",
    "ScenarioAnalysisInput",
    "ScenarioAnalysisResult",
    "ContinuousComplianceWorkflow",
    "ContinuousComplianceInput",
    "ComplianceMonitoringResult",
    "StakeholderEngagementWorkflow",
    "StakeholderEngagementInput",
    "StakeholderEngagementResult",
    "RegulatoryChangeMgmtWorkflow",
    "RegulatoryChangeMgmtInput",
    "RegulatoryChangeMgmtResult",
    "BoardGovernanceWorkflow",
    "BoardGovernanceInput",
    "BoardGovernanceResult",
    "ProfessionalAuditWorkflow",
    "ProfessionalAuditInput",
    "ProfessionalAuditResult",
]
