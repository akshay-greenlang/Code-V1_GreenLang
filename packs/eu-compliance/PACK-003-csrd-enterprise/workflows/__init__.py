# -*- coding: utf-8 -*-
"""
PACK-003 CSRD Enterprise Pack - Workflow Orchestration
=========================================================

Enterprise-grade workflow orchestrators for CSRD compliance operations at scale.
Each workflow coordinates GreenLang agents, data pipelines, AI engines, and
validation systems into end-to-end compliance processes for multi-tenant,
multi-entity enterprise deployments with 200+ entities.

Workflows:
    - EnterpriseReportingWorkflow: 10-phase annual CSRD reporting cycle with
      checkpoint/resume, multi-entity group dispatch, AI quality assessment,
      narrative generation, cross-framework alignment, and automated filing.
      Performance target: <60 minutes for 200+ entities.

    - MultiTenantOnboardingWorkflow: 6-phase tenant provisioning with SSO/SAML
      configuration, white-label branding, data residency compliance, feature
      activation, and rollback support if any phase fails.

    - PredictiveComplianceWorkflow: 5-phase AI-driven compliance forecasting
      with trend modeling, gap prediction using confidence intervals, Monte Carlo
      risk scoring (CRITICAL/HIGH/MEDIUM/LOW), and intervention action planning
      with estimated ROI.

    - RealTimeMonitoringWorkflow: 4-phase continuous IoT monitoring with device
      registration, stream processing, real-time anomaly detection, and
      multi-channel alert dispatch (webhook/email/Slack/Teams) with escalation.

    - CustomWorkflowExecutionWorkflow: 3-phase user-defined workflow runner
      with DAG-based step execution (topological sort), conditional branching,
      parallel fork/join, timer steps, and human-in-the-loop approval gates.

    - AuditorCollaborationWorkflow: 5-phase external audit engagement with
      evidence packaging per ISAE 3000/3410, iterative review cycles with
      comment threads, finding management, and assurance opinion issuance.

    - RegulatoryFilingWorkflow: 6-phase automated filing with ESEF/iXBRL
      package generation, pre-submission validation, internal approval routing,
      multi-registry submission (ESAP/national/EDGAR), acknowledgment tracking,
      and post-filing archival with provenance chain.

    - SupplyChainAssessmentWorkflow: 5-phase supply chain ESG assessment with
      multi-tier (1-4) supplier mapping, automated questionnaire dispatch,
      E/S/G scoring (0-100), risk tier assignment, corrective action plans,
      and Scope 3 upstream emission estimation.

Author: GreenLang Team
Version: 3.0.0
"""

from packs.eu_compliance.PACK_003_csrd_enterprise.workflows.enterprise_reporting import (
    EnterpriseReportingWorkflow,
    EnterpriseReportConfig,
    EnterpriseReportResult,
    GroupReportResult,
    CheckpointData,
    EntityConfig,
)
from packs.eu_compliance.PACK_003_csrd_enterprise.workflows.multi_tenant_onboarding import (
    MultiTenantOnboardingWorkflow,
    TenantRequest,
    OnboardingResult,
    SSOMetadata,
    BrandingConfig,
)
from packs.eu_compliance.PACK_003_csrd_enterprise.workflows.predictive_compliance import (
    PredictiveComplianceWorkflow,
    PredictiveComplianceInput,
    PredictiveComplianceResult,
    ComplianceTarget,
    MetricProjection,
    GapPrediction,
    InterventionAction,
)
from packs.eu_compliance.PACK_003_csrd_enterprise.workflows.real_time_monitoring import (
    RealTimeMonitoringWorkflow,
    MonitoringConfig,
    MonitoringSession,
    MonitoringSummary,
    DeviceConfig,
    AlertRule,
)
from packs.eu_compliance.PACK_003_csrd_enterprise.workflows.custom_workflow_execution import (
    CustomWorkflowExecutionWorkflow,
    WorkflowDefinition,
    StepDefinition,
    CustomWorkflowResult,
    ExecutionTrace,
)
from packs.eu_compliance.PACK_003_csrd_enterprise.workflows.auditor_collaboration import (
    AuditorCollaborationWorkflow,
    AuditEngagement,
    AuditCollaborationResult,
    AuditorProfile,
    AuditFinding,
    EvidenceItem,
)
from packs.eu_compliance.PACK_003_csrd_enterprise.workflows.regulatory_filing import (
    RegulatoryFilingWorkflow,
    FilingInput,
    FilingResult,
    FilingTargetConfig,
    SubmissionRecord,
)
from packs.eu_compliance.PACK_003_csrd_enterprise.workflows.supply_chain_assessment import (
    SupplyChainAssessmentWorkflow,
    SupplyChainAssessmentConfig,
    SupplyChainAssessmentResult,
    SupplierProfile,
    ESGScore,
    CorrectiveAction,
)

__all__ = [
    # Enterprise Reporting
    "EnterpriseReportingWorkflow",
    "EnterpriseReportConfig",
    "EnterpriseReportResult",
    "GroupReportResult",
    "CheckpointData",
    "EntityConfig",
    # Multi-Tenant Onboarding
    "MultiTenantOnboardingWorkflow",
    "TenantRequest",
    "OnboardingResult",
    "SSOMetadata",
    "BrandingConfig",
    # Predictive Compliance
    "PredictiveComplianceWorkflow",
    "PredictiveComplianceInput",
    "PredictiveComplianceResult",
    "ComplianceTarget",
    "MetricProjection",
    "GapPrediction",
    "InterventionAction",
    # Real-Time Monitoring
    "RealTimeMonitoringWorkflow",
    "MonitoringConfig",
    "MonitoringSession",
    "MonitoringSummary",
    "DeviceConfig",
    "AlertRule",
    # Custom Workflow Execution
    "CustomWorkflowExecutionWorkflow",
    "WorkflowDefinition",
    "StepDefinition",
    "CustomWorkflowResult",
    "ExecutionTrace",
    # Auditor Collaboration
    "AuditorCollaborationWorkflow",
    "AuditEngagement",
    "AuditCollaborationResult",
    "AuditorProfile",
    "AuditFinding",
    "EvidenceItem",
    # Regulatory Filing
    "RegulatoryFilingWorkflow",
    "FilingInput",
    "FilingResult",
    "FilingTargetConfig",
    "SubmissionRecord",
    # Supply Chain Assessment
    "SupplyChainAssessmentWorkflow",
    "SupplyChainAssessmentConfig",
    "SupplyChainAssessmentResult",
    "SupplierProfile",
    "ESGScore",
    "CorrectiveAction",
]
