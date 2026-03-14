# -*- coding: utf-8 -*-
"""
PACK-003 CSRD Enterprise Pack - Integration Layer
====================================================

Phase 4 integration layer for the CSRD Enterprise Pack that extends
PACK-002 Professional with enterprise-grade integrations: multi-tenant
workflow orchestration, tenant management, SSO (SAML/OAuth/OIDC/SCIM),
GraphQL API extensions, ML model lifecycle, auditor portal, plugin
marketplace, 10-step enterprise setup wizard, and 15-category health check.

Components:
    - EnterprisePackOrchestrator: Multi-tenant workflow orchestration with
      batch execution, scheduling, SLA enforcement, and checkpoint/resume
    - TenantBridge: Wraps platform TenantManager with CSRD-specific metadata,
      feature flags, tier migration, and cross-tenant benchmarking
    - SSOBridge: Unified SAML/OAuth/OIDC/SCIM interface with JIT provisioning,
      group-to-role mapping, and session management
    - GraphQLBridge: Extends Strawberry schema with CSRD types, tenant-scoped
      queries, field-level auth, complexity limits, and query analytics
    - MLBridge: Model lifecycle management with training, prediction,
      drift detection, anomaly detection, and SHAP/LIME explainability
    - AuditorBridge: Audit engagement management with evidence packaging
      per ISAE 3000/3410, finding management, and opinion issuance
    - MarketplaceBridge: Plugin discovery, installation, compatibility
      checking, usage tracking, and quota enforcement
    - EnterpriseSetupWizard: 10-step guided setup for enterprise deployments
    - EnterpriseHealthCheck: 15-category health verification system

Architecture:
    External Systems --> Integration Bridges --> Platform Components
                              |
                              v
    CSRD Enterprise Pack <-- Composition (not inheritance) <-- Zero duplication
                              |
                              v
    Tenants <-- Data Isolation <-- SSO <-- Auditors

Platform Integrations:
    - greenlang/auth/tenant.py (TenantManager)
    - greenlang/auth/saml_provider.py (SAMLProvider)
    - greenlang/auth/oauth_provider.py (OAuthProvider)
    - greenlang/auth/scim_provider.py (SCIMProvider)
    - greenlang/execution/infrastructure/api/graphql_schema.py
    - greenlang/extensions/ml/predictive/
    - greenlang/extensions/ml/drift_detection/
    - greenlang/extensions/ml/explainability/
    - greenlang/infrastructure/soc2_preparation/auditor_portal/
    - greenlang/ecosystem/marketplace/

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-003 CSRD Enterprise
Status: Production Ready
"""

# ---------------------------------------------------------------------------
# Enterprise Pack Orchestrator
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_003_csrd_enterprise.integrations.pack_orchestrator import (
    ApprovalChainConfig,
    EnterpriseOrchestratorConfig,
    EnterprisePackOrchestrator,
    EnterpriseQualityGateConfig,
    EnterpriseRetryConfig,
    EnterpriseWorkflowPhase,
    EnterpriseWorkflowType,
    ExecutionStatus,
    QualityGateId,
    QualityGateResult,
    QualityGateStatus,
    ScheduledWorkflow,
    SLAConfig,
    WorkflowCheckpoint,
    WorkflowResult,
)

# ---------------------------------------------------------------------------
# Tenant Bridge
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_003_csrd_enterprise.integrations.tenant_bridge import (
    CSRDTenantConfig,
    ResourceUsage,
    TenantBridge,
    TenantIsolationLevel,
    TenantProfile,
    TenantStatus,
    TenantTier,
)

# ---------------------------------------------------------------------------
# SSO Bridge
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_003_csrd_enterprise.integrations.sso_bridge import (
    AuthResult,
    AuthStatus,
    OAuthConfig,
    SAMLConfig,
    SCIMConfig,
    SSOBridge,
    SSOProtocol,
    SyncAction,
    SyncResult,
    UserProfile,
)

# ---------------------------------------------------------------------------
# GraphQL Bridge
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_003_csrd_enterprise.integrations.graphql_bridge import (
    ComplianceStatusType,
    EmissionReportType,
    FieldAuthRule,
    GraphQLBridge,
    QueryComplexityConfig,
    QueryLogEntry,
    SupplierScoreType,
    TenantDashboardType,
)

# ---------------------------------------------------------------------------
# ML Bridge
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_003_csrd_enterprise.integrations.ml_bridge import (
    AnomalyMethod,
    AnomalyResult,
    DriftResult,
    DriftSeverity,
    ExplainabilityResult,
    MLBridge,
    ModelRegistration,
    ModelStatus,
    ModelType,
    PredictionResult,
    TrainingResult,
)

# ---------------------------------------------------------------------------
# Auditor Bridge
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_003_csrd_enterprise.integrations.auditor_bridge import (
    AssuranceLevel,
    AuditEngagement,
    AuditFinding,
    AuditOpinion,
    AuditorAccess,
    AuditorBridge,
    AuditorPermission,
    EngagementProgress,
    EngagementStatus,
    EvidencePackage,
    FindingSeverity,
    FindingStatus,
    OpinionType,
)

# ---------------------------------------------------------------------------
# Marketplace Bridge
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_003_csrd_enterprise.integrations.marketplace_bridge import (
    CompatibilityLevel,
    CompatibilityResult,
    InstallResult,
    InstalledPlugin,
    MarketplaceBridge,
    PluginCategory,
    PluginInfo,
    PluginStatus,
    PluginUsageMetrics,
)

# ---------------------------------------------------------------------------
# Enterprise Setup Wizard
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_003_csrd_enterprise.integrations.setup_wizard import (
    APIKeyConfig,
    DataResidencyConfig,
    EnterpriseSetupWizard,
    EnterpriseWizardStep,
    EntityConfig,
    FrameworkConfig,
    IoTDeviceConfig,
    OrganizationProfile,
    SetupResult,
    SSOSetup,
    StepStatus,
    TierSelection,
    WhiteLabelConfig,
    WizardState,
    WizardStepState,
)

# ---------------------------------------------------------------------------
# Enterprise Health Check
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_003_csrd_enterprise.integrations.health_check import (
    CheckCategory,
    ComponentHealth,
    EnterpriseHealthCheck,
    HealthCheckConfig,
    HealthCheckResult,
    HealthSeverity,
    HealthStatus,
    RemediationSuggestion,
)

__all__ = [
    # --- Enterprise Pack Orchestrator ---
    "EnterprisePackOrchestrator",
    "EnterpriseOrchestratorConfig",
    "EnterpriseRetryConfig",
    "SLAConfig",
    "EnterpriseQualityGateConfig",
    "ApprovalChainConfig",
    "EnterpriseWorkflowType",
    "EnterpriseWorkflowPhase",
    "ExecutionStatus",
    "QualityGateId",
    "QualityGateStatus",
    "QualityGateResult",
    "WorkflowCheckpoint",
    "ScheduledWorkflow",
    "WorkflowResult",
    # --- Tenant Bridge ---
    "TenantBridge",
    "TenantProfile",
    "TenantTier",
    "TenantIsolationLevel",
    "TenantStatus",
    "CSRDTenantConfig",
    "ResourceUsage",
    # --- SSO Bridge ---
    "SSOBridge",
    "SSOProtocol",
    "AuthStatus",
    "SyncAction",
    "SAMLConfig",
    "OAuthConfig",
    "SCIMConfig",
    "UserProfile",
    "AuthResult",
    "SyncResult",
    # --- GraphQL Bridge ---
    "GraphQLBridge",
    "EmissionReportType",
    "ComplianceStatusType",
    "TenantDashboardType",
    "SupplierScoreType",
    "FieldAuthRule",
    "QueryComplexityConfig",
    "QueryLogEntry",
    # --- ML Bridge ---
    "MLBridge",
    "ModelType",
    "ModelStatus",
    "DriftSeverity",
    "AnomalyMethod",
    "ModelRegistration",
    "TrainingResult",
    "PredictionResult",
    "DriftResult",
    "AnomalyResult",
    "ExplainabilityResult",
    # --- Auditor Bridge ---
    "AuditorBridge",
    "AssuranceLevel",
    "EngagementStatus",
    "FindingSeverity",
    "FindingStatus",
    "OpinionType",
    "AuditorPermission",
    "AuditEngagement",
    "AuditorAccess",
    "EvidencePackage",
    "AuditFinding",
    "AuditOpinion",
    "EngagementProgress",
    # --- Marketplace Bridge ---
    "MarketplaceBridge",
    "PluginCategory",
    "PluginStatus",
    "CompatibilityLevel",
    "PluginInfo",
    "InstallResult",
    "InstalledPlugin",
    "PluginUsageMetrics",
    "CompatibilityResult",
    # --- Enterprise Setup Wizard ---
    "EnterpriseSetupWizard",
    "EnterpriseWizardStep",
    "StepStatus",
    "OrganizationProfile",
    "TierSelection",
    "SSOSetup",
    "WhiteLabelConfig",
    "DataResidencyConfig",
    "EntityConfig",
    "FrameworkConfig",
    "IoTDeviceConfig",
    "APIKeyConfig",
    "WizardStepState",
    "WizardState",
    "SetupResult",
    # --- Enterprise Health Check ---
    "EnterpriseHealthCheck",
    "HealthCheckConfig",
    "HealthCheckResult",
    "ComponentHealth",
    "HealthSeverity",
    "HealthStatus",
    "CheckCategory",
    "RemediationSuggestion",
]
