# -*- coding: utf-8 -*-
"""
PACK-003 CSRD Enterprise - Computation Engines

Ten specialized engines providing the computational backbone for
enterprise-tier CSRD reporting with multi-tenant SaaS capabilities:

    1.  MultiTenantEngine          - Tenant lifecycle & quota management
    2.  WhiteLabelEngine           - Brand customization & WCAG themes
    3.  PredictiveAnalyticsEngine  - Statistical forecasting & anomalies
    4.  NarrativeGenerationEngine  - AI-assisted narrative composition
    5.  WorkflowBuilderEngine      - Custom DAG workflow orchestration
    6.  IoTStreamingEngine         - Real-time sensor data integration
    7.  CarbonCreditEngine         - Carbon offset portfolio management
    8.  SupplyChainESGEngine       - Multi-tier supplier ESG scoring
    9.  FilingAutomationEngine     - Automated regulatory submission
    10. APIManagementEngine        - Enterprise API governance

Zero-Hallucination Guarantee:
    All numeric computations in every engine use deterministic
    mathematical and statistical formulas. LLM assistance is
    restricted to narrative prose generation with mandatory
    fact-checking against source data.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-003 CSRD Enterprise
Status: Production Ready
"""

from __future__ import annotations

# Engine 1: Multi-Tenant Lifecycle Management
from packs.eu_compliance.PACK_003_csrd_enterprise.engines.multi_tenant_engine import (
    IsolationLevel,
    MultiTenantEngine,
    QuotaViolation,
    ResourceUsage,
    TenantLifecycleStatus,
    TenantProvisionRequest,
    TenantResource,
    TenantStatus,
    TenantTier,
)

# Engine 2: White-Label Brand Customization
from packs.eu_compliance.PACK_003_csrd_enterprise.engines.white_label_engine import (
    BrandConfig,
    BrandTheme,
    BrandValidationIssue,
    BrandValidationSeverity,
    ColorVariant,
    SSLStatus,
    TemplateType,
    WhiteLabelEngine,
)

# Engine 3: Predictive Analytics & Forecasting
from packs.eu_compliance.PACK_003_csrd_enterprise.engines.predictive_analytics_engine import (
    AnomalyMethod,
    AnomalyPoint,
    AnomalyResult,
    AnomalySeverity,
    ForecastRequest,
    ForecastResult,
    HistoricalDataPoint,
    ModelType,
    PredictionPoint,
    PredictiveAnalyticsEngine,
)

# Engine 4: Narrative Generation with Fact-Checking
from packs.eu_compliance.PACK_003_csrd_enterprise.engines.narrative_generation_engine import (
    Citation,
    ESRSSection,
    FactCheckResult,
    FactCheckStatus,
    NarrativeGenerationEngine,
    NarrativeRequest,
    NarrativeResult,
    NarrativeTone,
    RevisionDiff,
)

# Engine 5: Workflow Builder & Execution
from packs.eu_compliance.PACK_003_csrd_enterprise.engines.workflow_builder_engine import (
    ConditionOperator,
    ExecutionStatus,
    StepResult,
    StepStatus,
    StepType,
    ValidationIssue,
    WorkflowBuilderEngine,
    WorkflowCondition,
    WorkflowDefinition,
    WorkflowExecution,
    WorkflowStatus,
    WorkflowStep,
)

# Engine 6: IoT Streaming & Real-Time Data
from packs.eu_compliance.PACK_003_csrd_enterprise.engines.iot_streaming_engine import (
    AggregatedReading,
    AlertSeverity,
    AlertType,
    DeviceLocation,
    DeviceProtocol,
    DeviceStatus,
    DeviceType,
    IoTDevice,
    IoTReading,
    IoTStreamingEngine,
    QualityFlag,
    StreamAlert,
)

# Engine 7: Carbon Credit Portfolio Management
from packs.eu_compliance.PACK_003_csrd_enterprise.engines.carbon_credit_engine import (
    CarbonCredit,
    CarbonCreditEngine,
    CreditPortfolio,
    CreditRegistry,
    CreditStatus,
    CreditType,
    NetZeroAccounting,
    RetirementReason,
)

# Engine 8: Supply Chain ESG Scoring
from packs.eu_compliance.PACK_003_csrd_enterprise.engines.supply_chain_esg_engine import (
    Action,
    ESGScore,
    Finding,
    ImprovementPlan,
    ImprovementStatus,
    QuestionnaireStatus,
    RiskTier,
    Supplier,
    SupplierQuestionnaire,
    SupplierTier,
    SupplyChainESGEngine,
)

# Engine 9: Filing Automation
from packs.eu_compliance.PACK_003_csrd_enterprise.engines.filing_automation_engine import (
    AuthMethod,
    DeadlineUrgency,
    FilingAutomationEngine,
    FilingDeadline,
    FilingFormat,
    FilingPackage,
    FilingStatus,
    FilingSubmission,
    FilingTarget,
    FilingTargetName,
    ValidationFinding,
    ValidationSeverity,
)

# Engine 10: API Management & Governance
from packs.eu_compliance.PACK_003_csrd_enterprise.engines.api_management_engine import (
    APIKey,
    APIManagementEngine,
    APIUsageMetrics,
    EndpointUsage,
    GraphQLConfig,
    KeyStatus,
    RateLimitAlgorithm,
    RateLimitPolicy,
    WebhookEvent,
    WebhookRegistration,
)

__all__: list[str] = [
    # Engine 1: Multi-Tenant
    "MultiTenantEngine",
    "TenantProvisionRequest",
    "TenantResource",
    "TenantStatus",
    "TenantTier",
    "TenantLifecycleStatus",
    "IsolationLevel",
    "ResourceUsage",
    "QuotaViolation",
    # Engine 2: White-Label
    "WhiteLabelEngine",
    "BrandConfig",
    "BrandTheme",
    "ColorVariant",
    "BrandValidationIssue",
    "BrandValidationSeverity",
    "TemplateType",
    "SSLStatus",
    # Engine 3: Predictive Analytics
    "PredictiveAnalyticsEngine",
    "ForecastRequest",
    "ForecastResult",
    "HistoricalDataPoint",
    "PredictionPoint",
    "AnomalyResult",
    "AnomalyPoint",
    "ModelType",
    "AnomalyMethod",
    "AnomalySeverity",
    # Engine 4: Narrative Generation
    "NarrativeGenerationEngine",
    "NarrativeRequest",
    "NarrativeResult",
    "NarrativeTone",
    "Citation",
    "FactCheckResult",
    "FactCheckStatus",
    "ESRSSection",
    "RevisionDiff",
    # Engine 5: Workflow Builder
    "WorkflowBuilderEngine",
    "WorkflowStep",
    "WorkflowCondition",
    "WorkflowDefinition",
    "WorkflowExecution",
    "StepResult",
    "StepType",
    "StepStatus",
    "ConditionOperator",
    "WorkflowStatus",
    "ExecutionStatus",
    "ValidationIssue",
    # Engine 6: IoT Streaming
    "IoTStreamingEngine",
    "IoTDevice",
    "IoTReading",
    "AggregatedReading",
    "StreamAlert",
    "DeviceType",
    "DeviceProtocol",
    "DeviceStatus",
    "DeviceLocation",
    "QualityFlag",
    "AlertType",
    "AlertSeverity",
    # Engine 7: Carbon Credits
    "CarbonCreditEngine",
    "CarbonCredit",
    "CreditPortfolio",
    "NetZeroAccounting",
    "CreditRegistry",
    "CreditType",
    "CreditStatus",
    "RetirementReason",
    # Engine 8: Supply Chain ESG
    "SupplyChainESGEngine",
    "Supplier",
    "ESGScore",
    "SupplierQuestionnaire",
    "ImprovementPlan",
    "Finding",
    "Action",
    "SupplierTier",
    "RiskTier",
    "QuestionnaireStatus",
    "ImprovementStatus",
    # Engine 9: Filing Automation
    "FilingAutomationEngine",
    "FilingTarget",
    "FilingPackage",
    "FilingSubmission",
    "FilingDeadline",
    "FilingTargetName",
    "FilingFormat",
    "FilingStatus",
    "ValidationFinding",
    "ValidationSeverity",
    "AuthMethod",
    "DeadlineUrgency",
    # Engine 10: API Management
    "APIManagementEngine",
    "APIKey",
    "RateLimitPolicy",
    "APIUsageMetrics",
    "EndpointUsage",
    "WebhookRegistration",
    "GraphQLConfig",
    "KeyStatus",
    "RateLimitAlgorithm",
    "WebhookEvent",
]
