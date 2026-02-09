# -*- coding: utf-8 -*-
"""
GL-DATA-SUP-001: GreenLang Supplier Questionnaire Processor Service SDK
========================================================================

This package provides supplier questionnaire creation, distribution,
response collection, validation, scoring, follow-up management,
analytics, and provenance tracking SDK for the GreenLang framework.
It supports:

- Multi-framework questionnaire templates (CDP, EcoVadis, GRI, Custom)
- Multi-channel distribution (email, portal, API, bulk upload)
- Supplier response collection with evidence attachment
- Multi-level validation (completeness, consistency, evidence, cross-field)
- Deterministic scoring with performance tier classification
- Automated follow-up and escalation management
- Campaign analytics with compliance gap identification
- Supplier benchmarking against peer cohorts
- SHA-256 provenance chain tracking for complete audit trails
- 12 Prometheus metrics for observability
- FastAPI REST API with 20 endpoints
- Thread-safe configuration with GL_SUPPLIER_QUEST_ env prefix

Key Components:
    - config: SupplierQuestionnaireConfig with GL_SUPPLIER_QUEST_ env prefix
    - template_builder: Questionnaire template creation engine
    - distribution_engine: Multi-channel questionnaire distribution engine
    - response_collector: Supplier response collection engine
    - validation_engine: Multi-level response validation engine
    - scoring_engine: Deterministic response scoring engine
    - followup_manager: Reminder and escalation management engine
    - analytics_engine: Campaign analytics and reporting engine
    - provenance: SHA-256 chain-hashed audit trails
    - metrics: 12 Prometheus metrics
    - api: FastAPI HTTP service
    - setup: SupplierQuestionnaireService facade

Example:
    >>> from greenlang.supplier_questionnaire import SupplierQuestionnaireService
    >>> service = SupplierQuestionnaireService()
    >>> template = service.create_template(
    ...     name="CDP Climate 2025", framework="cdp",
    ... )
    >>> print(template.status)
    draft

Agent ID: GL-DATA-SUP-001
Agent Name: Supplier Questionnaire Processor Agent
"""

__version__ = "1.0.0"
__agent_id__ = "GL-DATA-SUP-001"
__agent_name__ = "Supplier Questionnaire Processor Agent"

# SDK availability flag
SUPPLIER_QUESTIONNAIRE_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from greenlang.supplier_questionnaire.config import (
    SupplierQuestionnaireConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
from greenlang.supplier_questionnaire.provenance import ProvenanceTracker

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from greenlang.supplier_questionnaire.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    supplier_quest_templates_total,
    supplier_quest_distributions_total,
    supplier_quest_responses_total,
    supplier_quest_validations_total,
    supplier_quest_scores_total,
    supplier_quest_followups_total,
    supplier_quest_response_rate,
    supplier_quest_processing_duration_seconds,
    supplier_quest_active_campaigns,
    supplier_quest_pending_responses,
    supplier_quest_processing_errors_total,
    supplier_quest_data_quality_score,
    # Helper functions
    record_template,
    record_distribution,
    record_response,
    record_validation,
    record_score,
    record_followup,
    update_response_rate,
    record_processing_duration,
    update_active_campaigns,
    update_pending_responses,
    record_processing_error,
    record_data_quality,
)

# ---------------------------------------------------------------------------
# Core engines (Layer 2 SDK)
# ---------------------------------------------------------------------------
from greenlang.supplier_questionnaire.questionnaire_template import (
    QuestionnaireTemplateEngine,
)
from greenlang.supplier_questionnaire.distribution import DistributionEngine
from greenlang.supplier_questionnaire.response_collector import (
    ResponseCollectorEngine,
)
from greenlang.supplier_questionnaire.validation_engine import (
    ValidationEngine as QuestionnaireValidationEngine,
)
from greenlang.supplier_questionnaire.scoring_engine import ScoringEngine
from greenlang.supplier_questionnaire.follow_up import FollowUpEngine
from greenlang.supplier_questionnaire.analytics import AnalyticsEngine

# ---------------------------------------------------------------------------
# Models (Layer 2 SDK)
# ---------------------------------------------------------------------------
from greenlang.supplier_questionnaire.models import (
    # Enumerations
    Framework,
    QuestionType,
    QuestionnaireStatus,
    DistributionStatus,
    DistributionChannel,
    ResponseStatus,
    ValidationSeverity,
    ReminderType,
    EscalationLevel,
    CDPGrade,
    PerformanceTier,
    ReportFormat,
    # Core models
    TemplateQuestion,
    TemplateSection,
    Answer,
    ValidationCheck,
    ValidationSummary,
    QuestionnaireScore,
    # Request models
    CreateTemplateRequest,
    DistributeRequest,
    SubmitResponseRequest,
)

# ---------------------------------------------------------------------------
# Service setup facade and models
# ---------------------------------------------------------------------------
from greenlang.supplier_questionnaire.setup import (
    SupplierQuestionnaireService,
    configure_supplier_questionnaire,
    get_supplier_questionnaire,
    get_router,
    # Models
    QuestionnaireTemplate,
    Distribution,
    QuestionnaireResponse,
    ValidationResult,
    ScoringResult,
    FollowUpAction,
    CampaignAnalytics,
    QuestionnaireStatistics,
)

__all__ = [
    # Version
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # SDK flag
    "SUPPLIER_QUESTIONNAIRE_SDK_AVAILABLE",
    # Configuration
    "SupplierQuestionnaireConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Provenance
    "ProvenanceTracker",
    # Metric objects
    "PROMETHEUS_AVAILABLE",
    "supplier_quest_templates_total",
    "supplier_quest_distributions_total",
    "supplier_quest_responses_total",
    "supplier_quest_validations_total",
    "supplier_quest_scores_total",
    "supplier_quest_followups_total",
    "supplier_quest_response_rate",
    "supplier_quest_processing_duration_seconds",
    "supplier_quest_active_campaigns",
    "supplier_quest_pending_responses",
    "supplier_quest_processing_errors_total",
    "supplier_quest_data_quality_score",
    # Metric helper functions
    "record_template",
    "record_distribution",
    "record_response",
    "record_validation",
    "record_score",
    "record_followup",
    "update_response_rate",
    "record_processing_duration",
    "update_active_campaigns",
    "update_pending_responses",
    "record_processing_error",
    "record_data_quality",
    # Core engines (Layer 2)
    "QuestionnaireTemplateEngine",
    "DistributionEngine",
    "ResponseCollectorEngine",
    "QuestionnaireValidationEngine",
    "ScoringEngine",
    "FollowUpEngine",
    "AnalyticsEngine",
    # Layer 2 Enumerations
    "Framework",
    "QuestionType",
    "QuestionnaireStatus",
    "DistributionStatus",
    "DistributionChannel",
    "ResponseStatus",
    "ValidationSeverity",
    "ReminderType",
    "EscalationLevel",
    "CDPGrade",
    "PerformanceTier",
    "ReportFormat",
    # Layer 2 Core models
    "TemplateQuestion",
    "TemplateSection",
    "Answer",
    "ValidationCheck",
    "ValidationSummary",
    "QuestionnaireScore",
    # Layer 2 Request models
    "CreateTemplateRequest",
    "DistributeRequest",
    "SubmitResponseRequest",
    # Service setup facade
    "SupplierQuestionnaireService",
    "configure_supplier_questionnaire",
    "get_supplier_questionnaire",
    "get_router",
    # Models
    "QuestionnaireTemplate",
    "Distribution",
    "QuestionnaireResponse",
    "ValidationResult",
    "ScoringResult",
    "FollowUpAction",
    "CampaignAnalytics",
    "QuestionnaireStatistics",
]
