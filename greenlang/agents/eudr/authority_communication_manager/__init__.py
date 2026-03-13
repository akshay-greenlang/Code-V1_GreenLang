# -*- coding: utf-8 -*-
"""
AGENT-EUDR-040: Authority Communication Manager

EUDR Articles 15-19, 31 authority communication management providing
production-grade capabilities for information request handling, inspection
coordination, non-compliance processing, administrative appeal management,
secure document exchange, multi-channel notification routing, and
multi-language template rendering across all 27 EU member states.

The agent interfaces between operators/traders and competent authorities,
managing the complete regulatory communication lifecycle with encrypted
document handling, deadline tracking, and GDPR-compliant audit trails.

Core capabilities:
    1. RequestHandler           -- Process information requests from
       competent authorities per EUDR Article 17 with deadline tracking
       and response assembly from validated data sources
    2. InspectionCoordinator    -- Schedule and manage on-the-spot checks
       per EUDR Article 15 with state machine status tracking, findings
       documentation, and corrective action management
    3. NonComplianceManager     -- Handle violations and penalties per
       EUDR Article 16 with deterministic penalty calculation, severity
       classification, and corrective action tracking
    4. AppealProcessor          -- Manage administrative appeals per
       EUDR Article 19 with filing, deadline extensions, decision
       recording, and penalty suspension tracking
    5. DocumentExchange         -- Secure document sharing with AES-256
       encryption, integrity verification, GDPR-compliant storage, and
       right to erasure support
    6. NotificationRouter       -- Multi-channel notification delivery
       across email, API, portal, SMS, and webhook channels with retry
       logic and delivery confirmation
    7. TemplateEngine           -- Multi-language template rendering
       supporting all 24 official EU languages with placeholder
       substitution and language fallback

Foundational modules:
    - config.py       -- AuthorityCommunicationManagerConfig with
      GL_EUDR_ACM_ env var support (65+ settings)
    - models.py       -- Pydantic v2 data models with 13 enumerations,
      15+ core models, and health/status types
    - provenance.py   -- SHA-256 chain-hashed audit trail tracking
    - metrics.py      -- 45 Prometheus self-monitoring metrics (gl_eudr_acm_)

Agent ID: GL-EUDR-ACM-040
Module: greenlang.agents.eudr.authority_communication_manager
PRD: PRD-AGENT-EUDR-040
Regulation: EU 2023/1115 Articles 15, 16, 17, 19, 31

Example:
    >>> from greenlang.agents.eudr.authority_communication_manager import (
    ...     AuthorityCommunicationManagerConfig,
    ...     get_config,
    ...     CommunicationType,
    ...     LanguageCode,
    ... )
    >>> cfg = get_config()
    >>> print(cfg.deadline_urgent_hours)
    24

    >>> from greenlang.agents.eudr.authority_communication_manager import (
    ...     AuthorityCommunicationManagerService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> health = await service.health_check()
    >>> assert health["status"] in ("healthy", "degraded")

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-040 Authority Communication Manager (GL-EUDR-ACM-040)
Status: Production Ready
"""

from __future__ import annotations

__version__ = "1.0.0"
__agent_id__ = "GL-EUDR-ACM-040"

# ---------------------------------------------------------------------------
# Public API listing
# ---------------------------------------------------------------------------

__all__: list[str] = [
    # -- Metadata --
    "__version__",
    "__agent_id__",
    # -- Configuration --
    "AuthorityCommunicationManagerConfig",
    "get_config",
    "reset_config",
    # -- Enumerations (13) --
    "CommunicationType",
    "CommunicationStatus",
    "CommunicationPriority",
    "InformationRequestType",
    "InspectionType",
    "ViolationType",
    "ViolationSeverity",
    "AppealDecision",
    "DocumentType",
    "NotificationChannel",
    "RecipientType",
    "LanguageCode",
    "AuthorityType",
    # -- Core Models (15+) --
    "Communication",
    "InformationRequest",
    "Inspection",
    "NonCompliance",
    "Appeal",
    "Document",
    "Authority",
    "Notification",
    "Template",
    "CommunicationThread",
    "ResponseData",
    "DeadlineReminder",
    "ApprovalWorkflow",
    "HealthStatus",
    # -- Constants --
    "AGENT_ID",
    "AGENT_VERSION",
    "EU_MEMBER_STATES",
    "EU_LANGUAGES",
    # -- Provenance --
    "ProvenanceTracker",
    "GENESIS_HASH",
    # -- Metrics (selected) --
    "record_communication_created",
    "record_communication_sent",
    "record_communication_responded",
    "record_information_request_received",
    "record_inspection_scheduled",
    "record_non_compliance_issued",
    "record_appeal_filed",
    "record_document_exchanged",
    "record_notification_sent",
    "record_api_error",
    "set_pending_communications",
    "set_overdue_responses",
    "set_active_appeals",
    "set_pending_inspections",
    "set_open_non_compliance",
    # -- Engines (7) --
    "RequestHandler",
    "InspectionCoordinator",
    "NonComplianceManager",
    "AppealProcessor",
    "DocumentExchange",
    "NotificationRouter",
    "TemplateEngine",
    # -- Service Facade --
    "AuthorityCommunicationManagerService",
    "get_service",
    "reset_service",
    "lifespan",
]


# ---------------------------------------------------------------------------
# Lazy import machinery
# ---------------------------------------------------------------------------

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Configuration
    "AuthorityCommunicationManagerConfig": ("config", "AuthorityCommunicationManagerConfig"),
    "get_config": ("config", "get_config"),
    "reset_config": ("config", "reset_config"),
    # Config constants
    "EU_MEMBER_STATES": ("config", "EU_MEMBER_STATES"),
    "EU_LANGUAGES": ("config", "EU_LANGUAGES"),
    # Enumerations
    "CommunicationType": ("models", "CommunicationType"),
    "CommunicationStatus": ("models", "CommunicationStatus"),
    "CommunicationPriority": ("models", "CommunicationPriority"),
    "InformationRequestType": ("models", "InformationRequestType"),
    "InspectionType": ("models", "InspectionType"),
    "ViolationType": ("models", "ViolationType"),
    "ViolationSeverity": ("models", "ViolationSeverity"),
    "AppealDecision": ("models", "AppealDecision"),
    "DocumentType": ("models", "DocumentType"),
    "NotificationChannel": ("models", "NotificationChannel"),
    "RecipientType": ("models", "RecipientType"),
    "LanguageCode": ("models", "LanguageCode"),
    "AuthorityType": ("models", "AuthorityType"),
    # Core Models
    "Communication": ("models", "Communication"),
    "InformationRequest": ("models", "InformationRequest"),
    "Inspection": ("models", "Inspection"),
    "NonCompliance": ("models", "NonCompliance"),
    "Appeal": ("models", "Appeal"),
    "Document": ("models", "Document"),
    "Authority": ("models", "Authority"),
    "Notification": ("models", "Notification"),
    "Template": ("models", "Template"),
    "CommunicationThread": ("models", "CommunicationThread"),
    "ResponseData": ("models", "ResponseData"),
    "DeadlineReminder": ("models", "DeadlineReminder"),
    "ApprovalWorkflow": ("models", "ApprovalWorkflow"),
    "HealthStatus": ("models", "HealthStatus"),
    # Constants
    "AGENT_ID": ("models", "AGENT_ID"),
    "AGENT_VERSION": ("models", "AGENT_VERSION"),
    # Provenance
    "ProvenanceTracker": ("provenance", "ProvenanceTracker"),
    "GENESIS_HASH": ("provenance", "GENESIS_HASH"),
    # Metrics (counters)
    "record_communication_created": ("metrics", "record_communication_created"),
    "record_communication_sent": ("metrics", "record_communication_sent"),
    "record_communication_responded": ("metrics", "record_communication_responded"),
    "record_information_request_received": ("metrics", "record_information_request_received"),
    "record_inspection_scheduled": ("metrics", "record_inspection_scheduled"),
    "record_non_compliance_issued": ("metrics", "record_non_compliance_issued"),
    "record_appeal_filed": ("metrics", "record_appeal_filed"),
    "record_document_exchanged": ("metrics", "record_document_exchanged"),
    "record_notification_sent": ("metrics", "record_notification_sent"),
    "record_api_error": ("metrics", "record_api_error"),
    # Metrics (gauges)
    "set_pending_communications": ("metrics", "set_pending_communications"),
    "set_overdue_responses": ("metrics", "set_overdue_responses"),
    "set_active_appeals": ("metrics", "set_active_appeals"),
    "set_pending_inspections": ("metrics", "set_pending_inspections"),
    "set_open_non_compliance": ("metrics", "set_open_non_compliance"),
    # Engines
    "RequestHandler": ("request_handler", "RequestHandler"),
    "InspectionCoordinator": ("inspection_coordinator", "InspectionCoordinator"),
    "NonComplianceManager": ("non_compliance_manager", "NonComplianceManager"),
    "AppealProcessor": ("appeal_processor", "AppealProcessor"),
    "DocumentExchange": ("document_exchange", "DocumentExchange"),
    "NotificationRouter": ("notification_router", "NotificationRouter"),
    "TemplateEngine": ("template_engine", "TemplateEngine"),
    # Service Facade
    "AuthorityCommunicationManagerService": ("setup", "AuthorityCommunicationManagerService"),
    "get_service": ("setup", "get_service"),
    "reset_service": ("setup", "reset_service"),
    "lifespan": ("setup", "lifespan"),
}


def __getattr__(name: str) -> object:
    """Module-level __getattr__ for lazy imports.

    Enables ``from greenlang.agents.eudr.authority_communication_manager import X``
    without eagerly loading all submodules at package import time.

    Args:
        name: Attribute name to look up.

    Returns:
        The lazily imported object.

    Raises:
        AttributeError: If the name is not a known export.
    """
    if name in _LAZY_IMPORTS:
        module_suffix, attr_name = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(
            f"greenlang.agents.eudr.authority_communication_manager.{module_suffix}"
        )
        return getattr(mod, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def get_version() -> str:
    """Return the current module version string.

    Returns:
        Version string in semver format (e.g. "1.0.0").

    Example:
        >>> get_version()
        '1.0.0'
    """
    return __version__


def get_agent_info() -> dict:
    """Return agent identification and capability metadata.

    Returns:
        Dictionary with agent_id, version, regulation references,
        engine listing, and capability summary.

    Example:
        >>> info = get_agent_info()
        >>> info["agent_id"]
        'GL-EUDR-ACM-040'
        >>> info["engine_count"]
        7
    """
    return {
        "agent_id": __agent_id__,
        "version": __version__,
        "name": "Authority Communication Manager",
        "prd": "PRD-AGENT-EUDR-040",
        "regulation": "EU 2023/1115 (EUDR)",
        "articles": ["15", "16", "17", "19", "31"],
        "enforcement_date_large": "2025-12-30",
        "enforcement_date_sme": "2026-06-30",
        "member_states": 27,
        "eu_languages": 24,
        "communication_types": [
            "information_request", "inspection_notice",
            "non_compliance_notice", "penalty_notice",
            "appeal_acknowledgment", "appeal_decision",
            "compliance_confirmation", "corrective_action_order",
            "market_withdrawal_order", "general_correspondence",
            "dds_submission_receipt", "status_update",
        ],
        "engines": [
            "RequestHandler",
            "InspectionCoordinator",
            "NonComplianceManager",
            "AppealProcessor",
            "DocumentExchange",
            "NotificationRouter",
            "TemplateEngine",
        ],
        "engine_count": 7,
        "enum_count": 13,
        "core_model_count": 15,
        "metrics_count": 45,
        "db_prefix": "gl_eudr_acm_",
        "metrics_prefix": "gl_eudr_acm_",
        "env_prefix": "GL_EUDR_ACM_",
    }
