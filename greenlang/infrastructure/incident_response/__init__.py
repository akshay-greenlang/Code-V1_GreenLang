# -*- coding: utf-8 -*-
"""
GreenLang Incident Response Automation Module - SEC-010

Production-grade incident response automation for GreenLang Climate OS.
Provides automated incident detection, correlation, classification,
escalation, notification, and remediation through playbook execution.

This module implements SEC-010 Phase 1 from the GreenLang security program:
    - IncidentDetector: Aggregates alerts from Prometheus, Loki, GuardDuty, CloudTrail
    - IncidentCorrelator: Groups related alerts into incidents
    - IncidentClassifier: Determines severity and business impact
    - EscalationEngine: Manages escalation workflow and SLAs
    - Notifier: PagerDuty, Slack, Email, SMS notifications
    - PlaybookExecutor: Automated remediation playbooks
    - IncidentTracker: Incident lifecycle management

Quick start:
    >>> from greenlang.infrastructure.incident_response import (
    ...     IncidentDetector,
    ...     IncidentCorrelator,
    ...     IncidentClassifier,
    ...     PlaybookExecutor,
    ... )
    >>> detector = IncidentDetector()
    >>> alerts = await detector.detect_incidents()
    >>> correlator = IncidentCorrelator()
    >>> incidents = await correlator.correlate(alerts)

Security Compliance: SOC 2 CC7.2, ISO 27001 A.16, NIST IR

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
Version: 1.0.0
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Module version
__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from greenlang.infrastructure.incident_response.config import (
    # Constants
    DEFAULT_ESCALATION_THRESHOLDS,
    DEFAULT_RESPONSE_SLAS,
    DEFAULT_RESOLUTION_SLAS,
    # Alert Source Configs
    AlertSourceConfig,
    PrometheusAlertConfig,
    LokiAlertConfig,
    GuardDutyConfig,
    CloudTrailConfig,
    # Notification Configs
    PagerDutyConfig,
    SlackConfig,
    EmailConfig,
    SMSConfig,
    # Other Configs
    PlaybookConfig,
    JiraConfig,
    # Main Config
    IncidentResponseConfig,
    # Functions
    get_config,
    configure,
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

from greenlang.infrastructure.incident_response.models import (
    # Enums
    EscalationLevel,
    IncidentStatus,
    IncidentType,
    AlertSource,
    PlaybookStatus,
    TimelineEventType,
    # Models
    Alert,
    Incident,
    PlaybookStep,
    PlaybookExecution,
    TimelineEvent,
    PostMortem,
    IncidentMetricsSummary,
)

# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

from greenlang.infrastructure.incident_response.detector import (
    IncidentDetector,
    get_detector,
    reset_detector,
)

# ---------------------------------------------------------------------------
# Correlator
# ---------------------------------------------------------------------------

from greenlang.infrastructure.incident_response.correlator import (
    IncidentCorrelator,
    get_correlator,
    reset_correlator,
    ALERT_TYPE_TO_INCIDENT_TYPE,
)

# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

from greenlang.infrastructure.incident_response.classifier import (
    IncidentClassifier,
    SeverityLevel,
    BusinessImpact,
    SEVERITY_LEVELS,
    INCIDENT_TYPE_BASE_SEVERITY,
    get_classifier,
    reset_classifier,
)

# ---------------------------------------------------------------------------
# Escalator
# ---------------------------------------------------------------------------

from greenlang.infrastructure.incident_response.escalator import (
    EscalationEngine,
    OnCallResponder,
    EscalationRecord,
    get_escalation_engine,
    reset_escalation_engine,
)

# ---------------------------------------------------------------------------
# Notifier
# ---------------------------------------------------------------------------

from greenlang.infrastructure.incident_response.notifier import (
    Notifier,
    TEMPLATES,
    SEVERITY_EMOJI,
    get_notifier,
    reset_notifier,
)

# ---------------------------------------------------------------------------
# Playbook Executor
# ---------------------------------------------------------------------------

from greenlang.infrastructure.incident_response.playbook_executor import (
    # Base classes
    BasePlaybook,
    PlaybookResult,
    # Playbooks
    CredentialCompromisePlaybook,
    DDoSMitigationPlaybook,
    DataBreachPlaybook,
    MalwareContainmentPlaybook,
    AccessRevocationPlaybook,
    SessionHijackPlaybook,
    BruteForceResponsePlaybook,
    SQLInjectionResponsePlaybook,
    APIAbusePlaybook,
    InsiderThreatPlaybook,
    # Registry
    PLAYBOOKS,
    # Executor
    PlaybookExecutor,
    get_playbook_executor,
    reset_playbook_executor,
)

# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

from greenlang.infrastructure.incident_response.tracker import (
    IncidentTracker,
    get_tracker,
    reset_tracker,
)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

from greenlang.infrastructure.incident_response.metrics import (
    IncidentResponseMetrics,
    get_metrics,
)

# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

try:
    from greenlang.infrastructure.incident_response.api import (
        incident_router,
        FASTAPI_AVAILABLE,
    )
except ImportError:
    incident_router = None  # type: ignore[assignment]
    FASTAPI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Version
    "__version__",
    # Configuration
    "DEFAULT_ESCALATION_THRESHOLDS",
    "DEFAULT_RESPONSE_SLAS",
    "DEFAULT_RESOLUTION_SLAS",
    "AlertSourceConfig",
    "PrometheusAlertConfig",
    "LokiAlertConfig",
    "GuardDutyConfig",
    "CloudTrailConfig",
    "PagerDutyConfig",
    "SlackConfig",
    "EmailConfig",
    "SMSConfig",
    "PlaybookConfig",
    "JiraConfig",
    "IncidentResponseConfig",
    "get_config",
    "configure",
    # Enums
    "EscalationLevel",
    "IncidentStatus",
    "IncidentType",
    "AlertSource",
    "PlaybookStatus",
    "TimelineEventType",
    # Models
    "Alert",
    "Incident",
    "PlaybookStep",
    "PlaybookExecution",
    "TimelineEvent",
    "PostMortem",
    "IncidentMetricsSummary",
    # Detector
    "IncidentDetector",
    "get_detector",
    "reset_detector",
    # Correlator
    "IncidentCorrelator",
    "get_correlator",
    "reset_correlator",
    "ALERT_TYPE_TO_INCIDENT_TYPE",
    # Classifier
    "IncidentClassifier",
    "SeverityLevel",
    "BusinessImpact",
    "SEVERITY_LEVELS",
    "INCIDENT_TYPE_BASE_SEVERITY",
    "get_classifier",
    "reset_classifier",
    # Escalator
    "EscalationEngine",
    "OnCallResponder",
    "EscalationRecord",
    "get_escalation_engine",
    "reset_escalation_engine",
    # Notifier
    "Notifier",
    "TEMPLATES",
    "SEVERITY_EMOJI",
    "get_notifier",
    "reset_notifier",
    # Playbook Executor
    "BasePlaybook",
    "PlaybookResult",
    "CredentialCompromisePlaybook",
    "DDoSMitigationPlaybook",
    "DataBreachPlaybook",
    "MalwareContainmentPlaybook",
    "AccessRevocationPlaybook",
    "SessionHijackPlaybook",
    "BruteForceResponsePlaybook",
    "SQLInjectionResponsePlaybook",
    "APIAbusePlaybook",
    "InsiderThreatPlaybook",
    "PLAYBOOKS",
    "PlaybookExecutor",
    "get_playbook_executor",
    "reset_playbook_executor",
    # Tracker
    "IncidentTracker",
    "get_tracker",
    "reset_tracker",
    # Metrics
    "IncidentResponseMetrics",
    "get_metrics",
    # API
    "incident_router",
    "FASTAPI_AVAILABLE",
]

logger.debug("Incident response module loaded (version %s)", __version__)
