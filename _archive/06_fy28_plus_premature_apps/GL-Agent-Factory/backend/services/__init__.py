"""
GreenLang Agent Factory - Services Module

This module contains all core business services for the Agent Factory:
- Execution Service: Agent execution with provenance tracking
- Registry Service: Agent lifecycle and version management
- Calculation Engine: Zero-hallucination calculations
- Workflow Orchestration: Multi-agent pipeline management
- Audit Logging: Comprehensive audit trails
- Security Services: SOC2/ISO27001 compliance controls
- Compliance Services: Control checks and evidence generation
"""

import logging

logger = logging.getLogger(__name__)

# Build __all__ dynamically based on available modules
__all__ = []

# Core Services - Required
from services.execution.agent_execution_service import AgentExecutionService
__all__.append("AgentExecutionService")

from services.registry.agent_registry_service import AgentRegistryService
__all__.append("AgentRegistryService")

# Optional Core Services - gracefully handle missing modules
try:
    from services.calculation.calculation_engine_service import CalculationEngineService
    __all__.append("CalculationEngineService")
except (ImportError, AttributeError, Exception) as e:
    CalculationEngineService = None  # type: ignore
    logger.debug(f"CalculationEngineService not available: {e}")

try:
    from services.workflow.workflow_orchestration_service import WorkflowOrchestrationService
    __all__.append("WorkflowOrchestrationService")
except (ImportError, AttributeError, Exception) as e:
    WorkflowOrchestrationService = None  # type: ignore
    logger.debug(f"WorkflowOrchestrationService not available: {e}")

try:
    from services.audit.audit_logging_service import AuditLoggingService
    __all__.append("AuditLoggingService")
except (ImportError, AttributeError, Exception) as e:
    AuditLoggingService = None  # type: ignore
    logger.debug(f"AuditLoggingService not available: {e}")

# Audit Service
try:
    from services.audit.audit_service import (
        AuditService,
        AuditEventType,
        AuditSeverity,
        AuditEntry,
        AuditExportFormat,
    )
    __all__.extend([
        "AuditService",
        "AuditEventType",
        "AuditSeverity",
        "AuditEntry",
        "AuditExportFormat",
    ])
except (ImportError, AttributeError, Exception) as e:
    AuditService = None  # type: ignore
    AuditEventType = None  # type: ignore
    AuditSeverity = None  # type: ignore
    AuditEntry = None  # type: ignore
    AuditExportFormat = None  # type: ignore
    logger.debug(f"AuditService not available: {e}")

# Encryption Service
try:
    from services.security.encryption_service import (
        EncryptionService,
        EncryptionConfig,
        EncryptedField,
        DataClassification,
    )
    __all__.extend([
        "EncryptionService",
        "EncryptionConfig",
        "EncryptedField",
        "DataClassification",
    ])
except (ImportError, AttributeError, Exception) as e:
    EncryptionService = None  # type: ignore
    EncryptionConfig = None  # type: ignore
    EncryptedField = None  # type: ignore
    DataClassification = None  # type: ignore
    logger.debug(f"EncryptionService not available: {e}")

# Access Control Service
try:
    from services.security.access_control import (
        AccessControlService,
        AccessPolicy,
        AccessDecision,
        AccessContext,
        Action,
    )
    __all__.extend([
        "AccessControlService",
        "AccessPolicy",
        "AccessDecision",
        "AccessContext",
        "Action",
    ])
except (ImportError, AttributeError, Exception) as e:
    AccessControlService = None  # type: ignore
    AccessPolicy = None  # type: ignore
    AccessDecision = None  # type: ignore
    AccessContext = None  # type: ignore
    Action = None  # type: ignore
    logger.debug(f"AccessControlService not available: {e}")

# Secrets Service
try:
    from services.security.secrets_service import (
        SecretsService,
        SecretsConfig,
        SecretReference,
        APIKey,
    )
    __all__.extend([
        "SecretsService",
        "SecretsConfig",
        "SecretReference",
        "APIKey",
    ])
except (ImportError, AttributeError, Exception) as e:
    SecretsService = None  # type: ignore
    SecretsConfig = None  # type: ignore
    SecretReference = None  # type: ignore
    APIKey = None  # type: ignore
    logger.debug(f"SecretsService not available: {e}")

# Security Monitor
try:
    from services.security.security_monitor import (
        SecurityMonitor,
        SecurityEvent,
        SecurityAlert,
        ThreatLevel,
    )
    __all__.extend([
        "SecurityMonitor",
        "SecurityEvent",
        "SecurityAlert",
        "ThreatLevel",
    ])
except (ImportError, AttributeError, Exception) as e:
    SecurityMonitor = None  # type: ignore
    SecurityEvent = None  # type: ignore
    SecurityAlert = None  # type: ignore
    ThreatLevel = None  # type: ignore
    logger.debug(f"SecurityMonitor not available: {e}")

# Compliance Services
try:
    from services.compliance.compliance_checks import (
        ComplianceService,
        ComplianceFramework,
        ControlCheck,
        ComplianceReport,
    )
    __all__.extend([
        "ComplianceService",
        "ComplianceFramework",
        "ControlCheck",
        "ComplianceReport",
    ])
except (ImportError, AttributeError, Exception) as e:
    ComplianceService = None  # type: ignore
    ComplianceFramework = None  # type: ignore
    ControlCheck = None  # type: ignore
    ComplianceReport = None  # type: ignore
    logger.debug(f"ComplianceService not available: {e}")
