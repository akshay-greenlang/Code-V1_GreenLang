"""
GreenLang Security Services Module

This module provides comprehensive security capabilities for SOC2 and ISO27001 compliance:
- Encryption Service: Field-level encryption with AWS KMS integration
- Access Control: Attribute-based access control (ABAC) with time and IP restrictions
- Secrets Management: HashiCorp Vault integration for secret rotation
- Security Monitoring: Anomaly detection and SIEM integration
"""

import logging

logger = logging.getLogger(__name__)

__all__ = []

# Encryption Service
try:
    from services.security.encryption_service import (
        EncryptionService,
        EncryptionConfig,
        EncryptedField,
        KeyVersion,
    )
    __all__.extend([
        "EncryptionService",
        "EncryptionConfig",
        "EncryptedField",
        "KeyVersion",
    ])
except (ImportError, AttributeError) as e:
    EncryptionService = None  # type: ignore
    EncryptionConfig = None  # type: ignore
    EncryptedField = None  # type: ignore
    KeyVersion = None  # type: ignore
    logger.debug(f"EncryptionService not available: {e}")

# Access Control Service
try:
    from services.security.access_control import (
        AccessControlService,
        AccessPolicy,
        AccessDecision,
        AccessContext,
    )
    __all__.extend([
        "AccessControlService",
        "AccessPolicy",
        "AccessDecision",
        "AccessContext",
    ])
except (ImportError, AttributeError) as e:
    AccessControlService = None  # type: ignore
    AccessPolicy = None  # type: ignore
    AccessDecision = None  # type: ignore
    AccessContext = None  # type: ignore
    logger.debug(f"AccessControlService not available: {e}")

# Secrets Service
try:
    from services.security.secrets_service import (
        SecretsService,
        SecretsConfig,
        SecretReference,
    )
    __all__.extend([
        "SecretsService",
        "SecretsConfig",
        "SecretReference",
    ])
except (ImportError, AttributeError) as e:
    SecretsService = None  # type: ignore
    SecretsConfig = None  # type: ignore
    SecretReference = None  # type: ignore
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
except (ImportError, AttributeError) as e:
    SecurityMonitor = None  # type: ignore
    SecurityEvent = None  # type: ignore
    SecurityAlert = None  # type: ignore
    ThreatLevel = None  # type: ignore
    logger.debug(f"SecurityMonitor not available: {e}")
