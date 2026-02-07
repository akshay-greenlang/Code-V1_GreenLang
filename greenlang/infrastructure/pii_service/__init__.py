# -*- coding: utf-8 -*-
"""
GreenLang PII Service - SEC-011: PII Detection/Redaction Enhancements

Unified PII detection, redaction, tokenization, and enforcement service
for the GreenLang Climate OS platform. Provides production-grade PII
protection with complete audit trails for regulatory compliance.

Key Features:
    - **Secure Token Vault**: AES-256-GCM encryption (replaces legacy XOR)
    - **Real-time Enforcement**: Block/quarantine PII, not just alert
    - **Allowlist Manager**: Exclude test data and known safe patterns
    - **Unified PIIService**: Single interface over all PII modules
    - **Streaming Scanner**: Kafka/Kinesis real-time scanning

Subpackages:
    - enforcement: Real-time PII enforcement engine and middleware
    - allowlist: Allowlist management for false positive filtering
    - streaming: Kafka/Kinesis real-time PII scanning
    - remediation: Automated PII remediation engine
    - api: REST API routes for PII operations

Core Exports:
    - PIIService: Unified facade over all PII capabilities
    - SecureTokenVault: AES-256-GCM encrypted token vault
    - PIIEnforcementEngine: Core enforcement logic
    - PIIEnforcementMiddleware: FastAPI middleware
    - AllowlistManager: Allowlist for false positive filtering
    - PIIRemediationEngine: Automated PII remediation
    - VaultMigrator: Migration from XOR to AES-256

Quick Start:
    >>> from greenlang.infrastructure.pii_service import (
    ...     PIIService,
    ...     PIIServiceConfig,
    ...     SecureTokenVault,
    ...     PIIType,
    ... )
    >>> from greenlang.infrastructure.encryption_service import EncryptionService
    >>>
    >>> # Initialize
    >>> config = PIIServiceConfig()
    >>> encryption_svc = await EncryptionService.create(enc_config)
    >>> pii_service = await PIIService.create(config, encryption_svc)
    >>>
    >>> # Detect and redact
    >>> result = await pii_service.redact("SSN: 123-45-6789")
    >>> print(result.redacted_content)  # "SSN: [SSN]"
    >>>
    >>> # Tokenize (reversible)
    >>> token = await pii_service.tokenize("123-45-6789", PIIType.SSN, "tenant-1")
    >>> original = await pii_service.detokenize(token, "tenant-1", "user-1")

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-011 PII Detection/Redaction Enhancements
Status: Production Ready
"""

from __future__ import annotations

import logging

# Re-export enforcement components as the primary interface
from greenlang.infrastructure.pii_service.enforcement import (
    # Enums
    PIIType,
    EnforcementAction,
    TransformationType,
    ContextType,
    QuarantineStatus,
    # Policy Models
    EnforcementPolicy,
    EnforcementContext,
    DEFAULT_POLICIES,
    get_default_policy,
    # Action Models
    PIIDetection,
    ActionTaken,
    QuarantineItem,
    EnforcementResult,
    # Engine
    PIIEnforcementEngine,
    EnforcementConfig,
    get_enforcement_engine,
    reset_engine,
    # Protocols
    PIIScannerProtocol,
    AllowlistManagerProtocol,
    NotifierProtocol,
    QuarantineStorageProtocol,
    TokenVaultProtocol,
    # Middleware
    MiddlewareConfig,
    PIIEnforcementMiddleware,
    PIIEnforcementASGIMiddleware,
    PIIEnforcementDependency,
    create_pii_error_response,
)

# ---------------------------------------------------------------------------
# Allowlist Components
# ---------------------------------------------------------------------------

from greenlang.infrastructure.pii_service.allowlist import (
    # Models
    PatternType,
    AllowlistEntry,
    AllowlistConfig,
    # Manager
    AllowlistManager,
    # Exceptions
    AllowlistError,
    InvalidPatternError,
    EntryNotFoundError,
    EntryLimitExceededError,
    # Defaults
    DEFAULT_ALLOWLISTS,
    get_allowlist_for_type,
    get_default_allowlist_count,
    # Factory
    get_allowlist_manager,
    reset_allowlist_manager,
)

# ---------------------------------------------------------------------------
# Remediation Components
# ---------------------------------------------------------------------------

from greenlang.infrastructure.pii_service.remediation import (
    # Enums
    RemediationAction,
    RemediationStatus,
    SourceType,
    # Policy Models
    RemediationPolicy,
    PIIRemediationItem,
    DeletionCertificate,
    RemediationResult,
    # Engine
    RemediationConfig,
    PIIRemediationEngine,
    RemediationError,
    SourceConnectionError,
    RemediationExecutionError,
    # Defaults
    DEFAULT_REMEDIATION_POLICIES,
    get_default_policy as get_default_remediation_policy,
    get_all_default_policies,
    # Factory
    get_remediation_engine,
    reset_remediation_engine,
    # Jobs
    JobConfig,
    JobStatus,
    PIIRemediationJob,
    run_remediation_cron,
)

# ---------------------------------------------------------------------------
# Configuration (Phase 1)
# ---------------------------------------------------------------------------

from greenlang.infrastructure.pii_service.config import (
    # Enums
    PersistenceBackend,
    EnforcementMode,
    StreamingPlatform,
    RemediationAction as ConfigRemediationAction,
    # Component configs
    VaultConfig,
    EnforcementConfig as ServiceEnforcementConfig,
    AllowlistConfig as ServiceAllowlistConfig,
    StreamingConfig,
    RemediationConfig as ServiceRemediationConfig,
    ScannerConfig,
    # Main config
    PIIServiceConfig,
    # Functions
    get_pii_service_config,
    configure_pii_service as configure_pii_config,
)

# ---------------------------------------------------------------------------
# Models (Phase 1)
# ---------------------------------------------------------------------------

from greenlang.infrastructure.pii_service.models import (
    # Re-exported from pii_redaction
    RedactionStrategy,
    # Detection enums and models
    DetectionMethod,
    ConfidenceLevel,
    PIIDetection as ServicePIIDetection,
    DetectionOptions,
    # Token vault
    EncryptedTokenEntry,
    # Redaction
    RedactionResult,
    RedactionOptions,
    # Enforcement models
    EnforcementAction as ServiceEnforcementAction,
    EnforcementContext as ServiceEnforcementContext,
    ActionTaken as ServiceActionTaken,
    EnforcementResult as ServiceEnforcementResult,
    # Allowlist
    AllowlistEntry as ServiceAllowlistEntry,
    # API models
    DetectRequest,
    DetectResponse,
    RedactRequest,
    RedactResponse,
    TokenizeRequest,
    TokenizeResponse,
    DetokenizeRequest,
    DetokenizeResponse,
)

# ---------------------------------------------------------------------------
# Secure Token Vault (Phase 1)
# ---------------------------------------------------------------------------

from greenlang.infrastructure.pii_service.secure_vault import (
    SecureTokenVault,
    # Exceptions
    VaultError,
    TokenNotFoundError,
    UnauthorizedAccessError,
    TokenExpiredError,
    VaultCapacityError,
)

# ---------------------------------------------------------------------------
# Vault Migration (Phase 1)
# ---------------------------------------------------------------------------

from greenlang.infrastructure.pii_service.vault_migration import (
    VaultMigrator,
    MigrationResult,
    VerificationResult,
    MigrationProgress,
    LegacyTokenEntry,
)

# ---------------------------------------------------------------------------
# Unified PII Service (Phase 6)
# ---------------------------------------------------------------------------

from greenlang.infrastructure.pii_service.service import (
    PIIService,
    PIIEnforcementEngine as UnifiedPIIEnforcementEngine,
    AllowlistManager as UnifiedAllowlistManager,
    # Functions
    get_pii_service,
    configure_pii_service,
    # Constants
    DEFAULT_ENFORCEMENT_POLICIES,
    DEFAULT_ALLOWLIST_ENTRIES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Version
    "__version__",
    # Configuration
    "PersistenceBackend",
    "EnforcementMode",
    "StreamingPlatform",
    "ConfigRemediationAction",
    "VaultConfig",
    "ServiceEnforcementConfig",
    "ServiceAllowlistConfig",
    "StreamingConfig",
    "ServiceRemediationConfig",
    "ScannerConfig",
    "PIIServiceConfig",
    "get_pii_service_config",
    "configure_pii_config",
    # Models
    "RedactionStrategy",
    "DetectionMethod",
    "ConfidenceLevel",
    "ServicePIIDetection",
    "DetectionOptions",
    "EncryptedTokenEntry",
    "RedactionResult",
    "RedactionOptions",
    "ServiceEnforcementAction",
    "ServiceEnforcementContext",
    "ServiceActionTaken",
    "ServiceEnforcementResult",
    "ServiceAllowlistEntry",
    "DetectRequest",
    "DetectResponse",
    "RedactRequest",
    "RedactResponse",
    "TokenizeRequest",
    "TokenizeResponse",
    "DetokenizeRequest",
    "DetokenizeResponse",
    # Secure Token Vault
    "SecureTokenVault",
    "VaultError",
    "TokenNotFoundError",
    "UnauthorizedAccessError",
    "TokenExpiredError",
    "VaultCapacityError",
    # Vault Migration
    "VaultMigrator",
    "MigrationResult",
    "VerificationResult",
    "MigrationProgress",
    "LegacyTokenEntry",
    # Unified PII Service
    "PIIService",
    "UnifiedPIIEnforcementEngine",
    "UnifiedAllowlistManager",
    "get_pii_service",
    "configure_pii_service",
    "DEFAULT_ENFORCEMENT_POLICIES",
    "DEFAULT_ALLOWLIST_ENTRIES",
    # Enforcement - Enums
    "PIIType",
    "EnforcementAction",
    "TransformationType",
    "ContextType",
    "QuarantineStatus",
    # Enforcement - Policy Models
    "EnforcementPolicy",
    "EnforcementContext",
    "DEFAULT_POLICIES",
    "get_default_policy",
    # Enforcement - Action Models
    "PIIDetection",
    "ActionTaken",
    "QuarantineItem",
    "EnforcementResult",
    # Enforcement - Engine
    "PIIEnforcementEngine",
    "EnforcementConfig",
    "get_enforcement_engine",
    "reset_engine",
    # Enforcement - Protocols
    "PIIScannerProtocol",
    "AllowlistManagerProtocol",
    "NotifierProtocol",
    "QuarantineStorageProtocol",
    "TokenVaultProtocol",
    # Enforcement - Middleware
    "MiddlewareConfig",
    "PIIEnforcementMiddleware",
    "PIIEnforcementASGIMiddleware",
    "PIIEnforcementDependency",
    "create_pii_error_response",
    # Allowlist - Models
    "PatternType",
    "AllowlistEntry",
    "AllowlistConfig",
    # Allowlist - Manager
    "AllowlistManager",
    # Allowlist - Exceptions
    "AllowlistError",
    "InvalidPatternError",
    "EntryNotFoundError",
    "EntryLimitExceededError",
    # Allowlist - Defaults
    "DEFAULT_ALLOWLISTS",
    "get_allowlist_for_type",
    "get_default_allowlist_count",
    # Allowlist - Factory
    "get_allowlist_manager",
    "reset_allowlist_manager",
    # Remediation - Enums
    "RemediationAction",
    "RemediationStatus",
    "SourceType",
    # Remediation - Policy Models
    "RemediationPolicy",
    "PIIRemediationItem",
    "DeletionCertificate",
    "RemediationResult",
    # Remediation - Engine
    "RemediationConfig",
    "PIIRemediationEngine",
    "RemediationError",
    "SourceConnectionError",
    "RemediationExecutionError",
    # Remediation - Defaults
    "DEFAULT_REMEDIATION_POLICIES",
    "get_default_remediation_policy",
    "get_all_default_policies",
    # Remediation - Factory
    "get_remediation_engine",
    "reset_remediation_engine",
    # Remediation - Jobs
    "JobConfig",
    "JobStatus",
    "PIIRemediationJob",
    "run_remediation_cron",
]

logger.debug("PII Service module loaded: version=%s", __version__)
