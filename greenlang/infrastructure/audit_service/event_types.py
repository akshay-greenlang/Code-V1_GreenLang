# -*- coding: utf-8 -*-
"""
Audit Event Types - SEC-005: Centralized Audit Logging Service

Defines all audit event type enumerations for the unified audit logging system.
Consolidates 70+ event types from auth, RBAC, encryption, data, agent, system,
API, and compliance domains into a single coherent taxonomy.

**Design Principles:**
- Lowercase snake_case values for Loki label compatibility
- String-based enums for JSON serialization
- Hierarchical naming (category.subcategory.action)
- Extensible without breaking existing integrations

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

from enum import Enum


class AuditEventCategory(str, Enum):
    """Top-level categories for audit events.

    Used for routing, filtering, and dashboard organization.
    Maps to Loki stream labels for efficient querying.
    """

    AUTH = "auth"
    RBAC = "rbac"
    ENCRYPTION = "encryption"
    DATA = "data"
    AGENT = "agent"
    SYSTEM = "system"
    API = "api"
    COMPLIANCE = "compliance"


class AuditSeverity(str, Enum):
    """Severity levels for audit events.

    Follows syslog severity conventions for SIEM compatibility.
    Maps to Python logging levels for consistent handling.
    """

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditAction(str, Enum):
    """Standard CRUD actions for audit events.

    Normalized action vocabulary for consistent querying.
    """

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    EXPORT = "export"
    IMPORT = "import"
    GRANT = "grant"
    REVOKE = "revoke"
    VALIDATE = "validate"
    ENCRYPT = "encrypt"
    DECRYPT = "decrypt"


class AuditResult(str, Enum):
    """Outcome of an audited operation.

    Standardized result vocabulary for filtering and alerting.
    """

    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    ERROR = "error"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class UnifiedAuditEventType(str, Enum):
    """Unified enumeration of all 70+ auditable event types.

    Consolidates event types from:
    - SEC-001: JWT Authentication (17 events)
    - SEC-002: RBAC Authorization (13 events)
    - SEC-003: Encryption at Rest (20 events)
    - SEC-004: Data Access (8 events)
    - Agents: Execution lifecycle (8 events)
    - System: Health and operations (6 events)
    - API: Request lifecycle (6 events)
    - Compliance: Regulatory events (6 events)

    Values are lowercase snake_case for Loki label compatibility.
    """

    # =========================================================================
    # AUTH EVENTS (SEC-001) - 17 types
    # =========================================================================
    AUTH_LOGIN_SUCCESS = "auth.login_success"
    AUTH_LOGIN_FAILURE = "auth.login_failure"
    AUTH_LOGOUT = "auth.logout"
    AUTH_TOKEN_ISSUED = "auth.token_issued"
    AUTH_TOKEN_VALIDATED = "auth.token_validated"
    AUTH_TOKEN_REVOKED = "auth.token_revoked"
    AUTH_TOKEN_REFRESHED = "auth.token_refreshed"
    AUTH_TOKEN_EXPIRED = "auth.token_expired"
    AUTH_PASSWORD_CHANGED = "auth.password_changed"
    AUTH_PASSWORD_RESET_REQUESTED = "auth.password_reset_requested"
    AUTH_PASSWORD_RESET_COMPLETED = "auth.password_reset_completed"
    AUTH_MFA_SETUP = "auth.mfa_setup"
    AUTH_MFA_VERIFIED = "auth.mfa_verified"
    AUTH_MFA_FAILED = "auth.mfa_failed"
    AUTH_ACCOUNT_LOCKED = "auth.account_locked"
    AUTH_ACCOUNT_UNLOCKED = "auth.account_unlocked"
    AUTH_SESSION_EXPIRED = "auth.session_expired"

    # =========================================================================
    # RBAC EVENTS (SEC-002) - 13 types
    # =========================================================================
    RBAC_ROLE_CREATED = "rbac.role_created"
    RBAC_ROLE_UPDATED = "rbac.role_updated"
    RBAC_ROLE_DELETED = "rbac.role_deleted"
    RBAC_ROLE_ENABLED = "rbac.role_enabled"
    RBAC_ROLE_DISABLED = "rbac.role_disabled"
    RBAC_PERMISSION_GRANTED = "rbac.permission_granted"
    RBAC_PERMISSION_REVOKED = "rbac.permission_revoked"
    RBAC_ROLE_ASSIGNED = "rbac.role_assigned"
    RBAC_ROLE_REVOKED = "rbac.role_revoked"
    RBAC_ROLE_EXPIRED = "rbac.role_expired"
    RBAC_AUTHORIZATION_ALLOWED = "rbac.authorization_allowed"
    RBAC_AUTHORIZATION_DENIED = "rbac.authorization_denied"
    RBAC_CACHE_INVALIDATED = "rbac.cache_invalidated"

    # =========================================================================
    # ENCRYPTION EVENTS (SEC-003) - 20 types
    # =========================================================================
    ENCRYPTION_PERFORMED = "encryption.performed"
    DECRYPTION_PERFORMED = "encryption.decryption_performed"
    ENCRYPTION_FAILED = "encryption.failed"
    DECRYPTION_FAILED = "encryption.decryption_failed"
    ENCRYPTION_KEY_GENERATED = "encryption.key_generated"
    ENCRYPTION_KEY_ROTATED = "encryption.key_rotated"
    ENCRYPTION_KEY_ACCESSED = "encryption.key_accessed"
    ENCRYPTION_KEY_EXPIRED = "encryption.key_expired"
    ENCRYPTION_KEY_REVOKED = "encryption.key_revoked"
    ENCRYPTION_CACHE_HIT = "encryption.cache_hit"
    ENCRYPTION_CACHE_MISS = "encryption.cache_miss"
    ENCRYPTION_CACHE_INVALIDATED = "encryption.cache_invalidated"
    ENCRYPTION_CACHE_EVICTED = "encryption.cache_evicted"
    ENCRYPTION_KMS_CALL = "encryption.kms_call"
    ENCRYPTION_KMS_ERROR = "encryption.kms_error"
    ENCRYPTION_KMS_ENCRYPT_DEK = "encryption.kms_encrypt_dek"
    ENCRYPTION_KMS_DECRYPT_DEK = "encryption.kms_decrypt_dek"
    ENCRYPTION_FIELD_ENCRYPTED = "encryption.field_encrypted"
    ENCRYPTION_FIELD_DECRYPTED = "encryption.field_decrypted"
    ENCRYPTION_BATCH_OPERATION = "encryption.batch_operation"

    # =========================================================================
    # DATA ACCESS EVENTS - 8 types
    # =========================================================================
    DATA_READ = "data.read"
    DATA_WRITE = "data.write"
    DATA_DELETE = "data.delete"
    DATA_EXPORT = "data.export"
    DATA_IMPORT = "data.import"
    DATA_QUERY = "data.query"
    DATA_BULK_OPERATION = "data.bulk_operation"
    DATA_SCHEMA_CHANGED = "data.schema_changed"

    # =========================================================================
    # AGENT EVENTS - 8 types
    # =========================================================================
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"
    AGENT_CANCELLED = "agent.cancelled"
    AGENT_STEP_COMPLETED = "agent.step_completed"
    AGENT_VALIDATION_PASSED = "agent.validation_passed"
    AGENT_VALIDATION_FAILED = "agent.validation_failed"
    AGENT_PROVENANCE_RECORDED = "agent.provenance_recorded"

    # =========================================================================
    # SYSTEM EVENTS - 6 types
    # =========================================================================
    SYSTEM_STARTED = "system.started"
    SYSTEM_STOPPED = "system.stopped"
    SYSTEM_CONFIG_CHANGED = "system.config_changed"
    SYSTEM_HEALTH_CHECK = "system.health_check"
    SYSTEM_ERROR = "system.error"
    SYSTEM_MAINTENANCE = "system.maintenance"

    # =========================================================================
    # API EVENTS - 6 types
    # =========================================================================
    API_REQUEST_RECEIVED = "api.request_received"
    API_REQUEST_COMPLETED = "api.request_completed"
    API_REQUEST_FAILED = "api.request_failed"
    API_RATE_LIMITED = "api.rate_limited"
    API_WEBHOOK_SENT = "api.webhook_sent"
    API_WEBHOOK_FAILED = "api.webhook_failed"

    # =========================================================================
    # COMPLIANCE EVENTS - 6 types
    # =========================================================================
    COMPLIANCE_REPORT_GENERATED = "compliance.report_generated"
    COMPLIANCE_AUDIT_COMPLETED = "compliance.audit_completed"
    COMPLIANCE_VIOLATION_DETECTED = "compliance.violation_detected"
    COMPLIANCE_ATTESTATION_SIGNED = "compliance.attestation_signed"
    COMPLIANCE_DATA_RETENTION = "compliance.data_retention"
    COMPLIANCE_GDPR_REQUEST = "compliance.gdpr_request"

    @classmethod
    def get_category(cls, event_type: "UnifiedAuditEventType") -> AuditEventCategory:
        """Get the category for an event type.

        Args:
            event_type: The event type to categorize.

        Returns:
            The category for the event type.
        """
        prefix = event_type.value.split(".")[0]
        return AuditEventCategory(prefix)

    @classmethod
    def get_default_severity(
        cls, event_type: "UnifiedAuditEventType"
    ) -> AuditSeverity:
        """Get the default severity for an event type.

        Args:
            event_type: The event type to get severity for.

        Returns:
            The default severity level.
        """
        # Critical events
        critical_events = {
            cls.COMPLIANCE_VIOLATION_DETECTED,
            cls.SYSTEM_ERROR,
            cls.AUTH_ACCOUNT_LOCKED,
        }
        if event_type in critical_events:
            return AuditSeverity.CRITICAL

        # Error events
        error_events = {
            cls.AUTH_LOGIN_FAILURE,
            cls.ENCRYPTION_FAILED,
            cls.DECRYPTION_FAILED,
            cls.AGENT_FAILED,
            cls.API_REQUEST_FAILED,
            cls.API_WEBHOOK_FAILED,
            cls.ENCRYPTION_KMS_ERROR,
        }
        if event_type in error_events:
            return AuditSeverity.ERROR

        # Warning events
        warning_events = {
            cls.AUTH_MFA_FAILED,
            cls.RBAC_AUTHORIZATION_DENIED,
            cls.API_RATE_LIMITED,
            cls.AGENT_VALIDATION_FAILED,
            cls.AGENT_CANCELLED,
            cls.AUTH_PASSWORD_RESET_REQUESTED,
        }
        if event_type in warning_events:
            return AuditSeverity.WARNING

        # Default to INFO
        return AuditSeverity.INFO


# ---------------------------------------------------------------------------
# Mapping from legacy event types to unified types
# ---------------------------------------------------------------------------

# Maps AuthEventType values to UnifiedAuditEventType
AUTH_EVENT_TYPE_MAP = {
    "login_success": UnifiedAuditEventType.AUTH_LOGIN_SUCCESS,
    "login_failure": UnifiedAuditEventType.AUTH_LOGIN_FAILURE,
    "token_issued": UnifiedAuditEventType.AUTH_TOKEN_ISSUED,
    "token_validated": UnifiedAuditEventType.AUTH_TOKEN_VALIDATED,
    "token_revoked": UnifiedAuditEventType.AUTH_TOKEN_REVOKED,
    "token_refreshed": UnifiedAuditEventType.AUTH_TOKEN_REFRESHED,
    "logout": UnifiedAuditEventType.AUTH_LOGOUT,
    "password_changed": UnifiedAuditEventType.AUTH_PASSWORD_CHANGED,
    "password_reset_requested": UnifiedAuditEventType.AUTH_PASSWORD_RESET_REQUESTED,
    "mfa_setup": UnifiedAuditEventType.AUTH_MFA_SETUP,
    "mfa_verified": UnifiedAuditEventType.AUTH_MFA_VERIFIED,
    "mfa_failed": UnifiedAuditEventType.AUTH_MFA_FAILED,
    "account_locked": UnifiedAuditEventType.AUTH_ACCOUNT_LOCKED,
    "account_unlocked": UnifiedAuditEventType.AUTH_ACCOUNT_UNLOCKED,
    "permission_denied": UnifiedAuditEventType.RBAC_AUTHORIZATION_DENIED,
    "session_expired": UnifiedAuditEventType.AUTH_SESSION_EXPIRED,
    "rate_limited": UnifiedAuditEventType.API_RATE_LIMITED,
}

# Maps RBACAuditEventType values to UnifiedAuditEventType
RBAC_EVENT_TYPE_MAP = {
    "role_created": UnifiedAuditEventType.RBAC_ROLE_CREATED,
    "role_updated": UnifiedAuditEventType.RBAC_ROLE_UPDATED,
    "role_deleted": UnifiedAuditEventType.RBAC_ROLE_DELETED,
    "role_enabled": UnifiedAuditEventType.RBAC_ROLE_ENABLED,
    "role_disabled": UnifiedAuditEventType.RBAC_ROLE_DISABLED,
    "permission_granted": UnifiedAuditEventType.RBAC_PERMISSION_GRANTED,
    "permission_revoked": UnifiedAuditEventType.RBAC_PERMISSION_REVOKED,
    "role_assigned": UnifiedAuditEventType.RBAC_ROLE_ASSIGNED,
    "role_revoked": UnifiedAuditEventType.RBAC_ROLE_REVOKED,
    "role_expired": UnifiedAuditEventType.RBAC_ROLE_EXPIRED,
    "authorization_allowed": UnifiedAuditEventType.RBAC_AUTHORIZATION_ALLOWED,
    "authorization_denied": UnifiedAuditEventType.RBAC_AUTHORIZATION_DENIED,
    "cache_invalidated": UnifiedAuditEventType.RBAC_CACHE_INVALIDATED,
}

# Maps EncryptionAuditEventType values to UnifiedAuditEventType
ENCRYPTION_EVENT_TYPE_MAP = {
    "encryption_performed": UnifiedAuditEventType.ENCRYPTION_PERFORMED,
    "decryption_performed": UnifiedAuditEventType.DECRYPTION_PERFORMED,
    "encryption_failed": UnifiedAuditEventType.ENCRYPTION_FAILED,
    "decryption_failed": UnifiedAuditEventType.DECRYPTION_FAILED,
    "key_generated": UnifiedAuditEventType.ENCRYPTION_KEY_GENERATED,
    "key_rotated": UnifiedAuditEventType.ENCRYPTION_KEY_ROTATED,
    "key_accessed": UnifiedAuditEventType.ENCRYPTION_KEY_ACCESSED,
    "key_expired": UnifiedAuditEventType.ENCRYPTION_KEY_EXPIRED,
    "key_revoked": UnifiedAuditEventType.ENCRYPTION_KEY_REVOKED,
    "key_cache_hit": UnifiedAuditEventType.ENCRYPTION_CACHE_HIT,
    "key_cache_miss": UnifiedAuditEventType.ENCRYPTION_CACHE_MISS,
    "key_cache_invalidated": UnifiedAuditEventType.ENCRYPTION_CACHE_INVALIDATED,
    "key_cache_evicted": UnifiedAuditEventType.ENCRYPTION_CACHE_EVICTED,
    "kms_call": UnifiedAuditEventType.ENCRYPTION_KMS_CALL,
    "kms_error": UnifiedAuditEventType.ENCRYPTION_KMS_ERROR,
    "kms_encrypt_dek": UnifiedAuditEventType.ENCRYPTION_KMS_ENCRYPT_DEK,
    "kms_decrypt_dek": UnifiedAuditEventType.ENCRYPTION_KMS_DECRYPT_DEK,
    "field_encrypted": UnifiedAuditEventType.ENCRYPTION_FIELD_ENCRYPTED,
    "field_decrypted": UnifiedAuditEventType.ENCRYPTION_FIELD_DECRYPTED,
    "batch_encryption": UnifiedAuditEventType.ENCRYPTION_BATCH_OPERATION,
    "batch_decryption": UnifiedAuditEventType.ENCRYPTION_BATCH_OPERATION,
}


__all__ = [
    "AuditEventCategory",
    "AuditSeverity",
    "AuditAction",
    "AuditResult",
    "UnifiedAuditEventType",
    "AUTH_EVENT_TYPE_MAP",
    "RBAC_EVENT_TYPE_MAP",
    "ENCRYPTION_EVENT_TYPE_MAP",
]
