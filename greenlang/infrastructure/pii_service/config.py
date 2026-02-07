# -*- coding: utf-8 -*-
"""
PII Service Configuration - SEC-011: PII Detection/Redaction Enhancements

Centralized configuration for the PII Service components including:
- Secure Token Vault (AES-256-GCM encryption)
- Enforcement Engine (real-time blocking/redaction)
- Allowlist Manager (false positive filtering)
- Streaming Scanner (Kafka/Kinesis integration)
- Auto-Remediation (scheduled PII cleanup)

Configuration follows GreenLang patterns using pydantic-settings for
environment-based overrides and sensible production defaults.

Author: GreenLang Framework Team
Date: February 2026
PRD: SEC-011 PII Detection/Redaction Enhancements
Status: Production Ready
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PersistenceBackend(str, Enum):
    """Token vault persistence backend options."""

    POSTGRESQL = "postgresql"
    REDIS = "redis"
    S3 = "s3"
    MEMORY = "memory"  # For testing only


class EnforcementMode(str, Enum):
    """Enforcement engine operating modes."""

    DISABLED = "disabled"  # No enforcement, detection only
    AUDIT = "audit"  # Log detections but don't block
    WARN = "warn"  # Warn users but allow through
    ENFORCE = "enforce"  # Full enforcement with blocking


class StreamingPlatform(str, Enum):
    """Supported streaming platforms."""

    KAFKA = "kafka"
    KINESIS = "kinesis"


class RemediationAction(str, Enum):
    """Auto-remediation action types."""

    DELETE = "delete"
    ANONYMIZE = "anonymize"
    ARCHIVE = "archive"
    NOTIFY_ONLY = "notify_only"


# ---------------------------------------------------------------------------
# Component Configurations
# ---------------------------------------------------------------------------


class VaultConfig(BaseSettings):
    """Configuration for the Secure Token Vault.

    The vault stores encrypted, reversible tokens for PII values using
    AES-256-GCM encryption via SEC-003 EncryptionService.

    Attributes:
        token_ttl_days: Days until tokens expire and become unrecoverable.
        max_tokens_per_tenant: Maximum tokens per tenant to prevent abuse.
        persistence_backend: Storage backend for token persistence.
        encryption_key_id: KMS key alias for envelope encryption.
        enable_persistence: Whether to persist tokens to storage.
        cache_ttl_seconds: In-memory cache TTL for token lookups.
        cache_max_size: Maximum entries in the token cache.
        require_user_auth: Require user authentication for detokenization.
    """

    token_ttl_days: int = Field(
        default=90,
        ge=1,
        le=3650,
        description="Days until tokens expire (1-3650)",
    )
    max_tokens_per_tenant: int = Field(
        default=1_000_000,
        ge=1000,
        le=100_000_000,
        description="Maximum tokens per tenant",
    )
    persistence_backend: PersistenceBackend = Field(
        default=PersistenceBackend.POSTGRESQL,
        description="Token storage backend",
    )
    encryption_key_id: str = Field(
        default="alias/greenlang-pii-vault",
        description="KMS key ID for token encryption",
    )
    enable_persistence: bool = Field(
        default=True,
        description="Persist tokens to storage",
    )
    cache_ttl_seconds: int = Field(
        default=300,
        ge=0,
        le=3600,
        description="In-memory cache TTL (0 to disable)",
    )
    cache_max_size: int = Field(
        default=10000,
        ge=100,
        le=1_000_000,
        description="Maximum cache entries",
    )
    require_user_auth: bool = Field(
        default=True,
        description="Require user auth for detokenization",
    )

    model_config = {
        "env_prefix": "GL_PII_VAULT_",
        "extra": "ignore",
    }


class EnforcementConfig(BaseSettings):
    """Configuration for the PII Enforcement Engine.

    Controls real-time enforcement actions when PII is detected.

    Attributes:
        mode: Operating mode (disabled/audit/warn/enforce).
        scan_requests: Scan incoming API request bodies.
        scan_responses: Scan outgoing API response bodies.
        scan_logs: Scan log messages before output.
        default_action: Default action when no policy matches.
        exclude_paths: API paths to exclude from scanning.
        exclude_content_types: Content types to skip.
        min_confidence: Minimum confidence for enforcement.
        block_high_sensitivity: Block high-sensitivity PII regardless of policy.
        quarantine_enabled: Enable quarantine for blocked content.
        quarantine_ttl_hours: Hours to retain quarantined content.
        notification_enabled: Send notifications on enforcement.
    """

    mode: EnforcementMode = Field(
        default=EnforcementMode.ENFORCE,
        description="Enforcement operating mode",
    )
    scan_requests: bool = Field(
        default=True,
        description="Scan incoming request bodies",
    )
    scan_responses: bool = Field(
        default=True,
        description="Scan outgoing response bodies",
    )
    scan_logs: bool = Field(
        default=True,
        description="Scan log messages",
    )
    default_action: str = Field(
        default="allow",
        description="Default action when no policy matches",
    )
    exclude_paths: List[str] = Field(
        default_factory=lambda: ["/health", "/metrics", "/ready", "/live"],
        description="API paths to exclude from scanning",
    )
    exclude_content_types: List[str] = Field(
        default_factory=lambda: ["image/", "audio/", "video/", "application/octet-stream"],
        description="Content types to skip",
    )
    min_confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for enforcement",
    )
    block_high_sensitivity: bool = Field(
        default=True,
        description="Always block high-sensitivity PII (SSN, credit card, password)",
    )
    quarantine_enabled: bool = Field(
        default=True,
        description="Enable quarantine for blocked content",
    )
    quarantine_ttl_hours: int = Field(
        default=72,
        ge=1,
        le=720,
        description="Hours to retain quarantined content",
    )
    notification_enabled: bool = Field(
        default=True,
        description="Send notifications on enforcement",
    )

    model_config = {
        "env_prefix": "GL_PII_ENFORCEMENT_",
        "extra": "ignore",
    }


class AllowlistConfig(BaseSettings):
    """Configuration for the PII Allowlist Manager.

    Manages patterns to exclude from PII detection (false positive filtering).

    Attributes:
        enable_defaults: Include default allowlist patterns.
        cache_ttl_seconds: Cache TTL for allowlist lookups.
        max_entries_per_tenant: Maximum allowlist entries per tenant.
        require_reason: Require a reason for new allowlist entries.
        expiration_days: Default expiration for new entries (0 = never).
    """

    enable_defaults: bool = Field(
        default=True,
        description="Include default allowlist patterns",
    )
    cache_ttl_seconds: int = Field(
        default=60,
        ge=0,
        le=3600,
        description="Cache TTL for allowlist lookups",
    )
    max_entries_per_tenant: int = Field(
        default=10000,
        ge=100,
        le=1_000_000,
        description="Maximum entries per tenant",
    )
    require_reason: bool = Field(
        default=True,
        description="Require reason for new entries",
    )
    expiration_days: int = Field(
        default=0,
        ge=0,
        le=3650,
        description="Default expiration days (0 = never)",
    )

    model_config = {
        "env_prefix": "GL_PII_ALLOWLIST_",
        "extra": "ignore",
    }


class StreamingConfig(BaseSettings):
    """Configuration for streaming platform integration.

    Supports real-time PII scanning on Kafka and Kinesis streams.

    Attributes:
        enabled: Enable streaming scanner.
        platform: Streaming platform (kafka/kinesis).
        bootstrap_servers: Kafka bootstrap servers (comma-separated).
        input_topics: Topics to consume for scanning.
        output_topic: Topic for clean (redacted) messages.
        dlq_topic: Dead letter queue for blocked messages.
        consumer_group: Kafka consumer group ID.
        kinesis_stream_name: Kinesis stream name.
        kinesis_region: AWS region for Kinesis.
        batch_size: Messages per batch.
        batch_timeout_ms: Maximum wait for batch.
        max_poll_records: Maximum records per poll.
    """

    enabled: bool = Field(
        default=False,
        description="Enable streaming scanner",
    )
    platform: StreamingPlatform = Field(
        default=StreamingPlatform.KAFKA,
        description="Streaming platform",
    )
    bootstrap_servers: str = Field(
        default="localhost:9092",
        description="Kafka bootstrap servers",
    )
    input_topics: List[str] = Field(
        default_factory=lambda: ["raw-events"],
        description="Topics to consume",
    )
    output_topic: str = Field(
        default="clean-events",
        description="Topic for clean messages",
    )
    dlq_topic: str = Field(
        default="pii-blocked",
        description="Dead letter queue topic",
    )
    consumer_group: str = Field(
        default="pii-scanner",
        description="Kafka consumer group",
    )
    kinesis_stream_name: str = Field(
        default="",
        description="Kinesis stream name",
    )
    kinesis_region: str = Field(
        default="us-east-1",
        description="AWS region for Kinesis",
    )
    batch_size: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Messages per batch",
    )
    batch_timeout_ms: int = Field(
        default=1000,
        ge=100,
        le=30000,
        description="Max wait for batch (ms)",
    )
    max_poll_records: int = Field(
        default=500,
        ge=1,
        le=10000,
        description="Max records per poll",
    )

    model_config = {
        "env_prefix": "GL_PII_STREAMING_",
        "extra": "ignore",
    }


class RemediationConfig(BaseSettings):
    """Configuration for automated PII remediation.

    Controls scheduled cleanup of detected PII from storage.

    Attributes:
        enabled: Enable auto-remediation.
        default_action: Default remediation action.
        delay_hours: Grace period before remediation.
        requires_approval: Require approval before action.
        notify_on_action: Send notifications on remediation.
        schedule_cron: Cron expression for scheduled runs.
        batch_size: Records per remediation batch.
        dry_run: Log actions without executing.
        generate_certificates: Generate deletion certificates.
    """

    enabled: bool = Field(
        default=True,
        description="Enable auto-remediation",
    )
    default_action: RemediationAction = Field(
        default=RemediationAction.NOTIFY_ONLY,
        description="Default remediation action",
    )
    delay_hours: int = Field(
        default=72,
        ge=1,
        le=720,
        description="Grace period before action",
    )
    requires_approval: bool = Field(
        default=True,
        description="Require approval before action",
    )
    notify_on_action: bool = Field(
        default=True,
        description="Send notifications on action",
    )
    schedule_cron: str = Field(
        default="0 2 * * *",
        description="Cron schedule (default: 2 AM daily)",
    )
    batch_size: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Records per batch",
    )
    dry_run: bool = Field(
        default=False,
        description="Log actions without executing",
    )
    generate_certificates: bool = Field(
        default=True,
        description="Generate deletion certificates",
    )

    model_config = {
        "env_prefix": "GL_PII_REMEDIATION_",
        "extra": "ignore",
    }


class ScannerConfig(BaseSettings):
    """Configuration for PII detection scanners.

    Controls regex and ML-based detection settings.

    Attributes:
        enable_regex: Enable regex-based detection.
        enable_ml: Enable ML-based detection (Presidio).
        min_confidence: Minimum confidence threshold.
        context_window: Characters of context to capture.
        max_content_size_mb: Maximum content size to scan.
        scan_timeout_seconds: Timeout for scan operations.
        parallel_scans: Enable parallel scanning.
        max_workers: Maximum parallel scan workers.
    """

    enable_regex: bool = Field(
        default=True,
        description="Enable regex-based detection",
    )
    enable_ml: bool = Field(
        default=True,
        description="Enable ML-based detection (Presidio)",
    )
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold",
    )
    context_window: int = Field(
        default=50,
        ge=0,
        le=200,
        description="Characters of context to capture",
    )
    max_content_size_mb: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum content size (MB)",
    )
    scan_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Scan timeout (seconds)",
    )
    parallel_scans: bool = Field(
        default=True,
        description="Enable parallel scanning",
    )
    max_workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Maximum parallel workers",
    )

    model_config = {
        "env_prefix": "GL_PII_SCANNER_",
        "extra": "ignore",
    }


# ---------------------------------------------------------------------------
# Main Configuration
# ---------------------------------------------------------------------------


class PIIServiceConfig(BaseSettings):
    """Main configuration for the PII Service.

    Aggregates all component configurations with environment-based overrides.

    Attributes:
        service_name: Service identifier for logging.
        enable_metrics: Emit Prometheus metrics.
        enable_audit: Log operations for audit trail.
        vault: Token vault configuration.
        enforcement: Enforcement engine configuration.
        allowlist: Allowlist manager configuration.
        streaming: Streaming scanner configuration.
        remediation: Auto-remediation configuration.
        scanner: Scanner configuration.
    """

    service_name: str = Field(
        default="pii_service",
        description="Service name for logging/metrics",
    )
    enable_metrics: bool = Field(
        default=True,
        description="Emit Prometheus metrics",
    )
    enable_audit: bool = Field(
        default=True,
        description="Log operations for audit",
    )

    # Component configurations
    vault: VaultConfig = Field(default_factory=VaultConfig)
    enforcement: EnforcementConfig = Field(default_factory=EnforcementConfig)
    allowlist: AllowlistConfig = Field(default_factory=AllowlistConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    remediation: RemediationConfig = Field(default_factory=RemediationConfig)
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)

    model_config = {
        "env_prefix": "GL_PII_SERVICE_",
        "extra": "ignore",
    }

    @classmethod
    def for_environment(cls, environment: str) -> PIIServiceConfig:
        """Create configuration tuned for a specific environment.

        Args:
            environment: Environment name (dev/staging/prod).

        Returns:
            PIIServiceConfig with environment-appropriate settings.
        """
        if environment == "dev":
            return cls(
                enable_audit=False,
                vault=VaultConfig(
                    persistence_backend=PersistenceBackend.MEMORY,
                    token_ttl_days=7,
                    max_tokens_per_tenant=10000,
                ),
                enforcement=EnforcementConfig(
                    mode=EnforcementMode.AUDIT,
                ),
                streaming=StreamingConfig(enabled=False),
                remediation=RemediationConfig(
                    enabled=False,
                    dry_run=True,
                ),
            )
        elif environment == "staging":
            return cls(
                vault=VaultConfig(
                    token_ttl_days=30,
                    max_tokens_per_tenant=100000,
                ),
                enforcement=EnforcementConfig(
                    mode=EnforcementMode.WARN,
                ),
                remediation=RemediationConfig(
                    requires_approval=True,
                    dry_run=True,
                ),
            )
        else:  # production
            return cls(
                vault=VaultConfig(
                    token_ttl_days=90,
                    max_tokens_per_tenant=1_000_000,
                    require_user_auth=True,
                ),
                enforcement=EnforcementConfig(
                    mode=EnforcementMode.ENFORCE,
                    block_high_sensitivity=True,
                ),
                remediation=RemediationConfig(
                    requires_approval=True,
                    generate_certificates=True,
                ),
            )


# ---------------------------------------------------------------------------
# Global Configuration Instance
# ---------------------------------------------------------------------------

_global_config: Optional[PIIServiceConfig] = None


def get_pii_service_config() -> PIIServiceConfig:
    """Get or create the global PII service configuration.

    Returns:
        The global PIIServiceConfig instance.
    """
    global _global_config

    if _global_config is None:
        _global_config = PIIServiceConfig()

    return _global_config


def configure_pii_service(config: PIIServiceConfig) -> None:
    """Set the global PII service configuration.

    Args:
        config: Configuration to use globally.
    """
    global _global_config
    _global_config = config
    logger.info(
        "PII service configured: enforcement=%s vault=%s",
        config.enforcement.mode.value,
        config.vault.persistence_backend.value,
    )


__all__ = [
    # Enums
    "PersistenceBackend",
    "EnforcementMode",
    "StreamingPlatform",
    "RemediationAction",
    # Component configs
    "VaultConfig",
    "EnforcementConfig",
    "AllowlistConfig",
    "StreamingConfig",
    "RemediationConfig",
    "ScannerConfig",
    # Main config
    "PIIServiceConfig",
    # Functions
    "get_pii_service_config",
    "configure_pii_service",
]
