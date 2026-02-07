# -*- coding: utf-8 -*-
"""
Secrets Service Configuration - SEC-006

Configuration dataclass for the Secrets Service that wraps VaultClient
with additional tenant isolation, caching, and API features.

Example:
    >>> from greenlang.infrastructure.secrets_service import SecretsServiceConfig
    >>> config = SecretsServiceConfig(
    ...     vault_addr="https://vault.greenlang.svc:8200",
    ...     auth_method="kubernetes",
    ...     cache_ttl_seconds=300,
    ... )

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SecretsServiceConfig:
    """Configuration for the Secrets Service (SEC-006).

    The Secrets Service wraps the existing VaultClient with additional
    concerns: tenant isolation, multi-layer caching (memory + Redis),
    rotation scheduling via SecretsRotationManager, and REST API endpoints.

    Attributes:
        vault_addr: Vault server address (VAULT_ADDR env var fallback).
        vault_namespace: Vault namespace for enterprise deployments.
        auth_method: Authentication method (kubernetes, token, approle).
        kubernetes_role: Vault role for Kubernetes auth.
        cache_enabled: Whether to enable secret caching.
        cache_ttl_seconds: Default TTL for cached secrets (seconds).
        redis_cache_enabled: Whether to use Redis as L1 cache.
        redis_key_prefix: Prefix for Redis cache keys.
        memory_cache_ttl_seconds: TTL for in-memory L2 cache.
        rotation_check_interval: How often to check for pending rotations (seconds).
        rotation_enabled: Whether automatic rotation is enabled.
        tenant_path_prefix: Path prefix for tenant-scoped secrets.
        platform_path_prefix: Path prefix for platform-wide secrets.
        max_versions_to_keep: Maximum secret versions to retain.
        audit_enabled: Whether to emit audit events for secret access.
        metrics_enabled: Whether to record Prometheus metrics.
    """

    # Vault connection
    vault_addr: str = field(
        default_factory=lambda: os.getenv(
            "VAULT_ADDR", "https://vault.vault.svc.cluster.local:8200"
        )
    )
    vault_namespace: str = field(
        default_factory=lambda: os.getenv("VAULT_NAMESPACE", "")
    )
    auth_method: str = field(
        default_factory=lambda: os.getenv("VAULT_AUTH_METHOD", "kubernetes")
    )
    kubernetes_role: str = field(
        default_factory=lambda: os.getenv("VAULT_ROLE", "greenlang-api")
    )

    # Token auth (if auth_method == "token")
    vault_token: Optional[str] = field(
        default_factory=lambda: os.getenv("VAULT_TOKEN")
    )

    # AppRole auth (if auth_method == "approle")
    approle_role_id: Optional[str] = field(
        default_factory=lambda: os.getenv("VAULT_ROLE_ID")
    )
    approle_secret_id: Optional[str] = field(
        default_factory=lambda: os.getenv("VAULT_SECRET_ID")
    )

    # TLS configuration
    ca_cert_path: Optional[str] = field(
        default_factory=lambda: os.getenv("VAULT_CACERT", "/vault/tls/ca.crt")
    )
    skip_verify: bool = field(
        default_factory=lambda: os.getenv("VAULT_SKIP_VERIFY", "false").lower() == "true"
    )

    # Caching configuration
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes default

    # Redis L1 cache
    redis_cache_enabled: bool = True
    redis_key_prefix: str = "gl:secrets"
    redis_url: Optional[str] = field(
        default_factory=lambda: os.getenv("REDIS_URL")
    )

    # Memory L2 cache
    memory_cache_enabled: bool = True
    memory_cache_ttl_seconds: int = 30  # Short TTL for memory cache
    memory_cache_max_size: int = 1000  # Maximum entries in memory cache

    # Rotation configuration
    rotation_enabled: bool = True
    rotation_check_interval: int = 3600  # 1 hour

    # Path prefixes (multi-tenant support)
    tenant_path_prefix: str = "secret/data/tenants"
    platform_path_prefix: str = "secret/data/greenlang"

    # Version management
    max_versions_to_keep: int = 10

    # Audit and observability
    audit_enabled: bool = True
    metrics_enabled: bool = True

    # Retry configuration (passed to VaultClient)
    max_retries: int = 3
    base_retry_delay: float = 0.5
    max_retry_delay: float = 30.0

    # Timeout configuration
    connect_timeout: float = 10.0
    read_timeout: float = 30.0

    def to_vault_config_kwargs(self) -> dict:
        """Convert to kwargs suitable for VaultConfig initialization.

        Returns:
            Dictionary of keyword arguments for VaultConfig.
        """
        from greenlang.execution.infrastructure.secrets import VaultAuthMethod

        auth_method_map = {
            "kubernetes": VaultAuthMethod.KUBERNETES,
            "token": VaultAuthMethod.TOKEN,
            "approle": VaultAuthMethod.APPROLE,
            "aws_iam": VaultAuthMethod.AWS_IAM,
        }

        return {
            "addr": self.vault_addr,
            "namespace": self.vault_namespace,
            "auth_method": auth_method_map.get(
                self.auth_method.lower(), VaultAuthMethod.KUBERNETES
            ),
            "kubernetes_role": self.kubernetes_role,
            "token": self.vault_token,
            "approle_role_id": self.approle_role_id,
            "approle_secret_id": self.approle_secret_id,
            "ca_cert_path": self.ca_cert_path,
            "skip_verify": self.skip_verify,
            "cache_enabled": False,  # We handle caching at service layer
            "max_retries": self.max_retries,
            "base_retry_delay": self.base_retry_delay,
            "max_retry_delay": self.max_retry_delay,
            "connect_timeout": self.connect_timeout,
            "read_timeout": self.read_timeout,
        }
