# -*- coding: utf-8 -*-
"""
GreenLang Secrets Service - SEC-006: Secrets Management

Production-grade secrets management service wrapping the existing
VaultClient with additional features:

- Tenant isolation via TenantSecretContext
- Two-layer caching (Redis L1 + Memory L2)
- Integration with SecretsRotationManager
- REST API endpoints for secrets CRUD
- Factory methods for common credential types
- Audit logging and Prometheus metrics

Depends on:
    - greenlang.execution.infrastructure.secrets (VaultClient, SecretsRotationManager)
    - Redis for L1 caching
    - PostgreSQL for rotation state (via existing rotation manager)

Sub-modules:
    config           - SecretsServiceConfig dataclass
    secret_types     - SecretType enum, SecretMetadata, SecretReference
    tenant_context   - Tenant isolation and path building
    cache            - Two-layer secrets cache
    secrets_service  - Main SecretsService class
    metrics          - Prometheus metrics
    api              - FastAPI routers

Quick start:
    >>> from greenlang.infrastructure.secrets_service import (
    ...     SecretsService,
    ...     SecretsServiceConfig,
    ...     configure_secrets_service,
    ...     get_secrets_service,
    ... )
    >>> config = SecretsServiceConfig()
    >>> configure_secrets_service(config)
    >>> svc = get_secrets_service()
    >>> db_creds = await svc.get_database_credentials("main")

Re-exports from greenlang.execution.infrastructure.secrets:
    - VaultClient, VaultConfig, VaultSecret
    - SecretsRotationManager, RotationConfig, RotationResult
    - DatabaseCredentials, Certificate, AWSCredentials

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metrics (always available, no external dependencies beyond prometheus)
# ---------------------------------------------------------------------------

from greenlang.infrastructure.secrets_service.metrics import (  # noqa: E402
    SecretsMetrics,
    get_metrics,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from greenlang.infrastructure.secrets_service.config import (  # noqa: E402
    SecretsServiceConfig,
)

# ---------------------------------------------------------------------------
# Secret Types and Metadata
# ---------------------------------------------------------------------------

from greenlang.infrastructure.secrets_service.secret_types import (  # noqa: E402
    SecretType,
    SecretMetadata,
    SecretReference,
    SecretVersion,
)

# ---------------------------------------------------------------------------
# Tenant Context
# ---------------------------------------------------------------------------

from greenlang.infrastructure.secrets_service.tenant_context import (  # noqa: E402
    TenantSecretContext,
    TenantAccessDeniedError,
    InvalidSecretPathError,
    get_current_tenant,
    set_current_tenant,
    clear_tenant_context,
    build_tenant_secret_path,
    build_platform_secret_path,
)

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

from greenlang.infrastructure.secrets_service.cache import (  # noqa: E402
    SecretsCache,
    SecretsCacheConfig,
    MemoryCache,
    RedisSecretCache,
)

# ---------------------------------------------------------------------------
# Secrets Service
# ---------------------------------------------------------------------------

from greenlang.infrastructure.secrets_service.secrets_service import (  # noqa: E402
    SecretsService,
    SecretsServiceError,
    SecretNotFoundError,
    SecretAccessDeniedError,
    SecretValidationError,
    get_secrets_service,
    configure_secrets_service,
)

# ---------------------------------------------------------------------------
# API Router (may not be available if FastAPI not installed)
# ---------------------------------------------------------------------------

try:
    from greenlang.infrastructure.secrets_service.api import secrets_router
except ImportError:
    secrets_router = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Re-exports from existing secrets module
# ---------------------------------------------------------------------------

VAULT_AVAILABLE = False

try:
    from greenlang.execution.infrastructure.secrets import (
        VaultClient,
        VaultConfig,
        VaultAuthMethod,
        VaultSecret,
        VaultError,
        VaultAuthError,
        VaultSecretNotFoundError,
        VaultConnectionError,
        VaultPermissionError,
        DatabaseCredentials,
        AWSCredentials,
        Certificate,
        SecretsRotationManager,
        RotationConfig,
        RotationResult,
        RotationType,
    )

    VAULT_AVAILABLE = True
except ImportError as e:
    logger.warning("Vault client import failed: %s", e)

    # Define placeholders for type checking
    VaultClient = None  # type: ignore[misc, assignment]
    VaultConfig = None  # type: ignore[misc, assignment]
    VaultAuthMethod = None  # type: ignore[misc, assignment]
    VaultSecret = None  # type: ignore[misc, assignment]
    VaultError = Exception  # type: ignore[misc, assignment]
    VaultAuthError = Exception  # type: ignore[misc, assignment]
    VaultSecretNotFoundError = Exception  # type: ignore[misc, assignment]
    VaultConnectionError = Exception  # type: ignore[misc, assignment]
    VaultPermissionError = Exception  # type: ignore[misc, assignment]
    DatabaseCredentials = None  # type: ignore[misc, assignment]
    AWSCredentials = None  # type: ignore[misc, assignment]
    Certificate = None  # type: ignore[misc, assignment]
    SecretsRotationManager = None  # type: ignore[misc, assignment]
    RotationConfig = None  # type: ignore[misc, assignment]
    RotationResult = None  # type: ignore[misc, assignment]
    RotationType = None  # type: ignore[misc, assignment]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Metrics
    "SecretsMetrics",
    "get_metrics",
    # Configuration
    "SecretsServiceConfig",
    # Secret Types
    "SecretType",
    "SecretMetadata",
    "SecretReference",
    "SecretVersion",
    # Tenant Context
    "TenantSecretContext",
    "TenantAccessDeniedError",
    "InvalidSecretPathError",
    "get_current_tenant",
    "set_current_tenant",
    "clear_tenant_context",
    "build_tenant_secret_path",
    "build_platform_secret_path",
    # Cache
    "SecretsCache",
    "SecretsCacheConfig",
    "MemoryCache",
    "RedisSecretCache",
    # Service
    "SecretsService",
    "SecretsServiceError",
    "SecretNotFoundError",
    "SecretAccessDeniedError",
    "SecretValidationError",
    "get_secrets_service",
    "configure_secrets_service",
    # API Router
    "secrets_router",
    # Re-exports from Vault module
    "VaultClient",
    "VaultConfig",
    "VaultAuthMethod",
    "VaultSecret",
    "VaultError",
    "VaultAuthError",
    "VaultSecretNotFoundError",
    "VaultConnectionError",
    "VaultPermissionError",
    "DatabaseCredentials",
    "AWSCredentials",
    "Certificate",
    "SecretsRotationManager",
    "RotationConfig",
    "RotationResult",
    "RotationType",
    # Availability flag
    "VAULT_AVAILABLE",
]

__version__ = "1.0.0"
