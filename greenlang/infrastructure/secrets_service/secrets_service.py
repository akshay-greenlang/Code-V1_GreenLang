# -*- coding: utf-8 -*-
"""
Secrets Service - SEC-006

High-level secrets management service wrapping VaultClient with:
- Tenant isolation via TenantSecretContext
- Two-layer caching (Redis L1 + Memory L2)
- Rotation integration via SecretsRotationManager
- Audit logging and metrics
- Factory methods for common credential types

Example:
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

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from greenlang.infrastructure.secrets_service.config import SecretsServiceConfig
from greenlang.infrastructure.secrets_service.secret_types import (
    SecretMetadata,
    SecretReference,
    SecretType,
    SecretVersion,
)
from greenlang.infrastructure.secrets_service.tenant_context import (
    TenantSecretContext,
    TenantAccessDeniedError,
    get_current_tenant,
)
from greenlang.infrastructure.secrets_service.cache import (
    SecretsCache,
    SecretsCacheConfig,
)

if TYPE_CHECKING:
    from greenlang.execution.infrastructure.secrets import (
        VaultClient,
        VaultSecret,
        SecretsRotationManager,
        DatabaseCredentials,
        Certificate,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Histogram

    SECRET_OPERATIONS = Counter(
        "secrets_operations_total",
        "Total secret operations",
        ["operation", "secret_type", "status"],
    )
    SECRET_LATENCY = Histogram(
        "secrets_operation_duration_seconds",
        "Duration of secret operations",
        ["operation"],
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    SECRET_OPERATIONS = None
    SECRET_LATENCY = None


def _record_operation(
    operation: str,
    secret_type: str = "generic",
    status: str = "success",
) -> None:
    """Record a secret operation metric."""
    if METRICS_AVAILABLE and SECRET_OPERATIONS:
        SECRET_OPERATIONS.labels(
            operation=operation,
            secret_type=secret_type,
            status=status,
        ).inc()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SecretsServiceError(Exception):
    """Base exception for Secrets Service errors."""

    pass


class SecretNotFoundError(SecretsServiceError):
    """Secret was not found at the specified path."""

    def __init__(self, path: str, version: Optional[int] = None):
        self.path = path
        self.version = version
        msg = f"Secret not found at path '{path}'"
        if version is not None:
            msg += f" (version {version})"
        super().__init__(msg)


class SecretAccessDeniedError(SecretsServiceError):
    """Access to the secret was denied."""

    def __init__(self, path: str, reason: str = "Access denied"):
        self.path = path
        self.reason = reason
        super().__init__(f"Access denied for secret '{path}': {reason}")


class SecretValidationError(SecretsServiceError):
    """Secret data failed validation."""

    pass


# ---------------------------------------------------------------------------
# Global Instance
# ---------------------------------------------------------------------------

_secrets_service: ContextVar[Optional["SecretsService"]] = ContextVar(
    "secrets_service", default=None
)


def get_secrets_service() -> "SecretsService":
    """Get the global SecretsService instance.

    Returns:
        Configured SecretsService.

    Raises:
        RuntimeError: If service not configured.
    """
    svc = _secrets_service.get()
    if svc is None:
        raise RuntimeError(
            "SecretsService not configured. Call configure_secrets_service() first."
        )
    return svc


def configure_secrets_service(
    config: Optional[SecretsServiceConfig] = None,
    vault_client: Optional["VaultClient"] = None,
    rotation_manager: Optional["SecretsRotationManager"] = None,
    redis_client: Any = None,
) -> "SecretsService":
    """Configure the global SecretsService instance.

    Args:
        config: Service configuration.
        vault_client: Optional pre-configured VaultClient.
        rotation_manager: Optional pre-configured rotation manager.
        redis_client: Optional Redis client for caching.

    Returns:
        Configured SecretsService instance.
    """
    svc = SecretsService(
        config=config,
        vault_client=vault_client,
        rotation_manager=rotation_manager,
        redis_client=redis_client,
    )
    _secrets_service.set(svc)
    logger.info(
        "SecretsService configured",
        extra={"event_category": "secrets"},
    )
    return svc


# ---------------------------------------------------------------------------
# Secrets Service
# ---------------------------------------------------------------------------


class SecretsService:
    """High-level secrets management service.

    Wraps VaultClient with additional features:
    - Tenant isolation and path building
    - Two-layer caching (Redis + Memory)
    - Integration with SecretsRotationManager
    - Audit logging for compliance
    - Factory methods for common credential types

    Example:
        >>> config = SecretsServiceConfig()
        >>> svc = SecretsService(config)
        >>> await svc.connect()
        >>>
        >>> # Get a tenant-scoped secret
        >>> secret = await svc.get_secret("database/main", tenant_id="acme")
        >>>
        >>> # Get database credentials
        >>> creds = await svc.get_database_credentials("readwrite")
        >>>
        >>> # Trigger rotation
        >>> result = await svc.trigger_rotation("api-keys/external")

    Attributes:
        config: Service configuration.
        vault_client: Underlying VaultClient.
        rotation_manager: Secrets rotation manager.
        cache: Two-layer secrets cache.
        tenant_context: Tenant isolation context.
    """

    def __init__(
        self,
        config: Optional[SecretsServiceConfig] = None,
        vault_client: Optional["VaultClient"] = None,
        rotation_manager: Optional["SecretsRotationManager"] = None,
        redis_client: Any = None,
    ):
        """Initialize the Secrets Service.

        Args:
            config: Service configuration.
            vault_client: Optional pre-configured VaultClient.
            rotation_manager: Optional pre-configured rotation manager.
            redis_client: Optional Redis client for caching.
        """
        self.config = config or SecretsServiceConfig()

        # Vault client (lazy initialized if not provided)
        self._vault_client = vault_client
        self._vault_client_owned = vault_client is None

        # Rotation manager
        self._rotation_manager = rotation_manager
        self._rotation_manager_owned = rotation_manager is None

        # Cache
        cache_config = SecretsCacheConfig(
            redis_enabled=self.config.redis_cache_enabled,
            redis_ttl_seconds=self.config.cache_ttl_seconds,
            redis_key_prefix=self.config.redis_key_prefix,
            memory_enabled=self.config.memory_cache_enabled,
            memory_ttl_seconds=self.config.memory_cache_ttl_seconds,
            memory_max_size=self.config.memory_cache_max_size,
        )
        self.cache = SecretsCache(redis_client=redis_client, config=cache_config)

        # Tenant context
        self.tenant_context = TenantSecretContext(
            tenant_path_prefix=self.config.tenant_path_prefix,
            platform_path_prefix=self.config.platform_path_prefix,
        )

        # State
        self._connected = False
        self._lock = asyncio.Lock()

    @property
    def vault_client(self) -> "VaultClient":
        """Get the VaultClient, initializing if needed.

        Returns:
            VaultClient instance.

        Raises:
            RuntimeError: If not connected.
        """
        if self._vault_client is None:
            raise RuntimeError("SecretsService not connected. Call connect() first.")
        return self._vault_client

    @property
    def rotation_manager(self) -> Optional["SecretsRotationManager"]:
        """Get the rotation manager if configured.

        Returns:
            SecretsRotationManager or None.
        """
        return self._rotation_manager

    async def connect(self) -> None:
        """Connect to Vault and initialize the service.

        Creates VaultClient and RotationManager if not provided.
        """
        async with self._lock:
            if self._connected:
                return

            # Initialize Vault client
            if self._vault_client is None:
                from greenlang.execution.infrastructure.secrets import (
                    VaultClient,
                    VaultConfig,
                )

                vault_config = VaultConfig(**self.config.to_vault_config_kwargs())
                self._vault_client = VaultClient(vault_config)

            # Connect to Vault
            await self._vault_client.connect()

            # Initialize rotation manager
            if self._rotation_manager is None and self.config.rotation_enabled:
                from greenlang.execution.infrastructure.secrets import (
                    SecretsRotationManager,
                    RotationConfig,
                )

                rotation_config = RotationConfig(
                    rotation_check_interval=asyncio.timedelta(
                        seconds=self.config.rotation_check_interval
                    ),
                )
                self._rotation_manager = SecretsRotationManager(
                    self._vault_client, rotation_config
                )
                # Note: Not starting rotation manager here - call start_rotation() if needed

            self._connected = True
            logger.info(
                "SecretsService connected to Vault",
                extra={
                    "event_category": "secrets",
                    "vault_addr": self.config.vault_addr,
                },
            )

    async def close(self) -> None:
        """Close connections and cleanup resources."""
        async with self._lock:
            if not self._connected:
                return

            # Stop rotation manager
            if self._rotation_manager and self._rotation_manager_owned:
                await self._rotation_manager.stop()

            # Close Vault client
            if self._vault_client and self._vault_client_owned:
                await self._vault_client.close()

            self._connected = False
            logger.info(
                "SecretsService closed",
                extra={"event_category": "secrets"},
            )

    async def __aenter__(self) -> "SecretsService":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    # ---------------------------------------------------------------------------
    # Core Secret Operations
    # ---------------------------------------------------------------------------

    async def get_secret(
        self,
        path: str,
        tenant_id: Optional[str] = None,
        version: Optional[int] = None,
        bypass_cache: bool = False,
        user_roles: Optional[Set[str]] = None,
    ) -> Optional["VaultSecret"]:
        """Get a secret from Vault.

        Args:
            path: Secret path (relative to tenant/platform prefix).
            tenant_id: Tenant ID (uses context if not provided).
            version: Specific version to retrieve.
            bypass_cache: Skip cache and fetch directly from Vault.
            user_roles: Roles for access validation.

        Returns:
            VaultSecret or None if not found.

        Raises:
            TenantAccessDeniedError: If cross-tenant access denied.
            SecretAccessDeniedError: If Vault permission denied.
        """
        # Determine effective tenant
        effective_tenant = tenant_id or get_current_tenant()

        # Build full path
        is_platform = self.tenant_context.is_platform_secret(path)
        if path.startswith(self.config.tenant_path_prefix) or path.startswith(
            self.config.platform_path_prefix
        ):
            # Already a full path
            full_path = path
        else:
            full_path = self.tenant_context.build_path(
                path,
                tenant_id=effective_tenant,
                is_platform=is_platform,
            )

        # Validate access
        self.tenant_context.validate_access(
            effective_tenant, full_path, user_roles
        )

        # Check cache first
        if not bypass_cache:
            cached = await self.cache.get(full_path, version)
            if cached is not None:
                _record_operation("get", "generic", "cache_hit")
                # Reconstruct VaultSecret from cached data
                from greenlang.execution.infrastructure.secrets import VaultSecret

                return VaultSecret(
                    data=cached.get("data", {}),
                    metadata=cached.get("metadata", {}),
                )

        # Fetch from Vault
        try:
            from greenlang.execution.infrastructure.secrets import (
                VaultSecretNotFoundError,
                VaultPermissionError,
            )

            secret = await self.vault_client.get_secret(full_path, version)

            # Cache the result
            if secret:
                cache_data = {
                    "data": secret.data,
                    "metadata": secret.metadata,
                }
                current_version = secret.metadata.get("version")
                await self.cache.set(full_path, cache_data, version=current_version)

                _record_operation("get", "generic", "success")
                self._emit_audit(
                    "secret_read",
                    path=full_path,
                    tenant_id=effective_tenant,
                    version=current_version,
                )

            return secret

        except VaultSecretNotFoundError:
            _record_operation("get", "generic", "not_found")
            return None

        except VaultPermissionError as e:
            _record_operation("get", "generic", "permission_denied")
            raise SecretAccessDeniedError(full_path, str(e)) from e

    async def put_secret(
        self,
        path: str,
        data: Dict[str, Any],
        tenant_id: Optional[str] = None,
        cas: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_roles: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Create or update a secret.

        Args:
            path: Secret path.
            data: Secret data.
            tenant_id: Tenant ID.
            cas: Check-and-set version for optimistic locking.
            metadata: Custom metadata to store.
            user_roles: Roles for access validation.

        Returns:
            Vault response metadata.

        Raises:
            TenantAccessDeniedError: If cross-tenant access denied.
        """
        # Determine effective tenant
        effective_tenant = tenant_id or get_current_tenant()

        # Build full path
        is_platform = self.tenant_context.is_platform_secret(path)
        if path.startswith(self.config.tenant_path_prefix) or path.startswith(
            self.config.platform_path_prefix
        ):
            full_path = path
        else:
            full_path = self.tenant_context.build_path(
                path,
                tenant_id=effective_tenant,
                is_platform=is_platform,
            )

        # Validate access
        self.tenant_context.validate_access(
            effective_tenant, full_path, user_roles
        )

        # Write to Vault
        result = await self.vault_client.put_secret(full_path, data, cas=cas)

        # Invalidate cache for this path
        await self.cache.delete(full_path)

        _record_operation("put", SecretType.from_path(path).value, "success")
        self._emit_audit(
            "secret_write",
            path=full_path,
            tenant_id=effective_tenant,
            version=result.get("version"),
        )

        return result

    async def delete_secret(
        self,
        path: str,
        tenant_id: Optional[str] = None,
        versions: Optional[List[int]] = None,
        user_roles: Optional[Set[str]] = None,
    ) -> None:
        """Delete a secret (soft delete).

        Args:
            path: Secret path.
            tenant_id: Tenant ID.
            versions: Specific versions to delete.
            user_roles: Roles for access validation.
        """
        effective_tenant = tenant_id or get_current_tenant()

        # Build full path
        is_platform = self.tenant_context.is_platform_secret(path)
        if path.startswith(self.config.tenant_path_prefix) or path.startswith(
            self.config.platform_path_prefix
        ):
            full_path = path
        else:
            full_path = self.tenant_context.build_path(
                path,
                tenant_id=effective_tenant,
                is_platform=is_platform,
            )

        # Validate access
        self.tenant_context.validate_access(
            effective_tenant, full_path, user_roles
        )

        # Delete from Vault
        await self.vault_client.delete_secret(full_path, versions=versions)

        # Invalidate cache
        await self.cache.delete(full_path)

        _record_operation("delete", "generic", "success")
        self._emit_audit(
            "secret_delete",
            path=full_path,
            tenant_id=effective_tenant,
        )

    async def list_secrets(
        self,
        prefix: str,
        tenant_id: Optional[str] = None,
        user_roles: Optional[Set[str]] = None,
    ) -> List[str]:
        """List secrets under a prefix.

        Args:
            prefix: Path prefix to list.
            tenant_id: Tenant ID.
            user_roles: Roles for access validation.

        Returns:
            List of secret paths (relative to prefix).
        """
        effective_tenant = tenant_id or get_current_tenant()

        # Build full path
        is_platform = self.tenant_context.is_platform_secret(prefix)
        if prefix.startswith(self.config.tenant_path_prefix) or prefix.startswith(
            self.config.platform_path_prefix
        ):
            full_prefix = prefix
        else:
            full_prefix = self.tenant_context.build_path(
                prefix,
                tenant_id=effective_tenant,
                is_platform=is_platform,
            )

        # Validate access
        self.tenant_context.validate_access(
            effective_tenant, full_prefix, user_roles
        )

        # List from Vault
        # Note: VaultClient doesn't have a list method in the provided code
        # This would need to be added or use direct API call
        try:
            list_path = full_prefix.replace("secret/data/", "secret/metadata/")
            response = await self.vault_client._request(
                "LIST", f"/v1/{list_path}"
            )
            keys = response.get("data", {}).get("keys", [])
            _record_operation("list", "generic", "success")
            return keys
        except Exception as e:
            logger.warning(
                "Failed to list secrets at %s: %s",
                full_prefix,
                str(e),
                extra={"event_category": "secrets"},
            )
            return []

    async def get_secret_versions(
        self,
        path: str,
        tenant_id: Optional[str] = None,
        user_roles: Optional[Set[str]] = None,
    ) -> List[SecretVersion]:
        """Get version history for a secret.

        Args:
            path: Secret path.
            tenant_id: Tenant ID.
            user_roles: Roles for access validation.

        Returns:
            List of SecretVersion objects.
        """
        effective_tenant = tenant_id or get_current_tenant()

        # Build full path
        is_platform = self.tenant_context.is_platform_secret(path)
        if path.startswith(self.config.tenant_path_prefix) or path.startswith(
            self.config.platform_path_prefix
        ):
            full_path = path
        else:
            full_path = self.tenant_context.build_path(
                path,
                tenant_id=effective_tenant,
                is_platform=is_platform,
            )

        # Validate access
        self.tenant_context.validate_access(
            effective_tenant, full_path, user_roles
        )

        # Get metadata from Vault
        try:
            metadata_path = full_path.replace("secret/data/", "secret/metadata/")
            response = await self.vault_client._request(
                "GET", f"/v1/{metadata_path}"
            )
            versions_data = response.get("data", {}).get("versions", {})

            versions = []
            for version_num, version_info in versions_data.items():
                version_info["version"] = int(version_num)
                versions.append(SecretVersion.from_vault_version(version_info))

            # Sort by version descending
            versions.sort(key=lambda v: v.version, reverse=True)

            _record_operation("get_versions", "generic", "success")
            return versions

        except Exception as e:
            logger.warning(
                "Failed to get versions for %s: %s",
                full_path,
                str(e),
                extra={"event_category": "secrets"},
            )
            return []

    async def undelete_version(
        self,
        path: str,
        version: int,
        tenant_id: Optional[str] = None,
        user_roles: Optional[Set[str]] = None,
    ) -> bool:
        """Undelete a soft-deleted secret version.

        Args:
            path: Secret path.
            version: Version to restore.
            tenant_id: Tenant ID.
            user_roles: Roles for access validation.

        Returns:
            True if successful.
        """
        effective_tenant = tenant_id or get_current_tenant()

        # Build full path
        is_platform = self.tenant_context.is_platform_secret(path)
        if path.startswith(self.config.tenant_path_prefix) or path.startswith(
            self.config.platform_path_prefix
        ):
            full_path = path
        else:
            full_path = self.tenant_context.build_path(
                path,
                tenant_id=effective_tenant,
                is_platform=is_platform,
            )

        # Validate access
        self.tenant_context.validate_access(
            effective_tenant, full_path, user_roles
        )

        # Undelete in Vault
        try:
            undelete_path = full_path.replace("secret/data/", "secret/undelete/")
            await self.vault_client._request(
                "POST",
                f"/v1/{undelete_path}",
                json={"versions": [version]},
            )

            # Invalidate cache
            await self.cache.delete(full_path)

            _record_operation("undelete", "generic", "success")
            self._emit_audit(
                "secret_undelete",
                path=full_path,
                tenant_id=effective_tenant,
                version=version,
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to undelete version %d for %s: %s",
                version,
                full_path,
                str(e),
                extra={"event_category": "secrets"},
            )
            return False

    # ---------------------------------------------------------------------------
    # Factory Methods for Common Credential Types
    # ---------------------------------------------------------------------------

    async def get_database_credentials(
        self,
        role: str,
        tenant_id: Optional[str] = None,
    ) -> "DatabaseCredentials":
        """Get dynamic database credentials from Vault.

        Args:
            role: Database role name.
            tenant_id: Tenant ID for context.

        Returns:
            DatabaseCredentials with username, password, connection info.
        """
        # Database credentials come from the database secrets engine
        # These are dynamic and not tenant-scoped in the same way
        creds = await self.vault_client.get_database_credentials(role)

        _record_operation("get_database_creds", "database", "success")
        self._emit_audit(
            "database_credentials_issued",
            path=f"database/creds/{role}",
            tenant_id=tenant_id or get_current_tenant(),
        )

        return creds

    async def get_api_key(
        self,
        name: str,
        tenant_id: Optional[str] = None,
    ) -> Optional[str]:
        """Get an API key from Vault.

        Args:
            name: API key name.
            tenant_id: Tenant ID.

        Returns:
            API key value or None.
        """
        secret = await self.get_secret(
            f"api-keys/{name}",
            tenant_id=tenant_id,
        )

        if secret is None:
            return None

        _record_operation("get_api_key", "api_key", "success")
        return secret.data.get("key") or secret.data.get("api_key")

    async def get_certificate(
        self,
        role: str,
        common_name: str,
        alt_names: Optional[List[str]] = None,
        ttl: str = "168h",
    ) -> "Certificate":
        """Generate a TLS certificate from Vault PKI.

        Args:
            role: PKI role name.
            common_name: Certificate common name.
            alt_names: Subject alternative names.
            ttl: Certificate TTL.

        Returns:
            Certificate with cert, key, and CA chain.
        """
        cert = await self.vault_client.generate_certificate(
            role=role,
            common_name=common_name,
            alt_names=alt_names,
            ttl=ttl,
        )

        _record_operation("get_certificate", "certificate", "success")
        self._emit_audit(
            "certificate_issued",
            path=f"pki_int/issue/{role}",
            tenant_id=get_current_tenant(),
            metadata={"common_name": common_name},
        )

        return cert

    async def get_encryption_key(
        self,
        key_name: str,
        tenant_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get encryption key metadata from Vault Transit.

        Note: Does not return the actual key material (Vault Transit
        encrypts/decrypts without exposing keys).

        Args:
            key_name: Transit key name.
            tenant_id: Tenant ID for logging.

        Returns:
            Key metadata or None.
        """
        try:
            response = await self.vault_client._request(
                "GET", f"/v1/transit/keys/{key_name}"
            )

            _record_operation("get_encryption_key", "encryption_key", "success")
            self._emit_audit(
                "encryption_key_accessed",
                path=f"transit/keys/{key_name}",
                tenant_id=tenant_id or get_current_tenant(),
            )

            return response.get("data", {})

        except Exception:
            return None

    # ---------------------------------------------------------------------------
    # Rotation Integration
    # ---------------------------------------------------------------------------

    async def trigger_rotation(
        self,
        path: str,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Trigger manual rotation for a secret.

        Args:
            path: Secret path to rotate.
            tenant_id: Tenant ID.

        Returns:
            Rotation result.
        """
        if not self._rotation_manager:
            raise SecretsServiceError("Rotation manager not configured")

        effective_tenant = tenant_id or get_current_tenant()

        # Build full path
        full_path = self.tenant_context.build_path(
            path, tenant_id=effective_tenant
        )

        # Determine rotation type from path
        from greenlang.execution.infrastructure.secrets import RotationType

        secret_type = SecretType.from_path(path)
        rotation_type_map = {
            SecretType.DATABASE: RotationType.DATABASE_CREDENTIAL,
            SecretType.API_KEY: RotationType.API_KEY,
            SecretType.CERTIFICATE: RotationType.CERTIFICATE,
            SecretType.ENCRYPTION_KEY: RotationType.ENCRYPTION_KEY,
        }

        # Trigger rotation based on type
        rotation_type = rotation_type_map.get(
            secret_type, RotationType.STATIC_SECRET
        )

        if rotation_type == RotationType.DATABASE_CREDENTIAL:
            result = await self._rotation_manager.rotate_database_credentials(path)
        elif rotation_type == RotationType.API_KEY:
            result = await self._rotation_manager.rotate_api_keys(full_path)
        elif rotation_type == RotationType.CERTIFICATE:
            result = await self._rotation_manager.rotate_certificate(path)
        elif rotation_type == RotationType.ENCRYPTION_KEY:
            result = await self._rotation_manager.rotate_encryption_key(path)
        else:
            # For static secrets, trigger generic rotation
            result = await self._rotation_manager.rotate_api_keys(full_path)

        # Invalidate cache
        await self.cache.delete(full_path)

        _record_operation("rotate", secret_type.value, result.status.value)
        self._emit_audit(
            "secret_rotated",
            path=full_path,
            tenant_id=effective_tenant,
            metadata={"rotation_status": result.status.value},
        )

        return result.to_audit_log()

    def get_rotation_status(self) -> Dict[str, Any]:
        """Get current rotation status for all managed secrets.

        Returns:
            Dictionary of rotation schedules and status.
        """
        if not self._rotation_manager:
            return {"enabled": False, "schedules": {}}

        return {
            "enabled": True,
            "schedules": self._rotation_manager.get_rotation_status(),
        }

    def get_rotation_schedule(self) -> Dict[str, Any]:
        """Get the rotation schedule for all secrets.

        Returns:
            Dictionary of next rotation times.
        """
        if not self._rotation_manager:
            return {"enabled": False, "schedule": []}

        status = self._rotation_manager.get_rotation_status()
        schedule = []

        for key, info in status.items():
            if info.get("next_rotation"):
                schedule.append({
                    "path": info["identifier"],
                    "type": info["type"],
                    "next_rotation": info["next_rotation"],
                    "last_rotation": info.get("last_rotation"),
                })

        # Sort by next rotation
        schedule.sort(key=lambda x: x.get("next_rotation") or "")

        return {"enabled": True, "schedule": schedule}

    async def start_rotation(self) -> None:
        """Start the automatic rotation manager."""
        if self._rotation_manager:
            await self._rotation_manager.start()
            logger.info(
                "Secrets rotation manager started",
                extra={"event_category": "secrets"},
            )

    async def stop_rotation(self) -> None:
        """Stop the automatic rotation manager."""
        if self._rotation_manager:
            await self._rotation_manager.stop()
            logger.info(
                "Secrets rotation manager stopped",
                extra={"event_category": "secrets"},
            )

    # ---------------------------------------------------------------------------
    # Health and Status
    # ---------------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        """Check health of the secrets service.

        Returns:
            Health status including Vault connectivity.
        """
        vault_healthy = False
        vault_status = {}

        try:
            vault_status = await self.vault_client.health_check()
            vault_healthy = await self.vault_client.is_healthy()
        except Exception as e:
            vault_status = {"error": str(e)}

        rotation_healthy = True
        rotation_status = {"enabled": False}

        if self._rotation_manager:
            rotation_status = await self._rotation_manager.health_check()
            rotation_healthy = rotation_status.get("healthy", False)

        return {
            "healthy": vault_healthy and rotation_healthy,
            "vault": {
                "healthy": vault_healthy,
                "status": vault_status,
            },
            "rotation": rotation_status,
            "cache": self.cache.stats,
            "connected": self._connected,
        }

    async def get_stats(self) -> Dict[str, Any]:
        """Get operation statistics.

        Returns:
            Statistics including cache metrics.
        """
        return {
            "cache": self.cache.stats,
            "connected": self._connected,
            "rotation_enabled": self._rotation_manager is not None,
        }

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    def create_reference(
        self,
        path: str,
        tenant_id: Optional[str] = None,
        version: Optional[int] = None,
    ) -> SecretReference:
        """Create a lazy secret reference.

        Args:
            path: Secret path.
            tenant_id: Tenant ID.
            version: Specific version.

        Returns:
            SecretReference bound to this service.
        """
        return SecretReference(
            path=path,
            tenant_id=tenant_id,
            version=version,
        ).bind(self)

    def _emit_audit(
        self,
        action: str,
        path: str,
        tenant_id: Optional[str] = None,
        version: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit an audit log event.

        Args:
            action: Action performed.
            path: Secret path.
            tenant_id: Tenant ID.
            version: Secret version.
            metadata: Additional metadata.
        """
        if not self.config.audit_enabled:
            return

        logger.info(
            "Secrets audit: %s",
            action,
            extra={
                "event_category": "secrets",
                "action": action,
                "path": path,
                "tenant_id": tenant_id,
                "version": version,
                "metadata": metadata or {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
