# -*- coding: utf-8 -*-
"""
Secret Types and Metadata Models - SEC-006

Defines the SecretType enum, SecretMetadata dataclass, and SecretReference
for lazy loading of secrets in the Secrets Service.

Example:
    >>> from greenlang.infrastructure.secrets_service import SecretType, SecretMetadata
    >>> metadata = SecretMetadata(
    ...     path="tenants/acme-corp/database",
    ...     secret_type=SecretType.DATABASE,
    ...     version=3,
    ...     tenant_id="acme-corp",
    ... )

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from greenlang.infrastructure.secrets_service.secrets_service import SecretsService


class SecretType(str, Enum):
    """Types of secrets managed by the Secrets Service.

    Used for categorization, rotation scheduling, and audit logging.

    Members:
        DATABASE: Database credentials (username, password, connection strings).
        API_KEY: API keys and tokens for external service integrations.
        CERTIFICATE: TLS certificates and private keys.
        ENCRYPTION_KEY: Symmetric encryption keys for data protection.
        SERVICE_TOKEN: Internal service-to-service authentication tokens.
        OAUTH_CLIENT: OAuth2 client credentials (client_id, client_secret).
        SSH_KEY: SSH key pairs for infrastructure access.
        SIGNING_KEY: Asymmetric keys for signing operations.
        GENERIC: Generic key-value secrets not fitting other categories.
    """

    DATABASE = "database"
    API_KEY = "api_key"
    CERTIFICATE = "certificate"
    ENCRYPTION_KEY = "encryption_key"
    SERVICE_TOKEN = "service_token"
    OAUTH_CLIENT = "oauth_client"
    SSH_KEY = "ssh_key"
    SIGNING_KEY = "signing_key"
    GENERIC = "generic"

    @classmethod
    def from_path(cls, path: str) -> "SecretType":
        """Infer secret type from the path naming convention.

        Args:
            path: Secret path (e.g., "tenants/acme/database/main").

        Returns:
            Inferred SecretType.
        """
        path_lower = path.lower()

        if "database" in path_lower or "db" in path_lower:
            return cls.DATABASE
        if "api-key" in path_lower or "apikey" in path_lower:
            return cls.API_KEY
        if "certificate" in path_lower or "cert" in path_lower or "tls" in path_lower:
            return cls.CERTIFICATE
        if "encryption" in path_lower or "encrypt" in path_lower:
            return cls.ENCRYPTION_KEY
        if "service-token" in path_lower or "svc-token" in path_lower:
            return cls.SERVICE_TOKEN
        if "oauth" in path_lower:
            return cls.OAUTH_CLIENT
        if "ssh" in path_lower:
            return cls.SSH_KEY
        if "signing" in path_lower or "sign-key" in path_lower:
            return cls.SIGNING_KEY

        return cls.GENERIC


@dataclass
class SecretMetadata:
    """Metadata about a secret stored in Vault.

    Contains information about the secret's path, type, version, and
    lifecycle without exposing the actual secret data.

    Attributes:
        path: Full path to the secret in Vault (excluding mount point).
        secret_type: Categorization of the secret.
        version: Current version number of the secret.
        created_at: When this version was created.
        updated_at: When the secret was last modified.
        expires_at: When the secret expires (None if no expiry).
        tenant_id: Tenant ID for tenant-scoped secrets (None for platform).
        is_platform_secret: Whether this is a platform-wide secret.
        created_by: User or service that created this version.
        rotation_schedule: Cron expression for automatic rotation.
        last_rotated_at: When the secret was last rotated.
        next_rotation_at: When the next rotation is scheduled.
        tags: Key-value tags for categorization and filtering.
        checksum: SHA-256 checksum of the secret data for integrity.
    """

    path: str
    secret_type: SecretType = SecretType.GENERIC
    version: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    tenant_id: Optional[str] = None
    is_platform_secret: bool = False
    created_by: Optional[str] = None
    rotation_schedule: Optional[str] = None
    last_rotated_at: Optional[datetime] = None
    next_rotation_at: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)
    checksum: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        """Check if the secret has expired.

        Returns:
            True if expires_at is set and in the past.
        """
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) >= self.expires_at

    @property
    def is_rotation_due(self) -> bool:
        """Check if rotation is due based on schedule.

        Returns:
            True if next_rotation_at is set and in the past.
        """
        if self.next_rotation_at is None:
            return False
        return datetime.now(timezone.utc) >= self.next_rotation_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary representation.

        Returns:
            Dictionary with all metadata fields serialized.
        """
        return {
            "path": self.path,
            "secret_type": self.secret_type.value,
            "version": self.version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "tenant_id": self.tenant_id,
            "is_platform_secret": self.is_platform_secret,
            "created_by": self.created_by,
            "rotation_schedule": self.rotation_schedule,
            "last_rotated_at": (
                self.last_rotated_at.isoformat() if self.last_rotated_at else None
            ),
            "next_rotation_at": (
                self.next_rotation_at.isoformat() if self.next_rotation_at else None
            ),
            "tags": self.tags,
            "checksum": self.checksum,
            "is_expired": self.is_expired,
            "is_rotation_due": self.is_rotation_due,
        }

    @classmethod
    def from_vault_metadata(
        cls,
        path: str,
        vault_metadata: Dict[str, Any],
        tenant_id: Optional[str] = None,
    ) -> "SecretMetadata":
        """Create SecretMetadata from Vault response metadata.

        Args:
            path: Secret path.
            vault_metadata: Metadata dict from VaultSecret.metadata.
            tenant_id: Optional tenant ID override.

        Returns:
            Populated SecretMetadata instance.
        """
        created_time = vault_metadata.get("created_time")
        created_at = datetime.now(timezone.utc)
        if created_time:
            try:
                # Vault uses RFC 3339 format
                created_at = datetime.fromisoformat(
                    created_time.replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                pass

        custom_metadata = vault_metadata.get("custom_metadata", {}) or {}

        return cls(
            path=path,
            secret_type=SecretType.from_path(path),
            version=vault_metadata.get("version", 1),
            created_at=created_at,
            tenant_id=tenant_id or custom_metadata.get("tenant_id"),
            is_platform_secret=custom_metadata.get("is_platform", False),
            created_by=custom_metadata.get("created_by"),
            rotation_schedule=custom_metadata.get("rotation_schedule"),
            tags=custom_metadata.get("tags", {}),
        )


@dataclass
class SecretReference:
    """A lazy reference to a secret that fetches on first access.

    Useful for dependency injection where the secret value should not
    be loaded until actually needed, reducing unnecessary Vault calls.

    Attributes:
        path: Secret path in Vault.
        tenant_id: Tenant context for the secret.
        version: Specific version to fetch (None for latest).
        _cached_value: Internal cache of fetched secret data.
        _fetched: Whether the secret has been fetched.
    """

    path: str
    tenant_id: Optional[str] = None
    version: Optional[int] = None
    _cached_value: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _fetched: bool = field(default=False, repr=False)
    _service: Optional["SecretsService"] = field(default=None, repr=False)
    _fetch_callback: Optional[Callable[[], Dict[str, Any]]] = field(
        default=None, repr=False
    )

    def bind(self, service: "SecretsService") -> "SecretReference":
        """Bind a SecretsService instance for fetching.

        Args:
            service: SecretsService to use for fetching.

        Returns:
            Self for method chaining.
        """
        self._service = service
        return self

    def bind_callback(
        self, callback: Callable[[], Dict[str, Any]]
    ) -> "SecretReference":
        """Bind a custom fetch callback.

        Args:
            callback: Async callable that returns secret data.

        Returns:
            Self for method chaining.
        """
        self._fetch_callback = callback
        return self

    async def fetch(self) -> Dict[str, Any]:
        """Fetch the secret data, using cache if available.

        Returns:
            Secret data dictionary.

        Raises:
            RuntimeError: If no service or callback is bound.
        """
        if self._fetched and self._cached_value is not None:
            return self._cached_value

        if self._fetch_callback is not None:
            self._cached_value = self._fetch_callback()
            self._fetched = True
            return self._cached_value

        if self._service is None:
            raise RuntimeError(
                "SecretReference is not bound to a SecretsService. "
                "Call bind() or bind_callback() first."
            )

        secret = await self._service.get_secret(
            self.path, tenant_id=self.tenant_id, version=self.version
        )
        self._cached_value = secret.data if secret else {}
        self._fetched = True
        return self._cached_value

    async def get(self, key: str, default: Any = None) -> Any:
        """Fetch and get a specific key from the secret.

        Args:
            key: Key to retrieve from secret data.
            default: Default value if key not found.

        Returns:
            Value for the key or default.
        """
        data = await self.fetch()
        return data.get(key, default)

    def invalidate(self) -> None:
        """Invalidate the cached value, forcing re-fetch on next access."""
        self._cached_value = None
        self._fetched = False

    @staticmethod
    def compute_checksum(data: Dict[str, Any]) -> str:
        """Compute SHA-256 checksum for secret data.

        Args:
            data: Secret data dictionary.

        Returns:
            Hex-encoded SHA-256 checksum.
        """
        import json

        # Serialize deterministically
        serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass
class SecretVersion:
    """Represents a specific version of a secret.

    Attributes:
        version: Version number.
        created_at: When this version was created.
        destroyed: Whether this version has been destroyed.
        deletion_time: When this version was deleted (soft delete).
    """

    version: int
    created_at: datetime
    destroyed: bool = False
    deletion_time: Optional[datetime] = None

    @property
    def is_available(self) -> bool:
        """Check if this version is available for retrieval.

        Returns:
            True if not destroyed and not deleted.
        """
        if self.destroyed:
            return False
        if self.deletion_time and datetime.now(timezone.utc) >= self.deletion_time:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with version information.
        """
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "destroyed": self.destroyed,
            "deletion_time": (
                self.deletion_time.isoformat() if self.deletion_time else None
            ),
            "is_available": self.is_available,
        }

    @classmethod
    def from_vault_version(cls, version_data: Dict[str, Any]) -> "SecretVersion":
        """Create from Vault version metadata.

        Args:
            version_data: Version dict from Vault API.

        Returns:
            SecretVersion instance.
        """
        created_time = version_data.get("created_time")
        created_at = datetime.now(timezone.utc)
        if created_time:
            try:
                created_at = datetime.fromisoformat(
                    created_time.replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                pass

        deletion_time = version_data.get("deletion_time")
        deletion_dt = None
        if deletion_time:
            try:
                deletion_dt = datetime.fromisoformat(
                    deletion_time.replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                pass

        return cls(
            version=version_data.get("version", 0),
            created_at=created_at,
            destroyed=version_data.get("destroyed", False),
            deletion_time=deletion_dt,
        )
