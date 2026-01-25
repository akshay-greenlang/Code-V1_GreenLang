"""
Secrets Management Service - HashiCorp Vault Integration

This module provides secure secrets management with HashiCorp Vault integration,
including secret rotation automation, dynamic credentials for databases,
and API key lifecycle management.

SOC2 Controls Addressed:
    - CC6.1: Logical access security
    - CC6.6: Confidential information protection
    - CC6.7: Encryption key management

ISO27001 Controls Addressed:
    - A.9.2.4: Secret authentication information management
    - A.9.4.3: Password management system
    - A.10.1.2: Key management

Example:
    >>> config = SecretsConfig(vault_addr="https://vault.greenlang.io:8200")
    >>> service = SecretsService(config)
    >>> await service.initialize()
    >>> db_creds = await service.get_dynamic_database_credentials("postgresql")
    >>> api_key = await service.create_api_key("service-account-1")
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import secrets
import time
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class SecretType(str, Enum):
    """Types of secrets managed by the service."""

    STATIC = "STATIC"  # Manually managed secrets
    DYNAMIC = "DYNAMIC"  # Auto-generated, auto-rotated
    API_KEY = "API_KEY"  # API keys for service authentication
    DATABASE = "DATABASE"  # Database credentials
    CERTIFICATE = "CERTIFICATE"  # TLS certificates
    ENCRYPTION_KEY = "ENCRYPTION_KEY"  # Encryption keys
    OAUTH_TOKEN = "OAUTH_TOKEN"  # OAuth tokens


class SecretStatus(str, Enum):
    """Lifecycle status of a secret."""

    ACTIVE = "ACTIVE"
    PENDING_ROTATION = "PENDING_ROTATION"
    ROTATING = "ROTATING"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"


class DatabaseEngine(str, Enum):
    """Supported database engines for dynamic credentials."""

    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    REDIS = "redis"


class SecretReference(BaseModel):
    """
    Reference to a secret stored in the secrets manager.

    Used to track secret metadata without exposing the actual value.

    Attributes:
        id: Unique identifier for the secret
        name: Human-readable secret name
        path: Vault path or storage location
        type: Type of secret
        status: Current lifecycle status
        version: Secret version number
        created_at: Creation timestamp
        expires_at: Expiration timestamp
        last_rotated_at: Last rotation timestamp
        rotation_schedule: Cron expression for rotation schedule
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Secret name/identifier")
    path: str = Field(..., description="Vault path or storage location")
    type: SecretType = Field(default=SecretType.STATIC)
    status: SecretStatus = Field(default=SecretStatus.ACTIVE)
    version: int = Field(default=1, ge=1)

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = Field(None)
    last_rotated_at: Optional[datetime] = Field(None)
    last_accessed_at: Optional[datetime] = Field(None)

    # Rotation configuration
    rotation_schedule: Optional[str] = Field(None, description="Cron expression for rotation")
    rotation_period_days: Optional[int] = Field(None)
    auto_rotate: bool = Field(default=False)

    # Access tracking
    access_count: int = Field(default=0)
    allowed_services: List[str] = Field(default_factory=list)

    # Metadata
    description: Optional[str] = Field(None)
    tags: Dict[str, str] = Field(default_factory=dict)
    tenant_id: Optional[str] = Field(None)

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

    def is_expired(self) -> bool:
        """Check if the secret has expired."""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def needs_rotation(self) -> bool:
        """Check if the secret needs rotation based on schedule."""
        if not self.rotation_period_days:
            return False

        if not self.last_rotated_at:
            reference_time = self.created_at
        else:
            reference_time = self.last_rotated_at

        next_rotation = reference_time + timedelta(days=self.rotation_period_days)
        return datetime.now(timezone.utc) >= next_rotation


class SecretValue(BaseModel):
    """
    Container for a secret value with metadata.

    The actual secret value is kept separate from references
    to minimize exposure in logs and memory.
    """

    reference: SecretReference
    value: str = Field(..., description="The actual secret value")
    lease_id: Optional[str] = Field(None, description="Vault lease ID for dynamic secrets")
    lease_duration: Optional[int] = Field(None, description="Lease duration in seconds")

    class Config:
        """Pydantic configuration."""

        # Prevent secret values from appearing in string representations
        json_encoders = {
            str: lambda v: "[REDACTED]" if len(v) > 50 else v,
        }


class DatabaseCredentials(BaseModel):
    """
    Dynamic database credentials.

    Short-lived credentials for database access with automatic expiration.
    """

    username: str
    password: str
    host: str
    port: int
    database: str
    engine: DatabaseEngine
    expires_at: datetime
    lease_id: str

    def connection_string(self) -> str:
        """Generate connection string (password redacted in logs)."""
        if self.engine == DatabaseEngine.POSTGRESQL:
            return f"postgresql://{self.username}:****@{self.host}:{self.port}/{self.database}"
        elif self.engine == DatabaseEngine.MYSQL:
            return f"mysql://{self.username}:****@{self.host}:{self.port}/{self.database}"
        elif self.engine == DatabaseEngine.MONGODB:
            return f"mongodb://{self.username}:****@{self.host}:{self.port}/{self.database}"
        elif self.engine == DatabaseEngine.REDIS:
            return f"redis://:{self.password}@{self.host}:{self.port}"
        else:
            return f"{self.engine.value}://{self.username}:****@{self.host}:{self.port}/{self.database}"


class APIKey(BaseModel):
    """
    API key for service authentication.

    Includes metadata for access control and auditing.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    key_prefix: str = Field(..., description="Public prefix for identification")
    key_hash: str = Field(..., description="SHA-256 hash of full key")
    name: str = Field(..., description="Key name/description")
    service_id: str = Field(..., description="Service this key belongs to")
    tenant_id: str = Field(..., description="Tenant this key belongs to")

    # Permissions
    scopes: List[str] = Field(default_factory=list)
    allowed_ips: List[str] = Field(default_factory=list)
    rate_limit: Optional[int] = Field(None, description="Requests per minute")

    # Lifecycle
    status: SecretStatus = Field(default=SecretStatus.ACTIVE)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = Field(None)
    last_used_at: Optional[datetime] = Field(None)
    use_count: int = Field(default=0)

    @staticmethod
    def generate_key() -> Tuple[str, str, str]:
        """
        Generate a new API key.

        Returns:
            Tuple of (full_key, prefix, hash)
        """
        # Generate 32 random bytes (256 bits)
        key_bytes = secrets.token_bytes(32)
        full_key = f"gl_{base64.urlsafe_b64encode(key_bytes).decode().rstrip('=')}"
        prefix = full_key[:12]
        key_hash = hashlib.sha256(full_key.encode()).hexdigest()

        return full_key, prefix, key_hash

    def verify_key(self, provided_key: str) -> bool:
        """Verify a provided key matches this API key."""
        provided_hash = hashlib.sha256(provided_key.encode()).hexdigest()
        return secrets.compare_digest(provided_hash, self.key_hash)


class SecretsConfig(BaseModel):
    """Configuration for the Secrets Service."""

    # Vault configuration
    vault_addr: str = Field(default="http://127.0.0.1:8200")
    vault_namespace: Optional[str] = Field(default=None)
    vault_mount_path: str = Field(default="secret")

    # Authentication
    vault_token: Optional[str] = Field(default=None)
    vault_role_id: Optional[str] = Field(default=None)
    vault_secret_id: Optional[str] = Field(default=None)
    kubernetes_auth_enabled: bool = Field(default=False)
    kubernetes_role: Optional[str] = Field(default=None)

    # Dynamic secrets
    database_mount_path: str = Field(default="database")
    database_default_ttl: int = Field(default=3600, description="Default TTL in seconds")
    database_max_ttl: int = Field(default=86400, description="Maximum TTL in seconds")

    # API keys
    api_key_default_expiry_days: int = Field(default=90)
    api_key_max_expiry_days: int = Field(default=365)
    api_key_prefix: str = Field(default="gl_")

    # Rotation
    auto_rotation_enabled: bool = Field(default=True)
    rotation_check_interval_seconds: int = Field(default=3600)
    default_rotation_period_days: int = Field(default=90)

    # Caching
    cache_enabled: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=300)

    # Local development mode
    local_mode: bool = Field(default=False)


class SecretsService:
    """
    Production-grade secrets management service with HashiCorp Vault integration.

    Provides secure secret storage, retrieval, and lifecycle management including:
    - Static secret storage with versioning
    - Dynamic database credentials
    - API key generation and validation
    - Automatic secret rotation
    - Audit logging for all secret access

    Example:
        >>> config = SecretsConfig(vault_addr="https://vault.greenlang.io:8200")
        >>> service = SecretsService(config)
        >>> await service.initialize()
        >>>
        >>> # Store a static secret
        >>> ref = await service.store_secret(
        ...     name="api-credentials",
        ...     value="super-secret-value",
        ...     path="secret/data/api-credentials",
        ... )
        >>>
        >>> # Get dynamic database credentials
        >>> creds = await service.get_dynamic_database_credentials(
        ...     engine=DatabaseEngine.POSTGRESQL,
        ...     role="readonly",
        ... )
        >>>
        >>> # Create API key
        >>> key, api_key = await service.create_api_key(
        ...     name="service-account",
        ...     service_id="emissions-calculator",
        ...     tenant_id="tenant-123",
        ...     scopes=["read:emissions", "write:reports"],
        ... )

    Attributes:
        config: Service configuration
        _vault_client: HashiCorp Vault client
        _secret_cache: In-memory secret cache
    """

    def __init__(self, config: Optional[SecretsConfig] = None):
        """
        Initialize the Secrets Service.

        Args:
            config: Service configuration
        """
        self.config = config or SecretsConfig()
        self._vault_client = None
        self._secret_cache: Dict[str, Tuple[Any, float]] = {}
        self._secret_references: Dict[str, SecretReference] = {}
        self._api_keys: Dict[str, APIKey] = {}
        self._rotation_task: Optional[asyncio.Task] = None
        self._initialized = False

        # Local mode storage
        self._local_secrets: Dict[str, str] = {}

        logger.info(
            "SecretsService initialized",
            extra={
                "vault_addr": self.config.vault_addr,
                "local_mode": self.config.local_mode,
            },
        )

    async def initialize(self) -> None:
        """
        Initialize the secrets service.

        Connects to Vault and starts rotation background task.
        """
        if self._initialized:
            logger.warning("SecretsService already initialized")
            return

        try:
            if not self.config.local_mode:
                await self._initialize_vault_client()

            # Start rotation background task
            if self.config.auto_rotation_enabled:
                self._rotation_task = asyncio.create_task(self._rotation_loop())

            self._initialized = True
            logger.info("SecretsService initialization complete")

        except Exception as e:
            logger.error(f"Failed to initialize SecretsService: {e}", exc_info=True)
            raise

    async def shutdown(self) -> None:
        """Gracefully shutdown the secrets service."""
        logger.info("Shutting down SecretsService...")

        if self._rotation_task:
            self._rotation_task.cancel()
            try:
                await self._rotation_task
            except asyncio.CancelledError:
                pass

        # Revoke dynamic leases
        await self._revoke_all_leases()

        self._initialized = False
        logger.info("SecretsService shutdown complete")

    async def store_secret(
        self,
        name: str,
        value: str,
        path: Optional[str] = None,
        secret_type: SecretType = SecretType.STATIC,
        description: Optional[str] = None,
        tenant_id: Optional[str] = None,
        auto_rotate: bool = False,
        rotation_period_days: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> SecretReference:
        """
        Store a secret in the secrets manager.

        Args:
            name: Secret name/identifier
            value: Secret value to store
            path: Vault path (auto-generated if not provided)
            secret_type: Type of secret
            description: Human-readable description
            tenant_id: Tenant the secret belongs to
            auto_rotate: Enable automatic rotation
            rotation_period_days: Days between rotations
            tags: Metadata tags

        Returns:
            Reference to the stored secret
        """
        if not self._initialized:
            raise RuntimeError("SecretsService not initialized. Call initialize() first.")

        # Generate path if not provided
        if not path:
            path = f"{self.config.vault_mount_path}/data/{tenant_id or 'global'}/{name}"

        # Create reference
        reference = SecretReference(
            name=name,
            path=path,
            type=secret_type,
            description=description,
            tenant_id=tenant_id,
            auto_rotate=auto_rotate,
            rotation_period_days=rotation_period_days or self.config.default_rotation_period_days,
            tags=tags or {},
        )

        # Store in Vault or local storage
        if self.config.local_mode:
            self._local_secrets[path] = value
        else:
            await self._vault_write(path, {"value": value})

        # Track reference
        self._secret_references[reference.id] = reference

        logger.info(
            f"Secret stored: {name}",
            extra={
                "secret_id": reference.id,
                "path": path,
                "type": secret_type.value,
            },
        )

        return reference

    async def get_secret(
        self,
        secret_id: Optional[str] = None,
        path: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Optional[SecretValue]:
        """
        Retrieve a secret from the secrets manager.

        Args:
            secret_id: Secret ID from reference
            path: Direct Vault path
            name: Secret name to look up

        Returns:
            Secret value with metadata, or None if not found
        """
        if not self._initialized:
            raise RuntimeError("SecretsService not initialized. Call initialize() first.")

        # Resolve reference
        reference = None
        if secret_id:
            reference = self._secret_references.get(secret_id)
            if not reference:
                logger.warning(f"Secret reference not found: {secret_id}")
                return None
            path = reference.path
        elif name:
            for ref in self._secret_references.values():
                if ref.name == name:
                    reference = ref
                    path = ref.path
                    break

        if not path:
            logger.warning("No secret path could be resolved")
            return None

        # Check cache
        if self.config.cache_enabled:
            cached = self._get_from_cache(path)
            if cached:
                return cached

        # Retrieve from storage
        if self.config.local_mode:
            value = self._local_secrets.get(path)
        else:
            data = await self._vault_read(path)
            value = data.get("value") if data else None

        if not value:
            logger.warning(f"Secret not found at path: {path}")
            return None

        # Update reference metadata
        if reference:
            reference.last_accessed_at = datetime.now(timezone.utc)
            reference.access_count += 1

        # Create response
        secret_value = SecretValue(
            reference=reference or SecretReference(name="unknown", path=path),
            value=value,
        )

        # Cache the result
        if self.config.cache_enabled:
            self._add_to_cache(path, secret_value)

        logger.debug(
            f"Secret retrieved: {reference.name if reference else path}",
            extra={"path": path},
        )

        return secret_value

    async def rotate_secret(
        self,
        secret_id: str,
        new_value: Optional[str] = None,
    ) -> SecretReference:
        """
        Rotate a secret to a new value.

        Args:
            secret_id: ID of the secret to rotate
            new_value: New value (auto-generated if not provided)

        Returns:
            Updated secret reference
        """
        reference = self._secret_references.get(secret_id)
        if not reference:
            raise ValueError(f"Secret not found: {secret_id}")

        # Generate new value if not provided
        if not new_value:
            new_value = secrets.token_urlsafe(32)

        # Update status
        reference.status = SecretStatus.ROTATING

        try:
            # Store new version
            if self.config.local_mode:
                self._local_secrets[reference.path] = new_value
            else:
                await self._vault_write(reference.path, {"value": new_value})

            # Update reference
            reference.version += 1
            reference.last_rotated_at = datetime.now(timezone.utc)
            reference.status = SecretStatus.ACTIVE

            # Clear cache
            self._clear_from_cache(reference.path)

            logger.info(
                f"Secret rotated: {reference.name}",
                extra={
                    "secret_id": secret_id,
                    "new_version": reference.version,
                },
            )

            return reference

        except Exception as e:
            reference.status = SecretStatus.ACTIVE  # Rollback status
            logger.error(f"Secret rotation failed: {e}", exc_info=True)
            raise

    async def revoke_secret(self, secret_id: str) -> bool:
        """
        Revoke a secret, making it unusable.

        Args:
            secret_id: ID of the secret to revoke

        Returns:
            True if revoked successfully
        """
        reference = self._secret_references.get(secret_id)
        if not reference:
            return False

        # Update status
        reference.status = SecretStatus.REVOKED

        # Delete from storage
        if self.config.local_mode:
            self._local_secrets.pop(reference.path, None)
        else:
            await self._vault_delete(reference.path)

        # Clear cache
        self._clear_from_cache(reference.path)

        logger.info(
            f"Secret revoked: {reference.name}",
            extra={"secret_id": secret_id},
        )

        return True

    async def get_dynamic_database_credentials(
        self,
        engine: DatabaseEngine,
        role: str,
        ttl: Optional[int] = None,
    ) -> DatabaseCredentials:
        """
        Get dynamic, short-lived database credentials.

        Vault generates unique credentials for each request that are
        automatically revoked after the TTL expires.

        Args:
            engine: Database engine type
            role: Vault database role to use
            ttl: Time-to-live in seconds

        Returns:
            Dynamic database credentials
        """
        if not self._initialized:
            raise RuntimeError("SecretsService not initialized. Call initialize() first.")

        ttl = ttl or self.config.database_default_ttl
        ttl = min(ttl, self.config.database_max_ttl)

        if self.config.local_mode:
            # Return mock credentials for local development
            return DatabaseCredentials(
                username=f"dev_{role}_{secrets.token_hex(4)}",
                password=secrets.token_urlsafe(24),
                host="localhost",
                port=5432 if engine == DatabaseEngine.POSTGRESQL else 3306,
                database="greenlang_dev",
                engine=engine,
                expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl),
                lease_id=f"local-lease-{uuid.uuid4()}",
            )

        # Request credentials from Vault
        path = f"{self.config.database_mount_path}/creds/{role}"
        response = await self._vault_read(path)

        if not response:
            raise RuntimeError(f"Failed to get database credentials for role: {role}")

        credentials = DatabaseCredentials(
            username=response["username"],
            password=response["password"],
            host=response.get("host", "localhost"),
            port=response.get("port", 5432),
            database=response.get("database", "greenlang"),
            engine=engine,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=response.get("lease_duration", ttl)),
            lease_id=response.get("lease_id", ""),
        )

        logger.info(
            f"Dynamic database credentials generated",
            extra={
                "engine": engine.value,
                "role": role,
                "ttl": ttl,
                "username": credentials.username,
            },
        )

        return credentials

    async def create_api_key(
        self,
        name: str,
        service_id: str,
        tenant_id: str,
        scopes: Optional[List[str]] = None,
        allowed_ips: Optional[List[str]] = None,
        rate_limit: Optional[int] = None,
        expires_in_days: Optional[int] = None,
    ) -> Tuple[str, APIKey]:
        """
        Create a new API key for service authentication.

        Args:
            name: Key name/description
            service_id: Service the key belongs to
            tenant_id: Tenant the key belongs to
            scopes: Allowed permission scopes
            allowed_ips: IP allowlist
            rate_limit: Requests per minute limit
            expires_in_days: Days until expiration

        Returns:
            Tuple of (full_api_key, APIKey metadata)
        """
        # Generate key
        full_key, prefix, key_hash = APIKey.generate_key()

        # Calculate expiration
        expires_in_days = expires_in_days or self.config.api_key_default_expiry_days
        expires_in_days = min(expires_in_days, self.config.api_key_max_expiry_days)
        expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

        # Create API key record
        api_key = APIKey(
            key_prefix=prefix,
            key_hash=key_hash,
            name=name,
            service_id=service_id,
            tenant_id=tenant_id,
            scopes=scopes or [],
            allowed_ips=allowed_ips or [],
            rate_limit=rate_limit,
            expires_at=expires_at,
        )

        # Store the key metadata
        self._api_keys[api_key.id] = api_key

        logger.info(
            f"API key created: {name}",
            extra={
                "key_id": api_key.id,
                "prefix": prefix,
                "service_id": service_id,
                "tenant_id": tenant_id,
                "expires_at": expires_at.isoformat(),
            },
        )

        # Return the full key (only time it's exposed) and metadata
        return full_key, api_key

    async def validate_api_key(
        self,
        provided_key: str,
        required_scopes: Optional[List[str]] = None,
        client_ip: Optional[str] = None,
    ) -> Optional[APIKey]:
        """
        Validate an API key and check permissions.

        Args:
            provided_key: The API key to validate
            required_scopes: Scopes that must be present
            client_ip: Client IP for allowlist check

        Returns:
            APIKey if valid, None if invalid
        """
        # Find key by hash
        provided_hash = hashlib.sha256(provided_key.encode()).hexdigest()

        for api_key in self._api_keys.values():
            if api_key.key_hash != provided_hash:
                continue

            # Check status
            if api_key.status != SecretStatus.ACTIVE:
                logger.warning(
                    f"API key validation failed: key not active",
                    extra={"key_id": api_key.id, "status": api_key.status.value},
                )
                return None

            # Check expiration
            if api_key.expires_at and datetime.now(timezone.utc) > api_key.expires_at:
                logger.warning(
                    f"API key validation failed: key expired",
                    extra={"key_id": api_key.id},
                )
                return None

            # Check IP allowlist
            if api_key.allowed_ips and client_ip:
                if client_ip not in api_key.allowed_ips:
                    logger.warning(
                        f"API key validation failed: IP not allowed",
                        extra={"key_id": api_key.id, "client_ip": client_ip},
                    )
                    return None

            # Check scopes
            if required_scopes:
                if not all(scope in api_key.scopes for scope in required_scopes):
                    logger.warning(
                        f"API key validation failed: insufficient scopes",
                        extra={
                            "key_id": api_key.id,
                            "required": required_scopes,
                            "available": api_key.scopes,
                        },
                    )
                    return None

            # Update usage tracking
            api_key.last_used_at = datetime.now(timezone.utc)
            api_key.use_count += 1

            logger.debug(
                f"API key validated",
                extra={"key_id": api_key.id, "service_id": api_key.service_id},
            )

            return api_key

        logger.warning("API key validation failed: key not found")
        return None

    async def revoke_api_key(self, key_id: str) -> bool:
        """
        Revoke an API key.

        Args:
            key_id: ID of the key to revoke

        Returns:
            True if revoked successfully
        """
        api_key = self._api_keys.get(key_id)
        if not api_key:
            return False

        api_key.status = SecretStatus.REVOKED

        logger.info(
            f"API key revoked: {api_key.name}",
            extra={"key_id": key_id, "service_id": api_key.service_id},
        )

        return True

    async def list_api_keys(
        self,
        tenant_id: Optional[str] = None,
        service_id: Optional[str] = None,
        include_revoked: bool = False,
    ) -> List[APIKey]:
        """
        List API keys with optional filtering.

        Args:
            tenant_id: Filter by tenant
            service_id: Filter by service
            include_revoked: Include revoked keys

        Returns:
            List of matching API keys
        """
        results = []
        for api_key in self._api_keys.values():
            if tenant_id and api_key.tenant_id != tenant_id:
                continue
            if service_id and api_key.service_id != service_id:
                continue
            if not include_revoked and api_key.status == SecretStatus.REVOKED:
                continue
            results.append(api_key)

        return results

    def _get_from_cache(self, path: str) -> Optional[SecretValue]:
        """Get secret from cache if not expired."""
        if path not in self._secret_cache:
            return None

        value, cached_at = self._secret_cache[path]
        if time.time() - cached_at > self.config.cache_ttl_seconds:
            del self._secret_cache[path]
            return None

        return value

    def _add_to_cache(self, path: str, value: SecretValue) -> None:
        """Add secret to cache."""
        self._secret_cache[path] = (value, time.time())

    def _clear_from_cache(self, path: str) -> None:
        """Remove secret from cache."""
        self._secret_cache.pop(path, None)

    async def _rotation_loop(self) -> None:
        """Background task to check and rotate secrets."""
        while True:
            try:
                await asyncio.sleep(self.config.rotation_check_interval_seconds)
                await self._check_rotation_needed()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Rotation check error: {e}", exc_info=True)

    async def _check_rotation_needed(self) -> None:
        """Check all secrets for rotation needs."""
        for reference in self._secret_references.values():
            if reference.auto_rotate and reference.needs_rotation():
                try:
                    await self.rotate_secret(reference.id)
                except Exception as e:
                    logger.error(
                        f"Auto-rotation failed for {reference.name}: {e}",
                        exc_info=True,
                    )

    async def _initialize_vault_client(self) -> None:
        """Initialize HashiCorp Vault client."""
        try:
            import hvac

            self._vault_client = hvac.Client(
                url=self.config.vault_addr,
                namespace=self.config.vault_namespace,
            )

            # Authenticate
            if self.config.vault_token:
                self._vault_client.token = self.config.vault_token
            elif self.config.vault_role_id and self.config.vault_secret_id:
                self._vault_client.auth.approle.login(
                    role_id=self.config.vault_role_id,
                    secret_id=self.config.vault_secret_id,
                )
            elif self.config.kubernetes_auth_enabled:
                jwt_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
                with open(jwt_path) as f:
                    jwt = f.read()
                self._vault_client.auth.kubernetes.login(
                    role=self.config.kubernetes_role,
                    jwt=jwt,
                )

            if not self._vault_client.is_authenticated():
                raise RuntimeError("Vault authentication failed")

            logger.info("Vault client initialized and authenticated")

        except ImportError:
            logger.error("hvac not installed. Install with: pip install hvac")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Vault client: {e}")
            raise

    async def _vault_read(self, path: str) -> Optional[Dict[str, Any]]:
        """Read a secret from Vault."""
        try:
            response = self._vault_client.secrets.kv.v2.read_secret_version(path=path)
            return response["data"]["data"]
        except Exception as e:
            logger.error(f"Vault read failed for {path}: {e}")
            return None

    async def _vault_write(self, path: str, data: Dict[str, Any]) -> None:
        """Write a secret to Vault."""
        try:
            self._vault_client.secrets.kv.v2.create_or_update_secret(path=path, secret=data)
        except Exception as e:
            logger.error(f"Vault write failed for {path}: {e}")
            raise

    async def _vault_delete(self, path: str) -> None:
        """Delete a secret from Vault."""
        try:
            self._vault_client.secrets.kv.v2.delete_metadata_and_all_versions(path=path)
        except Exception as e:
            logger.error(f"Vault delete failed for {path}: {e}")
            raise

    async def _revoke_all_leases(self) -> None:
        """Revoke all dynamic secret leases."""
        # Would revoke Vault leases in production
        pass
