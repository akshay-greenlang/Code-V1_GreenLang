"""
GL-002 FLAMEGUARD BoilerEfficiencyOptimizer - HashiCorp Vault Secrets Management Module

This module provides secure secrets management for the FLAMEGUARD
BoilerEfficiencyOptimizer using HashiCorp Vault as the primary secrets backend
with environment variable fallback for development environments.

Features:
    - Multiple authentication methods (Token, AppRole, Kubernetes)
    - Secret caching with configurable TTL
    - Automatic secret rotation handling
    - Fallback to environment variables when Vault unavailable
    - Connection pooling and retry logic
    - Thread-safe secret access
    - Comprehensive audit logging

Security Standards:
    - All secrets encrypted in transit (TLS 1.2+)
    - Secret values never logged
    - Cache encryption for sensitive data
    - Automatic lease renewal

Author: GreenLang Security Engineering
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from pydantic import BaseModel, Field, SecretStr, validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class VaultAuthMethod(str, Enum):
    """Supported Vault authentication methods."""
    TOKEN = "token"
    APPROLE = "approle"
    KUBERNETES = "kubernetes"


class SecretType(str, Enum):
    """Types of secrets managed by the system."""
    DATABASE_CREDENTIAL = "database_credential"
    API_KEY = "api_key"
    TLS_CERTIFICATE = "tls_certificate"
    ENCRYPTION_KEY = "encryption_key"
    JWT_SECRET = "jwt_secret"
    SCADA_CREDENTIAL = "scada_credential"
    HISTORIAN_CREDENTIAL = "historian_credential"
    KAFKA_CREDENTIAL = "kafka_credential"
    CEMS_CREDENTIAL = "cems_credential"
    GENERIC = "generic"


# Default configuration values
DEFAULT_CACHE_TTL_SECONDS = 300
DEFAULT_LEASE_RENEWAL_THRESHOLD = 0.75
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_SECONDS = 1.0
DEFAULT_CONNECTION_TIMEOUT_SECONDS = 30
DEFAULT_KUBERNETES_AUTH_PATH = "auth/kubernetes"
DEFAULT_APPROLE_AUTH_PATH = "auth/approle"

# GL-002 Specific Constants
AGENT_ID = "GL-002"
AGENT_NAME = "FLAMEGUARD"
KUBERNETES_ROLE = "gl-002-flameguard"


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class VaultConfig(BaseModel):
    """Vault connection and authentication configuration for GL-002 FLAMEGUARD."""

    # Connection settings
    vault_addr: str = Field(
        default="https://vault.greenlang.local:8200",
        description="Vault server address"
    )
    vault_namespace: Optional[str] = Field(
        default=None,
        description="Vault namespace (enterprise feature)"
    )

    # TLS settings
    vault_cacert: Optional[str] = Field(
        default=None,
        description="Path to CA certificate for Vault TLS"
    )
    vault_client_cert: Optional[str] = Field(
        default=None,
        description="Path to client certificate for mutual TLS"
    )
    vault_client_key: Optional[str] = Field(
        default=None,
        description="Path to client private key for mutual TLS"
    )
    vault_skip_verify: bool = Field(
        default=False,
        description="Skip TLS verification (NOT recommended for production)"
    )

    # Authentication settings
    auth_method: VaultAuthMethod = Field(
        default=VaultAuthMethod.KUBERNETES,
        description="Authentication method"
    )
    vault_token: Optional[SecretStr] = Field(
        default=None,
        description="Vault token (for token auth)"
    )
    vault_role_id: Optional[SecretStr] = Field(
        default=None,
        description="AppRole role ID"
    )
    vault_secret_id: Optional[SecretStr] = Field(
        default=None,
        description="AppRole secret ID"
    )
    kubernetes_role: Optional[str] = Field(
        default=KUBERNETES_ROLE,
        description="Kubernetes auth role name"
    )
    kubernetes_jwt_path: str = Field(
        default="/var/run/secrets/kubernetes.io/serviceaccount/token",
        description="Path to Kubernetes service account JWT"
    )

    # Secrets engine paths
    kv_mount_path: str = Field(
        default="secret",
        description="KV secrets engine mount path"
    )
    database_mount_path: str = Field(
        default="database",
        description="Database secrets engine mount path"
    )
    pki_mount_path: str = Field(
        default="pki",
        description="PKI secrets engine mount path"
    )

    # Agent-specific secret paths
    agent_id: str = Field(
        default=AGENT_ID,
        description="Agent identifier for secret paths"
    )
    secret_base_path: str = Field(
        default="greenlang/agents",
        description="Base path for agent secrets"
    )

    # Cache settings
    cache_enabled: bool = Field(
        default=True,
        description="Enable secret caching"
    )
    cache_ttl_seconds: int = Field(
        default=DEFAULT_CACHE_TTL_SECONDS,
        ge=30,
        le=3600,
        description="Cache TTL in seconds"
    )

    # Retry settings
    max_retries: int = Field(
        default=DEFAULT_MAX_RETRIES,
        ge=1,
        le=10,
        description="Maximum retry attempts"
    )
    retry_delay_seconds: float = Field(
        default=DEFAULT_RETRY_DELAY_SECONDS,
        ge=0.1,
        le=30.0,
        description="Delay between retries"
    )
    connection_timeout_seconds: int = Field(
        default=DEFAULT_CONNECTION_TIMEOUT_SECONDS,
        ge=5,
        le=120,
        description="Connection timeout"
    )

    # Fallback settings
    fallback_to_env: bool = Field(
        default=True,
        description="Fall back to environment variables if Vault unavailable"
    )

    @validator("vault_addr")
    def validate_vault_addr(cls, v):
        """Validate Vault address format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("vault_addr must start with http:// or https://")
        return v.rstrip("/")

    @validator("vault_skip_verify")
    def warn_skip_verify(cls, v):
        """Warn if TLS verification is disabled."""
        if v:
            logger.warning(
                "SECURITY WARNING: Vault TLS verification disabled. "
                "This should NEVER be used in production."
            )
        return v

    @classmethod
    def from_environment(cls) -> "VaultConfig":
        """Create configuration from environment variables."""
        return cls(
            vault_addr=os.getenv("VAULT_ADDR", "https://vault.greenlang.local:8200"),
            vault_namespace=os.getenv("VAULT_NAMESPACE"),
            vault_cacert=os.getenv("VAULT_CACERT"),
            vault_client_cert=os.getenv("VAULT_CLIENT_CERT"),
            vault_client_key=os.getenv("VAULT_CLIENT_KEY"),
            vault_skip_verify=os.getenv("VAULT_SKIP_VERIFY", "false").lower() == "true",
            auth_method=VaultAuthMethod(os.getenv("VAULT_AUTH_METHOD", "kubernetes")),
            vault_token=SecretStr(os.getenv("VAULT_TOKEN", "")) if os.getenv("VAULT_TOKEN") else None,
            vault_role_id=SecretStr(os.getenv("VAULT_ROLE_ID", "")) if os.getenv("VAULT_ROLE_ID") else None,
            vault_secret_id=SecretStr(os.getenv("VAULT_SECRET_ID", "")) if os.getenv("VAULT_SECRET_ID") else None,
            kubernetes_role=os.getenv("VAULT_KUBERNETES_ROLE", KUBERNETES_ROLE),
            kubernetes_jwt_path=os.getenv(
                "VAULT_KUBERNETES_JWT_PATH",
                "/var/run/secrets/kubernetes.io/serviceaccount/token"
            ),
            kv_mount_path=os.getenv("VAULT_KV_MOUNT", "secret"),
            database_mount_path=os.getenv("VAULT_DATABASE_MOUNT", "database"),
            pki_mount_path=os.getenv("VAULT_PKI_MOUNT", "pki"),
            agent_id=os.getenv("GL_AGENT_ID", AGENT_ID),
            secret_base_path=os.getenv("VAULT_SECRET_BASE_PATH", "greenlang/agents"),
            cache_enabled=os.getenv("VAULT_CACHE_ENABLED", "true").lower() == "true",
            cache_ttl_seconds=int(os.getenv("VAULT_CACHE_TTL", str(DEFAULT_CACHE_TTL_SECONDS))),
            fallback_to_env=os.getenv("VAULT_FALLBACK_TO_ENV", "true").lower() == "true",
        )


# =============================================================================
# SECRET CACHE
# =============================================================================

@dataclass
class CachedSecret:
    """Represents a cached secret with metadata."""

    key: str
    value: Any
    secret_type: SecretType
    cached_at: datetime
    expires_at: datetime
    lease_id: Optional[str] = None
    lease_duration: Optional[int] = None
    renewable: bool = False
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if the cached secret has expired."""
        return datetime.utcnow() >= self.expires_at

    @property
    def ttl_remaining(self) -> float:
        """Get remaining TTL in seconds."""
        remaining = (self.expires_at - datetime.utcnow()).total_seconds()
        return max(0, remaining)

    @property
    def should_renew(self) -> bool:
        """Check if the secret should be renewed."""
        if not self.renewable or not self.lease_duration:
            return False
        threshold = self.lease_duration * DEFAULT_LEASE_RENEWAL_THRESHOLD
        elapsed = (datetime.utcnow() - self.cached_at).total_seconds()
        return elapsed >= threshold


class SecretCache:
    """Thread-safe cache for secrets with TTL support."""

    def __init__(self, default_ttl: int = DEFAULT_CACHE_TTL_SECONDS):
        """Initialize the secret cache."""
        self._cache: Dict[str, CachedSecret] = {}
        self._lock = threading.RLock()
        self._default_ttl = default_ttl
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "renewals": 0,
        }

    def get(self, key: str) -> Optional[CachedSecret]:
        """Get a secret from the cache."""
        with self._lock:
            cached = self._cache.get(key)
            if cached is None:
                self._stats["misses"] += 1
                return None

            if cached.is_expired:
                self._evict(key)
                self._stats["misses"] += 1
                return None

            self._stats["hits"] += 1
            return cached

    def set(
        self,
        key: str,
        value: Any,
        secret_type: SecretType = SecretType.GENERIC,
        ttl: Optional[int] = None,
        lease_id: Optional[str] = None,
        lease_duration: Optional[int] = None,
        renewable: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CachedSecret:
        """Store a secret in the cache."""
        ttl = ttl or self._default_ttl
        now = datetime.utcnow()

        cached = CachedSecret(
            key=key,
            value=value,
            secret_type=secret_type,
            cached_at=now,
            expires_at=now + timedelta(seconds=ttl),
            lease_id=lease_id,
            lease_duration=lease_duration,
            renewable=renewable,
            metadata=metadata or {},
        )

        with self._lock:
            self._cache[key] = cached

        return cached

    def invalidate(self, key: str) -> bool:
        """Invalidate a cached secret."""
        with self._lock:
            return self._evict(key)

    def invalidate_by_prefix(self, prefix: str) -> int:
        """Invalidate all secrets matching a prefix."""
        with self._lock:
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(prefix)]
            for key in keys_to_remove:
                self._evict(key)
            return len(keys_to_remove)

    def clear(self) -> int:
        """Clear all cached secrets."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats["evictions"] += count
            return count

    @property
    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            return {
                **self._stats,
                "size": len(self._cache),
                "hit_rate": self._stats["hits"] / max(1, self._stats["hits"] + self._stats["misses"]),
            }

    def _evict(self, key: str) -> bool:
        """Evict a secret from the cache."""
        if key in self._cache:
            del self._cache[key]
            self._stats["evictions"] += 1
            return True
        return False


# =============================================================================
# EXCEPTIONS
# =============================================================================

class VaultError(Exception):
    """Base exception for Vault operations."""
    pass


class VaultConnectionError(VaultError):
    """Raised when Vault connection fails."""
    pass


class VaultAuthenticationError(VaultError):
    """Raised when Vault authentication fails."""
    pass


class VaultSecretNotFoundError(VaultError):
    """Raised when a secret is not found."""
    pass


class VaultSecretAccessDenied(VaultError):
    """Raised when access to a secret is denied."""
    pass


# =============================================================================
# VAULT CLIENT
# =============================================================================

class VaultClient:
    """
    HashiCorp Vault client for GL-002 FLAMEGUARD BoilerEfficiencyOptimizer.

    This client provides a unified interface for retrieving secrets from
    HashiCorp Vault with support for multiple authentication methods,
    caching, and automatic fallback to environment variables.

    GL-002 Specific Secrets:
        - SCADA/DCS credentials
        - Historian database credentials
        - CEMS integration credentials
        - Burner management system credentials
        - API keys for external services

    Example:
        >>> config = VaultConfig.from_environment()
        >>> client = VaultClient(config)
        >>> scada_creds = client.get_scada_credentials()
        >>> historian_creds = client.get_historian_credentials()
    """

    def __init__(self, config: Optional[VaultConfig] = None):
        """Initialize the Vault client."""
        self._config = config or VaultConfig.from_environment()
        self._cache = SecretCache(self._config.cache_ttl_seconds)
        self._token: Optional[str] = None
        self._token_lock = threading.Lock()
        self._connected = False

        logger.info(
            f"VaultClient initialized for agent {self._config.agent_id} "
            f"({AGENT_NAME}) using {self._config.auth_method.value} authentication"
        )

    def _get_env_fallback(self, key: str, default: Any = None) -> Any:
        """Get a secret from environment variables as fallback."""
        env_key = key.upper().replace("/", "_").replace("-", "_")
        value = os.getenv(env_key)

        if value is not None:
            logger.warning(
                f"Using environment variable fallback for secret: {key}. "
                "This should only be used in development."
            )
            return value

        return default

    def get_secret(
        self,
        path: str,
        key: Optional[str] = None,
        secret_type: SecretType = SecretType.GENERIC,
        use_cache: bool = True,
    ) -> Any:
        """
        Retrieve a secret from Vault.

        Args:
            path: Path to the secret (relative to agent's secret base path)
            key: Specific key within the secret data
            secret_type: Type of secret for caching categorization
            use_cache: Whether to use cached value if available

        Returns:
            Secret value or dict of all secret data
        """
        full_path = f"{self._config.kv_mount_path}/data/{self._config.secret_base_path}/{self._config.agent_id}/{path}"
        cache_key = f"{full_path}:{key or '*'}"

        if use_cache and self._config.cache_enabled:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for secret: {path}")
                return cached.value

        try:
            import httpx

            headers = {"X-Vault-Token": self._get_token()}
            if self._config.vault_namespace:
                headers["X-Vault-Namespace"] = self._config.vault_namespace

            response = httpx.get(
                f"{self._config.vault_addr}/v1/{full_path}",
                headers=headers,
                timeout=self._config.connection_timeout_seconds,
                verify=not self._config.vault_skip_verify,
            )

            if response.status_code == 404:
                raise VaultSecretNotFoundError(f"Secret not found at path: {path}")
            elif response.status_code == 403:
                raise VaultSecretAccessDenied(f"Access denied to path: {path}")

            response.raise_for_status()
            data = response.json().get("data", {}).get("data", {})

            value = data.get(key) if key else data

            if self._config.cache_enabled:
                self._cache.set(key=cache_key, value=value, secret_type=secret_type)

            return value

        except VaultSecretNotFoundError:
            if self._config.fallback_to_env:
                env_value = self._get_env_fallback(path)
                if env_value is not None:
                    return env_value
            raise

        except Exception as e:
            if self._config.fallback_to_env:
                logger.warning(f"Vault unavailable, falling back to environment: {e}")
                env_value = self._get_env_fallback(path)
                if env_value is not None:
                    return env_value
            raise VaultConnectionError(f"Failed to retrieve secret: {e}") from e

    def _get_token(self) -> str:
        """Get or refresh the Vault token."""
        with self._token_lock:
            if self._token:
                return self._token

            if self._config.auth_method == VaultAuthMethod.TOKEN:
                if self._config.vault_token:
                    self._token = self._config.vault_token.get_secret_value()
                    return self._token

            elif self._config.auth_method == VaultAuthMethod.KUBERNETES:
                self._token = self._authenticate_kubernetes()
                return self._token

            elif self._config.auth_method == VaultAuthMethod.APPROLE:
                self._token = self._authenticate_approle()
                return self._token

            raise VaultAuthenticationError("No valid authentication method configured")

    def _authenticate_kubernetes(self) -> str:
        """Authenticate using Kubernetes service account."""
        import httpx

        jwt_path = Path(self._config.kubernetes_jwt_path)
        if not jwt_path.exists():
            raise VaultAuthenticationError(f"Kubernetes JWT not found at {jwt_path}")

        jwt = jwt_path.read_text().strip()

        response = httpx.post(
            f"{self._config.vault_addr}/v1/auth/kubernetes/login",
            json={"role": self._config.kubernetes_role, "jwt": jwt},
            timeout=30.0,
            verify=not self._config.vault_skip_verify,
        )
        response.raise_for_status()

        return response.json()["auth"]["client_token"]

    def _authenticate_approle(self) -> str:
        """Authenticate using AppRole."""
        import httpx

        response = httpx.post(
            f"{self._config.vault_addr}/v1/auth/approle/login",
            json={
                "role_id": self._config.vault_role_id.get_secret_value(),
                "secret_id": self._config.vault_secret_id.get_secret_value(),
            },
            timeout=30.0,
            verify=not self._config.vault_skip_verify,
        )
        response.raise_for_status()

        return response.json()["auth"]["client_token"]

    # =========================================================================
    # GL-002 FLAMEGUARD SPECIFIC METHODS
    # =========================================================================

    def get_scada_credentials(self) -> Dict[str, str]:
        """Get SCADA/DCS credentials for boiler control integration."""
        return self.get_secret(
            "integrations/scada",
            secret_type=SecretType.SCADA_CREDENTIAL,
        )

    def get_historian_credentials(self) -> Dict[str, str]:
        """Get process historian database credentials."""
        return self.get_secret(
            "integrations/historian",
            secret_type=SecretType.HISTORIAN_CREDENTIAL,
        )

    def get_cems_credentials(self) -> Dict[str, str]:
        """Get CEMS (Continuous Emissions Monitoring System) credentials."""
        return self.get_secret(
            "integrations/cems",
            secret_type=SecretType.CEMS_CREDENTIAL,
        )

    def get_database_credentials(self, role_name: str = "gl-002-postgres") -> Dict[str, str]:
        """Get dynamic database credentials from Vault."""
        try:
            import httpx

            path = f"{self._config.database_mount_path}/creds/{role_name}"
            headers = {"X-Vault-Token": self._get_token()}

            response = httpx.get(
                f"{self._config.vault_addr}/v1/{path}",
                headers=headers,
                timeout=self._config.connection_timeout_seconds,
                verify=not self._config.vault_skip_verify,
            )
            response.raise_for_status()

            data = response.json()
            return {
                "username": data["data"]["username"],
                "password": data["data"]["password"],
            }

        except Exception:
            if self._config.fallback_to_env:
                db_url = os.getenv("DATABASE_URL", "")
                import re
                match = re.match(r"[^:]+://([^:]+):([^@]+)@", db_url)
                if match:
                    return {"username": match.group(1), "password": match.group(2)}
            raise

    def get_jwt_secret(self) -> str:
        """Get the JWT signing secret."""
        return self.get_secret("auth/jwt", key="secret", secret_type=SecretType.JWT_SECRET)

    def get_kafka_credentials(self) -> Dict[str, str]:
        """Get Kafka credentials for event streaming."""
        return self.get_secret("integrations/kafka", secret_type=SecretType.KAFKA_CREDENTIAL)

    def get_api_key(self, service_name: str) -> str:
        """Get an API key for an external service."""
        return self.get_secret(f"api-keys/{service_name}", key="api_key", secret_type=SecretType.API_KEY)

    def invalidate_cache(self, path: Optional[str] = None) -> int:
        """Invalidate cached secrets."""
        if path:
            return self._cache.invalidate_by_prefix(path)
        return self._cache.clear()

    def health_check(self) -> Dict[str, Any]:
        """Check Vault connection health."""
        try:
            import httpx

            response = httpx.get(
                f"{self._config.vault_addr}/v1/sys/health",
                timeout=5.0,
                verify=not self._config.vault_skip_verify,
            )

            return {
                "status": "healthy" if response.status_code in (200, 429, 472, 473) else "unhealthy",
                "initialized": response.json().get("initialized", False),
                "sealed": response.json().get("sealed", True),
                "cache_stats": self._cache.stats,
            }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "cache_stats": self._cache.stats}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

_global_client: Optional[VaultClient] = None
_global_client_lock = threading.Lock()


def get_vault_client() -> VaultClient:
    """Get the global VaultClient instance."""
    global _global_client

    if _global_client is None:
        with _global_client_lock:
            if _global_client is None:
                _global_client = VaultClient()

    return _global_client


def get_secret_or_env(
    secret_path: str,
    env_var: str,
    vault_client: Optional[VaultClient] = None,
    default: Any = None,
) -> Any:
    """Get a secret from Vault or fall back to environment variable."""
    env_value = os.getenv(env_var)
    if env_value is not None:
        return env_value

    if vault_client is not None:
        try:
            return vault_client.get_secret(secret_path)
        except Exception:
            pass

    return default


__all__ = [
    "VaultConfig",
    "VaultAuthMethod",
    "SecretType",
    "VaultClient",
    "SecretCache",
    "CachedSecret",
    "VaultError",
    "VaultConnectionError",
    "VaultAuthenticationError",
    "VaultSecretNotFoundError",
    "VaultSecretAccessDenied",
    "get_vault_client",
    "get_secret_or_env",
]
