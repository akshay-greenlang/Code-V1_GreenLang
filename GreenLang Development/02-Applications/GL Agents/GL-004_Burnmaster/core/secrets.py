"""
GL-004 BURNMASTER Burner Optimization Agent - HashiCorp Vault Secrets Management Module

This module provides secure secrets management for the BURNMASTER
Burner Optimization Agent using HashiCorp Vault as the primary secrets backend
with environment variable fallback for development environments.

GL-004 Specific Features:
    - Flame scanner system credentials
    - Combustion analyzer credentials
    - Air-fuel ratio controller credentials
    - NOx/CO monitoring system credentials

Author: GreenLang Security Engineering
Version: 1.0.0
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    FLAME_SCANNER_CREDENTIAL = "flame_scanner_credential"
    COMBUSTION_ANALYZER_CREDENTIAL = "combustion_analyzer_credential"
    AIR_FUEL_CONTROLLER_CREDENTIAL = "air_fuel_controller_credential"
    EMISSIONS_MONITOR_CREDENTIAL = "emissions_monitor_credential"
    DCS_CREDENTIAL = "dcs_credential"
    HISTORIAN_CREDENTIAL = "historian_credential"
    KAFKA_CREDENTIAL = "kafka_credential"
    GENERIC = "generic"


DEFAULT_CACHE_TTL_SECONDS = 300
DEFAULT_LEASE_RENEWAL_THRESHOLD = 0.75
DEFAULT_MAX_RETRIES = 3

# GL-004 Specific Constants
AGENT_ID = "GL-004"
AGENT_NAME = "BURNMASTER"
KUBERNETES_ROLE = "gl-004-burnmaster"


# =============================================================================
# CONFIGURATION
# =============================================================================

class VaultConfig(BaseModel):
    """Vault configuration for GL-004 BURNMASTER."""

    vault_addr: str = Field(default="https://vault.greenlang.local:8200")
    vault_namespace: Optional[str] = Field(default=None)
    vault_cacert: Optional[str] = Field(default=None)
    vault_skip_verify: bool = Field(default=False)

    auth_method: VaultAuthMethod = Field(default=VaultAuthMethod.KUBERNETES)
    vault_token: Optional[SecretStr] = Field(default=None)
    vault_role_id: Optional[SecretStr] = Field(default=None)
    vault_secret_id: Optional[SecretStr] = Field(default=None)
    kubernetes_role: Optional[str] = Field(default=KUBERNETES_ROLE)
    kubernetes_jwt_path: str = Field(default="/var/run/secrets/kubernetes.io/serviceaccount/token")

    kv_mount_path: str = Field(default="secret")
    database_mount_path: str = Field(default="database")
    pki_mount_path: str = Field(default="pki")

    agent_id: str = Field(default=AGENT_ID)
    secret_base_path: str = Field(default="greenlang/agents")

    cache_enabled: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=DEFAULT_CACHE_TTL_SECONDS, ge=30, le=3600)
    max_retries: int = Field(default=DEFAULT_MAX_RETRIES, ge=1, le=10)
    connection_timeout_seconds: int = Field(default=30, ge=5, le=120)
    fallback_to_env: bool = Field(default=True)

    @validator("vault_addr")
    def validate_vault_addr(cls, v):
        if not v.startswith(("http://", "https://")):
            raise ValueError("vault_addr must start with http:// or https://")
        return v.rstrip("/")

    @classmethod
    def from_environment(cls) -> "VaultConfig":
        """Create configuration from environment variables."""
        return cls(
            vault_addr=os.getenv("VAULT_ADDR", "https://vault.greenlang.local:8200"),
            vault_namespace=os.getenv("VAULT_NAMESPACE"),
            vault_cacert=os.getenv("VAULT_CACERT"),
            vault_skip_verify=os.getenv("VAULT_SKIP_VERIFY", "false").lower() == "true",
            auth_method=VaultAuthMethod(os.getenv("VAULT_AUTH_METHOD", "kubernetes")),
            vault_token=SecretStr(os.getenv("VAULT_TOKEN", "")) if os.getenv("VAULT_TOKEN") else None,
            kubernetes_role=os.getenv("VAULT_KUBERNETES_ROLE", KUBERNETES_ROLE),
            agent_id=os.getenv("GL_AGENT_ID", AGENT_ID),
            cache_enabled=os.getenv("VAULT_CACHE_ENABLED", "true").lower() == "true",
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
    renewable: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() >= self.expires_at


class SecretCache:
    """Thread-safe cache for secrets with TTL support."""

    def __init__(self, default_ttl: int = DEFAULT_CACHE_TTL_SECONDS):
        self._cache: Dict[str, CachedSecret] = {}
        self._lock = threading.RLock()
        self._default_ttl = default_ttl
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def get(self, key: str) -> Optional[CachedSecret]:
        with self._lock:
            cached = self._cache.get(key)
            if cached is None:
                self._stats["misses"] += 1
                return None
            if cached.is_expired:
                del self._cache[key]
                self._stats["evictions"] += 1
                self._stats["misses"] += 1
                return None
            self._stats["hits"] += 1
            return cached

    def set(self, key: str, value: Any, secret_type: SecretType = SecretType.GENERIC,
            ttl: Optional[int] = None, **kwargs) -> CachedSecret:
        ttl = ttl or self._default_ttl
        now = datetime.utcnow()
        cached = CachedSecret(
            key=key, value=value, secret_type=secret_type,
            cached_at=now, expires_at=now + timedelta(seconds=ttl), **kwargs
        )
        with self._lock:
            self._cache[key] = cached
        return cached

    def invalidate(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> int:
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {**self._stats, "size": len(self._cache)}


# =============================================================================
# EXCEPTIONS
# =============================================================================

class VaultError(Exception):
    pass

class VaultConnectionError(VaultError):
    pass

class VaultAuthenticationError(VaultError):
    pass

class VaultSecretNotFoundError(VaultError):
    pass

class VaultSecretAccessDenied(VaultError):
    pass


# =============================================================================
# VAULT CLIENT
# =============================================================================

class VaultClient:
    """
    HashiCorp Vault client for GL-004 BURNMASTER Burner Optimization Agent.

    GL-004 Specific Secrets:
        - Flame scanner system credentials
        - Combustion analyzer credentials
        - Air-fuel ratio controller credentials
        - NOx/CO emissions monitoring credentials
        - DCS/PLC communication credentials
    """

    def __init__(self, config: Optional[VaultConfig] = None):
        self._config = config or VaultConfig.from_environment()
        self._cache = SecretCache(self._config.cache_ttl_seconds)
        self._token: Optional[str] = None
        self._token_lock = threading.Lock()

        logger.info(
            f"VaultClient initialized for agent {self._config.agent_id} "
            f"({AGENT_NAME}) using {self._config.auth_method.value} authentication"
        )

    def _get_env_fallback(self, key: str, default: Any = None) -> Any:
        env_key = key.upper().replace("/", "_").replace("-", "_")
        value = os.getenv(env_key)
        if value is not None:
            logger.warning(f"Using environment variable fallback for secret: {key}")
            return value
        return default

    def _get_token(self) -> str:
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

            raise VaultAuthenticationError("No valid authentication method configured")

    def _authenticate_kubernetes(self) -> str:
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

    def get_secret(self, path: str, key: Optional[str] = None,
                   secret_type: SecretType = SecretType.GENERIC, use_cache: bool = True) -> Any:
        full_path = f"{self._config.kv_mount_path}/data/{self._config.secret_base_path}/{self._config.agent_id}/{path}"
        cache_key = f"{full_path}:{key or '*'}"

        if use_cache and self._config.cache_enabled:
            cached = self._cache.get(cache_key)
            if cached is not None:
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
                raise VaultSecretNotFoundError(f"Secret not found: {path}")
            elif response.status_code == 403:
                raise VaultSecretAccessDenied(f"Access denied: {path}")

            response.raise_for_status()
            data = response.json().get("data", {}).get("data", {})
            value = data.get(key) if key else data

            if self._config.cache_enabled:
                self._cache.set(key=cache_key, value=value, secret_type=secret_type)

            return value

        except (VaultSecretNotFoundError, VaultSecretAccessDenied):
            if self._config.fallback_to_env:
                env_value = self._get_env_fallback(path)
                if env_value is not None:
                    return env_value
            raise

        except Exception as e:
            if self._config.fallback_to_env:
                env_value = self._get_env_fallback(path)
                if env_value is not None:
                    return env_value
            raise VaultConnectionError(f"Failed to retrieve secret: {e}") from e

    # =========================================================================
    # GL-004 BURNMASTER SPECIFIC METHODS
    # =========================================================================

    def get_flame_scanner_credentials(self) -> Dict[str, str]:
        """Get flame scanner system credentials."""
        return self.get_secret("integrations/flame-scanner", secret_type=SecretType.FLAME_SCANNER_CREDENTIAL)

    def get_combustion_analyzer_credentials(self) -> Dict[str, str]:
        """Get combustion analyzer credentials."""
        return self.get_secret("integrations/combustion-analyzer", secret_type=SecretType.COMBUSTION_ANALYZER_CREDENTIAL)

    def get_air_fuel_controller_credentials(self) -> Dict[str, str]:
        """Get air-fuel ratio controller credentials."""
        return self.get_secret("integrations/air-fuel-controller", secret_type=SecretType.AIR_FUEL_CONTROLLER_CREDENTIAL)

    def get_emissions_monitor_credentials(self) -> Dict[str, str]:
        """Get NOx/CO emissions monitoring system credentials."""
        return self.get_secret("integrations/emissions-monitor", secret_type=SecretType.EMISSIONS_MONITOR_CREDENTIAL)

    def get_dcs_credentials(self) -> Dict[str, str]:
        """Get DCS/PLC communication credentials."""
        return self.get_secret("integrations/dcs", secret_type=SecretType.DCS_CREDENTIAL)

    def get_historian_credentials(self) -> Dict[str, str]:
        """Get process historian credentials."""
        return self.get_secret("integrations/historian", secret_type=SecretType.HISTORIAN_CREDENTIAL)

    def get_database_credentials(self, role_name: str = "gl-004-postgres") -> Dict[str, str]:
        """Get dynamic database credentials."""
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
            return {"username": data["data"]["username"], "password": data["data"]["password"]}
        except Exception:
            if self._config.fallback_to_env:
                import re
                db_url = os.getenv("DATABASE_URL", "")
                match = re.match(r"[^:]+://([^:]+):([^@]+)@", db_url)
                if match:
                    return {"username": match.group(1), "password": match.group(2)}
            raise

    def get_kafka_credentials(self) -> Dict[str, str]:
        """Get Kafka credentials."""
        return self.get_secret("integrations/kafka", secret_type=SecretType.KAFKA_CREDENTIAL)

    def get_jwt_secret(self) -> str:
        """Get JWT signing secret."""
        return self.get_secret("auth/jwt", key="secret", secret_type=SecretType.JWT_SECRET)

    def get_api_key(self, service_name: str) -> str:
        """Get API key for external service."""
        return self.get_secret(f"api-keys/{service_name}", key="api_key", secret_type=SecretType.API_KEY)

    def invalidate_cache(self, path: Optional[str] = None) -> int:
        if path:
            return 1 if self._cache.invalidate(path) else 0
        return self._cache.clear()

    def health_check(self) -> Dict[str, Any]:
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
    global _global_client
    if _global_client is None:
        with _global_client_lock:
            if _global_client is None:
                _global_client = VaultClient()
    return _global_client


def get_secret_or_env(secret_path: str, env_var: str,
                      vault_client: Optional[VaultClient] = None, default: Any = None) -> Any:
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
    "VaultConfig", "VaultAuthMethod", "SecretType", "VaultClient",
    "SecretCache", "CachedSecret", "VaultError", "VaultConnectionError",
    "VaultAuthenticationError", "VaultSecretNotFoundError", "VaultSecretAccessDenied",
    "get_vault_client", "get_secret_or_env",
]
