"""
HashiCorp Vault Client for GreenLang Process Heat Platform
===========================================================

TASK-154/155: Secrets Management Implementation

This module provides a high-level Vault client with:
- Kubernetes authentication integration
- Token renewal and caching
- Retry logic with exponential backoff
- Support for multiple secrets engines (KV, Database, AWS, Transit, PKI)

Example:
    >>> from greenlang.infrastructure.secrets import VaultClient, VaultConfig
    >>> config = VaultConfig(addr="https://vault.vault.svc:8200")
    >>> client = VaultClient(config)
    >>> await client.authenticate()
    >>> secret = await client.get_secret("process-heat/database")
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar('T')


class VaultAuthMethod(str, Enum):
    """Supported Vault authentication methods."""
    KUBERNETES = "kubernetes"
    TOKEN = "token"
    APPROLE = "approle"
    AWS_IAM = "aws_iam"


class VaultError(Exception):
    """Base exception for Vault operations."""
    pass


class VaultAuthError(VaultError):
    """Authentication failed."""
    pass


class VaultSecretNotFoundError(VaultError):
    """Secret not found in Vault."""
    pass


class VaultConnectionError(VaultError):
    """Connection to Vault failed."""
    pass


class VaultPermissionError(VaultError):
    """Permission denied for Vault operation."""
    pass


@dataclass
class VaultConfig:
    """Configuration for Vault client."""
    addr: str = field(default_factory=lambda: os.getenv("VAULT_ADDR", "https://vault.vault.svc.cluster.local:8200"))
    namespace: str = field(default_factory=lambda: os.getenv("VAULT_NAMESPACE", ""))
    auth_method: VaultAuthMethod = VaultAuthMethod.KUBERNETES

    # Kubernetes auth
    kubernetes_role: str = field(default_factory=lambda: os.getenv("VAULT_ROLE", "process-heat-api"))
    kubernetes_auth_path: str = "kubernetes"
    service_account_token_path: str = "/var/run/secrets/kubernetes.io/serviceaccount/token"

    # Token auth
    token: Optional[str] = field(default_factory=lambda: os.getenv("VAULT_TOKEN"))

    # AppRole auth
    approle_role_id: Optional[str] = field(default_factory=lambda: os.getenv("VAULT_ROLE_ID"))
    approle_secret_id: Optional[str] = field(default_factory=lambda: os.getenv("VAULT_SECRET_ID"))

    # TLS configuration
    ca_cert_path: Optional[str] = field(default_factory=lambda: os.getenv("VAULT_CACERT", "/vault/tls/ca.crt"))
    skip_verify: bool = field(default_factory=lambda: os.getenv("VAULT_SKIP_VERIFY", "false").lower() == "true")

    # Retry configuration
    max_retries: int = 3
    base_retry_delay: float = 0.5
    max_retry_delay: float = 30.0

    # Cache configuration
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300

    # Token renewal
    token_renewal_threshold: float = 0.75  # Renew when 75% of TTL consumed

    # Timeout configuration
    connect_timeout: float = 10.0
    read_timeout: float = 30.0


@dataclass
class VaultSecret:
    """Represents a secret retrieved from Vault."""
    data: dict[str, Any]
    metadata: dict[str, Any]
    lease_id: Optional[str] = None
    lease_duration: int = 0
    renewable: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_expired(self) -> bool:
        """Check if the secret lease has expired."""
        if self.lease_duration == 0:
            return False
        expiry = self.created_at + timedelta(seconds=self.lease_duration)
        return datetime.utcnow() >= expiry

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the secret data."""
        return self.data.get(key, default)


@dataclass
class DatabaseCredentials:
    """Database credentials from Vault."""
    username: str
    password: str
    host: str
    port: int
    database: str
    ssl_mode: str = "require"
    lease_id: Optional[str] = None
    lease_duration: int = 0

    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}?sslmode={self.ssl_mode}"

    @property
    def async_connection_string(self) -> str:
        """Generate async PostgreSQL connection string."""
        return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}?ssl={self.ssl_mode}"


@dataclass
class AWSCredentials:
    """AWS credentials from Vault."""
    access_key_id: str
    secret_access_key: str
    session_token: Optional[str] = None
    region: str = "us-west-2"
    lease_id: Optional[str] = None
    lease_duration: int = 0

    def to_boto3_session_kwargs(self) -> dict[str, str]:
        """Convert to boto3 session kwargs."""
        kwargs = {
            "aws_access_key_id": self.access_key_id,
            "aws_secret_access_key": self.secret_access_key,
            "region_name": self.region,
        }
        if self.session_token:
            kwargs["aws_session_token"] = self.session_token
        return kwargs


@dataclass
class Certificate:
    """TLS certificate from Vault PKI."""
    certificate: str
    private_key: str
    ca_chain: list[str]
    serial_number: str
    expiration: datetime

    @property
    def is_expired(self) -> bool:
        """Check if the certificate has expired."""
        return datetime.utcnow() >= self.expiration

    @property
    def full_chain(self) -> str:
        """Get the full certificate chain."""
        return self.certificate + "\n" + "\n".join(self.ca_chain)


class SecretCache:
    """In-memory cache for secrets with TTL support."""

    def __init__(self, ttl_seconds: int = 300):
        self._cache: dict[str, tuple[Any, float]] = {}
        self._ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Get a cached value if not expired."""
        async with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]
                if time.time() < expiry:
                    return value
                del self._cache[key]
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a cached value with TTL."""
        async with self._lock:
            expiry = time.time() + (ttl or self._ttl_seconds)
            self._cache[key] = (value, expiry)

    async def delete(self, key: str) -> None:
        """Delete a cached value."""
        async with self._lock:
            self._cache.pop(key, None)

    async def clear(self) -> None:
        """Clear all cached values."""
        async with self._lock:
            self._cache.clear()

    async def cleanup_expired(self) -> int:
        """Remove expired entries and return count of removed entries."""
        async with self._lock:
            now = time.time()
            expired_keys = [k for k, (_, expiry) in self._cache.items() if now >= expiry]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)


class VaultClient:
    """
    High-level HashiCorp Vault client for GreenLang.

    Provides secure access to secrets with:
    - Kubernetes service account authentication
    - Automatic token renewal
    - Retry logic with exponential backoff
    - Secret caching
    - Support for multiple secrets engines

    Example:
        >>> client = VaultClient()
        >>> await client.authenticate()
        >>>
        >>> # Get static secrets
        >>> db_config = await client.get_secret("process-heat/database")
        >>>
        >>> # Get dynamic database credentials
        >>> creds = await client.get_database_credentials("process-heat-readonly")
        >>>
        >>> # Encrypt data
        >>> ciphertext = await client.encrypt_data("process-heat-data-key", b"sensitive data")
    """

    def __init__(self, config: Optional[VaultConfig] = None):
        """Initialize the Vault client."""
        self.config = config or VaultConfig()
        self._token: Optional[str] = None
        self._token_accessor: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._token_ttl: int = 0
        self._cache = SecretCache(self.config.cache_ttl_seconds) if self.config.cache_enabled else None
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()
        self._renewal_task: Optional[asyncio.Task] = None

    async def __aenter__(self) -> VaultClient:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """Establish connection and authenticate with Vault."""
        await self._create_client()
        await self.authenticate()
        self._start_token_renewal()

    async def close(self) -> None:
        """Close the Vault client and cleanup resources."""
        if self._renewal_task:
            self._renewal_task.cancel()
            try:
                await self._renewal_task
            except asyncio.CancelledError:
                pass

        if self._client:
            await self._client.aclose()
            self._client = None

        if self._cache:
            await self._cache.clear()

    async def _create_client(self) -> None:
        """Create the HTTP client for Vault communication."""
        ssl_context = True
        if self.config.ca_cert_path and os.path.exists(self.config.ca_cert_path):
            ssl_context = self.config.ca_cert_path
        elif self.config.skip_verify:
            ssl_context = False

        self._client = httpx.AsyncClient(
            base_url=self.config.addr,
            verify=ssl_context,
            timeout=httpx.Timeout(
                connect=self.config.connect_timeout,
                read=self.config.read_timeout,
                write=self.config.read_timeout,
                pool=self.config.connect_timeout,
            ),
            headers={"X-Vault-Request": "true"},
        )

    async def authenticate(self) -> None:
        """Authenticate with Vault using configured method."""
        if self.config.auth_method == VaultAuthMethod.KUBERNETES:
            await self._kubernetes_auth()
        elif self.config.auth_method == VaultAuthMethod.TOKEN:
            self._token = self.config.token
            if not self._token:
                raise VaultAuthError("Token not provided for token auth method")
            await self._lookup_token()
        elif self.config.auth_method == VaultAuthMethod.APPROLE:
            await self._approle_auth()
        else:
            raise VaultAuthError(f"Unsupported auth method: {self.config.auth_method}")

        logger.info("Successfully authenticated with Vault")

    async def _kubernetes_auth(self) -> None:
        """Authenticate using Kubernetes service account."""
        token_path = Path(self.config.service_account_token_path)
        if not token_path.exists():
            raise VaultAuthError(f"Service account token not found at {token_path}")

        jwt = token_path.read_text().strip()

        response = await self._request(
            "POST",
            f"/v1/auth/{self.config.kubernetes_auth_path}/login",
            json={
                "role": self.config.kubernetes_role,
                "jwt": jwt,
            },
            authenticated=False,
        )

        self._process_auth_response(response)

    async def _approle_auth(self) -> None:
        """Authenticate using AppRole."""
        if not self.config.approle_role_id or not self.config.approle_secret_id:
            raise VaultAuthError("AppRole role_id and secret_id are required")

        response = await self._request(
            "POST",
            "/v1/auth/approle/login",
            json={
                "role_id": self.config.approle_role_id,
                "secret_id": self.config.approle_secret_id,
            },
            authenticated=False,
        )

        self._process_auth_response(response)

    async def _lookup_token(self) -> None:
        """Look up current token to get TTL and accessor."""
        response = await self._request("GET", "/v1/auth/token/lookup-self")
        data = response.get("data", {})
        self._token_accessor = data.get("accessor")
        self._token_ttl = data.get("ttl", 0)
        self._token_expiry = datetime.utcnow() + timedelta(seconds=self._token_ttl)

    def _process_auth_response(self, response: dict[str, Any]) -> None:
        """Process authentication response and extract token info."""
        auth = response.get("auth", {})
        self._token = auth.get("client_token")
        self._token_accessor = auth.get("accessor")
        self._token_ttl = auth.get("lease_duration", 0)

        if self._token_ttl > 0:
            self._token_expiry = datetime.utcnow() + timedelta(seconds=self._token_ttl)
        else:
            self._token_expiry = None

        if not self._token:
            raise VaultAuthError("Failed to obtain token from authentication response")

    def _start_token_renewal(self) -> None:
        """Start background token renewal task."""
        if self._token_ttl > 0:
            self._renewal_task = asyncio.create_task(self._token_renewal_loop())

    async def _token_renewal_loop(self) -> None:
        """Background task to renew token before expiry."""
        while True:
            try:
                if not self._token_expiry or not self._token_ttl:
                    await asyncio.sleep(60)
                    continue

                # Calculate when to renew (at threshold of TTL)
                renew_at = self._token_expiry - timedelta(
                    seconds=self._token_ttl * (1 - self.config.token_renewal_threshold)
                )
                wait_seconds = max(0, (renew_at - datetime.utcnow()).total_seconds())

                if wait_seconds > 0:
                    await asyncio.sleep(wait_seconds)

                await self._renew_token()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Token renewal failed: {e}")
                await asyncio.sleep(30)

    async def _renew_token(self) -> None:
        """Renew the current token."""
        async with self._lock:
            try:
                response = await self._request("POST", "/v1/auth/token/renew-self")
                auth = response.get("auth", {})
                self._token_ttl = auth.get("lease_duration", 0)
                if self._token_ttl > 0:
                    self._token_expiry = datetime.utcnow() + timedelta(seconds=self._token_ttl)
                logger.debug("Token renewed successfully")
            except VaultError as e:
                logger.warning(f"Token renewal failed, re-authenticating: {e}")
                await self.authenticate()

    async def _request(
        self,
        method: str,
        path: str,
        authenticated: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """Make a request to Vault with retry logic."""
        if not self._client:
            await self._create_client()

        headers = kwargs.pop("headers", {})
        if authenticated and self._token:
            headers["X-Vault-Token"] = self._token
        if self.config.namespace:
            headers["X-Vault-Namespace"] = self.config.namespace

        last_error: Optional[Exception] = None

        for attempt in range(self.config.max_retries):
            try:
                response = await self._client.request(
                    method,
                    path,
                    headers=headers,
                    **kwargs,
                )

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 204:
                    return {}
                elif response.status_code == 403:
                    raise VaultPermissionError(f"Permission denied for {path}")
                elif response.status_code == 404:
                    raise VaultSecretNotFoundError(f"Secret not found: {path}")
                elif response.status_code == 503:
                    # Vault is sealed or unavailable
                    raise VaultConnectionError("Vault is sealed or unavailable")
                else:
                    error_data = response.json() if response.content else {}
                    errors = error_data.get("errors", [])
                    raise VaultError(f"Vault request failed: {errors}")

            except httpx.ConnectError as e:
                last_error = VaultConnectionError(f"Failed to connect to Vault: {e}")
            except httpx.TimeoutException as e:
                last_error = VaultConnectionError(f"Vault request timed out: {e}")
            except VaultError:
                raise
            except Exception as e:
                last_error = VaultError(f"Unexpected error: {e}")

            # Exponential backoff
            if attempt < self.config.max_retries - 1:
                delay = min(
                    self.config.base_retry_delay * (2 ** attempt),
                    self.config.max_retry_delay,
                )
                logger.warning(f"Vault request failed, retrying in {delay}s: {last_error}")
                await asyncio.sleep(delay)

        raise last_error or VaultError("Request failed after retries")

    # KV Secrets Engine Operations

    async def get_secret(self, path: str, version: Optional[int] = None) -> VaultSecret:
        """
        Get a secret from the KV v2 secrets engine.

        Args:
            path: Secret path (e.g., "process-heat/database")
            version: Optional specific version to retrieve

        Returns:
            VaultSecret object with secret data
        """
        cache_key = f"kv:{path}:{version}"

        if self._cache:
            cached = await self._cache.get(cache_key)
            if cached:
                return cached

        url = f"/v1/secret/data/{path}"
        if version is not None:
            url += f"?version={version}"

        response = await self._request("GET", url)

        secret = VaultSecret(
            data=response.get("data", {}).get("data", {}),
            metadata=response.get("data", {}).get("metadata", {}),
            lease_id=response.get("lease_id"),
            lease_duration=response.get("lease_duration", 0),
            renewable=response.get("renewable", False),
        )

        if self._cache:
            await self._cache.set(cache_key, secret)

        return secret

    async def put_secret(self, path: str, data: dict[str, Any], cas: Optional[int] = None) -> dict[str, Any]:
        """
        Create or update a secret in the KV v2 secrets engine.

        Args:
            path: Secret path
            data: Secret data to store
            cas: Check-and-set version for optimistic locking

        Returns:
            Metadata about the written secret
        """
        payload: dict[str, Any] = {"data": data}
        if cas is not None:
            payload["options"] = {"cas": cas}

        response = await self._request(
            "POST",
            f"/v1/secret/data/{path}",
            json=payload,
        )

        # Invalidate cache
        if self._cache:
            await self._cache.delete(f"kv:{path}:None")

        return response.get("data", {})

    async def delete_secret(self, path: str, versions: Optional[list[int]] = None) -> None:
        """
        Delete a secret from the KV v2 secrets engine.

        Args:
            path: Secret path
            versions: Optional list of versions to delete (soft delete)
        """
        if versions:
            await self._request(
                "POST",
                f"/v1/secret/delete/{path}",
                json={"versions": versions},
            )
        else:
            await self._request("DELETE", f"/v1/secret/data/{path}")

        # Invalidate cache
        if self._cache:
            await self._cache.delete(f"kv:{path}:None")

    # Database Secrets Engine Operations

    async def get_database_credentials(self, role: str) -> DatabaseCredentials:
        """
        Get dynamic database credentials from Vault.

        Args:
            role: Database role (e.g., "process-heat-readonly")

        Returns:
            DatabaseCredentials with username, password, and connection info
        """
        cache_key = f"db:{role}"

        if self._cache:
            cached = await self._cache.get(cache_key)
            if cached and not cached.is_expired:
                return cached

        response = await self._request("GET", f"/v1/database/creds/{role}")

        data = response.get("data", {})

        # Get static configuration for connection details
        db_config = await self.get_secret("process-heat/database")

        creds = DatabaseCredentials(
            username=data.get("username", ""),
            password=data.get("password", ""),
            host=db_config.get("host", "localhost"),
            port=int(db_config.get("port", 5432)),
            database=db_config.get("database", "process_heat"),
            ssl_mode=db_config.get("ssl_mode", "require"),
            lease_id=response.get("lease_id"),
            lease_duration=response.get("lease_duration", 0),
        )

        if self._cache:
            # Cache for half the lease duration
            ttl = max(60, creds.lease_duration // 2)
            await self._cache.set(cache_key, creds, ttl)

        return creds

    async def renew_lease(self, lease_id: str, increment: Optional[int] = None) -> dict[str, Any]:
        """
        Renew a lease.

        Args:
            lease_id: The lease ID to renew
            increment: Optional lease increment in seconds

        Returns:
            Lease renewal response
        """
        payload: dict[str, Any] = {"lease_id": lease_id}
        if increment is not None:
            payload["increment"] = increment

        return await self._request("POST", "/v1/sys/leases/renew", json=payload)

    async def revoke_lease(self, lease_id: str) -> None:
        """
        Revoke a lease.

        Args:
            lease_id: The lease ID to revoke
        """
        await self._request("POST", "/v1/sys/leases/revoke", json={"lease_id": lease_id})

    # AWS Secrets Engine Operations

    async def get_dynamic_aws_credentials(self, role: str) -> AWSCredentials:
        """
        Get dynamic AWS credentials from Vault.

        Args:
            role: AWS role (e.g., "process-heat-s3")

        Returns:
            AWSCredentials with access keys and session token
        """
        cache_key = f"aws:{role}"

        if self._cache:
            cached = await self._cache.get(cache_key)
            if cached:
                return cached

        response = await self._request("GET", f"/v1/aws/creds/{role}")

        data = response.get("data", {})

        creds = AWSCredentials(
            access_key_id=data.get("access_key", ""),
            secret_access_key=data.get("secret_key", ""),
            session_token=data.get("security_token"),
            lease_id=response.get("lease_id"),
            lease_duration=response.get("lease_duration", 0),
        )

        if self._cache:
            # Cache for half the lease duration
            ttl = max(60, creds.lease_duration // 2)
            await self._cache.set(cache_key, creds, ttl)

        return creds

    async def get_aws_sts_credentials(self, role: str, ttl: str = "1h") -> AWSCredentials:
        """
        Get AWS STS credentials from Vault.

        Args:
            role: AWS STS role
            ttl: TTL for the credentials

        Returns:
            AWSCredentials with STS session credentials
        """
        response = await self._request(
            "POST",
            f"/v1/aws/sts/{role}",
            json={"ttl": ttl},
        )

        data = response.get("data", {})

        return AWSCredentials(
            access_key_id=data.get("access_key", ""),
            secret_access_key=data.get("secret_key", ""),
            session_token=data.get("security_token"),
            lease_id=response.get("lease_id"),
            lease_duration=response.get("lease_duration", 0),
        )

    # Transit Secrets Engine Operations

    async def encrypt_data(self, key_name: str, plaintext: bytes, context: Optional[bytes] = None) -> str:
        """
        Encrypt data using the Transit secrets engine.

        Args:
            key_name: Transit key name
            plaintext: Data to encrypt
            context: Optional context for derived keys

        Returns:
            Ciphertext string (vault:v1:...)
        """
        payload: dict[str, Any] = {
            "plaintext": base64.b64encode(plaintext).decode("utf-8"),
        }
        if context:
            payload["context"] = base64.b64encode(context).decode("utf-8")

        response = await self._request(
            "POST",
            f"/v1/transit/encrypt/{key_name}",
            json=payload,
        )

        return response.get("data", {}).get("ciphertext", "")

    async def decrypt_data(self, key_name: str, ciphertext: str, context: Optional[bytes] = None) -> bytes:
        """
        Decrypt data using the Transit secrets engine.

        Args:
            key_name: Transit key name
            ciphertext: Ciphertext to decrypt (vault:v1:...)
            context: Optional context for derived keys

        Returns:
            Decrypted plaintext bytes
        """
        payload: dict[str, Any] = {"ciphertext": ciphertext}
        if context:
            payload["context"] = base64.b64encode(context).decode("utf-8")

        response = await self._request(
            "POST",
            f"/v1/transit/decrypt/{key_name}",
            json=payload,
        )

        plaintext_b64 = response.get("data", {}).get("plaintext", "")
        return base64.b64decode(plaintext_b64)

    async def encrypt_batch(
        self,
        key_name: str,
        items: list[bytes],
        context: Optional[bytes] = None,
    ) -> list[str]:
        """
        Encrypt multiple items in a single request.

        Args:
            key_name: Transit key name
            items: List of plaintext items to encrypt
            context: Optional context for derived keys

        Returns:
            List of ciphertext strings
        """
        batch_input = []
        for item in items:
            entry: dict[str, str] = {"plaintext": base64.b64encode(item).decode("utf-8")}
            if context:
                entry["context"] = base64.b64encode(context).decode("utf-8")
            batch_input.append(entry)

        response = await self._request(
            "POST",
            f"/v1/transit/encrypt/{key_name}",
            json={"batch_input": batch_input},
        )

        results = response.get("data", {}).get("batch_results", [])
        return [r.get("ciphertext", "") for r in results]

    async def decrypt_batch(
        self,
        key_name: str,
        items: list[str],
        context: Optional[bytes] = None,
    ) -> list[bytes]:
        """
        Decrypt multiple items in a single request.

        Args:
            key_name: Transit key name
            items: List of ciphertext items to decrypt
            context: Optional context for derived keys

        Returns:
            List of decrypted plaintext bytes
        """
        batch_input = []
        for item in items:
            entry: dict[str, str] = {"ciphertext": item}
            if context:
                entry["context"] = base64.b64encode(context).decode("utf-8")
            batch_input.append(entry)

        response = await self._request(
            "POST",
            f"/v1/transit/decrypt/{key_name}",
            json={"batch_input": batch_input},
        )

        results = response.get("data", {}).get("batch_results", [])
        return [base64.b64decode(r.get("plaintext", "")) for r in results]

    async def sign_data(self, key_name: str, data: bytes, hash_algorithm: str = "sha2-256") -> str:
        """
        Sign data using the Transit secrets engine.

        Args:
            key_name: Transit key name
            data: Data to sign
            hash_algorithm: Hash algorithm to use

        Returns:
            Signature string
        """
        response = await self._request(
            "POST",
            f"/v1/transit/sign/{key_name}/{hash_algorithm}",
            json={"input": base64.b64encode(data).decode("utf-8")},
        )

        return response.get("data", {}).get("signature", "")

    async def verify_signature(
        self,
        key_name: str,
        data: bytes,
        signature: str,
        hash_algorithm: str = "sha2-256",
    ) -> bool:
        """
        Verify a signature using the Transit secrets engine.

        Args:
            key_name: Transit key name
            data: Original data
            signature: Signature to verify
            hash_algorithm: Hash algorithm that was used

        Returns:
            True if signature is valid
        """
        response = await self._request(
            "POST",
            f"/v1/transit/verify/{key_name}/{hash_algorithm}",
            json={
                "input": base64.b64encode(data).decode("utf-8"),
                "signature": signature,
            },
        )

        return response.get("data", {}).get("valid", False)

    async def generate_hmac(self, key_name: str, data: bytes, algorithm: str = "sha2-256") -> str:
        """
        Generate HMAC using the Transit secrets engine.

        Args:
            key_name: Transit key name
            data: Data to HMAC
            algorithm: Hash algorithm

        Returns:
            HMAC string
        """
        response = await self._request(
            "POST",
            f"/v1/transit/hmac/{key_name}/{algorithm}",
            json={"input": base64.b64encode(data).decode("utf-8")},
        )

        return response.get("data", {}).get("hmac", "")

    async def verify_hmac(
        self,
        key_name: str,
        data: bytes,
        hmac: str,
        algorithm: str = "sha2-256",
    ) -> bool:
        """
        Verify HMAC using the Transit secrets engine.

        Args:
            key_name: Transit key name
            data: Original data
            hmac: HMAC to verify
            algorithm: Hash algorithm

        Returns:
            True if HMAC is valid
        """
        response = await self._request(
            "POST",
            f"/v1/transit/verify/{key_name}/{algorithm}",
            json={
                "input": base64.b64encode(data).decode("utf-8"),
                "hmac": hmac,
            },
        )

        return response.get("data", {}).get("valid", False)

    # PKI Secrets Engine Operations

    async def generate_certificate(
        self,
        role: str,
        common_name: str,
        alt_names: Optional[list[str]] = None,
        ip_sans: Optional[list[str]] = None,
        ttl: str = "168h",
    ) -> Certificate:
        """
        Generate a TLS certificate from the PKI secrets engine.

        Args:
            role: PKI role (e.g., "process-heat-services")
            common_name: Certificate common name
            alt_names: Optional list of SANs
            ip_sans: Optional list of IP SANs
            ttl: Certificate TTL

        Returns:
            Certificate with cert, key, and CA chain
        """
        payload: dict[str, Any] = {
            "common_name": common_name,
            "ttl": ttl,
        }
        if alt_names:
            payload["alt_names"] = ",".join(alt_names)
        if ip_sans:
            payload["ip_sans"] = ",".join(ip_sans)

        response = await self._request(
            "POST",
            f"/v1/pki_int/issue/{role}",
            json=payload,
        )

        data = response.get("data", {})

        # Parse expiration
        expiration = datetime.utcnow() + timedelta(hours=168)  # Default
        if "expiration" in data:
            expiration = datetime.fromtimestamp(data["expiration"])

        return Certificate(
            certificate=data.get("certificate", ""),
            private_key=data.get("private_key", ""),
            ca_chain=data.get("ca_chain", []),
            serial_number=data.get("serial_number", ""),
            expiration=expiration,
        )

    async def revoke_certificate(self, serial_number: str) -> None:
        """
        Revoke a certificate.

        Args:
            serial_number: Certificate serial number to revoke
        """
        await self._request(
            "POST",
            "/v1/pki_int/revoke",
            json={"serial_number": serial_number},
        )

    # Health Check

    async def health_check(self) -> dict[str, Any]:
        """
        Check Vault health status.

        Returns:
            Health status including initialization and seal status
        """
        try:
            response = await self._request(
                "GET",
                "/v1/sys/health",
                authenticated=False,
            )
            return response
        except VaultError:
            # Health endpoint returns various status codes
            return {"initialized": False, "sealed": True, "standby": False}

    async def is_healthy(self) -> bool:
        """
        Check if Vault is healthy and unsealed.

        Returns:
            True if Vault is initialized, unsealed, and active
        """
        try:
            health = await self.health_check()
            return (
                health.get("initialized", False)
                and not health.get("sealed", True)
            )
        except Exception:
            return False
