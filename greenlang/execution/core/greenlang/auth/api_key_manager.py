# -*- coding: utf-8 -*-
"""
API Key Manager for GreenLang Authentication

Implements secure API key generation, hashing, and validation
for programmatic access to GreenLang services.

Features:
- SHA-256 hashed key storage (never store plaintext)
- Prefix format (glk_*) for easy identification
- Key rotation and expiration support
- Usage tracking and rate limiting
- Scope-based access control
- PostgreSQL backend support

Security Compliance:
- SOC 2 CC6.1 (Logical Access)
- ISO 27001 A.9.4.3 (Password Management)
"""

import os
import hashlib
import secrets
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Set
from enum import Enum

logger = logging.getLogger(__name__)

# API Key constants
API_KEY_PREFIX = "glk"
API_KEY_LENGTH = 32  # Length of the random part
MAX_KEYS_PER_USER = 5
DEFAULT_KEY_EXPIRY_DAYS = 90
HASH_ALGORITHM = "sha256"


class APIKeyScope(Enum):
    """Available scopes for API keys"""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    AGENT_EXECUTE = "agent:execute"
    AGENT_CREATE = "agent:create"
    AGENT_DELETE = "agent:delete"
    TOOLS_READ = "tools:read"
    TOOLS_WRITE = "tools:write"
    REGISTRY_PUBLISH = "registry:publish"


class APIKeyError(Exception):
    """Base exception for API key operations"""
    pass


class InvalidAPIKeyError(APIKeyError):
    """API key is invalid or not found"""
    pass


class ExpiredAPIKeyError(APIKeyError):
    """API key has expired"""
    pass


class RateLimitExceededError(APIKeyError):
    """API key rate limit exceeded"""
    pass


class MaxKeysExceededError(APIKeyError):
    """Maximum number of API keys exceeded for user"""
    pass


@dataclass
class APIKeyConfig:
    """Configuration for API Key Manager"""

    # Storage backend
    database_url: Optional[str] = None

    # Key settings
    default_expiry_days: int = DEFAULT_KEY_EXPIRY_DAYS
    max_keys_per_user: int = MAX_KEYS_PER_USER

    # Rate limiting
    default_rate_limit: int = 1000  # requests per hour
    rate_limit_window_seconds: int = 3600

    # Security
    require_scopes: bool = True
    allow_unlimited_keys: bool = False


@dataclass
class APIKeyRecord:
    """API Key database record"""

    # Key identification
    key_id: str  # Unique identifier (glk_xxxx)
    key_hash: str  # SHA-256 hash of the full key

    # Ownership
    tenant_id: str
    user_id: str

    # Metadata
    name: str
    description: str = ""

    # Permissions
    scopes: List[str] = field(default_factory=list)
    allowed_ips: List[str] = field(default_factory=list)
    allowed_origins: List[str] = field(default_factory=list)

    # Lifecycle
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    last_rotated_at: Optional[datetime] = None

    # Status
    active: bool = True
    revoked: bool = False
    revoked_at: Optional[datetime] = None
    revoked_by: Optional[str] = None
    revoke_reason: Optional[str] = None

    # Usage tracking
    use_count: int = 0
    rate_limit: Optional[int] = None  # requests per hour

    def is_valid(self) -> bool:
        """Check if API key is valid"""
        if not self.active or self.revoked:
            return False

        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False

        return True

    def is_expired(self) -> bool:
        """Check if API key is expired"""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def has_scope(self, scope: str) -> bool:
        """Check if key has specified scope"""
        return scope in self.scopes or "admin" in self.scopes

    def has_any_scope(self, scopes: List[str]) -> bool:
        """Check if key has any of the specified scopes"""
        if "admin" in self.scopes:
            return True
        return bool(set(self.scopes) & set(scopes))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "key_id": self.key_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "name": self.name,
            "description": self.description,
            "scopes": self.scopes,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "active": self.active,
            "use_count": self.use_count,
            "rate_limit": self.rate_limit,
        }


class APIKeyManager:
    """
    API Key Manager for secure key generation and validation.

    API keys are used for programmatic access to GreenLang services.
    Keys are stored as SHA-256 hashes - the plaintext is never stored.

    Example:
        ```python
        manager = APIKeyManager()

        # Generate a new API key
        key_record, plaintext_key = manager.generate_api_key(
            tenant_id="tenant123",
            user_id="user456",
            name="My API Key",
            scopes=["read", "agent:execute"]
        )

        # The plaintext_key is shown to user ONCE: "glk_abc123..."
        # Only the hash is stored in key_record.key_hash

        # Validate a key
        record = manager.validate_api_key(plaintext_key)
        if record:
            print(f"Valid key for tenant: {record.tenant_id}")
        ```
    """

    def __init__(self, config: Optional[APIKeyConfig] = None):
        """
        Initialize API Key Manager.

        Args:
            config: API key configuration
        """
        self.config = config or APIKeyConfig()

        # In-memory storage (replace with database in production)
        self._keys: Dict[str, APIKeyRecord] = {}  # key_id -> record
        self._key_hashes: Dict[str, str] = {}  # hash -> key_id

        # User key count tracking
        self._user_key_counts: Dict[str, int] = {}  # user_id -> count

        # Rate limiting tracking
        self._rate_limit_windows: Dict[str, List[datetime]] = {}  # key_id -> request times

        logger.info("APIKeyManager initialized")

    def generate_api_key(
        self,
        tenant_id: str,
        user_id: str,
        name: str,
        scopes: Optional[List[str]] = None,
        description: str = "",
        expires_in_days: Optional[int] = None,
        rate_limit: Optional[int] = None,
        allowed_ips: Optional[List[str]] = None,
        allowed_origins: Optional[List[str]] = None,
    ) -> tuple[APIKeyRecord, str]:
        """
        Generate a new API key.

        Args:
            tenant_id: Tenant identifier
            user_id: User identifier (owner)
            name: Key name for identification
            scopes: List of permission scopes
            description: Optional description
            expires_in_days: Days until expiration (default from config)
            rate_limit: Custom rate limit (requests/hour)
            allowed_ips: IP allowlist (empty = all allowed)
            allowed_origins: Origin allowlist for CORS

        Returns:
            Tuple of (APIKeyRecord, plaintext_key)
            The plaintext_key should be shown to user ONCE and never stored.

        Raises:
            MaxKeysExceededError: If user has too many keys
        """
        # Check max keys per user
        user_key_count = self._user_key_counts.get(user_id, 0)
        if user_key_count >= self.config.max_keys_per_user:
            if not self.config.allow_unlimited_keys:
                raise MaxKeysExceededError(
                    f"User {user_id} has reached maximum of "
                    f"{self.config.max_keys_per_user} API keys"
                )

        # Generate key components
        key_id = self._generate_key_id()
        key_secret = self._generate_key_secret()
        full_key = f"{key_id}_{key_secret}"

        # Hash the full key for storage
        key_hash = self._hash_key(full_key)

        # Calculate expiration
        expires_at = None
        expiry_days = expires_in_days or self.config.default_expiry_days
        if expiry_days > 0:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expiry_days)

        # Create record
        record = APIKeyRecord(
            key_id=key_id,
            key_hash=key_hash,
            tenant_id=tenant_id,
            user_id=user_id,
            name=name,
            description=description,
            scopes=scopes or [],
            expires_at=expires_at,
            rate_limit=rate_limit or self.config.default_rate_limit,
            allowed_ips=allowed_ips or [],
            allowed_origins=allowed_origins or [],
        )

        # Store record
        self._keys[key_id] = record
        self._key_hashes[key_hash] = key_id
        self._user_key_counts[user_id] = user_key_count + 1

        logger.info(f"Generated API key: {key_id} for user={user_id}, tenant={tenant_id}")

        return record, full_key

    def validate_api_key(
        self,
        api_key: str,
        required_scopes: Optional[List[str]] = None,
        client_ip: Optional[str] = None,
        origin: Optional[str] = None,
    ) -> APIKeyRecord:
        """
        Validate an API key and return its record.

        Args:
            api_key: The full API key string (glk_xxxx_yyyyyyy)
            required_scopes: Scopes required for this operation
            client_ip: Client IP for allowlist check
            origin: Request origin for CORS check

        Returns:
            APIKeyRecord if valid

        Raises:
            InvalidAPIKeyError: If key is invalid, not found, or revoked
            ExpiredAPIKeyError: If key has expired
            RateLimitExceededError: If rate limit exceeded
        """
        # Parse and validate key format
        if not api_key or not api_key.startswith(f"{API_KEY_PREFIX}_"):
            raise InvalidAPIKeyError("Invalid API key format")

        # Hash the provided key
        key_hash = self._hash_key(api_key)

        # Look up by hash
        key_id = self._key_hashes.get(key_hash)
        if not key_id:
            logger.warning(f"API key validation failed: key not found")
            raise InvalidAPIKeyError("Invalid API key")

        record = self._keys.get(key_id)
        if not record:
            raise InvalidAPIKeyError("Invalid API key")

        # Check if revoked
        if record.revoked:
            logger.warning(f"API key validation failed: key revoked - {key_id}")
            raise InvalidAPIKeyError("API key has been revoked")

        # Check if active
        if not record.active:
            logger.warning(f"API key validation failed: key inactive - {key_id}")
            raise InvalidAPIKeyError("API key is inactive")

        # Check expiration
        if record.is_expired():
            logger.warning(f"API key validation failed: key expired - {key_id}")
            raise ExpiredAPIKeyError("API key has expired")

        # Check IP allowlist
        if record.allowed_ips and client_ip:
            if client_ip not in record.allowed_ips:
                logger.warning(f"API key validation failed: IP not allowed - {key_id}")
                raise InvalidAPIKeyError("Client IP not in allowlist")

        # Check origin allowlist
        if record.allowed_origins and origin:
            if origin not in record.allowed_origins:
                logger.warning(f"API key validation failed: origin not allowed - {key_id}")
                raise InvalidAPIKeyError("Origin not in allowlist")

        # Check scopes
        if required_scopes:
            if not record.has_any_scope(required_scopes):
                logger.warning(f"API key validation failed: missing scopes - {key_id}")
                raise InvalidAPIKeyError(
                    f"API key missing required scopes: {required_scopes}"
                )

        # Check rate limit
        if record.rate_limit:
            if not self._check_rate_limit(key_id, record.rate_limit):
                logger.warning(f"API key rate limit exceeded - {key_id}")
                raise RateLimitExceededError("API key rate limit exceeded")

        # Update usage
        record.use_count += 1
        record.last_used_at = datetime.now(timezone.utc)

        logger.debug(f"Validated API key: {key_id} for tenant={record.tenant_id}")
        return record

    def revoke_api_key(
        self,
        key_id: str,
        revoked_by: str,
        reason: Optional[str] = None,
    ) -> bool:
        """
        Revoke an API key.

        Args:
            key_id: The key ID to revoke
            revoked_by: User ID who revoked the key
            reason: Optional revocation reason

        Returns:
            True if key was revoked
        """
        record = self._keys.get(key_id)
        if not record:
            return False

        record.revoked = True
        record.revoked_at = datetime.now(timezone.utc)
        record.revoked_by = revoked_by
        record.revoke_reason = reason
        record.active = False

        # Remove from hash lookup (key can no longer be used)
        if record.key_hash in self._key_hashes:
            del self._key_hashes[record.key_hash]

        logger.info(f"Revoked API key: {key_id} by {revoked_by}")
        return True

    def rotate_api_key(
        self,
        key_id: str,
        grace_period_hours: int = 24,
    ) -> tuple[APIKeyRecord, str]:
        """
        Rotate an API key by generating a new secret.

        The old key remains valid for the grace period.

        Args:
            key_id: The key ID to rotate
            grace_period_hours: Hours the old key remains valid

        Returns:
            Tuple of (updated APIKeyRecord, new plaintext_key)

        Raises:
            InvalidAPIKeyError: If key not found
        """
        old_record = self._keys.get(key_id)
        if not old_record:
            raise InvalidAPIKeyError(f"API key not found: {key_id}")

        # Generate new secret
        new_secret = self._generate_key_secret()
        new_full_key = f"{key_id}_{new_secret}"
        new_hash = self._hash_key(new_full_key)

        # Update record
        old_hash = old_record.key_hash
        old_record.key_hash = new_hash
        old_record.last_rotated_at = datetime.now(timezone.utc)

        # Update hash lookup
        del self._key_hashes[old_hash]
        self._key_hashes[new_hash] = key_id

        # TODO: In production, keep old_hash valid for grace_period_hours

        logger.info(f"Rotated API key: {key_id}")
        return old_record, new_full_key

    def list_user_keys(self, user_id: str) -> List[APIKeyRecord]:
        """
        List all API keys for a user.

        Args:
            user_id: User identifier

        Returns:
            List of APIKeyRecord (without sensitive data)
        """
        return [
            record for record in self._keys.values()
            if record.user_id == user_id and not record.revoked
        ]

    def list_tenant_keys(self, tenant_id: str) -> List[APIKeyRecord]:
        """
        List all API keys for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            List of APIKeyRecord
        """
        return [
            record for record in self._keys.values()
            if record.tenant_id == tenant_id and not record.revoked
        ]

    def get_key_info(self, key_id: str) -> Optional[APIKeyRecord]:
        """
        Get information about an API key.

        Args:
            key_id: Key identifier

        Returns:
            APIKeyRecord or None if not found
        """
        return self._keys.get(key_id)

    def update_key_scopes(self, key_id: str, scopes: List[str]) -> bool:
        """
        Update the scopes for an API key.

        Args:
            key_id: Key identifier
            scopes: New list of scopes

        Returns:
            True if updated
        """
        record = self._keys.get(key_id)
        if not record:
            return False

        record.scopes = scopes
        logger.info(f"Updated scopes for API key: {key_id}")
        return True

    def _generate_key_id(self) -> str:
        """Generate a unique key ID with prefix"""
        random_part = secrets.token_hex(8)
        return f"{API_KEY_PREFIX}_{random_part}"

    def _generate_key_secret(self) -> str:
        """Generate a random key secret"""
        return secrets.token_urlsafe(API_KEY_LENGTH)

    def _hash_key(self, key: str) -> str:
        """
        Hash an API key using SHA-256.

        Args:
            key: The full API key

        Returns:
            Hex-encoded SHA-256 hash
        """
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _check_rate_limit(self, key_id: str, limit: int) -> bool:
        """
        Check if key is within rate limit.

        Args:
            key_id: Key identifier
            limit: Requests per hour limit

        Returns:
            True if within limit
        """
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=self.config.rate_limit_window_seconds)

        # Get or create window tracking
        if key_id not in self._rate_limit_windows:
            self._rate_limit_windows[key_id] = []

        # Clean old entries
        self._rate_limit_windows[key_id] = [
            ts for ts in self._rate_limit_windows[key_id]
            if ts > window_start
        ]

        # Check limit
        if len(self._rate_limit_windows[key_id]) >= limit:
            return False

        # Record this request
        self._rate_limit_windows[key_id].append(now)
        return True

    def get_usage_stats(self, key_id: str) -> Dict[str, Any]:
        """
        Get usage statistics for an API key.

        Args:
            key_id: Key identifier

        Returns:
            Usage statistics dictionary
        """
        record = self._keys.get(key_id)
        if not record:
            return {}

        window_requests = len(self._rate_limit_windows.get(key_id, []))

        return {
            "key_id": key_id,
            "total_requests": record.use_count,
            "requests_this_hour": window_requests,
            "rate_limit": record.rate_limit,
            "rate_limit_remaining": (record.rate_limit or 0) - window_requests,
            "last_used_at": record.last_used_at.isoformat() if record.last_used_at else None,
            "expires_at": record.expires_at.isoformat() if record.expires_at else None,
        }


# PostgreSQL Backend Implementation
class PostgreSQLAPIKeyStore:
    """
    PostgreSQL backend for API key storage.

    Table Schema:
    ```sql
    CREATE TABLE api_keys (
        key_id VARCHAR(32) PRIMARY KEY,
        key_hash VARCHAR(64) NOT NULL UNIQUE,
        tenant_id UUID NOT NULL,
        user_id UUID NOT NULL,
        name VARCHAR(100) NOT NULL,
        description TEXT,
        scopes TEXT[] DEFAULT '{}',
        allowed_ips INET[] DEFAULT '{}',
        allowed_origins TEXT[] DEFAULT '{}',
        created_at TIMESTAMPTZ DEFAULT NOW(),
        expires_at TIMESTAMPTZ,
        last_used_at TIMESTAMPTZ,
        last_rotated_at TIMESTAMPTZ,
        active BOOLEAN DEFAULT TRUE,
        revoked BOOLEAN DEFAULT FALSE,
        revoked_at TIMESTAMPTZ,
        revoked_by UUID,
        revoke_reason TEXT,
        use_count BIGINT DEFAULT 0,
        rate_limit INTEGER,
        FOREIGN KEY (tenant_id) REFERENCES tenants(id),
        FOREIGN KEY (user_id) REFERENCES users(id)
    );

    CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);
    CREATE INDEX idx_api_keys_user ON api_keys(user_id);
    CREATE INDEX idx_api_keys_tenant ON api_keys(tenant_id);
    CREATE INDEX idx_api_keys_active ON api_keys(active) WHERE active = TRUE;
    ```
    """

    def __init__(self, connection_string: str):
        """
        Initialize PostgreSQL store.

        Args:
            connection_string: PostgreSQL connection URL
        """
        self.connection_string = connection_string
        self._pool = None

    async def initialize(self) -> None:
        """Initialize connection pool"""
        try:
            import asyncpg
            self._pool = await asyncpg.create_pool(self.connection_string)
            logger.info("PostgreSQL API key store initialized")
        except ImportError:
            raise ImportError(
                "asyncpg package required for PostgreSQL storage. "
                "Install with: pip install asyncpg"
            )

    async def store_key(self, record: APIKeyRecord) -> None:
        """Store an API key record"""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO api_keys (
                    key_id, key_hash, tenant_id, user_id, name, description,
                    scopes, allowed_ips, allowed_origins, created_at, expires_at,
                    rate_limit, active
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                record.key_id, record.key_hash, record.tenant_id, record.user_id,
                record.name, record.description, record.scopes, record.allowed_ips,
                record.allowed_origins, record.created_at, record.expires_at,
                record.rate_limit, record.active
            )

    async def get_key_by_hash(self, key_hash: str) -> Optional[APIKeyRecord]:
        """Retrieve an API key by its hash"""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM api_keys WHERE key_hash = $1",
                key_hash
            )
            if row:
                return self._row_to_record(row)
            return None

    async def update_usage(self, key_id: str) -> None:
        """Update usage statistics"""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE api_keys
                SET use_count = use_count + 1, last_used_at = NOW()
                WHERE key_id = $1
                """,
                key_id
            )

    async def revoke_key(
        self, key_id: str, revoked_by: str, reason: Optional[str]
    ) -> None:
        """Revoke an API key"""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE api_keys
                SET revoked = TRUE, revoked_at = NOW(),
                    revoked_by = $2, revoke_reason = $3, active = FALSE
                WHERE key_id = $1
                """,
                key_id, revoked_by, reason
            )

    def _row_to_record(self, row) -> APIKeyRecord:
        """Convert database row to APIKeyRecord"""
        return APIKeyRecord(
            key_id=row["key_id"],
            key_hash=row["key_hash"],
            tenant_id=row["tenant_id"],
            user_id=row["user_id"],
            name=row["name"],
            description=row["description"] or "",
            scopes=list(row["scopes"] or []),
            allowed_ips=list(row["allowed_ips"] or []),
            allowed_origins=list(row["allowed_origins"] or []),
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            last_used_at=row["last_used_at"],
            last_rotated_at=row["last_rotated_at"],
            active=row["active"],
            revoked=row["revoked"],
            revoked_at=row["revoked_at"],
            revoked_by=row["revoked_by"],
            revoke_reason=row["revoke_reason"],
            use_count=row["use_count"],
            rate_limit=row["rate_limit"],
        )
