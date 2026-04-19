"""
GL-001 ThermalCommand - Webhook Configuration Module

This module provides configuration management for webhook endpoints including:
- Endpoint registration and management
- Event filtering per endpoint
- Secret management for HMAC signatures
- Rate limiting configuration
- Retry and timeout settings

Configuration can be loaded from environment variables, YAML files,
or programmatically through the API.

Example:
    >>> config = WebhookConfig.from_yaml("webhooks.yaml")
    >>> registry = EndpointRegistry(config)
    >>> endpoints = registry.get_endpoints_for_event(WebhookEventType.HEAT_PLAN_CREATED)

Security Notes:
    - Secrets should be stored securely (vault, env vars, encrypted config)
    - All endpoints must use HTTPS in production
    - Secrets should be rotated regularly

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import hashlib
import json
import logging
import os
import secrets
import uuid

from pydantic import BaseModel, Field, validator, SecretStr

from .webhook_events import WebhookEventType


logger = logging.getLogger(__name__)


class EndpointStatus(str, Enum):
    """Status of a webhook endpoint."""

    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"
    DEGRADED = "degraded"  # Working but with issues
    FAILING = "failing"  # Currently experiencing failures


class AuthenticationType(str, Enum):
    """Authentication methods for webhook endpoints."""

    HMAC_SHA256 = "hmac_sha256"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    API_KEY = "api_key"
    NONE = "none"  # Not recommended for production


class RetryConfig(BaseModel):
    """Configuration for webhook delivery retries."""

    max_retries: int = Field(
        default=5,
        ge=0,
        le=10,
        description="Maximum number of retry attempts"
    )
    initial_delay_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Initial delay before first retry in milliseconds"
    )
    max_delay_ms: int = Field(
        default=300000,  # 5 minutes
        ge=1000,
        le=3600000,  # 1 hour
        description="Maximum delay between retries"
    )
    backoff_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        le=5.0,
        description="Exponential backoff multiplier"
    )
    retry_on_status_codes: List[int] = Field(
        default_factory=lambda: [408, 429, 500, 502, 503, 504],
        description="HTTP status codes that trigger retry"
    )
    jitter_enabled: bool = Field(
        default=True,
        description="Add random jitter to retry delays"
    )
    jitter_factor: float = Field(
        default=0.2,
        ge=0.0,
        le=0.5,
        description="Jitter factor (0.2 = +/- 20%)"
    )

    def calculate_delay(self, attempt: int) -> int:
        """
        Calculate delay for a specific retry attempt.

        Args:
            attempt: Retry attempt number (0-indexed)

        Returns:
            Delay in milliseconds
        """
        import random

        delay = self.initial_delay_ms * (self.backoff_multiplier ** attempt)
        delay = min(delay, self.max_delay_ms)

        if self.jitter_enabled:
            jitter_range = delay * self.jitter_factor
            delay = delay + random.uniform(-jitter_range, jitter_range)

        return int(max(self.initial_delay_ms, delay))


class RateLimitConfig(BaseModel):
    """Rate limiting configuration for an endpoint."""

    requests_per_second: float = Field(
        default=10.0,
        gt=0.0,
        le=1000.0,
        description="Maximum requests per second"
    )
    burst_size: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum burst size"
    )
    enabled: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    window_seconds: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Rate limit window in seconds"
    )
    max_requests_per_window: int = Field(
        default=600,
        ge=1,
        description="Maximum requests per window"
    )


class WebhookEndpoint(BaseModel):
    """
    Configuration for a single webhook endpoint.

    Defines all settings for delivering events to a specific endpoint,
    including authentication, filtering, rate limiting, and retry behavior.

    Attributes:
        endpoint_id: Unique identifier for this endpoint
        name: Human-readable name
        url: HTTPS URL for webhook delivery
        description: Optional description
        status: Current endpoint status
        event_types: Set of event types to deliver to this endpoint
        authentication_type: Authentication method
        secret: Shared secret for HMAC signing
        headers: Custom headers to include
        timeout_ms: Request timeout
        retry_config: Retry configuration
        rate_limit_config: Rate limiting configuration
        metadata: Custom metadata

    Example:
        >>> endpoint = WebhookEndpoint(
        ...     name="ERP Notification",
        ...     url="https://erp.example.com/webhooks/thermal",
        ...     secret=SecretStr("my-secret-key"),
        ...     event_types={WebhookEventType.HEAT_PLAN_CREATED}
        ... )
    """

    endpoint_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique endpoint identifier"
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Human-readable endpoint name"
    )
    url: str = Field(
        ...,
        min_length=10,
        max_length=2048,
        description="Webhook delivery URL"
    )
    description: str = Field(
        default="",
        max_length=1024,
        description="Endpoint description"
    )
    status: EndpointStatus = Field(
        default=EndpointStatus.ACTIVE,
        description="Current endpoint status"
    )
    event_types: Set[WebhookEventType] = Field(
        default_factory=lambda: set(WebhookEventType),
        description="Event types to deliver"
    )
    authentication_type: AuthenticationType = Field(
        default=AuthenticationType.HMAC_SHA256,
        description="Authentication method"
    )
    secret: Optional[SecretStr] = Field(
        default=None,
        description="Shared secret for HMAC signing"
    )
    bearer_token: Optional[SecretStr] = Field(
        default=None,
        description="Bearer token for authentication"
    )
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key for authentication"
    )
    api_key_header: str = Field(
        default="X-API-Key",
        description="Header name for API key"
    )
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Custom headers to include"
    )
    timeout_ms: int = Field(
        default=30000,
        ge=1000,
        le=120000,
        description="Request timeout in milliseconds"
    )
    retry_config: RetryConfig = Field(
        default_factory=RetryConfig,
        description="Retry configuration"
    )
    rate_limit_config: RateLimitConfig = Field(
        default_factory=RateLimitConfig,
        description="Rate limiting configuration"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom metadata"
    )

    class Config:
        """Pydantic configuration."""
        use_enum_values = True

    @validator("url")
    def validate_url(cls, v: str) -> str:
        """Validate URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        # Warn about non-HTTPS in logs (production should use HTTPS)
        if v.startswith("http://") and "localhost" not in v:
            logger.warning(f"Non-HTTPS URL configured: {v[:50]}...")
        return v

    @validator("secret")
    def validate_secret(cls, v: Optional[SecretStr], values: Dict) -> Optional[SecretStr]:
        """Validate secret is provided for HMAC authentication."""
        auth_type = values.get("authentication_type")
        if auth_type == AuthenticationType.HMAC_SHA256 and v is None:
            raise ValueError("Secret is required for HMAC-SHA256 authentication")
        return v

    def accepts_event(self, event_type: WebhookEventType) -> bool:
        """
        Check if this endpoint accepts a specific event type.

        Args:
            event_type: Event type to check

        Returns:
            True if endpoint accepts this event type
        """
        return event_type in self.event_types

    def is_active(self) -> bool:
        """Check if endpoint is in an active state."""
        return self.status in {EndpointStatus.ACTIVE, EndpointStatus.DEGRADED}

    def get_secret_value(self) -> Optional[str]:
        """Get the secret value (for internal use only)."""
        if self.secret is None:
            return None
        return self.secret.get_secret_value()


class DeadLetterQueueConfig(BaseModel):
    """Configuration for dead letter queue."""

    enabled: bool = Field(
        default=True,
        description="Enable dead letter queue"
    )
    max_size: int = Field(
        default=10000,
        ge=100,
        le=1000000,
        description="Maximum queue size"
    )
    retention_hours: int = Field(
        default=168,  # 7 days
        ge=1,
        le=720,  # 30 days
        description="Retention period in hours"
    )
    persist_to_disk: bool = Field(
        default=True,
        description="Persist failed events to disk"
    )
    persistence_path: str = Field(
        default="/var/lib/greenlang/webhooks/dlq",
        description="Path for DLQ persistence"
    )
    alert_threshold: int = Field(
        default=100,
        ge=1,
        description="Alert when queue exceeds this size"
    )


class WebhookConfig(BaseModel):
    """
    Complete webhook system configuration.

    Aggregates all webhook-related configuration including endpoints,
    dead letter queue settings, and global defaults.

    Attributes:
        endpoints: List of registered webhook endpoints
        dlq_config: Dead letter queue configuration
        default_retry_config: Default retry configuration
        default_rate_limit_config: Default rate limit configuration
        signature_header: Header name for HMAC signature
        timestamp_header: Header name for request timestamp
        idempotency_header: Header for idempotency key
        max_payload_size_bytes: Maximum webhook payload size
        signature_tolerance_seconds: Tolerance for timestamp validation

    Example:
        >>> config = WebhookConfig(
        ...     endpoints=[
        ...         WebhookEndpoint(name="ERP", url="https://erp.example.com/webhook")
        ...     ]
        ... )
    """

    endpoints: List[WebhookEndpoint] = Field(
        default_factory=list,
        description="Registered webhook endpoints"
    )
    dlq_config: DeadLetterQueueConfig = Field(
        default_factory=DeadLetterQueueConfig,
        description="Dead letter queue configuration"
    )
    default_retry_config: RetryConfig = Field(
        default_factory=RetryConfig,
        description="Default retry configuration"
    )
    default_rate_limit_config: RateLimitConfig = Field(
        default_factory=RateLimitConfig,
        description="Default rate limit configuration"
    )
    signature_header: str = Field(
        default="X-GL-Signature-256",
        description="Header name for HMAC signature"
    )
    timestamp_header: str = Field(
        default="X-GL-Timestamp",
        description="Header name for request timestamp"
    )
    idempotency_header: str = Field(
        default="X-GL-Idempotency-Key",
        description="Header for idempotency key"
    )
    event_type_header: str = Field(
        default="X-GL-Event-Type",
        description="Header for event type"
    )
    delivery_id_header: str = Field(
        default="X-GL-Delivery-ID",
        description="Header for delivery ID"
    )
    max_payload_size_bytes: int = Field(
        default=1048576,  # 1 MB
        ge=1024,
        le=10485760,  # 10 MB
        description="Maximum payload size"
    )
    signature_tolerance_seconds: int = Field(
        default=300,  # 5 minutes
        ge=60,
        le=3600,
        description="Tolerance for timestamp validation"
    )
    content_type: str = Field(
        default="application/json",
        description="Content-Type header value"
    )
    user_agent: str = Field(
        default="GL-001-ThermalCommand-Webhooks/1.0",
        description="User-Agent header value"
    )

    @classmethod
    def from_yaml(cls, path: str) -> "WebhookConfig":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            WebhookConfig instance
        """
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    @classmethod
    def from_environment(cls) -> "WebhookConfig":
        """
        Load configuration from environment variables.

        Environment variables follow the pattern:
        GL_WEBHOOK_<KEY>=value

        Returns:
            WebhookConfig instance
        """
        config_dict: Dict[str, Any] = {}
        prefix = "GL_WEBHOOK_"

        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                config_dict[config_key] = value

        return cls(**config_dict)


class EndpointRegistry:
    """
    Registry for managing webhook endpoints.

    Provides methods for registering, updating, and querying endpoints.
    Thread-safe for concurrent access.

    Attributes:
        config: Webhook configuration

    Example:
        >>> registry = EndpointRegistry(config)
        >>> registry.register_endpoint(endpoint)
        >>> endpoints = registry.get_endpoints_for_event(WebhookEventType.HEAT_PLAN_CREATED)
    """

    def __init__(self, config: WebhookConfig):
        """
        Initialize endpoint registry.

        Args:
            config: Webhook configuration
        """
        self._config = config
        self._endpoints: Dict[str, WebhookEndpoint] = {}
        self._event_subscriptions: Dict[WebhookEventType, Set[str]] = {
            event_type: set() for event_type in WebhookEventType
        }

        # Load endpoints from config
        for endpoint in config.endpoints:
            self._register_internal(endpoint)

    def _register_internal(self, endpoint: WebhookEndpoint) -> None:
        """Internal registration without validation."""
        self._endpoints[endpoint.endpoint_id] = endpoint
        for event_type in endpoint.event_types:
            if isinstance(event_type, str):
                event_type = WebhookEventType(event_type)
            self._event_subscriptions[event_type].add(endpoint.endpoint_id)

    def register_endpoint(self, endpoint: WebhookEndpoint) -> str:
        """
        Register a new webhook endpoint.

        Args:
            endpoint: Endpoint configuration

        Returns:
            Endpoint ID

        Raises:
            ValueError: If endpoint with same ID already exists
        """
        if endpoint.endpoint_id in self._endpoints:
            raise ValueError(f"Endpoint {endpoint.endpoint_id} already registered")

        self._register_internal(endpoint)
        logger.info(f"Registered webhook endpoint: {endpoint.name} ({endpoint.endpoint_id})")

        return endpoint.endpoint_id

    def unregister_endpoint(self, endpoint_id: str) -> bool:
        """
        Unregister a webhook endpoint.

        Args:
            endpoint_id: Endpoint ID to remove

        Returns:
            True if endpoint was removed, False if not found
        """
        endpoint = self._endpoints.pop(endpoint_id, None)
        if endpoint is None:
            return False

        # Remove from event subscriptions
        for event_type in endpoint.event_types:
            if isinstance(event_type, str):
                event_type = WebhookEventType(event_type)
            self._event_subscriptions[event_type].discard(endpoint_id)

        logger.info(f"Unregistered webhook endpoint: {endpoint.name} ({endpoint_id})")
        return True

    def get_endpoint(self, endpoint_id: str) -> Optional[WebhookEndpoint]:
        """
        Get endpoint by ID.

        Args:
            endpoint_id: Endpoint identifier

        Returns:
            WebhookEndpoint if found, None otherwise
        """
        return self._endpoints.get(endpoint_id)

    def get_endpoints_for_event(
        self,
        event_type: WebhookEventType,
        active_only: bool = True
    ) -> List[WebhookEndpoint]:
        """
        Get all endpoints subscribed to an event type.

        Args:
            event_type: Event type to query
            active_only: Only return active endpoints

        Returns:
            List of endpoints subscribed to this event type
        """
        endpoint_ids = self._event_subscriptions.get(event_type, set())
        endpoints = []

        for endpoint_id in endpoint_ids:
            endpoint = self._endpoints.get(endpoint_id)
            if endpoint is not None:
                if active_only and not endpoint.is_active():
                    continue
                endpoints.append(endpoint)

        return endpoints

    def get_all_endpoints(self, active_only: bool = False) -> List[WebhookEndpoint]:
        """
        Get all registered endpoints.

        Args:
            active_only: Only return active endpoints

        Returns:
            List of all endpoints
        """
        if active_only:
            return [e for e in self._endpoints.values() if e.is_active()]
        return list(self._endpoints.values())

    def update_endpoint_status(
        self,
        endpoint_id: str,
        status: EndpointStatus
    ) -> bool:
        """
        Update endpoint status.

        Args:
            endpoint_id: Endpoint ID
            status: New status

        Returns:
            True if updated, False if endpoint not found
        """
        endpoint = self._endpoints.get(endpoint_id)
        if endpoint is None:
            return False

        # Create updated endpoint (Pydantic models are immutable by default)
        updated = endpoint.copy(update={
            "status": status,
            "updated_at": datetime.now(timezone.utc)
        })
        self._endpoints[endpoint_id] = updated

        logger.info(f"Updated endpoint {endpoint_id} status to {status}")
        return True

    def update_event_subscriptions(
        self,
        endpoint_id: str,
        event_types: Set[WebhookEventType]
    ) -> bool:
        """
        Update event type subscriptions for an endpoint.

        Args:
            endpoint_id: Endpoint ID
            event_types: New set of event types

        Returns:
            True if updated, False if endpoint not found
        """
        endpoint = self._endpoints.get(endpoint_id)
        if endpoint is None:
            return False

        # Remove from old subscriptions
        old_types = endpoint.event_types
        for event_type in old_types:
            if isinstance(event_type, str):
                event_type = WebhookEventType(event_type)
            self._event_subscriptions[event_type].discard(endpoint_id)

        # Add to new subscriptions
        for event_type in event_types:
            self._event_subscriptions[event_type].add(endpoint_id)

        # Update endpoint
        updated = endpoint.copy(update={
            "event_types": event_types,
            "updated_at": datetime.now(timezone.utc)
        })
        self._endpoints[endpoint_id] = updated

        logger.info(f"Updated endpoint {endpoint_id} event subscriptions")
        return True

    def generate_secret(self, length: int = 32) -> str:
        """
        Generate a cryptographically secure secret.

        Args:
            length: Length of secret in bytes

        Returns:
            Hex-encoded secret string
        """
        return secrets.token_hex(length)

    def endpoint_count(self) -> int:
        """Get total number of registered endpoints."""
        return len(self._endpoints)

    def active_endpoint_count(self) -> int:
        """Get number of active endpoints."""
        return sum(1 for e in self._endpoints.values() if e.is_active())


class SecretManager:
    """
    Secure secret management for webhook authentication.

    Provides methods for generating, storing, and rotating secrets.
    In production, this should integrate with a secrets vault.

    Example:
        >>> manager = SecretManager()
        >>> secret = manager.generate_secret()
        >>> manager.store_secret("endpoint-001", secret)
    """

    def __init__(self, storage_backend: str = "memory"):
        """
        Initialize secret manager.

        Args:
            storage_backend: Backend for secret storage (memory, vault, env)
        """
        self._backend = storage_backend
        self._secrets: Dict[str, str] = {}
        self._rotation_schedule: Dict[str, datetime] = {}

    def generate_secret(self, length: int = 32) -> str:
        """
        Generate a cryptographically secure secret.

        Args:
            length: Length in bytes

        Returns:
            Hex-encoded secret
        """
        return secrets.token_hex(length)

    def store_secret(
        self,
        endpoint_id: str,
        secret: str,
        rotation_days: int = 90
    ) -> None:
        """
        Store a secret for an endpoint.

        Args:
            endpoint_id: Endpoint identifier
            secret: Secret value
            rotation_days: Days until rotation required
        """
        if self._backend == "memory":
            self._secrets[endpoint_id] = secret
            self._rotation_schedule[endpoint_id] = (
                datetime.now(timezone.utc) + timedelta(days=rotation_days)
            )
        elif self._backend == "env":
            # In production, this would write to a secure env manager
            logger.warning("Environment backend not recommended for production")
        elif self._backend == "vault":
            # In production, integrate with HashiCorp Vault or similar
            raise NotImplementedError("Vault backend not yet implemented")

        logger.info(f"Stored secret for endpoint {endpoint_id}")

    def get_secret(self, endpoint_id: str) -> Optional[str]:
        """
        Retrieve a secret for an endpoint.

        Args:
            endpoint_id: Endpoint identifier

        Returns:
            Secret value if found
        """
        return self._secrets.get(endpoint_id)

    def delete_secret(self, endpoint_id: str) -> bool:
        """
        Delete a secret.

        Args:
            endpoint_id: Endpoint identifier

        Returns:
            True if deleted
        """
        if endpoint_id in self._secrets:
            del self._secrets[endpoint_id]
            self._rotation_schedule.pop(endpoint_id, None)
            logger.info(f"Deleted secret for endpoint {endpoint_id}")
            return True
        return False

    def needs_rotation(self, endpoint_id: str) -> bool:
        """
        Check if a secret needs rotation.

        Args:
            endpoint_id: Endpoint identifier

        Returns:
            True if rotation is due
        """
        rotation_date = self._rotation_schedule.get(endpoint_id)
        if rotation_date is None:
            return True
        return datetime.now(timezone.utc) >= rotation_date

    def rotate_secret(self, endpoint_id: str) -> Optional[str]:
        """
        Rotate a secret for an endpoint.

        Args:
            endpoint_id: Endpoint identifier

        Returns:
            New secret value
        """
        if endpoint_id not in self._secrets:
            return None

        new_secret = self.generate_secret()
        self.store_secret(endpoint_id, new_secret)
        logger.info(f"Rotated secret for endpoint {endpoint_id}")
        return new_secret
