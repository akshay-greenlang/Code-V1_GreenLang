# -*- coding: utf-8 -*-
"""
GreenLang Orchestrator Alert Webhooks
=====================================

Core webhook alerting module for operational incident response (FR-063).

This module implements:
- AlertType and AlertSeverity enums for categorizing alerts
- AlertPayload and WebhookConfig Pydantic models
- WebhookManager for dispatching alerts with retry logic
- AlertManager for integration with orchestrator
- HMAC-SHA256 payload signing for webhook verification

Author: GreenLang Team
Version: 1.0.0
GL-FOUND-X-001: Alert Webhooks Integration
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

import httpx
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class AlertType(str, Enum):
    """Types of alerts that can be triggered by the orchestrator."""

    RUN_FAILED = "run_failed"
    STEP_TIMEOUT = "step_timeout"
    POLICY_DENIAL = "policy_denial"
    SLO_BREACH = "slo_breach"
    RUN_SUCCEEDED = "run_succeeded"


class AlertSeverity(str, Enum):
    """Severity levels for alerts, ordered from most to least severe."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @classmethod
    def from_string(cls, value: str) -> "AlertSeverity":
        """Parse severity from string, case-insensitive."""
        return cls(value.lower())

    def __ge__(self, other: "AlertSeverity") -> bool:
        """Compare severity levels (CRITICAL >= HIGH >= MEDIUM >= LOW >= INFO)."""
        order = [cls.INFO, cls.LOW, cls.MEDIUM, cls.HIGH, cls.CRITICAL]
        return order.index(self) >= order.index(other)

    def __gt__(self, other: "AlertSeverity") -> bool:
        """Compare severity levels."""
        order = [cls.INFO, cls.LOW, cls.MEDIUM, cls.HIGH, cls.CRITICAL]
        return order.index(self) > order.index(other)

    def __le__(self, other: "AlertSeverity") -> bool:
        """Compare severity levels."""
        return not self.__gt__(other)

    def __lt__(self, other: "AlertSeverity") -> bool:
        """Compare severity levels."""
        return not self.__ge__(other)


class WebhookDeliveryStatus(str, Enum):
    """Status of a webhook delivery attempt."""

    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"


# =============================================================================
# MODELS
# =============================================================================


class AlertPayload(BaseModel):
    """
    Payload for an alert notification.

    Contains all relevant information about an operational event that
    triggered the alert.

    Attributes:
        alert_id: Unique identifier for this alert instance
        alert_type: Type of alert (RUN_FAILED, STEP_TIMEOUT, etc.)
        severity: Severity level of the alert
        run_id: Pipeline run ID that triggered the alert
        step_id: Optional step ID if alert is step-specific
        pipeline_id: Optional pipeline ID for context
        namespace: Namespace where the alert originated
        message: Human-readable alert message
        details: Additional structured details about the alert
        timestamp: When the alert was generated
        source: Source system/component that generated the alert
    """

    alert_id: str = Field(
        default_factory=lambda: f"alert-{uuid4().hex[:12]}",
        description="Unique alert identifier",
    )
    alert_type: AlertType = Field(..., description="Type of alert")
    severity: AlertSeverity = Field(..., description="Alert severity level")
    run_id: str = Field(..., description="Pipeline run ID")
    step_id: Optional[str] = Field(None, description="Step ID if applicable")
    pipeline_id: Optional[str] = Field(None, description="Pipeline ID")
    namespace: str = Field(default="default", description="Namespace")
    message: str = Field(..., description="Human-readable alert message")
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Additional alert details"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Alert timestamp",
    )
    source: str = Field(
        default="greenlang-orchestrator", description="Alert source system"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with ISO timestamp."""
        data = self.model_dump()
        data["timestamp"] = self.timestamp.isoformat()
        data["alert_type"] = self.alert_type.value
        data["severity"] = self.severity.value
        return data

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)


class WebhookConfig(BaseModel):
    """
    Configuration for a webhook endpoint.

    Attributes:
        webhook_id: Unique identifier for this webhook configuration
        name: Human-readable name for the webhook
        provider: Provider type (slack, discord, pagerduty, custom)
        url: Webhook endpoint URL (can use env var syntax ${VAR})
        secret: Optional secret for HMAC signing (can use env var syntax)
        routing_key: Optional routing key (for PagerDuty)
        events: List of event types this webhook subscribes to
        severity_threshold: Minimum severity to trigger this webhook
        retries: Number of retry attempts on failure
        timeout_seconds: HTTP request timeout
        enabled: Whether this webhook is active
        headers: Additional headers to include in requests
        metadata: Additional webhook metadata
    """

    webhook_id: str = Field(
        default_factory=lambda: f"webhook-{uuid4().hex[:8]}",
        description="Unique webhook identifier",
    )
    name: str = Field(..., description="Webhook name")
    provider: str = Field(default="custom", description="Provider type")
    url: Optional[str] = Field(None, description="Webhook URL")
    secret: Optional[str] = Field(None, description="HMAC signing secret")
    routing_key: Optional[str] = Field(None, description="Routing key (PagerDuty)")
    events: List[str] = Field(
        default_factory=list, description="Subscribed event types"
    )
    severity_threshold: AlertSeverity = Field(
        default=AlertSeverity.MEDIUM, description="Minimum severity threshold"
    )
    retries: int = Field(default=3, ge=0, le=10, description="Retry attempts")
    timeout_seconds: int = Field(default=30, ge=1, le=300, description="Request timeout")
    enabled: bool = Field(default=True, description="Whether webhook is enabled")
    headers: Dict[str, str] = Field(
        default_factory=dict, description="Additional HTTP headers"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("events", mode="before")
    @classmethod
    def normalize_events(cls, v: Any) -> List[str]:
        """Normalize event types to lowercase."""
        if isinstance(v, list):
            return [e.lower() if isinstance(e, str) else e for e in v]
        return v

    def resolve_url(self) -> Optional[str]:
        """Resolve URL, expanding environment variables."""
        if not self.url:
            return None
        return _expand_env_vars(self.url)

    def resolve_secret(self) -> Optional[str]:
        """Resolve secret, expanding environment variables."""
        if not self.secret:
            return None
        return _expand_env_vars(self.secret)

    def resolve_routing_key(self) -> Optional[str]:
        """Resolve routing key, expanding environment variables."""
        if not self.routing_key:
            return None
        return _expand_env_vars(self.routing_key)

    def subscribes_to(self, alert_type: AlertType) -> bool:
        """Check if this webhook subscribes to the given alert type."""
        if not self.events:
            return True  # Subscribe to all events if none specified
        return alert_type.value in self.events

    def meets_severity_threshold(self, severity: AlertSeverity) -> bool:
        """Check if severity meets the threshold for this webhook."""
        return severity >= self.severity_threshold


class WebhookDeliveryResult(BaseModel):
    """
    Result of a webhook delivery attempt.

    Attributes:
        webhook_id: ID of the webhook that was called
        alert_id: ID of the alert that was delivered
        status: Delivery status
        status_code: HTTP status code (if applicable)
        attempts: Number of delivery attempts made
        delivered_at: Timestamp of successful delivery
        error_message: Error message if delivery failed
        latency_ms: Request latency in milliseconds
    """

    webhook_id: str = Field(..., description="Webhook ID")
    alert_id: str = Field(..., description="Alert ID")
    status: WebhookDeliveryStatus = Field(..., description="Delivery status")
    status_code: Optional[int] = Field(None, description="HTTP status code")
    attempts: int = Field(default=1, description="Delivery attempts")
    delivered_at: Optional[datetime] = Field(None, description="Delivery timestamp")
    error_message: Optional[str] = Field(None, description="Error message")
    latency_ms: Optional[float] = Field(None, description="Request latency")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _expand_env_vars(value: str) -> str:
    """
    Expand environment variables in a string.

    Supports ${VAR_NAME} syntax for environment variable expansion.

    Args:
        value: String potentially containing ${VAR} patterns

    Returns:
        String with environment variables expanded
    """
    import re

    def replace_env(match: re.Match) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, "")

    return re.sub(r"\$\{([^}]+)\}", replace_env, value)


def compute_hmac_signature(
    payload: str, secret: str, algorithm: str = "sha256"
) -> str:
    """
    Compute HMAC signature for webhook payload verification.

    Args:
        payload: JSON payload string to sign
        secret: HMAC secret key
        algorithm: Hash algorithm (default: sha256)

    Returns:
        Hex-encoded HMAC signature
    """
    key = secret.encode("utf-8")
    message = payload.encode("utf-8")
    signature = hmac.new(key, message, hashlib.sha256).hexdigest()
    return f"sha256={signature}"


# =============================================================================
# WEBHOOK MANAGER
# =============================================================================


class WebhookManager:
    """
    Manages webhook registrations and alert dispatching.

    Provides:
    - Webhook registration per namespace
    - Alert dispatching with retry logic
    - HMAC-SHA256 payload signing
    - Delivery status tracking
    - Provider-specific formatting

    Example:
        >>> manager = WebhookManager()
        >>> config = WebhookConfig(name="slack", url="https://hooks.slack.com/...")
        >>> manager.register_webhook("production", config)
        >>> await manager.dispatch_alert("production", alert_payload)
    """

    def __init__(
        self,
        default_timeout: int = 30,
        default_retries: int = 3,
        base_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 60.0,
    ):
        """
        Initialize WebhookManager.

        Args:
            default_timeout: Default HTTP timeout in seconds
            default_retries: Default number of retry attempts
            base_backoff_seconds: Initial backoff delay for retries
            max_backoff_seconds: Maximum backoff delay
        """
        self._webhooks: Dict[str, Dict[str, WebhookConfig]] = {}
        self._delivery_history: List[WebhookDeliveryResult] = []
        self._providers: Dict[str, Callable] = {}
        self._default_timeout = default_timeout
        self._default_retries = default_retries
        self._base_backoff = base_backoff_seconds
        self._max_backoff = max_backoff_seconds
        self._http_client: Optional[httpx.AsyncClient] = None

        logger.info("WebhookManager initialized")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._default_timeout),
                follow_redirects=True,
            )
        return self._http_client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    def register_provider(
        self, provider_name: str, formatter: Callable[[AlertPayload, WebhookConfig], Dict[str, Any]]
    ) -> None:
        """
        Register a custom provider formatter.

        Args:
            provider_name: Name of the provider (e.g., "slack", "discord")
            formatter: Function that formats AlertPayload for the provider
        """
        self._providers[provider_name.lower()] = formatter
        logger.info(f"Registered provider formatter: {provider_name}")

    def register_webhook(
        self, namespace: str, config: WebhookConfig
    ) -> str:
        """
        Register a webhook for a namespace.

        Args:
            namespace: Namespace to register the webhook for
            config: Webhook configuration

        Returns:
            Webhook ID
        """
        if namespace not in self._webhooks:
            self._webhooks[namespace] = {}

        self._webhooks[namespace][config.webhook_id] = config
        logger.info(
            f"Registered webhook '{config.name}' (ID: {config.webhook_id}) "
            f"for namespace '{namespace}'"
        )
        return config.webhook_id

    def unregister_webhook(self, namespace: str, webhook_id: str) -> bool:
        """
        Unregister a webhook from a namespace.

        Args:
            namespace: Namespace the webhook is registered in
            webhook_id: ID of the webhook to remove

        Returns:
            True if webhook was removed, False if not found
        """
        if namespace in self._webhooks and webhook_id in self._webhooks[namespace]:
            del self._webhooks[namespace][webhook_id]
            logger.info(f"Unregistered webhook '{webhook_id}' from namespace '{namespace}'")
            return True
        return False

    def get_webhook(self, namespace: str, webhook_id: str) -> Optional[WebhookConfig]:
        """Get a specific webhook configuration."""
        return self._webhooks.get(namespace, {}).get(webhook_id)

    def list_webhooks(self, namespace: str) -> List[WebhookConfig]:
        """List all webhooks for a namespace."""
        return list(self._webhooks.get(namespace, {}).values())

    def get_all_namespaces(self) -> List[str]:
        """Get all namespaces with registered webhooks."""
        return list(self._webhooks.keys())

    async def dispatch_alert(
        self, namespace: str, alert: AlertPayload
    ) -> List[WebhookDeliveryResult]:
        """
        Dispatch an alert to all applicable webhooks in a namespace.

        Args:
            namespace: Namespace to dispatch to
            alert: Alert payload to send

        Returns:
            List of delivery results for each webhook
        """
        webhooks = self._webhooks.get(namespace, {})
        results = []

        # Filter applicable webhooks
        applicable_webhooks = [
            wh for wh in webhooks.values()
            if wh.enabled
            and wh.subscribes_to(alert.alert_type)
            and wh.meets_severity_threshold(alert.severity)
        ]

        if not applicable_webhooks:
            logger.debug(
                f"No applicable webhooks for alert {alert.alert_id} "
                f"in namespace '{namespace}'"
            )
            return results

        # Dispatch to all applicable webhooks concurrently
        tasks = [
            self._deliver_to_webhook(webhook, alert)
            for webhook in applicable_webhooks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                webhook = applicable_webhooks[i]
                processed_results.append(
                    WebhookDeliveryResult(
                        webhook_id=webhook.webhook_id,
                        alert_id=alert.alert_id,
                        status=WebhookDeliveryStatus.FAILED,
                        error_message=str(result),
                    )
                )
            else:
                processed_results.append(result)

        # Store delivery history
        self._delivery_history.extend(processed_results)

        return processed_results

    async def _deliver_to_webhook(
        self, webhook: WebhookConfig, alert: AlertPayload
    ) -> WebhookDeliveryResult:
        """
        Deliver an alert to a single webhook with retry logic.

        Args:
            webhook: Webhook configuration
            alert: Alert to deliver

        Returns:
            Delivery result
        """
        start_time = time.time()
        attempts = 0
        max_attempts = webhook.retries + 1
        last_error: Optional[str] = None
        last_status_code: Optional[int] = None

        # Format payload based on provider
        formatted_payload = self._format_payload(webhook, alert)
        payload_json = json.dumps(formatted_payload, sort_keys=True)

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "GreenLang-Orchestrator/1.0",
            "X-GreenLang-Alert-ID": alert.alert_id,
            "X-GreenLang-Alert-Type": alert.alert_type.value,
            **webhook.headers,
        }

        # Add HMAC signature if secret is configured
        secret = webhook.resolve_secret()
        if secret:
            signature = compute_hmac_signature(payload_json, secret)
            headers["X-GreenLang-Signature"] = signature

        # Resolve URL
        url = webhook.resolve_url()
        if not url:
            return WebhookDeliveryResult(
                webhook_id=webhook.webhook_id,
                alert_id=alert.alert_id,
                status=WebhookDeliveryStatus.FAILED,
                error_message="Webhook URL not configured or could not be resolved",
            )

        client = await self._get_client()

        while attempts < max_attempts:
            attempts += 1
            try:
                response = await client.post(
                    url,
                    content=payload_json,
                    headers=headers,
                    timeout=webhook.timeout_seconds,
                )

                last_status_code = response.status_code

                if 200 <= response.status_code < 300:
                    latency_ms = (time.time() - start_time) * 1000
                    logger.info(
                        f"Alert {alert.alert_id} delivered to webhook "
                        f"{webhook.webhook_id} (status: {response.status_code}, "
                        f"latency: {latency_ms:.1f}ms)"
                    )
                    return WebhookDeliveryResult(
                        webhook_id=webhook.webhook_id,
                        alert_id=alert.alert_id,
                        status=WebhookDeliveryStatus.DELIVERED,
                        status_code=response.status_code,
                        attempts=attempts,
                        delivered_at=datetime.now(timezone.utc),
                        latency_ms=latency_ms,
                    )

                # Non-success status code
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                logger.warning(
                    f"Webhook delivery failed (attempt {attempts}/{max_attempts}): "
                    f"{last_error}"
                )

            except httpx.TimeoutException as e:
                last_error = f"Timeout: {str(e)}"
                logger.warning(
                    f"Webhook delivery timeout (attempt {attempts}/{max_attempts}): "
                    f"{last_error}"
                )

            except httpx.RequestError as e:
                last_error = f"Request error: {str(e)}"
                logger.warning(
                    f"Webhook delivery error (attempt {attempts}/{max_attempts}): "
                    f"{last_error}"
                )

            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                logger.error(
                    f"Webhook delivery failed unexpectedly (attempt {attempts}/{max_attempts}): "
                    f"{last_error}",
                    exc_info=True,
                )

            # Exponential backoff before retry
            if attempts < max_attempts:
                backoff = min(
                    self._base_backoff * (2 ** (attempts - 1)),
                    self._max_backoff,
                )
                await asyncio.sleep(backoff)

        # All retries exhausted
        latency_ms = (time.time() - start_time) * 1000
        logger.error(
            f"Alert {alert.alert_id} delivery to webhook {webhook.webhook_id} "
            f"failed after {attempts} attempts: {last_error}"
        )
        return WebhookDeliveryResult(
            webhook_id=webhook.webhook_id,
            alert_id=alert.alert_id,
            status=WebhookDeliveryStatus.FAILED,
            status_code=last_status_code,
            attempts=attempts,
            error_message=last_error,
            latency_ms=latency_ms,
        )

    def _format_payload(
        self, webhook: WebhookConfig, alert: AlertPayload
    ) -> Dict[str, Any]:
        """
        Format alert payload based on provider.

        Args:
            webhook: Webhook configuration
            alert: Alert to format

        Returns:
            Formatted payload dictionary
        """
        provider = webhook.provider.lower()

        # Check for custom provider formatter
        if provider in self._providers:
            return self._providers[provider](alert, webhook)

        # Default to raw alert payload
        return alert.to_dict()

    def get_delivery_history(
        self,
        webhook_id: Optional[str] = None,
        alert_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[WebhookDeliveryResult]:
        """
        Get delivery history with optional filtering.

        Args:
            webhook_id: Filter by webhook ID
            alert_id: Filter by alert ID
            limit: Maximum number of results

        Returns:
            List of delivery results
        """
        results = self._delivery_history

        if webhook_id:
            results = [r for r in results if r.webhook_id == webhook_id]
        if alert_id:
            results = [r for r in results if r.alert_id == alert_id]

        return results[-limit:]

    def get_failure_count(self, namespace: str, webhook_id: str) -> int:
        """Get number of recent failures for a webhook."""
        results = [
            r for r in self._delivery_history[-100:]
            if r.webhook_id == webhook_id and r.status == WebhookDeliveryStatus.FAILED
        ]
        return len(results)


# =============================================================================
# ALERT MANAGER
# =============================================================================


class AlertManager:
    """
    High-level alert manager for orchestrator integration.

    Provides a simplified interface for emitting alerts and managing
    webhook configurations across namespaces.

    Example:
        >>> alert_manager = AlertManager()
        >>> await alert_manager.load_config("config/alerting.yaml")
        >>> await alert_manager.emit_run_failed(
        ...     namespace="production",
        ...     run_id="run-123",
        ...     error_message="Step 'calculate' timed out",
        ... )
    """

    def __init__(self, webhook_manager: Optional[WebhookManager] = None):
        """
        Initialize AlertManager.

        Args:
            webhook_manager: Optional WebhookManager instance (creates new if not provided)
        """
        self._webhook_manager = webhook_manager or WebhookManager()
        self._enabled = True
        self._default_severity_threshold = AlertSeverity.MEDIUM
        self._config_loaded = False

        logger.info("AlertManager initialized")

    @property
    def webhook_manager(self) -> WebhookManager:
        """Get the underlying WebhookManager."""
        return self._webhook_manager

    @property
    def enabled(self) -> bool:
        """Check if alerting is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable alerting."""
        self._enabled = value
        logger.info(f"Alerting {'enabled' if value else 'disabled'}")

    async def load_config(self, config_path: str) -> None:
        """
        Load alerting configuration from YAML file.

        Args:
            config_path: Path to alerting.yaml configuration file
        """
        import yaml

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            alerting_config = config.get("alerting", {})

            # Global settings
            self._enabled = alerting_config.get("enabled", True)
            threshold_str = alerting_config.get("default_severity_threshold", "medium")
            self._default_severity_threshold = AlertSeverity.from_string(threshold_str)

            # Load namespace configurations
            namespaces = alerting_config.get("namespaces", {})
            for namespace, ns_config in namespaces.items():
                webhooks = ns_config.get("webhooks", [])
                for wh_config in webhooks:
                    # Parse severity threshold
                    threshold = wh_config.get(
                        "severity_threshold",
                        self._default_severity_threshold.value,
                    )
                    if isinstance(threshold, str):
                        threshold = AlertSeverity.from_string(threshold)

                    webhook = WebhookConfig(
                        name=wh_config.get("name", "unnamed"),
                        provider=wh_config.get("provider", "custom"),
                        url=wh_config.get("url"),
                        secret=wh_config.get("secret"),
                        routing_key=wh_config.get("routing_key"),
                        events=wh_config.get("events", []),
                        severity_threshold=threshold,
                        retries=wh_config.get("retries", 3),
                        timeout_seconds=wh_config.get("timeout_seconds", 30),
                        enabled=wh_config.get("enabled", True),
                        headers=wh_config.get("headers", {}),
                        metadata=wh_config.get("metadata", {}),
                    )
                    self._webhook_manager.register_webhook(namespace, webhook)

            self._config_loaded = True
            logger.info(f"Loaded alerting configuration from {config_path}")

        except FileNotFoundError:
            logger.warning(f"Alerting config file not found: {config_path}")
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse alerting config: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load alerting config: {e}", exc_info=True)
            raise

    async def emit_alert(
        self,
        namespace: str,
        alert_type: AlertType,
        severity: AlertSeverity,
        run_id: str,
        message: str,
        step_id: Optional[str] = None,
        pipeline_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> List[WebhookDeliveryResult]:
        """
        Emit an alert to all applicable webhooks.

        Args:
            namespace: Namespace where the alert originated
            alert_type: Type of alert
            severity: Alert severity
            run_id: Pipeline run ID
            message: Human-readable alert message
            step_id: Optional step ID
            pipeline_id: Optional pipeline ID
            details: Optional additional details

        Returns:
            List of delivery results
        """
        if not self._enabled:
            logger.debug("Alerting disabled, skipping alert emission")
            return []

        alert = AlertPayload(
            alert_type=alert_type,
            severity=severity,
            run_id=run_id,
            step_id=step_id,
            pipeline_id=pipeline_id,
            namespace=namespace,
            message=message,
            details=details or {},
        )

        logger.info(
            f"Emitting {severity.value} alert [{alert_type.value}] "
            f"for run {run_id}: {message}"
        )

        return await self._webhook_manager.dispatch_alert(namespace, alert)

    async def emit_run_failed(
        self,
        namespace: str,
        run_id: str,
        error_message: str,
        pipeline_id: Optional[str] = None,
        step_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> List[WebhookDeliveryResult]:
        """Emit a RUN_FAILED alert."""
        return await self.emit_alert(
            namespace=namespace,
            alert_type=AlertType.RUN_FAILED,
            severity=AlertSeverity.HIGH,
            run_id=run_id,
            message=f"Pipeline run failed: {error_message}",
            step_id=step_id,
            pipeline_id=pipeline_id,
            details={"error": error_message, **(details or {})},
        )

    async def emit_step_timeout(
        self,
        namespace: str,
        run_id: str,
        step_id: str,
        timeout_seconds: int,
        pipeline_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> List[WebhookDeliveryResult]:
        """Emit a STEP_TIMEOUT alert."""
        return await self.emit_alert(
            namespace=namespace,
            alert_type=AlertType.STEP_TIMEOUT,
            severity=AlertSeverity.HIGH,
            run_id=run_id,
            message=f"Step '{step_id}' timed out after {timeout_seconds}s",
            step_id=step_id,
            pipeline_id=pipeline_id,
            details={"timeout_seconds": timeout_seconds, **(details or {})},
        )

    async def emit_policy_denial(
        self,
        namespace: str,
        run_id: str,
        policy_name: str,
        violation_message: str,
        pipeline_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> List[WebhookDeliveryResult]:
        """Emit a POLICY_DENIAL alert."""
        return await self.emit_alert(
            namespace=namespace,
            alert_type=AlertType.POLICY_DENIAL,
            severity=AlertSeverity.MEDIUM,
            run_id=run_id,
            message=f"Policy '{policy_name}' denied execution: {violation_message}",
            pipeline_id=pipeline_id,
            details={
                "policy_name": policy_name,
                "violation": violation_message,
                **(details or {}),
            },
        )

    async def emit_slo_breach(
        self,
        namespace: str,
        run_id: str,
        slo_name: str,
        expected_value: Any,
        actual_value: Any,
        pipeline_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> List[WebhookDeliveryResult]:
        """Emit an SLO_BREACH alert."""
        return await self.emit_alert(
            namespace=namespace,
            alert_type=AlertType.SLO_BREACH,
            severity=AlertSeverity.CRITICAL,
            run_id=run_id,
            message=f"SLO '{slo_name}' breached: expected {expected_value}, got {actual_value}",
            pipeline_id=pipeline_id,
            details={
                "slo_name": slo_name,
                "expected": expected_value,
                "actual": actual_value,
                **(details or {}),
            },
        )

    async def emit_run_succeeded(
        self,
        namespace: str,
        run_id: str,
        pipeline_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> List[WebhookDeliveryResult]:
        """Emit a RUN_SUCCEEDED alert (informational)."""
        message = f"Pipeline run completed successfully"
        if duration_ms:
            message += f" in {duration_ms:.1f}ms"

        return await self.emit_alert(
            namespace=namespace,
            alert_type=AlertType.RUN_SUCCEEDED,
            severity=AlertSeverity.INFO,
            run_id=run_id,
            message=message,
            pipeline_id=pipeline_id,
            details={"duration_ms": duration_ms, **(details or {})},
        )

    def register_webhook(
        self, namespace: str, config: WebhookConfig
    ) -> str:
        """Register a webhook for a namespace."""
        return self._webhook_manager.register_webhook(namespace, config)

    def unregister_webhook(self, namespace: str, webhook_id: str) -> bool:
        """Unregister a webhook from a namespace."""
        return self._webhook_manager.unregister_webhook(namespace, webhook_id)

    def list_webhooks(self, namespace: str) -> List[WebhookConfig]:
        """List all webhooks for a namespace."""
        return self._webhook_manager.list_webhooks(namespace)

    async def send_test_alert(
        self, namespace: str, webhook_id: Optional[str] = None
    ) -> List[WebhookDeliveryResult]:
        """
        Send a test alert to verify webhook configuration.

        Args:
            namespace: Namespace to send test alert to
            webhook_id: Optional specific webhook to test (tests all if not specified)

        Returns:
            List of delivery results
        """
        alert = AlertPayload(
            alert_type=AlertType.RUN_SUCCEEDED,
            severity=AlertSeverity.INFO,
            run_id="test-run-000",
            namespace=namespace,
            message="This is a test alert from GreenLang Orchestrator",
            details={"test": True, "timestamp": datetime.now(timezone.utc).isoformat()},
        )

        if webhook_id:
            webhook = self._webhook_manager.get_webhook(namespace, webhook_id)
            if not webhook:
                return [
                    WebhookDeliveryResult(
                        webhook_id=webhook_id,
                        alert_id=alert.alert_id,
                        status=WebhookDeliveryStatus.FAILED,
                        error_message="Webhook not found",
                    )
                ]
            return [await self._webhook_manager._deliver_to_webhook(webhook, alert)]

        return await self._webhook_manager.dispatch_alert(namespace, alert)

    async def close(self) -> None:
        """Close resources."""
        await self._webhook_manager.close()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "AlertType",
    "AlertSeverity",
    "WebhookDeliveryStatus",
    # Models
    "AlertPayload",
    "WebhookConfig",
    "WebhookDeliveryResult",
    # Managers
    "WebhookManager",
    "AlertManager",
    # Helpers
    "compute_hmac_signature",
]
