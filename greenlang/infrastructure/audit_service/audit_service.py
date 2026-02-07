# -*- coding: utf-8 -*-
"""
Centralized Audit Service - SEC-005: Centralized Audit Logging Service

Main orchestration service that ties together the collector, enricher, router,
and cache components. Provides the primary API for logging audit events with
convenience methods for common event types.

**Architecture:**
    Event -> AuditService.log_event()
          -> AuditCache.check_duplicate() [optional dedup]
          -> AuditEventEnricher.enrich()
          -> AuditEventCollector.collect()
          -> [background worker] AuditEventRouter.route()
                                 -> PostgreSQL (batch insert)
                                 -> Loki (JSON logs)
                                 -> Redis (pub/sub)

**Design Principles:**
- Non-blocking main API (events queued for async processing)
- Graceful shutdown with flush
- Comprehensive metrics for observability
- Backward compatible with legacy audit loggers

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .audit_cache import AuditCache, AuditCacheConfig
from .event_collector import AuditEventCollector, CollectorConfig
from .event_enricher import AuditEventEnricher, EnricherConfig
from .event_model import EventBuilder, UnifiedAuditEvent
from .event_router import AuditEventRouter, RouterConfig
from .event_types import (
    AuditAction,
    AuditEventCategory,
    AuditResult,
    AuditSeverity,
    UnifiedAuditEventType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Service Configuration
# ---------------------------------------------------------------------------


@dataclass
class AuditServiceConfig:
    """Configuration for the AuditService.

    Aggregates configuration for all sub-components.

    Attributes:
        collector: Collector configuration.
        enricher: Enricher configuration.
        router: Router configuration.
        cache: Cache configuration.
        enable_deduplication: Whether to deduplicate events (default True).
        auto_start: Whether to auto-start background worker (default True).
    """

    collector: CollectorConfig = field(default_factory=CollectorConfig)
    enricher: EnricherConfig = field(default_factory=EnricherConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    cache: AuditCacheConfig = field(default_factory=AuditCacheConfig)
    enable_deduplication: bool = True
    auto_start: bool = True


# ---------------------------------------------------------------------------
# Service Metrics
# ---------------------------------------------------------------------------


@dataclass
class AuditServiceMetrics:
    """Aggregated metrics for the audit service.

    Attributes:
        events_logged: Total events logged via log_event().
        events_deduplicated: Events skipped due to deduplication.
        events_enriched: Events successfully enriched.
        events_collected: Events successfully collected.
        events_routed: Events successfully routed.
    """

    events_logged: int = 0
    events_deduplicated: int = 0
    events_enriched: int = 0
    events_collected: int = 0
    events_routed: int = 0


# ---------------------------------------------------------------------------
# Centralized Audit Service
# ---------------------------------------------------------------------------


class AuditService:
    """Centralized audit logging service.

    Orchestrates event collection, enrichment, deduplication, and multi-destination
    routing for comprehensive audit trail capture.

    Example:
        >>> service = AuditService(db_pool=pool, redis_client=redis)
        >>> await service.start()
        >>> await service.log_event(event)
        >>> await service.log_auth_event(
        ...     event_type=UnifiedAuditEventType.AUTH_LOGIN_SUCCESS,
        ...     user_id="u-123",
        ...     tenant_id="t-corp",
        ...     client_ip="10.0.0.1",
        ... )
        >>> await service.stop()
    """

    def __init__(
        self,
        db_pool: Optional[Any] = None,
        redis_client: Optional[Any] = None,
        config: Optional[AuditServiceConfig] = None,
    ) -> None:
        """Initialize the audit service.

        Args:
            db_pool: Async PostgreSQL connection pool.
            redis_client: Async Redis client.
            config: Service configuration.
        """
        self._config = config or AuditServiceConfig()
        self._metrics = AuditServiceMetrics()
        self._running = False

        # Initialize sub-components
        self._cache = AuditCache(
            redis_client=redis_client,
            config=self._config.cache,
        )
        self._enricher = AuditEventEnricher(config=self._config.enricher)
        self._router = AuditEventRouter(
            db_pool=db_pool,
            redis_client=redis_client,
            config=self._config.router,
        )
        self._collector = AuditEventCollector(
            config=self._config.collector,
            on_batch_ready=self._on_batch_ready,
        )

    @property
    def is_running(self) -> bool:
        """Check if the service is running.

        Returns:
            True if the service is running.
        """
        return self._running

    @property
    def metrics(self) -> AuditServiceMetrics:
        """Get aggregated service metrics.

        Returns:
            Current metrics snapshot.
        """
        return self._metrics

    async def start(self) -> None:
        """Start the audit service.

        Starts the background worker for event processing.
        """
        if self._running:
            logger.warning("AuditService is already running")
            return

        await self._collector.start()
        self._running = True
        logger.info("AuditService started")

    async def stop(self, drain: bool = True, timeout: float = 10.0) -> None:
        """Stop the audit service.

        Args:
            drain: Whether to flush remaining events before stopping.
            timeout: Maximum time to wait for drain (seconds).
        """
        if not self._running:
            return

        await self._collector.stop(drain=drain, timeout=timeout)
        self._running = False
        logger.info(
            "AuditService stopped. Total logged: %d, deduplicated: %d",
            self._metrics.events_logged,
            self._metrics.events_deduplicated,
        )

    # -------------------------------------------------------------------------
    # Core Logging API
    # -------------------------------------------------------------------------

    async def log_event(
        self,
        event: UnifiedAuditEvent,
        skip_dedup: bool = False,
    ) -> bool:
        """Log an audit event.

        Main entry point for logging audit events. Events are enriched,
        optionally deduplicated, and queued for async routing.

        Args:
            event: The audit event to log.
            skip_dedup: Whether to skip deduplication check.

        Returns:
            True if event was logged, False if deduplicated or dropped.
        """
        self._metrics.events_logged += 1

        # Deduplication check
        if self._config.enable_deduplication and not skip_dedup:
            if await self._cache.check_duplicate(event.event_id):
                self._metrics.events_deduplicated += 1
                logger.debug(
                    "Duplicate event skipped: event_id=%s",
                    event.event_id,
                )
                return False

        # Enrich the event
        self._enricher.enrich(event)
        self._metrics.events_enriched += 1

        # Mark as processed (for dedup)
        if self._config.enable_deduplication and not skip_dedup:
            await self._cache.mark_processed(event.event_id)

        # Collect for async routing
        if await self._collector.collect(event):
            self._metrics.events_collected += 1
            return True

        return False

    async def log_event_sync(self, event: UnifiedAuditEvent) -> None:
        """Log an audit event synchronously (bypassing queue).

        Use for critical events that must be persisted immediately.
        Blocks until routing is complete.

        Args:
            event: The audit event to log.
        """
        self._metrics.events_logged += 1
        self._enricher.enrich(event)
        self._metrics.events_enriched += 1
        await self._router.route(event)
        self._metrics.events_routed += 1

    async def _on_batch_ready(self, events: List[UnifiedAuditEvent]) -> None:
        """Callback invoked when a batch is ready for routing.

        Args:
            events: Batch of events to route.
        """
        await self._router.route_batch(events)
        self._metrics.events_routed += len(events)

    # -------------------------------------------------------------------------
    # Convenience: Auth Events
    # -------------------------------------------------------------------------

    async def log_auth_event(
        self,
        event_type: UnifiedAuditEventType,
        *,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        result: AuditResult = AuditResult.SUCCESS,
        error_message: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **metadata: Any,
    ) -> bool:
        """Log an authentication-related audit event.

        Args:
            event_type: The auth event type.
            user_id: UUID of the user.
            tenant_id: UUID of the tenant.
            client_ip: Client IP address.
            user_agent: Client User-Agent header.
            session_id: Session identifier.
            result: Operation result.
            error_message: Error message if failed.
            correlation_id: Request correlation ID.
            **metadata: Additional metadata.

        Returns:
            True if event was logged.
        """
        event = (
            EventBuilder(event_type)
            .with_user(user_id=user_id, session_id=session_id)
            .with_tenant(tenant_id) if tenant_id else EventBuilder(event_type).with_user(user_id=user_id, session_id=session_id)
        )

        if tenant_id:
            event = event.with_tenant(tenant_id)
        if client_ip or user_agent:
            event = event.with_client(ip=client_ip, user_agent=user_agent)
        if correlation_id:
            event = event.with_correlation_id(correlation_id)
        if metadata:
            event = event.with_metadata(**metadata)

        event = event.with_result(result, error_message)

        return await self.log_event(event.build())

    # -------------------------------------------------------------------------
    # Convenience: RBAC Events
    # -------------------------------------------------------------------------

    async def log_rbac_event(
        self,
        event_type: UnifiedAuditEventType,
        *,
        actor_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        result: AuditResult = AuditResult.SUCCESS,
        client_ip: Optional[str] = None,
        correlation_id: Optional[str] = None,
        old_value: Optional[Dict[str, Any]] = None,
        new_value: Optional[Dict[str, Any]] = None,
        **metadata: Any,
    ) -> bool:
        """Log an RBAC-related audit event.

        Args:
            event_type: The RBAC event type.
            actor_id: UUID of the actor.
            tenant_id: UUID of the tenant.
            resource_type: Type of resource (role, permission, etc.).
            resource_id: Identifier of the resource.
            action: CRUD action performed.
            result: Operation result.
            client_ip: Client IP address.
            correlation_id: Request correlation ID.
            old_value: Previous state (for updates).
            new_value: New state (for creates/updates).
            **metadata: Additional metadata.

        Returns:
            True if event was logged.
        """
        builder = EventBuilder(event_type).with_user(user_id=actor_id)

        if tenant_id:
            builder = builder.with_tenant(tenant_id)
        if resource_type or resource_id:
            builder = builder.with_resource(
                resource_type=resource_type,
                resource_id=resource_id,
            )
        if action:
            builder = builder.with_action(action)
        if client_ip:
            builder = builder.with_client(ip=client_ip)
        if correlation_id:
            builder = builder.with_correlation_id(correlation_id)

        # Add old/new values to metadata
        if old_value:
            metadata["old_value"] = old_value
        if new_value:
            metadata["new_value"] = new_value
        if metadata:
            builder = builder.with_metadata(**metadata)

        builder = builder.with_result(result)

        return await self.log_event(builder.build())

    # -------------------------------------------------------------------------
    # Convenience: Encryption Events
    # -------------------------------------------------------------------------

    async def log_encryption_event(
        self,
        event_type: UnifiedAuditEventType,
        *,
        tenant_id: Optional[str] = None,
        key_version: Optional[str] = None,
        data_class: Optional[str] = None,
        operation: Optional[str] = None,
        duration_ms: Optional[float] = None,
        result: AuditResult = AuditResult.SUCCESS,
        error_message: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **metadata: Any,
    ) -> bool:
        """Log an encryption-related audit event.

        Args:
            event_type: The encryption event type.
            tenant_id: UUID of the tenant.
            key_version: Version of the encryption key.
            data_class: Data classification (pii, secret, etc.).
            operation: Operation performed (encrypt, decrypt, etc.).
            duration_ms: Operation duration in milliseconds.
            result: Operation result.
            error_message: Error message if failed.
            correlation_id: Request correlation ID.
            **metadata: Additional metadata.

        Returns:
            True if event was logged.
        """
        builder = EventBuilder(event_type)

        if tenant_id:
            builder = builder.with_tenant(tenant_id)
        if correlation_id:
            builder = builder.with_correlation_id(correlation_id)
        if duration_ms:
            builder = builder.with_request(duration_ms=duration_ms)

        # Add encryption-specific metadata
        if key_version:
            metadata["key_version"] = key_version
        if data_class:
            metadata["data_class"] = data_class
        if operation:
            metadata["operation"] = operation
        if metadata:
            builder = builder.with_metadata(**metadata)

        builder = builder.with_result(result, error_message)

        return await self.log_event(builder.build())

    # -------------------------------------------------------------------------
    # Convenience: Data Access Events
    # -------------------------------------------------------------------------

    async def log_data_event(
        self,
        event_type: UnifiedAuditEventType,
        *,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        result: AuditResult = AuditResult.SUCCESS,
        correlation_id: Optional[str] = None,
        **metadata: Any,
    ) -> bool:
        """Log a data access audit event.

        Args:
            event_type: The data event type.
            user_id: UUID of the user.
            tenant_id: UUID of the tenant.
            resource_type: Type of data resource.
            resource_id: Identifier of the data resource.
            action: CRUD action performed.
            result: Operation result.
            correlation_id: Request correlation ID.
            **metadata: Additional metadata.

        Returns:
            True if event was logged.
        """
        builder = EventBuilder(event_type).with_user(user_id=user_id)

        if tenant_id:
            builder = builder.with_tenant(tenant_id)
        if resource_type or resource_id:
            builder = builder.with_resource(
                resource_type=resource_type,
                resource_id=resource_id,
            )
        if action:
            builder = builder.with_action(action)
        if correlation_id:
            builder = builder.with_correlation_id(correlation_id)
        if metadata:
            builder = builder.with_metadata(**metadata)

        builder = builder.with_result(result)

        return await self.log_event(builder.build())

    # -------------------------------------------------------------------------
    # Convenience: Agent Events
    # -------------------------------------------------------------------------

    async def log_agent_event(
        self,
        event_type: UnifiedAuditEventType,
        *,
        agent_name: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        execution_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
        result: AuditResult = AuditResult.SUCCESS,
        error_message: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **metadata: Any,
    ) -> bool:
        """Log an agent execution audit event.

        Args:
            event_type: The agent event type.
            agent_name: Name of the agent.
            user_id: UUID of the user who triggered execution.
            tenant_id: UUID of the tenant.
            execution_id: Unique execution identifier.
            duration_ms: Execution duration in milliseconds.
            result: Execution result.
            error_message: Error message if failed.
            correlation_id: Request correlation ID.
            **metadata: Additional metadata.

        Returns:
            True if event was logged.
        """
        builder = EventBuilder(event_type).with_user(user_id=user_id)

        if tenant_id:
            builder = builder.with_tenant(tenant_id)
        if agent_name:
            builder = builder.with_resource(
                resource_type="agent",
                resource_name=agent_name,
            )
        if correlation_id:
            builder = builder.with_correlation_id(correlation_id)
        if duration_ms:
            builder = builder.with_request(duration_ms=duration_ms)

        # Add agent-specific metadata
        if execution_id:
            metadata["execution_id"] = execution_id
        if metadata:
            builder = builder.with_metadata(**metadata)

        builder = builder.with_result(result, error_message)

        return await self.log_event(builder.build())

    # -------------------------------------------------------------------------
    # Convenience: API Events
    # -------------------------------------------------------------------------

    async def log_api_event(
        self,
        event_type: UnifiedAuditEventType,
        *,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        request_path: Optional[str] = None,
        request_method: Optional[str] = None,
        response_status: Optional[int] = None,
        duration_ms: Optional[float] = None,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        result: AuditResult = AuditResult.SUCCESS,
        error_message: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **metadata: Any,
    ) -> bool:
        """Log an API request audit event.

        Args:
            event_type: The API event type.
            user_id: UUID of the user.
            tenant_id: UUID of the tenant.
            request_path: HTTP request path.
            request_method: HTTP method.
            response_status: HTTP response status code.
            duration_ms: Request duration in milliseconds.
            client_ip: Client IP address.
            user_agent: Client User-Agent header.
            result: Request result.
            error_message: Error message if failed.
            correlation_id: Request correlation ID.
            **metadata: Additional metadata.

        Returns:
            True if event was logged.
        """
        builder = EventBuilder(event_type).with_user(user_id=user_id)

        if tenant_id:
            builder = builder.with_tenant(tenant_id)
        if client_ip or user_agent:
            builder = builder.with_client(ip=client_ip, user_agent=user_agent)
        if correlation_id:
            builder = builder.with_correlation_id(correlation_id)

        builder = builder.with_request(
            path=request_path,
            method=request_method,
            status=response_status,
            duration_ms=duration_ms,
        )

        if metadata:
            builder = builder.with_metadata(**metadata)

        builder = builder.with_result(result, error_message)

        return await self.log_event(builder.build())

    # -------------------------------------------------------------------------
    # Component Access
    # -------------------------------------------------------------------------

    @property
    def cache(self) -> AuditCache:
        """Get the audit cache component.

        Returns:
            The audit cache instance.
        """
        return self._cache

    @property
    def enricher(self) -> AuditEventEnricher:
        """Get the event enricher component.

        Returns:
            The event enricher instance.
        """
        return self._enricher

    @property
    def router(self) -> AuditEventRouter:
        """Get the event router component.

        Returns:
            The event router instance.
        """
        return self._router

    @property
    def collector(self) -> AuditEventCollector:
        """Get the event collector component.

        Returns:
            The event collector instance.
        """
        return self._collector


# ---------------------------------------------------------------------------
# Global Service Instance
# ---------------------------------------------------------------------------

_global_audit_service: Optional[AuditService] = None


def get_audit_service() -> Optional[AuditService]:
    """Get the global audit service instance.

    Returns:
        The global audit service or None if not configured.
    """
    return _global_audit_service


async def configure_audit_service(
    db_pool: Optional[Any] = None,
    redis_client: Optional[Any] = None,
    config: Optional[AuditServiceConfig] = None,
) -> AuditService:
    """Configure and start the global audit service.

    Args:
        db_pool: Async PostgreSQL connection pool.
        redis_client: Async Redis client.
        config: Service configuration.

    Returns:
        The configured audit service.
    """
    global _global_audit_service

    if _global_audit_service is not None:
        await _global_audit_service.stop()

    _global_audit_service = AuditService(
        db_pool=db_pool,
        redis_client=redis_client,
        config=config,
    )

    if config is None or config.auto_start:
        await _global_audit_service.start()

    return _global_audit_service


async def shutdown_audit_service() -> None:
    """Shutdown the global audit service."""
    global _global_audit_service

    if _global_audit_service is not None:
        await _global_audit_service.stop()
        _global_audit_service = None


__all__ = [
    "AuditService",
    "AuditServiceConfig",
    "AuditServiceMetrics",
    "get_audit_service",
    "configure_audit_service",
    "shutdown_audit_service",
]
