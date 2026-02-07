# -*- coding: utf-8 -*-
"""
Secrets Health REST API Routes - SEC-006

FastAPI router for secrets service health and status:

  GET /health  - Vault health (sealed, initialized, standby)
  GET /status  - Service status (connected, authenticated)
  GET /stats   - Operation statistics

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import (
        APIRouter,
        Depends,
        Header,
        HTTPException,
        Request,
        status,
    )
    from pydantic import BaseModel, ConfigDict, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = object  # type: ignore[misc, assignment]
    Depends = None  # type: ignore[assignment]
    Header = None  # type: ignore[assignment]
    HTTPException = Exception  # type: ignore[misc, assignment]
    Request = None  # type: ignore[assignment]
    status = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[misc, assignment]
    ConfigDict = None  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class VaultHealthResponse(BaseModel):
        """Vault health status."""

        model_config = ConfigDict(from_attributes=True)

        initialized: bool = Field(..., description="Vault is initialized.")
        sealed: bool = Field(..., description="Vault is sealed.")
        standby: bool = Field(default=False, description="Vault is in standby mode.")
        performance_standby: bool = Field(
            default=False, description="Performance standby node."
        )
        replication_perf_mode: Optional[str] = Field(
            default=None, description="Replication performance mode."
        )
        replication_dr_mode: Optional[str] = Field(
            default=None, description="Disaster recovery replication mode."
        )
        server_time_utc: Optional[int] = Field(
            default=None, description="Server time in UTC epoch."
        )
        version: Optional[str] = Field(default=None, description="Vault version.")
        cluster_name: Optional[str] = Field(default=None, description="Cluster name.")
        cluster_id: Optional[str] = Field(default=None, description="Cluster ID.")

    class ServiceHealthResponse(BaseModel):
        """Overall service health."""

        model_config = ConfigDict(from_attributes=True)

        healthy: bool = Field(..., description="Overall health status.")
        vault_healthy: bool = Field(..., description="Vault is healthy.")
        rotation_healthy: bool = Field(
            default=True, description="Rotation manager is healthy."
        )
        cache_enabled: bool = Field(
            default=True, description="Caching is enabled."
        )
        connected: bool = Field(..., description="Service is connected to Vault.")
        vault: VaultHealthResponse = Field(..., description="Vault health details.")
        message: str = Field(default="", description="Health message.")

    class ServiceStatusResponse(BaseModel):
        """Service status details."""

        model_config = ConfigDict(from_attributes=True)

        connected: bool = Field(..., description="Connected to Vault.")
        authenticated: bool = Field(..., description="Authenticated with Vault.")
        vault_addr: str = Field(..., description="Vault server address.")
        vault_namespace: str = Field(default="", description="Vault namespace.")
        auth_method: str = Field(..., description="Authentication method.")
        token_ttl: Optional[int] = Field(
            default=None, description="Token TTL in seconds."
        )
        token_renewable: bool = Field(
            default=False, description="Token is renewable."
        )
        rotation_enabled: bool = Field(
            default=False, description="Rotation manager enabled."
        )
        cache_enabled: bool = Field(
            default=True, description="Caching is enabled."
        )
        lease_count: int = Field(default=0, description="Active leases count.")

    class CacheStatsResponse(BaseModel):
        """Cache statistics."""

        model_config = ConfigDict(from_attributes=True)

        memory_hits: int = Field(default=0, description="Memory cache hits.")
        memory_misses: int = Field(default=0, description="Memory cache misses.")
        memory_size: int = Field(default=0, description="Memory cache size.")
        memory_hit_rate: float = Field(
            default=0.0, description="Memory cache hit rate."
        )
        redis_hits: int = Field(default=0, description="Redis cache hits.")
        redis_misses: int = Field(default=0, description="Redis cache misses.")
        redis_hit_rate: float = Field(
            default=0.0, description="Redis cache hit rate."
        )

    class ServiceStatsResponse(BaseModel):
        """Service operation statistics."""

        model_config = ConfigDict(from_attributes=True)

        cache: CacheStatsResponse = Field(..., description="Cache statistics.")
        connected: bool = Field(..., description="Service is connected.")
        rotation_enabled: bool = Field(
            default=False, description="Rotation is enabled."
        )
        uptime_seconds: Optional[float] = Field(
            default=None, description="Service uptime in seconds."
        )


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def _get_secrets_service() -> Any:
    """FastAPI dependency for SecretsService."""
    try:
        from greenlang.infrastructure.secrets_service import get_secrets_service

        return get_secrets_service()
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail="Secrets service not configured.",
        )


def _get_correlation_id(
    x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID"),
) -> str:
    """Get or generate correlation ID."""
    import uuid

    return x_correlation_id or str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    health_router = APIRouter(
        tags=["Secrets Health"],
        responses={
            503: {"description": "Service Unavailable"},
        },
    )

    # -------------------------------------------------------------------------
    # Health Check
    # -------------------------------------------------------------------------

    @health_router.get(
        "/health",
        response_model=ServiceHealthResponse,
        summary="Health check",
        description="Check Vault and service health.",
        operation_id="secrets_health_check",
    )
    async def health_check(
        request: Request,
        secrets_service: Any = Depends(_get_secrets_service),
        correlation_id: str = Depends(_get_correlation_id),
    ) -> ServiceHealthResponse:
        """Check service and Vault health.

        Returns overall health status including Vault connectivity,
        seal status, and rotation manager health.
        """
        try:
            health = await secrets_service.health_check()

            vault_status = health.get("vault", {}).get("status", {})

            vault_health = VaultHealthResponse(
                initialized=vault_status.get("initialized", False),
                sealed=vault_status.get("sealed", True),
                standby=vault_status.get("standby", False),
                performance_standby=vault_status.get("performance_standby", False),
                replication_perf_mode=vault_status.get("replication_performance_mode"),
                replication_dr_mode=vault_status.get("replication_dr_mode"),
                server_time_utc=vault_status.get("server_time_utc"),
                version=vault_status.get("version"),
                cluster_name=vault_status.get("cluster_name"),
                cluster_id=vault_status.get("cluster_id"),
            )

            vault_healthy = health.get("vault", {}).get("healthy", False)
            rotation_healthy = health.get("rotation", {}).get("healthy", True)

            message = "All systems operational"
            if not vault_healthy:
                message = "Vault is unhealthy"
            elif vault_health.sealed:
                message = "Vault is sealed"
            elif not rotation_healthy:
                message = "Rotation manager has failures"

            return ServiceHealthResponse(
                healthy=health.get("healthy", False),
                vault_healthy=vault_healthy,
                rotation_healthy=rotation_healthy,
                cache_enabled=True,
                connected=health.get("connected", False),
                vault=vault_health,
                message=message,
            )

        except HTTPException:
            raise

        except Exception as exc:
            logger.exception(
                "Health check failed",
                extra={
                    "event_category": "secrets",
                    "correlation_id": correlation_id,
                },
            )

            # Return unhealthy response
            return ServiceHealthResponse(
                healthy=False,
                vault_healthy=False,
                rotation_healthy=False,
                cache_enabled=False,
                connected=False,
                vault=VaultHealthResponse(
                    initialized=False,
                    sealed=True,
                ),
                message=f"Health check failed: {exc}",
            )

    # -------------------------------------------------------------------------
    # Service Status
    # -------------------------------------------------------------------------

    @health_router.get(
        "/status",
        response_model=ServiceStatusResponse,
        summary="Service status",
        description="Get detailed service status.",
        operation_id="secrets_service_status",
    )
    async def service_status(
        request: Request,
        secrets_service: Any = Depends(_get_secrets_service),
        correlation_id: str = Depends(_get_correlation_id),
    ) -> ServiceStatusResponse:
        """Get detailed service status.

        Returns connection status, authentication state, and configuration.
        """
        try:
            health = await secrets_service.health_check()
            config = secrets_service.config

            # Determine if authenticated by checking if connected
            authenticated = health.get("connected", False)

            return ServiceStatusResponse(
                connected=health.get("connected", False),
                authenticated=authenticated,
                vault_addr=config.vault_addr,
                vault_namespace=config.vault_namespace,
                auth_method=config.auth_method,
                token_ttl=None,  # Would require token introspection
                token_renewable=True,  # Assumed
                rotation_enabled=secrets_service.rotation_manager is not None,
                cache_enabled=config.cache_enabled,
                lease_count=0,  # Would require lease tracking
            )

        except HTTPException:
            raise

        except Exception as exc:
            logger.exception(
                "Status check failed",
                extra={
                    "event_category": "secrets",
                    "correlation_id": correlation_id,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Status check failed: {exc}",
            )

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    @health_router.get(
        "/stats",
        response_model=ServiceStatsResponse,
        summary="Service statistics",
        description="Get operation statistics from cache and metrics.",
        operation_id="secrets_service_stats",
    )
    async def service_stats(
        request: Request,
        secrets_service: Any = Depends(_get_secrets_service),
        correlation_id: str = Depends(_get_correlation_id),
    ) -> ServiceStatsResponse:
        """Get service operation statistics.

        Returns cache hit rates, operation counts, and uptime.
        """
        try:
            stats = await secrets_service.get_stats()
            cache_stats = stats.get("cache", {})

            memory_stats = cache_stats.get("memory", {}) or {}
            redis_stats = cache_stats.get("redis", {}) or {}

            return ServiceStatsResponse(
                cache=CacheStatsResponse(
                    memory_hits=memory_stats.get("hits", 0),
                    memory_misses=memory_stats.get("misses", 0),
                    memory_size=memory_stats.get("size", 0),
                    memory_hit_rate=memory_stats.get("hit_rate", 0.0),
                    redis_hits=redis_stats.get("hits", 0),
                    redis_misses=redis_stats.get("misses", 0),
                    redis_hit_rate=redis_stats.get("hit_rate", 0.0),
                ),
                connected=stats.get("connected", False),
                rotation_enabled=stats.get("rotation_enabled", False),
                uptime_seconds=None,  # Would require startup timestamp tracking
            )

        except HTTPException:
            raise

        except Exception as exc:
            logger.exception(
                "Stats check failed",
                extra={
                    "event_category": "secrets",
                    "correlation_id": correlation_id,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Stats check failed: {exc}",
            )

else:
    health_router = None  # type: ignore[assignment]
    logger.warning("FastAPI not available - health_router is None")


__all__ = ["health_router"]
