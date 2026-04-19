# -*- coding: utf-8 -*-
"""
FastAPI Feature Serving Server for GreenLang Process Heat Agents

This module provides a FastAPI-based feature serving API for real-time
inference, including:
- GET /features/{entity_id} - Get features for a single entity
- POST /features/batch - Batch feature retrieval for multiple entities
- GET /features/online/{feature_view} - Get online features from a view
- Health and metrics endpoints

All endpoints include SHA-256 provenance tracking for regulatory compliance.

Example:
    >>> # Run the server
    >>> uvicorn greenlang.ml.feature_store.feature_server:app --host 0.0.0.0 --port 8000
    >>>
    >>> # Or import and mount in existing app
    >>> from greenlang.ml.feature_store.feature_server import create_app
    >>> app = create_app()
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import hashlib
import json
import logging
import time
import os

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, Query, Path, Depends, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class FeatureRequest(BaseModel):
    """Request model for single entity feature retrieval."""

    entity_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Entity identifier"
    )
    feature_refs: List[str] = Field(
        ...,
        min_items=1,
        description="List of feature references (e.g., 'gl002_boiler_features:efficiency')"
    )
    entity_type: str = Field(
        default="equipment_id",
        description="Type of entity (equipment_id or facility_id)"
    )
    include_provenance: bool = Field(
        default=True,
        description="Include provenance hash in response"
    )

    @validator('feature_refs')
    def validate_feature_refs(cls, v):
        """Validate feature reference format."""
        for ref in v:
            if ':' not in ref:
                raise ValueError(
                    f"Invalid feature reference '{ref}'. "
                    "Expected format: 'feature_view:feature_name'"
                )
        return v


class BatchFeatureRequest(BaseModel):
    """Request model for batch feature retrieval."""

    entity_ids: List[str] = Field(
        ...,
        min_items=1,
        max_items=1000,
        description="List of entity identifiers"
    )
    feature_refs: List[str] = Field(
        ...,
        min_items=1,
        description="List of feature references"
    )
    entity_type: str = Field(
        default="equipment_id",
        description="Type of entity"
    )
    include_provenance: bool = Field(
        default=True,
        description="Include provenance hashes"
    )


class OnlineFeatureRequest(BaseModel):
    """Request model for online feature view retrieval."""

    entity_ids: List[str] = Field(
        ...,
        min_items=1,
        max_items=1000,
        description="List of entity identifiers"
    )
    entity_type: str = Field(
        default="equipment_id",
        description="Type of entity"
    )


class FeatureValue(BaseModel):
    """Single feature value with metadata."""

    name: str = Field(..., description="Feature name")
    value: Optional[Union[float, int, str, bool, List[Any]]] = Field(
        None,
        description="Feature value"
    )
    timestamp: Optional[datetime] = Field(
        None,
        description="Feature timestamp"
    )


class FeatureResponse(BaseModel):
    """Response model for feature retrieval."""

    entity_id: str = Field(..., description="Entity identifier")
    features: Dict[str, Any] = Field(
        default_factory=dict,
        description="Feature name to value mapping"
    )
    feature_refs: List[str] = Field(
        default_factory=list,
        description="Requested feature references"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp"
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash"
    )
    cache_hit: bool = Field(
        default=False,
        description="Whether features were served from cache"
    )
    retrieval_time_ms: float = Field(
        default=0.0,
        description="Feature retrieval time in milliseconds"
    )


class BatchFeatureResponse(BaseModel):
    """Response model for batch feature retrieval."""

    results: List[FeatureResponse] = Field(
        default_factory=list,
        description="List of feature responses"
    )
    total_entities: int = Field(
        default=0,
        description="Total number of entities requested"
    )
    successful_entities: int = Field(
        default=0,
        description="Number of entities with successful retrieval"
    )
    failed_entities: int = Field(
        default=0,
        description="Number of entities with failed retrieval"
    )
    total_time_ms: float = Field(
        default=0.0,
        description="Total processing time in milliseconds"
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash for batch"
    )


class FeatureViewResponse(BaseModel):
    """Response model for feature view retrieval."""

    feature_view: str = Field(..., description="Feature view name")
    features: List[str] = Field(
        default_factory=list,
        description="Available features in this view"
    )
    entities: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Entity ID to features mapping"
    )
    description: Optional[str] = Field(
        None,
        description="Feature view description"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Health status")
    components: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Component health status"
    )
    version: str = Field(
        default="1.0.0",
        description="API version"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class MetricsResponse(BaseModel):
    """Metrics response."""

    requests_total: int = Field(
        default=0,
        description="Total number of requests"
    )
    requests_successful: int = Field(
        default=0,
        description="Number of successful requests"
    )
    requests_failed: int = Field(
        default=0,
        description="Number of failed requests"
    )
    avg_latency_ms: float = Field(
        default=0.0,
        description="Average request latency in milliseconds"
    )
    cache_hit_rate: float = Field(
        default=0.0,
        description="Cache hit rate (0-1)"
    )
    features_served: int = Field(
        default=0,
        description="Total features served"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(
        None,
        description="Detailed error information"
    )
    code: str = Field(..., description="Error code")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# =============================================================================
# METRICS TRACKER
# =============================================================================

class MetricsTracker:
    """Simple in-memory metrics tracker."""

    def __init__(self):
        self.requests_total = 0
        self.requests_successful = 0
        self.requests_failed = 0
        self.total_latency_ms = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.features_served = 0

    def record_request(
        self,
        success: bool,
        latency_ms: float,
        cache_hit: bool,
        features_count: int
    ):
        """Record a request."""
        self.requests_total += 1
        if success:
            self.requests_successful += 1
        else:
            self.requests_failed += 1
        self.total_latency_ms += latency_ms
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        self.features_served += features_count

    def get_metrics(self) -> MetricsResponse:
        """Get current metrics."""
        avg_latency = (
            self.total_latency_ms / self.requests_total
            if self.requests_total > 0 else 0.0
        )
        cache_hit_rate = (
            self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0 else 0.0
        )

        return MetricsResponse(
            requests_total=self.requests_total,
            requests_successful=self.requests_successful,
            requests_failed=self.requests_failed,
            avg_latency_ms=avg_latency,
            cache_hit_rate=cache_hit_rate,
            features_served=self.features_served
        )


# =============================================================================
# FEATURE SERVER CLASS
# =============================================================================

class FeatureServer:
    """
    Feature Serving Server for Process Heat Agents.

    This class provides the core feature serving logic, which can be
    used with FastAPI or other frameworks.

    Attributes:
        feature_store: ProcessHeatFeatureStore instance
        metrics: MetricsTracker instance
    """

    def __init__(self):
        """Initialize FeatureServer."""
        self._feature_store = None
        self.metrics = MetricsTracker()
        self._initialized = False

    def initialize(self):
        """Initialize the feature store connection."""
        if self._initialized:
            return

        try:
            from greenlang.ml.feature_store.feast_config import ProcessHeatFeatureStore
            self._feature_store = ProcessHeatFeatureStore(initialize_stores=True)
            self._initialized = True
            logger.info("FeatureServer initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize feature store: {e}")
            # Continue without store - will return mock data
            self._initialized = True

    @property
    def feature_store(self):
        """Get feature store, initializing if needed."""
        if not self._initialized:
            self.initialize()
        return self._feature_store

    def _calculate_provenance_hash(self, data: Dict[str, Any]) -> str:
        """Calculate SHA-256 provenance hash."""
        try:
            json_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(json_str.encode('utf-8')).hexdigest()
        except Exception:
            return ""

    async def get_features(
        self,
        entity_id: str,
        feature_refs: List[str],
        entity_type: str = "equipment_id",
        include_provenance: bool = True
    ) -> FeatureResponse:
        """
        Get features for a single entity.

        Args:
            entity_id: Entity identifier
            feature_refs: List of feature references
            entity_type: Type of entity
            include_provenance: Include provenance hash

        Returns:
            FeatureResponse with retrieved features
        """
        start_time = time.time()

        features: Dict[str, Any] = {}
        cache_hit = False

        try:
            if self.feature_store is not None:
                result = self.feature_store.get_online_features(
                    entity_ids=[entity_id],
                    feature_refs=feature_refs,
                    entity_type=entity_type
                )
                # Extract features for single entity
                for ref in feature_refs:
                    feature_name = ref.split(":")[-1]
                    values = result.features.get(feature_name, [])
                    features[feature_name] = values[0] if values else None
                cache_hit = result.cache_hit
            else:
                # Return mock data if store not available
                for ref in feature_refs:
                    feature_name = ref.split(":")[-1]
                    features[feature_name] = None

        except Exception as e:
            logger.error(f"Error retrieving features: {e}")
            for ref in feature_refs:
                feature_name = ref.split(":")[-1]
                features[feature_name] = None

        # Calculate retrieval time
        retrieval_time_ms = (time.time() - start_time) * 1000

        # Calculate provenance
        provenance_hash = None
        if include_provenance:
            provenance_data = {
                "entity_id": entity_id,
                "feature_refs": feature_refs,
                "features": features,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            provenance_hash = self._calculate_provenance_hash(provenance_data)

        # Record metrics
        self.metrics.record_request(
            success=True,
            latency_ms=retrieval_time_ms,
            cache_hit=cache_hit,
            features_count=len(features)
        )

        return FeatureResponse(
            entity_id=entity_id,
            features=features,
            feature_refs=feature_refs,
            provenance_hash=provenance_hash,
            cache_hit=cache_hit,
            retrieval_time_ms=retrieval_time_ms
        )

    async def get_batch_features(
        self,
        entity_ids: List[str],
        feature_refs: List[str],
        entity_type: str = "equipment_id",
        include_provenance: bool = True
    ) -> BatchFeatureResponse:
        """
        Get features for multiple entities.

        Args:
            entity_ids: List of entity identifiers
            feature_refs: List of feature references
            entity_type: Type of entity
            include_provenance: Include provenance hashes

        Returns:
            BatchFeatureResponse with retrieved features
        """
        start_time = time.time()

        results: List[FeatureResponse] = []
        successful = 0
        failed = 0

        try:
            if self.feature_store is not None:
                result = self.feature_store.get_online_features(
                    entity_ids=entity_ids,
                    feature_refs=feature_refs,
                    entity_type=entity_type
                )

                # Build response for each entity
                for i, entity_id in enumerate(entity_ids):
                    entity_features: Dict[str, Any] = {}
                    for ref in feature_refs:
                        feature_name = ref.split(":")[-1]
                        values = result.features.get(feature_name, [])
                        entity_features[feature_name] = values[i] if i < len(values) else None

                    # Calculate provenance for entity
                    provenance_hash = None
                    if include_provenance:
                        provenance_data = {
                            "entity_id": entity_id,
                            "features": entity_features
                        }
                        provenance_hash = self._calculate_provenance_hash(provenance_data)

                    results.append(FeatureResponse(
                        entity_id=entity_id,
                        features=entity_features,
                        feature_refs=feature_refs,
                        provenance_hash=provenance_hash,
                        cache_hit=result.cache_hit
                    ))
                    successful += 1

            else:
                # Return mock data if store not available
                for entity_id in entity_ids:
                    entity_features = {
                        ref.split(":")[-1]: None for ref in feature_refs
                    }
                    results.append(FeatureResponse(
                        entity_id=entity_id,
                        features=entity_features,
                        feature_refs=feature_refs
                    ))
                    successful += 1

        except Exception as e:
            logger.error(f"Error retrieving batch features: {e}")
            failed = len(entity_ids)

        # Calculate total time
        total_time_ms = (time.time() - start_time) * 1000

        # Calculate batch provenance
        batch_provenance = None
        if include_provenance:
            batch_data = {
                "entity_ids": entity_ids,
                "feature_refs": feature_refs,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            batch_provenance = self._calculate_provenance_hash(batch_data)

        # Record metrics
        self.metrics.record_request(
            success=failed == 0,
            latency_ms=total_time_ms,
            cache_hit=any(r.cache_hit for r in results),
            features_count=len(feature_refs) * successful
        )

        return BatchFeatureResponse(
            results=results,
            total_entities=len(entity_ids),
            successful_entities=successful,
            failed_entities=failed,
            total_time_ms=total_time_ms,
            provenance_hash=batch_provenance
        )

    async def get_feature_view_features(
        self,
        feature_view: str,
        entity_ids: List[str],
        entity_type: str = "equipment_id"
    ) -> FeatureViewResponse:
        """
        Get all features from a specific feature view.

        Args:
            feature_view: Feature view name
            entity_ids: List of entity identifiers
            entity_type: Type of entity

        Returns:
            FeatureViewResponse with features from the view
        """
        view_config = None
        features_list: List[str] = []
        description = None

        if self.feature_store is not None:
            view_config = self.feature_store.get_feature_view(feature_view)
            if view_config:
                features_list = view_config.features
                description = view_config.description

        # Build feature refs
        feature_refs = [f"{feature_view}:{f}" for f in features_list]

        # Get features
        entities: Dict[str, Dict[str, Any]] = {}

        if feature_refs:
            try:
                result = self.feature_store.get_online_features(
                    entity_ids=entity_ids,
                    feature_refs=feature_refs,
                    entity_type=entity_type
                )

                for i, entity_id in enumerate(entity_ids):
                    entity_features: Dict[str, Any] = {}
                    for f in features_list:
                        values = result.features.get(f, [])
                        entity_features[f] = values[i] if i < len(values) else None
                    entities[entity_id] = entity_features

            except Exception as e:
                logger.error(f"Error getting feature view features: {e}")

        return FeatureViewResponse(
            feature_view=feature_view,
            features=features_list,
            entities=entities,
            description=description
        )

    async def health_check(self) -> HealthResponse:
        """
        Perform health check.

        Returns:
            HealthResponse with component status
        """
        components: Dict[str, Dict[str, Any]] = {}

        if self.feature_store is not None:
            try:
                health = self.feature_store.health_check()
                components = health.get("components", {})
                status = health.get("status", "unknown")
            except Exception as e:
                status = "unhealthy"
                components["error"] = {"message": str(e)}
        else:
            status = "degraded"
            components["feature_store"] = {"status": "not_configured"}

        return HealthResponse(
            status=status,
            components=components
        )

    def get_metrics(self) -> MetricsResponse:
        """Get current metrics."""
        return self.metrics.get_metrics()


# =============================================================================
# GLOBAL FEATURE SERVER INSTANCE
# =============================================================================

feature_server = FeatureServer()


# =============================================================================
# FASTAPI APPLICATION FACTORY
# =============================================================================

def create_app() -> "FastAPI":
    """
    Create FastAPI application for feature serving.

    Returns:
        Configured FastAPI application

    Example:
        >>> app = create_app()
        >>> # Run with: uvicorn greenlang.ml.feature_store.feature_server:app
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required for feature server. "
            "Install with: pip install fastapi uvicorn"
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan management."""
        # Startup
        logger.info("Starting Feature Server...")
        feature_server.initialize()
        yield
        # Shutdown
        logger.info("Shutting down Feature Server...")
        if feature_server.feature_store is not None:
            feature_server.feature_store.close()

    app = FastAPI(
        title="GreenLang Feature Store API",
        description=(
            "Feature serving API for GreenLang Process Heat Agents. "
            "Provides real-time feature retrieval with SHA-256 provenance tracking."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # Add CORS middleware - SECURITY: Configure specific origins in production
    _cors_origins = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000")
    _allowed_origins = [origin.strip() for origin in _cors_origins.split(",") if origin.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Request-ID"],
    )

    # ==========================================================================
    # ENDPOINTS
    # ==========================================================================

    @app.get(
        "/features/{entity_id}",
        response_model=FeatureResponse,
        summary="Get features for a single entity",
        tags=["Features"]
    )
    async def get_entity_features(
        entity_id: str = Path(..., description="Entity identifier"),
        feature_refs: List[str] = Query(
            ...,
            description="Feature references (e.g., gl002_boiler_features:efficiency)"
        ),
        entity_type: str = Query(
            "equipment_id",
            description="Entity type"
        ),
        include_provenance: bool = Query(
            True,
            description="Include provenance hash"
        )
    ):
        """
        Get features for a single entity.

        Returns feature values for the specified entity and feature references.
        Includes SHA-256 provenance hash for audit trail.
        """
        return await feature_server.get_features(
            entity_id=entity_id,
            feature_refs=feature_refs,
            entity_type=entity_type,
            include_provenance=include_provenance
        )

    @app.post(
        "/features/batch",
        response_model=BatchFeatureResponse,
        summary="Batch feature retrieval",
        tags=["Features"]
    )
    async def get_batch_features(
        request: BatchFeatureRequest
    ):
        """
        Get features for multiple entities in a single request.

        Optimized for batch inference workloads. Supports up to 1000 entities
        per request.
        """
        return await feature_server.get_batch_features(
            entity_ids=request.entity_ids,
            feature_refs=request.feature_refs,
            entity_type=request.entity_type,
            include_provenance=request.include_provenance
        )

    @app.get(
        "/features/online/{feature_view}",
        response_model=FeatureViewResponse,
        summary="Get online features from a feature view",
        tags=["Features"]
    )
    async def get_online_features(
        feature_view: str = Path(..., description="Feature view name"),
        entity_ids: List[str] = Query(
            ...,
            description="List of entity identifiers"
        ),
        entity_type: str = Query(
            "equipment_id",
            description="Entity type"
        )
    ):
        """
        Get all features from a specific feature view.

        Returns all features defined in the feature view for the specified
        entities.
        """
        return await feature_server.get_feature_view_features(
            feature_view=feature_view,
            entity_ids=entity_ids,
            entity_type=entity_type
        )

    @app.get(
        "/feature-views",
        summary="List available feature views",
        tags=["Feature Views"]
    )
    async def list_feature_views():
        """
        List all available feature views.

        Returns a list of registered feature view names.
        """
        if feature_server.feature_store is not None:
            views = feature_server.feature_store.list_feature_views()
        else:
            views = []

        return {
            "feature_views": views,
            "count": len(views)
        }

    @app.get(
        "/feature-views/{view_name}",
        summary="Get feature view details",
        tags=["Feature Views"]
    )
    async def get_feature_view(
        view_name: str = Path(..., description="Feature view name")
    ):
        """
        Get details for a specific feature view.

        Returns configuration and metadata for the feature view.
        """
        if feature_server.feature_store is not None:
            view = feature_server.feature_store.get_feature_view(view_name)
            if view:
                return {
                    "name": view.name,
                    "entities": view.entities,
                    "features": view.features,
                    "ttl_hours": view.ttl_hours,
                    "source_table": view.source_table,
                    "description": view.description,
                    "tags": view.tags
                }

        raise HTTPException(
            status_code=404,
            detail=f"Feature view '{view_name}' not found"
        )

    @app.get(
        "/feature-services",
        summary="List available feature services",
        tags=["Feature Services"]
    )
    async def list_feature_services():
        """
        List all available feature services.

        Feature services group multiple feature views for common use cases.
        """
        if feature_server.feature_store is not None:
            services = feature_server.feature_store.list_feature_services()
        else:
            services = []

        return {
            "feature_services": services,
            "count": len(services)
        }

    @app.get(
        "/feature-services/{service_name}",
        summary="Get feature service details",
        tags=["Feature Services"]
    )
    async def get_feature_service(
        service_name: str = Path(..., description="Feature service name")
    ):
        """
        Get details for a specific feature service.
        """
        if feature_server.feature_store is not None:
            service = feature_server.feature_store.get_feature_service(service_name)
            if service:
                return {
                    "name": service.name,
                    "feature_views": service.feature_views,
                    "description": service.description,
                    "owner": service.owner,
                    "tags": service.tags
                }

        raise HTTPException(
            status_code=404,
            detail=f"Feature service '{service_name}' not found"
        )

    @app.get(
        "/agents/{agent_id}/features",
        summary="Get features for a specific agent",
        tags=["Agents"]
    )
    async def get_agent_features(
        agent_id: str = Path(
            ...,
            description="Agent identifier (e.g., GL-001, GL-002)"
        ),
        entity_ids: List[str] = Query(
            ...,
            description="Entity identifiers"
        )
    ):
        """
        Get features for a specific Process Heat agent.

        Maps agent ID to the appropriate feature view and returns features.
        """
        if feature_server.feature_store is not None:
            feature_view = feature_server.feature_store.get_agent_feature_view(agent_id)
            if feature_view:
                return await feature_server.get_feature_view_features(
                    feature_view=feature_view,
                    entity_ids=entity_ids
                )

        raise HTTPException(
            status_code=404,
            detail=f"No feature view found for agent '{agent_id}'"
        )

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Health check",
        tags=["Health"]
    )
    async def health_check():
        """
        Health check endpoint.

        Returns status of all feature store components.
        """
        return await feature_server.health_check()

    @app.get(
        "/health/live",
        summary="Liveness probe",
        tags=["Health"]
    )
    async def liveness():
        """
        Kubernetes liveness probe.

        Returns 200 if the server is running.
        """
        return {"status": "alive"}

    @app.get(
        "/health/ready",
        summary="Readiness probe",
        tags=["Health"]
    )
    async def readiness():
        """
        Kubernetes readiness probe.

        Returns 200 if the server is ready to accept requests.
        """
        health = await feature_server.health_check()
        if health.status == "unhealthy":
            raise HTTPException(status_code=503, detail="Service not ready")
        return {"status": "ready"}

    @app.get(
        "/metrics",
        response_model=MetricsResponse,
        summary="Get metrics",
        tags=["Metrics"]
    )
    async def get_metrics():
        """
        Get server metrics.

        Returns request counts, latencies, and cache statistics.
        """
        return feature_server.get_metrics()

    @app.get(
        "/",
        summary="API info",
        tags=["Info"]
    )
    async def root():
        """
        API root endpoint.

        Returns basic API information.
        """
        return {
            "name": "GreenLang Feature Store API",
            "version": "1.0.0",
            "description": "Feature serving for Process Heat agents",
            "docs": "/docs",
            "health": "/health"
        }

    return app


# =============================================================================
# APPLICATION INSTANCE
# =============================================================================

if FASTAPI_AVAILABLE:
    app = create_app()
else:
    app = None
    logger.warning(
        "FastAPI not available. Install with: pip install fastapi uvicorn"
    )
