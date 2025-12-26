# -*- coding: utf-8 -*-
"""GL-015 INSULSCAN - API module.

FastAPI-based REST API for Insulation Scanning and Thermal Assessment.

This module provides:
- Production-grade REST API routes
- Request/response schemas with validation
- Dependency injection for services
- Middleware for logging, auth, and provenance tracking

Usage:
    from gl_015_insulscan.api import app, router, create_router

    # Run with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8015)

Author: GL-BackendDeveloper
Version: 1.0.0
"""

from .routes import router, create_router
from .schemas import (
    # Request/Response models
    AnalyzeInsulationRequest,
    AnalyzeInsulationResponse,
    InsulationAnalysisResult,
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    BatchAnalysisItemResult,
    HotSpotDetectionRequest,
    HotSpotDetectionResponse,
    DetectedHotSpot,
    AssetConditionResponse,
    AssetHistoryResponse,
    HistoricalDataPoint,
    RepairRecommendation,
    RepairRecommendationRequest,
    RepairRecommendationResponse,
    # Health and metrics
    HealthResponse,
    ComponentHealth,
    MetricsResponse,
    MetricValue,
    # Error handling
    ErrorResponse,
    ErrorDetail,
    # Enums
    InsulationType,
    ConditionRating,
    DegradationMechanism,
    RepairPriority,
    HotSpotSeverity,
    # Utilities
    compute_hash,
    create_response_with_hash,
    # Constants
    AGENT_VERSION,
    AGENT_ID,
    AGENT_NAME,
)
from .dependencies import (
    # Settings
    Settings,
    get_settings,
    # Authentication
    User,
    verify_api_key,
    get_current_user,
    require_user,
    require_role,
    # Rate limiting
    RateLimiter,
    get_rate_limiter,
    rate_limiter,
    # Services
    get_orchestrator,
    get_thermal_analyzer,
    get_hotspot_detector,
    get_recommendation_engine,
    get_metrics_service,
    get_historical_data_service,
    get_audit_service,
    # Request context
    get_request_id,
    get_correlation_id,
    # Validation
    validate_asset_exists,
    validate_asset_access,
)
from .middleware import (
    RequestIDMiddleware,
    RequestLoggingMiddleware,
    RateLimitMiddleware,
    ErrorHandlingMiddleware,
    RequestValidationMiddleware,
    AuthenticationMiddleware,
    ProvenanceMiddleware,
    get_cors_config,
)

__all__ = [
    # Router
    "router",
    "create_router",
    # Request/Response schemas
    "AnalyzeInsulationRequest",
    "AnalyzeInsulationResponse",
    "InsulationAnalysisResult",
    "BatchAnalysisRequest",
    "BatchAnalysisResponse",
    "BatchAnalysisItemResult",
    "HotSpotDetectionRequest",
    "HotSpotDetectionResponse",
    "DetectedHotSpot",
    "AssetConditionResponse",
    "AssetHistoryResponse",
    "HistoricalDataPoint",
    "RepairRecommendation",
    "RepairRecommendationRequest",
    "RepairRecommendationResponse",
    # Health and metrics
    "HealthResponse",
    "ComponentHealth",
    "MetricsResponse",
    "MetricValue",
    # Error handling
    "ErrorResponse",
    "ErrorDetail",
    # Enums
    "InsulationType",
    "ConditionRating",
    "DegradationMechanism",
    "RepairPriority",
    "HotSpotSeverity",
    # Utilities
    "compute_hash",
    "create_response_with_hash",
    # Constants
    "AGENT_VERSION",
    "AGENT_ID",
    "AGENT_NAME",
    # Settings
    "Settings",
    "get_settings",
    # Authentication
    "User",
    "verify_api_key",
    "get_current_user",
    "require_user",
    "require_role",
    # Rate limiting
    "RateLimiter",
    "get_rate_limiter",
    "rate_limiter",
    # Services
    "get_orchestrator",
    "get_thermal_analyzer",
    "get_hotspot_detector",
    "get_recommendation_engine",
    "get_metrics_service",
    "get_historical_data_service",
    "get_audit_service",
    # Request context
    "get_request_id",
    "get_correlation_id",
    # Validation
    "validate_asset_exists",
    "validate_asset_access",
    # Middleware
    "RequestIDMiddleware",
    "RequestLoggingMiddleware",
    "RateLimitMiddleware",
    "ErrorHandlingMiddleware",
    "RequestValidationMiddleware",
    "AuthenticationMiddleware",
    "ProvenanceMiddleware",
    "get_cors_config",
]
