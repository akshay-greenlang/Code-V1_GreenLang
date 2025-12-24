"""
GL-011 FUELCRAFT - API Package

REST API for fuel mix optimization with provenance tracking.

This package provides:
- FastAPI-based REST endpoints for optimization runs
- Pydantic schemas for request/response validation
- Middleware for authentication, rate limiting, and audit logging
- Health check endpoints per IEC 61511 requirements

Endpoints:
- POST /runs - Create optimization run request
- GET /runs/{run_id} - Retrieve run status and metadata
- GET /runs/{run_id}/recommendation - Get optimized fuel plan
- GET /runs/{run_id}/explainability - Get SHAP-based explanations

Standards Compliance:
- IEC 61511 (Functional Safety)
- ISO 14064 (GHG Quantification)
- API data provenance via bundle_hash and SHA-256 audit trails
"""

from .schemas import (
    # Request Models
    RunRequest,
    RunStatus,
    FuelConstraints,
    CarbonConstraints,
    EffectiveTimeWindow,
    # Response Models
    RunResponse,
    RunStatusResponse,
    RecommendationResponse,
    ExplainabilityResponse,
    # Output Models
    FuelMixOutput,
    BlendRatioOutput,
    CostBreakdown,
    CarbonFootprint,
    # Health Models
    HealthResponse,
    ReadinessResponse,
    # Error Models
    ErrorResponse,
    ValidationErrorResponse,
)

from .rest_api import (
    FuelCraftAPI,
    create_app,
    app,
)

from .middleware import (
    RequestIDMiddleware,
    AuditLoggingMiddleware,
    RateLimitMiddleware,
    ErrorHandlingMiddleware,
    ProvenanceMiddleware,
)

__all__ = [
    # Request Models
    "RunRequest",
    "RunStatus",
    "FuelConstraints",
    "CarbonConstraints",
    "EffectiveTimeWindow",
    # Response Models
    "RunResponse",
    "RunStatusResponse",
    "RecommendationResponse",
    "ExplainabilityResponse",
    # Output Models
    "FuelMixOutput",
    "BlendRatioOutput",
    "CostBreakdown",
    "CarbonFootprint",
    # Health Models
    "HealthResponse",
    "ReadinessResponse",
    # Error Models
    "ErrorResponse",
    "ValidationErrorResponse",
    # API
    "FuelCraftAPI",
    "create_app",
    "app",
    # Middleware
    "RequestIDMiddleware",
    "AuditLoggingMiddleware",
    "RateLimitMiddleware",
    "ErrorHandlingMiddleware",
    "ProvenanceMiddleware",
]

__version__ = "1.0.0"
