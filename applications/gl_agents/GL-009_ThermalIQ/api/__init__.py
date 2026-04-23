"""
GL-009 ThermalIQ - API Module

REST and GraphQL API endpoints for thermal fluid property calculations,
exergy analysis, and Sankey diagram generation.

Provides:
- REST endpoints for thermal analysis operations
- GraphQL schema for flexible queries
- Fluid property calculations
- Exergy destruction analysis
- Sankey diagram generation
- Authentication and rate limiting middleware
"""

from .rest_api import create_app, router
from .graphql_schema import schema, ThermalQuery, ThermalMutation
from .middleware import (
    RateLimitMiddleware,
    AuthenticationMiddleware,
    AuditMiddleware,
    ProvenanceMiddleware,
    ErrorHandlingMiddleware,
)
from .api_schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    EfficiencyRequest,
    EfficiencyResponse,
    ExergyRequest,
    ExergyResponse,
    FluidPropertiesRequest,
    FluidPropertiesResponse,
    SankeyRequest,
    SankeyResponse,
    FluidRecommendationRequest,
    FluidRecommendationResponse,
    ErrorResponse,
)

__all__ = [
    # App and router
    "create_app",
    "router",
    # GraphQL
    "schema",
    "ThermalQuery",
    "ThermalMutation",
    # Middleware
    "RateLimitMiddleware",
    "AuthenticationMiddleware",
    "AuditMiddleware",
    "ProvenanceMiddleware",
    "ErrorHandlingMiddleware",
    # Request/Response schemas
    "AnalyzeRequest",
    "AnalyzeResponse",
    "EfficiencyRequest",
    "EfficiencyResponse",
    "ExergyRequest",
    "ExergyResponse",
    "FluidPropertiesRequest",
    "FluidPropertiesResponse",
    "SankeyRequest",
    "SankeyResponse",
    "FluidRecommendationRequest",
    "FluidRecommendationResponse",
    "ErrorResponse",
]
