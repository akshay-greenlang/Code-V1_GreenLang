"""
GL-016_Waterguard API Package

Multi-protocol API layer for the Waterguard cooling tower optimization system.
Provides REST, GraphQL, and gRPC interfaces for water chemistry monitoring,
optimization control, and compliance reporting.

Protocols:
    - REST API (FastAPI) - Primary interface at /api/v1/
    - GraphQL (Strawberry) - Query/mutation interface at /graphql
    - gRPC - High-performance streaming interface

Authentication:
    - JWT-based authentication with role-based access control
    - Roles: operator, engineer, admin
    - All API calls logged to audit trail

Author: GL-APIDeveloper
Version: 1.0.0
"""

from api.config import APIConfig, get_api_config
from api.api_schemas import (
    OptimizationRequest,
    OptimizationResponse,
    RecommendationApprovalRequest,
    ChemistryStateResponse,
    ComplianceReportResponse,
    SavingsReportResponse,
    HealthCheckResponse,
    BlowdownStatusResponse,
    DosingStatusResponse,
    RecommendationResponse,
)
from api.api_auth import (
    JWTAuthenticator,
    get_current_user,
    require_role,
    UserRole,
    TokenPayload,
)

__version__ = "1.0.0"
__all__ = [
    # Configuration
    "APIConfig",
    "get_api_config",
    # Schemas
    "OptimizationRequest",
    "OptimizationResponse",
    "RecommendationApprovalRequest",
    "ChemistryStateResponse",
    "ComplianceReportResponse",
    "SavingsReportResponse",
    "HealthCheckResponse",
    "BlowdownStatusResponse",
    "DosingStatusResponse",
    "RecommendationResponse",
    # Authentication
    "JWTAuthenticator",
    "get_current_user",
    "require_role",
    "UserRole",
    "TokenPayload",
]
