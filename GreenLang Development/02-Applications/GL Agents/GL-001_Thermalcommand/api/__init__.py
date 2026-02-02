"""
GL-001 ThermalCommand API Layer

Comprehensive GraphQL/gRPC API services for district heating optimization.
Implements GreenLang standard patterns: mTLS + OAuth2/JWT auth, rate limiting, audit trails.
"""

from .api_schemas import (
    DispatchPlan,
    AssetState,
    Constraint,
    KPI,
    DemandUpdate,
    AllocationRequest,
    AllocationResponse,
    ForecastData,
    AlarmEvent,
    MaintenanceTrigger,
    ExplainabilitySummary,
)

from .api_auth import (
    get_current_user,
    verify_token,
    authorize_action,
    ThermalCommandUser,
    Role,
)

from .graphql_api import (
    graphql_app,
    schema,
)

from .grpc_services import (
    ThermalCommandServicer,
    serve_grpc,
)

__version__ = "1.0.0"
__all__ = [
    # Schemas
    "DispatchPlan",
    "AssetState",
    "Constraint",
    "KPI",
    "DemandUpdate",
    "AllocationRequest",
    "AllocationResponse",
    "ForecastData",
    "AlarmEvent",
    "MaintenanceTrigger",
    "ExplainabilitySummary",
    # Auth
    "get_current_user",
    "verify_token",
    "authorize_action",
    "ThermalCommandUser",
    "Role",
    # GraphQL
    "graphql_app",
    "schema",
    # gRPC
    "ThermalCommandServicer",
    "serve_grpc",
]
