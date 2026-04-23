"""
GL-004 BURNMASTER API Module

Production-grade REST, GraphQL, and gRPC APIs for the Burner Optimization Agent.
Provides programmatic access to burner status, KPIs, recommendations, and control operations.

Components:
- REST API: FastAPI-based REST endpoints with OpenAPI documentation
- GraphQL API: Strawberry GraphQL schema with queries, mutations, and subscriptions
- gRPC API: High-performance gRPC services for industrial integrations
- WebSocket: Real-time streaming for status, recommendations, and alerts
- Authentication: JWT and API key authentication with RBAC

Usage:
    from api import app, create_app
    from api.rest_api import router
    from api.graphql_api import schema
    from api.grpc_services import BurnerOptimizationServicer
"""

# Version info
__version__ = "1.0.0"
__author__ = "GreenLang"
__description__ = "GL-004 BURNMASTER Burner Optimization API"

# Configuration
from .config import (
    APISettings,
    ServerConfig,
    SecurityConfig,
    RateLimitConfig,
    CORSConfig,
    WebSocketConfig,
    GRPCConfig,
    GraphQLConfig,
    AuditConfig,
    get_settings,
    get_environment,
    is_production,
    is_development,
    settings
)

# Schemas
from .api_schemas import (
    # Enums
    OperatingMode,
    BurnerState,
    RecommendationPriority,
    RecommendationStatus,
    AlertSeverity,
    AlertStatus,
    OptimizationState,
    UserRole,
    # Request/Response Models
    BurnerMetrics,
    BurnerStatusResponse,
    KPIValue,
    EmissionsKPIs,
    EfficiencyKPIs,
    OperationalKPIs,
    KPIDashboardResponse,
    RecommendationAction,
    RecommendationImpact,
    RecommendationResponse,
    AcceptRecommendationRequest,
    AcceptRecommendationResponse,
    OptimizationMetrics,
    OptimizationStatusResponse,
    ModeChangeRequest,
    ModeChangeResponse,
    HistoryRequest,
    HistoryDataPoint,
    HistoryResponse,
    AlertResponse,
    AlertAcknowledgeRequest,
    ServiceHealth,
    HealthResponse,
    WebSocketMessage
)

# Authentication
from .api_auth import (
    # Models
    Token,
    TokenData,
    User,
    APIKey,
    AuditLogEntry,
    # Functions
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_api_key,
    hash_api_key,
    validate_api_key,
    # Dependencies
    get_current_user,
    get_optional_user,
    require_roles,
    require_operator,
    require_engineer,
    require_admin,
    # Audit
    audit_logger,
    audit_action,
    RoleChecker,
    AuditLogger
)

# REST API
from .rest_api import router as rest_router

# GraphQL API
from .graphql_api import schema as graphql_schema

# gRPC Services
from .grpc_services import (
    BurnerOptimizationServicer,
    create_grpc_server,
    serve_grpc,
    get_proto_template,
    # Message classes
    BurnerStatus,
    KPIResponse,
    RecommendationMessage,
    AcceptRecommendationRequest as GRPCAcceptRecommendationRequest,
    AcceptRecommendationResponse as GRPCAcceptRecommendationResponse,
    ModeChangeRequest as GRPCModeChangeRequest,
    ModeChangeResponse as GRPCModeChangeResponse,
    UnitRequest,
    RecommendationsRequest,
    StreamRequest
)

# WebSocket
from .websocket_handler import (
    router as websocket_router,
    ConnectionManager,
    WebSocketConnection,
    ConnectionState,
    manager as ws_manager,
    notify_status_change,
    notify_new_recommendation,
    notify_alert,
    start_background_tasks
)

# Main Application
from .main import (
    app,
    create_app,
    run_server,
    limiter
)


# Public API
__all__ = [
    # Version
    "__version__",
    "__author__",
    "__description__",

    # Configuration
    "APISettings",
    "ServerConfig",
    "SecurityConfig",
    "RateLimitConfig",
    "CORSConfig",
    "WebSocketConfig",
    "GRPCConfig",
    "GraphQLConfig",
    "AuditConfig",
    "get_settings",
    "get_environment",
    "is_production",
    "is_development",
    "settings",

    # Enums
    "OperatingMode",
    "BurnerState",
    "RecommendationPriority",
    "RecommendationStatus",
    "AlertSeverity",
    "AlertStatus",
    "OptimizationState",
    "UserRole",

    # Schemas
    "BurnerMetrics",
    "BurnerStatusResponse",
    "KPIValue",
    "EmissionsKPIs",
    "EfficiencyKPIs",
    "OperationalKPIs",
    "KPIDashboardResponse",
    "RecommendationAction",
    "RecommendationImpact",
    "RecommendationResponse",
    "AcceptRecommendationRequest",
    "AcceptRecommendationResponse",
    "OptimizationMetrics",
    "OptimizationStatusResponse",
    "ModeChangeRequest",
    "ModeChangeResponse",
    "HistoryRequest",
    "HistoryDataPoint",
    "HistoryResponse",
    "AlertResponse",
    "AlertAcknowledgeRequest",
    "ServiceHealth",
    "HealthResponse",
    "WebSocketMessage",

    # Authentication
    "Token",
    "TokenData",
    "User",
    "APIKey",
    "AuditLogEntry",
    "verify_password",
    "get_password_hash",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "generate_api_key",
    "hash_api_key",
    "validate_api_key",
    "get_current_user",
    "get_optional_user",
    "require_roles",
    "require_operator",
    "require_engineer",
    "require_admin",
    "audit_logger",
    "audit_action",
    "RoleChecker",
    "AuditLogger",

    # REST API
    "rest_router",

    # GraphQL API
    "graphql_schema",

    # gRPC Services
    "BurnerOptimizationServicer",
    "create_grpc_server",
    "serve_grpc",
    "get_proto_template",
    "BurnerStatus",
    "KPIResponse",
    "RecommendationMessage",
    "GRPCAcceptRecommendationRequest",
    "GRPCAcceptRecommendationResponse",
    "GRPCModeChangeRequest",
    "GRPCModeChangeResponse",
    "UnitRequest",
    "RecommendationsRequest",
    "StreamRequest",

    # WebSocket
    "websocket_router",
    "ConnectionManager",
    "WebSocketConnection",
    "ConnectionState",
    "ws_manager",
    "notify_status_change",
    "notify_new_recommendation",
    "notify_alert",
    "start_background_tasks",

    # Main Application
    "app",
    "create_app",
    "run_server",
    "limiter"
]
