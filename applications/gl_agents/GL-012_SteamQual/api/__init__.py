"""
GL-012 SteamQual SteamQualityController API Layer

Production-grade REST API for industrial steam quality monitoring and control.
Implements GreenLang standard patterns: OAuth2/JWT auth, rate limiting, audit trails.

Features:
- Steam quality estimation from sensor data
- Carryover risk assessment
- Quality state monitoring
- Event management and alerting
- Control recommendations
- Quality metrics and KPIs

Latency Targets:
- Sensor-to-metric: < 5 seconds
- Event emission: < 10 seconds

Zero-Hallucination Guarantee:
All numeric calculations use deterministic thermodynamic formulas.
No LLM/ML models are used for regulatory or safety-critical values.
"""

from .schemas import (
    # Quality Estimation
    QualityEstimateRequest,
    QualityEstimateResponse,
    QualityEstimate,
    QualityLevel,
    MeasurementSource,
    SteamMeasurements,
    # Carryover Risk
    CarryoverRiskRequest,
    CarryoverRiskResponse,
    CarryoverRiskAssessment,
    CarryoverRiskLevel,
    CarryoverRiskFactors,
    # Quality State
    QualityStateResponse,
    QualityKPI,
    # Events
    EventsRequest,
    EventsResponse,
    QualityEvent,
    EventType,
    EventSeverity,
    # Recommendations
    RecommendationsRequest,
    RecommendationsResponse,
    QualityRecommendation,
    RecommendationType,
    RecommendationPriority,
    ControlAction,
    # Metrics
    MetricsRequest,
    MetricsResponse,
    QualityMetrics,
    # Common
    ErrorResponse,
    HealthStatus,
    ServiceHealth,
    SystemStatus,
)

from .auth import (
    # User and authentication
    get_current_user,
    verify_token,
    SteamQualUser,
    TokenResponse,
    TokenData,
    # Roles and permissions
    Role,
    Permission,
    ROLE_PERMISSIONS,
    # Authorization dependencies
    require_permissions,
    require_any_permission,
    require_roles,
    require_header_access,
    # API key management
    APIKeyInfo,
    generate_api_key,
    verify_api_key,
    # Audit logging
    AuditLogEntry,
    log_api_call,
    log_security_event,
    # Configuration
    AuthConfig,
    get_auth_config,
)

from .routes import (
    router as quality_router,
)

from .main import (
    app,
    Settings,
    get_settings,
)


__version__ = "1.0.0"

__all__ = [
    # Schemas - Quality Estimation
    "QualityEstimateRequest",
    "QualityEstimateResponse",
    "QualityEstimate",
    "QualityLevel",
    "MeasurementSource",
    "SteamMeasurements",
    # Schemas - Carryover Risk
    "CarryoverRiskRequest",
    "CarryoverRiskResponse",
    "CarryoverRiskAssessment",
    "CarryoverRiskLevel",
    "CarryoverRiskFactors",
    # Schemas - Quality State
    "QualityStateResponse",
    "QualityKPI",
    # Schemas - Events
    "EventsRequest",
    "EventsResponse",
    "QualityEvent",
    "EventType",
    "EventSeverity",
    # Schemas - Recommendations
    "RecommendationsRequest",
    "RecommendationsResponse",
    "QualityRecommendation",
    "RecommendationType",
    "RecommendationPriority",
    "ControlAction",
    # Schemas - Metrics
    "MetricsRequest",
    "MetricsResponse",
    "QualityMetrics",
    # Schemas - Common
    "ErrorResponse",
    "HealthStatus",
    "ServiceHealth",
    "SystemStatus",
    # Auth - User and authentication
    "get_current_user",
    "verify_token",
    "SteamQualUser",
    "TokenResponse",
    "TokenData",
    # Auth - Roles and permissions
    "Role",
    "Permission",
    "ROLE_PERMISSIONS",
    # Auth - Authorization dependencies
    "require_permissions",
    "require_any_permission",
    "require_roles",
    "require_header_access",
    # Auth - API key management
    "APIKeyInfo",
    "generate_api_key",
    "verify_api_key",
    # Auth - Audit logging
    "AuditLogEntry",
    "log_api_call",
    "log_security_event",
    # Auth - Configuration
    "AuthConfig",
    "get_auth_config",
    # Routes
    "quality_router",
    # Application
    "app",
    "Settings",
    "get_settings",
]
