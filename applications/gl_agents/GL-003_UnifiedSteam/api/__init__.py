"""
GL-003 UnifiedSteam SteamSystemOptimizer API Layer

Comprehensive GraphQL/gRPC/REST API services for steam system optimization.
Implements GreenLang standard patterns: mTLS + OAuth2/JWT auth, rate limiting, audit trails.

Features:
- Steam properties computation (IAPWS IF97)
- Enthalpy and mass balance calculations
- Desuperheater, condensate recovery, and network optimization
- Steam trap diagnostics and failure prediction
- Root cause analysis with causal inference
- KPI dashboards and climate impact reporting
- SHAP/LIME explainability for recommendations
"""

from .api_schemas import (
    # Steam Properties
    SteamPropertiesRequest,
    SteamPropertiesResponse,
    SteamState,
    # Balance Calculations
    EnthalpyBalanceRequest,
    EnthalpyBalanceResponse,
    MassBalanceRequest,
    MassBalanceResponse,
    # Optimization
    DesuperheaterOptimizationRequest,
    DesuperheaterOptimizationResponse,
    CondensateOptimizationRequest,
    CondensateOptimizationResponse,
    NetworkOptimizationRequest,
    NetworkOptimizationResponse,
    OptimizationRecommendation,
    # Trap Diagnostics
    TrapDiagnosticsRequest,
    TrapDiagnosticsResponse,
    TrapStatus,
    TrapFailurePrediction,
    # Root Cause Analysis
    RCARequest,
    RCAResponse,
    CausalFactor,
    CounterfactualScenario,
    # Explainability
    ExplainabilityRequest,
    ExplainabilityResponse,
    FeatureContribution,
    # KPIs and Climate Impact
    KPIDashboardResponse,
    ClimateImpactResponse,
    EnergyMetrics,
    EmissionsMetrics,
    # Common
    ErrorResponse,
    PaginationParams,
    TimeRangeFilter,
)

from .api_auth import (
    get_current_user,
    verify_token,
    authorize_action,
    SteamSystemUser,
    Role,
    Permission,
    require_permissions,
    require_roles,
)

from .graphql_api import (
    graphql_app,
    schema,
)

from .grpc_services import (
    SteamPropertiesServicer,
    OptimizationServicer,
    InferenceServicer,
    RCAServicer,
    serve_grpc,
)

from .rest_api import (
    router as rest_router,
)

from .config import (
    Settings,
    get_settings,
)

__version__ = "1.0.0"
__all__ = [
    # Schemas - Steam Properties
    "SteamPropertiesRequest",
    "SteamPropertiesResponse",
    "SteamState",
    # Schemas - Balance
    "EnthalpyBalanceRequest",
    "EnthalpyBalanceResponse",
    "MassBalanceRequest",
    "MassBalanceResponse",
    # Schemas - Optimization
    "DesuperheaterOptimizationRequest",
    "DesuperheaterOptimizationResponse",
    "CondensateOptimizationRequest",
    "CondensateOptimizationResponse",
    "NetworkOptimizationRequest",
    "NetworkOptimizationResponse",
    "OptimizationRecommendation",
    # Schemas - Trap Diagnostics
    "TrapDiagnosticsRequest",
    "TrapDiagnosticsResponse",
    "TrapStatus",
    "TrapFailurePrediction",
    # Schemas - RCA
    "RCARequest",
    "RCAResponse",
    "CausalFactor",
    "CounterfactualScenario",
    # Schemas - Explainability
    "ExplainabilityRequest",
    "ExplainabilityResponse",
    "FeatureContribution",
    # Schemas - KPIs
    "KPIDashboardResponse",
    "ClimateImpactResponse",
    "EnergyMetrics",
    "EmissionsMetrics",
    # Schemas - Common
    "ErrorResponse",
    "PaginationParams",
    "TimeRangeFilter",
    # Auth
    "get_current_user",
    "verify_token",
    "authorize_action",
    "SteamSystemUser",
    "Role",
    "Permission",
    "require_permissions",
    "require_roles",
    # GraphQL
    "graphql_app",
    "schema",
    # gRPC
    "SteamPropertiesServicer",
    "OptimizationServicer",
    "InferenceServicer",
    "RCAServicer",
    "serve_grpc",
    # REST
    "rest_router",
    # Config
    "Settings",
    "get_settings",
]
