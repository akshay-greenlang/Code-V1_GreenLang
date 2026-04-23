"""
GL-014 EXCHANGERPRO - API Dependencies

Dependency injection for FastAPI services.

Provides:
- Service factory functions for dependency injection
- Database session management
- Cache client management
- ML model loading
- Configuration management
"""

from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, Generator, Optional
import logging
import os

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer

logger = logging.getLogger(__name__)

# Security schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)

# Agent constants
AGENT_ID = "GL-014"
AGENT_VERSION = "1.0.0"


# =============================================================================
# Configuration
# =============================================================================

class Settings:
    """Application settings loaded from environment."""

    def __init__(self) -> None:
        # Database
        self.database_url: str = os.getenv(
            "DATABASE_URL",
            "postgresql://localhost:5432/exchangerpro"
        )
        self.database_pool_size: int = int(os.getenv("DATABASE_POOL_SIZE", "10"))

        # Redis cache
        self.redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "300"))

        # Authentication
        self.jwt_secret: str = os.getenv("JWT_SECRET", "")
        self.jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
        self.jwt_expiration_minutes: int = int(
            os.getenv("JWT_EXPIRATION_MINUTES", "30")
        )
        self.require_authentication: bool = os.getenv(
            "REQUIRE_AUTHENTICATION", "false"
        ).lower() == "true"

        # Rate limiting
        self.rate_limit_requests_per_minute: int = int(
            os.getenv("RATE_LIMIT_RPM", "60")
        )
        self.rate_limit_burst_size: int = int(
            os.getenv("RATE_LIMIT_BURST", "10")
        )

        # ML models
        self.fouling_model_path: str = os.getenv(
            "FOULING_MODEL_PATH",
            "./models/fouling_predictor.pkl"
        )
        self.optimization_model_path: str = os.getenv(
            "OPTIMIZATION_MODEL_PATH",
            "./models/cleaning_optimizer.pkl"
        )

        # Logging
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.log_request_body: bool = os.getenv(
            "LOG_REQUEST_BODY", "false"
        ).lower() == "true"
        self.audit_log_path: str = os.getenv(
            "AUDIT_LOG_PATH",
            "./logs/audit.log"
        )

        # Feature flags
        self.enable_explainability: bool = os.getenv(
            "ENABLE_EXPLAINABILITY", "true"
        ).lower() == "true"
        self.enable_what_if: bool = os.getenv(
            "ENABLE_WHAT_IF", "true"
        ).lower() == "true"

        # External services
        self.cmms_api_url: Optional[str] = os.getenv("CMMS_API_URL")
        self.historian_api_url: Optional[str] = os.getenv("HISTORIAN_API_URL")


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# =============================================================================
# User Model and Authentication
# =============================================================================

class User:
    """Authenticated user model."""

    def __init__(
        self,
        user_id: str,
        tenant_id: Optional[str] = None,
        email: Optional[str] = None,
        roles: Optional[list] = None,
        api_key_id: Optional[str] = None,
    ) -> None:
        self.user_id = user_id
        self.tenant_id = tenant_id
        self.email = email
        self.roles = roles or []
        self.api_key_id = api_key_id

    def has_role(self, role: str) -> bool:
        """Check if user has specified role."""
        return role in self.roles

    def has_any_role(self, roles: list) -> bool:
        """Check if user has any of the specified roles."""
        return any(role in self.roles for role in roles)


async def get_current_user(
    request: Request,
    api_key: Optional[str] = Depends(api_key_header),
    token: Optional[str] = Depends(oauth2_scheme),
    settings: Settings = Depends(get_settings),
) -> Optional[User]:
    """
    Get current authenticated user from request.

    Checks for authentication via:
    1. X-API-Key header
    2. Bearer token (OAuth2)
    3. Request state (set by middleware)

    Returns:
        User object if authenticated, None otherwise
    """
    # Check if user was set by middleware
    if hasattr(request.state, "user") and request.state.user:
        user_info = request.state.user
        return User(
            user_id=user_info.get("user_id", "unknown"),
            tenant_id=user_info.get("tenant_id"),
            email=user_info.get("email"),
            roles=user_info.get("roles", []),
            api_key_id=user_info.get("api_key_id"),
        )

    # If authentication required but no user, raise error
    if settings.require_authentication:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return None


async def require_user(
    user: Optional[User] = Depends(get_current_user),
) -> User:
    """
    Require authenticated user.

    Raises HTTPException if not authenticated.
    """
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_role(required_role: str):
    """
    Dependency factory requiring specific role.

    Usage:
        @router.get("/admin")
        async def admin_endpoint(user: User = Depends(require_role("admin"))):
            ...
    """
    async def role_checker(user: User = Depends(require_user)) -> User:
        if not user.has_role(required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required",
            )
        return user

    return role_checker


def require_any_role(required_roles: list):
    """
    Dependency factory requiring any of specified roles.

    Usage:
        @router.get("/operators")
        async def operator_endpoint(
            user: User = Depends(require_any_role(["operator", "admin"]))
        ):
            ...
    """
    async def role_checker(user: User = Depends(require_user)) -> User:
        if not user.has_any_role(required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"One of roles {required_roles} required",
            )
        return user

    return role_checker


# =============================================================================
# Service Dependencies
# =============================================================================

class ThermalCalculator:
    """
    Thermal KPIs calculator service.

    Placeholder for the actual thermal calculation engine.
    In production, this would be the deterministic calculation engine.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def compute_kpis(
        self,
        exchanger_id: str,
        hot_stream: Dict[str, Any],
        cold_stream: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Compute thermal KPIs for heat exchanger."""
        # Placeholder - actual implementation in core module
        raise NotImplementedError("ThermalCalculator.compute_kpis")

    async def compute_pressure_drop(
        self,
        exchanger_id: str,
        stream_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, float]:
        """Compute pressure drop."""
        raise NotImplementedError("ThermalCalculator.compute_pressure_drop")


class FoulingPredictor:
    """
    Fouling prediction service.

    Placeholder for the ML-based fouling prediction engine.
    """

    def __init__(self, settings: Settings, model_path: Optional[str] = None) -> None:
        self.settings = settings
        self.model_path = model_path or settings.fouling_model_path
        self._model = None

    async def load_model(self) -> None:
        """Load fouling prediction model."""
        # Placeholder - actual model loading
        pass

    async def predict(
        self,
        exchanger_id: str,
        current_state: Dict[str, float],
        operating_conditions: Dict[str, float],
        horizon_days: int,
    ) -> Dict[str, Any]:
        """Predict fouling progression."""
        raise NotImplementedError("FoulingPredictor.predict")

    async def get_forecast(
        self,
        exchanger_id: str,
    ) -> Dict[str, Any]:
        """Get stored fouling forecast."""
        raise NotImplementedError("FoulingPredictor.get_forecast")


class CleaningOptimizer:
    """
    Cleaning schedule optimization service.

    Placeholder for the cleaning optimization engine.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def optimize_schedule(
        self,
        exchanger_id: str,
        current_state: Dict[str, float],
        fouling_forecast: list,
        constraints: Dict[str, Any],
        objective: str,
        horizon_days: int,
    ) -> Dict[str, Any]:
        """Optimize cleaning schedule."""
        raise NotImplementedError("CleaningOptimizer.optimize_schedule")

    async def get_recommendations(
        self,
        exchanger_id: str,
    ) -> list:
        """Get cleaning recommendations."""
        raise NotImplementedError("CleaningOptimizer.get_recommendations")


class ExplainabilityService:
    """
    Explainability service for computations.

    Provides LIME, SHAP, and natural language explanations.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def explain_computation(
        self,
        computation_id: str,
        exchanger_id: str,
        explanation_type: str,
    ) -> Dict[str, Any]:
        """Generate explanation for a computation."""
        raise NotImplementedError("ExplainabilityService.explain_computation")


class WhatIfAnalyzer:
    """
    What-if scenario analysis service.
    """

    def __init__(
        self,
        settings: Settings,
        calculator: ThermalCalculator,
    ) -> None:
        self.settings = settings
        self.calculator = calculator

    async def analyze_scenarios(
        self,
        exchanger_id: str,
        base_conditions: Dict[str, float],
        scenarios: list,
        kpis: list,
    ) -> Dict[str, Any]:
        """Analyze what-if scenarios."""
        raise NotImplementedError("WhatIfAnalyzer.analyze_scenarios")


class AuditService:
    """
    Audit trail service.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def get_audit_trail(
        self,
        exchanger_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[list] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Get audit trail for exchanger."""
        raise NotImplementedError("AuditService.get_audit_trail")

    async def log_event(
        self,
        exchanger_id: str,
        event_type: str,
        event_data: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> str:
        """Log audit event."""
        raise NotImplementedError("AuditService.log_event")


class HistoricalDataService:
    """
    Historical KPI data service.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def get_kpi_history(
        self,
        exchanger_id: str,
        kpis: list,
        start_time: datetime,
        end_time: datetime,
        resolution: str,
    ) -> Dict[str, Any]:
        """Get historical KPI data."""
        raise NotImplementedError("HistoricalDataService.get_kpi_history")


class MetricsService:
    """
    Prometheus metrics service.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._start_time = datetime.now(timezone.utc)

    def get_uptime_seconds(self) -> float:
        """Get application uptime in seconds."""
        delta = datetime.now(timezone.utc) - self._start_time
        return delta.total_seconds()

    async def get_metrics(self) -> list:
        """Get current metrics."""
        # Placeholder metrics
        return [
            {
                "name": "exchangerpro_uptime_seconds",
                "value": self.get_uptime_seconds(),
                "labels": {"agent_id": AGENT_ID},
                "timestamp": datetime.now(timezone.utc),
            },
            {
                "name": "exchangerpro_requests_total",
                "value": 0,  # Placeholder
                "labels": {"agent_id": AGENT_ID},
                "timestamp": datetime.now(timezone.utc),
            },
        ]


# =============================================================================
# Dependency Injection Functions
# =============================================================================

def get_thermal_calculator(
    settings: Settings = Depends(get_settings),
) -> ThermalCalculator:
    """Get thermal calculator service."""
    return ThermalCalculator(settings)


def get_fouling_predictor(
    settings: Settings = Depends(get_settings),
) -> FoulingPredictor:
    """Get fouling predictor service."""
    return FoulingPredictor(settings)


def get_cleaning_optimizer(
    settings: Settings = Depends(get_settings),
) -> CleaningOptimizer:
    """Get cleaning optimizer service."""
    return CleaningOptimizer(settings)


def get_explainability_service(
    settings: Settings = Depends(get_settings),
) -> ExplainabilityService:
    """Get explainability service."""
    return ExplainabilityService(settings)


def get_what_if_analyzer(
    settings: Settings = Depends(get_settings),
    calculator: ThermalCalculator = Depends(get_thermal_calculator),
) -> WhatIfAnalyzer:
    """Get what-if analyzer service."""
    return WhatIfAnalyzer(settings, calculator)


def get_audit_service(
    settings: Settings = Depends(get_settings),
) -> AuditService:
    """Get audit service."""
    return AuditService(settings)


def get_historical_data_service(
    settings: Settings = Depends(get_settings),
) -> HistoricalDataService:
    """Get historical data service."""
    return HistoricalDataService(settings)


_metrics_service: Optional[MetricsService] = None


def get_metrics_service(
    settings: Settings = Depends(get_settings),
) -> MetricsService:
    """Get metrics service (singleton)."""
    global _metrics_service
    if _metrics_service is None:
        _metrics_service = MetricsService(settings)
    return _metrics_service


# =============================================================================
# Request Context Dependencies
# =============================================================================

async def get_request_id(request: Request) -> str:
    """Get request ID from request state."""
    return getattr(request.state, "request_id", "unknown")


async def get_correlation_id(request: Request) -> str:
    """Get correlation ID from request state."""
    return getattr(request.state, "correlation_id", "unknown")


# =============================================================================
# Exchanger Validation Dependencies
# =============================================================================

async def validate_exchanger_exists(
    exchanger_id: str,
    # repository: ExchangerRepository = Depends(get_exchanger_repository),
) -> str:
    """
    Validate that exchanger exists.

    In production, this would query the database.
    """
    # Placeholder - actual validation would query database
    if not exchanger_id or len(exchanger_id) < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid exchanger ID",
        )

    # For now, accept any non-empty ID
    # In production: repository.exists(exchanger_id)
    return exchanger_id


async def validate_exchanger_access(
    exchanger_id: str = Depends(validate_exchanger_exists),
    user: Optional[User] = Depends(get_current_user),
) -> str:
    """
    Validate user has access to exchanger.

    In production, this would check tenant isolation.
    """
    # Placeholder - actual implementation would check permissions
    # if user and user.tenant_id:
    #     if not repository.user_has_access(exchanger_id, user.tenant_id):
    #         raise HTTPException(status_code=403, detail="Access denied")
    return exchanger_id


# =============================================================================
# Export all dependencies
# =============================================================================

__all__ = [
    # Settings
    "Settings",
    "get_settings",
    # User/Auth
    "User",
    "get_current_user",
    "require_user",
    "require_role",
    "require_any_role",
    # Services
    "ThermalCalculator",
    "FoulingPredictor",
    "CleaningOptimizer",
    "ExplainabilityService",
    "WhatIfAnalyzer",
    "AuditService",
    "HistoricalDataService",
    "MetricsService",
    # Dependency functions
    "get_thermal_calculator",
    "get_fouling_predictor",
    "get_cleaning_optimizer",
    "get_explainability_service",
    "get_what_if_analyzer",
    "get_audit_service",
    "get_historical_data_service",
    "get_metrics_service",
    # Request context
    "get_request_id",
    "get_correlation_id",
    # Validation
    "validate_exchanger_exists",
    "validate_exchanger_access",
]
