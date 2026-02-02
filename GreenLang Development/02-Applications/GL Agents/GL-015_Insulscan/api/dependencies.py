"""
GL-015 INSULSCAN - API Dependencies

Dependency injection for FastAPI services.

Provides:
- get_orchestrator() - Dependency injection for INSULSCAN orchestrator
- get_settings() - Application settings dependency
- verify_api_key() - API key authentication
- rate_limiter() - Rate limiting dependency
- Service factory functions for thermal analysis, hot spot detection, etc.
"""

from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional
import hashlib
import logging
import os
import time

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer

logger = logging.getLogger(__name__)

# Security schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)

# Agent constants
AGENT_ID = "GL-015"
AGENT_NAME = "INSULSCAN"
AGENT_VERSION = "1.0.0"


# =============================================================================
# Configuration / Settings
# =============================================================================

class Settings:
    """Application settings loaded from environment variables."""

    def __init__(self) -> None:
        # Database
        self.database_url: str = os.getenv(
            "DATABASE_URL",
            "postgresql://localhost:5432/insulscan"
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

        # API Keys (comma-separated list for development)
        self.api_keys: str = os.getenv("API_KEYS", "")

        # Rate limiting
        self.rate_limit_requests_per_minute: int = int(
            os.getenv("RATE_LIMIT_RPM", "60")
        )
        self.rate_limit_burst_size: int = int(
            os.getenv("RATE_LIMIT_BURST", "10")
        )

        # Thermal analysis settings
        self.default_ambient_temperature_C: float = float(
            os.getenv("DEFAULT_AMBIENT_TEMP_C", "25.0")
        )
        self.default_emissivity: float = float(
            os.getenv("DEFAULT_EMISSIVITY", "0.95")
        )
        self.hotspot_threshold_C: float = float(
            os.getenv("HOTSPOT_THRESHOLD_C", "5.0")
        )

        # ML models
        self.thermal_model_path: str = os.getenv(
            "THERMAL_MODEL_PATH",
            "./models/thermal_analyzer.pkl"
        )
        self.hotspot_model_path: str = os.getenv(
            "HOTSPOT_MODEL_PATH",
            "./models/hotspot_detector.pkl"
        )
        self.degradation_model_path: str = os.getenv(
            "DEGRADATION_MODEL_PATH",
            "./models/degradation_predictor.pkl"
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
        self.enable_batch_processing: bool = os.getenv(
            "ENABLE_BATCH_PROCESSING", "true"
        ).lower() == "true"
        self.max_batch_size: int = int(os.getenv("MAX_BATCH_SIZE", "100"))

        # Energy cost defaults
        self.default_energy_cost_usd_per_kwh: float = float(
            os.getenv("DEFAULT_ENERGY_COST_USD_KWH", "0.10")
        )

        # External services
        self.cmms_api_url: Optional[str] = os.getenv("CMMS_API_URL")
        self.historian_api_url: Optional[str] = os.getenv("HISTORIAN_API_URL")
        self.thermal_camera_api_url: Optional[str] = os.getenv("THERMAL_CAMERA_API_URL")


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
        roles: Optional[List[str]] = None,
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

    def has_any_role(self, roles: List[str]) -> bool:
        """Check if user has any of the specified roles."""
        return any(role in self.roles for role in roles)


async def verify_api_key(
    request: Request,
    api_key: Optional[str] = Depends(api_key_header),
    settings: Settings = Depends(get_settings),
) -> Optional[User]:
    """
    Verify API key and return user information.

    This is the primary authentication dependency for API endpoints.

    Args:
        request: FastAPI request object
        api_key: API key from X-API-Key header
        settings: Application settings

    Returns:
        User object if authenticated, None if no auth provided

    Raises:
        HTTPException: 401 if authentication required but invalid
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

    # Validate API key if provided
    if api_key:
        valid_keys = _parse_api_keys(settings.api_keys)
        if api_key in valid_keys:
            return User(
                user_id=valid_keys[api_key].get("user_id", "api_user"),
                tenant_id=valid_keys[api_key].get("tenant_id"),
                roles=valid_keys[api_key].get("roles", ["user"]),
                api_key_id=valid_keys[api_key].get("key_id", "default"),
            )
        elif settings.require_authentication:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )

    # If authentication required but no valid key provided
    if settings.require_authentication:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return None


def _parse_api_keys(api_keys_str: str) -> Dict[str, Dict[str, Any]]:
    """Parse API keys from environment variable."""
    keys = {}
    if not api_keys_str:
        # Add development key if DEBUG mode
        if os.getenv("DEBUG", "false").lower() == "true":
            keys["dev-api-key-12345"] = {
                "user_id": "dev_user",
                "tenant_id": "dev_tenant",
                "roles": ["admin", "user"],
                "key_id": "key_dev",
            }
        return keys

    # Parse format: KEY1:USER_ID:TENANT_ID:ROLES,KEY2:USER_ID:TENANT_ID:ROLES
    for key_spec in api_keys_str.split(","):
        parts = key_spec.strip().split(":")
        if len(parts) >= 2:
            api_key = parts[0]
            user_id = parts[1]
            tenant_id = parts[2] if len(parts) > 2 else None
            roles = parts[3].split(";") if len(parts) > 3 else ["user"]

            keys[api_key] = {
                "user_id": user_id,
                "tenant_id": tenant_id,
                "roles": roles,
                "key_id": f"key_{hashlib.sha256(api_key.encode()).hexdigest()[:8]}",
            }

    return keys


async def get_current_user(
    request: Request,
    api_key: Optional[str] = Depends(api_key_header),
    token: Optional[str] = Depends(oauth2_scheme),
    settings: Settings = Depends(get_settings),
) -> Optional[User]:
    """
    Get current authenticated user from request.

    Supports both API key and OAuth2 Bearer token authentication.

    Returns:
        User object if authenticated, None otherwise
    """
    # First try API key
    user = await verify_api_key(request, api_key, settings)
    if user:
        return user

    # Try JWT token if provided
    if token and settings.jwt_secret:
        try:
            import jwt
            payload = jwt.decode(
                token,
                settings.jwt_secret,
                algorithms=[settings.jwt_algorithm]
            )
            return User(
                user_id=payload.get("sub", "unknown"),
                tenant_id=payload.get("tenant_id"),
                email=payload.get("email"),
                roles=payload.get("roles", []),
            )
        except Exception as e:
            logger.debug(f"JWT validation failed: {e}")
            if settings.require_authentication:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token",
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
    """Dependency factory requiring specific role."""
    async def role_checker(user: User = Depends(require_user)) -> User:
        if not user.has_role(required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required",
            )
        return user
    return role_checker


# =============================================================================
# Rate Limiting
# =============================================================================

class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10,
    ) -> None:
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.tokens: Dict[str, float] = {}
        self.last_update: Dict[str, float] = {}

    def check_rate_limit(self, client_id: str) -> bool:
        """Check if request is within rate limit."""
        now = time.time()

        if client_id not in self.tokens:
            self.tokens[client_id] = float(self.burst_size)
            self.last_update[client_id] = now

        # Add tokens based on time elapsed
        elapsed = now - self.last_update[client_id]
        token_rate = self.requests_per_minute / 60.0
        self.tokens[client_id] = min(
            self.burst_size,
            self.tokens[client_id] + elapsed * token_rate
        )
        self.last_update[client_id] = now

        # Check if token available
        if self.tokens[client_id] >= 1.0:
            self.tokens[client_id] -= 1.0
            return True

        return False

    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client."""
        return max(0, int(self.tokens.get(client_id, 0)))

    def get_retry_after(self, client_id: str) -> float:
        """Calculate seconds until next request allowed."""
        tokens_needed = 1.0 - self.tokens.get(client_id, 0)
        token_rate = self.requests_per_minute / 60.0
        return tokens_needed / token_rate if token_rate > 0 else 60.0


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter(
    settings: Settings = Depends(get_settings),
) -> RateLimiter:
    """Get rate limiter instance (singleton)."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(
            requests_per_minute=settings.rate_limit_requests_per_minute,
            burst_size=settings.rate_limit_burst_size,
        )
    return _rate_limiter


async def rate_limiter(
    request: Request,
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> None:
    """
    Rate limiting dependency.

    Call this dependency to enforce rate limits on endpoints.

    Raises:
        HTTPException: 429 if rate limit exceeded
    """
    # Get client identifier
    client_id = _get_client_id(request)

    if not limiter.check_rate_limit(client_id):
        retry_after = limiter.get_retry_after(client_id)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Retry after {int(retry_after)} seconds.",
            headers={
                "Retry-After": str(int(retry_after)),
                "X-RateLimit-Limit": str(limiter.requests_per_minute),
                "X-RateLimit-Remaining": "0",
            },
        )


def _get_client_id(request: Request) -> str:
    """Get client identifier for rate limiting."""
    # Check for API key first
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"key:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"

    # Fall back to IP
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return f"ip:{forwarded.split(',')[0].strip()}"

    client_ip = request.client.host if request.client else "unknown"
    return f"ip:{client_ip}"


# =============================================================================
# Service Classes (Placeholder implementations)
# =============================================================================

class InsulationOrchestrator:
    """
    Main orchestrator for INSULSCAN operations.

    Coordinates thermal analysis, hot spot detection, and recommendations.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize orchestrator and load models."""
        if not self._initialized:
            logger.info("Initializing INSULSCAN orchestrator...")
            # Load models, initialize connections, etc.
            self._initialized = True

    async def analyze_insulation(
        self,
        asset_id: str,
        measurement_data: Optional[Dict[str, float]] = None,
        include_recommendations: bool = True,
        include_explanations: bool = False,
    ) -> Dict[str, Any]:
        """
        Analyze insulation condition for an asset.

        This is a placeholder - actual implementation would use the core module.
        """
        raise NotImplementedError("InsulationOrchestrator.analyze_insulation")

    async def detect_hotspots(
        self,
        image_data: bytes,
        calibration_params: Dict[str, float],
    ) -> Dict[str, Any]:
        """Detect hot spots in thermal image."""
        raise NotImplementedError("InsulationOrchestrator.detect_hotspots")

    async def get_asset_condition(self, asset_id: str) -> Dict[str, Any]:
        """Get current condition for an asset."""
        raise NotImplementedError("InsulationOrchestrator.get_asset_condition")

    async def get_asset_history(
        self,
        asset_id: str,
        start_date: datetime,
        end_date: datetime,
        resolution: str = "1h",
    ) -> Dict[str, Any]:
        """Get historical data for an asset."""
        raise NotImplementedError("InsulationOrchestrator.get_asset_history")

    async def generate_recommendations(
        self,
        asset_ids: Optional[List[str]] = None,
        budget_limit: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate repair recommendations."""
        raise NotImplementedError("InsulationOrchestrator.generate_recommendations")


class ThermalAnalyzer:
    """Thermal analysis service for insulation assessment."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def calculate_heat_loss(
        self,
        surface_temp_C: float,
        ambient_temp_C: float,
        process_temp_C: Optional[float] = None,
        insulation_thickness_mm: Optional[float] = None,
        surface_area_m2: float = 1.0,
    ) -> Dict[str, float]:
        """Calculate heat loss through insulation."""
        raise NotImplementedError("ThermalAnalyzer.calculate_heat_loss")

    async def calculate_effectiveness(
        self,
        actual_heat_loss: float,
        design_heat_loss: float,
    ) -> float:
        """Calculate insulation effectiveness percentage."""
        if design_heat_loss <= 0:
            return 0.0
        return max(0.0, min(100.0, (1 - actual_heat_loss / design_heat_loss) * 100))


class HotSpotDetector:
    """Hot spot detection service for thermal images."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def detect_hotspots(
        self,
        image_data: bytes,
        calibration_params: Dict[str, float],
        threshold_C: float = 5.0,
    ) -> List[Dict[str, Any]]:
        """Detect hot spots in thermal image."""
        raise NotImplementedError("HotSpotDetector.detect_hotspots")


class RecommendationEngine:
    """Recommendation engine for repair and maintenance."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def generate_recommendations(
        self,
        analysis_results: List[Dict[str, Any]],
        budget_limit: Optional[float] = None,
        optimization_objective: str = "cost_benefit",
    ) -> List[Dict[str, Any]]:
        """Generate repair recommendations based on analysis results."""
        raise NotImplementedError("RecommendationEngine.generate_recommendations")


class MetricsService:
    """Prometheus metrics service."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._start_time = datetime.now(timezone.utc)
        self._request_count = 0
        self._analysis_count = 0
        self._hotspot_detections = 0

    def get_uptime_seconds(self) -> float:
        """Get application uptime in seconds."""
        delta = datetime.now(timezone.utc) - self._start_time
        return delta.total_seconds()

    def increment_request_count(self) -> None:
        """Increment request counter."""
        self._request_count += 1

    def increment_analysis_count(self) -> None:
        """Increment analysis counter."""
        self._analysis_count += 1

    def increment_hotspot_count(self, count: int = 1) -> None:
        """Increment hotspot detection counter."""
        self._hotspot_detections += count

    async def get_metrics(self) -> List[Dict[str, Any]]:
        """Get current metrics."""
        return [
            {
                "name": "insulscan_uptime_seconds",
                "value": self.get_uptime_seconds(),
                "labels": {"agent_id": AGENT_ID},
                "type": "gauge",
                "help": "INSULSCAN agent uptime in seconds",
            },
            {
                "name": "insulscan_requests_total",
                "value": self._request_count,
                "labels": {"agent_id": AGENT_ID},
                "type": "counter",
                "help": "Total number of API requests",
            },
            {
                "name": "insulscan_analyses_total",
                "value": self._analysis_count,
                "labels": {"agent_id": AGENT_ID},
                "type": "counter",
                "help": "Total number of insulation analyses",
            },
            {
                "name": "insulscan_hotspots_detected_total",
                "value": self._hotspot_detections,
                "labels": {"agent_id": AGENT_ID},
                "type": "counter",
                "help": "Total number of hot spots detected",
            },
        ]


class HistoricalDataService:
    """Historical data service for time-series queries."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def get_asset_history(
        self,
        asset_id: str,
        start_date: datetime,
        end_date: datetime,
        resolution: str = "1h",
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get historical data for an asset."""
        raise NotImplementedError("HistoricalDataService.get_asset_history")


class AuditService:
    """Audit logging service."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def log_event(
        self,
        event_type: str,
        asset_id: Optional[str] = None,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log an audit event."""
        raise NotImplementedError("AuditService.log_event")


# =============================================================================
# Dependency Injection Functions
# =============================================================================

# Singleton instances
_orchestrator: Optional[InsulationOrchestrator] = None
_metrics_service: Optional[MetricsService] = None


def get_orchestrator(
    settings: Settings = Depends(get_settings),
) -> InsulationOrchestrator:
    """Get INSULSCAN orchestrator instance (singleton)."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = InsulationOrchestrator(settings)
    return _orchestrator


def get_thermal_analyzer(
    settings: Settings = Depends(get_settings),
) -> ThermalAnalyzer:
    """Get thermal analyzer service."""
    return ThermalAnalyzer(settings)


def get_hotspot_detector(
    settings: Settings = Depends(get_settings),
) -> HotSpotDetector:
    """Get hot spot detector service."""
    return HotSpotDetector(settings)


def get_recommendation_engine(
    settings: Settings = Depends(get_settings),
) -> RecommendationEngine:
    """Get recommendation engine service."""
    return RecommendationEngine(settings)


def get_metrics_service(
    settings: Settings = Depends(get_settings),
) -> MetricsService:
    """Get metrics service (singleton)."""
    global _metrics_service
    if _metrics_service is None:
        _metrics_service = MetricsService(settings)
    return _metrics_service


def get_historical_data_service(
    settings: Settings = Depends(get_settings),
) -> HistoricalDataService:
    """Get historical data service."""
    return HistoricalDataService(settings)


def get_audit_service(
    settings: Settings = Depends(get_settings),
) -> AuditService:
    """Get audit service."""
    return AuditService(settings)


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
# Asset Validation Dependencies
# =============================================================================

async def validate_asset_exists(
    asset_id: str,
    # repository: AssetRepository = Depends(get_asset_repository),
) -> str:
    """
    Validate that asset exists.

    In production, this would query the database.
    """
    if not asset_id or len(asset_id) < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid asset ID",
        )

    # For now, accept any non-empty ID
    # In production: repository.exists(asset_id)
    return asset_id


async def validate_asset_access(
    asset_id: str = Depends(validate_asset_exists),
    user: Optional[User] = Depends(get_current_user),
) -> str:
    """
    Validate user has access to asset.

    In production, this would check tenant isolation.
    """
    # Placeholder - actual implementation would check permissions
    return asset_id


# =============================================================================
# Export all dependencies
# =============================================================================

__all__ = [
    # Settings
    "Settings",
    "get_settings",
    # User/Auth
    "User",
    "verify_api_key",
    "get_current_user",
    "require_user",
    "require_role",
    # Rate Limiting
    "RateLimiter",
    "get_rate_limiter",
    "rate_limiter",
    # Services
    "InsulationOrchestrator",
    "ThermalAnalyzer",
    "HotSpotDetector",
    "RecommendationEngine",
    "MetricsService",
    "HistoricalDataService",
    "AuditService",
    # Dependency functions
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
    # Constants
    "AGENT_ID",
    "AGENT_NAME",
    "AGENT_VERSION",
]
