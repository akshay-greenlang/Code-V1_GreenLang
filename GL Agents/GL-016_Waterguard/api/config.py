"""
GL-016_Waterguard API Configuration

Centralized configuration for the Waterguard API server including
host/port settings, authentication, rate limiting, and CORS policies.

Author: GL-APIDeveloper
Version: 1.0.0
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional
from functools import lru_cache


@dataclass
class RateLimitConfig:
    """Rate limiting configuration for API endpoints."""

    # Global rate limits
    requests_per_minute: int = 100
    requests_per_hour: int = 1000

    # Per-endpoint rate limits
    optimize_per_minute: int = 10
    status_per_minute: int = 60
    recommendations_per_minute: int = 30
    health_per_minute: int = 120

    # Burst limits
    burst_size: int = 20

    # Sliding window configuration
    window_size_seconds: int = 60

    def get_limit_string(self, endpoint: str) -> str:
        """Get rate limit string for an endpoint in format 'N/period'."""
        limits = {
            "optimize": f"{self.optimize_per_minute}/minute",
            "status": f"{self.status_per_minute}/minute",
            "recommendations": f"{self.recommendations_per_minute}/minute",
            "health": f"{self.health_per_minute}/minute",
        }
        return limits.get(endpoint, f"{self.requests_per_minute}/minute")


@dataclass
class CORSConfig:
    """CORS configuration for cross-origin requests."""

    # Allowed origins (production domains)
    allowed_origins: List[str] = field(default_factory=lambda: [
        "https://waterguard.greenlang.io",
        "https://app.greenlang.io",
        "https://admin.greenlang.io",
    ])

    # Development origins (enabled only in dev mode)
    dev_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
    ])

    # Allowed methods
    allowed_methods: List[str] = field(default_factory=lambda: [
        "GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"
    ])

    # Allowed headers
    allowed_headers: List[str] = field(default_factory=lambda: [
        "Authorization",
        "Content-Type",
        "X-Request-ID",
        "X-API-Key",
        "X-Correlation-ID",
    ])

    # Exposed headers (visible to JavaScript)
    expose_headers: List[str] = field(default_factory=lambda: [
        "X-Request-ID",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
    ])

    # Credentials support
    allow_credentials: bool = True

    # Preflight cache time (seconds)
    max_age: int = 600

    def get_all_origins(self, include_dev: bool = False) -> List[str]:
        """Get all allowed origins based on environment."""
        origins = self.allowed_origins.copy()
        if include_dev:
            origins.extend(self.dev_origins)
        return origins


@dataclass
class JWTConfig:
    """JWT authentication configuration."""

    # JWT settings
    secret_key: str = field(default_factory=lambda: os.getenv(
        "WATERGUARD_JWT_SECRET",
        "development-secret-change-in-production"
    ))
    algorithm: str = "HS256"

    # Token expiration (seconds)
    access_token_expire_seconds: int = 3600  # 1 hour
    refresh_token_expire_seconds: int = 604800  # 7 days

    # Token issuer/audience
    issuer: str = "waterguard.greenlang.io"
    audience: str = "waterguard-api"

    # Token type header
    token_type: str = "Bearer"

    # API key settings (alternative auth)
    api_key_header: str = "X-API-Key"
    api_key_prefix: str = "wg_"


@dataclass
class ServerConfig:
    """Server configuration for the API."""

    # Host and port
    host: str = field(default_factory=lambda: os.getenv("WATERGUARD_API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("WATERGUARD_API_PORT", "8000")))

    # Worker configuration
    workers: int = field(default_factory=lambda: int(os.getenv("WATERGUARD_API_WORKERS", "4")))

    # Debug mode
    debug: bool = field(default_factory=lambda: os.getenv("WATERGUARD_DEBUG", "false").lower() == "true")

    # Reload on changes (development only)
    reload: bool = field(default_factory=lambda: os.getenv("WATERGUARD_RELOAD", "false").lower() == "true")

    # Trusted hosts
    trusted_hosts: List[str] = field(default_factory=lambda: [
        "*.greenlang.io",
        "localhost",
        "127.0.0.1",
    ])

    # Request limits
    max_request_size_mb: int = 10
    request_timeout_seconds: int = 60


@dataclass
class LoggingConfig:
    """Logging configuration for API."""

    # Log level
    level: str = field(default_factory=lambda: os.getenv("WATERGUARD_LOG_LEVEL", "INFO"))

    # Log format
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # JSON logging for production
    json_logs: bool = field(default_factory=lambda: os.getenv("WATERGUARD_JSON_LOGS", "false").lower() == "true")

    # Access log configuration
    access_log: bool = True
    access_log_format: str = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

    # Audit log settings
    audit_log_enabled: bool = True
    audit_log_path: str = field(default_factory=lambda: os.getenv(
        "WATERGUARD_AUDIT_LOG_PATH",
        "/var/log/waterguard/audit.log"
    ))


@dataclass
class APIConfig:
    """
    Main API configuration aggregating all sub-configurations.

    Provides centralized access to all API settings including server,
    authentication, rate limiting, CORS, and logging.
    """

    # Sub-configurations
    server: ServerConfig = field(default_factory=ServerConfig)
    jwt: JWTConfig = field(default_factory=JWTConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    cors: CORSConfig = field(default_factory=CORSConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # API metadata
    api_title: str = "GL-016_Waterguard API"
    api_description: str = """
    Waterguard Cooling Tower Optimization API

    Multi-protocol API for cooling tower water chemistry optimization,
    providing real-time monitoring, ML-based recommendations, and
    compliance reporting for industrial water systems.

    ## Features

    - **Water Chemistry Monitoring**: Real-time pH, conductivity, TDS tracking
    - **Optimization Engine**: ML-driven setpoint recommendations
    - **Blowdown Control**: Automated cycles of concentration management
    - **Dosing Control**: Chemical injection rate optimization
    - **Compliance Reporting**: Regulatory compliance tracking
    - **Energy/Water Savings**: Resource conservation metrics

    ## Authentication

    All endpoints require JWT authentication via Bearer token.
    Include the token in the Authorization header:

    ```
    Authorization: Bearer <your-token>
    ```

    ## Rate Limiting

    Rate limits are enforced per user and per endpoint.
    Check X-RateLimit-* headers for current limits.
    """
    api_version: str = "1.0.0"

    # OpenAPI settings
    openapi_url: str = "/api/openapi.json"
    docs_url: str = "/api/docs"
    redoc_url: str = "/api/redoc"

    # GraphQL settings
    graphql_path: str = "/graphql"
    graphql_playground: bool = True

    # gRPC settings
    grpc_port: int = field(default_factory=lambda: int(os.getenv("WATERGUARD_GRPC_PORT", "50051")))
    grpc_max_message_size_mb: int = 10

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.server.debug or self.server.reload

    @property
    def cors_origins(self) -> List[str]:
        """Get CORS origins based on environment."""
        return self.cors.get_all_origins(include_dev=self.is_development)

    def get_uvicorn_config(self) -> dict:
        """Get configuration dictionary for uvicorn server."""
        return {
            "host": self.server.host,
            "port": self.server.port,
            "workers": self.server.workers if not self.server.reload else 1,
            "reload": self.server.reload,
            "log_level": self.logging.level.lower(),
            "access_log": self.logging.access_log,
        }


@lru_cache()
def get_api_config() -> APIConfig:
    """
    Get cached API configuration instance.

    Returns:
        APIConfig: Singleton configuration instance
    """
    return APIConfig()


def get_api_settings() -> dict:
    """
    Get API settings as a dictionary for FastAPI.

    Returns:
        dict: Settings dictionary for FastAPI app initialization
    """
    config = get_api_config()
    return {
        "title": config.api_title,
        "description": config.api_description,
        "version": config.api_version,
        "openapi_url": config.openapi_url,
        "docs_url": config.docs_url,
        "redoc_url": config.redoc_url,
        "debug": config.server.debug,
    }
