"""
GL-004 BURNMASTER API Configuration

Configuration settings for the Burner Optimization API including
server settings, security, rate limiting, and CORS configuration.
"""

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings
from typing import List, Optional
from functools import lru_cache
import os


class ServerConfig(BaseModel):
    """Server configuration settings."""
    host: str = Field(default="0.0.0.0", description="Server host address")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port number")
    workers: int = Field(default=4, ge=1, le=32, description="Number of worker processes")
    reload: bool = Field(default=False, description="Enable auto-reload for development")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="info", description="Logging level")
    request_timeout: int = Field(default=60, ge=1, le=300, description="Request timeout in seconds")
    max_request_size: int = Field(default=10 * 1024 * 1024, description="Maximum request body size")


class SecurityConfig(BaseModel):
    """Security and authentication settings."""
    jwt_secret_key: SecretStr = Field(default=SecretStr("your-super-secret-key-change-in-production"))
    jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    jwt_access_token_expire_minutes: int = Field(default=30, ge=5, le=1440)
    jwt_refresh_token_expire_days: int = Field(default=7, ge=1, le=30)
    api_key_header: str = Field(default="X-API-Key", description="Header name for API key")
    api_key_prefix: str = Field(default="gl_", description="Prefix for API keys")
    enable_api_key_auth: bool = Field(default=True, description="Enable API key authentication")
    password_min_length: int = Field(default=12, ge=8, le=128)
    max_login_attempts: int = Field(default=5, ge=1, le=10)
    lockout_duration_minutes: int = Field(default=15, ge=5, le=60)
    session_idle_timeout_minutes: int = Field(default=30, ge=5, le=120)


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    enabled: bool = Field(default=True, description="Enable rate limiting")
    default_limit: str = Field(default="100/minute", description="Default rate limit")
    status_endpoint_limit: str = Field(default="1000/minute")
    recommendations_endpoint_limit: str = Field(default="200/minute")
    action_endpoint_limit: str = Field(default="50/minute")
    history_endpoint_limit: str = Field(default="30/minute")
    health_endpoint_limit: str = Field(default="1000/minute")
    burst_multiplier: float = Field(default=1.5, ge=1.0, le=3.0)
    storage_backend: str = Field(default="memory", description="Rate limit storage backend")
    redis_url: Optional[str] = Field(default=None, description="Redis URL for distributed rate limiting")


class CORSConfig(BaseModel):
    """CORS configuration."""
    enabled: bool = Field(default=True, description="Enable CORS")
    allow_origins: List[str] = Field(default=["https://*.greenlang.io", "http://localhost:3000"])
    allow_credentials: bool = Field(default=True)
    allow_methods: List[str] = Field(default=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
    allow_headers: List[str] = Field(default=["*"])
    expose_headers: List[str] = Field(default=["X-Request-ID", "X-RateLimit-Remaining", "X-RateLimit-Reset"])
    max_age: int = Field(default=600, ge=0, le=86400)


class WebSocketConfig(BaseModel):
    """WebSocket configuration."""
    enabled: bool = Field(default=True)
    heartbeat_interval_seconds: int = Field(default=30, ge=5, le=120)
    max_connections_per_unit: int = Field(default=100, ge=1, le=1000)
    message_queue_size: int = Field(default=100, ge=10, le=1000)
    reconnect_timeout_seconds: int = Field(default=60, ge=10, le=300)
    compression_enabled: bool = Field(default=True)


class GRPCConfig(BaseModel):
    """gRPC server configuration."""
    enabled: bool = Field(default=True)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=50051, ge=1, le=65535)
    max_workers: int = Field(default=10, ge=1, le=100)
    max_message_size: int = Field(default=4 * 1024 * 1024)
    enable_reflection: bool = Field(default=True)
    enable_health_check: bool = Field(default=True)


class GraphQLConfig(BaseModel):
    """GraphQL configuration."""
    enabled: bool = Field(default=True)
    path: str = Field(default="/graphql")
    enable_playground: bool = Field(default=True)
    enable_subscriptions: bool = Field(default=True)
    max_query_depth: int = Field(default=10, ge=1, le=20)
    max_query_complexity: int = Field(default=100, ge=10, le=500)
    introspection_enabled: bool = Field(default=True)


class AuditConfig(BaseModel):
    """Audit logging configuration."""
    enabled: bool = Field(default=True)
    log_requests: bool = Field(default=True)
    log_responses: bool = Field(default=False)
    retention_days: int = Field(default=90, ge=30, le=365)
    sensitive_fields: List[str] = Field(default=["password", "token", "api_key", "secret"])
    storage_backend: str = Field(default="database")


class APISettings(BaseSettings):
    """Main API settings aggregating all configuration sections."""
    app_name: str = Field(default="GL-004 BURNMASTER API")
    app_version: str = Field(default="1.0.0")
    app_description: str = Field(default="Burner Optimization Agent REST, GraphQL, and gRPC API")
    environment: str = Field(default="development")
    server: ServerConfig = Field(default_factory=ServerConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    cors: CORSConfig = Field(default_factory=CORSConfig)
    websocket: WebSocketConfig = Field(default_factory=WebSocketConfig)
    grpc: GRPCConfig = Field(default_factory=GRPCConfig)
    graphql: GraphQLConfig = Field(default_factory=GraphQLConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    database_url: str = Field(default="postgresql://localhost:5432/burnmaster")
    redis_url: Optional[str] = Field(default=None)
    metrics_endpoint: Optional[str] = Field(default=None)
    tracing_endpoint: Optional[str] = Field(default=None)

    class Config:
        env_prefix = "BURNMASTER_"
        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = False


@lru_cache()
def get_settings() -> APISettings:
    """Get cached API settings."""
    return APISettings()


def get_environment() -> str:
    """Get current environment name."""
    return os.getenv("BURNMASTER_ENVIRONMENT", "development")


def is_production() -> bool:
    """Check if running in production environment."""
    return get_environment() == "production"


def is_development() -> bool:
    """Check if running in development environment."""
    return get_environment() == "development"


settings = get_settings()
