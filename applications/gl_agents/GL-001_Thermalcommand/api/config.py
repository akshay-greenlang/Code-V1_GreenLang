"""
GL-001 ThermalCommand API Configuration

Centralized configuration management for the ThermalCommand API.
Supports environment variables and .env files.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # ==========================================================================
    # Application Settings
    # ==========================================================================
    app_name: str = Field(default="GL-001 ThermalCommand API")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)
    environment: str = Field(default="development")

    # ==========================================================================
    # Server Settings
    # ==========================================================================
    host: str = Field(default="0.0.0.0", alias="TC_HOST")
    port: int = Field(default=8000, alias="TC_PORT")
    workers: int = Field(default=1, alias="TC_WORKERS")
    reload: bool = Field(default=False, alias="TC_RELOAD")

    # ==========================================================================
    # gRPC Settings
    # ==========================================================================
    grpc_enabled: bool = Field(default=True, alias="TC_GRPC_ENABLED")
    grpc_host: str = Field(default="0.0.0.0", alias="TC_GRPC_HOST")
    grpc_port: int = Field(default=50051, alias="TC_GRPC_PORT")
    grpc_max_workers: int = Field(default=10, alias="TC_GRPC_MAX_WORKERS")
    grpc_reflection_enabled: bool = Field(default=True, alias="TC_GRPC_REFLECTION")

    # ==========================================================================
    # Authentication Settings
    # ==========================================================================
    jwt_secret_key: str = Field(
        default="development-secret-key-change-in-production-12345",
        alias="TC_JWT_SECRET",
    )
    jwt_algorithm: str = Field(default="HS256", alias="TC_JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(default=30, alias="TC_JWT_ACCESS_EXPIRE")
    jwt_refresh_token_expire_days: int = Field(default=7, alias="TC_JWT_REFRESH_EXPIRE")

    # ==========================================================================
    # mTLS Settings
    # ==========================================================================
    mtls_enabled: bool = Field(default=False, alias="TC_MTLS_ENABLED")
    mtls_ca_cert_path: Optional[str] = Field(default=None, alias="TC_MTLS_CA_CERT")
    mtls_server_cert_path: Optional[str] = Field(default=None, alias="TC_MTLS_SERVER_CERT")
    mtls_server_key_path: Optional[str] = Field(default=None, alias="TC_MTLS_SERVER_KEY")
    mtls_verify_client: bool = Field(default=True, alias="TC_MTLS_VERIFY_CLIENT")

    # ==========================================================================
    # API Key Settings
    # ==========================================================================
    api_key_header: str = Field(default="X-API-Key", alias="TC_API_KEY_HEADER")
    api_key_prefix: str = Field(default="tc_", alias="TC_API_KEY_PREFIX")

    # ==========================================================================
    # CORS Settings
    # ==========================================================================
    cors_origins: str = Field(
        default="https://*.greenlang.io,http://localhost:3000",
        alias="TC_CORS_ORIGINS",
    )
    cors_allow_credentials: bool = Field(default=True, alias="TC_CORS_CREDENTIALS")

    # ==========================================================================
    # Trusted Hosts Settings
    # ==========================================================================
    allowed_hosts: str = Field(
        default="*.greenlang.io,localhost",
        alias="TC_ALLOWED_HOSTS",
    )

    # ==========================================================================
    # Rate Limiting Settings
    # ==========================================================================
    rate_limit_default: str = Field(default="100/minute", alias="TC_RATE_LIMIT_DEFAULT")
    rate_limit_auth: str = Field(default="10/minute", alias="TC_RATE_LIMIT_AUTH")
    rate_limit_allocation: str = Field(default="10/minute", alias="TC_RATE_LIMIT_ALLOCATION")

    # ==========================================================================
    # Database Settings (for production)
    # ==========================================================================
    database_url: Optional[str] = Field(default=None, alias="TC_DATABASE_URL")
    database_pool_size: int = Field(default=5, alias="TC_DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=10, alias="TC_DATABASE_MAX_OVERFLOW")

    # ==========================================================================
    # Redis Settings (for caching)
    # ==========================================================================
    redis_url: Optional[str] = Field(default=None, alias="TC_REDIS_URL")
    redis_cache_ttl: int = Field(default=300, alias="TC_REDIS_CACHE_TTL")

    # ==========================================================================
    # Logging Settings
    # ==========================================================================
    log_level: str = Field(default="INFO", alias="TC_LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        alias="TC_LOG_FORMAT",
    )
    log_json: bool = Field(default=False, alias="TC_LOG_JSON")

    # ==========================================================================
    # Monitoring Settings
    # ==========================================================================
    metrics_enabled: bool = Field(default=True, alias="TC_METRICS_ENABLED")
    tracing_enabled: bool = Field(default=False, alias="TC_TRACING_ENABLED")
    tracing_endpoint: Optional[str] = Field(default=None, alias="TC_TRACING_ENDPOINT")

    # ==========================================================================
    # Optimization Settings
    # ==========================================================================
    optimization_timeout_seconds: int = Field(default=30, alias="TC_OPT_TIMEOUT")
    optimization_max_iterations: int = Field(default=1000, alias="TC_OPT_MAX_ITER")
    optimization_tolerance: float = Field(default=0.001, alias="TC_OPT_TOLERANCE")

    @property
    def cors_origins_list(self) -> List[str]:
        """Get CORS origins as a list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def allowed_hosts_list(self) -> List[str]:
        """Get allowed hosts as a list."""
        return [host.strip() for host in self.allowed_hosts.split(",")]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses LRU cache to avoid re-reading environment variables on every call.

    Returns:
        Settings instance
    """
    return Settings()


# Convenience function for accessing settings
settings = get_settings()


# =============================================================================
# Environment-Specific Configurations
# =============================================================================

class DevelopmentSettings(Settings):
    """Development-specific settings."""
    debug: bool = True
    environment: str = "development"
    reload: bool = True
    log_level: str = "DEBUG"


class ProductionSettings(Settings):
    """Production-specific settings."""
    debug: bool = False
    environment: str = "production"
    reload: bool = False
    workers: int = 4
    log_level: str = "INFO"
    log_json: bool = True
    mtls_enabled: bool = True
    metrics_enabled: bool = True
    tracing_enabled: bool = True


class TestSettings(Settings):
    """Test-specific settings."""
    debug: bool = True
    environment: str = "test"
    database_url: str = "sqlite:///./test.db"
    jwt_secret_key: str = "test-secret-key-for-testing-only-32chars"


def get_settings_for_environment(env: str) -> Settings:
    """
    Get settings for a specific environment.

    Args:
        env: Environment name (development, production, test)

    Returns:
        Environment-specific settings
    """
    settings_map = {
        "development": DevelopmentSettings,
        "production": ProductionSettings,
        "test": TestSettings,
    }
    settings_class = settings_map.get(env, Settings)
    return settings_class()
