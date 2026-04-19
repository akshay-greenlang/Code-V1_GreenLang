"""
GL-003 UnifiedSteam API Configuration

Centralized configuration management for the SteamSystemOptimizer API.
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
    app_name: str = Field(default="GL-003 UnifiedSteam SteamSystemOptimizer API")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)
    environment: str = Field(default="development")

    # ==========================================================================
    # Server Settings
    # ==========================================================================
    host: str = Field(default="0.0.0.0", alias="STEAM_HOST")
    port: int = Field(default=8000, alias="STEAM_PORT")
    workers: int = Field(default=1, alias="STEAM_WORKERS")
    reload: bool = Field(default=False, alias="STEAM_RELOAD")

    # ==========================================================================
    # gRPC Settings
    # ==========================================================================
    grpc_enabled: bool = Field(default=True, alias="STEAM_GRPC_ENABLED")
    grpc_host: str = Field(default="0.0.0.0", alias="STEAM_GRPC_HOST")
    grpc_port: int = Field(default=50052, alias="STEAM_GRPC_PORT")
    grpc_max_workers: int = Field(default=10, alias="STEAM_GRPC_MAX_WORKERS")
    grpc_reflection_enabled: bool = Field(default=True, alias="STEAM_GRPC_REFLECTION")

    # ==========================================================================
    # Authentication Settings
    # ==========================================================================
    jwt_secret_key: str = Field(
        default="development-secret-key-change-in-production-12345",
        alias="STEAM_JWT_SECRET",
    )
    jwt_algorithm: str = Field(default="HS256", alias="STEAM_JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(default=30, alias="STEAM_JWT_ACCESS_EXPIRE")
    jwt_refresh_token_expire_days: int = Field(default=7, alias="STEAM_JWT_REFRESH_EXPIRE")

    # ==========================================================================
    # mTLS Settings
    # ==========================================================================
    mtls_enabled: bool = Field(default=False, alias="STEAM_MTLS_ENABLED")
    mtls_ca_cert_path: Optional[str] = Field(default=None, alias="STEAM_MTLS_CA_CERT")
    mtls_server_cert_path: Optional[str] = Field(default=None, alias="STEAM_MTLS_SERVER_CERT")
    mtls_server_key_path: Optional[str] = Field(default=None, alias="STEAM_MTLS_SERVER_KEY")
    mtls_verify_client: bool = Field(default=True, alias="STEAM_MTLS_VERIFY_CLIENT")

    # ==========================================================================
    # API Key Settings
    # ==========================================================================
    api_key_header: str = Field(default="X-API-Key", alias="STEAM_API_KEY_HEADER")
    api_key_prefix: str = Field(default="steam_", alias="STEAM_API_KEY_PREFIX")

    # ==========================================================================
    # CORS Settings
    # ==========================================================================
    cors_origins: str = Field(
        default="https://*.greenlang.io,http://localhost:3000",
        alias="STEAM_CORS_ORIGINS",
    )
    cors_allow_credentials: bool = Field(default=True, alias="STEAM_CORS_CREDENTIALS")

    # ==========================================================================
    # Trusted Hosts Settings
    # ==========================================================================
    allowed_hosts: str = Field(
        default="*.greenlang.io,localhost",
        alias="STEAM_ALLOWED_HOSTS",
    )

    # ==========================================================================
    # Rate Limiting Settings
    # ==========================================================================
    rate_limit_default: str = Field(default="100/minute", alias="STEAM_RATE_LIMIT_DEFAULT")
    rate_limit_auth: str = Field(default="10/minute", alias="STEAM_RATE_LIMIT_AUTH")
    rate_limit_optimization: str = Field(default="20/minute", alias="STEAM_RATE_LIMIT_OPT")
    rate_limit_properties: str = Field(default="500/minute", alias="STEAM_RATE_LIMIT_PROPS")
    rate_limit_diagnostics: str = Field(default="100/minute", alias="STEAM_RATE_LIMIT_DIAG")

    # ==========================================================================
    # Database Settings (for production)
    # ==========================================================================
    database_url: Optional[str] = Field(default=None, alias="STEAM_DATABASE_URL")
    database_pool_size: int = Field(default=5, alias="STEAM_DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=10, alias="STEAM_DATABASE_MAX_OVERFLOW")

    # ==========================================================================
    # Redis Settings (for caching)
    # ==========================================================================
    redis_url: Optional[str] = Field(default=None, alias="STEAM_REDIS_URL")
    redis_cache_ttl: int = Field(default=300, alias="STEAM_REDIS_CACHE_TTL")

    # ==========================================================================
    # Logging Settings
    # ==========================================================================
    log_level: str = Field(default="INFO", alias="STEAM_LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        alias="STEAM_LOG_FORMAT",
    )
    log_json: bool = Field(default=False, alias="STEAM_LOG_JSON")

    # ==========================================================================
    # Monitoring Settings
    # ==========================================================================
    metrics_enabled: bool = Field(default=True, alias="STEAM_METRICS_ENABLED")
    tracing_enabled: bool = Field(default=False, alias="STEAM_TRACING_ENABLED")
    tracing_endpoint: Optional[str] = Field(default=None, alias="STEAM_TRACING_ENDPOINT")

    # ==========================================================================
    # Steam Calculation Settings
    # ==========================================================================
    steam_calc_timeout_seconds: int = Field(default=10, alias="STEAM_CALC_TIMEOUT")
    steam_default_reference_pressure_kpa: float = Field(default=101.325, alias="STEAM_REF_PRESSURE")
    steam_default_reference_temp_c: float = Field(default=25.0, alias="STEAM_REF_TEMP")

    # ==========================================================================
    # Optimization Settings
    # ==========================================================================
    optimization_timeout_seconds: int = Field(default=60, alias="STEAM_OPT_TIMEOUT")
    optimization_max_iterations: int = Field(default=1000, alias="STEAM_OPT_MAX_ITER")
    optimization_tolerance: float = Field(default=0.001, alias="STEAM_OPT_TOLERANCE")
    optimization_default_horizon_hours: int = Field(default=24, alias="STEAM_OPT_HORIZON")

    # ==========================================================================
    # Trap Diagnostics Settings
    # ==========================================================================
    trap_diagnostic_timeout_seconds: int = Field(default=30, alias="STEAM_TRAP_TIMEOUT")
    trap_failure_threshold: float = Field(default=0.7, alias="STEAM_TRAP_FAIL_THRESH")
    trap_batch_max_size: int = Field(default=1000, alias="STEAM_TRAP_BATCH_MAX")

    # ==========================================================================
    # RCA Settings
    # ==========================================================================
    rca_timeout_seconds: int = Field(default=120, alias="STEAM_RCA_TIMEOUT")
    rca_max_causal_factors: int = Field(default=10, alias="STEAM_RCA_MAX_FACTORS")
    rca_min_confidence: float = Field(default=0.5, alias="STEAM_RCA_MIN_CONF")

    # ==========================================================================
    # Explainability Settings
    # ==========================================================================
    explainability_shap_enabled: bool = Field(default=True, alias="STEAM_SHAP_ENABLED")
    explainability_lime_enabled: bool = Field(default=True, alias="STEAM_LIME_ENABLED")
    explainability_max_features: int = Field(default=20, alias="STEAM_EXPLAIN_MAX_FEAT")

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
