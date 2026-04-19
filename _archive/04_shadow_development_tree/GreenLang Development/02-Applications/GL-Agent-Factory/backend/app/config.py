"""
Application Configuration

This module provides configuration management using Pydantic Settings
with environment variable support.

Environment variables can be set directly or via a .env file.

Example:
    >>> from app.config import get_settings
    >>> settings = get_settings()
    >>> print(settings.database_url)
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    All settings can be overridden via environment variables.
    Prefix: GREENLANG_ (e.g., GREENLANG_DATABASE_URL)
    """

    # Application
    app_name: str = Field(default="GreenLang Agent Factory")
    app_env: str = Field(default="development")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # API
    api_prefix: str = Field(default="/v1")
    api_version: str = Field(default="1.0.0")

    # Security
    jwt_secret: str = Field(default="change-me-in-production")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expiration_hours: int = Field(default=24)
    api_key_header: str = Field(default="X-API-Key")

    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"]
    )

    # Database
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/greenlang"
    )
    database_pool_size: int = Field(default=20)
    database_max_overflow: int = Field(default=10)
    database_pool_timeout: int = Field(default=30)

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0")
    redis_max_connections: int = Field(default=10)

    # Rate Limiting
    rate_limit_per_minute: int = Field(default=100)
    rate_limit_burst: int = Field(default=20)

    # Agent Execution
    execution_timeout_default: int = Field(default=300)
    execution_timeout_max: int = Field(default=3600)
    execution_max_concurrent: int = Field(default=100)

    # Storage (S3)
    s3_bucket: str = Field(default="greenlang-artifacts")
    s3_region: str = Field(default="us-east-1")
    s3_endpoint_url: Optional[str] = Field(default=None)

    # Observability
    otlp_endpoint: Optional[str] = Field(default=None)
    metrics_enabled: bool = Field(default=True)
    tracing_enabled: bool = Field(default=True)

    # Multi-tenancy
    default_tenant_quota_agents: int = Field(default=100)
    default_tenant_quota_executions_per_month: int = Field(default=10000)

    class Config:
        """Pydantic settings configuration."""

        env_prefix = "GREENLANG_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.

    Uses LRU cache to avoid re-reading environment on every call.

    Returns:
        Application settings
    """
    return Settings()
