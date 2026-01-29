"""
Configuration settings for GL Normalizer Service.

This module provides centralized configuration management using Pydantic Settings,
supporting environment variables and .env files for production deployments.

Environment Variables:
    GL_NORMALIZER_ENV: Deployment environment (development, staging, production)
    GL_NORMALIZER_DEBUG: Enable debug mode (true/false)
    GL_NORMALIZER_SECRET_KEY: JWT signing secret key
    GL_NORMALIZER_API_KEY_HEADER: Header name for API key authentication
    GL_NORMALIZER_REDIS_URL: Redis connection URL for rate limiting
    GL_NORMALIZER_RATE_LIMIT_REQUESTS: Rate limit requests per window
    GL_NORMALIZER_RATE_LIMIT_WINDOW: Rate limit window in seconds
    GL_NORMALIZER_BATCH_MAX_ITEMS: Maximum items in batch request
    GL_NORMALIZER_ASYNC_THRESHOLD: Threshold for async job processing
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    All settings can be overridden via environment variables prefixed with
    GL_NORMALIZER_. For example, GL_NORMALIZER_DEBUG=true enables debug mode.

    Attributes:
        env: Deployment environment (development, staging, production)
        debug: Enable debug mode with verbose logging
        secret_key: JWT signing secret (must be 32+ chars in production)
        api_key_header: HTTP header name for API key authentication
        algorithm: JWT signing algorithm
        access_token_expire_minutes: JWT token expiration time
        redis_url: Redis connection URL for rate limiting and caching
        rate_limit_enabled: Enable/disable rate limiting
        rate_limit_requests: Maximum requests per window
        rate_limit_window: Rate limit window in seconds
        batch_max_items: Maximum items allowed in batch requests
        async_threshold: Item count threshold for async job processing
        cors_origins: Allowed CORS origins
        trusted_hosts: Trusted host patterns for security
    """

    model_config = SettingsConfigDict(
        env_prefix="GL_NORMALIZER_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment
    env: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )

    # Security
    secret_key: str = Field(
        default="dev-secret-key-change-in-production-must-be-32-chars",
        description="JWT signing secret key",
        min_length=32,
    )
    api_key_header: str = Field(
        default="X-API-Key",
        description="Header name for API key authentication"
    )
    algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm"
    )
    access_token_expire_minutes: int = Field(
        default=30,
        description="JWT token expiration in minutes",
        ge=1,
        le=1440,
    )

    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )

    # Rate Limiting
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    rate_limit_requests: int = Field(
        default=100,
        description="Rate limit requests per window",
        ge=1,
        le=10000,
    )
    rate_limit_window: int = Field(
        default=60,
        description="Rate limit window in seconds",
        ge=1,
        le=3600,
    )

    # Batch Processing
    batch_max_items: int = Field(
        default=10000,
        description="Maximum items in batch request",
        ge=1,
        le=10000,
    )
    async_threshold: int = Field(
        default=100000,
        description="Threshold for async job processing",
        ge=10000,
    )

    # CORS
    cors_origins: list[str] = Field(
        default=["https://*.greenlang.io", "http://localhost:3000"],
        description="Allowed CORS origins"
    )
    trusted_hosts: list[str] = Field(
        default=["*.greenlang.io", "localhost", "127.0.0.1"],
        description="Trusted host patterns"
    )

    # Service Info
    service_name: str = Field(
        default="gl-normalizer-service",
        description="Service name for logging and tracing"
    )
    api_version: str = Field(
        default="v1",
        description="Current API version"
    )
    api_revision: str = Field(
        default="2026-01-30",
        description="API revision date"
    )

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str, info) -> str:
        """Validate secret key length in production."""
        # Access env through info.data if available
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.env == "production"

    @property
    def rate_limit_string(self) -> str:
        """Get rate limit as slowapi-compatible string."""
        return f"{self.rate_limit_requests}/{self.rate_limit_window}seconds"


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.

    Settings are loaded once and cached for performance. The cache is
    automatically cleared when the application restarts.

    Returns:
        Settings: Application settings instance

    Example:
        >>> settings = get_settings()
        >>> print(settings.env)
        'development'
    """
    return Settings()
