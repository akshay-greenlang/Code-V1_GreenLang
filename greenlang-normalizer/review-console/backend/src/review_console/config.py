"""
Configuration settings for Review Console Backend.

This module provides centralized configuration management using Pydantic Settings,
supporting environment variables and .env files for production deployments.

Environment Variables:
    REVIEW_CONSOLE_ENV: Deployment environment (development, staging, production)
    REVIEW_CONSOLE_DEBUG: Enable debug mode (true/false)
    REVIEW_CONSOLE_SECRET_KEY: JWT signing secret key
    REVIEW_CONSOLE_DATABASE_URL: PostgreSQL connection URL
    REVIEW_CONSOLE_REDIS_URL: Redis connection URL for rate limiting
    REVIEW_CONSOLE_RATE_LIMIT_REQUESTS: Rate limit requests per window
    REVIEW_CONSOLE_RATE_LIMIT_WINDOW: Rate limit window in seconds
    REVIEW_CONSOLE_VOCABULARY_SERVICE_URL: URL for vocabulary service (PR creation)
    REVIEW_CONSOLE_GITHUB_TOKEN: GitHub token for PR creation

Example:
    >>> from review_console.config import get_settings
    >>> settings = get_settings()
    >>> print(settings.env)
    'development'
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    All settings can be overridden via environment variables prefixed with
    REVIEW_CONSOLE_. For example, REVIEW_CONSOLE_DEBUG=true enables debug mode.

    Attributes:
        env: Deployment environment (development, staging, production)
        debug: Enable debug mode with verbose logging
        secret_key: JWT signing secret (must be 32+ chars in production)
        database_url: PostgreSQL connection URL
        redis_url: Redis connection URL for rate limiting and caching
        rate_limit_enabled: Enable/disable rate limiting
        rate_limit_requests: Maximum requests per window
        rate_limit_window: Rate limit window in seconds
        vocabulary_service_url: URL for vocabulary governance service
        github_token: GitHub token for vocabulary PR creation
        cors_origins: Allowed CORS origins
        trusted_hosts: Trusted host patterns for security
        default_page_size: Default pagination page size
        max_page_size: Maximum pagination page size
        confidence_threshold_needs_review: Confidence threshold below which items need review
    """

    model_config = SettingsConfigDict(
        env_prefix="REVIEW_CONSOLE_",
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
    algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm"
    )
    access_token_expire_minutes: int = Field(
        default=60,
        description="JWT token expiration in minutes",
        ge=1,
        le=1440,
    )

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/review_console",
        description="PostgreSQL connection URL"
    )
    database_pool_size: int = Field(
        default=5,
        description="Database connection pool size",
        ge=1,
        le=50,
    )
    database_max_overflow: int = Field(
        default=10,
        description="Database max overflow connections",
        ge=0,
        le=100,
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

    # External Services
    vocabulary_service_url: str = Field(
        default="http://localhost:8001",
        description="URL for vocabulary governance service"
    )
    github_token: str = Field(
        default="",
        description="GitHub token for vocabulary PR creation"
    )
    github_repo: str = Field(
        default="greenlang/greenlang-vocabularies",
        description="GitHub repository for vocabulary PRs"
    )

    # CORS
    cors_origins: list[str] = Field(
        default=["https://*.greenlang.io", "http://localhost:3000", "http://localhost:5173"],
        description="Allowed CORS origins"
    )
    trusted_hosts: list[str] = Field(
        default=["*.greenlang.io", "localhost", "127.0.0.1"],
        description="Trusted host patterns"
    )

    # Pagination
    default_page_size: int = Field(
        default=20,
        description="Default pagination page size",
        ge=1,
        le=100,
    )
    max_page_size: int = Field(
        default=100,
        description="Maximum pagination page size",
        ge=1,
        le=500,
    )

    # Review Queue Settings
    confidence_threshold_needs_review: float = Field(
        default=0.85,
        description="Confidence threshold below which items need review",
        ge=0.0,
        le=1.0,
    )
    auto_close_high_confidence: bool = Field(
        default=True,
        description="Automatically close items above high confidence threshold"
    )
    high_confidence_threshold: float = Field(
        default=0.95,
        description="Threshold for auto-closing high confidence matches",
        ge=0.0,
        le=1.0,
    )

    # Service Info
    service_name: str = Field(
        default="review-console",
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
    def validate_secret_key(cls, v: str) -> str:
        """Validate secret key length."""
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters")
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
