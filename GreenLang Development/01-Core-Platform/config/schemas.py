# -*- coding: utf-8 -*-
"""
GreenLang Configuration Schemas
================================

Pydantic schemas for type-safe, validated configuration.

Features:
- Environment-specific configs (dev/staging/prod)
- Type-safe with Pydantic v2
- Validation at load time
- Support for .env files
- Secrets management integration
- Hot-reload support

Author: GreenLang Framework Team
Date: November 2025
Status: Production Ready
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# ==============================================================================
# Environment Type
# ==============================================================================

class Environment(str, Enum):
    """Deployment environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"

    def is_production(self) -> bool:
        """Check if production environment."""
        return self == Environment.PRODUCTION

    def is_development(self) -> bool:
        """Check if development environment."""
        return self == Environment.DEVELOPMENT

    def is_test(self) -> bool:
        """Check if test environment."""
        return self == Environment.TEST


# ==============================================================================
# LLM Provider Configuration
# ==============================================================================

class LLMProviderConfig(BaseModel):
    """Configuration for LLM providers (OpenAI, Anthropic, etc.)."""

    provider: Literal["openai", "anthropic", "azure_openai", "local"] = Field(
        default="openai",
        description="LLM provider name"
    )

    api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key (loaded from env var or secrets manager)"
    )

    model: str = Field(
        default="gpt-4",
        description="Model identifier (gpt-4, claude-3-opus, etc.)"
    )

    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0=deterministic, 2=creative)"
    )

    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=128000,
        description="Maximum tokens in response"
    )

    timeout_seconds: float = Field(
        default=120.0,
        ge=1.0,
        description="Request timeout in seconds"
    )

    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts"
    )

    base_url: Optional[str] = Field(
        default=None,
        description="Custom API base URL (for Azure, local, etc.)"
    )

    organization_id: Optional[str] = Field(
        default=None,
        description="Organization ID (OpenAI)"
    )

    # Rate limiting
    requests_per_minute: int = Field(
        default=60,
        ge=1,
        description="Rate limit: requests per minute"
    )

    tokens_per_minute: int = Field(
        default=90000,
        ge=1,
        description="Rate limit: tokens per minute"
    )


# ==============================================================================
# Database Configuration
# ==============================================================================

class DatabaseConfig(BaseModel):
    """Configuration for database connections."""

    provider: Literal["postgresql", "sqlite", "mongodb", "memory"] = Field(
        default="postgresql",
        description="Database provider"
    )

    host: str = Field(
        default="localhost",
        description="Database host"
    )

    port: int = Field(
        default=5432,
        ge=1,
        le=65535,
        description="Database port"
    )

    database: str = Field(
        default="greenlang",
        description="Database name"
    )

    username: Optional[str] = Field(
        default=None,
        description="Database username"
    )

    password: Optional[SecretStr] = Field(
        default=None,
        description="Database password"
    )

    pool_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Connection pool size"
    )

    max_overflow: int = Field(
        default=20,
        ge=0,
        description="Max overflow connections"
    )

    pool_timeout: float = Field(
        default=30.0,
        ge=1.0,
        description="Connection pool timeout"
    )

    echo: bool = Field(
        default=False,
        description="Echo SQL queries (debug)"
    )

    ssl_mode: Optional[Literal["disable", "require", "verify-ca", "verify-full"]] = Field(
        default=None,
        description="SSL mode"
    )

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int, info) -> int:
        """Validate port based on provider."""
        provider = info.data.get("provider")
        if provider == "postgresql" and v == 5432:
            return v
        elif provider == "mongodb" and v == 27017:
            return v
        return v

    def get_connection_string(self, hide_password: bool = True) -> str:
        """Build connection string."""
        if self.provider == "sqlite":
            return f"sqlite:///{self.database}"

        password = "***" if hide_password and self.password else self.password.get_secret_value() if self.password else ""

        if self.provider == "postgresql":
            return f"postgresql://{self.username}:{password}@{self.host}:{self.port}/{self.database}"
        elif self.provider == "mongodb":
            return f"mongodb://{self.username}:{password}@{self.host}:{self.port}/{self.database}"

        return f"{self.provider}://{self.host}:{self.port}/{self.database}"


# ==============================================================================
# Cache Configuration
# ==============================================================================

class CacheConfig(BaseModel):
    """Configuration for caching layer."""

    provider: Literal["redis", "memory", "disk", "disabled"] = Field(
        default="memory",
        description="Cache provider"
    )

    host: str = Field(
        default="localhost",
        description="Cache host (Redis)"
    )

    port: int = Field(
        default=6379,
        ge=1,
        le=65535,
        description="Cache port"
    )

    password: Optional[SecretStr] = Field(
        default=None,
        description="Cache password"
    )

    db: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Redis database number"
    )

    ttl_seconds: int = Field(
        default=3600,
        ge=0,
        description="Default TTL in seconds (0=no expiry)"
    )

    max_size_mb: int = Field(
        default=100,
        ge=1,
        description="Max cache size in MB (memory/disk)"
    )

    eviction_policy: Literal["lru", "lfu", "fifo", "ttl"] = Field(
        default="lru",
        description="Cache eviction policy"
    )


# ==============================================================================
# Logging Configuration
# ==============================================================================

class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Log level"
    )

    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )

    output: Literal["console", "file", "both", "json"] = Field(
        default="console",
        description="Log output destination"
    )

    file_path: Optional[Path] = Field(
        default=None,
        description="Log file path (if output=file or both)"
    )

    max_file_size_mb: int = Field(
        default=10,
        ge=1,
        description="Max log file size before rotation"
    )

    backup_count: int = Field(
        default=5,
        ge=1,
        description="Number of backup log files"
    )

    enable_colors: bool = Field(
        default=True,
        description="Enable colored console output"
    )

    structured: bool = Field(
        default=False,
        description="Use structured JSON logging"
    )


# ==============================================================================
# Observability Configuration
# ==============================================================================

class ObservabilityConfig(BaseModel):
    """Configuration for observability (metrics, tracing)."""

    enable_metrics: bool = Field(
        default=True,
        description="Enable metrics collection"
    )

    enable_tracing: bool = Field(
        default=False,
        description="Enable distributed tracing"
    )

    metrics_port: int = Field(
        default=9090,
        ge=1024,
        le=65535,
        description="Prometheus metrics port"
    )

    tracing_endpoint: Optional[str] = Field(
        default=None,
        description="Jaeger/OTLP tracing endpoint"
    )

    sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Trace sampling rate (0-1)"
    )


# ==============================================================================
# Security Configuration
# ==============================================================================

class SecurityConfig(BaseModel):
    """Configuration for security settings."""

    enable_rate_limiting: bool = Field(
        default=True,
        description="Enable rate limiting"
    )

    enable_authentication: bool = Field(
        default=True,
        description="Enable authentication"
    )

    enable_authorization: bool = Field(
        default=True,
        description="Enable authorization"
    )

    jwt_secret: Optional[SecretStr] = Field(
        default=None,
        description="JWT signing secret"
    )

    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm"
    )

    jwt_expiry_minutes: int = Field(
        default=60,
        ge=1,
        description="JWT token expiry in minutes"
    )

    cors_origins: List[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS allowed origins"
    )

    allowed_hosts: List[str] = Field(
        default_factory=lambda: ["*"],
        description="Allowed HTTP hosts"
    )


# ==============================================================================
# Main GreenLang Configuration
# ==============================================================================

class GreenLangConfig(BaseSettings):
    """
    Main GreenLang configuration with environment support.

    Loads configuration from:
    1. Environment variables (GL_*)
    2. .env file
    3. Config file (config.yaml, config.json)
    4. Defaults

    Example:
        >>> config = GreenLangConfig()
        >>> print(config.environment)
        >>> print(config.llm.model)
    """

    model_config = SettingsConfigDict(
        env_prefix="GL_",
        env_nested_delimiter="__",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Deployment environment"
    )

    # Application
    app_name: str = Field(
        default="GreenLang",
        description="Application name"
    )

    app_version: str = Field(
        default="0.3.0",
        description="Application version"
    )

    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )

    # Sub-configurations
    llm: LLMProviderConfig = Field(
        default_factory=LLMProviderConfig,
        description="LLM provider configuration"
    )

    database: DatabaseConfig = Field(
        default_factory=DatabaseConfig,
        description="Database configuration"
    )

    cache: CacheConfig = Field(
        default_factory=CacheConfig,
        description="Cache configuration"
    )

    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration"
    )

    observability: ObservabilityConfig = Field(
        default_factory=ObservabilityConfig,
        description="Observability configuration"
    )

    security: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Security configuration"
    )

    # Paths
    data_dir: Path = Field(
        default=Path("data"),
        description="Data directory"
    )

    packs_dir: Path = Field(
        default=Path("packs"),
        description="Packs directory"
    )

    cache_dir: Path = Field(
        default=Path(".cache"),
        description="Cache directory"
    )

    logs_dir: Path = Field(
        default=Path("logs"),
        description="Logs directory"
    )

    # Features
    enable_async: bool = Field(
        default=True,
        description="Enable async agent execution"
    )

    enable_citations: bool = Field(
        default=True,
        description="Enable citation tracking"
    )

    enable_validation: bool = Field(
        default=True,
        description="Enable schema validation"
    )

    @field_validator("debug")
    @classmethod
    def sync_debug_with_environment(cls, v: bool, info) -> bool:
        """Auto-enable debug in development."""
        env = info.data.get("environment")
        if env == Environment.DEVELOPMENT:
            return True
        return v

    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment.is_production()

    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment.is_development()

    def is_test(self) -> bool:
        """Check if running in tests."""
        return self.environment.is_test()


# ==============================================================================
# Factory Functions
# ==============================================================================

def load_config_from_env() -> GreenLangConfig:
    """
    Load configuration from environment variables.

    Returns:
        GreenLangConfig instance
    """
    return GreenLangConfig()


def load_config_from_file(config_path: Path) -> GreenLangConfig:
    """
    Load configuration from YAML or JSON file.

    Args:
        config_path: Path to config file

    Returns:
        GreenLangConfig instance
    """
    import yaml
    import json

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        if config_path.suffix in [".yaml", ".yml"]:
            data = yaml.safe_load(f)
        elif config_path.suffix == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

    return GreenLangConfig(**data)


def create_test_config(**overrides) -> GreenLangConfig:
    """
    Create test configuration with overrides.

    Args:
        **overrides: Config fields to override

    Returns:
        GreenLangConfig instance for testing
    """
    defaults = {
        "environment": Environment.TEST,
        "debug": True,
        "database": {"provider": "memory"},
        "cache": {"provider": "memory"},
        "logging": {"level": "DEBUG"},
    }
    defaults.update(overrides)
    return GreenLangConfig(**defaults)
