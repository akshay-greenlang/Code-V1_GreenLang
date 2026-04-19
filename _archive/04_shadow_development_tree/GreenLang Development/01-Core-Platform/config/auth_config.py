"""
Authentication Configuration Module for GreenLang

This module provides configuration options for the GreenLang authentication system,
including database connection settings, permission storage backends, and security
policies.

Author: GreenLang Framework Team
Date: November 2025
"""

import os
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class StorageBackend(str, Enum):
    """Supported storage backends for authentication data."""
    MEMORY = "memory"
    POSTGRESQL = "postgresql"
    REDIS = "redis"  # Future support
    MONGODB = "mongodb"  # Future support


class DatabaseConfig(BaseModel):
    """PostgreSQL database configuration."""

    host: str = Field(
        default_factory=lambda: os.getenv("GREENLANG_DB_HOST", "localhost"),
        description="Database host"
    )
    port: int = Field(
        default_factory=lambda: int(os.getenv("GREENLANG_DB_PORT", "5432")),
        description="Database port"
    )
    database: str = Field(
        default_factory=lambda: os.getenv("GREENLANG_DB_NAME", "greenlang"),
        description="Database name"
    )
    username: str = Field(
        default_factory=lambda: os.getenv("GREENLANG_DB_USER", "greenlang"),
        description="Database username"
    )
    password: str = Field(
        default_factory=lambda: os.getenv("GREENLANG_DB_PASSWORD", ""),
        description="Database password"
    )
    pool_size: int = Field(
        default_factory=lambda: int(os.getenv("GREENLANG_DB_POOL_SIZE", "20")),
        description="Connection pool size"
    )
    max_overflow: int = Field(
        default_factory=lambda: int(os.getenv("GREENLANG_DB_MAX_OVERFLOW", "10")),
        description="Maximum overflow connections"
    )
    pool_timeout: int = Field(
        default_factory=lambda: int(os.getenv("GREENLANG_DB_POOL_TIMEOUT", "30")),
        description="Pool timeout in seconds"
    )
    echo: bool = Field(
        default_factory=lambda: os.getenv("GREENLANG_DB_ECHO", "false").lower() == "true",
        description="Echo SQL statements"
    )
    ssl_mode: Optional[str] = Field(
        default_factory=lambda: os.getenv("GREENLANG_DB_SSL_MODE"),
        description="SSL mode (require, prefer, disable)"
    )

    @validator('password')
    def validate_password(cls, v):
        """Ensure password is provided for production."""
        if not v and os.getenv("GREENLANG_ENV", "development") == "production":
            raise ValueError("Database password is required in production")
        return v

    @property
    def connection_url(self) -> str:
        """Generate database connection URL."""
        base_url = f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        if self.ssl_mode:
            base_url += f"?sslmode={self.ssl_mode}"
        return base_url

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backend initialization."""
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "username": self.username,
            "password": self.password,
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "echo": self.echo,
            "ssl_mode": self.ssl_mode
        }


class AuthConfig(BaseModel):
    """Main authentication configuration."""

    # Storage backend configuration
    storage_backend: StorageBackend = Field(
        default_factory=lambda: StorageBackend(os.getenv("GREENLANG_AUTH_BACKEND", "memory")),
        description="Storage backend for authentication data"
    )

    # Database configuration (for PostgreSQL backend)
    database: Optional[DatabaseConfig] = Field(
        default=None,
        description="Database configuration for PostgreSQL backend"
    )

    # Security settings
    session_timeout_minutes: int = Field(
        default_factory=lambda: int(os.getenv("GREENLANG_SESSION_TIMEOUT", "60")),
        description="Session timeout in minutes"
    )
    max_login_attempts: int = Field(
        default_factory=lambda: int(os.getenv("GREENLANG_MAX_LOGIN_ATTEMPTS", "5")),
        description="Maximum login attempts before lockout"
    )
    lockout_duration_minutes: int = Field(
        default_factory=lambda: int(os.getenv("GREENLANG_LOCKOUT_DURATION", "30")),
        description="Account lockout duration in minutes"
    )
    password_min_length: int = Field(
        default_factory=lambda: int(os.getenv("GREENLANG_PASSWORD_MIN_LENGTH", "12")),
        description="Minimum password length"
    )
    require_mfa: bool = Field(
        default_factory=lambda: os.getenv("GREENLANG_REQUIRE_MFA", "false").lower() == "true",
        description="Require multi-factor authentication"
    )

    # JWT settings
    jwt_secret_key: str = Field(
        default_factory=lambda: os.getenv("GREENLANG_JWT_SECRET", ""),
        description="Secret key for JWT signing"
    )
    jwt_algorithm: str = Field(
        default_factory=lambda: os.getenv("GREENLANG_JWT_ALGORITHM", "HS256"),
        description="JWT signing algorithm"
    )
    jwt_expiration_hours: int = Field(
        default_factory=lambda: int(os.getenv("GREENLANG_JWT_EXPIRATION", "24")),
        description="JWT token expiration in hours"
    )

    # Audit settings
    enable_audit_logging: bool = Field(
        default_factory=lambda: os.getenv("GREENLANG_AUDIT_LOGGING", "true").lower() == "true",
        description="Enable audit logging"
    )
    audit_log_retention_days: int = Field(
        default_factory=lambda: int(os.getenv("GREENLANG_AUDIT_RETENTION", "90")),
        description="Audit log retention in days"
    )

    # Permission evaluation settings
    permission_cache_ttl_seconds: int = Field(
        default_factory=lambda: int(os.getenv("GREENLANG_PERMISSION_CACHE_TTL", "300")),
        description="Permission cache TTL in seconds"
    )
    enable_permission_caching: bool = Field(
        default_factory=lambda: os.getenv("GREENLANG_PERMISSION_CACHING", "true").lower() == "true",
        description="Enable permission caching"
    )

    @validator('database', always=True)
    def validate_database_config(cls, v, values):
        """Ensure database config is provided for PostgreSQL backend."""
        if 'storage_backend' in values and values['storage_backend'] == StorageBackend.POSTGRESQL:
            if v is None:
                # Auto-create database config from environment
                v = DatabaseConfig()
        return v

    @validator('jwt_secret_key')
    def validate_jwt_secret(cls, v):
        """Ensure JWT secret is provided."""
        if not v:
            if os.getenv("GREENLANG_ENV", "development") == "production":
                raise ValueError("JWT secret key is required in production")
            else:
                # Generate a default secret for development
                import secrets
                v = secrets.token_urlsafe(32)
        return v

    def get_permission_store_config(self) -> Dict[str, Any]:
        """Get configuration for PermissionStore initialization."""
        config = {
            "storage_backend": self.storage_backend.value
        }
        if self.storage_backend == StorageBackend.POSTGRESQL and self.database:
            config["db_config"] = self.database.to_dict()
        return config

    @classmethod
    def from_env(cls) -> 'AuthConfig':
        """Create configuration from environment variables."""
        config = cls()

        # Auto-configure database if PostgreSQL backend is selected
        if config.storage_backend == StorageBackend.POSTGRESQL and not config.database:
            config.database = DatabaseConfig()

        return config

    @classmethod
    def for_testing(cls) -> 'AuthConfig':
        """Create configuration for testing with in-memory backend."""
        return cls(
            storage_backend=StorageBackend.MEMORY,
            jwt_secret_key="test-secret-key",
            enable_audit_logging=False,
            enable_permission_caching=False
        )

    @classmethod
    def for_production(cls, db_password: str, jwt_secret: str) -> 'AuthConfig':
        """
        Create production configuration with PostgreSQL backend.

        Args:
            db_password: Database password
            jwt_secret: JWT signing secret

        Returns:
            Production-ready configuration
        """
        return cls(
            storage_backend=StorageBackend.POSTGRESQL,
            database=DatabaseConfig(
                password=db_password,
                ssl_mode="require",
                pool_size=50,
                max_overflow=20
            ),
            jwt_secret_key=jwt_secret,
            require_mfa=True,
            enable_audit_logging=True,
            enable_permission_caching=True,
            session_timeout_minutes=30,
            max_login_attempts=3,
            lockout_duration_minutes=60
        )


# ==============================================================================
# Configuration Loader
# ==============================================================================

_auth_config: Optional[AuthConfig] = None


def load_auth_config(config_path: Optional[str] = None) -> AuthConfig:
    """
    Load authentication configuration.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Loaded configuration
    """
    global _auth_config

    if _auth_config is None:
        if config_path:
            # Load from file
            import json
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            _auth_config = AuthConfig(**config_data)
        else:
            # Load from environment
            _auth_config = AuthConfig.from_env()

    return _auth_config


def get_auth_config() -> AuthConfig:
    """
    Get current authentication configuration.

    Returns:
        Current configuration

    Raises:
        RuntimeError: If configuration not loaded
    """
    if _auth_config is None:
        raise RuntimeError("Authentication configuration not loaded. Call load_auth_config() first.")
    return _auth_config


def init_permission_store():
    """
    Initialize PermissionStore with current configuration.

    Returns:
        Configured PermissionStore instance
    """
    from greenlang.auth.permissions import PermissionStore

    config = get_auth_config()
    store_config = config.get_permission_store_config()

    return PermissionStore(**store_config)


__all__ = [
    'StorageBackend',
    'DatabaseConfig',
    'AuthConfig',
    'load_auth_config',
    'get_auth_config',
    'init_permission_store'
]