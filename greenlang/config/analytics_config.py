# -*- coding: utf-8 -*-
"""
Configuration for analytics dashboard and metrics system.

This module provides configuration settings for the WebSocket server,
metric collection, dashboard persistence, and alerting.
"""

import os
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field, validator


class RedisConfig(BaseSettings):
    """Redis configuration."""

    url: str = Field(
        default="redis://localhost:6379",
        env="REDIS_URL",
        description="Redis connection URL"
    )
    max_connections: int = Field(
        default=50,
        env="REDIS_MAX_CONNECTIONS",
        description="Maximum number of Redis connections"
    )
    decode_responses: bool = Field(
        default=False,
        description="Whether to decode Redis responses"
    )

    class Config:
        env_prefix = "REDIS_"


class WebSocketConfig(BaseSettings):
    """WebSocket server configuration."""

    host: str = Field(
        default="0.0.0.0",
        env="WS_HOST",
        description="WebSocket server host"
    )
    port: int = Field(
        default=8000,
        env="WS_PORT",
        description="WebSocket server port"
    )
    path: str = Field(
        default="/ws/metrics",
        env="WS_PATH",
        description="WebSocket endpoint path"
    )
    heartbeat_interval: int = Field(
        default=30,
        env="WS_HEARTBEAT_INTERVAL",
        description="Heartbeat interval in seconds"
    )
    max_message_size: int = Field(
        default=1024 * 1024,  # 1MB
        env="WS_MAX_MESSAGE_SIZE",
        description="Maximum message size in bytes"
    )
    compression_enabled: bool = Field(
        default=True,
        env="WS_COMPRESSION_ENABLED",
        description="Enable MessagePack compression"
    )

    class Config:
        env_prefix = "WS_"


class MetricCollectionConfig(BaseSettings):
    """Metric collection configuration."""

    collection_interval: int = Field(
        default=1,
        env="METRIC_COLLECTION_INTERVAL",
        description="Collection interval in seconds"
    )
    buffer_max_size: int = Field(
        default=100,
        env="METRIC_BUFFER_MAX_SIZE",
        description="Maximum buffer size before flush"
    )
    buffer_max_age: float = Field(
        default=1.0,
        env="METRIC_BUFFER_MAX_AGE",
        description="Maximum buffer age in seconds before flush"
    )

    # Retention policies
    raw_retention_hours: int = Field(
        default=1,
        env="METRIC_RAW_RETENTION_HOURS",
        description="Retention for raw 1s metrics (hours)"
    )
    short_retention_hours: int = Field(
        default=24,
        env="METRIC_SHORT_RETENTION_HOURS",
        description="Retention for 1m aggregated metrics (hours)"
    )
    medium_retention_days: int = Field(
        default=7,
        env="METRIC_MEDIUM_RETENTION_DAYS",
        description="Retention for 5m aggregated metrics (days)"
    )
    long_retention_days: int = Field(
        default=30,
        env="METRIC_LONG_RETENTION_DAYS",
        description="Retention for 1h aggregated metrics (days)"
    )
    archive_retention_days: int = Field(
        default=365,
        env="METRIC_ARCHIVE_RETENTION_DAYS",
        description="Retention for 1d aggregated metrics (days)"
    )

    class Config:
        env_prefix = "METRIC_"


class JWTConfig(BaseSettings):
    """JWT authentication configuration."""

    secret_key: str = Field(
        default="change-me-in-production",
        env="JWT_SECRET_KEY",
        description="JWT secret key for signing tokens"
    )
    algorithm: str = Field(
        default="HS256",
        env="JWT_ALGORITHM",
        description="JWT algorithm"
    )
    access_token_expire_minutes: int = Field(
        default=30,
        env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES",
        description="Access token expiration in minutes"
    )

    @validator("secret_key")
    def validate_secret_key(cls, v):
        """Validate secret key is not default in production."""
        if os.getenv("ENVIRONMENT") == "production" and v == "change-me-in-production":
            raise ValueError("JWT secret key must be changed in production")
        return v

    class Config:
        env_prefix = "JWT_"


class SMTPConfig(BaseSettings):
    """SMTP configuration for email notifications."""

    enabled: bool = Field(
        default=False,
        env="SMTP_ENABLED",
        description="Enable email notifications"
    )
    host: str = Field(
        default="localhost",
        env="SMTP_HOST",
        description="SMTP server host"
    )
    port: int = Field(
        default=587,
        env="SMTP_PORT",
        description="SMTP server port"
    )
    username: Optional[str] = Field(
        default=None,
        env="SMTP_USERNAME",
        description="SMTP username"
    )
    password: Optional[str] = Field(
        default=None,
        env="SMTP_PASSWORD",
        description="SMTP password"
    )
    use_tls: bool = Field(
        default=True,
        env="SMTP_USE_TLS",
        description="Use TLS encryption"
    )
    from_address: str = Field(
        default="alerts@greenlang.com",
        env="SMTP_FROM_ADDRESS",
        description="From email address"
    )

    class Config:
        env_prefix = "SMTP_"


class SlackConfig(BaseSettings):
    """Slack configuration for notifications."""

    enabled: bool = Field(
        default=False,
        env="SLACK_ENABLED",
        description="Enable Slack notifications"
    )
    webhook_url: Optional[str] = Field(
        default=None,
        env="SLACK_WEBHOOK_URL",
        description="Slack webhook URL"
    )

    class Config:
        env_prefix = "SLACK_"


class PagerDutyConfig(BaseSettings):
    """PagerDuty configuration for notifications."""

    enabled: bool = Field(
        default=False,
        env="PAGERDUTY_ENABLED",
        description="Enable PagerDuty notifications"
    )
    integration_key: Optional[str] = Field(
        default=None,
        env="PAGERDUTY_INTEGRATION_KEY",
        description="PagerDuty integration key"
    )

    class Config:
        env_prefix = "PAGERDUTY_"


class RateLimitConfig(BaseSettings):
    """Rate limiting configuration."""

    enabled: bool = Field(
        default=True,
        env="RATE_LIMIT_ENABLED",
        description="Enable rate limiting"
    )
    max_messages_per_minute: int = Field(
        default=1000,
        env="RATE_LIMIT_MAX_MESSAGES_PER_MINUTE",
        description="Maximum messages per minute per client"
    )
    window_seconds: int = Field(
        default=60,
        env="RATE_LIMIT_WINDOW_SECONDS",
        description="Rate limit window in seconds"
    )

    class Config:
        env_prefix = "RATE_LIMIT_"


class AnalyticsConfig(BaseSettings):
    """Main analytics configuration."""

    environment: str = Field(
        default="development",
        env="ENVIRONMENT",
        description="Environment (development, staging, production)"
    )
    debug: bool = Field(
        default=False,
        env="DEBUG",
        description="Enable debug mode"
    )
    log_level: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description="Logging level"
    )

    # Sub-configurations
    redis: RedisConfig = Field(default_factory=RedisConfig)
    websocket: WebSocketConfig = Field(default_factory=WebSocketConfig)
    metrics: MetricCollectionConfig = Field(default_factory=MetricCollectionConfig)
    jwt: JWTConfig = Field(default_factory=JWTConfig)
    smtp: SMTPConfig = Field(default_factory=SMTPConfig)
    slack: SlackConfig = Field(default_factory=SlackConfig)
    pagerduty: PagerDutyConfig = Field(default_factory=PagerDutyConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "environment": self.environment,
            "debug": self.debug,
            "log_level": self.log_level,
            "redis": {
                "url": self.redis.url,
                "max_connections": self.redis.max_connections
            },
            "websocket": {
                "host": self.websocket.host,
                "port": self.websocket.port,
                "path": self.websocket.path,
                "heartbeat_interval": self.websocket.heartbeat_interval,
                "compression_enabled": self.websocket.compression_enabled
            },
            "metrics": {
                "collection_interval": self.metrics.collection_interval,
                "buffer_max_size": self.metrics.buffer_max_size,
                "retention": {
                    "raw_hours": self.metrics.raw_retention_hours,
                    "short_hours": self.metrics.short_retention_hours,
                    "medium_days": self.metrics.medium_retention_days,
                    "long_days": self.metrics.long_retention_days,
                    "archive_days": self.metrics.archive_retention_days
                }
            },
            "notifications": {
                "smtp_enabled": self.smtp.enabled,
                "slack_enabled": self.slack.enabled,
                "pagerduty_enabled": self.pagerduty.enabled
            },
            "rate_limit": {
                "enabled": self.rate_limit.enabled,
                "max_messages_per_minute": self.rate_limit.max_messages_per_minute
            }
        }


# Global configuration instance
_config: Optional[AnalyticsConfig] = None


def get_config() -> AnalyticsConfig:
    """Get global analytics configuration.

    Returns:
        Analytics configuration instance
    """
    global _config
    if _config is None:
        _config = AnalyticsConfig()
    return _config


def reload_config() -> AnalyticsConfig:
    """Reload configuration from environment.

    Returns:
        Reloaded analytics configuration instance
    """
    global _config
    _config = AnalyticsConfig()
    return _config


# Export configuration classes and functions
__all__ = [
    "AnalyticsConfig",
    "RedisConfig",
    "WebSocketConfig",
    "MetricCollectionConfig",
    "JWTConfig",
    "SMTPConfig",
    "SlackConfig",
    "PagerDutyConfig",
    "RateLimitConfig",
    "get_config",
    "reload_config"
]
