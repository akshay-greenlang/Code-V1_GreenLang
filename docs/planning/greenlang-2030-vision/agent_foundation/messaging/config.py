# -*- coding: utf-8 -*-
"""
Message Broker Configuration

Centralized configuration for Redis Streams and Kafka brokers.
Supports environment variables, YAML files, and programmatic config.

Example:
    >>> config = MessagingConfig.from_env()
    >>> broker = RedisStreamsBroker(**config.redis_config)
"""

from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, validator
import os
import yaml
from pathlib import Path


class RedisConfig(BaseModel):
    """Redis Streams broker configuration."""

    # Connection
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis port")
    db: int = Field(default=0, ge=0, description="Redis database number")
    password: Optional[str] = Field(None, description="Redis password")
    username: Optional[str] = Field(None, description="Redis username (Redis 6+)")

    # Connection pool
    max_connections: int = Field(default=50, ge=1, description="Max connection pool size")
    socket_timeout: float = Field(default=5.0, ge=0.1, description="Socket timeout seconds")
    socket_connect_timeout: float = Field(default=5.0, ge=0.1, description="Connect timeout")

    # Performance
    decode_responses: bool = Field(default=False, description="Auto-decode responses")
    encoding: str = Field(default="utf-8", description="String encoding")

    # Persistence
    aof_enabled: bool = Field(default=True, description="Append-only file enabled")
    rdb_enabled: bool = Field(default=True, description="RDB snapshots enabled")

    # Streams
    max_stream_length: int = Field(default=100000, ge=1000, description="Max messages per stream")
    consumer_timeout_ms: int = Field(default=5000, ge=100, description="Consumer poll timeout")
    batch_size: int = Field(default=10, ge=1, le=1000, description="Messages per batch")

    # Reliability
    message_ttl_days: int = Field(default=7, ge=1, description="Message retention days")
    dlq_ttl_days: int = Field(default=30, ge=1, description="DLQ retention days")
    max_retries: int = Field(default=3, ge=0, description="Max retry attempts")

    @property
    def url(self) -> str:
        """Get Redis connection URL."""
        auth = ""
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        elif self.password:
            auth = f":{self.password}@"

        return f"redis://{auth}{self.host}:{self.port}/{self.db}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for broker initialization."""
        return {
            "redis_url": self.url,
            "max_connections": self.max_connections,
            "decode_responses": self.decode_responses,
            "encoding": self.encoding,
            "socket_timeout": self.socket_timeout,
            "socket_connect_timeout": self.socket_connect_timeout,
        }


class KafkaConfig(BaseModel):
    """Kafka broker configuration."""

    # Connection
    bootstrap_servers: str = Field(
        default="localhost:9092",
        description="Comma-separated Kafka brokers"
    )
    security_protocol: Literal["PLAINTEXT", "SSL", "SASL_PLAINTEXT", "SASL_SSL"] = Field(
        default="PLAINTEXT",
        description="Security protocol"
    )

    # Authentication
    sasl_mechanism: Optional[str] = Field(None, description="SASL mechanism")
    sasl_username: Optional[str] = Field(None, description="SASL username")
    sasl_password: Optional[str] = Field(None, description="SASL password")

    # Producer
    acks: Literal["0", "1", "all"] = Field(default="all", description="Acknowledgment mode")
    compression_type: Literal["none", "gzip", "snappy", "lz4", "zstd"] = Field(
        default="snappy",
        description="Compression algorithm"
    )
    batch_size: int = Field(default=16384, ge=1, description="Batch size in bytes")
    linger_ms: int = Field(default=10, ge=0, description="Batch linger time ms")

    # Consumer
    group_id: str = Field(default="greenlang_agents", description="Consumer group ID")
    auto_offset_reset: Literal["earliest", "latest"] = Field(
        default="earliest",
        description="Offset reset policy"
    )
    max_poll_records: int = Field(default=500, ge=1, description="Max records per poll")
    session_timeout_ms: int = Field(default=10000, ge=1000, description="Session timeout")

    # Topics
    num_partitions: int = Field(default=10, ge=1, description="Default partitions per topic")
    replication_factor: int = Field(default=3, ge=1, description="Replication factor")
    retention_hours: int = Field(default=168, ge=1, description="Message retention hours (7 days)")

    # Performance
    max_in_flight_requests: int = Field(default=5, ge=1, description="Max in-flight requests")
    buffer_memory: int = Field(default=33554432, ge=1, description="Producer buffer memory bytes")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Kafka client."""
        config = {
            "bootstrap_servers": self.bootstrap_servers.split(","),
            "security_protocol": self.security_protocol,
            "compression_type": self.compression_type,
        }

        if self.sasl_mechanism:
            config.update({
                "sasl_mechanism": self.sasl_mechanism,
                "sasl_plain_username": self.sasl_username,
                "sasl_plain_password": self.sasl_password,
            })

        return config


class MessagingConfig(BaseModel):
    """
    Complete messaging system configuration.

    Supports both Redis Streams and Kafka brokers.
    """

    # Broker selection
    broker_type: Literal["redis", "kafka"] = Field(
        default="redis",
        description="Message broker type"
    )

    # Redis configuration
    redis: RedisConfig = Field(
        default_factory=RedisConfig,
        description="Redis Streams config"
    )

    # Kafka configuration
    kafka: Optional[KafkaConfig] = Field(
        None,
        description="Kafka config (optional)"
    )

    # Monitoring
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    prometheus_port: int = Field(default=9090, ge=1024, le=65535, description="Prometheus port")
    health_check_interval_seconds: int = Field(
        default=30,
        ge=1,
        description="Health check interval"
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )
    log_messages: bool = Field(default=False, description="Log all messages (verbose)")

    # Reliability
    circuit_breaker_enabled: bool = Field(
        default=True,
        description="Enable circuit breaker"
    )
    circuit_breaker_threshold: int = Field(
        default=5,
        ge=1,
        description="Failures before opening circuit"
    )
    circuit_breaker_timeout_seconds: int = Field(
        default=60,
        ge=1,
        description="Circuit open timeout"
    )

    @classmethod
    def from_yaml(cls, file_path: str) -> "MessagingConfig":
        """
        Load configuration from YAML file.

        Args:
            file_path: Path to YAML configuration file

        Returns:
            MessagingConfig instance

        Example:
            >>> config = MessagingConfig.from_yaml("config/messaging.yaml")
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    @classmethod
    def from_env(cls) -> "MessagingConfig":
        """
        Load configuration from environment variables.

        Environment Variables:
            GREENLANG_BROKER_TYPE: redis or kafka
            GREENLANG_REDIS_HOST: Redis host
            GREENLANG_REDIS_PORT: Redis port
            GREENLANG_REDIS_PASSWORD: Redis password
            GREENLANG_KAFKA_BOOTSTRAP_SERVERS: Kafka brokers
            GREENLANG_LOG_LEVEL: Logging level

        Returns:
            MessagingConfig instance

        Example:
            >>> config = MessagingConfig.from_env()
        """
        redis_config = RedisConfig(
            host=os.getenv("GREENLANG_REDIS_HOST", "localhost"),
            port=int(os.getenv("GREENLANG_REDIS_PORT", "6379")),
            password=os.getenv("GREENLANG_REDIS_PASSWORD"),
            max_connections=int(os.getenv("GREENLANG_REDIS_MAX_CONNECTIONS", "50")),
        )

        kafka_config = None
        if os.getenv("GREENLANG_KAFKA_BOOTSTRAP_SERVERS"):
            kafka_config = KafkaConfig(
                bootstrap_servers=os.getenv("GREENLANG_KAFKA_BOOTSTRAP_SERVERS"),
                group_id=os.getenv("GREENLANG_KAFKA_GROUP_ID", "greenlang_agents"),
            )

        return cls(
            broker_type=os.getenv("GREENLANG_BROKER_TYPE", "redis"),
            redis=redis_config,
            kafka=kafka_config,
            log_level=os.getenv("GREENLANG_LOG_LEVEL", "INFO"),
            metrics_enabled=os.getenv("GREENLANG_METRICS_ENABLED", "true").lower() == "true",
        )

    def to_yaml(self, file_path: str) -> None:
        """
        Save configuration to YAML file.

        Args:
            file_path: Path to save YAML file
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    def get_broker_config(self) -> Dict[str, Any]:
        """Get broker-specific configuration."""
        if self.broker_type == "redis":
            return self.redis.to_dict()
        elif self.broker_type == "kafka":
            if self.kafka is None:
                raise ValueError("Kafka config not provided")
            return self.kafka.to_dict()
        else:
            raise ValueError(f"Unknown broker type: {self.broker_type}")


# Default configuration
DEFAULT_CONFIG = MessagingConfig()


def load_config(
    config_path: Optional[str] = None,
    use_env: bool = True,
) -> MessagingConfig:
    """
    Load messaging configuration.

    Priority:
        1. config_path (if provided)
        2. Environment variables (if use_env=True)
        3. Default configuration

    Args:
        config_path: Optional YAML config file path
        use_env: Whether to load from environment

    Returns:
        MessagingConfig instance
    """
    if config_path:
        return MessagingConfig.from_yaml(config_path)
    elif use_env:
        return MessagingConfig.from_env()
    else:
        return DEFAULT_CONFIG
