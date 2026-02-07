# -*- coding: utf-8 -*-
"""
Streaming PII Scanner Configuration - SEC-011

Configuration models for real-time PII scanning on Kafka and Kinesis streams.
Supports both platforms with separate configuration classes and a unified
streaming configuration that the scanners consume.

Features:
    - Kafka consumer/producer configuration with SASL authentication
    - Kinesis stream configuration with AWS region settings
    - Combined streaming configuration with backend selection
    - Dead letter queue (DLQ) configuration for blocked messages
    - Environment-based defaults (dev, staging, prod)
    - Pydantic validation with sensible production defaults

Configuration Hierarchy:
    StreamingConfig
    ├── KafkaConfig (when backend="kafka")
    ├── KinesisConfig (when backend="kinesis")
    └── Common settings (enforcement_mode, min_confidence, etc.)

Environment Variables:
    PII_KAFKA_* - Kafka-specific settings
    PII_KINESIS_* - Kinesis-specific settings
    PII_STREAMING_* - General streaming settings

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-011 PII Detection/Redaction Enhancements
Status: Production Ready
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StreamingBackend(str, Enum):
    """Supported streaming backends."""

    KAFKA = "kafka"
    KINESIS = "kinesis"


class EnforcementMode(str, Enum):
    """Enforcement action modes for stream processing.

    Determines what happens when PII is detected:
        - ALLOW: Log and pass through unchanged
        - REDACT: Replace PII with redacted values and forward
        - BLOCK: Send to DLQ, do not forward to output
    """

    ALLOW = "allow"
    REDACT = "redact"
    BLOCK = "block"


class KafkaSecurityProtocol(str, Enum):
    """Kafka security protocol options."""

    PLAINTEXT = "PLAINTEXT"
    SSL = "SSL"
    SASL_PLAINTEXT = "SASL_PLAINTEXT"
    SASL_SSL = "SASL_SSL"


class KafkaSASLMechanism(str, Enum):
    """Kafka SASL authentication mechanisms."""

    PLAIN = "PLAIN"
    SCRAM_SHA_256 = "SCRAM-SHA-256"
    SCRAM_SHA_512 = "SCRAM-SHA-512"
    GSSAPI = "GSSAPI"
    OAUTHBEARER = "OAUTHBEARER"


class KinesisShardIteratorType(str, Enum):
    """Kinesis shard iterator types for initial positioning."""

    AT_SEQUENCE_NUMBER = "AT_SEQUENCE_NUMBER"
    AFTER_SEQUENCE_NUMBER = "AFTER_SEQUENCE_NUMBER"
    AT_TIMESTAMP = "AT_TIMESTAMP"
    TRIM_HORIZON = "TRIM_HORIZON"
    LATEST = "LATEST"


# ---------------------------------------------------------------------------
# Kafka Configuration
# ---------------------------------------------------------------------------


class KafkaConfig(BaseSettings):
    """Configuration for Kafka streaming PII scanner.

    Covers both consumer and producer settings for the Kafka PII scanner,
    including authentication, topic routing, and consumer group management.

    Attributes:
        bootstrap_servers: List of Kafka broker addresses.
        consumer_group: Consumer group ID for coordination.
        input_topics: Topics to consume for PII scanning.
        output_topic: Topic for clean (scanned/redacted) messages.
        dlq_topic: Dead letter queue topic for blocked messages.
        auto_offset_reset: Initial offset strategy for new consumers.
        enable_auto_commit: Whether to auto-commit offsets.
        session_timeout_ms: Consumer session timeout.
        max_poll_records: Maximum records per poll.
        max_poll_interval_ms: Maximum interval between polls.
        security_protocol: Security protocol (PLAINTEXT, SSL, SASL_*).
        sasl_mechanism: SASL authentication mechanism.
        sasl_username: SASL username (if using SASL auth).
        sasl_password: SASL password (if using SASL auth).
        ssl_cafile: Path to CA certificate for SSL.
        ssl_certfile: Path to client certificate for mTLS.
        ssl_keyfile: Path to client key for mTLS.
        producer_acks: Acknowledgment level for producers.
        producer_retries: Number of retries for failed sends.
        producer_compression: Compression type for produced messages.
        consumer_fetch_min_bytes: Minimum bytes to fetch per request.
        consumer_fetch_max_wait_ms: Maximum wait time for fetch.

    Example:
        >>> config = KafkaConfig(
        ...     bootstrap_servers=["kafka1:9092", "kafka2:9092"],
        ...     consumer_group="pii-scanner-prod",
        ...     input_topics=["events.raw"],
        ...     output_topic="events.scanned",
        ...     dlq_topic="events.pii-blocked",
        ...     security_protocol=KafkaSecurityProtocol.SASL_SSL,
        ...     sasl_mechanism=KafkaSASLMechanism.SCRAM_SHA_256,
        ... )
    """

    model_config = {
        "env_prefix": "PII_KAFKA_",
        "extra": "ignore",
    }

    # Broker configuration
    bootstrap_servers: List[str] = Field(
        default_factory=lambda: ["localhost:9092"],
        description="Kafka broker addresses (host:port)",
    )

    # Consumer configuration
    consumer_group: str = Field(
        default="pii-scanner",
        min_length=1,
        max_length=255,
        description="Consumer group ID for coordination",
    )
    input_topics: List[str] = Field(
        default_factory=lambda: ["greenlang.events"],
        description="Topics to consume for PII scanning",
    )
    auto_offset_reset: str = Field(
        default="earliest",
        pattern=r"^(earliest|latest|none)$",
        description="Offset reset strategy for new consumers",
    )
    enable_auto_commit: bool = Field(
        default=True,
        description="Auto-commit consumed offsets",
    )
    session_timeout_ms: int = Field(
        default=30000,
        ge=1000,
        le=300000,
        description="Consumer session timeout (ms)",
    )
    max_poll_records: int = Field(
        default=500,
        ge=1,
        le=10000,
        description="Maximum records per poll",
    )
    max_poll_interval_ms: int = Field(
        default=300000,
        ge=10000,
        le=900000,
        description="Maximum interval between polls (ms)",
    )
    consumer_fetch_min_bytes: int = Field(
        default=1,
        ge=1,
        le=1048576,
        description="Minimum bytes per fetch request",
    )
    consumer_fetch_max_wait_ms: int = Field(
        default=500,
        ge=0,
        le=30000,
        description="Maximum wait time for fetch",
    )

    # Producer configuration
    output_topic: str = Field(
        default="greenlang.events.scanned",
        min_length=1,
        max_length=255,
        description="Topic for scanned/redacted messages",
    )
    dlq_topic: str = Field(
        default="greenlang.events.pii-blocked",
        min_length=1,
        max_length=255,
        description="Dead letter queue topic for blocked messages",
    )
    producer_acks: str = Field(
        default="all",
        pattern=r"^(0|1|all|-1)$",
        description="Producer acknowledgment level",
    )
    producer_retries: int = Field(
        default=3,
        ge=0,
        le=100,
        description="Number of retries for failed sends",
    )
    producer_compression: str = Field(
        default="gzip",
        pattern=r"^(none|gzip|snappy|lz4|zstd)$",
        description="Compression type for messages",
    )

    # Security configuration
    security_protocol: KafkaSecurityProtocol = Field(
        default=KafkaSecurityProtocol.PLAINTEXT,
        description="Security protocol for broker connections",
    )
    sasl_mechanism: Optional[KafkaSASLMechanism] = Field(
        default=None,
        description="SASL mechanism for authentication",
    )
    sasl_username: Optional[str] = Field(
        default=None,
        max_length=255,
        description="SASL username",
    )
    sasl_password: Optional[SecretStr] = Field(
        default=None,
        description="SASL password (sensitive)",
    )
    ssl_cafile: Optional[str] = Field(
        default=None,
        description="Path to CA certificate file",
    )
    ssl_certfile: Optional[str] = Field(
        default=None,
        description="Path to client certificate file",
    )
    ssl_keyfile: Optional[str] = Field(
        default=None,
        description="Path to client private key file",
    )

    @field_validator("bootstrap_servers", mode="before")
    @classmethod
    def parse_bootstrap_servers(cls, v: Any) -> List[str]:
        """Parse bootstrap servers from string or list."""
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @field_validator("input_topics", mode="before")
    @classmethod
    def parse_input_topics(cls, v: Any) -> List[str]:
        """Parse input topics from string or list."""
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        return v

    def get_consumer_config(self) -> Dict[str, Any]:
        """Get aiokafka consumer configuration dictionary.

        Returns:
            Dictionary of consumer configuration parameters.
        """
        config: Dict[str, Any] = {
            "bootstrap_servers": ",".join(self.bootstrap_servers),
            "group_id": self.consumer_group,
            "auto_offset_reset": self.auto_offset_reset,
            "enable_auto_commit": self.enable_auto_commit,
            "session_timeout_ms": self.session_timeout_ms,
            "max_poll_records": self.max_poll_records,
            "max_poll_interval_ms": self.max_poll_interval_ms,
            "fetch_min_bytes": self.consumer_fetch_min_bytes,
            "fetch_max_wait_ms": self.consumer_fetch_max_wait_ms,
        }

        # Add security configuration
        config.update(self._get_security_config())

        return config

    def get_producer_config(self) -> Dict[str, Any]:
        """Get aiokafka producer configuration dictionary.

        Returns:
            Dictionary of producer configuration parameters.
        """
        config: Dict[str, Any] = {
            "bootstrap_servers": ",".join(self.bootstrap_servers),
            "acks": self.producer_acks,
            "retries": self.producer_retries,
            "compression_type": self.producer_compression,
        }

        # Add security configuration
        config.update(self._get_security_config())

        return config

    def _get_security_config(self) -> Dict[str, Any]:
        """Get security-related configuration.

        Returns:
            Dictionary of security configuration parameters.
        """
        config: Dict[str, Any] = {
            "security_protocol": self.security_protocol.value,
        }

        if self.sasl_mechanism:
            config["sasl_mechanism"] = self.sasl_mechanism.value
            if self.sasl_username:
                config["sasl_plain_username"] = self.sasl_username
            if self.sasl_password:
                config["sasl_plain_password"] = self.sasl_password.get_secret_value()

        if self.ssl_cafile:
            config["ssl_cafile"] = self.ssl_cafile
        if self.ssl_certfile:
            config["ssl_certfile"] = self.ssl_certfile
        if self.ssl_keyfile:
            config["ssl_keyfile"] = self.ssl_keyfile

        return config


# ---------------------------------------------------------------------------
# Kinesis Configuration
# ---------------------------------------------------------------------------


class KinesisConfig(BaseSettings):
    """Configuration for Kinesis streaming PII scanner.

    Covers AWS Kinesis Data Streams settings for the PII scanner,
    including stream names, region, and checkpoint configuration.

    Attributes:
        stream_name: Name of the input Kinesis stream to consume.
        output_stream_name: Name of the output stream for scanned messages.
        dlq_stream_name: Name of the DLQ stream for blocked messages.
        region: AWS region for Kinesis streams.
        shard_iterator_type: Initial positioning for new consumers.
        checkpoint_interval_seconds: Interval for checkpointing position.
        max_records_per_batch: Maximum records to fetch per GetRecords call.
        idle_time_between_reads_ms: Wait time between read operations.
        use_enhanced_fan_out: Use Enhanced Fan-Out for dedicated throughput.
        consumer_name: Consumer name for Enhanced Fan-Out registration.
        aws_access_key_id: AWS access key (optional, uses default credentials if not set).
        aws_secret_access_key: AWS secret key (optional, uses default credentials if not set).
        aws_session_token: AWS session token for temporary credentials.
        endpoint_url: Custom endpoint URL (for LocalStack testing).

    Example:
        >>> config = KinesisConfig(
        ...     stream_name="greenlang-events-raw",
        ...     output_stream_name="greenlang-events-scanned",
        ...     dlq_stream_name="greenlang-events-pii-blocked",
        ...     region="us-west-2",
        ...     use_enhanced_fan_out=True,
        ...     consumer_name="pii-scanner-prod",
        ... )
    """

    model_config = {
        "env_prefix": "PII_KINESIS_",
        "extra": "ignore",
    }

    # Stream configuration
    stream_name: str = Field(
        default="greenlang-events",
        min_length=1,
        max_length=128,
        description="Input Kinesis stream name",
    )
    output_stream_name: str = Field(
        default="greenlang-events-scanned",
        min_length=1,
        max_length=128,
        description="Output stream for scanned messages",
    )
    dlq_stream_name: str = Field(
        default="greenlang-events-pii-blocked",
        min_length=1,
        max_length=128,
        description="DLQ stream for blocked messages",
    )

    # AWS region configuration
    region: str = Field(
        default="us-east-1",
        min_length=1,
        max_length=30,
        description="AWS region for Kinesis",
    )

    # Consumer configuration
    shard_iterator_type: KinesisShardIteratorType = Field(
        default=KinesisShardIteratorType.LATEST,
        description="Initial shard iterator position",
    )
    checkpoint_interval_seconds: int = Field(
        default=60,
        ge=5,
        le=300,
        description="Checkpoint interval (seconds)",
    )
    max_records_per_batch: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum records per GetRecords call",
    )
    idle_time_between_reads_ms: int = Field(
        default=1000,
        ge=100,
        le=30000,
        description="Wait time between read operations",
    )

    # Enhanced Fan-Out configuration
    use_enhanced_fan_out: bool = Field(
        default=False,
        description="Use Enhanced Fan-Out for dedicated throughput",
    )
    consumer_name: str = Field(
        default="pii-scanner",
        min_length=1,
        max_length=128,
        description="Consumer name for EFO registration",
    )

    # AWS credentials (optional - uses default credential chain if not set)
    aws_access_key_id: Optional[str] = Field(
        default=None,
        description="AWS access key ID (optional)",
    )
    aws_secret_access_key: Optional[SecretStr] = Field(
        default=None,
        description="AWS secret access key (optional)",
    )
    aws_session_token: Optional[str] = Field(
        default=None,
        description="AWS session token for temporary credentials",
    )

    # Custom endpoint (for LocalStack testing)
    endpoint_url: Optional[str] = Field(
        default=None,
        description="Custom endpoint URL (for testing)",
    )

    def get_boto3_config(self) -> Dict[str, Any]:
        """Get boto3 client configuration.

        Returns:
            Dictionary of boto3 client parameters.
        """
        config: Dict[str, Any] = {
            "region_name": self.region,
        }

        if self.endpoint_url:
            config["endpoint_url"] = self.endpoint_url

        if self.aws_access_key_id:
            config["aws_access_key_id"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            config["aws_secret_access_key"] = self.aws_secret_access_key.get_secret_value()
        if self.aws_session_token:
            config["aws_session_token"] = self.aws_session_token

        return config


# ---------------------------------------------------------------------------
# Combined Streaming Configuration
# ---------------------------------------------------------------------------


class StreamingConfig(BaseSettings):
    """Combined streaming configuration for PII scanner.

    This is the main configuration class that ties together Kafka and Kinesis
    settings with common enforcement and processing parameters.

    Attributes:
        enabled: Whether streaming scanner is enabled.
        backend: Which streaming backend to use (kafka/kinesis).
        kafka: Kafka-specific configuration.
        kinesis: Kinesis-specific configuration.
        enforcement_mode: How to handle detected PII.
        min_confidence: Minimum confidence threshold for PII detection.
        batch_timeout_ms: Maximum time to wait for batch completion.
        max_message_size_bytes: Maximum message size to process.
        preserve_message_headers: Whether to preserve original message headers.
        add_scan_metadata: Whether to add scan metadata to output messages.
        metrics_enabled: Whether to emit Prometheus metrics.
        health_check_interval_seconds: Interval for health checks.
        shutdown_timeout_seconds: Timeout for graceful shutdown.

    Example:
        >>> config = StreamingConfig(
        ...     enabled=True,
        ...     backend=StreamingBackend.KAFKA,
        ...     enforcement_mode=EnforcementMode.REDACT,
        ...     min_confidence=0.8,
        ... )
    """

    model_config = {
        "env_prefix": "PII_STREAMING_",
        "extra": "ignore",
    }

    # Enable/disable streaming
    enabled: bool = Field(
        default=True,
        description="Enable streaming PII scanner",
    )

    # Backend selection
    backend: StreamingBackend = Field(
        default=StreamingBackend.KAFKA,
        description="Streaming backend to use",
    )

    # Backend-specific configuration
    kafka: KafkaConfig = Field(
        default_factory=KafkaConfig,
        description="Kafka configuration",
    )
    kinesis: KinesisConfig = Field(
        default_factory=KinesisConfig,
        description="Kinesis configuration",
    )

    # Enforcement configuration
    enforcement_mode: EnforcementMode = Field(
        default=EnforcementMode.REDACT,
        description="Action when PII is detected",
    )
    min_confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for PII detection",
    )

    # Processing configuration
    batch_timeout_ms: int = Field(
        default=5000,
        ge=100,
        le=60000,
        description="Maximum wait for batch completion",
    )
    max_message_size_bytes: int = Field(
        default=1048576,  # 1 MB
        ge=1024,
        le=10485760,  # 10 MB
        description="Maximum message size to process",
    )

    # Message handling
    preserve_message_headers: bool = Field(
        default=True,
        description="Preserve original message headers in output",
    )
    add_scan_metadata: bool = Field(
        default=True,
        description="Add scan metadata to output messages",
    )

    # Operational configuration
    metrics_enabled: bool = Field(
        default=True,
        description="Emit Prometheus metrics",
    )
    health_check_interval_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Health check interval",
    )
    shutdown_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Graceful shutdown timeout",
    )

    @classmethod
    def for_environment(cls, environment: str) -> StreamingConfig:
        """Create configuration tuned for a specific environment.

        Args:
            environment: Environment name (dev/staging/prod).

        Returns:
            StreamingConfig with environment-appropriate settings.
        """
        if environment == "dev":
            return cls(
                enabled=False,
                enforcement_mode=EnforcementMode.ALLOW,
                kafka=KafkaConfig(
                    bootstrap_servers=["localhost:9092"],
                    consumer_group="pii-scanner-dev",
                ),
                kinesis=KinesisConfig(
                    endpoint_url="http://localhost:4566",  # LocalStack
                ),
            )
        elif environment == "staging":
            return cls(
                enabled=True,
                enforcement_mode=EnforcementMode.REDACT,
                min_confidence=0.7,
                kafka=KafkaConfig(
                    consumer_group="pii-scanner-staging",
                    max_poll_records=100,
                ),
            )
        else:  # production
            return cls(
                enabled=True,
                enforcement_mode=EnforcementMode.REDACT,
                min_confidence=0.8,
                kafka=KafkaConfig(
                    consumer_group="pii-scanner-prod",
                    max_poll_records=500,
                    security_protocol=KafkaSecurityProtocol.SASL_SSL,
                    sasl_mechanism=KafkaSASLMechanism.SCRAM_SHA_256,
                    producer_acks="all",
                ),
                kinesis=KinesisConfig(
                    use_enhanced_fan_out=True,
                    max_records_per_batch=100,
                ),
            )


# ---------------------------------------------------------------------------
# Global Configuration Instance
# ---------------------------------------------------------------------------


_global_streaming_config: Optional[StreamingConfig] = None


def get_streaming_config() -> StreamingConfig:
    """Get or create the global streaming configuration.

    Returns:
        The global StreamingConfig instance.
    """
    global _global_streaming_config

    if _global_streaming_config is None:
        _global_streaming_config = StreamingConfig()

    return _global_streaming_config


def configure_streaming(config: StreamingConfig) -> None:
    """Set the global streaming configuration.

    Args:
        config: Configuration to use globally.
    """
    global _global_streaming_config
    _global_streaming_config = config
    logger.info(
        "Streaming PII scanner configured: backend=%s mode=%s enabled=%s",
        config.backend.value,
        config.enforcement_mode.value,
        config.enabled,
    )


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "StreamingBackend",
    "EnforcementMode",
    "KafkaSecurityProtocol",
    "KafkaSASLMechanism",
    "KinesisShardIteratorType",
    # Configuration classes
    "KafkaConfig",
    "KinesisConfig",
    "StreamingConfig",
    # Functions
    "get_streaming_config",
    "configure_streaming",
]
