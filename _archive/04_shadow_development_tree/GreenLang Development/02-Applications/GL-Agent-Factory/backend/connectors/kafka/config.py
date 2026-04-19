"""
Kafka Configuration Module for GreenLang Agent Factory.

This module provides configuration models for Kafka connectivity, supporting
both producer and consumer configurations with comprehensive security options.

Features:
- SSL/TLS encryption configuration
- SASL authentication (PLAIN, SCRAM-SHA-256, SCRAM-SHA-512)
- Producer tuning (batching, compression, acks)
- Consumer tuning (auto-commit, offsets, isolation)
- Schema Registry configuration

Usage:
    from connectors.kafka.config import KafkaConfig, KafkaProducerConfig

    config = KafkaConfig(
        bootstrap_servers=["kafka-1:9092", "kafka-2:9092"],
        security_protocol="SASL_SSL",
    )
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class SecurityProtocol(str, Enum):
    """Kafka security protocols."""

    PLAINTEXT = "PLAINTEXT"
    SSL = "SSL"
    SASL_PLAINTEXT = "SASL_PLAINTEXT"
    SASL_SSL = "SASL_SSL"


class SASLMechanism(str, Enum):
    """SASL authentication mechanisms."""

    PLAIN = "PLAIN"
    SCRAM_SHA_256 = "SCRAM-SHA-256"
    SCRAM_SHA_512 = "SCRAM-SHA-512"
    GSSAPI = "GSSAPI"
    OAUTHBEARER = "OAUTHBEARER"


class CompressionType(str, Enum):
    """Message compression types."""

    NONE = "none"
    GZIP = "gzip"
    SNAPPY = "snappy"
    LZ4 = "lz4"
    ZSTD = "zstd"


class AcksMode(str, Enum):
    """Producer acknowledgment modes."""

    NO_ACK = "0"  # Fire and forget
    LEADER_ONLY = "1"  # Leader acknowledgment
    ALL = "all"  # All in-sync replicas (exactly-once)


class AutoOffsetReset(str, Enum):
    """Consumer auto-offset reset policies."""

    EARLIEST = "earliest"
    LATEST = "latest"
    NONE = "none"


class IsolationLevel(str, Enum):
    """Consumer isolation levels for transactional reads."""

    READ_UNCOMMITTED = "read_uncommitted"
    READ_COMMITTED = "read_committed"  # Required for exactly-once


class PartitionStrategy(str, Enum):
    """Partitioning strategies for agent events."""

    ROUND_ROBIN = "round_robin"  # Distribute evenly
    BY_AGENT_ID = "by_agent_id"  # Partition by agent
    BY_TENANT_ID = "by_tenant_id"  # Partition by tenant
    BY_EVENT_TYPE = "by_event_type"  # Partition by event type
    BY_KEY = "by_key"  # Custom key-based partitioning


# =============================================================================
# SSL Configuration
# =============================================================================


class SSLConfig(BaseModel):
    """SSL/TLS configuration for Kafka connections."""

    cafile: Optional[str] = Field(
        None,
        description="Path to CA certificate file for server verification",
    )
    certfile: Optional[str] = Field(
        None,
        description="Path to client certificate file for client authentication",
    )
    keyfile: Optional[str] = Field(
        None,
        description="Path to client private key file",
    )
    password: Optional[str] = Field(
        None,
        description="Password for encrypted private key",
    )
    crlfile: Optional[str] = Field(
        None,
        description="Path to certificate revocation list file",
    )
    check_hostname: bool = Field(
        True,
        description="Verify hostname against certificate",
    )

    def to_aiokafka_config(self) -> Dict[str, Any]:
        """Convert to aiokafka SSL configuration."""
        config = {}

        if self.cafile:
            config["ssl_cafile"] = self.cafile
        if self.certfile:
            config["ssl_certfile"] = self.certfile
        if self.keyfile:
            config["ssl_keyfile"] = self.keyfile
        if self.password:
            config["ssl_password"] = self.password
        if self.crlfile:
            config["ssl_crlfile"] = self.crlfile
        if self.check_hostname is not None:
            config["ssl_check_hostname"] = self.check_hostname

        return config


# =============================================================================
# SASL Configuration
# =============================================================================


class SASLConfig(BaseModel):
    """SASL authentication configuration."""

    mechanism: SASLMechanism = Field(
        SASLMechanism.PLAIN,
        description="SASL authentication mechanism",
    )
    username: Optional[str] = Field(
        None,
        description="SASL username",
    )
    password: Optional[str] = Field(
        None,
        description="SASL password (retrieve from vault in production)",
    )
    kerberos_service_name: str = Field(
        "kafka",
        description="Kerberos service name for GSSAPI",
    )
    kerberos_domain_name: Optional[str] = Field(
        None,
        description="Kerberos domain name",
    )
    oauth_token_provider: Optional[str] = Field(
        None,
        description="OAuth token provider class for OAUTHBEARER",
    )

    def to_aiokafka_config(self) -> Dict[str, Any]:
        """Convert to aiokafka SASL configuration."""
        config = {
            "sasl_mechanism": self.mechanism.value,
        }

        if self.mechanism in (SASLMechanism.PLAIN, SASLMechanism.SCRAM_SHA_256, SASLMechanism.SCRAM_SHA_512):
            config["sasl_plain_username"] = self.username
            config["sasl_plain_password"] = self.password

        if self.mechanism == SASLMechanism.GSSAPI:
            config["sasl_kerberos_service_name"] = self.kerberos_service_name
            if self.kerberos_domain_name:
                config["sasl_kerberos_domain_name"] = self.kerberos_domain_name

        return config


# =============================================================================
# Schema Registry Configuration
# =============================================================================


class SchemaRegistryConfig(BaseModel):
    """Confluent Schema Registry configuration."""

    url: str = Field(
        "http://localhost:8081",
        description="Schema Registry URL",
    )
    basic_auth_user: Optional[str] = Field(
        None,
        description="Basic auth username",
    )
    basic_auth_password: Optional[str] = Field(
        None,
        description="Basic auth password",
    )
    ssl_cafile: Optional[str] = Field(
        None,
        description="CA certificate for Schema Registry SSL",
    )
    auto_register_schemas: bool = Field(
        True,
        description="Automatically register new schemas",
    )
    subject_name_strategy: str = Field(
        "topic_name",
        description="Strategy for schema subject naming",
    )


# =============================================================================
# Producer Configuration
# =============================================================================


class KafkaProducerConfig(BaseModel):
    """Kafka producer-specific configuration."""

    acks: AcksMode = Field(
        AcksMode.ALL,
        description="Required acknowledgments (use 'all' for exactly-once)",
    )
    compression_type: CompressionType = Field(
        CompressionType.LZ4,
        description="Message compression type",
    )
    retries: int = Field(
        5,
        ge=0,
        le=100,
        description="Number of retries for failed sends",
    )
    retry_backoff_ms: int = Field(
        100,
        ge=0,
        description="Backoff time between retries (ms)",
    )
    batch_size: int = Field(
        16384,
        ge=0,
        description="Batch size in bytes",
    )
    linger_ms: int = Field(
        5,
        ge=0,
        description="Time to wait for batch accumulation (ms)",
    )
    buffer_memory: int = Field(
        33554432,
        ge=1048576,
        description="Total producer buffer memory (bytes)",
    )
    max_request_size: int = Field(
        1048576,
        ge=1024,
        description="Maximum request size (bytes)",
    )
    request_timeout_ms: int = Field(
        30000,
        ge=1000,
        description="Request timeout (ms)",
    )
    enable_idempotence: bool = Field(
        True,
        description="Enable idempotent producer (required for exactly-once)",
    )
    transactional_id: Optional[str] = Field(
        None,
        description="Transactional ID for exactly-once semantics",
    )
    max_in_flight_requests_per_connection: int = Field(
        5,
        ge=1,
        le=5,
        description="Max in-flight requests (must be <= 5 for idempotence)",
    )

    @field_validator("max_in_flight_requests_per_connection")
    @classmethod
    def validate_idempotence_constraint(cls, v: int, info) -> int:
        """Validate idempotence constraints."""
        # Note: In Pydantic v2, we can't access other fields easily in field_validator
        # This constraint should be enforced at runtime if idempotence is enabled
        if v > 5:
            raise ValueError(
                "max_in_flight_requests_per_connection must be <= 5 for idempotent producer"
            )
        return v

    def to_aiokafka_config(self) -> Dict[str, Any]:
        """Convert to aiokafka producer configuration."""
        config = {
            "acks": self.acks.value if self.acks != AcksMode.ALL else "all",
            "compression_type": self.compression_type.value,
            "max_batch_size": self.batch_size,
            "linger_ms": self.linger_ms,
            "request_timeout_ms": self.request_timeout_ms,
            "enable_idempotence": self.enable_idempotence,
        }

        if self.transactional_id:
            config["transactional_id"] = self.transactional_id

        return config


# =============================================================================
# Consumer Configuration
# =============================================================================


class KafkaConsumerConfig(BaseModel):
    """Kafka consumer-specific configuration."""

    group_id: str = Field(
        "gl-agent-factory",
        description="Consumer group ID",
    )
    auto_offset_reset: AutoOffsetReset = Field(
        AutoOffsetReset.EARLIEST,
        description="Offset reset policy when no committed offset exists",
    )
    enable_auto_commit: bool = Field(
        False,
        description="Auto-commit offsets (disable for exactly-once)",
    )
    auto_commit_interval_ms: int = Field(
        5000,
        ge=1000,
        description="Auto-commit interval (ms)",
    )
    max_poll_records: int = Field(
        500,
        ge=1,
        description="Maximum records per poll",
    )
    max_poll_interval_ms: int = Field(
        300000,
        ge=1000,
        description="Maximum interval between polls (ms)",
    )
    session_timeout_ms: int = Field(
        45000,
        ge=6000,
        description="Session timeout for consumer group membership (ms)",
    )
    heartbeat_interval_ms: int = Field(
        15000,
        ge=1000,
        description="Heartbeat interval (ms)",
    )
    fetch_min_bytes: int = Field(
        1,
        ge=1,
        description="Minimum bytes to fetch",
    )
    fetch_max_bytes: int = Field(
        52428800,
        ge=1024,
        description="Maximum bytes to fetch",
    )
    fetch_max_wait_ms: int = Field(
        500,
        ge=0,
        description="Maximum wait time for fetch (ms)",
    )
    isolation_level: IsolationLevel = Field(
        IsolationLevel.READ_COMMITTED,
        description="Transaction isolation level (use read_committed for exactly-once)",
    )

    def to_aiokafka_config(self) -> Dict[str, Any]:
        """Convert to aiokafka consumer configuration."""
        return {
            "group_id": self.group_id,
            "auto_offset_reset": self.auto_offset_reset.value,
            "enable_auto_commit": self.enable_auto_commit,
            "auto_commit_interval_ms": self.auto_commit_interval_ms,
            "max_poll_records": self.max_poll_records,
            "max_poll_interval_ms": self.max_poll_interval_ms,
            "session_timeout_ms": self.session_timeout_ms,
            "heartbeat_interval_ms": self.heartbeat_interval_ms,
            "fetch_min_bytes": self.fetch_min_bytes,
            "fetch_max_bytes": self.fetch_max_bytes,
            "fetch_max_wait_ms": self.fetch_max_wait_ms,
            "isolation_level": self.isolation_level.value,
        }


# =============================================================================
# Dead Letter Queue Configuration
# =============================================================================


class DeadLetterQueueConfig(BaseModel):
    """Dead letter queue configuration for failed messages."""

    enabled: bool = Field(
        True,
        description="Enable dead letter queue handling",
    )
    topic_suffix: str = Field(
        ".dlq",
        description="Suffix for DLQ topic names",
    )
    max_retries: int = Field(
        3,
        ge=0,
        le=10,
        description="Maximum retries before moving to DLQ",
    )
    retry_backoff_ms: int = Field(
        1000,
        ge=100,
        description="Initial backoff between retries (ms)",
    )
    retry_backoff_multiplier: float = Field(
        2.0,
        ge=1.0,
        le=10.0,
        description="Exponential backoff multiplier",
    )
    include_headers: bool = Field(
        True,
        description="Include original headers in DLQ message",
    )
    include_error_details: bool = Field(
        True,
        description="Include error details in DLQ headers",
    )


# =============================================================================
# Main Kafka Configuration
# =============================================================================


class KafkaConfig(BaseModel):
    """
    Complete Kafka configuration for GreenLang Agent Factory.

    This configuration supports:
    - Multiple bootstrap servers
    - SSL/TLS encryption
    - SASL authentication
    - Schema Registry integration
    - Producer and consumer tuning
    - Dead letter queue handling
    - Exactly-once semantics

    Example:
        config = KafkaConfig(
            bootstrap_servers=["kafka-1:9092", "kafka-2:9092"],
            security_protocol=SecurityProtocol.SASL_SSL,
            sasl=SASLConfig(
                mechanism=SASLMechanism.SCRAM_SHA_512,
                username="app-user",
                password="<from-vault>",
            ),
            producer=KafkaProducerConfig(
                enable_idempotence=True,
                transactional_id="gl-agent-producer-001",
            ),
        )
    """

    # Connection settings
    bootstrap_servers: List[str] = Field(
        ["localhost:9092"],
        description="Kafka broker addresses",
    )
    client_id: str = Field(
        "gl-agent-factory",
        description="Client identifier",
    )

    # Security
    security_protocol: SecurityProtocol = Field(
        SecurityProtocol.PLAINTEXT,
        description="Security protocol",
    )
    ssl: Optional[SSLConfig] = Field(
        None,
        description="SSL configuration",
    )
    sasl: Optional[SASLConfig] = Field(
        None,
        description="SASL authentication configuration",
    )

    # Schema Registry
    schema_registry: Optional[SchemaRegistryConfig] = Field(
        None,
        description="Schema Registry configuration",
    )

    # Producer settings
    producer: KafkaProducerConfig = Field(
        default_factory=KafkaProducerConfig,
        description="Producer configuration",
    )

    # Consumer settings
    consumer: KafkaConsumerConfig = Field(
        default_factory=KafkaConsumerConfig,
        description="Consumer configuration",
    )

    # Dead letter queue
    dlq: DeadLetterQueueConfig = Field(
        default_factory=DeadLetterQueueConfig,
        description="Dead letter queue configuration",
    )

    # Topic defaults
    default_topic_prefix: str = Field(
        "gl.agent.",
        description="Default prefix for agent topics",
    )
    default_partitions: int = Field(
        6,
        ge=1,
        description="Default partition count for new topics",
    )
    default_replication_factor: int = Field(
        3,
        ge=1,
        description="Default replication factor for new topics",
    )

    # Timeouts
    metadata_max_age_ms: int = Field(
        300000,
        ge=1000,
        description="How often to refresh cluster metadata (ms)",
    )
    connections_max_idle_ms: int = Field(
        540000,
        ge=1000,
        description="Close idle connections after this time (ms)",
    )

    def get_bootstrap_servers_string(self) -> str:
        """Get bootstrap servers as comma-separated string."""
        return ",".join(self.bootstrap_servers)

    def to_aiokafka_producer_config(self) -> Dict[str, Any]:
        """
        Generate aiokafka producer configuration dictionary.

        Returns:
            Configuration dict for AIOKafkaProducer
        """
        config = {
            "bootstrap_servers": self.get_bootstrap_servers_string(),
            "client_id": f"{self.client_id}-producer",
            "security_protocol": self.security_protocol.value,
            "metadata_max_age_ms": self.metadata_max_age_ms,
            "connections_max_idle_ms": self.connections_max_idle_ms,
        }

        # Add producer-specific settings
        config.update(self.producer.to_aiokafka_config())

        # Add SSL configuration
        if self.ssl and self.security_protocol in (SecurityProtocol.SSL, SecurityProtocol.SASL_SSL):
            config.update(self.ssl.to_aiokafka_config())

        # Add SASL configuration
        if self.sasl and self.security_protocol in (SecurityProtocol.SASL_PLAINTEXT, SecurityProtocol.SASL_SSL):
            config.update(self.sasl.to_aiokafka_config())

        return config

    def to_aiokafka_consumer_config(self) -> Dict[str, Any]:
        """
        Generate aiokafka consumer configuration dictionary.

        Returns:
            Configuration dict for AIOKafkaConsumer
        """
        config = {
            "bootstrap_servers": self.get_bootstrap_servers_string(),
            "client_id": f"{self.client_id}-consumer",
            "security_protocol": self.security_protocol.value,
            "metadata_max_age_ms": self.metadata_max_age_ms,
            "connections_max_idle_ms": self.connections_max_idle_ms,
        }

        # Add consumer-specific settings
        config.update(self.consumer.to_aiokafka_config())

        # Add SSL configuration
        if self.ssl and self.security_protocol in (SecurityProtocol.SSL, SecurityProtocol.SASL_SSL):
            config.update(self.ssl.to_aiokafka_config())

        # Add SASL configuration
        if self.sasl and self.security_protocol in (SecurityProtocol.SASL_PLAINTEXT, SecurityProtocol.SASL_SSL):
            config.update(self.sasl.to_aiokafka_config())

        return config


# =============================================================================
# Topic Configuration
# =============================================================================


class TopicConfig(BaseModel):
    """Configuration for a Kafka topic."""

    name: str = Field(
        ...,
        description="Topic name",
    )
    partitions: int = Field(
        6,
        ge=1,
        description="Number of partitions",
    )
    replication_factor: int = Field(
        3,
        ge=1,
        description="Replication factor",
    )
    retention_ms: int = Field(
        604800000,  # 7 days
        ge=0,
        description="Message retention time (ms)",
    )
    cleanup_policy: str = Field(
        "delete",
        description="Cleanup policy: delete or compact",
    )
    min_insync_replicas: int = Field(
        2,
        ge=1,
        description="Minimum in-sync replicas for acks=all",
    )


# =============================================================================
# Standard Topics for GreenLang
# =============================================================================


@dataclass
class GreenLangTopics:
    """Standard Kafka topics for GreenLang Agent Factory."""

    # Agent lifecycle events
    AGENT_EVENTS: str = "gl.agent.events"
    AGENT_CALCULATIONS: str = "gl.agent.calculations"
    AGENT_ALERTS: str = "gl.agent.alerts"
    AGENT_RECOMMENDATIONS: str = "gl.agent.recommendations"
    AGENT_HEALTH: str = "gl.agent.health"
    AGENT_CONFIG: str = "gl.agent.config"

    # Execution events
    EXECUTION_STARTED: str = "gl.execution.started"
    EXECUTION_COMPLETED: str = "gl.execution.completed"
    EXECUTION_FAILED: str = "gl.execution.failed"

    # Audit and compliance
    AUDIT_LOG: str = "gl.audit.log"
    COMPLIANCE_EVENTS: str = "gl.compliance.events"
    PROVENANCE: str = "gl.provenance"

    # Dead letter queues
    AGENT_EVENTS_DLQ: str = "gl.agent.events.dlq"
    EXECUTION_DLQ: str = "gl.execution.dlq"

    @classmethod
    def all_topics(cls) -> List[str]:
        """Get all standard topics."""
        return [
            cls.AGENT_EVENTS,
            cls.AGENT_CALCULATIONS,
            cls.AGENT_ALERTS,
            cls.AGENT_RECOMMENDATIONS,
            cls.AGENT_HEALTH,
            cls.AGENT_CONFIG,
            cls.EXECUTION_STARTED,
            cls.EXECUTION_COMPLETED,
            cls.EXECUTION_FAILED,
            cls.AUDIT_LOG,
            cls.COMPLIANCE_EVENTS,
            cls.PROVENANCE,
            cls.AGENT_EVENTS_DLQ,
            cls.EXECUTION_DLQ,
        ]


# =============================================================================
# Factory Functions
# =============================================================================


def create_production_config(
    bootstrap_servers: List[str],
    sasl_username: str,
    sasl_password: str,
    client_id: str = "gl-agent-factory",
    transactional_id: Optional[str] = None,
) -> KafkaConfig:
    """
    Create a production-ready Kafka configuration.

    Args:
        bootstrap_servers: Kafka broker addresses
        sasl_username: SASL username
        sasl_password: SASL password (from vault)
        client_id: Client identifier
        transactional_id: Optional transactional ID for exactly-once

    Returns:
        Production KafkaConfig
    """
    return KafkaConfig(
        bootstrap_servers=bootstrap_servers,
        client_id=client_id,
        security_protocol=SecurityProtocol.SASL_SSL,
        sasl=SASLConfig(
            mechanism=SASLMechanism.SCRAM_SHA_512,
            username=sasl_username,
            password=sasl_password,
        ),
        ssl=SSLConfig(
            check_hostname=True,
        ),
        producer=KafkaProducerConfig(
            acks=AcksMode.ALL,
            enable_idempotence=True,
            transactional_id=transactional_id,
            compression_type=CompressionType.LZ4,
            retries=5,
        ),
        consumer=KafkaConsumerConfig(
            enable_auto_commit=False,
            isolation_level=IsolationLevel.READ_COMMITTED,
            auto_offset_reset=AutoOffsetReset.EARLIEST,
        ),
        dlq=DeadLetterQueueConfig(
            enabled=True,
            max_retries=3,
        ),
    )


def create_development_config(
    bootstrap_servers: List[str] = None,
) -> KafkaConfig:
    """
    Create a development Kafka configuration.

    Args:
        bootstrap_servers: Kafka broker addresses (defaults to localhost)

    Returns:
        Development KafkaConfig
    """
    return KafkaConfig(
        bootstrap_servers=bootstrap_servers or ["localhost:9092"],
        client_id="gl-agent-factory-dev",
        security_protocol=SecurityProtocol.PLAINTEXT,
        producer=KafkaProducerConfig(
            acks=AcksMode.LEADER_ONLY,
            enable_idempotence=False,
            compression_type=CompressionType.NONE,
        ),
        consumer=KafkaConsumerConfig(
            enable_auto_commit=True,
            isolation_level=IsolationLevel.READ_UNCOMMITTED,
        ),
        dlq=DeadLetterQueueConfig(
            enabled=True,
            max_retries=1,
        ),
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enumerations
    "SecurityProtocol",
    "SASLMechanism",
    "CompressionType",
    "AcksMode",
    "AutoOffsetReset",
    "IsolationLevel",
    "PartitionStrategy",
    # Configuration models
    "SSLConfig",
    "SASLConfig",
    "SchemaRegistryConfig",
    "KafkaProducerConfig",
    "KafkaConsumerConfig",
    "DeadLetterQueueConfig",
    "KafkaConfig",
    "TopicConfig",
    "GreenLangTopics",
    # Factory functions
    "create_production_config",
    "create_development_config",
]
