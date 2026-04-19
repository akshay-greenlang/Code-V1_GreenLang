"""
Kafka Streaming Infrastructure for GreenLang Agent Factory.

This package provides enterprise-grade Kafka connectivity for publishing
and consuming agent events with support for exactly-once semantics,
Avro/JSON serialization, and dead letter queue handling.

Features:
- Async producer and consumer with aiokafka
- Exactly-once semantics with transactions
- Avro serialization with Schema Registry
- JSON serialization with validation
- Dead letter queue for failed messages
- Configurable partitioning strategies
- Comprehensive monitoring and statistics

Quick Start:
    from connectors.kafka import (
        AgentEventProducer,
        AgentEventConsumer,
        KafkaConfig,
        AgentCalculationCompleted,
        GreenLangTopics,
    )

    # Create configuration
    config = KafkaConfig(
        bootstrap_servers=["kafka:9092"],
        client_id="gl-agent-factory",
    )

    # Producer usage
    async with AgentEventProducer(config) as producer:
        event = AgentCalculationCompleted.create(
            agent_id="gl-001-carbon",
            calculation_type="scope1_emissions",
            formula_id="ghg.scope1.v1",
            input_params={"fuel_type": "natural_gas", "quantity": 1000},
            output_result={"emissions_kgco2e": 2500.0},
            processing_time_ms=45.2,
        )
        await producer.send_event(GreenLangTopics.AGENT_CALCULATIONS, event)

    # Consumer usage
    async with AgentEventConsumer(config) as consumer:
        @consumer.on(EventType.CALCULATION_COMPLETED)
        async def handle_calculation(event: AgentCalculationCompleted):
            print(f"Received: {event.calculation_type}")

        await consumer.subscribe([GreenLangTopics.AGENT_CALCULATIONS])
        await consumer.consume()

Architecture:
    Producer                          Consumer
    --------                          --------
    AgentEvent                        Topics
        |                                |
        v                                v
    Serializer                       Deserializer
    (JSON/Avro)                      (JSON/Avro)
        |                                |
        v                                v
    Partitioner                      Handler Registry
        |                                |
        v                                v
    Kafka Cluster <----------------> Kafka Cluster
        |                                |
        v                                v
    DLQ (failures)                   Offset Manager

Exactly-Once Semantics:
    Exactly-once delivery is achieved through:
    1. Idempotent producer (enable_idempotence=True)
    2. Transactional producer (transactional_id set)
    3. Consumer with read_committed isolation
    4. Manual offset commits after processing

    config = KafkaConfig(
        producer=KafkaProducerConfig(
            acks=AcksMode.ALL,
            enable_idempotence=True,
            transactional_id="gl-producer-001",
        ),
        consumer=KafkaConsumerConfig(
            enable_auto_commit=False,
            isolation_level=IsolationLevel.READ_COMMITTED,
        ),
    )

Event Types:
    - AgentCalculationCompleted: Calculation results with provenance
    - AgentAlertRaised: Alert events with severity levels
    - AgentRecommendationGenerated: Optimization recommendations
    - AgentHealthCheck: Health status events
    - AgentConfigurationChanged: Configuration change tracking

Topics:
    - gl.agent.events: General agent events
    - gl.agent.calculations: Calculation completed events
    - gl.agent.alerts: Alert events
    - gl.agent.recommendations: Recommendation events
    - gl.agent.health: Health check events
    - gl.agent.config: Configuration change events
    - gl.audit.log: Audit log events
    - gl.provenance: Provenance chain events

For more details, see the individual module documentation.
"""

# =============================================================================
# Configuration
# =============================================================================

from .config import (
    # Enumerations
    SecurityProtocol,
    SASLMechanism,
    CompressionType,
    AcksMode,
    AutoOffsetReset,
    IsolationLevel,
    PartitionStrategy,
    # Configuration models
    SSLConfig,
    SASLConfig,
    SchemaRegistryConfig,
    KafkaProducerConfig,
    KafkaConsumerConfig,
    DeadLetterQueueConfig,
    KafkaConfig,
    TopicConfig,
    GreenLangTopics,
    # Factory functions
    create_production_config,
    create_development_config,
)

# =============================================================================
# Events
# =============================================================================

from .events import (
    # Enumerations
    EventType,
    AlertSeverity,
    HealthStatus,
    RecommendationPriority,
    RecommendationCategory,
    # Metadata
    EventMetadata,
    # Base event
    AgentEvent,
    # Calculation events
    CalculationInput,
    CalculationOutput,
    AgentCalculationCompleted,
    # Alert events
    AlertContext,
    AgentAlertRaised,
    # Recommendation events
    RecommendationImpact,
    AgentRecommendationGenerated,
    # Health events
    HealthMetrics,
    DependencyHealth,
    AgentHealthCheck,
    # Configuration events
    ConfigChange,
    AgentConfigurationChanged,
    # Factory
    AgentEventFactory,
)

# =============================================================================
# Schemas
# =============================================================================

from .schemas import (
    SchemaVersion,
    AVRO_NAMESPACE,
    AgentEventSchemas,
    get_avro_schema,
    validate_json_schema,
)

# =============================================================================
# Serializers
# =============================================================================

from .serializers import (
    SerializationFormat,
    CompressionFormat,
    SerializerConfig,
    SchemaRegistryClient,
    AgentEventSerializer,
    AgentEventDeserializer,
)

# =============================================================================
# Producer
# =============================================================================

from .producer import (
    AgentEventProducer,
    AgentEventPartitioner,
    ProducerStatistics,
    SendResult,
    ProducerCallbackHandler,
    create_producer,
)

# =============================================================================
# Consumer
# =============================================================================

from .consumer import (
    AgentEventConsumer,
    EventHandlerRegistry,
    OffsetManager,
    ConsumerStatistics,
    ConsumedMessage,
    create_consumer,
    EventHandler,
    EventFilter,
)

# =============================================================================
# Version
# =============================================================================

__version__ = "1.0.0"

# =============================================================================
# All Exports
# =============================================================================

__all__ = [
    # Version
    "__version__",
    # Configuration enumerations
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
    "create_production_config",
    "create_development_config",
    # Event enumerations
    "EventType",
    "AlertSeverity",
    "HealthStatus",
    "RecommendationPriority",
    "RecommendationCategory",
    # Event models
    "EventMetadata",
    "AgentEvent",
    "CalculationInput",
    "CalculationOutput",
    "AgentCalculationCompleted",
    "AlertContext",
    "AgentAlertRaised",
    "RecommendationImpact",
    "AgentRecommendationGenerated",
    "HealthMetrics",
    "DependencyHealth",
    "AgentHealthCheck",
    "ConfigChange",
    "AgentConfigurationChanged",
    "AgentEventFactory",
    # Schemas
    "SchemaVersion",
    "AVRO_NAMESPACE",
    "AgentEventSchemas",
    "get_avro_schema",
    "validate_json_schema",
    # Serialization
    "SerializationFormat",
    "CompressionFormat",
    "SerializerConfig",
    "SchemaRegistryClient",
    "AgentEventSerializer",
    "AgentEventDeserializer",
    # Producer
    "AgentEventProducer",
    "AgentEventPartitioner",
    "ProducerStatistics",
    "SendResult",
    "ProducerCallbackHandler",
    "create_producer",
    # Consumer
    "AgentEventConsumer",
    "EventHandlerRegistry",
    "OffsetManager",
    "ConsumerStatistics",
    "ConsumedMessage",
    "create_consumer",
    "EventHandler",
    "EventFilter",
]
