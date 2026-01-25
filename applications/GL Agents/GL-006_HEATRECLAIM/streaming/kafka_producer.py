"""
Kafka Event Producer for GL-006 HEATRECLAIM

Implements topic-based event streaming for heat exchanger network optimization:
- heatreclaim.<site>.streams - Heat stream data updates
- heatreclaim.<site>.pinch - Pinch analysis results
- heatreclaim.<site>.hen - Heat Exchanger Network synthesis results
- heatreclaim.optimization - Optimization recommendations
- heatreclaim.alerts - Energy recovery alerts

Includes SHA-256 provenance tracking for regulatory compliance.
"""

import asyncio
import json
import logging
import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)


# =============================================================================
# Message Schema Definitions
# =============================================================================

class MessageType(str, Enum):
    """Message types for HEATRECLAIM events."""
    STREAM_UPDATE = "stream_update"
    PINCH_ANALYSIS = "pinch_analysis"
    HEN_SYNTHESIS = "hen_synthesis"
    OPTIMIZATION = "optimization"
    ALERT = "alert"
    EXERGY_ANALYSIS = "exergy_analysis"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class DataQuality(str, Enum):
    """Data quality indicators."""
    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"
    STALE = "stale"


class MessageHeader(BaseModel):
    """Standard message header for all HEATRECLAIM messages."""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType
    version: str = Field("1.0", description="Schema version")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source: str = Field(..., description="Source system identifier")
    correlation_id: Optional[str] = Field(None, description="For request-response correlation")
    provenance_hash: Optional[str] = Field(None, description="SHA-256 provenance hash")


class HeatStreamMessage(BaseModel):
    """Heat stream data update message."""
    header: MessageHeader
    site_id: str = Field(..., description="Site identifier")
    stream_id: str = Field(..., description="Heat stream identifier")
    stream_type: str = Field(..., description="HOT or COLD")

    # Stream properties
    T_supply_C: float = Field(..., description="Supply temperature (C)")
    T_target_C: float = Field(..., description="Target temperature (C)")
    m_dot_kg_s: float = Field(..., ge=0, description="Mass flow rate (kg/s)")
    Cp_kJ_kgK: float = Field(..., gt=0, description="Specific heat capacity (kJ/kg-K)")
    duty_kW: float = Field(..., description="Heat duty (kW)")

    # Data quality
    quality: DataQuality = Field(DataQuality.GOOD)
    source_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    model_config = ConfigDict(use_enum_values=True)

    def to_kafka_message(self) -> bytes:
        """Serialize to Kafka message bytes."""
        return json.dumps(self.model_dump(), default=str).encode('utf-8')


class PinchAnalysisMessage(BaseModel):
    """Pinch analysis results message."""
    header: MessageHeader
    site_id: str = Field(..., description="Site identifier")
    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Pinch results
    pinch_temperature_C: float = Field(..., description="Pinch temperature (C)")
    delta_t_min_C: float = Field(..., description="Minimum approach temperature (C)")
    minimum_hot_utility_kW: float = Field(..., ge=0, description="Minimum hot utility (kW)")
    minimum_cold_utility_kW: float = Field(..., ge=0, description="Minimum cold utility (kW)")
    maximum_heat_recovery_kW: float = Field(..., ge=0, description="Maximum heat recovery (kW)")

    # Composite curve data
    hot_composite_T_C: List[float] = Field(default_factory=list)
    hot_composite_H_kW: List[float] = Field(default_factory=list)
    cold_composite_T_C: List[float] = Field(default_factory=list)
    cold_composite_H_kW: List[float] = Field(default_factory=list)

    # Processing metadata
    computation_hash: str = Field(..., description="SHA-256 hash of inputs")
    processing_time_ms: float = Field(..., ge=0)

    model_config = ConfigDict(use_enum_values=True)

    def to_kafka_message(self) -> bytes:
        """Serialize to Kafka message bytes."""
        return json.dumps(self.model_dump(), default=str).encode('utf-8')


class HENSynthesisMessage(BaseModel):
    """Heat Exchanger Network synthesis results message."""
    header: MessageHeader
    site_id: str = Field(..., description="Site identifier")
    synthesis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Network design
    total_heat_exchangers: int = Field(..., ge=0)
    total_heat_exchange_kW: float = Field(..., ge=0)
    total_area_m2: float = Field(..., ge=0)
    total_capital_cost_usd: float = Field(..., ge=0)
    annual_operating_savings_usd: float = Field(..., ge=0)
    simple_payback_years: float = Field(..., ge=0)

    # Heat exchanger matches
    matches: List[Dict[str, Any]] = Field(default_factory=list)

    # Provenance
    computation_hash: str = Field(..., description="SHA-256 hash")
    processing_time_ms: float = Field(..., ge=0)

    model_config = ConfigDict(use_enum_values=True)

    def to_kafka_message(self) -> bytes:
        """Serialize to Kafka message bytes."""
        return json.dumps(self.model_dump(), default=str).encode('utf-8')


class OptimizationMessage(BaseModel):
    """Heat recovery optimization recommendation message."""
    header: MessageHeader
    site_id: str = Field(..., description="Site identifier")
    recommendation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Recommendation
    recommendation_type: str = Field(..., description="Type of optimization")
    description: str = Field(..., description="Human-readable description")
    priority: str = Field("medium", description="Priority: low, medium, high, critical")

    # Expected benefits
    energy_savings_kW: float = Field(..., ge=0)
    annual_cost_savings_usd: float = Field(..., ge=0)
    co2_reduction_kg_year: float = Field(..., ge=0)
    implementation_cost_usd: float = Field(..., ge=0)
    roi_percent: float = Field(...)

    # Implementation details
    affected_streams: List[str] = Field(default_factory=list)
    required_actions: List[str] = Field(default_factory=list)

    # Confidence
    confidence_percent: float = Field(..., ge=0, le=100)

    model_config = ConfigDict(use_enum_values=True)

    def to_kafka_message(self) -> bytes:
        """Serialize to Kafka message bytes."""
        return json.dumps(self.model_dump(), default=str).encode('utf-8')


class AlertMessage(BaseModel):
    """Heat recovery alert message."""
    header: MessageHeader
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    site_id: str = Field(..., description="Site identifier")

    # Alert details
    alert_type: str = Field(..., description="Alert type code")
    alert_name: str = Field(..., description="Human-readable alert name")
    severity: AlertSeverity
    description: str
    recommended_actions: List[str] = Field(default_factory=list)

    # Supporting data
    affected_streams: List[str] = Field(default_factory=list)
    current_value: Optional[float] = Field(None)
    threshold_value: Optional[float] = Field(None)
    potential_loss_kW: Optional[float] = Field(None)

    model_config = ConfigDict(use_enum_values=True)

    def to_kafka_message(self) -> bytes:
        """Serialize to Kafka message bytes."""
        return json.dumps(self.model_dump(), default=str).encode('utf-8')


# =============================================================================
# Kafka Producer Configuration
# =============================================================================

class KafkaProducerConfig(BaseModel):
    """Kafka producer configuration."""
    bootstrap_servers: str = Field(..., description="Comma-separated list of brokers")
    security_protocol: str = Field("SASL_SSL", description="Security protocol")
    sasl_mechanism: str = Field("PLAIN", description="SASL mechanism")
    sasl_username: Optional[str] = Field(None)
    sasl_password: Optional[str] = Field(None)
    ssl_ca_location: Optional[str] = Field(None)

    # Producer settings
    acks: str = Field("all", description="Acknowledgment level")
    retries: int = Field(3)
    compression_type: str = Field("gzip")
    enable_idempotence: bool = Field(True)


# =============================================================================
# Kafka Event Producer
# =============================================================================

class HeatReclaimKafkaProducer:
    """
    Kafka producer for GL-006 HEATRECLAIM event streaming.

    Topics:
    - heatreclaim.<site>.streams - Heat stream data updates
    - heatreclaim.<site>.pinch - Pinch analysis results
    - heatreclaim.<site>.hen - HEN synthesis results
    - heatreclaim.optimization - Optimization recommendations
    - heatreclaim.alerts - Energy recovery alerts
    """

    TOPIC_STREAMS = "heatreclaim.{site}.streams"
    TOPIC_PINCH = "heatreclaim.{site}.pinch"
    TOPIC_HEN = "heatreclaim.{site}.hen"
    TOPIC_OPTIMIZATION = "heatreclaim.optimization"
    TOPIC_ALERTS = "heatreclaim.alerts"

    def __init__(self, config: KafkaProducerConfig):
        """Initialize Kafka producer."""
        self.config = config
        self._producer = None
        self._started = False

        # Metrics
        self._messages_sent = 0
        self._messages_failed = 0
        self._bytes_sent = 0

    async def start(self) -> None:
        """Start the Kafka producer."""
        if self._started:
            return

        logger.info(f"Starting HEATRECLAIM Kafka producer for {self.config.bootstrap_servers}")

        # In production, initialize aiokafka or confluent-kafka producer
        # from aiokafka import AIOKafkaProducer
        # self._producer = AIOKafkaProducer(...)
        # await self._producer.start()

        self._started = True
        logger.info("HEATRECLAIM Kafka producer started")

    async def stop(self) -> None:
        """Stop the Kafka producer."""
        if not self._started:
            return

        logger.info("Stopping HEATRECLAIM Kafka producer")

        if self._producer:
            # await self._producer.flush()
            # await self._producer.stop()
            pass

        self._started = False
        logger.info(f"Producer stopped. Sent: {self._messages_sent}, Failed: {self._messages_failed}")

    def _compute_provenance_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash for audit trail."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    async def _produce(
        self,
        topic: str,
        key: Optional[str],
        value: bytes,
        headers: Optional[Dict[str, str]] = None
    ) -> bool:
        """Internal method to produce a message."""
        if not self._started:
            raise RuntimeError("Producer not started")

        try:
            # In production:
            # await self._producer.send_and_wait(topic, key, value, headers)

            self._messages_sent += 1
            self._bytes_sent += len(value)

            logger.debug(f"Produced message to {topic}, key={key}, size={len(value)}")
            return True

        except Exception as e:
            self._messages_failed += 1
            logger.error(f"Failed to produce message to {topic}: {e}")
            return False

    async def publish_stream_update(
        self,
        site_id: str,
        stream_id: str,
        stream_type: str,
        T_supply_C: float,
        T_target_C: float,
        m_dot_kg_s: float,
        Cp_kJ_kgK: float
    ) -> bool:
        """Publish heat stream update."""
        topic = self.TOPIC_STREAMS.format(site=site_id)

        duty_kW = m_dot_kg_s * Cp_kJ_kgK * abs(T_supply_C - T_target_C)

        message = HeatStreamMessage(
            header=MessageHeader(
                message_type=MessageType.STREAM_UPDATE,
                source=f"heatreclaim.{site_id}",
                provenance_hash=self._compute_provenance_hash({
                    "stream_id": stream_id,
                    "T_supply_C": T_supply_C,
                    "T_target_C": T_target_C,
                    "m_dot_kg_s": m_dot_kg_s
                })
            ),
            site_id=site_id,
            stream_id=stream_id,
            stream_type=stream_type,
            T_supply_C=T_supply_C,
            T_target_C=T_target_C,
            m_dot_kg_s=m_dot_kg_s,
            Cp_kJ_kgK=Cp_kJ_kgK,
            duty_kW=duty_kW
        )

        return await self._produce(
            topic=topic,
            key=stream_id,
            value=message.to_kafka_message(),
            headers={"message_type": "stream_update", "stream_type": stream_type}
        )

    async def publish_pinch_analysis(
        self,
        site_id: str,
        pinch_temperature_C: float,
        delta_t_min_C: float,
        minimum_hot_utility_kW: float,
        minimum_cold_utility_kW: float,
        maximum_heat_recovery_kW: float,
        computation_hash: str,
        processing_time_ms: float,
        hot_composite: Optional[tuple] = None,
        cold_composite: Optional[tuple] = None
    ) -> bool:
        """Publish pinch analysis results."""
        topic = self.TOPIC_PINCH.format(site=site_id)

        message = PinchAnalysisMessage(
            header=MessageHeader(
                message_type=MessageType.PINCH_ANALYSIS,
                source=f"heatreclaim.{site_id}",
                provenance_hash=computation_hash
            ),
            site_id=site_id,
            pinch_temperature_C=pinch_temperature_C,
            delta_t_min_C=delta_t_min_C,
            minimum_hot_utility_kW=minimum_hot_utility_kW,
            minimum_cold_utility_kW=minimum_cold_utility_kW,
            maximum_heat_recovery_kW=maximum_heat_recovery_kW,
            hot_composite_T_C=hot_composite[0] if hot_composite else [],
            hot_composite_H_kW=hot_composite[1] if hot_composite else [],
            cold_composite_T_C=cold_composite[0] if cold_composite else [],
            cold_composite_H_kW=cold_composite[1] if cold_composite else [],
            computation_hash=computation_hash,
            processing_time_ms=processing_time_ms
        )

        return await self._produce(
            topic=topic,
            key=site_id,
            value=message.to_kafka_message(),
            headers={
                "message_type": "pinch_analysis",
                "pinch_temp_C": str(pinch_temperature_C)
            }
        )

    async def publish_hen_synthesis(
        self,
        site_id: str,
        total_heat_exchangers: int,
        total_heat_exchange_kW: float,
        total_area_m2: float,
        total_capital_cost_usd: float,
        annual_operating_savings_usd: float,
        matches: List[Dict[str, Any]],
        computation_hash: str,
        processing_time_ms: float
    ) -> bool:
        """Publish HEN synthesis results."""
        topic = self.TOPIC_HEN.format(site=site_id)

        payback = (
            total_capital_cost_usd / annual_operating_savings_usd
            if annual_operating_savings_usd > 0 else float('inf')
        )

        message = HENSynthesisMessage(
            header=MessageHeader(
                message_type=MessageType.HEN_SYNTHESIS,
                source=f"heatreclaim.{site_id}",
                provenance_hash=computation_hash
            ),
            site_id=site_id,
            total_heat_exchangers=total_heat_exchangers,
            total_heat_exchange_kW=total_heat_exchange_kW,
            total_area_m2=total_area_m2,
            total_capital_cost_usd=total_capital_cost_usd,
            annual_operating_savings_usd=annual_operating_savings_usd,
            simple_payback_years=payback,
            matches=matches,
            computation_hash=computation_hash,
            processing_time_ms=processing_time_ms
        )

        return await self._produce(
            topic=topic,
            key=site_id,
            value=message.to_kafka_message(),
            headers={"message_type": "hen_synthesis"}
        )

    async def publish_optimization(
        self,
        site_id: str,
        recommendation_type: str,
        description: str,
        energy_savings_kW: float,
        annual_cost_savings_usd: float,
        co2_reduction_kg_year: float,
        implementation_cost_usd: float,
        affected_streams: List[str],
        required_actions: List[str],
        confidence_percent: float = 90.0,
        priority: str = "medium"
    ) -> bool:
        """Publish optimization recommendation."""
        roi = (
            (annual_cost_savings_usd / implementation_cost_usd * 100)
            if implementation_cost_usd > 0 else 0.0
        )

        message = OptimizationMessage(
            header=MessageHeader(
                message_type=MessageType.OPTIMIZATION,
                source=f"heatreclaim.{site_id}"
            ),
            site_id=site_id,
            recommendation_type=recommendation_type,
            description=description,
            priority=priority,
            energy_savings_kW=energy_savings_kW,
            annual_cost_savings_usd=annual_cost_savings_usd,
            co2_reduction_kg_year=co2_reduction_kg_year,
            implementation_cost_usd=implementation_cost_usd,
            roi_percent=roi,
            affected_streams=affected_streams,
            required_actions=required_actions,
            confidence_percent=confidence_percent
        )

        return await self._produce(
            topic=self.TOPIC_OPTIMIZATION,
            key=f"{site_id}.{recommendation_type}",
            value=message.to_kafka_message(),
            headers={
                "message_type": "optimization",
                "priority": priority,
                "site_id": site_id
            }
        )

    async def publish_alert(
        self,
        site_id: str,
        alert_type: str,
        alert_name: str,
        severity: AlertSeverity,
        description: str,
        recommended_actions: List[str],
        affected_streams: Optional[List[str]] = None,
        potential_loss_kW: Optional[float] = None
    ) -> bool:
        """Publish heat recovery alert."""
        message = AlertMessage(
            header=MessageHeader(
                message_type=MessageType.ALERT,
                source=f"heatreclaim.{site_id}"
            ),
            site_id=site_id,
            alert_type=alert_type,
            alert_name=alert_name,
            severity=severity,
            description=description,
            recommended_actions=recommended_actions,
            affected_streams=affected_streams or [],
            potential_loss_kW=potential_loss_kW
        )

        return await self._produce(
            topic=self.TOPIC_ALERTS,
            key=message.alert_id,
            value=message.to_kafka_message(),
            headers={
                "message_type": "alert",
                "severity": severity.value,
                "site_id": site_id,
                "alert_type": alert_type
            }
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get producer metrics."""
        total = self._messages_sent + self._messages_failed
        return {
            "messages_sent": self._messages_sent,
            "messages_failed": self._messages_failed,
            "bytes_sent": self._bytes_sent,
            "success_rate": self._messages_sent / total if total > 0 else 1.0
        }
