"""
GL-009 ThermalIQ - Kafka Event Producer

Implements topic-based event streaming for thermal fluid analysis:
- thermaliq.analysis.requests - Analysis request submissions
- thermaliq.analysis.results - Analysis completion events
- thermaliq.fluids.updates - Fluid property updates
- thermaliq.exergy.results - Exergy calculation results
- thermaliq.sankey.generated - Sankey diagram generation
- thermaliq.alerts - System alerts and notifications

Includes SHA-256 provenance tracking for regulatory compliance.
"""

import asyncio
import json
import logging
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ConfigDict

from .event_schemas import (
    EventType,
    MessageHeader,
    AnalysisRequestedEvent,
    AnalysisCompletedEvent,
    FluidPropertyUpdatedEvent,
    ExergyCalculatedEvent,
    SankeyGeneratedEvent,
    AlertEvent,
    AlertSeverity,
    DataQuality,
)

logger = logging.getLogger(__name__)


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
    retries: int = Field(3, description="Number of retries")
    retry_backoff_ms: int = Field(1000, description="Retry backoff in ms")
    compression_type: str = Field("gzip", description="Compression type")
    enable_idempotence: bool = Field(True, description="Enable idempotent producer")
    batch_size: int = Field(16384, description="Batch size in bytes")
    linger_ms: int = Field(5, description="Linger time in ms")

    # Schema registry
    schema_registry_url: Optional[str] = Field(None, description="Schema Registry URL")

    model_config = ConfigDict(extra="allow")


# =============================================================================
# ThermalIQ Kafka Producer
# =============================================================================

class ThermalIQKafkaProducer:
    """
    Kafka producer for GL-009 ThermalIQ event streaming.

    Topics:
    - thermaliq.analysis.requests - Analysis requests
    - thermaliq.analysis.results - Analysis results
    - thermaliq.fluids.updates - Fluid property updates
    - thermaliq.exergy.results - Exergy analysis results
    - thermaliq.sankey.generated - Sankey diagrams
    - thermaliq.alerts - System alerts
    """

    # Topic definitions
    TOPIC_ANALYSIS_REQUESTS = "thermaliq.analysis.requests"
    TOPIC_ANALYSIS_RESULTS = "thermaliq.analysis.results"
    TOPIC_FLUIDS_UPDATES = "thermaliq.fluids.updates"
    TOPIC_EXERGY_RESULTS = "thermaliq.exergy.results"
    TOPIC_SANKEY_GENERATED = "thermaliq.sankey.generated"
    TOPIC_ALERTS = "thermaliq.alerts"

    def __init__(self, config: KafkaProducerConfig):
        """
        Initialize Kafka producer.

        Args:
            config: Producer configuration
        """
        self.config = config
        self._producer = None
        self._started = False
        self._schema_registry = None

        # Metrics
        self._messages_sent = 0
        self._messages_failed = 0
        self._bytes_sent = 0
        self._last_error: Optional[str] = None

    async def start(self) -> None:
        """Start the Kafka producer."""
        if self._started:
            logger.warning("ThermalIQ Kafka producer already started")
            return

        logger.info(f"Starting ThermalIQ Kafka producer for {self.config.bootstrap_servers}")

        try:
            # In production, initialize aiokafka or confluent-kafka producer:
            # from aiokafka import AIOKafkaProducer
            # self._producer = AIOKafkaProducer(
            #     bootstrap_servers=self.config.bootstrap_servers,
            #     security_protocol=self.config.security_protocol,
            #     sasl_mechanism=self.config.sasl_mechanism,
            #     sasl_plain_username=self.config.sasl_username,
            #     sasl_plain_password=self.config.sasl_password,
            #     acks=self.config.acks,
            #     compression_type=self.config.compression_type,
            #     enable_idempotence=self.config.enable_idempotence,
            # )
            # await self._producer.start()

            # Initialize Schema Registry client if configured
            if self.config.schema_registry_url:
                logger.info(f"Connecting to Schema Registry: {self.config.schema_registry_url}")
                # from confluent_kafka.schema_registry import SchemaRegistryClient
                # self._schema_registry = SchemaRegistryClient(...)

            self._started = True
            logger.info("ThermalIQ Kafka producer started successfully")

        except Exception as e:
            logger.error(f"Failed to start Kafka producer: {e}")
            raise

    async def stop(self) -> None:
        """Stop the Kafka producer."""
        if not self._started:
            return

        logger.info("Stopping ThermalIQ Kafka producer")

        try:
            if self._producer:
                # await self._producer.flush()
                # await self._producer.stop()
                pass

            self._started = False
            logger.info(
                f"Producer stopped. Sent: {self._messages_sent}, "
                f"Failed: {self._messages_failed}, "
                f"Bytes: {self._bytes_sent}"
            )

        except Exception as e:
            logger.error(f"Error stopping producer: {e}")

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
        """
        Internal method to produce a message.

        Args:
            topic: Kafka topic
            key: Message key
            value: Message value (bytes)
            headers: Optional message headers

        Returns:
            True if successful, False otherwise
        """
        if not self._started:
            raise RuntimeError("Producer not started. Call start() first.")

        try:
            # Convert headers to Kafka format
            kafka_headers = [
                (k, v.encode("utf-8")) for k, v in (headers or {}).items()
            ]

            # In production:
            # await self._producer.send_and_wait(
            #     topic=topic,
            #     key=key.encode("utf-8") if key else None,
            #     value=value,
            #     headers=kafka_headers
            # )

            self._messages_sent += 1
            self._bytes_sent += len(value)

            logger.debug(
                f"Produced message to {topic}, key={key}, size={len(value)} bytes"
            )
            return True

        except Exception as e:
            self._messages_failed += 1
            self._last_error = str(e)
            logger.error(f"Failed to produce message to {topic}: {e}")
            return False

    # =========================================================================
    # Analysis Events
    # =========================================================================

    async def publish_analysis_request(
        self,
        request_id: str,
        streams: List[Dict[str, Any]],
        ambient_temperature_C: float = 25.0,
        ambient_pressure_kPa: float = 101.325,
        analysis_mode: str = "full",
        include_exergy: bool = True,
        include_sankey: bool = True,
        requester_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> bool:
        """
        Publish thermal analysis request event.

        Args:
            request_id: Unique request identifier
            streams: List of stream data dictionaries
            ambient_temperature_C: Ambient temperature
            ambient_pressure_kPa: Ambient pressure
            analysis_mode: Analysis mode (full, quick, detailed)
            include_exergy: Include exergy analysis
            include_sankey: Include Sankey diagram
            requester_id: ID of the requester
            correlation_id: Correlation ID for tracing

        Returns:
            True if published successfully
        """
        event = AnalysisRequestedEvent(
            header=MessageHeader(
                event_type=EventType.ANALYSIS_REQUESTED,
                source="thermaliq",
                correlation_id=correlation_id,
                provenance_hash=self._compute_provenance_hash({
                    "request_id": request_id,
                    "streams": streams
                })
            ),
            request_id=request_id,
            streams=streams,
            ambient_temperature_C=ambient_temperature_C,
            ambient_pressure_kPa=ambient_pressure_kPa,
            analysis_mode=analysis_mode,
            include_exergy=include_exergy,
            include_sankey=include_sankey,
            requester_id=requester_id,
        )

        return await self._produce(
            topic=self.TOPIC_ANALYSIS_REQUESTS,
            key=request_id,
            value=event.to_bytes(),
            headers={
                "event_type": EventType.ANALYSIS_REQUESTED.value,
                "request_id": request_id,
                "analysis_mode": analysis_mode,
            }
        )

    async def publish_analysis_result(
        self,
        request_id: str,
        status: str,
        total_heat_duty_kW: float,
        total_mass_flow_kg_s: float,
        first_law_efficiency_percent: float,
        second_law_efficiency_percent: Optional[float] = None,
        stream_results: Optional[List[Dict[str, Any]]] = None,
        exergy_destruction_kW: Optional[float] = None,
        exergy_efficiency_percent: Optional[float] = None,
        sankey_node_count: Optional[int] = None,
        sankey_link_count: Optional[int] = None,
        computation_hash: str = "",
        processing_time_ms: float = 0.0,
        correlation_id: Optional[str] = None,
    ) -> bool:
        """
        Publish analysis completion event.

        Args:
            request_id: Original request ID
            status: Completion status
            total_heat_duty_kW: Total heat duty
            total_mass_flow_kg_s: Total mass flow
            first_law_efficiency_percent: First law efficiency
            second_law_efficiency_percent: Second law efficiency (optional)
            stream_results: Per-stream results
            exergy_destruction_kW: Exergy destruction
            exergy_efficiency_percent: Exergetic efficiency
            sankey_node_count: Number of Sankey nodes
            sankey_link_count: Number of Sankey links
            computation_hash: Provenance hash
            processing_time_ms: Processing time
            correlation_id: Correlation ID

        Returns:
            True if published successfully
        """
        event = AnalysisCompletedEvent(
            header=MessageHeader(
                event_type=EventType.ANALYSIS_COMPLETED,
                source="thermaliq",
                correlation_id=correlation_id,
                provenance_hash=computation_hash or self._compute_provenance_hash({
                    "request_id": request_id,
                    "heat_duty": total_heat_duty_kW
                })
            ),
            request_id=request_id,
            status=status,
            total_heat_duty_kW=total_heat_duty_kW,
            total_mass_flow_kg_s=total_mass_flow_kg_s,
            first_law_efficiency_percent=first_law_efficiency_percent,
            second_law_efficiency_percent=second_law_efficiency_percent,
            stream_results=stream_results or [],
            exergy_destruction_kW=exergy_destruction_kW,
            exergy_efficiency_percent=exergy_efficiency_percent,
            sankey_node_count=sankey_node_count,
            sankey_link_count=sankey_link_count,
            computation_hash=computation_hash,
            processing_time_ms=processing_time_ms,
        )

        return await self._produce(
            topic=self.TOPIC_ANALYSIS_RESULTS,
            key=request_id,
            value=event.to_bytes(),
            headers={
                "event_type": EventType.ANALYSIS_COMPLETED.value,
                "request_id": request_id,
                "status": status,
            }
        )

    # =========================================================================
    # Fluid Property Events
    # =========================================================================

    async def publish_fluid_update(
        self,
        fluid_name: str,
        temperature_C: float,
        pressure_kPa: float,
        phase: str,
        density_kg_m3: float,
        specific_heat_kJ_kgK: float,
        enthalpy_kJ_kg: float,
        entropy_kJ_kgK: float,
        internal_energy_kJ_kg: float,
        viscosity_Pa_s: float,
        thermal_conductivity_W_mK: float,
        prandtl_number: float,
        quality: Optional[float] = None,
        data_source: str = "CoolProp",
        computation_hash: str = "",
    ) -> bool:
        """
        Publish fluid property update event.

        Args:
            fluid_name: Name of the fluid
            temperature_C: Temperature
            pressure_kPa: Pressure
            phase: Fluid phase
            density_kg_m3: Density
            specific_heat_kJ_kgK: Specific heat
            enthalpy_kJ_kg: Enthalpy
            entropy_kJ_kgK: Entropy
            internal_energy_kJ_kg: Internal energy
            viscosity_Pa_s: Viscosity
            thermal_conductivity_W_mK: Thermal conductivity
            prandtl_number: Prandtl number
            quality: Vapor quality (two-phase)
            data_source: Property data source
            computation_hash: Provenance hash

        Returns:
            True if published successfully
        """
        event = FluidPropertyUpdatedEvent(
            header=MessageHeader(
                event_type=EventType.FLUID_PROPERTY_UPDATED,
                source="thermaliq",
                provenance_hash=computation_hash or self._compute_provenance_hash({
                    "fluid": fluid_name,
                    "T": temperature_C,
                    "P": pressure_kPa
                })
            ),
            fluid_name=fluid_name,
            temperature_C=temperature_C,
            pressure_kPa=pressure_kPa,
            phase=phase,
            density_kg_m3=density_kg_m3,
            specific_heat_kJ_kgK=specific_heat_kJ_kgK,
            enthalpy_kJ_kg=enthalpy_kJ_kg,
            entropy_kJ_kgK=entropy_kJ_kgK,
            internal_energy_kJ_kg=internal_energy_kJ_kg,
            viscosity_Pa_s=viscosity_Pa_s,
            thermal_conductivity_W_mK=thermal_conductivity_W_mK,
            prandtl_number=prandtl_number,
            quality=quality,
            data_source=data_source,
            computation_hash=computation_hash,
        )

        return await self._produce(
            topic=self.TOPIC_FLUIDS_UPDATES,
            key=f"{fluid_name}:{temperature_C}:{pressure_kPa}",
            value=event.to_bytes(),
            headers={
                "event_type": EventType.FLUID_PROPERTY_UPDATED.value,
                "fluid_name": fluid_name,
                "phase": phase,
            }
        )

    # =========================================================================
    # Exergy Events
    # =========================================================================

    async def publish_exergy_result(
        self,
        request_id: str,
        dead_state_temperature_C: float,
        dead_state_pressure_kPa: float,
        total_exergy_input_kW: float,
        total_exergy_output_kW: float,
        total_exergy_destruction_kW: float,
        exergy_efficiency_percent: float,
        physical_exergy_kW: float,
        improvement_potential_kW: float,
        components: Optional[List[Dict[str, Any]]] = None,
        chemical_exergy_kW: Optional[float] = None,
        kinetic_exergy_kW: Optional[float] = None,
        potential_exergy_kW: Optional[float] = None,
        computation_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> bool:
        """
        Publish exergy analysis result event.

        Args:
            request_id: Request identifier
            dead_state_temperature_C: Dead state temperature
            dead_state_pressure_kPa: Dead state pressure
            total_exergy_input_kW: Total exergy input
            total_exergy_output_kW: Total exergy output
            total_exergy_destruction_kW: Total exergy destruction
            exergy_efficiency_percent: Exergetic efficiency
            physical_exergy_kW: Physical exergy
            improvement_potential_kW: Improvement potential
            components: Component-level breakdown
            chemical_exergy_kW: Chemical exergy
            kinetic_exergy_kW: Kinetic exergy
            potential_exergy_kW: Potential exergy
            computation_hash: Provenance hash
            processing_time_ms: Processing time

        Returns:
            True if published successfully
        """
        event = ExergyCalculatedEvent(
            header=MessageHeader(
                event_type=EventType.EXERGY_CALCULATED,
                source="thermaliq",
                provenance_hash=computation_hash or self._compute_provenance_hash({
                    "request_id": request_id,
                    "exergy_destruction": total_exergy_destruction_kW
                })
            ),
            request_id=request_id,
            dead_state_temperature_C=dead_state_temperature_C,
            dead_state_pressure_kPa=dead_state_pressure_kPa,
            total_exergy_input_kW=total_exergy_input_kW,
            total_exergy_output_kW=total_exergy_output_kW,
            total_exergy_destruction_kW=total_exergy_destruction_kW,
            exergy_efficiency_percent=exergy_efficiency_percent,
            physical_exergy_kW=physical_exergy_kW,
            chemical_exergy_kW=chemical_exergy_kW,
            kinetic_exergy_kW=kinetic_exergy_kW,
            potential_exergy_kW=potential_exergy_kW,
            components=components or [],
            improvement_potential_kW=improvement_potential_kW,
            computation_hash=computation_hash,
            processing_time_ms=processing_time_ms,
        )

        return await self._produce(
            topic=self.TOPIC_EXERGY_RESULTS,
            key=request_id,
            value=event.to_bytes(),
            headers={
                "event_type": EventType.EXERGY_CALCULATED.value,
                "request_id": request_id,
                "efficiency": str(round(exergy_efficiency_percent, 1)),
            }
        )

    # =========================================================================
    # Sankey Events
    # =========================================================================

    async def publish_sankey(
        self,
        request_id: str,
        diagram_type: str,
        nodes: List[Dict[str, Any]],
        links: List[Dict[str, Any]],
        total_input_kW: float,
        total_output_kW: float,
        total_losses_kW: float,
        layout_direction: str = "left_to_right",
        color_scheme: str = "thermal",
        computation_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> bool:
        """
        Publish Sankey diagram generation event.

        Args:
            request_id: Request identifier
            diagram_type: Type (energy or exergy)
            nodes: Sankey nodes
            links: Sankey links
            total_input_kW: Total input
            total_output_kW: Total output
            total_losses_kW: Total losses
            layout_direction: Layout direction
            color_scheme: Color scheme
            computation_hash: Provenance hash
            processing_time_ms: Processing time

        Returns:
            True if published successfully
        """
        event = SankeyGeneratedEvent(
            header=MessageHeader(
                event_type=EventType.SANKEY_GENERATED,
                source="thermaliq",
                provenance_hash=computation_hash or self._compute_provenance_hash({
                    "request_id": request_id,
                    "type": diagram_type
                })
            ),
            request_id=request_id,
            diagram_type=diagram_type,
            nodes=nodes,
            links=links,
            total_input_kW=total_input_kW,
            total_output_kW=total_output_kW,
            total_losses_kW=total_losses_kW,
            layout_direction=layout_direction,
            color_scheme=color_scheme,
            computation_hash=computation_hash,
            processing_time_ms=processing_time_ms,
        )

        return await self._produce(
            topic=self.TOPIC_SANKEY_GENERATED,
            key=request_id,
            value=event.to_bytes(),
            headers={
                "event_type": EventType.SANKEY_GENERATED.value,
                "request_id": request_id,
                "diagram_type": diagram_type,
            }
        )

    # =========================================================================
    # Alert Events
    # =========================================================================

    async def publish_alert(
        self,
        alert_type: str,
        alert_name: str,
        severity: AlertSeverity,
        description: str,
        recommended_actions: Optional[List[str]] = None,
        affected_streams: Optional[List[str]] = None,
        current_value: Optional[float] = None,
        threshold_value: Optional[float] = None,
        potential_efficiency_loss_percent: Optional[float] = None,
        potential_exergy_loss_kW: Optional[float] = None,
    ) -> bool:
        """
        Publish system alert event.

        Args:
            alert_type: Alert type code
            alert_name: Human-readable name
            severity: Alert severity
            description: Alert description
            recommended_actions: List of recommended actions
            affected_streams: List of affected stream IDs
            current_value: Current measured value
            threshold_value: Threshold that was exceeded
            potential_efficiency_loss_percent: Potential efficiency loss
            potential_exergy_loss_kW: Potential exergy loss

        Returns:
            True if published successfully
        """
        event = AlertEvent(
            header=MessageHeader(
                event_type=EventType.ALERT,
                source="thermaliq"
            ),
            alert_type=alert_type,
            alert_name=alert_name,
            severity=severity,
            description=description,
            recommended_actions=recommended_actions or [],
            affected_streams=affected_streams or [],
            current_value=current_value,
            threshold_value=threshold_value,
            potential_efficiency_loss_percent=potential_efficiency_loss_percent,
            potential_exergy_loss_kW=potential_exergy_loss_kW,
        )

        return await self._produce(
            topic=self.TOPIC_ALERTS,
            key=event.alert_id,
            value=event.to_bytes(),
            headers={
                "event_type": EventType.ALERT.value,
                "alert_type": alert_type,
                "severity": severity.value,
            }
        )

    # =========================================================================
    # Metrics
    # =========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """Get producer metrics."""
        total = self._messages_sent + self._messages_failed
        return {
            "messages_sent": self._messages_sent,
            "messages_failed": self._messages_failed,
            "bytes_sent": self._bytes_sent,
            "success_rate": self._messages_sent / total if total > 0 else 1.0,
            "last_error": self._last_error,
            "is_started": self._started,
        }

    async def flush(self) -> None:
        """Flush pending messages."""
        if self._producer:
            # await self._producer.flush()
            pass
        logger.debug("Producer flushed")
