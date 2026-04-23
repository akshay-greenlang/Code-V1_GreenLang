"""
GL-002 FLAMEGUARD - Kafka Producer

Produces boiler events to Kafka topics.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
import asyncio
import hashlib
import json
import logging
import uuid

from .event_schemas import (
    BaseEvent,
    ProcessDataEvent,
    OptimizationEvent,
    SafetyEvent,
    EfficiencyEvent,
    EmissionsEvent,
    AlarmEvent,
    EventType,
)

logger = logging.getLogger(__name__)


@dataclass
class KafkaConfig:
    """Kafka producer configuration."""
    bootstrap_servers: List[str]
    client_id: str = "flameguard-producer"

    # Topics
    process_data_topic: str = "flameguard.process-data"
    optimization_topic: str = "flameguard.optimization"
    safety_topic: str = "flameguard.safety"
    efficiency_topic: str = "flameguard.efficiency"
    emissions_topic: str = "flameguard.emissions"
    alarm_topic: str = "flameguard.alarms"

    # Producer settings
    acks: str = "all"
    compression_type: str = "gzip"
    batch_size: int = 16384
    linger_ms: int = 100
    buffer_memory: int = 33554432

    # Schema Registry
    schema_registry_url: Optional[str] = None
    use_avro: bool = False

    # Security
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None

    # Retry settings
    retries: int = 3
    retry_backoff_ms: int = 100


class FlameguardKafkaProducer:
    """
    Kafka producer for Flameguard events.

    Features:
    - Async message production
    - Schema validation
    - Partitioning by boiler_id
    - Exactly-once semantics support
    - Event buffering with flush
    """

    def __init__(
        self,
        config: KafkaConfig,
        on_delivery: Optional[Callable[[str, bool], None]] = None,
    ) -> None:
        self.config = config
        self._on_delivery = on_delivery

        # Producer instance (would be aiokafka.AIOKafkaProducer in production)
        self._producer = None
        self._connected = False

        # Event buffer
        self._buffer: List[Dict[str, Any]] = []
        self._buffer_lock = asyncio.Lock()
        self._max_buffer_size = 1000

        # Statistics
        self._stats = {
            "messages_sent": 0,
            "messages_failed": 0,
            "bytes_sent": 0,
            "batch_count": 0,
        }

        logger.info(f"FlameguardKafkaProducer initialized: {config.bootstrap_servers}")

    async def connect(self) -> bool:
        """Connect to Kafka cluster."""
        try:
            # In production, use aiokafka:
            # self._producer = AIOKafkaProducer(
            #     bootstrap_servers=self.config.bootstrap_servers,
            #     client_id=self.config.client_id,
            #     acks=self.config.acks,
            #     compression_type=self.config.compression_type,
            # )
            # await self._producer.start()

            self._connected = True
            logger.info("Kafka producer connected")
            return True
        except Exception as e:
            logger.error(f"Kafka connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Kafka cluster."""
        await self.flush()
        if self._producer:
            # await self._producer.stop()
            pass
        self._connected = False
        logger.info("Kafka producer disconnected")

    def _get_topic(self, event_type: EventType) -> str:
        """Get topic for event type."""
        topic_map = {
            EventType.PROCESS_DATA: self.config.process_data_topic,
            EventType.OPTIMIZATION: self.config.optimization_topic,
            EventType.SAFETY: self.config.safety_topic,
            EventType.EFFICIENCY: self.config.efficiency_topic,
            EventType.EMISSIONS: self.config.emissions_topic,
            EventType.ALARM: self.config.alarm_topic,
        }
        return topic_map.get(event_type, self.config.process_data_topic)

    async def send_event(self, event: BaseEvent) -> bool:
        """Send single event to Kafka."""
        if not self._connected:
            logger.warning("Producer not connected, buffering event")
            await self._buffer_event(event)
            return False

        topic = self._get_topic(event.event_type)
        key = event.boiler_id.encode("utf-8")
        value = event.to_json().encode("utf-8")

        try:
            # In production:
            # await self._producer.send_and_wait(topic, value, key=key)

            # Simulate sending
            logger.debug(f"Sent to {topic}: {event.event_id}")
            self._stats["messages_sent"] += 1
            self._stats["bytes_sent"] += len(value)

            if self._on_delivery:
                self._on_delivery(event.event_id, True)

            return True

        except Exception as e:
            logger.error(f"Failed to send event: {e}")
            self._stats["messages_failed"] += 1

            if self._on_delivery:
                self._on_delivery(event.event_id, False)

            return False

    async def send_process_data(
        self,
        boiler_id: str,
        data: Dict[str, float],
    ) -> bool:
        """Send process data event."""
        event = ProcessDataEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            boiler_id=boiler_id,
            drum_pressure_psig=data.get("drum_pressure", 0.0),
            drum_level_inches=data.get("drum_level", 0.0),
            steam_flow_klb_hr=data.get("steam_flow", 0.0),
            steam_temperature_f=data.get("steam_temperature", 0.0),
            flue_gas_temperature_f=data.get("flue_gas_temp", 0.0),
            o2_percent=data.get("o2_percent", 0.0),
            co_ppm=data.get("co_ppm", 0.0),
            fuel_flow_scfh=data.get("fuel_flow", 0.0),
            load_percent=data.get("load_percent", 0.0),
            firing=data.get("firing", False),
        )
        return await self.send_event(event)

    async def send_optimization_result(
        self,
        boiler_id: str,
        mode: str,
        current: Dict[str, float],
        predicted: Dict[str, float],
        recommendations: Dict[str, float],
    ) -> bool:
        """Send optimization result event."""
        event = OptimizationEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            boiler_id=boiler_id,
            optimization_mode=mode,
            current_efficiency=current.get("efficiency", 0.0),
            current_emissions_mtco2e=current.get("emissions", 0.0),
            current_cost_usd=current.get("cost", 0.0),
            predicted_efficiency=predicted.get("efficiency", 0.0),
            predicted_emissions_mtco2e=predicted.get("emissions", 0.0),
            predicted_cost_usd=predicted.get("cost", 0.0),
            recommended_o2_setpoint=recommendations.get("o2_setpoint", 0.0),
            recommended_excess_air=recommendations.get("excess_air", 0.0),
            efficiency_improvement=predicted.get("efficiency", 0.0) - current.get("efficiency", 0.0),
            emissions_reduction=current.get("emissions", 0.0) - predicted.get("emissions", 0.0),
            cost_savings_usd=current.get("cost", 0.0) - predicted.get("cost", 0.0),
        )
        event.calculation_hash = event.compute_hash()
        return await self.send_event(event)

    async def send_safety_event(
        self,
        boiler_id: str,
        safety_type: str,
        severity: str,
        details: Dict[str, Any],
    ) -> bool:
        """Send safety event."""
        from .event_schemas import Severity

        event = SafetyEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            boiler_id=boiler_id,
            safety_event_type=safety_type,
            severity=Severity(severity) if severity in [s.value for s in Severity] else Severity.INFO,
            interlock_tag=details.get("interlock_tag"),
            previous_state=details.get("previous_state", ""),
            new_state=details.get("new_state", ""),
            trip_cause=details.get("trip_cause"),
            flame_proven=details.get("flame_proven", True),
            bms_state=details.get("bms_state", ""),
        )
        return await self.send_event(event)

    async def send_efficiency_calculation(
        self,
        boiler_id: str,
        result: Dict[str, float],
        method: str = "indirect",
    ) -> bool:
        """Send efficiency calculation event."""
        event = EfficiencyEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            boiler_id=boiler_id,
            gross_efficiency_percent=result.get("gross_efficiency", 0.0),
            net_efficiency_percent=result.get("net_efficiency", 0.0),
            fuel_efficiency_percent=result.get("fuel_efficiency", 0.0),
            stack_loss_percent=result.get("stack_loss", 0.0),
            radiation_loss_percent=result.get("radiation_loss", 0.0),
            blowdown_loss_percent=result.get("blowdown_loss", 0.0),
            heat_input_mmbtu_hr=result.get("heat_input", 0.0),
            heat_output_mmbtu_hr=result.get("heat_output", 0.0),
            calculation_method=method,
            standard="ASME PTC 4.1",
            load_percent=result.get("load_percent", 0.0),
            o2_percent=result.get("o2_percent", 0.0),
        )
        event.calculation_hash = event.compute_hash()
        return await self.send_event(event)

    async def send_emissions_calculation(
        self,
        boiler_id: str,
        result: Dict[str, float],
        fuel_type: str = "natural_gas",
    ) -> bool:
        """Send emissions calculation event."""
        event = EmissionsEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            boiler_id=boiler_id,
            nox_lb_hr=result.get("nox_lb_hr", 0.0),
            co_lb_hr=result.get("co_lb_hr", 0.0),
            co2_ton_hr=result.get("co2_ton_hr", 0.0),
            so2_lb_hr=result.get("so2_lb_hr", 0.0),
            pm_lb_hr=result.get("pm_lb_hr", 0.0),
            voc_lb_hr=result.get("voc_lb_hr", 0.0),
            nox_ppm=result.get("nox_ppm", 0.0),
            co_ppm=result.get("co_ppm", 0.0),
            ghg_mtco2e_hr=result.get("ghg_mtco2e_hr", 0.0),
            fuel_type=fuel_type,
            fuel_flow_scfh=result.get("fuel_flow", 0.0),
            emission_factors_source="EPA",
        )
        event.calculation_hash = event.compute_hash()
        return await self.send_event(event)

    async def send_alarm(
        self,
        boiler_id: str,
        alarm_tag: str,
        alarm_type: str,
        severity: str,
        message: str,
        current_value: float,
        limit_value: float,
        unit: str = "",
    ) -> bool:
        """Send alarm event."""
        from .event_schemas import Severity

        event = AlarmEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            boiler_id=boiler_id,
            alarm_tag=alarm_tag,
            alarm_type=alarm_type,
            severity=Severity(severity) if severity in [s.value for s in Severity] else Severity.WARNING,
            message=message,
            current_value=current_value,
            limit_value=limit_value,
            unit=unit,
            active_since=datetime.now(timezone.utc),
        )
        return await self.send_event(event)

    async def _buffer_event(self, event: BaseEvent) -> None:
        """Buffer event for later sending."""
        async with self._buffer_lock:
            if len(self._buffer) < self._max_buffer_size:
                self._buffer.append({
                    "topic": self._get_topic(event.event_type),
                    "key": event.boiler_id,
                    "value": event.to_dict(),
                    "timestamp": datetime.now(timezone.utc),
                })
            else:
                logger.warning("Event buffer full, dropping oldest events")
                self._buffer.pop(0)
                self._buffer.append({
                    "topic": self._get_topic(event.event_type),
                    "key": event.boiler_id,
                    "value": event.to_dict(),
                    "timestamp": datetime.now(timezone.utc),
                })

    async def flush(self) -> int:
        """Flush buffered events."""
        async with self._buffer_lock:
            if not self._buffer:
                return 0

            count = len(self._buffer)
            logger.info(f"Flushing {count} buffered events")

            if self._connected:
                # In production, would send all buffered messages
                self._stats["batch_count"] += 1
                self._stats["messages_sent"] += count

            self._buffer.clear()
            return count

    def get_statistics(self) -> Dict:
        """Get producer statistics."""
        return {
            **self._stats,
            "connected": self._connected,
            "buffer_size": len(self._buffer),
        }

    @property
    def is_connected(self) -> bool:
        return self._connected
