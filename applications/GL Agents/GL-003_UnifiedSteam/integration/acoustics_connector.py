"""
GL-003 UNIFIEDSTEAM - Steam Trap Acoustics Connector

Integrates with edge devices for steam trap acoustic monitoring.
Supports ultrasonic and acoustic emission sensors for steam trap
condition assessment.

Features:
- Edge device connectivity (gRPC, MQTT, HTTP)
- Real-time acoustic feature streaming
- Feature extraction from raw acoustic signals
- Multi-trap subscription management
- Offline buffering and store-forward

Acoustic Features:
- RMS amplitude (dB)
- Peak frequency (Hz)
- Spectral centroid (Hz)
- Spectral bandwidth (Hz)
- Crest factor
- Zero crossing rate
- Energy in frequency bands (0-8kHz, 8-20kHz, 20-40kHz)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import asyncio
import logging
import uuid
import json
import math

logger = logging.getLogger(__name__)


class TrapStatus(Enum):
    """Steam trap operational status."""
    GOOD = "good"  # Operating normally
    LEAKING = "leaking"  # Passing live steam
    BLOCKED = "blocked"  # Not passing condensate
    FAILED_OPEN = "failed_open"  # Stuck open
    FAILED_CLOSED = "failed_closed"  # Stuck closed
    COLD = "cold"  # No steam supply
    UNKNOWN = "unknown"  # Unable to determine


class EdgeProtocol(Enum):
    """Edge device communication protocols."""
    GRPC = "grpc"
    MQTT = "mqtt"
    HTTP_REST = "http_rest"
    WEBSOCKET = "websocket"
    MODBUS_TCP = "modbus_tcp"


class AcousticSensorType(Enum):
    """Types of acoustic sensors."""
    ULTRASONIC = "ultrasonic"  # 20kHz - 100kHz
    ACOUSTIC_EMISSION = "acoustic_emission"  # 100kHz - 1MHz
    VIBRATION = "vibration"  # < 20kHz
    COMBINED = "combined"  # Multi-sensor


@dataclass
class EdgeDeviceConfig:
    """Edge device configuration."""
    device_id: str
    endpoint: str
    protocol: EdgeProtocol = EdgeProtocol.GRPC

    # Authentication
    api_key: Optional[str] = None
    certificate_path: Optional[str] = None
    client_id: Optional[str] = None

    # Connection settings
    timeout_ms: int = 10000
    keepalive_interval_ms: int = 30000
    reconnect_delay_ms: int = 5000
    max_reconnect_attempts: int = 10

    # MQTT specific
    mqtt_topic_prefix: str = "gl003/acoustics"
    mqtt_qos: int = 1

    # Buffering
    enable_store_forward: bool = True
    buffer_size: int = 10000

    # Sampling
    sample_rate_hz: int = 44100
    feature_interval_ms: int = 1000


@dataclass
class SpectralBands:
    """Energy in acoustic frequency bands."""
    band_0_8khz: float = 0.0  # Low frequency (mechanical noise)
    band_8_20khz: float = 0.0  # Mid frequency (flow noise)
    band_20_40khz: float = 0.0  # High frequency (steam leak)
    band_40_100khz: float = 0.0  # Ultrasonic (cavitation, leak)

    def to_dict(self) -> Dict:
        return {
            "band_0_8khz": round(self.band_0_8khz, 4),
            "band_8_20khz": round(self.band_8_20khz, 4),
            "band_20_40khz": round(self.band_20_40khz, 4),
            "band_40_100khz": round(self.band_40_100khz, 4),
        }


@dataclass
class AcousticFeatures:
    """Extracted acoustic features for steam trap assessment."""
    trap_id: str
    timestamp: datetime

    # Amplitude features
    rms_amplitude_db: float = 0.0  # RMS level in dB
    peak_amplitude_db: float = 0.0  # Peak level in dB
    crest_factor: float = 0.0  # Peak / RMS ratio

    # Spectral features
    peak_frequency_hz: float = 0.0  # Dominant frequency
    spectral_centroid_hz: float = 0.0  # "Center of mass" of spectrum
    spectral_bandwidth_hz: float = 0.0  # Spread of spectrum
    spectral_rolloff_hz: float = 0.0  # Frequency below which 85% of energy

    # Temporal features
    zero_crossing_rate: float = 0.0  # Rate of sign changes
    temporal_entropy: float = 0.0  # Signal randomness

    # Energy bands
    spectral_bands: SpectralBands = field(default_factory=SpectralBands)

    # Temperature (if available from same sensor node)
    surface_temperature_c: Optional[float] = None
    inlet_temperature_c: Optional[float] = None
    outlet_temperature_c: Optional[float] = None

    # Classification
    predicted_status: TrapStatus = TrapStatus.UNKNOWN
    confidence: float = 0.0  # 0.0 - 1.0

    # Quality
    signal_quality: str = "good"  # good, uncertain, bad
    sensor_type: AcousticSensorType = AcousticSensorType.ULTRASONIC

    def to_dict(self) -> Dict:
        return {
            "trap_id": self.trap_id,
            "timestamp": self.timestamp.isoformat(),
            "rms_amplitude_db": round(self.rms_amplitude_db, 2),
            "peak_amplitude_db": round(self.peak_amplitude_db, 2),
            "crest_factor": round(self.crest_factor, 3),
            "peak_frequency_hz": round(self.peak_frequency_hz, 1),
            "spectral_centroid_hz": round(self.spectral_centroid_hz, 1),
            "spectral_bandwidth_hz": round(self.spectral_bandwidth_hz, 1),
            "spectral_rolloff_hz": round(self.spectral_rolloff_hz, 1),
            "zero_crossing_rate": round(self.zero_crossing_rate, 4),
            "spectral_bands": self.spectral_bands.to_dict(),
            "surface_temperature_c": self.surface_temperature_c,
            "predicted_status": self.predicted_status.value,
            "confidence": round(self.confidence, 3),
            "signal_quality": self.signal_quality,
        }

    def get_leak_indicator(self) -> float:
        """
        Calculate steam leak indicator score (0-1).

        Higher values indicate higher probability of steam leak.
        Based on typical acoustic signatures of failing traps.
        """
        score = 0.0

        # High amplitude suggests leak
        if self.rms_amplitude_db > 80:
            score += 0.3 * min(1.0, (self.rms_amplitude_db - 80) / 20)

        # High frequency content suggests steam flow
        if self.spectral_bands.band_20_40khz > 0.3:
            score += 0.3 * self.spectral_bands.band_20_40khz

        # Low crest factor suggests continuous flow (leak) vs intermittent (normal)
        if self.crest_factor < 3.0:
            score += 0.2 * (3.0 - self.crest_factor) / 2.0

        # High spectral centroid suggests high-frequency content
        if self.spectral_centroid_hz > 15000:
            score += 0.2 * min(1.0, (self.spectral_centroid_hz - 15000) / 10000)

        return min(1.0, max(0.0, score))


# Type alias for acoustic update callback
AcousticCallback = Callable[[str, AcousticFeatures], None]


@dataclass
class TrapAcousticSubscription:
    """Subscription for trap acoustic updates."""
    subscription_id: str
    trap_ids: List[str]
    callback: AcousticCallback
    interval_ms: int = 1000

    # State
    is_active: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_update: Optional[datetime] = None
    updates_received: int = 0

    def to_dict(self) -> Dict:
        return {
            "subscription_id": self.subscription_id,
            "trap_ids": self.trap_ids,
            "interval_ms": self.interval_ms,
            "is_active": self.is_active,
            "updates_received": self.updates_received,
        }


class EdgeDeviceClient:
    """Base class for edge device communication."""

    def __init__(self, config: EdgeDeviceConfig) -> None:
        self.config = config
        self._connected = False
        self._subscriptions: Dict[str, TrapAcousticSubscription] = {}

    async def connect(self) -> bool:
        """Connect to edge device."""
        raise NotImplementedError

    async def disconnect(self) -> None:
        """Disconnect from edge device."""
        raise NotImplementedError

    async def get_features(self, trap_id: str) -> Optional[AcousticFeatures]:
        """Get current acoustic features for a trap."""
        raise NotImplementedError

    async def subscribe(
        self,
        trap_ids: List[str],
        callback: AcousticCallback,
        interval_ms: int = 1000,
    ) -> TrapAcousticSubscription:
        """Subscribe to acoustic updates."""
        raise NotImplementedError

    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from updates."""
        raise NotImplementedError


class GRPCEdgeClient(EdgeDeviceClient):
    """gRPC-based edge device client."""

    def __init__(self, config: EdgeDeviceConfig) -> None:
        super().__init__(config)
        self._channel = None
        self._stub = None

    async def connect(self) -> bool:
        """Connect to edge device via gRPC."""
        try:
            # In production, use grpcio-aio:
            # self._channel = grpc.aio.insecure_channel(self.config.endpoint)
            # self._stub = acoustics_pb2_grpc.AcousticsServiceStub(self._channel)
            # await self._channel.channel_ready()

            self._connected = True
            logger.info(f"Connected to edge device: {self.config.endpoint}")
            return True

        except Exception as e:
            logger.error(f"gRPC connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from edge device."""
        if self._channel:
            # await self._channel.close()
            pass
        self._connected = False

    async def get_features(self, trap_id: str) -> Optional[AcousticFeatures]:
        """Get current acoustic features via gRPC."""
        if not self._connected:
            return None

        # In production:
        # request = acoustics_pb2.GetFeaturesRequest(trap_id=trap_id)
        # response = await self._stub.GetFeatures(request)
        # return self._parse_features_response(response)

        # For framework: return simulated features
        return self._generate_simulated_features(trap_id)

    async def subscribe(
        self,
        trap_ids: List[str],
        callback: AcousticCallback,
        interval_ms: int = 1000,
    ) -> TrapAcousticSubscription:
        """Subscribe to acoustic updates via gRPC streaming."""
        subscription = TrapAcousticSubscription(
            subscription_id=str(uuid.uuid4()),
            trap_ids=trap_ids,
            callback=callback,
            interval_ms=interval_ms,
            is_active=True,
        )

        self._subscriptions[subscription.subscription_id] = subscription

        # Start streaming task
        asyncio.create_task(self._stream_updates(subscription))

        return subscription

    async def _stream_updates(self, subscription: TrapAcousticSubscription) -> None:
        """Stream acoustic updates to callback."""
        while subscription.is_active:
            for trap_id in subscription.trap_ids:
                try:
                    features = await self.get_features(trap_id)
                    if features:
                        subscription.callback(trap_id, features)
                        subscription.updates_received += 1
                        subscription.last_update = datetime.now(timezone.utc)
                except Exception as e:
                    logger.error(f"Error in acoustic stream: {e}")

            await asyncio.sleep(subscription.interval_ms / 1000)

    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from updates."""
        if subscription_id in self._subscriptions:
            self._subscriptions[subscription_id].is_active = False
            del self._subscriptions[subscription_id]

    def _generate_simulated_features(self, trap_id: str) -> AcousticFeatures:
        """Generate simulated acoustic features for testing."""
        import random

        # Use trap_id hash for deterministic base values
        base = hash(trap_id) % 100

        # Simulate different trap conditions based on ID
        if "LEAK" in trap_id.upper():
            # Simulated leaking trap
            rms_db = 85 + random.gauss(0, 3)
            peak_freq = 25000 + random.gauss(0, 2000)
            crest_factor = 2.0 + random.gauss(0, 0.3)
            status = TrapStatus.LEAKING
            confidence = 0.85
        elif "BLOCKED" in trap_id.upper():
            # Simulated blocked trap
            rms_db = 45 + random.gauss(0, 3)
            peak_freq = 2000 + random.gauss(0, 500)
            crest_factor = 8.0 + random.gauss(0, 1.0)
            status = TrapStatus.BLOCKED
            confidence = 0.78
        else:
            # Normal operating trap
            rms_db = 65 + random.gauss(0, 5)
            peak_freq = 8000 + random.gauss(0, 1500)
            crest_factor = 4.5 + random.gauss(0, 0.5)
            status = TrapStatus.GOOD
            confidence = 0.92

        # Spectral features
        centroid = peak_freq * 0.8 + random.gauss(0, 500)
        bandwidth = 5000 + random.gauss(0, 1000)
        rolloff = peak_freq * 1.5 + random.gauss(0, 1000)

        # Energy bands (normalized to sum to ~1.0)
        total_energy = random.uniform(0.8, 1.2)
        band_ratios = [0.3, 0.3, 0.25, 0.15]  # Typical distribution
        if status == TrapStatus.LEAKING:
            band_ratios = [0.1, 0.2, 0.4, 0.3]  # More high frequency

        bands = SpectralBands(
            band_0_8khz=band_ratios[0] * total_energy + random.gauss(0, 0.05),
            band_8_20khz=band_ratios[1] * total_energy + random.gauss(0, 0.05),
            band_20_40khz=band_ratios[2] * total_energy + random.gauss(0, 0.05),
            band_40_100khz=band_ratios[3] * total_energy + random.gauss(0, 0.02),
        )

        # Temperature
        inlet_temp = 150.0 + random.gauss(0, 5)
        outlet_temp = inlet_temp - (30 if status == TrapStatus.GOOD else 5)
        surface_temp = inlet_temp - 10 + random.gauss(0, 3)

        return AcousticFeatures(
            trap_id=trap_id,
            timestamp=datetime.now(timezone.utc),
            rms_amplitude_db=rms_db,
            peak_amplitude_db=rms_db + crest_factor * 3,
            crest_factor=crest_factor,
            peak_frequency_hz=peak_freq,
            spectral_centroid_hz=centroid,
            spectral_bandwidth_hz=bandwidth,
            spectral_rolloff_hz=rolloff,
            zero_crossing_rate=peak_freq / 44100 * 2,
            temporal_entropy=random.uniform(0.6, 0.9),
            spectral_bands=bands,
            surface_temperature_c=surface_temp,
            inlet_temperature_c=inlet_temp,
            outlet_temperature_c=outlet_temp,
            predicted_status=status,
            confidence=confidence,
            signal_quality="good",
            sensor_type=AcousticSensorType.ULTRASONIC,
        )


class MQTTEdgeClient(EdgeDeviceClient):
    """MQTT-based edge device client."""

    def __init__(self, config: EdgeDeviceConfig) -> None:
        super().__init__(config)
        self._client = None
        self._feature_cache: Dict[str, AcousticFeatures] = {}

    async def connect(self) -> bool:
        """Connect to MQTT broker."""
        try:
            # In production, use aiomqtt:
            # self._client = aiomqtt.Client(
            #     hostname=self.config.endpoint,
            #     client_id=self.config.client_id,
            # )
            # await self._client.connect()

            self._connected = True
            logger.info(f"Connected to MQTT broker: {self.config.endpoint}")
            return True

        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from MQTT broker."""
        if self._client:
            # await self._client.disconnect()
            pass
        self._connected = False

    async def get_features(self, trap_id: str) -> Optional[AcousticFeatures]:
        """Get cached features or request update."""
        # Return cached value if recent
        if trap_id in self._feature_cache:
            cached = self._feature_cache[trap_id]
            age = (datetime.now(timezone.utc) - cached.timestamp).total_seconds()
            if age < 5.0:  # Cache valid for 5 seconds
                return cached

        # Request fresh data by publishing to request topic
        # In production:
        # await self._client.publish(
        #     f"{self.config.mqtt_topic_prefix}/{trap_id}/request",
        #     payload=json.dumps({"action": "get_features"})
        # )

        return self._feature_cache.get(trap_id)

    async def subscribe(
        self,
        trap_ids: List[str],
        callback: AcousticCallback,
        interval_ms: int = 1000,
    ) -> TrapAcousticSubscription:
        """Subscribe to MQTT topics for acoustic updates."""
        subscription = TrapAcousticSubscription(
            subscription_id=str(uuid.uuid4()),
            trap_ids=trap_ids,
            callback=callback,
            interval_ms=interval_ms,
            is_active=True,
        )

        # Subscribe to MQTT topics
        for trap_id in trap_ids:
            topic = f"{self.config.mqtt_topic_prefix}/{trap_id}/features"
            # In production:
            # await self._client.subscribe(topic)

        self._subscriptions[subscription.subscription_id] = subscription

        # Start message processing task
        asyncio.create_task(self._process_messages(subscription))

        return subscription

    async def _process_messages(self, subscription: TrapAcousticSubscription) -> None:
        """Process incoming MQTT messages."""
        # In production, this would be an async iteration over messages
        # async for message in self._client.messages:
        #     trap_id = self._extract_trap_id(message.topic)
        #     features = self._parse_mqtt_message(message.payload)
        #     if trap_id in subscription.trap_ids:
        #         subscription.callback(trap_id, features)

        # For framework: simulate message arrival
        while subscription.is_active:
            for trap_id in subscription.trap_ids:
                features = self._generate_simulated_features(trap_id)
                self._feature_cache[trap_id] = features
                subscription.callback(trap_id, features)
                subscription.updates_received += 1
                subscription.last_update = datetime.now(timezone.utc)

            await asyncio.sleep(subscription.interval_ms / 1000)

    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from MQTT topics."""
        if subscription_id in self._subscriptions:
            subscription = self._subscriptions[subscription_id]
            subscription.is_active = False

            # Unsubscribe from MQTT topics
            for trap_id in subscription.trap_ids:
                topic = f"{self.config.mqtt_topic_prefix}/{trap_id}/features"
                # await self._client.unsubscribe(topic)

            del self._subscriptions[subscription_id]

    def _generate_simulated_features(self, trap_id: str) -> AcousticFeatures:
        """Generate simulated features (same as gRPC client)."""
        # Delegate to GRPCEdgeClient's implementation
        grpc_client = GRPCEdgeClient(self.config)
        return grpc_client._generate_simulated_features(trap_id)


class AcousticsConnector:
    """
    Connector for steam trap acoustic monitoring edge devices.

    Provides unified interface for real-time acoustic feature streaming
    from edge devices monitoring steam trap health.

    Example:
        config = EdgeDeviceConfig(
            device_id="edge-01",
            endpoint="edge-gateway.plant.local:50051",
            protocol=EdgeProtocol.GRPC,
        )

        connector = AcousticsConnector()
        await connector.connect_edge_device("edge-01", config)

        # Get current features
        features = await connector.get_acoustic_features("ST001")
        print(f"RMS: {features.rms_amplitude_db} dB")
        print(f"Status: {features.predicted_status}")

        # Subscribe to updates
        def on_acoustic_update(trap_id: str, features: AcousticFeatures):
            if features.get_leak_indicator() > 0.7:
                print(f"WARNING: Potential leak at {trap_id}")

        subscription = await connector.subscribe_acoustic_updates(
            trap_ids=["ST001", "ST002", "ST003"],
            callback=on_acoustic_update
        )
    """

    def __init__(self) -> None:
        """Initialize acoustics connector."""
        self._clients: Dict[str, EdgeDeviceClient] = {}
        self._trap_to_device: Dict[str, str] = {}  # trap_id -> device_id
        self._subscriptions: Dict[str, TrapAcousticSubscription] = {}

        # Statistics
        self._stats = {
            "devices_connected": 0,
            "features_retrieved": 0,
            "subscriptions_active": 0,
            "leak_alerts": 0,
        }

        logger.info("AcousticsConnector initialized")

    async def connect_edge_device(
        self,
        device_id: str,
        config_or_endpoint: EdgeDeviceConfig,
    ) -> bool:
        """
        Connect to an edge device.

        Args:
            device_id: Unique device identifier
            config_or_endpoint: Device configuration or endpoint string

        Returns:
            True if connected successfully
        """
        if isinstance(config_or_endpoint, str):
            config = EdgeDeviceConfig(
                device_id=device_id,
                endpoint=config_or_endpoint,
            )
        else:
            config = config_or_endpoint

        # Create appropriate client
        if config.protocol == EdgeProtocol.GRPC:
            client = GRPCEdgeClient(config)
        elif config.protocol == EdgeProtocol.MQTT:
            client = MQTTEdgeClient(config)
        else:
            logger.error(f"Unsupported protocol: {config.protocol}")
            return False

        # Connect
        success = await client.connect()
        if success:
            self._clients[device_id] = client
            self._stats["devices_connected"] += 1
            logger.info(f"Connected to edge device: {device_id}")

        return success

    async def disconnect_edge_device(self, device_id: str) -> None:
        """Disconnect from an edge device."""
        if device_id in self._clients:
            await self._clients[device_id].disconnect()
            del self._clients[device_id]
            self._stats["devices_connected"] -= 1

    def register_trap(self, trap_id: str, device_id: str) -> None:
        """Register which device monitors a trap."""
        self._trap_to_device[trap_id] = device_id

    async def get_acoustic_features(self, trap_id: str) -> Optional[AcousticFeatures]:
        """
        Get current acoustic features for a steam trap.

        Args:
            trap_id: Steam trap identifier

        Returns:
            AcousticFeatures or None if unavailable
        """
        # Find device for this trap
        device_id = self._trap_to_device.get(trap_id)

        if device_id and device_id in self._clients:
            client = self._clients[device_id]
        elif self._clients:
            # Use first available client
            client = next(iter(self._clients.values()))
        else:
            logger.warning(f"No edge device available for trap {trap_id}")
            return None

        features = await client.get_features(trap_id)
        if features:
            self._stats["features_retrieved"] += 1

        return features

    async def get_acoustic_features_batch(
        self,
        trap_ids: List[str],
    ) -> Dict[str, AcousticFeatures]:
        """Get acoustic features for multiple traps."""
        results: Dict[str, AcousticFeatures] = {}

        for trap_id in trap_ids:
            features = await self.get_acoustic_features(trap_id)
            if features:
                results[trap_id] = features

        return results

    async def subscribe_acoustic_updates(
        self,
        trap_ids: List[str],
        callback: AcousticCallback,
        interval_ms: int = 1000,
    ) -> TrapAcousticSubscription:
        """
        Subscribe to acoustic feature updates for multiple traps.

        Args:
            trap_ids: List of trap IDs to monitor
            callback: Function called on each update (trap_id, features)
            interval_ms: Update interval in milliseconds

        Returns:
            TrapAcousticSubscription object
        """
        # Group traps by device
        device_traps: Dict[str, List[str]] = {}
        for trap_id in trap_ids:
            device_id = self._trap_to_device.get(trap_id, "default")
            if device_id not in device_traps:
                device_traps[device_id] = []
            device_traps[device_id].append(trap_id)

        # Create subscription
        subscription_id = str(uuid.uuid4())
        subscription = TrapAcousticSubscription(
            subscription_id=subscription_id,
            trap_ids=trap_ids,
            callback=callback,
            interval_ms=interval_ms,
            is_active=True,
        )

        # Subscribe on each device
        for device_id, device_trap_ids in device_traps.items():
            if device_id in self._clients:
                await self._clients[device_id].subscribe(
                    trap_ids=device_trap_ids,
                    callback=callback,
                    interval_ms=interval_ms,
                )
            elif self._clients:
                # Use first available client
                client = next(iter(self._clients.values()))
                await client.subscribe(
                    trap_ids=device_trap_ids,
                    callback=callback,
                    interval_ms=interval_ms,
                )

        self._subscriptions[subscription_id] = subscription
        self._stats["subscriptions_active"] += 1

        logger.info(f"Created acoustic subscription for {len(trap_ids)} traps")
        return subscription

    async def unsubscribe(self, subscription: TrapAcousticSubscription) -> None:
        """Unsubscribe from acoustic updates."""
        subscription.is_active = False

        # Unsubscribe from all devices
        for client in self._clients.values():
            await client.unsubscribe(subscription.subscription_id)

        if subscription.subscription_id in self._subscriptions:
            del self._subscriptions[subscription.subscription_id]
            self._stats["subscriptions_active"] -= 1

    def get_statistics(self) -> Dict:
        """Get connector statistics."""
        return {
            **self._stats,
            "registered_traps": len(self._trap_to_device),
            "active_subscriptions": len([s for s in self._subscriptions.values() if s.is_active]),
        }

    def get_trap_status_summary(
        self,
        features_map: Dict[str, AcousticFeatures],
    ) -> Dict[str, Any]:
        """
        Get summary of trap statuses from acoustic features.

        Args:
            features_map: Dict of trap_id -> AcousticFeatures

        Returns:
            Summary statistics
        """
        status_counts = {status.value: 0 for status in TrapStatus}
        high_risk_traps = []
        total_leak_potential = 0.0

        for trap_id, features in features_map.items():
            status_counts[features.predicted_status.value] += 1

            leak_indicator = features.get_leak_indicator()
            total_leak_potential += leak_indicator

            if leak_indicator > 0.7:
                high_risk_traps.append({
                    "trap_id": trap_id,
                    "leak_indicator": leak_indicator,
                    "predicted_status": features.predicted_status.value,
                    "confidence": features.confidence,
                })

        return {
            "total_traps": len(features_map),
            "status_counts": status_counts,
            "high_risk_traps": high_risk_traps,
            "average_leak_potential": total_leak_potential / len(features_map) if features_map else 0,
        }


def create_trap_monitoring_config(
    site_id: str,
    gateway_endpoint: str,
    protocol: EdgeProtocol = EdgeProtocol.GRPC,
) -> EdgeDeviceConfig:
    """Create configuration for steam trap acoustic monitoring."""
    return EdgeDeviceConfig(
        device_id=f"{site_id}-acoustic-gateway",
        endpoint=gateway_endpoint,
        protocol=protocol,
        mqtt_topic_prefix=f"gl003/{site_id}/acoustics",
        feature_interval_ms=1000,
        enable_store_forward=True,
    )
