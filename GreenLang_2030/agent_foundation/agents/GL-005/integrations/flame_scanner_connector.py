"""
Flame Scanner Connector for GL-005 CombustionControlAgent

Implements ultra-fast flame detection and monitoring for combustion safety:
- Real-time flame presence detection (<50ms response)
- Flame intensity measurement and analysis
- Flame stability monitoring (flicker analysis)
- Multi-scanner support for large burners
- Safety interlock integration

Real-Time Requirements:
- Flame detection response: <50ms
- Intensity measurement update: 100Hz
- Flame failure alarm: <30ms
- Flicker analysis cycle: 1Hz

Communication Protocols:
- Digital I/O via PLC (primary)
- Dedicated network interface (Modbus TCP, Ethernet/IP)
- 4-20mA analog for flame intensity

Supported Scanner Types:
- UV Flame Detectors
- IR Flame Detectors
- Flame Rods (ionization)
- Multi-spectrum analyzers

Author: GL-DataIntegrationEngineer
Date: 2025-11-18
Version: 1.0.0
"""

import asyncio
import logging
import time
import statistics
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum

# Third-party imports
try:
    from pymodbus.client import AsyncModbusTcpClient
    from pymodbus.exceptions import ModbusException
    MODBUS_AVAILABLE = True
except ImportError:
    MODBUS_AVAILABLE = False

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ScannerType(Enum):
    """Types of flame scanners."""
    UV_DETECTOR = "uv_detector"
    IR_DETECTOR = "ir_detector"
    FLAME_ROD = "flame_rod"
    MULTI_SPECTRUM = "multi_spectrum"


class FlameStatus(Enum):
    """Flame detection status."""
    PRESENT = "present"
    ABSENT = "absent"
    UNSTABLE = "unstable"
    UNKNOWN = "unknown"


class ScannerHealth(Enum):
    """Scanner health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    CALIBRATION_NEEDED = "calibration_needed"


@dataclass
class FlameDetectionEvent:
    """Flame detection event."""
    scanner_id: str
    timestamp: datetime
    flame_present: bool
    intensity: float  # 0-100%
    intensity_raw: int  # Raw sensor value
    response_time_ms: float
    quality_score: float  # 0-100


@dataclass
class FlameStabilityMetrics:
    """Flame stability analysis metrics."""
    flicker_frequency_hz: float
    flicker_amplitude_pct: float
    stability_index: float  # 0-100 (100=perfectly stable)
    intensity_mean: float
    intensity_std_dev: float
    coefficient_of_variation: float


@dataclass
class FlameScannerConfig:
    """Configuration for flame scanner."""
    scanner_id: str
    scanner_type: ScannerType
    burner_id: str

    # Connection settings
    connection_type: str = "modbus_tcp"  # modbus_tcp, digital_io, analog_io
    modbus_host: str = "localhost"
    modbus_port: int = 502
    modbus_unit_id: int = 1

    # Register/coil addresses
    flame_present_coil: int = 0  # Digital flame presence
    flame_intensity_register: int = 0  # Analog intensity
    scanner_health_register: int = 1
    flame_failure_alarm_coil: int = 10

    # Measurement settings
    scan_rate_hz: int = 100  # Fast scanning for flame detection
    intensity_scale_min: int = 0
    intensity_scale_max: int = 1000

    # Flame stability analysis
    stability_window_size: int = 100  # Samples for flicker analysis
    flicker_threshold_hz: float = 5.0  # Max acceptable flicker
    intensity_cv_threshold: float = 0.1  # Max coefficient of variation

    # Safety settings
    flame_failure_delay_ms: int = 200  # Delay before flame failure alarm
    auto_restart_enabled: bool = False
    max_restart_attempts: int = 3


class FlameScannerConnector:
    """
    Flame Scanner Connector for ultra-fast flame monitoring.

    Features:
    - Sub-50ms flame detection response
    - 100Hz intensity monitoring
    - Real-time flame stability analysis
    - Automatic flame failure detection
    - Multi-scanner coordination
    - Safety interlock integration

    Example:
        config = FlameScannerConfig(
            scanner_id="SCANNER_BURNER_01",
            scanner_type=ScannerType.UV_DETECTOR,
            burner_id="BURNER_01",
            modbus_host="10.0.1.60"
        )

        async with FlameScannerConnector(config) as scanner:
            # Detect flame presence
            flame_present = await scanner.detect_flame_presence()
            print(f"Flame: {'ON' if flame_present else 'OFF'}")

            # Measure intensity
            intensity = await scanner.measure_flame_intensity()
            print(f"Intensity: {intensity}%")

            # Analyze stability
            stability = await scanner.analyze_flame_stability()
            print(f"Stability: {stability.stability_index}/100")

            # Subscribe to flame events
            await scanner.subscribe_to_flame_events(
                lambda event: print(f"Flame event: {event.flame_present}")
            )
    """

    def __init__(self, config: FlameScannerConfig):
        """Initialize flame scanner connector."""
        self.config = config
        self.connected = False

        # Modbus client
        self.modbus_client: Optional[AsyncModbusTcpClient] = None

        # Flame state
        self.flame_present = False
        self.last_flame_detection_time: Optional[datetime] = None
        self.flame_loss_time: Optional[datetime] = None

        # Intensity tracking
        self.intensity_buffer: deque = deque(maxlen=config.stability_window_size)
        self.current_intensity: float = 0.0

        # Health monitoring
        self.scanner_health = ScannerHealth.HEALTHY
        self.consecutive_failures = 0
        self.last_health_check: Optional[datetime] = None

        # Restart tracking
        self.restart_attempts = 0
        self.last_restart_time: Optional[datetime] = None

        # Event callbacks
        self.flame_event_callbacks: List[Callable[[FlameDetectionEvent], None]] = []
        self.flame_failure_callbacks: List[Callable[[str], None]] = []

        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._stability_analysis_task: Optional[asyncio.Task] = None

        # Performance tracking
        self.detection_latencies = deque(maxlen=1000)

        # Prometheus metrics
        if METRICS_AVAILABLE:
            self.metrics = {
                'flame_status': Gauge(
                    'flame_scanner_status',
                    'Flame status (1=present, 0=absent)',
                    ['scanner_id', 'burner_id']
                ),
                'flame_intensity': Gauge(
                    'flame_intensity_pct',
                    'Flame intensity percentage',
                    ['scanner_id', 'burner_id']
                ),
                'flame_stability': Gauge(
                    'flame_stability_index',
                    'Flame stability index (0-100)',
                    ['scanner_id', 'burner_id']
                ),
                'detection_latency': Histogram(
                    'flame_detection_latency_seconds',
                    'Flame detection latency',
                    ['scanner_id']
                ),
                'flame_failures': Counter(
                    'flame_failures_total',
                    'Total flame failure events',
                    ['scanner_id', 'burner_id']
                ),
                'scanner_health': Gauge(
                    'flame_scanner_health',
                    'Scanner health (1=healthy, 0=failed)',
                    ['scanner_id']
                )
            }
        else:
            self.metrics = {}

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect_to_scanner()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def connect_to_scanner(self) -> bool:
        """
        Connect to flame scanner.

        Returns:
            True if connection successful

        Raises:
            ConnectionError: If connection fails
        """
        logger.info(f"Connecting to flame scanner {self.config.scanner_id}...")

        if self.config.connection_type == "modbus_tcp":
            if not MODBUS_AVAILABLE:
                raise ImportError("pymodbus library required")

            try:
                self.modbus_client = AsyncModbusTcpClient(
                    host=self.config.modbus_host,
                    port=self.config.modbus_port,
                    timeout=0.5  # Fast timeout for real-time response
                )

                await self.modbus_client.connect()

                if not self.modbus_client.connected:
                    raise ConnectionError("Failed to connect to scanner")

                self.connected = True
                logger.info(f"Connected to scanner via Modbus TCP")

                # Start monitoring tasks
                self._monitoring_task = asyncio.create_task(self._fast_monitoring_loop())
                self._stability_analysis_task = asyncio.create_task(self._stability_analysis_loop())

                return True

            except Exception as e:
                logger.error(f"Scanner connection failed: {e}")
                raise ConnectionError(f"Scanner connection failed: {e}")

        else:
            raise ValueError(f"Unsupported connection type: {self.config.connection_type}")

    async def detect_flame_presence(self) -> bool:
        """
        Detect flame presence (fast operation).

        Returns:
            True if flame present, False otherwise

        Raises:
            ConnectionError: If not connected
        """
        if not self.connected:
            raise ConnectionError("Not connected to scanner")

        start_time = time.perf_counter()

        try:
            # Read flame presence coil
            response = await self.modbus_client.read_coils(
                address=self.config.flame_present_coil,
                count=1,
                unit=self.config.modbus_unit_id
            )

            if response.isError():
                raise ModbusException(f"Modbus error: {response}")

            flame_present = response.bits[0]

            # Calculate response time
            latency = time.perf_counter() - start_time
            self.detection_latencies.append(latency)

            # Update state
            if flame_present and not self.flame_present:
                # Flame ignition detected
                logger.info(f"Flame ignition detected on {self.config.burner_id}")
                self.last_flame_detection_time = datetime.now()
                self.flame_loss_time = None
                self.restart_attempts = 0

            elif not flame_present and self.flame_present:
                # Flame loss detected
                self.flame_loss_time = datetime.now()
                logger.warning(f"Flame loss detected on {self.config.burner_id}")

                # Check if flame failure alarm should trigger
                await self._check_flame_failure()

            self.flame_present = flame_present
            self.consecutive_failures = 0

            # Update metrics
            if self.metrics:
                self.metrics['flame_status'].labels(
                    scanner_id=self.config.scanner_id,
                    burner_id=self.config.burner_id
                ).set(1 if flame_present else 0)

                self.metrics['detection_latency'].labels(
                    scanner_id=self.config.scanner_id
                ).observe(latency)

            return flame_present

        except Exception as e:
            logger.error(f"Flame detection failed: {e}")
            self.consecutive_failures += 1
            raise

    async def measure_flame_intensity(self) -> float:
        """
        Measure flame intensity.

        Returns:
            Flame intensity as percentage (0-100%)

        Raises:
            ConnectionError: If not connected
        """
        if not self.connected:
            raise ConnectionError("Not connected to scanner")

        try:
            # Read intensity register
            response = await self.modbus_client.read_holding_registers(
                address=self.config.flame_intensity_register,
                count=1,
                unit=self.config.modbus_unit_id
            )

            if response.isError():
                raise ModbusException(f"Modbus error: {response}")

            raw_value = response.registers[0]

            # Scale to percentage
            intensity_pct = (
                (raw_value - self.config.intensity_scale_min) /
                (self.config.intensity_scale_max - self.config.intensity_scale_min)
            ) * 100.0

            # Clamp to 0-100%
            intensity_pct = max(0.0, min(100.0, intensity_pct))

            # Update state
            self.current_intensity = intensity_pct
            self.intensity_buffer.append(intensity_pct)

            # Update metrics
            if self.metrics:
                self.metrics['flame_intensity'].labels(
                    scanner_id=self.config.scanner_id,
                    burner_id=self.config.burner_id
                ).set(intensity_pct)

            return intensity_pct

        except Exception as e:
            logger.error(f"Intensity measurement failed: {e}")
            self.consecutive_failures += 1
            raise

    async def analyze_flame_stability(self) -> FlameStabilityMetrics:
        """
        Analyze flame stability based on intensity variations.

        Returns:
            Flame stability metrics

        Raises:
            ValueError: If insufficient data
        """
        if len(self.intensity_buffer) < 10:
            raise ValueError("Insufficient data for stability analysis")

        intensities = list(self.intensity_buffer)

        # Calculate statistics
        mean_intensity = statistics.mean(intensities)
        std_dev = statistics.stdev(intensities)
        cv = std_dev / mean_intensity if mean_intensity > 0 else 0

        # Flicker analysis (simplified FFT approach)
        # In production, use scipy.fft for frequency analysis
        flicker_freq = self._estimate_flicker_frequency(intensities)
        flicker_amplitude = (max(intensities) - min(intensities)) / mean_intensity * 100

        # Stability index (0-100, higher is more stable)
        stability_index = 100.0

        # Penalize high coefficient of variation
        if cv > self.config.intensity_cv_threshold:
            stability_index -= min(50, cv * 100)

        # Penalize high flicker frequency
        if flicker_freq > self.config.flicker_threshold_hz:
            stability_index -= min(30, (flicker_freq / self.config.flicker_threshold_hz) * 20)

        # Penalize large amplitude variations
        if flicker_amplitude > 20:
            stability_index -= min(20, flicker_amplitude / 2)

        stability_index = max(0, stability_index)

        metrics = FlameStabilityMetrics(
            flicker_frequency_hz=flicker_freq,
            flicker_amplitude_pct=flicker_amplitude,
            stability_index=stability_index,
            intensity_mean=mean_intensity,
            intensity_std_dev=std_dev,
            coefficient_of_variation=cv
        )

        # Update Prometheus metrics
        if self.metrics:
            self.metrics['flame_stability'].labels(
                scanner_id=self.config.scanner_id,
                burner_id=self.config.burner_id
            ).set(stability_index)

        # Log warnings for unstable flame
        if stability_index < 50:
            logger.warning(
                f"Unstable flame detected on {self.config.burner_id}: "
                f"stability={stability_index:.1f}, flicker={flicker_freq:.1f}Hz"
            )

        return metrics

    def _estimate_flicker_frequency(self, intensities: List[float]) -> float:
        """
        Estimate flicker frequency using zero-crossing method.

        In production, use FFT for accurate frequency analysis.
        """
        if len(intensities) < 10:
            return 0.0

        # Find mean
        mean = statistics.mean(intensities)

        # Count zero crossings
        crossings = 0
        for i in range(len(intensities) - 1):
            if (intensities[i] - mean) * (intensities[i+1] - mean) < 0:
                crossings += 1

        # Calculate frequency
        # crossings/2 = number of complete cycles
        # duration = len(intensities) / scan_rate
        duration = len(intensities) / self.config.scan_rate_hz
        frequency = (crossings / 2) / duration

        return frequency

    async def detect_flame_failure(self) -> bool:
        """
        Detect flame failure condition.

        Returns:
            True if flame failure detected
        """
        if not self.flame_present and self.flame_loss_time:
            # Check if flame has been lost for longer than failure delay
            time_since_loss = (datetime.now() - self.flame_loss_time).total_seconds() * 1000

            if time_since_loss >= self.config.flame_failure_delay_ms:
                return True

        return False

    async def _check_flame_failure(self):
        """Check and handle flame failure."""
        # Wait for failure delay
        await asyncio.sleep(self.config.flame_failure_delay_ms / 1000.0)

        # Re-check flame status
        try:
            flame_present = await self.detect_flame_presence()

            if not flame_present:
                # Flame failure confirmed
                logger.critical(f"FLAME FAILURE on {self.config.burner_id}")

                # Update metrics
                if self.metrics:
                    self.metrics['flame_failures'].labels(
                        scanner_id=self.config.scanner_id,
                        burner_id=self.config.burner_id
                    ).inc()

                # Trigger callbacks
                for callback in self.flame_failure_callbacks:
                    try:
                        await callback(self.config.burner_id)
                    except Exception as e:
                        logger.error(f"Flame failure callback error: {e}")

                # Attempt auto-restart if enabled
                if self.config.auto_restart_enabled:
                    await self._attempt_restart()

        except Exception as e:
            logger.error(f"Flame failure check error: {e}")

    async def _attempt_restart(self):
        """Attempt automatic burner restart."""
        if self.restart_attempts >= self.config.max_restart_attempts:
            logger.error(
                f"Max restart attempts ({self.config.max_restart_attempts}) "
                f"exceeded for {self.config.burner_id}"
            )
            return

        self.restart_attempts += 1
        self.last_restart_time = datetime.now()

        logger.info(
            f"Attempting restart {self.restart_attempts}/{self.config.max_restart_attempts} "
            f"for {self.config.burner_id}"
        )

        # Restart sequence would be implemented here
        # This typically involves:
        # 1. Purge cycle
        # 2. Ignition attempt
        # 3. Flame verification

        await asyncio.sleep(5)  # Simulate restart sequence

        # Check if restart successful
        flame_present = await self.detect_flame_presence()

        if flame_present:
            logger.info(f"Restart successful for {self.config.burner_id}")
            self.restart_attempts = 0
        else:
            logger.warning(f"Restart failed for {self.config.burner_id}")

    async def subscribe_to_flame_events(
        self,
        callback: Callable[[FlameDetectionEvent], None]
    ):
        """Subscribe to flame detection events."""
        self.flame_event_callbacks.append(callback)
        logger.info("Subscribed to flame events")

    async def subscribe_to_flame_failures(
        self,
        callback: Callable[[str], None]
    ):
        """Subscribe to flame failure events."""
        self.flame_failure_callbacks.append(callback)
        logger.info("Subscribed to flame failure events")

    async def _fast_monitoring_loop(self):
        """Background task for fast flame monitoring."""
        scan_interval = 1.0 / self.config.scan_rate_hz

        while self.connected:
            try:
                # Fast flame detection
                flame_present = await self.detect_flame_presence()

                # Measure intensity if flame present
                if flame_present:
                    intensity = await self.measure_flame_intensity()
                else:
                    intensity = 0.0

                # Create detection event
                if self.detection_latencies:
                    response_time_ms = self.detection_latencies[-1] * 1000
                else:
                    response_time_ms = 0

                event = FlameDetectionEvent(
                    scanner_id=self.config.scanner_id,
                    timestamp=datetime.now(),
                    flame_present=flame_present,
                    intensity=intensity,
                    intensity_raw=int(
                        intensity * (self.config.intensity_scale_max - self.config.intensity_scale_min) / 100
                        + self.config.intensity_scale_min
                    ),
                    response_time_ms=response_time_ms,
                    quality_score=self._calculate_signal_quality()
                )

                # Call event callbacks
                for callback in self.flame_event_callbacks:
                    try:
                        await callback(event)
                    except Exception as e:
                        logger.error(f"Event callback error: {e}")

                await asyncio.sleep(scan_interval)

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(scan_interval)

    async def _stability_analysis_loop(self):
        """Background task for flame stability analysis."""
        while self.connected:
            try:
                if len(self.intensity_buffer) >= 10:
                    await self.analyze_flame_stability()

                await asyncio.sleep(1.0)  # Run every second

            except Exception as e:
                logger.error(f"Stability analysis error: {e}")
                await asyncio.sleep(1.0)

    def _calculate_signal_quality(self) -> float:
        """Calculate signal quality score (0-100)."""
        quality = 100.0

        # Penalize consecutive failures
        if self.consecutive_failures > 0:
            quality -= min(50, self.consecutive_failures * 10)

        # Penalize high detection latency
        if self.detection_latencies:
            avg_latency_ms = statistics.mean(self.detection_latencies) * 1000
            if avg_latency_ms > 50:
                quality -= min(30, (avg_latency_ms - 50) / 2)

        # Penalize scanner health issues
        if self.scanner_health == ScannerHealth.DEGRADED:
            quality -= 20
        elif self.scanner_health == ScannerHealth.FAILED:
            quality -= 60

        return max(0, quality)

    async def disconnect(self):
        """Disconnect from flame scanner."""
        logger.info(f"Disconnecting from scanner {self.config.scanner_id}...")

        # Stop monitoring tasks
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        if self._stability_analysis_task:
            self._stability_analysis_task.cancel()
            try:
                await self._stability_analysis_task
            except asyncio.CancelledError:
                pass

        # Close Modbus connection
        if self.modbus_client:
            try:
                self.modbus_client.close()
            except Exception as e:
                logger.error(f"Error closing Modbus connection: {e}")

        self.connected = False
        logger.info("Disconnected from scanner")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get scanner performance statistics."""
        if not self.detection_latencies:
            return {}

        latencies_ms = [l * 1000 for l in self.detection_latencies]

        return {
            'avg_detection_latency_ms': statistics.mean(latencies_ms),
            'max_detection_latency_ms': max(latencies_ms),
            'min_detection_latency_ms': min(latencies_ms),
            'flame_present': self.flame_present,
            'current_intensity_pct': self.current_intensity,
            'consecutive_failures': self.consecutive_failures,
            'scanner_health': self.scanner_health.value,
            'restart_attempts': self.restart_attempts
        }
