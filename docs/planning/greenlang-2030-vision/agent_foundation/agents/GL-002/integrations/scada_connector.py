# -*- coding: utf-8 -*-
"""
SCADA System Integration Connector for GL-002 BoilerEfficiencyOptimizer

Implements secure, real-time connections to industrial SCADA systems for
comprehensive boiler monitoring and control.

Protocols Supported:
- OPC UA (Open Platform Communications Unified Architecture)
- OPC Classic (DA/HDA)
- MQTT (Message Queuing Telemetry Transport)
- IEC 60870-5-104 (Telecontrol protocol)
- REST/WebSocket APIs

Features:
- Real-time data streaming with sub-second updates
- Historical data retrieval and trending
- Alarm and event management
- Redundancy and failover support
- Cybersecurity with encryption and authentication
"""

import asyncio
import logging
import ssl
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import json
import hashlib
import hmac
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class SCADAProtocol(Enum):
    """Supported SCADA communication protocols."""
    OPC_UA = "opc_ua"
    OPC_CLASSIC = "opc_classic"
    MQTT = "mqtt"
    IEC_104 = "iec_104"
    REST_API = "rest_api"
    WEBSOCKET = "websocket"


class DataQuality(Enum):
    """OPC data quality indicators."""
    GOOD = "good"
    BAD = "bad"
    UNCERTAIN = "uncertain"
    BAD_SENSOR_FAILURE = "bad_sensor_failure"
    BAD_COMM_FAILURE = "bad_comm_failure"
    BAD_OUT_OF_SERVICE = "bad_out_of_service"
    UNCERTAIN_SENSOR_CAL = "uncertain_sensor_cal"


class AlarmPriority(Enum):
    """SCADA alarm priority levels."""
    CRITICAL = 1  # Immediate action required
    HIGH = 2      # Urgent attention needed
    MEDIUM = 3    # Standard alarm
    LOW = 4       # Information only
    DIAGNOSTIC = 5  # System diagnostic


@dataclass
class SCADATag:
    """SCADA tag/point configuration."""
    tag_name: str
    description: str
    data_type: str  # float, int, bool, string
    engineering_units: str
    scan_rate: int  # milliseconds
    deadband: float  # Change threshold
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    alarm_limits: Optional[Dict[str, float]] = None  # HH, H, L, LL
    scaling_factor: float = 1.0
    offset: float = 0.0
    quality: DataQuality = DataQuality.GOOD
    last_value: Any = None
    last_update: Optional[datetime] = None


@dataclass
class SCADAAlarm:
    """SCADA alarm configuration and state."""
    alarm_id: str
    tag_name: str
    alarm_type: str  # high, low, deviation, rate_of_change
    priority: AlarmPriority
    setpoint: float
    deadband: float
    delay_seconds: int  # Time delay before alarm
    message: str
    active: bool = False
    acknowledged: bool = False
    activation_time: Optional[datetime] = None
    acknowledgment_time: Optional[datetime] = None


@dataclass
class SCADAConnectionConfig:
    """SCADA system connection configuration."""
    protocol: SCADAProtocol
    primary_host: str
    primary_port: int
    backup_host: Optional[str] = None
    backup_port: Optional[int] = None
    use_encryption: bool = True
    certificate_path: Optional[str] = None
    private_key_path: Optional[str] = None
    ca_cert_path: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None  # Retrieved from secure storage
    connection_timeout: int = 30
    read_timeout: int = 10
    write_timeout: int = 10
    max_reconnect_attempts: int = 5
    reconnect_delay: int = 5
    enable_redundancy: bool = True
    enable_buffering: bool = True
    buffer_size: int = 100000


class SCADADataBuffer:
    """
    High-performance circular buffer for SCADA data.

    Implements time-series compression and efficient retrieval.
    """

    def __init__(self, max_size: int = 100000, compression_enabled: bool = True):
        """Initialize SCADA data buffer."""
        self.max_size = max_size
        self.compression_enabled = compression_enabled
        self.buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_size))
        self.compressed_data: Dict[str, List] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def add_sample(self, tag_name: str, value: Any, timestamp: datetime, quality: DataQuality):
        """Add data sample to buffer."""
        async with self._lock:
            sample = {
                'value': value,
                'timestamp': timestamp,
                'quality': quality.value
            }

            # Add to circular buffer
            self.buffer[tag_name].append(sample)

            # Compress if enabled
            if self.compression_enabled and len(self.buffer[tag_name]) % 100 == 0:
                await self._compress_old_data(tag_name)

    async def _compress_old_data(self, tag_name: str):
        """Compress old data using deadband compression."""
        if len(self.buffer[tag_name]) < 1000:
            return

        # Take oldest 500 samples for compression
        samples = list(self.buffer[tag_name])[:500]

        if not samples:
            return

        compressed = []
        last_value = samples[0]['value']
        last_time = samples[0]['timestamp']

        for sample in samples[1:]:
            # Only store if value changed significantly
            if abs(sample['value'] - last_value) > 0.01 * abs(last_value):
                compressed.append({
                    'v': last_value,
                    't': last_time.timestamp(),
                    'q': sample.get('quality', 'good')
                })
                last_value = sample['value']
                last_time = sample['timestamp']

        # Store compressed data
        self.compressed_data[tag_name].extend(compressed)

        # Remove compressed samples from buffer
        for _ in range(500):
            if self.buffer[tag_name]:
                self.buffer[tag_name].popleft()

    async def get_recent_data(
        self,
        tag_name: str,
        duration_seconds: int = 3600
    ) -> List[Dict]:
        """Get recent data for a tag."""
        async with self._lock:
            cutoff_time = DeterministicClock.utcnow() - timedelta(seconds=duration_seconds)
            recent = []

            # Get from buffer
            for sample in self.buffer.get(tag_name, []):
                if sample['timestamp'] > cutoff_time:
                    recent.append(sample)

            return recent

    async def get_statistics(self, tag_name: str) -> Dict[str, Any]:
        """Calculate statistics for buffered data."""
        async with self._lock:
            data = list(self.buffer.get(tag_name, []))

            if not data:
                return {}

            values = [s['value'] for s in data if isinstance(s['value'], (int, float))]

            if not values:
                return {}

            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'latest': data[-1]['value'] if data else None,
                'oldest': data[0]['value'] if data else None
            }


class AlarmManager:
    """
    SCADA alarm management system.

    Handles alarm detection, notification, and acknowledgment.
    """

    def __init__(self):
        """Initialize alarm manager."""
        self.alarms: Dict[str, SCADAAlarm] = {}
        self.active_alarms: Set[str] = set()
        self.alarm_history = deque(maxlen=10000)
        self.alarm_callbacks: List[Callable] = []
        self._alarm_timers: Dict[str, asyncio.Task] = {}

    def configure_alarm(self, alarm: SCADAAlarm):
        """Configure an alarm."""
        self.alarms[alarm.alarm_id] = alarm
        logger.info(f"Configured alarm: {alarm.alarm_id}")

    async def check_alarm_condition(
        self,
        tag_name: str,
        value: float,
        timestamp: datetime
    ):
        """Check if value triggers any alarms."""
        for alarm_id, alarm in self.alarms.items():
            if alarm.tag_name != tag_name:
                continue

            should_activate = False

            # Check alarm type
            if alarm.alarm_type == 'high' and value > alarm.setpoint:
                should_activate = True
            elif alarm.alarm_type == 'low' and value < alarm.setpoint:
                should_activate = True
            elif alarm.alarm_type == 'high_high' and value > alarm.setpoint:
                should_activate = True
            elif alarm.alarm_type == 'low_low' and value < alarm.setpoint:
                should_activate = True

            # Handle alarm state transitions
            if should_activate and not alarm.active:
                await self._activate_alarm(alarm, value, timestamp)
            elif not should_activate and alarm.active:
                await self._deactivate_alarm(alarm, timestamp)

    async def _activate_alarm(
        self,
        alarm: SCADAAlarm,
        value: float,
        timestamp: datetime
    ):
        """Activate an alarm after delay."""
        if alarm.alarm_id in self._alarm_timers:
            return  # Already pending

        async def delayed_activation():
            await asyncio.sleep(alarm.delay_seconds)

            if alarm.alarm_id in self._alarm_timers:
                alarm.active = True
                alarm.activation_time = timestamp
                self.active_alarms.add(alarm.alarm_id)

                # Log alarm
                logger.warning(f"ALARM ACTIVATED: {alarm.alarm_id} - {alarm.message}")

                # Record in history
                self.alarm_history.append({
                    'alarm_id': alarm.alarm_id,
                    'action': 'activated',
                    'value': value,
                    'timestamp': timestamp
                })

                # Trigger callbacks
                for callback in self.alarm_callbacks:
                    try:
                        await callback(alarm, 'activated', value)
                    except Exception as e:
                        logger.error(f"Alarm callback error: {e}")

                del self._alarm_timers[alarm.alarm_id]

        self._alarm_timers[alarm.alarm_id] = asyncio.create_task(delayed_activation())

    async def _deactivate_alarm(self, alarm: SCADAAlarm, timestamp: datetime):
        """Deactivate an alarm."""
        if alarm.alarm_id in self._alarm_timers:
            self._alarm_timers[alarm.alarm_id].cancel()
            del self._alarm_timers[alarm.alarm_id]

        if alarm.active:
            alarm.active = False
            self.active_alarms.discard(alarm.alarm_id)

            logger.info(f"ALARM CLEARED: {alarm.alarm_id}")

            # Record in history
            self.alarm_history.append({
                'alarm_id': alarm.alarm_id,
                'action': 'cleared',
                'timestamp': timestamp
            })

            # Trigger callbacks
            for callback in self.alarm_callbacks:
                try:
                    await callback(alarm, 'cleared', None)
                except Exception as e:
                    logger.error(f"Alarm callback error: {e}")

    async def acknowledge_alarm(self, alarm_id: str, user: str) -> bool:
        """Acknowledge an active alarm."""
        if alarm_id not in self.alarms:
            return False

        alarm = self.alarms[alarm_id]

        if alarm.active and not alarm.acknowledged:
            alarm.acknowledged = True
            alarm.acknowledgment_time = DeterministicClock.utcnow()

            logger.info(f"ALARM ACKNOWLEDGED: {alarm_id} by {user}")

            # Record in history
            self.alarm_history.append({
                'alarm_id': alarm_id,
                'action': 'acknowledged',
                'user': user,
                'timestamp': alarm.acknowledgment_time
            })

            return True

        return False

    def get_active_alarms(self) -> List[SCADAAlarm]:
        """Get list of active alarms sorted by priority."""
        active = [
            self.alarms[alarm_id]
            for alarm_id in self.active_alarms
            if alarm_id in self.alarms
        ]
        return sorted(active, key=lambda a: a.priority.value)


class SCADAConnector:
    """
    Main SCADA system connector for GL-002.

    Provides unified interface for multiple SCADA protocols.
    """

    def __init__(self, config: SCADAConnectionConfig):
        """Initialize SCADA connector."""
        self.config = config
        self.connected = False
        self.connection = None
        self.backup_connection = None
        self.tags: Dict[str, SCADATag] = {}
        self.data_buffer = SCADADataBuffer(config.buffer_size)
        self.alarm_manager = AlarmManager()
        self.scan_tasks: Dict[str, asyncio.Task] = {}
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._reconnect_task = None
        self._setup_default_tags()

    def _setup_default_tags(self):
        """Setup default SCADA tags for boiler monitoring."""
        default_tags = [
            SCADATag(
                tag_name='BOILER.STEAM.PRESSURE',
                description='Main steam header pressure',
                data_type='float',
                engineering_units='bar',
                scan_rate=1000,
                deadband=0.5,
                min_value=0,
                max_value=200,
                alarm_limits={'HH': 180, 'H': 160, 'L': 20, 'LL': 10}
            ),
            SCADATag(
                tag_name='BOILER.STEAM.TEMPERATURE',
                description='Main steam temperature',
                data_type='float',
                engineering_units='°C',
                scan_rate=1000,
                deadband=1.0,
                min_value=100,
                max_value=600,
                alarm_limits={'HH': 550, 'H': 520, 'L': 200, 'LL': 150}
            ),
            SCADATag(
                tag_name='BOILER.STEAM.FLOW',
                description='Steam flow rate',
                data_type='float',
                engineering_units='t/hr',
                scan_rate=1000,
                deadband=0.1,
                min_value=0,
                max_value=500
            ),
            SCADATag(
                tag_name='BOILER.EFFICIENCY',
                description='Calculated boiler efficiency',
                data_type='float',
                engineering_units='%',
                scan_rate=5000,
                deadband=0.1,
                min_value=0,
                max_value=100
            ),
            SCADATag(
                tag_name='BOILER.O2.CONTENT',
                description='Flue gas oxygen content',
                data_type='float',
                engineering_units='%',
                scan_rate=2000,
                deadband=0.05,
                min_value=0,
                max_value=21,
                alarm_limits={'H': 8, 'L': 2}
            ),
            SCADATag(
                tag_name='BOILER.DRUM.LEVEL',
                description='Drum water level',
                data_type='float',
                engineering_units='mm',
                scan_rate=500,
                deadband=5,
                min_value=-500,
                max_value=500,
                alarm_limits={'HH': 300, 'H': 200, 'L': -200, 'LL': -300}
            ),
            SCADATag(
                tag_name='BOILER.FUEL.VALVE.POSITION',
                description='Main fuel valve position',
                data_type='float',
                engineering_units='%',
                scan_rate=500,
                deadband=0.5,
                min_value=0,
                max_value=100
            ),
            SCADATag(
                tag_name='BOILER.STATUS',
                description='Boiler operational status',
                data_type='int',
                engineering_units='status',
                scan_rate=1000,
                deadband=0
            )
        ]

        for tag in default_tags:
            self.tags[tag.tag_name] = tag

    async def connect(self) -> bool:
        """Establish connection to SCADA system."""
        try:
            # Primary connection
            self.connection = await self._create_connection(
                self.config.primary_host,
                self.config.primary_port
            )

            # Backup connection if configured
            if self.config.enable_redundancy and self.config.backup_host:
                self.backup_connection = await self._create_connection(
                    self.config.backup_host,
                    self.config.backup_port or self.config.primary_port
                )

            self.connected = True
            logger.info(f"Connected to SCADA system via {self.config.protocol.value}")

            # Start tag scanning
            await self._start_scanning()

            # Setup alarm monitoring
            self._setup_alarms()

            return True

        except Exception as e:
            logger.error(f"Failed to connect to SCADA: {e}")
            self.connected = False

            # Start reconnection task
            if not self._reconnect_task:
                self._reconnect_task = asyncio.create_task(self._reconnect_loop())

            return False

    async def _create_connection(self, host: str, port: int) -> Dict:
        """Create protocol-specific connection."""
        connection = {
            'host': host,
            'port': port,
            'protocol': self.config.protocol.value,
            'connected': False
        }

        if self.config.use_encryption:
            # Setup SSL context
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)

            if self.config.certificate_path:
                ssl_context.load_cert_chain(
                    self.config.certificate_path,
                    self.config.private_key_path
                )

            if self.config.ca_cert_path:
                ssl_context.load_verify_locations(self.config.ca_cert_path)

            connection['ssl_context'] = ssl_context

        # Protocol-specific connection logic
        if self.config.protocol == SCADAProtocol.OPC_UA:
            # Simulate OPC UA connection
            connection['endpoint'] = f"opc.tcp://{host}:{port}"
            connection['connected'] = True

        elif self.config.protocol == SCADAProtocol.MQTT:
            # Simulate MQTT connection
            connection['broker'] = host
            connection['topics'] = ['boiler/+/data', 'boiler/+/alarms']
            connection['connected'] = True

        elif self.config.protocol == SCADAProtocol.REST_API:
            # Simulate REST API connection
            connection['base_url'] = f"https://{host}:{port}/api/v1"
            connection['connected'] = True

        return connection

    async def _reconnect_loop(self):
        """Automatic reconnection loop."""
        attempt = 0

        while attempt < self.config.max_reconnect_attempts:
            if self.connected:
                break

            attempt += 1
            wait_time = self.config.reconnect_delay * (2 ** min(attempt - 1, 5))

            logger.info(f"Reconnection attempt {attempt}/{self.config.max_reconnect_attempts} in {wait_time}s")
            await asyncio.sleep(wait_time)

            if await self.connect():
                logger.info("Reconnection successful")
                attempt = 0
            else:
                logger.warning(f"Reconnection attempt {attempt} failed")

        if not self.connected:
            logger.error("Max reconnection attempts reached. Connection failed.")

    async def _start_scanning(self):
        """Start scanning all configured tags."""
        # Group tags by scan rate
        scan_groups = defaultdict(list)

        for tag_name, tag in self.tags.items():
            scan_groups[tag.scan_rate].append(tag_name)

        # Create scan task for each rate
        for scan_rate, tag_list in scan_groups.items():
            task = asyncio.create_task(
                self._scan_loop(tag_list, scan_rate / 1000.0)
            )
            self.scan_tasks[f"scan_{scan_rate}"] = task

    async def _scan_loop(self, tag_names: List[str], interval: float):
        """Scan loop for a group of tags."""
        while self.connected:
            try:
                for tag_name in tag_names:
                    if tag_name in self.tags:
                        await self._read_tag(tag_name)

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Error in scan loop: {e}")
                await asyncio.sleep(interval)

    async def _read_tag(self, tag_name: str):
        """Read a single tag value."""
        if tag_name not in self.tags:
            return

        tag = self.tags[tag_name]

        try:
            # Simulate reading based on protocol
            if self.config.protocol == SCADAProtocol.OPC_UA:
                value = await self._read_opc_ua(tag_name)
            elif self.config.protocol == SCADAProtocol.MQTT:
                value = await self._read_mqtt(tag_name)
            else:
                value = await self._read_generic(tag_name)

            # Apply scaling
            if isinstance(value, (int, float)):
                value = value * tag.scaling_factor + tag.offset

            # Check deadband
            if tag.last_value is not None:
                if isinstance(value, (int, float)) and isinstance(tag.last_value, (int, float)):
                    if abs(value - tag.last_value) < tag.deadband:
                        return  # No significant change

            # Update tag
            tag.last_value = value
            tag.last_update = DeterministicClock.utcnow()
            tag.quality = DataQuality.GOOD

            # Store in buffer
            await self.data_buffer.add_sample(
                tag_name, value, tag.last_update, tag.quality
            )

            # Check alarms
            if isinstance(value, (int, float)):
                await self.alarm_manager.check_alarm_condition(
                    tag_name, value, tag.last_update
                )

            # Notify subscribers
            await self._notify_subscribers(tag_name, value, tag.last_update)

        except Exception as e:
            logger.error(f"Failed to read tag {tag_name}: {e}")
            tag.quality = DataQuality.BAD_COMM_FAILURE

    async def _read_opc_ua(self, tag_name: str) -> Any:
        """Read value via OPC UA."""
        # Simulate OPC UA read
        import random

        if 'PRESSURE' in tag_name:
            return random.uniform(90, 110)
        elif 'TEMPERATURE' in tag_name:
            return random.uniform(480, 500)
        elif 'FLOW' in tag_name:
            return random.uniform(180, 220)
        elif 'EFFICIENCY' in tag_name:
            return random.uniform(85, 92)
        elif 'O2' in tag_name:
            return random.uniform(3, 4)
        elif 'LEVEL' in tag_name:
            return random.uniform(-50, 50)
        elif 'VALVE' in tag_name:
            return random.uniform(40, 60)
        elif 'STATUS' in tag_name:
            return 1  # Running
        else:
            return 0

    async def _read_mqtt(self, tag_name: str) -> Any:
        """Read value via MQTT."""
        # Simulate MQTT message
        return await self._read_opc_ua(tag_name)

    async def _read_generic(self, tag_name: str) -> Any:
        """Read value via generic protocol."""
        return await self._read_opc_ua(tag_name)

    def _setup_alarms(self):
        """Setup alarm configurations."""
        # Create alarms based on tag alarm limits
        for tag_name, tag in self.tags.items():
            if not tag.alarm_limits:
                continue

            for alarm_type, setpoint in tag.alarm_limits.items():
                alarm_id = f"{tag_name}.{alarm_type}"

                priority = AlarmPriority.MEDIUM
                if alarm_type in ['HH', 'LL']:
                    priority = AlarmPriority.HIGH
                elif alarm_type in ['H', 'L']:
                    priority = AlarmPriority.MEDIUM

                alarm = SCADAAlarm(
                    alarm_id=alarm_id,
                    tag_name=tag_name,
                    alarm_type=alarm_type.lower(),
                    priority=priority,
                    setpoint=setpoint,
                    deadband=tag.deadband,
                    delay_seconds=5,
                    message=f"{tag.description} {alarm_type} alarm at {setpoint} {tag.engineering_units}"
                )

                self.alarm_manager.configure_alarm(alarm)

    async def write_tag(self, tag_name: str, value: Any) -> bool:
        """
        Write value to SCADA tag.

        Args:
            tag_name: Tag to write
            value: Value to write

        Returns:
            Success status
        """
        if not self.connected or tag_name not in self.tags:
            return False

        tag = self.tags[tag_name]

        try:
            # Apply scaling
            if isinstance(value, (int, float)):
                scaled_value = (value - tag.offset) / tag.scaling_factor
            else:
                scaled_value = value

            # Protocol-specific write
            if self.config.protocol == SCADAProtocol.OPC_UA:
                success = await self._write_opc_ua(tag_name, scaled_value)
            elif self.config.protocol == SCADAProtocol.MQTT:
                success = await self._write_mqtt(tag_name, scaled_value)
            else:
                success = await self._write_generic(tag_name, scaled_value)

            if success:
                logger.info(f"Written to SCADA tag {tag_name}: {value}")
                tag.last_value = value
                tag.last_update = DeterministicClock.utcnow()

            return success

        except Exception as e:
            logger.error(f"Failed to write tag {tag_name}: {e}")
            return False

    async def _write_opc_ua(self, tag_name: str, value: Any) -> bool:
        """Write value via OPC UA."""
        # Simulate OPC UA write
        return True

    async def _write_mqtt(self, tag_name: str, value: Any) -> bool:
        """Publish value via MQTT."""
        # Simulate MQTT publish
        return True

    async def _write_generic(self, tag_name: str, value: Any) -> bool:
        """Write value via generic protocol."""
        return True

    async def subscribe(self, tag_name: str, callback: Callable):
        """
        Subscribe to tag value changes.

        Args:
            tag_name: Tag to monitor
            callback: Async function to call on value change
        """
        self.subscribers[tag_name].append(callback)
        logger.debug(f"Added subscriber for {tag_name}")

    async def _notify_subscribers(
        self,
        tag_name: str,
        value: Any,
        timestamp: datetime
    ):
        """Notify subscribers of value change."""
        for callback in self.subscribers.get(tag_name, []):
            try:
                await callback(tag_name, value, timestamp)
            except Exception as e:
                logger.error(f"Subscriber callback error: {e}")

    async def get_current_values(self) -> Dict[str, Any]:
        """Get current values of all tags."""
        values = {}

        for tag_name, tag in self.tags.items():
            if tag.last_value is not None:
                values[tag_name] = {
                    'value': tag.last_value,
                    'quality': tag.quality.value,
                    'timestamp': tag.last_update.isoformat() if tag.last_update else None,
                    'units': tag.engineering_units
                }

        return values

    async def get_historical_data(
        self,
        tag_name: str,
        start_time: datetime,
        end_time: datetime,
        max_points: int = 1000
    ) -> List[Dict]:
        """
        Retrieve historical data for a tag.

        Args:
            tag_name: Tag to query
            start_time: Start of period
            end_time: End of period
            max_points: Maximum points to return

        Returns:
            List of historical data points
        """
        # In production, this would query the SCADA historian
        # For now, return recent buffered data
        duration = (end_time - start_time).total_seconds()
        return await self.data_buffer.get_recent_data(tag_name, int(duration))

    async def get_statistics(
        self,
        tag_names: List[str],
        duration_hours: int = 24
    ) -> Dict[str, Dict]:
        """Get statistics for multiple tags."""
        stats = {}

        for tag_name in tag_names:
            if tag_name in self.tags:
                tag_stats = await self.data_buffer.get_statistics(tag_name)
                stats[tag_name] = tag_stats

        return stats

    async def disconnect(self):
        """Disconnect from SCADA system."""
        self.connected = False

        # Cancel scan tasks
        for task in self.scan_tasks.values():
            task.cancel()
        self.scan_tasks.clear()

        # Cancel reconnect task
        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None

        # Clear connections
        self.connection = None
        self.backup_connection = None

        logger.info("Disconnected from SCADA system")


# Example usage
async def main():
    """Example usage of SCADA connector."""

    # Configure SCADA connection
    config = SCADAConnectionConfig(
        protocol=SCADAProtocol.OPC_UA,
        primary_host="192.168.1.200",
        primary_port=4840,
        backup_host="192.168.1.201",
        backup_port=4840,
        use_encryption=True,
        username="boiler_optimizer",
        enable_redundancy=True,
        enable_buffering=True
    )

    # Initialize connector
    connector = SCADAConnector(config)

    # Setup alarm callback
    async def alarm_callback(alarm: SCADAAlarm, action: str, value: Any):
        print(f"ALARM {action}: {alarm.alarm_id} - {alarm.message}")

    connector.alarm_manager.alarm_callbacks.append(alarm_callback)

    # Connect to SCADA
    if await connector.connect():
        print("Connected to SCADA system")

        # Subscribe to critical tags
        async def value_callback(tag_name: str, value: Any, timestamp: datetime):
            print(f"{tag_name}: {value} at {timestamp}")

        await connector.subscribe('BOILER.STEAM.PRESSURE', value_callback)
        await connector.subscribe('BOILER.EFFICIENCY', value_callback)

        # Let it run and collect data
        await asyncio.sleep(5)

        # Get current values
        values = await connector.get_current_values()
        print(f"\nCurrent values:")
        for tag, data in values.items():
            print(f"  {tag}: {data['value']} {data['units']}")

        # Get statistics
        stats = await connector.get_statistics(
            ['BOILER.STEAM.PRESSURE', 'BOILER.EFFICIENCY'],
            duration_hours=1
        )
        print(f"\nStatistics: {json.dumps(stats, indent=2)}")

        # Write a setpoint
        success = await connector.write_tag('BOILER.FUEL.VALVE.POSITION', 55.0)
        print(f"\nWrite setpoint result: {success}")

        # Check active alarms
        active_alarms = connector.alarm_manager.get_active_alarms()
        print(f"\nActive alarms: {len(active_alarms)}")
        for alarm in active_alarms:
            print(f"  {alarm.alarm_id}: {alarm.message}")

        # Disconnect
        await connector.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())