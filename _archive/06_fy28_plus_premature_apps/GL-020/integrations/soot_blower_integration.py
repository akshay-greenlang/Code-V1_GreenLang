"""
GL-020 ECONOPULSE - Soot Blower Control Integration Module

Enterprise-grade soot blower control system integration providing:
- Blower status monitoring (running, idle, fault)
- Cleaning cycle history retrieval
- Cleaning cycle triggering with safety interlocks
- Zone-based cleaning control
- Media consumption tracking (steam/air)
- Cleaning effectiveness feedback loop

Thread-safe with circuit breaker pattern and safety interlock enforcement.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import asyncio
import logging
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class BlowerStatus(Enum):
    """Soot blower operational status."""
    IDLE = "idle"
    RUNNING = "running"
    ADVANCING = "advancing"
    RETRACTING = "retracting"
    PAUSED = "paused"
    FAULT = "fault"
    MAINTENANCE = "maintenance"
    DISABLED = "disabled"


class CleaningMediaType(Enum):
    """Type of cleaning media used by soot blower."""
    STEAM = "steam"
    COMPRESSED_AIR = "compressed_air"
    SONIC = "sonic"
    WATER = "water"


class InterlockType(Enum):
    """Types of safety interlocks."""
    STEAM_PRESSURE_LOW = "steam_pressure_low"
    STEAM_PRESSURE_HIGH = "steam_pressure_high"
    STEAM_TEMPERATURE_HIGH = "steam_temperature_high"
    BOILER_LOAD_LOW = "boiler_load_low"
    BLOWER_OVERTRAVEL = "blower_overtravel"
    BLOWER_STUCK = "blower_stuck"
    DRIVE_FAULT = "drive_fault"
    EMERGENCY_STOP = "emergency_stop"
    MAINTENANCE_LOCKOUT = "maintenance_lockout"
    SEQUENCE_TIMEOUT = "sequence_timeout"


class ZoneStatus(Enum):
    """Cleaning zone status."""
    CLEAN = "clean"
    DIRTY = "dirty"
    UNKNOWN = "unknown"
    CLEANING = "cleaning"
    BYPASSED = "bypassed"


class CommandStatus(Enum):
    """Status of issued commands."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SafetyInterlock:
    """Safety interlock definition and status."""
    interlock_type: InterlockType
    description: str
    is_active: bool = False
    trip_value: Optional[float] = None
    reset_value: Optional[float] = None
    current_value: Optional[float] = None
    last_trip_time: Optional[datetime] = None
    auto_reset: bool = False
    tag_name: str = ""

    def check_trip_condition(self, value: float) -> bool:
        """Check if value would trip this interlock."""
        if self.trip_value is None:
            return False

        if self.interlock_type in (
            InterlockType.STEAM_PRESSURE_LOW,
            InterlockType.BOILER_LOAD_LOW,
        ):
            return value < self.trip_value
        else:
            return value > self.trip_value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "interlock_type": self.interlock_type.value,
            "description": self.description,
            "is_active": self.is_active,
            "trip_value": self.trip_value,
            "reset_value": self.reset_value,
            "current_value": self.current_value,
            "last_trip_time": (
                self.last_trip_time.isoformat() if self.last_trip_time else None
            ),
            "auto_reset": self.auto_reset,
        }


@dataclass
class CleaningZone:
    """Economizer cleaning zone definition."""
    zone_id: str
    name: str
    description: str
    blower_ids: List[str]
    priority: int = 1  # 1 = highest priority
    status: ZoneStatus = ZoneStatus.UNKNOWN
    fouling_factor: float = 0.0  # 0.0 = clean, 1.0 = fully fouled
    last_cleaned: Optional[datetime] = None
    min_cleaning_interval_hours: float = 4.0
    enabled: bool = True

    def needs_cleaning(self) -> bool:
        """Check if zone needs cleaning based on fouling and time."""
        if not self.enabled:
            return False

        if self.fouling_factor > 0.7:
            return True

        if self.last_cleaned:
            hours_since_clean = (
                datetime.now() - self.last_cleaned
            ).total_seconds() / 3600

            if hours_since_clean < self.min_cleaning_interval_hours:
                return False

        return self.fouling_factor > 0.3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "description": self.description,
            "blower_ids": self.blower_ids,
            "priority": self.priority,
            "status": self.status.value,
            "fouling_factor": self.fouling_factor,
            "last_cleaned": (
                self.last_cleaned.isoformat() if self.last_cleaned else None
            ),
            "enabled": self.enabled,
        }


@dataclass
class CleaningCycle:
    """Record of a cleaning cycle execution."""
    cycle_id: str
    zone_id: str
    blower_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: CommandStatus = CommandStatus.PENDING
    media_type: CleaningMediaType = CleaningMediaType.STEAM
    media_consumed_kg: float = 0.0
    steam_pressure_avg_kpa: float = 0.0
    steam_temperature_avg_c: float = 0.0
    duration_seconds: float = 0.0
    dp_before_kpa: float = 0.0
    dp_after_kpa: float = 0.0
    effectiveness_score: float = 0.0  # 0-100
    fault_message: Optional[str] = None
    operator_id: str = ""
    trigger_type: str = "automatic"  # automatic, manual, scheduled

    def calculate_effectiveness(self) -> float:
        """
        Calculate cleaning effectiveness based on differential pressure change.

        Returns:
            Effectiveness score from 0 to 100.
        """
        if self.dp_before_kpa <= 0:
            return 0.0

        dp_reduction = (self.dp_before_kpa - self.dp_after_kpa) / self.dp_before_kpa
        effectiveness = min(100.0, max(0.0, dp_reduction * 100))

        self.effectiveness_score = effectiveness
        return effectiveness

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cycle_id": self.cycle_id,
            "zone_id": self.zone_id,
            "blower_id": self.blower_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status.value,
            "media_type": self.media_type.value,
            "media_consumed_kg": self.media_consumed_kg,
            "steam_pressure_avg_kpa": self.steam_pressure_avg_kpa,
            "steam_temperature_avg_c": self.steam_temperature_avg_c,
            "duration_seconds": self.duration_seconds,
            "dp_before_kpa": self.dp_before_kpa,
            "dp_after_kpa": self.dp_after_kpa,
            "effectiveness_score": self.effectiveness_score,
            "fault_message": self.fault_message,
            "trigger_type": self.trigger_type,
        }


@dataclass
class MediaConsumption:
    """Media consumption tracking data."""
    media_type: CleaningMediaType
    total_consumed_kg: float = 0.0
    consumption_rate_kg_hr: float = 0.0
    cost_per_kg: float = 0.0
    total_cost: float = 0.0
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    cycle_count: int = 0

    def calculate_totals(self) -> None:
        """Calculate total cost."""
        self.total_cost = self.total_consumed_kg * self.cost_per_kg

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "media_type": self.media_type.value,
            "total_consumed_kg": self.total_consumed_kg,
            "consumption_rate_kg_hr": self.consumption_rate_kg_hr,
            "cost_per_kg": self.cost_per_kg,
            "total_cost": self.total_cost,
            "period_start": (
                self.period_start.isoformat() if self.period_start else None
            ),
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "cycle_count": self.cycle_count,
        }


@dataclass
class CleaningEffectiveness:
    """Cleaning effectiveness analysis data."""
    zone_id: str
    analysis_period_hours: float = 24.0
    cycles_analyzed: int = 0
    avg_effectiveness_score: float = 0.0
    min_effectiveness_score: float = 0.0
    max_effectiveness_score: float = 0.0
    avg_dp_reduction_percent: float = 0.0
    avg_media_consumption_kg: float = 0.0
    recommended_cleaning_interval_hours: float = 0.0
    fouling_rate_kpa_per_hour: float = 0.0
    trend: str = "stable"  # improving, degrading, stable

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "zone_id": self.zone_id,
            "analysis_period_hours": self.analysis_period_hours,
            "cycles_analyzed": self.cycles_analyzed,
            "avg_effectiveness_score": self.avg_effectiveness_score,
            "min_effectiveness_score": self.min_effectiveness_score,
            "max_effectiveness_score": self.max_effectiveness_score,
            "avg_dp_reduction_percent": self.avg_dp_reduction_percent,
            "avg_media_consumption_kg": self.avg_media_consumption_kg,
            "recommended_cleaning_interval_hours": self.recommended_cleaning_interval_hours,
            "fouling_rate_kpa_per_hour": self.fouling_rate_kpa_per_hour,
            "trend": self.trend,
        }


@dataclass
class SootBlowerConfig:
    """Soot blower system configuration."""
    controller_id: str
    controller_name: str
    protocol: str  # "modbus_tcp", "opc_ua", "proprietary"
    host: str
    port: int
    media_type: CleaningMediaType = CleaningMediaType.STEAM
    steam_pressure_setpoint_kpa: float = 1200.0
    steam_temp_max_c: float = 350.0
    cycle_timeout_seconds: float = 300.0
    retract_timeout_seconds: float = 60.0
    min_boiler_load_percent: float = 40.0
    max_concurrent_blowers: int = 2


@dataclass
class BlowerCommand:
    """Command to soot blower system."""
    command_id: str
    command_type: str  # "start_cycle", "stop_cycle", "enable", "disable", "reset"
    blower_id: Optional[str] = None
    zone_id: Optional[str] = None
    status: CommandStatus = CommandStatus.PENDING
    issued_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    operator_id: str = ""
    reason: str = ""
    error_message: Optional[str] = None


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitBreakerState:
        with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                if self._last_failure_time:
                    elapsed = (datetime.now() - self._last_failure_time).total_seconds()
                    if elapsed >= self.recovery_timeout:
                        self._state = CircuitBreakerState.HALF_OPEN
            return self._state

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitBreakerState.OPEN

    def is_available(self) -> bool:
        return self.state != CircuitBreakerState.OPEN


# =============================================================================
# Soot Blower Controller
# =============================================================================

class SootBlowerController:
    """
    Enterprise soot blower control system interface.

    Provides:
    - Blower status monitoring
    - Cleaning cycle management with safety interlocks
    - Zone-based cleaning control
    - Media consumption tracking
    - Cleaning effectiveness analysis

    All operations enforce safety interlocks before execution.
    """

    def __init__(
        self,
        config: SootBlowerConfig,
        scada_client=None,
        vault_client=None,
    ):
        """
        Initialize soot blower controller.

        Args:
            config: Soot blower system configuration
            scada_client: Optional SCADA client for tag communication
            vault_client: Optional vault client for credentials
        """
        self.config = config
        self.scada_client = scada_client
        self.vault_client = vault_client

        self._connected = False
        self._lock = threading.RLock()

        # Zones and blowers
        self._zones: Dict[str, CleaningZone] = {}
        self._blower_status: Dict[str, BlowerStatus] = {}

        # Interlocks
        self._interlocks: Dict[InterlockType, SafetyInterlock] = {}
        self._init_default_interlocks()

        # Cycle tracking
        self._active_cycles: Dict[str, CleaningCycle] = {}
        self._cycle_history: List[CleaningCycle] = []
        self._max_history_size = 10000

        # Media consumption
        self._media_consumption = MediaConsumption(
            media_type=config.media_type,
            period_start=datetime.now(),
        )

        # Commands
        self._pending_commands: Dict[str, BlowerCommand] = {}

        # Circuit breaker
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=60.0,
        )

        # Background tasks
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None

        # Callbacks for status changes
        self._status_callbacks: List[Callable[[str, BlowerStatus], None]] = []
        self._cycle_callbacks: List[Callable[[CleaningCycle], None]] = []

        logger.info(
            f"Initialized SootBlowerController {config.controller_id} "
            f"at {config.host}:{config.port}"
        )

    def _init_default_interlocks(self) -> None:
        """Initialize default safety interlocks."""
        self._interlocks = {
            InterlockType.STEAM_PRESSURE_LOW: SafetyInterlock(
                interlock_type=InterlockType.STEAM_PRESSURE_LOW,
                description="Steam pressure below minimum for effective cleaning",
                trip_value=800.0,  # kPa
                reset_value=900.0,
                auto_reset=True,
                tag_name="ECON.SB.STEAM_PRESSURE",
            ),
            InterlockType.STEAM_PRESSURE_HIGH: SafetyInterlock(
                interlock_type=InterlockType.STEAM_PRESSURE_HIGH,
                description="Steam pressure exceeds safe limit",
                trip_value=1500.0,  # kPa
                reset_value=1400.0,
                auto_reset=True,
                tag_name="ECON.SB.STEAM_PRESSURE",
            ),
            InterlockType.STEAM_TEMPERATURE_HIGH: SafetyInterlock(
                interlock_type=InterlockType.STEAM_TEMPERATURE_HIGH,
                description="Steam temperature exceeds equipment limit",
                trip_value=400.0,  # Celsius
                reset_value=380.0,
                auto_reset=True,
                tag_name="ECON.SB.STEAM_TEMP",
            ),
            InterlockType.BOILER_LOAD_LOW: SafetyInterlock(
                interlock_type=InterlockType.BOILER_LOAD_LOW,
                description="Boiler load too low for safe blowing",
                trip_value=self.config.min_boiler_load_percent,
                reset_value=self.config.min_boiler_load_percent + 5.0,
                auto_reset=True,
                tag_name="BOILER.LOAD.PERCENT",
            ),
            InterlockType.BLOWER_OVERTRAVEL: SafetyInterlock(
                interlock_type=InterlockType.BLOWER_OVERTRAVEL,
                description="Blower lance position exceeds travel limit",
                auto_reset=False,
            ),
            InterlockType.BLOWER_STUCK: SafetyInterlock(
                interlock_type=InterlockType.BLOWER_STUCK,
                description="Blower lance stuck or drive fault",
                auto_reset=False,
            ),
            InterlockType.EMERGENCY_STOP: SafetyInterlock(
                interlock_type=InterlockType.EMERGENCY_STOP,
                description="Emergency stop activated",
                auto_reset=False,
            ),
            InterlockType.SEQUENCE_TIMEOUT: SafetyInterlock(
                interlock_type=InterlockType.SEQUENCE_TIMEOUT,
                description="Cleaning sequence timed out",
                trip_value=self.config.cycle_timeout_seconds,
                auto_reset=True,
            ),
        }

    @property
    def is_connected(self) -> bool:
        """Check if controller is connected."""
        with self._lock:
            return self._connected

    async def connect(self) -> bool:
        """
        Connect to soot blower control system.

        Returns:
            True if connection successful.
        """
        if not self._circuit_breaker.is_available():
            logger.warning("Soot blower connection blocked by circuit breaker")
            return False

        try:
            if self.config.protocol == "modbus_tcp":
                await self._connect_modbus()
            elif self.config.protocol == "opc_ua":
                await self._connect_opc_ua()
            else:
                # Use SCADA client if provided
                if self.scada_client and not self.scada_client.is_connected:
                    await self.scada_client.connect()
                self._connected = True

            self._circuit_breaker.record_success()
            logger.info(f"Connected to soot blower controller {self.config.controller_id}")
            return True

        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.error(f"Failed to connect to soot blower controller: {e}")
            return False

    async def _connect_modbus(self) -> None:
        """Connect via Modbus TCP."""
        from pymodbus.client import AsyncModbusTcpClient

        self._modbus_client = AsyncModbusTcpClient(
            host=self.config.host,
            port=self.config.port,
            timeout=10,
        )
        await self._modbus_client.connect()
        self._connected = True

    async def _connect_opc_ua(self) -> None:
        """Connect via OPC UA."""
        from asyncua import Client

        endpoint = f"opc.tcp://{self.config.host}:{self.config.port}"
        self._opc_client = Client(url=endpoint)
        await self._opc_client.connect()
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from soot blower controller."""
        try:
            self._running = False

            if self._monitor_task:
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass

            if hasattr(self, '_modbus_client') and self._modbus_client:
                self._modbus_client.close()

            if hasattr(self, '_opc_client') and self._opc_client:
                await self._opc_client.disconnect()

            self._connected = False
            logger.info("Disconnected from soot blower controller")

        except Exception as e:
            logger.error(f"Error disconnecting from soot blower controller: {e}")

    def register_zone(self, zone: CleaningZone) -> None:
        """Register a cleaning zone."""
        with self._lock:
            self._zones[zone.zone_id] = zone
            logger.info(f"Registered zone: {zone.zone_id} - {zone.name}")

    def get_zone(self, zone_id: str) -> Optional[CleaningZone]:
        """Get a cleaning zone by ID."""
        with self._lock:
            return self._zones.get(zone_id)

    def get_all_zones(self) -> List[CleaningZone]:
        """Get all registered cleaning zones."""
        with self._lock:
            return list(self._zones.values())

    async def get_blower_status(self, blower_id: str) -> BlowerStatus:
        """
        Get current status of a soot blower.

        Args:
            blower_id: ID of the blower

        Returns:
            Current BlowerStatus
        """
        try:
            if self.scada_client:
                # Read status from SCADA
                tag_name = f"ECON.SB.{blower_id}.STATUS"
                tag_value = await self.scada_client.read_tag(tag_name)

                if tag_value and tag_value.is_good():
                    status_code = int(tag_value.value)
                    status = self._decode_blower_status(status_code)

                    with self._lock:
                        self._blower_status[blower_id] = status

                    return status

            # Return cached status if SCADA read fails
            with self._lock:
                return self._blower_status.get(blower_id, BlowerStatus.UNKNOWN)

        except Exception as e:
            logger.error(f"Error getting blower status for {blower_id}: {e}")
            return BlowerStatus.FAULT

    def _decode_blower_status(self, status_code: int) -> BlowerStatus:
        """Decode numeric status code to BlowerStatus."""
        status_map = {
            0: BlowerStatus.IDLE,
            1: BlowerStatus.RUNNING,
            2: BlowerStatus.ADVANCING,
            3: BlowerStatus.RETRACTING,
            4: BlowerStatus.PAUSED,
            5: BlowerStatus.FAULT,
            6: BlowerStatus.MAINTENANCE,
            7: BlowerStatus.DISABLED,
        }
        return status_map.get(status_code, BlowerStatus.UNKNOWN)

    async def get_all_blower_statuses(self) -> Dict[str, BlowerStatus]:
        """Get status of all registered blowers."""
        statuses = {}

        for zone in self._zones.values():
            for blower_id in zone.blower_ids:
                statuses[blower_id] = await self.get_blower_status(blower_id)

        return statuses

    async def check_interlocks(self) -> Tuple[bool, List[SafetyInterlock]]:
        """
        Check all safety interlocks.

        Returns:
            Tuple of (all_clear, list_of_active_interlocks)
        """
        active_interlocks = []

        for interlock in self._interlocks.values():
            if interlock.tag_name and self.scada_client:
                # Read current value from SCADA
                tag_value = await self.scada_client.read_tag(interlock.tag_name)

                if tag_value and tag_value.is_good():
                    interlock.current_value = float(tag_value.value)

                    # Check trip condition
                    if interlock.check_trip_condition(interlock.current_value):
                        if not interlock.is_active:
                            interlock.is_active = True
                            interlock.last_trip_time = datetime.now()
                            logger.warning(
                                f"Interlock tripped: {interlock.interlock_type.value} "
                                f"- {interlock.description}"
                            )
                    elif interlock.auto_reset:
                        # Check reset condition
                        if interlock.reset_value:
                            if not interlock.check_trip_condition(interlock.reset_value):
                                interlock.is_active = False

            if interlock.is_active:
                active_interlocks.append(interlock)

        all_clear = len(active_interlocks) == 0
        return all_clear, active_interlocks

    async def trigger_cleaning_cycle(
        self,
        zone_id: str,
        operator_id: str = "",
        trigger_type: str = "manual",
        force: bool = False,
    ) -> Tuple[bool, str, Optional[CleaningCycle]]:
        """
        Trigger a cleaning cycle for a zone.

        Args:
            zone_id: ID of the zone to clean
            operator_id: ID of the operator triggering the cycle
            trigger_type: "manual", "automatic", or "scheduled"
            force: Override minimum interval check (NOT interlock override)

        Returns:
            Tuple of (success, message, cycle_record)
        """
        zone = self.get_zone(zone_id)
        if not zone:
            return False, f"Zone not found: {zone_id}", None

        if not zone.enabled:
            return False, f"Zone is disabled: {zone_id}", None

        # Check if zone was recently cleaned (unless forced)
        if not force and zone.last_cleaned:
            hours_since_clean = (
                datetime.now() - zone.last_cleaned
            ).total_seconds() / 3600

            if hours_since_clean < zone.min_cleaning_interval_hours:
                return (
                    False,
                    f"Zone was cleaned {hours_since_clean:.1f} hours ago. "
                    f"Minimum interval is {zone.min_cleaning_interval_hours} hours.",
                    None,
                )

        # Check safety interlocks (CANNOT be overridden)
        interlocks_clear, active_interlocks = await self.check_interlocks()
        if not interlocks_clear:
            interlock_names = [i.interlock_type.value for i in active_interlocks]
            return (
                False,
                f"Safety interlocks active: {', '.join(interlock_names)}",
                None,
            )

        # Check concurrent blower limit
        active_count = len([
            c for c in self._active_cycles.values()
            if c.status in (CommandStatus.EXECUTING, CommandStatus.PENDING)
        ])

        if active_count >= self.config.max_concurrent_blowers:
            return (
                False,
                f"Maximum concurrent blowers ({self.config.max_concurrent_blowers}) reached",
                None,
            )

        # Record pre-cleaning differential pressure
        dp_before = await self._read_zone_dp(zone_id)

        # Create cycle record
        cycle = CleaningCycle(
            cycle_id=f"{zone_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            zone_id=zone_id,
            blower_id=zone.blower_ids[0] if zone.blower_ids else "",
            start_time=datetime.now(),
            media_type=self.config.media_type,
            dp_before_kpa=dp_before,
            operator_id=operator_id,
            trigger_type=trigger_type,
        )

        try:
            # Update zone status
            zone.status = ZoneStatus.CLEANING

            # Issue start command
            success = await self._issue_start_command(zone, cycle)

            if success:
                cycle.status = CommandStatus.EXECUTING

                with self._lock:
                    self._active_cycles[cycle.cycle_id] = cycle

                # Start monitoring task for this cycle
                asyncio.create_task(self._monitor_cycle(cycle))

                logger.info(f"Started cleaning cycle {cycle.cycle_id} for zone {zone_id}")
                return True, f"Cleaning cycle started: {cycle.cycle_id}", cycle

            else:
                cycle.status = CommandStatus.FAILED
                cycle.fault_message = "Failed to issue start command"
                return False, "Failed to issue start command", cycle

        except Exception as e:
            cycle.status = CommandStatus.FAILED
            cycle.fault_message = str(e)
            logger.error(f"Error starting cleaning cycle for zone {zone_id}: {e}")
            return False, str(e), cycle

    async def _issue_start_command(
        self, zone: CleaningZone, cycle: CleaningCycle
    ) -> bool:
        """Issue start command to soot blower system."""
        if self.scada_client:
            from .scada_integration import SetpointWriteRequest

            # Write start command to SCADA
            request = SetpointWriteRequest(
                tag_name=f"ECON.SB.{zone.zone_id}.CMD_START",
                value=1,
                node_id=f"ns=2;s=ECON.SB.{zone.zone_id}.StartCommand",
                requires_confirmation=True,
                timeout_seconds=5.0,
                reason=f"Start cleaning cycle {cycle.cycle_id}",
                operator_id=cycle.operator_id,
            )

            result = await self.scada_client.write_setpoint(request)
            return result.status.value == "success"

        return True  # Simulated success if no SCADA client

    async def _monitor_cycle(self, cycle: CleaningCycle) -> None:
        """Monitor a cleaning cycle until completion."""
        start_time = datetime.now()
        zone = self.get_zone(cycle.zone_id)

        try:
            while True:
                # Check timeout
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > self.config.cycle_timeout_seconds:
                    cycle.status = CommandStatus.FAILED
                    cycle.fault_message = "Cycle timeout"
                    logger.warning(f"Cleaning cycle {cycle.cycle_id} timed out")
                    break

                # Check blower status
                status = await self.get_blower_status(cycle.blower_id)

                if status == BlowerStatus.IDLE:
                    # Cycle completed
                    cycle.status = CommandStatus.COMPLETED
                    cycle.end_time = datetime.now()
                    cycle.duration_seconds = elapsed

                    # Read post-cleaning DP
                    cycle.dp_after_kpa = await self._read_zone_dp(cycle.zone_id)

                    # Calculate effectiveness
                    cycle.calculate_effectiveness()

                    # Update zone
                    if zone:
                        zone.last_cleaned = datetime.now()
                        zone.status = ZoneStatus.CLEAN

                    logger.info(
                        f"Cleaning cycle {cycle.cycle_id} completed. "
                        f"Effectiveness: {cycle.effectiveness_score:.1f}%"
                    )
                    break

                elif status == BlowerStatus.FAULT:
                    cycle.status = CommandStatus.FAILED
                    cycle.fault_message = "Blower fault detected"
                    logger.error(f"Cleaning cycle {cycle.cycle_id} failed - blower fault")
                    break

                # Update media consumption
                await self._update_media_consumption(cycle)

                # Check interlocks during operation
                interlocks_clear, active_interlocks = await self.check_interlocks()
                if not interlocks_clear:
                    # Stop cycle due to interlock
                    await self.stop_cleaning_cycle(cycle.cycle_id, "Interlock trip")
                    cycle.status = CommandStatus.FAILED
                    cycle.fault_message = f"Interlock: {active_interlocks[0].interlock_type.value}"
                    break

                await asyncio.sleep(1.0)  # Poll every second

        except asyncio.CancelledError:
            cycle.status = CommandStatus.CANCELLED
            logger.info(f"Cleaning cycle {cycle.cycle_id} was cancelled")

        except Exception as e:
            cycle.status = CommandStatus.FAILED
            cycle.fault_message = str(e)
            logger.error(f"Error monitoring cycle {cycle.cycle_id}: {e}")

        finally:
            # Move to history
            with self._lock:
                self._active_cycles.pop(cycle.cycle_id, None)
                self._cycle_history.append(cycle)

                # Trim history
                if len(self._cycle_history) > self._max_history_size:
                    self._cycle_history = self._cycle_history[-self._max_history_size:]

            # Notify callbacks
            for callback in self._cycle_callbacks:
                try:
                    callback(cycle)
                except Exception as e:
                    logger.error(f"Cycle callback error: {e}")

    async def stop_cleaning_cycle(
        self, cycle_id: str, reason: str = ""
    ) -> Tuple[bool, str]:
        """
        Stop an active cleaning cycle.

        Args:
            cycle_id: ID of the cycle to stop
            reason: Reason for stopping

        Returns:
            Tuple of (success, message)
        """
        cycle = self._active_cycles.get(cycle_id)
        if not cycle:
            return False, f"Cycle not found: {cycle_id}"

        try:
            if self.scada_client:
                from .scada_integration import SetpointWriteRequest

                request = SetpointWriteRequest(
                    tag_name=f"ECON.SB.{cycle.zone_id}.CMD_STOP",
                    value=1,
                    node_id=f"ns=2;s=ECON.SB.{cycle.zone_id}.StopCommand",
                    requires_confirmation=True,
                    timeout_seconds=5.0,
                    reason=reason,
                )

                result = await self.scada_client.write_setpoint(request)

                if result.status.value == "success":
                    cycle.status = CommandStatus.CANCELLED
                    cycle.end_time = datetime.now()
                    cycle.fault_message = reason
                    logger.info(f"Stopped cleaning cycle {cycle_id}: {reason}")
                    return True, "Cycle stopped successfully"

            return False, "Failed to issue stop command"

        except Exception as e:
            logger.error(f"Error stopping cycle {cycle_id}: {e}")
            return False, str(e)

    async def _read_zone_dp(self, zone_id: str) -> float:
        """Read differential pressure for a zone."""
        if self.scada_client:
            tag_name = f"ECON.DP.{zone_id}"
            tag_value = await self.scada_client.read_tag(tag_name)

            if tag_value and tag_value.is_good():
                return float(tag_value.value)

        return 0.0

    async def _update_media_consumption(self, cycle: CleaningCycle) -> None:
        """Update media consumption during active cycle."""
        if self.scada_client:
            # Read current steam flow
            tag_value = await self.scada_client.read_tag("ECON.SB.STEAM_FLOW")

            if tag_value and tag_value.is_good():
                flow_kg_hr = float(tag_value.value)

                # Integrate consumption (1 second interval)
                consumption_increment = flow_kg_hr / 3600
                cycle.media_consumed_kg += consumption_increment

                with self._lock:
                    self._media_consumption.total_consumed_kg += consumption_increment
                    self._media_consumption.consumption_rate_kg_hr = flow_kg_hr

            # Read pressure and temperature
            pressure_tag = await self.scada_client.read_tag("ECON.SB.STEAM_PRESSURE")
            temp_tag = await self.scada_client.read_tag("ECON.SB.STEAM_TEMP")

            if pressure_tag and pressure_tag.is_good():
                cycle.steam_pressure_avg_kpa = float(pressure_tag.value)

            if temp_tag and temp_tag.is_good():
                cycle.steam_temperature_avg_c = float(temp_tag.value)

    def get_cycle_history(
        self,
        zone_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[CleaningCycle]:
        """
        Get cleaning cycle history with optional filters.

        Args:
            zone_id: Optional zone filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum records to return

        Returns:
            List of CleaningCycle records
        """
        with self._lock:
            cycles = self._cycle_history.copy()

        # Apply filters
        if zone_id:
            cycles = [c for c in cycles if c.zone_id == zone_id]

        if start_time:
            cycles = [c for c in cycles if c.start_time >= start_time]

        if end_time:
            cycles = [c for c in cycles if c.start_time <= end_time]

        # Sort by start time descending and limit
        cycles.sort(key=lambda c: c.start_time, reverse=True)
        return cycles[:limit]

    def get_media_consumption(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> MediaConsumption:
        """
        Get media consumption data for a time period.

        Args:
            start_time: Start of period (default: current tracking period)
            end_time: End of period (default: now)

        Returns:
            MediaConsumption summary
        """
        with self._lock:
            consumption = MediaConsumption(
                media_type=self.config.media_type,
                period_start=start_time or self._media_consumption.period_start,
                period_end=end_time or datetime.now(),
            )

            # Sum consumption from cycle history
            for cycle in self._cycle_history:
                if start_time and cycle.start_time < start_time:
                    continue
                if end_time and cycle.start_time > end_time:
                    continue

                consumption.total_consumed_kg += cycle.media_consumed_kg
                consumption.cycle_count += 1

            # Calculate rate
            period_hours = (
                (consumption.period_end - consumption.period_start).total_seconds()
                / 3600
            )

            if period_hours > 0:
                consumption.consumption_rate_kg_hr = (
                    consumption.total_consumed_kg / period_hours
                )

            consumption.calculate_totals()
            return consumption

    def analyze_cleaning_effectiveness(
        self,
        zone_id: str,
        analysis_period_hours: float = 24.0,
    ) -> CleaningEffectiveness:
        """
        Analyze cleaning effectiveness for a zone.

        Args:
            zone_id: Zone to analyze
            analysis_period_hours: Period to analyze

        Returns:
            CleaningEffectiveness analysis
        """
        cutoff = datetime.now() - timedelta(hours=analysis_period_hours)

        cycles = [
            c for c in self._cycle_history
            if c.zone_id == zone_id
            and c.start_time >= cutoff
            and c.status == CommandStatus.COMPLETED
        ]

        if not cycles:
            return CleaningEffectiveness(
                zone_id=zone_id,
                analysis_period_hours=analysis_period_hours,
            )

        # Calculate statistics
        effectiveness_scores = [c.effectiveness_score for c in cycles]
        media_consumption = [c.media_consumed_kg for c in cycles]

        analysis = CleaningEffectiveness(
            zone_id=zone_id,
            analysis_period_hours=analysis_period_hours,
            cycles_analyzed=len(cycles),
            avg_effectiveness_score=sum(effectiveness_scores) / len(effectiveness_scores),
            min_effectiveness_score=min(effectiveness_scores),
            max_effectiveness_score=max(effectiveness_scores),
            avg_media_consumption_kg=sum(media_consumption) / len(media_consumption),
        )

        # Calculate DP reduction
        dp_reductions = []
        for c in cycles:
            if c.dp_before_kpa > 0:
                reduction = (c.dp_before_kpa - c.dp_after_kpa) / c.dp_before_kpa * 100
                dp_reductions.append(reduction)

        if dp_reductions:
            analysis.avg_dp_reduction_percent = sum(dp_reductions) / len(dp_reductions)

        # Calculate fouling rate
        if len(cycles) >= 2:
            cycles_sorted = sorted(cycles, key=lambda c: c.start_time)
            first = cycles_sorted[0]
            last = cycles_sorted[-1]

            time_diff_hours = (last.start_time - first.start_time).total_seconds() / 3600
            dp_diff = last.dp_before_kpa - first.dp_after_kpa

            if time_diff_hours > 0:
                analysis.fouling_rate_kpa_per_hour = dp_diff / time_diff_hours

        # Determine trend
        if len(cycles) >= 3:
            recent_scores = effectiveness_scores[-3:]
            earlier_scores = effectiveness_scores[:3]

            recent_avg = sum(recent_scores) / len(recent_scores)
            earlier_avg = sum(earlier_scores) / len(earlier_scores)

            if recent_avg > earlier_avg + 5:
                analysis.trend = "improving"
            elif recent_avg < earlier_avg - 5:
                analysis.trend = "degrading"
            else:
                analysis.trend = "stable"

        # Calculate recommended interval
        if analysis.fouling_rate_kpa_per_hour > 0:
            # Time to reach 0.7 fouling factor (assuming baseline DP relationship)
            analysis.recommended_cleaning_interval_hours = min(
                24.0,
                max(2.0, 1.0 / analysis.fouling_rate_kpa_per_hour * 10)
            )

        return analysis

    def register_status_callback(
        self, callback: Callable[[str, BlowerStatus], None]
    ) -> None:
        """Register callback for blower status changes."""
        self._status_callbacks.append(callback)

    def register_cycle_callback(
        self, callback: Callable[[CleaningCycle], None]
    ) -> None:
        """Register callback for cycle completion."""
        self._cycle_callbacks.append(callback)

    def get_interlock_status(self) -> Dict[str, SafetyInterlock]:
        """Get status of all interlocks."""
        return dict(self._interlocks)

    async def reset_interlock(
        self, interlock_type: InterlockType, operator_id: str = ""
    ) -> Tuple[bool, str]:
        """
        Manually reset a safety interlock.

        Args:
            interlock_type: Interlock to reset
            operator_id: Operator performing reset

        Returns:
            Tuple of (success, message)
        """
        interlock = self._interlocks.get(interlock_type)
        if not interlock:
            return False, f"Unknown interlock: {interlock_type.value}"

        if interlock.auto_reset:
            return False, "This interlock auto-resets when condition clears"

        # Verify condition has cleared
        if interlock.tag_name and self.scada_client:
            tag_value = await self.scada_client.read_tag(interlock.tag_name)

            if tag_value and tag_value.is_good():
                interlock.current_value = float(tag_value.value)

                if interlock.check_trip_condition(interlock.current_value):
                    return False, "Cannot reset - trip condition still active"

        interlock.is_active = False
        logger.info(
            f"Interlock {interlock_type.value} reset by {operator_id}"
        )
        return True, "Interlock reset successfully"

    async def close(self) -> None:
        """Clean up resources."""
        await self.disconnect()

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        active_interlocks = [
            i.to_dict()
            for i in self._interlocks.values()
            if i.is_active
        ]

        return {
            "controller_id": self.config.controller_id,
            "connected": self._connected,
            "circuit_breaker_state": self._circuit_breaker.state.name,
            "zones": len(self._zones),
            "active_cycles": len(self._active_cycles),
            "active_interlocks": active_interlocks,
            "total_cycles_completed": len(self._cycle_history),
            "media_consumption_kg": self._media_consumption.total_consumed_kg,
        }
