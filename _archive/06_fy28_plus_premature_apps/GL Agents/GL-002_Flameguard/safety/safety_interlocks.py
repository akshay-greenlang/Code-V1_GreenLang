"""
GL-002 FLAMEGUARD - Safety Interlocks

Safety interlock management per IEC 61511.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class InterlockStatus(Enum):
    """Interlock status."""
    NORMAL = "normal"
    ALARM = "alarm"
    TRIP = "trip"
    BYPASSED = "bypassed"


@dataclass
class SafetyInterlock:
    """Safety interlock definition."""
    tag: str
    description: str
    trip_high: Optional[float] = None
    trip_low: Optional[float] = None
    alarm_high: Optional[float] = None
    alarm_low: Optional[float] = None
    unit: str = ""
    sil_level: int = 2
    current_value: float = 0.0
    status: InterlockStatus = InterlockStatus.NORMAL
    bypassed: bool = False
    bypass_reason: Optional[str] = None
    bypass_expiry: Optional[datetime] = None


class SafetyInterlockManager:
    """
    Manages safety interlocks per IEC 61511.

    Features:
    - High/low trip and alarm logic
    - Bypass management with expiry
    - SIL level tracking
    - Trip history
    """

    def __init__(
        self,
        boiler_id: str,
        trip_callback: Optional[Callable[[str, str], None]] = None,
        alarm_callback: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self.boiler_id = boiler_id
        self._trip_callback = trip_callback
        self._alarm_callback = alarm_callback

        # Interlocks
        self._interlocks: Dict[str, SafetyInterlock] = {}

        # Trip state
        self._tripped = False
        self._trip_causes: List[str] = []

        # Initialize standard interlocks
        self._init_standard_interlocks()

        logger.info(f"SafetyInterlockManager initialized for {boiler_id}")

    def _init_standard_interlocks(self) -> None:
        """Initialize standard boiler safety interlocks."""
        interlocks = [
            SafetyInterlock(
                tag="STEAM_PRESSURE",
                description="High steam pressure",
                trip_high=150.0,
                alarm_high=140.0,
                unit="psig",
                sil_level=2,
            ),
            SafetyInterlock(
                tag="DRUM_LEVEL",
                description="Drum level",
                trip_high=8.0,
                trip_low=-4.0,
                alarm_high=6.0,
                alarm_low=-2.0,
                unit="inches",
                sil_level=3,
            ),
            SafetyInterlock(
                tag="FUEL_PRESSURE",
                description="Fuel gas pressure",
                trip_high=25.0,
                trip_low=2.0,
                alarm_high=22.0,
                alarm_low=5.0,
                unit="psig",
                sil_level=2,
            ),
            SafetyInterlock(
                tag="COMBUSTION_AIR",
                description="Combustion air pressure",
                trip_low=0.5,
                alarm_low=1.0,
                unit="in WC",
                sil_level=2,
            ),
            SafetyInterlock(
                tag="FLUE_GAS_TEMP",
                description="Flue gas temperature",
                trip_high=700.0,
                alarm_high=650.0,
                unit="°F",
                sil_level=1,
            ),
        ]

        for interlock in interlocks:
            self._interlocks[interlock.tag] = interlock

    def update_value(
        self,
        tag: str,
        value: float,
    ) -> InterlockStatus:
        """
        Update interlock value and check status.

        Returns current status.
        """
        if tag not in self._interlocks:
            return InterlockStatus.NORMAL

        interlock = self._interlocks[tag]
        interlock.current_value = value
        prev_status = interlock.status

        # Skip if bypassed
        if interlock.bypassed:
            # Check bypass expiry
            if (interlock.bypass_expiry and
                datetime.now(timezone.utc) > interlock.bypass_expiry):
                self._clear_bypass(tag)
            else:
                return InterlockStatus.BYPASSED

        # Check trip conditions
        if interlock.trip_high is not None and value >= interlock.trip_high:
            interlock.status = InterlockStatus.TRIP
            self._handle_trip(tag, f"High trip: {value:.1f} >= {interlock.trip_high}")
        elif interlock.trip_low is not None and value <= interlock.trip_low:
            interlock.status = InterlockStatus.TRIP
            self._handle_trip(tag, f"Low trip: {value:.1f} <= {interlock.trip_low}")
        # Check alarm conditions
        elif interlock.alarm_high is not None and value >= interlock.alarm_high:
            interlock.status = InterlockStatus.ALARM
            if prev_status != InterlockStatus.ALARM:
                self._handle_alarm(tag, f"High alarm: {value:.1f}")
        elif interlock.alarm_low is not None and value <= interlock.alarm_low:
            interlock.status = InterlockStatus.ALARM
            if prev_status != InterlockStatus.ALARM:
                self._handle_alarm(tag, f"Low alarm: {value:.1f}")
        else:
            interlock.status = InterlockStatus.NORMAL

        return interlock.status

    def _handle_trip(self, tag: str, message: str) -> None:
        """Handle trip condition."""
        self._tripped = True
        self._trip_causes.append(f"{tag}: {message}")

        logger.critical(f"SAFETY TRIP [{self.boiler_id}] {tag}: {message}")

        if self._trip_callback:
            self._trip_callback(tag, message)

    def _handle_alarm(self, tag: str, message: str) -> None:
        """Handle alarm condition."""
        logger.warning(f"SAFETY ALARM [{self.boiler_id}] {tag}: {message}")

        if self._alarm_callback:
            self._alarm_callback(tag, message)

    def set_bypass(
        self,
        tag: str,
        reason: str,
        duration_minutes: int = 60,
        operator: str = "unknown",
    ) -> bool:
        """
        Set bypass on interlock.

        Returns True if bypass set successfully.
        """
        if tag not in self._interlocks:
            return False

        interlock = self._interlocks[tag]

        # Check if bypass allowed (SIL3+ cannot be bypassed)
        if interlock.sil_level >= 3:
            logger.error(f"Cannot bypass SIL{interlock.sil_level} interlock: {tag}")
            return False

        interlock.bypassed = True
        interlock.bypass_reason = f"{reason} by {operator}"
        interlock.bypass_expiry = datetime.now(timezone.utc).replace(
            minute=datetime.now(timezone.utc).minute + duration_minutes
        )
        interlock.status = InterlockStatus.BYPASSED

        logger.warning(
            f"Interlock bypassed [{self.boiler_id}] {tag}: {reason} "
            f"for {duration_minutes} min by {operator}"
        )
        return True

    def _clear_bypass(self, tag: str) -> None:
        """Clear bypass on interlock."""
        if tag in self._interlocks:
            interlock = self._interlocks[tag]
            interlock.bypassed = False
            interlock.bypass_reason = None
            interlock.bypass_expiry = None
            interlock.status = InterlockStatus.NORMAL
            logger.info(f"Bypass cleared: {tag}")

    def clear_all_bypasses(self) -> None:
        """Clear all bypasses."""
        for tag in self._interlocks:
            self._clear_bypass(tag)

    def reset_trip(self, operator: str) -> bool:
        """Reset trip condition."""
        if not self._tripped:
            return True

        # Check all interlocks are normal
        for interlock in self._interlocks.values():
            if interlock.status == InterlockStatus.TRIP:
                return False

        self._tripped = False
        self._trip_causes.clear()
        logger.info(f"Trip reset by {operator}: {self.boiler_id}")
        return True

    @property
    def is_tripped(self) -> bool:
        """Check if any interlock is tripped."""
        return self._tripped

    def get_status(self) -> Dict:
        """Get interlock status summary."""
        return {
            "boiler_id": self.boiler_id,
            "tripped": self._tripped,
            "trip_causes": list(self._trip_causes),
            "interlocks": {
                tag: {
                    "value": i.current_value,
                    "status": i.status.value,
                    "unit": i.unit,
                    "bypassed": i.bypassed,
                }
                for tag, i in self._interlocks.items()
            },
            "bypassed_count": sum(1 for i in self._interlocks.values() if i.bypassed),
            "alarm_count": sum(
                1 for i in self._interlocks.values()
                if i.status == InterlockStatus.ALARM
            ),
        }
