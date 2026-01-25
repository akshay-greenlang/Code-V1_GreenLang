"""
GL-002 FLAMEGUARD - Burner Management System

BMS implementation per NFPA 85 with state machine control
for safe boiler startup, operation, and shutdown sequences.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Callable, Dict, List, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)


class BurnerState(Enum):
    """Burner operating states per NFPA 85."""
    OFFLINE = auto()
    COLD_STANDBY = auto()
    PRE_PURGE = auto()
    PILOT_LIGHT_TRIAL = auto()
    PILOT_PROVEN = auto()
    MAIN_FLAME_TRIAL = auto()
    MAIN_FLAME_PROVEN = auto()
    FIRING = auto()
    POST_PURGE = auto()
    LOCKOUT = auto()
    EMERGENCY_SHUTDOWN = auto()


@dataclass
class BurnerPermissive:
    """Safety permissive condition."""
    name: str
    description: str
    satisfied: bool = False
    required_for_states: List[BurnerState] = field(default_factory=list)
    bypass_allowed: bool = False
    bypassed: bool = False
    last_check: Optional[datetime] = None


class BurnerManagementSystem:
    """
    Burner Management System per NFPA 85.

    Implements:
    - Safe startup sequence with purge
    - Flame detection and proving
    - Safety interlocks and permissives
    - Emergency shutdown
    - State machine control
    """

    # Timing requirements per NFPA 85
    PRE_PURGE_TIME_S = 300  # 5 minutes minimum
    PILOT_TRIAL_TIME_S = 10
    MAIN_FLAME_TRIAL_TIME_S = 10
    FLAME_FAILURE_RESPONSE_S = 4
    POST_PURGE_TIME_S = 60

    def __init__(
        self,
        boiler_id: str,
        trip_callback: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """Initialize BMS."""
        self.boiler_id = boiler_id
        self._trip_callback = trip_callback

        # State
        self._state = BurnerState.OFFLINE
        self._prev_state = BurnerState.OFFLINE
        self._state_entry_time: Optional[datetime] = None
        self._lockout_reason: Optional[str] = None

        # Permissives
        self._permissives: Dict[str, BurnerPermissive] = self._init_permissives()

        # Flame status
        self._flame_signal = 0.0
        self._flame_proven = False
        self._pilot_proven = False

        # Trip history
        self._trip_history: List[Dict] = []

        logger.info(f"BMS initialized for {boiler_id}")

    def _init_permissives(self) -> Dict[str, BurnerPermissive]:
        """Initialize standard permissives."""
        return {
            "drum_level_ok": BurnerPermissive(
                name="drum_level_ok",
                description="Drum level within limits",
                required_for_states=[
                    BurnerState.PRE_PURGE,
                    BurnerState.PILOT_LIGHT_TRIAL,
                    BurnerState.FIRING,
                ],
            ),
            "steam_pressure_ok": BurnerPermissive(
                name="steam_pressure_ok",
                description="Steam pressure below high limit",
                required_for_states=[BurnerState.FIRING],
            ),
            "fuel_pressure_ok": BurnerPermissive(
                name="fuel_pressure_ok",
                description="Fuel pressure within limits",
                required_for_states=[
                    BurnerState.PILOT_LIGHT_TRIAL,
                    BurnerState.FIRING,
                ],
            ),
            "combustion_air_ok": BurnerPermissive(
                name="combustion_air_ok",
                description="FD fan running, air pressure OK",
                required_for_states=[
                    BurnerState.PRE_PURGE,
                    BurnerState.PILOT_LIGHT_TRIAL,
                    BurnerState.FIRING,
                ],
            ),
            "purge_complete": BurnerPermissive(
                name="purge_complete",
                description="Pre-purge completed",
                required_for_states=[BurnerState.PILOT_LIGHT_TRIAL],
            ),
            "flame_scanner_ok": BurnerPermissive(
                name="flame_scanner_ok",
                description="Flame scanner healthy",
                required_for_states=[
                    BurnerState.PILOT_LIGHT_TRIAL,
                    BurnerState.FIRING,
                ],
            ),
            "no_lockout": BurnerPermissive(
                name="no_lockout",
                description="No active lockout condition",
                required_for_states=[
                    BurnerState.PRE_PURGE,
                    BurnerState.PILOT_LIGHT_TRIAL,
                ],
            ),
        }

    @property
    def state(self) -> BurnerState:
        """Get current state."""
        return self._state

    @property
    def is_firing(self) -> bool:
        """Check if burner is firing."""
        return self._state in [
            BurnerState.MAIN_FLAME_PROVEN,
            BurnerState.FIRING,
        ]

    def update_permissive(
        self,
        name: str,
        satisfied: bool,
    ) -> None:
        """Update a permissive condition."""
        if name in self._permissives:
            self._permissives[name].satisfied = satisfied
            self._permissives[name].last_check = datetime.now(timezone.utc)

            # Check for trip conditions
            if not satisfied and self.is_firing:
                perm = self._permissives[name]
                if self._state in perm.required_for_states:
                    self._trip(f"Permissive lost: {name}")

    def update_flame_signal(self, signal_percent: float) -> None:
        """Update flame signal."""
        self._flame_signal = signal_percent
        self._flame_proven = signal_percent >= 10.0

        # Check for flame failure during firing
        if self.is_firing and not self._flame_proven:
            logger.critical(f"FLAME FAILURE: {self.boiler_id}")
            self._trip("Flame failure")

    async def start_sequence(self) -> bool:
        """
        Start the burner startup sequence.

        Returns True if startup initiated successfully.
        """
        if self._state != BurnerState.OFFLINE:
            logger.warning(f"Cannot start - state is {self._state}")
            return False

        # Check critical permissives
        if not self._check_permissives_for_state(BurnerState.PRE_PURGE):
            logger.error("Permissives not satisfied for startup")
            return False

        # Begin pre-purge
        self._transition_to(BurnerState.PRE_PURGE)

        # Run purge timer
        logger.info(f"Starting pre-purge for {self.boiler_id}")
        await asyncio.sleep(self.PRE_PURGE_TIME_S)

        # Check permissives after purge
        if not self._check_permissives_for_state(BurnerState.PILOT_LIGHT_TRIAL):
            self._trip("Permissives lost during purge")
            return False

        # Mark purge complete
        self._permissives["purge_complete"].satisfied = True

        # Pilot trial
        self._transition_to(BurnerState.PILOT_LIGHT_TRIAL)
        logger.info(f"Pilot trial for {self.boiler_id}")

        # Wait for pilot proving time
        await asyncio.sleep(self.PILOT_TRIAL_TIME_S)

        if not self._flame_proven:
            self._trip("Pilot not proven")
            return False

        self._pilot_proven = True
        self._transition_to(BurnerState.PILOT_PROVEN)

        # Main flame trial
        self._transition_to(BurnerState.MAIN_FLAME_TRIAL)
        await asyncio.sleep(self.MAIN_FLAME_TRIAL_TIME_S)

        if not self._flame_proven:
            self._trip("Main flame not proven")
            return False

        # Success - transition to firing
        self._transition_to(BurnerState.FIRING)
        logger.info(f"Burner firing: {self.boiler_id}")
        return True

    async def stop_sequence(self) -> None:
        """Stop the burner with post-purge."""
        if not self.is_firing:
            logger.info("Burner already stopped")
            return

        # Stop firing
        self._transition_to(BurnerState.POST_PURGE)
        logger.info(f"Post-purge for {self.boiler_id}")

        await asyncio.sleep(self.POST_PURGE_TIME_S)

        self._transition_to(BurnerState.OFFLINE)
        self._permissives["purge_complete"].satisfied = False
        logger.info(f"Burner stopped: {self.boiler_id}")

    def emergency_stop(self, reason: str = "Operator request") -> None:
        """Immediate emergency shutdown."""
        logger.critical(f"EMERGENCY STOP: {self.boiler_id} - {reason}")
        self._trip(reason)

    def _trip(self, reason: str) -> None:
        """Execute safety trip."""
        self._prev_state = self._state
        self._state = BurnerState.LOCKOUT
        self._lockout_reason = reason
        self._state_entry_time = datetime.now(timezone.utc)
        self._flame_proven = False
        self._pilot_proven = False

        # Record trip
        self._trip_history.append({
            "timestamp": datetime.now(timezone.utc),
            "from_state": self._prev_state.name,
            "reason": reason,
        })

        # Invoke callback
        if self._trip_callback:
            self._trip_callback(self.boiler_id, reason)

        logger.critical(f"BMS TRIP: {self.boiler_id} - {reason}")

    def reset_lockout(self, operator: str) -> bool:
        """
        Reset lockout condition.

        Requires operator authorization.
        """
        if self._state != BurnerState.LOCKOUT:
            return False

        # Check that cause is cleared
        if not self._check_permissives_for_state(BurnerState.OFFLINE):
            logger.warning("Cannot reset - permissives not satisfied")
            return False

        self._state = BurnerState.OFFLINE
        self._lockout_reason = None
        self._permissives["no_lockout"].satisfied = True
        self._permissives["purge_complete"].satisfied = False

        logger.info(f"Lockout reset by {operator}: {self.boiler_id}")
        return True

    def _transition_to(self, new_state: BurnerState) -> None:
        """Transition to new state."""
        self._prev_state = self._state
        self._state = new_state
        self._state_entry_time = datetime.now(timezone.utc)
        logger.debug(f"BMS state: {self._prev_state.name} -> {new_state.name}")

    def _check_permissives_for_state(self, target_state: BurnerState) -> bool:
        """Check if all permissives are satisfied for target state."""
        for perm in self._permissives.values():
            if target_state in perm.required_for_states:
                if not perm.satisfied and not perm.bypassed:
                    logger.debug(f"Permissive not satisfied: {perm.name}")
                    return False
        return True

    def get_status(self) -> Dict:
        """Get BMS status."""
        return {
            "boiler_id": self.boiler_id,
            "state": self._state.name,
            "prev_state": self._prev_state.name,
            "flame_signal": self._flame_signal,
            "flame_proven": self._flame_proven,
            "pilot_proven": self._pilot_proven,
            "lockout_reason": self._lockout_reason,
            "permissives": {
                name: {
                    "satisfied": p.satisfied,
                    "bypassed": p.bypassed,
                }
                for name, p in self._permissives.items()
            },
            "trip_count": len(self._trip_history),
        }
