"""
GL-002 FLAMEGUARD - Flame Detector

Flame detection and monitoring per NFPA 85.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class FlameStatus(Enum):
    """Flame detection status."""
    NO_FLAME = "no_flame"
    FLAME_PRESENT = "flame_present"
    UNSTABLE = "unstable"
    SCANNER_FAULT = "scanner_fault"


@dataclass
class FlameScanner:
    """Flame scanner configuration."""
    scanner_id: str
    scanner_type: str  # UV, IR, combination
    signal_percent: float = 0.0
    healthy: bool = True
    last_update: Optional[datetime] = None


class FlameDetector:
    """
    Flame detection system per NFPA 85.

    Features:
    - Multiple scanner support with voting logic
    - Flame failure detection within 4 seconds
    - Scanner self-check integration
    - Signal quality monitoring
    """

    FLAME_FAILURE_TIME_S = 4.0  # NFPA 85 requirement
    MIN_SIGNAL_PERCENT = 10.0  # Minimum for flame proven
    UNSTABLE_THRESHOLD = 20.0  # Below this = unstable

    def __init__(
        self,
        boiler_id: str,
        voting_logic: str = "2oo3",  # 1oo1, 1oo2, 2oo2, 2oo3
        failure_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Initialize flame detector."""
        self.boiler_id = boiler_id
        self.voting_logic = voting_logic
        self._failure_callback = failure_callback

        # Scanners
        self._scanners: Dict[str, FlameScanner] = {}

        # State
        self._status = FlameStatus.NO_FLAME
        self._flame_proven = False
        self._signal_percent = 0.0
        self._flame_loss_time: Optional[datetime] = None

        # History
        self._flame_events: List[Dict] = []

        logger.info(f"FlameDetector initialized for {boiler_id} with {voting_logic}")

    def add_scanner(
        self,
        scanner_id: str,
        scanner_type: str = "UV",
    ) -> None:
        """Add a flame scanner."""
        self._scanners[scanner_id] = FlameScanner(
            scanner_id=scanner_id,
            scanner_type=scanner_type,
        )
        logger.info(f"Added flame scanner: {scanner_id} ({scanner_type})")

    def update_scanner(
        self,
        scanner_id: str,
        signal_percent: float,
        healthy: bool = True,
    ) -> None:
        """Update scanner reading."""
        if scanner_id not in self._scanners:
            return

        scanner = self._scanners[scanner_id]
        scanner.signal_percent = signal_percent
        scanner.healthy = healthy
        scanner.last_update = datetime.now(timezone.utc)

        # Re-evaluate flame status
        self._evaluate_flame_status()

    def _evaluate_flame_status(self) -> None:
        """Evaluate overall flame status using voting logic."""
        if not self._scanners:
            self._status = FlameStatus.NO_FLAME
            self._flame_proven = False
            return

        # Get healthy scanner signals
        healthy_signals = [
            s.signal_percent for s in self._scanners.values()
            if s.healthy
        ]

        if not healthy_signals:
            self._status = FlameStatus.SCANNER_FAULT
            self._flame_proven = False
            return

        # Apply voting logic
        flame_detected = self._apply_voting(healthy_signals)

        # Calculate combined signal
        self._signal_percent = sum(healthy_signals) / len(healthy_signals)

        # Determine status
        if flame_detected:
            if self._signal_percent >= self.MIN_SIGNAL_PERCENT:
                if self._signal_percent < self.UNSTABLE_THRESHOLD:
                    self._status = FlameStatus.UNSTABLE
                else:
                    self._status = FlameStatus.FLAME_PRESENT
                self._flame_proven = True
                self._flame_loss_time = None
            else:
                self._handle_flame_loss()
        else:
            self._handle_flame_loss()

    def _apply_voting(self, signals: List[float]) -> bool:
        """Apply voting logic to scanner signals."""
        threshold = self.MIN_SIGNAL_PERCENT
        flames = [s >= threshold for s in signals]
        n = len(flames)

        if self.voting_logic == "1oo1":
            return flames[0] if n >= 1 else False
        elif self.voting_logic == "1oo2":
            return any(flames) if n >= 2 else False
        elif self.voting_logic == "2oo2":
            return all(flames) if n >= 2 else False
        elif self.voting_logic == "2oo3":
            return sum(flames) >= 2 if n >= 3 else all(flames)
        else:
            return any(flames)

    def _handle_flame_loss(self) -> None:
        """Handle flame loss detection."""
        now = datetime.now(timezone.utc)

        if self._flame_loss_time is None:
            self._flame_loss_time = now
            logger.warning(f"Flame signal low: {self.boiler_id}")

        # Check failure time
        elapsed = (now - self._flame_loss_time).total_seconds()
        if elapsed >= self.FLAME_FAILURE_TIME_S:
            self._status = FlameStatus.NO_FLAME
            self._flame_proven = False

            if self._failure_callback:
                self._failure_callback(self.boiler_id)

            self._flame_events.append({
                "timestamp": now,
                "event": "flame_failure",
                "signal": self._signal_percent,
            })

            logger.critical(f"FLAME FAILURE: {self.boiler_id}")

    def is_flame_proven(self) -> bool:
        """Check if flame is proven."""
        return self._flame_proven

    @property
    def status(self) -> FlameStatus:
        """Get flame status."""
        return self._status

    @property
    def signal_percent(self) -> float:
        """Get combined signal percentage."""
        return self._signal_percent

    def get_status(self) -> Dict:
        """Get detailed status."""
        return {
            "boiler_id": self.boiler_id,
            "status": self._status.value,
            "flame_proven": self._flame_proven,
            "signal_percent": round(self._signal_percent, 1),
            "voting_logic": self.voting_logic,
            "scanners": {
                sid: {
                    "type": s.scanner_type,
                    "signal": s.signal_percent,
                    "healthy": s.healthy,
                }
                for sid, s in self._scanners.items()
            },
            "event_count": len(self._flame_events),
        }
