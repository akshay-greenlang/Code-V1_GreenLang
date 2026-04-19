"""
Fail-Safe Design Framework - IEC 61511 Compliant Fail-Safe Implementations

This module provides fail-safe design patterns and implementations
per IEC 61511 for Safety Instrumented Systems.

Components:
- VotingLogic: 1oo1, 1oo2, 2oo3 voting implementations
- Watchdog: Watchdog timer framework
- ManualOverride: Override management with logging
- SafeTransition: Safe state transition logic
- DeenergizeToTrip: De-energize-to-trip implementation

Reference: IEC 61511-1 Clause 11

Example:
    >>> from greenlang.safety.failsafe import VotingLogic, Watchdog
    >>> voter = VotingLogic("2oo3")
    >>> result = voter.evaluate([True, True, False])
"""

from greenlang.safety.failsafe.voting_logic import (
    VotingLogic,
    VotingArchitecture,
    VotingResult,
)
from greenlang.safety.failsafe.watchdog import (
    Watchdog,
    WatchdogConfig,
    WatchdogState,
)
from greenlang.safety.failsafe.manual_override import (
    ManualOverride,
    OverrideRequest,
    OverrideRecord,
)
from greenlang.safety.failsafe.safe_transition import (
    SafeTransition,
    TransitionConfig,
    TransitionResult,
)
from greenlang.safety.failsafe.deenergize_to_trip import (
    DeenergizeToTrip,
    TripChannel,
    TripResult,
)

__all__ = [
    # Voting Logic
    "VotingLogic",
    "VotingArchitecture",
    "VotingResult",
    # Watchdog
    "Watchdog",
    "WatchdogConfig",
    "WatchdogState",
    # Manual Override
    "ManualOverride",
    "OverrideRequest",
    "OverrideRecord",
    # Safe Transition
    "SafeTransition",
    "TransitionConfig",
    "TransitionResult",
    # De-energize to Trip
    "DeenergizeToTrip",
    "TripChannel",
    "TripResult",
]

__version__ = "1.0.0"
