"""
GreenLang Deterministic Clock - Controlled Timestamp Generation

This module provides deterministic time management for testing and auditing.

Features:
- Freezable clock for deterministic testing
- Thread-safe singleton pattern
- UTC-aware timestamps
- Microsecond removal for consistency

Author: GreenLang Team
Date: 2025-11-21
"""

import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional


class DeterministicClock:
    """
    A deterministic clock that can be frozen for testing and auditing.

    This clock ensures consistent timestamp generation across the application,
    with the ability to freeze time for deterministic testing.
    """

    _instance = None
    _lock = threading.Lock()
    _frozen_time: Optional[datetime] = None
    _time_offset: float = 0.0

    def __new__(cls):
        """Singleton pattern to ensure single clock instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    @classmethod
    def now(cls, tz=None) -> datetime:
        """
        Get current time, either real or frozen.

        Args:
            tz: Timezone info (defaults to UTC if frozen)

        Returns:
            Current datetime
        """
        instance = cls()
        if instance._frozen_time is not None:
            if tz is not None:
                return instance._frozen_time.replace(tzinfo=tz)
            return instance._frozen_time

        # Apply offset for testing time progression
        if instance._time_offset != 0:
            base_time = datetime.now(tz or timezone.utc)
            return base_time.replace(microsecond=0)  # Remove microseconds for determinism

        return datetime.now(tz or timezone.utc).replace(microsecond=0)

    @classmethod
    def utcnow(cls) -> datetime:
        """Get current UTC time."""
        return cls.now(timezone.utc)

    @classmethod
    def freeze(cls, frozen_time: Optional[datetime] = None):
        """
        Freeze clock at specific time.

        Args:
            frozen_time: Time to freeze at (defaults to current time)
        """
        instance = cls()
        if frozen_time is None:
            frozen_time = datetime.now(timezone.utc).replace(microsecond=0)
        instance._frozen_time = frozen_time

    @classmethod
    def unfreeze(cls):
        """Unfreeze the clock."""
        instance = cls()
        instance._frozen_time = None
        instance._time_offset = 0.0

    @classmethod
    @contextmanager
    def frozen(cls, frozen_time: Optional[datetime] = None):
        """
        Context manager for temporarily freezing time.

        Usage:
            with DeterministicClock.frozen(datetime(2025, 1, 1)):
                # All timestamps will be 2025-01-01
                pass
        """
        cls.freeze(frozen_time)
        try:
            yield
        finally:
            cls.unfreeze()


# Convenience functions
def now() -> datetime:
    """Get current deterministic time."""
    return DeterministicClock.now()


def utcnow() -> datetime:
    """Get current deterministic UTC time."""
    return DeterministicClock.utcnow()


def freeze_time(frozen_time: Optional[datetime] = None):
    """Freeze time for testing."""
    DeterministicClock.freeze(frozen_time)


def unfreeze_time():
    """Unfreeze time."""
    DeterministicClock.unfreeze()


__all__ = [
    'DeterministicClock',
    'now',
    'utcnow',
    'freeze_time',
    'unfreeze_time',
]
