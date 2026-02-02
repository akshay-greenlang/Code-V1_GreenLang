"""
Tests for GreenLang Deterministic Clock

Tests clock management, time freezing, and deterministic timestamp generation.
"""

import pytest
from datetime import datetime, timezone

from greenlang.determinism.clock import (
    DeterministicClock,
    now,
    utcnow,
    freeze_time,
    unfreeze_time,
)


class TestDeterministicClock:
    """Test DeterministicClock functionality."""

    def setup_method(self):
        """Reset clock before each test."""
        DeterministicClock.unfreeze()

    def teardown_method(self):
        """Clean up after each test."""
        DeterministicClock.unfreeze()

    def test_singleton_pattern(self):
        """Test that DeterministicClock uses singleton pattern."""
        clock1 = DeterministicClock()
        clock2 = DeterministicClock()
        assert clock1 is clock2

    def test_now_returns_datetime(self):
        """Test that now() returns a datetime object."""
        result = DeterministicClock.now()
        assert isinstance(result, datetime)

    def test_utcnow_returns_utc_datetime(self):
        """Test that utcnow() returns UTC datetime."""
        result = DeterministicClock.utcnow()
        assert isinstance(result, datetime)
        assert result.tzinfo == timezone.utc

    def test_freeze_time(self):
        """Test freezing time at specific moment."""
        frozen_dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        DeterministicClock.freeze(frozen_dt)

        result = DeterministicClock.now()
        assert result == frozen_dt

    def test_freeze_time_defaults_to_current(self):
        """Test that freeze without argument uses current time."""
        DeterministicClock.freeze()

        time1 = DeterministicClock.now()
        time2 = DeterministicClock.now()

        assert time1 == time2

    def test_unfreeze_time(self):
        """Test unfreezing time."""
        frozen_dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        DeterministicClock.freeze(frozen_dt)
        DeterministicClock.unfreeze()

        result = DeterministicClock.now()
        assert result != frozen_dt

    def test_frozen_context_manager(self):
        """Test frozen() context manager."""
        frozen_dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        with DeterministicClock.frozen(frozen_dt):
            assert DeterministicClock.now() == frozen_dt

        # After context, time should be unfrozen
        assert DeterministicClock.now() != frozen_dt

    def test_convenience_functions(self):
        """Test convenience functions (now, utcnow, freeze_time, unfreeze_time)."""
        frozen_dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        freeze_time(frozen_dt)
        assert now() == frozen_dt
        assert utcnow() == frozen_dt

        unfreeze_time()
        assert now() != frozen_dt

    def test_microseconds_removed_for_determinism(self):
        """Test that microseconds are removed for determinism."""
        result = DeterministicClock.now()
        assert result.microsecond == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
