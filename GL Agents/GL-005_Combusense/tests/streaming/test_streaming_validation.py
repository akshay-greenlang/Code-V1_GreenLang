# -*- coding: utf-8 -*-
"""
Streaming Data Validation Tests for GL-005 CombustionSense
==========================================================

Tests for real-time streaming data validation including:
    - Range validation
    - Rate-of-change monitoring
    - Gap detection
    - Spike detection
    - Frozen value detection

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import random
from datetime import datetime, timedelta
from typing import List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from monitoring.streaming_validator import (
    StreamingDataValidator,
    DataPoint,
    ValidationSpec,
    ValidationResult,
    QualityFlag,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def validator() -> StreamingDataValidator:
    """Create configured streaming validator."""
    v = StreamingDataValidator(window_size=100)

    # Register combustion parameters
    specs = [
        ValidationSpec(
            parameter="O2",
            range_min=0.0,
            range_max=25.0,
            max_rate_of_change=2.0,
            max_gap_seconds=5.0,
            spike_threshold=4.0,
        ),
        ValidationSpec(
            parameter="CO",
            range_min=0.0,
            range_max=5000.0,
            max_rate_of_change=100.0,
            max_gap_seconds=5.0,
            spike_threshold=4.0,
        ),
        ValidationSpec(
            parameter="flame_signal",
            range_min=0.0,
            range_max=20.0,
            max_rate_of_change=5.0,
            max_gap_seconds=1.0,
            spike_threshold=3.0,
        ),
    ]

    for spec in specs:
        v.register_parameter(spec)

    return v


def create_data_points(
    parameter: str,
    values: List[float],
    start_time: datetime = None,
    interval_seconds: float = 1.0
) -> List[DataPoint]:
    """Helper to create data points."""
    if start_time is None:
        start_time = datetime.now()

    return [
        DataPoint(
            parameter=parameter,
            value=v,
            timestamp=start_time + timedelta(seconds=i * interval_seconds),
            source_id=f"{parameter}-PRIMARY",
        )
        for i, v in enumerate(values)
    ]


# =============================================================================
# RANGE VALIDATION TESTS
# =============================================================================

class TestRangeValidation:
    """Test range validation."""

    @pytest.mark.parametrize("value,expected_result", [
        (3.5, ValidationResult.VALID),
        (0.0, ValidationResult.VALID),
        (25.0, ValidationResult.VALID),
        (-1.0, ValidationResult.INVALID),
        (26.0, ValidationResult.INVALID),
    ])
    def test_o2_range_validation(
        self,
        validator: StreamingDataValidator,
        value: float,
        expected_result: ValidationResult
    ):
        """Test O2 range validation."""
        dp = DataPoint(
            parameter="O2",
            value=value,
            timestamp=datetime.now(),
            source_id="O2-PRIMARY",
        )

        report = validator.validate(dp)

        assert report.result == expected_result, \
            f"Value {value} should be {expected_result.value}"

    @pytest.mark.parametrize("value,expected_result", [
        (50.0, ValidationResult.VALID),
        (0.0, ValidationResult.VALID),
        (4999.0, ValidationResult.VALID),
        (-10.0, ValidationResult.INVALID),
        (5500.0, ValidationResult.INVALID),
    ])
    def test_co_range_validation(
        self,
        validator: StreamingDataValidator,
        value: float,
        expected_result: ValidationResult
    ):
        """Test CO range validation."""
        dp = DataPoint(
            parameter="CO",
            value=value,
            timestamp=datetime.now(),
            source_id="CO-PRIMARY",
        )

        report = validator.validate(dp)

        assert report.result == expected_result


# =============================================================================
# RATE OF CHANGE TESTS
# =============================================================================

class TestRateOfChange:
    """Test rate-of-change validation."""

    def test_normal_rate_of_change(self, validator: StreamingDataValidator):
        """Test normal rate of change passes."""
        # 1% O2 per second is within limit of 2%/s
        points = create_data_points("O2", [3.0, 3.5, 4.0, 4.5, 5.0])

        for dp in points:
            report = validator.validate(dp)
            assert report.result == ValidationResult.VALID

    def test_excessive_rate_of_change(self, validator: StreamingDataValidator):
        """Test excessive rate of change detected."""
        # First establish baseline
        dp1 = DataPoint(
            parameter="O2",
            value=3.0,
            timestamp=datetime.now(),
            source_id="O2-PRIMARY",
        )
        validator.validate(dp1)

        # Then jump 5% in 1 second (exceeds 2%/s limit)
        dp2 = DataPoint(
            parameter="O2",
            value=8.0,
            timestamp=dp1.timestamp + timedelta(seconds=1),
            source_id="O2-PRIMARY",
        )
        report = validator.validate(dp2)

        assert report.result == ValidationResult.INVALID
        assert any("rate of change" in issue.lower() for issue in report.issues)

    def test_rate_of_change_with_time_gap(self, validator: StreamingDataValidator):
        """Test rate of change calculation accounts for time."""
        # First point
        dp1 = DataPoint(
            parameter="O2",
            value=3.0,
            timestamp=datetime.now(),
            source_id="O2-PRIMARY",
        )
        validator.validate(dp1)

        # Second point 10 seconds later with 5% change
        # Rate = 5% / 10s = 0.5%/s, which is under 2%/s limit
        dp2 = DataPoint(
            parameter="O2",
            value=8.0,
            timestamp=dp1.timestamp + timedelta(seconds=10),
            source_id="O2-PRIMARY",
        )
        report = validator.validate(dp2)

        assert report.result == ValidationResult.VALID


# =============================================================================
# GAP DETECTION TESTS
# =============================================================================

class TestGapDetection:
    """Test data gap detection."""

    def test_no_gap_detected(self, validator: StreamingDataValidator):
        """Test normal data rate has no gap."""
        points = create_data_points("O2", [3.0, 3.1, 3.2], interval_seconds=2.0)

        for dp in points:
            report = validator.validate(dp)
            gap_issues = [i for i in report.issues if "gap" in i.lower()]
            assert len(gap_issues) == 0

    def test_gap_detected(self, validator: StreamingDataValidator):
        """Test gap is detected when data exceeds max gap."""
        dp1 = DataPoint(
            parameter="O2",
            value=3.0,
            timestamp=datetime.now(),
            source_id="O2-PRIMARY",
        )
        validator.validate(dp1)

        # Second point 10 seconds later (exceeds 5s max gap)
        dp2 = DataPoint(
            parameter="O2",
            value=3.1,
            timestamp=dp1.timestamp + timedelta(seconds=10),
            source_id="O2-PRIMARY",
        )
        report = validator.validate(dp2)

        gap_issues = [i for i in report.issues if "gap" in i.lower()]
        assert len(gap_issues) > 0

    def test_flame_signal_fast_gap_detection(self, validator: StreamingDataValidator):
        """Test flame signal has faster gap detection."""
        dp1 = DataPoint(
            parameter="flame_signal",
            value=10.0,
            timestamp=datetime.now(),
            source_id="FLAME-PRIMARY",
        )
        validator.validate(dp1)

        # Second point 2 seconds later (exceeds 1s max gap for flame)
        dp2 = DataPoint(
            parameter="flame_signal",
            value=10.1,
            timestamp=dp1.timestamp + timedelta(seconds=2),
            source_id="FLAME-PRIMARY",
        )
        report = validator.validate(dp2)

        gap_issues = [i for i in report.issues if "gap" in i.lower()]
        assert len(gap_issues) > 0


# =============================================================================
# SPIKE DETECTION TESTS
# =============================================================================

class TestSpikeDetection:
    """Test spike (outlier) detection."""

    def test_no_spike_in_normal_data(self, validator: StreamingDataValidator):
        """Test no spike detected in normal data."""
        random.seed(42)

        # Build up window with normal variation
        base_time = datetime.now()
        for i in range(20):
            dp = DataPoint(
                parameter="O2",
                value=3.0 + random.gauss(0, 0.1),
                timestamp=base_time + timedelta(seconds=i),
                source_id="O2-PRIMARY",
            )
            report = validator.validate(dp)

        # Normal value should not be flagged as spike
        spike_issues = [i for i in report.issues if "spike" in i.lower()]
        assert len(spike_issues) == 0

    def test_spike_detected(self, validator: StreamingDataValidator):
        """Test spike is detected for outlier value."""
        random.seed(42)

        # Build up window with stable values
        base_time = datetime.now()
        for i in range(15):
            dp = DataPoint(
                parameter="O2",
                value=3.0 + random.gauss(0, 0.05),  # Very stable
                timestamp=base_time + timedelta(seconds=i),
                source_id="O2-PRIMARY",
            )
            validator.validate(dp)

        # Now send a spike
        spike_dp = DataPoint(
            parameter="O2",
            value=8.0,  # Way outside normal range
            timestamp=base_time + timedelta(seconds=16),
            source_id="O2-PRIMARY",
        )
        report = validator.validate(spike_dp)

        spike_issues = [i for i in report.issues if "spike" in i.lower()]
        assert len(spike_issues) > 0


# =============================================================================
# FROZEN VALUE TESTS
# =============================================================================

class TestFrozenValueDetection:
    """Test frozen value detection."""

    def test_frozen_value_detected(self, validator: StreamingDataValidator):
        """Test frozen value is detected."""
        base_time = datetime.now()

        # Send 15 identical values over 35 seconds
        for i in range(15):
            dp = DataPoint(
                parameter="O2",
                value=3.5,  # Exactly same value
                timestamp=base_time + timedelta(seconds=i * 3),
                source_id="O2-PRIMARY",
            )
            report = validator.validate(dp)

        frozen_issues = [i for i in report.issues if "frozen" in i.lower()]
        assert len(frozen_issues) > 0

    def test_small_variation_not_frozen(self, validator: StreamingDataValidator):
        """Test small variation is not detected as frozen."""
        random.seed(42)
        base_time = datetime.now()

        for i in range(15):
            dp = DataPoint(
                parameter="O2",
                value=3.5 + random.gauss(0, 0.01),  # Very small variation
                timestamp=base_time + timedelta(seconds=i * 3),
                source_id="O2-PRIMARY",
            )
            report = validator.validate(dp)

        frozen_issues = [i for i in report.issues if "frozen" in i.lower()]
        assert len(frozen_issues) == 0


# =============================================================================
# BATCH VALIDATION TESTS
# =============================================================================

class TestBatchValidation:
    """Test batch validation."""

    def test_batch_validation(self, validator: StreamingDataValidator):
        """Test batch validation of multiple points."""
        points = create_data_points("O2", [3.0, 3.1, 3.2, 3.3, 3.4])

        reports = validator.validate_batch(points)

        assert len(reports) == 5
        assert all(r.result == ValidationResult.VALID for r in reports)

    def test_batch_mixed_validity(self, validator: StreamingDataValidator):
        """Test batch with mix of valid and invalid."""
        base_time = datetime.now()

        points = [
            DataPoint("O2", 3.0, base_time, "O2-PRIMARY"),
            DataPoint("O2", 3.1, base_time + timedelta(seconds=1), "O2-PRIMARY"),
            DataPoint("O2", 30.0, base_time + timedelta(seconds=2), "O2-PRIMARY"),  # Out of range
        ]

        reports = validator.validate_batch(points)

        assert len(reports) == 3
        assert reports[0].result == ValidationResult.VALID
        assert reports[1].result == ValidationResult.VALID
        assert reports[2].result == ValidationResult.INVALID


# =============================================================================
# WINDOW STATISTICS TESTS
# =============================================================================

class TestWindowStatistics:
    """Test window statistics calculation."""

    def test_window_statistics(self, validator: StreamingDataValidator):
        """Test window statistics are calculated correctly."""
        # Add enough samples
        base_time = datetime.now()
        for i in range(20):
            dp = DataPoint(
                parameter="O2",
                value=3.0 + (i * 0.1),
                timestamp=base_time + timedelta(seconds=i),
                source_id="O2-PRIMARY",
            )
            validator.validate(dp)

        stats = validator.get_window_statistics("O2")

        assert stats is not None
        assert stats["count"] == 20
        assert 3.9 <= stats["mean"] <= 4.0
        assert stats["min"] == 3.0
        assert stats["max"] == 4.9

    def test_unknown_parameter_statistics(self, validator: StreamingDataValidator):
        """Test statistics for unknown parameter returns None."""
        stats = validator.get_window_statistics("UNKNOWN")
        assert stats is None


# =============================================================================
# QUALITY FLAG TESTS
# =============================================================================

class TestQualityFlags:
    """Test quality flag assignment."""

    def test_good_quality_for_valid(self, validator: StreamingDataValidator):
        """Test GOOD quality for valid data."""
        dp = DataPoint(
            parameter="O2",
            value=3.5,
            timestamp=datetime.now(),
            source_id="O2-PRIMARY",
        )

        report = validator.validate(dp)

        assert report.quality_assigned == QualityFlag.GOOD

    def test_bad_quality_for_invalid(self, validator: StreamingDataValidator):
        """Test BAD quality for invalid data."""
        dp = DataPoint(
            parameter="O2",
            value=30.0,  # Out of range
            timestamp=datetime.now(),
            source_id="O2-PRIMARY",
        )

        report = validator.validate(dp)

        assert report.quality_assigned == QualityFlag.BAD


# =============================================================================
# STREAMING TESTS INIT
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
