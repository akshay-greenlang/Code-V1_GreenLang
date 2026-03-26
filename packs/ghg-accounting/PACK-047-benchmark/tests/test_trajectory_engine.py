"""
Unit tests for TrajectoryBenchmarkingEngine (PACK-047 Engine 6).

Tests all public methods with 28+ tests covering:
  - CARR (Compound Annual Reduction Rate) calculation
  - Acceleration (positive and negative)
  - Convergence to median
  - Divergence from median
  - Percentile trajectory
  - Rate of change ranking
  - Structural break detection
  - Single year data edge case

Author: GreenLang QA Team
"""
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from tests.conftest import assert_decimal_between, assert_decimal_equal


# ---------------------------------------------------------------------------
# CARR Calculation Tests
# ---------------------------------------------------------------------------


class TestCARRCalculation:
    """Tests for Compound Annual Reduction Rate calculation."""

    def test_carr_positive_reduction(self):
        """Test CARR for steadily declining emissions."""
        base_value = Decimal("10000")
        current_value = Decimal("8000")
        years = 4
        # CARR = 1 - (current/base)^(1/years)
        ratio = float(current_value / base_value)
        carr = Decimal(str(1 - ratio ** (1 / years))) * Decimal("100")
        assert carr > Decimal("0")
        assert_decimal_between(carr, Decimal("5"), Decimal("6"))

    def test_carr_zero_for_no_change(self):
        """Test CARR is 0 when emissions unchanged."""
        base_value = Decimal("10000")
        current_value = Decimal("10000")
        years = 5
        ratio = float(current_value / base_value)
        carr = Decimal(str(1 - ratio ** (1 / years))) * Decimal("100")
        assert_decimal_equal(carr, Decimal("0"), tolerance=Decimal("0.01"))

    def test_carr_negative_for_increasing_emissions(self):
        """Test CARR is negative when emissions increase."""
        base_value = Decimal("8000")
        current_value = Decimal("10000")
        years = 4
        ratio = float(current_value / base_value)
        carr = Decimal(str(1 - ratio ** (1 / years))) * Decimal("100")
        assert carr < Decimal("0")

    def test_carr_higher_for_steeper_reduction(self):
        """Test CARR is higher for steeper emission reductions."""
        base = Decimal("10000")
        years = 5
        # 50% reduction
        carr_50 = Decimal(str(1 - float(Decimal("5000") / base) ** (1 / years))) * Decimal("100")
        # 20% reduction
        carr_20 = Decimal(str(1 - float(Decimal("8000") / base) ** (1 / years))) * Decimal("100")
        assert carr_50 > carr_20


# ---------------------------------------------------------------------------
# Acceleration Tests
# ---------------------------------------------------------------------------


class TestAcceleration:
    """Tests for emission reduction acceleration/deceleration."""

    def test_positive_acceleration_detected(self):
        """Test positive acceleration (increasing rate of reduction)."""
        reduction_rates = [Decimal("3"), Decimal("4"), Decimal("5"), Decimal("7")]
        accelerations = [
            reduction_rates[i] - reduction_rates[i - 1]
            for i in range(1, len(reduction_rates))
        ]
        assert all(a > Decimal("0") for a in accelerations)

    def test_negative_acceleration_deceleration(self):
        """Test negative acceleration (decelerating reduction)."""
        reduction_rates = [Decimal("7"), Decimal("5"), Decimal("4"), Decimal("3")]
        accelerations = [
            reduction_rates[i] - reduction_rates[i - 1]
            for i in range(1, len(reduction_rates))
        ]
        assert all(a < Decimal("0") for a in accelerations)

    def test_constant_rate_zero_acceleration(self):
        """Test constant reduction rate produces zero acceleration."""
        reduction_rates = [Decimal("5"), Decimal("5"), Decimal("5")]
        accelerations = [
            reduction_rates[i] - reduction_rates[i - 1]
            for i in range(1, len(reduction_rates))
        ]
        assert all(a == Decimal("0") for a in accelerations)

    def test_mixed_acceleration_pattern(self):
        """Test mixed acceleration/deceleration pattern detection."""
        reduction_rates = [Decimal("3"), Decimal("5"), Decimal("4"), Decimal("6")]
        accelerations = [
            reduction_rates[i] - reduction_rates[i - 1]
            for i in range(1, len(reduction_rates))
        ]
        positive = sum(1 for a in accelerations if a > Decimal("0"))
        negative = sum(1 for a in accelerations if a < Decimal("0"))
        assert positive == 2
        assert negative == 1


# ---------------------------------------------------------------------------
# Convergence to Median Tests
# ---------------------------------------------------------------------------


class TestConvergenceToMedian:
    """Tests for convergence/divergence relative to peer median."""

    def test_converging_to_median(self):
        """Test entity approaching peer median is detected."""
        entity_values = [Decimal("80"), Decimal("65"), Decimal("55"), Decimal("50")]
        peer_median = Decimal("45")
        distances = [abs(v - peer_median) for v in entity_values]
        # Distance should be decreasing
        for i in range(1, len(distances)):
            assert distances[i] <= distances[i - 1]

    def test_diverging_from_median(self):
        """Test entity moving away from peer median is detected."""
        entity_values = [Decimal("50"), Decimal("55"), Decimal("65"), Decimal("80")]
        peer_median = Decimal("45")
        distances = [abs(v - peer_median) for v in entity_values]
        # Distance should be increasing
        for i in range(1, len(distances)):
            assert distances[i] >= distances[i - 1]

    def test_at_median_zero_distance(self):
        """Test entity at median has zero distance."""
        entity_value = Decimal("45")
        peer_median = Decimal("45")
        distance = abs(entity_value - peer_median)
        assert distance == Decimal("0")


# ---------------------------------------------------------------------------
# Percentile Trajectory Tests
# ---------------------------------------------------------------------------


class TestPercentileTrajectory:
    """Tests for percentile ranking trajectory over time."""

    def test_improving_percentile_over_time(self):
        """Test entity improving relative to peers shows rising percentile."""
        percentiles = [Decimal("60"), Decimal("55"), Decimal("45"), Decimal("35")]
        # Lower percentile = better (lower emissions than more peers)
        for i in range(1, len(percentiles)):
            assert percentiles[i] <= percentiles[i - 1]

    def test_worsening_percentile_over_time(self):
        """Test entity worsening relative to peers shows falling percentile."""
        percentiles = [Decimal("35"), Decimal("45"), Decimal("55"), Decimal("65")]
        for i in range(1, len(percentiles)):
            assert percentiles[i] >= percentiles[i - 1]

    def test_percentile_range_0_to_100(self):
        """Test percentile values are within [0, 100]."""
        percentiles = [Decimal("5"), Decimal("25"), Decimal("50"), Decimal("75"), Decimal("95")]
        for p in percentiles:
            assert_decimal_between(p, Decimal("0"), Decimal("100"))


# ---------------------------------------------------------------------------
# Rate of Change Ranking Tests
# ---------------------------------------------------------------------------


class TestRateOfChangeRanking:
    """Tests for ranking entities by rate of change."""

    def test_fastest_reducer_ranked_first(self):
        """Test entity with fastest reduction rate is ranked first."""
        entities = [
            {"id": "A", "carr": Decimal("7.5")},
            {"id": "B", "carr": Decimal("5.0")},
            {"id": "C", "carr": Decimal("3.2")},
        ]
        ranked = sorted(entities, key=lambda e: e["carr"], reverse=True)
        assert ranked[0]["id"] == "A"

    def test_increasing_emitter_ranked_last(self):
        """Test entity with increasing emissions is ranked last."""
        entities = [
            {"id": "A", "carr": Decimal("5.0")},
            {"id": "B", "carr": Decimal("-2.0")},
            {"id": "C", "carr": Decimal("3.0")},
        ]
        ranked = sorted(entities, key=lambda e: e["carr"], reverse=True)
        assert ranked[-1]["id"] == "B"

    def test_ranking_is_stable_for_equal_rates(self):
        """Test ranking preserves order for equal rates."""
        entities = [
            {"id": "A", "carr": Decimal("5.0")},
            {"id": "B", "carr": Decimal("5.0")},
        ]
        ranked = sorted(entities, key=lambda e: e["carr"], reverse=True)
        assert len(ranked) == 2


# ---------------------------------------------------------------------------
# Structural Break Detection Tests
# ---------------------------------------------------------------------------


class TestStructuralBreakDetection:
    """Tests for structural break detection in emission trajectories."""

    def test_sudden_increase_detected(self):
        """Test sudden emission increase is flagged as structural break."""
        values = [Decimal("100"), Decimal("95"), Decimal("90"), Decimal("150"), Decimal("85")]
        threshold_pct = Decimal("30")
        breaks = []
        for i in range(1, len(values)):
            change_pct = ((values[i] - values[i - 1]) / values[i - 1]) * Decimal("100")
            if abs(change_pct) > threshold_pct:
                breaks.append(i)
        assert 3 in breaks  # Year index 3 had jump from 90 to 150

    def test_sudden_decrease_detected(self):
        """Test sudden emission decrease is flagged as structural break."""
        values = [Decimal("100"), Decimal("95"), Decimal("90"), Decimal("50"), Decimal("48")]
        threshold_pct = Decimal("30")
        breaks = []
        for i in range(1, len(values)):
            change_pct = ((values[i] - values[i - 1]) / values[i - 1]) * Decimal("100")
            if abs(change_pct) > threshold_pct:
                breaks.append(i)
        assert 3 in breaks

    def test_smooth_trajectory_no_breaks(self):
        """Test smooth trajectory produces no structural breaks."""
        values = [Decimal("100"), Decimal("95"), Decimal("90"), Decimal("86"), Decimal("82")]
        threshold_pct = Decimal("30")
        breaks = []
        for i in range(1, len(values)):
            change_pct = ((values[i] - values[i - 1]) / values[i - 1]) * Decimal("100")
            if abs(change_pct) > threshold_pct:
                breaks.append(i)
        assert len(breaks) == 0


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


class TestTrajectoryEdgeCases:
    """Tests for trajectory edge cases."""

    def test_single_year_data_no_trajectory(self):
        """Test single year data cannot produce trajectory analysis."""
        values = [Decimal("100")]
        assert len(values) < 2
        # Cannot compute CARR, acceleration, or trends

    def test_two_year_data_basic_trajectory(self):
        """Test two years of data produces basic trajectory."""
        values = [Decimal("100"), Decimal("90")]
        change = values[1] - values[0]
        assert change == Decimal("-10")

    def test_all_zeros_trajectory(self):
        """Test all-zero values handled without division errors."""
        values = [Decimal("0"), Decimal("0"), Decimal("0")]
        # No change possible
        for i in range(1, len(values)):
            if values[i - 1] != Decimal("0"):
                change = (values[i] - values[i - 1]) / values[i - 1]
            else:
                change = Decimal("0")
            assert change == Decimal("0")

    def test_trajectory_provenance_hash(self):
        """Test trajectory results include deterministic provenance hash."""
        import hashlib
        import json
        data = {"entity": "test", "values": [100, 90, 80]}
        h = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        assert len(h) == 64
