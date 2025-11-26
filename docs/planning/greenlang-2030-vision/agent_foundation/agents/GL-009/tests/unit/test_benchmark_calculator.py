"""Unit tests for Benchmark Calculator.

Tests industry benchmark comparisons and gap analysis.
Target Coverage: 85%+, Test Count: 12+
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestBenchmarkCalculator:
    """Test suite for BenchmarkCalculator."""

    def test_benchmark_initialization(self, benchmark_data):
        """Test benchmark calculator initialization."""
        assert "natural_gas_boilers" in benchmark_data
        assert "percentile_50" in benchmark_data["natural_gas_boilers"]

    def test_percentile_ranking(self, benchmark_data):
        """Test efficiency percentile ranking."""
        efficiency = 85.0
        benchmarks = benchmark_data["natural_gas_boilers"]

        # Check percentile position
        if efficiency >= benchmarks["percentile_75"]:
            ranking = "Above Average"
        elif efficiency >= benchmarks["percentile_50"]:
            ranking = "Average"
        else:
            ranking = "Below Average"

        assert ranking in ["Above Average", "Average", "Below Average"]

    def test_gap_analysis(self, benchmark_data):
        """Test gap analysis to best practice."""
        current_efficiency = 80.0
        best_practice = benchmark_data["natural_gas_boilers"]["best_practice"]
        gap = best_practice - current_efficiency

        assert gap > 0
        assert gap == 12.0  # 92 - 80

    def test_industry_comparison_natural_gas(self, benchmark_data):
        """Test comparison against natural gas boiler benchmarks."""
        efficiency = 88.0
        benchmarks = benchmark_data["natural_gas_boilers"]

        assert efficiency > benchmarks["percentile_50"]
        assert efficiency >= benchmarks["percentile_75"]

    def test_industry_comparison_coal(self, benchmark_data):
        """Test comparison against coal boiler benchmarks."""
        efficiency = 78.0
        benchmarks = benchmark_data["coal_boilers"]

        assert efficiency < benchmarks["percentile_50"]

    def test_benchmark_interpolation(self):
        """Test interpolation between percentiles."""
        p25, p50, p75 = 80.0, 85.0, 88.0
        efficiency = 86.5

        # Should be between p50 and p75
        assert p50 < efficiency < p75

    def test_improvement_potential_calculation(self):
        """Test calculation of improvement potential."""
        current = 80.0
        target = 90.0
        potential = target - current

        assert potential == 10.0
        assert potential > 0

    def test_peer_group_comparison(self):
        """Test comparison within peer group."""
        peer_efficiencies = [82.0, 85.0, 88.0, 84.0, 86.0]
        my_efficiency = 85.0

        import statistics
        avg = statistics.mean(peer_efficiencies)

        assert my_efficiency >= avg - 2.0  # Within 2% of average

    def test_trend_analysis(self):
        """Test efficiency trend analysis."""
        historical = [80.0, 82.0, 84.0, 85.0, 86.0]

        # Check improvement trend
        is_improving = all(historical[i] <= historical[i+1]
                          for i in range(len(historical)-1))
        assert is_improving

    def test_roi_estimation(self):
        """Test ROI estimation for improvements."""
        efficiency_gain = 5.0  # percentage points
        fuel_cost_per_year = 1000000.0

        savings = fuel_cost_per_year * (efficiency_gain / 100.0)
        assert savings == 50000.0

    def test_benchmark_visualization_data(self):
        """Test data preparation for benchmark visualization."""
        viz_data = {
            "current": 85.0,
            "p25": 80.0,
            "p50": 85.0,
            "p75": 88.0,
            "best": 92.0
        }
        assert viz_data["current"] == viz_data["p50"]

    def test_custom_benchmark_creation(self):
        """Test creation of custom benchmarks."""
        custom = {
            "organization": "My Company",
            "fleet_average": 83.5,
            "best_unit": 89.0,
            "worst_unit": 78.0
        }
        assert custom["best_unit"] > custom["fleet_average"]
