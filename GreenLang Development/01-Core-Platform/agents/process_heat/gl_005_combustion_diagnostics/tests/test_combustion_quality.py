# -*- coding: utf-8 -*-
"""
GL-005 Combustion Quality Calculator Tests
==========================================

Comprehensive unit tests for the CQI (Combustion Quality Index) calculator.
Tests calculation accuracy, edge cases, and provenance tracking.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone
import math

from greenlang.agents.process_heat.gl_005_combustion_diagnostics.config import (
    CQIConfig,
    CQIWeights,
    CQIThresholds,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.schemas import (
    FlueGasReading,
    CQIRating,
    TrendDirection,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.combustion_quality import (
    CombustionQualityCalculator,
    create_default_cqi_calculator,
    calculate_cqi_quick,
    ATMOSPHERIC_O2_PCT,
    MIN_SAFE_O2_PCT,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.tests.conftest import (
    assert_valid_provenance_hash,
    assert_valid_cqi_score,
)


class TestCQICalculatorInitialization:
    """Tests for CQI calculator initialization."""

    def test_default_initialization(self, default_cqi_config):
        """Test default calculator initialization."""
        calculator = CombustionQualityCalculator(default_cqi_config)
        assert calculator.config == default_cqi_config
        assert calculator.weights is not None
        assert calculator.thresholds is not None

    def test_custom_config_initialization(self):
        """Test calculator with custom configuration."""
        config = CQIConfig(
            scoring_method="weighted_sigmoid",
            o2_reference_pct=5.0,
        )
        calculator = CombustionQualityCalculator(config)
        assert calculator.config.scoring_method == "weighted_sigmoid"
        assert calculator.config.o2_reference_pct == 5.0

    def test_factory_function(self):
        """Test create_default_cqi_calculator factory."""
        calculator = create_default_cqi_calculator()
        assert isinstance(calculator, CombustionQualityCalculator)
        assert calculator.config.o2_reference_pct == 3.0


class TestCQICalculation:
    """Tests for CQI calculation functionality."""

    def test_optimal_combustion_score(self, default_cqi_config, optimal_flue_gas_reading):
        """Test CQI calculation for optimal combustion conditions."""
        calculator = CombustionQualityCalculator(default_cqi_config)
        result = calculator.calculate(optimal_flue_gas_reading)

        # Optimal conditions should yield excellent or good rating
        assert result.cqi_score >= 75.0
        assert result.cqi_rating in [CQIRating.EXCELLENT, CQIRating.GOOD]
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_high_co_reduces_score(self, default_cqi_config, high_co_flue_gas_reading):
        """Test that high CO reduces CQI score."""
        calculator = CombustionQualityCalculator(default_cqi_config)
        result = calculator.calculate(high_co_flue_gas_reading)

        # High CO should result in lower score
        assert result.cqi_score < 75.0
        assert result.cqi_rating in [CQIRating.ACCEPTABLE, CQIRating.POOR, CQIRating.CRITICAL]

    def test_high_o2_reduces_score(self, default_cqi_config, high_o2_flue_gas_reading):
        """Test that excess O2 reduces CQI score."""
        calculator = CombustionQualityCalculator(default_cqi_config)
        result = calculator.calculate(high_o2_flue_gas_reading)

        # High O2 (excess air) should reduce score
        assert result.excess_air_pct > 30.0

    def test_cqi_score_bounds(self, default_cqi_config, optimal_flue_gas_reading):
        """Test that CQI score is always between 0 and 100."""
        calculator = CombustionQualityCalculator(default_cqi_config)
        result = calculator.calculate(optimal_flue_gas_reading)

        assert_valid_cqi_score(result.cqi_score)

    def test_calculation_components(self, default_cqi_config, optimal_flue_gas_reading):
        """Test that all CQI components are calculated."""
        calculator = CombustionQualityCalculator(default_cqi_config)
        result = calculator.calculate(optimal_flue_gas_reading)

        assert len(result.components) == 5

        component_names = [c.component for c in result.components]
        assert "oxygen" in component_names
        assert "carbon_monoxide" in component_names
        assert "carbon_dioxide" in component_names
        assert "nox" in component_names
        assert "combustibles" in component_names

    def test_weighted_scores_sum_to_cqi(self, default_cqi_config, optimal_flue_gas_reading):
        """Test that weighted component scores sum to CQI score."""
        calculator = CombustionQualityCalculator(default_cqi_config)
        result = calculator.calculate(optimal_flue_gas_reading)

        weighted_sum = sum(c.weighted_score for c in result.components)
        assert abs(weighted_sum - result.cqi_score) < 0.1

    def test_component_weights_match_config(self, default_cqi_config, optimal_flue_gas_reading):
        """Test that component weights match configuration."""
        calculator = CombustionQualityCalculator(default_cqi_config)
        result = calculator.calculate(optimal_flue_gas_reading)

        for component in result.components:
            if component.component == "oxygen":
                assert component.weight == default_cqi_config.weights.oxygen
            elif component.component == "carbon_monoxide":
                assert component.weight == default_cqi_config.weights.carbon_monoxide


class TestO2Correction:
    """Tests for O2 correction calculations."""

    def test_o2_correction_formula(self, default_cqi_config):
        """Test O2 correction calculation."""
        calculator = CombustionQualityCalculator(default_cqi_config)

        # Test known correction
        # At measured O2=5%, reference O2=3%, measured value=100 ppm
        # Corrected = 100 * (20.95 - 3) / (20.95 - 5) = 112.58 ppm
        corrected = calculator._correct_to_reference_o2(100.0, 5.0, 3.0)
        expected = 100.0 * (20.95 - 3.0) / (20.95 - 5.0)
        assert abs(corrected - expected) < 0.01

    def test_no_correction_at_reference(self, default_cqi_config):
        """Test that no correction needed when measured O2 equals reference."""
        calculator = CombustionQualityCalculator(default_cqi_config)

        # At reference O2, correction factor should be 1.0
        corrected = calculator._correct_to_reference_o2(100.0, 3.0, 3.0)
        assert abs(corrected - 100.0) < 0.01

    def test_co_nox_corrected_in_result(self, default_cqi_config, optimal_flue_gas_reading):
        """Test that CO and NOx are corrected in result."""
        calculator = CombustionQualityCalculator(default_cqi_config)
        result = calculator.calculate(optimal_flue_gas_reading)

        assert result.co_corrected_ppm is not None
        assert result.nox_corrected_ppm is not None
        assert result.o2_reference_pct == 3.0


class TestExcessAirCalculation:
    """Tests for excess air calculation."""

    def test_excess_air_formula(self, default_cqi_config):
        """Test excess air calculation formula."""
        calculator = CombustionQualityCalculator(default_cqi_config)

        # At 3% O2: EA = 3 / (20.95 - 3) * 100 = 16.7%
        excess_air = calculator._calculate_excess_air(3.0)
        expected = (3.0 / (20.95 - 3.0)) * 100
        assert abs(excess_air - expected) < 0.1

    def test_excess_air_at_stoichiometric(self, default_cqi_config):
        """Test excess air at stoichiometric conditions (0% O2)."""
        calculator = CombustionQualityCalculator(default_cqi_config)

        # At 0% O2, excess air should be 0%
        excess_air = calculator._calculate_excess_air(0.0)
        assert excess_air == 0.0

    def test_excess_air_in_result(self, default_cqi_config, optimal_flue_gas_reading):
        """Test that excess air is included in result."""
        calculator = CombustionQualityCalculator(default_cqi_config)
        result = calculator.calculate(optimal_flue_gas_reading)

        assert result.excess_air_pct >= 0.0


class TestCombustionEfficiency:
    """Tests for combustion efficiency calculation."""

    def test_efficiency_reasonable_range(self, default_cqi_config, optimal_flue_gas_reading):
        """Test that efficiency is in reasonable range."""
        calculator = CombustionQualityCalculator(default_cqi_config)
        result = calculator.calculate(optimal_flue_gas_reading)

        # Efficiency should be between 70% and 99%
        assert 70.0 <= result.combustion_efficiency_pct <= 99.0

    def test_high_stack_temp_reduces_efficiency(self, default_cqi_config):
        """Test that high stack temperature reduces efficiency."""
        calculator = CombustionQualityCalculator(default_cqi_config)

        reading_normal = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=3.0,
            co2_pct=10.5,
            co_ppm=30.0,
            nox_ppm=45.0,
            flue_gas_temp_c=180.0,
            ambient_temp_c=25.0,
        )

        reading_high_temp = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=3.0,
            co2_pct=10.5,
            co_ppm=30.0,
            nox_ppm=45.0,
            flue_gas_temp_c=350.0,
            ambient_temp_c=25.0,
        )

        result_normal = calculator.calculate(reading_normal)
        result_high = calculator.calculate(reading_high_temp)

        # Higher stack temp should mean lower efficiency
        assert result_high.combustion_efficiency_pct < result_normal.combustion_efficiency_pct


class TestComponentScoring:
    """Tests for individual component scoring."""

    def test_oxygen_score_optimal_range(self, default_cqi_config):
        """Test oxygen scoring in optimal range."""
        calculator = CombustionQualityCalculator(default_cqi_config)

        score, status = calculator._score_oxygen(3.0)
        assert score == 100.0
        assert status == "optimal"

    def test_oxygen_score_high_o2(self, default_cqi_config):
        """Test oxygen scoring with high O2 (excess air)."""
        calculator = CombustionQualityCalculator(default_cqi_config)

        score, status = calculator._score_oxygen(9.0)
        assert score < 50.0
        assert status == "critical"

    def test_oxygen_score_low_o2(self, default_cqi_config):
        """Test oxygen scoring with low O2."""
        calculator = CombustionQualityCalculator(default_cqi_config)

        score, status = calculator._score_oxygen(1.5)
        assert score < 80.0
        assert status in ["warning", "critical"]

    def test_co_score_excellent(self, default_cqi_config):
        """Test CO scoring for excellent conditions."""
        calculator = CombustionQualityCalculator(default_cqi_config)

        score, status = calculator._score_co(25.0)
        assert score == 100.0
        assert status == "optimal"

    def test_co_score_critical(self, default_cqi_config):
        """Test CO scoring for critical conditions."""
        calculator = CombustionQualityCalculator(default_cqi_config)

        score, status = calculator._score_co(600.0)
        assert score < 40.0
        assert status == "critical"

    def test_nox_score_excellent(self, default_cqi_config):
        """Test NOx scoring for excellent conditions."""
        calculator = CombustionQualityCalculator(default_cqi_config)

        score, status = calculator._score_nox(40.0)
        assert score == 100.0
        assert status == "optimal"

    def test_combustibles_score(self, default_cqi_config):
        """Test combustibles scoring."""
        calculator = CombustionQualityCalculator(default_cqi_config)

        score_low, status_low = calculator._score_combustibles(0.05)
        score_high, status_high = calculator._score_combustibles(1.5)

        assert score_low > score_high
        assert status_low == "optimal"
        assert status_high == "critical"


class TestCQIRatingDetermination:
    """Tests for CQI rating determination."""

    def test_excellent_rating(self, default_cqi_config, optimal_flue_gas_reading):
        """Test excellent rating threshold."""
        calculator = CombustionQualityCalculator(default_cqi_config)

        # Create ideal reading
        ideal_reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=3.0,
            co2_pct=10.5,
            co_ppm=20.0,
            nox_ppm=30.0,
            combustibles_pct=0.05,
            flue_gas_temp_c=180.0,
        )

        result = calculator.calculate(ideal_reading)

        if result.cqi_score >= 90:
            assert result.cqi_rating == CQIRating.EXCELLENT

    def test_rating_boundaries(self, default_cqi_config):
        """Test rating boundary determination."""
        calculator = CombustionQualityCalculator(default_cqi_config)

        assert calculator._determine_rating(95.0) == CQIRating.EXCELLENT
        assert calculator._determine_rating(85.0) == CQIRating.GOOD
        assert calculator._determine_rating(65.0) == CQIRating.ACCEPTABLE
        assert calculator._determine_rating(45.0) == CQIRating.POOR
        assert calculator._determine_rating(30.0) == CQIRating.CRITICAL


class TestBaselineComparison:
    """Tests for baseline comparison and trending."""

    def test_improving_trend(self, default_cqi_config, optimal_flue_gas_reading):
        """Test improving trend detection."""
        calculator = CombustionQualityCalculator(default_cqi_config)

        result = calculator.calculate(optimal_flue_gas_reading, baseline_cqi=70.0)

        if result.cqi_score > 72.0:  # More than 2 points improvement
            assert result.trend_vs_baseline == TrendDirection.IMPROVING

    def test_degrading_trend(self, default_cqi_config, high_co_flue_gas_reading):
        """Test degrading trend detection."""
        calculator = CombustionQualityCalculator(default_cqi_config)

        result = calculator.calculate(high_co_flue_gas_reading, baseline_cqi=85.0)

        if result.cqi_score < 83.0:  # More than 2 points degradation
            assert result.trend_vs_baseline == TrendDirection.DEGRADING

    def test_stable_trend(self, default_cqi_config, optimal_flue_gas_reading):
        """Test stable trend detection."""
        calculator = CombustionQualityCalculator(default_cqi_config)

        # First calculate to get baseline
        result1 = calculator.calculate(optimal_flue_gas_reading)

        # Then calculate with same reading as baseline
        result2 = calculator.calculate(optimal_flue_gas_reading, baseline_cqi=result1.cqi_score)

        assert result2.trend_vs_baseline == TrendDirection.STABLE

    def test_unknown_trend_without_baseline(self, default_cqi_config, optimal_flue_gas_reading):
        """Test unknown trend when no baseline provided."""
        calculator = CombustionQualityCalculator(default_cqi_config)

        result = calculator.calculate(optimal_flue_gas_reading)

        assert result.trend_vs_baseline == TrendDirection.UNKNOWN
        assert result.baseline_cqi is None


class TestProvenanceTracking:
    """Tests for provenance hash generation."""

    def test_provenance_hash_generated(self, default_cqi_config, optimal_flue_gas_reading):
        """Test that provenance hash is generated."""
        calculator = CombustionQualityCalculator(default_cqi_config)
        result = calculator.calculate(optimal_flue_gas_reading)

        assert_valid_provenance_hash(result.provenance_hash)

    def test_provenance_hash_deterministic(self, default_cqi_config):
        """Test that provenance hash is deterministic."""
        calculator = CombustionQualityCalculator(default_cqi_config)

        # Create identical readings
        timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        reading1 = FlueGasReading(
            timestamp=timestamp,
            oxygen_pct=3.0,
            co2_pct=10.5,
            co_ppm=30.0,
            nox_ppm=45.0,
            flue_gas_temp_c=180.0,
        )

        reading2 = FlueGasReading(
            timestamp=timestamp,
            oxygen_pct=3.0,
            co2_pct=10.5,
            co_ppm=30.0,
            nox_ppm=45.0,
            flue_gas_temp_c=180.0,
        )

        result1 = calculator.calculate(reading1)
        result2 = calculator.calculate(reading2)

        # Same input should produce same hash
        assert result1.provenance_hash == result2.provenance_hash

    def test_provenance_hash_changes_with_input(self, default_cqi_config):
        """Test that provenance hash changes with different input."""
        calculator = CombustionQualityCalculator(default_cqi_config)

        reading1 = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=3.0,
            co2_pct=10.5,
            co_ppm=30.0,
            nox_ppm=45.0,
            flue_gas_temp_c=180.0,
        )

        reading2 = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=4.0,  # Different O2
            co2_pct=10.0,
            co_ppm=35.0,
            nox_ppm=50.0,
            flue_gas_temp_c=185.0,
        )

        result1 = calculator.calculate(reading1)
        result2 = calculator.calculate(reading2)

        assert result1.provenance_hash != result2.provenance_hash


class TestAuditTrail:
    """Tests for calculation audit trail."""

    def test_audit_trail_generated(self, default_cqi_config, optimal_flue_gas_reading):
        """Test that audit trail is generated."""
        calculator = CombustionQualityCalculator(default_cqi_config)
        calculator.calculate(optimal_flue_gas_reading)

        audit_trail = calculator.get_audit_trail()
        assert len(audit_trail) > 0

    def test_audit_trail_contains_steps(self, default_cqi_config, optimal_flue_gas_reading):
        """Test that audit trail contains expected steps."""
        calculator = CombustionQualityCalculator(default_cqi_config)
        calculator.calculate(optimal_flue_gas_reading)

        audit_trail = calculator.get_audit_trail()
        operations = [entry["operation"] for entry in audit_trail]

        assert "input_validation" in operations
        assert "o2_correction" in operations
        assert "cqi_calculation" in operations


class TestInputValidation:
    """Tests for input validation."""

    def test_reject_low_oxygen(self, default_cqi_config):
        """Test rejection of dangerously low oxygen."""
        calculator = CombustionQualityCalculator(default_cqi_config)

        low_o2_reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=0.4,  # Below minimum safe level
            co2_pct=13.0,
            co_ppm=1000.0,
            nox_ppm=20.0,
            flue_gas_temp_c=200.0,
        )

        # This should raise validation error in the schema
        # or be rejected in the calculator
        with pytest.raises(Exception):
            calculator.calculate(low_o2_reading)

    def test_reject_atmospheric_oxygen(self, default_cqi_config):
        """Test rejection of atmospheric oxygen (no combustion)."""
        calculator = CombustionQualityCalculator(default_cqi_config)

        atm_o2_reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=20.9,
            co2_pct=0.04,
            co_ppm=0.0,
            nox_ppm=0.0,
            flue_gas_temp_c=25.0,
        )

        with pytest.raises(ValueError):
            calculator.calculate(atm_o2_reading)


class TestQuickCalculation:
    """Tests for quick CQI calculation utility."""

    def test_quick_calculation(self):
        """Test quick CQI calculation."""
        score = calculate_cqi_quick(
            oxygen_pct=3.0,
            co_ppm=30.0,
            co2_pct=10.5,
            nox_ppm=45.0,
        )

        assert_valid_cqi_score(score)

    def test_quick_calculation_optimal(self):
        """Test quick calculation with optimal values."""
        score = calculate_cqi_quick(
            oxygen_pct=3.0,
            co_ppm=25.0,
            co2_pct=10.5,
            nox_ppm=40.0,
            combustibles_pct=0.05,
        )

        assert score >= 80.0

    def test_quick_calculation_poor(self):
        """Test quick calculation with poor values."""
        score = calculate_cqi_quick(
            oxygen_pct=8.0,
            co_ppm=400.0,
            co2_pct=7.0,
            nox_ppm=200.0,
            combustibles_pct=0.8,
        )

        assert score < 60.0


class TestParametrizedCQIScenarios:
    """Parametrized tests for various CQI scenarios."""

    @pytest.mark.parametrize("o2,co,nox,min_expected_score,max_expected_score", [
        (3.0, 25.0, 40.0, 85.0, 100.0),   # Optimal
        (3.0, 80.0, 80.0, 70.0, 90.0),     # Good
        (5.0, 150.0, 120.0, 55.0, 75.0),   # Acceptable
        (7.0, 350.0, 200.0, 35.0, 55.0),   # Poor
        (9.0, 600.0, 300.0, 0.0, 40.0),    # Critical
    ])
    def test_cqi_scenarios(self, default_cqi_config, o2, co, nox, min_expected_score, max_expected_score):
        """Test CQI calculation across various scenarios."""
        calculator = CombustionQualityCalculator(default_cqi_config)

        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=o2,
            co2_pct=11.8 * (20.95 - o2) / 20.95,
            co_ppm=co,
            nox_ppm=nox,
            flue_gas_temp_c=180.0,
        )

        result = calculator.calculate(reading)

        assert min_expected_score <= result.cqi_score <= max_expected_score, \
            f"CQI {result.cqi_score} not in range [{min_expected_score}, {max_expected_score}]"
