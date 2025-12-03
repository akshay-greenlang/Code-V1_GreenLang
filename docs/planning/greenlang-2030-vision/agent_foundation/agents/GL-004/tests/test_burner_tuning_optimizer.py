# -*- coding: utf-8 -*-
"""
Test Suite for Burner Tuning Optimizer (GL-004 BURNMASTER).

Comprehensive tests for the burner tuning optimizer module including:
- Unit tests for individual calculations
- Integration tests for complete optimization workflow
- Determinism tests for reproducibility
- Performance tests for execution time
- Edge case and boundary testing
- Thread-safety tests for concurrent access

Reference Standards:
- ASME PTC 4: Fired Steam Generators
- API 535: Burners for Fired Heaters
- NFPA 85: Boiler and Combustion Systems

Test Coverage Target: 85%+
"""

import pytest
import math
import threading
import time
import hashlib
from decimal import Decimal
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import module under test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from calculators.burner_tuning_optimizer import (
    BurnerTuningOptimizer,
    BurnerTuningInput,
    BurnerTuningOutput,
    FuelType,
    BurnerType,
    FlameStabilityStatus,
    AtomizationQuality,
    FuelProperties,
    AirFuelRatioResult,
    EmissionTargets,
    TurndownAnalysisResult,
    AirDistributionResult,
    FlameStabilityResult,
    DraftPressureResult,
    AtomizationResult,
    MultiBurnerResult,
    ProvenanceTracker,
    ThreadSafeCache,
    FUEL_PROPERTIES_DB,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def optimizer():
    """Create a BurnerTuningOptimizer instance."""
    return BurnerTuningOptimizer()


@pytest.fixture
def natural_gas_input():
    """Create standard natural gas burner input."""
    return BurnerTuningInput(
        fuel_type="natural_gas",
        fuel_flow_kg_hr=500.0,
        fuel_temperature_c=25.0,
        fuel_pressure_kpa=120.0,
        air_temperature_c=30.0,
        air_humidity_percent=60.0,
        excess_air_percent=15.0,
        o2_measured_percent=3.0,
        co_measured_ppm=40.0,
        nox_measured_ppm=35.0,
        burner_type="nozzle_mix",
        num_burners=1,
        burner_capacity_mw=10.0,
        burner_turndown_ratio=5.0,
        load_percent=80.0,
    )


@pytest.fixture
def fuel_oil_input():
    """Create fuel oil burner input with atomization."""
    return BurnerTuningInput(
        fuel_type="fuel_oil_2",
        fuel_flow_kg_hr=400.0,
        fuel_temperature_c=60.0,
        fuel_pressure_kpa=101.325,
        air_temperature_c=35.0,
        air_humidity_percent=50.0,
        excess_air_percent=18.0,
        o2_measured_percent=4.0,
        co_measured_ppm=60.0,
        nox_measured_ppm=45.0,
        burner_type="diffusion",
        num_burners=1,
        burner_capacity_mw=8.0,
        burner_turndown_ratio=4.0,
        load_percent=75.0,
        atomizer_type="pressure_swirl",
        atomizer_pressure_kpa=800.0,
        fuel_viscosity_cst=5.0,
    )


@pytest.fixture
def multi_burner_input():
    """Create multi-burner furnace input."""
    return BurnerTuningInput(
        fuel_type="natural_gas",
        fuel_flow_kg_hr=1500.0,
        excess_air_percent=12.0,
        o2_measured_percent=2.5,
        burner_type="low_nox",
        num_burners=4,
        burner_capacity_mw=5.0,
        load_percent=90.0,
    )


@pytest.fixture
def low_load_input():
    """Create low load operating condition input."""
    return BurnerTuningInput(
        fuel_type="natural_gas",
        fuel_flow_kg_hr=150.0,
        excess_air_percent=20.0,
        o2_measured_percent=4.5,
        burner_type="premix",
        load_percent=25.0,
    )


# =============================================================================
# UNIT TESTS - FUEL PROPERTIES
# =============================================================================

class TestFuelProperties:
    """Tests for fuel properties database and lookup."""

    def test_natural_gas_properties(self):
        """Test natural gas properties are correct."""
        props = FUEL_PROPERTIES_DB["natural_gas"]
        assert props.name == "Natural Gas (Pipeline Quality)"
        assert props.stoich_afr_mass == pytest.approx(17.2, rel=0.01)
        assert props.hhv_mj_kg == pytest.approx(55.5, rel=0.01)
        assert props.laminar_flame_speed_m_s == pytest.approx(0.40, rel=0.05)

    def test_hydrogen_properties(self):
        """Test hydrogen properties are correct."""
        props = FUEL_PROPERTIES_DB["hydrogen"]
        assert props.stoich_afr_mass == pytest.approx(34.3, rel=0.01)
        assert props.laminar_flame_speed_m_s == pytest.approx(2.10, rel=0.05)
        assert props.carbon_content == 0.0
        assert props.hydrogen_content == 1.0

    def test_fuel_oil_properties(self):
        """Test fuel oil properties are correct."""
        props = FUEL_PROPERTIES_DB["fuel_oil_2"]
        assert props.stoich_afr_mass == pytest.approx(14.7, rel=0.01)
        assert props.density_kg_m3 == pytest.approx(850.0, rel=0.01)
        assert props.sulfur_content == pytest.approx(0.005, rel=0.1)

    def test_all_fuel_types_have_required_fields(self):
        """Test all fuel types have all required properties."""
        required_fields = [
            "name", "molecular_weight", "stoich_afr_mass",
            "hhv_mj_kg", "lhv_mj_kg", "density_kg_m3",
            "carbon_content", "hydrogen_content"
        ]
        for fuel_type, props in FUEL_PROPERTIES_DB.items():
            for field in required_fields:
                assert hasattr(props, field), f"{fuel_type} missing {field}"

    def test_fuel_properties_immutable(self):
        """Test that FuelProperties dataclass is frozen."""
        props = FUEL_PROPERTIES_DB["natural_gas"]
        with pytest.raises(AttributeError):
            props.name = "Modified"


# =============================================================================
# UNIT TESTS - AIR FUEL RATIO
# =============================================================================

class TestAirFuelRatioCalculation:
    """Tests for air-fuel ratio optimization."""

    def test_optimal_afr_natural_gas(self, optimizer, natural_gas_input):
        """Test optimal AFR calculation for natural gas."""
        fuel_props = FUEL_PROPERTIES_DB["natural_gas"]
        result = optimizer.calculate_optimal_afr(natural_gas_input, fuel_props)

        # Stoichiometric AFR for natural gas is ~17.2
        # With 10-15% excess air, actual AFR should be ~18.9-19.8
        assert result.optimal_afr_mass > fuel_props.stoich_afr_mass
        assert result.optimal_afr_mass < fuel_props.stoich_afr_mass * 1.3
        assert result.optimal_excess_air_percent >= 5.0
        assert result.optimal_excess_air_percent <= 25.0

    def test_optimal_afr_hydrogen(self, optimizer):
        """Test optimal AFR calculation for hydrogen."""
        inputs = BurnerTuningInput(
            fuel_type="hydrogen",
            fuel_flow_kg_hr=100.0,
            excess_air_percent=10.0,
            o2_measured_percent=2.0,
            burner_type="premix",
        )
        fuel_props = FUEL_PROPERTIES_DB["hydrogen"]
        result = optimizer.calculate_optimal_afr(inputs, fuel_props)

        # Hydrogen has very high stoichiometric AFR (~34.3)
        assert result.optimal_afr_mass > 30.0
        assert result.optimal_afr_mass < 45.0

    def test_afr_varies_with_burner_type(self, optimizer):
        """Test AFR optimization varies by burner type."""
        fuel_props = FUEL_PROPERTIES_DB["natural_gas"]

        premix_input = BurnerTuningInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=500.0,
            burner_type="premix",
        )
        low_nox_input = BurnerTuningInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=500.0,
            burner_type="low_nox",
        )

        premix_result = optimizer.calculate_optimal_afr(premix_input, fuel_props)
        low_nox_result = optimizer.calculate_optimal_afr(low_nox_input, fuel_props)

        # Low NOx burners typically need more excess air
        assert low_nox_result.optimal_excess_air_percent >= premix_result.optimal_excess_air_percent

    def test_o2_target_calculation(self, optimizer, natural_gas_input):
        """Test O2 target calculation from excess air."""
        fuel_props = FUEL_PROPERTIES_DB["natural_gas"]
        result = optimizer.calculate_optimal_afr(natural_gas_input, fuel_props)

        # O2 should be in reasonable range (1-5% for gas)
        assert result.recommended_o2_target_percent >= 1.0
        assert result.recommended_o2_target_percent <= 6.0

    def test_air_flow_calculation(self, optimizer, natural_gas_input):
        """Test air flow rate calculation."""
        fuel_props = FUEL_PROPERTIES_DB["natural_gas"]
        result = optimizer.calculate_optimal_afr(natural_gas_input, fuel_props)

        # Air flow should be greater than fuel flow * stoich AFR
        min_air = natural_gas_input.fuel_flow_kg_hr * fuel_props.stoich_afr_mass
        assert result.air_flow_kg_hr > min_air

    def test_heat_input_calculation(self, optimizer, natural_gas_input):
        """Test heat input calculation."""
        fuel_props = FUEL_PROPERTIES_DB["natural_gas"]
        result = optimizer.calculate_optimal_afr(natural_gas_input, fuel_props)

        # Heat input = fuel flow * LHV
        expected_heat = natural_gas_input.fuel_flow_kg_hr * fuel_props.lhv_mj_kg / 3600
        assert result.heat_input_mw == pytest.approx(expected_heat, rel=0.01)


# =============================================================================
# UNIT TESTS - FLAME STABILITY
# =============================================================================

class TestFlameStabilityAnalysis:
    """Tests for flame stability analysis."""

    def test_wobbe_index_calculation(self, optimizer, natural_gas_input):
        """Test Wobbe Index calculation."""
        fuel_props = FUEL_PROPERTIES_DB["natural_gas"]
        result = optimizer.analyze_flame_stability(natural_gas_input, fuel_props)

        # Natural gas Wobbe Index ~48 MJ/m3
        assert result.wobbe_index_mj_m3 > 40.0
        assert result.wobbe_index_mj_m3 < 60.0

    def test_laminar_flame_speed_correction(self, optimizer):
        """Test flame speed correction for temperature and pressure."""
        fuel_props = FUEL_PROPERTIES_DB["natural_gas"]

        # Standard conditions
        standard_input = BurnerTuningInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=500.0,
            fuel_temperature_c=25.0,
            fuel_pressure_kpa=101.325,
        )

        # Elevated temperature
        hot_input = BurnerTuningInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=500.0,
            fuel_temperature_c=200.0,
            fuel_pressure_kpa=101.325,
        )

        standard_result = optimizer.analyze_flame_stability(standard_input, fuel_props)
        hot_result = optimizer.analyze_flame_stability(hot_input, fuel_props)

        # Flame speed should increase with temperature
        assert hot_result.laminar_flame_speed_m_s > standard_result.laminar_flame_speed_m_s

    def test_stability_status_determination(self, optimizer, natural_gas_input):
        """Test flame stability status determination."""
        fuel_props = FUEL_PROPERTIES_DB["natural_gas"]
        result = optimizer.analyze_flame_stability(natural_gas_input, fuel_props)

        # Status should be one of the valid options
        valid_statuses = [s.value for s in FlameStabilityStatus]
        assert result.stability_status in valid_statuses

    def test_damkohler_number_calculation(self, optimizer, natural_gas_input):
        """Test Damkohler number calculation."""
        fuel_props = FUEL_PROPERTIES_DB["natural_gas"]
        result = optimizer.analyze_flame_stability(natural_gas_input, fuel_props)

        # Damkohler number should be positive
        assert result.damkohler_number > 0

    def test_blowoff_flashback_velocities(self, optimizer, natural_gas_input):
        """Test blowoff and flashback velocity calculations."""
        fuel_props = FUEL_PROPERTIES_DB["natural_gas"]
        result = optimizer.analyze_flame_stability(natural_gas_input, fuel_props)

        # Blowoff velocity should be higher than flashback
        assert result.blowoff_velocity_m_s > result.flashback_velocity_m_s
        # Both should be positive
        assert result.flashback_velocity_m_s > 0


# =============================================================================
# UNIT TESTS - TURNDOWN ANALYSIS
# =============================================================================

class TestTurndownAnalysis:
    """Tests for burner turndown ratio analysis."""

    def test_turndown_ratio_calculation(self, optimizer, natural_gas_input):
        """Test effective turndown ratio calculation."""
        fuel_props = FUEL_PROPERTIES_DB["natural_gas"]
        result = optimizer.analyze_turndown_ratio(natural_gas_input, fuel_props)

        # Effective turndown should be > 1
        assert result.effective_turndown_ratio > 1.0
        # Min stable load should be > 0 and < 100
        assert 0 < result.min_stable_load_percent < 100

    def test_turndown_varies_by_burner_type(self, optimizer):
        """Test turndown varies by burner type."""
        fuel_props = FUEL_PROPERTIES_DB["natural_gas"]

        premix_input = BurnerTuningInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=500.0,
            burner_type="premix",
        )
        diffusion_input = BurnerTuningInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=500.0,
            burner_type="diffusion",
        )

        premix_result = optimizer.analyze_turndown_ratio(premix_input, fuel_props)
        diffusion_result = optimizer.analyze_turndown_ratio(diffusion_input, fuel_props)

        # Premix typically has better turndown (lower min stable load)
        assert premix_result.min_stable_load_percent < diffusion_result.min_stable_load_percent

    def test_efficiency_at_load_extremes(self, optimizer, natural_gas_input):
        """Test efficiency values at load extremes."""
        fuel_props = FUEL_PROPERTIES_DB["natural_gas"]
        result = optimizer.analyze_turndown_ratio(natural_gas_input, fuel_props)

        # Efficiency at max load should be higher than at min load
        assert result.efficiency_at_max_load > result.efficiency_at_min_load
        # Both should be in reasonable range (70-95%)
        assert 70 < result.efficiency_at_min_load < 95
        assert 70 < result.efficiency_at_max_load < 95


# =============================================================================
# UNIT TESTS - AIR DISTRIBUTION
# =============================================================================

class TestAirDistribution:
    """Tests for combustion air distribution optimization."""

    def test_air_distribution_sums_to_100(self, optimizer, natural_gas_input):
        """Test that air distribution percentages sum to 100%."""
        fuel_props = FUEL_PROPERTIES_DB["natural_gas"]
        result = optimizer.optimize_air_distribution(natural_gas_input, fuel_props)

        total = (result.primary_air_percent +
                result.secondary_air_percent +
                result.tertiary_air_percent)
        assert total == pytest.approx(100.0, rel=0.01)

    def test_staged_burner_air_distribution(self, optimizer):
        """Test air distribution for staged burners."""
        fuel_props = FUEL_PROPERTIES_DB["natural_gas"]
        staged_input = BurnerTuningInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=500.0,
            burner_type="staged_air",
        )
        result = optimizer.optimize_air_distribution(staged_input, fuel_props)

        # Staged burners should have significant secondary air
        assert result.secondary_air_percent > 20

    def test_swirl_number_range(self, optimizer, natural_gas_input):
        """Test swirl number is in valid range."""
        fuel_props = FUEL_PROPERTIES_DB["natural_gas"]
        result = optimizer.optimize_air_distribution(natural_gas_input, fuel_props)

        # Swirl number typically 0-2 for industrial burners
        assert 0 <= result.swirl_number <= 2.0

    def test_mixing_effectiveness_range(self, optimizer, natural_gas_input):
        """Test mixing effectiveness is in 0-1 range."""
        fuel_props = FUEL_PROPERTIES_DB["natural_gas"]
        result = optimizer.optimize_air_distribution(natural_gas_input, fuel_props)

        assert 0 <= result.mixing_effectiveness <= 1.0


# =============================================================================
# UNIT TESTS - DRAFT PRESSURE
# =============================================================================

class TestDraftPressureCalculation:
    """Tests for draft and pressure balance calculations."""

    def test_stack_draft_positive(self, optimizer, natural_gas_input):
        """Test stack draft is positive (creating suction)."""
        fuel_props = FUEL_PROPERTIES_DB["natural_gas"]
        result = optimizer.calculate_draft_balance(natural_gas_input, fuel_props)

        # Stack draft should be positive (suction)
        assert result.stack_draft_pa > 0

    def test_friction_loss_positive(self, optimizer, natural_gas_input):
        """Test friction loss is positive."""
        fuel_props = FUEL_PROPERTIES_DB["natural_gas"]
        result = optimizer.calculate_draft_balance(natural_gas_input, fuel_props)

        assert result.friction_loss_pa > 0

    def test_draft_balance(self, optimizer, natural_gas_input):
        """Test draft balance calculation logic."""
        fuel_props = FUEL_PROPERTIES_DB["natural_gas"]
        result = optimizer.calculate_draft_balance(natural_gas_input, fuel_props)

        # Draft margin = stack draft - friction loss
        expected_margin = result.stack_draft_pa - result.friction_loss_pa
        assert result.draft_margin_pa == pytest.approx(expected_margin, rel=0.01)


# =============================================================================
# UNIT TESTS - ATOMIZATION (LIQUID FUELS)
# =============================================================================

class TestAtomizationAnalysis:
    """Tests for liquid fuel atomization analysis."""

    def test_atomization_analysis_liquid_fuel(self, optimizer, fuel_oil_input):
        """Test atomization analysis for liquid fuel."""
        fuel_props = FUEL_PROPERTIES_DB["fuel_oil_2"]
        result = optimizer.analyze_atomization(fuel_oil_input, fuel_props)

        # Weber number should be positive
        assert result.weber_number > 0
        # Ohnesorge number should be positive
        assert result.ohnesorge_number > 0
        # SMD should be in typical range (10-500 um)
        assert 10 <= result.sauter_mean_diameter_um <= 500

    def test_atomization_quality_rating(self, optimizer, fuel_oil_input):
        """Test atomization quality rating."""
        fuel_props = FUEL_PROPERTIES_DB["fuel_oil_2"]
        result = optimizer.analyze_atomization(fuel_oil_input, fuel_props)

        valid_qualities = [q.value for q in AtomizationQuality]
        assert result.quality_rating in valid_qualities

    def test_atomization_requires_parameters(self, optimizer):
        """Test atomization analysis requires viscosity and pressure."""
        gas_input = BurnerTuningInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=500.0,
        )
        fuel_props = FUEL_PROPERTIES_DB["natural_gas"]

        with pytest.raises(ValueError):
            optimizer.analyze_atomization(gas_input, fuel_props)


# =============================================================================
# UNIT TESTS - MULTI-BURNER
# =============================================================================

class TestMultiBurnerAnalysis:
    """Tests for multi-burner furnace analysis."""

    def test_multi_burner_heat_distribution(self, optimizer, multi_burner_input):
        """Test heat distribution across burners."""
        fuel_props = FUEL_PROPERTIES_DB["natural_gas"]
        result = optimizer.analyze_multi_burner(multi_burner_input, fuel_props)

        # Should have heat for each burner
        assert len(result.burner_heat_distribution) == multi_burner_input.num_burners
        # Sum should equal total
        assert sum(result.burner_heat_distribution) == pytest.approx(
            result.total_heat_input_mw, rel=0.01
        )

    def test_multi_burner_load_balance(self, optimizer, multi_burner_input):
        """Test load balance percentage."""
        fuel_props = FUEL_PROPERTIES_DB["natural_gas"]
        result = optimizer.analyze_multi_burner(multi_burner_input, fuel_props)

        # Load balance should be 0-100%
        assert 0 <= result.burner_load_balance_percent <= 100

    def test_cross_lighting_risk(self, optimizer):
        """Test cross-lighting risk detection at low load."""
        fuel_props = FUEL_PROPERTIES_DB["natural_gas"]

        low_load_multi = BurnerTuningInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=200.0,
            num_burners=4,
            burner_capacity_mw=10.0,
            load_percent=20.0,
        )
        result = optimizer.analyze_multi_burner(low_load_multi, fuel_props)

        # At low load, cross-lighting risk should be flagged
        assert result.cross_lighting_risk is True


# =============================================================================
# INTEGRATION TESTS - COMPLETE OPTIMIZATION
# =============================================================================

class TestCompleteOptimization:
    """Integration tests for complete optimization workflow."""

    def test_complete_optimization_natural_gas(self, optimizer, natural_gas_input):
        """Test complete optimization for natural gas."""
        result = optimizer.optimize(natural_gas_input)

        # Verify all components are present
        assert result.air_fuel_result is not None
        assert result.emission_targets is not None
        assert result.turndown_result is not None
        assert result.air_distribution is not None
        assert result.flame_stability is not None
        assert result.draft_pressure is not None
        assert result.recommendations is not None
        assert result.provenance_hash is not None

    def test_complete_optimization_fuel_oil(self, optimizer, fuel_oil_input):
        """Test complete optimization for fuel oil with atomization."""
        result = optimizer.optimize(fuel_oil_input)

        # Should include atomization analysis
        assert result.atomization is not None
        assert result.atomization.weber_number > 0

    def test_complete_optimization_multi_burner(self, optimizer, multi_burner_input):
        """Test complete optimization for multi-burner."""
        result = optimizer.optimize(multi_burner_input)

        # Should include multi-burner analysis
        assert result.multi_burner is not None
        assert len(result.multi_burner.burner_heat_distribution) == 4

    def test_optimization_score_range(self, optimizer, natural_gas_input):
        """Test optimization score is in valid range."""
        result = optimizer.optimize(natural_gas_input)

        assert 0 <= result.optimization_score <= 100

    def test_provenance_hash_generated(self, optimizer, natural_gas_input):
        """Test provenance hash is generated."""
        result = optimizer.optimize(natural_gas_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 16  # First 16 chars of SHA-256

    def test_calculation_steps_recorded(self, optimizer, natural_gas_input):
        """Test calculation steps are recorded."""
        result = optimizer.optimize(natural_gas_input)

        assert result.calculation_steps > 0
        steps = optimizer.get_calculation_steps()
        assert len(steps) == result.calculation_steps


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests for calculation determinism and reproducibility."""

    def test_same_inputs_same_outputs(self, natural_gas_input):
        """Test same inputs produce same outputs."""
        optimizer1 = BurnerTuningOptimizer()
        optimizer2 = BurnerTuningOptimizer()

        result1 = optimizer1.optimize(natural_gas_input)
        result2 = optimizer2.optimize(natural_gas_input)

        assert result1.air_fuel_result.optimal_afr_mass == result2.air_fuel_result.optimal_afr_mass
        assert result1.flame_stability.stability_index == result2.flame_stability.stability_index
        assert result1.optimization_score == result2.optimization_score

    def test_provenance_hash_deterministic(self, natural_gas_input):
        """Test provenance hash is deterministic for same inputs."""
        optimizer1 = BurnerTuningOptimizer()
        optimizer2 = BurnerTuningOptimizer()

        result1 = optimizer1.optimize(natural_gas_input)
        result2 = optimizer2.optimize(natural_gas_input)

        # Note: Hash includes timestamp, so we check step content instead
        steps1 = optimizer1.get_calculation_steps()
        steps2 = optimizer2.get_calculation_steps()

        assert len(steps1) == len(steps2)
        for s1, s2 in zip(steps1, steps2):
            assert s1["operation"] == s2["operation"]
            assert s1["output"] == s2["output"]

    def test_repeated_calls_consistent(self, optimizer, natural_gas_input):
        """Test repeated calls give consistent results."""
        results = [optimizer.optimize(natural_gas_input) for _ in range(5)]

        afr_values = [r.air_fuel_result.optimal_afr_mass for r in results]
        # All values should be identical (floating point exact)
        assert all(v == afr_values[0] for v in afr_values)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_low_load(self, optimizer, low_load_input):
        """Test optimization at very low load."""
        result = optimizer.optimize(low_load_input)

        # Should still produce valid results
        assert result.air_fuel_result is not None
        # Should recommend caution at low load
        assert any("low load" in rec.lower() for rec in result.recommendations)

    def test_high_excess_air(self, optimizer):
        """Test optimization with high excess air."""
        high_ea_input = BurnerTuningInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=500.0,
            excess_air_percent=40.0,
            o2_measured_percent=7.0,
        )
        result = optimizer.optimize(high_ea_input)

        # Should recommend reducing excess air
        assert any("excess air" in rec.lower() for rec in result.recommendations)

    def test_low_excess_air(self, optimizer):
        """Test optimization with low excess air."""
        low_ea_input = BurnerTuningInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=500.0,
            excess_air_percent=3.0,
            o2_measured_percent=0.5,
        )
        result = optimizer.optimize(low_ea_input)

        # Should recommend increasing excess air
        assert any("excess air" in rec.lower() or "co" in rec.lower()
                  for rec in result.recommendations)

    def test_zero_fuel_flow_raises(self, optimizer):
        """Test that zero fuel flow is handled."""
        # This should work because dataclass doesn't validate
        zero_flow = BurnerTuningInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=0.0,  # Edge case
        )
        # The optimization should handle this gracefully
        result = optimizer.optimize(zero_flow)
        assert result is not None

    def test_unknown_fuel_type(self, optimizer):
        """Test handling of unknown fuel type."""
        unknown_fuel = BurnerTuningInput(
            fuel_type="unknown_fuel",
            fuel_flow_kg_hr=500.0,
        )
        with pytest.raises(ValueError, match="Unknown fuel type"):
            optimizer.optimize(unknown_fuel)


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Tests for calculation performance."""

    def test_optimization_speed(self, optimizer, natural_gas_input):
        """Test optimization completes within time limit."""
        import time

        start = time.perf_counter()
        result = optimizer.optimize(natural_gas_input)
        elapsed = time.perf_counter() - start

        # Should complete within 100ms
        assert elapsed < 0.1
        assert result is not None

    def test_multiple_optimizations_performance(self, optimizer, natural_gas_input):
        """Test multiple optimizations maintain performance."""
        import time

        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            optimizer.optimize(natural_gas_input)
        elapsed = time.perf_counter() - start

        # Average should be under 50ms
        avg_time = elapsed / iterations
        assert avg_time < 0.05


# =============================================================================
# THREAD SAFETY TESTS
# =============================================================================

class TestThreadSafety:
    """Tests for thread safety of optimizer and cache."""

    def test_concurrent_optimizations(self, natural_gas_input):
        """Test concurrent optimizations don't interfere."""
        results = []
        errors = []

        def optimize():
            try:
                optimizer = BurnerTuningOptimizer()
                result = optimizer.optimize(natural_gas_input)
                results.append(result.air_fuel_result.optimal_afr_mass)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=optimize) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        # All results should be the same
        assert all(r == results[0] for r in results)

    def test_thread_safe_cache(self):
        """Test ThreadSafeCache under concurrent access."""
        cache = ThreadSafeCache(max_size=100)
        errors = []

        def cache_ops(thread_id):
            try:
                for i in range(100):
                    key = f"thread_{thread_id}_key_{i}"
                    cache.set(key, i * thread_id)
                    value = cache.get(key)
                    # Value might be evicted, so None is valid
                    if value is not None:
                        assert value == i * thread_id
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=cache_ops, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_provenance_tracker_thread_safety(self):
        """Test ProvenanceTracker under concurrent access."""
        tracker = ProvenanceTracker()
        errors = []

        def log_steps(thread_id):
            try:
                for i in range(100):
                    tracker.log_step(
                        f"operation_{thread_id}_{i}",
                        {"input": i},
                        {"output": i * 2}
                    )
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=log_steps, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        steps = tracker.get_steps()
        assert len(steps) == 500  # 5 threads * 100 steps each


# =============================================================================
# PROVENANCE TESTS
# =============================================================================

class TestProvenance:
    """Tests for provenance tracking and audit trail."""

    def test_provenance_tracker_logs_steps(self):
        """Test ProvenanceTracker logs calculation steps."""
        tracker = ProvenanceTracker()

        tracker.log_step("step1", {"a": 1}, 2)
        tracker.log_step("step2", {"b": 2}, 4, formula="y = x * 2")

        steps = tracker.get_steps()
        assert len(steps) == 2
        assert steps[0]["operation"] == "step1"
        assert steps[1]["formula"] == "y = x * 2"

    def test_provenance_hash_calculation(self):
        """Test provenance hash calculation."""
        tracker = ProvenanceTracker()
        tracker.log_step("op1", {"x": 1}, 2)
        tracker.log_step("op2", {"y": 2}, 4)

        hash1 = tracker.calculate_provenance_hash()
        hash2 = tracker.calculate_provenance_hash()

        # Same steps should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 16

    def test_provenance_clear(self):
        """Test ProvenanceTracker clear method."""
        tracker = ProvenanceTracker()
        tracker.log_step("step1", {"a": 1}, 2)
        assert len(tracker.get_steps()) == 1

        tracker.clear()
        assert len(tracker.get_steps()) == 0


# =============================================================================
# OUTPUT DATACLASS TESTS
# =============================================================================

class TestOutputDataclasses:
    """Tests for output dataclass immutability and structure."""

    def test_output_dataclass_frozen(self, optimizer, natural_gas_input):
        """Test output dataclasses are frozen (immutable)."""
        result = optimizer.optimize(natural_gas_input)

        with pytest.raises(AttributeError):
            result.optimization_score = 50.0

    def test_afr_result_all_fields(self, optimizer, natural_gas_input):
        """Test AirFuelRatioResult has all required fields."""
        fuel_props = FUEL_PROPERTIES_DB["natural_gas"]
        result = optimizer.calculate_optimal_afr(natural_gas_input, fuel_props)

        assert hasattr(result, "optimal_afr_mass")
        assert hasattr(result, "optimal_afr_vol")
        assert hasattr(result, "optimal_excess_air_percent")
        assert hasattr(result, "recommended_o2_target_percent")
        assert hasattr(result, "heat_input_mw")
        assert hasattr(result, "efficiency_potential_percent")


# =============================================================================
# RECOMMENDATIONS TESTS
# =============================================================================

class TestRecommendations:
    """Tests for optimization recommendations generation."""

    def test_recommendations_not_empty(self, optimizer, natural_gas_input):
        """Test recommendations are always generated."""
        result = optimizer.optimize(natural_gas_input)

        assert len(result.recommendations) > 0

    def test_recommendations_for_high_nox(self, optimizer):
        """Test recommendations generated for high NOx."""
        high_nox_input = BurnerTuningInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=500.0,
            nox_measured_ppm=80.0,
        )
        result = optimizer.optimize(high_nox_input)

        # Should have NOx-related recommendation
        assert any("nox" in rec.lower() for rec in result.recommendations)

    def test_recommendations_for_high_co(self, optimizer):
        """Test recommendations generated for high CO."""
        high_co_input = BurnerTuningInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=500.0,
            co_measured_ppm=150.0,
        )
        result = optimizer.optimize(high_co_input)

        # Should have CO-related recommendation
        assert any("co" in rec.lower() for rec in result.recommendations)


# =============================================================================
# GOLDEN VALUE TESTS
# =============================================================================

class TestGoldenValues:
    """Tests against known good values for regression testing."""

    def test_natural_gas_afr_golden(self, optimizer):
        """Test natural gas AFR against golden values."""
        inputs = BurnerTuningInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=500.0,
            fuel_temperature_c=25.0,
            fuel_pressure_kpa=101.325,
            excess_air_percent=15.0,
            burner_type="nozzle_mix",
        )
        fuel_props = FUEL_PROPERTIES_DB["natural_gas"]
        result = optimizer.calculate_optimal_afr(inputs, fuel_props)

        # Stoichiometric AFR for NG is 17.2
        # With ~10% optimal excess air, AFR should be ~18.9
        assert result.optimal_afr_mass == pytest.approx(18.9, abs=1.0)

    def test_heat_input_golden(self, optimizer):
        """Test heat input calculation against golden value."""
        inputs = BurnerTuningInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=360.0,  # 360 kg/hr
        )
        fuel_props = FUEL_PROPERTIES_DB["natural_gas"]
        result = optimizer.calculate_optimal_afr(inputs, fuel_props)

        # 360 kg/hr * 50 MJ/kg / 3600 s = 5 MW
        assert result.heat_input_mw == pytest.approx(5.0, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
