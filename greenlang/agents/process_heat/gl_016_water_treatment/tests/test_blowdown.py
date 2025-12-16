"""
GL-016 WATERGUARD Agent - Blowdown Optimizer Tests

Unit tests for BlowdownOptimizer covering:
- Cycles of concentration calculation
- Blowdown rate optimization
- Energy savings calculations
- Water savings calculations
- Heat recovery potential
- Flash steam calculations
- Provenance tracking

Author: GL-TestEngineer
Target Coverage: 85%+
"""

import pytest
from datetime import datetime, timezone
from typing import Tuple

from greenlang.agents.process_heat.gl_016_water_treatment.schemas import (
    BlowdownInput,
    BlowdownOutput,
    BlowdownType,
)

from greenlang.agents.process_heat.gl_016_water_treatment.blowdown import (
    BlowdownOptimizer,
    BlowdownConstants,
    calculate_makeup_requirement,
)


class TestBlowdownOptimizerInitialization:
    """Test BlowdownOptimizer initialization."""

    def test_default_initialization(self):
        """Test optimizer initializes with defaults."""
        optimizer = BlowdownOptimizer()
        assert optimizer.boiler_efficiency == 0.82

    def test_custom_efficiency(self):
        """Test optimizer with custom boiler efficiency."""
        optimizer = BlowdownOptimizer(boiler_efficiency=0.85)
        assert optimizer.boiler_efficiency == 0.85


class TestCyclesOfConcentration:
    """Test cycles of concentration calculations."""

    @pytest.fixture
    def optimizer(self):
        return BlowdownOptimizer()

    @pytest.mark.parametrize("boiler_tds,feedwater_tds,expected_cycles", [
        (2000.0, 50.0, 40.0),
        (1500.0, 100.0, 15.0),
        (2500.0, 250.0, 10.0),
        (3000.0, 300.0, 10.0),
        (1000.0, 200.0, 5.0),
        (500.0, 500.0, 1.0),  # Same TDS = 1 cycle
    ])
    def test_cycles_calculation(self, optimizer, boiler_tds, feedwater_tds, expected_cycles):
        """Test cycles calculation: Cycles = Boiler TDS / Feedwater TDS."""
        result = optimizer.calculate_cycles_of_concentration(boiler_tds, feedwater_tds)
        assert result == pytest.approx(expected_cycles, rel=0.01)

    def test_zero_feedwater_tds_returns_one(self, optimizer):
        """Test zero feedwater TDS returns 1 cycle."""
        result = optimizer.calculate_cycles_of_concentration(2000.0, 0.0)
        assert result == 1.0

    def test_negative_feedwater_tds_returns_one(self, optimizer):
        """Test negative feedwater TDS returns 1 cycle."""
        result = optimizer.calculate_cycles_of_concentration(2000.0, -50.0)
        assert result == 1.0


class TestBlowdownRateCalculation:
    """Test blowdown rate calculations."""

    @pytest.fixture
    def optimizer(self):
        return BlowdownOptimizer()

    @pytest.mark.parametrize("cycles,expected_blowdown_pct", [
        (5.0, 20.0),    # 100/5 = 20%
        (10.0, 10.0),   # 100/10 = 10%
        (4.0, 25.0),    # 100/4 = 25%
        (8.0, 12.5),    # 100/8 = 12.5%
        (2.0, 50.0),    # 100/2 = 50%
        (20.0, 5.0),    # 100/20 = 5%
    ])
    def test_blowdown_rate_from_cycles(self, optimizer, cycles, expected_blowdown_pct):
        """Test blowdown rate calculation: BD% = 100/Cycles."""
        result = optimizer.calculate_blowdown_rate(cycles)
        assert result == pytest.approx(expected_blowdown_pct, rel=0.01)

    def test_one_cycle_returns_100_percent(self, optimizer):
        """Test 1 cycle returns 100% blowdown."""
        result = optimizer.calculate_blowdown_rate(1.0)
        assert result == 100.0

    def test_less_than_one_cycle_returns_100_percent(self, optimizer):
        """Test < 1 cycle returns 100% blowdown."""
        result = optimizer.calculate_blowdown_rate(0.5)
        assert result == 100.0


class TestBlowdownFlowCalculation:
    """Test blowdown flow calculations."""

    @pytest.fixture
    def optimizer(self):
        return BlowdownOptimizer()

    @pytest.mark.parametrize("steam_flow,blowdown_pct,expected_flow", [
        (50000.0, 10.0, 5556.0),   # 50000 * (10/(100-10)) = 5556
        (50000.0, 5.0, 2632.0),    # 50000 * (5/(100-5)) = 2632
        (100000.0, 10.0, 11111.0), # 100000 * (10/90) = 11111
        (50000.0, 20.0, 12500.0),  # 50000 * (20/80) = 12500
    ])
    def test_blowdown_flow_calculation(self, optimizer, steam_flow, blowdown_pct, expected_flow):
        """Test blowdown flow calculation."""
        result = optimizer.calculate_blowdown_flow(steam_flow, blowdown_pct)
        assert result == pytest.approx(expected_flow, rel=0.01)

    def test_100_percent_blowdown_equals_steam_flow(self, optimizer):
        """Test 100% blowdown equals steam flow."""
        result = optimizer.calculate_blowdown_flow(50000.0, 100.0)
        assert result == 50000.0


class TestOptimalCyclesCalculation:
    """Test optimal cycles calculation."""

    @pytest.fixture
    def optimizer(self):
        return BlowdownOptimizer()

    @pytest.mark.parametrize("feedwater_tds,tds_limit,expected_cycles", [
        (50.0, 2500.0, 10.0),   # Capped at MAX_CYCLES_RECOMMENDED (10)
        (100.0, 2500.0, 10.0),  # Capped at MAX_CYCLES_RECOMMENDED
        (250.0, 2500.0, 10.0),  # 2500/250 = 10
        (500.0, 2500.0, 5.0),   # 2500/500 = 5
        (833.0, 2500.0, 3.0),   # 2500/833 = 3.0 (at MIN_CYCLES)
    ])
    def test_optimal_cycles_calculation(self, optimizer, feedwater_tds, tds_limit, expected_cycles):
        """Test optimal cycles calculation."""
        result = optimizer.calculate_optimal_cycles(feedwater_tds, tds_limit)
        assert result == pytest.approx(expected_cycles, rel=0.05)

    def test_optimal_cycles_respects_minimum(self, optimizer):
        """Test optimal cycles respects minimum of 3."""
        result = optimizer.calculate_optimal_cycles(1000.0, 2500.0)
        assert result >= BlowdownConstants.MIN_CYCLES

    def test_optimal_cycles_respects_maximum(self, optimizer):
        """Test optimal cycles respects maximum of 10."""
        result = optimizer.calculate_optimal_cycles(10.0, 2500.0)
        assert result <= BlowdownConstants.MAX_CYCLES_RECOMMENDED


class TestEnergySavingsCalculation:
    """Test energy savings calculations."""

    @pytest.fixture
    def optimizer(self):
        return BlowdownOptimizer()

    def test_energy_savings_positive_reduction(self, optimizer):
        """Test positive energy savings with blowdown reduction."""
        savings = optimizer.calculate_energy_savings(
            current_blowdown_lb_hr=10000.0,
            optimal_blowdown_lb_hr=5000.0,
            pressure_psig=450.0,
            heat_recovery_enabled=False,
        )
        assert savings > 0

    def test_energy_savings_no_reduction(self, optimizer):
        """Test no energy savings when no reduction possible."""
        savings = optimizer.calculate_energy_savings(
            current_blowdown_lb_hr=5000.0,
            optimal_blowdown_lb_hr=5000.0,  # Same
            pressure_psig=450.0,
            heat_recovery_enabled=False,
        )
        assert savings == 0

    def test_energy_savings_negative_reduction(self, optimizer):
        """Test zero savings when optimal > current."""
        savings = optimizer.calculate_energy_savings(
            current_blowdown_lb_hr=5000.0,
            optimal_blowdown_lb_hr=8000.0,  # Higher
            pressure_psig=450.0,
            heat_recovery_enabled=False,
        )
        assert savings == 0

    def test_heat_recovery_reduces_savings(self, optimizer):
        """Test heat recovery reduces potential savings."""
        savings_no_hr = optimizer.calculate_energy_savings(
            current_blowdown_lb_hr=10000.0,
            optimal_blowdown_lb_hr=5000.0,
            pressure_psig=450.0,
            heat_recovery_enabled=False,
        )
        savings_with_hr = optimizer.calculate_energy_savings(
            current_blowdown_lb_hr=10000.0,
            optimal_blowdown_lb_hr=5000.0,
            pressure_psig=450.0,
            heat_recovery_enabled=True,
        )
        assert savings_with_hr < savings_no_hr

    def test_higher_pressure_more_energy(self, optimizer):
        """Test higher pressure blowdown contains more energy."""
        savings_low_p = optimizer.calculate_energy_savings(
            current_blowdown_lb_hr=10000.0,
            optimal_blowdown_lb_hr=5000.0,
            pressure_psig=150.0,
            heat_recovery_enabled=False,
        )
        savings_high_p = optimizer.calculate_energy_savings(
            current_blowdown_lb_hr=10000.0,
            optimal_blowdown_lb_hr=5000.0,
            pressure_psig=450.0,
            heat_recovery_enabled=False,
        )
        assert savings_high_p > savings_low_p


class TestWaterSavingsCalculation:
    """Test water savings calculations."""

    @pytest.fixture
    def optimizer(self):
        return BlowdownOptimizer()

    def test_water_savings_positive_reduction(self, optimizer):
        """Test positive water savings with blowdown reduction."""
        savings = optimizer.calculate_water_savings(
            current_blowdown_lb_hr=10000.0,
            optimal_blowdown_lb_hr=5000.0,
        )
        assert savings > 0
        # 5000 lb/hr reduction / 8.34 lb/gal = 600 gal/hr
        # 600 gal/hr * 8760 hr/yr / 1000 = 5254 kgal/yr
        assert savings == pytest.approx(5250.0, rel=0.05)

    def test_water_savings_no_reduction(self, optimizer):
        """Test no water savings when no reduction."""
        savings = optimizer.calculate_water_savings(
            current_blowdown_lb_hr=5000.0,
            optimal_blowdown_lb_hr=5000.0,
        )
        assert savings == 0


class TestTotalSavingsCalculation:
    """Test total savings calculations."""

    @pytest.fixture
    def optimizer(self):
        return BlowdownOptimizer()

    def test_total_savings_calculation(self, optimizer):
        """Test total savings includes fuel, water, and chemical costs."""
        total = optimizer.calculate_total_savings(
            energy_savings_mmbtu=1000.0,
            water_savings_kgal=5000.0,
            fuel_cost=5.0,       # $/MMBTU
            water_cost=3.0,      # $/kgal
            chemical_cost=2.0,   # $/kgal
        )
        # Expected: 1000*5 + 5000*3 + 5000*2 = 5000 + 15000 + 10000 = $30,000
        assert total == pytest.approx(30000.0, rel=0.01)


class TestHeatRecoveryPotential:
    """Test heat recovery potential calculations."""

    @pytest.fixture
    def optimizer(self):
        return BlowdownOptimizer()

    def test_heat_recovery_returns_tuple(self, optimizer):
        """Test heat recovery returns tuple of (heat, flash_steam)."""
        result = optimizer.calculate_heat_recovery_potential(
            blowdown_flow_lb_hr=5000.0,
            boiler_pressure_psig=450.0,
            flash_tank_pressure_psig=5.0,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_heat_recovery_positive(self, optimizer):
        """Test heat recovery potential is positive."""
        heat_recovery, flash_steam = optimizer.calculate_heat_recovery_potential(
            blowdown_flow_lb_hr=5000.0,
            boiler_pressure_psig=450.0,
            flash_tank_pressure_psig=5.0,
        )
        assert heat_recovery > 0
        assert flash_steam > 0

    def test_flash_steam_increases_with_pressure(self, optimizer):
        """Test flash steam increases with higher boiler pressure."""
        _, flash_low_p = optimizer.calculate_heat_recovery_potential(
            blowdown_flow_lb_hr=5000.0,
            boiler_pressure_psig=150.0,
            flash_tank_pressure_psig=5.0,
        )
        _, flash_high_p = optimizer.calculate_heat_recovery_potential(
            blowdown_flow_lb_hr=5000.0,
            boiler_pressure_psig=450.0,
            flash_tank_pressure_psig=5.0,
        )
        assert flash_high_p > flash_low_p


class TestBlowdownOptimization:
    """Test full blowdown optimization."""

    @pytest.fixture
    def optimizer(self):
        return BlowdownOptimizer()

    def test_optimize_returns_output(self, optimizer, blowdown_input_optimize):
        """Test optimize returns BlowdownOutput."""
        result = optimizer.optimize(blowdown_input_optimize)
        assert isinstance(result, BlowdownOutput)

    def test_optimize_all_fields_populated(self, optimizer, blowdown_input_optimize):
        """Test optimize populates all output fields."""
        result = optimizer.optimize(blowdown_input_optimize)

        assert result.current_cycles_of_concentration > 0
        assert result.current_blowdown_rate_pct > 0
        assert result.current_blowdown_flow_lb_hr > 0
        assert result.optimal_cycles_of_concentration > 0
        assert result.optimal_blowdown_rate_pct > 0
        assert result.optimal_blowdown_flow_lb_hr > 0
        assert result.energy_savings_mmbtu_yr >= 0
        assert result.water_savings_kgal_yr >= 0
        assert result.total_savings_usd_yr >= 0
        assert result.optimization_status == "complete"
        assert result.provenance_hash is not None
        assert result.processing_time_ms > 0

    def test_optimize_high_tds_scenario(self, optimizer, blowdown_input_high_tds):
        """Test optimization with high TDS input."""
        result = optimizer.optimize(blowdown_input_high_tds)

        # High TDS = high cycles = low blowdown rate
        assert result.current_cycles_of_concentration > 10
        # Should recommend more blowdown
        assert result.optimal_blowdown_rate_pct <= result.current_blowdown_rate_pct

    def test_optimize_generates_recommendations(self, optimizer, blowdown_input_optimize):
        """Test optimization generates recommendations."""
        result = optimizer.optimize(blowdown_input_optimize)
        assert isinstance(result.recommendations, list)

    def test_optimize_within_limits_flag(self, optimizer, blowdown_input_optimize):
        """Test within_limits flag is set correctly."""
        result = optimizer.optimize(blowdown_input_optimize)
        assert isinstance(result.within_limits, bool)


class TestSteamPropertyInterpolation:
    """Test steam property interpolation."""

    @pytest.fixture
    def optimizer(self):
        return BlowdownOptimizer()

    def test_liquid_enthalpy_at_known_pressure(self, optimizer):
        """Test liquid enthalpy at known pressure point."""
        # At 150 psig, hf = 339 BTU/lb (per constants)
        hf = optimizer._get_liquid_enthalpy(150.0)
        assert hf == pytest.approx(339.0, rel=0.02)

    def test_liquid_enthalpy_interpolation(self, optimizer):
        """Test liquid enthalpy interpolation between points."""
        hf = optimizer._get_liquid_enthalpy(125.0)  # Between 100 and 150
        # Should be between hf_100 (309) and hf_150 (339)
        assert 309 < hf < 339

    def test_latent_heat_at_known_pressure(self, optimizer):
        """Test latent heat at known pressure point."""
        hfg = optimizer._get_latent_heat(150.0)
        assert hfg == pytest.approx(857.0, rel=0.02)

    def test_boundary_pressure_handling(self, optimizer):
        """Test boundary pressure handling."""
        # Below minimum
        hf_low = optimizer._get_liquid_enthalpy(-10.0)
        assert hf_low == BlowdownConstants.STEAM_PROPERTIES[0][1]

        # Above maximum
        hf_high = optimizer._get_liquid_enthalpy(700.0)
        assert hf_high == BlowdownConstants.STEAM_PROPERTIES[600][1]


class TestCalculateMakeupRequirement:
    """Test makeup water requirement calculation."""

    def test_makeup_calculation(self):
        """Test makeup water requirement calculation."""
        makeup = calculate_makeup_requirement(
            steam_flow_lb_hr=50000.0,
            condensate_return_pct=80.0,
            blowdown_rate_pct=5.0,
        )
        # Losses: Steam lost (20%) + Blowdown (5%) + Other (2%) = 27%
        # Makeup = 50000 * 0.27 = 13500 lb/hr
        expected = 50000 * 0.20 + 50000 * 0.05 + 50000 * 0.02
        assert makeup == pytest.approx(expected, rel=0.01)

    @pytest.mark.parametrize("condensate_return,blowdown_rate", [
        (90.0, 3.0),
        (70.0, 8.0),
        (50.0, 10.0),
        (100.0, 2.0),  # All condensate returned
        (0.0, 10.0),   # No condensate return
    ])
    def test_makeup_various_scenarios(self, condensate_return, blowdown_rate):
        """Test makeup calculation for various scenarios."""
        makeup = calculate_makeup_requirement(
            steam_flow_lb_hr=50000.0,
            condensate_return_pct=condensate_return,
            blowdown_rate_pct=blowdown_rate,
        )
        assert makeup > 0


class TestProvenanceTracking:
    """Test provenance hash calculation."""

    @pytest.fixture
    def optimizer(self):
        return BlowdownOptimizer()

    def test_provenance_hash_generated(self, optimizer, blowdown_input_optimize):
        """Test provenance hash is generated."""
        result = optimizer.optimize(blowdown_input_optimize)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_provenance_hash_reproducible(self, optimizer, blowdown_input_optimize):
        """Test same input produces same provenance hash."""
        hash1 = optimizer._calculate_provenance_hash(blowdown_input_optimize)
        hash2 = optimizer._calculate_provenance_hash(blowdown_input_optimize)
        assert hash1 == hash2


class TestBlowdownConstants:
    """Test blowdown constants values."""

    def test_water_density(self):
        """Test water density constant."""
        assert BlowdownConstants.WATER_DENSITY_LB_GAL == 8.34

    def test_hours_per_year(self):
        """Test hours per year constant."""
        assert BlowdownConstants.HOURS_PER_YEAR == 8760

    def test_steam_properties_defined(self):
        """Test steam properties table is defined."""
        assert len(BlowdownConstants.STEAM_PROPERTIES) > 0
        assert 0 in BlowdownConstants.STEAM_PROPERTIES
        assert 100 in BlowdownConstants.STEAM_PROPERTIES
        assert 300 in BlowdownConstants.STEAM_PROPERTIES

    def test_steam_properties_format(self):
        """Test steam properties have correct format."""
        for pressure, props in BlowdownConstants.STEAM_PROPERTIES.items():
            assert len(props) == 3  # (sat_temp, hf, hfg)
            sat_temp, hf, hfg = props
            assert sat_temp > 212  # Above atmospheric boiling
            assert hf > 0  # Positive enthalpy
            assert hfg > 0  # Positive latent heat

    def test_blowdown_limits(self):
        """Test blowdown limits are reasonable."""
        assert BlowdownConstants.MIN_BLOWDOWN_PCT >= 0.5
        assert BlowdownConstants.MAX_BLOWDOWN_PCT <= 15
        assert BlowdownConstants.MIN_CYCLES >= 2
        assert BlowdownConstants.MAX_CYCLES_RECOMMENDED <= 15


class TestRecommendationGeneration:
    """Test recommendation generation logic."""

    @pytest.fixture
    def optimizer(self):
        return BlowdownOptimizer()

    def test_recommendations_for_low_cycles(self, optimizer):
        """Test recommendations for low cycles of concentration."""
        input_data = BlowdownInput(
            continuous_blowdown_rate_pct=20.0,  # High blowdown
            blowdown_type=BlowdownType.CONTINUOUS,
            boiler_tds_ppm=500.0,  # Low TDS
            feedwater_tds_ppm=100.0,  # High feedwater TDS
            tds_max_ppm=2500.0,
            steam_flow_rate_lb_hr=50000.0,
            operating_pressure_psig=450.0,
        )
        result = optimizer.optimize(input_data)

        # Current cycles = 500/100 = 5 (low)
        # Optimal cycles = 2500/100 = 10 (capped)
        # Should recommend increasing cycles
        rec_text = " ".join(result.recommendations).lower()
        assert "increase" in rec_text or "cycles" in rec_text

    def test_recommendations_for_high_cycles(self, optimizer):
        """Test recommendations for excessively high cycles."""
        input_data = BlowdownInput(
            continuous_blowdown_rate_pct=2.0,  # Very low blowdown
            blowdown_type=BlowdownType.CONTINUOUS,
            boiler_tds_ppm=6000.0,  # Very high TDS
            feedwater_tds_ppm=100.0,
            tds_max_ppm=2500.0,
            steam_flow_rate_lb_hr=50000.0,
            operating_pressure_psig=450.0,
        )
        result = optimizer.optimize(input_data)

        # Current cycles = 6000/100 = 60 (very high)
        # Should warn about carryover/scaling risk
        rec_text = " ".join(result.recommendations).lower()
        assert "warning" in rec_text or "carryover" in rec_text or "scaling" in rec_text

    def test_recommendations_for_no_heat_recovery(self, optimizer, blowdown_input_optimize):
        """Test recommendations for heat recovery opportunity."""
        result = optimizer.optimize(blowdown_input_optimize)

        if result.energy_savings_mmbtu_yr > 100:
            rec_text = " ".join(result.recommendations).lower()
            assert "heat recovery" in rec_text


class TestPerformance:
    """Performance tests for blowdown optimizer."""

    @pytest.fixture
    def optimizer(self):
        return BlowdownOptimizer()

    @pytest.mark.performance
    def test_optimization_performance(self, optimizer, blowdown_input_optimize):
        """Test optimization completes within performance target."""
        import time
        start = time.perf_counter()
        result = optimizer.optimize(blowdown_input_optimize)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50  # Should complete in < 50ms
        assert result.processing_time_ms < 50

    @pytest.mark.performance
    def test_batch_optimization_performance(self, optimizer, blowdown_input_optimize):
        """Test batch optimization maintains throughput."""
        import time
        num_optimizations = 100

        start = time.perf_counter()
        for _ in range(num_optimizations):
            optimizer.optimize(blowdown_input_optimize)
        elapsed_s = time.perf_counter() - start

        throughput = num_optimizations / elapsed_s
        assert throughput > 50  # At least 50 optimizations/second
