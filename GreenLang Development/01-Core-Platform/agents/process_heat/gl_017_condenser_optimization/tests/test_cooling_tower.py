"""
GL-017 CONDENSYNC Agent - Cooling Tower Optimizer Tests

Unit tests for CoolingTowerOptimizer including thermal efficiency,
water balance, chemistry compliance, and blowdown optimization.

Coverage targets:
    - Thermal efficiency calculation
    - Evaporation and drift loss
    - Cycles of concentration
    - Chemistry compliance checking
    - Scaling/corrosion potential
    - Blowdown optimization
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from greenlang.agents.process_heat.gl_017_condenser_optimization.cooling_tower import (
    CoolingTowerOptimizer,
    CoolingTowerConstants,
    CoolingTowerReading,
)
from greenlang.agents.process_heat.gl_017_condenser_optimization.config import (
    CoolingTowerConfig,
)
from greenlang.agents.process_heat.gl_017_condenser_optimization.schemas import (
    CoolingTowerResult,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def cooling_tower_config():
    """Create default cooling tower configuration."""
    return CoolingTowerConfig()


@pytest.fixture
def optimizer(cooling_tower_config):
    """Create CoolingTowerOptimizer instance."""
    return CoolingTowerOptimizer(cooling_tower_config)


# =============================================================================
# CONSTANTS TESTS
# =============================================================================

class TestCoolingTowerConstants:
    """Test CoolingTowerConstants values."""

    def test_water_density(self):
        """Test water density constant."""
        assert CoolingTowerConstants.WATER_DENSITY_LB_GAL == 8.34

    def test_evaporation_factor(self):
        """Test evaporation factor."""
        assert CoolingTowerConstants.EVAPORATION_FACTOR == 0.0008

    def test_drift_loss_values(self):
        """Test drift loss values."""
        assert CoolingTowerConstants.TYPICAL_DRIFT_LOSS_PCT < CoolingTowerConstants.OLD_DRIFT_LOSS_PCT

    def test_min_cycles(self):
        """Test minimum cycles value."""
        assert CoolingTowerConstants.MIN_CYCLES == 1.5

    def test_water_costs(self):
        """Test water cost constants exist."""
        assert CoolingTowerConstants.WATER_COST_PER_1000_GAL > 0
        assert CoolingTowerConstants.SEWER_COST_PER_1000_GAL > 0


# =============================================================================
# OPTIMIZER INITIALIZATION TESTS
# =============================================================================

class TestOptimizerInitialization:
    """Test optimizer initialization."""

    def test_basic_initialization(self, cooling_tower_config):
        """Test optimizer initializes correctly."""
        optimizer = CoolingTowerOptimizer(cooling_tower_config)
        assert optimizer is not None
        assert optimizer.config == cooling_tower_config

    def test_history_empty(self, optimizer):
        """Test history is empty on initialization."""
        assert len(optimizer._history) == 0


# =============================================================================
# THERMAL EFFICIENCY TESTS
# =============================================================================

class TestThermalEfficiency:
    """Test thermal efficiency calculation."""

    def test_basic_efficiency(self, optimizer):
        """Test basic efficiency calculation."""
        efficiency = optimizer._calculate_thermal_efficiency(
            range_f=20.0,
            approach_f=8.0,
            wet_bulb_f=78.0,
        )

        # Effectiveness = 20 / (20 + 8) = 71.4%
        # May be adjusted by design ratios
        assert 50 < efficiency < 150

    def test_zero_range_approach(self, optimizer):
        """Test efficiency with zero values."""
        efficiency = optimizer._calculate_thermal_efficiency(
            range_f=0.0,
            approach_f=0.0,
            wet_bulb_f=78.0,
        )

        assert efficiency == 0.0

    def test_higher_range_better_efficiency(self, optimizer):
        """Test higher range gives better efficiency."""
        eff_low = optimizer._calculate_thermal_efficiency(
            range_f=10.0,
            approach_f=8.0,
            wet_bulb_f=78.0,
        )

        eff_high = optimizer._calculate_thermal_efficiency(
            range_f=25.0,
            approach_f=8.0,
            wet_bulb_f=78.0,
        )

        assert eff_high > eff_low

    def test_lower_approach_better_efficiency(self, optimizer):
        """Test lower approach gives better efficiency."""
        eff_low = optimizer._calculate_thermal_efficiency(
            range_f=20.0,
            approach_f=5.0,
            wet_bulb_f=78.0,
        )

        eff_high_approach = optimizer._calculate_thermal_efficiency(
            range_f=20.0,
            approach_f=12.0,
            wet_bulb_f=78.0,
        )

        assert eff_low > eff_high_approach


# =============================================================================
# L/G RATIO TESTS
# =============================================================================

class TestLGRatio:
    """Test liquid-to-gas ratio calculation."""

    def test_basic_lg_ratio(self, optimizer):
        """Test basic L/G ratio calculation."""
        lg = optimizer._calculate_lg_ratio(
            circulation_gpm=100000.0,
            range_f=20.0,
            wet_bulb_f=78.0,
        )

        # Typical L/G is 0.8 - 1.5
        assert 0.5 < lg < 2.0

    def test_higher_range_lower_lg(self, optimizer):
        """Test higher range needs lower L/G."""
        lg_low_range = optimizer._calculate_lg_ratio(
            circulation_gpm=100000.0,
            range_f=10.0,
            wet_bulb_f=78.0,
        )

        lg_high_range = optimizer._calculate_lg_ratio(
            circulation_gpm=100000.0,
            range_f=25.0,
            wet_bulb_f=78.0,
        )

        assert lg_high_range < lg_low_range


# =============================================================================
# EVAPORATION TESTS
# =============================================================================

class TestEvaporation:
    """Test evaporation calculation."""

    def test_evaporation_formula(self, optimizer):
        """Test evaporation calculation formula."""
        evap = optimizer._calculate_evaporation(
            circulation_gpm=100000.0,
            range_f=20.0,
        )

        # E = C * R * Factor = 100000 * 20 * 0.0008 = 1600 GPM
        expected = 100000.0 * 20.0 * CoolingTowerConstants.EVAPORATION_FACTOR
        assert evap == pytest.approx(expected, rel=0.01)

    def test_evaporation_vs_range(self, optimizer):
        """Test evaporation increases with range."""
        evap_10 = optimizer._calculate_evaporation(100000.0, 10.0)
        evap_20 = optimizer._calculate_evaporation(100000.0, 20.0)

        assert evap_20 == 2 * evap_10

    def test_evaporation_vs_flow(self, optimizer):
        """Test evaporation increases with flow."""
        evap_50k = optimizer._calculate_evaporation(50000.0, 20.0)
        evap_100k = optimizer._calculate_evaporation(100000.0, 20.0)

        assert evap_100k == 2 * evap_50k


# =============================================================================
# DRIFT LOSS TESTS
# =============================================================================

class TestDriftLoss:
    """Test drift loss calculation."""

    def test_drift_formula(self, optimizer):
        """Test drift loss calculation."""
        drift = optimizer._calculate_drift(100000.0)

        # D = C * 0.005% = 100000 * 0.00005 = 5 GPM
        expected = 100000.0 * CoolingTowerConstants.TYPICAL_DRIFT_LOSS_PCT / 100
        assert drift == pytest.approx(expected, rel=0.01)

    def test_drift_vs_flow(self, optimizer):
        """Test drift increases with flow."""
        drift_50k = optimizer._calculate_drift(50000.0)
        drift_100k = optimizer._calculate_drift(100000.0)

        assert drift_100k == 2 * drift_50k


# =============================================================================
# CYCLES OF CONCENTRATION TESTS
# =============================================================================

class TestCyclesOfConcentration:
    """Test cycles of concentration calculation."""

    def test_cycles_from_conductivity(self, optimizer):
        """Test cycles from conductivity method."""
        cycles = optimizer._calculate_cycles(
            makeup_gpm=None,
            evaporation_gpm=800.0,
            drift_gpm=5.0,
            makeup_conductivity=500.0,
            tower_conductivity=2500.0,
        )

        # Cycles = 2500 / 500 = 5.0
        assert cycles == pytest.approx(5.0, rel=0.01)

    def test_cycles_from_mass_balance(self, optimizer):
        """Test cycles from mass balance method."""
        cycles = optimizer._calculate_cycles(
            makeup_gpm=1000.0,
            evaporation_gpm=800.0,
            drift_gpm=5.0,
            makeup_conductivity=None,
            tower_conductivity=None,
        )

        # Blowdown = Makeup - Evap - Drift = 1000 - 800 - 5 = 195
        # Cycles = Makeup / Blowdown = 1000 / 195 = 5.13
        assert 4.0 < cycles < 6.0

    def test_cycles_default(self, optimizer):
        """Test cycles defaults to target when no data."""
        cycles = optimizer._calculate_cycles(
            makeup_gpm=None,
            evaporation_gpm=800.0,
            drift_gpm=5.0,
            makeup_conductivity=None,
            tower_conductivity=None,
        )

        assert cycles == optimizer.config.target_cycles_concentration


# =============================================================================
# BLOWDOWN CALCULATION TESTS
# =============================================================================

class TestBlowdownCalculation:
    """Test blowdown calculation."""

    def test_required_blowdown(self, optimizer):
        """Test required blowdown calculation."""
        blowdown = optimizer._calculate_required_blowdown(
            evaporation_gpm=800.0,
            drift_gpm=5.0,
            cycles=5.0,
        )

        # BD = E / (C - 1) - D = 800 / 4 - 5 = 195 GPM
        expected = 800.0 / (5.0 - 1.0) - 5.0
        assert blowdown == pytest.approx(expected, rel=0.01)

    def test_blowdown_minimum_cycles(self, optimizer):
        """Test blowdown with minimum cycles."""
        blowdown = optimizer._calculate_required_blowdown(
            evaporation_gpm=800.0,
            drift_gpm=5.0,
            cycles=1.0,  # Will be raised to 1.5
        )

        # Should handle cycles <= 1
        assert blowdown > 0

    def test_higher_cycles_lower_blowdown(self, optimizer):
        """Test higher cycles means lower blowdown."""
        bd_low = optimizer._calculate_required_blowdown(800.0, 5.0, 3.0)
        bd_high = optimizer._calculate_required_blowdown(800.0, 5.0, 6.0)

        assert bd_high < bd_low


# =============================================================================
# MAKEUP CALCULATION TESTS
# =============================================================================

class TestMakeupCalculation:
    """Test makeup water calculation."""

    def test_makeup_formula(self, optimizer):
        """Test makeup water formula."""
        makeup = optimizer._calculate_required_makeup(
            evaporation_gpm=800.0,
            blowdown_gpm=195.0,
            drift_gpm=5.0,
        )

        # M = E + BD + D = 800 + 195 + 5 = 1000 GPM
        expected = 800.0 + 195.0 + 5.0
        assert makeup == pytest.approx(expected, rel=0.01)


# =============================================================================
# CHEMISTRY COMPLIANCE TESTS
# =============================================================================

class TestChemistryCompliance:
    """Test chemistry compliance checking."""

    def test_compliant_chemistry(self, optimizer):
        """Test compliant chemistry."""
        compliant, deviations = optimizer._check_chemistry_compliance(
            ph=8.0,
            calcium_ppm=400.0,
            silica_ppm=100.0,
            chlorides_ppm=200.0,
            conductivity=2000.0,
        )

        assert compliant is True
        assert len(deviations) == 0

    def test_ph_out_of_range(self, optimizer):
        """Test pH out of range detection."""
        compliant, deviations = optimizer._check_chemistry_compliance(
            ph=10.0,  # Too high
            calcium_ppm=400.0,
            silica_ppm=100.0,
            chlorides_ppm=200.0,
            conductivity=2000.0,
        )

        assert compliant is False
        assert any("pH" in d for d in deviations)

    def test_calcium_exceeded(self, optimizer):
        """Test calcium limit exceeded."""
        compliant, deviations = optimizer._check_chemistry_compliance(
            ph=8.0,
            calcium_ppm=1000.0,  # Above limit
            silica_ppm=100.0,
            chlorides_ppm=200.0,
            conductivity=2000.0,
        )

        assert compliant is False
        assert any("Calcium" in d for d in deviations)

    def test_silica_exceeded(self, optimizer):
        """Test silica limit exceeded."""
        compliant, deviations = optimizer._check_chemistry_compliance(
            ph=8.0,
            calcium_ppm=400.0,
            silica_ppm=200.0,  # Above limit
            chlorides_ppm=200.0,
            conductivity=2000.0,
        )

        assert compliant is False
        assert any("Silica" in d for d in deviations)

    def test_multiple_deviations(self, optimizer):
        """Test multiple chemistry deviations."""
        compliant, deviations = optimizer._check_chemistry_compliance(
            ph=6.0,  # Low
            calcium_ppm=1000.0,  # High
            silica_ppm=200.0,  # High
            chlorides_ppm=600.0,  # High
            conductivity=4000.0,  # High
        )

        assert compliant is False
        assert len(deviations) >= 4


# =============================================================================
# SCALING POTENTIAL TESTS
# =============================================================================

class TestScalingPotential:
    """Test scaling potential assessment."""

    def test_unknown_scaling(self, optimizer):
        """Test unknown scaling when data missing."""
        potential = optimizer._assess_scaling_potential(
            ph=None,
            calcium_ppm=None,
            temperature_f=100.0,
        )

        assert potential == "unknown"

    def test_low_scaling(self, optimizer):
        """Test low scaling potential."""
        potential = optimizer._assess_scaling_potential(
            ph=7.5,
            calcium_ppm=200.0,
            temperature_f=90.0,
        )

        assert potential == "low"

    def test_high_scaling(self, optimizer):
        """Test high scaling potential."""
        potential = optimizer._assess_scaling_potential(
            ph=9.0,  # High pH
            calcium_ppm=600.0,  # High calcium
            temperature_f=110.0,  # High temp
        )

        assert potential in ["moderate", "high"]


# =============================================================================
# CORROSION POTENTIAL TESTS
# =============================================================================

class TestCorrosionPotential:
    """Test corrosion potential assessment."""

    def test_unknown_corrosion(self, optimizer):
        """Test unknown corrosion when data missing."""
        potential = optimizer._assess_corrosion_potential(
            ph=None,
            chlorides_ppm=None,
        )

        assert potential == "unknown"

    def test_low_corrosion(self, optimizer):
        """Test low corrosion potential."""
        potential = optimizer._assess_corrosion_potential(
            ph=8.0,
            chlorides_ppm=100.0,
        )

        assert potential == "low"

    def test_high_corrosion(self, optimizer):
        """Test high corrosion potential."""
        potential = optimizer._assess_corrosion_potential(
            ph=6.5,  # Low pH
            chlorides_ppm=400.0,  # High chlorides
        )

        assert potential == "high"


# =============================================================================
# OPTIMAL CYCLES TESTS
# =============================================================================

class TestOptimalCycles:
    """Test optimal cycles calculation."""

    def test_limited_by_conductivity(self, optimizer):
        """Test cycles limited by conductivity."""
        cycles = optimizer._calculate_optimal_cycles(
            makeup_conductivity=1000.0,  # High
            calcium_ppm=None,
            silica_ppm=None,
        )

        # Max cond = 3000, so max cycles = 3
        assert cycles <= 3.0

    def test_limited_by_calcium(self, optimizer):
        """Test cycles limited by calcium."""
        cycles = optimizer._calculate_optimal_cycles(
            makeup_conductivity=None,
            calcium_ppm=200.0,  # High
            silica_ppm=None,
        )

        # Max Ca = 800, so max cycles = 4
        assert cycles <= 4.5

    def test_no_limits(self, optimizer):
        """Test optimal cycles with no limiting factors."""
        cycles = optimizer._calculate_optimal_cycles(
            makeup_conductivity=None,
            calcium_ppm=None,
            silica_ppm=None,
        )

        # Should be config max * 0.9
        assert cycles == optimizer.config.max_cycles_concentration * 0.9


# =============================================================================
# ANALYZE COOLING TOWER TESTS
# =============================================================================

class TestAnalyzeCoolingTower:
    """Test main analysis method."""

    def test_basic_analysis(self, optimizer):
        """Test basic cooling tower analysis."""
        result = optimizer.analyze_cooling_tower(
            hot_water_temp_f=105.0,
            cold_water_temp_f=85.0,
            wet_bulb_temp_f=78.0,
            circulation_flow_gpm=100000.0,
        )

        assert isinstance(result, CoolingTowerResult)

    def test_result_components(self, optimizer):
        """Test all result components are populated."""
        result = optimizer.analyze_cooling_tower(
            hot_water_temp_f=105.0,
            cold_water_temp_f=85.0,
            wet_bulb_temp_f=78.0,
            circulation_flow_gpm=100000.0,
            makeup_flow_gpm=1000.0,
            blowdown_flow_gpm=200.0,
            makeup_conductivity_umhos=500.0,
            tower_conductivity_umhos=2500.0,
            ph=8.0,
            calcium_ppm=400.0,
        )

        assert result.thermal_efficiency_pct > 0
        assert result.approach_f == 7.0  # 85 - 78
        assert result.range_f == 20.0  # 105 - 85
        assert result.cycles_of_concentration > 0
        assert result.evaporation_rate_gpm > 0

    def test_range_and_approach(self, optimizer):
        """Test range and approach calculations."""
        result = optimizer.analyze_cooling_tower(
            hot_water_temp_f=105.0,
            cold_water_temp_f=85.0,
            wet_bulb_temp_f=78.0,
            circulation_flow_gpm=100000.0,
        )

        assert result.range_f == pytest.approx(20.0, rel=0.01)
        assert result.approach_f == pytest.approx(7.0, rel=0.01)


# =============================================================================
# BLOWDOWN OPTIMIZATION TESTS
# =============================================================================

class TestBlowdownOptimization:
    """Test blowdown optimization method."""

    def test_optimize_blowdown(self, optimizer):
        """Test blowdown optimization."""
        result = optimizer.optimize_blowdown(
            current_cycles=3.0,
            evaporation_gpm=800.0,
            drift_gpm=5.0,
            makeup_conductivity_umhos=500.0,
        )

        assert "current_cycles" in result
        assert "target_cycles" in result
        assert "current_blowdown_gpm" in result
        assert "optimal_blowdown_gpm" in result

    def test_savings_calculation(self, optimizer):
        """Test savings are calculated."""
        result = optimizer.optimize_blowdown(
            current_cycles=3.0,  # Low cycles
            evaporation_gpm=800.0,
            drift_gpm=5.0,
            makeup_conductivity_umhos=500.0,
        )

        # Should recommend higher cycles and show savings
        assert result["target_cycles"] > 3.0
        assert result["blowdown_reduction_gpm"] > 0
        assert result["annual_savings_usd"] > 0

    def test_no_savings_at_optimal(self, optimizer):
        """Test no savings when already at optimal."""
        result = optimizer.optimize_blowdown(
            current_cycles=5.5,  # Near optimal
            evaporation_gpm=800.0,
            drift_gpm=5.0,
            makeup_conductivity_umhos=500.0,
            target_cycles=5.5,
        )

        assert result["blowdown_reduction_gpm"] == pytest.approx(0.0, abs=1.0)


# =============================================================================
# WATER SAVINGS TESTS
# =============================================================================

class TestWaterSavings:
    """Test water savings calculation."""

    def test_positive_savings(self, optimizer):
        """Test positive water savings."""
        savings = optimizer._calculate_water_savings(
            actual_blowdown=300.0,
            optimal_blowdown=200.0,
        )

        assert savings == 100.0

    def test_no_savings(self, optimizer):
        """Test no savings when already optimal."""
        savings = optimizer._calculate_water_savings(
            actual_blowdown=200.0,
            optimal_blowdown=200.0,
        )

        assert savings == 0.0

    def test_no_negative_savings(self, optimizer):
        """Test no negative savings."""
        savings = optimizer._calculate_water_savings(
            actual_blowdown=150.0,  # Below optimal
            optimal_blowdown=200.0,
        )

        assert savings == 0.0


# =============================================================================
# CHEMICAL SAVINGS TESTS
# =============================================================================

class TestChemicalSavings:
    """Test chemical savings calculation."""

    def test_savings_with_higher_cycles(self, optimizer):
        """Test savings when increasing cycles."""
        savings = optimizer._calculate_chemical_savings(
            current_cycles=3.0,
            optimal_cycles=6.0,
        )

        # Higher cycles = 50% reduction
        assert savings == pytest.approx(50.0, rel=0.1)

    def test_no_savings_when_optimal(self, optimizer):
        """Test no savings when already optimal."""
        savings = optimizer._calculate_chemical_savings(
            current_cycles=6.0,
            optimal_cycles=6.0,
        )

        assert savings == 0.0

    def test_no_savings_when_above_optimal(self, optimizer):
        """Test no savings when above optimal."""
        savings = optimizer._calculate_chemical_savings(
            current_cycles=7.0,
            optimal_cycles=6.0,
        )

        assert savings == 0.0


# =============================================================================
# HISTORY TESTS
# =============================================================================

class TestCoolingTowerHistory:
    """Test cooling tower history management."""

    def test_record_reading(self, optimizer):
        """Test recording readings."""
        optimizer._record_reading(105.0, 85.0, 78.0, 5.0, 200.0)

        assert len(optimizer._history) == 1

    def test_get_performance_trend(self, optimizer):
        """Test retrieving performance trend."""
        for i in range(5):
            optimizer._record_reading(105.0 + i, 85.0, 78.0, 5.0, 200.0)

        trends = optimizer.get_performance_trend(hours=24)

        assert "range" in trends
        assert "approach" in trends
        assert "cycles" in trends
        assert "blowdown" in trends


# =============================================================================
# CALCULATION COUNT TESTS
# =============================================================================

class TestCalculationCount:
    """Test calculation counting."""

    def test_count_increments(self, optimizer):
        """Test calculation count increments."""
        initial = optimizer.calculation_count

        optimizer.analyze_cooling_tower(
            hot_water_temp_f=105.0,
            cold_water_temp_f=85.0,
            wet_bulb_temp_f=78.0,
            circulation_flow_gpm=100000.0,
        )

        assert optimizer.calculation_count == initial + 1
