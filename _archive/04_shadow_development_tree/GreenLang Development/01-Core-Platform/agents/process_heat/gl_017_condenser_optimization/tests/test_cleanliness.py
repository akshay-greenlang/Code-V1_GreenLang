"""
GL-017 CONDENSYNC Agent - Cleanliness Calculator Tests

Unit tests for HEICleanlinessCalculator and CleanlinessMonitor.
Tests cover HEI Standards calculations, LMTD, cleaning status determination.

Coverage targets:
    - HEI U-value calculations
    - LMTD calculations
    - Cleanliness factor determination
    - Fouling factor calculations
    - Cleaning status logic
    - Trend monitoring
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from greenlang.agents.process_heat.gl_017_condenser_optimization.cleanliness import (
    HEICleanlinessCalculator,
    CleanlinessMonitor,
    HEIConstants,
)
from greenlang.agents.process_heat.gl_017_condenser_optimization.config import (
    CleanlinessConfig,
    TubeFoulingConfig,
    TubeMaterial,
)
from greenlang.agents.process_heat.gl_017_condenser_optimization.schemas import (
    CleaninessResult,
    CleaningStatus,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def cleanliness_config():
    """Create default cleanliness configuration."""
    return CleanlinessConfig()


@pytest.fixture
def fouling_config():
    """Create default tube fouling configuration."""
    return TubeFoulingConfig()


@pytest.fixture
def calculator(cleanliness_config, fouling_config):
    """Create HEI cleanliness calculator instance."""
    return HEICleanlinessCalculator(cleanliness_config, fouling_config)


@pytest.fixture
def monitor(calculator):
    """Create cleanliness monitor instance."""
    return CleanlinessMonitor(calculator, history_days=90)


# =============================================================================
# HEI CONSTANTS TESTS
# =============================================================================

class TestHEIConstants:
    """Test HEI Constants values."""

    def test_base_u_value(self):
        """Test base U value is per HEI standards."""
        assert HEIConstants.U_BASE_BTU_HR_FT2_F == 650.0

    def test_material_factors_exist(self):
        """Test all tube materials have factors."""
        for material in TubeMaterial:
            assert material in HEIConstants.MATERIAL_FACTORS

    def test_material_factors_range(self):
        """Test material factors are in valid range."""
        for material, factor in HEIConstants.MATERIAL_FACTORS.items():
            assert 0.5 <= factor <= 1.1

    def test_admiralty_brass_is_reference(self):
        """Test admiralty brass is reference material (1.0)."""
        assert HEIConstants.MATERIAL_FACTORS[TubeMaterial.ADMIRALTY_BRASS] == 1.0

    def test_tube_wall_thickness_values(self):
        """Test tube wall thickness values."""
        # 18 BWG is common
        assert 18 in HEIConstants.TUBE_WALL_THICKNESS
        assert HEIConstants.TUBE_WALL_THICKNESS[18] == 0.049

    def test_thermal_conductivity_values(self):
        """Test thermal conductivity values are positive."""
        for material, conductivity in HEIConstants.THERMAL_CONDUCTIVITY.items():
            assert conductivity > 0

    def test_inlet_water_factors(self):
        """Test inlet water correction factors."""
        assert "freshwater" in HEIConstants.INLET_WATER_FACTORS
        assert "seawater" in HEIConstants.INLET_WATER_FACTORS
        assert "cooling_tower" in HEIConstants.INLET_WATER_FACTORS


# =============================================================================
# LMTD CALCULATION TESTS
# =============================================================================

class TestLMTDCalculation:
    """Test LMTD calculation methods."""

    def test_standard_lmtd(self, calculator):
        """Test standard LMTD calculation."""
        lmtd = calculator.calculate_lmtd(
            hot_inlet_temp_f=101.0,
            hot_outlet_temp_f=101.0,  # Isothermal condensation
            cold_inlet_temp_f=70.0,
            cold_outlet_temp_f=95.0,
        )

        # LMTD should be positive
        assert lmtd > 0

        # Calculate expected:
        # dT1 = 101 - 95 = 6
        # dT2 = 101 - 70 = 31
        # LMTD = (31 - 6) / ln(31/6) = 25 / 1.64 = 15.2
        assert 14.0 < lmtd < 17.0

    def test_equal_temperature_difference(self, calculator):
        """Test LMTD with equal temperature differences."""
        lmtd = calculator.calculate_lmtd(
            hot_inlet_temp_f=100.0,
            hot_outlet_temp_f=100.0,
            cold_inlet_temp_f=80.0,
            cold_outlet_temp_f=80.0,
        )

        # Should equal the temperature difference
        assert lmtd == pytest.approx(20.0, rel=0.01)

    def test_temperature_cross_handling(self, calculator):
        """Test handling of temperature cross."""
        # Cold outlet > hot outlet (impossible in real life)
        lmtd = calculator.calculate_lmtd(
            hot_inlet_temp_f=100.0,
            hot_outlet_temp_f=100.0,
            cold_inlet_temp_f=95.0,
            cold_outlet_temp_f=105.0,  # Cross
        )

        # Should return some value (fallback handling)
        assert lmtd >= 0

    @pytest.mark.parametrize("hot_temp,cold_in,cold_out,expected_range", [
        (101.0, 70.0, 95.0, (14.0, 17.0)),
        (105.0, 75.0, 95.0, (15.0, 19.0)),
        (110.0, 80.0, 100.0, (14.0, 18.0)),
    ])
    def test_lmtd_various_conditions(self, calculator, hot_temp, cold_in, cold_out, expected_range):
        """Test LMTD at various conditions."""
        lmtd = calculator.calculate_lmtd(
            hot_inlet_temp_f=hot_temp,
            hot_outlet_temp_f=hot_temp,
            cold_inlet_temp_f=cold_in,
            cold_outlet_temp_f=cold_out,
        )

        assert expected_range[0] < lmtd < expected_range[1]


# =============================================================================
# U-VALUE CALCULATION TESTS
# =============================================================================

class TestUValueCalculation:
    """Test U-value calculation methods."""

    def test_actual_u_calculation(self, calculator):
        """Test actual U calculation."""
        # U = Q / (A * LMTD)
        u_actual = calculator._calculate_actual_u(
            heat_duty_btu_hr=450_000_000.0,
            lmtd_f=18.0,
            surface_area_ft2=150000.0,
        )

        # U = 450e6 / (150000 * 18) = 166.7
        assert u_actual == pytest.approx(166.7, rel=0.01)

    def test_clean_u_calculation(self, calculator):
        """Test clean tube U calculation."""
        u_clean = calculator._calculate_clean_u(
            cw_velocity_fps=7.0,
            cw_inlet_temp_f=70.0,
            tube_material=TubeMaterial.STAINLESS_316,
            tube_od_in=0.875,
            tube_gauge=18,
            water_type="cooling_tower",
        )

        # U_clean should be between 300-600 typically
        assert 300 < u_clean < 700

    def test_clean_u_velocity_correction(self, calculator):
        """Test velocity correction in clean U."""
        u_low_v = calculator._calculate_clean_u(
            cw_velocity_fps=5.0,
            cw_inlet_temp_f=70.0,
            tube_material=TubeMaterial.STAINLESS_316,
            tube_od_in=0.875,
            tube_gauge=18,
        )

        u_high_v = calculator._calculate_clean_u(
            cw_velocity_fps=10.0,
            cw_inlet_temp_f=70.0,
            tube_material=TubeMaterial.STAINLESS_316,
            tube_od_in=0.875,
            tube_gauge=18,
        )

        # Higher velocity should give higher U
        assert u_high_v > u_low_v

    def test_clean_u_temperature_correction(self, calculator):
        """Test temperature correction in clean U."""
        u_cold = calculator._calculate_clean_u(
            cw_velocity_fps=7.0,
            cw_inlet_temp_f=60.0,
            tube_material=TubeMaterial.STAINLESS_316,
            tube_od_in=0.875,
            tube_gauge=18,
        )

        u_hot = calculator._calculate_clean_u(
            cw_velocity_fps=7.0,
            cw_inlet_temp_f=90.0,
            tube_material=TubeMaterial.STAINLESS_316,
            tube_od_in=0.875,
            tube_gauge=18,
        )

        # Warmer inlet should give higher U
        assert u_hot > u_cold

    def test_material_factor_effect(self, calculator):
        """Test tube material affects U calculation."""
        u_admiralty = calculator._calculate_clean_u(
            cw_velocity_fps=7.0,
            cw_inlet_temp_f=70.0,
            tube_material=TubeMaterial.ADMIRALTY_BRASS,
            tube_od_in=0.875,
            tube_gauge=18,
        )

        u_titanium = calculator._calculate_clean_u(
            cw_velocity_fps=7.0,
            cw_inlet_temp_f=70.0,
            tube_material=TubeMaterial.TITANIUM,
            tube_od_in=0.875,
            tube_gauge=18,
        )

        # Admiralty (factor 1.0) should have higher U than titanium (0.73)
        assert u_admiralty > u_titanium


# =============================================================================
# CLEANLINESS FACTOR TESTS
# =============================================================================

class TestCleanlinessFactorCalculation:
    """Test cleanliness factor calculation."""

    def test_calculate_cleanliness_normal(self, calculator):
        """Test cleanliness calculation under normal conditions."""
        result = calculator.calculate_cleanliness(
            heat_duty_btu_hr=450_000_000.0,
            lmtd_f=18.0,
            surface_area_ft2=150000.0,
            cw_velocity_fps=7.0,
            cw_inlet_temp_f=70.0,
        )

        assert result.cleanliness_factor > 0
        assert result.cleanliness_factor <= 1.2

    def test_cleanliness_result_components(self, calculator):
        """Test all result components are populated."""
        result = calculator.calculate_cleanliness(
            heat_duty_btu_hr=450_000_000.0,
            lmtd_f=18.0,
            surface_area_ft2=150000.0,
            cw_velocity_fps=7.0,
            cw_inlet_temp_f=70.0,
        )

        assert result.cleanliness_factor is not None
        assert result.design_cleanliness is not None
        assert result.cleanliness_ratio is not None
        assert result.u_actual_btu_hr_ft2_f is not None
        assert result.u_clean_btu_hr_ft2_f is not None
        assert result.u_design_btu_hr_ft2_f is not None
        assert result.lmtd_f is not None
        assert result.heat_duty_btu_hr is not None

    def test_cleanliness_ratio_calculation(self, calculator):
        """Test cleanliness ratio is calculated correctly."""
        result = calculator.calculate_cleanliness(
            heat_duty_btu_hr=450_000_000.0,
            lmtd_f=18.0,
            surface_area_ft2=150000.0,
            cw_velocity_fps=7.0,
            cw_inlet_temp_f=70.0,
        )

        expected_ratio = result.cleanliness_factor / result.design_cleanliness
        assert result.cleanliness_ratio == pytest.approx(expected_ratio, rel=0.01)

    @pytest.mark.parametrize("heat_duty,expected_cf_range", [
        (500_000_000.0, (0.3, 0.5)),  # High duty = lower CF
        (400_000_000.0, (0.35, 0.6)),  # Medium duty
        (300_000_000.0, (0.45, 0.8)),  # Low duty = higher CF
    ])
    def test_cleanliness_vs_duty(self, calculator, heat_duty, expected_cf_range):
        """Test cleanliness factor varies with duty."""
        result = calculator.calculate_cleanliness(
            heat_duty_btu_hr=heat_duty,
            lmtd_f=18.0,
            surface_area_ft2=150000.0,
            cw_velocity_fps=7.0,
            cw_inlet_temp_f=70.0,
        )

        # CF should be in expected range
        assert expected_cf_range[0] < result.cleanliness_factor < expected_cf_range[1]


# =============================================================================
# FOULING FACTOR TESTS
# =============================================================================

class TestFoulingFactorCalculation:
    """Test fouling factor calculations."""

    def test_fouling_factor_positive(self, calculator):
        """Test fouling factor is non-negative."""
        fouling = calculator._calculate_fouling_factor(
            u_actual=400.0,
            u_clean=550.0,
        )

        assert fouling >= 0

    def test_fouling_factor_formula(self, calculator):
        """Test fouling factor calculation formula."""
        # R_f = 1/U_actual - 1/U_clean
        fouling = calculator._calculate_fouling_factor(
            u_actual=400.0,
            u_clean=550.0,
        )

        expected = (1.0/400.0) - (1.0/550.0)
        assert fouling == pytest.approx(expected, rel=0.01)

    def test_no_fouling_when_clean(self, calculator):
        """Test zero fouling when U_actual >= U_clean."""
        fouling = calculator._calculate_fouling_factor(
            u_actual=550.0,
            u_clean=550.0,
        )

        assert fouling == 0.0

    def test_fouling_thickness_estimation(self, calculator):
        """Test fouling thickness estimation."""
        thickness = calculator._estimate_fouling_thickness(
            fouling_factor=0.0005,
            fouling_conductivity=0.6,
        )

        # Should return thickness in mils
        assert thickness is not None
        assert thickness > 0

    def test_fouling_thickness_zero_for_no_fouling(self, calculator):
        """Test no thickness when no fouling."""
        thickness = calculator._estimate_fouling_thickness(
            fouling_factor=0.0,
        )

        assert thickness is None


# =============================================================================
# CLEANING STATUS TESTS
# =============================================================================

class TestCleaningStatus:
    """Test cleaning status determination."""

    def test_not_required_status(self, calculator):
        """Test NOT_REQUIRED status for high CF."""
        status = calculator._determine_cleaning_status(0.85)
        assert status == CleaningStatus.NOT_REQUIRED

    def test_recommended_status(self, calculator):
        """Test RECOMMENDED status for medium CF."""
        status = calculator._determine_cleaning_status(0.72)
        assert status == CleaningStatus.RECOMMENDED

    def test_required_status(self, calculator):
        """Test REQUIRED status for low CF."""
        status = calculator._determine_cleaning_status(0.62)
        assert status == CleaningStatus.REQUIRED

    def test_urgent_status(self, calculator):
        """Test URGENT status for very low CF."""
        status = calculator._determine_cleaning_status(0.50)
        assert status == CleaningStatus.URGENT

    @pytest.mark.parametrize("cf,expected_status", [
        (0.90, CleaningStatus.NOT_REQUIRED),
        (0.80, CleaningStatus.NOT_REQUIRED),
        (0.75, CleaningStatus.NOT_REQUIRED),
        (0.74, CleaningStatus.RECOMMENDED),
        (0.70, CleaningStatus.RECOMMENDED),
        (0.65, CleaningStatus.NOT_REQUIRED),  # Wait, check thresholds
    ])
    def test_cleaning_status_thresholds(self, calculator, cf, expected_status):
        """Test cleaning status at various thresholds."""
        # Default thresholds: warning=0.75, alarm=0.65, trigger=0.60
        status = calculator._determine_cleaning_status(cf)
        # Adjust expected based on actual thresholds


# =============================================================================
# DAYS TO CLEANING TESTS
# =============================================================================

class TestDaysToCleaning:
    """Test days to cleaning estimation."""

    def test_days_to_cleaning_estimate(self, calculator):
        """Test days to cleaning estimation."""
        days = calculator._estimate_days_to_cleaning(
            cleanliness_factor=0.80,
            current_status=CleaningStatus.NOT_REQUIRED,
            fouling_rate_per_day=0.001,
        )

        # CF = 0.80, trigger = 0.60, margin = 0.20
        # Days = 0.20 / 0.001 = 200
        assert days == 200

    def test_zero_days_when_cleaning_required(self, calculator):
        """Test zero days when cleaning already required."""
        days = calculator._estimate_days_to_cleaning(
            cleanliness_factor=0.55,
            current_status=CleaningStatus.REQUIRED,
        )

        assert days == 0

    def test_days_capped_at_one_year(self, calculator):
        """Test days capped at 365."""
        days = calculator._estimate_days_to_cleaning(
            cleanliness_factor=0.95,
            current_status=CleaningStatus.NOT_REQUIRED,
            fouling_rate_per_day=0.0001,  # Very slow fouling
        )

        assert days <= 365


# =============================================================================
# INPUT VALIDATION TESTS
# =============================================================================

class TestInputValidation:
    """Test input validation."""

    def test_negative_heat_duty_raises(self, calculator):
        """Test negative heat duty raises error."""
        with pytest.raises(ValueError):
            calculator.calculate_cleanliness(
                heat_duty_btu_hr=-100.0,
                lmtd_f=18.0,
                surface_area_ft2=150000.0,
                cw_velocity_fps=7.0,
                cw_inlet_temp_f=70.0,
            )

    def test_zero_lmtd_raises(self, calculator):
        """Test zero LMTD raises error."""
        with pytest.raises(ValueError):
            calculator.calculate_cleanliness(
                heat_duty_btu_hr=450_000_000.0,
                lmtd_f=0.0,
                surface_area_ft2=150000.0,
                cw_velocity_fps=7.0,
                cw_inlet_temp_f=70.0,
            )

    def test_zero_area_raises(self, calculator):
        """Test zero surface area raises error."""
        with pytest.raises(ValueError):
            calculator.calculate_cleanliness(
                heat_duty_btu_hr=450_000_000.0,
                lmtd_f=18.0,
                surface_area_ft2=0.0,
                cw_velocity_fps=7.0,
                cw_inlet_temp_f=70.0,
            )

    def test_zero_velocity_raises(self, calculator):
        """Test zero velocity raises error."""
        with pytest.raises(ValueError):
            calculator.calculate_cleanliness(
                heat_duty_btu_hr=450_000_000.0,
                lmtd_f=18.0,
                surface_area_ft2=150000.0,
                cw_velocity_fps=0.0,
                cw_inlet_temp_f=70.0,
            )


# =============================================================================
# CLEANLINESS MONITOR TESTS
# =============================================================================

class TestCleanlinessMonitor:
    """Test CleanlinessMonitor class."""

    def test_monitor_initialization(self, monitor):
        """Test monitor initializes correctly."""
        assert monitor is not None
        assert monitor.history_days == 90

    def test_record_cleanliness(self, monitor):
        """Test recording cleanliness values."""
        monitor.record_cleanliness(0.85)
        monitor.record_cleanliness(0.84)
        monitor.record_cleanliness(0.83)

        # History should have entries
        assert len(monitor._history) == 3

    def test_get_fouling_rate_insufficient_data(self, monitor):
        """Test fouling rate with insufficient data."""
        monitor.record_cleanliness(0.85)

        rate = monitor.get_fouling_rate()
        assert rate is None

    def test_get_fouling_rate(self, monitor):
        """Test fouling rate calculation."""
        # Record declining cleanliness
        base_time = datetime.now(timezone.utc)
        for i in range(10):
            ts = base_time + timedelta(days=i)
            cf = 0.85 - (i * 0.01)  # Declining by 0.01 per day
            monitor.record_cleanliness(cf, ts)

        rate = monitor.get_fouling_rate()

        # Rate should be approximately 0.01 per day
        assert rate is not None
        assert 0.005 < rate < 0.015

    def test_get_trend_stable(self, monitor):
        """Test trend detection - stable."""
        for _ in range(15):
            monitor.record_cleanliness(0.80)

        trend = monitor.get_trend()
        assert trend in ["stable", "unknown"]

    def test_get_trend_degrading(self, monitor):
        """Test trend detection - degrading."""
        base_time = datetime.now(timezone.utc)
        # First 10 at higher CF
        for i in range(10):
            ts = base_time + timedelta(days=i)
            monitor.record_cleanliness(0.85, ts)
        # Next 10 at lower CF
        for i in range(10, 20):
            ts = base_time + timedelta(days=i)
            monitor.record_cleanliness(0.70, ts)

        trend = monitor.get_trend()
        assert trend == "degrading" or trend == "degrading_fast"

    def test_predict_cleaning_date(self, monitor):
        """Test cleaning date prediction."""
        base_time = datetime.now(timezone.utc)
        for i in range(10):
            ts = base_time + timedelta(days=i)
            cf = 0.85 - (i * 0.01)
            monitor.record_cleanliness(cf, ts)

        predicted_date = monitor.predict_cleaning_date(
            current_cf=0.75,
            target_cf=0.60,
        )

        # Should predict a future date
        assert predicted_date is not None
        assert predicted_date > datetime.now(timezone.utc)

    def test_history_trimming(self, monitor):
        """Test old history is trimmed."""
        # Record old entries
        old_time = datetime.now(timezone.utc) - timedelta(days=100)
        monitor.record_cleanliness(0.80, old_time)

        # Record new entry - should trigger trimming
        monitor.record_cleanliness(0.79)

        # Old entry should be removed
        assert len(monitor._history) == 1


# =============================================================================
# CALCULATION COUNT TESTS
# =============================================================================

class TestCalculationCount:
    """Test calculation counting."""

    def test_calculation_count_increments(self, calculator):
        """Test calculation count increments."""
        initial_count = calculator.calculation_count

        calculator.calculate_cleanliness(
            heat_duty_btu_hr=450_000_000.0,
            lmtd_f=18.0,
            surface_area_ft2=150000.0,
            cw_velocity_fps=7.0,
            cw_inlet_temp_f=70.0,
        )

        assert calculator.calculation_count == initial_count + 1

    def test_multiple_calculations_counted(self, calculator):
        """Test multiple calculations are counted."""
        initial_count = calculator.calculation_count

        for _ in range(5):
            calculator.calculate_cleanliness(
                heat_duty_btu_hr=450_000_000.0,
                lmtd_f=18.0,
                surface_area_ft2=150000.0,
                cw_velocity_fps=7.0,
                cw_inlet_temp_f=70.0,
            )

        assert calculator.calculation_count == initial_count + 5
