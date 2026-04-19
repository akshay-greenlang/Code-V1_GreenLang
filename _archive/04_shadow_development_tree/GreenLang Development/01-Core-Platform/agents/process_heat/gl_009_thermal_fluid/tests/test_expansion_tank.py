"""
Unit tests for GL-009 THERMALIQ Agent Expansion Tank Sizing

Tests expansion tank sizing validation per API 660, thermal expansion
calculations, NPSH analysis, and nitrogen blanket requirements.
"""

import pytest
import math
from typing import Dict, Any

from greenlang.agents.process_heat.gl_009_thermal_fluid.expansion_tank import (
    ExpansionTankAnalyzer,
    NPSHAnalysis,
    GRAVITY_FT_S2,
    GALLONS_PER_FT3,
    FT_HEAD_PER_PSI,
    EXPANSION_SAFETY_FACTOR,
    MIN_TANK_LEVEL_PCT,
    MAX_TANK_LEVEL_PCT,
    NPSH_SAFETY_MARGIN_FT,
    calculate_expansion,
    size_tank,
)
from greenlang.agents.process_heat.gl_009_thermal_fluid.config import (
    ExpansionTankConfig,
    PumpConfig,
)
from greenlang.agents.process_heat.gl_009_thermal_fluid.schemas import (
    ThermalFluidType,
    ExpansionTankSizing,
    ExpansionTankData,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def expansion_analyzer():
    """Create expansion tank analyzer instance."""
    return ExpansionTankAnalyzer(fluid_type=ThermalFluidType.THERMINOL_66)


@pytest.fixture
def expansion_analyzer_with_configs():
    """Create analyzer with custom configs."""
    pump_config = PumpConfig(
        npsh_required_ft=12.0,
        design_flow_gpm=600.0,
    )
    tank_config = ExpansionTankConfig(
        volume_gallons=1500.0,
        blanket_pressure_psig=3.0,
    )
    return ExpansionTankAnalyzer(
        fluid_type=ThermalFluidType.THERMINOL_66,
        pump_config=pump_config,
        tank_config=tank_config,
    )


@pytest.fixture
def typical_analysis_params():
    """Create typical analysis parameters."""
    return {
        "tank_volume_gallons": 1000.0,
        "system_volume_gallons": 5000.0,
        "cold_temp_f": 70.0,
        "hot_temp_f": 600.0,
    }


@pytest.fixture
def expansion_tank_data():
    """Create expansion tank operating data."""
    return ExpansionTankData(
        tank_id="ET-001",
        total_volume_gallons=1000.0,
        current_level_pct=45.0,
        current_temperature_f=400.0,
        system_volume_gallons=5000.0,
        max_operating_temp_f=600.0,
        cold_fill_temp_f=70.0,
    )


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestExpansionTankAnalyzerInit:
    """Tests for ExpansionTankAnalyzer initialization."""

    def test_default_initialization(self, expansion_analyzer):
        """Test analyzer initializes with defaults."""
        assert expansion_analyzer.fluid_type == ThermalFluidType.THERMINOL_66
        assert expansion_analyzer._calculation_count == 0
        assert expansion_analyzer.pump_config is not None
        assert expansion_analyzer.tank_config is not None

    def test_custom_configs(self, expansion_analyzer_with_configs):
        """Test analyzer with custom configs."""
        assert expansion_analyzer_with_configs.pump_config.npsh_required_ft == 12.0
        assert expansion_analyzer_with_configs.tank_config.volume_gallons == 1500.0


# =============================================================================
# THERMAL EXPANSION TESTS
# =============================================================================

class TestThermalExpansion:
    """Tests for thermal expansion calculations."""

    def test_expansion_calculation(self, expansion_analyzer):
        """Test basic expansion volume calculation."""
        result = expansion_analyzer.calculate_expansion_volume(
            system_volume_gallons=5000.0,
            cold_temp_f=70.0,
            hot_temp_f=600.0,
        )

        assert "expansion_volume_gallons" in result
        assert "expansion_percentage" in result
        assert result["expansion_volume_gallons"] > 0
        # Typical expansion 15-25% for Therminol 66 from 70F to 600F
        assert 15.0 <= result["expansion_percentage"] <= 30.0

    def test_expansion_increases_with_temperature_range(self, expansion_analyzer):
        """Test expansion increases with larger temperature range."""
        result_small = expansion_analyzer.calculate_expansion_volume(
            system_volume_gallons=5000.0,
            cold_temp_f=70.0,
            hot_temp_f=300.0,
        )

        result_large = expansion_analyzer.calculate_expansion_volume(
            system_volume_gallons=5000.0,
            cold_temp_f=70.0,
            hot_temp_f=600.0,
        )

        assert result_large["expansion_volume_gallons"] > result_small["expansion_volume_gallons"]

    def test_expansion_proportional_to_system_volume(self, expansion_analyzer):
        """Test expansion is proportional to system volume."""
        result_small = expansion_analyzer.calculate_expansion_volume(
            system_volume_gallons=2500.0,
            cold_temp_f=70.0,
            hot_temp_f=600.0,
        )

        result_large = expansion_analyzer.calculate_expansion_volume(
            system_volume_gallons=5000.0,
            cold_temp_f=70.0,
            hot_temp_f=600.0,
        )

        # Should be approximately 2x
        ratio = result_large["expansion_volume_gallons"] / result_small["expansion_volume_gallons"]
        assert 1.9 <= ratio <= 2.1

    def test_expansion_zero_at_same_temp(self, expansion_analyzer):
        """Test expansion is zero at same temperature."""
        result = expansion_analyzer.calculate_expansion_volume(
            system_volume_gallons=5000.0,
            cold_temp_f=500.0,
            hot_temp_f=500.0,
        )

        assert abs(result["expansion_volume_gallons"]) < 1.0
        assert abs(result["expansion_percentage"]) < 0.1

    def test_hot_system_volume_correct(self, expansion_analyzer):
        """Test hot system volume is calculated correctly."""
        result = expansion_analyzer.calculate_expansion_volume(
            system_volume_gallons=5000.0,
            cold_temp_f=70.0,
            hot_temp_f=600.0,
        )

        expected_hot = result["cold_system_volume_gallons"] + result["expansion_volume_gallons"]
        assert abs(result["hot_system_volume_gallons"] - expected_hot) < 1.0


# =============================================================================
# TANK SIZING TESTS
# =============================================================================

class TestTankSizing:
    """Tests for expansion tank sizing."""

    def test_size_expansion_tank(self, expansion_analyzer):
        """Test tank sizing calculation."""
        result = expansion_analyzer.size_expansion_tank(
            system_volume_gallons=5000.0,
            cold_temp_f=70.0,
            hot_temp_f=600.0,
        )

        assert "required_volume_gallons" in result
        assert "recommended_size_gallons" in result
        assert "expected_cold_level_pct" in result
        assert "expected_hot_level_pct" in result
        assert "design_standard" in result

    def test_recommended_size_exceeds_required(self, expansion_analyzer):
        """Test recommended size exceeds required volume."""
        result = expansion_analyzer.size_expansion_tank(
            system_volume_gallons=5000.0,
            cold_temp_f=70.0,
            hot_temp_f=600.0,
        )

        assert result["recommended_size_gallons"] >= result["required_volume_gallons"]

    def test_recommended_is_standard_size(self, expansion_analyzer):
        """Test recommended size is a standard tank size."""
        standard_sizes = [100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]

        result = expansion_analyzer.size_expansion_tank(
            system_volume_gallons=5000.0,
            cold_temp_f=70.0,
            hot_temp_f=600.0,
        )

        assert result["recommended_size_gallons"] in standard_sizes

    def test_safety_factor_applied(self, expansion_analyzer):
        """Test safety factor is applied."""
        result = expansion_analyzer.size_expansion_tank(
            system_volume_gallons=5000.0,
            cold_temp_f=70.0,
            hot_temp_f=600.0,
            safety_factor=1.5,
        )

        assert result["safety_factor_applied"] == 1.5

    def test_level_predictions_reasonable(self, expansion_analyzer):
        """Test level predictions are reasonable."""
        result = expansion_analyzer.size_expansion_tank(
            system_volume_gallons=5000.0,
            cold_temp_f=70.0,
            hot_temp_f=600.0,
        )

        # Cold level should be low
        assert 10 <= result["expected_cold_level_pct"] <= 50

        # Hot level should be higher but not overflow
        assert result["expected_hot_level_pct"] > result["expected_cold_level_pct"]
        assert result["expected_hot_level_pct"] <= 90

    def test_design_standard_api_660(self, expansion_analyzer):
        """Test design standard is API 660."""
        result = expansion_analyzer.size_expansion_tank(
            system_volume_gallons=5000.0,
            cold_temp_f=70.0,
            hot_temp_f=600.0,
        )

        assert result["design_standard"] == "API_660"


# =============================================================================
# FULL ANALYSIS TESTS
# =============================================================================

class TestFullAnalysis:
    """Tests for full expansion tank analysis."""

    def test_analyze_returns_sizing(
        self, expansion_analyzer, typical_analysis_params
    ):
        """Test analyze returns ExpansionTankSizing."""
        result = expansion_analyzer.analyze(**typical_analysis_params)

        assert isinstance(result, ExpansionTankSizing)

    def test_analyze_sizing_adequacy(self, expansion_analyzer):
        """Test analyze determines sizing adequacy."""
        # Adequate tank
        result_adequate = expansion_analyzer.analyze(
            tank_volume_gallons=1500.0,
            system_volume_gallons=5000.0,
            cold_temp_f=70.0,
            hot_temp_f=600.0,
        )
        assert result_adequate.sizing_adequate == True

        # Undersized tank
        result_undersized = expansion_analyzer.analyze(
            tank_volume_gallons=500.0,
            system_volume_gallons=5000.0,
            cold_temp_f=70.0,
            hot_temp_f=600.0,
        )
        assert result_undersized.sizing_adequate == False

    def test_analyze_returns_expansion_percentage(
        self, expansion_analyzer, typical_analysis_params
    ):
        """Test analyze returns expansion percentage."""
        result = expansion_analyzer.analyze(**typical_analysis_params)

        assert 10.0 <= result.thermal_expansion_pct <= 30.0

    def test_analyze_returns_levels(
        self, expansion_analyzer, typical_analysis_params
    ):
        """Test analyze returns cold/hot levels."""
        result = expansion_analyzer.analyze(**typical_analysis_params)

        assert 0 <= result.cold_level_pct <= 100
        assert 0 <= result.hot_level_pct <= 100
        assert result.hot_level_pct > result.cold_level_pct

    def test_analyze_with_current_level(self, expansion_analyzer):
        """Test analyze with current level deviation."""
        result = expansion_analyzer.analyze(
            tank_volume_gallons=1000.0,
            system_volume_gallons=5000.0,
            cold_temp_f=70.0,
            hot_temp_f=600.0,
            current_level_pct=75.0,
        )

        assert result.current_level_deviation_pct is not None

    def test_analyze_returns_npsh(
        self, expansion_analyzer, typical_analysis_params
    ):
        """Test analyze returns NPSH values."""
        result = expansion_analyzer.analyze(**typical_analysis_params)

        assert result.required_npsh_ft >= 0
        assert result.available_npsh_ft > 0
        assert result.npsh_margin_ft is not None

    def test_analyze_returns_recommendations(self, expansion_analyzer):
        """Test analyze returns recommendations for issues."""
        # Undersized tank should have recommendations
        result = expansion_analyzer.analyze(
            tank_volume_gallons=500.0,
            system_volume_gallons=5000.0,
            cold_temp_f=70.0,
            hot_temp_f=600.0,
        )

        assert len(result.recommendations) > 0

    def test_analyze_calculation_standard(
        self, expansion_analyzer, typical_analysis_params
    ):
        """Test calculation standard is recorded."""
        result = expansion_analyzer.analyze(**typical_analysis_params)

        assert result.calculation_standard == "API_660"


# =============================================================================
# NPSH ANALYSIS TESTS
# =============================================================================

class TestNPSHAnalysis:
    """Tests for NPSH (Net Positive Suction Head) analysis."""

    def test_npsh_analysis_positive(self, expansion_analyzer):
        """Test NPSH analysis returns positive value for normal conditions."""
        result = expansion_analyzer.analyze(
            tank_volume_gallons=1000.0,
            system_volume_gallons=5000.0,
            cold_temp_f=70.0,
            hot_temp_f=600.0,
            tank_elevation_ft=10.0,
            pump_centerline_ft=0.0,
        )

        assert result.available_npsh_ft > 0

    def test_npsh_margin_calculated(self, expansion_analyzer):
        """Test NPSH margin is calculated correctly."""
        result = expansion_analyzer.analyze(
            tank_volume_gallons=1000.0,
            system_volume_gallons=5000.0,
            cold_temp_f=70.0,
            hot_temp_f=600.0,
            tank_elevation_ft=20.0,
        )

        expected_margin = result.available_npsh_ft - result.required_npsh_ft
        assert abs(result.npsh_margin_ft - expected_margin) < 0.5

    def test_npsh_increases_with_elevation(self, expansion_analyzer):
        """Test NPSH increases with tank elevation."""
        result_low = expansion_analyzer.analyze(
            tank_volume_gallons=1000.0,
            system_volume_gallons=5000.0,
            cold_temp_f=70.0,
            hot_temp_f=600.0,
            tank_elevation_ft=5.0,
        )

        result_high = expansion_analyzer.analyze(
            tank_volume_gallons=1000.0,
            system_volume_gallons=5000.0,
            cold_temp_f=70.0,
            hot_temp_f=600.0,
            tank_elevation_ft=30.0,
        )

        assert result_high.available_npsh_ft > result_low.available_npsh_ft


# =============================================================================
# OPERATING LEVEL VALIDATION TESTS
# =============================================================================

class TestOperatingLevelValidation:
    """Tests for operating level validation."""

    def test_validate_normal_level(self, expansion_analyzer, expansion_tank_data):
        """Test validation of normal operating level."""
        result = expansion_analyzer.validate_operating_level(
            tank_data=expansion_tank_data,
            current_temp_f=400.0,
        )

        assert result["status"] in ["normal", "warning", "alarm"]

    def test_validate_low_level_alarm(self, expansion_analyzer, expansion_tank_data):
        """Test low level triggers alarm."""
        expansion_tank_data.current_level_pct = 5.0  # Below MIN_TANK_LEVEL_PCT

        result = expansion_analyzer.validate_operating_level(
            tank_data=expansion_tank_data,
            current_temp_f=400.0,
        )

        assert result["status"] == "alarm"
        assert len(result["warnings"]) > 0

    def test_validate_high_level_alarm(self, expansion_analyzer, expansion_tank_data):
        """Test high level triggers alarm."""
        expansion_tank_data.current_level_pct = 95.0  # Above MAX_TANK_LEVEL_PCT

        result = expansion_analyzer.validate_operating_level(
            tank_data=expansion_tank_data,
            current_temp_f=400.0,
        )

        assert result["status"] == "alarm"
        assert len(result["warnings"]) > 0

    def test_validate_level_deviation_warning(
        self, expansion_analyzer, expansion_tank_data
    ):
        """Test significant deviation triggers warning."""
        # Set level very different from expected
        expansion_tank_data.current_level_pct = 80.0  # Much higher than expected

        result = expansion_analyzer.validate_operating_level(
            tank_data=expansion_tank_data,
            current_temp_f=200.0,  # Low temp = expected low level
        )

        assert result["status"] in ["warning", "alarm"]


# =============================================================================
# NITROGEN REQUIREMENTS TESTS
# =============================================================================

class TestNitrogenRequirements:
    """Tests for nitrogen blanket calculations."""

    def test_nitrogen_requirements_calculated(self, expansion_analyzer):
        """Test nitrogen requirements are calculated."""
        result = expansion_analyzer.calculate_nitrogen_requirements(
            tank_volume_gallons=1000.0,
            tank_pressure_psig=2.0,
            operating_temp_f=600.0,
        )

        assert "vapor_space_gallons" in result
        assert "operating_volume_scf" in result
        assert "initial_purge_scf" in result
        assert "blanket_pressure_psig" in result
        assert "recommendations" in result

    def test_nitrogen_volume_proportional_to_tank(self, expansion_analyzer):
        """Test nitrogen volume is proportional to tank size."""
        result_small = expansion_analyzer.calculate_nitrogen_requirements(
            tank_volume_gallons=500.0,
            tank_pressure_psig=2.0,
            operating_temp_f=600.0,
        )

        result_large = expansion_analyzer.calculate_nitrogen_requirements(
            tank_volume_gallons=1000.0,
            tank_pressure_psig=2.0,
            operating_temp_f=600.0,
        )

        ratio = result_large["vapor_space_gallons"] / result_small["vapor_space_gallons"]
        assert 1.9 <= ratio <= 2.1

    def test_purge_volume_exceeds_operating(self, expansion_analyzer):
        """Test initial purge volume exceeds operating volume."""
        result = expansion_analyzer.calculate_nitrogen_requirements(
            tank_volume_gallons=1000.0,
        )

        assert result["initial_purge_scf"] > result["operating_volume_scf"]

    def test_nitrogen_recommendations(self, expansion_analyzer):
        """Test nitrogen recommendations are provided."""
        result = expansion_analyzer.calculate_nitrogen_requirements(
            tank_volume_gallons=1000.0,
        )

        assert len(result["recommendations"]) > 0


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_calculate_expansion_function(self):
        """Test calculate_expansion convenience function."""
        expansion = calculate_expansion(
            fluid_type=ThermalFluidType.THERMINOL_66,
            system_volume_gallons=5000.0,
            cold_temp_f=70.0,
            hot_temp_f=600.0,
        )

        assert expansion > 0
        # Typical 15-25% expansion
        expected_min = 5000 * 0.12
        expected_max = 5000 * 0.30
        assert expected_min <= expansion <= expected_max

    def test_size_tank_function(self):
        """Test size_tank convenience function."""
        recommended_size = size_tank(
            fluid_type=ThermalFluidType.THERMINOL_66,
            system_volume_gallons=5000.0,
            cold_temp_f=70.0,
            hot_temp_f=600.0,
        )

        assert recommended_size > 0
        # Should be standard size
        standard_sizes = [100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]
        assert recommended_size in standard_sizes


# =============================================================================
# TANK HEIGHT ESTIMATION TESTS
# =============================================================================

class TestTankHeightEstimation:
    """Tests for tank height estimation."""

    def test_estimate_tank_height(self, expansion_analyzer):
        """Test tank height estimation from volume."""
        height = expansion_analyzer._estimate_tank_height(1000.0)

        # For 2:1 L/D ratio, typical tank ~8-12 ft height for 1000 gal
        assert 5.0 <= height <= 15.0

    def test_height_increases_with_volume(self, expansion_analyzer):
        """Test height increases with volume."""
        height_small = expansion_analyzer._estimate_tank_height(500.0)
        height_large = expansion_analyzer._estimate_tank_height(2000.0)

        assert height_large > height_small


# =============================================================================
# CALCULATION COUNT TESTS
# =============================================================================

class TestCalculationCount:
    """Tests for calculation counting."""

    def test_calculation_count_increments(
        self, expansion_analyzer, typical_analysis_params
    ):
        """Test calculation count increments."""
        assert expansion_analyzer.calculation_count == 0

        expansion_analyzer.analyze(**typical_analysis_params)
        assert expansion_analyzer.calculation_count == 1

        expansion_analyzer.calculate_expansion_volume(
            system_volume_gallons=5000.0,
            cold_temp_f=70.0,
            hot_temp_f=600.0,
        )
        assert expansion_analyzer.calculation_count == 2


# =============================================================================
# CONSTANTS TESTS
# =============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_gravity_constant(self):
        """Test gravity constant value."""
        assert abs(GRAVITY_FT_S2 - 32.174) < 0.001

    def test_gallons_per_ft3(self):
        """Test gallons per cubic foot conversion."""
        assert abs(GALLONS_PER_FT3 - 7.48052) < 0.001

    def test_ft_head_per_psi(self):
        """Test feet of head per psi (for water)."""
        assert abs(FT_HEAD_PER_PSI - 2.31) < 0.01

    def test_expansion_safety_factor(self):
        """Test expansion safety factor."""
        assert EXPANSION_SAFETY_FACTOR == 1.1

    def test_level_limits(self):
        """Test level limit constants."""
        assert MIN_TANK_LEVEL_PCT == 10.0
        assert MAX_TANK_LEVEL_PCT == 90.0

    def test_npsh_safety_margin(self):
        """Test NPSH safety margin constant."""
        assert NPSH_SAFETY_MARGIN_FT == 3.0


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests for calculation determinism."""

    def test_same_input_same_output(self, typical_analysis_params):
        """Test same input produces identical output."""
        analyzer1 = ExpansionTankAnalyzer(fluid_type=ThermalFluidType.THERMINOL_66)
        analyzer2 = ExpansionTankAnalyzer(fluid_type=ThermalFluidType.THERMINOL_66)

        result1 = analyzer1.analyze(**typical_analysis_params)
        result2 = analyzer2.analyze(**typical_analysis_params)

        assert result1.required_volume_gallons == result2.required_volume_gallons
        assert result1.thermal_expansion_pct == result2.thermal_expansion_pct
        assert result1.cold_level_pct == result2.cold_level_pct
        assert result1.hot_level_pct == result2.hot_level_pct
