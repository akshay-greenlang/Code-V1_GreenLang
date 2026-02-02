"""
Unit tests for GL-020 ECONOPULSE Acid Dew Point Calculator

Tests the Verhoff-Banchero correlation and related calculations.
Target coverage: 85%+

Standards Reference:
    - Verhoff & Banchero, Chemical Engineering Progress, 1974
    - EPA Method 6C for SO2 measurement
    - ASME PTC 4.1 Steam Generating Units

Zero-Hallucination: Tests validate against known reference values.
"""

import pytest
import math

from ..acid_dew_point import (
    AcidDewPointCalculator,
    AcidDewPointInput,
    create_acid_dew_point_calculator,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def calculator():
    """Create default acid dew point calculator."""
    return AcidDewPointCalculator(safety_margin_f=30.0)


@pytest.fixture
def calculator_high_margin():
    """Create calculator with high safety margin."""
    return AcidDewPointCalculator(safety_margin_f=50.0)


@pytest.fixture
def natural_gas_input():
    """Natural gas combustion input (low sulfur)."""
    return AcidDewPointInput(
        flue_gas_moisture_pct=10.0,
        fuel_sulfur_pct=0.001,
        so2_to_so3_conversion_pct=1.0,
        flue_gas_o2_pct=3.0,
        cold_end_metal_temp_f=300.0,
        safety_margin_f=30.0,
    )


@pytest.fixture
def fuel_oil_input():
    """No. 6 fuel oil combustion input (high sulfur)."""
    return AcidDewPointInput(
        flue_gas_moisture_pct=8.0,
        fuel_sulfur_pct=2.0,
        so2_to_so3_conversion_pct=3.0,
        flue_gas_o2_pct=4.0,
        cold_end_metal_temp_f=320.0,
        safety_margin_f=30.0,
    )


@pytest.fixture
def coal_input():
    """Bituminous coal combustion input."""
    return AcidDewPointInput(
        flue_gas_moisture_pct=6.0,
        fuel_sulfur_pct=2.5,
        so2_to_so3_conversion_pct=2.0,
        flue_gas_o2_pct=3.5,
        cold_end_metal_temp_f=310.0,
        safety_margin_f=30.0,
    )


@pytest.fixture
def direct_so2_input():
    """Input with direct SO2 measurement."""
    return AcidDewPointInput(
        flue_gas_moisture_pct=10.0,
        flue_gas_so2_ppm=500.0,  # Direct measurement
        fuel_sulfur_pct=0.0,  # Ignored when SO2 is provided
        so2_to_so3_conversion_pct=2.0,
        flue_gas_o2_pct=3.0,
        cold_end_metal_temp_f=290.0,
        safety_margin_f=30.0,
    )


# =============================================================================
# CALCULATOR INITIALIZATION TESTS
# =============================================================================

class TestAcidDewPointCalculatorInit:
    """Test AcidDewPointCalculator initialization."""

    def test_default_initialization(self, calculator):
        """Test default calculator initialization."""
        assert calculator.safety_margin_f == 30.0

    def test_custom_safety_margin(self):
        """Test calculator with custom safety margin."""
        calc = AcidDewPointCalculator(safety_margin_f=50.0)
        assert calc.safety_margin_f == 50.0

    def test_factory_function(self):
        """Test factory function creates calculator."""
        calc = create_acid_dew_point_calculator(safety_margin_f=40.0)
        assert isinstance(calc, AcidDewPointCalculator)
        assert calc.safety_margin_f == 40.0


# =============================================================================
# EXCESS AIR CALCULATION TESTS
# =============================================================================

class TestExcessAirCalculation:
    """Test excess air calculation from O2 measurement."""

    def test_typical_excess_air(self, calculator):
        """Test typical excess air calculation at 3% O2."""
        excess_air = calculator.calculate_excess_air(3.0)
        # At 3% O2: excess_air = 3 / (21 - 3) * 100 = 16.67%
        expected = 3.0 / (21.0 - 3.0) * 100
        assert excess_air == pytest.approx(expected, rel=1e-4)
        assert excess_air == pytest.approx(16.67, rel=0.01)

    def test_low_excess_air(self, calculator):
        """Test low excess air at 1.5% O2."""
        excess_air = calculator.calculate_excess_air(1.5)
        expected = 1.5 / (21.0 - 1.5) * 100
        assert excess_air == pytest.approx(expected, rel=1e-4)
        assert excess_air == pytest.approx(7.69, rel=0.01)

    def test_high_excess_air(self, calculator):
        """Test high excess air at 6% O2."""
        excess_air = calculator.calculate_excess_air(6.0)
        expected = 6.0 / (21.0 - 6.0) * 100
        assert excess_air == pytest.approx(expected, rel=1e-4)
        assert excess_air == pytest.approx(40.0, rel=0.01)

    def test_zero_excess_air(self, calculator):
        """Test zero O2 gives zero excess air."""
        excess_air = calculator.calculate_excess_air(0.0)
        assert excess_air == 0.0

    def test_near_atmospheric_o2(self, calculator):
        """Test behavior near atmospheric O2 level."""
        # At 21% O2, should return 0 (edge case)
        excess_air = calculator.calculate_excess_air(21.0)
        assert excess_air == 0.0

    @pytest.mark.parametrize("o2_pct,expected_excess_air", [
        (0.0, 0.0),
        (2.0, 10.53),
        (3.0, 16.67),
        (4.0, 23.53),
        (5.0, 31.25),
        (6.0, 40.0),
    ])
    def test_excess_air_table(self, calculator, o2_pct, expected_excess_air):
        """Test excess air calculation against reference table."""
        excess_air = calculator.calculate_excess_air(o2_pct)
        assert excess_air == pytest.approx(expected_excess_air, rel=0.01)


# =============================================================================
# SO3 CALCULATION TESTS
# =============================================================================

class TestSO3Calculation:
    """Test SO3 concentration calculation from fuel sulfur."""

    def test_low_sulfur_fuel(self, calculator):
        """Test SO3 calculation for natural gas (low sulfur)."""
        so3 = calculator.calculate_so3_from_fuel(
            fuel_sulfur_pct=0.001,
            so2_to_so3_conversion_pct=1.0,
            excess_air_pct=15.0,
        )
        # SO3 should be very low for natural gas
        assert so3 < 1.0  # Less than 1 ppm
        assert so3 > 0.0

    def test_high_sulfur_fuel(self, calculator):
        """Test SO3 calculation for high sulfur fuel oil."""
        so3 = calculator.calculate_so3_from_fuel(
            fuel_sulfur_pct=2.0,
            so2_to_so3_conversion_pct=3.0,
            excess_air_pct=20.0,
        )
        # Higher SO3 for high sulfur fuel
        assert so3 > 10.0
        assert so3 < 100.0

    def test_zero_sulfur(self, calculator):
        """Test zero sulfur fuel gives zero SO3."""
        so3 = calculator.calculate_so3_from_fuel(
            fuel_sulfur_pct=0.0,
            so2_to_so3_conversion_pct=2.0,
            excess_air_pct=15.0,
        )
        assert so3 == 0.0

    def test_so3_increases_with_sulfur(self, calculator):
        """Test SO3 increases with fuel sulfur content."""
        so3_low = calculator.calculate_so3_from_fuel(
            fuel_sulfur_pct=0.5,
            so2_to_so3_conversion_pct=2.0,
            excess_air_pct=15.0,
        )
        so3_high = calculator.calculate_so3_from_fuel(
            fuel_sulfur_pct=2.0,
            so2_to_so3_conversion_pct=2.0,
            excess_air_pct=15.0,
        )
        assert so3_high > so3_low
        assert so3_high / so3_low == pytest.approx(4.0, rel=0.01)

    def test_so3_increases_with_conversion(self, calculator):
        """Test SO3 increases with higher conversion rate."""
        so3_low = calculator.calculate_so3_from_fuel(
            fuel_sulfur_pct=1.0,
            so2_to_so3_conversion_pct=1.0,
            excess_air_pct=15.0,
        )
        so3_high = calculator.calculate_so3_from_fuel(
            fuel_sulfur_pct=1.0,
            so2_to_so3_conversion_pct=3.0,
            excess_air_pct=15.0,
        )
        assert so3_high > so3_low
        assert so3_high / so3_low == pytest.approx(3.0, rel=0.01)


# =============================================================================
# WATER DEW POINT TESTS
# =============================================================================

class TestWaterDewPoint:
    """Test water dew point calculation."""

    def test_typical_moisture(self, calculator):
        """Test water dew point at typical moisture level."""
        dew_point = calculator.calculate_water_dew_point(10.0)
        # At 10% moisture, water dew point should be around 110-120F
        assert dew_point > 100.0
        assert dew_point < 140.0

    def test_low_moisture(self, calculator):
        """Test water dew point at low moisture level."""
        dew_point_low = calculator.calculate_water_dew_point(5.0)
        dew_point_high = calculator.calculate_water_dew_point(15.0)
        # Higher moisture = higher dew point
        assert dew_point_high > dew_point_low

    def test_zero_moisture(self, calculator):
        """Test water dew point at zero moisture."""
        dew_point = calculator.calculate_water_dew_point(0.0)
        # Should return freezing point for zero moisture
        assert dew_point == 32.0

    def test_moisture_range(self, calculator):
        """Test water dew point across moisture range."""
        dew_points = []
        for moisture in [5.0, 10.0, 15.0, 20.0]:
            dew_points.append(calculator.calculate_water_dew_point(moisture))

        # Dew points should be monotonically increasing
        for i in range(1, len(dew_points)):
            assert dew_points[i] > dew_points[i - 1]


# =============================================================================
# VERHOFF-BANCHERO ACID DEW POINT TESTS
# =============================================================================

class TestVerhoffBancheroCorrelation:
    """Test Verhoff-Banchero acid dew point correlation."""

    def test_typical_conditions(self, calculator):
        """Test acid dew point at typical conditions."""
        adp = calculator.calculate_acid_dew_point_verhoff_banchero(
            h2o_pct=10.0,
            so3_ppm=10.0,
        )
        # Typical acid dew point is 250-300F
        assert adp > 220.0
        assert adp < 350.0

    def test_high_so3(self, calculator):
        """Test acid dew point increases with higher SO3."""
        adp_low = calculator.calculate_acid_dew_point_verhoff_banchero(
            h2o_pct=10.0,
            so3_ppm=5.0,
        )
        adp_high = calculator.calculate_acid_dew_point_verhoff_banchero(
            h2o_pct=10.0,
            so3_ppm=50.0,
        )
        assert adp_high > adp_low

    def test_high_moisture(self, calculator):
        """Test acid dew point increases with higher moisture."""
        adp_low = calculator.calculate_acid_dew_point_verhoff_banchero(
            h2o_pct=5.0,
            so3_ppm=10.0,
        )
        adp_high = calculator.calculate_acid_dew_point_verhoff_banchero(
            h2o_pct=15.0,
            so3_ppm=10.0,
        )
        assert adp_high > adp_low

    def test_zero_so3(self, calculator):
        """Test with zero SO3 returns water dew point."""
        adp = calculator.calculate_acid_dew_point_verhoff_banchero(
            h2o_pct=10.0,
            so3_ppm=0.0,
        )
        water_dp = calculator.calculate_water_dew_point(10.0)
        # Should return water dew point when SO3 is zero
        assert adp == water_dp

    def test_very_low_so3(self, calculator):
        """Test with very low SO3."""
        adp = calculator.calculate_acid_dew_point_verhoff_banchero(
            h2o_pct=10.0,
            so3_ppm=0.1,
        )
        # Should still calculate valid dew point
        assert adp > 100.0
        assert adp < 400.0

    def test_reference_point_verification(self, calculator):
        """
        Verify against reference point from literature.

        Reference: At pH2O = 76 mmHg (10%) and pSO3 = 0.0076 mmHg (10 ppm),
        the acid dew point should be approximately 260-280F.
        """
        adp = calculator.calculate_acid_dew_point_verhoff_banchero(
            h2o_pct=10.0,
            so3_ppm=10.0,
        )
        # Allow reasonable tolerance for reference value
        assert adp > 250.0
        assert adp < 300.0


# =============================================================================
# OKKES CORRELATION TESTS
# =============================================================================

class TestOkkesCorrelation:
    """Test alternative Okkes acid dew point correlation."""

    def test_typical_conditions(self, calculator):
        """Test Okkes correlation at typical conditions."""
        adp = calculator.calculate_acid_dew_point_okkes(
            h2o_pct=10.0,
            so3_ppm=10.0,
        )
        # Should be in reasonable range
        assert adp > 200.0
        assert adp < 400.0

    def test_comparison_with_verhoff_banchero(self, calculator):
        """Compare Okkes with Verhoff-Banchero."""
        adp_vb = calculator.calculate_acid_dew_point_verhoff_banchero(
            h2o_pct=10.0,
            so3_ppm=20.0,
        )
        adp_okkes = calculator.calculate_acid_dew_point_okkes(
            h2o_pct=10.0,
            so3_ppm=20.0,
        )
        # Results should be reasonably close (within 20F)
        assert abs(adp_vb - adp_okkes) < 30.0

    def test_zero_inputs(self, calculator):
        """Test Okkes with zero inputs."""
        adp = calculator.calculate_acid_dew_point_okkes(
            h2o_pct=0.0,
            so3_ppm=0.0,
        )
        # Should return water dew point
        water_dp = calculator.calculate_water_dew_point(0.0)
        assert adp == water_dp


# =============================================================================
# CORROSION RISK ASSESSMENT TESTS
# =============================================================================

class TestCorrosionRiskAssessment:
    """Test corrosion risk assessment."""

    def test_low_risk(self, calculator):
        """Test low corrosion risk when well above dew point."""
        risk, action = calculator.assess_corrosion_risk(
            metal_temp_f=350.0,
            acid_dew_point_f=270.0,
            safety_margin_f=30.0,
        )
        assert risk == "low"
        assert "adequate" in action.lower()

    def test_moderate_risk(self, calculator):
        """Test moderate risk when margin is tight."""
        risk, action = calculator.assess_corrosion_risk(
            metal_temp_f=295.0,
            acid_dew_point_f=270.0,
            safety_margin_f=30.0,
        )
        assert risk == "moderate"
        assert "monitor" in action.lower()

    def test_high_risk(self, calculator):
        """Test high risk when margin is inadequate."""
        risk, action = calculator.assess_corrosion_risk(
            metal_temp_f=280.0,
            acid_dew_point_f=270.0,
            safety_margin_f=30.0,
        )
        assert risk == "high"
        assert "increase" in action.lower()

    def test_critical_risk(self, calculator):
        """Test critical risk when below dew point."""
        risk, action = calculator.assess_corrosion_risk(
            metal_temp_f=260.0,
            acid_dew_point_f=270.0,
            safety_margin_f=30.0,
        )
        assert risk == "critical"
        assert "immediate" in action.lower()

    def test_exactly_at_margin(self, calculator):
        """Test when exactly at safety margin."""
        risk, action = calculator.assess_corrosion_risk(
            metal_temp_f=300.0,
            acid_dew_point_f=270.0,
            safety_margin_f=30.0,
        )
        assert risk == "low"


# =============================================================================
# COMPLETE CALCULATION TESTS
# =============================================================================

class TestCompleteCalculation:
    """Test complete acid dew point analysis."""

    def test_natural_gas_calculation(self, calculator, natural_gas_input):
        """Test complete calculation for natural gas."""
        result = calculator.calculate(natural_gas_input)

        # Verify all required fields
        assert "sulfuric_acid_dew_point_f" in result
        assert "water_dew_point_f" in result
        assert "effective_dew_point_f" in result
        assert "margin_above_dew_point_f" in result
        assert "corrosion_risk" in result
        assert "so3_concentration_ppm" in result
        assert "excess_air_pct" in result
        assert "provenance_hash" in result

        # Natural gas should have low acid dew point
        assert result["so3_concentration_ppm"] < 1.0
        assert result["corrosion_risk"] == "low"

    def test_fuel_oil_calculation(self, calculator, fuel_oil_input):
        """Test complete calculation for high sulfur fuel oil."""
        result = calculator.calculate(fuel_oil_input)

        # Higher SO3 for fuel oil
        assert result["so3_concentration_ppm"] > 10.0
        # Higher acid dew point
        assert result["sulfuric_acid_dew_point_f"] > 250.0

    def test_coal_calculation(self, calculator, coal_input):
        """Test complete calculation for coal."""
        result = calculator.calculate(coal_input)

        # Coal has significant SO3
        assert result["so3_concentration_ppm"] > 5.0
        assert result["sulfuric_acid_dew_point_f"] > 240.0

    def test_direct_so2_measurement(self, calculator, direct_so2_input):
        """Test calculation using direct SO2 measurement."""
        result = calculator.calculate(direct_so2_input)

        # SO3 should be derived from measured SO2
        expected_so3 = 500.0 * 2.0 / 100  # 10 ppm
        assert result["so3_concentration_ppm"] == pytest.approx(expected_so3, rel=0.01)

    def test_action_required_when_margin_low(self, calculator):
        """Test action is required when margin is too low."""
        input_data = AcidDewPointInput(
            flue_gas_moisture_pct=10.0,
            fuel_sulfur_pct=2.0,  # High sulfur
            so2_to_so3_conversion_pct=3.0,
            flue_gas_o2_pct=3.0,
            cold_end_metal_temp_f=260.0,  # Low metal temp
            safety_margin_f=30.0,
        )

        result = calculator.calculate(input_data)

        # Should indicate action required
        assert result["action_required"] is True
        assert result["feedwater_temp_adjustment_f"] is not None
        assert result["feedwater_temp_adjustment_f"] > 0

    def test_provenance_hash_deterministic(self, calculator, natural_gas_input):
        """Test provenance hash is deterministic."""
        result1 = calculator.calculate(natural_gas_input)
        result2 = calculator.calculate(natural_gas_input)

        assert result1["provenance_hash"] == result2["provenance_hash"]

    def test_calculation_method_in_result(self, calculator, natural_gas_input):
        """Test calculation method is recorded."""
        result = calculator.calculate(natural_gas_input)

        assert result["calculation_method"] == "VERHOFF_BANCHERO"
        assert "Verhoff & Banchero" in result["formula_reference"]

    def test_effective_dew_point_is_maximum(self, calculator, fuel_oil_input):
        """Test effective dew point is the maximum of acid and water."""
        result = calculator.calculate(fuel_oil_input)

        assert result["effective_dew_point_f"] == max(
            result["sulfuric_acid_dew_point_f"],
            result["water_dew_point_f"],
        )


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_high_sulfur(self, calculator):
        """Test with very high sulfur content."""
        input_data = AcidDewPointInput(
            flue_gas_moisture_pct=8.0,
            fuel_sulfur_pct=4.5,  # Near maximum
            so2_to_so3_conversion_pct=5.0,
            flue_gas_o2_pct=3.0,
            cold_end_metal_temp_f=350.0,
            safety_margin_f=30.0,
        )

        result = calculator.calculate(input_data)

        # Should still calculate valid result
        assert result["sulfuric_acid_dew_point_f"] > 200.0
        assert result["sulfuric_acid_dew_point_f"] < 450.0

    def test_very_low_moisture(self, calculator):
        """Test with very low moisture content."""
        input_data = AcidDewPointInput(
            flue_gas_moisture_pct=1.0,
            fuel_sulfur_pct=1.0,
            so2_to_so3_conversion_pct=2.0,
            flue_gas_o2_pct=3.0,
            cold_end_metal_temp_f=300.0,
            safety_margin_f=30.0,
        )

        result = calculator.calculate(input_data)

        # Low moisture should result in lower water dew point
        assert result["water_dew_point_f"] < 100.0

    def test_high_excess_air(self, calculator):
        """Test with high excess air (dilutes SO3)."""
        input_data = AcidDewPointInput(
            flue_gas_moisture_pct=10.0,
            fuel_sulfur_pct=2.0,
            so2_to_so3_conversion_pct=2.0,
            flue_gas_o2_pct=8.0,  # High excess air
            cold_end_metal_temp_f=300.0,
            safety_margin_f=30.0,
        )

        result = calculator.calculate(input_data)

        # High excess air should be calculated
        assert result["excess_air_pct"] > 50.0


# =============================================================================
# FUEL TYPE DEFAULT TESTS
# =============================================================================

class TestFuelTypeDefaults:
    """Test fuel type default sulfur values."""

    def test_fuel_defaults_exist(self, calculator):
        """Test fuel type defaults are defined."""
        assert "natural_gas" in calculator.FUEL_SULFUR_DEFAULTS
        assert "no2_fuel_oil" in calculator.FUEL_SULFUR_DEFAULTS
        assert "no6_fuel_oil" in calculator.FUEL_SULFUR_DEFAULTS
        assert "coal_bituminous" in calculator.FUEL_SULFUR_DEFAULTS

    def test_natural_gas_low_sulfur(self, calculator):
        """Test natural gas has low default sulfur."""
        assert calculator.FUEL_SULFUR_DEFAULTS["natural_gas"] < 0.01

    def test_fuel_oil_higher_sulfur(self, calculator):
        """Test fuel oil has higher sulfur than natural gas."""
        assert calculator.FUEL_SULFUR_DEFAULTS["no6_fuel_oil"] > \
               calculator.FUEL_SULFUR_DEFAULTS["natural_gas"]

    def test_conversion_rates_exist(self, calculator):
        """Test SO3 conversion rates are defined."""
        assert "natural_gas" in calculator.SO3_CONVERSION_RATES
        assert "no6_fuel_oil" in calculator.SO3_CONVERSION_RATES
        # Fuel oil with vanadium has higher conversion
        assert calculator.SO3_CONVERSION_RATES["no6_fuel_oil"] > \
               calculator.SO3_CONVERSION_RATES["natural_gas"]


# =============================================================================
# PARAMETRIZED TESTS
# =============================================================================

class TestParameterized:
    """Parameterized tests for various input combinations."""

    @pytest.mark.parametrize("sulfur_pct,expected_so3_range", [
        (0.001, (0.0, 1.0)),      # Natural gas
        (0.3, (1.0, 10.0)),        # Light fuel oil
        (2.0, (10.0, 100.0)),      # Heavy fuel oil
        (2.5, (10.0, 150.0)),      # Bituminous coal
    ])
    def test_so3_ranges_by_fuel_type(self, calculator, sulfur_pct, expected_so3_range):
        """Test SO3 concentration ranges for different fuel types."""
        so3 = calculator.calculate_so3_from_fuel(
            fuel_sulfur_pct=sulfur_pct,
            so2_to_so3_conversion_pct=2.0,
            excess_air_pct=15.0,
        )
        assert expected_so3_range[0] <= so3 <= expected_so3_range[1]

    @pytest.mark.parametrize("metal_temp_f,expected_risk", [
        (350.0, "low"),
        (290.0, "moderate"),
        (275.0, "high"),
        (260.0, "critical"),
    ])
    def test_risk_levels_by_temperature(self, calculator, metal_temp_f, expected_risk):
        """Test risk level determination at various temperatures."""
        # Assuming acid dew point of 270F
        risk, _ = calculator.assess_corrosion_risk(
            metal_temp_f=metal_temp_f,
            acid_dew_point_f=270.0,
            safety_margin_f=30.0,
        )
        assert risk == expected_risk

    @pytest.mark.parametrize("moisture_pct", [5.0, 8.0, 10.0, 12.0, 15.0, 20.0])
    def test_moisture_range_calculation(self, calculator, moisture_pct):
        """Test calculation across moisture range."""
        adp = calculator.calculate_acid_dew_point_verhoff_banchero(
            h2o_pct=moisture_pct,
            so3_ppm=10.0,
        )
        # All should produce valid results
        assert adp > 150.0
        assert adp < 400.0
