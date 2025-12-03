"""
GL-018 FLUEFLOW - Combustion Analyzer Unit Tests

Comprehensive unit tests for CombustionAnalyzer with 95%+ coverage target.
Tests all methods, edge cases, error handling, and provenance tracking.

Target Coverage: 95%+
Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import sys
import os
from decimal import Decimal
from typing import Dict

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from calculators.combustion_analyzer import (
    CombustionAnalyzer,
    CombustionInput,
    CombustionOutput,
    FuelType,
    GasBasis,
    FUEL_PROPERTIES,
    REFERENCE_O2_PERCENT,
    TYPICAL_H2O_PERCENT_WET,
    calculate_excess_air_from_O2,
    convert_wet_to_dry,
    convert_dry_to_wet,
)
from calculators.provenance import verify_provenance


# =============================================================================
# KNOWN TEST VALUES (ASME PTC 4.1 Reference)
# =============================================================================

KNOWN_VALUES = {
    "excess_air_O2_3.5": 20.0,
    "excess_air_O2_4.0": 23.5,
    "excess_air_O2_5.0": 31.25,
    "lambda_20pct_excess": 1.2,
    "lambda_25pct_excess": 1.25,
}


# =============================================================================
# TEST CLASS: CombustionAnalyzer
# =============================================================================

@pytest.mark.unit
@pytest.mark.calculator
@pytest.mark.critical
class TestCombustionAnalyzer:
    """Comprehensive test suite for CombustionAnalyzer."""

    # =========================================================================
    # INITIALIZATION TESTS
    # =========================================================================

    def test_initialization(self):
        """Test CombustionAnalyzer initializes correctly."""
        analyzer = CombustionAnalyzer()

        assert analyzer.VERSION == "1.0.0"
        assert analyzer.NAME == "CombustionAnalyzer"
        assert analyzer._tracker is None  # Not initialized until calculate()

    # =========================================================================
    # HAPPY PATH TESTS - NATURAL GAS
    # =========================================================================

    def test_natural_gas_optimal_combustion(self, combustion_analyzer, natural_gas_combustion_input):
        """Test natural gas combustion with optimal conditions."""
        result, provenance = combustion_analyzer.calculate(natural_gas_combustion_input)

        # Validate outputs
        assert isinstance(result, CombustionOutput)
        assert result.excess_air_pct == pytest.approx(20.0, rel=0.01)
        assert result.excess_O2_pct == pytest.approx(3.5, rel=0.01)
        assert result.stoichiometric_ratio == pytest.approx(1.2, rel=0.001)
        assert result.CO2_max_pct == pytest.approx(11.8, rel=0.01)
        assert result.O2_dry_pct == 3.5
        assert result.CO2_dry_pct == 12.0
        assert result.is_complete_combustion is True
        assert result.combustion_quality_rating in ["Excellent", "Good"]
        assert result.combustion_quality_index > 70.0

        # Validate provenance
        assert provenance.calculator_name == "CombustionAnalyzer"
        assert provenance.calculator_version == "1.0.0"
        assert len(provenance.calculation_steps) >= 9
        assert verify_provenance(provenance) is True
        assert len(provenance.provenance_hash) == 64

    def test_natural_gas_high_efficiency(self, combustion_analyzer):
        """Test natural gas with high efficiency (low CO, optimal O2)."""
        inputs = CombustionInput(
            O2_pct=3.0,
            CO2_pct=12.5,
            CO_ppm=30.0,
            NOx_ppm=120.0,
            flue_gas_temp_c=170.0,
            ambient_temp_c=20.0,
            fuel_type=FuelType.NATURAL_GAS.value,
            gas_basis=GasBasis.DRY.value
        )

        result, provenance = combustion_analyzer.calculate(inputs)

        assert result.excess_air_pct == pytest.approx(16.7, rel=0.02)
        assert result.is_complete_combustion is True
        assert result.combustion_quality_rating == "Excellent"
        assert result.combustion_quality_index >= 85.0

    # =========================================================================
    # HAPPY PATH TESTS - FUEL OIL
    # =========================================================================

    def test_fuel_oil_combustion(self, combustion_analyzer, fuel_oil_combustion_input):
        """Test fuel oil combustion analysis."""
        result, provenance = combustion_analyzer.calculate(fuel_oil_combustion_input)

        # Fuel oil has higher CO2_max than natural gas
        assert result.CO2_max_pct == pytest.approx(15.5, rel=0.01)
        assert result.excess_air_pct == pytest.approx(17.6, rel=0.02)
        assert result.stoichiometric_ratio == pytest.approx(1.176, rel=0.01)
        assert result.is_complete_combustion is True
        assert verify_provenance(provenance) is True

    def test_fuel_oil_with_sulfur(self, combustion_analyzer):
        """Test fuel oil with sulfur content."""
        inputs = CombustionInput(
            O2_pct=3.5,
            CO2_pct=13.5,
            CO_ppm=80.0,
            NOx_ppm=250.0,
            SO2_ppm=200.0,  # Fuel oil typically has sulfur
            flue_gas_temp_c=210.0,
            ambient_temp_c=25.0,
            fuel_type=FuelType.FUEL_OIL.value,
            gas_basis=GasBasis.DRY.value
        )

        result, provenance = combustion_analyzer.calculate(inputs)

        assert result.is_complete_combustion is True
        assert provenance.inputs["SO2_ppm"] == 200.0

    # =========================================================================
    # HAPPY PATH TESTS - COAL
    # =========================================================================

    def test_coal_combustion(self, combustion_analyzer, coal_combustion_input):
        """Test coal combustion analysis."""
        result, provenance = combustion_analyzer.calculate(coal_combustion_input)

        # Coal has highest CO2_max
        assert result.CO2_max_pct == pytest.approx(18.5, rel=0.01)
        assert result.excess_air_pct > 0
        assert result.stoichiometric_ratio > 1.0
        assert verify_provenance(provenance) is True

    def test_coal_high_CO(self, combustion_analyzer):
        """Test coal with high CO (incomplete combustion)."""
        inputs = CombustionInput(
            O2_pct=5.0,
            CO2_pct=15.0,
            CO_ppm=450.0,  # Above 400 ppm threshold
            NOx_ppm=400.0,
            SO2_ppm=600.0,
            flue_gas_temp_c=280.0,
            ambient_temp_c=25.0,
            fuel_type=FuelType.COAL.value,
            gas_basis=GasBasis.DRY.value
        )

        result, provenance = combustion_analyzer.calculate(inputs)

        assert result.is_complete_combustion is False  # CO > 400 ppm
        assert result.combustion_quality_rating in ["Fair", "Poor", "Critical"]

    # =========================================================================
    # HAPPY PATH TESTS - OTHER FUELS
    # =========================================================================

    @pytest.mark.parametrize("fuel_type,expected_CO2_max,expected_stoich_air", [
        (FuelType.NATURAL_GAS.value, 11.8, 17.2),
        (FuelType.FUEL_OIL.value, 15.5, 14.5),
        (FuelType.COAL.value, 18.5, 9.5),
        (FuelType.DIESEL.value, 15.3, 14.3),
        (FuelType.PROPANE.value, 13.7, 15.7),
        (FuelType.BIOMASS.value, 20.2, 6.0),
    ])
    def test_all_fuel_types(self, combustion_analyzer, fuel_type, expected_CO2_max, expected_stoich_air):
        """Test combustion analysis for all fuel types."""
        inputs = CombustionInput(
            O2_pct=3.5,
            CO2_pct=expected_CO2_max * 0.9,  # 90% of theoretical max
            CO_ppm=50.0,
            NOx_ppm=150.0,
            flue_gas_temp_c=180.0,
            ambient_temp_c=25.0,
            fuel_type=fuel_type,
            gas_basis=GasBasis.DRY.value
        )

        result, provenance = combustion_analyzer.calculate(inputs)

        assert result.CO2_max_pct == pytest.approx(expected_CO2_max, rel=0.01)
        assert verify_provenance(provenance) is True

    # =========================================================================
    # EDGE CASE TESTS
    # =========================================================================

    def test_low_O2_rich_combustion(self, combustion_analyzer, low_O2_combustion_input):
        """Test rich combustion (low O2, high CO)."""
        result, provenance = combustion_analyzer.calculate(low_O2_combustion_input)

        assert result.excess_air_pct == pytest.approx(5.0, rel=0.05)
        assert result.stoichiometric_ratio == pytest.approx(1.05, rel=0.01)
        assert result.is_complete_combustion is False  # CO = 800 ppm > 400
        assert result.combustion_quality_rating in ["Poor", "Critical"]

    def test_high_O2_lean_combustion(self, combustion_analyzer, high_O2_combustion_input):
        """Test lean combustion (high O2, excessive air)."""
        result, provenance = combustion_analyzer.calculate(high_O2_combustion_input)

        assert result.excess_air_pct == pytest.approx(90.9, rel=0.02)
        assert result.stoichiometric_ratio == pytest.approx(1.909, rel=0.01)
        assert result.is_complete_combustion is True  # CO low
        assert result.combustion_quality_rating in ["Fair", "Poor", "Critical"]  # Too much excess air

    def test_wet_to_dry_conversion(self, combustion_analyzer, wet_basis_combustion_input):
        """Test wet basis to dry basis conversion."""
        result, provenance = combustion_analyzer.calculate(wet_basis_combustion_input)

        # Wet basis values should be converted to dry
        assert result.O2_dry_pct == pytest.approx(3.33, rel=0.02)  # 3.0% wet → 3.33% dry
        assert result.CO2_dry_pct == pytest.approx(12.0, rel=0.02)  # 10.8% wet → 12.0% dry
        assert verify_provenance(provenance) is True

    def test_zero_O2_stoichiometric(self, combustion_analyzer):
        """Test stoichiometric combustion (O2 ≈ 0)."""
        inputs = CombustionInput(
            O2_pct=0.1,
            CO2_pct=11.7,
            CO_ppm=200.0,
            NOx_ppm=100.0,
            flue_gas_temp_c=200.0,
            ambient_temp_c=25.0,
            fuel_type=FuelType.NATURAL_GAS.value,
            gas_basis=GasBasis.DRY.value
        )

        result, provenance = combustion_analyzer.calculate(inputs)

        assert result.excess_air_pct == pytest.approx(0.48, rel=0.1)
        assert result.stoichiometric_ratio == pytest.approx(1.0048, rel=0.01)

    def test_high_temperature_flue_gas(self, combustion_analyzer):
        """Test high temperature flue gas (>1000°C)."""
        inputs = CombustionInput(
            O2_pct=5.0,
            CO2_pct=10.0,
            CO_ppm=100.0,
            NOx_ppm=300.0,
            flue_gas_temp_c=1100.0,  # Very high temperature
            ambient_temp_c=25.0,
            fuel_type=FuelType.COAL.value,
            gas_basis=GasBasis.DRY.value
        )

        result, provenance = combustion_analyzer.calculate(inputs)

        assert result.flue_gas_volume_nm3_kg > 0
        assert verify_provenance(provenance) is True

    def test_low_temperature_flue_gas(self, combustion_analyzer):
        """Test low temperature flue gas (near minimum)."""
        inputs = CombustionInput(
            O2_pct=8.0,
            CO2_pct=6.0,
            CO_ppm=50.0,
            NOx_ppm=80.0,
            flue_gas_temp_c=60.0,  # Low temperature
            ambient_temp_c=20.0,
            fuel_type=FuelType.NATURAL_GAS.value,
            gas_basis=GasBasis.DRY.value
        )

        result, provenance = combustion_analyzer.calculate(inputs)

        assert result.flue_gas_volume_nm3_kg > 0
        assert verify_provenance(provenance) is True

    # =========================================================================
    # ERROR HANDLING TESTS
    # =========================================================================

    def test_invalid_O2_negative(self, combustion_analyzer):
        """Test invalid O2 (negative) raises ValueError."""
        inputs = CombustionInput(
            O2_pct=-1.0,  # Invalid
            CO2_pct=12.0,
            CO_ppm=50.0,
            NOx_ppm=150.0,
            flue_gas_temp_c=180.0,
            ambient_temp_c=25.0,
            fuel_type=FuelType.NATURAL_GAS.value,
            gas_basis=GasBasis.DRY.value
        )

        with pytest.raises(ValueError, match="O2 concentration .* out of range"):
            combustion_analyzer.calculate(inputs)

    def test_invalid_O2_too_high(self, combustion_analyzer):
        """Test invalid O2 (>21%) raises ValueError."""
        inputs = CombustionInput(
            O2_pct=22.0,  # Invalid (>21%)
            CO2_pct=12.0,
            CO_ppm=50.0,
            NOx_ppm=150.0,
            flue_gas_temp_c=180.0,
            ambient_temp_c=25.0,
            fuel_type=FuelType.NATURAL_GAS.value,
            gas_basis=GasBasis.DRY.value
        )

        with pytest.raises(ValueError, match="O2 concentration .* out of range"):
            combustion_analyzer.calculate(inputs)

    def test_invalid_CO2_negative(self, combustion_analyzer):
        """Test invalid CO2 (negative) raises ValueError."""
        inputs = CombustionInput(
            O2_pct=3.5,
            CO2_pct=-5.0,  # Invalid
            CO_ppm=50.0,
            NOx_ppm=150.0,
            flue_gas_temp_c=180.0,
            ambient_temp_c=25.0,
            fuel_type=FuelType.NATURAL_GAS.value,
            gas_basis=GasBasis.DRY.value
        )

        with pytest.raises(ValueError, match="CO2 concentration .* out of range"):
            combustion_analyzer.calculate(inputs)

    def test_invalid_CO_negative(self, combustion_analyzer):
        """Test invalid CO (negative) raises ValueError."""
        inputs = CombustionInput(
            O2_pct=3.5,
            CO2_pct=12.0,
            CO_ppm=-10.0,  # Invalid
            NOx_ppm=150.0,
            flue_gas_temp_c=180.0,
            ambient_temp_c=25.0,
            fuel_type=FuelType.NATURAL_GAS.value,
            gas_basis=GasBasis.DRY.value
        )

        with pytest.raises(ValueError, match="CO concentration cannot be negative"):
            combustion_analyzer.calculate(inputs)

    def test_invalid_NOx_negative(self, combustion_analyzer):
        """Test invalid NOx (negative) raises ValueError."""
        inputs = CombustionInput(
            O2_pct=3.5,
            CO2_pct=12.0,
            CO_ppm=50.0,
            NOx_ppm=-50.0,  # Invalid
            flue_gas_temp_c=180.0,
            ambient_temp_c=25.0,
            fuel_type=FuelType.NATURAL_GAS.value,
            gas_basis=GasBasis.DRY.value
        )

        with pytest.raises(ValueError, match="NOx concentration cannot be negative"):
            combustion_analyzer.calculate(inputs)

    def test_invalid_flue_gas_temp_too_low(self, combustion_analyzer):
        """Test invalid flue gas temperature (too low) raises ValueError."""
        inputs = CombustionInput(
            O2_pct=3.5,
            CO2_pct=12.0,
            CO_ppm=50.0,
            NOx_ppm=150.0,
            flue_gas_temp_c=30.0,  # Too low (<50°C)
            ambient_temp_c=25.0,
            fuel_type=FuelType.NATURAL_GAS.value,
            gas_basis=GasBasis.DRY.value
        )

        with pytest.raises(ValueError, match="Flue gas temperature .* out of range"):
            combustion_analyzer.calculate(inputs)

    def test_invalid_flue_gas_temp_too_high(self, combustion_analyzer):
        """Test invalid flue gas temperature (too high) raises ValueError."""
        inputs = CombustionInput(
            O2_pct=3.5,
            CO2_pct=12.0,
            CO_ppm=50.0,
            NOx_ppm=150.0,
            flue_gas_temp_c=1300.0,  # Too high (>1200°C)
            ambient_temp_c=25.0,
            fuel_type=FuelType.NATURAL_GAS.value,
            gas_basis=GasBasis.DRY.value
        )

        with pytest.raises(ValueError, match="Flue gas temperature .* out of range"):
            combustion_analyzer.calculate(inputs)

    def test_invalid_ambient_temp_too_low(self, combustion_analyzer):
        """Test invalid ambient temperature (too low) raises ValueError."""
        inputs = CombustionInput(
            O2_pct=3.5,
            CO2_pct=12.0,
            CO_ppm=50.0,
            NOx_ppm=150.0,
            flue_gas_temp_c=180.0,
            ambient_temp_c=-30.0,  # Too low (<-20°C)
            fuel_type=FuelType.NATURAL_GAS.value,
            gas_basis=GasBasis.DRY.value
        )

        with pytest.raises(ValueError, match="Ambient temperature .* out of range"):
            combustion_analyzer.calculate(inputs)

    def test_unknown_fuel_type(self, combustion_analyzer):
        """Test unknown fuel type raises ValueError."""
        inputs = CombustionInput(
            O2_pct=3.5,
            CO2_pct=12.0,
            CO_ppm=50.0,
            NOx_ppm=150.0,
            flue_gas_temp_c=180.0,
            ambient_temp_c=25.0,
            fuel_type="Unknown Fuel",  # Invalid
            gas_basis=GasBasis.DRY.value
        )

        with pytest.raises(ValueError, match="Unknown fuel type"):
            combustion_analyzer.calculate(inputs)

    # =========================================================================
    # PROVENANCE TESTS
    # =========================================================================

    def test_provenance_determinism(self, combustion_analyzer, natural_gas_combustion_input):
        """Test provenance hash is deterministic (same input → same hash)."""
        result1, provenance1 = combustion_analyzer.calculate(natural_gas_combustion_input)
        result2, provenance2 = combustion_analyzer.calculate(natural_gas_combustion_input)

        # Same inputs must produce same provenance hash
        assert provenance1.provenance_hash == provenance2.provenance_hash

        # Same outputs
        assert result1.excess_air_pct == result2.excess_air_pct
        assert result1.combustion_quality_index == result2.combustion_quality_index

    def test_provenance_steps_completeness(self, combustion_analyzer, natural_gas_combustion_input):
        """Test provenance includes all calculation steps."""
        result, provenance = combustion_analyzer.calculate(natural_gas_combustion_input)

        # Should have at least 9 steps for full analysis
        assert len(provenance.calculation_steps) >= 9

        # Check step numbers are sequential
        step_numbers = [step.step_number for step in provenance.calculation_steps]
        assert step_numbers == list(range(1, len(step_numbers) + 1))

        # Check each step has required fields
        for step in provenance.calculation_steps:
            assert step.description != ""
            assert step.operation != ""
            assert step.formula != ""

    def test_provenance_metadata(self, combustion_analyzer, natural_gas_combustion_input):
        """Test provenance includes correct metadata."""
        result, provenance = combustion_analyzer.calculate(natural_gas_combustion_input)

        assert "standards" in provenance.metadata
        assert "ASME PTC 4.1" in provenance.metadata["standards"]
        assert "EPA Method 19" in provenance.metadata["standards"]
        assert provenance.metadata["domain"] == "Combustion Analysis"

    # =========================================================================
    # PERFORMANCE TESTS
    # =========================================================================

    @pytest.mark.performance
    def test_calculation_speed(self, combustion_analyzer, natural_gas_combustion_input, benchmark):
        """Test calculation meets performance target (<5ms)."""
        result = benchmark(combustion_analyzer.calculate, natural_gas_combustion_input)

        # Execution time should be <5ms
        assert benchmark.stats.stats.mean < 0.005  # 5ms

    @pytest.mark.performance
    def test_batch_processing_throughput(self, combustion_analyzer, benchmark_dataset):
        """Test batch processing throughput (target: >1000 records/sec)."""
        import time

        start_time = time.time()
        results = []

        for input_data in benchmark_dataset[:100]:  # Test 100 records
            result, provenance = combustion_analyzer.calculate(input_data)
            results.append(result)

        end_time = time.time()
        duration = end_time - start_time
        throughput = 100 / duration

        assert throughput > 1000  # Target: >1000 records/sec
        assert len(results) == 100


# =============================================================================
# STANDALONE FUNCTION TESTS
# =============================================================================

@pytest.mark.unit
class TestStandaloneFunctions:
    """Test standalone utility functions."""

    @pytest.mark.parametrize("O2_pct,expected_excess_air", [
        (0.0, 0.0),
        (1.0, 5.0),
        (2.0, 10.53),
        (3.0, 16.67),
        (3.5, 20.0),
        (4.0, 23.53),
        (5.0, 31.25),
        (10.0, 90.91),
    ])
    def test_calculate_excess_air_from_O2(self, O2_pct, expected_excess_air):
        """Test excess air calculation from O2 against known values."""
        result = calculate_excess_air_from_O2(O2_pct)
        assert result == pytest.approx(expected_excess_air, rel=0.01)

    def test_calculate_excess_air_invalid_O2_negative(self):
        """Test excess air calculation with negative O2 raises ValueError."""
        with pytest.raises(ValueError):
            calculate_excess_air_from_O2(-1.0)

    def test_calculate_excess_air_invalid_O2_too_high(self):
        """Test excess air calculation with O2 >= 21% raises ValueError."""
        with pytest.raises(ValueError):
            calculate_excess_air_from_O2(21.0)

    @pytest.mark.parametrize("wet_value,h2o_pct,expected_dry", [
        (3.0, 10.0, 3.33),
        (5.0, 10.0, 5.56),
        (10.0, 10.0, 11.11),
        (3.0, 15.0, 3.53),
        (8.0, 5.0, 8.42),
    ])
    def test_convert_wet_to_dry(self, wet_value, h2o_pct, expected_dry):
        """Test wet to dry conversion."""
        result = convert_wet_to_dry(wet_value, h2o_pct)
        assert result == pytest.approx(expected_dry, rel=0.01)

    @pytest.mark.parametrize("dry_value,h2o_pct,expected_wet", [
        (3.33, 10.0, 3.0),
        (5.56, 10.0, 5.0),
        (11.11, 10.0, 10.0),
        (3.53, 15.0, 3.0),
        (8.42, 5.0, 8.0),
    ])
    def test_convert_dry_to_wet(self, dry_value, h2o_pct, expected_wet):
        """Test dry to wet conversion."""
        result = convert_dry_to_wet(dry_value, h2o_pct)
        assert result == pytest.approx(expected_wet, rel=0.01)

    def test_wet_dry_roundtrip(self):
        """Test wet/dry conversion roundtrip."""
        original = 5.0
        h2o = 10.0

        dry = convert_wet_to_dry(original, h2o)
        wet = convert_dry_to_wet(dry, h2o)

        assert wet == pytest.approx(original, rel=1e-6)
