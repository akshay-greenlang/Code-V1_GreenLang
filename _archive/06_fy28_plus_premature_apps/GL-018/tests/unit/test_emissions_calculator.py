"""
GL-018 FLUEFLOW - Emissions Calculator Unit Tests

Comprehensive unit tests for EmissionsCalculator with 95%+ coverage target.
Tests EPA Method 19 compliance and unit conversions.

Target Coverage: 95%+
Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from calculators.emissions_calculator import (
    EmissionsCalculator,
    EmissionsInput,
    EmissionsOutput,
    MW_NOX,
    MW_CO,
    MW_SO2,
    MOLAR_VOLUME_NM3_KMOL,
    convert_ppm_to_mg_nm3,
    convert_mg_nm3_to_ppm,
    correct_to_reference_O2,
    calculate_CO_CO2_ratio,
)
from calculators.provenance import verify_provenance


@pytest.mark.unit
@pytest.mark.calculator
class TestEmissionsCalculator:
    """Comprehensive test suite for EmissionsCalculator."""

    def test_initialization(self):
        """Test EmissionsCalculator initializes correctly."""
        calculator = EmissionsCalculator()
        assert calculator.VERSION == "1.0.0"
        assert calculator.NAME == "EmissionsCalculator"

    def test_natural_gas_emissions(self, emissions_calculator, natural_gas_emissions_input):
        """Test natural gas emissions analysis."""
        result, provenance = emissions_calculator.calculate(natural_gas_emissions_input)

        assert isinstance(result, EmissionsOutput)
        assert result.NOx_mg_nm3 > 0
        assert result.CO_mg_nm3 > 0
        assert result.NOx_kg_hr > 0
        assert result.CO_kg_hr > 0
        assert result.CO_CO2_ratio < 0.01  # Good combustion
        assert result.NOx_compliance_status in ["Compliant", "Compliant (Good Margin)"]
        assert verify_provenance(provenance) is True

    @pytest.mark.parametrize("NOx_ppm,expected_mg_nm3", [
        (50.0, 102.62),
        (100.0, 205.25),
        (150.0, 307.87),
        (200.0, 410.50),
    ])
    def test_NOx_ppm_to_mg_conversion(self, emissions_calculator, NOx_ppm, expected_mg_nm3):
        """Test NOx ppm to mg/Nm³ conversion."""
        inputs = EmissionsInput(
            NOx_ppm=NOx_ppm,
            CO_ppm=50.0,
            SO2_ppm=0.0,
            CO2_pct=12.0,
            O2_pct=3.5,
            flue_gas_temp_c=180.0,
            flue_gas_flow_nm3_hr=50000.0,
            fuel_type="Natural Gas"
        )

        result, provenance = emissions_calculator.calculate(inputs)
        assert result.NOx_mg_nm3 == pytest.approx(expected_mg_nm3, rel=0.01)

    @pytest.mark.parametrize("CO_ppm,expected_mg_nm3", [
        (50.0, 62.47),
        (100.0, 124.93),
        (200.0, 249.87),
        (400.0, 499.73),
    ])
    def test_CO_ppm_to_mg_conversion(self, emissions_calculator, CO_ppm, expected_mg_nm3):
        """Test CO ppm to mg/Nm³ conversion."""
        inputs = EmissionsInput(
            NOx_ppm=150.0,
            CO_ppm=CO_ppm,
            SO2_ppm=0.0,
            CO2_pct=12.0,
            O2_pct=3.5,
            flue_gas_temp_c=180.0,
            flue_gas_flow_nm3_hr=50000.0,
            fuel_type="Natural Gas"
        )

        result, provenance = emissions_calculator.calculate(inputs)
        assert result.CO_mg_nm3 == pytest.approx(expected_mg_nm3, rel=0.01)

    def test_O2_correction(self, emissions_calculator):
        """Test O2 reference correction (EPA Method 19)."""
        # Measured at 5% O2, correct to 3% O2 reference
        inputs = EmissionsInput(
            NOx_ppm=150.0,
            CO_ppm=50.0,
            SO2_ppm=0.0,
            CO2_pct=10.0,
            O2_pct=5.0,  # Measured at 5% O2
            flue_gas_temp_c=180.0,
            flue_gas_flow_nm3_hr=50000.0,
            fuel_type="Natural Gas",
            reference_O2_pct=3.0  # Correct to 3% O2
        )

        result, provenance = emissions_calculator.calculate(inputs)

        # Correction factor = (21-3)/(21-5) = 18/16 = 1.125
        # Corrected values should be higher
        assert result.NOx_mg_nm3_corrected > result.NOx_mg_nm3
        assert result.CO_mg_nm3_corrected > result.CO_mg_nm3

    @pytest.mark.parametrize("CO_ppm,CO2_pct,expected_ratio", [
        (50.0, 12.0, 0.000417),  # Excellent
        (100.0, 12.0, 0.000833),  # Good
        (200.0, 10.0, 0.002000),  # Fair
        (500.0, 8.0, 0.006250),  # Poor
    ])
    def test_CO_CO2_ratio(self, emissions_calculator, CO_ppm, CO2_pct, expected_ratio):
        """Test CO/CO2 ratio calculation."""
        inputs = EmissionsInput(
            NOx_ppm=150.0,
            CO_ppm=CO_ppm,
            SO2_ppm=0.0,
            CO2_pct=CO2_pct,
            O2_pct=3.5,
            flue_gas_temp_c=180.0,
            flue_gas_flow_nm3_hr=50000.0,
            fuel_type="Natural Gas"
        )

        result, provenance = emissions_calculator.calculate(inputs)
        assert result.CO_CO2_ratio == pytest.approx(expected_ratio, rel=0.01)

    def test_mass_emission_rate(self, emissions_calculator):
        """Test mass emission rate calculation."""
        inputs = EmissionsInput(
            NOx_ppm=100.0,  # → ~205 mg/Nm³
            CO_ppm=50.0,    # → ~62.5 mg/Nm³
            SO2_ppm=0.0,
            CO2_pct=12.0,
            O2_pct=3.5,
            flue_gas_temp_c=180.0,
            flue_gas_flow_nm3_hr=50000.0,  # 50,000 Nm³/hr
            fuel_type="Natural Gas"
        )

        result, provenance = emissions_calculator.calculate(inputs)

        # NOx: 205 mg/Nm³ × 50,000 Nm³/hr ÷ 1,000,000 = 10.25 kg/hr
        assert result.NOx_kg_hr == pytest.approx(10.25, rel=0.02)

        # CO: 62.5 mg/Nm³ × 50,000 Nm³/hr ÷ 1,000,000 = 3.125 kg/hr
        assert result.CO_kg_hr == pytest.approx(3.125, rel=0.02)

    def test_compliance_check_good_margin(self, emissions_calculator):
        """Test emissions compliance with good margin."""
        inputs = EmissionsInput(
            NOx_ppm=50.0,  # Well below limit
            CO_ppm=30.0,   # Well below limit
            SO2_ppm=0.0,
            CO2_pct=12.0,
            O2_pct=3.0,
            flue_gas_temp_c=180.0,
            flue_gas_flow_nm3_hr=50000.0,
            fuel_type="Natural Gas",
            reference_O2_pct=3.0
        )

        result, provenance = emissions_calculator.calculate(inputs)

        assert result.NOx_compliance_status == "Compliant (Good Margin)"
        assert result.CO_compliance_status == "Compliant (Good Margin)"

    def test_compliance_check_near_limit(self, emissions_calculator):
        """Test emissions near compliance limit."""
        inputs = EmissionsInput(
            NOx_ppm=195.0,  # Near 200 ppm limit
            CO_ppm=390.0,   # Near 400 ppm limit
            SO2_ppm=0.0,
            CO2_pct=12.0,
            O2_pct=3.0,
            flue_gas_temp_c=180.0,
            flue_gas_flow_nm3_hr=50000.0,
            fuel_type="Natural Gas",
            reference_O2_pct=3.0
        )

        result, provenance = emissions_calculator.calculate(inputs)

        assert "Compliant" in result.NOx_compliance_status
        assert "Compliant" in result.CO_compliance_status

    def test_invalid_NOx_negative(self, emissions_calculator):
        """Test negative NOx raises ValueError."""
        inputs = EmissionsInput(
            NOx_ppm=-10.0,  # Invalid
            CO_ppm=50.0,
            SO2_ppm=0.0,
            CO2_pct=12.0,
            O2_pct=3.5,
            flue_gas_temp_c=180.0,
            flue_gas_flow_nm3_hr=50000.0,
            fuel_type="Natural Gas"
        )

        with pytest.raises(ValueError, match="NOx concentration cannot be negative"):
            emissions_calculator.calculate(inputs)

    def test_invalid_flow_rate(self, emissions_calculator):
        """Test invalid flow rate raises ValueError."""
        inputs = EmissionsInput(
            NOx_ppm=150.0,
            CO_ppm=50.0,
            SO2_ppm=0.0,
            CO2_pct=12.0,
            O2_pct=3.5,
            flue_gas_temp_c=180.0,
            flue_gas_flow_nm3_hr=0.0,  # Invalid
            fuel_type="Natural Gas"
        )

        with pytest.raises(ValueError, match="Flue gas flow rate must be positive"):
            emissions_calculator.calculate(inputs)

    def test_provenance_determinism(self, emissions_calculator, natural_gas_emissions_input):
        """Test provenance determinism."""
        result1, provenance1 = emissions_calculator.calculate(natural_gas_emissions_input)
        result2, provenance2 = emissions_calculator.calculate(natural_gas_emissions_input)

        assert provenance1.provenance_hash == provenance2.provenance_hash


@pytest.mark.unit
class TestStandaloneFunctions:
    """Test standalone functions."""

    def test_convert_ppm_to_mg_nm3(self):
        """Test ppm to mg/Nm³ conversion."""
        # NOx: 100 ppm × 46 / 22.414 = 205.25 mg/Nm³
        result = convert_ppm_to_mg_nm3(100.0, MW_NOX)
        assert result == pytest.approx(205.25, rel=0.01)

        # CO: 100 ppm × 28 / 22.414 = 124.93 mg/Nm³
        result = convert_ppm_to_mg_nm3(100.0, MW_CO)
        assert result == pytest.approx(124.93, rel=0.01)

    def test_convert_mg_nm3_to_ppm(self):
        """Test mg/Nm³ to ppm conversion."""
        result = convert_mg_nm3_to_ppm(205.25, MW_NOX)
        assert result == pytest.approx(100.0, rel=0.01)

    def test_correct_to_reference_O2(self):
        """Test O2 correction."""
        # Measured at 5% O2, correct to 3% O2
        # Factor = (21-3)/(21-5) = 18/16 = 1.125
        result = correct_to_reference_O2(100.0, 5.0, 3.0)
        assert result == pytest.approx(112.5, rel=0.01)

    def test_calculate_CO_CO2_ratio(self):
        """Test CO/CO2 ratio calculation."""
        ratio = calculate_CO_CO2_ratio(50.0, 12.0)
        assert ratio == pytest.approx(0.000417, rel=0.01)
