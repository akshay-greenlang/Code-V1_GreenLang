"""
GL-011 FUELCRAFT - Heating Value Calculator Tests

Unit tests for HeatingValueCalculator including HHV, LHV,
Wobbe Index calculations, and gas composition analysis.
Tests validate accuracy against known reference values.
"""

import pytest
import math
from datetime import datetime, timezone

from greenlang.agents.process_heat.gl_011_fuel_optimization.heating_value import (
    HeatingValueCalculator,
    HeatingValueInput,
    HeatingValueResult,
    WobbeIndexResult,
    GAS_COMPONENTS,
    LIQUID_FUEL_PROPERTIES,
    SOLID_FUEL_PROPERTIES,
)


class TestHeatingValueCalculator:
    """Tests for HeatingValueCalculator class."""

    def test_initialization(self, heating_value_calculator):
        """Test calculator initialization."""
        calc = heating_value_calculator

        assert calc.reference_temp_f == 60.0
        assert calc.reference_pressure_psia == 14.696
        assert calc.calculation_count == 0

    def test_initialization_custom_reference(self):
        """Test calculator with custom reference temperature."""
        calc = HeatingValueCalculator(reference_temp_f=70.0)

        assert calc.reference_temp_f == 70.0
        assert calc.reference_pressure_psia == 14.696


class TestGasHeatingValueCalculation:
    """Tests for gas heating value calculations."""

    def test_pure_methane_hhv(self, heating_value_calculator):
        """Test HHV calculation for pure methane."""
        input_data = HeatingValueInput(
            fuel_type="methane",
            methane_pct=100.0,
        )

        result = heating_value_calculator.calculate_gas_heating_value(input_data)

        # Reference value for pure methane: 1010 BTU/SCF
        assert result.hhv_btu_scf == pytest.approx(1010.0, rel=0.01)
        assert result.lhv_btu_scf == pytest.approx(909.4, rel=0.01)
        assert result.specific_gravity == pytest.approx(0.5539, rel=0.01)

    def test_natural_gas_composition(self, heating_value_calculator, heating_value_input_natural_gas):
        """Test HHV for typical natural gas composition."""
        result = heating_value_calculator.calculate_gas_heating_value(
            heating_value_input_natural_gas
        )

        # Pipeline quality natural gas: ~1010-1040 BTU/SCF
        assert 1000.0 <= result.hhv_btu_scf <= 1050.0
        assert 900.0 <= result.lhv_btu_scf <= 950.0
        assert result.specific_gravity > 0
        assert result.wobbe_index is not None

    def test_biogas_composition(self, heating_value_calculator, heating_value_input_biogas):
        """Test HHV for biogas composition."""
        result = heating_value_calculator.calculate_gas_heating_value(
            heating_value_input_biogas
        )

        # Biogas: ~550-650 BTU/SCF (high CO2 content)
        assert 500.0 <= result.hhv_btu_scf <= 700.0
        assert result.specific_gravity > 0.6  # Higher than natural gas due to CO2

    @pytest.mark.parametrize("methane,ethane,propane,nitrogen,co2,expected_hhv_min,expected_hhv_max", [
        (95.0, 2.5, 0.5, 1.5, 0.5, 1000.0, 1020.0),  # Pipeline quality
        (98.0, 1.0, 0.2, 0.5, 0.3, 990.0, 1010.0),   # High methane
        (90.0, 5.0, 2.0, 2.0, 1.0, 1010.0, 1040.0),  # Rich gas
        (85.0, 7.0, 4.0, 2.5, 1.5, 1030.0, 1060.0),  # Very rich gas
    ])
    def test_various_gas_compositions(
        self,
        heating_value_calculator,
        methane,
        ethane,
        propane,
        nitrogen,
        co2,
        expected_hhv_min,
        expected_hhv_max,
    ):
        """Test HHV for various gas compositions."""
        input_data = HeatingValueInput(
            fuel_type="test_gas",
            methane_pct=methane,
            ethane_pct=ethane,
            propane_pct=propane,
            nitrogen_pct=nitrogen,
            co2_pct=co2,
        )

        result = heating_value_calculator.calculate_gas_heating_value(input_data)

        assert expected_hhv_min <= result.hhv_btu_scf <= expected_hhv_max

    def test_hydrogen_blend(self, heating_value_calculator):
        """Test HHV for natural gas with hydrogen blend."""
        input_data = HeatingValueInput(
            fuel_type="h2_blend",
            methane_pct=80.0,
            hydrogen_pct=20.0,
        )

        result = heating_value_calculator.calculate_gas_heating_value(input_data)

        # H2 has lower HHV per SCF but higher per lb
        # 80% CH4 (1010) + 20% H2 (324) = ~872 BTU/SCF
        assert 850.0 <= result.hhv_btu_scf <= 900.0
        assert result.specific_gravity < 0.5539  # H2 is much lighter

    def test_default_composition_fallback(self, heating_value_calculator):
        """Test fallback to default composition when none provided."""
        input_data = HeatingValueInput(fuel_type="natural_gas")

        result = heating_value_calculator.calculate_gas_heating_value(input_data)

        # Should use default natural gas composition
        assert result.hhv_btu_scf > 0
        assert len(result.warnings) > 0  # Warning about using default

    def test_composition_normalization(self, heating_value_calculator):
        """Test composition normalization when not summing to 100%."""
        input_data = HeatingValueInput(
            fuel_type="test",
            methane_pct=90.0,
            ethane_pct=5.0,
            # Total: 95%, should be normalized
        )

        result = heating_value_calculator.calculate_gas_heating_value(input_data)

        assert result.hhv_btu_scf > 0
        assert len(result.warnings) > 0  # Warning about normalization

    def test_provenance_hash_deterministic(self, heating_value_calculator):
        """Test provenance hash is deterministic for same input."""
        input_data = HeatingValueInput(
            fuel_type="test",
            methane_pct=95.0,
            ethane_pct=3.0,
            propane_pct=1.0,
            nitrogen_pct=1.0,
        )

        result1 = heating_value_calculator.calculate_gas_heating_value(input_data)
        result2 = heating_value_calculator.calculate_gas_heating_value(input_data)

        assert result1.provenance_hash == result2.provenance_hash

    def test_calculation_count_increments(self, heating_value_calculator):
        """Test calculation count increments with each call."""
        initial_count = heating_value_calculator.calculation_count

        input_data = HeatingValueInput(fuel_type="test", methane_pct=95.0)
        heating_value_calculator.calculate_gas_heating_value(input_data)

        assert heating_value_calculator.calculation_count == initial_count + 1

        heating_value_calculator.calculate_gas_heating_value(input_data)
        assert heating_value_calculator.calculation_count == initial_count + 2


class TestLiquidHeatingValueCalculation:
    """Tests for liquid fuel heating value calculations."""

    def test_no2_fuel_oil(self, heating_value_calculator):
        """Test HHV for #2 fuel oil."""
        result = heating_value_calculator.calculate_liquid_heating_value("no2_fuel_oil")

        # Reference: ~19,580 BTU/lb
        assert result.hhv_btu_lb == pytest.approx(19580, rel=0.01)
        assert result.lhv_btu_lb == pytest.approx(18410, rel=0.01)

    def test_no6_fuel_oil(self, heating_value_calculator):
        """Test HHV for #6 fuel oil."""
        result = heating_value_calculator.calculate_liquid_heating_value("no6_fuel_oil")

        # Reference: ~18,300 BTU/lb
        assert result.hhv_btu_lb == pytest.approx(18300, rel=0.01)

    def test_diesel(self, heating_value_calculator):
        """Test HHV for diesel."""
        result = heating_value_calculator.calculate_liquid_heating_value("diesel")

        # Reference: ~19,300 BTU/lb
        assert result.hhv_btu_lb == pytest.approx(19300, rel=0.01)

    def test_lpg_propane(self, heating_value_calculator):
        """Test HHV for LPG propane."""
        result = heating_value_calculator.calculate_liquid_heating_value("lpg_propane")

        # Reference: ~21,500 BTU/lb
        assert result.hhv_btu_lb == pytest.approx(21500, rel=0.01)

    def test_api_gravity_correlation(self, heating_value_calculator):
        """Test HHV calculation using API gravity correlation."""
        # API gravity of 38 (typical diesel)
        result = heating_value_calculator.calculate_liquid_heating_value(
            "custom_oil",
            api_gravity=38.0,
        )

        # Should produce reasonable HHV
        assert 18000 <= result.hhv_btu_lb <= 20000
        assert len(result.warnings) > 0  # Warning about using correlation

    def test_unknown_fuel_no_api_gravity_raises(self, heating_value_calculator):
        """Test error when unknown fuel has no API gravity."""
        with pytest.raises(ValueError, match="Unknown fuel type"):
            heating_value_calculator.calculate_liquid_heating_value("unknown_fuel")


class TestSolidHeatingValueCalculation:
    """Tests for solid fuel heating value calculations."""

    def test_bituminous_coal(self, heating_value_calculator):
        """Test HHV for bituminous coal."""
        result = heating_value_calculator.calculate_solid_heating_value("coal_bituminous")

        # Reference: ~12,500 BTU/lb
        assert result.hhv_btu_lb == pytest.approx(12500, rel=0.05)

    def test_anthracite_coal(self, heating_value_calculator):
        """Test HHV for anthracite coal."""
        result = heating_value_calculator.calculate_solid_heating_value("coal_anthracite")

        # Reference: ~13,000 BTU/lb
        assert result.hhv_btu_lb == pytest.approx(13000, rel=0.05)

    def test_sub_bituminous_coal(self, heating_value_calculator):
        """Test HHV for sub-bituminous coal."""
        result = heating_value_calculator.calculate_solid_heating_value("coal_sub_bituminous")

        # Reference: ~9,500 BTU/lb
        assert result.hhv_btu_lb == pytest.approx(9500, rel=0.05)

    def test_biomass_wood(self, heating_value_calculator):
        """Test HHV for wood biomass."""
        result = heating_value_calculator.calculate_solid_heating_value("biomass_wood")

        # Reference: ~8,500 BTU/lb
        assert result.hhv_btu_lb == pytest.approx(8500, rel=0.05)

    def test_moisture_correction(self, heating_value_calculator):
        """Test moisture content correction."""
        # Default moisture
        result_default = heating_value_calculator.calculate_solid_heating_value(
            "biomass_wood"
        )

        # Higher moisture (30% vs default 25%)
        result_wet = heating_value_calculator.calculate_solid_heating_value(
            "biomass_wood",
            moisture_pct=30.0,
        )

        # Higher moisture = lower HHV
        assert result_wet.hhv_btu_lb < result_default.hhv_btu_lb
        assert len(result_wet.warnings) > 0

    def test_ash_correction(self, heating_value_calculator):
        """Test ash content correction."""
        # Default ash
        result_default = heating_value_calculator.calculate_solid_heating_value(
            "coal_bituminous"
        )

        # Higher ash (12% vs default 8%)
        result_high_ash = heating_value_calculator.calculate_solid_heating_value(
            "coal_bituminous",
            ash_pct=12.0,
        )

        # Higher ash = lower HHV
        assert result_high_ash.hhv_btu_lb < result_default.hhv_btu_lb

    def test_unknown_solid_fuel_raises(self, heating_value_calculator):
        """Test error for unknown solid fuel."""
        with pytest.raises(ValueError, match="Unknown solid fuel"):
            heating_value_calculator.calculate_solid_heating_value("unknown_solid")


class TestWobbeIndexCalculation:
    """Tests for Wobbe Index calculations."""

    def test_wobbe_index_formula(self, heating_value_calculator):
        """Test Wobbe Index calculation formula: WI = HHV / sqrt(SG)."""
        hhv = 1020.0
        sg = 0.60

        result = heating_value_calculator.calculate_wobbe_index(hhv, sg)

        expected_wi = hhv / math.sqrt(sg)
        assert result.wobbe_index == pytest.approx(expected_wi, rel=0.001)

    def test_wobbe_index_natural_gas(self, heating_value_calculator):
        """Test Wobbe Index for typical natural gas."""
        result = heating_value_calculator.calculate_wobbe_index(
            hhv_btu_scf=1020.0,
            specific_gravity=0.60,
        )

        # Typical natural gas Wobbe: ~1300-1380
        assert 1300 <= result.wobbe_index <= 1400
        assert result.interchangeable is True

    def test_wobbe_interchangeability_excellent(self, heating_value_calculator):
        """Test excellent interchangeability (< 2% deviation)."""
        result = heating_value_calculator.calculate_wobbe_index(
            hhv_btu_scf=1020.0,
            specific_gravity=0.5636,  # Gives WI ~1358
            reference_wobbe=1360.0,
        )

        assert result.interchangeability_status == "excellent"
        assert result.interchangeable is True
        assert result.deviation_pct < 2.0

    def test_wobbe_interchangeability_acceptable(self, heating_value_calculator):
        """Test acceptable interchangeability (2-5% deviation)."""
        result = heating_value_calculator.calculate_wobbe_index(
            hhv_btu_scf=1020.0,
            specific_gravity=0.52,  # Gives WI ~1415
            reference_wobbe=1360.0,
        )

        assert result.interchangeability_status == "acceptable"
        assert result.interchangeable is True
        assert 2.0 <= result.deviation_pct <= 5.0

    def test_wobbe_interchangeability_marginal(self, heating_value_calculator):
        """Test marginal interchangeability (5-10% deviation)."""
        result = heating_value_calculator.calculate_wobbe_index(
            hhv_btu_scf=1020.0,
            specific_gravity=0.45,  # Gives WI ~1520
            reference_wobbe=1360.0,
        )

        assert result.interchangeability_status == "marginal"
        assert result.interchangeable is False
        assert 5.0 < result.deviation_pct <= 10.0

    def test_wobbe_not_interchangeable(self, heating_value_calculator):
        """Test not interchangeable (> 10% deviation)."""
        result = heating_value_calculator.calculate_wobbe_index(
            hhv_btu_scf=600.0,  # Biogas
            specific_gravity=0.90,
            reference_wobbe=1360.0,
        )

        assert result.interchangeability_status == "not_interchangeable"
        assert result.interchangeable is False
        assert result.deviation_pct > 10.0

    def test_modified_wobbe_index_temperature(self, heating_value_calculator):
        """Test modified Wobbe Index with temperature correction."""
        # At standard 60F
        result_60 = heating_value_calculator.calculate_wobbe_index(
            hhv_btu_scf=1020.0,
            specific_gravity=0.60,
            temperature_f=60.0,
        )

        # At higher temperature
        result_80 = heating_value_calculator.calculate_wobbe_index(
            hhv_btu_scf=1020.0,
            specific_gravity=0.60,
            temperature_f=80.0,
        )

        # Modified WI should be different
        assert result_60.wobbe_index_modified != result_80.wobbe_index_modified

    def test_wobbe_zero_sg_raises(self, heating_value_calculator):
        """Test error when specific gravity is zero."""
        with pytest.raises(ValueError, match="Specific gravity must be positive"):
            heating_value_calculator.calculate_wobbe_index(1020.0, 0.0)


class TestHHVToLHVConversion:
    """Tests for HHV to LHV conversion."""

    def test_hhv_to_lhv_natural_gas(self, heating_value_calculator):
        """Test HHV to LHV conversion for natural gas."""
        hhv = 1020.0  # BTU/SCF (typical natural gas)

        # Typical H content for natural gas: ~25%
        lhv = heating_value_calculator.convert_hhv_to_lhv(
            hhv=hhv,
            hydrogen_content_pct=25.0,
            moisture_content_pct=0.0,
        )

        # LHV should be ~10% lower than HHV for natural gas
        assert lhv < hhv
        assert lhv == pytest.approx(hhv * 0.9, rel=0.05)

    def test_hhv_to_lhv_fuel_oil(self, heating_value_calculator):
        """Test HHV to LHV conversion for fuel oil."""
        hhv = 19580.0  # BTU/lb (#2 fuel oil)

        # Fuel oil: ~13% H content
        lhv = heating_value_calculator.convert_hhv_to_lhv(
            hhv=hhv,
            hydrogen_content_pct=13.0,
            moisture_content_pct=0.0,
        )

        # LHV should be ~6% lower for fuel oil
        assert lhv < hhv
        expected_lhv = 18410.0
        assert lhv == pytest.approx(expected_lhv, rel=0.05)

    def test_hhv_to_lhv_with_moisture(self, heating_value_calculator):
        """Test HHV to LHV conversion with moisture content."""
        hhv = 8500.0  # BTU/lb (wood biomass)

        # With 25% moisture
        lhv_wet = heating_value_calculator.convert_hhv_to_lhv(
            hhv=hhv,
            hydrogen_content_pct=6.0,
            moisture_content_pct=25.0,
        )

        # Without moisture
        lhv_dry = heating_value_calculator.convert_hhv_to_lhv(
            hhv=hhv,
            hydrogen_content_pct=6.0,
            moisture_content_pct=0.0,
        )

        # Moisture reduces LHV
        assert lhv_wet < lhv_dry


class TestGasComponentConstants:
    """Tests for gas component constant values."""

    def test_methane_properties(self):
        """Test methane reference properties."""
        methane = GAS_COMPONENTS["methane"]

        assert methane.name == "Methane"
        assert methane.molecular_weight == pytest.approx(16.043, rel=0.001)
        assert methane.hhv_btu_scf == pytest.approx(1010.0, rel=0.01)
        assert methane.specific_gravity == pytest.approx(0.5539, rel=0.01)

    def test_ethane_properties(self):
        """Test ethane reference properties."""
        ethane = GAS_COMPONENTS["ethane"]

        assert ethane.name == "Ethane"
        assert ethane.molecular_weight == pytest.approx(30.07, rel=0.001)
        assert ethane.hhv_btu_scf == pytest.approx(1769.7, rel=0.01)

    def test_propane_properties(self):
        """Test propane reference properties."""
        propane = GAS_COMPONENTS["propane"]

        assert propane.name == "Propane"
        assert propane.hhv_btu_scf == pytest.approx(2516.1, rel=0.01)

    def test_hydrogen_properties(self):
        """Test hydrogen reference properties."""
        hydrogen = GAS_COMPONENTS["hydrogen"]

        assert hydrogen.name == "Hydrogen"
        assert hydrogen.molecular_weight == pytest.approx(2.016, rel=0.001)
        assert hydrogen.hhv_btu_scf == pytest.approx(324.2, rel=0.01)
        assert hydrogen.specific_gravity == pytest.approx(0.0696, rel=0.01)

    def test_inert_components_zero_hhv(self):
        """Test inert components have zero HHV."""
        nitrogen = GAS_COMPONENTS["nitrogen"]
        co2 = GAS_COMPONENTS["carbon_dioxide"]

        assert nitrogen.hhv_btu_scf == 0.0
        assert co2.hhv_btu_scf == 0.0

    def test_all_components_have_required_properties(self):
        """Test all components have required properties."""
        for name, component in GAS_COMPONENTS.items():
            assert component.name is not None
            assert component.molecular_weight > 0
            assert component.hhv_btu_scf >= 0
            assert component.lhv_btu_scf >= 0
            assert component.specific_gravity > 0


class TestLiquidFuelConstants:
    """Tests for liquid fuel constant values."""

    def test_fuel_oil_properties(self):
        """Test fuel oil reference properties."""
        no2 = LIQUID_FUEL_PROPERTIES["no2_fuel_oil"]

        assert no2["hhv_btu_lb"] == 19580
        assert no2["lhv_btu_lb"] == 18410
        assert no2["density_lb_gal"] == pytest.approx(7.21, rel=0.01)

    def test_all_liquid_fuels_have_properties(self):
        """Test all liquid fuels have required properties."""
        for name, props in LIQUID_FUEL_PROPERTIES.items():
            assert props["hhv_btu_lb"] > 0
            assert props["lhv_btu_lb"] > 0
            assert props["density_lb_gal"] > 0


class TestSolidFuelConstants:
    """Tests for solid fuel constant values."""

    def test_coal_properties(self):
        """Test coal reference properties."""
        bituminous = SOLID_FUEL_PROPERTIES["coal_bituminous"]

        assert bituminous["hhv_btu_lb"] == 12500
        assert bituminous["moisture_pct"] == 6.0
        assert bituminous["ash_pct"] == 8.0

    def test_all_solid_fuels_have_properties(self):
        """Test all solid fuels have required properties."""
        for name, props in SOLID_FUEL_PROPERTIES.items():
            assert props["hhv_btu_lb"] > 0
            assert props["lhv_btu_lb"] > 0
            assert "moisture_pct" in props
            assert "ash_pct" in props


class TestResultMetadata:
    """Tests for result metadata and provenance."""

    def test_calculation_method_field(self, heating_value_calculator):
        """Test calculation method is recorded."""
        input_data = HeatingValueInput(fuel_type="test", methane_pct=95.0)
        result = heating_value_calculator.calculate_gas_heating_value(input_data)

        assert result.calculation_method == "ASTM_D3588"

    def test_reference_conditions_field(self, heating_value_calculator):
        """Test reference conditions are recorded."""
        input_data = HeatingValueInput(fuel_type="test", methane_pct=95.0)
        result = heating_value_calculator.calculate_gas_heating_value(input_data)

        assert "60F" in result.reference_conditions
        assert "14.696 psia" in result.reference_conditions

    def test_timestamp_field(self, heating_value_calculator):
        """Test timestamp is recorded."""
        input_data = HeatingValueInput(fuel_type="test", methane_pct=95.0)
        result = heating_value_calculator.calculate_gas_heating_value(input_data)

        assert result.timestamp is not None
        assert result.timestamp.tzinfo is not None

    def test_provenance_hash_is_sha256(self, heating_value_calculator):
        """Test provenance hash is SHA-256 format."""
        input_data = HeatingValueInput(fuel_type="test", methane_pct=95.0)
        result = heating_value_calculator.calculate_gas_heating_value(input_data)

        assert len(result.provenance_hash) == 64  # SHA-256 hex length
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)
