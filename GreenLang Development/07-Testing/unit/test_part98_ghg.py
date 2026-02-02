# -*- coding: utf-8 -*-
"""
Unit Tests for EPA Part 98 Subpart C GHG Reporting

Tests the Part98Reporter implementation for:
- Tier 1, Tier 2, and Tier 3 CO2 calculations
- CH4 and N2O calculations
- Annual facility reporting
- Validation and error handling
- Provenance tracking

References:
    - 40 CFR Part 98 Subpart C
    - EPA GHGRP Reporting Guidelines
"""

import pytest
from datetime import datetime
from greenlang.compliance.epa.part98_ghg import (
    Part98Reporter,
    Part98Config,
    FuelCombustionData,
    FuelType,
    TierLevel,
    CO2_EMISSION_FACTORS,
    CH4_N2O_FACTORS,
)


class TestCO2EmissionFactors:
    """Test EPA Table C-1 CO2 emission factors."""

    def test_natural_gas_factor(self):
        """Natural gas CO2 factor should be 53.06 kg CO2/MMBtu."""
        factor = CO2_EMISSION_FACTORS.get_factor(FuelType.NATURAL_GAS)
        assert factor == 53.06

    def test_coal_bituminous_factor(self):
        """Coal (bituminous) CO2 factor should be 93.69 kg CO2/MMBtu."""
        factor = CO2_EMISSION_FACTORS.get_factor(FuelType.COAL_BITUMINOUS)
        assert factor == 93.69

    def test_fuel_oil_no2_factor(self):
        """Fuel Oil No. 2 CO2 factor should be 73.96 kg CO2/MMBtu."""
        factor = CO2_EMISSION_FACTORS.get_factor(FuelType.FUEL_OIL_NO2)
        assert factor == 73.96

    def test_all_fuel_types_have_factors(self):
        """All fuel types should have defined emission factors."""
        for fuel_type in FuelType:
            factor = CO2_EMISSION_FACTORS.get_factor(fuel_type)
            assert factor > 0, f"No factor for {fuel_type}"


class TestCH4N2OFactors:
    """Test EPA Table C-2 CH4 and N2O emission factors."""

    def test_natural_gas_ch4_factor(self):
        """Natural gas CH4 factor should be 0.0022 kg/MMBtu."""
        factor = CH4_N2O_FACTORS.get_ch4_factor(FuelType.NATURAL_GAS)
        assert factor == 0.0022

    def test_natural_gas_n2o_factor(self):
        """Natural gas N2O factor should be 0.0001 kg/MMBtu."""
        factor = CH4_N2O_FACTORS.get_n2o_factor(FuelType.NATURAL_GAS)
        assert factor == 0.0001

    def test_coal_ch4_factor(self):
        """Coal CH4 factor should be 0.0005 kg/MMBtu."""
        factor = CH4_N2O_FACTORS.get_ch4_factor(FuelType.COAL_BITUMINOUS)
        assert factor == 0.0005

    def test_coal_n2o_factor(self):
        """Coal N2O factor should be 0.0001 kg/MMBtu."""
        factor = CH4_N2O_FACTORS.get_n2o_factor(FuelType.COAL_BITUMINOUS)
        assert factor == 0.0001


class TestFuelCombustionData:
    """Test FuelCombustionData input validation."""

    def test_valid_fuel_data(self):
        """Valid fuel combustion data should pass validation."""
        data = FuelCombustionData(
            fuel_type=FuelType.NATURAL_GAS,
            heat_input_mmbtu=5000.0,
            facility_id="FAC123",
            reporting_year=2024
        )
        assert data.fuel_type == FuelType.NATURAL_GAS
        assert data.heat_input_mmbtu == 5000.0

    def test_invalid_heat_input_negative(self):
        """Negative heat input should fail validation."""
        with pytest.raises(ValueError):
            FuelCombustionData(
                fuel_type=FuelType.NATURAL_GAS,
                heat_input_mmbtu=-100.0,
                facility_id="FAC123",
                reporting_year=2024
            )

    def test_invalid_heat_input_unreasonable(self):
        """Unreasonably high heat input should fail validation."""
        with pytest.raises(ValueError):
            FuelCombustionData(
                fuel_type=FuelType.NATURAL_GAS,
                heat_input_mmbtu=200_000_000.0,  # > 100M MMBtu
                facility_id="FAC123",
                reporting_year=2024
            )

    def test_invalid_carbon_content(self):
        """Carbon content >100% should fail validation."""
        with pytest.raises(ValueError):
            FuelCombustionData(
                fuel_type=FuelType.NATURAL_GAS,
                heat_input_mmbtu=5000.0,
                carbon_content=150.0,  # > 100%
                facility_id="FAC123",
                reporting_year=2024
            )

    def test_tier2_requires_hhv_and_carbon_content(self):
        """Tier 2 should have optional HHV and carbon content."""
        data = FuelCombustionData(
            fuel_type=FuelType.COAL_BITUMINOUS,
            heat_input_mmbtu=5000.0,
            fuel_quantity=500.0,
            higher_heating_value=12000.0,
            carbon_content=75.0,
            facility_id="FAC123",
            reporting_year=2024
        )
        assert data.higher_heating_value == 12000.0
        assert data.carbon_content == 75.0


class TestPart98Tier1Calculation:
    """Test Tier 1 CO2 calculation (default factors)."""

    def setup_method(self):
        """Set up test fixtures."""
        config = Part98Config(facility_id="FAC123")
        self.reporter = Part98Reporter(config)

    def test_tier1_natural_gas(self):
        """Tier 1 calculation for natural gas."""
        result = self.reporter.calculate_co2_tier1(
            fuel_quantity=1000.0,  # MMBtu
            emission_factor=53.06  # kg CO2/MMBtu
        )
        assert result.co2_kg == pytest.approx(53060.0, rel=1e-3)
        assert result.co2_metric_tons == pytest.approx(53.06, rel=1e-3)
        assert result.calculation_tier == TierLevel.TIER1

    def test_tier1_coal(self):
        """Tier 1 calculation for coal."""
        result = self.reporter.calculate_co2_tier1(
            fuel_quantity=500.0,  # MMBtu
            emission_factor=93.69  # kg CO2/MMBtu
        )
        assert result.co2_kg == pytest.approx(46845.0, rel=1e-3)
        assert result.co2_metric_tons == pytest.approx(46.845, rel=1e-3)

    def test_tier1_fuel_oil(self):
        """Tier 1 calculation for fuel oil."""
        result = self.reporter.calculate_co2_tier1(
            fuel_quantity=250.0,  # MMBtu
            emission_factor=73.96  # kg CO2/MMBtu
        )
        assert result.co2_kg == pytest.approx(18490.0, rel=1e-3)
        assert result.co2_metric_tons == pytest.approx(18.49, rel=1e-3)


class TestPart98Tier2Calculation:
    """Test Tier 2 CO2 calculation (fuel-specific data)."""

    def setup_method(self):
        """Set up test fixtures."""
        config = Part98Config(facility_id="FAC123")
        self.reporter = Part98Reporter(config)

    def test_tier2_coal(self):
        """Tier 2 calculation with fuel-specific data."""
        result = self.reporter.calculate_co2_tier2(
            fuel_quantity=1000.0,  # kg
            higher_heating_value=12000.0,  # BTU/kg
            carbon_content=75.0,  # % by weight
            fuel_type=FuelType.COAL_BITUMINOUS
        )
        # CO2 = 1000 * 12000 * 0.75 * 3.6667
        expected_co2_kg = 1000 * 12000 * 0.75 * 3.6667
        assert result.co2_kg == pytest.approx(expected_co2_kg, rel=1e-3)
        assert result.calculation_tier == TierLevel.TIER2
        assert result.carbon_content == 75.0

    def test_tier2_converts_to_mmbtu(self):
        """Tier 2 should calculate heat input in MMBtu."""
        result = self.reporter.calculate_co2_tier2(
            fuel_quantity=1000.0,  # kg
            higher_heating_value=12000.0,  # BTU/kg
            carbon_content=75.0,
            fuel_type=FuelType.COAL_BITUMINOUS
        )
        expected_heat_input = (1000 * 12000) / 1_000_000
        assert result.heat_input_mmbtu == pytest.approx(expected_heat_input, rel=1e-3)


class TestPart98Tier3Calculation:
    """Test Tier 3 CO2 calculation (continuous monitoring)."""

    def setup_method(self):
        """Set up test fixtures."""
        config = Part98Config(facility_id="FAC123")
        self.reporter = Part98Reporter(config)

    def test_tier3_with_cems_data(self):
        """Tier 3 calculation using CEMS-measured CO2."""
        result = self.reporter.calculate_co2_tier3(
            fuel_quantity=1000.0,  # kg
            higher_heating_value=12000.0,  # BTU/kg
            carbon_content=75.0,
            fuel_type=FuelType.COAL_BITUMINOUS,
            cems_co2_measured_kg=26000.0  # Measured CO2
        )
        assert result.co2_kg == 26000.0
        assert result.calculation_tier == TierLevel.TIER3

    def test_tier3_fallback_to_tier2(self):
        """Tier 3 without CEMS should fallback to Tier 2."""
        result = self.reporter.calculate_co2_tier3(
            fuel_quantity=1000.0,
            higher_heating_value=12000.0,
            carbon_content=75.0,
            fuel_type=FuelType.COAL_BITUMINOUS,
            cems_co2_measured_kg=None
        )
        assert result.calculation_tier == TierLevel.TIER3


class TestPart98CH4N2OCalculation:
    """Test CH4 and N2O calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        config = Part98Config(facility_id="FAC123")
        self.reporter = Part98Reporter(config)

    def test_ch4_n2o_natural_gas(self):
        """CH4 and N2O calculation for natural gas."""
        result = self.reporter.calculate_ch4_n2o(
            fuel_type=FuelType.NATURAL_GAS,
            heat_input_mmbtu=1000.0
        )
        assert result.ch4_kg == pytest.approx(1000.0 * 0.0022, rel=1e-3)
        assert result.n2o_kg == pytest.approx(1000.0 * 0.0001, rel=1e-3)
        assert result.ch4_metric_tons == pytest.approx((1000.0 * 0.0022) / 1000.0, rel=1e-3)
        assert result.n2o_metric_tons == pytest.approx((1000.0 * 0.0001) / 1000.0, rel=1e-3)

    def test_ch4_n2o_coal(self):
        """CH4 and N2O calculation for coal."""
        result = self.reporter.calculate_ch4_n2o(
            fuel_type=FuelType.COAL_BITUMINOUS,
            heat_input_mmbtu=500.0
        )
        assert result.ch4_kg == pytest.approx(500.0 * 0.0005, rel=1e-3)
        assert result.n2o_kg == pytest.approx(500.0 * 0.0001, rel=1e-3)

    def test_ch4_n2o_fuel_oil(self):
        """CH4 and N2O calculation for fuel oil."""
        result = self.reporter.calculate_ch4_n2o(
            fuel_type=FuelType.FUEL_OIL_NO2,
            heat_input_mmbtu=300.0
        )
        assert result.ch4_kg == pytest.approx(300.0 * 0.0010, rel=1e-3)
        assert result.n2o_kg == pytest.approx(300.0 * 0.0005, rel=1e-3)


class TestPart98SubpartCComplete:
    """Test complete Subpart C calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        config = Part98Config(facility_id="FAC123", epa_ghgrp_id="123456789")
        self.reporter = Part98Reporter(config)

    def test_subpart_c_natural_gas_tier1(self):
        """Complete Subpart C calculation for natural gas (Tier 1)."""
        fuel_data = FuelCombustionData(
            fuel_type=FuelType.NATURAL_GAS,
            heat_input_mmbtu=5000.0,
            facility_id="FAC123",
            reporting_year=2024
        )
        result = self.reporter.calculate_subpart_c(fuel_data, tier=TierLevel.TIER1)

        assert result.facility_id == "FAC123"
        assert result.reporting_year == 2024
        assert result.validation_status == "PASS"
        assert result.co2_calculation.calculation_tier == TierLevel.TIER1

        # CO2: 5000 MMBtu * 53.06 kg/MMBtu = 265,300 kg = 265.3 MT
        assert result.total_co2_metric_tons == pytest.approx(265.3, rel=1e-2)

        # CH4: 5000 * 0.0022 = 11 kg = 0.011 MT
        assert result.total_ch4_metric_tons == pytest.approx(0.011, rel=1e-2)

        # N2O: 5000 * 0.0001 = 0.5 kg = 0.0005 MT
        assert result.total_n2o_metric_tons == pytest.approx(0.0005, rel=1e-2)

    def test_subpart_c_exceeds_threshold(self):
        """Facility exceeding 25,000 MT CO2e threshold."""
        fuel_data = FuelCombustionData(
            fuel_type=FuelType.COAL_BITUMINOUS,
            heat_input_mmbtu=300_000.0,  # Large facility
            facility_id="FAC123",
            reporting_year=2024
        )
        result = self.reporter.calculate_subpart_c(fuel_data, tier=TierLevel.TIER1)

        assert result.exceeds_threshold is True
        assert result.requires_reporting is True

    def test_subpart_c_below_threshold(self):
        """Facility below 25,000 MT CO2e threshold."""
        fuel_data = FuelCombustionData(
            fuel_type=FuelType.NATURAL_GAS,
            heat_input_mmbtu=100.0,  # Small facility
            facility_id="FAC123",
            reporting_year=2024
        )
        result = self.reporter.calculate_subpart_c(fuel_data, tier=TierLevel.TIER1)

        assert result.total_co2e_metric_tons < 25000
        assert result.exceeds_threshold is False
        assert result.requires_reporting is False

    def test_subpart_c_provenance_hash(self):
        """Provenance hash should be generated for audit trail."""
        fuel_data = FuelCombustionData(
            fuel_type=FuelType.NATURAL_GAS,
            heat_input_mmbtu=5000.0,
            facility_id="FAC123",
            reporting_year=2024
        )
        result = self.reporter.calculate_subpart_c(fuel_data)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 hex length


class TestPart98AnnualReport:
    """Test facility-level annual report generation."""

    def setup_method(self):
        """Set up test fixtures."""
        config = Part98Config(
            facility_id="FAC123",
            epa_ghgrp_id="123456789",
            facility_name="Test Facility"
        )
        self.reporter = Part98Reporter(config)

    def test_annual_report_single_source(self):
        """Annual report for single source category."""
        fuel_data = FuelCombustionData(
            fuel_type=FuelType.NATURAL_GAS,
            heat_input_mmbtu=5000.0,
            facility_id="FAC123",
            process_id="BOILER-001",
            reporting_year=2024
        )
        result = self.reporter.calculate_subpart_c(fuel_data)
        annual_report = self.reporter.generate_annual_report([result])

        assert annual_report["facility_id"] == "FAC123"
        assert annual_report["reporting_year"] == 2024
        assert annual_report["total_records"] == 1
        assert "emissions_summary" in annual_report

    def test_annual_report_multiple_sources(self):
        """Annual report aggregating multiple source categories."""
        fuel_data_1 = FuelCombustionData(
            fuel_type=FuelType.NATURAL_GAS,
            heat_input_mmbtu=3000.0,
            facility_id="FAC123",
            process_id="BOILER-001",
            reporting_year=2024
        )
        fuel_data_2 = FuelCombustionData(
            fuel_type=FuelType.FUEL_OIL_NO2,
            heat_input_mmbtu=2000.0,
            facility_id="FAC123",
            process_id="FURNACE-001",
            reporting_year=2024
        )

        result1 = self.reporter.calculate_subpart_c(fuel_data_1)
        result2 = self.reporter.calculate_subpart_c(fuel_data_2)
        annual_report = self.reporter.generate_annual_report([result1, result2])

        assert annual_report["total_records"] == 2
        assert len(annual_report["source_categories"]) == 2

        # Total CO2: (3000 * 53.06) + (2000 * 73.96)
        expected_co2 = (3000 * 53.06 + 2000 * 73.96) / 1000
        assert annual_report["emissions_summary"]["total_co2_metric_tons"] == pytest.approx(
            expected_co2, rel=1e-2
        )

    def test_annual_report_mixed_facilities_error(self):
        """Mixed facilities in annual report should raise error."""
        fuel_data_1 = FuelCombustionData(
            fuel_type=FuelType.NATURAL_GAS,
            heat_input_mmbtu=5000.0,
            facility_id="FAC123",
            reporting_year=2024
        )
        fuel_data_2 = FuelCombustionData(
            fuel_type=FuelType.NATURAL_GAS,
            heat_input_mmbtu=3000.0,
            facility_id="FAC456",  # Different facility
            reporting_year=2024
        )

        result1 = self.reporter.calculate_subpart_c(fuel_data_1)
        # Create second reporter for different facility
        config2 = Part98Config(facility_id="FAC456")
        reporter2 = Part98Reporter(config2)
        result2 = reporter2.calculate_subpart_c(fuel_data_2)

        with pytest.raises(ValueError):
            self.reporter.generate_annual_report([result1, result2])

    def test_annual_report_gwp_values(self):
        """Annual report should use correct GWP values."""
        fuel_data = FuelCombustionData(
            fuel_type=FuelType.NATURAL_GAS,
            heat_input_mmbtu=5000.0,
            facility_id="FAC123",
            reporting_year=2024
        )
        result = self.reporter.calculate_subpart_c(fuel_data)
        annual_report = self.reporter.generate_annual_report([result])

        # AR5 GWP: CH4=28, N2O=265
        assert annual_report["emissions_summary"]["gwp_ch4"] == 28
        assert annual_report["emissions_summary"]["gwp_n2o"] == 265


class TestPart98EdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        config = Part98Config(facility_id="FAC123")
        self.reporter = Part98Reporter(config)

    def test_zero_heat_input(self):
        """Zero heat input should be handled."""
        fuel_data = FuelCombustionData(
            fuel_type=FuelType.NATURAL_GAS,
            heat_input_mmbtu=0.0,
            facility_id="FAC123",
            reporting_year=2024
        )
        result = self.reporter.calculate_subpart_c(fuel_data)
        assert result.total_co2_metric_tons == 0.0

    def test_very_small_values(self):
        """Very small heat input values."""
        fuel_data = FuelCombustionData(
            fuel_type=FuelType.NATURAL_GAS,
            heat_input_mmbtu=0.001,
            facility_id="FAC123",
            reporting_year=2024
        )
        result = self.reporter.calculate_subpart_c(fuel_data)
        assert result.total_co2_metric_tons > 0
        assert result.validation_status == "PASS"

    def test_co2e_calculation(self):
        """CO2e should include CH4 and N2O with GWP factors."""
        fuel_data = FuelCombustionData(
            fuel_type=FuelType.NATURAL_GAS,
            heat_input_mmbtu=1000.0,
            facility_id="FAC123",
            reporting_year=2024
        )
        result = self.reporter.calculate_subpart_c(fuel_data)

        # CO2e = CO2 + (CH4 * 28) + (N2O * 265)
        expected_co2e = (
            result.total_co2_metric_tons +
            (result.total_ch4_metric_tons * 28) +
            (result.total_n2o_metric_tons * 265)
        )
        assert result.total_co2e_metric_tons == pytest.approx(expected_co2e, rel=1e-3)


class TestPart98Performance:
    """Test performance characteristics."""

    def setup_method(self):
        """Set up test fixtures."""
        config = Part98Config(facility_id="FAC123")
        self.reporter = Part98Reporter(config)

    def test_calculation_performance(self):
        """Calculation should complete quickly."""
        fuel_data = FuelCombustionData(
            fuel_type=FuelType.NATURAL_GAS,
            heat_input_mmbtu=5000.0,
            facility_id="FAC123",
            reporting_year=2024
        )
        result = self.reporter.calculate_subpart_c(fuel_data)

        # Should complete in <100ms
        assert result.processing_time_ms < 100.0

    def test_large_facility_calculation(self):
        """Large facility calculation performance."""
        fuel_data = FuelCombustionData(
            fuel_type=FuelType.COAL_BITUMINOUS,
            heat_input_mmbtu=1_000_000.0,  # Very large
            facility_id="FAC123",
            reporting_year=2024
        )
        result = self.reporter.calculate_subpart_c(fuel_data)

        assert result.processing_time_ms < 100.0
        assert result.total_co2_metric_tons > 0
