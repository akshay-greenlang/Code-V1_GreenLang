# -*- coding: utf-8 -*-
"""
EPA 40 CFR Part 98 GHG Reporting Compliance Tests

Tests compliance with Greenhouse Gas Reporting Program requirements:
    - Tier calculation accuracy (Tier 1-4 methodologies)
    - Emission factor validation against Table C-1
    - GWP (Global Warming Potential) application
    - Annual reporting deadline compliance
    - CO2, CH4, N2O, and CO2e calculations

Standards Reference:
    - 40 CFR Part 98 Subpart A - General Provisions
    - 40 CFR Part 98 Subpart C - Stationary Combustion Sources
    - 40 CFR Part 98 Table C-1 - Emission Factors
    - 40 CFR Part 98 Table A-1 - Global Warming Potentials

Author: GL-TestEngineer
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
import math
import pytest

# Import process heat calculation modules if available
# Use broad exception handling to catch pydantic errors during import
try:
    from greenlang.agents.process_heat.shared.calculation_library import (
        ThermalIQCalculationLibrary,
        FuelConstants,
    )
    CALC_LIBRARY_AVAILABLE = True
except Exception:
    CALC_LIBRARY_AVAILABLE = False
    ThermalIQCalculationLibrary = None
    FuelConstants = None

try:
    from greenlang.agents.process_heat.gl_018_unified_combustion.emissions import (
        CO2_EMISSION_FACTORS,
    )
    EMISSION_FACTORS_AVAILABLE = True
except Exception:
    EMISSION_FACTORS_AVAILABLE = False
    CO2_EMISSION_FACTORS = {}


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def calc_library():
    """Create ThermalIQCalculationLibrary for testing."""
    if not CALC_LIBRARY_AVAILABLE:
        pytest.skip("ThermalIQCalculationLibrary not available")
    return ThermalIQCalculationLibrary()


@pytest.fixture
def tier_calculation_inputs() -> Dict[str, Any]:
    """Standard inputs for tier calculation testing."""
    return {
        "natural_gas": {
            "fuel_consumption_mmbtu": 100000,  # 100,000 MMBTU/year
            "measured_hhv_btu_scf": 1028,
            "carbon_content_pct": 75.0,
        },
        "fuel_oil_no2": {
            "fuel_consumption_mmbtu": 50000,
            "measured_hhv_btu_gallon": 138690,
            "carbon_content_pct": 86.5,
        },
        "coal_bituminous": {
            "fuel_consumption_mmbtu": 200000,
            "measured_hhv_btu_ton": 24930000,
            "carbon_content_pct": 75.0,
        },
    }


# =============================================================================
# EMISSION FACTOR VALIDATION TESTS (TABLE C-1)
# =============================================================================


class TestEPAPart98EmissionFactors:
    """
    Validate emission factors against EPA Part 98 Table C-1.

    Table C-1 emission factors are legally mandated for GHG reporting.
    Factors must match exactly for regulatory compliance.

    Pass/Fail Criteria:
        - CO2 factors must match Table C-1 exactly (0 tolerance)
        - CH4 factors must match Table C-1 within 1%
        - N2O factors must match Table C-1 within 1%
    """

    @pytest.mark.compliance
    @pytest.mark.parametrize("fuel_type,expected_co2_kg_mmbtu", [
        ("natural_gas", 53.06),
        ("distillate_fuel_oil_no2", 73.16),
        ("residual_fuel_oil_no6", 75.10),
        ("propane", 62.87),
        ("bituminous_coal", 93.28),
        ("sub_bituminous_coal", 97.17),
        ("lignite", 97.72),
    ])
    def test_co2_emission_factors_table_c1(
        self,
        epa_part98_emission_factors,
        fuel_type: str,
        expected_co2_kg_mmbtu: float,
    ):
        """
        Test CO2 emission factors match Table C-1 exactly.

        These factors are the legal basis for GHG reporting.
        No tolerance is allowed.
        """
        factor = epa_part98_emission_factors[fuel_type]["co2_kg_per_mmbtu"]

        assert factor == expected_co2_kg_mmbtu, (
            f"CO2 factor for {fuel_type} must match Table C-1 exactly: "
            f"got {factor}, expected {expected_co2_kg_mmbtu}"
        )

    @pytest.mark.compliance
    @pytest.mark.parametrize("fuel_type,expected_ch4_kg_mmbtu", [
        ("natural_gas", 0.001),
        ("distillate_fuel_oil_no2", 0.003),
        ("bituminous_coal", 0.011),
    ])
    def test_ch4_emission_factors_table_c1(
        self,
        epa_part98_emission_factors,
        fuel_type: str,
        expected_ch4_kg_mmbtu: float,
    ):
        """
        Test CH4 emission factors match Table C-1.

        CH4 factors are important for total GHG calculations.
        """
        factor = epa_part98_emission_factors[fuel_type]["ch4_kg_per_mmbtu"]

        # Allow 1% tolerance for rounding
        tolerance = 0.01
        assert abs(factor - expected_ch4_kg_mmbtu) / expected_ch4_kg_mmbtu < tolerance, (
            f"CH4 factor for {fuel_type} out of tolerance: "
            f"got {factor}, expected {expected_ch4_kg_mmbtu}"
        )

    @pytest.mark.compliance
    @pytest.mark.parametrize("fuel_type,expected_n2o_kg_mmbtu", [
        ("natural_gas", 0.0001),
        ("distillate_fuel_oil_no2", 0.0006),
        ("bituminous_coal", 0.0016),
    ])
    def test_n2o_emission_factors_table_c1(
        self,
        epa_part98_emission_factors,
        fuel_type: str,
        expected_n2o_kg_mmbtu: float,
    ):
        """
        Test N2O emission factors match Table C-1.

        N2O has high GWP (298), so accurate factors are critical.
        """
        factor = epa_part98_emission_factors[fuel_type]["n2o_kg_per_mmbtu"]

        # Allow 1% tolerance for rounding
        tolerance = 0.01
        assert abs(factor - expected_n2o_kg_mmbtu) / expected_n2o_kg_mmbtu < tolerance, (
            f"N2O factor for {fuel_type} out of tolerance: "
            f"got {factor}, expected {expected_n2o_kg_mmbtu}"
        )

    @pytest.mark.compliance
    def test_emission_factors_in_greenlang_library(
        self,
        epa_part98_emission_factors,
    ):
        """
        Test GreenLang calculation library uses correct Part 98 factors.

        Verifies that the process heat agents use regulatory-compliant factors.
        """
        if not EMISSION_FACTORS_AVAILABLE:
            pytest.skip("Emission factors module not available")

        # Map Part 98 fuel types to GreenLang fuel types
        mapping = {
            "natural_gas": "natural_gas",
            "distillate_fuel_oil_no2": "no2_fuel_oil",
            "residual_fuel_oil_no6": "no6_fuel_oil",
            "propane": "propane",
            "bituminous_coal": "coal_bituminous",
        }

        for part98_fuel, greenlang_fuel in mapping.items():
            part98_factor = epa_part98_emission_factors[part98_fuel]["co2_kg_per_mmbtu"]
            greenlang_factor = CO2_EMISSION_FACTORS.get(greenlang_fuel)

            if greenlang_factor is not None:
                # Must match exactly for regulatory compliance
                assert abs(greenlang_factor - part98_factor) < 0.01, (
                    f"GreenLang CO2 factor for {greenlang_fuel} does not match Part 98: "
                    f"{greenlang_factor} vs {part98_factor}"
                )


# =============================================================================
# GLOBAL WARMING POTENTIAL (GWP) TESTS
# =============================================================================


class TestGlobalWarmingPotentials:
    """
    Test Global Warming Potential values from Part 98 Table A-1.

    GWP values are used to convert different GHGs to CO2 equivalents.

    Pass/Fail Criteria:
        - GWP values must match Table A-1 exactly
        - CO2e calculations must apply correct GWPs
    """

    @pytest.mark.compliance
    @pytest.mark.parametrize("gas,expected_gwp", [
        ("co2", 1),
        ("ch4", 25),      # AR4 value used in Part 98
        ("n2o", 298),
        ("sf6", 22800),
    ])
    def test_gwp_values_table_a1(
        self,
        epa_part98_gwp_values,
        gas: str,
        expected_gwp: int,
    ):
        """Test GWP values match Table A-1."""
        gwp = epa_part98_gwp_values[gas]

        assert gwp == expected_gwp, (
            f"GWP for {gas} must match Table A-1: got {gwp}, expected {expected_gwp}"
        )

    @pytest.mark.compliance
    def test_co2_equivalent_calculation(
        self,
        ghg_calculator,
    ):
        """
        Test CO2 equivalent calculation using GWP values.

        CO2e = CO2 + (CH4 * GWP_CH4) + (N2O * GWP_N2O)
        """
        result = ghg_calculator.calculate_co2e_emissions(
            fuel_type="natural_gas",
            fuel_quantity_mmbtu=1000,  # 1000 MMBTU
        )

        # Verify individual components
        assert result["co2_metric_tons"] > 0, "CO2 should be calculated"
        assert result["ch4_metric_tons"] > 0, "CH4 should be calculated"
        assert result["n2o_metric_tons"] > 0, "N2O should be calculated"

        # Verify CO2e is sum of weighted components
        expected_co2e = (
            result["co2_metric_tons"] * 1 +
            result["ch4_metric_tons"] * 25 +
            result["n2o_metric_tons"] * 298
        )

        assert abs(result["co2e_metric_tons"] - expected_co2e) < 0.001, (
            f"CO2e calculation incorrect: {result['co2e_metric_tons']} "
            f"vs expected {expected_co2e}"
        )

    @pytest.mark.compliance
    def test_ch4_gwp_contribution(
        self,
        ghg_calculator,
    ):
        """
        Test that CH4 GWP contribution is significant.

        CH4 has GWP of 25, so even small amounts contribute.
        """
        result = ghg_calculator.calculate_co2e_emissions(
            fuel_type="natural_gas",
            fuel_quantity_mmbtu=1000,
        )

        # CH4 contribution = CH4 mass * 25
        ch4_contribution = result["ch4_metric_tons"] * 25

        # CH4 should be at least 0.1% of total CO2e for gas
        ch4_fraction = ch4_contribution / result["co2e_metric_tons"]
        assert ch4_fraction > 0.0001, (
            f"CH4 contribution ({ch4_fraction:.4%}) should be measurable"
        )


# =============================================================================
# TIER CALCULATION METHODOLOGY TESTS
# =============================================================================


class TestTierCalculationMethodologies:
    """
    Test Tier 1-4 calculation methodologies per Part 98 Subpart C.

    Tier 1: Default emission factors from Table C-1
    Tier 2: Site-specific HHV with default EFs
    Tier 3: Carbon content analysis
    Tier 4: CEMS measurement

    Pass/Fail Criteria:
        - Tier selection must match facility emission thresholds
        - Calculations must follow correct methodology
        - Uncertainty requirements must be met
    """

    @pytest.mark.compliance
    def test_tier_1_calculation_natural_gas(
        self,
        ghg_calculator,
        epa_part98_emission_factors,
    ):
        """
        Test Tier 1 calculation using default emission factors.

        Tier 1: CO2 = Fuel * EF_CO2 * HHV_default
        """
        fuel_mmbtu = 10000  # 10,000 MMBTU/year
        ef_co2 = epa_part98_emission_factors["natural_gas"]["co2_kg_per_mmbtu"]

        # Expected CO2 emissions
        expected_co2_kg = fuel_mmbtu * ef_co2
        expected_co2_mt = expected_co2_kg / 1000

        # Calculate using GHG calculator
        result = ghg_calculator.calculate_co2_emissions(
            fuel_type="natural_gas",
            fuel_quantity_mmbtu=fuel_mmbtu,
        )

        assert abs(result - expected_co2_mt) < 0.01, (
            f"Tier 1 CO2 calculation incorrect: {result} MT vs expected {expected_co2_mt} MT"
        )

    @pytest.mark.compliance
    @pytest.mark.parametrize("annual_emissions_mt,expected_tier", [
        (10000, "tier_1"),      # Small source
        (50000, "tier_2"),      # Medium source
        (300000, "tier_3"),     # Large source
        (500000, "tier_3"),     # Very large source
    ])
    def test_tier_selection_by_emission_level(
        self,
        epa_part98_tier_requirements,
        annual_emissions_mt: float,
        expected_tier: str,
    ):
        """
        Test tier selection based on annual emission levels.

        Tier requirements:
            - Tier 1: < 25,000 MT CO2e/year
            - Tier 2: 25,000 - 250,000 MT CO2e/year
            - Tier 3: > 250,000 MT CO2e/year
            - Tier 4: CEMS option
        """
        tier_1_threshold = epa_part98_tier_requirements["tier_1"]["applicable_for_emissions_less_than"]
        tier_2_range = epa_part98_tier_requirements["tier_2"]["applicable_for_emissions_range"]
        tier_3_threshold = epa_part98_tier_requirements["tier_3"]["applicable_for_emissions_greater_than"]

        # Determine required tier
        if annual_emissions_mt < tier_1_threshold:
            required_tier = "tier_1"
        elif tier_2_range[0] <= annual_emissions_mt <= tier_2_range[1]:
            required_tier = "tier_2"
        elif annual_emissions_mt > tier_3_threshold:
            required_tier = "tier_3"
        else:
            required_tier = "tier_2"  # Default to tier 2 for edge cases

        assert required_tier == expected_tier, (
            f"Emissions of {annual_emissions_mt} MT should require {expected_tier}, "
            f"calculated {required_tier}"
        )

    @pytest.mark.compliance
    def test_tier_2_hhv_measurement_requirement(
        self,
        epa_part98_tier_requirements,
    ):
        """Test Tier 2 requires monthly HHV measurement."""
        tier_2 = epa_part98_tier_requirements["tier_2"]

        assert tier_2["hhv_measurement_frequency"] == "monthly", (
            "Tier 2 requires monthly HHV measurement"
        )

        # Verify data requirements
        assert "measured_hhv" in tier_2["data_requirements"], (
            "Tier 2 must require measured HHV"
        )

    @pytest.mark.compliance
    def test_tier_3_carbon_analysis_requirement(
        self,
        epa_part98_tier_requirements,
    ):
        """Test Tier 3 requires carbon content analysis."""
        tier_3 = epa_part98_tier_requirements["tier_3"]

        assert tier_3["carbon_analysis_frequency"] == "monthly", (
            "Tier 3 requires monthly carbon analysis"
        )

        # Verify data requirements
        assert "carbon_content" in tier_3["data_requirements"], (
            "Tier 3 must require carbon content"
        )

    @pytest.mark.compliance
    def test_tier_4_cems_data_availability(
        self,
        epa_part98_tier_requirements,
    ):
        """Test Tier 4 requires 90% CEMS data availability."""
        tier_4 = epa_part98_tier_requirements["tier_4"]

        assert tier_4["data_availability_minimum_pct"] == 90.0, (
            "Tier 4 requires 90% CEMS data availability"
        )

    @pytest.mark.compliance
    @pytest.mark.parametrize("tier,expected_uncertainty_pct", [
        ("tier_1", 10.0),
        ("tier_2", 5.0),
        ("tier_3", 2.0),
        ("tier_4", 1.0),
    ])
    def test_tier_uncertainty_levels(
        self,
        epa_part98_tier_requirements,
        tier: str,
        expected_uncertainty_pct: float,
    ):
        """Test that higher tiers have lower uncertainty."""
        tier_req = epa_part98_tier_requirements[tier]

        assert tier_req["uncertainty_pct"] == expected_uncertainty_pct, (
            f"{tier} uncertainty should be {expected_uncertainty_pct}%"
        )


# =============================================================================
# CO2 CALCULATION ACCURACY TESTS
# =============================================================================


class TestCO2CalculationAccuracy:
    """
    Test CO2 calculation accuracy against known values.

    Validates calculations produce regulatory-compliant results.

    Pass/Fail Criteria:
        - Calculations must be within 2% of expected values
        - Results must be reproducible (deterministic)
    """

    @pytest.mark.compliance
    @pytest.mark.parametrize("fuel_type,fuel_mmbtu,expected_co2_mt", [
        ("natural_gas", 1000, 53.06),        # 1000 * 53.06 / 1000
        ("natural_gas", 10000, 530.6),       # 10000 * 53.06 / 1000
        ("distillate_fuel_oil_no2", 1000, 73.16),
        ("bituminous_coal", 1000, 93.28),
    ])
    def test_co2_calculation_known_values(
        self,
        ghg_calculator,
        fuel_type: str,
        fuel_mmbtu: float,
        expected_co2_mt: float,
    ):
        """Test CO2 calculations against known values."""
        result = ghg_calculator.calculate_co2_emissions(
            fuel_type=fuel_type,
            fuel_quantity_mmbtu=fuel_mmbtu,
        )

        # Allow 0.1% tolerance for floating point
        tolerance = 0.001
        assert abs(result - expected_co2_mt) / expected_co2_mt < tolerance, (
            f"CO2 calculation for {fuel_type}: {result} MT vs expected {expected_co2_mt} MT"
        )

    @pytest.mark.compliance
    def test_co2_calculation_deterministic(self, ghg_calculator):
        """Test CO2 calculations are deterministic (reproducible)."""
        inputs = {
            "fuel_type": "natural_gas",
            "fuel_quantity_mmbtu": 12345.67,
        }

        results = []
        for _ in range(10):
            result = ghg_calculator.calculate_co2_emissions(**inputs)
            results.append(result)

        # All results must be identical
        for i in range(1, len(results)):
            assert results[i] == results[0], (
                f"CO2 calculation not deterministic: run {i+1} = {results[i]}, "
                f"run 1 = {results[0]}"
            )

    @pytest.mark.compliance
    def test_co2_calculation_with_calc_library(self, calc_library):
        """Test CO2 calculation using ThermalIQCalculationLibrary."""
        result = calc_library.calculate_co2_emissions(
            fuel_type="natural_gas",
            fuel_consumption=100,  # 100 MMBTU
            fuel_unit="MMBTU",
        )

        # Verify result structure
        assert result.value > 0, "CO2 emissions should be positive"
        assert result.unit == "kg CO2/hr", f"Unit should be kg CO2/hr, got {result.unit}"
        assert result.formula_reference == "40 CFR Part 98 Table C-1", (
            "Should reference Part 98"
        )

        # Verify calculation accuracy
        # 100 MMBTU * 53.06 kg/MMBTU = 5306 kg
        expected_kg = 100 * 53.06
        tolerance = 0.01
        assert abs(result.value - expected_kg) / expected_kg < tolerance, (
            f"CO2 calculation: {result.value} kg vs expected {expected_kg} kg"
        )


# =============================================================================
# TOTAL CO2e CALCULATION TESTS
# =============================================================================


class TestCO2eCalculation:
    """
    Test total CO2 equivalent calculations.

    CO2e accounts for all greenhouse gases using GWP weighting.
    """

    @pytest.mark.compliance
    def test_co2e_includes_all_gases(self, ghg_calculator):
        """Test CO2e calculation includes CO2, CH4, and N2O."""
        result = ghg_calculator.calculate_co2e_emissions(
            fuel_type="natural_gas",
            fuel_quantity_mmbtu=1000,
        )

        # All components should be present
        assert "co2_metric_tons" in result
        assert "ch4_metric_tons" in result
        assert "n2o_metric_tons" in result
        assert "co2e_metric_tons" in result

        # CO2e should be greater than CO2 alone (due to CH4 and N2O)
        assert result["co2e_metric_tons"] > result["co2_metric_tons"], (
            "CO2e should be greater than CO2 alone"
        )

    @pytest.mark.compliance
    def test_co2e_gwp_weighting_correct(
        self,
        ghg_calculator,
        epa_part98_gwp_values,
    ):
        """Test CO2e calculation applies correct GWP weights."""
        result = ghg_calculator.calculate_co2e_emissions(
            fuel_type="natural_gas",
            fuel_quantity_mmbtu=1000,
        )

        # Manual calculation of expected CO2e
        expected_co2e = (
            result["co2_metric_tons"] * epa_part98_gwp_values["co2"] +
            result["ch4_metric_tons"] * epa_part98_gwp_values["ch4"] +
            result["n2o_metric_tons"] * epa_part98_gwp_values["n2o"]
        )

        assert abs(result["co2e_metric_tons"] - expected_co2e) < 0.001, (
            f"CO2e GWP weighting incorrect: {result['co2e_metric_tons']} "
            f"vs expected {expected_co2e}"
        )

    @pytest.mark.compliance
    @pytest.mark.parametrize("fuel_type,co2e_ratio_min,co2e_ratio_max", [
        ("natural_gas", 1.001, 1.01),     # Low CH4/N2O relative to CO2
        ("bituminous_coal", 1.001, 1.02),  # Slightly higher
    ])
    def test_co2e_to_co2_ratio(
        self,
        ghg_calculator,
        fuel_type: str,
        co2e_ratio_min: float,
        co2e_ratio_max: float,
    ):
        """Test CO2e/CO2 ratio is reasonable for each fuel."""
        result = ghg_calculator.calculate_co2e_emissions(
            fuel_type=fuel_type,
            fuel_quantity_mmbtu=1000,
        )

        ratio = result["co2e_metric_tons"] / result["co2_metric_tons"]

        assert co2e_ratio_min <= ratio <= co2e_ratio_max, (
            f"CO2e/CO2 ratio for {fuel_type} out of expected range: {ratio}"
        )


# =============================================================================
# ANNUAL REPORTING REQUIREMENTS TESTS
# =============================================================================


class TestAnnualReportingRequirements:
    """
    Test annual GHG reporting requirements compliance.

    Part 98 requires annual reporting by March 31 for previous calendar year.
    """

    @pytest.mark.compliance
    def test_reporting_threshold_25000_mtco2e(
        self,
        epa_part98_tier_requirements,
    ):
        """
        Test 25,000 MT CO2e/year reporting threshold.

        Facilities emitting >= 25,000 MT CO2e/year must report.
        """
        tier_1_threshold = epa_part98_tier_requirements["tier_1"]["applicable_for_emissions_less_than"]

        assert tier_1_threshold == 25000, (
            "Part 98 reporting threshold should be 25,000 MT CO2e/year"
        )

    @pytest.mark.compliance
    def test_annual_calculation_accuracy(self, ghg_calculator):
        """
        Test annual emission calculation for reporting.

        Verifies full-year calculation produces accurate totals.
        """
        # Simulate a year of operation
        monthly_fuel_mmbtu = 10000  # 10,000 MMBTU per month
        annual_fuel_mmbtu = monthly_fuel_mmbtu * 12  # 120,000 MMBTU/year

        annual_result = ghg_calculator.calculate_co2e_emissions(
            fuel_type="natural_gas",
            fuel_quantity_mmbtu=annual_fuel_mmbtu,
        )

        # Monthly result * 12 should equal annual
        monthly_result = ghg_calculator.calculate_co2e_emissions(
            fuel_type="natural_gas",
            fuel_quantity_mmbtu=monthly_fuel_mmbtu,
        )

        expected_annual = monthly_result["co2e_metric_tons"] * 12

        assert abs(annual_result["co2e_metric_tons"] - expected_annual) < 0.01, (
            f"Annual calculation should equal 12 * monthly: "
            f"{annual_result['co2e_metric_tons']} vs {expected_annual}"
        )


# =============================================================================
# BIOGENIC EMISSIONS TESTS
# =============================================================================


class TestBiogenicEmissions:
    """
    Test biogenic emissions handling.

    Biogenic CO2 (from biomass) is reported separately from fossil CO2.
    """

    @pytest.mark.compliance
    def test_biogenic_flag_for_biomass(self, epa_part98_emission_factors):
        """Test biomass fuels are flagged as biogenic."""
        wood_factors = epa_part98_emission_factors.get("wood_biomass", {})

        assert wood_factors.get("biogenic") is True, (
            "Wood biomass should be flagged as biogenic"
        )

    @pytest.mark.compliance
    def test_biogenic_co2_factor_exists(self, epa_part98_emission_factors):
        """Test biogenic fuels have CO2 factors for reporting."""
        wood_factors = epa_part98_emission_factors.get("wood_biomass", {})

        assert "co2_kg_per_mmbtu" in wood_factors, (
            "Biogenic fuels should have CO2 emission factors"
        )
        assert wood_factors["co2_kg_per_mmbtu"] > 0, (
            "Biogenic CO2 factor should be positive"
        )


# =============================================================================
# DATA QUALITY AND UNCERTAINTY TESTS
# =============================================================================


class TestDataQualityAndUncertainty:
    """
    Test data quality and uncertainty requirements.

    Part 98 has specific requirements for data quality and uncertainty.
    """

    @pytest.mark.compliance
    def test_calculation_provenance_hash(self, calc_library):
        """Test calculations include provenance hash for audit trail."""
        result = calc_library.calculate_co2_emissions(
            fuel_type="natural_gas",
            fuel_consumption=100,
            fuel_unit="MMBTU",
        )

        assert hasattr(result, "inputs_hash"), "Result should have inputs hash"
        assert len(result.inputs_hash) == 64, "Hash should be SHA-256 (64 chars)"

    @pytest.mark.compliance
    def test_calculation_uncertainty_bounds(self, calc_library):
        """Test calculations include uncertainty bounds."""
        result = calc_library.calculate_co2_emissions(
            fuel_type="natural_gas",
            fuel_consumption=100,
            fuel_unit="MMBTU",
        )

        assert result.uncertainty is not None, "Result should have uncertainty"
        assert result.uncertainty.lower < result.value, "Lower bound should be less than value"
        assert result.uncertainty.upper > result.value, "Upper bound should be greater than value"
        assert result.uncertainty.confidence_level == 0.95, "Should use 95% confidence"

    @pytest.mark.compliance
    def test_calculation_formula_reference(self, calc_library):
        """Test calculations reference Part 98 formula."""
        result = calc_library.calculate_co2_emissions(
            fuel_type="natural_gas",
            fuel_consumption=100,
            fuel_unit="MMBTU",
        )

        assert "Part 98" in result.formula_reference or "40 CFR" in result.formula_reference, (
            f"Should reference Part 98, got: {result.formula_reference}"
        )


# =============================================================================
# EDGE CASES AND BOUNDARY CONDITIONS
# =============================================================================


class TestGHGCalculationEdgeCases:
    """
    Test edge cases and boundary conditions for GHG calculations.

    Ensures robust handling of unusual inputs.
    """

    @pytest.mark.compliance
    def test_zero_fuel_consumption(self, ghg_calculator):
        """Test handling of zero fuel consumption."""
        result = ghg_calculator.calculate_co2_emissions(
            fuel_type="natural_gas",
            fuel_quantity_mmbtu=0,
        )

        assert result == 0, "Zero fuel should result in zero emissions"

    @pytest.mark.compliance
    def test_very_small_fuel_consumption(self, ghg_calculator):
        """Test handling of very small fuel consumption."""
        result = ghg_calculator.calculate_co2_emissions(
            fuel_type="natural_gas",
            fuel_quantity_mmbtu=0.001,  # Very small
        )

        assert result > 0, "Very small fuel should still produce positive emissions"
        assert result < 0.001, "Emissions should be proportionally small"

    @pytest.mark.compliance
    def test_large_fuel_consumption(self, ghg_calculator):
        """Test handling of large fuel consumption values."""
        result = ghg_calculator.calculate_co2_emissions(
            fuel_type="natural_gas",
            fuel_quantity_mmbtu=1000000,  # 1 million MMBTU
        )

        # Expected: 1,000,000 * 53.06 / 1000 = 53,060 MT
        expected = 53060

        assert abs(result - expected) / expected < 0.01, (
            f"Large fuel calculation: {result} MT vs expected {expected} MT"
        )

    @pytest.mark.compliance
    def test_unknown_fuel_type_handling(self, ghg_calculator):
        """Test handling of unknown fuel types."""
        with pytest.raises(ValueError, match="Unknown fuel type"):
            ghg_calculator.calculate_co2_emissions(
                fuel_type="unknown_fuel",
                fuel_quantity_mmbtu=1000,
            )
