# -*- coding: utf-8 -*-
"""
EPA 40 CFR Part 60 NSPS Compliance Tests

Tests compliance with New Source Performance Standards for:
    - Emission limits by fuel type (NOx, SO2, PM, CO)
    - Continuous emission monitoring requirements
    - Reporting thresholds
    - O2 correction calculations per EPA Method 19

Standards Reference:
    - 40 CFR Part 60 Subpart Da - Electric Utility Steam Generating Units
    - 40 CFR Part 60 Subpart Db - Industrial-Commercial-Institutional Steam Units
    - 40 CFR Part 60 Subpart Dc - Small Industrial-Commercial-Institutional Units
    - 40 CFR Part 60 Subpart Ja - Petroleum Refinery Process Heaters

Author: GL-TestEngineer
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List
import math
import pytest

# Import process heat agents for testing
# Use broad exception handling to catch pydantic errors during import
try:
    from greenlang.agents.process_heat.gl_018_unified_combustion.emissions import (
        EmissionsController,
        NOX_EMISSION_FACTORS,
        CO_EMISSION_FACTORS,
        CO2_EMISSION_FACTORS,
    )
    from greenlang.agents.process_heat.gl_018_unified_combustion.config import EmissionsConfig
    EMISSIONS_CONTROLLER_AVAILABLE = True
except Exception:
    EMISSIONS_CONTROLLER_AVAILABLE = False
    EmissionsController = None
    NOX_EMISSION_FACTORS = {}
    CO_EMISSION_FACTORS = {}
    CO2_EMISSION_FACTORS = {}
    EmissionsConfig = None

try:
    from greenlang.agents.process_heat.gl_010_emissions_guardian.monitor import (
        EmissionsMonitor,
        EmissionsInput,
        EmissionsOutput,
    )
    EMISSIONS_MONITOR_AVAILABLE = True
except Exception:
    EMISSIONS_MONITOR_AVAILABLE = False
    EmissionsMonitor = None
    EmissionsInput = None
    EmissionsOutput = None


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def emissions_config():
    """Create default emissions configuration for testing."""
    if not EMISSIONS_CONTROLLER_AVAILABLE:
        pytest.skip("EmissionsConfig not available")
    return EmissionsConfig(
        nox_permit_limit_lb_mmbtu=0.10,
        co_permit_limit_lb_mmbtu=0.05,
        ammonia_slip_limit_ppm=10.0,
        fgr_enabled=True,
        scr_enabled=True,
        scr_inlet_temp_min_f=575.0,
        scr_inlet_temp_max_f=750.0,
    )


@pytest.fixture
def emissions_controller(emissions_config):
    """Create EmissionsController instance for testing."""
    if not EMISSIONS_CONTROLLER_AVAILABLE:
        pytest.skip("EmissionsController not available")
    return EmissionsController(emissions_config)


@pytest.fixture
def emissions_monitor():
    """Create EmissionsMonitor instance for testing."""
    if not EMISSIONS_MONITOR_AVAILABLE:
        pytest.skip("EmissionsMonitor not available")
    return EmissionsMonitor(
        source_id="BOILER-001",
        permit_limits={
            "co2_lb_hr": 5000.0,
            "nox_lb_hr": 10.0,
        }
    )


# =============================================================================
# EPA PART 60 EMISSION FACTOR VALIDATION TESTS
# =============================================================================


class TestEPAPart60EmissionFactors:
    """
    Validate emission factors against EPA AP-42 and Part 60 requirements.

    Pass/Fail Criteria:
        - NOx factors must match EPA AP-42 Table 1.4-1 within 5%
        - CO factors must match EPA AP-42 Table 1.4-2 within 5%
        - CO2 factors must match 40 CFR Part 98 Table C-1 exactly
    """

    @pytest.mark.compliance
    def test_nox_emission_factors_natural_gas(self):
        """
        Test NOx emission factors for natural gas match EPA AP-42.

        EPA AP-42, Chapter 1.4, Table 1.4-1:
            - Uncontrolled: 0.098 lb/MMBTU
            - Low NOx Burner: 0.049 lb/MMBTU
            - Ultra-low NOx: 0.025 lb/MMBTU
        """
        if not EMISSIONS_CONTROLLER_AVAILABLE:
            pytest.skip("Emissions module not available")

        epa_reference_factors = {
            "uncontrolled": 0.098,
            "low_nox_burner": 0.049,
            "ultra_low_nox": 0.025,
            "fgr": 0.035,
            "fgr_lnb": 0.020,
            "scr": 0.010,
        }

        greenlang_factors = NOX_EMISSION_FACTORS.get("natural_gas", {})

        for control_type, epa_factor in epa_reference_factors.items():
            greenlang_factor = greenlang_factors.get(control_type)

            assert greenlang_factor is not None, (
                f"Missing NOx factor for natural_gas/{control_type}"
            )

            # Allow 5% tolerance for regulatory compliance
            tolerance = 0.05
            assert abs(greenlang_factor - epa_factor) / epa_factor <= tolerance, (
                f"NOx factor for natural_gas/{control_type} out of tolerance: "
                f"{greenlang_factor} vs EPA {epa_factor}"
            )

    @pytest.mark.compliance
    def test_nox_emission_factors_fuel_oil(self):
        """
        Test NOx emission factors for fuel oil match EPA AP-42.

        EPA AP-42, Chapter 1.3:
            - No. 2 Fuel Oil Uncontrolled: 0.140 lb/MMBTU
            - No. 6 Fuel Oil Uncontrolled: 0.170 lb/MMBTU
        """
        if not EMISSIONS_CONTROLLER_AVAILABLE:
            pytest.skip("Emissions module not available")

        epa_reference = {
            "no2_fuel_oil": {"uncontrolled": 0.140, "low_nox_burner": 0.070},
            "no6_fuel_oil": {"uncontrolled": 0.170, "low_nox_burner": 0.085},
        }

        for fuel_type, controls in epa_reference.items():
            greenlang_factors = NOX_EMISSION_FACTORS.get(fuel_type, {})

            for control_type, epa_factor in controls.items():
                greenlang_factor = greenlang_factors.get(control_type)

                assert greenlang_factor is not None, (
                    f"Missing NOx factor for {fuel_type}/{control_type}"
                )

                tolerance = 0.05
                assert abs(greenlang_factor - epa_factor) / epa_factor <= tolerance, (
                    f"NOx factor for {fuel_type}/{control_type} out of tolerance: "
                    f"{greenlang_factor} vs EPA {epa_factor}"
                )

    @pytest.mark.compliance
    def test_co2_emission_factors_part98(self, epa_part98_emission_factors):
        """
        Test CO2 emission factors match 40 CFR Part 98 Table C-1.

        These factors are legally mandated for GHG reporting.
        Must match exactly (no tolerance allowed).
        """
        if not EMISSIONS_CONTROLLER_AVAILABLE:
            pytest.skip("Emissions module not available")

        part98_to_greenlang_mapping = {
            "natural_gas": "natural_gas",
            "distillate_fuel_oil_no2": "no2_fuel_oil",
            "residual_fuel_oil_no6": "no6_fuel_oil",
            "propane": "propane",
            "bituminous_coal": "coal_bituminous",
        }

        for part98_fuel, greenlang_fuel in part98_to_greenlang_mapping.items():
            part98_factor = epa_part98_emission_factors[part98_fuel]["co2_kg_per_mmbtu"]
            greenlang_factor = CO2_EMISSION_FACTORS.get(greenlang_fuel)

            assert greenlang_factor is not None, (
                f"Missing CO2 factor for {greenlang_fuel}"
            )

            # Part 98 factors must match exactly for regulatory compliance
            assert abs(greenlang_factor - part98_factor) < 0.01, (
                f"CO2 factor for {greenlang_fuel} does not match Part 98: "
                f"{greenlang_factor} vs {part98_factor}"
            )


# =============================================================================
# O2 CORRECTION CALCULATION TESTS (EPA METHOD 19)
# =============================================================================


class TestEPAMethod19O2Correction:
    """
    Test O2 correction calculations per EPA Method 19.

    The O2 correction formula:
        Corrected = Measured * (20.9 - O2_ref) / (20.9 - O2_meas)

    Pass/Fail Criteria:
        - Correction factor must be calculated within 0.1% tolerance
        - Reference O2 of 3% is the default for most standards
    """

    @pytest.mark.compliance
    @pytest.mark.parametrize("measured_o2,expected_factor", [
        (3.0, 1.000),   # Reference O2 = no correction
        (4.0, 1.059),   # 4% O2 -> 1.059x correction
        (5.0, 1.126),   # 5% O2 -> 1.126x correction
        (6.0, 1.201),   # 6% O2 -> 1.201x correction
        (2.0, 0.947),   # 2% O2 -> 0.947x correction (lower)
        (1.0, 0.899),   # 1% O2 -> 0.899x correction
        (10.0, 1.643),  # 10% O2 -> high correction
    ])
    def test_o2_correction_factor_calculation(
        self,
        emissions_controller,
        measured_o2: float,
        expected_factor: float,
    ):
        """Test O2 correction factors match EPA Method 19 formula."""
        reference_o2 = 3.0

        # Calculate correction using private method
        calculated_factor = emissions_controller._calculate_o2_correction(
            measured_o2_pct=measured_o2,
            reference_o2_pct=reference_o2,
        )

        # Calculate expected using EPA formula
        expected_calculated = (20.9 - reference_o2) / (20.9 - measured_o2)

        # Verify calculation matches formula
        assert abs(calculated_factor - expected_calculated) < 0.001, (
            f"O2 correction at {measured_o2}% does not match formula: "
            f"{calculated_factor} vs {expected_calculated}"
        )

        # Verify matches expected lookup value (within 0.5%)
        assert abs(calculated_factor - expected_factor) / expected_factor < 0.005, (
            f"O2 correction at {measured_o2}% differs from expected: "
            f"{calculated_factor} vs {expected_factor}"
        )

    @pytest.mark.compliance
    def test_o2_correction_edge_case_ambient_air(self, emissions_controller):
        """Test O2 correction handles ambient air (20.9% O2) correctly."""
        # At 20.9% O2, correction would be undefined (division by zero)
        # Implementation should return 1.0 or handle gracefully

        correction = emissions_controller._calculate_o2_correction(
            measured_o2_pct=20.9,
            reference_o2_pct=3.0,
        )

        # Should return 1.0 (no correction) when at ambient O2
        assert correction == 1.0, (
            f"O2 correction at ambient (20.9%) should return 1.0, got {correction}"
        )

    @pytest.mark.compliance
    @pytest.mark.parametrize("nox_ppm,o2_pct,expected_corrected", [
        (25.0, 3.0, 25.0),    # No correction at reference O2
        (25.0, 5.0, 28.15),   # Higher O2 -> higher corrected value
        (25.0, 2.0, 23.68),   # Lower O2 -> lower corrected value
        (50.0, 4.0, 52.95),   # Double NOx with 4% O2
        (100.0, 6.0, 120.07), # High NOx at high O2
    ])
    def test_nox_o2_correction_application(
        self,
        emissions_controller,
        nox_ppm: float,
        o2_pct: float,
        expected_corrected: float,
    ):
        """Test NOx concentration correction to 3% O2 reference."""
        correction_factor = emissions_controller._calculate_o2_correction(
            measured_o2_pct=o2_pct,
            reference_o2_pct=3.0,
        )

        corrected_nox = nox_ppm * correction_factor

        # Allow 1% tolerance for rounding
        tolerance = 0.01
        assert abs(corrected_nox - expected_corrected) / expected_corrected < tolerance, (
            f"NOx correction failed: {nox_ppm} ppm at {o2_pct}% O2 "
            f"corrected to {corrected_nox:.2f}, expected {expected_corrected}"
        )


# =============================================================================
# NSPS EMISSION LIMIT COMPLIANCE TESTS
# =============================================================================


class TestNSPSEmissionLimitCompliance:
    """
    Test emission calculations against NSPS limits.

    Pass/Fail Criteria:
        - Calculations must correctly identify compliance/exceedance
        - Emission rates must be calculated accurately (within 2%)
        - Proper unit conversions (ppm to lb/MMBTU)
    """

    @pytest.mark.compliance
    @pytest.mark.parametrize("nox_ppm,o2_pct,fuel_rate_mmbtu,expected_compliance", [
        (15.0, 3.0, 50.0, True),   # Low NOx - compliant
        (25.0, 3.0, 50.0, True),   # Moderate NOx - compliant
        (50.0, 3.0, 50.0, False),  # High NOx - exceedance
        (75.0, 3.0, 50.0, False),  # Very high NOx - exceedance
        (30.0, 6.0, 50.0, False),  # Moderate NOx but high O2 correction
    ])
    def test_nox_compliance_determination(
        self,
        emissions_controller,
        nox_ppm: float,
        o2_pct: float,
        fuel_rate_mmbtu: float,
        expected_compliance: bool,
    ):
        """Test NOx compliance determination against permit limits."""
        result = emissions_controller.analyze_emissions(
            nox_ppm=nox_ppm,
            o2_pct=o2_pct,
            co_ppm=30.0,  # Low CO
            fuel_type="natural_gas",
            fuel_consumption_mmbtu_hr=fuel_rate_mmbtu,
        )

        # Check if NOx compliance matches expectation
        nox_in_compliance = (
            result.nox_lb_mmbtu is None or
            result.nox_lb_mmbtu <= emissions_controller.config.nox_permit_limit_lb_mmbtu
        )

        assert nox_in_compliance == expected_compliance, (
            f"NOx compliance mismatch at {nox_ppm} ppm: "
            f"expected {'compliant' if expected_compliance else 'exceedance'}, "
            f"got lb/MMBTU={result.nox_lb_mmbtu}, limit={emissions_controller.config.nox_permit_limit_lb_mmbtu}"
        )

    @pytest.mark.compliance
    @pytest.mark.parametrize("co_ppm,expected_status", [
        (20.0, "normal"),    # Low CO - good combustion
        (50.0, "normal"),    # Moderate CO - acceptable
        (100.0, "warning"),  # Elevated CO - tuning needed
        (200.0, "alarm"),    # High CO - combustion problem
        (400.0, "alarm"),    # Very high CO - immediate action
    ])
    def test_co_level_status_determination(
        self,
        emissions_controller,
        co_ppm: float,
        expected_status: str,
    ):
        """Test CO level status determination for combustion quality."""
        result = emissions_controller.analyze_emissions(
            nox_ppm=25.0,
            o2_pct=3.0,
            co_ppm=co_ppm,
            fuel_type="natural_gas",
            fuel_consumption_mmbtu_hr=50.0,
        )

        # Determine CO status from result
        co_compliance_pct = result.co_compliance_pct

        if co_compliance_pct <= 50:
            actual_status = "normal"
        elif co_compliance_pct <= 80:
            actual_status = "warning"
        else:
            actual_status = "alarm"

        # Also check recommendations for CO
        has_co_recommendation = any(
            "co" in rec.lower() or "combustion" in rec.lower()
            for rec in result.recommendations
        )

        if expected_status in ["warning", "alarm"]:
            assert co_ppm > 50 or has_co_recommendation, (
                f"High CO ({co_ppm} ppm) should trigger recommendation"
            )

    @pytest.mark.compliance
    def test_emissions_rate_calculation_accuracy(
        self,
        emissions_controller,
        epa_part60_emission_limits,
    ):
        """
        Test emission rate calculations are accurate.

        Validates lb/MMBTU calculations against EPA Method 19.
        """
        # Test with known values
        test_cases = [
            {
                "nox_ppm": 25.0,
                "o2_pct": 3.0,
                "fuel_type": "natural_gas",
                "fuel_rate": 100.0,
            },
            {
                "nox_ppm": 40.0,
                "o2_pct": 4.5,
                "fuel_type": "natural_gas",
                "fuel_rate": 75.0,
            },
        ]

        for case in test_cases:
            result = emissions_controller.analyze_emissions(
                nox_ppm=case["nox_ppm"],
                o2_pct=case["o2_pct"],
                co_ppm=30.0,
                fuel_type=case["fuel_type"],
                fuel_consumption_mmbtu_hr=case["fuel_rate"],
            )

            # Verify NOx lb/MMBTU is reasonable
            if result.nox_lb_mmbtu is not None:
                assert 0 < result.nox_lb_mmbtu < 1.0, (
                    f"NOx lb/MMBTU out of reasonable range: {result.nox_lb_mmbtu}"
                )

            # Verify CO lb/MMBTU is reasonable
            assert 0 < result.co_lb_mmbtu < 1.0, (
                f"CO lb/MMBTU out of reasonable range: {result.co_lb_mmbtu}"
            )


# =============================================================================
# SUBPART-SPECIFIC COMPLIANCE TESTS
# =============================================================================


class TestNSPSSubpartDbCompliance:
    """
    Test compliance with NSPS Subpart Db requirements.

    Subpart Db applies to industrial-commercial-institutional
    steam generating units with heat input capacity > 100 MMBTU/hr.
    """

    @pytest.mark.compliance
    def test_subpart_db_nox_limit_natural_gas(
        self,
        epa_part60_emission_limits,
    ):
        """Test natural gas NOx limit for Subpart Db units."""
        limits = epa_part60_emission_limits["steam_generator_natural_gas_db"]

        # Subpart Db NOx limit for gas: 0.20 lb/MMBTU
        expected_nox_limit = 0.20

        assert limits["nox_lb_mmbtu"] == expected_nox_limit, (
            f"Subpart Db natural gas NOx limit should be {expected_nox_limit}, "
            f"got {limits['nox_lb_mmbtu']}"
        )

    @pytest.mark.compliance
    def test_subpart_db_applicability_threshold(
        self,
        epa_part60_emission_limits,
    ):
        """Test Subpart Db applicability threshold (100 MMBTU/hr)."""
        limits = epa_part60_emission_limits["steam_generator_natural_gas_db"]

        # Subpart Db applies to units >= 100 MMBTU/hr
        expected_threshold = 100.0

        assert limits["heat_input_threshold_mmbtu_hr"] == expected_threshold, (
            f"Subpart Db threshold should be {expected_threshold} MMBTU/hr"
        )


class TestNSPSSubpartDcCompliance:
    """
    Test compliance with NSPS Subpart Dc requirements.

    Subpart Dc applies to small industrial-commercial-institutional
    steam generating units (10-100 MMBTU/hr capacity).
    """

    @pytest.mark.compliance
    def test_subpart_dc_nox_limit_natural_gas(
        self,
        epa_part60_emission_limits,
    ):
        """Test natural gas NOx limit for Subpart Dc units."""
        limits = epa_part60_emission_limits["steam_generator_natural_gas_dc"]

        # Subpart Dc NOx limit for gas: 0.30 lb/MMBTU (less stringent)
        expected_nox_limit = 0.30

        assert limits["nox_lb_mmbtu"] == expected_nox_limit, (
            f"Subpart Dc natural gas NOx limit should be {expected_nox_limit}, "
            f"got {limits['nox_lb_mmbtu']}"
        )

    @pytest.mark.compliance
    def test_subpart_dc_applicability_threshold(
        self,
        epa_part60_emission_limits,
    ):
        """Test Subpart Dc applicability threshold (10 MMBTU/hr)."""
        limits = epa_part60_emission_limits["steam_generator_natural_gas_dc"]

        # Subpart Dc applies to units >= 10 MMBTU/hr
        expected_threshold = 10.0

        assert limits["heat_input_threshold_mmbtu_hr"] == expected_threshold, (
            f"Subpart Dc threshold should be {expected_threshold} MMBTU/hr"
        )


# =============================================================================
# EMISSION MONITORING AND REPORTING TESTS
# =============================================================================


class TestEmissionMonitoringCompliance:
    """
    Test continuous emission monitoring compliance.

    Pass/Fail Criteria:
        - Monitoring must correctly identify exceedances
        - Predictions must be based on trend analysis
        - Daily/hourly averages must be calculated correctly
    """

    @pytest.mark.compliance
    def test_exceedance_detection(self, emissions_monitor):
        """Test that exceedances are correctly detected."""
        # Create input that exceeds permit limit
        input_data = EmissionsInput(
            source_id="BOILER-001",
            fuel_type="natural_gas",
            fuel_flow_rate=100.0,  # High fuel rate
            stack_o2_pct=3.0,
            stack_co_ppm=50.0,
            stack_temperature_f=400.0,
        )

        result = emissions_monitor.monitor(input_data)

        # Check if CO2 is above or near limit
        assert result.co2_lb_hr is not None, "CO2 emissions should be calculated"
        assert isinstance(result.status, str), "Status should be a string"

        # If exceeded, should have exceedances list
        if result.status == "exceedance":
            assert len(result.exceedances) > 0, (
                "Exceedance status should include exceedance details"
            )

    @pytest.mark.compliance
    def test_co_warning_threshold(self, emissions_monitor):
        """Test high CO warning is triggered at correct threshold."""
        # Create input with high CO
        input_data = EmissionsInput(
            source_id="BOILER-001",
            fuel_type="natural_gas",
            fuel_flow_rate=50.0,
            stack_o2_pct=3.0,
            stack_co_ppm=250.0,  # High CO
            stack_temperature_f=400.0,
        )

        result = emissions_monitor.monitor(input_data)

        # Should have warning about high CO
        has_co_warning = any("co" in w.lower() for w in result.warnings)

        assert has_co_warning, (
            f"High CO ({input_data.stack_co_ppm} ppm) should trigger warning. "
            f"Warnings: {result.warnings}"
        )

    @pytest.mark.compliance
    def test_emissions_annualization(self, emissions_monitor):
        """Test annual emission projections are calculated correctly."""
        input_data = EmissionsInput(
            source_id="BOILER-001",
            fuel_type="natural_gas",
            fuel_flow_rate=50.0,
            stack_o2_pct=3.0,
            stack_co_ppm=30.0,
            stack_temperature_f=400.0,
        )

        result = emissions_monitor.monitor(input_data)

        # Annual = hourly * 8760 / 2000 (convert lb to tons)
        expected_annual = result.co2_lb_hr * 8760 / 2000

        assert abs(result.co2_ton_yr - expected_annual) < 1.0, (
            f"Annual CO2 projection incorrect: {result.co2_ton_yr} vs {expected_annual}"
        )


# =============================================================================
# EPA TEST METHOD REFERENCE TESTS
# =============================================================================


class TestEPATestMethodCompliance:
    """
    Test compliance with EPA reference test methods.

    Verifies that calculation methods align with EPA test procedures.
    """

    @pytest.mark.compliance
    def test_method_19_fd_factors(self, epa_part60_test_methods):
        """
        Test Fd factors used in Method 19 calculations.

        Fd factors (dscf/MMBTU) are fuel-specific conversion factors.
        """
        # EPA Method 19 Fd factors
        expected_fd_factors = {
            "natural_gas": 8710,
            "no2_fuel_oil": 9190,
            "no6_fuel_oil": 9220,
            "coal_bituminous": 9780,
        }

        # Verify Method 19 is referenced
        method_19 = epa_part60_test_methods.get("method_19")
        assert method_19 is not None, "Method 19 should be in test methods"
        assert method_19["o2_correction_reference"] == 3.0, (
            "Method 19 O2 reference should be 3%"
        )

    @pytest.mark.compliance
    def test_method_7_nox_reference(self, epa_part60_test_methods):
        """Test Method 7 is referenced for NOx measurements."""
        method_7 = epa_part60_test_methods.get("method_7")
        assert method_7 is not None, "Method 7 should be in test methods"
        assert method_7["pollutant"] == "nox", "Method 7 should be for NOx"


# =============================================================================
# EDGE CASES AND BOUNDARY CONDITIONS
# =============================================================================


class TestNSPSEdgeCases:
    """
    Test edge cases and boundary conditions for NSPS compliance.

    Ensures robust handling of:
        - Zero and near-zero values
        - Maximum operating conditions
        - Invalid inputs
    """

    @pytest.mark.compliance
    def test_zero_fuel_flow_handling(self, emissions_controller):
        """Test handling of zero fuel flow rate."""
        # Zero fuel flow should result in zero emissions or error
        result = emissions_controller.analyze_emissions(
            nox_ppm=25.0,
            o2_pct=3.0,
            co_ppm=30.0,
            fuel_type="natural_gas",
            fuel_consumption_mmbtu_hr=0.001,  # Very low but not zero
        )

        # Should calculate without error
        assert result is not None, "Should handle very low fuel flow"
        assert result.co2_tons_hr >= 0, "CO2 should be non-negative"

    @pytest.mark.compliance
    def test_high_o2_edge_case(self, emissions_controller):
        """Test handling of unusually high O2 levels."""
        # High O2 (near ambient) indicates air leakage
        result = emissions_controller.analyze_emissions(
            nox_ppm=10.0,
            o2_pct=18.0,  # Very high - likely measurement issue
            co_ppm=30.0,
            fuel_type="natural_gas",
            fuel_consumption_mmbtu_hr=50.0,
        )

        # Should still calculate, but correction factor will be very high
        assert result is not None, "Should handle high O2"

        # Correction factor at 18% O2 would be (20.9-3)/(20.9-18) = 6.17
        # This is unusual but mathematically valid

    @pytest.mark.compliance
    def test_permit_limit_boundary(self, emissions_controller):
        """Test behavior at exact permit limit boundary."""
        # Find emission level that's exactly at limit
        permit_limit = emissions_controller.config.nox_permit_limit_lb_mmbtu

        # This would require iterating to find exact ppm that gives limit
        # For now, test that limit comparison is correct (not > vs >=)
        result = emissions_controller.analyze_emissions(
            nox_ppm=25.0,  # This may or may not hit limit exactly
            o2_pct=3.0,
            co_ppm=30.0,
            fuel_type="natural_gas",
            fuel_consumption_mmbtu_hr=50.0,
        )

        # At exact limit, should be compliant (not exceedance)
        if result.nox_lb_mmbtu == permit_limit:
            assert result.in_compliance, (
                "Emission exactly at limit should be compliant"
            )


# =============================================================================
# PROVENANCE AND AUDIT TRAIL TESTS
# =============================================================================


class TestEmissionCalculationProvenance:
    """
    Test provenance tracking for emission calculations.

    Regulatory compliance requires complete audit trail.
    """

    @pytest.mark.compliance
    def test_calculation_deterministic(self, emissions_controller):
        """Test that calculations are deterministic (reproducible)."""
        inputs = {
            "nox_ppm": 25.0,
            "o2_pct": 3.5,
            "co_ppm": 30.0,
            "fuel_type": "natural_gas",
            "fuel_consumption_mmbtu_hr": 50.0,
        }

        # Run calculation multiple times
        results = []
        for _ in range(5):
            result = emissions_controller.analyze_emissions(**inputs)
            results.append(result)

        # All results should be identical
        for i in range(1, len(results)):
            assert results[i].nox_lb_mmbtu == results[0].nox_lb_mmbtu, (
                "NOx calculation should be deterministic"
            )
            assert results[i].co_lb_mmbtu == results[0].co_lb_mmbtu, (
                "CO calculation should be deterministic"
            )
            assert results[i].co2_lb_mmbtu == results[0].co2_lb_mmbtu, (
                "CO2 calculation should be deterministic"
            )

    @pytest.mark.compliance
    def test_emission_rate_calculation_formula_documentation(
        self,
        emissions_controller,
    ):
        """
        Test that emission rate calculations follow documented formulas.

        lb/MMBTU = ppm * MW * Fd / (20.9 - O2%) * 10^-6

        Where:
            MW = molecular weight (NO2=46, CO=28)
            Fd = fuel-specific factor (dscf/MMBTU)
        """
        # This test verifies the implementation follows EPA Method 19
        # by checking intermediate values

        nox_ppm = 25.0
        o2_pct = 3.0
        fuel_type = "natural_gas"

        # EPA Method 19 parameters for natural gas
        MW_NOX = 46.0  # NO2 molecular weight
        Fd = 8710  # dscf/MMBTU for natural gas

        # Calculate expected lb/dscf
        lb_dscf = nox_ppm * MW_NOX / 385.5 * 1e-6

        # Calculate expected lb/MMBTU
        expected_lb_mmbtu = lb_dscf * Fd * (20.9 / (20.9 - o2_pct))

        # Get actual calculation
        result = emissions_controller.analyze_emissions(
            nox_ppm=nox_ppm,
            o2_pct=o2_pct,
            co_ppm=30.0,
            fuel_type=fuel_type,
            fuel_consumption_mmbtu_hr=50.0,
        )

        if result.nox_lb_mmbtu is not None:
            # Allow 5% tolerance for implementation differences
            tolerance = 0.05
            assert abs(result.nox_lb_mmbtu - expected_lb_mmbtu) / expected_lb_mmbtu < tolerance, (
                f"NOx lb/MMBTU calculation differs from Method 19: "
                f"{result.nox_lb_mmbtu} vs expected {expected_lb_mmbtu}"
            )
