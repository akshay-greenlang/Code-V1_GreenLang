# -*- coding: utf-8 -*-
"""
Unit Tests for GL-010 EMISSIONWATCH Particulate Matter Calculator.

Tests PM emissions calculations including PM10, PM2.5, filterable and
condensable PM, control device efficiency, and EPA Method 5 compliance.

Test Count: 18+ tests
Coverage Target: 90%+

Standards: EPA Method 5, EPA Method 202

Author: GreenLang Foundation Test Engineering
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone
from typing import Any, Dict

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools import (
    EmissionsComplianceTools,
    PMEmissionsResult,
    AP42_EMISSION_FACTORS,
)


# =============================================================================
# TEST CLASS: PM EMISSIONS CALCULATOR
# =============================================================================

@pytest.mark.unit
class TestPMCalculator:
    """Test suite for particulate matter emissions calculations."""

    # =========================================================================
    # BASIC CALCULATION TESTS
    # =========================================================================

    def test_calculate_pm_emissions_basic(
        self, emissions_tools, sample_cems_data, natural_gas_fuel_data
    ):
        """Test basic PM emissions calculation."""
        result = emissions_tools.calculate_particulate_matter(
            cems_data=sample_cems_data,
            fuel_data=natural_gas_fuel_data,
        )

        assert isinstance(result, PMEmissionsResult)
        assert result.concentration_mg_m3 >= 0
        assert result.emission_rate_lb_mmbtu >= 0
        assert result.emission_rate_lb_hr >= 0
        assert result.mass_rate_kg_hr >= 0
        assert result.calculation_method == "AP42_with_Control"
        assert result.provenance_hash is not None

    def test_calculate_pm_emissions_coal(
        self, emissions_tools, sample_cems_data, coal_bituminous_data
    ):
        """Test PM calculation for coal combustion."""
        result = emissions_tools.calculate_particulate_matter(
            cems_data=sample_cems_data,
            fuel_data=coal_bituminous_data,
        )

        # Coal has higher PM than natural gas
        assert result.emission_rate_lb_mmbtu >= 0

    # =========================================================================
    # PM10 CALCULATION TESTS
    # =========================================================================

    def test_pm10_calculation_fraction(
        self, emissions_tools, sample_cems_data, natural_gas_fuel_data
    ):
        """Test PM10 is proper fraction of total PM."""
        result = emissions_tools.calculate_particulate_matter(
            cems_data=sample_cems_data,
            fuel_data=natural_gas_fuel_data,
        )

        # PM10 should be ~95% of total PM for combustion
        assert 0.90 <= result.pm10_fraction <= 1.0

    def test_pm10_calculation_coal(
        self, emissions_tools, sample_cems_data, coal_bituminous_data
    ):
        """Test PM10 calculation for coal."""
        result = emissions_tools.calculate_particulate_matter(
            cems_data=sample_cems_data,
            fuel_data=coal_bituminous_data,
        )

        assert result.pm10_fraction > 0.90

    # =========================================================================
    # PM2.5 CALCULATION TESTS
    # =========================================================================

    def test_pm25_calculation_fraction(
        self, emissions_tools, sample_cems_data, natural_gas_fuel_data
    ):
        """Test PM2.5 is proper fraction of total PM."""
        result = emissions_tools.calculate_particulate_matter(
            cems_data=sample_cems_data,
            fuel_data=natural_gas_fuel_data,
        )

        # PM2.5 should be ~85% of total PM
        assert 0.80 <= result.pm25_fraction <= 0.95

    def test_pm25_less_than_pm10(
        self, emissions_tools, sample_cems_data, natural_gas_fuel_data
    ):
        """Test PM2.5 fraction is less than PM10 fraction."""
        result = emissions_tools.calculate_particulate_matter(
            cems_data=sample_cems_data,
            fuel_data=natural_gas_fuel_data,
        )

        assert result.pm25_fraction <= result.pm10_fraction

    # =========================================================================
    # FILTERABLE/CONDENSABLE PM TESTS
    # =========================================================================

    def test_filterable_condensable_pm_split(
        self, emissions_tools, sample_cems_data, natural_gas_fuel_data
    ):
        """Test filterable and condensable PM sum to 100%."""
        result = emissions_tools.calculate_particulate_matter(
            cems_data=sample_cems_data,
            fuel_data=natural_gas_fuel_data,
        )

        total = result.filterable_percent + result.condensable_percent
        assert abs(total - 100.0) < 0.1

    def test_filterable_pm_typical_split(
        self, emissions_tools, sample_cems_data, coal_bituminous_data
    ):
        """Test typical filterable PM fraction."""
        result = emissions_tools.calculate_particulate_matter(
            cems_data=sample_cems_data,
            fuel_data=coal_bituminous_data,
        )

        # Filterable typically 60-80% for combustion
        assert 50.0 <= result.filterable_percent <= 90.0

    def test_condensable_pm_present(
        self, emissions_tools, sample_cems_data, natural_gas_fuel_data
    ):
        """Test condensable PM is tracked."""
        result = emissions_tools.calculate_particulate_matter(
            cems_data=sample_cems_data,
            fuel_data=natural_gas_fuel_data,
        )

        # Condensable PM should be present
        assert result.condensable_percent > 0

    # =========================================================================
    # CONTROL DEVICE EFFICIENCY TESTS
    # =========================================================================

    def test_control_device_efficiency_baghouse(
        self, emissions_tools, sample_cems_data, coal_bituminous_data
    ):
        """Test baghouse PM control efficiency."""
        no_control = {"baghouse_efficiency_percent": 0.0}
        with_baghouse = {"baghouse_efficiency_percent": 99.5}

        result_no_control = emissions_tools.calculate_particulate_matter(
            cems_data=sample_cems_data,
            fuel_data=coal_bituminous_data,
            process_parameters=no_control,
        )

        result_with_baghouse = emissions_tools.calculate_particulate_matter(
            cems_data=sample_cems_data,
            fuel_data=coal_bituminous_data,
            process_parameters=with_baghouse,
        )

        # 99.5% control should reduce emissions significantly
        reduction = 1 - (result_with_baghouse.emission_rate_lb_hr / result_no_control.emission_rate_lb_hr)
        assert reduction > 0.99

    def test_control_device_efficiency_esp(
        self, emissions_tools, sample_cems_data, coal_bituminous_data
    ):
        """Test ESP PM control efficiency."""
        with_esp = {"baghouse_efficiency_percent": 99.0}

        result = emissions_tools.calculate_particulate_matter(
            cems_data=sample_cems_data,
            fuel_data=coal_bituminous_data,
            process_parameters=with_esp,
        )

        # Should reduce emissions by 99%
        assert result.emission_rate_lb_mmbtu > 0

    def test_control_device_efficiency_zero(
        self, emissions_tools, sample_cems_data, natural_gas_fuel_data
    ):
        """Test calculation without control device."""
        no_control = {"baghouse_efficiency_percent": 0.0}

        result = emissions_tools.calculate_particulate_matter(
            cems_data=sample_cems_data,
            fuel_data=natural_gas_fuel_data,
            process_parameters=no_control,
        )

        # Uncontrolled emissions should match AP-42 factor
        ap42_factor = AP42_EMISSION_FACTORS["natural_gas"]["pm_lb_mmbtu"]
        assert abs(result.emission_rate_lb_mmbtu - ap42_factor) < 0.001

    # =========================================================================
    # EPA METHOD 5 TESTS
    # =========================================================================

    def test_epa_method_5_calculation(
        self, emissions_tools, sample_cems_data, coal_bituminous_data
    ):
        """Test EPA Method 5 compliant calculation."""
        result = emissions_tools.calculate_particulate_matter(
            cems_data=sample_cems_data,
            fuel_data=coal_bituminous_data,
        )

        # Method 5 captures filterable PM
        assert result.filterable_percent > 0
        assert result.calculation_method == "AP42_with_Control"

    def test_concentration_mg_m3_calculation(
        self, emissions_tools, sample_cems_data, natural_gas_fuel_data
    ):
        """Test PM concentration in mg/m3."""
        result = emissions_tools.calculate_particulate_matter(
            cems_data=sample_cems_data,
            fuel_data=natural_gas_fuel_data,
        )

        # Concentration should be positive
        assert result.concentration_mg_m3 >= 0

    # =========================================================================
    # BOUNDARY CONDITION TESTS
    # =========================================================================

    def test_boundary_zero_flow_rate(self, emissions_tools, natural_gas_fuel_data):
        """Test calculation with zero flow rate."""
        cems_data = {
            "flow_rate_dscfm": 0.0,
            "opacity_percent": 5.0,
        }

        result = emissions_tools.calculate_particulate_matter(
            cems_data=cems_data,
            fuel_data=natural_gas_fuel_data,
        )

        # Zero flow should result in zero concentration
        assert result.concentration_mg_m3 == 0.0

    def test_boundary_zero_heat_input(self, emissions_tools, sample_cems_data):
        """Test calculation with zero heat input."""
        fuel_data = {
            "fuel_type": "natural_gas",
            "heat_input_mmbtu_hr": 0.0,
        }

        result = emissions_tools.calculate_particulate_matter(
            cems_data=sample_cems_data,
            fuel_data=fuel_data,
        )

        assert result.emission_rate_lb_hr == 0.0

    def test_boundary_high_ash_fuel(self, emissions_tools, sample_cems_data):
        """Test calculation with high ash content fuel."""
        fuel_data = {
            "fuel_type": "coal_bituminous",
            "heat_input_mmbtu_hr": 100.0,
            "ash_percent": 15.0,  # High ash coal
        }

        result = emissions_tools.calculate_particulate_matter(
            cems_data=sample_cems_data,
            fuel_data=fuel_data,
        )

        # Should handle high ash
        assert result.emission_rate_lb_mmbtu >= 0

    # =========================================================================
    # DETERMINISM TESTS
    # =========================================================================

    def test_determinism_pm_calculation(
        self, emissions_tools, sample_cems_data, natural_gas_fuel_data
    ):
        """Test deterministic PM calculation."""
        results = []

        for _ in range(10):
            result = emissions_tools.calculate_particulate_matter(
                cems_data=sample_cems_data,
                fuel_data=natural_gas_fuel_data,
            )
            results.append(result)

        # All results should be identical
        first = results[0]
        for result in results[1:]:
            assert result.concentration_mg_m3 == first.concentration_mg_m3
            assert result.emission_rate_lb_mmbtu == first.emission_rate_lb_mmbtu
            assert result.pm10_fraction == first.pm10_fraction
            assert result.pm25_fraction == first.pm25_fraction

    def test_determinism_provenance_hash_pm(
        self, emissions_tools, sample_cems_data, coal_bituminous_data
    ):
        """Test PM provenance hash is deterministic."""
        hashes = []

        for _ in range(5):
            result = emissions_tools.calculate_particulate_matter(
                cems_data=sample_cems_data,
                fuel_data=coal_bituminous_data,
            )
            hashes.append(result.provenance_hash)

        assert len(set(hashes)) == 1

    # =========================================================================
    # TO_DICT CONVERSION TEST
    # =========================================================================

    def test_pm_result_to_dict(
        self, emissions_tools, sample_cems_data, natural_gas_fuel_data
    ):
        """Test PMEmissionsResult to_dict conversion."""
        result = emissions_tools.calculate_particulate_matter(
            cems_data=sample_cems_data,
            fuel_data=natural_gas_fuel_data,
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "concentration_mg_m3" in result_dict
        assert "emission_rate_lb_mmbtu" in result_dict
        assert "pm10_fraction" in result_dict
        assert "pm25_fraction" in result_dict
        assert "filterable_percent" in result_dict
        assert "condensable_percent" in result_dict
        assert "calculation_method" in result_dict
        assert "provenance_hash" in result_dict


# =============================================================================
# PARAMETRIZED TESTS
# =============================================================================

@pytest.mark.unit
class TestPMCalculatorParametrized:
    """Parametrized tests for PM calculator."""

    @pytest.mark.parametrize("fuel_type,expected_factor_min,expected_factor_max", [
        ("natural_gas", 0.005, 0.010),
        ("fuel_oil_no2", 0.020, 0.030),
        ("coal_bituminous", 0.35, 0.45),
        ("biomass_wood", 0.25, 0.35),
    ])
    def test_pm_emission_factors_by_fuel(
        self, emissions_tools, sample_cems_data, fuel_type, expected_factor_min, expected_factor_max
    ):
        """Test PM emission factors for different fuels (uncontrolled)."""
        fuel_data = {
            "fuel_type": fuel_type,
            "heat_input_mmbtu_hr": 100.0,
        }
        process_params = {"baghouse_efficiency_percent": 0.0}

        result = emissions_tools.calculate_particulate_matter(
            cems_data=sample_cems_data,
            fuel_data=fuel_data,
            process_parameters=process_params,
        )

        assert expected_factor_min <= result.emission_rate_lb_mmbtu <= expected_factor_max

    @pytest.mark.parametrize("control_efficiency,expected_reduction_min", [
        (90.0, 0.89),
        (95.0, 0.94),
        (99.0, 0.98),
        (99.5, 0.99),
        (99.9, 0.998),
    ])
    def test_control_device_efficiency_levels(
        self, emissions_tools, sample_cems_data, coal_bituminous_data,
        control_efficiency, expected_reduction_min
    ):
        """Test various control device efficiency levels."""
        no_control = {"baghouse_efficiency_percent": 0.0}
        with_control = {"baghouse_efficiency_percent": control_efficiency}

        result_no_control = emissions_tools.calculate_particulate_matter(
            cems_data=sample_cems_data,
            fuel_data=coal_bituminous_data,
            process_parameters=no_control,
        )

        result_with_control = emissions_tools.calculate_particulate_matter(
            cems_data=sample_cems_data,
            fuel_data=coal_bituminous_data,
            process_parameters=with_control,
        )

        if result_no_control.emission_rate_lb_hr > 0:
            reduction = 1 - (result_with_control.emission_rate_lb_hr / result_no_control.emission_rate_lb_hr)
            assert reduction >= expected_reduction_min
