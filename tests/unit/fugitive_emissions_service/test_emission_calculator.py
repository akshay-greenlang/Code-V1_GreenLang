# -*- coding: utf-8 -*-
"""
Unit tests for EmissionCalculatorEngine (Engine 2 of 7) - AGENT-MRV-005

Tests all 5 calculation methods:
  1. Average Emission Factor (15 tests)
  2. Screening Ranges (10 tests)
  3. Correlation Equation (8 tests)
  4. Engineering Estimate (10 tests - pneumatic, tank, coal mine, wastewater)
  5. Direct Measurement (3 tests)
  6. Batch Processing (4 tests)

Validates GWP application, Decimal precision, known value calculations,
error handling, and provenance hashing.

Target: 50 tests, ~540 lines.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-005 Fugitive Emissions (GL-MRV-SCOPE1-005)
"""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from greenlang.fugitive_emissions.emission_calculator import (
    EmissionCalculatorEngine,
    CalculationMethod,
    CalculationStatus,
    EngineeringSubmethod,
    GasResult,
    _D,
    _quantize,
    _PRECISION,
    _ZERO,
    _ONE,
    _HOURS_PER_YEAR,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EIGHT_DP = Decimal("0.00000001")


def _build_mock_source_db() -> MagicMock:
    """Build a mock FugitiveSourceDatabaseEngine with default lookups."""
    db = MagicMock()

    def _component_ef(component_type: str, service_type: str) -> Dict[str, Any]:
        _FACTORS = {
            ("valve", "gas"): Decimal("0.00597"),
            ("valve", "light_liquid"): Decimal("0.00403"),
            ("pump", "light_liquid"): Decimal("0.01140"),
            ("compressor", "gas"): Decimal("0.22800"),
            ("connector", "gas"): Decimal("0.00183"),
            ("flange", "gas"): Decimal("0.00083"),
        }
        key = (component_type.lower(), service_type.lower())
        ef = _FACTORS.get(key)
        if ef is None:
            return None
        return {"ef_decimal": ef, "source": "EPA-453/R-95-017"}

    def _screening_factor(component_type: str, service_type: str) -> Dict:
        _SF = {
            ("valve", "gas"): {
                "leak_ef_decimal": Decimal("0.02680"),
                "no_leak_ef_decimal": Decimal("0.00006"),
            },
            ("connector", "gas"): {
                "leak_ef_decimal": Decimal("0.01130"),
                "no_leak_ef_decimal": Decimal("0.00020"),
            },
        }
        key = (component_type.lower(), service_type.lower())
        return _SF.get(key)

    def _correlation(ct: str, st: str) -> Dict:
        return {
            "a_decimal": Decimal("-6.36040"),
            "b_decimal": Decimal("0.79690"),
            "default_zero_ef_decimal": Decimal("0.000020"),
        }

    def _coal(rank: str) -> Dict:
        _COAL = {
            "BITUMINOUS": Decimal("10"),
            "ANTHRACITE": Decimal("18"),
        }
        ef = _COAL.get(rank.upper())
        if ef is None:
            return None
        return {"ef_m3_per_tonne_decimal": ef}

    def _wastewater(tt: str) -> Dict:
        _WW = {
            "ANAEROBIC_LAGOON_DEEP": Decimal("0.8"),
            "SEPTIC_SYSTEM": Decimal("0.5"),
        }
        mcf = _WW.get(tt.upper())
        if mcf is None:
            return None
        return {
            "bo_decimal": Decimal("0.25"),
            "mcf_decimal": mcf,
            "n2o_ef_decimal": Decimal("0.005"),
        }

    def _pneumatic(dt: str) -> Dict:
        _PN = {
            "high_bleed": Decimal("37.8"),
            "low_bleed": Decimal("0.9440"),
        }
        rate = _PN.get(dt.lower())
        if rate is None:
            return None
        return {"rate_m3_per_day_decimal": rate}

    def _gwp(gas: str, source: str = None) -> Dict:
        _GWP = {
            ("CH4", "AR6"): Decimal("27.9"),
            ("CO2", "AR6"): Decimal("1"),
            ("N2O", "AR6"): Decimal("273"),
        }
        key = (gas.upper(), (source or "AR6").upper())
        gwp = _GWP.get(key, _ONE)
        return {"gwp_decimal": gwp}

    def _mole_fraction(species: str) -> Decimal:
        _MF = {"CH4": Decimal("0.950"), "CO2": Decimal("0.010")}
        return _MF.get(species, _ZERO)

    db.get_component_ef.side_effect = _component_ef
    db.get_screening_factor.side_effect = _screening_factor
    db.get_correlation_coefficients.side_effect = _correlation
    db.get_coal_methane_factor.side_effect = _coal
    db.get_wastewater_factor.side_effect = _wastewater
    db.get_pneumatic_rate.side_effect = _pneumatic
    db.get_gwp.side_effect = _gwp
    db.get_mole_fraction.side_effect = _mole_fraction
    return db


@pytest.fixture
def calc() -> EmissionCalculatorEngine:
    """Create an EmissionCalculatorEngine with mocked source database."""
    db = _build_mock_source_db()
    return EmissionCalculatorEngine(source_database=db)


@pytest.fixture
def calc_real():
    """Create an EmissionCalculatorEngine with real source database."""
    from greenlang.fugitive_emissions.fugitive_source_database import (
        FugitiveSourceDatabaseEngine,
    )
    db = FugitiveSourceDatabaseEngine()
    return EmissionCalculatorEngine(source_database=db)


# ===========================================================================
# TestAverageEFMethod - 15 tests
# ===========================================================================


class TestAverageEFMethod:
    """Test Average Emission Factor calculation method."""

    def test_valve_gas_100_components_known_value(self, calc):
        """100 valves * 0.00597 kg/hr * 8760 hr * 0.95 = 49691.34 kg TOC * 0.95."""
        result = calc.calculate({
            "method": "AVERAGE_EMISSION_FACTOR",
            "component_type": "valve", "service_type": "gas",
            "component_count": 100, "operating_hours": 8760,
            "gas_fraction": 0.95,
        })
        assert result["status"] == "SUCCESS"
        ch4_entry = next(
            g for g in result["emissions_by_gas"] if g["gas"] == "CH4"
        )
        # 100 * 0.00597 * 8760 = 52297.2 kg TOC, * 0.95 = 49682.34
        toc_kg = Decimal("100") * Decimal("0.00597") * Decimal("8760")
        expected_ch4 = _quantize(toc_kg * Decimal("0.95"))
        assert Decimal(ch4_entry["emission_kg"]) == expected_ch4

    def test_valve_gas_52_3_tonnes_known_value(self, calc):
        """100 valves * 0.00597 kg/hr * 8760 hr = 52.2972 tonnes TOC."""
        result = calc.calculate_average_ef({
            "component_type": "valve", "service_type": "gas",
            "component_count": 100, "operating_hours": 8760,
            "gas_fraction": 1.0,
        })
        toc_kg = result["raw_emissions_by_gas"]["CH4"]
        # 100 * 0.00597 * 8760 = 52297.2 kg
        expected = _quantize(Decimal("100") * Decimal("0.00597") * Decimal("8760"))
        assert toc_kg == expected

    def test_result_contains_provenance_hash(self, calc):
        result = calc.calculate({
            "method": "AVERAGE_EMISSION_FACTOR",
            "component_type": "valve", "service_type": "gas",
            "component_count": 50,
        })
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_result_contains_calculation_id(self, calc):
        result = calc.calculate({
            "method": "AVERAGE_EMISSION_FACTOR",
            "component_type": "valve", "service_type": "gas",
            "component_count": 10,
        })
        assert result["calculation_id"].startswith("calc_")

    def test_result_contains_processing_time(self, calc):
        result = calc.calculate({
            "method": "AVERAGE_EMISSION_FACTOR",
            "component_type": "valve", "service_type": "gas",
            "component_count": 10,
        })
        assert result["processing_time_ms"] >= 0

    def test_gwp_applied_to_ch4(self, calc):
        result = calc.calculate({
            "method": "AVERAGE_EMISSION_FACTOR",
            "component_type": "valve", "service_type": "gas",
            "component_count": 1, "operating_hours": 1,
            "gas_fraction": 1.0, "gwp_source": "AR6",
        })
        ch4 = next(g for g in result["emissions_by_gas"] if g["gas"] == "CH4")
        # CO2e = emission * GWP(27.9)
        assert Decimal(ch4["co2e_kg"]) > Decimal(ch4["emission_kg"])

    def test_pump_light_liquid(self, calc):
        result = calc.calculate({
            "method": "AVERAGE_EMISSION_FACTOR",
            "component_type": "pump", "service_type": "light_liquid",
            "component_count": 25, "operating_hours": 8760,
            "gas_fraction": 0.90,
        })
        assert result["status"] == "SUCCESS"

    def test_compressor_gas(self, calc):
        result = calc.calculate({
            "method": "AVERAGE_EMISSION_FACTOR",
            "component_type": "compressor", "service_type": "gas",
            "component_count": 5, "gas_fraction": 0.95,
        })
        assert result["status"] == "SUCCESS"

    def test_missing_component_type_raises(self, calc):
        result = calc.calculate({
            "method": "AVERAGE_EMISSION_FACTOR",
            "service_type": "gas", "component_count": 10,
        })
        assert result["status"] == "ERROR"
        assert "component_type" in result["error"]

    def test_missing_service_type_raises(self, calc):
        result = calc.calculate({
            "method": "AVERAGE_EMISSION_FACTOR",
            "component_type": "valve", "component_count": 10,
        })
        assert result["status"] == "ERROR"

    def test_zero_component_count_raises(self, calc):
        result = calc.calculate({
            "method": "AVERAGE_EMISSION_FACTOR",
            "component_type": "valve", "service_type": "gas",
            "component_count": 0,
        })
        assert result["status"] == "ERROR"

    def test_default_operating_hours_8760(self, calc):
        result = calc.calculate_average_ef({
            "component_type": "valve", "service_type": "gas",
            "component_count": 1, "gas_fraction": 1.0,
        })
        assert result["details"]["operating_hours"] == "8760"

    def test_custom_gas_fraction(self, calc):
        result = calc.calculate_average_ef({
            "component_type": "valve", "service_type": "gas",
            "component_count": 1, "gas_fraction": 0.80,
        })
        assert result["details"]["gas_fraction_ch4"] == "0.80"

    def test_invalid_gas_fraction_raises(self, calc):
        result = calc.calculate({
            "method": "AVERAGE_EMISSION_FACTOR",
            "component_type": "valve", "service_type": "gas",
            "component_count": 10, "gas_fraction": 1.5,
        })
        assert result["status"] == "ERROR"

    def test_invalid_method_raises(self, calc):
        result = calc.calculate({"method": "NONEXISTENT_METHOD"})
        assert result["status"] == "ERROR"
        assert "Unknown calculation method" in result["error"]


# ===========================================================================
# TestScreeningMethod - 10 tests
# ===========================================================================


class TestScreeningMethod:
    """Test Screening Ranges calculation method."""

    def test_basic_screening_calculation(self, calc):
        result = calc.calculate({
            "method": "SCREENING_RANGES",
            "component_type": "valve", "service_type": "gas",
            "leak_count": 5, "no_leak_count": 95,
            "operating_hours": 8760, "gas_fraction": 0.95,
        })
        assert result["status"] == "SUCCESS"

    def test_screening_leak_vs_no_leak(self, calc):
        """Verify leak_count drives higher emissions than no_leak_count."""
        result_high_leak = calc.calculate_screening({
            "component_type": "valve", "service_type": "gas",
            "leak_count": 50, "no_leak_count": 50,
            "gas_fraction": 1.0,
        })
        result_low_leak = calc.calculate_screening({
            "component_type": "valve", "service_type": "gas",
            "leak_count": 5, "no_leak_count": 95,
            "gas_fraction": 1.0,
        })
        high_ch4 = result_high_leak["raw_emissions_by_gas"]["CH4"]
        low_ch4 = result_low_leak["raw_emissions_by_gas"]["CH4"]
        assert high_ch4 > low_ch4

    def test_screening_known_values(self, calc):
        """5 leakers * 0.0268 + 95 no-leak * 0.00006 = 0.1397 kg/hr TOC."""
        result = calc.calculate_screening({
            "component_type": "valve", "service_type": "gas",
            "leak_count": 5, "no_leak_count": 95,
            "operating_hours": 8760, "gas_fraction": 1.0,
        })
        details = result["details"]
        leak_em = Decimal(details["leak_emission_kg"])
        no_leak_em = Decimal(details["no_leak_emission_kg"])
        expected_leak = _quantize(Decimal("5") * Decimal("0.02680") * Decimal("8760"))
        assert leak_em == expected_leak

    def test_screening_missing_component_type_error(self, calc):
        result = calc.calculate({
            "method": "SCREENING_RANGES",
            "service_type": "gas", "leak_count": 5, "no_leak_count": 95,
        })
        assert result["status"] == "ERROR"

    def test_screening_negative_counts_error(self, calc):
        result = calc.calculate({
            "method": "SCREENING_RANGES",
            "component_type": "valve", "service_type": "gas",
            "leak_count": -1, "no_leak_count": 100,
        })
        assert result["status"] == "ERROR"

    def test_screening_zero_total_count_error(self, calc):
        result = calc.calculate({
            "method": "SCREENING_RANGES",
            "component_type": "valve", "service_type": "gas",
            "leak_count": 0, "no_leak_count": 0,
        })
        assert result["status"] == "ERROR"

    def test_screening_all_leakers(self, calc):
        result = calc.calculate_screening({
            "component_type": "valve", "service_type": "gas",
            "leak_count": 100, "no_leak_count": 0,
            "gas_fraction": 1.0,
        })
        ch4 = result["raw_emissions_by_gas"]["CH4"]
        assert ch4 > _ZERO

    def test_screening_all_no_leakers(self, calc):
        result = calc.calculate_screening({
            "component_type": "valve", "service_type": "gas",
            "leak_count": 0, "no_leak_count": 100,
            "gas_fraction": 1.0,
        })
        ch4 = result["raw_emissions_by_gas"]["CH4"]
        assert ch4 > _ZERO  # no-leak EF is small but non-zero

    def test_screening_formula_in_details(self, calc):
        result = calc.calculate_screening({
            "component_type": "valve", "service_type": "gas",
            "leak_count": 5, "no_leak_count": 95,
        })
        assert "formula" in result["details"]

    def test_screening_threshold_in_details(self, calc):
        result = calc.calculate_screening({
            "component_type": "valve", "service_type": "gas",
            "leak_count": 5, "no_leak_count": 95,
        })
        assert result["details"]["threshold_ppmv"] == 10000


# ===========================================================================
# TestCorrelationMethod - 8 tests
# ===========================================================================


class TestCorrelationMethod:
    """Test Correlation Equation calculation method."""

    def test_basic_correlation(self, calc):
        result = calc.calculate({
            "method": "CORRELATION_EQUATION",
            "component_type": "valve", "service_type": "gas",
            "screening_values_ppmv": [1000, 5000, 50000],
            "gas_fraction": 0.95,
        })
        assert result["status"] == "SUCCESS"

    def test_correlation_higher_ppmv_higher_emission(self, calc):
        result_low = calc.calculate_correlation({
            "component_type": "valve", "service_type": "gas",
            "screening_values_ppmv": [100],
            "gas_fraction": 1.0,
        })
        result_high = calc.calculate_correlation({
            "component_type": "valve", "service_type": "gas",
            "screening_values_ppmv": [100000],
            "gas_fraction": 1.0,
        })
        low_ch4 = result_low["raw_emissions_by_gas"]["CH4"]
        high_ch4 = result_high["raw_emissions_by_gas"]["CH4"]
        assert high_ch4 > low_ch4

    def test_correlation_zero_ppmv_uses_default(self, calc):
        result = calc.calculate_correlation({
            "component_type": "valve", "service_type": "gas",
            "screening_values_ppmv": [0],
            "gas_fraction": 1.0,
        })
        ch4 = result["raw_emissions_by_gas"]["CH4"]
        assert ch4 > _ZERO

    def test_correlation_dict_format_input(self, calc):
        result = calc.calculate({
            "method": "CORRELATION_EQUATION",
            "component_type": "valve", "service_type": "gas",
            "screening_values": [{"ppmv": 10000}, {"ppmv": 50000}],
            "gas_fraction": 0.95,
        })
        assert result["status"] == "SUCCESS"

    def test_correlation_missing_values_error(self, calc):
        result = calc.calculate({
            "method": "CORRELATION_EQUATION",
            "component_type": "valve", "service_type": "gas",
        })
        assert result["status"] == "ERROR"

    def test_correlation_component_count_in_details(self, calc):
        result = calc.calculate_correlation({
            "component_type": "valve", "service_type": "gas",
            "screening_values_ppmv": [100, 200, 300],
            "gas_fraction": 1.0,
        })
        assert result["details"]["component_count"] == 3

    def test_correlation_formula_in_details(self, calc):
        result = calc.calculate_correlation({
            "component_type": "valve", "service_type": "gas",
            "screening_values_ppmv": [1000],
        })
        assert "formula" in result["details"]

    def test_correlation_equation_math(self, calc):
        """Verify: log10(kg/hr) = -6.36040 + 0.79690 * log10(10000)."""
        result = calc.calculate_correlation({
            "component_type": "valve", "service_type": "gas",
            "screening_values_ppmv": [10000],
            "operating_hours": 1, "gas_fraction": 1.0,
        })
        # log10(10000) = 4.0
        # log_rate = -6.36040 + 0.79690 * 4.0 = -6.36040 + 3.18760 = -3.17280
        # rate = 10^(-3.17280) ~ 0.000671 kg/hr
        ch4 = result["raw_emissions_by_gas"]["CH4"]
        assert Decimal("0.0005") < ch4 < Decimal("0.001")


# ===========================================================================
# TestEngineeringMethod - 10 tests
# ===========================================================================


class TestEngineeringMethod:
    """Test Engineering Estimate sub-methods."""

    # --- Pneumatic ---

    def test_pneumatic_high_bleed(self, calc):
        result = calc.calculate({
            "method": "ENGINEERING_ESTIMATE",
            "engineering_type": "PNEUMATIC",
            "device_type": "high_bleed", "device_count": 10,
            "operating_hours": 8760, "gas_fraction": 0.95,
        })
        assert result["status"] == "SUCCESS"
        ch4 = next(g for g in result["emissions_by_gas"] if g["gas"] == "CH4")
        assert Decimal(ch4["emission_kg"]) > _ZERO

    def test_pneumatic_zero_count_error(self, calc):
        result = calc.calculate({
            "method": "ENGINEERING_ESTIMATE",
            "engineering_type": "PNEUMATIC",
            "device_type": "high_bleed", "device_count": 0,
        })
        assert result["status"] == "ERROR"

    def test_pneumatic_formula_in_details(self, calc):
        result = calc.calculate({
            "method": "ENGINEERING_ESTIMATE",
            "engineering_type": "PNEUMATIC",
            "device_type": "low_bleed", "device_count": 5,
        })
        assert "formula" in result["calculation_details"]

    # --- Tank Loss ---

    def test_tank_loss_pass_through(self, calc):
        result = calc.calculate({
            "method": "ENGINEERING_ESTIMATE",
            "engineering_type": "TANK_LOSS",
            "annual_loss_kg": 500.0, "gas_fraction": 0.10,
        })
        assert result["status"] == "SUCCESS"

    def test_tank_loss_negative_error(self, calc):
        result = calc.calculate({
            "method": "ENGINEERING_ESTIMATE",
            "engineering_type": "TANK_LOSS",
            "annual_loss_kg": -100.0,
        })
        assert result["status"] == "ERROR"

    # --- Coal Mine ---

    def test_coal_mine_bituminous(self, calc):
        result = calc.calculate({
            "method": "ENGINEERING_ESTIMATE",
            "engineering_type": "COAL_MINE",
            "coal_production_tonnes": 10000,
            "coal_rank": "BITUMINOUS",
            "recovery_fraction": 0.0,
        })
        assert result["status"] == "SUCCESS"

    def test_coal_mine_with_recovery(self, calc):
        result_no_recovery = calc.calculate({
            "method": "ENGINEERING_ESTIMATE",
            "engineering_type": "COAL_MINE",
            "coal_production_tonnes": 10000,
            "coal_rank": "BITUMINOUS",
            "recovery_fraction": 0.0,
        })
        result_with_recovery = calc.calculate({
            "method": "ENGINEERING_ESTIMATE",
            "engineering_type": "COAL_MINE",
            "coal_production_tonnes": 10000,
            "coal_rank": "BITUMINOUS",
            "recovery_fraction": 0.5,
        })
        co2e_no = Decimal(result_no_recovery["total_co2e_kg"])
        co2e_yes = Decimal(result_with_recovery["total_co2e_kg"])
        assert co2e_yes < co2e_no

    # --- Wastewater ---

    def test_wastewater_ch4_calculation(self, calc):
        result = calc.calculate({
            "method": "ENGINEERING_ESTIMATE",
            "engineering_type": "WASTEWATER",
            "bod_load_kg": 5000,
            "treatment_type": "ANAEROBIC_LAGOON_DEEP",
            "recovery_fraction": 0.0,
        })
        assert result["status"] == "SUCCESS"

    def test_wastewater_with_n2o(self, calc):
        result = calc.calculate({
            "method": "ENGINEERING_ESTIMATE",
            "engineering_type": "WASTEWATER",
            "bod_load_kg": 5000,
            "treatment_type": "ANAEROBIC_LAGOON_DEEP",
            "nitrogen_load_kg": 200,
        })
        assert result["status"] == "SUCCESS"
        gases = {g["gas"] for g in result["emissions_by_gas"]}
        assert "CH4" in gases
        assert "N2O" in gases

    def test_engineering_invalid_submethod_error(self, calc):
        result = calc.calculate({
            "method": "ENGINEERING_ESTIMATE",
            "engineering_type": "NONEXISTENT",
        })
        assert result["status"] == "ERROR"


# ===========================================================================
# TestDirectMeasurement - 3 tests
# ===========================================================================


class TestDirectMeasurement:
    """Test Direct Measurement method."""

    def test_direct_ch4_only(self, calc):
        result = calc.calculate({
            "method": "DIRECT_MEASUREMENT",
            "measured_ch4_kg": 150.5,
        })
        assert result["status"] == "SUCCESS"
        ch4 = next(g for g in result["emissions_by_gas"] if g["gas"] == "CH4")
        assert Decimal(ch4["emission_kg"]) == _quantize(Decimal("150.5"))

    def test_direct_multiple_gases(self, calc):
        result = calc.calculate({
            "method": "DIRECT_MEASUREMENT",
            "measured_ch4_kg": 100, "measured_co2_kg": 50,
            "measured_n2o_kg": 5,
        })
        assert result["status"] == "SUCCESS"
        gases = {g["gas"] for g in result["emissions_by_gas"]}
        assert gases == {"CH4", "CO2", "N2O"}

    def test_direct_no_values_error(self, calc):
        result = calc.calculate({
            "method": "DIRECT_MEASUREMENT",
            "measured_ch4_kg": 0, "measured_co2_kg": 0,
        })
        assert result["status"] == "ERROR"


# ===========================================================================
# TestBatch - 4 tests
# ===========================================================================


class TestBatch:
    """Test batch calculation processing."""

    def test_batch_success(self, calc):
        records = [
            {"method": "AVERAGE_EMISSION_FACTOR",
             "component_type": "valve", "service_type": "gas",
             "component_count": 100},
            {"method": "DIRECT_MEASUREMENT", "measured_ch4_kg": 50.0},
        ]
        result = calc.calculate_batch(records)
        assert result["successful"] == 2
        assert result["failed"] == 0
        assert result["total_records"] == 2
        assert "provenance_hash" in result

    def test_batch_partial_failure(self, calc):
        records = [
            {"method": "AVERAGE_EMISSION_FACTOR",
             "component_type": "valve", "service_type": "gas",
             "component_count": 100},
            {"method": "NONEXISTENT_METHOD"},  # will fail
        ]
        result = calc.calculate_batch(records, continue_on_error=True)
        assert result["successful"] == 1
        assert result["failed"] == 1

    def test_batch_stop_on_error(self, calc):
        records = [
            {"method": "NONEXISTENT_METHOD"},
            {"method": "DIRECT_MEASUREMENT", "measured_ch4_kg": 50.0},
        ]
        result = calc.calculate_batch(records, continue_on_error=False)
        assert result["failed"] >= 1
        # Second record may not have been processed
        assert len(result["results"]) <= 2

    def test_batch_aggregates_co2e(self, calc):
        records = [
            {"method": "DIRECT_MEASUREMENT", "measured_ch4_kg": 100.0},
            {"method": "DIRECT_MEASUREMENT", "measured_ch4_kg": 200.0},
        ]
        result = calc.calculate_batch(records)
        total_co2e = Decimal(result["summary"]["total_co2e_kg"])
        assert total_co2e > _ZERO
