# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-011 HeatCoolingCalculatorEngine (Engine 3).

Tests district heating with regional EFs, electric cooling with COP,
absorption cooling, free cooling, distribution loss, storage loss,
GJ-to-kWh conversion, technology comparison, batch processing,
provenance hash, calculation trace, singleton, and reset.

Target: 80 tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

import pytest

from greenlang.agents.mrv.steam_heat_purchase.heat_cooling_calculator import (
    HeatCoolingCalculatorEngine,
    HeatingCoolingType,
    CoolingCategory,
    TraceStep,
)


# ===========================================================================
# Precision helpers (match engine internals)
# ===========================================================================

_PRECISION = Decimal("0.00000001")
_ZERO = Decimal("0")
_ONE = Decimal("1")
_THOUSAND = Decimal("1000")
_GJ_TO_KWH = Decimal("277.778")


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_heat_cooling_singleton():
    """Reset the HeatCoolingCalculatorEngine singleton."""
    HeatCoolingCalculatorEngine.reset()
    yield
    HeatCoolingCalculatorEngine.reset()


@pytest.fixture
def calc() -> HeatCoolingCalculatorEngine:
    """Return a fresh HeatCoolingCalculatorEngine instance."""
    return HeatCoolingCalculatorEngine()


# ===========================================================================
# Singleton Tests
# ===========================================================================


class TestHeatCoolingSingleton:
    """Tests for the singleton pattern."""

    def test_singleton_same_instance(self):
        """Multiple instantiations return the same object."""
        c1 = HeatCoolingCalculatorEngine()
        c2 = HeatCoolingCalculatorEngine()
        assert c1 is c2

    def test_reset_creates_new_instance(self):
        """After reset, a new instance is created."""
        c1 = HeatCoolingCalculatorEngine()
        HeatCoolingCalculatorEngine.reset()
        c2 = HeatCoolingCalculatorEngine()
        assert c1 is not c2


# ===========================================================================
# Enumerations Tests
# ===========================================================================


class TestEnumerations:
    """Tests for HeatingCoolingType and CoolingCategory enums."""

    def test_heating_cooling_types(self):
        """HeatingCoolingType has expected members."""
        assert HeatingCoolingType.DISTRICT_HEATING == "DISTRICT_HEATING"
        assert HeatingCoolingType.ELECTRIC_COOLING == "ELECTRIC_COOLING"
        assert HeatingCoolingType.ABSORPTION_COOLING == "ABSORPTION_COOLING"
        assert HeatingCoolingType.FREE_COOLING == "FREE_COOLING"

    def test_cooling_categories(self):
        """CoolingCategory has expected members."""
        assert CoolingCategory.ELECTRIC == "ELECTRIC"
        assert CoolingCategory.ABSORPTION == "ABSORPTION"
        assert CoolingCategory.FREE == "FREE"
        assert CoolingCategory.STORAGE == "STORAGE"


# ===========================================================================
# TraceStep Tests
# ===========================================================================


class TestTraceStep:
    """Tests for TraceStep dataclass."""

    def test_trace_step_to_dict(self):
        """TraceStep.to_dict returns correct dictionary."""
        step = TraceStep(
            step_number=1,
            description="Test step",
            formula="A = B + C",
            inputs={"B": "100", "C": "200"},
            output="300",
            unit="kgCO2e",
        )
        d = step.to_dict()
        assert d["step_number"] == 1
        assert d["description"] == "Test step"
        assert d["formula"] == "A = B + C"
        assert d["output"] == "300"
        assert d["unit"] == "kgCO2e"


# ===========================================================================
# District Heating Tests
# ===========================================================================


class TestDistrictHeating:
    """Tests for calculate_district_heating method."""

    def test_germany_dh_basic(self, calc):
        """Germany DH: 500 GJ, 72 kgCO2e/GJ, 12% loss.
        Adjusted = 500 / (1 - 0.12) = 568.18 GJ
        Emissions = 568.18 x 72 = 40,909 kgCO2e
        """
        result = calc.calculate_district_heating(
            consumption_gj="500",
            region="germany",
        )
        assert result["status"] == "SUCCESS"
        adjusted = (Decimal("500") / (Decimal("1") - Decimal("0.12"))).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        expected_emissions = (adjusted * Decimal("72")).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        assert Decimal(result["emissions_kgco2e"]) == pytest.approx(
            expected_emissions, rel=Decimal("0.01"),
        )

    def test_denmark_dh(self, calc):
        """Denmark DH: 1000 GJ, 36 kgCO2e/GJ, 10% loss."""
        result = calc.calculate_district_heating(
            consumption_gj="1000",
            region="denmark",
        )
        assert result["status"] == "SUCCESS"
        adjusted = (Decimal("1000") / Decimal("0.90")).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        expected = (adjusted * Decimal("36")).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        assert Decimal(result["emissions_kgco2e"]) == pytest.approx(
            expected, rel=Decimal("0.01"),
        )

    def test_dh_with_supplier_ef_override(self, calc):
        """Supplier-specific EF overrides regional default."""
        result = calc.calculate_district_heating(
            consumption_gj="500",
            region="germany",
            supplier_ef="50.0",
        )
        assert result["status"] == "SUCCESS"
        assert result["ef_source"] == "supplier_specific"
        # With supplier_ef=50, should be different from default 72
        adjusted = (Decimal("500") / (Decimal("1") - Decimal("0.12"))).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        expected = (adjusted * Decimal("50")).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        assert Decimal(result["emissions_kgco2e"]) == pytest.approx(
            expected, rel=Decimal("0.01"),
        )

    def test_dh_with_custom_distribution_loss(self, calc):
        """Custom distribution loss override."""
        result = calc.calculate_district_heating(
            consumption_gj="1000",
            region="germany",
            distribution_loss_pct="0.05",
        )
        assert result["status"] == "SUCCESS"
        adjusted = (Decimal("1000") / Decimal("0.95")).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        expected = (adjusted * Decimal("72")).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        assert Decimal(result["emissions_kgco2e"]) == pytest.approx(
            expected, rel=Decimal("0.01"),
        )

    def test_dh_unknown_region_uses_global_default(self, calc):
        """Unknown region falls back to global_default (EF=70, loss=0.12)."""
        result = calc.calculate_district_heating(
            consumption_gj="500",
            region="mars",
        )
        assert result["status"] == "SUCCESS"

    def test_dh_none_consumption_raises(self, calc):
        """None consumption returns error result."""
        result = calc.calculate_district_heating(
            consumption_gj=None,
            region="germany",
        )
        assert result["status"] == "ERROR"

    def test_dh_zero_consumption_raises(self, calc):
        """Zero consumption returns error result."""
        result = calc.calculate_district_heating(
            consumption_gj="0",
            region="germany",
        )
        assert result["status"] == "ERROR"

    def test_dh_negative_consumption_raises(self, calc):
        """Negative consumption returns error result."""
        result = calc.calculate_district_heating(
            consumption_gj="-100",
            region="germany",
        )
        assert result["status"] == "ERROR"

    def test_dh_result_has_provenance_hash(self, calc):
        """District heating result includes provenance hash."""
        result = calc.calculate_district_heating(
            consumption_gj="500",
            region="germany",
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_dh_result_has_trace_steps(self, calc):
        """District heating result includes trace_steps."""
        result = calc.calculate_district_heating(
            consumption_gj="500",
            region="germany",
        )
        assert "trace_steps" in result
        assert len(result["trace_steps"]) > 0

    def test_dh_result_has_emissions_tco2e(self, calc):
        """District heating result includes emissions_tco2e."""
        result = calc.calculate_district_heating(
            consumption_gj="500",
            region="germany",
        )
        assert "emissions_tco2e" in result
        kgco2e = Decimal(result["emissions_kgco2e"])
        tco2e = Decimal(result["emissions_tco2e"])
        expected_tco2e = (kgco2e / _THOUSAND).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        assert tco2e == expected_tco2e

    def test_dh_calculation_type_is_district_heating(self, calc):
        """Result has calculation_type DISTRICT_HEATING."""
        result = calc.calculate_district_heating(
            consumption_gj="500",
            region="denmark",
        )
        assert result["calculation_type"] == "DISTRICT_HEATING"


# ===========================================================================
# Electric Cooling Tests
# ===========================================================================


class TestElectricCooling:
    """Tests for calculate_electric_cooling method."""

    def test_electric_cooling_basic(self, calc):
        """300 GJ / COP 6.0 = 50 GJ elec -> 50 x 277.778 = 13,889 kWh.
        Emissions = 13,889 x 0.450 = 6,250 kgCO2e (with default grid EF).
        """
        result = calc.calculate_electric_cooling(
            cooling_output_gj="300",
            technology="centrifugal_chiller",
            cop="6.0",
            grid_ef_kwh="0.450",
        )
        assert result["status"] == "SUCCESS"
        elec_gj = (Decimal("300") / Decimal("6.0")).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        elec_kwh = (elec_gj * _GJ_TO_KWH).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        expected = (elec_kwh * Decimal("0.450")).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        assert Decimal(result["emissions_kgco2e"]) == pytest.approx(
            expected, rel=Decimal("0.01"),
        )

    def test_electric_cooling_custom_grid_ef(self, calc):
        """Custom grid EF of 0.436 kgCO2e/kWh."""
        result = calc.calculate_electric_cooling(
            cooling_output_gj="300",
            technology="centrifugal_chiller",
            cop="6.0",
            grid_ef_kwh="0.436",
        )
        assert result["status"] == "SUCCESS"
        elec_gj = Decimal("300") / Decimal("6.0")
        elec_kwh = (elec_gj * _GJ_TO_KWH).quantize(_PRECISION, rounding=ROUND_HALF_UP)
        expected = (elec_kwh * Decimal("0.436")).quantize(_PRECISION, rounding=ROUND_HALF_UP)
        assert Decimal(result["emissions_kgco2e"]) == pytest.approx(
            expected, rel=Decimal("0.01"),
        )

    def test_electric_cooling_screw_chiller(self, calc):
        """Screw chiller with COP 4.5."""
        result = calc.calculate_electric_cooling(
            cooling_output_gj="200",
            technology="screw_chiller",
        )
        assert result["status"] == "SUCCESS"
        assert result["technology"] == "screw_chiller"

    def test_electric_cooling_none_output_error(self, calc):
        """None cooling output returns error."""
        result = calc.calculate_electric_cooling(
            cooling_output_gj=None,
            technology="centrifugal_chiller",
        )
        assert result["status"] == "ERROR"

    def test_electric_cooling_zero_output_error(self, calc):
        """Zero cooling output returns error."""
        result = calc.calculate_electric_cooling(
            cooling_output_gj="0",
            technology="centrifugal_chiller",
        )
        assert result["status"] == "ERROR"

    def test_electric_cooling_has_provenance_hash(self, calc):
        """Electric cooling result includes provenance hash."""
        result = calc.calculate_electric_cooling(
            cooling_output_gj="300",
            technology="centrifugal_chiller",
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_electric_cooling_has_trace(self, calc):
        """Electric cooling result has trace_steps."""
        result = calc.calculate_electric_cooling(
            cooling_output_gj="300",
            technology="centrifugal_chiller",
        )
        assert "trace_steps" in result
        assert len(result["trace_steps"]) > 0


# ===========================================================================
# Absorption Cooling Tests
# ===========================================================================


class TestAbsorptionCooling:
    """Tests for calculate_absorption_cooling method."""

    def test_absorption_cooling_basic(self, calc):
        """300 GJ / COP 1.2 = 250 GJ heat -> 250 x heat_ef."""
        result = calc.calculate_absorption_cooling(
            cooling_output_gj="300",
            technology="absorption_double",
            cop="1.2",
            heat_source_ef="66.0",
        )
        assert result["status"] == "SUCCESS"
        heat_gj = (Decimal("300") / Decimal("1.2")).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        expected = (heat_gj * Decimal("66.0")).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        assert Decimal(result["emissions_kgco2e"]) == pytest.approx(
            expected, rel=Decimal("0.01"),
        )

    def test_absorption_single_effect(self, calc):
        """Single effect absorption chiller with COP 0.7."""
        result = calc.calculate_absorption_cooling(
            cooling_output_gj="200",
            technology="absorption_single",
        )
        assert result["status"] == "SUCCESS"

    def test_absorption_triple_effect(self, calc):
        """Triple effect absorption chiller with COP 1.6."""
        result = calc.calculate_absorption_cooling(
            cooling_output_gj="200",
            technology="absorption_triple",
        )
        assert result["status"] == "SUCCESS"

    def test_absorption_none_output_error(self, calc):
        """None cooling output returns error."""
        result = calc.calculate_absorption_cooling(
            cooling_output_gj=None,
        )
        assert result["status"] == "ERROR"

    def test_absorption_has_provenance_hash(self, calc):
        """Absorption cooling result includes provenance hash."""
        result = calc.calculate_absorption_cooling(
            cooling_output_gj="300",
            technology="absorption_double",
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# Free Cooling Tests
# ===========================================================================


class TestFreeCooling:
    """Tests for calculate_free_cooling method."""

    def test_free_cooling_basic(self, calc):
        """300 GJ x 277.778 / COP 20 = 4167 kWh -> x grid_ef."""
        result = calc.calculate_free_cooling(
            cooling_output_gj="300",
            cop="20",
            grid_ef_kwh="0.450",
        )
        assert result["status"] == "SUCCESS"
        cooling_kwh = (Decimal("300") * _GJ_TO_KWH).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        pump_kwh = (cooling_kwh / Decimal("20")).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        expected = (pump_kwh * Decimal("0.450")).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        assert Decimal(result["emissions_kgco2e"]) == pytest.approx(
            expected, rel=Decimal("0.01"),
        )

    def test_free_cooling_uses_default_cop_20(self, calc):
        """Without COP override, free cooling uses default COP 20."""
        result = calc.calculate_free_cooling(
            cooling_output_gj="100",
        )
        assert result["status"] == "SUCCESS"
        assert Decimal(result["cop"]) == Decimal("20.0")

    def test_free_cooling_none_output_error(self, calc):
        """None cooling output returns error."""
        result = calc.calculate_free_cooling(
            cooling_output_gj=None,
        )
        assert result["status"] == "ERROR"

    def test_free_cooling_low_emissions(self, calc):
        """Free cooling has very low emissions due to high COP."""
        result_free = calc.calculate_free_cooling(
            cooling_output_gj="300",
            grid_ef_kwh="0.450",
        )
        # Compare against electric cooling at same output
        HeatCoolingCalculatorEngine.reset()
        calc2 = HeatCoolingCalculatorEngine()
        result_elec = calc2.calculate_electric_cooling(
            cooling_output_gj="300",
            technology="centrifugal_chiller",
            grid_ef_kwh="0.450",
        )
        # Free cooling should produce much less emissions than electric
        assert Decimal(result_free["emissions_kgco2e"]) < Decimal(result_elec["emissions_kgco2e"])


# ===========================================================================
# Distribution Loss Tests
# ===========================================================================


class TestDistributionLoss:
    """Tests for apply_distribution_loss method."""

    def test_distribution_loss_12_percent(self, calc):
        """500 / (1 - 0.12) = 568.18 GJ."""
        adjusted = calc.apply_distribution_loss("500", "0.12")
        expected = (Decimal("500") / Decimal("0.88")).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        assert adjusted == expected

    def test_distribution_loss_zero(self, calc):
        """Zero loss returns original value."""
        adjusted = calc.apply_distribution_loss("500", "0")
        assert adjusted == Decimal("500").quantize(_PRECISION, rounding=ROUND_HALF_UP)

    def test_distribution_loss_100_percent_raises(self, calc):
        """100% loss (loss_pct=1.0) raises ValueError."""
        with pytest.raises(ValueError, match="must be < 1.0"):
            calc.apply_distribution_loss("500", "1.0")

    def test_distribution_loss_greater_than_1_raises(self, calc):
        """Loss percentage >= 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="must be < 1.0"):
            calc.apply_distribution_loss("500", "1.5")


# ===========================================================================
# Storage Loss Tests
# ===========================================================================


class TestStorageLoss:
    """Tests for apply_storage_loss method."""

    def test_storage_loss_5_percent(self, calc):
        """500 x (1 - 0.05) = 475 GJ effective."""
        effective = calc.apply_storage_loss("500", "0.05")
        expected = (Decimal("500") * Decimal("0.95")).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        assert effective == expected

    def test_storage_loss_zero(self, calc):
        """Zero storage loss returns original value."""
        effective = calc.apply_storage_loss("500", "0")
        expected = Decimal("500").quantize(_PRECISION, rounding=ROUND_HALF_UP)
        assert effective == expected

    def test_storage_loss_10_percent(self, calc):
        """10% storage loss."""
        effective = calc.apply_storage_loss("1000", "0.10")
        expected = (Decimal("1000") * Decimal("0.90")).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        assert effective == expected


# ===========================================================================
# GJ to kWh Conversion Tests
# ===========================================================================


class TestGJtoKWhConversion:
    """Tests for convert_gj_to_kwh method."""

    def test_1_gj_to_kwh(self, calc):
        """1 GJ = 277.778 kWh."""
        result = calc.convert_gj_to_kwh("1")
        expected = _GJ_TO_KWH.quantize(_PRECISION, rounding=ROUND_HALF_UP)
        assert result == expected

    def test_10_gj_to_kwh(self, calc):
        """10 GJ = 2777.78 kWh."""
        result = calc.convert_gj_to_kwh("10")
        expected = (Decimal("10") * _GJ_TO_KWH).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        assert result == expected


# ===========================================================================
# Compute Electrical Input Tests
# ===========================================================================


class TestComputeElectricalInput:
    """Tests for compute_electrical_input method."""

    def test_electrical_input_basic(self, calc):
        """300 GJ / COP 6.0 = 50 GJ."""
        result = calc.compute_electrical_input("300", "6.0")
        expected = (Decimal("300") / Decimal("6.0")).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        assert result == expected


# ===========================================================================
# Compute Heat Input Tests
# ===========================================================================


class TestComputeHeatInput:
    """Tests for compute_heat_input method."""

    def test_heat_input_basic(self, calc):
        """300 GJ / COP 1.2 = 250 GJ."""
        result = calc.compute_heat_input("300", "1.2")
        expected = (Decimal("300") / Decimal("1.2")).quantize(
            _PRECISION, rounding=ROUND_HALF_UP,
        )
        assert result == expected


# ===========================================================================
# Heating Emissions (Request-Based) Tests
# ===========================================================================


class TestHeatingEmissionsRequest:
    """Tests for calculate_heating_emissions request-based method."""

    def test_heating_emissions_request(self, calc):
        """Heating emissions via request dict."""
        result = calc.calculate_heating_emissions({
            "consumption_gj": 500,
            "region": "germany",
        })
        assert result["status"] == "SUCCESS"
        assert Decimal(result["emissions_kgco2e"]) > _ZERO

    def test_heating_emissions_request_with_supplier_ef(self, calc):
        """Heating emissions with supplier EF override via request."""
        result = calc.calculate_heating_emissions({
            "consumption_gj": 500,
            "region": "germany",
            "supplier_ef": 50.0,
        })
        assert result["status"] == "SUCCESS"
        assert result["ef_source"] == "supplier_specific"


# ===========================================================================
# Cooling Emissions (Request-Based) Tests
# ===========================================================================


class TestCoolingEmissionsRequest:
    """Tests for calculate_cooling_emissions request-based method."""

    def test_cooling_emissions_electric(self, calc):
        """Cooling emissions via request for electric chiller."""
        result = calc.calculate_cooling_emissions({
            "cooling_output_gj": 200,
            "technology": "centrifugal_chiller",
        })
        assert result["status"] == "SUCCESS"

    def test_cooling_emissions_absorption(self, calc):
        """Cooling emissions via request for absorption chiller."""
        result = calc.calculate_cooling_emissions({
            "cooling_output_gj": 200,
            "technology": "absorption_double",
        })
        assert result["status"] == "SUCCESS"

    def test_cooling_emissions_free(self, calc):
        """Cooling emissions via request for free cooling."""
        result = calc.calculate_cooling_emissions({
            "cooling_output_gj": 200,
            "technology": "free_cooling",
        })
        assert result["status"] == "SUCCESS"


# ===========================================================================
# Health Check Tests
# ===========================================================================


class TestHealthCheck:
    """Tests for health_check method."""

    def test_health_check_returns_healthy(self, calc):
        """health_check returns healthy status."""
        result = calc.health_check()
        assert result["status"] == "healthy"
        assert result["engine"] == "heat_cooling_calculator"
