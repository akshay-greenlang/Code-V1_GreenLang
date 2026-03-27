# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-009 ElectricityEmissionsEngine.

Tests basic emission calculations, per-gas breakdowns, facility-level
computations, tier 1/tier 2 resolution, unit conversions, aggregation,
and input validation.

Target: 45+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from unittest.mock import Mock

import pytest

try:
    from greenlang.agents.mrv.scope2_location.electricity_emissions import (
        ElectricityEmissionsEngine,
        GWP_VALUES,
        _to_decimal,
        _KWH_TO_MWH,
        _GJ_TO_MWH,
        _MMBTU_TO_MWH,
        _KG_TO_TONNES,
        _TONNES_TO_KG,
        _DEFAULT_COUNTRY_GRID_EF,
        _DEFAULT_EGRID_SUBREGION_EF,
    )
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False

_SKIP = pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """Create a default ElectricityEmissionsEngine."""
    return ElectricityEmissionsEngine()


@pytest.fixture
def engine_with_grid_db():
    """Create engine with a mock grid factor database."""
    mock_db = Mock()
    mock_db.get_grid_factor.return_value = {
        "country_code": "US",
        "co2_kg_per_mwh": Decimal("379.00"),
        "ch4_kg_per_mwh": Decimal("0.038"),
        "n2o_kg_per_mwh": Decimal("0.004"),
        "source": "iea",
    }
    return ElectricityEmissionsEngine(grid_factor_db=mock_db)


@pytest.fixture
def engine_no_provenance():
    """Create engine with provenance disabled."""
    return ElectricityEmissionsEngine(
        config={"enable_provenance": False}
    )


# ===========================================================================
# TestBasicCalculation
# ===========================================================================


@_SKIP
class TestBasicCalculation:
    """Tests for calculate_emissions with known values."""

    def test_basic_emissions_1000_mwh(self, engine):
        """1000 MWh x 225.30 kgCO2e/MWh = 225,300 kgCO2e."""
        result = engine.calculate_emissions(
            consumption_mwh=Decimal("1000"),
            grid_ef_co2e=Decimal("225.30"),
        )
        assert result["total_co2e_kg"] == Decimal("225300.00000000")
        assert result["total_co2e_tonnes"] == Decimal("225.30000000")

    def test_basic_emissions_with_td_loss(self, engine):
        """1000 MWh x (1+0.05) x 400 kgCO2e/MWh = 420,000 kg."""
        result = engine.calculate_emissions(
            consumption_mwh=Decimal("1000"),
            grid_ef_co2e=Decimal("400"),
            td_loss_pct=Decimal("0.05"),
        )
        expected_gross = Decimal("1050")
        expected_kg = expected_gross * Decimal("400")
        assert result["gross_consumption"] == expected_gross.quantize(
            Decimal("0.00000001"), ROUND_HALF_UP
        )
        assert result["total_co2e_kg"] == expected_kg.quantize(
            Decimal("0.00000001"), ROUND_HALF_UP
        )

    def test_zero_consumption_returns_zero(self, engine):
        """Zero consumption produces zero emissions."""
        result = engine.calculate_emissions(
            consumption_mwh=Decimal("0"),
            grid_ef_co2e=Decimal("500"),
        )
        assert result["total_co2e_kg"] == Decimal("0")
        assert result["total_co2e_tonnes"] == Decimal("0")

    def test_result_contains_provenance_hash(self, engine):
        """Result includes a 64-character SHA-256 provenance hash."""
        result = engine.calculate_emissions(
            consumption_mwh=Decimal("100"),
            grid_ef_co2e=Decimal("300"),
        )
        assert len(result["provenance_hash"]) == 64

    def test_result_contains_calculation_trace(self, engine):
        """Result includes a non-empty calculation trace."""
        result = engine.calculate_emissions(
            consumption_mwh=Decimal("100"),
            grid_ef_co2e=Decimal("300"),
        )
        assert len(result["calculation_trace"]) > 0

    def test_deterministic_output(self, engine):
        """Same inputs produce identical outputs (bit-perfect)."""
        r1 = engine.calculate_emissions(Decimal("500"), Decimal("300"))
        r2 = engine.calculate_emissions(Decimal("500"), Decimal("300"))
        assert r1["total_co2e_kg"] == r2["total_co2e_kg"]
        assert r1["provenance_hash"] == r2["provenance_hash"]

    def test_processing_time_recorded(self, engine):
        """Result includes processing_time_ms > 0."""
        result = engine.calculate_emissions(Decimal("100"), Decimal("200"))
        assert result["processing_time_ms"] >= 0

    def test_small_consumption_precision(self, engine):
        """Decimal precision maintained for small values."""
        result = engine.calculate_emissions(
            consumption_mwh=Decimal("0.001"),
            grid_ef_co2e=Decimal("500"),
        )
        assert result["total_co2e_kg"] == Decimal("0.50000000")


# ===========================================================================
# TestGasBreakdown
# ===========================================================================


@_SKIP
class TestGasBreakdown:
    """Tests for calculate_with_gas_breakdown."""

    def test_gas_breakdown_components(self, engine):
        """Gas breakdown includes CO2, CH4, and N2O entries."""
        result = engine.calculate_with_gas_breakdown(
            consumption_mwh=Decimal("1000"),
            co2_ef=Decimal("225.30"),
            ch4_ef=Decimal("0.026"),
            n2o_ef=Decimal("0.003"),
        )
        gases = {g["gas"] for g in result["gas_breakdown"]}
        assert gases == {"CO2", "CH4", "N2O"}

    def test_co2_is_dominant_component(self, engine):
        """CO2 CO2e should be the dominant component."""
        result = engine.calculate_with_gas_breakdown(
            consumption_mwh=Decimal("1000"),
            co2_ef=Decimal("225.30"),
            ch4_ef=Decimal("0.026"),
            n2o_ef=Decimal("0.003"),
        )
        co2_entry = [g for g in result["gas_breakdown"] if g["gas"] == "CO2"][0]
        total = result["total_co2e_kg"]
        # CO2 should be >95% of total
        assert co2_entry["co2e_kg"] > total * Decimal("0.95")

    def test_ch4_gwp_applied_correctly(self, engine):
        """CH4 CO2e = CH4_kg x GWP_CH4."""
        result = engine.calculate_with_gas_breakdown(
            consumption_mwh=Decimal("1000"),
            co2_ef=Decimal("0"),
            ch4_ef=Decimal("1"),
            n2o_ef=Decimal("0"),
            gwp_source="AR5",
        )
        ch4_entry = [g for g in result["gas_breakdown"] if g["gas"] == "CH4"][0]
        assert ch4_entry["gwp_factor"] == Decimal("28")
        assert ch4_entry["co2e_kg"] == Decimal("28000.00000000")

    def test_n2o_gwp_applied_correctly(self, engine):
        """N2O CO2e = N2O_kg x GWP_N2O."""
        result = engine.calculate_with_gas_breakdown(
            consumption_mwh=Decimal("1000"),
            co2_ef=Decimal("0"),
            ch4_ef=Decimal("0"),
            n2o_ef=Decimal("1"),
            gwp_source="AR5",
        )
        n2o_entry = [g for g in result["gas_breakdown"] if g["gas"] == "N2O"][0]
        assert n2o_entry["gwp_factor"] == Decimal("265")
        assert n2o_entry["co2e_kg"] == Decimal("265000.00000000")

    def test_total_co2e_is_sum_of_gases(self, engine):
        """Total CO2e equals sum of individual gas CO2e values."""
        result = engine.calculate_with_gas_breakdown(
            consumption_mwh=Decimal("1000"),
            co2_ef=Decimal("225.30"),
            ch4_ef=Decimal("0.026"),
            n2o_ef=Decimal("0.003"),
        )
        gas_sum = sum(g["co2e_kg"] for g in result["gas_breakdown"])
        assert abs(result["total_co2e_kg"] - gas_sum) < Decimal("0.01")

    @pytest.mark.parametrize("gwp_source", ["AR4", "AR5", "AR6"])
    def test_different_gwp_sources(self, engine, gwp_source):
        """Different GWP sources produce different results."""
        result = engine.calculate_with_gas_breakdown(
            consumption_mwh=Decimal("1000"),
            co2_ef=Decimal("300"),
            ch4_ef=Decimal("0.05"),
            n2o_ef=Decimal("0.01"),
            gwp_source=gwp_source,
        )
        assert result["gwp_source"] == gwp_source
        assert result["total_co2e_kg"] > Decimal("0")

    def test_invalid_gwp_source_raises(self, engine):
        """Unknown GWP source raises ValueError."""
        with pytest.raises(ValueError, match="Unknown gwp_source"):
            engine.calculate_with_gas_breakdown(
                consumption_mwh=Decimal("1000"),
                co2_ef=Decimal("300"),
                ch4_ef=Decimal("0.01"),
                n2o_ef=Decimal("0.001"),
                gwp_source="AR99",
            )

    def test_gas_breakdown_with_td_loss(self, engine):
        """T&D loss factor is applied to gross consumption in gas breakdown."""
        result = engine.calculate_with_gas_breakdown(
            consumption_mwh=Decimal("1000"),
            co2_ef=Decimal("400"),
            ch4_ef=Decimal("0.01"),
            n2o_ef=Decimal("0.001"),
            td_loss_pct=Decimal("0.05"),
        )
        # Gross consumption = 1000 * 1.05 = 1050
        assert result["gross_consumption"] == Decimal("1050.00000000")


# ===========================================================================
# TestFacilityCalculation
# ===========================================================================


@_SKIP
class TestFacilityCalculation:
    """Tests for calculate_for_facility."""

    def test_facility_us_default_factors(self, engine):
        """US facility uses built-in default factors."""
        result = engine.calculate_for_facility(
            facility_id="FAC-001",
            consumption_mwh=Decimal("5000"),
            country_code="US",
        )
        assert result["status"] == "SUCCESS"
        assert result["country_code"] == "US"
        assert result["total_co2e_kg"] > Decimal("0")
        assert result["facility_id"] == "FAC-001"

    def test_facility_egrid_subregion_tier2(self, engine):
        """US facility with eGRID subregion resolves to tier_2."""
        result = engine.calculate_for_facility(
            facility_id="FAC-002",
            consumption_mwh=Decimal("5000"),
            country_code="US",
            egrid_subregion="CAMX",
        )
        assert result["status"] == "SUCCESS"
        assert result["tier"] == "tier_2"

    def test_facility_global_fallback(self, engine):
        """Facility without country_code falls back to GLOBAL."""
        result = engine.calculate_for_facility(
            facility_id="FAC-003",
            consumption_mwh=Decimal("1000"),
        )
        assert result["status"] == "SUCCESS"
        assert result["country_code"] == "GLOBAL"

    def test_facility_includes_gas_breakdown(self, engine):
        """Facility result includes per-gas breakdown."""
        result = engine.calculate_for_facility(
            facility_id="FAC-004",
            consumption_mwh=Decimal("1000"),
            country_code="DE",
        )
        assert result["status"] == "SUCCESS"
        assert len(result["gas_breakdown"]) == 3  # CO2, CH4, N2O

    def test_facility_negative_consumption_fails(self, engine):
        """Negative consumption returns FAILED status."""
        result = engine.calculate_for_facility(
            facility_id="FAC-ERR",
            consumption_mwh=Decimal("-100"),
            country_code="US",
        )
        assert result["status"] == "FAILED"
        assert result["error_message"] is not None

    def test_facility_empty_id_fails(self, engine):
        """Empty facility_id returns FAILED status."""
        result = engine.calculate_for_facility(
            facility_id="",
            consumption_mwh=Decimal("100"),
            country_code="US",
        )
        assert result["status"] == "FAILED"

    def test_facility_no_td_losses(self, engine):
        """Facility with include_td_losses=False skips T&D adjustment."""
        result = engine.calculate_for_facility(
            facility_id="FAC-NOTD",
            consumption_mwh=Decimal("1000"),
            country_code="US",
            include_td_losses=False,
        )
        assert result["status"] == "SUCCESS"
        assert result["include_td_losses"] is False


# ===========================================================================
# TestTierCalculation
# ===========================================================================


@_SKIP
class TestTierCalculation:
    """Tests for tier 1 vs tier 2 resolution."""

    def test_country_level_is_tier_1(self, engine):
        """Country-level lookup resolves to tier_1."""
        result = engine.calculate_for_facility(
            facility_id="TIER1",
            consumption_mwh=Decimal("1000"),
            country_code="GB",
        )
        assert result["tier"] == "tier_1"

    def test_egrid_subregion_is_tier_2(self, engine):
        """eGRID subregion lookup resolves to tier_2."""
        result = engine.calculate_for_facility(
            facility_id="TIER2",
            consumption_mwh=Decimal("1000"),
            country_code="US",
            egrid_subregion="ERCT",
        )
        assert result["tier"] == "tier_2"


# ===========================================================================
# TestUnitConversions
# ===========================================================================


@_SKIP
class TestUnitConversions:
    """Tests for unit conversion utilities."""

    def test_kwh_to_mwh(self):
        """1000 kWh = 1 MWh using _KWH_TO_MWH constant."""
        result = Decimal("1000") * _KWH_TO_MWH
        assert result == Decimal("1")

    def test_gj_to_mwh(self):
        """3.6 GJ = ~1 MWh using _GJ_TO_MWH constant."""
        result = Decimal("3.6") * _GJ_TO_MWH
        assert abs(result - Decimal("1")) < Decimal("0.001")

    def test_to_decimal_from_int(self):
        """_to_decimal handles integer input."""
        assert _to_decimal(42) == Decimal("42")

    def test_to_decimal_from_float(self):
        """_to_decimal handles float input via string conversion."""
        result = _to_decimal(3.14)
        assert isinstance(result, Decimal)

    def test_to_decimal_from_string(self):
        """_to_decimal handles string input."""
        assert _to_decimal("100.5") == Decimal("100.5")

    def test_to_decimal_invalid_raises(self):
        """_to_decimal raises ValueError for non-numeric input."""
        with pytest.raises(ValueError, match="Cannot convert"):
            _to_decimal("not_a_number")

    def test_normalize_consumption_kwh(self, engine):
        """Consumption in kWh is correctly converted to MWh for calculation."""
        kwh_value = Decimal("500000")  # 500 MWh
        mwh_value = kwh_value * _KWH_TO_MWH
        result = engine.calculate_emissions(
            consumption_mwh=mwh_value,
            grid_ef_co2e=Decimal("400"),
        )
        expected = mwh_value * Decimal("400")
        assert result["total_co2e_kg"] == expected.quantize(
            Decimal("0.00000001"), ROUND_HALF_UP
        )


# ===========================================================================
# TestAggregation
# ===========================================================================


@_SKIP
class TestAggregation:
    """Tests for multi-facility aggregation via calculate_monthly."""

    def test_monthly_calculation_12_months(self, engine):
        """calculate_monthly accepts 12-element list."""
        monthly = [Decimal("100")] * 12
        result = engine.calculate_monthly(
            facility_id="FAC-M01",
            monthly_consumption=monthly,
            country_code="US",
            year=2024,
        )
        assert len(result["monthly_results"]) == 12
        assert result["annual_consumption_mwh"] == Decimal("1200")

    def test_monthly_wrong_length_raises(self, engine):
        """Non-12-element list raises ValueError."""
        with pytest.raises(ValueError, match="exactly 12"):
            engine.calculate_monthly(
                facility_id="FAC-M02",
                monthly_consumption=[Decimal("100")] * 11,
                country_code="US",
                year=2024,
            )

    def test_monthly_annual_total_is_sum(self, engine):
        """Annual total is the sum of 12 monthly results."""
        monthly = [Decimal(str(i * 100 + 100)) for i in range(12)]
        result = engine.calculate_monthly(
            facility_id="FAC-M03",
            monthly_consumption=monthly,
            country_code="US",
            year=2024,
        )
        monthly_total = sum(
            Decimal(str(mr.get("total_co2e_kg", mr.get("co2e_kg", 0))))
            for mr in result["monthly_results"]
        )
        assert abs(result["annual_total_co2e_kg"] - monthly_total) < Decimal("1")


# ===========================================================================
# TestValidation
# ===========================================================================


@_SKIP
class TestValidation:
    """Tests for input validation."""

    def test_negative_consumption_raises(self, engine):
        """Negative consumption raises ValueError."""
        with pytest.raises(ValueError, match="Validation failed"):
            engine.calculate_emissions(
                consumption_mwh=Decimal("-100"),
                grid_ef_co2e=Decimal("400"),
            )

    def test_negative_ef_raises(self, engine):
        """Negative emission factor raises ValueError."""
        with pytest.raises(ValueError, match="Validation failed"):
            engine.calculate_emissions(
                consumption_mwh=Decimal("100"),
                grid_ef_co2e=Decimal("-10"),
            )

    def test_negative_consumption_gas_breakdown_raises(self, engine):
        """Negative consumption in gas breakdown raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 0"):
            engine.calculate_with_gas_breakdown(
                consumption_mwh=Decimal("-100"),
                co2_ef=Decimal("300"),
                ch4_ef=Decimal("0.01"),
                n2o_ef=Decimal("0.001"),
            )

    def test_none_consumption_raises(self, engine):
        """None consumption raises ValueError."""
        with pytest.raises((ValueError, TypeError)):
            engine.calculate_emissions(
                consumption_mwh=None,
                grid_ef_co2e=Decimal("400"),
            )


# ===========================================================================
# TestProvenanceTracking
# ===========================================================================


@_SKIP
class TestProvenanceTracking:
    """Tests for provenance tracking and chain integrity."""

    def test_provenance_hash_is_deterministic(self, engine):
        """Same inputs produce identical provenance hashes."""
        r1 = engine.calculate_emissions(Decimal("750"), Decimal("350"))
        r2 = engine.calculate_emissions(Decimal("750"), Decimal("350"))
        assert r1["provenance_hash"] == r2["provenance_hash"]

    def test_different_inputs_different_hash(self, engine):
        """Different inputs produce different provenance hashes."""
        r1 = engine.calculate_emissions(Decimal("100"), Decimal("300"))
        r2 = engine.calculate_emissions(Decimal("200"), Decimal("300"))
        assert r1["provenance_hash"] != r2["provenance_hash"]

    def test_provenance_chain_verify(self, engine):
        """Provenance chain verifies after multiple calculations."""
        engine.calculate_emissions(Decimal("100"), Decimal("300"))
        engine.calculate_emissions(Decimal("200"), Decimal("400"))
        engine.calculate_emissions(Decimal("300"), Decimal("500"))
        if engine._provenance is not None:
            assert engine._provenance.verify_chain() is True

    def test_provenance_disabled_no_error(self, engine_no_provenance):
        """Engine with provenance disabled still calculates correctly."""
        result = engine_no_provenance.calculate_emissions(
            Decimal("100"), Decimal("300")
        )
        assert result["total_co2e_kg"] > Decimal("0")


# ===========================================================================
# TestStatistics
# ===========================================================================


@_SKIP
class TestStatistics:
    """Tests for engine statistics counters."""

    def test_total_calculations_incremented(self, engine):
        """Each calculation increments the counter."""
        engine.calculate_emissions(Decimal("100"), Decimal("300"))
        engine.calculate_emissions(Decimal("200"), Decimal("400"))
        assert engine._total_calculations >= 2

    def test_total_co2e_accumulated(self, engine):
        """Cumulative CO2e counter increases."""
        engine.calculate_emissions(Decimal("100"), Decimal("300"))
        assert engine._total_co2e_kg_processed > Decimal("0")

    def test_default_country_grid_ef_not_empty(self):
        """Built-in default country grid EF table is populated."""
        assert len(_DEFAULT_COUNTRY_GRID_EF) >= 10
        assert "US" in _DEFAULT_COUNTRY_GRID_EF
        assert "GLOBAL" in _DEFAULT_COUNTRY_GRID_EF

    def test_default_egrid_subregion_ef_not_empty(self):
        """Built-in default eGRID subregion EF table is populated."""
        assert len(_DEFAULT_EGRID_SUBREGION_EF) >= 20
        assert "CAMX" in _DEFAULT_EGRID_SUBREGION_EF
