# -*- coding: utf-8 -*-
"""
Unit tests for MarketEmissionsCalculatorEngine

AGENT-MRV-010: Scope 2 Market-Based Emissions Agent

Tests covered emissions, uncovered emissions, market-based totals, gas breakdown,
facility calculation, batch processing, instrument-specific methods (renewable,
supplier-specific, PPA), aggregation (by instrument, facility, period, coverage),
unit conversions, and validation.

Target: ~80 tests, 85%+ coverage.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from unittest.mock import Mock

import pytest

try:
    from greenlang.scope2_market.market_emissions_calculator import (
        MarketEmissionsCalculatorEngine,
        GWP_VALUES,
        DEFAULT_GAS_FRACTIONS,
        RENEWABLE_EF,
        BIOGENIC_SOURCES,
        _to_decimal,
        _KWH_TO_MWH,
        _GJ_TO_MWH,
        _MMBTU_TO_MWH,
        _TJ_TO_MWH,
        _KG_TO_TONNES,
        _TONNES_TO_KG,
        _KGCO2E_KWH_TO_MWH,
        _DEFAULT_RESIDUAL_MIX_EF,
        _VALID_INSTRUMENT_TYPES,
    )
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False

_SKIP = pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine not available")

_Q_INTERNAL = Decimal("0.00000001")
_Q_OUTPUT = Decimal("0.001")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create a default MarketEmissionsCalculatorEngine."""
    return MarketEmissionsCalculatorEngine()


@pytest.fixture
def engine_no_provenance():
    """Create engine with provenance disabled."""
    return MarketEmissionsCalculatorEngine(
        config={"enable_provenance": False}
    )


@pytest.fixture
def engine_ar6():
    """Create engine with AR6 GWP as default."""
    return MarketEmissionsCalculatorEngine(
        config={"default_gwp_source": "AR6"}
    )


@pytest.fixture
def engine_with_region():
    """Create engine with a specific default region."""
    return MarketEmissionsCalculatorEngine(
        config={"default_region": "US"}
    )


# ---------------------------------------------------------------------------
# Helper: build a covered result dict
# ---------------------------------------------------------------------------


def _make_covered_result(
    instrument_type: str = "REC",
    mwh: str = "500",
    co2e_kg: str = "0.000",
    co2e_tonnes: str = "0.000",
    coverage_type: str = "COVERED",
) -> dict:
    """Build a minimal covered-result dictionary for market-based tests."""
    return {
        "instrument_type": instrument_type,
        "consumption_mwh": Decimal(mwh),
        "total_co2e_kg": Decimal(co2e_kg),
        "total_co2e_tonnes": Decimal(co2e_tonnes),
        "coverage_type": coverage_type,
    }


def _make_uncovered_result(
    mwh: str = "200",
    co2e_kg: str = "85000.000",
    co2e_tonnes: str = "85.000",
    region: str = "US",
) -> dict:
    """Build a minimal uncovered-result dictionary for market-based tests."""
    return {
        "consumption_mwh": Decimal(mwh),
        "total_co2e_kg": Decimal(co2e_kg),
        "total_co2e_tonnes": Decimal(co2e_tonnes),
        "region": region,
        "coverage_type": "UNCOVERED",
    }


# ===========================================================================
# 1. TestCoveredEmissions
# ===========================================================================


@_SKIP
class TestCoveredEmissions:
    """Tests for calculate_covered_emissions."""

    def test_renewable_rec_zero_emissions(self, engine):
        """REC with ef=0.000 yields zero emissions."""
        result = engine.calculate_covered_emissions(
            instrument_type="REC",
            mwh=Decimal("1000"),
            ef_kgco2e_kwh=Decimal("0.000"),
        )
        assert result["total_co2e_kg"] == Decimal("0.000")
        assert result["total_co2e_tonnes"] == Decimal("0.000")
        assert result["coverage_type"] == "COVERED"
        assert result["instrument_type"] == "REC"

    def test_fossil_instrument_positive_emissions(self, engine):
        """Non-zero EF produces expected positive emissions."""
        # 500 MWh * 0.425 kgCO2e/kWh * 1000 = 212,500 kgCO2e
        result = engine.calculate_covered_emissions(
            instrument_type="SUPPLIER_SPECIFIC",
            mwh=Decimal("500"),
            ef_kgco2e_kwh=Decimal("0.425"),
        )
        expected_kg = Decimal("212500.000")
        assert result["total_co2e_kg"] == expected_kg
        assert result["total_co2e_tonnes"] == Decimal("212.500")

    def test_ef_conversion_kwh_to_mwh(self, engine):
        """EF is correctly converted from kgCO2e/kWh to kgCO2e/MWh (x1000)."""
        result = engine.calculate_covered_emissions(
            instrument_type="GO",
            mwh=Decimal("1"),
            ef_kgco2e_kwh=Decimal("0.350"),
        )
        assert result["ef_kgco2e_mwh"] == Decimal("350.00000000")
        assert result["total_co2e_kg"] == Decimal("350.000")

    def test_zero_mwh_zero_emissions(self, engine):
        """Zero MWh produces zero emissions regardless of EF."""
        result = engine.calculate_covered_emissions(
            instrument_type="PPA",
            mwh=Decimal("0"),
            ef_kgco2e_kwh=Decimal("0.500"),
        )
        assert result["total_co2e_kg"] == Decimal("0.000")

    def test_provenance_hash_present(self, engine):
        """Result includes a 64-character SHA-256 provenance hash."""
        result = engine.calculate_covered_emissions(
            instrument_type="REC",
            mwh=Decimal("100"),
            ef_kgco2e_kwh=Decimal("0.000"),
        )
        assert len(result["provenance_hash"]) == 64

    def test_calculation_trace_non_empty(self, engine):
        """Result includes a non-empty calculation trace."""
        result = engine.calculate_covered_emissions(
            instrument_type="REC",
            mwh=Decimal("100"),
            ef_kgco2e_kwh=Decimal("0.000"),
        )
        assert len(result["calculation_trace"]) > 0

    def test_processing_time_recorded(self, engine):
        """processing_time_ms is recorded and >= 0."""
        result = engine.calculate_covered_emissions(
            instrument_type="REC",
            mwh=Decimal("100"),
            ef_kgco2e_kwh=Decimal("0.000"),
        )
        assert result["processing_time_ms"] >= 0

    def test_deterministic_output(self, engine):
        """Same inputs produce identical outputs (bit-perfect)."""
        r1 = engine.calculate_covered_emissions("REC", Decimal("500"), Decimal("0.100"))
        r2 = engine.calculate_covered_emissions("REC", Decimal("500"), Decimal("0.100"))
        assert r1["total_co2e_kg"] == r2["total_co2e_kg"]
        assert r1["provenance_hash"] == r2["provenance_hash"]

    def test_negative_mwh_raises(self, engine):
        """Negative mwh raises ValueError."""
        with pytest.raises(ValueError, match="mwh must be >= 0"):
            engine.calculate_covered_emissions("REC", Decimal("-1"), Decimal("0.000"))

    def test_negative_ef_raises(self, engine):
        """Negative EF raises ValueError."""
        with pytest.raises(ValueError, match="ef_kgco2e_kwh must be >= 0"):
            engine.calculate_covered_emissions("REC", Decimal("100"), Decimal("-0.01"))

    def test_invalid_instrument_type_raises(self, engine):
        """Unrecognized instrument type raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized instrument_type"):
            engine.calculate_covered_emissions("INVALID_TYPE", Decimal("100"), Decimal("0.5"))

    @pytest.mark.parametrize("instrument_type", [
        "REC", "GO", "I-REC", "REGO", "LGC", "GEC", "TIGR",
        "PPA", "GREEN_TARIFF", "VPPA", "DIRECT_LINE",
        "SUPPLIER_SPECIFIC", "RESIDUAL_MIX", "OTHER",
    ])
    def test_all_valid_instrument_types_accepted(self, engine, instrument_type):
        """All valid instrument types are accepted."""
        result = engine.calculate_covered_emissions(
            instrument_type=instrument_type,
            mwh=Decimal("10"),
            ef_kgco2e_kwh=Decimal("0.100"),
        )
        assert result["instrument_type"] == instrument_type


# ===========================================================================
# 2. TestUncoveredEmissions
# ===========================================================================


@_SKIP
class TestUncoveredEmissions:
    """Tests for calculate_uncovered_emissions."""

    def test_us_residual_mix_default(self, engine):
        """US residual mix factor (0.425 kgCO2e/kWh) applied correctly."""
        # 1000 MWh * 0.425 * 1000 = 425,000 kgCO2e
        result = engine.calculate_uncovered_emissions(
            mwh=Decimal("1000"),
            region="US",
        )
        assert result["total_co2e_kg"] == Decimal("425000.000")
        assert result["total_co2e_tonnes"] == Decimal("425.000")
        assert result["coverage_type"] == "UNCOVERED"

    def test_de_residual_mix(self, engine):
        """DE (Germany) residual mix factor 0.560 applied correctly."""
        # 500 MWh * 0.560 * 1000 = 280,000 kgCO2e
        result = engine.calculate_uncovered_emissions(
            mwh=Decimal("500"),
            region="DE",
        )
        assert result["total_co2e_kg"] == Decimal("280000.000")
        assert result["residual_mix_ef_kgco2e_kwh"] == Decimal("0.560")

    def test_gb_residual_mix(self, engine):
        """GB residual mix factor 0.320 applied correctly."""
        result = engine.calculate_uncovered_emissions(
            mwh=Decimal("100"),
            region="GB",
        )
        assert result["total_co2e_kg"] == Decimal("32000.000")

    def test_user_provided_ef_overrides_default(self, engine):
        """User-provided EF overrides the built-in default."""
        result = engine.calculate_uncovered_emissions(
            mwh=Decimal("1000"),
            region="US",
            residual_mix_ef=Decimal("0.500"),
        )
        assert result["total_co2e_kg"] == Decimal("500000.000")
        assert result["ef_source"] == "USER_PROVIDED"

    def test_zero_mwh_returns_zero(self, engine):
        """Zero MWh produces zero emissions."""
        result = engine.calculate_uncovered_emissions(
            mwh=Decimal("0"),
            region="US",
        )
        assert result["total_co2e_kg"] == Decimal("0.000")

    def test_global_fallback_for_unknown_region(self, engine):
        """Unknown region falls back to GLOBAL residual mix factor."""
        result = engine.calculate_uncovered_emissions(
            mwh=Decimal("100"),
            region="XY_UNKNOWN",
        )
        # GLOBAL fallback: 0.450 kgCO2e/kWh -> 100 * 450 = 45,000 kgCO2e
        assert result["total_co2e_kg"] == Decimal("45000.000")

    def test_negative_mwh_raises(self, engine):
        """Negative MWh raises ValueError."""
        with pytest.raises(ValueError, match="mwh must be >= 0"):
            engine.calculate_uncovered_emissions(
                mwh=Decimal("-100"),
                region="US",
            )

    def test_negative_residual_ef_raises(self, engine):
        """Negative residual mix EF raises ValueError."""
        with pytest.raises(ValueError, match="residual_mix_ef must be >= 0"):
            engine.calculate_uncovered_emissions(
                mwh=Decimal("100"),
                region="US",
                residual_mix_ef=Decimal("-0.01"),
            )

    def test_region_case_insensitive(self, engine):
        """Region lookup is case-insensitive."""
        r1 = engine.calculate_uncovered_emissions(Decimal("100"), "us")
        r2 = engine.calculate_uncovered_emissions(Decimal("100"), "US")
        assert r1["total_co2e_kg"] == r2["total_co2e_kg"]

    @pytest.mark.parametrize("region,expected_ef", [
        ("US", Decimal("0.425")),
        ("EU", Decimal("0.420")),
        ("FR", Decimal("0.055")),
        ("JP", Decimal("0.470")),
        ("AU", Decimal("0.680")),
        ("ZA", Decimal("0.950")),
        ("BR", Decimal("0.090")),
        ("GLOBAL", Decimal("0.450")),
    ])
    def test_built_in_residual_mix_regions(self, engine, region, expected_ef):
        """Built-in residual mix EFs match expected values."""
        result = engine.calculate_uncovered_emissions(
            mwh=Decimal("1"),
            region=region,
        )
        assert result["residual_mix_ef_kgco2e_kwh"] == expected_ef


# ===========================================================================
# 3. TestMarketBasedTotal
# ===========================================================================


@_SKIP
class TestMarketBasedTotal:
    """Tests for calculate_market_based."""

    def test_combined_covered_and_uncovered(self, engine):
        """Total market-based = sum(covered) + uncovered."""
        covered = [
            _make_covered_result("REC", "500", "0.000", "0.000"),
            _make_covered_result("SUPPLIER_SPECIFIC", "300", "75000.000", "75.000"),
        ]
        uncovered = _make_uncovered_result("200", "85000.000", "85.000")

        result = engine.calculate_market_based(
            total_mwh=Decimal("1000"),
            covered_results=covered,
            uncovered_result=uncovered,
        )
        assert result["total_co2e_kg"] == Decimal("160000.000")
        assert result["total_co2e_tonnes"] == Decimal("160.000")

    def test_coverage_percentage(self, engine):
        """Coverage percentage computed correctly."""
        covered = [_make_covered_result("REC", "800", "0.000")]
        uncovered = _make_uncovered_result("200", "85000.000")

        result = engine.calculate_market_based(
            total_mwh=Decimal("1000"),
            covered_results=covered,
            uncovered_result=uncovered,
        )
        assert result["coverage_pct"] == Decimal("80.000")

    def test_mwh_balance_check_pass(self, engine):
        """MWh balance check passes when covered + uncovered = total."""
        covered = [_make_covered_result("REC", "700", "0.000")]
        uncovered = _make_uncovered_result("300", "127500.000")

        result = engine.calculate_market_based(
            total_mwh=Decimal("1000"),
            covered_results=covered,
            uncovered_result=uncovered,
        )
        assert result["mwh_balance_check"] == "PASS"

    def test_mwh_balance_check_warn(self, engine):
        """MWh balance warns when covered + uncovered != total."""
        covered = [_make_covered_result("REC", "700", "0.000")]
        uncovered = _make_uncovered_result("200", "85000.000")

        result = engine.calculate_market_based(
            total_mwh=Decimal("1000"),
            covered_results=covered,
            uncovered_result=uncovered,
        )
        # 700 + 200 = 900 vs 1000 => WARN
        assert result["mwh_balance_check"] == "WARN"

    def test_instrument_count(self, engine):
        """instrument_count reflects covered_results length."""
        covered = [
            _make_covered_result("REC", "200", "0.000"),
            _make_covered_result("GO", "300", "0.000"),
            _make_covered_result("PPA", "100", "25000.000"),
        ]
        uncovered = _make_uncovered_result("400", "170000.000")

        result = engine.calculate_market_based(
            total_mwh=Decimal("1000"),
            covered_results=covered,
            uncovered_result=uncovered,
        )
        assert result["instrument_count"] == 3

    def test_zero_total_mwh(self, engine):
        """Zero total MWh yields zero coverage percentage."""
        result = engine.calculate_market_based(
            total_mwh=Decimal("0"),
            covered_results=[],
            uncovered_result=_make_uncovered_result("0", "0.000"),
        )
        assert result["coverage_pct"] == Decimal("0.000")
        assert result["total_co2e_kg"] == Decimal("0.000")

    def test_negative_total_mwh_raises(self, engine):
        """Negative total_mwh raises ValueError."""
        with pytest.raises(ValueError, match="total_mwh must be >= 0"):
            engine.calculate_market_based(
                total_mwh=Decimal("-1"),
                covered_results=[],
                uncovered_result=_make_uncovered_result(),
            )

    def test_100_percent_covered(self, engine):
        """Fully covered (100%) with zero uncovered."""
        covered = [_make_covered_result("REC", "1000", "0.000")]
        uncovered = _make_uncovered_result("0", "0.000")

        result = engine.calculate_market_based(
            total_mwh=Decimal("1000"),
            covered_results=covered,
            uncovered_result=uncovered,
        )
        assert result["coverage_pct"] == Decimal("100.000")
        assert result["uncovered_co2e_kg"] == Decimal("0.000")

    def test_provenance_hash_64_chars(self, engine):
        """Result includes a 64-character SHA-256 provenance hash."""
        covered = [_make_covered_result()]
        uncovered = _make_uncovered_result()
        result = engine.calculate_market_based(
            total_mwh=Decimal("700"),
            covered_results=covered,
            uncovered_result=uncovered,
        )
        assert len(result["provenance_hash"]) == 64

    def test_covered_uncovered_mwh_in_output(self, engine):
        """Output includes separated covered_mwh and uncovered_mwh."""
        covered = [_make_covered_result("REC", "600", "0.000")]
        uncovered = _make_uncovered_result("400", "170000.000")

        result = engine.calculate_market_based(
            total_mwh=Decimal("1000"),
            covered_results=covered,
            uncovered_result=uncovered,
        )
        assert result["covered_mwh"] == Decimal("600.000")
        assert result["uncovered_mwh"] == Decimal("400.000")


# ===========================================================================
# 4. TestGasBreakdown
# ===========================================================================


@_SKIP
class TestGasBreakdown:
    """Tests for calculate_with_gas_breakdown."""

    def test_default_fractions_applied(self, engine):
        """Default gas fractions (95% CO2, 3% CH4, 2% N2O) are applied."""
        result = engine.calculate_with_gas_breakdown(
            total_mwh=Decimal("1000"),
            ef=Decimal("0.425"),
        )
        gases = {g["gas"] for g in result["gas_breakdown"]}
        assert gases == {"CO2", "CH4", "N2O"}

    def test_co2_dominant_component(self, engine):
        """CO2 fraction (95%) is the dominant gas component."""
        result = engine.calculate_with_gas_breakdown(
            total_mwh=Decimal("1000"),
            ef=Decimal("0.425"),
        )
        co2 = [g for g in result["gas_breakdown"] if g["gas"] == "CO2"][0]
        assert co2["fraction"] == Decimal("0.95")
        # 1000 * 425 * 0.95 = 403,750 kgCO2e
        assert co2["co2e_kg"] == Decimal("403750.000")

    def test_ch4_gwp_ar5(self, engine):
        """CH4 uses GWP=28 for AR5."""
        result = engine.calculate_with_gas_breakdown(
            total_mwh=Decimal("1000"),
            ef=Decimal("0.425"),
            gwp_source="AR5",
        )
        ch4 = [g for g in result["gas_breakdown"] if g["gas"] == "CH4"][0]
        assert ch4["gwp_factor"] == Decimal("28")
        # Total = 425,000; CH4 fraction = 0.03 -> 12,750 co2e_kg
        # mass = 12750 / 28 = 455.357... -> 455.357
        assert ch4["co2e_kg"] == Decimal("12750.000")

    def test_n2o_gwp_ar5(self, engine):
        """N2O uses GWP=265 for AR5."""
        result = engine.calculate_with_gas_breakdown(
            total_mwh=Decimal("1000"),
            ef=Decimal("0.425"),
            gwp_source="AR5",
        )
        n2o = [g for g in result["gas_breakdown"] if g["gas"] == "N2O"][0]
        assert n2o["gwp_factor"] == Decimal("265")
        # N2O fraction = 0.02 -> 8,500 co2e_kg
        assert n2o["co2e_kg"] == Decimal("8500.000")

    def test_ar6_gwp_values(self, engine):
        """AR6 GWP values (CH4=27.9, N2O=273) applied correctly."""
        result = engine.calculate_with_gas_breakdown(
            total_mwh=Decimal("100"),
            ef=Decimal("0.400"),
            gwp_source="AR6",
        )
        ch4 = [g for g in result["gas_breakdown"] if g["gas"] == "CH4"][0]
        n2o = [g for g in result["gas_breakdown"] if g["gas"] == "N2O"][0]
        assert ch4["gwp_factor"] == Decimal("27.9")
        assert n2o["gwp_factor"] == Decimal("273")

    def test_custom_gas_fractions(self, engine):
        """Custom gas fractions override defaults."""
        custom = {"co2": Decimal("0.90"), "ch4": Decimal("0.05"), "n2o": Decimal("0.05")}
        result = engine.calculate_with_gas_breakdown(
            total_mwh=Decimal("100"),
            ef=Decimal("0.500"),
            gas_fractions=custom,
        )
        co2 = [g for g in result["gas_breakdown"] if g["gas"] == "CO2"][0]
        assert co2["fraction"] == Decimal("0.90")
        assert result["gas_fractions_used"] == custom

    def test_fractions_not_summing_to_1_raises(self, engine):
        """Gas fractions that do not sum to 1.0 raise ValueError."""
        bad_fractions = {"co2": Decimal("0.80"), "ch4": Decimal("0.05"), "n2o": Decimal("0.05")}
        with pytest.raises(ValueError, match="must sum to 1.0"):
            engine.calculate_with_gas_breakdown(
                total_mwh=Decimal("100"),
                ef=Decimal("0.500"),
                gas_fractions=bad_fractions,
            )

    def test_missing_fraction_key_raises(self, engine):
        """Missing gas fraction key raises ValueError."""
        incomplete = {"co2": Decimal("0.95"), "ch4": Decimal("0.05")}
        with pytest.raises(ValueError, match="Missing gas fraction keys"):
            engine.calculate_with_gas_breakdown(
                total_mwh=Decimal("100"),
                ef=Decimal("0.500"),
                gas_fractions=incomplete,
            )

    def test_unknown_gwp_source_raises(self, engine):
        """Unknown GWP source raises ValueError."""
        with pytest.raises(ValueError, match="Unknown gwp_source"):
            engine.calculate_with_gas_breakdown(
                total_mwh=Decimal("100"),
                ef=Decimal("0.500"),
                gwp_source="AR99",
            )

    def test_zero_ef_zero_breakdown(self, engine):
        """Zero EF yields zero for all gas components."""
        result = engine.calculate_with_gas_breakdown(
            total_mwh=Decimal("1000"),
            ef=Decimal("0.000"),
        )
        for g in result["gas_breakdown"]:
            assert g["co2e_kg"] == Decimal("0.000")


# ===========================================================================
# 5. TestFacilityCalculation
# ===========================================================================


@_SKIP
class TestFacilityCalculation:
    """Tests for calculate_for_facility."""

    def test_mixed_purchases(self, engine):
        """Facility with covered + uncovered purchases computes correctly."""
        purchases = [
            {"mwh": Decimal("500"), "instrument_type": "REC", "ef_kgco2e_kwh": Decimal("0.000")},
            {"mwh": Decimal("300"), "instrument_type": "SUPPLIER_SPECIFIC", "ef_kgco2e_kwh": Decimal("0.350")},
            {"mwh": Decimal("200"), "instrument_type": "RESIDUAL_MIX", "ef_kgco2e_kwh": Decimal("0"), "region": "US"},
        ]
        result = engine.calculate_for_facility("FAC-001", purchases)

        assert result["status"] == "SUCCESS"
        assert result["facility_id"] == "FAC-001"
        assert result["total_mwh"] == Decimal("1000.000")
        assert result["covered_mwh"] == Decimal("800.000")
        assert result["uncovered_mwh"] == Decimal("200.000")
        assert result["instrument_count"] == 2  # REC + SUPPLIER_SPECIFIC

    def test_all_renewable_zero_emissions(self, engine):
        """All-renewable facility yields zero emissions."""
        purchases = [
            {"mwh": Decimal("1000"), "instrument_type": "REC", "ef_kgco2e_kwh": Decimal("0.000")},
        ]
        result = engine.calculate_for_facility("FAC-GREEN", purchases)

        assert result["status"] == "SUCCESS"
        assert result["total_co2e_kg"] == Decimal("0.000")
        assert result["coverage_pct"] == Decimal("100.000")

    def test_empty_facility_id_raises(self, engine):
        """Empty facility_id raises ValueError."""
        with pytest.raises(ValueError, match="facility_id must not be empty"):
            engine.calculate_for_facility("", [])

    def test_whitespace_facility_id_raises(self, engine):
        """Whitespace-only facility_id raises ValueError."""
        with pytest.raises(ValueError, match="facility_id must not be empty"):
            engine.calculate_for_facility("   ", [])

    def test_empty_purchases_returns_zero(self, engine):
        """No purchases yields zero total."""
        result = engine.calculate_for_facility("FAC-EMPTY", [])
        assert result["status"] == "SUCCESS"
        assert result["total_co2e_kg"] == Decimal("0.000")


# ===========================================================================
# 6. TestBatchProcessing
# ===========================================================================


@_SKIP
class TestBatchProcessing:
    """Tests for calculate_batch."""

    def test_batch_multiple_requests(self, engine):
        """Batch processes multiple requests and sums total."""
        requests = [
            {"instrument_type": "REC", "mwh": Decimal("500"), "ef_kgco2e_kwh": Decimal("0.000")},
            {"instrument_type": "SUPPLIER_SPECIFIC", "mwh": Decimal("300"), "ef_kgco2e_kwh": Decimal("0.350")},
        ]
        result = engine.calculate_batch(requests)

        assert result["success_count"] == 2
        assert result["failure_count"] == 0
        assert len(result["results"]) == 2
        # REC: 0, SUPPLIER: 300*350=105,000
        assert result["total_co2e_kg"] == Decimal("105000.000")

    def test_batch_error_isolation(self, engine):
        """A bad request in a batch does not prevent others from succeeding."""
        requests = [
            {"instrument_type": "REC", "mwh": Decimal("100"), "ef_kgco2e_kwh": Decimal("0.000")},
            {"instrument_type": "INVALID", "mwh": Decimal("100"), "ef_kgco2e_kwh": Decimal("0.300")},
            {"instrument_type": "GO", "mwh": Decimal("100"), "ef_kgco2e_kwh": Decimal("0.000")},
        ]
        result = engine.calculate_batch(requests)

        assert result["success_count"] == 2
        assert result["failure_count"] == 1
        assert result["results"][1]["status"] == "FAILED"

    def test_batch_empty_list(self, engine):
        """Empty batch returns zero totals."""
        result = engine.calculate_batch([])
        assert result["success_count"] == 0
        assert result["failure_count"] == 0
        assert result["total_co2e_kg"] == Decimal("0.000")

    def test_batch_residual_mix_routing(self, engine):
        """RESIDUAL_MIX instrument routes to uncovered calculation."""
        requests = [
            {"instrument_type": "RESIDUAL_MIX", "mwh": Decimal("500"), "ef_kgco2e_kwh": Decimal("0"), "region": "US"},
        ]
        result = engine.calculate_batch(requests)
        assert result["success_count"] == 1
        # Should use US residual mix 0.425: 500 * 425 = 212,500
        assert result["total_co2e_kg"] == Decimal("212500.000")

    def test_batch_provenance_hash(self, engine):
        """Batch result includes a 64-character provenance hash."""
        result = engine.calculate_batch([
            {"instrument_type": "REC", "mwh": Decimal("100"), "ef_kgco2e_kwh": Decimal("0")},
        ])
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# 7. TestInstrumentSpecificMethods
# ===========================================================================


@_SKIP
class TestInstrumentSpecificMethods:
    """Tests for calculate_renewable_instrument, calculate_supplier_specific, calculate_ppa_emissions."""

    # --- Renewable Instrument ---

    def test_renewable_instrument_zero_emissions(self, engine):
        """Renewable instrument yields zero emissions."""
        result = engine.calculate_renewable_instrument(
            instrument_type="REC",
            mwh=Decimal("1000"),
        )
        assert result["total_co2e_kg"] == Decimal("0.000")
        assert result["is_renewable"] is True

    def test_renewable_negative_mwh_raises(self, engine):
        """Negative MWh raises ValueError."""
        with pytest.raises(ValueError, match="mwh must be >= 0"):
            engine.calculate_renewable_instrument("REC", Decimal("-100"))

    # --- Supplier-Specific ---

    def test_supplier_specific_calculation(self, engine):
        """Supplier-specific EF method calculates correctly."""
        # 500 MWh * 0.300 kgCO2e/kWh * 1000 = 150,000 kgCO2e
        result = engine.calculate_supplier_specific(
            supplier_ef=Decimal("0.300"),
            mwh=Decimal("500"),
        )
        assert result["total_co2e_kg"] == Decimal("150000.000")
        assert result["instrument_type"] == "SUPPLIER_SPECIFIC"
        assert result["market_method_rank"] == 1

    def test_supplier_specific_negative_ef_raises(self, engine):
        """Negative supplier EF raises ValueError."""
        with pytest.raises(ValueError, match="supplier_ef must be >= 0"):
            engine.calculate_supplier_specific(Decimal("-0.01"), Decimal("100"))

    def test_supplier_specific_negative_mwh_raises(self, engine):
        """Negative MWh raises ValueError."""
        with pytest.raises(ValueError, match="mwh must be >= 0"):
            engine.calculate_supplier_specific(Decimal("0.300"), Decimal("-1"))

    # --- PPA Emissions ---

    def test_ppa_solar_zero_ef(self, engine):
        """Solar PPA yields zero emissions."""
        result = engine.calculate_ppa_emissions(
            ppa_source="solar",
            mwh=Decimal("500"),
        )
        assert result["total_co2e_kg"] == Decimal("0.000")
        assert result["is_renewable_ppa"] is True

    def test_ppa_wind_zero_ef(self, engine):
        """Wind PPA yields zero emissions."""
        result = engine.calculate_ppa_emissions(
            ppa_source="wind",
            mwh=Decimal("1000"),
        )
        assert result["total_co2e_kg"] == Decimal("0.000")
        assert result["ppa_source"] == "wind"

    def test_ppa_custom_ef(self, engine):
        """PPA with custom EF for non-renewable source."""
        result = engine.calculate_ppa_emissions(
            ppa_source="natural_gas",
            mwh=Decimal("100"),
            custom_ef=Decimal("0.400"),
        )
        assert result["total_co2e_kg"] == Decimal("40000.000")
        assert result["ppa_ef_source"] == "CUSTOM_PPA_EF"

    def test_ppa_biogenic_source(self, engine):
        """Biogenic PPA source yields zero EF."""
        result = engine.calculate_ppa_emissions(
            ppa_source="biomass",
            mwh=Decimal("200"),
        )
        assert result["total_co2e_kg"] == Decimal("0.000")
        assert result["is_biogenic_ppa"] is True

    def test_ppa_unknown_source_no_ef_raises(self, engine):
        """Unknown non-renewable PPA source without custom_ef raises ValueError."""
        with pytest.raises(ValueError, match="not renewable/biogenic"):
            engine.calculate_ppa_emissions(
                ppa_source="coal",
                mwh=Decimal("100"),
            )


# ===========================================================================
# 8. TestAggregation
# ===========================================================================


@_SKIP
class TestAggregation:
    """Tests for aggregate_by_instrument, aggregate_by_facility, aggregate_by_period, aggregate_by_coverage."""

    # --- aggregate_by_instrument ---

    def test_aggregate_by_instrument(self, engine):
        """Results grouped by instrument type."""
        results = [
            _make_covered_result("REC", "500", "0.000", "0.000"),
            _make_covered_result("REC", "300", "0.000", "0.000"),
            _make_covered_result("GO", "200", "20000.000", "20.000"),
        ]
        agg = engine.aggregate_by_instrument(results)

        assert "REC" in agg["by_instrument"]
        assert agg["by_instrument"]["REC"]["calculation_count"] == 2
        assert agg["by_instrument"]["REC"]["consumption_mwh"] == Decimal("800.000")
        assert agg["by_instrument"]["GO"]["calculation_count"] == 1
        assert agg["instrument_type_count"] == 2

    def test_aggregate_by_instrument_empty(self, engine):
        """Empty results produce zero grand total."""
        agg = engine.aggregate_by_instrument([])
        assert agg["grand_total"]["total_co2e_kg"] == Decimal("0.000")
        assert agg["instrument_type_count"] == 0

    # --- aggregate_by_facility ---

    def test_aggregate_by_facility(self, engine):
        """Results grouped by facility_id."""
        results = [
            {"facility_id": "FAC-A", "total_co2e_kg": Decimal("10000"), "total_co2e_tonnes": Decimal("10"), "total_mwh": Decimal("100")},
            {"facility_id": "FAC-A", "total_co2e_kg": Decimal("20000"), "total_co2e_tonnes": Decimal("20"), "total_mwh": Decimal("200")},
            {"facility_id": "FAC-B", "total_co2e_kg": Decimal("5000"), "total_co2e_tonnes": Decimal("5"), "total_mwh": Decimal("50")},
        ]
        agg = engine.aggregate_by_facility(results)

        assert agg["facility_count"] == 2
        assert agg["by_facility"]["FAC-A"]["total_co2e_kg"] == Decimal("30000.000")
        assert agg["by_facility"]["FAC-B"]["total_co2e_kg"] == Decimal("5000.000")
        assert agg["grand_total"]["total_co2e_kg"] == Decimal("35000.000")

    def test_aggregate_by_facility_empty(self, engine):
        """Empty results produce zero grand total."""
        agg = engine.aggregate_by_facility([])
        assert agg["facility_count"] == 0
        assert agg["grand_total"]["total_co2e_kg"] == Decimal("0.000")

    # --- aggregate_by_period ---

    def test_aggregate_by_period_annual(self, engine):
        """Annual aggregation groups by year."""
        results = [
            {"total_co2e_kg": Decimal("10000"), "total_co2e_tonnes": Decimal("10"), "consumption_mwh": Decimal("100"), "timestamp": "2026-01-15T00:00:00"},
            {"total_co2e_kg": Decimal("20000"), "total_co2e_tonnes": Decimal("20"), "consumption_mwh": Decimal("200"), "timestamp": "2026-06-15T00:00:00"},
            {"total_co2e_kg": Decimal("5000"), "total_co2e_tonnes": Decimal("5"), "consumption_mwh": Decimal("50"), "timestamp": "2025-12-01T00:00:00"},
        ]
        agg = engine.aggregate_by_period(results, period_type="annual")

        assert "2026" in agg["by_period"]
        assert "2025" in agg["by_period"]
        assert agg["by_period"]["2026"]["total_co2e_kg"] == Decimal("30000.000")
        assert agg["period_count"] == 2

    def test_aggregate_by_period_quarterly(self, engine):
        """Quarterly aggregation groups by year-quarter."""
        results = [
            {"total_co2e_kg": Decimal("10000"), "total_co2e_tonnes": Decimal("10"), "consumption_mwh": Decimal("100"), "month": 1, "year": 2026},
            {"total_co2e_kg": Decimal("20000"), "total_co2e_tonnes": Decimal("20"), "consumption_mwh": Decimal("200"), "month": 4, "year": 2026},
        ]
        agg = engine.aggregate_by_period(results, period_type="quarterly")

        assert "2026-Q1" in agg["by_period"]
        assert "2026-Q2" in agg["by_period"]

    def test_aggregate_by_period_invalid_raises(self, engine):
        """Invalid period_type raises ValueError."""
        with pytest.raises(ValueError, match="period_type must be one of"):
            engine.aggregate_by_period([], period_type="weekly")

    # --- aggregate_by_coverage ---

    def test_aggregate_by_coverage(self, engine):
        """Results split into covered and uncovered buckets."""
        results = [
            _make_covered_result("REC", "500", "0.000", "0.000", "COVERED"),
            _make_covered_result("GO", "300", "30000.000", "30.000", "COVERED"),
            _make_uncovered_result("200", "85000.000", "85.000"),
        ]
        agg = engine.aggregate_by_coverage(results)

        assert agg["covered"]["calculation_count"] == 2
        assert agg["uncovered"]["calculation_count"] == 1
        assert agg["covered"]["total_co2e_kg"] == Decimal("30000.000")
        assert agg["uncovered"]["total_co2e_kg"] == Decimal("85000.000")
        assert agg["grand_total"]["total_co2e_kg"] == Decimal("115000.000")

    def test_aggregate_by_coverage_percentages(self, engine):
        """Coverage percentages (emissions and consumption) computed."""
        results = [
            {"coverage_type": "COVERED", "total_co2e_kg": Decimal("50000"), "consumption_mwh": Decimal("800")},
            {"coverage_type": "UNCOVERED", "total_co2e_kg": Decimal("50000"), "consumption_mwh": Decimal("200")},
        ]
        agg = engine.aggregate_by_coverage(results)

        # 50/50 split for emissions
        assert agg["covered"]["pct_of_emissions"] == Decimal("50.000")
        assert agg["uncovered"]["pct_of_emissions"] == Decimal("50.000")
        # 80/20 split for consumption
        assert agg["covered"]["pct_of_consumption"] == Decimal("80.000")
        assert agg["uncovered"]["pct_of_consumption"] == Decimal("20.000")


# ===========================================================================
# 9. TestUnitConversions
# ===========================================================================


@_SKIP
class TestUnitConversions:
    """Tests for kwh_to_mwh, gj_to_mwh, mmbtu_to_mwh, normalize_consumption."""

    def test_kwh_to_mwh(self, engine):
        """1000 kWh = 1.0 MWh."""
        result = engine.kwh_to_mwh(Decimal("1000"))
        assert result == Decimal("1.00000000")

    def test_kwh_to_mwh_small(self, engine):
        """1 kWh = 0.001 MWh."""
        result = engine.kwh_to_mwh(Decimal("1"))
        assert result == Decimal("0.00100000")

    def test_gj_to_mwh(self, engine):
        """3.6 GJ = ~1.0 MWh."""
        result = engine.gj_to_mwh(Decimal("3.6"))
        expected = (Decimal("3.6") * _GJ_TO_MWH).quantize(
            Decimal("0.00000001"), ROUND_HALF_UP
        )
        assert result == expected

    def test_mmbtu_to_mwh(self, engine):
        """1 MMBTU = 0.293071 MWh."""
        result = engine.mmbtu_to_mwh(Decimal("1"))
        assert result == Decimal("0.29307100")

    def test_normalize_kwh(self, engine):
        """normalize_consumption converts kWh to MWh."""
        result = engine.normalize_consumption(Decimal("5000"), "kWh")
        assert result == Decimal("5.00000000")

    def test_normalize_mwh_passthrough(self, engine):
        """normalize_consumption with MWh is passthrough."""
        result = engine.normalize_consumption(Decimal("100"), "MWh")
        assert result == Decimal("100.00000000")

    def test_normalize_gj(self, engine):
        """normalize_consumption converts GJ to MWh."""
        result = engine.normalize_consumption(Decimal("10"), "GJ")
        expected = (Decimal("10") * _GJ_TO_MWH).quantize(
            Decimal("0.00000001"), ROUND_HALF_UP
        )
        assert result == expected

    def test_normalize_mmbtu(self, engine):
        """normalize_consumption converts MMBTU to MWh."""
        result = engine.normalize_consumption(Decimal("10"), "MMBTU")
        expected = (Decimal("10") * _MMBTU_TO_MWH).quantize(
            Decimal("0.00000001"), ROUND_HALF_UP
        )
        assert result == expected

    def test_normalize_tj(self, engine):
        """normalize_consumption converts TJ to MWh."""
        result = engine.normalize_consumption(Decimal("1"), "TJ")
        assert result == Decimal("277.77800000")

    def test_normalize_unsupported_unit_raises(self, engine):
        """Unsupported unit raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported energy unit"):
            engine.normalize_consumption(Decimal("100"), "BTU")

    def test_negative_kwh_raises(self, engine):
        """Negative kWh raises ValueError."""
        with pytest.raises(ValueError, match="kwh must be >= 0"):
            engine.kwh_to_mwh(Decimal("-1"))

    def test_negative_gj_raises(self, engine):
        """Negative GJ raises ValueError."""
        with pytest.raises(ValueError, match="gj must be >= 0"):
            engine.gj_to_mwh(Decimal("-1"))

    def test_negative_mmbtu_raises(self, engine):
        """Negative MMBTU raises ValueError."""
        with pytest.raises(ValueError, match="mmbtu must be >= 0"):
            engine.mmbtu_to_mwh(Decimal("-1"))

    def test_normalize_negative_raises(self, engine):
        """Negative quantity raises ValueError."""
        with pytest.raises(ValueError, match="quantity must be >= 0"):
            engine.normalize_consumption(Decimal("-100"), "kWh")

    def test_unit_case_insensitive(self, engine):
        """Unit strings are case-insensitive."""
        r1 = engine.normalize_consumption(Decimal("100"), "kwh")
        r2 = engine.normalize_consumption(Decimal("100"), "KWH")
        assert r1 == r2


# ===========================================================================
# 10. TestValidation
# ===========================================================================


@_SKIP
class TestValidation:
    """Tests for validate_consumption and validate_emission_factor."""

    # --- validate_consumption ---

    def test_valid_consumption_no_errors(self, engine):
        """Valid consumption returns empty list."""
        errors = engine.validate_consumption(Decimal("500"))
        assert errors == []

    def test_negative_consumption(self, engine):
        """Negative consumption returns error."""
        errors = engine.validate_consumption(Decimal("-1"))
        assert any("must be >= 0" in e for e in errors)

    def test_zero_consumption_warning(self, engine):
        """Zero consumption returns warning."""
        errors = engine.validate_consumption(Decimal("0"))
        assert any("WARNING" in e and "0" in e for e in errors)

    def test_huge_consumption_warning(self, engine):
        """Very large consumption returns warning."""
        errors = engine.validate_consumption(Decimal("2000000"))
        assert any("WARNING" in e and "1,000,000" in e for e in errors)

    def test_invalid_type_consumption(self, engine):
        """Non-convertible value returns error."""
        errors = engine.validate_consumption("not_a_number")
        assert len(errors) > 0

    # --- validate_emission_factor ---

    def test_valid_ef_no_errors(self, engine):
        """Valid EF returns empty list."""
        errors = engine.validate_emission_factor(Decimal("0.425"))
        assert errors == []

    def test_zero_ef_acceptable(self, engine):
        """Zero EF is acceptable (renewable instruments)."""
        errors = engine.validate_emission_factor(Decimal("0.000"))
        assert errors == []

    def test_negative_ef_error(self, engine):
        """Negative EF returns error."""
        errors = engine.validate_emission_factor(Decimal("-0.1"))
        assert any("must be >= 0" in e for e in errors)

    def test_high_ef_warning(self, engine):
        """EF > 2.0 returns warning."""
        errors = engine.validate_emission_factor(Decimal("2.5"))
        assert any("WARNING" in e and "2.0" in e for e in errors)

    def test_invalid_type_ef(self, engine):
        """Non-convertible EF returns error."""
        errors = engine.validate_emission_factor("not_valid")
        assert len(errors) > 0


# ===========================================================================
# 11. TestStatisticsAndReset
# ===========================================================================


@_SKIP
class TestStatisticsAndReset:
    """Tests for get_statistics and reset."""

    def test_initial_statistics(self, engine):
        """Fresh engine has zero counters."""
        stats = engine.get_statistics()
        assert stats["total_calculations"] == 0
        assert stats["total_batches"] == 0
        assert stats["total_co2e_kg_processed"] == Decimal("0")
        assert stats["provenance_enabled"] is True

    def test_statistics_after_calculation(self, engine):
        """Counters increment after calculations."""
        engine.calculate_covered_emissions("REC", Decimal("100"), Decimal("0.000"))
        engine.calculate_uncovered_emissions(Decimal("100"), "US")
        stats = engine.get_statistics()

        assert stats["total_calculations"] == 2
        assert stats["total_covered_mwh"] == Decimal("100")
        assert stats["total_uncovered_mwh"] == Decimal("100")

    def test_reset_clears_counters(self, engine):
        """Reset sets all counters back to zero."""
        engine.calculate_covered_emissions("REC", Decimal("1000"), Decimal("0.500"))
        engine.reset()
        stats = engine.get_statistics()

        assert stats["total_calculations"] == 0
        assert stats["total_co2e_kg_processed"] == Decimal("0")

    def test_statistics_include_reference_data(self, engine):
        """Statistics include reference lists."""
        stats = engine.get_statistics()

        assert "REC" in stats["supported_instrument_types"]
        assert "US" in stats["supported_residual_mix_regions"]
        assert "AR5" in stats["supported_gwp_sources"]
        assert "solar" in stats["supported_renewable_sources"]


# ===========================================================================
# 12. TestProvenanceDisabled
# ===========================================================================


@_SKIP
class TestProvenanceDisabled:
    """Tests when provenance is disabled."""

    def test_calculation_works_without_provenance(self, engine_no_provenance):
        """Calculations still work when provenance is disabled."""
        result = engine_no_provenance.calculate_covered_emissions(
            instrument_type="REC",
            mwh=Decimal("100"),
            ef_kgco2e_kwh=Decimal("0.000"),
        )
        assert result["total_co2e_kg"] == Decimal("0.000")
        # provenance_hash is still computed (inline SHA-256), but no tracker entries
        assert len(result["provenance_hash"]) == 64

    def test_statistics_show_provenance_disabled(self, engine_no_provenance):
        """Statistics reflect provenance_enabled=False."""
        stats = engine_no_provenance.get_statistics()
        assert stats["provenance_enabled"] is False
        assert stats["provenance_entry_count"] == 0


# ===========================================================================
# 13. TestExternalResidualMixDB
# ===========================================================================


@_SKIP
class TestExternalResidualMixDB:
    """Tests for external residual mix database integration."""

    def test_external_db_used_when_provided(self):
        """External residual mix DB is queried first."""
        mock_db = Mock()
        mock_db.get_residual_mix_factor.return_value = {
            "ef_kgco2e_kwh": Decimal("0.550"),
            "source": "EXTERNAL_TEST_DB",
        }
        engine = MarketEmissionsCalculatorEngine(residual_mix_db=mock_db)

        result = engine.calculate_uncovered_emissions(
            mwh=Decimal("100"),
            region="US",
        )
        # Should use external DB value (0.550) not built-in (0.425)
        assert result["total_co2e_kg"] == Decimal("55000.000")
        assert result["ef_source"] == "EXTERNAL_TEST_DB"

    def test_fallback_on_external_db_failure(self):
        """Built-in default used when external DB raises exception."""
        mock_db = Mock()
        mock_db.get_residual_mix_factor.side_effect = Exception("DB connection error")
        engine = MarketEmissionsCalculatorEngine(residual_mix_db=mock_db)

        result = engine.calculate_uncovered_emissions(
            mwh=Decimal("100"),
            region="US",
        )
        # Should fallback to built-in US default (0.425)
        assert result["total_co2e_kg"] == Decimal("42500.000")

    def test_external_db_returns_none_falls_back(self):
        """Built-in default used when external DB returns None."""
        mock_db = Mock()
        mock_db.get_residual_mix_factor.return_value = None
        engine = MarketEmissionsCalculatorEngine(residual_mix_db=mock_db)

        result = engine.calculate_uncovered_emissions(
            mwh=Decimal("100"),
            region="DE",
        )
        # Fallback to DE built-in: 0.560 -> 100 * 560 = 56,000
        assert result["total_co2e_kg"] == Decimal("56000.000")
