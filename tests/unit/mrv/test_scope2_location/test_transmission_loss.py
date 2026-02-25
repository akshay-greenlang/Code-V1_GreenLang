# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-009 TransmissionLossEngine.

Tests T&D loss calculation, gross/net conversions, loss emission
calculations, country factor lookups, custom factor management,
proportional allocation, on-site generation deductions, validation,
and statistical comparisons.

Target: 35+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

import pytest

try:
    from greenlang.scope2_location.transmission_loss import (
        TransmissionLossEngine,
        TDLossResult,
        TD_LOSS_FACTORS,
        REGIONAL_GROUPS,
        VALID_COUNTRY_CODES,
    )
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False

_SKIP = pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine not available")

# Precision constants matching the engine
_PRECISION = Decimal("0.00000001")
_OUTPUT = Decimal("0.001")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """Create a default TransmissionLossEngine."""
    return TransmissionLossEngine()


@pytest.fixture
def engine_no_fallback():
    """Create engine with fallback_to_world disabled."""
    return TransmissionLossEngine(
        config={"fallback_to_world": False}
    )


@pytest.fixture
def engine_high_precision():
    """Create engine with 6-digit output precision."""
    return TransmissionLossEngine(
        config={"default_precision": 6}
    )


# ===========================================================================
# TestTDLossCalculation
# ===========================================================================


@_SKIP
class TestTDLossCalculation:
    """Tests for calculate_td_loss with known country values."""

    def test_us_1000_mwh(self, engine):
        """US: 1000 MWh x (1 + 0.050) = 1050.000 MWh gross."""
        result = engine.calculate_td_loss(
            net_consumption_mwh=Decimal("1000"),
            country_code="US",
        )
        assert isinstance(result, TDLossResult)
        assert result.gross_consumption_mwh == Decimal("1050.000")
        assert result.loss_mwh == Decimal("50.000")
        assert result.td_loss_pct == Decimal("0.050")

    def test_gb_5000_mwh(self, engine):
        """GB: 5000 MWh x (1 + 0.077) = 5385.000 MWh gross."""
        result = engine.calculate_td_loss(
            net_consumption_mwh=Decimal("5000"),
            country_code="GB",
        )
        assert result.gross_consumption_mwh == Decimal("5385.000")
        assert result.loss_mwh == Decimal("385.000")

    def test_in_high_losses(self, engine):
        """India: 1000 MWh x (1 + 0.194) = 1194.000 MWh gross."""
        result = engine.calculate_td_loss(
            net_consumption_mwh=Decimal("1000"),
            country_code="IN",
        )
        assert result.gross_consumption_mwh == Decimal("1194.000")
        assert result.loss_mwh == Decimal("194.000")

    def test_custom_td_pct_override(self, engine):
        """Custom T&D percentage overrides country default."""
        result = engine.calculate_td_loss(
            net_consumption_mwh=Decimal("1000"),
            country_code="US",
            custom_td_pct=Decimal("0.10"),
        )
        assert result.td_loss_pct == Decimal("0.10")
        assert result.gross_consumption_mwh == Decimal("1100.000")
        assert result.method == "custom"

    def test_with_grid_ef_calculates_loss_emissions(self, engine):
        """When grid_ef_co2e is provided, loss emissions are calculated."""
        result = engine.calculate_td_loss(
            net_consumption_mwh=Decimal("1000"),
            country_code="US",
            grid_ef_co2e=Decimal("400"),
        )
        # Loss emissions = 1000 x 0.050 x 400 = 20,000 kg
        expected = (Decimal("1000") * Decimal("0.050") * Decimal("400")).quantize(
            _OUTPUT, ROUND_HALF_UP
        )
        assert result.loss_emissions_kg == expected

    def test_zero_consumption(self, engine):
        """Zero consumption returns zero gross and zero loss."""
        result = engine.calculate_td_loss(
            net_consumption_mwh=Decimal("0"),
            country_code="US",
        )
        assert result.gross_consumption_mwh == Decimal("0.000")
        assert result.loss_mwh == Decimal("0.000")

    def test_negative_consumption_raises(self, engine):
        """Negative consumption raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 0"):
            engine.calculate_td_loss(
                net_consumption_mwh=Decimal("-100"),
                country_code="US",
            )

    def test_result_has_provenance_hash(self, engine):
        """Result includes a 64-character SHA-256 provenance hash."""
        result = engine.calculate_td_loss(
            net_consumption_mwh=Decimal("1000"),
            country_code="US",
        )
        assert len(result.provenance_hash) == 64

    def test_result_has_calculation_steps(self, engine):
        """Result includes non-empty calculation steps for audit."""
        result = engine.calculate_td_loss(
            net_consumption_mwh=Decimal("1000"),
            country_code="US",
        )
        assert len(result.calculation_steps) >= 3

    def test_deterministic_provenance(self, engine):
        """Same inputs produce identical provenance hashes."""
        r1 = engine.calculate_td_loss(Decimal("1000"), "US")
        r2 = engine.calculate_td_loss(Decimal("1000"), "US")
        assert r1.provenance_hash == r2.provenance_hash

    def test_lowercase_country_code(self, engine):
        """Lowercase country codes are normalized."""
        result = engine.calculate_td_loss(Decimal("1000"), "us")
        assert result.country_code == "US"


# ===========================================================================
# TestGrossNetConversion
# ===========================================================================


@_SKIP
class TestGrossNetConversion:
    """Tests for get_gross_consumption and get_net_consumption."""

    def test_get_gross_consumption_us(self, engine):
        """1000 MWh x (1 + 0.05) = 1050 MWh."""
        gross = engine.get_gross_consumption(
            net_mwh=Decimal("1000"),
            td_loss_pct=Decimal("0.05"),
        )
        expected = (Decimal("1000") * Decimal("1.05")).quantize(
            _PRECISION, ROUND_HALF_UP
        )
        assert gross == expected

    def test_get_net_consumption_roundtrip(self, engine):
        """Net -> Gross -> Net roundtrip preserves value (within precision)."""
        original_net = Decimal("1000")
        td_pct = Decimal("0.077")
        gross = engine.get_gross_consumption(original_net, td_pct)
        recovered_net = engine.get_net_consumption(gross, td_pct)
        assert abs(recovered_net - original_net) < Decimal("0.01")

    def test_get_net_consumption_from_gross(self, engine):
        """1050 MWh / (1 + 0.05) = 1000 MWh."""
        net = engine.get_net_consumption(
            gross_mwh=Decimal("1050"),
            td_loss_pct=Decimal("0.05"),
        )
        expected = (Decimal("1050") / Decimal("1.05")).quantize(
            _PRECISION, ROUND_HALF_UP
        )
        assert net == expected

    def test_negative_net_raises(self, engine):
        """Negative net_mwh raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 0"):
            engine.get_gross_consumption(
                net_mwh=Decimal("-100"),
                td_loss_pct=Decimal("0.05"),
            )

    def test_negative_gross_raises(self, engine):
        """Negative gross_mwh raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 0"):
            engine.get_net_consumption(
                gross_mwh=Decimal("-100"),
                td_loss_pct=Decimal("0.05"),
            )


# ===========================================================================
# TestLossEmissions
# ===========================================================================


@_SKIP
class TestLossEmissions:
    """Tests for calculate_loss_emissions."""

    def test_loss_emissions_formula(self, engine):
        """loss_emissions = 1000 x 0.05 x 400 = 20,000 kg."""
        result = engine.calculate_loss_emissions(
            net_mwh=Decimal("1000"),
            td_loss_pct=Decimal("0.05"),
            grid_ef_co2e=Decimal("400"),
        )
        expected = (Decimal("1000") * Decimal("0.05") * Decimal("400")).quantize(
            _PRECISION, ROUND_HALF_UP
        )
        assert result == expected

    def test_loss_emissions_india(self, engine):
        """India: 1000 x 0.194 x 708 = 137,352 kg."""
        result = engine.calculate_loss_emissions(
            net_mwh=Decimal("1000"),
            td_loss_pct=Decimal("0.194"),
            grid_ef_co2e=Decimal("708"),
        )
        expected = (Decimal("1000") * Decimal("0.194") * Decimal("708")).quantize(
            _PRECISION, ROUND_HALF_UP
        )
        assert result == expected

    def test_zero_td_loss_zero_emissions(self, engine):
        """Zero T&D loss produces zero loss emissions."""
        result = engine.calculate_loss_emissions(
            net_mwh=Decimal("1000"),
            td_loss_pct=Decimal("0"),
            grid_ef_co2e=Decimal("400"),
        )
        assert result == Decimal("0")

    def test_negative_net_raises(self, engine):
        """Negative net_mwh raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 0"):
            engine.calculate_loss_emissions(
                net_mwh=Decimal("-100"),
                td_loss_pct=Decimal("0.05"),
                grid_ef_co2e=Decimal("400"),
            )


# ===========================================================================
# TestFactorLookup
# ===========================================================================


@_SKIP
class TestFactorLookup:
    """Tests for get_td_loss_factor across 50+ countries."""

    @pytest.mark.parametrize("country,expected_pct", [
        ("US", Decimal("0.050")),
        ("GB", Decimal("0.077")),
        ("DE", Decimal("0.040")),
        ("FR", Decimal("0.060")),
        ("IN", Decimal("0.194")),
        ("CN", Decimal("0.058")),
        ("JP", Decimal("0.050")),
        ("BR", Decimal("0.156")),
        ("AU", Decimal("0.055")),
        ("SG", Decimal("0.025")),
        ("KE", Decimal("0.200")),
        ("NG", Decimal("0.180")),
    ])
    def test_known_countries(self, engine, country, expected_pct):
        """Known country T&D loss factors match reference values."""
        result = engine.get_td_loss_factor(country)
        assert result == expected_pct

    def test_world_average(self, engine):
        """WORLD average is 8.3%."""
        result = engine.get_td_loss_factor("WORLD")
        assert result == Decimal("0.083")

    def test_unknown_country_fallback(self, engine):
        """Unknown country falls back to WORLD average."""
        result = engine.get_td_loss_factor("ZZ")
        assert result == Decimal("0.083")

    def test_unknown_country_no_fallback_raises(self, engine_no_fallback):
        """Without fallback, unknown country raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            engine_no_fallback.get_td_loss_factor("ZZ")

    def test_factor_database_has_53_plus_countries(self):
        """T&D database has 53+ country entries (plus WORLD)."""
        assert len(TD_LOSS_FACTORS) >= 53

    def test_all_factors_are_decimal(self):
        """All T&D factors in the database are Decimal instances."""
        for code, record in TD_LOSS_FACTORS.items():
            assert isinstance(record["td_loss_pct"], Decimal), (
                f"Factor for {code} is not Decimal"
            )


# ===========================================================================
# TestCustomFactors
# ===========================================================================


@_SKIP
class TestCustomFactors:
    """Tests for set_custom_factor, remove_custom_factor."""

    def test_set_custom_factor(self, engine):
        """Custom factor overrides built-in."""
        engine.set_custom_factor("US", Decimal("0.065"))
        assert engine.get_td_loss_factor("US") == Decimal("0.065")

    def test_remove_custom_factor(self, engine):
        """Removing custom factor reverts to built-in."""
        engine.set_custom_factor("US", Decimal("0.065"))
        removed = engine.remove_custom_factor("US")
        assert removed is True
        assert engine.get_td_loss_factor("US") == Decimal("0.050")

    def test_remove_nonexistent_returns_false(self, engine):
        """Removing a nonexistent custom factor returns False."""
        assert engine.remove_custom_factor("ZZ") is False

    def test_list_custom_factors_empty(self, engine):
        """list_custom_factors returns empty dict initially."""
        assert engine.list_custom_factors() == {}

    def test_list_custom_factors_after_set(self, engine):
        """list_custom_factors returns set factors."""
        engine.set_custom_factor("US", Decimal("0.065"), name="Custom US")
        factors = engine.list_custom_factors()
        assert "US" in factors
        assert factors["US"]["td_loss_pct"] == str(Decimal("0.065"))

    def test_invalid_custom_factor_raises(self, engine):
        """Custom factor > 50% raises ValueError."""
        with pytest.raises(ValueError, match="Invalid custom"):
            engine.set_custom_factor("US", Decimal("0.60"))


# ===========================================================================
# TestAllocation
# ===========================================================================


@_SKIP
class TestAllocation:
    """Tests for allocate_proportional."""

    def test_proportional_equal_shares(self, engine):
        """Equal consumption shares split losses equally."""
        allocations = engine.allocate_proportional(
            total_loss_mwh=Decimal("100"),
            consumption_shares={
                "A": Decimal("500"),
                "B": Decimal("500"),
            },
        )
        assert allocations["A"] == Decimal("50.000")
        assert allocations["B"] == Decimal("50.000")

    def test_proportional_unequal_shares(self, engine):
        """Unequal shares produce proportional allocations."""
        allocations = engine.allocate_proportional(
            total_loss_mwh=Decimal("100"),
            consumption_shares={
                "A": Decimal("750"),
                "B": Decimal("250"),
            },
        )
        assert allocations["A"] == Decimal("75.000")
        assert allocations["B"] == Decimal("25.000")

    def test_proportional_empty_shares_raises(self, engine):
        """Empty consumption_shares raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            engine.allocate_proportional(
                total_loss_mwh=Decimal("100"),
                consumption_shares={},
            )


# ===========================================================================
# TestOnsiteGeneration
# ===========================================================================


@_SKIP
class TestOnsiteGeneration:
    """Tests for deduct_onsite_generation."""

    def test_deduction_basic(self, engine):
        """1000 - 200 = 800 MWh net grid consumption."""
        result = engine.deduct_onsite_generation(
            total_consumption_mwh=Decimal("1000"),
            onsite_generation_mwh=Decimal("200"),
        )
        assert result == Decimal("800.000")

    def test_deduction_clamps_to_zero(self, engine):
        """On-site generation > total is clamped to zero."""
        result = engine.deduct_onsite_generation(
            total_consumption_mwh=Decimal("100"),
            onsite_generation_mwh=Decimal("500"),
        )
        assert result == Decimal("0.000")

    def test_deduction_equal_values(self, engine):
        """Equal total and on-site produces zero."""
        result = engine.deduct_onsite_generation(
            total_consumption_mwh=Decimal("1000"),
            onsite_generation_mwh=Decimal("1000"),
        )
        assert result == Decimal("0.000")

    def test_negative_total_raises(self, engine):
        """Negative total_consumption raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 0"):
            engine.deduct_onsite_generation(
                total_consumption_mwh=Decimal("-100"),
                onsite_generation_mwh=Decimal("50"),
            )

    def test_negative_onsite_raises(self, engine):
        """Negative onsite_generation raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 0"):
            engine.deduct_onsite_generation(
                total_consumption_mwh=Decimal("1000"),
                onsite_generation_mwh=Decimal("-50"),
            )


# ===========================================================================
# TestValidation
# ===========================================================================


@_SKIP
class TestValidation:
    """Tests for validate_td_loss_factor and validate_country_code."""

    def test_valid_factor_no_errors(self, engine):
        """Valid T&D factor produces empty error list."""
        errors = engine.validate_td_loss_factor(Decimal("0.05"))
        assert errors == []

    def test_negative_factor_flagged(self, engine):
        """Negative T&D factor is flagged."""
        errors = engine.validate_td_loss_factor(Decimal("-0.01"))
        assert len(errors) > 0

    def test_above_50_pct_flagged(self, engine):
        """T&D factor > 50% is flagged."""
        errors = engine.validate_td_loss_factor(Decimal("0.55"))
        assert any("exceeds maximum" in e for e in errors)

    def test_zero_factor_valid(self, engine):
        """Zero T&D factor is valid."""
        errors = engine.validate_td_loss_factor(Decimal("0"))
        assert errors == []

    def test_validate_known_country_code(self, engine):
        """Known country code validates as True."""
        assert engine.validate_country_code("US") is True
        assert engine.validate_country_code("GB") is True
        assert engine.validate_country_code("WORLD") is True

    def test_validate_unknown_country_code(self, engine):
        """Unknown country code validates as False."""
        assert engine.validate_country_code("ZZ") is False

    def test_validate_custom_country_code(self, engine):
        """Custom country code validates as True after set."""
        engine.set_custom_factor("XX", Decimal("0.08"))
        assert engine.validate_country_code("XX") is True


# ===========================================================================
# TestStatistics
# ===========================================================================


@_SKIP
class TestStatistics:
    """Tests for compare_countries and get_highest/lowest_loss_countries."""

    def test_compare_countries_basic(self, engine):
        """compare_countries returns correct ordering."""
        result = engine.compare_countries(["US", "GB", "IN"])
        assert result["country_count"] == 3
        assert Decimal(result["lowest"]["td_loss_pct"]) == Decimal("0.050")
        assert Decimal(result["highest"]["td_loss_pct"]) == Decimal("0.194")

    def test_compare_countries_range(self, engine):
        """Range is highest - lowest."""
        result = engine.compare_countries(["US", "IN"])
        expected_range = Decimal("0.194") - Decimal("0.050")
        assert Decimal(result["range_td_loss_pct"]) == expected_range.quantize(
            Decimal("0.0001"), ROUND_HALF_UP
        )

    def test_compare_empty_list_raises(self, engine):
        """Empty codes list raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            engine.compare_countries([])

    def test_get_highest_loss_countries(self, engine):
        """get_highest_loss_countries returns sorted descending."""
        top5 = engine.get_highest_loss_countries(top_n=5)
        assert len(top5) == 5
        pcts = [Decimal(c["td_loss_pct"]) for c in top5]
        assert pcts == sorted(pcts, reverse=True)

    def test_get_lowest_loss_countries(self, engine):
        """get_lowest_loss_countries returns sorted ascending."""
        bottom5 = engine.get_lowest_loss_countries(top_n=5)
        assert len(bottom5) == 5
        pcts = [Decimal(c["td_loss_pct"]) for c in bottom5]
        assert pcts == sorted(pcts)

    def test_get_statistics_summary(self, engine):
        """get_statistics returns valid statistical summary."""
        stats = engine.get_statistics()
        assert stats["country_count"] >= 50
        assert Decimal(stats["min_td_loss_pct"]) < Decimal(stats["max_td_loss_pct"])
        assert stats["world_average_td_loss_pct"] == str(Decimal("0.083"))

    def test_regional_average_eu(self, engine):
        """EU regional average is within reasonable range (4-10%)."""
        avg = engine.get_regional_average("EU")
        assert Decimal("0.03") < avg < Decimal("0.12")

    def test_regional_average_unknown_raises(self, engine):
        """Unknown region raises ValueError."""
        with pytest.raises(ValueError, match="Unknown region"):
            engine.get_regional_average("ANTARCTIC")

    def test_td_loss_result_to_dict(self, engine):
        """TDLossResult.to_dict produces serializable dictionary."""
        result = engine.calculate_td_loss(Decimal("1000"), "US")
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["country_code"] == "US"
        assert d["gross_consumption_mwh"] == "1050.000"
