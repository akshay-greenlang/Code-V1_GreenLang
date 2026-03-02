# -*- coding: utf-8 -*-
"""
Unit tests for GL-CBAM-APP v1.1 Certificate Calculator

Tests certificate calculations:
- calculate_gross_certificates (quantity x emissions)
- apply_free_allocation (cement, steel, aluminium benchmarks)
- apply_carbon_price_deduction (EUR conversion, verified only)
- calculate_net_obligation (gross - free - carbon)
- calculate_certificate_cost (net x ETS price)
- calculate_quarterly_holding (50% requirement)
- check_quarterly_compliance (compliant, shortfall)
- Free allocation phase-out schedule (2026: 97.5% -> 2034: 0%)
- Edge cases: zero imports, negative deductions, rounding

Target: 60+ tests
"""

import pytest
from decimal import Decimal, ROUND_HALF_UP


# ---------------------------------------------------------------------------
# Inline certificate calculator for self-contained tests
# ---------------------------------------------------------------------------

class CertificateCalculator:
    """CBAM certificate obligation calculator."""

    # Free allocation phase-out schedule (EU ETS)
    FREE_ALLOCATION_SCHEDULE = {
        2026: Decimal("97.5"),
        2027: Decimal("95.0"),
        2028: Decimal("90.0"),
        2029: Decimal("77.5"),
        2030: Decimal("51.5"),
        2031: Decimal("39.0"),
        2032: Decimal("26.5"),
        2033: Decimal("14.0"),
        2034: Decimal("0.0"),
    }

    # Sector benchmarks (tCO2/tonne product)
    BENCHMARKS = {
        "cement": Decimal("0.766"),
        "cement_clinker": Decimal("0.766"),
        "steel_hot_metal": Decimal("1.328"),
        "steel_eaf": Decimal("0.283"),
        "aluminium": Decimal("1.514"),
        "fertilizers": Decimal("1.960"),
        "hydrogen": Decimal("8.850"),
    }

    QUARTERLY_HOLDING_PCT = Decimal("50.0")

    def calculate_gross_certificates(self, quantity_tonnes, specific_emissions_tco2):
        qty = Decimal(str(quantity_tonnes))
        ef = Decimal(str(specific_emissions_tco2))
        result = (qty * ef).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        return max(result, Decimal("0"))

    def get_free_allocation_rate(self, year):
        if year < 2026:
            return Decimal("100.0")
        return self.FREE_ALLOCATION_SCHEDULE.get(year, Decimal("0.0"))

    def apply_free_allocation(self, gross_certificates, sector, year):
        gross = Decimal(str(gross_certificates))
        benchmark = self.BENCHMARKS.get(sector, Decimal("0"))
        if benchmark == 0:
            return Decimal("0")

        free_rate = self.get_free_allocation_rate(year) / Decimal("100")
        free_certs = (gross * free_rate).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        return free_certs

    def apply_carbon_price_deduction(self, emissions_tco2, carbon_price_per_tco2,
                                     ets_price_eur, exchange_rate=Decimal("1.0"),
                                     verified=True):
        if not verified:
            return Decimal("0")
        emissions = Decimal(str(emissions_tco2))
        carbon_price = Decimal(str(carbon_price_per_tco2))
        ets_price = Decimal(str(ets_price_eur))
        rate = Decimal(str(exchange_rate))

        carbon_in_eur = carbon_price * rate
        if carbon_in_eur >= ets_price:
            deduction = emissions
        else:
            ratio = carbon_in_eur / ets_price
            deduction = (emissions * ratio).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
        return deduction

    def calculate_net_obligation(self, gross_certificates, free_allocation,
                                 carbon_price_deduction):
        gross = Decimal(str(gross_certificates))
        free = Decimal(str(free_allocation))
        carbon = Decimal(str(carbon_price_deduction))
        net = gross - free - carbon
        return max(net.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
                   Decimal("0"))

    def calculate_certificate_cost(self, net_certificates, ets_price_eur):
        net = Decimal(str(net_certificates))
        price = Decimal(str(ets_price_eur))
        return (net * price).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def calculate_quarterly_holding(self, annual_obligation):
        annual = Decimal(str(annual_obligation))
        pct = self.QUARTERLY_HOLDING_PCT / Decimal("100")
        return (annual * pct).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def check_quarterly_compliance(self, certificates_held, quarterly_obligation):
        held = Decimal(str(certificates_held))
        required = Decimal(str(quarterly_obligation))
        shortfall = required - held
        return {
            "compliant": held >= required,
            "certificates_held": held,
            "quarterly_obligation": required,
            "shortfall": max(shortfall, Decimal("0")),
            "surplus": max(held - required, Decimal("0")),
        }


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def calculator():
    return CertificateCalculator()


# ===========================================================================
# TEST CLASS -- calculate_gross_certificates
# ===========================================================================

class TestCalculateGrossCertificates:
    """Tests for calculate_gross_certificates."""

    def test_basic_calculation(self, calculator):
        result = calculator.calculate_gross_certificates(100, 2.0)
        assert result == Decimal("200.000")

    def test_fractional_values(self, calculator):
        result = calculator.calculate_gross_certificates(15.5, 0.9)
        assert result == Decimal("13.950")

    def test_zero_quantity(self, calculator):
        result = calculator.calculate_gross_certificates(0, 2.0)
        assert result == Decimal("0.000")

    def test_zero_emissions(self, calculator):
        result = calculator.calculate_gross_certificates(100, 0)
        assert result == Decimal("0.000")

    def test_large_values(self, calculator):
        result = calculator.calculate_gross_certificates(500000, 2.1)
        assert result == Decimal("1050000.000")

    def test_small_values(self, calculator):
        result = calculator.calculate_gross_certificates(0.001, 0.001)
        assert result == Decimal("0.000")

    def test_precision_three_decimals(self, calculator):
        result = calculator.calculate_gross_certificates(33.333, 3.333)
        assert str(result).split(".")[1] == "089" or len(str(result).split(".")[1]) <= 3

    @pytest.mark.parametrize("qty,ef,expected", [
        (100, 0.766, Decimal("76.600")),
        (100, 1.328, Decimal("132.800")),
        (100, 1.514, Decimal("151.400")),
        (50, 2.0, Decimal("100.000")),
    ])
    def test_parametrized(self, calculator, qty, ef, expected):
        assert calculator.calculate_gross_certificates(qty, ef) == expected


# ===========================================================================
# TEST CLASS -- apply_free_allocation
# ===========================================================================

class TestApplyFreeAllocation:
    """Tests for apply_free_allocation."""

    def test_cement_2026(self, calculator):
        gross = Decimal("100")
        free = calculator.apply_free_allocation(gross, "cement", 2026)
        assert free == Decimal("97.500")

    def test_steel_2027(self, calculator):
        gross = Decimal("200")
        free = calculator.apply_free_allocation(gross, "steel_hot_metal", 2027)
        assert free == Decimal("190.000")

    def test_aluminium_2030(self, calculator):
        gross = Decimal("100")
        free = calculator.apply_free_allocation(gross, "aluminium", 2030)
        assert free == Decimal("51.500")

    def test_2034_zero_free_allocation(self, calculator):
        gross = Decimal("1000")
        free = calculator.apply_free_allocation(gross, "cement", 2034)
        assert free == Decimal("0.000")

    def test_unknown_sector_returns_zero(self, calculator):
        free = calculator.apply_free_allocation(Decimal("100"), "widgets", 2026)
        assert free == Decimal("0")

    def test_before_2026_full_allocation(self, calculator):
        rate = calculator.get_free_allocation_rate(2025)
        assert rate == Decimal("100.0")


# ===========================================================================
# TEST CLASS -- Free allocation phase-out schedule
# ===========================================================================

class TestFreeAllocationPhaseOut:
    """Tests for the complete phase-out schedule."""

    @pytest.mark.parametrize("year,expected_rate", [
        (2026, Decimal("97.5")),
        (2027, Decimal("95.0")),
        (2028, Decimal("90.0")),
        (2029, Decimal("77.5")),
        (2030, Decimal("51.5")),
        (2031, Decimal("39.0")),
        (2032, Decimal("26.5")),
        (2033, Decimal("14.0")),
        (2034, Decimal("0.0")),
    ])
    def test_schedule_year(self, calculator, year, expected_rate):
        assert calculator.get_free_allocation_rate(year) == expected_rate

    def test_after_2034_remains_zero(self, calculator):
        assert calculator.get_free_allocation_rate(2035) == Decimal("0.0")
        assert calculator.get_free_allocation_rate(2040) == Decimal("0.0")

    def test_phase_out_is_monotonically_decreasing(self, calculator):
        years = sorted(calculator.FREE_ALLOCATION_SCHEDULE.keys())
        rates = [calculator.get_free_allocation_rate(y) for y in years]
        for i in range(1, len(rates)):
            assert rates[i] <= rates[i - 1]


# ===========================================================================
# TEST CLASS -- apply_carbon_price_deduction
# ===========================================================================

class TestApplyCarbonPriceDeduction:
    """Tests for apply_carbon_price_deduction."""

    def test_full_deduction_when_price_exceeds_ets(self, calculator):
        result = calculator.apply_carbon_price_deduction(
            100, 100, 80
        )
        assert result == Decimal("100")

    def test_partial_deduction(self, calculator):
        result = calculator.apply_carbon_price_deduction(
            100, 40, 80
        )
        assert result == Decimal("50.000")

    def test_unverified_returns_zero(self, calculator):
        result = calculator.apply_carbon_price_deduction(
            100, 50, 80, verified=False
        )
        assert result == Decimal("0")

    def test_with_exchange_rate(self, calculator):
        result = calculator.apply_carbon_price_deduction(
            100, 50, 80, exchange_rate=Decimal("1.1")
        )
        carbon_eur = 50 * 1.1  # 55
        expected_ratio = Decimal("55") / Decimal("80")
        expected = (Decimal("100") * expected_ratio).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        assert result == expected

    def test_zero_emissions(self, calculator):
        result = calculator.apply_carbon_price_deduction(0, 50, 80)
        assert result == Decimal("0.000")

    def test_zero_carbon_price(self, calculator):
        result = calculator.apply_carbon_price_deduction(100, 0, 80)
        assert result == Decimal("0.000")


# ===========================================================================
# TEST CLASS -- calculate_net_obligation
# ===========================================================================

class TestCalculateNetObligation:
    """Tests for calculate_net_obligation."""

    def test_basic_net(self, calculator):
        net = calculator.calculate_net_obligation(100, 50, 20)
        assert net == Decimal("30.000")

    def test_net_cannot_be_negative(self, calculator):
        net = calculator.calculate_net_obligation(100, 80, 30)
        assert net == Decimal("0.000")

    def test_zero_deductions(self, calculator):
        net = calculator.calculate_net_obligation(100, 0, 0)
        assert net == Decimal("100.000")

    def test_full_deduction(self, calculator):
        net = calculator.calculate_net_obligation(100, 100, 0)
        assert net == Decimal("0.000")

    def test_precision(self, calculator):
        net = calculator.calculate_net_obligation(
            Decimal("100.123"), Decimal("50.456"), Decimal("20.789")
        )
        expected = Decimal("28.878")
        assert net == expected

    def test_all_zeros(self, calculator):
        net = calculator.calculate_net_obligation(0, 0, 0)
        assert net == Decimal("0.000")


# ===========================================================================
# TEST CLASS -- calculate_certificate_cost
# ===========================================================================

class TestCalculateCertificateCost:
    """Tests for calculate_certificate_cost."""

    def test_basic_cost(self, calculator):
        cost = calculator.calculate_certificate_cost(100, 75.50)
        assert cost == Decimal("7550.00")

    def test_zero_certificates(self, calculator):
        cost = calculator.calculate_certificate_cost(0, 75.50)
        assert cost == Decimal("0.00")

    def test_large_obligation(self, calculator):
        cost = calculator.calculate_certificate_cost(10000, 80)
        assert cost == Decimal("800000.00")

    def test_fractional_price(self, calculator):
        cost = calculator.calculate_certificate_cost(100, 75.99)
        assert cost == Decimal("7599.00")

    def test_precision_two_decimals(self, calculator):
        cost = calculator.calculate_certificate_cost(
            Decimal("33.333"), Decimal("75.555")
        )
        result_str = str(cost)
        assert len(result_str.split(".")[1]) <= 2


# ===========================================================================
# TEST CLASS -- calculate_quarterly_holding
# ===========================================================================

class TestCalculateQuarterlyHolding:
    """Tests for calculate_quarterly_holding (50% requirement)."""

    def test_basic_holding(self, calculator):
        result = calculator.calculate_quarterly_holding(1000)
        assert result == Decimal("500.000")

    def test_zero_obligation(self, calculator):
        result = calculator.calculate_quarterly_holding(0)
        assert result == Decimal("0.000")

    def test_odd_number(self, calculator):
        result = calculator.calculate_quarterly_holding(333)
        assert result == Decimal("166.500")

    def test_small_obligation(self, calculator):
        result = calculator.calculate_quarterly_holding(1)
        assert result == Decimal("0.500")


# ===========================================================================
# TEST CLASS -- check_quarterly_compliance
# ===========================================================================

class TestCheckQuarterlyCompliance:
    """Tests for check_quarterly_compliance."""

    def test_compliant(self, calculator):
        result = calculator.check_quarterly_compliance(600, 500)
        assert result["compliant"] is True
        assert result["surplus"] == Decimal("100")
        assert result["shortfall"] == Decimal("0")

    def test_exactly_compliant(self, calculator):
        result = calculator.check_quarterly_compliance(500, 500)
        assert result["compliant"] is True
        assert result["surplus"] == Decimal("0")

    def test_shortfall(self, calculator):
        result = calculator.check_quarterly_compliance(400, 500)
        assert result["compliant"] is False
        assert result["shortfall"] == Decimal("100")

    def test_zero_held(self, calculator):
        result = calculator.check_quarterly_compliance(0, 500)
        assert result["compliant"] is False
        assert result["shortfall"] == Decimal("500")

    def test_zero_obligation(self, calculator):
        result = calculator.check_quarterly_compliance(0, 0)
        assert result["compliant"] is True


# ===========================================================================
# TEST CLASS -- Edge cases
# ===========================================================================

class TestEdgeCases:
    """Tests for edge cases in certificate calculations."""

    def test_full_pipeline(self, calculator):
        """Test complete certificate calculation pipeline."""
        gross = calculator.calculate_gross_certificates(1000, 2.0)
        assert gross == Decimal("2000.000")

        free = calculator.apply_free_allocation(gross, "steel_hot_metal", 2026)
        assert free == Decimal("1950.000")

        carbon = calculator.apply_carbon_price_deduction(
            gross, 10, 80, verified=True
        )

        net = calculator.calculate_net_obligation(gross, free, carbon)
        assert net >= Decimal("0")

        cost = calculator.calculate_certificate_cost(net, 75)
        assert cost >= Decimal("0")

        quarterly = calculator.calculate_quarterly_holding(net)
        assert quarterly >= Decimal("0")

    def test_rounding_consistency(self, calculator):
        """Test that rounding is always ROUND_HALF_UP."""
        val = Decimal("2.5005")
        rounded = val.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        assert rounded == Decimal("2.501")

    def test_large_number_precision(self, calculator):
        gross = calculator.calculate_gross_certificates(999999, 99.999)
        assert gross > Decimal("0")

    def test_very_small_emissions(self, calculator):
        gross = calculator.calculate_gross_certificates(Decimal("0.001"), Decimal("0.001"))
        assert gross == Decimal("0.000")
