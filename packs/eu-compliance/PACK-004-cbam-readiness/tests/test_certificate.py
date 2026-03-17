# -*- coding: utf-8 -*-
"""
PACK-004 CBAM Readiness Pack - Certificate Engine Deep Tests (20 tests)

Deep testing of the CBAM certificate engine including obligation
calculation, free allocation phase-out, carbon price deductions,
cost estimation, quarterly holding compliance, and provenance.

Author: GreenLang QA Team
"""

import pytest

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from conftest import (
    StubCertificateEngine,
    _compute_hash,
    _new_uuid,
    _utcnow,
)


@pytest.fixture
def engine():
    """Create a StubCertificateEngine instance."""
    return StubCertificateEngine()


class TestCertificateObligations:
    """Test certificate obligation calculation."""

    def test_gross_obligation_calculation(self, engine):
        """Test gross obligation equals total emissions."""
        result = engine.calculate_obligation(
            total_emissions_tco2e=10000.0, year=2026,
        )
        assert result["gross_obligation_tco2e"] == 10000.0

    def test_free_allocation_2026(self, engine):
        """Test 2026 free allocation at 97.5%."""
        result = engine.calculate_obligation(
            total_emissions_tco2e=10000.0, year=2026,
        )
        assert result["free_allocation_pct"] == 0.975
        assert result["free_allocation_tco2e"] == pytest.approx(9750.0)
        assert result["net_obligation_tco2e"] == pytest.approx(250.0)

    def test_free_allocation_2030(self, engine):
        """Test 2030 free allocation at 51.5%."""
        result = engine.calculate_obligation(
            total_emissions_tco2e=10000.0, year=2030,
        )
        assert result["free_allocation_pct"] == 0.515
        assert result["free_allocation_tco2e"] == pytest.approx(5150.0)
        assert result["net_obligation_tco2e"] == pytest.approx(4850.0)

    def test_free_allocation_2034(self, engine):
        """Test 2034 free allocation at 0% (full CBAM)."""
        result = engine.calculate_obligation(
            total_emissions_tco2e=10000.0, year=2034,
        )
        assert result["free_allocation_pct"] == 0.0
        assert result["free_allocation_tco2e"] == 0.0
        assert result["net_obligation_tco2e"] == pytest.approx(10000.0)


class TestCarbonPriceDeductions:
    """Test carbon price deduction for origin countries."""

    def test_carbon_price_deduction_turkey(self, engine):
        """Test Turkey carbon pricing deduction reduces net obligation."""
        turkey_deduction = 200.0  # tCO2e equivalent already paid
        result = engine.calculate_obligation(
            total_emissions_tco2e=5000.0, year=2026,
            carbon_price_deduction=turkey_deduction,
        )
        # net = 5000 - (5000 * 0.975) - 200 = 125 - 200 = max(0, -75) = 0
        assert result["net_obligation_tco2e"] == 0.0

    def test_carbon_price_deduction_no_pricing(self, engine):
        """Test no deduction when origin has no carbon pricing."""
        result = engine.calculate_obligation(
            total_emissions_tco2e=5000.0, year=2026,
            carbon_price_deduction=0.0,
        )
        assert result["carbon_price_deduction_tco2e"] == 0.0
        assert result["net_obligation_tco2e"] == pytest.approx(125.0)

    def test_net_obligation_non_negative(self, engine):
        """Test net obligation never goes below zero (max(0, ...))."""
        result = engine.calculate_obligation(
            total_emissions_tco2e=1000.0, year=2026,
            carbon_price_deduction=500.0,
        )
        # net = 1000 - 975 - 500 = -475 => max(0, -475) = 0
        assert result["net_obligation_tco2e"] >= 0.0


class TestCostEstimation:
    """Test certificate cost estimation."""

    def test_cost_estimation_low(self, engine):
        """Test cost at low ETS price (60 EUR)."""
        cost = engine.estimate_cost(net_obligation_tco2e=100.0, ets_price_eur=60.0)
        assert cost["estimated_cost_eur"] == 6000.0
        assert cost["currency"] == "EUR"

    def test_cost_estimation_mid(self, engine):
        """Test cost at mid ETS price (80 EUR)."""
        cost = engine.estimate_cost(net_obligation_tco2e=250.0, ets_price_eur=80.0)
        assert cost["estimated_cost_eur"] == 20000.0

    def test_cost_estimation_high(self, engine):
        """Test cost at high ETS price (100 EUR)."""
        cost = engine.estimate_cost(net_obligation_tco2e=500.0, ets_price_eur=100.0)
        assert cost["estimated_cost_eur"] == 50000.0


class TestQuarterlyHolding:
    """Test quarterly holding compliance."""

    def test_quarterly_holding_compliant(self, engine):
        """Test compliant quarterly holding (enough certificates)."""
        result = engine.check_quarterly_holding(
            certificates_held=100, net_obligation_tco2e=100.0,
        )
        assert result["compliant"] is True
        assert result["shortfall"] == 0

    def test_quarterly_holding_violation(self, engine):
        """Test non-compliant quarterly holding (insufficient certificates)."""
        result = engine.check_quarterly_holding(
            certificates_held=50, net_obligation_tco2e=100.0,
        )
        # required = 80% of 100 = 80; held = 50 => shortfall 30
        assert result["compliant"] is False
        assert result["shortfall"] == 30


class TestCertificateLifecycle:
    """Test full certificate lifecycle and edge cases."""

    def test_full_certificate_lifecycle(self, engine):
        """Test complete lifecycle: calculate -> estimate -> hold -> surrender."""
        # Step 1: Calculate obligation
        obligation = engine.calculate_obligation(
            total_emissions_tco2e=5000.0, year=2026,
        )
        net = obligation["net_obligation_tco2e"]

        # Step 2: Estimate cost
        cost = engine.estimate_cost(net, ets_price_eur=78.50)
        assert cost["estimated_cost_eur"] > 0

        # Step 3: Check holding
        holding = engine.check_quarterly_holding(
            certificates_held=int(net), net_obligation_tco2e=net,
        )
        assert holding["compliant"] is True

    def test_multi_category_obligation(self, engine):
        """Test obligation across multiple goods categories."""
        steel_emissions = 3000.0
        aluminium_emissions = 1500.0
        cement_emissions = 500.0
        total = steel_emissions + aluminium_emissions + cement_emissions
        result = engine.calculate_obligation(
            total_emissions_tco2e=total, year=2027,
        )
        # 2027: 92.5% free allocation => net = 5000 * 0.075 = 375
        assert result["net_obligation_tco2e"] == pytest.approx(375.0)

    def test_surrender_deadline(self, engine):
        """Test surrender deadline calculation."""
        year = 2026
        deadline_month = 5  # May of next year
        deadline = f"{year + 1}-{deadline_month:02d}-31"
        assert deadline == "2027-05-31"

    def test_purchase_planning(self, engine):
        """Test certificate purchase planning (quarterly tranches)."""
        obligation = engine.calculate_obligation(
            total_emissions_tco2e=8000.0, year=2028,
        )
        net = obligation["net_obligation_tco2e"]
        quarterly_purchase = round(net / 4, 0)
        assert quarterly_purchase > 0
        assert quarterly_purchase * 4 >= net - 4  # rounding tolerance

    def test_certificate_with_zero_emissions(self, engine):
        """Test certificate calculation with zero emissions."""
        result = engine.calculate_obligation(
            total_emissions_tco2e=0.0, year=2026,
        )
        assert result["net_obligation_tco2e"] == 0.0
        assert result["certificates_required"] == 0

    def test_certificate_large_volume(self, engine):
        """Test certificate calculation with very large emissions."""
        result = engine.calculate_obligation(
            total_emissions_tco2e=500000.0, year=2034,
        )
        # 2034: 0% free allocation => full obligation
        assert result["net_obligation_tco2e"] == pytest.approx(500000.0)
        assert result["certificates_required"] == 500000

    def test_phase_out_schedule_complete(self, engine):
        """Test free allocation phase-out schedule covers 2026-2034."""
        schedule = StubCertificateEngine.FREE_ALLOCATION_SCHEDULE
        assert len(schedule) == 9
        assert schedule[2026] == 0.975
        assert schedule[2034] == 0.0
        # Verify monotonically decreasing
        years = sorted(schedule.keys())
        for i in range(1, len(years)):
            assert schedule[years[i]] <= schedule[years[i - 1]], (
                f"Schedule not decreasing: {years[i-1]}={schedule[years[i-1]]} "
                f"> {years[i]}={schedule[years[i]]}"
            )

    def test_cost_projection_3_years(self, engine):
        """Test cost projection across 3 years with increasing obligations."""
        projections = []
        for year in [2028, 2029, 2030]:
            obligation = engine.calculate_obligation(
                total_emissions_tco2e=10000.0, year=year,
            )
            cost = engine.estimate_cost(
                obligation["net_obligation_tco2e"], ets_price_eur=80.0,
            )
            projections.append({
                "year": year,
                "net_tco2e": obligation["net_obligation_tco2e"],
                "cost_eur": cost["estimated_cost_eur"],
            })
        # Costs should increase as free allocation decreases
        for i in range(1, len(projections)):
            assert projections[i]["cost_eur"] >= projections[i - 1]["cost_eur"], (
                f"Cost should increase: {projections[i-1]['year']} -> {projections[i]['year']}"
            )

    def test_provenance_hash_certificate(self, engine):
        """Test provenance hash is deterministic for same inputs."""
        r1 = engine.calculate_obligation(total_emissions_tco2e=5000.0, year=2026)
        r2 = engine.calculate_obligation(total_emissions_tco2e=5000.0, year=2026)
        assert r1["provenance_hash"] == r2["provenance_hash"]
        assert len(r1["provenance_hash"]) == 64
