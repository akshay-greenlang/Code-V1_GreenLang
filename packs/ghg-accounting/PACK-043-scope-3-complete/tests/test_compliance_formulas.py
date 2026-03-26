# -*- coding: utf-8 -*-
"""
Unit tests for Compliance Formulas -- PACK-043
=================================================

Reference calculation validation against SBTi, PCAF, GHG Protocol
Scope 3 Standard, and Paris Agreement pathway formulas.

Coverage target: 85%+
Total tests: ~40
"""

from decimal import Decimal

import pytest

from tests.conftest import SBTI_15C_ANNUAL_RATE, SBTI_WB2C_ANNUAL_RATE


# =============================================================================
# SBTi 1.5C Pathway: 4.2%/yr Reduction
# =============================================================================


class TestSBTi15CPathway:
    """Test SBTi 1.5C pathway formula: compound 4.2% annual reduction."""

    def test_single_year_reduction(self):
        """Year 1: base * (1 - 0.042) = base * 0.958."""
        base = Decimal("300000")
        after_1yr = base * (Decimal("1") - SBTI_15C_ANNUAL_RATE / Decimal("100"))
        assert after_1yr == Decimal("287400")

    def test_5_year_compound_reduction(self):
        """5 years: base * (1 - 0.042)^5."""
        base = Decimal("300000")
        target = base * (Decimal("1") - Decimal("0.042")) ** 5
        assert Decimal("237000") < target < Decimal("243000")

    def test_11_year_compound_2019_to_2030(self):
        """11 years (2019->2030): ~37% compound reduction."""
        base = Decimal("300000")
        target = base * (Decimal("1") - Decimal("0.042")) ** 11
        reduction_pct = (base - target) / base * Decimal("100")
        assert Decimal("36") < reduction_pct < Decimal("40")


# =============================================================================
# SBTi WB2C Pathway: 2.5%/yr Reduction
# =============================================================================


class TestSBTiWB2CPathway:
    """Test SBTi well-below 2C pathway formula."""

    def test_single_year_wb2c(self):
        base = Decimal("300000")
        after_1yr = base * (Decimal("1") - SBTI_WB2C_ANNUAL_RATE / Decimal("100"))
        assert after_1yr == Decimal("292500")

    def test_11_year_wb2c(self):
        base = Decimal("300000")
        target = base * (Decimal("1") - Decimal("0.025")) ** 11
        reduction_pct = (base - target) / base * Decimal("100")
        assert Decimal("24") < reduction_pct < Decimal("27")

    def test_wb2c_less_ambitious_than_15c(self):
        base = Decimal("300000")
        target_15c = base * (Decimal("1") - Decimal("0.042")) ** 11
        target_wb2c = base * (Decimal("1") - Decimal("0.025")) ** 11
        assert target_wb2c > target_15c


# =============================================================================
# PCAF Attribution: Invested / EVIC x Investee Emissions
# =============================================================================


class TestPCAFAttribution:
    """Test PCAF attribution factor calculation."""

    def test_basic_attribution(self):
        """50M invested / 500M EVIC = 10% attribution."""
        invested = Decimal("50000000")
        evic = Decimal("500000000")
        attribution = invested / evic
        assert attribution == Decimal("0.1")

    def test_financed_scope12(self):
        """10% of 25,000 tCO2e = 2,500 tCO2e financed."""
        attribution = Decimal("0.1")
        investee_s12 = Decimal("25000")
        financed = attribution * investee_s12
        assert financed == Decimal("2500")

    def test_financed_scope3(self):
        """10% of 120,000 tCO2e = 12,000 tCO2e financed."""
        attribution = Decimal("0.1")
        investee_s3 = Decimal("120000")
        financed = attribution * investee_s3
        assert financed == Decimal("12000")

    def test_portfolio_financed_sum(self, sample_pcaf_portfolio):
        """Total financed emissions across portfolio."""
        total = Decimal("0")
        for asset_class in sample_pcaf_portfolio["asset_classes"].values():
            for inv in asset_class["investments"]:
                attr = inv["invested_amount_usd"] / inv["evic_usd"]
                total += attr * (inv["investee_scope12_tco2e"] + inv["investee_scope3_tco2e"])
        assert total > Decimal("0")


# =============================================================================
# MACC: Cost/tCO2e Ranking Validation
# =============================================================================


class TestMACCRanking:
    """Test MACC cost-effectiveness ranking."""

    def test_macc_ordering(self, sample_macc_interventions):
        """Interventions should be rankable by cost/tCO2e."""
        ranked = sorted(sample_macc_interventions, key=lambda x: x["cost_per_tco2e"])
        for i in range(1, len(ranked)):
            assert ranked[i]["cost_per_tco2e"] >= ranked[i - 1]["cost_per_tco2e"]

    def test_cost_per_tco2e_formula(self, sample_macc_interventions):
        """cost_per_tco2e = annual_cost / abatement."""
        for intv in sample_macc_interventions:
            expected = intv["annual_cost_usd"] / intv["abatement_tco2e"]
            assert expected == pytest.approx(intv["cost_per_tco2e"], abs=Decimal("0.01"))

    def test_negative_cost_means_savings(self, sample_macc_interventions):
        negatives = [i for i in sample_macc_interventions if i["cost_per_tco2e"] < 0]
        for intv in negatives:
            assert intv["annual_cost_usd"] < Decimal("0")


# =============================================================================
# NPV Calculation: Discount Rate x Cash Flows
# =============================================================================


class TestNPVFormula:
    """Test NPV calculation formula."""

    def test_npv_basic(self):
        """NPV = sum(CF_t / (1+r)^t) - initial_investment."""
        cfs = [Decimal("100000")] * 10
        r = Decimal("0.08")
        investment = Decimal("500000")
        npv = -investment + sum(cf / (Decimal("1") + r) ** t for t, cf in enumerate(cfs, 1))
        assert npv > Decimal("0")

    def test_npv_zero_discount(self):
        """At 0% discount rate, NPV = sum(CFs) - investment."""
        cfs = [Decimal("100000")] * 10
        investment = Decimal("500000")
        npv = -investment + sum(cfs)
        assert npv == Decimal("500000")

    def test_npv_high_discount_reduces(self):
        """Higher discount rate should reduce NPV."""
        cfs = [Decimal("100000")] * 10
        investment = Decimal("500000")
        npv_low = -investment + sum(cf / Decimal("1.05") ** t for t, cf in enumerate(cfs, 1))
        npv_high = -investment + sum(cf / Decimal("1.15") ** t for t, cf in enumerate(cfs, 1))
        assert npv_high < npv_low


# =============================================================================
# Pro-Rata: Days in Period / 365 x Annual Emissions
# =============================================================================


class TestProRataFormula:
    """Test pro-rata calculation formula."""

    def test_full_year(self):
        annual = Decimal("22000")
        days = 365
        pro_rata = annual * Decimal(str(days)) / Decimal("365")
        assert pro_rata == annual

    def test_half_year(self):
        annual = Decimal("22000")
        days = 183  # approx half year
        pro_rata = annual * Decimal(str(days)) / Decimal("365")
        assert Decimal("10000") < pro_rata < Decimal("12000")

    def test_single_day(self):
        annual = Decimal("365000")
        days = 1
        pro_rata = annual * Decimal(str(days)) / Decimal("365")
        assert pro_rata == Decimal("1000")

    def test_acquisition_jul1(self):
        """Acquisition on July 1 = 184 days remaining."""
        annual = Decimal("22000")
        days = 184
        pro_rata = annual * Decimal(str(days)) / Decimal("365")
        assert Decimal("11000") < pro_rata < Decimal("11200")


# =============================================================================
# Equity Share: Ownership % x Entity Emissions
# =============================================================================


class TestEquityShareFormula:
    """Test equity share consolidation formula."""

    def test_100pct_ownership(self):
        emissions = Decimal("45000")
        ownership = Decimal("100")
        reported = emissions * ownership / Decimal("100")
        assert reported == emissions

    def test_50pct_ownership(self):
        emissions = Decimal("22000")
        ownership = Decimal("50")
        reported = emissions * ownership / Decimal("100")
        assert reported == Decimal("11000")

    def test_40pct_ownership(self):
        emissions = Decimal("35000")
        ownership = Decimal("40")
        reported = emissions * ownership / Decimal("100")
        assert reported == Decimal("14000")

    def test_25pct_ownership(self):
        emissions = Decimal("40000")
        ownership = Decimal("25")
        reported = emissions * ownership / Decimal("100")
        assert reported == Decimal("10000")

    def test_sum_of_equity_shares(self, sample_entity_hierarchy):
        """Sum of equity-share-adjusted emissions for all entities."""
        total = Decimal("0")
        total += sample_entity_hierarchy["parent"]["scope3_tco2e"]
        for sub in sample_entity_hierarchy["subsidiaries"]:
            total += sub["scope3_tco2e"] * sub["equity_pct"] / Decimal("100")
        for jv in sample_entity_hierarchy["joint_ventures"]:
            total += jv["scope3_tco2e"] * jv["equity_pct"] / Decimal("100")
        assert total > Decimal("0")
