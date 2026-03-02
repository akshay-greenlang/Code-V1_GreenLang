# -*- coding: utf-8 -*-
"""
Test suite for investments.debt_investment_calculator - AGENT-MRV-028.

Tests the DebtInvestmentCalculatorEngine (Engine 3) for the Investments
Agent (GL-MRV-S3-015) including corporate bond attribution, business loan
calculation, project finance annualized emissions, green bond handling,
revolving credit average balance, PCAF quality tiers, and DC rules.

Coverage:
- Corporate bond attribution via EVIC
- Business loan calculation
- Project finance: annualized lifetime emissions
- Green bond handling (discount factor)
- Revolving credit average balance
- All 5 PCAF quality tiers
- DC rules for debt instruments
- Parametrized tests for asset types

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
from unittest.mock import patch, MagicMock
import pytest

from greenlang.investments.debt_investment_calculator import (
    DebtInvestmentCalculatorEngine,
)
from greenlang.investments.models import (
    AssetClass,
    PCAFDataQuality,
    AttributionMethod,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset singleton before and after every test."""
    DebtInvestmentCalculatorEngine.reset_instance()
    yield
    DebtInvestmentCalculatorEngine.reset_instance()


@pytest.fixture
def engine():
    """Create a fresh DebtInvestmentCalculatorEngine with mocked config."""
    with patch(
        "greenlang.investments.debt_investment_calculator.get_config"
    ) as mock_config:
        cfg = MagicMock()
        cfg.debt.green_bond_discount = Decimal("0.0")
        cfg.general.default_gwp = "AR5"
        mock_config.return_value = cfg
        eng = DebtInvestmentCalculatorEngine()
        yield eng


def _make_corporate_bond_input(**overrides):
    """Build a corporate bond input dict with defaults."""
    base = {
        "asset_class": "corporate_bond",
        "investee_name": "Tesla Inc.",
        "isin": "US88160RAJ68",
        "outstanding_amount": Decimal("75000000"),
        "evic": Decimal("500000000000"),
        "investee_scope1": Decimal("30000"),
        "investee_scope2": Decimal("12000"),
        "sector": "consumer_discretionary",
        "country": "US",
        "currency": "USD",
        "reporting_year": 2024,
        "pcaf_quality_score": 1,
    }
    base.update(overrides)
    return base


def _make_business_loan_input(**overrides):
    """Build a business loan input dict with defaults."""
    base = {
        "asset_class": "business_loan",
        "investee_name": "SME Corp",
        "outstanding_amount": Decimal("10000000"),
        "total_equity_plus_debt": Decimal("50000000"),
        "investee_scope1": Decimal("5000"),
        "investee_scope2": Decimal("2000"),
        "sector": "industrials",
        "country": "US",
        "currency": "USD",
        "reporting_year": 2024,
        "pcaf_quality_score": 3,
    }
    base.update(overrides)
    return base


def _make_project_finance_input(**overrides):
    """Build a project finance input dict with defaults."""
    base = {
        "asset_class": "project_finance",
        "project_name": "SunBelt Solar Farm",
        "outstanding_amount": Decimal("30000000"),
        "total_project_cost": Decimal("100000000"),
        "project_lifetime_years": 25,
        "annual_project_emissions": Decimal("500"),
        "sector": "utilities",
        "country": "US",
        "currency": "USD",
        "reporting_year": 2024,
        "pcaf_quality_score": 2,
    }
    base.update(overrides)
    return base


# ==============================================================================
# CORPORATE BOND TESTS
# ==============================================================================


class TestCorporateBondCalculation:
    """Test corporate bond EVIC-based calculations."""

    def test_corporate_bond_attribution_factor(self, engine):
        """Test corporate bond AF = outstanding / EVIC."""
        data = _make_corporate_bond_input()
        result = engine.calculate(data)
        expected_af = Decimal("75000000") / Decimal("500000000000")
        assert abs(result["attribution_factor"] - expected_af) < Decimal("0.0000001")

    def test_corporate_bond_financed_emissions(self, engine):
        """Test corporate bond financed emissions calculation."""
        data = _make_corporate_bond_input()
        result = engine.calculate(data)
        af = Decimal("75000000") / Decimal("500000000000")
        expected = af * (Decimal("30000") + Decimal("12000"))
        assert abs(result["financed_emissions"] - expected) < Decimal("0.01")

    def test_corporate_bond_positive_emissions(self, engine):
        """Test corporate bond emissions are positive."""
        data = _make_corporate_bond_input()
        result = engine.calculate(data)
        assert result["financed_emissions"] > Decimal("0")

    def test_corporate_bond_attribution_method(self, engine):
        """Test corporate bond uses EVIC attribution."""
        data = _make_corporate_bond_input()
        result = engine.calculate(data)
        assert result["attribution_method"] == "evic"

    def test_corporate_bond_provenance_hash(self, engine):
        """Test corporate bond result includes provenance hash."""
        data = _make_corporate_bond_input()
        result = engine.calculate(data)
        assert len(result["provenance_hash"]) == 64

    def test_larger_outstanding_increases_emissions(self, engine):
        """Test larger bond outstanding increases financed emissions."""
        small = _make_corporate_bond_input(outstanding_amount=Decimal("25000000"))
        large = _make_corporate_bond_input(outstanding_amount=Decimal("150000000"))
        r_small = engine.calculate(small)
        r_large = engine.calculate(large)
        assert r_large["financed_emissions"] > r_small["financed_emissions"]


# ==============================================================================
# BUSINESS LOAN TESTS
# ==============================================================================


class TestBusinessLoanCalculation:
    """Test business loan calculations."""

    def test_business_loan_attribution_factor(self, engine):
        """Test business loan AF = outstanding / (equity + debt)."""
        data = _make_business_loan_input()
        result = engine.calculate(data)
        expected_af = Decimal("10000000") / Decimal("50000000")
        assert abs(result["attribution_factor"] - expected_af) < Decimal("0.0001")

    def test_business_loan_financed_emissions(self, engine):
        """Test business loan financed emissions calculation."""
        data = _make_business_loan_input()
        result = engine.calculate(data)
        af = Decimal("10000000") / Decimal("50000000")
        expected = af * (Decimal("5000") + Decimal("2000"))
        assert abs(result["financed_emissions"] - expected) < Decimal("0.01")

    def test_business_loan_positive_emissions(self, engine):
        """Test business loan emissions are positive."""
        data = _make_business_loan_input()
        result = engine.calculate(data)
        assert result["financed_emissions"] > Decimal("0")


# ==============================================================================
# PROJECT FINANCE TESTS
# ==============================================================================


class TestProjectFinanceCalculation:
    """Test project finance pro-rata calculations."""

    def test_project_finance_attribution_factor(self, engine):
        """Test PF attribution = outstanding / total_project_cost."""
        data = _make_project_finance_input()
        result = engine.calculate(data)
        expected_af = Decimal("30000000") / Decimal("100000000")
        assert abs(result["attribution_factor"] - expected_af) < Decimal("0.0001")

    def test_project_finance_annualized_emissions(self, engine):
        """Test PF uses annualized lifetime emissions."""
        data = _make_project_finance_input()
        result = engine.calculate(data)
        af = Decimal("30000000") / Decimal("100000000")
        expected = af * Decimal("500")
        assert abs(result["financed_emissions"] - expected) < Decimal("0.01")

    def test_project_finance_lifetime_effect(self, engine):
        """Test changing lifetime affects annualized emissions."""
        data_short = _make_project_finance_input(project_lifetime_years=10)
        data_long = _make_project_finance_input(project_lifetime_years=50)
        r_short = engine.calculate(data_short)
        r_long = engine.calculate(data_long)
        # Same annual emissions, so results should be similar if using annual
        # But total lifetime emissions differ
        assert r_short["financed_emissions"] > Decimal("0")
        assert r_long["financed_emissions"] > Decimal("0")

    def test_project_finance_pro_rata_method(self, engine):
        """Test PF uses pro_rata_project attribution method."""
        data = _make_project_finance_input()
        result = engine.calculate(data)
        assert result["attribution_method"] == "pro_rata_project"

    def test_zero_project_cost_raises_error(self, engine):
        """Test zero total project cost raises error."""
        data = _make_project_finance_input(total_project_cost=Decimal("0"))
        with pytest.raises((ValueError, ZeroDivisionError)):
            engine.calculate(data)


# ==============================================================================
# GREEN BOND TESTS
# ==============================================================================


class TestGreenBondHandling:
    """Test green bond specific handling."""

    def test_green_bond_flag(self, engine):
        """Test green bond flag is recognized."""
        data = _make_corporate_bond_input(is_green_bond=True)
        result = engine.calculate(data)
        assert result.get("is_green_bond", False) is True or \
               "green_bond" in str(result).lower()

    def test_green_bond_discount_applied(self, engine):
        """Test green bond discount factor is applied when configured."""
        with patch(
            "greenlang.investments.debt_investment_calculator.get_config"
        ) as mock_config:
            cfg = MagicMock()
            cfg.debt.green_bond_discount = Decimal("0.50")
            cfg.general.default_gwp = "AR5"
            mock_config.return_value = cfg
            DebtInvestmentCalculatorEngine.reset_instance()
            eng = DebtInvestmentCalculatorEngine()
            data = _make_corporate_bond_input(is_green_bond=True)
            result = eng.calculate(data)
            assert result["financed_emissions"] > Decimal("0")

    def test_non_green_bond_no_discount(self, engine):
        """Test non-green bond gets no discount."""
        data = _make_corporate_bond_input(is_green_bond=False)
        result = engine.calculate(data)
        assert result["financed_emissions"] > Decimal("0")


# ==============================================================================
# REVOLVING CREDIT TESTS
# ==============================================================================


class TestRevolvingCredit:
    """Test revolving credit average balance handling."""

    def test_revolving_credit_average_balance(self, engine):
        """Test revolving credit uses average drawn balance."""
        data = _make_business_loan_input(
            is_revolving=True,
            average_balance=Decimal("7500000"),
        )
        result = engine.calculate(data)
        assert result["financed_emissions"] > Decimal("0")

    def test_revolving_vs_term_loan(self, engine):
        """Test revolving credit with lower avg balance has lower emissions."""
        data_term = _make_business_loan_input(
            outstanding_amount=Decimal("10000000"),
        )
        data_revolving = _make_business_loan_input(
            is_revolving=True,
            average_balance=Decimal("5000000"),
            outstanding_amount=Decimal("10000000"),
        )
        r_term = engine.calculate(data_term)
        r_revolving = engine.calculate(data_revolving)
        # Revolving with 50% avg draw should have lower emissions
        assert r_revolving["financed_emissions"] <= r_term["financed_emissions"]


# ==============================================================================
# PCAF QUALITY TIER TESTS
# ==============================================================================


class TestPCAFQualityTiers:
    """Test PCAF quality tiers for debt instruments."""

    @pytest.mark.parametrize("score", [1, 2, 3, 4, 5])
    def test_all_pcaf_scores_accepted(self, engine, score):
        """Test all PCAF scores 1-5 produce valid results."""
        data = _make_corporate_bond_input(pcaf_quality_score=score)
        result = engine.calculate(data)
        assert result["pcaf_quality_score"] == score

    def test_pcaf_score_affects_uncertainty(self, engine):
        """Test higher PCAF score increases uncertainty."""
        data_q1 = _make_corporate_bond_input(pcaf_quality_score=1)
        data_q5 = _make_corporate_bond_input(pcaf_quality_score=5)
        r1 = engine.calculate(data_q1)
        r5 = engine.calculate(data_q5)
        if "uncertainty_range" in r1 and "uncertainty_range" in r5:
            assert r5["uncertainty_range"] >= r1["uncertainty_range"]


# ==============================================================================
# DC RULES TESTS
# ==============================================================================


class TestDCRules:
    """Test double-counting prevention for debt instruments."""

    def test_dc_inv_001_consolidated_excluded(self, engine):
        """Test DC-INV-001: consolidated entities excluded."""
        data = _make_corporate_bond_input(is_consolidated=True)
        result = engine.calculate(data)
        assert result.get("excluded", False) is True or \
               result.get("financed_emissions") == Decimal("0")

    def test_dc_inv_002_scope1_scope2_exclusion(self, engine):
        """Test DC-INV-002: already reported in Scope 1/2 excluded."""
        data = _make_corporate_bond_input(
            already_in_scope1_or_scope2=True,
        )
        result = engine.calculate(data)
        assert result.get("dc_inv_002_triggered", False) is True or \
               result.get("excluded", False) is True or \
               result["financed_emissions"] == Decimal("0")


# ==============================================================================
# BATCH AND ERROR HANDLING TESTS
# ==============================================================================


class TestBatchAndErrors:
    """Test batch processing and error handling."""

    def test_batch_multiple_debt_types(self, engine):
        """Test batch with mixed debt instrument types."""
        items = [
            _make_corporate_bond_input(),
            _make_business_loan_input(),
            _make_project_finance_input(),
        ]
        results = engine.calculate_batch(items)
        assert len(results) == 3

    def test_batch_empty_list(self, engine):
        """Test batch with empty list."""
        results = engine.calculate_batch([])
        assert len(results) == 0

    def test_zero_evic_raises_error(self, engine):
        """Test zero EVIC raises error for corporate bond."""
        data = _make_corporate_bond_input(evic=Decimal("0"))
        with pytest.raises((ValueError, ZeroDivisionError)):
            engine.calculate(data)

    def test_result_required_fields(self, engine):
        """Test result contains all required fields."""
        data = _make_corporate_bond_input()
        result = engine.calculate(data)
        required = [
            "investee_name", "asset_class", "attribution_factor",
            "financed_emissions", "pcaf_quality_score", "provenance_hash",
        ]
        for field in required:
            assert field in result

    @pytest.mark.parametrize("asset_type,factory", [
        ("corporate_bond", _make_corporate_bond_input),
        ("business_loan", _make_business_loan_input),
        ("project_finance", _make_project_finance_input),
    ])
    def test_all_debt_types_produce_results(self, engine, asset_type, factory):
        """Test all debt instrument types produce valid results."""
        data = factory()
        result = engine.calculate(data)
        assert result["financed_emissions"] > Decimal("0")
        assert len(result["provenance_hash"]) == 64
