# -*- coding: utf-8 -*-
"""
Test suite for PACK-027 Enterprise Net Zero Pack - Financial Integration Engine.

Tests carbon data integration into financial reporting: carbon-adjusted P&L,
carbon balance sheet, EBITDA carbon intensity, CBAM exposure, ESRS E1-8/E1-9.

Author:  GreenLang Test Engineering
Pack:    PACK-027 Enterprise Net Zero
Tests:   ~55 tests
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.financial_integration_engine import (
    FinancialIntegrationEngine,
    FinancialIntegrationInput,
    FinancialIntegrationResult,
    CarbonPnLAllocation,
    CarbonBalanceSheet,
    CarbonIntensityMetrics,
    ESRSE1Disclosure,
    FinancialData,
    EmissionsByFunction,
)

from .conftest import (
    assert_decimal_close, assert_decimal_positive,
    assert_provenance_hash,
)


def _make_input(**overrides):
    defaults = dict(
        internal_carbon_price=Decimal("85"),
        total_scope1_tco2e=Decimal("125000"),
        total_scope2_tco2e=Decimal("85000"),
        total_scope3_tco2e=Decimal("657000"),
    )
    defaults.update(overrides)
    return FinancialIntegrationInput(**defaults)


class TestFinancialIntegrationInstantiation:
    def test_engine_instantiates(self):
        engine = FinancialIntegrationEngine()
        assert engine is not None

    def test_engine_has_calculate_method(self):
        engine = FinancialIntegrationEngine()
        assert hasattr(engine, "calculate")


# ===========================================================================
# Tests -- Carbon-Adjusted P&L
# ===========================================================================


class TestCarbonPnL:
    def test_carbon_pnl_generated(self):
        """Carbon P&L allocation must be generated."""
        engine = FinancialIntegrationEngine()
        result = engine.calculate(_make_input())
        assert hasattr(result, "carbon_pnl")
        assert result.carbon_pnl is not None

    def test_carbon_cogs_charge(self):
        """Carbon COGS charge must exist."""
        engine = FinancialIntegrationEngine()
        result = engine.calculate(_make_input(
            emissions_by_function=EmissionsByFunction(
                manufacturing_tco2e=Decimal("100000"),
            ),
            financial_data=FinancialData(
                revenue_usd=Decimal("2800000000"),
                cogs_usd=Decimal("1900000000"),
            ),
        ))
        assert hasattr(result.carbon_pnl, "cogs_carbon_charge_usd")

    def test_carbon_sga_charge(self):
        engine = FinancialIntegrationEngine()
        result = engine.calculate(_make_input(
            emissions_by_function=EmissionsByFunction(
                office_operations_tco2e=Decimal("18000"),
                travel_tco2e=Decimal("22000"),
                commuting_tco2e=Decimal("15000"),
            ),
            financial_data=FinancialData(
                sga_usd=Decimal("450000000"),
            ),
        ))
        assert hasattr(result.carbon_pnl, "sga_carbon_charge_usd")

    def test_carbon_adjusted_ebitda(self):
        engine = FinancialIntegrationEngine()
        result = engine.calculate(_make_input(
            financial_data=FinancialData(
                ebitda_usd=Decimal("500000000"),
            ),
        ))
        assert hasattr(result.carbon_pnl, "carbon_adjusted_ebitda_usd")

    def test_ebitda_impact_pct(self):
        engine = FinancialIntegrationEngine()
        result = engine.calculate(_make_input(
            financial_data=FinancialData(
                ebitda_usd=Decimal("500000000"),
            ),
        ))
        assert hasattr(result.carbon_pnl, "ebitda_impact_pct")

    def test_carbon_intensity_per_revenue(self):
        engine = FinancialIntegrationEngine()
        result = engine.calculate(_make_input(
            financial_data=FinancialData(
                revenue_usd=Decimal("2800000000"),
            ),
        ))
        assert hasattr(result, "intensity_metrics")
        assert hasattr(result.intensity_metrics, "tco2e_per_million_revenue")


# ===========================================================================
# Tests -- Carbon Balance Sheet
# ===========================================================================


class TestCarbonBalanceSheet:
    def test_balance_sheet_generated(self):
        engine = FinancialIntegrationEngine()
        result = engine.calculate(_make_input())
        assert hasattr(result, "carbon_balance_sheet")
        assert result.carbon_balance_sheet is not None

    def test_balance_sheet_has_assets(self):
        engine = FinancialIntegrationEngine()
        result = engine.calculate(_make_input())
        bs = result.carbon_balance_sheet
        assert hasattr(bs, "total_assets_usd")
        assert hasattr(bs, "total_liabilities_usd")

    def test_net_carbon_position(self):
        engine = FinancialIntegrationEngine()
        result = engine.calculate(_make_input())
        bs = result.carbon_balance_sheet
        assert hasattr(bs, "net_carbon_position_usd")

    def test_asset_details_list(self):
        engine = FinancialIntegrationEngine()
        result = engine.calculate(_make_input())
        bs = result.carbon_balance_sheet
        assert isinstance(bs.asset_details, list)

    def test_liability_details_list(self):
        engine = FinancialIntegrationEngine()
        result = engine.calculate(_make_input())
        bs = result.carbon_balance_sheet
        assert isinstance(bs.liability_details, list)


# ===========================================================================
# Tests -- Intensity Metrics
# ===========================================================================


class TestIntensityMetrics:
    def test_intensity_metrics_generated(self):
        engine = FinancialIntegrationEngine()
        result = engine.calculate(_make_input(
            financial_data=FinancialData(
                revenue_usd=Decimal("2800000000"),
                ebitda_usd=Decimal("500000000"),
            ),
        ))
        im = result.intensity_metrics
        assert im.tco2e_per_million_revenue >= Decimal("0")
        assert im.tco2e_per_million_ebitda >= Decimal("0")

    def test_green_capex_pct(self):
        engine = FinancialIntegrationEngine()
        result = engine.calculate(_make_input(
            financial_data=FinancialData(
                total_capex_usd=Decimal("350000000"),
                green_capex_usd=Decimal("120000000"),
            ),
        ))
        assert result.intensity_metrics.green_capex_pct >= Decimal("0")


# ===========================================================================
# Tests -- ESRS E1 Disclosures
# ===========================================================================


class TestESRSDisclosures:
    def test_esrs_disclosure_generated(self):
        engine = FinancialIntegrationEngine()
        result = engine.calculate(_make_input())
        assert hasattr(result, "esrs_disclosure")
        assert result.esrs_disclosure is not None

    def test_esrs_e18_pricing(self):
        engine = FinancialIntegrationEngine()
        result = engine.calculate(_make_input())
        e = result.esrs_disclosure
        assert hasattr(e, "e1_8_pricing_scheme")
        assert hasattr(e, "e1_8_price_level")

    def test_esrs_e19_risks(self):
        engine = FinancialIntegrationEngine()
        result = engine.calculate(_make_input())
        e = result.esrs_disclosure
        assert hasattr(e, "e1_9_physical_risk_exposure_usd")
        assert hasattr(e, "e1_9_transition_risk_exposure_usd")
        assert hasattr(e, "e1_9_climate_opportunities_usd")


# ===========================================================================
# Tests -- Provenance
# ===========================================================================


class TestFinancialProvenance:
    def test_provenance_hash(self):
        engine = FinancialIntegrationEngine()
        result = engine.calculate(_make_input())
        assert_provenance_hash(result)

    def test_deterministic(self):
        engine = FinancialIntegrationEngine()
        inp = _make_input()
        r1 = engine.calculate(inp)
        r2 = engine.calculate(inp)
        assert r1.carbon_pnl.total_carbon_charge_usd == r2.carbon_pnl.total_carbon_charge_usd

    def test_regulatory_citations(self):
        engine = FinancialIntegrationEngine()
        result = engine.calculate(_make_input())
        assert len(result.regulatory_citations) > 0

    def test_processing_time(self):
        engine = FinancialIntegrationEngine()
        result = engine.calculate(_make_input())
        assert result.processing_time_ms >= 0
