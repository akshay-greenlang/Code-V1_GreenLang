# -*- coding: utf-8 -*-
"""
Test suite for PACK-027 Enterprise Net Zero Pack - Carbon Pricing Engine.

Tests internal carbon price management ($50-$200/tCO2e), shadow pricing,
carbon-adjusted NPV, CBAM exposure, and business unit allocation.

Author:  GreenLang Test Engineering
Pack:    PACK-027 Enterprise Net Zero
Tests:   ~45 tests
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.carbon_pricing_engine import (
    CarbonPricingEngine,
    CarbonPricingApproach,
    CarbonPricingInput,
    CarbonPricingResult,
    InvestmentAppraisal,
    CBAMExposure,
    BUCarbonAllocation,
    BusinessUnitEmissions,
    InvestmentProposal,
    CBAMImport,
)

from .conftest import (
    assert_decimal_close, assert_decimal_positive, assert_provenance_hash,
)


def _make_input(**overrides):
    defaults = dict(internal_carbon_price=Decimal("85"))
    defaults.update(overrides)
    return CarbonPricingInput(**defaults)


class TestCarbonPricingInstantiation:
    def test_engine_instantiates(self):
        engine = CarbonPricingEngine()
        assert engine is not None

    def test_engine_has_calculate_method(self):
        engine = CarbonPricingEngine()
        assert hasattr(engine, "calculate")


class TestShadowPricing:
    def test_shadow_price_basic(self):
        engine = CarbonPricingEngine()
        result = engine.calculate(_make_input(
            total_scope1_tco2e=Decimal("250000"),
        ))
        assert result.total_carbon_charge_usd >= Decimal("0")

    def test_pricing_approach_default(self):
        engine = CarbonPricingEngine()
        result = engine.calculate(_make_input())
        assert result is not None

    def test_carbon_pnl_present(self):
        engine = CarbonPricingEngine()
        result = engine.calculate(_make_input(
            total_scope1_tco2e=Decimal("100000"),
            total_scope2_tco2e=Decimal("50000"),
        ))
        assert hasattr(result, "carbon_pnl")

    def test_carbon_liability_present(self):
        engine = CarbonPricingEngine()
        result = engine.calculate(_make_input(
            total_scope1_tco2e=Decimal("100000"),
        ))
        assert hasattr(result, "carbon_liability")


class TestInvestmentAppraisal:
    def test_investment_appraisals(self):
        engine = CarbonPricingEngine()
        result = engine.calculate(_make_input(
            investment_proposals=[
                InvestmentProposal(
                    project_name="Solar Farm Phase 1",
                    capex_usd=Decimal("15000000"),
                    annual_savings_usd=Decimal("1000000"),
                    annual_emission_reduction_tco2e=Decimal("8500"),
                    lifetime_years=25,
                ),
            ],
        ))
        assert len(result.investment_appraisals) == 1
        appraisal = result.investment_appraisals[0]
        assert hasattr(appraisal, "standard_npv_usd")
        assert hasattr(appraisal, "carbon_adjusted_npv_usd")

    def test_carbon_adjusted_npv_higher(self):
        engine = CarbonPricingEngine()
        result = engine.calculate(_make_input(
            investment_proposals=[
                InvestmentProposal(
                    project_name="Heat Pump Retrofit",
                    capex_usd=Decimal("3500000"),
                    annual_savings_usd=Decimal("500000"),
                    annual_emission_reduction_tco2e=Decimal("2800"),
                    lifetime_years=20,
                ),
            ],
        ))
        appraisal = result.investment_appraisals[0]
        assert appraisal.carbon_adjusted_npv_usd >= appraisal.standard_npv_usd

    def test_multiple_proposals(self):
        engine = CarbonPricingEngine()
        result = engine.calculate(_make_input(
            investment_proposals=[
                InvestmentProposal(
                    project_name=f"Project {i}",
                    capex_usd=Decimal("5000000"),
                    annual_savings_usd=Decimal("500000"),
                    annual_emission_reduction_tco2e=Decimal("3000"),
                    lifetime_years=15,
                )
                for i in range(3)
            ],
        ))
        assert len(result.investment_appraisals) == 3


class TestCBAMExposure:
    def test_cbam_calculation(self):
        engine = CarbonPricingEngine()
        result = engine.calculate(_make_input(
            cbam_imports=[
                CBAMImport(
                    product_category="steel",
                    import_origin="CN",
                    annual_tonnes=Decimal("5000"),
                    embedded_tco2e_per_tonne=Decimal("2.1"),
                ),
                CBAMImport(
                    product_category="aluminium",
                    import_origin="IN",
                    annual_tonnes=Decimal("2000"),
                    embedded_tco2e_per_tonne=Decimal("12.5"),
                ),
            ],
        ))
        assert result.total_cbam_cost_usd >= Decimal("0")

    def test_cbam_exposures_by_product(self):
        engine = CarbonPricingEngine()
        result = engine.calculate(_make_input(
            cbam_imports=[
                CBAMImport(
                    product_category="steel",
                    import_origin="CN",
                    annual_tonnes=Decimal("5000"),
                    embedded_tco2e_per_tonne=Decimal("2.1"),
                ),
                CBAMImport(
                    product_category="cement",
                    import_origin="TR",
                    annual_tonnes=Decimal("10000"),
                    embedded_tco2e_per_tonne=Decimal("0.65"),
                ),
            ],
        ))
        assert len(result.cbam_exposures) == 2


class TestBUAllocation:
    def test_bu_carbon_allocation(self):
        engine = CarbonPricingEngine()
        result = engine.calculate(_make_input(
            business_units=[
                BusinessUnitEmissions(bu_name="Manufacturing EU",
                                      total_emissions_tco2e=Decimal("45000"),
                                      revenue_usd=Decimal("850000000")),
                BusinessUnitEmissions(bu_name="Manufacturing US",
                                      total_emissions_tco2e=Decimal("38000"),
                                      revenue_usd=Decimal("720000000")),
                BusinessUnitEmissions(bu_name="Logistics",
                                      total_emissions_tco2e=Decimal("12000"),
                                      revenue_usd=Decimal("350000000")),
            ],
        ))
        assert len(result.bu_allocations) == 3

    def test_bu_carbon_charge_positive(self):
        engine = CarbonPricingEngine()
        result = engine.calculate(_make_input(
            business_units=[
                BusinessUnitEmissions(bu_name="BU1",
                                      total_emissions_tco2e=Decimal("50000"),
                                      revenue_usd=Decimal("500000000")),
            ],
        ))
        for alloc in result.bu_allocations:
            assert alloc.carbon_charge_usd >= Decimal("0")


class TestCarbonPricingProvenance:
    def test_provenance_hash(self):
        engine = CarbonPricingEngine()
        result = engine.calculate(_make_input(
            total_scope1_tco2e=Decimal("250000"),
        ))
        assert_provenance_hash(result)

    def test_deterministic(self):
        engine = CarbonPricingEngine()
        inp = _make_input(total_scope1_tco2e=Decimal("250000"))
        r1 = engine.calculate(inp)
        r2 = engine.calculate(inp)
        assert r1.total_carbon_charge_usd == r2.total_carbon_charge_usd

    def test_esrs_e18_output(self):
        """Must produce ESRS E1-8 internal carbon pricing disclosure."""
        engine = CarbonPricingEngine()
        result = engine.calculate(_make_input(
            total_scope1_tco2e=Decimal("250000"),
        ))
        assert hasattr(result, "esrs_e1_8_disclosure")
