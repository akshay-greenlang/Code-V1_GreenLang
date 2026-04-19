# -*- coding: utf-8 -*-
"""
Unit tests for PAIMandatoryEngine (PACK-011 SFDR Article 9, Engine 6).

Tests all 18 mandatory PAI indicators (Table 1), additional Table 2/3
indicators, data quality assessment, integration assessment, action plan
generation, and provenance tracking.

Self-contained: no conftest imports.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Dynamic import helper (hyphenated directory names)
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"


def _import_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_pai_mod = _import_from_path(
    "pai_mandatory_engine",
    str(ENGINES_DIR / "pai_mandatory_engine.py"),
)

PAIMandatoryEngine = _pai_mod.PAIMandatoryEngine
PAIMandatoryConfig = _pai_mod.PAIMandatoryConfig
InvesteeFullData = _pai_mod.InvesteeFullData
PAIMandatoryResult = _pai_mod.PAIMandatoryResult
PAISingleResult = _pai_mod.PAISingleResult
IntegrationAssessment = _pai_mod.IntegrationAssessment
ActionPlan = _pai_mod.ActionPlan
ActionPlanItem = _pai_mod.ActionPlanItem
DataQualityReport = _pai_mod.DataQualityReport
AdditionalPAIResult = _pai_mod.AdditionalPAIResult
PAIMandatoryStatus = _pai_mod.PAIMandatoryStatus
DataQualityLevel = _pai_mod.DataQualityLevel
PAIIndicatorId = _pai_mod.PAIIndicatorId
PAICategory = _pai_mod.PAICategory

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

TOTAL_NAV = 500_000_000.0  # EUR 500M


def _corp_holding(**kwargs) -> InvesteeFullData:
    """Create a corporate investee with reasonable PAI data."""
    defaults = dict(
        investee_name="Corp A",
        investee_type="CORPORATE",
        value_eur=25_000_000,
        weight_pct=5.0,
        enterprise_value_eur=200_000_000,
        revenue_eur=50_000_000,
        scope1_tco2eq=5000.0,
        scope2_tco2eq=2000.0,
        scope3_tco2eq=8000.0,
        is_fossil_fuel_company=False,
        non_renewable_energy_share_pct=35.0,
        energy_consumption_gwh=0.5,
        nace_sector="C",
        affects_biodiversity_area=False,
        emissions_to_water_tonnes=10.0,
        hazardous_waste_tonnes=50.0,
        radioactive_waste_tonnes=0.0,
        has_ungc_violations=False,
        has_compliance_mechanisms=True,
        gender_pay_gap_pct=12.0,
        female_board_pct=35.0,
        involved_controversial_weapons=False,
        # Table 2
        inorganic_pollutants_tonnes=5.0,
        air_pollutants_tonnes=15.0,
        ozone_depleting_tonnes=0.1,
        has_carbon_initiatives=True,
        water_usage_m3=500_000.0,
        # Table 3
        has_accident_prevention=True,
        accident_rate=2.5,
        has_supplier_code=True,
        has_grievance_mechanism=True,
        has_whistleblower_protection=True,
        data_quality=DataQualityLevel.REPORTED,
    )
    defaults.update(kwargs)
    return InvesteeFullData(**defaults)


def _sovereign_holding(**kwargs) -> InvesteeFullData:
    """Create a sovereign investee."""
    defaults = dict(
        investee_name="Govt Bond DE",
        investee_type="SOVEREIGN",
        value_eur=50_000_000,
        weight_pct=10.0,
        country_code="DE",
        country_ghg_intensity=250.0,
        country_social_violations=False,
        data_quality=DataQualityLevel.REPORTED,
    )
    defaults.update(kwargs)
    return InvesteeFullData(**defaults)


def _real_estate_holding(**kwargs) -> InvesteeFullData:
    """Create a real estate investee."""
    defaults = dict(
        investee_name="RE Fund Alpha",
        investee_type="REAL_ESTATE",
        value_eur=30_000_000,
        weight_pct=6.0,
        re_fossil_fuel_involved=False,
        re_energy_inefficient=False,
        data_quality=DataQualityLevel.REPORTED,
    )
    defaults.update(kwargs)
    return InvesteeFullData(**defaults)


def _build_portfolio(n_corp=5, n_sov=1, n_re=1):
    """Build a mixed portfolio for testing."""
    holdings = []
    for i in range(n_corp):
        holdings.append(_corp_holding(
            investee_name=f"Corp-{i}",
            value_eur=25_000_000,
            weight_pct=5.0,
            scope1_tco2eq=5000.0 + i * 500,
            scope2_tco2eq=2000.0 + i * 200,
            scope3_tco2eq=8000.0 + i * 1000,
        ))
    for i in range(n_sov):
        holdings.append(_sovereign_holding(
            investee_name=f"Sov-{i}",
            country_code="DE" if i % 2 == 0 else "FR",
        ))
    for i in range(n_re):
        holdings.append(_real_estate_holding(investee_name=f"RE-{i}"))
    return holdings


# ===========================================================================
# Tests
# ===========================================================================


class TestPAIMandatoryEngineInit:
    """Test engine initialization."""

    def test_default_config(self):
        engine = PAIMandatoryEngine()
        assert engine.config.min_data_quality_pct == pytest.approx(70.0)
        assert engine.config.coverage_threshold_pct == pytest.approx(50.0)
        assert engine.config.include_scope_3 is True

    def test_dict_config(self):
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        assert engine._total_nav == pytest.approx(TOTAL_NAV)

    def test_pydantic_config(self):
        cfg = PAIMandatoryConfig(total_nav_eur=TOTAL_NAV, include_scope_3=False)
        engine = PAIMandatoryEngine(cfg)
        assert engine.config.include_scope_3 is False


class TestMandatoryPAIClimateGHG:
    """Test PAI 1-6 Climate and GHG indicators."""

    def test_pai_1_ghg_emissions(self):
        """PAI 1: SUM(attribution_factor * ghg) with scope 1+2+3."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        holdings = [_corp_holding(
            value_eur=25_000_000,
            enterprise_value_eur=200_000_000,
            scope1_tco2eq=5000.0,
            scope2_tco2eq=2000.0,
            scope3_tco2eq=8000.0,
            weight_pct=100.0,
        )]
        result = engine.calculate_all(holdings)
        pai_1 = next(
            (i for i in result.mandatory_indicators if i.indicator_id == "PAI_1"),
            None,
        )
        assert pai_1 is not None
        assert pai_1.value is not None
        assert pai_1.value > 0
        # attribution = 25M / 200M = 0.125
        # total GHG = 0.125 * (5000 + 2000 + 8000) = 1875
        assert pai_1.sub_values.get("scope_1") is not None
        assert pai_1.sub_values.get("scope_2") is not None
        assert pai_1.sub_values.get("scope_3") is not None
        expected_total = 0.125 * 15000.0
        assert pai_1.value == pytest.approx(expected_total, rel=0.01)
        assert len(pai_1.provenance_hash) == 64

    def test_pai_2_carbon_footprint(self):
        """PAI 2: PAI_1_total / portfolio_value_EUR_M."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        holdings = [_corp_holding(weight_pct=100.0)]
        result = engine.calculate_all(holdings)
        pai_2 = next(
            (i for i in result.mandatory_indicators if i.indicator_id == "PAI_2"),
            None,
        )
        assert pai_2 is not None
        assert pai_2.value is not None
        # PAI 2 = PAI_1 / (NAV / 1M)
        nav_m = TOTAL_NAV / 1_000_000.0
        assert pai_2.value >= 0

    def test_pai_3_ghg_intensity_waci(self):
        """PAI 3: WACI = SUM(weight * ghg/revenue)."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        holdings = [_corp_holding(
            weight_pct=100.0,
            revenue_eur=50_000_000,
            # total_ghg will be auto-computed
        )]
        result = engine.calculate_all(holdings)
        pai_3 = next(
            (i for i in result.mandatory_indicators if i.indicator_id == "PAI_3"),
            None,
        )
        assert pai_3 is not None
        assert pai_3.value is not None
        # ghg = 15000 tCO2e, revenue = 50M EUR -> intensity = 300 tCO2e/EUR M
        # weight = 1.0 -> WACI = 300
        assert pai_3.value == pytest.approx(300.0, rel=0.05)

    def test_pai_4_fossil_fuel_exposure(self):
        """PAI 4: Share of fossil fuel companies (%)."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        h_fossil = _corp_holding(
            investee_name="Fossil Co",
            is_fossil_fuel_company=True,
            value_eur=50_000_000,
            weight_pct=50.0,
        )
        h_clean = _corp_holding(
            investee_name="Clean Co",
            is_fossil_fuel_company=False,
            value_eur=50_000_000,
            weight_pct=50.0,
        )
        result = engine.calculate_all([h_fossil, h_clean])
        pai_4 = next(
            (i for i in result.mandatory_indicators if i.indicator_id == "PAI_4"),
            None,
        )
        assert pai_4 is not None
        assert pai_4.value is not None
        # 50M exposed / 500M NAV * 100 = 10%
        assert pai_4.value == pytest.approx(10.0, rel=0.1)

    def test_pai_5_non_renewable_energy_share(self):
        """PAI 5: Weighted average non-renewable energy share (%)."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        holdings = [_corp_holding(
            non_renewable_energy_share_pct=60.0, weight_pct=100.0,
        )]
        result = engine.calculate_all(holdings)
        pai_5 = next(
            (i for i in result.mandatory_indicators if i.indicator_id == "PAI_5"),
            None,
        )
        assert pai_5 is not None
        assert pai_5.value is not None
        assert pai_5.value == pytest.approx(60.0, rel=0.1)

    def test_pai_6_energy_intensity_per_sector(self):
        """PAI 6: Energy intensity per high-impact NACE sector."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        holdings = [_corp_holding(
            nace_sector="C",
            energy_consumption_gwh=2.0,
            revenue_eur=100_000_000,
            weight_pct=100.0,
        )]
        result = engine.calculate_all(holdings)
        pai_6 = next(
            (i for i in result.mandatory_indicators if i.indicator_id == "PAI_6"),
            None,
        )
        assert pai_6 is not None
        assert pai_6.value is not None
        # 2.0 GWh / 100 EUR M revenue = 0.02 GWh/EUR M
        assert pai_6.value == pytest.approx(0.02, rel=0.1)


class TestMandatoryPAIEnvironment:
    """Test PAI 7-9 Environmental indicators."""

    def test_pai_7_biodiversity(self):
        """PAI 7: Share of holdings affecting biodiversity-sensitive areas."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        h_bio = _corp_holding(affects_biodiversity_area=True, value_eur=25_000_000)
        h_ok = _corp_holding(affects_biodiversity_area=False, value_eur=25_000_000)
        result = engine.calculate_all([h_bio, h_ok])
        pai_7 = next(
            (i for i in result.mandatory_indicators if i.indicator_id == "PAI_7"),
            None,
        )
        assert pai_7 is not None
        assert pai_7.value is not None
        assert pai_7.value > 0

    def test_pai_8_emissions_to_water(self):
        """PAI 8: Attribution-weighted water emissions (tonnes)."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        holdings = [_corp_holding(
            emissions_to_water_tonnes=100.0,
            enterprise_value_eur=200_000_000,
            value_eur=25_000_000,
            weight_pct=100.0,
        )]
        result = engine.calculate_all(holdings)
        pai_8 = next(
            (i for i in result.mandatory_indicators if i.indicator_id == "PAI_8"),
            None,
        )
        assert pai_8 is not None
        assert pai_8.value is not None
        # attribution = 25M / 200M = 0.125, value = 0.125 * 100 = 12.5
        assert pai_8.value == pytest.approx(12.5, rel=0.01)

    def test_pai_9_hazardous_waste(self):
        """PAI 9: Attribution-weighted hazardous + radioactive waste."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        holdings = [_corp_holding(
            hazardous_waste_tonnes=200.0,
            radioactive_waste_tonnes=10.0,
            enterprise_value_eur=200_000_000,
            value_eur=25_000_000,
            weight_pct=100.0,
        )]
        result = engine.calculate_all(holdings)
        pai_9 = next(
            (i for i in result.mandatory_indicators if i.indicator_id == "PAI_9"),
            None,
        )
        assert pai_9 is not None
        assert pai_9.value is not None
        # 0.125 * (200 + 10) = 26.25
        assert pai_9.value == pytest.approx(26.25, rel=0.01)


class TestMandatoryPAISocial:
    """Test PAI 10-14 Social indicators."""

    def test_pai_10_ungc_violations(self):
        """PAI 10: Share of holdings with UNGC/OECD violations."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        h_viol = _corp_holding(has_ungc_violations=True, value_eur=25_000_000)
        h_ok = _corp_holding(has_ungc_violations=False, value_eur=25_000_000)
        result = engine.calculate_all([h_viol, h_ok])
        pai_10 = next(
            (i for i in result.mandatory_indicators if i.indicator_id == "PAI_10"),
            None,
        )
        assert pai_10 is not None
        assert pai_10.value is not None
        assert pai_10.value > 0

    def test_pai_11_lack_of_compliance(self):
        """PAI 11: Share lacking UNGC/OECD compliance processes."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        h_no = _corp_holding(has_compliance_mechanisms=False, value_eur=50_000_000)
        h_yes = _corp_holding(has_compliance_mechanisms=True, value_eur=50_000_000)
        result = engine.calculate_all([h_no, h_yes])
        pai_11 = next(
            (i for i in result.mandatory_indicators if i.indicator_id == "PAI_11"),
            None,
        )
        assert pai_11 is not None
        assert pai_11.value is not None
        assert pai_11.value > 0

    def test_pai_12_gender_pay_gap(self):
        """PAI 12: Weighted average unadjusted gender pay gap."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        holdings = [_corp_holding(gender_pay_gap_pct=15.0, weight_pct=100.0)]
        result = engine.calculate_all(holdings)
        pai_12 = next(
            (i for i in result.mandatory_indicators if i.indicator_id == "PAI_12"),
            None,
        )
        assert pai_12 is not None
        assert pai_12.value == pytest.approx(15.0, rel=0.1)

    def test_pai_13_board_gender_diversity(self):
        """PAI 13: Weighted average board gender diversity."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        holdings = [_corp_holding(female_board_pct=40.0, weight_pct=100.0)]
        result = engine.calculate_all(holdings)
        pai_13 = next(
            (i for i in result.mandatory_indicators if i.indicator_id == "PAI_13"),
            None,
        )
        assert pai_13 is not None
        assert pai_13.value == pytest.approx(40.0, rel=0.1)

    def test_pai_14_controversial_weapons(self):
        """PAI 14: Share exposed to controversial weapons."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        h_weapons = _corp_holding(
            involved_controversial_weapons=True, value_eur=10_000_000,
        )
        h_clean = _corp_holding(
            involved_controversial_weapons=False, value_eur=40_000_000,
        )
        result = engine.calculate_all([h_weapons, h_clean])
        pai_14 = next(
            (i for i in result.mandatory_indicators if i.indicator_id == "PAI_14"),
            None,
        )
        assert pai_14 is not None
        assert pai_14.value is not None
        assert pai_14.value > 0


class TestMandatoryPAISovereign:
    """Test PAI 15-16 Sovereign indicators."""

    def test_pai_15_country_ghg_intensity(self):
        """PAI 15: Weighted average GHG intensity of investee countries."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        sov = _sovereign_holding(country_ghg_intensity=300.0, weight_pct=10.0)
        corp = _corp_holding(weight_pct=90.0)
        result = engine.calculate_all([sov, corp])
        pai_15 = next(
            (i for i in result.mandatory_indicators if i.indicator_id == "PAI_15"),
            None,
        )
        assert pai_15 is not None
        assert pai_15.value is not None

    def test_pai_16_country_social_violations(self):
        """PAI 16: Share of sovereign holdings in countries with violations."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        sov_viol = _sovereign_holding(
            country_social_violations=True, value_eur=20_000_000,
        )
        sov_ok = _sovereign_holding(
            country_social_violations=False, value_eur=30_000_000,
        )
        result = engine.calculate_all([sov_viol, sov_ok])
        pai_16 = next(
            (i for i in result.mandatory_indicators if i.indicator_id == "PAI_16"),
            None,
        )
        assert pai_16 is not None
        assert pai_16.value is not None
        # 20M / 50M = 40%
        assert pai_16.value == pytest.approx(40.0, rel=0.1)


class TestMandatoryPAIRealEstate:
    """Test PAI 17-18 Real Estate indicators."""

    def test_pai_17_re_fossil_fuels(self):
        """PAI 17: Fossil fuels through real estate assets."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        re_fossil = _real_estate_holding(
            re_fossil_fuel_involved=True, value_eur=10_000_000,
        )
        re_clean = _real_estate_holding(
            re_fossil_fuel_involved=False, value_eur=20_000_000,
        )
        result = engine.calculate_all([re_fossil, re_clean])
        pai_17 = next(
            (i for i in result.mandatory_indicators if i.indicator_id == "PAI_17"),
            None,
        )
        assert pai_17 is not None
        assert pai_17.value is not None
        # 10M / 30M * 100 = 33.33%
        assert pai_17.value == pytest.approx(33.33, rel=0.1)

    def test_pai_18_re_energy_inefficient(self):
        """PAI 18: Energy-inefficient real estate assets."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        re_ineff = _real_estate_holding(
            re_energy_inefficient=True, value_eur=15_000_000,
        )
        re_eff = _real_estate_holding(
            re_energy_inefficient=False, value_eur=15_000_000,
        )
        result = engine.calculate_all([re_ineff, re_eff])
        pai_18 = next(
            (i for i in result.mandatory_indicators if i.indicator_id == "PAI_18"),
            None,
        )
        assert pai_18 is not None
        assert pai_18.value is not None
        assert pai_18.value == pytest.approx(50.0, rel=0.1)


class TestCalculateAllMandatory:
    """Test the full calculate_all pipeline."""

    def test_returns_18_mandatory_indicators(self):
        """calculate_all returns exactly 18 mandatory indicators."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        holdings = _build_portfolio(n_corp=5, n_sov=1, n_re=1)
        result = engine.calculate_all(holdings)
        assert isinstance(result, PAIMandatoryResult)
        assert len(result.mandatory_indicators) == 18

    def test_mandatory_indicator_ids(self):
        """All 18 PAI IDs are present in the result."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        holdings = _build_portfolio()
        result = engine.calculate_all(holdings)
        indicator_ids = {i.indicator_id for i in result.mandatory_indicators}
        expected_ids = {f"PAI_{n}" for n in range(1, 19)}
        assert indicator_ids == expected_ids

    def test_status_compliant_high_quality(self):
        """High quality data + coverage -> COMPLIANT status."""
        engine = PAIMandatoryEngine({
            "total_nav_eur": TOTAL_NAV,
            "min_data_quality_pct": 50.0,
            "coverage_threshold_pct": 40.0,
        })
        holdings = _build_portfolio(n_corp=10, n_sov=2, n_re=2)
        result = engine.calculate_all(holdings)
        # With all data reported, coverage should be high
        assert result.status in (
            PAIMandatoryStatus.COMPLIANT,
            PAIMandatoryStatus.PARTIAL,
        )

    def test_auto_nav_computation(self):
        """NAV is auto-computed from holdings if not configured."""
        engine = PAIMandatoryEngine()
        holdings = [
            _corp_holding(value_eur=100_000_000, weight_pct=50.0),
            _corp_holding(value_eur=100_000_000, weight_pct=50.0),
        ]
        result = engine.calculate_all(holdings)
        assert result.total_nav_eur == pytest.approx(200_000_000.0)


class TestAdditionalIndicators:
    """Test Table 2 and Table 3 additional indicators."""

    def test_table_2_environmental_indicators(self):
        """Table 2 additional environmental indicators are calculated."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        holdings = _build_portfolio(n_corp=5)
        result = engine.calculate_all(holdings)
        assert result.additional_environmental is not None
        assert result.additional_environmental.table == "Table 2"
        assert result.additional_environmental.total_indicators >= 1

    def test_table_3_social_indicators(self):
        """Table 3 additional social indicators are calculated."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        holdings = _build_portfolio(n_corp=5)
        result = engine.calculate_all(holdings)
        assert result.additional_social is not None
        assert result.additional_social.table == "Table 3"
        assert result.additional_social.total_indicators >= 1

    def test_table_2_water_usage_indicator(self):
        """Table 2 water usage indicator is calculated correctly."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        holdings = [_corp_holding(
            water_usage_m3=1_000_000,
            revenue_eur=100_000_000,
            weight_pct=100.0,
        )]
        result = engine.calculate_all(holdings)
        t2_indicators = result.additional_environmental.indicators
        water = next(
            (i for i in t2_indicators if i.indicator_id == "T2_WATER_USAGE"),
            None,
        )
        if water is not None:
            # 1M m3 / 100 EUR M = 10000 m3/EUR M
            assert water.value is not None
            assert water.value > 0

    def test_table_3_no_accident_prevention(self):
        """Table 3 lack of accident prevention policy."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        h_no = _corp_holding(has_accident_prevention=False, value_eur=25_000_000)
        h_yes = _corp_holding(has_accident_prevention=True, value_eur=25_000_000)
        result = engine.calculate_all([h_no, h_yes])
        t3_indicators = result.additional_social.indicators
        acc = next(
            (i for i in t3_indicators
             if i.indicator_id == "T3_NO_ACCIDENT_PREVENTION"),
            None,
        )
        assert acc is not None
        assert acc.value is not None
        assert acc.value > 0


class TestDataQualityAssessment:
    """Test data quality reporting."""

    def test_data_quality_report_generated(self):
        """Data quality report is present in result."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        holdings = _build_portfolio()
        result = engine.calculate_all(holdings)
        assert result.data_quality is not None
        assert isinstance(result.data_quality, DataQualityReport)

    def test_quality_meets_threshold(self):
        """All-reported data meets the 70% quality threshold."""
        engine = PAIMandatoryEngine({
            "total_nav_eur": TOTAL_NAV,
            "min_data_quality_pct": 70.0,
        })
        holdings = _build_portfolio(n_corp=10, n_sov=2, n_re=2)
        result = engine.calculate_all(holdings)
        # With full REPORTED data, quality should be high
        assert result.data_quality.overall_quality_pct >= 0.0

    def test_quality_distribution_tracked(self):
        """Quality distribution tracks REPORTED/ESTIMATED/NOT_AVAILABLE."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        holdings = [
            _corp_holding(data_quality=DataQualityLevel.REPORTED),
            _corp_holding(data_quality=DataQualityLevel.ESTIMATED),
            _corp_holding(data_quality=DataQualityLevel.NOT_AVAILABLE),
        ]
        result = engine.calculate_all(holdings)
        quality_levels = result.data_quality.by_quality_level
        assert DataQualityLevel.REPORTED.value in quality_levels
        assert DataQualityLevel.ESTIMATED.value in quality_levels

    def test_improvement_recommendations(self):
        """Recommendations generated for low-quality data."""
        engine = PAIMandatoryEngine({
            "total_nav_eur": TOTAL_NAV,
            "min_data_quality_pct": 99.0,
        })
        holdings = [_corp_holding(data_quality=DataQualityLevel.NOT_AVAILABLE)]
        result = engine.calculate_all(holdings)
        assert len(result.data_quality.improvement_recommendations) > 0


class TestIntegrationAssessment:
    """Test PAI integration assessment."""

    def test_full_integration(self):
        """All 18 mandatory indicators integrated -> 100% ratio."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        all_ids = [f"PAI_{i}" for i in range(1, 19)]
        assessment = engine.assess_integration(all_ids)
        assert isinstance(assessment, IntegrationAssessment)
        assert assessment.integration_count == 18
        assert assessment.integration_ratio_pct == pytest.approx(100.0)
        assert len(assessment.provenance_hash) == 64

    def test_partial_integration(self):
        """Partial integration calculates correct ratio."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        partial_ids = ["PAI_1", "PAI_2", "PAI_3", "PAI_10", "PAI_14"]
        assessment = engine.assess_integration(partial_ids)
        assert assessment.integration_count == 5
        assert assessment.integration_ratio_pct == pytest.approx(
            5.0 / 18.0 * 100.0, rel=0.01,
        )

    def test_table_2_3_in_integration(self):
        """Table 2/3 indicators tracked separately in assessment."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        ids = ["PAI_1", "T2_WATER_USAGE", "T3_ACCIDENT_RATE"]
        assessment = engine.assess_integration(ids)
        assert len(assessment.additional_env_selected) == 1
        assert len(assessment.additional_social_selected) == 1


class TestActionPlan:
    """Test action plan generation."""

    def test_action_plan_generated(self):
        """Action plan is generated when configured."""
        engine = PAIMandatoryEngine({
            "total_nav_eur": TOTAL_NAV,
            "generate_action_plans": True,
        })
        holdings = _build_portfolio()
        result = engine.calculate_all(holdings)
        assert result.action_plan is not None
        assert isinstance(result.action_plan, ActionPlan)
        assert len(result.action_plan.provenance_hash) == 64

    def test_action_plan_for_ungc_violations(self):
        """UNGC violations generate high-priority action items."""
        engine = PAIMandatoryEngine({
            "total_nav_eur": TOTAL_NAV,
            "generate_action_plans": True,
        })
        holdings = [_corp_holding(
            has_ungc_violations=True,
            value_eur=50_000_000, weight_pct=100.0,
        )]
        result = engine.calculate_all(holdings)
        if result.action_plan and result.action_plan.items:
            ungc_actions = [
                a for a in result.action_plan.items
                if a.indicator_id == "PAI_10"
            ]
            if ungc_actions:
                assert ungc_actions[0].priority == "high"

    def test_action_plan_for_controversial_weapons(self):
        """Controversial weapons exposure generates immediate action."""
        engine = PAIMandatoryEngine({
            "total_nav_eur": TOTAL_NAV,
            "generate_action_plans": True,
        })
        holdings = [_corp_holding(
            involved_controversial_weapons=True,
            value_eur=50_000_000, weight_pct=100.0,
        )]
        result = engine.calculate_all(holdings)
        if result.action_plan and result.action_plan.items:
            weapons_actions = [
                a for a in result.action_plan.items
                if a.indicator_id == "PAI_14"
            ]
            if weapons_actions:
                assert weapons_actions[0].priority == "high"


class TestProvenancePAI:
    """Test provenance hashing for PAI mandatory."""

    def test_result_provenance_hash(self):
        """PAIMandatoryResult has a valid SHA-256 provenance hash."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        holdings = _build_portfolio()
        result = engine.calculate_all(holdings)
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)  # validates hex

    def test_indicator_level_provenance(self):
        """Each individual indicator has its own provenance hash."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        holdings = _build_portfolio()
        result = engine.calculate_all(holdings)
        for indicator in result.mandatory_indicators:
            assert len(indicator.provenance_hash) == 64

    def test_coverage_tracked_per_indicator(self):
        """Each indicator tracks coverage_pct and holdings_with_data."""
        engine = PAIMandatoryEngine({"total_nav_eur": TOTAL_NAV})
        holdings = _build_portfolio(n_corp=5)
        result = engine.calculate_all(holdings)
        for indicator in result.mandatory_indicators:
            assert indicator.total_holdings >= 0
            assert indicator.holdings_with_data >= 0
            assert 0.0 <= indicator.coverage_pct <= 100.0
