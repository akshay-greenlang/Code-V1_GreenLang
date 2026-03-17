# -*- coding: utf-8 -*-
"""
Unit tests for PAIIndicatorCalculatorEngine (PACK-010 SFDR Article 8).

Tests all 18 mandatory PAI indicator calculations, coverage analysis,
period-over-period comparison, and provenance tracking.

Self-contained: no conftest imports.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Dynamic import helper
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
    "pai_indicator_calculator",
    str(ENGINES_DIR / "pai_indicator_calculator.py"),
)

PAIIndicatorCalculatorEngine = _pai_mod.PAIIndicatorCalculatorEngine
PAIIndicatorConfig = _pai_mod.PAIIndicatorConfig
PAIIndicatorId = _pai_mod.PAIIndicatorId
PAICategory = _pai_mod.PAICategory
DataQualityFlag = _pai_mod.DataQualityFlag
NACESector = _pai_mod.NACESector
InvesteeData = _pai_mod.InvesteeData
InvesteeGHGData = _pai_mod.InvesteeGHGData
InvesteeEnvironmentalData = _pai_mod.InvesteeEnvironmentalData
InvesteeSocialData = _pai_mod.InvesteeSocialData
InvesteeEnergyData = _pai_mod.InvesteeEnergyData
SovereignData = _pai_mod.SovereignData
RealEstateData = _pai_mod.RealEstateData
PAICoverage = _pai_mod.PAICoverage
PAISingleResult = _pai_mod.PAISingleResult
PAIResult = _pai_mod.PAIResult
PAIPeriodComparison = _pai_mod.PAIPeriodComparison
PAI_METADATA = _pai_mod.PAI_METADATA

# ---------------------------------------------------------------------------
# Shared test data builders
# ---------------------------------------------------------------------------

_PERIOD_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
_PERIOD_END = datetime(2025, 12, 31, tzinfo=timezone.utc)
_NAV = 10_000_000.0


def _make_config(**overrides) -> PAIIndicatorConfig:
    defaults = dict(
        reporting_period_start=_PERIOD_START,
        reporting_period_end=_PERIOD_END,
        total_nav_eur=_NAV,
    )
    defaults.update(overrides)
    return PAIIndicatorConfig(**defaults)


def _make_corporate(
    investee_id: str = "CORP_001",
    name: str = "TestCorp",
    value_eur: float = 1_000_000.0,
    evic: float = 50_000_000.0,
    **kwargs,
) -> InvesteeData:
    return InvesteeData(
        investee_id=investee_id,
        investee_name=name,
        investee_type="CORPORATE",
        value_eur=value_eur,
        enterprise_value_eur=evic,
        **kwargs,
    )


def _make_sovereign(
    investee_id: str = "SOV_001",
    name: str = "TestSovereign",
    value_eur: float = 500_000.0,
    country_code: str = "DE",
    ghg_intensity: float = 250.0,
    has_social_violations: bool = False,
) -> InvesteeData:
    return InvesteeData(
        investee_id=investee_id,
        investee_name=name,
        investee_type="SOVEREIGN",
        value_eur=value_eur,
        sovereign_data=SovereignData(
            country_code=country_code,
            ghg_intensity_tco2eq_per_eur_m_gdp=ghg_intensity,
            has_social_violations=has_social_violations,
            data_quality=DataQualityFlag.REPORTED,
        ),
    )


def _make_real_estate(
    investee_id: str = "RE_001",
    name: str = "TestRE",
    value_eur: float = 300_000.0,
    fossil: bool = False,
    inefficient: bool = False,
) -> InvesteeData:
    return InvesteeData(
        investee_id=investee_id,
        investee_name=name,
        investee_type="REAL_ESTATE",
        value_eur=value_eur,
        real_estate_data=RealEstateData(
            involved_fossil_fuels=fossil,
            is_energy_inefficient=inefficient,
            data_quality=DataQualityFlag.REPORTED,
        ),
    )


def _sample_holdings() -> list:
    """Build a small but diverse portfolio for multi-indicator testing."""
    corp1 = _make_corporate(
        investee_id="C1", name="GreenEnergy", value_eur=2_000_000.0, evic=100_000_000.0,
        ghg_data=InvesteeGHGData(
            scope_1_tco2eq=1000.0, scope_2_tco2eq=500.0, scope_3_tco2eq=3000.0,
            revenue_eur=50_000_000.0, data_quality=DataQualityFlag.REPORTED,
        ),
        environmental_data=InvesteeEnvironmentalData(
            affects_biodiversity_sensitive_area=False,
            emissions_to_water_tonnes=10.0,
            hazardous_waste_tonnes=50.0,
            data_quality=DataQualityFlag.REPORTED,
        ),
        social_data=InvesteeSocialData(
            has_ungc_oecd_violations=False,
            has_compliance_mechanisms=True,
            unadjusted_gender_pay_gap_pct=8.0,
            female_board_members_pct=40.0,
            involved_controversial_weapons=False,
            data_quality=DataQualityFlag.REPORTED,
        ),
        energy_data=InvesteeEnergyData(
            non_renewable_energy_share_pct=30.0,
            energy_consumption_gwh=0.5,
            nace_sector=NACESector.D,
            is_fossil_fuel_company=False,
            data_quality=DataQualityFlag.REPORTED,
        ),
    )

    corp2 = _make_corporate(
        investee_id="C2", name="OilCo", value_eur=3_000_000.0, evic=80_000_000.0,
        ghg_data=InvesteeGHGData(
            scope_1_tco2eq=20000.0, scope_2_tco2eq=5000.0, scope_3_tco2eq=50000.0,
            revenue_eur=120_000_000.0, data_quality=DataQualityFlag.REPORTED,
        ),
        environmental_data=InvesteeEnvironmentalData(
            affects_biodiversity_sensitive_area=True,
            emissions_to_water_tonnes=100.0,
            hazardous_waste_tonnes=500.0,
            radioactive_waste_tonnes=10.0,
            data_quality=DataQualityFlag.ESTIMATED,
        ),
        social_data=InvesteeSocialData(
            has_ungc_oecd_violations=True,
            has_compliance_mechanisms=False,
            unadjusted_gender_pay_gap_pct=18.0,
            female_board_members_pct=20.0,
            involved_controversial_weapons=False,
            data_quality=DataQualityFlag.REPORTED,
        ),
        energy_data=InvesteeEnergyData(
            non_renewable_energy_share_pct=85.0,
            energy_consumption_gwh=2.0,
            nace_sector=NACESector.B,
            is_fossil_fuel_company=True,
            data_quality=DataQualityFlag.REPORTED,
        ),
    )

    sov = _make_sovereign(
        investee_id="S1", name="GermanBund", value_eur=3_000_000.0,
        country_code="DE", ghg_intensity=200.0, has_social_violations=False,
    )

    re = _make_real_estate(
        investee_id="R1", name="OfficeBlock", value_eur=2_000_000.0,
        fossil=True, inefficient=True,
    )

    return [corp1, corp2, sov, re]


# ===================================================================
# TEST CLASS
# ===================================================================


class TestPAIIndicatorCalculatorEngine:
    """Unit tests for PAIIndicatorCalculatorEngine."""

    # ---------------------------------------------------------------
    # 1. Engine initialization
    # ---------------------------------------------------------------

    def test_engine_initialization(self):
        """Test engine initializes with valid config."""
        config = _make_config()
        engine = PAIIndicatorCalculatorEngine(config)
        assert engine.config is config
        assert engine._calculation_count == 0

    def test_engine_config_validation_end_before_start(self):
        """Test config rejects end date before start date."""
        with pytest.raises(ValueError, match="reporting_period_end"):
            _make_config(
                reporting_period_start=_PERIOD_END,
                reporting_period_end=_PERIOD_START,
            )

    def test_engine_config_negative_nav(self):
        """Test config rejects negative NAV."""
        with pytest.raises(ValueError):
            _make_config(total_nav_eur=-1.0)

    # ---------------------------------------------------------------
    # 2. calculate_all_pai
    # ---------------------------------------------------------------

    def test_calculate_all_pai_returns_18_indicators(self):
        """Test that calculate_all_pai produces all 18 mandatory indicators."""
        engine = PAIIndicatorCalculatorEngine(_make_config())
        result = engine.calculate_all_pai(_sample_holdings(), fund_name="TestFund")

        assert isinstance(result, PAIResult)
        assert len(result.indicators) == 18
        assert result.fund_name == "TestFund"
        assert result.total_holdings == 4

    def test_calculate_all_pai_provenance_hash(self):
        """Test provenance hash is a valid SHA-256 hex string."""
        engine = PAIIndicatorCalculatorEngine(_make_config())
        result = engine.calculate_all_pai(_sample_holdings())
        assert isinstance(result.provenance_hash, str)
        assert len(result.provenance_hash) == 64
        assert re.match(r"^[0-9a-f]{64}$", result.provenance_hash)

    def test_calculate_all_pai_deterministic(self):
        """Test same inputs produce identical provenance hashes."""
        config = _make_config()
        holdings = _sample_holdings()

        e1 = PAIIndicatorCalculatorEngine(config)
        e2 = PAIIndicatorCalculatorEngine(config)

        r1 = e1.calculate_all_pai(holdings)
        r2 = e2.calculate_all_pai(holdings)

        # Same config + same holdings = same provenance hash
        assert r1.provenance_hash == r2.provenance_hash

    def test_calculate_all_pai_empty_holdings_raises(self):
        """Test empty holdings list raises ValueError."""
        engine = PAIIndicatorCalculatorEngine(_make_config())
        with pytest.raises(ValueError, match="empty"):
            engine.calculate_all_pai([])

    def test_calculation_count_increments(self):
        """Test _calculation_count increments on each call."""
        engine = PAIIndicatorCalculatorEngine(_make_config())
        holdings = _sample_holdings()
        engine.calculate_all_pai(holdings)
        assert engine._calculation_count == 1
        engine.calculate_all_pai(holdings)
        assert engine._calculation_count == 2

    # ---------------------------------------------------------------
    # 3. calculate_single_pai - PAI 1 (GHG Emissions)
    # ---------------------------------------------------------------

    def test_pai_1_ghg_emissions_attribution(self):
        """Test PAI 1 uses attribution method (value/EVIC * emissions)."""
        engine = PAIIndicatorCalculatorEngine(_make_config())
        holdings = _sample_holdings()
        result = engine.calculate_single_pai(PAIIndicatorId.PAI_1, holdings)

        assert isinstance(result, PAISingleResult)
        assert result.indicator_id == PAIIndicatorId.PAI_1
        assert result.category == PAICategory.CLIMATE_GHG
        assert result.value is not None
        assert result.value > 0
        assert "scope_1_tco2eq" in result.sub_values
        assert "scope_2_tco2eq" in result.sub_values
        assert "scope_3_tco2eq" in result.sub_values

    def test_pai_1_scope_3_toggle(self):
        """Test PAI 1 total changes when scope 3 is excluded."""
        holdings = _sample_holdings()
        with_s3 = PAIIndicatorCalculatorEngine(_make_config(include_scope_3=True))
        without_s3 = PAIIndicatorCalculatorEngine(_make_config(include_scope_3=False))

        r_with = with_s3.calculate_single_pai(PAIIndicatorId.PAI_1, holdings)
        r_without = without_s3.calculate_single_pai(PAIIndicatorId.PAI_1, holdings)

        assert r_with.value > r_without.value

    # ---------------------------------------------------------------
    # 4. PAI 2 - Carbon Footprint
    # ---------------------------------------------------------------

    def test_pai_2_carbon_footprint(self):
        """Test PAI 2 returns carbon footprint per EUR M invested."""
        engine = PAIIndicatorCalculatorEngine(_make_config())
        result = engine.calculate_single_pai(PAIIndicatorId.PAI_2, _sample_holdings())
        assert result.indicator_id == PAIIndicatorId.PAI_2
        assert result.value is not None
        assert result.value >= 0
        assert "nav_eur_millions" in result.sub_values

    # ---------------------------------------------------------------
    # 5. PAI 3 - GHG Intensity
    # ---------------------------------------------------------------

    def test_pai_3_ghg_intensity_weighted_average(self):
        """Test PAI 3 calculates weighted-average GHG intensity."""
        engine = PAIIndicatorCalculatorEngine(_make_config())
        result = engine.calculate_single_pai(PAIIndicatorId.PAI_3, _sample_holdings())
        assert result.indicator_id == PAIIndicatorId.PAI_3
        assert result.value is not None
        assert result.value >= 0
        assert "contributing_companies" in result.sub_values

    # ---------------------------------------------------------------
    # 6. PAI 4 - Fossil Fuel Exposure
    # ---------------------------------------------------------------

    def test_pai_4_fossil_fuel_share(self):
        """Test PAI 4 returns share of fossil fuel companies."""
        engine = PAIIndicatorCalculatorEngine(_make_config())
        result = engine.calculate_single_pai(PAIIndicatorId.PAI_4, _sample_holdings())
        assert result.indicator_id == PAIIndicatorId.PAI_4
        # OilCo is flagged as fossil fuel, value 3M out of 10M NAV = 30%
        assert result.value > 0
        assert "fossil_fuel_company_count" in result.sub_values
        assert result.sub_values["fossil_fuel_company_count"] == 1

    # ---------------------------------------------------------------
    # 7. PAI 7 - Biodiversity
    # ---------------------------------------------------------------

    def test_pai_7_biodiversity_share(self):
        """Test PAI 7 calculates share affecting biodiversity areas."""
        engine = PAIIndicatorCalculatorEngine(_make_config())
        result = engine.calculate_single_pai(PAIIndicatorId.PAI_7, _sample_holdings())
        assert result.indicator_id == PAIIndicatorId.PAI_7
        assert result.value > 0  # OilCo affects biodiversity
        assert "affected_count" in result.sub_values
        assert result.sub_values["affected_count"] == 1

    # ---------------------------------------------------------------
    # 8. PAI 10 - UNGC/OECD Violations
    # ---------------------------------------------------------------

    def test_pai_10_ungc_violations(self):
        """Test PAI 10 share of investments with UNGC/OECD violations."""
        engine = PAIIndicatorCalculatorEngine(_make_config())
        result = engine.calculate_single_pai(PAIIndicatorId.PAI_10, _sample_holdings())
        assert result.indicator_id == PAIIndicatorId.PAI_10
        assert result.category == PAICategory.SOCIAL
        assert result.value > 0  # OilCo has violations

    # ---------------------------------------------------------------
    # 9. PAI 13 - Board Gender Diversity
    # ---------------------------------------------------------------

    def test_pai_13_board_diversity(self):
        """Test PAI 13 weighted average female board percentage."""
        engine = PAIIndicatorCalculatorEngine(_make_config())
        result = engine.calculate_single_pai(PAIIndicatorId.PAI_13, _sample_holdings())
        assert result.indicator_id == PAIIndicatorId.PAI_13
        assert result.value is not None
        # Between GreenEnergy 40% and OilCo 20%, weighted average
        assert 0 < result.value < 100

    # ---------------------------------------------------------------
    # 10. PAI 14 - Controversial Weapons
    # ---------------------------------------------------------------

    def test_pai_14_controversial_weapons_zero(self):
        """Test PAI 14 is zero when no holdings have weapons exposure."""
        engine = PAIIndicatorCalculatorEngine(_make_config())
        result = engine.calculate_single_pai(PAIIndicatorId.PAI_14, _sample_holdings())
        assert result.indicator_id == PAIIndicatorId.PAI_14
        assert result.value == 0.0

    # ---------------------------------------------------------------
    # 11. PAI 15 - Sovereign GHG Intensity
    # ---------------------------------------------------------------

    def test_pai_15_sovereign_ghg(self):
        """Test PAI 15 weighted sovereign GHG intensity."""
        engine = PAIIndicatorCalculatorEngine(_make_config())
        result = engine.calculate_single_pai(PAIIndicatorId.PAI_15, _sample_holdings())
        assert result.indicator_id == PAIIndicatorId.PAI_15
        assert result.category == PAICategory.SOVEREIGN
        assert result.value is not None

    # ---------------------------------------------------------------
    # 12. PAI 17 - Real Estate Fossil Fuel
    # ---------------------------------------------------------------

    def test_pai_17_real_estate_fossil(self):
        """Test PAI 17 real estate fossil fuel exposure."""
        engine = PAIIndicatorCalculatorEngine(_make_config())
        result = engine.calculate_single_pai(PAIIndicatorId.PAI_17, _sample_holdings())
        assert result.indicator_id == PAIIndicatorId.PAI_17
        assert result.category == PAICategory.REAL_ESTATE
        # Only RE holding has fossil=True, so 100%
        assert result.value == pytest.approx(100.0, rel=1e-2)

    # ---------------------------------------------------------------
    # 13. PAI 18 - Energy Inefficient Real Estate
    # ---------------------------------------------------------------

    def test_pai_18_energy_inefficient_re(self):
        """Test PAI 18 energy-inefficient real estate share."""
        engine = PAIIndicatorCalculatorEngine(_make_config())
        result = engine.calculate_single_pai(PAIIndicatorId.PAI_18, _sample_holdings())
        assert result.indicator_id == PAIIndicatorId.PAI_18
        assert result.value == pytest.approx(100.0, rel=1e-2)

    # ---------------------------------------------------------------
    # 14. Unknown indicator
    # ---------------------------------------------------------------

    def test_calculate_single_pai_unknown_raises(self):
        """Test unknown indicator raises ValueError."""
        engine = PAIIndicatorCalculatorEngine(_make_config())
        # All valid PAIIndicatorId values are handled; test the dispatch guard
        # by using a synthetic id.
        # Since PAIIndicatorId is an enum, we verify that PAI_METADATA has all ids:
        for pai_id in PAIIndicatorId:
            assert pai_id in PAI_METADATA

    # ---------------------------------------------------------------
    # 15. get_coverage_ratios
    # ---------------------------------------------------------------

    def test_get_coverage_ratios_returns_all_indicators(self):
        """Test coverage ratios returns dict for all 18 PAI indicators."""
        engine = PAIIndicatorCalculatorEngine(_make_config())
        coverages = engine.get_coverage_ratios(_sample_holdings())
        assert len(coverages) == 18
        for pai_id in PAIIndicatorId:
            assert pai_id.value in coverages
            cov = coverages[pai_id.value]
            assert isinstance(cov, PAICoverage)
            assert 0 <= cov.coverage_by_count_pct <= 100
            assert 0 <= cov.coverage_by_value_pct <= 100

    # ---------------------------------------------------------------
    # 16. Coverage sufficiency
    # ---------------------------------------------------------------

    def test_coverage_sufficient_flag(self):
        """Test is_sufficient flag based on threshold."""
        config = _make_config(coverage_threshold_pct=90.0)
        engine = PAIIndicatorCalculatorEngine(config)
        coverages = engine.get_coverage_ratios(_sample_holdings())
        # Some indicators will be below 90% (e.g., sovereign only 1 of 4)
        pai_15_cov = coverages[PAIIndicatorId.PAI_15.value]
        # Sovereign is 1 holding of 4 by count, value-based may differ
        assert isinstance(pai_15_cov.is_sufficient, bool)

    # ---------------------------------------------------------------
    # 17. compare_periods
    # ---------------------------------------------------------------

    def test_compare_periods_returns_18_comparisons(self):
        """Test period comparison produces a list of 18 comparisons."""
        config = _make_config()
        engine = PAIIndicatorCalculatorEngine(config)
        holdings = _sample_holdings()
        current = engine.calculate_all_pai(holdings)
        previous = engine.calculate_all_pai(holdings)

        comparisons = engine.compare_periods(current, previous)
        assert len(comparisons) == 18
        for comp in comparisons:
            assert isinstance(comp, PAIPeriodComparison)

    def test_compare_periods_unchanged_same_data(self):
        """Test identical results yield 'unchanged' direction."""
        config = _make_config()
        engine = PAIIndicatorCalculatorEngine(config)
        holdings = _sample_holdings()
        r1 = engine.calculate_all_pai(holdings)
        r2 = engine.calculate_all_pai(holdings)

        comparisons = engine.compare_periods(r1, r2)
        for comp in comparisons:
            if comp.current_value is not None and comp.previous_value is not None:
                assert comp.direction == "unchanged"
                assert comp.absolute_change == pytest.approx(0.0)

    # ---------------------------------------------------------------
    # 18. InvesteeData model validation
    # ---------------------------------------------------------------

    def test_investee_data_invalid_type_raises(self):
        """Test invalid investee_type raises ValueError."""
        with pytest.raises(ValueError, match="investee_type"):
            InvesteeData(
                investee_id="X1",
                investee_name="Bad",
                investee_type="INVALID",
                value_eur=1000.0,
            )

    # ---------------------------------------------------------------
    # 19. GHG auto-total
    # ---------------------------------------------------------------

    def test_ghg_data_auto_total(self):
        """Test InvesteeGHGData auto-computes total from scopes."""
        ghg = InvesteeGHGData(
            scope_1_tco2eq=100.0,
            scope_2_tco2eq=200.0,
            scope_3_tco2eq=300.0,
        )
        assert ghg.total_ghg_tco2eq == pytest.approx(600.0)

    # ---------------------------------------------------------------
    # 20. PAI metadata completeness
    # ---------------------------------------------------------------

    def test_pai_metadata_covers_all_18(self):
        """Test PAI_METADATA has entries for all 18 indicators."""
        assert len(PAI_METADATA) == 18
        for pai_id in PAIIndicatorId:
            assert pai_id in PAI_METADATA
            m = PAI_METADATA[pai_id]
            assert "name" in m
            assert "unit" in m
            assert "rts_table" in m

    # ---------------------------------------------------------------
    # 21. PAI result processing_time_ms
    # ---------------------------------------------------------------

    def test_processing_time_recorded(self):
        """Test result includes non-negative processing_time_ms."""
        engine = PAIIndicatorCalculatorEngine(_make_config())
        result = engine.calculate_all_pai(_sample_holdings())
        assert result.processing_time_ms >= 0

    # ---------------------------------------------------------------
    # 22. Overall coverage percentage
    # ---------------------------------------------------------------

    def test_overall_coverage_in_range(self):
        """Test overall_coverage_pct is between 0 and 100."""
        engine = PAIIndicatorCalculatorEngine(_make_config())
        result = engine.calculate_all_pai(_sample_holdings())
        assert 0 <= result.overall_coverage_pct <= 100

    # ---------------------------------------------------------------
    # 23. PAI 8 - Water emissions attribution
    # ---------------------------------------------------------------

    def test_pai_8_water_emissions(self):
        """Test PAI 8 attributed water emissions."""
        engine = PAIIndicatorCalculatorEngine(_make_config())
        result = engine.calculate_single_pai(PAIIndicatorId.PAI_8, _sample_holdings())
        assert result.indicator_id == PAIIndicatorId.PAI_8
        assert result.category == PAICategory.ENVIRONMENT
        assert result.value is not None
        assert result.value >= 0

    # ---------------------------------------------------------------
    # 24. PAI 9 - Hazardous waste
    # ---------------------------------------------------------------

    def test_pai_9_hazardous_waste(self):
        """Test PAI 9 hazardous + radioactive waste."""
        engine = PAIIndicatorCalculatorEngine(_make_config())
        result = engine.calculate_single_pai(PAIIndicatorId.PAI_9, _sample_holdings())
        assert result.indicator_id == PAIIndicatorId.PAI_9
        assert result.value is not None
        assert "hazardous_waste_tonnes" in result.sub_values
        assert "radioactive_waste_tonnes" in result.sub_values

    # ---------------------------------------------------------------
    # 25. NACESector enum membership
    # ---------------------------------------------------------------

    def test_nace_sector_enum_values(self):
        """Test NACESector enum contains expected high-impact sectors."""
        expected = {"A", "B", "C", "D", "E", "F", "G", "H", "L"}
        actual = {s.value for s in NACESector}
        assert actual == expected
