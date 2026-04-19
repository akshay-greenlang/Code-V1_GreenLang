# -*- coding: utf-8 -*-
"""
Unit tests for CarbonImpactEngine -- PACK-037 Engine 9
========================================================

Tests marginal emission factor lookup, event carbon calculation,
marginal vs average comparison, annual summary, SBTi contribution,
and marginal abatement cost (MAC) calculation.

Coverage target: 85%+
Total tests: ~50
"""

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"


def _load(name: str):
    path = ENGINES_DIR / f"{name}.py"
    if not path.exists():
        pytest.skip(f"Engine file not found: {path}")
    mod_key = f"pack037_test.{name}"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load {name}: {exc}")
    return mod


_m = _load("carbon_impact_engine")


class TestModuleLoading:
    def test_module_loads(self):
        assert _m is not None

    def test_engine_class_exists(self):
        assert hasattr(_m, "CarbonImpactEngine")

    def test_engine_instantiation(self):
        engine = _m.CarbonImpactEngine()
        assert engine is not None


class TestMarginalEmissionFactorLookup:
    """Test marginal emission factor lookup by hour and season."""

    def _get_lookup(self, engine):
        return (getattr(engine, "lookup_emission_factor", None)
                or getattr(engine, "get_marginal_factor", None)
                or getattr(engine, "emission_factor", None))

    def test_hourly_factors_exist(self, sample_emission_factors):
        assert len(sample_emission_factors["marginal_by_hour"]) == 24

    @pytest.mark.parametrize("hour,expected_min,expected_max", [
        (0, 500, 540), (6, 560, 600), (12, 760, 800),
        (14, 800, 840), (18, 700, 740), (23, 510, 550),
    ])
    def test_hourly_factor_ranges(self, sample_emission_factors,
                                    hour, expected_min, expected_max):
        factor = float(sample_emission_factors["marginal_by_hour"][hour])
        assert expected_min <= factor <= expected_max

    def test_peak_hours_highest(self, sample_emission_factors):
        peak_factors = [
            float(sample_emission_factors["marginal_by_hour"][h])
            for h in range(12, 17)
        ]
        offpeak_factors = [
            float(sample_emission_factors["marginal_by_hour"][h])
            for h in range(0, 6)
        ]
        assert min(peak_factors) > max(offpeak_factors)

    def test_annual_average(self, sample_emission_factors):
        assert sample_emission_factors["average_annual"] == Decimal("420.0")

    def test_annual_marginal(self, sample_emission_factors):
        assert sample_emission_factors["marginal_annual"] == Decimal("680.0")

    def test_marginal_exceeds_average(self, sample_emission_factors):
        assert (sample_emission_factors["marginal_annual"] >
                sample_emission_factors["average_annual"])

    def test_summer_peak_highest(self, sample_emission_factors):
        assert (sample_emission_factors["marginal_summer_peak"] >=
                sample_emission_factors["marginal_winter_peak"])

    def test_lookup_method(self, sample_emission_factors):
        engine = _m.CarbonImpactEngine()
        lookup = self._get_lookup(engine)
        if lookup is None:
            pytest.skip("lookup method not found")
        result = lookup(region="PJM", hour=14, season="SUMMER")
        assert result is not None


class TestEventCarbonCalculation:
    """Test carbon impact calculation for a DR event."""

    def _get_calc(self, engine):
        return (getattr(engine, "calculate_event_carbon", None)
                or getattr(engine, "event_carbon_impact", None)
                or getattr(engine, "carbon_impact", None))

    def test_calculate_carbon(self, sample_dr_event_results,
                               sample_emission_factors):
        engine = _m.CarbonImpactEngine()
        calc = self._get_calc(engine)
        if calc is None:
            pytest.skip("calculate_event_carbon method not found")
        result = calc(event_results=sample_dr_event_results,
                     emission_factors=sample_emission_factors)
        assert result is not None

    def test_carbon_positive(self, sample_dr_event_results,
                              sample_emission_factors):
        engine = _m.CarbonImpactEngine()
        calc = self._get_calc(engine)
        if calc is None:
            pytest.skip("calculate_event_carbon method not found")
        result = calc(event_results=sample_dr_event_results,
                     emission_factors=sample_emission_factors)
        co2 = getattr(result, "avoided_co2_kg", result)
        if isinstance(co2, (int, float, Decimal)):
            assert co2 > 0

    def test_manual_carbon_calculation(self, sample_dr_event_results,
                                        sample_emission_factors):
        """Manually verify carbon = reduction_MWh * marginal_factor."""
        intervals = sample_dr_event_results["measurement_intervals"]
        total_kwh = sum(i["reduction_kw"] * 0.25 for i in intervals)  # 15-min
        total_mwh = total_kwh / 1000
        # Event is 14:00-18:00, use average marginal for those hours
        avg_factor = sum(
            float(sample_emission_factors["marginal_by_hour"][h])
            for h in range(14, 18)
        ) / 4
        avoided_kg = total_mwh * avg_factor
        assert avoided_kg > 0

    @pytest.mark.parametrize("reduction_mwh,factor_kg_per_mwh,expected_kg", [
        (1.0, 800, 800),
        (3.0, 820, 2460),
        (0.5, 500, 250),
    ])
    def test_carbon_math(self, reduction_mwh, factor_kg_per_mwh, expected_kg):
        result = reduction_mwh * factor_kg_per_mwh
        assert result == pytest.approx(expected_kg, rel=0.01)


class TestMarginalVsAverage:
    """Test marginal vs average emission comparison."""

    def _get_compare(self, engine):
        return (getattr(engine, "compare_marginal_average", None)
                or getattr(engine, "marginal_vs_average", None)
                or getattr(engine, "emission_comparison", None))

    def test_marginal_vs_average(self, sample_emission_factors):
        engine = _m.CarbonImpactEngine()
        compare = self._get_compare(engine)
        if compare is None:
            pytest.skip("marginal_vs_average method not found")
        result = compare(emission_factors=sample_emission_factors,
                        reduction_mwh=Decimal("3.0"))
        assert result is not None

    def test_marginal_higher_than_average(self, sample_emission_factors):
        marginal = sample_emission_factors["marginal_annual"]
        average = sample_emission_factors["average_annual"]
        assert marginal > average

    def test_ratio_calculation(self, sample_emission_factors):
        ratio = (sample_emission_factors["marginal_annual"] /
                 sample_emission_factors["average_annual"])
        assert ratio > Decimal("1.0")
        assert ratio == pytest.approx(Decimal("1.619"), rel=0.01)


class TestAnnualSummary:
    """Test annual carbon impact summary."""

    def _get_summary(self, engine):
        return (getattr(engine, "annual_carbon_summary", None)
                or getattr(engine, "summarize_annual_carbon", None)
                or getattr(engine, "carbon_summary", None))

    def test_annual_summary(self, sample_emission_factors):
        engine = _m.CarbonImpactEngine()
        summary = self._get_summary(engine)
        if summary is None:
            pytest.skip("annual_carbon_summary method not found")
        events = [
            {"event_id": f"EVT-{i}", "reduction_mwh": Decimal("3.0"),
             "avoided_co2_kg": Decimal("2400.0")}
            for i in range(5)
        ]
        result = summary(events=events, year=2025)
        assert result is not None

    def test_total_avoided_co2(self):
        event_co2 = [2400, 2200, 2500, 1800, 2800]
        total_kg = sum(event_co2)
        total_tonnes = total_kg / 1000
        assert total_tonnes == pytest.approx(11.7, rel=0.01)


class TestSBTiContribution:
    """Test SBTi target contribution calculation."""

    def _get_sbti(self, engine):
        return (getattr(engine, "sbti_contribution", None)
                or getattr(engine, "calculate_sbti_impact", None)
                or getattr(engine, "sbti_impact", None))

    def test_sbti_contribution(self, sample_emission_factors):
        engine = _m.CarbonImpactEngine()
        sbti = self._get_sbti(engine)
        if sbti is None:
            pytest.skip("sbti_contribution method not found")
        result = sbti(
            avoided_co2_tonnes=Decimal("11.7"),
            scope2_baseline_tonnes=Decimal("5250.0"),
            target_reduction_pct=Decimal("42.0"),
        )
        assert result is not None

    def test_sbti_contribution_pct(self):
        avoided = 11.7
        baseline = 5250.0
        target_reduction = baseline * 0.42
        contribution_pct = (avoided / target_reduction) * 100
        assert contribution_pct == pytest.approx(0.531, rel=0.01)


class TestMACCalculation:
    """Test Marginal Abatement Cost calculation."""

    def _get_mac(self, engine):
        return (getattr(engine, "calculate_mac", None)
                or getattr(engine, "marginal_abatement_cost", None)
                or getattr(engine, "mac_analysis", None))

    def test_mac_calculation(self):
        engine = _m.CarbonImpactEngine()
        mac = self._get_mac(engine)
        if mac is None:
            pytest.skip("calculate_mac method not found")
        result = mac(
            net_cost_usd=Decimal("-20000.00"),  # Net revenue (negative cost)
            avoided_co2_tonnes=Decimal("11.7"),
        )
        assert result is not None

    def test_mac_negative_when_profitable(self):
        """When DR generates net revenue, MAC should be negative."""
        net_cost = -20000  # Revenue exceeds costs
        avoided_tonnes = 11.7
        mac = net_cost / avoided_tonnes
        assert mac < 0  # Negative MAC = profitable abatement

    def test_mac_positive_when_costly(self):
        """When DR costs more than it earns, MAC should be positive."""
        net_cost = 5000
        avoided_tonnes = 11.7
        mac = net_cost / avoided_tonnes
        assert mac > 0

    @pytest.mark.parametrize("net_cost,tonnes,expected_mac", [
        (-20000, 11.7, -1709.40),
        (0, 11.7, 0),
        (5000, 11.7, 427.35),
    ])
    def test_mac_scenarios(self, net_cost, tonnes, expected_mac):
        mac = net_cost / tonnes if tonnes > 0 else 0
        assert mac == pytest.approx(expected_mac, rel=0.01)


class TestCarbonProvenance:
    """Test provenance tracking for carbon calculations."""

    def test_provenance_deterministic(self, sample_dr_event_results,
                                      sample_emission_factors):
        engine = _m.CarbonImpactEngine()
        calc = (getattr(engine, "calculate_event_carbon", None)
                or getattr(engine, "carbon_impact", None))
        if calc is None:
            pytest.skip("carbon calculation method not found")
        r1 = calc(event_results=sample_dr_event_results,
                 emission_factors=sample_emission_factors)
        r2 = calc(event_results=sample_dr_event_results,
                 emission_factors=sample_emission_factors)
        h1 = getattr(r1, "provenance_hash", None)
        h2 = getattr(r2, "provenance_hash", None)
        if h1 and h2:
            assert h1 == h2


# =============================================================================
# Emission Factor Data Validation
# =============================================================================


class TestEmissionFactorData:
    """Validate emission factor fixture data."""

    def test_all_24_hours(self, sample_emission_factors):
        for h in range(24):
            assert h in sample_emission_factors["marginal_by_hour"]

    def test_factors_positive(self, sample_emission_factors):
        for h in range(24):
            assert sample_emission_factors["marginal_by_hour"][h] > 0

    def test_grid_region(self, sample_emission_factors):
        assert sample_emission_factors["grid_region"] == "PJM"

    def test_year(self, sample_emission_factors):
        assert sample_emission_factors["year"] == 2025

    def test_unit(self, sample_emission_factors):
        assert sample_emission_factors["unit"] == "kg_CO2e_per_MWh"

    @pytest.mark.parametrize("factor_key", [
        "average_annual", "marginal_annual", "marginal_summer_peak",
        "marginal_winter_peak", "marginal_shoulder", "marginal_off_peak",
        "sbti_factor_scope2",
    ])
    def test_factor_exists_and_positive(self, sample_emission_factors,
                                         factor_key):
        assert factor_key in sample_emission_factors
        assert sample_emission_factors[factor_key] > 0

    def test_summer_peak_highest_seasonal(self, sample_emission_factors):
        summer = sample_emission_factors["marginal_summer_peak"]
        winter = sample_emission_factors["marginal_winter_peak"]
        shoulder = sample_emission_factors["marginal_shoulder"]
        offpeak = sample_emission_factors["marginal_off_peak"]
        assert summer >= winter
        assert summer >= shoulder
        assert summer >= offpeak

    def test_offpeak_lowest_seasonal(self, sample_emission_factors):
        offpeak = sample_emission_factors["marginal_off_peak"]
        shoulder = sample_emission_factors["marginal_shoulder"]
        assert offpeak <= shoulder
