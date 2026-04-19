# -*- coding: utf-8 -*-
"""Test suite for PACK-029 - Corrective Action Engine (Engine 6)."""
import asyncio
import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.corrective_action_engine import (
    CorrectiveActionEngine, CorrectiveActionInput, CorrectiveActionResult,
    AvailableInitiative, InitiativeCategory,
)
from .conftest import assert_provenance_hash, assert_processing_time, timed_block


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_initiatives():
    return [
        AvailableInitiative(
            initiative_id="INI-001", name="Renewable Procurement",
            category=InitiativeCategory.RENEWABLE_ENERGY,
            annual_reduction_tco2e=Decimal("15000"),
            cost_per_tco2e=Decimal("25"), capex=Decimal("2500000"),
        ),
        AvailableInitiative(
            initiative_id="INI-002", name="Energy Efficiency",
            category=InitiativeCategory.ENERGY_EFFICIENCY,
            annual_reduction_tco2e=Decimal("8000"),
            cost_per_tco2e=Decimal("15"), capex=Decimal("500000"),
        ),
        AvailableInitiative(
            initiative_id="INI-003", name="Fleet Electrification",
            category=InitiativeCategory.ELECTRIFICATION,
            annual_reduction_tco2e=Decimal("5000"),
            cost_per_tco2e=Decimal("60"), capex=Decimal("1500000"),
        ),
    ]


def _make_input(**kwargs):
    defaults = dict(
        entity_name="GreenCorp Industries",
        current_emissions_tco2e=Decimal("183000"),
        target_emissions_tco2e=Decimal("117740"),
        target_year=2030,
        current_year=2024,
        available_initiatives=_make_initiatives(),
    )
    defaults.update(kwargs)
    return CorrectiveActionInput(**defaults)


class TestInstantiation:
    def test_creates(self):
        assert CorrectiveActionEngine() is not None

    def test_version(self):
        assert CorrectiveActionEngine().engine_version == "1.0.0"

    def test_has_calculate(self):
        assert hasattr(CorrectiveActionEngine(), "calculate")

    def test_has_batch(self):
        assert hasattr(CorrectiveActionEngine(), "calculate_batch")

    def test_categories(self):
        cats = CorrectiveActionEngine().get_initiative_categories()
        assert isinstance(cats, list)
        assert len(cats) > 0


class TestBasicCalculation:
    def test_basic_result(self):
        r = _run(CorrectiveActionEngine().calculate(_make_input()))
        assert r is not None
        assert r.entity_name == "GreenCorp Industries"

    def test_gap_quantification(self):
        r = _run(CorrectiveActionEngine().calculate(_make_input()))
        assert r.gap is not None
        assert isinstance(r.gap.gap_tco2e, Decimal)
        assert r.gap.gap_tco2e > Decimal("0")

    def test_selected_initiatives(self):
        r = _run(CorrectiveActionEngine().calculate(_make_input()))
        assert isinstance(r.selected_initiatives, list)
        assert len(r.selected_initiatives) > 0

    def test_total_portfolio_reduction(self):
        r = _run(CorrectiveActionEngine().calculate(_make_input()))
        assert isinstance(r.total_portfolio_reduction_tco2e, Decimal)
        assert r.total_portfolio_reduction_tco2e > Decimal("0")

    def test_provenance(self):
        assert_provenance_hash(_run(CorrectiveActionEngine().calculate(_make_input())))

    def test_processing_time(self):
        assert_processing_time(_run(CorrectiveActionEngine().calculate(_make_input())))


class TestGapAnalysis:
    def test_gap_pct(self):
        r = _run(CorrectiveActionEngine().calculate(_make_input()))
        assert isinstance(r.gap.gap_pct, Decimal)

    def test_remaining_years(self):
        r = _run(CorrectiveActionEngine().calculate(_make_input()))
        assert r.gap.remaining_years > 0

    def test_urgency(self):
        r = _run(CorrectiveActionEngine().calculate(_make_input()))
        assert r.gap.urgency in ("low", "medium", "high", "critical")


class TestCatchUpTimeline:
    def test_catch_up_exists(self):
        r = _run(CorrectiveActionEngine().calculate(_make_input()))
        assert r.catch_up_timeline is not None

    def test_catch_up_fields(self):
        r = _run(CorrectiveActionEngine().calculate(_make_input()))
        ct = r.catch_up_timeline
        assert isinstance(ct.catch_up_rate_pct, Decimal)
        assert isinstance(ct.is_achievable, bool)


class TestInvestmentAnalysis:
    def test_investment_exists(self):
        r = _run(CorrectiveActionEngine().calculate(_make_input()))
        assert r.investment_analysis is not None

    def test_investment_fields(self):
        r = _run(CorrectiveActionEngine().calculate(_make_input()))
        ia = r.investment_analysis
        assert isinstance(ia.total_capex, Decimal)
        assert isinstance(ia.cost_per_tco2e_abated, Decimal)


class TestAcceleratedScenarios:
    def test_scenarios(self):
        r = _run(CorrectiveActionEngine().calculate(_make_input()))
        assert isinstance(r.accelerated_scenarios, list)


class TestUrgencyLevel:
    def test_urgency(self):
        r = _run(CorrectiveActionEngine().calculate(_make_input()))
        assert r.urgency_level in ("low", "medium", "high", "critical")


class TestScales:
    @pytest.mark.parametrize("current,target", [
        (50000, 29000), (200000, 116000), (1000000, 580000),
        (5000000, 2900000), (50000000, 29000000),
    ])
    def test_various_scales(self, current, target):
        r = _run(CorrectiveActionEngine().calculate(_make_input(
            current_emissions_tco2e=Decimal(str(current)),
            target_emissions_tco2e=Decimal(str(target)),
        )))
        assert r is not None

    @pytest.mark.parametrize("entity", ["Corp A", "Corp B", "Corp C"])
    def test_entities(self, entity):
        r = _run(CorrectiveActionEngine().calculate(_make_input(entity_name=entity)))
        assert r.entity_name == entity

    @pytest.mark.parametrize("year", [2026, 2028, 2030, 2035, 2040])
    def test_target_years(self, year):
        r = _run(CorrectiveActionEngine().calculate(_make_input(target_year=year)))
        assert r is not None


class TestDecimalPrecision:
    def test_gap_decimal(self):
        r = _run(CorrectiveActionEngine().calculate(_make_input()))
        assert isinstance(r.gap.gap_tco2e, Decimal)

    @pytest.mark.parametrize("val", ["183456.789", "999999.999", "1000000.001"])
    def test_precision(self, val):
        r = _run(CorrectiveActionEngine().calculate(_make_input(
            current_emissions_tco2e=Decimal(val))))
        assert isinstance(r.gap.gap_tco2e, Decimal)


class TestRecommendations:
    def test_recommendations(self):
        assert isinstance(_run(CorrectiveActionEngine().calculate(_make_input())).recommendations, list)

    def test_warnings(self):
        assert isinstance(_run(CorrectiveActionEngine().calculate(_make_input())).warnings, list)

    def test_data_quality(self):
        r = _run(CorrectiveActionEngine().calculate(_make_input()))
        assert r.data_quality in ("high", "medium", "low", "estimated")


class TestPerformance:
    def test_under_1_second(self):
        with timed_block(max_ms=1000):
            _run(CorrectiveActionEngine().calculate(_make_input()))

    def test_benchmark(self):
        e = CorrectiveActionEngine()
        inp = _make_input()
        with timed_block(max_ms=10000):
            for _ in range(100):
                _run(e.calculate(inp))


class TestBatch:
    def test_batch(self):
        inputs = [_make_input(entity_name=f"Corp {i}") for i in range(3)]
        results = _run(CorrectiveActionEngine().calculate_batch(inputs))
        assert len(results) == 3


class TestEdgeCases:
    def test_no_gap(self):
        r = _run(CorrectiveActionEngine().calculate(_make_input(
            current_emissions_tco2e=Decimal("100000"),
            target_emissions_tco2e=Decimal("120000"),
        )))
        assert r is not None

    def test_no_initiatives(self):
        r = _run(CorrectiveActionEngine().calculate(_make_input(available_initiatives=[])))
        assert r is not None

    def test_model_dump(self):
        d = _run(CorrectiveActionEngine().calculate(_make_input())).model_dump()
        assert isinstance(d, dict)

    def test_sha256(self):
        h = _run(CorrectiveActionEngine().calculate(_make_input())).provenance_hash
        assert len(h) == 64
        int(h, 16)

    def test_budget_constraint(self):
        r = _run(CorrectiveActionEngine().calculate(_make_input(
            budget_constraint=Decimal("1000000"))))
        assert r is not None

    def test_gap_coverage(self):
        r = _run(CorrectiveActionEngine().calculate(_make_input()))
        assert isinstance(r.gap_coverage_pct, Decimal)
