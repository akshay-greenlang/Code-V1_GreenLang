# -*- coding: utf-8 -*-
"""Test suite for PACK-029 - Initiative Scheduler Engine (Engine 8)."""
import asyncio
import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.initiative_scheduler_engine import (
    InitiativeSchedulerEngine, InitiativeSchedulerInput, InitiativeSchedulerResult,
    SchedulableInitiative, InitiativeDependency, InitiativeCategory,
)
from .conftest import assert_provenance_hash, assert_processing_time, timed_block


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_initiatives():
    return [
        SchedulableInitiative(
            initiative_id="INI-001", name="LED Retrofit",
            category=InitiativeCategory.ENERGY_EFFICIENCY,
            annual_reduction_tco2e=Decimal("3000"),
            cost_per_tco2e=Decimal("20"), capex=Decimal("200000"),
            trl=9, implementation_years=1,
        ),
        SchedulableInitiative(
            initiative_id="INI-002", name="HVAC Optimization",
            category=InitiativeCategory.ENERGY_EFFICIENCY,
            annual_reduction_tco2e=Decimal("8000"),
            cost_per_tco2e=Decimal("35"), capex=Decimal("800000"),
            trl=9, implementation_years=2,
        ),
        SchedulableInitiative(
            initiative_id="INI-003", name="Solar PV",
            category=InitiativeCategory.RENEWABLE_ENERGY,
            annual_reduction_tco2e=Decimal("18000"),
            cost_per_tco2e=Decimal("50"), capex=Decimal("3500000"),
            trl=9, implementation_years=2,
        ),
        SchedulableInitiative(
            initiative_id="INI-004", name="Fleet Electrification",
            category=InitiativeCategory.ELECTRIFICATION,
            annual_reduction_tco2e=Decimal("5000"),
            cost_per_tco2e=Decimal("60"), capex=Decimal("1500000"),
            trl=9, implementation_years=2,
        ),
        SchedulableInitiative(
            initiative_id="INI-005", name="Process Heat Pump",
            category=InitiativeCategory.ELECTRIFICATION,
            annual_reduction_tco2e=Decimal("12000"),
            cost_per_tco2e=Decimal("45"), capex=Decimal("2000000"),
            trl=8, implementation_years=2,
        ),
    ]


def _make_input(**kwargs):
    defaults = dict(
        entity_name="GreenCorp Industries",
        initiatives=_make_initiatives(),
        start_year=2024,
        target_year=2035,
        total_budget=Decimal("10000000"),
        annual_budget=Decimal("2000000"),
    )
    defaults.update(kwargs)
    return InitiativeSchedulerInput(**defaults)


class TestInstantiation:
    def test_creates(self):
        assert InitiativeSchedulerEngine() is not None

    def test_version(self):
        assert InitiativeSchedulerEngine().engine_version == "1.0.0"

    def test_has_calculate(self):
        assert hasattr(InitiativeSchedulerEngine(), "calculate")

    def test_has_batch(self):
        assert hasattr(InitiativeSchedulerEngine(), "calculate_batch")

    def test_trl_categories(self):
        cats = InitiativeSchedulerEngine().get_trl_categories()
        assert isinstance(cats, (list, dict))


class TestBasicScheduling:
    def test_basic_result(self):
        r = _run(InitiativeSchedulerEngine().calculate(_make_input()))
        assert r is not None
        assert r.entity_name == "GreenCorp Industries"

    def test_scheduled_initiatives(self):
        r = _run(InitiativeSchedulerEngine().calculate(_make_input()))
        assert len(r.scheduled_initiatives) > 0

    def test_annual_summary(self):
        r = _run(InitiativeSchedulerEngine().calculate(_make_input()))
        assert len(r.annual_summary) > 0

    def test_total_portfolio_reduction(self):
        r = _run(InitiativeSchedulerEngine().calculate(_make_input()))
        assert isinstance(r.total_portfolio_reduction_tco2e, Decimal)
        assert r.total_portfolio_reduction_tco2e > Decimal("0")

    def test_total_portfolio_cost(self):
        r = _run(InitiativeSchedulerEngine().calculate(_make_input()))
        assert isinstance(r.total_portfolio_cost, Decimal)

    def test_budget_feasible(self):
        r = _run(InitiativeSchedulerEngine().calculate(_make_input()))
        assert isinstance(r.budget_feasible, bool)

    def test_provenance(self):
        assert_provenance_hash(_run(InitiativeSchedulerEngine().calculate(_make_input())))

    def test_processing_time(self):
        assert_processing_time(_run(InitiativeSchedulerEngine().calculate(_make_input())))


class TestCriticalPath:
    def test_critical_path_exists(self):
        r = _run(InitiativeSchedulerEngine().calculate(_make_input()))
        assert r.critical_path is not None

    def test_critical_path_length(self):
        r = _run(InitiativeSchedulerEngine().calculate(_make_input()))
        assert r.critical_path.critical_path_length_years >= 0

    def test_earliest_completion(self):
        r = _run(InitiativeSchedulerEngine().calculate(_make_input()))
        assert r.critical_path.earliest_completion_year > 0


class TestScheduledInitiativeFields:
    def test_start_year(self):
        r = _run(InitiativeSchedulerEngine().calculate(_make_input()))
        for si in r.scheduled_initiatives:
            assert si.scheduled_start_year >= 2024

    def test_priority_score(self):
        r = _run(InitiativeSchedulerEngine().calculate(_make_input()))
        for si in r.scheduled_initiatives:
            assert isinstance(si.priority_score, Decimal)

    def test_phases(self):
        r = _run(InitiativeSchedulerEngine().calculate(_make_input()))
        for si in r.scheduled_initiatives:
            assert isinstance(si.phases, list)


class TestAnnualSummary:
    def test_annual_has_year(self):
        r = _run(InitiativeSchedulerEngine().calculate(_make_input()))
        for a in r.annual_summary:
            assert a.year >= 2024

    def test_annual_has_cost(self):
        r = _run(InitiativeSchedulerEngine().calculate(_make_input()))
        for a in r.annual_summary:
            assert isinstance(a.annual_cost, Decimal)

    def test_annual_has_reduction(self):
        r = _run(InitiativeSchedulerEngine().calculate(_make_input()))
        for a in r.annual_summary:
            assert isinstance(a.total_reduction_tco2e, Decimal)


class TestDependencies:
    def test_with_dependencies(self):
        deps = [
            InitiativeDependency(initiative_id="INI-002", depends_on_id="INI-001"),
            InitiativeDependency(initiative_id="INI-005", depends_on_id="INI-002"),
        ]
        r = _run(InitiativeSchedulerEngine().calculate(_make_input(dependencies=deps)))
        assert r is not None


class TestScales:
    @pytest.mark.parametrize("budget", [
        Decimal("1000000"), Decimal("5000000"), Decimal("20000000"),
        Decimal("100000000"),
    ])
    def test_various_budgets(self, budget):
        r = _run(InitiativeSchedulerEngine().calculate(_make_input(
            total_budget=budget, annual_budget=budget / Decimal("10"))))
        assert r is not None

    @pytest.mark.parametrize("entity", ["Corp A", "Corp B", "Corp C"])
    def test_entities(self, entity):
        r = _run(InitiativeSchedulerEngine().calculate(_make_input(entity_name=entity)))
        assert r.entity_name == entity

    @pytest.mark.parametrize("target_year", [2030, 2035, 2040, 2050])
    def test_target_years(self, target_year):
        r = _run(InitiativeSchedulerEngine().calculate(_make_input(target_year=target_year)))
        assert r is not None


class TestDecimalPrecision:
    def test_cost_decimal(self):
        r = _run(InitiativeSchedulerEngine().calculate(_make_input()))
        assert isinstance(r.total_portfolio_cost, Decimal)

    def test_reduction_decimal(self):
        r = _run(InitiativeSchedulerEngine().calculate(_make_input()))
        assert isinstance(r.total_portfolio_reduction_tco2e, Decimal)


class TestRecommendations:
    def test_recommendations(self):
        assert isinstance(_run(InitiativeSchedulerEngine().calculate(_make_input())).recommendations, list)

    def test_warnings(self):
        assert isinstance(_run(InitiativeSchedulerEngine().calculate(_make_input())).warnings, list)

    def test_data_quality(self):
        r = _run(InitiativeSchedulerEngine().calculate(_make_input()))
        assert r.data_quality in ("high", "medium", "low", "estimated")


class TestPerformance:
    def test_under_1_second(self):
        with timed_block(max_ms=1000):
            _run(InitiativeSchedulerEngine().calculate(_make_input()))

    def test_benchmark(self):
        e = InitiativeSchedulerEngine()
        inp = _make_input()
        with timed_block(max_ms=10000):
            for _ in range(100):
                _run(e.calculate(inp))


class TestBatch:
    def test_batch(self):
        inputs = [_make_input(entity_name=f"Corp {i}") for i in range(3)]
        results = _run(InitiativeSchedulerEngine().calculate_batch(inputs))
        assert len(results) == 3


class TestEdgeCases:
    def test_single_initiative(self):
        r = _run(InitiativeSchedulerEngine().calculate(_make_input(
            initiatives=[_make_initiatives()[0]])))
        assert r is not None

    def test_model_dump(self):
        d = _run(InitiativeSchedulerEngine().calculate(_make_input())).model_dump()
        assert isinstance(d, dict)

    def test_sha256(self):
        h = _run(InitiativeSchedulerEngine().calculate(_make_input())).provenance_hash
        assert len(h) == 64
        int(h, 16)

    def test_no_budget(self):
        r = _run(InitiativeSchedulerEngine().calculate(_make_input(
            total_budget=Decimal("0"), annual_budget=Decimal("0"))))
        assert r is not None

    @pytest.mark.parametrize("trl", [5, 6, 7, 8, 9])
    def test_min_trl(self, trl):
        r = _run(InitiativeSchedulerEngine().calculate(_make_input(min_trl_for_deployment=trl)))
        assert r is not None
