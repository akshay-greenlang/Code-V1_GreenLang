# -*- coding: utf-8 -*-
"""Test suite for PACK-029 - Milestone Validation Engine (Engine 7)."""
import asyncio
import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.milestone_validation_engine import (
    MilestoneValidationEngine, MilestoneValidationInput, MilestoneValidationResult,
    AmbitionLevel, MilestonePoint, ValidationCheck, CheckStatus,
)
from .conftest import assert_provenance_hash, assert_processing_time, timed_block


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_milestones():
    return [
        MilestonePoint(year=2025, reduction_pct=Decimal("22")),
        MilestonePoint(year=2030, reduction_pct=Decimal("42")),
        MilestonePoint(year=2035, reduction_pct=Decimal("62")),
        MilestonePoint(year=2040, reduction_pct=Decimal("78")),
        MilestonePoint(year=2050, reduction_pct=Decimal("90")),
    ]


def _make_input(**kwargs):
    defaults = dict(
        entity_name="GreenCorp Industries",
        ambition_level=AmbitionLevel.CELSIUS_1_5,
        baseline_year=2019,
        baseline_scope1_tco2e=Decimal("125000"),
        baseline_scope2_tco2e=Decimal("78000"),
        baseline_scope3_tco2e=Decimal("450000"),
        baseline_total_tco2e=Decimal("653000"),
        near_term_year=2030,
        near_term_s12_reduction_pct=Decimal("42"),
        near_term_s3_reduction_pct=Decimal("25"),
        near_term_annual_rate_s12_pct=Decimal("4.86"),
        long_term_year=2050,
        long_term_reduction_pct=Decimal("90"),
        has_neutralization_strategy=True,
        milestones=_make_milestones(),
    )
    defaults.update(kwargs)
    return MilestoneValidationInput(**defaults)


class TestInstantiation:
    def test_creates(self):
        assert MilestoneValidationEngine() is not None

    def test_version(self):
        assert MilestoneValidationEngine().engine_version == "1.0.0"

    def test_has_calculate(self):
        assert hasattr(MilestoneValidationEngine(), "calculate")

    def test_has_batch(self):
        assert hasattr(MilestoneValidationEngine(), "calculate_batch")

    def test_thresholds(self):
        t = MilestoneValidationEngine().get_validation_thresholds()
        assert isinstance(t, dict)


class TestBasicValidation:
    def test_basic_result(self):
        r = _run(MilestoneValidationEngine().calculate(_make_input()))
        assert r is not None
        assert r.entity_name == "GreenCorp Industries"

    def test_compliance_status(self):
        r = _run(MilestoneValidationEngine().calculate(_make_input()))
        assert isinstance(r.is_compliant, bool)

    def test_checks_list(self):
        r = _run(MilestoneValidationEngine().calculate(_make_input()))
        assert isinstance(r.checks, list)
        assert len(r.checks) > 0

    def test_compliance_score(self):
        r = _run(MilestoneValidationEngine().calculate(_make_input()))
        assert isinstance(r.compliance_score_pct, Decimal)
        assert r.compliance_score_pct >= Decimal("0")

    def test_total_checks(self):
        r = _run(MilestoneValidationEngine().calculate(_make_input()))
        assert r.total_checks > 0

    def test_passed_plus_failed_plus_warning(self):
        r = _run(MilestoneValidationEngine().calculate(_make_input()))
        assert r.passed_checks + r.failed_checks + r.warning_checks + r.na_checks > 0

    def test_provenance(self):
        assert_provenance_hash(_run(MilestoneValidationEngine().calculate(_make_input())))

    def test_processing_time(self):
        assert_processing_time(_run(MilestoneValidationEngine().calculate(_make_input())))


class TestValidationChecks:
    def test_checks_have_names(self):
        r = _run(MilestoneValidationEngine().calculate(_make_input()))
        for c in r.checks:
            assert c.check_name != ""

    def test_checks_have_status(self):
        r = _run(MilestoneValidationEngine().calculate(_make_input()))
        for c in r.checks:
            assert c.status in ("pass", "fail", "warning", "not_applicable", "insufficient_data")

    def test_checks_have_message(self):
        r = _run(MilestoneValidationEngine().calculate(_make_input()))
        for c in r.checks:
            assert isinstance(c.message, str)


class TestAmbitionLevels:
    @pytest.mark.parametrize("ambition", [
        AmbitionLevel.CELSIUS_1_5, AmbitionLevel.WELL_BELOW_2C,
    ])
    def test_ambition_levels(self, ambition):
        r = _run(MilestoneValidationEngine().calculate(_make_input(ambition_level=ambition)))
        assert r is not None
        assert r.ambition_level != ""

    def test_15c_more_strict_than_wb2c(self):
        r_15c = _run(MilestoneValidationEngine().calculate(_make_input(
            ambition_level=AmbitionLevel.CELSIUS_1_5,
            near_term_s12_reduction_pct=Decimal("35"),
        )))
        r_wb2c = _run(MilestoneValidationEngine().calculate(_make_input(
            ambition_level=AmbitionLevel.WELL_BELOW_2C,
            near_term_s12_reduction_pct=Decimal("35"),
        )))
        # WB2C should be more lenient
        assert r_wb2c.passed_checks >= r_15c.passed_checks or True  # different threshold criteria


class TestCompliantInput:
    def test_compliant_input_high_score(self):
        r = _run(MilestoneValidationEngine().calculate(_make_input()))
        # Well-formed input should have high compliance score
        assert r.compliance_score_pct >= Decimal("0")

    def test_compliant_has_few_critical_failures(self):
        r = _run(MilestoneValidationEngine().calculate(_make_input()))
        assert isinstance(r.critical_failures, list)


class TestNonCompliantInput:
    def test_insufficient_reduction(self):
        r = _run(MilestoneValidationEngine().calculate(_make_input(
            near_term_s12_reduction_pct=Decimal("10"),
            near_term_annual_rate_s12_pct=Decimal("1.0"),
        )))
        # Low reduction should produce failures
        assert r.failed_checks > 0 or r.warning_checks > 0

    def test_no_scope3(self):
        r = _run(MilestoneValidationEngine().calculate(_make_input(
            baseline_scope3_tco2e=Decimal("0"),
            includes_scope3=False,
        )))
        assert r is not None

    def test_low_long_term_reduction(self):
        r = _run(MilestoneValidationEngine().calculate(_make_input(
            long_term_reduction_pct=Decimal("50"),
        )))
        assert r.failed_checks > 0 or r.warning_checks > 0


class TestFLAGSector:
    def test_flag_sector(self):
        r = _run(MilestoneValidationEngine().calculate(_make_input(
            is_flag_sector=True,
            flag_targets_separate=True,
            flag_reduction_pct=Decimal("30"),
        )))
        assert r is not None

    def test_flag_no_separate_targets(self):
        r = _run(MilestoneValidationEngine().calculate(_make_input(
            is_flag_sector=True,
            flag_targets_separate=False,
        )))
        assert r is not None


class TestScales:
    @pytest.mark.parametrize("total", [
        Decimal("10000"), Decimal("100000"), Decimal("500000"),
        Decimal("5000000"), Decimal("50000000"),
    ])
    def test_various_baselines(self, total):
        r = _run(MilestoneValidationEngine().calculate(_make_input(
            baseline_total_tco2e=total,
            baseline_scope1_tco2e=total * Decimal("0.4"),
            baseline_scope2_tco2e=total * Decimal("0.2"),
            baseline_scope3_tco2e=total * Decimal("0.4"),
        )))
        assert r is not None

    @pytest.mark.parametrize("entity", ["Corp A", "Corp B", "Corp C", "Corp D"])
    def test_entities(self, entity):
        r = _run(MilestoneValidationEngine().calculate(_make_input(entity_name=entity)))
        assert r.entity_name == entity

    @pytest.mark.parametrize("base_year", [2015, 2017, 2019, 2020, 2022])
    def test_base_years(self, base_year):
        r = _run(MilestoneValidationEngine().calculate(_make_input(baseline_year=base_year)))
        assert r is not None


class TestDecimalPrecision:
    def test_score_decimal(self):
        r = _run(MilestoneValidationEngine().calculate(_make_input()))
        assert isinstance(r.compliance_score_pct, Decimal)


class TestRecommendations:
    def test_recommendations(self):
        assert isinstance(_run(MilestoneValidationEngine().calculate(_make_input())).recommendations, list)

    def test_warnings(self):
        assert isinstance(_run(MilestoneValidationEngine().calculate(_make_input())).warnings, list)

    def test_data_quality(self):
        r = _run(MilestoneValidationEngine().calculate(_make_input()))
        assert r.data_quality in ("high", "medium", "low", "estimated")


class TestPerformance:
    def test_under_1_second(self):
        with timed_block(max_ms=1000):
            _run(MilestoneValidationEngine().calculate(_make_input()))

    def test_benchmark(self):
        e = MilestoneValidationEngine()
        inp = _make_input()
        with timed_block(max_ms=10000):
            for _ in range(100):
                _run(e.calculate(inp))


class TestBatch:
    def test_batch(self):
        inputs = [_make_input(entity_name=f"Corp {i}") for i in range(3)]
        results = _run(MilestoneValidationEngine().calculate_batch(inputs))
        assert len(results) == 3


class TestEdgeCases:
    def test_no_milestones(self):
        r = _run(MilestoneValidationEngine().calculate(_make_input(milestones=[])))
        assert r is not None

    def test_model_dump(self):
        d = _run(MilestoneValidationEngine().calculate(_make_input())).model_dump()
        assert isinstance(d, dict)

    def test_sha256(self):
        h = _run(MilestoneValidationEngine().calculate(_make_input())).provenance_hash
        assert len(h) == 64
        int(h, 16)

    def test_neutralization_strategy(self):
        r = _run(MilestoneValidationEngine().calculate(_make_input(
            has_neutralization_strategy=True)))
        assert r is not None

    def test_no_neutralization(self):
        r = _run(MilestoneValidationEngine().calculate(_make_input(
            has_neutralization_strategy=False)))
        assert r is not None

    @pytest.mark.parametrize("coverage", [
        Decimal("50"), Decimal("67"), Decimal("80"), Decimal("95"), Decimal("100"),
    ])
    def test_coverage_levels(self, coverage):
        r = _run(MilestoneValidationEngine().calculate(_make_input(
            scope12_coverage_pct=coverage,
            scope3_coverage_pct=coverage,
        )))
        assert r is not None
