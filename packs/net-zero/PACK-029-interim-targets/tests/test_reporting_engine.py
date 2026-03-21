# -*- coding: utf-8 -*-
"""Test suite for PACK-029 - Reporting Engine (Engine 10)."""
import asyncio
import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.reporting_engine import (
    ReportingEngine, ReportingInput, ReportingResult,
    ReportType, AssuranceLevel, EmissionsData, TargetData, MilestoneData,
)
from .conftest import assert_provenance_hash, assert_processing_time, timed_block


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_emissions_current():
    return EmissionsData(
        reporting_year=2024,
        scope_1_tco2e=Decimal("107500"),
        scope_2_location_tco2e=Decimal("73100"),
        scope_2_market_tco2e=Decimal("67080"),
        scope_3_tco2e=Decimal("405000"),
        total_tco2e=Decimal("579580"),
        revenue=Decimal("2650000000"),
        revenue_currency="USD",
        intensity_metric="tCO2e/M_USD",
        is_verified=True,
        verification_standard="ISO 14064-3",
    )


def _make_emissions_previous():
    return EmissionsData(
        reporting_year=2023,
        scope_1_tco2e=Decimal("115000"),
        scope_2_location_tco2e=Decimal("78000"),
        scope_2_market_tco2e=Decimal("72000"),
        scope_3_tco2e=Decimal("420000"),
        total_tco2e=Decimal("607000"),
        revenue=Decimal("2500000000"),
        revenue_currency="USD",
        intensity_metric="tCO2e/M_USD",
        is_verified=True,
        verification_standard="ISO 14064-3",
    )


def _make_targets():
    return [
        TargetData(
            target_reference="NT-S12-2030",
            scope="scope_1_2",
            base_year=2019,
            base_year_emissions_tco2e=Decimal("203000"),
            target_year=2030,
            target_reduction_pct=Decimal("42"),
            target_coverage="scope_1_2",
            is_sbti_validated=True,
            is_near_term=True,
            year_set=2022,
        ),
        TargetData(
            target_reference="LT-ALL-2050",
            scope="all_scopes",
            base_year=2019,
            base_year_emissions_tco2e=Decimal("653000"),
            target_year=2050,
            target_reduction_pct=Decimal("90"),
            target_coverage="all_scopes",
            is_sbti_validated=True,
            is_near_term=False,
            year_set=2022,
        ),
    ]


def _make_milestones():
    return [
        MilestoneData(year=2025, target_reduction_pct=Decimal("22"),
                      actual_reduction_pct=Decimal("14"), achieved=False),
        MilestoneData(year=2030, target_reduction_pct=Decimal("42"),
                      actual_reduction_pct=Decimal("0"), achieved=False),
    ]


def _make_input(**kwargs):
    defaults = dict(
        entity_name="GreenCorp Industries",
        report_types=[ReportType.SBTI_PROGRESS],
        emissions_current=_make_emissions_current(),
        emissions_previous=_make_emissions_previous(),
        targets=_make_targets(),
        milestones=_make_milestones(),
        assurance_level=AssuranceLevel.LIMITED,
    )
    defaults.update(kwargs)
    return ReportingInput(**defaults)


class TestInstantiation:
    def test_creates(self):
        assert ReportingEngine() is not None

    def test_version(self):
        assert ReportingEngine().engine_version == "1.0.0"

    def test_has_calculate(self):
        assert hasattr(ReportingEngine(), "calculate")

    def test_has_batch(self):
        assert hasattr(ReportingEngine(), "calculate_batch")

    def test_report_types(self):
        types = ReportingEngine().get_supported_report_types()
        assert isinstance(types, (list, dict))


class TestBasicReporting:
    def test_basic_result(self):
        r = _run(ReportingEngine().calculate(_make_input()))
        assert r is not None
        assert r.entity_name == "GreenCorp Industries"

    def test_report_types_generated(self):
        r = _run(ReportingEngine().calculate(_make_input()))
        assert isinstance(r.report_types_generated, list)
        assert len(r.report_types_generated) > 0

    def test_provenance(self):
        assert_provenance_hash(_run(ReportingEngine().calculate(_make_input())))

    def test_processing_time(self):
        assert_processing_time(_run(ReportingEngine().calculate(_make_input())))


class TestSBTiReport:
    def test_sbti_report_generated(self):
        r = _run(ReportingEngine().calculate(_make_input(
            report_types=[ReportType.SBTI_PROGRESS])))
        assert r.sbti_report is not None

    def test_sbti_report_fields(self):
        r = _run(ReportingEngine().calculate(_make_input(
            report_types=[ReportType.SBTI_PROGRESS])))
        rpt = r.sbti_report
        assert isinstance(rpt.overall_progress_pct, Decimal)
        assert isinstance(rpt.on_track, bool)

    def test_sbti_report_milestones(self):
        r = _run(ReportingEngine().calculate(_make_input(
            report_types=[ReportType.SBTI_PROGRESS])))
        rpt = r.sbti_report
        assert rpt.milestones_total >= 0

    def test_sbti_annual_rate(self):
        r = _run(ReportingEngine().calculate(_make_input(
            report_types=[ReportType.SBTI_PROGRESS])))
        assert isinstance(r.sbti_report.annual_reduction_rate_pct, Decimal)


class TestCDPResponse:
    def test_cdp_report_generated(self):
        r = _run(ReportingEngine().calculate(_make_input(
            report_types=[ReportType.CDP_C4])))
        assert r.cdp_response is not None

    def test_cdp_target_count(self):
        r = _run(ReportingEngine().calculate(_make_input(
            report_types=[ReportType.CDP_C4])))
        assert r.cdp_response.target_count >= 0

    def test_cdp_c41a_rows(self):
        r = _run(ReportingEngine().calculate(_make_input(
            report_types=[ReportType.CDP_C4])))
        assert isinstance(r.cdp_response.c4_1a_rows, list)


class TestTCFDMetrics:
    def test_tcfd_generated(self):
        r = _run(ReportingEngine().calculate(_make_input(
            report_types=[ReportType.TCFD_METRICS])))
        assert r.tcfd_metrics is not None

    def test_tcfd_metrics_dict(self):
        r = _run(ReportingEngine().calculate(_make_input(
            report_types=[ReportType.TCFD_METRICS])))
        assert isinstance(r.tcfd_metrics.metrics, dict)

    def test_tcfd_yoy_change(self):
        r = _run(ReportingEngine().calculate(_make_input(
            report_types=[ReportType.TCFD_METRICS])))
        assert isinstance(r.tcfd_metrics.year_over_year_change, dict)


class TestPublicDisclosure:
    def test_public_generated(self):
        r = _run(ReportingEngine().calculate(_make_input(
            report_types=[ReportType.PUBLIC_DISCLOSURE])))
        assert r.public_disclosure is not None

    def test_public_summary(self):
        r = _run(ReportingEngine().calculate(_make_input(
            report_types=[ReportType.PUBLIC_DISCLOSURE])))
        assert isinstance(r.public_disclosure.executive_summary, str)
        assert len(r.public_disclosure.executive_summary) > 0

    def test_public_highlights(self):
        r = _run(ReportingEngine().calculate(_make_input(
            report_types=[ReportType.PUBLIC_DISCLOSURE])))
        assert isinstance(r.public_disclosure.progress_highlights, list)


class TestAssuranceEvidence:
    def test_assurance_generated(self):
        r = _run(ReportingEngine().calculate(_make_input(
            report_types=[ReportType.ISO14064_ASSURANCE])))
        assert r.assurance_evidence is not None

    def test_assurance_level(self):
        r = _run(ReportingEngine().calculate(_make_input(
            report_types=[ReportType.ISO14064_ASSURANCE])))
        assert r.assurance_evidence.assurance_level != ""

    def test_assurance_methodology(self):
        r = _run(ReportingEngine().calculate(_make_input(
            report_types=[ReportType.ISO14064_ASSURANCE])))
        assert isinstance(r.assurance_evidence.methodology_documentation, list)

    def test_assurance_data_lineage(self):
        r = _run(ReportingEngine().calculate(_make_input(
            report_types=[ReportType.ISO14064_ASSURANCE])))
        assert isinstance(r.assurance_evidence.data_lineage, list)


class TestConsistencyCheck:
    def test_consistency_multi_framework(self):
        r = _run(ReportingEngine().calculate(_make_input(
            report_types=[ReportType.MULTI_FRAMEWORK])))
        assert r.consistency_check is not None

    def test_consistency_status(self):
        r = _run(ReportingEngine().calculate(_make_input(
            report_types=[ReportType.MULTI_FRAMEWORK])))
        assert r.consistency_check.overall_status != ""

    def test_consistency_frameworks(self):
        r = _run(ReportingEngine().calculate(_make_input(
            report_types=[ReportType.MULTI_FRAMEWORK])))
        assert isinstance(r.consistency_check.frameworks_checked, list)


class TestMultipleReportTypes:
    def test_two_report_types(self):
        r = _run(ReportingEngine().calculate(_make_input(
            report_types=[ReportType.SBTI_PROGRESS, ReportType.CDP_C4])))
        assert r.sbti_report is not None
        assert r.cdp_response is not None

    def test_all_report_types(self):
        r = _run(ReportingEngine().calculate(_make_input(
            report_types=[ReportType.SBTI_PROGRESS, ReportType.CDP_C4,
                          ReportType.TCFD_METRICS, ReportType.PUBLIC_DISCLOSURE,
                          ReportType.ISO14064_ASSURANCE, ReportType.MULTI_FRAMEWORK])))
        assert r is not None
        assert len(r.report_types_generated) >= 1


class TestAssuranceLevels:
    @pytest.mark.parametrize("level", [AssuranceLevel.LIMITED, AssuranceLevel.REASONABLE])
    def test_assurance_levels(self, level):
        r = _run(ReportingEngine().calculate(_make_input(
            assurance_level=level,
            report_types=[ReportType.ISO14064_ASSURANCE])))
        assert r is not None


class TestScales:
    @pytest.mark.parametrize("total", [
        Decimal("50000"), Decimal("200000"), Decimal("1000000"),
        Decimal("5000000"), Decimal("50000000"),
    ])
    def test_various_emissions_scales(self, total):
        emissions = EmissionsData(
            reporting_year=2024,
            scope_1_tco2e=total * Decimal("0.4"),
            scope_2_location_tco2e=total * Decimal("0.2"),
            scope_2_market_tco2e=total * Decimal("0.15"),
            scope_3_tco2e=total * Decimal("0.25"),
            total_tco2e=total,
            revenue=Decimal("1000000000"),
            revenue_currency="USD",
            intensity_metric="tCO2e/M_USD",
            is_verified=True,
            verification_standard="ISO 14064-3",
        )
        r = _run(ReportingEngine().calculate(_make_input(
            emissions_current=emissions)))
        assert r is not None

    @pytest.mark.parametrize("entity", ["Corp A", "Corp B", "Corp C"])
    def test_entities(self, entity):
        r = _run(ReportingEngine().calculate(_make_input(entity_name=entity)))
        assert r.entity_name == entity


class TestDecimalPrecision:
    def test_sbti_progress_decimal(self):
        r = _run(ReportingEngine().calculate(_make_input(
            report_types=[ReportType.SBTI_PROGRESS])))
        assert isinstance(r.sbti_report.overall_progress_pct, Decimal)


class TestRecommendations:
    def test_recommendations(self):
        assert isinstance(_run(ReportingEngine().calculate(_make_input())).recommendations, list)

    def test_warnings(self):
        assert isinstance(_run(ReportingEngine().calculate(_make_input())).warnings, list)

    def test_data_quality(self):
        r = _run(ReportingEngine().calculate(_make_input()))
        assert r.data_quality in ("high", "medium", "low", "estimated")


class TestPerformance:
    def test_under_1_second(self):
        with timed_block(max_ms=1000):
            _run(ReportingEngine().calculate(_make_input()))

    def test_benchmark(self):
        e = ReportingEngine()
        inp = _make_input()
        with timed_block(max_ms=10000):
            for _ in range(50):
                _run(e.calculate(inp))


class TestBatch:
    def test_batch(self):
        inputs = [_make_input(entity_name=f"Corp {i}") for i in range(3)]
        results = _run(ReportingEngine().calculate_batch(inputs))
        assert len(results) == 3


class TestEdgeCases:
    def test_single_report_type(self):
        for rt in ReportType:
            r = _run(ReportingEngine().calculate(_make_input(report_types=[rt])))
            assert r is not None

    def test_no_previous_emissions(self):
        r = _run(ReportingEngine().calculate(_make_input(emissions_previous=None)))
        assert r is not None

    def test_model_dump(self):
        d = _run(ReportingEngine().calculate(_make_input())).model_dump()
        assert isinstance(d, dict)

    def test_sha256(self):
        h = _run(ReportingEngine().calculate(_make_input())).provenance_hash
        assert len(h) == 64
        int(h, 16)

    def test_recalculation_flag(self):
        r = _run(ReportingEngine().calculate(_make_input(
            has_recalculation=True,
            recalculation_reason="Methodology update")))
        assert r is not None

    def test_with_sector(self):
        r = _run(ReportingEngine().calculate(_make_input(sector="manufacturing")))
        assert r is not None

    def test_with_country(self):
        r = _run(ReportingEngine().calculate(_make_input(country="US")))
        assert r is not None

    def test_no_milestones(self):
        r = _run(ReportingEngine().calculate(_make_input(milestones=[])))
        assert r is not None

    def test_reasonable_assurance(self):
        r = _run(ReportingEngine().calculate(_make_input(
            assurance_level=AssuranceLevel.REASONABLE,
            report_types=[ReportType.ISO14064_ASSURANCE])))
        assert r is not None
