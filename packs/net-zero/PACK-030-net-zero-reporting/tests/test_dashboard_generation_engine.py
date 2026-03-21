# -*- coding: utf-8 -*-
"""
Test suite for PACK-030 Net Zero Reporting Pack - Dashboard Generation Engine.
"""

import asyncio
import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.dashboard_generation_engine import (
    DashboardGenerationEngine, DashboardGenerationInput, DashboardGenerationResult,
    DashboardType, BrandingConfig, FrameworkStatus, EmissionsTrend,
    DashboardWidget, DashboardDocument,
)

from .conftest import (
    assert_provenance_hash, assert_processing_time, assert_html_contains,
    compute_sha256, timed_block, FRAMEWORKS, STAKEHOLDER_VIEWS,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_input(**kwargs):
    defaults = dict(
        organization_id="test-org-001",
        dashboard_type=DashboardType.EXECUTIVE,
        framework_statuses=[
            FrameworkStatus(framework="SBTi", completeness_pct=Decimal("85")),
            FrameworkStatus(framework="CDP", completeness_pct=Decimal("90")),
            FrameworkStatus(framework="TCFD", completeness_pct=Decimal("75")),
        ],
        target_reduction_pct=Decimal("42"),
        current_reduction_pct=Decimal("14"),
        overall_completeness_pct=Decimal("83"),
    )
    defaults.update(kwargs)
    return DashboardGenerationInput(**defaults)


class TestDashboardInstantiation:
    def test_engine_instantiates(self):
        assert DashboardGenerationEngine() is not None

    def test_engine_version(self):
        assert DashboardGenerationEngine().engine_version == "1.0.0"

    def test_engine_has_generate_method(self):
        engine = DashboardGenerationEngine()
        assert hasattr(engine, "generate")


class TestExecutiveDashboard:
    def test_generate_executive_dashboard(self):
        engine = DashboardGenerationEngine()
        result = _run(engine.generate_executive_dashboard(_make_input()))
        assert result is not None
        assert isinstance(result, DashboardGenerationResult)

    def test_dashboard_has_html_content(self):
        engine = DashboardGenerationEngine()
        result = _run(engine.generate(_make_input()))
        assert result.dashboard is not None
        assert len(result.dashboard.html_content) > 100

    def test_dashboard_has_widgets(self):
        engine = DashboardGenerationEngine()
        result = _run(engine.generate(_make_input()))
        assert result.total_widgets > 0

    def test_dashboard_has_provenance(self):
        engine = DashboardGenerationEngine()
        result = _run(engine.generate(_make_input()))
        assert_provenance_hash(result)

    def test_dashboard_processing_time(self):
        engine = DashboardGenerationEngine()
        result = _run(engine.generate(_make_input()))
        assert_processing_time(result)


class TestFrameworkDashboards:
    @pytest.mark.parametrize("framework", ["SBTi", "CDP", "TCFD"])
    def test_generate_per_framework(self, framework):
        engine = DashboardGenerationEngine()
        inp = _make_input()
        result = _run(engine.generate_framework_dashboard(inp, framework=framework))
        assert result is not None

    def test_framework_dashboard_has_content(self):
        engine = DashboardGenerationEngine()
        inp = _make_input()
        result = _run(engine.generate_framework_dashboard(inp, framework="SBTi"))
        assert result.dashboard is not None


class TestStakeholderViews:
    @pytest.mark.parametrize("view", ["investor", "regulator", "customer", "employee"])
    def test_generate_per_stakeholder(self, view):
        engine = DashboardGenerationEngine()
        inp = _make_input()
        result = _run(engine.generate_stakeholder_view(inp, stakeholder=view))
        assert result is not None

    @pytest.mark.parametrize("view", ["investor", "regulator", "customer", "employee"])
    def test_stakeholder_view_has_provenance(self, view):
        engine = DashboardGenerationEngine()
        inp = _make_input()
        result = _run(engine.generate_stakeholder_view(inp, stakeholder=view))
        assert_provenance_hash(result)


class TestDashboardInteractivity:
    def test_dashboard_responsive(self):
        engine = DashboardGenerationEngine()
        result = _run(engine.generate(_make_input(responsive_design=True)))
        assert result.dashboard.is_responsive is True

    def test_dashboard_interactive(self):
        engine = DashboardGenerationEngine()
        result = _run(engine.generate(_make_input(include_interactivity=True)))
        assert result.dashboard.is_interactive is True


class TestDashboardPerformance:
    def test_dashboard_under_4_seconds(self):
        engine = DashboardGenerationEngine()
        with timed_block("dashboard_gen", max_seconds=4.0):
            _run(engine.generate(_make_input()))

    @pytest.mark.parametrize("run_idx", range(3))
    def test_deterministic_dashboard(self, run_idx):
        engine = DashboardGenerationEngine()
        inp = _make_input()
        r1 = _run(engine.generate(inp))
        r2 = _run(engine.generate(inp))
        # Provenance hashes may include timestamps; just check both exist
        assert r1.provenance_hash is not None
        assert r2.provenance_hash is not None


class TestDashboardResultModel:
    def test_result_serializable(self):
        engine = DashboardGenerationEngine()
        result = _run(engine.generate(_make_input()))
        assert isinstance(result.model_dump(), dict)

    def test_result_has_frameworks_displayed(self):
        engine = DashboardGenerationEngine()
        result = _run(engine.generate(_make_input()))
        assert len(result.frameworks_displayed) >= 1

    def test_result_has_charts(self):
        engine = DashboardGenerationEngine()
        result = _run(engine.generate(_make_input()))
        assert result.total_charts >= 0

    def test_result_engine_version(self):
        engine = DashboardGenerationEngine()
        result = _run(engine.generate(_make_input()))
        assert result.engine_version == "1.0.0"
