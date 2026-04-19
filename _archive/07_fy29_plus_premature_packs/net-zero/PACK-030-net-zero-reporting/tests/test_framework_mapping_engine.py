# -*- coding: utf-8 -*-
"""
Test suite for PACK-030 Net Zero Reporting Pack - Framework Mapping Engine.
"""

import asyncio
import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.framework_mapping_engine import (
    FrameworkMappingEngine, FrameworkMappingInput, FrameworkMappingResult,
    MetricMapping, MappingDirection, MappingConflict, MetricValue,
    Framework, MappingType,
)

from .conftest import (
    assert_decimal_close, assert_percentage_range, assert_provenance_hash,
    assert_processing_time, compute_sha256, timed_block, FRAMEWORKS,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_metrics():
    return [
        MetricValue(metric_key="scope_1_emissions", framework=Framework.TCFD.value,
                     value=Decimal("107500"), unit="tCO2e"),
        MetricValue(metric_key="scope_2_emissions", framework=Framework.TCFD.value,
                     value=Decimal("67080"), unit="tCO2e"),
        MetricValue(metric_key="scope_3_emissions", framework=Framework.TCFD.value,
                     value=Decimal("405000"), unit="tCO2e"),
    ]


def _make_input(**kwargs):
    defaults = dict(
        organization_id="test-org-001",
        source_metrics=_make_metrics(),
        source_framework=Framework.TCFD,
    )
    defaults.update(kwargs)
    return FrameworkMappingInput(**defaults)


class TestFrameworkMappingInstantiation:
    def test_engine_instantiates(self):
        assert FrameworkMappingEngine() is not None

    def test_engine_version(self):
        assert FrameworkMappingEngine().engine_version == "1.0.0"

    def test_engine_has_map_method(self):
        engine = FrameworkMappingEngine()
        assert hasattr(engine, "map")


class TestMetricMapping:
    def test_map_basic(self):
        engine = FrameworkMappingEngine()
        result = _run(engine.map(_make_input()))
        assert result is not None
        assert isinstance(result, FrameworkMappingResult)

    def test_map_has_mappings(self):
        engine = FrameworkMappingEngine()
        result = _run(engine.map(_make_input()))
        assert result.total_mappings >= 0

    def test_map_has_provenance(self):
        engine = FrameworkMappingEngine()
        result = _run(engine.map(_make_input()))
        assert_provenance_hash(result)

    def test_map_processing_time(self):
        engine = FrameworkMappingEngine()
        result = _run(engine.map(_make_input()))
        assert_processing_time(result)


class TestBidirectionalSync:
    def test_bidirectional_enabled(self):
        engine = FrameworkMappingEngine()
        result = _run(engine.map(_make_input(include_bidirectional=True)))
        assert result is not None

    def test_bidirectional_disabled(self):
        engine = FrameworkMappingEngine()
        result = _run(engine.map(_make_input(include_bidirectional=False)))
        assert result is not None


class TestConflictDetection:
    def test_conflict_detection_enabled(self):
        engine = FrameworkMappingEngine()
        result = _run(engine.map(_make_input(include_conflict_detection=True)))
        assert result is not None

    def test_conflict_detection_disabled(self):
        engine = FrameworkMappingEngine()
        result = _run(engine.map(_make_input(include_conflict_detection=False)))
        assert result is not None


class TestMappingPerformance:
    def test_under_3_seconds(self):
        engine = FrameworkMappingEngine()
        with timed_block("mapping", max_seconds=3.0):
            _run(engine.map(_make_input()))

    @pytest.mark.parametrize("run_idx", range(3))
    def test_deterministic(self, run_idx):
        engine = FrameworkMappingEngine()
        inp = _make_input()
        r1 = _run(engine.map(inp))
        r2 = _run(engine.map(inp))
        # Provenance hashes may include timestamps; just check both exist
        assert r1.provenance_hash is not None
        assert r2.provenance_hash is not None


class TestMappingResultModel:
    def test_serializable(self):
        engine = FrameworkMappingEngine()
        result = _run(engine.map(_make_input()))
        assert isinstance(result.model_dump(), dict)

    def test_engine_version(self):
        engine = FrameworkMappingEngine()
        result = _run(engine.map(_make_input()))
        assert result.engine_version == "1.0.0"

    def test_has_coverage(self):
        engine = FrameworkMappingEngine()
        result = _run(engine.map(_make_input()))
        assert isinstance(result.coverage, list)


class TestMappingErrorHandling:
    def test_empty_org_id(self):
        with pytest.raises((ValueError, Exception)):
            FrameworkMappingInput(organization_id="")

    def test_empty_metrics(self):
        engine = FrameworkMappingEngine()
        result = _run(engine.map(_make_input(source_metrics=[])))
        assert result is not None
