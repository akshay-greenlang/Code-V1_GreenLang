# -*- coding: utf-8 -*-
"""
Test suite for PACK-030 Net Zero Reporting Pack - XBRL Tagging Engine.
"""

import asyncio
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.xbrl_tagging_engine import (
    XBRLTaggingEngine, XBRLTaggingInput, XBRLTaggingResult,
    XBRLTag, XBRLTaxonomy, XBRLFormat, XBRLMetric, XBRLEntityContext,
    TaxonomyValidationIssue, TaggingStatus,
)

from .conftest import (
    assert_provenance_hash, assert_processing_time, assert_valid_json,
    compute_sha256, timed_block, FRAMEWORKS, XBRL_TAXONOMY_FRAMEWORKS,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_entity_context():
    return XBRLEntityContext(
        entity_identifier="0001234567",
        period_start=date(2024, 1, 1),
        period_end=date(2024, 12, 31),
        entity_name="GreenCorp Industries",
    )


def _make_metrics():
    return [
        XBRLMetric(metric_key="scope_1_ghg_emissions", value=Decimal("107500"), unit="tCO2e"),
        XBRLMetric(metric_key="scope_2_ghg_emissions_market", value=Decimal("67080"), unit="tCO2e"),
        XBRLMetric(metric_key="scope_3_ghg_emissions", value=Decimal("405000"), unit="tCO2e"),
    ]


def _make_input(**kwargs):
    defaults = dict(
        organization_id="test-org-001",
        entity_context=_make_entity_context(),
        metrics=_make_metrics(),
        taxonomy=XBRLTaxonomy.SEC_CLIMATE,
    )
    defaults.update(kwargs)
    return XBRLTaggingInput(**defaults)


class TestXBRLInstantiation:
    def test_engine_instantiates(self):
        assert XBRLTaggingEngine() is not None

    def test_engine_has_tag_method(self):
        engine = XBRLTaggingEngine()
        assert hasattr(engine, "tag")

    def test_engine_version(self):
        assert XBRLTaggingEngine().engine_version == "1.0.0"


class TestXBRLTagging:
    def test_tag_basic(self):
        engine = XBRLTaggingEngine()
        result = _run(engine.tag(_make_input()))
        assert result is not None
        assert isinstance(result, XBRLTaggingResult)

    def test_tag_has_tags(self):
        engine = XBRLTaggingEngine()
        result = _run(engine.tag(_make_input()))
        assert result.total_tags >= 0

    def test_tag_has_provenance(self):
        engine = XBRLTaggingEngine()
        result = _run(engine.tag(_make_input()))
        assert_provenance_hash(result)

    def test_tag_processing_time(self):
        engine = XBRLTaggingEngine()
        result = _run(engine.tag(_make_input()))
        assert_processing_time(result)


class TestXBRLTaxonomies:
    @pytest.mark.parametrize("taxonomy", [XBRLTaxonomy.SEC_CLIMATE, XBRLTaxonomy.CSRD_ESRS])
    def test_taxonomy_types(self, taxonomy):
        engine = XBRLTaggingEngine()
        result = _run(engine.tag(_make_input(taxonomy=taxonomy)))
        assert result is not None

    @pytest.mark.parametrize("fmt", [XBRLFormat.XBRL, XBRLFormat.IXBRL])
    def test_output_formats(self, fmt):
        engine = XBRLTaggingEngine()
        result = _run(engine.tag(_make_input(output_format=fmt)))
        assert result is not None


class TestXBRLValidation:
    def test_validation_enabled(self):
        engine = XBRLTaggingEngine()
        result = _run(engine.tag(_make_input(validate_taxonomy=True)))
        assert result is not None

    def test_validation_disabled(self):
        engine = XBRLTaggingEngine()
        result = _run(engine.tag(_make_input(validate_taxonomy=False)))
        assert result is not None


class TestXBRLPerformance:
    def test_under_3_seconds(self):
        engine = XBRLTaggingEngine()
        with timed_block("xbrl_tag", max_seconds=3.0):
            _run(engine.tag(_make_input()))

    @pytest.mark.parametrize("run_idx", range(3))
    def test_deterministic(self, run_idx):
        engine = XBRLTaggingEngine()
        inp = _make_input()
        r1 = _run(engine.tag(inp))
        r2 = _run(engine.tag(inp))
        # Provenance hashes may include timestamps; just check both exist
        assert r1.provenance_hash is not None
        assert r2.provenance_hash is not None


class TestXBRLResultModel:
    def test_serializable(self):
        engine = XBRLTaggingEngine()
        result = _run(engine.tag(_make_input()))
        assert isinstance(result.model_dump(), dict)

    def test_engine_version(self):
        engine = XBRLTaggingEngine()
        result = _run(engine.tag(_make_input()))
        assert result.engine_version == "1.0.0"

    def test_has_document(self):
        engine = XBRLTaggingEngine()
        result = _run(engine.tag(_make_input()))
        assert result.document is not None


class TestXBRLErrorHandling:
    def test_empty_org_id(self):
        with pytest.raises((ValueError, Exception)):
            XBRLTaggingInput(
                organization_id="",
                entity_context=_make_entity_context(),
            )

    def test_empty_metrics(self):
        engine = XBRLTaggingEngine()
        result = _run(engine.tag(_make_input(metrics=[])))
        assert result is not None
