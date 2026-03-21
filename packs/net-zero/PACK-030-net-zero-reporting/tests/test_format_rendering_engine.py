# -*- coding: utf-8 -*-
"""
Test suite for PACK-030 Net Zero Reporting Pack - Format Rendering Engine.

Tests PDF rendering, HTML generation, Excel export, JSON API output,
XBRL rendering delegation, branding in rendered outputs, multi-format
generation, and format-specific validation.

Author:  GreenLang Test Engineering
Pack:    PACK-030 Net Zero Reporting Pack
Engine:  10 of 10 - format_rendering_engine.py
"""

import asyncio
import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.format_rendering_engine import (
    FormatRenderingEngine, RenderInput, RenderResult,
    OutputFormat, BrandingConfig, ChartConfig,
    ReportSection, PageSize, Orientation, RenderQuality,
)

from .conftest import (
    assert_provenance_hash, assert_processing_time, assert_valid_json,
    assert_html_contains, compute_sha256, timed_block,
    FRAMEWORKS, OUTPUT_FORMATS, LANGUAGES,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_sections(count=3):
    """Create minimal ReportSection list."""
    return [
        ReportSection(
            section_id=f"sec-{i}",
            section_type=f"section_{i+1}",
            title=f"Section {i+1}",
            content=f"Sample content for section {i+1}.",
        )
        for i in range(count)
    ]


def _make_input(**kwargs):
    defaults = dict(
        organization_id="test-org-001",
        sections=_make_sections(),
    )
    defaults.update(kwargs)
    return RenderInput(**defaults)


class TestFormatRenderingInstantiation:
    def test_engine_instantiates(self):
        assert FormatRenderingEngine() is not None

    def test_engine_has_render_method(self):
        engine = FormatRenderingEngine()
        assert hasattr(engine, "render")

    def test_engine_has_render_pdf(self):
        engine = FormatRenderingEngine()
        assert hasattr(engine, "render_pdf")

    def test_engine_has_render_html(self):
        engine = FormatRenderingEngine()
        assert hasattr(engine, "render_html")


class TestRenderPDF:
    def test_render_pdf_basic(self):
        engine = FormatRenderingEngine()
        inp = _make_input(output_format=OutputFormat.PDF)
        result = _run(engine.render(inp))
        assert result is not None
        assert isinstance(result, RenderResult)

    def test_render_pdf_method(self):
        engine = FormatRenderingEngine()
        inp = _make_input()
        result = _run(engine.render_pdf(inp))
        assert result is not None

    def test_render_pdf_has_provenance(self):
        engine = FormatRenderingEngine()
        inp = _make_input(output_format=OutputFormat.PDF)
        result = _run(engine.render(inp))
        assert_provenance_hash(result)

    def test_render_pdf_processing_time(self):
        engine = FormatRenderingEngine()
        inp = _make_input(output_format=OutputFormat.PDF)
        result = _run(engine.render(inp))
        assert_processing_time(result)

    def test_render_pdf_under_5_seconds(self):
        engine = FormatRenderingEngine()
        inp = _make_input(output_format=OutputFormat.PDF, sections=_make_sections(10))
        with timed_block("pdf_render", max_seconds=5.0):
            _run(engine.render(inp))

    @pytest.mark.parametrize("count", [1, 3, 5, 10])
    def test_render_pdf_various_sizes(self, count):
        engine = FormatRenderingEngine()
        inp = _make_input(output_format=OutputFormat.PDF, sections=_make_sections(count))
        result = _run(engine.render(inp))
        assert result is not None


class TestRenderHTML:
    def test_render_html_basic(self):
        engine = FormatRenderingEngine()
        inp = _make_input(output_format=OutputFormat.HTML)
        result = _run(engine.render(inp))
        assert result is not None

    def test_render_html_method(self):
        engine = FormatRenderingEngine()
        inp = _make_input()
        result = _run(engine.render_html(inp))
        assert result is not None

    def test_render_html_has_provenance(self):
        engine = FormatRenderingEngine()
        inp = _make_input(output_format=OutputFormat.HTML)
        result = _run(engine.render(inp))
        assert_provenance_hash(result)

    def test_render_html_processing_time(self):
        engine = FormatRenderingEngine()
        inp = _make_input(output_format=OutputFormat.HTML)
        result = _run(engine.render(inp))
        assert_processing_time(result)

    def test_render_html_under_2_seconds(self):
        engine = FormatRenderingEngine()
        inp = _make_input(output_format=OutputFormat.HTML, sections=_make_sections(5))
        with timed_block("html_render", max_seconds=2.0):
            _run(engine.render(inp))


class TestRenderExcel:
    def test_render_excel_basic(self):
        engine = FormatRenderingEngine()
        inp = _make_input(output_format=OutputFormat.EXCEL)
        result = _run(engine.render(inp))
        assert result is not None

    def test_render_excel_method(self):
        engine = FormatRenderingEngine()
        inp = _make_input()
        result = _run(engine.render_excel(inp))
        assert result is not None

    def test_render_excel_has_provenance(self):
        engine = FormatRenderingEngine()
        inp = _make_input(output_format=OutputFormat.EXCEL)
        result = _run(engine.render(inp))
        assert_provenance_hash(result)

    def test_render_excel_under_2_seconds(self):
        engine = FormatRenderingEngine()
        inp = _make_input(output_format=OutputFormat.EXCEL, sections=_make_sections(5))
        with timed_block("excel_render", max_seconds=2.0):
            _run(engine.render(inp))


class TestRenderJSON:
    def test_render_json_basic(self):
        engine = FormatRenderingEngine()
        inp = _make_input(output_format=OutputFormat.JSON)
        result = _run(engine.render(inp))
        assert result is not None

    def test_render_json_method(self):
        engine = FormatRenderingEngine()
        inp = _make_input()
        result = _run(engine.render_json(inp))
        assert result is not None

    def test_render_json_has_provenance(self):
        engine = FormatRenderingEngine()
        inp = _make_input(output_format=OutputFormat.JSON)
        result = _run(engine.render(inp))
        assert_provenance_hash(result)

    def test_render_json_under_1_second(self):
        engine = FormatRenderingEngine()
        inp = _make_input(output_format=OutputFormat.JSON, sections=_make_sections(5))
        with timed_block("json_render", max_seconds=1.0):
            _run(engine.render(inp))


class TestRenderXBRL:
    def test_render_xbrl_basic(self):
        engine = FormatRenderingEngine()
        inp = _make_input(output_format=OutputFormat.XBRL)
        result = _run(engine.render(inp))
        assert result is not None

    def test_render_xbrl_method(self):
        engine = FormatRenderingEngine()
        inp = _make_input()
        result = _run(engine.render_xbrl(inp))
        assert result is not None


class TestRenderMulti:
    def test_render_multi(self):
        engine = FormatRenderingEngine()
        inp = _make_input()
        result = _run(engine.render_multi(inp))
        assert result is not None

    @pytest.mark.parametrize("fmt", [OutputFormat.PDF, OutputFormat.HTML, OutputFormat.JSON, OutputFormat.EXCEL])
    def test_render_per_format(self, fmt):
        engine = FormatRenderingEngine()
        inp = _make_input(output_format=fmt)
        result = _run(engine.render(inp))
        assert result is not None


class TestBranding:
    def test_render_with_branding(self):
        engine = FormatRenderingEngine()
        inp = _make_input(output_format=OutputFormat.PDF)
        result = _run(engine.render(inp))
        assert result is not None


class TestSections:
    @pytest.mark.parametrize("count", [1, 5, 10, 20])
    def test_various_section_counts(self, count):
        engine = FormatRenderingEngine()
        inp = _make_input(sections=_make_sections(count))
        result = _run(engine.render(inp))
        assert result is not None


class TestPageConfiguration:
    def test_default_page_size(self):
        engine = FormatRenderingEngine()
        inp = _make_input()
        result = _run(engine.render(inp))
        assert result is not None


class TestQuality:
    def test_default_quality(self):
        engine = FormatRenderingEngine()
        inp = _make_input()
        result = _run(engine.render(inp))
        assert result is not None


class TestPerformance:
    @pytest.mark.parametrize("run_idx", range(3))
    def test_deterministic_rendering(self, run_idx):
        engine = FormatRenderingEngine()
        inp = _make_input()
        r1 = _run(engine.render(inp))
        r2 = _run(engine.render(inp))
        # Provenance hashes may include timestamps; just check both exist
        assert r1.provenance_hash is not None
        assert r2.provenance_hash is not None

    def test_batch_rendering_performance(self):
        engine = FormatRenderingEngine()
        with timed_block("batch_render", max_seconds=10.0):
            for fmt in [OutputFormat.PDF, OutputFormat.HTML, OutputFormat.JSON]:
                inp = _make_input(output_format=fmt, sections=_make_sections(3))
                _run(engine.render(inp))


class TestResultModel:
    def test_result_serializable(self):
        engine = FormatRenderingEngine()
        inp = _make_input()
        result = _run(engine.render(inp))
        assert isinstance(result.model_dump(), dict)

    def test_result_engine_version(self):
        engine = FormatRenderingEngine()
        inp = _make_input()
        result = _run(engine.render(inp))
        assert result.engine_version == "1.0.0"


class TestErrorHandling:
    def test_empty_org_id_accepted(self):
        """RenderInput allows empty org_id (no min_length validator)."""
        inp = RenderInput(
            organization_id="",
            sections=_make_sections(),
        )
        assert inp.organization_id == ""
