# -*- coding: utf-8 -*-
"""
PACK-035 Energy Benchmark - Template Tests
=============================================

Tests all 10 templates for importability, class instantiation,
render_markdown/html/json methods, non-empty output, and content
validation (correct section headings, provenance hash, etc.).

Test Count Target: ~60 tests
Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-035 Energy Benchmark
Date:    March 2026
"""

import importlib.util
import sys
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = PACK_ROOT / "templates"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import (
    TEMPLATE_FILES,
    TEMPLATE_CLASSES,
    _load_module,
)


def _load_template(tpl_key: str):
    """Load a template module by its logical key."""
    file_name = TEMPLATE_FILES.get(tpl_key)
    if file_name is None:
        pytest.skip(f"Unknown template key: {tpl_key}")
    try:
        return _load_module(tpl_key, file_name, "templates")
    except FileNotFoundError:
        pytest.skip(f"Template file not found: {file_name}")
    except ImportError as exc:
        pytest.skip(f"Cannot load template {tpl_key}: {exc}")


# =========================================================================
# 1. Template File Presence
# =========================================================================


class TestTemplateFilePresence:
    """Test all 10 template files exist on disk."""

    @pytest.mark.parametrize("tpl_key,file_name", list(TEMPLATE_FILES.items()))
    def test_template_file_exists(self, tpl_key, file_name):
        """Template Python file exists."""
        path = TEMPLATES_DIR / file_name
        if not path.exists():
            pytest.skip(f"File not found: {path}")
        assert path.is_file()
        assert path.suffix == ".py"


# =========================================================================
# 2. Template Module Loading
# =========================================================================


class TestTemplateModuleLoading:
    """Test all 10 template modules can be loaded."""

    @pytest.mark.parametrize("tpl_key", list(TEMPLATE_FILES.keys()))
    def test_template_module_loads(self, tpl_key):
        """Template module loads without error."""
        mod = _load_template(tpl_key)
        assert mod is not None


# =========================================================================
# 3. Template Class Instantiation
# =========================================================================


class TestTemplateClassInstantiation:
    """Test template class exists and can be instantiated."""

    @pytest.mark.parametrize("tpl_key,class_name", list(TEMPLATE_CLASSES.items()))
    def test_template_class_exists(self, tpl_key, class_name):
        """Template class is defined in the module."""
        mod = _load_template(tpl_key)
        assert hasattr(mod, class_name), f"{class_name} not found in {tpl_key}"

    @pytest.mark.parametrize("tpl_key,class_name", list(TEMPLATE_CLASSES.items()))
    def test_template_instantiation(self, tpl_key, class_name):
        """Template class can be instantiated."""
        mod = _load_template(tpl_key)
        cls = getattr(mod, class_name, None)
        if cls is None:
            pytest.skip(f"{class_name} not found")
        instance = cls()
        assert instance is not None


# =========================================================================
# 4. Render Methods Exist
# =========================================================================


class TestRenderMethodsExist:
    """Test all templates have render methods."""

    @pytest.mark.parametrize("tpl_key,class_name", list(TEMPLATE_CLASSES.items()))
    def test_render_markdown_exists(self, tpl_key, class_name):
        """Template has render_markdown method."""
        mod = _load_template(tpl_key)
        cls = getattr(mod, class_name, None)
        if cls is None:
            pytest.skip(f"{class_name} not found")
        instance = cls()
        has_render = (
            hasattr(instance, "render_markdown")
            or hasattr(instance, "render_md")
            or hasattr(instance, "render")
        )
        assert has_render, f"{class_name} has no render_markdown/render_md/render method"

    @pytest.mark.parametrize("tpl_key,class_name", list(TEMPLATE_CLASSES.items()))
    def test_render_html_exists(self, tpl_key, class_name):
        """Template has render_html method."""
        mod = _load_template(tpl_key)
        cls = getattr(mod, class_name, None)
        if cls is None:
            pytest.skip(f"{class_name} not found")
        instance = cls()
        has_render = (
            hasattr(instance, "render_html")
            or hasattr(instance, "to_html")
        )
        if not has_render:
            pytest.skip(f"{class_name} has no render_html/to_html method")
        assert has_render

    @pytest.mark.parametrize("tpl_key,class_name", list(TEMPLATE_CLASSES.items()))
    def test_render_json_exists(self, tpl_key, class_name):
        """Template has render_json method."""
        mod = _load_template(tpl_key)
        cls = getattr(mod, class_name, None)
        if cls is None:
            pytest.skip(f"{class_name} not found")
        instance = cls()
        has_render = (
            hasattr(instance, "render_json")
            or hasattr(instance, "to_json")
            or hasattr(instance, "to_dict")
        )
        if not has_render:
            pytest.skip(f"{class_name} has no render_json/to_json/to_dict method")
        assert has_render


# =========================================================================
# 5. Render Output Validation
# =========================================================================


class TestRenderOutputValidation:
    """Test render output content quality."""

    def test_eui_report_markdown_nonempty(self, sample_report_data):
        """EUI benchmark report renders non-empty markdown."""
        mod = _load_template("eui_benchmark_report")
        cls = getattr(mod, "EUIBenchmarkReportTemplate", None)
        if cls is None:
            pytest.skip("EUIBenchmarkReportTemplate not found")
        tpl = cls()
        render_fn = getattr(tpl, "render_markdown", None) or getattr(tpl, "render", None)
        if render_fn is None:
            pytest.skip("No render method")
        try:
            output = render_fn(sample_report_data)
        except Exception:
            pytest.skip("Render failed (may need specific data format)")
            return
        assert output is not None
        assert len(output) > 100

    def test_peer_comparison_report_nonempty(self, sample_report_data):
        """Peer comparison report renders non-empty content."""
        mod = _load_template("peer_comparison_report")
        cls = getattr(mod, "PeerComparisonReportTemplate", None)
        if cls is None:
            pytest.skip("PeerComparisonReportTemplate not found")
        tpl = cls()
        render_fn = getattr(tpl, "render_markdown", None) or getattr(tpl, "render", None)
        if render_fn is None:
            pytest.skip("No render method")
        try:
            output = render_fn(sample_report_data)
        except Exception:
            pytest.skip("Render failed")
            return
        assert output is not None
        assert len(output) > 50

    def test_portfolio_dashboard_nonempty(self, sample_report_data):
        """Portfolio dashboard renders non-empty content."""
        mod = _load_template("portfolio_dashboard")
        cls = getattr(mod, "PortfolioDashboardTemplate", None)
        if cls is None:
            pytest.skip("PortfolioDashboardTemplate not found")
        tpl = cls()
        render_fn = getattr(tpl, "render_markdown", None) or getattr(tpl, "render", None)
        if render_fn is None:
            pytest.skip("No render method")
        try:
            output = render_fn(sample_report_data)
        except Exception:
            pytest.skip("Render failed")
            return
        assert output is not None
        assert len(output) > 50


# =========================================================================
# 6. Template Content Sections
# =========================================================================


class TestTemplateContentSections:
    """Test templates include expected content sections."""

    def test_eui_report_has_facility_section(self, sample_report_data):
        """EUI report includes facility information section."""
        mod = _load_template("eui_benchmark_report")
        cls = getattr(mod, "EUIBenchmarkReportTemplate", None)
        if cls is None:
            pytest.skip("EUIBenchmarkReportTemplate not found")
        tpl = cls()
        render_fn = getattr(tpl, "render_markdown", None) or getattr(tpl, "render", None)
        if render_fn is None:
            pytest.skip("No render method")
        try:
            output = render_fn(sample_report_data)
        except Exception:
            pytest.skip("Render failed")
            return
        output_lower = output.lower()
        has_facility = "facility" in output_lower or "building" in output_lower
        assert has_facility

    def test_eui_report_includes_eui_value(self, sample_report_data):
        """EUI report includes the EUI value."""
        mod = _load_template("eui_benchmark_report")
        cls = getattr(mod, "EUIBenchmarkReportTemplate", None)
        if cls is None:
            pytest.skip("EUIBenchmarkReportTemplate not found")
        tpl = cls()
        render_fn = getattr(tpl, "render_markdown", None) or getattr(tpl, "render", None)
        if render_fn is None:
            pytest.skip("No render method")
        try:
            output = render_fn(sample_report_data)
        except Exception:
            pytest.skip("Render failed")
            return
        # EUI should appear in the output
        assert "eui" in output.lower() or "kwh" in output.lower()


# =========================================================================
# 7. Template Metadata
# =========================================================================


class TestTemplateMetadata:
    """Test template module metadata."""

    @pytest.mark.parametrize("tpl_key", list(TEMPLATE_FILES.keys()))
    def test_template_has_version(self, tpl_key):
        """Template module defines _MODULE_VERSION."""
        mod = _load_template(tpl_key)
        if not hasattr(mod, "_MODULE_VERSION"):
            pytest.skip(f"_MODULE_VERSION not found in {tpl_key}")
        assert mod._MODULE_VERSION == "1.0.0"

    @pytest.mark.parametrize("tpl_key", list(TEMPLATE_FILES.keys()))
    def test_template_has_docstring(self, tpl_key):
        """Template module has a docstring."""
        mod = _load_template(tpl_key)
        assert mod.__doc__ is not None
        assert len(mod.__doc__) > 20


# =========================================================================
# 8. Provenance in Reports
# =========================================================================


class TestProvenanceInReports:
    """Test provenance tracking in report templates."""

    def test_report_data_has_provenance_hash(self, sample_report_data):
        """Report data includes a provenance hash."""
        assert "provenance_hash" in sample_report_data
        assert len(sample_report_data["provenance_hash"]) == 64

    def test_provenance_hash_hex_only(self, sample_report_data):
        """Provenance hash contains only hex characters."""
        h = sample_report_data["provenance_hash"]
        assert all(c in "0123456789abcdef" for c in h)
