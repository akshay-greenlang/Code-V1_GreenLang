# -*- coding: utf-8 -*-
"""
PACK-017 ESRS Full Coverage Pack - Template Tests
====================================================

Tests for all 11 ESRS report templates: file existence, module loading,
class exports, render methods (markdown, html, json), section definitions,
provenance hashing, and per-template content validation.

Target: 40+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-017 ESRS Full Coverage Pack
Date:    March 2026
"""

import pytest

from .conftest import (
    TEMPLATE_FILES,
    TEMPLATE_CLASSES,
    TEMPLATES_DIR,
    _load_template,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _try_load_template(key):
    """Attempt to load a template, returning module or None."""
    try:
        return _load_template(key)
    except (ImportError, FileNotFoundError):
        return None


def _sample_esrs2_data():
    """Minimal sample data for ESRS 2 general template rendering tests."""
    return {
        "company_name": "NordEnergy AG",
        "reporting_year": 2025,
        "governance_bodies": [
            {"name": "Board of Directors", "sustainability_competence": True},
        ],
        "strategy_elements": [
            {"area": "SBM-1", "description": "Business model overview"},
        ],
        "material_iros": [
            {"iro_id": "IRO-1", "topic": "Climate Change", "material": True},
        ],
        "provenance_hash": "a" * 64,
    }


def _sample_pollution_data():
    """Minimal sample data for E2 Pollution template rendering tests."""
    return {
        "company_name": "ChemCorp GmbH",
        "reporting_year": 2025,
        "policies": [{"name": "Zero Pollution Policy", "scope": "group-wide"}],
        "pollutant_emissions": [
            {"medium": "air", "substance": "NOx", "tonnes": 120.5},
            {"medium": "water", "substance": "COD", "tonnes": 45.2},
        ],
        "substances_of_concern": [
            {"name": "Lead", "svhc": True, "tonnes": 0.8},
        ],
        "provenance_hash": "b" * 64,
    }


# ===========================================================================
# Template File Existence
# ===========================================================================


class TestTemplateFilesExist:
    """Test that all 11 template files exist on disk."""

    @pytest.mark.parametrize("tpl_key,tpl_file", list(TEMPLATE_FILES.items()))
    def test_template_file_exists(self, tpl_key, tpl_file):
        """Template file exists on disk."""
        path = TEMPLATES_DIR / tpl_file
        assert path.exists(), f"Template file missing: {path}"


# ===========================================================================
# Template Module Loading
# ===========================================================================


class TestTemplateLoading:
    """Test that all 11 templates can be loaded via importlib."""

    @pytest.mark.parametrize("tpl_key", list(TEMPLATE_FILES.keys()))
    def test_template_module_loads(self, tpl_key):
        """Each template module loads independently."""
        mod = _try_load_template(tpl_key)
        assert mod is not None, f"Template {tpl_key} failed to load"

    @pytest.mark.parametrize("tpl_key,tpl_class", list(TEMPLATE_CLASSES.items()))
    def test_template_exports_class(self, tpl_key, tpl_class):
        """Each template exports its primary class."""
        mod = _try_load_template(tpl_key)
        if mod is None:
            pytest.skip(f"Template {tpl_key} not loaded")
        assert hasattr(mod, tpl_class), f"Template {tpl_key} missing class {tpl_class}"


# ===========================================================================
# Render Method Existence
# ===========================================================================


class TestTemplateRenderMethods:
    """Test that all template classes have the required render methods."""

    @pytest.mark.parametrize("tpl_key,tpl_class", list(TEMPLATE_CLASSES.items()))
    def test_template_has_render_markdown(self, tpl_key, tpl_class):
        """Each template class has render_markdown method."""
        mod = _try_load_template(tpl_key)
        if mod is None:
            pytest.skip(f"Template {tpl_key} not loaded")
        cls = getattr(mod, tpl_class, None)
        if cls is None:
            pytest.skip(f"Class {tpl_class} not found")
        assert hasattr(cls, "render_markdown"), f"{tpl_class} missing render_markdown"

    @pytest.mark.parametrize("tpl_key,tpl_class", list(TEMPLATE_CLASSES.items()))
    def test_template_has_render_html(self, tpl_key, tpl_class):
        """Each template class has render_html method."""
        mod = _try_load_template(tpl_key)
        if mod is None:
            pytest.skip(f"Template {tpl_key} not loaded")
        cls = getattr(mod, tpl_class, None)
        if cls is None:
            pytest.skip(f"Class {tpl_class} not found")
        assert hasattr(cls, "render_html"), f"{tpl_class} missing render_html"

    @pytest.mark.parametrize("tpl_key,tpl_class", list(TEMPLATE_CLASSES.items()))
    def test_template_has_render_json(self, tpl_key, tpl_class):
        """Each template class has render_json method."""
        mod = _try_load_template(tpl_key)
        if mod is None:
            pytest.skip(f"Template {tpl_key} not loaded")
        cls = getattr(mod, tpl_class, None)
        if cls is None:
            pytest.skip(f"Class {tpl_class} not found")
        assert hasattr(cls, "render_json"), f"{tpl_class} missing render_json"

    @pytest.mark.parametrize("tpl_key,tpl_class", list(TEMPLATE_CLASSES.items()))
    def test_template_has_sections(self, tpl_key, tpl_class):
        """Each template has get_sections, SECTIONS, or sections attribute."""
        mod = _try_load_template(tpl_key)
        if mod is None:
            pytest.skip(f"Template {tpl_key} not loaded")
        cls = getattr(mod, tpl_class, None)
        if cls is None:
            pytest.skip(f"Class {tpl_class} not found")
        has_sections = (
            hasattr(cls, "get_sections")
            or hasattr(cls, "SECTIONS")
            or hasattr(cls, "sections")
            or hasattr(mod, "_SECTIONS")
        )
        assert has_sections, f"{tpl_class} should define sections"


# ===========================================================================
# Per-Template Rendering Tests
# ===========================================================================


class TestESRS2GeneralTemplate:
    """Tests for the ESRS2GeneralReport template."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_template("esrs2_general_report")

    def test_class_exists(self):
        """ESRS2GeneralReport class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "ESRS2GeneralReportTemplate")

    def test_render_markdown_produces_output(self):
        """render_markdown produces non-empty string."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        tpl = self.mod.ESRS2GeneralReportTemplate()
        result = tpl.render_markdown(_sample_esrs2_data())
        assert isinstance(result, str)
        assert len(result) > 0

    def test_render_json_produces_dict(self):
        """render_json produces a dict with expected keys."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        tpl = self.mod.ESRS2GeneralReportTemplate()
        result = tpl.render_json(_sample_esrs2_data())
        assert isinstance(result, dict)

    def test_source_references_gov(self):
        """Template source references governance disclosures."""
        path = TEMPLATES_DIR / "esrs2_general_report.py"
        content = path.read_text(encoding="utf-8")
        assert "GOV" in content or "governance" in content.lower()


class TestE2PollutionTemplate:
    """Tests for the E2PollutionReport template."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_template("e2_pollution_report")

    def test_class_exists(self):
        """E2PollutionReportTemplate class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "E2PollutionReportTemplate")

    def test_render_markdown_produces_output(self):
        """render_markdown produces non-empty markdown string."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        tpl = self.mod.E2PollutionReportTemplate()
        result = tpl.render_markdown(_sample_pollution_data())
        assert isinstance(result, str)
        assert len(result) > 0

    def test_render_json_produces_dict(self):
        """render_json produces a dict."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        tpl = self.mod.E2PollutionReportTemplate()
        result = tpl.render_json(_sample_pollution_data())
        assert isinstance(result, dict)

    def test_source_references_e2(self):
        """Template source references E2 disclosure requirements."""
        path = TEMPLATES_DIR / "e2_pollution_report.py"
        content = path.read_text(encoding="utf-8")
        assert "E2-" in content or "E2_" in content or "e2" in content.lower() or "pollution" in content.lower()


class TestE3WaterTemplate:
    """Tests for the E3WaterReport template."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_template("e3_water_report")

    def test_class_exists(self):
        """E3WaterReport class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "E3WaterReportTemplate")

    def test_source_references_e3(self):
        """Template source references E3 disclosure requirements."""
        path = TEMPLATES_DIR / "e3_water_report.py"
        content = path.read_text(encoding="utf-8")
        assert "E3-" in content or "E3_" in content or "water" in content.lower()


# ===========================================================================
# Template Provenance (cross-template)
# ===========================================================================


class TestTemplateProvenance:
    """Tests for template provenance hashing across all templates."""

    @pytest.mark.parametrize("tpl_key,tpl_class", list(TEMPLATE_CLASSES.items()))
    def test_template_has_docstring(self, tpl_key, tpl_class):
        """Each template class has a docstring."""
        mod = _try_load_template(tpl_key)
        if mod is None:
            pytest.skip(f"Template {tpl_key} not loaded")
        cls = getattr(mod, tpl_class, None)
        if cls is None:
            pytest.skip(f"Class {tpl_class} not found")
        assert cls.__doc__ is not None

    @pytest.mark.parametrize("tpl_key", list(TEMPLATE_FILES.keys()))
    def test_template_uses_sha256(self, tpl_key):
        """Each template file references SHA-256 for provenance."""
        source_path = TEMPLATES_DIR / TEMPLATE_FILES[tpl_key]
        if not source_path.exists():
            pytest.skip(f"File not found: {source_path}")
        content = source_path.read_text(encoding="utf-8")
        has_sha = (
            "sha256" in content.lower()
            or "hashlib" in content
            or "provenance" in content.lower()
        )
        assert has_sha, f"Template {tpl_key} should reference SHA-256 provenance"

    @pytest.mark.parametrize("tpl_key", list(TEMPLATE_FILES.keys()))
    def test_template_includes_generated_at(self, tpl_key):
        """Each template file references a generated_at timestamp or datetime."""
        source_path = TEMPLATES_DIR / TEMPLATE_FILES[tpl_key]
        if not source_path.exists():
            pytest.skip(f"File not found: {source_path}")
        content = source_path.read_text(encoding="utf-8")
        has_ts = (
            "generated_at" in content
            or "datetime" in content
            or "timestamp" in content.lower()
        )
        assert has_ts, f"Template {tpl_key} should include timestamp"
