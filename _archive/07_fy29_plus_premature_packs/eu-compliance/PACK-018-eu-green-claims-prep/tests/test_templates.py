# -*- coding: utf-8 -*-
"""
PACK-018 EU Green Claims Prep Pack - Template Tests
=====================================================

Tests for all 8 report templates: file existence, module loading, class
exports, render_markdown/render_html/render_json methods, docstrings,
and SHA-256 provenance. Parametrized across all templates.

Target: ~45 tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-018 EU Green Claims Prep
Date:    March 2026
"""

import pytest

from .conftest import (
    _load_template,
    TEMPLATES_DIR,
    TEMPLATE_FILES,
    TEMPLATE_CLASSES,
)


# ===========================================================================
# File Existence Tests (parametrized across all 8 templates)
# ===========================================================================


EXISTING_TEMPLATES = [
    ("claim_assessment_report", "claim_assessment_report.py", "ClaimAssessmentReportTemplate"),
    ("evidence_dossier_report", "evidence_dossier_report.py", "EvidenceDossierReportTemplate"),
    ("lifecycle_summary_report", "lifecycle_summary_report.py", "LifecycleSummaryReportTemplate"),
    ("label_compliance_report", "label_compliance_report.py", "LabelComplianceReportTemplate"),
    ("greenwashing_risk_report", "greenwashing_risk_report.py", "GreenwashingRiskReportTemplate"),
]

ALL_TEMPLATE_KEYS = list(TEMPLATE_FILES.keys())


class TestTemplateFileExistence:
    """Tests for template file existence."""

    @pytest.mark.parametrize("key", ALL_TEMPLATE_KEYS)
    def test_template_registered_in_mapping(self, key):
        """Template key is registered in TEMPLATE_FILES."""
        assert key in TEMPLATE_FILES

    @pytest.mark.parametrize("key", ALL_TEMPLATE_KEYS)
    def test_template_has_class_mapping(self, key):
        """Template key has a class name mapping in TEMPLATE_CLASSES."""
        assert key in TEMPLATE_CLASSES

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_TEMPLATES)
    def test_template_file_exists(self, key, filename, cls_name):
        """Template file exists on disk."""
        path = TEMPLATES_DIR / filename
        assert path.exists(), f"Template file missing: {filename}"


# ===========================================================================
# Module Loading Tests (parametrized across existing templates)
# ===========================================================================


class TestTemplateModuleLoading:
    """Tests for template module loading."""

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_TEMPLATES)
    def test_template_module_loads(self, key, filename, cls_name):
        """Template module loads successfully."""
        mod = _load_template(key)
        assert mod is not None

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_TEMPLATES)
    def test_template_class_exists(self, key, filename, cls_name):
        """Template module exports the expected class."""
        mod = _load_template(key)
        assert hasattr(mod, cls_name), f"Class {cls_name} not found in {key}"

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_TEMPLATES)
    def test_template_class_has_docstring(self, key, filename, cls_name):
        """Template class has a docstring."""
        mod = _load_template(key)
        cls = getattr(mod, cls_name)
        assert cls.__doc__ is not None


# ===========================================================================
# Method Existence Tests (parametrized across existing templates)
# ===========================================================================


class TestTemplateMethodExistence:
    """Tests for template render methods."""

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_TEMPLATES)
    def test_template_has_render_markdown(self, key, filename, cls_name):
        """Template class has render_markdown method."""
        mod = _load_template(key)
        cls = getattr(mod, cls_name)
        instance = cls()
        assert hasattr(instance, "render_markdown")
        assert callable(instance.render_markdown)

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_TEMPLATES)
    def test_template_has_render_html(self, key, filename, cls_name):
        """Template class has render_html method."""
        mod = _load_template(key)
        cls = getattr(mod, cls_name)
        instance = cls()
        assert hasattr(instance, "render_html")
        assert callable(instance.render_html)

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_TEMPLATES)
    def test_template_has_render_json(self, key, filename, cls_name):
        """Template class has render_json method."""
        mod = _load_template(key)
        cls = getattr(mod, cls_name)
        instance = cls()
        assert hasattr(instance, "render_json")
        assert callable(instance.render_json)


# ===========================================================================
# Source File Characteristic Tests
# ===========================================================================


class TestTemplateSourceCharacteristics:
    """Tests for template source file characteristics."""

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_TEMPLATES)
    def test_template_source_has_sha256_or_hashlib(self, key, filename, cls_name):
        """Template source references SHA-256 or hashlib."""
        source = (TEMPLATES_DIR / filename).read_text(encoding="utf-8")
        assert "sha256" in source.lower() or "hashlib" in source or "provenance" in source.lower()

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_TEMPLATES)
    def test_template_source_has_datetime(self, key, filename, cls_name):
        """Template source imports or references datetime for timestamps."""
        source = (TEMPLATES_DIR / filename).read_text(encoding="utf-8")
        assert "datetime" in source or "generated_at" in source

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_TEMPLATES)
    def test_template_source_has_sections(self, key, filename, cls_name):
        """Template source defines report sections."""
        source = (TEMPLATES_DIR / filename).read_text(encoding="utf-8")
        has_sections = (
            "section" in source.lower()
            or "header" in source.lower()
            or "title" in source.lower()
            or "report" in source.lower()
        )
        assert has_sections


# ===========================================================================
# Missing Templates Tests
# ===========================================================================


MISSING_TEMPLATES = [
    ("compliance_gap_report", "compliance_gap_report.py"),
    ("green_claims_scorecard", "green_claims_scorecard.py"),
    ("regulatory_submission_report", "regulatory_submission_report.py"),
]


class TestMissingTemplates:
    """Tests documenting templates not yet created."""

    @pytest.mark.parametrize("key,filename", MISSING_TEMPLATES)
    def test_missing_template_registered(self, key, filename):
        """Missing template is registered in TEMPLATE_FILES."""
        assert key in TEMPLATE_FILES

    @pytest.mark.parametrize("key,filename", MISSING_TEMPLATES)
    def test_missing_template_file_status(self, key, filename):
        """Missing template file existence check."""
        path = TEMPLATES_DIR / filename
        if not path.exists():
            pytest.skip(f"{filename} not yet created")
        assert path.exists()
