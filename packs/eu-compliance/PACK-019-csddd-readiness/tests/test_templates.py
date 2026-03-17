# -*- coding: utf-8 -*-
"""
PACK-019 CSDDD Readiness Pack - Template Tests
================================================

Tests all 8 template classes for instantiation, render(), render_section(),
get_sections(), validate_data(), render_markdown(), render_json(), and
provenance tracking.

Test count target: ~45 tests
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import (
    TEMPLATE_CLASSES,
    TEMPLATE_FILES,
    TEMPLATES_DIR,
    _load_template,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEMPLATE_KEYS = list(TEMPLATE_FILES.keys())


def _sample_render_data() -> Dict[str, Any]:
    """Return sample data suitable for any template render() call."""
    return {
        "reporting_year": 2027,
        "entity_name": "EuroManufacturing AG",
        "entity_id": "TEST-001",
        "company_name": "EuroManufacturing AG",
        "sector": "MANUFACTURING",
        "country": "DE",
        "employee_count": 6000,
        "turnover_eur": 2_000_000_000,
        "scope": {
            "in_scope": True,
            "tier": "group_1",
            "compliance_deadline": "2027-07-26",
            "employee_count": 6000,
            "net_turnover_eur": 2_000_000_000,
            "is_eu_based": True,
        },
        "article_statuses": {
            "art_5": {"status": "partially_compliant", "score": 55.0},
            "art_6": {"status": "non_compliant", "score": 20.0},
            "art_7": {"status": "non_compliant", "score": 10.0},
            "art_8": {"status": "non_compliant", "score": 0.0},
            "art_9": {"status": "non_compliant", "score": 0.0},
            "art_10": {"status": "partially_compliant", "score": 40.0},
            "art_11": {"status": "non_compliant", "score": 0.0},
            "art_15": {"status": "non_compliant", "score": 15.0},
            "art_22": {"status": "non_compliant", "score": 0.0},
        },
        "gaps": [
            {"article": "art_6", "gap": "No impact identification process", "priority": "critical"},
            {"article": "art_7", "gap": "No prevention measures", "priority": "critical"},
        ],
        "readiness_score": 18.5,
        "readiness_level": "not_ready",
        "recommendations": [
            {"action": "Establish impact identification process", "priority": "critical"},
            {"action": "Develop prevention action plan", "priority": "high"},
        ],
        "adverse_impacts": [
            {
                "impact_id": "AI-001",
                "type": "HUMAN_RIGHTS",
                "category": "forced_labour",
                "severity": "CRITICAL",
                "likelihood": "LIKELY",
            },
        ],
        "prevention_measures": [
            {
                "measure_id": "PM-001",
                "type": "PREVENTION",
                "description": "Supplier code of conduct",
                "effectiveness_score": 0.7,
            },
        ],
        "grievance_cases": [
            {
                "case_id": "GC-001",
                "status": "RESOLVED",
                "days_to_resolve": 45,
            },
        ],
        "stakeholder_engagements": [
            {
                "engagement_id": "SE-001",
                "stakeholder_group": "TRADE_UNIONS",
                "method": "formal_consultation",
                "meaningful": True,
            },
        ],
        "climate_targets": [
            {
                "target_id": "CT-001",
                "scope": "SCOPE_1",
                "target_year": 2030,
                "reduction_pct": 42,
                "aligned_with_15c": True,
            },
        ],
        "value_chain": {
            "tiers": [
                {"tier": 1, "name": "Direct Suppliers", "supplier_count": 45},
                {"tier": 2, "name": "Sub-tier Suppliers", "supplier_count": 120},
            ],
            "suppliers": [
                {
                    "supplier_id": "SUP-001",
                    "name": "MetalWorks Co.",
                    "tier": 1,
                    "country": "CN",
                    "risk_score": 7.5,
                    "risk_level": "HIGH",
                },
            ],
        },
        "civil_liability": {
            "due_diligence_performed": True,
            "prevention_measures_taken": True,
            "damage_estimate_eur": 5_000_000,
        },
        "overall_score": 35.0,
        "action_items": [
            {"article": "art_6", "action": "Implement impact identification", "priority": "critical"},
        ],
        "risk_summary": {
            "critical": 2,
            "high": 3,
            "medium": 4,
            "low": 1,
        },
        "trend_analysis": {
            "current_score": 35.0,
            "previous_score": None,
            "trend": "baseline",
        },
        "monitoring_kpis": [],
        "emissions_data": {
            "scope_1": 15000,
            "scope_2": 8000,
            "scope_3": 120000,
            "total": 143000,
            "unit": "tCO2e",
        },
    }


# ---------------------------------------------------------------------------
# 1. File existence
# ---------------------------------------------------------------------------


class TestTemplateFilesExist:
    """Verify all template source files are present on disk."""

    @pytest.mark.parametrize("tpl_key", TEMPLATE_KEYS)
    def test_template_file_exists(self, tpl_key: str):
        filepath = TEMPLATES_DIR / TEMPLATE_FILES[tpl_key]
        assert filepath.exists(), f"Missing template file: {filepath}"


# ---------------------------------------------------------------------------
# 2. Module loading
# ---------------------------------------------------------------------------


class TestTemplateModuleLoading:
    """Verify all template modules load without import errors."""

    @pytest.mark.parametrize("tpl_key", TEMPLATE_KEYS)
    def test_template_module_loads(self, tpl_key: str):
        mod = _load_template(tpl_key)
        assert mod is not None


# ---------------------------------------------------------------------------
# 3. Class instantiation
# ---------------------------------------------------------------------------


class TestTemplateInstantiation:
    """Verify all template classes can be instantiated."""

    @pytest.mark.parametrize("tpl_key", TEMPLATE_KEYS)
    def test_template_class_exists(self, tpl_key: str):
        mod = _load_template(tpl_key)
        cls_name = TEMPLATE_CLASSES[tpl_key]
        cls = getattr(mod, cls_name, None)
        assert cls is not None, f"Class {cls_name} not found in {tpl_key}"

    @pytest.mark.parametrize("tpl_key", TEMPLATE_KEYS)
    def test_template_instantiation_no_args(self, tpl_key: str):
        mod = _load_template(tpl_key)
        cls = getattr(mod, TEMPLATE_CLASSES[tpl_key])
        instance = cls()
        assert instance is not None


# ---------------------------------------------------------------------------
# 4. get_sections()
# ---------------------------------------------------------------------------


class TestTemplateSections:
    """Verify get_sections() returns non-empty section lists."""

    @pytest.mark.parametrize("tpl_key", TEMPLATE_KEYS)
    def test_get_sections_returns_list(self, tpl_key: str):
        mod = _load_template(tpl_key)
        cls = getattr(mod, TEMPLATE_CLASSES[tpl_key])
        instance = cls()
        sections = instance.get_sections()
        assert isinstance(sections, list)
        assert len(sections) >= 2, "Templates should define at least 2 sections"

    @pytest.mark.parametrize("tpl_key", TEMPLATE_KEYS)
    def test_sections_are_strings(self, tpl_key: str):
        mod = _load_template(tpl_key)
        cls = getattr(mod, TEMPLATE_CLASSES[tpl_key])
        instance = cls()
        for section in instance.get_sections():
            assert isinstance(section, str)


# ---------------------------------------------------------------------------
# 5. render()
# ---------------------------------------------------------------------------


class TestTemplateRender:
    """Verify render() produces structured results with provenance."""

    @pytest.mark.parametrize("tpl_key", TEMPLATE_KEYS)
    def test_render_returns_dict(self, tpl_key: str):
        mod = _load_template(tpl_key)
        cls = getattr(mod, TEMPLATE_CLASSES[tpl_key])
        instance = cls()
        result = instance.render(_sample_render_data())
        assert isinstance(result, dict)

    @pytest.mark.parametrize("tpl_key", TEMPLATE_KEYS)
    def test_render_has_report_id(self, tpl_key: str):
        mod = _load_template(tpl_key)
        cls = getattr(mod, TEMPLATE_CLASSES[tpl_key])
        instance = cls()
        result = instance.render(_sample_render_data())
        assert "report_id" in result
        assert isinstance(result["report_id"], str)
        assert len(result["report_id"]) > 0

    @pytest.mark.parametrize("tpl_key", TEMPLATE_KEYS)
    def test_render_has_provenance_hash(self, tpl_key: str):
        mod = _load_template(tpl_key)
        cls = getattr(mod, TEMPLATE_CLASSES[tpl_key])
        instance = cls()
        result = instance.render(_sample_render_data())
        assert "provenance_hash" in result
        assert isinstance(result["provenance_hash"], str)
        assert len(result["provenance_hash"]) == 64

    @pytest.mark.parametrize("tpl_key", TEMPLATE_KEYS)
    def test_render_has_generated_at(self, tpl_key: str):
        mod = _load_template(tpl_key)
        cls = getattr(mod, TEMPLATE_CLASSES[tpl_key])
        instance = cls()
        result = instance.render(_sample_render_data())
        assert "generated_at" in result
        assert isinstance(result["generated_at"], str)
        assert len(result["generated_at"]) > 0

    @pytest.mark.parametrize("tpl_key", TEMPLATE_KEYS)
    def test_render_populates_sections(self, tpl_key: str):
        mod = _load_template(tpl_key)
        cls = getattr(mod, TEMPLATE_CLASSES[tpl_key])
        instance = cls()
        result = instance.render(_sample_render_data())
        sections = instance.get_sections()
        for section in sections:
            assert section in result, f"Section '{section}' not in render output"


# ---------------------------------------------------------------------------
# 6. render_section()
# ---------------------------------------------------------------------------


class TestTemplateRenderSection:
    """Verify render_section() works for individual sections."""

    @pytest.mark.parametrize("tpl_key", TEMPLATE_KEYS)
    def test_render_first_section(self, tpl_key: str):
        mod = _load_template(tpl_key)
        cls = getattr(mod, TEMPLATE_CLASSES[tpl_key])
        instance = cls()
        sections = instance.get_sections()
        result = instance.render_section(sections[0], _sample_render_data())
        assert isinstance(result, dict)

    @pytest.mark.parametrize("tpl_key", TEMPLATE_KEYS)
    def test_render_invalid_section_raises(self, tpl_key: str):
        mod = _load_template(tpl_key)
        cls = getattr(mod, TEMPLATE_CLASSES[tpl_key])
        instance = cls()
        with pytest.raises(ValueError):
            instance.render_section("nonexistent_section_xyz", _sample_render_data())


# ---------------------------------------------------------------------------
# 7. render_markdown()
# ---------------------------------------------------------------------------


class TestTemplateRenderMarkdown:
    """Verify render_markdown() produces non-empty Markdown strings."""

    @pytest.mark.parametrize("tpl_key", TEMPLATE_KEYS)
    def test_render_markdown_returns_string(self, tpl_key: str):
        mod = _load_template(tpl_key)
        cls = getattr(mod, TEMPLATE_CLASSES[tpl_key])
        instance = cls()
        md = instance.render_markdown(_sample_render_data())
        assert isinstance(md, str)
        assert len(md) > 100, "Markdown output should be substantial"

    @pytest.mark.parametrize("tpl_key", TEMPLATE_KEYS)
    def test_render_markdown_contains_header(self, tpl_key: str):
        mod = _load_template(tpl_key)
        cls = getattr(mod, TEMPLATE_CLASSES[tpl_key])
        instance = cls()
        md = instance.render_markdown(_sample_render_data())
        assert "#" in md, "Markdown should contain at least one header"
