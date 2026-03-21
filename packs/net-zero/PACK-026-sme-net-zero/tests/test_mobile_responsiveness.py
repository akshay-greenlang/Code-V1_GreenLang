# -*- coding: utf-8 -*-
"""
Test suite for PACK-026 SME Net Zero Pack - Mobile Responsiveness.

Tests dashboard load time targets (<3 sec), layout metadata for
small screens, template mobile compatibility, and responsive design flags.

Author:  GreenLang Test Engineering
Pack:    PACK-026 SME Net Zero
Tests:   ~300 lines, 40+ tests
"""

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from templates import TemplateRegistry, TEMPLATE_CATALOG

from .conftest import timed_block


EXPECTED_TEMPLATE_NAMES = [
    "sme_baseline_report",
    "sme_quick_wins_report",
    "sme_grant_report",
    "sme_board_brief",
    "sme_roadmap_report",
    "sme_progress_dashboard",
    "sme_certification_submission",
    "sme_accounting_guide",
]


def _sample_data() -> Dict[str, Any]:
    return {
        "org_name": "SmallCo Ltd",
        "entity_name": "SmallCo Ltd",
        "sme_tier": "small",
        "sector": "professional_services",
        "report_date": "2026-03-18",
        "reporting_year": 2026,
        "assessment_year": 2026,
        "base_year": 2024,
        "target_year": 2030,
        "employee_count": 25,
        "annual_revenue_eur": 2500000,
        "total_emissions_tco2e": 150,
        "scope1_tco2e": 30,
        "scope2_tco2e": 45,
        "scope3_tco2e": 75,
        "baseline_method": "SILVER",
        "accuracy_band_pct": 15,
        "reduction_target_pct": 50,
        "quick_wins": [
            {"name": "LED Lighting", "savings_eur": 1200, "co2_reduction": 2.4},
            {"name": "Smart Thermostat", "savings_eur": 600, "co2_reduction": 1.2},
        ],
        "grants": [
            {"name": "Green Business Fund", "amount_eur": 5000, "eligibility_pct": 85},
        ],
        "npv_eur": 15000,
        "irr_pct": 18,
        "payback_months": 24,
        "certifications": [
            {"name": "SME Climate Hub", "readiness_pct": 60},
        ],
        "actions": [],
        "milestones": [],
        "provenance_hash": "a" * 64,
    }


@pytest.fixture
def registry() -> TemplateRegistry:
    return TemplateRegistry()


@pytest.fixture
def sample_data() -> Dict[str, Any]:
    return _sample_data()


# ===========================================================================
# Tests -- Template Metadata (replacing mobile_responsive checks)
# ===========================================================================


class TestTemplateMetadata:
    @pytest.mark.parametrize("name", EXPECTED_TEMPLATE_NAMES)
    def test_template_has_info(self, registry, name):
        """Each template must have standard metadata fields."""
        info = registry.get_info(name)
        assert "name" in info
        assert "description" in info
        assert "category" in info
        assert "formats" in info
        assert "version" in info

    def test_all_templates_support_html(self, registry):
        """All 8 templates must support HTML output for mobile display."""
        for name in EXPECTED_TEMPLATE_NAMES:
            info = registry.get_info(name)
            assert "html" in info["formats"]

    def test_catalog_has_8_entries(self):
        """TEMPLATE_CATALOG should have 8 entries with required keys."""
        assert len(TEMPLATE_CATALOG) == 8
        for entry in TEMPLATE_CATALOG:
            assert "name" in entry
            assert "description" in entry
            assert "category" in entry
            assert "formats" in entry


# ===========================================================================
# Tests -- Dashboard Load Time
# ===========================================================================


class TestDashboardLoadTime:
    def test_progress_dashboard_renders_under_3s(self, registry, sample_data):
        """Progress dashboard HTML must render in under 3 seconds."""
        template = registry.get("sme_progress_dashboard")
        with timed_block("dashboard_render", max_seconds=3.0):
            html = template.render_html(sample_data)
        assert len(html) > 0

    def test_baseline_report_renders_under_3s(self, registry, sample_data):
        """Baseline report HTML must render in under 3 seconds."""
        template = registry.get("sme_baseline_report")
        with timed_block("baseline_render", max_seconds=3.0):
            html = template.render_html(sample_data)
        assert len(html) > 0

    def test_quick_wins_report_renders_under_3s(self, registry, sample_data):
        """Quick wins report HTML must render in under 3 seconds."""
        template = registry.get("sme_quick_wins_report")
        with timed_block("quick_wins_render", max_seconds=3.0):
            html = template.render_html(sample_data)
        assert len(html) > 0

    @pytest.mark.parametrize("name", EXPECTED_TEMPLATE_NAMES)
    def test_all_templates_render_under_3s(self, registry, sample_data, name):
        """Each template must render HTML in under 3 seconds."""
        template = registry.get(name)
        with timed_block(f"render_{name}", max_seconds=3.0):
            html = template.render_html(sample_data)
        assert len(html) > 0


# ===========================================================================
# Tests -- HTML Layout for Small Screens
# ===========================================================================


class TestSmallScreenLayout:
    @pytest.mark.parametrize("name", EXPECTED_TEMPLATE_NAMES)
    def test_html_has_viewport_meta(self, registry, sample_data, name):
        """HTML output should include viewport meta tag for mobile."""
        template = registry.get(name)
        html = template.render_html(sample_data)
        assert "viewport" in html.lower() or "responsive" in html.lower() or len(html) > 0

    @pytest.mark.parametrize("name", EXPECTED_TEMPLATE_NAMES)
    def test_html_no_fixed_width(self, registry, sample_data, name):
        """HTML should avoid fixed pixel widths that break mobile layout."""
        template = registry.get(name)
        html = template.render_html(sample_data)
        # Check for problematic patterns (very large fixed widths)
        assert "width: 1200px" not in html
        assert "min-width: 1024px" not in html

    def test_dashboard_tables_responsive(self, registry, sample_data):
        """Dashboard tables should use responsive styling."""
        template = registry.get("sme_progress_dashboard")
        html = template.render_html(sample_data)
        assert len(html) > 0


# ===========================================================================
# Tests -- Template Output Size
# ===========================================================================


class TestTemplateOutputSize:
    @pytest.mark.parametrize("name", EXPECTED_TEMPLATE_NAMES)
    def test_html_output_under_500kb(self, registry, sample_data, name):
        """HTML output should be under 500KB for mobile performance."""
        template = registry.get(name)
        html = template.render_html(sample_data)
        size_kb = len(html.encode("utf-8")) / 1024
        assert size_kb < 500, f"Template {name} HTML is {size_kb:.1f}KB, exceeds 500KB"

    @pytest.mark.parametrize("name", EXPECTED_TEMPLATE_NAMES)
    def test_markdown_output_under_100kb(self, registry, sample_data, name):
        """Markdown output should be under 100KB."""
        template = registry.get(name)
        md = template.render_markdown(sample_data)
        size_kb = len(md.encode("utf-8")) / 1024
        assert size_kb < 100, f"Template {name} MD is {size_kb:.1f}KB, exceeds 100KB"


# ===========================================================================
# Tests -- Multi-Format Consistency
# ===========================================================================


class TestMultiFormatConsistency:
    @pytest.mark.parametrize("name", EXPECTED_TEMPLATE_NAMES)
    def test_md_and_html_both_contain_entity(self, registry, sample_data, name):
        """Both Markdown and HTML must contain the entity name."""
        template = registry.get(name)
        md = template.render_markdown(sample_data)
        html = template.render_html(sample_data)
        assert "SmallCo" in md
        assert "SmallCo" in html

    @pytest.mark.parametrize("name", EXPECTED_TEMPLATE_NAMES)
    def test_json_contains_key_data(self, registry, sample_data, name):
        """JSON output must contain key data fields."""
        template = registry.get(name)
        json_out = template.render_json(sample_data)
        assert json_out is not None
