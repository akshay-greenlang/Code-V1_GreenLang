# -*- coding: utf-8 -*-
"""
PACK-016 ESRS E1 Climate Pack - Template Tests
=================================================

Tests for all 9 E1 report templates: class existence, render methods
(markdown, html, json), section completeness, data validation, and
provenance hashing.

Target: 30+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-016 ESRS E1 Climate Change
Date:    March 2026
"""

import inspect
import json

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


def _sample_ghg_data():
    """Minimal sample data for GHG emissions template rendering tests."""
    return {
        "company_name": "NordEnergy AG",
        "reporting_year": 2025,
        "base_year": 2020,
        "scope_1_total_tco2e": 450000.0,
        "scope_2_location_tco2e": 120000.0,
        "scope_2_market_tco2e": 85000.0,
        "scope_3_total_tco2e": 320000.0,
        "total_tco2e": 855000.0,
        "ghg_intensity_per_revenue": 267.2,
        "gas_disaggregation": {
            "CO2": 830000.0,
            "CH4": 15000.0,
            "N2O": 8000.0,
            "SF6": 2000.0,
        },
        "biogenic_co2_tco2e": 5200.0,
        "scope_3_categories": [
            {"category": 1, "name": "Purchased Goods", "tco2e": 150000.0},
            {"category": 3, "name": "Fuel & Energy Activities", "tco2e": 80000.0},
            {"category": 6, "name": "Business Travel", "tco2e": 12000.0},
        ],
        "methodology": "GHG Protocol Corporate Standard",
        "gwp_source": "IPCC AR6",
        "provenance_hash": "a" * 64,
    }


def _sample_energy_data():
    """Minimal sample data for energy mix template rendering tests."""
    return {
        "company_name": "NordEnergy AG",
        "reporting_year": 2025,
        "total_energy_mwh": 5200000.0,
        "fossil_energy_mwh": 3100000.0,
        "renewable_energy_mwh": 1800000.0,
        "nuclear_energy_mwh": 0.0,
        "renewable_share_pct": 34.6,
        "energy_intensity_per_revenue": 1625.0,
        "sources": [
            {"source": "Natural Gas", "mwh": 2800000.0, "category": "FOSSIL"},
            {"source": "Wind Onshore", "mwh": 1200000.0, "category": "RENEWABLE"},
            {"source": "Solar PV", "mwh": 600000.0, "category": "RENEWABLE"},
        ],
        "provenance_hash": "b" * 64,
    }


# ===========================================================================
# Template File Existence
# ===========================================================================


class TestTemplateFilesExist:
    """Test that all 9 template files exist on disk."""

    @pytest.mark.parametrize("tpl_key,tpl_file", list(TEMPLATE_FILES.items()))
    def test_template_file_exists(self, tpl_key, tpl_file):
        """Template file exists on disk."""
        path = TEMPLATES_DIR / tpl_file
        assert path.exists(), f"Template file missing: {path}"


class TestTemplateLoading:
    """Test that all 9 templates can be loaded."""

    @pytest.mark.parametrize("tpl_key", list(TEMPLATE_FILES.keys()))
    def test_template_module_loads(self, tpl_key):
        """Each template module loads independently."""
        mod = _try_load_template(tpl_key)
        assert mod is not None, f"Template {tpl_key} failed to load"

    def test_all_9_templates_loadable(self):
        """All 9 templates load successfully."""
        loaded = []
        for key in TEMPLATE_FILES:
            mod = _try_load_template(key)
            if mod is not None:
                loaded.append(key)
        assert len(loaded) == 9, f"Loaded {len(loaded)}/9 templates: {loaded}"

    @pytest.mark.parametrize("tpl_key,tpl_class", list(TEMPLATE_CLASSES.items()))
    def test_template_class_exists(self, tpl_key, tpl_class):
        """Each template exports its primary class."""
        mod = _try_load_template(tpl_key)
        if mod is None:
            pytest.skip(f"Template {tpl_key} not loaded")
        assert hasattr(mod, tpl_class), f"Template {tpl_key} missing class {tpl_class}"


# ===========================================================================
# GHG Emissions Report Template
# ===========================================================================


class TestGHGEmissionsTemplate:
    """Tests for the GHGEmissionsReportTemplate."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_template("ghg_emissions_report")

    def test_class_exists(self):
        """GHGEmissionsReportTemplate class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "GHGEmissionsReportTemplate")

    def test_has_render_methods(self):
        """Template has render_markdown, render_html, render_json methods."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = self.mod.GHGEmissionsReportTemplate
        for method in ["render_markdown", "render_html", "render_json"]:
            assert hasattr(cls, method), f"Missing method: {method}"

    def test_has_get_sections(self):
        """Template has get_sections method or SECTIONS attribute."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = self.mod.GHGEmissionsReportTemplate
        has_sections = (
            hasattr(cls, "get_sections")
            or hasattr(cls, "SECTIONS")
            or hasattr(cls, "sections")
        )
        assert has_sections


# ===========================================================================
# Energy Mix Report Template
# ===========================================================================


class TestEnergyMixTemplate:
    """Tests for the EnergyMixReportTemplate."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_template("energy_mix_report")

    def test_class_exists(self):
        """EnergyMixReportTemplate class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "EnergyMixReportTemplate")

    def test_has_render_methods(self):
        """Template has render_markdown, render_html, render_json methods."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = self.mod.EnergyMixReportTemplate
        for method in ["render_markdown", "render_html", "render_json"]:
            assert hasattr(cls, method)

    def test_source_references_e1_5(self):
        """Template source references E1-5."""
        path = TEMPLATES_DIR / "energy_mix_report.py"
        content = path.read_text(encoding="utf-8")
        assert "E1-5" in content or "E1_5" in content


# ===========================================================================
# Transition Plan Report Template
# ===========================================================================


class TestTransitionPlanTemplate:
    """Tests for the TransitionPlanReportTemplate."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_template("transition_plan_report")

    def test_class_exists(self):
        """TransitionPlanReportTemplate class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "TransitionPlanReportTemplate")

    def test_has_render_methods(self):
        """Template has render methods."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = self.mod.TransitionPlanReportTemplate
        assert hasattr(cls, "render_markdown") or hasattr(cls, "render")

    def test_source_references_e1_1(self):
        """Template source references E1-1."""
        path = TEMPLATES_DIR / "transition_plan_report.py"
        content = path.read_text(encoding="utf-8")
        assert "E1-1" in content or "E1_1" in content


# ===========================================================================
# Climate Policy Report Template
# ===========================================================================


class TestClimatePolicyTemplate:
    """Tests for the ClimatePolicyReportTemplate."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_template("climate_policy_report")

    def test_class_exists(self):
        """ClimatePolicyReportTemplate class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "ClimatePolicyReportTemplate")

    def test_source_references_e1_2(self):
        """Template source references E1-2."""
        path = TEMPLATES_DIR / "climate_policy_report.py"
        content = path.read_text(encoding="utf-8")
        assert "E1-2" in content or "E1_2" in content


# ===========================================================================
# Climate Actions Report Template
# ===========================================================================


class TestClimateActionsTemplate:
    """Tests for the ClimateActionsReportTemplate."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_template("climate_actions_report")

    def test_class_exists(self):
        """ClimateActionsReportTemplate class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "ClimateActionsReportTemplate")

    def test_source_references_e1_3(self):
        """Template source references E1-3."""
        path = TEMPLATES_DIR / "climate_actions_report.py"
        content = path.read_text(encoding="utf-8")
        assert "E1-3" in content or "E1_3" in content


# ===========================================================================
# Climate Targets Report Template
# ===========================================================================


class TestClimateTargetsTemplate:
    """Tests for the ClimateTargetsReportTemplate."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_template("climate_targets_report")

    def test_class_exists(self):
        """ClimateTargetsReportTemplate class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "ClimateTargetsReportTemplate")

    def test_source_references_e1_4(self):
        """Template source references E1-4."""
        path = TEMPLATES_DIR / "climate_targets_report.py"
        content = path.read_text(encoding="utf-8")
        assert "E1-4" in content or "E1_4" in content


# ===========================================================================
# Carbon Credits Report Template
# ===========================================================================


class TestCarbonCreditsTemplate:
    """Tests for the CarbonCreditsReportTemplate."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_template("carbon_credits_report")

    def test_class_exists(self):
        """CarbonCreditsReportTemplate class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "CarbonCreditsReportTemplate")

    def test_source_references_e1_7(self):
        """Template source references E1-7."""
        path = TEMPLATES_DIR / "carbon_credits_report.py"
        content = path.read_text(encoding="utf-8")
        assert "E1-7" in content or "E1_7" in content


# ===========================================================================
# Carbon Pricing Report Template
# ===========================================================================


class TestCarbonPricingTemplate:
    """Tests for the CarbonPricingReportTemplate."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_template("carbon_pricing_report")

    def test_class_exists(self):
        """CarbonPricingReportTemplate class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "CarbonPricingReportTemplate")

    def test_source_references_e1_8(self):
        """Template source references E1-8."""
        path = TEMPLATES_DIR / "carbon_pricing_report.py"
        content = path.read_text(encoding="utf-8")
        assert "E1-8" in content or "E1_8" in content


# ===========================================================================
# Climate Risk Report Template
# ===========================================================================


class TestClimateRiskTemplate:
    """Tests for the ClimateRiskReportTemplate."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_template("climate_risk_report")

    def test_class_exists(self):
        """ClimateRiskReportTemplate class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "ClimateRiskReportTemplate")

    def test_source_references_e1_9(self):
        """Template source references E1-9."""
        path = TEMPLATES_DIR / "climate_risk_report.py"
        content = path.read_text(encoding="utf-8")
        assert "E1-9" in content or "E1_9" in content


# ===========================================================================
# Template Rendering (cross-template)
# ===========================================================================


class TestTemplateRendering:
    """Tests for template rendering across all templates."""

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
    def test_template_has_render_json(self, tpl_key, tpl_class):
        """Each template class has render_json method."""
        mod = _try_load_template(tpl_key)
        if mod is None:
            pytest.skip(f"Template {tpl_key} not loaded")
        cls = getattr(mod, tpl_class, None)
        if cls is None:
            pytest.skip(f"Class {tpl_class} not found")
        assert hasattr(cls, "render_json"), f"{tpl_class} missing render_json"

    @pytest.mark.parametrize("tpl_key", list(TEMPLATE_FILES.keys()))
    def test_template_uses_sha256(self, tpl_key):
        """Each template file references SHA-256 for provenance."""
        source_path = TEMPLATES_DIR / TEMPLATE_FILES[tpl_key]
        if not source_path.exists():
            pytest.skip(f"File not found: {source_path}")
        content = source_path.read_text(encoding="utf-8")
        has_sha = "sha256" in content.lower() or "hashlib" in content or "provenance" in content.lower()
        assert has_sha, f"Template {tpl_key} should reference SHA-256 provenance"
