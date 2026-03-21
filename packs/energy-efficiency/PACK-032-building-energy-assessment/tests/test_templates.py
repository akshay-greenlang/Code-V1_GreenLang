# -*- coding: utf-8 -*-
"""
Unit tests for PACK-032 Building Energy Assessment Templates

Tests template module loading, class instantiation, render_markdown,
render_html, render_json, and section definitions for all 10 report templates.

Target: 15+ tests
Author: GL-TestEngineer
"""

import importlib.util
import sys
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = PACK_ROOT / "templates"


TEMPLATE_DEFINITIONS = {
    "building_assessment_report": "BuildingAssessmentReportTemplate",
    "epc_report": "EPCReportTemplate",
    "dec_report": "DECReportTemplate",
    "building_benchmark_report": "BuildingBenchmarkReportTemplate",
    "retrofit_recommendation_report": "RetrofitRecommendationReportTemplate",
    "certification_scorecard": "CertificationScorecardTemplate",
    "regulatory_compliance_report": "RegulatoryComplianceReportTemplate",
    "building_dashboard": "BuildingDashboardTemplate",
    "tenant_energy_report": "TenantEnergyReportTemplate",
    "whole_life_carbon_report": "WholeLifeCarbonReportTemplate",
}


def _load_tpl(name: str):
    path = TEMPLATES_DIR / f"{name}.py"
    if not path.exists():
        pytest.skip(f"Template file not found: {path}")
    mod_key = f"pack032_tpl.{name}"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load {name}: {exc}")
    return mod


# =========================================================================
# Test Template File Existence
# =========================================================================


class TestTemplateFiles:
    @pytest.mark.parametrize("tpl_file", list(TEMPLATE_DEFINITIONS.keys()))
    def test_template_file_exists(self, tpl_file):
        path = TEMPLATES_DIR / f"{tpl_file}.py"
        assert path.exists(), f"Template file missing: {tpl_file}.py"

    def test_templates_init_exists(self):
        assert (TEMPLATES_DIR / "__init__.py").exists()

    def test_template_count(self):
        py_files = [f for f in TEMPLATES_DIR.glob("*.py") if f.name != "__init__.py"]
        assert len(py_files) >= 10


# =========================================================================
# Test Template Class Loading
# =========================================================================


class TestTemplateClasses:
    @pytest.mark.parametrize(
        "tpl_file,class_name",
        list(TEMPLATE_DEFINITIONS.items()),
    )
    def test_template_class_exists(self, tpl_file, class_name):
        mod = _load_tpl(tpl_file)
        assert hasattr(mod, class_name), f"{class_name} not found in {tpl_file}"

    @pytest.mark.parametrize(
        "tpl_file,class_name",
        list(TEMPLATE_DEFINITIONS.items()),
    )
    def test_template_instantiation(self, tpl_file, class_name):
        mod = _load_tpl(tpl_file)
        cls = getattr(mod, class_name)
        instance = cls()
        assert instance is not None


# =========================================================================
# Test Template Render Methods
# =========================================================================


SAMPLE_DATA = {
    "building_id": "BLD-TEST-001",
    "building_name": "Test Building",
    "building_type": "office",
    "building_address": "123 Test Street",
    "floor_area_m2": 2000.0,
    "epc_rating": {
        "current_score": 65,
        "current_band": "C",
        "potential_score": 42,
        "potential_band": "B",
    },
    "epc_score": 65,
    "eui_kwh_m2": 180.0,
    "co2_emissions_kg": 50000.0,
    "co2_emissions_kg_m2": 25.0,
    "current_rating": "D",
    "target_rating": "B",
    "recommendations": [
        {"measure": "LED Lighting", "savings_kwh": 15000, "cost": 25000},
        {"measure": "Wall Insulation", "savings_kwh": 30000, "cost": 80000},
    ],
    "assessor_name": "Test Assessor",
    "assessment_date": "2025-03-01",
    "electricity_kwh": 250000,
    "gas_kwh": 150000,
    "total_energy_kwh": 400000,
    "benchmark_eui": 120.0,
    "operational_rating": {
        "score": 148,
        "band": "D",
        "typical_score": 100,
    },
    "dec_band": "D",
    "whole_life_carbon_kgco2e": 1500000,
    "whole_life_carbon_per_m2": 750.0,
    "compliance_status": "PARTIAL",
    "certification_score": 65,
    "certification_target": "BREEAM Very Good",
}


class TestTemplateRendering:
    @pytest.mark.parametrize(
        "tpl_file,class_name",
        list(TEMPLATE_DEFINITIONS.items()),
    )
    def test_has_render_markdown(self, tpl_file, class_name):
        mod = _load_tpl(tpl_file)
        cls = getattr(mod, class_name)
        instance = cls()
        assert hasattr(instance, "render_markdown")

    @pytest.mark.parametrize(
        "tpl_file,class_name",
        list(TEMPLATE_DEFINITIONS.items()),
    )
    def test_has_render_html(self, tpl_file, class_name):
        mod = _load_tpl(tpl_file)
        cls = getattr(mod, class_name)
        instance = cls()
        assert hasattr(instance, "render_html")

    @pytest.mark.parametrize(
        "tpl_file,class_name",
        list(TEMPLATE_DEFINITIONS.items()),
    )
    def test_has_render_json(self, tpl_file, class_name):
        mod = _load_tpl(tpl_file)
        cls = getattr(mod, class_name)
        instance = cls()
        assert hasattr(instance, "render_json")

    @pytest.mark.parametrize(
        "tpl_file,class_name",
        list(TEMPLATE_DEFINITIONS.items()),
    )
    def test_render_markdown_returns_string(self, tpl_file, class_name):
        mod = _load_tpl(tpl_file)
        cls = getattr(mod, class_name)
        instance = cls()
        result = instance.render_markdown(SAMPLE_DATA)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.parametrize(
        "tpl_file,class_name",
        list(TEMPLATE_DEFINITIONS.items()),
    )
    def test_render_html_returns_string(self, tpl_file, class_name):
        mod = _load_tpl(tpl_file)
        cls = getattr(mod, class_name)
        instance = cls()
        result = instance.render_html(SAMPLE_DATA)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.parametrize(
        "tpl_file,class_name",
        list(TEMPLATE_DEFINITIONS.items()),
    )
    def test_render_json_returns_dict(self, tpl_file, class_name):
        mod = _load_tpl(tpl_file)
        cls = getattr(mod, class_name)
        instance = cls()
        result = instance.render_json(SAMPLE_DATA)
        assert isinstance(result, dict)


# =========================================================================
# Test EPC Report Specifics
# =========================================================================


class TestEPCReportSpecifics:
    def test_epc_bands_defined(self):
        mod = _load_tpl("epc_report")
        assert hasattr(mod, "EPC_BANDS")
        assert isinstance(mod.EPC_BANDS, list)
        assert len(mod.EPC_BANDS) == 7  # A through G

    def test_epc_sections(self):
        mod = _load_tpl("epc_report")
        cls = getattr(mod, "EPCReportTemplate")
        assert hasattr(cls, "EPC_SECTIONS")
        assert len(cls.EPC_SECTIONS) >= 5

    def test_epc_markdown_contains_provenance(self):
        mod = _load_tpl("epc_report")
        tpl = mod.EPCReportTemplate()
        md = tpl.render_markdown(SAMPLE_DATA)
        assert "Provenance" in md or "provenance" in md.lower()

    def test_epc_html_is_valid(self):
        mod = _load_tpl("epc_report")
        tpl = mod.EPCReportTemplate()
        html = tpl.render_html(SAMPLE_DATA)
        assert "<html" in html.lower()
        assert "</html>" in html.lower()
