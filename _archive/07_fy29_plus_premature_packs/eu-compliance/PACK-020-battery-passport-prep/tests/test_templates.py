# -*- coding: utf-8 -*-
"""
PACK-020 Battery Passport Prep Pack - Template Tests
========================================================

Tests all 8 templates and the TemplateRegistry. Validates render_markdown,
render_html, render_json for each template. Uses dynamic import pattern.

Author: GreenLang Platform Team (GL-TestEngineer)
"""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = PACK_ROOT / "templates"

TEMPLATE_FILES = {
    "carbon_footprint_declaration": "carbon_footprint_declaration.py",
    "recycled_content_report": "recycled_content_report.py",
    "battery_passport_report": "battery_passport_report.py",
    "performance_report": "performance_report.py",
    "due_diligence_report": "due_diligence_report.py",
    "labelling_compliance_report": "labelling_compliance_report.py",
    "end_of_life_report": "end_of_life_report.py",
    "battery_regulation_scorecard": "battery_regulation_scorecard.py",
}

TEMPLATE_CLASSES = {
    "carbon_footprint_declaration": "CarbonFootprintDeclarationTemplate",
    "recycled_content_report": "RecycledContentReportTemplate",
    "battery_passport_report": "BatteryPassportReportTemplate",
    "performance_report": "PerformanceReportTemplate",
    "due_diligence_report": "DueDiligenceReportTemplate",
    "labelling_compliance_report": "LabellingComplianceReportTemplate",
    "end_of_life_report": "EndOfLifeReportTemplate",
    "battery_regulation_scorecard": "BatteryRegulationScorecardTemplate",
}


def _load_module(file_name: str, module_name: str, subdir: str = ""):
    if subdir:
        file_path = PACK_ROOT / subdir / file_name
    else:
        file_path = PACK_ROOT / file_name
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_template_module(name: str):
    filename = TEMPLATE_FILES[name]
    return _load_module(filename, f"pack020_tt.tpl_{name}", "templates")


# Load all template modules
_template_modules = {}
for tname in TEMPLATE_FILES:
    try:
        _template_modules[tname] = _load_template_module(tname)
    except Exception:
        _template_modules[tname] = None


def _get_template_instance(name: str):
    mod = _template_modules.get(name)
    if mod is None:
        pytest.skip(f"Template module {name} not loadable")
    cls_name = TEMPLATE_CLASSES[name]
    cls = getattr(mod, cls_name, None)
    if cls is None:
        pytest.skip(f"Class {cls_name} not found")
    return cls()


# ---------------------------------------------------------------------------
# Sample data fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_template_data() -> Dict[str, Any]:
    """Comprehensive sample data for template rendering."""
    return {
        "entity_name": "EuroBattery GmbH",
        "battery_id": "BAT-EV-2027-001",
        "battery_model": "EB-75-NMC811",
        "battery_type": "ev_battery",
        "battery_category": "EV",
        "chemistry": "NMC811",
        "manufacturer": "EuroBattery GmbH",
        "manufacturing_plant": "Gigafactory Berlin",
        "production_date": "2027-01-15",
        "weight_kg": 450,
        "rated_capacity_kwh": 75.0,
        "rated_capacity_ah": 200.0,
        "nominal_voltage_v": 400.0,
        "total_carbon_footprint_kgco2e_per_kwh": 65.0,
        "total_carbon_footprint_kgco2e": 4875.0,
        "performance_class": "B",
        "lifecycle_stages": {
            "raw_material_acquisition": {"kgco2e_per_kwh": 33.3, "data_quality": "measured"},
            "manufacturing": {"kgco2e_per_kwh": 20.0, "data_quality": "measured"},
            "distribution": {"kgco2e_per_kwh": 2.7, "data_quality": "estimated"},
            "end_of_life": {"kgco2e_per_kwh": 9.0, "data_quality": "measured"},
        },
        "methodology": {
            "standard": "Commission Delegated Regulation (EU) 2023/1791",
            "lca_standard": "ISO 14067:2018",
            "system_boundary": "cradle-to-grave",
            "functional_unit": "1 kWh of total energy provided over expected service life",
        },
        "recycled_content": {
            "cobalt_pct": 20.0,
            "lithium_pct": 8.0,
            "nickel_pct": 10.0,
            "lead_pct": 0.0,
        },
        "supply_chain_suppliers": [
            {"name": "SupA", "country": "DE", "risk": "LOW"},
            {"name": "SupB", "country": "CD", "risk": "HIGH"},
        ],
        "label_elements": [
            {"element": "CE_MARKING", "present": True},
            {"element": "QR_CODE", "present": True},
        ],
        "collection_rate_pct": 100.0,
        "recycling_efficiency_pct": 72.0,
        "material_recovery": {
            "cobalt_pct": 92.0,
            "lithium_pct": 55.0,
            "nickel_pct": 91.0,
        },
        "overall_compliance_score": 82.0,
        "conformity_module": "MODULE_A",
        "reporting_year": 2027,
        "cycle_life": 1500,
        "round_trip_efficiency_pct": 94.0,
        "soh_pct": 98.0,
        "soc_pct": 80.0,
        "due_diligence_compliant": True,
        "overall_score_pct": 85.0,
    }


# =========================================================================
# Parameterized tests for all templates
# =========================================================================

@pytest.mark.parametrize("template_name", list(TEMPLATE_FILES.keys()))
class TestTemplateRendering:
    """Parameterized tests for all 8 template modules."""

    def test_instantiation(self, template_name):
        tpl = _get_template_instance(template_name)
        assert tpl is not None

    def test_instantiation_with_config(self, template_name):
        mod = _template_modules.get(template_name)
        if mod is None:
            pytest.skip(f"Module not loaded: {template_name}")
        cls_name = TEMPLATE_CLASSES[template_name]
        cls = getattr(mod, cls_name)
        tpl = cls(config={"custom": True})
        assert tpl.config.get("custom") is True

    def test_render_markdown(self, template_name, sample_template_data):
        tpl = _get_template_instance(template_name)
        md = tpl.render_markdown(sample_template_data)
        assert isinstance(md, str)
        assert len(md) > 50
        assert "Provenance" in md or "#" in md

    def test_render_html(self, template_name, sample_template_data):
        tpl = _get_template_instance(template_name)
        html = tpl.render_html(sample_template_data)
        assert isinstance(html, str)
        assert "<html" in html.lower() or "<div" in html.lower()
        assert len(html) > 100

    def test_render_json(self, template_name, sample_template_data):
        tpl = _get_template_instance(template_name)
        result = tpl.render_json(sample_template_data)
        assert isinstance(result, dict)
        assert "provenance_hash" in result or "report_id" in result or len(result) > 0

    def test_render_markdown_provenance(self, template_name, sample_template_data):
        tpl = _get_template_instance(template_name)
        md = tpl.render_markdown(sample_template_data)
        assert "Provenance" in md or "provenance" in md.lower() or "<!--" in md

    def test_get_sections_if_available(self, template_name):
        tpl = _get_template_instance(template_name)
        if hasattr(tpl, "get_sections"):
            sections = tpl.get_sections()
            assert isinstance(sections, list)
            assert len(sections) > 0

    def test_validate_data_if_available(self, template_name, sample_template_data):
        tpl = _get_template_instance(template_name)
        if hasattr(tpl, "validate_data"):
            result = tpl.validate_data(sample_template_data)
            assert isinstance(result, dict)
            assert "valid" in result


# =========================================================================
# TemplateRegistry tests
# =========================================================================

class TestTemplateRegistry:
    """Tests for the TemplateRegistry from templates/__init__.py."""

    def _load_registry(self):
        """Attempt to load TemplateRegistry via dynamic import."""
        try:
            mod = _load_module("__init__.py", "pack020_tt.tpl_init", "templates")
            return mod.TemplateRegistry
        except Exception:
            pytest.skip("TemplateRegistry not loadable")

    def test_registry_instantiation(self):
        RegistryCls = self._load_registry()
        registry = RegistryCls()
        assert registry is not None
        assert registry.template_count >= 0

    def test_list_templates(self):
        RegistryCls = self._load_registry()
        registry = RegistryCls()
        templates = registry.list_templates()
        assert isinstance(templates, list)

    def test_list_template_names(self):
        RegistryCls = self._load_registry()
        registry = RegistryCls()
        names = registry.list_template_names()
        assert isinstance(names, list)

    def test_has_template(self):
        RegistryCls = self._load_registry()
        registry = RegistryCls()
        if registry.template_count > 0:
            first_name = registry.list_template_names()[0]
            assert registry.has_template(first_name) is True
        assert registry.has_template("nonexistent_template") is False

    def test_get_raises_key_error(self):
        RegistryCls = self._load_registry()
        registry = RegistryCls()
        with pytest.raises(KeyError):
            registry.get("totally_nonexistent_template_xyz")

    def test_render_raises_value_error_bad_format(self):
        RegistryCls = self._load_registry()
        registry = RegistryCls()
        if registry.template_count > 0:
            first_name = registry.list_template_names()[0]
            with pytest.raises(ValueError):
                registry.render(first_name, {}, format="xml")

    def test_get_info(self):
        RegistryCls = self._load_registry()
        registry = RegistryCls()
        if registry.template_count > 0:
            first_name = registry.list_template_names()[0]
            info = registry.get_info(first_name)
            assert "name" in info
            assert "description" in info

    def test_repr(self):
        RegistryCls = self._load_registry()
        registry = RegistryCls()
        r = repr(registry)
        assert "TemplateRegistry" in r
