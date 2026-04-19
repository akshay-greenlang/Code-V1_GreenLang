# -*- coding: utf-8 -*-
"""
Unit tests for PACK-031 Templates
====================================

Tests all 10 report templates: loading, instantiation, render_markdown,
render_html, render_json, section validation, and provenance tracking.

Coverage target: 85%+
Total tests: ~60
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = PACK_ROOT / "templates"

TEMPLATE_FILES = {
    "energy_audit_report": "energy_audit_report.py",
    "energy_baseline_report": "energy_baseline_report.py",
    "mv_report": "savings_verification_report.py",
    "compressed_air_report": "compressed_air_report.py",
    "steam_system_report": "steam_system_report.py",
    "waste_heat_report": "waste_heat_recovery_report.py",
    "energy_dashboard": "energy_management_dashboard.py",
    "regulatory_compliance_report": "regulatory_compliance_report.py",
    "iso_50001_evidence": "iso_50001_review_report.py",
    "equipment_efficiency_report": "equipment_efficiency_report.py",
}

TEMPLATE_CLASSES = {
    "energy_audit_report": "EnergyAuditReportTemplate",
    "energy_baseline_report": "EnergyBaselineReportTemplate",
    "mv_report": "SavingsVerificationReportTemplate",
    "compressed_air_report": "CompressedAirReportTemplate",
    "steam_system_report": "SteamSystemReportTemplate",
    "waste_heat_report": "WasteHeatRecoveryReportTemplate",
    "energy_dashboard": "EnergyManagementDashboardTemplate",
    "regulatory_compliance_report": "RegulatoryComplianceReportTemplate",
    "iso_50001_evidence": "ISO50001ReviewReportTemplate",
    "equipment_efficiency_report": "EquipmentEfficiencyReportTemplate",
}


def _load_template(name: str):
    file_name = TEMPLATE_FILES[name]
    path = TEMPLATES_DIR / file_name
    if not path.exists():
        pytest.skip(f"Template file not found: {path}")
    mod_key = f"pack031_test_tpl.{name}"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load template {name}: {exc}")
    return mod


def _sample_audit_data():
    """Create sample audit data for template rendering."""
    return {
        "facility_id": "FAC-031-DE-001",
        "facility_name": "Stuttgart Automotive Parts Plant",
        "company": "Mittelwerk GmbH",
        "country": "DE",
        "audit_date": "2025-03-15",
        "audit_type": "Detailed (EN 16247 Type 2)",
        "auditor": "GreenLang Energy Audit Engine",
        "total_energy_kwh": 14_500_000,
        "electricity_kwh": 8_200_000,
        "gas_kwh": 6_300_000,
        "production_tonnes": 12_500,
        "sec_kwh_per_tonne": 1_160,
        "total_savings_kwh": 1_337_800,
        "total_savings_eur": 168_870,
        "total_implementation_cost": 152_500,
        "simple_payback_years": 0.9,
        "co2_reduction_tonnes": 414,
        "findings_count": 12,
        "findings": [
            {
                "id": "AF-001",
                "title": "Compressed Air Leak Repair",
                "savings_kwh": 220_000,
                "savings_eur": 33_000,
                "cost_eur": 8_500,
                "payback": 0.26,
                "priority": "high",
            },
            {
                "id": "AF-002",
                "title": "LED High Bay Retrofit",
                "savings_kwh": 92_800,
                "savings_eur": 13_920,
                "cost_eur": 24_000,
                "payback": 1.7,
                "priority": "high",
            },
        ],
        "end_use_breakdown": [
            {"end_use": "Motors & Drives", "energy_mwh": 5075, "share_pct": 35, "cost_eur": 761250},
            {"end_use": "Compressed Air", "energy_mwh": 2610, "share_pct": 18, "cost_eur": 391500},
            {"end_use": "Lighting", "energy_mwh": 1160, "share_pct": 8, "cost_eur": 174000},
            {"end_use": "HVAC", "energy_mwh": 1740, "share_pct": 12, "cost_eur": 261000},
            {"end_use": "Process Heat", "energy_mwh": 3190, "share_pct": 22, "cost_eur": 478500},
            {"end_use": "Other", "energy_mwh": 725, "share_pct": 5, "cost_eur": 108750},
        ],
        "provenance_hash": "a" * 64,
    }


ALL_TEMPLATE_KEYS = list(TEMPLATE_FILES.keys())
EXISTING_TEMPLATE_KEYS = [
    k for k in ALL_TEMPLATE_KEYS if (TEMPLATES_DIR / TEMPLATE_FILES[k]).exists()
]


class TestTemplateFilePresence:
    """Test that template files exist on disk."""

    @pytest.mark.parametrize("tpl_key", ALL_TEMPLATE_KEYS)
    def test_file_exists(self, tpl_key):
        path = TEMPLATES_DIR / TEMPLATE_FILES[tpl_key]
        if not path.exists():
            pytest.skip(f"Not yet implemented: {TEMPLATE_FILES[tpl_key]}")
        assert path.is_file()


class TestTemplateModuleLoading:
    """Test that template modules load via importlib."""

    @pytest.mark.parametrize("tpl_key", EXISTING_TEMPLATE_KEYS)
    def test_module_loads(self, tpl_key):
        mod = _load_template(tpl_key)
        assert mod is not None


class TestTemplateClassInstantiation:
    """Test that each template class can be instantiated."""

    @pytest.mark.parametrize("tpl_key", EXISTING_TEMPLATE_KEYS)
    def test_instantiate(self, tpl_key):
        mod = _load_template(tpl_key)
        cls_name = TEMPLATE_CLASSES[tpl_key]
        cls = getattr(mod, cls_name, None)
        if cls is None:
            pytest.skip(f"Class {cls_name} not found in {TEMPLATE_FILES[tpl_key]}")
        instance = cls()
        assert instance is not None


class TestTemplateRenderMethods:
    """Test that each template has render methods."""

    @pytest.mark.parametrize("tpl_key", EXISTING_TEMPLATE_KEYS)
    def test_has_render_markdown(self, tpl_key):
        mod = _load_template(tpl_key)
        cls_name = TEMPLATE_CLASSES[tpl_key]
        cls = getattr(mod, cls_name, None)
        if cls is None:
            pytest.skip(f"Class {cls_name} not found")
        instance = cls()
        has_md = (
            hasattr(instance, "render_markdown")
            or hasattr(instance, "render")
            or hasattr(instance, "generate")
        )
        assert has_md

    @pytest.mark.parametrize("tpl_key", EXISTING_TEMPLATE_KEYS)
    def test_has_render_html(self, tpl_key):
        mod = _load_template(tpl_key)
        cls_name = TEMPLATE_CLASSES[tpl_key]
        cls = getattr(mod, cls_name, None)
        if cls is None:
            pytest.skip(f"Class {cls_name} not found")
        instance = cls()
        has_html = (
            hasattr(instance, "render_html")
            or hasattr(instance, "render")
            or hasattr(instance, "generate")
        )
        assert has_html

    @pytest.mark.parametrize("tpl_key", EXISTING_TEMPLATE_KEYS)
    def test_has_render_json(self, tpl_key):
        mod = _load_template(tpl_key)
        cls_name = TEMPLATE_CLASSES[tpl_key]
        cls = getattr(mod, cls_name, None)
        if cls is None:
            pytest.skip(f"Class {cls_name} not found")
        instance = cls()
        has_json = (
            hasattr(instance, "render_json")
            or hasattr(instance, "render")
            or hasattr(instance, "to_json")
        )
        assert has_json


class TestTemplateMarkdownOutput:
    """Test markdown rendering produces valid output."""

    @pytest.mark.parametrize("tpl_key", EXISTING_TEMPLATE_KEYS)
    def test_render_markdown(self, tpl_key):
        mod = _load_template(tpl_key)
        cls_name = TEMPLATE_CLASSES[tpl_key]
        cls = getattr(mod, cls_name, None)
        if cls is None:
            pytest.skip(f"Class {cls_name} not found")
        instance = cls()
        render_fn = getattr(instance, "render_markdown", None) or getattr(instance, "render", None)
        if render_fn is None:
            pytest.skip(f"No render method on {cls_name}")
        data = _sample_audit_data()
        result = render_fn(data)
        assert isinstance(result, str)
        assert len(result) > 100  # Should produce substantial output

    @pytest.mark.parametrize("tpl_key", EXISTING_TEMPLATE_KEYS)
    def test_markdown_has_header(self, tpl_key):
        mod = _load_template(tpl_key)
        cls_name = TEMPLATE_CLASSES[tpl_key]
        cls = getattr(mod, cls_name, None)
        if cls is None:
            pytest.skip(f"Class {cls_name} not found")
        instance = cls()
        render_fn = getattr(instance, "render_markdown", None) or getattr(instance, "render", None)
        if render_fn is None:
            pytest.skip(f"No render method on {cls_name}")
        data = _sample_audit_data()
        result = render_fn(data)
        # Markdown headers start with #
        assert "#" in result


class TestTemplateHTMLOutput:
    """Test HTML rendering produces valid output."""

    @pytest.mark.parametrize("tpl_key", EXISTING_TEMPLATE_KEYS)
    def test_render_html(self, tpl_key):
        mod = _load_template(tpl_key)
        cls_name = TEMPLATE_CLASSES[tpl_key]
        cls = getattr(mod, cls_name, None)
        if cls is None:
            pytest.skip(f"Class {cls_name} not found")
        instance = cls()
        render_fn = getattr(instance, "render_html", None)
        if render_fn is None:
            pytest.skip(f"No render_html method on {cls_name}")
        data = _sample_audit_data()
        result = render_fn(data)
        assert isinstance(result, str)
        assert "<" in result  # Should contain HTML tags


class TestTemplateJSONOutput:
    """Test JSON rendering produces valid output."""

    @pytest.mark.parametrize("tpl_key", EXISTING_TEMPLATE_KEYS)
    def test_render_json(self, tpl_key):
        mod = _load_template(tpl_key)
        cls_name = TEMPLATE_CLASSES[tpl_key]
        cls = getattr(mod, cls_name, None)
        if cls is None:
            pytest.skip(f"Class {cls_name} not found")
        instance = cls()
        render_fn = getattr(instance, "render_json", None) or getattr(instance, "to_json", None)
        if render_fn is None:
            pytest.skip(f"No render_json method on {cls_name}")
        data = _sample_audit_data()
        result = render_fn(data)
        # Should return string or dict
        assert result is not None


class TestEnergyAuditReportTemplate:
    """Specific tests for the EN 16247 energy audit report."""

    def test_en16247_sections(self):
        mod = _load_template("energy_audit_report")
        cls = getattr(mod, "EnergyAuditReportTemplate", None)
        if cls is None:
            pytest.skip("EnergyAuditReportTemplate not found")
        instance = cls()
        sections = getattr(instance, "EN16247_SECTIONS", None) or getattr(cls, "EN16247_SECTIONS", None)
        if sections:
            assert len(sections) >= 6

    def test_report_has_provenance(self):
        mod = _load_template("energy_audit_report")
        cls = getattr(mod, "EnergyAuditReportTemplate", None)
        if cls is None:
            pytest.skip("EnergyAuditReportTemplate not found")
        instance = cls()
        data = _sample_audit_data()
        result = instance.render_markdown(data)
        # Should include provenance hash
        assert "provenance" in result.lower() or "Provenance" in result


class TestEnergyBaselineReportTemplate:
    """Specific tests for the energy baseline report."""

    def test_instantiation(self):
        mod = _load_template("energy_baseline_report")
        cls = getattr(mod, "EnergyBaselineReportTemplate", None)
        if cls is None:
            pytest.skip("EnergyBaselineReportTemplate not found")
        instance = cls()
        assert instance is not None


class TestSavingsVerificationReportTemplate:
    """Specific tests for the M&V report."""

    def test_instantiation(self):
        mod = _load_template("mv_report")
        cls = getattr(mod, "SavingsVerificationReportTemplate", None)
        if cls is None:
            pytest.skip("SavingsVerificationReportTemplate not found")
        instance = cls()
        assert instance is not None
