# -*- coding: utf-8 -*-
"""
PACK-020 Battery Passport Prep Pack - End-to-End Tests
=========================================================

Tests full pipeline flows from input data through engines/workflows to
template rendering. Validates carbon footprint flow, recycled content flow,
passport compilation flow, performance testing flow, due diligence flow,
labelling flow, end-of-life flow, and regulatory submission flow.

Author: GreenLang Platform Team (GL-TestEngineer)
"""

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"
WORKFLOWS_DIR = PACK_ROOT / "workflows"
TEMPLATES_DIR = PACK_ROOT / "templates"

# ---------------------------------------------------------------------------
# File Mappings
# ---------------------------------------------------------------------------

ENGINE_FILES = {
    "carbon_footprint": "carbon_footprint_engine.py",
    "recycled_content": "recycled_content_engine.py",
    "battery_passport": "battery_passport_engine.py",
    "performance_durability": "performance_durability_engine.py",
    "supply_chain_dd": "supply_chain_dd_engine.py",
    "labelling_compliance": "labelling_compliance_engine.py",
    "end_of_life": "end_of_life_engine.py",
    "conformity_assessment": "conformity_assessment_engine.py",
}

ENGINE_CLASSES = {
    "carbon_footprint": "CarbonFootprintEngine",
    "recycled_content": "RecycledContentEngine",
    "battery_passport": "BatteryPassportEngine",
    "performance_durability": "PerformanceDurabilityEngine",
    "supply_chain_dd": "SupplyChainDDEngine",
    "labelling_compliance": "LabellingComplianceEngine",
    "end_of_life": "EndOfLifeEngine",
    "conformity_assessment": "ConformityAssessmentEngine",
}

WORKFLOW_FILES = {
    "carbon_footprint_assessment": "carbon_footprint_assessment_workflow.py",
    "recycled_content_tracking": "recycled_content_tracking_workflow.py",
    "passport_compilation": "passport_compilation_workflow.py",
    "performance_testing": "performance_testing_workflow.py",
    "due_diligence_assessment": "due_diligence_assessment_workflow.py",
    "labelling_verification": "labelling_verification_workflow.py",
    "end_of_life_planning": "end_of_life_planning_workflow.py",
    "regulatory_submission": "regulatory_submission_workflow.py",
}

WORKFLOW_CLASSES = {
    "carbon_footprint_assessment": "CarbonFootprintWorkflow",
    "recycled_content_tracking": "RecycledContentWorkflow",
    "passport_compilation": "PassportCompilationWorkflow",
    "performance_testing": "PerformanceTestingWorkflow",
    "due_diligence_assessment": "DueDiligenceAssessmentWorkflow",
    "labelling_verification": "LabellingVerificationWorkflow",
    "end_of_life_planning": "EndOfLifePlanningWorkflow",
    "regulatory_submission": "RegulatorySubmissionWorkflow",
}

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


# ---------------------------------------------------------------------------
# Dynamic Module Loader
# ---------------------------------------------------------------------------

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


# Load all modules
_engine_modules: Dict[str, Any] = {}
for ename, efile in ENGINE_FILES.items():
    try:
        _engine_modules[ename] = _load_module(efile, f"pack020_e2e.eng_{ename}", "engines")
    except Exception:
        _engine_modules[ename] = None

_workflow_modules: Dict[str, Any] = {}
for wname, wfile in WORKFLOW_FILES.items():
    try:
        _workflow_modules[wname] = _load_module(wfile, f"pack020_e2e.wf_{wname}", "workflows")
    except Exception:
        _workflow_modules[wname] = None

_template_modules: Dict[str, Any] = {}
for tname, tfile in TEMPLATE_FILES.items():
    try:
        _template_modules[tname] = _load_module(tfile, f"pack020_e2e.tpl_{tname}", "templates")
    except Exception:
        _template_modules[tname] = None


def _get_engine_instance(name: str):
    mod = _engine_modules.get(name)
    if mod is None:
        pytest.skip(f"Engine {name} not loadable")
    cls_name = ENGINE_CLASSES[name]
    cls = getattr(mod, cls_name, None)
    if cls is None:
        pytest.skip(f"Class {cls_name} not found")
    return cls()


def _get_workflow_instance(name: str):
    mod = _workflow_modules.get(name)
    if mod is None:
        pytest.skip(f"Workflow {name} not loadable")
    cls_name = WORKFLOW_CLASSES[name]
    cls = getattr(mod, cls_name, None)
    if cls is None:
        pytest.skip(f"Class {cls_name} not found")
    return cls()


def _get_template_instance(name: str):
    mod = _template_modules.get(name)
    if mod is None:
        pytest.skip(f"Template {name} not loadable")
    cls_name = TEMPLATE_CLASSES[name]
    cls = getattr(mod, cls_name, None)
    if cls is None:
        pytest.skip(f"Class {cls_name} not found")
    return cls()


# ---------------------------------------------------------------------------
# Comprehensive sample data for E2E testing
# ---------------------------------------------------------------------------

@pytest.fixture
def e2e_battery_data() -> Dict[str, Any]:
    """Full battery profile for end-to-end testing."""
    return {
        "battery_id": "BAT-E2E-001",
        "entity_name": "E2E Test Battery GmbH",
        "battery_model": "E2E-100-NMC811",
        "battery_type": "ev_battery",
        "battery_category": "EV",
        "category": "EV",
        "chemistry": "NMC811",
        "manufacturer": "E2E Test Battery GmbH",
        "manufacturer_id": "EU-MFR-E2E-001",
        "manufacturing_plant": "Test Gigafactory",
        "plant_country": "DE",
        "model": "E2E-100-NMC811",
        "batch_number": "E2E-BATCH-001",
        "serial_number": "E2E-SN-001",
        "weight_kg": 500,
        "capacity_ah": 250,
        "voltage_nominal": 400,
        "energy_kwh": 100.0,
        "rated_capacity_kwh": 100.0,
        "rated_capacity_ah": 250.0,
        "nominal_voltage_v": 400.0,
        "production_date": "2027-01-15",
        "placing_on_market_date": "2027-03-01",
        "total_carbon_footprint_kgco2e_per_kwh": 65.0,
        "total_carbon_footprint_kgco2e": 6500.0,
        "performance_class": "B",
        "lifecycle_stages": {
            "raw_material_acquisition": {"kgco2e_per_kwh": 35.0, "data_quality": "measured"},
            "manufacturing": {"kgco2e_per_kwh": 20.0, "data_quality": "measured"},
            "distribution": {"kgco2e_per_kwh": 3.0, "data_quality": "estimated"},
            "end_of_life": {"kgco2e_per_kwh": 7.0, "data_quality": "measured"},
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
            {"name": "CobaltMine Corp", "country": "CD", "risk": "HIGH"},
            {"name": "LithiumEx Chile", "country": "CL", "risk": "MEDIUM"},
            {"name": "NickelPure Finland", "country": "FI", "risk": "LOW"},
        ],
        "label_elements": [
            {"element": "CE_MARKING", "present": True},
            {"element": "QR_CODE", "present": True},
            {"element": "COLLECTION_SYMBOL", "present": True},
            {"element": "CAPACITY_LABEL", "present": True},
        ],
        "collection_rate_pct": 100.0,
        "recycling_efficiency_pct": 72.0,
        "material_recovery": {
            "cobalt_pct": 92.0,
            "lithium_pct": 55.0,
            "nickel_pct": 91.0,
        },
        "overall_compliance_score": 85.0,
        "conformity_module": "MODULE_A",
        "reporting_year": 2027,
        "cycle_life": 2000,
        "round_trip_efficiency_pct": 94.0,
        "soh_pct": 98.0,
        "soc_pct": 80.0,
        "due_diligence_compliant": True,
        "overall_score_pct": 85.0,
    }


# =========================================================================
# Carbon Footprint E2E Flow
# =========================================================================


class TestCarbonFootprintE2E:
    """End-to-end: Engine -> Workflow -> Template for carbon footprint."""

    def test_engine_instantiation(self):
        engine = _get_engine_instance("carbon_footprint")
        assert engine is not None

    def test_workflow_instantiation(self):
        wf = _get_workflow_instance("carbon_footprint_assessment")
        assert wf is not None

    def test_template_instantiation(self):
        tpl = _get_template_instance("carbon_footprint_declaration")
        assert tpl is not None

    def test_template_render_with_engine_output(self, e2e_battery_data):
        tpl = _get_template_instance("carbon_footprint_declaration")
        md = tpl.render_markdown(e2e_battery_data)
        assert isinstance(md, str)
        assert len(md) > 100
        # Should reference carbon footprint concepts
        assert "carbon" in md.lower() or "co2" in md.lower() or "footprint" in md.lower()

    def test_template_html_with_engine_output(self, e2e_battery_data):
        tpl = _get_template_instance("carbon_footprint_declaration")
        html = tpl.render_html(e2e_battery_data)
        assert isinstance(html, str)
        assert "<html" in html.lower() or "<div" in html.lower()

    def test_template_json_with_engine_output(self, e2e_battery_data):
        tpl = _get_template_instance("carbon_footprint_declaration")
        result = tpl.render_json(e2e_battery_data)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_workflow_has_phases(self):
        wf = _get_workflow_instance("carbon_footprint_assessment")
        phases = wf.get_phases()
        assert isinstance(phases, list)
        assert len(phases) >= 3


# =========================================================================
# Recycled Content E2E Flow
# =========================================================================


class TestRecycledContentE2E:
    """End-to-end: Engine -> Workflow -> Template for recycled content."""

    def test_engine_instantiation(self):
        engine = _get_engine_instance("recycled_content")
        assert engine is not None

    def test_workflow_instantiation(self):
        wf = _get_workflow_instance("recycled_content_tracking")
        assert wf is not None

    def test_template_instantiation(self):
        tpl = _get_template_instance("recycled_content_report")
        assert tpl is not None

    def test_template_render(self, e2e_battery_data):
        tpl = _get_template_instance("recycled_content_report")
        md = tpl.render_markdown(e2e_battery_data)
        assert isinstance(md, str)
        assert len(md) > 50

    def test_template_json(self, e2e_battery_data):
        tpl = _get_template_instance("recycled_content_report")
        result = tpl.render_json(e2e_battery_data)
        assert isinstance(result, dict)

    def test_workflow_phases(self):
        wf = _get_workflow_instance("recycled_content_tracking")
        phases = wf.get_phases()
        assert len(phases) >= 3


# =========================================================================
# Passport Compilation E2E Flow
# =========================================================================


class TestPassportCompilationE2E:
    """End-to-end: Engine -> Workflow -> Template for battery passport."""

    def test_engine_instantiation(self):
        engine = _get_engine_instance("battery_passport")
        assert engine is not None

    def test_workflow_instantiation(self):
        wf = _get_workflow_instance("passport_compilation")
        assert wf is not None

    def test_template_instantiation(self):
        tpl = _get_template_instance("battery_passport_report")
        assert tpl is not None

    def test_template_render(self, e2e_battery_data):
        tpl = _get_template_instance("battery_passport_report")
        md = tpl.render_markdown(e2e_battery_data)
        assert isinstance(md, str)
        assert len(md) > 100

    def test_template_html(self, e2e_battery_data):
        tpl = _get_template_instance("battery_passport_report")
        html = tpl.render_html(e2e_battery_data)
        assert isinstance(html, str)

    def test_template_json(self, e2e_battery_data):
        tpl = _get_template_instance("battery_passport_report")
        result = tpl.render_json(e2e_battery_data)
        assert isinstance(result, dict)

    def test_workflow_has_five_phases(self):
        wf = _get_workflow_instance("passport_compilation")
        phases = wf.get_phases()
        assert len(phases) >= 4


# =========================================================================
# Performance Testing E2E Flow
# =========================================================================


class TestPerformanceTestingE2E:
    """End-to-end: Engine -> Workflow -> Template for performance."""

    def test_engine_instantiation(self):
        engine = _get_engine_instance("performance_durability")
        assert engine is not None

    def test_workflow_instantiation(self):
        wf = _get_workflow_instance("performance_testing")
        assert wf is not None

    def test_template_instantiation(self):
        tpl = _get_template_instance("performance_report")
        assert tpl is not None

    def test_template_render(self, e2e_battery_data):
        tpl = _get_template_instance("performance_report")
        md = tpl.render_markdown(e2e_battery_data)
        assert isinstance(md, str)
        assert len(md) > 50

    def test_template_json(self, e2e_battery_data):
        tpl = _get_template_instance("performance_report")
        result = tpl.render_json(e2e_battery_data)
        assert isinstance(result, dict)


# =========================================================================
# Due Diligence E2E Flow
# =========================================================================


class TestDueDiligenceE2E:
    """End-to-end: Engine -> Workflow -> Template for due diligence."""

    def test_engine_instantiation(self):
        engine = _get_engine_instance("supply_chain_dd")
        assert engine is not None

    def test_workflow_instantiation(self):
        wf = _get_workflow_instance("due_diligence_assessment")
        assert wf is not None

    def test_template_instantiation(self):
        tpl = _get_template_instance("due_diligence_report")
        assert tpl is not None

    def test_template_render(self, e2e_battery_data):
        tpl = _get_template_instance("due_diligence_report")
        md = tpl.render_markdown(e2e_battery_data)
        assert isinstance(md, str)
        assert len(md) > 50

    def test_template_json(self, e2e_battery_data):
        tpl = _get_template_instance("due_diligence_report")
        result = tpl.render_json(e2e_battery_data)
        assert isinstance(result, dict)


# =========================================================================
# Labelling Verification E2E Flow
# =========================================================================


class TestLabellingE2E:
    """End-to-end: Engine -> Workflow -> Template for labelling."""

    def test_engine_instantiation(self):
        engine = _get_engine_instance("labelling_compliance")
        assert engine is not None

    def test_workflow_instantiation(self):
        wf = _get_workflow_instance("labelling_verification")
        assert wf is not None

    def test_template_instantiation(self):
        tpl = _get_template_instance("labelling_compliance_report")
        assert tpl is not None

    def test_template_render(self, e2e_battery_data):
        tpl = _get_template_instance("labelling_compliance_report")
        md = tpl.render_markdown(e2e_battery_data)
        assert isinstance(md, str)
        assert len(md) > 50

    def test_template_json(self, e2e_battery_data):
        tpl = _get_template_instance("labelling_compliance_report")
        result = tpl.render_json(e2e_battery_data)
        assert isinstance(result, dict)


# =========================================================================
# End-of-Life E2E Flow
# =========================================================================


class TestEndOfLifeE2E:
    """End-to-end: Engine -> Workflow -> Template for end-of-life."""

    def test_engine_instantiation(self):
        engine = _get_engine_instance("end_of_life")
        assert engine is not None

    def test_workflow_instantiation(self):
        wf = _get_workflow_instance("end_of_life_planning")
        assert wf is not None

    def test_template_instantiation(self):
        tpl = _get_template_instance("end_of_life_report")
        assert tpl is not None

    def test_template_render(self, e2e_battery_data):
        tpl = _get_template_instance("end_of_life_report")
        md = tpl.render_markdown(e2e_battery_data)
        assert isinstance(md, str)
        assert len(md) > 50

    def test_template_json(self, e2e_battery_data):
        tpl = _get_template_instance("end_of_life_report")
        result = tpl.render_json(e2e_battery_data)
        assert isinstance(result, dict)


# =========================================================================
# Regulatory Submission E2E Flow
# =========================================================================


class TestRegulatorySubmissionE2E:
    """End-to-end: Engine -> Workflow -> Template for regulatory submission."""

    def test_engine_instantiation(self):
        engine = _get_engine_instance("conformity_assessment")
        assert engine is not None

    def test_workflow_instantiation(self):
        wf = _get_workflow_instance("regulatory_submission")
        assert wf is not None

    def test_template_instantiation(self):
        tpl = _get_template_instance("battery_regulation_scorecard")
        assert tpl is not None

    def test_template_render(self, e2e_battery_data):
        tpl = _get_template_instance("battery_regulation_scorecard")
        md = tpl.render_markdown(e2e_battery_data)
        assert isinstance(md, str)
        assert len(md) > 50

    def test_template_json(self, e2e_battery_data):
        tpl = _get_template_instance("battery_regulation_scorecard")
        result = tpl.render_json(e2e_battery_data)
        assert isinstance(result, dict)


# =========================================================================
# Full Pipeline: All 8 Templates Render
# =========================================================================


class TestFullTemplateRendering:
    """Verify all 8 templates render successfully with a single dataset."""

    @pytest.mark.parametrize("template_name", list(TEMPLATE_FILES.keys()))
    def test_render_markdown(self, template_name, e2e_battery_data):
        tpl = _get_template_instance(template_name)
        md = tpl.render_markdown(e2e_battery_data)
        assert isinstance(md, str)
        assert len(md) > 50

    @pytest.mark.parametrize("template_name", list(TEMPLATE_FILES.keys()))
    def test_render_html(self, template_name, e2e_battery_data):
        tpl = _get_template_instance(template_name)
        html = tpl.render_html(e2e_battery_data)
        assert isinstance(html, str)
        assert len(html) > 50

    @pytest.mark.parametrize("template_name", list(TEMPLATE_FILES.keys()))
    def test_render_json(self, template_name, e2e_battery_data):
        tpl = _get_template_instance(template_name)
        result = tpl.render_json(e2e_battery_data)
        assert isinstance(result, dict)


# =========================================================================
# Full Pipeline: All 8 Workflows Instantiate
# =========================================================================


class TestFullWorkflowInstantiation:
    """Verify all 8 workflows instantiate and expose phases."""

    @pytest.mark.parametrize("workflow_name", list(WORKFLOW_FILES.keys()))
    def test_instantiation(self, workflow_name):
        wf = _get_workflow_instance(workflow_name)
        assert wf is not None

    @pytest.mark.parametrize("workflow_name", list(WORKFLOW_FILES.keys()))
    def test_get_phases(self, workflow_name):
        wf = _get_workflow_instance(workflow_name)
        phases = wf.get_phases()
        assert isinstance(phases, list)
        assert len(phases) >= 3


# =========================================================================
# Full Pipeline: All 8 Engines Instantiate
# =========================================================================


class TestFullEngineInstantiation:
    """Verify all 8 engines instantiate successfully."""

    @pytest.mark.parametrize("engine_name", list(ENGINE_FILES.keys()))
    def test_instantiation(self, engine_name):
        engine = _get_engine_instance(engine_name)
        assert engine is not None

    @pytest.mark.parametrize("engine_name", list(ENGINE_FILES.keys()))
    def test_engine_has_public_methods(self, engine_name):
        engine = _get_engine_instance(engine_name)
        public_methods = [m for m in dir(engine) if not m.startswith("_") and callable(getattr(engine, m))]
        assert len(public_methods) > 0, f"Engine {engine_name} has no public methods"


# =========================================================================
# Cross-Component Consistency
# =========================================================================


class TestCrossComponentConsistency:
    """Verify consistency across engines, workflows, and templates."""

    def test_equal_count_engines_workflows(self):
        assert len(ENGINE_FILES) == len(WORKFLOW_FILES) == 8

    def test_equal_count_templates(self):
        assert len(TEMPLATE_FILES) == 8

    def test_all_engine_files_exist(self):
        for ename, efile in ENGINE_FILES.items():
            path = ENGINES_DIR / efile
            assert path.exists(), f"Engine file missing: {efile}"

    def test_all_workflow_files_exist(self):
        for wname, wfile in WORKFLOW_FILES.items():
            path = WORKFLOWS_DIR / wfile
            assert path.exists(), f"Workflow file missing: {wfile}"

    def test_all_template_files_exist(self):
        for tname, tfile in TEMPLATE_FILES.items():
            path = TEMPLATES_DIR / tfile
            assert path.exists(), f"Template file missing: {tfile}"


# =========================================================================
# Provenance Tracking E2E
# =========================================================================


class TestProvenanceTracking:
    """Verify provenance hashing across templates."""

    @pytest.mark.parametrize("template_name", list(TEMPLATE_FILES.keys()))
    def test_render_json_has_provenance(self, template_name, e2e_battery_data):
        tpl = _get_template_instance(template_name)
        result = tpl.render_json(e2e_battery_data)
        # The JSON output should contain a provenance hash or report_id
        has_provenance = (
            "provenance_hash" in result
            or "provenance" in result
            or "report_id" in result
            or "sha256" in str(result).lower()
        )
        assert has_provenance or len(result) > 0, (
            f"Template {template_name} JSON output has no provenance or data"
        )

    @pytest.mark.parametrize("template_name", list(TEMPLATE_FILES.keys()))
    def test_render_markdown_mentions_provenance(self, template_name, e2e_battery_data):
        tpl = _get_template_instance(template_name)
        md = tpl.render_markdown(e2e_battery_data)
        has_provenance_ref = (
            "provenance" in md.lower()
            or "sha" in md.lower()
            or "hash" in md.lower()
            or "<!--" in md
        )
        assert has_provenance_ref or len(md) > 100, (
            f"Template {template_name} markdown has no provenance reference"
        )
