# -*- coding: utf-8 -*-
"""
PACK-020 Battery Passport Prep Pack - Test Configuration
=========================================================

Shared test infrastructure for all PACK-020 test modules.
Provides dynamic module loading, path constants, and reusable
sample data fixtures for EU Battery Regulation testing.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-020 Battery Passport Prep Pack
Date:    March 2026
"""

import importlib.util
import sys
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Path Constants
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"
WORKFLOWS_DIR = PACK_ROOT / "workflows"
TEMPLATES_DIR = PACK_ROOT / "templates"
INTEGRATIONS_DIR = PACK_ROOT / "integrations"
CONFIG_DIR = PACK_ROOT / "config"
PRESETS_DIR = CONFIG_DIR / "presets"
DEMO_DIR = CONFIG_DIR / "demo"
MANIFEST_PATH = PACK_ROOT / "pack.yaml"

# ---------------------------------------------------------------------------
# File Mappings
# ---------------------------------------------------------------------------

ENGINE_FILES: Dict[str, str] = {
    "carbon_footprint": "carbon_footprint_engine.py",
    "recycled_content": "recycled_content_engine.py",
    "battery_passport": "battery_passport_engine.py",
    "performance_durability": "performance_durability_engine.py",
    "supply_chain_dd": "supply_chain_dd_engine.py",
    "labelling_compliance": "labelling_compliance_engine.py",
    "end_of_life": "end_of_life_engine.py",
    "conformity_assessment": "conformity_assessment_engine.py",
}

ENGINE_CLASSES: Dict[str, str] = {
    "carbon_footprint": "CarbonFootprintEngine",
    "recycled_content": "RecycledContentEngine",
    "battery_passport": "BatteryPassportEngine",
    "performance_durability": "PerformanceDurabilityEngine",
    "supply_chain_dd": "SupplyChainDDEngine",
    "labelling_compliance": "LabellingComplianceEngine",
    "end_of_life": "EndOfLifeEngine",
    "conformity_assessment": "ConformityAssessmentEngine",
}

WORKFLOW_FILES: Dict[str, str] = {
    "carbon_footprint_assessment": "carbon_footprint_assessment_workflow.py",
    "recycled_content_tracking": "recycled_content_tracking_workflow.py",
    "passport_compilation": "passport_compilation_workflow.py",
    "performance_testing": "performance_testing_workflow.py",
    "due_diligence_assessment": "due_diligence_assessment_workflow.py",
    "labelling_verification": "labelling_verification_workflow.py",
    "end_of_life_planning": "end_of_life_planning_workflow.py",
    "regulatory_submission": "regulatory_submission_workflow.py",
}

WORKFLOW_CLASSES: Dict[str, str] = {
    "carbon_footprint_assessment": "CarbonFootprintWorkflow",
    "recycled_content_tracking": "RecycledContentWorkflow",
    "passport_compilation": "PassportCompilationWorkflow",
    "performance_testing": "PerformanceTestingWorkflow",
    "due_diligence_assessment": "DueDiligenceAssessmentWorkflow",
    "labelling_verification": "LabellingVerificationWorkflow",
    "end_of_life_planning": "EndOfLifePlanningWorkflow",
    "regulatory_submission": "RegulatorySubmissionWorkflow",
}

TEMPLATE_FILES: Dict[str, str] = {
    "carbon_footprint_declaration": "carbon_footprint_declaration.py",
    "recycled_content_report": "recycled_content_report.py",
    "battery_passport_report": "battery_passport_report.py",
    "performance_report": "performance_report.py",
    "due_diligence_report": "due_diligence_report.py",
    "labelling_compliance_report": "labelling_compliance_report.py",
    "end_of_life_report": "end_of_life_report.py",
    "battery_regulation_scorecard": "battery_regulation_scorecard.py",
}

TEMPLATE_CLASSES: Dict[str, str] = {
    "carbon_footprint_declaration": "CarbonFootprintDeclarationTemplate",
    "recycled_content_report": "RecycledContentReportTemplate",
    "battery_passport_report": "BatteryPassportReportTemplate",
    "performance_report": "PerformanceReportTemplate",
    "due_diligence_report": "DueDiligenceReportTemplate",
    "labelling_compliance_report": "LabellingComplianceReportTemplate",
    "end_of_life_report": "EndOfLifeReportTemplate",
    "battery_regulation_scorecard": "BatteryRegulationScorecardTemplate",
}

INTEGRATION_FILES: Dict[str, str] = {
    "pack_orchestrator": "pack_orchestrator.py",
    "mrv_bridge": "mrv_bridge.py",
    "csrd_pack_bridge": "csrd_pack_bridge.py",
    "supply_chain_bridge": "supply_chain_bridge.py",
    "eudr_bridge": "eudr_bridge.py",
    "taxonomy_bridge": "taxonomy_bridge.py",
    "csddd_bridge": "csddd_bridge.py",
    "data_bridge": "data_bridge.py",
    "health_check": "health_check.py",
    "setup_wizard": "setup_wizard.py",
}

INTEGRATION_CLASSES: Dict[str, str] = {
    "pack_orchestrator": "BatteryPassportOrchestrator",
    "mrv_bridge": "MRVBridge",
    "csrd_pack_bridge": "CSRDPackBridge",
    "supply_chain_bridge": "SupplyChainBridge",
    "eudr_bridge": "EUDRBridge",
    "taxonomy_bridge": "TaxonomyBridge",
    "csddd_bridge": "CSDDDBridge",
    "data_bridge": "DataBridge",
    "health_check": "BatteryPassportHealthCheck",
    "setup_wizard": "BatteryPassportSetupWizard",
}

PRESET_FILES: List[str] = [
    "ev_battery.yaml",
    "industrial_storage.yaml",
    "lmt_battery.yaml",
    "portable_battery.yaml",
    "sli_battery.yaml",
    "cell_manufacturer.yaml",
]


# ---------------------------------------------------------------------------
# Dynamic Module Loader
# ---------------------------------------------------------------------------


def _load_module(file_path: Path, namespace: str) -> Any:
    """Load a Python module from file path using importlib.

    Uses ``pack020_test.*`` namespace to avoid collisions with other packs.
    Modules are cached in ``sys.modules`` for session reuse.
    """
    mod_name = f"pack020_test.{namespace}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_engine(name: str) -> Any:
    """Load a PACK-020 engine module by short name."""
    filename = ENGINE_FILES.get(name)
    if filename is None:
        raise KeyError(f"Unknown engine: {name}")
    return _load_module(ENGINES_DIR / filename, f"engines.{name}")


def _load_workflow(name: str) -> Any:
    """Load a PACK-020 workflow module by short name."""
    filename = WORKFLOW_FILES.get(name)
    if filename is None:
        raise KeyError(f"Unknown workflow: {name}")
    return _load_module(WORKFLOWS_DIR / filename, f"workflows.{name}")


def _load_template(name: str) -> Any:
    """Load a PACK-020 template module by short name."""
    filename = TEMPLATE_FILES.get(name)
    if filename is None:
        raise KeyError(f"Unknown template: {name}")
    return _load_module(TEMPLATES_DIR / filename, f"templates.{name}")


def _load_integration(name: str) -> Any:
    """Load a PACK-020 integration module by short name."""
    filename = INTEGRATION_FILES.get(name)
    if filename is None:
        raise KeyError(f"Unknown integration: {name}")
    return _load_module(INTEGRATIONS_DIR / filename, f"integrations.{name}")


# ---------------------------------------------------------------------------
# Sample Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_battery_profile() -> Dict[str, Any]:
    """Sample EV battery profile for testing."""
    return {
        "battery_id": "BAT-EV-2027-001",
        "category": "EV",
        "chemistry": "NMC811",
        "manufacturer": "EuroBattery GmbH",
        "manufacturer_id": "EU-MFR-001",
        "manufacturing_plant": "Gigafactory Berlin",
        "plant_country": "DE",
        "model": "EB-75-NMC811",
        "batch_number": "BATCH-2027-Q1-001",
        "serial_number": "SN-EV-2027-00001",
        "weight_kg": Decimal("450"),
        "capacity_ah": Decimal("75"),
        "voltage_nominal": Decimal("400"),
        "energy_kwh": Decimal("75"),
        "production_date": "2027-01-15",
        "placing_on_market_date": "2027-03-01",
    }


@pytest.fixture
def sample_carbon_footprint() -> Dict[str, Any]:
    """Sample carbon footprint data for lifecycle assessment."""
    return {
        "battery_id": "BAT-EV-2027-001",
        "total_co2e_kg": Decimal("5625"),
        "per_kwh_co2e_kg": Decimal("75"),
        "lifecycle_breakdown": {
            "raw_material_extraction": Decimal("2812.5"),
            "manufacturing": Decimal("1406.25"),
            "distribution": Decimal("281.25"),
            "end_of_life": Decimal("1125"),
        },
        "methodology": "PEFCR_BATTERIES_2025",
        "calculation_date": "2027-02-01",
        "third_party_verified": True,
        "verifier": "TUV Rheinland",
    }


@pytest.fixture
def sample_recycled_content() -> Dict[str, Any]:
    """Sample recycled content data for Art 8 compliance."""
    return {
        "battery_id": "BAT-EV-2027-001",
        "reporting_year": 2027,
        "cobalt_total_kg": Decimal("12.5"),
        "cobalt_recycled_kg": Decimal("2.5"),
        "cobalt_recycled_pct": Decimal("20"),
        "lithium_total_kg": Decimal("8.0"),
        "lithium_recycled_kg": Decimal("0.64"),
        "lithium_recycled_pct": Decimal("8"),
        "nickel_total_kg": Decimal("35.0"),
        "nickel_recycled_kg": Decimal("3.5"),
        "nickel_recycled_pct": Decimal("10"),
        "lead_total_kg": Decimal("0"),
        "lead_recycled_kg": Decimal("0"),
        "lead_recycled_pct": Decimal("0"),
    }


@pytest.fixture
def sample_performance_metrics() -> Dict[str, Any]:
    """Sample battery performance and durability metrics."""
    return {
        "battery_id": "BAT-EV-2027-001",
        "rated_capacity_ah": Decimal("75"),
        "min_capacity_ah": Decimal("60"),
        "remaining_capacity_pct": Decimal("95"),
        "voltage_nominal": Decimal("400"),
        "voltage_min": Decimal("320"),
        "voltage_max": Decimal("440"),
        "power_capability_w": Decimal("150000"),
        "cycle_life": 1500,
        "calendar_life_years": 10,
        "efficiency_pct": Decimal("95"),
        "internal_resistance_mohm": Decimal("50"),
        "soh_pct": Decimal("98"),
        "soc_pct": Decimal("80"),
        "c_rate": Decimal("2.0"),
        "temperature_min_c": -20,
        "temperature_max_c": 45,
        "test_date": "2027-02-15",
    }


@pytest.fixture
def sample_supply_chain() -> List[Dict[str, Any]]:
    """Sample supply chain data for due diligence assessment."""
    return [
        {
            "supplier_id": "SUP-001",
            "name": "CobaltMine Corp",
            "material": "COBALT",
            "country": "CD",
            "tier": 3,
            "risk_level": "HIGH",
            "oecd_compliant": False,
            "third_party_audited": False,
        },
        {
            "supplier_id": "SUP-002",
            "name": "LithiumEx Chile",
            "material": "LITHIUM",
            "country": "CL",
            "tier": 2,
            "risk_level": "MEDIUM",
            "oecd_compliant": True,
            "third_party_audited": True,
        },
        {
            "supplier_id": "SUP-003",
            "name": "NickelPure Finland",
            "material": "NICKEL",
            "country": "FI",
            "tier": 1,
            "risk_level": "LOW",
            "oecd_compliant": True,
            "third_party_audited": True,
        },
        {
            "supplier_id": "SUP-004",
            "name": "GraphiteWorks China",
            "material": "NATURAL_GRAPHITE",
            "country": "CN",
            "tier": 2,
            "risk_level": "MEDIUM",
            "oecd_compliant": True,
            "third_party_audited": False,
        },
    ]


@pytest.fixture
def sample_label_requirements() -> List[Dict[str, Any]]:
    """Sample labelling requirements for compliance check."""
    return [
        {"element": "CE_MARKING", "required": True, "present": True},
        {"element": "QR_CODE", "required": True, "present": True},
        {"element": "COLLECTION_SYMBOL", "required": True, "present": True},
        {"element": "CAPACITY_LABEL", "required": True, "present": True},
        {"element": "HAZARDOUS_SUBSTANCE", "required": False, "present": False},
        {"element": "BATTERY_CHEMISTRY", "required": True, "present": False},
        {"element": "CARBON_FOOTPRINT", "required": True, "present": True},
        {"element": "SEPARATE_COLLECTION", "required": True, "present": True},
    ]


@pytest.fixture
def sample_end_of_life() -> Dict[str, Any]:
    """Sample end-of-life management data."""
    return {
        "battery_category": "EV",
        "collection_rate_pct": Decimal("65"),
        "recycling_efficiency_pct": Decimal("70"),
        "material_recovery": {
            "cobalt_pct": Decimal("90"),
            "lithium_pct": Decimal("55"),
            "nickel_pct": Decimal("90"),
            "copper_pct": Decimal("90"),
        },
        "dismantling_info_available": True,
        "safety_info_available": True,
        "second_life_assessment": True,
        "reporting_year": 2027,
    }


@pytest.fixture
def sample_conformity_data() -> Dict[str, Any]:
    """Sample conformity assessment data."""
    return {
        "battery_id": "BAT-EV-2027-001",
        "conformity_module": "MODULE_A",
        "eu_declaration_of_conformity": True,
        "technical_documentation": True,
        "ce_marking_applied": True,
        "notified_body_required": False,
        "notified_body_id": None,
        "harmonised_standards_applied": ["EN 62660-1", "EN 62660-2", "EN 62660-3"],
        "assessment_date": "2027-02-20",
    }
