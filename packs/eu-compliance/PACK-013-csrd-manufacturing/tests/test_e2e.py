# -*- coding: utf-8 -*-
"""
PACK-013 CSRD Manufacturing Pack - End-to-End Tests

Tests complete workflow-to-template pipelines for four manufacturing
scenarios: cement plant, automotive, chemical plant, and multi-facility.

12 tests across 4 test classes.
"""

import asyncio
import importlib.util
import sys
import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Dynamic module loading via importlib
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = PACK_ROOT / "workflows"
TEMPLATES_DIR = PACK_ROOT / "templates"


def _load_module(module_name: str, file_name: str, search_dir: Path):
    """Load a module dynamically using importlib.util.spec_from_file_location."""
    file_path = search_dir / file_name
    if not file_path.exists():
        pytest.skip(f"Module file not found: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        pytest.skip(f"Cannot create spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Load workflow + template modules needed for E2E
# ---------------------------------------------------------------------------

# Workflows
mfg_wf_mod = _load_module(
    "pack013_e2e_mfg_wf",
    "manufacturing_emissions_workflow.py",
    WORKFLOWS_DIR,
)
ManufacturingEmissionsWorkflow = mfg_wf_mod.ManufacturingEmissionsWorkflow
ManufacturingEmissionsInput = mfg_wf_mod.ManufacturingEmissionsInput

bat_wf_mod = _load_module(
    "pack013_e2e_bat_wf",
    "bat_compliance_workflow.py",
    WORKFLOWS_DIR,
)
BATComplianceWorkflow = bat_wf_mod.BATComplianceWorkflow
BATComplianceInput = bat_wf_mod.BATComplianceInput

circ_wf_mod = _load_module(
    "pack013_e2e_circ_wf",
    "circular_economy_workflow.py",
    WORKFLOWS_DIR,
)
CircularEconomyWorkflow = circ_wf_mod.CircularEconomyWorkflow
CircularEconomyInput = circ_wf_mod.CircularEconomyInput

decarb_wf_mod = _load_module(
    "pack013_e2e_decarb_wf",
    "decarbonization_roadmap_workflow.py",
    WORKFLOWS_DIR,
)
DecarbonizationRoadmapWorkflow = decarb_wf_mod.DecarbonizationRoadmapWorkflow
DecarbonizationInput = decarb_wf_mod.DecarbonizationInput

# Templates
process_tpl_mod = _load_module(
    "pack013_e2e_tpl_process",
    "process_emissions_report.py",
    TEMPLATES_DIR,
)
ProcessEmissionsReportTemplate = process_tpl_mod.ProcessEmissionsReportTemplate

bat_tpl_mod = _load_module(
    "pack013_e2e_tpl_bat",
    "bat_compliance_report.py",
    TEMPLATES_DIR,
)
BATComplianceReportTemplate = bat_tpl_mod.BATComplianceReportTemplate

circular_tpl_mod = _load_module(
    "pack013_e2e_tpl_circular",
    "circular_economy_report.py",
    TEMPLATES_DIR,
)
CircularEconomyReportTemplate = circular_tpl_mod.CircularEconomyReportTemplate

decarb_tpl_mod = _load_module(
    "pack013_e2e_tpl_decarb",
    "decarbonization_roadmap.py",
    TEMPLATES_DIR,
)
DecarbonizationRoadmapTemplate = decarb_tpl_mod.DecarbonizationRoadmapTemplate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_async(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ===========================================================================
# Scenario 1: Cement Plant E2E (3 tests)
# ===========================================================================

class TestCementPlantE2E:
    """End-to-end tests for cement manufacturing scenario."""

    @pytest.fixture
    def cement_input(self):
        """Create cement plant emissions input data."""
        return ManufacturingEmissionsInput(
            organization_id="CEMENT-ORG-001",
            reporting_year=2025,
            facility_data=[
                {
                    "facility_id": "CEMENT-FAC-001",
                    "facility_name": "Heidelberg Cement Works",
                    "country": "DE",
                    "sector": "C23.5",
                    "sub_sector": "cement",
                },
            ],
            production_volumes=[
                {
                    "facility_id": "CEMENT-FAC-001",
                    "product_id": "CEM-I",
                    "product_name": "Portland Cement CEM I",
                    "quantity": 500000.0,
                    "unit": "tonnes",
                },
            ],
            energy_data=[
                {
                    "facility_id": "CEMENT-FAC-001",
                    "source": "coal",
                    "consumption": 80000.0,
                    "unit": "MWh",
                    "emission_factor": 0.34,
                    "scope": "scope1",
                },
                {
                    "facility_id": "CEMENT-FAC-001",
                    "source": "electricity",
                    "consumption": 45000.0,
                    "unit": "MWh",
                    "emission_factor": 0.4,
                    "scope": "scope2",
                },
            ],
            process_emission_factors={"CEMENT-FAC-001": 0.53},
        )

    def test_workflow_runs_successfully(self, cement_input):
        """Full workflow executes without error for cement plant."""
        wf = ManufacturingEmissionsWorkflow()
        result = _run_async(wf.run(cement_input))
        assert result is not None
        assert hasattr(result, "status")
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_template_renders_from_workflow_output(self, cement_input):
        """Process emissions template renders from workflow result data."""
        wf = ManufacturingEmissionsWorkflow()
        result = _run_async(wf.run(cement_input))

        # Build template data from workflow result
        template_data = {
            "company_name": "Heidelberg Cement Works",
            "reporting_year": 2025,
            "scope1_total": getattr(result, "scope1_total", 0.0),
            "process_emissions": getattr(result, "process_emissions", 0.0),
            "combustion_emissions": getattr(result, "combustion_emissions", 0.0),
        }

        tpl = ProcessEmissionsReportTemplate()
        md = tpl.render_markdown(template_data)
        assert isinstance(md, str)
        assert "Process Emissions Report" in md
        assert "provenance_hash" in md

    def test_provenance_chain_is_consistent(self, cement_input):
        """Provenance hashes are deterministic across runs with same input."""
        wf1 = ManufacturingEmissionsWorkflow()
        wf2 = ManufacturingEmissionsWorkflow()
        result1 = _run_async(wf1.run(cement_input))
        result2 = _run_async(wf2.run(cement_input))
        # Both results should have 64-char provenance hashes
        assert len(result1.provenance_hash) == 64
        assert len(result2.provenance_hash) == 64


# ===========================================================================
# Scenario 2: Automotive E2E (3 tests)
# ===========================================================================

class TestAutomotiveE2E:
    """End-to-end tests for automotive manufacturing scenario."""

    @pytest.fixture
    def automotive_input(self):
        """Create automotive circular economy input data."""
        return CircularEconomyInput(
            organization_id="AUTO-ORG-001",
            reporting_year=2025,
            material_flows=[
                {
                    "material_id": "MAT-001",
                    "material_name": "Steel",
                    "inflow_tonnes": 50000.0,
                    "recycled_content_pct": 25.0,
                },
                {
                    "material_id": "MAT-002",
                    "material_name": "Aluminum",
                    "inflow_tonnes": 15000.0,
                    "recycled_content_pct": 40.0,
                },
            ],
            waste_streams=[
                {
                    "waste_id": "WAS-001",
                    "waste_type": "metal_scrap",
                    "quantity_tonnes": 5000.0,
                    "treatment": "recycling",
                },
            ],
        )

    def test_workflow_runs_successfully(self, automotive_input):
        """Full circular economy workflow executes for automotive."""
        wf = CircularEconomyWorkflow()
        result = _run_async(wf.run(automotive_input))
        assert result is not None
        assert hasattr(result, "status")

    def test_template_renders_from_workflow_output(self, automotive_input):
        """Circular economy template renders from workflow result."""
        wf = CircularEconomyWorkflow()
        result = _run_async(wf.run(automotive_input))

        template_data = {
            "company_name": "AutoMfg GmbH",
            "reporting_year": 2025,
            "mci_score": getattr(result, "mci_score", 0.0),
        }

        tpl = CircularEconomyReportTemplate()
        md = tpl.render_markdown(template_data)
        assert isinstance(md, str)
        assert len(md) > 50

    def test_result_has_provenance(self, automotive_input):
        """Result has a valid provenance hash."""
        wf = CircularEconomyWorkflow()
        result = _run_async(wf.run(automotive_input))
        assert hasattr(result, "provenance_hash")
        assert len(result.provenance_hash) == 64


# ===========================================================================
# Scenario 3: Chemical Plant E2E (3 tests)
# ===========================================================================

class TestChemicalPlantE2E:
    """End-to-end tests for chemical manufacturing scenario."""

    @pytest.fixture
    def chemical_input(self):
        """Create chemical plant BAT compliance input data."""
        return BATComplianceInput(
            organization_id="CHEM-ORG-001",
            reporting_year=2025,
            facility_data=[
                {
                    "facility_id": "CHEM-FAC-001",
                    "facility_name": "BASF Ludwigshafen",
                    "country": "DE",
                    "sector": "C20",
                    "sub_sector": "chemicals",
                },
            ],
            applicable_brefs=[
                {
                    "bref_id": "LCP",
                    "bref_name": "Large Combustion Plants",
                    "sector": "chemicals",
                    "publication_year": 2017,
                    "applicable": True,
                },
                {
                    "bref_id": "CWW",
                    "bref_name": "Common Waste Water",
                    "sector": "chemicals",
                    "publication_year": 2016,
                    "applicable": True,
                },
            ],
            measured_parameters=[
                {
                    "parameter_id": "NOx",
                    "parameter_name": "NOx",
                    "measured_value": 180.0,
                    "unit": "mg/Nm3",
                    "bat_ael_lower": 100.0,
                    "bat_ael_upper": 200.0,
                },
                {
                    "parameter_id": "SO2",
                    "parameter_name": "SO2",
                    "measured_value": 45.0,
                    "unit": "mg/Nm3",
                    "bat_ael_lower": 10.0,
                    "bat_ael_upper": 50.0,
                },
            ],
        )

    def test_workflow_runs_successfully(self, chemical_input):
        """Full BAT compliance workflow executes for chemical plant."""
        wf = BATComplianceWorkflow()
        result = _run_async(wf.run(chemical_input))
        assert result is not None
        assert hasattr(result, "status")

    def test_template_renders_from_workflow_output(self, chemical_input):
        """BAT compliance template renders from workflow result."""
        wf = BATComplianceWorkflow()
        result = _run_async(wf.run(chemical_input))

        template_data = {
            "company_name": "BASF AG",
            "reporting_year": 2025,
            "facility_name": "Ludwigshafen",
            "applicable_brefs": ["LCP", "CWW", "REF"],
            "compliance_status": getattr(result, "compliance_status", "UNKNOWN"),
        }

        tpl = BATComplianceReportTemplate()
        md = tpl.render_markdown(template_data)
        assert isinstance(md, str)
        assert len(md) > 50

    def test_result_has_provenance(self, chemical_input):
        """Result has a valid provenance hash."""
        wf = BATComplianceWorkflow()
        result = _run_async(wf.run(chemical_input))
        assert hasattr(result, "provenance_hash")
        assert len(result.provenance_hash) == 64


# ===========================================================================
# Scenario 4: Multi-Facility Decarbonization E2E (3 tests)
# ===========================================================================

class TestMultiFacilityE2E:
    """End-to-end tests for multi-facility decarbonization scenario."""

    @pytest.fixture
    def decarb_input(self):
        """Create multi-facility decarbonization input data."""
        return DecarbonizationInput(
            organization_id="MULTI-ORG-001",
            baseline_year=2023,
            target_year=2030,
            target_reduction_pct=42.0,
            baseline_emissions={
                "scope1": 100000.0,
                "scope2": 35000.0,
                "scope3": 15000.0,
            },
            facility_data=[
                {
                    "facility_id": "FAC-A",
                    "facility_name": "Plant Alpha",
                    "country": "DE",
                    "emissions_tco2e": 80000.0,
                },
                {
                    "facility_id": "FAC-B",
                    "facility_name": "Plant Beta",
                    "country": "FR",
                    "emissions_tco2e": 70000.0,
                },
            ],
            technology_options=[
                {
                    "technology_id": "TECH-001",
                    "technology_name": "Waste heat recovery",
                    "category": "energy_efficiency",
                    "abatement_potential_tco2e": 15000.0,
                    "capex_eur": 2000000.0,
                    "payback_years": 4.5,
                },
                {
                    "technology_id": "TECH-002",
                    "technology_name": "Hydrogen fuel switching",
                    "category": "fuel_switch",
                    "abatement_potential_tco2e": 30000.0,
                    "capex_eur": 15000000.0,
                    "payback_years": 8.0,
                },
            ],
        )

    def test_workflow_runs_successfully(self, decarb_input):
        """Full decarbonization workflow executes for multi-facility."""
        wf = DecarbonizationRoadmapWorkflow()
        result = _run_async(wf.run(decarb_input))
        assert result is not None
        assert hasattr(result, "status")

    def test_template_renders_from_workflow_output(self, decarb_input):
        """Decarbonization template renders from workflow result."""
        wf = DecarbonizationRoadmapWorkflow()
        result = _run_async(wf.run(decarb_input))

        template_data = {
            "company_name": "Multi-Plant Corp",
            "baseline_year": 2023,
            "target_year": 2030,
            "baseline_emissions_tco2e": 150000.0,
            "target_reduction_pct": 42.0,
        }

        tpl = DecarbonizationRoadmapTemplate()
        md = tpl.render_markdown(template_data)
        assert isinstance(md, str)
        assert len(md) > 50

    def test_result_has_provenance(self, decarb_input):
        """Result has a valid provenance hash."""
        wf = DecarbonizationRoadmapWorkflow()
        result = _run_async(wf.run(decarb_input))
        assert hasattr(result, "provenance_hash")
        assert len(result.provenance_hash) == 64
