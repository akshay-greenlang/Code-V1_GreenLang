# -*- coding: utf-8 -*-
"""
PACK-013 CSRD Manufacturing Pack - Workflow Tests

Tests all 8 workflow orchestrators: initialization, async execution,
phase ordering, and provenance hash generation.

31 tests across 8 test classes.
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


def _load_module(module_name: str, file_name: str, search_dir: Path = WORKFLOWS_DIR):
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
# Load all 8 workflow modules
# ---------------------------------------------------------------------------

mfg_emissions_mod = _load_module(
    "pack013_mfg_emissions_wf",
    "manufacturing_emissions_workflow.py",
)
ManufacturingEmissionsWorkflow = mfg_emissions_mod.ManufacturingEmissionsWorkflow
ManufacturingEmissionsInput = mfg_emissions_mod.ManufacturingEmissionsInput

product_pcf_mod = _load_module(
    "pack013_product_pcf_wf",
    "product_pcf_workflow.py",
)
ProductPCFWorkflow = product_pcf_mod.ProductPCFWorkflow
ProductPCFInput = product_pcf_mod.ProductPCFInput

circular_econ_mod = _load_module(
    "pack013_circular_economy_wf",
    "circular_economy_workflow.py",
)
CircularEconomyWorkflow = circular_econ_mod.CircularEconomyWorkflow
CircularEconomyInput = circular_econ_mod.CircularEconomyInput

bat_mod = _load_module(
    "pack013_bat_compliance_wf",
    "bat_compliance_workflow.py",
)
BATComplianceWorkflow = bat_mod.BATComplianceWorkflow
BATComplianceInput = bat_mod.BATComplianceInput

supply_chain_mod = _load_module(
    "pack013_supply_chain_wf",
    "supply_chain_assessment_workflow.py",
)
SupplyChainAssessmentWorkflow = supply_chain_mod.SupplyChainAssessmentWorkflow
SupplyChainInput = supply_chain_mod.SupplyChainInput

esrs_mod = _load_module(
    "pack013_esrs_mfg_wf",
    "esrs_manufacturing_workflow.py",
)
ESRSManufacturingWorkflow = esrs_mod.ESRSManufacturingWorkflow
ESRSManufacturingInput = esrs_mod.ESRSManufacturingInput

decarb_mod = _load_module(
    "pack013_decarb_roadmap_wf",
    "decarbonization_roadmap_workflow.py",
)
DecarbonizationRoadmapWorkflow = decarb_mod.DecarbonizationRoadmapWorkflow
DecarbonizationInput = decarb_mod.DecarbonizationInput

reg_comp_mod = _load_module(
    "pack013_reg_compliance_wf",
    "regulatory_compliance_workflow.py",
)
RegulatoryComplianceWorkflow = reg_comp_mod.RegulatoryComplianceWorkflow
RegulatoryComplianceInput = reg_comp_mod.RegulatoryComplianceInput


# ---------------------------------------------------------------------------
# Shared helper: run async
# ---------------------------------------------------------------------------

def _run_async(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Shared test data factories
# ---------------------------------------------------------------------------

def _make_facility(fac_id="FAC-001", country="DE", sector="C23"):
    return {
        "facility_id": fac_id,
        "facility_name": f"Plant {fac_id}",
        "country": country,
        "sector": sector,
        "sub_sector": "cement",
    }


def _make_production(fac_id="FAC-001", qty=10000.0):
    return {
        "facility_id": fac_id,
        "product_id": "PROD-001",
        "product_name": "Portland Cement",
        "quantity": qty,
        "unit": "tonnes",
    }


def _make_energy(fac_id="FAC-001", source="electricity", consumption=5000.0):
    return {
        "facility_id": fac_id,
        "source": source,
        "consumption": consumption,
        "unit": "MWh",
        "emission_factor": 0.4,
        "scope": "scope2",
    }


# ===========================================================================
# 1. Manufacturing Emissions Workflow (4 tests)
# ===========================================================================

class TestManufacturingEmissionsWorkflow:
    """Tests for the 4-phase manufacturing emissions workflow."""

    def test_init_creates_workflow_id(self):
        """Workflow initializes with a unique workflow_id."""
        wf = ManufacturingEmissionsWorkflow()
        assert hasattr(wf, "workflow_id")
        assert isinstance(wf.workflow_id, str)
        assert len(wf.workflow_id) > 0

    def test_run_basic_returns_result(self):
        """Running with minimal valid input returns a result object."""
        inp = ManufacturingEmissionsInput(
            organization_id="ORG-001",
            reporting_year=2025,
            facility_data=[_make_facility()],
            production_volumes=[_make_production()],
            energy_data=[_make_energy()],
        )
        wf = ManufacturingEmissionsWorkflow()
        result = _run_async(wf.run(inp))
        assert result is not None
        assert hasattr(result, "status")
        assert hasattr(result, "workflow_id")

    def test_phases_list_has_four_entries(self):
        """PHASE_ORDER contains exactly 4 phases."""
        assert hasattr(ManufacturingEmissionsWorkflow, "PHASE_ORDER")
        assert len(ManufacturingEmissionsWorkflow.PHASE_ORDER) == 4
        expected = [
            "data_collection", "process_calculation",
            "energy_analysis", "consolidation",
        ]
        assert ManufacturingEmissionsWorkflow.PHASE_ORDER == expected

    def test_result_has_provenance_hash(self):
        """Result contains a non-empty provenance_hash field."""
        inp = ManufacturingEmissionsInput(
            organization_id="ORG-002",
            reporting_year=2025,
            facility_data=[_make_facility("FAC-002")],
            production_volumes=[_make_production("FAC-002")],
            energy_data=[_make_energy("FAC-002")],
        )
        wf = ManufacturingEmissionsWorkflow()
        result = _run_async(wf.run(inp))
        assert hasattr(result, "provenance_hash")
        assert isinstance(result.provenance_hash, str)
        assert len(result.provenance_hash) == 64  # SHA-256


# ===========================================================================
# 2. Product PCF Workflow (4 tests)
# ===========================================================================

class TestProductPCFWorkflow:
    """Tests for the 5-phase product PCF workflow."""

    def test_init_creates_workflow_id(self):
        wf = ProductPCFWorkflow()
        assert hasattr(wf, "workflow_id")
        assert len(wf.workflow_id) > 0

    def test_run_basic_returns_result(self):
        inp = ProductPCFInput(
            organization_id="ORG-001",
            product_id="PROD-001",
            product_name="Steel Beam",
            reporting_year=2025,
        )
        result = _run_async(ProductPCFWorkflow().run(inp))
        assert result is not None
        assert hasattr(result, "status")

    def test_phases_list_has_five_entries(self):
        assert hasattr(ProductPCFWorkflow, "PHASE_ORDER")
        assert len(ProductPCFWorkflow.PHASE_ORDER) == 5
        expected = [
            "product_selection", "bom_mapping",
            "lifecycle_assessment", "allocation", "pcf_generation",
        ]
        assert ProductPCFWorkflow.PHASE_ORDER == expected

    def test_result_has_provenance_hash(self):
        inp = ProductPCFInput(
            organization_id="ORG-001",
            product_id="PROD-002",
            product_name="Aluminum Sheet",
            reporting_year=2025,
        )
        result = _run_async(ProductPCFWorkflow().run(inp))
        assert hasattr(result, "provenance_hash")
        assert len(result.provenance_hash) == 64


# ===========================================================================
# 3. Circular Economy Workflow (4 tests)
# ===========================================================================

class TestCircularEconomyWorkflow:
    """Tests for the 4-phase circular economy workflow."""

    def test_init_creates_workflow_id(self):
        wf = CircularEconomyWorkflow()
        assert hasattr(wf, "workflow_id")

    def test_run_basic_returns_result(self):
        inp = CircularEconomyInput(
            organization_id="ORG-001",
            reporting_year=2025,
        )
        result = _run_async(CircularEconomyWorkflow().run(inp))
        assert result is not None
        assert hasattr(result, "status")

    def test_phases_list_has_four_entries(self):
        assert hasattr(CircularEconomyWorkflow, "PHASE_ORDER")
        assert len(CircularEconomyWorkflow.PHASE_ORDER) == 4
        expected = [
            "material_flow_mapping", "waste_analysis",
            "circularity_metrics", "epr_compliance",
        ]
        assert CircularEconomyWorkflow.PHASE_ORDER == expected

    def test_result_has_provenance_hash(self):
        inp = CircularEconomyInput(
            organization_id="ORG-001",
            reporting_year=2025,
        )
        result = _run_async(CircularEconomyWorkflow().run(inp))
        assert hasattr(result, "provenance_hash")
        assert len(result.provenance_hash) == 64


# ===========================================================================
# 4. BAT Compliance Workflow (4 tests)
# ===========================================================================

class TestBATComplianceWorkflow:
    """Tests for the 4-phase BAT/BREF compliance workflow."""

    def test_init_creates_workflow_id(self):
        wf = BATComplianceWorkflow()
        assert hasattr(wf, "workflow_id")

    def test_run_basic_returns_result(self):
        inp = BATComplianceInput(
            organization_id="ORG-001",
            reporting_year=2025,
        )
        result = _run_async(BATComplianceWorkflow().run(inp))
        assert result is not None
        assert hasattr(result, "status")

    def test_phases_list_has_four_entries(self):
        assert hasattr(BATComplianceWorkflow, "PHASE_ORDER")
        assert len(BATComplianceWorkflow.PHASE_ORDER) == 4
        expected = [
            "bref_identification", "performance_assessment",
            "gap_analysis", "transformation_planning",
        ]
        assert BATComplianceWorkflow.PHASE_ORDER == expected

    def test_result_has_provenance_hash(self):
        inp = BATComplianceInput(
            organization_id="ORG-001",
            reporting_year=2025,
        )
        result = _run_async(BATComplianceWorkflow().run(inp))
        assert hasattr(result, "provenance_hash")
        assert len(result.provenance_hash) == 64


# ===========================================================================
# 5. Supply Chain Assessment Workflow (4 tests)
# ===========================================================================

class TestSupplyChainWorkflow:
    """Tests for the 5-phase supply chain assessment workflow."""

    def test_init_creates_workflow_id(self):
        wf = SupplyChainAssessmentWorkflow()
        assert hasattr(wf, "workflow_id")

    def test_run_basic_returns_result(self):
        inp = SupplyChainInput(
            organization_id="ORG-001",
            reporting_year=2025,
        )
        result = _run_async(SupplyChainAssessmentWorkflow().run(inp))
        assert result is not None
        assert hasattr(result, "status")

    def test_phases_list_has_five_entries(self):
        assert hasattr(SupplyChainAssessmentWorkflow, "PHASE_ORDER")
        assert len(SupplyChainAssessmentWorkflow.PHASE_ORDER) == 5
        expected = [
            "supplier_inventory", "data_collection",
            "emission_calculation", "hotspot_analysis",
            "engagement_planning",
        ]
        assert SupplyChainAssessmentWorkflow.PHASE_ORDER == expected

    def test_result_has_provenance_hash(self):
        inp = SupplyChainInput(
            organization_id="ORG-001",
            reporting_year=2025,
        )
        result = _run_async(SupplyChainAssessmentWorkflow().run(inp))
        assert hasattr(result, "provenance_hash")
        assert len(result.provenance_hash) == 64


# ===========================================================================
# 6. ESRS Manufacturing Workflow (4 tests)
# ===========================================================================

class TestESRSManufacturingWorkflow:
    """Tests for the 4-phase ESRS manufacturing workflow."""

    def test_init_creates_workflow_id(self):
        wf = ESRSManufacturingWorkflow()
        assert hasattr(wf, "workflow_id")

    def test_run_basic_returns_result(self):
        inp = ESRSManufacturingInput(
            organization_id="ORG-001",
            reporting_year=2025,
            company_name="Test Manufacturing GmbH",
        )
        result = _run_async(ESRSManufacturingWorkflow().run(inp))
        assert result is not None
        assert hasattr(result, "status")

    def test_phases_list_has_four_entries(self):
        assert hasattr(ESRSManufacturingWorkflow, "PHASE_ORDER")
        assert len(ESRSManufacturingWorkflow.PHASE_ORDER) == 4
        expected = [
            "materiality_assessment", "data_point_collection",
            "disclosure_generation", "audit_preparation",
        ]
        assert ESRSManufacturingWorkflow.PHASE_ORDER == expected

    def test_result_has_provenance_hash(self):
        inp = ESRSManufacturingInput(
            organization_id="ORG-001",
            reporting_year=2025,
            company_name="Test Manufacturing GmbH",
        )
        result = _run_async(ESRSManufacturingWorkflow().run(inp))
        assert hasattr(result, "provenance_hash")
        assert len(result.provenance_hash) == 64


# ===========================================================================
# 7. Decarbonization Roadmap Workflow (4 tests)
# ===========================================================================

class TestDecarbonizationWorkflow:
    """Tests for the 5-phase decarbonization roadmap workflow."""

    def test_init_creates_workflow_id(self):
        wf = DecarbonizationRoadmapWorkflow()
        assert hasattr(wf, "workflow_id")

    def test_run_basic_returns_result(self):
        inp = DecarbonizationInput(
            organization_id="ORG-001",
            baseline_year=2023,
            target_year=2030,
            target_reduction_pct=42.0,
        )
        result = _run_async(DecarbonizationRoadmapWorkflow().run(inp))
        assert result is not None
        assert hasattr(result, "status")

    def test_phases_list_has_five_entries(self):
        assert hasattr(DecarbonizationRoadmapWorkflow, "PHASE_ORDER")
        assert len(DecarbonizationRoadmapWorkflow.PHASE_ORDER) == 5
        expected = [
            "baseline_assessment", "technology_evaluation",
            "target_setting", "investment_planning",
            "monitoring_setup",
        ]
        assert DecarbonizationRoadmapWorkflow.PHASE_ORDER == expected

    def test_result_has_provenance_hash(self):
        inp = DecarbonizationInput(
            organization_id="ORG-001",
            baseline_year=2023,
            target_year=2030,
            target_reduction_pct=42.0,
        )
        result = _run_async(DecarbonizationRoadmapWorkflow().run(inp))
        assert hasattr(result, "provenance_hash")
        assert len(result.provenance_hash) == 64


# ===========================================================================
# 8. Regulatory Compliance Workflow (3 tests)
# ===========================================================================

class TestRegulatoryComplianceWorkflow:
    """Tests for the 3-phase regulatory compliance workflow."""

    def test_init_creates_workflow_id(self):
        wf = RegulatoryComplianceWorkflow()
        assert hasattr(wf, "workflow_id")

    def test_phases_list_has_three_entries(self):
        assert hasattr(RegulatoryComplianceWorkflow, "PHASE_ORDER")
        assert len(RegulatoryComplianceWorkflow.PHASE_ORDER) == 3
        expected = [
            "regulation_mapping", "compliance_assessment",
            "action_planning",
        ]
        assert RegulatoryComplianceWorkflow.PHASE_ORDER == expected

    def test_run_basic_returns_result_with_provenance(self):
        inp = RegulatoryComplianceInput(
            organization_id="ORG-001",
            reporting_year=2025,
        )
        result = _run_async(RegulatoryComplianceWorkflow().run(inp))
        assert result is not None
        assert hasattr(result, "status")
        assert hasattr(result, "provenance_hash")
        assert len(result.provenance_hash) == 64
