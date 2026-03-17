# -*- coding: utf-8 -*-
"""
PACK-012 CSRD Financial Service Pack - Workflow Tests
========================================================

Tests all 8 workflow orchestrators for CSRD Financial Institution compliance:
FinancedEmissionsWorkflow (5-phase), GARBTARWorkflow (4-phase),
InsuranceEmissionsWorkflow (4-phase), ClimateStressTestWorkflow (5-phase),
FSMaterialityWorkflow (4-phase), TransitionPlanWorkflow (4-phase),
Pillar3ReportingWorkflow (4-phase), RegulatoryIntegrationWorkflow (3-phase).

Self-contained: does NOT import from conftest.
Test count: 30 tests
"""

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from datetime import datetime, date

import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACK_ROOT.parent.parent.parent


def _import_from_path(module_name: str, file_path: str):
    """Import a module from an absolute file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Import all 8 workflow modules
# ---------------------------------------------------------------------------

WF_DIR = str(PACK_ROOT / "workflows")

_wf_financed = _import_from_path(
    "pack012_wf_financed",
    os.path.join(WF_DIR, "financed_emissions_workflow.py"),
)
_wf_gar_btar = _import_from_path(
    "pack012_wf_gar_btar",
    os.path.join(WF_DIR, "gar_btar_workflow.py"),
)
_wf_insurance = _import_from_path(
    "pack012_wf_insurance",
    os.path.join(WF_DIR, "insurance_emissions_workflow.py"),
)
_wf_stress = _import_from_path(
    "pack012_wf_stress",
    os.path.join(WF_DIR, "climate_stress_test_workflow.py"),
)
_wf_materiality = _import_from_path(
    "pack012_wf_materiality",
    os.path.join(WF_DIR, "fs_materiality_workflow.py"),
)
_wf_transition = _import_from_path(
    "pack012_wf_transition",
    os.path.join(WF_DIR, "transition_plan_workflow.py"),
)
_wf_pillar3 = _import_from_path(
    "pack012_wf_pillar3",
    os.path.join(WF_DIR, "pillar3_reporting_workflow.py"),
)
_wf_regulatory = _import_from_path(
    "pack012_wf_regulatory",
    os.path.join(WF_DIR, "regulatory_integration_workflow.py"),
)

# Workflow classes
FinancedEmissionsWorkflow = _wf_financed.FinancedEmissionsWorkflow
FinancedEmissionsInput = _wf_financed.FinancedEmissionsInput
FinancedEmissionsResult = _wf_financed.FinancedEmissionsResult
CounterpartyExposure = _wf_financed.CounterpartyExposure

GARBTARWorkflow = _wf_gar_btar.GARBTARWorkflow
GARBTARInput = _wf_gar_btar.GARBTARInput
GARBTARResult = _wf_gar_btar.GARBTARResult
GARAsset = _wf_gar_btar.GARAsset

InsuranceEmissionsWorkflow = _wf_insurance.InsuranceEmissionsWorkflow
InsuranceEmissionsInput = _wf_insurance.InsuranceEmissionsInput
InsuranceEmissionsResult = _wf_insurance.InsuranceEmissionsResult
InsurancePolicy = _wf_insurance.InsurancePolicy

ClimateStressTestWorkflow = _wf_stress.ClimateStressTestWorkflow
ClimateStressTestInput = _wf_stress.ClimateStressTestInput
ClimateStressTestResult = _wf_stress.ClimateStressTestResult

FSMaterialityWorkflow = _wf_materiality.FSMaterialityWorkflow
FSMaterialityInput = _wf_materiality.FSMaterialityInput
FSMaterialityResult = _wf_materiality.FSMaterialityResult

TransitionPlanWorkflow = _wf_transition.TransitionPlanWorkflow
TransitionPlanInput = _wf_transition.TransitionPlanInput
TransitionPlanResult = _wf_transition.TransitionPlanResult

Pillar3ReportingWorkflow = _wf_pillar3.Pillar3ReportingWorkflow
Pillar3ReportingInput = _wf_pillar3.Pillar3ReportingInput
Pillar3ReportingResult = _wf_pillar3.Pillar3ReportingResult

RegulatoryIntegrationWorkflow = _wf_regulatory.RegulatoryIntegrationWorkflow
RegulatoryIntegrationInput = _wf_regulatory.RegulatoryIntegrationInput
RegulatoryIntegrationResult = _wf_regulatory.RegulatoryIntegrationResult


# ===========================================================================
# Test: FinancedEmissionsWorkflow (5-phase)
# ===========================================================================


class TestFinancedEmissionsWorkflow:
    """Tests for the 5-phase PCAF financed emissions workflow."""

    def test_instantiation(self):
        """Workflow can be instantiated without arguments."""
        wf = FinancedEmissionsWorkflow()
        assert wf is not None
        assert hasattr(wf, "workflow_id")
        assert wf.WORKFLOW_NAME == "financed_emissions"

    def test_phase_order_has_five_phases(self):
        """Phase order list contains exactly 5 phases."""
        wf = FinancedEmissionsWorkflow()
        assert len(wf.PHASE_ORDER) == 5
        assert "data_collection" in wf.PHASE_ORDER
        assert "reporting" in wf.PHASE_ORDER

    @pytest.mark.asyncio
    async def test_run_produces_result(self):
        """Running with valid input produces a result with provenance hash."""
        wf = FinancedEmissionsWorkflow()
        inp = FinancedEmissionsInput(
            organization_id="org-test-001",
            reporting_period="2025",
            exposures=[
                CounterpartyExposure(
                    counterparty_id="cp-001",
                    asset_class="LISTED_EQUITY",
                    outstanding_amount=1000000.0,
                    scope1_emissions=100.0,
                    total_equity_plus_debt=5000000.0,
                )
            ],
        )
        result = await wf.run(inp)
        assert result is not None
        assert hasattr(result, "status")
        assert hasattr(result, "provenance_hash")
        assert hasattr(result, "phases")
        assert len(result.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_phase_count_in_result(self):
        """Result contains exactly 5 phase results."""
        wf = FinancedEmissionsWorkflow()
        inp = FinancedEmissionsInput(
            organization_id="org-test-002",
            reporting_period="2025",
        )
        result = await wf.run(inp)
        assert len(result.phases) == 5

    @pytest.mark.asyncio
    async def test_result_fields_populated(self):
        """Key result fields are populated after execution."""
        wf = FinancedEmissionsWorkflow()
        inp = FinancedEmissionsInput(
            organization_id="org-test-003",
            reporting_period="2025",
            exposures=[
                CounterpartyExposure(
                    counterparty_id="cp-002",
                    asset_class="BUSINESS_LOANS",
                    outstanding_amount=500000.0,
                    scope1_emissions=50.0,
                    scope2_emissions=20.0,
                    total_equity_plus_debt=2000000.0,
                    data_quality_score=2,
                )
            ],
        )
        result = await wf.run(inp)
        assert result.workflow_name == "financed_emissions"
        assert hasattr(result, "total_financed_emissions_tco2e")
        assert hasattr(result, "waci")
        assert hasattr(result, "weighted_data_quality_score")


# ===========================================================================
# Test: GARBTARWorkflow (4-phase)
# ===========================================================================


class TestGARBTARWorkflow:
    """Tests for the 4-phase GAR/BTAR workflow."""

    def test_instantiation(self):
        """Workflow can be instantiated without arguments."""
        wf = GARBTARWorkflow()
        assert wf is not None
        assert hasattr(wf, "workflow_id")
        assert wf.WORKFLOW_NAME == "gar_btar"

    def test_phase_order_has_four_phases(self):
        """Phase order list contains exactly 4 phases."""
        wf = GARBTARWorkflow()
        assert len(wf.PHASE_ORDER) == 4

    @pytest.mark.asyncio
    async def test_run_produces_result_with_provenance(self):
        """Running produces a result with a 64-char SHA-256 provenance hash."""
        wf = GARBTARWorkflow()
        inp = GARBTARInput(
            organization_id="org-test-010",
            reporting_date="2025-12-31",
            assets=[
                GARAsset(
                    asset_id="asset-001",
                    gross_carrying_amount=1000000.0,
                    taxonomy_aligned_pct=30.0,
                    dnsh_compliant=True,
                    minimum_safeguards=True,
                )
            ],
        )
        result = await wf.run(inp)
        assert result is not None
        assert len(result.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_result_has_gar_btar(self):
        """Result includes GAR and BTAR percentage fields."""
        wf = GARBTARWorkflow()
        inp = GARBTARInput(
            organization_id="org-test-011",
            reporting_date="2025-12-31",
        )
        result = await wf.run(inp)
        assert hasattr(result, "gar_pct")
        assert hasattr(result, "btar_pct")
        assert hasattr(result, "gar_by_objective")


# ===========================================================================
# Test: InsuranceEmissionsWorkflow (4-phase)
# ===========================================================================


class TestInsuranceEmissionsWorkflow:
    """Tests for the 4-phase insurance emissions workflow."""

    def test_instantiation(self):
        """Workflow can be instantiated without arguments."""
        wf = InsuranceEmissionsWorkflow()
        assert wf is not None
        assert wf.WORKFLOW_NAME == "insurance_emissions"

    def test_phase_order_has_four_phases(self):
        """Phase order list contains exactly 4 phases."""
        wf = InsuranceEmissionsWorkflow()
        assert len(wf.PHASE_ORDER) == 4

    @pytest.mark.asyncio
    async def test_run_with_valid_input(self):
        """Running with policies produces a result."""
        wf = InsuranceEmissionsWorkflow()
        inp = InsuranceEmissionsInput(
            organization_id="org-test-020",
            reporting_period="2025",
            policies=[
                InsurancePolicy(
                    policy_id="pol-001",
                    gross_written_premium=500000.0,
                    line_of_business="PROPERTY",
                    scope1_emissions=30.0,
                    policyholder_revenue=2000000.0,
                )
            ],
        )
        result = await wf.run(inp)
        assert result is not None
        assert len(result.provenance_hash) == 64
        assert hasattr(result, "gross_attributed_emissions_tco2e")

    @pytest.mark.asyncio
    async def test_result_phases_populated(self):
        """All 4 phase results are populated."""
        wf = InsuranceEmissionsWorkflow()
        inp = InsuranceEmissionsInput(
            organization_id="org-test-021",
            reporting_period="2025",
        )
        result = await wf.run(inp)
        assert len(result.phases) == 4
        for phase in result.phases:
            assert hasattr(phase, "phase_name")
            assert hasattr(phase, "status")


# ===========================================================================
# Test: ClimateStressTestWorkflow (5-phase)
# ===========================================================================


class TestClimateStressTestWorkflow:
    """Tests for the 5-phase climate stress test workflow."""

    def test_instantiation(self):
        """Workflow can be instantiated without arguments."""
        wf = ClimateStressTestWorkflow()
        assert wf is not None
        assert wf.WORKFLOW_NAME == "climate_stress_test"

    def test_phase_order_has_five_phases(self):
        """Phase order list contains exactly 5 phases."""
        wf = ClimateStressTestWorkflow()
        assert len(wf.PHASE_ORDER) == 5

    @pytest.mark.asyncio
    async def test_run_produces_result(self):
        """Running produces result with scenario count."""
        wf = ClimateStressTestWorkflow()
        inp = ClimateStressTestInput(
            organization_id="org-test-030",
            reporting_date="2025-12-31",
        )
        result = await wf.run(inp)
        assert result is not None
        assert len(result.provenance_hash) == 64
        assert hasattr(result, "scenarios_tested")
        assert hasattr(result, "max_credit_loss_pct")

    @pytest.mark.asyncio
    async def test_result_has_risk_exposure(self):
        """Result includes physical and transition risk exposure metrics."""
        wf = ClimateStressTestWorkflow()
        inp = ClimateStressTestInput(
            organization_id="org-test-031",
            reporting_date="2025-12-31",
        )
        result = await wf.run(inp)
        assert hasattr(result, "physical_risk_exposure_pct")
        assert hasattr(result, "transition_risk_exposure_pct")


# ===========================================================================
# Test: FSMaterialityWorkflow (4-phase)
# ===========================================================================


class TestFSMaterialityWorkflow:
    """Tests for the 4-phase FI double materiality workflow."""

    def test_instantiation(self):
        """Workflow can be instantiated without arguments."""
        wf = FSMaterialityWorkflow()
        assert wf is not None
        assert wf.WORKFLOW_NAME == "fs_materiality"

    def test_phase_order_has_four_phases(self):
        """Phase order list contains exactly 4 phases."""
        wf = FSMaterialityWorkflow()
        assert len(wf.PHASE_ORDER) == 4

    @pytest.mark.asyncio
    async def test_run_produces_result(self):
        """Running produces a result with material topics count."""
        wf = FSMaterialityWorkflow()
        inp = FSMaterialityInput(
            organization_id="org-test-040",
            reporting_period="2025",
        )
        result = await wf.run(inp)
        assert result is not None
        assert len(result.provenance_hash) == 64
        assert hasattr(result, "material_topics_count")
        assert hasattr(result, "double_material_count")


# ===========================================================================
# Test: TransitionPlanWorkflow (4-phase)
# ===========================================================================


class TestTransitionPlanWorkflow:
    """Tests for the 4-phase FI transition plan workflow."""

    def test_instantiation(self):
        """Workflow can be instantiated without arguments."""
        wf = TransitionPlanWorkflow()
        assert wf is not None
        assert wf.WORKFLOW_NAME == "transition_plan"

    def test_phase_order_has_four_phases(self):
        """Phase order list contains exactly 4 phases."""
        wf = TransitionPlanWorkflow()
        assert len(wf.PHASE_ORDER) == 4

    @pytest.mark.asyncio
    async def test_run_produces_result(self):
        """Running produces a result with credibility score."""
        wf = TransitionPlanWorkflow()
        inp = TransitionPlanInput(
            organization_id="org-test-050",
            reporting_period="2025",
            institution_name="GL Test Bank",
            commitment_framework="NZBA",
            net_zero_target_year=2050,
        )
        result = await wf.run(inp)
        assert result is not None
        assert len(result.provenance_hash) == 64
        assert hasattr(result, "credibility_score")
        assert hasattr(result, "sectors_with_targets")


# ===========================================================================
# Test: Pillar3ReportingWorkflow (4-phase)
# ===========================================================================


class TestPillar3ReportingWorkflow:
    """Tests for the 4-phase EBA Pillar 3 ESG ITS workflow."""

    def test_instantiation(self):
        """Workflow can be instantiated without arguments."""
        wf = Pillar3ReportingWorkflow()
        assert wf is not None
        assert wf.WORKFLOW_NAME == "pillar3_reporting"

    def test_phase_order_has_four_phases(self):
        """Phase order list contains exactly 4 phases."""
        wf = Pillar3ReportingWorkflow()
        assert len(wf.PHASE_ORDER) == 4

    @pytest.mark.asyncio
    async def test_run_produces_result(self):
        """Running produces a result with templates populated count."""
        wf = Pillar3ReportingWorkflow()
        inp = Pillar3ReportingInput(
            organization_id="org-test-060",
            reporting_date="2025-12-31",
            institution_name="GL Test Credit Institution",
        )
        result = await wf.run(inp)
        assert result is not None
        assert len(result.provenance_hash) == 64
        assert hasattr(result, "templates_populated")
        assert hasattr(result, "filing_ready")

    @pytest.mark.asyncio
    async def test_result_has_validation_fields(self):
        """Result includes data quality and completeness fields."""
        wf = Pillar3ReportingWorkflow()
        inp = Pillar3ReportingInput(
            organization_id="org-test-061",
            reporting_date="2025-12-31",
        )
        result = await wf.run(inp)
        assert hasattr(result, "data_quality_score")
        assert hasattr(result, "completeness_pct")
        assert hasattr(result, "issues_count")


# ===========================================================================
# Test: RegulatoryIntegrationWorkflow (3-phase)
# ===========================================================================


class TestRegulatoryIntegrationWorkflow:
    """Tests for the 3-phase cross-regulatory mapping workflow."""

    def test_instantiation(self):
        """Workflow can be instantiated without arguments."""
        wf = RegulatoryIntegrationWorkflow()
        assert wf is not None
        assert wf.WORKFLOW_NAME == "regulatory_integration"

    def test_phase_order_has_three_phases(self):
        """Phase order list contains exactly 3 phases."""
        wf = RegulatoryIntegrationWorkflow()
        assert len(wf.PHASE_ORDER) == 3

    @pytest.mark.asyncio
    async def test_run_produces_result(self):
        """Running produces a result with coverage and gap data."""
        wf = RegulatoryIntegrationWorkflow()
        inp = RegulatoryIntegrationInput(
            organization_id="org-test-070",
        )
        result = await wf.run(inp)
        assert result is not None
        assert len(result.provenance_hash) == 64
        assert hasattr(result, "coverage_pct")
        assert hasattr(result, "gap_count")
        assert hasattr(result, "regulations_mapped")

    @pytest.mark.asyncio
    async def test_result_phases_populated(self):
        """All 3 phase results are populated."""
        wf = RegulatoryIntegrationWorkflow()
        inp = RegulatoryIntegrationInput(
            organization_id="org-test-071",
        )
        result = await wf.run(inp)
        assert len(result.phases) == 3
        for phase in result.phases:
            assert hasattr(phase, "phase_name")
            assert hasattr(phase, "status")
