# -*- coding: utf-8 -*-
"""
PACK-011 SFDR Article 9 Pack - Workflow Tests
================================================

Tests all 8 workflow orchestrators for SFDR Article 9 compliance:
AnnexIIIDisclosure, AnnexVReporting, SustainableVerification,
ImpactReporting, BenchmarkMonitoring, PAIMandatory,
DowngradeMonitoring, and RegulatoryUpdate.

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

_wf_annex_iii = _import_from_path(
    "pack011_wf_annex_iii",
    os.path.join(WF_DIR, "annex_iii_disclosure.py"),
)
_wf_annex_v = _import_from_path(
    "pack011_wf_annex_v",
    os.path.join(WF_DIR, "annex_v_reporting.py"),
)
_wf_sustainable = _import_from_path(
    "pack011_wf_sustainable",
    os.path.join(WF_DIR, "sustainable_verification.py"),
)
_wf_impact = _import_from_path(
    "pack011_wf_impact",
    os.path.join(WF_DIR, "impact_reporting.py"),
)
_wf_benchmark = _import_from_path(
    "pack011_wf_benchmark",
    os.path.join(WF_DIR, "benchmark_monitoring.py"),
)
_wf_pai = _import_from_path(
    "pack011_wf_pai",
    os.path.join(WF_DIR, "pai_mandatory_workflow.py"),
)
_wf_downgrade = _import_from_path(
    "pack011_wf_downgrade",
    os.path.join(WF_DIR, "downgrade_monitoring.py"),
)
_wf_regulatory = _import_from_path(
    "pack011_wf_regulatory",
    os.path.join(WF_DIR, "regulatory_update.py"),
)

# Workflow classes
AnnexIIIDisclosureWorkflow = _wf_annex_iii.AnnexIIIDisclosureWorkflow
AnnexIIIDisclosureInput = _wf_annex_iii.AnnexIIIDisclosureInput
AnnexIIIDisclosureResult = _wf_annex_iii.AnnexIIIDisclosureResult
SustainableObjective = _wf_annex_iii.SustainableObjective
SustainableObjectiveType = _wf_annex_iii.SustainableObjectiveType
WorkflowStatus = _wf_annex_iii.WorkflowStatus

AnnexVReportingWorkflow = _wf_annex_v.AnnexVReportingWorkflow
AnnexVReportingInput = _wf_annex_v.AnnexVReportingInput
AnnexVReportingResult = _wf_annex_v.AnnexVReportingResult

SustainableVerificationWorkflow = _wf_sustainable.SustainableVerificationWorkflow
SustainableVerificationInput = _wf_sustainable.SustainableVerificationInput
SustainableVerificationResult = _wf_sustainable.SustainableVerificationResult

ImpactReportingWorkflow = _wf_impact.ImpactReportingWorkflow
ImpactReportingInput = _wf_impact.ImpactReportingInput
ImpactReportingResult = _wf_impact.ImpactReportingResult

BenchmarkMonitoringWorkflow = _wf_benchmark.BenchmarkMonitoringWorkflow
BenchmarkMonitoringInput = _wf_benchmark.BenchmarkMonitoringInput
BenchmarkMonitoringResult = _wf_benchmark.BenchmarkMonitoringResult

PAIMandatoryWorkflow = _wf_pai.PAIMandatoryWorkflow
PAIMandatoryInput = _wf_pai.PAIMandatoryInput
PAIMandatoryResult = _wf_pai.PAIMandatoryResult

DowngradeMonitoringWorkflow = _wf_downgrade.DowngradeMonitoringWorkflow
DowngradeMonitoringInput = _wf_downgrade.DowngradeMonitoringInput
DowngradeMonitoringResult = _wf_downgrade.DowngradeMonitoringResult

RegulatoryUpdateWorkflow = _wf_regulatory.RegulatoryUpdateWorkflow
RegulatoryUpdateInput = _wf_regulatory.RegulatoryUpdateInput
RegulatoryUpdateResult = _wf_regulatory.RegulatoryUpdateResult


# ===========================================================================
# Test: AnnexIIIDisclosureWorkflow (5-phase)
# ===========================================================================


class TestAnnexIIIDisclosureWorkflow:
    """Tests for the 5-phase Annex III pre-contractual disclosure workflow."""

    def test_instantiation(self):
        """Workflow can be instantiated without arguments."""
        wf = AnnexIIIDisclosureWorkflow()
        assert wf is not None
        assert hasattr(wf, "workflow_id")
        assert wf.WORKFLOW_NAME == "annex_iii_disclosure"

    def test_phase_order_has_five_phases(self):
        """Phase order list contains exactly 5 phases."""
        wf = AnnexIIIDisclosureWorkflow()
        assert len(wf.PHASE_ORDER) == 5

    @pytest.mark.asyncio
    async def test_run_produces_result(self):
        """Running with valid input produces a result with provenance hash."""
        wf = AnnexIIIDisclosureWorkflow()
        inp = AnnexIIIDisclosureInput(
            organization_id="org-test-001",
            product_name="GL Deep Green Equity Fund",
            reporting_date=date.today().isoformat(),
            sustainable_objectives=[
                SustainableObjective(
                    name="Carbon Reduction",
                    objective_type=SustainableObjectiveType.CARBON_REDUCTION,
                    sustainability_indicators=["ghg_intensity_reduction"],
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
        wf = AnnexIIIDisclosureWorkflow()
        inp = AnnexIIIDisclosureInput(
            organization_id="org-test-002",
            product_name="Test Climate Fund",
            reporting_date="2026-01-01",
            sustainable_objectives=[
                SustainableObjective(
                    name="Climate Mitigation",
                    objective_type=SustainableObjectiveType.CLIMATE_CHANGE_MITIGATION,
                    sustainability_indicators=["co2_avoided"],
                )
            ],
        )
        result = await wf.run(inp)
        assert len(result.phases) == 5

    @pytest.mark.asyncio
    async def test_result_fields_populated(self):
        """Key result fields are populated after execution."""
        wf = AnnexIIIDisclosureWorkflow()
        inp = AnnexIIIDisclosureInput(
            organization_id="org-test-003",
            product_name="GL Sustainable Impact Fund",
            reporting_date="2026-03-15",
            sustainable_objectives=[
                SustainableObjective(
                    name="Environmental",
                    objective_type=SustainableObjectiveType.ENVIRONMENTAL,
                    sustainability_indicators=["renewable_energy_share"],
                )
            ],
        )
        result = await wf.run(inp)
        assert result.product_name == "GL Sustainable Impact Fund"
        assert hasattr(result, "is_article_9_eligible")
        assert result.template_sections_total == 11
        assert result.sustainable_investment_commitment_pct >= 0.0


# ===========================================================================
# Test: AnnexVReportingWorkflow
# ===========================================================================


class TestAnnexVReportingWorkflow:
    """Tests for the Annex V periodic reporting workflow."""

    def test_instantiation(self):
        """Workflow can be instantiated without arguments."""
        wf = AnnexVReportingWorkflow()
        assert wf is not None
        assert hasattr(wf, "workflow_id")

    @pytest.mark.asyncio
    async def test_run_produces_result_with_provenance(self):
        """Running produces a result with a 64-char SHA-256 provenance hash."""
        wf = AnnexVReportingWorkflow()
        inp = AnnexVReportingInput(
            organization_id="org-test-010",
            product_name="GL Article 9 Bond Fund",
            reporting_period_start="2025-01-01",
            reporting_period_end="2025-12-31",
        )
        result = await wf.run(inp)
        assert result is not None
        assert len(result.provenance_hash) == 64
        assert result.product_name == "GL Article 9 Bond Fund"

    @pytest.mark.asyncio
    async def test_result_has_reporting_period(self):
        """Result includes the reporting period range."""
        wf = AnnexVReportingWorkflow()
        inp = AnnexVReportingInput(
            organization_id="org-test-011",
            product_name="Test Periodic Fund",
            reporting_period_start="2025-01-01",
            reporting_period_end="2025-12-31",
        )
        result = await wf.run(inp)
        assert hasattr(result, "reporting_period")
        assert hasattr(result, "actual_sustainable_investment_pct")

    @pytest.mark.asyncio
    async def test_phase_results_populated(self):
        """All phase results are populated."""
        wf = AnnexVReportingWorkflow()
        inp = AnnexVReportingInput(
            organization_id="org-test-012",
            product_name="Phase Check Fund",
            reporting_period_start="2025-01-01",
            reporting_period_end="2025-12-31",
        )
        result = await wf.run(inp)
        assert len(result.phases) >= 4
        for phase in result.phases:
            assert hasattr(phase, "phase_name")
            assert hasattr(phase, "status")


# ===========================================================================
# Test: SustainableVerificationWorkflow (4-phase)
# ===========================================================================


class TestSustainableVerificationWorkflow:
    """Tests for the 4-phase sustainable investment verification workflow."""

    def test_instantiation(self):
        """Workflow can be instantiated without arguments."""
        wf = SustainableVerificationWorkflow()
        assert wf is not None

    @pytest.mark.asyncio
    async def test_run_with_valid_input(self):
        """Running produces a result with compliance status."""
        wf = SustainableVerificationWorkflow()
        inp = SustainableVerificationInput(
            organization_id="org-test-020",
            product_name="GL 100% Sustainable Fund",
            verification_date="2026-03-15",
        )
        result = await wf.run(inp)
        assert result is not None
        assert len(result.provenance_hash) == 64
        assert hasattr(result, "sustainable_pct")
        assert hasattr(result, "fully_compliant_holdings")

    @pytest.mark.asyncio
    async def test_phase_count(self):
        """Result contains 4 phase results."""
        wf = SustainableVerificationWorkflow()
        inp = SustainableVerificationInput(
            organization_id="org-test-021",
            product_name="Verification Test",
            verification_date="2026-01-01",
        )
        result = await wf.run(inp)
        assert len(result.phases) == 4


# ===========================================================================
# Test: ImpactReportingWorkflow (4-phase)
# ===========================================================================


class TestImpactReportingWorkflow:
    """Tests for the 4-phase impact measurement and reporting workflow."""

    def test_instantiation(self):
        """Workflow can be instantiated without arguments."""
        wf = ImpactReportingWorkflow()
        assert wf is not None

    @pytest.mark.asyncio
    async def test_run_with_valid_input(self):
        """Running produces result with impact metrics."""
        wf = ImpactReportingWorkflow()
        inp = ImpactReportingInput(
            organization_id="org-test-030",
            product_name="GL Impact Alpha Fund",
            reporting_period_start="2025-01-01",
            reporting_period_end="2025-12-31",
            sustainable_objective="Carbon emissions reduction via renewable energy investments",
        )
        result = await wf.run(inp)
        assert result is not None
        assert len(result.provenance_hash) == 64
        assert hasattr(result, "kpis_defined")
        assert hasattr(result, "impact_metrics_calculated")

    @pytest.mark.asyncio
    async def test_phase_count(self):
        """Result contains 4 phase results."""
        wf = ImpactReportingWorkflow()
        inp = ImpactReportingInput(
            organization_id="org-test-031",
            product_name="Phase Count Fund",
            reporting_period_start="2025-01-01",
            reporting_period_end="2025-12-31",
            sustainable_objective="Environmental sustainability",
        )
        result = await wf.run(inp)
        assert len(result.phases) == 4


# ===========================================================================
# Test: BenchmarkMonitoringWorkflow (4-phase)
# ===========================================================================


class TestBenchmarkMonitoringWorkflow:
    """Tests for the 4-phase EU Climate Benchmark monitoring workflow."""

    def test_instantiation(self):
        """Workflow can be instantiated without arguments."""
        wf = BenchmarkMonitoringWorkflow()
        assert wf is not None

    @pytest.mark.asyncio
    async def test_run_ctb_benchmark(self):
        """Running with CTB benchmark produces alignment result."""
        wf = BenchmarkMonitoringWorkflow()
        inp = BenchmarkMonitoringInput(
            organization_id="org-test-040",
            product_name="GL CTB-Aligned Fund",
            reporting_date="2026-03-15",
            benchmark_type="CTB",
            benchmark_name="MSCI Europe Climate Change CTB",
        )
        result = await wf.run(inp)
        assert result is not None
        assert len(result.provenance_hash) == 64
        assert hasattr(result, "alignment_status")
        assert hasattr(result, "ghg_reduction_pct")

    @pytest.mark.asyncio
    async def test_run_pab_benchmark(self):
        """Running with PAB benchmark produces alignment result."""
        wf = BenchmarkMonitoringWorkflow()
        inp = BenchmarkMonitoringInput(
            organization_id="org-test-041",
            product_name="GL PAB-Aligned Fund",
            reporting_date="2026-03-15",
            benchmark_type="PAB",
            benchmark_name="S&P Eurozone LargeMidCap PAB",
        )
        result = await wf.run(inp)
        assert result is not None
        assert result.benchmark_type == "PAB"

    @pytest.mark.asyncio
    async def test_phase_count(self):
        """Result contains 4 phase results."""
        wf = BenchmarkMonitoringWorkflow()
        inp = BenchmarkMonitoringInput(
            organization_id="org-test-042",
            product_name="Phase Count Fund",
            reporting_date="2026-01-01",
            benchmark_name="MSCI World CTB Index",
        )
        result = await wf.run(inp)
        assert len(result.phases) == 4


# ===========================================================================
# Test: PAIMandatoryWorkflow (4-phase)
# ===========================================================================


class TestPAIMandatoryWorkflow:
    """Tests for the 4-phase mandatory PAI indicator workflow."""

    def test_instantiation(self):
        """Workflow can be instantiated without arguments."""
        wf = PAIMandatoryWorkflow()
        assert wf is not None

    @pytest.mark.asyncio
    async def test_run_with_valid_input(self):
        """Running produces PAI assessment result."""
        wf = PAIMandatoryWorkflow()
        inp = PAIMandatoryInput(
            organization_id="org-test-050",
            product_name="GL Full PAI Fund",
            reporting_date="2026-03-15",
            reporting_period_start="2025-01-01",
            reporting_period_end="2025-12-31",
        )
        result = await wf.run(inp)
        assert result is not None
        assert len(result.provenance_hash) == 64
        assert hasattr(result, "total_indicators")
        assert hasattr(result, "indicators_calculated")

    @pytest.mark.asyncio
    async def test_all_14_mandatory_indicators(self):
        """Result indicates 14 mandatory indicators targeted."""
        wf = PAIMandatoryWorkflow()
        inp = PAIMandatoryInput(
            organization_id="org-test-051",
            product_name="Full PAI Test",
            reporting_date="2026-01-01",
            reporting_period_start="2025-01-01",
            reporting_period_end="2025-12-31",
        )
        result = await wf.run(inp)
        assert result.total_indicators >= 14

    @pytest.mark.asyncio
    async def test_phase_count(self):
        """Result contains 4 phase results."""
        wf = PAIMandatoryWorkflow()
        inp = PAIMandatoryInput(
            organization_id="org-test-052",
            product_name="Phase Count Fund",
            reporting_date="2026-01-01",
            reporting_period_start="2025-01-01",
            reporting_period_end="2025-12-31",
        )
        result = await wf.run(inp)
        assert len(result.phases) == 4


# ===========================================================================
# Test: DowngradeMonitoringWorkflow (4-phase)
# ===========================================================================


class TestDowngradeMonitoringWorkflow:
    """Tests for the 4-phase Article 9 downgrade monitoring workflow."""

    def test_instantiation(self):
        """Workflow can be instantiated without arguments."""
        wf = DowngradeMonitoringWorkflow()
        assert wf is not None

    @pytest.mark.asyncio
    async def test_run_with_valid_input(self):
        """Running produces downgrade risk result."""
        wf = DowngradeMonitoringWorkflow()
        inp = DowngradeMonitoringInput(
            organization_id="org-test-060",
            product_name="GL Dark Green Fund",
            assessment_date="2026-03-15",
        )
        result = await wf.run(inp)
        assert result is not None
        assert len(result.provenance_hash) == 64
        assert hasattr(result, "risk_level")
        assert hasattr(result, "risk_score")
        assert hasattr(result, "triggers_active")

    @pytest.mark.asyncio
    async def test_default_classification_is_article_9(self):
        """Default classification is ARTICLE_9."""
        wf = DowngradeMonitoringWorkflow()
        inp = DowngradeMonitoringInput(
            organization_id="org-test-061",
            product_name="Classification Check",
            assessment_date="2026-01-01",
        )
        result = await wf.run(inp)
        assert result.current_classification == "ARTICLE_9"

    @pytest.mark.asyncio
    async def test_phase_count(self):
        """Result contains 4 phase results."""
        wf = DowngradeMonitoringWorkflow()
        inp = DowngradeMonitoringInput(
            organization_id="org-test-062",
            product_name="Phase Count Fund",
            assessment_date="2026-01-01",
        )
        result = await wf.run(inp)
        assert len(result.phases) == 4


# ===========================================================================
# Test: RegulatoryUpdateWorkflow (3-phase)
# ===========================================================================


class TestRegulatoryUpdateWorkflow:
    """Tests for the 3-phase regulatory change management workflow."""

    def test_instantiation(self):
        """Workflow can be instantiated without arguments."""
        wf = RegulatoryUpdateWorkflow()
        assert wf is not None

    @pytest.mark.asyncio
    async def test_run_with_valid_input(self):
        """Running produces regulatory update result."""
        wf = RegulatoryUpdateWorkflow()
        inp = RegulatoryUpdateInput(
            organization_id="org-test-070",
            product_name="GL Regulatory Watch Fund",
            assessment_date="2026-03-15",
        )
        result = await wf.run(inp)
        assert result is not None
        assert len(result.provenance_hash) == 64
        assert hasattr(result, "changes_detected")
        assert hasattr(result, "migration_actions")

    @pytest.mark.asyncio
    async def test_phase_count(self):
        """Result contains 3 phase results."""
        wf = RegulatoryUpdateWorkflow()
        inp = RegulatoryUpdateInput(
            organization_id="org-test-071",
            product_name="Phase Count Fund",
            assessment_date="2026-01-01",
        )
        result = await wf.run(inp)
        assert len(result.phases) == 3

    @pytest.mark.asyncio
    async def test_result_tracks_disclosures_affected(self):
        """Result tracks number of disclosures affected by changes."""
        wf = RegulatoryUpdateWorkflow()
        inp = RegulatoryUpdateInput(
            organization_id="org-test-072",
            product_name="Disclosure Tracker Fund",
            assessment_date="2026-03-15",
        )
        result = await wf.run(inp)
        assert hasattr(result, "disclosures_affected")
        assert hasattr(result, "processes_affected")
        assert result.disclosures_affected >= 0
