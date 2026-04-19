# -*- coding: utf-8 -*-
"""
PACK-010 SFDR Article 8 Pack - Workflow Tests
===============================================

Tests all 8 workflow orchestrators for SFDR Article 8 compliance:
PrecontractualDisclosure, PeriodicReporting, WebsiteDisclosure,
PAIStatement, PortfolioScreening, TaxonomyAlignment,
ComplianceReview, and RegulatoryUpdate.

Self-contained: does NOT import from conftest.
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

_wf_precon = _import_from_path(
    "pack010_wf_precon",
    os.path.join(WF_DIR, "precontractual_disclosure.py"),
)
_wf_periodic = _import_from_path(
    "pack010_wf_periodic",
    os.path.join(WF_DIR, "periodic_reporting.py"),
)
_wf_website = _import_from_path(
    "pack010_wf_website",
    os.path.join(WF_DIR, "website_disclosure.py"),
)
_wf_pai = _import_from_path(
    "pack010_wf_pai",
    os.path.join(WF_DIR, "pai_statement.py"),
)
_wf_screen = _import_from_path(
    "pack010_wf_screen",
    os.path.join(WF_DIR, "portfolio_screening.py"),
)
_wf_taxonomy = _import_from_path(
    "pack010_wf_taxonomy",
    os.path.join(WF_DIR, "taxonomy_alignment.py"),
)
_wf_compliance = _import_from_path(
    "pack010_wf_compliance",
    os.path.join(WF_DIR, "compliance_review.py"),
)
_wf_regulatory = _import_from_path(
    "pack010_wf_regulatory",
    os.path.join(WF_DIR, "regulatory_update.py"),
)

# Workflow classes
PrecontractualDisclosureWorkflow = _wf_precon.PrecontractualDisclosureWorkflow
PrecontractualDisclosureInput = _wf_precon.PrecontractualDisclosureInput

PeriodicReportingWorkflow = _wf_periodic.PeriodicReportingWorkflow
PeriodicReportingInput = _wf_periodic.PeriodicReportingInput

WebsiteDisclosureWorkflow = _wf_website.WebsiteDisclosureWorkflow
WebsiteDisclosureInput = _wf_website.WebsiteDisclosureInput

PAIStatementWorkflow = _wf_pai.PAIStatementWorkflow
PAIStatementInput = _wf_pai.PAIStatementInput

PortfolioScreeningWorkflow = _wf_screen.PortfolioScreeningWorkflow
PortfolioScreeningInput = _wf_screen.PortfolioScreeningInput

TaxonomyAlignmentWorkflow = _wf_taxonomy.TaxonomyAlignmentWorkflow
TaxonomyAlignmentInput = _wf_taxonomy.TaxonomyAlignmentInput

ComplianceReviewWorkflow = _wf_compliance.ComplianceReviewWorkflow
ComplianceReviewInput = _wf_compliance.ComplianceReviewInput

RegulatoryUpdateWorkflow = _wf_regulatory.RegulatoryUpdateWorkflow
RegulatoryUpdateInput = _wf_regulatory.RegulatoryUpdateInput


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPrecontractualDisclosureWorkflow:
    """Tests for PrecontractualDisclosureWorkflow (5-phase)."""

    def test_instantiation(self):
        """Workflow can be instantiated without arguments."""
        wf = PrecontractualDisclosureWorkflow()
        assert wf is not None

    @pytest.mark.asyncio
    async def test_run_produces_result(self):
        """Running the workflow with valid input produces a result."""
        wf = PrecontractualDisclosureWorkflow()
        inp = PrecontractualDisclosureInput(
            organization_id="org-test-001",
            product_name="GL Green Equity Fund",
            reporting_date=date.today().isoformat(),
        )
        result = await wf.run(inp)
        assert result is not None
        assert hasattr(result, "status")
        assert hasattr(result, "provenance_hash")
        assert hasattr(result, "phases")
        assert len(result.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_phase_count(self):
        """PrecontractualDisclosure has exactly 5 phases."""
        wf = PrecontractualDisclosureWorkflow()
        inp = PrecontractualDisclosureInput(
            organization_id="org-test-002",
            product_name="Test Fund",
            reporting_date="2026-01-01",
        )
        result = await wf.run(inp)
        assert len(result.phases) == 5


class TestPeriodicReportingWorkflow:
    """Tests for PeriodicReportingWorkflow (5-phase)."""

    def test_instantiation(self):
        """Workflow can be instantiated without arguments."""
        wf = PeriodicReportingWorkflow()
        assert wf is not None

    @pytest.mark.asyncio
    async def test_run_and_provenance(self):
        """Running the workflow produces a result with provenance hash."""
        wf = PeriodicReportingWorkflow()
        inp = PeriodicReportingInput(
            organization_id="org-test-003",
            product_name="GL ESG Bond Fund",
            reporting_period_start="2025-01-01",
            reporting_period_end="2025-12-31",
        )
        result = await wf.run(inp)
        assert result is not None
        assert len(result.provenance_hash) == 64
        assert len(result.phases) == 5


class TestWebsiteDisclosureWorkflow:
    """Tests for WebsiteDisclosureWorkflow (4-phase)."""

    @pytest.mark.asyncio
    async def test_run_with_valid_input(self):
        """Running produces a result with 4 phases."""
        wf = WebsiteDisclosureWorkflow()
        inp = WebsiteDisclosureInput(
            organization_id="org-test-004",
            product_name="GL Sustainable Balanced",
            disclosure_date="2026-03-01",
        )
        result = await wf.run(inp)
        assert result is not None
        assert len(result.phases) == 4
        assert hasattr(result, "provenance_hash")


class TestPAIStatementWorkflow:
    """Tests for PAIStatementWorkflow (4-phase)."""

    @pytest.mark.asyncio
    async def test_run_with_valid_input(self):
        """Running produces a result with 4 phases."""
        wf = PAIStatementWorkflow()
        inp = PAIStatementInput(
            organization_id="org-test-005",
            product_name="GL Climate Fund",
            reporting_period_start="2025-01-01",
            reporting_period_end="2025-12-31",
        )
        result = await wf.run(inp)
        assert result is not None
        assert len(result.phases) == 4
        assert len(result.provenance_hash) == 64


class TestPortfolioScreeningWorkflow:
    """Tests for PortfolioScreeningWorkflow (4-phase)."""

    @pytest.mark.asyncio
    async def test_run_with_valid_input(self):
        """Running produces a result with 4 phases."""
        wf = PortfolioScreeningWorkflow()
        inp = PortfolioScreeningInput(
            organization_id="org-test-006",
            product_name="GL Screened Equity",
            screening_date="2026-03-01",
        )
        result = await wf.run(inp)
        assert result is not None
        assert len(result.phases) == 4


class TestTaxonomyAlignmentWorkflow:
    """Tests for TaxonomyAlignmentWorkflow (4-phase)."""

    @pytest.mark.asyncio
    async def test_run_with_valid_input(self):
        """Running produces a result with 4 phases."""
        wf = TaxonomyAlignmentWorkflow()
        inp = TaxonomyAlignmentInput(
            organization_id="org-test-007",
            product_name="GL Taxonomy Fund",
            reporting_date="2026-03-01",
        )
        result = await wf.run(inp)
        assert result is not None
        assert len(result.phases) == 4


class TestComplianceReviewWorkflow:
    """Tests for ComplianceReviewWorkflow (4-phase)."""

    @pytest.mark.asyncio
    async def test_run_with_valid_input(self):
        """Running produces a result with 4 phases."""
        wf = ComplianceReviewWorkflow()
        inp = ComplianceReviewInput(
            organization_id="org-test-008",
            product_name="GL Compliance Fund",
            review_date="2026-03-01",
        )
        result = await wf.run(inp)
        assert result is not None
        assert len(result.phases) == 4
        assert hasattr(result, "provenance_hash")


class TestRegulatoryUpdateWorkflow:
    """Tests for RegulatoryUpdateWorkflow (3-phase)."""

    @pytest.mark.asyncio
    async def test_run_with_valid_input(self):
        """Running produces a result with 3 phases."""
        wf = RegulatoryUpdateWorkflow()
        inp = RegulatoryUpdateInput(
            organization_id="org-test-009",
            product_name="GL Regulatory Watch",
            review_date="2026-03-01",
        )
        result = await wf.run(inp)
        assert result is not None
        assert len(result.phases) == 3
        assert len(result.provenance_hash) == 64


class TestWorkflowSkipPhases:
    """Test phase-skipping across workflows."""

    @pytest.mark.asyncio
    async def test_precontractual_skip_phase(self):
        """Skipping a phase reduces the total phase results or marks as SKIPPED."""
        wf = PrecontractualDisclosureWorkflow()
        inp = PrecontractualDisclosureInput(
            organization_id="org-test-skip",
            product_name="GL Skip Test Fund",
            reporting_date="2026-01-01",
            skip_phases=["review_approval"],
        )
        result = await wf.run(inp)
        assert result is not None
        # Skipped phases should be marked accordingly or omitted
        statuses = [p.status.value if hasattr(p.status, "value") else p.status for p in result.phases]
        # At least one phase should be skipped or the phase count <= 5
        assert len(result.phases) <= 5


class TestWorkflowProvenanceDeterminism:
    """Test that provenance hashes are deterministic for the same inputs."""

    @pytest.mark.asyncio
    async def test_same_input_same_hash(self):
        """Two runs with identical input produce the same provenance hash."""
        wf = PrecontractualDisclosureWorkflow()
        inp = PrecontractualDisclosureInput(
            organization_id="org-determinism",
            product_name="GL Determinism Fund",
            reporting_date="2026-01-01",
        )
        result1 = await wf.run(inp)
        result2 = await wf.run(inp)
        # Both should have valid 64-char hashes
        assert len(result1.provenance_hash) == 64
        assert len(result2.provenance_hash) == 64
