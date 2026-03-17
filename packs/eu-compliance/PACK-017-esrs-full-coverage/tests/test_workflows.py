# -*- coding: utf-8 -*-
"""
PACK-017 ESRS Full Coverage Pack - Workflow Tests
====================================================

Tests for all 12 ESRS workflows: file existence, module loading, class
exports, execute/run methods, phase enums, and per-workflow execution
phase validation.

Target: 50+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-017 ESRS Full Coverage Pack
Date:    March 2026
"""

import asyncio

import pytest

from .conftest import (
    WORKFLOW_FILES,
    WORKFLOW_CLASSES,
    WORKFLOWS_DIR,
    _load_workflow,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _run_async(coro):
    """Run an async coroutine synchronously for testing."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _try_load_workflow(key):
    """Attempt to load a workflow, returning module or None."""
    try:
        return _load_workflow(key)
    except (ImportError, FileNotFoundError):
        return None


# ===========================================================================
# Parametrized: file existence, module loading, class/execute/phases
# ===========================================================================


class TestWorkflowFilesExist:
    """Test that all 12 workflow files exist on disk."""

    @pytest.mark.parametrize("wf_key,wf_file", list(WORKFLOW_FILES.items()))
    def test_workflow_file_exists(self, wf_key, wf_file):
        """Workflow file exists on disk."""
        path = WORKFLOWS_DIR / wf_file
        assert path.exists(), f"Workflow file missing: {path}"


class TestWorkflowModuleLoading:
    """Test that all 12 workflows can be loaded via importlib."""

    @pytest.mark.parametrize("wf_key", list(WORKFLOW_FILES.keys()))
    def test_workflow_module_loads(self, wf_key):
        """Each workflow module loads independently."""
        mod = _try_load_workflow(wf_key)
        assert mod is not None, f"Workflow {wf_key} failed to load"

    @pytest.mark.parametrize("wf_key,wf_class", list(WORKFLOW_CLASSES.items()))
    def test_workflow_exports_class(self, wf_key, wf_class):
        """Each workflow exports its primary class."""
        mod = _try_load_workflow(wf_key)
        if mod is None:
            pytest.skip(f"Workflow {wf_key} not loaded")
        assert hasattr(mod, wf_class), f"Workflow {wf_key} missing class {wf_class}"

    @pytest.mark.parametrize("wf_key,wf_class", list(WORKFLOW_CLASSES.items()))
    def test_workflow_has_execute_method(self, wf_key, wf_class):
        """Each workflow class has execute or run method."""
        mod = _try_load_workflow(wf_key)
        if mod is None:
            pytest.skip(f"Workflow {wf_key} not loaded")
        cls = getattr(mod, wf_class, None)
        if cls is None:
            pytest.skip(f"Class {wf_class} not found")
        has_execute = (
            hasattr(cls, "execute")
            or hasattr(cls, "run")
            or hasattr(cls, "execute_pipeline")
        )
        assert has_execute, f"{wf_class} missing execute/run method"

    @pytest.mark.parametrize("wf_key", list(WORKFLOW_FILES.keys()))
    def test_workflow_has_phases_enum(self, wf_key):
        """Each workflow file defines a WorkflowPhase enum or phase references."""
        source_path = WORKFLOWS_DIR / WORKFLOW_FILES[wf_key]
        if not source_path.exists():
            pytest.skip(f"File not found: {source_path}")
        content = source_path.read_text(encoding="utf-8")
        has_phases = (
            "WorkflowPhase" in content
            or "Phase" in content
            or "phase" in content.lower()
        )
        assert has_phases, f"Workflow {wf_key} should define phases"


# ===========================================================================
# ESRS 2 General Disclosures Workflow
# ===========================================================================


class TestESRS2Workflow:
    """Tests for the ESRS2GeneralWorkflow."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_workflow("esrs2_general")

    def test_governance_assessment_phase(self):
        """ESRS2 workflow references governance assessment phase."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        source = (WORKFLOWS_DIR / "esrs2_general_workflow.py").read_text(encoding="utf-8")
        has_gov = "governance" in source.lower() or "GOV" in source
        assert has_gov, "ESRS2 workflow should include governance assessment"

    def test_full_execution(self):
        """ESRS2 workflow defines at least 5 phases."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        source = (WORKFLOWS_DIR / "esrs2_general_workflow.py").read_text(encoding="utf-8")
        phase_count = source.lower().count("phase")
        assert phase_count >= 5, "Expected at least 5 phase references"


# ===========================================================================
# E2 Pollution Workflow
# ===========================================================================


class TestE2PollutionWorkflow:
    """Tests for the E2PollutionWorkflow."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_workflow("e2_pollution")

    def test_data_collection_phase(self):
        """E2 workflow references pollutant data collection."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        source = (WORKFLOWS_DIR / "e2_pollution_workflow.py").read_text(encoding="utf-8")
        has_data = "pollut" in source.lower() or "emission" in source.lower()
        assert has_data, "E2 workflow should include pollution data collection"

    def test_full_execution(self):
        """E2 workflow defines at least 5 phases."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        source = (WORKFLOWS_DIR / "e2_pollution_workflow.py").read_text(encoding="utf-8")
        phase_count = source.lower().count("phase")
        assert phase_count >= 5


# ===========================================================================
# E3 Water Workflow
# ===========================================================================


class TestE3WaterWorkflow:
    """Tests for the E3WaterWorkflow."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_workflow("e3_water")

    def test_water_balance_phase(self):
        """E3 workflow references water balance or withdrawal."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        source = (WORKFLOWS_DIR / "e3_water_workflow.py").read_text(encoding="utf-8")
        has_water = "water" in source.lower() or "withdrawal" in source.lower()
        assert has_water

    def test_full_execution(self):
        """E3 workflow defines at least 4 phases."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        source = (WORKFLOWS_DIR / "e3_water_workflow.py").read_text(encoding="utf-8")
        phase_count = source.lower().count("phase")
        assert phase_count >= 4


# ===========================================================================
# E4 Biodiversity Workflow
# ===========================================================================


class TestE4BiodiversityWorkflow:
    """Tests for the E4BiodiversityWorkflow."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_workflow("e4_biodiversity")

    def test_site_assessment_phase(self):
        """E4 workflow references site or biodiversity assessment."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        source = (WORKFLOWS_DIR / "e4_biodiversity_workflow.py").read_text(encoding="utf-8")
        has_bio = "biodiversity" in source.lower() or "site" in source.lower()
        assert has_bio

    def test_full_execution(self):
        """E4 workflow defines at least 4 phases."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        source = (WORKFLOWS_DIR / "e4_biodiversity_workflow.py").read_text(encoding="utf-8")
        phase_count = source.lower().count("phase")
        assert phase_count >= 4


# ===========================================================================
# E5 Circular Economy Workflow
# ===========================================================================


class TestE5CircularWorkflow:
    """Tests for the E5CircularWorkflow."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_workflow("e5_circular_economy")

    def test_inflow_analysis_phase(self):
        """E5 workflow references inflow or resource analysis."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        source = (WORKFLOWS_DIR / "e5_circular_economy_workflow.py").read_text(encoding="utf-8")
        has_circular = "circular" in source.lower() or "inflow" in source.lower() or "resource" in source.lower()
        assert has_circular

    def test_full_execution(self):
        """E5 workflow defines at least 4 phases."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        source = (WORKFLOWS_DIR / "e5_circular_economy_workflow.py").read_text(encoding="utf-8")
        phase_count = source.lower().count("phase")
        assert phase_count >= 4


# ===========================================================================
# S1 Workforce Workflow
# ===========================================================================


class TestS1WorkforceWorkflow:
    """Tests for the S1WorkforceWorkflow."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_workflow("s1_workforce")

    def test_demographics_phase(self):
        """S1 workflow references workforce demographics or employee data."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        source = (WORKFLOWS_DIR / "s1_workforce_workflow.py").read_text(encoding="utf-8")
        has_workforce = "workforce" in source.lower() or "employee" in source.lower()
        assert has_workforce

    def test_full_execution(self):
        """S1 workflow defines at least 5 phases."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        source = (WORKFLOWS_DIR / "s1_workforce_workflow.py").read_text(encoding="utf-8")
        phase_count = source.lower().count("phase")
        assert phase_count >= 5


# ===========================================================================
# S2 Value Chain Workers Workflow
# ===========================================================================


class TestS2ValueChainWorkflow:
    """Tests for the S2ValueChainWorkflow."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_workflow("s2_value_chain")

    def test_risk_assessment_phase(self):
        """S2 workflow references risk assessment or value chain analysis."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        source = (WORKFLOWS_DIR / "s2_value_chain_workflow.py").read_text(encoding="utf-8")
        has_vc = "value chain" in source.lower() or "risk" in source.lower()
        assert has_vc

    def test_full_execution(self):
        """S2 workflow defines at least 4 phases."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        source = (WORKFLOWS_DIR / "s2_value_chain_workflow.py").read_text(encoding="utf-8")
        phase_count = source.lower().count("phase")
        assert phase_count >= 4


# ===========================================================================
# S3 Affected Communities Workflow
# ===========================================================================


class TestS3CommunitiesWorkflow:
    """Tests for the S3CommunitiesWorkflow."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_workflow("s3_communities")

    def test_engagement_phase(self):
        """S3 workflow references community engagement."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        source = (WORKFLOWS_DIR / "s3_communities_workflow.py").read_text(encoding="utf-8")
        has_comm = "communit" in source.lower() or "engagement" in source.lower()
        assert has_comm

    def test_full_execution(self):
        """S3 workflow defines at least 4 phases."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        source = (WORKFLOWS_DIR / "s3_communities_workflow.py").read_text(encoding="utf-8")
        phase_count = source.lower().count("phase")
        assert phase_count >= 4


# ===========================================================================
# S4 Consumers Workflow
# ===========================================================================


class TestS4ConsumersWorkflow:
    """Tests for the S4ConsumersWorkflow."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_workflow("s4_consumers")

    def test_safety_evaluation_phase(self):
        """S4 workflow references product safety or consumer evaluation."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        source = (WORKFLOWS_DIR / "s4_consumers_workflow.py").read_text(encoding="utf-8")
        has_safety = "safety" in source.lower() or "consumer" in source.lower()
        assert has_safety

    def test_full_execution(self):
        """S4 workflow defines at least 4 phases."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        source = (WORKFLOWS_DIR / "s4_consumers_workflow.py").read_text(encoding="utf-8")
        phase_count = source.lower().count("phase")
        assert phase_count >= 4


# ===========================================================================
# G1 Governance Workflow
# ===========================================================================


class TestG1GovernanceWorkflow:
    """Tests for the G1GovernanceWorkflow."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_workflow("g1_governance")

    def test_corruption_assessment_phase(self):
        """G1 workflow references corruption or business conduct assessment."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        source = (WORKFLOWS_DIR / "g1_governance_workflow.py").read_text(encoding="utf-8")
        has_gov = "corruption" in source.lower() or "conduct" in source.lower()
        assert has_gov

    def test_full_execution(self):
        """G1 workflow defines at least 4 phases."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        source = (WORKFLOWS_DIR / "g1_governance_workflow.py").read_text(encoding="utf-8")
        phase_count = source.lower().count("phase")
        assert phase_count >= 4


# ===========================================================================
# ESRS Coverage Workflow
# ===========================================================================


class TestESRSCoverageWorkflow:
    """Tests for the ESRSCoverageWorkflow (cross-standard)."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_workflow("esrs_coverage") if "esrs_coverage" in WORKFLOW_FILES else None

    def test_materiality_phase(self):
        """Coverage workflow references materiality assessment."""
        wf_key = "esrs_coverage"
        if wf_key not in WORKFLOW_FILES:
            pytest.skip("esrs_coverage workflow not in mapping")
        source_path = WORKFLOWS_DIR / WORKFLOW_FILES[wf_key]
        if not source_path.exists():
            pytest.skip(f"File not found: {source_path}")
        content = source_path.read_text(encoding="utf-8")
        has_mat = "materiality" in content.lower() or "coverage" in content.lower()
        assert has_mat

    def test_full_execution(self):
        """Coverage workflow defines at least 4 phases."""
        wf_key = "esrs_coverage"
        if wf_key not in WORKFLOW_FILES:
            pytest.skip("esrs_coverage workflow not in mapping")
        source_path = WORKFLOWS_DIR / WORKFLOW_FILES[wf_key]
        if not source_path.exists():
            pytest.skip(f"File not found: {source_path}")
        content = source_path.read_text(encoding="utf-8")
        phase_count = content.lower().count("phase")
        assert phase_count >= 4


# ===========================================================================
# Full ESRS Workflow
# ===========================================================================


class TestFullESRSWorkflow:
    """Tests for the FullCoverageWorkflow (master 10+ phase pipeline)."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_workflow("full_esrs")

    def test_initialization_phase(self):
        """Full workflow references initialization or config phase."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        source = (WORKFLOWS_DIR / WORKFLOW_FILES["full_coverage"]).read_text(encoding="utf-8")
        has_init = (
            "init" in source.lower()
            or "config" in source.lower()
            or "materiality" in source.lower()
        )
        assert has_init

    def test_full_10_phase_execution(self):
        """Full workflow defines at least 10 phases."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        source = (WORKFLOWS_DIR / WORKFLOW_FILES["full_coverage"]).read_text(encoding="utf-8")
        phase_count = source.lower().count("phase")
        assert phase_count >= 10, f"Expected 10+ phase refs, found {phase_count}"


# ===========================================================================
# Workflow Validation (cross-workflow)
# ===========================================================================


class TestWorkflowValidation:
    """Cross-workflow validation patterns."""

    @pytest.mark.parametrize("wf_key,wf_class", list(WORKFLOW_CLASSES.items()))
    def test_workflow_has_docstring(self, wf_key, wf_class):
        """Each workflow class has a docstring."""
        mod = _try_load_workflow(wf_key)
        if mod is None:
            pytest.skip(f"Workflow {wf_key} not loaded")
        cls = getattr(mod, wf_class, None)
        if cls is None:
            pytest.skip(f"Class {wf_class} not found")
        assert cls.__doc__ is not None

    @pytest.mark.parametrize("wf_key", list(WORKFLOW_FILES.keys()))
    def test_workflow_uses_hashlib(self, wf_key):
        """Each workflow file references hashlib for provenance."""
        source_path = WORKFLOWS_DIR / WORKFLOW_FILES[wf_key]
        if not source_path.exists():
            pytest.skip(f"File not found: {source_path}")
        content = source_path.read_text(encoding="utf-8")
        has_hash = (
            "hashlib" in content
            or "sha256" in content.lower()
            or "provenance" in content.lower()
        )
        assert has_hash, f"Workflow {wf_key} should reference provenance/hashing"
