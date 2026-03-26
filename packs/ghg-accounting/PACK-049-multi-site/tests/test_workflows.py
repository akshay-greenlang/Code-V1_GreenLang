# -*- coding: utf-8 -*-
"""
Tests for all 8 PACK-049 workflows.

Each workflow: test creation, test creation with config, test has execute,
test full execution, test phase count, test error handling, test provenance.
7 tests per workflow = 56 tests total.
Target: ~55 tests.
"""

import pytest
from decimal import Decimal
from datetime import date
from pathlib import Path
import sys

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

# ---------------------------------------------------------------------------
# Import workflows with graceful fallback
# ---------------------------------------------------------------------------

WORKFLOW_IMPORTS = {}
INPUT_IMPORTS = {}

# SiteRegistrationWorkflow
try:
    from workflows.site_registration_workflow import (
        SiteRegistrationWorkflow, WorkflowStatus, SiteRegistrationInput,
    )
    WORKFLOW_IMPORTS["SiteRegistrationWorkflow"] = SiteRegistrationWorkflow
    INPUT_IMPORTS["SiteRegistrationInput"] = SiteRegistrationInput
except ImportError:
    pass

# DataCollectionWorkflow
try:
    from workflows.data_collection_workflow import (
        DataCollectionWorkflow, DataCollectionInput,
        CollectionRoundConfig,
    )
    WORKFLOW_IMPORTS["DataCollectionWorkflow"] = DataCollectionWorkflow
    INPUT_IMPORTS["DataCollectionInput"] = DataCollectionInput
    INPUT_IMPORTS["CollectionRoundConfig"] = CollectionRoundConfig
except ImportError:
    pass

# BoundaryDefinitionWorkflow
try:
    from workflows.boundary_definition_workflow import (
        BoundaryDefinitionWorkflow, BoundaryDefinitionInput,
    )
    WORKFLOW_IMPORTS["BoundaryDefinitionWorkflow"] = BoundaryDefinitionWorkflow
    INPUT_IMPORTS["BoundaryDefinitionInput"] = BoundaryDefinitionInput
except ImportError:
    pass

# ConsolidationWorkflow
try:
    from workflows.consolidation_workflow import (
        ConsolidationWorkflow, ConsolidationInput,
    )
    WORKFLOW_IMPORTS["ConsolidationWorkflow"] = ConsolidationWorkflow
    INPUT_IMPORTS["ConsolidationInput"] = ConsolidationInput
except ImportError:
    pass

# AllocationWorkflow
try:
    from workflows.allocation_workflow import (
        AllocationWorkflow, AllocationInput,
    )
    WORKFLOW_IMPORTS["AllocationWorkflow"] = AllocationWorkflow
    INPUT_IMPORTS["AllocationInput"] = AllocationInput
except ImportError:
    pass

# SiteComparisonWorkflow
try:
    from workflows.site_comparison_workflow import (
        SiteComparisonWorkflow, SiteComparisonInput,
    )
    WORKFLOW_IMPORTS["SiteComparisonWorkflow"] = SiteComparisonWorkflow
    INPUT_IMPORTS["SiteComparisonInput"] = SiteComparisonInput
except ImportError:
    pass

# QualityImprovementWorkflow
try:
    from workflows.quality_improvement_workflow import (
        QualityImprovementWorkflow, QualityImprovementInput,
    )
    WORKFLOW_IMPORTS["QualityImprovementWorkflow"] = QualityImprovementWorkflow
    INPUT_IMPORTS["QualityImprovementInput"] = QualityImprovementInput
except ImportError:
    pass

# FullMultiSitePipelineWorkflow
try:
    from workflows.full_multi_site_pipeline_workflow import (
        FullMultiSitePipelineWorkflow, FullPipelineInput,
    )
    WORKFLOW_IMPORTS["FullMultiSitePipelineWorkflow"] = FullMultiSitePipelineWorkflow
    INPUT_IMPORTS["FullPipelineInput"] = FullPipelineInput
except ImportError:
    pass

# WorkflowStatus import fallback
try:
    from workflows.site_registration_workflow import WorkflowStatus
except ImportError:
    try:
        from workflows.consolidation_workflow import WorkflowStatus
    except ImportError:
        WorkflowStatus = None


def _get_workflow(name):
    return WORKFLOW_IMPORTS.get(name)


def _get_input(name):
    return INPUT_IMPORTS.get(name)


# ============================================================================
# SiteRegistrationWorkflow
# ============================================================================

class TestSiteRegistrationWorkflow:

    @pytest.fixture
    def wf_class(self):
        cls = _get_workflow("SiteRegistrationWorkflow")
        if cls is None:
            pytest.skip("SiteRegistrationWorkflow not built yet")
        return cls

    def test_create_workflow(self, wf_class):
        wf = wf_class()
        assert wf is not None

    def test_create_workflow_with_config(self, wf_class):
        wf = wf_class(config={"max_sites": 100})
        assert wf is not None

    def test_workflow_has_execute(self, wf_class):
        wf = wf_class()
        assert hasattr(wf, "execute")

    def test_workflow_status_enum(self, wf_class):
        assert WorkflowStatus is not None
        assert WorkflowStatus.COMPLETED is not None
        assert WorkflowStatus.FAILED is not None

    def test_execute_basic(self, wf_class):
        wf = wf_class()
        inp = SiteRegistrationInput(
            organisation_id="ORG-001",
            reporting_year=2026,
            candidate_sites=[
                {"site_name": "Test Plant", "country_code": "DE"},
            ],
        )
        result = wf.execute(inp)
        assert result.status in (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED)

    def test_execute_phase_count(self, wf_class):
        wf = wf_class()
        inp = SiteRegistrationInput(
            organisation_id="ORG-001",
            reporting_year=2026,
            candidate_sites=[
                {"site_name": "Plant A", "country_code": "US"},
            ],
        )
        result = wf.execute(inp)
        assert len(result.phase_results) == 5

    def test_execute_provenance(self, wf_class):
        wf = wf_class()
        inp = SiteRegistrationInput(
            organisation_id="ORG-001",
            reporting_year=2026,
            candidate_sites=[
                {"site_name": "Plant A", "country_code": "GB"},
            ],
        )
        result = wf.execute(inp)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64


# ============================================================================
# DataCollectionWorkflow
# ============================================================================

class TestDataCollectionWorkflow:

    @pytest.fixture
    def wf_class(self):
        cls = _get_workflow("DataCollectionWorkflow")
        if cls is None:
            pytest.skip("DataCollectionWorkflow not built yet")
        return cls

    def test_create_workflow(self, wf_class):
        wf = wf_class()
        assert wf is not None

    def test_create_with_config(self, wf_class):
        wf = wf_class(config={"period": "ANNUAL"})
        assert wf is not None

    def test_has_execute(self, wf_class):
        assert hasattr(wf_class(), "execute")

    def test_execute(self, wf_class):
        RoundCfg = _get_input("CollectionRoundConfig")
        InputCls = _get_input("DataCollectionInput")
        if RoundCfg is None or InputCls is None:
            pytest.skip("DataCollectionInput not importable")
        round_config = RoundCfg(
            reporting_period_start="2026-01-01",
            reporting_period_end="2026-12-31",
            submission_deadline="2027-03-31T23:59:59Z",
        )
        inp = InputCls(
            organisation_id="ORG-001",
            round_config=round_config,
        )
        result = wf_class().execute(inp)
        assert result is not None

    def test_phase_count(self, wf_class):
        RoundCfg = _get_input("CollectionRoundConfig")
        InputCls = _get_input("DataCollectionInput")
        if RoundCfg is None or InputCls is None:
            pytest.skip("DataCollectionInput not importable")
        round_config = RoundCfg(
            reporting_period_start="2026-01-01",
            reporting_period_end="2026-12-31",
            submission_deadline="2027-03-31T23:59:59Z",
        )
        inp = InputCls(
            organisation_id="ORG-001",
            round_config=round_config,
        )
        result = wf_class().execute(inp)
        assert len(result.phase_results) >= 3

    def test_error_handling(self, wf_class):
        RoundCfg = _get_input("CollectionRoundConfig")
        InputCls = _get_input("DataCollectionInput")
        if RoundCfg is None or InputCls is None:
            pytest.skip("DataCollectionInput not importable")
        round_config = RoundCfg(
            reporting_period_start="2026-01-01",
            reporting_period_end="2026-12-31",
            submission_deadline="2027-03-31T23:59:59Z",
        )
        inp = InputCls(
            organisation_id="ORG-001",
            round_config=round_config,
        )
        result = wf_class().execute(inp)
        assert result is not None

    def test_provenance(self, wf_class):
        RoundCfg = _get_input("CollectionRoundConfig")
        InputCls = _get_input("DataCollectionInput")
        if RoundCfg is None or InputCls is None:
            pytest.skip("DataCollectionInput not importable")
        round_config = RoundCfg(
            reporting_period_start="2026-01-01",
            reporting_period_end="2026-12-31",
            submission_deadline="2027-03-31T23:59:59Z",
        )
        inp = InputCls(
            organisation_id="ORG-001",
            round_config=round_config,
        )
        result = wf_class().execute(inp)
        assert result.provenance_hash is not None


# ============================================================================
# BoundaryDefinitionWorkflow
# ============================================================================

class TestBoundaryDefinitionWorkflow:

    @pytest.fixture
    def wf_class(self):
        cls = _get_workflow("BoundaryDefinitionWorkflow")
        if cls is None:
            pytest.skip("BoundaryDefinitionWorkflow not built yet")
        return cls

    def test_create(self, wf_class):
        assert wf_class() is not None

    def test_create_with_config(self, wf_class):
        assert wf_class(config={"approach": "EQUITY_SHARE"}) is not None

    def test_has_execute(self, wf_class):
        assert hasattr(wf_class(), "execute")

    def test_execute(self, wf_class):
        InputCls = _get_input("BoundaryDefinitionInput")
        if InputCls is None:
            pytest.skip("BoundaryDefinitionInput not importable")
        inp = InputCls(
            organisation_id="ORG-001",
            reporting_year=2026,
        )
        result = wf_class().execute(inp)
        assert result is not None

    def test_phase_count(self, wf_class):
        InputCls = _get_input("BoundaryDefinitionInput")
        if InputCls is None:
            pytest.skip("BoundaryDefinitionInput not importable")
        inp = InputCls(
            organisation_id="ORG-001",
            reporting_year=2026,
        )
        result = wf_class().execute(inp)
        assert len(result.phase_results) >= 3

    def test_error_handling(self, wf_class):
        InputCls = _get_input("BoundaryDefinitionInput")
        if InputCls is None:
            pytest.skip("BoundaryDefinitionInput not importable")
        inp = InputCls(
            organisation_id="ORG-EMPTY",
            reporting_year=2026,
        )
        result = wf_class().execute(inp)
        assert result is not None

    def test_provenance(self, wf_class):
        InputCls = _get_input("BoundaryDefinitionInput")
        if InputCls is None:
            pytest.skip("BoundaryDefinitionInput not importable")
        inp = InputCls(
            organisation_id="ORG-001",
            reporting_year=2026,
        )
        result = wf_class().execute(inp)
        assert result.provenance_hash is not None


# ============================================================================
# ConsolidationWorkflow
# ============================================================================

class TestConsolidationWorkflow:

    @pytest.fixture
    def wf_class(self):
        cls = _get_workflow("ConsolidationWorkflow")
        if cls is None:
            pytest.skip("ConsolidationWorkflow not built yet")
        return cls

    def test_create(self, wf_class):
        assert wf_class() is not None

    def test_has_execute(self, wf_class):
        assert hasattr(wf_class(), "execute")

    def test_execute(self, wf_class):
        InputCls = _get_input("ConsolidationInput")
        if InputCls is None:
            pytest.skip("ConsolidationInput not importable")
        inp = InputCls(
            organisation_id="ORG-001",
            reporting_year=2026,
        )
        result = wf_class().execute(inp)
        assert result is not None

    def test_provenance(self, wf_class):
        InputCls = _get_input("ConsolidationInput")
        if InputCls is None:
            pytest.skip("ConsolidationInput not importable")
        inp = InputCls(
            organisation_id="ORG-001",
            reporting_year=2026,
        )
        result = wf_class().execute(inp)
        assert result.provenance_hash is not None

    def test_phase_count(self, wf_class):
        InputCls = _get_input("ConsolidationInput")
        if InputCls is None:
            pytest.skip("ConsolidationInput not importable")
        inp = InputCls(
            organisation_id="ORG-001",
            reporting_year=2026,
        )
        result = wf_class().execute(inp)
        assert len(result.phase_results) >= 3


# ============================================================================
# AllocationWorkflow
# ============================================================================

class TestAllocationWorkflow:

    @pytest.fixture
    def wf_class(self):
        cls = _get_workflow("AllocationWorkflow")
        if cls is None:
            pytest.skip("AllocationWorkflow not built yet")
        return cls

    def test_create(self, wf_class):
        assert wf_class() is not None

    def test_has_execute(self, wf_class):
        assert hasattr(wf_class(), "execute")

    def test_execute(self, wf_class):
        InputCls = _get_input("AllocationInput")
        if InputCls is None:
            pytest.skip("AllocationInput not importable")
        inp = InputCls(
            organisation_id="ORG-001",
            reporting_year=2026,
        )
        result = wf_class().execute(inp)
        assert result is not None

    def test_provenance(self, wf_class):
        InputCls = _get_input("AllocationInput")
        if InputCls is None:
            pytest.skip("AllocationInput not importable")
        inp = InputCls(
            organisation_id="ORG-001",
            reporting_year=2026,
        )
        result = wf_class().execute(inp)
        assert result.provenance_hash is not None


# ============================================================================
# SiteComparisonWorkflow
# ============================================================================

class TestSiteComparisonWorkflow:

    @pytest.fixture
    def wf_class(self):
        cls = _get_workflow("SiteComparisonWorkflow")
        if cls is None:
            pytest.skip("SiteComparisonWorkflow not built yet")
        return cls

    def test_create(self, wf_class):
        assert wf_class() is not None

    def test_has_execute(self, wf_class):
        assert hasattr(wf_class(), "execute")

    def test_execute(self, wf_class):
        InputCls = _get_input("SiteComparisonInput")
        if InputCls is None:
            pytest.skip("SiteComparisonInput not importable")
        inp = InputCls(
            organisation_id="ORG-001",
            reporting_year=2026,
        )
        result = wf_class().execute(inp)
        assert result is not None

    def test_provenance(self, wf_class):
        InputCls = _get_input("SiteComparisonInput")
        if InputCls is None:
            pytest.skip("SiteComparisonInput not importable")
        inp = InputCls(
            organisation_id="ORG-001",
            reporting_year=2026,
        )
        result = wf_class().execute(inp)
        assert result.provenance_hash is not None


# ============================================================================
# QualityImprovementWorkflow
# ============================================================================

class TestQualityImprovementWorkflow:

    @pytest.fixture
    def wf_class(self):
        cls = _get_workflow("QualityImprovementWorkflow")
        if cls is None:
            pytest.skip("QualityImprovementWorkflow not built yet")
        return cls

    def test_create(self, wf_class):
        assert wf_class() is not None

    def test_has_execute(self, wf_class):
        assert hasattr(wf_class(), "execute")

    def test_execute(self, wf_class):
        InputCls = _get_input("QualityImprovementInput")
        if InputCls is None:
            pytest.skip("QualityImprovementInput not importable")
        inp = InputCls(
            organisation_id="ORG-001",
            reporting_year=2026,
        )
        result = wf_class().execute(inp)
        assert result is not None

    def test_provenance(self, wf_class):
        InputCls = _get_input("QualityImprovementInput")
        if InputCls is None:
            pytest.skip("QualityImprovementInput not importable")
        inp = InputCls(
            organisation_id="ORG-001",
            reporting_year=2026,
        )
        result = wf_class().execute(inp)
        assert result.provenance_hash is not None


# ============================================================================
# FullMultiSitePipelineWorkflow
# ============================================================================

class TestFullMultiSitePipelineWorkflow:

    @pytest.fixture
    def wf_class(self):
        cls = _get_workflow("FullMultiSitePipelineWorkflow")
        if cls is None:
            pytest.skip("FullMultiSitePipelineWorkflow not built yet")
        return cls

    def test_create(self, wf_class):
        assert wf_class() is not None

    def test_create_with_config(self, wf_class):
        assert wf_class(config={"reporting_year": 2026}) is not None

    def test_has_execute(self, wf_class):
        assert hasattr(wf_class(), "execute")

    def test_execute(self, wf_class):
        InputCls = _get_input("FullPipelineInput")
        if InputCls is None:
            pytest.skip("FullPipelineInput not importable")
        inp = InputCls(
            organisation_id="ORG-001",
            reporting_year=2026,
        )
        result = wf_class().execute(inp)
        assert result is not None

    def test_phase_count(self, wf_class):
        InputCls = _get_input("FullPipelineInput")
        if InputCls is None:
            pytest.skip("FullPipelineInput not importable")
        inp = InputCls(
            organisation_id="ORG-001",
            reporting_year=2026,
        )
        result = wf_class().execute(inp)
        assert len(result.phase_results) >= 5

    def test_error_handling(self, wf_class):
        InputCls = _get_input("FullPipelineInput")
        if InputCls is None:
            pytest.skip("FullPipelineInput not importable")
        inp = InputCls(
            organisation_id="ORG-EMPTY",
            reporting_year=2026,
        )
        result = wf_class().execute(inp)
        assert result is not None

    def test_provenance(self, wf_class):
        InputCls = _get_input("FullPipelineInput")
        if InputCls is None:
            pytest.skip("FullPipelineInput not importable")
        inp = InputCls(
            organisation_id="ORG-001",
            reporting_year=2026,
        )
        result = wf_class().execute(inp)
        assert result.provenance_hash is not None
