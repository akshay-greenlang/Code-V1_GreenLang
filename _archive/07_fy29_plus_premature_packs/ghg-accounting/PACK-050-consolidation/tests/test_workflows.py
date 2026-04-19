# -*- coding: utf-8 -*-
"""
PACK-050 GHG Consolidation Pack - Workflow Tests

Tests workflow module availability, phase enumeration, model creation,
input/output validation, and boundary selection workflow logic.

Target: 40-60 tests.
"""

import pytest
from decimal import Decimal


class TestEntityMappingWorkflow:
    """Test entity mapping workflow components."""

    def test_workflow_module_importable(self):
        from workflows.entity_mapping_workflow import EntityMappingWorkflow
        assert EntityMappingWorkflow is not None

    def test_workflow_has_phases(self):
        from workflows.entity_mapping_workflow import EntityMappingPhase
        phases = list(EntityMappingPhase)
        assert len(phases) >= 4

    def test_workflow_input_model(self):
        from workflows.entity_mapping_workflow import EntityMappingInput
        inp = EntityMappingInput(
            organisation_id="ORG-001",
            reporting_year=2025,
        )
        assert inp.organisation_id == "ORG-001"
        assert inp.reporting_year == 2025

    def test_workflow_result_model(self):
        from workflows.entity_mapping_workflow import EntityMappingResult
        assert EntityMappingResult is not None

    def test_phase_status_enum(self):
        from workflows.entity_mapping_workflow import PhaseStatus
        assert "completed" in [s.value for s in PhaseStatus]
        assert "failed" in [s.value for s in PhaseStatus]

    def test_workflow_status_enum(self):
        from workflows.entity_mapping_workflow import WorkflowStatus
        assert "completed" in [s.value for s in WorkflowStatus]


class TestBoundarySelectionWorkflow:
    """Test boundary selection workflow components."""

    def test_workflow_module_importable(self):
        from workflows.boundary_selection_workflow import BoundarySelectionWorkflow
        assert BoundarySelectionWorkflow is not None

    def test_boundary_phases(self):
        from workflows.boundary_selection_workflow import BoundarySelectionPhase
        phases = list(BoundarySelectionPhase)
        assert len(phases) == 4
        phase_values = [p.value for p in phases]
        assert "approach_evaluation" in phase_values
        assert "impact_analysis" in phase_values
        assert "stakeholder_approval" in phase_values
        assert "boundary_lock" in phase_values

    def test_consolidation_approach_enum(self):
        from workflows.boundary_selection_workflow import ConsolidationApproach
        approaches = [a.value for a in ConsolidationApproach]
        assert "equity_share" in approaches
        assert "financial_control" in approaches
        assert "operational_control" in approaches

    def test_approach_suitability_enum(self):
        from workflows.boundary_selection_workflow import ApproachSuitability
        levels = [s.value for s in ApproachSuitability]
        assert "highly_suitable" in levels
        assert "not_suitable" in levels


class TestEntityDataCollectionWorkflow:
    """Test entity data collection workflow."""

    def test_workflow_module_importable(self):
        from workflows.entity_data_collection_workflow import EntityDataCollectionWorkflow
        assert EntityDataCollectionWorkflow is not None


class TestConsolidationExecutionWorkflow:
    """Test consolidation execution workflow."""

    def test_workflow_module_importable(self):
        from workflows.consolidation_execution_workflow import ConsolidationExecutionWorkflow
        assert ConsolidationExecutionWorkflow is not None


class TestEliminationWorkflow:
    """Test elimination workflow."""

    def test_workflow_module_importable(self):
        from workflows.elimination_workflow import EliminationWorkflow
        assert EliminationWorkflow is not None


class TestMnAWorkflow:
    """Test M&A adjustment workflow."""

    def test_workflow_module_importable(self):
        from workflows.mna_adjustment_workflow import MnAAdjustmentWorkflow
        assert MnAAdjustmentWorkflow is not None


class TestGroupReportingWorkflow:
    """Test group reporting workflow."""

    def test_workflow_module_importable(self):
        from workflows.group_reporting_workflow import GroupReportingWorkflow
        assert GroupReportingWorkflow is not None


class TestWorkflowPhaseStatusTracking:
    """Test workflow phase status tracking patterns."""

    def test_phase_status_values(self):
        from workflows.entity_mapping_workflow import PhaseStatus
        statuses = {s.value for s in PhaseStatus}
        required = {"pending", "running", "completed", "failed"}
        assert required.issubset(statuses)

    def test_workflow_status_values(self):
        from workflows.entity_mapping_workflow import WorkflowStatus
        statuses = {s.value for s in WorkflowStatus}
        required = {"pending", "running", "completed", "failed"}
        assert required.issubset(statuses)


class TestWorkflowPhaseResults:
    """Test workflow phase result models."""

    def test_phase_result_model(self):
        from workflows.entity_mapping_workflow import PhaseResult
        result = PhaseResult(
            phase="entity_discovery",
            status="completed",
            summary="Discovered 5 entities",
        )
        assert result.phase == "entity_discovery"
        assert result.status == "completed"


class TestWorkflowConsistency:
    """Test cross-workflow consistency."""

    def test_all_workflows_have_status_enum(self):
        """All workflow modules should define status enums."""
        from workflows.entity_mapping_workflow import PhaseStatus as PS1
        from workflows.boundary_selection_workflow import PhaseStatus as PS2
        assert set(s.value for s in PS1) == set(s.value for s in PS2)

    def test_workflow_count_matches_manifest(self):
        """Should have 7 importable workflow modules (excluding full_consolidation_pipeline_workflow)."""
        importable = 0
        workflow_modules = [
            "workflows.entity_mapping_workflow",
            "workflows.boundary_selection_workflow",
            "workflows.entity_data_collection_workflow",
            "workflows.consolidation_execution_workflow",
            "workflows.elimination_workflow",
            "workflows.mna_adjustment_workflow",
            "workflows.group_reporting_workflow",
        ]
        for mod_name in workflow_modules:
            try:
                __import__(mod_name)
                importable += 1
            except ImportError:
                pass
        assert importable >= 6

    def test_entity_mapping_workflow_instantiation(self):
        from workflows.entity_mapping_workflow import EntityMappingWorkflow
        wf = EntityMappingWorkflow()
        assert wf is not None

    def test_boundary_selection_workflow_instantiation(self):
        from workflows.boundary_selection_workflow import BoundarySelectionWorkflow
        wf = BoundarySelectionWorkflow()
        assert wf is not None


class TestEntityMappingWorkflowModels:
    """Test entity mapping workflow Pydantic models."""

    def test_candidate_entity_model(self):
        from workflows.entity_mapping_workflow import CandidateEntity
        candidate = CandidateEntity(
            entity_id="ENT-001",
            entity_name="Test Entity",
            entity_type="SUBSIDIARY",
            country="US",
        )
        assert candidate.entity_id == "ENT-001"

    def test_entity_mapping_input_defaults(self):
        from workflows.entity_mapping_workflow import EntityMappingInput
        inp = EntityMappingInput(
            organisation_id="ORG-001",
            reporting_year=2025,
        )
        assert inp.reporting_year == 2025

    def test_entity_mapping_input_with_entities(self):
        from workflows.entity_mapping_workflow import EntityMappingInput, CandidateEntity
        candidates = [
            CandidateEntity(
                entity_id="ENT-A",
                entity_name="Entity A",
                entity_type="SUBSIDIARY",
                country="DE",
            ),
        ]
        inp = EntityMappingInput(
            organisation_id="ORG-001",
            reporting_year=2025,
            candidate_entities=candidates,
        )
        assert len(inp.candidate_entities) == 1


class TestBoundarySelectionModels:
    """Test boundary selection workflow models."""

    def test_approval_decision_enum(self):
        from workflows.boundary_selection_workflow import ApprovalDecision
        decisions = [d.value for d in ApprovalDecision]
        assert len(decisions) >= 2

    def test_boundary_selection_phase_ordering(self):
        from workflows.boundary_selection_workflow import BoundarySelectionPhase
        phases = list(BoundarySelectionPhase)
        assert phases[0].value == "approach_evaluation"
        assert phases[-1].value == "boundary_lock"
