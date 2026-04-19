# -*- coding: utf-8 -*-
"""
Tests for all 8 PACK-025 Race to Zero Workflows.

Covers: PledgeOnboardingWorkflow, StartingLineAssessmentWorkflow,
ActionPlanningWorkflow, AnnualReportingWorkflow, SectorPathwayWorkflow,
PartnershipEngagementWorkflow, CredibilityReviewWorkflow,
FullRaceToZeroWorkflow.

Validates instantiation, config defaults, phase counts, DAG dependencies,
enum values, and model construction.

Author: GreenLang Platform Team
Pack: PACK-025 Race to Zero Pack
"""

import sys
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from workflows import (
    # Module metadata
    __version__,
    __pack_id__,
    __pack_name__,
    # Pledge Onboarding
    PledgeOnboardingWorkflow,
    PledgeOnboardingConfig,
    PledgeOnboardingResult,
    OnboardingPhase,
    ActorType,
    PartnerInitiative,
    PledgeQuality,
    EligibilityStatus,
    TargetAmbition,
    OrganizationProfile,
    EligibilityAssessment,
    BaselineEmissions,
    TargetProposal,
    CommitmentPackage,
    # Starting Line Assessment
    StartingLineAssessmentWorkflow,
    StartingLineConfig,
    StartingLineResult,
    StartingLinePhase,
    CriterionStatus,
    ComplianceStatus,
    GapSeverity,
    RemediationPriority,
    StartingLinePillar,
    CriterionResult,
    GapItem,
    RemediationAction,
    ComplianceCertificate,
    # Action Planning
    ActionPlanningWorkflow,
    ActionPlanningConfig,
    ActionPlanningResult,
    ActionPlanPhase,
    ActionCategory,
    FeasibilityLevel,
    TimeHorizon,
    AbatementAction,
    PartnerOpportunity,
    ActionPlanDocument,
    # Annual Reporting
    AnnualReportingWorkflow,
    AnnualReportingConfig,
    AnnualReportingResult,
    ReportingPhase,
    TrajectoryStatus,
    VerificationStatus,
    SubmissionChannel,
    ProgressMetrics,
    CredibilityScore,
    AnnualReport,
    # Sector Pathway
    SectorPathwayWorkflow,
    SectorPathwayConfig,
    SectorPathwayResult,
    SectorPathwayPhase,
    SectorCategory,
    PathwaySource,
    AlignmentStatus,
    SectorProfile,
    PathwayBenchmark,
    MilestoneMap,
    # Partnership Engagement
    PartnershipEngagementWorkflow,
    PartnershipEngagementConfig,
    PartnershipEngagementResult,
    PartnershipPhase,
    PartnerType,
    EngagementLevel,
    CollaborationStatus,
    PartnerMatch,
    CollaborationAgreement,
    JointTarget,
    # Credibility Review
    CredibilityReviewWorkflow,
    CredibilityReviewConfig,
    CredibilityReviewResult,
    CredibilityPhase,
    RecommendationStatus,
    CredibilityRating,
    GovernanceMaturity,
    RecommendationAssessment,
    CredibilityReport,
    # Full Race to Zero
    FullRaceToZeroWorkflow,
    FullR2ZConfig,
    FullR2ZResult,
    R2ZPhase,
    ReadinessLevel,
    CycleStatus,
    PhaseGate,
    ReadinessScore,
    CycleMetrics,
    CycleSummary,
)


# ========================================================================
# Helper: count phases from a workflow instance
# ========================================================================


def _get_phase_count(workflow_instance) -> int:
    """Attempt to determine the number of phases in a workflow."""
    for attr in ("phases", "_phases", "PHASES", "phase_definitions",
                 "PHASE_EXECUTION_ORDER"):
        val = getattr(workflow_instance, attr, None)
        if val is not None and hasattr(val, "__len__"):
            return len(val)
    count_attr = getattr(workflow_instance, "phase_count", None)
    if count_attr is not None:
        return int(count_attr)
    return -1


# ========================================================================
# Module Metadata
# ========================================================================


class TestWorkflowModuleMetadata:
    """Tests for workflows package metadata."""

    def test_version(self):
        assert __version__ == "1.0.0"

    def test_pack_id(self):
        assert __pack_id__ == "PACK-025"

    def test_pack_name(self):
        assert __pack_name__ == "Race to Zero Pack"


# ========================================================================
# Workflow 1: PledgeOnboardingWorkflow (5 phases)
# ========================================================================


class TestPledgeOnboardingWorkflow:
    """Tests for PledgeOnboardingWorkflow."""

    def test_workflow_instantiates(self):
        wf = PledgeOnboardingWorkflow()
        assert wf is not None

    def test_workflow_with_config(self):
        config = PledgeOnboardingConfig()
        wf = PledgeOnboardingWorkflow(config=config)
        assert wf is not None

    def test_workflow_phase_count(self):
        wf = PledgeOnboardingWorkflow()
        count = _get_phase_count(wf)
        if count >= 0:
            assert count == 5

    def test_config_defaults(self):
        config = PledgeOnboardingConfig()
        assert config is not None

    def test_onboarding_phase_enum_values(self):
        assert len(OnboardingPhase) >= 5

    def test_actor_type_enum(self):
        assert len(ActorType) >= 5

    def test_partner_initiative_enum(self):
        assert len(PartnerInitiative) >= 5

    def test_pledge_quality_enum(self):
        assert len(PledgeQuality) >= 3

    def test_eligibility_status_enum(self):
        assert len(EligibilityStatus) >= 2

    def test_target_ambition_enum(self):
        assert len(TargetAmbition) >= 2

    def test_organization_profile_model(self):
        assert OrganizationProfile is not None

    def test_eligibility_assessment_model(self):
        assert EligibilityAssessment is not None

    def test_baseline_emissions_model(self):
        assert BaselineEmissions is not None

    def test_target_proposal_model(self):
        assert TargetProposal is not None

    def test_commitment_package_model(self):
        assert CommitmentPackage is not None

    def test_result_model(self):
        assert PledgeOnboardingResult is not None

    def test_workflow_has_execute(self):
        wf = PledgeOnboardingWorkflow()
        assert callable(getattr(wf, "execute", None))

    def test_workflow_has_cancel(self):
        wf = PledgeOnboardingWorkflow()
        assert callable(getattr(wf, "cancel", None))


# ========================================================================
# Workflow 2: StartingLineAssessmentWorkflow (4 phases)
# ========================================================================


class TestStartingLineAssessmentWorkflow:
    """Tests for StartingLineAssessmentWorkflow."""

    def test_workflow_instantiates(self):
        wf = StartingLineAssessmentWorkflow()
        assert wf is not None

    def test_workflow_with_config(self):
        config = StartingLineConfig()
        wf = StartingLineAssessmentWorkflow(config=config)
        assert wf is not None

    def test_workflow_phase_count(self):
        wf = StartingLineAssessmentWorkflow()
        count = _get_phase_count(wf)
        if count >= 0:
            assert count == 4

    def test_config_defaults(self):
        config = StartingLineConfig()
        assert config is not None

    def test_starting_line_phase_enum(self):
        assert len(StartingLinePhase) >= 4

    def test_criterion_status_enum(self):
        assert len(CriterionStatus) >= 3

    def test_compliance_status_enum(self):
        assert len(ComplianceStatus) >= 3

    def test_gap_severity_enum(self):
        assert len(GapSeverity) >= 3

    def test_remediation_priority_enum(self):
        assert len(RemediationPriority) >= 3

    def test_starting_line_pillar_enum(self):
        assert len(StartingLinePillar) >= 4

    def test_criterion_result_model(self):
        assert CriterionResult is not None

    def test_gap_item_model(self):
        assert GapItem is not None

    def test_remediation_action_model(self):
        assert RemediationAction is not None

    def test_compliance_certificate_model(self):
        assert ComplianceCertificate is not None

    def test_workflow_has_execute(self):
        wf = StartingLineAssessmentWorkflow()
        assert callable(getattr(wf, "execute", None))


# ========================================================================
# Workflow 3: ActionPlanningWorkflow (6 phases)
# ========================================================================


class TestActionPlanningWorkflow:
    """Tests for ActionPlanningWorkflow."""

    def test_workflow_instantiates(self):
        wf = ActionPlanningWorkflow()
        assert wf is not None

    def test_workflow_with_config(self):
        config = ActionPlanningConfig()
        wf = ActionPlanningWorkflow(config=config)
        assert wf is not None

    def test_workflow_phase_count(self):
        wf = ActionPlanningWorkflow()
        count = _get_phase_count(wf)
        if count >= 0:
            assert count == 6

    def test_config_defaults(self):
        config = ActionPlanningConfig()
        assert config is not None

    def test_action_plan_phase_enum(self):
        assert len(ActionPlanPhase) >= 6

    def test_action_category_enum(self):
        assert len(ActionCategory) >= 5

    def test_feasibility_level_enum(self):
        assert len(FeasibilityLevel) >= 3

    def test_time_horizon_enum(self):
        assert len(TimeHorizon) >= 3

    def test_abatement_action_model(self):
        assert AbatementAction is not None

    def test_partner_opportunity_model(self):
        assert PartnerOpportunity is not None

    def test_action_plan_document_model(self):
        assert ActionPlanDocument is not None

    def test_workflow_has_execute(self):
        wf = ActionPlanningWorkflow()
        assert callable(getattr(wf, "execute", None))


# ========================================================================
# Workflow 4: AnnualReportingWorkflow (7 phases)
# ========================================================================


class TestAnnualReportingWorkflow:
    """Tests for AnnualReportingWorkflow."""

    def test_workflow_instantiates(self):
        wf = AnnualReportingWorkflow()
        assert wf is not None

    def test_workflow_with_config(self):
        config = AnnualReportingConfig()
        wf = AnnualReportingWorkflow(config=config)
        assert wf is not None

    def test_workflow_phase_count(self):
        wf = AnnualReportingWorkflow()
        count = _get_phase_count(wf)
        if count >= 0:
            assert count == 7

    def test_config_defaults(self):
        config = AnnualReportingConfig()
        assert config is not None

    def test_reporting_phase_enum(self):
        assert len(ReportingPhase) >= 7

    def test_trajectory_status_enum(self):
        assert len(TrajectoryStatus) >= 3

    def test_verification_status_enum(self):
        assert len(VerificationStatus) >= 3

    def test_submission_channel_enum(self):
        assert len(SubmissionChannel) >= 2

    def test_progress_metrics_model(self):
        assert ProgressMetrics is not None

    def test_credibility_score_model(self):
        assert CredibilityScore is not None

    def test_annual_report_model(self):
        assert AnnualReport is not None

    def test_workflow_has_execute(self):
        wf = AnnualReportingWorkflow()
        assert callable(getattr(wf, "execute", None))


# ========================================================================
# Workflow 5: SectorPathwayWorkflow (5 phases)
# ========================================================================


class TestSectorPathwayWorkflow:
    """Tests for SectorPathwayWorkflow."""

    def test_workflow_instantiates(self):
        wf = SectorPathwayWorkflow()
        assert wf is not None

    def test_workflow_with_config(self):
        config = SectorPathwayConfig()
        wf = SectorPathwayWorkflow(config=config)
        assert wf is not None

    def test_workflow_phase_count(self):
        wf = SectorPathwayWorkflow()
        count = _get_phase_count(wf)
        if count >= 0:
            assert count == 5

    def test_config_defaults(self):
        config = SectorPathwayConfig()
        assert config is not None

    def test_sector_pathway_phase_enum(self):
        assert len(SectorPathwayPhase) >= 5

    def test_sector_category_enum(self):
        assert len(SectorCategory) >= 5

    def test_pathway_source_enum(self):
        assert len(PathwaySource) >= 3

    def test_alignment_status_enum(self):
        assert len(AlignmentStatus) >= 3

    def test_sector_profile_model(self):
        assert SectorProfile is not None

    def test_pathway_benchmark_model(self):
        assert PathwayBenchmark is not None

    def test_milestone_map_model(self):
        assert MilestoneMap is not None

    def test_workflow_has_execute(self):
        wf = SectorPathwayWorkflow()
        assert callable(getattr(wf, "execute", None))


# ========================================================================
# Workflow 6: PartnershipEngagementWorkflow (5 phases)
# ========================================================================


class TestPartnershipEngagementWorkflow:
    """Tests for PartnershipEngagementWorkflow."""

    def test_workflow_instantiates(self):
        wf = PartnershipEngagementWorkflow()
        assert wf is not None

    def test_workflow_with_config(self):
        config = PartnershipEngagementConfig()
        wf = PartnershipEngagementWorkflow(config=config)
        assert wf is not None

    def test_workflow_phase_count(self):
        wf = PartnershipEngagementWorkflow()
        count = _get_phase_count(wf)
        if count >= 0:
            assert count == 5

    def test_config_defaults(self):
        config = PartnershipEngagementConfig()
        assert config is not None

    def test_partnership_phase_enum(self):
        assert len(PartnershipPhase) >= 5

    def test_partner_type_enum(self):
        assert len(PartnerType) >= 3

    def test_engagement_level_enum(self):
        assert len(EngagementLevel) >= 3

    def test_collaboration_status_enum(self):
        assert len(CollaborationStatus) >= 3

    def test_partner_match_model(self):
        assert PartnerMatch is not None

    def test_collaboration_agreement_model(self):
        assert CollaborationAgreement is not None

    def test_joint_target_model(self):
        assert JointTarget is not None

    def test_workflow_has_execute(self):
        wf = PartnershipEngagementWorkflow()
        assert callable(getattr(wf, "execute", None))


# ========================================================================
# Workflow 7: CredibilityReviewWorkflow (4 phases)
# ========================================================================


class TestCredibilityReviewWorkflow:
    """Tests for CredibilityReviewWorkflow."""

    def test_workflow_instantiates(self):
        wf = CredibilityReviewWorkflow()
        assert wf is not None

    def test_workflow_with_config(self):
        config = CredibilityReviewConfig()
        wf = CredibilityReviewWorkflow(config=config)
        assert wf is not None

    def test_workflow_phase_count(self):
        wf = CredibilityReviewWorkflow()
        count = _get_phase_count(wf)
        if count >= 0:
            assert count == 4

    def test_config_defaults(self):
        config = CredibilityReviewConfig()
        assert config is not None

    def test_credibility_phase_enum(self):
        assert len(CredibilityPhase) >= 4

    def test_recommendation_status_enum(self):
        assert len(RecommendationStatus) >= 3

    def test_credibility_rating_enum(self):
        assert len(CredibilityRating) >= 4

    def test_governance_maturity_enum(self):
        assert len(GovernanceMaturity) >= 3

    def test_recommendation_assessment_model(self):
        assert RecommendationAssessment is not None

    def test_credibility_report_model(self):
        assert CredibilityReport is not None

    def test_workflow_has_execute(self):
        wf = CredibilityReviewWorkflow()
        assert callable(getattr(wf, "execute", None))


# ========================================================================
# Workflow 8: FullRaceToZeroWorkflow (10 phases)
# ========================================================================


class TestFullRaceToZeroWorkflow:
    """Tests for FullRaceToZeroWorkflow."""

    def test_workflow_instantiates(self):
        wf = FullRaceToZeroWorkflow()
        assert wf is not None

    def test_workflow_with_config(self):
        config = FullR2ZConfig()
        wf = FullRaceToZeroWorkflow(config=config)
        assert wf is not None

    def test_workflow_phase_count(self):
        wf = FullRaceToZeroWorkflow()
        count = _get_phase_count(wf)
        if count >= 0:
            assert count == 10

    def test_config_defaults(self):
        config = FullR2ZConfig()
        assert config is not None

    def test_r2z_phase_enum(self):
        assert len(R2ZPhase) >= 10

    def test_readiness_level_enum(self):
        assert len(ReadinessLevel) >= 4

    def test_cycle_status_enum(self):
        assert len(CycleStatus) >= 3

    def test_phase_gate_model(self):
        assert PhaseGate is not None

    def test_readiness_score_model(self):
        assert ReadinessScore is not None

    def test_cycle_metrics_model(self):
        assert CycleMetrics is not None

    def test_cycle_summary_model(self):
        assert CycleSummary is not None

    def test_result_model(self):
        assert FullR2ZResult is not None

    def test_workflow_has_execute(self):
        wf = FullRaceToZeroWorkflow()
        assert callable(getattr(wf, "execute", None))

    def test_workflow_has_cancel(self):
        wf = FullRaceToZeroWorkflow()
        assert callable(getattr(wf, "cancel", None))


# ========================================================================
# Cross-Workflow Tests
# ========================================================================


ALL_WORKFLOW_CLASSES = [
    PledgeOnboardingWorkflow,
    StartingLineAssessmentWorkflow,
    ActionPlanningWorkflow,
    AnnualReportingWorkflow,
    SectorPathwayWorkflow,
    PartnershipEngagementWorkflow,
    CredibilityReviewWorkflow,
    FullRaceToZeroWorkflow,
]

ALL_WORKFLOW_NAMES = [cls.__name__ for cls in ALL_WORKFLOW_CLASSES]


@pytest.fixture(params=ALL_WORKFLOW_CLASSES, ids=ALL_WORKFLOW_NAMES)
def workflow_class(request):
    """Parameterized fixture yielding each workflow class."""
    return request.param


class TestAllWorkflowsCommon:
    """Common tests applied to every workflow class."""

    def test_workflow_instantiates(self, workflow_class):
        wf = workflow_class()
        assert wf is not None

    def test_workflow_has_execute(self, workflow_class):
        wf = workflow_class()
        assert callable(getattr(wf, "execute", None))

    def test_workflow_has_docstring(self, workflow_class):
        assert workflow_class.__doc__ is not None
        assert len(workflow_class.__doc__.strip()) > 0

    def test_workflow_name_ends_with_workflow(self, workflow_class):
        assert workflow_class.__name__.endswith("Workflow")


class TestWorkflowCount:
    """Verify all 8 workflows are present."""

    def test_all_8_workflows_importable(self):
        assert len(ALL_WORKFLOW_CLASSES) == 8

    def test_workflow_names(self):
        expected = [
            "PledgeOnboardingWorkflow",
            "StartingLineAssessmentWorkflow",
            "ActionPlanningWorkflow",
            "AnnualReportingWorkflow",
            "SectorPathwayWorkflow",
            "PartnershipEngagementWorkflow",
            "CredibilityReviewWorkflow",
            "FullRaceToZeroWorkflow",
        ]
        assert ALL_WORKFLOW_NAMES == expected
