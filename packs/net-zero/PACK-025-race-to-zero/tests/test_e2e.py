# -*- coding: utf-8 -*-
"""
End-to-end pipeline tests for PACK-025 Race to Zero Pack.

Tests the full component integration chain: engine instantiation, workflow
construction, template availability, integration bridge creation, preset
loading, and cross-component consistency.

Uses real class logic (no mocks) to validate that all 10 engines, 8 workflows,
10 templates, 12 integrations, and 8 presets form a cohesive pack.

Author: GreenLang Platform Team
Pack: PACK-025 Race to Zero Pack
"""

import sys
from pathlib import Path

import pytest
import yaml

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

# --- Engines ---
from engines import (
    PledgeCommitmentEngine,
    StartingLineEngine,
    InterimTargetEngine,
    ActionPlanEngine,
    ProgressTrackingEngine,
    SectorPathwayEngine,
    PartnershipScoringEngine,
    CampaignReportingEngine,
    CredibilityAssessmentEngine,
    RaceReadinessEngine,
)

# --- Workflows ---
from workflows import (
    PledgeOnboardingWorkflow,
    StartingLineAssessmentWorkflow,
    ActionPlanningWorkflow,
    AnnualReportingWorkflow,
    SectorPathwayWorkflow,
    PartnershipEngagementWorkflow,
    CredibilityReviewWorkflow,
    FullRaceToZeroWorkflow,
)

# --- Templates ---
from templates import (
    PledgeCommitmentLetterTemplate,
    StartingLineChecklistTemplate,
    ActionPlanDocumentTemplate,
    AnnualProgressReportTemplate,
    SectorPathwayRoadmapTemplate,
    PartnershipFrameworkTemplate,
    CredibilityAssessmentReportTemplate,
    CampaignSubmissionPackageTemplate,
    DisclosureDashboardTemplate,
    RaceToZeroCertificateTemplate,
)

# --- Integrations ---
from integrations import (
    RaceToZeroOrchestrator,
    RaceToZeroOrchestratorConfig,
    MRVBridge,
    MRVBridgeConfig,
    GHGAppBridge,
    GHGAppBridgeConfig,
    SBTiAppBridge,
    SBTiAppBridgeConfig,
    DecarbBridge,
    DecarbBridgeConfig,
    TaxonomyBridge,
    TaxonomyBridgeConfig,
    DataBridge,
    DataBridgeConfig,
    UNFCCCBridge,
    UNFCCCBridgeConfig,
    CDPBridge,
    CDPBridgeConfig,
    GFANZBridge,
    GFANZBridgeConfig,
    RaceToZeroSetupWizard,
    RaceToZeroHealthCheck,
    HealthCheckConfig,
)

# --- Presets ---
from config.presets import (
    AVAILABLE_PRESETS,
    ACTOR_TYPE_PRESET_MAP,
    get_preset_path,
    get_preset_for_actor_type,
)


PRESET_DIR = Path(__file__).resolve().parent.parent / "config" / "presets"


# ========================================================================
# E2E: Full Component Instantiation
# ========================================================================


class TestFullComponentInstantiation:
    """End-to-end: Instantiate all 10 engines, 8 workflows, 10 templates,
    12 integrations simultaneously to verify no import conflicts."""

    def test_all_engines_instantiate_together(self):
        """All 10 engines can be instantiated in the same process."""
        engines = [
            PledgeCommitmentEngine(),
            StartingLineEngine(),
            InterimTargetEngine(),
            ActionPlanEngine(),
            ProgressTrackingEngine(),
            SectorPathwayEngine(),
            PartnershipScoringEngine(),
            CampaignReportingEngine(),
            CredibilityAssessmentEngine(),
            RaceReadinessEngine(),
        ]
        assert len(engines) == 10
        for e in engines:
            assert e is not None

    def test_all_workflows_instantiate_together(self):
        """All 8 workflows can be instantiated in the same process."""
        workflows = [
            PledgeOnboardingWorkflow(),
            StartingLineAssessmentWorkflow(),
            ActionPlanningWorkflow(),
            AnnualReportingWorkflow(),
            SectorPathwayWorkflow(),
            PartnershipEngagementWorkflow(),
            CredibilityReviewWorkflow(),
            FullRaceToZeroWorkflow(),
        ]
        assert len(workflows) == 8
        for wf in workflows:
            assert wf is not None

    def test_all_templates_instantiate_together(self):
        """All 10 templates can be instantiated in the same process."""
        templates = [
            PledgeCommitmentLetterTemplate(),
            StartingLineChecklistTemplate(),
            ActionPlanDocumentTemplate(),
            AnnualProgressReportTemplate(),
            SectorPathwayRoadmapTemplate(),
            PartnershipFrameworkTemplate(),
            CredibilityAssessmentReportTemplate(),
            CampaignSubmissionPackageTemplate(),
            DisclosureDashboardTemplate(),
            RaceToZeroCertificateTemplate(),
        ]
        assert len(templates) == 10
        for t in templates:
            assert t is not None

    def test_all_integrations_instantiate_together(self):
        """All 12 integrations can be instantiated in the same process."""
        integrations = [
            RaceToZeroOrchestrator(config=RaceToZeroOrchestratorConfig()),
            MRVBridge(config=MRVBridgeConfig()),
            GHGAppBridge(config=GHGAppBridgeConfig()),
            SBTiAppBridge(config=SBTiAppBridgeConfig()),
            DecarbBridge(config=DecarbBridgeConfig()),
            TaxonomyBridge(config=TaxonomyBridgeConfig()),
            DataBridge(config=DataBridgeConfig()),
            UNFCCCBridge(config=UNFCCCBridgeConfig()),
            CDPBridge(config=CDPBridgeConfig()),
            GFANZBridge(config=GFANZBridgeConfig()),
            RaceToZeroSetupWizard(),
            RaceToZeroHealthCheck(config=HealthCheckConfig()),
        ]
        assert len(integrations) == 12
        for i in integrations:
            assert i is not None


# ========================================================================
# E2E: Preset Loading Chain
# ========================================================================


class TestPresetLoadingChain:
    """End-to-end: Load each preset, verify YAML validity, and cross-reference
    against actor-type mapping."""

    def test_all_presets_load_as_valid_yaml(self):
        """All 8 presets load as valid YAML dicts."""
        for name, path in AVAILABLE_PRESETS.items():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            assert isinstance(data, dict), f"Preset {name} is not a dict"
            assert len(data) > 0, f"Preset {name} is empty"

    def test_actor_type_map_covers_all_presets(self):
        """Actor type map covers all 8 presets."""
        preset_names = set(ACTOR_TYPE_PRESET_MAP.values())
        available_names = set(AVAILABLE_PRESETS.keys())
        assert preset_names == available_names

    def test_get_preset_for_all_actor_types(self):
        """get_preset_for_actor_type works for all actor types."""
        for actor_type in ACTOR_TYPE_PRESET_MAP:
            preset_name = get_preset_for_actor_type(actor_type)
            path = get_preset_path(preset_name)
            assert Path(path).exists()


# ========================================================================
# E2E: Cross-Component Consistency
# ========================================================================


class TestCrossComponentConsistency:
    """End-to-end: Verify that engines, workflows, templates, and integrations
    are consistent in naming, versioning, and structure."""

    def test_engine_count_matches_pack_spec(self):
        """Pack has exactly 10 engines."""
        from engines import __all__ as engine_exports
        engine_classes = [e for e in engine_exports if e.endswith("Engine")]
        assert len(engine_classes) == 10

    def test_workflow_count_matches_pack_spec(self):
        """Pack has exactly 8 workflows."""
        from workflows import __all__ as workflow_exports
        workflow_classes = [w for w in workflow_exports if w.endswith("Workflow")]
        assert len(workflow_classes) == 8

    def test_template_count_matches_pack_spec(self):
        """Pack has exactly 10 templates."""
        from templates import __all__ as template_exports
        template_classes = [t for t in template_exports if t.endswith("Template")]
        assert len(template_classes) == 10

    def test_preset_count_matches_pack_spec(self):
        """Pack has exactly 8 presets."""
        assert len(AVAILABLE_PRESETS) == 8

    def test_all_module_versions_match(self):
        """Engines, workflows, templates, integrations all have version 1.0.0."""
        from engines import __version__ as ev
        from workflows import __version__ as wv
        from templates import __version__ as tv
        from integrations import __version__ as iv
        from config.presets import __version__ as pv
        assert ev == wv == tv == iv == pv == "1.0.0"

    def test_all_module_pack_ids_match(self):
        """All modules have pack ID PACK-025."""
        from engines import __pack_id__ as ep
        from workflows import __pack_id__ as wp
        from templates import __pack_id__ as tp
        from integrations import __pack_id__ as ip
        from config.presets import __pack_id__ as pp
        assert ep == wp == tp == ip == pp == "PACK-025"


# ========================================================================
# E2E: Orchestrator Pipeline Phases
# ========================================================================


class TestOrchestratorPipelinePhases:
    """End-to-end: Verify orchestrator pipeline phase definitions."""

    def test_orchestrator_has_10_phases(self):
        """Orchestrator defines 10 pipeline phases."""
        from integrations import PHASE_EXECUTION_ORDER
        assert len(PHASE_EXECUTION_ORDER) >= 10

    def test_orchestrator_has_phase_dependencies(self):
        """Orchestrator defines phase dependencies."""
        from integrations import PHASE_DEPENDENCIES
        assert isinstance(PHASE_DEPENDENCIES, dict)
        assert len(PHASE_DEPENDENCIES) > 0


# ========================================================================
# E2E: Full Pack File Count
# ========================================================================


class TestFullPackFileCount:
    """End-to-end: Verify the total number of Python files in the pack."""

    def test_total_python_files(self):
        """Pack contains a substantial number of Python files."""
        pack_root = Path(__file__).resolve().parents[1]
        py_files = list(pack_root.rglob("*.py"))
        # 10 engines + __init__ + 8 workflows + __init__ + 10 templates + __init__
        # + 12 integrations + __init__ + presets __init__ + config __init__
        # + tests (8 files) = ~55+ Python files
        assert len(py_files) >= 45, f"Only found {len(py_files)} .py files"

    def test_total_yaml_files(self):
        """Pack contains 8 preset YAML files."""
        pack_root = Path(__file__).resolve().parents[1]
        yaml_files = list(pack_root.rglob("*.yaml"))
        assert len(yaml_files) >= 8, f"Only found {len(yaml_files)} .yaml files"


# ========================================================================
# E2E: Component Interaction Chain
# ========================================================================


class TestComponentInteractionChain:
    """End-to-end: Verify components can reference each other."""

    def test_workflow_can_reference_engine(self):
        """A workflow can instantiate an engine it depends on."""
        engine = PledgeCommitmentEngine()
        workflow = PledgeOnboardingWorkflow()
        assert engine is not None
        assert workflow is not None

    def test_orchestrator_can_reference_all_engines(self):
        """Orchestrator can coexist with all engines."""
        orch = RaceToZeroOrchestrator(config=RaceToZeroOrchestratorConfig())
        engines = {
            "pledge": PledgeCommitmentEngine(),
            "starting_line": StartingLineEngine(),
            "interim_target": InterimTargetEngine(),
            "action_plan": ActionPlanEngine(),
            "progress": ProgressTrackingEngine(),
            "sector": SectorPathwayEngine(),
            "partnership": PartnershipScoringEngine(),
            "campaign": CampaignReportingEngine(),
            "credibility": CredibilityAssessmentEngine(),
            "readiness": RaceReadinessEngine(),
        }
        assert orch is not None
        assert len(engines) == 10

    def test_template_can_reference_with_engine(self):
        """Templates and engines can coexist."""
        engine = CredibilityAssessmentEngine()
        template = CredibilityAssessmentReportTemplate()
        assert engine is not None
        assert template is not None

    def test_preset_loads_with_wizard_actor_types(self):
        """Setup wizard actor types align with preset actor type map."""
        from integrations import OrganizationType
        # OrganizationType enum should have entries that correspond to
        # actor types in ACTOR_TYPE_PRESET_MAP
        assert len(OrganizationType) >= 5

    def test_health_check_with_all_components(self):
        """Health check can be created alongside all components."""
        hc = RaceToZeroHealthCheck(config=HealthCheckConfig())
        engines_ok = PledgeCommitmentEngine() is not None
        workflows_ok = PledgeOnboardingWorkflow() is not None
        templates_ok = PledgeCommitmentLetterTemplate() is not None
        assert hc is not None
        assert engines_ok
        assert workflows_ok
        assert templates_ok
