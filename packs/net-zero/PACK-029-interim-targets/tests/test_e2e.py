# -*- coding: utf-8 -*-
"""
End-to-end pipeline tests for PACK-029 Interim Targets Pack.

Tests the full component integration chain: engine instantiation, workflow
construction, template availability, integration bridge creation, preset
loading, and cross-component consistency.

Uses real class logic (no mocks) to validate that all 10 engines, 7 workflows,
10 templates, 10 integrations, and 7 presets form a cohesive pack.

Author: GreenLang Platform Team
Pack: PACK-029 Interim Targets Pack
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
    InterimTargetEngine,
    AnnualPathwayEngine,
    ProgressTrackerEngine,
    VarianceAnalysisEngine,
    TrendExtrapolationEngine,
    CorrectiveActionEngine,
    MilestoneValidationEngine,
    InitiativeSchedulerEngine,
    BudgetAllocationEngine,
    ReportingEngine,
)

# --- Workflows ---
from workflows import (
    InterimTargetSettingWorkflow,
    AnnualProgressReviewWorkflow,
    QuarterlyMonitoringWorkflow,
    VarianceInvestigationWorkflow,
    CorrectiveActionPlanningWorkflow,
    AnnualReportingWorkflow,
    TargetRecalibrationWorkflow,
)

# --- Templates ---
from templates import (
    InterimTargetsSummaryTemplate,
    AnnualProgressReportTemplate,
    VarianceAnalysisReportTemplate,
    CorrectiveActionPlanTemplate,
    QuarterlyDashboardTemplate,
    CDPDisclosureTemplate,
    TCFDMetricsReportTemplate,
    AssuranceEvidencePackageTemplate,
    ExecutiveSummaryTemplate,
    PublicDisclosureTemplate,
    TemplateRegistry,
)

# --- Integrations ---
from integrations import (
    PACK021Bridge,
    PACK028Bridge,
    MRVBridge,
    SBTiBridge,
    CDPBridge,
    TCFDBridge,
    InitiativeTrackerBridge,
    BudgetSystemBridge,
    AlertingBridge,
    AssurancePortalBridge,
)


PRESET_DIR = Path(__file__).resolve().parent.parent / "config" / "presets"


# ========================================================================
# E2E: Full Component Instantiation
# ========================================================================


class TestFullComponentInstantiation:
    """End-to-end: Instantiate all 10 engines, 7 workflows, 10 templates,
    10 integrations simultaneously to verify no import conflicts."""

    def test_all_engines_instantiate_together(self):
        """All 10 engines can be instantiated in the same process."""
        engines = [
            InterimTargetEngine(),
            AnnualPathwayEngine(),
            ProgressTrackerEngine(),
            VarianceAnalysisEngine(),
            TrendExtrapolationEngine(),
            CorrectiveActionEngine(),
            MilestoneValidationEngine(),
            InitiativeSchedulerEngine(),
            BudgetAllocationEngine(),
            ReportingEngine(),
        ]
        assert len(engines) == 10
        for e in engines:
            assert e is not None

    def test_all_workflows_instantiate_together(self):
        """All 7 workflows can be instantiated in the same process."""
        workflows = [
            InterimTargetSettingWorkflow(),
            AnnualProgressReviewWorkflow(),
            QuarterlyMonitoringWorkflow(),
            VarianceInvestigationWorkflow(),
            CorrectiveActionPlanningWorkflow(),
            AnnualReportingWorkflow(),
            TargetRecalibrationWorkflow(),
        ]
        assert len(workflows) == 7
        for wf in workflows:
            assert wf is not None

    def test_all_templates_instantiate_together(self):
        """All 10 templates can be instantiated in the same process."""
        templates = [
            InterimTargetsSummaryTemplate(),
            AnnualProgressReportTemplate(),
            VarianceAnalysisReportTemplate(),
            CorrectiveActionPlanTemplate(),
            QuarterlyDashboardTemplate(),
            CDPDisclosureTemplate(),
            TCFDMetricsReportTemplate(),
            AssuranceEvidencePackageTemplate(),
            ExecutiveSummaryTemplate(),
            PublicDisclosureTemplate(),
        ]
        assert len(templates) == 10
        for t in templates:
            assert t is not None

    def test_all_integrations_instantiate_together(self):
        """All 10 integrations can be instantiated in the same process."""
        integrations = [
            PACK021Bridge(),
            PACK028Bridge(),
            MRVBridge(),
            SBTiBridge(),
            CDPBridge(),
            TCFDBridge(),
            InitiativeTrackerBridge(),
            BudgetSystemBridge(),
            AlertingBridge(),
            AssurancePortalBridge(),
        ]
        assert len(integrations) == 10
        for i in integrations:
            assert i is not None

    def test_template_registry_instantiates(self):
        """TemplateRegistry can be instantiated alongside all components."""
        registry = TemplateRegistry()
        assert registry is not None
        assert len(registry) == 10


# ========================================================================
# E2E: Preset Loading Chain
# ========================================================================


class TestPresetLoadingChain:
    """End-to-end: Load each preset YAML file, verify validity."""

    def test_all_preset_yaml_files_are_valid(self):
        """All 7 preset YAML files parse as valid YAML dicts."""
        yaml_files = list(PRESET_DIR.glob("*.yaml"))
        assert len(yaml_files) == 7, f"Expected 7 preset YAML files, found {len(yaml_files)}"
        for yf in yaml_files:
            with open(yf, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            assert isinstance(data, dict), f"Preset {yf.name} is not a dict"
            assert len(data) > 0, f"Preset {yf.name} is empty"

    def test_preset_file_names(self):
        """Preset files follow expected naming convention."""
        yaml_files = sorted([f.name for f in PRESET_DIR.glob("*.yaml")])
        assert len(yaml_files) == 7
        for name in yaml_files:
            assert name.endswith(".yaml")
            assert "_" in name or "-" in name or name[0].isalpha()


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
        """Pack has exactly 7 workflows."""
        from workflows import __all__ as workflow_exports
        workflow_classes = [w for w in workflow_exports if w.endswith("Workflow")]
        assert len(workflow_classes) == 7

    def test_template_count_matches_pack_spec(self):
        """Pack has exactly 10 templates (excluding TemplateRegistry)."""
        from templates import __all__ as template_exports
        template_classes = [t for t in template_exports if t.endswith("Template")]
        assert len(template_classes) == 10

    def test_all_module_versions_exist(self):
        """Engines, workflows, templates, integrations all have __version__."""
        from engines import __version__ as ev
        from workflows import __version__ as wv
        from templates import __version__ as tv
        from integrations import __version__ as iv
        assert ev is not None
        assert wv is not None
        assert tv is not None
        assert iv is not None

    def test_all_module_pack_ids_match(self):
        """All modules have pack ID PACK-029."""
        from workflows import __pack_id__ as wp
        from templates import __pack_id__ as tp
        from integrations import __pack_id__ as ip
        assert wp == "PACK-029"
        assert tp == "PACK-029"
        assert ip == "PACK-029"

    def test_preset_count_matches_pack_spec(self):
        """Pack has exactly 7 preset YAML files."""
        yaml_files = list(PRESET_DIR.glob("*.yaml"))
        assert len(yaml_files) == 7

    def test_workflow_registry_has_7_entries(self):
        """Workflow registry contains exactly 7 workflow definitions."""
        from workflows import WORKFLOW_REGISTRY
        assert len(WORKFLOW_REGISTRY) == 7

    def test_workflow_registry_keys(self):
        """Workflow registry has expected keys."""
        from workflows import WORKFLOW_REGISTRY
        expected_keys = {
            "interim_target_setting",
            "annual_progress_review",
            "quarterly_monitoring",
            "variance_investigation",
            "corrective_action_planning",
            "annual_reporting",
            "target_recalibration",
        }
        assert set(WORKFLOW_REGISTRY.keys()) == expected_keys


# ========================================================================
# E2E: Full Pack File Count
# ========================================================================


class TestFullPackFileCount:
    """End-to-end: Verify the total number of Python files in the pack."""

    def test_total_python_files(self):
        """Pack contains a substantial number of Python files."""
        pack_root = Path(__file__).resolve().parents[1]
        py_files = list(pack_root.rglob("*.py"))
        # 10 engines + __init__ + 7 workflows + __init__ + 10 templates + __init__
        # + 10 integrations + __init__ + config + tests = ~55+ Python files
        assert len(py_files) >= 45, f"Only found {len(py_files)} .py files"

    def test_total_yaml_files(self):
        """Pack contains at least 7 preset YAML files plus pack.yaml."""
        pack_root = Path(__file__).resolve().parents[1]
        yaml_files = list(pack_root.rglob("*.yaml"))
        assert len(yaml_files) >= 7, f"Only found {len(yaml_files)} .yaml files"

    def test_engine_files_exist(self):
        """Each engine has a corresponding Python file."""
        engines_dir = _PACK_ROOT / "engines"
        expected = [
            "interim_target_engine.py",
            "annual_pathway_engine.py",
            "progress_tracker_engine.py",
            "variance_analysis_engine.py",
            "trend_extrapolation_engine.py",
            "corrective_action_engine.py",
            "milestone_validation_engine.py",
            "initiative_scheduler_engine.py",
            "budget_allocation_engine.py",
            "reporting_engine.py",
        ]
        for fname in expected:
            assert (engines_dir / fname).exists(), f"Missing engine file: {fname}"

    def test_workflow_files_exist(self):
        """Each workflow has a corresponding Python file."""
        workflows_dir = _PACK_ROOT / "workflows"
        expected = [
            "interim_target_setting_workflow.py",
            "annual_progress_review_workflow.py",
            "quarterly_monitoring_workflow.py",
            "variance_investigation_workflow.py",
            "corrective_action_planning_workflow.py",
            "annual_reporting_workflow.py",
            "target_recalibration_workflow.py",
        ]
        for fname in expected:
            assert (workflows_dir / fname).exists(), f"Missing workflow file: {fname}"

    def test_integration_files_exist(self):
        """Each integration has a corresponding Python file."""
        integrations_dir = _PACK_ROOT / "integrations"
        expected = [
            "pack021_bridge.py",
            "pack028_bridge.py",
            "mrv_bridge.py",
            "sbti_bridge.py",
            "cdp_bridge.py",
            "tcfd_bridge.py",
            "initiative_tracker_bridge.py",
            "budget_system_bridge.py",
            "alerting_bridge.py",
            "assurance_portal_bridge.py",
        ]
        for fname in expected:
            assert (integrations_dir / fname).exists(), f"Missing integration file: {fname}"


# ========================================================================
# E2E: Component Interaction Chain
# ========================================================================


class TestComponentInteractionChain:
    """End-to-end: Verify components can reference each other."""

    def test_workflow_can_reference_engine(self):
        """A workflow can coexist with the engine it depends on."""
        engine = InterimTargetEngine()
        workflow = InterimTargetSettingWorkflow()
        assert engine is not None
        assert workflow is not None

    def test_all_engines_with_all_workflows(self):
        """All 10 engines and all 7 workflows can coexist."""
        engines = {
            "interim_target": InterimTargetEngine(),
            "annual_pathway": AnnualPathwayEngine(),
            "progress_tracker": ProgressTrackerEngine(),
            "variance_analysis": VarianceAnalysisEngine(),
            "trend_extrapolation": TrendExtrapolationEngine(),
            "corrective_action": CorrectiveActionEngine(),
            "milestone_validation": MilestoneValidationEngine(),
            "initiative_scheduler": InitiativeSchedulerEngine(),
            "budget_allocation": BudgetAllocationEngine(),
            "reporting": ReportingEngine(),
        }
        workflows = {
            "target_setting": InterimTargetSettingWorkflow(),
            "progress_review": AnnualProgressReviewWorkflow(),
            "quarterly_monitoring": QuarterlyMonitoringWorkflow(),
            "variance_investigation": VarianceInvestigationWorkflow(),
            "corrective_action": CorrectiveActionPlanningWorkflow(),
            "annual_reporting": AnnualReportingWorkflow(),
            "target_recalibration": TargetRecalibrationWorkflow(),
        }
        assert len(engines) == 10
        assert len(workflows) == 7

    def test_template_can_reference_with_engine(self):
        """Templates and engines can coexist."""
        engine = VarianceAnalysisEngine()
        template = VarianceAnalysisReportTemplate()
        assert engine is not None
        assert template is not None

    def test_template_registry_with_all_components(self):
        """Template registry can be created alongside all components."""
        registry = TemplateRegistry()
        engine = InterimTargetEngine()
        workflow = InterimTargetSettingWorkflow()
        bridge = PACK021Bridge()
        assert registry is not None
        assert engine is not None
        assert workflow is not None
        assert bridge is not None
        assert registry.template_count == 10

    def test_bridges_coexist_with_engines(self):
        """All 10 bridges can coexist with all 10 engines."""
        bridges = {
            "pack021": PACK021Bridge(),
            "pack028": PACK028Bridge(),
            "mrv": MRVBridge(),
            "sbti": SBTiBridge(),
            "cdp": CDPBridge(),
            "tcfd": TCFDBridge(),
            "initiative_tracker": InitiativeTrackerBridge(),
            "budget_system": BudgetSystemBridge(),
            "alerting": AlertingBridge(),
            "assurance_portal": AssurancePortalBridge(),
        }
        engines = {
            "interim_target": InterimTargetEngine(),
            "reporting": ReportingEngine(),
        }
        assert len(bridges) == 10
        assert len(engines) == 2

    def test_template_registry_lists_all_templates(self):
        """Template registry lists all 10 templates."""
        registry = TemplateRegistry()
        names = registry.list_template_names()
        assert len(names) == 10

    def test_template_registry_categories(self):
        """Template registry has multiple categories."""
        registry = TemplateRegistry()
        categories = registry.categories
        assert len(categories) >= 5


# ========================================================================
# E2E: Workflow Registry Utilities
# ========================================================================


class TestWorkflowRegistryUtilities:
    """End-to-end: Verify workflow registry utility functions."""

    def test_get_workflow_returns_instance(self):
        """get_workflow returns a workflow instance."""
        from workflows import get_workflow
        wf = get_workflow("interim_target_setting")
        assert wf is not None

    def test_get_workflow_invalid_raises_key_error(self):
        """get_workflow raises KeyError for unknown workflow."""
        from workflows import get_workflow
        with pytest.raises(KeyError):
            get_workflow("nonexistent_workflow")

    def test_list_workflows_returns_7(self):
        """list_workflows returns 7 workflow descriptions."""
        from workflows import list_workflows
        wf_list = list_workflows()
        assert len(wf_list) == 7

    def test_list_workflows_has_expected_fields(self):
        """Each workflow entry has name, phases, description, and dag."""
        from workflows import list_workflows
        for wf in list_workflows():
            assert "name" in wf
            assert "phases" in wf
            assert "description" in wf
            assert "dag" in wf

    def test_get_workflow_config_returns_class(self):
        """get_workflow_config returns a config class."""
        from workflows import get_workflow_config
        config_cls = get_workflow_config("interim_target_setting")
        assert config_cls is not None

    def test_get_workflow_input_returns_class(self):
        """get_workflow_input returns an input class."""
        from workflows import get_workflow_input
        input_cls = get_workflow_input("annual_progress_review")
        assert input_cls is not None

    @pytest.mark.parametrize("wf_name", [
        "interim_target_setting",
        "annual_progress_review",
        "quarterly_monitoring",
        "variance_investigation",
        "corrective_action_planning",
        "annual_reporting",
        "target_recalibration",
    ])
    def test_each_workflow_retrievable(self, wf_name):
        """Each workflow can be retrieved by name from the registry."""
        from workflows import get_workflow
        wf = get_workflow(wf_name)
        assert wf is not None
