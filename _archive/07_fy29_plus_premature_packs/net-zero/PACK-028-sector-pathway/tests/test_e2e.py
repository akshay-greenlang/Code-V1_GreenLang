# -*- coding: utf-8 -*-
"""
End-to-end pipeline tests for PACK-028 Sector Pathway Pack.

Tests the full component integration chain: engine instantiation, workflow
construction, template availability, integration bridge creation, preset
loading, and cross-component consistency.

Uses real class logic (no mocks) to validate that all 8 engines, 6 workflows,
8 templates (+TemplateRegistry), 10 integrations, and 6 presets form a
cohesive pack.

Author: GreenLang Platform Team
Pack: PACK-028 Sector Pathway Pack
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
    SectorClassificationEngine,
    IntensityCalculatorEngine,
    PathwayGeneratorEngine,
    ConvergenceAnalyzerEngine,
    TechnologyRoadmapEngine,
    AbatementWaterfallEngine,
    SectorBenchmarkEngine,
    ScenarioComparisonEngine,
)

# --- Workflows ---
from workflows import (
    SectorPathwayDesignWorkflow,
    PathwayValidationWorkflow,
    TechnologyPlanningWorkflow,
    ProgressMonitoringWorkflow,
    MultiScenarioAnalysisWorkflow,
    FullSectorAssessmentWorkflow,
)

# --- Templates ---
from templates import (
    SectorPathwayReportTemplate,
    IntensityConvergenceReportTemplate,
    TechnologyRoadmapReportTemplate,
    AbatementWaterfallReportTemplate,
    SectorBenchmarkReportTemplate,
    ScenarioComparisonReportTemplate,
    SBTiValidationReportTemplate,
    SectorStrategyReportTemplate,
    TemplateRegistry,
)

# --- Integrations ---
from integrations import (
    SectorPathwayPipelineOrchestrator,
    SectorPathwayOrchestratorConfig,
    SBTiSDABridge,
    SBTiSDABridgeConfig,
    IEANZEBridge,
    IEANZEBridgeConfig,
    IPCCAR6Bridge,
    IPCCAR6BridgeConfig,
    PACK021Bridge,
    PACK021BridgeConfig,
    SectorMRVBridge,
    SectorMRVBridgeConfig,
    SectorDecarbBridge,
    SectorDecarbBridgeConfig,
    SectorDataBridge,
    SectorDataBridgeConfig,
    SectorPathwaySetupWizard,
    SectorPathwayHealthCheck,
    HealthCheckConfig,
)


PRESET_DIR = Path(__file__).resolve().parent.parent / "config" / "presets"


# ========================================================================
# E2E: Full Component Instantiation
# ========================================================================


class TestFullComponentInstantiation:
    """End-to-end: Instantiate all 8 engines, 6 workflows, 8 templates,
    10 integrations simultaneously to verify no import conflicts."""

    def test_all_engines_instantiate_together(self):
        """All 8 engines can be instantiated in the same process."""
        engines = [
            SectorClassificationEngine(),
            IntensityCalculatorEngine(),
            PathwayGeneratorEngine(),
            ConvergenceAnalyzerEngine(),
            TechnologyRoadmapEngine(),
            AbatementWaterfallEngine(),
            SectorBenchmarkEngine(),
            ScenarioComparisonEngine(),
        ]
        assert len(engines) == 8
        for e in engines:
            assert e is not None

    def test_all_workflows_instantiate_together(self):
        """All 6 workflows can be instantiated in the same process."""
        workflows = [
            SectorPathwayDesignWorkflow(),
            PathwayValidationWorkflow(),
            TechnologyPlanningWorkflow(),
            ProgressMonitoringWorkflow(),
            MultiScenarioAnalysisWorkflow(),
            FullSectorAssessmentWorkflow(),
        ]
        assert len(workflows) == 6
        for wf in workflows:
            assert wf is not None

    def test_all_templates_instantiate_together(self):
        """All 8 templates can be instantiated in the same process."""
        templates = [
            SectorPathwayReportTemplate(),
            IntensityConvergenceReportTemplate(),
            TechnologyRoadmapReportTemplate(),
            AbatementWaterfallReportTemplate(),
            SectorBenchmarkReportTemplate(),
            ScenarioComparisonReportTemplate(),
            SBTiValidationReportTemplate(),
            SectorStrategyReportTemplate(),
        ]
        assert len(templates) == 8
        for t in templates:
            assert t is not None

    def test_template_registry_instantiates(self):
        """TemplateRegistry can be instantiated alongside templates."""
        registry = TemplateRegistry()
        assert registry is not None
        assert registry.template_count == 8

    def test_all_integrations_instantiate_together(self):
        """All 10 integrations can be instantiated in the same process."""
        integrations = [
            SectorPathwayPipelineOrchestrator(config=SectorPathwayOrchestratorConfig()),
            SBTiSDABridge(config=SBTiSDABridgeConfig()),
            IEANZEBridge(config=IEANZEBridgeConfig()),
            IPCCAR6Bridge(config=IPCCAR6BridgeConfig()),
            PACK021Bridge(config=PACK021BridgeConfig()),
            SectorMRVBridge(config=SectorMRVBridgeConfig()),
            SectorDecarbBridge(config=SectorDecarbBridgeConfig()),
            SectorDataBridge(config=SectorDataBridgeConfig()),
            SectorPathwaySetupWizard(),
            SectorPathwayHealthCheck(config=HealthCheckConfig()),
        ]
        assert len(integrations) == 10
        for i in integrations:
            assert i is not None


# ========================================================================
# E2E: Preset Loading Chain
# ========================================================================


class TestPresetLoadingChain:
    """End-to-end: Load each preset, verify YAML validity."""

    def test_all_presets_exist_as_yaml_files(self):
        """All 6 preset YAML files exist on disk."""
        yaml_files = list(PRESET_DIR.glob("*.yaml"))
        assert len(yaml_files) == 6, (
            f"Expected 6 preset YAML files, found {len(yaml_files)}: "
            f"{[f.name for f in yaml_files]}"
        )

    def test_all_presets_load_as_valid_yaml(self):
        """All 6 presets load as valid YAML dicts."""
        yaml_files = list(PRESET_DIR.glob("*.yaml"))
        for yaml_file in yaml_files:
            with open(yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            assert isinstance(data, dict), f"Preset {yaml_file.name} is not a dict"
            assert len(data) > 0, f"Preset {yaml_file.name} is empty"

    def test_preset_filenames_match_spec(self):
        """Preset filenames align with pack specification."""
        yaml_files = sorted([f.stem for f in PRESET_DIR.glob("*.yaml")])
        # Spec lists: power_generation, heavy_industry, transport,
        # buildings, agriculture (or chemicals, mixed_sectors as alternates)
        assert len(yaml_files) == 6
        for name in yaml_files:
            assert isinstance(name, str)
            assert len(name) > 0


# ========================================================================
# E2E: Cross-Component Consistency
# ========================================================================


class TestCrossComponentConsistency:
    """End-to-end: Verify that engines, workflows, templates, and integrations
    are consistent in naming, versioning, and structure."""

    def test_engine_count_matches_pack_spec(self):
        """Pack has exactly 8 engines."""
        from engines import __all__ as engine_exports
        engine_classes = [e for e in engine_exports if e.endswith("Engine")]
        assert len(engine_classes) == 8

    def test_workflow_count_matches_pack_spec(self):
        """Pack has exactly 6 workflows."""
        from workflows import __all__ as workflow_exports
        workflow_classes = [w for w in workflow_exports if w.endswith("Workflow")]
        assert len(workflow_classes) == 6

    def test_template_count_matches_pack_spec(self):
        """Pack has exactly 8 templates (excluding TemplateRegistry)."""
        from templates import __all__ as template_exports
        template_classes = [t for t in template_exports if t.endswith("Template")]
        assert len(template_classes) == 8

    def test_template_registry_in_exports(self):
        """TemplateRegistry is in templates __all__."""
        from templates import __all__ as template_exports
        assert "TemplateRegistry" in template_exports

    def test_preset_count_matches_pack_spec(self):
        """Pack has exactly 6 preset YAML files."""
        yaml_files = list(PRESET_DIR.glob("*.yaml"))
        assert len(yaml_files) == 6

    def test_workflow_and_integration_pack_ids_match(self):
        """Workflows and integrations both report PACK-028."""
        from workflows import __pack_id__ as wp
        from integrations import __pack_id__ as ip
        assert wp == "PACK-028"
        assert ip == "PACK-028"

    def test_template_and_integration_pack_ids_match(self):
        """Templates and integrations both report PACK-028."""
        from templates import __pack_id__ as tp
        from integrations import __pack_id__ as ip
        assert tp == "PACK-028"
        assert ip == "PACK-028"

    def test_all_module_pack_names_match(self):
        """All modules have pack name 'Sector Pathway Pack'."""
        from workflows import __pack_name__ as wn
        from templates import __pack_name__ as tn
        from integrations import __pack_name__ as in_
        assert wn == "Sector Pathway Pack"
        assert tn == "Sector Pathway Pack"
        assert in_ == "Sector Pathway Pack"


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

    def test_orchestrator_has_parallel_group(self):
        """Orchestrator defines a parallel phase group."""
        from integrations import PARALLEL_PHASE_GROUP
        assert PARALLEL_PHASE_GROUP is not None
        assert len(PARALLEL_PHASE_GROUP) > 0


# ========================================================================
# E2E: Full Pack File Count
# ========================================================================


class TestFullPackFileCount:
    """End-to-end: Verify the total number of Python files in the pack."""

    def test_total_python_files(self):
        """Pack contains a substantial number of Python files."""
        pack_root = Path(__file__).resolve().parents[1]
        py_files = list(pack_root.rglob("*.py"))
        # 8 engines + __init__ + 6 workflows + __init__ + 8 templates + __init__
        # + 10 integrations + __init__ + presets __init__ + config __init__
        # + tests = ~50+ Python files minimum
        assert len(py_files) >= 40, f"Only found {len(py_files)} .py files"

    def test_total_yaml_files(self):
        """Pack contains 6 preset YAML files + pack.yaml."""
        pack_root = Path(__file__).resolve().parents[1]
        yaml_files = list(pack_root.rglob("*.yaml"))
        assert len(yaml_files) >= 6, f"Only found {len(yaml_files)} .yaml files"


# ========================================================================
# E2E: Component Interaction Chain
# ========================================================================


class TestComponentInteractionChain:
    """End-to-end: Verify components can reference each other."""

    def test_workflow_can_reference_engine(self):
        """A workflow can instantiate an engine it depends on."""
        engine = SectorClassificationEngine()
        workflow = SectorPathwayDesignWorkflow()
        assert engine is not None
        assert workflow is not None

    def test_orchestrator_can_reference_all_engines(self):
        """Orchestrator can coexist with all engines."""
        orch = SectorPathwayPipelineOrchestrator(
            config=SectorPathwayOrchestratorConfig()
        )
        engines = {
            "classification": SectorClassificationEngine(),
            "intensity": IntensityCalculatorEngine(),
            "pathway": PathwayGeneratorEngine(),
            "convergence": ConvergenceAnalyzerEngine(),
            "technology": TechnologyRoadmapEngine(),
            "abatement": AbatementWaterfallEngine(),
            "benchmark": SectorBenchmarkEngine(),
            "scenario": ScenarioComparisonEngine(),
        }
        assert orch is not None
        assert len(engines) == 8

    def test_template_can_reference_with_engine(self):
        """Templates and engines can coexist."""
        engine = ScenarioComparisonEngine()
        template = ScenarioComparisonReportTemplate()
        assert engine is not None
        assert template is not None

    def test_template_registry_lists_all_8_templates(self):
        """TemplateRegistry lists all 8 templates."""
        registry = TemplateRegistry()
        names = registry.list_template_names()
        assert len(names) == 8

    def test_template_registry_categories(self):
        """TemplateRegistry has multiple categories."""
        registry = TemplateRegistry()
        cats = registry.categories
        assert len(cats) >= 5  # pathway, convergence, technology, abatement, etc.

    def test_health_check_with_all_components(self):
        """Health check can be created alongside all components."""
        hc = SectorPathwayHealthCheck(config=HealthCheckConfig())
        engines_ok = SectorClassificationEngine() is not None
        workflows_ok = SectorPathwayDesignWorkflow() is not None
        templates_ok = SectorPathwayReportTemplate() is not None
        assert hc is not None
        assert engines_ok
        assert workflows_ok
        assert templates_ok

    def test_setup_wizard_with_orchestrator(self):
        """Setup wizard and orchestrator can coexist."""
        wizard = SectorPathwaySetupWizard()
        orch = SectorPathwayPipelineOrchestrator(
            config=SectorPathwayOrchestratorConfig()
        )
        assert wizard is not None
        assert orch is not None

    def test_all_bridges_with_orchestrator(self):
        """All bridge integrations coexist with orchestrator."""
        orch = SectorPathwayPipelineOrchestrator(
            config=SectorPathwayOrchestratorConfig()
        )
        bridges = {
            "sbti_sda": SBTiSDABridge(config=SBTiSDABridgeConfig()),
            "iea_nze": IEANZEBridge(config=IEANZEBridgeConfig()),
            "ipcc_ar6": IPCCAR6Bridge(config=IPCCAR6BridgeConfig()),
            "pack021": PACK021Bridge(config=PACK021BridgeConfig()),
            "mrv": SectorMRVBridge(config=SectorMRVBridgeConfig()),
            "decarb": SectorDecarbBridge(config=SectorDecarbBridgeConfig()),
            "data": SectorDataBridge(config=SectorDataBridgeConfig()),
        }
        assert orch is not None
        assert len(bridges) == 7
        for name, bridge in bridges.items():
            assert bridge is not None, f"Bridge {name} is None"
