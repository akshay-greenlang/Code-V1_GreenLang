# -*- coding: utf-8 -*-
"""
Tests for PACK-028 Sector Pathway Pack package __init__ modules.

Validates that each sub-package (__init__.py) correctly exposes its
public API, version metadata, and all expected classes/enums.

Author: GreenLang Platform Team
Pack: PACK-028 Sector Pathway Pack
"""

import sys
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))


# ========================================================================
# Root __init__
# ========================================================================


class TestRootInit:
    """Tests for PACK-028 root __init__.py exports."""

    def test_version_exported(self):
        """Root __init__ exports __version__."""
        init_path = _PACK_ROOT / "__init__.py"
        assert init_path.exists()
        # Import from the module namespace
        import importlib
        spec = importlib.util.spec_from_file_location("pack028", init_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "__version__")
        assert mod.__version__ == "1.0.0"

    def test_pack_id_exported(self):
        """Root __init__ exports __pack_id__ = PACK-028."""
        import importlib
        spec = importlib.util.spec_from_file_location(
            "pack028_id", _PACK_ROOT / "__init__.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "__pack_id__")
        assert mod.__pack_id__ == "PACK-028"

    def test_pack_name_exported(self):
        """Root __init__ exports __pack_name__."""
        import importlib
        spec = importlib.util.spec_from_file_location(
            "pack028_name", _PACK_ROOT / "__init__.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "__pack_name__")
        assert mod.__pack_name__ == "Sector Pathway Pack"

    def test_all_exports_count(self):
        """Root __init__ __all__ has at least 30 exports."""
        import importlib
        spec = importlib.util.spec_from_file_location(
            "pack028_all", _PACK_ROOT / "__init__.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "__all__")
        assert isinstance(mod.__all__, list)
        # 8 engines + 6 workflows + 8 templates + TemplateRegistry
        # + 10 integrations + metadata = 36+
        assert len(mod.__all__) >= 30

    def test_all_8_engines_in_root_all(self):
        """Root __all__ includes all 8 engine class names."""
        import importlib
        spec = importlib.util.spec_from_file_location(
            "pack028_eng", _PACK_ROOT / "__init__.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        expected_engines = [
            "SectorClassificationEngine",
            "IntensityCalculatorEngine",
            "PathwayGeneratorEngine",
            "ConvergenceAnalyzerEngine",
            "TechnologyRoadmapEngine",
            "AbatementWaterfallEngine",
            "SectorBenchmarkEngine",
            "ScenarioComparisonEngine",
        ]
        for eng in expected_engines:
            assert eng in mod.__all__, f"Engine {eng} not in root __all__"

    def test_all_6_workflows_in_root_all(self):
        """Root __all__ includes all 6 workflow class names."""
        import importlib
        spec = importlib.util.spec_from_file_location(
            "pack028_wf", _PACK_ROOT / "__init__.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        expected_workflows = [
            "SectorPathwayDesignWorkflow",
            "PathwayValidationWorkflow",
            "TechnologyPlanningWorkflow",
            "ProgressMonitoringWorkflow",
            "MultiScenarioAnalysisWorkflow",
            "FullSectorAssessmentWorkflow",
        ]
        for wf in expected_workflows:
            assert wf in mod.__all__, f"Workflow {wf} not in root __all__"

    def test_all_8_templates_in_root_all(self):
        """Root __all__ includes all 8 template class names."""
        import importlib
        spec = importlib.util.spec_from_file_location(
            "pack028_tmpl", _PACK_ROOT / "__init__.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        expected_templates = [
            "SectorPathwayReportTemplate",
            "IntensityConvergenceReportTemplate",
            "TechnologyRoadmapReportTemplate",
            "AbatementWaterfallReportTemplate",
            "SectorBenchmarkReportTemplate",
            "ScenarioComparisonReportTemplate",
            "SBTiValidationReportTemplate",
            "SectorStrategyReportTemplate",
        ]
        for tmpl in expected_templates:
            assert tmpl in mod.__all__, f"Template {tmpl} not in root __all__"

    def test_template_registry_in_root_all(self):
        """Root __all__ includes TemplateRegistry."""
        import importlib
        spec = importlib.util.spec_from_file_location(
            "pack028_reg", _PACK_ROOT / "__init__.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert "TemplateRegistry" in mod.__all__

    def test_all_10_integrations_in_root_all(self):
        """Root __all__ includes all 10 integration class names."""
        import importlib
        spec = importlib.util.spec_from_file_location(
            "pack028_int", _PACK_ROOT / "__init__.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        expected_integrations = [
            "SectorPathwayPipelineOrchestrator",
            "SBTiSDABridge",
            "IEANZEBridge",
            "IPCCAR6Bridge",
            "PACK021Bridge",
            "SectorMRVBridge",
            "SectorDecarbBridge",
            "SectorDataBridge",
            "SectorPathwaySetupWizard",
            "SectorPathwayHealthCheck",
        ]
        for integ in expected_integrations:
            assert integ in mod.__all__, f"Integration {integ} not in root __all__"


# ========================================================================
# Engines __init__
# ========================================================================


class TestEnginesInit:
    """Tests for engines/__init__.py exports."""

    def test_imports_all_8_engines(self):
        """engines __all__ includes 8 engine classes."""
        from engines import __all__
        engine_classes = [e for e in __all__ if e.endswith("Engine")]
        assert len(engine_classes) == 8

    def test_version_exported(self):
        """engines/__init__.py exports __version__."""
        from engines import __version__
        assert __version__ == "1.0.0"

    def test_pack_identifier_exported(self):
        """engines/__init__.py exports a pack identifier."""
        import engines
        # engines uses __pack__ instead of __pack_id__
        has_pack = hasattr(engines, "__pack__") or hasattr(engines, "__pack_id__")
        assert has_pack
        pack_val = getattr(engines, "__pack__", None) or getattr(engines, "__pack_id__", None)
        assert pack_val == "PACK-028"

    def test_pack_name_exported(self):
        """engines/__init__.py exports __pack_name__."""
        from engines import __pack_name__
        assert __pack_name__ == "Sector Pathway Pack"

    def test_all_list_is_list(self):
        """engines __all__ is a list."""
        from engines import __all__
        assert isinstance(__all__, list)

    def test_engine_count_helper(self):
        """engines provides get_engine_count() -> 8."""
        from engines import get_engine_count
        assert get_engine_count() == 8

    def test_loaded_engines_helper(self):
        """engines provides get_loaded_engines() returning engine names."""
        from engines import get_loaded_engines
        loaded = get_loaded_engines()
        assert isinstance(loaded, list)
        assert len(loaded) == 8

    @pytest.mark.parametrize("engine_name", [
        "SectorClassificationEngine",
        "IntensityCalculatorEngine",
        "PathwayGeneratorEngine",
        "ConvergenceAnalyzerEngine",
        "TechnologyRoadmapEngine",
        "AbatementWaterfallEngine",
        "SectorBenchmarkEngine",
        "ScenarioComparisonEngine",
    ])
    def test_engine_importable(self, engine_name):
        """Each engine class is importable from engines package."""
        import engines
        cls = getattr(engines, engine_name, None)
        assert cls is not None, f"{engine_name} not importable from engines"


# ========================================================================
# Workflows __init__
# ========================================================================


class TestWorkflowsInit:
    """Tests for workflows/__init__.py exports."""

    def test_imports_all_6_workflows(self):
        """workflows __all__ includes 6 workflow classes."""
        from workflows import __all__
        workflow_classes = [w for w in __all__ if w.endswith("Workflow")]
        assert len(workflow_classes) == 6

    def test_version_exported(self):
        """workflows/__init__.py exports __version__."""
        from workflows import __version__
        assert __version__ == "1.0.0"

    def test_pack_id_exported(self):
        """workflows/__init__.py exports __pack_id__ = PACK-028."""
        from workflows import __pack_id__
        assert __pack_id__ == "PACK-028"

    def test_pack_name_exported(self):
        """workflows/__init__.py exports __pack_name__."""
        from workflows import __pack_name__
        assert __pack_name__ == "Sector Pathway Pack"

    def test_all_list_has_many_exports(self):
        """workflows __all__ has a large number of exports."""
        from workflows import __all__
        assert isinstance(__all__, list)
        # 6 workflows + configs + results + enums + models + constants
        assert len(__all__) >= 50

    @pytest.mark.parametrize("workflow_name", [
        "SectorPathwayDesignWorkflow",
        "PathwayValidationWorkflow",
        "TechnologyPlanningWorkflow",
        "ProgressMonitoringWorkflow",
        "MultiScenarioAnalysisWorkflow",
        "FullSectorAssessmentWorkflow",
    ])
    def test_workflow_importable(self, workflow_name):
        """Each workflow class is importable from workflows package."""
        import workflows
        cls = getattr(workflows, workflow_name, None)
        assert cls is not None, f"{workflow_name} not importable from workflows"

    def test_sector_pathway_design_exports(self):
        """SectorPathwayDesignWorkflow related exports are available."""
        from workflows import (
            SectorPathwayDesignWorkflow,
            SectorPathwayDesignConfig,
            SectorPathwayDesignInput,
            SectorPathwayDesignResult,
        )
        assert all(
            c is not None
            for c in [
                SectorPathwayDesignWorkflow,
                SectorPathwayDesignConfig,
                SectorPathwayDesignInput,
                SectorPathwayDesignResult,
            ]
        )

    def test_full_sector_assessment_exports(self):
        """FullSectorAssessmentWorkflow related exports are available."""
        from workflows import (
            FullSectorAssessmentWorkflow,
            FullSectorAssessmentConfig,
            FullSectorAssessmentInput,
            FullSectorAssessmentResult,
        )
        assert all(
            c is not None
            for c in [
                FullSectorAssessmentWorkflow,
                FullSectorAssessmentConfig,
                FullSectorAssessmentInput,
                FullSectorAssessmentResult,
            ]
        )


# ========================================================================
# Templates __init__
# ========================================================================


class TestTemplatesInit:
    """Tests for templates/__init__.py exports."""

    def test_imports_all_8_templates(self):
        """templates __all__ includes 8 template classes."""
        from templates import __all__
        template_classes = [t for t in __all__ if t.endswith("Template")]
        assert len(template_classes) == 8

    def test_version_exported(self):
        """templates/__init__.py exports __version__."""
        from templates import __version__
        assert __version__ is not None

    def test_pack_id_exported(self):
        """templates/__init__.py exports __pack_id__ = PACK-028."""
        from templates import __pack_id__
        assert __pack_id__ == "PACK-028"

    def test_pack_name_exported(self):
        """templates/__init__.py exports __pack_name__."""
        from templates import __pack_name__
        assert __pack_name__ == "Sector Pathway Pack"

    def test_template_registry_exported(self):
        """templates/__init__.py exports TemplateRegistry."""
        from templates import TemplateRegistry
        assert TemplateRegistry is not None

    def test_template_catalog_exported(self):
        """templates/__init__.py exports TEMPLATE_CATALOG."""
        from templates import TEMPLATE_CATALOG
        assert isinstance(TEMPLATE_CATALOG, list)
        assert len(TEMPLATE_CATALOG) == 8

    @pytest.mark.parametrize("template_name", [
        "SectorPathwayReportTemplate",
        "IntensityConvergenceReportTemplate",
        "TechnologyRoadmapReportTemplate",
        "AbatementWaterfallReportTemplate",
        "SectorBenchmarkReportTemplate",
        "ScenarioComparisonReportTemplate",
        "SBTiValidationReportTemplate",
        "SectorStrategyReportTemplate",
    ])
    def test_template_importable(self, template_name):
        """Each template class is importable from templates package."""
        import templates
        cls = getattr(templates, template_name, None)
        assert cls is not None, f"{template_name} not importable from templates"


# ========================================================================
# Integrations __init__
# ========================================================================


class TestIntegrationsInit:
    """Tests for integrations/__init__.py exports."""

    def test_version_exported(self):
        """integrations/__init__.py exports __version__."""
        from integrations import __version__
        assert __version__ == "1.0.0"

    def test_pack_id_exported(self):
        """integrations/__init__.py exports __pack_id__ = PACK-028."""
        from integrations import __pack_id__
        assert __pack_id__ == "PACK-028"

    def test_all_list_has_many_exports(self):
        """integrations __all__ has a large number of exports."""
        from integrations import __all__
        assert isinstance(__all__, list)
        # 10 integration classes + configs + models + enums + constants
        assert len(__all__) >= 80

    def test_orchestrator_exported(self):
        """SectorPathwayPipelineOrchestrator is exported."""
        from integrations import SectorPathwayPipelineOrchestrator
        assert SectorPathwayPipelineOrchestrator is not None

    def test_setup_wizard_exported(self):
        """SectorPathwaySetupWizard is exported."""
        from integrations import SectorPathwaySetupWizard
        assert SectorPathwaySetupWizard is not None

    def test_health_check_exported(self):
        """SectorPathwayHealthCheck is exported."""
        from integrations import SectorPathwayHealthCheck
        assert SectorPathwayHealthCheck is not None

    @pytest.mark.parametrize("bridge_name", [
        "SBTiSDABridge",
        "IEANZEBridge",
        "IPCCAR6Bridge",
        "PACK021Bridge",
        "SectorMRVBridge",
        "SectorDecarbBridge",
        "SectorDataBridge",
    ])
    def test_bridge_importable(self, bridge_name):
        """Each bridge class is importable from integrations package."""
        import integrations
        cls = getattr(integrations, bridge_name, None)
        assert cls is not None, f"{bridge_name} not importable from integrations"

    def test_phase_constants_exported(self):
        """Phase-related constants are exported."""
        from integrations import (
            PHASE_DEPENDENCIES,
            PHASE_EXECUTION_ORDER,
            PARALLEL_PHASE_GROUP,
        )
        assert PHASE_DEPENDENCIES is not None
        assert PHASE_EXECUTION_ORDER is not None
        assert PARALLEL_PHASE_GROUP is not None

    def test_sector_mapping_constants_exported(self):
        """Sector mapping constants are exported."""
        from integrations import (
            SECTOR_NACE_MAPPING,
            SECTOR_ROUTING_GROUPS,
        )
        assert isinstance(SECTOR_NACE_MAPPING, dict)
        assert isinstance(SECTOR_ROUTING_GROUPS, dict)


# ========================================================================
# Presets __init__
# ========================================================================


class TestPresetsInit:
    """Tests for config/presets/__init__.py."""

    def test_presets_init_exists(self):
        """config/presets/__init__.py exists."""
        init_path = _PACK_ROOT / "config" / "presets" / "__init__.py"
        assert init_path.exists()

    def test_presets_init_importable(self):
        """config/presets module is importable."""
        from config import presets
        assert presets is not None

    def test_preset_yaml_files_on_disk(self):
        """All 6 preset YAML files exist on disk."""
        preset_dir = _PACK_ROOT / "config" / "presets"
        yaml_files = list(preset_dir.glob("*.yaml"))
        assert len(yaml_files) == 6


# ========================================================================
# Cross-Module Consistency
# ========================================================================


class TestCrossModuleConsistency:
    """Verify consistency across all __init__.py modules."""

    def test_all_modules_report_pack_028(self):
        """All modules with __pack_id__ report PACK-028."""
        from workflows import __pack_id__ as wpi
        from templates import __pack_id__ as tpi
        from integrations import __pack_id__ as ipi
        assert wpi == "PACK-028"
        assert tpi == "PACK-028"
        assert ipi == "PACK-028"

    def test_engines_pack_matches(self):
        """Engines module pack identifier is PACK-028."""
        import engines
        pack_val = getattr(engines, "__pack__", None) or getattr(engines, "__pack_id__", None)
        assert pack_val == "PACK-028"

    def test_all_modules_report_sector_pathway_pack(self):
        """All modules with __pack_name__ report 'Sector Pathway Pack'."""
        from workflows import __pack_name__ as wpn
        from templates import __pack_name__ as tpn
        from integrations import __pack_name__ as ipn
        import engines
        epn = getattr(engines, "__pack_name__", None)
        assert wpn == "Sector Pathway Pack"
        assert tpn == "Sector Pathway Pack"
        assert ipn == "Sector Pathway Pack"
        if epn is not None:
            assert epn == "Sector Pathway Pack"
