# -*- coding: utf-8 -*-
"""
Tests for PACK-029 Interim Targets Pack package __init__ modules.

Validates that each sub-package (__init__.py) correctly exposes its
public API, version metadata, and all expected classes/enums.

Author: GreenLang Platform Team
Pack: PACK-029 Interim Targets Pack
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
    """Tests for the root PACK-029 __init__.py exports."""

    def test_root_package_imports(self):
        """Root package can be imported via sys.path."""
        root_init = _PACK_ROOT / "__init__.py"
        assert root_init.exists()

    def test_root_version(self):
        """Root __init__ exports __version__."""
        # Read and exec to avoid relative import issues
        init_path = _PACK_ROOT / "__init__.py"
        content = init_path.read_text(encoding="utf-8")
        assert '__version__ = "1.0.0"' in content

    def test_root_pack_id(self):
        """Root __init__ exports __pack_id__."""
        init_path = _PACK_ROOT / "__init__.py"
        content = init_path.read_text(encoding="utf-8")
        assert '__pack_id__ = "PACK-029"' in content

    def test_root_pack_name(self):
        """Root __init__ exports __pack_name__."""
        init_path = _PACK_ROOT / "__init__.py"
        content = init_path.read_text(encoding="utf-8")
        assert '__pack_name__ = "Interim Targets Pack"' in content

    def test_root_all_exports_engines(self):
        """Root __all__ includes all 10 engine class names."""
        init_path = _PACK_ROOT / "__init__.py"
        content = init_path.read_text(encoding="utf-8")
        engine_names = [
            "InterimTargetEngine",
            "AnnualPathwayEngine",
            "ProgressTrackerEngine",
            "VarianceAnalysisEngine",
            "TrendExtrapolationEngine",
            "CorrectiveActionEngine",
            "MilestoneValidationEngine",
            "InitiativeSchedulerEngine",
            "BudgetAllocationEngine",
            "ReportingEngine",
        ]
        for name in engine_names:
            assert name in content, f"Root __init__ missing engine: {name}"

    def test_root_all_exports_workflows(self):
        """Root __all__ includes all 7 workflow class names."""
        init_path = _PACK_ROOT / "__init__.py"
        content = init_path.read_text(encoding="utf-8")
        workflow_names = [
            "InterimTargetSettingWorkflow",
            "AnnualProgressReviewWorkflow",
            "QuarterlyMonitoringWorkflow",
            "VarianceInvestigationWorkflow",
            "CorrectiveActionPlanningWorkflow",
            "AnnualReportingWorkflow",
            "TargetRecalibrationWorkflow",
        ]
        for name in workflow_names:
            assert name in content, f"Root __init__ missing workflow: {name}"

    def test_root_all_exports_templates(self):
        """Root __all__ includes all 10 template class names."""
        init_path = _PACK_ROOT / "__init__.py"
        content = init_path.read_text(encoding="utf-8")
        template_names = [
            "InterimTargetsSummaryTemplate",
            "AnnualProgressReportTemplate",
            "VarianceAnalysisReportTemplate",
            "CorrectiveActionPlanTemplate",
            "QuarterlyDashboardTemplate",
            "CDPDisclosureTemplate",
            "TCFDMetricsReportTemplate",
            "AssuranceEvidencePackageTemplate",
            "ExecutiveSummaryTemplate",
            "PublicDisclosureTemplate",
            "TemplateRegistry",
        ]
        for name in template_names:
            assert name in content, f"Root __init__ missing template: {name}"

    def test_root_all_exports_integrations(self):
        """Root __all__ includes all 10 integration bridge names."""
        init_path = _PACK_ROOT / "__init__.py"
        content = init_path.read_text(encoding="utf-8")
        integration_names = [
            "PACK021Bridge",
            "PACK028Bridge",
            "MRVBridge",
            "SBTiBridge",
            "CDPBridge",
            "TCFDBridge",
            "InitiativeTrackerBridge",
            "BudgetSystemBridge",
            "AlertingBridge",
            "AssurancePortalBridge",
        ]
        for name in integration_names:
            assert name in content, f"Root __init__ missing integration: {name}"

    def test_root_all_export_count(self):
        """Root __all__ has at least 38 entries (10+7+11+10)."""
        init_path = _PACK_ROOT / "__init__.py"
        content = init_path.read_text(encoding="utf-8")
        # Count items in __all__ list
        in_all = False
        count = 0
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("__all__"):
                in_all = True
                continue
            if in_all:
                if stripped.startswith("]"):
                    break
                if stripped.startswith('"') or stripped.startswith("'"):
                    count += 1
        assert count >= 38, f"Root __all__ has only {count} entries, expected >= 38"


# ========================================================================
# Engines __init__
# ========================================================================


class TestEnginesInit:
    """Tests for engines/__init__.py exports."""

    def test_imports_all_10_engines(self):
        from engines import __all__
        engine_classes = [e for e in __all__ if e.endswith("Engine")]
        assert len(engine_classes) == 10

    def test_version_exported(self):
        from engines import __version__
        assert __version__ is not None
        assert isinstance(__version__, str)

    def test_pack_attribute_exported(self):
        """Engines module has a pack identifier attribute."""
        import engines
        has_pack = (
            hasattr(engines, "__pack__")
            or hasattr(engines, "__pack_id__")
        )
        assert has_pack

    def test_pack_name_exported(self):
        from engines import __pack_name__
        assert __pack_name__ == "Interim Targets Pack"

    def test_engines_count_attribute(self):
        from engines import __engines_count__
        assert __engines_count__ == 10

    def test_get_loaded_engines(self):
        from engines import get_loaded_engines
        loaded = get_loaded_engines()
        assert isinstance(loaded, list)
        assert len(loaded) == 10

    def test_get_engine_count(self):
        from engines import get_engine_count
        assert get_engine_count() == 10

    def test_get_loaded_engine_count(self):
        from engines import get_loaded_engine_count
        assert get_loaded_engine_count() == 10

    @pytest.mark.parametrize("engine_name", [
        "InterimTargetEngine",
        "AnnualPathwayEngine",
        "ProgressTrackerEngine",
        "VarianceAnalysisEngine",
        "TrendExtrapolationEngine",
        "CorrectiveActionEngine",
        "MilestoneValidationEngine",
        "InitiativeSchedulerEngine",
        "BudgetAllocationEngine",
        "ReportingEngine",
    ])
    def test_engine_importable(self, engine_name):
        """Each engine class is importable from engines module."""
        import engines
        cls = getattr(engines, engine_name, None)
        assert cls is not None, f"Engine {engine_name} not importable"

    @pytest.mark.parametrize("engine_name", [
        "InterimTargetEngine",
        "AnnualPathwayEngine",
        "ProgressTrackerEngine",
        "VarianceAnalysisEngine",
        "TrendExtrapolationEngine",
        "CorrectiveActionEngine",
        "MilestoneValidationEngine",
        "InitiativeSchedulerEngine",
        "BudgetAllocationEngine",
        "ReportingEngine",
    ])
    def test_engine_instantiable(self, engine_name):
        """Each engine class can be instantiated."""
        import engines
        cls = getattr(engines, engine_name)
        instance = cls()
        assert instance is not None


# ========================================================================
# Workflows __init__
# ========================================================================


class TestWorkflowsInit:
    """Tests for workflows/__init__.py exports."""

    def test_imports_all_7_workflows(self):
        from workflows import __all__
        workflow_classes = [w for w in __all__ if w.endswith("Workflow")]
        assert len(workflow_classes) == 7

    def test_version_exported(self):
        from workflows import __version__
        assert __version__ == "1.0.0"

    def test_pack_id_exported(self):
        from workflows import __pack_id__
        assert __pack_id__ == "PACK-029"

    def test_pack_name_exported(self):
        from workflows import __pack_name__
        assert __pack_name__ == "Interim Targets Pack"

    def test_all_list_has_many_exports(self):
        from workflows import __all__
        assert isinstance(__all__, list)
        # 7 workflows + configs + results + enums + models + registry/utils
        assert len(__all__) >= 50

    def test_workflow_registry_exported(self):
        from workflows import WORKFLOW_REGISTRY
        assert isinstance(WORKFLOW_REGISTRY, dict)
        assert len(WORKFLOW_REGISTRY) == 7

    def test_get_workflow_exported(self):
        from workflows import get_workflow
        assert callable(get_workflow)

    def test_list_workflows_exported(self):
        from workflows import list_workflows
        assert callable(list_workflows)

    def test_run_workflow_exported(self):
        from workflows import run_workflow
        assert callable(run_workflow)

    @pytest.mark.parametrize("workflow_name", [
        "InterimTargetSettingWorkflow",
        "AnnualProgressReviewWorkflow",
        "QuarterlyMonitoringWorkflow",
        "VarianceInvestigationWorkflow",
        "CorrectiveActionPlanningWorkflow",
        "AnnualReportingWorkflow",
        "TargetRecalibrationWorkflow",
    ])
    def test_workflow_importable(self, workflow_name):
        """Each workflow class is importable from workflows module."""
        import workflows
        cls = getattr(workflows, workflow_name, None)
        assert cls is not None, f"Workflow {workflow_name} not importable"

    @pytest.mark.parametrize("workflow_name", [
        "InterimTargetSettingWorkflow",
        "AnnualProgressReviewWorkflow",
        "QuarterlyMonitoringWorkflow",
        "VarianceInvestigationWorkflow",
        "CorrectiveActionPlanningWorkflow",
        "AnnualReportingWorkflow",
        "TargetRecalibrationWorkflow",
    ])
    def test_workflow_instantiable(self, workflow_name):
        """Each workflow class can be instantiated."""
        import workflows
        cls = getattr(workflows, workflow_name)
        instance = cls()
        assert instance is not None


# ========================================================================
# Templates __init__
# ========================================================================


class TestTemplatesInit:
    """Tests for templates/__init__.py exports."""

    def test_imports_all_10_templates(self):
        from templates import __all__
        template_classes = [t for t in __all__ if t.endswith("Template")]
        assert len(template_classes) == 10

    def test_version_exported(self):
        from templates import __version__
        assert __version__ is not None

    def test_pack_id_exported(self):
        from templates import __pack_id__
        assert __pack_id__ == "PACK-029"

    def test_pack_name_exported(self):
        from templates import __pack_name__
        assert __pack_name__ == "Interim Targets Pack"

    def test_template_registry_exported(self):
        from templates import TemplateRegistry
        assert TemplateRegistry is not None

    def test_template_catalog_exported(self):
        from templates import TEMPLATE_CATALOG
        assert isinstance(TEMPLATE_CATALOG, list)
        assert len(TEMPLATE_CATALOG) == 10

    @pytest.mark.parametrize("template_name", [
        "InterimTargetsSummaryTemplate",
        "AnnualProgressReportTemplate",
        "VarianceAnalysisReportTemplate",
        "CorrectiveActionPlanTemplate",
        "QuarterlyDashboardTemplate",
        "CDPDisclosureTemplate",
        "TCFDMetricsReportTemplate",
        "AssuranceEvidencePackageTemplate",
        "ExecutiveSummaryTemplate",
        "PublicDisclosureTemplate",
    ])
    def test_template_importable(self, template_name):
        """Each template class is importable from templates module."""
        import templates
        cls = getattr(templates, template_name, None)
        assert cls is not None, f"Template {template_name} not importable"

    @pytest.mark.parametrize("template_name", [
        "InterimTargetsSummaryTemplate",
        "AnnualProgressReportTemplate",
        "VarianceAnalysisReportTemplate",
        "CorrectiveActionPlanTemplate",
        "QuarterlyDashboardTemplate",
        "CDPDisclosureTemplate",
        "TCFDMetricsReportTemplate",
        "AssuranceEvidencePackageTemplate",
        "ExecutiveSummaryTemplate",
        "PublicDisclosureTemplate",
    ])
    def test_template_instantiable(self, template_name):
        """Each template class can be instantiated."""
        import templates
        cls = getattr(templates, template_name)
        instance = cls()
        assert instance is not None


# ========================================================================
# Integrations __init__
# ========================================================================


class TestIntegrationsInit:
    """Tests for integrations/__init__.py exports."""

    def test_version_exported(self):
        from integrations import __version__
        assert __version__ == "1.0.0"

    def test_pack_id_exported(self):
        from integrations import __pack_id__
        assert __pack_id__ == "PACK-029"

    def test_pack_name_exported(self):
        from integrations import __pack_name__
        assert __pack_name__ == "Interim Targets Pack"

    def test_all_list_has_many_exports(self):
        from integrations import __all__
        assert isinstance(__all__, list)
        # 10 bridge classes + 10 config classes + many models + enums + utilities
        assert len(__all__) >= 80

    @pytest.mark.parametrize("bridge_name", [
        "PACK021Bridge",
        "PACK028Bridge",
        "MRVBridge",
        "SBTiBridge",
        "CDPBridge",
        "TCFDBridge",
        "InitiativeTrackerBridge",
        "BudgetSystemBridge",
        "AlertingBridge",
        "AssurancePortalBridge",
    ])
    def test_bridge_importable(self, bridge_name):
        """Each bridge class is importable from integrations module."""
        import integrations
        cls = getattr(integrations, bridge_name, None)
        assert cls is not None, f"Bridge {bridge_name} not importable"

    @pytest.mark.parametrize("bridge_name", [
        "PACK021Bridge",
        "PACK028Bridge",
        "MRVBridge",
        "SBTiBridge",
        "CDPBridge",
        "TCFDBridge",
        "InitiativeTrackerBridge",
        "BudgetSystemBridge",
        "AlertingBridge",
        "AssurancePortalBridge",
    ])
    def test_bridge_instantiable(self, bridge_name):
        """Each bridge class can be instantiated."""
        import integrations
        cls = getattr(integrations, bridge_name)
        instance = cls()
        assert instance is not None

    def test_circuit_breaker_exported(self):
        from integrations import CircuitBreaker
        assert CircuitBreaker is not None

    def test_rate_limiter_exported(self):
        from integrations import AsyncRateLimiter
        assert AsyncRateLimiter is not None

    def test_response_cache_exported(self):
        from integrations import AsyncResponseCache
        assert AsyncResponseCache is not None

    def test_api_key_rotator_exported(self):
        from integrations import APIKeyRotator
        assert APIKeyRotator is not None

    def test_health_check_exported(self):
        from integrations import integration_health_check
        assert callable(integration_health_check)

    def test_retry_async_exported(self):
        from integrations import retry_async
        assert callable(retry_async)

    def test_timeout_async_exported(self):
        from integrations import timeout_async
        assert callable(timeout_async)


# ========================================================================
# Config / Presets __init__
# ========================================================================


class TestConfigPresetsInit:
    """Tests for config/presets/__init__.py."""

    def test_presets_init_exists(self):
        presets_init = _PACK_ROOT / "config" / "presets" / "__init__.py"
        assert presets_init.exists()

    def test_presets_yaml_count(self):
        presets_dir = _PACK_ROOT / "config" / "presets"
        yaml_files = list(presets_dir.glob("*.yaml"))
        assert len(yaml_files) == 7

    @pytest.mark.parametrize("preset_file", [
        "sbti_1_5c_pathway.yaml",
        "sbti_wb2c_pathway.yaml",
        "annual_review.yaml",
        "corrective_action.yaml",
        "quarterly_monitoring.yaml",
        "scope_3_extended.yaml",
        "sector_specific.yaml",
    ])
    def test_preset_file_exists(self, preset_file):
        """Each expected preset YAML file exists."""
        presets_dir = _PACK_ROOT / "config" / "presets"
        assert (presets_dir / preset_file).exists(), f"Missing preset: {preset_file}"

    @pytest.mark.parametrize("preset_file", [
        "sbti_1_5c_pathway.yaml",
        "sbti_wb2c_pathway.yaml",
        "annual_review.yaml",
        "corrective_action.yaml",
        "quarterly_monitoring.yaml",
        "scope_3_extended.yaml",
        "sector_specific.yaml",
    ])
    def test_preset_file_valid_yaml(self, preset_file):
        """Each preset YAML file is valid YAML."""
        import yaml
        presets_dir = _PACK_ROOT / "config" / "presets"
        with open(presets_dir / preset_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), f"Preset {preset_file} is not a dict"
        assert len(data) > 0, f"Preset {preset_file} is empty"
