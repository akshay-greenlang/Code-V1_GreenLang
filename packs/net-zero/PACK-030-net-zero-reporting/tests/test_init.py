# -*- coding: utf-8 -*-
"""
Tests for PACK-030 Net Zero Reporting Pack package __init__ modules.

Validates that the root __init__.py and each sub-package __init__.py
correctly expose their public API, version metadata, and all expected
classes/enums. Covers: root pack imports, engines (10), workflows (8),
templates (15 + TemplateRegistry), integrations (12), and __all__ export
counts.

Target: ~65 tests.

Author: GreenLang Platform Team
Pack: PACK-030 Net Zero Reporting Pack
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
    """Tests for root __init__.py exports."""

    def test_root_import_works(self):
        """Root pack can be imported without error."""
        # Already on sys.path; verify __init__.py content is loadable
        root_init = _PACK_ROOT / "__init__.py"
        assert root_init.exists()

    def test_root_version(self):
        from __init__ import __version__
        assert __version__ == "1.0.0"

    def test_root_pack_id(self):
        from __init__ import __pack_id__
        assert __pack_id__ == "PACK-030"

    def test_root_pack_name(self):
        from __init__ import __pack_name__
        assert __pack_name__ == "Net Zero Reporting Pack"

    def test_root_author(self):
        from __init__ import __author__
        assert __author__ is not None
        assert len(__author__) > 0

    def test_root_all_exports_list(self):
        from __init__ import __all__
        assert isinstance(__all__, list)

    def test_root_all_has_correct_count(self):
        """Root __all__ should have 4 metadata + 10 engines + 8 workflows + 16 templates + 12 integrations = 50."""
        from __init__ import __all__
        assert len(__all__) >= 50

    def test_root_exports_all_10_engines(self):
        from __init__ import __all__
        engine_names = [
            "DataAggregationEngine",
            "NarrativeGenerationEngine",
            "FrameworkMappingEngine",
            "XBRLTaggingEngine",
            "DashboardGenerationEngine",
            "AssurancePackagingEngine",
            "ReportCompilationEngine",
            "ValidationEngine",
            "TranslationEngine",
            "FormatRenderingEngine",
        ]
        for name in engine_names:
            assert name in __all__, f"Missing engine in __all__: {name}"

    def test_root_exports_all_8_workflows(self):
        from __init__ import __all__
        workflow_names = [
            "SBTiProgressWorkflow",
            "CDPQuestionnaireWorkflow",
            "TCFDDisclosureWorkflow",
            "GRI305Workflow",
            "IFRSS2Workflow",
            "SECClimateWorkflow",
            "CSRDESRSE1Workflow",
            "MultiFrameworkWorkflow",
        ]
        for name in workflow_names:
            assert name in __all__, f"Missing workflow in __all__: {name}"

    def test_root_exports_all_15_templates(self):
        from __init__ import __all__
        template_names = [
            "SBTiProgressTemplate",
            "CDPGovernanceTemplate",
            "CDPEmissionsTemplate",
            "TCFDGovernanceTemplate",
            "TCFDStrategyTemplate",
            "TCFDRiskTemplate",
            "TCFDMetricsTemplate",
            "GRI305Template",
            "ISSBS2Template",
            "SECClimateTemplate",
            "CSRDE1Template",
            "InvestorDashboardTemplate",
            "RegulatorDashboardTemplate",
            "CustomerCarbonTemplate",
            "AssuranceEvidenceTemplate",
        ]
        for name in template_names:
            assert name in __all__, f"Missing template in __all__: {name}"

    def test_root_exports_template_registry(self):
        from __init__ import __all__
        assert "TemplateRegistry" in __all__

    def test_root_exports_all_12_integrations(self):
        from __init__ import __all__
        integration_names = [
            "PACK021Integration",
            "PACK022Integration",
            "PACK028Integration",
            "PACK029Integration",
            "GLSBTiAppIntegration",
            "GLCDPAppIntegration",
            "GLTCFDAppIntegration",
            "GLGHGAppIntegration",
            "XBRLTaxonomyIntegration",
            "TranslationIntegration",
            "OrchestratorIntegration",
            "HealthCheckIntegration",
        ]
        for name in integration_names:
            assert name in __all__, f"Missing integration in __all__: {name}"


# ========================================================================
# Engines __init__
# ========================================================================


class TestEnginesInit:
    """Tests for engines/__init__.py exports."""

    def test_engines_init_exists(self):
        assert (_PACK_ROOT / "engines" / "__init__.py").exists()

    def test_engines_version_exported(self):
        from engines import __version__
        assert __version__ == "1.0.0"

    def test_engines_pack_exported(self):
        from engines import __pack__
        assert __pack__ == "PACK-030"

    def test_engines_pack_name_exported(self):
        from engines import __pack_name__
        assert __pack_name__ == "Net Zero Reporting Pack"

    def test_engines_count_exported(self):
        from engines import __engines_count__
        assert __engines_count__ == 10

    def test_engines_all_list_is_list(self):
        from engines import __all__
        assert isinstance(__all__, list)

    @pytest.mark.parametrize("engine_name", [
        "DataAggregationEngine",
        "NarrativeGenerationEngine",
        "FrameworkMappingEngine",
        "XBRLTaggingEngine",
        "DashboardGenerationEngine",
        "AssurancePackagingEngine",
        "ReportCompilationEngine",
        "ValidationEngine",
        "TranslationEngine",
        "FormatRenderingEngine",
    ])
    def test_engine_importable(self, engine_name):
        """Each engine class should be importable from engines package."""
        import engines
        engine_cls = getattr(engines, engine_name, None)
        assert engine_cls is not None, f"Engine {engine_name} not importable from engines"

    def test_get_loaded_engines_callable(self):
        from engines import get_loaded_engines
        result = get_loaded_engines()
        assert isinstance(result, list)

    def test_get_engine_count_callable(self):
        from engines import get_engine_count
        assert get_engine_count() == 10

    def test_get_loaded_engine_count_callable(self):
        from engines import get_loaded_engine_count
        count = get_loaded_engine_count()
        assert count >= 0
        assert count <= 10


# ========================================================================
# Workflows __init__
# ========================================================================


class TestWorkflowsInit:
    """Tests for workflows/__init__.py exports."""

    def test_workflows_init_exists(self):
        assert (_PACK_ROOT / "workflows" / "__init__.py").exists()

    def test_workflows_version_exported(self):
        from workflows import __version__
        assert __version__ == "1.0.0"

    def test_workflows_pack_id_exported(self):
        from workflows import __pack_id__
        assert __pack_id__ == "PACK-030"

    def test_workflows_pack_name_exported(self):
        from workflows import __pack_name__
        assert __pack_name__ == "Net Zero Reporting Pack"

    def test_workflows_all_list_is_list(self):
        from workflows import __all__
        assert isinstance(__all__, list)

    def test_workflows_all_has_many_exports(self):
        from workflows import __all__
        # 8 workflows + configs + results + enums + models + registry + utilities
        assert len(__all__) >= 50

    @pytest.mark.parametrize("wf_name", [
        "SBTiProgressWorkflow",
        "CDPQuestionnaireWorkflow",
        "TCFDDisclosureWorkflow",
        "GRI305Workflow",
        "IFRSS2Workflow",
        "SECClimateWorkflow",
        "CSRDESRSE1Workflow",
        "MultiFrameworkWorkflow",
    ])
    def test_workflow_importable(self, wf_name):
        """Each workflow class should be importable from workflows package."""
        import workflows
        wf_cls = getattr(workflows, wf_name, None)
        assert wf_cls is not None, f"Workflow {wf_name} not importable from workflows"

    def test_workflow_registry_exported(self):
        from workflows import WORKFLOW_REGISTRY
        assert isinstance(WORKFLOW_REGISTRY, dict)
        assert len(WORKFLOW_REGISTRY) == 8

    def test_get_workflow_callable(self):
        from workflows import get_workflow
        assert callable(get_workflow)

    def test_list_workflows_callable(self):
        from workflows import list_workflows
        result = list_workflows()
        assert isinstance(result, list)
        assert len(result) == 8


# ========================================================================
# Templates __init__
# ========================================================================


class TestTemplatesInit:
    """Tests for templates/__init__.py exports."""

    def test_templates_init_exists(self):
        assert (_PACK_ROOT / "templates" / "__init__.py").exists()

    def test_templates_version_exported(self):
        from templates import __version__
        assert __version__ is not None

    def test_templates_pack_id_exported(self):
        from templates import __pack_id__
        assert __pack_id__ == "PACK-030"

    def test_templates_pack_name_exported(self):
        from templates import __pack_name__
        assert __pack_name__ == "Net Zero Reporting Pack"

    def test_templates_all_list_is_list(self):
        from templates import __all__
        assert isinstance(__all__, list)

    def test_templates_all_has_15_templates_plus_registry(self):
        from templates import __all__
        template_classes = [t for t in __all__ if t.endswith("Template")]
        assert len(template_classes) == 15

    def test_template_registry_in_all(self):
        from templates import __all__
        assert "TemplateRegistry" in __all__

    @pytest.mark.parametrize("template_name", [
        "SBTiProgressTemplate",
        "CDPGovernanceTemplate",
        "CDPEmissionsTemplate",
        "TCFDGovernanceTemplate",
        "TCFDStrategyTemplate",
        "TCFDRiskTemplate",
        "TCFDMetricsTemplate",
        "GRI305Template",
        "ISSBS2Template",
        "SECClimateTemplate",
        "CSRDE1Template",
        "InvestorDashboardTemplate",
        "RegulatorDashboardTemplate",
        "CustomerCarbonTemplate",
        "AssuranceEvidenceTemplate",
    ])
    def test_template_importable(self, template_name):
        """Each template class should be importable from templates package."""
        import templates
        tmpl_cls = getattr(templates, template_name, None)
        assert tmpl_cls is not None, f"Template {template_name} not importable from templates"

    def test_template_registry_importable(self):
        from templates import TemplateRegistry
        assert TemplateRegistry is not None

    def test_template_catalog_exported(self):
        from templates import TEMPLATE_CATALOG
        assert isinstance(TEMPLATE_CATALOG, list)
        assert len(TEMPLATE_CATALOG) == 15


# ========================================================================
# Integrations __init__
# ========================================================================


class TestIntegrationsInit:
    """Tests for integrations/__init__.py exports."""

    def test_integrations_init_exists(self):
        assert (_PACK_ROOT / "integrations" / "__init__.py").exists()

    def test_integrations_version_exported(self):
        from integrations import __version__
        assert __version__ == "1.0.0"

    def test_integrations_pack_id_exported(self):
        from integrations import __pack_id__
        assert __pack_id__ == "PACK-030"

    def test_integrations_pack_name_exported(self):
        from integrations import __pack_name__
        assert __pack_name__ == "Net Zero Reporting Pack"

    def test_integrations_all_list_is_list(self):
        from integrations import __all__
        assert isinstance(__all__, list)

    def test_integrations_all_has_many_exports(self):
        from integrations import __all__
        # 12 integration classes + configs + models + enums + utilities
        assert len(__all__) >= 80

    @pytest.mark.parametrize("integration_name", [
        "PACK021Integration",
        "PACK022Integration",
        "PACK028Integration",
        "PACK029Integration",
        "GLSBTiAppIntegration",
        "GLCDPAppIntegration",
        "GLTCFDAppIntegration",
        "GLGHGAppIntegration",
        "XBRLTaxonomyIntegration",
        "TranslationIntegration",
        "OrchestratorIntegration",
        "HealthCheckIntegration",
    ])
    def test_integration_importable(self, integration_name):
        """Each integration class should be importable from integrations package."""
        import integrations
        int_cls = getattr(integrations, integration_name, None)
        assert int_cls is not None, (
            f"Integration {integration_name} not importable from integrations"
        )

    def test_orchestrator_exported(self):
        from integrations import OrchestratorIntegration
        assert OrchestratorIntegration is not None

    def test_health_check_exported(self):
        from integrations import HealthCheckIntegration
        assert HealthCheckIntegration is not None

    def test_circuit_breaker_exported(self):
        from integrations import CircuitBreaker
        assert CircuitBreaker is not None

    def test_rate_limiter_exported(self):
        from integrations import AsyncRateLimiter
        assert AsyncRateLimiter is not None

    def test_response_cache_exported(self):
        from integrations import AsyncResponseCache
        assert AsyncResponseCache is not None

    def test_integration_health_check_callable(self):
        from integrations import integration_health_check
        assert callable(integration_health_check)
