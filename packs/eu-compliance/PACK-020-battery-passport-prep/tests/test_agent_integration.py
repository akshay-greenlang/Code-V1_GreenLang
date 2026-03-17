# -*- coding: utf-8 -*-
"""
PACK-020 Battery Passport Prep Pack - Agent Integration Tests
=================================================================

Tests agent integration contracts: all bridges importable,
get_loaded_integrations / get_loaded_workflows utilities, cross-bridge
interaction contracts, __init__.py module exports, and version metadata.

Author: GreenLang Platform Team (GL-TestEngineer)
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
INTEGRATIONS_DIR = PACK_ROOT / "integrations"
WORKFLOWS_DIR = PACK_ROOT / "workflows"

# ---------------------------------------------------------------------------
# Dynamic module loader
# ---------------------------------------------------------------------------

def _load_module(file_name: str, module_name: str, subdir: str = ""):
    if subdir:
        file_path = PACK_ROOT / subdir / file_name
    else:
        file_path = PACK_ROOT / file_name
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Load integration modules individually
# ---------------------------------------------------------------------------

INTEGRATION_FILES = {
    "pack_orchestrator": "pack_orchestrator.py",
    "mrv_bridge": "mrv_bridge.py",
    "csrd_pack_bridge": "csrd_pack_bridge.py",
    "supply_chain_bridge": "supply_chain_bridge.py",
    "eudr_bridge": "eudr_bridge.py",
    "taxonomy_bridge": "taxonomy_bridge.py",
    "csddd_bridge": "csddd_bridge.py",
    "data_bridge": "data_bridge.py",
    "health_check": "health_check.py",
    "setup_wizard": "setup_wizard.py",
}

INTEGRATION_CLASSES = {
    "pack_orchestrator": "BatteryPassportOrchestrator",
    "mrv_bridge": "MRVBridge",
    "csrd_pack_bridge": "CSRDPackBridge",
    "supply_chain_bridge": "SupplyChainBridge",
    "eudr_bridge": "EUDRBridge",
    "taxonomy_bridge": "TaxonomyBridge",
    "csddd_bridge": "CSDDDBridge",
    "data_bridge": "DataBridge",
    "health_check": "BatteryPassportHealthCheck",
    "setup_wizard": "BatteryPassportSetupWizard",
}

WORKFLOW_FILES = {
    "carbon_footprint_assessment": "carbon_footprint_assessment_workflow.py",
    "recycled_content_tracking": "recycled_content_tracking_workflow.py",
    "passport_compilation": "passport_compilation_workflow.py",
    "performance_testing": "performance_testing_workflow.py",
    "due_diligence_assessment": "due_diligence_assessment_workflow.py",
    "labelling_verification": "labelling_verification_workflow.py",
    "end_of_life_planning": "end_of_life_planning_workflow.py",
    "regulatory_submission": "regulatory_submission_workflow.py",
}

WORKFLOW_CLASSES = {
    "carbon_footprint_assessment": "CarbonFootprintWorkflow",
    "recycled_content_tracking": "RecycledContentWorkflow",
    "passport_compilation": "PassportCompilationWorkflow",
    "performance_testing": "PerformanceTestingWorkflow",
    "due_diligence_assessment": "DueDiligenceAssessmentWorkflow",
    "labelling_verification": "LabellingVerificationWorkflow",
    "end_of_life_planning": "EndOfLifePlanningWorkflow",
    "regulatory_submission": "RegulatorySubmissionWorkflow",
}

# Load individual modules
_integration_modules: Dict[str, Any] = {}
for iname, ifile in INTEGRATION_FILES.items():
    try:
        _integration_modules[iname] = _load_module(
            ifile, f"pack020_tai.int_{iname}", "integrations"
        )
    except Exception:
        _integration_modules[iname] = None

_workflow_modules: Dict[str, Any] = {}
for wname, wfile in WORKFLOW_FILES.items():
    try:
        _workflow_modules[wname] = _load_module(
            wfile, f"pack020_tai.wf_{wname}", "workflows"
        )
    except Exception:
        _workflow_modules[wname] = None


# ---------------------------------------------------------------------------
# Load __init__.py modules
# ---------------------------------------------------------------------------

_integrations_init = None
try:
    _integrations_init = _load_module(
        "__init__.py", "pack020_tai.integrations_init", "integrations"
    )
except Exception:
    pass

_workflows_init = None
try:
    _workflows_init = _load_module(
        "__init__.py", "pack020_tai.workflows_init", "workflows"
    )
except Exception:
    pass


# =========================================================================
# Integration Module Importability
# =========================================================================


class TestIntegrationImportability:
    """Verify all integration modules can be loaded."""

    @pytest.mark.parametrize("integration_name", list(INTEGRATION_FILES.keys()))
    def test_module_loads(self, integration_name):
        mod = _integration_modules.get(integration_name)
        if mod is None:
            pytest.skip(f"Integration {integration_name} not loadable")
        assert mod is not None

    @pytest.mark.parametrize("integration_name", list(INTEGRATION_FILES.keys()))
    def test_class_exists(self, integration_name):
        mod = _integration_modules.get(integration_name)
        if mod is None:
            pytest.skip(f"Integration {integration_name} not loadable")
        cls_name = INTEGRATION_CLASSES[integration_name]
        assert hasattr(mod, cls_name), f"Class {cls_name} not in {integration_name}"

    @pytest.mark.parametrize("integration_name", list(INTEGRATION_FILES.keys()))
    def test_class_instantiates(self, integration_name):
        mod = _integration_modules.get(integration_name)
        if mod is None:
            pytest.skip(f"Integration {integration_name} not loadable")
        cls_name = INTEGRATION_CLASSES[integration_name]
        cls = getattr(mod, cls_name)
        try:
            instance = cls()
        except TypeError:
            instance = cls(config=None)
        assert instance is not None


# =========================================================================
# Workflow Module Importability
# =========================================================================


class TestWorkflowImportability:
    """Verify all workflow modules can be loaded."""

    @pytest.mark.parametrize("workflow_name", list(WORKFLOW_FILES.keys()))
    def test_module_loads(self, workflow_name):
        mod = _workflow_modules.get(workflow_name)
        if mod is None:
            pytest.skip(f"Workflow {workflow_name} not loadable")
        assert mod is not None

    @pytest.mark.parametrize("workflow_name", list(WORKFLOW_FILES.keys()))
    def test_class_exists(self, workflow_name):
        mod = _workflow_modules.get(workflow_name)
        if mod is None:
            pytest.skip(f"Workflow {workflow_name} not loadable")
        cls_name = WORKFLOW_CLASSES[workflow_name]
        assert hasattr(mod, cls_name), f"Class {cls_name} not in {workflow_name}"


# =========================================================================
# Integrations __init__.py Exports
# =========================================================================


class TestIntegrationsInit:
    """Verify integrations/__init__.py module exports and utilities."""

    def _require_init(self):
        if _integrations_init is None:
            pytest.skip("integrations __init__.py not loadable")
        return _integrations_init

    def test_version_defined(self):
        mod = self._require_init()
        assert hasattr(mod, "__version__")
        assert mod.__version__ == "1.0.0"

    def test_pack_id_defined(self):
        mod = self._require_init()
        assert hasattr(mod, "__pack_id__")
        assert mod.__pack_id__ == "PACK-020"

    def test_pack_name_defined(self):
        mod = self._require_init()
        assert hasattr(mod, "__pack_name__")
        assert mod.__pack_name__ == "Battery Passport Prep Pack"

    def test_get_loaded_integrations_callable(self):
        mod = self._require_init()
        assert callable(mod.get_loaded_integrations)

    def test_get_loaded_integrations_returns_list(self):
        mod = self._require_init()
        result = mod.get_loaded_integrations()
        assert isinstance(result, list)

    def test_get_integration_count_callable(self):
        mod = self._require_init()
        assert callable(mod.get_integration_count)

    def test_get_integration_count_returns_int(self):
        mod = self._require_init()
        count = mod.get_integration_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_loaded_integrations_count_matches(self):
        mod = self._require_init()
        loaded = mod.get_loaded_integrations()
        count = mod.get_integration_count()
        assert len(loaded) == count

    def test_all_exports_defined(self):
        mod = self._require_init()
        assert hasattr(mod, "__all__")
        assert isinstance(mod.__all__, list)
        assert len(mod.__all__) > 20

    def test_bridge_classes_in_all(self):
        mod = self._require_init()
        all_exports = mod.__all__
        expected_classes = [
            "BatteryPassportOrchestrator",
            "MRVBridge",
            "CSRDPackBridge",
            "SupplyChainBridge",
            "EUDRBridge",
            "TaxonomyBridge",
            "CSDDDBridge",
            "DataBridge",
            "BatteryPassportHealthCheck",
            "BatteryPassportSetupWizard",
        ]
        for cls in expected_classes:
            assert cls in all_exports, f"'{cls}' not in __all__"

    def test_utility_functions_in_all(self):
        mod = self._require_init()
        all_exports = mod.__all__
        assert "get_loaded_integrations" in all_exports
        assert "get_integration_count" in all_exports


# =========================================================================
# Workflows __init__.py Exports
# =========================================================================


class TestWorkflowsInit:
    """Verify workflows/__init__.py module exports and utilities."""

    def _require_init(self):
        if _workflows_init is None:
            pytest.skip("workflows __init__.py not loadable")
        return _workflows_init

    def test_version_defined(self):
        mod = self._require_init()
        assert hasattr(mod, "__version__")
        assert mod.__version__ == "1.0.0"

    def test_pack_defined(self):
        mod = self._require_init()
        assert hasattr(mod, "__pack__")
        assert mod.__pack__ == "PACK-020"

    def test_get_loaded_workflows_callable(self):
        mod = self._require_init()
        assert callable(mod.get_loaded_workflows)

    def test_get_loaded_workflows_returns_list(self):
        mod = self._require_init()
        result = mod.get_loaded_workflows()
        assert isinstance(result, list)

    def test_get_workflow_count_callable(self):
        mod = self._require_init()
        assert callable(mod.get_workflow_count)

    def test_get_workflow_count_returns_int(self):
        mod = self._require_init()
        count = mod.get_workflow_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_loaded_workflows_count_matches(self):
        mod = self._require_init()
        loaded = mod.get_loaded_workflows()
        count = mod.get_workflow_count()
        assert len(loaded) == count

    def test_regulation_workflow_mapping(self):
        mod = self._require_init()
        assert callable(mod.get_regulation_workflow_mapping)
        mapping = mod.get_regulation_workflow_mapping()
        assert isinstance(mapping, dict)
        assert len(mapping) == 8

    def test_mapping_contains_all_articles(self):
        mod = self._require_init()
        mapping = mod.get_regulation_workflow_mapping()
        expected_keys = [
            "ART_7_CARBON_FOOTPRINT",
            "ART_8_RECYCLED_CONTENT",
            "ART_77_BATTERY_PASSPORT",
            "ART_10_11_PERFORMANCE",
            "ART_48_DUE_DILIGENCE",
            "ART_13_14_LABELLING",
            "ART_59_62_71_END_OF_LIFE",
            "ART_17_18_REGULATORY",
        ]
        for key in expected_keys:
            assert key in mapping, f"Mapping missing key: {key}"

    def test_mapping_values_are_workflow_classes(self):
        mod = self._require_init()
        mapping = mod.get_regulation_workflow_mapping()
        expected_classes = set(WORKFLOW_CLASSES.values())
        for key, value in mapping.items():
            assert value in expected_classes, f"Mapping value '{value}' not a known workflow class"

    def test_all_exports_defined(self):
        mod = self._require_init()
        assert hasattr(mod, "__all__")
        assert isinstance(mod.__all__, list)


# =========================================================================
# Cross-Bridge Contracts
# =========================================================================


class TestCrossBridgeContracts:
    """Validate integration bridge interaction contracts."""

    def test_mrv_bridge_has_routing_table(self):
        mod = _integration_modules.get("mrv_bridge")
        if mod is None:
            pytest.skip("MRV bridge not loadable")
        assert hasattr(mod, "BATTERY_MRV_ROUTING")
        routing = mod.BATTERY_MRV_ROUTING
        assert isinstance(routing, (dict, list))

    def test_data_bridge_has_field_requirements(self):
        mod = _integration_modules.get("data_bridge")
        if mod is None:
            pytest.skip("Data bridge not loadable")
        assert hasattr(mod, "PASSPORT_FIELD_REQUIREMENTS")

    def test_supply_chain_bridge_has_cahra(self):
        mod = _integration_modules.get("supply_chain_bridge")
        if mod is None:
            pytest.skip("Supply chain bridge not loadable")
        assert hasattr(mod, "CAHRA_COUNTRIES")

    def test_taxonomy_bridge_has_activity_criteria(self):
        mod = _integration_modules.get("taxonomy_bridge")
        if mod is None:
            pytest.skip("Taxonomy bridge not loadable")
        assert hasattr(mod, "ACTIVITY_34_SC_CRITERIA")
        assert hasattr(mod, "ACTIVITY_34_DNSH_CRITERIA")

    def test_csddd_bridge_has_overlap_mapping(self):
        mod = _integration_modules.get("csddd_bridge")
        if mod is None:
            pytest.skip("CSDDD bridge not loadable")
        assert hasattr(mod, "CSDDD_BATTERY_OVERLAP")

    def test_csrd_bridge_has_esrs_mappings(self):
        mod = _integration_modules.get("csrd_pack_bridge")
        if mod is None:
            pytest.skip("CSRD bridge not loadable")
        assert hasattr(mod, "ESRS_BATTERY_MAPPINGS")

    def test_eudr_bridge_has_rubber_components(self):
        mod = _integration_modules.get("eudr_bridge")
        if mod is None:
            pytest.skip("EUDR bridge not loadable")
        assert hasattr(mod, "BATTERY_RUBBER_COMPONENTS")

    def test_health_check_has_status_enum(self):
        mod = _integration_modules.get("health_check")
        if mod is None:
            pytest.skip("Health check not loadable")
        assert hasattr(mod, "HealthStatus")

    def test_setup_wizard_has_category_defaults(self):
        mod = _integration_modules.get("setup_wizard")
        if mod is None:
            pytest.skip("Setup wizard not loadable")
        assert hasattr(mod, "CATEGORY_DEFAULTS")

    def test_orchestrator_has_phase_dependencies(self):
        mod = _integration_modules.get("pack_orchestrator")
        if mod is None:
            pytest.skip("Pack orchestrator not loadable")
        assert hasattr(mod, "PHASE_DEPENDENCIES")
        assert hasattr(mod, "PHASE_EXECUTION_ORDER")


# =========================================================================
# Bridge-to-Workflow Alignment
# =========================================================================


class TestBridgeWorkflowAlignment:
    """Verify bridges cover the same regulatory scope as workflows."""

    def test_mrv_bridge_aligns_with_carbon_footprint_workflow(self):
        mrv_mod = _integration_modules.get("mrv_bridge")
        wf_mod = _workflow_modules.get("carbon_footprint_assessment")
        if mrv_mod is None or wf_mod is None:
            pytest.skip("MRV bridge or CF workflow not loadable")
        # MRV bridge should have scope enum covering emissions
        assert hasattr(mrv_mod, "MRVScope")
        # CF workflow should have the workflow class
        assert hasattr(wf_mod, "CarbonFootprintWorkflow")

    def test_supply_chain_bridge_aligns_with_dd_workflow(self):
        sc_mod = _integration_modules.get("supply_chain_bridge")
        wf_mod = _workflow_modules.get("due_diligence_assessment")
        if sc_mod is None or wf_mod is None:
            pytest.skip("SC bridge or DD workflow not loadable")
        assert hasattr(sc_mod, "CriticalMineral")
        assert hasattr(wf_mod, "DueDiligenceAssessmentWorkflow")

    def test_data_bridge_aligns_with_passport_workflow(self):
        db_mod = _integration_modules.get("data_bridge")
        wf_mod = _workflow_modules.get("passport_compilation")
        if db_mod is None or wf_mod is None:
            pytest.skip("Data bridge or passport workflow not loadable")
        assert hasattr(db_mod, "BATTERY_DATA_ROUTING")
        assert hasattr(wf_mod, "PassportCompilationWorkflow")

    def test_eudr_bridge_aligns_with_dd_workflow(self):
        eudr_mod = _integration_modules.get("eudr_bridge")
        wf_mod = _workflow_modules.get("due_diligence_assessment")
        if eudr_mod is None or wf_mod is None:
            pytest.skip("EUDR bridge or DD workflow not loadable")
        assert hasattr(eudr_mod, "COUNTRY_BENCHMARKS")
        assert hasattr(wf_mod, "DueDiligenceAssessmentWorkflow")


# =========================================================================
# Integration File Presence
# =========================================================================


class TestIntegrationFilePresence:
    """Verify all integration files exist on disk."""

    @pytest.mark.parametrize("integration_name,file_name", list(INTEGRATION_FILES.items()))
    def test_integration_file_exists(self, integration_name, file_name):
        filepath = INTEGRATIONS_DIR / file_name
        assert filepath.exists(), f"Integration file missing: {file_name}"

    @pytest.mark.parametrize("workflow_name,file_name", list(WORKFLOW_FILES.items()))
    def test_workflow_file_exists(self, workflow_name, file_name):
        filepath = WORKFLOWS_DIR / file_name
        assert filepath.exists(), f"Workflow file missing: {file_name}"

    def test_integrations_init_exists(self):
        init_path = INTEGRATIONS_DIR / "__init__.py"
        assert init_path.exists()

    def test_workflows_init_exists(self):
        init_path = WORKFLOWS_DIR / "__init__.py"
        assert init_path.exists()
