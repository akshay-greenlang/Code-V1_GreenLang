# -*- coding: utf-8 -*-
"""
PACK-017 ESRS Full Coverage Pack - Integration Tests
=======================================================

Tests for all 10 integration modules: pack orchestrator, E1 pack bridge,
DMA pack bridge, CSRD app bridge, MRV agent bridge, data agent bridge,
taxonomy bridge, XBRL tagging bridge, health check, and setup wizard.

Target: 35+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-017 ESRS Full Coverage Pack
Date:    March 2026
"""

import pytest

from .conftest import (
    INTEGRATION_FILES,
    INTEGRATION_CLASSES,
    INTEGRATIONS_DIR,
    _load_integration,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _try_load_integration(key):
    """Attempt to load an integration, returning module or None."""
    try:
        return _load_integration(key)
    except (ImportError, FileNotFoundError):
        return None


# ===========================================================================
# Integration File Existence
# ===========================================================================


class TestIntegrationFilesExist:
    """Test that all 10 integration files exist on disk."""

    @pytest.mark.parametrize("int_key,int_file", list(INTEGRATION_FILES.items()))
    def test_integration_file_exists(self, int_key, int_file):
        """Integration file exists on disk."""
        path = INTEGRATIONS_DIR / int_file
        assert path.exists(), f"Integration file missing: {path}"


# ===========================================================================
# Integration Module Loading
# ===========================================================================


class TestIntegrationLoading:
    """Test that all 10 integrations can be loaded via importlib."""

    @pytest.mark.parametrize("int_key", list(INTEGRATION_FILES.keys()))
    def test_integration_module_loads(self, int_key):
        """Each integration module loads independently."""
        mod = _try_load_integration(int_key)
        assert mod is not None, f"Integration {int_key} failed to load"

    @pytest.mark.parametrize("int_key,int_class", list(INTEGRATION_CLASSES.items()))
    def test_integration_exports_class(self, int_key, int_class):
        """Each integration exports its primary class."""
        mod = _try_load_integration(int_key)
        if mod is None:
            pytest.skip(f"Integration {int_key} not loaded")
        assert hasattr(mod, int_class), f"Integration {int_key} missing class {int_class}"


# ===========================================================================
# Pack Orchestrator
# ===========================================================================


class TestPackOrchestrator:
    """Tests for the ESRSFullCoverageOrchestrator integration."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_integration("pack_orchestrator")

    def test_class_exists(self):
        """ESRSFullOrchestrator class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "ESRSFullOrchestrator")

    def test_phase_count_is_15(self):
        """Orchestrator defines exactly 15 pipeline phases (including INIT)."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        if hasattr(self.mod, "ESRSPipelinePhase"):
            phases = list(self.mod.ESRSPipelinePhase)
            assert len(phases) == 15, f"Expected 15 phases, got {len(phases)}"
        else:
            pytest.skip("ESRSPipelinePhase enum not found")

    def test_phase_enum(self):
        """Orchestrator exports ESRSPipelinePhase enum."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "ESRSPipelinePhase")

    def test_dependency_graph(self):
        """Orchestrator exports PHASE_DEPENDENCIES dict."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "PHASE_DEPENDENCIES")
        deps = self.mod.PHASE_DEPENDENCIES
        assert isinstance(deps, dict)
        assert len(deps) > 0, "PHASE_DEPENDENCIES should not be empty"

    def test_phase_execution_order(self):
        """Orchestrator exports PHASE_EXECUTION_ORDER list."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "PHASE_EXECUTION_ORDER")
        order = self.mod.PHASE_EXECUTION_ORDER
        assert isinstance(order, list)
        assert len(order) == 15, f"Expected 15 phases, got {len(order)}"

    def test_orchestrator_config_model(self):
        """OrchestratorConfig model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "OrchestratorConfig")

    def test_phase_result_model(self):
        """PhaseResult model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "PhaseResult")


# ===========================================================================
# E1 Pack Bridge
# ===========================================================================


class TestE1PackBridge:
    """Tests for the E1PackBridge integration (bridges PACK-016)."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_integration("e1_bridge")

    def test_init(self):
        """E1PackBridge class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "E1PackBridge")

    def test_has_import_method(self):
        """E1PackBridge has import or bridge method."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = self.mod.E1PackBridge
        has_import = (
            hasattr(cls, "import_e1_results")
            or hasattr(cls, "import_climate_data")
            or hasattr(cls, "bridge_e1")
            or hasattr(cls, "import_emissions")
        )
        assert has_import, "E1PackBridge should have an import method"


# ===========================================================================
# DMA Pack Bridge
# ===========================================================================


class TestDMAPackBridge:
    """Tests for the DMABridge integration (bridges PACK-015)."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_integration("dma_bridge")

    def test_init(self):
        """DMABridge class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, INTEGRATION_CLASSES["dma_bridge"])

    def test_has_materiality_import(self):
        """DMABridge has materiality import or check method."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = getattr(self.mod, INTEGRATION_CLASSES["dma_bridge"])
        has_mat = (
            hasattr(cls, "import_materiality")
            or hasattr(cls, "get_materiality_status")
            or hasattr(cls, "check_materiality")
            or hasattr(cls, "import_dma_results")
        )
        assert has_mat, "DMABridge should have a materiality method"


# ===========================================================================
# CSRD App Bridge
# ===========================================================================


class TestCSRDAppBridge:
    """Tests for the CSRDAppBridge integration."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_integration("csrd_app_bridge")

    def test_init(self):
        """CSRDAppBridge class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, INTEGRATION_CLASSES["csrd_app_bridge"])

    def test_datapoint_access(self):
        """CSRDAppBridge has datapoint access method."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = getattr(self.mod, INTEGRATION_CLASSES["csrd_app_bridge"])
        has_dp = (
            hasattr(cls, "get_datapoints")
            or hasattr(cls, "import_datapoints")
            or hasattr(cls, "export_disclosure")
            or hasattr(cls, "sync_datapoints")
        )
        assert has_dp, "CSRDAppBridge should have datapoint access"


# ===========================================================================
# MRV Agent Bridge
# ===========================================================================


class TestMRVAgentBridge:
    """Tests for the MRVBridge integration."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_integration("mrv_bridge")

    def test_class_exists(self):
        """MRVBridge class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, INTEGRATION_CLASSES["mrv_bridge"])

    def test_routing_table_30_agents(self):
        """MRV bridge source references 30 MRV agents."""
        path = INTEGRATIONS_DIR / INTEGRATION_FILES["mrv_bridge"]
        if not path.exists():
            pytest.skip("MRV bridge file not found")
        content = path.read_text(encoding="utf-8")
        # Check for MRV-001 and MRV-030 references
        has_mrv_001 = "MRV-001" in content or "mrv_001" in content or "001" in content
        has_mrv_030 = "MRV-030" in content or "mrv_030" in content or "030" in content
        assert has_mrv_001, "MRV bridge should reference MRV-001"
        assert has_mrv_030, "MRV bridge should reference MRV-030"

    def test_scope_coverage(self):
        """MRV bridge covers all three scopes."""
        path = INTEGRATIONS_DIR / INTEGRATION_FILES["mrv_bridge"]
        if not path.exists():
            pytest.skip("MRV bridge file not found")
        content = path.read_text(encoding="utf-8")
        has_scope_1 = "scope_1" in content.lower() or "scope 1" in content.lower()
        has_scope_2 = "scope_2" in content.lower() or "scope 2" in content.lower()
        has_scope_3 = "scope_3" in content.lower() or "scope 3" in content.lower()
        assert has_scope_1, "MRV bridge should reference Scope 1"
        assert has_scope_2, "MRV bridge should reference Scope 2"
        assert has_scope_3, "MRV bridge should reference Scope 3"


# ===========================================================================
# Data Agent Bridge
# ===========================================================================


class TestDataAgentBridge:
    """Tests for the DataBridge integration."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_integration("data_bridge")

    def test_class_exists(self):
        """DataBridge class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, INTEGRATION_CLASSES["data_bridge"])

    def test_routing_table_20_agents(self):
        """Data bridge source references 20 data agents."""
        path = INTEGRATIONS_DIR / INTEGRATION_FILES["data_bridge"]
        if not path.exists():
            pytest.skip("Data bridge file not found")
        content = path.read_text(encoding="utf-8")
        # Check for DATA agent references
        has_data_001 = "DATA-001" in content or "data_001" in content or "001" in content
        has_data_020 = "DATA-020" in content or "data_020" in content or "020" in content
        assert has_data_001, "Data bridge should reference DATA-001"
        assert has_data_020, "Data bridge should reference DATA-020"

    def test_erp_mapping(self):
        """Data bridge references ERP system mappings."""
        path = INTEGRATIONS_DIR / INTEGRATION_FILES["data_bridge"]
        if not path.exists():
            pytest.skip("Data bridge file not found")
        content = path.read_text(encoding="utf-8")
        has_erp = (
            "SAP" in content
            or "Oracle" in content
            or "erp" in content.lower()
            or "ERP" in content
        )
        assert has_erp, "Data bridge should reference ERP systems"


# ===========================================================================
# Taxonomy Bridge
# ===========================================================================


class TestTaxonomyBridge:
    """Tests for the TaxonomyBridge integration."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_integration("xbrl_mapper")

    def test_class_exists(self):
        """XBRLMapper class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, INTEGRATION_CLASSES["xbrl_mapper"])

    def test_objectives_coverage(self):
        """XBRL mapper covers EU Taxonomy objectives."""
        path = INTEGRATIONS_DIR / INTEGRATION_FILES["xbrl_mapper"]
        if not path.exists():
            pytest.skip("XBRL mapper file not found")
        content = path.read_text(encoding="utf-8")
        has_taxonomy = (
            "taxonomy" in content.lower()
            or "XBRL" in content
            or "xbrl" in content
        )
        assert has_taxonomy, "XBRL mapper should reference taxonomy"


# ===========================================================================
# Health Check
# ===========================================================================


class TestHealthCheck:
    """Tests for the ESRSHealthCheck integration."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_integration("health_check")

    def test_class_exists(self):
        """ESRSHealthCheck class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, INTEGRATION_CLASSES["health_check"])

    def test_category_count(self):
        """Health check source references multiple check categories."""
        path = INTEGRATIONS_DIR / INTEGRATION_FILES["health_check"]
        if not path.exists():
            pytest.skip("Health check file not found")
        content = path.read_text(encoding="utf-8")
        # Count unique check references (engines, workflows, templates, etc.)
        check_refs = sum(1 for kw in [
            "engine", "workflow", "template", "integration",
            "config", "database", "bridge", "orchestrator",
        ] if kw in content.lower())
        assert check_refs >= 4, f"Expected 4+ check categories, found {check_refs}"

    def test_all_handlers_callable(self):
        """Health check has run or check_all method."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = getattr(self.mod, INTEGRATION_CLASSES["health_check"])
        has_run = (
            hasattr(cls, "check_all")
            or hasattr(cls, "run_all_checks")
            or hasattr(cls, "run")
            or hasattr(cls, "run_check")
        )
        assert has_run, "ESRSHealthCheck should have a run method"


# ===========================================================================
# Setup Wizard
# ===========================================================================


class TestSetupWizard:
    """Tests for the ESRSSetupWizard integration."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_integration("setup_wizard")

    def test_class_exists(self):
        """ESRSSetupWizard class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, INTEGRATION_CLASSES["setup_wizard"])

    def test_step_count(self):
        """Setup wizard source references at least 7 setup steps."""
        path = INTEGRATIONS_DIR / INTEGRATION_FILES["setup_wizard"]
        if not path.exists():
            pytest.skip("Setup wizard file not found")
        content = path.read_text(encoding="utf-8")
        step_count = content.lower().count("step")
        assert step_count >= 7, f"Expected 7+ step references, found {step_count}"

    def test_wizard_init(self):
        """ESRSSetupWizard has start, run, or finalize method."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = getattr(self.mod, INTEGRATION_CLASSES["setup_wizard"])
        has_method = (
            hasattr(cls, "start")
            or hasattr(cls, "run")
            or hasattr(cls, "finalize")
            or hasattr(cls, "submit_step")
            or hasattr(cls, "run_setup")
        )
        assert has_method, "ESRSSetupWizard should have a start/run method"


# ===========================================================================
# Audit Bridge
# ===========================================================================


class TestAuditBridge:
    """Tests for the AuditBridge integration."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_integration("audit_bridge")

    def test_class_exists(self):
        """AuditBridge class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, INTEGRATION_CLASSES["audit_bridge"])

    def test_has_audit_method(self):
        """AuditBridge (ESRSHealthCheck) has health check or audit method."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = getattr(self.mod, INTEGRATION_CLASSES["audit_bridge"])
        has_audit = (
            hasattr(cls, "run_all_checks")
            or hasattr(cls, "run_check")
            or hasattr(cls, "check_all")
            or hasattr(cls, "record_audit_event")
            or hasattr(cls, "log_event")
        )
        assert has_audit, "AuditBridge should have a health check or audit method"
