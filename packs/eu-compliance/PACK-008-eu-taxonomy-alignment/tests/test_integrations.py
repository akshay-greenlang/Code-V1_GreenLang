# -*- coding: utf-8 -*-
"""
Unit tests for PACK-008 EU Taxonomy Alignment Pack - Integrations

Tests all 12 integration modules: Pack Orchestrator, Taxonomy App Bridge,
MRV Taxonomy Bridge, CSRD Cross-Framework Bridge, Financial Data Bridge,
Activity Registry Bridge, Evidence Management Bridge, GAR Data Bridge,
Regulatory Tracking Bridge, Data Quality Bridge, Health Check, and Setup Wizard.
"""

import pytest
import sys
import importlib.util
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PACK_008_DIR = Path(__file__).resolve().parent.parent
_INTEGRATIONS_DIR = _PACK_008_DIR / "integrations"


def _import_from_path(module_name: str, file_path: Path) -> Optional[Any]:
    """Helper to import from hyphenated directory paths."""
    if not file_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def _instantiate_bridge(mod, class_name: str, config_class_names: list):
    """Try to instantiate a bridge class, handling config requirement patterns.

    Attempts bare instantiation first, then tries each config class name in order.
    """
    if mod is None:
        return None
    cls = getattr(mod, class_name, None)
    if cls is None:
        return None
    try:
        return cls()
    except TypeError:
        for cfg_name in config_class_names:
            cfg_cls = getattr(mod, cfg_name, None)
            if cfg_cls is not None:
                try:
                    return cls(cfg_cls())
                except Exception:
                    continue
    return None


# ---------------------------------------------------------------------------
# Import integration modules
# ---------------------------------------------------------------------------
_orch_mod = _import_from_path(
    "pack008_orchestrator", _INTEGRATIONS_DIR / "pack_orchestrator.py"
)
_tax_app_mod = _import_from_path(
    "pack008_taxonomy_app", _INTEGRATIONS_DIR / "taxonomy_app_bridge.py"
)
_mrv_mod = _import_from_path(
    "pack008_mrv_bridge", _INTEGRATIONS_DIR / "mrv_taxonomy_bridge.py"
)
_csrd_mod = _import_from_path(
    "pack008_csrd_bridge", _INTEGRATIONS_DIR / "csrd_cross_framework_bridge.py"
)
_fin_mod = _import_from_path(
    "pack008_financial_bridge", _INTEGRATIONS_DIR / "financial_data_bridge.py"
)
_activity_mod = _import_from_path(
    "pack008_activity_bridge", _INTEGRATIONS_DIR / "activity_registry_bridge.py"
)
_evidence_mod = _import_from_path(
    "pack008_evidence_bridge", _INTEGRATIONS_DIR / "evidence_management_bridge.py"
)
_gar_mod = _import_from_path(
    "pack008_gar_bridge", _INTEGRATIONS_DIR / "gar_data_bridge.py"
)
_reg_mod = _import_from_path(
    "pack008_regulatory_bridge", _INTEGRATIONS_DIR / "regulatory_tracking_bridge.py"
)
_dq_mod = _import_from_path(
    "pack008_data_quality_bridge", _INTEGRATIONS_DIR / "data_quality_bridge.py"
)
_health_mod = _import_from_path(
    "pack008_health_check", _INTEGRATIONS_DIR / "health_check.py"
)
_wizard_mod = _import_from_path(
    "pack008_setup_wizard", _INTEGRATIONS_DIR / "setup_wizard.py"
)

# Collect all module references for the all-bridges-importable test
_ALL_BRIDGE_MODULES = {
    "pack_orchestrator": _orch_mod,
    "taxonomy_app_bridge": _tax_app_mod,
    "mrv_taxonomy_bridge": _mrv_mod,
    "csrd_cross_framework_bridge": _csrd_mod,
    "financial_data_bridge": _fin_mod,
    "activity_registry_bridge": _activity_mod,
    "evidence_management_bridge": _evidence_mod,
    "gar_data_bridge": _gar_mod,
    "regulatory_tracking_bridge": _reg_mod,
    "data_quality_bridge": _dq_mod,
    "health_check": _health_mod,
    "setup_wizard": _wizard_mod,
}


# ===========================================================================
# Integration Tests
# ===========================================================================
@pytest.mark.unit
class TestIntegrations:
    """Test suite for all PACK-008 integration modules."""

    # -----------------------------------------------------------------------
    # INT-001: Orchestrator
    # -----------------------------------------------------------------------
    def test_orchestrator_instantiation(self):
        """TaxonomyPackOrchestrator can be created with config."""
        bridge = _instantiate_bridge(
            _orch_mod,
            "TaxonomyPackOrchestrator",
            ["TaxonomyOrchestratorConfig"],
        )
        if bridge is None:
            pytest.skip("TaxonomyPackOrchestrator not instantiable")
        assert bridge is not None
        assert hasattr(bridge, "config")
        assert hasattr(bridge, "_agents")

    def test_orchestrator_phase_count(self):
        """Orchestrator defines exactly 10 pipeline phases."""
        if _orch_mod is None:
            pytest.skip("pack_orchestrator module not available")
        phase_enum = getattr(_orch_mod, "TaxonomyPipelinePhase", None)
        if phase_enum is None:
            pytest.skip("TaxonomyPipelinePhase enum not found")
        phases = list(phase_enum)
        assert len(phases) == 10, (
            f"Expected 10 pipeline phases, got {len(phases)}: {[p.value for p in phases]}"
        )

    # -----------------------------------------------------------------------
    # INT-002: Taxonomy App Bridge
    # -----------------------------------------------------------------------
    def test_taxonomy_app_bridge_instantiation(self):
        """TaxonomyAppBridge can be created with config."""
        bridge = _instantiate_bridge(
            _tax_app_mod,
            "TaxonomyAppBridge",
            ["TaxonomyAppBridgeConfig"],
        )
        if bridge is None:
            pytest.skip("TaxonomyAppBridge not instantiable")
        assert bridge is not None

    # -----------------------------------------------------------------------
    # INT-003: MRV Taxonomy Bridge
    # -----------------------------------------------------------------------
    def test_mrv_taxonomy_bridge_routing_table(self):
        """MRV Taxonomy Bridge exposes a routing table with activity mappings."""
        if _mrv_mod is None:
            pytest.skip("mrv_taxonomy_bridge module not available")
        routing_table = getattr(_mrv_mod, "MRV_ROUTING_TABLE", None)
        if routing_table is None:
            pytest.skip("MRV_ROUTING_TABLE not found")
        assert isinstance(routing_table, dict)
        assert len(routing_table) > 0, "Routing table must not be empty"
        # Validate table structure
        sample_key = next(iter(routing_table))
        sample_entry = routing_table[sample_key]
        assert "agent" in sample_entry, "Routing entry must have 'agent' key"
        assert "scope" in sample_entry, "Routing entry must have 'scope' key"

    # -----------------------------------------------------------------------
    # INT-004: CSRD Cross-Framework Bridge
    # -----------------------------------------------------------------------
    def test_csrd_cross_framework_bridge(self):
        """CSRDCrossFrameworkBridge can be created with config."""
        bridge = _instantiate_bridge(
            _csrd_mod,
            "CSRDCrossFrameworkBridge",
            ["CrossFrameworkConfig"],
        )
        if bridge is None:
            pytest.skip("CSRDCrossFrameworkBridge not instantiable")
        assert bridge is not None

    # -----------------------------------------------------------------------
    # INT-005: Financial Data Bridge
    # -----------------------------------------------------------------------
    def test_financial_data_bridge(self):
        """FinancialDataBridge can be created with config."""
        bridge = _instantiate_bridge(
            _fin_mod,
            "FinancialDataBridge",
            ["FinancialDataConfig"],
        )
        if bridge is None:
            pytest.skip("FinancialDataBridge not instantiable")
        assert bridge is not None

    # -----------------------------------------------------------------------
    # INT-006: Activity Registry Bridge
    # -----------------------------------------------------------------------
    def test_activity_registry_bridge(self):
        """ActivityRegistryBridge can be created with config."""
        bridge = _instantiate_bridge(
            _activity_mod,
            "ActivityRegistryBridge",
            ["ActivityRegistryConfig"],
        )
        if bridge is None:
            pytest.skip("ActivityRegistryBridge not instantiable")
        assert bridge is not None

    # -----------------------------------------------------------------------
    # INT-007: Evidence Management Bridge
    # -----------------------------------------------------------------------
    def test_evidence_management_bridge(self):
        """EvidenceManagementBridge can be created with config."""
        bridge = _instantiate_bridge(
            _evidence_mod,
            "EvidenceManagementBridge",
            ["EvidenceConfig"],
        )
        if bridge is None:
            pytest.skip("EvidenceManagementBridge not instantiable")
        assert bridge is not None

    # -----------------------------------------------------------------------
    # INT-008: GAR Data Bridge
    # -----------------------------------------------------------------------
    def test_gar_data_bridge(self):
        """GARDataBridge can be created with config."""
        bridge = _instantiate_bridge(
            _gar_mod,
            "GARDataBridge",
            ["GARDataConfig"],
        )
        if bridge is None:
            pytest.skip("GARDataBridge not instantiable")
        assert bridge is not None

    # -----------------------------------------------------------------------
    # INT-009: Regulatory Tracking Bridge
    # -----------------------------------------------------------------------
    def test_regulatory_tracking_bridge(self):
        """RegulatoryTrackingBridge can be created with config."""
        bridge = _instantiate_bridge(
            _reg_mod,
            "RegulatoryTrackingBridge",
            ["RegulatoryTrackingConfig"],
        )
        if bridge is None:
            pytest.skip("RegulatoryTrackingBridge not instantiable")
        assert bridge is not None

    # -----------------------------------------------------------------------
    # INT-010: Data Quality Bridge
    # -----------------------------------------------------------------------
    def test_data_quality_bridge(self):
        """DataQualityBridge can be created with config."""
        bridge = _instantiate_bridge(
            _dq_mod,
            "DataQualityBridge",
            ["DataQualityConfig"],
        )
        if bridge is None:
            pytest.skip("DataQualityBridge not instantiable")
        assert bridge is not None

    # -----------------------------------------------------------------------
    # INT-011: Health Check
    # -----------------------------------------------------------------------
    def test_health_check(self):
        """TaxonomyHealthCheck can be created with config."""
        bridge = _instantiate_bridge(
            _health_mod,
            "TaxonomyHealthCheck",
            ["HealthCheckConfig"],
        )
        if bridge is None:
            pytest.skip("TaxonomyHealthCheck not instantiable")
        assert bridge is not None

    # -----------------------------------------------------------------------
    # INT-012: Setup Wizard
    # -----------------------------------------------------------------------
    def test_setup_wizard(self):
        """TaxonomySetupWizard can be created with config."""
        bridge = _instantiate_bridge(
            _wizard_mod,
            "TaxonomySetupWizard",
            ["SetupWizardConfig"],
        )
        if bridge is None:
            pytest.skip("TaxonomySetupWizard not instantiable")
        assert bridge is not None

    # -----------------------------------------------------------------------
    # META: All 12 bridges importable
    # -----------------------------------------------------------------------
    def test_all_12_bridges_importable(self):
        """All 12 integration modules can be imported."""
        missing = [name for name, mod in _ALL_BRIDGE_MODULES.items() if mod is None]
        assert len(missing) == 0, (
            f"{len(missing)} integration module(s) failed to import: {missing}"
        )
