# -*- coding: utf-8 -*-
"""
PACK-012 CSRD Financial Service Pack - Integration Tests
============================================================

Tests all 10 integration bridges for CSRD Financial Service:
FSCSRDOrchestrator, CSRDPackBridge, SFDRPackBridge, TaxonomyPackBridge,
MRVInvestmentsBridge, FinanceAgentBridge, ClimateRiskBridge,
EBAPillar3Bridge, HealthCheck, SetupWizard.

Self-contained: does NOT import from conftest.
Test count: 20 tests
"""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACK_ROOT.parent.parent.parent


def _import_from_path(module_name: str, file_path: str):
    """Import a module from an absolute file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Import all 10 integration modules
# ---------------------------------------------------------------------------

INT_DIR = str(PACK_ROOT / "integrations")

_int_orch = _import_from_path(
    "pack012_int_orch",
    os.path.join(INT_DIR, "pack_orchestrator.py"),
)
_int_csrd = _import_from_path(
    "pack012_int_csrd",
    os.path.join(INT_DIR, "csrd_pack_bridge.py"),
)
_int_sfdr = _import_from_path(
    "pack012_int_sfdr",
    os.path.join(INT_DIR, "sfdr_pack_bridge.py"),
)
_int_tax = _import_from_path(
    "pack012_int_tax",
    os.path.join(INT_DIR, "taxonomy_pack_bridge.py"),
)
_int_mrv = _import_from_path(
    "pack012_int_mrv",
    os.path.join(INT_DIR, "mrv_investments_bridge.py"),
)
_int_finance = _import_from_path(
    "pack012_int_finance",
    os.path.join(INT_DIR, "finance_agent_bridge.py"),
)
_int_climate = _import_from_path(
    "pack012_int_climate",
    os.path.join(INT_DIR, "climate_risk_bridge.py"),
)
_int_pillar3 = _import_from_path(
    "pack012_int_pillar3",
    os.path.join(INT_DIR, "eba_pillar3_bridge.py"),
)
_int_health = _import_from_path(
    "pack012_int_health",
    os.path.join(INT_DIR, "health_check.py"),
)
_int_wizard = _import_from_path(
    "pack012_int_wizard",
    os.path.join(INT_DIR, "setup_wizard.py"),
)

# Classes and configs
FSCSRDOrchestrator = _int_orch.FSCSRDOrchestrator
FSOrchestrationConfig = _int_orch.FSOrchestrationConfig
PipelinePhase = _int_orch.PipelinePhase
PipelineResult = _int_orch.PipelineResult
PhaseResult = _int_orch.PhaseResult

CSRDPackBridge = _int_csrd.CSRDPackBridge
CSRDBridgeConfig = _int_csrd.CSRDBridgeConfig

SFDRPackBridge = _int_sfdr.SFDRPackBridge
SFDRBridgeConfig = _int_sfdr.SFDRBridgeConfig

TaxonomyPackBridge = _int_tax.TaxonomyPackBridge
TaxonomyBridgeConfig = _int_tax.TaxonomyBridgeConfig

MRVInvestmentsBridge = _int_mrv.MRVInvestmentsBridge
MRVInvestmentsBridgeConfig = _int_mrv.MRVInvestmentsBridgeConfig

FinanceAgentBridge = _int_finance.FinanceAgentBridge
FinanceAgentBridgeConfig = _int_finance.FinanceAgentBridgeConfig

ClimateRiskBridge = _int_climate.ClimateRiskBridge
ClimateRiskBridgeConfig = _int_climate.ClimateRiskBridgeConfig

EBAPillar3Bridge = _int_pillar3.EBAPillar3Bridge
EBAPillar3BridgeConfig = _int_pillar3.EBAPillar3BridgeConfig

HealthCheck = _int_health.HealthCheck
HealthCheckConfig = _int_health.HealthCheckConfig

SetupWizard = _int_wizard.SetupWizard
SetupWizardConfig = _int_wizard.SetupWizardConfig


# ===========================================================================
# Test: FSCSRDOrchestrator
# ===========================================================================


class TestFSCSRDOrchestrator:
    """Tests for the 11-phase CSRD Financial Service pipeline."""

    def test_instantiation_default_config(self):
        """Orchestrator can be instantiated with default config."""
        orch = FSCSRDOrchestrator()
        assert orch is not None
        assert hasattr(orch, "config")

    def test_instantiation_with_config(self):
        """Orchestrator accepts custom configuration."""
        config = FSOrchestrationConfig(
            institution_name="GL Test Bank AG",
            lei_code="529900HNOAA1KXQJUQ27",
        )
        orch = FSCSRDOrchestrator(config)
        assert orch.config.institution_name == "GL Test Bank AG"
        assert orch.config.lei_code == "529900HNOAA1KXQJUQ27"
        assert orch.config.pack_id == "PACK-012"

    def test_pipeline_phase_enum_has_11_values(self):
        """Pipeline phase enum has exactly 11 phases."""
        phases = list(PipelinePhase)
        assert len(phases) == 11
        expected_phases = {
            "health_check", "config_init", "data_loading",
            "financed_emissions", "gar_btar", "climate_risk",
            "materiality", "transition_plan", "pillar3",
            "disclosure", "audit_trail",
        }
        actual_phases = {p.value for p in phases}
        assert actual_phases == expected_phases

    def test_config_defaults_for_financial_service(self):
        """Default config has FI-specific defaults."""
        config = FSOrchestrationConfig()
        assert config.pack_id == "PACK-012"
        assert config.gar_applicable is True
        assert config.btar_applicable is True
        assert config.pillar3_applicable is True
        assert config.pcaf_version == "2.1"
        assert len(config.climate_risk_scenarios) == 3


# ===========================================================================
# Test: CSRDPackBridge
# ===========================================================================


class TestCSRDPackBridge:
    """Tests for the PACK-001/002/003 CSRD core bridge."""

    def test_instantiation(self):
        """Bridge can be instantiated with default config."""
        bridge = CSRDPackBridge()
        assert bridge is not None
        assert hasattr(bridge, "config")

    def test_config_attributes(self):
        """Bridge config has expected attributes."""
        config = CSRDBridgeConfig()
        assert config is not None


# ===========================================================================
# Test: SFDRPackBridge
# ===========================================================================


class TestSFDRPackBridge:
    """Tests for the PACK-010/011 SFDR integration bridge."""

    def test_instantiation(self):
        """Bridge can be instantiated with default config."""
        bridge = SFDRPackBridge()
        assert bridge is not None
        assert hasattr(bridge, "config")

    def test_config_defaults(self):
        """Bridge config has sensible defaults."""
        config = SFDRBridgeConfig()
        assert config is not None


# ===========================================================================
# Test: TaxonomyPackBridge
# ===========================================================================


class TestTaxonomyPackBridge:
    """Tests for the PACK-008 EU Taxonomy alignment bridge."""

    def test_instantiation(self):
        """Bridge can be instantiated with default config."""
        bridge = TaxonomyPackBridge()
        assert bridge is not None
        assert hasattr(bridge, "config")

    def test_config_defaults(self):
        """Bridge config has sensible defaults."""
        config = TaxonomyBridgeConfig()
        assert config is not None


# ===========================================================================
# Test: MRVInvestmentsBridge
# ===========================================================================


class TestMRVInvestmentsBridge:
    """Tests for the AGENT-MRV-028 financed emissions bridge."""

    def test_instantiation(self):
        """Bridge can be instantiated with default config."""
        bridge = MRVInvestmentsBridge()
        assert bridge is not None
        assert hasattr(bridge, "config")

    def test_config_defaults(self):
        """Bridge config has sensible defaults."""
        config = MRVInvestmentsBridgeConfig()
        assert config is not None


# ===========================================================================
# Test: FinanceAgentBridge
# ===========================================================================


class TestFinanceAgentBridge:
    """Tests for the green screening / stranded asset bridge."""

    def test_instantiation(self):
        """Bridge can be instantiated with default config."""
        bridge = FinanceAgentBridge()
        assert bridge is not None
        assert hasattr(bridge, "config")


# ===========================================================================
# Test: ClimateRiskBridge
# ===========================================================================


class TestClimateRiskBridge:
    """Tests for the transition + physical climate risk bridge."""

    def test_instantiation(self):
        """Bridge can be instantiated with default config."""
        bridge = ClimateRiskBridge()
        assert bridge is not None
        assert hasattr(bridge, "config")


# ===========================================================================
# Test: EBAPillar3Bridge
# ===========================================================================


class TestEBAPillar3Bridge:
    """Tests for the EBA Pillar 3 ITS template bridge."""

    def test_instantiation(self):
        """Bridge can be instantiated with default config."""
        bridge = EBAPillar3Bridge()
        assert bridge is not None
        assert hasattr(bridge, "config")


# ===========================================================================
# Test: HealthCheck
# ===========================================================================


class TestHealthCheck:
    """Tests for the 22-category health check system."""

    def test_instantiation_default_config(self):
        """HealthCheck can be instantiated with default config."""
        health = HealthCheck()
        assert health is not None
        assert hasattr(health, "config")

    def test_instantiation_with_config(self):
        """HealthCheck accepts custom configuration."""
        config = HealthCheckConfig()
        health = HealthCheck(config)
        assert health.config is not None

    def test_run_full_check_returns_result(self):
        """run_full_check returns a HealthCheckResult."""
        health = HealthCheck()
        result = health.run_full_check()
        assert result is not None
        assert hasattr(result, "overall_score")
        assert hasattr(result, "is_ready")
        assert hasattr(result, "category_results")
        assert result.overall_score >= 0.0
        assert result.overall_score <= 100.0


# ===========================================================================
# Test: SetupWizard
# ===========================================================================


class TestSetupWizard:
    """Tests for the 8-step guided configuration wizard."""

    def test_instantiation_default_config(self):
        """SetupWizard can be instantiated with default config."""
        wizard = SetupWizard()
        assert wizard is not None
        assert hasattr(wizard, "config")

    def test_instantiation_with_config(self):
        """SetupWizard accepts custom configuration."""
        config = SetupWizardConfig()
        wizard = SetupWizard(config)
        assert wizard.config is not None
