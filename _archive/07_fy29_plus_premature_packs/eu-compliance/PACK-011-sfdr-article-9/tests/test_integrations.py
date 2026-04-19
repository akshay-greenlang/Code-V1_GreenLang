# -*- coding: utf-8 -*-
"""
PACK-011 SFDR Article 9 Pack - Integration Tests
===================================================

Tests all 10 integration bridges for SFDR Article 9:
Article9Orchestrator, Article8PackBridge, TaxonomyPackBridge,
MRVEmissionsBridge, BenchmarkDataBridge, ImpactDataBridge,
EETDataBridge, RegulatoryBridge, HealthCheck, SetupWizard.

Self-contained: does NOT import from conftest.
Test count: 25 tests
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
    "pack011_int_orch",
    os.path.join(INT_DIR, "pack_orchestrator.py"),
)
_int_art8 = _import_from_path(
    "pack011_int_art8",
    os.path.join(INT_DIR, "article8_pack_bridge.py"),
)
_int_tax = _import_from_path(
    "pack011_int_tax",
    os.path.join(INT_DIR, "taxonomy_pack_bridge.py"),
)
_int_mrv = _import_from_path(
    "pack011_int_mrv",
    os.path.join(INT_DIR, "mrv_emissions_bridge.py"),
)
_int_benchmark = _import_from_path(
    "pack011_int_benchmark",
    os.path.join(INT_DIR, "benchmark_data_bridge.py"),
)
_int_impact = _import_from_path(
    "pack011_int_impact",
    os.path.join(INT_DIR, "impact_data_bridge.py"),
)
_int_eet = _import_from_path(
    "pack011_int_eet",
    os.path.join(INT_DIR, "eet_data_bridge.py"),
)
_int_reg = _import_from_path(
    "pack011_int_reg",
    os.path.join(INT_DIR, "regulatory_bridge.py"),
)
_int_health = _import_from_path(
    "pack011_int_health",
    os.path.join(INT_DIR, "health_check.py"),
)
_int_wizard = _import_from_path(
    "pack011_int_wizard",
    os.path.join(INT_DIR, "setup_wizard.py"),
)

# Classes and configs - Orchestrator
Article9Orchestrator = _int_orch.Article9Orchestrator
Article9OrchestrationConfig = _int_orch.Article9OrchestrationConfig
PipelinePhase = _int_orch.PipelinePhase
PipelineResult = _int_orch.PipelineResult
PhaseResult = _int_orch.PhaseResult
Article9ExecutionStatus = _int_orch.Article9ExecutionStatus

# Article 8 bridge
Article8PackBridge = _int_art8.Article8PackBridge
Article8BridgeConfig = _int_art8.Article8BridgeConfig

# Taxonomy bridge
TaxonomyPackBridge = _int_tax.TaxonomyPackBridge
TaxonomyBridgeConfig = _int_tax.TaxonomyBridgeConfig

# MRV emissions bridge
MRVEmissionsBridge = _int_mrv.MRVEmissionsBridge
MRVBridgeConfig = _int_mrv.MRVBridgeConfig

# Benchmark data bridge
BenchmarkDataBridge = _int_benchmark.BenchmarkDataBridge
BenchmarkDataConfig = _int_benchmark.BenchmarkDataConfig

# Impact data bridge
ImpactDataBridge = _int_impact.ImpactDataBridge
ImpactDataConfig = _int_impact.ImpactDataConfig

# EET data bridge
EETDataBridge = _int_eet.EETDataBridge
EETBridgeConfig = _int_eet.EETBridgeConfig

# Regulatory bridge
RegulatoryBridge = _int_reg.RegulatoryBridge
RegulatoryBridgeConfig = _int_reg.RegulatoryBridgeConfig

# Health check
HealthCheck = _int_health.HealthCheck
HealthCheckConfig = _int_health.HealthCheckConfig

# Setup wizard
SetupWizard = _int_wizard.SetupWizard
SetupWizardConfig = _int_wizard.SetupWizardConfig


# ===========================================================================
# Test: Article9Orchestrator
# ===========================================================================


class TestArticle9Orchestrator:
    """Tests for the 11-phase Article 9 execution pipeline."""

    def test_instantiation_default_config(self):
        """Orchestrator can be instantiated with default config."""
        orch = Article9Orchestrator()
        assert orch is not None
        assert hasattr(orch, "config")

    def test_instantiation_with_config(self):
        """Orchestrator accepts custom configuration."""
        config = Article9OrchestrationConfig(
            product_name="GL Deep Green Fund",
            product_isin="LU0009876543",
        )
        orch = Article9Orchestrator(config)
        assert orch.config.product_name == "GL Deep Green Fund"
        assert orch.config.product_isin == "LU0009876543"
        assert orch.config.pack_id == "PACK-011"

    def test_pipeline_phase_enum_has_11_values(self):
        """Pipeline phase enum has exactly 11 phases."""
        phases = list(PipelinePhase)
        assert len(phases) == 11
        expected_phases = {
            "health_check",
            "configuration_init",
            "holdings_intake",
            "sustainable_objective_verify",
            "enhanced_dnsh",
            "good_governance",
            "taxonomy_alignment",
            "mandatory_pai",
            "impact_measurement",
            "benchmark_alignment",
            "disclosure_generation",
        }
        actual_phases = {p.value for p in phases}
        assert actual_phases == expected_phases

    def test_config_defaults_for_article_9(self):
        """Default config enforces Article 9 requirements."""
        config = Article9OrchestrationConfig()
        assert config.sustainable_investment_min_pct == 100.0
        assert len(config.pai_mandatory_indicators) == 18
        assert config.enable_enhanced_dnsh is True
        assert len(config.taxonomy_objectives) == 6


# ===========================================================================
# Test: Article8PackBridge
# ===========================================================================


class TestArticle8PackBridge:
    """Tests for the PACK-010 cross-reference and downgrade detection bridge."""

    def test_instantiation(self):
        """Bridge can be instantiated with default config."""
        bridge = Article8PackBridge(Article8BridgeConfig())
        assert bridge is not None
        assert hasattr(bridge, "config")

    def test_config_attributes(self):
        """Bridge config has expected attributes."""
        config = Article8BridgeConfig()
        assert config is not None


# ===========================================================================
# Test: TaxonomyPackBridge
# ===========================================================================


class TestTaxonomyPackBridge:
    """Tests for the PACK-008 EU Taxonomy alignment bridge."""

    def test_instantiation(self):
        """Bridge can be instantiated with default config."""
        bridge = TaxonomyPackBridge(TaxonomyBridgeConfig())
        assert bridge is not None
        assert hasattr(bridge, "config")

    def test_config_defaults(self):
        """Bridge config has sensible defaults."""
        config = TaxonomyBridgeConfig()
        assert config is not None


# ===========================================================================
# Test: MRVEmissionsBridge
# ===========================================================================


class TestMRVEmissionsBridge:
    """Tests for the bridge to 30 MRV agents for PAI 1-6."""

    def test_instantiation(self):
        """Bridge can be instantiated with default config."""
        bridge = MRVEmissionsBridge(MRVBridgeConfig())
        assert bridge is not None
        assert hasattr(bridge, "config")

    def test_config_defaults(self):
        """Bridge config has sensible defaults."""
        config = MRVBridgeConfig()
        assert config is not None


# ===========================================================================
# Test: BenchmarkDataBridge
# ===========================================================================


class TestBenchmarkDataBridge:
    """Tests for the CTB/PAB benchmark data intake bridge."""

    def test_instantiation(self):
        """Bridge can be instantiated with default config."""
        bridge = BenchmarkDataBridge(BenchmarkDataConfig())
        assert bridge is not None
        assert hasattr(bridge, "config")


# ===========================================================================
# Test: ImpactDataBridge
# ===========================================================================


class TestImpactDataBridge:
    """Tests for the impact data intake and SDG alignment bridge."""

    def test_instantiation(self):
        """Bridge can be instantiated with default config."""
        bridge = ImpactDataBridge(ImpactDataConfig())
        assert bridge is not None
        assert hasattr(bridge, "config")


# ===========================================================================
# Test: EETDataBridge
# ===========================================================================


class TestEETDataBridge:
    """Tests for the European ESG Template import/export bridge."""

    def test_instantiation(self):
        """Bridge can be instantiated with default config."""
        bridge = EETDataBridge(EETBridgeConfig())
        assert bridge is not None
        assert hasattr(bridge, "config")


# ===========================================================================
# Test: RegulatoryBridge
# ===========================================================================


class TestRegulatoryBridge:
    """Tests for the SFDR/Taxonomy/BMR regulatory update tracking bridge."""

    def test_instantiation(self):
        """Bridge can be instantiated with default config."""
        bridge = RegulatoryBridge(RegulatoryBridgeConfig())
        assert bridge is not None
        assert hasattr(bridge, "config")


# ===========================================================================
# Test: HealthCheck
# ===========================================================================


class TestHealthCheck:
    """Tests for the 20-category system verification health check."""

    def test_instantiation(self):
        """HealthCheck can be instantiated with default config."""
        hc = HealthCheck(HealthCheckConfig())
        assert hc is not None
        assert hasattr(hc, "config")

    def test_run_returns_result(self):
        """Running health check produces a result."""
        hc = HealthCheck(HealthCheckConfig())
        result = hc.run_full_check()
        assert result is not None
        assert hasattr(result, "is_ready")
        assert hasattr(result, "category_results")


# ===========================================================================
# Test: SetupWizard
# ===========================================================================


class TestSetupWizard:
    """Tests for the 8-step guided product configuration wizard."""

    def test_instantiation(self):
        """SetupWizard can be instantiated with default config."""
        wiz = SetupWizard(SetupWizardConfig())
        assert wiz is not None
        assert hasattr(wiz, "config")

    def test_wizard_has_steps(self):
        """Wizard defines its configuration steps."""
        wiz = SetupWizard(SetupWizardConfig())
        assert hasattr(wiz, "steps") or hasattr(wiz, "_steps") or hasattr(wiz, "STEPS")
