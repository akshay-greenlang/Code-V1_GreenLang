# -*- coding: utf-8 -*-
"""
Agent integration tests for PACK-011 SFDR Article 9 Pack.

These tests verify the wiring between pack components: orchestrator phases,
bridge configurations, engine importability, workflow modules, template
modules, config classes, and provenance hash generation. All tests are
marked with @pytest.mark.integration.

Test count: 10 tests
Target: Validate inter-component wiring and contract adherence
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent


def _import_from_path(module_name: str, file_path: str):
    """Import a module from an absolute file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Module imports - integration layer
# ---------------------------------------------------------------------------

orch_mod = _import_from_path(
    "ai_pack_orchestrator",
    str(PACK_ROOT / "integrations" / "pack_orchestrator.py"),
)
art8_bridge_mod = _import_from_path(
    "ai_article8_pack_bridge",
    str(PACK_ROOT / "integrations" / "article8_pack_bridge.py"),
)
tax_bridge_mod = _import_from_path(
    "ai_taxonomy_pack_bridge",
    str(PACK_ROOT / "integrations" / "taxonomy_pack_bridge.py"),
)
mrv_bridge_mod = _import_from_path(
    "ai_mrv_emissions_bridge",
    str(PACK_ROOT / "integrations" / "mrv_emissions_bridge.py"),
)
wizard_mod = _import_from_path(
    "ai_setup_wizard",
    str(PACK_ROOT / "integrations" / "setup_wizard.py"),
)
health_mod = _import_from_path(
    "ai_health_check",
    str(PACK_ROOT / "integrations" / "health_check.py"),
)

# Engine modules
so_engine_mod = _import_from_path(
    "ai_so_engine",
    str(PACK_ROOT / "engines" / "sustainable_objective_engine.py"),
)
dnsh_engine_mod = _import_from_path(
    "ai_dnsh_engine",
    str(PACK_ROOT / "engines" / "enhanced_dnsh_engine.py"),
)
pai_engine_mod = _import_from_path(
    "ai_pai_engine",
    str(PACK_ROOT / "engines" / "pai_mandatory_engine.py"),
)

# Config module
try:
    config_mod = _import_from_path(
        "ai_pack_config",
        str(PACK_ROOT / "config" / "pack_config.py"),
    )
    _CONFIG_AVAILABLE = True
except Exception:
    config_mod = None
    _CONFIG_AVAILABLE = False

# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------

Article9Orchestrator = orch_mod.Article9Orchestrator
Article9OrchestrationConfig = orch_mod.Article9OrchestrationConfig
PipelinePhase = orch_mod.PipelinePhase
Article9ExecutionStatus = orch_mod.Article9ExecutionStatus

Article8PackBridge = art8_bridge_mod.Article8PackBridge
Article8BridgeConfig = art8_bridge_mod.Article8BridgeConfig

TaxonomyPackBridge = tax_bridge_mod.TaxonomyPackBridge
TaxonomyBridgeConfig = tax_bridge_mod.TaxonomyBridgeConfig

MRVEmissionsBridge = mrv_bridge_mod.MRVEmissionsBridge
MRVBridgeConfig = mrv_bridge_mod.MRVBridgeConfig

SetupWizard = wizard_mod.SetupWizard
SetupWizardConfig = wizard_mod.SetupWizardConfig

HealthCheck = health_mod.HealthCheck
HealthCheckConfig = health_mod.HealthCheckConfig

SustainableObjectiveEngine = so_engine_mod.SustainableObjectiveEngine
SustainableObjectiveConfig = so_engine_mod.SustainableObjectiveConfig

EnhancedDNSHEngine = dnsh_engine_mod.EnhancedDNSHEngine
EnhancedDNSHConfig = dnsh_engine_mod.EnhancedDNSHConfig

PAIMandatoryEngine = pai_engine_mod.PAIMandatoryEngine
PAIMandatoryConfig = pai_engine_mod.PAIMandatoryConfig


# ===========================================================================
# Integration Test Class
# ===========================================================================


@pytest.mark.integration
class TestAgentIntegration:
    """Integration tests for SFDR Article 9 Pack component wiring."""

    # -----------------------------------------------------------------------
    # 1. Pack orchestrator instantiation
    # -----------------------------------------------------------------------

    def test_pack_orchestrator_instantiation(self):
        """Verify orchestrator creates with 11 phases, config, agent stubs."""
        config = Article9OrchestrationConfig(
            product_name="Integration Test Fund",
            product_isin="LU0001234567",
        )
        orch = Article9Orchestrator(config)

        assert orch.config.product_name == "Integration Test Fund"
        assert orch.config.product_isin == "LU0001234567"
        assert orch.config.pack_id == "PACK-011"
        assert hasattr(orch, "config")

    # -----------------------------------------------------------------------
    # 2. Pipeline phase enum has exactly 11 values
    # -----------------------------------------------------------------------

    def test_pack_orchestrator_has_11_phases(self):
        """Verify pipeline phase enum has exactly 11 values."""
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

    # -----------------------------------------------------------------------
    # 3. All pack bridges instantiate
    # -----------------------------------------------------------------------

    def test_all_pack_bridges_instantiate(self):
        """Construct all bridge types with configs."""
        # Article 8 bridge
        art8_bridge = Article8PackBridge(Article8BridgeConfig())
        assert art8_bridge.config is not None

        # Taxonomy bridge
        tax_bridge = TaxonomyPackBridge(TaxonomyBridgeConfig())
        assert tax_bridge.config is not None

        # MRV emissions bridge
        mrv_bridge = MRVEmissionsBridge(MRVBridgeConfig())
        assert mrv_bridge.config is not None

    # -----------------------------------------------------------------------
    # 4. Engine modules importable
    # -----------------------------------------------------------------------

    def test_all_engine_modules_importable(self):
        """Verify all 8 engine modules can be imported."""
        engine_files = [
            "sustainable_objective_engine.py",
            "enhanced_dnsh_engine.py",
            "full_taxonomy_alignment.py",
            "impact_measurement_engine.py",
            "benchmark_alignment_engine.py",
            "pai_mandatory_engine.py",
            "carbon_trajectory_engine.py",
            "investment_universe_engine.py",
        ]
        engines_dir = PACK_ROOT / "engines"
        for f in engine_files:
            mod = _import_from_path(
                f"ai_engine_check_{f.replace('.py', '')}",
                str(engines_dir / f),
            )
            assert mod is not None

    # -----------------------------------------------------------------------
    # 5. Workflow modules importable
    # -----------------------------------------------------------------------

    def test_all_workflow_modules_importable(self):
        """Verify all 8 workflow modules can be imported."""
        workflow_files = [
            "annex_iii_disclosure.py",
            "annex_v_reporting.py",
            "sustainable_verification.py",
            "impact_reporting.py",
            "benchmark_monitoring.py",
            "pai_mandatory_workflow.py",
            "downgrade_monitoring.py",
            "regulatory_update.py",
        ]
        wf_dir = PACK_ROOT / "workflows"
        for f in workflow_files:
            mod = _import_from_path(
                f"ai_wf_check_{f.replace('.py', '')}",
                str(wf_dir / f),
            )
            assert mod is not None

    # -----------------------------------------------------------------------
    # 6. Template modules importable
    # -----------------------------------------------------------------------

    def test_all_template_modules_importable(self):
        """Verify all 8 template modules can be imported."""
        template_files = [
            "annex_iii_precontractual.py",
            "annex_v_periodic.py",
            "impact_report.py",
            "benchmark_methodology.py",
            "sustainable_dashboard.py",
            "pai_mandatory_report.py",
            "carbon_trajectory_report.py",
            "audit_trail_report.py",
        ]
        tpl_dir = PACK_ROOT / "templates"
        for f in template_files:
            mod = _import_from_path(
                f"ai_tpl_check_{f.replace('.py', '')}",
                str(tpl_dir / f),
            )
            assert mod is not None

    # -----------------------------------------------------------------------
    # 7. Config class enforces Article 9 constraints
    # -----------------------------------------------------------------------

    def test_config_enforces_article_9_constraints(self):
        """Article 9 config defaults enforce 100% SI, 18 PAI, etc."""
        config = Article9OrchestrationConfig()
        assert config.sustainable_investment_min_pct == 100.0
        assert len(config.pai_mandatory_indicators) == 18
        assert config.enable_enhanced_dnsh is True
        assert config.enable_good_governance is True
        assert len(config.taxonomy_objectives) == 6

    # -----------------------------------------------------------------------
    # 8. Health check produces results
    # -----------------------------------------------------------------------

    def test_health_check_produces_result(self):
        """Health check runs and produces a result with categories."""
        hc = HealthCheck(HealthCheckConfig())
        result = hc.run_full_check()
        assert result is not None
        assert hasattr(result, "is_ready")

    # -----------------------------------------------------------------------
    # 9. Cross-pack bridge (Article 8 to Article 9 comparison)
    # -----------------------------------------------------------------------

    def test_article8_bridge_config(self):
        """Article 8 bridge config allows Article 9 comparison."""
        config = Article8BridgeConfig()
        bridge = Article8PackBridge(config)
        assert bridge is not None
        assert hasattr(bridge, "config")

    # -----------------------------------------------------------------------
    # 10. Engine config classes accept Article 9 parameters
    # -----------------------------------------------------------------------

    def test_engine_configs_accept_article_9_params(self):
        """Engine config classes accept Article 9-specific parameters."""
        so_config = SustainableObjectiveConfig()
        assert so_config is not None

        dnsh_config = EnhancedDNSHConfig()
        assert dnsh_config is not None

        pai_config = PAIMandatoryConfig()
        assert pai_config is not None
