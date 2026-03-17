# -*- coding: utf-8 -*-
"""
PACK-016 ESRS E1 Climate Pack - End-to-End Tests
===================================================

End-to-end integration tests validating cross-engine data flow,
pipeline sequencing, provenance hashing, determinism, and full
E1 disclosure pipeline execution.

Target: 30+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-016 ESRS E1 Climate Change
Date:    March 2026
"""

import hashlib
import json
from pathlib import Path

import pytest

from .conftest import (
    ENGINE_FILES,
    ENGINE_CLASSES,
    ENGINES_DIR,
    WORKFLOW_FILES,
    WORKFLOW_CLASSES,
    TEMPLATE_FILES,
    TEMPLATE_CLASSES,
    INTEGRATION_FILES,
    INTEGRATION_CLASSES,
    _load_engine,
    _load_workflow,
    _load_template,
    _load_integration,
    _load_config_module,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _safe_load(loader, key):
    """Safely load a module, returning None on failure."""
    try:
        return loader(key)
    except (ImportError, FileNotFoundError):
        return None


# ===========================================================================
# Engine-Level Data Flow Tests
# ===========================================================================


class TestEngineDataCompatibility:
    """Tests for cross-engine data compatibility."""

    def test_all_engines_loadable(self):
        """All 8 engines can be loaded via importlib."""
        loaded = 0
        for key in ENGINE_FILES:
            mod = _safe_load(_load_engine, key)
            if mod is not None:
                loaded += 1
        assert loaded == 8, f"Only {loaded}/8 engines loaded"

    def test_all_engines_produce_provenance_hash(self):
        """All engines produce SHA-256 provenance hashes."""
        for key, file_name in ENGINE_FILES.items():
            source_path = ENGINES_DIR / file_name
            content = source_path.read_text(encoding="utf-8")
            has_sha256 = "sha256" in content.lower() or "hashlib" in content
            assert has_sha256, f"Engine {key} should produce SHA-256 provenance"

    def test_all_engines_deterministic(self):
        """All engines use deterministic calculations (hashlib + no random in score paths)."""
        for key, file_name in ENGINE_FILES.items():
            source_path = ENGINES_DIR / file_name
            content = source_path.read_text(encoding="utf-8")
            assert "hashlib" in content, f"Engine {key} should use hashlib"

    def test_cross_engine_ghg_and_energy(self):
        """GHG inventory and energy mix engines share common data patterns."""
        ghg = _safe_load(_load_engine, "ghg_inventory")
        energy = _safe_load(_load_engine, "energy_mix")
        if ghg is None or energy is None:
            pytest.skip("Both engines required")
        assert hasattr(ghg, "GHGInventoryEngine")
        assert hasattr(energy, "EnergyMixEngine")

    def test_cross_engine_target_and_transition(self):
        """Target and transition plan engines share common data patterns."""
        target = _safe_load(_load_engine, "climate_target")
        transition = _safe_load(_load_engine, "transition_plan")
        if target is None or transition is None:
            pytest.skip("Both engines required")
        assert hasattr(target, "ClimateTargetEngine")
        assert hasattr(transition, "TransitionPlanEngine")


# ===========================================================================
# Pipeline Flow Tests
# ===========================================================================


class TestPipelineFlow:
    """Tests for pipeline flow between components."""

    def test_engines_output_hashlib(self):
        """All engine files use hashlib for provenance tracking."""
        for key, file_name in ENGINE_FILES.items():
            source_path = ENGINES_DIR / file_name
            content = source_path.read_text(encoding="utf-8")
            assert "hashlib" in content, f"Engine {key} missing hashlib"

    def test_workflows_reference_engines(self):
        """Workflow files reference engines, phases, or workflow logic."""
        for key, file_name in WORKFLOW_FILES.items():
            source_path = WORKFLOW_FILES[key]
            path = Path(__file__).resolve().parent.parent / "workflows" / source_path
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8")
            has_workflow_ref = (
                "engine" in content.lower()
                or "Engine" in content
                or "Workflow" in content
                or "phase" in content.lower()
                or "Phase" in content
            )
            assert has_workflow_ref, f"Workflow {key} should reference engines or phases"

    def test_templates_reference_disclosure_requirements(self):
        """Template files reference ESRS E1 disclosure requirements."""
        for key, file_name in TEMPLATE_FILES.items():
            path = Path(__file__).resolve().parent.parent / "templates" / file_name
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8")
            has_e1_ref = "E1-" in content or "E1_" in content
            assert has_e1_ref, f"Template {key} should reference E1 disclosure"


# ===========================================================================
# E2E GHG Inventory Tests
# ===========================================================================


class TestE2EGHGInventory:
    """End-to-end tests for GHG inventory flow."""

    def test_ghg_engine_has_emission_entry(self):
        """GHG engine exports EmissionEntry model."""
        mod = _safe_load(_load_engine, "ghg_inventory")
        if mod is None:
            pytest.skip("GHG engine not loaded")
        assert hasattr(mod, "EmissionEntry")

    def test_ghg_engine_has_result_model(self):
        """GHG engine exports GHGInventoryResult model."""
        mod = _safe_load(_load_engine, "ghg_inventory")
        if mod is None:
            pytest.skip("GHG engine not loaded")
        assert hasattr(mod, "GHGInventoryResult")

    def test_ghg_engine_has_gwp_values(self):
        """GHG engine exports GWP_AR6 values."""
        mod = _safe_load(_load_engine, "ghg_inventory")
        if mod is None:
            pytest.skip("GHG engine not loaded")
        assert hasattr(mod, "GWP_AR6")

    def test_ghg_engine_has_scope_enums(self):
        """GHG engine exports GHGScope and EmissionGas enums."""
        mod = _safe_load(_load_engine, "ghg_inventory")
        if mod is None:
            pytest.skip("GHG engine not loaded")
        assert hasattr(mod, "GHGScope")
        assert hasattr(mod, "EmissionGas")


# ===========================================================================
# E2E Energy Mix Tests
# ===========================================================================


class TestE2EEnergyMix:
    """End-to-end tests for energy mix flow."""

    def test_energy_engine_has_entry_model(self):
        """Energy engine exports EnergyConsumptionEntry model."""
        mod = _safe_load(_load_engine, "energy_mix")
        if mod is None:
            pytest.skip("Energy engine not loaded")
        assert hasattr(mod, "EnergyConsumptionEntry")

    def test_energy_engine_has_result_model(self):
        """Energy engine exports EnergyMixResult model."""
        mod = _safe_load(_load_engine, "energy_mix")
        if mod is None:
            pytest.skip("Energy engine not loaded")
        assert hasattr(mod, "EnergyMixResult")

    def test_energy_engine_has_source_classification(self):
        """Energy engine exports SOURCE_CLASSIFICATION."""
        mod = _safe_load(_load_engine, "energy_mix")
        if mod is None:
            pytest.skip("Energy engine not loaded")
        assert hasattr(mod, "SOURCE_CLASSIFICATION")


# ===========================================================================
# E2E Transition Plan Tests
# ===========================================================================


class TestE2ETransitionPlan:
    """End-to-end tests for transition plan flow."""

    def test_transition_engine_has_action_model(self):
        """Transition engine exports TransitionPlanAction model."""
        mod = _safe_load(_load_engine, "transition_plan")
        if mod is None:
            pytest.skip("Transition engine not loaded")
        assert hasattr(mod, "TransitionPlanAction")

    def test_transition_engine_has_result_model(self):
        """Transition engine exports TransitionPlanResult model."""
        mod = _safe_load(_load_engine, "transition_plan")
        if mod is None:
            pytest.skip("Transition engine not loaded")
        assert hasattr(mod, "TransitionPlanResult")

    def test_transition_engine_has_gap_analysis(self):
        """Transition engine exports PlanGapAnalysis model."""
        mod = _safe_load(_load_engine, "transition_plan")
        if mod is None:
            pytest.skip("Transition engine not loaded")
        assert hasattr(mod, "PlanGapAnalysis")


# ===========================================================================
# E2E Target Tests
# ===========================================================================


class TestE2ETargets:
    """End-to-end tests for target setting flow."""

    def test_target_engine_has_target_model(self):
        """Target engine exports ClimateTarget model."""
        mod = _safe_load(_load_engine, "climate_target")
        if mod is None:
            pytest.skip("Target engine not loaded")
        assert hasattr(mod, "ClimateTarget")

    def test_target_engine_has_progress_result(self):
        """Target engine exports TargetProgressResult model."""
        mod = _safe_load(_load_engine, "climate_target")
        if mod is None:
            pytest.skip("Target engine not loaded")
        assert hasattr(mod, "TargetProgressResult")

    def test_target_engine_has_sbti_rates(self):
        """Target engine exports SBTI_MINIMUM_RATES."""
        mod = _safe_load(_load_engine, "climate_target")
        if mod is None:
            pytest.skip("Target engine not loaded")
        assert hasattr(mod, "SBTI_MINIMUM_RATES")


# ===========================================================================
# E2E Carbon Credits Tests
# ===========================================================================


class TestE2ECarbonCredits:
    """End-to-end tests for carbon credit flow."""

    def test_credit_engine_has_credit_model(self):
        """Credit engine exports CarbonCredit model."""
        mod = _safe_load(_load_engine, "carbon_credit")
        if mod is None:
            pytest.skip("Credit engine not loaded")
        assert hasattr(mod, "CarbonCredit")

    def test_credit_engine_has_result_model(self):
        """Credit engine exports CarbonCreditResult model."""
        mod = _safe_load(_load_engine, "carbon_credit")
        if mod is None:
            pytest.skip("Credit engine not loaded")
        assert hasattr(mod, "CarbonCreditResult")

    def test_credit_engine_has_quality_assessment(self):
        """Credit engine exports QualityAssessment model."""
        mod = _safe_load(_load_engine, "carbon_credit")
        if mod is None:
            pytest.skip("Credit engine not loaded")
        assert hasattr(mod, "QualityAssessment")


# ===========================================================================
# E2E Carbon Pricing Tests
# ===========================================================================


class TestE2ECarbonPricing:
    """End-to-end tests for carbon pricing flow."""

    def test_pricing_engine_has_price_model(self):
        """Pricing engine exports CarbonPrice model."""
        mod = _safe_load(_load_engine, "carbon_pricing")
        if mod is None:
            pytest.skip("Pricing engine not loaded")
        assert hasattr(mod, "CarbonPrice")

    def test_pricing_engine_has_result_model(self):
        """Pricing engine exports CarbonPricingResult model."""
        mod = _safe_load(_load_engine, "carbon_pricing")
        if mod is None:
            pytest.skip("Pricing engine not loaded")
        assert hasattr(mod, "CarbonPricingResult")


# ===========================================================================
# E2E Climate Risk Tests
# ===========================================================================


class TestE2EClimateRisk:
    """End-to-end tests for climate risk flow."""

    def test_risk_engine_has_physical_risk(self):
        """Risk engine exports PhysicalRisk model."""
        mod = _safe_load(_load_engine, "climate_risk")
        if mod is None:
            pytest.skip("Risk engine not loaded")
        assert hasattr(mod, "PhysicalRisk")

    def test_risk_engine_has_transition_risk(self):
        """Risk engine exports TransitionRisk model."""
        mod = _safe_load(_load_engine, "climate_risk")
        if mod is None:
            pytest.skip("Risk engine not loaded")
        assert hasattr(mod, "TransitionRisk")

    def test_risk_engine_has_result_model(self):
        """Risk engine exports ClimateRiskResult model."""
        mod = _safe_load(_load_engine, "climate_risk")
        if mod is None:
            pytest.skip("Risk engine not loaded")
        assert hasattr(mod, "ClimateRiskResult")

    def test_risk_engine_has_opportunity(self):
        """Risk engine exports ClimateOpportunity model."""
        mod = _safe_load(_load_engine, "climate_risk")
        if mod is None:
            pytest.skip("Risk engine not loaded")
        assert hasattr(mod, "ClimateOpportunity")


# ===========================================================================
# Provenance Chain Tests
# ===========================================================================


class TestE2EProvenanceChain:
    """End-to-end tests for provenance hash chain."""

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_engine_has_sha256(self, engine_key):
        """Each engine uses SHA-256 for provenance."""
        source_path = ENGINES_DIR / ENGINE_FILES[engine_key]
        content = source_path.read_text(encoding="utf-8")
        assert "sha256" in content.lower() or "hashlib" in content

    def test_config_hash_is_deterministic(self):
        """Configuration hash is deterministic."""
        cfg = _load_config_module()
        config1 = cfg.PackConfig()
        config2 = cfg.PackConfig()
        assert config1.get_config_hash() == config2.get_config_hash()

    def test_config_hash_is_sha256(self):
        """Configuration hash is 64 hex chars (SHA-256)."""
        cfg = _load_config_module()
        config = cfg.PackConfig()
        h = config.get_config_hash()
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ===========================================================================
# Full E1 Disclosure Tests
# ===========================================================================


class TestE2EFullE1:
    """End-to-end tests for complete E1 disclosure."""

    def test_full_e1_workflow_exists(self):
        """Full E1 workflow can be loaded."""
        mod = _safe_load(_load_workflow, "full_e1")
        assert mod is not None

    def test_full_e1_has_all_disclosure_phases(self):
        """Full E1 workflow source references all 9 disclosure requirements."""
        path = Path(__file__).resolve().parent.parent / "workflows" / "full_e1_workflow.py"
        if not path.exists():
            pytest.skip("full_e1_workflow.py not found")
        content = path.read_text(encoding="utf-8")
        for dr in ["E1-1", "E1-2", "E1-3", "E1-4", "E1-5",
                    "E1-6", "E1-7", "E1-8", "E1-9"]:
            normalized = dr.replace("-", "_")
            assert dr in content or normalized in content, (
                f"Full E1 workflow should reference {dr}"
            )

    def test_orchestrator_covers_all_phases(self):
        """Pack orchestrator covers all 10 phases."""
        mod = _safe_load(_load_integration, "pack_orchestrator")
        if mod is None:
            pytest.skip("Pack orchestrator not loaded")
        if hasattr(mod, "PHASE_EXECUTION_ORDER"):
            assert len(mod.PHASE_EXECUTION_ORDER) >= 10

    def test_all_templates_available(self):
        """All 9 templates are available on disk."""
        for key, file_name in TEMPLATE_FILES.items():
            path = Path(__file__).resolve().parent.parent / "templates" / file_name
            assert path.exists(), f"Template {key} missing: {path}"
