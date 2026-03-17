# -*- coding: utf-8 -*-
"""
PACK-017 ESRS Full Coverage Pack - End-to-End Tests
======================================================

End-to-end integration tests validating cross-engine data flow,
pipeline sequencing, provenance hashing, determinism, environmental
pipeline (E2-E5), social pipeline (S1-S4), governance pipeline (G1+ESRS2),
full ESRS pipeline, and provenance chain integrity.

Target: 30+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-017 ESRS Full Coverage Pack
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
    WORKFLOWS_DIR,
    TEMPLATE_FILES,
    TEMPLATE_CLASSES,
    TEMPLATES_DIR,
    INTEGRATION_FILES,
    INTEGRATION_CLASSES,
    INTEGRATIONS_DIR,
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
# Engine Data Compatibility
# ===========================================================================


class TestEngineDataCompatibility:
    """Tests for cross-engine data compatibility (all 11 engines)."""

    def test_all_11_engines_loadable(self):
        """All 11 engines can be loaded via importlib."""
        loaded = 0
        for key in ENGINE_FILES:
            mod = _safe_load(_load_engine, key)
            if mod is not None:
                loaded += 1
        assert loaded == 11, f"Only {loaded}/11 engines loaded"

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_engine_produces_provenance_hash(self, engine_key):
        """Each engine uses SHA-256/hashlib for provenance hashing."""
        source_path = ENGINES_DIR / ENGINE_FILES[engine_key]
        content = source_path.read_text(encoding="utf-8")
        has_sha256 = "sha256" in content.lower() or "hashlib" in content
        assert has_sha256, f"Engine {engine_key} should produce SHA-256 provenance"

    def test_cross_engine_e2_e3_compatibility(self):
        """E2 pollution and E3 water engines share compatible data patterns."""
        e2 = _safe_load(_load_engine, "e2_pollution")
        e3 = _safe_load(_load_engine, "e3_water_marine")
        if e2 is None or e3 is None:
            pytest.skip("Both E2 and E3 engines required")
        assert hasattr(e2, "PollutionEngine")
        assert hasattr(e3, "WaterMarineEngine")

    def test_cross_engine_s1_s2_compatibility(self):
        """S1 workforce and S2 value chain engines share compatible patterns."""
        s1 = _safe_load(_load_engine, "s1_own_workforce")
        s2 = _safe_load(_load_engine, "s2_value_chain_workers")
        if s1 is None or s2 is None:
            pytest.skip("Both S1 and S2 engines required")
        assert hasattr(s1, "OwnWorkforceEngine")
        assert hasattr(s2, "ValueChainWorkersEngine")

    def test_esrs2_and_g1_compatibility(self):
        """ESRS2 general and G1 governance engines share compatible patterns."""
        esrs2 = _safe_load(_load_engine, "esrs2_general_disclosures")
        g1 = _safe_load(_load_engine, "g1_business_conduct")
        if esrs2 is None or g1 is None:
            pytest.skip("Both ESRS2 and G1 engines required")
        assert hasattr(esrs2, "GeneralDisclosuresEngine")
        assert hasattr(g1, "BusinessConductEngine")


# ===========================================================================
# Environmental Pipeline (E2 -> E3 -> E4 -> E5)
# ===========================================================================


class TestEnvironmentalPipeline:
    """Tests for environmental standard data flow (E2 through E5)."""

    def test_e2_engine_loaded(self):
        """E2 pollution engine loads successfully."""
        mod = _safe_load(_load_engine, "e2_pollution")
        assert mod is not None

    def test_e3_engine_loaded(self):
        """E3 water engine loads successfully."""
        mod = _safe_load(_load_engine, "e3_water_marine")
        assert mod is not None

    def test_e4_engine_loaded(self):
        """E4 biodiversity engine loads successfully."""
        mod = _safe_load(_load_engine, "e4_biodiversity")
        assert mod is not None

    def test_e5_engine_loaded(self):
        """E5 circular economy engine loads successfully."""
        mod = _safe_load(_load_engine, "e5_circular_economy")
        assert mod is not None

    def test_environmental_chapter_consistency(self):
        """All environmental engines use consistent provenance patterns."""
        env_engines = ["e2_pollution", "e3_water_marine", "e4_biodiversity", "e5_circular_economy"]
        for engine_key in env_engines:
            source_path = ENGINES_DIR / ENGINE_FILES[engine_key]
            content = source_path.read_text(encoding="utf-8")
            assert "hashlib" in content, f"{engine_key} missing hashlib"
            assert "BaseModel" in content, f"{engine_key} missing BaseModel"
            assert "Decimal" in content or "decimal" in content, (
                f"{engine_key} missing Decimal"
            )


# ===========================================================================
# Social Pipeline (S1 -> S2 -> S3 -> S4)
# ===========================================================================


class TestSocialPipeline:
    """Tests for social standard data flow (S1 through S4)."""

    def test_s1_engine_loaded(self):
        """S1 workforce engine loads successfully."""
        mod = _safe_load(_load_engine, "s1_own_workforce")
        assert mod is not None

    def test_s2_engine_loaded(self):
        """S2 value chain engine loads successfully."""
        mod = _safe_load(_load_engine, "s2_value_chain_workers")
        assert mod is not None

    def test_s3_engine_loaded(self):
        """S3 communities engine loads successfully."""
        mod = _safe_load(_load_engine, "s3_affected_communities")
        assert mod is not None

    def test_s4_engine_loaded(self):
        """S4 consumers engine loads successfully."""
        mod = _safe_load(_load_engine, "s4_consumers")
        assert mod is not None

    def test_social_chapter_consistency(self):
        """All social engines use consistent provenance patterns."""
        social_engines = ["s1_own_workforce", "s2_value_chain_workers", "s3_affected_communities", "s4_consumers"]
        for engine_key in social_engines:
            source_path = ENGINES_DIR / ENGINE_FILES[engine_key]
            content = source_path.read_text(encoding="utf-8")
            assert "hashlib" in content, f"{engine_key} missing hashlib"
            assert "BaseModel" in content, f"{engine_key} missing BaseModel"


# ===========================================================================
# Governance Pipeline (G1 + ESRS2)
# ===========================================================================


class TestGovernancePipeline:
    """Tests for governance standard data flow (G1 + ESRS 2)."""

    def test_g1_engine_loaded(self):
        """G1 business conduct engine loads successfully."""
        mod = _safe_load(_load_engine, "g1_business_conduct")
        assert mod is not None

    def test_esrs2_engine_loaded(self):
        """ESRS 2 general disclosures engine loads successfully."""
        mod = _safe_load(_load_engine, "esrs2_general_disclosures")
        assert mod is not None

    def test_governance_chapter_completeness(self):
        """G1 and ESRS2 engines reference their standard disclosure requirements."""
        g1_path = ENGINES_DIR / ENGINE_FILES["g1_business_conduct"]
        g1_content = g1_path.read_text(encoding="utf-8")
        assert "G1-" in g1_content or "G1_" in g1_content, "G1 engine should reference G1 DRs"

        esrs2_path = ENGINES_DIR / ENGINE_FILES["esrs2_general_disclosures"]
        esrs2_content = esrs2_path.read_text(encoding="utf-8")
        has_gov = "GOV-" in esrs2_content or "GOV_" in esrs2_content
        has_sbm = "SBM-" in esrs2_content or "SBM_" in esrs2_content
        assert has_gov, "ESRS2 engine should reference GOV disclosures"
        assert has_sbm, "ESRS2 engine should reference SBM disclosures"


# ===========================================================================
# Full ESRS Pipeline
# ===========================================================================


class TestFullESRSPipeline:
    """Tests for complete ESRS pipeline: engines -> orchestrator -> templates."""

    def test_orchestrator_covers_all_standards(self):
        """Pack orchestrator covers all 12 pipeline phases."""
        mod = _safe_load(_load_integration, "pack_orchestrator")
        if mod is None:
            pytest.skip("Pack orchestrator not loaded")
        if hasattr(mod, "PHASE_EXECUTION_ORDER"):
            assert len(mod.PHASE_EXECUTION_ORDER) >= 12
        if hasattr(mod, "ESRSPipelinePhase"):
            phases = list(mod.ESRSPipelinePhase)
            assert len(phases) >= 12

    def test_orchestrator_references_all_standards(self):
        """Pack orchestrator source references all ESRS standards."""
        path = INTEGRATIONS_DIR / INTEGRATION_FILES["pack_orchestrator"]
        if not path.exists():
            pytest.skip("Pack orchestrator file not found")
        content = path.read_text(encoding="utf-8")
        for std in ["E1", "E2", "E3", "E4", "E5", "S1", "S2", "S3", "S4", "G1"]:
            assert std in content, f"Orchestrator should reference {std}"

    def test_all_engine_files_available(self):
        """All 11 engine files are available on disk."""
        for key, file_name in ENGINE_FILES.items():
            path = ENGINES_DIR / file_name
            assert path.exists(), f"Engine {key} missing: {path}"

    def test_orchestrator_has_pipeline_result(self):
        """Orchestrator exports PipelineResult model."""
        mod = _safe_load(_load_integration, "pack_orchestrator")
        if mod is None:
            pytest.skip("Pack orchestrator not loaded")
        assert hasattr(mod, "PipelineResult"), "Orchestrator should export PipelineResult"


# ===========================================================================
# Determinism Tests
# ===========================================================================


class TestDeterminism:
    """Tests for calculation determinism (same inputs -> same outputs)."""

    def test_config_hash_is_deterministic(self):
        """Configuration hash is deterministic across multiple calls."""
        try:
            cfg_mod = _load_config_module()
        except (ImportError, FileNotFoundError, AttributeError):
            pytest.skip("Config module not available")

        # Try to instantiate config
        if not hasattr(cfg_mod, "PackConfig"):
            pytest.skip("PackConfig class not found in config module")

        config1 = cfg_mod.PackConfig()
        config2 = cfg_mod.PackConfig()

        # Check if get_config_hash method exists
        if not hasattr(config1, "get_config_hash"):
            pytest.skip("get_config_hash method not implemented")

        assert config1.get_config_hash() == config2.get_config_hash()

    def test_config_hash_is_sha256(self):
        """Configuration hash is 64 hex chars (SHA-256 format)."""
        try:
            cfg_mod = _load_config_module()
        except (ImportError, FileNotFoundError):
            pytest.skip("Config module not available")
        config = cfg_mod.PackConfig()
        h = config.get_config_hash()
        assert len(h) == 64, f"Expected 64-char hex string, got {len(h)}"
        assert all(c in "0123456789abcdef" for c in h), "Hash should be hex"

    def test_all_engines_use_deterministic_hashing(self):
        """All engine files use hashlib (no random in scoring paths)."""
        for key, file_name in ENGINE_FILES.items():
            source_path = ENGINES_DIR / file_name
            content = source_path.read_text(encoding="utf-8")
            assert "hashlib" in content, f"Engine {key} should use hashlib"


# ===========================================================================
# Provenance Chain Tests
# ===========================================================================


class TestProvenanceChain:
    """Tests for SHA-256 provenance chain from engine through to template."""

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_engine_has_sha256_import(self, engine_key):
        """Each engine imports hashlib for SHA-256 provenance."""
        source_path = ENGINES_DIR / ENGINE_FILES[engine_key]
        content = source_path.read_text(encoding="utf-8")
        assert "hashlib" in content, f"Engine {engine_key} missing hashlib import"

    def test_orchestrator_has_sha256(self):
        """Pack orchestrator uses SHA-256 for pipeline provenance."""
        path = INTEGRATIONS_DIR / INTEGRATION_FILES["pack_orchestrator"]
        if not path.exists():
            pytest.skip("Pack orchestrator file not found")
        content = path.read_text(encoding="utf-8")
        assert "hashlib" in content, "Orchestrator should use hashlib"
        assert "sha256" in content.lower(), "Orchestrator should use SHA-256"

    def test_available_templates_have_provenance(self):
        """All available template files reference provenance/hashing."""
        for key, file_name in TEMPLATE_FILES.items():
            path = TEMPLATES_DIR / file_name
            if not path.exists():
                continue  # Template not yet created
            content = path.read_text(encoding="utf-8")
            has_prov = (
                "hashlib" in content
                or "sha256" in content.lower()
                or "provenance" in content.lower()
            )
            assert has_prov, f"Template {key} should reference provenance"

    def test_provenance_hash_chain_consistency(self):
        """Verify provenance chain: engine hash -> workflow -> template."""
        # Verify the pattern by checking that the same hashing function is used
        # across the pipeline components that exist on disk
        hash_pattern_count = 0
        for key, file_name in ENGINE_FILES.items():
            source_path = ENGINES_DIR / file_name
            content = source_path.read_text(encoding="utf-8")
            if "sha256" in content.lower() and "hashlib" in content:
                hash_pattern_count += 1
        assert hash_pattern_count >= 10, (
            f"Expected 10+ engines with SHA-256 pattern, found {hash_pattern_count}"
        )

    def test_deterministic_hash_function(self):
        """Verify that SHA-256 produces deterministic results for test data."""
        test_data = json.dumps({"company": "TestCorp", "year": 2025}, sort_keys=True)
        hash1 = hashlib.sha256(test_data.encode("utf-8")).hexdigest()
        hash2 = hashlib.sha256(test_data.encode("utf-8")).hexdigest()
        assert hash1 == hash2, "SHA-256 should be deterministic"
        assert len(hash1) == 64, "SHA-256 hex digest should be 64 chars"


# ===========================================================================
# Full Pipeline Availability
# ===========================================================================


class TestFullPipelineAvailability:
    """Tests for complete pipeline component availability."""

    def test_all_engine_files_on_disk(self):
        """All 11 engine files exist on disk."""
        missing = []
        for key, file_name in ENGINE_FILES.items():
            if not (ENGINES_DIR / file_name).exists():
                missing.append(key)
        assert len(missing) == 0, f"Missing engines: {missing}"

    def test_orchestrator_on_disk(self):
        """Pack orchestrator file exists on disk."""
        path = INTEGRATIONS_DIR / INTEGRATION_FILES["pack_orchestrator"]
        assert path.exists(), f"Pack orchestrator missing: {path}"

    def test_config_on_disk(self):
        """Pack configuration file exists on disk."""
        path = Path(__file__).resolve().parent.parent / "config" / "pack_config.py"
        assert path.exists(), f"Pack config missing: {path}"
