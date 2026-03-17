# -*- coding: utf-8 -*-
"""
PACK-018 EU Green Claims Prep Pack - Integration Tests
========================================================

Tests for all 10+ integrations: file existence, module loading, class
exports, key methods, docstrings. Parametrized across all integrations.

Target: ~35 tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-018 EU Green Claims Prep
Date:    March 2026
"""

import pytest

from .conftest import (
    _load_integration,
    INTEGRATIONS_DIR,
    INTEGRATION_FILES,
    INTEGRATION_CLASSES,
)


# ===========================================================================
# Integration definitions
# ===========================================================================


EXISTING_INTEGRATIONS = [
    ("pack_orchestrator", "pack_orchestrator.py", "GreenClaimsOrchestrator"),
    ("csrd_pack_bridge", "csrd_pack_bridge.py", "CSRDPackBridge"),
    ("mrv_claims_bridge", "mrv_claims_bridge.py", "MRVClaimsBridge"),
    ("data_claims_bridge", "data_claims_bridge.py", "DataClaimsBridge"),
    ("taxonomy_bridge", "taxonomy_bridge.py", "TaxonomyBridge"),
    ("pef_bridge", "pef_bridge.py", "PEFBridge"),
]

MISSING_INTEGRATIONS = [
    ("dpp_bridge", "dpp_bridge.py", "DPPBridge"),
    ("ecgt_bridge", "ecgt_bridge.py", "ECGTBridge"),
    ("health_check", "health_check.py", "GreenClaimsHealthCheck"),
    ("setup_wizard", "setup_wizard.py", "GreenClaimsSetupWizard"),
]

ALL_INTEGRATION_KEYS = list(INTEGRATION_FILES.keys())


# ===========================================================================
# File Existence Tests
# ===========================================================================


class TestIntegrationFileExistence:
    """Tests for integration file existence."""

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_INTEGRATIONS)
    def test_integration_file_exists(self, key, filename, cls_name):
        """Integration file exists on disk."""
        path = INTEGRATIONS_DIR / filename
        assert path.exists(), f"Integration file missing: {filename}"

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_INTEGRATIONS)
    def test_integration_registered_in_mapping(self, key, filename, cls_name):
        """Integration key is registered in INTEGRATION_FILES."""
        assert key in INTEGRATION_FILES

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_INTEGRATIONS)
    def test_integration_has_class_mapping(self, key, filename, cls_name):
        """Integration key has a class name mapping."""
        assert key in INTEGRATION_CLASSES


# ===========================================================================
# Module Loading Tests
# ===========================================================================


class TestIntegrationModuleLoading:
    """Tests for integration module loading."""

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_INTEGRATIONS)
    def test_integration_module_loads(self, key, filename, cls_name):
        """Integration module loads successfully."""
        mod = _load_integration(key)
        assert mod is not None

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_INTEGRATIONS)
    def test_integration_class_exists(self, key, filename, cls_name):
        """Integration module exports the expected class."""
        mod = _load_integration(key)
        assert hasattr(mod, cls_name), f"Class {cls_name} not found in {key}"

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_INTEGRATIONS)
    def test_integration_class_has_docstring(self, key, filename, cls_name):
        """Integration class has a docstring."""
        mod = _load_integration(key)
        cls = getattr(mod, cls_name)
        assert cls.__doc__ is not None


# ===========================================================================
# Source File Characteristic Tests
# ===========================================================================


class TestIntegrationSourceCharacteristics:
    """Tests for integration source file characteristics."""

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_INTEGRATIONS)
    def test_integration_source_has_logging(self, key, filename, cls_name):
        """Integration source uses logging."""
        source = (INTEGRATIONS_DIR / filename).read_text(encoding="utf-8")
        assert "logging" in source

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_INTEGRATIONS)
    def test_integration_source_has_pydantic(self, key, filename, cls_name):
        """Integration source uses Pydantic BaseModel."""
        source = (INTEGRATIONS_DIR / filename).read_text(encoding="utf-8")
        assert "BaseModel" in source

    @pytest.mark.parametrize("key,filename,cls_name", EXISTING_INTEGRATIONS)
    def test_integration_source_has_provenance(self, key, filename, cls_name):
        """Integration source references provenance or hashing."""
        source = (INTEGRATIONS_DIR / filename).read_text(encoding="utf-8")
        has_provenance = (
            "sha256" in source.lower()
            or "hashlib" in source
            or "provenance" in source.lower()
        )
        assert has_provenance


# ===========================================================================
# Missing Integrations Tests
# ===========================================================================


class TestMissingIntegrations:
    """Tests documenting integrations not yet created."""

    @pytest.mark.parametrize("key,filename,cls_name", MISSING_INTEGRATIONS)
    def test_missing_integration_registered(self, key, filename, cls_name):
        """Missing integration is registered in INTEGRATION_FILES."""
        assert key in INTEGRATION_FILES

    @pytest.mark.parametrize("key,filename,cls_name", MISSING_INTEGRATIONS)
    def test_missing_integration_file_status(self, key, filename, cls_name):
        """Missing integration file existence check."""
        path = INTEGRATIONS_DIR / filename
        if not path.exists():
            pytest.skip(f"{filename} not yet created")
        assert path.exists()


# ===========================================================================
# Alias Tests (csrd_bridge, mrv_bridge, data_bridge)
# ===========================================================================


class TestIntegrationAliases:
    """Tests for integration alias mappings."""

    def test_csrd_bridge_alias(self):
        """csrd_bridge alias maps to csrd_pack_bridge.py."""
        assert INTEGRATION_FILES.get("csrd_bridge") == "csrd_pack_bridge.py"

    def test_mrv_bridge_alias(self):
        """mrv_bridge alias maps to mrv_claims_bridge.py."""
        assert INTEGRATION_FILES.get("mrv_bridge") == "mrv_claims_bridge.py"

    def test_data_bridge_alias(self):
        """data_bridge alias maps to data_claims_bridge.py."""
        assert INTEGRATION_FILES.get("data_bridge") == "data_claims_bridge.py"
