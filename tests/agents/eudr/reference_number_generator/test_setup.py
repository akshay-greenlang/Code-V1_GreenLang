# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-EUDR-038 Reference Number Generator -- setup.py & __init__.py

Tests the service facade, module metadata, lazy imports, convenience
functions, and package-level exports. 20+ tests.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest


# ====================================================================
# Test: Module Metadata
# ====================================================================


class TestModuleMetadata:
    """Test __init__.py module-level metadata and convenience functions."""

    def test_version(self):
        from greenlang.agents.eudr.reference_number_generator import __version__
        assert __version__ == "1.0.0"

    def test_agent_id(self):
        from greenlang.agents.eudr.reference_number_generator import __agent_id__
        assert __agent_id__ == "GL-EUDR-RNG-038"

    def test_get_version(self):
        from greenlang.agents.eudr.reference_number_generator import get_version
        assert get_version() == "1.0.0"

    def test_get_agent_info(self):
        from greenlang.agents.eudr.reference_number_generator import get_agent_info
        info = get_agent_info()
        assert isinstance(info, dict)
        assert info["agent_id"] == "GL-EUDR-RNG-038"
        assert info["version"] == "1.0.0"
        assert info["name"] == "Reference Number Generator"
        assert info["prd"] == "PRD-AGENT-EUDR-038"
        assert info["regulation"] == "EU 2023/1115 (EUDR)"

    def test_agent_info_articles(self):
        from greenlang.agents.eudr.reference_number_generator import get_agent_info
        info = get_agent_info()
        assert "4" in info["articles"]
        assert "9" in info["articles"]
        assert "31" in info["articles"]
        assert "33" in info["articles"]

    def test_agent_info_engine_count(self):
        from greenlang.agents.eudr.reference_number_generator import get_agent_info
        info = get_agent_info()
        assert info["engine_count"] == 7

    def test_agent_info_enum_count(self):
        from greenlang.agents.eudr.reference_number_generator import get_agent_info
        info = get_agent_info()
        assert info["enum_count"] == 12

    def test_agent_info_model_count(self):
        from greenlang.agents.eudr.reference_number_generator import get_agent_info
        info = get_agent_info()
        assert info["core_model_count"] == 15

    def test_agent_info_metrics_count(self):
        from greenlang.agents.eudr.reference_number_generator import get_agent_info
        info = get_agent_info()
        assert info["metrics_count"] == 18

    def test_agent_info_member_states(self):
        from greenlang.agents.eudr.reference_number_generator import get_agent_info
        info = get_agent_info()
        assert info["member_states_supported"] == 27

    def test_agent_info_prefixes(self):
        from greenlang.agents.eudr.reference_number_generator import get_agent_info
        info = get_agent_info()
        assert info["db_prefix"] == "gl_eudr_rng_"
        assert info["metrics_prefix"] == "gl_eudr_rng_"
        assert info["env_prefix"] == "GL_EUDR_RNG_"

    def test_agent_info_enforcement_dates(self):
        from greenlang.agents.eudr.reference_number_generator import get_agent_info
        info = get_agent_info()
        assert info["enforcement_date_large"] == "2025-12-30"
        assert info["enforcement_date_sme"] == "2026-06-30"

    def test_agent_info_engines_list(self):
        from greenlang.agents.eudr.reference_number_generator import get_agent_info
        info = get_agent_info()
        expected_engines = [
            "NumberGenerator",
            "FormatValidator",
            "SequenceManager",
            "BatchProcessor",
            "CollisionDetector",
            "LifecycleManager",
            "VerificationService",
        ]
        assert info["engines"] == expected_engines


# ====================================================================
# Test: Lazy Imports - Configuration
# ====================================================================


class TestLazyImportsConfig:
    """Test lazy import of configuration classes."""

    def test_import_config_class(self):
        from greenlang.agents.eudr.reference_number_generator import (
            ReferenceNumberGeneratorConfig,
        )
        assert ReferenceNumberGeneratorConfig is not None

    def test_import_get_config(self):
        from greenlang.agents.eudr.reference_number_generator import get_config
        cfg = get_config()
        assert cfg is not None
        assert cfg.reference_prefix == "EUDR"

    def test_import_reset_config(self):
        from greenlang.agents.eudr.reference_number_generator import reset_config
        reset_config()  # Should not raise


# ====================================================================
# Test: Lazy Imports - Models
# ====================================================================


class TestLazyImportsModels:
    """Test lazy import of model classes and enums."""

    def test_import_member_state_code(self):
        from greenlang.agents.eudr.reference_number_generator import MemberStateCode
        assert len(MemberStateCode) == 27

    def test_import_reference_number_status(self):
        from greenlang.agents.eudr.reference_number_generator.models import (
            ReferenceNumberStatus,
        )
        assert len(ReferenceNumberStatus) == 7

    def test_import_checksum_algorithm(self):
        from greenlang.agents.eudr.reference_number_generator import ChecksumAlgorithm
        assert len(ChecksumAlgorithm) == 4

    def test_import_validation_result(self):
        from greenlang.agents.eudr.reference_number_generator.models import (
            ValidationResult,
        )
        assert len(ValidationResult) == 9

    def test_import_constants(self):
        from greenlang.agents.eudr.reference_number_generator import (
            AGENT_ID,
            AGENT_VERSION,
        )
        # EU_MEMBER_STATES is in config.py, import directly
        from greenlang.agents.eudr.reference_number_generator.config import (
            EU_MEMBER_STATES,
        )
        assert AGENT_ID == "GL-EUDR-RNG-038"
        assert AGENT_VERSION == "1.0.0"
        assert len(EU_MEMBER_STATES) == 27


# ====================================================================
# Test: Lazy Imports - Provenance
# ====================================================================


class TestLazyImportsProvenance:
    """Test lazy import of provenance classes."""

    def test_import_provenance_tracker(self):
        from greenlang.agents.eudr.reference_number_generator import ProvenanceTracker
        tracker = ProvenanceTracker()
        assert tracker is not None

    def test_import_genesis_hash(self):
        from greenlang.agents.eudr.reference_number_generator import GENESIS_HASH
        assert GENESIS_HASH == "0" * 64


# ====================================================================
# Test: Lazy Imports - Engines
# ====================================================================


class TestLazyImportsEngines:
    """Test lazy import of engine classes."""

    def test_import_number_generator(self):
        from greenlang.agents.eudr.reference_number_generator import NumberGenerator
        engine = NumberGenerator()
        assert engine is not None

    def test_import_format_validator(self):
        from greenlang.agents.eudr.reference_number_generator import FormatValidator
        engine = FormatValidator()
        assert engine is not None

    def test_import_sequence_manager(self):
        from greenlang.agents.eudr.reference_number_generator import SequenceManager
        engine = SequenceManager()
        assert engine is not None


# ====================================================================
# Test: Invalid Lazy Import
# ====================================================================


class TestInvalidLazyImport:
    """Test that invalid attribute access raises AttributeError."""

    def test_invalid_attribute_raises(self):
        import greenlang.agents.eudr.reference_number_generator as pkg
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = pkg.NonExistentClass

    def test_invalid_attribute_message(self):
        import greenlang.agents.eudr.reference_number_generator as pkg
        with pytest.raises(AttributeError) as exc_info:
            _ = pkg.ThisDoesNotExist
        assert "ThisDoesNotExist" in str(exc_info.value)


# ====================================================================
# Test: __all__ Exports
# ====================================================================


class TestAllExports:
    """Test __all__ export list completeness."""

    def test_all_is_list(self):
        from greenlang.agents.eudr.reference_number_generator import __all__
        assert isinstance(__all__, list)

    def test_all_has_metadata(self):
        from greenlang.agents.eudr.reference_number_generator import __all__
        assert "__version__" in __all__
        assert "__agent_id__" in __all__

    def test_all_has_config(self):
        from greenlang.agents.eudr.reference_number_generator import __all__
        assert "ReferenceNumberGeneratorConfig" in __all__
        assert "get_config" in __all__
        assert "reset_config" in __all__

    def test_all_has_engines(self):
        from greenlang.agents.eudr.reference_number_generator import __all__
        assert "NumberGenerator" in __all__
        assert "FormatValidator" in __all__
        assert "SequenceManager" in __all__
        assert "BatchProcessor" in __all__
        assert "CollisionDetector" in __all__
        assert "LifecycleManager" in __all__
        assert "VerificationService" in __all__

    def test_all_has_provenance(self):
        from greenlang.agents.eudr.reference_number_generator import __all__
        assert "ProvenanceTracker" in __all__
        assert "GENESIS_HASH" in __all__

    def test_all_count(self):
        from greenlang.agents.eudr.reference_number_generator import __all__
        assert len(__all__) >= 60
