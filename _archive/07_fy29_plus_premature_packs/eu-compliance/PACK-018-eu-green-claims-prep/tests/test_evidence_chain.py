# -*- coding: utf-8 -*-
"""
PACK-018 EU Green Claims Prep Pack - Evidence Chain Engine Tests
================================================================

Unit tests for EvidenceChainEngine covering enums (EvidenceStatus, EvidenceType,
ChainVerificationStatus, DocumentTier), models (DocumentRecord, EvidenceChain,
ChainValidationResult), constants, and engine methods (build_evidence_chain,
validate_chain_completeness, check_document_validity, calculate_chain_strength).

Target: ~50 tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-018 EU Green Claims Prep
Date:    March 2026
"""

from decimal import Decimal

import pytest

from .conftest import _load_engine, ENGINES_DIR


# ---------------------------------------------------------------------------
# Module-scoped engine loading
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mod():
    """Load the Evidence Chain engine module."""
    return _load_engine("evidence_chain")


@pytest.fixture
def engine(mod):
    """Create a fresh EvidenceChainEngine instance."""
    return mod.EvidenceChainEngine()


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestEvidenceChainEnums:
    """Tests for Evidence Chain engine enums."""

    def test_evidence_status_count(self, mod):
        """EvidenceStatus has exactly 5 values."""
        assert len(mod.EvidenceStatus) == 5

    def test_evidence_status_values(self, mod):
        """EvidenceStatus contains expected statuses."""
        values = {m.value for m in mod.EvidenceStatus}
        expected = {"pending", "collected", "validated", "expired", "rejected"}
        assert values == expected

    def test_evidence_type_count(self, mod):
        """EvidenceType has exactly 10 values."""
        assert len(mod.EvidenceType) == 10

    def test_evidence_type_certification(self, mod):
        """EvidenceType includes CERTIFICATION."""
        assert mod.EvidenceType.CERTIFICATION.value == "certification"

    def test_evidence_type_lca_study(self, mod):
        """EvidenceType includes LCA_STUDY."""
        assert mod.EvidenceType.LCA_STUDY.value == "lca_study"

    def test_evidence_type_supplier_declaration(self, mod):
        """EvidenceType includes SUPPLIER_DECLARATION."""
        assert mod.EvidenceType.SUPPLIER_DECLARATION.value == "supplier_declaration"

    def test_evidence_type_offset_registry(self, mod):
        """EvidenceType includes OFFSET_REGISTRY."""
        assert mod.EvidenceType.OFFSET_REGISTRY.value == "offset_registry"

    def test_evidence_type_laboratory_result(self, mod):
        """EvidenceType includes LABORATORY_RESULT."""
        assert mod.EvidenceType.LABORATORY_RESULT.value == "laboratory_result"

    def test_evidence_type_monitoring_data(self, mod):
        """EvidenceType includes MONITORING_DATA."""
        assert mod.EvidenceType.MONITORING_DATA.value == "monitoring_data"

    def test_chain_verification_status_count(self, mod):
        """ChainVerificationStatus has exactly 4 values."""
        assert len(mod.ChainVerificationStatus) == 4

    def test_chain_verification_status_values(self, mod):
        """ChainVerificationStatus contains expected values."""
        values = {m.value for m in mod.ChainVerificationStatus}
        expected = {
            "verified", "partially_verified",
            "unverified", "verification_failed",
        }
        assert values == expected

    def test_document_tier_count(self, mod):
        """DocumentTier has exactly 3 values."""
        assert len(mod.DocumentTier) == 3

    def test_document_tier_values(self, mod):
        """DocumentTier contains primary, secondary, tertiary."""
        values = {m.value for m in mod.DocumentTier}
        assert values == {"primary", "secondary", "tertiary"}


# ===========================================================================
# Constants Tests
# ===========================================================================


class TestEvidenceChainConstants:
    """Tests for Evidence Chain engine constants."""

    def test_core_evidence_types_exist(self, mod):
        """CORE_EVIDENCE_TYPES list exists with at least 2 entries."""
        assert len(mod.CORE_EVIDENCE_TYPES) >= 2

    def test_evidence_type_weights_exist(self, mod):
        """EVIDENCE_TYPE_WEIGHTS dict exists with 10 entries."""
        assert len(mod.EVIDENCE_TYPE_WEIGHTS) == 10

    def test_evidence_type_weights_values_are_decimal(self, mod):
        """EVIDENCE_TYPE_WEIGHTS values are all Decimal."""
        for val in mod.EVIDENCE_TYPE_WEIGHTS.values():
            assert isinstance(val, Decimal)

    def test_chain_strength_weights_exist(self, mod):
        """CHAIN_STRENGTH_WEIGHTS dict exists with 5 entries."""
        assert len(mod.CHAIN_STRENGTH_WEIGHTS) == 5

    def test_chain_strength_weights_keys(self, mod):
        """CHAIN_STRENGTH_WEIGHTS has expected dimension keys."""
        expected_keys = {
            "document_quality", "chain_completeness", "temporal_validity",
            "verification_depth", "traceability",
        }
        assert set(mod.CHAIN_STRENGTH_WEIGHTS.keys()) == expected_keys

    def test_validity_warning_days(self, mod):
        """VALIDITY_WARNING_DAYS is 90."""
        assert mod.VALIDITY_WARNING_DAYS == 90


# ===========================================================================
# Model Tests
# ===========================================================================


class TestDocumentRecordModel:
    """Tests for DocumentRecord Pydantic model."""

    def test_create_valid_document(self, mod):
        """Create a valid DocumentRecord with required fields."""
        doc = mod.DocumentRecord(
            title="Test Certificate",
            evidence_type=mod.EvidenceType.CERTIFICATION,
        )
        assert doc.title == "Test Certificate"
        assert doc.evidence_type == mod.EvidenceType.CERTIFICATION

    def test_document_has_auto_id(self, mod):
        """DocumentRecord auto-generates doc_id."""
        doc = mod.DocumentRecord(
            title="Test",
            evidence_type=mod.EvidenceType.TEST_REPORT,
        )
        assert doc.doc_id is not None
        assert len(doc.doc_id) > 0

    def test_document_default_status(self, mod):
        """DocumentRecord defaults status to PENDING."""
        doc = mod.DocumentRecord(
            title="Test",
            evidence_type=mod.EvidenceType.MEASUREMENT,
        )
        assert doc.status == mod.EvidenceStatus.PENDING

    def test_document_default_tier(self, mod):
        """DocumentRecord defaults tier to SECONDARY."""
        doc = mod.DocumentRecord(
            title="Test",
            evidence_type=mod.EvidenceType.MEASUREMENT,
        )
        assert doc.tier == mod.DocumentTier.SECONDARY

    def test_document_empty_title_raises_error(self, mod):
        """DocumentRecord rejects empty title."""
        with pytest.raises(Exception):
            mod.DocumentRecord(
                title="   ",
                evidence_type=mod.EvidenceType.CERTIFICATION,
            )

    def test_document_sha256_hash_field(self, mod):
        """DocumentRecord has sha256_hash field."""
        doc = mod.DocumentRecord(
            title="Test",
            evidence_type=mod.EvidenceType.AUDIT_REPORT,
        )
        assert hasattr(doc, "sha256_hash")


class TestEvidenceChainModel:
    """Tests for EvidenceChain Pydantic model."""

    def test_create_evidence_chain(self, mod):
        """Create an EvidenceChain with required fields."""
        chain = mod.EvidenceChain(claim_id="CLM-001")
        assert chain.claim_id == "CLM-001"
        assert chain.documents == []

    def test_chain_default_verification_status(self, mod):
        """EvidenceChain defaults verification_status to UNVERIFIED."""
        chain = mod.EvidenceChain(claim_id="CLM-001")
        assert chain.verification_status == mod.ChainVerificationStatus.UNVERIFIED

    def test_chain_has_provenance_hash(self, mod):
        """EvidenceChain has provenance_hash field."""
        chain = mod.EvidenceChain(claim_id="CLM-001")
        assert hasattr(chain, "provenance_hash")

    def test_chain_strength_score_default(self, mod):
        """EvidenceChain defaults strength_score to 0.00."""
        chain = mod.EvidenceChain(claim_id="CLM-001")
        assert chain.strength_score == Decimal("0.00")


# ===========================================================================
# Engine Method Tests
# ===========================================================================


class TestEvidenceChainEngine:
    """Tests for EvidenceChainEngine methods."""

    def test_engine_instantiation(self, mod):
        """Engine can be instantiated."""
        engine = mod.EvidenceChainEngine()
        assert engine is not None

    def test_engine_has_build_evidence_chain(self, engine):
        """Engine has build_evidence_chain method."""
        assert hasattr(engine, "build_evidence_chain")
        assert callable(engine.build_evidence_chain)

    def test_engine_has_validate_chain_completeness(self, engine):
        """Engine has validate_chain_completeness method."""
        assert hasattr(engine, "validate_chain_completeness")
        assert callable(engine.validate_chain_completeness)

    def test_engine_has_check_document_validity(self, engine):
        """Engine has check_document_validity method."""
        assert hasattr(engine, "check_document_validity")
        assert callable(engine.check_document_validity)

    def test_engine_has_calculate_chain_strength(self, engine):
        """Engine has calculate_chain_strength method."""
        assert hasattr(engine, "calculate_chain_strength")
        assert callable(engine.calculate_chain_strength)

    def test_engine_has_docstring(self, mod):
        """EvidenceChainEngine class has a docstring."""
        assert mod.EvidenceChainEngine.__doc__ is not None


# ===========================================================================
# Provenance and Source Checks
# ===========================================================================


class TestEvidenceChainProvenance:
    """Tests for source file characteristics and provenance."""

    def test_engine_source_has_sha256(self):
        """Engine source uses SHA-256 for provenance."""
        source = (ENGINES_DIR / "evidence_chain_engine.py").read_text(
            encoding="utf-8"
        )
        assert "sha256" in source.lower() or "hashlib" in source

    def test_engine_source_has_decimal(self):
        """Engine source uses Decimal arithmetic."""
        source = (ENGINES_DIR / "evidence_chain_engine.py").read_text(
            encoding="utf-8"
        )
        assert "Decimal" in source

    def test_engine_source_has_basemodel(self):
        """Engine source uses Pydantic BaseModel."""
        source = (ENGINES_DIR / "evidence_chain_engine.py").read_text(
            encoding="utf-8"
        )
        assert "BaseModel" in source

    def test_engine_source_has_logging(self):
        """Engine source uses logging."""
        source = (ENGINES_DIR / "evidence_chain_engine.py").read_text(
            encoding="utf-8"
        )
        assert "logging" in source

    def test_engine_source_references_article10(self):
        """Engine source references Article 10 of Green Claims Directive."""
        source = (ENGINES_DIR / "evidence_chain_engine.py").read_text(
            encoding="utf-8"
        )
        assert "Article 10" in source

    def test_engine_file_exists(self):
        """Engine source file exists on disk."""
        assert (ENGINES_DIR / "evidence_chain_engine.py").exists()
