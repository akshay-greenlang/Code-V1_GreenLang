# -*- coding: utf-8 -*-
"""
PACK-018 EU Green Claims Prep Pack - Trader Obligation Engine Tests
====================================================================

Unit tests for TraderObligationEngine covering enums (ArticleReference,
ComplianceStatus, ClaimLifecycleStage), models (ObligationItem,
ClaimLifecycleRecord, TraderObligationResult), constants
(ARTICLE_3_CHECKLIST, RECORD_RETENTION_YEARS, ARTICLE_WEIGHTS,
VALID_TRANSITIONS, REMEDIATION_PRIORITY), and engine methods
(assess_article3-8, calculate_overall_compliance,
generate_remediation_timeline, track_claim_lifecycle).

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
    """Load the Trader Obligation engine module."""
    return _load_engine("trader_obligation")


@pytest.fixture
def engine(mod):
    """Create a fresh TraderObligationEngine instance."""
    return mod.TraderObligationEngine()


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestTraderObligationEnums:
    """Tests for Trader Obligation engine enums."""

    # -- ArticleReference --

    def test_article_reference_count(self, mod):
        """ArticleReference has exactly 6 values."""
        assert len(mod.ArticleReference) == 6

    def test_article_reference_article3(self, mod):
        """ArticleReference includes ARTICLE_3."""
        assert mod.ArticleReference.ARTICLE_3.value == "article_3"

    def test_article_reference_article4(self, mod):
        """ArticleReference includes ARTICLE_4."""
        assert mod.ArticleReference.ARTICLE_4.value == "article_4"

    def test_article_reference_article5(self, mod):
        """ArticleReference includes ARTICLE_5."""
        assert mod.ArticleReference.ARTICLE_5.value == "article_5"

    def test_article_reference_article6(self, mod):
        """ArticleReference includes ARTICLE_6."""
        assert mod.ArticleReference.ARTICLE_6.value == "article_6"

    def test_article_reference_article7(self, mod):
        """ArticleReference includes ARTICLE_7."""
        assert mod.ArticleReference.ARTICLE_7.value == "article_7"

    def test_article_reference_article8(self, mod):
        """ArticleReference includes ARTICLE_8."""
        assert mod.ArticleReference.ARTICLE_8.value == "article_8"

    def test_article_reference_values_set(self, mod):
        """ArticleReference values match expected set."""
        values = {m.value for m in mod.ArticleReference}
        expected = {
            "article_3", "article_4", "article_5",
            "article_6", "article_7", "article_8",
        }
        assert values == expected

    # -- ComplianceStatus --

    def test_compliance_status_count(self, mod):
        """ComplianceStatus has exactly 4 values."""
        assert len(mod.ComplianceStatus) == 4

    def test_compliance_status_compliant(self, mod):
        """ComplianceStatus includes COMPLIANT."""
        assert mod.ComplianceStatus.COMPLIANT.value == "compliant"

    def test_compliance_status_partially_compliant(self, mod):
        """ComplianceStatus includes PARTIALLY_COMPLIANT."""
        assert mod.ComplianceStatus.PARTIALLY_COMPLIANT.value == "partially_compliant"

    def test_compliance_status_non_compliant(self, mod):
        """ComplianceStatus includes NON_COMPLIANT."""
        assert mod.ComplianceStatus.NON_COMPLIANT.value == "non_compliant"

    def test_compliance_status_not_applicable(self, mod):
        """ComplianceStatus includes NOT_APPLICABLE."""
        assert mod.ComplianceStatus.NOT_APPLICABLE.value == "not_applicable"

    def test_compliance_status_values_set(self, mod):
        """ComplianceStatus values match expected set."""
        values = {m.value for m in mod.ComplianceStatus}
        expected = {
            "compliant", "partially_compliant",
            "non_compliant", "not_applicable",
        }
        assert values == expected

    # -- ClaimLifecycleStage --

    def test_claim_lifecycle_stage_count(self, mod):
        """ClaimLifecycleStage has exactly 7 values."""
        assert len(mod.ClaimLifecycleStage) == 7

    def test_claim_lifecycle_stage_draft(self, mod):
        """ClaimLifecycleStage includes DRAFT."""
        assert mod.ClaimLifecycleStage.DRAFT.value == "draft"

    def test_claim_lifecycle_stage_substantiated(self, mod):
        """ClaimLifecycleStage includes SUBSTANTIATED."""
        assert mod.ClaimLifecycleStage.SUBSTANTIATED.value == "substantiated"

    def test_claim_lifecycle_stage_submitted(self, mod):
        """ClaimLifecycleStage includes SUBMITTED_FOR_VERIFICATION."""
        assert mod.ClaimLifecycleStage.SUBMITTED_FOR_VERIFICATION.value == "submitted_for_verification"

    def test_claim_lifecycle_stage_verified(self, mod):
        """ClaimLifecycleStage includes VERIFIED."""
        assert mod.ClaimLifecycleStage.VERIFIED.value == "verified"

    def test_claim_lifecycle_stage_published(self, mod):
        """ClaimLifecycleStage includes PUBLISHED."""
        assert mod.ClaimLifecycleStage.PUBLISHED.value == "published"

    def test_claim_lifecycle_stage_expired(self, mod):
        """ClaimLifecycleStage includes EXPIRED."""
        assert mod.ClaimLifecycleStage.EXPIRED.value == "expired"

    def test_claim_lifecycle_stage_withdrawn(self, mod):
        """ClaimLifecycleStage includes WITHDRAWN."""
        assert mod.ClaimLifecycleStage.WITHDRAWN.value == "withdrawn"

    def test_claim_lifecycle_stage_values_set(self, mod):
        """ClaimLifecycleStage values match expected set."""
        values = {m.value for m in mod.ClaimLifecycleStage}
        expected = {
            "draft", "substantiated", "submitted_for_verification",
            "verified", "published", "expired", "withdrawn",
        }
        assert values == expected


# ===========================================================================
# Constants Tests
# ===========================================================================


class TestTraderObligationConstants:
    """Tests for Trader Obligation engine constants."""

    def test_article_3_checklist_exists(self, mod):
        """ARTICLE_3_CHECKLIST list exists with 8 items."""
        assert len(mod.ARTICLE_3_CHECKLIST) == 8

    def test_article_3_checklist_first_item(self, mod):
        """First ARTICLE_3_CHECKLIST item references scientific evidence."""
        assert "scientific evidence" in mod.ARTICLE_3_CHECKLIST[0].lower()

    def test_article_3_checklist_all_strings(self, mod):
        """All ARTICLE_3_CHECKLIST items are non-empty strings."""
        for item in mod.ARTICLE_3_CHECKLIST:
            assert isinstance(item, str)
            assert len(item.strip()) > 0

    def test_record_retention_years(self, mod):
        """RECORD_RETENTION_YEARS is 5."""
        assert mod.RECORD_RETENTION_YEARS == 5

    def test_article_weights_exists(self, mod):
        """ARTICLE_WEIGHTS dict exists with 6 entries."""
        assert len(mod.ARTICLE_WEIGHTS) == 6

    def test_article_weights_keys(self, mod):
        """ARTICLE_WEIGHTS has keys for all 6 articles."""
        expected_keys = {
            "article_3", "article_4", "article_5",
            "article_6", "article_7", "article_8",
        }
        assert set(mod.ARTICLE_WEIGHTS.keys()) == expected_keys

    def test_article_weights_are_decimal(self, mod):
        """All ARTICLE_WEIGHTS values are Decimal."""
        for weight in mod.ARTICLE_WEIGHTS.values():
            assert isinstance(weight, Decimal)

    def test_article_weights_sum_to_100(self, mod):
        """ARTICLE_WEIGHTS values sum to 100."""
        total = sum(mod.ARTICLE_WEIGHTS.values())
        assert total == Decimal("100")

    def test_article_3_weight_is_30(self, mod):
        """Article 3 weight is 30 (highest priority)."""
        assert mod.ARTICLE_WEIGHTS["article_3"] == Decimal("30")

    def test_valid_transitions_exists(self, mod):
        """VALID_TRANSITIONS dict exists with 7 entries."""
        assert len(mod.VALID_TRANSITIONS) == 7

    def test_valid_transitions_draft_can_move_to_substantiated(self, mod):
        """DRAFT can transition to SUBSTANTIATED."""
        assert "substantiated" in mod.VALID_TRANSITIONS["draft"]

    def test_valid_transitions_expired_is_terminal(self, mod):
        """EXPIRED is a terminal state with no transitions."""
        assert mod.VALID_TRANSITIONS["expired"] == []

    def test_valid_transitions_withdrawn_is_terminal(self, mod):
        """WITHDRAWN is a terminal state with no transitions."""
        assert mod.VALID_TRANSITIONS["withdrawn"] == []

    def test_remediation_priority_exists(self, mod):
        """REMEDIATION_PRIORITY dict exists with 6 entries."""
        assert len(mod.REMEDIATION_PRIORITY) == 6

    def test_remediation_priority_article3_highest(self, mod):
        """Article 3 has highest remediation priority (1)."""
        assert mod.REMEDIATION_PRIORITY["article_3"] == 1


# ===========================================================================
# Model Tests
# ===========================================================================


class TestObligationItemModel:
    """Tests for ObligationItem Pydantic model."""

    def test_create_obligation_item(self, mod):
        """Create an ObligationItem with required fields."""
        item = mod.ObligationItem(
            article=mod.ArticleReference.ARTICLE_3,
            description="Test obligation requirement",
        )
        assert item.article == mod.ArticleReference.ARTICLE_3
        assert item.description == "Test obligation requirement"

    def test_obligation_item_auto_id(self, mod):
        """ObligationItem auto-generates obligation_id."""
        item = mod.ObligationItem(
            article=mod.ArticleReference.ARTICLE_4,
            description="Test obligation",
        )
        assert item.obligation_id is not None
        assert len(item.obligation_id) > 0

    def test_obligation_item_default_status(self, mod):
        """ObligationItem defaults status to NON_COMPLIANT."""
        item = mod.ObligationItem(
            article=mod.ArticleReference.ARTICLE_5,
            description="Test obligation",
        )
        assert item.status == mod.ComplianceStatus.NON_COMPLIANT

    def test_obligation_item_with_evidence_ref(self, mod):
        """ObligationItem can include evidence_ref."""
        item = mod.ObligationItem(
            article=mod.ArticleReference.ARTICLE_7,
            description="Verification requirement",
            status=mod.ComplianceStatus.COMPLIANT,
            evidence_ref="DOC-001",
        )
        assert item.evidence_ref == "DOC-001"
        assert item.status == mod.ComplianceStatus.COMPLIANT

    def test_obligation_item_with_remediation_note(self, mod):
        """ObligationItem can include remediation_note."""
        item = mod.ObligationItem(
            article=mod.ArticleReference.ARTICLE_8,
            description="Record keeping requirement",
            remediation_note="Maintain records for 5 years",
        )
        assert item.remediation_note == "Maintain records for 5 years"

    def test_obligation_item_paragraph_field(self, mod):
        """ObligationItem has paragraph field default empty."""
        item = mod.ObligationItem(
            article=mod.ArticleReference.ARTICLE_3,
            description="Paragraph test",
            paragraph="3(1)(a)",
        )
        assert item.paragraph == "3(1)(a)"


class TestClaimLifecycleRecordModel:
    """Tests for ClaimLifecycleRecord Pydantic model."""

    def test_create_lifecycle_record(self, mod):
        """Create a ClaimLifecycleRecord with defaults."""
        record = mod.ClaimLifecycleRecord()
        assert record.claim_id is not None
        assert record.current_stage == mod.ClaimLifecycleStage.DRAFT
        assert record.transitions == []

    def test_lifecycle_record_custom_claim_id(self, mod):
        """ClaimLifecycleRecord can set custom claim_id."""
        record = mod.ClaimLifecycleRecord(claim_id="CLM-TEST-001")
        assert record.claim_id == "CLM-TEST-001"

    def test_lifecycle_record_has_timestamps(self, mod):
        """ClaimLifecycleRecord has created_at and last_updated."""
        record = mod.ClaimLifecycleRecord()
        assert record.created_at is not None
        assert record.last_updated is not None


class TestTraderObligationResultModel:
    """Tests for TraderObligationResult Pydantic model."""

    def test_create_result_with_defaults(self, mod):
        """Create a TraderObligationResult with defaults."""
        result = mod.TraderObligationResult()
        assert result.result_id is not None
        assert result.overall_compliance == mod.ComplianceStatus.NON_COMPLIANT
        assert result.overall_score == Decimal("0")
        assert result.obligations == []
        assert result.issues == []

    def test_result_has_provenance_hash_field(self, mod):
        """TraderObligationResult has provenance_hash field."""
        result = mod.TraderObligationResult()
        assert hasattr(result, "provenance_hash")

    def test_result_has_processing_time_field(self, mod):
        """TraderObligationResult has processing_time_ms field."""
        result = mod.TraderObligationResult()
        assert hasattr(result, "processing_time_ms")

    def test_result_has_article_scores_field(self, mod):
        """TraderObligationResult has article_scores field."""
        result = mod.TraderObligationResult()
        assert isinstance(result.article_scores, dict)

    def test_result_has_remediation_items_field(self, mod):
        """TraderObligationResult has remediation_items field."""
        result = mod.TraderObligationResult()
        assert isinstance(result.remediation_items, list)


# ===========================================================================
# Engine Method Tests
# ===========================================================================


class TestTraderObligationEngine:
    """Tests for TraderObligationEngine methods."""

    def test_engine_instantiation(self, mod):
        """Engine can be instantiated."""
        engine = mod.TraderObligationEngine()
        assert engine is not None

    def test_engine_has_assess_article3(self, engine):
        """Engine has assess_article3 method."""
        assert hasattr(engine, "assess_article3")
        assert callable(engine.assess_article3)

    def test_engine_has_assess_article4(self, engine):
        """Engine has assess_article4 method."""
        assert hasattr(engine, "assess_article4")
        assert callable(engine.assess_article4)

    def test_engine_has_assess_article5(self, engine):
        """Engine has assess_article5 method."""
        assert hasattr(engine, "assess_article5")
        assert callable(engine.assess_article5)

    def test_engine_has_assess_article6(self, engine):
        """Engine has assess_article6 method."""
        assert hasattr(engine, "assess_article6")
        assert callable(engine.assess_article6)

    def test_engine_has_assess_article7(self, engine):
        """Engine has assess_article7 method."""
        assert hasattr(engine, "assess_article7")
        assert callable(engine.assess_article7)

    def test_engine_has_assess_article8(self, engine):
        """Engine has assess_article8 method."""
        assert hasattr(engine, "assess_article8")
        assert callable(engine.assess_article8)

    def test_engine_has_calculate_overall_compliance(self, engine):
        """Engine has calculate_overall_compliance method."""
        assert hasattr(engine, "calculate_overall_compliance")
        assert callable(engine.calculate_overall_compliance)

    def test_engine_has_generate_remediation_timeline(self, engine):
        """Engine has generate_remediation_timeline method."""
        assert hasattr(engine, "generate_remediation_timeline")
        assert callable(engine.generate_remediation_timeline)

    def test_engine_has_track_claim_lifecycle(self, engine):
        """Engine has track_claim_lifecycle method."""
        assert hasattr(engine, "track_claim_lifecycle")
        assert callable(engine.track_claim_lifecycle)

    def test_engine_has_docstring(self, mod):
        """TraderObligationEngine class has a docstring."""
        assert mod.TraderObligationEngine.__doc__ is not None

    def test_engine_has_engine_id(self, engine):
        """Engine instance has engine_id attribute."""
        assert hasattr(engine, "engine_id")
        assert len(engine.engine_id) > 0

    def test_engine_has_engine_version(self, engine):
        """Engine instance has engine_version attribute."""
        assert hasattr(engine, "engine_version")
        assert engine.engine_version == "1.0.0"


# ===========================================================================
# Provenance and Source Checks
# ===========================================================================


class TestTraderObligationProvenance:
    """Tests for source file characteristics and provenance."""

    def test_engine_source_has_sha256(self):
        """Engine source uses SHA-256 for provenance."""
        source = (ENGINES_DIR / "trader_obligation_engine.py").read_text(
            encoding="utf-8"
        )
        assert "sha256" in source.lower() or "hashlib" in source

    def test_engine_source_has_decimal(self):
        """Engine source uses Decimal arithmetic."""
        source = (ENGINES_DIR / "trader_obligation_engine.py").read_text(
            encoding="utf-8"
        )
        assert "Decimal" in source

    def test_engine_source_has_basemodel(self):
        """Engine source uses Pydantic BaseModel."""
        source = (ENGINES_DIR / "trader_obligation_engine.py").read_text(
            encoding="utf-8"
        )
        assert "BaseModel" in source

    def test_engine_source_has_round_half_up(self):
        """Engine source uses ROUND_HALF_UP for regulatory consistency."""
        source = (ENGINES_DIR / "trader_obligation_engine.py").read_text(
            encoding="utf-8"
        )
        assert "ROUND_HALF_UP" in source

    def test_engine_source_references_article8(self):
        """Engine source references Article 8."""
        source = (ENGINES_DIR / "trader_obligation_engine.py").read_text(
            encoding="utf-8"
        )
        assert "Article 8" in source

    def test_engine_source_references_com_2023_166(self):
        """Engine source references the Green Claims Directive COM/2023/166."""
        source = (ENGINES_DIR / "trader_obligation_engine.py").read_text(
            encoding="utf-8"
        )
        assert "COM/2023/166" in source

    def test_engine_source_has_logging(self):
        """Engine source uses logging."""
        source = (ENGINES_DIR / "trader_obligation_engine.py").read_text(
            encoding="utf-8"
        )
        assert "logging" in source

    def test_engine_file_exists(self):
        """Engine source file exists on disk."""
        assert (ENGINES_DIR / "trader_obligation_engine.py").exists()


# ===========================================================================
# Sample Data Tests
# ===========================================================================


class TestTraderSampleData:
    """Tests using trader profile fixtures from conftest."""

    def test_sample_trader_profile(self, sample_trader_profile):
        """sample_trader_profile has expected fields."""
        assert "entity_id" in sample_trader_profile
        assert "entity_name" in sample_trader_profile
        assert "entity_type" in sample_trader_profile
        assert "entity_size" in sample_trader_profile

    def test_sample_trader_is_medium(self, sample_trader_profile):
        """Sample trader is MEDIUM entity."""
        assert sample_trader_profile["entity_size"] == "MEDIUM"

    def test_sample_trader_has_turnover(self, sample_trader_profile):
        """Sample trader has annual turnover."""
        assert isinstance(
            sample_trader_profile["annual_turnover_eur"], Decimal
        )

    def test_sample_sme_profile(self, sample_sme_profile):
        """sample_sme_profile has expected fields."""
        assert sample_sme_profile["entity_size"] == "MICRO"
        assert sample_sme_profile["employee_count"] < 10

    def test_sample_sme_turnover(self, sample_sme_profile):
        """SME profile turnover is under 2M EUR."""
        assert sample_sme_profile["annual_turnover_eur"] < Decimal("2000000")
