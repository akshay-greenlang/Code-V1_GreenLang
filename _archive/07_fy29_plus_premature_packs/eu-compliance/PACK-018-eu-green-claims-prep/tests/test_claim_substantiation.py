# -*- coding: utf-8 -*-
"""
PACK-018 EU Green Claims Prep Pack - Claim Substantiation Engine Tests
=======================================================================

Unit tests for ClaimSubstantiationEngine covering enums (ClaimType 16 values,
ClaimRiskLevel, SubstantiationLevel, EvidenceType, LifecycleStage), models
(EnvironmentalClaim, ClaimEvidence, SubstantiationAssessment,
ClaimCompletenessResult), constants, and engine methods (assess_claim,
calculate_substantiation_score, check_compliance, validate_claim_completeness).

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
    """Load the Claim Substantiation engine module."""
    return _load_engine("claim_substantiation")


@pytest.fixture
def engine(mod):
    """Create a fresh ClaimSubstantiationEngine instance."""
    return mod.ClaimSubstantiationEngine()


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestClaimSubstantiationEnums:
    """Tests for Claim Substantiation engine enums."""

    def test_claim_type_count(self, mod):
        """ClaimType has exactly 16 values."""
        assert len(mod.ClaimType) == 16

    def test_claim_type_carbon_neutral(self, mod):
        """ClaimType includes CARBON_NEUTRAL."""
        assert mod.ClaimType.CARBON_NEUTRAL.value == "carbon_neutral"

    def test_claim_type_climate_positive(self, mod):
        """ClaimType includes CLIMATE_POSITIVE."""
        assert mod.ClaimType.CLIMATE_POSITIVE.value == "climate_positive"

    def test_claim_type_net_zero(self, mod):
        """ClaimType includes NET_ZERO."""
        assert mod.ClaimType.NET_ZERO.value == "net_zero"

    def test_claim_type_eco_friendly(self, mod):
        """ClaimType includes ECO_FRIENDLY."""
        assert mod.ClaimType.ECO_FRIENDLY.value == "eco_friendly"

    def test_claim_type_sustainable(self, mod):
        """ClaimType includes SUSTAINABLE."""
        assert mod.ClaimType.SUSTAINABLE.value == "sustainable"

    def test_claim_type_recyclable(self, mod):
        """ClaimType includes RECYCLABLE."""
        assert mod.ClaimType.RECYCLABLE.value == "recyclable"

    def test_claim_type_biodegradable(self, mod):
        """ClaimType includes BIODEGRADABLE."""
        assert mod.ClaimType.BIODEGRADABLE.value == "biodegradable"

    def test_claim_type_compostable(self, mod):
        """ClaimType includes COMPOSTABLE."""
        assert mod.ClaimType.COMPOSTABLE.value == "compostable"

    def test_claim_type_plastic_free(self, mod):
        """ClaimType includes PLASTIC_FREE."""
        assert mod.ClaimType.PLASTIC_FREE.value == "plastic_free"

    def test_claim_type_zero_waste(self, mod):
        """ClaimType includes ZERO_WASTE."""
        assert mod.ClaimType.ZERO_WASTE.value == "zero_waste"

    def test_claim_type_is_str_enum(self, mod):
        """ClaimType members are string-valued enums."""
        for member in mod.ClaimType:
            assert isinstance(member.value, str)

    def test_claim_risk_level_count(self, mod):
        """ClaimRiskLevel has exactly 4 values."""
        assert len(mod.ClaimRiskLevel) == 4

    def test_claim_risk_level_values(self, mod):
        """ClaimRiskLevel contains low, medium, high, critical."""
        values = {m.value for m in mod.ClaimRiskLevel}
        assert values == {"low", "medium", "high", "critical"}

    def test_substantiation_level_count(self, mod):
        """SubstantiationLevel has exactly 5 values."""
        assert len(mod.SubstantiationLevel) == 5

    def test_substantiation_level_values(self, mod):
        """SubstantiationLevel values span excellent to insufficient."""
        values = {m.value for m in mod.SubstantiationLevel}
        expected = {"excellent", "good", "moderate", "weak", "insufficient"}
        assert values == expected

    def test_evidence_type_count(self, mod):
        """EvidenceType has exactly 6 values."""
        assert len(mod.EvidenceType) == 6

    def test_evidence_type_lca_study(self, mod):
        """EvidenceType includes LCA_STUDY."""
        assert mod.EvidenceType.LCA_STUDY.value == "lca_study"

    def test_lifecycle_stage_count(self, mod):
        """LifecycleStage has exactly 6 values."""
        assert len(mod.LifecycleStage) == 6

    def test_lifecycle_stage_values(self, mod):
        """LifecycleStage covers full product lifecycle."""
        values = {m.value for m in mod.LifecycleStage}
        expected = {
            "raw_materials", "manufacturing", "transportation",
            "distribution", "use", "end_of_life",
        }
        assert values == expected


# ===========================================================================
# Constants Tests
# ===========================================================================


class TestClaimSubstantiationConstants:
    """Tests for Claim Substantiation engine constants."""

    def test_dimension_weights_exist(self, mod):
        """DIMENSION_WEIGHTS dict exists with 5 entries."""
        assert len(mod.DIMENSION_WEIGHTS) == 5

    def test_dimension_weights_sum_to_100(self, mod):
        """DIMENSION_WEIGHTS sum to exactly 100."""
        total = sum(mod.DIMENSION_WEIGHTS.values())
        assert total == Decimal("100")

    def test_dimension_weight_keys(self, mod):
        """DIMENSION_WEIGHTS has correct dimension keys."""
        expected_keys = {
            "scientific_validity", "data_quality", "scope_completeness",
            "verification_independence", "transparency",
        }
        assert set(mod.DIMENSION_WEIGHTS.keys()) == expected_keys

    def test_substantiation_thresholds_exist(self, mod):
        """SUBSTANTIATION_THRESHOLDS dict exists."""
        assert len(mod.SUBSTANTIATION_THRESHOLDS) >= 4

    def test_compliance_threshold_value(self, mod):
        """COMPLIANCE_THRESHOLD is Decimal 50."""
        assert mod.COMPLIANCE_THRESHOLD == Decimal("50")

    def test_claim_requirements_exist(self, mod):
        """CLAIM_REQUIREMENTS has entries for all 16 claim types."""
        assert len(mod.CLAIM_REQUIREMENTS) == 16

    def test_claim_requirements_carbon_neutral(self, mod):
        """CLAIM_REQUIREMENTS for carbon_neutral has required_evidence."""
        reqs = mod.CLAIM_REQUIREMENTS["carbon_neutral"]
        assert "required_evidence" in reqs
        assert "required_lifecycle_stages" in reqs
        assert "risk_level" in reqs

    def test_claim_type_descriptions_exist(self, mod):
        """CLAIM_TYPE_DESCRIPTIONS has 16 entries."""
        assert len(mod.CLAIM_TYPE_DESCRIPTIONS) == 16


# ===========================================================================
# Model Tests
# ===========================================================================


class TestEnvironmentalClaimModel:
    """Tests for EnvironmentalClaim Pydantic model."""

    def test_create_valid_claim(self, mod):
        """Create a valid EnvironmentalClaim with required fields."""
        claim = mod.EnvironmentalClaim(
            claim_text="Our product is recyclable",
            claim_type=mod.ClaimType.RECYCLABLE,
            product_or_org="Test Product",
        )
        assert claim.claim_type == mod.ClaimType.RECYCLABLE
        assert claim.claim_text == "Our product is recyclable"

    def test_claim_has_auto_id(self, mod):
        """EnvironmentalClaim auto-generates claim_id."""
        claim = mod.EnvironmentalClaim(
            claim_text="Test claim",
            claim_type=mod.ClaimType.GREEN,
            product_or_org="Test",
        )
        assert claim.claim_id is not None
        assert len(claim.claim_id) > 0

    def test_claim_default_lifecycle_stages(self, mod):
        """EnvironmentalClaim defaults lifecycle_stages_covered to empty list."""
        claim = mod.EnvironmentalClaim(
            claim_text="Test",
            claim_type=mod.ClaimType.GREEN,
            product_or_org="Test",
        )
        assert claim.lifecycle_stages_covered == []

    def test_claim_empty_text_raises_error(self, mod):
        """EnvironmentalClaim rejects empty claim text."""
        with pytest.raises(Exception):
            mod.EnvironmentalClaim(
                claim_text="   ",
                claim_type=mod.ClaimType.GREEN,
                product_or_org="Test",
            )


class TestClaimEvidenceModel:
    """Tests for ClaimEvidence Pydantic model."""

    def test_create_valid_evidence(self, mod):
        """Create a valid ClaimEvidence with required fields."""
        evidence = mod.ClaimEvidence(
            evidence_type=mod.EvidenceType.LCA_STUDY,
            source="Test Lab",
        )
        assert evidence.evidence_type == mod.EvidenceType.LCA_STUDY
        assert evidence.source == "Test Lab"

    def test_evidence_default_third_party(self, mod):
        """ClaimEvidence defaults is_third_party to False."""
        evidence = mod.ClaimEvidence(
            evidence_type=mod.EvidenceType.MEASUREMENT,
            source="Internal",
        )
        assert evidence.is_third_party is False

    def test_evidence_data_quality_default(self, mod):
        """ClaimEvidence defaults data_quality_score to 0."""
        evidence = mod.ClaimEvidence(
            evidence_type=mod.EvidenceType.TEST_REPORT,
            source="Lab",
        )
        assert evidence.data_quality_score == Decimal("0")


class TestSubstantiationAssessmentModel:
    """Tests for SubstantiationAssessment Pydantic model."""

    def test_create_assessment(self, mod):
        """Create a SubstantiationAssessment with defaults."""
        assessment = mod.SubstantiationAssessment(claim_id="CLM-001")
        assert assessment.claim_id == "CLM-001"
        assert assessment.overall_score == Decimal("0.00")
        assert assessment.compliant is False

    def test_assessment_has_provenance_field(self, mod):
        """SubstantiationAssessment has provenance_hash field."""
        assessment = mod.SubstantiationAssessment(claim_id="CLM-001")
        assert hasattr(assessment, "provenance_hash")

    def test_assessment_has_engine_version(self, mod):
        """SubstantiationAssessment has engine_version field."""
        assessment = mod.SubstantiationAssessment(claim_id="CLM-001")
        assert assessment.engine_version == "1.0.0"


# ===========================================================================
# Engine Method Tests
# ===========================================================================


class TestClaimSubstantiationEngine:
    """Tests for ClaimSubstantiationEngine methods."""

    def test_engine_instantiation(self, mod):
        """Engine can be instantiated."""
        engine = mod.ClaimSubstantiationEngine()
        assert engine is not None

    def test_engine_has_assess_claim(self, engine):
        """Engine has assess_claim method."""
        assert hasattr(engine, "assess_claim")
        assert callable(engine.assess_claim)

    def test_engine_has_calculate_substantiation_score(self, engine):
        """Engine has calculate_substantiation_score method."""
        assert hasattr(engine, "calculate_substantiation_score")
        assert callable(engine.calculate_substantiation_score)

    def test_engine_has_check_compliance(self, engine):
        """Engine has check_compliance method."""
        assert hasattr(engine, "check_compliance")
        assert callable(engine.check_compliance)

    def test_engine_has_validate_claim_completeness(self, engine):
        """Engine has validate_claim_completeness method."""
        assert hasattr(engine, "validate_claim_completeness")
        assert callable(engine.validate_claim_completeness)

    def test_engine_version(self, engine):
        """Engine has version string."""
        assert engine.engine_version == "1.0.0"

    def test_engine_has_docstring(self, mod):
        """ClaimSubstantiationEngine class has a docstring."""
        assert mod.ClaimSubstantiationEngine.__doc__ is not None
        assert "substantiation" in mod.ClaimSubstantiationEngine.__doc__.lower()


# ===========================================================================
# Provenance and Source Checks
# ===========================================================================


class TestClaimSubstantiationProvenance:
    """Tests for source file characteristics and provenance."""

    def test_engine_source_has_sha256(self):
        """Engine source uses SHA-256 for provenance."""
        source = (ENGINES_DIR / "claim_substantiation_engine.py").read_text(
            encoding="utf-8"
        )
        assert "sha256" in source.lower() or "hashlib" in source

    def test_engine_source_has_decimal(self):
        """Engine source uses Decimal arithmetic."""
        source = (ENGINES_DIR / "claim_substantiation_engine.py").read_text(
            encoding="utf-8"
        )
        assert "Decimal" in source

    def test_engine_source_has_basemodel(self):
        """Engine source uses Pydantic BaseModel."""
        source = (ENGINES_DIR / "claim_substantiation_engine.py").read_text(
            encoding="utf-8"
        )
        assert "BaseModel" in source

    def test_engine_source_has_logging(self):
        """Engine source uses logging."""
        source = (ENGINES_DIR / "claim_substantiation_engine.py").read_text(
            encoding="utf-8"
        )
        assert "logging" in source

    def test_engine_source_references_article3(self):
        """Engine source references Article 3 of Green Claims Directive."""
        source = (ENGINES_DIR / "claim_substantiation_engine.py").read_text(
            encoding="utf-8"
        )
        assert "Article 3" in source

    def test_engine_source_references_article4(self):
        """Engine source references Article 4 of Green Claims Directive."""
        source = (ENGINES_DIR / "claim_substantiation_engine.py").read_text(
            encoding="utf-8"
        )
        assert "Article 4" in source

    def test_engine_file_exists(self):
        """Engine source file exists on disk."""
        assert (ENGINES_DIR / "claim_substantiation_engine.py").exists()
