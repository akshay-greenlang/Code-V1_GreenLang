# -*- coding: utf-8 -*-
"""
PACK-018 EU Green Claims Prep Pack - Greenwashing Detection Engine Tests
=========================================================================

Unit tests for GreenwashingDetectionEngine covering enums (GreenwashingSin 7
values, AlertSeverity, ProhibitedPractice, ClaimType), models
(GreenwashingAlert, ClaimScreeningInput), constants (VAGUE_KEYWORDS,
VAGUE_PHRASES, OFFSET_NEUTRALITY_KEYWORDS, PROHIBITED_PATTERNS,
SEVERITY_RISK_WEIGHTS), and engine methods (screen_claim, detect_seven_sins,
check_prohibited_practices, calculate_risk_score, screen_portfolio).

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
    """Load the Greenwashing Detection engine module."""
    return _load_engine("greenwashing_detection")


@pytest.fixture
def engine(mod):
    """Create a fresh GreenwashingDetectionEngine instance."""
    return mod.GreenwashingDetectionEngine()


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestGreenwashingDetectionEnums:
    """Tests for Greenwashing Detection engine enums."""

    def test_greenwashing_sin_count(self, mod):
        """GreenwashingSin has exactly 7 values (TerraChoice Seven Sins)."""
        assert len(mod.GreenwashingSin) == 7

    def test_greenwashing_sin_hidden_tradeoff(self, mod):
        """GreenwashingSin includes HIDDEN_TRADEOFF."""
        assert mod.GreenwashingSin.HIDDEN_TRADEOFF.value == "hidden_tradeoff"

    def test_greenwashing_sin_no_proof(self, mod):
        """GreenwashingSin includes NO_PROOF."""
        assert mod.GreenwashingSin.NO_PROOF.value == "no_proof"

    def test_greenwashing_sin_vagueness(self, mod):
        """GreenwashingSin includes VAGUENESS."""
        assert mod.GreenwashingSin.VAGUENESS.value == "vagueness"

    def test_greenwashing_sin_false_labels(self, mod):
        """GreenwashingSin includes FALSE_LABELS."""
        assert mod.GreenwashingSin.FALSE_LABELS.value == "false_labels"

    def test_greenwashing_sin_irrelevance(self, mod):
        """GreenwashingSin includes IRRELEVANCE."""
        assert mod.GreenwashingSin.IRRELEVANCE.value == "irrelevance"

    def test_greenwashing_sin_lesser_of_two_evils(self, mod):
        """GreenwashingSin includes LESSER_OF_TWO_EVILS."""
        assert mod.GreenwashingSin.LESSER_OF_TWO_EVILS.value == "lesser_of_two_evils"

    def test_greenwashing_sin_fibbing(self, mod):
        """GreenwashingSin includes FIBBING."""
        assert mod.GreenwashingSin.FIBBING.value == "fibbing"

    def test_alert_severity_count(self, mod):
        """AlertSeverity has exactly 4 values."""
        assert len(mod.AlertSeverity) == 4

    def test_alert_severity_values(self, mod):
        """AlertSeverity contains low, medium, high, critical."""
        values = {m.value for m in mod.AlertSeverity}
        assert values == {"low", "medium", "high", "critical"}

    def test_prohibited_practice_count(self, mod):
        """ProhibitedPractice has exactly 6 values."""
        assert len(mod.ProhibitedPractice) == 6

    def test_prohibited_practice_generic_excellence(self, mod):
        """ProhibitedPractice includes GENERIC_EXCELLENCE_CLAIM."""
        assert mod.ProhibitedPractice.GENERIC_EXCELLENCE_CLAIM.value == "generic_excellence_claim"

    def test_prohibited_practice_offset_neutrality(self, mod):
        """ProhibitedPractice includes OFFSET_NEUTRALITY_CLAIM."""
        assert mod.ProhibitedPractice.OFFSET_NEUTRALITY_CLAIM.value == "offset_neutrality_claim"

    def test_prohibited_practice_uncertified_label(self, mod):
        """ProhibitedPractice includes UNCERTIFIED_LABEL."""
        assert mod.ProhibitedPractice.UNCERTIFIED_LABEL.value == "uncertified_label"

    def test_claim_type_count(self, mod):
        """ClaimType has exactly 6 values."""
        assert len(mod.ClaimType) == 6

    def test_claim_type_values(self, mod):
        """ClaimType covers product, corporate, service etc."""
        values = {m.value for m in mod.ClaimType}
        expected = {"product", "corporate", "service", "process", "label", "marketing"}
        assert values == expected


# ===========================================================================
# Constants Tests
# ===========================================================================


class TestGreenwashingDetectionConstants:
    """Tests for Greenwashing Detection engine constants."""

    def test_vague_keywords_exist(self, mod):
        """VAGUE_KEYWORDS list exists with at least 10 entries."""
        assert len(mod.VAGUE_KEYWORDS) >= 10

    def test_vague_keywords_contains_green(self, mod):
        """VAGUE_KEYWORDS includes 'green'."""
        assert "green" in mod.VAGUE_KEYWORDS

    def test_vague_keywords_contains_eco(self, mod):
        """VAGUE_KEYWORDS includes 'eco'."""
        assert "eco" in mod.VAGUE_KEYWORDS

    def test_vague_keywords_contains_sustainable(self, mod):
        """VAGUE_KEYWORDS includes 'sustainable'."""
        assert "sustainable" in mod.VAGUE_KEYWORDS

    def test_vague_phrases_exist(self, mod):
        """VAGUE_PHRASES list exists with at least 10 entries."""
        assert len(mod.VAGUE_PHRASES) >= 10

    def test_offset_neutrality_keywords_exist(self, mod):
        """OFFSET_NEUTRALITY_KEYWORDS list exists with at least 10 entries."""
        assert len(mod.OFFSET_NEUTRALITY_KEYWORDS) >= 10

    def test_offset_neutrality_contains_carbon_neutral(self, mod):
        """OFFSET_NEUTRALITY_KEYWORDS includes 'carbon neutral'."""
        assert "carbon neutral" in mod.OFFSET_NEUTRALITY_KEYWORDS

    def test_future_claim_indicators_exist(self, mod):
        """FUTURE_CLAIM_INDICATORS list exists with at least 10 entries."""
        assert len(mod.FUTURE_CLAIM_INDICATORS) >= 10

    def test_legal_requirement_phrases_exist(self, mod):
        """LEGAL_REQUIREMENT_PHRASES list exists with entries."""
        assert len(mod.LEGAL_REQUIREMENT_PHRASES) >= 5

    def test_prohibited_patterns_count(self, mod):
        """PROHIBITED_PATTERNS has 6 entries for all prohibited practices."""
        assert len(mod.PROHIBITED_PATTERNS) == 6

    def test_severity_risk_weights_count(self, mod):
        """SEVERITY_RISK_WEIGHTS has 4 entries."""
        assert len(mod.SEVERITY_RISK_WEIGHTS) == 4

    def test_severity_risk_weights_critical_highest(self, mod):
        """Critical severity has the highest risk weight."""
        weights = mod.SEVERITY_RISK_WEIGHTS
        assert weights["critical"] > weights["high"]
        assert weights["high"] > weights["medium"]
        assert weights["medium"] > weights["low"]

    def test_sin_descriptions_exist(self, mod):
        """SIN_DESCRIPTIONS dict exists with 7 entries."""
        assert len(mod.SIN_DESCRIPTIONS) == 7


# ===========================================================================
# Model Tests
# ===========================================================================


class TestGreenwashingAlertModel:
    """Tests for GreenwashingAlert Pydantic model."""

    def test_create_valid_alert(self, mod):
        """Create a valid GreenwashingAlert."""
        alert = mod.GreenwashingAlert(
            sin_type="vagueness",
            severity=mod.AlertSeverity.HIGH.value,
            claim_text="Our product is eco-friendly",
            description="Vague claim detected",
            recommendation="Replace with specific measurable claim",
        )
        assert alert.severity == mod.AlertSeverity.HIGH.value

    def test_alert_has_auto_id(self, mod):
        """GreenwashingAlert auto-generates alert_id."""
        alert = mod.GreenwashingAlert(
            sin_type="no_proof",
            severity=mod.AlertSeverity.MEDIUM.value,
            claim_text="We are sustainable",
            description="No proof found",
            recommendation="Provide third-party certification or evidence",
        )
        assert alert.alert_id is not None


class TestClaimScreeningInputModel:
    """Tests for ClaimScreeningInput Pydantic model."""

    def test_create_screening_input(self, mod):
        """Create a valid ClaimScreeningInput."""
        inp = mod.ClaimScreeningInput(
            claim_text="Our product is eco-friendly",
        )
        assert "eco-friendly" in inp.claim_text


# ===========================================================================
# Engine Method Tests
# ===========================================================================


class TestGreenwashingDetectionEngine:
    """Tests for GreenwashingDetectionEngine methods."""

    def test_engine_instantiation(self, mod):
        """Engine can be instantiated."""
        engine = mod.GreenwashingDetectionEngine()
        assert engine is not None

    def test_engine_has_screen_claim(self, engine):
        """Engine has screen_claim method."""
        assert hasattr(engine, "screen_claim")
        assert callable(engine.screen_claim)

    def test_engine_has_detect_seven_sins(self, engine):
        """Engine has detect_seven_sins method."""
        assert hasattr(engine, "detect_seven_sins")
        assert callable(engine.detect_seven_sins)

    def test_engine_has_check_prohibited_practices(self, engine):
        """Engine has check_prohibited_practices method."""
        assert hasattr(engine, "check_prohibited_practices")
        assert callable(engine.check_prohibited_practices)

    def test_engine_has_calculate_risk_score(self, engine):
        """Engine has calculate_risk_score method."""
        assert hasattr(engine, "calculate_risk_score")
        assert callable(engine.calculate_risk_score)

    def test_engine_has_screen_portfolio(self, engine):
        """Engine has screen_portfolio method."""
        assert hasattr(engine, "screen_portfolio")
        assert callable(engine.screen_portfolio)

    def test_engine_has_docstring(self, mod):
        """GreenwashingDetectionEngine has a docstring."""
        assert mod.GreenwashingDetectionEngine.__doc__ is not None
        assert "greenwashing" in mod.GreenwashingDetectionEngine.__doc__.lower()


# ===========================================================================
# Provenance and Source Checks
# ===========================================================================


class TestGreenwashingDetectionProvenance:
    """Tests for source file characteristics and provenance."""

    def test_engine_source_has_sha256(self):
        """Engine source uses SHA-256 for provenance."""
        source = (ENGINES_DIR / "greenwashing_detection_engine.py").read_text(
            encoding="utf-8"
        )
        assert "sha256" in source.lower() or "hashlib" in source

    def test_engine_source_has_decimal(self):
        """Engine source uses Decimal arithmetic."""
        source = (ENGINES_DIR / "greenwashing_detection_engine.py").read_text(
            encoding="utf-8"
        )
        assert "Decimal" in source

    def test_engine_source_has_basemodel(self):
        """Engine source uses Pydantic BaseModel."""
        source = (ENGINES_DIR / "greenwashing_detection_engine.py").read_text(
            encoding="utf-8"
        )
        assert "BaseModel" in source

    def test_engine_source_references_terrachoice(self):
        """Engine source references TerraChoice Seven Sins."""
        source = (ENGINES_DIR / "greenwashing_detection_engine.py").read_text(
            encoding="utf-8"
        )
        assert "TerraChoice" in source or "Seven Sins" in source

    def test_engine_file_exists(self):
        """Engine source file exists on disk."""
        assert (ENGINES_DIR / "greenwashing_detection_engine.py").exists()
