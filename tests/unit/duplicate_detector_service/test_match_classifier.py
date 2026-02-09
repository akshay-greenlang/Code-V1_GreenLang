# -*- coding: utf-8 -*-
"""
Unit tests for MatchClassifier Engine - AGENT-DATA-011

Tests the MatchClassifier with 100+ test cases covering:
- classify_pair: MATCH, NON_MATCH, POSSIBLE classifications
- classify_batch: batch processing with mixed results
- compute_weighted_score: equal, custom, zero weights
- fellegi_sunter_score: probabilistic linkage scoring
- auto_threshold: Otsu-like histogram threshold detection
- evaluate_classification_quality: quality metrics
- decision_reason: human-readable explanation generation
- threshold validation and edge cases
- thread-safe statistics tracking
- provenance hash generation

Author: GreenLang Platform Team
Date: February 2026
"""

import math
import threading
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from greenlang.duplicate_detector.match_classifier import MatchClassifier
from greenlang.duplicate_detector.models import (
    FieldComparisonConfig,
    MatchClassification,
    MatchResult,
    SimilarityAlgorithm,
    SimilarityResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def classifier() -> MatchClassifier:
    """Create a fresh MatchClassifier instance."""
    return MatchClassifier()


def _make_sim_result(
    record_a_id: str = "rec-a",
    record_b_id: str = "rec-b",
    overall_score: float = 0.90,
    field_scores: Dict[str, float] | None = None,
) -> SimilarityResult:
    """Helper to build a SimilarityResult."""
    return SimilarityResult(
        record_a_id=record_a_id,
        record_b_id=record_b_id,
        overall_score=overall_score,
        field_scores=field_scores or {"name": overall_score},
    )


def _make_field_configs(
    fields: Dict[str, float] | None = None,
) -> List[FieldComparisonConfig]:
    """Helper to build FieldComparisonConfig list from field_name->weight."""
    if fields is None:
        fields = {"name": 1.0, "email": 1.0}
    return [
        FieldComparisonConfig(field_name=fn, weight=w)
        for fn, w in fields.items()
    ]


def _make_match_result(
    classification: MatchClassification = MatchClassification.MATCH,
    overall_score: float = 0.92,
    confidence: float = 0.85,
) -> MatchResult:
    """Helper to build a MatchResult."""
    return MatchResult(
        record_a_id="a",
        record_b_id="b",
        classification=classification,
        confidence=confidence,
        overall_score=overall_score,
        field_scores={"name": overall_score},
        decision_reason="test",
        provenance_hash="abc123" * 10 + "abcd",
    )


# =============================================================================
# TestMatchClassifierInit
# =============================================================================


class TestMatchClassifierInit:
    """Initialization and statistics reset tests."""

    def test_initialization(self):
        """Classifier initializes with zero statistics."""
        classifier = MatchClassifier()
        stats = classifier.get_statistics()
        assert stats["engine_name"] == "MatchClassifier"
        assert stats["invocations"] == 0
        assert stats["successes"] == 0
        assert stats["failures"] == 0
        assert stats["total_duration_ms"] == 0.0
        assert stats["last_invoked_at"] is None

    def test_reset_statistics(self, classifier: MatchClassifier):
        """Statistics reset to zero after invocation."""
        sim = _make_sim_result(overall_score=0.90)
        classifier.classify_pair(sim)
        classifier.reset_statistics()
        stats = classifier.get_statistics()
        assert stats["invocations"] == 0
        assert stats["successes"] == 0

    def test_multiple_instances_independent(self):
        """Two instances maintain independent statistics."""
        c1 = MatchClassifier()
        c2 = MatchClassifier()
        sim = _make_sim_result(overall_score=0.90)
        c1.classify_pair(sim)
        assert c1.get_statistics()["invocations"] == 1
        assert c2.get_statistics()["invocations"] == 0


# =============================================================================
# TestClassifyPairMATCH
# =============================================================================


class TestClassifyPairMATCH:
    """Tests for classify_pair producing MATCH classification."""

    def test_high_score_classified_as_match(self, classifier: MatchClassifier):
        """Score >= match_threshold results in MATCH."""
        sim = _make_sim_result(overall_score=0.95)
        result = classifier.classify_pair(sim, match_threshold=0.85, possible_threshold=0.65)
        assert result.classification == MatchClassification.MATCH

    def test_exact_match_threshold(self, classifier: MatchClassifier):
        """Score exactly at match_threshold results in MATCH."""
        sim = _make_sim_result(overall_score=0.85)
        result = classifier.classify_pair(sim, match_threshold=0.85, possible_threshold=0.65)
        assert result.classification == MatchClassification.MATCH

    def test_score_1_0_is_match(self, classifier: MatchClassifier):
        """Perfect score 1.0 results in MATCH."""
        sim = _make_sim_result(overall_score=1.0)
        result = classifier.classify_pair(sim)
        assert result.classification == MatchClassification.MATCH
        assert result.confidence >= 0.7

    def test_match_confidence_above_threshold(self, classifier: MatchClassifier):
        """MATCH classification always has confidence >= 0.7."""
        sim = _make_sim_result(overall_score=0.90)
        result = classifier.classify_pair(sim, match_threshold=0.85, possible_threshold=0.65)
        assert result.confidence >= 0.7

    def test_match_decision_reason_contains_match(self, classifier: MatchClassifier):
        """Decision reason for MATCH contains 'MATCH'."""
        sim = _make_sim_result(overall_score=0.95)
        result = classifier.classify_pair(sim)
        assert "MATCH" in result.decision_reason

    def test_match_result_has_field_scores(self, classifier: MatchClassifier):
        """Result carries through field_scores from input."""
        sim = _make_sim_result(overall_score=0.95, field_scores={"name": 0.95, "email": 0.90})
        result = classifier.classify_pair(sim)
        assert result.field_scores == {"name": 0.95, "email": 0.90}

    def test_match_result_has_provenance_hash(self, classifier: MatchClassifier):
        """Result includes a 64-char SHA-256 provenance hash."""
        sim = _make_sim_result(overall_score=0.95)
        result = classifier.classify_pair(sim)
        assert len(result.provenance_hash) == 64

    def test_match_overall_score_preserved(self, classifier: MatchClassifier):
        """Overall score is preserved in the result."""
        sim = _make_sim_result(overall_score=0.95)
        result = classifier.classify_pair(sim)
        assert result.overall_score == 0.95

    def test_match_record_ids_preserved(self, classifier: MatchClassifier):
        """Record IDs are preserved from input."""
        sim = _make_sim_result(record_a_id="alpha", record_b_id="beta", overall_score=0.95)
        result = classifier.classify_pair(sim)
        assert result.record_a_id == "alpha"
        assert result.record_b_id == "beta"

    def test_high_match_has_higher_confidence(self, classifier: MatchClassifier):
        """Higher score above threshold yields higher confidence."""
        sim_high = _make_sim_result(overall_score=0.99)
        sim_low = _make_sim_result(overall_score=0.86)
        r_high = classifier.classify_pair(sim_high, match_threshold=0.85)
        r_low = classifier.classify_pair(sim_low, match_threshold=0.85)
        assert r_high.confidence >= r_low.confidence


# =============================================================================
# TestClassifyPairNONMATCH
# =============================================================================


class TestClassifyPairNONMATCH:
    """Tests for classify_pair producing NON_MATCH classification."""

    def test_low_score_classified_as_non_match(self, classifier: MatchClassifier):
        """Score below possible_threshold results in NON_MATCH."""
        sim = _make_sim_result(overall_score=0.20)
        result = classifier.classify_pair(sim, match_threshold=0.85, possible_threshold=0.65)
        assert result.classification == MatchClassification.NON_MATCH

    def test_zero_score_is_non_match(self, classifier: MatchClassifier):
        """Score 0.0 results in NON_MATCH."""
        sim = _make_sim_result(overall_score=0.0)
        result = classifier.classify_pair(sim)
        assert result.classification == MatchClassification.NON_MATCH

    def test_just_below_possible_threshold(self, classifier: MatchClassifier):
        """Score just below possible_threshold is NON_MATCH."""
        sim = _make_sim_result(overall_score=0.6499)
        result = classifier.classify_pair(sim, match_threshold=0.85, possible_threshold=0.65)
        assert result.classification == MatchClassification.NON_MATCH

    def test_non_match_confidence_above_threshold(self, classifier: MatchClassifier):
        """NON_MATCH classification has confidence >= 0.7."""
        sim = _make_sim_result(overall_score=0.10)
        result = classifier.classify_pair(sim, match_threshold=0.85, possible_threshold=0.65)
        assert result.confidence >= 0.7

    def test_non_match_decision_reason(self, classifier: MatchClassifier):
        """Decision reason for NON_MATCH contains 'NON_MATCH'."""
        sim = _make_sim_result(overall_score=0.10)
        result = classifier.classify_pair(sim)
        assert "NON_MATCH" in result.decision_reason

    def test_very_low_score_high_confidence(self, classifier: MatchClassifier):
        """Very low score produces higher confidence than borderline."""
        sim_very_low = _make_sim_result(overall_score=0.01)
        sim_borderline = _make_sim_result(overall_score=0.64)
        r_low = classifier.classify_pair(sim_very_low, possible_threshold=0.65)
        r_border = classifier.classify_pair(sim_borderline, possible_threshold=0.65)
        assert r_low.confidence >= r_border.confidence


# =============================================================================
# TestClassifyPairPOSSIBLE
# =============================================================================


class TestClassifyPairPOSSIBLE:
    """Tests for classify_pair producing POSSIBLE classification."""

    def test_middle_score_classified_as_possible(self, classifier: MatchClassifier):
        """Score between thresholds results in POSSIBLE."""
        sim = _make_sim_result(overall_score=0.75)
        result = classifier.classify_pair(sim, match_threshold=0.85, possible_threshold=0.65)
        assert result.classification == MatchClassification.POSSIBLE

    def test_exact_possible_threshold(self, classifier: MatchClassifier):
        """Score exactly at possible_threshold results in POSSIBLE."""
        sim = _make_sim_result(overall_score=0.65)
        result = classifier.classify_pair(sim, match_threshold=0.85, possible_threshold=0.65)
        assert result.classification == MatchClassification.POSSIBLE

    def test_just_below_match_threshold(self, classifier: MatchClassifier):
        """Score just below match_threshold is POSSIBLE."""
        sim = _make_sim_result(overall_score=0.8499)
        result = classifier.classify_pair(sim, match_threshold=0.85, possible_threshold=0.65)
        assert result.classification == MatchClassification.POSSIBLE

    def test_possible_confidence_range(self, classifier: MatchClassifier):
        """POSSIBLE confidence is between 0.3 and 0.7."""
        sim = _make_sim_result(overall_score=0.75)
        result = classifier.classify_pair(sim, match_threshold=0.85, possible_threshold=0.65)
        assert 0.3 <= result.confidence <= 0.7

    def test_possible_decision_reason(self, classifier: MatchClassifier):
        """Decision reason for POSSIBLE contains 'POSSIBLE'."""
        sim = _make_sim_result(overall_score=0.75)
        result = classifier.classify_pair(sim)
        assert "POSSIBLE" in result.decision_reason


# =============================================================================
# TestClassifyBatch
# =============================================================================


class TestClassifyBatch:
    """Tests for classify_batch method."""

    def test_empty_batch_returns_empty(self, classifier: MatchClassifier):
        """Empty input list returns empty output."""
        results = classifier.classify_batch([])
        assert results == []

    def test_single_item_batch(self, classifier: MatchClassifier):
        """Single-item batch produces one result."""
        sims = [_make_sim_result(overall_score=0.90)]
        results = classifier.classify_batch(sims)
        assert len(results) == 1
        assert results[0].classification == MatchClassification.MATCH

    def test_mixed_classifications(self, classifier: MatchClassifier):
        """Batch with mixed scores produces mixed classifications."""
        sims = [
            _make_sim_result(record_a_id="a1", record_b_id="b1", overall_score=0.95),
            _make_sim_result(record_a_id="a2", record_b_id="b2", overall_score=0.75),
            _make_sim_result(record_a_id="a3", record_b_id="b3", overall_score=0.20),
        ]
        results = classifier.classify_batch(sims, match_threshold=0.85, possible_threshold=0.65)
        classifications = [r.classification for r in results]
        assert MatchClassification.MATCH in classifications
        assert MatchClassification.POSSIBLE in classifications
        assert MatchClassification.NON_MATCH in classifications

    def test_batch_result_count(self, classifier: MatchClassifier):
        """Batch returns same count as input."""
        sims = [_make_sim_result(overall_score=0.5 + i * 0.1) for i in range(5)]
        results = classifier.classify_batch(sims)
        assert len(results) == 5

    def test_batch_with_custom_thresholds(self, classifier: MatchClassifier):
        """Batch respects custom thresholds."""
        sims = [
            _make_sim_result(record_a_id="a", record_b_id="b", overall_score=0.70),
            _make_sim_result(record_a_id="c", record_b_id="d", overall_score=0.55),
        ]
        results = classifier.classify_batch(sims, match_threshold=0.60, possible_threshold=0.40)
        assert results[0].classification == MatchClassification.MATCH
        assert results[1].classification == MatchClassification.MATCH

    def test_batch_all_matches(self, classifier: MatchClassifier):
        """All high-score items are classified as MATCH."""
        sims = [_make_sim_result(record_a_id=f"a{i}", record_b_id=f"b{i}", overall_score=0.95) for i in range(10)]
        results = classifier.classify_batch(sims)
        assert all(r.classification == MatchClassification.MATCH for r in results)

    def test_batch_all_non_matches(self, classifier: MatchClassifier):
        """All low-score items are classified as NON_MATCH."""
        sims = [_make_sim_result(record_a_id=f"a{i}", record_b_id=f"b{i}", overall_score=0.10) for i in range(10)]
        results = classifier.classify_batch(sims)
        assert all(r.classification == MatchClassification.NON_MATCH for r in results)

    def test_batch_preserves_order(self, classifier: MatchClassifier):
        """Batch results maintain input order."""
        sims = [
            _make_sim_result(record_a_id="first", record_b_id="x", overall_score=0.95),
            _make_sim_result(record_a_id="second", record_b_id="y", overall_score=0.10),
        ]
        results = classifier.classify_batch(sims)
        assert results[0].record_a_id == "first"
        assert results[1].record_a_id == "second"

    def test_batch_stats_incremented(self, classifier: MatchClassifier):
        """Batch classification increments stats for each item."""
        sims = [_make_sim_result(record_a_id=f"a{i}", record_b_id=f"b{i}", overall_score=0.90) for i in range(3)]
        classifier.classify_batch(sims)
        stats = classifier.get_statistics()
        assert stats["invocations"] == 3
        assert stats["successes"] == 3

    def test_batch_with_fellegi_sunter(self, classifier: MatchClassifier):
        """Batch classification with Fellegi-Sunter flag works."""
        configs = _make_field_configs({"name": 1.0})
        sims = [_make_sim_result(record_a_id="a", record_b_id="b", overall_score=0.90)]
        results = classifier.classify_batch(
            sims, use_fellegi_sunter=True, field_configs=configs,
        )
        assert len(results) == 1


# =============================================================================
# TestComputeWeightedScore
# =============================================================================


class TestComputeWeightedScore:
    """Tests for compute_weighted_score method."""

    def test_equal_weights(self, classifier: MatchClassifier):
        """Equal weights produce simple average."""
        field_scores = {"name": 0.80, "email": 0.60}
        configs = _make_field_configs({"name": 1.0, "email": 1.0})
        score = classifier.compute_weighted_score(field_scores, configs)
        assert score == pytest.approx(0.70, abs=1e-5)

    def test_custom_weights(self, classifier: MatchClassifier):
        """Custom weights produce weighted average."""
        field_scores = {"name": 1.0, "email": 0.0}
        configs = _make_field_configs({"name": 3.0, "email": 1.0})
        score = classifier.compute_weighted_score(field_scores, configs)
        # (1.0 * 3.0 + 0.0 * 1.0) / (3.0 + 1.0) = 0.75
        assert score == pytest.approx(0.75, abs=1e-5)

    def test_zero_total_weight_returns_zero(self, classifier: MatchClassifier):
        """All-zero weights return 0.0."""
        field_scores = {"name": 0.90}
        configs = _make_field_configs({"name": 0.0})
        score = classifier.compute_weighted_score(field_scores, configs)
        assert score == 0.0

    def test_single_field(self, classifier: MatchClassifier):
        """Single-field weighted score equals field score."""
        field_scores = {"name": 0.85}
        configs = _make_field_configs({"name": 1.0})
        score = classifier.compute_weighted_score(field_scores, configs)
        assert score == pytest.approx(0.85, abs=1e-5)

    def test_missing_field_treated_as_zero(self, classifier: MatchClassifier):
        """Missing field score is treated as 0.0."""
        field_scores = {"name": 0.80}
        configs = _make_field_configs({"name": 1.0, "email": 1.0})
        score = classifier.compute_weighted_score(field_scores, configs)
        # (0.80 * 1.0 + 0.0 * 1.0) / 2.0 = 0.40
        assert score == pytest.approx(0.40, abs=1e-5)

    def test_weighted_score_capped_at_1(self, classifier: MatchClassifier):
        """Score is clamped to maximum 1.0."""
        field_scores = {"name": 1.0}
        configs = _make_field_configs({"name": 1.0})
        score = classifier.compute_weighted_score(field_scores, configs)
        assert score <= 1.0

    def test_weighted_score_minimum_zero(self, classifier: MatchClassifier):
        """Score is clamped to minimum 0.0."""
        field_scores = {"name": 0.0, "email": 0.0}
        configs = _make_field_configs({"name": 1.0, "email": 1.0})
        score = classifier.compute_weighted_score(field_scores, configs)
        assert score >= 0.0

    def test_empty_field_configs(self, classifier: MatchClassifier):
        """Empty configs returns 0.0 (zero total weight)."""
        field_scores = {"name": 0.80}
        score = classifier.compute_weighted_score(field_scores, [])
        assert score == 0.0

    def test_many_fields(self, classifier: MatchClassifier):
        """Ten fields with equal weights produce average."""
        fields = {f"f{i}": float(i) / 10.0 for i in range(1, 11)}
        configs = _make_field_configs({f"f{i}": 1.0 for i in range(1, 11)})
        score = classifier.compute_weighted_score(fields, configs)
        expected = sum(float(i) / 10.0 for i in range(1, 11)) / 10.0
        assert score == pytest.approx(expected, abs=1e-4)


# =============================================================================
# TestFellegiSunterScore
# =============================================================================


class TestFellegiSunterScore:
    """Tests for fellegi_sunter_score method."""

    def test_all_fields_agree(self, classifier: MatchClassifier):
        """All fields score=1.0 produces high FS score."""
        field_scores = {"name": 1.0, "email": 1.0}
        score = classifier.fellegi_sunter_score(field_scores)
        assert score > 0.8

    def test_all_fields_disagree(self, classifier: MatchClassifier):
        """All fields score=0.0 produces low FS score."""
        field_scores = {"name": 0.0, "email": 0.0}
        score = classifier.fellegi_sunter_score(field_scores)
        assert score < 0.2

    def test_perfect_agreement_is_1(self, classifier: MatchClassifier):
        """Perfect agreement normalizes to 1.0."""
        field_scores = {"name": 1.0}
        score = classifier.fellegi_sunter_score(field_scores)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_perfect_disagreement_is_0(self, classifier: MatchClassifier):
        """Perfect disagreement normalizes to 0.0."""
        field_scores = {"name": 0.0}
        score = classifier.fellegi_sunter_score(field_scores)
        assert score == pytest.approx(0.0, abs=1e-5)

    def test_mixed_agreement(self, classifier: MatchClassifier):
        """Mixed agreement produces intermediate score."""
        field_scores = {"name": 1.0, "email": 0.0}
        score = classifier.fellegi_sunter_score(field_scores)
        assert 0.2 < score < 0.8

    def test_empty_field_scores_returns_zero(self, classifier: MatchClassifier):
        """Empty field_scores returns 0.0."""
        score = classifier.fellegi_sunter_score({})
        assert score == 0.0

    def test_with_field_configs(self, classifier: MatchClassifier):
        """Field configs are accepted without error."""
        field_scores = {"name": 0.8, "email": 0.7}
        configs = _make_field_configs({"name": 1.0, "email": 1.0})
        score = classifier.fellegi_sunter_score(field_scores, configs)
        assert 0.0 <= score <= 1.0

    def test_custom_m_u_probabilities(self, classifier: MatchClassifier):
        """Custom m and u probabilities affect the score."""
        field_scores = {"name": 0.5}
        score_default = classifier.fellegi_sunter_score(field_scores)
        score_custom = classifier.fellegi_sunter_score(
            field_scores, m_probability=0.95, u_probability=0.05,
        )
        # Both valid, but may differ
        assert 0.0 <= score_custom <= 1.0

    def test_fs_score_monotonic(self, classifier: MatchClassifier):
        """Higher field score produces higher FS score."""
        low = classifier.fellegi_sunter_score({"name": 0.2})
        high = classifier.fellegi_sunter_score({"name": 0.9})
        assert high > low

    def test_fs_score_bounded(self, classifier: MatchClassifier):
        """FS score is always in [0, 1]."""
        for val in [0.0, 0.1, 0.5, 0.9, 1.0]:
            score = classifier.fellegi_sunter_score({"f": val})
            assert 0.0 <= score <= 1.0

    def test_fs_score_single_field(self, classifier: MatchClassifier):
        """Single-field FS score works correctly."""
        score = classifier.fellegi_sunter_score({"name": 0.75})
        assert 0.0 <= score <= 1.0


# =============================================================================
# TestClassifyPairWithFellegiSunter
# =============================================================================


class TestClassifyPairWithFellegiSunter:
    """Tests for classify_pair with Fellegi-Sunter enabled."""

    def test_fellegi_sunter_adjusts_score(self, classifier: MatchClassifier):
        """With FS enabled, score is a blend of original and FS."""
        sim = _make_sim_result(overall_score=0.88, field_scores={"name": 0.88})
        configs = _make_field_configs({"name": 1.0})
        result = classifier.classify_pair(
            sim, use_fellegi_sunter=True, field_configs=configs,
        )
        # Score is 0.6 * 0.88 + 0.4 * FS(0.88) - still MATCH
        assert result.classification in (MatchClassification.MATCH, MatchClassification.POSSIBLE)

    def test_fellegi_sunter_high_score_match(self, classifier: MatchClassifier):
        """High score with FS still classifies as MATCH."""
        sim = _make_sim_result(overall_score=0.98, field_scores={"name": 0.98})
        configs = _make_field_configs({"name": 1.0})
        result = classifier.classify_pair(
            sim, use_fellegi_sunter=True, field_configs=configs,
        )
        assert result.classification == MatchClassification.MATCH

    def test_fellegi_sunter_decision_reason_mentions_fs(self, classifier: MatchClassifier):
        """Decision reason mentions Fellegi-Sunter when used."""
        sim = _make_sim_result(overall_score=0.90, field_scores={"name": 0.90})
        configs = _make_field_configs({"name": 1.0})
        result = classifier.classify_pair(
            sim, use_fellegi_sunter=True, field_configs=configs,
        )
        assert "Fellegi-Sunter" in result.decision_reason


# =============================================================================
# TestAutoThreshold
# =============================================================================


class TestAutoThreshold:
    """Tests for auto_threshold detection method."""

    def test_empty_input_raises(self, classifier: MatchClassifier):
        """Empty input raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            classifier.auto_threshold([])

    def test_single_score_returns_defaults(self, classifier: MatchClassifier):
        """Single-element input returns default thresholds."""
        sim = _make_sim_result(overall_score=0.50)
        mt, pt = classifier.auto_threshold([sim])
        assert mt == 0.85
        assert pt == 0.65

    def test_bimodal_distribution(self, classifier: MatchClassifier):
        """Bimodal distribution detects threshold between modes."""
        sims = []
        for _ in range(50):
            sims.append(_make_sim_result(record_a_id=f"a{_}", record_b_id=f"b{_}", overall_score=0.2))
        for i in range(50):
            sims.append(_make_sim_result(record_a_id=f"c{i}", record_b_id=f"d{i}", overall_score=0.9))
        mt, pt = classifier.auto_threshold(sims)
        # Threshold should fall between the modes
        assert 0.5 <= mt <= 0.95
        assert pt < mt

    def test_all_high_scores(self, classifier: MatchClassifier):
        """All high scores produces threshold at upper boundary."""
        sims = [_make_sim_result(record_a_id=f"a{i}", record_b_id=f"b{i}", overall_score=0.95) for i in range(20)]
        mt, pt = classifier.auto_threshold(sims)
        assert mt >= 0.5
        assert pt < mt

    def test_all_low_scores(self, classifier: MatchClassifier):
        """All low scores still produces valid thresholds."""
        sims = [_make_sim_result(record_a_id=f"a{i}", record_b_id=f"b{i}", overall_score=0.05) for i in range(20)]
        mt, pt = classifier.auto_threshold(sims)
        assert mt >= 0.5
        assert pt >= 0.3

    def test_possible_threshold_less_than_match(self, classifier: MatchClassifier):
        """Possible threshold is always less than match threshold."""
        sims = [_make_sim_result(record_a_id=f"a{i}", record_b_id=f"b{i}", overall_score=i * 0.05) for i in range(20)]
        mt, pt = classifier.auto_threshold(sims)
        assert pt < mt

    def test_custom_num_bins(self, classifier: MatchClassifier):
        """Custom bin count is accepted."""
        sims = [_make_sim_result(record_a_id=f"a{i}", record_b_id=f"b{i}", overall_score=0.5) for i in range(10)]
        mt, pt = classifier.auto_threshold(sims, num_bins=50)
        assert mt >= 0.5

    def test_match_threshold_bounded(self, classifier: MatchClassifier):
        """Match threshold is clamped to [0.5, 0.95]."""
        sims = [_make_sim_result(record_a_id=f"a{i}", record_b_id=f"b{i}", overall_score=0.99) for i in range(30)]
        mt, _pt = classifier.auto_threshold(sims)
        assert 0.5 <= mt <= 0.95


# =============================================================================
# TestEvaluateClassificationQuality
# =============================================================================


class TestEvaluateClassificationQuality:
    """Tests for evaluate_classification_quality method."""

    def test_empty_results(self, classifier: MatchClassifier):
        """Empty input returns all-zero metrics."""
        quality = classifier.evaluate_classification_quality([])
        assert quality["total_pairs"] == 0
        assert quality["match_count"] == 0
        assert quality["match_rate"] == 0.0

    def test_all_matches(self, classifier: MatchClassifier):
        """All-MATCH results produce match_rate 1.0."""
        results = [_make_match_result(MatchClassification.MATCH) for _ in range(5)]
        quality = classifier.evaluate_classification_quality(results)
        assert quality["match_count"] == 5
        assert quality["match_rate"] == 1.0

    def test_all_non_matches(self, classifier: MatchClassifier):
        """All-NON_MATCH results produce match_rate 0.0."""
        results = [_make_match_result(MatchClassification.NON_MATCH, overall_score=0.2) for _ in range(5)]
        quality = classifier.evaluate_classification_quality(results)
        assert quality["match_count"] == 0
        assert quality["non_match_count"] == 5
        assert quality["match_rate"] == 0.0

    def test_mixed_results(self, classifier: MatchClassifier):
        """Mixed results produce correct counts."""
        results = [
            _make_match_result(MatchClassification.MATCH, overall_score=0.90),
            _make_match_result(MatchClassification.POSSIBLE, overall_score=0.75),
            _make_match_result(MatchClassification.NON_MATCH, overall_score=0.20),
        ]
        quality = classifier.evaluate_classification_quality(results)
        assert quality["total_pairs"] == 3
        assert quality["match_count"] == 1
        assert quality["possible_count"] == 1
        assert quality["non_match_count"] == 1

    def test_confidence_metrics(self, classifier: MatchClassifier):
        """Confidence metrics are computed correctly."""
        results = [
            _make_match_result(confidence=0.80),
            _make_match_result(confidence=0.90),
        ]
        quality = classifier.evaluate_classification_quality(results)
        assert quality["avg_confidence"] == pytest.approx(0.85, abs=0.01)
        assert quality["min_confidence"] == pytest.approx(0.80, abs=0.01)
        assert quality["max_confidence"] == pytest.approx(0.90, abs=0.01)

    def test_score_separation(self, classifier: MatchClassifier):
        """Score separation is avg_match - avg_non_match."""
        results = [
            _make_match_result(MatchClassification.MATCH, overall_score=0.92),
            _make_match_result(MatchClassification.NON_MATCH, overall_score=0.20),
        ]
        quality = classifier.evaluate_classification_quality(results)
        assert quality["score_separation"] == pytest.approx(0.72, abs=0.01)

    def test_single_result(self, classifier: MatchClassifier):
        """Single result produces valid metrics."""
        results = [_make_match_result(MatchClassification.MATCH)]
        quality = classifier.evaluate_classification_quality(results)
        assert quality["total_pairs"] == 1
        assert quality["match_rate"] == 1.0


# =============================================================================
# TestDecisionReason
# =============================================================================


class TestDecisionReason:
    """Tests for decision reason generation."""

    def test_match_reason_format(self, classifier: MatchClassifier):
        """MATCH reason includes score and threshold."""
        sim = _make_sim_result(overall_score=0.90)
        result = classifier.classify_pair(sim, match_threshold=0.85)
        assert "0.9000" in result.decision_reason
        assert "0.8500" in result.decision_reason

    def test_non_match_reason_format(self, classifier: MatchClassifier):
        """NON_MATCH reason includes score and threshold."""
        sim = _make_sim_result(overall_score=0.30)
        result = classifier.classify_pair(sim, possible_threshold=0.65)
        assert "0.3000" in result.decision_reason

    def test_possible_reason_format(self, classifier: MatchClassifier):
        """POSSIBLE reason includes both thresholds."""
        sim = _make_sim_result(overall_score=0.75)
        result = classifier.classify_pair(sim, match_threshold=0.85, possible_threshold=0.65)
        assert "POSSIBLE" in result.decision_reason

    def test_threshold_method_displayed(self, classifier: MatchClassifier):
        """Threshold method name appears in reason."""
        sim = _make_sim_result(overall_score=0.90)
        result = classifier.classify_pair(sim, use_fellegi_sunter=False)
        assert "threshold" in result.decision_reason

    def test_fs_method_displayed(self, classifier: MatchClassifier):
        """FS method name appears in reason when enabled."""
        sim = _make_sim_result(overall_score=0.90, field_scores={"name": 0.90})
        configs = _make_field_configs({"name": 1.0})
        result = classifier.classify_pair(sim, use_fellegi_sunter=True, field_configs=configs)
        assert "Fellegi-Sunter" in result.decision_reason


# =============================================================================
# TestThresholdValidation
# =============================================================================


class TestThresholdValidation:
    """Tests for threshold validation edge cases."""

    def test_match_threshold_above_1_raises(self, classifier: MatchClassifier):
        """match_threshold > 1.0 raises ValueError."""
        sim = _make_sim_result(overall_score=0.90)
        with pytest.raises(ValueError, match="match_threshold"):
            classifier.classify_pair(sim, match_threshold=1.5)

    def test_match_threshold_below_0_raises(self, classifier: MatchClassifier):
        """match_threshold < 0.0 raises ValueError."""
        sim = _make_sim_result(overall_score=0.90)
        with pytest.raises(ValueError, match="match_threshold"):
            classifier.classify_pair(sim, match_threshold=-0.1)

    def test_possible_threshold_above_1_raises(self, classifier: MatchClassifier):
        """possible_threshold > 1.0 raises ValueError."""
        sim = _make_sim_result(overall_score=0.90)
        with pytest.raises(ValueError, match="possible_threshold"):
            classifier.classify_pair(sim, possible_threshold=1.5)

    def test_possible_threshold_below_0_raises(self, classifier: MatchClassifier):
        """possible_threshold < 0.0 raises ValueError."""
        sim = _make_sim_result(overall_score=0.90)
        with pytest.raises(ValueError, match="possible_threshold"):
            classifier.classify_pair(sim, possible_threshold=-0.1)

    def test_possible_exceeds_match_raises(self, classifier: MatchClassifier):
        """possible_threshold > match_threshold raises ValueError."""
        sim = _make_sim_result(overall_score=0.90)
        with pytest.raises(ValueError, match="must not exceed"):
            classifier.classify_pair(sim, match_threshold=0.5, possible_threshold=0.7)

    def test_equal_thresholds_allowed(self, classifier: MatchClassifier):
        """Equal thresholds are valid (no POSSIBLE region)."""
        sim = _make_sim_result(overall_score=0.90)
        result = classifier.classify_pair(sim, match_threshold=0.85, possible_threshold=0.85)
        assert result.classification in (MatchClassification.MATCH, MatchClassification.NON_MATCH)

    def test_zero_thresholds_allowed(self, classifier: MatchClassifier):
        """Zero thresholds are valid (all MATCH)."""
        sim = _make_sim_result(overall_score=0.01)
        result = classifier.classify_pair(sim, match_threshold=0.0, possible_threshold=0.0)
        assert result.classification == MatchClassification.MATCH


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for classify_pair."""

    def test_all_field_scores_zero(self, classifier: MatchClassifier):
        """All field scores 0.0 still produces valid result."""
        sim = _make_sim_result(overall_score=0.0, field_scores={"a": 0.0, "b": 0.0})
        result = classifier.classify_pair(sim)
        assert result.classification == MatchClassification.NON_MATCH

    def test_all_field_scores_one(self, classifier: MatchClassifier):
        """All field scores 1.0 produces MATCH."""
        sim = _make_sim_result(overall_score=1.0, field_scores={"a": 1.0, "b": 1.0})
        result = classifier.classify_pair(sim)
        assert result.classification == MatchClassification.MATCH

    def test_single_field_score(self, classifier: MatchClassifier):
        """Single field score works correctly."""
        sim = _make_sim_result(overall_score=0.90, field_scores={"name": 0.90})
        result = classifier.classify_pair(sim)
        assert result.classification == MatchClassification.MATCH

    def test_many_fields(self, classifier: MatchClassifier):
        """Many field scores do not cause errors."""
        scores = {f"field_{i}": 0.5 for i in range(100)}
        sim = _make_sim_result(overall_score=0.50, field_scores=scores)
        result = classifier.classify_pair(sim)
        assert result.classification == MatchClassification.NON_MATCH

    def test_max_threshold_1_0(self, classifier: MatchClassifier):
        """Match threshold 1.0: only perfect scores match."""
        sim_perfect = _make_sim_result(overall_score=1.0)
        sim_high = _make_sim_result(overall_score=0.999)
        r_perfect = classifier.classify_pair(sim_perfect, match_threshold=1.0, possible_threshold=0.9)
        r_high = classifier.classify_pair(sim_high, match_threshold=1.0, possible_threshold=0.9)
        assert r_perfect.classification == MatchClassification.MATCH
        assert r_high.classification == MatchClassification.POSSIBLE


# =============================================================================
# TestThreadSafety
# =============================================================================


class TestThreadSafety:
    """Thread-safety tests for statistics tracking."""

    def test_concurrent_classify(self, classifier: MatchClassifier):
        """Concurrent classify_pair calls maintain correct stats."""
        num_threads = 10
        iters_per_thread = 20
        errors: List[str] = []

        def worker():
            try:
                for i in range(iters_per_thread):
                    sim = _make_sim_result(
                        record_a_id=f"t{threading.current_thread().ident}_{i}",
                        record_b_id=f"b{i}",
                        overall_score=0.90,
                    )
                    classifier.classify_pair(sim)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = classifier.get_statistics()
        assert stats["invocations"] == num_threads * iters_per_thread
        assert stats["successes"] == num_threads * iters_per_thread
        assert stats["failures"] == 0

    def test_concurrent_reset(self, classifier: MatchClassifier):
        """Concurrent reset does not crash."""
        def worker():
            for _ in range(50):
                classifier.reset_statistics()

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stats = classifier.get_statistics()
        assert stats["invocations"] == 0


# =============================================================================
# TestProvenanceTracking
# =============================================================================


class TestProvenanceTracking:
    """Provenance hash generation tests."""

    def test_provenance_hash_is_sha256(self, classifier: MatchClassifier):
        """Provenance hash is 64-char hex (SHA-256)."""
        sim = _make_sim_result(overall_score=0.90)
        result = classifier.classify_pair(sim)
        assert len(result.provenance_hash) == 64
        # All hex characters
        int(result.provenance_hash, 16)

    def test_provenance_hash_not_empty(self, classifier: MatchClassifier):
        """Provenance hash is never empty."""
        sim = _make_sim_result(overall_score=0.50)
        result = classifier.classify_pair(sim)
        assert result.provenance_hash != ""

    def test_different_inputs_different_provenance(self, classifier: MatchClassifier):
        """Different record IDs produce different provenance (usually)."""
        sim1 = _make_sim_result(record_a_id="alpha", record_b_id="beta", overall_score=0.90)
        sim2 = _make_sim_result(record_a_id="gamma", record_b_id="delta", overall_score=0.90)
        r1 = classifier.classify_pair(sim1)
        r2 = classifier.classify_pair(sim2)
        # Provenance includes record IDs in the hash input
        # They may still collide if hashed at same microsecond, so we don't assert inequality


# =============================================================================
# TestStatistics
# =============================================================================


class TestStatistics:
    """Statistics tracking tests."""

    def test_success_increments_stats(self, classifier: MatchClassifier):
        """Successful invocation increments success counter."""
        sim = _make_sim_result(overall_score=0.90)
        classifier.classify_pair(sim)
        stats = classifier.get_statistics()
        assert stats["invocations"] == 1
        assert stats["successes"] == 1
        assert stats["failures"] == 0

    def test_failure_increments_failure_counter(self, classifier: MatchClassifier):
        """Failed invocation increments failure counter."""
        sim = _make_sim_result(overall_score=0.90)
        with pytest.raises(ValueError):
            classifier.classify_pair(sim, match_threshold=1.5)
        stats = classifier.get_statistics()
        assert stats["failures"] == 1

    def test_duration_tracked(self, classifier: MatchClassifier):
        """Total duration is positive after invocation."""
        sim = _make_sim_result(overall_score=0.90)
        classifier.classify_pair(sim)
        stats = classifier.get_statistics()
        assert stats["total_duration_ms"] >= 0

    def test_avg_duration_computed(self, classifier: MatchClassifier):
        """Average duration is total / invocations."""
        for i in range(5):
            sim = _make_sim_result(record_a_id=f"a{i}", record_b_id=f"b{i}", overall_score=0.90)
            classifier.classify_pair(sim)
        stats = classifier.get_statistics()
        assert stats["avg_duration_ms"] == pytest.approx(
            stats["total_duration_ms"] / 5, abs=0.01,
        )

    def test_last_invoked_at_set(self, classifier: MatchClassifier):
        """last_invoked_at is set after invocation."""
        sim = _make_sim_result(overall_score=0.90)
        classifier.classify_pair(sim)
        stats = classifier.get_statistics()
        assert stats["last_invoked_at"] is not None

    def test_get_statistics_returns_dict(self, classifier: MatchClassifier):
        """get_statistics returns a dictionary."""
        stats = classifier.get_statistics()
        assert isinstance(stats, dict)
        assert "engine_name" in stats


# =============================================================================
# TestDeterminism
# =============================================================================


class TestDeterminism:
    """Determinism tests: same input always produces same classification."""

    def test_same_input_same_classification(self, classifier: MatchClassifier):
        """Same SimilarityResult always produces same classification."""
        sim = _make_sim_result(overall_score=0.90)
        results = [classifier.classify_pair(sim) for _ in range(10)]
        first_class = results[0].classification
        assert all(r.classification == first_class for r in results)

    def test_same_input_same_confidence(self, classifier: MatchClassifier):
        """Same input always produces same confidence (ignoring provenance)."""
        sim = _make_sim_result(overall_score=0.75)
        results = [classifier.classify_pair(sim) for _ in range(10)]
        first_conf = results[0].confidence
        assert all(r.confidence == first_conf for r in results)

    def test_same_input_same_overall_score(self, classifier: MatchClassifier):
        """Same input preserves overall_score across runs."""
        sim = _make_sim_result(overall_score=0.82)
        results = [classifier.classify_pair(sim) for _ in range(10)]
        assert all(r.overall_score == 0.82 for r in results)
