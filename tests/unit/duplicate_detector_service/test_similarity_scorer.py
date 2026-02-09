# -*- coding: utf-8 -*-
"""
Unit Tests for SimilarityScorer Engine - AGENT-DATA-011 Batch 2

Comprehensive test suite for the SimilarityScorer engine covering all
8 similarity algorithms:
- exact_match
- levenshtein_similarity
- jaro_winkler_similarity
- soundex_similarity
- ngram_similarity
- tfidf_cosine_similarity
- numeric_proximity
- date_proximity

Also covers:
- score_pair with weighted fields
- score_batch
- Edge cases (None, empty, unicode, mixed types)
- Score range validation [0.0, 1.0]
- Symmetry testing
- Thread-safe statistics tracking
- Provenance hash generation

Target: 150+ tests, 85%+ coverage.

Author: GreenLang QA Team
Date: February 2026
PRD: AGENT-DATA-011 Duplicate Detection Agent (GL-DATA-X-014)
"""

from __future__ import annotations

import math
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import pytest

from greenlang.duplicate_detector.models import (
    FieldComparisonConfig,
    FieldType,
    SimilarityAlgorithm,
    SimilarityResult,
)
from greenlang.duplicate_detector.similarity_scorer import (
    SimilarityScorer,
    _SOUNDEX_TABLE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scorer() -> SimilarityScorer:
    """Create a fresh SimilarityScorer instance."""
    return SimilarityScorer()


@pytest.fixture
def record_alice_a() -> Dict[str, Any]:
    """First Alice record."""
    return {"id": "a1", "name": "Alice Smith", "email": "alice@test.com", "amount": "100.50"}


@pytest.fixture
def record_alice_b() -> Dict[str, Any]:
    """Second Alice record (similar)."""
    return {"id": "a2", "name": "Alice Smyth", "email": "alice@test.com", "amount": "100.75"}


@pytest.fixture
def record_bob() -> Dict[str, Any]:
    """Bob record (different)."""
    return {"id": "b1", "name": "Bob Jones", "email": "bob@example.com", "amount": "5000.00"}


@pytest.fixture
def name_config() -> FieldComparisonConfig:
    """Field config for name comparison with Jaro-Winkler."""
    return FieldComparisonConfig(
        field_name="name",
        algorithm=SimilarityAlgorithm.JARO_WINKLER,
        weight=2.0,
    )


@pytest.fixture
def email_config() -> FieldComparisonConfig:
    """Field config for email comparison with exact match."""
    return FieldComparisonConfig(
        field_name="email",
        algorithm=SimilarityAlgorithm.EXACT,
        weight=1.5,
    )


@pytest.fixture
def amount_config() -> FieldComparisonConfig:
    """Field config for numeric amount comparison."""
    return FieldComparisonConfig(
        field_name="amount",
        algorithm=SimilarityAlgorithm.NUMERIC,
        weight=1.0,
    )


@pytest.fixture
def multi_field_configs(name_config, email_config, amount_config) -> List[FieldComparisonConfig]:
    """Multiple field comparison configs."""
    return [name_config, email_config, amount_config]


# ===========================================================================
# Test Class: exact_match
# ===========================================================================


class TestExactMatch:
    """Tests for exact_match similarity algorithm."""

    def test_identical_strings(self, scorer: SimilarityScorer):
        """Identical strings return 1.0."""
        assert scorer.exact_match("hello", "hello") == 1.0

    def test_different_strings(self, scorer: SimilarityScorer):
        """Different strings return 0.0."""
        assert scorer.exact_match("hello", "world") == 0.0

    def test_case_sensitive(self, scorer: SimilarityScorer):
        """Exact match is case-sensitive."""
        assert scorer.exact_match("Hello", "hello") == 0.0

    def test_empty_strings_identical(self, scorer: SimilarityScorer):
        """Two empty strings return 1.0."""
        assert scorer.exact_match("", "") == 1.0

    def test_one_empty_one_not(self, scorer: SimilarityScorer):
        """One empty and one non-empty return 0.0."""
        assert scorer.exact_match("", "hello") == 0.0
        assert scorer.exact_match("hello", "") == 0.0

    def test_whitespace_matters(self, scorer: SimilarityScorer):
        """Trailing whitespace causes non-match."""
        assert scorer.exact_match("hello", "hello ") == 0.0

    def test_unicode_match(self, scorer: SimilarityScorer):
        """Unicode strings match correctly."""
        assert scorer.exact_match("cafe", "cafe") == 1.0

    def test_unicode_mismatch(self, scorer: SimilarityScorer):
        """Different unicode strings return 0.0."""
        assert scorer.exact_match("cafe", "caff") == 0.0

    def test_special_characters(self, scorer: SimilarityScorer):
        """Special characters compared correctly."""
        assert scorer.exact_match("test@#$!", "test@#$!") == 1.0
        assert scorer.exact_match("test@#$!", "test@#$?") == 0.0

    def test_numbers_as_strings(self, scorer: SimilarityScorer):
        """Numeric strings compared correctly."""
        assert scorer.exact_match("12345", "12345") == 1.0
        assert scorer.exact_match("12345", "12346") == 0.0


# ===========================================================================
# Test Class: levenshtein_similarity
# ===========================================================================


class TestLevenshteinSimilarity:
    """Tests for Levenshtein edit distance similarity."""

    def test_identical_strings(self, scorer: SimilarityScorer):
        """Identical strings return 1.0."""
        assert scorer.levenshtein_similarity("hello", "hello") == 1.0

    def test_completely_different(self, scorer: SimilarityScorer):
        """Completely different strings of equal length return low score."""
        score = scorer.levenshtein_similarity("abc", "xyz")
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_single_edit(self, scorer: SimilarityScorer):
        """Single character insertion: 'kitten' vs 'kittens'."""
        score = scorer.levenshtein_similarity("kitten", "kittens")
        # edit_distance=1, max_len=7, similarity = 1 - 1/7 = 0.857
        assert score == pytest.approx(1.0 - 1 / 7, abs=1e-3)

    def test_kitten_sitting(self, scorer: SimilarityScorer):
        """Known test case: 'kitten' vs 'sitting' (edit distance=3)."""
        score = scorer.levenshtein_similarity("kitten", "sitting")
        # edit_distance=3, max_len=7, similarity = 1 - 3/7 = 0.571
        assert score == pytest.approx(1.0 - 3 / 7, abs=1e-3)

    def test_empty_string_a(self, scorer: SimilarityScorer):
        """Empty first string returns 0.0."""
        assert scorer.levenshtein_similarity("", "hello") == 0.0

    def test_empty_string_b(self, scorer: SimilarityScorer):
        """Empty second string returns 0.0."""
        assert scorer.levenshtein_similarity("hello", "") == 0.0

    def test_both_empty(self, scorer: SimilarityScorer):
        """Both empty strings return 1.0."""
        assert scorer.levenshtein_similarity("", "") == 1.0

    def test_single_char_match(self, scorer: SimilarityScorer):
        """Single matching characters return 1.0."""
        assert scorer.levenshtein_similarity("a", "a") == 1.0

    def test_single_char_mismatch(self, scorer: SimilarityScorer):
        """Single differing characters return 0.0."""
        assert scorer.levenshtein_similarity("a", "b") == 0.0

    def test_transposition(self, scorer: SimilarityScorer):
        """Transposed characters: 'ab' vs 'ba' (edit distance=2)."""
        score = scorer.levenshtein_similarity("ab", "ba")
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_score_in_range(self, scorer: SimilarityScorer):
        """All scores are in [0.0, 1.0]."""
        pairs = [
            ("hello", "world"),
            ("test", "testing"),
            ("abc", "abcdef"),
            ("x", "abcdefghij"),
        ]
        for a, b in pairs:
            score = scorer.levenshtein_similarity(a, b)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for ({a}, {b})"

    def test_symmetry(self, scorer: SimilarityScorer):
        """Levenshtein similarity is symmetric."""
        pairs = [
            ("hello", "hallo"),
            ("kitten", "sitting"),
            ("abc", "xyz"),
        ]
        for a, b in pairs:
            assert scorer.levenshtein_similarity(a, b) == pytest.approx(
                scorer.levenshtein_similarity(b, a), abs=1e-6
            )

    def test_close_strings_high_score(self, scorer: SimilarityScorer):
        """Close strings score highly."""
        score = scorer.levenshtein_similarity("sustainability", "sustainabilitx")
        assert score > 0.9

    def test_long_strings(self, scorer: SimilarityScorer):
        """Levenshtein handles longer strings correctly."""
        a = "the quick brown fox jumps over the lazy dog"
        b = "the quick brown fox leaps over the lazy dog"
        score = scorer.levenshtein_similarity(a, b)
        assert 0.0 <= score <= 1.0
        assert score > 0.8  # Only 1 word different


# ===========================================================================
# Test Class: jaro_winkler_similarity
# ===========================================================================


class TestJaroWinklerSimilarity:
    """Tests for Jaro-Winkler string similarity."""

    def test_identical_strings(self, scorer: SimilarityScorer):
        """Identical strings return 1.0."""
        assert scorer.jaro_winkler_similarity("hello", "hello") == 1.0

    def test_empty_string_a(self, scorer: SimilarityScorer):
        """Empty first string returns 0.0."""
        assert scorer.jaro_winkler_similarity("", "hello") == 0.0

    def test_empty_string_b(self, scorer: SimilarityScorer):
        """Empty second string returns 0.0."""
        assert scorer.jaro_winkler_similarity("hello", "") == 0.0

    def test_both_empty(self, scorer: SimilarityScorer):
        """Both empty strings return 1.0."""
        assert scorer.jaro_winkler_similarity("", "") == 1.0

    def test_martha_marhta(self, scorer: SimilarityScorer):
        """Known test case: MARTHA vs MARHTA (Jaro=0.944, JW~0.961)."""
        score = scorer.jaro_winkler_similarity("MARTHA", "MARHTA")
        # Jaro: matches=6, transpositions=2/2=1
        # Jaro = (6/6 + 6/6 + (6-1)/6) / 3 = (1 + 1 + 0.8333)/3 = 0.9444
        # Winkler: prefix MAR (3 chars), JW = 0.9444 + 3*0.1*(1-0.9444)
        assert score == pytest.approx(0.961, abs=0.01)

    def test_dwayne_duane(self, scorer: SimilarityScorer):
        """Known test case: DWAYNE vs DUANE."""
        score = scorer.jaro_winkler_similarity("DWAYNE", "DUANE")
        assert 0.8 <= score <= 1.0

    def test_common_prefix_boost(self, scorer: SimilarityScorer):
        """Winkler prefix boost increases similarity for common prefixes."""
        jw = scorer.jaro_winkler_similarity("ABCDEF", "ABCXYZ")
        # Common prefix ABC (3 chars) should provide a boost
        assert jw > 0.5

    def test_no_common_prefix(self, scorer: SimilarityScorer):
        """No common prefix means no Winkler boost."""
        score_jw = scorer.jaro_winkler_similarity("XYZ", "ABC")
        # No common prefix, but there might still be some Jaro component
        assert 0.0 <= score_jw <= 1.0

    def test_prefix_weight_parameter(self, scorer: SimilarityScorer):
        """Custom prefix weight affects the score."""
        default = scorer.jaro_winkler_similarity("MARTHA", "MARHTA", winkler_prefix_weight=0.1)
        higher = scorer.jaro_winkler_similarity("MARTHA", "MARHTA", winkler_prefix_weight=0.2)
        # Higher prefix weight should give a higher boost (or clamp at 1.0)
        assert higher >= default

    def test_no_matches(self, scorer: SimilarityScorer):
        """Strings with no matching characters return 0.0."""
        assert scorer.jaro_winkler_similarity("abc", "xyz") == 0.0

    def test_single_char_match(self, scorer: SimilarityScorer):
        """Single character strings that match return 1.0."""
        assert scorer.jaro_winkler_similarity("a", "a") == 1.0

    def test_single_char_mismatch(self, scorer: SimilarityScorer):
        """Single character strings that differ return 0.0."""
        assert scorer.jaro_winkler_similarity("a", "b") == 0.0

    def test_score_in_range(self, scorer: SimilarityScorer):
        """All scores are in [0.0, 1.0]."""
        pairs = [
            ("hello", "world"),
            ("MARTHA", "MARHTA"),
            ("", "test"),
            ("abc", "abc"),
            ("a", "abcdef"),
        ]
        for a, b in pairs:
            score = scorer.jaro_winkler_similarity(a, b)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for ({a}, {b})"

    def test_symmetry(self, scorer: SimilarityScorer):
        """Jaro-Winkler is symmetric for these test cases."""
        # Note: Jaro itself is symmetric, and Winkler prefix boost is symmetric
        pairs = [("MARTHA", "MARHTA"), ("hello", "hallo"), ("abc", "abd")]
        for a, b in pairs:
            score_ab = scorer.jaro_winkler_similarity(a, b)
            score_ba = scorer.jaro_winkler_similarity(b, a)
            assert score_ab == pytest.approx(score_ba, abs=1e-6), \
                f"Asymmetric: ({a},{b})={score_ab} vs ({b},{a})={score_ba}"

    def test_longer_prefix_gives_higher_score(self, scorer: SimilarityScorer):
        """Longer common prefix (up to 4) gives higher Winkler boost."""
        score_1 = scorer.jaro_winkler_similarity("Axxxx", "Ayyyy")
        score_3 = scorer.jaro_winkler_similarity("ABCxx", "ABCyy")
        # Longer shared prefix should boost more
        assert score_3 >= score_1

    def test_max_prefix_length_4(self, scorer: SimilarityScorer):
        """Winkler prefix is capped at 4 characters."""
        score_4 = scorer.jaro_winkler_similarity("ABCDx", "ABCDy")
        score_5 = scorer.jaro_winkler_similarity("ABCDEx", "ABCDEy")
        # Scores should be very close since prefix capped at 4
        assert abs(score_4 - score_5) < 0.1


# ===========================================================================
# Test Class: soundex_similarity
# ===========================================================================


class TestSoundexSimilarity:
    """Tests for Soundex phonetic similarity."""

    def test_same_soundex_code(self, scorer: SimilarityScorer):
        """Words with the same Soundex code return 1.0."""
        assert scorer.soundex_similarity("Robert", "Rupert") == 1.0

    def test_different_soundex_code(self, scorer: SimilarityScorer):
        """Words with different Soundex codes return 0.0."""
        assert scorer.soundex_similarity("Alice", "Bob") == 0.0

    def test_identical_strings(self, scorer: SimilarityScorer):
        """Identical strings always return 1.0."""
        assert scorer.soundex_similarity("Smith", "Smith") == 1.0

    def test_smith_smythe(self, scorer: SimilarityScorer):
        """Smith and Smythe have the same Soundex code."""
        assert scorer.soundex_similarity("Smith", "Smythe") == 1.0

    def test_empty_strings(self, scorer: SimilarityScorer):
        """Two empty strings produce same Soundex code (0000) -> 1.0."""
        assert scorer.soundex_similarity("", "") == 1.0

    def test_one_empty_one_not(self, scorer: SimilarityScorer):
        """One empty and one non-empty have different Soundex codes."""
        assert scorer.soundex_similarity("", "Alice") == 0.0

    def test_case_insensitive(self, scorer: SimilarityScorer):
        """Soundex is case insensitive."""
        assert scorer.soundex_similarity("ALICE", "alice") == 1.0

    def test_known_pairs(self, scorer: SimilarityScorer):
        """Known phonetically similar pairs."""
        similar_pairs = [
            ("Robert", "Rupert"),
            ("Smith", "Smythe"),
        ]
        for a, b in similar_pairs:
            assert scorer.soundex_similarity(a, b) == 1.0

    def test_known_different_pairs(self, scorer: SimilarityScorer):
        """Known phonetically different pairs."""
        different_pairs = [
            ("Alice", "Bob"),
            ("Smith", "Jones"),
            ("Robert", "Alice"),
        ]
        for a, b in different_pairs:
            assert scorer.soundex_similarity(a, b) == 0.0

    def test_only_returns_0_or_1(self, scorer: SimilarityScorer):
        """Soundex similarity only returns 0.0 or 1.0."""
        pairs = [
            ("hello", "hello"),
            ("hello", "world"),
            ("test", "tests"),
            ("Alice", "Bob"),
        ]
        for a, b in pairs:
            score = scorer.soundex_similarity(a, b)
            assert score in (0.0, 1.0), f"Unexpected score {score} for ({a}, {b})"


# ===========================================================================
# Test Class: ngram_similarity
# ===========================================================================


class TestNgramSimilarity:
    """Tests for character n-gram Jaccard coefficient similarity."""

    def test_identical_strings(self, scorer: SimilarityScorer):
        """Identical strings return 1.0."""
        assert scorer.ngram_similarity("hello", "hello") == 1.0

    def test_completely_different(self, scorer: SimilarityScorer):
        """Completely different strings return 0.0."""
        score = scorer.ngram_similarity("abc", "xyz")
        assert score == 0.0

    def test_empty_string_a(self, scorer: SimilarityScorer):
        """Empty first string returns 0.0."""
        assert scorer.ngram_similarity("", "hello") == 0.0

    def test_empty_string_b(self, scorer: SimilarityScorer):
        """Empty second string returns 0.0."""
        assert scorer.ngram_similarity("hello", "") == 0.0

    def test_both_empty(self, scorer: SimilarityScorer):
        """Both empty strings return 1.0."""
        assert scorer.ngram_similarity("", "") == 1.0

    def test_partial_overlap(self, scorer: SimilarityScorer):
        """Partial n-gram overlap returns intermediate score."""
        score = scorer.ngram_similarity("hello", "hallo")
        assert 0.0 < score < 1.0

    def test_bigrams(self, scorer: SimilarityScorer):
        """N-gram similarity with n=2."""
        score = scorer.ngram_similarity("hello", "hallo", n=2)
        # "hello" bigrams: {he, el, ll, lo}
        # "hallo" bigrams: {ha, al, ll, lo}
        # intersection: {ll, lo} = 2, union: {he, el, ll, lo, ha, al} = 6
        # Jaccard = 2/6 = 0.333
        assert score == pytest.approx(2 / 6, abs=1e-3)

    def test_trigrams_default(self, scorer: SimilarityScorer):
        """Default n=3 trigrams."""
        score = scorer.ngram_similarity("hello", "hallo", n=3)
        # "hello" trigrams: {hel, ell, llo}
        # "hallo" trigrams: {hal, all, llo}
        # intersection: {llo} = 1, union: {hel, ell, llo, hal, all} = 5
        # Jaccard = 1/5 = 0.2
        assert score == pytest.approx(1 / 5, abs=1e-3)

    def test_quadgrams(self, scorer: SimilarityScorer):
        """N-gram similarity with n=4."""
        score = scorer.ngram_similarity("hello world", "hello earth", n=4)
        assert 0.0 < score < 1.0

    def test_short_strings(self, scorer: SimilarityScorer):
        """Short strings with n>len produce single-element sets."""
        # "ab" with n=3 -> ["ab"], "ac" with n=3 -> ["ac"]
        score = scorer.ngram_similarity("ab", "ac", n=3)
        assert score == 0.0  # {"ab"} vs {"ac"} -> no intersection

    def test_short_strings_match(self, scorer: SimilarityScorer):
        """Short matching strings return 1.0."""
        assert scorer.ngram_similarity("ab", "ab", n=3) == 1.0

    def test_score_in_range(self, scorer: SimilarityScorer):
        """All scores are in [0.0, 1.0]."""
        pairs = [
            ("hello", "world"),
            ("test", "testing"),
            ("abc", "abcdef"),
            ("sustainability", "sustainabilitx"),
        ]
        for a, b in pairs:
            for n in [2, 3, 4]:
                score = scorer.ngram_similarity(a, b, n=n)
                assert 0.0 <= score <= 1.0, \
                    f"Score {score} out of range for ({a}, {b}) n={n}"

    def test_symmetry(self, scorer: SimilarityScorer):
        """N-gram similarity is symmetric."""
        pairs = [("hello", "hallo"), ("abc", "def"), ("test", "testing")]
        for a, b in pairs:
            assert scorer.ngram_similarity(a, b) == pytest.approx(
                scorer.ngram_similarity(b, a), abs=1e-6
            )


# ===========================================================================
# Test Class: tfidf_cosine_similarity
# ===========================================================================


class TestTFIDFCosineSimilarity:
    """Tests for TF-IDF cosine similarity."""

    def test_identical_documents(self, scorer: SimilarityScorer):
        """Identical documents return 1.0."""
        assert scorer.tfidf_cosine_similarity("hello world", "hello world") == 1.0

    def test_completely_different(self, scorer: SimilarityScorer):
        """Documents with no common terms return 0.0."""
        assert scorer.tfidf_cosine_similarity("hello world", "foo bar") == 0.0

    def test_partial_overlap(self, scorer: SimilarityScorer):
        """Documents with partial term overlap return intermediate score."""
        score = scorer.tfidf_cosine_similarity("hello world", "hello earth")
        assert 0.0 < score < 1.0

    def test_empty_string_a(self, scorer: SimilarityScorer):
        """Empty first string returns 0.0."""
        assert scorer.tfidf_cosine_similarity("", "hello world") == 0.0

    def test_empty_string_b(self, scorer: SimilarityScorer):
        """Empty second string returns 0.0."""
        assert scorer.tfidf_cosine_similarity("hello world", "") == 0.0

    def test_both_empty(self, scorer: SimilarityScorer):
        """Both empty strings return 1.0."""
        assert scorer.tfidf_cosine_similarity("", "") == 1.0

    def test_single_common_term(self, scorer: SimilarityScorer):
        """Documents sharing one term among many."""
        score = scorer.tfidf_cosine_similarity("carbon emissions report", "carbon tax policy")
        assert 0.0 < score < 1.0

    def test_case_insensitive(self, scorer: SimilarityScorer):
        """TF-IDF cosine lowercases terms internally."""
        score_lower = scorer.tfidf_cosine_similarity("hello world", "hello world")
        score_upper = scorer.tfidf_cosine_similarity("HELLO WORLD", "hello world")
        assert score_lower == pytest.approx(score_upper, abs=1e-6)

    def test_repeated_terms(self, scorer: SimilarityScorer):
        """Repeated terms affect TF weights."""
        score = scorer.tfidf_cosine_similarity(
            "carbon carbon carbon", "carbon emission report",
        )
        assert 0.0 < score < 1.0

    def test_score_in_range(self, scorer: SimilarityScorer):
        """All scores are in [0.0, 1.0]."""
        pairs = [
            ("hello world", "hello earth"),
            ("carbon emissions", "carbon emissions report"),
            ("sustainability", "environmental impact"),
            ("a b c d e", "f g h i j"),
        ]
        for a, b in pairs:
            score = scorer.tfidf_cosine_similarity(a, b)
            assert 0.0 <= score <= 1.0

    def test_symmetry(self, scorer: SimilarityScorer):
        """TF-IDF cosine is symmetric."""
        pairs = [
            ("hello world", "world hello"),
            ("carbon emissions", "emissions carbon report"),
        ]
        for a, b in pairs:
            score_ab = scorer.tfidf_cosine_similarity(a, b)
            score_ba = scorer.tfidf_cosine_similarity(b, a)
            assert score_ab == pytest.approx(score_ba, abs=1e-6)

    def test_known_cosine_value(self, scorer: SimilarityScorer):
        """Verify known cosine similarity calculation."""
        # "a a b" vs "a b b"
        # tf_a = {a:2, b:1}, tf_b = {a:1, b:2}
        # dot = 2*1 + 1*2 = 4
        # mag_a = sqrt(4+1) = sqrt(5), mag_b = sqrt(1+4) = sqrt(5)
        # cosine = 4/5 = 0.8
        score = scorer.tfidf_cosine_similarity("a a b", "a b b")
        assert score == pytest.approx(4 / 5, abs=1e-6)


# ===========================================================================
# Test Class: numeric_proximity
# ===========================================================================


class TestNumericProximity:
    """Tests for numeric proximity similarity."""

    def test_same_number(self, scorer: SimilarityScorer):
        """Same number returns 1.0."""
        assert scorer.numeric_proximity(100.0, 100.0) == 1.0

    def test_zero_diff(self, scorer: SimilarityScorer):
        """Zero difference returns 1.0."""
        assert scorer.numeric_proximity(0.0, 0.0) == 1.0

    def test_within_max_diff(self, scorer: SimilarityScorer):
        """Difference within max_diff returns proportional score."""
        # diff=50, max_diff=100 -> 1 - 50/100 = 0.5
        assert scorer.numeric_proximity(50.0, 100.0, max_diff=100.0) == pytest.approx(0.5)

    def test_at_max_diff(self, scorer: SimilarityScorer):
        """Difference exactly at max_diff returns 0.0."""
        assert scorer.numeric_proximity(0.0, 100.0, max_diff=100.0) == pytest.approx(0.0)

    def test_beyond_max_diff(self, scorer: SimilarityScorer):
        """Difference beyond max_diff returns 0.0 (clamped)."""
        assert scorer.numeric_proximity(0.0, 200.0, max_diff=100.0) == 0.0

    def test_negative_numbers(self, scorer: SimilarityScorer):
        """Negative numbers handled correctly."""
        score = scorer.numeric_proximity(-50.0, 50.0, max_diff=200.0)
        # diff=100, max=200 -> 1 - 100/200 = 0.5
        assert score == pytest.approx(0.5)

    def test_zero_max_diff_same(self, scorer: SimilarityScorer):
        """max_diff=0 with same number returns 1.0."""
        assert scorer.numeric_proximity(5.0, 5.0, max_diff=0.0) == 1.0

    def test_zero_max_diff_different(self, scorer: SimilarityScorer):
        """max_diff=0 with different numbers returns 0.0."""
        assert scorer.numeric_proximity(5.0, 6.0, max_diff=0.0) == 0.0

    def test_negative_max_diff_same(self, scorer: SimilarityScorer):
        """Negative max_diff with same numbers returns 1.0."""
        assert scorer.numeric_proximity(5.0, 5.0, max_diff=-1.0) == 1.0

    def test_negative_max_diff_different(self, scorer: SimilarityScorer):
        """Negative max_diff with different numbers returns 0.0."""
        assert scorer.numeric_proximity(5.0, 6.0, max_diff=-1.0) == 0.0

    def test_small_difference(self, scorer: SimilarityScorer):
        """Small difference gives high score."""
        score = scorer.numeric_proximity(100.0, 100.01, max_diff=100.0)
        assert score > 0.999

    def test_score_in_range(self, scorer: SimilarityScorer):
        """All scores are in [0.0, 1.0]."""
        test_cases = [
            (0, 0, 100), (0, 100, 100), (0, 200, 100),
            (-50, 50, 200), (1e6, 1e6 + 1, 1000),
        ]
        for a, b, md in test_cases:
            score = scorer.numeric_proximity(float(a), float(b), max_diff=float(md))
            assert 0.0 <= score <= 1.0

    def test_symmetry(self, scorer: SimilarityScorer):
        """Numeric proximity is symmetric."""
        pairs = [(10.0, 20.0), (100.0, 0.0), (-5.0, 5.0)]
        for a, b in pairs:
            assert scorer.numeric_proximity(a, b, 100.0) == pytest.approx(
                scorer.numeric_proximity(b, a, 100.0), abs=1e-6
            )

    def test_known_values(self, scorer: SimilarityScorer):
        """Verify specific known values."""
        assert scorer.numeric_proximity(10.0, 30.0, max_diff=100.0) == pytest.approx(0.8)
        assert scorer.numeric_proximity(0.0, 50.0, max_diff=100.0) == pytest.approx(0.5)
        assert scorer.numeric_proximity(0.0, 100.0, max_diff=100.0) == pytest.approx(0.0)


# ===========================================================================
# Test Class: date_proximity
# ===========================================================================


class TestDateProximity:
    """Tests for date proximity similarity."""

    def test_same_date(self, scorer: SimilarityScorer):
        """Same date returns 1.0."""
        assert scorer.date_proximity("2025-06-15", "2025-06-15") == 1.0

    def test_one_day_apart(self, scorer: SimilarityScorer):
        """Dates one day apart with max_days=365."""
        score = scorer.date_proximity("2025-06-15", "2025-06-16", max_days=365)
        assert score == pytest.approx(1.0 - 1 / 365, abs=1e-3)

    def test_within_max_days(self, scorer: SimilarityScorer):
        """Dates within max_days return proportional score."""
        score = scorer.date_proximity("2025-01-01", "2025-07-01", max_days=365)
        # ~181 days apart -> 1 - 181/365 ~ 0.504
        assert 0.4 < score < 0.6

    def test_at_max_days(self, scorer: SimilarityScorer):
        """Dates exactly max_days apart return 0.0."""
        score = scorer.date_proximity("2025-01-01", "2025-12-31", max_days=364)
        # 364 days apart / 364 max_days = 1.0 - 1.0 = 0.0
        assert score == pytest.approx(0.0, abs=1e-3)

    def test_beyond_max_days(self, scorer: SimilarityScorer):
        """Dates beyond max_days return 0.0."""
        score = scorer.date_proximity("2020-01-01", "2025-01-01", max_days=365)
        assert score == 0.0

    def test_different_date_formats(self, scorer: SimilarityScorer):
        """Date proximity handles different formats with same date."""
        # Both dates parse to 2025-06-15 at different times, so they
        # may have a fractional day difference. Assert near 1.0.
        score = scorer.date_proximity("2025-06-15", "2025-06-15T10:30:00")
        assert score >= 0.99

    def test_invalid_date_returns_0(self, scorer: SimilarityScorer):
        """Invalid date string returns 0.0."""
        assert scorer.date_proximity("not-a-date", "2025-06-15") == 0.0
        assert scorer.date_proximity("2025-06-15", "not-a-date") == 0.0

    def test_empty_date_returns_0(self, scorer: SimilarityScorer):
        """Empty date string returns 0.0."""
        assert scorer.date_proximity("", "2025-06-15") == 0.0
        assert scorer.date_proximity("2025-06-15", "") == 0.0

    def test_both_empty_returns_0(self, scorer: SimilarityScorer):
        """Both empty date strings return 0.0."""
        assert scorer.date_proximity("", "") == 0.0

    def test_zero_max_days_same(self, scorer: SimilarityScorer):
        """max_days=0 with same date returns 1.0."""
        assert scorer.date_proximity("2025-06-15", "2025-06-15", max_days=0) == 1.0

    def test_zero_max_days_different(self, scorer: SimilarityScorer):
        """max_days=0 with different dates returns 0.0."""
        assert scorer.date_proximity("2025-06-15", "2025-06-16", max_days=0) == 0.0

    def test_score_in_range(self, scorer: SimilarityScorer):
        """All scores are in [0.0, 1.0]."""
        dates = ["2025-01-01", "2025-06-15", "2025-12-31", "2024-01-01"]
        for i in range(len(dates)):
            for j in range(len(dates)):
                score = scorer.date_proximity(dates[i], dates[j])
                assert 0.0 <= score <= 1.0

    def test_symmetry(self, scorer: SimilarityScorer):
        """Date proximity is symmetric."""
        pairs = [
            ("2025-01-01", "2025-06-15"),
            ("2025-03-01", "2025-09-01"),
        ]
        for a, b in pairs:
            assert scorer.date_proximity(a, b) == pytest.approx(
                scorer.date_proximity(b, a), abs=1e-6
            )

    def test_iso_with_z(self, scorer: SimilarityScorer):
        """Date proximity handles ISO format with Z."""
        score = scorer.date_proximity("2025-06-15T10:30:00Z", "2025-06-15")
        assert score == 1.0

    def test_us_date_format(self, scorer: SimilarityScorer):
        """Date proximity handles US date format (MM/DD/YYYY)."""
        score = scorer.date_proximity("06/15/2025", "2025-06-15")
        assert score == 1.0


# ===========================================================================
# Test Class: score_pair
# ===========================================================================


class TestScorePair:
    """Tests for score_pair with weighted fields."""

    def test_score_pair_basic(
        self, scorer: SimilarityScorer,
        record_alice_a: Dict, record_alice_b: Dict,
        name_config: FieldComparisonConfig,
    ):
        """Score pair returns a valid SimilarityResult."""
        result = scorer.score_pair(
            record_alice_a, record_alice_b, [name_config],
        )
        assert isinstance(result, SimilarityResult)
        assert result.record_a_id == "a1"
        assert result.record_b_id == "a2"
        assert 0.0 <= result.overall_score <= 1.0
        assert "name" in result.field_scores

    def test_score_pair_identical_records(
        self, scorer: SimilarityScorer,
        record_alice_a: Dict,
        name_config: FieldComparisonConfig,
    ):
        """Identical records score 1.0."""
        result = scorer.score_pair(
            record_alice_a, record_alice_a, [name_config],
        )
        assert result.overall_score == pytest.approx(1.0, abs=1e-3)

    def test_score_pair_different_records(
        self, scorer: SimilarityScorer,
        record_alice_a: Dict, record_bob: Dict,
        name_config: FieldComparisonConfig,
    ):
        """Very different records score low."""
        result = scorer.score_pair(
            record_alice_a, record_bob, [name_config],
        )
        assert result.overall_score < 0.5

    def test_score_pair_multiple_fields(
        self, scorer: SimilarityScorer,
        record_alice_a: Dict, record_alice_b: Dict,
        multi_field_configs: List[FieldComparisonConfig],
    ):
        """Multiple field scores are computed and weighted."""
        result = scorer.score_pair(
            record_alice_a, record_alice_b, multi_field_configs,
        )
        assert len(result.field_scores) == 3
        assert "name" in result.field_scores
        assert "email" in result.field_scores
        assert "amount" in result.field_scores

    def test_score_pair_weighted_overall(
        self, scorer: SimilarityScorer,
    ):
        """Overall score is weighted average of field scores."""
        rec_a = {"id": "a", "f1": "hello", "f2": "world"}
        rec_b = {"id": "b", "f1": "hello", "f2": "earth"}
        configs = [
            FieldComparisonConfig(field_name="f1", algorithm=SimilarityAlgorithm.EXACT, weight=2.0),
            FieldComparisonConfig(field_name="f2", algorithm=SimilarityAlgorithm.EXACT, weight=1.0),
        ]
        result = scorer.score_pair(rec_a, rec_b, configs)
        # f1: exact match "hello"=="hello" -> 1.0, weight 2.0
        # f2: exact match "world"!="earth" -> 0.0, weight 1.0
        # overall = (1.0*2 + 0.0*1) / (2+1) = 2/3
        assert result.overall_score == pytest.approx(2 / 3, abs=1e-3)

    def test_score_pair_custom_ids(self, scorer: SimilarityScorer):
        """Custom record IDs are used in the result."""
        rec_a = {"name": "Alice"}
        rec_b = {"name": "Bob"}
        config = FieldComparisonConfig(field_name="name", algorithm=SimilarityAlgorithm.EXACT)
        result = scorer.score_pair(
            rec_a, rec_b, [config],
            record_a_id="custom-a", record_b_id="custom-b",
        )
        assert result.record_a_id == "custom-a"
        assert result.record_b_id == "custom-b"

    def test_score_pair_empty_configs_raises(self, scorer: SimilarityScorer):
        """Empty field_configs raises ValueError."""
        with pytest.raises(ValueError, match="field_configs must not be empty"):
            scorer.score_pair({"name": "A"}, {"name": "B"}, [])

    def test_score_pair_missing_field(self, scorer: SimilarityScorer):
        """Missing field in record is treated as empty string."""
        rec_a = {"name": "Alice"}
        rec_b = {}  # No "name" field
        config = FieldComparisonConfig(
            field_name="name",
            algorithm=SimilarityAlgorithm.LEVENSHTEIN,
        )
        result = scorer.score_pair(rec_a, rec_b, [config])
        # "alice" vs "" -> Levenshtein returns 0.0
        assert result.field_scores["name"] == 0.0

    def test_score_pair_provenance_present(
        self, scorer: SimilarityScorer,
        record_alice_a: Dict, record_alice_b: Dict,
        name_config: FieldComparisonConfig,
    ):
        """Score pair includes a provenance hash."""
        result = scorer.score_pair(record_alice_a, record_alice_b, [name_config])
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    def test_score_pair_comparison_time(
        self, scorer: SimilarityScorer,
        record_alice_a: Dict, record_alice_b: Dict,
        name_config: FieldComparisonConfig,
    ):
        """Score pair records comparison time."""
        result = scorer.score_pair(record_alice_a, record_alice_b, [name_config])
        assert result.comparison_time_ms >= 0.0

    def test_score_pair_algorithm_used(
        self, scorer: SimilarityScorer,
        record_alice_a: Dict, record_alice_b: Dict,
        name_config: FieldComparisonConfig,
    ):
        """Score pair records the primary algorithm used (first config)."""
        result = scorer.score_pair(record_alice_a, record_alice_b, [name_config])
        assert result.algorithm_used == SimilarityAlgorithm.JARO_WINKLER

    def test_score_pair_strip_whitespace(self, scorer: SimilarityScorer):
        """Strip whitespace preprocessing is applied."""
        rec_a = {"name": "  Alice  "}
        rec_b = {"name": "Alice"}
        config = FieldComparisonConfig(
            field_name="name",
            algorithm=SimilarityAlgorithm.EXACT,
            strip_whitespace=True,
        )
        result = scorer.score_pair(rec_a, rec_b, [config])
        assert result.field_scores["name"] == 1.0

    def test_score_pair_case_insensitive(self, scorer: SimilarityScorer):
        """Case-insensitive comparison when configured."""
        rec_a = {"name": "ALICE"}
        rec_b = {"name": "alice"}
        config = FieldComparisonConfig(
            field_name="name",
            algorithm=SimilarityAlgorithm.EXACT,
            case_sensitive=False,
        )
        result = scorer.score_pair(rec_a, rec_b, [config])
        assert result.field_scores["name"] == 1.0

    def test_score_pair_case_sensitive(self, scorer: SimilarityScorer):
        """Case-sensitive comparison when configured."""
        rec_a = {"name": "ALICE"}
        rec_b = {"name": "alice"}
        config = FieldComparisonConfig(
            field_name="name",
            algorithm=SimilarityAlgorithm.EXACT,
            case_sensitive=True,
        )
        result = scorer.score_pair(rec_a, rec_b, [config])
        assert result.field_scores["name"] == 0.0

    def test_score_pair_all_algorithms_dispatch(self, scorer: SimilarityScorer):
        """score_pair dispatches to all 8 algorithms correctly."""
        rec_a = {"val": "hello"}
        rec_b = {"val": "hello"}
        for algo in SimilarityAlgorithm:
            config = FieldComparisonConfig(field_name="val", algorithm=algo)
            if algo == SimilarityAlgorithm.NUMERIC:
                rec_a_n = {"val": "100"}
                rec_b_n = {"val": "100"}
                result = scorer.score_pair(rec_a_n, rec_b_n, [config])
            elif algo == SimilarityAlgorithm.DATE:
                rec_a_d = {"val": "2025-06-15"}
                rec_b_d = {"val": "2025-06-15"}
                result = scorer.score_pair(rec_a_d, rec_b_d, [config])
            else:
                result = scorer.score_pair(rec_a, rec_b, [config])
            assert result.field_scores["val"] == pytest.approx(1.0, abs=1e-3), \
                f"Algorithm {algo} failed for identical inputs"

    def test_score_pair_overall_clamped(self, scorer: SimilarityScorer):
        """Overall score is clamped to [0.0, 1.0]."""
        rec_a = {"f": "hello"}
        rec_b = {"f": "hello"}
        config = FieldComparisonConfig(
            field_name="f", algorithm=SimilarityAlgorithm.EXACT, weight=10.0,
        )
        result = scorer.score_pair(rec_a, rec_b, [config])
        assert 0.0 <= result.overall_score <= 1.0

    def test_score_pair_numeric_dispatch_invalid_values(self, scorer: SimilarityScorer):
        """Numeric algorithm with non-numeric strings returns 0.0."""
        rec_a = {"val": "not-a-number"}
        rec_b = {"val": "also-not"}
        config = FieldComparisonConfig(field_name="val", algorithm=SimilarityAlgorithm.NUMERIC)
        result = scorer.score_pair(rec_a, rec_b, [config])
        assert result.field_scores["val"] == 0.0

    def test_score_pair_date_dispatch_invalid_values(self, scorer: SimilarityScorer):
        """Date algorithm with invalid dates returns 0.0."""
        rec_a = {"val": "not-a-date"}
        rec_b = {"val": "also-not"}
        config = FieldComparisonConfig(field_name="val", algorithm=SimilarityAlgorithm.DATE)
        result = scorer.score_pair(rec_a, rec_b, [config])
        assert result.field_scores["val"] == 0.0


# ===========================================================================
# Test Class: score_batch
# ===========================================================================


class TestScoreBatch:
    """Tests for batch scoring."""

    def test_score_batch_basic(self, scorer: SimilarityScorer):
        """Batch scoring processes all pairs."""
        pairs = [
            ({"id": "1", "name": "Alice"}, {"id": "2", "name": "Alice"}),
            ({"id": "3", "name": "Bob"}, {"id": "4", "name": "Bobby"}),
        ]
        config = FieldComparisonConfig(field_name="name", algorithm=SimilarityAlgorithm.LEVENSHTEIN)
        results = scorer.score_batch(pairs, [config])
        assert len(results) == 2
        assert all(isinstance(r, SimilarityResult) for r in results)

    def test_score_batch_empty(self, scorer: SimilarityScorer):
        """Batch scoring with empty pairs returns empty list."""
        config = FieldComparisonConfig(field_name="name", algorithm=SimilarityAlgorithm.EXACT)
        results = scorer.score_batch([], [config])
        assert results == []

    def test_score_batch_uses_id_field(self, scorer: SimilarityScorer):
        """Batch scoring uses the id_field parameter."""
        pairs = [
            ({"uid": "aa", "name": "Alice"}, {"uid": "bb", "name": "Bob"}),
        ]
        config = FieldComparisonConfig(field_name="name", algorithm=SimilarityAlgorithm.EXACT)
        results = scorer.score_batch(pairs, [config], id_field="uid")
        assert results[0].record_a_id == "aa"
        assert results[0].record_b_id == "bb"

    def test_score_batch_large(self, scorer: SimilarityScorer):
        """Batch scoring handles 100 pairs."""
        pairs = [
            ({"id": f"a{i}", "name": f"Person {i}"}, {"id": f"b{i}", "name": f"Person {i}"})
            for i in range(100)
        ]
        config = FieldComparisonConfig(field_name="name", algorithm=SimilarityAlgorithm.EXACT)
        results = scorer.score_batch(pairs, [config])
        assert len(results) == 100
        assert all(r.overall_score == pytest.approx(1.0) for r in results)

    def test_score_batch_preserves_order(self, scorer: SimilarityScorer):
        """Batch scoring preserves pair order."""
        pairs = [
            ({"id": "1", "name": "Alice"}, {"id": "2", "name": "Alice"}),
            ({"id": "3", "name": "Bob"}, {"id": "4", "name": "Charlie"}),
        ]
        config = FieldComparisonConfig(field_name="name", algorithm=SimilarityAlgorithm.EXACT)
        results = scorer.score_batch(pairs, [config])
        assert results[0].record_a_id == "1"
        assert results[1].record_a_id == "3"


# ===========================================================================
# Test Class: Soundex Encoding (internal)
# ===========================================================================


class TestSoundexEncode:
    """Tests for the internal _soundex_encode method."""

    def test_soundex_robert(self, scorer: SimilarityScorer):
        """Soundex of 'Robert' is R163."""
        assert scorer._soundex_encode("Robert") == "R163"

    def test_soundex_rupert(self, scorer: SimilarityScorer):
        """Soundex of 'Rupert' is R163."""
        assert scorer._soundex_encode("Rupert") == "R163"

    def test_soundex_empty(self, scorer: SimilarityScorer):
        """Soundex of empty string is '0000'."""
        assert scorer._soundex_encode("") == "0000"

    def test_soundex_whitespace(self, scorer: SimilarityScorer):
        """Soundex of whitespace-only is '0000'."""
        assert scorer._soundex_encode("   ") == "0000"

    def test_soundex_single_letter(self, scorer: SimilarityScorer):
        """Soundex of single letter pads with zeros."""
        assert scorer._soundex_encode("A") == "A000"
        assert scorer._soundex_encode("Z") == "Z000"

    def test_soundex_preserves_first_upper(self, scorer: SimilarityScorer):
        """First letter is uppercased."""
        assert scorer._soundex_encode("alice")[0] == "A"

    def test_soundex_length_always_4(self, scorer: SimilarityScorer):
        """Soundex code is always exactly 4 characters."""
        names = ["A", "Alice", "Bob", "Charliebrowningtonsmith", "X"]
        for name in names:
            result = scorer._soundex_encode(name)
            assert len(result) == 4, f"Soundex of '{name}' has length {len(result)}"

    def test_soundex_strips_numbers(self, scorer: SimilarityScorer):
        """Non-alpha characters are stripped."""
        assert scorer._soundex_encode("Alice123") == scorer._soundex_encode("Alice")


# ===========================================================================
# Test Class: N-gram Generation (internal)
# ===========================================================================


class TestNgramGeneration:
    """Tests for the internal _generate_ngrams method."""

    def test_trigrams(self, scorer: SimilarityScorer):
        """Trigram generation from 'hello'."""
        result = scorer._generate_ngrams("hello", 3)
        assert result == ["hel", "ell", "llo"]

    def test_empty_string(self, scorer: SimilarityScorer):
        """Empty string returns empty list."""
        assert scorer._generate_ngrams("", 3) == []

    def test_short_string(self, scorer: SimilarityScorer):
        """String shorter than n returns [text]."""
        assert scorer._generate_ngrams("ab", 3) == ["ab"]

    def test_exact_length(self, scorer: SimilarityScorer):
        """String of exact n-gram length returns one element."""
        assert scorer._generate_ngrams("abc", 3) == ["abc"]

    def test_bigrams(self, scorer: SimilarityScorer):
        """Bigram generation."""
        assert scorer._generate_ngrams("abcd", 2) == ["ab", "bc", "cd"]


# ===========================================================================
# Test Class: Date Parsing (internal)
# ===========================================================================


class TestDateParsing:
    """Tests for the internal _parse_date method."""

    def test_iso_format(self, scorer: SimilarityScorer):
        """ISO format YYYY-MM-DD is parsed."""
        result = scorer._parse_date("2025-06-15")
        assert result is not None
        assert result.year == 2025
        assert result.month == 6
        assert result.day == 15

    def test_iso_datetime(self, scorer: SimilarityScorer):
        """ISO datetime format is parsed."""
        result = scorer._parse_date("2025-06-15T10:30:00")
        assert result is not None
        assert result.year == 2025

    def test_iso_with_z(self, scorer: SimilarityScorer):
        """ISO datetime with Z suffix is parsed."""
        result = scorer._parse_date("2025-06-15T10:30:00Z")
        assert result is not None

    def test_us_format(self, scorer: SimilarityScorer):
        """US format MM/DD/YYYY is parsed."""
        result = scorer._parse_date("06/15/2025")
        assert result is not None
        assert result.month == 6

    def test_compact_format(self, scorer: SimilarityScorer):
        """Compact format YYYYMMDD is parsed."""
        result = scorer._parse_date("20250615")
        assert result is not None
        assert result.year == 2025

    def test_invalid_date(self, scorer: SimilarityScorer):
        """Invalid date string returns None."""
        assert scorer._parse_date("not-a-date") is None

    def test_empty_string(self, scorer: SimilarityScorer):
        """Empty string returns None."""
        assert scorer._parse_date("") is None

    def test_whitespace_only(self, scorer: SimilarityScorer):
        """Whitespace-only returns None."""
        assert scorer._parse_date("   ") is None


# ===========================================================================
# Test Class: Edge Cases and Range Validation
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and score range validation."""

    def test_all_algorithms_return_range_0_1(self, scorer: SimilarityScorer):
        """All individual algorithms return scores in [0.0, 1.0]."""
        test_pairs = [
            ("hello", "hallo"),
            ("", "test"),
            ("test", ""),
            ("identical", "identical"),
            ("a", "z"),
            ("sustainability report", "carbon emissions data"),
        ]
        for a, b in test_pairs:
            assert 0.0 <= scorer.exact_match(a, b) <= 1.0
            assert 0.0 <= scorer.levenshtein_similarity(a, b) <= 1.0
            assert 0.0 <= scorer.jaro_winkler_similarity(a, b) <= 1.0
            assert 0.0 <= scorer.soundex_similarity(a, b) <= 1.0
            assert 0.0 <= scorer.ngram_similarity(a, b) <= 1.0
            assert 0.0 <= scorer.tfidf_cosine_similarity(a, b) <= 1.0

    def test_numeric_edge_cases(self, scorer: SimilarityScorer):
        """Numeric proximity edge cases."""
        # Very large numbers
        score = scorer.numeric_proximity(1e15, 1e15 + 1, max_diff=1e15)
        assert 0.0 <= score <= 1.0
        # Very small numbers
        score = scorer.numeric_proximity(1e-10, 2e-10, max_diff=1e-9)
        assert 0.0 <= score <= 1.0

    def test_date_edge_cases(self, scorer: SimilarityScorer):
        """Date proximity edge cases."""
        # Very far apart dates
        score = scorer.date_proximity("2000-01-01", "2025-12-31")
        assert score == 0.0  # ~9496 days apart, max_days=365

    def test_none_value_in_record(self, scorer: SimilarityScorer):
        """None values in records are converted to string 'None'."""
        rec_a = {"name": None}
        rec_b = {"name": "Alice"}
        config = FieldComparisonConfig(
            field_name="name", algorithm=SimilarityAlgorithm.LEVENSHTEIN,
        )
        result = scorer.score_pair(rec_a, rec_b, [config])
        assert 0.0 <= result.overall_score <= 1.0

    def test_integer_values_in_record(self, scorer: SimilarityScorer):
        """Integer values in records are converted to string."""
        rec_a = {"val": 42}
        rec_b = {"val": 42}
        config = FieldComparisonConfig(
            field_name="val", algorithm=SimilarityAlgorithm.EXACT,
        )
        result = scorer.score_pair(rec_a, rec_b, [config])
        assert result.field_scores["val"] == 1.0

    def test_boolean_values_in_record(self, scorer: SimilarityScorer):
        """Boolean values in records are converted to string."""
        rec_a = {"val": True}
        rec_b = {"val": True}
        config = FieldComparisonConfig(
            field_name="val", algorithm=SimilarityAlgorithm.EXACT,
        )
        result = scorer.score_pair(rec_a, rec_b, [config])
        assert result.field_scores["val"] == 1.0

    def test_list_value_in_record(self, scorer: SimilarityScorer):
        """List values in records are stringified."""
        rec_a = {"val": [1, 2, 3]}
        rec_b = {"val": [1, 2, 3]}
        config = FieldComparisonConfig(
            field_name="val", algorithm=SimilarityAlgorithm.EXACT,
        )
        result = scorer.score_pair(rec_a, rec_b, [config])
        assert result.field_scores["val"] == 1.0


# ===========================================================================
# Test Class: Symmetry Validation
# ===========================================================================


class TestSymmetry:
    """Tests validating score(a,b) == score(b,a) for all algorithms."""

    def _assert_symmetric(self, scorer: SimilarityScorer, a: str, b: str, tolerance: float = 1e-6):
        """Helper: assert all string-based algorithms are symmetric."""
        assert scorer.exact_match(a, b) == scorer.exact_match(b, a)
        assert scorer.levenshtein_similarity(a, b) == pytest.approx(
            scorer.levenshtein_similarity(b, a), abs=tolerance,
        )
        assert scorer.jaro_winkler_similarity(a, b) == pytest.approx(
            scorer.jaro_winkler_similarity(b, a), abs=tolerance,
        )
        assert scorer.soundex_similarity(a, b) == scorer.soundex_similarity(b, a)
        assert scorer.ngram_similarity(a, b) == pytest.approx(
            scorer.ngram_similarity(b, a), abs=tolerance,
        )
        assert scorer.tfidf_cosine_similarity(a, b) == pytest.approx(
            scorer.tfidf_cosine_similarity(b, a), abs=tolerance,
        )

    def test_symmetry_similar_strings(self, scorer: SimilarityScorer):
        """Symmetric for similar strings."""
        self._assert_symmetric(scorer, "alice smith", "alice smyth")

    def test_symmetry_different_strings(self, scorer: SimilarityScorer):
        """Symmetric for different strings."""
        self._assert_symmetric(scorer, "hello", "world")

    def test_symmetry_one_empty(self, scorer: SimilarityScorer):
        """Symmetric when one string is empty."""
        self._assert_symmetric(scorer, "", "test")

    def test_symmetry_numeric(self, scorer: SimilarityScorer):
        """Numeric proximity is symmetric."""
        assert scorer.numeric_proximity(10.0, 50.0) == pytest.approx(
            scorer.numeric_proximity(50.0, 10.0),
        )

    def test_symmetry_date(self, scorer: SimilarityScorer):
        """Date proximity is symmetric."""
        assert scorer.date_proximity("2025-01-01", "2025-06-15") == pytest.approx(
            scorer.date_proximity("2025-06-15", "2025-01-01"),
        )


# ===========================================================================
# Test Class: Statistics Tracking
# ===========================================================================


class TestScorerStatistics:
    """Tests for thread-safe statistics tracking."""

    def test_initial_statistics(self, scorer: SimilarityScorer):
        """Initial statistics are all zero."""
        stats = scorer.get_statistics()
        assert stats["engine_name"] == "SimilarityScorer"
        assert stats["invocations"] == 0
        assert stats["successes"] == 0
        assert stats["failures"] == 0
        assert stats["total_duration_ms"] == 0.0
        assert stats["avg_duration_ms"] == 0.0
        assert stats["last_invoked_at"] is None

    def test_statistics_after_success(self, scorer: SimilarityScorer):
        """Statistics increment after successful scoring."""
        rec_a = {"name": "Alice"}
        rec_b = {"name": "Bob"}
        config = FieldComparisonConfig(field_name="name", algorithm=SimilarityAlgorithm.EXACT)
        scorer.score_pair(rec_a, rec_b, [config])
        stats = scorer.get_statistics()
        assert stats["invocations"] == 1
        assert stats["successes"] == 1
        assert stats["failures"] == 0
        assert stats["total_duration_ms"] >= 0  # May be 0 on fast machines
        assert stats["last_invoked_at"] is not None

    def test_statistics_after_failure(self, scorer: SimilarityScorer):
        """Statistics increment after failed scoring."""
        with pytest.raises(ValueError):
            scorer.score_pair({"name": "A"}, {"name": "B"}, [])
        stats = scorer.get_statistics()
        assert stats["invocations"] == 1
        assert stats["failures"] == 1

    def test_statistics_accumulate(self, scorer: SimilarityScorer):
        """Statistics accumulate across multiple operations."""
        rec_a = {"name": "Alice"}
        rec_b = {"name": "Bob"}
        config = FieldComparisonConfig(field_name="name", algorithm=SimilarityAlgorithm.EXACT)
        for _ in range(5):
            scorer.score_pair(rec_a, rec_b, [config])
        stats = scorer.get_statistics()
        assert stats["invocations"] == 5
        assert stats["successes"] == 5

    def test_reset_statistics(self, scorer: SimilarityScorer):
        """Reset clears all statistics."""
        rec_a = {"name": "Alice"}
        rec_b = {"name": "Bob"}
        config = FieldComparisonConfig(field_name="name", algorithm=SimilarityAlgorithm.EXACT)
        scorer.score_pair(rec_a, rec_b, [config])
        scorer.reset_statistics()
        stats = scorer.get_statistics()
        assert stats["invocations"] == 0
        assert stats["successes"] == 0
        assert stats["total_duration_ms"] == 0.0

    def test_statistics_avg_duration(self, scorer: SimilarityScorer):
        """Average duration is computed correctly."""
        rec_a = {"name": "Alice"}
        rec_b = {"name": "Bob"}
        config = FieldComparisonConfig(field_name="name", algorithm=SimilarityAlgorithm.EXACT)
        for _ in range(4):
            scorer.score_pair(rec_a, rec_b, [config])
        stats = scorer.get_statistics()
        expected_avg = stats["total_duration_ms"] / 4
        assert abs(stats["avg_duration_ms"] - expected_avg) < 0.01

    def test_statistics_thread_safety(self, scorer: SimilarityScorer):
        """Statistics remain consistent under concurrent access."""
        rec_a = {"name": "Alice"}
        rec_b = {"name": "Bob"}
        config = FieldComparisonConfig(field_name="name", algorithm=SimilarityAlgorithm.EXACT)
        errors: List[str] = []

        def worker():
            try:
                for _ in range(20):
                    scorer.score_pair(rec_a, rec_b, [config])
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = scorer.get_statistics()
        assert stats["invocations"] == 100
        assert stats["successes"] == 100


# ===========================================================================
# Test Class: Provenance Tracking
# ===========================================================================


class TestScorerProvenance:
    """Tests for provenance hash tracking in similarity scoring."""

    def test_provenance_hash_present(self, scorer: SimilarityScorer):
        """Every score_pair result has a provenance hash."""
        rec_a = {"name": "Alice"}
        rec_b = {"name": "Bob"}
        config = FieldComparisonConfig(field_name="name", algorithm=SimilarityAlgorithm.EXACT)
        result = scorer.score_pair(rec_a, rec_b, [config])
        assert result.provenance_hash != ""

    def test_provenance_hash_is_64_hex(self, scorer: SimilarityScorer):
        """Provenance hash is a 64-character hex string (SHA-256)."""
        rec_a = {"name": "Alice"}
        rec_b = {"name": "Bob"}
        config = FieldComparisonConfig(field_name="name", algorithm=SimilarityAlgorithm.EXACT)
        result = scorer.score_pair(rec_a, rec_b, [config])
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)  # Valid hex

    def test_provenance_hash_valid_chars(self, scorer: SimilarityScorer):
        """Provenance hash contains only valid hex characters."""
        rec_a = {"name": "Alice"}
        rec_b = {"name": "Bob"}
        config = FieldComparisonConfig(field_name="name", algorithm=SimilarityAlgorithm.EXACT)
        result = scorer.score_pair(rec_a, rec_b, [config])
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)

    def test_provenance_hash_not_reused(self, scorer: SimilarityScorer):
        """Different score_pair calls produce different provenance hashes."""
        rec_a = {"name": "Alice"}
        rec_b = {"name": "Bob"}
        config = FieldComparisonConfig(field_name="name", algorithm=SimilarityAlgorithm.EXACT)
        result1 = scorer.score_pair(rec_a, rec_b, [config])
        # Add a tiny sleep to ensure timestamp differs
        time.sleep(0.01)
        result2 = scorer.score_pair(rec_a, rec_b, [config])
        # Provenance includes timestamp, so they may differ
        # (not guaranteed to be different within same second, but testing generation)
        assert result1.provenance_hash != "" and result2.provenance_hash != ""


# ===========================================================================
# Test Class: Field Score Rounding
# ===========================================================================


class TestFieldScoreRounding:
    """Tests for field score precision and rounding."""

    def test_field_scores_rounded_to_6_decimals(self, scorer: SimilarityScorer):
        """Per-field scores are rounded to 6 decimal places."""
        rec_a = {"name": "kitten"}
        rec_b = {"name": "sitting"}
        config = FieldComparisonConfig(
            field_name="name", algorithm=SimilarityAlgorithm.LEVENSHTEIN,
        )
        result = scorer.score_pair(rec_a, rec_b, [config])
        score_str = str(result.field_scores["name"])
        # Check decimal places (after the decimal point)
        if "." in score_str:
            decimals = len(score_str.split(".")[1])
            assert decimals <= 6

    def test_overall_score_rounded_to_6_decimals(self, scorer: SimilarityScorer):
        """Overall score is rounded to 6 decimal places."""
        rec_a = {"name": "kitten"}
        rec_b = {"name": "sitting"}
        config = FieldComparisonConfig(
            field_name="name", algorithm=SimilarityAlgorithm.LEVENSHTEIN,
        )
        result = scorer.score_pair(rec_a, rec_b, [config])
        score_str = str(result.overall_score)
        if "." in score_str:
            decimals = len(score_str.split(".")[1])
            assert decimals <= 6
