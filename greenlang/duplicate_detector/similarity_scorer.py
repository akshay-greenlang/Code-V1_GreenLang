# -*- coding: utf-8 -*-
"""
Similarity Scorer Engine - AGENT-DATA-011: Duplicate Detection (GL-DATA-X-014)

Computes pairwise field-level similarity scores using 8 deterministic
algorithms: exact match, Levenshtein edit distance, Jaro-Winkler,
Soundex, character n-gram Jaccard, TF-IDF cosine, numeric proximity,
and date proximity.

Zero-Hallucination Guarantees:
    - All 8 similarity algorithms implemented from scratch
    - No external fuzzy matching libraries (no fuzzywuzzy/jellyfish)
    - All computations are deterministic Python arithmetic
    - No ML/LLM calls in similarity scoring path
    - Provenance recorded for every scoring operation

Supported Algorithms (8):
    EXACT:              Binary 1.0/0.0 match
    LEVENSHTEIN:        1 - (edit_distance / max_len)
    JARO_WINKLER:       Jaro similarity with Winkler prefix bonus
    SOUNDEX:            Phonetic encoding match (1.0 or 0.0)
    NGRAM:              Character n-gram Jaccard coefficient
    TFIDF_COSINE:       Term frequency cosine similarity
    NUMERIC:            1 - (|a-b| / max_diff), clamped [0,1]
    DATE:               1 - (|days_diff| / max_days), clamped [0,1]

Example:
    >>> from greenlang.duplicate_detector.similarity_scorer import SimilarityScorer
    >>> scorer = SimilarityScorer()
    >>> result = scorer.score_pair(
    ...     record_a={"name": "Alice Smith"},
    ...     record_b={"name": "Alise Smith"},
    ...     field_configs=[FieldComparisonConfig(field_name="name")],
    ... )
    >>> print(f"Overall: {result.overall_score:.3f}")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-011 Duplicate Detection Agent (GL-DATA-X-014)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.duplicate_detector.models import (
    FieldComparisonConfig,
    SimilarityAlgorithm,
    SimilarityResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SimilarityScorer",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_provenance(operation: str, data_repr: str) -> str:
    """Compute SHA-256 provenance hash."""
    payload = f"{operation}:{data_repr}:{_utcnow().isoformat()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Soundex table
# ---------------------------------------------------------------------------

_SOUNDEX_TABLE: Dict[str, str] = {
    "b": "1", "f": "1", "p": "1", "v": "1",
    "c": "2", "g": "2", "j": "2", "k": "2",
    "q": "2", "s": "2", "x": "2", "z": "2",
    "d": "3", "t": "3",
    "l": "4",
    "m": "5", "n": "5",
    "r": "6",
}


# =============================================================================
# SimilarityScorer
# =============================================================================


class SimilarityScorer:
    """Similarity scoring engine with 8 deterministic algorithms.

    Computes per-field similarity scores between record pairs using
    configurable algorithms and weights. All algorithms are implemented
    from scratch with no external fuzzy matching dependencies.

    This engine follows GreenLang's zero-hallucination principle:
    all similarity computations are deterministic arithmetic operations.

    Attributes:
        _stats_lock: Threading lock for stats updates.
        _invocations: Total invocation count.
        _successes: Total successful invocations.
        _failures: Total failed invocations.
        _total_duration_ms: Cumulative processing time.

    Example:
        >>> scorer = SimilarityScorer()
        >>> score = scorer.levenshtein_similarity("kitten", "sitting")
        >>> print(f"Similarity: {score:.3f}")
    """

    def __init__(self) -> None:
        """Initialize SimilarityScorer with empty statistics."""
        self._stats_lock = threading.Lock()
        self._invocations: int = 0
        self._successes: int = 0
        self._failures: int = 0
        self._total_duration_ms: float = 0.0
        self._last_invoked_at: Optional[datetime] = None
        logger.info("SimilarityScorer initialized")

    # ------------------------------------------------------------------
    # Public API - Pair scoring
    # ------------------------------------------------------------------

    def score_pair(
        self,
        record_a: Dict[str, Any],
        record_b: Dict[str, Any],
        field_configs: List[FieldComparisonConfig],
        record_a_id: str = "",
        record_b_id: str = "",
    ) -> SimilarityResult:
        """Score similarity between two records across configured fields.

        Computes per-field similarity scores and a weighted overall
        score based on the field comparison configurations.

        Args:
            record_a: First record dictionary.
            record_b: Second record dictionary.
            field_configs: Per-field comparison configurations.
            record_a_id: Optional identifier for record A.
            record_b_id: Optional identifier for record B.

        Returns:
            SimilarityResult with per-field and overall scores.

        Raises:
            ValueError: If field_configs is empty.
        """
        start_time = time.monotonic()
        rid_a = record_a_id or str(record_a.get("id", "a"))
        rid_b = record_b_id or str(record_b.get("id", "b"))

        try:
            if not field_configs:
                raise ValueError("field_configs must not be empty")

            field_scores: Dict[str, float] = {}
            total_weight = 0.0
            weighted_sum = 0.0
            primary_algo = field_configs[0].algorithm

            for config in field_configs:
                val_a = str(record_a.get(config.field_name, ""))
                val_b = str(record_b.get(config.field_name, ""))

                # Apply preprocessing
                if config.strip_whitespace:
                    val_a = val_a.strip()
                    val_b = val_b.strip()
                if not config.case_sensitive:
                    val_a = val_a.lower()
                    val_b = val_b.lower()

                score = self._compute_field_score(val_a, val_b, config)
                field_scores[config.field_name] = round(score, 6)

                weighted_sum += score * config.weight
                total_weight += config.weight

            overall = weighted_sum / total_weight if total_weight > 0 else 0.0
            overall = max(0.0, min(1.0, round(overall, 6)))

            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            provenance = _compute_provenance(
                "score_pair", f"{rid_a}:{rid_b}:{overall}",
            )

            result = SimilarityResult(
                record_a_id=rid_a,
                record_b_id=rid_b,
                field_scores=field_scores,
                overall_score=overall,
                algorithm_used=primary_algo,
                comparison_time_ms=round(elapsed_ms, 3),
                provenance_hash=provenance,
            )

            self._record_success(time.monotonic() - start_time)
            return result

        except Exception as e:
            self._record_failure(time.monotonic() - start_time)
            logger.error("Scoring pair (%s, %s) failed: %s", rid_a, rid_b, e)
            raise

    def score_batch(
        self,
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
        field_configs: List[FieldComparisonConfig],
        id_field: str = "id",
    ) -> List[SimilarityResult]:
        """Score similarity for a batch of record pairs.

        Args:
            pairs: List of (record_a, record_b) tuples.
            field_configs: Per-field comparison configurations.
            id_field: Field name for record identifier.

        Returns:
            List of SimilarityResult instances.
        """
        if not pairs:
            return []

        logger.info("Scoring batch of %d pairs", len(pairs))
        results: List[SimilarityResult] = []

        for record_a, record_b in pairs:
            rid_a = str(record_a.get(id_field, "a"))
            rid_b = str(record_b.get(id_field, "b"))
            result = self.score_pair(
                record_a, record_b, field_configs, rid_a, rid_b,
            )
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Public API - Individual algorithms
    # ------------------------------------------------------------------

    def exact_match(self, a: str, b: str) -> float:
        """Binary exact match: 1.0 if identical, 0.0 otherwise."""
        return 1.0 if a == b else 0.0

    def levenshtein_similarity(self, a: str, b: str) -> float:
        """Levenshtein edit distance normalized to similarity.

        Implements Wagner-Fischer DP algorithm. Normalized as:
        similarity = 1 - (edit_distance / max(len(a), len(b)))

        Args:
            a: First string.
            b: Second string.

        Returns:
            Similarity score (0.0 to 1.0).
        """
        if a == b:
            return 1.0
        if not a or not b:
            return 0.0

        len_a, len_b = len(a), len(b)
        prev_row: List[int] = list(range(len_b + 1))
        curr_row: List[int] = [0] * (len_b + 1)

        for i in range(1, len_a + 1):
            curr_row[0] = i
            for j in range(1, len_b + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                curr_row[j] = min(
                    prev_row[j] + 1,
                    curr_row[j - 1] + 1,
                    prev_row[j - 1] + cost,
                )
            prev_row, curr_row = curr_row, prev_row

        edit_distance = prev_row[len_b]
        return 1.0 - (edit_distance / max(len_a, len_b))

    def jaro_winkler_similarity(
        self, a: str, b: str, winkler_prefix_weight: float = 0.1,
    ) -> float:
        """Jaro-Winkler string similarity.

        Jaro = (1/3) * (m/|a| + m/|b| + (m-t)/m)
        Winkler = Jaro + L * p * (1 - Jaro)

        Args:
            a: First string.
            b: Second string.
            winkler_prefix_weight: Winkler prefix weight (default 0.1).

        Returns:
            Jaro-Winkler similarity (0.0 to 1.0).
        """
        if a == b:
            return 1.0
        if not a or not b:
            return 0.0

        len_a, len_b = len(a), len(b)
        match_distance = max(len_a, len_b) // 2 - 1
        if match_distance < 0:
            match_distance = 0

        a_matches = [False] * len_a
        b_matches = [False] * len_b
        matches = 0
        transpositions = 0

        for i in range(len_a):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len_b)
            for j in range(start, end):
                if b_matches[j] or a[i] != b[j]:
                    continue
                a_matches[i] = True
                b_matches[j] = True
                matches += 1
                break

        if matches == 0:
            return 0.0

        k = 0
        for i in range(len_a):
            if not a_matches[i]:
                continue
            while not b_matches[k]:
                k += 1
            if a[i] != b[k]:
                transpositions += 1
            k += 1

        jaro = (
            matches / len_a + matches / len_b
            + (matches - transpositions / 2) / matches
        ) / 3.0

        prefix_len = 0
        for i in range(min(4, min(len_a, len_b))):
            if a[i] == b[i]:
                prefix_len += 1
            else:
                break

        winkler = jaro + prefix_len * winkler_prefix_weight * (1.0 - jaro)
        return max(0.0, min(1.0, winkler))

    def soundex_similarity(self, a: str, b: str) -> float:
        """Soundex phonetic similarity: 1.0 if codes match, 0.0 otherwise."""
        return 1.0 if self._soundex_encode(a) == self._soundex_encode(b) else 0.0

    def ngram_similarity(self, a: str, b: str, n: int = 3) -> float:
        """Character n-gram Jaccard coefficient similarity.

        Args:
            a: First string.
            b: Second string.
            n: N-gram size (default 3).

        Returns:
            Jaccard coefficient (0.0 to 1.0).
        """
        if a == b:
            return 1.0
        if not a or not b:
            return 0.0

        ngrams_a = self._generate_ngrams(a, n)
        ngrams_b = self._generate_ngrams(b, n)
        if not ngrams_a and not ngrams_b:
            return 1.0
        if not ngrams_a or not ngrams_b:
            return 0.0

        set_a, set_b = set(ngrams_a), set(ngrams_b)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def tfidf_cosine_similarity(self, a: str, b: str) -> float:
        """Term frequency cosine similarity between two strings.

        Args:
            a: First string.
            b: Second string.

        Returns:
            Cosine similarity (0.0 to 1.0).
        """
        if a == b:
            return 1.0
        if not a or not b:
            return 0.0

        terms_a = a.lower().split()
        terms_b = b.lower().split()
        if not terms_a or not terms_b:
            return 0.0

        tf_a: Dict[str, float] = {}
        for t in terms_a:
            tf_a[t] = tf_a.get(t, 0.0) + 1.0
        tf_b: Dict[str, float] = {}
        for t in terms_b:
            tf_b[t] = tf_b.get(t, 0.0) + 1.0

        common = set(tf_a.keys()) & set(tf_b.keys())
        dot = sum(tf_a[t] * tf_b[t] for t in common)
        mag_a = math.sqrt(sum(v * v for v in tf_a.values()))
        mag_b = math.sqrt(sum(v * v for v in tf_b.values()))

        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / (mag_a * mag_b)))

    def numeric_proximity(
        self, a: float, b: float, max_diff: float = 100.0,
    ) -> float:
        """Numeric proximity: 1 - (|a-b| / max_diff), clamped [0,1]."""
        if max_diff <= 0:
            return 1.0 if a == b else 0.0
        return max(0.0, min(1.0, 1.0 - abs(a - b) / max_diff))

    def date_proximity(
        self, a: str, b: str, max_days: int = 365,
    ) -> float:
        """Date proximity: 1 - (|days_diff| / max_days), clamped [0,1]."""
        date_a = self._parse_date(a)
        date_b = self._parse_date(b)
        if date_a is None or date_b is None:
            return 0.0
        if max_days <= 0:
            return 1.0 if date_a == date_b else 0.0
        days_diff = abs((date_a - date_b).days)
        return max(0.0, min(1.0, 1.0 - days_diff / max_days))

    def get_statistics(self) -> Dict[str, Any]:
        """Return current engine operational statistics."""
        with self._stats_lock:
            avg_ms = 0.0
            if self._invocations > 0:
                avg_ms = self._total_duration_ms / self._invocations
            return {
                "engine_name": "SimilarityScorer",
                "invocations": self._invocations,
                "successes": self._successes,
                "failures": self._failures,
                "total_duration_ms": round(self._total_duration_ms, 3),
                "avg_duration_ms": round(avg_ms, 3),
                "last_invoked_at": (
                    self._last_invoked_at.isoformat()
                    if self._last_invoked_at else None
                ),
            }

    def reset_statistics(self) -> None:
        """Reset all operational statistics to zero."""
        with self._stats_lock:
            self._invocations = 0
            self._successes = 0
            self._failures = 0
            self._total_duration_ms = 0.0
            self._last_invoked_at = None

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _compute_field_score(
        self, val_a: str, val_b: str, config: FieldComparisonConfig,
    ) -> float:
        """Dispatch to the appropriate algorithm for a single field."""
        algo = config.algorithm

        if algo == SimilarityAlgorithm.EXACT:
            return self.exact_match(val_a, val_b)
        elif algo == SimilarityAlgorithm.LEVENSHTEIN:
            return self.levenshtein_similarity(val_a, val_b)
        elif algo == SimilarityAlgorithm.JARO_WINKLER:
            return self.jaro_winkler_similarity(val_a, val_b)
        elif algo == SimilarityAlgorithm.SOUNDEX:
            return self.soundex_similarity(val_a, val_b)
        elif algo == SimilarityAlgorithm.NGRAM:
            return self.ngram_similarity(val_a, val_b)
        elif algo == SimilarityAlgorithm.TFIDF_COSINE:
            return self.tfidf_cosine_similarity(val_a, val_b)
        elif algo == SimilarityAlgorithm.NUMERIC:
            try:
                num_a = float(val_a) if val_a else 0.0
                num_b = float(val_b) if val_b else 0.0
            except ValueError:
                return 0.0
            return self.numeric_proximity(num_a, num_b)
        elif algo == SimilarityAlgorithm.DATE:
            return self.date_proximity(val_a, val_b)
        else:
            logger.warning("Unknown algorithm %s, defaulting to exact", algo)
            return self.exact_match(val_a, val_b)

    def _soundex_encode(self, value: str) -> str:
        """Encode a string using the Soundex algorithm."""
        if not value or not value.strip():
            return "0000"
        s = "".join(c for c in value.strip().upper() if c.isalpha())
        if not s:
            return "0000"
        first = s[0]
        coded: List[str] = [first]
        prev = _SOUNDEX_TABLE.get(first.lower(), "0")
        for ch in s[1:]:
            code = _SOUNDEX_TABLE.get(ch.lower(), "0")
            if code != "0" and code != prev:
                coded.append(code)
                if len(coded) == 4:
                    break
            prev = code
        return "".join(coded).ljust(4, "0")[:4]

    def _generate_ngrams(self, text: str, n: int) -> List[str]:
        """Generate character n-grams."""
        if len(text) < n:
            return [text] if text else []
        return [text[i:i + n] for i in range(len(text) - n + 1)]

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse a date string into a datetime object."""
        if not date_str or not date_str.strip():
            return None
        s = date_str.strip()
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
                     "%Y-%m-%dT%H:%M:%S.%f", "%m/%d/%Y", "%d-%m-%Y",
                     "%d/%m/%Y", "%Y%m%d"):
            try:
                return datetime.strptime(s[:26], fmt)
            except (ValueError, IndexError):
                continue
        return None

    def _record_success(self, elapsed_seconds: float) -> None:
        """Record a successful invocation."""
        ms = elapsed_seconds * 1000.0
        with self._stats_lock:
            self._invocations += 1
            self._successes += 1
            self._total_duration_ms += ms
            self._last_invoked_at = _utcnow()

    def _record_failure(self, elapsed_seconds: float) -> None:
        """Record a failed invocation."""
        ms = elapsed_seconds * 1000.0
        with self._stats_lock:
            self._invocations += 1
            self._failures += 1
            self._total_duration_ms += ms
            self._last_invoked_at = _utcnow()
