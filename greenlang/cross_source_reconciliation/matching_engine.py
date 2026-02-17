# -*- coding: utf-8 -*-
"""
Matching Engine - AGENT-DATA-015: Cross-Source Reconciliation

Engine 2 of 7. Cross-source record matching using composite keys, fuzzy
matching (Jaro-Winkler, Levenshtein), temporal alignment across different
reporting granularities, and hash-based blocking for scalable pairwise
comparison.

Zero-Hallucination: All similarity computations use deterministic Python
arithmetic. No external ML libraries or LLM calls for scoring. All match
decisions are traceable via SHA-256 provenance hashes.

Matching Strategies:
    1. Exact   -- composite key equality on entity+period+metric
    2. Fuzzy   -- Jaro-Winkler similarity on string fields, numeric
                  proximity on numeric fields, date proximity on date fields
    3. Composite -- multi-field weighted matching with configurable weights
    4. Temporal -- align records across granularities
                   (daily/monthly/quarterly/annual) then match on entity
    5. Blocking -- hash-based blocking on prefix keys to reduce O(n*m)

Example:
    >>> from greenlang.cross_source_reconciliation.matching_engine import (
    ...     MatchingEngine,
    ... )
    >>> engine = MatchingEngine()
    >>> results = engine.match_records(
    ...     source_a_records=[{"entity_id": "E1", "period": "2025-Q1", "value": 100}],
    ...     source_b_records=[{"entity_id": "E1", "period": "2025-Q1", "value": 102}],
    ...     key_fields=["entity_id", "period"],
    ...     strategy="exact",
    ... )
    >>> assert len(results) == 1

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-015 Cross-Source Reconciliation (GL-DATA-X-018)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graceful imports for optional sibling modules (metrics, models, config,
# provenance).  When the full SDK is wired together these provide real
# Prometheus counters and a shared provenance tracker.  When running the
# engine in isolation (tests, notebooks) the engine still works.
# ---------------------------------------------------------------------------

try:
    from greenlang.cross_source_reconciliation import metrics as _metrics_mod
    _METRICS_AVAILABLE = True
except ImportError:
    _metrics_mod = None  # type: ignore[assignment]
    _METRICS_AVAILABLE = False

try:
    from greenlang.cross_source_reconciliation.provenance import (  # type: ignore[import-untyped]
        ProvenanceTracker as _ExtProvenanceTracker,
        get_provenance_tracker as _ext_get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _ExtProvenanceTracker = None  # type: ignore[assignment, misc]
    _ext_get_provenance_tracker = None  # type: ignore[assignment]
    _PROVENANCE_AVAILABLE = False

try:
    from greenlang.cross_source_reconciliation.config import (  # type: ignore[import-untyped]
        get_config as _ext_get_config,
    )
    _CONFIG_AVAILABLE = True
except ImportError:
    _ext_get_config = None  # type: ignore[assignment]
    _CONFIG_AVAILABLE = False

try:
    from greenlang.cross_source_reconciliation.models import (  # type: ignore[import-untyped]
        MatchStrategy as _ExtMatchStrategy,
        MatchStatus as _ExtMatchStatus,
        TemporalGranularity as _ExtTemporalGranularity,
        FieldType as _ExtFieldType,
        MatchKey as _ExtMatchKey,
        MatchResult as _ExtMatchResult,
        BatchMatchResult as _ExtBatchMatchResult,
        SourceDefinition as _ExtSourceDefinition,
        SchemaMapping as _ExtSchemaMapping,
    )
    _MODELS_AVAILABLE = True
except ImportError:
    _ExtMatchStrategy = None  # type: ignore[assignment, misc]
    _ExtMatchStatus = None  # type: ignore[assignment, misc]
    _ExtTemporalGranularity = None  # type: ignore[assignment, misc]
    _ExtFieldType = None  # type: ignore[assignment, misc]
    _ExtMatchKey = None  # type: ignore[assignment, misc]
    _ExtMatchResult = None  # type: ignore[assignment, misc]
    _ExtBatchMatchResult = None  # type: ignore[assignment, misc]
    _ExtSourceDefinition = None  # type: ignore[assignment, misc]
    _ExtSchemaMapping = None  # type: ignore[assignment, misc]
    _MODELS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Local metric helper stubs (delegate when metrics module is present)
# ---------------------------------------------------------------------------

def _inc_records_matched(strategy: str, count: int = 1) -> None:
    """Increment the records-matched counter by strategy.

    Delegates to ``metrics.inc_records_matched(strategy, count)`` when
    the metrics module is available.

    Args:
        strategy: Matching strategy name.
        count: Number of records matched.
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        try:
            _metrics_mod.inc_records_matched(strategy, count)
        except AttributeError:
            pass


def _observe_match_confidence(confidence: float) -> None:
    """Observe a match confidence score.

    Delegates to ``metrics.observe_confidence(confidence)`` when the
    metrics module is available.

    Args:
        confidence: Confidence value (0.0-1.0).
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        try:
            _metrics_mod.observe_confidence(confidence)
        except AttributeError:
            pass


def _observe_duration(operation: str, duration: float) -> None:
    """Observe processing duration in seconds.

    The underlying ``metrics.observe_duration`` accepts only the
    duration value (no operation label), so the operation argument is
    logged but not forwarded.

    Args:
        operation: Operation label (logged, not forwarded).
        duration: Duration in seconds.
    """
    if _METRICS_AVAILABLE and _metrics_mod is not None:
        try:
            _metrics_mod.observe_duration(duration)
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Lightweight provenance wrapper
# ---------------------------------------------------------------------------

class _InternalProvenanceTracker:
    """Minimal provenance tracker used when the full SDK module is absent.

    Provides ``build_hash`` and ``add_to_chain`` methods that mirror the
    real ProvenanceTracker API so the engine code can call them without
    conditional branches.
    """

    GENESIS_HASH: str = hashlib.sha256(
        b"greenlang-cross-source-reconciliation-genesis"
    ).hexdigest()

    def __init__(self) -> None:
        """Initialize internal provenance tracker."""
        self._last_hash: str = self.GENESIS_HASH

    def build_hash(self, data: Any) -> str:
        """Compute a deterministic SHA-256 hash.

        Args:
            data: Arbitrary data to hash.

        Returns:
            Hex-encoded SHA-256 string.
        """
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def add_to_chain(
        self,
        operation: str,
        input_hash: str,
        output_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Append an operation to the chain and return the new chain hash.

        Args:
            operation: Name of the operation.
            input_hash: SHA-256 hash of the input.
            output_hash: SHA-256 hash of the output.
            metadata: Optional extra metadata.

        Returns:
            New chain hash.
        """
        combined = json.dumps({
            "previous": self._last_hash,
            "input": input_hash,
            "output": output_hash,
            "operation": operation,
            "timestamp": _utcnow().isoformat(),
        }, sort_keys=True)
        chain_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        self._last_hash = chain_hash
        return chain_hash


def _get_provenance_tracker() -> Any:
    """Return the best available provenance tracker instance.

    Prefers the SDK singleton; falls back to the lightweight internal one.

    Returns:
        A provenance tracker with ``build_hash`` and ``add_to_chain``.
    """
    if _PROVENANCE_AVAILABLE and _ext_get_provenance_tracker is not None:
        try:
            return _ext_get_provenance_tracker()
        except Exception:
            pass
    return _InternalProvenanceTracker()


# ---------------------------------------------------------------------------
# Local enumerations (used when models.py is absent)
# ---------------------------------------------------------------------------


class MatchStrategy(str, Enum):
    """Strategy for cross-source record matching.

    EXACT: Composite key equality on entity+period+metric.
    FUZZY: Jaro-Winkler / numeric / date similarity with threshold.
    COMPOSITE: Multi-field weighted matching with per-field weights.
    TEMPORAL: Align records to common temporal granularity then match.
    BLOCKING: Hash-based blocking to reduce comparison space.
    """

    EXACT = "exact"
    FUZZY = "fuzzy"
    COMPOSITE = "composite"
    TEMPORAL = "temporal"
    BLOCKING = "blocking"


class MatchStatus(str, Enum):
    """Status of a match result.

    MATCHED: Records were successfully matched.
    UNMATCHED_A: Record from source A has no match in source B.
    UNMATCHED_B: Record from source B has no match in source A.
    AMBIGUOUS: Multiple candidate matches above threshold.
    BELOW_THRESHOLD: Best candidate score is below the threshold.
    """

    MATCHED = "matched"
    UNMATCHED_A = "unmatched_a"
    UNMATCHED_B = "unmatched_b"
    AMBIGUOUS = "ambiguous"
    BELOW_THRESHOLD = "below_threshold"


class TemporalGranularity(str, Enum):
    """Temporal reporting granularity for period alignment.

    DAILY: Daily granularity.
    WEEKLY: Weekly granularity.
    MONTHLY: Monthly granularity.
    QUARTERLY: Quarterly granularity.
    ANNUAL: Annual granularity.
    """

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class FieldType(str, Enum):
    """Field data type for match-key computation.

    STRING: Text field compared via string similarity.
    NUMERIC: Number field compared via numeric proximity.
    DATE: Date/datetime field compared via day proximity.
    CATEGORICAL: Categorical field compared via exact equality.
    """

    STRING = "string"
    NUMERIC = "numeric"
    DATE = "date"
    CATEGORICAL = "categorical"


# ---------------------------------------------------------------------------
# Local data models (used when models.py is absent)
# ---------------------------------------------------------------------------


@dataclass
class MatchKey:
    """Composite key extracted from a record for matching.

    Attributes:
        fields: Ordered mapping of field name to normalised value.
        composite_key: Concatenated string representation.
        provenance_hash: SHA-256 hash of the composite key.
    """

    fields: Dict[str, Any] = field(default_factory=dict)
    composite_key: str = ""
    provenance_hash: str = ""


@dataclass
class MatchResult:
    """Result of matching a single pair of records.

    Attributes:
        match_id: Unique identifier for this match.
        record_a: Record from source A.
        record_b: Record from source B (None if unmatched).
        source_a_id: Identifier for source A.
        source_b_id: Identifier for source B.
        status: Match status enum value.
        confidence: Match confidence score (0.0-1.0).
        strategy: Strategy used for this match.
        field_scores: Per-field similarity scores.
        match_key: Composite match key used.
        processing_time_ms: Wall-clock processing time in ms.
        provenance_hash: SHA-256 provenance hash.
        created_at: ISO-formatted creation timestamp.
        metadata: Optional additional metadata.
    """

    match_id: str = ""
    record_a: Dict[str, Any] = field(default_factory=dict)
    record_b: Optional[Dict[str, Any]] = None
    source_a_id: str = ""
    source_b_id: str = ""
    status: str = "matched"
    confidence: float = 0.0
    strategy: str = "exact"
    field_scores: Dict[str, float] = field(default_factory=dict)
    match_key: str = ""
    processing_time_ms: float = 0.0
    provenance_hash: str = ""
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchMatchResult:
    """Aggregate result of matching across multiple source pairs.

    Attributes:
        batch_id: Unique identifier for this batch.
        total_records_a: Total records from all A-side sources.
        total_records_b: Total records from all B-side sources.
        total_matched: Number of successfully matched pairs.
        total_unmatched_a: Records in A with no match in B.
        total_unmatched_b: Records in B with no match in A.
        total_ambiguous: Matches with multiple candidates.
        match_results: All individual MatchResult objects.
        pair_stats: Per-source-pair statistics.
        mean_confidence: Average confidence across all matches.
        min_confidence: Minimum confidence across all matches.
        processing_time_ms: Total wall-clock processing time in ms.
        provenance_hash: SHA-256 provenance hash for the batch.
        created_at: ISO-formatted creation timestamp.
    """

    batch_id: str = ""
    total_records_a: int = 0
    total_records_b: int = 0
    total_matched: int = 0
    total_unmatched_a: int = 0
    total_unmatched_b: int = 0
    total_ambiguous: int = 0
    match_results: List[MatchResult] = field(default_factory=list)
    pair_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    mean_confidence: float = 0.0
    min_confidence: float = 1.0
    processing_time_ms: float = 0.0
    provenance_hash: str = ""
    created_at: str = ""


@dataclass
class SourceDefinition:
    """Metadata about a registered data source.

    Attributes:
        source_id: Unique identifier for the source.
        name: Human-readable name.
        source_type: Type of data source (erp, utility, meter, etc.).
        priority: Priority rank (lower = higher priority).
        credibility_score: Credibility score (0.0-1.0).
    """

    source_id: str = ""
    name: str = ""
    source_type: str = ""
    priority: int = 0
    credibility_score: float = 1.0


@dataclass
class SchemaMapping:
    """Mapping between a source column and a canonical column.

    Attributes:
        source_column: Column name in the source.
        canonical_column: Canonical column name.
        transform: Optional transformation to apply.
        field_type: Data type of the field.
    """

    source_column: str = ""
    canonical_column: str = ""
    transform: str = ""
    field_type: str = "string"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Small epsilon to avoid division by zero.
_EPS: float = 1e-15

#: Default Jaro-Winkler common prefix scaling factor.
_JW_PREFIX_SCALE: float = 0.1

#: Maximum prefix length considered for Jaro-Winkler bonus.
_JW_MAX_PREFIX: int = 4

#: Default maximum numeric difference for proximity scoring.
_DEFAULT_MAX_DIFF: float = 1000.0

#: Default maximum day difference for date proximity scoring.
_DEFAULT_MAX_DAYS: int = 365

#: Default match confidence threshold.
_DEFAULT_THRESHOLD: float = 0.85

#: Default blocking prefix length (first N chars of each field).
_DEFAULT_BLOCK_PREFIX_LEN: int = 3

#: Temporal granularity ordering (finest to coarsest).
_GRANULARITY_ORDER: Dict[str, int] = {
    "daily": 0,
    "weekly": 1,
    "monthly": 2,
    "quarterly": 3,
    "annual": 4,
}

#: Months per temporal granularity (for aggregation alignment).
_MONTHS_PER_GRANULARITY: Dict[str, int] = {
    "daily": 0,
    "weekly": 0,
    "monthly": 1,
    "quarterly": 3,
    "annual": 12,
}


# ============================================================================
# MatchingEngine
# ============================================================================


class MatchingEngine:
    """Cross-source record matching engine.

    Implements five matching strategies (exact, fuzzy, composite, temporal,
    blocking) for reconciling records across different data sources.
    Produces MatchResult objects with confidence scores and SHA-256
    provenance hashes for every match decision.

    Zero-Hallucination: All similarity computations (Jaro-Winkler,
    Levenshtein, numeric proximity, date proximity) use deterministic
    Python arithmetic via the ``math`` module. No external ML libraries
    or LLM calls. Match confidence is a deterministic function of field
    similarity scores.

    Attributes:
        _provenance: SHA-256 provenance tracker.
        _config: Optional configuration object.

    Example:
        >>> engine = MatchingEngine()
        >>> results = engine.match_records(
        ...     source_a_records=[{"entity_id": "E1", "period": "2025-Q1"}],
        ...     source_b_records=[{"entity_id": "E1", "period": "2025-Q1"}],
        ...     key_fields=["entity_id", "period"],
        ...     strategy="exact",
        ... )
        >>> assert results[0].confidence == 1.0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize MatchingEngine.

        Args:
            config: Optional configuration override. Falls back to the
                SDK singleton from ``get_config()`` when available, or
                defaults when the config module is absent.
        """
        if config is not None:
            self._config = config
        elif _CONFIG_AVAILABLE and _ext_get_config is not None:
            try:
                self._config = _ext_get_config()
            except Exception:
                self._config = None
        else:
            self._config = None

        self._provenance = _get_provenance_tracker()
        logger.info("MatchingEngine initialized")

    # ------------------------------------------------------------------
    # Public API: match_records dispatcher
    # ------------------------------------------------------------------

    def match_records(
        self,
        source_a_records: List[Dict[str, Any]],
        source_b_records: List[Dict[str, Any]],
        key_fields: List[str],
        strategy: str = "exact",
        threshold: float = _DEFAULT_THRESHOLD,
        source_a_id: str = "source_a",
        source_b_id: str = "source_b",
        field_weights: Optional[Dict[str, float]] = None,
        blocking_fields: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[MatchResult]:
        """Match records from two sources using the specified strategy.

        This is the primary dispatcher that routes to strategy-specific
        matching methods based on the ``strategy`` parameter.

        Args:
            source_a_records: List of record dicts from source A.
            source_b_records: List of record dicts from source B.
            key_fields: List of field names to use as match keys.
            strategy: One of ``exact``, ``fuzzy``, ``composite``,
                ``temporal``, ``blocking``.
            threshold: Minimum confidence score (0.0-1.0) to accept
                a match. Ignored for exact matching.
            source_a_id: Identifier for source A.
            source_b_id: Identifier for source B.
            field_weights: Per-field weights for composite matching.
                Defaults to equal weights when None.
            blocking_fields: Fields used for hash-based blocking.
                Required for blocking strategy. Defaults to key_fields
                when None.
            **kwargs: Additional arguments forwarded to strategy methods.

        Returns:
            List of MatchResult objects with confidence scores and
            provenance hashes.

        Raises:
            ValueError: If strategy name is not recognised.
        """
        start_t = time.time()
        strategy_lower = strategy.lower().strip()

        logger.info(
            "match_records: strategy=%s, source_a=%d records, "
            "source_b=%d records, key_fields=%s, threshold=%.3f",
            strategy_lower,
            len(source_a_records),
            len(source_b_records),
            key_fields,
            threshold,
        )

        dispatcher: Dict[str, Any] = {
            "exact": self._dispatch_exact,
            "fuzzy": self._dispatch_fuzzy,
            "composite": self._dispatch_composite,
            "temporal": self._dispatch_temporal,
            "blocking": self._dispatch_blocking,
        }

        fn = dispatcher.get(strategy_lower)
        if fn is None:
            raise ValueError(
                f"Unrecognised matching strategy: {strategy!r}. "
                f"Supported: {sorted(dispatcher.keys())}"
            )

        results = fn(
            source_a_records=source_a_records,
            source_b_records=source_b_records,
            key_fields=key_fields,
            threshold=threshold,
            source_a_id=source_a_id,
            source_b_id=source_b_id,
            field_weights=field_weights,
            blocking_fields=blocking_fields,
            **kwargs,
        )

        elapsed_ms = (time.time() - start_t) * 1000.0
        matched_count = sum(1 for r in results if r.status == MatchStatus.MATCHED.value)

        # Record provenance for the overall match operation
        self._record_operation_provenance(
            strategy_lower, source_a_id, source_b_id,
            len(source_a_records), len(source_b_records),
            matched_count, results,
        )

        # Emit metrics
        _inc_records_matched(strategy_lower, matched_count)
        _observe_duration(f"match_{strategy_lower}", elapsed_ms / 1000.0)

        logger.info(
            "match_records complete: strategy=%s, matched=%d/%d, "
            "elapsed=%.1fms",
            strategy_lower,
            matched_count,
            len(source_a_records),
            elapsed_ms,
        )

        return results

    # ------------------------------------------------------------------
    # Dispatcher helpers (unpack kwargs to typed calls)
    # ------------------------------------------------------------------

    def _dispatch_exact(
        self,
        source_a_records: List[Dict[str, Any]],
        source_b_records: List[Dict[str, Any]],
        key_fields: List[str],
        source_a_id: str,
        source_b_id: str,
        **kwargs: Any,
    ) -> List[MatchResult]:
        """Dispatch to exact matching.

        Args:
            source_a_records: Records from source A.
            source_b_records: Records from source B.
            key_fields: Fields for composite key.
            source_a_id: Source A identifier.
            source_b_id: Source B identifier.
            **kwargs: Ignored extra arguments.

        Returns:
            List of MatchResult objects.
        """
        return self.match_exact(
            source_a_records, source_b_records,
            key_fields, source_a_id, source_b_id,
        )

    def _dispatch_fuzzy(
        self,
        source_a_records: List[Dict[str, Any]],
        source_b_records: List[Dict[str, Any]],
        key_fields: List[str],
        threshold: float,
        source_a_id: str,
        source_b_id: str,
        **kwargs: Any,
    ) -> List[MatchResult]:
        """Dispatch to fuzzy matching.

        Args:
            source_a_records: Records from source A.
            source_b_records: Records from source B.
            key_fields: Fields for fuzzy comparison.
            threshold: Minimum similarity threshold.
            source_a_id: Source A identifier.
            source_b_id: Source B identifier.
            **kwargs: Ignored extra arguments.

        Returns:
            List of MatchResult objects.
        """
        return self.match_fuzzy(
            source_a_records, source_b_records,
            key_fields, threshold, source_a_id, source_b_id,
        )

    def _dispatch_composite(
        self,
        source_a_records: List[Dict[str, Any]],
        source_b_records: List[Dict[str, Any]],
        key_fields: List[str],
        threshold: float,
        source_a_id: str,
        source_b_id: str,
        field_weights: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> List[MatchResult]:
        """Dispatch to composite matching.

        Args:
            source_a_records: Records from source A.
            source_b_records: Records from source B.
            key_fields: Fields for composite comparison.
            threshold: Minimum weighted score threshold.
            source_a_id: Source A identifier.
            source_b_id: Source B identifier.
            field_weights: Per-field weights.
            **kwargs: Ignored extra arguments.

        Returns:
            List of MatchResult objects.
        """
        weights = field_weights or {f: 1.0 for f in key_fields}
        return self.match_composite(
            source_a_records, source_b_records,
            key_fields, weights, threshold,
            source_a_id, source_b_id,
        )

    def _dispatch_temporal(
        self,
        source_a_records: List[Dict[str, Any]],
        source_b_records: List[Dict[str, Any]],
        key_fields: List[str],
        threshold: float,
        source_a_id: str,
        source_b_id: str,
        **kwargs: Any,
    ) -> List[MatchResult]:
        """Dispatch to temporal matching.

        Temporal matching uses ``entity_field``, ``value_field``,
        ``timestamp_field``, ``granularity_a``, ``granularity_b``, and
        ``target_granularity`` from kwargs when provided, or falls back
        to sensible defaults based on ``key_fields``.

        Args:
            source_a_records: Records from source A.
            source_b_records: Records from source B.
            key_fields: Fields for entity matching after alignment.
            threshold: Minimum confidence threshold.
            source_a_id: Source A identifier.
            source_b_id: Source B identifier.
            **kwargs: Must include temporal parameters.

        Returns:
            List of MatchResult objects.
        """
        entity_field = kwargs.get("entity_field", key_fields[0] if key_fields else "entity_id")
        value_field = kwargs.get("value_field", "value")
        timestamp_field = kwargs.get("timestamp_field", "timestamp")
        granularity_a = kwargs.get("granularity_a", "monthly")
        granularity_b = kwargs.get("granularity_b", "monthly")
        target_granularity = kwargs.get("target_granularity", "quarterly")

        return self.match_temporal(
            source_a_records, source_b_records,
            entity_field, value_field, timestamp_field,
            granularity_a, granularity_b, target_granularity,
            threshold, source_a_id, source_b_id,
        )

    def _dispatch_blocking(
        self,
        source_a_records: List[Dict[str, Any]],
        source_b_records: List[Dict[str, Any]],
        key_fields: List[str],
        threshold: float,
        source_a_id: str,
        source_b_id: str,
        blocking_fields: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[MatchResult]:
        """Dispatch to blocking + matching.

        Args:
            source_a_records: Records from source A.
            source_b_records: Records from source B.
            key_fields: Fields for similarity comparison within blocks.
            threshold: Minimum similarity threshold.
            source_a_id: Source A identifier.
            source_b_id: Source B identifier.
            blocking_fields: Fields used for block keys. Defaults to
                key_fields when None.
            **kwargs: Additional strategy selection (defaults to fuzzy).

        Returns:
            List of MatchResult objects.
        """
        b_fields = blocking_fields or key_fields
        inner_strategy = kwargs.get("inner_strategy", "fuzzy")
        return self.match_with_blocking(
            source_a_records, source_b_records,
            key_fields, b_fields, inner_strategy,
            threshold, source_a_id, source_b_id,
        )

    # ------------------------------------------------------------------
    # 1. Exact Matching
    # ------------------------------------------------------------------

    def match_exact(
        self,
        records_a: List[Dict[str, Any]],
        records_b: List[Dict[str, Any]],
        key_fields: List[str],
        source_a_id: str = "source_a",
        source_b_id: str = "source_b",
    ) -> List[MatchResult]:
        """Match records using exact composite key equality.

        Builds a composite key from ``key_fields`` for every record in
        both sources and matches where keys are identical. Matched
        pairs receive confidence 1.0. Unmatched records from each
        source are included with UNMATCHED_A / UNMATCHED_B status.

        Args:
            records_a: Records from source A.
            records_b: Records from source B.
            key_fields: Field names forming the composite key.
            source_a_id: Identifier for source A.
            source_b_id: Identifier for source B.

        Returns:
            List of MatchResult for matched and unmatched records.
        """
        start_t = time.time()
        results: List[MatchResult] = []

        logger.debug(
            "match_exact: records_a=%d, records_b=%d, key_fields=%s",
            len(records_a), len(records_b), key_fields,
        )

        # Build index for source B keyed by composite key
        b_index: Dict[str, List[Dict[str, Any]]] = {}
        for rec_b in records_b:
            mk = self.compute_match_key(rec_b, key_fields)
            b_index.setdefault(mk.composite_key, []).append(rec_b)

        matched_b_keys: set = set()

        for rec_a in records_a:
            mk_a = self.compute_match_key(rec_a, key_fields)
            candidates = b_index.get(mk_a.composite_key, [])

            if candidates:
                # Take first candidate for 1:1 matching
                rec_b = candidates[0]
                matched_b_keys.add(mk_a.composite_key)

                prov = self._compute_provenance(
                    "match_exact", rec_a, rec_b,
                )
                elapsed = (time.time() - start_t) * 1000.0

                results.append(MatchResult(
                    match_id=str(uuid.uuid4()),
                    record_a=rec_a,
                    record_b=rec_b,
                    source_a_id=source_a_id,
                    source_b_id=source_b_id,
                    status=MatchStatus.MATCHED.value,
                    confidence=1.0,
                    strategy="exact",
                    field_scores={f: 1.0 for f in key_fields},
                    match_key=mk_a.composite_key,
                    processing_time_ms=round(elapsed, 3),
                    provenance_hash=prov,
                    created_at=_utcnow().isoformat(),
                ))

                _observe_match_confidence(1.0)
            else:
                prov = self._compute_provenance(
                    "match_exact_unmatched_a", rec_a, {},
                )
                elapsed = (time.time() - start_t) * 1000.0

                results.append(MatchResult(
                    match_id=str(uuid.uuid4()),
                    record_a=rec_a,
                    record_b=None,
                    source_a_id=source_a_id,
                    source_b_id=source_b_id,
                    status=MatchStatus.UNMATCHED_A.value,
                    confidence=0.0,
                    strategy="exact",
                    field_scores={},
                    match_key=mk_a.composite_key,
                    processing_time_ms=round(elapsed, 3),
                    provenance_hash=prov,
                    created_at=_utcnow().isoformat(),
                ))

        # Unmatched records from source B
        for key_str, b_recs in b_index.items():
            if key_str not in matched_b_keys:
                for rec_b in b_recs:
                    prov = self._compute_provenance(
                        "match_exact_unmatched_b", {}, rec_b,
                    )
                    elapsed = (time.time() - start_t) * 1000.0

                    results.append(MatchResult(
                        match_id=str(uuid.uuid4()),
                        record_a={},
                        record_b=rec_b,
                        source_a_id=source_a_id,
                        source_b_id=source_b_id,
                        status=MatchStatus.UNMATCHED_B.value,
                        confidence=0.0,
                        strategy="exact",
                        field_scores={},
                        match_key=key_str,
                        processing_time_ms=round(
                            (time.time() - start_t) * 1000.0, 3,
                        ),
                        provenance_hash=prov,
                        created_at=_utcnow().isoformat(),
                    ))

        logger.debug(
            "match_exact complete: %d results",
            len(results),
        )
        return results

    # ------------------------------------------------------------------
    # 2. Fuzzy Matching
    # ------------------------------------------------------------------

    def match_fuzzy(
        self,
        records_a: List[Dict[str, Any]],
        records_b: List[Dict[str, Any]],
        key_fields: List[str],
        threshold: float = _DEFAULT_THRESHOLD,
        source_a_id: str = "source_a",
        source_b_id: str = "source_b",
    ) -> List[MatchResult]:
        """Match records using fuzzy similarity on key fields.

        For each record in source A, computes similarity against every
        record in source B using Jaro-Winkler for string fields,
        numeric proximity for numeric fields, and date proximity for
        date fields. The overall score is the unweighted average of
        per-field scores. The best candidate above threshold is selected.

        Args:
            records_a: Records from source A.
            records_b: Records from source B.
            key_fields: Fields to compare for similarity.
            threshold: Minimum average similarity to accept a match.
            source_a_id: Identifier for source A.
            source_b_id: Identifier for source B.

        Returns:
            List of MatchResult for matched and unmatched records.
        """
        start_t = time.time()
        results: List[MatchResult] = []
        matched_b_indices: set = set()

        logger.debug(
            "match_fuzzy: records_a=%d, records_b=%d, threshold=%.3f",
            len(records_a), len(records_b), threshold,
        )

        for rec_a in records_a:
            best_score = 0.0
            best_idx = -1
            best_field_scores: Dict[str, float] = {}

            for j, rec_b in enumerate(records_b):
                if j in matched_b_indices:
                    continue

                field_scores = self._compute_field_similarities(
                    rec_a, rec_b, key_fields,
                )
                if not field_scores:
                    continue

                avg_score = sum(field_scores.values()) / len(field_scores)

                if avg_score > best_score:
                    best_score = avg_score
                    best_idx = j
                    best_field_scores = field_scores

            if best_score >= threshold and best_idx >= 0:
                matched_b_indices.add(best_idx)
                rec_b = records_b[best_idx]
                prov = self._compute_provenance(
                    "match_fuzzy", rec_a, rec_b,
                )
                elapsed = (time.time() - start_t) * 1000.0

                results.append(MatchResult(
                    match_id=str(uuid.uuid4()),
                    record_a=rec_a,
                    record_b=rec_b,
                    source_a_id=source_a_id,
                    source_b_id=source_b_id,
                    status=MatchStatus.MATCHED.value,
                    confidence=round(best_score, 6),
                    strategy="fuzzy",
                    field_scores=best_field_scores,
                    match_key=self.compute_match_key(
                        rec_a, key_fields,
                    ).composite_key,
                    processing_time_ms=round(elapsed, 3),
                    provenance_hash=prov,
                    created_at=_utcnow().isoformat(),
                ))
                _observe_match_confidence(best_score)
            else:
                status = (
                    MatchStatus.BELOW_THRESHOLD.value
                    if best_score > 0
                    else MatchStatus.UNMATCHED_A.value
                )
                prov = self._compute_provenance(
                    "match_fuzzy_unmatched", rec_a, {},
                )
                elapsed = (time.time() - start_t) * 1000.0

                results.append(MatchResult(
                    match_id=str(uuid.uuid4()),
                    record_a=rec_a,
                    record_b=None,
                    source_a_id=source_a_id,
                    source_b_id=source_b_id,
                    status=status,
                    confidence=round(best_score, 6),
                    strategy="fuzzy",
                    field_scores=best_field_scores,
                    match_key=self.compute_match_key(
                        rec_a, key_fields,
                    ).composite_key,
                    processing_time_ms=round(elapsed, 3),
                    provenance_hash=prov,
                    created_at=_utcnow().isoformat(),
                ))

        # Unmatched B records
        for j, rec_b in enumerate(records_b):
            if j not in matched_b_indices:
                prov = self._compute_provenance(
                    "match_fuzzy_unmatched_b", {}, rec_b,
                )
                results.append(MatchResult(
                    match_id=str(uuid.uuid4()),
                    record_a={},
                    record_b=rec_b,
                    source_a_id=source_a_id,
                    source_b_id=source_b_id,
                    status=MatchStatus.UNMATCHED_B.value,
                    confidence=0.0,
                    strategy="fuzzy",
                    field_scores={},
                    match_key=self.compute_match_key(
                        rec_b, key_fields,
                    ).composite_key,
                    processing_time_ms=round(
                        (time.time() - start_t) * 1000.0, 3,
                    ),
                    provenance_hash=prov,
                    created_at=_utcnow().isoformat(),
                ))

        logger.debug("match_fuzzy complete: %d results", len(results))
        return results

    # ------------------------------------------------------------------
    # 3. Composite (Weighted Multi-Field) Matching
    # ------------------------------------------------------------------

    def match_composite(
        self,
        records_a: List[Dict[str, Any]],
        records_b: List[Dict[str, Any]],
        key_fields: List[str],
        field_weights: Dict[str, float],
        threshold: float = _DEFAULT_THRESHOLD,
        source_a_id: str = "source_a",
        source_b_id: str = "source_b",
    ) -> List[MatchResult]:
        """Match records using multi-field weighted composite scoring.

        Each field contributes a similarity score multiplied by its
        weight. The final score is the weighted average:
        ``score = sum(weight_i * sim_i) / sum(weight_i)``.

        Args:
            records_a: Records from source A.
            records_b: Records from source B.
            key_fields: Fields to compare.
            field_weights: Weight for each field (higher = more
                important). Fields not in this dict get weight 1.0.
            threshold: Minimum weighted score to accept.
            source_a_id: Identifier for source A.
            source_b_id: Identifier for source B.

        Returns:
            List of MatchResult for matched and unmatched records.
        """
        start_t = time.time()
        results: List[MatchResult] = []
        matched_b_indices: set = set()

        logger.debug(
            "match_composite: records_a=%d, records_b=%d, "
            "weights=%s, threshold=%.3f",
            len(records_a), len(records_b), field_weights, threshold,
        )

        for rec_a in records_a:
            best_score = 0.0
            best_idx = -1
            best_field_scores: Dict[str, float] = {}

            for j, rec_b in enumerate(records_b):
                if j in matched_b_indices:
                    continue

                field_scores = self._compute_field_similarities(
                    rec_a, rec_b, key_fields,
                )
                if not field_scores:
                    continue

                # Weighted average
                weighted_sum = 0.0
                weight_total = 0.0
                for f_name, f_sim in field_scores.items():
                    w = field_weights.get(f_name, 1.0)
                    weighted_sum += w * f_sim
                    weight_total += w

                if weight_total > _EPS:
                    score = weighted_sum / weight_total
                else:
                    score = 0.0

                if score > best_score:
                    best_score = score
                    best_idx = j
                    best_field_scores = field_scores

            if best_score >= threshold and best_idx >= 0:
                matched_b_indices.add(best_idx)
                rec_b = records_b[best_idx]
                prov = self._compute_provenance(
                    "match_composite", rec_a, rec_b,
                )
                elapsed = (time.time() - start_t) * 1000.0

                results.append(MatchResult(
                    match_id=str(uuid.uuid4()),
                    record_a=rec_a,
                    record_b=rec_b,
                    source_a_id=source_a_id,
                    source_b_id=source_b_id,
                    status=MatchStatus.MATCHED.value,
                    confidence=round(best_score, 6),
                    strategy="composite",
                    field_scores=best_field_scores,
                    match_key=self.compute_match_key(
                        rec_a, key_fields,
                    ).composite_key,
                    processing_time_ms=round(elapsed, 3),
                    provenance_hash=prov,
                    created_at=_utcnow().isoformat(),
                    metadata={"field_weights": field_weights},
                ))
                _observe_match_confidence(best_score)
            else:
                status = (
                    MatchStatus.BELOW_THRESHOLD.value
                    if best_score > 0
                    else MatchStatus.UNMATCHED_A.value
                )
                prov = self._compute_provenance(
                    "match_composite_unmatched", rec_a, {},
                )
                elapsed = (time.time() - start_t) * 1000.0

                results.append(MatchResult(
                    match_id=str(uuid.uuid4()),
                    record_a=rec_a,
                    record_b=None,
                    source_a_id=source_a_id,
                    source_b_id=source_b_id,
                    status=status,
                    confidence=round(best_score, 6),
                    strategy="composite",
                    field_scores=best_field_scores,
                    match_key=self.compute_match_key(
                        rec_a, key_fields,
                    ).composite_key,
                    processing_time_ms=round(elapsed, 3),
                    provenance_hash=prov,
                    created_at=_utcnow().isoformat(),
                    metadata={"field_weights": field_weights},
                ))

        # Unmatched B records
        for j, rec_b in enumerate(records_b):
            if j not in matched_b_indices:
                prov = self._compute_provenance(
                    "match_composite_unmatched_b", {}, rec_b,
                )
                results.append(MatchResult(
                    match_id=str(uuid.uuid4()),
                    record_a={},
                    record_b=rec_b,
                    source_a_id=source_a_id,
                    source_b_id=source_b_id,
                    status=MatchStatus.UNMATCHED_B.value,
                    confidence=0.0,
                    strategy="composite",
                    field_scores={},
                    match_key=self.compute_match_key(
                        rec_b, key_fields,
                    ).composite_key,
                    processing_time_ms=round(
                        (time.time() - start_t) * 1000.0, 3,
                    ),
                    provenance_hash=prov,
                    created_at=_utcnow().isoformat(),
                ))

        logger.debug("match_composite complete: %d results", len(results))
        return results

    # ------------------------------------------------------------------
    # 4. Temporal Matching
    # ------------------------------------------------------------------

    def match_temporal(
        self,
        records_a: List[Dict[str, Any]],
        records_b: List[Dict[str, Any]],
        entity_field: str,
        value_field: str,
        timestamp_field: str,
        granularity_a: str,
        granularity_b: str,
        target_granularity: str,
        threshold: float = _DEFAULT_THRESHOLD,
        source_a_id: str = "source_a",
        source_b_id: str = "source_b",
    ) -> List[MatchResult]:
        """Match records after aligning to a common temporal granularity.

        Records from both sources are aggregated to the
        ``target_granularity`` (e.g. monthly records summed to quarterly)
        and then matched on entity + aligned period. Value comparison
        uses numeric proximity for confidence scoring.

        Supported granularity conversions:
            - daily -> monthly, quarterly, annual
            - monthly -> quarterly, annual
            - quarterly -> annual

        Args:
            records_a: Records from source A.
            records_b: Records from source B.
            entity_field: Field name for entity identifier.
            value_field: Field name for the numeric value.
            timestamp_field: Field name for the timestamp/period.
            granularity_a: Temporal granularity of source A records.
            granularity_b: Temporal granularity of source B records.
            target_granularity: Target granularity for alignment.
            threshold: Minimum confidence threshold.
            source_a_id: Identifier for source A.
            source_b_id: Identifier for source B.

        Returns:
            List of MatchResult for aligned and matched records.
        """
        start_t = time.time()
        results: List[MatchResult] = []

        logger.debug(
            "match_temporal: records_a=%d (%s), records_b=%d (%s), "
            "target=%s",
            len(records_a), granularity_a,
            len(records_b), granularity_b,
            target_granularity,
        )

        # Aggregate both sources to target granularity
        agg_a = self._aggregate_temporal(
            records_a, entity_field, value_field,
            timestamp_field, granularity_a, target_granularity,
        )
        agg_b = self._aggregate_temporal(
            records_b, entity_field, value_field,
            timestamp_field, granularity_b, target_granularity,
        )

        # Build index for aggregated B by entity+period
        b_index: Dict[str, Dict[str, Any]] = {}
        for rec_b in agg_b:
            key = f"{rec_b.get(entity_field, '')}||{rec_b.get('aligned_period', '')}"
            b_index[key] = rec_b

        matched_b_keys: set = set()

        for rec_a in agg_a:
            key = f"{rec_a.get(entity_field, '')}||{rec_a.get('aligned_period', '')}"
            rec_b = b_index.get(key)

            if rec_b is not None:
                matched_b_keys.add(key)

                # Compute value proximity for confidence
                val_a = self._to_float(rec_a.get(value_field, 0))
                val_b = self._to_float(rec_b.get(value_field, 0))
                max_val = max(abs(val_a), abs(val_b), _EPS)
                confidence = 1.0 - min(abs(val_a - val_b) / max_val, 1.0)

                prov = self._compute_provenance(
                    "match_temporal", rec_a, rec_b,
                )
                elapsed = (time.time() - start_t) * 1000.0

                status = (
                    MatchStatus.MATCHED.value
                    if confidence >= threshold
                    else MatchStatus.BELOW_THRESHOLD.value
                )

                results.append(MatchResult(
                    match_id=str(uuid.uuid4()),
                    record_a=rec_a,
                    record_b=rec_b,
                    source_a_id=source_a_id,
                    source_b_id=source_b_id,
                    status=status,
                    confidence=round(confidence, 6),
                    strategy="temporal",
                    field_scores={
                        entity_field: 1.0,
                        "aligned_period": 1.0,
                        value_field: round(confidence, 6),
                    },
                    match_key=key,
                    processing_time_ms=round(elapsed, 3),
                    provenance_hash=prov,
                    created_at=_utcnow().isoformat(),
                    metadata={
                        "granularity_a": granularity_a,
                        "granularity_b": granularity_b,
                        "target_granularity": target_granularity,
                    },
                ))
                _observe_match_confidence(confidence)
            else:
                prov = self._compute_provenance(
                    "match_temporal_unmatched_a", rec_a, {},
                )
                elapsed = (time.time() - start_t) * 1000.0

                results.append(MatchResult(
                    match_id=str(uuid.uuid4()),
                    record_a=rec_a,
                    record_b=None,
                    source_a_id=source_a_id,
                    source_b_id=source_b_id,
                    status=MatchStatus.UNMATCHED_A.value,
                    confidence=0.0,
                    strategy="temporal",
                    field_scores={},
                    match_key=key,
                    processing_time_ms=round(elapsed, 3),
                    provenance_hash=prov,
                    created_at=_utcnow().isoformat(),
                ))

        # Unmatched B records
        for key, rec_b in b_index.items():
            if key not in matched_b_keys:
                prov = self._compute_provenance(
                    "match_temporal_unmatched_b", {}, rec_b,
                )
                results.append(MatchResult(
                    match_id=str(uuid.uuid4()),
                    record_a={},
                    record_b=rec_b,
                    source_a_id=source_a_id,
                    source_b_id=source_b_id,
                    status=MatchStatus.UNMATCHED_B.value,
                    confidence=0.0,
                    strategy="temporal",
                    field_scores={},
                    match_key=key,
                    processing_time_ms=round(
                        (time.time() - start_t) * 1000.0, 3,
                    ),
                    provenance_hash=prov,
                    created_at=_utcnow().isoformat(),
                ))

        logger.debug("match_temporal complete: %d results", len(results))
        return results

    # ------------------------------------------------------------------
    # 5. Blocking + Matching
    # ------------------------------------------------------------------

    def match_with_blocking(
        self,
        records_a: List[Dict[str, Any]],
        records_b: List[Dict[str, Any]],
        key_fields: List[str],
        blocking_fields: List[str],
        strategy: str = "fuzzy",
        threshold: float = _DEFAULT_THRESHOLD,
        source_a_id: str = "source_a",
        source_b_id: str = "source_b",
    ) -> List[MatchResult]:
        """Match records using hash-based blocking to reduce comparisons.

        Creates blocks using the first N characters of each blocking
        field. Only records within the same block are compared, reducing
        the O(n*m) comparison space to smaller within-block subsets.

        After blocking, the specified ``strategy`` (exact or fuzzy) is
        applied within each block.

        Args:
            records_a: Records from source A.
            records_b: Records from source B.
            key_fields: Fields for similarity comparison within blocks.
            blocking_fields: Fields used to build block keys.
            strategy: Inner matching strategy (``exact`` or ``fuzzy``).
            threshold: Minimum similarity threshold (for fuzzy).
            source_a_id: Identifier for source A.
            source_b_id: Identifier for source B.

        Returns:
            List of MatchResult for matched and unmatched records.
        """
        start_t = time.time()

        logger.debug(
            "match_with_blocking: records_a=%d, records_b=%d, "
            "blocking_fields=%s, inner_strategy=%s",
            len(records_a), len(records_b), blocking_fields, strategy,
        )

        # Create blocks for both sources
        blocks_a = self._create_blocks(records_a, blocking_fields)
        blocks_b = self._create_blocks(records_b, blocking_fields)

        all_results: List[MatchResult] = []
        matched_a_indices: set = set()
        matched_b_indices: set = set()

        # Process each block that exists in both sources
        all_block_keys = set(blocks_a.keys()) | set(blocks_b.keys())

        for block_key in sorted(all_block_keys):
            block_recs_a = blocks_a.get(block_key, [])
            block_recs_b = blocks_b.get(block_key, [])

            if not block_recs_a or not block_recs_b:
                # One side is empty in this block; produce unmatched
                for rec_a in block_recs_a:
                    rec_a_id = id(rec_a)
                    if rec_a_id not in matched_a_indices:
                        matched_a_indices.add(rec_a_id)
                        prov = self._compute_provenance(
                            "match_blocking_unmatched_a", rec_a, {},
                        )
                        all_results.append(MatchResult(
                            match_id=str(uuid.uuid4()),
                            record_a=rec_a,
                            record_b=None,
                            source_a_id=source_a_id,
                            source_b_id=source_b_id,
                            status=MatchStatus.UNMATCHED_A.value,
                            confidence=0.0,
                            strategy="blocking",
                            field_scores={},
                            match_key=block_key,
                            processing_time_ms=round(
                                (time.time() - start_t) * 1000.0, 3,
                            ),
                            provenance_hash=prov,
                            created_at=_utcnow().isoformat(),
                            metadata={"block_key": block_key},
                        ))
                for rec_b in block_recs_b:
                    rec_b_id = id(rec_b)
                    if rec_b_id not in matched_b_indices:
                        matched_b_indices.add(rec_b_id)
                        prov = self._compute_provenance(
                            "match_blocking_unmatched_b", {}, rec_b,
                        )
                        all_results.append(MatchResult(
                            match_id=str(uuid.uuid4()),
                            record_a={},
                            record_b=rec_b,
                            source_a_id=source_a_id,
                            source_b_id=source_b_id,
                            status=MatchStatus.UNMATCHED_B.value,
                            confidence=0.0,
                            strategy="blocking",
                            field_scores={},
                            match_key=block_key,
                            processing_time_ms=round(
                                (time.time() - start_t) * 1000.0, 3,
                            ),
                            provenance_hash=prov,
                            created_at=_utcnow().isoformat(),
                            metadata={"block_key": block_key},
                        ))
                continue

            # Match within the block using the inner strategy
            if strategy == "exact":
                block_results = self.match_exact(
                    block_recs_a, block_recs_b,
                    key_fields, source_a_id, source_b_id,
                )
            else:
                block_results = self.match_fuzzy(
                    block_recs_a, block_recs_b,
                    key_fields, threshold, source_a_id, source_b_id,
                )

            # Tag results with blocking metadata
            for result in block_results:
                result.strategy = "blocking"
                result.metadata["block_key"] = block_key
                result.metadata["inner_strategy"] = strategy

                # Track matched indices to avoid duplicates
                if result.record_a:
                    matched_a_indices.add(id(result.record_a))
                if result.record_b:
                    matched_b_indices.add(id(result.record_b))

            all_results.extend(block_results)

        logger.debug(
            "match_with_blocking complete: %d results from %d blocks",
            len(all_results), len(all_block_keys),
        )
        return all_results

    # ------------------------------------------------------------------
    # 6. Batch Matching (Multi-Source)
    # ------------------------------------------------------------------

    def match_batch(
        self,
        source_records_map: Dict[str, List[Dict[str, Any]]],
        key_fields: List[str],
        strategy: str = "exact",
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> BatchMatchResult:
        """Match records across all source pairs in a batch.

        For N sources, performs pairwise matching across all N*(N-1)/2
        unique pairs. Applies transitive linking: if A matches B and
        B matches C, then A is linked to C.

        Args:
            source_records_map: Mapping of source_id to list of records.
            key_fields: Fields for matching.
            strategy: Matching strategy to use.
            threshold: Minimum confidence threshold.

        Returns:
            BatchMatchResult with aggregate statistics and all individual
            match results.
        """
        start_t = time.time()
        batch_id = str(uuid.uuid4())

        source_ids = sorted(source_records_map.keys())
        all_results: List[MatchResult] = []
        pair_stats: Dict[str, Dict[str, Any]] = {}

        total_a = 0
        total_b = 0

        logger.info(
            "match_batch: %d sources, strategy=%s, threshold=%.3f",
            len(source_ids), strategy, threshold,
        )

        # Pairwise matching across all source pairs
        for i in range(len(source_ids)):
            for j in range(i + 1, len(source_ids)):
                src_a_id = source_ids[i]
                src_b_id = source_ids[j]
                recs_a = source_records_map[src_a_id]
                recs_b = source_records_map[src_b_id]

                total_a += len(recs_a)
                total_b += len(recs_b)

                pair_results = self.match_records(
                    recs_a, recs_b, key_fields,
                    strategy=strategy,
                    threshold=threshold,
                    source_a_id=src_a_id,
                    source_b_id=src_b_id,
                )
                all_results.extend(pair_results)

                # Compute pair statistics
                matched = sum(
                    1 for r in pair_results
                    if r.status == MatchStatus.MATCHED.value
                )
                unmatched_a = sum(
                    1 for r in pair_results
                    if r.status == MatchStatus.UNMATCHED_A.value
                )
                unmatched_b = sum(
                    1 for r in pair_results
                    if r.status == MatchStatus.UNMATCHED_B.value
                )
                confidences = [
                    r.confidence for r in pair_results
                    if r.status == MatchStatus.MATCHED.value
                ]

                pair_key = f"{src_a_id}||{src_b_id}"
                pair_stats[pair_key] = {
                    "source_a_id": src_a_id,
                    "source_b_id": src_b_id,
                    "records_a": len(recs_a),
                    "records_b": len(recs_b),
                    "matched": matched,
                    "unmatched_a": unmatched_a,
                    "unmatched_b": unmatched_b,
                    "mean_confidence": (
                        round(sum(confidences) / len(confidences), 6)
                        if confidences else 0.0
                    ),
                }

        # Transitive linking via union-find
        all_results = self._apply_transitive_links(all_results)

        # Aggregate statistics
        total_matched = sum(
            1 for r in all_results
            if r.status == MatchStatus.MATCHED.value
        )
        total_unmatched_a = sum(
            1 for r in all_results
            if r.status == MatchStatus.UNMATCHED_A.value
        )
        total_unmatched_b = sum(
            1 for r in all_results
            if r.status == MatchStatus.UNMATCHED_B.value
        )
        total_ambiguous = sum(
            1 for r in all_results
            if r.status == MatchStatus.AMBIGUOUS.value
        )

        matched_confidences = [
            r.confidence for r in all_results
            if r.status == MatchStatus.MATCHED.value
        ]
        mean_conf = (
            round(sum(matched_confidences) / len(matched_confidences), 6)
            if matched_confidences else 0.0
        )
        min_conf = (
            round(min(matched_confidences), 6)
            if matched_confidences else 1.0
        )

        elapsed_ms = (time.time() - start_t) * 1000.0

        # Build provenance for the batch
        prov_input = self._provenance.build_hash({
            "operation": "match_batch_input",
            "sources": source_ids,
            "strategy": strategy,
            "threshold": threshold,
            "key_fields": key_fields,
        })
        prov_output = self._provenance.build_hash({
            "operation": "match_batch_output",
            "batch_id": batch_id,
            "total_matched": total_matched,
            "mean_confidence": mean_conf,
        })
        batch_prov = self._provenance.add_to_chain(
            operation="match_batch",
            input_hash=prov_input,
            output_hash=prov_output,
            metadata={
                "sources": source_ids,
                "strategy": strategy,
                "total_matched": total_matched,
            },
        )

        result = BatchMatchResult(
            batch_id=batch_id,
            total_records_a=total_a,
            total_records_b=total_b,
            total_matched=total_matched,
            total_unmatched_a=total_unmatched_a,
            total_unmatched_b=total_unmatched_b,
            total_ambiguous=total_ambiguous,
            match_results=all_results,
            pair_stats=pair_stats,
            mean_confidence=mean_conf,
            min_confidence=min_conf,
            processing_time_ms=round(elapsed_ms, 3),
            provenance_hash=batch_prov,
            created_at=_utcnow().isoformat(),
        )

        logger.info(
            "match_batch complete: %d sources, %d pairs, %d matched, "
            "mean_conf=%.4f, elapsed=%.1fms",
            len(source_ids),
            len(pair_stats),
            total_matched,
            mean_conf,
            elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # 7. Compute Match Key
    # ------------------------------------------------------------------

    def compute_match_key(
        self,
        record: Dict[str, Any],
        key_fields: List[str],
    ) -> MatchKey:
        """Extract and normalise key fields from a record.

        Builds a composite key string by normalising each field value
        (lowercased, stripped) and joining with ``||`` as separator.

        Args:
            record: Source record dictionary.
            key_fields: Ordered list of fields to include in the key.

        Returns:
            MatchKey with normalised fields, composite key string,
            and provenance hash.
        """
        fields: Dict[str, Any] = {}
        parts: List[str] = []

        for f in key_fields:
            raw_val = record.get(f, "")
            normalised = self._normalise_key_value(raw_val)
            fields[f] = normalised
            parts.append(normalised)

        composite = "||".join(parts)

        prov = self._provenance.build_hash({
            "engine": "matching",
            "operation": "compute_match_key",
            "fields": fields,
            "composite_key": composite,
        })

        return MatchKey(
            fields=fields,
            composite_key=composite,
            provenance_hash=prov,
        )

    # ==================================================================
    # Similarity algorithms (pure Python, zero-hallucination)
    # ==================================================================

    # ------------------------------------------------------------------
    # Jaro-Winkler Similarity
    # ------------------------------------------------------------------

    @staticmethod
    def _jaro_winkler_similarity(s1: str, s2: str) -> float:
        """Compute Jaro-Winkler similarity between two strings.

        The Jaro similarity is computed first, measuring transpositions
        and matching characters within a window. The Winkler extension
        adds a bonus for common prefixes (up to 4 characters) with a
        scaling factor of 0.1.

        Args:
            s1: First string.
            s2: Second string.

        Returns:
            Similarity score in [0.0, 1.0]. Returns 1.0 for identical
            strings; 0.0 when no characters match.
        """
        if s1 == s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        len1 = len(s1)
        len2 = len(s2)

        # Maximum matching window
        match_distance = max(len1, len2) // 2 - 1
        if match_distance < 0:
            match_distance = 0

        s1_matches = [False] * len1
        s2_matches = [False] * len2

        matches = 0
        transpositions = 0

        # Find matching characters
        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)

            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

        if matches == 0:
            return 0.0

        # Count transpositions
        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

        jaro = (
            matches / len1
            + matches / len2
            + (matches - transpositions / 2.0) / matches
        ) / 3.0

        # Winkler prefix bonus
        prefix_len = 0
        for i in range(min(len1, len2, _JW_MAX_PREFIX)):
            if s1[i] == s2[i]:
                prefix_len += 1
            else:
                break

        return jaro + prefix_len * _JW_PREFIX_SCALE * (1.0 - jaro)

    # ------------------------------------------------------------------
    # Levenshtein Distance
    # ------------------------------------------------------------------

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance between two strings.

        Uses dynamic programming with O(min(m,n)) space optimisation
        (two-row approach).

        Args:
            s1: First string.
            s2: Second string.

        Returns:
            Minimum number of single-character edits (insertions,
            deletions, substitutions) to transform s1 into s2.
        """
        if s1 == s2:
            return 0
        if not s1:
            return len(s2)
        if not s2:
            return len(s1)

        # Ensure s1 is the shorter string for space efficiency
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        len1 = len(s1)
        len2 = len(s2)

        # Previous and current row
        prev_row = list(range(len1 + 1))
        curr_row = [0] * (len1 + 1)

        for j in range(1, len2 + 1):
            curr_row[0] = j
            for i in range(1, len1 + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                curr_row[i] = min(
                    curr_row[i - 1] + 1,      # insertion
                    prev_row[i] + 1,           # deletion
                    prev_row[i - 1] + cost,    # substitution
                )
            prev_row, curr_row = curr_row, prev_row

        return prev_row[len1]

    # ------------------------------------------------------------------
    # Numeric Proximity
    # ------------------------------------------------------------------

    @staticmethod
    def _numeric_proximity(
        a: float,
        b: float,
        max_diff: float = _DEFAULT_MAX_DIFF,
    ) -> float:
        """Compute numeric proximity between two values.

        Returns 1.0 when a == b, decaying linearly to 0.0 as the
        absolute difference reaches ``max_diff``.

        Args:
            a: First numeric value.
            b: Second numeric value.
            max_diff: Maximum difference for scaling. When the absolute
                difference reaches or exceeds this, proximity is 0.0.

        Returns:
            Proximity score in [0.0, 1.0].
        """
        if max_diff <= 0:
            return 1.0 if abs(a - b) < _EPS else 0.0

        return 1.0 - min(abs(a - b) / max_diff, 1.0)

    # ------------------------------------------------------------------
    # Date Proximity
    # ------------------------------------------------------------------

    @staticmethod
    def _date_proximity(
        a: datetime,
        b: datetime,
        max_days: int = _DEFAULT_MAX_DAYS,
    ) -> float:
        """Compute date proximity between two datetimes.

        Returns 1.0 when dates are identical, decaying linearly to 0.0
        as the day difference reaches ``max_days``.

        Args:
            a: First datetime.
            b: Second datetime.
            max_days: Maximum day difference for scaling.

        Returns:
            Proximity score in [0.0, 1.0].
        """
        if max_days <= 0:
            return 1.0 if a == b else 0.0

        day_diff = abs((a - b).days)
        return 1.0 - min(day_diff / max_days, 1.0)

    # ==================================================================
    # Private helpers -- field similarity computation
    # ==================================================================

    def _compute_field_similarities(
        self,
        rec_a: Dict[str, Any],
        rec_b: Dict[str, Any],
        key_fields: List[str],
    ) -> Dict[str, float]:
        """Compute per-field similarity between two records.

        Automatically detects field types and applies the appropriate
        similarity function:
            - String values: Jaro-Winkler similarity
            - Numeric values: Numeric proximity
            - Datetime values: Date proximity
            - None / missing: 0.0 if one side is missing

        Args:
            rec_a: Record from source A.
            rec_b: Record from source B.
            key_fields: Fields to compare.

        Returns:
            Mapping of field name to similarity score [0.0, 1.0].
        """
        scores: Dict[str, float] = {}

        for f in key_fields:
            val_a = rec_a.get(f)
            val_b = rec_b.get(f)

            # Handle missing values
            if val_a is None and val_b is None:
                scores[f] = 1.0
                continue
            if val_a is None or val_b is None:
                scores[f] = 0.0
                continue

            # Detect type and compute similarity
            score = self._compute_single_field_similarity(val_a, val_b)
            scores[f] = round(score, 6)

        return scores

    def _compute_single_field_similarity(
        self,
        val_a: Any,
        val_b: Any,
    ) -> float:
        """Compute similarity between two values with auto type detection.

        Args:
            val_a: Value from record A.
            val_b: Value from record B.

        Returns:
            Similarity score in [0.0, 1.0].
        """
        # Both are datetime
        if isinstance(val_a, datetime) and isinstance(val_b, datetime):
            return self._date_proximity(val_a, val_b)

        # Both are numeric
        if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            float_a = float(val_a)
            float_b = float(val_b)
            max_val = max(abs(float_a), abs(float_b), 1.0)
            return self._numeric_proximity(float_a, float_b, max_val)

        # Convert to strings and use Jaro-Winkler
        str_a = str(val_a).strip().lower()
        str_b = str(val_b).strip().lower()

        # Exact string match shortcut
        if str_a == str_b:
            return 1.0

        return self._jaro_winkler_similarity(str_a, str_b)

    # ==================================================================
    # Private helpers -- temporal aggregation
    # ==================================================================

    def _aggregate_temporal(
        self,
        records: List[Dict[str, Any]],
        entity_field: str,
        value_field: str,
        timestamp_field: str,
        source_granularity: str,
        target_granularity: str,
    ) -> List[Dict[str, Any]]:
        """Aggregate records to a target temporal granularity.

        Groups records by entity and target period, then sums values
        within each group. Produces records with an ``aligned_period``
        field representing the target period.

        Supported conversions:
            - monthly -> quarterly: group Jan-Mar, Apr-Jun, etc.
            - monthly -> annual: group Jan-Dec
            - daily -> monthly: group by YYYY-MM
            - quarterly -> annual: group Q1-Q4

        If source and target granularity are the same, records pass
        through with only the ``aligned_period`` field added.

        Args:
            records: Source records.
            entity_field: Field name for entity identifier.
            value_field: Field name for the numeric value.
            timestamp_field: Field for timestamp/period string.
            source_granularity: Granularity of the input records.
            target_granularity: Desired output granularity.

        Returns:
            List of aggregated records with ``aligned_period`` field.
        """
        source_order = _GRANULARITY_ORDER.get(source_granularity, 2)
        target_order = _GRANULARITY_ORDER.get(target_granularity, 2)

        aggregated: Dict[str, Dict[str, Any]] = {}

        for rec in records:
            entity = str(rec.get(entity_field, ""))
            ts_raw = rec.get(timestamp_field, "")
            value = self._to_float(rec.get(value_field, 0))

            aligned_period = self._align_period(
                ts_raw, source_granularity, target_granularity,
            )

            group_key = f"{entity}||{aligned_period}"

            if group_key not in aggregated:
                aggregated[group_key] = {
                    entity_field: entity,
                    "aligned_period": aligned_period,
                    value_field: 0.0,
                    "_count": 0,
                    "_source_records": [],
                }

            aggregated[group_key][value_field] += value
            aggregated[group_key]["_count"] += 1
            aggregated[group_key]["_source_records"].append(rec)

        # Convert to list and clean up internal fields
        result: List[Dict[str, Any]] = []
        for group in aggregated.values():
            output_rec = {
                entity_field: group[entity_field],
                "aligned_period": group["aligned_period"],
                value_field: round(group[value_field], 6),
                "record_count": group["_count"],
            }
            result.append(output_rec)

        logger.debug(
            "_aggregate_temporal: %d records -> %d groups "
            "(%s -> %s)",
            len(records), len(result),
            source_granularity, target_granularity,
        )

        return result

    def _align_period(
        self,
        timestamp_raw: Any,
        source_granularity: str,
        target_granularity: str,
    ) -> str:
        """Align a timestamp/period string to the target granularity.

        Parses various period formats and converts to the target
        granularity representation.

        Supported input formats:
            - ``YYYY-MM-DD`` (daily)
            - ``YYYY-MM`` (monthly)
            - ``YYYY-QN`` (quarterly, e.g. ``2025-Q1``)
            - ``YYYY`` (annual)
            - datetime objects

        Args:
            timestamp_raw: Raw timestamp or period string.
            source_granularity: Source granularity.
            target_granularity: Target granularity.

        Returns:
            Aligned period string (e.g. ``2025-Q1``, ``2025``,
            ``2025-01``).
        """
        year, month = self._parse_period(timestamp_raw)

        if target_granularity == "annual":
            return str(year)
        elif target_granularity == "quarterly":
            quarter = (month - 1) // 3 + 1
            return f"{year}-Q{quarter}"
        elif target_granularity == "monthly":
            return f"{year}-{month:02d}"
        elif target_granularity == "daily":
            # Cannot disaggregate; return monthly
            return f"{year}-{month:02d}"
        else:
            return f"{year}-{month:02d}"

    @staticmethod
    def _parse_period(timestamp_raw: Any) -> Tuple[int, int]:
        """Parse a period/timestamp into (year, month).

        Handles multiple common formats:
            - ``datetime`` objects
            - ``YYYY-MM-DD`` strings
            - ``YYYY-MM`` strings
            - ``YYYY-QN`` strings
            - ``YYYY`` strings
            - Integer year values

        Args:
            timestamp_raw: Raw timestamp or period representation.

        Returns:
            Tuple of (year, month). Month defaults to 1 when only
            year is available.
        """
        if isinstance(timestamp_raw, datetime):
            return timestamp_raw.year, timestamp_raw.month

        ts_str = str(timestamp_raw).strip()

        # Try YYYY-QN format
        if "-Q" in ts_str.upper():
            parts = ts_str.upper().split("-Q")
            try:
                year = int(parts[0])
                quarter = int(parts[1])
                month = (quarter - 1) * 3 + 1
                return year, month
            except (ValueError, IndexError):
                pass

        # Try YYYY-MM-DD or YYYY-MM
        if "-" in ts_str:
            parts = ts_str.split("-")
            try:
                year = int(parts[0])
                month = int(parts[1]) if len(parts) >= 2 else 1
                return year, max(1, min(12, month))
            except (ValueError, IndexError):
                pass

        # Try bare year
        try:
            year = int(ts_str)
            return year, 1
        except ValueError:
            pass

        # Fallback
        return 2000, 1

    # ==================================================================
    # Private helpers -- blocking
    # ==================================================================

    def _create_blocks(
        self,
        records: List[Dict[str, Any]],
        blocking_fields: List[str],
        prefix_len: int = _DEFAULT_BLOCK_PREFIX_LEN,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Create hash-based blocks from records using field prefixes.

        For each record, concatenates the first ``prefix_len`` characters
        of each blocking field to form a block key. Records with the same
        block key are grouped together.

        Args:
            records: Records to block.
            blocking_fields: Fields used for block key generation.
            prefix_len: Number of leading characters per field.

        Returns:
            Mapping of block key to list of records in that block.
        """
        blocks: Dict[str, List[Dict[str, Any]]] = {}

        for rec in records:
            key_parts: List[str] = []
            for f in blocking_fields:
                raw = str(rec.get(f, "")).strip().lower()
                prefix = raw[:prefix_len] if len(raw) >= prefix_len else raw
                key_parts.append(prefix)

            block_key = "||".join(key_parts)
            blocks.setdefault(block_key, []).append(rec)

        logger.debug(
            "_create_blocks: %d records -> %d blocks "
            "(fields=%s, prefix_len=%d)",
            len(records), len(blocks), blocking_fields, prefix_len,
        )
        return blocks

    # ==================================================================
    # Private helpers -- transitive linking
    # ==================================================================

    def _apply_transitive_links(
        self,
        results: List[MatchResult],
    ) -> List[MatchResult]:
        """Apply transitive linking to batch match results.

        Uses union-find to identify transitive clusters: if record A
        matches record B and record B matches record C, all three are
        linked. This is recorded in the metadata of each MatchResult.

        Args:
            results: All match results from pairwise matching.

        Returns:
            Updated results with ``transitive_group`` in metadata.
        """
        # Build union-find for matched record keys
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            """Find root with path compression."""
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            """Union two elements."""
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        # Build edges from matched results
        for r in results:
            if r.status != MatchStatus.MATCHED.value:
                continue

            key_a = self._record_identity_key(r.record_a, r.source_a_id)
            key_b = self._record_identity_key(
                r.record_b or {}, r.source_b_id,
            )

            if key_a and key_b:
                parent.setdefault(key_a, key_a)
                parent.setdefault(key_b, key_b)
                union(key_a, key_b)

        # Assign transitive group IDs
        group_map: Dict[str, str] = {}
        group_counter = 0

        for r in results:
            if r.status != MatchStatus.MATCHED.value:
                continue

            key_a = self._record_identity_key(r.record_a, r.source_a_id)
            if key_a:
                root = find(key_a)
                if root not in group_map:
                    group_counter += 1
                    group_map[root] = f"TG-{group_counter:06d}"
                r.metadata["transitive_group"] = group_map[root]

        return results

    @staticmethod
    def _record_identity_key(
        record: Dict[str, Any],
        source_id: str,
    ) -> str:
        """Build a stable identity key for a record within a source.

        Uses the record's JSON representation combined with the source
        ID to produce a unique key.

        Args:
            record: Record dictionary.
            source_id: Source identifier.

        Returns:
            String key, or empty string if record is empty.
        """
        if not record:
            return ""
        content = json.dumps(record, sort_keys=True, default=str)
        return f"{source_id}::{content}"

    # ==================================================================
    # Private helpers -- normalisation
    # ==================================================================

    @staticmethod
    def _normalise_key_value(value: Any) -> str:
        """Normalise a key field value for consistent comparison.

        Strips whitespace, lowercases strings, and converts other types
        to string representation.

        Args:
            value: Raw field value.

        Returns:
            Normalised string representation.
        """
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip().lower()
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value).strip().lower()

    @staticmethod
    def _to_float(value: Any) -> float:
        """Safely convert a value to float.

        Args:
            value: Value to convert.

        Returns:
            Float representation, or 0.0 on conversion failure.
        """
        if value is None:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    # ==================================================================
    # Private helpers -- provenance
    # ==================================================================

    def _compute_provenance(
        self,
        operation: str,
        input_data: Any,
        output_data: Any,
    ) -> str:
        """Compute a SHA-256 provenance hash for a match operation.

        Args:
            operation: Name of the operation (e.g. ``match_exact``).
            input_data: Input data for the operation.
            output_data: Output data for the operation.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        return self._provenance.build_hash({
            "engine": "matching",
            "operation": operation,
            "input": input_data,
            "output": output_data,
        })

    def _record_operation_provenance(
        self,
        strategy: str,
        source_a_id: str,
        source_b_id: str,
        count_a: int,
        count_b: int,
        matched: int,
        results: List[MatchResult],
    ) -> str:
        """Record provenance for a complete match_records operation.

        Adds an entry to the provenance chain linking the input
        parameters to the output statistics.

        Args:
            strategy: Matching strategy used.
            source_a_id: Source A identifier.
            source_b_id: Source B identifier.
            count_a: Number of records from source A.
            count_b: Number of records from source B.
            matched: Number of matched pairs.
            results: All match results.

        Returns:
            Chain hash from the provenance tracker.
        """
        confidences = [
            r.confidence for r in results
            if r.status == MatchStatus.MATCHED.value
        ]
        mean_conf = (
            round(sum(confidences) / len(confidences), 6)
            if confidences else 0.0
        )

        input_hash = self._provenance.build_hash({
            "operation": "match_records_input",
            "strategy": strategy,
            "source_a_id": source_a_id,
            "source_b_id": source_b_id,
            "count_a": count_a,
            "count_b": count_b,
        })
        output_hash = self._provenance.build_hash({
            "operation": "match_records_output",
            "matched": matched,
            "total_results": len(results),
            "mean_confidence": mean_conf,
        })

        return self._provenance.add_to_chain(
            operation=f"match_records_{strategy}",
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={
                "strategy": strategy,
                "source_a_id": source_a_id,
                "source_b_id": source_b_id,
                "matched": matched,
                "mean_confidence": mean_conf,
            },
        )


# ============================================================================
# Module exports
# ============================================================================

__all__ = ["MatchingEngine"]
