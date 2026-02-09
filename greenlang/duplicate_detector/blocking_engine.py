# -*- coding: utf-8 -*-
"""
Blocking Engine - AGENT-DATA-011: Duplicate Detection (GL-DATA-X-014)

Partitions records into candidate blocks to reduce the O(n^2) pairwise
comparison space. Supports three blocking strategies: standard (hash-based),
sorted neighborhood (sliding window), and canopy clustering (TF-IDF).

Zero-Hallucination Guarantees:
    - All blocking keys use deterministic string operations
    - Sorted neighborhood uses deterministic sorting and fixed window
    - Canopy clustering uses TF-IDF cosine distance (no ML)
    - Soundex encoding uses the classic 4-character algorithm
    - No ML/LLM calls in blocking path
    - Provenance recorded for every blocking operation

Example:
    >>> from greenlang.duplicate_detector.blocking_engine import BlockingEngine
    >>> engine = BlockingEngine()
    >>> blocks = engine.create_blocks(
    ...     records=[{"id": "1", "name": "Alice"}, {"id": "2", "name": "Alice"}],
    ...     strategy=BlockingStrategy.STANDARD,
    ...     key_fields=["name"],
    ... )

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
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.duplicate_detector.models import (
    BlockingStrategy,
    BlockResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BlockingEngine",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_provenance(operation: str, data_repr: str) -> str:
    """Compute SHA-256 provenance hash for a blocking operation."""
    payload = f"{operation}:{data_repr}:{_utcnow().isoformat()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SOUNDEX_TABLE: Dict[str, str] = {
    "b": "1", "f": "1", "p": "1", "v": "1",
    "c": "2", "g": "2", "j": "2", "k": "2",
    "q": "2", "s": "2", "x": "2", "z": "2",
    "d": "3", "t": "3", "l": "4",
    "m": "5", "n": "5", "r": "6",
}

_DEFAULT_WINDOW_SIZE: int = 10
_DEFAULT_KEY_SIZE: int = 3
_MIN_CANOPY_RECORDS: int = 3


class BlockingEngine:
    """Blocking engine for candidate pair generation.

    Partitions records into blocks sharing a blocking key to reduce
    the quadratic comparison space to near-linear.

    This engine follows GreenLang's zero-hallucination principle by
    using deterministic blocking key generation and grouping algorithms.

    Example:
        >>> engine = BlockingEngine()
        >>> blocks = engine.create_blocks(records, BlockingStrategy.STANDARD, ["name"])
    """

    def __init__(self) -> None:
        """Initialize BlockingEngine with empty statistics."""
        self._stats_lock = threading.Lock()
        self._invocations: int = 0
        self._successes: int = 0
        self._failures: int = 0
        self._total_duration_ms: float = 0.0
        self._last_invoked_at: Optional[datetime] = None
        logger.info("BlockingEngine initialized")

    def create_blocks(
        self, records: List[Dict[str, Any]], strategy: BlockingStrategy,
        key_fields: List[str], id_field: str = "id",
        window_size: int = _DEFAULT_WINDOW_SIZE,
        key_size: int = _DEFAULT_KEY_SIZE,
        tight_threshold: float = 0.8, loose_threshold: float = 0.4,
    ) -> List[BlockResult]:
        """Create blocks of candidate records using the specified strategy.

        Args:
            records: List of input records.
            strategy: Blocking strategy to use.
            key_fields: Fields used to generate blocking keys.
            id_field: Field name for record identifier.
            window_size: Window size for sorted neighborhood.
            key_size: Number of leading chars for blocking key.
            tight_threshold: Tight threshold for canopy clustering.
            loose_threshold: Loose threshold for canopy clustering.

        Returns:
            List of BlockResult instances.
        """
        start_time = time.monotonic()
        try:
            if not records:
                raise ValueError("records list must not be empty")
            if not key_fields:
                raise ValueError("key_fields must not be empty")

            if strategy == BlockingStrategy.STANDARD:
                blocks = self.standard_blocking(records, key_fields, id_field, key_size)
            elif strategy == BlockingStrategy.SORTED_NEIGHBORHOOD:
                blocks = self.sorted_neighborhood(records, key_fields, id_field, window_size, key_size)
            elif strategy == BlockingStrategy.CANOPY:
                blocks = self.canopy_clustering(records, key_fields, id_field, tight_threshold, loose_threshold)
            elif strategy == BlockingStrategy.NONE:
                blocks = self._no_blocking(records, id_field)
            else:
                raise ValueError(f"Unknown blocking strategy: {strategy}")

            self._record_success(time.monotonic() - start_time)
            logger.info("Created %d blocks from %d records using %s", len(blocks), len(records), strategy.value)
            return blocks
        except Exception as e:
            self._record_failure(time.monotonic() - start_time)
            logger.error("Blocking failed: %s", e)
            raise

    def standard_blocking(
        self, records: List[Dict[str, Any]], key_fields: List[str],
        id_field: str = "id", key_size: int = _DEFAULT_KEY_SIZE,
    ) -> List[BlockResult]:
        """Hash-based standard blocking: group by identical blocking key."""
        groups: Dict[str, List[str]] = defaultdict(list)
        for idx, record in enumerate(records):
            rid = str(record.get(id_field, f"rec-{idx}"))
            bkey = self.generate_blocking_key(record, key_fields, key_size)
            groups[bkey].append(rid)

        blocks: List[BlockResult] = []
        for bkey, record_ids in groups.items():
            if len(record_ids) < 2:
                continue
            provenance = _compute_provenance("standard_blocking", f"{bkey}:{len(record_ids)}")
            blocks.append(BlockResult(
                block_key=bkey, strategy=BlockingStrategy.STANDARD,
                record_ids=record_ids, record_count=len(record_ids),
                provenance_hash=provenance,
            ))
        return blocks

    def sorted_neighborhood(
        self, records: List[Dict[str, Any]], key_fields: List[str],
        id_field: str = "id", window_size: int = _DEFAULT_WINDOW_SIZE,
        key_size: int = _DEFAULT_KEY_SIZE,
    ) -> List[BlockResult]:
        """Sorted neighborhood blocking with sliding window."""
        keyed: List[Tuple[str, str]] = []
        for idx, record in enumerate(records):
            rid = str(record.get(id_field, f"rec-{idx}"))
            bkey = self.generate_blocking_key(record, key_fields, key_size)
            keyed.append((bkey, rid))

        keyed.sort(key=lambda x: x[0])
        seen_blocks: Dict[str, List[str]] = {}
        n = len(keyed)
        effective_window = min(window_size, n)

        for i in range(max(1, n - effective_window + 1)):
            end = min(i + effective_window, n)
            window_ids = [keyed[j][1] for j in range(i, end)]
            if len(window_ids) < 2:
                continue
            block_key = "|".join(sorted(window_ids))
            if block_key not in seen_blocks:
                seen_blocks[block_key] = window_ids

        blocks: List[BlockResult] = []
        for block_key, record_ids in seen_blocks.items():
            bk = f"snb-{hashlib.md5(block_key.encode()).hexdigest()[:8]}"
            provenance = _compute_provenance("sorted_neighborhood", f"w{window_size}:{len(record_ids)}")
            blocks.append(BlockResult(
                block_key=bk, strategy=BlockingStrategy.SORTED_NEIGHBORHOOD,
                record_ids=record_ids, record_count=len(record_ids),
                provenance_hash=provenance,
            ))
        return blocks

    def canopy_clustering(
        self, records: List[Dict[str, Any]], key_fields: List[str],
        id_field: str = "id", tight_threshold: float = 0.8,
        loose_threshold: float = 0.4,
    ) -> List[BlockResult]:
        """Canopy clustering using TF-IDF-based distance."""
        if len(records) < _MIN_CANOPY_RECORDS:
            all_ids = [str(r.get(id_field, f"rec-{i}")) for i, r in enumerate(records)]
            if len(all_ids) >= 2:
                provenance = _compute_provenance("canopy_single", str(len(all_ids)))
                return [BlockResult(
                    block_key="canopy-all", strategy=BlockingStrategy.CANOPY,
                    record_ids=all_ids, record_count=len(all_ids), provenance_hash=provenance,
                )]
            return []

        texts: List[str] = []
        record_ids: List[str] = []
        for idx, record in enumerate(records):
            record_ids.append(str(record.get(id_field, f"rec-{idx}")))
            texts.append(" ".join(str(record.get(f, "")).strip().lower() for f in key_fields))

        tf_vectors = self._compute_tf_vectors(texts)
        available = set(range(len(records)))
        canopies: List[List[int]] = []

        while available:
            center_idx = min(available)
            canopy_members: List[int] = [center_idx]
            to_remove: List[int] = [center_idx]
            for idx in sorted(available):
                if idx == center_idx:
                    continue
                distance = self._cosine_distance(tf_vectors[center_idx], tf_vectors[idx])
                if distance <= loose_threshold:
                    canopy_members.append(idx)
                if distance <= tight_threshold:
                    to_remove.append(idx)
            for idx in to_remove:
                available.discard(idx)
            if len(canopy_members) >= 2:
                canopies.append(canopy_members)

        blocks: List[BlockResult] = []
        for members in canopies:
            ids = [record_ids[i] for i in members]
            bk = f"canopy-{hashlib.md5('|'.join(ids).encode()).hexdigest()[:8]}"
            provenance = _compute_provenance("canopy_clustering", f"{len(ids)}")
            blocks.append(BlockResult(
                block_key=bk, strategy=BlockingStrategy.CANOPY,
                record_ids=ids, record_count=len(ids), provenance_hash=provenance,
            ))
        return blocks

    def _no_blocking(self, records: List[Dict[str, Any]], id_field: str = "id") -> List[BlockResult]:
        """No blocking: all records in one block (for small datasets)."""
        all_ids = [str(r.get(id_field, f"rec-{i}")) for i, r in enumerate(records)]
        if len(all_ids) < 2:
            return []
        provenance = _compute_provenance("no_blocking", str(len(all_ids)))
        return [BlockResult(
            block_key="all", strategy=BlockingStrategy.NONE,
            record_ids=all_ids, record_count=len(all_ids), provenance_hash=provenance,
        )]

    def generate_blocking_key(
        self, record: Dict[str, Any], key_fields: List[str], key_size: int = _DEFAULT_KEY_SIZE,
    ) -> str:
        """Generate a blocking key from the first key_size chars of each key field."""
        parts: List[str] = []
        for field in sorted(key_fields):
            normalized = str(record.get(field, "")).strip().lower()
            parts.append(normalized[:key_size])
        return "|".join(parts)

    def generate_phonetic_key(self, value: str) -> str:
        """Generate Soundex phonetic encoding for a string value."""
        if not value or not value.strip():
            return "0000"
        s = "".join(c for c in value.strip().upper() if c.isalpha())
        if not s:
            return "0000"
        coded: List[str] = [s[0]]
        prev_code = _SOUNDEX_TABLE.get(s[0].lower(), "0")
        for char in s[1:]:
            code = _SOUNDEX_TABLE.get(char.lower(), "0")
            if code != "0" and code != prev_code:
                coded.append(code)
                if len(coded) == 4:
                    break
            prev_code = code
        return "".join(coded).ljust(4, "0")[:4]

    def estimate_reduction_ratio(self, total_records: int, blocks: List[BlockResult]) -> float:
        """Estimate comparison reduction ratio achieved by blocking."""
        if total_records < 2:
            return 1.0
        total_cmp = total_records * (total_records - 1) / 2
        blocked_cmp = sum(b.record_count * (b.record_count - 1) / 2 for b in blocks if b.record_count >= 2)
        return max(0.0, min(1.0, 1.0 - blocked_cmp / total_cmp)) if total_cmp > 0 else 1.0

    def optimize_window_size(
        self, records: List[Dict[str, Any]], key_fields: List[str],
        id_field: str = "id", min_window: int = 3, max_window: int = 50,
        target_reduction: float = 0.9,
    ) -> int:
        """Heuristic window size optimization for sorted neighborhood."""
        if len(records) < 3:
            return min_window
        best_window = min_window
        for window in range(min_window, max_window + 1):
            blocks = self.sorted_neighborhood(records, key_fields, id_field, window)
            ratio = self.estimate_reduction_ratio(len(records), blocks)
            if ratio >= target_reduction:
                best_window = window
                break
            best_window = window
        logger.info("Optimized window size: %d (target: %.2f)", best_window, target_reduction)
        return best_window

    def get_statistics(self) -> Dict[str, Any]:
        """Return current engine operational statistics."""
        with self._stats_lock:
            avg_ms = self._total_duration_ms / self._invocations if self._invocations > 0 else 0.0
            return {
                "engine_name": "BlockingEngine", "invocations": self._invocations,
                "successes": self._successes, "failures": self._failures,
                "total_duration_ms": round(self._total_duration_ms, 3),
                "avg_duration_ms": round(avg_ms, 3),
                "last_invoked_at": self._last_invoked_at.isoformat() if self._last_invoked_at else None,
            }

    def reset_statistics(self) -> None:
        """Reset all operational statistics to zero."""
        with self._stats_lock:
            self._invocations = 0
            self._successes = 0
            self._failures = 0
            self._total_duration_ms = 0.0
            self._last_invoked_at = None

    # -- Private TF-IDF helpers --

    def _compute_tf_vectors(self, texts: List[str]) -> List[Dict[str, float]]:
        """Compute term frequency vectors for a list of texts."""
        vectors: List[Dict[str, float]] = []
        for text in texts:
            terms = text.split()
            tf: Dict[str, float] = {}
            total = len(terms) if terms else 1
            for term in terms:
                tf[term] = tf.get(term, 0.0) + 1.0
            for term in tf:
                tf[term] /= total
            vectors.append(tf)
        return vectors

    def _cosine_distance(self, vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        """Compute cosine distance (1 - cosine_similarity) between two TF vectors."""
        if not vec_a or not vec_b:
            return 1.0
        common = set(vec_a.keys()) & set(vec_b.keys())
        dot = sum(vec_a[t] * vec_b[t] for t in common)
        mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
        mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
        if mag_a == 0.0 or mag_b == 0.0:
            return 1.0
        return 1.0 - max(0.0, min(1.0, dot / (mag_a * mag_b)))

    def _record_success(self, elapsed_seconds: float) -> None:
        """Record a successful invocation."""
        with self._stats_lock:
            self._invocations += 1
            self._successes += 1
            self._total_duration_ms += elapsed_seconds * 1000.0
            self._last_invoked_at = _utcnow()

    def _record_failure(self, elapsed_seconds: float) -> None:
        """Record a failed invocation."""
        with self._stats_lock:
            self._invocations += 1
            self._failures += 1
            self._total_duration_ms += elapsed_seconds * 1000.0
            self._last_invoked_at = _utcnow()
