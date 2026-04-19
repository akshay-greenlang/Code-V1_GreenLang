"""
Data deduplication utilities for GreenLang.

Provides hash-based deduplication, fuzzy matching, and duplicate detection
for ensuring data integrity across connectors and pipelines.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
from collections import defaultdict
import difflib
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DeduplicationStrategy(Enum):
    """Deduplication strategies."""
    EXACT_MATCH = "exact_match"          # Exact hash match
    FUZZY_MATCH = "fuzzy_match"          # Similar content detection
    KEY_BASED = "key_based"              # Match on specific keys
    TIME_WINDOW = "time_window"          # Within time window
    COMPOSITE = "composite"               # Multiple strategies


class DuplicateAction(Enum):
    """Actions to take on duplicate detection."""
    SKIP = "skip"                        # Skip the duplicate
    UPDATE = "update"                    # Update existing record
    MERGE = "merge"                      # Merge with existing
    VERSION = "version"                  # Create new version
    ALERT = "alert"                      # Alert but process


@dataclass
class DuplicateRecord:
    """Information about a duplicate record."""
    original_hash: str
    duplicate_hash: str
    original_timestamp: datetime
    duplicate_timestamp: datetime
    similarity_score: float
    strategy_used: DeduplicationStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeduplicationResult:
    """Result of deduplication operation."""
    is_duplicate: bool
    original_record: Optional[Any] = None
    similarity_score: float = 0.0
    strategy_matched: Optional[DeduplicationStrategy] = None
    suggested_action: DuplicateAction = DuplicateAction.SKIP


class HashGenerator:
    """
    Generate deterministic hashes for data deduplication.

    Supports multiple hash algorithms and custom key extraction.
    """

    @staticmethod
    def generate_hash(
        data: Any,
        algorithm: str = "sha256",
        keys: Optional[List[str]] = None
    ) -> str:
        """
        Generate hash for data.

        Args:
            data: Data to hash
            algorithm: Hash algorithm (sha256, md5, sha1)
            keys: Specific keys to include in hash (for dicts)

        Returns:
            Hex digest of hash
        """
        # Select hash algorithm
        if algorithm == "sha256":
            hasher = hashlib.sha256()
        elif algorithm == "md5":
            hasher = hashlib.md5()
        elif algorithm == "sha1":
            hasher = hashlib.sha1()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Prepare data for hashing
        if isinstance(data, dict):
            if keys:
                # Hash only specified keys
                hash_data = {k: data.get(k) for k in keys if k in data}
            else:
                # Hash all keys in sorted order
                hash_data = dict(sorted(data.items()))
            content = json.dumps(hash_data, sort_keys=True, default=str)
        elif isinstance(data, (list, tuple)):
            content = json.dumps(data, sort_keys=True, default=str)
        elif isinstance(data, str):
            content = data
        else:
            # Convert to string for other types
            content = str(data)

        hasher.update(content.encode('utf-8'))
        return hasher.hexdigest()

    @staticmethod
    def generate_composite_hash(
        data: Any,
        hash_fields: Dict[str, List[str]]
    ) -> str:
        """
        Generate composite hash from multiple field groups.

        Args:
            data: Data to hash
            hash_fields: Dictionary of field groups to hash

        Returns:
            Composite hash string

        Example:
            composite_hash = HashGenerator.generate_composite_hash(
                data=shipment,
                hash_fields={
                    "primary": ["shipment_id", "origin", "destination"],
                    "secondary": ["weight", "volume"]
                }
            )
        """
        hashes = []
        for group_name, fields in hash_fields.items():
            group_hash = HashGenerator.generate_hash(data, keys=fields)
            hashes.append(f"{group_name}:{group_hash[:8]}")

        return "_".join(hashes)


class DeduplicationCache:
    """
    In-memory cache for deduplication with TTL support.

    Maintains a fixed-size cache of recent hashes for fast duplicate detection.
    """

    def __init__(
        self,
        max_size: int = 100000,
        ttl: Optional[timedelta] = None
    ):
        """
        Initialize deduplication cache.

        Args:
            max_size: Maximum number of hashes to cache
            ttl: Time to live for cache entries
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._access_count: Dict[str, int] = defaultdict(int)

    def add(self, hash_value: str, data: Any) -> bool:
        """
        Add hash to cache.

        Args:
            hash_value: Hash to add
            data: Associated data

        Returns:
            True if added, False if duplicate
        """
        # Check if already exists
        if hash_value in self._cache:
            self._access_count[hash_value] += 1
            return False

        # Check cache size
        if len(self._cache) >= self.max_size:
            self._evict_lru()

        # Add to cache
        self._cache[hash_value] = (data, datetime.now())
        self._access_count[hash_value] = 1
        return True

    def contains(self, hash_value: str) -> bool:
        """
        Check if hash exists in cache.

        Args:
            hash_value: Hash to check

        Returns:
            True if exists and not expired
        """
        if hash_value not in self._cache:
            return False

        _, timestamp = self._cache[hash_value]

        # Check TTL
        if self.ttl and datetime.now() - timestamp > self.ttl:
            del self._cache[hash_value]
            del self._access_count[hash_value]
            return False

        self._access_count[hash_value] += 1
        return True

    def get(self, hash_value: str) -> Optional[Any]:
        """
        Get data associated with hash.

        Args:
            hash_value: Hash to retrieve

        Returns:
            Associated data if exists
        """
        if not self.contains(hash_value):
            return None

        data, _ = self._cache[hash_value]
        return data

    def _evict_lru(self):
        """Evict least recently used entry."""
        # Find least accessed entry
        lru_hash = min(self._access_count.keys(), key=lambda k: self._access_count[k])
        del self._cache[lru_hash]
        del self._access_count[lru_hash]

    def clear_expired(self):
        """Remove expired entries."""
        if not self.ttl:
            return

        current_time = datetime.now()
        expired = [
            hash_value for hash_value, (_, timestamp) in self._cache.items()
            if current_time - timestamp > self.ttl
        ]

        for hash_value in expired:
            del self._cache[hash_value]
            del self._access_count[hash_value]

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hit_count": sum(self._access_count.values()),
            "unique_hashes": len(self._cache),
            "ttl": str(self.ttl) if self.ttl else "No TTL"
        }


class FuzzyMatcher:
    """
    Fuzzy matching for near-duplicate detection.

    Uses similarity algorithms to detect records that are similar but not identical.
    """

    @staticmethod
    def calculate_similarity(
        data1: Any,
        data2: Any,
        threshold: float = 0.85
    ) -> Tuple[float, bool]:
        """
        Calculate similarity between two data items.

        Args:
            data1: First data item
            data2: Second data item
            threshold: Similarity threshold for match

        Returns:
            Tuple of (similarity_score, is_match)
        """
        # Convert to strings for comparison
        str1 = json.dumps(data1, sort_keys=True) if not isinstance(data1, str) else data1
        str2 = json.dumps(data2, sort_keys=True) if not isinstance(data2, str) else data2

        # Calculate similarity ratio
        similarity = difflib.SequenceMatcher(None, str1, str2).ratio()

        return similarity, similarity >= threshold

    @staticmethod
    def find_similar(
        target: Any,
        candidates: List[Any],
        threshold: float = 0.85,
        top_n: int = 5
    ) -> List[Tuple[Any, float]]:
        """
        Find similar items from a list of candidates.

        Args:
            target: Target item to match
            candidates: List of candidates
            threshold: Minimum similarity threshold
            top_n: Return top N matches

        Returns:
            List of (candidate, similarity_score) tuples
        """
        matches = []

        for candidate in candidates:
            score, is_match = FuzzyMatcher.calculate_similarity(
                target,
                candidate,
                threshold
            )

            if is_match:
                matches.append((candidate, score))

        # Sort by similarity score
        matches.sort(key=lambda x: x[1], reverse=True)

        return matches[:top_n]


class DataDeduplicator:
    """
    Main deduplication engine for GreenLang.

    Combines multiple strategies for comprehensive duplicate detection.
    """

    def __init__(
        self,
        strategy: DeduplicationStrategy = DeduplicationStrategy.EXACT_MATCH,
        cache_size: int = 100000,
        ttl: Optional[timedelta] = None,
        similarity_threshold: float = 0.85
    ):
        """
        Initialize data deduplicator.

        Args:
            strategy: Deduplication strategy to use
            cache_size: Size of deduplication cache
            ttl: Time to live for cache entries
            similarity_threshold: Threshold for fuzzy matching
        """
        self.strategy = strategy
        self.similarity_threshold = similarity_threshold

        # Initialize cache
        self.cache = DeduplicationCache(max_size=cache_size, ttl=ttl)

        # Hash generator
        self.hash_generator = HashGenerator()

        # Fuzzy matcher
        self.fuzzy_matcher = FuzzyMatcher()

        # Statistics
        self.stats = {
            "total_processed": 0,
            "duplicates_found": 0,
            "unique_records": 0,
            "fuzzy_matches": 0
        }

        # Duplicate history
        self.duplicate_history: List[DuplicateRecord] = []

    def is_duplicate(
        self,
        data: Any,
        keys: Optional[List[str]] = None,
        custom_hash: Optional[str] = None
    ) -> DeduplicationResult:
        """
        Check if data is a duplicate.

        Args:
            data: Data to check
            keys: Specific keys for key-based deduplication
            custom_hash: Pre-computed hash to use

        Returns:
            Deduplication result

        Example:
            dedup = DataDeduplicator(strategy=DeduplicationStrategy.KEY_BASED)

            result = dedup.is_duplicate(
                data=shipment,
                keys=["shipment_id", "origin", "destination"]
            )

            if result.is_duplicate:
                logger.warning(f"Duplicate found: {result.similarity_score}")
            else:
                process_record(shipment)
        """
        self.stats["total_processed"] += 1

        # Generate hash
        if custom_hash:
            hash_value = custom_hash
        elif keys and self.strategy == DeduplicationStrategy.KEY_BASED:
            hash_value = self.hash_generator.generate_hash(data, keys=keys)
        else:
            hash_value = self.hash_generator.generate_hash(data)

        # Check based on strategy
        if self.strategy == DeduplicationStrategy.EXACT_MATCH:
            return self._check_exact_match(hash_value, data)

        elif self.strategy == DeduplicationStrategy.FUZZY_MATCH:
            return self._check_fuzzy_match(data)

        elif self.strategy == DeduplicationStrategy.KEY_BASED:
            return self._check_key_based(hash_value, data, keys)

        elif self.strategy == DeduplicationStrategy.COMPOSITE:
            return self._check_composite(data, keys)

        else:
            # Default to exact match
            return self._check_exact_match(hash_value, data)

    def _check_exact_match(
        self,
        hash_value: str,
        data: Any
    ) -> DeduplicationResult:
        """Check for exact hash match."""
        if self.cache.contains(hash_value):
            self.stats["duplicates_found"] += 1

            original = self.cache.get(hash_value)

            # Record duplicate
            self.duplicate_history.append(
                DuplicateRecord(
                    original_hash=hash_value,
                    duplicate_hash=hash_value,
                    original_timestamp=datetime.now(),
                    duplicate_timestamp=datetime.now(),
                    similarity_score=1.0,
                    strategy_used=DeduplicationStrategy.EXACT_MATCH
                )
            )

            return DeduplicationResult(
                is_duplicate=True,
                original_record=original,
                similarity_score=1.0,
                strategy_matched=DeduplicationStrategy.EXACT_MATCH,
                suggested_action=DuplicateAction.SKIP
            )

        # Not a duplicate - add to cache
        self.cache.add(hash_value, data)
        self.stats["unique_records"] += 1

        return DeduplicationResult(
            is_duplicate=False,
            similarity_score=0.0,
            strategy_matched=DeduplicationStrategy.EXACT_MATCH
        )

    def _check_fuzzy_match(self, data: Any) -> DeduplicationResult:
        """Check for fuzzy match."""
        # Get all cached items for comparison
        candidates = []
        for hash_value in list(self.cache._cache.keys())[:1000]:  # Limit comparison
            candidate = self.cache.get(hash_value)
            if candidate:
                candidates.append(candidate)

        # Find similar items
        matches = self.fuzzy_matcher.find_similar(
            data,
            candidates,
            self.similarity_threshold,
            top_n=1
        )

        if matches:
            original, score = matches[0]
            self.stats["fuzzy_matches"] += 1
            self.stats["duplicates_found"] += 1

            return DeduplicationResult(
                is_duplicate=True,
                original_record=original,
                similarity_score=score,
                strategy_matched=DeduplicationStrategy.FUZZY_MATCH,
                suggested_action=DuplicateAction.MERGE if score < 0.95 else DuplicateAction.SKIP
            )

        # Not a duplicate
        hash_value = self.hash_generator.generate_hash(data)
        self.cache.add(hash_value, data)
        self.stats["unique_records"] += 1

        return DeduplicationResult(
            is_duplicate=False,
            similarity_score=0.0,
            strategy_matched=DeduplicationStrategy.FUZZY_MATCH
        )

    def _check_key_based(
        self,
        hash_value: str,
        data: Any,
        keys: Optional[List[str]]
    ) -> DeduplicationResult:
        """Check based on specific keys."""
        if self.cache.contains(hash_value):
            self.stats["duplicates_found"] += 1
            original = self.cache.get(hash_value)

            # Check if non-key fields differ
            if isinstance(data, dict) and isinstance(original, dict):
                differs = any(
                    data.get(k) != original.get(k)
                    for k in data.keys()
                    if k not in (keys or [])
                )

                action = DuplicateAction.UPDATE if differs else DuplicateAction.SKIP
            else:
                action = DuplicateAction.SKIP

            return DeduplicationResult(
                is_duplicate=True,
                original_record=original,
                similarity_score=1.0,
                strategy_matched=DeduplicationStrategy.KEY_BASED,
                suggested_action=action
            )

        # Not a duplicate
        self.cache.add(hash_value, data)
        self.stats["unique_records"] += 1

        return DeduplicationResult(
            is_duplicate=False,
            similarity_score=0.0,
            strategy_matched=DeduplicationStrategy.KEY_BASED
        )

    def _check_composite(
        self,
        data: Any,
        keys: Optional[List[str]]
    ) -> DeduplicationResult:
        """Check using multiple strategies."""
        # First check exact match
        hash_value = self.hash_generator.generate_hash(data)
        exact_result = self._check_exact_match(hash_value, data)

        if exact_result.is_duplicate:
            return exact_result

        # Then check fuzzy match
        fuzzy_result = self._check_fuzzy_match(data)

        if fuzzy_result.is_duplicate:
            return fuzzy_result

        # Finally check key-based if keys provided
        if keys:
            key_hash = self.hash_generator.generate_hash(data, keys=keys)
            return self._check_key_based(key_hash, data, keys)

        return DeduplicationResult(
            is_duplicate=False,
            similarity_score=0.0,
            strategy_matched=DeduplicationStrategy.COMPOSITE
        )

    def batch_deduplicate(
        self,
        records: List[Any],
        keys: Optional[List[str]] = None,
        action: DuplicateAction = DuplicateAction.SKIP
    ) -> Tuple[List[Any], List[DuplicateRecord]]:
        """
        Deduplicate a batch of records.

        Args:
            records: List of records to deduplicate
            keys: Keys for key-based deduplication
            action: Action to take on duplicates

        Returns:
            Tuple of (unique_records, duplicate_records)

        Example:
            unique, duplicates = dedup.batch_deduplicate(
                records=shipments,
                keys=["shipment_id"],
                action=DuplicateAction.SKIP
            )
        """
        unique_records = []
        duplicate_records = []

        for record in records:
            result = self.is_duplicate(record, keys=keys)

            if result.is_duplicate:
                duplicate_records.append(
                    DuplicateRecord(
                        original_hash="",
                        duplicate_hash="",
                        original_timestamp=datetime.now(),
                        duplicate_timestamp=datetime.now(),
                        similarity_score=result.similarity_score,
                        strategy_used=result.strategy_matched or self.strategy,
                        metadata={"action": action.value}
                    )
                )

                # Handle based on action
                if action == DuplicateAction.UPDATE:
                    # Replace original with new version
                    unique_records = [
                        record if r != result.original_record else record
                        for r in unique_records
                    ]
                elif action == DuplicateAction.MERGE:
                    # Merge records (for dicts)
                    if isinstance(record, dict) and isinstance(result.original_record, dict):
                        merged = {**result.original_record, **record}
                        unique_records = [
                            merged if r == result.original_record else r
                            for r in unique_records
                        ]
                # SKIP action - do nothing

            else:
                unique_records.append(record)

        return unique_records, duplicate_records

    def get_statistics(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        return {
            **self.stats,
            "cache_stats": self.cache.get_statistics(),
            "duplicate_history_size": len(self.duplicate_history),
            "deduplication_rate": (
                self.stats["duplicates_found"] / self.stats["total_processed"]
                if self.stats["total_processed"] > 0 else 0
            )
        }

    def clear_cache(self):
        """Clear deduplication cache."""
        self.cache = DeduplicationCache(
            max_size=self.cache.max_size,
            ttl=self.cache.ttl
        )
        logger.info("Deduplication cache cleared")

    def export_duplicate_report(self, path: str):
        """
        Export duplicate detection report.

        Args:
            path: Path to export report
        """
        report = {
            "statistics": self.get_statistics(),
            "duplicate_history": [
                {
                    "original_hash": d.original_hash,
                    "duplicate_hash": d.duplicate_hash,
                    "similarity_score": d.similarity_score,
                    "strategy": d.strategy_used.value,
                    "timestamp": d.duplicate_timestamp.isoformat()
                }
                for d in self.duplicate_history[-100:]  # Last 100 duplicates
            ]
        }

        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Duplicate report exported to {path}")


class ConnectorDeduplicator(ABC):
    """
    Abstract base class for connector-specific deduplication.

    To be implemented by each connector for custom deduplication logic.
    """

    def __init__(self):
        """Initialize connector deduplicator."""
        self.deduplicator = DataDeduplicator(
            strategy=self.get_strategy(),
            cache_size=self.get_cache_size()
        )

    @abstractmethod
    def get_strategy(self) -> DeduplicationStrategy:
        """Get deduplication strategy for this connector."""
        pass

    @abstractmethod
    def get_cache_size(self) -> int:
        """Get cache size for this connector."""
        pass

    @abstractmethod
    def get_dedup_keys(self, record_type: str) -> List[str]:
        """Get deduplication keys for a record type."""
        pass

    def deduplicate_record(
        self,
        record: Dict[str, Any],
        record_type: str
    ) -> DeduplicationResult:
        """
        Deduplicate a single record.

        Args:
            record: Record to check
            record_type: Type of record

        Returns:
            Deduplication result
        """
        keys = self.get_dedup_keys(record_type)
        return self.deduplicator.is_duplicate(record, keys=keys)

    def deduplicate_batch(
        self,
        records: List[Dict[str, Any]],
        record_type: str
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Deduplicate a batch of records.

        Args:
            records: Records to deduplicate
            record_type: Type of records

        Returns:
            Tuple of (unique_records, duplicate_count)
        """
        keys = self.get_dedup_keys(record_type)
        unique, duplicates = self.deduplicator.batch_deduplicate(
            records,
            keys=keys
        )

        logger.info(
            f"Deduplicated {len(records)} {record_type} records: "
            f"{len(unique)} unique, {len(duplicates)} duplicates"
        )

        return unique, len(duplicates)