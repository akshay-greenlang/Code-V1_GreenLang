# SAP Deduplication Cache
# Idempotency and duplicate detection for SAP transactions

"""
Deduplication Cache
===================

Provides idempotency and deduplication for SAP transaction processing
using Redis-backed cache.

Features:
---------
- Track processed transaction IDs
- Redis-based distributed cache
- Configurable TTL (default: 7 days)
- Transaction ID hashing for efficiency
- Duplicate detection
- Batch duplicate checking
- Cache statistics

Usage:
------
```python
from connectors.sap.utils.deduplication import DeduplicationCache

# Initialize cache
cache = DeduplicationCache(ttl_days=7)

# Check if transaction already processed
if cache.is_duplicate("PO-12345"):
    print("Already processed")
else:
    # Process transaction
    process_transaction()
    # Mark as processed
    cache.mark_processed("PO-12345")

# Batch checking
transaction_ids = ["PO-12345", "PO-12346", "PO-12347"]
duplicates = cache.filter_duplicates(transaction_ids)
# Returns only IDs not yet processed
```
"""

import hashlib
import logging
from typing import List, Optional, Set

import redis
from redis import Redis
from redis.exceptions import RedisError

# Configure logger
logger = logging.getLogger(__name__)


class DeduplicationCache:
    """
    Redis-backed deduplication cache for SAP transactions.

    Uses Redis sets to track processed transaction IDs with configurable TTL.
    """

    def __init__(
        self,
        ttl_days: int = 7,
        redis_client: Optional[Redis] = None,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "sap:dedup",
        hash_ids: bool = True,
    ):
        """
        Initialize the deduplication cache.

        Args:
            ttl_days: Time-to-live in days (default: 7)
            redis_client: Existing Redis client (optional)
            redis_url: Redis connection URL (default: localhost)
            key_prefix: Prefix for Redis keys (default: "sap:dedup")
            hash_ids: Hash transaction IDs for efficiency (default: True)
        """
        self.ttl_seconds = ttl_days * 24 * 60 * 60
        self.key_prefix = key_prefix
        self.hash_ids = hash_ids

        # Initialize Redis client
        if redis_client:
            self.redis = redis_client
        else:
            try:
                self.redis = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                )
                # Test connection
                self.redis.ping()
                logger.info(f"Deduplication cache connected to Redis at {redis_url}")
            except RedisError as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise

    def is_duplicate(self, transaction_id: str, entity_type: str = "default") -> bool:
        """
        Check if a transaction ID has already been processed.

        Args:
            transaction_id: SAP transaction/document ID
            entity_type: Type of entity (purchase_order, delivery, etc.)

        Returns:
            True if transaction already processed, False otherwise
        """
        key = self._get_key(entity_type)
        hashed_id = self._hash_id(transaction_id) if self.hash_ids else transaction_id

        try:
            exists = self.redis.sismember(key, hashed_id)
            if exists:
                logger.debug(
                    f"Duplicate transaction detected: {transaction_id} ({entity_type})"
                )
            return bool(exists)
        except RedisError as e:
            logger.error(f"Redis error checking duplicate: {e}")
            # Fail open - assume not duplicate if Redis is down
            return False

    def mark_processed(
        self, transaction_id: str, entity_type: str = "default"
    ) -> bool:
        """
        Mark a transaction ID as processed.

        Args:
            transaction_id: SAP transaction/document ID
            entity_type: Type of entity (purchase_order, delivery, etc.)

        Returns:
            True if successfully marked, False on error
        """
        key = self._get_key(entity_type)
        hashed_id = self._hash_id(transaction_id) if self.hash_ids else transaction_id

        try:
            # Add to set with TTL
            pipe = self.redis.pipeline()
            pipe.sadd(key, hashed_id)
            pipe.expire(key, self.ttl_seconds)
            pipe.execute()

            logger.debug(
                f"Marked transaction as processed: {transaction_id} ({entity_type})"
            )
            return True
        except RedisError as e:
            logger.error(f"Redis error marking processed: {e}")
            return False

    def filter_duplicates(
        self, transaction_ids: List[str], entity_type: str = "default"
    ) -> List[str]:
        """
        Filter out duplicate transaction IDs from a list.

        Args:
            transaction_ids: List of SAP transaction IDs
            entity_type: Type of entity

        Returns:
            List of transaction IDs that have not been processed
        """
        if not transaction_ids:
            return []

        key = self._get_key(entity_type)
        non_duplicates = []

        try:
            # Check all IDs in batch
            for transaction_id in transaction_ids:
                hashed_id = (
                    self._hash_id(transaction_id) if self.hash_ids else transaction_id
                )
                if not self.redis.sismember(key, hashed_id):
                    non_duplicates.append(transaction_id)

            duplicates_count = len(transaction_ids) - len(non_duplicates)
            if duplicates_count > 0:
                logger.info(
                    f"Filtered out {duplicates_count} duplicates from {len(transaction_ids)} "
                    f"transactions ({entity_type})"
                )

            return non_duplicates
        except RedisError as e:
            logger.error(f"Redis error filtering duplicates: {e}")
            # Fail open - return all IDs if Redis is down
            return transaction_ids

    def mark_batch_processed(
        self, transaction_ids: List[str], entity_type: str = "default"
    ) -> int:
        """
        Mark multiple transaction IDs as processed in a batch.

        Args:
            transaction_ids: List of SAP transaction IDs
            entity_type: Type of entity

        Returns:
            Number of IDs successfully marked
        """
        if not transaction_ids:
            return 0

        key = self._get_key(entity_type)
        hashed_ids = [
            (self._hash_id(tid) if self.hash_ids else tid) for tid in transaction_ids
        ]

        try:
            # Add all IDs in one operation
            pipe = self.redis.pipeline()
            pipe.sadd(key, *hashed_ids)
            pipe.expire(key, self.ttl_seconds)
            results = pipe.execute()

            count = results[0]  # Number of IDs added
            logger.info(
                f"Marked {count} transactions as processed in batch ({entity_type})"
            )
            return count
        except RedisError as e:
            logger.error(f"Redis error marking batch processed: {e}")
            return 0

    def get_stats(self, entity_type: str = "default") -> dict:
        """
        Get cache statistics for an entity type.

        Args:
            entity_type: Type of entity

        Returns:
            Dictionary with cache statistics
        """
        key = self._get_key(entity_type)

        try:
            count = self.redis.scard(key)
            ttl = self.redis.ttl(key)

            return {
                "entity_type": entity_type,
                "processed_count": count,
                "ttl_seconds": ttl if ttl > 0 else self.ttl_seconds,
            }
        except RedisError as e:
            logger.error(f"Redis error getting stats: {e}")
            return {
                "entity_type": entity_type,
                "processed_count": 0,
                "ttl_seconds": self.ttl_seconds,
            }

    def clear(self, entity_type: str = "default") -> bool:
        """
        Clear the deduplication cache for an entity type.

        Args:
            entity_type: Type of entity

        Returns:
            True if successfully cleared, False on error
        """
        key = self._get_key(entity_type)

        try:
            self.redis.delete(key)
            logger.info(f"Cleared deduplication cache for {entity_type}")
            return True
        except RedisError as e:
            logger.error(f"Redis error clearing cache: {e}")
            return False

    def _get_key(self, entity_type: str) -> str:
        """
        Get Redis key for an entity type.

        Args:
            entity_type: Type of entity

        Returns:
            Redis key string
        """
        return f"{self.key_prefix}:{entity_type}"

    def _hash_id(self, transaction_id: str) -> str:
        """
        Hash a transaction ID for efficient storage.

        Args:
            transaction_id: Transaction ID to hash

        Returns:
            SHA256 hash of the transaction ID
        """
        return hashlib.sha256(transaction_id.encode()).hexdigest()
