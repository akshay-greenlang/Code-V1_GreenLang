# -*- coding: utf-8 -*-
"""
Transaction Manager with Dead Letter Queue for GreenLang

Provides transaction management, retry logic, and dead letter queue
for failed ERP data extraction operations.
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import traceback

import redis
from redis import Redis
from redis.exceptions import RedisError
import httpx

logger = logging.getLogger(__name__)


class TransactionStatus(Enum):
    """Transaction processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"
    RETRY = "retry"


@dataclass
class Transaction:
    """Transaction data structure."""
    id: str
    source_system: str  # SAP, Oracle, Workday
    entity_type: str  # purchase_orders, suppliers, etc.
    query: Dict[str, Any]
    status: TransactionStatus
    attempts: int = 0
    max_attempts: int = 3
    created_at: str = ""
    updated_at: str = ""
    error_message: Optional[str] = None
    error_details: Optional[Dict] = None
    result_count: int = 0
    processing_time_ms: int = 0

    def __post_init__(self):
        """Initialize timestamps if not provided."""
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.updated_at:
            self.updated_at = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['status'] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transaction':
        """Create from dictionary."""
        data['status'] = TransactionStatus(data['status'])
        return cls(**data)


class DeadLetterQueue:
    """
    Dead Letter Queue for failed transactions.

    Stores failed transactions for manual review and retry.
    """

    def __init__(self, redis_client: Optional[Redis] = None,
                 redis_url: str = "redis://localhost:6379/0"):
        """Initialize Dead Letter Queue."""
        self.key_prefix = "dlq"

        # Initialize Redis client
        if redis_client:
            self.redis = redis_client
        else:
            try:
                self.redis = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5
                )
                self.redis.ping()
                logger.info("Dead Letter Queue connected to Redis")
            except RedisError as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise

    def add(self, transaction: Transaction, error: Exception) -> bool:
        """
        Add failed transaction to dead letter queue.

        Args:
            transaction: Failed transaction
            error: Exception that caused failure

        Returns:
            True if successfully added
        """
        try:
            # Update transaction with error details
            transaction.status = TransactionStatus.DEAD_LETTER
            transaction.error_message = str(error)
            transaction.error_details = {
                'exception_type': type(error).__name__,
                'traceback': traceback.format_exc(),
                'timestamp': datetime.utcnow().isoformat()
            }
            transaction.updated_at = datetime.utcnow().isoformat()

            # Store in Redis with expiration (30 days)
            key = f"{self.key_prefix}:{transaction.source_system}:{transaction.id}"
            self.redis.setex(
                key,
                30 * 24 * 60 * 60,  # 30 days TTL
                json.dumps(transaction.to_dict())
            )

            # Add to DLQ set for tracking
            dlq_set_key = f"{self.key_prefix}:transactions"
            self.redis.sadd(dlq_set_key, key)

            # Track by source system
            system_key = f"{self.key_prefix}:system:{transaction.source_system}"
            self.redis.sadd(system_key, transaction.id)

            logger.warning(f"Transaction {transaction.id} added to DLQ: {error}")
            return True

        except RedisError as e:
            logger.error(f"Failed to add transaction to DLQ: {e}")
            return False

    def get(self, transaction_id: str, source_system: str) -> Optional[Transaction]:
        """
        Get transaction from dead letter queue.

        Args:
            transaction_id: Transaction ID
            source_system: Source system (SAP, Oracle, Workday)

        Returns:
            Transaction if found, None otherwise
        """
        try:
            key = f"{self.key_prefix}:{source_system}:{transaction_id}"
            data = self.redis.get(key)

            if data:
                return Transaction.from_dict(json.loads(data))
            return None

        except Exception as e:
            logger.error(f"Error retrieving from DLQ: {e}")
            return None

    def list_all(self, source_system: Optional[str] = None,
                 limit: int = 100) -> List[Transaction]:
        """
        List transactions in dead letter queue.

        Args:
            source_system: Filter by source system
            limit: Maximum number of transactions to return

        Returns:
            List of failed transactions
        """
        try:
            transactions = []

            if source_system:
                # Get transactions for specific system
                system_key = f"{self.key_prefix}:system:{source_system}"
                transaction_ids = self.redis.smembers(system_key)

                for tid in list(transaction_ids)[:limit]:
                    transaction = self.get(tid, source_system)
                    if transaction:
                        transactions.append(transaction)
            else:
                # Get all transactions
                dlq_set_key = f"{self.key_prefix}:transactions"
                keys = self.redis.smembers(dlq_set_key)

                for key in list(keys)[:limit]:
                    data = self.redis.get(key)
                    if data:
                        transactions.append(Transaction.from_dict(json.loads(data)))

            return transactions

        except Exception as e:
            logger.error(f"Error listing DLQ transactions: {e}")
            return []

    def retry(self, transaction_id: str, source_system: str) -> bool:
        """
        Move transaction from DLQ back to processing queue.

        Args:
            transaction_id: Transaction ID to retry
            source_system: Source system

        Returns:
            True if successfully moved for retry
        """
        try:
            transaction = self.get(transaction_id, source_system)
            if not transaction:
                return False

            # Reset for retry
            transaction.status = TransactionStatus.RETRY
            transaction.attempts = 0
            transaction.error_message = None
            transaction.error_details = None
            transaction.updated_at = datetime.utcnow().isoformat()

            # Remove from DLQ
            self.remove(transaction_id, source_system)

            # Add back to processing queue (would need TransactionManager instance)
            logger.info(f"Transaction {transaction_id} marked for retry")
            return True

        except Exception as e:
            logger.error(f"Error retrying transaction: {e}")
            return False

    def remove(self, transaction_id: str, source_system: str) -> bool:
        """
        Remove transaction from dead letter queue.

        Args:
            transaction_id: Transaction ID
            source_system: Source system

        Returns:
            True if successfully removed
        """
        try:
            key = f"{self.key_prefix}:{source_system}:{transaction_id}"

            # Remove from Redis
            self.redis.delete(key)

            # Remove from tracking sets
            dlq_set_key = f"{self.key_prefix}:transactions"
            self.redis.srem(dlq_set_key, key)

            system_key = f"{self.key_prefix}:system:{source_system}"
            self.redis.srem(system_key, transaction_id)

            logger.info(f"Transaction {transaction_id} removed from DLQ")
            return True

        except Exception as e:
            logger.error(f"Error removing from DLQ: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get DLQ statistics.

        Returns:
            Dictionary with DLQ stats
        """
        try:
            dlq_set_key = f"{self.key_prefix}:transactions"
            total_count = self.redis.scard(dlq_set_key)

            # Count by system
            systems = ['SAP', 'Oracle', 'Workday']
            by_system = {}
            for system in systems:
                system_key = f"{self.key_prefix}:system:{system}"
                by_system[system] = self.redis.scard(system_key)

            return {
                'total_transactions': total_count,
                'by_system': by_system,
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting DLQ stats: {e}")
            return {}


class TransactionManager:
    """
    Manages ERP data extraction transactions with retry logic.

    Features:
    - Transaction tracking and status management
    - Automatic retry with exponential backoff
    - Dead letter queue for failed transactions
    - Performance metrics and monitoring
    - Circuit breaker pattern for failing systems
    """

    def __init__(self, redis_client: Optional[Redis] = None,
                 redis_url: str = "redis://localhost:6379/0"):
        """Initialize Transaction Manager."""
        self.key_prefix = "transaction"
        self.dlq = DeadLetterQueue(redis_client, redis_url)

        # Circuit breaker state
        self.circuit_breakers = {}
        self.circuit_breaker_threshold = 5  # Failures before opening
        self.circuit_breaker_timeout = 300  # Seconds before retry

        # Initialize Redis client
        if redis_client:
            self.redis = redis_client
        else:
            try:
                self.redis = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5
                )
                self.redis.ping()
                logger.info("Transaction Manager connected to Redis")
            except RedisError as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise

    def create_transaction(self, source_system: str, entity_type: str,
                          query: Dict[str, Any]) -> Transaction:
        """
        Create a new transaction.

        Args:
            source_system: ERP system (SAP, Oracle, Workday)
            entity_type: Entity type being queried
            query: Query parameters

        Returns:
            New transaction object
        """
        import uuid
        transaction = Transaction(
            id=str(uuid.uuid4()),
            source_system=source_system,
            entity_type=entity_type,
            query=query,
            status=TransactionStatus.PENDING
        )

        # Store in Redis
        self._store_transaction(transaction)

        # Add to pending queue
        queue_key = f"{self.key_prefix}:queue:pending"
        self.redis.lpush(queue_key, transaction.id)

        logger.info(f"Created transaction {transaction.id} for {source_system}/{entity_type}")
        return transaction

    def _store_transaction(self, transaction: Transaction):
        """Store transaction in Redis."""
        key = f"{self.key_prefix}:{transaction.id}"
        self.redis.setex(
            key,
            24 * 60 * 60,  # 24 hours TTL
            json.dumps(transaction.to_dict())
        )

    def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        """Get transaction by ID."""
        key = f"{self.key_prefix}:{transaction_id}"
        data = self.redis.get(key)

        if data:
            return Transaction.from_dict(json.loads(data))
        return None

    async def process_transaction(self, transaction: Transaction,
                                 processor: Callable) -> bool:
        """
        Process a transaction with retry logic.

        Args:
            transaction: Transaction to process
            processor: Async function to process the transaction

        Returns:
            True if successful, False otherwise
        """
        # Check circuit breaker
        if self._is_circuit_open(transaction.source_system):
            logger.warning(f"Circuit breaker open for {transaction.source_system}")
            self.dlq.add(transaction, Exception("Circuit breaker open"))
            return False

        start_time = datetime.utcnow()
        transaction.status = TransactionStatus.PROCESSING
        transaction.attempts += 1
        self._store_transaction(transaction)

        try:
            # Process the transaction
            result = await processor(transaction.query)

            # Update transaction on success
            transaction.status = TransactionStatus.SUCCESS
            transaction.result_count = len(result) if isinstance(result, list) else 1
            transaction.processing_time_ms = int(
                (datetime.utcnow() - start_time).total_seconds() * 1000
            )
            transaction.updated_at = datetime.utcnow().isoformat()
            self._store_transaction(transaction)

            # Reset circuit breaker on success
            self._reset_circuit_breaker(transaction.source_system)

            logger.info(f"Transaction {transaction.id} completed successfully")
            return True

        except Exception as e:
            logger.error(f"Transaction {transaction.id} failed: {str(e)}")

            # Update circuit breaker
            self._record_failure(transaction.source_system)

            # Check if should retry
            if transaction.attempts < transaction.max_attempts:
                # Calculate backoff delay
                delay = self._calculate_backoff(transaction.attempts)

                # Update transaction for retry
                transaction.status = TransactionStatus.RETRY
                transaction.error_message = str(e)
                transaction.updated_at = datetime.utcnow().isoformat()
                self._store_transaction(transaction)

                # Add to retry queue with delay
                retry_key = f"{self.key_prefix}:queue:retry"
                retry_time = datetime.utcnow() + timedelta(seconds=delay)
                self.redis.zadd(
                    retry_key,
                    {transaction.id: retry_time.timestamp()}
                )

                logger.info(f"Transaction {transaction.id} scheduled for retry in {delay}s")
                return False
            else:
                # Max attempts reached - move to DLQ
                self.dlq.add(transaction, e)
                return False

    def _calculate_backoff(self, attempt: int) -> int:
        """Calculate exponential backoff delay."""
        base_delay = 5  # Base delay in seconds
        max_delay = 300  # Maximum delay
        delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
        return delay

    def _is_circuit_open(self, source_system: str) -> bool:
        """Check if circuit breaker is open for a system."""
        breaker = self.circuit_breakers.get(source_system, {})

        if breaker.get('state') == 'open':
            # Check if timeout has passed
            opened_at = breaker.get('opened_at', 0)
            if datetime.utcnow().timestamp() - opened_at > self.circuit_breaker_timeout:
                # Try to close circuit
                self.circuit_breakers[source_system] = {
                    'state': 'half_open',
                    'failures': 0
                }
                return False
            return True
        return False

    def _record_failure(self, source_system: str):
        """Record a failure for circuit breaker."""
        if source_system not in self.circuit_breakers:
            self.circuit_breakers[source_system] = {
                'state': 'closed',
                'failures': 0
            }

        breaker = self.circuit_breakers[source_system]
        breaker['failures'] += 1

        if breaker['failures'] >= self.circuit_breaker_threshold:
            breaker['state'] = 'open'
            breaker['opened_at'] = datetime.utcnow().timestamp()
            logger.warning(f"Circuit breaker opened for {source_system}")

    def _reset_circuit_breaker(self, source_system: str):
        """Reset circuit breaker for a system."""
        if source_system in self.circuit_breakers:
            self.circuit_breakers[source_system] = {
                'state': 'closed',
                'failures': 0
            }

    async def process_pending_transactions(self, processor_map: Dict[str, Callable]):
        """
        Process all pending transactions.

        Args:
            processor_map: Map of source systems to processor functions
        """
        queue_key = f"{self.key_prefix}:queue:pending"

        while True:
            # Get next pending transaction
            transaction_id = self.redis.rpop(queue_key)
            if not transaction_id:
                break

            transaction = self.get_transaction(transaction_id)
            if not transaction:
                continue

            # Get processor for source system
            processor = processor_map.get(transaction.source_system)
            if not processor:
                logger.error(f"No processor for {transaction.source_system}")
                self.dlq.add(transaction, Exception("No processor available"))
                continue

            # Process transaction
            await self.process_transaction(transaction, processor)

    async def process_retry_queue(self, processor_map: Dict[str, Callable]):
        """Process transactions in retry queue that are due."""
        retry_key = f"{self.key_prefix}:queue:retry"
        current_time = datetime.utcnow().timestamp()

        # Get due transactions
        due_transactions = self.redis.zrangebyscore(
            retry_key,
            0,
            current_time
        )

        for transaction_id in due_transactions:
            # Remove from retry queue
            self.redis.zrem(retry_key, transaction_id)

            # Get and process transaction
            transaction = self.get_transaction(transaction_id)
            if not transaction:
                continue

            processor = processor_map.get(transaction.source_system)
            if processor:
                await self.process_transaction(transaction, processor)

    def get_stats(self) -> Dict[str, Any]:
        """Get transaction manager statistics."""
        try:
            # Count transactions by status
            pending_count = self.redis.llen(f"{self.key_prefix}:queue:pending")
            retry_count = self.redis.zcard(f"{self.key_prefix}:queue:retry")

            # Get DLQ stats
            dlq_stats = self.dlq.get_stats()

            # Get circuit breaker states
            breaker_states = {}
            for system, breaker in self.circuit_breakers.items():
                breaker_states[system] = {
                    'state': breaker['state'],
                    'failures': breaker.get('failures', 0)
                }

            return {
                'pending_transactions': pending_count,
                'retry_transactions': retry_count,
                'dead_letter_queue': dlq_stats,
                'circuit_breakers': breaker_states,
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}


__all__ = ["TransactionManager", "DeadLetterQueue", "Transaction", "TransactionStatus"]