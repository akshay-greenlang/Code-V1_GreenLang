"""
Transaction management for GreenLang database operations.

Provides ACID-compliant transaction context managers with automatic rollback,
logging, and retry logic for critical data operations.
"""

import logging
import asyncio
import time
from contextlib import contextmanager, asynccontextmanager
from typing import Optional, Any, Dict, Callable, List
from datetime import datetime
import traceback
import uuid
from enum import Enum
from dataclasses import dataclass, field
from greenlang.determinism import deterministic_uuid, DeterministicClock

logger = logging.getLogger(__name__)


class TransactionState(Enum):
    """Transaction states for tracking."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class IsolationLevel(Enum):
    """Database isolation levels."""
    READ_UNCOMMITTED = "READ UNCOMMITTED"
    READ_COMMITTED = "READ COMMITTED"
    REPEATABLE_READ = "REPEATABLE READ"
    SERIALIZABLE = "SERIALIZABLE"


@dataclass
class TransactionLog:
    """Transaction log entry for audit trail."""
    transaction_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    state: TransactionState = TransactionState.PENDING
    operations: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    rollback_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TransactionManager:
    """
    Manages database transactions with automatic rollback and logging.

    Features:
    - ACID compliance
    - Automatic rollback on failure
    - Transaction logging and audit trail
    - Nested transaction support
    - Deadlock detection and retry
    - Performance monitoring
    """

    def __init__(
        self,
        db_connection: Any,
        isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED,
        max_retries: int = 3,
        retry_delay: float = 0.1
    ):
        """
        Initialize transaction manager.

        Args:
            db_connection: Database connection object
            isolation_level: Transaction isolation level
            max_retries: Maximum retry attempts on deadlock
            retry_delay: Delay between retries in seconds
        """
        self.db_connection = db_connection
        self.isolation_level = isolation_level
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.transaction_logs: Dict[str, TransactionLog] = {}
        self._active_transactions: Dict[str, Any] = {}
        self._savepoints: Dict[str, List[str]] = {}

    @contextmanager
    def transaction(
        self,
        name: Optional[str] = None,
        timeout: Optional[float] = None,
        readonly: bool = False
    ):
        """
        Transaction context manager with automatic rollback.

        Args:
            name: Optional transaction name for logging
            timeout: Transaction timeout in seconds
            readonly: Whether this is a read-only transaction

        Yields:
            Transaction cursor or connection

        Example:
            with transaction_manager.transaction("update_emissions") as tx:
                tx.execute("UPDATE emissions SET value = ? WHERE id = ?", (100, 1))
                tx.execute("INSERT INTO audit_log VALUES (?)", (log_entry,))
                # Automatically commits on success, rolls back on exception
        """
        transaction_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))
        transaction_name = name or f"transaction_{transaction_id[:8]}"

        # Create transaction log
        log_entry = TransactionLog(
            transaction_id=transaction_id,
            start_time=DeterministicClock.now(),
            metadata={
                "name": transaction_name,
                "readonly": readonly,
                "isolation_level": self.isolation_level.value
            }
        )
        self.transaction_logs[transaction_id] = log_entry

        # Start transaction with retry logic
        cursor = None
        retry_count = 0

        while retry_count <= self.max_retries:
            try:
                # Begin transaction
                cursor = self.db_connection.cursor()

                # Set isolation level
                if not readonly:
                    cursor.execute(f"SET TRANSACTION ISOLATION LEVEL {self.isolation_level.value}")

                cursor.execute("BEGIN TRANSACTION")
                log_entry.state = TransactionState.IN_PROGRESS
                self._active_transactions[transaction_id] = cursor

                logger.info(f"Started transaction: {transaction_name} (ID: {transaction_id})")

                # Set timeout if specified
                if timeout:
                    cursor.execute(f"SET LOCAL statement_timeout = {int(timeout * 1000)}")

                # Yield cursor to execute operations
                yield cursor

                # Commit on success
                cursor.execute("COMMIT")
                log_entry.state = TransactionState.COMMITTED
                log_entry.end_time = DeterministicClock.now()

                duration = (log_entry.end_time - log_entry.start_time).total_seconds()
                logger.info(f"Committed transaction: {transaction_name} (Duration: {duration:.3f}s)")

                break  # Success, exit retry loop

            except Exception as e:
                # Rollback on any error
                if cursor:
                    try:
                        cursor.execute("ROLLBACK")
                        log_entry.state = TransactionState.ROLLED_BACK
                        log_entry.rollback_reason = str(e)
                        log_entry.error = traceback.format_exc()

                        logger.error(f"Rolled back transaction: {transaction_name} - {str(e)}")

                    except Exception as rollback_error:
                        logger.error(f"Error during rollback: {rollback_error}")
                        log_entry.state = TransactionState.FAILED

                # Check if we should retry (deadlock/timeout)
                if self._is_retryable_error(e) and retry_count < self.max_retries:
                    retry_count += 1
                    logger.warning(f"Retrying transaction {transaction_name} (attempt {retry_count}/{self.max_retries})")
                    time.sleep(self.retry_delay * (2 ** retry_count))  # Exponential backoff
                    continue

                # Re-raise if not retryable or max retries exceeded
                raise

            finally:
                # Cleanup
                if transaction_id in self._active_transactions:
                    del self._active_transactions[transaction_id]
                if cursor:
                    cursor.close()
                log_entry.end_time = log_entry.end_time or DeterministicClock.now()

    @asynccontextmanager
    async def async_transaction(
        self,
        name: Optional[str] = None,
        timeout: Optional[float] = None
    ):
        """
        Async transaction context manager.

        Example:
            async with transaction_manager.async_transaction("async_update") as tx:
                await tx.execute("UPDATE data SET processed = true WHERE id = ?", (id,))
        """
        transaction_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))
        transaction_name = name or f"async_transaction_{transaction_id[:8]}"

        log_entry = TransactionLog(
            transaction_id=transaction_id,
            start_time=DeterministicClock.now(),
            metadata={"name": transaction_name, "async": True}
        )
        self.transaction_logs[transaction_id] = log_entry

        connection = None
        try:
            # Start async transaction
            connection = await self.db_connection.acquire()
            transaction = connection.transaction()

            await transaction.start()
            log_entry.state = TransactionState.IN_PROGRESS

            logger.info(f"Started async transaction: {transaction_name}")

            yield connection

            # Commit on success
            await transaction.commit()
            log_entry.state = TransactionState.COMMITTED
            log_entry.end_time = DeterministicClock.now()

            logger.info(f"Committed async transaction: {transaction_name}")

        except Exception as e:
            # Rollback on error
            if transaction:
                await transaction.rollback()
                log_entry.state = TransactionState.ROLLED_BACK
                log_entry.rollback_reason = str(e)

            logger.error(f"Rolled back async transaction: {transaction_name} - {str(e)}")
            raise

        finally:
            if connection:
                await self.db_connection.release(connection)
            log_entry.end_time = log_entry.end_time or DeterministicClock.now()

    def savepoint(self, name: str, transaction_id: str) -> str:
        """
        Create a savepoint within a transaction.

        Args:
            name: Savepoint name
            transaction_id: Parent transaction ID

        Returns:
            Savepoint identifier
        """
        if transaction_id not in self._active_transactions:
            raise ValueError(f"No active transaction with ID: {transaction_id}")

        cursor = self._active_transactions[transaction_id]
        savepoint_id = f"sp_{name}_{int(time.time() * 1000)}"

        cursor.execute(f"SAVEPOINT {savepoint_id}")

        if transaction_id not in self._savepoints:
            self._savepoints[transaction_id] = []
        self._savepoints[transaction_id].append(savepoint_id)

        logger.debug(f"Created savepoint: {savepoint_id}")
        return savepoint_id

    def rollback_to_savepoint(self, savepoint_id: str, transaction_id: str):
        """
        Rollback to a specific savepoint.

        Args:
            savepoint_id: Savepoint identifier
            transaction_id: Parent transaction ID
        """
        if transaction_id not in self._active_transactions:
            raise ValueError(f"No active transaction with ID: {transaction_id}")

        cursor = self._active_transactions[transaction_id]
        cursor.execute(f"ROLLBACK TO SAVEPOINT {savepoint_id}")

        logger.info(f"Rolled back to savepoint: {savepoint_id}")

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Check if error is retryable (deadlock, timeout, etc.).

        Args:
            error: Exception to check

        Returns:
            True if retryable
        """
        error_msg = str(error).lower()
        retryable_patterns = [
            "deadlock",
            "lock timeout",
            "connection timeout",
            "serialization failure",
            "concurrent update"
        ]
        return any(pattern in error_msg for pattern in retryable_patterns)

    def get_transaction_log(self, transaction_id: str) -> Optional[TransactionLog]:
        """
        Get transaction log by ID.

        Args:
            transaction_id: Transaction identifier

        Returns:
            Transaction log entry if found
        """
        return self.transaction_logs.get(transaction_id)

    def get_transaction_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        state: Optional[TransactionState] = None
    ) -> List[TransactionLog]:
        """
        Get transaction history with filters.

        Args:
            start_time: Filter by start time
            end_time: Filter by end time
            state: Filter by transaction state

        Returns:
            List of matching transaction logs
        """
        logs = list(self.transaction_logs.values())

        if start_time:
            logs = [l for l in logs if l.start_time >= start_time]
        if end_time:
            logs = [l for l in logs if l.start_time <= end_time]
        if state:
            logs = [l for l in logs if l.state == state]

        return sorted(logs, key=lambda x: x.start_time, reverse=True)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get transaction metrics for monitoring.

        Returns:
            Dictionary with transaction metrics
        """
        logs = list(self.transaction_logs.values())

        if not logs:
            return {
                "total_transactions": 0,
                "committed": 0,
                "rolled_back": 0,
                "failed": 0,
                "average_duration": 0
            }

        committed = [l for l in logs if l.state == TransactionState.COMMITTED]
        durations = [
            (l.end_time - l.start_time).total_seconds()
            for l in committed
            if l.end_time
        ]

        return {
            "total_transactions": len(logs),
            "committed": len([l for l in logs if l.state == TransactionState.COMMITTED]),
            "rolled_back": len([l for l in logs if l.state == TransactionState.ROLLED_BACK]),
            "failed": len([l for l in logs if l.state == TransactionState.FAILED]),
            "average_duration": sum(durations) / len(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "active_transactions": len(self._active_transactions)
        }


# Decorator for transactional methods
def transactional(
    isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED,
    timeout: Optional[float] = None,
    max_retries: int = 3
):
    """
    Decorator to make a method transactional.

    Args:
        isolation_level: Transaction isolation level
        timeout: Transaction timeout in seconds
        max_retries: Maximum retry attempts

    Example:
        @transactional(isolation_level=IsolationLevel.SERIALIZABLE)
        def update_emissions(self, data):
            # All database operations in this method will be wrapped in a transaction
            self.db.execute("UPDATE emissions SET ...")
            self.db.execute("INSERT INTO audit_log ...")
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            # Get or create transaction manager
            if not hasattr(self, '_transaction_manager'):
                raise ValueError("Object must have _transaction_manager attribute")

            with self._transaction_manager.transaction(
                name=func.__name__,
                timeout=timeout
            ):
                return func(self, *args, **kwargs)

        async def async_wrapper(self, *args, **kwargs):
            if not hasattr(self, '_transaction_manager'):
                raise ValueError("Object must have _transaction_manager attribute")

            async with self._transaction_manager.async_transaction(
                name=func.__name__,
                timeout=timeout
            ):
                return await func(self, *args, **kwargs)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator