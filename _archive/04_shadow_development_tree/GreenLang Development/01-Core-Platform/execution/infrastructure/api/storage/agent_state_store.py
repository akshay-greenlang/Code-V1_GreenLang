"""
Agent State Persistent Storage for GreenLang.

This module provides storage backends for persisting agent state and
calculation results, enabling:
- Agent state checkpointing and recovery
- Calculation result caching
- Job tracking and audit trails

Storage Backends:
- InMemoryAgentStateStore: Testing and development
- SQLiteAgentStateStore: File-based persistence
- PostgresAgentStateStore: Production deployments

Example:
    >>> from greenlang.infrastructure.api.storage import StorageFactory
    >>> store = StorageFactory.get_agent_store({"backend": "sqlite"})
    >>> await store.save_agent_state("agent-1", {"status": "running"})
    >>> state = await store.get_agent_state("agent-1")
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def _datetime_to_iso(dt: Optional[datetime]) -> Optional[str]:
    """Convert datetime to ISO string for storage."""
    if dt is None:
        return None
    return dt.isoformat()


def _iso_to_datetime(iso_str: Optional[str]) -> Optional[datetime]:
    """Convert ISO string back to datetime."""
    if iso_str is None:
        return None
    return datetime.fromisoformat(iso_str)


def _calculate_provenance_hash(data: Dict[str, Any]) -> str:
    """Calculate SHA-256 hash for audit trail."""
    data_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(data_str.encode()).hexdigest()


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------


class AgentState(BaseModel):
    """Model for agent state persistence."""

    agent_id: str = Field(..., description="Unique agent identifier")
    agent_type: str = Field(default="", description="Type of agent")
    state: Dict[str, Any] = Field(default_factory=dict, description="Agent state data")
    status: str = Field(default="idle", description="Agent status")
    last_activity_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last activity timestamp"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="State creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CalculationResult(BaseModel):
    """Model for calculation result persistence."""

    job_id: str = Field(..., description="Unique job identifier")
    agent_id: str = Field(default="", description="Agent that performed calculation")
    calculation_type: str = Field(default="", description="Type of calculation")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Calculation inputs")
    result: Dict[str, Any] = Field(default_factory=dict, description="Calculation result")
    status: str = Field(default="pending", description="Job status")
    error_message: Optional[str] = Field(default=None, description="Error if failed")
    started_at: Optional[datetime] = Field(default=None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Job creation timestamp"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit")
    processing_time_ms: Optional[float] = Field(default=None, description="Processing duration")


class AgentStateStoreConfig(BaseModel):
    """Configuration for agent state storage backend."""

    backend: str = Field(
        default="memory",
        description="Storage backend: 'memory', 'sqlite', or 'postgres'"
    )
    sqlite_path: str = Field(
        default="agent_state.db",
        description="Path to SQLite database file"
    )
    postgres_dsn: str = Field(
        default="",
        description="PostgreSQL connection string"
    )
    postgres_pool_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="PostgreSQL connection pool size"
    )
    enable_wal_mode: bool = Field(
        default=True,
        description="Enable SQLite WAL mode for better concurrency"
    )
    state_ttl_seconds: int = Field(
        default=86400 * 7,  # 7 days
        description="Time-to-live for agent states"
    )
    result_ttl_seconds: int = Field(
        default=86400 * 30,  # 30 days
        description="Time-to-live for calculation results"
    )


# -----------------------------------------------------------------------------
# Abstract Base Class
# -----------------------------------------------------------------------------


@runtime_checkable
class AgentStateStore(Protocol):
    """
    Protocol defining the agent state storage interface.

    All storage backends must implement this protocol to ensure
    consistent behavior across different persistence layers.
    """

    async def save_agent_state(self, agent_id: str, state: Dict[str, Any]) -> None:
        """
        Save or update agent state.

        Args:
            agent_id: Unique agent identifier
            state: State data to persist
        """
        ...

    async def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve agent state by ID.

        Args:
            agent_id: Unique agent identifier

        Returns:
            State dictionary if found, None otherwise
        """
        ...

    async def list_agent_states(self) -> List[Dict[str, Any]]:
        """
        List all agent states.

        Returns:
            List of state dictionaries
        """
        ...

    async def delete_agent_state(self, agent_id: str) -> bool:
        """
        Delete agent state.

        Args:
            agent_id: Unique agent identifier

        Returns:
            True if deleted, False if not found
        """
        ...

    async def save_calculation_result(
        self,
        job_id: str,
        result: Dict[str, Any]
    ) -> None:
        """
        Save calculation result.

        Args:
            job_id: Unique job identifier
            result: Result data to persist
        """
        ...

    async def get_calculation_result(
        self,
        job_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve calculation result by job ID.

        Args:
            job_id: Unique job identifier

        Returns:
            Result dictionary if found, None otherwise
        """
        ...

    async def list_calculation_results(
        self,
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List calculation results with optional filtering.

        Args:
            agent_id: Filter by agent ID
            status: Filter by status
            limit: Maximum results to return
            offset: Number of results to skip

        Returns:
            List of result dictionaries
        """
        ...

    async def close(self) -> None:
        """Close any open connections and release resources."""
        ...


class BaseAgentStateStore(ABC):
    """
    Abstract base class for agent state storage implementations.

    Provides common functionality and enforces the AgentStateStore interface.
    """

    @abstractmethod
    async def save_agent_state(self, agent_id: str, state: Dict[str, Any]) -> None:
        """Save or update agent state."""
        pass

    @abstractmethod
    async def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve agent state by ID."""
        pass

    @abstractmethod
    async def list_agent_states(self) -> List[Dict[str, Any]]:
        """List all agent states."""
        pass

    @abstractmethod
    async def delete_agent_state(self, agent_id: str) -> bool:
        """Delete agent state."""
        pass

    @abstractmethod
    async def save_calculation_result(
        self,
        job_id: str,
        result: Dict[str, Any]
    ) -> None:
        """Save calculation result."""
        pass

    @abstractmethod
    async def get_calculation_result(
        self,
        job_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve calculation result by job ID."""
        pass

    @abstractmethod
    async def list_calculation_results(
        self,
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List calculation results with optional filtering."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close any open connections and release resources."""
        pass


# -----------------------------------------------------------------------------
# InMemory Implementation (Default, Testing)
# -----------------------------------------------------------------------------


class InMemoryAgentStateStore(BaseAgentStateStore):
    """
    In-memory agent state storage for testing and development.

    This implementation stores all data in memory and does not persist
    across restarts. It is thread-safe using asyncio locks.

    Attributes:
        _states: Dictionary mapping agent_id to AgentState
        _results: Dictionary mapping job_id to CalculationResult

    Example:
        >>> store = InMemoryAgentStateStore()
        >>> await store.save_agent_state("agent-1", {"status": "running"})
        >>> state = await store.get_agent_state("agent-1")
        >>> assert state["status"] == "running"
    """

    def __init__(self):
        """Initialize in-memory storage."""
        self._states: Dict[str, AgentState] = {}
        self._results: Dict[str, CalculationResult] = {}
        self._lock = asyncio.Lock()
        self._provenance_log: List[Dict[str, Any]] = []

        logger.info("InMemoryAgentStateStore initialized")

    async def save_agent_state(self, agent_id: str, state: Dict[str, Any]) -> None:
        """
        Save or update agent state in memory.

        Args:
            agent_id: Unique agent identifier
            state: State data to persist
        """
        async with self._lock:
            now = datetime.utcnow()
            provenance_hash = _calculate_provenance_hash({
                "agent_id": agent_id,
                "state": state
            })

            # Check if exists for update
            existing = self._states.get(agent_id)
            created_at = existing.created_at if existing else now

            agent_state = AgentState(
                agent_id=agent_id,
                agent_type=state.get("agent_type", ""),
                state=state,
                status=state.get("status", "idle"),
                last_activity_at=now,
                created_at=created_at,
                updated_at=now,
                provenance_hash=provenance_hash,
                metadata=state.get("metadata", {})
            )

            self._states[agent_id] = agent_state

            # Log provenance
            self._provenance_log.append({
                "action": "save_agent_state",
                "agent_id": agent_id,
                "timestamp": now.isoformat(),
                "hash": provenance_hash
            })

            logger.debug(f"Saved agent state for {agent_id} to memory")

    async def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve agent state from memory.

        Args:
            agent_id: Unique agent identifier

        Returns:
            State dictionary if found, None otherwise
        """
        async with self._lock:
            agent_state = self._states.get(agent_id)
            if agent_state:
                logger.debug(f"Retrieved agent state for {agent_id} from memory")
                return agent_state.dict()
            return None

    async def list_agent_states(self) -> List[Dict[str, Any]]:
        """
        List all agent states from memory.

        Returns:
            List of state dictionaries
        """
        async with self._lock:
            states = [s.dict() for s in self._states.values()]
            logger.debug(f"Listed {len(states)} agent states from memory")
            return states

    async def delete_agent_state(self, agent_id: str) -> bool:
        """
        Delete agent state from memory.

        Args:
            agent_id: Unique agent identifier

        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            if agent_id in self._states:
                del self._states[agent_id]

                self._provenance_log.append({
                    "action": "delete_agent_state",
                    "agent_id": agent_id,
                    "timestamp": datetime.utcnow().isoformat()
                })

                logger.debug(f"Deleted agent state for {agent_id} from memory")
                return True
            return False

    async def save_calculation_result(
        self,
        job_id: str,
        result: Dict[str, Any]
    ) -> None:
        """
        Save calculation result in memory.

        Args:
            job_id: Unique job identifier
            result: Result data to persist
        """
        async with self._lock:
            now = datetime.utcnow()
            provenance_hash = _calculate_provenance_hash({
                "job_id": job_id,
                "result": result
            })

            # Check if exists for update
            existing = self._results.get(job_id)
            created_at = existing.created_at if existing else now
            started_at = existing.started_at if existing else result.get("started_at")

            calc_result = CalculationResult(
                job_id=job_id,
                agent_id=result.get("agent_id", ""),
                calculation_type=result.get("calculation_type", ""),
                inputs=result.get("inputs", {}),
                result=result.get("result", {}),
                status=result.get("status", "pending"),
                error_message=result.get("error_message"),
                started_at=started_at,
                completed_at=result.get("completed_at"),
                created_at=created_at,
                provenance_hash=provenance_hash,
                processing_time_ms=result.get("processing_time_ms")
            )

            self._results[job_id] = calc_result

            logger.debug(f"Saved calculation result for job {job_id} to memory")

    async def get_calculation_result(
        self,
        job_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve calculation result from memory.

        Args:
            job_id: Unique job identifier

        Returns:
            Result dictionary if found, None otherwise
        """
        async with self._lock:
            calc_result = self._results.get(job_id)
            if calc_result:
                logger.debug(f"Retrieved calculation result for job {job_id}")
                return calc_result.dict()
            return None

    async def list_calculation_results(
        self,
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List calculation results from memory with optional filtering.

        Args:
            agent_id: Filter by agent ID
            status: Filter by status
            limit: Maximum results to return
            offset: Number of results to skip

        Returns:
            List of result dictionaries
        """
        async with self._lock:
            results = list(self._results.values())

            # Apply filters
            if agent_id:
                results = [r for r in results if r.agent_id == agent_id]
            if status:
                results = [r for r in results if r.status == status]

            # Sort by created_at descending
            results.sort(key=lambda r: r.created_at, reverse=True)

            # Apply pagination
            paginated = results[offset:offset + limit]

            logger.debug(f"Listed {len(paginated)} calculation results")
            return [r.dict() for r in paginated]

    async def close(self) -> None:
        """Clear memory storage."""
        async with self._lock:
            self._states.clear()
            self._results.clear()
            self._provenance_log.clear()
            logger.info("InMemoryAgentStateStore closed")

    def get_provenance_log(self) -> List[Dict[str, Any]]:
        """Get the provenance audit log (for testing/debugging)."""
        return list(self._provenance_log)


# -----------------------------------------------------------------------------
# SQLite Implementation (File-based Persistence)
# -----------------------------------------------------------------------------


class SQLiteAgentStateStore(BaseAgentStateStore):
    """
    SQLite-based agent state storage for file-based persistence.

    Suitable for single-node deployments or development environments
    that require data persistence across restarts.

    Features:
        - WAL mode for better concurrent access
        - Automatic schema migration
        - TTL-based cleanup

    Attributes:
        db_path: Path to the SQLite database file

    Example:
        >>> store = SQLiteAgentStateStore("agent_state.db")
        >>> await store.initialize()
        >>> await store.save_agent_state("agent-1", {"status": "running"})
    """

    def __init__(self, db_path: str = "agent_state.db", enable_wal: bool = True):
        """
        Initialize SQLite agent state store.

        Args:
            db_path: Path to the SQLite database file
            enable_wal: Enable WAL mode for better concurrency
        """
        self.db_path = db_path
        self._enable_wal = enable_wal
        self._connection: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()
        self._initialized = False

        logger.info(f"SQLiteAgentStateStore initialized with path: {db_path}")

    async def initialize(self) -> None:
        """
        Initialize the database schema.

        Creates tables if they don't exist and applies migrations.
        """
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            await asyncio.get_event_loop().run_in_executor(
                None, self._create_tables
            )
            self._initialized = True
            logger.info("SQLiteAgentStateStore database initialized")

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a database connection."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False
            )
            self._connection.row_factory = sqlite3.Row

            if self._enable_wal:
                self._connection.execute("PRAGMA journal_mode=WAL")

        return self._connection

    def _create_tables(self) -> None:
        """Create database tables."""
        conn = self._get_connection()

        # Agent states table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_states (
                agent_id TEXT PRIMARY KEY,
                agent_type TEXT DEFAULT '',
                state TEXT NOT NULL,
                status TEXT DEFAULT 'idle',
                last_activity_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                provenance_hash TEXT,
                metadata TEXT
            )
        """)

        # Calculation results table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS calculation_results (
                job_id TEXT PRIMARY KEY,
                agent_id TEXT DEFAULT '',
                calculation_type TEXT DEFAULT '',
                inputs TEXT NOT NULL,
                result TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                error_message TEXT,
                started_at TEXT,
                completed_at TEXT,
                created_at TEXT NOT NULL,
                provenance_hash TEXT,
                processing_time_ms REAL
            )
        """)

        # Indexes
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_states_status
            ON agent_states(status)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_calculation_results_agent_id
            ON calculation_results(agent_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_calculation_results_status
            ON calculation_results(status)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_calculation_results_created_at
            ON calculation_results(created_at DESC)
        """)

        conn.commit()

    async def save_agent_state(self, agent_id: str, state: Dict[str, Any]) -> None:
        """
        Save or update agent state in SQLite.

        Args:
            agent_id: Unique agent identifier
            state: State data to persist
        """
        await self.initialize()

        async with self._lock:
            def _save():
                conn = self._get_connection()
                now = datetime.utcnow()
                provenance_hash = _calculate_provenance_hash({
                    "agent_id": agent_id,
                    "state": state
                })

                # Check if exists
                cursor = conn.execute(
                    "SELECT created_at FROM agent_states WHERE agent_id = ?",
                    (agent_id,)
                )
                row = cursor.fetchone()
                created_at = row["created_at"] if row else _datetime_to_iso(now)

                conn.execute("""
                    INSERT OR REPLACE INTO agent_states (
                        agent_id, agent_type, state, status, last_activity_at,
                        created_at, updated_at, provenance_hash, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    agent_id,
                    state.get("agent_type", ""),
                    json.dumps(state, default=str),
                    state.get("status", "idle"),
                    _datetime_to_iso(now),
                    created_at,
                    _datetime_to_iso(now),
                    provenance_hash,
                    json.dumps(state.get("metadata", {}))
                ))
                conn.commit()

            await asyncio.get_event_loop().run_in_executor(None, _save)
            logger.debug(f"Saved agent state for {agent_id} to SQLite")

    async def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve agent state from SQLite.

        Args:
            agent_id: Unique agent identifier

        Returns:
            State dictionary if found, None otherwise
        """
        await self.initialize()

        async with self._lock:
            def _get():
                conn = self._get_connection()
                cursor = conn.execute(
                    "SELECT * FROM agent_states WHERE agent_id = ?",
                    (agent_id,)
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                return {
                    "agent_id": row["agent_id"],
                    "agent_type": row["agent_type"],
                    "state": json.loads(row["state"]),
                    "status": row["status"],
                    "last_activity_at": _iso_to_datetime(row["last_activity_at"]),
                    "created_at": _iso_to_datetime(row["created_at"]),
                    "updated_at": _iso_to_datetime(row["updated_at"]),
                    "provenance_hash": row["provenance_hash"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
                }

            result = await asyncio.get_event_loop().run_in_executor(None, _get)
            if result:
                logger.debug(f"Retrieved agent state for {agent_id}")
            return result

    async def list_agent_states(self) -> List[Dict[str, Any]]:
        """
        List all agent states from SQLite.

        Returns:
            List of state dictionaries
        """
        await self.initialize()

        async with self._lock:
            def _list():
                conn = self._get_connection()
                cursor = conn.execute(
                    "SELECT * FROM agent_states ORDER BY updated_at DESC"
                )
                rows = cursor.fetchall()

                states = []
                for row in rows:
                    states.append({
                        "agent_id": row["agent_id"],
                        "agent_type": row["agent_type"],
                        "state": json.loads(row["state"]),
                        "status": row["status"],
                        "last_activity_at": _iso_to_datetime(row["last_activity_at"]),
                        "created_at": _iso_to_datetime(row["created_at"]),
                        "updated_at": _iso_to_datetime(row["updated_at"]),
                        "provenance_hash": row["provenance_hash"],
                        "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
                    })
                return states

            states = await asyncio.get_event_loop().run_in_executor(None, _list)
            logger.debug(f"Listed {len(states)} agent states")
            return states

    async def delete_agent_state(self, agent_id: str) -> bool:
        """
        Delete agent state from SQLite.

        Args:
            agent_id: Unique agent identifier

        Returns:
            True if deleted, False if not found
        """
        await self.initialize()

        async with self._lock:
            def _delete():
                conn = self._get_connection()

                cursor = conn.execute(
                    "SELECT 1 FROM agent_states WHERE agent_id = ?",
                    (agent_id,)
                )
                if cursor.fetchone() is None:
                    return False

                conn.execute(
                    "DELETE FROM agent_states WHERE agent_id = ?",
                    (agent_id,)
                )
                conn.commit()
                return True

            deleted = await asyncio.get_event_loop().run_in_executor(None, _delete)
            if deleted:
                logger.debug(f"Deleted agent state for {agent_id}")
            return deleted

    async def save_calculation_result(
        self,
        job_id: str,
        result: Dict[str, Any]
    ) -> None:
        """
        Save calculation result to SQLite.

        Args:
            job_id: Unique job identifier
            result: Result data to persist
        """
        await self.initialize()

        async with self._lock:
            def _save():
                conn = self._get_connection()
                now = datetime.utcnow()
                provenance_hash = _calculate_provenance_hash({
                    "job_id": job_id,
                    "result": result
                })

                # Check if exists
                cursor = conn.execute(
                    "SELECT created_at, started_at FROM calculation_results WHERE job_id = ?",
                    (job_id,)
                )
                row = cursor.fetchone()
                created_at = row["created_at"] if row else _datetime_to_iso(now)
                started_at = row["started_at"] if row else result.get("started_at")
                if started_at and isinstance(started_at, datetime):
                    started_at = _datetime_to_iso(started_at)

                completed_at = result.get("completed_at")
                if completed_at and isinstance(completed_at, datetime):
                    completed_at = _datetime_to_iso(completed_at)

                conn.execute("""
                    INSERT OR REPLACE INTO calculation_results (
                        job_id, agent_id, calculation_type, inputs, result,
                        status, error_message, started_at, completed_at,
                        created_at, provenance_hash, processing_time_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    job_id,
                    result.get("agent_id", ""),
                    result.get("calculation_type", ""),
                    json.dumps(result.get("inputs", {}), default=str),
                    json.dumps(result.get("result", {}), default=str),
                    result.get("status", "pending"),
                    result.get("error_message"),
                    started_at,
                    completed_at,
                    created_at,
                    provenance_hash,
                    result.get("processing_time_ms")
                ))
                conn.commit()

            await asyncio.get_event_loop().run_in_executor(None, _save)
            logger.debug(f"Saved calculation result for job {job_id}")

    async def get_calculation_result(
        self,
        job_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve calculation result from SQLite.

        Args:
            job_id: Unique job identifier

        Returns:
            Result dictionary if found, None otherwise
        """
        await self.initialize()

        async with self._lock:
            def _get():
                conn = self._get_connection()
                cursor = conn.execute(
                    "SELECT * FROM calculation_results WHERE job_id = ?",
                    (job_id,)
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                return {
                    "job_id": row["job_id"],
                    "agent_id": row["agent_id"],
                    "calculation_type": row["calculation_type"],
                    "inputs": json.loads(row["inputs"]),
                    "result": json.loads(row["result"]),
                    "status": row["status"],
                    "error_message": row["error_message"],
                    "started_at": _iso_to_datetime(row["started_at"]),
                    "completed_at": _iso_to_datetime(row["completed_at"]),
                    "created_at": _iso_to_datetime(row["created_at"]),
                    "provenance_hash": row["provenance_hash"],
                    "processing_time_ms": row["processing_time_ms"]
                }

            result = await asyncio.get_event_loop().run_in_executor(None, _get)
            if result:
                logger.debug(f"Retrieved calculation result for job {job_id}")
            return result

    async def list_calculation_results(
        self,
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List calculation results from SQLite with optional filtering.

        Args:
            agent_id: Filter by agent ID
            status: Filter by status
            limit: Maximum results to return
            offset: Number of results to skip

        Returns:
            List of result dictionaries
        """
        await self.initialize()

        async with self._lock:
            def _list():
                conn = self._get_connection()

                # Build query with filters
                query = "SELECT * FROM calculation_results WHERE 1=1"
                params = []

                if agent_id:
                    query += " AND agent_id = ?"
                    params.append(agent_id)
                if status:
                    query += " AND status = ?"
                    params.append(status)

                query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])

                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

                results = []
                for row in rows:
                    results.append({
                        "job_id": row["job_id"],
                        "agent_id": row["agent_id"],
                        "calculation_type": row["calculation_type"],
                        "inputs": json.loads(row["inputs"]),
                        "result": json.loads(row["result"]),
                        "status": row["status"],
                        "error_message": row["error_message"],
                        "started_at": _iso_to_datetime(row["started_at"]),
                        "completed_at": _iso_to_datetime(row["completed_at"]),
                        "created_at": _iso_to_datetime(row["created_at"]),
                        "provenance_hash": row["provenance_hash"],
                        "processing_time_ms": row["processing_time_ms"]
                    })
                return results

            results = await asyncio.get_event_loop().run_in_executor(None, _list)
            logger.debug(f"Listed {len(results)} calculation results")
            return results

    async def close(self) -> None:
        """Close the database connection."""
        async with self._lock:
            if self._connection:
                self._connection.close()
                self._connection = None
            self._initialized = False
            logger.info("SQLiteAgentStateStore closed")


# -----------------------------------------------------------------------------
# PostgreSQL Implementation (Production)
# -----------------------------------------------------------------------------


class PostgresAgentStateStore(BaseAgentStateStore):
    """
    PostgreSQL-based agent state storage for production deployments.

    Provides high availability, concurrent access, and advanced
    querying capabilities for distributed systems.

    Features:
        - Connection pooling with asyncpg
        - Automatic schema migration
        - JSONB for efficient state storage
        - TTL-based cleanup with pg_cron

    Attributes:
        dsn: PostgreSQL connection string
        pool_size: Maximum number of connections in pool

    Example:
        >>> store = PostgresAgentStateStore(
        ...     dsn="postgresql://user:pass@localhost/greenlang"
        ... )
        >>> await store.initialize()
        >>> await store.save_agent_state("agent-1", {"status": "running"})
    """

    def __init__(
        self,
        dsn: str,
        pool_size: int = 10,
        pool_min_size: int = 2
    ):
        """
        Initialize PostgreSQL agent state store.

        Args:
            dsn: PostgreSQL connection string
            pool_size: Maximum connections in pool
            pool_min_size: Minimum connections in pool
        """
        self.dsn = dsn
        self.pool_size = pool_size
        self.pool_min_size = pool_min_size
        self._pool = None
        self._initialized = False
        self._lock = asyncio.Lock()

        logger.info(f"PostgresAgentStateStore initialized (pool_size={pool_size})")

    async def initialize(self) -> None:
        """
        Initialize the database connection pool and schema.

        Creates tables if they don't exist and applies migrations.
        """
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            try:
                import asyncpg
            except ImportError:
                raise ImportError(
                    "asyncpg is required for PostgresAgentStateStore. "
                    "Install it with: pip install asyncpg"
                )

            self._pool = await asyncpg.create_pool(
                self.dsn,
                min_size=self.pool_min_size,
                max_size=self.pool_size
            )

            await self._create_tables()
            self._initialized = True
            logger.info("PostgresAgentStateStore database initialized")

    async def _create_tables(self) -> None:
        """Create database tables."""
        async with self._pool.acquire() as conn:
            # Agent states table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_states (
                    agent_id TEXT PRIMARY KEY,
                    agent_type TEXT DEFAULT '',
                    state JSONB NOT NULL DEFAULT '{}',
                    status TEXT DEFAULT 'idle',
                    last_activity_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    provenance_hash TEXT,
                    metadata JSONB DEFAULT '{}'
                )
            """)

            # Calculation results table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS calculation_results (
                    job_id TEXT PRIMARY KEY,
                    agent_id TEXT DEFAULT '',
                    calculation_type TEXT DEFAULT '',
                    inputs JSONB NOT NULL DEFAULT '{}',
                    result JSONB NOT NULL DEFAULT '{}',
                    status TEXT DEFAULT 'pending',
                    error_message TEXT,
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    provenance_hash TEXT,
                    processing_time_ms DOUBLE PRECISION
                )
            """)

            # Indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_states_status
                ON agent_states(status)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_states_updated_at
                ON agent_states(updated_at DESC)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_calculation_results_agent_id
                ON calculation_results(agent_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_calculation_results_status
                ON calculation_results(status)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_calculation_results_created_at
                ON calculation_results(created_at DESC)
            """)

    async def save_agent_state(self, agent_id: str, state: Dict[str, Any]) -> None:
        """
        Save or update agent state in PostgreSQL.

        Args:
            agent_id: Unique agent identifier
            state: State data to persist
        """
        await self.initialize()

        now = datetime.utcnow()
        provenance_hash = _calculate_provenance_hash({
            "agent_id": agent_id,
            "state": state
        })

        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO agent_states (
                    agent_id, agent_type, state, status, last_activity_at,
                    created_at, updated_at, provenance_hash, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (agent_id) DO UPDATE SET
                    agent_type = EXCLUDED.agent_type,
                    state = EXCLUDED.state,
                    status = EXCLUDED.status,
                    last_activity_at = EXCLUDED.last_activity_at,
                    updated_at = EXCLUDED.updated_at,
                    provenance_hash = EXCLUDED.provenance_hash,
                    metadata = EXCLUDED.metadata
            """,
                agent_id,
                state.get("agent_type", ""),
                json.dumps(state),
                state.get("status", "idle"),
                now,
                now,
                now,
                provenance_hash,
                json.dumps(state.get("metadata", {}))
            )

        logger.debug(f"Saved agent state for {agent_id} to PostgreSQL")

    async def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve agent state from PostgreSQL.

        Args:
            agent_id: Unique agent identifier

        Returns:
            State dictionary if found, None otherwise
        """
        await self.initialize()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM agent_states WHERE agent_id = $1",
                agent_id
            )

            if row is None:
                return None

            state = json.loads(row["state"]) if isinstance(row["state"], str) else row["state"]
            metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]

            return {
                "agent_id": row["agent_id"],
                "agent_type": row["agent_type"],
                "state": state,
                "status": row["status"],
                "last_activity_at": row["last_activity_at"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "provenance_hash": row["provenance_hash"],
                "metadata": metadata
            }

    async def list_agent_states(self) -> List[Dict[str, Any]]:
        """
        List all agent states from PostgreSQL.

        Returns:
            List of state dictionaries
        """
        await self.initialize()

        states = []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM agent_states ORDER BY updated_at DESC"
            )

            for row in rows:
                state = json.loads(row["state"]) if isinstance(row["state"], str) else row["state"]
                metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]

                states.append({
                    "agent_id": row["agent_id"],
                    "agent_type": row["agent_type"],
                    "state": state,
                    "status": row["status"],
                    "last_activity_at": row["last_activity_at"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "provenance_hash": row["provenance_hash"],
                    "metadata": metadata
                })

        logger.debug(f"Listed {len(states)} agent states from PostgreSQL")
        return states

    async def delete_agent_state(self, agent_id: str) -> bool:
        """
        Delete agent state from PostgreSQL.

        Args:
            agent_id: Unique agent identifier

        Returns:
            True if deleted, False if not found
        """
        await self.initialize()

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM agent_states WHERE agent_id = $1",
                agent_id
            )
            deleted = result.split()[-1] != "0"

        if deleted:
            logger.debug(f"Deleted agent state for {agent_id} from PostgreSQL")
        return deleted

    async def save_calculation_result(
        self,
        job_id: str,
        result: Dict[str, Any]
    ) -> None:
        """
        Save calculation result to PostgreSQL.

        Args:
            job_id: Unique job identifier
            result: Result data to persist
        """
        await self.initialize()

        provenance_hash = _calculate_provenance_hash({
            "job_id": job_id,
            "result": result
        })

        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO calculation_results (
                    job_id, agent_id, calculation_type, inputs, result,
                    status, error_message, started_at, completed_at,
                    created_at, provenance_hash, processing_time_ms
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW(), $10, $11)
                ON CONFLICT (job_id) DO UPDATE SET
                    result = EXCLUDED.result,
                    status = EXCLUDED.status,
                    error_message = EXCLUDED.error_message,
                    completed_at = EXCLUDED.completed_at,
                    provenance_hash = EXCLUDED.provenance_hash,
                    processing_time_ms = EXCLUDED.processing_time_ms
            """,
                job_id,
                result.get("agent_id", ""),
                result.get("calculation_type", ""),
                json.dumps(result.get("inputs", {})),
                json.dumps(result.get("result", {})),
                result.get("status", "pending"),
                result.get("error_message"),
                result.get("started_at"),
                result.get("completed_at"),
                provenance_hash,
                result.get("processing_time_ms")
            )

        logger.debug(f"Saved calculation result for job {job_id} to PostgreSQL")

    async def get_calculation_result(
        self,
        job_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve calculation result from PostgreSQL.

        Args:
            job_id: Unique job identifier

        Returns:
            Result dictionary if found, None otherwise
        """
        await self.initialize()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM calculation_results WHERE job_id = $1",
                job_id
            )

            if row is None:
                return None

            inputs = json.loads(row["inputs"]) if isinstance(row["inputs"], str) else row["inputs"]
            result = json.loads(row["result"]) if isinstance(row["result"], str) else row["result"]

            return {
                "job_id": row["job_id"],
                "agent_id": row["agent_id"],
                "calculation_type": row["calculation_type"],
                "inputs": inputs,
                "result": result,
                "status": row["status"],
                "error_message": row["error_message"],
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
                "created_at": row["created_at"],
                "provenance_hash": row["provenance_hash"],
                "processing_time_ms": row["processing_time_ms"]
            }

    async def list_calculation_results(
        self,
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List calculation results from PostgreSQL with optional filtering.

        Args:
            agent_id: Filter by agent ID
            status: Filter by status
            limit: Maximum results to return
            offset: Number of results to skip

        Returns:
            List of result dictionaries
        """
        await self.initialize()

        results = []
        async with self._pool.acquire() as conn:
            # Build query with filters
            query = "SELECT * FROM calculation_results WHERE 1=1"
            params = []
            param_num = 1

            if agent_id:
                query += f" AND agent_id = ${param_num}"
                params.append(agent_id)
                param_num += 1
            if status:
                query += f" AND status = ${param_num}"
                params.append(status)
                param_num += 1

            query += f" ORDER BY created_at DESC LIMIT ${param_num} OFFSET ${param_num + 1}"
            params.extend([limit, offset])

            rows = await conn.fetch(query, *params)

            for row in rows:
                inputs = json.loads(row["inputs"]) if isinstance(row["inputs"], str) else row["inputs"]
                result = json.loads(row["result"]) if isinstance(row["result"], str) else row["result"]

                results.append({
                    "job_id": row["job_id"],
                    "agent_id": row["agent_id"],
                    "calculation_type": row["calculation_type"],
                    "inputs": inputs,
                    "result": result,
                    "status": row["status"],
                    "error_message": row["error_message"],
                    "started_at": row["started_at"],
                    "completed_at": row["completed_at"],
                    "created_at": row["created_at"],
                    "provenance_hash": row["provenance_hash"],
                    "processing_time_ms": row["processing_time_ms"]
                })

        logger.debug(f"Listed {len(results)} calculation results from PostgreSQL")
        return results

    async def close(self) -> None:
        """Close the connection pool."""
        async with self._lock:
            if self._pool:
                await self._pool.close()
                self._pool = None
            self._initialized = False
            logger.info("PostgresAgentStateStore closed")


# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------

__all__ = [
    "AgentState",
    "CalculationResult",
    "AgentStateStore",
    "AgentStateStoreConfig",
    "BaseAgentStateStore",
    "InMemoryAgentStateStore",
    "SQLiteAgentStateStore",
    "PostgresAgentStateStore",
]
