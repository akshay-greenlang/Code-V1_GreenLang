"""
CheckpointManager - Pipeline checkpointing and recovery system

This module implements checkpointing capabilities for GreenLang pipelines,
allowing them to save state between stages and resume after failures.

Example:
    >>> from greenlang.pipeline.checkpointing import CheckpointManager
    >>> manager = CheckpointManager(strategy="file", base_path="/tmp/checkpoints")
    >>> manager.save_checkpoint(pipeline_id, stage_name, state_data)
    >>> restored = manager.load_checkpoint(pipeline_id)
"""

import json
import hashlib
import pickle
import sqlite3
import logging
from typing import Dict, List, Optional, Any, Union, Literal
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import Json
import redis
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class CheckpointStrategy(str, Enum):
    """Supported checkpoint storage strategies."""
    FILE = "file"
    DATABASE = "database"
    REDIS = "redis"
    MEMORY = "memory"
    SQLITE = "sqlite"


class CheckpointStatus(str, Enum):
    """Checkpoint status indicators."""
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RESUMED = "resumed"
    EXPIRED = "expired"


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    pipeline_id: str
    stage_name: str
    stage_index: int
    timestamp: datetime
    status: CheckpointStatus
    checksum: str
    data_size: int
    error_message: Optional[str] = None
    resume_count: int = 0
    parent_checkpoint_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['status'] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['status'] = CheckpointStatus(data['status'])
        return cls(**data)


@dataclass
class PipelineCheckpoint:
    """Complete checkpoint for a pipeline execution."""
    metadata: CheckpointMetadata
    state_data: Dict[str, Any]
    completed_stages: List[str]
    pending_stages: List[str]
    agent_outputs: Dict[str, Any]
    provenance_hashes: Dict[str, str]

    def calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum for checkpoint integrity."""
        checkpoint_str = json.dumps({
            'state_data': self.state_data,
            'completed_stages': self.completed_stages,
            'agent_outputs': str(self.agent_outputs)
        }, sort_keys=True)
        return hashlib.sha256(checkpoint_str.encode()).hexdigest()


class CheckpointStorage(ABC):
    """Abstract base class for checkpoint storage backends."""

    @abstractmethod
    def save(self, checkpoint_id: str, checkpoint: PipelineCheckpoint) -> bool:
        """Save checkpoint to storage."""
        pass

    @abstractmethod
    def load(self, checkpoint_id: str) -> Optional[PipelineCheckpoint]:
        """Load checkpoint from storage."""
        pass

    @abstractmethod
    def exists(self, checkpoint_id: str) -> bool:
        """Check if checkpoint exists."""
        pass

    @abstractmethod
    def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        pass

    @abstractmethod
    def list_checkpoints(self, pipeline_id: Optional[str] = None) -> List[CheckpointMetadata]:
        """List available checkpoints."""
        pass

    @abstractmethod
    def cleanup_old(self, retention_days: int) -> int:
        """Clean up checkpoints older than retention period."""
        pass


class FileCheckpointStorage(CheckpointStorage):
    """File-based checkpoint storage using JSON."""

    def __init__(self, base_path: Union[str, Path]):
        """Initialize file storage."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized file checkpoint storage at {self.base_path}")

    def _get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get path for checkpoint file."""
        return self.base_path / f"{checkpoint_id}.checkpoint.json"

    def save(self, checkpoint_id: str, checkpoint: PipelineCheckpoint) -> bool:
        """Save checkpoint to JSON file."""
        try:
            checkpoint_path = self._get_checkpoint_path(checkpoint_id)

            # Prepare data for serialization
            data = {
                'metadata': checkpoint.metadata.to_dict(),
                'state_data': checkpoint.state_data,
                'completed_stages': checkpoint.completed_stages,
                'pending_stages': checkpoint.pending_stages,
                'agent_outputs': checkpoint.agent_outputs,
                'provenance_hashes': checkpoint.provenance_hashes
            }

            # Write atomically using temp file
            temp_path = checkpoint_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            # Move to final location
            temp_path.replace(checkpoint_path)

            logger.info(f"Saved checkpoint {checkpoint_id} to {checkpoint_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint_id}: {str(e)}")
            return False

    def load(self, checkpoint_id: str) -> Optional[PipelineCheckpoint]:
        """Load checkpoint from JSON file."""
        try:
            checkpoint_path = self._get_checkpoint_path(checkpoint_id)

            if not checkpoint_path.exists():
                return None

            with open(checkpoint_path, 'r') as f:
                data = json.load(f)

            # Reconstruct checkpoint
            metadata = CheckpointMetadata.from_dict(data['metadata'])
            checkpoint = PipelineCheckpoint(
                metadata=metadata,
                state_data=data['state_data'],
                completed_stages=data['completed_stages'],
                pending_stages=data['pending_stages'],
                agent_outputs=data['agent_outputs'],
                provenance_hashes=data['provenance_hashes']
            )

            # Verify checksum
            expected_checksum = checkpoint.calculate_checksum()
            if metadata.checksum != expected_checksum:
                logger.warning(f"Checksum mismatch for checkpoint {checkpoint_id}")

            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {str(e)}")
            return None

    def exists(self, checkpoint_id: str) -> bool:
        """Check if checkpoint file exists."""
        return self._get_checkpoint_path(checkpoint_id).exists()

    def delete(self, checkpoint_id: str) -> bool:
        """Delete checkpoint file."""
        try:
            checkpoint_path = self._get_checkpoint_path(checkpoint_id)
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {str(e)}")
            return False

    def list_checkpoints(self, pipeline_id: Optional[str] = None) -> List[CheckpointMetadata]:
        """List checkpoints in directory."""
        checkpoints = []
        pattern = f"{pipeline_id}*.checkpoint.json" if pipeline_id else "*.checkpoint.json"

        for checkpoint_file in self.base_path.glob(pattern):
            try:
                checkpoint = self.load(checkpoint_file.stem.replace('.checkpoint', ''))
                if checkpoint:
                    checkpoints.append(checkpoint.metadata)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint metadata from {checkpoint_file}: {e}")

        return sorted(checkpoints, key=lambda x: x.timestamp, reverse=True)

    def cleanup_old(self, retention_days: int) -> int:
        """Clean up old checkpoint files."""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        deleted_count = 0

        for checkpoint_file in self.base_path.glob("*.checkpoint.json"):
            try:
                # Check file modification time
                mtime = datetime.fromtimestamp(checkpoint_file.stat().st_mtime)
                if mtime < cutoff_date:
                    checkpoint_file.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted old checkpoint: {checkpoint_file}")
            except Exception as e:
                logger.error(f"Failed to delete {checkpoint_file}: {e}")

        return deleted_count


class DatabaseCheckpointStorage(CheckpointStorage):
    """PostgreSQL-based checkpoint storage."""

    def __init__(self, connection_string: str):
        """Initialize database storage."""
        self.connection_string = connection_string
        self._create_tables()
        logger.info("Initialized database checkpoint storage")

    @contextmanager
    def _get_connection(self):
        """Get database connection context manager."""
        conn = psycopg2.connect(self.connection_string)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _create_tables(self):
        """Create checkpoint tables if they don't exist."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS pipeline_checkpoints (
                        checkpoint_id VARCHAR(255) PRIMARY KEY,
                        pipeline_id VARCHAR(255) NOT NULL,
                        stage_name VARCHAR(255) NOT NULL,
                        stage_index INTEGER NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        status VARCHAR(50) NOT NULL,
                        checksum VARCHAR(64) NOT NULL,
                        data_size INTEGER NOT NULL,
                        error_message TEXT,
                        resume_count INTEGER DEFAULT 0,
                        parent_checkpoint_id VARCHAR(255),
                        state_data JSONB NOT NULL,
                        completed_stages JSONB NOT NULL,
                        pending_stages JSONB NOT NULL,
                        agent_outputs JSONB NOT NULL,
                        provenance_hashes JSONB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_pipeline_id (pipeline_id),
                        INDEX idx_timestamp (timestamp)
                    )
                """)

    def save(self, checkpoint_id: str, checkpoint: PipelineCheckpoint) -> bool:
        """Save checkpoint to database."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO pipeline_checkpoints (
                            checkpoint_id, pipeline_id, stage_name, stage_index,
                            timestamp, status, checksum, data_size, error_message,
                            resume_count, parent_checkpoint_id, state_data,
                            completed_stages, pending_stages, agent_outputs,
                            provenance_hashes
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (checkpoint_id) DO UPDATE SET
                            timestamp = EXCLUDED.timestamp,
                            status = EXCLUDED.status,
                            state_data = EXCLUDED.state_data,
                            completed_stages = EXCLUDED.completed_stages,
                            pending_stages = EXCLUDED.pending_stages,
                            agent_outputs = EXCLUDED.agent_outputs
                    """, (
                        checkpoint_id,
                        checkpoint.metadata.pipeline_id,
                        checkpoint.metadata.stage_name,
                        checkpoint.metadata.stage_index,
                        checkpoint.metadata.timestamp,
                        checkpoint.metadata.status.value,
                        checkpoint.metadata.checksum,
                        checkpoint.metadata.data_size,
                        checkpoint.metadata.error_message,
                        checkpoint.metadata.resume_count,
                        checkpoint.metadata.parent_checkpoint_id,
                        Json(checkpoint.state_data),
                        Json(checkpoint.completed_stages),
                        Json(checkpoint.pending_stages),
                        Json(checkpoint.agent_outputs),
                        Json(checkpoint.provenance_hashes)
                    ))

            logger.info(f"Saved checkpoint {checkpoint_id} to database")
            return True

        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint_id} to database: {str(e)}")
            return False

    def load(self, checkpoint_id: str) -> Optional[PipelineCheckpoint]:
        """Load checkpoint from database."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT * FROM pipeline_checkpoints
                        WHERE checkpoint_id = %s
                    """, (checkpoint_id,))

                    row = cur.fetchone()
                    if not row:
                        return None

                    # Reconstruct checkpoint
                    metadata = CheckpointMetadata(
                        pipeline_id=row[1],
                        stage_name=row[2],
                        stage_index=row[3],
                        timestamp=row[4],
                        status=CheckpointStatus(row[5]),
                        checksum=row[6],
                        data_size=row[7],
                        error_message=row[8],
                        resume_count=row[9],
                        parent_checkpoint_id=row[10]
                    )

                    checkpoint = PipelineCheckpoint(
                        metadata=metadata,
                        state_data=row[11],
                        completed_stages=row[12],
                        pending_stages=row[13],
                        agent_outputs=row[14],
                        provenance_hashes=row[15]
                    )

                    return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id} from database: {str(e)}")
            return None

    def exists(self, checkpoint_id: str) -> bool:
        """Check if checkpoint exists in database."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 1 FROM pipeline_checkpoints
                        WHERE checkpoint_id = %s
                    """, (checkpoint_id,))
                    return cur.fetchone() is not None
        except Exception:
            return False

    def delete(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from database."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        DELETE FROM pipeline_checkpoints
                        WHERE checkpoint_id = %s
                    """, (checkpoint_id,))
                    return cur.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {str(e)}")
            return False

    def list_checkpoints(self, pipeline_id: Optional[str] = None) -> List[CheckpointMetadata]:
        """List checkpoints from database."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    if pipeline_id:
                        cur.execute("""
                            SELECT pipeline_id, stage_name, stage_index, timestamp,
                                   status, checksum, data_size, error_message,
                                   resume_count, parent_checkpoint_id
                            FROM pipeline_checkpoints
                            WHERE pipeline_id = %s
                            ORDER BY timestamp DESC
                        """, (pipeline_id,))
                    else:
                        cur.execute("""
                            SELECT pipeline_id, stage_name, stage_index, timestamp,
                                   status, checksum, data_size, error_message,
                                   resume_count, parent_checkpoint_id
                            FROM pipeline_checkpoints
                            ORDER BY timestamp DESC
                        """)

                    checkpoints = []
                    for row in cur.fetchall():
                        metadata = CheckpointMetadata(
                            pipeline_id=row[0],
                            stage_name=row[1],
                            stage_index=row[2],
                            timestamp=row[3],
                            status=CheckpointStatus(row[4]),
                            checksum=row[5],
                            data_size=row[6],
                            error_message=row[7],
                            resume_count=row[8],
                            parent_checkpoint_id=row[9]
                        )
                        checkpoints.append(metadata)

                    return checkpoints

        except Exception as e:
            logger.error(f"Failed to list checkpoints: {str(e)}")
            return []

    def cleanup_old(self, retention_days: int) -> int:
        """Clean up old checkpoints from database."""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        DELETE FROM pipeline_checkpoints
                        WHERE timestamp < %s
                    """, (cutoff_date,))
                    return cur.rowcount
        except Exception as e:
            logger.error(f"Failed to cleanup old checkpoints: {str(e)}")
            return 0


class RedisCheckpointStorage(CheckpointStorage):
    """Redis-based checkpoint storage for fast in-memory access."""

    def __init__(self, host: str = 'localhost', port: int = 6379,
                 db: int = 0, password: Optional[str] = None,
                 ttl_seconds: int = 86400):  # 24 hours default TTL
        """Initialize Redis storage."""
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False  # Use binary for pickle
        )
        self.ttl_seconds = ttl_seconds
        logger.info(f"Initialized Redis checkpoint storage at {host}:{port}")

    def _get_checkpoint_key(self, checkpoint_id: str) -> str:
        """Get Redis key for checkpoint."""
        return f"checkpoint:{checkpoint_id}"

    def _get_metadata_key(self, checkpoint_id: str) -> str:
        """Get Redis key for checkpoint metadata."""
        return f"checkpoint:metadata:{checkpoint_id}"

    def save(self, checkpoint_id: str, checkpoint: PipelineCheckpoint) -> bool:
        """Save checkpoint to Redis."""
        try:
            checkpoint_key = self._get_checkpoint_key(checkpoint_id)
            metadata_key = self._get_metadata_key(checkpoint_id)

            # Serialize checkpoint using pickle for efficiency
            checkpoint_data = pickle.dumps(checkpoint)
            metadata_data = json.dumps(checkpoint.metadata.to_dict())

            # Save with TTL
            pipe = self.redis_client.pipeline()
            pipe.setex(checkpoint_key, self.ttl_seconds, checkpoint_data)
            pipe.setex(metadata_key, self.ttl_seconds, metadata_data)

            # Add to pipeline index
            pipe.sadd(f"pipelines:{checkpoint.metadata.pipeline_id}", checkpoint_id)
            pipe.expire(f"pipelines:{checkpoint.metadata.pipeline_id}", self.ttl_seconds)

            pipe.execute()

            logger.info(f"Saved checkpoint {checkpoint_id} to Redis with TTL {self.ttl_seconds}s")
            return True

        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint_id} to Redis: {str(e)}")
            return False

    def load(self, checkpoint_id: str) -> Optional[PipelineCheckpoint]:
        """Load checkpoint from Redis."""
        try:
            checkpoint_key = self._get_checkpoint_key(checkpoint_id)
            checkpoint_data = self.redis_client.get(checkpoint_key)

            if not checkpoint_data:
                return None

            checkpoint = pickle.loads(checkpoint_data)

            # Refresh TTL on access
            self.redis_client.expire(checkpoint_key, self.ttl_seconds)

            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id} from Redis: {str(e)}")
            return None

    def exists(self, checkpoint_id: str) -> bool:
        """Check if checkpoint exists in Redis."""
        return bool(self.redis_client.exists(self._get_checkpoint_key(checkpoint_id)))

    def delete(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from Redis."""
        try:
            checkpoint_key = self._get_checkpoint_key(checkpoint_id)
            metadata_key = self._get_metadata_key(checkpoint_id)

            # Get pipeline ID before deletion
            checkpoint = self.load(checkpoint_id)

            pipe = self.redis_client.pipeline()
            pipe.delete(checkpoint_key)
            pipe.delete(metadata_key)

            # Remove from pipeline index
            if checkpoint:
                pipe.srem(f"pipelines:{checkpoint.metadata.pipeline_id}", checkpoint_id)

            results = pipe.execute()
            return any(results)

        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {str(e)}")
            return False

    def list_checkpoints(self, pipeline_id: Optional[str] = None) -> List[CheckpointMetadata]:
        """List checkpoints from Redis."""
        try:
            checkpoints = []

            if pipeline_id:
                # Get checkpoints for specific pipeline
                checkpoint_ids = self.redis_client.smembers(f"pipelines:{pipeline_id}")
            else:
                # Get all checkpoint metadata keys
                checkpoint_ids = []
                for key in self.redis_client.scan_iter("checkpoint:metadata:*"):
                    checkpoint_id = key.decode().replace("checkpoint:metadata:", "")
                    checkpoint_ids.append(checkpoint_id)

            for checkpoint_id in checkpoint_ids:
                metadata_key = self._get_metadata_key(checkpoint_id)
                metadata_data = self.redis_client.get(metadata_key)

                if metadata_data:
                    metadata_dict = json.loads(metadata_data)
                    metadata = CheckpointMetadata.from_dict(metadata_dict)
                    checkpoints.append(metadata)

            return sorted(checkpoints, key=lambda x: x.timestamp, reverse=True)

        except Exception as e:
            logger.error(f"Failed to list checkpoints: {str(e)}")
            return []

    def cleanup_old(self, retention_days: int) -> int:
        """Clean up old checkpoints (Redis handles this via TTL)."""
        # Redis automatically expires keys based on TTL
        # This method returns 0 as cleanup is automatic
        logger.info("Redis handles cleanup automatically via TTL")
        return 0


class MemoryCheckpointStorage(CheckpointStorage):
    """In-memory checkpoint storage for testing and development."""

    def __init__(self):
        """Initialize memory storage."""
        self.checkpoints: Dict[str, PipelineCheckpoint] = {}
        logger.info("Initialized in-memory checkpoint storage")

    def save(self, checkpoint_id: str, checkpoint: PipelineCheckpoint) -> bool:
        """Save checkpoint to memory."""
        try:
            self.checkpoints[checkpoint_id] = checkpoint
            logger.info(f"Saved checkpoint {checkpoint_id} to memory")
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint_id}: {str(e)}")
            return False

    def load(self, checkpoint_id: str) -> Optional[PipelineCheckpoint]:
        """Load checkpoint from memory."""
        return self.checkpoints.get(checkpoint_id)

    def exists(self, checkpoint_id: str) -> bool:
        """Check if checkpoint exists in memory."""
        return checkpoint_id in self.checkpoints

    def delete(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from memory."""
        if checkpoint_id in self.checkpoints:
            del self.checkpoints[checkpoint_id]
            return True
        return False

    def list_checkpoints(self, pipeline_id: Optional[str] = None) -> List[CheckpointMetadata]:
        """List checkpoints in memory."""
        checkpoints = []
        for checkpoint in self.checkpoints.values():
            if pipeline_id is None or checkpoint.metadata.pipeline_id == pipeline_id:
                checkpoints.append(checkpoint.metadata)
        return sorted(checkpoints, key=lambda x: x.timestamp, reverse=True)

    def cleanup_old(self, retention_days: int) -> int:
        """Clean up old checkpoints from memory."""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        to_delete = []

        for checkpoint_id, checkpoint in self.checkpoints.items():
            if checkpoint.metadata.timestamp < cutoff_date:
                to_delete.append(checkpoint_id)

        for checkpoint_id in to_delete:
            del self.checkpoints[checkpoint_id]

        return len(to_delete)


class CheckpointManager:
    """
    Main checkpoint manager for pipeline execution.

    Handles checkpoint creation, recovery, and lifecycle management.
    """

    def __init__(self,
                 strategy: Union[CheckpointStrategy, str] = CheckpointStrategy.FILE,
                 **storage_kwargs):
        """
        Initialize CheckpointManager.

        Args:
            strategy: Storage strategy to use
            **storage_kwargs: Strategy-specific configuration
        """
        self.strategy = CheckpointStrategy(strategy) if isinstance(strategy, str) else strategy
        self.storage = self._create_storage(**storage_kwargs)
        self.active_checkpoints: Dict[str, str] = {}  # pipeline_id -> checkpoint_id
        logger.info(f"Initialized CheckpointManager with {self.strategy.value} strategy")

    def _create_storage(self, **kwargs) -> CheckpointStorage:
        """Create storage backend based on strategy."""
        if self.strategy == CheckpointStrategy.FILE:
            base_path = kwargs.get('base_path', '/tmp/greenlang/checkpoints')
            return FileCheckpointStorage(base_path)

        elif self.strategy == CheckpointStrategy.DATABASE:
            connection_string = kwargs.get('connection_string')
            if not connection_string:
                raise ValueError("Database strategy requires connection_string")
            return DatabaseCheckpointStorage(connection_string)

        elif self.strategy == CheckpointStrategy.REDIS:
            return RedisCheckpointStorage(
                host=kwargs.get('host', 'localhost'),
                port=kwargs.get('port', 6379),
                db=kwargs.get('db', 0),
                password=kwargs.get('password'),
                ttl_seconds=kwargs.get('ttl_seconds', 86400)
            )

        elif self.strategy == CheckpointStrategy.MEMORY:
            return MemoryCheckpointStorage()

        elif self.strategy == CheckpointStrategy.SQLITE:
            # SQLite implementation (simplified database storage)
            db_path = kwargs.get('db_path', '/tmp/greenlang/checkpoints.db')
            connection_string = f"sqlite:///{db_path}"
            return DatabaseCheckpointStorage(connection_string)

        else:
            raise ValueError(f"Unknown checkpoint strategy: {self.strategy}")

    def create_checkpoint(self,
                         pipeline_id: str,
                         stage_name: str,
                         stage_index: int,
                         state_data: Dict[str, Any],
                         completed_stages: List[str],
                         pending_stages: List[str],
                         agent_outputs: Dict[str, Any],
                         provenance_hashes: Optional[Dict[str, str]] = None) -> str:
        """
        Create a new checkpoint.

        Args:
            pipeline_id: Unique pipeline identifier
            stage_name: Current stage name
            stage_index: Current stage index
            state_data: Pipeline state to preserve
            completed_stages: List of completed stage names
            pending_stages: List of pending stage names
            agent_outputs: Outputs from completed agents
            provenance_hashes: Provenance tracking hashes

        Returns:
            Checkpoint ID
        """
        # Generate checkpoint ID
        checkpoint_id = f"{pipeline_id}_{stage_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Calculate data size
        data_size = len(json.dumps({
            'state_data': state_data,
            'agent_outputs': agent_outputs
        }, default=str))

        # Create checkpoint
        checkpoint = PipelineCheckpoint(
            metadata=CheckpointMetadata(
                pipeline_id=pipeline_id,
                stage_name=stage_name,
                stage_index=stage_index,
                timestamp=datetime.now(),
                status=CheckpointStatus.IN_PROGRESS,
                checksum="",  # Will be calculated
                data_size=data_size,
                parent_checkpoint_id=self.active_checkpoints.get(pipeline_id)
            ),
            state_data=state_data,
            completed_stages=completed_stages,
            pending_stages=pending_stages,
            agent_outputs=agent_outputs,
            provenance_hashes=provenance_hashes or {}
        )

        # Calculate and set checksum
        checkpoint.metadata.checksum = checkpoint.calculate_checksum()

        # Save checkpoint
        if self.storage.save(checkpoint_id, checkpoint):
            self.active_checkpoints[pipeline_id] = checkpoint_id
            logger.info(f"Created checkpoint {checkpoint_id} for pipeline {pipeline_id} at stage {stage_name}")
            return checkpoint_id
        else:
            raise RuntimeError(f"Failed to create checkpoint for pipeline {pipeline_id}")

    def update_checkpoint_status(self,
                                checkpoint_id: str,
                                status: CheckpointStatus,
                                error_message: Optional[str] = None) -> bool:
        """
        Update checkpoint status.

        Args:
            checkpoint_id: Checkpoint to update
            status: New status
            error_message: Optional error message

        Returns:
            Success indicator
        """
        checkpoint = self.storage.load(checkpoint_id)
        if checkpoint:
            checkpoint.metadata.status = status
            if error_message:
                checkpoint.metadata.error_message = error_message
            return self.storage.save(checkpoint_id, checkpoint)
        return False

    def load_latest_checkpoint(self, pipeline_id: str) -> Optional[PipelineCheckpoint]:
        """
        Load the most recent checkpoint for a pipeline.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            Latest checkpoint or None
        """
        checkpoints = self.storage.list_checkpoints(pipeline_id)

        # Find latest valid checkpoint
        for metadata in checkpoints:
            if metadata.status in [CheckpointStatus.COMPLETED, CheckpointStatus.IN_PROGRESS]:
                checkpoint_id = f"{metadata.pipeline_id}_{metadata.stage_name}_{metadata.timestamp.strftime('%Y%m%d_%H%M%S')}"
                checkpoint = self.storage.load(checkpoint_id)
                if checkpoint:
                    logger.info(f"Loaded checkpoint {checkpoint_id} for pipeline {pipeline_id}")
                    return checkpoint

        return None

    def resume_pipeline(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """
        Resume a pipeline from its latest checkpoint.

        Args:
            pipeline_id: Pipeline to resume

        Returns:
            Resume context with checkpoint data
        """
        checkpoint = self.load_latest_checkpoint(pipeline_id)

        if not checkpoint:
            logger.warning(f"No checkpoint found for pipeline {pipeline_id}")
            return None

        # Update resume count
        checkpoint.metadata.resume_count += 1
        checkpoint.metadata.status = CheckpointStatus.RESUMED

        # Save updated checkpoint
        checkpoint_id = f"{checkpoint.metadata.pipeline_id}_{checkpoint.metadata.stage_name}_{checkpoint.metadata.timestamp.strftime('%Y%m%d_%H%M%S')}"
        self.storage.save(checkpoint_id, checkpoint)

        # Prepare resume context
        resume_context = {
            'checkpoint_id': checkpoint_id,
            'pipeline_id': pipeline_id,
            'resume_from_stage': checkpoint.metadata.stage_name,
            'resume_from_index': checkpoint.metadata.stage_index,
            'completed_stages': checkpoint.completed_stages,
            'pending_stages': checkpoint.pending_stages,
            'state_data': checkpoint.state_data,
            'agent_outputs': checkpoint.agent_outputs,
            'provenance_hashes': checkpoint.provenance_hashes,
            'resume_count': checkpoint.metadata.resume_count
        }

        logger.info(f"Resuming pipeline {pipeline_id} from stage {checkpoint.metadata.stage_name} (attempt #{checkpoint.metadata.resume_count})")

        return resume_context

    def get_checkpoint_history(self, pipeline_id: str) -> List[CheckpointMetadata]:
        """
        Get checkpoint history for a pipeline.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            List of checkpoint metadata
        """
        return self.storage.list_checkpoints(pipeline_id)

    def visualize_checkpoint_chain(self, pipeline_id: str) -> str:
        """
        Create a visual representation of checkpoint chain.

        Args:
            pipeline_id: Pipeline to visualize

        Returns:
            ASCII art visualization
        """
        checkpoints = self.get_checkpoint_history(pipeline_id)

        if not checkpoints:
            return f"No checkpoints found for pipeline {pipeline_id}"

        lines = [f"Checkpoint Chain for Pipeline: {pipeline_id}"]
        lines.append("=" * 60)

        for i, checkpoint in enumerate(checkpoints):
            indent = "  " * i
            status_symbol = {
                CheckpointStatus.COMPLETED: "✓",
                CheckpointStatus.IN_PROGRESS: "⟳",
                CheckpointStatus.FAILED: "✗",
                CheckpointStatus.RESUMED: "↺",
                CheckpointStatus.EXPIRED: "⌛"
            }.get(checkpoint.status, "?")

            lines.append(
                f"{indent}{status_symbol} {checkpoint.stage_name} "
                f"[{checkpoint.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"({checkpoint.data_size} bytes)"
            )

            if checkpoint.error_message:
                lines.append(f"{indent}  └─ Error: {checkpoint.error_message[:50]}...")

            if checkpoint.resume_count > 0:
                lines.append(f"{indent}  └─ Resumed {checkpoint.resume_count} time(s)")

        return "\n".join(lines)

    def cleanup_completed_pipelines(self, pipeline_ids: List[str]) -> int:
        """
        Clean up checkpoints for completed pipelines.

        Args:
            pipeline_ids: List of completed pipeline IDs

        Returns:
            Number of checkpoints deleted
        """
        deleted_count = 0

        for pipeline_id in pipeline_ids:
            checkpoints = self.storage.list_checkpoints(pipeline_id)
            for checkpoint_meta in checkpoints:
                checkpoint_id = f"{checkpoint_meta.pipeline_id}_{checkpoint_meta.stage_name}_{checkpoint_meta.timestamp.strftime('%Y%m%d_%H%M%S')}"
                if self.storage.delete(checkpoint_id):
                    deleted_count += 1

            # Remove from active checkpoints
            self.active_checkpoints.pop(pipeline_id, None)

        logger.info(f"Cleaned up {deleted_count} checkpoints for {len(pipeline_ids)} pipelines")
        return deleted_count

    def auto_cleanup(self, retention_days: int = 7) -> int:
        """
        Automatically clean up old checkpoints.

        Args:
            retention_days: Days to retain checkpoints

        Returns:
            Number of checkpoints cleaned up
        """
        count = self.storage.cleanup_old(retention_days)
        logger.info(f"Auto-cleanup removed {count} checkpoints older than {retention_days} days")
        return count

    def export_checkpoint(self, checkpoint_id: str, export_path: Path) -> bool:
        """
        Export checkpoint to file for backup/transfer.

        Args:
            checkpoint_id: Checkpoint to export
            export_path: Path to export file

        Returns:
            Success indicator
        """
        try:
            checkpoint = self.storage.load(checkpoint_id)
            if not checkpoint:
                return False

            export_data = {
                'checkpoint_id': checkpoint_id,
                'metadata': checkpoint.metadata.to_dict(),
                'checkpoint_data': {
                    'state_data': checkpoint.state_data,
                    'completed_stages': checkpoint.completed_stages,
                    'pending_stages': checkpoint.pending_stages,
                    'agent_outputs': checkpoint.agent_outputs,
                    'provenance_hashes': checkpoint.provenance_hashes
                },
                'export_timestamp': datetime.now().isoformat(),
                'export_version': '1.0'
            }

            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Exported checkpoint {checkpoint_id} to {export_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export checkpoint: {str(e)}")
            return False

    def import_checkpoint(self, import_path: Path) -> Optional[str]:
        """
        Import checkpoint from file.

        Args:
            import_path: Path to import file

        Returns:
            Imported checkpoint ID or None
        """
        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)

            # Reconstruct checkpoint
            metadata = CheckpointMetadata.from_dict(import_data['metadata'])
            checkpoint_data = import_data['checkpoint_data']

            checkpoint = PipelineCheckpoint(
                metadata=metadata,
                state_data=checkpoint_data['state_data'],
                completed_stages=checkpoint_data['completed_stages'],
                pending_stages=checkpoint_data['pending_stages'],
                agent_outputs=checkpoint_data['agent_outputs'],
                provenance_hashes=checkpoint_data['provenance_hashes']
            )

            checkpoint_id = import_data['checkpoint_id']
            if self.storage.save(checkpoint_id, checkpoint):
                logger.info(f"Imported checkpoint {checkpoint_id} from {import_path}")
                return checkpoint_id

            return None

        except Exception as e:
            logger.error(f"Failed to import checkpoint: {str(e)}")
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get checkpoint statistics.

        Returns:
            Dictionary with checkpoint statistics
        """
        all_checkpoints = self.storage.list_checkpoints()

        stats = {
            'total_checkpoints': len(all_checkpoints),
            'active_pipelines': len(self.active_checkpoints),
            'storage_strategy': self.strategy.value,
            'checkpoints_by_status': {},
            'average_data_size': 0,
            'total_resume_count': 0,
            'oldest_checkpoint': None,
            'newest_checkpoint': None
        }

        if all_checkpoints:
            # Count by status
            for checkpoint in all_checkpoints:
                status = checkpoint.status.value
                stats['checkpoints_by_status'][status] = stats['checkpoints_by_status'].get(status, 0) + 1

            # Calculate averages
            total_size = sum(c.data_size for c in all_checkpoints)
            stats['average_data_size'] = total_size / len(all_checkpoints)
            stats['total_resume_count'] = sum(c.resume_count for c in all_checkpoints)

            # Find oldest and newest
            sorted_checkpoints = sorted(all_checkpoints, key=lambda x: x.timestamp)
            stats['oldest_checkpoint'] = sorted_checkpoints[0].timestamp.isoformat()
            stats['newest_checkpoint'] = sorted_checkpoints[-1].timestamp.isoformat()

        return stats