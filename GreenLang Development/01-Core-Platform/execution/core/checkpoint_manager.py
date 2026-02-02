# -*- coding: utf-8 -*-
"""
Smart Retry from Checkpoint Manager (FR-074)

This module implements checkpoint-based retry for long-running pipeline recovery
in the GL-FOUND-X-001 GreenLang Orchestrator.

Author: GreenLang Team
Version: 1.0.0
GL-FOUND-X-001: Smart Retry from Checkpoint (FR-074)
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_RETENTION_DAYS = 7
DEFAULT_MAX_RETRY_COUNT = 3
CHECKPOINT_HASH_ALGORITHM = "sha256"


class CheckpointStatus(str, Enum):
    """Status of a checkpoint state."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELED = "canceled"


class StepIdempotencyBehavior(str, Enum):
    """Idempotency behavior for step execution."""
    IDEMPOTENT = "idempotent"
    NON_IDEMPOTENT = "non_idempotent"
    UNKNOWN = "unknown"


class CheckpointState(BaseModel):
    """State of a single step checkpoint."""

    run_id: str = Field(..., description="Parent run identifier", min_length=1, max_length=255)
    step_id: str = Field(..., description="Step identifier", min_length=1, max_length=255)
    status: CheckpointStatus = Field(default=CheckpointStatus.PENDING)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[str] = Field(default_factory=list)
    idempotency_key: str = Field(default="")
    attempt: int = Field(default=1, ge=1)
    error_message: Optional[str] = Field(None)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False, "extra": "forbid"}

    @field_validator("created_at", "updated_at", "started_at", "completed_at", mode="before")
    @classmethod
    def ensure_utc(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v

    @staticmethod
    def generate_idempotency_key(plan_hash: str, step_id: str, attempt: int) -> str:
        """Generate idempotency key from plan hash, step ID, and attempt."""
        key_input = f"{plan_hash}:{step_id}:{attempt}"
        return hashlib.sha256(key_input.encode("utf-8")).hexdigest()

    def compute_output_hash(self) -> str:
        output_str = json.dumps(self.outputs, sort_keys=True, default=str)
        return hashlib.sha256(output_str.encode("utf-8")).hexdigest()

    def is_terminal(self) -> bool:
        return self.status in {CheckpointStatus.COMPLETED, CheckpointStatus.FAILED, CheckpointStatus.SKIPPED, CheckpointStatus.CANCELED}

    def is_resumable(self) -> bool:
        return self.status == CheckpointStatus.COMPLETED

    def mark_started(self) -> None:
        self.status = CheckpointStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def mark_completed(self, outputs=None) -> None:
        self.status = CheckpointStatus.COMPLETED
        if outputs:
            self.outputs = outputs
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def mark_failed(self, error_message: str) -> None:
        self.status = CheckpointStatus.FAILED
        self.error_message = error_message
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id, "step_id": self.step_id, "status": self.status.value,
            "outputs": self.outputs, "artifacts": self.artifacts, "idempotency_key": self.idempotency_key,
            "attempt": self.attempt, "error_message": self.error_message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_at": self.created_at.isoformat(), "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data):
        data = data.copy()
        if isinstance(data.get("status"), str):
            data["status"] = CheckpointStatus(data["status"])
        return cls(**data)


class RunCheckpoint(BaseModel):
    """Complete checkpoint state for a pipeline run."""

    run_id: str = Field(..., min_length=1, max_length=255)
    plan_id: str = Field(..., min_length=1, max_length=255)
    plan_hash: str = Field(...)
    pipeline_id: str = Field(default="")
    step_checkpoints: Dict[str, CheckpointState] = Field(default_factory=dict)
    last_successful_step: Optional[str] = Field(None)
    retry_count: int = Field(default=0, ge=0)
    parent_run_id: Optional[str] = Field(None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False, "extra": "forbid"}

    @field_validator("created_at", "updated_at", "expires_at", mode="before")
    @classmethod
    def ensure_utc(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v

    def get_step_checkpoint(self, step_id: str):
        return self.step_checkpoints.get(step_id)

    def set_step_checkpoint(self, step_id: str, state: CheckpointState) -> None:
        self.step_checkpoints[step_id] = state
        self.updated_at = datetime.now(timezone.utc)
        if state.status == CheckpointStatus.COMPLETED:
            self.last_successful_step = step_id

    def get_completed_steps(self) -> List[str]:
        return [s for s, st in self.step_checkpoints.items() if st.status == CheckpointStatus.COMPLETED]

    def get_failed_steps(self) -> List[str]:
        return [s for s, st in self.step_checkpoints.items() if st.status == CheckpointStatus.FAILED]

    def get_pending_steps(self, all_step_ids: List[str]) -> List[str]:
        completed = set(self.get_completed_steps())
        return [s for s in all_step_ids if s not in completed]

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def set_expiration(self, retention_days: int) -> None:
        self.expires_at = datetime.now(timezone.utc) + timedelta(days=retention_days)

    def compute_state_hash(self) -> str:
        state_dict = {"run_id": self.run_id, "plan_id": self.plan_id, "plan_hash": self.plan_hash,
            "step_checkpoints": {k: v.to_dict() for k, v in self.step_checkpoints.items()}, "retry_count": self.retry_count}
        state_str = json.dumps(state_dict, sort_keys=True, default=str)
        return hashlib.sha256(state_str.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id, "plan_id": self.plan_id, "plan_hash": self.plan_hash, "pipeline_id": self.pipeline_id,
            "step_checkpoints": {k: v.to_dict() for k, v in self.step_checkpoints.items()},
            "last_successful_step": self.last_successful_step, "retry_count": self.retry_count,
            "parent_run_id": self.parent_run_id, "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(), "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data):
        data = data.copy()
        if "step_checkpoints" in data:
            data["step_checkpoints"] = {k: CheckpointState.from_dict(v) for k, v in data["step_checkpoints"].items()}
        return cls(**data)


@runtime_checkable
class CheckpointStore(Protocol):
    """Protocol defining the CheckpointStore interface."""
    async def save(self, checkpoint: RunCheckpoint) -> bool: ...
    async def load(self, run_id: str) -> Optional[RunCheckpoint]: ...
    async def delete(self, run_id: str) -> bool: ...
    async def exists(self, run_id: str) -> bool: ...
    async def list_checkpoints(self, pipeline_id=None, limit: int = 100, offset: int = 0) -> List[RunCheckpoint]: ...
    async def cleanup_expired(self) -> int: ...


class InMemoryCheckpointStore:
    """In-memory checkpoint store for testing and development."""

    def __init__(self):
        self._checkpoints: Dict[str, RunCheckpoint] = {}
        logger.info("Initialized InMemoryCheckpointStore")

    async def save(self, checkpoint: RunCheckpoint) -> bool:
        try:
            self._checkpoints[checkpoint.run_id] = checkpoint
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False

    async def load(self, run_id: str):
        checkpoint = self._checkpoints.get(run_id)
        if checkpoint and checkpoint.is_expired():
            await self.delete(run_id)
            return None
        return checkpoint

    async def delete(self, run_id: str) -> bool:
        if run_id in self._checkpoints:
            del self._checkpoints[run_id]
            return True
        return False

    async def exists(self, run_id: str) -> bool:
        checkpoint = self._checkpoints.get(run_id)
        if checkpoint and checkpoint.is_expired():
            await self.delete(run_id)
            return False
        return run_id in self._checkpoints

    async def list_checkpoints(self, pipeline_id=None, limit: int = 100, offset: int = 0):
        checkpoints = list(self._checkpoints.values())
        if pipeline_id:
            checkpoints = [c for c in checkpoints if c.pipeline_id == pipeline_id]
        checkpoints = [c for c in checkpoints if not c.is_expired()]
        checkpoints.sort(key=lambda c: c.created_at, reverse=True)
        return checkpoints[offset:offset + limit]

    async def cleanup_expired(self) -> int:
        expired_ids = [run_id for run_id, checkpoint in self._checkpoints.items() if checkpoint.is_expired()]
        for run_id in expired_ids:
            del self._checkpoints[run_id]
        return len(expired_ids)

    async def clear(self) -> None:
        self._checkpoints.clear()


class CheckpointManager:
    """Manager for checkpoint operations with audit trail integration."""

    def __init__(self, store, event_store=None, retention_days: int = DEFAULT_CHECKPOINT_RETENTION_DAYS, max_retry_count: int = DEFAULT_MAX_RETRY_COUNT):
        self._store = store
        self._event_store = event_store
        self._retention_days = retention_days
        self._max_retry_count = max_retry_count
        logger.info(f"Initialized CheckpointManager: retention_days={retention_days}, max_retry_count={max_retry_count}")

    async def create_run_checkpoint(self, run_id: str, plan_id: str, plan_hash: str, pipeline_id: str = "", metadata=None) -> RunCheckpoint:
        checkpoint = RunCheckpoint(run_id=run_id, plan_id=plan_id, plan_hash=plan_hash, pipeline_id=pipeline_id, metadata=metadata or {})
        checkpoint.set_expiration(self._retention_days)
        await self._store.save(checkpoint)
        await self._emit_checkpoint_event(run_id=run_id, event_type="CHECKPOINT_CREATED", payload={"plan_id": plan_id, "plan_hash": plan_hash, "pipeline_id": pipeline_id})
        logger.info(f"Created run checkpoint for {run_id}")
        return checkpoint

    async def save_checkpoint(self, run_id: str, step_id: str, state: CheckpointState) -> bool:
        checkpoint = await self._store.load(run_id)
        if checkpoint is None:
            logger.warning(f"No run checkpoint found for {run_id}, creating new one")
            checkpoint = RunCheckpoint(run_id=run_id, plan_id="unknown", plan_hash="unknown")
            checkpoint.set_expiration(self._retention_days)
        checkpoint.set_step_checkpoint(step_id, state)
        success = await self._store.save(checkpoint)
        if success:
            await self._emit_checkpoint_event(run_id=run_id, event_type="CHECKPOINT_SAVED", step_id=step_id, payload={"status": state.status.value, "attempt": state.attempt})
        return success

    async def get_checkpoint(self, run_id: str, step_id: str):
        checkpoint = await self._store.load(run_id)
        return checkpoint.get_step_checkpoint(step_id) if checkpoint else None

    async def get_run_checkpoint(self, run_id: str):
        return await self._store.load(run_id)

    async def get_resumable_steps(self, run_id: str) -> List[str]:
        checkpoint = await self._store.load(run_id)
        return checkpoint.get_completed_steps() if checkpoint else []

    async def verify_idempotency(self, run_id: str, step_id: str, plan_hash: str, attempt: int) -> bool:
        checkpoint = await self._store.load(run_id)
        if checkpoint is None:
            return True
        step_state = checkpoint.get_step_checkpoint(step_id)
        if step_state is None:
            return True
        expected_key = CheckpointState.generate_idempotency_key(plan_hash, step_id, attempt)
        if step_state.idempotency_key == expected_key and step_state.status == CheckpointStatus.COMPLETED:
            logger.info(f"Skipping duplicate execution for {run_id}/{step_id}")
            return False
        return True

    async def clear_checkpoint(self, run_id: str) -> bool:
        success = await self._store.delete(run_id)
        if success:
            await self._emit_checkpoint_event(run_id=run_id, event_type="CHECKPOINT_CLEARED", payload={})
            logger.info(f"Cleared checkpoint for {run_id}")
        return success

    async def prepare_retry(self, original_run_id: str, new_run_id: str, skip_succeeded: bool = True, force_rerun_steps=None):
        original = await self._store.load(original_run_id)
        if original is None:
            logger.warning(f"No checkpoint found for original run {original_run_id}")
            return None
        if original.retry_count >= self._max_retry_count:
            logger.warning(f"Max retry count ({self._max_retry_count}) reached for {original_run_id}")
            return None
        new_checkpoint = RunCheckpoint(run_id=new_run_id, plan_id=original.plan_id, plan_hash=original.plan_hash,
            pipeline_id=original.pipeline_id, parent_run_id=original.run_id, retry_count=original.retry_count + 1,
            metadata={**original.metadata, "retry_from": original_run_id, "skip_succeeded": skip_succeeded, "force_rerun_steps": force_rerun_steps or []})
        force_rerun_set = set(force_rerun_steps or [])
        for step_id, state in original.step_checkpoints.items():
            if skip_succeeded and state.is_resumable() and step_id not in force_rerun_set:
                new_checkpoint.set_step_checkpoint(step_id, state)
        new_checkpoint.set_expiration(self._retention_days)
        await self._store.save(new_checkpoint)
        await self._emit_checkpoint_event(run_id=new_run_id, event_type="CHECKPOINT_RETRY_PREPARED",
            payload={"original_run_id": original_run_id, "retry_count": new_checkpoint.retry_count, "skip_succeeded": skip_succeeded,
                     "force_rerun_steps": force_rerun_steps or [], "skipped_steps": list(new_checkpoint.step_checkpoints.keys())})
        logger.info(f"Prepared retry checkpoint: {new_run_id} from {original_run_id} (retry #{new_checkpoint.retry_count})")
        return new_checkpoint

    async def validate_schema_compatibility(self, original_run_id: str, new_plan_hash: str) -> bool:
        original = await self._store.load(original_run_id)
        if original is None:
            return True
        if original.plan_hash != new_plan_hash:
            logger.warning(f"Plan hash mismatch for {original_run_id}")
            return False
        return True

    async def get_non_idempotent_steps(self, run_id: str, step_metadata):
        checkpoint = await self._store.load(run_id)
        if checkpoint is None:
            return []
        non_idempotent = []
        for step_id, state in checkpoint.step_checkpoints.items():
            if state.status == CheckpointStatus.COMPLETED:
                step_meta = step_metadata.get(step_id, {})
                behavior = step_meta.get("idempotency_behavior", StepIdempotencyBehavior.UNKNOWN)
                if behavior in {StepIdempotencyBehavior.NON_IDEMPOTENT, StepIdempotencyBehavior.UNKNOWN}:
                    non_idempotent.append(step_id)
        return non_idempotent

    async def cleanup_expired_checkpoints(self) -> int:
        count = await self._store.cleanup_expired()
        if count > 0:
            logger.info(f"Cleaned up {count} expired checkpoints")
        return count

    async def _emit_checkpoint_event(self, run_id: str, event_type: str, payload, step_id=None) -> None:
        if self._event_store is None:
            return
        try:
            from greenlang.orchestrator.audit.event_store import EventType, RunEvent, GENESIS_HASH
            prev_hash = await self._event_store.get_latest_hash(run_id)
            event = RunEvent(event_id=str(uuid4()), run_id=run_id, step_id=step_id, event_type=EventType.ARTIFACT_WRITTEN,
                timestamp=datetime.now(timezone.utc), payload={"checkpoint_event": event_type, **payload}, prev_event_hash=prev_hash, event_hash="")
            await self._event_store.append(event)
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to emit checkpoint audit event: {e}")


class CheckpointExecutionContract(BaseModel):
    """Execution contract passed to agents with checkpoint context."""
    run_id: str = Field(...)
    step_id: str = Field(...)
    idempotency_key: str = Field(...)
    attempt: int = Field(default=1)
    is_retry: bool = Field(default=False)
    previous_outputs: Dict[str, Any] = Field(default_factory=dict)
    checkpoint_enabled: bool = Field(default=True)

    model_config = {"frozen": True, "extra": "forbid"}


__all__ = [
    "CheckpointStatus", "StepIdempotencyBehavior", "CheckpointState", "RunCheckpoint",
    "CheckpointExecutionContract", "CheckpointStore", "InMemoryCheckpointStore",
    "CheckpointManager", "DEFAULT_CHECKPOINT_RETENTION_DAYS",
    "DEFAULT_MAX_RETRY_COUNT", "CHECKPOINT_HASH_ALGORITHM",
]
