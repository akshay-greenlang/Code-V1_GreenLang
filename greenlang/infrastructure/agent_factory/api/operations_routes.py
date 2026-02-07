# -*- coding: utf-8 -*-
"""
Operations Routes - Async 202 Accepted pattern for long-running operations.

Router prefix: /api/v1/factory/operations

Implements the async 202 Accepted pattern for long-running Agent Factory
operations (deploy, rollback, pack, publish, migrate). Each operation
returns an operation_id that can be polled for status, progress, and
cancellation.

Endpoints:
    POST   /                  - Create a new async operation (returns 202).
    GET    /{operation_id}    - Poll operation status and progress.
    DELETE /{operation_id}    - Request cancellation of a running operation.
    GET    /                  - List operations with optional filters.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/factory/operations", tags=["Operations"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class OperationCreateRequest(BaseModel):
    """Request body for creating a new async operation."""

    operation_type: str = Field(
        ...,
        description="Operation type: deploy, rollback, pack, publish, migrate.",
    )
    agent_key: str = Field(..., description="Target agent key.")
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Operation-specific parameters."
    )
    idempotency_key: Optional[str] = Field(
        None, description="Client-supplied idempotency key (auto-generated if omitted)."
    )


class OperationResponse(BaseModel):
    """Representation of an async operation."""

    operation_id: str = Field(..., description="Unique operation identifier.")
    operation_type: str = Field(..., description="deploy, rollback, pack, publish, migrate.")
    agent_key: Optional[str] = Field(None, description="Target agent key.")
    status: str = Field(..., description="pending, running, completed, failed, cancelled.")
    progress_pct: int = Field(0, ge=0, le=100, description="Completion percentage.")
    poll_url: str = Field(..., description="URL to poll for status updates.")
    started_at: Optional[str] = Field(None, description="ISO timestamp when execution started.")
    completed_at: Optional[str] = Field(None, description="ISO timestamp when execution finished.")
    result: Optional[Dict[str, Any]] = Field(None, description="Output payload on completion.")
    error_message: Optional[str] = Field(None, description="Error details on failure.")


class OperationListResponse(BaseModel):
    """Paginated list of operations."""

    operations: List[OperationResponse]
    total: int


class OperationCancelResponse(BaseModel):
    """Response after requesting cancellation."""

    operation_id: str
    status: str
    message: str


# ---------------------------------------------------------------------------
# Operation Manager
# ---------------------------------------------------------------------------

# Valid operation types (must match the DB CHECK constraint).
_VALID_OPERATION_TYPES = frozenset({"deploy", "rollback", "pack", "publish", "migrate"})


class OperationManager:
    """Manages the full lifecycle of async operations.

    Responsibilities:
        - Create operations with idempotency guarantees.
        - Track status, progress, and results in-memory (backed by DB in prod).
        - Dispatch background tasks for each operation type.
        - Support cancellation via cooperative flag.

    In production the in-memory stores are replaced by PostgreSQL
    (infrastructure.agent_operations) and Redis (idempotency keys with
    24 h TTL).
    """

    def __init__(self) -> None:
        """Initialise operation stores and handler registry."""
        # In-memory stores (replaced by DB + Redis in production).
        self._operations: Dict[str, Dict[str, Any]] = {}
        self._idempotency_cache: Dict[str, str] = {}  # idempotency_key -> operation_id
        self._cancel_flags: Dict[str, bool] = {}
        self._handlers: Dict[str, Callable[..., Coroutine[Any, Any, Dict[str, Any]]]] = {}

        # Register built-in operation handlers.
        self.register_handler("deploy", self._handle_deploy)
        self.register_handler("rollback", self._handle_rollback)
        self.register_handler("pack", self._handle_pack)
        self.register_handler("publish", self._handle_publish)
        self.register_handler("migrate", self._handle_migrate)

    # -- Handler registration -------------------------------------------------

    def register_handler(
        self,
        operation_type: str,
        handler: Callable[..., Coroutine[Any, Any, Dict[str, Any]]],
    ) -> None:
        """Register an async handler for a given operation type.

        Args:
            operation_type: One of the valid operation types.
            handler: Async callable ``(operation_id, agent_key, params) -> dict``.
        """
        self._handlers[operation_type] = handler

    # -- CRUD -----------------------------------------------------------------

    def create_operation(
        self,
        operation_type: str,
        agent_key: str,
        params: Dict[str, Any],
        idempotency_key: Optional[str],
        created_by: str,
    ) -> Dict[str, Any]:
        """Create a new pending operation.

        If an idempotency key already maps to an existing operation, the
        existing operation record is returned without creating a duplicate.

        Args:
            operation_type: Type of operation to create.
            agent_key: Target agent key.
            params: Operation-specific input parameters.
            idempotency_key: Optional client-supplied key for deduplication.
            created_by: Actor / caller identity.

        Returns:
            Operation record dict.

        Raises:
            ValueError: If the operation type is not recognised.
        """
        if operation_type not in _VALID_OPERATION_TYPES:
            raise ValueError(
                f"Invalid operation_type '{operation_type}'. "
                f"Must be one of {sorted(_VALID_OPERATION_TYPES)}."
            )

        # Generate or normalise idempotency key.
        idem_key = idempotency_key or self._generate_idempotency_key(
            operation_type, agent_key, params
        )

        # Return existing operation if idempotency key already used.
        if idem_key in self._idempotency_cache:
            existing_id = self._idempotency_cache[idem_key]
            logger.info(
                "Idempotent hit: returning existing operation %s for key %s",
                existing_id,
                idem_key,
            )
            return self._operations[existing_id]

        operation_id = uuid.uuid4().hex
        now = _now_iso()

        record: Dict[str, Any] = {
            "operation_id": operation_id,
            "operation_type": operation_type,
            "agent_key": agent_key,
            "idempotency_key": idem_key,
            "status": "pending",
            "progress_pct": 0,
            "input_params": params,
            "result": None,
            "error_message": None,
            "started_at": None,
            "completed_at": None,
            "cancelled_at": None,
            "created_by": created_by,
            "created_at": now,
            "updated_at": now,
        }

        self._operations[operation_id] = record
        self._idempotency_cache[idem_key] = operation_id
        self._cancel_flags[operation_id] = False

        logger.info(
            "Operation created: id=%s type=%s agent=%s",
            operation_id,
            operation_type,
            agent_key,
        )
        return record

    def get_operation(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an operation by ID.

        Args:
            operation_id: The unique operation identifier.

        Returns:
            Operation record dict, or ``None`` if not found.
        """
        return self._operations.get(operation_id)

    def list_operations(
        self,
        status: Optional[str] = None,
        agent_key: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[List[Dict[str, Any]], int]:
        """List operations with optional filters.

        Args:
            status: Filter by operation status.
            agent_key: Filter by target agent key.
            limit: Maximum results to return.
            offset: Number of results to skip.

        Returns:
            Tuple of (filtered operation list, total matching count).
        """
        ops = list(self._operations.values())

        # Most recent first.
        ops.sort(key=lambda o: o["created_at"], reverse=True)

        if status:
            ops = [o for o in ops if o["status"] == status]
        if agent_key:
            ops = [o for o in ops if o["agent_key"] == agent_key]

        total = len(ops)
        return ops[offset : offset + limit], total

    def request_cancellation(self, operation_id: str) -> Dict[str, Any]:
        """Request cooperative cancellation of a running operation.

        Args:
            operation_id: The operation to cancel.

        Returns:
            Updated operation record dict.

        Raises:
            KeyError: If the operation does not exist.
            ValueError: If the operation is not in a cancellable state.
        """
        record = self._operations.get(operation_id)
        if record is None:
            raise KeyError(f"Operation '{operation_id}' not found.")

        if record["status"] in ("completed", "failed", "cancelled"):
            raise ValueError(
                f"Operation '{operation_id}' is already in terminal state "
                f"'{record['status']}' and cannot be cancelled."
            )

        self._cancel_flags[operation_id] = True
        record["status"] = "cancelled"
        record["cancelled_at"] = _now_iso()
        record["updated_at"] = _now_iso()

        logger.info("Cancellation requested for operation %s", operation_id)
        return record

    # -- Background execution -------------------------------------------------

    def dispatch(self, operation_id: str) -> None:
        """Dispatch the operation handler as an asyncio background task.

        Args:
            operation_id: The operation to execute.

        Raises:
            KeyError: If the operation does not exist.
            ValueError: If no handler is registered for the operation type.
        """
        record = self._operations.get(operation_id)
        if record is None:
            raise KeyError(f"Operation '{operation_id}' not found.")

        handler = self._handlers.get(record["operation_type"])
        if handler is None:
            raise ValueError(
                f"No handler registered for operation type '{record['operation_type']}'."
            )

        asyncio.create_task(
            self._execute_handler(operation_id, handler),
            name=f"op-{operation_id[:12]}",
        )

    async def _execute_handler(
        self,
        operation_id: str,
        handler: Callable[..., Coroutine[Any, Any, Dict[str, Any]]],
    ) -> None:
        """Wrap handler execution with status tracking and error handling.

        Args:
            operation_id: The operation being executed.
            handler: The async handler to invoke.
        """
        record = self._operations[operation_id]
        record["status"] = "running"
        record["started_at"] = _now_iso()
        record["updated_at"] = _now_iso()

        logger.info("Operation started: %s (%s)", operation_id, record["operation_type"])

        try:
            result = await handler(
                operation_id,
                record["agent_key"],
                record["input_params"],
            )

            # Check for cooperative cancellation.
            if self._cancel_flags.get(operation_id, False):
                record["status"] = "cancelled"
                record["cancelled_at"] = _now_iso()
                logger.info("Operation cancelled during execution: %s", operation_id)
            else:
                record["status"] = "completed"
                record["progress_pct"] = 100
                record["result"] = result
                logger.info("Operation completed: %s", operation_id)

        except Exception as exc:
            record["status"] = "failed"
            record["error_message"] = str(exc)
            logger.error(
                "Operation failed: %s - %s", operation_id, exc, exc_info=True
            )

        finally:
            record["completed_at"] = _now_iso()
            record["updated_at"] = _now_iso()

    def update_progress(self, operation_id: str, progress_pct: int) -> None:
        """Update the progress percentage of a running operation.

        Args:
            operation_id: The operation to update.
            progress_pct: New progress value (0-100).

        Raises:
            KeyError: If the operation does not exist.
        """
        record = self._operations.get(operation_id)
        if record is None:
            raise KeyError(f"Operation '{operation_id}' not found.")

        record["progress_pct"] = max(0, min(100, progress_pct))
        record["updated_at"] = _now_iso()

    def is_cancelled(self, operation_id: str) -> bool:
        """Check whether cancellation has been requested.

        Args:
            operation_id: The operation to check.

        Returns:
            ``True`` if cancellation was requested.
        """
        return self._cancel_flags.get(operation_id, False)

    # -- Built-in handlers ----------------------------------------------------

    async def _handle_deploy(
        self, operation_id: str, agent_key: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle a deploy operation.

        Args:
            operation_id: Current operation ID (for progress updates).
            agent_key: Target agent key.
            params: Deploy parameters (version, environment, strategy, etc.).

        Returns:
            Result dict with deployment details.
        """
        version = params.get("version", "0.1.0")
        environment = params.get("environment", "dev")
        strategy = params.get("strategy", "rolling")

        self.update_progress(operation_id, 10)

        # Simulate validation phase.
        await asyncio.sleep(0.05)
        if self.is_cancelled(operation_id):
            return {"cancelled": True}
        self.update_progress(operation_id, 30)

        # Simulate build / packaging phase.
        await asyncio.sleep(0.05)
        if self.is_cancelled(operation_id):
            return {"cancelled": True}
        self.update_progress(operation_id, 60)

        # Simulate deployment phase.
        await asyncio.sleep(0.05)
        if self.is_cancelled(operation_id):
            return {"cancelled": True}
        self.update_progress(operation_id, 90)

        # Simulate health-check phase.
        await asyncio.sleep(0.02)
        self.update_progress(operation_id, 100)

        logger.info(
            "Deploy handler complete: %s v%s -> %s (%s)",
            agent_key, version, environment, strategy,
        )

        return {
            "agent_key": agent_key,
            "version": version,
            "environment": environment,
            "strategy": strategy,
            "deployment_id": uuid.uuid4().hex[:16],
        }

    async def _handle_rollback(
        self, operation_id: str, agent_key: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle a rollback operation.

        Args:
            operation_id: Current operation ID.
            agent_key: Target agent key.
            params: Rollback parameters (target_version, reason).

        Returns:
            Result dict with rollback details.
        """
        target_version = params.get("target_version", "previous")
        reason = params.get("reason", "")

        self.update_progress(operation_id, 20)
        await asyncio.sleep(0.05)
        if self.is_cancelled(operation_id):
            return {"cancelled": True}

        self.update_progress(operation_id, 60)
        await asyncio.sleep(0.05)

        self.update_progress(operation_id, 100)

        logger.info("Rollback handler complete: %s -> %s", agent_key, target_version)

        return {
            "agent_key": agent_key,
            "from_version": "current",
            "to_version": target_version,
            "reason": reason,
            "rollback_id": uuid.uuid4().hex[:16],
        }

    async def _handle_pack(
        self, operation_id: str, agent_key: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle a pack (build) operation.

        Args:
            operation_id: Current operation ID.
            agent_key: Target agent key.
            params: Pack parameters.

        Returns:
            Result dict with pack details.
        """
        self.update_progress(operation_id, 25)
        await asyncio.sleep(0.05)
        self.update_progress(operation_id, 75)
        await asyncio.sleep(0.05)
        self.update_progress(operation_id, 100)

        checksum = hashlib.sha256(
            f"{agent_key}:{params}".encode()
        ).hexdigest()[:32]

        logger.info("Pack handler complete: %s checksum=%s", agent_key, checksum)

        return {
            "agent_key": agent_key,
            "pack_checksum": checksum,
            "pack_url": f"s3://greenlang-agent-packs/{agent_key}/{checksum}.tar.gz",
        }

    async def _handle_publish(
        self, operation_id: str, agent_key: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle a publish operation (push to Agent Hub).

        Args:
            operation_id: Current operation ID.
            agent_key: Target agent key.
            params: Publish parameters.

        Returns:
            Result dict with publish details.
        """
        self.update_progress(operation_id, 30)
        await asyncio.sleep(0.05)
        self.update_progress(operation_id, 80)
        await asyncio.sleep(0.03)
        self.update_progress(operation_id, 100)

        logger.info("Publish handler complete: %s", agent_key)

        return {
            "agent_key": agent_key,
            "published_version": params.get("version", "0.1.0"),
            "hub_url": f"https://hub.greenlang.ai/agents/{agent_key}",
        }

    async def _handle_migrate(
        self, operation_id: str, agent_key: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle a migrate operation (schema or config migration).

        Args:
            operation_id: Current operation ID.
            agent_key: Target agent key.
            params: Migration parameters.

        Returns:
            Result dict with migration details.
        """
        self.update_progress(operation_id, 20)
        await asyncio.sleep(0.05)
        self.update_progress(operation_id, 50)
        await asyncio.sleep(0.05)
        self.update_progress(operation_id, 100)

        logger.info("Migrate handler complete: %s", agent_key)

        return {
            "agent_key": agent_key,
            "migration_type": params.get("migration_type", "config"),
            "applied_steps": params.get("steps", 0),
        }

    # -- Helpers --------------------------------------------------------------

    @staticmethod
    def _generate_idempotency_key(
        operation_type: str, agent_key: str, params: Dict[str, Any]
    ) -> str:
        """Deterministically generate an idempotency key from inputs.

        Args:
            operation_type: Type of operation.
            agent_key: Target agent key.
            params: Operation parameters.

        Returns:
            SHA-256 hex digest (first 32 chars) as idempotency key.
        """
        payload = f"{operation_type}:{agent_key}:{sorted(params.items())}"
        return hashlib.sha256(payload.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _build_poll_url(request: Request, operation_id: str) -> str:
    """Construct the absolute poll URL for an operation.

    Args:
        request: The incoming FastAPI request (used for base URL).
        operation_id: The operation identifier.

    Returns:
        Fully-qualified poll URL string.
    """
    base = str(request.base_url).rstrip("/")
    return f"{base}/api/v1/factory/operations/{operation_id}"


def _to_response(record: Dict[str, Any], poll_url: str) -> OperationResponse:
    """Convert an internal operation record to an API response model.

    Args:
        record: Internal operation dict from OperationManager.
        poll_url: The absolute poll URL for this operation.

    Returns:
        Pydantic OperationResponse instance.
    """
    return OperationResponse(
        operation_id=record["operation_id"],
        operation_type=record["operation_type"],
        agent_key=record.get("agent_key"),
        status=record["status"],
        progress_pct=record["progress_pct"],
        poll_url=poll_url,
        started_at=record.get("started_at"),
        completed_at=record.get("completed_at"),
        result=record.get("result"),
        error_message=record.get("error_message"),
    )


# ---------------------------------------------------------------------------
# Singleton manager
# ---------------------------------------------------------------------------

operation_manager = OperationManager()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/", status_code=202, response_model=OperationResponse)
async def create_operation(
    body: OperationCreateRequest, request: Request
) -> OperationResponse:
    """Create a new async operation (returns 202 Accepted).

    The operation is queued and dispatched as a background task. Clients
    should poll the ``poll_url`` in the response to track progress.

    Args:
        body: Operation creation request payload.
        request: FastAPI request (used for URL generation).

    Returns:
        OperationResponse with ``status='pending'`` and a poll URL.
    """
    try:
        record = operation_manager.create_operation(
            operation_type=body.operation_type,
            agent_key=body.agent_key,
            params=body.params,
            idempotency_key=body.idempotency_key,
            created_by=request.headers.get("X-User-Id", "anonymous"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    poll_url = _build_poll_url(request, record["operation_id"])

    # Dispatch background execution only for newly created (pending) operations.
    if record["status"] == "pending":
        try:
            operation_manager.dispatch(record["operation_id"])
        except (KeyError, ValueError) as exc:
            logger.error("Failed to dispatch operation: %s", exc)
            raise HTTPException(status_code=500, detail="Failed to dispatch operation.") from exc

    logger.info(
        "POST /operations -> 202 operation_id=%s type=%s agent=%s",
        record["operation_id"],
        body.operation_type,
        body.agent_key,
    )

    return _to_response(record, poll_url)


@router.get("/{operation_id}", response_model=OperationResponse)
async def get_operation(operation_id: str, request: Request) -> OperationResponse:
    """Poll the status and progress of an operation.

    Args:
        operation_id: The unique operation identifier.
        request: FastAPI request (used for URL generation).

    Returns:
        Current OperationResponse with updated status and progress.
    """
    record = operation_manager.get_operation(operation_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Operation '{operation_id}' not found.")

    poll_url = _build_poll_url(request, operation_id)
    return _to_response(record, poll_url)


@router.delete("/{operation_id}", response_model=OperationCancelResponse)
async def cancel_operation(operation_id: str) -> OperationCancelResponse:
    """Request cancellation of a running or pending operation.

    Cancellation is cooperative -- the operation handler checks a flag
    at each phase boundary and exits early if set.

    Args:
        operation_id: The operation to cancel.

    Returns:
        OperationCancelResponse confirming the cancellation request.
    """
    try:
        record = operation_manager.request_cancellation(operation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    logger.info("DELETE /operations/%s -> cancelled", operation_id)

    return OperationCancelResponse(
        operation_id=operation_id,
        status=record["status"],
        message="Cancellation requested. The operation will stop at the next safe checkpoint.",
    )


@router.get("/", response_model=OperationListResponse)
async def list_operations(
    request: Request,
    status: Optional[str] = Query(None, description="Filter by status."),
    agent_key: Optional[str] = Query(None, description="Filter by agent key."),
    limit: int = Query(20, ge=1, le=100, description="Max results."),
    offset: int = Query(0, ge=0, description="Results offset."),
) -> OperationListResponse:
    """List operations with optional status and agent_key filters.

    Args:
        request: FastAPI request (used for URL generation).
        status: Optional status filter.
        agent_key: Optional agent key filter.
        limit: Maximum number of results.
        offset: Number of results to skip.

    Returns:
        Paginated OperationListResponse.
    """
    ops, total = operation_manager.list_operations(
        status=status,
        agent_key=agent_key,
        limit=limit,
        offset=offset,
    )

    responses = [
        _to_response(o, _build_poll_url(request, o["operation_id"])) for o in ops
    ]

    return OperationListResponse(operations=responses, total=total)
