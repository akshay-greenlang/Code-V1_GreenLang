# -*- coding: utf-8 -*-
"""
DAG Orchestrator Setup - AGENT-FOUND-001: GreenLang DAG Orchestrator

Provides ``configure_dag_orchestrator(app)`` which wires up the DAG
execution engine (executor, node runner, checkpoint store, provenance
tracker) and mounts the REST API.

Also exposes ``get_dag_orchestrator(app)`` for programmatic access and
the ``DAGOrchestrator`` facade class.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.orchestrator.dag_setup import configure_dag_orchestrator
    >>> app = FastAPI()
    >>> configure_dag_orchestrator(app)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-001 GreenLang Orchestrator
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from greenlang.orchestrator.checkpoint_store import (
    DAGCheckpointStore,
    create_checkpoint_store,
)
from greenlang.orchestrator.config import OrchestratorConfig, get_config
from greenlang.orchestrator.dag_builder import DAGBuilder
from greenlang.orchestrator.dag_executor import (
    DAGExecutor,
    ExecutionOptions,
)
from greenlang.orchestrator.dag_validator import validate_dag
from greenlang.orchestrator.models import (
    DAGCheckpoint,
    DAGWorkflow,
    ExecutionStatus,
    ExecutionTrace,
)
from greenlang.orchestrator.node_runner import NodeRunner
from greenlang.orchestrator.provenance import ProvenanceTracker
from greenlang.orchestrator.topological_sort import level_grouping

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment, misc]
    FASTAPI_AVAILABLE = False


# ===================================================================
# DAGOrchestrator facade
# ===================================================================


class DAGOrchestrator:
    """Unified facade over the DAG execution engine.

    Manages DAG workflow CRUD, execution, checkpointing, provenance,
    and query operations.

    Attributes:
        config: Orchestrator configuration.
        executor: Core DAG execution engine.
        node_runner: Node execution handler.
        checkpoint_store: Checkpoint storage backend.
        provenance_tracker: Provenance tracking engine.
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
    ) -> None:
        """Initialize the DAG Orchestrator facade.

        Args:
            config: Configuration (loaded from env if not provided).
        """
        self.config = config or get_config()

        # Initialize components
        self.node_runner = NodeRunner(self.config)
        self.checkpoint_store: DAGCheckpointStore = create_checkpoint_store(
            strategy=self.config.checkpoint_strategy,
            checkpoint_dir=self.config.checkpoint_dir,
            db_connection_string=self.config.db_connection_string,
        )
        self.provenance_tracker = ProvenanceTracker()
        self.executor = DAGExecutor(
            config=self.config,
            node_runner=self.node_runner,
            checkpoint_store=self.checkpoint_store,
            provenance_tracker=self.provenance_tracker,
        )

        # In-memory registry for DAG definitions
        self._dag_registry: Dict[str, DAGWorkflow] = {}

        # In-memory execution trace store
        self._execution_store: Dict[str, ExecutionTrace] = {}

        logger.info("DAGOrchestrator facade created")

    # ------------------------------------------------------------------
    # DAG CRUD
    # ------------------------------------------------------------------

    def create_dag(self, data: Dict[str, Any]) -> DAGWorkflow:
        """Create a new DAG workflow from a data dictionary.

        Args:
            data: DAG creation data (matches CreateDAGRequest fields).

        Returns:
            Created DAGWorkflow.

        Raises:
            ValueError: If validation fails or DAG already exists.
        """
        dag = DAGWorkflow.from_dict(data)
        if not dag.dag_id:
            # Auto-generate ID
            import hashlib
            content = f"{dag.name}:{sorted(dag.nodes.keys())}"
            hash_hex = hashlib.sha256(content.encode()).hexdigest()[:12]
            dag.dag_id = f"dag_{hash_hex}"

        # Validate
        errors = validate_dag(dag)
        if errors:
            raise ValueError(
                f"DAG validation failed: "
                + "; ".join(e.message for e in errors)
            )

        if dag.dag_id in self._dag_registry:
            raise ValueError(f"DAG '{dag.dag_id}' already exists")

        dag.hash = dag.calculate_hash()
        self._dag_registry[dag.dag_id] = dag
        logger.info("Created DAG: id=%s name='%s'", dag.dag_id, dag.name)
        return dag

    def get_dag(self, dag_id: str) -> Optional[DAGWorkflow]:
        """Retrieve a DAG workflow by ID.

        Args:
            dag_id: DAG identifier.

        Returns:
            DAGWorkflow or None.
        """
        return self._dag_registry.get(dag_id)

    def list_dags(self) -> List[DAGWorkflow]:
        """List all registered DAG workflows.

        Returns:
            List of DAGWorkflow instances.
        """
        return list(self._dag_registry.values())

    def update_dag(
        self, dag_id: str, updates: Dict[str, Any],
    ) -> Optional[DAGWorkflow]:
        """Update an existing DAG workflow.

        Args:
            dag_id: DAG identifier.
            updates: Dictionary of fields to update.

        Returns:
            Updated DAGWorkflow or None if not found.

        Raises:
            ValueError: If updated DAG fails validation.
        """
        existing = self._dag_registry.get(dag_id)
        if existing is None:
            return None

        # Merge updates into existing definition
        existing_dict = existing.to_dict()
        existing_dict.update(updates)
        existing_dict["dag_id"] = dag_id

        updated = DAGWorkflow.from_dict(existing_dict)
        errors = validate_dag(updated)
        if errors:
            raise ValueError(
                f"Updated DAG validation failed: "
                + "; ".join(e.message for e in errors)
            )

        updated.hash = updated.calculate_hash()
        self._dag_registry[dag_id] = updated
        logger.info("Updated DAG: id=%s", dag_id)
        return updated

    def delete_dag(self, dag_id: str) -> bool:
        """Delete a DAG workflow.

        Args:
            dag_id: DAG identifier.

        Returns:
            True if deleted, False if not found.
        """
        if dag_id in self._dag_registry:
            del self._dag_registry[dag_id]
            logger.info("Deleted DAG: id=%s", dag_id)
            return True
        return False

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute_dag(
        self,
        dag_id: str,
        input_data: Dict[str, Any],
        execution_options: Optional[Dict[str, Any]] = None,
        agent_registry: Optional[Dict[str, Any]] = None,
    ) -> ExecutionTrace:
        """Execute a DAG workflow.

        Args:
            dag_id: DAG to execute.
            input_data: Input data for the execution.
            execution_options: Optional execution options.
            agent_registry: Mapping from agent_id to agent instances.

        Returns:
            ExecutionTrace with results.

        Raises:
            KeyError: If DAG not found.
        """
        dag = self._dag_registry.get(dag_id)
        if dag is None:
            raise KeyError(f"DAG '{dag_id}' not found")

        opts_dict = execution_options or {}
        opts = ExecutionOptions(
            checkpoint_enabled=opts_dict.get("checkpoint_enabled", True),
            deterministic_mode=opts_dict.get("deterministic_mode", True),
            agent_registry=agent_registry or opts_dict.get("agent_registry", {}),
        )

        trace = await self.executor.execute(dag, input_data, opts)
        self._execution_store[trace.execution_id] = trace
        return trace

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution.

        Args:
            execution_id: Execution to cancel.

        Returns:
            True if cancellation requested.
        """
        if execution_id in self._execution_store:
            return await self.executor.cancel(execution_id)
        return False

    async def resume_execution(
        self,
        execution_id: str,
        agent_registry: Optional[Dict[str, Any]] = None,
    ) -> ExecutionTrace:
        """Resume a failed execution from checkpoint.

        Args:
            execution_id: Execution to resume.
            agent_registry: Agent registry for resumed execution.

        Returns:
            New ExecutionTrace.

        Raises:
            KeyError: If execution not found.
            ValueError: If execution cannot be resumed.
        """
        old_trace = self._execution_store.get(execution_id)
        if old_trace is None:
            raise KeyError(f"Execution '{execution_id}' not found")

        dag = self._dag_registry.get(old_trace.dag_id)
        if dag is None:
            raise KeyError(f"DAG '{old_trace.dag_id}' no longer exists")

        opts = ExecutionOptions(
            checkpoint_enabled=True,
            deterministic_mode=True,
            resume_execution_id=execution_id,
            agent_registry=agent_registry or {},
        )

        trace = await self.executor.execute(
            dag, old_trace.input_data, opts,
        )
        self._execution_store[trace.execution_id] = trace
        return trace

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    def get_execution(self, execution_id: str) -> Optional[ExecutionTrace]:
        """Get an execution trace.

        Args:
            execution_id: Execution identifier.

        Returns:
            ExecutionTrace or None.
        """
        return self._execution_store.get(execution_id)

    def list_executions(
        self,
        dag_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[ExecutionTrace]:
        """List executions with optional filters.

        Args:
            dag_id: Filter by DAG ID.
            status: Filter by execution status.

        Returns:
            List of ExecutionTraces.
        """
        results = list(self._execution_store.values())
        if dag_id:
            results = [t for t in results if t.dag_id == dag_id]
        if status:
            results = [t for t in results if t.status.value == status]
        return results

    def get_provenance(self, execution_id: str) -> Optional[str]:
        """Get provenance chain JSON for an execution.

        Args:
            execution_id: Execution identifier.

        Returns:
            JSON string or None.
        """
        provenances = self.provenance_tracker.get_trace(execution_id)
        if not provenances:
            return None
        return self.provenance_tracker.export_json(execution_id)

    def get_checkpoints(
        self, execution_id: str,
    ) -> List[DAGCheckpoint]:
        """Get all checkpoints for an execution.

        Args:
            execution_id: Execution identifier.

        Returns:
            List of DAGCheckpoints.
        """
        return self.checkpoint_store.list_checkpoints(execution_id)

    def delete_checkpoints(self, execution_id: str) -> int:
        """Delete all checkpoints for an execution.

        Args:
            execution_id: Execution identifier.

        Returns:
            Number of checkpoints deleted.
        """
        return self.checkpoint_store.delete(execution_id)

    # ------------------------------------------------------------------
    # Import/Export
    # ------------------------------------------------------------------

    def import_dag_yaml(self, yaml_content: str) -> DAGWorkflow:
        """Import a DAG from YAML content.

        Args:
            yaml_content: YAML string.

        Returns:
            Imported DAGWorkflow.
        """
        dag = DAGWorkflow.from_yaml(yaml_content)
        errors = validate_dag(dag)
        if errors:
            raise ValueError(
                f"Imported DAG validation failed: "
                + "; ".join(e.message for e in errors)
            )
        dag.hash = dag.calculate_hash()
        self._dag_registry[dag.dag_id] = dag
        return dag

    def export_dag_yaml(self, dag_id: str) -> Optional[str]:
        """Export a DAG as YAML string.

        Args:
            dag_id: DAG identifier.

        Returns:
            YAML string or None.
        """
        dag = self._dag_registry.get(dag_id)
        if dag is None:
            return None
        return dag.to_yaml()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics summary.

        Returns:
            Dictionary with metric summaries.
        """
        return {
            "registered_dags": len(self._dag_registry),
            "total_executions": len(self._execution_store),
            "executions_by_status": self._count_by_status(),
            "checkpoint_strategy": self.config.checkpoint_strategy,
            "provenance_enabled": self.config.enable_provenance,
            "determinism_enabled": self.config.enable_determinism,
        }

    def _count_by_status(self) -> Dict[str, int]:
        """Count executions by status."""
        counts: Dict[str, int] = {}
        for trace in self._execution_store.values():
            status = trace.status.value
            counts[status] = counts.get(status, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Shutdown the orchestrator and release resources."""
        self.node_runner.shutdown()
        logger.info("DAGOrchestrator shut down")


# ===================================================================
# FastAPI integration
# ===================================================================


def configure_dag_orchestrator(
    app: Any,
    config: Optional[OrchestratorConfig] = None,
) -> DAGOrchestrator:
    """Configure the DAG Orchestrator on a FastAPI application.

    Creates the DAGOrchestrator, stores it in app.state, and
    mounts the DAG API router.

    Args:
        app: FastAPI application instance.
        config: Optional configuration.

    Returns:
        DAGOrchestrator instance.
    """
    orchestrator = DAGOrchestrator(config)
    app.state.dag_orchestrator = orchestrator

    # Mount DAG router
    try:
        from greenlang.orchestrator.api.dag_router import dag_router
        if dag_router is not None:
            app.include_router(dag_router)
            logger.info("DAG Orchestrator API router mounted")
    except ImportError:
        logger.warning("DAG router not available; API not mounted")

    logger.info("DAG Orchestrator configured on app")
    return orchestrator


def get_dag_orchestrator(app: Any) -> DAGOrchestrator:
    """Get the DAGOrchestrator instance from app state.

    Args:
        app: FastAPI application instance.

    Returns:
        DAGOrchestrator instance.

    Raises:
        RuntimeError: If orchestrator not configured.
    """
    orchestrator = getattr(app.state, "dag_orchestrator", None)
    if orchestrator is None:
        raise RuntimeError(
            "DAG Orchestrator not configured. "
            "Call configure_dag_orchestrator(app) first."
        )
    return orchestrator


__all__ = [
    "DAGOrchestrator",
    "configure_dag_orchestrator",
    "get_dag_orchestrator",
]
