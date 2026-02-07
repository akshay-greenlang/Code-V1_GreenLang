# -*- coding: utf-8 -*-
"""
DAG Orchestrator REST API Router - AGENT-FOUND-001: GreenLang DAG Orchestrator

FastAPI router providing 20 endpoints for DAG workflow management,
execution, checkpointing, provenance, and monitoring.

All endpoints are mounted under ``/api/v1/orchestrator``.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-001 GreenLang Orchestrator
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None  # type: ignore[assignment, misc]
    logger.warning("FastAPI not available; dag_router is None")


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class CreateDAGRequest(BaseModel):
        """Request body for creating a new DAG workflow."""
        name: str = Field(..., description="Workflow name")
        description: str = Field("", description="Workflow description")
        version: str = Field("1.0.0", description="Version string")
        nodes: Dict[str, Any] = Field(..., description="Node definitions")
        default_retry_policy: Optional[Dict[str, Any]] = Field(
            None, description="DAG-level default retry policy",
        )
        default_timeout_policy: Optional[Dict[str, Any]] = Field(
            None, description="DAG-level default timeout policy",
        )
        on_failure: str = Field("fail_fast", description="DAG failure strategy")
        max_parallel_nodes: int = Field(10, ge=1, le=500)
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class UpdateDAGRequest(BaseModel):
        """Request body for updating a DAG workflow."""
        name: Optional[str] = None
        description: Optional[str] = None
        version: Optional[str] = None
        nodes: Optional[Dict[str, Any]] = None
        default_retry_policy: Optional[Dict[str, Any]] = None
        default_timeout_policy: Optional[Dict[str, Any]] = None
        on_failure: Optional[str] = None
        max_parallel_nodes: Optional[int] = Field(None, ge=1, le=500)
        metadata: Optional[Dict[str, Any]] = None

    class ExecuteDAGRequest(BaseModel):
        """Request body for executing a DAG workflow."""
        input_data: Dict[str, Any] = Field(
            default_factory=dict, description="Input data",
        )
        execution_options: Dict[str, Any] = Field(
            default_factory=dict, description="Execution options",
        )

    class ImportDAGRequest(BaseModel):
        """Request body for importing a DAG from YAML."""
        yaml_content: str = Field(..., description="YAML content")

    class ResumeExecutionRequest(BaseModel):
        """Request body for resuming an execution."""
        agent_registry: Dict[str, Any] = Field(
            default_factory=dict,
            description="Agent registry for resumed execution",
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _get_orchestrator(request: "Request") -> Any:
    """Extract the DAGOrchestrator from app state."""
    svc = getattr(request.app.state, "dag_orchestrator", None)
    if svc is None:
        raise HTTPException(
            status_code=503,
            detail="DAG Orchestrator not configured",
        )
    return svc


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

dag_router: Optional[Any] = None

if FASTAPI_AVAILABLE:
    dag_router = APIRouter(
        prefix="/api/v1/orchestrator",
        tags=["orchestrator"],
    )

    # =================================================================
    # DAG CRUD endpoints
    # =================================================================

    @dag_router.post("/dags", status_code=201, summary="Create DAG workflow")
    async def create_dag(
        body: CreateDAGRequest,
        request: Request,
    ) -> JSONResponse:
        """Create a new DAG workflow definition."""
        orchestrator = _get_orchestrator(request)
        try:
            dag = orchestrator.create_dag(body.model_dump())
            return JSONResponse(
                status_code=201,
                content={"dag": dag.to_dict(), "message": "DAG created"},
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @dag_router.get("/dags", summary="List DAG workflows")
    async def list_dags(request: Request) -> JSONResponse:
        """List all DAG workflow definitions."""
        orchestrator = _get_orchestrator(request)
        dags = orchestrator.list_dags()
        return JSONResponse(
            content={
                "dags": [d.to_dict() for d in dags],
                "count": len(dags),
            }
        )

    @dag_router.get("/dags/{dag_id}", summary="Get DAG workflow")
    async def get_dag(dag_id: str, request: Request) -> JSONResponse:
        """Get a specific DAG workflow definition."""
        orchestrator = _get_orchestrator(request)
        dag = orchestrator.get_dag(dag_id)
        if dag is None:
            raise HTTPException(status_code=404, detail="DAG not found")
        return JSONResponse(content={"dag": dag.to_dict()})

    @dag_router.put("/dags/{dag_id}", summary="Update DAG workflow")
    async def update_dag(
        dag_id: str,
        body: UpdateDAGRequest,
        request: Request,
    ) -> JSONResponse:
        """Update an existing DAG workflow definition."""
        orchestrator = _get_orchestrator(request)
        try:
            dag = orchestrator.update_dag(dag_id, body.model_dump(exclude_none=True))
            if dag is None:
                raise HTTPException(status_code=404, detail="DAG not found")
            return JSONResponse(
                content={"dag": dag.to_dict(), "message": "DAG updated"},
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @dag_router.delete("/dags/{dag_id}", summary="Delete DAG workflow")
    async def delete_dag(dag_id: str, request: Request) -> JSONResponse:
        """Delete a DAG workflow definition."""
        orchestrator = _get_orchestrator(request)
        deleted = orchestrator.delete_dag(dag_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="DAG not found")
        return JSONResponse(content={"message": "DAG deleted"})

    # =================================================================
    # DAG operations endpoints
    # =================================================================

    @dag_router.post(
        "/dags/{dag_id}/validate", summary="Validate DAG",
    )
    async def validate_dag_endpoint(
        dag_id: str, request: Request,
    ) -> JSONResponse:
        """Validate a DAG workflow for structural correctness."""
        orchestrator = _get_orchestrator(request)
        dag = orchestrator.get_dag(dag_id)
        if dag is None:
            raise HTTPException(status_code=404, detail="DAG not found")

        from greenlang.orchestrator.dag_validator import validate_dag
        errors = validate_dag(dag)
        return JSONResponse(
            content={
                "valid": len(errors) == 0,
                "errors": [e.to_dict() for e in errors],
                "error_count": len(errors),
            }
        )

    @dag_router.post(
        "/dags/{dag_id}/execute",
        status_code=202,
        summary="Execute DAG",
    )
    async def execute_dag(
        dag_id: str,
        body: ExecuteDAGRequest,
        request: Request,
    ) -> JSONResponse:
        """Start execution of a DAG workflow."""
        orchestrator = _get_orchestrator(request)
        try:
            trace = await orchestrator.execute_dag(
                dag_id=dag_id,
                input_data=body.input_data,
                execution_options=body.execution_options,
            )
            return JSONResponse(
                status_code=202,
                content={
                    "execution_id": trace.execution_id,
                    "dag_id": trace.dag_id,
                    "status": trace.status.value,
                    "message": "Execution started",
                },
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    # =================================================================
    # Execution endpoints
    # =================================================================

    @dag_router.get("/executions", summary="List executions")
    async def list_executions(
        request: Request,
        dag_id: Optional[str] = Query(None),
        status: Optional[str] = Query(None),
    ) -> JSONResponse:
        """List DAG executions with optional filters."""
        orchestrator = _get_orchestrator(request)
        executions = orchestrator.list_executions(
            dag_id=dag_id, status=status,
        )
        return JSONResponse(
            content={
                "executions": [e.to_dict() for e in executions],
                "count": len(executions),
            }
        )

    @dag_router.get(
        "/executions/{execution_id}", summary="Get execution details",
    )
    async def get_execution(
        execution_id: str, request: Request,
    ) -> JSONResponse:
        """Get detailed status of a DAG execution."""
        orchestrator = _get_orchestrator(request)
        trace = orchestrator.get_execution(execution_id)
        if trace is None:
            raise HTTPException(
                status_code=404, detail="Execution not found",
            )
        return JSONResponse(content={"execution": trace.to_dict()})

    @dag_router.get(
        "/executions/{execution_id}/trace",
        summary="Get execution trace",
    )
    async def get_execution_trace(
        execution_id: str, request: Request,
    ) -> JSONResponse:
        """Get full execution trace with node timings and provenance."""
        orchestrator = _get_orchestrator(request)
        trace = orchestrator.get_execution(execution_id)
        if trace is None:
            raise HTTPException(
                status_code=404, detail="Execution not found",
            )
        return JSONResponse(content=trace.to_dict())

    @dag_router.post(
        "/executions/{execution_id}/cancel",
        summary="Cancel execution",
    )
    async def cancel_execution(
        execution_id: str, request: Request,
    ) -> JSONResponse:
        """Cancel a running DAG execution."""
        orchestrator = _get_orchestrator(request)
        cancelled = await orchestrator.cancel_execution(execution_id)
        if not cancelled:
            raise HTTPException(
                status_code=404, detail="Execution not found or not running",
            )
        return JSONResponse(
            content={"message": "Execution cancellation requested"},
        )

    @dag_router.post(
        "/executions/{execution_id}/resume",
        status_code=202,
        summary="Resume execution from checkpoint",
    )
    async def resume_execution(
        execution_id: str,
        body: ResumeExecutionRequest,
        request: Request,
    ) -> JSONResponse:
        """Resume a failed execution from its last checkpoint."""
        orchestrator = _get_orchestrator(request)
        try:
            trace = await orchestrator.resume_execution(
                execution_id=execution_id,
                agent_registry=body.agent_registry,
            )
            return JSONResponse(
                status_code=202,
                content={
                    "execution_id": trace.execution_id,
                    "status": trace.status.value,
                    "message": "Execution resumed",
                },
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @dag_router.get(
        "/executions/{execution_id}/provenance",
        summary="Get provenance chain",
    )
    async def get_provenance(
        execution_id: str, request: Request,
    ) -> JSONResponse:
        """Get the provenance chain for an execution."""
        orchestrator = _get_orchestrator(request)
        provenance_json = orchestrator.get_provenance(execution_id)
        if provenance_json is None:
            raise HTTPException(
                status_code=404, detail="Provenance not found",
            )
        import json
        return JSONResponse(content=json.loads(provenance_json))

    # =================================================================
    # Checkpoint endpoints
    # =================================================================

    @dag_router.get(
        "/checkpoints/{execution_id}",
        summary="Get checkpoints",
    )
    async def get_checkpoints(
        execution_id: str, request: Request,
    ) -> JSONResponse:
        """Get all checkpoints for an execution."""
        orchestrator = _get_orchestrator(request)
        checkpoints = orchestrator.get_checkpoints(execution_id)
        return JSONResponse(
            content={
                "checkpoints": [cp.to_dict() for cp in checkpoints],
                "count": len(checkpoints),
            }
        )

    @dag_router.delete(
        "/checkpoints/{execution_id}",
        summary="Delete checkpoints",
    )
    async def delete_checkpoints(
        execution_id: str, request: Request,
    ) -> JSONResponse:
        """Delete all checkpoints for an execution."""
        orchestrator = _get_orchestrator(request)
        count = orchestrator.delete_checkpoints(execution_id)
        return JSONResponse(
            content={"deleted": count, "message": "Checkpoints deleted"},
        )

    # =================================================================
    # Monitoring endpoints
    # =================================================================

    @dag_router.get("/metrics", summary="Get orchestrator metrics")
    async def get_metrics(request: Request) -> JSONResponse:
        """Get orchestrator metrics summary."""
        orchestrator = _get_orchestrator(request)
        return JSONResponse(content=orchestrator.get_metrics())

    # =================================================================
    # Import/Export endpoints
    # =================================================================

    @dag_router.post("/dags/import", summary="Import DAG from YAML")
    async def import_dag(
        body: ImportDAGRequest, request: Request,
    ) -> JSONResponse:
        """Import a DAG workflow from YAML content."""
        orchestrator = _get_orchestrator(request)
        try:
            dag = orchestrator.import_dag_yaml(body.yaml_content)
            return JSONResponse(
                status_code=201,
                content={"dag": dag.to_dict(), "message": "DAG imported"},
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @dag_router.get(
        "/dags/{dag_id}/export", summary="Export DAG to YAML",
    )
    async def export_dag(
        dag_id: str, request: Request,
    ) -> JSONResponse:
        """Export a DAG workflow as YAML."""
        orchestrator = _get_orchestrator(request)
        yaml_content = orchestrator.export_dag_yaml(dag_id)
        if yaml_content is None:
            raise HTTPException(status_code=404, detail="DAG not found")
        return JSONResponse(content={"yaml": yaml_content})

    @dag_router.get(
        "/dags/{dag_id}/visualize",
        summary="Get DAG visualization (Mermaid)",
    )
    async def visualize_dag(
        dag_id: str, request: Request,
    ) -> JSONResponse:
        """Get Mermaid diagram representation of a DAG."""
        orchestrator = _get_orchestrator(request)
        dag = orchestrator.get_dag(dag_id)
        if dag is None:
            raise HTTPException(status_code=404, detail="DAG not found")

        # Generate Mermaid syntax
        lines = ["graph TD"]
        for nid, node in sorted(dag.nodes.items()):
            label = f"{nid}[{nid}\\n{node.agent_id}]"
            lines.append(f"    {label}")
            for dep in node.depends_on:
                lines.append(f"    {dep} --> {nid}")

        mermaid = "\n".join(lines)
        return JSONResponse(
            content={"dag_id": dag_id, "mermaid": mermaid},
        )

    # =================================================================
    # Health endpoint
    # =================================================================

    @dag_router.get("/health", summary="Health check")
    async def health_check(request: Request) -> JSONResponse:
        """DAG Orchestrator health check."""
        return JSONResponse(
            content={
                "status": "healthy",
                "service": "dag-orchestrator",
                "version": "1.0.0",
            }
        )


__all__ = ["dag_router"]
