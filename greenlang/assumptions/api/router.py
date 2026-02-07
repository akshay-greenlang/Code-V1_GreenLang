# -*- coding: utf-8 -*-
"""
Assumptions Registry REST API Router - AGENT-FOUND-004: Assumptions Registry

FastAPI router providing 20 endpoints for assumption management,
scenario management, validation, dependency analysis, and provenance.

All endpoints are mounted under ``/api/v1/assumptions``.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-004 Assumptions Registry
Status: Production Ready
"""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import  (no `from __future__ import annotations` here)
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None  # type: ignore[assignment, misc]
    logger.warning("FastAPI not available; assumptions router is None")


# ---------------------------------------------------------------------------
# Pydantic request/response models (only when FastAPI is available)
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class CreateAssumptionRequest(BaseModel):
        """Request body for creating an assumption."""
        assumption_id: str = Field(..., description="Unique assumption identifier")
        name: str = Field(..., description="Human-readable name")
        description: str = Field("", description="Detailed description")
        category: str = Field("custom", description="Assumption category")
        data_type: str = Field("float", description="Data type")
        value: Any = Field(..., description="Initial value")
        unit: Optional[str] = Field(None, description="Unit of measurement")
        default_value: Optional[Any] = Field(None, description="Default value")
        user_id: str = Field("system", description="User creating the assumption")
        change_reason: str = Field("Initial creation", description="Reason")
        metadata_source: str = Field("user_defined", description="Source")
        metadata_tags: Optional[List[str]] = Field(None, description="Tags")
        validation_rules: Optional[List[Dict[str, Any]]] = Field(
            None, description="Validation rules",
        )

    class UpdateAssumptionRequest(BaseModel):
        """Request body for updating an assumption value."""
        value: Any = Field(..., description="New value")
        user_id: str = Field("system", description="User making the change")
        reason: str = Field("Value update", description="Reason for change")

    class SetValueRequest(BaseModel):
        """Request body for setting an assumption value."""
        value: Any = Field(..., description="New value")
        user_id: str = Field("system", description="User making the change")
        reason: str = Field("Value update", description="Reason for change")

    class ValidateRequest(BaseModel):
        """Request body for validating a value."""
        assumption_id: str = Field(..., description="Assumption to validate against")
        value: Any = Field(..., description="Value to validate")

    class CreateScenarioRequest(BaseModel):
        """Request body for creating a scenario."""
        name: str = Field(..., description="Scenario name")
        description: str = Field("", description="Scenario description")
        scenario_type: str = Field("custom", description="Scenario type")
        overrides: Optional[Dict[str, Any]] = Field(None, description="Overrides")
        user_id: str = Field("system", description="User creating the scenario")
        tags: Optional[List[str]] = Field(None, description="Tags")

    class UpdateScenarioRequest(BaseModel):
        """Request body for updating a scenario."""
        name: Optional[str] = Field(None, description="New name")
        description: Optional[str] = Field(None, description="New description")
        overrides: Optional[Dict[str, Any]] = Field(None, description="New overrides")
        is_active: Optional[bool] = Field(None, description="Active flag")
        tags: Optional[List[str]] = Field(None, description="New tags")

    class ExportRequest(BaseModel):
        """Request body for export."""
        user_id: str = Field("system", description="User performing export")

    class ImportRequest(BaseModel):
        """Request body for import."""
        data: Dict[str, Any] = Field(..., description="Import data")
        user_id: str = Field("system", description="User performing import")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    router = APIRouter(
        prefix="/api/v1/assumptions",
        tags=["assumptions"],
    )
else:
    router = None  # type: ignore[assignment]


def _get_service(request: Request) -> Any:
    """Extract AssumptionsService from app state.

    Args:
        request: FastAPI request object.

    Returns:
        AssumptionsService instance.

    Raises:
        HTTPException: If service is not configured.
    """
    service = getattr(request.app.state, "assumptions_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Assumptions service not configured",
        )
    return service


if FASTAPI_AVAILABLE:

    # 1. Health check
    @router.get("/health")
    async def health() -> Dict[str, str]:
        """Assumptions registry health check endpoint."""
        return {"status": "healthy", "service": "assumptions-registry"}

    # 2. Prometheus metrics endpoint
    @router.get("/metrics")
    async def metrics_endpoint(request: Request) -> Dict[str, Any]:
        """Get assumptions registry metrics summary."""
        service = _get_service(request)
        return {
            "assumptions_count": service.registry.count,
            "scenarios_count": service.scenario_manager.count,
            "provenance_entries": service.provenance.entry_count,
        }

    # 3. Create assumption
    @router.post("/")
    async def create_assumption(
        body: CreateAssumptionRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Create a new assumption."""
        service = _get_service(request)
        try:
            assumption = service.registry.create(
                assumption_id=body.assumption_id,
                name=body.name,
                description=body.description,
                category=body.category,
                data_type=body.data_type,
                value=body.value,
                unit=body.unit,
                default_value=body.default_value,
                user_id=body.user_id,
                change_reason=body.change_reason,
                metadata_source=body.metadata_source,
                metadata_tags=body.metadata_tags,
                validation_rules=body.validation_rules,
            )
            return assumption.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 4. List assumptions
    @router.get("/")
    async def list_assumptions(
        category: Optional[str] = Query(None),
        search: Optional[str] = Query(None),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List assumptions with optional filtering."""
        service = _get_service(request)
        assumptions = service.registry.list(category=category, search=search)
        return {
            "assumptions": [a.model_dump(mode="json") for a in assumptions],
            "count": len(assumptions),
        }

    # 5. Get assumption
    @router.get("/{assumption_id}")
    async def get_assumption(
        assumption_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a specific assumption by ID."""
        service = _get_service(request)
        assumption = service.registry.get(assumption_id)
        if assumption is None:
            raise HTTPException(
                status_code=404,
                detail=f"Assumption {assumption_id} not found",
            )
        return assumption.model_dump(mode="json")

    # 6. Update assumption
    @router.put("/{assumption_id}")
    async def update_assumption(
        assumption_id: str,
        body: UpdateAssumptionRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Update an assumption value."""
        service = _get_service(request)
        try:
            assumption = service.registry.update(
                assumption_id=assumption_id,
                value=body.value,
                user_id=body.user_id,
                reason=body.reason,
            )
            return assumption.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 7. Delete assumption
    @router.delete("/{assumption_id}")
    async def delete_assumption(
        assumption_id: str,
        user_id: str = Query("system"),
        reason: str = Query("Deletion"),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """Delete an assumption."""
        service = _get_service(request)
        try:
            deleted = service.registry.delete(
                assumption_id=assumption_id,
                user_id=user_id,
                reason=reason,
            )
            if not deleted:
                raise HTTPException(
                    status_code=404,
                    detail=f"Assumption {assumption_id} not found",
                )
            return {"assumption_id": assumption_id, "deleted": True}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 8. Get assumption versions
    @router.get("/{assumption_id}/versions")
    async def get_versions(
        assumption_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get version history for an assumption."""
        service = _get_service(request)
        try:
            versions = service.registry.get_versions(assumption_id)
            return {
                "assumption_id": assumption_id,
                "versions": [v.model_dump(mode="json") for v in versions],
                "count": len(versions),
            }
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    # 9. Get assumption value
    @router.get("/{assumption_id}/value")
    async def get_value(
        assumption_id: str,
        scenario_id: Optional[str] = Query(None),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """Get the resolved value for an assumption."""
        service = _get_service(request)
        try:
            value = service.registry.get_value(assumption_id)
            source = "baseline"

            # Check scenario override
            if scenario_id:
                override_value, override_source = (
                    service.scenario_manager.resolve_value(
                        assumption_id, scenario_id,
                    )
                )
                if override_value is not None:
                    value = override_value
                    source = override_source

            return {
                "assumption_id": assumption_id,
                "value": value,
                "value_source": source,
                "scenario_id": scenario_id,
            }
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    # 10. Set assumption value
    @router.put("/{assumption_id}/value")
    async def set_value(
        assumption_id: str,
        body: SetValueRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Set a new value for an assumption."""
        service = _get_service(request)
        try:
            service.registry.set_value(
                assumption_id=assumption_id,
                value=body.value,
                user_id=body.user_id,
                reason=body.reason,
            )
            return {
                "assumption_id": assumption_id,
                "value": body.value,
                "updated": True,
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 11. Validate value
    @router.post("/validate")
    async def validate_value(
        body: ValidateRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Validate a value against an assumption's rules."""
        service = _get_service(request)
        assumption = service.registry.get(body.assumption_id)
        if assumption is None:
            raise HTTPException(
                status_code=404,
                detail=f"Assumption {body.assumption_id} not found",
            )
        result = service.validator.validate(assumption, body.value)
        return result.model_dump(mode="json")

    # 12. Create scenario
    @router.post("/scenarios")
    async def create_scenario(
        body: CreateScenarioRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Create a new scenario."""
        service = _get_service(request)
        try:
            scenario = service.scenario_manager.create(
                name=body.name,
                description=body.description,
                scenario_type=body.scenario_type,
                overrides=body.overrides,
                user_id=body.user_id,
                tags=body.tags,
            )
            return scenario.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 13. List scenarios
    @router.get("/scenarios/list")
    async def list_scenarios(
        scenario_type: Optional[str] = Query(None),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List all scenarios."""
        service = _get_service(request)
        scenarios = service.scenario_manager.list(scenario_type=scenario_type)
        return {
            "scenarios": [s.model_dump(mode="json") for s in scenarios],
            "count": len(scenarios),
        }

    # 14. Get scenario
    @router.get("/scenarios/{scenario_id}")
    async def get_scenario(
        scenario_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a specific scenario by ID."""
        service = _get_service(request)
        scenario = service.scenario_manager.get(scenario_id)
        if scenario is None:
            raise HTTPException(
                status_code=404,
                detail=f"Scenario {scenario_id} not found",
            )
        return scenario.model_dump(mode="json")

    # 15. Update scenario
    @router.put("/scenarios/{scenario_id}")
    async def update_scenario(
        scenario_id: str,
        body: UpdateScenarioRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Update a scenario."""
        service = _get_service(request)
        try:
            scenario = service.scenario_manager.update(
                scenario_id=scenario_id,
                name=body.name,
                description=body.description,
                overrides=body.overrides,
                is_active=body.is_active,
                tags=body.tags,
            )
            return scenario.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 16. Delete scenario
    @router.delete("/scenarios/{scenario_id}")
    async def delete_scenario(
        scenario_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Delete a scenario."""
        service = _get_service(request)
        try:
            deleted = service.scenario_manager.delete(scenario_id)
            if not deleted:
                raise HTTPException(
                    status_code=404,
                    detail=f"Scenario {scenario_id} not found",
                )
            return {"scenario_id": scenario_id, "deleted": True}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 17. Get dependency graph
    @router.get("/{assumption_id}/dependencies")
    async def get_dependencies(
        assumption_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get dependency graph for an assumption."""
        service = _get_service(request)
        impact = service.dependency_tracker.get_impact(assumption_id)
        node = service.dependency_tracker.get_node(assumption_id)
        return {
            "assumption_id": assumption_id,
            "node": node.model_dump(mode="json") if node else None,
            "impact": impact,
        }

    # 18. Get sensitivity analysis
    @router.get("/{assumption_id}/sensitivity")
    async def get_sensitivity(
        assumption_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get sensitivity analysis for an assumption."""
        service = _get_service(request)
        assumption = service.registry.get(assumption_id)
        if assumption is None:
            raise HTTPException(
                status_code=404,
                detail=f"Assumption {assumption_id} not found",
            )

        # Collect scenario values
        scenario_values: Dict[str, Any] = {}
        for scenario in service.scenario_manager.list():
            if assumption_id in scenario.overrides:
                scenario_values[scenario.name] = scenario.overrides[assumption_id]
            else:
                scenario_values[scenario.name] = assumption.current_value

        result: Dict[str, Any] = {
            "assumption_id": assumption_id,
            "baseline_value": assumption.current_value,
            "scenario_values": scenario_values,
            "dependency_count": len(assumption.used_by),
            "dependent_calculations": assumption.used_by,
        }

        # Range analysis for numeric types
        numeric_values = [
            v for v in scenario_values.values()
            if isinstance(v, (int, float))
        ]
        if numeric_values:
            result["min_value"] = min(numeric_values)
            result["max_value"] = max(numeric_values)
            result["range"] = max(numeric_values) - min(numeric_values)
            if assumption.current_value and assumption.current_value != 0:
                result["range_pct"] = (
                    (max(numeric_values) - min(numeric_values))
                    / abs(assumption.current_value)
                    * 100
                )

        return result

    # 19. Export all
    @router.post("/export")
    async def export_all(
        body: ExportRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Export all assumptions."""
        service = _get_service(request)
        return service.registry.export_all(user_id=body.user_id)

    # 20. Import
    @router.post("/import")
    async def import_all(
        body: ImportRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Import assumptions from export data."""
        service = _get_service(request)
        count = service.registry.import_all(
            data=body.data, user_id=body.user_id,
        )
        return {"imported_count": count}


__all__ = [
    "router",
]
