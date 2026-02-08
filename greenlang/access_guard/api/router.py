# -*- coding: utf-8 -*-
"""
Access Guard REST API Router - AGENT-FOUND-006: Access & Policy Guard

FastAPI router providing 20 endpoints for access control, policy
management, audit logging, classification, rate limiting, OPA
integration, and provenance tracking.

All endpoints are mounted under ``/api/v1/access-guard``.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-006 Access & Policy Guard
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
    logger.warning("FastAPI not available; access guard router is None")


# ---------------------------------------------------------------------------
# Pydantic request/response models (only when FastAPI is available)
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class CheckAccessRequest(BaseModel):
        """Request body for checking access."""
        principal: Dict[str, Any] = Field(..., description="Principal data")
        resource: Dict[str, Any] = Field(..., description="Resource data")
        action: str = Field(..., description="Action to check")
        context: Optional[Dict[str, Any]] = Field(None, description="ABAC context")
        source_ip: Optional[str] = Field(None, description="Source IP")

    class AddPolicyRequest(BaseModel):
        """Request body for adding a policy."""
        policy_id: str = Field(..., description="Unique policy ID")
        name: str = Field(..., description="Policy name")
        description: str = Field("", description="Policy description")
        version: str = Field("1.0.0", description="Policy version")
        enabled: bool = Field(True, description="Whether policy is active")
        rules: List[Dict[str, Any]] = Field(default_factory=list, description="Policy rules")
        parent_policy_id: Optional[str] = Field(None, description="Parent policy ID")
        tenant_id: Optional[str] = Field(None, description="Tenant scope")
        applies_to: Optional[List[str]] = Field(None, description="Resource types")
        created_by: Optional[str] = Field(None, description="Creator")

    class UpdatePolicyRequest(BaseModel):
        """Request body for updating a policy."""
        name: Optional[str] = Field(None, description="New name")
        description: Optional[str] = Field(None, description="New description")
        enabled: Optional[bool] = Field(None, description="New enabled state")
        rules: Optional[List[Dict[str, Any]]] = Field(None, description="New rules")
        applies_to: Optional[List[str]] = Field(None, description="New scope")

    class ClassifyResourceRequest(BaseModel):
        """Request body for classifying a resource."""
        resource_id: str = Field(..., description="Resource ID")
        resource_type: str = Field(..., description="Resource type")
        tenant_id: str = Field(..., description="Tenant ID")
        classification: str = Field("internal", description="Initial classification")
        attributes: Optional[Dict[str, Any]] = Field(None, description="Resource attributes")

    class AddRegoRequest(BaseModel):
        """Request body for adding a Rego policy."""
        policy_id: str = Field(..., description="Rego policy ID")
        rego_source: str = Field(..., description="Rego source code")

    class SimulateRequest(BaseModel):
        """Request body for policy simulation."""
        requests: List[Dict[str, Any]] = Field(..., description="Test requests")
        policy_ids: Optional[List[str]] = Field(None, description="Specific policies")

    class ComplianceReportRequest(BaseModel):
        """Request body for generating a compliance report."""
        tenant_id: str = Field(..., description="Tenant ID")
        start_date: str = Field(..., description="ISO start date")
        end_date: str = Field(..., description="ISO end date")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    router = APIRouter(
        prefix="/api/v1/access-guard",
        tags=["access-guard"],
    )
else:
    router = None  # type: ignore[assignment]


def _get_service(request: Request) -> Any:
    """Extract AccessGuardService from app state.

    Args:
        request: FastAPI request object.

    Returns:
        AccessGuardService instance.

    Raises:
        HTTPException: If the service is not configured.
    """
    service = getattr(request.app.state, "access_guard_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Access Guard service not configured",
        )
    return service


# ---------------------------------------------------------------------------
# Endpoint definitions
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    # 1. Health check
    @router.get("/health", summary="Access Guard health check")
    async def health_check(request: Request) -> Dict[str, Any]:
        """Return access guard service health status."""
        service = _get_service(request)
        return {
            "status": "healthy",
            "started": service._started,
            "policies_loaded": service.policy_engine.count,
            "audit_events": service.audit_logger.count,
        }

    # 2. Check access
    @router.post("/check", summary="Check access for a request")
    async def check_access(body: CheckAccessRequest, request: Request) -> Dict[str, Any]:
        """Evaluate an access request against loaded policies."""
        from greenlang.access_guard.models import (
            AccessRequest, Principal, Resource,
        )
        service = _get_service(request)
        try:
            principal = Principal(**body.principal)
            resource = Resource(**body.resource)
            access_req = AccessRequest(
                principal=principal,
                resource=resource,
                action=body.action,
                context=body.context or {},
                source_ip=body.source_ip,
            )
            result = service.check_access(access_req)
            return result.model_dump(mode="json")
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # 3. List policies
    @router.get("/policies", summary="List all policies")
    async def list_policies(
        request: Request,
        tenant_id: Optional[str] = Query(None),
        resource_type: Optional[str] = Query(None),
    ) -> List[Dict[str, Any]]:
        """List policies with optional filters."""
        service = _get_service(request)
        policies = service.policy_engine.list_policies(tenant_id, resource_type)
        return [p.model_dump(mode="json") for p in policies]

    # 4. Add policy
    @router.post("/policies", summary="Add a policy", status_code=201)
    async def add_policy(body: AddPolicyRequest, request: Request) -> Dict[str, Any]:
        """Add a new policy to the engine."""
        from greenlang.access_guard.models import Policy, PolicyRule
        service = _get_service(request)
        try:
            rules = [PolicyRule(**r) for r in body.rules]
            policy = Policy(
                policy_id=body.policy_id,
                name=body.name,
                description=body.description,
                version=body.version,
                enabled=body.enabled,
                rules=rules,
                parent_policy_id=body.parent_policy_id,
                tenant_id=body.tenant_id,
                applies_to=body.applies_to or [],
                created_by=body.created_by,
            )
            policy_hash = service.policy_engine.add_policy(policy)
            service.provenance.record(
                "policy", body.policy_id, "create", policy_hash,
                user_id=body.created_by or "system",
            )
            return {
                "policy_id": body.policy_id,
                "provenance_hash": policy_hash,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # 5. Get policy
    @router.get("/policies/{policy_id}", summary="Get a policy by ID")
    async def get_policy(policy_id: str, request: Request) -> Dict[str, Any]:
        """Get a specific policy."""
        service = _get_service(request)
        policy = service.policy_engine.get_policy(policy_id)
        if policy is None:
            raise HTTPException(status_code=404, detail=f"Policy not found: {policy_id}")
        return policy.model_dump(mode="json")

    # 6. Update policy
    @router.put("/policies/{policy_id}", summary="Update a policy")
    async def update_policy(
        policy_id: str, body: UpdatePolicyRequest, request: Request,
    ) -> Dict[str, Any]:
        """Update an existing policy."""
        from greenlang.access_guard.models import PolicyRule
        service = _get_service(request)
        try:
            rules = None
            if body.rules is not None:
                rules = [PolicyRule(**r) for r in body.rules]
            policy = service.policy_engine.update_policy(
                policy_id,
                name=body.name,
                description=body.description,
                enabled=body.enabled,
                rules=rules,
                applies_to=body.applies_to,
            )
            service.provenance.record(
                "policy", policy_id, "update", policy.provenance_hash,
            )
            return policy.model_dump(mode="json")
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Policy not found: {policy_id}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # 7. Delete policy
    @router.delete("/policies/{policy_id}", summary="Delete a policy")
    async def delete_policy(policy_id: str, request: Request) -> Dict[str, Any]:
        """Remove a policy from the engine."""
        service = _get_service(request)
        removed = service.policy_engine.remove_policy(policy_id)
        if not removed:
            raise HTTPException(status_code=404, detail=f"Policy not found: {policy_id}")
        service.provenance.record(
            "policy", policy_id, "delete", "", user_id="system",
        )
        return {"removed": True, "policy_id": policy_id}

    # 8. Get effective rules
    @router.get("/rules/effective", summary="Get effective rules")
    async def get_effective_rules(
        request: Request,
        tenant_id: Optional[str] = Query(None),
        resource_type: Optional[str] = Query(None),
    ) -> List[Dict[str, Any]]:
        """Get all effective rules for a tenant/resource type."""
        service = _get_service(request)
        rules = service.policy_engine.get_effective_rules(tenant_id, resource_type)
        return [r.model_dump(mode="json") for r in rules]

    # 9. Classify resource
    @router.post("/classify", summary="Classify a resource")
    async def classify_resource(
        body: ClassifyResourceRequest, request: Request,
    ) -> Dict[str, Any]:
        """Classify a resource by sensitivity level."""
        from greenlang.access_guard.models import DataClassification, Resource
        service = _get_service(request)
        try:
            resource = Resource(
                resource_id=body.resource_id,
                resource_type=body.resource_type,
                tenant_id=body.tenant_id,
                classification=DataClassification(body.classification),
                attributes=body.attributes or {},
            )
            level = service.classifier.classify(resource)
            return {
                "resource_id": body.resource_id,
                "classification": level.value,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # 10. Get rate limit quota
    @router.get("/rate-limits/{tenant_id}/{principal_id}", summary="Get rate limit quota")
    async def get_rate_limit_quota(
        tenant_id: str, principal_id: str, request: Request,
        role: Optional[str] = Query(None),
    ) -> Dict[str, Any]:
        """Get remaining rate limit quota for a principal."""
        service = _get_service(request)
        quota = service.rate_limiter.get_remaining_quota(
            tenant_id, principal_id, role,
        )
        return {"tenant_id": tenant_id, "principal_id": principal_id, **quota}

    # 11. Reset rate limits
    @router.delete(
        "/rate-limits/{tenant_id}/{principal_id}",
        summary="Reset rate limits",
    )
    async def reset_rate_limits(
        tenant_id: str, principal_id: str, request: Request,
    ) -> Dict[str, Any]:
        """Reset rate limits for a principal."""
        service = _get_service(request)
        service.rate_limiter.reset_limits(tenant_id, principal_id)
        return {"reset": True, "tenant_id": tenant_id, "principal_id": principal_id}

    # 12. Get audit events
    @router.get("/audit/events", summary="Get audit events")
    async def get_audit_events(
        request: Request,
        tenant_id: Optional[str] = Query(None),
        event_type: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> List[Dict[str, Any]]:
        """Get audit events with optional filters."""
        service = _get_service(request)
        events = service.audit_logger.get_events(
            tenant_id=tenant_id,
            event_type=event_type,
            limit=limit,
            offset=offset,
        )
        return [e.model_dump(mode="json") for e in events]

    # 13. Get audit event by ID
    @router.get("/audit/events/{event_id}", summary="Get an audit event by ID")
    async def get_audit_event(event_id: str, request: Request) -> Dict[str, Any]:
        """Get a single audit event."""
        service = _get_service(request)
        event = service.audit_logger.get_event(event_id)
        if event is None:
            raise HTTPException(
                status_code=404, detail=f"Audit event not found: {event_id}",
            )
        return event.model_dump(mode="json")

    # 14. Generate compliance report
    @router.post("/audit/compliance-report", summary="Generate compliance report")
    async def generate_compliance_report(
        body: ComplianceReportRequest, request: Request,
    ) -> Dict[str, Any]:
        """Generate a compliance report for a tenant."""
        from datetime import datetime
        service = _get_service(request)
        try:
            start = datetime.fromisoformat(body.start_date)
            end = datetime.fromisoformat(body.end_date)
            report = service.audit_logger.generate_compliance_report(
                body.tenant_id, start, end,
            )
            return report.model_dump(mode="json")
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # 15. Add Rego policy
    @router.post("/opa/policies", summary="Add a Rego policy", status_code=201)
    async def add_rego_policy(body: AddRegoRequest, request: Request) -> Dict[str, Any]:
        """Add an OPA Rego policy."""
        service = _get_service(request)
        try:
            policy_hash = service.opa_client.add_rego_policy(
                body.policy_id, body.rego_source,
            )
            service.provenance.record(
                "rego", body.policy_id, "create", policy_hash,
            )
            return {"policy_id": body.policy_id, "hash": policy_hash}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # 16. List Rego policies
    @router.get("/opa/policies", summary="List Rego policies")
    async def list_rego_policies(request: Request) -> List[Dict[str, Any]]:
        """List all registered Rego policies."""
        service = _get_service(request)
        return service.opa_client.list_rego_policies()

    # 17. Delete Rego policy
    @router.delete("/opa/policies/{policy_id}", summary="Delete a Rego policy")
    async def delete_rego_policy(policy_id: str, request: Request) -> Dict[str, Any]:
        """Remove a Rego policy."""
        service = _get_service(request)
        removed = service.opa_client.remove_rego_policy(policy_id)
        if not removed:
            raise HTTPException(
                status_code=404, detail=f"Rego policy not found: {policy_id}",
            )
        service.provenance.record("rego", policy_id, "delete", "")
        return {"removed": True, "policy_id": policy_id}

    # 18. Simulate policies
    @router.post("/simulate", summary="Simulate policy evaluation")
    async def simulate_policies(
        body: SimulateRequest, request: Request,
    ) -> Dict[str, Any]:
        """Run a policy simulation against test requests."""
        from greenlang.access_guard.models import AccessRequest, Principal, Resource
        service = _get_service(request)
        try:
            test_requests = []
            for req_data in body.requests:
                principal = Principal(**req_data.get("principal", {}))
                resource = Resource(**req_data.get("resource", {}))
                access_req = AccessRequest(
                    principal=principal,
                    resource=resource,
                    action=req_data.get("action", "read"),
                    context=req_data.get("context", {}),
                )
                test_requests.append(access_req)

            # Temporarily enable simulation
            original_sim = service.config.simulation_mode
            service.config.simulation_mode = True
            try:
                results = []
                for req in test_requests:
                    result = service.check_access(req)
                    results.append(result.model_dump(mode="json"))
            finally:
                service.config.simulation_mode = original_sim

            return {
                "test_requests": len(test_requests),
                "results": results,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # 19. Get provenance chain
    @router.get(
        "/provenance/{entity_id}",
        summary="Get provenance chain for an entity",
    )
    async def get_provenance_chain(
        entity_id: str, request: Request,
    ) -> List[Dict[str, Any]]:
        """Get the provenance chain for a policy or entity."""
        service = _get_service(request)
        chain = service.provenance.get_chain(entity_id)
        return [e.model_dump(mode="json") for e in chain]

    # 20. Get metrics summary
    @router.get("/metrics", summary="Get metrics summary")
    async def get_metrics(request: Request) -> Dict[str, Any]:
        """Get service metrics summary."""
        service = _get_service(request)
        return service.get_metrics()


__all__ = [
    "router",
    "FASTAPI_AVAILABLE",
]
