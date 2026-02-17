# -*- coding: utf-8 -*-
"""
Validation Rule Engine REST API Router - AGENT-DATA-019

FastAPI router providing 20 REST API endpoints for the Validation
Rule Engine service at ``/api/v1/validation-rules``.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-019 Validation Rule Engine (GL-DATA-X-022)
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, HTTPException, Query
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None  # type: ignore[assignment, misc]

router: Optional[Any] = None

if FASTAPI_AVAILABLE:
    router = APIRouter(prefix="/api/v1/validation-rules", tags=["validation-rules"])

    # ------------------------------------------------------------------
    # Lazy service accessor
    # ------------------------------------------------------------------
    def _get_service():
        from greenlang.validation_rule_engine.setup import get_validation_rule_engine
        svc = get_validation_rule_engine()
        if svc is None:
            raise HTTPException(status_code=503, detail="Validation Rule Engine service not initialized")
        return svc

    # 1. POST /rules
    @router.post("/rules")
    async def register_rule(body: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new validation rule."""
        svc = _get_service()
        return svc.register_rule(**body)

    # 2. GET /rules
    @router.get("/rules")
    async def list_rules(
        rule_type: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> List[Dict[str, Any]]:
        """List registered validation rules with optional filters."""
        svc = _get_service()
        return svc.search_rules(
            rule_type=rule_type, severity=severity,
            status=status, tag=tag,
            limit=limit, offset=offset,
        )

    # 3. GET /rules/{rule_id}
    @router.get("/rules/{rule_id}")
    async def get_rule(rule_id: str) -> Dict[str, Any]:
        """Get validation rule details."""
        svc = _get_service()
        result = svc.get_rule(rule_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Rule {rule_id} not found")
        return result

    # 4. PUT /rules/{rule_id}
    @router.put("/rules/{rule_id}")
    async def update_rule(rule_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing validation rule."""
        svc = _get_service()
        result = svc.update_rule(rule_id, **body)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Rule {rule_id} not found")
        return result

    # 5. DELETE /rules/{rule_id}
    @router.delete("/rules/{rule_id}")
    async def delete_rule(rule_id: str) -> Dict[str, Any]:
        """Delete a validation rule (soft delete)."""
        svc = _get_service()
        success = svc.delete_rule(rule_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Rule {rule_id} not found")
        return {"status": "deleted", "rule_id": rule_id}

    # 6. POST /rule-sets
    @router.post("/rule-sets")
    async def create_rule_set(body: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new rule set (named collection of rules)."""
        svc = _get_service()
        return svc.create_rule_set(**body)

    # 7. GET /rule-sets
    @router.get("/rule-sets")
    async def list_rule_sets(
        pack_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> List[Dict[str, Any]]:
        """List rule sets with optional filters."""
        svc = _get_service()
        return svc.list_rule_sets(
            pack_type=pack_type, status=status,
            limit=limit, offset=offset,
        )

    # 8. GET /rule-sets/{set_id}
    @router.get("/rule-sets/{set_id}")
    async def get_rule_set(set_id: str) -> Dict[str, Any]:
        """Get rule set details including member rules."""
        svc = _get_service()
        result = svc.get_rule_set(set_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Rule set {set_id} not found")
        return result

    # 9. PUT /rule-sets/{set_id}
    @router.put("/rule-sets/{set_id}")
    async def update_rule_set(set_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Update a rule set (add/remove rules, change metadata)."""
        svc = _get_service()
        result = svc.update_rule_set(set_id, **body)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Rule set {set_id} not found")
        return result

    # 10. DELETE /rule-sets/{set_id}
    @router.delete("/rule-sets/{set_id}")
    async def delete_rule_set(set_id: str) -> Dict[str, Any]:
        """Delete a rule set (soft delete)."""
        svc = _get_service()
        success = svc.delete_rule_set(set_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Rule set {set_id} not found")
        return {"status": "deleted", "set_id": set_id}

    # 11. POST /evaluate
    @router.post("/evaluate")
    async def evaluate_rules(body: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate validation rules against a dataset or record."""
        svc = _get_service()
        return svc.evaluate(**body)

    # 12. POST /evaluate/batch
    @router.post("/evaluate/batch")
    async def batch_evaluate(body: Dict[str, Any]) -> Dict[str, Any]:
        """Batch evaluate rules across multiple datasets."""
        svc = _get_service()
        return svc.batch_evaluate(**body)

    # 13. GET /evaluations/{eval_id}
    @router.get("/evaluations/{eval_id}")
    async def get_evaluation(eval_id: str) -> Dict[str, Any]:
        """Get evaluation result details."""
        svc = _get_service()
        result = svc.get_evaluation(eval_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Evaluation {eval_id} not found")
        return result

    # 14. POST /conflicts/detect
    @router.post("/conflicts/detect")
    async def detect_conflicts(body: Dict[str, Any]) -> Dict[str, Any]:
        """Detect conflicts between rules in a rule set."""
        svc = _get_service()
        return svc.detect_conflicts(**body)

    # 15. GET /conflicts
    @router.get("/conflicts")
    async def list_conflicts(
        set_id: Optional[str] = None,
        conflict_type: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> List[Dict[str, Any]]:
        """List detected rule conflicts with optional filters."""
        svc = _get_service()
        return svc.list_conflicts(
            set_id=set_id, conflict_type=conflict_type,
            severity=severity, limit=limit, offset=offset,
        )

    # 16. POST /packs/{pack_name}/apply
    @router.post("/packs/{pack_name}/apply")
    async def apply_pack(pack_name: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a rule pack (import and activate its rule sets)."""
        svc = _get_service()
        return svc.apply_pack(pack_name, **body)

    # 17. GET /packs
    @router.get("/packs")
    async def list_packs(
        framework: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> List[Dict[str, Any]]:
        """List available rule packs."""
        svc = _get_service()
        return svc.list_packs(
            framework=framework, limit=limit, offset=offset,
        )

    # 18. POST /reports
    @router.post("/reports")
    async def generate_report(body: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a validation report."""
        svc = _get_service()
        return svc.generate_report(**body)

    # 19. POST /pipeline
    @router.post("/pipeline")
    async def run_pipeline(body: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full validation rule engine pipeline."""
        svc = _get_service()
        return svc.run_pipeline(**body)

    # 20. GET /health
    @router.get("/health")
    async def health_check() -> Dict[str, Any]:
        """Health check for the validation rule engine service."""
        svc = _get_service()
        return svc.get_health()


__all__ = ["router"]
