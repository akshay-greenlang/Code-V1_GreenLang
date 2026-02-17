# -*- coding: utf-8 -*-
"""
Data Lineage Tracker REST API Router - AGENT-DATA-018

FastAPI router providing 20 REST API endpoints for the Data Lineage
Tracker service at ``/api/v1/data-lineage``.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-018 Data Lineage Tracker (GL-DATA-X-021)
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
    router = APIRouter(prefix="/api/v1/data-lineage", tags=["data-lineage"])

    # ------------------------------------------------------------------
    # Lazy service accessor
    # ------------------------------------------------------------------
    def _get_service():
        from greenlang.data_lineage_tracker.setup import get_data_lineage_tracker
        svc = get_data_lineage_tracker()
        if svc is None:
            raise HTTPException(status_code=503, detail="Data Lineage Tracker service not initialized")
        return svc

    # 1. POST /assets
    @router.post("/assets")
    async def register_asset(body: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new data asset."""
        svc = _get_service()
        return svc.register_asset(**body)

    # 2. GET /assets
    @router.get("/assets")
    async def list_assets(
        asset_type: Optional[str] = None,
        owner: Optional[str] = None,
        classification: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> List[Dict[str, Any]]:
        """List registered assets with optional filters."""
        svc = _get_service()
        return svc.search_assets(
            asset_type=asset_type, owner=owner,
            classification=classification, status=status,
            limit=limit, offset=offset,
        )

    # 3. GET /assets/{asset_id}
    @router.get("/assets/{asset_id}")
    async def get_asset(asset_id: str) -> Dict[str, Any]:
        """Get asset details."""
        svc = _get_service()
        result = svc.get_asset(asset_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Asset {asset_id} not found")
        return result

    # 4. PUT /assets/{asset_id}
    @router.put("/assets/{asset_id}")
    async def update_asset(asset_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Update asset metadata."""
        svc = _get_service()
        result = svc.update_asset(asset_id, **body)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Asset {asset_id} not found")
        return result

    # 5. DELETE /assets/{asset_id}
    @router.delete("/assets/{asset_id}")
    async def delete_asset(asset_id: str) -> Dict[str, Any]:
        """Deregister asset (soft delete)."""
        svc = _get_service()
        success = svc.delete_asset(asset_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Asset {asset_id} not found")
        return {"status": "archived", "asset_id": asset_id}

    # 6. POST /transformations
    @router.post("/transformations")
    async def record_transformation(body: Dict[str, Any]) -> Dict[str, Any]:
        """Record a transformation event."""
        svc = _get_service()
        return svc.record_transformation(**body)

    # 7. GET /transformations
    @router.get("/transformations")
    async def list_transformations(
        transformation_type: Optional[str] = None,
        agent_id: Optional[str] = None,
        pipeline_id: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> List[Dict[str, Any]]:
        """List transformation events."""
        svc = _get_service()
        return svc.list_transformations(
            transformation_type=transformation_type,
            agent_id=agent_id, pipeline_id=pipeline_id,
            limit=limit, offset=offset,
        )

    # 8. POST /edges
    @router.post("/edges")
    async def create_edge(body: Dict[str, Any]) -> Dict[str, Any]:
        """Create a lineage edge."""
        svc = _get_service()
        return svc.create_edge(**body)

    # 9. GET /edges
    @router.get("/edges")
    async def list_edges(
        source_asset_id: Optional[str] = None,
        target_asset_id: Optional[str] = None,
        edge_type: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1000),
    ) -> List[Dict[str, Any]]:
        """List lineage edges."""
        svc = _get_service()
        return svc.list_edges(
            source_asset_id=source_asset_id,
            target_asset_id=target_asset_id,
            edge_type=edge_type, limit=limit,
        )

    # 10. GET /graph
    @router.get("/graph")
    async def get_graph() -> Dict[str, Any]:
        """Get the full lineage graph."""
        svc = _get_service()
        return svc.get_graph()

    # 11. GET /graph/subgraph/{asset_id}
    @router.get("/graph/subgraph/{asset_id}")
    async def get_subgraph(
        asset_id: str,
        depth: int = Query(3, ge=1, le=50),
    ) -> Dict[str, Any]:
        """Extract subgraph centered on asset."""
        svc = _get_service()
        return svc.get_subgraph(asset_id, depth=depth)

    # 12. GET /backward/{asset_id}
    @router.get("/backward/{asset_id}")
    async def backward_lineage(
        asset_id: str,
        max_depth: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Backward lineage traversal to sources."""
        svc = _get_service()
        return svc.analyze_backward(asset_id, max_depth=max_depth)

    # 13. GET /forward/{asset_id}
    @router.get("/forward/{asset_id}")
    async def forward_lineage(
        asset_id: str,
        max_depth: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Forward lineage traversal to consumers."""
        svc = _get_service()
        return svc.analyze_forward(asset_id, max_depth=max_depth)

    # 14. POST /impact
    @router.post("/impact")
    async def run_impact_analysis(body: Dict[str, Any]) -> Dict[str, Any]:
        """Run impact analysis for an asset."""
        svc = _get_service()
        asset_id = body.get("asset_id", "")
        direction = body.get("direction", "forward")
        max_depth = body.get("max_depth")
        return svc.run_impact_analysis(asset_id, direction=direction, max_depth=max_depth)

    # 15. POST /validate
    @router.post("/validate")
    async def validate_lineage(body: Dict[str, Any]) -> Dict[str, Any]:
        """Validate lineage completeness and consistency."""
        svc = _get_service()
        scope = body.get("scope", "full")
        include_freshness = body.get("include_freshness", True)
        include_coverage = body.get("include_coverage", True)
        return svc.validate_lineage(
            scope=scope, include_freshness=include_freshness,
            include_coverage=include_coverage,
        )

    # 16. GET /validate/{validation_id}
    @router.get("/validate/{validation_id}")
    async def get_validation(validation_id: str) -> Dict[str, Any]:
        """Get validation result details."""
        svc = _get_service()
        result = svc.get_validation(validation_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Validation {validation_id} not found")
        return result

    # 17. POST /reports
    @router.post("/reports")
    async def generate_report(body: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a lineage report."""
        svc = _get_service()
        return svc.generate_report(**body)

    # 18. POST /pipeline
    @router.post("/pipeline")
    async def run_pipeline(body: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full lineage tracking pipeline."""
        svc = _get_service()
        return svc.run_pipeline(**body)

    # 19. GET /health
    @router.get("/health")
    async def health_check() -> Dict[str, Any]:
        """Health check."""
        svc = _get_service()
        return svc.get_health()

    # 20. GET /stats
    @router.get("/stats")
    async def get_stats() -> Dict[str, Any]:
        """Service statistics."""
        svc = _get_service()
        return svc.get_statistics()


__all__ = ["router"]
