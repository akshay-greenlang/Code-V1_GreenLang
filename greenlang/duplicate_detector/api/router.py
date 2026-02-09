# -*- coding: utf-8 -*-
"""
Duplicate Detection REST API Router - AGENT-DATA-011

FastAPI router providing 20 endpoints for record deduplication,
fingerprinting, blocking, similarity comparison, match classification,
cluster resolution, record merging, pipeline execution, job management,
rule management, and health/statistics.

All endpoints are mounted under ``/api/v1/dedup``.

Endpoints:
    1.  POST   /jobs                  - Create dedup job
    2.  GET    /jobs                  - List jobs
    3.  GET    /jobs/{job_id}         - Get job details
    4.  DELETE /jobs/{job_id}         - Cancel job
    5.  POST   /fingerprint           - Fingerprint records
    6.  POST   /block                 - Create blocks
    7.  POST   /compare               - Compare pairs
    8.  POST   /classify              - Classify matches
    9.  GET    /matches               - List matches
    10. GET    /matches/{match_id}    - Get match details
    11. POST   /clusters              - Form clusters
    12. GET    /clusters              - List clusters
    13. GET    /clusters/{cluster_id} - Get cluster details
    14. POST   /merge                 - Execute merge
    15. GET    /merge/{merge_id}      - Get merge result
    16. POST   /pipeline              - Run full pipeline
    17. POST   /rules                 - Create dedup rule
    18. GET    /rules                 - List rules
    19. GET    /health                - Health check
    20. GET    /stats                 - Statistics

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-011 Duplicate Detection Agent (GL-DATA-X-014)
Status: Production Ready
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import (no `from __future__ import annotations` here)
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None  # type: ignore[assignment, misc]
    logger.warning(
        "FastAPI not available; duplicate detector router is None"
    )


# ---------------------------------------------------------------------------
# Pydantic request/response models (only when FastAPI is available)
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class CreateJobBody(BaseModel):
        """Request body for creating a dedup job."""
        dataset_ids: List[str] = Field(
            ..., description="List of dataset identifiers to deduplicate",
        )
        rule_id: Optional[str] = Field(
            None, description="Optional dedup rule ID to apply",
        )

    class FingerprintBody(BaseModel):
        """Request body for fingerprinting records."""
        records: List[Dict[str, Any]] = Field(
            ..., description="List of record dicts to fingerprint",
        )
        field_set: Optional[List[str]] = Field(
            None, description="Fields to include in fingerprint (all if omitted)",
        )
        algorithm: Optional[str] = Field(
            None, description="Algorithm (sha256, simhash, minhash)",
        )

    class BlockBody(BaseModel):
        """Request body for creating blocks."""
        records: List[Dict[str, Any]] = Field(
            ..., description="List of record dicts",
        )
        strategy: Optional[str] = Field(
            None,
            description="Blocking strategy (sorted_neighborhood, standard, canopy, none)",
        )
        key_fields: Optional[List[str]] = Field(
            None, description="Fields to use for blocking key generation",
        )

    class CompareBody(BaseModel):
        """Request body for comparing pairs."""
        block_results: Dict[str, Any] = Field(
            ..., description="Block results dict containing candidate pairs under 'pairs' key",
        )
        field_configs: Optional[List[Dict[str, Any]]] = Field(
            None, description="Per-field comparison config with algorithm and weight",
        )

    class ClassifyBody(BaseModel):
        """Request body for classifying matches."""
        comparisons: List[Dict[str, Any]] = Field(
            ..., description="List of comparison result dicts with 'overall_score'",
        )
        thresholds: Optional[Dict[str, float]] = Field(
            None, description="Override thresholds dict with 'match' and 'possible' keys",
        )

    class ClusterBody(BaseModel):
        """Request body for forming clusters."""
        matches: List[Dict[str, Any]] = Field(
            ..., description="List of match dicts with record_a_id, record_b_id",
        )
        algorithm: Optional[str] = Field(
            None, description="Clustering algorithm (union_find, connected_components)",
        )

    class MergeBody(BaseModel):
        """Request body for executing merge."""
        clusters: List[Dict[str, Any]] = Field(
            ..., description="List of cluster dicts with 'members'",
        )
        records: List[Dict[str, Any]] = Field(
            ..., description="Full record list",
        )
        strategy: Optional[str] = Field(
            None,
            description="Merge strategy (keep_first, keep_latest, keep_most_complete, merge_fields, golden_record)",
        )

    class PipelineBody(BaseModel):
        """Request body for running the full dedup pipeline."""
        records: List[Dict[str, Any]] = Field(
            ..., description="List of record dicts to deduplicate",
        )
        rule: Optional[Dict[str, Any]] = Field(
            None, description="Optional dedup rule configuration",
        )
        options: Optional[Dict[str, Any]] = Field(
            None, description="Optional pipeline options overriding config defaults",
        )

    class CreateRuleBody(BaseModel):
        """Request body for creating a dedup rule."""
        name: str = Field(
            ..., description="Rule display name",
        )
        description: Optional[str] = Field(
            None, description="Rule description",
        )
        field_weights: Optional[List[Dict[str, Any]]] = Field(
            None, description="Per-field weight configurations",
        )
        match_threshold: Optional[float] = Field(
            None, description="Match classification threshold",
        )
        possible_threshold: Optional[float] = Field(
            None, description="Possible classification threshold",
        )
        blocking_strategy: Optional[str] = Field(
            None, description="Blocking strategy to use",
        )
        blocking_key_fields: Optional[List[str]] = Field(
            None, description="Fields for blocking key generation",
        )
        merge_strategy: Optional[str] = Field(
            None, description="Merge strategy",
        )
        is_active: bool = Field(
            default=True, description="Whether the rule is active",
        )

    class ReportBody(BaseModel):
        """Request body for generating a report."""
        job_id: str = Field(
            ..., description="Job identifier to generate report for",
        )
        format: str = Field(
            default="json", description="Report format (json, markdown, text)",
        )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    router = APIRouter(
        prefix="/api/v1/dedup",
        tags=["Duplicate Detection"],
    )
else:
    router = None  # type: ignore[assignment]


def _get_service(request: Request) -> Any:
    """Extract DuplicateDetectorService from app state.

    Args:
        request: FastAPI request object.

    Returns:
        DuplicateDetectorService instance.

    Raises:
        HTTPException: If service is not configured.
    """
    service = getattr(
        request.app.state, "duplicate_detector_service", None,
    )
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Duplicate detector service not configured",
        )
    return service


if FASTAPI_AVAILABLE:

    # ------------------------------------------------------------------
    # 1. Create dedup job
    # ------------------------------------------------------------------
    @router.post("/jobs")
    async def create_job(
        body: CreateJobBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Create a new deduplication job."""
        service = _get_service(request)
        try:
            result = service.create_dedup_job(
                dataset_ids=body.dataset_ids,
                rule_id=body.rule_id,
            )
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 2. List jobs
    # ------------------------------------------------------------------
    @router.get("/jobs")
    async def list_jobs(
        status: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List deduplication jobs with optional status filter."""
        service = _get_service(request)
        return service.list_jobs(
            status=status,
            limit=limit,
            offset=offset,
        )

    # ------------------------------------------------------------------
    # 3. Get job details
    # ------------------------------------------------------------------
    @router.get("/jobs/{job_id}")
    async def get_job(
        job_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a deduplication job by ID."""
        service = _get_service(request)
        job = service.get_job(job_id)
        if job is None:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found",
            )
        return job

    # ------------------------------------------------------------------
    # 4. Cancel job
    # ------------------------------------------------------------------
    @router.delete("/jobs/{job_id}")
    async def cancel_job(
        job_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Cancel a deduplication job."""
        service = _get_service(request)
        result = service.cancel_job(job_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found",
            )
        return result

    # ------------------------------------------------------------------
    # 5. Fingerprint records
    # ------------------------------------------------------------------
    @router.post("/fingerprint")
    async def fingerprint_records(
        body: FingerprintBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Fingerprint records using a deterministic hashing algorithm."""
        service = _get_service(request)
        try:
            result = service.fingerprint_records(
                records=body.records,
                field_set=body.field_set,
                algorithm=body.algorithm,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 6. Create blocks
    # ------------------------------------------------------------------
    @router.post("/block")
    async def create_blocks(
        body: BlockBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Create blocking partitions for candidate pair generation."""
        service = _get_service(request)
        try:
            result = service.create_blocks(
                records=body.records,
                strategy=body.strategy,
                key_fields=body.key_fields,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 7. Compare pairs
    # ------------------------------------------------------------------
    @router.post("/compare")
    async def compare_pairs(
        body: CompareBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Compare candidate record pairs for similarity."""
        service = _get_service(request)
        try:
            result = service.compare_pairs(
                block_results=body.block_results,
                field_configs=body.field_configs,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 8. Classify matches
    # ------------------------------------------------------------------
    @router.post("/classify")
    async def classify_matches(
        body: ClassifyBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Classify comparison results into match/possible/non-match."""
        service = _get_service(request)
        try:
            result = service.classify_matches(
                comparisons=body.comparisons,
                thresholds=body.thresholds,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 9. List matches
    # ------------------------------------------------------------------
    @router.get("/matches")
    async def list_matches(
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List match results across all classification runs."""
        service = _get_service(request)
        matches = list(service._matches.values())
        page = matches[offset:offset + limit]
        return {
            "matches": page,
            "count": len(page),
            "total": len(matches),
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------
    # 10. Get match details
    # ------------------------------------------------------------------
    @router.get("/matches/{match_id}")
    async def get_match_details(
        match_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get details of a specific match."""
        service = _get_service(request)
        match = service.get_match_details(match_id)
        if match is None:
            raise HTTPException(
                status_code=404,
                detail=f"Match {match_id} not found",
            )
        return match

    # ------------------------------------------------------------------
    # 11. Form clusters
    # ------------------------------------------------------------------
    @router.post("/clusters")
    async def form_clusters(
        body: ClusterBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Form duplicate clusters from matched pairs."""
        service = _get_service(request)
        try:
            result = service.form_clusters(
                matches=body.matches,
                algorithm=body.algorithm,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 12. List clusters
    # ------------------------------------------------------------------
    @router.get("/clusters")
    async def list_clusters(
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List duplicate clusters across all clustering runs."""
        service = _get_service(request)
        clusters = list(service._clusters.values())
        page = clusters[offset:offset + limit]
        return {
            "clusters": page,
            "count": len(page),
            "total": len(clusters),
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------
    # 13. Get cluster details
    # ------------------------------------------------------------------
    @router.get("/clusters/{cluster_id}")
    async def get_cluster_details(
        cluster_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get details of a specific cluster."""
        service = _get_service(request)
        cluster = service.get_cluster_details(cluster_id)
        if cluster is None:
            raise HTTPException(
                status_code=404,
                detail=f"Cluster {cluster_id} not found",
            )
        return cluster

    # ------------------------------------------------------------------
    # 14. Execute merge
    # ------------------------------------------------------------------
    @router.post("/merge")
    async def merge_duplicates(
        body: MergeBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Merge duplicate records within clusters into golden records."""
        service = _get_service(request)
        try:
            result = service.merge_duplicates(
                clusters=body.clusters,
                records=body.records,
                strategy=body.strategy,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 15. Get merge result
    # ------------------------------------------------------------------
    @router.get("/merge/{merge_id}")
    async def get_merge_result(
        merge_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a merge result by ID."""
        service = _get_service(request)
        result = service.get_merge_result(merge_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Merge result {merge_id} not found",
            )
        return result.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 16. Run full pipeline
    # ------------------------------------------------------------------
    @router.post("/pipeline")
    async def run_pipeline(
        body: PipelineBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Run the full deduplication pipeline end-to-end."""
        service = _get_service(request)
        try:
            result = service.run_pipeline(
                records=body.records,
                rule=body.rule,
                options=body.options,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 17. Create dedup rule
    # ------------------------------------------------------------------
    @router.post("/rules")
    async def create_rule(
        body: CreateRuleBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Create a new deduplication rule."""
        service = _get_service(request)
        try:
            rule_config = body.model_dump(exclude_none=True)
            result = service.create_rule(rule_config=rule_config)
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 18. List rules
    # ------------------------------------------------------------------
    @router.get("/rules")
    async def list_rules(
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List deduplication rules."""
        service = _get_service(request)
        return service.list_rules(
            limit=limit,
            offset=offset,
        )

    # ------------------------------------------------------------------
    # 19. Health check
    # ------------------------------------------------------------------
    @router.get("/health")
    async def health(
        request: Request,
    ) -> Dict[str, Any]:
        """Duplicate detector service health check endpoint."""
        service = _get_service(request)
        return service.health_check()

    # ------------------------------------------------------------------
    # 20. Statistics
    # ------------------------------------------------------------------
    @router.get("/stats")
    async def stats(
        request: Request,
    ) -> Dict[str, Any]:
        """Get aggregated duplicate detection statistics."""
        service = _get_service(request)
        result = service.get_statistics()
        return result.model_dump(mode="json")


__all__ = [
    "router",
]
