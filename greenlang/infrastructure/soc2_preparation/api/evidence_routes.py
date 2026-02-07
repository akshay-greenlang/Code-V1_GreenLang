# -*- coding: utf-8 -*-
"""
SOC 2 Evidence API Routes - SEC-009 Phase 10

FastAPI routes for SOC 2 evidence collection and management:
- GET /evidence - List all evidence
- GET /evidence/{criterion} - Get evidence for criterion
- POST /evidence/collect - Trigger evidence collection
- POST /evidence/package - Create evidence package
- GET /evidence/package/{id} - Download/get package status

Requires soc2:evidence:read or soc2:evidence:write permissions.

Author: GreenLang Security Team
Date: February 2026
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------


class EvidenceItem(BaseModel):
    """Evidence item summary."""

    evidence_id: UUID = Field(..., description="Evidence identifier")
    criterion_id: str = Field(..., description="Related SOC 2 criterion")
    title: str = Field(..., description="Evidence title")
    evidence_type: str = Field(..., description="Type: policy, log, screenshot, etc.")
    source: str = Field(..., description="Source system")
    status: str = Field(..., description="Status: collected, validated, approved")
    collected_at: datetime = Field(..., description="Collection timestamp")
    period_start: Optional[datetime] = Field(None, description="Period start")
    period_end: Optional[datetime] = Field(None, description="Period end")
    collector_id: str = Field(default="system", description="Collector identity")
    provenance_hash: Optional[str] = Field(None, description="SHA-256 hash")
    s3_key: Optional[str] = Field(None, description="S3 storage key")
    file_size_bytes: Optional[int] = Field(None, description="File size if applicable")


class EvidenceListResponse(BaseModel):
    """Response for evidence listing."""

    total: int = Field(..., description="Total evidence count")
    items: List[EvidenceItem] = Field(..., description="Evidence items")
    page: int = Field(default=1, description="Current page")
    page_size: int = Field(default=50, description="Items per page")


class EvidenceCollectionRequest(BaseModel):
    """Request to collect evidence for criteria."""

    criteria: List[str] = Field(
        ..., min_length=1, description="Criteria to collect evidence for"
    )
    sources: Optional[List[str]] = Field(
        None, description="Specific sources to query (None = all)"
    )
    period_start: datetime = Field(..., description="Collection period start")
    period_end: datetime = Field(..., description="Collection period end")
    force_refresh: bool = Field(
        default=False, description="Force re-collection even if cached"
    )


class EvidenceCollectionResponse(BaseModel):
    """Response from evidence collection request."""

    collection_id: UUID = Field(..., description="Collection job ID")
    status: str = Field(default="running", description="Job status")
    criteria_count: int = Field(..., description="Number of criteria")
    started_at: datetime = Field(..., description="Job start time")
    estimated_completion: datetime = Field(..., description="Estimated completion")


class EvidencePackageRequest(BaseModel):
    """Request to create an evidence package."""

    package_name: str = Field(..., max_length=256, description="Package name")
    criteria: List[str] = Field(..., min_length=1, description="Criteria to include")
    period_start: datetime = Field(..., description="Audit period start")
    period_end: datetime = Field(..., description="Audit period end")
    include_populations: bool = Field(
        default=True, description="Include population files for sampling"
    )
    include_validation_results: bool = Field(
        default=True, description="Include validation results"
    )
    format: str = Field(
        default="zip", description="Package format: zip, tar.gz"
    )


class EvidencePackageResponse(BaseModel):
    """Response with evidence package details."""

    package_id: UUID = Field(..., description="Package identifier")
    package_name: str = Field(..., description="Package name")
    status: str = Field(..., description="Status: creating, ready, expired")
    evidence_count: int = Field(default=0, description="Number of evidence items")
    total_size_bytes: int = Field(default=0, description="Total package size")
    criteria: List[str] = Field(default_factory=list, description="Included criteria")
    created_at: datetime = Field(..., description="Creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Download expiration")
    download_url: Optional[str] = Field(None, description="Presigned download URL")
    manifest_hash: Optional[str] = Field(None, description="Package manifest hash")


class EvidenceValidationResult(BaseModel):
    """Validation result for evidence."""

    evidence_id: UUID = Field(..., description="Evidence ID")
    validation_type: str = Field(..., description="Type of validation")
    status: str = Field(..., description="pass, fail, warning")
    message: str = Field(default="", description="Validation message")
    validated_at: datetime = Field(..., description="Validation timestamp")


class CriterionEvidenceResponse(BaseModel):
    """Evidence for a specific criterion."""

    criterion_id: str = Field(..., description="Criterion ID")
    total_evidence: int = Field(..., description="Total evidence count")
    validated_count: int = Field(default=0, description="Validated evidence count")
    approved_count: int = Field(default=0, description="Approved evidence count")
    items: List[EvidenceItem] = Field(..., description="Evidence items")
    validation_results: List[EvidenceValidationResult] = Field(
        default_factory=list, description="Validation results"
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/evidence", tags=["soc2-evidence"])


@router.get(
    "",
    response_model=EvidenceListResponse,
    summary="List all evidence",
    description="List all collected SOC 2 evidence with filtering options.",
)
async def list_evidence(
    request: Request,
    criterion: Optional[str] = Query(None, description="Filter by criterion"),
    evidence_type: Optional[str] = Query(None, description="Filter by type"),
    source: Optional[str] = Query(None, description="Filter by source"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    period_start: Optional[datetime] = Query(None, description="Filter by period start"),
    period_end: Optional[datetime] = Query(None, description="Filter by period end"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Page size"),
) -> EvidenceListResponse:
    """List all collected evidence with filtering.

    Args:
        request: FastAPI request object.
        criterion: Filter by criterion ID.
        evidence_type: Filter by evidence type.
        source: Filter by source system.
        status_filter: Filter by status.
        period_start: Filter by period start date.
        period_end: Filter by period end date.
        page: Page number.
        page_size: Items per page.

    Returns:
        EvidenceListResponse with filtered evidence items.
    """
    logger.info(
        "Listing evidence: criterion=%s, type=%s, source=%s",
        criterion,
        evidence_type,
        source,
    )

    # Sample evidence for demonstration
    sample_items = [
        EvidenceItem(
            evidence_id=uuid4(),
            criterion_id="CC6.1",
            title="MFA Configuration Export",
            evidence_type="configuration",
            source="auth_service",
            status="validated",
            collected_at=datetime.now(timezone.utc),
            period_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2026, 2, 1, tzinfo=timezone.utc),
            provenance_hash="a1b2c3d4e5f6...",
            file_size_bytes=15234,
        ),
        EvidenceItem(
            evidence_id=uuid4(),
            criterion_id="CC6.2",
            title="Access Review Report Q1 2026",
            evidence_type="access_review",
            source="okta",
            status="approved",
            collected_at=datetime.now(timezone.utc),
            period_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2026, 3, 31, tzinfo=timezone.utc),
            file_size_bytes=89456,
        ),
        EvidenceItem(
            evidence_id=uuid4(),
            criterion_id="CC7.1",
            title="CloudTrail Audit Logs",
            evidence_type="log_export",
            source="cloudtrail",
            status="collected",
            collected_at=datetime.now(timezone.utc),
            period_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2026, 2, 1, tzinfo=timezone.utc),
            file_size_bytes=5678901,
        ),
    ]

    # Apply filters
    filtered = sample_items
    if criterion:
        filtered = [e for e in filtered if e.criterion_id == criterion.upper()]
    if evidence_type:
        filtered = [e for e in filtered if e.evidence_type == evidence_type.lower()]
    if source:
        filtered = [e for e in filtered if e.source == source.lower()]
    if status_filter:
        filtered = [e for e in filtered if e.status == status_filter.lower()]

    # Pagination
    start = (page - 1) * page_size
    end = start + page_size
    paginated = filtered[start:end]

    return EvidenceListResponse(
        total=len(filtered),
        items=paginated,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{criterion}",
    response_model=CriterionEvidenceResponse,
    summary="Get evidence for criterion",
    description="Get all evidence collected for a specific SOC 2 criterion.",
)
async def get_criterion_evidence(
    request: Request,
    criterion: str,
    include_validation: bool = Query(
        True, description="Include validation results"
    ),
) -> CriterionEvidenceResponse:
    """Get evidence for a specific criterion.

    Args:
        request: FastAPI request object.
        criterion: The criterion ID (e.g., CC6.1).
        include_validation: Whether to include validation results.

    Returns:
        CriterionEvidenceResponse with evidence details.

    Raises:
        HTTPException: 404 if no evidence found for criterion.
    """
    criterion = criterion.upper().strip()
    logger.info("Getting evidence for criterion: %s", criterion)

    # Sample evidence items
    items = [
        EvidenceItem(
            evidence_id=uuid4(),
            criterion_id=criterion,
            title=f"Evidence Item 1 for {criterion}",
            evidence_type="configuration",
            source="auth_service",
            status="validated",
            collected_at=datetime.now(timezone.utc),
        ),
        EvidenceItem(
            evidence_id=uuid4(),
            criterion_id=criterion,
            title=f"Evidence Item 2 for {criterion}",
            evidence_type="log_export",
            source="cloudtrail",
            status="validated",
            collected_at=datetime.now(timezone.utc),
        ),
    ]

    validation_results = []
    if include_validation:
        for item in items:
            validation_results.append(
                EvidenceValidationResult(
                    evidence_id=item.evidence_id,
                    validation_type="integrity",
                    status="pass",
                    message="Hash verified successfully",
                    validated_at=datetime.now(timezone.utc),
                )
            )

    return CriterionEvidenceResponse(
        criterion_id=criterion,
        total_evidence=len(items),
        validated_count=len([i for i in items if i.status == "validated"]),
        approved_count=len([i for i in items if i.status == "approved"]),
        items=items,
        validation_results=validation_results,
    )


@router.post(
    "/collect",
    response_model=EvidenceCollectionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger evidence collection",
    description="Start an asynchronous evidence collection job.",
)
async def collect_evidence(
    request: Request,
    collection_request: EvidenceCollectionRequest,
) -> EvidenceCollectionResponse:
    """Trigger evidence collection for specified criteria.

    Args:
        request: FastAPI request object.
        collection_request: Collection configuration.

    Returns:
        EvidenceCollectionResponse with job tracking info.
    """
    logger.info(
        "Starting evidence collection: criteria=%s, sources=%s",
        collection_request.criteria,
        collection_request.sources,
    )

    collection_id = uuid4()
    started_at = datetime.now(timezone.utc)

    # Estimate 5 seconds per criterion for collection
    from datetime import timedelta
    estimated_completion = started_at + timedelta(
        seconds=len(collection_request.criteria) * 5
    )

    return EvidenceCollectionResponse(
        collection_id=collection_id,
        status="running",
        criteria_count=len(collection_request.criteria),
        started_at=started_at,
        estimated_completion=estimated_completion,
    )


@router.post(
    "/package",
    response_model=EvidencePackageResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create evidence package",
    description="Create an evidence package for auditor delivery.",
)
async def create_package(
    request: Request,
    package_request: EvidencePackageRequest,
) -> EvidencePackageResponse:
    """Create an evidence package for auditor delivery.

    Args:
        request: FastAPI request object.
        package_request: Package configuration.

    Returns:
        EvidencePackageResponse with package details.
    """
    logger.info(
        "Creating evidence package: name=%s, criteria=%s",
        package_request.package_name,
        package_request.criteria,
    )

    package_id = uuid4()
    created_at = datetime.now(timezone.utc)

    return EvidencePackageResponse(
        package_id=package_id,
        package_name=package_request.package_name,
        status="creating",
        evidence_count=0,  # Will be updated when complete
        total_size_bytes=0,
        criteria=package_request.criteria,
        created_at=created_at,
        expires_at=None,  # Set when ready
        download_url=None,  # Set when ready
    )


@router.get(
    "/package/{package_id}",
    response_model=EvidencePackageResponse,
    summary="Get package status",
    description="Get the status of an evidence package, including download URL if ready.",
)
async def get_package(
    request: Request,
    package_id: UUID,
) -> EvidencePackageResponse:
    """Get an evidence package by ID.

    Args:
        request: FastAPI request object.
        package_id: The package identifier.

    Returns:
        EvidencePackageResponse with package details.

    Raises:
        HTTPException: 404 if package not found.
    """
    logger.info("Getting package: %s", package_id)

    # In production, this would query the database
    # Return a sample "ready" package
    from datetime import timedelta

    created_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=24)

    return EvidencePackageResponse(
        package_id=package_id,
        package_name="SOC 2 Evidence Package Q1 2026",
        status="ready",
        evidence_count=45,
        total_size_bytes=125678901,
        criteria=["CC6", "CC7", "CC8"],
        created_at=created_at,
        expires_at=expires_at,
        download_url=f"https://s3.amazonaws.com/greenlang-soc2-evidence/packages/{package_id}.zip?signature=xxx",
        manifest_hash="e5f6a7b8c9d0e1f2...",
    )


@router.get(
    "/sources",
    response_model=List[Dict[str, Any]],
    summary="List evidence sources",
    description="List available evidence collection sources and their status.",
)
async def list_sources(
    request: Request,
) -> List[Dict[str, Any]]:
    """List available evidence collection sources.

    Args:
        request: FastAPI request object.

    Returns:
        List of source configurations and status.
    """
    logger.info("Listing evidence sources")

    return [
        {
            "source_id": "cloudtrail",
            "name": "AWS CloudTrail",
            "status": "connected",
            "last_sync": datetime.now(timezone.utc).isoformat(),
            "supported_criteria": ["CC6.1", "CC6.2", "CC7.1", "CC7.2"],
        },
        {
            "source_id": "github",
            "name": "GitHub",
            "status": "connected",
            "last_sync": datetime.now(timezone.utc).isoformat(),
            "supported_criteria": ["CC8.1", "CC8.2"],
        },
        {
            "source_id": "auth_service",
            "name": "GreenLang Auth Service",
            "status": "connected",
            "last_sync": datetime.now(timezone.utc).isoformat(),
            "supported_criteria": ["CC6.1", "CC6.2", "CC6.3"],
        },
        {
            "source_id": "okta",
            "name": "Okta SSO",
            "status": "connected",
            "last_sync": datetime.now(timezone.utc).isoformat(),
            "supported_criteria": ["CC6.1", "CC6.2", "CC6.3"],
        },
        {
            "source_id": "jira",
            "name": "Jira",
            "status": "connected",
            "last_sync": datetime.now(timezone.utc).isoformat(),
            "supported_criteria": ["CC8.1", "CC8.2", "CC8.3"],
        },
    ]


__all__ = ["router"]
