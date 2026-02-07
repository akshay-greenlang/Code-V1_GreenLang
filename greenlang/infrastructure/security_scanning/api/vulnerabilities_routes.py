# -*- coding: utf-8 -*-
"""
Vulnerability Management REST API Routes - SEC-007

FastAPI router for vulnerability CRUD and lifecycle management:

    GET  /vulnerabilities            - List vulnerabilities with filters
    GET  /vulnerabilities/{id}       - Get vulnerability details with findings
    POST /vulnerabilities/{id}/accept    - Risk acceptance workflow
    POST /vulnerabilities/{id}/remediate - Mark vulnerability as fixed
    GET  /vulnerabilities/stats      - Vulnerability statistics

Author: GreenLang Framework Team
Date: February 2026
PRD: SEC-007 Security Scanning Pipeline
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
    from pydantic import BaseModel, ConfigDict, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = object  # type: ignore[misc, assignment]
    Depends = None  # type: ignore[assignment]
    HTTPException = Exception  # type: ignore[misc, assignment]
    Query = None  # type: ignore[assignment]
    Request = None  # type: ignore[assignment]
    status = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[misc, assignment]
    ConfigDict = None  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class VulnerabilityResponse(BaseModel):
        """Vulnerability response model."""

        model_config = ConfigDict(from_attributes=True)

        id: UUID = Field(..., description="Vulnerability UUID")
        cve: Optional[str] = Field(None, description="CVE identifier")
        title: str = Field(..., description="Vulnerability title")
        description: Optional[str] = Field(None, description="Detailed description")
        severity: str = Field(..., description="Severity level")
        status: str = Field(..., description="Lifecycle status")
        cvss_score: Optional[float] = Field(None, description="CVSS 3.1 base score")
        epss_score: Optional[float] = Field(None, description="EPSS probability")
        risk_score: float = Field(..., description="Calculated risk score (0-10)")
        is_exploited: bool = Field(False, description="Known to be exploited in wild")
        is_kev: bool = Field(False, description="In CISA KEV catalog")
        discovered_at: datetime = Field(..., description="Discovery timestamp")
        sla_due_at: Optional[datetime] = Field(None, description="SLA due date")
        resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp")
        package_name: Optional[str] = Field(None, description="Affected package")
        package_version: Optional[str] = Field(None, description="Affected version")
        fixed_version: Optional[str] = Field(None, description="Fixed version")
        remediation_guidance: Optional[str] = Field(None, description="Fix guidance")
        findings_count: int = Field(0, description="Number of associated findings")

    class VulnerabilityDetailResponse(VulnerabilityResponse):
        """Detailed vulnerability response with findings."""

        findings: List[Dict[str, Any]] = Field(
            default_factory=list, description="Associated findings"
        )
        exceptions: List[Dict[str, Any]] = Field(
            default_factory=list, description="Risk exceptions"
        )

    class VulnerabilitiesListResponse(BaseModel):
        """Paginated list of vulnerabilities."""

        items: List[VulnerabilityResponse] = Field(..., description="Vulnerabilities")
        total: int = Field(..., description="Total matching count")
        limit: int = Field(..., description="Page size")
        offset: int = Field(..., description="Offset")
        has_more: bool = Field(..., description="More results available")

    class RiskAcceptanceRequest(BaseModel):
        """Request to accept risk for a vulnerability."""

        exception_type: str = Field(
            "risk_acceptance",
            description="Exception type: risk_acceptance, false_positive, wont_fix, deferred",
        )
        reason: str = Field(..., min_length=10, description="Reason for acceptance")
        business_justification: Optional[str] = Field(
            None, description="Business justification"
        )
        compensating_controls: Optional[str] = Field(
            None, description="Compensating controls in place"
        )
        approved_by: str = Field(..., description="Approver identifier")
        expires_at: Optional[datetime] = Field(None, description="Exception expiry")
        is_permanent: bool = Field(False, description="Permanent exception")

    class RemediationRequest(BaseModel):
        """Request to mark vulnerability as remediated."""

        remediated_by: Optional[str] = Field(None, description="User who fixed")
        notes: Optional[str] = Field(None, description="Remediation notes")

    class VulnerabilityStatsResponse(BaseModel):
        """Vulnerability statistics response."""

        total: int = Field(..., description="Total vulnerabilities")
        open: int = Field(..., description="Open vulnerabilities")
        resolved: int = Field(..., description="Resolved vulnerabilities")
        accepted: int = Field(..., description="Risk-accepted vulnerabilities")
        by_severity: Dict[str, int] = Field(..., description="Count by severity")
        kev_count: int = Field(..., description="KEV vulnerabilities")
        sla_breached: int = Field(..., description="SLA breached count")
        avg_risk_score: float = Field(..., description="Average risk score")
        avg_mttr_days: float = Field(..., description="Average time to remediation")


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def _get_vulnerability_service() -> Any:
    """FastAPI dependency to get VulnerabilityService.

    Returns:
        The VulnerabilityService instance.

    Raises:
        HTTPException 503: If service is not available.
    """
    try:
        from greenlang.infrastructure.security_scanning.vulnerability_service import (
            get_vulnerability_service,
        )

        service = get_vulnerability_service()
        if service is None:
            raise HTTPException(
                status_code=503,
                detail="Vulnerability service is not configured.",
            )
        return service
    except ImportError as exc:
        logger.error("Vulnerability service not available: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Vulnerability service is not available.",
        )


def _get_tenant_id(request: Request) -> Optional[UUID]:
    """Extract tenant ID from request headers.

    Args:
        request: FastAPI Request.

    Returns:
        Tenant UUID or None.
    """
    tenant_str = request.headers.get("x-tenant-id")
    if tenant_str:
        try:
            return UUID(tenant_str)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Router Definition
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    from greenlang.infrastructure.security_scanning.vulnerability_service import (
        VulnerabilityFilter,
        VulnerabilitySeverity,
        VulnerabilityStatus,
        RiskAcceptance,
        ExceptionType,
    )

    vulnerabilities_router = APIRouter(
        prefix="/vulnerabilities",
        tags=["Vulnerabilities"],
        responses={
            400: {"description": "Bad Request"},
            403: {"description": "Forbidden"},
            404: {"description": "Vulnerability Not Found"},
            500: {"description": "Internal Server Error"},
            503: {"description": "Service Unavailable"},
        },
    )

    @vulnerabilities_router.get(
        "",
        response_model=VulnerabilitiesListResponse,
        summary="List vulnerabilities",
        description="Query vulnerabilities with filters, sorted by risk priority.",
        operation_id="list_vulnerabilities",
    )
    async def list_vulnerabilities(
        request: Request,
        severity: Optional[str] = Query(
            None, description="Filter by severity: critical, high, medium, low, info"
        ),
        status_filter: Optional[str] = Query(
            None,
            alias="status",
            description="Filter by status: open, in_progress, resolved, accepted",
        ),
        cve: Optional[str] = Query(None, description="Filter by CVE identifier"),
        scanner: Optional[str] = Query(None, description="Filter by scanner"),
        package_name: Optional[str] = Query(
            None, description="Filter by package name (partial match)"
        ),
        is_kev: Optional[bool] = Query(None, description="Filter KEV vulnerabilities"),
        sla_breached: Optional[bool] = Query(
            None, description="Filter by SLA breach status"
        ),
        since: Optional[datetime] = Query(
            None, description="Discovered after (ISO 8601)"
        ),
        until: Optional[datetime] = Query(
            None, description="Discovered before (ISO 8601)"
        ),
        limit: int = Query(50, ge=1, le=500, description="Page size"),
        offset: int = Query(0, ge=0, description="Offset"),
        service: Any = Depends(_get_vulnerability_service),
    ) -> VulnerabilitiesListResponse:
        """List vulnerabilities with optional filters.

        Results are sorted by severity (critical first), then risk score, then age.
        """
        tenant_id = _get_tenant_id(request)

        # Parse severity filter
        severities = None
        if severity:
            try:
                severities = [VulnerabilitySeverity(severity.lower())]
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid severity '{severity}'",
                )

        # Parse status filter
        statuses = None
        if status_filter:
            try:
                statuses = [VulnerabilityStatus(status_filter.lower())]
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid status '{status_filter}'",
                )

        filters = VulnerabilityFilter(
            severities=severities,
            statuses=statuses,
            cve=cve,
            scanner=scanner,
            package_name=package_name,
            is_kev=is_kev,
            sla_breached=sla_breached,
            since=since,
            until=until,
            tenant_id=tenant_id,
            limit=limit + 1,  # Fetch one extra to check has_more
            offset=offset,
        )

        try:
            vulnerabilities = await service.get_vulnerabilities(filters)

            has_more = len(vulnerabilities) > limit
            if has_more:
                vulnerabilities = vulnerabilities[:limit]

            items = [
                VulnerabilityResponse(
                    id=v.id,
                    cve=v.cve,
                    title=v.title,
                    description=v.description,
                    severity=v.severity.value,
                    status=v.status.value,
                    cvss_score=v.cvss_score,
                    epss_score=v.epss_score,
                    risk_score=v.risk_score,
                    is_exploited=v.is_exploited,
                    is_kev=v.is_kev,
                    discovered_at=v.discovered_at,
                    sla_due_at=v.sla_due_at,
                    resolved_at=v.resolved_at,
                    package_name=v.package_name,
                    package_version=v.package_version,
                    fixed_version=v.fixed_version,
                    remediation_guidance=v.remediation_guidance,
                    findings_count=v.findings_count,
                )
                for v in vulnerabilities
            ]

            return VulnerabilitiesListResponse(
                items=items,
                total=len(items),  # For cursor pagination; real total requires count query
                limit=limit,
                offset=offset,
                has_more=has_more,
            )

        except Exception as exc:
            logger.exception("Failed to list vulnerabilities")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list vulnerabilities: {exc}",
            )

    @vulnerabilities_router.get(
        "/{vulnerability_id}",
        response_model=VulnerabilityDetailResponse,
        summary="Get vulnerability details",
        description="Retrieve detailed vulnerability information including findings.",
        operation_id="get_vulnerability",
    )
    async def get_vulnerability(
        vulnerability_id: UUID,
        service: Any = Depends(_get_vulnerability_service),
    ) -> VulnerabilityDetailResponse:
        """Get detailed vulnerability by ID."""
        try:
            vuln = await service.get_vulnerability(vulnerability_id)
        except Exception as exc:
            logger.exception("Failed to get vulnerability %s", vulnerability_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get vulnerability: {exc}",
            )

        if vuln is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Vulnerability '{vulnerability_id}' not found.",
            )

        # TODO: Fetch findings and exceptions from service
        return VulnerabilityDetailResponse(
            id=vuln.id,
            cve=vuln.cve,
            title=vuln.title,
            description=vuln.description,
            severity=vuln.severity.value,
            status=vuln.status.value,
            cvss_score=vuln.cvss_score,
            epss_score=vuln.epss_score,
            risk_score=vuln.risk_score,
            is_exploited=vuln.is_exploited,
            is_kev=vuln.is_kev,
            discovered_at=vuln.discovered_at,
            sla_due_at=vuln.sla_due_at,
            resolved_at=vuln.resolved_at,
            package_name=vuln.package_name,
            package_version=vuln.package_version,
            fixed_version=vuln.fixed_version,
            remediation_guidance=vuln.remediation_guidance,
            findings_count=vuln.findings_count,
            findings=[],
            exceptions=[],
        )

    @vulnerabilities_router.post(
        "/{vulnerability_id}/accept",
        status_code=status.HTTP_200_OK,
        summary="Accept risk for vulnerability",
        description="Record risk acceptance decision for a vulnerability.",
        operation_id="accept_vulnerability_risk",
    )
    async def accept_vulnerability_risk(
        vulnerability_id: UUID,
        request_body: RiskAcceptanceRequest,
        service: Any = Depends(_get_vulnerability_service),
    ) -> Dict[str, Any]:
        """Accept risk for a vulnerability (create exception)."""
        try:
            exception_type = ExceptionType(request_body.exception_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid exception type '{request_body.exception_type}'",
            )

        acceptance = RiskAcceptance(
            vulnerability_id=vulnerability_id,
            exception_type=exception_type,
            reason=request_body.reason,
            business_justification=request_body.business_justification,
            compensating_controls=request_body.compensating_controls,
            approved_by=request_body.approved_by,
            expires_at=request_body.expires_at,
            is_permanent=request_body.is_permanent,
        )

        try:
            success = await service.accept_risk(acceptance)
            if success:
                return {
                    "status": "accepted",
                    "vulnerability_id": str(vulnerability_id),
                    "exception_type": exception_type.value,
                    "approved_by": request_body.approved_by,
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to record risk acceptance.",
                )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )
        except Exception as exc:
            logger.exception("Failed to accept risk for %s", vulnerability_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to accept risk: {exc}",
            )

    @vulnerabilities_router.post(
        "/{vulnerability_id}/remediate",
        status_code=status.HTTP_200_OK,
        summary="Mark vulnerability as remediated",
        description="Mark a vulnerability as fixed/resolved.",
        operation_id="remediate_vulnerability",
    )
    async def remediate_vulnerability(
        vulnerability_id: UUID,
        request_body: RemediationRequest,
        service: Any = Depends(_get_vulnerability_service),
    ) -> Dict[str, Any]:
        """Mark vulnerability as remediated."""
        try:
            success = await service.mark_remediated(
                vulnerability_id=vulnerability_id,
                remediated_by=request_body.remediated_by,
                notes=request_body.notes,
            )
            if success:
                return {
                    "status": "resolved",
                    "vulnerability_id": str(vulnerability_id),
                    "remediated_by": request_body.remediated_by,
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Vulnerability '{vulnerability_id}' not found or already resolved.",
                )
        except Exception as exc:
            logger.exception("Failed to remediate %s", vulnerability_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to mark remediated: {exc}",
            )

    @vulnerabilities_router.get(
        "/stats",
        response_model=VulnerabilityStatsResponse,
        summary="Get vulnerability statistics",
        description="Retrieve aggregated vulnerability statistics for dashboard.",
        operation_id="get_vulnerability_stats",
    )
    async def get_vulnerability_stats(
        request: Request,
        since: Optional[datetime] = Query(None, description="Start of time range"),
        until: Optional[datetime] = Query(None, description="End of time range"),
        service: Any = Depends(_get_vulnerability_service),
    ) -> VulnerabilityStatsResponse:
        """Get aggregated vulnerability statistics."""
        tenant_id = _get_tenant_id(request)

        try:
            stats = await service.get_statistics(
                since=since,
                until=until,
                tenant_id=tenant_id,
            )

            return VulnerabilityStatsResponse(
                total=stats.get("total", 0),
                open=stats.get("open", 0),
                resolved=stats.get("resolved", 0),
                accepted=stats.get("accepted", 0),
                by_severity=stats.get("by_severity", {}),
                kev_count=stats.get("kev_count", 0),
                sla_breached=stats.get("sla_breached", 0),
                avg_risk_score=stats.get("avg_risk_score", 0.0),
                avg_mttr_days=stats.get("avg_mttr_days", 0.0),
            )

        except Exception as exc:
            logger.exception("Failed to get vulnerability stats")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get statistics: {exc}",
            )

    # SEC-001: Apply authentication and permission protection
    try:
        from greenlang.infrastructure.auth_service.route_protector import (
            protect_router,
        )

        protect_router(vulnerabilities_router)
    except ImportError:
        pass  # auth_service not available

else:
    vulnerabilities_router = None  # type: ignore[assignment]
    logger.warning("FastAPI not available - vulnerabilities_router is None")


__all__ = ["vulnerabilities_router"]
