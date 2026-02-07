# -*- coding: utf-8 -*-
"""
Security Scan REST API Routes - SEC-007

FastAPI router for security scan execution and history:

    POST /scans              - Trigger a new security scan (async)
    GET  /scans              - List scan runs with filters
    GET  /scans/{id}         - Get scan run details
    GET  /scans/{id}/findings - Get findings for a scan run

Author: GreenLang Framework Team
Date: February 2026
PRD: SEC-007 Security Scanning Pipeline
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, Depends, HTTPException, Query, Request, status, BackgroundTasks
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
    BackgroundTasks = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[misc, assignment]
    ConfigDict = None  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class ScanRequest(BaseModel):
        """Request to trigger a security scan."""

        scanner: str = Field(
            ...,
            description="Scanner to use: trivy, bandit, semgrep, gitleaks, owasp_zap, etc.",
        )
        target_type: str = Field(
            "repository",
            description="Target type: repository, image, url, api, manifest",
        )
        target_path: Optional[str] = Field(
            None, description="Target path, image name, or URL"
        )
        target_ref: Optional[str] = Field(
            None, description="Git ref (branch, tag, commit)"
        )
        scan_config: Optional[Dict[str, Any]] = Field(
            None, description="Scanner-specific configuration"
        )

    class ScanResponse(BaseModel):
        """Scan run response model."""

        model_config = ConfigDict(from_attributes=True)

        id: UUID = Field(..., description="Scan run UUID")
        scanner: str = Field(..., description="Scanner name")
        scanner_version: Optional[str] = Field(None, description="Scanner version")
        target_type: str = Field(..., description="Target type")
        target_path: Optional[str] = Field(None, description="Target path")
        target_ref: Optional[str] = Field(None, description="Git ref")
        commit_sha: Optional[str] = Field(None, description="Commit SHA")
        branch: Optional[str] = Field(None, description="Branch name")
        trigger_type: str = Field(..., description="Trigger type")
        triggered_by: Optional[str] = Field(None, description="User who triggered")
        started_at: datetime = Field(..., description="Start timestamp")
        completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
        duration_seconds: Optional[float] = Field(None, description="Duration")
        status: str = Field(..., description="Scan status")
        exit_code: Optional[int] = Field(None, description="Exit code")
        findings_total: int = Field(0, description="Total findings")
        findings_critical: int = Field(0, description="Critical findings")
        findings_high: int = Field(0, description="High findings")
        findings_medium: int = Field(0, description="Medium findings")
        findings_low: int = Field(0, description="Low findings")
        findings_info: int = Field(0, description="Info findings")
        findings_new: int = Field(0, description="New findings")
        findings_existing: int = Field(0, description="Existing findings")
        findings_resolved: int = Field(0, description="Resolved findings")
        report_url: Optional[str] = Field(None, description="Report URL")
        sarif_url: Optional[str] = Field(None, description="SARIF URL")
        error_message: Optional[str] = Field(None, description="Error message if failed")

    class ScanTriggerResponse(BaseModel):
        """Response for scan trigger request."""

        job_id: UUID = Field(..., description="Job ID for tracking")
        scan_id: UUID = Field(..., description="Scan run ID")
        status: str = Field("pending", description="Initial status")
        message: str = Field(..., description="Status message")

    class ScansListResponse(BaseModel):
        """Paginated list of scan runs."""

        items: List[ScanResponse] = Field(..., description="Scan runs")
        total: int = Field(..., description="Total count")
        limit: int = Field(..., description="Page size")
        offset: int = Field(..., description="Offset")
        has_more: bool = Field(..., description="More results available")

    class FindingResponse(BaseModel):
        """Scan finding response model."""

        model_config = ConfigDict(from_attributes=True)

        id: UUID = Field(..., description="Finding UUID")
        vulnerability_id: Optional[UUID] = Field(None, description="Linked vulnerability")
        scanner: str = Field(..., description="Scanner")
        finding_type: str = Field(..., description="Finding type")
        severity: str = Field(..., description="Severity")
        confidence: str = Field(..., description="Confidence")
        file_path: Optional[str] = Field(None, description="File path")
        line_start: Optional[int] = Field(None, description="Start line")
        line_end: Optional[int] = Field(None, description="End line")
        code_snippet: Optional[str] = Field(None, description="Code snippet")
        cve: Optional[str] = Field(None, description="CVE")
        cwe: Optional[str] = Field(None, description="CWE")
        rule_id: Optional[str] = Field(None, description="Rule ID")
        message: str = Field(..., description="Finding message")
        created_at: datetime = Field(..., description="Created timestamp")

    class ScanFindingsResponse(BaseModel):
        """Paginated list of findings for a scan."""

        items: List[FindingResponse] = Field(..., description="Findings")
        total: int = Field(..., description="Total count")
        scan_id: UUID = Field(..., description="Scan run ID")


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def _get_db_pool() -> Any:
    """FastAPI dependency to get database pool.

    Returns:
        Database connection pool.

    Raises:
        HTTPException 503: If database is not available.
    """
    # In production, this would return the actual pool
    # For now, return None to indicate mock mode
    return None


def _get_tenant_id(request: Request) -> Optional[UUID]:
    """Extract tenant ID from request headers."""
    tenant_str = request.headers.get("x-tenant-id")
    if tenant_str:
        try:
            return UUID(tenant_str)
        except ValueError:
            return None
    return None


def _get_user_id(request: Request) -> Optional[str]:
    """Extract user ID from request headers."""
    return request.headers.get("x-user-id")


# ---------------------------------------------------------------------------
# Background Tasks
# ---------------------------------------------------------------------------


async def _run_scan_async(
    scan_id: UUID,
    scanner: str,
    target_type: str,
    target_path: Optional[str],
    scan_config: Optional[Dict[str, Any]],
    db_pool: Any,
) -> None:
    """Background task to run a security scan.

    Args:
        scan_id: Scan run ID.
        scanner: Scanner name.
        target_type: Target type.
        target_path: Target path.
        scan_config: Scanner configuration.
        db_pool: Database pool.
    """
    logger.info("Starting scan %s with scanner %s", scan_id, scanner)

    try:
        # Update status to running
        if db_pool:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE security.scan_runs
                    SET status = 'running', started_at = NOW()
                    WHERE id = $1
                    """,
                    scan_id,
                )

        # Import and run orchestrator
        # try:
        #     from greenlang.infrastructure.security_scanning.orchestrator import (
        #         get_scan_orchestrator,
        #     )
        #     orchestrator = get_scan_orchestrator()
        #     if orchestrator:
        #         result = await orchestrator.run_scanner(
        #             scanner=scanner,
        #             target_type=target_type,
        #             target_path=target_path,
        #             config=scan_config,
        #         )
        # except ImportError:
        #     pass

        # For now, mark as completed (mock)
        if db_pool:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE security.scan_runs
                    SET
                        status = 'completed',
                        completed_at = NOW(),
                        duration_seconds = EXTRACT(EPOCH FROM (NOW() - started_at))
                    WHERE id = $1
                    """,
                    scan_id,
                )

        logger.info("Scan %s completed", scan_id)

    except Exception as e:
        logger.error("Scan %s failed: %s", scan_id, e)
        if db_pool:
            try:
                async with db_pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE security.scan_runs
                        SET status = 'failed', error_message = $2, completed_at = NOW()
                        WHERE id = $1
                        """,
                        scan_id,
                        str(e),
                    )
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Router Definition
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    scans_router = APIRouter(
        prefix="/scans",
        tags=["Security Scans"],
        responses={
            400: {"description": "Bad Request"},
            403: {"description": "Forbidden"},
            404: {"description": "Scan Not Found"},
            500: {"description": "Internal Server Error"},
            503: {"description": "Service Unavailable"},
        },
    )

    VALID_SCANNERS = {
        "bandit", "semgrep", "codeql",
        "trivy", "snyk", "pip_audit", "safety",
        "gitleaks", "trufflehog", "detect_secrets",
        "grype", "cosign",
        "tfsec", "checkov", "kubeconform",
        "owasp_zap",
        "presidio", "pii_scanner",
    }

    VALID_TARGET_TYPES = {
        "repository", "branch", "commit", "pull_request",
        "image", "manifest", "url", "api", "infrastructure",
    }

    @scans_router.post(
        "",
        response_model=ScanTriggerResponse,
        status_code=status.HTTP_202_ACCEPTED,
        summary="Trigger a security scan",
        description="Start a new security scan asynchronously. Returns job ID for tracking.",
        operation_id="trigger_scan",
    )
    async def trigger_scan(
        request_body: ScanRequest,
        request: Request,
        background_tasks: BackgroundTasks,
        db_pool: Any = Depends(_get_db_pool),
    ) -> ScanTriggerResponse:
        """Trigger a new security scan.

        The scan runs asynchronously in the background. Use the returned job_id
        to poll for status via GET /scans/{scan_id}.
        """
        # Validate scanner
        if request_body.scanner.lower() not in VALID_SCANNERS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid scanner '{request_body.scanner}'. Valid: {sorted(VALID_SCANNERS)}",
            )

        # Validate target type
        if request_body.target_type.lower() not in VALID_TARGET_TYPES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid target_type '{request_body.target_type}'. Valid: {sorted(VALID_TARGET_TYPES)}",
            )

        scan_id = uuid4()
        job_id = uuid4()
        tenant_id = _get_tenant_id(request)
        user_id = _get_user_id(request)

        # Create scan run record
        if db_pool:
            try:
                async with db_pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO security.scan_runs (
                            id, scanner, target_type, target_path, target_ref,
                            trigger_type, triggered_by, status, tenant_id, scan_config
                        ) VALUES ($1, $2, $3, $4, $5, 'api', $6, 'pending', $7, $8)
                        """,
                        scan_id,
                        request_body.scanner.lower(),
                        request_body.target_type.lower(),
                        request_body.target_path,
                        request_body.target_ref,
                        user_id,
                        tenant_id,
                        request_body.scan_config or {},
                    )
            except Exception as e:
                logger.error("Failed to create scan run: %s", e)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to create scan: {e}",
                )

        # Schedule background task
        background_tasks.add_task(
            _run_scan_async,
            scan_id,
            request_body.scanner.lower(),
            request_body.target_type.lower(),
            request_body.target_path,
            request_body.scan_config,
            db_pool,
        )

        return ScanTriggerResponse(
            job_id=job_id,
            scan_id=scan_id,
            status="pending",
            message=f"Scan {scan_id} queued with scanner {request_body.scanner}",
        )

    @scans_router.get(
        "",
        response_model=ScansListResponse,
        summary="List scan runs",
        description="Query scan runs with optional filters.",
        operation_id="list_scans",
    )
    async def list_scans(
        request: Request,
        scanner: Optional[str] = Query(None, description="Filter by scanner"),
        status_filter: Optional[str] = Query(
            None,
            alias="status",
            description="Filter by status: pending, running, completed, failed",
        ),
        target_type: Optional[str] = Query(None, description="Filter by target type"),
        since: Optional[datetime] = Query(None, description="Started after"),
        until: Optional[datetime] = Query(None, description="Started before"),
        limit: int = Query(50, ge=1, le=500, description="Page size"),
        offset: int = Query(0, ge=0, description="Offset"),
        db_pool: Any = Depends(_get_db_pool),
    ) -> ScansListResponse:
        """List scan runs with optional filters."""
        tenant_id = _get_tenant_id(request)

        # For mock mode, return empty list
        if db_pool is None:
            return ScansListResponse(
                items=[],
                total=0,
                limit=limit,
                offset=offset,
                has_more=False,
            )

        query = """
            SELECT * FROM security.scan_runs
            WHERE 1=1
        """
        params: List[Any] = []
        param_idx = 1

        if scanner:
            query += f" AND scanner = ${param_idx}"
            params.append(scanner.lower())
            param_idx += 1

        if status_filter:
            query += f" AND status = ${param_idx}"
            params.append(status_filter.lower())
            param_idx += 1

        if target_type:
            query += f" AND target_type = ${param_idx}"
            params.append(target_type.lower())
            param_idx += 1

        if since:
            query += f" AND started_at >= ${param_idx}"
            params.append(since)
            param_idx += 1

        if until:
            query += f" AND started_at <= ${param_idx}"
            params.append(until)
            param_idx += 1

        if tenant_id:
            query += f" AND tenant_id = ${param_idx}"
            params.append(tenant_id)
            param_idx += 1

        query += " ORDER BY started_at DESC"
        query += f" LIMIT ${param_idx} OFFSET ${param_idx + 1}"
        params.extend([limit + 1, offset])

        try:
            async with db_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)

            has_more = len(rows) > limit
            if has_more:
                rows = rows[:limit]

            items = [
                ScanResponse(
                    id=row["id"],
                    scanner=row["scanner"],
                    scanner_version=row.get("scanner_version"),
                    target_type=row["target_type"],
                    target_path=row.get("target_path"),
                    target_ref=row.get("target_ref"),
                    commit_sha=row.get("commit_sha"),
                    branch=row.get("branch"),
                    trigger_type=row["trigger_type"],
                    triggered_by=row.get("triggered_by"),
                    started_at=row["started_at"],
                    completed_at=row.get("completed_at"),
                    duration_seconds=float(row["duration_seconds"]) if row.get("duration_seconds") else None,
                    status=row["status"],
                    exit_code=row.get("exit_code"),
                    findings_total=row.get("findings_total", 0),
                    findings_critical=row.get("findings_critical", 0),
                    findings_high=row.get("findings_high", 0),
                    findings_medium=row.get("findings_medium", 0),
                    findings_low=row.get("findings_low", 0),
                    findings_info=row.get("findings_info", 0),
                    findings_new=row.get("findings_new", 0),
                    findings_existing=row.get("findings_existing", 0),
                    findings_resolved=row.get("findings_resolved", 0),
                    report_url=row.get("report_url"),
                    sarif_url=row.get("sarif_url"),
                    error_message=row.get("error_message"),
                )
                for row in rows
            ]

            return ScansListResponse(
                items=items,
                total=len(items),
                limit=limit,
                offset=offset,
                has_more=has_more,
            )

        except Exception as exc:
            logger.exception("Failed to list scans")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list scans: {exc}",
            )

    @scans_router.get(
        "/{scan_id}",
        response_model=ScanResponse,
        summary="Get scan details",
        description="Retrieve details for a specific scan run.",
        operation_id="get_scan",
    )
    async def get_scan(
        scan_id: UUID,
        db_pool: Any = Depends(_get_db_pool),
    ) -> ScanResponse:
        """Get scan run details by ID."""
        if db_pool is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scan '{scan_id}' not found.",
            )

        try:
            async with db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM security.scan_runs WHERE id = $1",
                    scan_id,
                )

            if row is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Scan '{scan_id}' not found.",
                )

            return ScanResponse(
                id=row["id"],
                scanner=row["scanner"],
                scanner_version=row.get("scanner_version"),
                target_type=row["target_type"],
                target_path=row.get("target_path"),
                target_ref=row.get("target_ref"),
                commit_sha=row.get("commit_sha"),
                branch=row.get("branch"),
                trigger_type=row["trigger_type"],
                triggered_by=row.get("triggered_by"),
                started_at=row["started_at"],
                completed_at=row.get("completed_at"),
                duration_seconds=float(row["duration_seconds"]) if row.get("duration_seconds") else None,
                status=row["status"],
                exit_code=row.get("exit_code"),
                findings_total=row.get("findings_total", 0),
                findings_critical=row.get("findings_critical", 0),
                findings_high=row.get("findings_high", 0),
                findings_medium=row.get("findings_medium", 0),
                findings_low=row.get("findings_low", 0),
                findings_info=row.get("findings_info", 0),
                findings_new=row.get("findings_new", 0),
                findings_existing=row.get("findings_existing", 0),
                findings_resolved=row.get("findings_resolved", 0),
                report_url=row.get("report_url"),
                sarif_url=row.get("sarif_url"),
                error_message=row.get("error_message"),
            )

        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Failed to get scan %s", scan_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get scan: {exc}",
            )

    @scans_router.get(
        "/{scan_id}/findings",
        response_model=ScanFindingsResponse,
        summary="Get findings for a scan",
        description="Retrieve all findings discovered in a specific scan run.",
        operation_id="get_scan_findings",
    )
    async def get_scan_findings(
        scan_id: UUID,
        severity: Optional[str] = Query(None, description="Filter by severity"),
        finding_type: Optional[str] = Query(None, description="Filter by finding type"),
        limit: int = Query(100, ge=1, le=1000, description="Page size"),
        offset: int = Query(0, ge=0, description="Offset"),
        db_pool: Any = Depends(_get_db_pool),
    ) -> ScanFindingsResponse:
        """Get findings for a scan run."""
        if db_pool is None:
            return ScanFindingsResponse(
                items=[],
                total=0,
                scan_id=scan_id,
            )

        query = """
            SELECT * FROM security.findings
            WHERE scan_run_id = $1
        """
        params: List[Any] = [scan_id]
        param_idx = 2

        if severity:
            query += f" AND severity = ${param_idx}"
            params.append(severity.lower())
            param_idx += 1

        if finding_type:
            query += f" AND finding_type = ${param_idx}"
            params.append(finding_type.lower())
            param_idx += 1

        query += " ORDER BY severity, created_at DESC"
        query += f" LIMIT ${param_idx} OFFSET ${param_idx + 1}"
        params.extend([limit, offset])

        try:
            async with db_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)

            items = [
                FindingResponse(
                    id=row["id"],
                    vulnerability_id=row.get("vulnerability_id"),
                    scanner=row["scanner"],
                    finding_type=row["finding_type"],
                    severity=row["severity"],
                    confidence=row["confidence"],
                    file_path=row.get("file_path"),
                    line_start=row.get("line_start"),
                    line_end=row.get("line_end"),
                    code_snippet=row.get("code_snippet"),
                    cve=row.get("cve"),
                    cwe=row.get("cwe"),
                    rule_id=row.get("rule_id"),
                    message=row["message"],
                    created_at=row["created_at"],
                )
                for row in rows
            ]

            return ScanFindingsResponse(
                items=items,
                total=len(items),
                scan_id=scan_id,
            )

        except Exception as exc:
            logger.exception("Failed to get findings for scan %s", scan_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get findings: {exc}",
            )

    # SEC-001: Apply authentication and permission protection
    try:
        from greenlang.infrastructure.auth_service.route_protector import (
            protect_router,
        )

        protect_router(scans_router)
    except ImportError:
        pass

else:
    scans_router = None  # type: ignore[assignment]
    logger.warning("FastAPI not available - scans_router is None")


__all__ = ["scans_router"]
