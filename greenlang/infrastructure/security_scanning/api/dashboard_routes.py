# -*- coding: utf-8 -*-
"""
Security Dashboard REST API Routes - SEC-007

FastAPI router for security dashboard and statistics:

    GET /dashboard          - Summary statistics
    GET /dashboard/trends   - 90-day trend data
    GET /dashboard/coverage - Scanner coverage
    GET /dashboard/sla      - SLA compliance metrics

Author: GreenLang Framework Team
Date: February 2026
PRD: SEC-007 Security Scanning Pipeline
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
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

    class DashboardSummaryResponse(BaseModel):
        """Security dashboard summary response."""

        # Vulnerability counts
        total_vulnerabilities: int = Field(..., description="Total vulnerabilities")
        open_vulnerabilities: int = Field(..., description="Open vulnerabilities")
        resolved_vulnerabilities: int = Field(..., description="Resolved vulnerabilities")
        accepted_vulnerabilities: int = Field(..., description="Risk-accepted vulnerabilities")

        # By severity
        critical_count: int = Field(..., description="Critical vulnerabilities")
        high_count: int = Field(..., description="High vulnerabilities")
        medium_count: int = Field(..., description="Medium vulnerabilities")
        low_count: int = Field(..., description="Low vulnerabilities")

        # Risk indicators
        kev_count: int = Field(..., description="KEV vulnerabilities")
        sla_breached_count: int = Field(..., description="SLA breached count")
        avg_risk_score: float = Field(..., description="Average risk score")

        # Remediation metrics
        avg_mttr_days: float = Field(..., description="Average MTTR in days")
        remediation_velocity: float = Field(..., description="Vulns resolved per week")

        # Scan metrics
        total_scans_24h: int = Field(..., description="Scans in last 24 hours")
        findings_24h: int = Field(..., description="Findings in last 24 hours")
        scanners_active: int = Field(..., description="Active scanner types")

        # Timestamps
        last_scan_at: Optional[datetime] = Field(None, description="Last scan timestamp")
        generated_at: datetime = Field(..., description="Report generation time")

    class TrendDataPoint(BaseModel):
        """Single trend data point."""

        date: datetime = Field(..., description="Data point timestamp")
        open: int = Field(..., description="Open vulnerabilities")
        resolved: int = Field(..., description="Resolved vulnerabilities")
        critical: int = Field(..., description="Critical count")
        high: int = Field(..., description="High count")
        medium: int = Field(..., description="Medium count")
        low: int = Field(..., description="Low count")

    class TrendsResponse(BaseModel):
        """90-day vulnerability trend response."""

        time_range_days: int = Field(90, description="Trend period in days")
        data_points: List[TrendDataPoint] = Field(..., description="Daily data points")
        summary: Dict[str, Any] = Field(..., description="Trend summary statistics")

    class ScannerCoverage(BaseModel):
        """Coverage data for a single scanner."""

        scanner: str = Field(..., description="Scanner name")
        scanner_type: str = Field(..., description="Scanner type (sast, sca, etc)")
        last_scan_at: Optional[datetime] = Field(None, description="Last scan")
        scans_7d: int = Field(..., description="Scans in last 7 days")
        findings_7d: int = Field(..., description="Findings in last 7 days")
        avg_duration_seconds: float = Field(..., description="Average scan duration")
        success_rate: float = Field(..., description="Success rate (0-100)")
        status: str = Field(..., description="healthy, degraded, or inactive")

    class CoverageResponse(BaseModel):
        """Scanner coverage response."""

        scanners: List[ScannerCoverage] = Field(..., description="Scanner coverage data")
        total_scanners: int = Field(..., description="Total configured scanners")
        healthy_scanners: int = Field(..., description="Healthy scanner count")
        coverage_score: float = Field(..., description="Overall coverage score (0-100)")

    class SLAStatusBySeverity(BaseModel):
        """SLA status for a single severity level."""

        severity: str = Field(..., description="Severity level")
        total_open: int = Field(..., description="Total open")
        within_sla: int = Field(..., description="Within SLA")
        approaching_sla: int = Field(..., description="Approaching SLA (< 2 days)")
        breached_sla: int = Field(..., description="SLA breached")
        compliance_rate: float = Field(..., description="SLA compliance rate (0-100)")
        max_sla_days: int = Field(..., description="SLA target days")

    class SLAComplianceResponse(BaseModel):
        """SLA compliance metrics response."""

        overall_compliance_rate: float = Field(..., description="Overall compliance (0-100)")
        total_open: int = Field(..., description="Total open vulnerabilities")
        within_sla: int = Field(..., description="Within SLA")
        approaching_sla: int = Field(..., description="Approaching SLA")
        breached_sla: int = Field(..., description="SLA breached")
        by_severity: List[SLAStatusBySeverity] = Field(..., description="Breakdown by severity")
        oldest_breach_date: Optional[datetime] = Field(None, description="Oldest breach date")
        average_age_days: float = Field(..., description="Average vulnerability age")


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def _get_vulnerability_service() -> Any:
    """FastAPI dependency to get VulnerabilityService."""
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


def _get_db_pool() -> Any:
    """FastAPI dependency to get database pool."""
    return None  # Mock mode


def _get_tenant_id(request: Request) -> Optional[UUID]:
    """Extract tenant ID from request headers."""
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

    dashboard_router = APIRouter(
        prefix="/dashboard",
        tags=["Security Dashboard"],
        responses={
            403: {"description": "Forbidden"},
            500: {"description": "Internal Server Error"},
            503: {"description": "Service Unavailable"},
        },
    )

    @dashboard_router.get(
        "",
        response_model=DashboardSummaryResponse,
        summary="Get dashboard summary",
        description="Retrieve security dashboard summary statistics.",
        operation_id="get_dashboard_summary",
    )
    async def get_dashboard_summary(
        request: Request,
        service: Any = Depends(_get_vulnerability_service),
        db_pool: Any = Depends(_get_db_pool),
    ) -> DashboardSummaryResponse:
        """Get security dashboard summary."""
        tenant_id = _get_tenant_id(request)

        try:
            # Get vulnerability statistics
            stats = await service.get_statistics(tenant_id=tenant_id)

            # Get scan metrics from database
            scans_24h = 0
            findings_24h = 0
            scanners_active = 0
            last_scan_at = None

            if db_pool:
                async with db_pool.acquire() as conn:
                    # Scans in last 24h
                    row = await conn.fetchrow(
                        """
                        SELECT
                            COUNT(*) as scans,
                            SUM(findings_total) as findings,
                            COUNT(DISTINCT scanner) as scanners,
                            MAX(completed_at) as last_scan
                        FROM security.scan_runs
                        WHERE started_at >= NOW() - INTERVAL '24 hours'
                          AND status = 'completed'
                        """
                    )
                    if row:
                        scans_24h = row["scans"] or 0
                        findings_24h = row["findings"] or 0
                        scanners_active = row["scanners"] or 0
                        last_scan_at = row["last_scan"]

            # Calculate remediation velocity (vulns resolved per week, last 30 days)
            remediation_velocity = 0.0
            if db_pool:
                async with db_pool.acquire() as conn:
                    row = await conn.fetchrow(
                        """
                        SELECT COUNT(*) as resolved
                        FROM security.vulnerabilities
                        WHERE resolved_at >= NOW() - INTERVAL '30 days'
                        """
                    )
                    if row and row["resolved"]:
                        remediation_velocity = (row["resolved"] / 30) * 7

            return DashboardSummaryResponse(
                total_vulnerabilities=stats.get("total", 0),
                open_vulnerabilities=stats.get("open", 0),
                resolved_vulnerabilities=stats.get("resolved", 0),
                accepted_vulnerabilities=stats.get("accepted", 0),
                critical_count=stats.get("by_severity", {}).get("critical", 0),
                high_count=stats.get("by_severity", {}).get("high", 0),
                medium_count=stats.get("by_severity", {}).get("medium", 0),
                low_count=stats.get("by_severity", {}).get("low", 0),
                kev_count=stats.get("kev_count", 0),
                sla_breached_count=stats.get("sla_breached", 0),
                avg_risk_score=stats.get("avg_risk_score", 0.0),
                avg_mttr_days=stats.get("avg_mttr_days", 0.0),
                remediation_velocity=remediation_velocity,
                total_scans_24h=scans_24h,
                findings_24h=findings_24h,
                scanners_active=scanners_active,
                last_scan_at=last_scan_at,
                generated_at=datetime.utcnow(),
            )

        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Failed to get dashboard summary")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get dashboard: {exc}",
            )

    @dashboard_router.get(
        "/trends",
        response_model=TrendsResponse,
        summary="Get vulnerability trends",
        description="Retrieve 90-day vulnerability trend data.",
        operation_id="get_vulnerability_trends",
    )
    async def get_vulnerability_trends(
        request: Request,
        days: int = Query(90, ge=7, le=365, description="Number of days"),
        db_pool: Any = Depends(_get_db_pool),
    ) -> TrendsResponse:
        """Get vulnerability trends over time."""
        tenant_id = _get_tenant_id(request)

        # Generate mock trend data for now
        data_points = []
        start_date = datetime.utcnow() - timedelta(days=days)

        for i in range(days):
            date = start_date + timedelta(days=i)
            # Simulated trend (in production, query from DB)
            base = 100 + i * 2  # Gradual increase
            data_points.append(
                TrendDataPoint(
                    date=date,
                    open=max(0, base - (i % 7) * 3),
                    resolved=i * 2,
                    critical=max(0, 5 - (i // 10)),
                    high=max(0, 15 - (i // 5)),
                    medium=30 + (i % 10),
                    low=50 + (i % 15),
                )
            )

        if db_pool:
            # In production, query actual data
            # SELECT date_trunc('day', discovered_at) as date, ...
            pass

        # Calculate summary
        if data_points:
            first_open = data_points[0].open
            last_open = data_points[-1].open
            change = last_open - first_open
            change_pct = (change / first_open * 100) if first_open > 0 else 0
        else:
            change = 0
            change_pct = 0

        summary = {
            "period_start": start_date.isoformat(),
            "period_end": datetime.utcnow().isoformat(),
            "open_change": change,
            "open_change_percent": round(change_pct, 1),
            "total_resolved": sum(d.resolved for d in data_points),
            "avg_daily_findings": round(sum(d.open for d in data_points) / max(1, len(data_points)), 1),
        }

        return TrendsResponse(
            time_range_days=days,
            data_points=data_points,
            summary=summary,
        )

    @dashboard_router.get(
        "/coverage",
        response_model=CoverageResponse,
        summary="Get scanner coverage",
        description="Retrieve scanner coverage and health metrics.",
        operation_id="get_scanner_coverage",
    )
    async def get_scanner_coverage(
        request: Request,
        db_pool: Any = Depends(_get_db_pool),
    ) -> CoverageResponse:
        """Get scanner coverage metrics."""
        tenant_id = _get_tenant_id(request)

        # Define expected scanners
        expected_scanners = [
            ("bandit", "sast"),
            ("semgrep", "sast"),
            ("codeql", "sast"),
            ("trivy", "sca"),
            ("snyk", "sca"),
            ("gitleaks", "secret"),
            ("trufflehog", "secret"),
            ("tfsec", "iac"),
            ("checkov", "iac"),
            ("owasp_zap", "dast"),
        ]

        scanners = []
        healthy = 0

        if db_pool:
            async with db_pool.acquire() as conn:
                for scanner_name, scanner_type in expected_scanners:
                    row = await conn.fetchrow(
                        """
                        SELECT
                            MAX(completed_at) as last_scan,
                            COUNT(*) FILTER (WHERE started_at >= NOW() - INTERVAL '7 days') as scans_7d,
                            SUM(findings_total) FILTER (WHERE started_at >= NOW() - INTERVAL '7 days') as findings_7d,
                            AVG(duration_seconds) as avg_duration,
                            COUNT(*) FILTER (WHERE status = 'completed') * 100.0 / NULLIF(COUNT(*), 0) as success_rate
                        FROM security.scan_runs
                        WHERE scanner = $1
                        """,
                        scanner_name,
                    )

                    if row and row["last_scan"]:
                        is_healthy = (
                            row["scans_7d"] > 0 and
                            (row["success_rate"] or 0) >= 80
                        )
                        scanner_status = "healthy" if is_healthy else "degraded"
                        if is_healthy:
                            healthy += 1
                    else:
                        scanner_status = "inactive"

                    scanners.append(
                        ScannerCoverage(
                            scanner=scanner_name,
                            scanner_type=scanner_type,
                            last_scan_at=row["last_scan"] if row else None,
                            scans_7d=row["scans_7d"] if row else 0,
                            findings_7d=row["findings_7d"] or 0 if row else 0,
                            avg_duration_seconds=float(row["avg_duration"] or 0) if row else 0,
                            success_rate=float(row["success_rate"] or 0) if row else 0,
                            status=scanner_status,
                        )
                    )
        else:
            # Mock data when no database
            for scanner_name, scanner_type in expected_scanners:
                scanners.append(
                    ScannerCoverage(
                        scanner=scanner_name,
                        scanner_type=scanner_type,
                        last_scan_at=None,
                        scans_7d=0,
                        findings_7d=0,
                        avg_duration_seconds=0.0,
                        success_rate=0.0,
                        status="inactive",
                    )
                )

        coverage_score = (healthy / len(expected_scanners) * 100) if expected_scanners else 0

        return CoverageResponse(
            scanners=scanners,
            total_scanners=len(expected_scanners),
            healthy_scanners=healthy,
            coverage_score=round(coverage_score, 1),
        )

    @dashboard_router.get(
        "/sla",
        response_model=SLAComplianceResponse,
        summary="Get SLA compliance",
        description="Retrieve SLA compliance metrics by severity.",
        operation_id="get_sla_compliance",
    )
    async def get_sla_compliance(
        request: Request,
        service: Any = Depends(_get_vulnerability_service),
    ) -> SLAComplianceResponse:
        """Get SLA compliance metrics."""
        tenant_id = _get_tenant_id(request)

        try:
            sla_report = await service.get_sla_status(tenant_id=tenant_id)

            # Build by-severity breakdown
            severity_sla = {
                "critical": 1,
                "high": 7,
                "medium": 30,
                "low": 90,
            }

            by_severity = []
            for severity, max_days in severity_sla.items():
                sev_data = sla_report.by_severity.get(severity, {})
                within = sev_data.get("within", 0)
                breached = sev_data.get("breached", 0)
                total = within + breached

                compliance_rate = (within / total * 100) if total > 0 else 100.0

                by_severity.append(
                    SLAStatusBySeverity(
                        severity=severity,
                        total_open=total,
                        within_sla=within,
                        approaching_sla=0,  # Would need more detailed data
                        breached_sla=breached,
                        compliance_rate=round(compliance_rate, 1),
                        max_sla_days=max_days,
                    )
                )

            # Calculate overall compliance
            overall = (
                (sla_report.within_sla / sla_report.total_open * 100)
                if sla_report.total_open > 0
                else 100.0
            )

            return SLAComplianceResponse(
                overall_compliance_rate=round(overall, 1),
                total_open=sla_report.total_open,
                within_sla=sla_report.within_sla,
                approaching_sla=sla_report.approaching_sla,
                breached_sla=sla_report.breached_sla,
                by_severity=by_severity,
                oldest_breach_date=sla_report.oldest_breach,
                average_age_days=round(sla_report.average_age_days, 1),
            )

        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Failed to get SLA compliance")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get SLA compliance: {exc}",
            )

    # SEC-001: Apply authentication and permission protection
    try:
        from greenlang.infrastructure.auth_service.route_protector import (
            protect_router,
        )

        protect_router(dashboard_router)
    except ImportError:
        pass

else:
    dashboard_router = None  # type: ignore[assignment]
    logger.warning("FastAPI not available - dashboard_router is None")


__all__ = ["dashboard_router"]
