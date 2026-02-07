# -*- coding: utf-8 -*-
"""
Compliance Automation API Routes - SEC-010 Phase 5

FastAPI routes for multi-compliance automation including ISO 27001, GDPR,
PCI-DSS, CCPA, and LGPD. Provides endpoints for compliance status, DSAR
processing, consent management, and framework-specific operations.

Endpoints:
    Compliance Status:
        GET  /compliance/status - Overall compliance dashboard
        GET  /compliance/iso27001 - ISO 27001 compliance status
        GET  /compliance/iso27001/soa - Statement of Applicability
        GET  /compliance/gdpr - GDPR compliance status
        GET  /compliance/pci-dss - PCI-DSS compliance status

    DSAR Processing:
        POST /dsar - Submit DSAR request (public)
        GET  /dsar - List DSAR requests
        GET  /dsar/{request_id} - Get DSAR details
        POST /dsar/{request_id}/verify - Verify identity
        POST /dsar/{request_id}/execute - Execute DSAR
        GET  /dsar/{request_id}/download - Download data export

    Consent Management:
        POST /consent - Record consent
        GET  /consent/{user_id} - Get user consent summary
        DELETE /consent/{user_id}/{purpose} - Revoke consent
        GET  /consent/{user_id}/audit - Get consent audit trail

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-010 Security Operations Automation Platform
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, EmailStr, Field

from greenlang.infrastructure.compliance_automation.models import (
    ComplianceFramework,
    ComplianceStatus,
    ConsentPurpose,
    DSARStatus,
    DSARType,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Compliance Automation"])


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------


class ComplianceDashboardResponse(BaseModel):
    """Response model for overall compliance dashboard."""

    overall_score: float = Field(..., description="Overall compliance score (0-100)")
    last_assessed: Optional[datetime] = Field(None, description="Last assessment time")
    frameworks: Dict[str, Any] = Field(default_factory=dict, description="Framework statuses")
    pending_dsars: int = Field(0, description="Pending DSAR requests")
    dsar_sla_compliance: float = Field(100.0, description="DSAR SLA compliance %")
    next_assessment: Optional[datetime] = Field(None, description="Next scheduled assessment")


class FrameworkStatusResponse(BaseModel):
    """Response model for framework compliance status."""

    framework: str
    framework_name: str
    version: str
    score: float
    status: str
    controls_total: int
    controls_compliant: int
    controls_non_compliant: int
    controls_not_applicable: int
    gaps_count: int
    last_assessed: Optional[datetime]
    next_assessment: Optional[datetime]


class SOAResponse(BaseModel):
    """Response model for Statement of Applicability."""

    framework: str
    version: str
    generated_at: datetime
    summary: Dict[str, Any]
    controls: List[Dict[str, Any]]


class DSARSubmitRequest(BaseModel):
    """Request model for DSAR submission."""

    request_type: DSARType = Field(..., description="Type of DSAR request")
    subject_email: EmailStr = Field(..., description="Email of the data subject")
    subject_name: str = Field("", description="Name of the data subject")
    regulation: str = Field("gdpr", description="Regulation (gdpr, ccpa, lgpd)")
    description: str = Field("", description="Request description")


class DSARResponse(BaseModel):
    """Response model for DSAR details."""

    id: str
    request_number: str
    request_type: str
    subject_email: str
    status: str
    submitted_at: datetime
    due_date: datetime
    days_remaining: int
    is_overdue: bool
    completed_at: Optional[datetime]
    export_url: Optional[str]


class DSARListResponse(BaseModel):
    """Response model for DSAR list."""

    total: int
    pending: int
    completed: int
    overdue: int
    requests: List[DSARResponse]


class ConsentRecordRequest(BaseModel):
    """Request model for recording consent."""

    user_id: str = Field(..., description="User ID")
    purpose: ConsentPurpose = Field(..., description="Consent purpose")
    source: str = Field("api", description="How consent was obtained")


class ConsentSummaryResponse(BaseModel):
    """Response model for user consent summary."""

    user_id: str
    total_purposes: int
    active_consents: int
    revoked_consents: int
    consents_by_purpose: Dict[str, bool]
    last_updated: Optional[datetime]


class ConsentAuditResponse(BaseModel):
    """Response model for consent audit trail."""

    user_id: str
    entries: List[Dict[str, Any]]
    total_entries: int


class VerifyIdentityRequest(BaseModel):
    """Request model for identity verification."""

    method: str = Field("email", description="Verification method")
    verification_data: Optional[Dict[str, Any]] = Field(None, description="Verification data")


class ExecuteDSARRequest(BaseModel):
    """Request model for executing a DSAR."""

    export_format: str = Field("json", description="Export format (json, csv)")


# ---------------------------------------------------------------------------
# Compliance Status Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/compliance/status",
    response_model=ComplianceDashboardResponse,
    summary="Get overall compliance dashboard",
    description="Returns an overview of compliance status across all frameworks.",
)
async def get_compliance_dashboard() -> ComplianceDashboardResponse:
    """Get the overall compliance dashboard."""
    logger.info("Fetching compliance dashboard")

    # In production, fetch from actual assessment data
    from greenlang.infrastructure.compliance_automation.iso27001 import ISO27001Mapper
    from greenlang.infrastructure.compliance_automation.gdpr import DSARProcessor

    # Get ISO 27001 status
    iso_mapper = ISO27001Mapper()
    iso_status = await iso_mapper.calculate_compliance_score()

    # Get DSAR stats
    dsar_processor = DSARProcessor()
    pending_dsars = await dsar_processor.get_pending_requests()
    overdue_dsars = await dsar_processor.get_overdue_requests()

    # Calculate SLA compliance
    if pending_dsars:
        sla_compliance = ((len(pending_dsars) - len(overdue_dsars)) / len(pending_dsars)) * 100
    else:
        sla_compliance = 100.0

    return ComplianceDashboardResponse(
        overall_score=iso_status.score,
        last_assessed=iso_status.last_assessed,
        frameworks={
            "iso27001": {
                "score": iso_status.score,
                "status": iso_status.status,
            },
            "gdpr": {
                "score": 95.0,  # Placeholder
                "status": "compliant",
            },
            "pci_dss": {
                "score": 100.0,  # Placeholder (not storing PAN)
                "status": "compliant",
            },
        },
        pending_dsars=len(pending_dsars),
        dsar_sla_compliance=sla_compliance,
        next_assessment=iso_status.next_assessment,
    )


@router.get(
    "/compliance/iso27001",
    response_model=FrameworkStatusResponse,
    summary="Get ISO 27001 compliance status",
    description="Returns detailed ISO 27001:2022 compliance status.",
)
async def get_iso27001_status() -> FrameworkStatusResponse:
    """Get ISO 27001 compliance status."""
    logger.info("Fetching ISO 27001 compliance status")

    from greenlang.infrastructure.compliance_automation.iso27001 import ISO27001Mapper

    mapper = ISO27001Mapper()
    status = await mapper.calculate_compliance_score()

    return FrameworkStatusResponse(
        framework="iso27001",
        framework_name="ISO/IEC 27001:2022",
        version="2022",
        score=status.score,
        status=status.status,
        controls_total=status.controls_total,
        controls_compliant=status.controls_compliant,
        controls_non_compliant=status.controls_non_compliant,
        controls_not_applicable=status.controls_not_applicable,
        gaps_count=len(status.gaps),
        last_assessed=status.last_assessed,
        next_assessment=status.next_assessment,
    )


@router.get(
    "/compliance/iso27001/soa",
    response_model=SOAResponse,
    summary="Get ISO 27001 Statement of Applicability",
    description="Returns the Statement of Applicability (SoA) for ISO 27001.",
)
async def get_iso27001_soa() -> SOAResponse:
    """Get ISO 27001 Statement of Applicability."""
    logger.info("Generating ISO 27001 SoA")

    from greenlang.infrastructure.compliance_automation.iso27001 import ISO27001Mapper

    mapper = ISO27001Mapper()
    soa = await mapper.generate_soa()

    return SOAResponse(
        framework="iso27001",
        version=soa.get("version", "2022"),
        generated_at=datetime.now(timezone.utc),
        summary=soa.get("summary", {}),
        controls=soa.get("controls", []),
    )


@router.get(
    "/compliance/gdpr",
    response_model=FrameworkStatusResponse,
    summary="Get GDPR compliance status",
    description="Returns GDPR compliance status including DSAR metrics.",
)
async def get_gdpr_status() -> FrameworkStatusResponse:
    """Get GDPR compliance status."""
    logger.info("Fetching GDPR compliance status")

    # GDPR compliance is assessed through DSAR processing and consent management
    from greenlang.infrastructure.compliance_automation.gdpr import DSARProcessor

    processor = DSARProcessor()
    pending = await processor.get_pending_requests()
    overdue = await processor.get_overdue_requests()

    # Calculate compliance score based on DSAR SLA adherence
    if pending:
        score = ((len(pending) - len(overdue)) / len(pending)) * 100
    else:
        score = 100.0

    return FrameworkStatusResponse(
        framework="gdpr",
        framework_name="EU General Data Protection Regulation",
        version="2018",
        score=score,
        status="compliant" if score >= 95 else "partial",
        controls_total=8,  # GDPR Articles 15-22
        controls_compliant=8 if score >= 95 else 6,
        controls_non_compliant=0 if score >= 95 else 2,
        controls_not_applicable=0,
        gaps_count=0 if score >= 95 else 1,
        last_assessed=datetime.now(timezone.utc),
        next_assessment=None,
    )


@router.get(
    "/compliance/pci-dss",
    response_model=FrameworkStatusResponse,
    summary="Get PCI-DSS compliance status",
    description="Returns PCI-DSS v4.0 compliance status.",
)
async def get_pci_dss_status() -> FrameworkStatusResponse:
    """Get PCI-DSS compliance status."""
    logger.info("Fetching PCI-DSS compliance status")

    from greenlang.infrastructure.compliance_automation.pci_dss import (
        CardDataMapper,
        EncryptionChecker,
    )

    # Check CDE scope
    mapper = CardDataMapper()
    scope = await mapper.identify_cde_scope()

    # Check encryption
    checker = EncryptionChecker()
    assessment = await checker.run_full_assessment()

    score = 100.0 if assessment["overall_compliant"] else 85.0

    return FrameworkStatusResponse(
        framework="pci_dss",
        framework_name="Payment Card Industry Data Security Standard",
        version="4.0",
        score=score,
        status="compliant" if assessment["overall_compliant"] else "partial",
        controls_total=12,
        controls_compliant=12 if assessment["overall_compliant"] else 10,
        controls_non_compliant=0 if assessment["overall_compliant"] else 2,
        controls_not_applicable=0,
        gaps_count=0 if assessment["overall_compliant"] else len(assessment.get("recommendations", [])),
        last_assessed=datetime.now(timezone.utc),
        next_assessment=None,
    )


# ---------------------------------------------------------------------------
# DSAR Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/dsar",
    response_model=DSARResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a DSAR request",
    description="Submit a new Data Subject Access Request (public endpoint).",
)
async def submit_dsar(request: DSARSubmitRequest) -> DSARResponse:
    """Submit a new DSAR request."""
    logger.info(
        "DSAR submission: type=%s, email=%s",
        request.request_type.value,
        request.subject_email,
    )

    from greenlang.infrastructure.compliance_automation.models import DSARRequest
    from greenlang.infrastructure.compliance_automation.gdpr import DSARProcessor

    processor = DSARProcessor()

    dsar = DSARRequest(
        request_type=request.request_type,
        subject_email=request.subject_email,
        subject_name=request.subject_name,
    )

    submitted = await processor.submit_request(dsar)

    return DSARResponse(
        id=submitted.id,
        request_number=submitted.request_number,
        request_type=submitted.request_type.value,
        subject_email=submitted.subject_email,
        status=submitted.status.value,
        submitted_at=submitted.submitted_at,
        due_date=submitted.due_date,
        days_remaining=submitted.days_remaining,
        is_overdue=submitted.is_overdue,
        completed_at=submitted.completed_at,
        export_url=submitted.export_file_url,
    )


@router.get(
    "/dsar",
    response_model=DSARListResponse,
    summary="List DSAR requests",
    description="List all DSAR requests with optional filtering.",
)
async def list_dsars(
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
) -> DSARListResponse:
    """List DSAR requests."""
    logger.info("Listing DSARs: status=%s, limit=%d, offset=%d", status_filter, limit, offset)

    from greenlang.infrastructure.compliance_automation.gdpr import DSARProcessor

    processor = DSARProcessor()
    pending = await processor.get_pending_requests()
    overdue = await processor.get_overdue_requests()

    # Convert to response format
    requests_list: List[DSARResponse] = []
    for req in pending[offset:offset + limit]:
        requests_list.append(DSARResponse(
            id=req.id,
            request_number=req.request_number,
            request_type=req.request_type.value,
            subject_email=req.subject_email,
            status=req.status.value,
            submitted_at=req.submitted_at,
            due_date=req.due_date,
            days_remaining=req.days_remaining,
            is_overdue=req.is_overdue,
            completed_at=req.completed_at,
            export_url=req.export_file_url,
        ))

    return DSARListResponse(
        total=len(pending),
        pending=len([r for r in pending if r.status != DSARStatus.COMPLETED]),
        completed=0,  # Would need to track completed separately
        overdue=len(overdue),
        requests=requests_list,
    )


@router.get(
    "/dsar/{request_id}",
    response_model=DSARResponse,
    summary="Get DSAR details",
    description="Get detailed information about a specific DSAR request.",
)
async def get_dsar(request_id: str) -> DSARResponse:
    """Get DSAR details by ID."""
    logger.info("Fetching DSAR: %s", request_id)

    from greenlang.infrastructure.compliance_automation.gdpr import DSARProcessor

    processor = DSARProcessor()
    dsar = await processor.get_request(request_id)

    if not dsar:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"DSAR request not found: {request_id}",
        )

    return DSARResponse(
        id=dsar.id,
        request_number=dsar.request_number,
        request_type=dsar.request_type.value,
        subject_email=dsar.subject_email,
        status=dsar.status.value,
        submitted_at=dsar.submitted_at,
        due_date=dsar.due_date,
        days_remaining=dsar.days_remaining,
        is_overdue=dsar.is_overdue,
        completed_at=dsar.completed_at,
        export_url=dsar.export_file_url,
    )


@router.post(
    "/dsar/{request_id}/verify",
    summary="Verify identity for DSAR",
    description="Verify the identity of the data subject.",
)
async def verify_dsar_identity(
    request_id: str,
    verification: VerifyIdentityRequest,
) -> Dict[str, Any]:
    """Verify identity for DSAR."""
    logger.info("Verifying identity for DSAR: %s, method=%s", request_id, verification.method)

    from greenlang.infrastructure.compliance_automation.gdpr import DSARProcessor

    processor = DSARProcessor()

    try:
        result = await processor.verify_identity(
            request_id=request_id,
            method=verification.method,
            verification_data=verification.verification_data,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    return {
        "request_id": request_id,
        "verified": result.success,
        "method": result.method,
        "confidence": result.confidence,
        "verified_at": result.verified_at.isoformat() if result.verified_at else None,
    }


@router.post(
    "/dsar/{request_id}/execute",
    summary="Execute DSAR request",
    description="Execute the DSAR request (access, erasure, etc.).",
)
async def execute_dsar(
    request_id: str,
    execution: ExecuteDSARRequest,
) -> Dict[str, Any]:
    """Execute DSAR request."""
    logger.info("Executing DSAR: %s", request_id)

    from greenlang.infrastructure.compliance_automation.gdpr import DSARProcessor

    processor = DSARProcessor()
    dsar = await processor.get_request(request_id)

    if not dsar:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"DSAR request not found: {request_id}",
        )

    # Execute based on request type
    if dsar.request_type == DSARType.ACCESS:
        result = await processor.execute_access(request_id, execution.export_format)
    elif dsar.request_type == DSARType.ERASURE:
        result = await processor.execute_erasure(request_id)
    elif dsar.request_type == DSARType.PORTABILITY:
        result = await processor.execute_portability(request_id, execution.export_format)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported request type: {dsar.request_type.value}",
        )

    return {
        "request_id": request_id,
        "success": result.success,
        "request_type": result.request_type.value,
        "records_affected": result.records_affected,
        "export_url": result.export_url,
        "deletion_certificate_id": result.deletion_certificate_id,
        "completed_at": result.completed_at.isoformat() if result.completed_at else None,
        "errors": result.errors,
    }


@router.get(
    "/dsar/{request_id}/download",
    summary="Download DSAR export",
    description="Get download URL for DSAR data export.",
)
async def download_dsar_export(request_id: str) -> Dict[str, Any]:
    """Get download URL for DSAR export."""
    logger.info("Getting download URL for DSAR: %s", request_id)

    from greenlang.infrastructure.compliance_automation.gdpr import DSARProcessor

    processor = DSARProcessor()
    dsar = await processor.get_request(request_id)

    if not dsar:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"DSAR request not found: {request_id}",
        )

    if not dsar.export_file_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Export not available for this request",
        )

    return {
        "request_id": request_id,
        "download_url": dsar.export_file_url,
        "expires_in_hours": 24,
    }


# ---------------------------------------------------------------------------
# Consent Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/consent",
    status_code=status.HTTP_201_CREATED,
    summary="Record consent",
    description="Record a new consent grant for a user.",
)
async def record_consent(request: ConsentRecordRequest) -> Dict[str, Any]:
    """Record consent grant."""
    logger.info("Recording consent: user=%s, purpose=%s", request.user_id, request.purpose.value)

    from greenlang.infrastructure.compliance_automation.gdpr import ConsentManager

    manager = ConsentManager()
    record = await manager.record_consent(
        user_id=request.user_id,
        purpose=request.purpose,
        source=request.source,
    )

    return {
        "consent_id": record.id,
        "user_id": record.user_id,
        "purpose": record.purpose.value,
        "granted_at": record.granted_at.isoformat(),
        "consent_version": record.consent_version,
    }


@router.get(
    "/consent/{user_id}",
    response_model=ConsentSummaryResponse,
    summary="Get user consent summary",
    description="Get a summary of all consents for a user.",
)
async def get_user_consent(user_id: str) -> ConsentSummaryResponse:
    """Get user consent summary."""
    logger.info("Fetching consent summary for user: %s", user_id)

    from greenlang.infrastructure.compliance_automation.gdpr import ConsentManager

    manager = ConsentManager()
    summary = await manager.get_user_consent_summary(user_id)

    return ConsentSummaryResponse(
        user_id=summary.user_id,
        total_purposes=summary.total_purposes,
        active_consents=summary.active_consents,
        revoked_consents=summary.revoked_consents,
        consents_by_purpose=summary.consents_by_purpose,
        last_updated=summary.last_updated,
    )


@router.delete(
    "/consent/{user_id}/{purpose}",
    status_code=status.HTTP_200_OK,
    summary="Revoke consent",
    description="Revoke consent for a specific purpose.",
)
async def revoke_consent(
    user_id: str,
    purpose: ConsentPurpose,
    reason: Optional[str] = Query(None, description="Reason for revocation"),
) -> Dict[str, Any]:
    """Revoke consent."""
    logger.info("Revoking consent: user=%s, purpose=%s", user_id, purpose.value)

    from greenlang.infrastructure.compliance_automation.gdpr import ConsentManager

    manager = ConsentManager()
    revoked = await manager.revoke_consent(
        user_id=user_id,
        purpose=purpose,
        reason=reason,
    )

    if not revoked:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active consent found for user {user_id} and purpose {purpose.value}",
        )

    return {
        "user_id": user_id,
        "purpose": purpose.value,
        "revoked": True,
        "revoked_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get(
    "/consent/{user_id}/audit",
    response_model=ConsentAuditResponse,
    summary="Get consent audit trail",
    description="Get the audit trail of consent changes for a user.",
)
async def get_consent_audit(
    user_id: str,
    purpose: Optional[ConsentPurpose] = Query(None, description="Filter by purpose"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
) -> ConsentAuditResponse:
    """Get consent audit trail."""
    logger.info("Fetching consent audit for user: %s", user_id)

    from greenlang.infrastructure.compliance_automation.gdpr import ConsentManager

    manager = ConsentManager()
    entries = await manager.audit_consent_trail(
        user_id=user_id,
        purpose=purpose,
    )

    # Convert to response format
    entry_dicts = []
    for entry in entries[:limit]:
        entry_dicts.append({
            "id": entry.id,
            "action": entry.action,
            "purpose": entry.purpose.value,
            "timestamp": entry.timestamp.isoformat(),
            "performed_by": entry.performed_by,
            "details": entry.details,
        })

    return ConsentAuditResponse(
        user_id=user_id,
        entries=entry_dicts,
        total_entries=len(entries),
    )


__all__ = [
    "router",
]
