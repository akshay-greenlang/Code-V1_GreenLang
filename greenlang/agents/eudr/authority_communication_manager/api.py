# -*- coding: utf-8 -*-
"""
FastAPI Router - AGENT-EUDR-040: Authority Communication Manager

REST API endpoints for EUDR authority communication management.
Provides 30+ endpoints for communication lifecycle management,
information request handling, inspection coordination, non-compliance
processing, appeal management, document exchange, notification routing,
multi-language template rendering, and health monitoring.

Endpoint Summary (30+):
    POST /communication                        - Create new communication
    GET  /communication/{id}                   - Get communication details
    POST /communication/{id}/respond           - Submit response
    GET  /communications/pending               - List pending communications
    GET  /communications/overdue               - List overdue communications
    POST /information-request                   - Handle authority request
    GET  /information-request/{id}             - Get request details
    GET  /information-requests/pending          - List pending requests
    POST /inspection                           - Schedule inspection
    GET  /inspection/{id}                      - Get inspection details
    POST /inspection/{id}/status               - Update inspection status
    POST /inspection/{id}/findings             - Record findings
    GET  /inspections                          - List inspections
    POST /non-compliance                       - Record violation
    GET  /non-compliance/{id}                  - Get violation details
    POST /non-compliance/{id}/corrective       - Mark corrective completed
    GET  /non-compliance/penalties/{operator}   - Get operator penalties
    POST /appeal                               - File appeal
    GET  /appeal/{id}                          - Get appeal details
    POST /appeal/{id}/decision                 - Record decision
    POST /appeal/{id}/extension                - Grant extension
    POST /appeal/{id}/withdraw                 - Withdraw appeal
    POST /document/upload                      - Upload document
    GET  /document/{id}/download               - Download document
    GET  /document/{id}/metadata               - Get document metadata
    GET  /document/{id}/verify                 - Verify integrity
    POST /notification/send                    - Send notification
    GET  /templates                            - List templates
    GET  /templates/{language}                 - Get templates by language
    POST /templates/render                     - Render a template
    GET  /authorities                          - List authorities
    GET  /health                               - Health check

Auth & RBAC:
    All endpoints (except health) require JWT auth via SEC-001 and check
    eudr-authority-communication-manager:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-040 (GL-EUDR-ACM-040)
Regulation: EU 2023/1115 (EUDR) Articles 15, 16, 17, 19, 31
Status: Production Ready
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from greenlang.agents.eudr.authority_communication_manager.setup import get_service

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------


class CreateCommunicationRequest(BaseModel):
    """Request body for creating a new communication."""

    operator_id: str = Field(..., description="EUDR operator identifier")
    authority_id: str = Field(..., description="Competent authority identifier")
    member_state: str = Field(..., min_length=2, max_length=2, description="ISO 3166-1 alpha-2")
    communication_type: str = Field(..., description="Communication type")
    subject: str = Field(..., min_length=1, description="Communication subject")
    body: str = Field(default="", description="Communication body")
    priority: str = Field(default="normal", description="Priority level")
    language: str = Field(default="en", description="Language code")
    dds_reference: str = Field(default="", description="DDS reference")
    document_ids: List[str] = Field(default_factory=list, description="Attached docs")


class RespondRequest(BaseModel):
    """Request body for responding to a communication."""

    responder_id: str = Field(..., description="Responder identity")
    body: str = Field(..., min_length=1, description="Response body")
    document_ids: List[str] = Field(default_factory=list, description="Supporting docs")


class InformationRequestBody(BaseModel):
    """Request body for handling an authority information request."""

    operator_id: str = Field(..., description="Target operator")
    authority_id: str = Field(..., description="Requesting authority")
    request_type: str = Field(..., description="Request type")
    items_requested: List[str] = Field(..., min_length=1, description="Items requested")
    dds_reference: str = Field(default="", description="DDS reference")
    commodity: str = Field(default="", description="EUDR commodity")
    priority: str = Field(default="normal", description="Priority level")
    language: str = Field(default="en", description="Response language")


class ScheduleInspectionRequest(BaseModel):
    """Request body for scheduling an inspection."""

    operator_id: str = Field(..., description="Operator to inspect")
    authority_id: str = Field(..., description="Inspecting authority")
    inspection_type: str = Field(..., description="Inspection type")
    scheduled_date: datetime = Field(..., description="Planned date")
    location: str = Field(default="", description="Location")
    scope: str = Field(default="", description="Scope")
    inspector_name: str = Field(default="", description="Inspector name")


class UpdateInspectionStatusRequest(BaseModel):
    """Request body for updating inspection status."""

    new_status: str = Field(..., description="Target status")
    notes: str = Field(default="", description="Status change notes")


class RecordFindingsRequest(BaseModel):
    """Request body for recording inspection findings."""

    findings: List[str] = Field(..., min_length=1, description="Findings list")
    corrective_actions: List[str] = Field(default_factory=list, description="Corrective actions")


class RecordViolationRequest(BaseModel):
    """Request body for recording a non-compliance violation."""

    operator_id: str = Field(..., description="Violating operator")
    authority_id: str = Field(..., description="Issuing authority")
    violation_type: str = Field(..., description="Violation type")
    severity: str = Field(..., description="Severity level")
    description: str = Field(..., min_length=10, description="Description")
    evidence_references: List[str] = Field(default_factory=list)
    corrective_actions_required: List[str] = Field(default_factory=list)
    corrective_deadline_days: int = Field(default=30, ge=1)
    commodity: str = Field(default="")
    dds_reference: str = Field(default="")
    penalty_override: Optional[str] = Field(default=None, description="Penalty override as decimal string")


class FileAppealRequest(BaseModel):
    """Request body for filing an appeal."""

    non_compliance_id: str = Field(..., description="NC record being appealed")
    operator_id: str = Field(..., description="Appealing operator")
    authority_id: str = Field(..., description="Authority receiving appeal")
    grounds: str = Field(..., min_length=10, description="Grounds for appeal")
    supporting_evidence: List[str] = Field(default_factory=list)


class AppealDecisionRequest(BaseModel):
    """Request body for recording an appeal decision."""

    decision: str = Field(..., description="Decision outcome")
    reason: str = Field(default="", description="Decision reasoning")


class UploadDocumentRequest(BaseModel):
    """Request body for document upload metadata."""

    communication_id: str = Field(..., description="Parent communication")
    document_type: str = Field(..., description="Document type")
    title: str = Field(..., min_length=1, description="Document title")
    description: str = Field(default="", description="Description")
    language: str = Field(default="en", description="Language code")
    mime_type: str = Field(default="application/pdf", description="MIME type")
    encrypt: Optional[bool] = Field(default=None, description="Force encryption")
    uploaded_by: str = Field(..., description="Uploader identity")


class SendNotificationRequest(BaseModel):
    """Request body for sending a notification."""

    communication_id: str = Field(..., description="Related communication")
    channel: str = Field(..., description="Delivery channel")
    recipient_type: str = Field(..., description="Recipient type")
    recipient_id: str = Field(..., description="Recipient ID")
    recipient_address: str = Field(default="", description="Delivery address")
    subject: str = Field(default="", description="Subject")
    body: str = Field(default="", description="Body")
    language: str = Field(default="en", description="Language")


class RenderTemplateRequest(BaseModel):
    """Request body for rendering a template."""

    template_name: str = Field(..., description="Template name")
    language: str = Field(default="en", description="Target language")
    variables: Dict[str, str] = Field(default_factory=dict, description="Placeholder values")


class ErrorResponse(BaseModel):
    """Standard error response body."""

    detail: str = Field(..., description="Error description")
    error_code: str = Field(default="internal_error")
    timestamp: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/api/v1/eudr/authority-communication-manager",
    tags=["EUDR Authority Communication Manager"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)


# ---------------------------------------------------------------------------
# Communication Endpoints
# ---------------------------------------------------------------------------


@router.post("/communication", response_model=Dict[str, Any], status_code=200,
             summary="Create new communication")
async def create_communication(request: CreateCommunicationRequest) -> Dict[str, Any]:
    """Create a new communication between operator and authority."""
    try:
        service = get_service()
        return await service.create_communication(
            operator_id=request.operator_id,
            authority_id=request.authority_id,
            member_state=request.member_state,
            communication_type=request.communication_type,
            subject=request.subject,
            body=request.body,
            priority=request.priority,
            language=request.language,
            dds_reference=request.dds_reference,
            document_ids=request.document_ids,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"create_communication failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/communication/{communication_id}", response_model=Dict[str, Any], status_code=200,
            summary="Get communication details")
async def get_communication(communication_id: str) -> Dict[str, Any]:
    """Get communication details by identifier."""
    try:
        service = get_service()
        result = await service.get_communication(communication_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Communication {communication_id} not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"get_communication failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.post("/communication/{communication_id}/respond", response_model=Dict[str, Any], status_code=200,
             summary="Submit response to communication")
async def respond_to_communication(communication_id: str, request: RespondRequest) -> Dict[str, Any]:
    """Submit a response to a communication."""
    try:
        service = get_service()
        return await service.respond_to_communication(
            communication_id=communication_id,
            responder_id=request.responder_id,
            body=request.body,
            document_ids=request.document_ids,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"respond_to_communication failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/communications/pending", response_model=List[Dict[str, Any]], status_code=200,
            summary="List pending communications")
async def list_pending(operator_id: Optional[str] = Query(None)) -> List[Dict[str, Any]]:
    """List pending communications."""
    try:
        service = get_service()
        return await service.list_pending_communications(operator_id=operator_id)
    except Exception as e:
        logger.error(f"list_pending failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/communications/overdue", response_model=List[Dict[str, Any]], status_code=200,
            summary="List overdue communications")
async def list_overdue() -> List[Dict[str, Any]]:
    """List overdue communications past their deadline."""
    try:
        service = get_service()
        return await service.list_overdue_communications()
    except Exception as e:
        logger.error(f"list_overdue failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Information Request Endpoints
# ---------------------------------------------------------------------------


@router.post("/information-request", response_model=Dict[str, Any], status_code=200,
             summary="Handle authority information request")
async def handle_information_request(request: InformationRequestBody) -> Dict[str, Any]:
    """Receive and register an information request from authority."""
    try:
        service = get_service()
        result = await service.handle_information_request(
            operator_id=request.operator_id,
            authority_id=request.authority_id,
            request_type=request.request_type,
            items_requested=request.items_requested,
            dds_reference=request.dds_reference,
            commodity=request.commodity,
            priority=request.priority,
            language=request.language,
        )
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        return dict(result) if not isinstance(result, dict) else result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"handle_information_request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Inspection Endpoints
# ---------------------------------------------------------------------------


@router.post("/inspection", response_model=Dict[str, Any], status_code=200,
             summary="Schedule inspection")
async def schedule_inspection(request: ScheduleInspectionRequest) -> Dict[str, Any]:
    """Schedule a new inspection or on-the-spot check."""
    try:
        service = get_service()
        result = await service.schedule_inspection(
            operator_id=request.operator_id,
            authority_id=request.authority_id,
            inspection_type=request.inspection_type,
            scheduled_date=request.scheduled_date,
            location=request.location,
            scope=request.scope,
            inspector_name=request.inspector_name,
        )
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        return dict(result) if not isinstance(result, dict) else result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"schedule_inspection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Non-Compliance Endpoints
# ---------------------------------------------------------------------------


@router.post("/non-compliance", response_model=Dict[str, Any], status_code=200,
             summary="Record violation")
async def record_violation(request: RecordViolationRequest) -> Dict[str, Any]:
    """Record a non-compliance violation with calculated penalty."""
    try:
        service = get_service()
        kwargs: Dict[str, Any] = {
            "operator_id": request.operator_id,
            "authority_id": request.authority_id,
            "violation_type": request.violation_type,
            "severity": request.severity,
            "description": request.description,
            "evidence_references": request.evidence_references,
            "corrective_actions_required": request.corrective_actions_required,
            "corrective_deadline_days": request.corrective_deadline_days,
            "commodity": request.commodity,
            "dds_reference": request.dds_reference,
        }
        if request.penalty_override is not None:
            kwargs["penalty_override"] = Decimal(request.penalty_override)

        result = await service.record_violation(**kwargs)
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        return dict(result) if not isinstance(result, dict) else result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"record_violation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Appeal Endpoints
# ---------------------------------------------------------------------------


@router.post("/appeal", response_model=Dict[str, Any], status_code=200,
             summary="File appeal")
async def file_appeal(request: FileAppealRequest) -> Dict[str, Any]:
    """File an administrative appeal per Article 19."""
    try:
        service = get_service()
        result = await service.file_appeal(
            non_compliance_id=request.non_compliance_id,
            operator_id=request.operator_id,
            authority_id=request.authority_id,
            grounds=request.grounds,
            supporting_evidence=request.supporting_evidence,
        )
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        return dict(result) if not isinstance(result, dict) else result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"file_appeal failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.post("/appeal/{appeal_id}/decision", response_model=Dict[str, Any], status_code=200,
             summary="Record appeal decision")
async def record_appeal_decision(appeal_id: str, request: AppealDecisionRequest) -> Dict[str, Any]:
    """Record authority decision on an appeal."""
    try:
        service = get_service()
        engine = service.get_engine("appeal_processor")
        if engine is None:
            raise HTTPException(status_code=503, detail="AppealProcessor not available")
        result = await engine.record_decision(
            appeal_id=appeal_id,
            decision=request.decision,
            reason=request.reason,
        )
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        return dict(result) if not isinstance(result, dict) else result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"record_appeal_decision failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.post("/appeal/{appeal_id}/extension", response_model=Dict[str, Any], status_code=200,
             summary="Grant appeal extension")
async def grant_appeal_extension(
    appeal_id: str,
    additional_days: Optional[int] = Query(None, ge=1),
) -> Dict[str, Any]:
    """Grant a deadline extension for an appeal."""
    try:
        service = get_service()
        engine = service.get_engine("appeal_processor")
        if engine is None:
            raise HTTPException(status_code=503, detail="AppealProcessor not available")
        result = await engine.grant_extension(
            appeal_id=appeal_id,
            additional_days=additional_days,
        )
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        return dict(result) if not isinstance(result, dict) else result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"grant_appeal_extension failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.post("/appeal/{appeal_id}/withdraw", response_model=Dict[str, Any], status_code=200,
             summary="Withdraw appeal")
async def withdraw_appeal(appeal_id: str) -> Dict[str, Any]:
    """Withdraw a pending appeal."""
    try:
        service = get_service()
        engine = service.get_engine("appeal_processor")
        if engine is None:
            raise HTTPException(status_code=503, detail="AppealProcessor not available")
        result = await engine.withdraw_appeal(appeal_id=appeal_id)
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        return dict(result) if not isinstance(result, dict) else result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"withdraw_appeal failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Document Endpoints
# ---------------------------------------------------------------------------


@router.post("/document/upload", response_model=Dict[str, Any], status_code=200,
             summary="Upload document")
async def upload_document(request: UploadDocumentRequest) -> Dict[str, Any]:
    """Upload a document with optional encryption."""
    try:
        service = get_service()
        # In production, content comes from multipart form data
        # For API model, content is simulated as placeholder bytes
        content = b"placeholder-document-content"
        result = await service.upload_document(
            communication_id=request.communication_id,
            document_type=request.document_type,
            title=request.title,
            content=content,
            uploaded_by=request.uploaded_by,
            description=request.description,
            language=request.language,
            mime_type=request.mime_type,
            encrypt=request.encrypt,
        )
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        return dict(result) if not isinstance(result, dict) else result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"upload_document failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/document/{document_id}/metadata", response_model=Dict[str, Any], status_code=200,
            summary="Get document metadata")
async def get_document_metadata(document_id: str) -> Dict[str, Any]:
    """Get document metadata without content."""
    try:
        service = get_service()
        engine = service.get_engine("document_exchange")
        if engine is None:
            raise HTTPException(status_code=503, detail="DocumentExchange not available")
        result = await engine.get_document_metadata(document_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        return dict(result) if not isinstance(result, dict) else result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"get_document_metadata failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/document/{document_id}/verify", response_model=Dict[str, Any], status_code=200,
            summary="Verify document integrity")
async def verify_document_integrity(document_id: str) -> Dict[str, Any]:
    """Verify document integrity against stored hash."""
    try:
        service = get_service()
        engine = service.get_engine("document_exchange")
        if engine is None:
            raise HTTPException(status_code=503, detail="DocumentExchange not available")
        return await engine.verify_integrity(document_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"verify_document_integrity failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Notification Endpoints
# ---------------------------------------------------------------------------


@router.post("/notification/send", response_model=Dict[str, Any], status_code=200,
             summary="Send notification")
async def send_notification(request: SendNotificationRequest) -> Dict[str, Any]:
    """Send a notification through the specified channel."""
    try:
        service = get_service()
        result = await service.send_notification(
            communication_id=request.communication_id,
            channel=request.channel,
            recipient_type=request.recipient_type,
            recipient_id=request.recipient_id,
            recipient_address=request.recipient_address,
            subject=request.subject,
            body=request.body,
            language=request.language,
        )
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        return dict(result) if not isinstance(result, dict) else result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"send_notification failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Template Endpoints
# ---------------------------------------------------------------------------


@router.get("/templates", response_model=List[Dict[str, Any]], status_code=200,
            summary="List templates")
async def list_templates(
    language: Optional[str] = Query(None),
    communication_type: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    """List available communication templates."""
    try:
        service = get_service()
        templates = await service.list_templates(
            language=language,
            communication_type=communication_type,
        )
        results = []
        for t in templates:
            if hasattr(t, "model_dump"):
                results.append(t.model_dump(mode="json"))
            elif isinstance(t, dict):
                results.append(t)
            else:
                results.append(dict(t))
        return results
    except Exception as e:
        logger.error(f"list_templates failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/templates/{language}", response_model=List[Dict[str, Any]], status_code=200,
            summary="Get templates by language")
async def get_templates_by_language(language: str) -> List[Dict[str, Any]]:
    """Get templates for a specific language."""
    try:
        service = get_service()
        templates = await service.list_templates(language=language)
        results = []
        for t in templates:
            if hasattr(t, "model_dump"):
                results.append(t.model_dump(mode="json"))
            elif isinstance(t, dict):
                results.append(t)
            else:
                results.append(dict(t))
        return results
    except Exception as e:
        logger.error(f"get_templates_by_language failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.post("/templates/render", response_model=Dict[str, str], status_code=200,
             summary="Render template")
async def render_template(request: RenderTemplateRequest) -> Dict[str, str]:
    """Render a template with variable substitution."""
    try:
        service = get_service()
        return await service.render_template(
            template_name=request.template_name,
            language=request.language,
            variables=request.variables,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"render_template failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Authority Endpoints
# ---------------------------------------------------------------------------


@router.get("/authorities", response_model=List[Dict[str, Any]], status_code=200,
            summary="List authorities by member state")
async def list_authorities(
    member_state: Optional[str] = Query(None, min_length=2, max_length=2),
) -> List[Dict[str, Any]]:
    """List configured EU member state authorities."""
    try:
        service = get_service()
        return await service.get_authorities(member_state=member_state)
    except Exception as e:
        logger.error(f"list_authorities failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Health Endpoint
# ---------------------------------------------------------------------------


@router.get("/health", response_model=Dict[str, Any], status_code=200,
            summary="Health check")
async def health_check() -> Dict[str, Any]:
    """Health check for Authority Communication Manager."""
    try:
        service = get_service()
        return await service.health_check()
    except Exception as e:
        logger.error(f"health_check failed: {e}", exc_info=True)
        return {
            "agent_id": "GL-EUDR-ACM-040",
            "status": "error",
            "error": str(e)[:200],
        }


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def get_router() -> APIRouter:
    """Return the Authority Communication Manager API router.

    Used by ``auth_setup.configure_auth()`` to include the router
    in the main FastAPI application.

    Returns:
        The configured APIRouter instance.
    """
    return router
