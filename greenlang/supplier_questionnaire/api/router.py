# -*- coding: utf-8 -*-
"""
Supplier Questionnaire Processor REST API Router - AGENT-DATA-008

FastAPI router providing 20 endpoints for questionnaire template management,
distribution, response collection, validation, scoring, follow-up management,
analytics, statistics, and health monitoring.

All endpoints are mounted under ``/api/v1/questionnaires``.

Endpoints:
    1.  POST   /v1/templates                          - Create template
    2.  GET    /v1/templates                           - List templates
    3.  GET    /v1/templates/{template_id}             - Get template
    4.  PUT    /v1/templates/{template_id}             - Update template
    5.  POST   /v1/templates/{template_id}/clone       - Clone template
    6.  POST   /v1/distribute                          - Distribute questionnaire
    7.  GET    /v1/distributions                       - List distributions
    8.  GET    /v1/distributions/{dist_id}             - Get distribution
    9.  POST   /v1/responses                           - Submit response
    10. GET    /v1/responses                           - List responses
    11. GET    /v1/responses/{response_id}             - Get response
    12. POST   /v1/responses/{response_id}/validate    - Validate response
    13. POST   /v1/score                               - Score response
    14. GET    /v1/scores/{score_id}                   - Get score
    15. GET    /v1/scores/supplier/{supplier_id}       - Get supplier scores
    16. POST   /v1/followup                            - Trigger follow-up
    17. GET    /v1/followup/{campaign_id}              - Get follow-up status
    18. GET    /v1/analytics/{campaign_id}             - Get analytics
    19. GET    /health                                 - Health check
    20. GET    /v1/statistics                          - Statistics

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-008 Supplier Questionnaire Processor
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
        "FastAPI not available; supplier questionnaire router is None"
    )


# ---------------------------------------------------------------------------
# Pydantic request/response models (only when FastAPI is available)
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class CreateTemplateBody(BaseModel):
        """Request body for creating a questionnaire template."""
        name: str = Field(
            ..., description="Template display name",
        )
        framework: str = Field(
            default="custom",
            description="Questionnaire framework (cdp, ecovadis, gri, custom)",
        )
        version: str = Field(
            default="1.0", description="Template version string",
        )
        description: str = Field(
            default="", description="Template description",
        )
        sections: List[Dict[str, Any]] = Field(
            default_factory=list,
            description="Ordered list of section definitions with questions",
        )
        language: str = Field(
            default="en", description="ISO 639-1 language code",
        )
        tags: List[str] = Field(
            default_factory=list, description="Classification tags",
        )

    class UpdateTemplateBody(BaseModel):
        """Request body for updating a questionnaire template."""
        name: Optional[str] = Field(
            None, description="New template name",
        )
        description: Optional[str] = Field(
            None, description="New description",
        )
        sections: Optional[List[Dict[str, Any]]] = Field(
            None, description="New sections",
        )
        status: Optional[str] = Field(
            None, description="New status (draft, active, archived)",
        )
        tags: Optional[List[str]] = Field(
            None, description="New tags",
        )

    class CloneTemplateBody(BaseModel):
        """Request body for cloning a template."""
        new_name: Optional[str] = Field(
            None, description="Name for the cloned template",
        )
        new_version: Optional[str] = Field(
            None, description="Version for the cloned template",
        )

    class DistributeBody(BaseModel):
        """Request body for distributing a questionnaire."""
        template_id: str = Field(
            ..., description="Template ID to distribute",
        )
        supplier_id: str = Field(
            ..., description="Target supplier identifier",
        )
        supplier_name: str = Field(
            ..., description="Target supplier display name",
        )
        supplier_email: str = Field(
            ..., description="Supplier contact email",
        )
        campaign_id: Optional[str] = Field(
            None, description="Campaign identifier (auto-generated if omitted)",
        )
        channel: str = Field(
            default="email",
            description="Distribution channel (email, portal, api, bulk)",
        )
        deadline: Optional[str] = Field(
            None, description="Response deadline (ISO 8601)",
        )

    class SubmitResponseBody(BaseModel):
        """Request body for submitting a questionnaire response."""
        distribution_id: str = Field(
            ..., description="Linked distribution identifier",
        )
        supplier_id: str = Field(
            ..., description="Responding supplier identifier",
        )
        supplier_name: str = Field(
            ..., description="Responding supplier display name",
        )
        answers: Dict[str, Any] = Field(
            default_factory=dict, description="Answers keyed by question_id",
        )
        evidence_files: List[str] = Field(
            default_factory=list, description="Evidence file references",
        )
        channel: str = Field(
            default="portal", description="Submission channel",
        )

    class ValidateResponseBody(BaseModel):
        """Request body for validating a response."""
        level: str = Field(
            default="completeness",
            description="Validation level (completeness, consistency, evidence, cross_field)",
        )

    class ScoreResponseBody(BaseModel):
        """Request body for scoring a response."""
        response_id: str = Field(
            ..., description="Response ID to score",
        )
        framework: Optional[str] = Field(
            None,
            description="Scoring framework override (uses template framework if omitted)",
        )

    class TriggerFollowUpBody(BaseModel):
        """Request body for triggering a follow-up action."""
        distribution_id: str = Field(
            ..., description="Distribution ID to follow up on",
        )
        action_type: str = Field(
            default="reminder",
            description="Follow-up type (reminder, escalation)",
        )
        message: str = Field(
            default="",
            description="Follow-up message content",
        )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    router = APIRouter(
        prefix="/api/v1/questionnaires",
        tags=["Supplier Questionnaires"],
    )
else:
    router = None  # type: ignore[assignment]


def _get_service(request: Request) -> Any:
    """Extract SupplierQuestionnaireService from app state.

    Args:
        request: FastAPI request object.

    Returns:
        SupplierQuestionnaireService instance.

    Raises:
        HTTPException: If service is not configured.
    """
    service = getattr(
        request.app.state, "supplier_questionnaire_service", None,
    )
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Supplier questionnaire service not configured",
        )
    return service


if FASTAPI_AVAILABLE:

    # ------------------------------------------------------------------
    # 1. Create template
    # ------------------------------------------------------------------
    @router.post("/v1/templates")
    async def create_template(
        body: CreateTemplateBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Create a new questionnaire template."""
        service = _get_service(request)
        try:
            template = service.create_template(
                name=body.name,
                framework=body.framework,
                version=body.version,
                description=body.description,
                sections=body.sections,
                language=body.language,
                tags=body.tags,
            )
            return template.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 2. List templates
    # ------------------------------------------------------------------
    @router.get("/v1/templates")
    async def list_templates(
        framework: Optional[str] = Query(None),
        status: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List questionnaire templates with optional filters."""
        service = _get_service(request)
        templates = service.list_templates(
            framework=framework,
            status=status,
            limit=limit,
            offset=offset,
        )
        return {
            "templates": [t.model_dump(mode="json") for t in templates],
            "count": len(templates),
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------
    # 3. Get template
    # ------------------------------------------------------------------
    @router.get("/v1/templates/{template_id}")
    async def get_template(
        template_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a questionnaire template by ID."""
        service = _get_service(request)
        template = service.get_template(template_id)
        if template is None:
            raise HTTPException(
                status_code=404,
                detail=f"Template {template_id} not found",
            )
        return template.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 4. Update template
    # ------------------------------------------------------------------
    @router.put("/v1/templates/{template_id}")
    async def update_template(
        template_id: str,
        body: UpdateTemplateBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Update an existing questionnaire template."""
        service = _get_service(request)
        try:
            template = service.update_template(
                template_id=template_id,
                name=body.name,
                description=body.description,
                sections=body.sections,
                status=body.status,
                tags=body.tags,
            )
            return template.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    # ------------------------------------------------------------------
    # 5. Clone template
    # ------------------------------------------------------------------
    @router.post("/v1/templates/{template_id}/clone")
    async def clone_template(
        template_id: str,
        body: CloneTemplateBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Clone an existing template to create a new version."""
        service = _get_service(request)
        try:
            template = service.clone_template(
                template_id=template_id,
                new_name=body.new_name,
                new_version=body.new_version,
            )
            return template.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    # ------------------------------------------------------------------
    # 6. Distribute questionnaire
    # ------------------------------------------------------------------
    @router.post("/v1/distribute")
    async def distribute(
        body: DistributeBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Distribute a questionnaire to a supplier."""
        service = _get_service(request)
        try:
            dist = service.distribute(
                template_id=body.template_id,
                supplier_id=body.supplier_id,
                supplier_name=body.supplier_name,
                supplier_email=body.supplier_email,
                campaign_id=body.campaign_id,
                channel=body.channel,
                deadline=body.deadline,
            )
            return dist.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 7. List distributions
    # ------------------------------------------------------------------
    @router.get("/v1/distributions")
    async def list_distributions(
        campaign_id: Optional[str] = Query(None),
        supplier_id: Optional[str] = Query(None),
        status: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List questionnaire distributions with optional filters."""
        service = _get_service(request)
        distributions = service.list_distributions(
            campaign_id=campaign_id,
            supplier_id=supplier_id,
            status=status,
            limit=limit,
            offset=offset,
        )
        return {
            "distributions": [
                d.model_dump(mode="json") for d in distributions
            ],
            "count": len(distributions),
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------
    # 8. Get distribution
    # ------------------------------------------------------------------
    @router.get("/v1/distributions/{dist_id}")
    async def get_distribution(
        dist_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a questionnaire distribution by ID."""
        service = _get_service(request)
        dist = service.get_distribution(dist_id)
        if dist is None:
            raise HTTPException(
                status_code=404,
                detail=f"Distribution {dist_id} not found",
            )
        return dist.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 9. Submit response
    # ------------------------------------------------------------------
    @router.post("/v1/responses")
    async def submit_response(
        body: SubmitResponseBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Submit a questionnaire response."""
        service = _get_service(request)
        try:
            response = service.submit_response(
                distribution_id=body.distribution_id,
                supplier_id=body.supplier_id,
                supplier_name=body.supplier_name,
                answers=body.answers,
                evidence_files=body.evidence_files,
                channel=body.channel,
            )
            return response.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 10. List responses
    # ------------------------------------------------------------------
    @router.get("/v1/responses")
    async def list_responses(
        supplier_id: Optional[str] = Query(None),
        template_id: Optional[str] = Query(None),
        status: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List questionnaire responses with optional filters."""
        service = _get_service(request)
        responses = service.list_responses(
            supplier_id=supplier_id,
            template_id=template_id,
            status=status,
            limit=limit,
            offset=offset,
        )
        return {
            "responses": [r.model_dump(mode="json") for r in responses],
            "count": len(responses),
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------
    # 11. Get response
    # ------------------------------------------------------------------
    @router.get("/v1/responses/{response_id}")
    async def get_response(
        response_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a questionnaire response by ID."""
        service = _get_service(request)
        response = service.get_response(response_id)
        if response is None:
            raise HTTPException(
                status_code=404,
                detail=f"Response {response_id} not found",
            )
        return response.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 12. Validate response
    # ------------------------------------------------------------------
    @router.post("/v1/responses/{response_id}/validate")
    async def validate_response(
        response_id: str,
        body: ValidateResponseBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Validate a questionnaire response."""
        service = _get_service(request)
        try:
            result = service.validate_response(
                response_id=response_id,
                level=body.level,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    # ------------------------------------------------------------------
    # 13. Score response
    # ------------------------------------------------------------------
    @router.post("/v1/score")
    async def score_response(
        body: ScoreResponseBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Score a questionnaire response."""
        service = _get_service(request)
        try:
            result = service.score_response(
                response_id=body.response_id,
                framework=body.framework,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 14. Get score
    # ------------------------------------------------------------------
    @router.get("/v1/scores/{score_id}")
    async def get_score(
        score_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a scoring result by ID."""
        service = _get_service(request)
        score = service.get_score(score_id)
        if score is None:
            raise HTTPException(
                status_code=404,
                detail=f"Score {score_id} not found",
            )
        return score.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 15. Get supplier scores
    # ------------------------------------------------------------------
    @router.get("/v1/scores/supplier/{supplier_id}")
    async def get_supplier_scores(
        supplier_id: str,
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """Get all scores for a specific supplier."""
        service = _get_service(request)
        scores = service.get_supplier_scores(
            supplier_id=supplier_id,
            limit=limit,
            offset=offset,
        )
        return {
            "scores": [s.model_dump(mode="json") for s in scores],
            "count": len(scores),
            "supplier_id": supplier_id,
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------
    # 16. Trigger follow-up
    # ------------------------------------------------------------------
    @router.post("/v1/followup")
    async def trigger_followup(
        body: TriggerFollowUpBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Trigger a follow-up action (reminder or escalation)."""
        service = _get_service(request)
        try:
            if body.action_type == "escalation":
                action = service.escalate(
                    distribution_id=body.distribution_id,
                    message=body.message,
                )
            else:
                action = service.trigger_reminder(
                    distribution_id=body.distribution_id,
                    message=body.message,
                )
            return action.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 17. Get follow-up status
    # ------------------------------------------------------------------
    @router.get("/v1/followup/{campaign_id}")
    async def get_followup_status(
        campaign_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get follow-up status for a campaign."""
        service = _get_service(request)
        reminders = service.get_due_reminders(campaign_id)
        return {
            "campaign_id": campaign_id,
            "pending_reminders": [
                r.model_dump(mode="json") for r in reminders
            ],
            "count": len(reminders),
        }

    # ------------------------------------------------------------------
    # 18. Get analytics
    # ------------------------------------------------------------------
    @router.get("/v1/analytics/{campaign_id}")
    async def get_analytics(
        campaign_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get campaign analytics and compliance gaps."""
        service = _get_service(request)
        analytics = service.get_campaign_analytics(campaign_id)
        return analytics.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 19. Health check
    # ------------------------------------------------------------------
    @router.get("/health")
    async def health(
        request: Request,
    ) -> Dict[str, Any]:
        """Supplier questionnaire service health check endpoint."""
        service = _get_service(request)
        return service.health_check()

    # ------------------------------------------------------------------
    # 20. Statistics
    # ------------------------------------------------------------------
    @router.get("/v1/statistics")
    async def get_statistics(
        request: Request,
    ) -> Dict[str, Any]:
        """Get supplier questionnaire service statistics."""
        service = _get_service(request)
        stats = service.get_statistics()
        return stats.model_dump(mode="json")


__all__ = [
    "router",
]
