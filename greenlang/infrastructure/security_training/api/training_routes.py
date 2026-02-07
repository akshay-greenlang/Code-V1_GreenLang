# -*- coding: utf-8 -*-
"""
Security Training REST API Router - SEC-010

FastAPI APIRouter providing REST endpoints for the GreenLang security training
platform. Supports training management, assessments, phishing campaigns,
and security scoring.

All endpoints use dependency injection for services and include proper error
handling with standard HTTP status codes.

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.infrastructure.security_training.api.training_routes import (
    ...     training_router,
    ... )
    >>> app = FastAPI()
    >>> app.include_router(training_router)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, Depends, HTTPException, Query, status
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = object  # type: ignore[misc, assignment]
    Depends = None  # type: ignore[assignment]
    HTTPException = Exception  # type: ignore[misc, assignment]
    Query = None  # type: ignore[assignment]
    status = None  # type: ignore[assignment]
    JSONResponse = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[misc, assignment]
    Field = None  # type: ignore[assignment]


from greenlang.infrastructure.security_training.models import (
    CampaignStatus,
    TemplateType,
)
from greenlang.infrastructure.security_training.content_library import ContentLibrary
from greenlang.infrastructure.security_training.curriculum_mapper import CurriculumMapper
from greenlang.infrastructure.security_training.assessment_engine import AssessmentEngine
from greenlang.infrastructure.security_training.phishing_simulator import PhishingSimulator
from greenlang.infrastructure.security_training.completion_tracker import CompletionTracker
from greenlang.infrastructure.security_training.security_scorer import SecurityScorer
from greenlang.infrastructure.security_training.metrics import (
    record_training_completion,
    record_training_attempt,
    record_campaign_status,
    record_phishing_emails_sent,
    update_phishing_metrics,
    record_certificate_issued,
)


# ---------------------------------------------------------------------------
# Service Singletons
# ---------------------------------------------------------------------------

_library: Optional[ContentLibrary] = None
_mapper: Optional[CurriculumMapper] = None
_engine: Optional[AssessmentEngine] = None
_simulator: Optional[PhishingSimulator] = None
_tracker: Optional[CompletionTracker] = None
_scorer: Optional[SecurityScorer] = None


def _get_library() -> ContentLibrary:
    """Get or create ContentLibrary singleton."""
    global _library
    if _library is None:
        _library = ContentLibrary()
    return _library


def _get_mapper() -> CurriculumMapper:
    """Get or create CurriculumMapper singleton."""
    global _mapper
    if _mapper is None:
        _mapper = CurriculumMapper(_get_library())
    return _mapper


def _get_engine() -> AssessmentEngine:
    """Get or create AssessmentEngine singleton."""
    global _engine
    if _engine is None:
        _engine = AssessmentEngine(_get_library())
    return _engine


def _get_simulator() -> PhishingSimulator:
    """Get or create PhishingSimulator singleton."""
    global _simulator
    if _simulator is None:
        _simulator = PhishingSimulator()
    return _simulator


def _get_tracker() -> CompletionTracker:
    """Get or create CompletionTracker singleton."""
    global _tracker
    if _tracker is None:
        _tracker = CompletionTracker(_get_library())
    return _tracker


def _get_scorer() -> SecurityScorer:
    """Get or create SecurityScorer singleton."""
    global _scorer
    if _scorer is None:
        _scorer = SecurityScorer()
    return _scorer


# ---------------------------------------------------------------------------
# Request/Response Schemas
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class CourseResponse(BaseModel):
        """Course information response."""

        id: str
        title: str
        description: str
        duration_minutes: int
        content_type: str
        role_required: Optional[str]
        passing_score: int
        is_mandatory: bool
        tags: List[str]
        prerequisites: List[str]

    class CourseListResponse(BaseModel):
        """Paginated course list response."""

        items: List[CourseResponse]
        total: int
        page: int
        page_size: int

    class CourseContentResponse(BaseModel):
        """Full course content response."""

        course_id: str
        modules: List[Dict[str, Any]]
        version: int

    class UserProgressResponse(BaseModel):
        """User training progress response."""

        user_id: str
        total_required: int
        total_completed: int
        total_in_progress: int
        total_overdue: int
        completion_rate: float
        average_score: Optional[float]
        certificates: List[str]
        expiring_soon: List[str]

    class CurriculumItemResponse(BaseModel):
        """Curriculum item response."""

        course_id: str
        course_title: str
        is_required: bool
        is_completed: bool
        is_overdue: bool
        due_date: Optional[str]

    class StartCourseResponse(BaseModel):
        """Start course response."""

        completion_id: str
        course_id: str
        started_at: str

    class AssessmentSubmission(BaseModel):
        """Assessment submission request."""

        answers: Dict[str, int] = Field(
            ...,
            description="Map of question_id to selected option index",
        )
        time_taken_seconds: int = Field(
            default=0,
            ge=0,
            description="Time taken to complete the quiz",
        )

    class AssessmentResultResponse(BaseModel):
        """Assessment result response."""

        score: int
        passed: bool
        total_questions: int
        correct_answers: int
        attempt_number: int
        certificate_id: Optional[str]
        feedback: Dict[str, str]

    class CertificateResponse(BaseModel):
        """Certificate response."""

        id: str
        course_id: str
        course_title: str
        issued_at: str
        expires_at: str
        verification_code: str
        score: int

    class CertificateVerifyResponse(BaseModel):
        """Certificate verification response."""

        valid: bool
        certificate: Optional[CertificateResponse]

    class TeamComplianceResponse(BaseModel):
        """Team compliance response."""

        team_id: str
        team_name: str
        total_members: int
        compliant_members: int
        overdue_members: int
        compliance_rate: float
        average_score: float

    class CreateCampaignRequest(BaseModel):
        """Create phishing campaign request."""

        name: str = Field(..., min_length=1, max_length=256)
        template_type: str = Field(...)
        target_user_ids: List[str] = Field(default_factory=list)
        target_roles: List[str] = Field(default_factory=list)
        scheduled_at: Optional[str] = None

    class CampaignResponse(BaseModel):
        """Phishing campaign response."""

        id: str
        name: str
        template_type: str
        status: str
        created_at: str
        sent_at: Optional[str]
        target_count: int

    class CampaignMetricsResponse(BaseModel):
        """Campaign metrics response."""

        campaign_id: str
        total_sent: int
        total_opened: int
        total_clicked: int
        total_credentials: int
        total_reported: int
        open_rate: float
        click_rate: float
        credential_rate: float
        report_rate: float

    class SecurityScoreResponse(BaseModel):
        """Security score response."""

        user_id: str
        score: int
        components: Dict[str, float]
        calculated_at: str
        trend: str
        suggestions: List[str]

    class LeaderboardEntryResponse(BaseModel):
        """Leaderboard entry response."""

        rank: int
        user_id: str
        score: int
        trend: str

    class LeaderboardResponse(BaseModel):
        """Leaderboard response."""

        team_id: Optional[str]
        entries: List[LeaderboardEntryResponse]
        organization_average: float


# ---------------------------------------------------------------------------
# Router Definition
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    training_router = APIRouter(
        prefix="/api/v1/secops",
        tags=["Security Training"],
        responses={
            400: {"description": "Bad Request"},
            404: {"description": "Not Found"},
            422: {"description": "Validation Error"},
            500: {"description": "Internal Server Error"},
        },
    )

    # -----------------------------------------------------------------------
    # Training Endpoints
    # -----------------------------------------------------------------------

    @training_router.get(
        "/training/courses",
        response_model=CourseListResponse,
        summary="List training courses",
        description="Retrieve a list of available training courses.",
        operation_id="list_courses",
    )
    async def list_courses(
        role: Optional[str] = Query(None, description="Filter by role"),
        tag: Optional[str] = Query(None, description="Filter by tag"),
        mandatory_only: bool = Query(False, description="Only mandatory courses"),
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
        library: ContentLibrary = Depends(_get_library),
    ) -> CourseListResponse:
        """List available training courses with optional filters."""
        courses = await library.list_courses(
            role_filter=role,
            tag_filter=tag,
            mandatory_only=mandatory_only,
        )

        # Paginate
        start = (page - 1) * page_size
        end = start + page_size
        page_courses = courses[start:end]

        items = [
            CourseResponse(
                id=c.id,
                title=c.title,
                description=c.description,
                duration_minutes=c.duration_minutes,
                content_type=c.content_type.value,
                role_required=c.role_required,
                passing_score=c.passing_score,
                is_mandatory=c.is_mandatory,
                tags=c.tags,
                prerequisites=c.prerequisites,
            )
            for c in page_courses
        ]

        return CourseListResponse(
            items=items,
            total=len(courses),
            page=page,
            page_size=page_size,
        )

    @training_router.get(
        "/training/courses/{course_id}",
        response_model=Dict[str, Any],
        summary="Get course content",
        description="Retrieve full content for a specific course.",
        operation_id="get_course_content",
    )
    async def get_course_content(
        course_id: str,
        library: ContentLibrary = Depends(_get_library),
    ) -> Dict[str, Any]:
        """Get full course content including modules."""
        course = await library.get_course(course_id)
        if course is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Course '{course_id}' not found",
            )

        content = await library.get_course_content(course_id)

        return {
            "course": {
                "id": course.id,
                "title": course.title,
                "description": course.description,
                "duration_minutes": course.duration_minutes,
                "passing_score": course.passing_score,
            },
            "modules": [
                {
                    "id": m.id,
                    "title": m.title,
                    "content_html": m.content_html,
                    "video_url": m.video_url,
                    "duration_minutes": m.duration_minutes,
                    "order": m.order,
                }
                for m in (content.modules if content else [])
            ],
            "question_count": len(content.questions) if content else 0,
            "version": content.version if content else 1,
        }

    @training_router.get(
        "/training/my-progress",
        response_model=UserProgressResponse,
        summary="Get user progress",
        description="Get current user's training progress.",
        operation_id="get_user_progress",
    )
    async def get_user_progress(
        user_id: str = Query(..., description="User ID"),
        tracker: CompletionTracker = Depends(_get_tracker),
    ) -> UserProgressResponse:
        """Get user's overall training progress."""
        progress = await tracker.get_user_progress(user_id)

        return UserProgressResponse(
            user_id=progress.user_id,
            total_required=progress.total_required,
            total_completed=progress.total_completed,
            total_in_progress=progress.total_in_progress,
            total_overdue=progress.total_overdue,
            completion_rate=progress.completion_rate,
            average_score=progress.average_score,
            certificates=progress.certificates,
            expiring_soon=progress.expiring_soon,
        )

    @training_router.get(
        "/training/my-curriculum",
        response_model=List[CurriculumItemResponse],
        summary="Get user curriculum",
        description="Get required training curriculum for the user.",
        operation_id="get_user_curriculum",
    )
    async def get_user_curriculum(
        user_id: str = Query(..., description="User ID"),
        roles: str = Query("", description="Comma-separated roles"),
        mapper: CurriculumMapper = Depends(_get_mapper),
    ) -> List[CurriculumItemResponse]:
        """Get user's required training curriculum."""

        class MockUser:
            def __init__(self, uid: str, r: List[str]) -> None:
                self._id = uid
                self._roles = r

            @property
            def id(self) -> str:
                return self._id

            @property
            def roles(self) -> List[str]:
                return self._roles

        role_list = [r.strip() for r in roles.split(",") if r.strip()]
        user = MockUser(user_id, role_list)
        curriculum = await mapper.get_curriculum(user)

        return [
            CurriculumItemResponse(
                course_id=item.course.id,
                course_title=item.course.title,
                is_required=item.is_required,
                is_completed=item.is_completed,
                is_overdue=item.is_overdue,
                due_date=item.due_date.isoformat() if item.due_date else None,
            )
            for item in curriculum.items
        ]

    @training_router.post(
        "/training/courses/{course_id}/start",
        response_model=StartCourseResponse,
        status_code=201,
        summary="Start course",
        description="Record that a user has started a course.",
        operation_id="start_course",
    )
    async def start_course(
        course_id: str,
        user_id: str = Query(..., description="User ID"),
        tracker: CompletionTracker = Depends(_get_tracker),
        library: ContentLibrary = Depends(_get_library),
    ) -> StartCourseResponse:
        """Start a training course."""
        course = await library.get_course(course_id)
        if course is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Course '{course_id}' not found",
            )

        completion = await tracker.start_course(user_id, course_id)

        return StartCourseResponse(
            completion_id=completion.id,
            course_id=completion.course_id,
            started_at=completion.started_at.isoformat(),
        )

    @training_router.post(
        "/training/courses/{course_id}/complete",
        response_model=Dict[str, Any],
        summary="Complete course",
        description="Mark a course as complete (for content-only courses).",
        operation_id="complete_course",
    )
    async def complete_course(
        course_id: str,
        user_id: str = Query(..., description="User ID"),
        tracker: CompletionTracker = Depends(_get_tracker),
        library: ContentLibrary = Depends(_get_library),
    ) -> Dict[str, Any]:
        """Mark course as complete (content-only, no assessment)."""
        course = await library.get_course(course_id)
        if course is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Course '{course_id}' not found",
            )

        completion = await tracker.record_completion(
            user_id=user_id,
            course_id=course_id,
            score=100,
            passed=True,
        )

        return {
            "completion_id": completion.id,
            "course_id": course_id,
            "completed_at": completion.completed_at.isoformat() if completion.completed_at else None,
            "passed": True,
        }

    @training_router.post(
        "/training/courses/{course_id}/assessment",
        response_model=AssessmentResultResponse,
        summary="Submit assessment",
        description="Submit quiz answers for grading.",
        operation_id="submit_assessment",
    )
    async def submit_assessment(
        course_id: str,
        submission: AssessmentSubmission,
        user_id: str = Query(..., description="User ID"),
        engine: AssessmentEngine = Depends(_get_engine),
        library: ContentLibrary = Depends(_get_library),
    ) -> AssessmentResultResponse:
        """Submit assessment for grading."""
        course = await library.get_course(course_id)
        if course is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Course '{course_id}' not found",
            )

        from greenlang.infrastructure.security_training.models import QuizSubmission

        quiz_submission = QuizSubmission(
            user_id=user_id,
            course_id=course_id,
            answers=submission.answers,
            time_taken_seconds=submission.time_taken_seconds,
        )

        record_training_attempt(course_id)
        result = await engine.grade_assessment(quiz_submission)
        record_training_completion(course_id, result.passed, result.score)

        if result.certificate:
            record_certificate_issued(course_id)

        return AssessmentResultResponse(
            score=result.score,
            passed=result.passed,
            total_questions=result.total_questions,
            correct_answers=result.correct_answers,
            attempt_number=result.attempt_number,
            certificate_id=result.certificate.id if result.certificate else None,
            feedback=result.feedback,
        )

    @training_router.get(
        "/training/certificates",
        response_model=List[CertificateResponse],
        summary="List user certificates",
        description="Get all certificates for a user.",
        operation_id="list_certificates",
    )
    async def list_certificates(
        user_id: str = Query(..., description="User ID"),
        engine: AssessmentEngine = Depends(_get_engine),
    ) -> List[CertificateResponse]:
        """List user's certificates."""
        # In production, query from database
        certificates = []
        for code, cert in engine._certificates.items():
            if cert.user_id == user_id:
                certificates.append(
                    CertificateResponse(
                        id=cert.id,
                        course_id=cert.course_id,
                        course_title=cert.course_title,
                        issued_at=cert.issued_at.isoformat(),
                        expires_at=cert.expires_at.isoformat(),
                        verification_code=cert.verification_code,
                        score=cert.score,
                    )
                )
        return certificates

    @training_router.get(
        "/training/certificates/{code}/verify",
        response_model=CertificateVerifyResponse,
        summary="Verify certificate",
        description="Verify a certificate by its verification code.",
        operation_id="verify_certificate",
    )
    async def verify_certificate(
        code: str,
        engine: AssessmentEngine = Depends(_get_engine),
    ) -> CertificateVerifyResponse:
        """Verify certificate authenticity."""
        cert = await engine.verify_certificate(code)

        if cert is None:
            return CertificateVerifyResponse(valid=False, certificate=None)

        return CertificateVerifyResponse(
            valid=True,
            certificate=CertificateResponse(
                id=cert.id,
                course_id=cert.course_id,
                course_title=cert.course_title,
                issued_at=cert.issued_at.isoformat(),
                expires_at=cert.expires_at.isoformat(),
                verification_code=cert.verification_code,
                score=cert.score,
            ),
        )

    @training_router.get(
        "/training/team-compliance",
        response_model=TeamComplianceResponse,
        summary="Get team compliance",
        description="Get training compliance statistics for a team.",
        operation_id="get_team_compliance",
    )
    async def get_team_compliance(
        team_id: str = Query(..., description="Team ID"),
        tracker: CompletionTracker = Depends(_get_tracker),
    ) -> TeamComplianceResponse:
        """Get team compliance statistics."""
        compliance = await tracker.get_team_compliance(team_id)

        return TeamComplianceResponse(
            team_id=compliance.team_id,
            team_name=compliance.team_name,
            total_members=compliance.total_members,
            compliant_members=compliance.compliant_members,
            overdue_members=compliance.overdue_members,
            compliance_rate=compliance.compliance_rate,
            average_score=compliance.average_score,
        )

    # -----------------------------------------------------------------------
    # Phishing Campaign Endpoints
    # -----------------------------------------------------------------------

    @training_router.post(
        "/phishing/campaigns",
        response_model=CampaignResponse,
        status_code=201,
        summary="Create campaign",
        description="Create a new phishing simulation campaign.",
        operation_id="create_phishing_campaign",
    )
    async def create_phishing_campaign(
        request: CreateCampaignRequest,
        created_by: str = Query(..., description="Creator user ID"),
        simulator: PhishingSimulator = Depends(_get_simulator),
    ) -> CampaignResponse:
        """Create a new phishing campaign."""
        try:
            template_type = TemplateType(request.template_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid template_type '{request.template_type}'",
            )

        scheduled_at = None
        if request.scheduled_at:
            scheduled_at = datetime.fromisoformat(request.scheduled_at)

        campaign = await simulator.create_campaign(
            name=request.name,
            template_type=template_type,
            target_users=request.target_user_ids,
            target_roles=request.target_roles,
            scheduled_at=scheduled_at,
            created_by=created_by,
        )

        record_campaign_status("draft")

        return CampaignResponse(
            id=campaign.id,
            name=campaign.name,
            template_type=campaign.template_type.value,
            status=campaign.status.value,
            created_at=campaign.created_at.isoformat(),
            sent_at=campaign.sent_at.isoformat() if campaign.sent_at else None,
            target_count=campaign.target_count,
        )

    @training_router.get(
        "/phishing/campaigns",
        response_model=List[CampaignResponse],
        summary="List campaigns",
        description="List all phishing campaigns.",
        operation_id="list_phishing_campaigns",
    )
    async def list_phishing_campaigns(
        status_filter: Optional[str] = Query(None, alias="status"),
        limit: int = Query(50, ge=1, le=200),
        simulator: PhishingSimulator = Depends(_get_simulator),
    ) -> List[CampaignResponse]:
        """List phishing campaigns."""
        campaign_status = None
        if status_filter:
            try:
                campaign_status = CampaignStatus(status_filter)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid status '{status_filter}'",
                )

        campaigns = await simulator.list_campaigns(
            status_filter=campaign_status,
            limit=limit,
        )

        return [
            CampaignResponse(
                id=c.id,
                name=c.name,
                template_type=c.template_type.value,
                status=c.status.value,
                created_at=c.created_at.isoformat(),
                sent_at=c.sent_at.isoformat() if c.sent_at else None,
                target_count=c.target_count,
            )
            for c in campaigns
        ]

    @training_router.get(
        "/phishing/campaigns/{campaign_id}",
        response_model=Dict[str, Any],
        summary="Get campaign",
        description="Get details of a specific campaign.",
        operation_id="get_phishing_campaign",
    )
    async def get_phishing_campaign(
        campaign_id: str,
        simulator: PhishingSimulator = Depends(_get_simulator),
    ) -> Dict[str, Any]:
        """Get campaign details."""
        campaign = await simulator.get_campaign(campaign_id)
        if campaign is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Campaign '{campaign_id}' not found",
            )

        metrics = await simulator.get_campaign_metrics(campaign_id)

        return {
            "campaign": {
                "id": campaign.id,
                "name": campaign.name,
                "template_type": campaign.template_type.value,
                "status": campaign.status.value,
                "created_at": campaign.created_at.isoformat(),
                "sent_at": campaign.sent_at.isoformat() if campaign.sent_at else None,
                "completed_at": campaign.completed_at.isoformat() if campaign.completed_at else None,
                "target_count": campaign.target_count,
            },
            "metrics": {
                "total_sent": metrics.total_sent,
                "total_opened": metrics.total_opened,
                "total_clicked": metrics.total_clicked,
                "total_reported": metrics.total_reported,
                "click_rate": metrics.click_rate,
                "report_rate": metrics.report_rate,
            },
        }

    @training_router.put(
        "/phishing/campaigns/{campaign_id}",
        response_model=CampaignResponse,
        summary="Update campaign",
        description="Update a draft campaign.",
        operation_id="update_phishing_campaign",
    )
    async def update_phishing_campaign(
        campaign_id: str,
        request: Dict[str, Any],
        simulator: PhishingSimulator = Depends(_get_simulator),
    ) -> CampaignResponse:
        """Update a campaign."""
        campaign = await simulator.update_campaign(campaign_id, request)
        if campaign is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Campaign '{campaign_id}' not found",
            )

        return CampaignResponse(
            id=campaign.id,
            name=campaign.name,
            template_type=campaign.template_type.value,
            status=campaign.status.value,
            created_at=campaign.created_at.isoformat(),
            sent_at=campaign.sent_at.isoformat() if campaign.sent_at else None,
            target_count=campaign.target_count,
        )

    @training_router.post(
        "/phishing/campaigns/{campaign_id}/send",
        response_model=Dict[str, Any],
        summary="Send campaign emails",
        description="Send phishing simulation emails for a campaign.",
        operation_id="send_phishing_campaign",
    )
    async def send_phishing_campaign(
        campaign_id: str,
        simulator: PhishingSimulator = Depends(_get_simulator),
    ) -> Dict[str, Any]:
        """Send phishing emails for a campaign."""
        try:
            count = await simulator.send_phishing_emails(campaign_id)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )

        record_campaign_status("running")
        record_phishing_emails_sent(campaign_id, count)

        return {
            "campaign_id": campaign_id,
            "emails_sent": count,
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }

    @training_router.get(
        "/phishing/campaigns/{campaign_id}/metrics",
        response_model=CampaignMetricsResponse,
        summary="Get campaign metrics",
        description="Get detailed metrics for a campaign.",
        operation_id="get_campaign_metrics",
    )
    async def get_campaign_metrics(
        campaign_id: str,
        simulator: PhishingSimulator = Depends(_get_simulator),
    ) -> CampaignMetricsResponse:
        """Get campaign metrics."""
        try:
            metrics = await simulator.get_campaign_metrics(campaign_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Campaign '{campaign_id}' not found",
            )

        update_phishing_metrics(
            campaign_id,
            metrics.click_rate,
            metrics.report_rate,
            metrics.credential_rate,
        )

        return CampaignMetricsResponse(
            campaign_id=metrics.campaign_id,
            total_sent=metrics.total_sent,
            total_opened=metrics.total_opened,
            total_clicked=metrics.total_clicked,
            total_credentials=metrics.total_credentials,
            total_reported=metrics.total_reported,
            open_rate=metrics.open_rate,
            click_rate=metrics.click_rate,
            credential_rate=metrics.credential_rate,
            report_rate=metrics.report_rate,
        )

    @training_router.post(
        "/phishing/track/{campaign_id}/{user_id}/open",
        response_model=Dict[str, bool],
        summary="Track email open",
        description="Track email open via tracking pixel (internal).",
        operation_id="track_phishing_open",
    )
    async def track_phishing_open(
        campaign_id: str,
        user_id: str,
        simulator: PhishingSimulator = Depends(_get_simulator),
    ) -> Dict[str, bool]:
        """Track email open event."""
        success = await simulator.track_open(campaign_id, user_id)
        return {"tracked": success}

    @training_router.post(
        "/phishing/track/{campaign_id}/{user_id}/click",
        response_model=Dict[str, bool],
        summary="Track link click",
        description="Track link click (internal).",
        operation_id="track_phishing_click",
    )
    async def track_phishing_click(
        campaign_id: str,
        user_id: str,
        user_agent: Optional[str] = Query(None),
        ip_address: Optional[str] = Query(None),
        simulator: PhishingSimulator = Depends(_get_simulator),
    ) -> Dict[str, bool]:
        """Track link click event."""
        success = await simulator.track_click(
            campaign_id, user_id, user_agent, ip_address
        )
        return {"tracked": success}

    # -----------------------------------------------------------------------
    # Security Score Endpoints
    # -----------------------------------------------------------------------

    @training_router.get(
        "/security-score",
        response_model=SecurityScoreResponse,
        summary="Get security score",
        description="Get security score for a user.",
        operation_id="get_security_score",
    )
    async def get_security_score(
        user_id: str = Query(..., description="User ID"),
        scorer: SecurityScorer = Depends(_get_scorer),
    ) -> SecurityScoreResponse:
        """Get user's security score."""
        breakdown = await scorer.get_score_breakdown(user_id)

        return SecurityScoreResponse(
            user_id=breakdown.user_id,
            score=breakdown.total_score,
            components={c.name: c.raw_score for c in breakdown.components},
            calculated_at=breakdown.calculated_at.isoformat(),
            trend=breakdown.trend,
            suggestions=breakdown.suggestions,
        )

    @training_router.get(
        "/security-score/leaderboard",
        response_model=LeaderboardResponse,
        summary="Get leaderboard",
        description="Get security score leaderboard.",
        operation_id="get_security_leaderboard",
    )
    async def get_security_leaderboard(
        team_id: Optional[str] = Query(None, description="Filter by team"),
        limit: int = Query(10, ge=1, le=100),
        scorer: SecurityScorer = Depends(_get_scorer),
    ) -> LeaderboardResponse:
        """Get security score leaderboard."""
        entries = await scorer.get_leaderboard(team_id=team_id, limit=limit)
        org_average = await scorer.get_organization_average()

        return LeaderboardResponse(
            team_id=team_id,
            entries=[
                LeaderboardEntryResponse(
                    rank=e.rank,
                    user_id=e.user_id,
                    score=e.score,
                    trend=e.trend,
                )
                for e in entries
            ],
            organization_average=org_average,
        )

    # Apply authentication/authorization protection
    try:
        from greenlang.infrastructure.auth_service.route_protector import (
            protect_router,
        )
        protect_router(training_router)
    except ImportError:
        pass  # auth_service not available

else:
    training_router = None  # type: ignore[assignment]
    logger.warning("FastAPI not available - training_router is None")


__all__ = ["training_router"]
