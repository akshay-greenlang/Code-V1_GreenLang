# -*- coding: utf-8 -*-
"""
Security Training Data Models - SEC-010

Pydantic v2 models for the GreenLang security training platform. Provides
strongly-typed data structures for courses, assessments, phishing campaigns,
training completions, certificates, and security scores.

All datetime fields use UTC. All models enforce strict validation via Pydantic v2
field validators and model configuration.

Models:
    - TemplateType: Enum of phishing template types
    - CampaignStatus: Lifecycle status of a phishing campaign
    - Course: Training course definition
    - Module: Course module with content
    - Question: Assessment question with options
    - CourseContent: Full course content including modules and questions
    - TrainingCompletion: User's completion record for a course
    - Certificate: Completion certificate with verification code
    - PhishingCampaign: Phishing simulation campaign definition
    - PhishingResult: Individual phishing test result
    - CampaignMetrics: Aggregated campaign metrics
    - SecurityScore: Employee security posture score
    - UserProgress: Aggregated user training progress
    - QuizSubmission: Quiz answer submission

Example:
    >>> from greenlang.infrastructure.security_training.models import (
    ...     Course, TrainingCompletion, PhishingCampaign
    ... )
    >>> course = Course(
    ...     id="security_awareness",
    ...     title="Security Awareness Fundamentals",
    ...     description="Annual security awareness training",
    ...     duration_minutes=45,
    ...     content_type="interactive",
    ...     role_required=None,
    ...     passing_score=80,
    ... )
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TemplateType(str, Enum):
    """Supported phishing email template types.

    Each type simulates a different social engineering attack vector.
    """

    CREDENTIAL_HARVEST = "credential_harvest"
    """Fake login page requesting username/password."""

    MALICIOUS_ATTACHMENT = "malicious_attachment"
    """Email with suspicious attachment (simulated)."""

    FAKE_INVOICE = "fake_invoice"
    """Fake invoice or payment request."""

    URGENT_ACTION = "urgent_action"
    """Urgent action required - account suspension, etc."""

    CEO_FRAUD = "ceo_fraud"
    """Business email compromise - impersonating executive."""

    IT_SUPPORT = "it_support"
    """Fake IT support request for credentials."""


class CampaignStatus(str, Enum):
    """Lifecycle status of a phishing campaign.

    Transitions follow: DRAFT -> SCHEDULED -> RUNNING -> COMPLETED
    Campaigns can be CANCELLED from DRAFT or SCHEDULED states.
    """

    DRAFT = "draft"
    """Campaign is being configured, not yet ready to send."""

    SCHEDULED = "scheduled"
    """Campaign is scheduled to send at a future time."""

    RUNNING = "running"
    """Campaign emails have been sent, collecting results."""

    COMPLETED = "completed"
    """Campaign has ended, all results collected."""

    CANCELLED = "cancelled"
    """Campaign was cancelled before completion."""


class ContentType(str, Enum):
    """Type of training content delivery."""

    VIDEO = "video"
    """Video-based training module."""

    INTERACTIVE = "interactive"
    """Interactive web-based training."""

    DOCUMENT = "document"
    """Document/PDF-based training."""

    QUIZ_ONLY = "quiz_only"
    """Assessment-only (no content, just quiz)."""


# ---------------------------------------------------------------------------
# Course Models
# ---------------------------------------------------------------------------


class Course(BaseModel):
    """Training course definition.

    Represents a single training course in the catalog with its metadata,
    requirements, and scoring configuration.

    Attributes:
        id: Unique course identifier (lowercase alphanumeric with underscores).
        title: Human-readable course title.
        description: Detailed description of course content and objectives.
        duration_minutes: Estimated course duration in minutes.
        content_type: Type of content delivery (video, interactive, etc.).
        role_required: Role that requires this course (None = all employees).
        passing_score: Minimum score to pass the assessment (0-100).
        is_mandatory: Whether this course is mandatory for the required role.
        tags: Free-form tags for categorization.
        prerequisites: List of course IDs that must be completed first.
        created_at: When the course was created.
        updated_at: When the course was last updated.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
    )

    id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Unique course identifier.",
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Human-readable course title.",
    )
    description: str = Field(
        default="",
        max_length=4096,
        description="Detailed course description.",
    )
    duration_minutes: int = Field(
        ...,
        ge=5,
        le=480,
        description="Estimated course duration in minutes.",
    )
    content_type: ContentType = Field(
        default=ContentType.INTERACTIVE,
        description="Type of content delivery.",
    )
    role_required: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Role that requires this course (None = all employees).",
    )
    passing_score: int = Field(
        default=80,
        ge=0,
        le=100,
        description="Minimum passing score (0-100).",
    )
    is_mandatory: bool = Field(
        default=True,
        description="Whether this course is mandatory.",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorization.",
    )
    prerequisites: List[str] = Field(
        default_factory=list,
        description="Course IDs that must be completed first.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Course creation timestamp (UTC).",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp (UTC).",
    )

    @field_validator("id")
    @classmethod
    def validate_course_id(cls, v: str) -> str:
        """Validate course ID format."""
        v_lower = v.strip().lower()
        if not re.match(r"^[a-z][a-z0-9_]{0,63}$", v_lower):
            raise ValueError(
                f"Course ID '{v}' must start with a letter and contain "
                f"only lowercase letters, numbers, and underscores."
            )
        return v_lower

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, v: List[str]) -> List[str]:
        """Normalize tags to lowercase and deduplicate."""
        seen: set[str] = set()
        result: List[str] = []
        for tag in v:
            tag_lower = tag.strip().lower()
            if tag_lower and tag_lower not in seen:
                seen.add(tag_lower)
                result.append(tag_lower)
        return result


class Module(BaseModel):
    """Course module with content.

    A module is a single unit of learning within a course.

    Attributes:
        id: Unique module identifier.
        title: Module title.
        content_html: HTML content for the module.
        video_url: Optional video URL for video content.
        duration_minutes: Estimated module duration.
        order: Display order within the course.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique module identifier.",
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Module title.",
    )
    content_html: str = Field(
        default="",
        max_length=65536,
        description="HTML content for the module.",
    )
    video_url: Optional[str] = Field(
        default=None,
        max_length=2048,
        description="Video URL for video-based modules.",
    )
    duration_minutes: int = Field(
        default=10,
        ge=1,
        le=120,
        description="Estimated module duration in minutes.",
    )
    order: int = Field(
        default=0,
        ge=0,
        description="Display order within the course.",
    )


class Question(BaseModel):
    """Assessment question with multiple choice options.

    Attributes:
        id: Unique question identifier.
        text: Question text.
        options: List of answer options (typically 4).
        correct_option: Index of the correct option (0-based).
        explanation: Explanation shown after answering.
        difficulty: Question difficulty level (1-5).
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique question identifier.",
    )
    text: str = Field(
        ...,
        min_length=10,
        max_length=2048,
        description="Question text.",
    )
    options: List[str] = Field(
        ...,
        min_length=2,
        max_length=6,
        description="Answer options.",
    )
    correct_option: int = Field(
        ...,
        ge=0,
        description="Index of correct option (0-based).",
    )
    explanation: str = Field(
        default="",
        max_length=2048,
        description="Explanation shown after answering.",
    )
    difficulty: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Difficulty level (1=easy, 5=hard).",
    )

    @model_validator(mode="after")
    def validate_correct_option(self) -> "Question":
        """Ensure correct_option is within valid range."""
        if self.correct_option >= len(self.options):
            raise ValueError(
                f"correct_option ({self.correct_option}) must be less than "
                f"number of options ({len(self.options)})."
            )
        return self


class CourseContent(BaseModel):
    """Full course content including modules and assessment questions.

    Attributes:
        id: Course identifier.
        course_id: Reference to the parent course.
        modules: List of course modules.
        questions: Question pool for assessment.
        version: Content version for updates.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Content identifier.",
    )
    course_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Parent course identifier.",
    )
    modules: List[Module] = Field(
        default_factory=list,
        description="Course modules.",
    )
    questions: List[Question] = Field(
        default_factory=list,
        description="Assessment question pool.",
    )
    version: int = Field(
        default=1,
        ge=1,
        description="Content version number.",
    )


# ---------------------------------------------------------------------------
# Completion & Certificate Models
# ---------------------------------------------------------------------------


class TrainingCompletion(BaseModel):
    """User's completion record for a course.

    Tracks the user's progress through a course including start time,
    completion time, score, and pass/fail status.

    Attributes:
        id: Unique completion record identifier.
        user_id: User who completed the training.
        course_id: Course that was completed.
        started_at: When the user started the course.
        completed_at: When the user completed the course (None if in progress).
        score: Assessment score (0-100, None if not yet assessed).
        passed: Whether the user passed (None if not yet assessed).
        certificate_id: ID of issued certificate (None if not passed).
        attempts: Number of quiz attempts.
        time_spent_minutes: Total time spent on the course.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique completion record identifier.",
    )
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="User identifier.",
    )
    course_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Course identifier.",
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the user started the course.",
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="When the user completed the course.",
    )
    score: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Assessment score (0-100).",
    )
    passed: Optional[bool] = Field(
        default=None,
        description="Whether the user passed.",
    )
    certificate_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Certificate ID if passed.",
    )
    attempts: int = Field(
        default=0,
        ge=0,
        description="Number of quiz attempts.",
    )
    time_spent_minutes: int = Field(
        default=0,
        ge=0,
        description="Total time spent on the course.",
    )


class Certificate(BaseModel):
    """Completion certificate with verification code.

    Certificates are issued when a user passes a course assessment.
    They include a unique verification code for authenticity checks.

    Attributes:
        id: Unique certificate identifier.
        user_id: User who earned the certificate.
        course_id: Course that was completed.
        issued_at: When the certificate was issued.
        expires_at: When the certificate expires.
        verification_code: Unique code for verification.
        score: Score achieved on the assessment.
        user_name: User's display name (for certificate display).
        course_title: Course title (for certificate display).
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique certificate identifier.",
    )
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="User identifier.",
    )
    course_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Course identifier.",
    )
    issued_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Certificate issue timestamp.",
    )
    expires_at: datetime = Field(
        ...,
        description="Certificate expiration timestamp.",
    )
    verification_code: str = Field(
        default_factory=lambda: f"GL-{uuid.uuid4().hex[:12].upper()}",
        min_length=10,
        max_length=32,
        description="Unique verification code.",
    )
    score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Score achieved.",
    )
    user_name: str = Field(
        default="",
        max_length=256,
        description="User's display name.",
    )
    course_title: str = Field(
        default="",
        max_length=256,
        description="Course title.",
    )

    @model_validator(mode="after")
    def validate_expiry(self) -> "Certificate":
        """Ensure expires_at is after issued_at."""
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must be after issued_at.")
        return self


# ---------------------------------------------------------------------------
# Phishing Campaign Models
# ---------------------------------------------------------------------------


class PhishingCampaign(BaseModel):
    """Phishing simulation campaign definition.

    Manages the configuration and lifecycle of a phishing simulation
    campaign targeting a group of users.

    Attributes:
        id: Unique campaign identifier.
        name: Campaign name for identification.
        template_type: Type of phishing template to use.
        status: Current campaign status.
        created_at: When the campaign was created.
        scheduled_at: When the campaign is scheduled to send (if scheduled).
        sent_at: When emails were sent (if running/completed).
        completed_at: When the campaign ended.
        target_count: Number of target users.
        target_user_ids: List of target user IDs.
        target_roles: Target users by role (alternative to explicit list).
        created_by: User who created the campaign.
        metrics: Aggregated campaign metrics.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique campaign identifier.",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Campaign name.",
    )
    template_type: TemplateType = Field(
        ...,
        description="Type of phishing template.",
    )
    status: CampaignStatus = Field(
        default=CampaignStatus.DRAFT,
        description="Current campaign status.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Campaign creation timestamp.",
    )
    scheduled_at: Optional[datetime] = Field(
        default=None,
        description="Scheduled send time.",
    )
    sent_at: Optional[datetime] = Field(
        default=None,
        description="Actual send time.",
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Campaign completion time.",
    )
    target_count: int = Field(
        default=0,
        ge=0,
        description="Number of target users.",
    )
    target_user_ids: List[str] = Field(
        default_factory=list,
        description="Explicit list of target user IDs.",
    )
    target_roles: List[str] = Field(
        default_factory=list,
        description="Target users by role.",
    )
    created_by: str = Field(
        default="",
        max_length=256,
        description="Creator user ID.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional campaign metadata.",
    )


class PhishingResult(BaseModel):
    """Individual phishing test result for a single user.

    Tracks each user's interaction with a phishing email.

    Attributes:
        id: Unique result identifier.
        campaign_id: Parent campaign identifier.
        user_id: Target user identifier.
        sent_at: When the email was sent.
        opened_at: When the email was opened (via tracking pixel).
        clicked_at: When the link was clicked.
        reported_at: When the user reported the email as phishing.
        credentials_entered: Whether credentials were entered.
        user_agent: Browser user agent if link was clicked.
        ip_address: IP address if link was clicked.
        training_enrolled: Whether user was auto-enrolled in training.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique result identifier.",
    )
    campaign_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Campaign identifier.",
    )
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Target user identifier.",
    )
    sent_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Email send timestamp.",
    )
    opened_at: Optional[datetime] = Field(
        default=None,
        description="Email open timestamp.",
    )
    clicked_at: Optional[datetime] = Field(
        default=None,
        description="Link click timestamp.",
    )
    reported_at: Optional[datetime] = Field(
        default=None,
        description="Phishing report timestamp.",
    )
    credentials_entered: bool = Field(
        default=False,
        description="Whether credentials were submitted.",
    )
    user_agent: Optional[str] = Field(
        default=None,
        max_length=512,
        description="Browser user agent.",
    )
    ip_address: Optional[str] = Field(
        default=None,
        max_length=45,
        description="IP address.",
    )
    training_enrolled: bool = Field(
        default=False,
        description="Whether user was auto-enrolled in training.",
    )


class CampaignMetrics(BaseModel):
    """Aggregated phishing campaign metrics.

    Attributes:
        campaign_id: Campaign identifier.
        total_sent: Total emails sent.
        total_opened: Number of emails opened.
        total_clicked: Number of links clicked.
        total_credentials: Number of credential submissions.
        total_reported: Number of users who reported the phishing.
        open_rate: Percentage of emails opened.
        click_rate: Percentage of links clicked.
        credential_rate: Percentage of credential submissions.
        report_rate: Percentage of users who reported.
    """

    model_config = ConfigDict(extra="forbid")

    campaign_id: str = Field(
        ...,
        description="Campaign identifier.",
    )
    total_sent: int = Field(
        default=0,
        ge=0,
        description="Total emails sent.",
    )
    total_opened: int = Field(
        default=0,
        ge=0,
        description="Number opened.",
    )
    total_clicked: int = Field(
        default=0,
        ge=0,
        description="Number clicked.",
    )
    total_credentials: int = Field(
        default=0,
        ge=0,
        description="Credential submissions.",
    )
    total_reported: int = Field(
        default=0,
        ge=0,
        description="Number reported.",
    )
    open_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Open rate (0.0-1.0).",
    )
    click_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Click rate (0.0-1.0).",
    )
    credential_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Credential submission rate.",
    )
    report_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Report rate.",
    )


# ---------------------------------------------------------------------------
# Security Score Models
# ---------------------------------------------------------------------------


class SecurityScore(BaseModel):
    """Employee security posture score.

    Composite score based on multiple security factors including training
    completion, phishing resistance, MFA usage, password hygiene, and
    security incident history.

    Attributes:
        id: Unique score record identifier.
        user_id: User identifier.
        score: Overall security score (0-100).
        components: Breakdown of score by component.
        calculated_at: When the score was calculated.
        previous_score: Previous score for trend tracking.
        rank: User's rank within their team.
        percentile: User's percentile within organization.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique score record identifier.",
    )
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="User identifier.",
    )
    score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Overall security score (0-100).",
    )
    components: Dict[str, float] = Field(
        default_factory=dict,
        description="Score breakdown by component.",
    )
    calculated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Score calculation timestamp.",
    )
    previous_score: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Previous score for trend tracking.",
    )
    rank: Optional[int] = Field(
        default=None,
        ge=1,
        description="Rank within team.",
    )
    percentile: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Percentile within organization.",
    )


class UserProgress(BaseModel):
    """Aggregated user training progress.

    Summarizes a user's overall training status across all courses.

    Attributes:
        user_id: User identifier.
        total_required: Total required courses.
        total_completed: Completed courses.
        total_in_progress: In-progress courses.
        total_overdue: Overdue courses.
        completion_rate: Completion percentage.
        average_score: Average assessment score.
        certificates: List of valid certificate IDs.
        expiring_soon: Courses expiring within 30 days.
        security_score: Current security score.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(
        ...,
        description="User identifier.",
    )
    total_required: int = Field(
        default=0,
        ge=0,
        description="Total required courses.",
    )
    total_completed: int = Field(
        default=0,
        ge=0,
        description="Completed courses.",
    )
    total_in_progress: int = Field(
        default=0,
        ge=0,
        description="In-progress courses.",
    )
    total_overdue: int = Field(
        default=0,
        ge=0,
        description="Overdue courses.",
    )
    completion_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Completion rate (0.0-1.0).",
    )
    average_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Average assessment score.",
    )
    certificates: List[str] = Field(
        default_factory=list,
        description="Valid certificate IDs.",
    )
    expiring_soon: List[str] = Field(
        default_factory=list,
        description="Course IDs expiring soon.",
    )
    security_score: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Current security score.",
    )


class QuizSubmission(BaseModel):
    """Quiz answer submission.

    Represents a user's answers to a quiz for grading.

    Attributes:
        user_id: User submitting the quiz.
        course_id: Course the quiz belongs to.
        answers: Map of question_id to selected option index.
        submitted_at: Submission timestamp.
        time_taken_seconds: Time taken to complete the quiz.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(
        ...,
        description="User identifier.",
    )
    course_id: str = Field(
        ...,
        description="Course identifier.",
    )
    answers: Dict[str, int] = Field(
        ...,
        description="Map of question_id to selected option index.",
    )
    submitted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Submission timestamp.",
    )
    time_taken_seconds: int = Field(
        default=0,
        ge=0,
        description="Time taken in seconds.",
    )


__all__ = [
    "CampaignMetrics",
    "CampaignStatus",
    "Certificate",
    "ContentType",
    "Course",
    "CourseContent",
    "Module",
    "PhishingCampaign",
    "PhishingResult",
    "Question",
    "QuizSubmission",
    "SecurityScore",
    "TemplateType",
    "TrainingCompletion",
    "UserProgress",
]
