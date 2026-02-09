# -*- coding: utf-8 -*-
"""
Supplier Questionnaire Processor Data Models - AGENT-DATA-008

Pydantic v2 data models for the Supplier Questionnaire Processor SDK.
Defines enumerations, core data models, and request wrappers for
questionnaire template management, distribution, response collection,
validation, scoring, follow-up, and analytics.

Models:
    - Enumerations: Framework, QuestionType, QuestionnaireStatus,
        DistributionStatus, DistributionChannel, ResponseStatus,
        ValidationSeverity, ReminderType, EscalationLevel, CDPGrade,
        PerformanceTier, ReportFormat
    - Core models: TemplateQuestion, TemplateSection, QuestionnaireTemplate,
        Distribution, QuestionnaireResponse, Answer, ValidationCheck,
        ValidationSummary, QuestionnaireScore, FollowUpAction,
        CampaignAnalytics
    - Request models: CreateTemplateRequest, DistributeRequest,
        SubmitResponseRequest

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-008 Supplier Questionnaire Processor
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# =============================================================================
# Enumerations
# =============================================================================


class Framework(str, Enum):
    """Supported sustainability reporting frameworks."""

    CDP_CLIMATE = "cdp_climate"
    CDP_WATER = "cdp_water"
    CDP_FORESTS = "cdp_forests"
    ECOVADIS = "ecovadis"
    DJSI = "djsi"
    GRI = "gri"
    SASB = "sasb"
    TCFD = "tcfd"
    TNFD = "tnfd"
    SBT = "sbt"
    CUSTOM = "custom"


class QuestionType(str, Enum):
    """Question input types for questionnaire templates."""

    TEXT = "text"
    NUMERIC = "numeric"
    SINGLE_CHOICE = "single_choice"
    MULTI_CHOICE = "multi_choice"
    YES_NO = "yes_no"
    DATE = "date"
    FILE_UPLOAD = "file_upload"
    TABLE = "table"
    PERCENTAGE = "percentage"
    CURRENCY = "currency"


class QuestionnaireStatus(str, Enum):
    """Lifecycle status of a questionnaire template."""

    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class DistributionStatus(str, Enum):
    """Delivery status of a distributed questionnaire."""

    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    OPENED = "opened"
    IN_PROGRESS = "in_progress"
    SUBMITTED = "submitted"
    BOUNCED = "bounced"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class DistributionChannel(str, Enum):
    """Delivery channels for questionnaire distribution."""

    EMAIL = "email"
    PORTAL = "portal"
    API = "api"
    BULK_UPLOAD = "bulk_upload"


class ResponseStatus(str, Enum):
    """Status of a supplier's questionnaire response."""

    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    SUBMITTED = "submitted"
    VALIDATED = "validated"
    SCORED = "scored"
    REOPENED = "reopened"
    REJECTED = "rejected"


class ValidationSeverity(str, Enum):
    """Severity levels for validation checks."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ReminderType(str, Enum):
    """Types of follow-up reminders."""

    GENTLE = "gentle"
    FIRM = "firm"
    URGENT = "urgent"
    FINAL = "final"


class EscalationLevel(str, Enum):
    """Escalation levels for non-responsive suppliers."""

    LEVEL_1 = "level_1"
    LEVEL_2 = "level_2"
    LEVEL_3 = "level_3"
    LEVEL_4 = "level_4"
    LEVEL_5 = "level_5"


class CDPGrade(str, Enum):
    """CDP scoring grades (A to D-minus, plus F)."""

    A = "A"
    A_MINUS = "A-"
    B = "B"
    B_MINUS = "B-"
    C = "C"
    C_MINUS = "C-"
    D = "D"
    D_MINUS = "D-"
    F = "F"


class PerformanceTier(str, Enum):
    """Supplier performance tiers based on questionnaire scores."""

    LEADER = "leader"
    ADVANCED = "advanced"
    INTERMEDIATE = "intermediate"
    BEGINNER = "beginner"
    LAGGARD = "laggard"


class ReportFormat(str, Enum):
    """Supported report output formats."""

    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"


# =============================================================================
# Core Data Models
# =============================================================================


class TemplateQuestion(BaseModel):
    """A single question within a questionnaire template section.

    Attributes:
        question_id: Unique identifier for this question.
        code: Short question code (e.g. "C1.1a").
        text: Full question text.
        question_type: Input type for this question.
        required: Whether an answer is mandatory.
        choices: Available choices for single/multi-choice questions.
        help_text: Guidance text for respondents.
        weight: Scoring weight for this question (0.0-10.0).
        framework_ref: Reference to framework-specific question ID.
        translations: Translations of text keyed by ISO 639-1 code.
        validation_rules: Validation constraints as key-value pairs.
        order: Display order within the section.
    """

    question_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this question",
    )
    code: str = Field(
        default="", description="Short question code",
    )
    text: str = Field(
        ..., description="Full question text",
    )
    question_type: QuestionType = Field(
        default=QuestionType.TEXT,
        description="Input type for this question",
    )
    required: bool = Field(
        default=True, description="Whether an answer is mandatory",
    )
    choices: List[str] = Field(
        default_factory=list,
        description="Available choices for single/multi-choice",
    )
    help_text: str = Field(
        default="", description="Guidance text for respondents",
    )
    weight: float = Field(
        default=1.0, ge=0.0, le=10.0,
        description="Scoring weight for this question",
    )
    framework_ref: str = Field(
        default="", description="Framework-specific question reference",
    )
    translations: Dict[str, str] = Field(
        default_factory=dict,
        description="Translations keyed by ISO 639-1 code",
    )
    validation_rules: Dict[str, Any] = Field(
        default_factory=dict,
        description="Validation constraints as key-value pairs",
    )
    order: int = Field(
        default=0, ge=0, description="Display order within section",
    )

    model_config = {"extra": "forbid"}


class TemplateSection(BaseModel):
    """A section within a questionnaire template.

    Attributes:
        section_id: Unique identifier for this section.
        name: Section display name.
        description: Section description.
        order: Display order within the template.
        questions: List of questions in this section.
        weight: Scoring weight for this section (0.0-10.0).
        framework_ref: Framework-specific section reference.
        translations: Translations of name/description by ISO 639-1 code.
    """

    section_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this section",
    )
    name: str = Field(
        ..., description="Section display name",
    )
    description: str = Field(
        default="", description="Section description",
    )
    order: int = Field(
        default=0, ge=0, description="Display order within template",
    )
    questions: List[TemplateQuestion] = Field(
        default_factory=list,
        description="Questions in this section",
    )
    weight: float = Field(
        default=1.0, ge=0.0, le=10.0,
        description="Scoring weight for this section",
    )
    framework_ref: str = Field(
        default="", description="Framework-specific section reference",
    )
    translations: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="Translations of name/description by ISO 639-1 code",
    )

    model_config = {"extra": "forbid"}


class QuestionnaireTemplate(BaseModel):
    """A complete questionnaire template definition.

    Attributes:
        template_id: Unique identifier for this template.
        name: Template display name.
        framework: Target reporting framework.
        version: Template version (incremented on updates).
        status: Current lifecycle status.
        sections: Ordered list of template sections.
        language: Primary language (ISO 639-1 code).
        supported_languages: List of supported language codes.
        description: Template description.
        created_at: Timestamp when the template was created.
        updated_at: Timestamp of the last update.
        created_by: User who created the template.
        tags: Free-form tags for organisation.
        provenance_hash: SHA-256 provenance chain hash.
    """

    template_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this template",
    )
    name: str = Field(
        ..., description="Template display name",
    )
    framework: Framework = Field(
        ..., description="Target reporting framework",
    )
    version: int = Field(
        default=1, ge=1, description="Template version number",
    )
    status: QuestionnaireStatus = Field(
        default=QuestionnaireStatus.DRAFT,
        description="Current lifecycle status",
    )
    sections: List[TemplateSection] = Field(
        default_factory=list,
        description="Ordered list of template sections",
    )
    language: str = Field(
        default="en", description="Primary language (ISO 639-1)",
    )
    supported_languages: List[str] = Field(
        default_factory=lambda: ["en"],
        description="Supported language codes",
    )
    description: str = Field(
        default="", description="Template description",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="Last update timestamp",
    )
    created_by: str = Field(
        default="system", description="User who created the template",
    )
    tags: List[str] = Field(
        default_factory=list, description="Free-form tags",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance chain hash",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v


class Answer(BaseModel):
    """A single answer to a questionnaire question.

    Attributes:
        question_id: Question this answer responds to.
        value: The answer value (text, number, list, etc.).
        unit: Optional unit for numeric answers.
        confidence: Respondent confidence level (0.0-1.0).
        evidence_refs: References to supporting evidence/documents.
        notes: Additional notes or comments.
    """

    question_id: str = Field(
        ..., description="Question this answer responds to",
    )
    value: Any = Field(
        ..., description="The answer value",
    )
    unit: str = Field(
        default="", description="Unit for numeric answers",
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Respondent confidence level",
    )
    evidence_refs: List[str] = Field(
        default_factory=list,
        description="References to supporting evidence",
    )
    notes: str = Field(
        default="", description="Additional notes or comments",
    )

    model_config = {"extra": "forbid"}

    @field_validator("question_id")
    @classmethod
    def validate_question_id(cls, v: str) -> str:
        """Validate question_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("question_id must be non-empty")
        return v


class Distribution(BaseModel):
    """Record of a questionnaire distributed to a supplier.

    Attributes:
        distribution_id: Unique distribution identifier.
        template_id: Template that was distributed.
        supplier_id: Target supplier identifier.
        supplier_name: Supplier display name.
        supplier_email: Supplier contact email.
        campaign_id: Campaign this distribution belongs to.
        channel: Delivery channel used.
        status: Current delivery status.
        access_token: Secure portal access token.
        deadline: Submission deadline date.
        sent_at: Timestamp when sent.
        delivered_at: Timestamp when delivered.
        opened_at: Timestamp when first opened.
        submitted_at: Timestamp when response submitted.
        created_at: Creation timestamp.
        reminder_count: Number of reminders sent.
        provenance_hash: SHA-256 provenance chain hash.
    """

    distribution_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique distribution identifier",
    )
    template_id: str = Field(
        ..., description="Template that was distributed",
    )
    supplier_id: str = Field(
        ..., description="Target supplier identifier",
    )
    supplier_name: str = Field(
        default="", description="Supplier display name",
    )
    supplier_email: str = Field(
        default="", description="Supplier contact email",
    )
    campaign_id: str = Field(
        default="", description="Campaign this distribution belongs to",
    )
    channel: DistributionChannel = Field(
        default=DistributionChannel.EMAIL,
        description="Delivery channel used",
    )
    status: DistributionStatus = Field(
        default=DistributionStatus.PENDING,
        description="Current delivery status",
    )
    access_token: str = Field(
        default="", description="Secure portal access token",
    )
    deadline: Optional[date] = Field(
        None, description="Submission deadline date",
    )
    sent_at: Optional[datetime] = Field(
        None, description="Timestamp when sent",
    )
    delivered_at: Optional[datetime] = Field(
        None, description="Timestamp when delivered",
    )
    opened_at: Optional[datetime] = Field(
        None, description="Timestamp when first opened",
    )
    submitted_at: Optional[datetime] = Field(
        None, description="Timestamp when response submitted",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Creation timestamp",
    )
    reminder_count: int = Field(
        default=0, ge=0,
        description="Number of reminders sent",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance chain hash",
    )

    model_config = {"extra": "forbid"}

    @field_validator("template_id")
    @classmethod
    def validate_template_id(cls, v: str) -> str:
        """Validate template_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("template_id must be non-empty")
        return v

    @field_validator("supplier_id")
    @classmethod
    def validate_supplier_id(cls, v: str) -> str:
        """Validate supplier_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("supplier_id must be non-empty")
        return v


class QuestionnaireResponse(BaseModel):
    """A supplier's response to a distributed questionnaire.

    Attributes:
        response_id: Unique response identifier.
        distribution_id: Distribution this response belongs to.
        template_id: Template the response is for.
        supplier_id: Responding supplier identifier.
        answers: List of answers to template questions.
        status: Current response status.
        language: Language used in the response.
        completion_pct: Response completion percentage (0-100).
        submitted_at: Timestamp when submitted.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
        confirmation_token: Acknowledgement confirmation token.
        provenance_hash: SHA-256 provenance chain hash.
    """

    response_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique response identifier",
    )
    distribution_id: str = Field(
        ..., description="Distribution this response belongs to",
    )
    template_id: str = Field(
        ..., description="Template the response is for",
    )
    supplier_id: str = Field(
        ..., description="Responding supplier identifier",
    )
    answers: List[Answer] = Field(
        default_factory=list,
        description="List of answers to template questions",
    )
    status: ResponseStatus = Field(
        default=ResponseStatus.DRAFT,
        description="Current response status",
    )
    language: str = Field(
        default="en", description="Language used in the response",
    )
    completion_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Response completion percentage",
    )
    submitted_at: Optional[datetime] = Field(
        None, description="Timestamp when submitted",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="Last update timestamp",
    )
    confirmation_token: str = Field(
        default="", description="Acknowledgement confirmation token",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance chain hash",
    )

    model_config = {"extra": "forbid"}

    @field_validator("distribution_id")
    @classmethod
    def validate_distribution_id(cls, v: str) -> str:
        """Validate distribution_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("distribution_id must be non-empty")
        return v

    @field_validator("supplier_id")
    @classmethod
    def validate_supplier_id(cls, v: str) -> str:
        """Validate supplier_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("supplier_id must be non-empty")
        return v


class ValidationCheck(BaseModel):
    """A single validation check result.

    Attributes:
        check_id: Unique check identifier.
        check_type: Category of validation check.
        question_id: Question the check applies to (if applicable).
        severity: Severity level of the check result.
        passed: Whether the check passed.
        message: Human-readable check result message.
        expected: Expected value or pattern (if applicable).
        actual: Actual value found (if applicable).
        suggestion: Suggested fix (if check failed).
    """

    check_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique check identifier",
    )
    check_type: str = Field(
        ..., description="Category of validation check",
    )
    question_id: str = Field(
        default="", description="Question the check applies to",
    )
    severity: ValidationSeverity = Field(
        default=ValidationSeverity.ERROR,
        description="Severity level",
    )
    passed: bool = Field(
        default=True, description="Whether the check passed",
    )
    message: str = Field(
        default="", description="Human-readable check result message",
    )
    expected: str = Field(
        default="", description="Expected value or pattern",
    )
    actual: str = Field(
        default="", description="Actual value found",
    )
    suggestion: str = Field(
        default="", description="Suggested fix if check failed",
    )

    model_config = {"extra": "forbid"}


class ValidationSummary(BaseModel):
    """Aggregated validation summary for a questionnaire response.

    Attributes:
        response_id: Response that was validated.
        template_id: Template used for validation.
        checks: List of individual validation check results.
        total_checks: Total number of checks performed.
        passed_checks: Number of checks that passed.
        failed_checks: Number of checks that failed.
        warning_count: Number of warning-level issues.
        error_count: Number of error-level issues.
        is_valid: Overall validation result.
        data_quality_score: Data quality score (0-100).
        validated_at: Timestamp when validation was performed.
        provenance_hash: SHA-256 provenance chain hash.
    """

    response_id: str = Field(
        default="", description="Response that was validated",
    )
    template_id: str = Field(
        default="", description="Template used for validation",
    )
    checks: List[ValidationCheck] = Field(
        default_factory=list,
        description="Individual validation check results",
    )
    total_checks: int = Field(
        default=0, ge=0,
        description="Total number of checks performed",
    )
    passed_checks: int = Field(
        default=0, ge=0,
        description="Number of checks that passed",
    )
    failed_checks: int = Field(
        default=0, ge=0,
        description="Number of checks that failed",
    )
    warning_count: int = Field(
        default=0, ge=0,
        description="Number of warning-level issues",
    )
    error_count: int = Field(
        default=0, ge=0,
        description="Number of error-level issues",
    )
    is_valid: bool = Field(
        default=True, description="Overall validation result",
    )
    data_quality_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Data quality score (0-100)",
    )
    validated_at: datetime = Field(
        default_factory=_utcnow,
        description="Validation timestamp",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance chain hash",
    )

    model_config = {"extra": "forbid"}


class QuestionnaireScore(BaseModel):
    """Scoring result for a questionnaire response.

    Attributes:
        score_id: Unique score identifier.
        response_id: Response that was scored.
        template_id: Template used for scoring.
        supplier_id: Supplier who was scored.
        framework: Framework used for scoring methodology.
        raw_score: Raw calculated score.
        normalized_score: Score normalized to 0-100 scale.
        cdp_grade: CDP letter grade (if CDP framework).
        performance_tier: Performance tier classification.
        section_scores: Per-section score breakdown.
        scored_at: Timestamp when scoring was performed.
        methodology: Scoring methodology description.
        provenance_hash: SHA-256 provenance chain hash.
    """

    score_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique score identifier",
    )
    response_id: str = Field(
        default="", description="Response that was scored",
    )
    template_id: str = Field(
        default="", description="Template used for scoring",
    )
    supplier_id: str = Field(
        default="", description="Supplier who was scored",
    )
    framework: Framework = Field(
        default=Framework.CUSTOM,
        description="Framework used for scoring",
    )
    raw_score: float = Field(
        default=0.0, description="Raw calculated score",
    )
    normalized_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Score normalized to 0-100 scale",
    )
    cdp_grade: Optional[CDPGrade] = Field(
        None, description="CDP letter grade (if applicable)",
    )
    performance_tier: PerformanceTier = Field(
        default=PerformanceTier.BEGINNER,
        description="Performance tier classification",
    )
    section_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-section score breakdown",
    )
    scored_at: datetime = Field(
        default_factory=_utcnow,
        description="Scoring timestamp",
    )
    methodology: str = Field(
        default="", description="Scoring methodology description",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance chain hash",
    )

    model_config = {"extra": "forbid"}


class FollowUpAction(BaseModel):
    """A scheduled or executed follow-up action.

    Attributes:
        action_id: Unique action identifier.
        distribution_id: Distribution this action is for.
        campaign_id: Campaign this action belongs to.
        supplier_id: Target supplier identifier.
        reminder_type: Type of reminder.
        escalation_level: Escalation level (if escalated).
        status: Current action status (scheduled, sent, cancelled).
        scheduled_at: Scheduled execution time.
        sent_at: Actual execution time.
        message: Reminder message content.
        provenance_hash: SHA-256 provenance chain hash.
    """

    action_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique action identifier",
    )
    distribution_id: str = Field(
        ..., description="Distribution this action is for",
    )
    campaign_id: str = Field(
        default="", description="Campaign this action belongs to",
    )
    supplier_id: str = Field(
        default="", description="Target supplier identifier",
    )
    reminder_type: ReminderType = Field(
        default=ReminderType.GENTLE,
        description="Type of reminder",
    )
    escalation_level: Optional[EscalationLevel] = Field(
        None, description="Escalation level if escalated",
    )
    status: str = Field(
        default="scheduled", description="Current action status",
    )
    scheduled_at: Optional[datetime] = Field(
        None, description="Scheduled execution time",
    )
    sent_at: Optional[datetime] = Field(
        None, description="Actual execution time",
    )
    message: str = Field(
        default="", description="Reminder message content",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance chain hash",
    )

    model_config = {"extra": "forbid"}

    @field_validator("distribution_id")
    @classmethod
    def validate_distribution_id(cls, v: str) -> str:
        """Validate distribution_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("distribution_id must be non-empty")
        return v


class CampaignAnalytics(BaseModel):
    """Aggregated analytics for a questionnaire campaign.

    Attributes:
        campaign_id: Campaign identifier.
        total_distributions: Total questionnaires distributed.
        total_responses: Total responses received.
        response_rate: Response rate as percentage (0-100).
        avg_score: Average score across all responses.
        avg_completion_pct: Average completion percentage.
        avg_data_quality: Average data quality score.
        score_distribution: Score histogram buckets.
        status_breakdown: Count by distribution status.
        section_avg_scores: Average score per section.
        generated_at: Timestamp when analytics were generated.
        provenance_hash: SHA-256 provenance chain hash.
    """

    campaign_id: str = Field(
        default="", description="Campaign identifier",
    )
    total_distributions: int = Field(
        default=0, ge=0,
        description="Total questionnaires distributed",
    )
    total_responses: int = Field(
        default=0, ge=0,
        description="Total responses received",
    )
    response_rate: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Response rate as percentage",
    )
    avg_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Average score across all responses",
    )
    avg_completion_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Average completion percentage",
    )
    avg_data_quality: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Average data quality score",
    )
    score_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Score histogram buckets",
    )
    status_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Count by distribution status",
    )
    section_avg_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Average score per section",
    )
    generated_at: datetime = Field(
        default_factory=_utcnow,
        description="Analytics generation timestamp",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance chain hash",
    )

    model_config = {"extra": "forbid"}


# =============================================================================
# Request Models
# =============================================================================


class CreateTemplateRequest(BaseModel):
    """Request body for creating a new questionnaire template.

    Attributes:
        name: Template display name.
        framework: Target reporting framework.
        sections: Initial sections to include.
        language: Primary language (ISO 639-1 code).
        description: Template description.
        tags: Free-form tags.
    """

    name: str = Field(
        ..., description="Template display name",
    )
    framework: Framework = Field(
        ..., description="Target reporting framework",
    )
    sections: List[TemplateSection] = Field(
        default_factory=list,
        description="Initial sections to include",
    )
    language: str = Field(
        default="en", description="Primary language (ISO 639-1)",
    )
    description: str = Field(
        default="", description="Template description",
    )
    tags: List[str] = Field(
        default_factory=list, description="Free-form tags",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v


class DistributeRequest(BaseModel):
    """Request body for distributing a questionnaire to suppliers.

    Attributes:
        template_id: Template to distribute.
        supplier_list: List of supplier dicts (id, name, email).
        channel: Delivery channel.
        campaign_id: Campaign to associate with.
        deadline_days: Number of days until deadline.
    """

    template_id: str = Field(
        ..., description="Template to distribute",
    )
    supplier_list: List[Dict[str, str]] = Field(
        ..., description="List of supplier dicts with id, name, email",
    )
    channel: DistributionChannel = Field(
        default=DistributionChannel.EMAIL,
        description="Delivery channel",
    )
    campaign_id: str = Field(
        default="", description="Campaign to associate with",
    )
    deadline_days: int = Field(
        default=30, ge=1, le=365,
        description="Days until deadline",
    )

    model_config = {"extra": "forbid"}

    @field_validator("template_id")
    @classmethod
    def validate_template_id(cls, v: str) -> str:
        """Validate template_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("template_id must be non-empty")
        return v


class SubmitResponseRequest(BaseModel):
    """Request body for submitting a questionnaire response.

    Attributes:
        distribution_id: Distribution being responded to.
        answers: List of answers to template questions.
        language: Language used in the response.
    """

    distribution_id: str = Field(
        ..., description="Distribution being responded to",
    )
    answers: List[Answer] = Field(
        ..., description="List of answers to template questions",
    )
    language: str = Field(
        default="en", description="Language used in the response",
    )

    model_config = {"extra": "forbid"}

    @field_validator("distribution_id")
    @classmethod
    def validate_distribution_id(cls, v: str) -> str:
        """Validate distribution_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("distribution_id must be non-empty")
        return v


__all__ = [
    # Enumerations
    "Framework",
    "QuestionType",
    "QuestionnaireStatus",
    "DistributionStatus",
    "DistributionChannel",
    "ResponseStatus",
    "ValidationSeverity",
    "ReminderType",
    "EscalationLevel",
    "CDPGrade",
    "PerformanceTier",
    "ReportFormat",
    # Core models
    "TemplateQuestion",
    "TemplateSection",
    "QuestionnaireTemplate",
    "Answer",
    "Distribution",
    "QuestionnaireResponse",
    "ValidationCheck",
    "ValidationSummary",
    "QuestionnaireScore",
    "FollowUpAction",
    "CampaignAnalytics",
    # Request models
    "CreateTemplateRequest",
    "DistributeRequest",
    "SubmitResponseRequest",
]
