# -*- coding: utf-8 -*-
"""
Pydantic models for Supplier Engagement Agent.

Includes consent, campaign, email, portal, and analytics models.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any, Literal
from enum import Enum
from pydantic import BaseModel, Field, EmailStr, field_validator, ConfigDict


# Enums
class ConsentStatus(str, Enum):
    """Consent status for GDPR/CCPA/CAN-SPAM compliance."""
    OPTED_IN = "opted_in"
    OPTED_OUT = "opted_out"
    PENDING = "pending"


class LawfulBasis(str, Enum):
    """GDPR lawful basis for processing."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGITIMATE_INTEREST = "legitimate_interest"


class CampaignStatus(str, Enum):
    """Campaign lifecycle status."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class EmailStatus(str, Enum):
    """Email delivery status."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    OPENED = "opened"
    CLICKED = "clicked"
    BOUNCED = "bounced"
    FAILED = "failed"


class UploadStatus(str, Enum):
    """Data upload status."""
    IN_PROGRESS = "in_progress"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"


class BadgeType(str, Enum):
    """Gamification badges."""
    EARLY_ADOPTER = "early_adopter"
    DATA_CHAMPION = "data_champion"
    COMPLETE_PROFILE = "complete_profile"
    QUALITY_LEADER = "quality_leader"
    FAST_RESPONDER = "fast_responder"


# Consent Models
class ConsentRecord(BaseModel):
    """GDPR/CCPA/CAN-SPAM compliant consent record."""
    model_config = ConfigDict(use_enum_values=True)

    supplier_id: str = Field(..., description="Unique supplier identifier")
    email_address: EmailStr = Field(..., description="Contact email")
    consent_status: ConsentStatus = Field(
        default=ConsentStatus.PENDING,
        description="Current consent status"
    )
    lawful_basis: LawfulBasis = Field(
        default=LawfulBasis.LEGITIMATE_INTEREST,
        description="GDPR lawful basis"
    )
    country: str = Field(..., description="Supplier country (ISO 3166-1 alpha-2)")
    consent_date: Optional[datetime] = Field(
        default=None,
        description="Date consent was granted"
    )
    opt_out_date: Optional[datetime] = Field(
        default=None,
        description="Date supplier opted out"
    )
    opt_out_reason: Optional[str] = Field(
        default=None,
        description="Reason for opting out"
    )
    data_processing_agreement_url: Optional[str] = Field(
        default=None,
        description="Link to DPA"
    )
    retention_period_days: int = Field(
        default=730,
        description="Data retention period (GDPR Article 17)"
    )
    last_contacted: Optional[datetime] = Field(
        default=None,
        description="Last contact timestamp"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    @field_validator('country')
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Validate ISO country code."""
        if len(v) not in [2, 5]:  # ISO 3166-1 or US-CA format
            raise ValueError(f"Invalid country code: {v}")
        return v.upper()


# Campaign Models
class EmailTemplate(BaseModel):
    """Email template with personalization."""
    model_config = ConfigDict(use_enum_values=True)

    template_id: str = Field(..., description="Template identifier")
    subject: str = Field(..., description="Email subject line")
    body_html: str = Field(..., description="HTML version")
    body_text: str = Field(..., description="Plain text version")
    language: str = Field(default="en", description="ISO 639-1 language code")
    personalization_fields: List[str] = Field(
        default_factory=list,
        description="Fields to personalize (e.g., {contact_name})"
    )

    @field_validator('subject')
    @classmethod
    def validate_subject(cls, v: str) -> str:
        """Ensure subject is not empty."""
        if not v.strip():
            raise ValueError("Subject cannot be empty")
        return v


class EmailSequence(BaseModel):
    """Multi-touch email sequence."""
    model_config = ConfigDict(use_enum_values=True)

    sequence_id: str = Field(..., description="Sequence identifier")
    name: str = Field(..., description="Sequence name")
    touches: List[Dict[str, Any]] = Field(
        ...,
        description="Email touches with day offset"
    )

    @field_validator('touches')
    @classmethod
    def validate_touches(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure touches have required fields."""
        for idx, touch in enumerate(v):
            if 'day_offset' not in touch or 'template' not in touch:
                raise ValueError(f"Touch {idx} missing day_offset or template")
        return v


class Campaign(BaseModel):
    """Supplier engagement campaign."""
    model_config = ConfigDict(use_enum_values=True)

    campaign_id: str = Field(..., description="Campaign identifier")
    name: str = Field(..., description="Campaign name")
    target_suppliers: List[str] = Field(..., description="Supplier IDs to target")
    email_sequence: EmailSequence = Field(..., description="Email sequence")
    start_date: datetime = Field(..., description="Campaign start")
    end_date: datetime = Field(..., description="Campaign end")
    status: CampaignStatus = Field(
        default=CampaignStatus.DRAFT,
        description="Campaign status"
    )
    response_rate_target: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Target response rate"
    )

    # Metrics
    emails_sent: int = Field(default=0, ge=0)
    emails_delivered: int = Field(default=0, ge=0)
    emails_opened: int = Field(default=0, ge=0)
    emails_clicked: int = Field(default=0, ge=0)
    portal_visits: int = Field(default=0, ge=0)
    data_submissions: int = Field(default=0, ge=0)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def response_rate(self) -> float:
        """Calculate response rate."""
        if self.emails_sent == 0:
            return 0.0
        return self.data_submissions / self.emails_sent


# Email Models
class EmailMessage(BaseModel):
    """Email message to send."""
    model_config = ConfigDict(use_enum_values=True)

    message_id: str = Field(..., description="Message identifier")
    campaign_id: str = Field(..., description="Campaign ID")
    supplier_id: str = Field(..., description="Supplier ID")
    to_email: EmailStr = Field(..., description="Recipient email")
    subject: str = Field(..., description="Email subject")
    body_html: str = Field(..., description="HTML body")
    body_text: str = Field(..., description="Plain text body")
    status: EmailStatus = Field(
        default=EmailStatus.PENDING,
        description="Email status"
    )
    scheduled_send: datetime = Field(..., description="Scheduled send time")
    sent_at: Optional[datetime] = Field(default=None)
    delivered_at: Optional[datetime] = Field(default=None)
    opened_at: Optional[datetime] = Field(default=None)
    clicked_at: Optional[datetime] = Field(default=None)
    unsubscribe_url: str = Field(..., description="Unsubscribe link (mandatory)")
    tracking_metadata: Dict[str, Any] = Field(default_factory=dict)


class EmailResult(BaseModel):
    """Result of email send operation."""
    model_config = ConfigDict(use_enum_values=True)

    success: bool
    message_id: str
    supplier_id: str
    status: EmailStatus
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Portal Models
class SupplierPortalSession(BaseModel):
    """Supplier portal authentication session."""
    model_config = ConfigDict(use_enum_values=True)

    session_id: str = Field(..., description="Session identifier")
    supplier_id: str = Field(..., description="Supplier ID")
    email: EmailStr = Field(..., description="Supplier email")
    magic_link_token: Optional[str] = Field(
        default=None,
        description="Magic link token (passwordless)"
    )
    oauth_provider: Optional[str] = Field(
        default=None,
        description="OAuth provider (google, microsoft)"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(..., description="Session expiry")
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    ip_address: Optional[str] = Field(default=None)


class DataUpload(BaseModel):
    """Supplier data upload."""
    model_config = ConfigDict(use_enum_values=True)

    upload_id: str = Field(..., description="Upload identifier")
    supplier_id: str = Field(..., description="Supplier ID")
    campaign_id: str = Field(..., description="Campaign ID")
    file_name: str = Field(..., description="Uploaded file name")
    file_type: str = Field(..., description="File type (csv, xlsx, json)")
    file_size_bytes: int = Field(..., ge=0, description="File size")
    status: UploadStatus = Field(
        default=UploadStatus.IN_PROGRESS,
        description="Upload status"
    )
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    validated_at: Optional[datetime] = Field(default=None)
    validation_errors: List[str] = Field(default_factory=list)
    data_quality_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="DQI score"
    )
    records_count: int = Field(default=0, ge=0)


class ValidationResult(BaseModel):
    """Live validation result for portal."""
    model_config = ConfigDict(use_enum_values=True)

    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    data_quality_score: float = Field(ge=0.0, le=1.0)
    completeness_percentage: float = Field(ge=0.0, le=100.0)
    field_validations: Dict[str, bool] = Field(default_factory=dict)


# Gamification Models
class SupplierBadge(BaseModel):
    """Gamification badge."""
    model_config = ConfigDict(use_enum_values=True)

    badge_type: BadgeType
    earned_at: datetime = Field(default_factory=datetime.utcnow)
    criteria_met: str = Field(..., description="Criteria description")


class SupplierProgress(BaseModel):
    """Supplier progress tracking."""
    model_config = ConfigDict(use_enum_values=True)

    supplier_id: str
    campaign_id: str
    completion_percentage: float = Field(ge=0.0, le=100.0)
    data_quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    badges_earned: List[SupplierBadge] = Field(default_factory=list)
    leaderboard_rank: Optional[int] = Field(default=None, ge=1)
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class Leaderboard(BaseModel):
    """Campaign leaderboard."""
    model_config = ConfigDict(use_enum_values=True)

    campaign_id: str
    entries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ranked supplier entries"
    )
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# Analytics Models
class CampaignAnalytics(BaseModel):
    """Campaign performance analytics."""
    model_config = ConfigDict(use_enum_values=True)

    campaign_id: str
    campaign_name: str

    # Email metrics
    emails_sent: int = Field(ge=0)
    emails_delivered: int = Field(ge=0)
    emails_opened: int = Field(ge=0)
    emails_clicked: int = Field(ge=0)
    emails_bounced: int = Field(ge=0)

    # Portal metrics
    portal_visits: int = Field(ge=0)
    unique_visitors: int = Field(ge=0)
    data_submissions: int = Field(ge=0)

    # Calculated rates
    delivery_rate: float = Field(ge=0.0, le=1.0)
    open_rate: float = Field(ge=0.0, le=1.0)
    click_rate: float = Field(ge=0.0, le=1.0)
    response_rate: float = Field(ge=0.0, le=1.0)

    # Time metrics
    avg_time_to_response_hours: Optional[float] = Field(default=None, ge=0.0)

    # Quality metrics
    avg_data_quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    generated_at: datetime = Field(default_factory=datetime.utcnow)
