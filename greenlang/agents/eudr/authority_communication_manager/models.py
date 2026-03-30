# -*- coding: utf-8 -*-
"""
Authority Communication Manager Models - AGENT-EUDR-040

Pydantic v2 models for authority communication management, information
request handling, inspection coordination, non-compliance processing,
appeal management, secure document exchange, multi-channel notification
routing, and multi-language template rendering.

All models use Decimal for penalty amounts to ensure deterministic,
bit-perfect reproducibility in compliance calculations.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-040 Authority Communication Manager (GL-EUDR-ACM-040)
Regulation: EU 2023/1115 (EUDR) Articles 15, 16, 17, 19, 31
Status: Production Ready
"""
from __future__ import annotations

import enum
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import Field
from greenlang.schemas import GreenLangBase


# ---------------------------------------------------------------------------
# Enums (13)
# ---------------------------------------------------------------------------


class CommunicationType(str, enum.Enum):
    """Types of authority communications per EUDR."""

    INFORMATION_REQUEST = "information_request"
    INSPECTION_NOTICE = "inspection_notice"
    NON_COMPLIANCE_NOTICE = "non_compliance_notice"
    PENALTY_NOTICE = "penalty_notice"
    APPEAL_ACKNOWLEDGMENT = "appeal_acknowledgment"
    APPEAL_DECISION = "appeal_decision"
    COMPLIANCE_CONFIRMATION = "compliance_confirmation"
    CORRECTIVE_ACTION_ORDER = "corrective_action_order"
    MARKET_WITHDRAWAL_ORDER = "market_withdrawal_order"
    GENERAL_CORRESPONDENCE = "general_correspondence"
    DDS_SUBMISSION_RECEIPT = "dds_submission_receipt"
    STATUS_UPDATE = "status_update"


class CommunicationStatus(str, enum.Enum):
    """Lifecycle status of a communication."""

    DRAFT = "draft"
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    RESPONDED = "responded"
    OVERDUE = "overdue"
    ESCALATED = "escalated"
    CLOSED = "closed"
    ARCHIVED = "archived"


class CommunicationPriority(str, enum.Enum):
    """Priority classification for communications."""

    URGENT = "urgent"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    ROUTINE = "routine"


class InformationRequestType(str, enum.Enum):
    """Types of information requests per EUDR Article 17."""

    DDS_CLARIFICATION = "dds_clarification"
    SUPPLY_CHAIN_EVIDENCE = "supply_chain_evidence"
    GEOLOCATION_VERIFICATION = "geolocation_verification"
    DEFORESTATION_EVIDENCE = "deforestation_evidence"
    LEGALITY_DOCUMENTATION = "legality_documentation"
    RISK_ASSESSMENT_DETAILS = "risk_assessment_details"
    MITIGATION_MEASURES = "mitigation_measures"
    COMMODITY_TRACEABILITY = "commodity_traceability"
    SUPPLIER_DOCUMENTATION = "supplier_documentation"
    AUDIT_REPORT_REQUEST = "audit_report_request"


class InspectionType(str, enum.Enum):
    """Types of on-the-spot checks per EUDR Article 15."""

    ANNOUNCED = "announced"
    UNANNOUNCED = "unannounced"
    FOLLOW_UP = "follow_up"
    REMOTE = "remote"
    DOCUMENT_REVIEW = "document_review"
    PHYSICAL_INSPECTION = "physical_inspection"


class ViolationType(str, enum.Enum):
    """EUDR violation categories per Article 16."""

    MISSING_DDS = "missing_dds"
    INCOMPLETE_DDS = "incomplete_dds"
    FALSE_INFORMATION = "false_information"
    DEFORESTATION_LINK = "deforestation_link"
    LEGALITY_VIOLATION = "legality_violation"
    TRACEABILITY_FAILURE = "traceability_failure"
    RISK_ASSESSMENT_FAILURE = "risk_assessment_failure"
    INSUFFICIENT_MITIGATION = "insufficient_mitigation"
    RECORD_KEEPING_FAILURE = "record_keeping_failure"
    NON_COOPERATION = "non_cooperation"
    REPEATED_VIOLATION = "repeated_violation"


class ViolationSeverity(str, enum.Enum):
    """Severity classification for violations."""

    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


class AppealDecision(str, enum.Enum):
    """Administrative appeal decision outcomes per Article 19."""

    PENDING = "pending"
    UPHELD = "upheld"
    PARTIALLY_UPHELD = "partially_upheld"
    OVERTURNED = "overturned"
    DISMISSED = "dismissed"
    WITHDRAWN = "withdrawn"
    REFERRED = "referred"


class DocumentType(str, enum.Enum):
    """Types of documents exchanged with authorities."""

    DDS_STATEMENT = "dds_statement"
    RISK_ASSESSMENT = "risk_assessment"
    MITIGATION_REPORT = "mitigation_report"
    SUPPLY_CHAIN_MAP = "supply_chain_map"
    GEOLOCATION_DATA = "geolocation_data"
    SATELLITE_IMAGERY = "satellite_imagery"
    AUDIT_REPORT = "audit_report"
    CERTIFICATE = "certificate"
    INVOICE = "invoice"
    CUSTOMS_DECLARATION = "customs_declaration"
    LEGAL_OPINION = "legal_opinion"
    APPEAL_SUBMISSION = "appeal_submission"
    CORRECTIVE_ACTION_PLAN = "corrective_action_plan"
    EVIDENCE_PACKAGE = "evidence_package"
    OTHER = "other"


class NotificationChannel(str, enum.Enum):
    """Notification delivery channels."""

    EMAIL = "email"
    API = "api"
    PORTAL = "portal"
    SMS = "sms"
    WEBHOOK = "webhook"


class RecipientType(str, enum.Enum):
    """Types of communication recipients."""

    AUTHORITY = "authority"
    OPERATOR = "operator"
    TRADER = "trader"
    LEGAL_REPRESENTATIVE = "legal_representative"
    COMPLIANCE_OFFICER = "compliance_officer"
    AUDITOR = "auditor"
    SYSTEM = "system"


class LanguageCode(str, enum.Enum):
    """24 official EU languages per Treaty on European Union."""

    BG = "bg"  # Bulgarian
    CS = "cs"  # Czech
    DA = "da"  # Danish
    DE = "de"  # German
    EL = "el"  # Greek
    EN = "en"  # English
    ES = "es"  # Spanish
    ET = "et"  # Estonian
    FI = "fi"  # Finnish
    FR = "fr"  # French
    GA = "ga"  # Irish
    HR = "hr"  # Croatian
    HU = "hu"  # Hungarian
    IT = "it"  # Italian
    LT = "lt"  # Lithuanian
    LV = "lv"  # Latvian
    MT = "mt"  # Maltese
    NL = "nl"  # Dutch
    PL = "pl"  # Polish
    PT = "pt"  # Portuguese
    RO = "ro"  # Romanian
    SK = "sk"  # Slovak
    SL = "sl"  # Slovenian
    SV = "sv"  # Swedish


class AuthorityType(str, enum.Enum):
    """Types of competent authorities under EUDR."""

    NATIONAL_COMPETENT = "national_competent"
    CUSTOMS = "customs"
    ENVIRONMENTAL = "environmental"
    FORESTRY = "forestry"
    TRADE = "trade"
    JUDICIAL = "judicial"
    EUROPEAN_COMMISSION = "european_commission"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_ID = "GL-EUDR-ACM-040"
AGENT_VERSION = "1.0.0"

EUDR_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood",
]

DEADLINE_HOURS_MAP: Dict[str, int] = {
    "urgent": 24,
    "high": 72,
    "normal": 120,
    "low": 240,
    "routine": 360,
}


# ---------------------------------------------------------------------------
# Pydantic Models (15+)
# ---------------------------------------------------------------------------


class Authority(GreenLangBase):
    """Competent authority within an EU member state.

    Represents a specific regulatory authority responsible for
    EUDR enforcement in a given member state.
    """

    authority_id: str = Field(..., description="Unique authority identifier")
    member_state: str = Field(
        ..., min_length=2, max_length=2, description="ISO 3166-1 alpha-2 country code"
    )
    name: str = Field(..., description="Official authority name")
    authority_type: AuthorityType = AuthorityType.NATIONAL_COMPETENT
    contact_email: str = Field(default="", description="Official contact email")
    api_endpoint: str = Field(default="", description="Authority API endpoint URL")
    preferred_language: LanguageCode = LanguageCode.EN
    timezone: str = Field(default="Europe/Brussels", description="Authority timezone")
    active: bool = True

    model_config = {"frozen": False, "extra": "ignore"}


class Communication(GreenLangBase):
    """A communication thread between operator and authority.

    Represents the top-level container for all messages exchanged
    within a single regulatory communication context per EUDR
    Article 31 record-keeping requirements.
    """

    communication_id: str = Field(..., description="Unique communication identifier")
    thread_id: str = Field(default="", description="Parent thread identifier")
    operator_id: str = Field(..., description="EUDR operator identifier")
    authority_id: str = Field(..., description="Competent authority identifier")
    member_state: str = Field(
        ..., min_length=2, max_length=2, description="Member state code"
    )
    communication_type: CommunicationType
    status: CommunicationStatus = CommunicationStatus.DRAFT
    priority: CommunicationPriority = CommunicationPriority.NORMAL
    subject: str = Field(..., min_length=1, description="Communication subject")
    body: str = Field(default="", description="Communication body text")
    language: LanguageCode = LanguageCode.EN
    reference_number: str = Field(
        default="", description="Authority reference number"
    )
    dds_reference: str = Field(
        default="", description="Related DDS statement reference"
    )
    deadline: Optional[datetime] = Field(
        default=None, description="Response deadline"
    )
    responded_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    received_at: Optional[datetime] = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    document_ids: List[str] = Field(
        default_factory=list, description="Attached document IDs"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class InformationRequest(GreenLangBase):
    """Information request from competent authority per Article 17.

    Captures the specific information items requested, deadline,
    and compliance response requirements.
    """

    request_id: str = Field(..., description="Unique request identifier")
    communication_id: str = Field(..., description="Parent communication ID")
    operator_id: str = Field(..., description="Target operator identifier")
    authority_id: str = Field(..., description="Requesting authority ID")
    request_type: InformationRequestType
    items_requested: List[str] = Field(
        default_factory=list, description="Specific items requested"
    )
    dds_reference: str = Field(default="", description="Related DDS reference")
    commodity: str = Field(default="", description="EUDR commodity context")
    deadline: Optional[datetime] = None
    response_submitted: bool = False
    response_accepted: bool = False
    response_notes: str = Field(default="", description="Authority response notes")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class Inspection(GreenLangBase):
    """On-the-spot check or inspection per EUDR Article 15.

    Tracks scheduled, ongoing, and completed inspections including
    findings and follow-up actions.
    """

    inspection_id: str = Field(..., description="Unique inspection identifier")
    communication_id: str = Field(..., description="Parent communication ID")
    operator_id: str = Field(..., description="Operator being inspected")
    authority_id: str = Field(..., description="Inspecting authority ID")
    inspection_type: InspectionType
    scheduled_date: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    location: str = Field(default="", description="Inspection location")
    inspector_name: str = Field(default="", description="Lead inspector name")
    scope: str = Field(
        default="", description="Inspection scope description"
    )
    findings: List[str] = Field(
        default_factory=list, description="Inspection findings"
    )
    corrective_actions: List[str] = Field(
        default_factory=list, description="Required corrective actions"
    )
    follow_up_date: Optional[datetime] = None
    status: str = Field(default="scheduled", description="Inspection status")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class NonCompliance(GreenLangBase):
    """Non-compliance record per EUDR Article 16.

    Tracks violations, severity, penalties, and corrective action
    requirements with complete audit trail.
    """

    non_compliance_id: str = Field(
        ..., description="Unique non-compliance identifier"
    )
    communication_id: str = Field(..., description="Parent communication ID")
    operator_id: str = Field(..., description="Operator identifier")
    authority_id: str = Field(..., description="Issuing authority ID")
    violation_type: ViolationType
    severity: ViolationSeverity
    description: str = Field(..., description="Violation description")
    evidence_references: List[str] = Field(
        default_factory=list, description="Evidence document IDs"
    )
    penalty_amount: Optional[Decimal] = Field(
        default=None, ge=0, description="Penalty amount in EUR"
    )
    penalty_currency: str = Field(default="EUR", description="Penalty currency")
    corrective_actions_required: List[str] = Field(
        default_factory=list, description="Required corrective actions"
    )
    corrective_deadline: Optional[datetime] = None
    corrective_completed: bool = False
    appeal_id: Optional[str] = Field(
        default=None, description="Related appeal ID if appealed"
    )
    issued_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    resolved_at: Optional[datetime] = None
    commodity: str = Field(default="", description="Related commodity")
    dds_reference: str = Field(default="", description="Related DDS reference")
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class Appeal(GreenLangBase):
    """Administrative appeal per EUDR Article 19.

    Tracks the full lifecycle of an appeal from filing through
    decision, including extensions and referrals.
    """

    appeal_id: str = Field(..., description="Unique appeal identifier")
    communication_id: str = Field(..., description="Parent communication ID")
    non_compliance_id: str = Field(
        ..., description="Non-compliance being appealed"
    )
    operator_id: str = Field(..., description="Appealing operator")
    authority_id: str = Field(..., description="Authority receiving appeal")
    grounds: str = Field(..., description="Grounds for appeal")
    supporting_evidence: List[str] = Field(
        default_factory=list, description="Supporting document IDs"
    )
    decision: AppealDecision = AppealDecision.PENDING
    decision_reason: str = Field(default="", description="Decision reasoning")
    decision_date: Optional[datetime] = None
    filing_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    deadline: Optional[datetime] = None
    extensions_granted: int = Field(default=0, ge=0)
    penalty_suspended: bool = Field(
        default=True, description="Whether penalty is suspended during appeal"
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class Document(GreenLangBase):
    """Document exchanged between operator and authority.

    Includes encryption metadata for sensitive documents and
    integrity hashing for tamper detection.
    """

    document_id: str = Field(..., description="Unique document identifier")
    communication_id: str = Field(..., description="Parent communication ID")
    document_type: DocumentType
    title: str = Field(..., description="Document title")
    description: str = Field(default="", description="Document description")
    file_path: str = Field(default="", description="Storage path or S3 key")
    file_size_bytes: int = Field(default=0, ge=0, description="File size")
    mime_type: str = Field(default="application/pdf", description="MIME type")
    language: LanguageCode = LanguageCode.EN
    encrypted: bool = Field(default=False, description="Whether content is encrypted")
    encryption_key_id: str = Field(
        default="", description="Encryption key identifier"
    )
    integrity_hash: str = Field(
        default="", description="SHA-256 hash of file contents"
    )
    uploaded_by: str = Field(default="", description="Uploader identity")
    uploaded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class Notification(GreenLangBase):
    """Multi-channel notification record.

    Tracks delivery attempts and confirmations across email,
    API, portal, and other notification channels.
    """

    notification_id: str = Field(
        ..., description="Unique notification identifier"
    )
    communication_id: str = Field(..., description="Related communication ID")
    channel: NotificationChannel
    recipient_type: RecipientType
    recipient_id: str = Field(..., description="Recipient identifier")
    recipient_address: str = Field(
        default="", description="Delivery address (email, URL, etc.)"
    )
    subject: str = Field(default="", description="Notification subject")
    body: str = Field(default="", description="Notification body")
    language: LanguageCode = LanguageCode.EN
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    delivery_status: str = Field(
        default="pending", description="pending/sent/delivered/failed/bounced"
    )
    retry_count: int = Field(default=0, ge=0, description="Delivery retry count")
    max_retries: int = Field(default=3, ge=0, description="Maximum retries")
    error_message: str = Field(default="", description="Last error message")
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class Template(GreenLangBase):
    """Multi-language communication template.

    Supports placeholders for dynamic content injection and
    language-specific formatting rules.
    """

    template_id: str = Field(..., description="Unique template identifier")
    template_name: str = Field(..., description="Template name for lookup")
    communication_type: CommunicationType
    language: LanguageCode = LanguageCode.EN
    subject_template: str = Field(
        default="", description="Subject line template with placeholders"
    )
    body_template: str = Field(
        default="", description="Body text template with placeholders"
    )
    placeholders: List[str] = Field(
        default_factory=list, description="Available placeholder names"
    )
    version: str = Field(default="1.0", description="Template version")
    active: bool = True
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    model_config = {"frozen": False, "extra": "ignore"}


class CommunicationThread(GreenLangBase):
    """Thread grouping related communications.

    Provides chronological ordering and summary for a conversation
    between operator and authority on a specific matter.
    """

    thread_id: str = Field(..., description="Unique thread identifier")
    operator_id: str = Field(..., description="Operator identifier")
    authority_id: str = Field(..., description="Authority identifier")
    subject: str = Field(..., description="Thread subject")
    communication_ids: List[str] = Field(
        default_factory=list, description="Communications in this thread"
    )
    status: CommunicationStatus = CommunicationStatus.PENDING
    priority: CommunicationPriority = CommunicationPriority.NORMAL
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_activity_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    model_config = {"frozen": False, "extra": "ignore"}


class ResponseData(GreenLangBase):
    """Response data for a communication or information request.

    Captures the response content, attached documents, and
    submission metadata.
    """

    response_id: str = Field(..., description="Unique response identifier")
    communication_id: str = Field(
        ..., description="Communication being responded to"
    )
    responder_id: str = Field(..., description="Responder identity")
    responder_type: RecipientType = RecipientType.OPERATOR
    body: str = Field(default="", description="Response body text")
    document_ids: List[str] = Field(
        default_factory=list, description="Attached document IDs"
    )
    submitted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    accepted: Optional[bool] = None
    review_notes: str = Field(default="", description="Authority review notes")
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class DeadlineReminder(GreenLangBase):
    """Automated deadline reminder record.

    Generated 48 hours before a communication deadline to ensure
    timely response submission.
    """

    reminder_id: str = Field(..., description="Unique reminder identifier")
    communication_id: str = Field(..., description="Related communication ID")
    operator_id: str = Field(..., description="Operator to remind")
    deadline: datetime = Field(..., description="The approaching deadline")
    hours_remaining: int = Field(
        ..., description="Hours until deadline at reminder time"
    )
    reminder_sent_at: Optional[datetime] = None
    notification_ids: List[str] = Field(
        default_factory=list, description="Notification IDs for this reminder"
    )
    escalated: bool = Field(
        default=False, description="Whether this was escalated"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    model_config = {"frozen": False, "extra": "ignore"}


class ApprovalWorkflow(GreenLangBase):
    """Internal approval workflow for outgoing communications.

    Ensures that responses to authorities go through proper review
    before submission, per enterprise compliance policies.
    """

    workflow_id: str = Field(..., description="Unique workflow identifier")
    communication_id: str = Field(..., description="Related communication ID")
    initiated_by: str = Field(..., description="Workflow initiator")
    approver_id: str = Field(..., description="Designated approver")
    status: str = Field(
        default="pending_review",
        description="pending_review/approved/rejected/revision_requested",
    )
    comments: str = Field(default="", description="Approver comments")
    submitted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    reviewed_at: Optional[datetime] = None
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class HealthStatus(GreenLangBase):
    """Health check response for the Authority Communication Manager."""

    agent_id: str = AGENT_ID
    status: str = "healthy"
    version: str = AGENT_VERSION
    engines: Dict[str, str] = Field(default_factory=dict)
    database: bool = False
    redis: bool = False
    uptime_seconds: float = 0.0
    pending_communications: int = 0
    overdue_communications: int = 0
    active_inspections: int = 0
    open_appeals: int = 0

    model_config = {"frozen": False, "extra": "ignore"}
