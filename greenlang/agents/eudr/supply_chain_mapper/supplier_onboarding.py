# -*- coding: utf-8 -*-
"""
SupplierOnboardingEngine - AGENT-EUDR-001 Feature 8: Supplier Onboarding Workflow

Implements a structured multi-step supplier onboarding workflow for EU Deforestation
Regulation (EUDR) compliance. Provides secure link generation, real-time EUDR
requirement validation, mobile GPS capture support, sub-tier supplier collection,
completion tracking, automated reminders, bulk import, and auto-creation of
supply chain graph nodes and edges from completed onboarding data.

Wizard Steps (6):
    1. Company Information: Legal name, country, registration ID, contact
    2. Commodities: EUDR commodities produced/handled, HS/CN codes
    3. Production Plots: GPS coordinates, polygon boundaries (>4ha), area
    4. Certifications: FSC, RSPO, RA, ISO, organic certifications
    5. Declarations: EUDR compliance declarations, deforestation-free commitment
    6. Sub-Tier Suppliers: Upstream supplier names, commodities, countries

Integrations:
    - graph_engine.SupplyChainGraphEngine: Auto-create nodes/edges on completion
    - geolocation_linker.GeolocationLinker: Register plots during step 3
    - AGENT-DATA-008 Supplier Questionnaire Processor: Structured questionnaire handling
    - AGENT-DATA-002 Excel/CSV Normalizer: Bulk supplier import from spreadsheets

Zero-Hallucination Guarantees:
    - All validation rules are deterministic (regex, range checks, WGS84 bounds)
    - Completion percentage is pure arithmetic (fields_completed / total_fields)
    - Token generation uses cryptographic UUID4 + HMAC-SHA256, no ML involved
    - GPS coordinate validation uses WGS84 bounds and decimal precision checks
    - Polygon requirement (>4 ha) enforced per EUDR Article 9(1)(d)
    - SHA-256 provenance hashes on all onboarding mutations

Performance Targets:
    - Onboarding completion: 70%+ within 14 days of invitation
    - Token validation: <5ms per lookup
    - Bulk import: 10,000 suppliers per batch
    - GPS coordinate capture: HTML5 Geolocation API compatible

Non-Functional:
    - Multi-language: EN, FR, DE, ES, PT, ID
    - Mobile-friendly: GPS capture on iOS/Android browsers
    - Secure: Tokens expire after configurable TTL (default 30 days)

Example:
    >>> engine = SupplierOnboardingEngine()
    >>> session = engine.create_onboarding_session(
    ...     operator_id="op-001",
    ...     graph_id="graph-001",
    ...     supplier_name="Fazenda Verde",
    ...     supplier_email="verde@example.com",
    ...     commodity="soya",
    ... )
    >>> assert session.token is not None
    >>> assert session.completion_pct == 0.0

    >>> engine.submit_step(session.session_id, "company_info", {
    ...     "legal_name": "Fazenda Verde Ltda",
    ...     "country_code": "BR",
    ...     "registration_id": "BR-12345678",
    ...     "contact_name": "Maria Silva",
    ...     "contact_email": "maria@fazendaverde.com.br",
    ...     "contact_phone": "+5511999999999",
    ... })
    >>> updated = engine.get_session(session.session_id)
    >>> assert updated.completion_pct > 0.0

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-001 (Feature 8: Supplier Onboarding and Discovery Workflow)
Agent ID: GL-EUDR-SCM-001
Regulation: EU 2023/1115 (EUDR), Articles 4(2), 9, 10
Status: Production Ready
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import logging
import re
import secrets
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other serializable).

    Returns:
        SHA-256 hex digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _generate_id(prefix: str) -> str:
    """Generate a prefixed unique identifier.

    Args:
        prefix: String prefix for the identifier.

    Returns:
        Formatted identifier string like ``prefix-uuid4_hex[:12]``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Supported onboarding languages per PRD NFR.
SUPPORTED_LANGUAGES: Tuple[str, ...] = ("en", "fr", "de", "es", "pt", "id")

#: Default token expiry in days.
DEFAULT_TOKEN_EXPIRY_DAYS: int = 30

#: Default reminder intervals in days from invitation.
DEFAULT_REMINDER_DAYS: Tuple[int, ...] = (3, 7, 10, 14)

#: Maximum suppliers per bulk import batch.
MAX_BULK_IMPORT_BATCH: int = 10_000

#: Minimum GPS coordinate decimal precision for EUDR compliance.
MIN_COORDINATE_PRECISION: int = 6

#: EUDR polygon area threshold in hectares (>4ha requires polygon).
POLYGON_AREA_THRESHOLD_HA: float = 4.0

#: HMAC signing key prefix for token generation.
_TOKEN_HMAC_PREFIX: str = "GL-EUDR-ONBOARD"

#: WGS84 coordinate bounds.
LAT_MIN: float = -90.0
LAT_MAX: float = 90.0
LON_MIN: float = -180.0
LON_MAX: float = 180.0

#: EUDR deforestation cutoff date.
EUDR_DEFORESTATION_CUTOFF: str = "2020-12-31"

#: CSV/Excel required columns for bulk import.
BULK_IMPORT_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "supplier_name",
    "country_code",
    "commodity",
)

#: All fields by step for completion tracking.
STEP_FIELDS: Dict[str, List[str]] = {
    "company_info": [
        "legal_name",
        "country_code",
        "registration_id",
        "contact_name",
        "contact_email",
        "contact_phone",
    ],
    "commodities": [
        "commodities",
        "hs_codes",
    ],
    "plots": [
        "plots",
    ],
    "certifications": [
        "certifications",
    ],
    "declarations": [
        "deforestation_free_declaration",
        "legality_declaration",
    ],
    "sub_tier_suppliers": [
        "sub_tier_suppliers",
    ],
}

#: Ordered wizard steps.
WIZARD_STEPS: Tuple[str, ...] = (
    "company_info",
    "commodities",
    "plots",
    "certifications",
    "declarations",
    "sub_tier_suppliers",
)

#: Total number of required field groups across all steps.
TOTAL_FIELD_GROUPS: int = sum(len(fields) for fields in STEP_FIELDS.values())


# ===========================================================================
# Enumerations
# ===========================================================================


class OnboardingStatus(str, Enum):
    """Status of a supplier onboarding session.

    INVITED: Session created, invitation sent, no data submitted yet.
    IN_PROGRESS: Supplier has started submitting data.
    COMPLETED: All wizard steps submitted and validated.
    EXPIRED: Token expired before completion.
    CANCELLED: Session manually cancelled by operator.
    """

    INVITED = "invited"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ReminderType(str, Enum):
    """Type of automated reminder.

    EMAIL: Email notification to supplier contact.
    WEBHOOK: Webhook callback to external system.
    IN_APP: In-application notification to operator dashboard.
    """

    EMAIL = "email"
    WEBHOOK = "webhook"
    IN_APP = "in_app"


class ReminderStatus(str, Enum):
    """Delivery status of a reminder.

    PENDING: Reminder scheduled but not yet sent.
    SENT: Reminder successfully delivered.
    FAILED: Delivery attempt failed.
    SKIPPED: Reminder skipped (session completed or cancelled).
    """

    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    SKIPPED = "skipped"


class ValidationSeverity(str, Enum):
    """Severity level for validation findings.

    ERROR: Blocks submission; must be fixed before proceeding.
    WARNING: Does not block but flags potential compliance issue.
    INFO: Informational feedback for the supplier.
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class BulkImportStatus(str, Enum):
    """Status of a bulk import operation.

    PENDING: Import queued for processing.
    PROCESSING: Import currently being processed.
    COMPLETED: All rows processed successfully.
    PARTIAL: Some rows succeeded, some failed.
    FAILED: Import failed entirely.
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


# ===========================================================================
# Protocol Interfaces
# ===========================================================================


class GraphEngineProtocol(Protocol):
    """Protocol for SupplyChainGraphEngine integration.

    Decouples onboarding from the concrete graph engine implementation
    to support testing with mocks and alternative storage backends.
    """

    async def add_node(
        self,
        graph_id: str,
        node_type: Any,
        operator_name: str,
        country_code: str,
        **kwargs: Any,
    ) -> str:
        """Add a node to the supply chain graph."""
        ...

    async def add_edge(
        self,
        graph_id: str,
        source_node_id: str,
        target_node_id: str,
        commodity: str,
        quantity: Decimal,
        **kwargs: Any,
    ) -> str:
        """Add an edge to the supply chain graph."""
        ...


class GeolocationLinkerProtocol(Protocol):
    """Protocol for GeolocationLinker integration.

    Decouples onboarding from the concrete geolocation linker
    implementation for testability.
    """

    def link_producer_to_plot(
        self,
        producer_node_id: str,
        plot_id: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        polygon_coordinates: Optional[List[List[float]]] = None,
        area_hectares: Optional[float] = None,
        commodity: Optional[str] = None,
        country_code: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Link a producer node to a plot with geolocation data."""
        ...


class NotificationServiceProtocol(Protocol):
    """Protocol for email/webhook notification delivery.

    Decouples onboarding reminders from the concrete notification
    transport to support testing and pluggable delivery backends.
    """

    def send_email(
        self,
        to_email: str,
        subject: str,
        body: str,
        language: str,
    ) -> bool:
        """Send an email notification. Returns True on success."""
        ...

    def send_webhook(
        self,
        url: str,
        payload: Dict[str, Any],
    ) -> bool:
        """Send a webhook notification. Returns True on success."""
        ...


class QuestionnaireProcessorProtocol(Protocol):
    """Protocol for AGENT-DATA-008 Supplier Questionnaire Processor.

    Processes structured questionnaire responses into standardized
    supply chain data.
    """

    def process_response(
        self,
        questionnaire_id: str,
        responses: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process questionnaire responses. Returns normalized data."""
        ...


# ===========================================================================
# Pydantic Data Models
# ===========================================================================


class PlotData(BaseModel):
    """Production plot geolocation data captured during onboarding.

    Attributes:
        plot_id: Unique identifier for this plot.
        latitude: GPS latitude in WGS84 decimal degrees.
        longitude: GPS longitude in WGS84 decimal degrees.
        polygon_coordinates: Optional polygon ring coordinates for plots >4ha.
        area_hectares: Plot area in hectares.
        commodity: EUDR commodity produced on this plot.
        country_code: ISO 3166-1 alpha-2 country code.
        capture_method: How coordinates were captured (gps, manual, upload).
        capture_accuracy_m: GPS capture accuracy in meters (from HTML5 API).
    """

    model_config = ConfigDict(from_attributes=True)

    plot_id: str = Field(
        default_factory=lambda: _generate_id("PLOT"),
        description="Unique identifier for this plot",
    )
    latitude: float = Field(
        ...,
        ge=LAT_MIN,
        le=LAT_MAX,
        description="GPS latitude in WGS84 decimal degrees",
    )
    longitude: float = Field(
        ...,
        ge=LON_MIN,
        le=LON_MAX,
        description="GPS longitude in WGS84 decimal degrees",
    )
    polygon_coordinates: Optional[List[List[float]]] = Field(
        None,
        description="Polygon ring coordinates [[lon, lat], ...] for plots >4ha",
    )
    area_hectares: float = Field(
        ...,
        gt=0,
        description="Plot area in hectares",
    )
    commodity: str = Field(
        ...,
        description="EUDR commodity produced on this plot",
    )
    country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    capture_method: str = Field(
        default="manual",
        description="Coordinate capture method: gps, manual, upload",
    )
    capture_accuracy_m: Optional[float] = Field(
        None,
        ge=0,
        description="GPS capture accuracy in meters (from HTML5 Geolocation API)",
    )

    @field_validator("country_code")
    @classmethod
    def normalize_country_code(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        return v.upper().strip()

    @field_validator("capture_method")
    @classmethod
    def validate_capture_method(cls, v: str) -> str:
        """Validate capture method is one of the allowed values."""
        allowed = {"gps", "manual", "upload"}
        if v.lower().strip() not in allowed:
            raise ValueError(
                f"capture_method must be one of {sorted(allowed)}, got '{v}'"
            )
        return v.lower().strip()


class CertificationData(BaseModel):
    """Certification record captured during onboarding.

    Attributes:
        certification_type: Type of certification (FSC, RSPO, RA, ISO, organic).
        certificate_number: Certificate number or registration ID.
        issuing_body: Name of the certifying body.
        valid_from: Start date of certification validity.
        valid_to: Expiry date of certification.
        scope: Optional scope description.
    """

    model_config = ConfigDict(from_attributes=True)

    certification_type: str = Field(
        ...,
        description="Certification type (FSC, RSPO, RA, ISO, organic)",
    )
    certificate_number: str = Field(
        ...,
        description="Certificate number or registration ID",
    )
    issuing_body: str = Field(
        ...,
        description="Name of the certifying body",
    )
    valid_from: Optional[datetime] = Field(
        None,
        description="Start date of certification validity",
    )
    valid_to: Optional[datetime] = Field(
        None,
        description="Expiry date of certification",
    )
    scope: Optional[str] = Field(
        None,
        description="Scope description of the certification",
    )


class SubTierSupplierData(BaseModel):
    """Sub-tier supplier information collected during onboarding.

    Attributes:
        supplier_name: Legal name of the sub-tier supplier.
        country_code: ISO 3166-1 alpha-2 country code.
        commodities: EUDR commodities supplied.
        relationship_type: How this supplier relates (direct, indirect).
        contact_email: Optional contact email for recursive onboarding.
        estimated_volume_pct: Estimated percentage of total supply volume.
    """

    model_config = ConfigDict(from_attributes=True)

    supplier_name: str = Field(
        ...,
        min_length=1,
        description="Legal name of the sub-tier supplier",
    )
    country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    commodities: List[str] = Field(
        default_factory=list,
        description="EUDR commodities supplied",
    )
    relationship_type: str = Field(
        default="direct",
        description="Relationship type: direct or indirect",
    )
    contact_email: Optional[str] = Field(
        None,
        description="Contact email for recursive onboarding",
    )
    estimated_volume_pct: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Estimated percentage of total supply volume",
    )

    @field_validator("country_code")
    @classmethod
    def normalize_country_code(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        return v.upper().strip()


class OnboardingStepResult(BaseModel):
    """Result of validating and submitting a single onboarding step.

    Attributes:
        step_name: Name of the wizard step.
        is_valid: Whether the step data passed all validation rules.
        errors: List of validation errors (severity=ERROR).
        warnings: List of validation warnings (severity=WARNING).
        info_messages: List of informational messages.
        fields_completed: Number of fields completed in this step.
        fields_total: Total number of fields in this step.
        submitted_at: Timestamp of submission.
        provenance_hash: SHA-256 hash of the step data.
    """

    model_config = ConfigDict(from_attributes=True)

    step_name: str = Field(..., description="Name of the wizard step")
    is_valid: bool = Field(default=True, description="Whether step data is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    info_messages: List[str] = Field(
        default_factory=list, description="Informational messages"
    )
    fields_completed: int = Field(default=0, ge=0)
    fields_total: int = Field(default=0, ge=0)
    submitted_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


class ReminderRecord(BaseModel):
    """Record of a scheduled or sent reminder.

    Attributes:
        reminder_id: Unique identifier for this reminder.
        session_id: Onboarding session this reminder belongs to.
        reminder_type: Type of notification (email, webhook, in_app).
        status: Delivery status.
        scheduled_at: When the reminder is/was scheduled.
        sent_at: When the reminder was actually sent.
        target_email: Email address for email reminders.
        target_url: Webhook URL for webhook reminders.
        attempt_count: Number of delivery attempts.
        error_message: Error message if delivery failed.
    """

    model_config = ConfigDict(from_attributes=True)

    reminder_id: str = Field(
        default_factory=lambda: _generate_id("RMD"),
        description="Unique identifier for this reminder",
    )
    session_id: str = Field(..., description="Onboarding session ID")
    reminder_type: ReminderType = Field(
        default=ReminderType.EMAIL,
        description="Notification type",
    )
    status: ReminderStatus = Field(
        default=ReminderStatus.PENDING,
        description="Delivery status",
    )
    scheduled_at: datetime = Field(
        default_factory=_utcnow,
        description="When the reminder is scheduled",
    )
    sent_at: Optional[datetime] = Field(None, description="When actually sent")
    target_email: Optional[str] = Field(None, description="Email target")
    target_url: Optional[str] = Field(None, description="Webhook URL target")
    attempt_count: int = Field(default=0, ge=0, description="Delivery attempts")
    error_message: Optional[str] = Field(None, description="Error on failure")


class OnboardingSession(BaseModel):
    """Complete onboarding session state for a single supplier.

    Tracks all wizard steps, submitted data, completion percentage,
    reminders, and token lifecycle for one supplier onboarding flow.

    Attributes:
        session_id: Unique session identifier.
        operator_id: ID of the operator who initiated onboarding.
        graph_id: Target supply chain graph ID.
        supplier_name: Display name of the supplier being onboarded.
        supplier_email: Contact email for the supplier.
        commodity: Primary EUDR commodity for this supplier.
        token: Secure unique token for the onboarding link.
        token_expires_at: Token expiry timestamp.
        status: Current session status.
        language: Preferred language code.
        steps_completed: Set of completed step names.
        step_data: Raw data submitted per step.
        step_results: Validation results per step.
        completion_pct: Percentage of total fields completed (0-100).
        company_info: Validated company information.
        commodities_data: Validated commodity data.
        plots_data: Validated plot geolocation data.
        certifications_data: Validated certification data.
        declarations_data: Validated declaration data.
        sub_tier_suppliers_data: Validated sub-tier supplier data.
        reminders: Scheduled/sent reminders.
        created_node_ids: Graph node IDs created from this session.
        created_edge_ids: Graph edge IDs created from this session.
        created_at: Session creation timestamp.
        updated_at: Last modification timestamp.
        completed_at: Session completion timestamp.
        provenance_hash: SHA-256 hash of the full session state.
    """

    model_config = ConfigDict(from_attributes=True)

    session_id: str = Field(
        default_factory=lambda: _generate_id("ONB"),
        description="Unique session identifier",
    )
    operator_id: str = Field(..., description="Operator who initiated onboarding")
    graph_id: str = Field(..., description="Target supply chain graph ID")
    supplier_name: str = Field(..., description="Supplier display name")
    supplier_email: str = Field(..., description="Supplier contact email")
    commodity: str = Field(..., description="Primary EUDR commodity")
    token: str = Field(default="", description="Secure onboarding link token")
    token_expires_at: Optional[datetime] = Field(
        None, description="Token expiry timestamp"
    )
    status: OnboardingStatus = Field(
        default=OnboardingStatus.INVITED,
        description="Current session status",
    )
    language: str = Field(default="en", description="Preferred language code")
    steps_completed: List[str] = Field(
        default_factory=list, description="Completed step names"
    )
    step_data: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Raw data per step"
    )
    step_results: Dict[str, OnboardingStepResult] = Field(
        default_factory=dict, description="Validation results per step"
    )
    completion_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Completion percentage"
    )
    company_info: Dict[str, Any] = Field(
        default_factory=dict, description="Validated company info"
    )
    commodities_data: List[str] = Field(
        default_factory=list, description="Validated commodity list"
    )
    plots_data: List[PlotData] = Field(
        default_factory=list, description="Validated plot data"
    )
    certifications_data: List[CertificationData] = Field(
        default_factory=list, description="Validated certifications"
    )
    declarations_data: Dict[str, Any] = Field(
        default_factory=dict, description="Validated declarations"
    )
    sub_tier_suppliers_data: List[SubTierSupplierData] = Field(
        default_factory=list, description="Validated sub-tier suppliers"
    )
    reminders: List[ReminderRecord] = Field(
        default_factory=list, description="Reminder records"
    )
    created_node_ids: List[str] = Field(
        default_factory=list, description="Graph node IDs created"
    )
    created_edge_ids: List[str] = Field(
        default_factory=list, description="Graph edge IDs created"
    )
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 session hash")

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Validate language is one of the supported languages."""
        v = v.lower().strip()
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"language must be one of {list(SUPPORTED_LANGUAGES)}, got '{v}'"
            )
        return v


class BulkImportResult(BaseModel):
    """Result of a bulk supplier import operation.

    Attributes:
        import_id: Unique identifier for this import.
        status: Overall import status.
        total_rows: Total number of rows in the input.
        rows_succeeded: Number of rows successfully imported.
        rows_failed: Number of rows that failed validation.
        sessions_created: List of session IDs created.
        errors: Per-row error details (row_number -> error message).
        processing_time_ms: Total processing time in milliseconds.
        provenance_hash: SHA-256 hash of the import result.
    """

    model_config = ConfigDict(from_attributes=True)

    import_id: str = Field(
        default_factory=lambda: _generate_id("IMP"),
        description="Unique import identifier",
    )
    status: BulkImportStatus = Field(
        default=BulkImportStatus.PENDING,
        description="Overall import status",
    )
    total_rows: int = Field(default=0, ge=0)
    rows_succeeded: int = Field(default=0, ge=0)
    rows_failed: int = Field(default=0, ge=0)
    sessions_created: List[str] = Field(
        default_factory=list, description="Session IDs created"
    )
    errors: Dict[int, str] = Field(
        default_factory=dict, description="Row number -> error"
    )
    processing_time_ms: float = Field(default=0.0, ge=0.0)
    provenance_hash: str = Field(default="")


class OnboardingMetrics(BaseModel):
    """Aggregated metrics for onboarding operations.

    Attributes:
        total_sessions: Total sessions created.
        sessions_by_status: Count by status.
        average_completion_pct: Mean completion percentage.
        median_completion_days: Median days to completion.
        completion_within_14_days_pct: Percentage completed within 14 days.
        reminders_sent: Total reminders sent.
        bulk_imports: Total bulk imports processed.
    """

    model_config = ConfigDict(from_attributes=True)

    total_sessions: int = Field(default=0, ge=0)
    sessions_by_status: Dict[str, int] = Field(default_factory=dict)
    average_completion_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    median_completion_days: Optional[float] = Field(None)
    completion_within_14_days_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    reminders_sent: int = Field(default=0, ge=0)
    bulk_imports: int = Field(default=0, ge=0)


# ===========================================================================
# SupplierOnboardingEngine
# ===========================================================================


class SupplierOnboardingEngine:
    """Supplier onboarding workflow engine for EUDR compliance.

    Orchestrates the multi-step supplier onboarding process from invitation
    through data collection, validation, and graph node creation. Provides
    secure token-based access, real-time EUDR requirement validation,
    mobile GPS capture support, automated reminders, and bulk import.

    Attributes:
        _sessions: In-memory session store indexed by session_id.
        _token_index: Token -> session_id reverse index for fast lookup.
        _graph_engine: Optional graph engine for auto-creating nodes/edges.
        _geo_linker: Optional geolocation linker for plot registration.
        _notification_service: Optional notification service for reminders.
        _questionnaire_processor: Optional AGENT-DATA-008 processor.
        _signing_key: HMAC signing key for token generation.
        _token_expiry_days: Token TTL in days.
        _reminder_days: Scheduled reminder intervals.

    Example:
        >>> engine = SupplierOnboardingEngine()
        >>> session = engine.create_onboarding_session(
        ...     operator_id="op-001",
        ...     graph_id="graph-001",
        ...     supplier_name="Fazenda Verde",
        ...     supplier_email="verde@example.com",
        ...     commodity="soya",
        ... )
        >>> assert session.status == OnboardingStatus.INVITED
    """

    def __init__(
        self,
        graph_engine: Optional[GraphEngineProtocol] = None,
        geo_linker: Optional[GeolocationLinkerProtocol] = None,
        notification_service: Optional[NotificationServiceProtocol] = None,
        questionnaire_processor: Optional[QuestionnaireProcessorProtocol] = None,
        signing_key: Optional[str] = None,
        token_expiry_days: int = DEFAULT_TOKEN_EXPIRY_DAYS,
        reminder_days: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """Initialize the SupplierOnboardingEngine.

        Args:
            graph_engine: Optional graph engine for auto node/edge creation.
            geo_linker: Optional geolocation linker for plot registration.
            notification_service: Optional notification service.
            questionnaire_processor: Optional AGENT-DATA-008 processor.
            signing_key: HMAC key for token signing. Auto-generated if None.
            token_expiry_days: Token TTL in days (default 30).
            reminder_days: Reminder schedule in days from invitation.
        """
        self._sessions: Dict[str, OnboardingSession] = {}
        self._token_index: Dict[str, str] = {}
        self._graph_engine = graph_engine
        self._geo_linker = geo_linker
        self._notification_service = notification_service
        self._questionnaire_processor = questionnaire_processor
        self._signing_key = signing_key or secrets.token_hex(32)
        self._token_expiry_days = token_expiry_days
        self._reminder_days = reminder_days or DEFAULT_REMINDER_DAYS
        self._bulk_import_count: int = 0

        logger.info(
            "SupplierOnboardingEngine initialized: "
            "token_expiry=%dd, reminder_schedule=%s, "
            "graph_engine=%s, geo_linker=%s, notifications=%s",
            self._token_expiry_days,
            self._reminder_days,
            "attached" if graph_engine else "none",
            "attached" if geo_linker else "none",
            "attached" if notification_service else "none",
        )

    # ------------------------------------------------------------------
    # Token Management
    # ------------------------------------------------------------------

    def _generate_token(self, session_id: str) -> str:
        """Generate a secure, unique onboarding token.

        Uses UUID4 for uniqueness and HMAC-SHA256 for tamper resistance.
        The token combines a random component with a signed session binding
        to prevent token reuse across sessions.

        Args:
            session_id: Session ID to bind the token to.

        Returns:
            Hexadecimal token string (64 characters).
        """
        random_part = secrets.token_hex(16)
        message = f"{_TOKEN_HMAC_PREFIX}:{session_id}:{random_part}"
        signature = hmac.new(
            self._signing_key.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()
        return signature

    def _is_token_valid(self, session: OnboardingSession) -> bool:
        """Check whether a session token is still valid (not expired).

        Args:
            session: The onboarding session to check.

        Returns:
            True if the token has not expired, False otherwise.
        """
        if not session.token:
            return False
        if session.token_expires_at is None:
            return False
        return _utcnow() <= session.token_expires_at

    def validate_token(self, token: str) -> Optional[OnboardingSession]:
        """Validate an onboarding token and return the associated session.

        Performs token lookup, expiry check, and session status validation.
        Returns None if the token is invalid, expired, or the session is
        in a terminal state.

        Args:
            token: The onboarding token to validate.

        Returns:
            The OnboardingSession if valid, None otherwise.
        """
        session_id = self._token_index.get(token)
        if session_id is None:
            logger.warning("Token validation failed: unknown token")
            return None

        session = self._sessions.get(session_id)
        if session is None:
            logger.warning("Token validation failed: session %s not found", session_id)
            return None

        if not self._is_token_valid(session):
            logger.warning(
                "Token validation failed: token expired for session %s", session_id
            )
            session.status = OnboardingStatus.EXPIRED
            session.updated_at = _utcnow()
            return None

        if session.status in (OnboardingStatus.CANCELLED, OnboardingStatus.EXPIRED):
            logger.warning(
                "Token validation failed: session %s is %s",
                session_id,
                session.status.value,
            )
            return None

        return session

    # ------------------------------------------------------------------
    # Session Lifecycle
    # ------------------------------------------------------------------

    def create_onboarding_session(
        self,
        operator_id: str,
        graph_id: str,
        supplier_name: str,
        supplier_email: str,
        commodity: str,
        language: str = "en",
        token_expiry_days: Optional[int] = None,
    ) -> OnboardingSession:
        """Create a new supplier onboarding session with a secure token.

        Generates a unique session with HMAC-signed token, schedules
        automated reminders, and prepares the wizard state machine.

        Args:
            operator_id: ID of the operator initiating onboarding.
            graph_id: Target supply chain graph.
            supplier_name: Legal name of the supplier.
            supplier_email: Contact email for the supplier.
            commodity: Primary EUDR commodity.
            language: Preferred language code (default 'en').
            token_expiry_days: Custom token TTL (overrides default).

        Returns:
            Created OnboardingSession with token and scheduled reminders.

        Raises:
            ValueError: If required fields are empty or invalid.
        """
        start_time = time.monotonic()

        # Validate inputs
        if not operator_id or not operator_id.strip():
            raise ValueError("operator_id must be non-empty")
        if not graph_id or not graph_id.strip():
            raise ValueError("graph_id must be non-empty")
        if not supplier_name or not supplier_name.strip():
            raise ValueError("supplier_name must be non-empty")
        if not supplier_email or not supplier_email.strip():
            raise ValueError("supplier_email must be non-empty")
        if not commodity or not commodity.strip():
            raise ValueError("commodity must be non-empty")

        # Validate email format (basic)
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", supplier_email.strip()):
            raise ValueError(f"Invalid email format: {supplier_email}")

        expiry_days = token_expiry_days or self._token_expiry_days

        session = OnboardingSession(
            operator_id=operator_id.strip(),
            graph_id=graph_id.strip(),
            supplier_name=supplier_name.strip(),
            supplier_email=supplier_email.strip(),
            commodity=commodity.strip().lower(),
            language=language.lower().strip(),
            token_expires_at=_utcnow() + timedelta(days=expiry_days),
        )

        # Generate secure token
        token = self._generate_token(session.session_id)
        session.token = token

        # Schedule reminders
        session.reminders = self._schedule_reminders(session)

        # Compute provenance hash
        session.provenance_hash = _compute_hash(session)

        # Store session and token index
        self._sessions[session.session_id] = session
        self._token_index[token] = session.session_id

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Onboarding session created: session=%s, supplier=%s, "
            "commodity=%s, token_expires=%s, reminders=%d, elapsed=%.1fms",
            session.session_id,
            supplier_name,
            commodity,
            session.token_expires_at,
            len(session.reminders),
            elapsed_ms,
        )

        return session

    def get_session(self, session_id: str) -> Optional[OnboardingSession]:
        """Retrieve an onboarding session by ID.

        Args:
            session_id: The session identifier.

        Returns:
            The OnboardingSession if found, None otherwise.
        """
        return self._sessions.get(session_id)

    def get_session_by_token(self, token: str) -> Optional[OnboardingSession]:
        """Retrieve an onboarding session by its token.

        Performs full token validation including expiry check.

        Args:
            token: The onboarding token.

        Returns:
            The OnboardingSession if valid, None otherwise.
        """
        return self.validate_token(token)

    def cancel_session(self, session_id: str) -> Optional[OnboardingSession]:
        """Cancel an onboarding session.

        Marks the session as cancelled, skips pending reminders, and
        removes the token from the index.

        Args:
            session_id: The session to cancel.

        Returns:
            Updated session, or None if not found.
        """
        session = self._sessions.get(session_id)
        if session is None:
            logger.warning("Cancel failed: session %s not found", session_id)
            return None

        session.status = OnboardingStatus.CANCELLED
        session.updated_at = _utcnow()

        # Skip pending reminders
        for reminder in session.reminders:
            if reminder.status == ReminderStatus.PENDING:
                reminder.status = ReminderStatus.SKIPPED

        # Remove token from index
        if session.token in self._token_index:
            del self._token_index[session.token]

        session.provenance_hash = _compute_hash(session)

        logger.info("Onboarding session cancelled: session=%s", session_id)
        return session

    def list_sessions(
        self,
        operator_id: Optional[str] = None,
        status: Optional[OnboardingStatus] = None,
        commodity: Optional[str] = None,
    ) -> List[OnboardingSession]:
        """List onboarding sessions with optional filters.

        Args:
            operator_id: Filter by operator ID.
            status: Filter by session status.
            commodity: Filter by commodity.

        Returns:
            Filtered list of OnboardingSession objects.
        """
        results: List[OnboardingSession] = []
        for session in self._sessions.values():
            if operator_id and session.operator_id != operator_id:
                continue
            if status and session.status != status:
                continue
            if commodity and session.commodity != commodity.lower():
                continue
            results.append(session)
        return results

    # ------------------------------------------------------------------
    # Wizard Step Submission
    # ------------------------------------------------------------------

    def submit_step(
        self,
        session_id: str,
        step_name: str,
        data: Dict[str, Any],
    ) -> OnboardingStepResult:
        """Submit data for a single onboarding wizard step.

        Validates the data against EUDR requirements, stores the result,
        updates the session completion percentage, and transitions the
        session status from INVITED to IN_PROGRESS on first submission.

        Args:
            session_id: Target onboarding session.
            step_name: Name of the wizard step to submit.
            data: Key-value data for the step.

        Returns:
            OnboardingStepResult with validation details.

        Raises:
            ValueError: If session_id or step_name is invalid.
        """
        start_time = time.monotonic()

        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")

        if step_name not in WIZARD_STEPS:
            raise ValueError(
                f"Invalid step name '{step_name}'. "
                f"Valid steps: {list(WIZARD_STEPS)}"
            )

        if session.status in (OnboardingStatus.COMPLETED, OnboardingStatus.CANCELLED,
                              OnboardingStatus.EXPIRED):
            raise ValueError(
                f"Cannot submit to session in '{session.status.value}' state"
            )

        # Check token validity
        if not self._is_token_valid(session):
            session.status = OnboardingStatus.EXPIRED
            session.updated_at = _utcnow()
            raise ValueError("Session token has expired")

        # Validate the step data
        result = self._validate_step(step_name, data, session)

        # Store step data and result
        session.step_data[step_name] = data
        session.step_results[step_name] = result

        # Apply validated data to session if valid
        if result.is_valid:
            self._apply_step_data(session, step_name, data)
            if step_name not in session.steps_completed:
                session.steps_completed.append(step_name)

        # Transition status on first submission
        if session.status == OnboardingStatus.INVITED:
            session.status = OnboardingStatus.IN_PROGRESS

        # Recalculate completion percentage
        session.completion_pct = self._calculate_completion(session)

        # Check if all steps are complete
        if set(session.steps_completed) == set(WIZARD_STEPS):
            session.status = OnboardingStatus.COMPLETED
            session.completed_at = _utcnow()
            # Skip remaining reminders
            for reminder in session.reminders:
                if reminder.status == ReminderStatus.PENDING:
                    reminder.status = ReminderStatus.SKIPPED

        session.updated_at = _utcnow()
        session.provenance_hash = _compute_hash(session)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Step submitted: session=%s, step=%s, valid=%s, "
            "completion=%.1f%%, errors=%d, elapsed=%.1fms",
            session_id,
            step_name,
            result.is_valid,
            session.completion_pct,
            len(result.errors),
            elapsed_ms,
        )

        return result

    def get_next_step(self, session_id: str) -> Optional[str]:
        """Get the next incomplete step in the wizard sequence.

        Args:
            session_id: Target onboarding session.

        Returns:
            Name of the next step, or None if all steps are complete.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return None

        for step in WIZARD_STEPS:
            if step not in session.steps_completed:
                return step
        return None

    # ------------------------------------------------------------------
    # Step Validation
    # ------------------------------------------------------------------

    def _validate_step(
        self,
        step_name: str,
        data: Dict[str, Any],
        session: OnboardingSession,
    ) -> OnboardingStepResult:
        """Validate step data against EUDR requirements.

        Routes to step-specific validators and returns a unified result.

        Args:
            step_name: Wizard step name.
            data: Step data to validate.
            session: Current onboarding session for context.

        Returns:
            OnboardingStepResult with errors, warnings, and field counts.
        """
        validators: Dict[str, Callable[..., OnboardingStepResult]] = {
            "company_info": self._validate_company_info,
            "commodities": self._validate_commodities,
            "plots": self._validate_plots,
            "certifications": self._validate_certifications,
            "declarations": self._validate_declarations,
            "sub_tier_suppliers": self._validate_sub_tier_suppliers,
        }

        validator = validators.get(step_name)
        if validator is None:
            return OnboardingStepResult(
                step_name=step_name,
                is_valid=False,
                errors=[f"No validator for step '{step_name}'"],
            )

        return validator(data, session)

    def _validate_company_info(
        self,
        data: Dict[str, Any],
        session: OnboardingSession,
    ) -> OnboardingStepResult:
        """Validate company information step.

        Required fields: legal_name, country_code, registration_id,
        contact_name, contact_email, contact_phone.

        Args:
            data: Company info data.
            session: Current session for context.

        Returns:
            Step validation result.
        """
        errors: List[str] = []
        warnings: List[str] = []
        info_msgs: List[str] = []
        completed = 0
        total = len(STEP_FIELDS["company_info"])

        # legal_name
        legal_name = data.get("legal_name", "")
        if not legal_name or not str(legal_name).strip():
            errors.append("legal_name is required")
        else:
            completed += 1

        # country_code
        country_code = data.get("country_code", "")
        if not country_code or not str(country_code).strip():
            errors.append("country_code is required")
        elif len(str(country_code).strip()) != 2:
            errors.append("country_code must be a 2-letter ISO 3166-1 alpha-2 code")
        else:
            completed += 1

        # registration_id
        reg_id = data.get("registration_id", "")
        if not reg_id or not str(reg_id).strip():
            errors.append("registration_id is required")
        else:
            completed += 1

        # contact_name
        contact_name = data.get("contact_name", "")
        if not contact_name or not str(contact_name).strip():
            errors.append("contact_name is required")
        else:
            completed += 1

        # contact_email
        contact_email = data.get("contact_email", "")
        if not contact_email or not str(contact_email).strip():
            errors.append("contact_email is required")
        elif not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", str(contact_email).strip()):
            errors.append(f"Invalid email format: {contact_email}")
        else:
            completed += 1

        # contact_phone
        contact_phone = data.get("contact_phone", "")
        if not contact_phone or not str(contact_phone).strip():
            warnings.append("contact_phone is recommended but not required")
        else:
            completed += 1

        return OnboardingStepResult(
            step_name="company_info",
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            info_messages=info_msgs,
            fields_completed=completed,
            fields_total=total,
            provenance_hash=_compute_hash(data),
        )

    def _validate_commodities(
        self,
        data: Dict[str, Any],
        session: OnboardingSession,
    ) -> OnboardingStepResult:
        """Validate commodities step.

        Required: at least one commodity. Optional: HS codes.

        Args:
            data: Commodities data.
            session: Current session.

        Returns:
            Step validation result.
        """
        errors: List[str] = []
        warnings: List[str] = []
        completed = 0
        total = len(STEP_FIELDS["commodities"])

        valid_commodities = {
            "cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood",
            "beef", "leather", "chocolate", "palm_oil", "natural_rubber",
            "tyres", "soybean_oil", "soybean_meal", "timber", "furniture",
            "paper", "charcoal",
        }

        commodities = data.get("commodities", [])
        if not commodities or not isinstance(commodities, list) or len(commodities) == 0:
            errors.append("At least one EUDR commodity is required")
        else:
            invalid = [c for c in commodities if str(c).lower() not in valid_commodities]
            if invalid:
                errors.append(f"Invalid commodities: {invalid}")
            else:
                completed += 1

        hs_codes = data.get("hs_codes", [])
        if hs_codes and isinstance(hs_codes, list) and len(hs_codes) > 0:
            completed += 1
        else:
            warnings.append("HS codes are recommended for customs compliance")

        return OnboardingStepResult(
            step_name="commodities",
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            fields_completed=completed,
            fields_total=total,
            provenance_hash=_compute_hash(data),
        )

    def _validate_plots(
        self,
        data: Dict[str, Any],
        session: OnboardingSession,
    ) -> OnboardingStepResult:
        """Validate production plots step with EUDR geolocation requirements.

        Validates GPS coordinates (WGS84, 6+ decimal precision), enforces
        polygon requirement for plots >4 hectares per Article 9(1)(d),
        and checks for HTML5 Geolocation API capture metadata.

        Args:
            data: Plots data with list of plot entries.
            session: Current session.

        Returns:
            Step validation result.
        """
        errors: List[str] = []
        warnings: List[str] = []
        info_msgs: List[str] = []
        completed = 0
        total = len(STEP_FIELDS["plots"])

        plots = data.get("plots", [])
        if not plots or not isinstance(plots, list) or len(plots) == 0:
            errors.append("At least one production plot is required")
        else:
            all_plots_valid = True
            for i, plot in enumerate(plots):
                plot_errors = self._validate_single_plot(plot, i)
                if plot_errors:
                    all_plots_valid = False
                    errors.extend(plot_errors)

            if all_plots_valid:
                completed += 1

            # Info about mobile capture support
            gps_plots = [
                p for p in plots
                if isinstance(p, dict) and p.get("capture_method") == "gps"
            ]
            if gps_plots:
                info_msgs.append(
                    f"{len(gps_plots)} plot(s) captured via mobile GPS"
                )

        return OnboardingStepResult(
            step_name="plots",
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            info_messages=info_msgs,
            fields_completed=completed,
            fields_total=total,
            provenance_hash=_compute_hash(data),
        )

    def _validate_single_plot(
        self,
        plot: Any,
        index: int,
    ) -> List[str]:
        """Validate a single plot entry against EUDR requirements.

        Args:
            plot: Plot data dictionary.
            index: Index in the plots list (for error messages).

        Returns:
            List of error messages (empty if valid).
        """
        errors: List[str] = []
        prefix = f"Plot[{index}]"

        if not isinstance(plot, dict):
            return [f"{prefix}: must be a dictionary"]

        # Latitude
        lat = plot.get("latitude")
        if lat is None:
            errors.append(f"{prefix}: latitude is required")
        else:
            try:
                lat = float(lat)
                if not (LAT_MIN <= lat <= LAT_MAX):
                    errors.append(
                        f"{prefix}: latitude must be between {LAT_MIN} and {LAT_MAX}"
                    )
                elif self._count_decimal_places(lat) < MIN_COORDINATE_PRECISION:
                    errors.append(
                        f"{prefix}: latitude must have at least "
                        f"{MIN_COORDINATE_PRECISION} decimal places for EUDR compliance"
                    )
            except (TypeError, ValueError):
                errors.append(f"{prefix}: latitude must be a valid number")

        # Longitude
        lon = plot.get("longitude")
        if lon is None:
            errors.append(f"{prefix}: longitude is required")
        else:
            try:
                lon = float(lon)
                if not (LON_MIN <= lon <= LON_MAX):
                    errors.append(
                        f"{prefix}: longitude must be between {LON_MIN} and {LON_MAX}"
                    )
                elif self._count_decimal_places(lon) < MIN_COORDINATE_PRECISION:
                    errors.append(
                        f"{prefix}: longitude must have at least "
                        f"{MIN_COORDINATE_PRECISION} decimal places for EUDR compliance"
                    )
            except (TypeError, ValueError):
                errors.append(f"{prefix}: longitude must be a valid number")

        # Area
        area = plot.get("area_hectares")
        if area is None:
            errors.append(f"{prefix}: area_hectares is required")
        else:
            try:
                area = float(area)
                if area <= 0:
                    errors.append(f"{prefix}: area_hectares must be > 0")
                elif area > POLYGON_AREA_THRESHOLD_HA:
                    # Polygon required for plots > 4 ha
                    polygon = plot.get("polygon_coordinates")
                    if not polygon or not isinstance(polygon, list) or len(polygon) < 3:
                        errors.append(
                            f"{prefix}: polygon_coordinates required for plots "
                            f">{POLYGON_AREA_THRESHOLD_HA} ha per EUDR Article 9(1)(d)"
                        )
            except (TypeError, ValueError):
                errors.append(f"{prefix}: area_hectares must be a valid number")

        # Commodity
        commodity = plot.get("commodity")
        if not commodity or not str(commodity).strip():
            errors.append(f"{prefix}: commodity is required")

        # Country code
        country_code = plot.get("country_code")
        if not country_code or not str(country_code).strip():
            errors.append(f"{prefix}: country_code is required")
        elif len(str(country_code).strip()) != 2:
            errors.append(f"{prefix}: country_code must be 2 characters")

        return errors

    def _validate_certifications(
        self,
        data: Dict[str, Any],
        session: OnboardingSession,
    ) -> OnboardingStepResult:
        """Validate certifications step.

        Certifications are optional but recommended. Validates format
        when provided.

        Args:
            data: Certifications data.
            session: Current session.

        Returns:
            Step validation result.
        """
        errors: List[str] = []
        warnings: List[str] = []
        completed = 0
        total = len(STEP_FIELDS["certifications"])

        certs = data.get("certifications", [])
        if not certs or not isinstance(certs, list) or len(certs) == 0:
            warnings.append(
                "No certifications provided. Certifications such as FSC, "
                "RSPO, or Rainforest Alliance can support EUDR compliance."
            )
            # Certifications are optional -- mark as completed with warning
            completed += 1
        else:
            valid = True
            for i, cert in enumerate(certs):
                if not isinstance(cert, dict):
                    errors.append(f"Certification[{i}]: must be a dictionary")
                    valid = False
                    continue
                if not cert.get("certification_type"):
                    errors.append(
                        f"Certification[{i}]: certification_type is required"
                    )
                    valid = False
                if not cert.get("certificate_number"):
                    errors.append(
                        f"Certification[{i}]: certificate_number is required"
                    )
                    valid = False
            if valid:
                completed += 1

        return OnboardingStepResult(
            step_name="certifications",
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            fields_completed=completed,
            fields_total=total,
            provenance_hash=_compute_hash(data),
        )

    def _validate_declarations(
        self,
        data: Dict[str, Any],
        session: OnboardingSession,
    ) -> OnboardingStepResult:
        """Validate EUDR compliance declarations step.

        Required: deforestation-free declaration and legality declaration.

        Args:
            data: Declaration data.
            session: Current session.

        Returns:
            Step validation result.
        """
        errors: List[str] = []
        warnings: List[str] = []
        completed = 0
        total = len(STEP_FIELDS["declarations"])

        # Deforestation-free declaration
        df_decl = data.get("deforestation_free_declaration")
        if df_decl is None or df_decl is False:
            errors.append(
                "Deforestation-free declaration is required per EUDR Article 4(2)"
            )
        elif df_decl is True:
            completed += 1

        # Legality declaration
        leg_decl = data.get("legality_declaration")
        if leg_decl is None or leg_decl is False:
            errors.append(
                "Legality declaration is required per EUDR Article 3(b)"
            )
        elif leg_decl is True:
            completed += 1

        return OnboardingStepResult(
            step_name="declarations",
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            fields_completed=completed,
            fields_total=total,
            provenance_hash=_compute_hash(data),
        )

    def _validate_sub_tier_suppliers(
        self,
        data: Dict[str, Any],
        session: OnboardingSession,
    ) -> OnboardingStepResult:
        """Validate sub-tier supplier information step.

        Sub-tier suppliers are optional but collection is required as a step.
        Validates format when entries are provided.

        Args:
            data: Sub-tier supplier data.
            session: Current session.

        Returns:
            Step validation result.
        """
        errors: List[str] = []
        warnings: List[str] = []
        info_msgs: List[str] = []
        completed = 0
        total = len(STEP_FIELDS["sub_tier_suppliers"])

        suppliers = data.get("sub_tier_suppliers", [])
        if not suppliers or not isinstance(suppliers, list) or len(suppliers) == 0:
            warnings.append(
                "No sub-tier suppliers declared. EUDR requires transparency "
                "into upstream supply chain."
            )
            # Step still counts as submitted/completed
            completed += 1
        else:
            valid = True
            for i, sup in enumerate(suppliers):
                if not isinstance(sup, dict):
                    errors.append(f"SubTierSupplier[{i}]: must be a dictionary")
                    valid = False
                    continue
                if not sup.get("supplier_name"):
                    errors.append(f"SubTierSupplier[{i}]: supplier_name is required")
                    valid = False
                if not sup.get("country_code"):
                    errors.append(f"SubTierSupplier[{i}]: country_code is required")
                    valid = False
                elif len(str(sup["country_code"]).strip()) != 2:
                    errors.append(
                        f"SubTierSupplier[{i}]: country_code must be 2 characters"
                    )
                    valid = False

                # Recursive onboarding suggestion
                if sup.get("contact_email"):
                    info_msgs.append(
                        f"SubTierSupplier[{i}] '{sup.get('supplier_name', '')}' "
                        f"has contact email -- eligible for recursive onboarding"
                    )

            if valid:
                completed += 1

        return OnboardingStepResult(
            step_name="sub_tier_suppliers",
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            info_messages=info_msgs,
            fields_completed=completed,
            fields_total=total,
            provenance_hash=_compute_hash(data),
        )

    # ------------------------------------------------------------------
    # Step Data Application
    # ------------------------------------------------------------------

    def _apply_step_data(
        self,
        session: OnboardingSession,
        step_name: str,
        data: Dict[str, Any],
    ) -> None:
        """Apply validated step data to the session state.

        Routes to step-specific application logic that stores the
        normalized data into the session's typed fields.

        Args:
            session: Target session.
            step_name: Step that was validated.
            data: Validated step data.
        """
        appliers: Dict[str, Callable[..., None]] = {
            "company_info": self._apply_company_info,
            "commodities": self._apply_commodities,
            "plots": self._apply_plots,
            "certifications": self._apply_certifications,
            "declarations": self._apply_declarations,
            "sub_tier_suppliers": self._apply_sub_tier_suppliers,
        }
        applier = appliers.get(step_name)
        if applier:
            applier(session, data)

    def _apply_company_info(
        self,
        session: OnboardingSession,
        data: Dict[str, Any],
    ) -> None:
        """Store validated company information."""
        session.company_info = {
            "legal_name": str(data.get("legal_name", "")).strip(),
            "country_code": str(data.get("country_code", "")).upper().strip(),
            "registration_id": str(data.get("registration_id", "")).strip(),
            "contact_name": str(data.get("contact_name", "")).strip(),
            "contact_email": str(data.get("contact_email", "")).strip(),
            "contact_phone": str(data.get("contact_phone", "")).strip(),
        }

    def _apply_commodities(
        self,
        session: OnboardingSession,
        data: Dict[str, Any],
    ) -> None:
        """Store validated commodity list."""
        commodities = data.get("commodities", [])
        session.commodities_data = [str(c).lower() for c in commodities]

    def _apply_plots(
        self,
        session: OnboardingSession,
        data: Dict[str, Any],
    ) -> None:
        """Store validated plot data as PlotData objects."""
        plots_raw = data.get("plots", [])
        session.plots_data = []
        for plot_dict in plots_raw:
            if isinstance(plot_dict, dict):
                try:
                    plot = PlotData(
                        latitude=float(plot_dict["latitude"]),
                        longitude=float(plot_dict["longitude"]),
                        area_hectares=float(plot_dict["area_hectares"]),
                        commodity=str(plot_dict["commodity"]),
                        country_code=str(plot_dict["country_code"]),
                        polygon_coordinates=plot_dict.get("polygon_coordinates"),
                        capture_method=plot_dict.get("capture_method", "manual"),
                        capture_accuracy_m=plot_dict.get("capture_accuracy_m"),
                    )
                    session.plots_data.append(plot)
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning("Failed to parse plot data: %s", e)

    def _apply_certifications(
        self,
        session: OnboardingSession,
        data: Dict[str, Any],
    ) -> None:
        """Store validated certification data."""
        certs_raw = data.get("certifications", [])
        session.certifications_data = []
        for cert_dict in certs_raw:
            if isinstance(cert_dict, dict):
                try:
                    cert = CertificationData(
                        certification_type=cert_dict["certification_type"],
                        certificate_number=cert_dict["certificate_number"],
                        issuing_body=cert_dict.get("issuing_body", ""),
                        scope=cert_dict.get("scope"),
                    )
                    session.certifications_data.append(cert)
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning("Failed to parse certification data: %s", e)

    def _apply_declarations(
        self,
        session: OnboardingSession,
        data: Dict[str, Any],
    ) -> None:
        """Store validated declaration data."""
        session.declarations_data = {
            "deforestation_free_declaration": bool(
                data.get("deforestation_free_declaration", False)
            ),
            "legality_declaration": bool(data.get("legality_declaration", False)),
            "declaration_date": _utcnow().isoformat(),
        }

    def _apply_sub_tier_suppliers(
        self,
        session: OnboardingSession,
        data: Dict[str, Any],
    ) -> None:
        """Store validated sub-tier supplier data."""
        suppliers_raw = data.get("sub_tier_suppliers", [])
        session.sub_tier_suppliers_data = []
        for sup_dict in suppliers_raw:
            if isinstance(sup_dict, dict):
                try:
                    supplier = SubTierSupplierData(
                        supplier_name=sup_dict["supplier_name"],
                        country_code=sup_dict["country_code"],
                        commodities=sup_dict.get("commodities", []),
                        relationship_type=sup_dict.get("relationship_type", "direct"),
                        contact_email=sup_dict.get("contact_email"),
                        estimated_volume_pct=sup_dict.get("estimated_volume_pct"),
                    )
                    session.sub_tier_suppliers_data.append(supplier)
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning("Failed to parse sub-tier supplier: %s", e)

    # ------------------------------------------------------------------
    # Completion Tracking
    # ------------------------------------------------------------------

    def _calculate_completion(self, session: OnboardingSession) -> float:
        """Calculate the onboarding completion percentage.

        Computed as (total fields completed across all submitted steps /
        total field groups across all steps) * 100. This is a deterministic
        arithmetic calculation with zero hallucination.

        Args:
            session: The onboarding session.

        Returns:
            Completion percentage (0.0 - 100.0).
        """
        if TOTAL_FIELD_GROUPS == 0:
            return 0.0

        fields_completed = 0
        for step_name, result in session.step_results.items():
            fields_completed += result.fields_completed

        pct = (fields_completed / TOTAL_FIELD_GROUPS) * 100.0
        return round(min(pct, 100.0), 1)

    # ------------------------------------------------------------------
    # Reminder System
    # ------------------------------------------------------------------

    def _schedule_reminders(
        self,
        session: OnboardingSession,
    ) -> List[ReminderRecord]:
        """Schedule automated reminders for an onboarding session.

        Creates reminders at the configured intervals from the session
        creation date. Reminders are in PENDING status until processed.

        Args:
            session: The onboarding session.

        Returns:
            List of scheduled ReminderRecord objects.
        """
        reminders: List[ReminderRecord] = []
        for days in self._reminder_days:
            scheduled_at = session.created_at + timedelta(days=days)
            reminder = ReminderRecord(
                session_id=session.session_id,
                reminder_type=ReminderType.EMAIL,
                status=ReminderStatus.PENDING,
                scheduled_at=scheduled_at,
                target_email=session.supplier_email,
            )
            reminders.append(reminder)
        return reminders

    def process_due_reminders(self) -> List[ReminderRecord]:
        """Process all due reminders across all active sessions.

        Finds reminders where scheduled_at <= now and status is PENDING,
        then attempts delivery via the notification service. Updates
        reminder status to SENT or FAILED accordingly.

        Returns:
            List of processed ReminderRecord objects.
        """
        processed: List[ReminderRecord] = []
        now = _utcnow()

        for session in self._sessions.values():
            if session.status in (
                OnboardingStatus.COMPLETED,
                OnboardingStatus.CANCELLED,
                OnboardingStatus.EXPIRED,
            ):
                continue

            for reminder in session.reminders:
                if (
                    reminder.status == ReminderStatus.PENDING
                    and reminder.scheduled_at <= now
                ):
                    self._send_reminder(session, reminder)
                    processed.append(reminder)

        if processed:
            logger.info("Processed %d due reminders", len(processed))

        return processed

    def _send_reminder(
        self,
        session: OnboardingSession,
        reminder: ReminderRecord,
    ) -> None:
        """Send a single reminder notification.

        Attempts delivery through the notification service. Falls back
        to marking as sent if no notification service is configured
        (for testing and development).

        Args:
            session: The onboarding session.
            reminder: The reminder to send.
        """
        reminder.attempt_count += 1

        if self._notification_service is None:
            # No notification service -- mark as sent for dev/testing
            reminder.status = ReminderStatus.SENT
            reminder.sent_at = _utcnow()
            logger.info(
                "Reminder %s marked as sent (no notification service): "
                "session=%s, email=%s",
                reminder.reminder_id,
                session.session_id,
                reminder.target_email,
            )
            return

        try:
            subject = self._get_reminder_subject(session)
            body = self._get_reminder_body(session)

            success = self._notification_service.send_email(
                to_email=reminder.target_email or session.supplier_email,
                subject=subject,
                body=body,
                language=session.language,
            )

            if success:
                reminder.status = ReminderStatus.SENT
                reminder.sent_at = _utcnow()
            else:
                reminder.status = ReminderStatus.FAILED
                reminder.error_message = "Notification service returned False"

        except Exception as e:
            reminder.status = ReminderStatus.FAILED
            reminder.error_message = str(e)
            logger.error(
                "Reminder delivery failed: session=%s, reminder=%s, error=%s",
                session.session_id,
                reminder.reminder_id,
                e,
            )

    def _get_reminder_subject(self, session: OnboardingSession) -> str:
        """Generate reminder email subject line.

        Args:
            session: The onboarding session.

        Returns:
            Subject line string.
        """
        return (
            f"EUDR Onboarding Reminder: Please complete your supplier profile "
            f"({session.supplier_name})"
        )

    def _get_reminder_body(self, session: OnboardingSession) -> str:
        """Generate reminder email body text.

        Args:
            session: The onboarding session.

        Returns:
            Body text string.
        """
        return (
            f"Dear {session.supplier_name},\n\n"
            f"Your EUDR supplier onboarding is {session.completion_pct:.0f}% complete.\n"
            f"Please submit the remaining information to ensure EU Deforestation "
            f"Regulation compliance.\n\n"
            f"Steps completed: {len(session.steps_completed)}/{len(WIZARD_STEPS)}\n"
            f"Onboarding link: [Use your unique token to access the portal]\n\n"
            f"This link expires on {session.token_expires_at}.\n\n"
            f"Thank you,\n"
            f"GreenLang EUDR Compliance Platform"
        )

    # ------------------------------------------------------------------
    # Bulk Import
    # ------------------------------------------------------------------

    def bulk_import_from_csv(
        self,
        csv_content: str,
        operator_id: str,
        graph_id: str,
        language: str = "en",
    ) -> BulkImportResult:
        """Import multiple suppliers from CSV data.

        Parses CSV content and creates onboarding sessions for each valid
        row. Required columns: supplier_name, country_code, commodity.
        Optional columns: supplier_email, contact_name, hs_code.

        Integration: Compatible with AGENT-DATA-002 Excel/CSV Normalizer output.

        Args:
            csv_content: Raw CSV string content.
            operator_id: Operator initiating the import.
            graph_id: Target supply chain graph.
            language: Default language for all sessions.

        Returns:
            BulkImportResult with per-row status.

        Raises:
            ValueError: If CSV is empty or missing required columns.
        """
        start_time = time.monotonic()
        self._bulk_import_count += 1

        result = BulkImportResult()

        if not csv_content or not csv_content.strip():
            result.status = BulkImportStatus.FAILED
            result.errors[0] = "CSV content is empty"
            return result

        result.status = BulkImportStatus.PROCESSING

        try:
            reader = csv.DictReader(io.StringIO(csv_content))

            # Validate required columns
            if reader.fieldnames is None:
                result.status = BulkImportStatus.FAILED
                result.errors[0] = "CSV has no header row"
                return result

            headers_lower = {h.lower().strip() for h in reader.fieldnames}
            missing = []
            for col in BULK_IMPORT_REQUIRED_COLUMNS:
                if col not in headers_lower:
                    missing.append(col)

            if missing:
                result.status = BulkImportStatus.FAILED
                result.errors[0] = f"Missing required columns: {missing}"
                return result

            rows = list(reader)
            result.total_rows = len(rows)

            if result.total_rows > MAX_BULK_IMPORT_BATCH:
                result.status = BulkImportStatus.FAILED
                result.errors[0] = (
                    f"Batch size {result.total_rows} exceeds maximum "
                    f"{MAX_BULK_IMPORT_BATCH}"
                )
                return result

            for row_num, row in enumerate(rows, start=1):
                try:
                    # Normalize column names to lowercase
                    normalized = {
                        k.lower().strip(): v.strip() if v else ""
                        for k, v in row.items()
                    }

                    supplier_name = normalized.get("supplier_name", "")
                    country_code = normalized.get("country_code", "")
                    commodity = normalized.get("commodity", "")
                    supplier_email = normalized.get(
                        "supplier_email",
                        f"onboarding+{uuid.uuid4().hex[:8]}@placeholder.local",
                    )

                    if not supplier_name:
                        result.errors[row_num] = "supplier_name is empty"
                        result.rows_failed += 1
                        continue

                    if not country_code or len(country_code) != 2:
                        result.errors[row_num] = "country_code must be 2 characters"
                        result.rows_failed += 1
                        continue

                    if not commodity:
                        result.errors[row_num] = "commodity is empty"
                        result.rows_failed += 1
                        continue

                    # Validate email format
                    if not re.match(
                        r"^[^@\s]+@[^@\s]+\.[^@\s]+$", supplier_email
                    ):
                        supplier_email = (
                            f"onboarding+{uuid.uuid4().hex[:8]}@placeholder.local"
                        )

                    session = self.create_onboarding_session(
                        operator_id=operator_id,
                        graph_id=graph_id,
                        supplier_name=supplier_name,
                        supplier_email=supplier_email,
                        commodity=commodity,
                        language=language,
                    )

                    # Auto-submit company_info if additional data is available
                    contact_name = normalized.get("contact_name", "")
                    if contact_name and supplier_email:
                        company_data = {
                            "legal_name": supplier_name,
                            "country_code": country_code,
                            "registration_id": normalized.get("registration_id", "N/A"),
                            "contact_name": contact_name,
                            "contact_email": supplier_email,
                            "contact_phone": normalized.get("contact_phone", ""),
                        }
                        try:
                            self.submit_step(
                                session.session_id, "company_info", company_data
                            )
                        except ValueError:
                            pass  # Non-critical: session created regardless

                    result.sessions_created.append(session.session_id)
                    result.rows_succeeded += 1

                except Exception as e:
                    result.errors[row_num] = str(e)
                    result.rows_failed += 1

        except csv.Error as e:
            result.status = BulkImportStatus.FAILED
            result.errors[0] = f"CSV parsing error: {e}"
            return result

        # Determine final status
        if result.rows_failed == 0 and result.rows_succeeded > 0:
            result.status = BulkImportStatus.COMPLETED
        elif result.rows_succeeded > 0 and result.rows_failed > 0:
            result.status = BulkImportStatus.PARTIAL
        elif result.rows_succeeded == 0:
            result.status = BulkImportStatus.FAILED

        result.processing_time_ms = (time.monotonic() - start_time) * 1000
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Bulk import completed: import=%s, status=%s, "
            "total=%d, succeeded=%d, failed=%d, elapsed=%.1fms",
            result.import_id,
            result.status.value,
            result.total_rows,
            result.rows_succeeded,
            result.rows_failed,
            result.processing_time_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Graph Node/Edge Auto-Creation
    # ------------------------------------------------------------------

    async def create_graph_entities(
        self,
        session_id: str,
    ) -> Dict[str, Any]:
        """Auto-create supply chain graph nodes and edges from completed onboarding.

        Creates a supplier node in the graph engine, registers plots via the
        geolocation linker, and creates edges from sub-tier suppliers. Only
        processes sessions in COMPLETED status.

        Args:
            session_id: The completed onboarding session.

        Returns:
            Dictionary with created node_ids, edge_ids, and plot_link_ids.

        Raises:
            ValueError: If session is not found or not completed.
            RuntimeError: If graph_engine is not configured.
        """
        start_time = time.monotonic()

        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")

        if session.status != OnboardingStatus.COMPLETED:
            raise ValueError(
                f"Session {session_id} is not completed "
                f"(status={session.status.value})"
            )

        if self._graph_engine is None:
            raise RuntimeError(
                "graph_engine is not configured. Cannot create graph entities."
            )

        result: Dict[str, Any] = {
            "node_ids": [],
            "edge_ids": [],
            "plot_link_ids": [],
        }

        # Create the supplier node
        node_type_str = self._infer_node_type(session)
        country_code = session.company_info.get("country_code", "XX")
        operator_name = session.company_info.get(
            "legal_name", session.supplier_name
        )

        node_id = await self._graph_engine.add_node(
            graph_id=session.graph_id,
            node_type=node_type_str,
            operator_name=operator_name,
            country_code=country_code,
            operator_id=session.session_id,
            commodities=session.commodities_data,
            certifications=[
                c.certificate_number for c in session.certifications_data
            ],
        )
        result["node_ids"].append(node_id)
        session.created_node_ids.append(node_id)

        # Register plots via geolocation linker
        if self._geo_linker and session.plots_data:
            for plot in session.plots_data:
                try:
                    link_result = self._geo_linker.link_producer_to_plot(
                        producer_node_id=node_id,
                        plot_id=plot.plot_id,
                        latitude=plot.latitude,
                        longitude=plot.longitude,
                        polygon_coordinates=plot.polygon_coordinates,
                        area_hectares=plot.area_hectares,
                        commodity=plot.commodity,
                        country_code=plot.country_code,
                    )
                    link_id = link_result.get("link_id", "")
                    if link_id:
                        result["plot_link_ids"].append(link_id)
                except Exception as e:
                    logger.error(
                        "Failed to link plot %s: %s", plot.plot_id, e
                    )

        # Create sub-tier supplier nodes and edges
        for sub_sup in session.sub_tier_suppliers_data:
            try:
                sub_node_id = await self._graph_engine.add_node(
                    graph_id=session.graph_id,
                    node_type="producer",
                    operator_name=sub_sup.supplier_name,
                    country_code=sub_sup.country_code,
                    operator_id=f"subtier-{sub_sup.supplier_name[:20]}",
                    commodities=sub_sup.commodities,
                )
                result["node_ids"].append(sub_node_id)
                session.created_node_ids.append(sub_node_id)

                # Create edge from sub-tier supplier to the onboarded supplier
                commodity = (
                    sub_sup.commodities[0] if sub_sup.commodities
                    else session.commodity
                )
                edge_id = await self._graph_engine.add_edge(
                    graph_id=session.graph_id,
                    source_node_id=sub_node_id,
                    target_node_id=node_id,
                    commodity=commodity,
                    quantity=Decimal("0"),
                    product_description=f"Supply from {sub_sup.supplier_name}",
                )
                result["edge_ids"].append(edge_id)
                session.created_edge_ids.append(edge_id)

            except Exception as e:
                logger.error(
                    "Failed to create sub-tier node for %s: %s",
                    sub_sup.supplier_name,
                    e,
                )

        session.updated_at = _utcnow()
        session.provenance_hash = _compute_hash(session)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Graph entities created from onboarding: session=%s, "
            "nodes=%d, edges=%d, plots=%d, elapsed=%.1fms",
            session_id,
            len(result["node_ids"]),
            len(result["edge_ids"]),
            len(result["plot_link_ids"]),
            elapsed_ms,
        )

        return result

    def _infer_node_type(self, session: OnboardingSession) -> str:
        """Infer the supply chain node type from onboarding data.

        If the supplier has production plots, they are a producer.
        Otherwise defaults to trader.

        Args:
            session: Completed onboarding session.

        Returns:
            Node type string (producer, trader, etc.).
        """
        if session.plots_data:
            return "producer"
        return "trader"

    # ------------------------------------------------------------------
    # Metrics and Reporting
    # ------------------------------------------------------------------

    def get_metrics(self) -> OnboardingMetrics:
        """Calculate aggregated onboarding metrics.

        All calculations are deterministic arithmetic -- no LLM involvement.

        Returns:
            OnboardingMetrics with completion rates and reminder counts.
        """
        metrics = OnboardingMetrics()
        metrics.total_sessions = len(self._sessions)
        metrics.bulk_imports = self._bulk_import_count

        status_counts: Dict[str, int] = {}
        completion_pcts: List[float] = []
        completion_days: List[float] = []
        within_14_count = 0

        for session in self._sessions.values():
            status_key = session.status.value
            status_counts[status_key] = status_counts.get(status_key, 0) + 1
            completion_pcts.append(session.completion_pct)

            # Count reminders sent
            for reminder in session.reminders:
                if reminder.status == ReminderStatus.SENT:
                    metrics.reminders_sent += 1

            # Track completion timing
            if session.status == OnboardingStatus.COMPLETED and session.completed_at:
                days = (session.completed_at - session.created_at).total_seconds() / 86400
                completion_days.append(days)
                if days <= 14:
                    within_14_count += 1

        metrics.sessions_by_status = status_counts

        if completion_pcts:
            metrics.average_completion_pct = round(
                sum(completion_pcts) / len(completion_pcts), 1
            )

        if completion_days:
            sorted_days = sorted(completion_days)
            mid = len(sorted_days) // 2
            if len(sorted_days) % 2 == 0 and len(sorted_days) > 1:
                metrics.median_completion_days = round(
                    (sorted_days[mid - 1] + sorted_days[mid]) / 2, 1
                )
            else:
                metrics.median_completion_days = round(sorted_days[mid], 1)

        completed_count = status_counts.get("completed", 0)
        if completed_count > 0:
            metrics.completion_within_14_days_pct = round(
                (within_14_count / completed_count) * 100, 1
            )

        return metrics

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    @staticmethod
    def _count_decimal_places(value: float) -> int:
        """Count the number of decimal places in a float value.

        Uses string representation to determine precision. This is a
        deterministic calculation used for GPS coordinate precision
        validation per EUDR requirements.

        Args:
            value: The floating-point value to check.

        Returns:
            Number of decimal places.
        """
        str_val = str(value)
        if "." not in str_val:
            return 0
        return len(str_val.split(".")[1])

    def generate_onboarding_link(
        self,
        session_id: str,
        base_url: str = "https://app.greenlang.io/eudr/onboarding",
    ) -> Optional[str]:
        """Generate the full onboarding URL for a supplier.

        Creates a URL combining the base application URL with the secure
        session token. This URL can be shared via email, SMS, or embedded
        in a QR code for mobile access.

        Args:
            session_id: The onboarding session.
            base_url: Base application URL.

        Returns:
            Full onboarding URL, or None if session not found.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return None
        return f"{base_url}?token={session.token}"

    def get_mobile_gps_config(self) -> Dict[str, Any]:
        """Get HTML5 Geolocation API configuration for mobile GPS capture.

        Returns configuration parameters suitable for use with the
        browser Geolocation API to capture plot coordinates on mobile
        devices (iOS Safari, Chrome for Android).

        Returns:
            Configuration dictionary for HTML5 Geolocation API.
        """
        return {
            "enableHighAccuracy": True,
            "timeout": 30000,
            "maximumAge": 0,
            "requiredPrecision": MIN_COORDINATE_PRECISION,
            "minAccuracyMeters": 10.0,
            "supportedCaptureMethods": ["gps", "manual", "upload"],
            "eudrRequirements": {
                "minDecimalPlaces": MIN_COORDINATE_PRECISION,
                "polygonThresholdHa": POLYGON_AREA_THRESHOLD_HA,
                "coordinateSystem": "WGS84",
                "srid": 4326,
            },
        }

    @property
    def session_count(self) -> int:
        """Total number of onboarding sessions."""
        return len(self._sessions)


# ===========================================================================
# Public API
# ===========================================================================

__all__ = [
    # Constants
    "SUPPORTED_LANGUAGES",
    "DEFAULT_TOKEN_EXPIRY_DAYS",
    "DEFAULT_REMINDER_DAYS",
    "MAX_BULK_IMPORT_BATCH",
    "MIN_COORDINATE_PRECISION",
    "POLYGON_AREA_THRESHOLD_HA",
    "WIZARD_STEPS",
    "STEP_FIELDS",
    "TOTAL_FIELD_GROUPS",
    "BULK_IMPORT_REQUIRED_COLUMNS",
    # Enumerations
    "OnboardingStatus",
    "ReminderType",
    "ReminderStatus",
    "ValidationSeverity",
    "BulkImportStatus",
    # Protocols
    "GraphEngineProtocol",
    "GeolocationLinkerProtocol",
    "NotificationServiceProtocol",
    "QuestionnaireProcessorProtocol",
    # Data Models
    "PlotData",
    "CertificationData",
    "SubTierSupplierData",
    "OnboardingStepResult",
    "ReminderRecord",
    "OnboardingSession",
    "BulkImportResult",
    "OnboardingMetrics",
    # Engine
    "SupplierOnboardingEngine",
]
