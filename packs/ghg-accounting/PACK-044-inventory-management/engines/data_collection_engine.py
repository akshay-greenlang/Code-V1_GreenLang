# -*- coding: utf-8 -*-
"""
DataCollectionEngine - PACK-044 Inventory Management Engine 2
==============================================================

Data collection campaign management engine for GHG inventory programmes.
Orchestrates the systematic collection of activity data from facilities,
business units, and third-party suppliers through structured campaigns,
automated data requests, assignment tracking, validation, and coverage
analysis.

A campaign represents a coordinated effort to gather all required activity
data for one inventory period.  Within a campaign, individual data requests
are sent to data owners (facility managers, procurement teams, fleet
managers, etc.).  Each request is tracked through assignment, submission,
validation, and acceptance stages.

Core Capabilities:
    1. Campaign creation linked to an inventory period
    2. Data request generation for specific scopes / categories / facilities
    3. Assignment of requests to data owners with due dates
    4. Submission tracking (submitted, validated, accepted, rejected)
    5. Automated reminder escalation based on due date proximity
    6. Data validation against expected ranges and formats
    7. Coverage analysis (% of expected data points collected)
    8. Campaign progress dashboard metrics

Calculation Methodology:
    Campaign Progress:
        progress_pct = (accepted_requests / total_requests) * 100

    Coverage Rate:
        coverage_pct = (collected_data_points / expected_data_points) * 100

    Timeliness Score:
        on_time_pct = (on_time_submissions / total_submissions) * 100

    Reminder Escalation:
        days_overdue = (today - due_date).days
        escalation_level = 1 if days_overdue <= 7 else 2 if days_overdue <= 14 else 3

Regulatory References:
    - GHG Protocol Corporate Standard (Revised), Ch 6-9 (Data Requirements)
    - ISO 14064-1:2018, Clause 5.3 (Quantification)
    - GHG Protocol Scope 3 Standard, Ch 7 (Collecting Data)
    - EU CSRD / ESRS E1-6 (Disclosure Requirements on Data)
    - CDP Climate Change Questionnaire, C6-C7 (Data Collection)

Zero-Hallucination:
    - All progress metrics use deterministic Decimal arithmetic
    - Reminder escalation based on date arithmetic only
    - Validation uses range checks, not ML predictions
    - No LLM involvement in any data collection logic
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-044 Inventory Management
Engine:  2 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _today() -> date:
    """Return current UTC date."""
    return datetime.now(timezone.utc).date()


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)


def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RequestStatus(str, Enum):
    """Data request lifecycle status.

    DRAFT:       Request created but not yet sent.
    SENT:        Request dispatched to assignee.
    ACKNOWLEDGED: Assignee has acknowledged receipt.
    IN_PROGRESS: Assignee is actively working on data.
    SUBMITTED:   Data submitted by assignee; awaiting validation.
    VALIDATED:   Data passed automated validation checks.
    ACCEPTED:    Data reviewed and accepted into inventory.
    REJECTED:    Data rejected; requires resubmission.
    OVERDUE:     Past due date with no submission.
    CANCELLED:   Request cancelled (no longer needed).
    """
    DRAFT = "draft"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    SUBMITTED = "submitted"
    VALIDATED = "validated"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"


class CampaignStatus(str, Enum):
    """Campaign lifecycle status.

    PLANNING:     Campaign being set up; requests being drafted.
    ACTIVE:       Campaign launched; requests dispatched.
    IN_REVIEW:    All data submitted; under review.
    COMPLETED:    All data accepted; campaign finished.
    CLOSED:       Campaign closed (may have incomplete data).
    """
    PLANNING = "planning"
    ACTIVE = "active"
    IN_REVIEW = "in_review"
    COMPLETED = "completed"
    CLOSED = "closed"


class DataScope(str, Enum):
    """GHG scope for which data is requested.

    SCOPE_1:  Direct emissions.
    SCOPE_2:  Indirect energy emissions.
    SCOPE_3:  Other indirect emissions.
    ALL:      All scopes.
    """
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"
    ALL = "all"


class EscalationLevel(str, Enum):
    """Reminder escalation level.

    NONE:     Not overdue; no escalation.
    LEVEL_1:  1-7 days overdue; standard reminder.
    LEVEL_2:  8-14 days overdue; escalated reminder.
    LEVEL_3:  15+ days overdue; management escalation.
    """
    NONE = "none"
    LEVEL_1 = "level_1"
    LEVEL_2 = "level_2"
    LEVEL_3 = "level_3"


class ValidationSeverity(str, Enum):
    """Severity level for data validation findings.

    ERROR:   Data cannot be accepted; must be corrected.
    WARNING: Data may be accepted but should be reviewed.
    INFO:    Informational finding; no action required.
    """
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# ---------------------------------------------------------------------------
# Pydantic Models -- Data Request
# ---------------------------------------------------------------------------


class ValidationFinding(BaseModel):
    """A single finding from data validation.

    Attributes:
        finding_id: Unique finding ID.
        field_name: Name of the data field with the finding.
        severity: Severity level.
        message: Human-readable description.
        expected_range: Expected value range (if applicable).
        actual_value: Actual submitted value.
    """
    finding_id: str = Field(default_factory=_new_uuid, description="Finding ID")
    field_name: str = Field(default="", max_length=200, description="Field name")
    severity: ValidationSeverity = Field(
        default=ValidationSeverity.INFO, description="Severity"
    )
    message: str = Field(default="", max_length=1000, description="Message")
    expected_range: str = Field(default="", description="Expected range")
    actual_value: str = Field(default="", description="Actual value")


class DataSubmission(BaseModel):
    """A data submission against a data request.

    Attributes:
        submission_id: Unique submission ID.
        request_id: Parent data request ID.
        submitted_by: User who submitted the data.
        submitted_at: Submission timestamp.
        data_payload: The submitted data (key-value pairs).
        file_references: References to uploaded files.
        validation_findings: Results of automated validation.
        validation_passed: Whether all ERROR findings are resolved.
        reviewer_notes: Notes from the reviewer.
        accepted: Whether the submission was accepted.
        accepted_at: Acceptance timestamp.
        accepted_by: User who accepted.
    """
    submission_id: str = Field(default_factory=_new_uuid, description="Submission ID")
    request_id: str = Field(default="", description="Parent request ID")
    submitted_by: str = Field(default="", max_length=300, description="Submitted by")
    submitted_at: datetime = Field(
        default_factory=_utcnow, description="Submission time"
    )
    data_payload: Dict[str, Any] = Field(
        default_factory=dict, description="Submitted data"
    )
    file_references: List[str] = Field(
        default_factory=list, description="Uploaded file refs"
    )
    validation_findings: List[ValidationFinding] = Field(
        default_factory=list, description="Validation findings"
    )
    validation_passed: bool = Field(default=False, description="Validation passed?")
    reviewer_notes: str = Field(default="", max_length=2000, description="Reviewer notes")
    accepted: bool = Field(default=False, description="Accepted?")
    accepted_at: Optional[datetime] = Field(default=None, description="Accepted at")
    accepted_by: str = Field(default="", description="Accepted by")


class ReminderRecord(BaseModel):
    """Record of a reminder sent for a data request.

    Attributes:
        reminder_id: Unique reminder ID.
        request_id: Parent data request ID.
        sent_at: When the reminder was sent.
        escalation_level: Escalation level of the reminder.
        recipient: Who the reminder was sent to.
        message: Reminder message content.
    """
    reminder_id: str = Field(default_factory=_new_uuid, description="Reminder ID")
    request_id: str = Field(default="", description="Request ID")
    sent_at: datetime = Field(default_factory=_utcnow, description="Sent at")
    escalation_level: EscalationLevel = Field(
        default=EscalationLevel.NONE, description="Escalation level"
    )
    recipient: str = Field(default="", max_length=300, description="Recipient")
    message: str = Field(default="", max_length=2000, description="Message")


class DataRequest(BaseModel):
    """A single data collection request within a campaign.

    Attributes:
        request_id: Unique request identifier.
        campaign_id: Parent campaign ID.
        scope: GHG scope being requested.
        category: Emission category (e.g. 'stationary_combustion').
        facility_id: Target facility ID.
        facility_name: Target facility name.
        entity_id: Target entity ID.
        assigned_to: Data owner assigned to fulfill the request.
        assigned_at: Assignment timestamp.
        due_date: Data submission deadline.
        status: Current request status.
        data_fields_requested: List of specific data fields expected.
        instructions: Instructions for the data owner.
        submissions: Submission history.
        reminders: Reminder history.
        escalation_level: Current escalation level.
        created_at: Creation timestamp.
        created_by: User who created the request.
    """
    request_id: str = Field(default_factory=_new_uuid, description="Request ID")
    campaign_id: str = Field(default="", description="Campaign ID")
    scope: DataScope = Field(default=DataScope.ALL, description="GHG scope")
    category: str = Field(default="", max_length=200, description="Emission category")
    facility_id: str = Field(default="", description="Facility ID")
    facility_name: str = Field(default="", max_length=300, description="Facility name")
    entity_id: str = Field(default="", description="Entity ID")
    assigned_to: str = Field(default="", max_length=300, description="Assigned to")
    assigned_at: Optional[datetime] = Field(default=None, description="Assigned at")
    due_date: Optional[date] = Field(default=None, description="Due date")
    status: RequestStatus = Field(
        default=RequestStatus.DRAFT, description="Request status"
    )
    data_fields_requested: List[str] = Field(
        default_factory=list, description="Expected data fields"
    )
    instructions: str = Field(default="", max_length=5000, description="Instructions")
    submissions: List[DataSubmission] = Field(
        default_factory=list, description="Submissions"
    )
    reminders: List[ReminderRecord] = Field(
        default_factory=list, description="Reminders"
    )
    escalation_level: EscalationLevel = Field(
        default=EscalationLevel.NONE, description="Escalation level"
    )
    created_at: datetime = Field(default_factory=_utcnow, description="Created at")
    created_by: str = Field(default="", max_length=300, description="Created by")


# ---------------------------------------------------------------------------
# Pydantic Models -- Campaign
# ---------------------------------------------------------------------------


class CollectionProgress(BaseModel):
    """Progress metrics for a data collection campaign.

    Attributes:
        total_requests: Total number of data requests.
        draft_count: Number in DRAFT status.
        sent_count: Number sent but not yet submitted.
        submitted_count: Number submitted (pending review).
        validated_count: Number that passed validation.
        accepted_count: Number accepted into inventory.
        rejected_count: Number rejected.
        overdue_count: Number past due date.
        cancelled_count: Number cancelled.
        progress_pct: Overall progress percentage (accepted / total * 100).
        coverage_pct: Data coverage percentage.
        on_time_pct: Timeliness percentage.
        expected_data_points: Total expected data points across all requests.
        collected_data_points: Data points that have been collected.
    """
    total_requests: int = Field(default=0, description="Total requests")
    draft_count: int = Field(default=0, description="Draft count")
    sent_count: int = Field(default=0, description="Sent count")
    submitted_count: int = Field(default=0, description="Submitted count")
    validated_count: int = Field(default=0, description="Validated count")
    accepted_count: int = Field(default=0, description="Accepted count")
    rejected_count: int = Field(default=0, description="Rejected count")
    overdue_count: int = Field(default=0, description="Overdue count")
    cancelled_count: int = Field(default=0, description="Cancelled count")
    progress_pct: Decimal = Field(default=Decimal("0"), description="Progress %")
    coverage_pct: Decimal = Field(default=Decimal("0"), description="Coverage %")
    on_time_pct: Decimal = Field(default=Decimal("0"), description="On-time %")
    expected_data_points: int = Field(default=0, description="Expected data points")
    collected_data_points: int = Field(default=0, description="Collected data points")


class CollectionCampaign(BaseModel):
    """A data collection campaign for one inventory period.

    Attributes:
        campaign_id: Unique campaign identifier.
        period_id: Linked inventory period ID.
        organisation_id: Organisation ID.
        campaign_name: Display name.
        status: Campaign lifecycle status.
        start_date: Campaign start date.
        end_date: Campaign end date (overall deadline).
        requests: Data requests within this campaign.
        progress: Calculated progress metrics.
        created_at: Creation timestamp.
        created_by: User who created the campaign.
        notes: Campaign notes.
    """
    campaign_id: str = Field(default_factory=_new_uuid, description="Campaign ID")
    period_id: str = Field(default="", description="Period ID")
    organisation_id: str = Field(default="", description="Organisation ID")
    campaign_name: str = Field(default="", max_length=500, description="Campaign name")
    status: CampaignStatus = Field(
        default=CampaignStatus.PLANNING, description="Campaign status"
    )
    start_date: Optional[date] = Field(default=None, description="Start date")
    end_date: Optional[date] = Field(default=None, description="End date")
    requests: List[DataRequest] = Field(
        default_factory=list, description="Data requests"
    )
    progress: CollectionProgress = Field(
        default_factory=CollectionProgress, description="Progress metrics"
    )
    created_at: datetime = Field(default_factory=_utcnow, description="Created at")
    created_by: str = Field(default="", max_length=300, description="Created by")
    notes: str = Field(default="", max_length=5000, description="Notes")


# ---------------------------------------------------------------------------
# Pydantic Models -- Result
# ---------------------------------------------------------------------------


class DataCollectionResult(BaseModel):
    """Complete result from a data collection engine operation.

    Attributes:
        result_id: Unique result ID.
        operation: Name of the operation performed.
        campaign: The campaign (after operation).
        request: The data request (if operation targets a single request).
        submission: The submission (if a submission operation).
        reminders_generated: Reminder records generated this operation.
        progress: Updated campaign progress metrics.
        warnings: Operational warnings.
        calculated_at: Timestamp.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    operation: str = Field(default="", description="Operation name")
    campaign: Optional[CollectionCampaign] = Field(
        default=None, description="Campaign"
    )
    request: Optional[DataRequest] = Field(default=None, description="Data request")
    submission: Optional[DataSubmission] = Field(
        default=None, description="Submission"
    )
    reminders_generated: List[ReminderRecord] = Field(
        default_factory=list, description="Reminders generated"
    )
    progress: Optional[CollectionProgress] = Field(
        default=None, description="Progress metrics"
    )
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    calculated_at: datetime = Field(default_factory=_utcnow, description="Timestamp")
    processing_time_ms: Decimal = Field(
        default=Decimal("0"), description="Processing time (ms)"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")


# ---------------------------------------------------------------------------
# Model Rebuild (resolve forward references from __future__ annotations)
# ---------------------------------------------------------------------------

ValidationFinding.model_rebuild()
DataSubmission.model_rebuild()
ReminderRecord.model_rebuild()
DataRequest.model_rebuild()
CollectionProgress.model_rebuild()
CollectionCampaign.model_rebuild()
DataCollectionResult.model_rebuild()


# ---------------------------------------------------------------------------
# Validation Rules
# ---------------------------------------------------------------------------

# Default validation ranges for common data fields.
# Each entry: { field_pattern: (min_value, max_value, unit) }
DEFAULT_VALIDATION_RANGES: Dict[str, Tuple[float, float, str]] = {
    "electricity_kwh": (0, 1_000_000_000, "kWh"),
    "natural_gas_m3": (0, 500_000_000, "m3"),
    "diesel_litres": (0, 100_000_000, "litres"),
    "petrol_litres": (0, 100_000_000, "litres"),
    "steam_kg": (0, 500_000_000, "kg"),
    "refrigerant_kg": (0, 100_000, "kg"),
    "waste_tonnes": (0, 10_000_000, "tonnes"),
    "distance_km": (0, 50_000_000, "km"),
    "floor_area_m2": (0, 10_000_000, "m2"),
    "headcount": (0, 1_000_000, "persons"),
    "revenue_usd": (0, 1_000_000_000_000, "USD"),
    "production_tonnes": (0, 500_000_000, "tonnes"),
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class DataCollectionEngine:
    """Data collection campaign management engine.

    Orchestrates the systematic collection of activity data for GHG
    inventories through structured campaigns, automated requests, and
    comprehensive tracking.

    All progress metrics are computed via deterministic Decimal arithmetic.
    No LLM involvement in any collection or validation logic.

    Attributes:
        _campaigns: In-memory registry of campaigns.

    Example:
        >>> engine = DataCollectionEngine()
        >>> result = engine.create_campaign(
        ...     period_id="per-001",
        ...     organisation_id="org-001",
        ...     campaign_name="FY2025 Data Collection",
        ... )
        >>> campaign_id = result.campaign.campaign_id
        >>> result = engine.add_request(
        ...     campaign_id=campaign_id,
        ...     scope=DataScope.SCOPE_1,
        ...     category="stationary_combustion",
        ...     facility_id="fac-001",
        ... )
    """

    def __init__(self) -> None:
        """Initialise DataCollectionEngine."""
        self._campaigns: Dict[str, CollectionCampaign] = {}
        logger.info("DataCollectionEngine v%s initialised", _MODULE_VERSION)

    # ------------------------------------------------------------------
    # Public API -- Campaign Management
    # ------------------------------------------------------------------

    def create_campaign(
        self,
        period_id: str,
        organisation_id: str,
        campaign_name: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        created_by: str = "system",
        notes: str = "",
    ) -> DataCollectionResult:
        """Create a new data collection campaign.

        Args:
            period_id: Linked inventory period ID.
            organisation_id: Organisation ID.
            campaign_name: Display name for the campaign.
            start_date: Campaign start date.
            end_date: Campaign overall deadline.
            created_by: User creating the campaign.
            notes: Campaign notes.

        Returns:
            DataCollectionResult with the created campaign.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []

        if start_date and end_date and end_date <= start_date:
            raise ValueError(
                f"Campaign end_date ({end_date}) must be after "
                f"start_date ({start_date})"
            )

        campaign = CollectionCampaign(
            period_id=period_id,
            organisation_id=organisation_id,
            campaign_name=campaign_name,
            start_date=start_date or _today(),
            end_date=end_date,
            created_by=created_by,
            notes=notes,
        )

        self._campaigns[campaign.campaign_id] = campaign
        logger.info(
            "Created campaign '%s' for period %s [%s]",
            campaign_name, period_id, campaign.campaign_id,
        )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000)
        result = DataCollectionResult(
            operation="create_campaign",
            campaign=campaign,
            progress=campaign.progress,
            warnings=warnings,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def launch_campaign(
        self,
        campaign_id: str,
    ) -> DataCollectionResult:
        """Launch a campaign, changing status to ACTIVE and sending all requests.

        All DRAFT requests within the campaign are transitioned to SENT.

        Args:
            campaign_id: Campaign ID to launch.

        Returns:
            DataCollectionResult with updated campaign.

        Raises:
            KeyError: If campaign not found.
            ValueError: If campaign has no requests or is not in PLANNING.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []

        campaign = self._get_campaign(campaign_id)

        if campaign.status != CampaignStatus.PLANNING:
            raise ValueError(
                f"Campaign must be in PLANNING to launch, "
                f"current status: {campaign.status.value}"
            )

        if not campaign.requests:
            raise ValueError("Campaign has no data requests to launch")

        # Transition all DRAFT requests to SENT.
        sent_count = 0
        for req in campaign.requests:
            if req.status == RequestStatus.DRAFT:
                req.status = RequestStatus.SENT
                sent_count += 1

        campaign.status = CampaignStatus.ACTIVE
        self._recalculate_progress(campaign)

        logger.info(
            "Launched campaign '%s': %d requests sent",
            campaign.campaign_name, sent_count,
        )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000)
        result = DataCollectionResult(
            operation="launch_campaign",
            campaign=campaign,
            progress=campaign.progress,
            warnings=warnings,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def close_campaign(
        self,
        campaign_id: str,
    ) -> DataCollectionResult:
        """Close a campaign.  Marks it COMPLETED if all accepted, else CLOSED.

        Args:
            campaign_id: Campaign ID.

        Returns:
            DataCollectionResult with updated campaign.

        Raises:
            KeyError: If campaign not found.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []

        campaign = self._get_campaign(campaign_id)
        self._recalculate_progress(campaign)

        # Determine final status.
        all_accepted = campaign.progress.accepted_count == campaign.progress.total_requests
        if all_accepted and campaign.progress.total_requests > 0:
            campaign.status = CampaignStatus.COMPLETED
        else:
            campaign.status = CampaignStatus.CLOSED
            incomplete = (
                campaign.progress.total_requests - campaign.progress.accepted_count
                - campaign.progress.cancelled_count
            )
            if incomplete > 0:
                warnings.append(
                    f"{incomplete} request(s) were not accepted at campaign close"
                )

        logger.info(
            "Campaign '%s' closed with status %s",
            campaign.campaign_name, campaign.status.value,
        )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000)
        result = DataCollectionResult(
            operation="close_campaign",
            campaign=campaign,
            progress=campaign.progress,
            warnings=warnings,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Public API -- Data Request Management
    # ------------------------------------------------------------------

    def add_request(
        self,
        campaign_id: str,
        scope: DataScope = DataScope.ALL,
        category: str = "",
        facility_id: str = "",
        facility_name: str = "",
        entity_id: str = "",
        assigned_to: str = "",
        due_date: Optional[date] = None,
        data_fields_requested: Optional[List[str]] = None,
        instructions: str = "",
        created_by: str = "system",
    ) -> DataCollectionResult:
        """Add a data request to a campaign.

        Args:
            campaign_id: Parent campaign ID.
            scope: GHG scope for the request.
            category: Emission category.
            facility_id: Target facility.
            facility_name: Target facility name.
            entity_id: Target entity.
            assigned_to: Data owner.
            due_date: Submission deadline.
            data_fields_requested: Specific data fields expected.
            instructions: Instructions for the data owner.
            created_by: User creating the request.

        Returns:
            DataCollectionResult with the new request.

        Raises:
            KeyError: If campaign not found.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []

        campaign = self._get_campaign(campaign_id)

        # Default due date to campaign end date if not specified.
        effective_due = due_date or campaign.end_date

        request = DataRequest(
            campaign_id=campaign_id,
            scope=scope,
            category=category,
            facility_id=facility_id,
            facility_name=facility_name,
            entity_id=entity_id,
            assigned_to=assigned_to,
            assigned_at=_utcnow() if assigned_to else None,
            due_date=effective_due,
            data_fields_requested=data_fields_requested or [],
            instructions=instructions,
            created_by=created_by,
        )

        # Auto-send if campaign is already active.
        if campaign.status == CampaignStatus.ACTIVE:
            request.status = RequestStatus.SENT

        campaign.requests.append(request)
        self._recalculate_progress(campaign)

        logger.info(
            "Added request for %s/%s to campaign '%s' [%s]",
            scope.value, category or "all", campaign.campaign_name,
            request.request_id,
        )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000)
        result = DataCollectionResult(
            operation="add_request",
            campaign=campaign,
            request=request,
            progress=campaign.progress,
            warnings=warnings,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def assign_request(
        self,
        campaign_id: str,
        request_id: str,
        assigned_to: str,
        due_date: Optional[date] = None,
    ) -> DataCollectionResult:
        """Assign (or reassign) a data request to a data owner.

        Args:
            campaign_id: Campaign ID.
            request_id: Request ID to assign.
            assigned_to: Data owner to assign to.
            due_date: Optional updated due date.

        Returns:
            DataCollectionResult with updated request.

        Raises:
            KeyError: If campaign or request not found.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []

        campaign = self._get_campaign(campaign_id)
        request = self._find_request(campaign, request_id)

        old_assignee = request.assigned_to
        request.assigned_to = assigned_to
        request.assigned_at = _utcnow()
        if due_date:
            request.due_date = due_date

        if old_assignee and old_assignee != assigned_to:
            warnings.append(
                f"Request reassigned from '{old_assignee}' to '{assigned_to}'"
            )

        logger.info(
            "Assigned request %s to '%s'", request_id, assigned_to,
        )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000)
        result = DataCollectionResult(
            operation="assign_request",
            campaign=campaign,
            request=request,
            progress=campaign.progress,
            warnings=warnings,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Public API -- Submission & Validation
    # ------------------------------------------------------------------

    def submit_data(
        self,
        campaign_id: str,
        request_id: str,
        data_payload: Dict[str, Any],
        submitted_by: str = "",
        file_references: Optional[List[str]] = None,
    ) -> DataCollectionResult:
        """Submit data for a request and run automated validation.

        Args:
            campaign_id: Campaign ID.
            request_id: Request ID.
            data_payload: Submitted data as key-value pairs.
            submitted_by: User submitting the data.
            file_references: Optional uploaded file references.

        Returns:
            DataCollectionResult with submission and validation findings.

        Raises:
            KeyError: If campaign or request not found.
            ValueError: If request status does not allow submission.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []

        campaign = self._get_campaign(campaign_id)
        request = self._find_request(campaign, request_id)

        # Validate that request is in a submittable state.
        submittable_statuses = {
            RequestStatus.SENT,
            RequestStatus.ACKNOWLEDGED,
            RequestStatus.IN_PROGRESS,
            RequestStatus.REJECTED,  # allow resubmission after rejection
            RequestStatus.OVERDUE,   # allow late submission
        }
        if request.status not in submittable_statuses:
            raise ValueError(
                f"Request in status '{request.status.value}' cannot receive submissions"
            )

        # Run validation.
        findings = self._validate_submission(data_payload, request.data_fields_requested)
        has_errors = any(f.severity == ValidationSeverity.ERROR for f in findings)

        submission = DataSubmission(
            request_id=request_id,
            submitted_by=submitted_by,
            data_payload=data_payload,
            file_references=file_references or [],
            validation_findings=findings,
            validation_passed=not has_errors,
        )

        request.submissions.append(submission)

        # Update request status based on validation.
        if has_errors:
            request.status = RequestStatus.SUBMITTED
            warnings.append(
                f"Submission has {sum(1 for f in findings if f.severity == ValidationSeverity.ERROR)} "
                f"validation error(s) that must be resolved"
            )
        else:
            request.status = RequestStatus.VALIDATED

        self._recalculate_progress(campaign)

        logger.info(
            "Data submitted for request %s (validation: %s)",
            request_id, "PASS" if not has_errors else "FAIL",
        )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000)
        result = DataCollectionResult(
            operation="submit_data",
            campaign=campaign,
            request=request,
            submission=submission,
            progress=campaign.progress,
            warnings=warnings,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def accept_submission(
        self,
        campaign_id: str,
        request_id: str,
        submission_id: str,
        accepted_by: str = "system",
        reviewer_notes: str = "",
    ) -> DataCollectionResult:
        """Accept a data submission, marking the request as ACCEPTED.

        Args:
            campaign_id: Campaign ID.
            request_id: Request ID.
            submission_id: Submission ID to accept.
            accepted_by: User accepting.
            reviewer_notes: Notes from the reviewer.

        Returns:
            DataCollectionResult with accepted request.

        Raises:
            KeyError: If campaign, request, or submission not found.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []

        campaign = self._get_campaign(campaign_id)
        request = self._find_request(campaign, request_id)
        submission = self._find_submission(request, submission_id)

        submission.accepted = True
        submission.accepted_at = _utcnow()
        submission.accepted_by = accepted_by
        submission.reviewer_notes = reviewer_notes

        request.status = RequestStatus.ACCEPTED
        self._recalculate_progress(campaign)

        # Check if all requests are now accepted -> move campaign to IN_REVIEW.
        if (
            campaign.status == CampaignStatus.ACTIVE
            and campaign.progress.accepted_count == campaign.progress.total_requests
        ):
            campaign.status = CampaignStatus.IN_REVIEW

        logger.info(
            "Submission %s accepted for request %s by %s",
            submission_id, request_id, accepted_by,
        )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000)
        result = DataCollectionResult(
            operation="accept_submission",
            campaign=campaign,
            request=request,
            submission=submission,
            progress=campaign.progress,
            warnings=warnings,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def reject_submission(
        self,
        campaign_id: str,
        request_id: str,
        submission_id: str,
        rejected_by: str = "system",
        reason: str = "",
    ) -> DataCollectionResult:
        """Reject a data submission, requiring resubmission.

        Args:
            campaign_id: Campaign ID.
            request_id: Request ID.
            submission_id: Submission ID to reject.
            rejected_by: User rejecting.
            reason: Reason for rejection.

        Returns:
            DataCollectionResult with rejected request.

        Raises:
            KeyError: If campaign, request, or submission not found.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []

        campaign = self._get_campaign(campaign_id)
        request = self._find_request(campaign, request_id)
        submission = self._find_submission(request, submission_id)

        submission.accepted = False
        submission.reviewer_notes = reason

        request.status = RequestStatus.REJECTED
        self._recalculate_progress(campaign)

        logger.info(
            "Submission %s rejected for request %s: %s",
            submission_id, request_id, reason,
        )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000)
        result = DataCollectionResult(
            operation="reject_submission",
            campaign=campaign,
            request=request,
            submission=submission,
            progress=campaign.progress,
            warnings=warnings,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Public API -- Reminders & Escalation
    # ------------------------------------------------------------------

    def generate_reminders(
        self,
        campaign_id: str,
    ) -> DataCollectionResult:
        """Generate reminders for overdue or approaching-deadline requests.

        Scans all active requests in the campaign and generates reminder
        records based on due date proximity and escalation rules.

        Args:
            campaign_id: Campaign ID.

        Returns:
            DataCollectionResult with generated reminders.

        Raises:
            KeyError: If campaign not found.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []
        reminders_generated: List[ReminderRecord] = []

        campaign = self._get_campaign(campaign_id)
        today = _today()

        active_statuses = {
            RequestStatus.SENT,
            RequestStatus.ACKNOWLEDGED,
            RequestStatus.IN_PROGRESS,
        }

        for req in campaign.requests:
            if req.status not in active_statuses:
                continue
            if req.due_date is None:
                continue

            days_until_due = (req.due_date - today).days

            # Determine escalation level.
            if days_until_due < 0:
                days_overdue = abs(days_until_due)
                if days_overdue <= 7:
                    level = EscalationLevel.LEVEL_1
                    msg = (
                        f"Data request for {req.category} at {req.facility_name} "
                        f"is {days_overdue} day(s) overdue. Please submit urgently."
                    )
                elif days_overdue <= 14:
                    level = EscalationLevel.LEVEL_2
                    msg = (
                        f"ESCALATION: Data request for {req.category} at "
                        f"{req.facility_name} is {days_overdue} days overdue."
                    )
                else:
                    level = EscalationLevel.LEVEL_3
                    msg = (
                        f"CRITICAL: Data request for {req.category} at "
                        f"{req.facility_name} is {days_overdue} days overdue. "
                        f"Management intervention required."
                    )

                # Mark request as overdue if not already.
                if req.status != RequestStatus.OVERDUE:
                    req.status = RequestStatus.OVERDUE

                req.escalation_level = level

            elif days_until_due <= 3:
                # Approaching deadline reminder.
                level = EscalationLevel.LEVEL_1
                msg = (
                    f"Reminder: Data request for {req.category} at "
                    f"{req.facility_name} is due in {days_until_due} day(s)."
                )
            else:
                continue  # Not yet approaching deadline.

            reminder = ReminderRecord(
                request_id=req.request_id,
                escalation_level=level,
                recipient=req.assigned_to,
                message=msg,
            )
            req.reminders.append(reminder)
            reminders_generated.append(reminder)

        self._recalculate_progress(campaign)

        if reminders_generated:
            logger.info(
                "Generated %d reminders for campaign '%s'",
                len(reminders_generated), campaign.campaign_name,
            )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000)
        result = DataCollectionResult(
            operation="generate_reminders",
            campaign=campaign,
            reminders_generated=reminders_generated,
            progress=campaign.progress,
            warnings=warnings,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Public API -- Coverage Analysis
    # ------------------------------------------------------------------

    def calculate_coverage(
        self,
        campaign_id: str,
    ) -> DataCollectionResult:
        """Calculate data coverage for a campaign.

        Computes the percentage of expected data points that have been
        collected across all requests in the campaign.

        Args:
            campaign_id: Campaign ID.

        Returns:
            DataCollectionResult with updated coverage metrics.

        Raises:
            KeyError: If campaign not found.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []

        campaign = self._get_campaign(campaign_id)
        self._recalculate_progress(campaign)

        # Calculate detailed coverage by scope.
        scope_coverage: Dict[str, Dict[str, int]] = {}
        for req in campaign.requests:
            scope_key = req.scope.value
            if scope_key not in scope_coverage:
                scope_coverage[scope_key] = {"expected": 0, "collected": 0}
            expected_fields = len(req.data_fields_requested) or 1
            scope_coverage[scope_key]["expected"] += expected_fields
            if req.status in (RequestStatus.VALIDATED, RequestStatus.ACCEPTED):
                scope_coverage[scope_key]["collected"] += expected_fields

        for scope_key, counts in scope_coverage.items():
            pct = _safe_pct(
                _decimal(counts["collected"]), _decimal(counts["expected"])
            )
            if pct < Decimal("80"):
                warnings.append(
                    f"{scope_key} coverage at {_round_val(pct, 1)}% "
                    f"(below 80% threshold)"
                )

        logger.info(
            "Coverage analysis for campaign '%s': %s%%",
            campaign.campaign_name, campaign.progress.coverage_pct,
        )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000)
        result = DataCollectionResult(
            operation="calculate_coverage",
            campaign=campaign,
            progress=campaign.progress,
            warnings=warnings,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Public API -- Retrieval
    # ------------------------------------------------------------------

    def get_campaign(self, campaign_id: str) -> CollectionCampaign:
        """Retrieve a campaign by ID.

        Args:
            campaign_id: Campaign identifier.

        Returns:
            The CollectionCampaign.

        Raises:
            KeyError: If campaign not found.
        """
        return self._get_campaign(campaign_id)

    def list_campaigns(
        self,
        organisation_id: Optional[str] = None,
        period_id: Optional[str] = None,
    ) -> List[CollectionCampaign]:
        """List campaigns, optionally filtered.

        Args:
            organisation_id: Filter by organisation.
            period_id: Filter by inventory period.

        Returns:
            List of matching CollectionCampaign objects.
        """
        results: List[CollectionCampaign] = []
        for campaign in self._campaigns.values():
            if organisation_id and campaign.organisation_id != organisation_id:
                continue
            if period_id and campaign.period_id != period_id:
                continue
            results.append(campaign)
        return results

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _get_campaign(self, campaign_id: str) -> CollectionCampaign:
        """Retrieve a campaign by ID or raise KeyError."""
        if campaign_id not in self._campaigns:
            raise KeyError(f"Campaign not found: {campaign_id}")
        return self._campaigns[campaign_id]

    def _find_request(
        self,
        campaign: CollectionCampaign,
        request_id: str,
    ) -> DataRequest:
        """Find a request within a campaign or raise KeyError."""
        for req in campaign.requests:
            if req.request_id == request_id:
                return req
        raise KeyError(
            f"Request {request_id} not found in campaign {campaign.campaign_id}"
        )

    def _find_submission(
        self,
        request: DataRequest,
        submission_id: str,
    ) -> DataSubmission:
        """Find a submission within a request or raise KeyError."""
        for sub in request.submissions:
            if sub.submission_id == submission_id:
                return sub
        raise KeyError(
            f"Submission {submission_id} not found in request {request.request_id}"
        )

    def _validate_submission(
        self,
        data_payload: Dict[str, Any],
        expected_fields: List[str],
    ) -> List[ValidationFinding]:
        """Run automated validation checks on submitted data.

        Checks:
        1. All expected fields are present.
        2. Values are within expected ranges (if defined).
        3. No negative values for quantity fields.

        Args:
            data_payload: Submitted data.
            expected_fields: List of expected field names.

        Returns:
            List of ValidationFinding objects.
        """
        findings: List[ValidationFinding] = []

        # Check for missing expected fields.
        for field_name in expected_fields:
            if field_name not in data_payload:
                findings.append(ValidationFinding(
                    field_name=field_name,
                    severity=ValidationSeverity.ERROR,
                    message=f"Required field '{field_name}' is missing",
                ))

        # Validate values against known ranges.
        for field_name, value in data_payload.items():
            # Skip non-numeric values.
            try:
                numeric_val = float(value)
            except (TypeError, ValueError):
                continue

            # Check for negative values.
            if numeric_val < 0:
                findings.append(ValidationFinding(
                    field_name=field_name,
                    severity=ValidationSeverity.ERROR,
                    message=f"Field '{field_name}' has negative value: {value}",
                    actual_value=str(value),
                ))
                continue

            # Check against known ranges.
            if field_name in DEFAULT_VALIDATION_RANGES:
                min_val, max_val, unit = DEFAULT_VALIDATION_RANGES[field_name]
                if numeric_val < min_val or numeric_val > max_val:
                    findings.append(ValidationFinding(
                        field_name=field_name,
                        severity=ValidationSeverity.WARNING,
                        message=(
                            f"Field '{field_name}' value {value} outside "
                            f"expected range [{min_val}, {max_val}] {unit}"
                        ),
                        expected_range=f"[{min_val}, {max_val}] {unit}",
                        actual_value=str(value),
                    ))

        return findings

    def _recalculate_progress(self, campaign: CollectionCampaign) -> None:
        """Recalculate progress metrics for a campaign.

        Updates the campaign's progress object with current counts and
        percentages based on request statuses.

        Args:
            campaign: The campaign to recalculate.
        """
        total = len(campaign.requests)
        if total == 0:
            campaign.progress = CollectionProgress()
            return

        status_counts: Dict[RequestStatus, int] = {}
        for req in campaign.requests:
            status_counts[req.status] = status_counts.get(req.status, 0) + 1

        accepted = status_counts.get(RequestStatus.ACCEPTED, 0)
        validated = status_counts.get(RequestStatus.VALIDATED, 0)
        submitted = status_counts.get(RequestStatus.SUBMITTED, 0)
        cancelled = status_counts.get(RequestStatus.CANCELLED, 0)
        overdue = status_counts.get(RequestStatus.OVERDUE, 0)
        rejected = status_counts.get(RequestStatus.REJECTED, 0)

        # Calculate data points.
        expected_points = sum(
            len(req.data_fields_requested) or 1 for req in campaign.requests
        )
        collected_points = sum(
            len(req.data_fields_requested) or 1
            for req in campaign.requests
            if req.status in (RequestStatus.VALIDATED, RequestStatus.ACCEPTED)
        )

        # On-time calculation: accepted requests submitted before due date.
        total_completed = accepted + validated
        on_time_count = 0
        for req in campaign.requests:
            if req.status not in (RequestStatus.ACCEPTED, RequestStatus.VALIDATED):
                continue
            if req.due_date and req.submissions:
                last_sub = req.submissions[-1]
                if last_sub.submitted_at.date() <= req.due_date:
                    on_time_count += 1
                else:
                    on_time_count += 0  # explicit for clarity
            else:
                on_time_count += 1  # no due date means on-time by default

        campaign.progress = CollectionProgress(
            total_requests=total,
            draft_count=status_counts.get(RequestStatus.DRAFT, 0),
            sent_count=(
                status_counts.get(RequestStatus.SENT, 0)
                + status_counts.get(RequestStatus.ACKNOWLEDGED, 0)
                + status_counts.get(RequestStatus.IN_PROGRESS, 0)
            ),
            submitted_count=submitted,
            validated_count=validated,
            accepted_count=accepted,
            rejected_count=rejected,
            overdue_count=overdue,
            cancelled_count=cancelled,
            progress_pct=_round_val(_safe_pct(_decimal(accepted), _decimal(total)), 2),
            coverage_pct=_round_val(
                _safe_pct(_decimal(collected_points), _decimal(expected_points)), 2
            ),
            on_time_pct=_round_val(
                _safe_pct(_decimal(on_time_count), _decimal(total_completed))
                if total_completed > 0 else Decimal("0"),
                2,
            ),
            expected_data_points=expected_points,
            collected_data_points=collected_points,
        )
