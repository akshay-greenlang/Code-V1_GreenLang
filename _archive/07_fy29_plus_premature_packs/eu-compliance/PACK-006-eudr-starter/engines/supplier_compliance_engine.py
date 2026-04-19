# -*- coding: utf-8 -*-
"""
SupplierComplianceEngine - PACK-006 EUDR Starter Engine 5
===========================================================

Supplier due diligence status tracking engine for EUDR compliance.
Manages supplier profiles, tracks due diligence completion lifecycle,
calculates data completeness scores, and manages certification
validity across the supply chain.

Key Capabilities:
    - Supplier registration and profile management
    - Due diligence status lifecycle tracking
    - Data completeness scoring across 12+ required fields
    - Certification tracking and validity checks
    - Supplier prioritization by risk and completeness
    - Data request generation for incomplete suppliers
    - Engagement event tracking
    - Compliance calendar management
    - Supplier dashboard aggregation

DD Status Lifecycle:
    NOT_STARTED -> IN_PROGRESS -> COMPLETE -> VERIFIED -> EXPIRED

Zero-Hallucination:
    - All scoring uses deterministic completeness formulas
    - No LLM involvement in any status or scoring path
    - SHA-256 provenance hashing on every output
    - Pydantic validation at all input/output boundaries

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-006 EUDR Starter
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SupplierDDStatus(str, Enum):
    """Due diligence status lifecycle."""

    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETE = "COMPLETE"
    VERIFIED = "VERIFIED"
    EXPIRED = "EXPIRED"

class SupplierTier(str, Enum):
    """Supply chain tier classification."""

    TIER_1 = "TIER_1"
    TIER_2 = "TIER_2"
    TIER_3 = "TIER_3"
    TIER_4_PLUS = "TIER_4_PLUS"

class EngagementType(str, Enum):
    """Types of supplier engagement events."""

    INITIAL_CONTACT = "INITIAL_CONTACT"
    DATA_REQUEST = "DATA_REQUEST"
    DATA_RECEIVED = "DATA_RECEIVED"
    AUDIT = "AUDIT"
    CERTIFICATION_REVIEW = "CERTIFICATION_REVIEW"
    CORRECTIVE_ACTION = "CORRECTIVE_ACTION"
    FOLLOW_UP = "FOLLOW_UP"
    SITE_VISIT = "SITE_VISIT"
    TRAINING = "TRAINING"
    TERMINATION = "TERMINATION"

class PriorityLevel(str, Enum):
    """Supplier action priority levels."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class CertificationStatus(str, Enum):
    """Certification validity status."""

    VALID = "VALID"
    EXPIRING_SOON = "EXPIRING_SOON"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"
    NOT_VERIFIED = "NOT_VERIFIED"

# ---------------------------------------------------------------------------
# Required Fields for Completeness Scoring
# ---------------------------------------------------------------------------

REQUIRED_SUPPLIER_FIELDS: List[Dict[str, Any]] = [
    {"field": "legal_name", "weight": 10, "description": "Legal entity name"},
    {"field": "address", "weight": 8, "description": "Registered address"},
    {"field": "country", "weight": 10, "description": "Country of operation"},
    {"field": "eori_number", "weight": 7, "description": "EORI number (EU operators)"},
    {"field": "contact_person", "weight": 5, "description": "Primary contact person"},
    {"field": "contact_email", "weight": 5, "description": "Contact email address"},
    {"field": "commodities", "weight": 10, "description": "Commodities supplied"},
    {"field": "source_countries", "weight": 10, "description": "Countries of production/sourcing"},
    {"field": "geolocation_data", "weight": 12, "description": "Geolocation of production plots"},
    {"field": "certifications", "weight": 8, "description": "Active certifications"},
    {"field": "risk_assessment", "weight": 8, "description": "Risk assessment completed"},
    {"field": "dd_declaration", "weight": 7, "description": "DD declaration signed"},
]

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class SupplierProfile(BaseModel):
    """Complete supplier profile for EUDR compliance."""

    supplier_id: str = Field(default_factory=_new_uuid, description="Unique supplier identifier")
    legal_name: str = Field(..., description="Legal entity name")
    trade_name: Optional[str] = Field(None, description="Trading/brand name")
    address: Optional[str] = Field(None, description="Registered address")
    country: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    eori_number: Optional[str] = Field(None, description="EORI number")
    contact_person: Optional[str] = Field(None, description="Primary contact name")
    contact_email: Optional[str] = Field(None, description="Contact email")
    contact_phone: Optional[str] = Field(None, description="Contact phone")
    tier: int = Field(default=1, ge=1, le=10, description="Supply chain tier")
    commodities: List[str] = Field(default_factory=list, description="EUDR commodities supplied")
    source_countries: List[str] = Field(default_factory=list, description="Countries of sourcing")
    certifications: List[str] = Field(default_factory=list, description="Active certifications")
    dd_status: SupplierDDStatus = Field(
        default=SupplierDDStatus.NOT_STARTED, description="DD status"
    )
    risk_score: Optional[float] = Field(None, ge=0, le=100, description="Latest risk score")
    geolocation_data: Optional[Dict[str, Any]] = Field(None, description="Geolocation data")
    risk_assessment: Optional[Dict[str, Any]] = Field(None, description="Risk assessment data")
    dd_declaration: Optional[Dict[str, Any]] = Field(None, description="DD declaration data")
    registered_at: datetime = Field(default_factory=utcnow, description="Registration timestamp")
    updated_at: datetime = Field(default_factory=utcnow, description="Last update timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class DDStatusUpdate(BaseModel):
    """Result of a DD status update operation."""

    supplier_id: str = Field(..., description="Supplier identifier")
    previous_status: SupplierDDStatus = Field(..., description="Previous DD status")
    new_status: SupplierDDStatus = Field(..., description="Updated DD status")
    is_valid_transition: bool = Field(default=True, description="Whether transition is valid")
    transition_reason: Optional[str] = Field(None, description="Reason for transition")
    updated_at: datetime = Field(default_factory=utcnow, description="Update timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class CompletenessScore(BaseModel):
    """Data completeness score for a supplier."""

    supplier_id: str = Field(..., description="Supplier identifier")
    overall_score: float = Field(..., ge=0, le=100, description="Overall completeness 0-100")
    total_fields: int = Field(default=0, description="Total required fields")
    completed_fields: int = Field(default=0, description="Completed fields")
    missing_fields: List[str] = Field(default_factory=list, description="Missing field names")
    field_scores: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-field scoring"
    )
    is_sufficient_for_dd: bool = Field(default=False, description="Whether data is DD-sufficient")
    scored_at: datetime = Field(default_factory=utcnow, description="Scoring timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class CertificationRecord(BaseModel):
    """Certification record for a supplier."""

    record_id: str = Field(default_factory=_new_uuid, description="Record identifier")
    supplier_id: str = Field(..., description="Supplier identifier")
    certification_name: str = Field(..., description="Certification name")
    certification_body: Optional[str] = Field(None, description="Issuing certification body")
    certificate_number: Optional[str] = Field(None, description="Certificate number")
    issue_date: Optional[datetime] = Field(None, description="Issue date")
    expiry_date: Optional[datetime] = Field(None, description="Expiry date")
    status: CertificationStatus = Field(
        default=CertificationStatus.NOT_VERIFIED, description="Certification status"
    )
    scope: Optional[str] = Field(None, description="Certification scope")
    commodities_covered: List[str] = Field(default_factory=list, description="Commodities covered")
    recorded_at: datetime = Field(default_factory=utcnow, description="Record timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class CertValidation(BaseModel):
    """Certification validity check result."""

    certification_name: str = Field(..., description="Certification name")
    status: CertificationStatus = Field(..., description="Current status")
    is_valid: bool = Field(default=False, description="Whether currently valid")
    days_until_expiry: Optional[int] = Field(None, description="Days until expiry")
    expiry_date: Optional[datetime] = Field(None, description="Expiry date")
    issues: List[str] = Field(default_factory=list, description="Validation issues")

class PrioritizedSupplier(BaseModel):
    """Supplier with priority ranking for DD actions."""

    supplier_id: str = Field(..., description="Supplier identifier")
    supplier_name: str = Field(default="", description="Supplier name")
    priority: PriorityLevel = Field(..., description="Action priority")
    priority_score: float = Field(default=0.0, description="Numeric priority score")
    risk_score: float = Field(default=0.0, description="Risk score")
    completeness_score: float = Field(default=0.0, description="Data completeness")
    dd_status: SupplierDDStatus = Field(
        default=SupplierDDStatus.NOT_STARTED, description="DD status"
    )
    recommended_actions: List[str] = Field(
        default_factory=list, description="Recommended actions"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class DataRequest(BaseModel):
    """Data request generated for a supplier."""

    request_id: str = Field(default_factory=_new_uuid, description="Request identifier")
    supplier_id: str = Field(..., description="Target supplier identifier")
    supplier_name: str = Field(default="", description="Supplier name")
    missing_data: List[Dict[str, str]] = Field(
        default_factory=list, description="Missing data items"
    )
    priority: PriorityLevel = Field(default=PriorityLevel.MEDIUM, description="Request priority")
    deadline: Optional[datetime] = Field(None, description="Response deadline")
    generated_at: datetime = Field(default_factory=utcnow, description="Generation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class EngagementRecord(BaseModel):
    """Record of a supplier engagement event."""

    record_id: str = Field(default_factory=_new_uuid, description="Record identifier")
    supplier_id: str = Field(..., description="Supplier identifier")
    event_type: EngagementType = Field(..., description="Type of engagement")
    description: str = Field(default="", description="Event description")
    participants: List[str] = Field(default_factory=list, description="Participants")
    outcome: Optional[str] = Field(None, description="Event outcome")
    follow_up_required: bool = Field(default=False, description="Whether follow-up needed")
    follow_up_date: Optional[datetime] = Field(None, description="Follow-up due date")
    recorded_at: datetime = Field(default_factory=utcnow, description="Record timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class ComplianceCalendarEntry(BaseModel):
    """A single compliance calendar entry."""

    entry_id: str = Field(default_factory=_new_uuid, description="Entry identifier")
    title: str = Field(..., description="Calendar entry title")
    description: str = Field(default="", description="Entry description")
    due_date: datetime = Field(..., description="Due date")
    category: str = Field(default="", description="Entry category")
    is_overdue: bool = Field(default=False, description="Whether entry is overdue")
    days_remaining: int = Field(default=0, description="Days until due")

class ComplianceCalendar(BaseModel):
    """Compliance calendar for a supplier."""

    supplier_id: str = Field(..., description="Supplier identifier")
    entries: List[ComplianceCalendarEntry] = Field(
        default_factory=list, description="Calendar entries"
    )
    overdue_count: int = Field(default=0, description="Number of overdue items")
    upcoming_30d_count: int = Field(default=0, description="Items due in next 30 days")
    generated_at: datetime = Field(default_factory=utcnow, description="Generation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class SupplierDashboard(BaseModel):
    """Aggregated dashboard for a supplier."""

    supplier_id: str = Field(..., description="Supplier identifier")
    supplier_name: str = Field(default="", description="Supplier name")
    dd_status: SupplierDDStatus = Field(
        default=SupplierDDStatus.NOT_STARTED, description="DD status"
    )
    completeness_score: float = Field(default=0.0, description="Data completeness")
    risk_score: Optional[float] = Field(None, description="Risk score")
    active_certifications: int = Field(default=0, description="Active certification count")
    expiring_certifications: int = Field(default=0, description="Certifications expiring soon")
    pending_actions: int = Field(default=0, description="Pending action items")
    last_engagement: Optional[datetime] = Field(None, description="Last engagement date")
    compliance_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Compliance summary"
    )
    generated_at: datetime = Field(default_factory=utcnow, description="Generation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# ---------------------------------------------------------------------------
# Valid Status Transitions
# ---------------------------------------------------------------------------

VALID_TRANSITIONS: Dict[SupplierDDStatus, List[SupplierDDStatus]] = {
    SupplierDDStatus.NOT_STARTED: [SupplierDDStatus.IN_PROGRESS],
    SupplierDDStatus.IN_PROGRESS: [SupplierDDStatus.COMPLETE, SupplierDDStatus.NOT_STARTED],
    SupplierDDStatus.COMPLETE: [SupplierDDStatus.VERIFIED, SupplierDDStatus.IN_PROGRESS],
    SupplierDDStatus.VERIFIED: [SupplierDDStatus.EXPIRED, SupplierDDStatus.IN_PROGRESS],
    SupplierDDStatus.EXPIRED: [SupplierDDStatus.IN_PROGRESS],
}

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SupplierComplianceEngine:
    """
    Supplier Due Diligence Status Tracking Engine.

    Manages supplier profiles, tracks due diligence completion lifecycle
    (NOT_STARTED -> IN_PROGRESS -> COMPLETE -> VERIFIED -> EXPIRED),
    calculates data completeness scores, and manages certification validity.

    All scoring and status transitions are deterministic with complete
    provenance tracking. No LLM involvement in any computation path.

    Attributes:
        config: Optional engine configuration
        _suppliers: In-memory supplier store
        _certifications: In-memory certification store
        _engagements: In-memory engagement store

    Example:
        >>> engine = SupplierComplianceEngine()
        >>> profile = engine.register_supplier({"legal_name": "AcmeCo", "country": "BR"})
        >>> score = engine.calculate_data_completeness(profile.supplier_id)
        >>> assert 0 <= score.overall_score <= 100
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize SupplierComplianceEngine.

        Args:
            config: Optional configuration dictionary with keys:
                - dd_expiry_days: Days until DD verification expires (default: 365)
                - cert_expiry_warning_days: Warning days before cert expiry (default: 60)
                - min_completeness_for_dd: Minimum completeness for DD (default: 70)
        """
        self.config = config or {}
        self._suppliers: Dict[str, SupplierProfile] = {}
        self._certifications: Dict[str, List[CertificationRecord]] = {}
        self._engagements: Dict[str, List[EngagementRecord]] = {}
        self._dd_expiry_days: int = self.config.get("dd_expiry_days", 365)
        self._cert_warning_days: int = self.config.get("cert_expiry_warning_days", 60)
        self._min_completeness: float = self.config.get("min_completeness_for_dd", 70.0)
        logger.info("SupplierComplianceEngine initialized (version=%s)", _MODULE_VERSION)

    # -------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------

    def register_supplier(self, supplier_data: Dict[str, Any]) -> SupplierProfile:
        """Register a new supplier in the compliance system.

        Args:
            supplier_data: Dictionary with supplier information:
                - legal_name (required), country (required)
                - address, eori_number, contact_person, contact_email
                - commodities (list), source_countries (list)
                - certifications (list), tier (int)

        Returns:
            SupplierProfile with assigned ID and initial status.

        Raises:
            ValueError: If required fields are missing.
        """
        legal_name = supplier_data.get("legal_name")
        country = supplier_data.get("country")

        if not legal_name:
            raise ValueError("legal_name is required for supplier registration")
        if not country:
            raise ValueError("country is required for supplier registration")

        profile = SupplierProfile(
            legal_name=legal_name,
            trade_name=supplier_data.get("trade_name"),
            address=supplier_data.get("address"),
            country=country.upper(),
            eori_number=supplier_data.get("eori_number"),
            contact_person=supplier_data.get("contact_person"),
            contact_email=supplier_data.get("contact_email"),
            contact_phone=supplier_data.get("contact_phone"),
            tier=supplier_data.get("tier", 1),
            commodities=supplier_data.get("commodities", []),
            source_countries=[c.upper() for c in supplier_data.get("source_countries", [])],
            certifications=supplier_data.get("certifications", []),
            dd_status=SupplierDDStatus.NOT_STARTED,
            geolocation_data=supplier_data.get("geolocation_data"),
            risk_assessment=supplier_data.get("risk_assessment"),
            dd_declaration=supplier_data.get("dd_declaration"),
        )
        profile.provenance_hash = _compute_hash(profile)

        self._suppliers[profile.supplier_id] = profile
        self._certifications[profile.supplier_id] = []
        self._engagements[profile.supplier_id] = []

        logger.info("Registered supplier: %s (%s)", profile.legal_name, profile.supplier_id)
        return profile

    def update_dd_status(
        self, supplier_id: str, status: str, reason: Optional[str] = None
    ) -> DDStatusUpdate:
        """Update the due diligence status of a supplier.

        Validates that the requested transition follows the allowed lifecycle:
        NOT_STARTED -> IN_PROGRESS -> COMPLETE -> VERIFIED -> EXPIRED

        Args:
            supplier_id: Supplier identifier.
            status: New DD status value.
            reason: Optional reason for the transition.

        Returns:
            DDStatusUpdate with transition details.

        Raises:
            ValueError: If supplier not found or invalid transition.
        """
        supplier = self._suppliers.get(supplier_id)
        if not supplier:
            raise ValueError(f"Supplier not found: {supplier_id}")

        try:
            new_status = SupplierDDStatus(status)
        except ValueError:
            raise ValueError(f"Invalid DD status: {status}")

        previous = supplier.dd_status
        allowed = VALID_TRANSITIONS.get(previous, [])
        is_valid = new_status in allowed

        if is_valid:
            supplier.dd_status = new_status
            supplier.updated_at = utcnow()
            supplier.provenance_hash = _compute_hash(supplier)
            logger.info(
                "DD status updated for %s: %s -> %s",
                supplier_id, previous.value, new_status.value,
            )
        else:
            logger.warning(
                "Invalid DD status transition for %s: %s -> %s",
                supplier_id, previous.value, new_status.value,
            )

        result = DDStatusUpdate(
            supplier_id=supplier_id,
            previous_status=previous,
            new_status=new_status,
            is_valid_transition=is_valid,
            transition_reason=reason,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def calculate_data_completeness(self, supplier_id: str) -> CompletenessScore:
        """Calculate data completeness score for a supplier.

        Evaluates 12 required data fields with weighted scoring. Each field
        contributes to the overall completeness score based on its importance
        weight.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            CompletenessScore with overall and per-field results.

        Raises:
            ValueError: If supplier not found.
        """
        supplier = self._suppliers.get(supplier_id)
        if not supplier:
            raise ValueError(f"Supplier not found: {supplier_id}")

        supplier_dict = supplier.model_dump()
        total_weight = sum(f["weight"] for f in REQUIRED_SUPPLIER_FIELDS)
        achieved_weight = 0
        completed = 0
        missing: List[str] = []
        field_scores: List[Dict[str, Any]] = []

        for field_def in REQUIRED_SUPPLIER_FIELDS:
            field_name = field_def["field"]
            weight = field_def["weight"]
            description = field_def["description"]

            value = supplier_dict.get(field_name)
            is_complete = self._field_has_value(value)

            if is_complete:
                achieved_weight += weight
                completed += 1

            if not is_complete:
                missing.append(field_name)

            field_scores.append({
                "field": field_name,
                "description": description,
                "weight": weight,
                "is_complete": is_complete,
                "score": weight if is_complete else 0,
            })

        overall = round((achieved_weight / total_weight) * 100.0, 2) if total_weight > 0 else 0.0
        is_sufficient = overall >= self._min_completeness

        result = CompletenessScore(
            supplier_id=supplier_id,
            overall_score=overall,
            total_fields=len(REQUIRED_SUPPLIER_FIELDS),
            completed_fields=completed,
            missing_fields=missing,
            field_scores=field_scores,
            is_sufficient_for_dd=is_sufficient,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def track_certification(
        self, supplier_id: str, cert: Dict[str, Any]
    ) -> CertificationRecord:
        """Track a certification for a supplier.

        Args:
            supplier_id: Supplier identifier.
            cert: Certification data dictionary with keys:
                - certification_name (required)
                - certification_body, certificate_number
                - issue_date, expiry_date, scope
                - commodities_covered (list)

        Returns:
            CertificationRecord with assigned ID and status.

        Raises:
            ValueError: If supplier not found or cert name missing.
        """
        if supplier_id not in self._suppliers:
            raise ValueError(f"Supplier not found: {supplier_id}")

        cert_name = cert.get("certification_name")
        if not cert_name:
            raise ValueError("certification_name is required")

        # Determine status based on expiry
        now = utcnow()
        expiry = cert.get("expiry_date")
        status = CertificationStatus.NOT_VERIFIED

        if expiry:
            if isinstance(expiry, str):
                try:
                    expiry = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
                except ValueError:
                    expiry = None

            if isinstance(expiry, datetime):
                if expiry < now:
                    status = CertificationStatus.EXPIRED
                elif (expiry - now).days <= self._cert_warning_days:
                    status = CertificationStatus.EXPIRING_SOON
                else:
                    status = CertificationStatus.VALID

        issue_date = cert.get("issue_date")
        if issue_date and isinstance(issue_date, str):
            try:
                issue_date = datetime.fromisoformat(issue_date.replace("Z", "+00:00"))
            except ValueError:
                issue_date = None

        record = CertificationRecord(
            supplier_id=supplier_id,
            certification_name=cert_name,
            certification_body=cert.get("certification_body"),
            certificate_number=cert.get("certificate_number"),
            issue_date=issue_date if isinstance(issue_date, datetime) else None,
            expiry_date=expiry if isinstance(expiry, datetime) else None,
            status=status,
            scope=cert.get("scope"),
            commodities_covered=cert.get("commodities_covered", []),
        )
        record.provenance_hash = _compute_hash(record)

        if supplier_id not in self._certifications:
            self._certifications[supplier_id] = []
        self._certifications[supplier_id].append(record)

        logger.info(
            "Tracked certification '%s' for supplier %s (status: %s)",
            cert_name, supplier_id, status.value,
        )
        return record

    def check_certification_validity(
        self, supplier_id: str
    ) -> List[CertValidation]:
        """Check the validity of all certifications for a supplier.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            List of CertValidation results for each certification.
        """
        certs = self._certifications.get(supplier_id, [])
        now = utcnow()
        results: List[CertValidation] = []

        for cert in certs:
            issues: List[str] = []
            is_valid = False
            days_until_expiry = None

            if cert.status == CertificationStatus.REVOKED:
                issues.append("Certification has been revoked")
            elif cert.expiry_date:
                delta = cert.expiry_date - now
                days_until_expiry = delta.days

                if days_until_expiry < 0:
                    issues.append(f"Certification expired {abs(days_until_expiry)} days ago")
                elif days_until_expiry <= self._cert_warning_days:
                    issues.append(f"Certification expires in {days_until_expiry} days")
                    is_valid = True
                else:
                    is_valid = True
            else:
                issues.append("No expiry date recorded, cannot verify validity")

            status = cert.status
            if cert.expiry_date and cert.expiry_date < now:
                status = CertificationStatus.EXPIRED
            elif cert.expiry_date and days_until_expiry and days_until_expiry <= self._cert_warning_days:
                status = CertificationStatus.EXPIRING_SOON

            results.append(CertValidation(
                certification_name=cert.certification_name,
                status=status,
                is_valid=is_valid,
                days_until_expiry=days_until_expiry,
                expiry_date=cert.expiry_date,
                issues=issues,
            ))

        return results

    def prioritize_suppliers(
        self, suppliers: List[str], risk_scores: Dict[str, float]
    ) -> List[PrioritizedSupplier]:
        """Prioritize suppliers for DD action based on risk and completeness.

        Higher risk + lower completeness = higher priority.

        Args:
            suppliers: List of supplier IDs to prioritize.
            risk_scores: Dictionary mapping supplier_id to risk score (0-100).

        Returns:
            List of PrioritizedSupplier sorted by priority (highest first).
        """
        results: List[PrioritizedSupplier] = []

        for sid in suppliers:
            supplier = self._suppliers.get(sid)
            if not supplier:
                continue

            risk = risk_scores.get(sid, 50.0)
            try:
                completeness = self.calculate_data_completeness(sid)
                comp_score = completeness.overall_score
            except ValueError:
                comp_score = 0.0

            # Priority score: higher risk + lower completeness = higher priority
            priority_score = (risk * 0.6) + ((100.0 - comp_score) * 0.4)

            # Status multiplier: NOT_STARTED gets higher priority
            status_multiplier = {
                SupplierDDStatus.NOT_STARTED: 1.3,
                SupplierDDStatus.IN_PROGRESS: 1.0,
                SupplierDDStatus.COMPLETE: 0.7,
                SupplierDDStatus.VERIFIED: 0.5,
                SupplierDDStatus.EXPIRED: 1.2,
            }
            priority_score *= status_multiplier.get(supplier.dd_status, 1.0)
            priority_score = round(min(priority_score, 100.0), 2)

            # Classify priority
            if priority_score >= 75:
                priority = PriorityLevel.CRITICAL
            elif priority_score >= 50:
                priority = PriorityLevel.HIGH
            elif priority_score >= 25:
                priority = PriorityLevel.MEDIUM
            else:
                priority = PriorityLevel.LOW

            # Generate recommended actions
            actions = self._generate_recommended_actions(supplier, comp_score, risk)

            item = PrioritizedSupplier(
                supplier_id=sid,
                supplier_name=supplier.legal_name,
                priority=priority,
                priority_score=priority_score,
                risk_score=risk,
                completeness_score=comp_score,
                dd_status=supplier.dd_status,
                recommended_actions=actions,
            )
            item.provenance_hash = _compute_hash(item)
            results.append(item)

        results.sort(key=lambda x: x.priority_score, reverse=True)
        return results

    def generate_data_request(self, supplier_id: str) -> DataRequest:
        """Generate a data request for missing supplier information.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            DataRequest with missing data items and deadline.

        Raises:
            ValueError: If supplier not found.
        """
        supplier = self._suppliers.get(supplier_id)
        if not supplier:
            raise ValueError(f"Supplier not found: {supplier_id}")

        completeness = self.calculate_data_completeness(supplier_id)

        missing_data: List[Dict[str, str]] = []
        for field_name in completeness.missing_fields:
            field_def = next(
                (f for f in REQUIRED_SUPPLIER_FIELDS if f["field"] == field_name),
                None,
            )
            if field_def:
                missing_data.append({
                    "field": field_name,
                    "description": field_def["description"],
                    "priority": "HIGH" if field_def["weight"] >= 10 else "MEDIUM",
                })

        # Set deadline based on DD status
        deadline_days = {
            SupplierDDStatus.NOT_STARTED: 30,
            SupplierDDStatus.IN_PROGRESS: 14,
            SupplierDDStatus.EXPIRED: 7,
        }
        days = deadline_days.get(supplier.dd_status, 30)
        deadline = utcnow() + timedelta(days=days)

        # Priority based on completeness
        if completeness.overall_score < 30:
            priority = PriorityLevel.CRITICAL
        elif completeness.overall_score < 60:
            priority = PriorityLevel.HIGH
        else:
            priority = PriorityLevel.MEDIUM

        request = DataRequest(
            supplier_id=supplier_id,
            supplier_name=supplier.legal_name,
            missing_data=missing_data,
            priority=priority,
            deadline=deadline,
        )
        request.provenance_hash = _compute_hash(request)
        return request

    def track_engagement(
        self, supplier_id: str, event: Dict[str, Any]
    ) -> EngagementRecord:
        """Track an engagement event with a supplier.

        Args:
            supplier_id: Supplier identifier.
            event: Event dictionary with keys:
                - event_type (required), description
                - participants (list), outcome
                - follow_up_required (bool), follow_up_date

        Returns:
            EngagementRecord for the tracked event.

        Raises:
            ValueError: If supplier not found or event_type missing.
        """
        if supplier_id not in self._suppliers:
            raise ValueError(f"Supplier not found: {supplier_id}")

        event_type_str = event.get("event_type")
        if not event_type_str:
            raise ValueError("event_type is required")

        try:
            event_type = EngagementType(event_type_str)
        except ValueError:
            raise ValueError(f"Invalid engagement type: {event_type_str}")

        follow_up_date = event.get("follow_up_date")
        if follow_up_date and isinstance(follow_up_date, str):
            try:
                follow_up_date = datetime.fromisoformat(follow_up_date.replace("Z", "+00:00"))
            except ValueError:
                follow_up_date = None

        record = EngagementRecord(
            supplier_id=supplier_id,
            event_type=event_type,
            description=event.get("description", ""),
            participants=event.get("participants", []),
            outcome=event.get("outcome"),
            follow_up_required=event.get("follow_up_required", False),
            follow_up_date=follow_up_date if isinstance(follow_up_date, datetime) else None,
        )
        record.provenance_hash = _compute_hash(record)

        if supplier_id not in self._engagements:
            self._engagements[supplier_id] = []
        self._engagements[supplier_id].append(record)

        return record

    def get_compliance_calendar(self, supplier_id: str) -> ComplianceCalendar:
        """Get the compliance calendar for a supplier.

        Generates calendar entries for upcoming DD deadlines, certification
        renewals, and scheduled engagement follow-ups.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            ComplianceCalendar with upcoming entries.

        Raises:
            ValueError: If supplier not found.
        """
        supplier = self._suppliers.get(supplier_id)
        if not supplier:
            raise ValueError(f"Supplier not found: {supplier_id}")

        now = utcnow()
        entries: List[ComplianceCalendarEntry] = []

        # Certification renewals
        for cert in self._certifications.get(supplier_id, []):
            if cert.expiry_date:
                days_remaining = (cert.expiry_date - now).days
                entries.append(ComplianceCalendarEntry(
                    title=f"Certification renewal: {cert.certification_name}",
                    description=f"Certificate {cert.certificate_number or 'N/A'} expires",
                    due_date=cert.expiry_date,
                    category="CERTIFICATION",
                    is_overdue=days_remaining < 0,
                    days_remaining=days_remaining,
                ))

        # Engagement follow-ups
        for eng in self._engagements.get(supplier_id, []):
            if eng.follow_up_required and eng.follow_up_date:
                days_remaining = (eng.follow_up_date - now).days
                entries.append(ComplianceCalendarEntry(
                    title=f"Follow-up: {eng.event_type.value}",
                    description=eng.description,
                    due_date=eng.follow_up_date,
                    category="ENGAGEMENT",
                    is_overdue=days_remaining < 0,
                    days_remaining=days_remaining,
                ))

        # DD verification renewal
        if supplier.dd_status == SupplierDDStatus.VERIFIED:
            renewal_date = supplier.updated_at + timedelta(days=self._dd_expiry_days)
            days_remaining = (renewal_date - now).days
            entries.append(ComplianceCalendarEntry(
                title="DD verification renewal required",
                description="Due diligence verification expires and must be renewed",
                due_date=renewal_date,
                category="DD_VERIFICATION",
                is_overdue=days_remaining < 0,
                days_remaining=days_remaining,
            ))

        entries.sort(key=lambda e: e.due_date)
        overdue = sum(1 for e in entries if e.is_overdue)
        upcoming_30d = sum(1 for e in entries if 0 <= e.days_remaining <= 30)

        calendar = ComplianceCalendar(
            supplier_id=supplier_id,
            entries=entries,
            overdue_count=overdue,
            upcoming_30d_count=upcoming_30d,
        )
        calendar.provenance_hash = _compute_hash(calendar)
        return calendar

    def get_supplier_dashboard(self, supplier_id: str) -> SupplierDashboard:
        """Get an aggregated dashboard view for a supplier.

        Combines DD status, completeness, risk, certifications, and
        engagement data into a single dashboard view.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            SupplierDashboard with aggregated supplier data.

        Raises:
            ValueError: If supplier not found.
        """
        supplier = self._suppliers.get(supplier_id)
        if not supplier:
            raise ValueError(f"Supplier not found: {supplier_id}")

        # Calculate completeness
        try:
            completeness = self.calculate_data_completeness(supplier_id)
            comp_score = completeness.overall_score
        except ValueError:
            comp_score = 0.0

        # Check certifications
        cert_validations = self.check_certification_validity(supplier_id)
        active_certs = sum(1 for cv in cert_validations if cv.is_valid)
        expiring_certs = sum(
            1 for cv in cert_validations
            if cv.status == CertificationStatus.EXPIRING_SOON
        )

        # Get calendar for pending actions
        try:
            calendar = self.get_compliance_calendar(supplier_id)
            pending_actions = calendar.overdue_count + calendar.upcoming_30d_count
        except ValueError:
            pending_actions = 0

        # Last engagement
        engagements = self._engagements.get(supplier_id, [])
        last_engagement = engagements[-1].recorded_at if engagements else None

        # Compliance summary
        compliance_summary = {
            "dd_status": supplier.dd_status.value,
            "completeness_score": comp_score,
            "is_dd_sufficient": comp_score >= self._min_completeness,
            "active_certifications": active_certs,
            "total_engagements": len(engagements),
        }

        dashboard = SupplierDashboard(
            supplier_id=supplier_id,
            supplier_name=supplier.legal_name,
            dd_status=supplier.dd_status,
            completeness_score=comp_score,
            risk_score=supplier.risk_score,
            active_certifications=active_certs,
            expiring_certifications=expiring_certs,
            pending_actions=pending_actions,
            last_engagement=last_engagement,
            compliance_summary=compliance_summary,
        )
        dashboard.provenance_hash = _compute_hash(dashboard)
        return dashboard

    # -------------------------------------------------------------------
    # Private: Helpers
    # -------------------------------------------------------------------

    def _field_has_value(self, value: Any) -> bool:
        """Check if a field has a meaningful value.

        Args:
            value: Field value to check.

        Returns:
            True if the value is non-empty.
        """
        if value is None:
            return False
        if isinstance(value, str) and value.strip() == "":
            return False
        if isinstance(value, (list, dict)) and len(value) == 0:
            return False
        return True

    def _generate_recommended_actions(
        self, supplier: SupplierProfile, completeness: float, risk: float
    ) -> List[str]:
        """Generate recommended actions for a supplier.

        Args:
            supplier: Supplier profile.
            completeness: Data completeness score.
            risk: Risk score.

        Returns:
            List of recommended action strings.
        """
        actions: List[str] = []

        if supplier.dd_status == SupplierDDStatus.NOT_STARTED:
            actions.append("Initiate due diligence process")
        if supplier.dd_status == SupplierDDStatus.EXPIRED:
            actions.append("Renew expired due diligence verification")

        if completeness < 50:
            actions.append("Request missing supplier data (completeness below 50%)")
        elif completeness < self._min_completeness:
            actions.append(f"Complete supplier data to reach {self._min_completeness}% threshold")

        if risk >= 75:
            actions.append("Perform enhanced due diligence (critical risk)")
        elif risk >= 50:
            actions.append("Conduct additional risk mitigation measures")

        if not supplier.certifications:
            actions.append("Request sustainability certifications")

        if not supplier.geolocation_data:
            actions.append("Request geolocation data for production plots")

        return actions
