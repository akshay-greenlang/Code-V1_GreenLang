# -*- coding: utf-8 -*-
"""
SupplierManagementEngine - PACK-004 CBAM Readiness Engine 4
=============================================================

Supplier and installation management engine for CBAM compliance. Handles
the complete supplier lifecycle: registration, installation tracking,
emission data request/submission workflow, data quality scoring, and
dashboard aggregation.

Supplier Data Workflow:
    1. Register supplier with profile and EORI
    2. Add production installations with capacity and goods categories
    3. Request emission data for a specific period
    4. Supplier submits emission data per installation
    5. Review and accept/reject submissions with quality scoring
    6. Accepted data flows into quarterly reports

Submission Statuses:
    - DRAFT: Supplier is preparing data
    - SUBMITTED: Data sent for review
    - REVIEWED: Under compliance review
    - ACCEPTED: Data accepted for reporting
    - REJECTED: Data rejected, resubmission required

Data Quality Score (0-100):
    - Completeness: All required fields populated
    - Timeliness: Submitted before deadline
    - Accuracy: Emission intensity within plausible ranges
    - Methodology: Actual vs. default factor usage
    - Consistency: Agreement with historical data

Zero-Hallucination:
    - All scoring uses deterministic arithmetic
    - No LLM involvement in quality assessment
    - SHA-256 provenance hashing on every submission

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-004 CBAM Readiness
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

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

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class VerificationStatus(str, Enum):
    """Supplier verification status."""

    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"

class SubmissionStatus(str, Enum):
    """Emission data submission lifecycle status."""

    DRAFT = "DRAFT"
    SUBMITTED = "SUBMITTED"
    REVIEWED = "REVIEWED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"

class CBAMGoodsCategory(str, Enum):
    """CBAM goods categories (mirrored for self-containment)."""

    CEMENT = "cement"
    IRON_STEEL = "iron_steel"
    ALUMINIUM = "aluminium"
    FERTILIZERS = "fertilizers"
    ELECTRICITY = "electricity"
    HYDROGEN = "hydrogen"

class CalculationMethod(str, Enum):
    """Emission calculation method."""

    ACTUAL = "actual"
    DEFAULT = "default"
    COUNTRY_DEFAULT = "country_default"

# ---------------------------------------------------------------------------
# Plausible emission intensity ranges by goods category (tCO2e/t product)
# Used for data quality scoring
# ---------------------------------------------------------------------------

PLAUSIBLE_INTENSITY_RANGES: Dict[str, Dict[str, float]] = {
    "cement": {"min": 0.3, "max": 1.2},
    "iron_steel": {"min": 0.2, "max": 3.5},
    "aluminium": {"min": 0.3, "max": 12.0},
    "fertilizers": {"min": 0.2, "max": 3.0},
    "electricity": {"min": 0.01, "max": 1.5},
    "hydrogen": {"min": 0.2, "max": 12.0},
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class Installation(BaseModel):
    """Production installation registered under a supplier.

    Represents a physical production facility that manufactures CBAM-covered
    goods. Each installation has specific goods categories, capacity, and
    an emission profile.
    """

    installation_id: str = Field(
        default_factory=_new_uuid,
        description="Unique installation identifier",
    )
    name: str = Field(
        ..., min_length=1, max_length=300,
        description="Installation name",
    )
    country: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    address: str = Field(
        "", max_length=500,
        description="Installation address",
    )
    goods_categories: List[CBAMGoodsCategory] = Field(
        default_factory=list,
        description="CBAM goods categories produced at this installation",
    )
    cn_codes: List[str] = Field(
        default_factory=list,
        description="CN codes of products manufactured",
    )
    emission_type: str = Field(
        "direct",
        description="Primary emission type: direct, indirect, or both",
    )
    capacity_tonnes_per_year: float = Field(
        0.0, ge=0,
        description="Annual production capacity in tonnes",
    )
    registered_at: datetime = Field(
        default_factory=utcnow,
        description="Registration timestamp",
    )

    @field_validator("country")
    @classmethod
    def uppercase_country(cls, v: str) -> str:
        """Ensure country code is uppercase."""
        return v.strip().upper()

class SupplierProfile(BaseModel):
    """Supplier profile for CBAM data management.

    Contains the supplier's identification, installations, verification
    status, and data quality metrics.
    """

    supplier_id: str = Field(
        default_factory=_new_uuid,
        description="Unique supplier identifier",
    )
    name: str = Field(
        ..., min_length=1, max_length=300,
        description="Legal name of the supplier",
    )
    country: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    eori_number: Optional[str] = Field(
        None, max_length=17,
        description="EORI number (if applicable)",
    )
    tax_id: Optional[str] = Field(
        None, max_length=50,
        description="Tax identification number",
    )
    address: str = Field(
        "", max_length=500,
        description="Registered address",
    )
    contact_name: str = Field(
        "", max_length=200,
        description="Primary contact person",
    )
    contact_email: str = Field(
        "", max_length=200,
        description="Contact email address",
    )
    installations: List[Installation] = Field(
        default_factory=list,
        description="Registered production installations",
    )
    verification_status: VerificationStatus = Field(
        VerificationStatus.PENDING,
        description="Current verification status",
    )
    data_quality_score: float = Field(
        0.0, ge=0, le=100,
        description="Overall data quality score (0-100)",
    )
    registration_date: datetime = Field(
        default_factory=utcnow,
        description="Date supplier was registered",
    )
    last_submission_date: Optional[datetime] = Field(
        None, description="Date of most recent emission data submission",
    )

    @field_validator("country")
    @classmethod
    def uppercase_country(cls, v: str) -> str:
        """Ensure country code is uppercase."""
        return v.strip().upper()

class EmissionSubmission(BaseModel):
    """Emission data submission from a supplier for a specific period.

    Contains the emission data for a single installation and period,
    including quality scoring and review status.
    """

    submission_id: str = Field(
        default_factory=_new_uuid,
        description="Unique submission identifier",
    )
    supplier_id: str = Field(
        ..., description="Supplier who submitted the data",
    )
    installation_id: str = Field(
        ..., description="Installation the data applies to",
    )
    period: str = Field(
        ..., description="Reporting period (e.g., 'Q1-2027' or '2027')",
    )
    goods_category: CBAMGoodsCategory = Field(
        ..., description="CBAM goods category",
    )
    cn_codes: List[str] = Field(
        default_factory=list,
        description="CN codes covered by this submission",
    )
    direct_emissions_tco2e: float = Field(
        ..., ge=0,
        description="Total direct emissions (tCO2e)",
    )
    indirect_emissions_tco2e: float = Field(
        ..., ge=0,
        description="Total indirect emissions (tCO2e)",
    )
    production_quantity_tonnes: float = Field(
        ..., gt=0,
        description="Total production quantity (tonnes)",
    )
    calculation_method: CalculationMethod = Field(
        CalculationMethod.ACTUAL,
        description="Emission calculation method used by supplier",
    )
    verification_document: Optional[str] = Field(
        None, max_length=500,
        description="Reference to verification document",
    )
    notes: str = Field(
        "", max_length=2000,
        description="Supplementary notes from supplier",
    )
    status: SubmissionStatus = Field(
        SubmissionStatus.DRAFT,
        description="Current submission status",
    )
    quality_score: float = Field(
        0.0, ge=0, le=100,
        description="Data quality score (0-100)",
    )
    review_comments: str = Field(
        "", max_length=2000,
        description="Comments from the reviewer",
    )
    reviewer_id: Optional[str] = Field(
        None, description="ID of the reviewer",
    )
    submitted_at: Optional[datetime] = Field(
        None, description="Timestamp of submission",
    )
    reviewed_at: Optional[datetime] = Field(
        None, description="Timestamp of review",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="Record creation timestamp",
    )
    provenance_hash: str = Field(
        "", description="SHA-256 hash for audit trail",
    )

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SupplierManagementEngine:
    """Supplier and installation management engine for CBAM compliance.

    Manages the full supplier lifecycle from registration through emission
    data submission, review, and quality assessment. Provides dashboards
    and filtering capabilities for supplier portfolio management.

    Zero-Hallucination Guarantees:
        - Data quality scoring uses deterministic weighted arithmetic
        - No LLM involvement in any assessment or decision
        - SHA-256 provenance hashing on every submission

    Example:
        >>> engine = SupplierManagementEngine()
        >>> supplier = SupplierProfile(
        ...     name="Steel Works GmbH",
        ...     country="TR",
        ... )
        >>> supplier_id = engine.register_supplier(supplier)
        >>> assert supplier_id is not None
    """

    def __init__(self) -> None:
        """Initialize SupplierManagementEngine."""
        self._suppliers: Dict[str, SupplierProfile] = {}
        self._submissions: Dict[str, EmissionSubmission] = {}
        self._data_requests: Dict[str, Dict[str, Any]] = {}
        logger.info("SupplierManagementEngine initialized (v%s)", _MODULE_VERSION)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_supplier(self, profile: SupplierProfile) -> str:
        """Register a new supplier in the system.

        Args:
            profile: Complete supplier profile.

        Returns:
            The supplier_id of the registered supplier.

        Raises:
            ValueError: If a supplier with the same EORI already exists.
        """
        # Check for duplicate EORI
        if profile.eori_number:
            for existing in self._suppliers.values():
                if existing.eori_number == profile.eori_number:
                    raise ValueError(
                        f"Supplier with EORI {profile.eori_number} already registered: "
                        f"{existing.supplier_id}"
                    )

        self._suppliers[profile.supplier_id] = profile

        logger.info(
            "Supplier registered [%s]: %s (%s)",
            profile.supplier_id,
            profile.name,
            profile.country,
        )

        return profile.supplier_id

    def add_installation(
        self,
        supplier_id: str,
        installation: Installation,
    ) -> str:
        """Add a production installation to an existing supplier.

        Args:
            supplier_id: ID of the supplier.
            installation: Installation details.

        Returns:
            The installation_id.

        Raises:
            ValueError: If supplier is not found.
        """
        supplier = self._get_supplier(supplier_id)
        supplier.installations.append(installation)

        logger.info(
            "Installation added [%s] to supplier [%s]: %s (%s)",
            installation.installation_id,
            supplier_id,
            installation.name,
            installation.country,
        )

        return installation.installation_id

    def request_emission_data(
        self,
        supplier_id: str,
        period: str,
        goods_categories: Optional[List[str]] = None,
    ) -> str:
        """Send a data request to a supplier for emission data.

        Creates a data request record and returns a request ID. In a
        production system, this would trigger an email or API notification
        to the supplier.

        Args:
            supplier_id: ID of the supplier.
            period: Reporting period (e.g., 'Q1-2027').
            goods_categories: Optional filter for specific categories.

        Returns:
            Request ID for tracking.

        Raises:
            ValueError: If supplier is not found.
        """
        supplier = self._get_supplier(supplier_id)

        request_id = _new_uuid()
        self._data_requests[request_id] = {
            "request_id": request_id,
            "supplier_id": supplier_id,
            "supplier_name": supplier.name,
            "period": period,
            "goods_categories": goods_categories or [],
            "status": "SENT",
            "requested_at": utcnow().isoformat(),
            "installations": [i.installation_id for i in supplier.installations],
        }

        logger.info(
            "Data request [%s] sent to supplier [%s] for period %s",
            request_id,
            supplier_id,
            period,
        )

        return request_id

    def submit_emission_data(
        self,
        submission: EmissionSubmission,
    ) -> str:
        """Submit emission data from a supplier.

        Validates the submission, computes quality score and provenance hash,
        and stores the submission for review.

        Args:
            submission: Complete emission data submission.

        Returns:
            The submission_id.

        Raises:
            ValueError: If supplier or installation is not found.
        """
        # Validate supplier exists
        supplier = self._get_supplier(submission.supplier_id)

        # Validate installation belongs to supplier
        install_ids = {i.installation_id for i in supplier.installations}
        if submission.installation_id not in install_ids:
            raise ValueError(
                f"Installation {submission.installation_id} not found for "
                f"supplier {submission.supplier_id}"
            )

        # Auto-score quality
        submission.quality_score = self.score_data_quality(submission)
        submission.status = SubmissionStatus.SUBMITTED
        submission.submitted_at = utcnow()
        submission.provenance_hash = _compute_hash(submission)

        # Update supplier last submission date
        supplier.last_submission_date = utcnow()

        self._submissions[submission.submission_id] = submission

        logger.info(
            "Emission data submitted [%s]: supplier=%s, install=%s, "
            "period=%s, quality=%.1f",
            submission.submission_id,
            submission.supplier_id,
            submission.installation_id,
            submission.period,
            submission.quality_score,
        )

        return submission.submission_id

    def review_submission(
        self,
        submission_id: str,
        reviewer: str,
        decision: str,
        comments: str = "",
    ) -> Dict[str, Any]:
        """Review and accept/reject an emission data submission.

        Args:
            submission_id: ID of the submission to review.
            reviewer: ID of the reviewer.
            decision: 'ACCEPTED' or 'REJECTED'.
            comments: Review comments.

        Returns:
            Dictionary with review outcome.

        Raises:
            ValueError: If submission not found or invalid decision.
        """
        if submission_id not in self._submissions:
            raise ValueError(f"Submission not found: {submission_id}")

        if decision not in ("ACCEPTED", "REJECTED"):
            raise ValueError(f"Invalid decision: {decision}. Must be ACCEPTED or REJECTED")

        submission = self._submissions[submission_id]
        submission.status = SubmissionStatus(decision)
        submission.reviewer_id = reviewer
        submission.review_comments = comments
        submission.reviewed_at = utcnow()
        submission.provenance_hash = _compute_hash(submission)

        # Update supplier quality score if accepted
        if decision == "ACCEPTED":
            self._update_supplier_quality(submission.supplier_id)

        logger.info(
            "Submission [%s] reviewed by %s: %s - %s",
            submission_id,
            reviewer,
            decision,
            comments[:100] if comments else "No comments",
        )

        return {
            "submission_id": submission_id,
            "decision": decision,
            "reviewer": reviewer,
            "comments": comments,
            "reviewed_at": submission.reviewed_at.isoformat(),
            "quality_score": submission.quality_score,
        }

    def score_data_quality(
        self,
        submission: EmissionSubmission,
    ) -> float:
        """Score the data quality of an emission submission (0-100).

        Scoring components (weighted):
            - Completeness (25%): All required fields populated
            - Methodology (25%): Actual > country_default > default
            - Plausibility (25%): Emission intensity within expected range
            - Documentation (15%): Verification document provided
            - Consistency (10%): Matches historical patterns

        Args:
            submission: EmissionSubmission to score.

        Returns:
            Quality score between 0 and 100.
        """
        completeness = self._score_completeness(submission)
        methodology = self._score_methodology(submission)
        plausibility = self._score_plausibility(submission)
        documentation = self._score_documentation(submission)
        consistency = self._score_consistency(submission)

        # Weighted average
        score = (
            completeness * Decimal("0.25")
            + methodology * Decimal("0.25")
            + plausibility * Decimal("0.25")
            + documentation * Decimal("0.15")
            + consistency * Decimal("0.10")
        )

        return float(score.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))

    def get_supplier_dashboard(
        self,
        supplier_id: str,
    ) -> Dict[str, Any]:
        """Get a dashboard summary for a supplier.

        Aggregates submission history, quality metrics, installation count,
        and compliance status into a single dashboard view.

        Args:
            supplier_id: ID of the supplier.

        Returns:
            Dictionary with dashboard metrics.

        Raises:
            ValueError: If supplier is not found.
        """
        supplier = self._get_supplier(supplier_id)
        submissions = self.get_submission_history(supplier_id)

        # Calculate metrics
        total_submissions = len(submissions)
        accepted = sum(1 for s in submissions if s.status == SubmissionStatus.ACCEPTED)
        rejected = sum(1 for s in submissions if s.status == SubmissionStatus.REJECTED)
        pending = sum(
            1 for s in submissions
            if s.status in (SubmissionStatus.SUBMITTED, SubmissionStatus.REVIEWED)
        )

        avg_quality = Decimal("0")
        if submissions:
            total_score = sum(_decimal(s.quality_score) for s in submissions)
            avg_quality = total_score / Decimal(str(len(submissions)))

        total_direct = sum(s.direct_emissions_tco2e for s in submissions)
        total_indirect = sum(s.indirect_emissions_tco2e for s in submissions)
        total_quantity = sum(s.production_quantity_tonnes for s in submissions)

        return {
            "supplier_id": supplier_id,
            "supplier_name": supplier.name,
            "country": supplier.country,
            "verification_status": supplier.verification_status.value,
            "installation_count": len(supplier.installations),
            "total_submissions": total_submissions,
            "accepted_submissions": accepted,
            "rejected_submissions": rejected,
            "pending_submissions": pending,
            "acceptance_rate": round(accepted / max(total_submissions, 1) * 100, 1),
            "average_quality_score": float(
                avg_quality.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
            ),
            "total_direct_emissions_tco2e": round(total_direct, 4),
            "total_indirect_emissions_tco2e": round(total_indirect, 4),
            "total_production_tonnes": round(total_quantity, 2),
            "last_submission_date": (
                supplier.last_submission_date.isoformat()
                if supplier.last_submission_date
                else None
            ),
            "goods_categories": list(set(
                cat.value
                for inst in supplier.installations
                for cat in inst.goods_categories
            )),
        }

    def list_suppliers(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SupplierProfile]:
        """List suppliers with optional filtering.

        Supported filters:
            - country: ISO 3166-1 alpha-2 country code
            - verification_status: VerificationStatus value
            - goods_category: CBAMGoodsCategory value
            - min_quality_score: Minimum data quality score

        Args:
            filters: Optional dictionary of filter criteria.

        Returns:
            List of matching SupplierProfile objects.
        """
        suppliers = list(self._suppliers.values())

        if not filters:
            return sorted(suppliers, key=lambda s: s.name)

        filtered = suppliers
        if "country" in filters:
            country = filters["country"].upper()
            filtered = [s for s in filtered if s.country == country]

        if "verification_status" in filters:
            status = filters["verification_status"]
            filtered = [
                s for s in filtered
                if s.verification_status.value == status
                or s.verification_status == status
            ]

        if "goods_category" in filters:
            cat = filters["goods_category"]
            filtered = [
                s for s in filtered
                if any(
                    c.value == cat or c == cat
                    for inst in s.installations
                    for c in inst.goods_categories
                )
            ]

        if "min_quality_score" in filters:
            min_score = float(filters["min_quality_score"])
            filtered = [s for s in filtered if s.data_quality_score >= min_score]

        return sorted(filtered, key=lambda s: s.name)

    def get_submission_history(
        self,
        supplier_id: str,
    ) -> List[EmissionSubmission]:
        """Get all emission data submissions for a supplier.

        Args:
            supplier_id: ID of the supplier.

        Returns:
            List of EmissionSubmission sorted by submission date (newest first).
        """
        submissions = [
            s for s in self._submissions.values()
            if s.supplier_id == supplier_id
        ]
        submissions.sort(
            key=lambda s: s.submitted_at or s.created_at,
            reverse=True,
        )
        return submissions

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def supplier_count(self) -> int:
        """Number of registered suppliers."""
        return len(self._suppliers)

    @property
    def submission_count(self) -> int:
        """Number of emission data submissions."""
        return len(self._submissions)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_supplier(self, supplier_id: str) -> SupplierProfile:
        """Look up a supplier by ID, raising ValueError if not found."""
        if supplier_id not in self._suppliers:
            raise ValueError(f"Supplier not found: {supplier_id}")
        return self._suppliers[supplier_id]

    def _update_supplier_quality(self, supplier_id: str) -> None:
        """Recalculate overall supplier quality score from accepted submissions."""
        supplier = self._suppliers.get(supplier_id)
        if not supplier:
            return

        accepted = [
            s for s in self._submissions.values()
            if s.supplier_id == supplier_id and s.status == SubmissionStatus.ACCEPTED
        ]
        if accepted:
            total = sum(_decimal(s.quality_score) for s in accepted)
            avg = total / Decimal(str(len(accepted)))
            supplier.data_quality_score = float(
                avg.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
            )

    def _score_completeness(self, submission: EmissionSubmission) -> Decimal:
        """Score completeness of submission fields (0-100)."""
        required_fields = [
            submission.supplier_id,
            submission.installation_id,
            submission.period,
            submission.goods_category,
        ]
        numeric_fields = [
            submission.direct_emissions_tco2e,
            submission.indirect_emissions_tco2e,
            submission.production_quantity_tonnes,
        ]

        filled = sum(1 for f in required_fields if f)
        filled += sum(1 for n in numeric_fields if n > 0)
        total = len(required_fields) + len(numeric_fields)

        return _decimal(filled / max(total, 1) * 100)

    def _score_methodology(self, submission: EmissionSubmission) -> Decimal:
        """Score the calculation methodology used (0-100)."""
        method_scores = {
            CalculationMethod.ACTUAL: 100,
            CalculationMethod.COUNTRY_DEFAULT: 60,
            CalculationMethod.DEFAULT: 30,
        }
        return _decimal(method_scores.get(submission.calculation_method, 30))

    def _score_plausibility(self, submission: EmissionSubmission) -> Decimal:
        """Score emission intensity plausibility (0-100)."""
        if submission.production_quantity_tonnes <= 0:
            return Decimal("0")

        total_emissions = (
            submission.direct_emissions_tco2e + submission.indirect_emissions_tco2e
        )
        intensity = total_emissions / submission.production_quantity_tonnes

        cat = submission.goods_category.value
        ranges = PLAUSIBLE_INTENSITY_RANGES.get(cat, {"min": 0.0, "max": 50.0})

        if ranges["min"] <= intensity <= ranges["max"]:
            return Decimal("100")
        elif intensity < ranges["min"]:
            # Below minimum - slightly suspicious but not impossible
            ratio = intensity / max(ranges["min"], 0.001)
            return _decimal(max(ratio * 80, 20))
        else:
            # Above maximum - increasingly suspicious
            ratio = ranges["max"] / max(intensity, 0.001)
            return _decimal(max(ratio * 80, 10))

    def _score_documentation(self, submission: EmissionSubmission) -> Decimal:
        """Score documentation quality (0-100)."""
        score = Decimal("40")  # Base score for having a submission

        if submission.verification_document:
            score += Decimal("40")

        if submission.notes and len(submission.notes) > 10:
            score += Decimal("10")

        if submission.cn_codes:
            score += Decimal("10")

        return min(score, Decimal("100"))

    def _score_consistency(self, submission: EmissionSubmission) -> Decimal:
        """Score consistency with historical submissions (0-100).

        Compares the current submission's emission intensity with the
        average of prior accepted submissions for the same installation.
        """
        prior = [
            s for s in self._submissions.values()
            if s.installation_id == submission.installation_id
            and s.status == SubmissionStatus.ACCEPTED
            and s.submission_id != submission.submission_id
        ]

        if not prior:
            return Decimal("70")  # No history, neutral score

        # Calculate historical average intensity
        hist_intensities = []
        for s in prior:
            if s.production_quantity_tonnes > 0:
                intensity = (
                    (s.direct_emissions_tco2e + s.indirect_emissions_tco2e)
                    / s.production_quantity_tonnes
                )
                hist_intensities.append(intensity)

        if not hist_intensities:
            return Decimal("70")

        avg_hist = sum(hist_intensities) / len(hist_intensities)

        if submission.production_quantity_tonnes > 0:
            current = (
                (submission.direct_emissions_tco2e + submission.indirect_emissions_tco2e)
                / submission.production_quantity_tonnes
            )
        else:
            return Decimal("50")

        if avg_hist > 0:
            deviation = abs(current - avg_hist) / avg_hist
            if deviation < 0.1:
                return Decimal("100")
            elif deviation < 0.25:
                return Decimal("80")
            elif deviation < 0.5:
                return Decimal("60")
            else:
                return Decimal("30")

        return Decimal("70")
