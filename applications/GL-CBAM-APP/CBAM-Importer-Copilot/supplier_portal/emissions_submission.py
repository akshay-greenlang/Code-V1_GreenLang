# -*- coding: utf-8 -*-
"""
GL-CBAM-APP Supplier Portal - Emissions Submission Engine

Thread-safe singleton engine for managing emissions data submissions from
third-country installation operators.  Handles submission lifecycle (draft ->
submitted -> under_review -> accepted/rejected -> amended), validation,
total embedded emissions calculation, CN code checks, and data quality
scoring.

All numeric calculations use ``Decimal`` with ``ROUND_HALF_UP`` to satisfy
the zero-hallucination requirement for regulatory values.

Reference:
  - EU CBAM Regulation 2023/956, Art. 35-36
  - EU Implementing Regulation 2023/1773, Art. 3-8
  - GHG Protocol, Chapter 8 (Scope 2 embedded emissions)

Version: 1.1.0
Author: GreenLang CBAM Team
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import logging
import threading
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from supplier_portal.models import (
    CalculationMethod,
    CBAMSector,
    CN_CODE_PATTERN,
    CN_CODE_SECTORS,
    DataQualityScore,
    EmissionsDataSubmission,
    ExportFormat,
    MATERIALITY_THRESHOLD,
    PrecursorEmission,
    REPORTING_PERIOD_PATTERN,
    SubmissionStatus,
    SupportingDocument,
    ValidationIssue,
    _quantize,
    _utc_now,
    DECIMAL_PLACES,
    DECIMAL_QUANTIZE,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================


class SubmissionError(Exception):
    """Base exception for emissions submission operations."""


class SubmissionNotFoundError(SubmissionError):
    """Raised when a requested submission does not exist."""


class SubmissionStateError(SubmissionError):
    """Raised when an operation is invalid for the current submission status."""


class SubmissionValidationError(SubmissionError):
    """Raised when submission data fails validation."""

    def __init__(self, message: str, issues: Optional[List[ValidationIssue]] = None):
        super().__init__(message)
        self.issues = issues or []


class ExportError(SubmissionError):
    """Raised when data export fails."""


# ============================================================================
# JSON ENCODER FOR DECIMALS
# ============================================================================


class _DecimalEncoder(json.JSONEncoder):
    """JSON encoder that converts Decimal to string for serialization."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json")
        return super().default(obj)


# ============================================================================
# CALCULATION METHOD PERIOD RULES
# ============================================================================

# After Q2 2024, only EU implementing regulation method and direct measurement
# are accepted for the definitive CBAM period.
_TRANSITIONAL_METHODS = frozenset({
    CalculationMethod.EU_IMPLEMENTING_REG,
    CalculationMethod.GHG_PROTOCOL,
    CalculationMethod.NATIONAL_METHOD,
    CalculationMethod.DEFAULT_VALUES,
    CalculationMethod.DIRECT_MEASUREMENT,
})

_DEFINITIVE_METHODS = frozenset({
    CalculationMethod.EU_IMPLEMENTING_REG,
    CalculationMethod.DIRECT_MEASUREMENT,
})

# Transitional period end: Q2 2025
_DEFINITIVE_START_YEAR = 2025
_DEFINITIVE_START_QUARTER = 3


# ============================================================================
# EMISSIONS SUBMISSION ENGINE
# ============================================================================


class EmissionsSubmissionEngine:
    """
    Thread-safe engine for CBAM emissions data submission management.

    Handles the full submission lifecycle: creation, validation, review,
    amendment, data quality scoring, document attachment, history tracking,
    and data export.

    Thread Safety:
      All public methods acquire self._lock (RLock) before mutating state.

    Args:
        registry: SupplierRegistryEngine instance for supplier/installation lookups.

    Example:
        >>> engine = EmissionsSubmissionEngine(registry=registry)
        >>> submission = EmissionsDataSubmission(
        ...     installation_id="INST-ABC",
        ...     reporting_period_year=2026, reporting_period_quarter=1,
        ...     cn_code="72061000", product_description="Iron",
        ...     quantity_mt=Decimal("1000"),
        ...     direct_emissions_tCO2e_per_mt=Decimal("1.5"),
        ...     indirect_emissions_tCO2e_per_mt=Decimal("0.3"),
        ...     total_embedded_emissions_tCO2e_per_mt=Decimal("1.8"),
        ...     calculation_method=CalculationMethod.EU_IMPLEMENTING_REG,
        ... )
        >>> result = engine.submit_emissions(submission)
        >>> assert result.submission_status == SubmissionStatus.SUBMITTED
    """

    _instance: Optional["EmissionsSubmissionEngine"] = None
    _singleton_lock = threading.Lock()

    def __new__(cls, registry: Any = None) -> "EmissionsSubmissionEngine":
        """Singleton pattern: return existing instance if available."""
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self, registry: Any = None) -> None:
        """Initialize submission stores and lock."""
        if self._initialized:
            return
        self._lock = threading.RLock()
        self._registry = registry
        self._submissions: Dict[str, EmissionsDataSubmission] = {}
        self._submission_history: Dict[str, List[EmissionsDataSubmission]] = {}
        self._documents: Dict[str, List[SupportingDocument]] = {}
        self._audit_trail: List[Dict[str, Any]] = []
        self._initialized = True
        logger.info("EmissionsSubmissionEngine initialized")

    # ------------------------------------------------------------------
    # Submission Lifecycle
    # ------------------------------------------------------------------

    def submit_emissions(
        self, submission: EmissionsDataSubmission
    ) -> EmissionsDataSubmission:
        """
        Submit emissions data for a product at an installation.

        Validates the submission data, assigns a submission_id, computes
        the total embedded emissions, sets status to SUBMITTED, and
        records provenance.

        Args:
            submission: The emissions data to submit.

        Returns:
            The submitted EmissionsDataSubmission with assigned ID and status.

        Raises:
            SubmissionValidationError: If the submission fails validation.
        """
        start_time = datetime.now(timezone.utc)

        with self._lock:
            # Validate
            issues = self.validate_emissions_data(submission)
            errors = [i for i in issues if i.severity == "error"]
            if errors:
                raise SubmissionValidationError(
                    f"Submission has {len(errors)} validation error(s)",
                    issues=issues,
                )

            # Assign ID and set status
            submission_id = self._generate_id("SUB")
            total = self.calculate_total_embedded_emissions(submission)

            submission = submission.model_copy(
                update={
                    "submission_id": submission_id,
                    "submission_status": SubmissionStatus.SUBMITTED,
                    "total_embedded_emissions_tCO2e_per_mt": total,
                    "submitted_at": datetime.now(timezone.utc),
                    "version": 1,
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
            )

            # Compute provenance hash
            submission = submission.model_copy(
                update={
                    "provenance_hash": self._compute_provenance_hash(
                        submission.model_dump(mode="json")
                    )
                }
            )

            # Compute data quality score
            dq_score = self.calculate_data_quality_score(submission)
            submission = submission.model_copy(
                update={"data_quality_score": dq_score}
            )

            self._submissions[submission_id] = submission
            self._submission_history[submission_id] = [submission]
            self._documents[submission_id] = []

            self._record_audit(
                action="emissions_submitted",
                resource_type="submission",
                resource_id=submission_id,
                details={
                    "installation_id": submission.installation_id,
                    "period": submission.reporting_period,
                    "cn_code": submission.cn_code,
                    "total_emissions": str(total),
                },
            )

            duration_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000
            logger.info(
                "Emissions submitted: %s (installation=%s, period=%s) in %.1f ms",
                submission_id,
                submission.installation_id,
                submission.reporting_period,
                duration_ms,
            )

        return submission

    def get_submission(self, submission_id: str) -> EmissionsDataSubmission:
        """
        Retrieve a single submission by ID.

        Args:
            submission_id: Unique submission identifier.

        Returns:
            The EmissionsDataSubmission.

        Raises:
            SubmissionNotFoundError: If the submission does not exist.
        """
        with self._lock:
            return self._get_submission_or_raise(submission_id)

    def get_submissions(
        self,
        installation_id: Optional[str] = None,
        period: Optional[str] = None,
        status: Optional[SubmissionStatus] = None,
        supplier_id: Optional[str] = None,
    ) -> List[EmissionsDataSubmission]:
        """
        Retrieve submissions with optional filters.

        Args:
            installation_id: Filter by installation.
            period: Filter by reporting period (YYYYQN format).
            status: Filter by submission status.
            supplier_id: Filter by supplier (via registry lookup).

        Returns:
            List of matching submissions.
        """
        with self._lock:
            results = list(self._submissions.values())

            if installation_id:
                results = [
                    s for s in results if s.installation_id == installation_id
                ]

            if period:
                results = [
                    s for s in results if s.reporting_period == period
                ]

            if status:
                results = [
                    s for s in results if s.submission_status == status
                ]

            if supplier_id and self._registry:
                try:
                    installations = self._registry.get_installations(supplier_id)
                    inst_ids = {i.installation_id for i in installations}
                    results = [
                        s for s in results if s.installation_id in inst_ids
                    ]
                except Exception:
                    logger.warning(
                        "Could not filter by supplier_id: %s", supplier_id
                    )

            return results

    def amend_submission(
        self,
        submission_id: str,
        updates: Dict[str, Any],
        amendment_reason: Optional[str] = None,
    ) -> EmissionsDataSubmission:
        """
        Create a new version of an existing submission (amendment).

        Preserves the original in history and creates a new version
        with the updated fields.

        Args:
            submission_id: ID of the submission to amend.
            updates: Dictionary of field names to new values.
            amendment_reason: Reason for the amendment.

        Returns:
            The new versioned EmissionsDataSubmission.

        Raises:
            SubmissionNotFoundError: If the submission does not exist.
            SubmissionStateError: If the submission is in a state that
                cannot be amended (e.g., DRAFT).
        """
        with self._lock:
            current = self._get_submission_or_raise(submission_id)

            # Only submitted, accepted, or rejected submissions can be amended
            amendable_states = {
                SubmissionStatus.SUBMITTED,
                SubmissionStatus.ACCEPTED,
                SubmissionStatus.REJECTED,
                SubmissionStatus.UNDER_REVIEW,
            }
            if current.submission_status not in amendable_states:
                raise SubmissionStateError(
                    f"Cannot amend submission in '{current.submission_status.value}' state"
                )

            # Mark current as amended
            current = current.model_copy(
                update={
                    "submission_status": SubmissionStatus.AMENDED,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            # Store the amended version in history
            history = self._submission_history.get(submission_id, [])
            history.append(current)

            # Build new version
            allowed_fields = {
                "direct_emissions_tCO2e_per_mt",
                "indirect_emissions_tCO2e_per_mt",
                "total_embedded_emissions_tCO2e_per_mt",
                "quantity_mt",
                "calculation_method",
                "electricity_source",
                "grid_emission_factor",
                "precursor_emissions",
                "product_description",
                "carbon_price_paid_eur_per_tco2",
                "carbon_price_instrument",
            }

            update_dict: Dict[str, Any] = {}
            for field, value in updates.items():
                if field in allowed_fields:
                    update_dict[field] = value

            new_version = current.submission_status
            update_dict.update({
                "submission_status": SubmissionStatus.SUBMITTED,
                "version": current.version + 1,
                "submitted_at": datetime.now(timezone.utc),
                "reviewed_at": None,
                "reviewer_notes": None,
                "reviewer_id": None,
                "updated_at": datetime.now(timezone.utc),
            })

            amended = current.model_copy(update=update_dict)

            # Recalculate total
            total = self.calculate_total_embedded_emissions(amended)
            amended = amended.model_copy(
                update={
                    "total_embedded_emissions_tCO2e_per_mt": total,
                    "provenance_hash": self._compute_provenance_hash(
                        amended.model_dump(mode="json")
                    ),
                }
            )

            # Recompute data quality
            dq_score = self.calculate_data_quality_score(amended)
            amended = amended.model_copy(
                update={"data_quality_score": dq_score}
            )

            self._submissions[submission_id] = amended
            self._submission_history[submission_id] = history + [amended]

            self._record_audit(
                action="emissions_amended",
                resource_type="submission",
                resource_id=submission_id,
                details={
                    "new_version": amended.version,
                    "reason": amendment_reason,
                    "updated_fields": list(update_dict.keys()),
                },
            )

            logger.info(
                "Submission amended: %s -> v%d",
                submission_id,
                amended.version,
            )

        return amended

    def review_submission(
        self,
        submission_id: str,
        reviewer_id: str,
        outcome: str,
        notes: Optional[str] = None,
    ) -> EmissionsDataSubmission:
        """
        Review an emissions submission (accept, reject, or request amendment).

        Args:
            submission_id: ID of the submission to review.
            reviewer_id: ID of the reviewer performing the action.
            outcome: Review decision: 'accept', 'reject', or 'request_amendment'.
            notes: Optional reviewer notes.

        Returns:
            The updated EmissionsDataSubmission.

        Raises:
            SubmissionNotFoundError: If the submission does not exist.
            SubmissionStateError: If the submission is not in a reviewable state.
            ValueError: If outcome is not a valid decision.
        """
        with self._lock:
            submission = self._get_submission_or_raise(submission_id)

            reviewable = {SubmissionStatus.SUBMITTED, SubmissionStatus.UNDER_REVIEW}
            if submission.submission_status not in reviewable:
                raise SubmissionStateError(
                    f"Cannot review submission in '{submission.submission_status.value}' state"
                )

            outcome_map = {
                "accept": SubmissionStatus.ACCEPTED,
                "reject": SubmissionStatus.REJECTED,
                "request_amendment": SubmissionStatus.UNDER_REVIEW,
            }

            if outcome not in outcome_map:
                raise ValueError(
                    f"Invalid outcome '{outcome}'. Must be one of: {list(outcome_map.keys())}"
                )

            new_status = outcome_map[outcome]

            submission = submission.model_copy(
                update={
                    "submission_status": new_status,
                    "reviewed_at": datetime.now(timezone.utc),
                    "reviewer_id": reviewer_id,
                    "reviewer_notes": notes,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            submission = submission.model_copy(
                update={
                    "provenance_hash": self._compute_provenance_hash(
                        submission.model_dump(mode="json")
                    )
                }
            )

            self._submissions[submission_id] = submission

            # Update history
            history = self._submission_history.get(submission_id, [])
            history.append(submission)
            self._submission_history[submission_id] = history

            self._record_audit(
                action="emissions_reviewed",
                resource_type="submission",
                resource_id=submission_id,
                details={
                    "reviewer_id": reviewer_id,
                    "outcome": outcome,
                    "new_status": new_status.value,
                },
            )

            logger.info(
                "Submission reviewed: %s -> %s by %s",
                submission_id,
                new_status.value,
                reviewer_id,
            )

        return submission

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_emissions_data(
        self, submission: EmissionsDataSubmission
    ) -> List[ValidationIssue]:
        """
        Comprehensive validation of an emissions data submission.

        Checks:
          - Direct emissions are non-negative
          - Indirect emissions are non-negative
          - Total = direct + indirect + precursors (within tolerance)
          - CN code is in CBAM Annex I
          - Calculation method allowed for the period
          - Precursor emissions reference valid data
          - Quantity is positive

        Args:
            submission: The submission to validate.

        Returns:
            List of ValidationIssue objects (empty if valid).
        """
        issues: List[ValidationIssue] = []

        # Check direct emissions
        if submission.direct_emissions_tCO2e_per_mt < Decimal("0"):
            issues.append(
                ValidationIssue(
                    field="direct_emissions_tCO2e_per_mt",
                    code="NEGATIVE_DIRECT_EMISSIONS",
                    message="Direct emissions must be non-negative",
                    severity="error",
                )
            )

        # Check indirect emissions
        if submission.indirect_emissions_tCO2e_per_mt < Decimal("0"):
            issues.append(
                ValidationIssue(
                    field="indirect_emissions_tCO2e_per_mt",
                    code="NEGATIVE_INDIRECT_EMISSIONS",
                    message="Indirect emissions must be non-negative",
                    severity="error",
                )
            )

        # Check quantity
        if submission.quantity_mt <= Decimal("0"):
            issues.append(
                ValidationIssue(
                    field="quantity_mt",
                    code="INVALID_QUANTITY",
                    message="Quantity must be positive",
                    severity="error",
                )
            )

        # Validate CN code
        if not self.validate_cn_code(submission.cn_code):
            issues.append(
                ValidationIssue(
                    field="cn_code",
                    code="INVALID_CN_CODE",
                    message=(
                        f"CN code '{submission.cn_code}' is not in CBAM Annex I"
                    ),
                    severity="error",
                )
            )

        # Validate calculation method for period
        if not self.validate_calculation_method(
            submission.calculation_method,
            submission.reporting_period_year,
            submission.reporting_period_quarter,
        ):
            issues.append(
                ValidationIssue(
                    field="calculation_method",
                    code="INVALID_METHOD_FOR_PERIOD",
                    message=(
                        f"Calculation method '{submission.calculation_method.value}' "
                        f"is not allowed for period {submission.reporting_period}"
                    ),
                    severity="error",
                )
            )

        # Validate total = direct + indirect + precursors
        issues.extend(self._validate_total_consistency(submission))

        # Validate precursor emissions
        issues.extend(self._validate_precursors(submission))

        # Warn on missing grid emission factor for indirect emissions
        if (
            submission.indirect_emissions_tCO2e_per_mt > Decimal("0")
            and submission.grid_emission_factor is None
        ):
            issues.append(
                ValidationIssue(
                    field="grid_emission_factor",
                    code="MISSING_GRID_EF",
                    message=(
                        "Grid emission factor should be provided when "
                        "indirect emissions are non-zero"
                    ),
                    severity="warning",
                )
            )

        # Warn on high emissions intensity
        if submission.direct_emissions_tCO2e_per_mt > Decimal("50"):
            issues.append(
                ValidationIssue(
                    field="direct_emissions_tCO2e_per_mt",
                    code="HIGH_DIRECT_EMISSIONS",
                    message=(
                        "Direct emissions of >50 tCO2e/mt is unusually high; "
                        "please verify"
                    ),
                    severity="warning",
                )
            )

        if issues:
            error_count = sum(1 for i in issues if i.severity == "error")
            warning_count = sum(1 for i in issues if i.severity == "warning")
            logger.info(
                "Validation: %d error(s), %d warning(s) for submission %s",
                error_count,
                warning_count,
                submission.submission_id,
            )

        return issues

    def calculate_total_embedded_emissions(
        self, submission: EmissionsDataSubmission
    ) -> Decimal:
        """
        Calculate total specific embedded emissions (tCO2e/mt).

        Formula: total = direct + indirect + SUM(precursor_qty * precursor_ef)

        ZERO-HALLUCINATION: Pure deterministic arithmetic using Decimal.

        Args:
            submission: The submission containing emissions data.

        Returns:
            Total specific embedded emissions as Decimal.
        """
        direct = submission.direct_emissions_tCO2e_per_mt
        indirect = submission.indirect_emissions_tCO2e_per_mt

        precursor_sum = Decimal("0")
        if submission.precursor_emissions:
            for precursor in submission.precursor_emissions:
                contribution = _quantize(
                    precursor.quantity_per_unit
                    * precursor.embedded_emissions_tCO2e_per_mt
                )
                precursor_sum += contribution
            precursor_sum = _quantize(precursor_sum)

        total = _quantize(direct + indirect + precursor_sum)
        return total

    def validate_cn_code(self, cn_code: str) -> bool:
        """
        Check whether a CN code is covered by CBAM Annex I.

        Validates both the 8-digit format and that the 4-digit heading
        maps to a known CBAM sector.

        Args:
            cn_code: 8-digit CN code string.

        Returns:
            True if the CN code is valid and CBAM-covered.
        """
        if not CN_CODE_PATTERN.match(cn_code):
            return False

        heading = cn_code[:4]
        return heading in CN_CODE_SECTORS

    def validate_calculation_method(
        self,
        method: CalculationMethod,
        year: int,
        quarter: int,
    ) -> bool:
        """
        Check whether a calculation method is allowed for the given period.

        During the transitional period (before Q3 2025), all methods are
        accepted.  From Q3 2025 onwards (definitive period), only the EU
        implementing regulation method and direct measurement are accepted.

        Args:
            method: The calculation method to check.
            year: Reporting year.
            quarter: Reporting quarter.

        Returns:
            True if the method is allowed.
        """
        is_definitive = (
            year > _DEFINITIVE_START_YEAR
            or (year == _DEFINITIVE_START_YEAR and quarter >= _DEFINITIVE_START_QUARTER)
        )

        if is_definitive:
            return method in _DEFINITIVE_METHODS

        return method in _TRANSITIONAL_METHODS

    # ------------------------------------------------------------------
    # Document Management
    # ------------------------------------------------------------------

    def attach_document(
        self,
        submission_id: str,
        document: SupportingDocument,
    ) -> SupportingDocument:
        """
        Attach a supporting document to a submission.

        Args:
            submission_id: ID of the parent submission.
            document: The document metadata to attach.

        Returns:
            The attached SupportingDocument with assigned doc_id.

        Raises:
            SubmissionNotFoundError: If the submission does not exist.
        """
        with self._lock:
            self._get_submission_or_raise(submission_id)

            document = document.model_copy(
                update={
                    "submission_id": submission_id,
                    "upload_date": datetime.now(timezone.utc),
                }
            )

            if submission_id not in self._documents:
                self._documents[submission_id] = []
            self._documents[submission_id].append(document)

            self._record_audit(
                action="document_attached",
                resource_type="submission",
                resource_id=submission_id,
                details={
                    "doc_id": document.doc_id,
                    "doc_type": document.doc_type.value,
                    "filename": document.filename,
                },
            )

            logger.info(
                "Document attached to %s: %s (%s)",
                submission_id,
                document.doc_id,
                document.filename,
            )

        return document

    def get_documents(self, submission_id: str) -> List[SupportingDocument]:
        """
        Retrieve all documents attached to a submission.

        Args:
            submission_id: Submission identifier.

        Returns:
            List of SupportingDocument objects.
        """
        with self._lock:
            return list(self._documents.get(submission_id, []))

    # ------------------------------------------------------------------
    # History & Export
    # ------------------------------------------------------------------

    def get_submission_history(
        self, submission_id: str
    ) -> List[EmissionsDataSubmission]:
        """
        Retrieve the full version history of a submission.

        Args:
            submission_id: Submission identifier.

        Returns:
            List of EmissionsDataSubmission versions, ordered chronologically.

        Raises:
            SubmissionNotFoundError: If the submission does not exist.
        """
        with self._lock:
            self._get_submission_or_raise(submission_id)
            return list(self._submission_history.get(submission_id, []))

    def export_submissions(
        self,
        installation_id: str,
        period: Optional[str] = None,
        fmt: ExportFormat = ExportFormat.JSON,
    ) -> bytes:
        """
        Export submission data in the requested format.

        Args:
            installation_id: Filter to this installation.
            period: Optional reporting period filter (YYYYQN).
            fmt: Export format: CSV, JSON, or XML.

        Returns:
            Byte content of the exported data.

        Raises:
            ExportError: If the export format is unsupported or no data found.
        """
        with self._lock:
            submissions = self.get_submissions(
                installation_id=installation_id, period=period
            )

            if not submissions:
                raise ExportError(
                    f"No submissions found for installation '{installation_id}'"
                    + (f" period '{period}'" if period else "")
                )

            if fmt == ExportFormat.JSON:
                return self._export_json(submissions)
            elif fmt == ExportFormat.CSV:
                return self._export_csv(submissions)
            elif fmt == ExportFormat.XML:
                return self._export_xml(submissions)
            else:
                raise ExportError(f"Unsupported export format: {fmt}")

    # ------------------------------------------------------------------
    # Data Quality Scoring
    # ------------------------------------------------------------------

    def calculate_data_quality_score(
        self, submission: EmissionsDataSubmission
    ) -> DataQualityScore:
        """
        Calculate a data quality score for a submission.

        Evaluates four dimensions (0-100 each):
          - Completeness: Required fields populated
          - Consistency: Values are internally consistent
          - Timeliness: Submitted within the reporting deadline
          - Accuracy: Calculation method and verification status

        ZERO-HALLUCINATION: Deterministic scoring using Decimal arithmetic.

        Args:
            submission: The submission to score.

        Returns:
            DataQualityScore with dimension and overall scores.
        """
        completeness = self._score_completeness(submission)
        consistency = self._score_consistency(submission)
        timeliness = self._score_timeliness(submission)
        accuracy = self._score_accuracy(submission)

        overall = _quantize(
            completeness * Decimal("0.30")
            + consistency * Decimal("0.25")
            + timeliness * Decimal("0.20")
            + accuracy * Decimal("0.25")
        )

        return DataQualityScore(
            completeness=completeness,
            consistency=consistency,
            timeliness=timeliness,
            accuracy=accuracy,
            overall=overall,
        )

    # ------------------------------------------------------------------
    # Reset (for testing)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all in-memory stores. For testing only."""
        with self._lock:
            self._submissions.clear()
            self._submission_history.clear()
            self._documents.clear()
            self._audit_trail.clear()
            logger.warning("EmissionsSubmissionEngine stores reset")

    # ------------------------------------------------------------------
    # Private: Validation Helpers
    # ------------------------------------------------------------------

    def _validate_total_consistency(
        self, submission: EmissionsDataSubmission
    ) -> List[ValidationIssue]:
        """Validate total emissions = direct + indirect + precursors."""
        issues: List[ValidationIssue] = []

        expected = self.calculate_total_embedded_emissions(submission)
        actual = submission.total_embedded_emissions_tCO2e_per_mt

        tolerance = Decimal("0.01")
        if abs(actual - expected) > tolerance:
            issues.append(
                ValidationIssue(
                    field="total_embedded_emissions_tCO2e_per_mt",
                    code="TOTAL_MISMATCH",
                    message=(
                        f"Total embedded emissions ({actual}) does not match "
                        f"computed total ({expected}): "
                        f"direct + indirect + precursors"
                    ),
                    severity="error",
                )
            )

        return issues

    def _validate_precursors(
        self, submission: EmissionsDataSubmission
    ) -> List[ValidationIssue]:
        """Validate precursor emission records."""
        issues: List[ValidationIssue] = []

        if not submission.precursor_emissions:
            return issues

        for idx, precursor in enumerate(submission.precursor_emissions):
            prefix = f"precursor_emissions[{idx}]"

            if not CN_CODE_PATTERN.match(precursor.precursor_cn_code):
                issues.append(
                    ValidationIssue(
                        field=f"{prefix}.precursor_cn_code",
                        code="INVALID_PRECURSOR_CN",
                        message=(
                            f"Precursor CN code '{precursor.precursor_cn_code}' "
                            f"is not a valid 8-digit code"
                        ),
                        severity="error",
                    )
                )

            if precursor.quantity_per_unit < Decimal("0"):
                issues.append(
                    ValidationIssue(
                        field=f"{prefix}.quantity_per_unit",
                        code="NEGATIVE_PRECURSOR_QTY",
                        message="Precursor quantity per unit must be non-negative",
                        severity="error",
                    )
                )

            if precursor.embedded_emissions_tCO2e_per_mt < Decimal("0"):
                issues.append(
                    ValidationIssue(
                        field=f"{prefix}.embedded_emissions_tCO2e_per_mt",
                        code="NEGATIVE_PRECURSOR_EMISSIONS",
                        message="Precursor embedded emissions must be non-negative",
                        severity="error",
                    )
                )

            # Validate precursor installation reference
            if (
                precursor.origin_installation_id
                and self._registry
            ):
                try:
                    self._registry.get_installation(
                        precursor.origin_installation_id
                    )
                except Exception:
                    issues.append(
                        ValidationIssue(
                            field=f"{prefix}.origin_installation_id",
                            code="INVALID_PRECURSOR_INSTALLATION",
                            message=(
                                f"Precursor installation "
                                f"'{precursor.origin_installation_id}' not found"
                            ),
                            severity="warning",
                        )
                    )

        return issues

    # ------------------------------------------------------------------
    # Private: Scoring Helpers
    # ------------------------------------------------------------------

    def _score_completeness(self, submission: EmissionsDataSubmission) -> Decimal:
        """Score completeness (0-100) based on populated fields."""
        total_fields = 10
        populated = 0

        if submission.cn_code:
            populated += 1
        if submission.product_description:
            populated += 1
        if submission.quantity_mt > Decimal("0"):
            populated += 1
        if submission.direct_emissions_tCO2e_per_mt >= Decimal("0"):
            populated += 1
        if submission.indirect_emissions_tCO2e_per_mt >= Decimal("0"):
            populated += 1
        if submission.calculation_method:
            populated += 1
        if submission.electricity_source:
            populated += 1
        if submission.grid_emission_factor is not None:
            populated += 1
        if submission.precursor_emissions:
            populated += 1
        if submission.carbon_price_paid_eur_per_tco2 is not None:
            populated += 1

        score = _quantize(
            Decimal(populated) / Decimal(total_fields) * Decimal("100")
        )
        return min(score, Decimal("100"))

    def _score_consistency(self, submission: EmissionsDataSubmission) -> Decimal:
        """Score consistency (0-100) based on internal data coherence."""
        score = Decimal("100")
        deductions = Decimal("0")

        # Check total matches components
        expected = self.calculate_total_embedded_emissions(submission)
        if abs(submission.total_embedded_emissions_tCO2e_per_mt - expected) > Decimal("0.01"):
            deductions += Decimal("30")

        # Check indirect emissions with grid factor
        if (
            submission.indirect_emissions_tCO2e_per_mt > Decimal("0")
            and submission.grid_emission_factor is None
        ):
            deductions += Decimal("15")

        # Check for zero direct emissions (suspicious for most sectors)
        if submission.direct_emissions_tCO2e_per_mt == Decimal("0"):
            deductions += Decimal("10")

        score = max(score - deductions, Decimal("0"))
        return _quantize(score)

    def _score_timeliness(self, submission: EmissionsDataSubmission) -> Decimal:
        """Score timeliness (0-100) based on submission timing."""
        score = Decimal("80")  # Default if we cannot determine timing

        if submission.submitted_at:
            # Calculate expected deadline (end of month following quarter end)
            year = submission.reporting_period_year
            quarter = submission.reporting_period_quarter
            quarter_end_month = quarter * 3

            # Deadline is last day of month following quarter end
            if quarter_end_month >= 12:
                deadline_year = year + 1
                deadline_month = 1
            else:
                deadline_year = year
                deadline_month = quarter_end_month + 1

            from calendar import monthrange
            _, last_day = monthrange(deadline_year, deadline_month)
            deadline = datetime(
                deadline_year, deadline_month, last_day,
                tzinfo=timezone.utc,
            )

            if submission.submitted_at <= deadline:
                # On time - full score with bonus for early
                days_before = (deadline - submission.submitted_at).days
                bonus = min(Decimal(days_before) * Decimal("0.5"), Decimal("20"))
                score = _quantize(Decimal("80") + bonus)
            else:
                # Late - deduct points
                days_late = (submission.submitted_at - deadline).days
                penalty = min(Decimal(days_late) * Decimal("2"), Decimal("60"))
                score = _quantize(Decimal("80") - penalty)

        return max(min(score, Decimal("100")), Decimal("0"))

    def _score_accuracy(self, submission: EmissionsDataSubmission) -> Decimal:
        """Score accuracy (0-100) based on methodology and verification."""
        score = Decimal("50")  # Base score

        # Better methods score higher
        method_scores = {
            CalculationMethod.DIRECT_MEASUREMENT: Decimal("30"),
            CalculationMethod.EU_IMPLEMENTING_REG: Decimal("25"),
            CalculationMethod.GHG_PROTOCOL: Decimal("20"),
            CalculationMethod.NATIONAL_METHOD: Decimal("15"),
            CalculationMethod.DEFAULT_VALUES: Decimal("5"),
        }
        score += method_scores.get(submission.calculation_method, Decimal("0"))

        # Bonus for supporting documents
        docs = self._documents.get(submission.submission_id, [])
        if docs:
            doc_bonus = min(Decimal(len(docs)) * Decimal("5"), Decimal("20"))
            score += doc_bonus

        return _quantize(min(score, Decimal("100")))

    # ------------------------------------------------------------------
    # Private: Export Helpers
    # ------------------------------------------------------------------

    def _export_json(self, submissions: List[EmissionsDataSubmission]) -> bytes:
        """Export submissions as JSON bytes."""
        data = [s.model_dump(mode="json") for s in submissions]
        return json.dumps(
            {"submissions": data, "count": len(data)},
            cls=_DecimalEncoder,
            indent=2,
        ).encode("utf-8")

    def _export_csv(self, submissions: List[EmissionsDataSubmission]) -> bytes:
        """Export submissions as CSV bytes."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        headers = [
            "submission_id",
            "installation_id",
            "reporting_period",
            "cn_code",
            "product_description",
            "quantity_mt",
            "direct_emissions_tCO2e_per_mt",
            "indirect_emissions_tCO2e_per_mt",
            "total_embedded_emissions_tCO2e_per_mt",
            "calculation_method",
            "submission_status",
            "version",
            "submitted_at",
        ]
        writer.writerow(headers)

        for sub in submissions:
            writer.writerow([
                sub.submission_id,
                sub.installation_id,
                sub.reporting_period,
                sub.cn_code,
                sub.product_description,
                str(sub.quantity_mt),
                str(sub.direct_emissions_tCO2e_per_mt),
                str(sub.indirect_emissions_tCO2e_per_mt),
                str(sub.total_embedded_emissions_tCO2e_per_mt),
                sub.calculation_method.value,
                sub.submission_status.value,
                sub.version,
                sub.submitted_at.isoformat() if sub.submitted_at else "",
            ])

        return output.getvalue().encode("utf-8")

    def _export_xml(self, submissions: List[EmissionsDataSubmission]) -> bytes:
        """Export submissions as XML bytes."""
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append("<CBAMSubmissions>")

        for sub in submissions:
            lines.append("  <Submission>")
            lines.append(f"    <SubmissionId>{sub.submission_id}</SubmissionId>")
            lines.append(
                f"    <InstallationId>{sub.installation_id}</InstallationId>"
            )
            lines.append(
                f"    <ReportingPeriod>{sub.reporting_period}</ReportingPeriod>"
            )
            lines.append(f"    <CNCode>{sub.cn_code}</CNCode>")
            lines.append(
                f"    <ProductDescription>{_xml_escape(sub.product_description)}"
                f"</ProductDescription>"
            )
            lines.append(f"    <QuantityMT>{sub.quantity_mt}</QuantityMT>")
            lines.append(
                f"    <DirectEmissions>{sub.direct_emissions_tCO2e_per_mt}"
                f"</DirectEmissions>"
            )
            lines.append(
                f"    <IndirectEmissions>{sub.indirect_emissions_tCO2e_per_mt}"
                f"</IndirectEmissions>"
            )
            lines.append(
                f"    <TotalEmbeddedEmissions>"
                f"{sub.total_embedded_emissions_tCO2e_per_mt}"
                f"</TotalEmbeddedEmissions>"
            )
            lines.append(
                f"    <CalculationMethod>{sub.calculation_method.value}"
                f"</CalculationMethod>"
            )
            lines.append(
                f"    <Status>{sub.submission_status.value}</Status>"
            )
            lines.append(f"    <Version>{sub.version}</Version>")
            lines.append(
                f"    <SubmittedAt>"
                f"{sub.submitted_at.isoformat() if sub.submitted_at else ''}"
                f"</SubmittedAt>"
            )
            lines.append("  </Submission>")

        lines.append("</CBAMSubmissions>")
        return "\n".join(lines).encode("utf-8")

    # ------------------------------------------------------------------
    # Private: General Helpers
    # ------------------------------------------------------------------

    def _get_submission_or_raise(
        self, submission_id: str
    ) -> EmissionsDataSubmission:
        """Retrieve submission or raise SubmissionNotFoundError."""
        submission = self._submissions.get(submission_id)
        if submission is None:
            raise SubmissionNotFoundError(
                f"Submission '{submission_id}' not found"
            )
        return submission

    def _generate_id(self, prefix: str) -> str:
        """Generate a unique ID with the given prefix."""
        short = uuid.uuid4().hex[:12].upper()
        return f"{prefix}-{short}"

    def _compute_provenance_hash(self, data: Any) -> str:
        """Compute SHA-256 provenance hash."""
        serialized = json.dumps(data, sort_keys=True, cls=_DecimalEncoder)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _record_audit(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append an entry to the in-memory audit trail."""
        entry = {
            "audit_id": self._generate_id("AUD"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "details": details or {},
        }
        self._audit_trail.append(entry)


# ============================================================================
# MODULE HELPERS
# ============================================================================


def _xml_escape(text: str) -> str:
    """Escape XML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
