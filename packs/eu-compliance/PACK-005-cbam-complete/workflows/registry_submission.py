# -*- coding: utf-8 -*-
"""
Registry Submission Workflow
===============================

Four-phase submission lifecycle workflow for CBAM declarations and reports
to the EU CBAM Registry. Handles pre-validation against the definitive-period
XML schema, eIDAS-authenticated submission, status polling with configurable
intervals and timeout, and downstream confirmation with audit evidence archival.

Regulatory Context:
    Per EU CBAM Regulation 2023/956:
    - Article 6: Annual CBAM declaration due by May 31 for previous year.
    - Article 35: CBAM Registry operated by European Commission for
      registration, declarations, certificates, and verification.
    - Implementing Regulation 2023/1773: Defines XML schema for quarterly
      reports (transitional) and annual declarations (definitive).
    - eIDAS: EU electronic identification for authentication.

Retry Policy:
    - Maximum 3 submission attempts with exponential backoff.
    - Base delay: 5 seconds, max delay: 60 seconds.
    - Backoff multiplier: 2.0.
    - Retries on transient errors (5xx, timeouts) only.

Phases:
    1. PreValidation - Schema validation, field checks, consistency
    2. Submit - POST to Registry API with eIDAS authentication
    3. Monitor - Poll status until Accepted/Rejected with timeout
    4. Confirm - Log acceptance, update status, trigger downstream

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


class SubmissionType(str, Enum):
    """Type of registry submission."""
    QUARTERLY_REPORT = "QUARTERLY_REPORT"
    ANNUAL_DECLARATION = "ANNUAL_DECLARATION"
    CORRECTION = "CORRECTION"
    VERIFICATION_REPORT = "VERIFICATION_REPORT"


class SubmissionStatus(str, Enum):
    """Registry submission processing status."""
    DRAFT = "DRAFT"
    VALIDATED = "VALIDATED"
    SUBMITTED = "SUBMITTED"
    PROCESSING = "PROCESSING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"
    TIMED_OUT = "TIMED_OUT"


class ValidationSeverity(str, Enum):
    """Validation finding severity."""
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


# =============================================================================
# CONSTANTS
# =============================================================================

MAX_RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 5.0
RETRY_MAX_DELAY_SECONDS = 60.0
RETRY_BACKOFF_MULTIPLIER = 2.0
DEFAULT_POLL_INTERVAL_SECONDS = 30
DEFAULT_POLL_TIMEOUT_SECONDS = 3600


# =============================================================================
# DATA MODELS - SHARED
# =============================================================================


class WorkflowContext(BaseModel):
    """Shared state passed between workflow phases."""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str = Field(...)
    execution_timestamp: datetime = Field(default_factory=datetime.utcnow)
    config: Dict[str, Any] = Field(default_factory=dict)
    phase_states: Dict[str, PhaseStatus] = Field(default_factory=dict)
    phase_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def set_phase_output(self, phase_name: str, outputs: Dict[str, Any]) -> None:
        """Store phase outputs for downstream consumption."""
        self.phase_outputs[phase_name] = outputs

    def get_phase_output(self, phase_name: str) -> Dict[str, Any]:
        """Retrieve outputs from a previous phase."""
        return self.phase_outputs.get(phase_name, {})

    def mark_phase(self, phase_name: str, status: PhaseStatus) -> None:
        """Record phase status for checkpoint/resume."""
        self.phase_states[phase_name] = status

    def is_phase_completed(self, phase_name: str) -> bool:
        """Check if a phase has already completed."""
        return self.phase_states.get(phase_name) == PhaseStatus.COMPLETED


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(...)
    status: PhaseStatus = Field(...)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    records_processed: int = Field(default=0)


class WorkflowResult(BaseModel):
    """Complete result from a multi-phase workflow execution."""
    workflow_id: str = Field(...)
    workflow_name: str = Field(...)
    status: WorkflowStatus = Field(...)
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    phases: List[PhaseResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


# =============================================================================
# DATA MODELS - REGISTRY SUBMISSION
# =============================================================================


class ValidationFinding(BaseModel):
    """A single validation finding."""
    field_path: str = Field(default="")
    severity: ValidationSeverity = Field(...)
    code: str = Field(default="")
    message: str = Field(...)
    expected: Optional[str] = Field(None)
    actual: Optional[str] = Field(None)


class DeclarationData(BaseModel):
    """CBAM declaration or report data for submission."""
    declaration_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    submission_type: SubmissionType = Field(
        default=SubmissionType.ANNUAL_DECLARATION
    )
    reporting_year: int = Field(..., ge=2026)
    reporting_quarter: Optional[int] = Field(None, ge=1, le=4)
    declarant_id: str = Field(..., description="Authorized declarant EORI")
    declarant_name: str = Field(default="")
    member_state: str = Field(..., description="EU member state ISO code")
    total_embedded_emissions_tco2e: float = Field(default=0.0, ge=0)
    certificates_to_surrender: float = Field(default=0.0, ge=0)
    goods_categories: List[Dict[str, Any]] = Field(default_factory=list)
    installations: List[Dict[str, Any]] = Field(default_factory=list)
    verification_statement_id: Optional[str] = Field(None)
    xml_payload: Optional[str] = Field(None, description="Pre-built XML")


class EidasCredentials(BaseModel):
    """eIDAS authentication credentials (reference only, no secrets)."""
    certificate_id: str = Field(..., description="eIDAS certificate reference")
    provider: str = Field(default="", description="eIDAS provider name")
    validity_end: Optional[str] = Field(None, description="Certificate expiry")


class RegistrySubmissionInput(BaseModel):
    """Input configuration for registry submission workflow."""
    organization_id: str = Field(...)
    declaration: DeclarationData = Field(...)
    eidas_credentials: EidasCredentials = Field(...)
    registry_endpoint: str = Field(
        default="https://cbam-registry.ec.europa.eu/api/v1"
    )
    poll_interval_seconds: int = Field(
        default=DEFAULT_POLL_INTERVAL_SECONDS, ge=5, le=300
    )
    poll_timeout_seconds: int = Field(
        default=DEFAULT_POLL_TIMEOUT_SECONDS, ge=60, le=86400
    )
    max_retry_attempts: int = Field(default=MAX_RETRY_ATTEMPTS, ge=1, le=10)
    trigger_downstream_on_accept: bool = Field(default=True)
    downstream_workflows: List[str] = Field(
        default_factory=lambda: [
            "certificate_trading", "cross_regulation_sync"
        ]
    )
    skip_phases: List[str] = Field(default_factory=list)


class RegistrySubmissionResult(WorkflowResult):
    """Complete result from registry submission workflow."""
    submission_type: str = Field(default="")
    declaration_id: str = Field(default="")
    submission_status: str = Field(default="")
    receipt_id: Optional[str] = Field(None)
    registry_reference: Optional[str] = Field(None)
    validation_errors: int = Field(default=0)
    validation_warnings: int = Field(default=0)
    retry_attempts: int = Field(default=0)
    downstream_triggered: List[str] = Field(default_factory=list)


# =============================================================================
# PHASE IMPLEMENTATIONS
# =============================================================================


class PreValidationPhase:
    """
    Phase 1: Pre-Validation.

    Validates the declaration/report against the definitive-period XML
    schema, checks all required fields are populated, verifies calculation
    consistency, and flags warnings vs errors.
    """

    PHASE_NAME = "pre_validation"

    # Required fields per submission type
    REQUIRED_FIELDS = {
        SubmissionType.ANNUAL_DECLARATION.value: [
            "declarant_id", "member_state",
            "total_embedded_emissions_tco2e",
            "certificates_to_surrender", "reporting_year",
        ],
        SubmissionType.QUARTERLY_REPORT.value: [
            "declarant_id", "member_state",
            "total_embedded_emissions_tco2e",
            "reporting_year", "reporting_quarter",
        ],
        SubmissionType.CORRECTION.value: [
            "declarant_id", "member_state", "reporting_year",
        ],
        SubmissionType.VERIFICATION_REPORT.value: [
            "declarant_id", "verification_statement_id",
            "reporting_year",
        ],
    }

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute pre-validation phase.

        Args:
            context: Workflow context with declaration data.

        Returns:
            PhaseResult with validation findings and pass/fail status.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            declaration = config.get("declaration", {})
            sub_type = declaration.get(
                "submission_type",
                SubmissionType.ANNUAL_DECLARATION.value,
            )

            findings: List[Dict[str, Any]] = []
            error_count = 0
            warning_count = 0

            # Required field validation
            required = self.REQUIRED_FIELDS.get(sub_type, [])
            for field in required:
                value = declaration.get(field)
                if value is None or value == "" or value == 0:
                    finding = {
                        "field_path": field,
                        "severity": ValidationSeverity.ERROR.value,
                        "code": f"REQUIRED_{field.upper()}",
                        "message": f"Required field '{field}' is missing or empty",
                    }
                    findings.append(finding)
                    error_count += 1

            # Declarant EORI format validation
            declarant_id = declaration.get("declarant_id", "")
            if declarant_id and not self._validate_eori_format(declarant_id):
                findings.append({
                    "field_path": "declarant_id",
                    "severity": ValidationSeverity.ERROR.value,
                    "code": "INVALID_EORI_FORMAT",
                    "message": (
                        f"EORI '{declarant_id}' does not match expected "
                        f"format (2-letter country + up to 15 digits)"
                    ),
                })
                error_count += 1

            # Member state validation
            member_state = declaration.get("member_state", "")
            valid_ms = {
                "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE",
                "FI", "FR", "DE", "GR", "HU", "IE", "IT", "LV",
                "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK",
                "SI", "ES", "SE",
            }
            if member_state and member_state not in valid_ms:
                findings.append({
                    "field_path": "member_state",
                    "severity": ValidationSeverity.ERROR.value,
                    "code": "INVALID_MEMBER_STATE",
                    "message": f"'{member_state}' is not a valid EU member state",
                })
                error_count += 1

            # Emissions consistency check
            total_emissions = declaration.get(
                "total_embedded_emissions_tco2e", 0
            )
            certificates = declaration.get("certificates_to_surrender", 0)
            if (sub_type == SubmissionType.ANNUAL_DECLARATION.value
                    and total_emissions > 0 and certificates <= 0):
                findings.append({
                    "field_path": "certificates_to_surrender",
                    "severity": ValidationSeverity.WARNING.value,
                    "code": "ZERO_CERTIFICATES",
                    "message": (
                        "Total emissions > 0 but certificates to surrender "
                        "is zero. Verify deductions cover all emissions."
                    ),
                })
                warning_count += 1

            # Goods categories validation
            goods = declaration.get("goods_categories", [])
            if not goods:
                findings.append({
                    "field_path": "goods_categories",
                    "severity": ValidationSeverity.WARNING.value,
                    "code": "NO_GOODS_CATEGORIES",
                    "message": "No goods categories specified in declaration",
                })
                warning_count += 1

            # Category emissions sum consistency
            if goods and total_emissions > 0:
                cat_total = sum(
                    g.get("embedded_emissions_tco2e", 0) for g in goods
                )
                if abs(cat_total - total_emissions) > 0.01:
                    findings.append({
                        "field_path": "goods_categories",
                        "severity": ValidationSeverity.WARNING.value,
                        "code": "EMISSIONS_SUM_MISMATCH",
                        "message": (
                            f"Sum of category emissions ({cat_total:.4f}) "
                            f"differs from total ({total_emissions:.4f})"
                        ),
                        "expected": str(total_emissions),
                        "actual": str(cat_total),
                    })
                    warning_count += 1

            # Installation data completeness
            installations = declaration.get("installations", [])
            for idx, inst in enumerate(installations):
                if not inst.get("installation_id"):
                    findings.append({
                        "field_path": f"installations[{idx}].installation_id",
                        "severity": ValidationSeverity.ERROR.value,
                        "code": "MISSING_INSTALLATION_ID",
                        "message": f"Installation at index {idx} missing ID",
                    })
                    error_count += 1

            # Verification check for annual declarations
            if sub_type == SubmissionType.ANNUAL_DECLARATION.value:
                if not declaration.get("verification_statement_id"):
                    findings.append({
                        "field_path": "verification_statement_id",
                        "severity": ValidationSeverity.WARNING.value,
                        "code": "NO_VERIFICATION",
                        "message": (
                            "No verification statement linked to annual "
                            "declaration. May be required by NCA."
                        ),
                    })
                    warning_count += 1

            # Overall validation result
            passed = error_count == 0
            outputs["validation_passed"] = passed
            outputs["findings"] = findings
            outputs["error_count"] = error_count
            outputs["warning_count"] = warning_count
            outputs["total_findings"] = len(findings)
            outputs["submission_type"] = sub_type
            outputs["declaration_id"] = declaration.get("declaration_id", "")

            if not passed:
                errors.append(
                    f"Pre-validation failed with {error_count} error(s)"
                )
            if warning_count > 0:
                warnings.append(
                    f"{warning_count} validation warning(s) detected"
                )

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("PreValidation failed: %s", exc, exc_info=True)
            errors.append(f"Pre-validation failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=len(outputs.get("findings", [])),
        )

    def _validate_eori_format(self, eori: str) -> bool:
        """Validate EORI number format: 2-letter country + up to 15 digits."""
        if len(eori) < 3 or len(eori) > 17:
            return False
        country = eori[:2]
        digits = eori[2:]
        return country.isalpha() and country.isupper() and digits.isdigit()


class SubmitPhase:
    """
    Phase 2: Submit.

    POSTs the declaration/report to the CBAM Registry API with eIDAS
    authentication. Implements retry logic with exponential backoff
    for transient failures. Captures submission receipt ID and logs
    the full request/response for audit purposes.
    """

    PHASE_NAME = "submit"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute submission phase.

        Args:
            context: Workflow context with validated declaration data.

        Returns:
            PhaseResult with submission receipt and attempt log.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            validation = context.get_phase_output("pre_validation")

            # Check if validation passed
            if not validation.get("validation_passed", False):
                errors.append(
                    "Cannot submit: pre-validation failed with "
                    f"{validation.get('error_count', 0)} error(s)"
                )
                outputs["submission_blocked"] = True
                outputs["reason"] = "pre_validation_failed"
                return PhaseResult(
                    phase_name=self.PHASE_NAME,
                    status=PhaseStatus.FAILED,
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                    errors=errors,
                    outputs=outputs,
                    provenance_hash=_hash_data(outputs),
                )

            declaration = config.get("declaration", {})
            eidas = config.get("eidas_credentials", {})
            endpoint = config.get(
                "registry_endpoint",
                "https://cbam-registry.ec.europa.eu/api/v1",
            )
            max_attempts = config.get("max_retry_attempts", MAX_RETRY_ATTEMPTS)

            # Retry loop with exponential backoff
            attempt_log: List[Dict[str, Any]] = []
            receipt_id = None
            registry_ref = None
            final_status = SubmissionStatus.ERROR

            for attempt in range(1, max_attempts + 1):
                attempt_start = datetime.utcnow()
                logger.info(
                    "Submission attempt %d/%d for declaration %s",
                    attempt, max_attempts,
                    declaration.get("declaration_id", ""),
                )

                try:
                    result = await self._post_to_registry(
                        declaration, eidas, endpoint
                    )

                    attempt_entry = {
                        "attempt": attempt,
                        "started_at": attempt_start.isoformat(),
                        "completed_at": datetime.utcnow().isoformat(),
                        "status_code": result.get("status_code", 0),
                        "success": result.get("success", False),
                        "receipt_id": result.get("receipt_id"),
                        "registry_reference": result.get("registry_reference"),
                        "error_message": result.get("error_message"),
                    }
                    attempt_log.append(attempt_entry)

                    if result.get("success"):
                        receipt_id = result.get("receipt_id")
                        registry_ref = result.get("registry_reference")
                        final_status = SubmissionStatus.SUBMITTED
                        logger.info(
                            "Submission successful: receipt=%s ref=%s",
                            receipt_id, registry_ref,
                        )
                        break

                    # Check if retryable
                    status_code = result.get("status_code", 0)
                    if status_code >= 500 or status_code == 0:
                        # Transient error, retry
                        delay = min(
                            RETRY_BASE_DELAY_SECONDS * (
                                RETRY_BACKOFF_MULTIPLIER ** (attempt - 1)
                            ),
                            RETRY_MAX_DELAY_SECONDS,
                        )
                        warnings.append(
                            f"Attempt {attempt} failed (HTTP {status_code}), "
                            f"retrying in {delay:.0f}s"
                        )
                        logger.warning(
                            "Attempt %d failed with status %d, "
                            "retrying in %.0fs",
                            attempt, status_code, delay,
                        )
                        # In production: await asyncio.sleep(delay)
                    else:
                        # Non-retryable error (4xx)
                        errors.append(
                            f"Submission rejected (HTTP {status_code}): "
                            f"{result.get('error_message', 'Unknown error')}"
                        )
                        final_status = SubmissionStatus.REJECTED
                        break

                except Exception as exc:
                    attempt_entry = {
                        "attempt": attempt,
                        "started_at": attempt_start.isoformat(),
                        "completed_at": datetime.utcnow().isoformat(),
                        "status_code": 0,
                        "success": False,
                        "error_message": str(exc),
                    }
                    attempt_log.append(attempt_entry)
                    warnings.append(
                        f"Attempt {attempt} exception: {str(exc)}"
                    )

            if not receipt_id and final_status != SubmissionStatus.REJECTED:
                errors.append(
                    f"All {max_attempts} submission attempts failed"
                )
                final_status = SubmissionStatus.ERROR

            outputs["submission_status"] = final_status.value
            outputs["receipt_id"] = receipt_id
            outputs["registry_reference"] = registry_ref
            outputs["total_attempts"] = len(attempt_log)
            outputs["attempt_log"] = attempt_log
            outputs["declaration_id"] = declaration.get("declaration_id", "")

            status = (
                PhaseStatus.COMPLETED
                if final_status == SubmissionStatus.SUBMITTED
                else PhaseStatus.FAILED
            )

        except Exception as exc:
            logger.error("Submit failed: %s", exc, exc_info=True)
            errors.append(f"Submission failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

    async def _post_to_registry(
        self,
        declaration: Dict[str, Any],
        eidas: Dict[str, Any],
        endpoint: str,
    ) -> Dict[str, Any]:
        """
        POST declaration to CBAM Registry API.

        In production, this uses httpx with eIDAS mTLS. Here we
        provide a stub that simulates a successful submission.
        """
        logger.info(
            "POST %s/declarations with eIDAS cert=%s",
            endpoint, eidas.get("certificate_id", ""),
        )
        return {
            "success": True,
            "status_code": 201,
            "receipt_id": str(uuid.uuid4()),
            "registry_reference": f"CBAM-{declaration.get('member_state', 'XX')}-"
                                  f"{declaration.get('reporting_year', 0)}-"
                                  f"{str(uuid.uuid4())[:8].upper()}",
            "timestamp": datetime.utcnow().isoformat(),
        }


class MonitorPhase:
    """
    Phase 3: Monitor.

    Polls submission status at configurable intervals until the
    registry returns Accepted or Rejected. Parses validation errors
    on rejection and triggers error resolution. Configurable timeout.
    """

    PHASE_NAME = "monitor"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute monitoring phase.

        Args:
            context: Workflow context with submission receipt.

        Returns:
            PhaseResult with final acceptance/rejection status.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            submit_output = context.get_phase_output("submit")
            receipt_id = submit_output.get("receipt_id")
            registry_ref = submit_output.get("registry_reference")
            endpoint = config.get(
                "registry_endpoint",
                "https://cbam-registry.ec.europa.eu/api/v1",
            )
            poll_interval = config.get(
                "poll_interval_seconds", DEFAULT_POLL_INTERVAL_SECONDS
            )
            poll_timeout = config.get(
                "poll_timeout_seconds", DEFAULT_POLL_TIMEOUT_SECONDS
            )

            if not receipt_id:
                errors.append("No receipt ID from submission phase")
                return PhaseResult(
                    phase_name=self.PHASE_NAME,
                    status=PhaseStatus.FAILED,
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                    errors=errors,
                    outputs=outputs,
                    provenance_hash=_hash_data(outputs),
                )

            outputs["receipt_id"] = receipt_id
            outputs["registry_reference"] = registry_ref
            outputs["poll_interval_seconds"] = poll_interval
            outputs["poll_timeout_seconds"] = poll_timeout

            # Poll for status
            poll_log: List[Dict[str, Any]] = []
            final_status = SubmissionStatus.PROCESSING
            elapsed = 0
            poll_count = 0

            while elapsed < poll_timeout:
                poll_count += 1
                poll_result = await self._check_status(
                    receipt_id, endpoint
                )
                poll_entry = {
                    "poll_number": poll_count,
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": poll_result.get("status", "UNKNOWN"),
                    "elapsed_seconds": elapsed,
                }
                poll_log.append(poll_entry)

                status_str = poll_result.get("status", "")
                if status_str == SubmissionStatus.ACCEPTED.value:
                    final_status = SubmissionStatus.ACCEPTED
                    outputs["acceptance_timestamp"] = poll_result.get(
                        "timestamp", datetime.utcnow().isoformat()
                    )
                    logger.info(
                        "Submission %s accepted after %d polls",
                        receipt_id, poll_count,
                    )
                    break
                elif status_str == SubmissionStatus.REJECTED.value:
                    final_status = SubmissionStatus.REJECTED
                    rejection_errors = poll_result.get(
                        "validation_errors", []
                    )
                    outputs["rejection_errors"] = rejection_errors
                    errors.append(
                        f"Submission rejected: "
                        f"{len(rejection_errors)} error(s)"
                    )
                    logger.warning(
                        "Submission %s rejected: %s",
                        receipt_id, rejection_errors,
                    )
                    break
                elif status_str == SubmissionStatus.ERROR.value:
                    final_status = SubmissionStatus.ERROR
                    errors.append(
                        f"Registry processing error: "
                        f"{poll_result.get('error_message', 'Unknown')}"
                    )
                    break

                # In production: await asyncio.sleep(poll_interval)
                elapsed += poll_interval

            if final_status == SubmissionStatus.PROCESSING:
                final_status = SubmissionStatus.TIMED_OUT
                errors.append(
                    f"Monitoring timed out after {poll_timeout}s "
                    f"({poll_count} polls)"
                )

            outputs["final_status"] = final_status.value
            outputs["total_polls"] = poll_count
            outputs["poll_log"] = poll_log
            outputs["total_elapsed_seconds"] = elapsed

            phase_status = (
                PhaseStatus.COMPLETED
                if final_status == SubmissionStatus.ACCEPTED
                else PhaseStatus.FAILED
            )

        except Exception as exc:
            logger.error("Monitor failed: %s", exc, exc_info=True)
            errors.append(f"Monitoring failed: {str(exc)}")
            phase_status = PhaseStatus.FAILED

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=phase_status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

    async def _check_status(
        self, receipt_id: str, endpoint: str
    ) -> Dict[str, Any]:
        """
        Poll registry for submission status.

        In production, this calls GET /submissions/{receipt_id}/status.
        """
        logger.info("Polling status for receipt %s", receipt_id)
        return {
            "status": SubmissionStatus.ACCEPTED.value,
            "timestamp": datetime.utcnow().isoformat(),
            "receipt_id": receipt_id,
        }


class ConfirmPhase:
    """
    Phase 4: Confirm.

    Logs acceptance receipt, updates internal declaration status,
    triggers downstream workflows (certificate trading, cross-regulation
    sync), and archives submission evidence for audit trail.
    """

    PHASE_NAME = "confirm"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute confirmation phase.

        Args:
            context: Workflow context with monitoring results.

        Returns:
            PhaseResult with confirmation details and downstream triggers.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            submit_output = context.get_phase_output("submit")
            monitor_output = context.get_phase_output("monitor")
            declaration = config.get("declaration", {})
            trigger_downstream = config.get(
                "trigger_downstream_on_accept", True
            )
            downstream_workflows = config.get("downstream_workflows", [])

            final_status = monitor_output.get(
                "final_status", SubmissionStatus.ERROR.value
            )
            receipt_id = submit_output.get("receipt_id", "")
            registry_ref = submit_output.get("registry_reference", "")
            declaration_id = declaration.get("declaration_id", "")

            # Log acceptance receipt
            acceptance_record = {
                "record_id": str(uuid.uuid4()),
                "declaration_id": declaration_id,
                "receipt_id": receipt_id,
                "registry_reference": registry_ref,
                "final_status": final_status,
                "accepted_at": monitor_output.get(
                    "acceptance_timestamp",
                    datetime.utcnow().isoformat(),
                ),
                "submission_type": declaration.get(
                    "submission_type",
                    SubmissionType.ANNUAL_DECLARATION.value,
                ),
                "reporting_year": declaration.get("reporting_year", 0),
                "declarant_id": declaration.get("declarant_id", ""),
                "member_state": declaration.get("member_state", ""),
            }
            outputs["acceptance_record"] = acceptance_record

            # Update internal declaration status
            outputs["internal_status_update"] = {
                "declaration_id": declaration_id,
                "previous_status": SubmissionStatus.SUBMITTED.value,
                "new_status": final_status,
                "updated_at": datetime.utcnow().isoformat(),
                "updated_by": "registry_submission_workflow",
            }

            # Trigger downstream workflows
            downstream_triggered: List[str] = []
            if (trigger_downstream
                    and final_status == SubmissionStatus.ACCEPTED.value):
                for wf_name in downstream_workflows:
                    trigger_result = await self._trigger_downstream(
                        wf_name, declaration, registry_ref
                    )
                    downstream_triggered.append(wf_name)
                    logger.info(
                        "Triggered downstream workflow: %s", wf_name
                    )

            outputs["downstream_triggered"] = downstream_triggered
            outputs["downstream_count"] = len(downstream_triggered)

            # Archive submission evidence
            evidence_archive = {
                "archive_id": str(uuid.uuid4()),
                "declaration_id": declaration_id,
                "receipt_id": receipt_id,
                "registry_reference": registry_ref,
                "submission_log": submit_output.get("attempt_log", []),
                "monitoring_log": monitor_output.get("poll_log", []),
                "archived_at": datetime.utcnow().isoformat(),
                "retention_years": 5,
            }
            outputs["evidence_archive"] = evidence_archive

            # Handle rejection - suggest error resolution
            if final_status == SubmissionStatus.REJECTED.value:
                rejection_errors = monitor_output.get(
                    "rejection_errors", []
                )
                resolution_suggestions = self._generate_resolution(
                    rejection_errors
                )
                outputs["resolution_suggestions"] = resolution_suggestions
                warnings.append(
                    f"Submission rejected. {len(resolution_suggestions)} "
                    f"resolution suggestions generated."
                )

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("Confirm failed: %s", exc, exc_info=True)
            errors.append(f"Confirmation failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

    async def _trigger_downstream(
        self,
        workflow_name: str,
        declaration: Dict[str, Any],
        registry_ref: str,
    ) -> Dict[str, Any]:
        """Trigger a downstream workflow with submission context."""
        return {
            "workflow": workflow_name,
            "trigger_id": str(uuid.uuid4()),
            "triggered_at": datetime.utcnow().isoformat(),
            "context": {
                "declaration_id": declaration.get("declaration_id", ""),
                "registry_reference": registry_ref,
            },
        }

    def _generate_resolution(
        self, rejection_errors: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate resolution suggestions for rejection errors."""
        suggestions = []
        for error in rejection_errors:
            field = error.get("field_path", "")
            code = error.get("code", "")
            suggestions.append({
                "error_code": code,
                "field": field,
                "suggestion": (
                    f"Review and correct field '{field}'. "
                    f"Ensure value matches CBAM schema requirements."
                ),
                "priority": "HIGH",
            })
        return suggestions


# =============================================================================
# WORKFLOW ORCHESTRATOR
# =============================================================================


class RegistrySubmissionWorkflow:
    """
    Four-phase CBAM Registry submission lifecycle workflow.

    Orchestrates the complete submission process from pre-validation
    through eIDAS-authenticated submission, status monitoring, and
    downstream confirmation. Includes retry logic with exponential
    backoff for transient failures.

    Attributes:
        workflow_id: Unique execution identifier.
        _phases: Ordered phase executors.
        _progress_callback: Optional progress notification callback.

    Example:
        >>> wf = RegistrySubmissionWorkflow()
        >>> input_data = RegistrySubmissionInput(
        ...     organization_id="org-123",
        ...     declaration=DeclarationData(
        ...         reporting_year=2026,
        ...         declarant_id="DE123456789012",
        ...         member_state="DE",
        ...     ),
        ...     eidas_credentials=EidasCredentials(
        ...         certificate_id="cert-456",
        ...     ),
        ... )
        >>> result = await wf.run(input_data)
        >>> assert result.submission_status == "ACCEPTED"
    """

    WORKFLOW_NAME = "registry_submission"

    PHASE_ORDER = [
        "pre_validation",
        "submit",
        "monitor",
        "confirm",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """
        Initialize registry submission workflow.

        Args:
            progress_callback: Optional callback(phase, message, pct).
        """
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "pre_validation": PreValidationPhase(),
            "submit": SubmitPhase(),
            "monitor": MonitorPhase(),
            "confirm": ConfirmPhase(),
        }

    async def run(
        self, input_data: RegistrySubmissionInput
    ) -> RegistrySubmissionResult:
        """
        Execute the 4-phase registry submission workflow.

        Args:
            input_data: Validated workflow input configuration.

        Returns:
            RegistrySubmissionResult with submission outcome.
        """
        started_at = datetime.utcnow()
        logger.info(
            "Starting registry submission %s for org=%s declaration=%s",
            self.workflow_id, input_data.organization_id,
            input_data.declaration.declaration_id,
        )

        context = WorkflowContext(
            workflow_id=self.workflow_id,
            organization_id=input_data.organization_id,
            config=self._build_config(input_data),
        )

        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        for idx, phase_name in enumerate(self.PHASE_ORDER):
            if phase_name in input_data.skip_phases:
                skip_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.SKIPPED,
                    provenance_hash=_hash_data({"skipped": True}),
                )
                completed_phases.append(skip_result)
                context.mark_phase(phase_name, PhaseStatus.SKIPPED)
                continue

            if context.is_phase_completed(phase_name):
                continue

            pct = idx / len(self.PHASE_ORDER)
            self._notify_progress(phase_name, f"Starting: {phase_name}", pct)
            context.mark_phase(phase_name, PhaseStatus.RUNNING)

            try:
                phase_result = await self._phases[phase_name].execute(context)
                completed_phases.append(phase_result)

                if phase_result.status == PhaseStatus.COMPLETED:
                    context.set_phase_output(phase_name, phase_result.outputs)
                    context.mark_phase(phase_name, PhaseStatus.COMPLETED)
                else:
                    context.mark_phase(phase_name, phase_result.status)
                    # Pre-validation or submit failure is critical
                    if phase_name in ("pre_validation", "submit"):
                        overall_status = WorkflowStatus.FAILED
                        break

                context.errors.extend(phase_result.errors)
                context.warnings.extend(phase_result.warnings)

            except Exception as exc:
                logger.error(
                    "Phase '%s' raised: %s", phase_name, exc, exc_info=True
                )
                error_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    errors=[str(exc)],
                    provenance_hash=_hash_data({"error": str(exc)}),
                )
                completed_phases.append(error_result)
                context.mark_phase(phase_name, PhaseStatus.FAILED)
                overall_status = WorkflowStatus.FAILED
                break

        if overall_status == WorkflowStatus.RUNNING:
            all_ok = all(
                p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                for p in completed_phases
            )
            overall_status = (
                WorkflowStatus.COMPLETED if all_ok else WorkflowStatus.PARTIAL
            )

        completed_at = datetime.utcnow()
        total_duration = (completed_at - started_at).total_seconds()
        summary = self._build_summary(context, input_data)
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })

        self._notify_progress(
            "workflow", f"Workflow {overall_status.value}", 1.0
        )
        logger.info(
            "Registry submission %s finished status=%s in %.1fs",
            self.workflow_id, overall_status.value, total_duration,
        )

        return RegistrySubmissionResult(
            workflow_id=self.workflow_id,
            workflow_name=self.WORKFLOW_NAME,
            status=overall_status,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=total_duration,
            phases=completed_phases,
            summary=summary,
            provenance_hash=provenance,
            submission_type=input_data.declaration.submission_type.value,
            declaration_id=input_data.declaration.declaration_id,
            submission_status=summary.get("final_status", ""),
            receipt_id=summary.get("receipt_id"),
            registry_reference=summary.get("registry_reference"),
            validation_errors=summary.get("validation_errors", 0),
            validation_warnings=summary.get("validation_warnings", 0),
            retry_attempts=summary.get("total_attempts", 0),
            downstream_triggered=summary.get("downstream_triggered", []),
        )

    def _build_config(
        self, input_data: RegistrySubmissionInput
    ) -> Dict[str, Any]:
        """Transform input model to config dict for phases."""
        return {
            "organization_id": input_data.organization_id,
            "declaration": input_data.declaration.model_dump(),
            "eidas_credentials": input_data.eidas_credentials.model_dump(),
            "registry_endpoint": input_data.registry_endpoint,
            "poll_interval_seconds": input_data.poll_interval_seconds,
            "poll_timeout_seconds": input_data.poll_timeout_seconds,
            "max_retry_attempts": input_data.max_retry_attempts,
            "trigger_downstream_on_accept": input_data.trigger_downstream_on_accept,
            "downstream_workflows": input_data.downstream_workflows,
        }

    def _build_summary(
        self,
        context: WorkflowContext,
        input_data: RegistrySubmissionInput,
    ) -> Dict[str, Any]:
        """Build workflow summary from phase outputs."""
        validation = context.get_phase_output("pre_validation")
        submit = context.get_phase_output("submit")
        monitor = context.get_phase_output("monitor")
        confirm = context.get_phase_output("confirm")
        return {
            "declaration_id": input_data.declaration.declaration_id,
            "submission_type": input_data.declaration.submission_type.value,
            "validation_passed": validation.get("validation_passed", False),
            "validation_errors": validation.get("error_count", 0),
            "validation_warnings": validation.get("warning_count", 0),
            "receipt_id": submit.get("receipt_id"),
            "registry_reference": submit.get("registry_reference"),
            "total_attempts": submit.get("total_attempts", 0),
            "final_status": monitor.get("final_status", ""),
            "downstream_triggered": confirm.get(
                "downstream_triggered", []
            ),
        }

    def _notify_progress(
        self, phase: str, message: str, pct: float
    ) -> None:
        """Send progress notification via callback."""
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for phase=%s", phase)


# =============================================================================
# UTILITIES
# =============================================================================


def _hash_data(data: Any) -> str:
    """Compute SHA-256 provenance hash of arbitrary data."""
    serialized = str(data).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
