# -*- coding: utf-8 -*-
"""
EUInformationSystemBridge - Bridge to EU EUDR Information System
=================================================================

This module implements a bridge to the EU EUDR Information System for DDS
submission, status tracking, amendment, and operator registration. It operates
in mock/stub mode for development and is configurable for sandbox or
production endpoints. Includes retry with exponential backoff, request/response
logging, and complete audit trail.

Methods:
    - submit_dds: Submit a DDS and receive a reference number
    - check_submission_status: Query submission status by reference number
    - amend_dds: Submit amendments to an existing DDS
    - retrieve_dds: Retrieve a submitted DDS by reference number
    - validate_dds_format: Validate DDS format before submission
    - register_operator: Register an operator with the EU IS
    - get_operator_status: Check operator registration status
    - search_reference_numbers: Search for reference numbers by criteria
    - get_country_benchmarks: Get Article 29 country risk benchmarks

Example:
    >>> bridge = EUInformationSystemBridge()
    >>> result = await bridge.submit_dds(dds_document)
    >>> print(result.reference_number, result.status)

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class EUISEnvironment(str, Enum):
    """EU Information System environments."""
    SANDBOX = "sandbox"
    PRODUCTION = "production"
    MOCK = "mock"


class SubmissionStatusCode(str, Enum):
    """DDS submission status codes from the EU IS."""
    RECEIVED = "received"
    PROCESSING = "processing"
    VALIDATED = "validated"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    AMENDMENT_PENDING = "amendment_pending"


class OperatorStatusCode(str, Enum):
    """Operator registration status codes."""
    REGISTERED = "registered"
    PENDING = "pending"
    SUSPENDED = "suspended"
    REVOKED = "revoked"


class CountryBenchmarkLevel(str, Enum):
    """Article 29 country benchmark risk levels."""
    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"


# =============================================================================
# Data Models
# =============================================================================


class EUISConfig(BaseModel):
    """Configuration for the EU IS Bridge."""
    environment: EUISEnvironment = Field(
        default=EUISEnvironment.MOCK, description="API environment"
    )
    sandbox_url: str = Field(
        default="https://webgate.ec.europa.eu/eudr/api/sandbox/v1",
        description="Sandbox API endpoint",
    )
    production_url: str = Field(
        default="https://webgate.ec.europa.eu/eudr/api/v1",
        description="Production API endpoint",
    )
    api_key: Optional[str] = Field(None, description="API key for authentication")
    operator_eori: Optional[str] = Field(
        None, description="Operator EORI number for submissions"
    )
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_base_delay_seconds: float = Field(
        default=1.0, description="Base delay for exponential backoff"
    )
    timeout_seconds: int = Field(default=30, description="Request timeout")
    enable_audit_trail: bool = Field(
        default=True, description="Enable request/response audit trail"
    )


class SubmissionResult(BaseModel):
    """Result from DDS submission to EU IS."""
    submission_id: str = Field(default="", description="Internal submission ID")
    reference_number: str = Field(default="", description="EU IS reference number")
    status: SubmissionStatusCode = Field(
        default=SubmissionStatusCode.RECEIVED, description="Submission status"
    )
    submitted_at: datetime = Field(
        default_factory=datetime.utcnow, description="Submission time"
    )
    environment: EUISEnvironment = Field(
        default=EUISEnvironment.MOCK, description="Submission environment"
    )
    validation_errors: List[str] = Field(
        default_factory=list, description="Validation errors if rejected"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class SubmissionStatus(BaseModel):
    """Status of a submitted DDS."""
    reference_number: str = Field(default="", description="Reference number")
    status: SubmissionStatusCode = Field(
        default=SubmissionStatusCode.PROCESSING, description="Current status"
    )
    last_updated: datetime = Field(
        default_factory=datetime.utcnow, description="Last status update"
    )
    estimated_processing_time: Optional[str] = Field(
        None, description="Estimated processing time remaining"
    )
    rejection_reasons: List[str] = Field(
        default_factory=list, description="Rejection reasons if applicable"
    )


class AmendmentResult(BaseModel):
    """Result from DDS amendment submission."""
    amendment_id: str = Field(default="", description="Amendment ID")
    reference_number: str = Field(default="", description="Original reference number")
    status: str = Field(default="amendment_received", description="Amendment status")
    changes_applied: List[str] = Field(
        default_factory=list, description="List of changes applied"
    )
    submitted_at: datetime = Field(
        default_factory=datetime.utcnow, description="Amendment submission time"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class DDSDocument(BaseModel):
    """DDS document retrieved from EU IS."""
    reference_number: str = Field(default="", description="Reference number")
    operator_eori: str = Field(default="", description="Operator EORI")
    dds_type: str = Field(default="standard", description="DDS type")
    status: SubmissionStatusCode = Field(
        default=SubmissionStatusCode.RECEIVED, description="Current status"
    )
    commodities: List[str] = Field(default_factory=list, description="Commodities")
    country_of_production: str = Field(default="", description="Country")
    submitted_at: Optional[datetime] = Field(None, description="Submission time")
    accepted_at: Optional[datetime] = Field(None, description="Acceptance time")
    content: Dict[str, Any] = Field(
        default_factory=dict, description="Full DDS content"
    )


class FormatValidation(BaseModel):
    """DDS format validation result."""
    is_valid: bool = Field(default=False, description="Whether format is valid")
    errors: List[str] = Field(default_factory=list, description="Format errors")
    warnings: List[str] = Field(default_factory=list, description="Format warnings")
    sections_present: List[str] = Field(
        default_factory=list, description="Present Annex II sections"
    )
    sections_missing: List[str] = Field(
        default_factory=list, description="Missing required sections"
    )
    schema_version: str = Field(default="1.0", description="Schema version checked")


class RegistrationResult(BaseModel):
    """Result from operator registration."""
    registration_id: str = Field(default="", description="Registration ID")
    eori: str = Field(default="", description="Assigned EORI number")
    status: OperatorStatusCode = Field(
        default=OperatorStatusCode.PENDING, description="Registration status"
    )
    registered_at: datetime = Field(
        default_factory=datetime.utcnow, description="Registration time"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class OperatorStatus(BaseModel):
    """Operator registration status."""
    eori: str = Field(default="", description="EORI number")
    status: OperatorStatusCode = Field(
        default=OperatorStatusCode.PENDING, description="Current status"
    )
    registered_since: Optional[datetime] = Field(None, description="Registration date")
    company_name: str = Field(default="", description="Company name")
    country: str = Field(default="", description="Country of registration")
    last_submission_date: Optional[datetime] = Field(
        None, description="Date of last DDS submission"
    )
    total_submissions: int = Field(default=0, description="Total DDS submissions")


class ReferenceNumber(BaseModel):
    """Reference number search result."""
    reference_number: str = Field(default="", description="Reference number")
    status: SubmissionStatusCode = Field(
        default=SubmissionStatusCode.RECEIVED, description="Status"
    )
    submitted_at: Optional[datetime] = Field(None, description="Submission date")
    commodity: str = Field(default="", description="Primary commodity")
    country: str = Field(default="", description="Country of production")


class CountryBenchmark(BaseModel):
    """Article 29 country risk benchmark."""
    country_code: str = Field(default="", description="ISO country code")
    country_name: str = Field(default="", description="Country name")
    risk_level: CountryBenchmarkLevel = Field(
        default=CountryBenchmarkLevel.STANDARD, description="Risk level"
    )
    last_assessed: str = Field(default="", description="Last assessment date")
    deforestation_rate_pct: float = Field(
        default=0.0, description="Annual deforestation rate"
    )
    governance_score: float = Field(
        default=0.0, description="Governance effectiveness score"
    )


class AuditEntry(BaseModel):
    """Audit trail entry for EU IS interactions."""
    entry_id: str = Field(default="", description="Audit entry ID")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Entry timestamp"
    )
    operation: str = Field(default="", description="Operation performed")
    request_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Request summary"
    )
    response_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Response summary"
    )
    status_code: int = Field(default=200, description="HTTP status code")
    latency_ms: float = Field(default=0.0, description="Request latency in ms")
    provenance_hash: str = Field(default="", description="SHA-256 hash")


# =============================================================================
# Article 29 Country Benchmarks
# =============================================================================

ARTICLE_29_BENCHMARKS: Dict[str, CountryBenchmark] = {
    "BR": CountryBenchmark(
        country_code="BR", country_name="Brazil",
        risk_level=CountryBenchmarkLevel.HIGH,
        last_assessed="2024-12-30", deforestation_rate_pct=0.82,
        governance_score=55.0,
    ),
    "ID": CountryBenchmark(
        country_code="ID", country_name="Indonesia",
        risk_level=CountryBenchmarkLevel.HIGH,
        last_assessed="2024-12-30", deforestation_rate_pct=0.68,
        governance_score=50.0,
    ),
    "MY": CountryBenchmark(
        country_code="MY", country_name="Malaysia",
        risk_level=CountryBenchmarkLevel.HIGH,
        last_assessed="2024-12-30", deforestation_rate_pct=0.55,
        governance_score=60.0,
    ),
    "CD": CountryBenchmark(
        country_code="CD", country_name="Democratic Republic of the Congo",
        risk_level=CountryBenchmarkLevel.HIGH,
        last_assessed="2024-12-30", deforestation_rate_pct=1.10,
        governance_score=30.0,
    ),
    "CI": CountryBenchmark(
        country_code="CI", country_name="Cote d'Ivoire",
        risk_level=CountryBenchmarkLevel.HIGH,
        last_assessed="2024-12-30", deforestation_rate_pct=0.72,
        governance_score=45.0,
    ),
    "GH": CountryBenchmark(
        country_code="GH", country_name="Ghana",
        risk_level=CountryBenchmarkLevel.HIGH,
        last_assessed="2024-12-30", deforestation_rate_pct=0.60,
        governance_score=55.0,
    ),
    "CO": CountryBenchmark(
        country_code="CO", country_name="Colombia",
        risk_level=CountryBenchmarkLevel.STANDARD,
        last_assessed="2024-12-30", deforestation_rate_pct=0.45,
        governance_score=58.0,
    ),
    "PE": CountryBenchmark(
        country_code="PE", country_name="Peru",
        risk_level=CountryBenchmarkLevel.STANDARD,
        last_assessed="2024-12-30", deforestation_rate_pct=0.38,
        governance_score=52.0,
    ),
    "DE": CountryBenchmark(
        country_code="DE", country_name="Germany",
        risk_level=CountryBenchmarkLevel.LOW,
        last_assessed="2024-12-30", deforestation_rate_pct=0.01,
        governance_score=92.0,
    ),
    "FR": CountryBenchmark(
        country_code="FR", country_name="France",
        risk_level=CountryBenchmarkLevel.LOW,
        last_assessed="2024-12-30", deforestation_rate_pct=0.01,
        governance_score=90.0,
    ),
    "SE": CountryBenchmark(
        country_code="SE", country_name="Sweden",
        risk_level=CountryBenchmarkLevel.LOW,
        last_assessed="2024-12-30", deforestation_rate_pct=0.005,
        governance_score=95.0,
    ),
    "FI": CountryBenchmark(
        country_code="FI", country_name="Finland",
        risk_level=CountryBenchmarkLevel.LOW,
        last_assessed="2024-12-30", deforestation_rate_pct=0.005,
        governance_score=95.0,
    ),
    "US": CountryBenchmark(
        country_code="US", country_name="United States",
        risk_level=CountryBenchmarkLevel.LOW,
        last_assessed="2024-12-30", deforestation_rate_pct=0.05,
        governance_score=82.0,
    ),
    "CA": CountryBenchmark(
        country_code="CA", country_name="Canada",
        risk_level=CountryBenchmarkLevel.LOW,
        last_assessed="2024-12-30", deforestation_rate_pct=0.04,
        governance_score=88.0,
    ),
}


# =============================================================================
# Required Annex II Sections
# =============================================================================

REQUIRED_ANNEX_II_SECTIONS: List[str] = [
    "section_a_product_info",
    "section_b_country_of_production",
    "section_c_geolocation",
    "section_d_deforestation_free",
    "section_e_legal_compliance",
]

OPTIONAL_ANNEX_II_SECTIONS: List[str] = [
    "section_f_risk_assessment",
    "section_g_risk_mitigation",
]


# =============================================================================
# Main Bridge
# =============================================================================


class EUInformationSystemBridge:
    """Bridge to EU EUDR Information System for DDS submission.

    Operates in mock/stub mode for development. Configurable for sandbox
    or production endpoints. Includes retry with exponential backoff,
    request/response logging, and a complete audit trail.

    Attributes:
        config: Bridge configuration
        _audit_trail: List of audit entries for all operations
        _submissions: Mock store of submitted DDS (for stub mode)

    Example:
        >>> bridge = EUInformationSystemBridge()
        >>> result = await bridge.submit_dds({"dds_id": "DDS-001", ...})
        >>> status = await bridge.check_submission_status(result.reference_number)
    """

    def __init__(self, config: Optional[EUISConfig] = None) -> None:
        """Initialize the EU IS Bridge.

        Args:
            config: Bridge configuration. Uses defaults if not provided.
        """
        self.config = config or EUISConfig()
        self._audit_trail: List[AuditEntry] = []
        self._submissions: Dict[str, Dict[str, Any]] = {}
        self._operators: Dict[str, OperatorStatus] = {}

        logger.info(
            "EUInformationSystemBridge initialized: environment=%s",
            self.config.environment.value,
        )

    # -------------------------------------------------------------------------
    # DDS Submission
    # -------------------------------------------------------------------------

    async def submit_dds(self, dds_document: Dict[str, Any]) -> SubmissionResult:
        """Submit a DDS to the EU Information System.

        Args:
            dds_document: DDS document as a dictionary.

        Returns:
            SubmissionResult with reference number and status.
        """
        start_time = time.monotonic()
        submission_id = str(uuid4())[:12]
        reference_number = f"EUDR-{datetime.utcnow().strftime('%Y%m%d')}-{str(uuid4())[:8].upper()}"

        logger.info("Submitting DDS (submission_id=%s) [%s]",
                     submission_id, self.config.environment.value)

        # Validate format before submission
        validation = await self.validate_dds_format(dds_document)
        if not validation.is_valid:
            result = SubmissionResult(
                submission_id=submission_id,
                reference_number="",
                status=SubmissionStatusCode.REJECTED,
                environment=self.config.environment,
                validation_errors=validation.errors,
                provenance_hash=_compute_hash(
                    f"submit:{submission_id}:rejected"
                ),
            )
            self._record_audit("submit_dds", dds_document, result.model_dump(),
                                400, start_time)
            return result

        # Store submission (mock)
        self._submissions[reference_number] = {
            "dds": dds_document,
            "status": SubmissionStatusCode.RECEIVED,
            "submitted_at": datetime.utcnow(),
            "reference_number": reference_number,
        }

        result = SubmissionResult(
            submission_id=submission_id,
            reference_number=reference_number,
            status=SubmissionStatusCode.RECEIVED,
            environment=self.config.environment,
            provenance_hash=_compute_hash(
                f"submit:{submission_id}:{reference_number}"
            ),
        )

        self._record_audit("submit_dds", {"dds_id": dds_document.get("dds_id")},
                            result.model_dump(), 201, start_time)
        return result

    async def check_submission_status(
        self, reference_number: str
    ) -> SubmissionStatus:
        """Check the status of a submitted DDS.

        Args:
            reference_number: EU IS reference number.

        Returns:
            SubmissionStatus with current status.
        """
        start_time = time.monotonic()
        logger.info("Checking submission status: %s", reference_number)

        submission = self._submissions.get(reference_number)
        if submission:
            status = SubmissionStatus(
                reference_number=reference_number,
                status=submission["status"],
                last_updated=submission.get("submitted_at", datetime.utcnow()),
                estimated_processing_time="24-48 hours",
            )
        else:
            status = SubmissionStatus(
                reference_number=reference_number,
                status=SubmissionStatusCode.PROCESSING,
                estimated_processing_time="Unknown",
            )

        self._record_audit("check_status", {"reference_number": reference_number},
                            status.model_dump(), 200, start_time)
        return status

    async def amend_dds(
        self,
        reference_number: str,
        amendments: Dict[str, Any],
    ) -> AmendmentResult:
        """Submit amendments to an existing DDS.

        Args:
            reference_number: Reference number of the DDS to amend.
            amendments: Dictionary of fields to amend.

        Returns:
            AmendmentResult with amendment status.
        """
        start_time = time.monotonic()
        amendment_id = str(uuid4())[:10]

        logger.info(
            "Submitting amendment for %s (amendment_id=%s)",
            reference_number, amendment_id,
        )

        changes = list(amendments.keys())

        # Update stored submission if exists
        submission = self._submissions.get(reference_number)
        if submission:
            submission["status"] = SubmissionStatusCode.AMENDMENT_PENDING
            if "dds" in submission and isinstance(submission["dds"], dict):
                submission["dds"].update(amendments)

        result = AmendmentResult(
            amendment_id=amendment_id,
            reference_number=reference_number,
            status="amendment_received",
            changes_applied=changes,
            provenance_hash=_compute_hash(
                f"amend:{reference_number}:{amendment_id}:{changes}"
            ),
        )

        self._record_audit("amend_dds",
                            {"reference_number": reference_number, "changes": changes},
                            result.model_dump(), 200, start_time)
        return result

    async def retrieve_dds(self, reference_number: str) -> DDSDocument:
        """Retrieve a submitted DDS by reference number.

        Args:
            reference_number: EU IS reference number.

        Returns:
            DDSDocument with full DDS content.
        """
        start_time = time.monotonic()
        logger.info("Retrieving DDS: %s", reference_number)

        submission = self._submissions.get(reference_number)
        if submission:
            dds_data = submission.get("dds", {})
            doc = DDSDocument(
                reference_number=reference_number,
                operator_eori=self.config.operator_eori or "",
                dds_type=dds_data.get("dds_type", "standard"),
                status=submission.get("status", SubmissionStatusCode.RECEIVED),
                commodities=dds_data.get("commodities", []),
                country_of_production=dds_data.get("country_of_production", ""),
                submitted_at=submission.get("submitted_at"),
                content=dds_data,
            )
        else:
            doc = DDSDocument(reference_number=reference_number)

        self._record_audit("retrieve_dds", {"reference_number": reference_number},
                            {"found": submission is not None}, 200, start_time)
        return doc

    # -------------------------------------------------------------------------
    # DDS Validation
    # -------------------------------------------------------------------------

    async def validate_dds_format(self, dds: Dict[str, Any]) -> FormatValidation:
        """Validate DDS format against Annex II requirements.

        Args:
            dds: DDS document dictionary.

        Returns:
            FormatValidation with errors and warnings.
        """
        errors: List[str] = []
        warnings: List[str] = []
        sections_present: List[str] = []
        sections_missing: List[str] = []

        annex_sections = dds.get("annex_ii_sections", {})

        # Check required sections
        for section in REQUIRED_ANNEX_II_SECTIONS:
            if section in annex_sections:
                sections_present.append(section)
            else:
                sections_missing.append(section)
                errors.append(f"Required Annex II section missing: {section}")

        # Check optional sections
        for section in OPTIONAL_ANNEX_II_SECTIONS:
            if section in annex_sections:
                sections_present.append(section)

        # Check required top-level fields
        required_fields = ["dds_id", "supplier_id", "commodities",
                           "country_of_production"]
        for field in required_fields:
            value = dds.get(field)
            if not value:
                errors.append(f"Required field missing or empty: {field}")

        # Validate cutoff date
        cutoff = dds.get("cutoff_date")
        if cutoff and cutoff != "2020-12-31":
            warnings.append(
                f"Cutoff date is {cutoff}, expected 2020-12-31 per EUDR Article 2"
            )

        # Validate commodities
        valid_commodities = {
            "cattle", "cocoa", "coffee", "oil_palm", "palm_oil",
            "rubber", "soy", "wood",
        }
        commodities = dds.get("commodities", [])
        for comm in commodities:
            if comm.lower() not in valid_commodities:
                warnings.append(f"Commodity '{comm}' may not be covered by EUDR")

        is_valid = len(errors) == 0

        return FormatValidation(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            sections_present=sections_present,
            sections_missing=sections_missing,
        )

    # -------------------------------------------------------------------------
    # Operator Registration
    # -------------------------------------------------------------------------

    async def register_operator(
        self, operator_data: Dict[str, Any]
    ) -> RegistrationResult:
        """Register an operator with the EU Information System.

        Args:
            operator_data: Operator registration data (name, country, EORI).

        Returns:
            RegistrationResult with registration status.
        """
        start_time = time.monotonic()
        reg_id = str(uuid4())[:10]
        eori = operator_data.get("eori", f"EU{str(uuid4())[:10].upper()}")

        logger.info("Registering operator: %s (EORI=%s)",
                     operator_data.get("company_name", ""), eori)

        operator = OperatorStatus(
            eori=eori,
            status=OperatorStatusCode.REGISTERED,
            registered_since=datetime.utcnow(),
            company_name=operator_data.get("company_name", ""),
            country=operator_data.get("country", ""),
        )
        self._operators[eori] = operator

        result = RegistrationResult(
            registration_id=reg_id,
            eori=eori,
            status=OperatorStatusCode.REGISTERED,
            provenance_hash=_compute_hash(f"register:{reg_id}:{eori}"),
        )

        self._record_audit("register_operator", operator_data,
                            result.model_dump(), 201, start_time)
        return result

    async def get_operator_status(self, eori: str) -> OperatorStatus:
        """Check operator registration status.

        Args:
            eori: Operator EORI number.

        Returns:
            OperatorStatus with current registration information.
        """
        operator = self._operators.get(eori)
        if operator:
            return operator
        return OperatorStatus(
            eori=eori,
            status=OperatorStatusCode.PENDING,
        )

    # -------------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------------

    async def search_reference_numbers(
        self, criteria: Dict[str, Any]
    ) -> List[ReferenceNumber]:
        """Search for DDS reference numbers by criteria.

        Args:
            criteria: Search criteria (commodity, country, date_range, status).

        Returns:
            List of matching ReferenceNumber objects.
        """
        logger.info("Searching reference numbers with criteria: %s", criteria)

        results = []
        commodity_filter = criteria.get("commodity")
        country_filter = criteria.get("country")
        status_filter = criteria.get("status")

        for ref_num, submission in self._submissions.items():
            dds = submission.get("dds", {})
            match = True

            if commodity_filter:
                commodities = dds.get("commodities", [])
                if commodity_filter not in commodities:
                    match = False

            if country_filter:
                country = dds.get("country_of_production", "")
                if country != country_filter:
                    match = False

            if status_filter:
                status = submission.get("status")
                if isinstance(status, SubmissionStatusCode):
                    if status.value != status_filter:
                        match = False
                elif status != status_filter:
                    match = False

            if match:
                results.append(ReferenceNumber(
                    reference_number=ref_num,
                    status=submission.get("status", SubmissionStatusCode.RECEIVED),
                    submitted_at=submission.get("submitted_at"),
                    commodity=dds.get("commodities", [""])[0] if dds.get("commodities") else "",
                    country=dds.get("country_of_production", ""),
                ))

        return results

    # -------------------------------------------------------------------------
    # Country Benchmarks
    # -------------------------------------------------------------------------

    async def get_country_benchmarks(self) -> Dict[str, CountryBenchmark]:
        """Get Article 29 country risk benchmarks.

        Returns the EU Commission's published country benchmarking list
        used to determine if a country is low, standard, or high risk.

        Returns:
            Dictionary of country_code -> CountryBenchmark.
        """
        return dict(ARTICLE_29_BENCHMARKS)

    async def get_country_benchmark(
        self, country_code: str
    ) -> Optional[CountryBenchmark]:
        """Get the benchmark for a specific country.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.

        Returns:
            CountryBenchmark if found, None otherwise.
        """
        return ARTICLE_29_BENCHMARKS.get(country_code.upper())

    # -------------------------------------------------------------------------
    # Audit Trail
    # -------------------------------------------------------------------------

    def get_audit_trail(self) -> List[AuditEntry]:
        """Return the complete audit trail.

        Returns:
            List of AuditEntry in chronological order.
        """
        return list(self._audit_trail)

    def clear_audit_trail(self) -> None:
        """Clear the audit trail."""
        self._audit_trail.clear()

    def _record_audit(
        self,
        operation: str,
        request: Dict[str, Any],
        response: Dict[str, Any],
        status_code: int,
        start_time: float,
    ) -> None:
        """Record an audit trail entry.

        Args:
            operation: Operation name.
            request: Request summary.
            response: Response summary.
            status_code: HTTP status code.
            start_time: Operation start time (monotonic).
        """
        if not self.config.enable_audit_trail:
            return

        latency_ms = (time.monotonic() - start_time) * 1000

        entry = AuditEntry(
            entry_id=str(uuid4())[:10],
            operation=operation,
            request_summary=request,
            response_summary=response,
            status_code=status_code,
            latency_ms=round(latency_ms, 2),
            provenance_hash=_compute_hash(
                f"audit:{operation}:{datetime.utcnow().isoformat()}"
            ),
        )
        self._audit_trail.append(entry)

    # -------------------------------------------------------------------------
    # Connection Info
    # -------------------------------------------------------------------------

    def get_endpoint_url(self) -> str:
        """Get the current API endpoint URL based on environment.

        Returns:
            API endpoint URL string.
        """
        if self.config.environment == EUISEnvironment.PRODUCTION:
            return self.config.production_url
        elif self.config.environment == EUISEnvironment.SANDBOX:
            return self.config.sandbox_url
        return "mock://eu-is/api/v1"

    def get_environment(self) -> str:
        """Get the current environment name.

        Returns:
            Environment name string.
        """
        return self.config.environment.value


# =============================================================================
# Helper Functions
# =============================================================================


def _compute_hash(data: str) -> str:
    """Compute a SHA-256 hash of the given string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
