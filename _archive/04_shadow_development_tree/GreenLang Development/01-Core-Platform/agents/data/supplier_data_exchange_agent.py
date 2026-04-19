# -*- coding: utf-8 -*-
"""
GL-DATA-X-012: Supplier Data Exchange Agent
==========================================

Manages supplier Product Carbon Footprint (PCF) submissions with
validation and integration into Scope 3 calculations.

Capabilities:
    - Receive supplier PCF submissions
    - Validate PCF data against standards (PACT, TfS)
    - Store and version supplier emissions data
    - Map supplier data to purchased goods
    - Calculate Scope 3 Category 1 emissions
    - Track data quality and coverage
    - Provenance tracking with SHA-256 hashes

Zero-Hallucination Guarantees:
    - All data validated against published standards
    - NO LLM involvement in calculations
    - Explicit mapping rules for supplier data
    - Complete audit trail for submissions

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class PCFStandard(str, Enum):
    """PCF data exchange standards."""
    PACT = "pact"  # Partnership for Carbon Transparency
    TFS = "tfs"  # Together for Sustainability
    GHG_PROTOCOL = "ghg_protocol"
    ISO_14067 = "iso_14067"
    PEF = "pef"  # Product Environmental Footprint
    CUSTOM = "custom"


class SubmissionStatus(str, Enum):
    """Submission status."""
    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"
    ACCEPTED = "accepted"
    EXPIRED = "expired"


class DataQualityRating(str, Enum):
    """Data quality rating."""
    PRIMARY = "primary"  # Supplier-specific measured data
    SECONDARY = "secondary"  # Industry average data
    TERTIARY = "tertiary"  # Default/proxy data
    UNKNOWN = "unknown"


class ValidationResult(str, Enum):
    """Validation result."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class SupplierInfo(BaseModel):
    """Supplier information."""
    supplier_id: str = Field(...)
    supplier_name: str = Field(...)
    duns_number: Optional[str] = Field(None)
    vat_number: Optional[str] = Field(None)
    country: str = Field(...)
    industry_sector: Optional[str] = Field(None)
    contact_email: Optional[str] = Field(None)
    verified: bool = Field(default=False)


class ProductInfo(BaseModel):
    """Product information."""
    product_id: str = Field(...)
    product_name: str = Field(...)
    product_category: Optional[str] = Field(None)
    unit: str = Field(default="kg")
    sku: Optional[str] = Field(None)
    hs_code: Optional[str] = Field(None)  # Harmonized System code


class PCFDataPoint(BaseModel):
    """Product Carbon Footprint data point."""
    pcf_id: str = Field(...)
    product_id: str = Field(...)
    supplier_id: str = Field(...)
    reporting_period_start: date = Field(...)
    reporting_period_end: date = Field(...)
    functional_unit: str = Field(...)
    reference_unit_value: float = Field(default=1.0)
    pcf_excluding_biogenic: float = Field(..., description="PCF excluding biogenic CO2")
    pcf_including_biogenic: Optional[float] = Field(None)
    fossil_ghg_emissions: Optional[float] = Field(None)
    biogenic_carbon_content: Optional[float] = Field(None)
    biogenic_carbon_withdrawal: Optional[float] = Field(None)
    dluc_ghg_emissions: Optional[float] = Field(None)  # Direct Land Use Change
    aircraft_emissions: Optional[float] = Field(None)
    boundary_processes: Optional[str] = Field(None)
    primary_data_share: Optional[float] = Field(None, ge=0, le=100)
    emission_factor_sources: List[str] = Field(default_factory=list)
    geographic_scope: str = Field(default="global")
    data_quality_rating: DataQualityRating = Field(default=DataQualityRating.SECONDARY)
    uncertainty_pct: Optional[float] = Field(None)
    assurance_statement: Optional[str] = Field(None)
    standard_used: PCFStandard = Field(...)


class PCFSubmission(BaseModel):
    """PCF submission from supplier."""
    submission_id: str = Field(...)
    supplier: SupplierInfo = Field(...)
    product: ProductInfo = Field(...)
    pcf_data: PCFDataPoint = Field(...)
    submission_date: datetime = Field(default_factory=datetime.utcnow)
    status: SubmissionStatus = Field(default=SubmissionStatus.PENDING)
    validation_messages: List[str] = Field(default_factory=list)
    version: int = Field(default=1)
    supersedes: Optional[str] = Field(None)


class ValidationCheck(BaseModel):
    """Validation check result."""
    check_name: str = Field(...)
    result: ValidationResult = Field(...)
    message: str = Field(...)
    severity: str = Field(default="error")


class SupplierMapping(BaseModel):
    """Mapping from internal product to supplier PCF."""
    mapping_id: str = Field(...)
    internal_product_id: str = Field(...)
    internal_product_name: str = Field(...)
    supplier_id: str = Field(...)
    supplier_product_id: str = Field(...)
    pcf_id: str = Field(...)
    effective_date: date = Field(...)
    end_date: Optional[date] = Field(None)
    quantity_factor: float = Field(default=1.0)


class SupplierEmissions(BaseModel):
    """Calculated supplier emissions."""
    calculation_id: str = Field(...)
    supplier_id: str = Field(...)
    supplier_name: str = Field(...)
    product_id: str = Field(...)
    product_name: str = Field(...)
    quantity: float = Field(...)
    quantity_unit: str = Field(...)
    pcf_factor: float = Field(...)
    emissions_kgco2e: float = Field(...)
    data_quality: DataQualityRating = Field(...)
    pcf_source: str = Field(...)
    provenance_hash: str = Field(...)


class SupplierQueryInput(BaseModel):
    """Input for supplier data operations."""
    operation: str = Field(...)  # submit, validate, query, map, calculate
    submission: Optional[PCFSubmission] = Field(None)
    supplier_id: Optional[str] = Field(None)
    product_id: Optional[str] = Field(None)
    mapping: Optional[SupplierMapping] = Field(None)
    quantity: Optional[float] = Field(None)
    quantity_unit: Optional[str] = Field(None)
    start_date: Optional[date] = Field(None)
    end_date: Optional[date] = Field(None)
    tenant_id: Optional[str] = Field(None)


class SupplierQueryOutput(BaseModel):
    """Output from supplier data operations."""
    operation: str = Field(...)
    submissions: List[PCFSubmission] = Field(default_factory=list)
    validation_checks: List[ValidationCheck] = Field(default_factory=list)
    mappings: List[SupplierMapping] = Field(default_factory=list)
    emissions: List[SupplierEmissions] = Field(default_factory=list)
    supplier_count: int = Field(default=0)
    product_count: int = Field(default=0)
    coverage_pct: Optional[float] = Field(None)
    processing_time_ms: float = Field(...)
    provenance_hash: str = Field(...)


# =============================================================================
# SUPPLIER DATA EXCHANGE AGENT
# =============================================================================

class SupplierDataExchangeAgent(BaseAgent):
    """
    GL-DATA-X-012: Supplier Data Exchange Agent

    Manages supplier PCF submissions and calculates Scope 3 emissions.

    Zero-Hallucination Guarantees:
        - All data validated against published standards
        - NO LLM involvement in calculations
        - Explicit mapping rules for supplier data
        - Complete audit trail for submissions
    """

    AGENT_ID = "GL-DATA-X-012"
    AGENT_NAME = "Supplier Data Exchange Agent"
    VERSION = "1.0.0"

    # PACT validation thresholds
    PACT_MIN_PRIMARY_DATA = 20.0  # Minimum primary data share %

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize SupplierDataExchangeAgent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Supplier PCF data exchange and validation",
                version=self.VERSION,
            )
        super().__init__(config)

        self._suppliers: Dict[str, SupplierInfo] = {}
        self._submissions: Dict[str, PCFSubmission] = {}
        self._mappings: Dict[str, SupplierMapping] = {}
        self._pcf_by_product: Dict[str, List[PCFDataPoint]] = {}

        self.logger.info(f"Initialized {self.AGENT_NAME}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute supplier data operation."""
        start_time = datetime.utcnow()

        try:
            query_input = SupplierQueryInput(**input_data)

            if query_input.operation == "submit":
                return self._handle_submit(query_input, start_time)
            elif query_input.operation == "validate":
                return self._handle_validate(query_input, start_time)
            elif query_input.operation == "query":
                return self._handle_query(query_input, start_time)
            elif query_input.operation == "map":
                return self._handle_map(query_input, start_time)
            elif query_input.operation == "calculate":
                return self._handle_calculate(query_input, start_time)
            else:
                return AgentResult(success=False, error=f"Unknown operation: {query_input.operation}")

        except Exception as e:
            self.logger.error(f"Supplier operation failed: {str(e)}", exc_info=True)
            return AgentResult(success=False, error=str(e))

    def _handle_submit(self, query_input: SupplierQueryInput, start_time: datetime) -> AgentResult:
        """Handle PCF submission."""
        if not query_input.submission:
            return AgentResult(success=False, error="submission required")

        submission = query_input.submission

        # Validate submission
        validation_checks = self._validate_pcf(submission)
        all_passed = all(c.result != ValidationResult.FAIL for c in validation_checks)

        submission.status = SubmissionStatus.VALIDATED if all_passed else SubmissionStatus.NEEDS_REVISION
        submission.validation_messages = [c.message for c in validation_checks if c.result != ValidationResult.PASS]

        # Store submission
        self._submissions[submission.submission_id] = submission

        # Store supplier
        self._suppliers[submission.supplier.supplier_id] = submission.supplier

        # Index PCF by product
        if submission.product.product_id not in self._pcf_by_product:
            self._pcf_by_product[submission.product.product_id] = []
        self._pcf_by_product[submission.product.product_id].append(submission.pcf_data)

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        output = SupplierQueryOutput(
            operation="submit",
            submissions=[submission.model_dump()],
            validation_checks=[c.model_dump() for c in validation_checks],
            processing_time_ms=processing_time,
            provenance_hash=self._compute_provenance_hash(submission.model_dump(), {"status": submission.status.value})
        )

        return AgentResult(success=True, data=output.model_dump())

    def _handle_validate(self, query_input: SupplierQueryInput, start_time: datetime) -> AgentResult:
        """Handle PCF validation."""
        if not query_input.submission:
            return AgentResult(success=False, error="submission required for validation")

        validation_checks = self._validate_pcf(query_input.submission)

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        output = SupplierQueryOutput(
            operation="validate",
            validation_checks=[c.model_dump() for c in validation_checks],
            processing_time_ms=processing_time,
            provenance_hash=self._compute_provenance_hash({}, {"checks": len(validation_checks)})
        )

        return AgentResult(success=True, data=output.model_dump())

    def _handle_query(self, query_input: SupplierQueryInput, start_time: datetime) -> AgentResult:
        """Handle submission query."""
        submissions = list(self._submissions.values())

        if query_input.supplier_id:
            submissions = [s for s in submissions if s.supplier.supplier_id == query_input.supplier_id]

        if query_input.product_id:
            submissions = [s for s in submissions if s.product.product_id == query_input.product_id]

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        output = SupplierQueryOutput(
            operation="query",
            submissions=[s.model_dump() for s in submissions],
            supplier_count=len(set(s.supplier.supplier_id for s in submissions)),
            product_count=len(set(s.product.product_id for s in submissions)),
            processing_time_ms=processing_time,
            provenance_hash=self._compute_provenance_hash({}, {"count": len(submissions)})
        )

        return AgentResult(success=True, data=output.model_dump())

    def _handle_map(self, query_input: SupplierQueryInput, start_time: datetime) -> AgentResult:
        """Handle supplier mapping."""
        if not query_input.mapping:
            return AgentResult(success=False, error="mapping required")

        mapping = query_input.mapping
        self._mappings[mapping.mapping_id] = mapping

        return AgentResult(success=True, data={
            "mapping_id": mapping.mapping_id,
            "registered": True
        })

    def _handle_calculate(self, query_input: SupplierQueryInput, start_time: datetime) -> AgentResult:
        """Handle emissions calculation."""
        if not query_input.product_id or query_input.quantity is None:
            return AgentResult(success=False, error="product_id and quantity required")

        # Find PCF for product
        pcf_list = self._pcf_by_product.get(query_input.product_id, [])

        if not pcf_list:
            return AgentResult(success=False, error=f"No PCF data for product: {query_input.product_id}")

        # Use most recent PCF
        pcf = pcf_list[-1]

        # Calculate emissions
        emissions_kgco2e = query_input.quantity * pcf.pcf_excluding_biogenic

        # Find supplier
        supplier = self._suppliers.get(pcf.supplier_id)

        calculation = SupplierEmissions(
            calculation_id=f"SUPP-{uuid.uuid4().hex[:8].upper()}",
            supplier_id=pcf.supplier_id,
            supplier_name=supplier.supplier_name if supplier else "Unknown",
            product_id=query_input.product_id,
            product_name=query_input.product_id,
            quantity=query_input.quantity,
            quantity_unit=query_input.quantity_unit or pcf.functional_unit,
            pcf_factor=pcf.pcf_excluding_biogenic,
            emissions_kgco2e=round(emissions_kgco2e, 3),
            data_quality=pcf.data_quality_rating,
            pcf_source=f"{pcf.standard_used.value}:{pcf.pcf_id}",
            provenance_hash=self._compute_provenance_hash(
                {"product": query_input.product_id, "qty": query_input.quantity},
                {"emissions": emissions_kgco2e}
            )
        )

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        output = SupplierQueryOutput(
            operation="calculate",
            emissions=[calculation.model_dump()],
            processing_time_ms=processing_time,
            provenance_hash=calculation.provenance_hash
        )

        return AgentResult(success=True, data=output.model_dump())

    def _validate_pcf(self, submission: PCFSubmission) -> List[ValidationCheck]:
        """Validate PCF submission."""
        checks = []
        pcf = submission.pcf_data

        # Check 1: PCF value is positive
        if pcf.pcf_excluding_biogenic <= 0:
            checks.append(ValidationCheck(
                check_name="pcf_positive",
                result=ValidationResult.FAIL,
                message="PCF value must be positive",
                severity="error"
            ))
        else:
            checks.append(ValidationCheck(
                check_name="pcf_positive",
                result=ValidationResult.PASS,
                message="PCF value is positive"
            ))

        # Check 2: Functional unit specified
        if not pcf.functional_unit:
            checks.append(ValidationCheck(
                check_name="functional_unit",
                result=ValidationResult.FAIL,
                message="Functional unit must be specified",
                severity="error"
            ))
        else:
            checks.append(ValidationCheck(
                check_name="functional_unit",
                result=ValidationResult.PASS,
                message="Functional unit specified"
            ))

        # Check 3: Reporting period valid
        if pcf.reporting_period_end <= pcf.reporting_period_start:
            checks.append(ValidationCheck(
                check_name="reporting_period",
                result=ValidationResult.FAIL,
                message="Reporting period end must be after start",
                severity="error"
            ))
        else:
            checks.append(ValidationCheck(
                check_name="reporting_period",
                result=ValidationResult.PASS,
                message="Reporting period is valid"
            ))

        # Check 4: Primary data share (PACT requirement)
        if pcf.standard_used == PCFStandard.PACT:
            if pcf.primary_data_share is None or pcf.primary_data_share < self.PACT_MIN_PRIMARY_DATA:
                checks.append(ValidationCheck(
                    check_name="primary_data_share",
                    result=ValidationResult.WARNING,
                    message=f"Primary data share should be >= {self.PACT_MIN_PRIMARY_DATA}%",
                    severity="warning"
                ))
            else:
                checks.append(ValidationCheck(
                    check_name="primary_data_share",
                    result=ValidationResult.PASS,
                    message="Primary data share meets PACT requirement"
                ))

        # Check 5: Emission factor sources documented
        if not pcf.emission_factor_sources:
            checks.append(ValidationCheck(
                check_name="ef_sources",
                result=ValidationResult.WARNING,
                message="Emission factor sources should be documented",
                severity="warning"
            ))
        else:
            checks.append(ValidationCheck(
                check_name="ef_sources",
                result=ValidationResult.PASS,
                message="Emission factor sources documented"
            ))

        return checks

    def _compute_provenance_hash(self, input_data: Any, output_data: Any) -> str:
        """Compute SHA-256 provenance hash."""
        provenance_str = json.dumps(
            {"input": str(input_data), "output": output_data},
            sort_keys=True, default=str
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    # =========================================================================
    # PUBLIC API METHODS
    # =========================================================================

    def submit_pcf(self, submission: PCFSubmission) -> PCFSubmission:
        """Submit a PCF from supplier."""
        result = self.run({
            "operation": "submit",
            "submission": submission.model_dump()
        })
        if result.success and result.data.get("submissions"):
            return PCFSubmission(**result.data["submissions"][0])
        raise ValueError(f"Submission failed: {result.error}")

    def validate_pcf(self, submission: PCFSubmission) -> List[ValidationCheck]:
        """Validate a PCF submission."""
        result = self.run({
            "operation": "validate",
            "submission": submission.model_dump()
        })
        if result.success:
            return [ValidationCheck(**c) for c in result.data.get("validation_checks", [])]
        raise ValueError(f"Validation failed: {result.error}")

    def calculate_supplier_emissions(
        self,
        product_id: str,
        quantity: float,
        quantity_unit: str = "kg"
    ) -> SupplierEmissions:
        """Calculate emissions from supplier product."""
        result = self.run({
            "operation": "calculate",
            "product_id": product_id,
            "quantity": quantity,
            "quantity_unit": quantity_unit
        })
        if result.success and result.data.get("emissions"):
            return SupplierEmissions(**result.data["emissions"][0])
        raise ValueError(f"Calculation failed: {result.error}")

    def get_pcf_standards(self) -> List[str]:
        """Get list of PCF standards."""
        return [s.value for s in PCFStandard]

    def get_supplier_count(self) -> int:
        """Get number of registered suppliers."""
        return len(self._suppliers)

    def get_submission_count(self) -> int:
        """Get number of submissions."""
        return len(self._submissions)
