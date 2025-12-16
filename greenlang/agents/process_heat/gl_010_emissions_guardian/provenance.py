# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian - Provenance Tracking Module

This module implements comprehensive provenance tracking for regulatory audit
compliance in emissions monitoring systems. All calculations, data transformations,
and decisions are tracked with SHA-256 hashes to ensure complete reproducibility
and audit trail integrity.

Provenance tracking is essential for:
    - EPA Part 98 GHG reporting verification
    - Title V permit compliance demonstration
    - Third-party verification audits
    - ISO 14064 GHG verification requirements
    - Legal defensibility of reported emissions

Standards Compliance:
    - EPA 40 CFR Part 98 recordkeeping requirements
    - EPA 40 CFR Part 75 QA/QC documentation
    - ISO 14064-1:2018 GHG verification requirements
    - GHG Protocol verification guidance

Example:
    >>> from greenlang.agents.process_heat.gl_010_emissions_guardian.provenance import (
    ...     ProvenanceTracker,
    ...     ProvenanceRecord,
    ...     DataLineage,
    ... )
    >>> tracker = ProvenanceTracker()
    >>> record = tracker.track_calculation(
    ...     input_data={"fuel_consumption": 1500, "fuel_type": "natural_gas"},
    ...     output_data={"co2_tons": 150.5},
    ...     calculation_type="ghg_emissions",
    ... )

Author: GreenLang Process Heat Team
Version: 2.0.0
"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union
import hashlib
import json
import logging
import uuid
from functools import wraps
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class ProvenanceEventType(str, Enum):
    """Types of provenance events tracked."""
    DATA_INPUT = "data_input"  # Raw data ingestion
    DATA_TRANSFORM = "data_transform"  # Data transformation
    CALCULATION = "calculation"  # Emissions calculation
    VALIDATION = "validation"  # Data validation
    AGGREGATION = "aggregation"  # Data aggregation
    COMPLIANCE_CHECK = "compliance_check"  # Compliance assessment
    REPORT_GENERATION = "report_generation"  # Report creation
    EXPORT = "export"  # Data export
    CORRECTION = "correction"  # Data correction
    APPROVAL = "approval"  # Review/approval action
    SUBMISSION = "submission"  # Regulatory submission


class DataSourceType(str, Enum):
    """Types of data sources."""
    CEMS = "cems"  # Continuous emissions monitoring
    FUEL_METER = "fuel_meter"  # Fuel flow measurement
    LAB_ANALYSIS = "lab_analysis"  # Laboratory fuel analysis
    MANUAL_ENTRY = "manual_entry"  # Manual data entry
    ERP_SYSTEM = "erp_system"  # Enterprise system
    CALCULATED = "calculated"  # Derived value
    DEFAULT = "default"  # Regulatory default value
    EXTERNAL_API = "external_api"  # External data source


class HashAlgorithm(str, Enum):
    """Supported hash algorithms."""
    SHA256 = "sha256"
    SHA384 = "sha384"
    SHA512 = "sha512"


class VerificationStatus(str, Enum):
    """Provenance verification status."""
    VERIFIED = "verified"
    FAILED = "failed"
    PENDING = "pending"
    NOT_VERIFIED = "not_verified"


# =============================================================================
# PROVENANCE DATA MODELS
# =============================================================================


class DataSource(BaseModel):
    """
    Data source metadata for provenance tracking.

    Documents the origin of input data including source type,
    timestamp, and quality indicators.

    Attributes:
        source_id: Unique source identifier
        source_type: Type of data source
        source_name: Human-readable source name
        timestamp: Data timestamp

    Example:
        >>> source = DataSource(
        ...     source_id="CEMS-001",
        ...     source_type=DataSourceType.CEMS,
        ...     source_name="Stack 1 CEMS",
        ...     timestamp=datetime.now(timezone.utc),
        ... )
    """

    source_id: str = Field(
        ...,
        description="Unique source identifier"
    )
    source_type: DataSourceType = Field(
        ...,
        description="Type of data source"
    )
    source_name: str = Field(
        default="",
        description="Human-readable source name"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Data timestamp"
    )

    # Quality metadata
    quality_indicator: Optional[str] = Field(
        default=None,
        description="Data quality indicator"
    )
    uncertainty_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Data uncertainty (%)"
    )

    # Calibration status (for instruments)
    calibration_status: Optional[str] = Field(
        default=None,
        description="Instrument calibration status"
    )
    last_calibration: Optional[datetime] = Field(
        default=None,
        description="Last calibration timestamp"
    )

    # External references
    external_reference: Optional[str] = Field(
        default=None,
        description="External system reference"
    )

    class Config:
        use_enum_values = True


class CalculationStep(BaseModel):
    """
    Individual calculation step for audit trail.

    Documents each step in a calculation with inputs, formula,
    and result for complete reproducibility.

    Attributes:
        step_number: Step sequence number
        description: Step description
        formula: Formula or equation used
        inputs: Input values for this step

    Example:
        >>> step = CalculationStep(
        ...     step_number=1,
        ...     description="Calculate CO2 mass emissions",
        ...     formula="E_CO2 = Fuel * EF_CO2",
        ...     inputs={"Fuel": 1500, "EF_CO2": 53.06},
        ...     result=79590.0,
        ...     result_unit="kg",
        ... )
    """

    step_number: int = Field(
        ...,
        ge=1,
        description="Step sequence number"
    )
    description: str = Field(
        ...,
        description="Step description"
    )
    formula: Optional[str] = Field(
        default=None,
        description="Formula or equation used"
    )
    formula_reference: Optional[str] = Field(
        default=None,
        description="Formula regulatory reference"
    )

    # Inputs
    inputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Input values for this step"
    )
    input_units: Dict[str, str] = Field(
        default_factory=dict,
        description="Units for input values"
    )

    # Result
    result: Any = Field(
        ...,
        description="Step result"
    )
    result_unit: Optional[str] = Field(
        default=None,
        description="Result unit"
    )

    # Method reference
    method_code: Optional[str] = Field(
        default=None,
        description="Calculation method code"
    )

    def to_hash_string(self) -> str:
        """Convert step to string for hashing."""
        return json.dumps({
            "step": self.step_number,
            "formula": self.formula,
            "inputs": self.inputs,
            "result": self.result,
        }, sort_keys=True, default=str)


class ProvenanceRecord(BaseModel):
    """
    Complete provenance record for an operation.

    Captures all information needed to reproduce and verify
    a calculation or data transformation.

    Attributes:
        record_id: Unique record identifier
        event_type: Type of provenance event
        timestamp: Event timestamp
        input_hash: SHA-256 hash of input data

    Example:
        >>> record = ProvenanceRecord(
        ...     record_id="PROV-2024-001",
        ...     event_type=ProvenanceEventType.CALCULATION,
        ...     description="CO2 emissions calculation",
        ...     input_hash="abc123...",
        ...     output_hash="def456...",
        ... )
    """

    record_id: str = Field(
        default_factory=lambda: f"PROV-{uuid.uuid4().hex[:12].upper()}",
        description="Unique record identifier"
    )
    event_type: ProvenanceEventType = Field(
        ...,
        description="Type of provenance event"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp"
    )
    description: str = Field(
        default="",
        description="Event description"
    )

    # Entity identification
    entity_type: str = Field(
        default="",
        description="Type of entity (source, facility, report)"
    )
    entity_id: str = Field(
        default="",
        description="Entity identifier"
    )

    # Input provenance
    input_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Input data snapshot"
    )
    input_hash: str = Field(
        ...,
        description="SHA-256 hash of input data"
    )
    input_sources: List[DataSource] = Field(
        default_factory=list,
        description="Data sources for inputs"
    )

    # Output provenance
    output_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Output data snapshot"
    )
    output_hash: str = Field(
        ...,
        description="SHA-256 hash of output data"
    )

    # Calculation details
    calculation_steps: List[CalculationStep] = Field(
        default_factory=list,
        description="Detailed calculation steps"
    )
    method_reference: Optional[str] = Field(
        default=None,
        description="Methodology reference"
    )
    parameters_used: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters/factors used"
    )

    # Chain linking
    parent_record_ids: List[str] = Field(
        default_factory=list,
        description="Parent provenance record IDs"
    )
    chain_hash: Optional[str] = Field(
        default=None,
        description="Chain hash linking to parents"
    )

    # Configuration
    config_hash: Optional[str] = Field(
        default=None,
        description="Configuration hash"
    )
    software_version: str = Field(
        default="2.0.0",
        description="Software version"
    )
    agent_id: str = Field(
        default="GL-010",
        description="Agent identifier"
    )

    # User/system attribution
    user_id: Optional[str] = Field(
        default=None,
        description="User who triggered event"
    )
    system_id: Optional[str] = Field(
        default=None,
        description="System identifier"
    )

    # Verification
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.NOT_VERIFIED,
        description="Verification status"
    )
    verified_at: Optional[datetime] = Field(
        default=None,
        description="Verification timestamp"
    )
    verified_by: Optional[str] = Field(
        default=None,
        description="Verifier identifier"
    )

    class Config:
        use_enum_values = True

    def calculate_record_hash(self) -> str:
        """Calculate complete record hash."""
        hash_content = {
            "record_id": self.record_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "parent_record_ids": self.parent_record_ids,
        }
        hash_string = json.dumps(hash_content, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()


class DataLineage(BaseModel):
    """
    Complete data lineage for an emission value.

    Traces the full history of data from source to final
    reported value for audit purposes.

    Attributes:
        lineage_id: Unique lineage identifier
        target_field: Final data field name
        final_value: Final calculated value
        provenance_chain: Chain of provenance records
    """

    lineage_id: str = Field(
        default_factory=lambda: f"LIN-{uuid.uuid4().hex[:12].upper()}",
        description="Unique lineage identifier"
    )
    target_field: str = Field(
        ...,
        description="Final data field name"
    )
    final_value: Any = Field(
        ...,
        description="Final calculated value"
    )
    final_unit: Optional[str] = Field(
        default=None,
        description="Final value unit"
    )

    # Provenance chain
    provenance_chain: List[ProvenanceRecord] = Field(
        default_factory=list,
        description="Chain of provenance records"
    )
    chain_length: int = Field(
        default=0,
        ge=0,
        description="Number of steps in chain"
    )
    chain_integrity_hash: Optional[str] = Field(
        default=None,
        description="Hash of complete chain"
    )

    # Source data
    original_sources: List[DataSource] = Field(
        default_factory=list,
        description="Original data sources"
    )

    # Timeline
    first_timestamp: Optional[datetime] = Field(
        default=None,
        description="First event timestamp"
    )
    last_timestamp: Optional[datetime] = Field(
        default=None,
        description="Last event timestamp"
    )

    # Verification
    fully_verified: bool = Field(
        default=False,
        description="All records verified"
    )

    def calculate_chain_hash(self) -> str:
        """Calculate hash of complete provenance chain."""
        chain_data = [r.calculate_record_hash() for r in self.provenance_chain]
        chain_string = "|".join(chain_data)
        return hashlib.sha256(chain_string.encode()).hexdigest()


# =============================================================================
# PROVENANCE TRACKER
# =============================================================================


class ProvenanceTracker:
    """
    Comprehensive provenance tracking for emissions calculations.

    Implements SHA-256 hashing for complete audit trails as required
    by EPA regulations and GHG verification standards.

    Features:
        - SHA-256 hashing of all inputs and outputs
        - Chain linking for data lineage
        - Calculation step documentation
        - Verification support
        - Export for audit packages

    Attributes:
        algorithm: Hash algorithm (default: SHA-256)
        records: List of provenance records
        config_hash: Configuration hash

    Example:
        >>> tracker = ProvenanceTracker()
        >>> record = tracker.track_calculation(
        ...     input_data={"fuel": 1500},
        ...     output_data={"emissions": 150.5},
        ...     calculation_type="ghg_emissions",
        ... )
        >>> verified = tracker.verify_record(record.record_id)
    """

    def __init__(
        self,
        algorithm: HashAlgorithm = HashAlgorithm.SHA256,
        config: Optional[Dict[str, Any]] = None,
        retain_data: bool = True,
    ) -> None:
        """
        Initialize provenance tracker.

        Args:
            algorithm: Hash algorithm to use
            config: Configuration dictionary to hash
            retain_data: Whether to retain full data in records
        """
        self.algorithm = algorithm
        self.retain_data = retain_data
        self.records: Dict[str, ProvenanceRecord] = {}
        self.lineages: Dict[str, DataLineage] = {}

        # Hash configuration
        self.config_hash = self._hash_data(config) if config else None

        logger.info(
            f"ProvenanceTracker initialized with {algorithm.value} algorithm"
        )

    def _get_hasher(self) -> Any:
        """Get hash function based on configured algorithm."""
        if self.algorithm == HashAlgorithm.SHA256:
            return hashlib.sha256()
        elif self.algorithm == HashAlgorithm.SHA384:
            return hashlib.sha384()
        elif self.algorithm == HashAlgorithm.SHA512:
            return hashlib.sha512()
        else:
            return hashlib.sha256()

    def _hash_data(self, data: Any) -> str:
        """
        Calculate hash of data.

        Args:
            data: Data to hash (dict, list, or primitive)

        Returns:
            Hex digest of hash
        """
        if data is None:
            return hashlib.sha256(b"null").hexdigest()

        try:
            # Convert to JSON string for consistent hashing
            data_str = json.dumps(data, sort_keys=True, default=str)
            hasher = self._get_hasher()
            hasher.update(data_str.encode())
            return hasher.hexdigest()
        except (TypeError, ValueError) as e:
            logger.warning(f"Hash calculation failed: {e}")
            # Fall back to string representation
            hasher = self._get_hasher()
            hasher.update(str(data).encode())
            return hasher.hexdigest()

    def track_input(
        self,
        data: Dict[str, Any],
        source: DataSource,
        entity_type: str = "",
        entity_id: str = "",
        description: str = "",
    ) -> ProvenanceRecord:
        """
        Track data input event.

        Args:
            data: Input data dictionary
            source: Data source metadata
            entity_type: Type of entity
            entity_id: Entity identifier
            description: Event description

        Returns:
            ProvenanceRecord for this input

        Example:
            >>> source = DataSource(
            ...     source_id="CEMS-001",
            ...     source_type=DataSourceType.CEMS,
            ... )
            >>> record = tracker.track_input(
            ...     data={"nox_ppm": 125.5},
            ...     source=source,
            ...     entity_type="stack",
            ...     entity_id="STACK-001",
            ... )
        """
        input_hash = self._hash_data(data)

        record = ProvenanceRecord(
            event_type=ProvenanceEventType.DATA_INPUT,
            description=description or f"Data input from {source.source_type}",
            entity_type=entity_type,
            entity_id=entity_id,
            input_data=data if self.retain_data else None,
            input_hash=input_hash,
            output_data=data if self.retain_data else None,
            output_hash=input_hash,
            input_sources=[source],
            config_hash=self.config_hash,
        )

        self.records[record.record_id] = record
        logger.debug(f"Tracked input: {record.record_id}")

        return record

    def track_calculation(
        self,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        calculation_type: str,
        steps: Optional[List[CalculationStep]] = None,
        method_reference: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        parent_records: Optional[List[str]] = None,
        entity_type: str = "",
        entity_id: str = "",
        description: str = "",
    ) -> ProvenanceRecord:
        """
        Track calculation event with full audit trail.

        Args:
            input_data: Input data for calculation
            output_data: Calculation results
            calculation_type: Type of calculation
            steps: Detailed calculation steps
            method_reference: Regulatory method reference
            parameters: Parameters/factors used
            parent_records: Parent provenance record IDs
            entity_type: Type of entity
            entity_id: Entity identifier
            description: Event description

        Returns:
            ProvenanceRecord for this calculation

        Example:
            >>> record = tracker.track_calculation(
            ...     input_data={"fuel_mmbtu": 1500, "ef_kg_mmbtu": 53.06},
            ...     output_data={"co2_kg": 79590.0},
            ...     calculation_type="ghg_emissions",
            ...     method_reference="40 CFR 98.33(a)(3)",
            ...     steps=[
            ...         CalculationStep(
            ...             step_number=1,
            ...             description="Calculate CO2 mass",
            ...             formula="CO2 = Fuel * EF",
            ...             inputs={"Fuel": 1500, "EF": 53.06},
            ...             result=79590.0,
            ...         ),
            ...     ],
            ... )
        """
        input_hash = self._hash_data(input_data)
        output_hash = self._hash_data(output_data)

        # Calculate chain hash if parents exist
        chain_hash = None
        if parent_records:
            parent_hashes = [
                self.records[pid].output_hash
                for pid in parent_records
                if pid in self.records
            ]
            if parent_hashes:
                chain_data = input_hash + "|" + "|".join(parent_hashes)
                chain_hash = self._hash_data(chain_data)

        record = ProvenanceRecord(
            event_type=ProvenanceEventType.CALCULATION,
            description=description or f"Calculation: {calculation_type}",
            entity_type=entity_type,
            entity_id=entity_id,
            input_data=input_data if self.retain_data else None,
            input_hash=input_hash,
            output_data=output_data if self.retain_data else None,
            output_hash=output_hash,
            calculation_steps=steps or [],
            method_reference=method_reference,
            parameters_used=parameters or {},
            parent_record_ids=parent_records or [],
            chain_hash=chain_hash,
            config_hash=self.config_hash,
        )

        self.records[record.record_id] = record
        logger.debug(f"Tracked calculation: {record.record_id}")

        return record

    def track_validation(
        self,
        input_data: Dict[str, Any],
        validation_result: Dict[str, Any],
        rules_applied: List[str],
        parent_records: Optional[List[str]] = None,
        entity_type: str = "",
        entity_id: str = "",
        description: str = "",
    ) -> ProvenanceRecord:
        """
        Track data validation event.

        Args:
            input_data: Data that was validated
            validation_result: Validation results
            rules_applied: List of validation rules applied
            parent_records: Parent provenance record IDs
            entity_type: Type of entity
            entity_id: Entity identifier
            description: Event description

        Returns:
            ProvenanceRecord for this validation
        """
        input_hash = self._hash_data(input_data)
        output_hash = self._hash_data(validation_result)

        record = ProvenanceRecord(
            event_type=ProvenanceEventType.VALIDATION,
            description=description or "Data validation",
            entity_type=entity_type,
            entity_id=entity_id,
            input_data=input_data if self.retain_data else None,
            input_hash=input_hash,
            output_data=validation_result if self.retain_data else None,
            output_hash=output_hash,
            parameters_used={"rules_applied": rules_applied},
            parent_record_ids=parent_records or [],
            config_hash=self.config_hash,
        )

        self.records[record.record_id] = record
        logger.debug(f"Tracked validation: {record.record_id}")

        return record

    def track_aggregation(
        self,
        source_records: List[str],
        aggregated_data: Dict[str, Any],
        aggregation_method: str,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
        entity_type: str = "",
        entity_id: str = "",
        description: str = "",
    ) -> ProvenanceRecord:
        """
        Track data aggregation event.

        Args:
            source_records: Source provenance record IDs
            aggregated_data: Aggregated results
            aggregation_method: Method used for aggregation
            period_start: Aggregation period start
            period_end: Aggregation period end
            entity_type: Type of entity
            entity_id: Entity identifier
            description: Event description

        Returns:
            ProvenanceRecord for this aggregation
        """
        # Collect input data from source records
        input_summary = {
            "source_records": source_records,
            "source_count": len(source_records),
            "period_start": period_start.isoformat() if period_start else None,
            "period_end": period_end.isoformat() if period_end else None,
        }
        input_hash = self._hash_data(input_summary)
        output_hash = self._hash_data(aggregated_data)

        record = ProvenanceRecord(
            event_type=ProvenanceEventType.AGGREGATION,
            description=description or f"Aggregation: {aggregation_method}",
            entity_type=entity_type,
            entity_id=entity_id,
            input_data=input_summary if self.retain_data else None,
            input_hash=input_hash,
            output_data=aggregated_data if self.retain_data else None,
            output_hash=output_hash,
            parameters_used={
                "aggregation_method": aggregation_method,
                "period_start": period_start.isoformat() if period_start else None,
                "period_end": period_end.isoformat() if period_end else None,
            },
            parent_record_ids=source_records,
            config_hash=self.config_hash,
        )

        self.records[record.record_id] = record
        logger.debug(f"Tracked aggregation: {record.record_id}")

        return record

    def track_compliance_check(
        self,
        emission_data: Dict[str, Any],
        compliance_result: Dict[str, Any],
        limits_checked: Dict[str, float],
        parent_records: Optional[List[str]] = None,
        entity_type: str = "",
        entity_id: str = "",
        description: str = "",
    ) -> ProvenanceRecord:
        """
        Track compliance check event.

        Args:
            emission_data: Emission data checked
            compliance_result: Compliance check results
            limits_checked: Permit limits checked against
            parent_records: Parent provenance record IDs
            entity_type: Type of entity
            entity_id: Entity identifier
            description: Event description

        Returns:
            ProvenanceRecord for this compliance check
        """
        input_hash = self._hash_data(emission_data)
        output_hash = self._hash_data(compliance_result)

        record = ProvenanceRecord(
            event_type=ProvenanceEventType.COMPLIANCE_CHECK,
            description=description or "Compliance assessment",
            entity_type=entity_type,
            entity_id=entity_id,
            input_data=emission_data if self.retain_data else None,
            input_hash=input_hash,
            output_data=compliance_result if self.retain_data else None,
            output_hash=output_hash,
            parameters_used={"limits_checked": limits_checked},
            parent_record_ids=parent_records or [],
            config_hash=self.config_hash,
        )

        self.records[record.record_id] = record
        logger.debug(f"Tracked compliance check: {record.record_id}")

        return record

    def track_report_generation(
        self,
        source_records: List[str],
        report_data: Dict[str, Any],
        report_type: str,
        reporting_period: Optional[str] = None,
        entity_type: str = "",
        entity_id: str = "",
        description: str = "",
    ) -> ProvenanceRecord:
        """
        Track report generation event.

        Args:
            source_records: Source provenance record IDs
            report_data: Report data/summary
            report_type: Type of report (Part98, TitleV, etc.)
            reporting_period: Reporting period identifier
            entity_type: Type of entity
            entity_id: Entity identifier
            description: Event description

        Returns:
            ProvenanceRecord for this report
        """
        input_summary = {
            "source_records": source_records,
            "report_type": report_type,
            "reporting_period": reporting_period,
        }
        input_hash = self._hash_data(input_summary)
        output_hash = self._hash_data(report_data)

        record = ProvenanceRecord(
            event_type=ProvenanceEventType.REPORT_GENERATION,
            description=description or f"Report generation: {report_type}",
            entity_type=entity_type,
            entity_id=entity_id,
            input_data=input_summary if self.retain_data else None,
            input_hash=input_hash,
            output_data=report_data if self.retain_data else None,
            output_hash=output_hash,
            parameters_used={
                "report_type": report_type,
                "reporting_period": reporting_period,
            },
            parent_record_ids=source_records,
            config_hash=self.config_hash,
        )

        self.records[record.record_id] = record
        logger.debug(f"Tracked report generation: {record.record_id}")

        return record

    def verify_record(self, record_id: str) -> Tuple[bool, str]:
        """
        Verify integrity of a provenance record.

        Args:
            record_id: Record ID to verify

        Returns:
            Tuple of (verification_passed, message)

        Example:
            >>> passed, message = tracker.verify_record("PROV-ABC123")
            >>> print(f"Verified: {passed}, {message}")
        """
        if record_id not in self.records:
            return False, f"Record not found: {record_id}"

        record = self.records[record_id]

        # Verify input hash
        if record.input_data is not None:
            calculated_input_hash = self._hash_data(record.input_data)
            if calculated_input_hash != record.input_hash:
                record.verification_status = VerificationStatus.FAILED
                return False, "Input hash mismatch"

        # Verify output hash
        if record.output_data is not None:
            calculated_output_hash = self._hash_data(record.output_data)
            if calculated_output_hash != record.output_hash:
                record.verification_status = VerificationStatus.FAILED
                return False, "Output hash mismatch"

        # Verify parent chain
        for parent_id in record.parent_record_ids:
            if parent_id not in self.records:
                return False, f"Parent record not found: {parent_id}"

        record.verification_status = VerificationStatus.VERIFIED
        record.verified_at = datetime.now(timezone.utc)

        logger.info(f"Record verified: {record_id}")
        return True, "Verification passed"

    def verify_chain(self, record_id: str) -> Tuple[bool, List[str]]:
        """
        Verify complete provenance chain for a record.

        Args:
            record_id: Record ID to verify chain from

        Returns:
            Tuple of (all_verified, list of verification messages)
        """
        messages = []
        all_verified = True

        if record_id not in self.records:
            return False, [f"Record not found: {record_id}"]

        # Collect all records in chain
        to_verify = [record_id]
        verified = set()

        while to_verify:
            current_id = to_verify.pop(0)
            if current_id in verified:
                continue

            verified.add(current_id)
            passed, message = self.verify_record(current_id)
            messages.append(f"{current_id}: {message}")

            if not passed:
                all_verified = False

            # Add parents to verification queue
            record = self.records[current_id]
            for parent_id in record.parent_record_ids:
                if parent_id not in verified:
                    to_verify.append(parent_id)

        return all_verified, messages

    def build_lineage(
        self,
        target_record_id: str,
        target_field: str,
        final_value: Any,
        final_unit: Optional[str] = None,
    ) -> DataLineage:
        """
        Build complete data lineage from a target record.

        Args:
            target_record_id: Target provenance record ID
            target_field: Name of target data field
            final_value: Final calculated value
            final_unit: Unit of final value

        Returns:
            DataLineage object with complete chain

        Example:
            >>> lineage = tracker.build_lineage(
            ...     target_record_id="PROV-XYZ789",
            ...     target_field="annual_co2_tons",
            ...     final_value=150.5,
            ...     final_unit="tons",
            ... )
        """
        if target_record_id not in self.records:
            raise ValueError(f"Record not found: {target_record_id}")

        # Collect chain
        chain = []
        sources = []
        to_process = [target_record_id]
        processed = set()

        while to_process:
            current_id = to_process.pop(0)
            if current_id in processed:
                continue

            processed.add(current_id)
            record = self.records[current_id]
            chain.append(record)

            # Collect sources
            sources.extend(record.input_sources)

            # Add parents
            for parent_id in record.parent_record_ids:
                if parent_id not in processed and parent_id in self.records:
                    to_process.append(parent_id)

        # Sort chain by timestamp
        chain.sort(key=lambda r: r.timestamp)

        lineage = DataLineage(
            target_field=target_field,
            final_value=final_value,
            final_unit=final_unit,
            provenance_chain=chain,
            chain_length=len(chain),
            original_sources=sources,
            first_timestamp=chain[0].timestamp if chain else None,
            last_timestamp=chain[-1].timestamp if chain else None,
        )

        # Calculate chain integrity hash
        lineage.chain_integrity_hash = lineage.calculate_chain_hash()

        # Store lineage
        self.lineages[lineage.lineage_id] = lineage

        logger.info(f"Built lineage: {lineage.lineage_id} with {len(chain)} records")

        return lineage

    def get_record(self, record_id: str) -> Optional[ProvenanceRecord]:
        """Get a provenance record by ID."""
        return self.records.get(record_id)

    def get_records_by_entity(
        self,
        entity_type: str,
        entity_id: str,
    ) -> List[ProvenanceRecord]:
        """Get all records for an entity."""
        return [
            r for r in self.records.values()
            if r.entity_type == entity_type and r.entity_id == entity_id
        ]

    def get_records_by_type(
        self,
        event_type: ProvenanceEventType,
    ) -> List[ProvenanceRecord]:
        """Get all records of a specific type."""
        return [
            r for r in self.records.values()
            if r.event_type == event_type
        ]

    def export_audit_package(
        self,
        record_ids: Optional[List[str]] = None,
        include_data: bool = True,
    ) -> Dict[str, Any]:
        """
        Export provenance records for audit package.

        Args:
            record_ids: Specific records to export (None = all)
            include_data: Include full data in export

        Returns:
            Dictionary with audit package contents
        """
        records_to_export = (
            [self.records[rid] for rid in record_ids if rid in self.records]
            if record_ids
            else list(self.records.values())
        )

        package = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "algorithm": self.algorithm.value,
            "config_hash": self.config_hash,
            "record_count": len(records_to_export),
            "records": [],
        }

        for record in records_to_export:
            record_data = record.dict()
            if not include_data:
                record_data.pop("input_data", None)
                record_data.pop("output_data", None)
            package["records"].append(record_data)

        # Calculate package hash
        package["package_hash"] = self._hash_data(package["records"])

        logger.info(f"Exported audit package with {len(records_to_export)} records")

        return package

    def clear_records(self, older_than_days: Optional[int] = None) -> int:
        """
        Clear provenance records.

        Args:
            older_than_days: Only clear records older than this (None = all)

        Returns:
            Number of records cleared
        """
        if older_than_days is None:
            count = len(self.records)
            self.records.clear()
            self.lineages.clear()
            logger.info(f"Cleared all {count} provenance records")
            return count

        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        to_remove = [
            rid for rid, record in self.records.items()
            if record.timestamp < cutoff
        ]

        for rid in to_remove:
            del self.records[rid]

        logger.info(f"Cleared {len(to_remove)} provenance records older than {older_than_days} days")
        return len(to_remove)


# =============================================================================
# DECORATOR FOR AUTOMATIC PROVENANCE TRACKING
# =============================================================================


T = TypeVar('T')


def track_provenance(
    event_type: ProvenanceEventType = ProvenanceEventType.CALCULATION,
    description: str = "",
    method_reference: Optional[str] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to automatically track provenance for a function.

    Args:
        event_type: Type of provenance event
        description: Event description
        method_reference: Regulatory method reference

    Returns:
        Decorated function with provenance tracking

    Example:
        >>> @track_provenance(
        ...     event_type=ProvenanceEventType.CALCULATION,
        ...     description="CO2 emissions calculation",
        ...     method_reference="40 CFR 98.33(a)(3)",
        ... )
        ... def calculate_co2(fuel_mmbtu: float, ef: float) -> float:
        ...     return fuel_mmbtu * ef
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Get tracker from kwargs or create new one
            tracker = kwargs.pop('_provenance_tracker', None)
            if tracker is None:
                # No tracker provided, execute without tracking
                return func(*args, **kwargs)

            # Capture input data
            input_data = {
                "args": [str(a) for a in args],
                "kwargs": {k: str(v) for k, v in kwargs.items()},
            }

            # Execute function
            result = func(*args, **kwargs)

            # Capture output data
            output_data = {"result": result}

            # Track provenance
            tracker.track_calculation(
                input_data=input_data,
                output_data=output_data,
                calculation_type=func.__name__,
                description=description or f"Function: {func.__name__}",
                method_reference=method_reference,
            )

            return result

        return wrapper
    return decorator


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def calculate_hash(data: Any, algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> str:
    """
    Calculate hash of data using specified algorithm.

    Args:
        data: Data to hash
        algorithm: Hash algorithm

    Returns:
        Hex digest of hash
    """
    data_str = json.dumps(data, sort_keys=True, default=str)

    if algorithm == HashAlgorithm.SHA256:
        return hashlib.sha256(data_str.encode()).hexdigest()
    elif algorithm == HashAlgorithm.SHA384:
        return hashlib.sha384(data_str.encode()).hexdigest()
    elif algorithm == HashAlgorithm.SHA512:
        return hashlib.sha512(data_str.encode()).hexdigest()
    else:
        return hashlib.sha256(data_str.encode()).hexdigest()


def verify_hash(
    data: Any,
    expected_hash: str,
    algorithm: HashAlgorithm = HashAlgorithm.SHA256,
) -> bool:
    """
    Verify data against expected hash.

    Args:
        data: Data to verify
        expected_hash: Expected hash value
        algorithm: Hash algorithm

    Returns:
        True if hash matches
    """
    calculated = calculate_hash(data, algorithm)
    return calculated == expected_hash


def create_provenance_chain_hash(
    hashes: List[str],
    algorithm: HashAlgorithm = HashAlgorithm.SHA256,
) -> str:
    """
    Create chain hash from list of hashes.

    Args:
        hashes: List of hash values
        algorithm: Hash algorithm

    Returns:
        Chain hash
    """
    chain_str = "|".join(hashes)
    return calculate_hash(chain_str, algorithm)


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "ProvenanceEventType",
    "DataSourceType",
    "HashAlgorithm",
    "VerificationStatus",
    # Models
    "DataSource",
    "CalculationStep",
    "ProvenanceRecord",
    "DataLineage",
    # Tracker
    "ProvenanceTracker",
    # Decorator
    "track_provenance",
    # Utilities
    "calculate_hash",
    "verify_hash",
    "create_provenance_chain_hash",
]
