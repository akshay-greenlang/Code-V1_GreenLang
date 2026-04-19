# -*- coding: utf-8 -*-
"""
Retirement Workflow
========================

3-phase workflow for carbon credit retirement on registries within PACK-024
Carbon Neutral Pack.  Validates credits before retirement, executes the
retirement on the appropriate registry, and confirms retirement with
serial number tracking and certificate generation.

Phases:
    1. PreRetirementValidation  -- Validate credit eligibility and ownership
    2. RegistryExecution        -- Execute retirement on registry platform
    3. Confirmation             -- Confirm retirement, capture certificates

Regulatory references:
    - PAS 2060:2014 (Section 9: Retirement requirements)
    - Verra VCS Registry Requirements V4.5
    - Gold Standard Registry Requirements
    - ICVCM Assessment Framework (Tracking)
    - ISO 14064-1:2018 (Cancellation requirements)

Author: GreenLang Team
Version: 24.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "24.0.0"

def _new_uuid() -> str:
    return uuid.uuid4().hex

def _compute_hash(data: Any) -> str:
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(str(data).encode("utf-8")).hexdigest()

# =============================================================================
# ENUMS
# =============================================================================

class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"

class RetirementPhase(str, Enum):
    PRE_RETIREMENT_VALIDATION = "pre_retirement_validation"
    REGISTRY_EXECUTION = "registry_execution"
    CONFIRMATION = "confirmation"

class RegistryType(str, Enum):
    VERRA = "verra"
    GOLD_STANDARD = "gold_standard"
    ACR = "acr"
    CAR = "car"
    CDM = "cdm"
    PURO = "puro"

class RetirementStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PROCESSING = "processing"
    RETIRED = "retired"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    REVERSED = "reversed"

class RetirementPurpose(str, Enum):
    CARBON_NEUTRALITY = "carbon_neutrality"
    NET_ZERO = "net_zero"
    VOLUNTARY_CANCELLATION = "voluntary_cancellation"
    COMPLIANCE = "compliance"
    CORPORATE_SOCIAL_RESPONSIBILITY = "csr"

# =============================================================================
# REFERENCE DATA
# =============================================================================

# Registry retirement endpoints (conceptual)
REGISTRY_ENDPOINTS: Dict[str, Dict[str, str]] = {
    "verra": {
        "name": "Verra VCS Registry",
        "retirement_api": "registry.verra.org/api/v1/retire",
        "confirmation_api": "registry.verra.org/api/v1/confirm",
        "certificate_url": "registry.verra.org/certificates",
    },
    "gold_standard": {
        "name": "Gold Standard Impact Registry",
        "retirement_api": "registry.goldstandard.org/api/v2/retire",
        "confirmation_api": "registry.goldstandard.org/api/v2/confirm",
        "certificate_url": "registry.goldstandard.org/certificates",
    },
    "acr": {
        "name": "American Carbon Registry",
        "retirement_api": "acr.verra.org/api/v1/retire",
        "confirmation_api": "acr.verra.org/api/v1/confirm",
        "certificate_url": "acr.verra.org/certificates",
    },
    "car": {
        "name": "Climate Action Reserve",
        "retirement_api": "thereserve2.apx.com/api/retire",
        "confirmation_api": "thereserve2.apx.com/api/confirm",
        "certificate_url": "thereserve2.apx.com/certificates",
    },
    "cdm": {
        "name": "CDM Registry",
        "retirement_api": "cdm.unfccc.int/api/retire",
        "confirmation_api": "cdm.unfccc.int/api/confirm",
        "certificate_url": "cdm.unfccc.int/certificates",
    },
    "puro": {
        "name": "Puro.earth Registry",
        "retirement_api": "app.puro.earth/api/retire",
        "confirmation_api": "app.puro.earth/api/confirm",
        "certificate_url": "app.puro.earth/certificates",
    },
}

# PAS 2060 retirement requirements
PAS2060_RETIREMENT_WINDOW_MONTHS = 12
PAS2060_BENEFICIARY_MUST_BE_NAMED = True
PAS2060_PURPOSE_MUST_STATE_NEUTRALITY = True

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    phase_name: str = Field(...)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class RetirementRecord(BaseModel):
    record_id: str = Field(default="")
    registry: RegistryType = Field(...)
    serial_numbers: List[str] = Field(default_factory=list)
    serial_number_start: str = Field(default="")
    serial_number_end: str = Field(default="")
    volume_tco2e: float = Field(default=0.0, ge=0.0)
    vintage_year: int = Field(default=2024)
    project_id: str = Field(default="")
    project_name: str = Field(default="")
    retirement_date: Optional[datetime] = Field(None)
    retirement_purpose: RetirementPurpose = Field(default=RetirementPurpose.CARBON_NEUTRALITY)
    beneficiary_name: str = Field(default="")
    beneficiary_country: str = Field(default="")
    retirement_reason: str = Field(default="Carbon neutrality per PAS 2060")
    status: RetirementStatus = Field(default=RetirementStatus.PENDING)
    certificate_url: str = Field(default="")
    registry_transaction_id: str = Field(default="")

class RegistryConfirmation(BaseModel):
    confirmation_id: str = Field(default="")
    registry: RegistryType = Field(...)
    transaction_id: str = Field(default="")
    retirement_date: datetime = Field(default_factory=utcnow)
    volume_tco2e: float = Field(default=0.0, ge=0.0)
    certificate_url: str = Field(default="")
    certificate_hash: str = Field(default="")
    is_final: bool = Field(default=False)
    confirmation_timestamp: datetime = Field(default_factory=utcnow)

class RetirementBatch(BaseModel):
    batch_id: str = Field(default="")
    records: List[RetirementRecord] = Field(default_factory=list)
    confirmations: List[RegistryConfirmation] = Field(default_factory=list)
    total_volume_tco2e: float = Field(default=0.0, ge=0.0)
    total_records: int = Field(default=0)
    successful_records: int = Field(default=0)
    failed_records: int = Field(default=0)

class RetirementConfig(BaseModel):
    org_name: str = Field(default="")
    reporting_year: int = Field(default=2025, ge=2015, le=2050)
    beneficiary_name: str = Field(default="")
    beneficiary_country: str = Field(default="")
    retirement_purpose: RetirementPurpose = Field(default=RetirementPurpose.CARBON_NEUTRALITY)
    credits: List[Dict[str, Any]] = Field(default_factory=list)
    pas2060_compliance: bool = Field(default=True)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

class RetirementResult(BaseModel):
    workflow_id: str = Field(...)
    workflow_name: str = Field(default="retirement")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    batch: Optional[RetirementBatch] = Field(None)
    total_retired_tco2e: float = Field(default=0.0)
    total_certificates: int = Field(default=0)
    registries_used: List[str] = Field(default_factory=list)
    pas2060_compliant: bool = Field(default=False)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class RetirementWorkflow:
    """
    3-phase retirement workflow for PACK-024.

    Validates credit eligibility, executes retirement on carbon credit
    registries, and confirms retirement with certificate tracking.
    Ensures PAS 2060 compliance for retirement timing, beneficiary
    naming, and purpose documentation.

    Attributes:
        workflow_id: Unique execution identifier.
    """

    def __init__(self) -> None:
        self.workflow_id: str = _new_uuid()
        self._phase_results: List[PhaseResult] = []
        self._records: List[RetirementRecord] = []
        self._confirmations: List[RegistryConfirmation] = []
        self._batch: Optional[RetirementBatch] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(self, config: RetirementConfig) -> RetirementResult:
        """Execute the 3-phase retirement workflow."""
        started_at = utcnow()
        self.logger.info(
            "Starting retirement workflow %s, credits=%d",
            self.workflow_id, len(config.credits),
        )
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_pre_retirement_validation(config)
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise ValueError("Pre-retirement validation failed")

            phase2 = await self._phase_registry_execution(config)
            self._phase_results.append(phase2)

            phase3 = await self._phase_confirmation(config)
            self._phase_results.append(phase3)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Retirement workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()
        total_retired = sum(r.volume_tco2e for r in self._records if r.status == RetirementStatus.CONFIRMED)
        registries_used = list(set(r.registry.value for r in self._records))

        self._batch = RetirementBatch(
            batch_id=_new_uuid(),
            records=self._records,
            confirmations=self._confirmations,
            total_volume_tco2e=total_retired,
            total_records=len(self._records),
            successful_records=len([r for r in self._records if r.status == RetirementStatus.CONFIRMED]),
            failed_records=len([r for r in self._records if r.status == RetirementStatus.FAILED]),
        )

        result = RetirementResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            batch=self._batch,
            total_retired_tco2e=round(total_retired, 2),
            total_certificates=len(self._confirmations),
            registries_used=registries_used,
            pas2060_compliant=config.pas2060_compliance,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"})
        )
        return result

    async def _phase_pre_retirement_validation(self, config: RetirementConfig) -> PhaseResult:
        """Validate credit eligibility and ownership before retirement."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        if not config.credits:
            errors.append("No credits provided for retirement")

        # PAS 2060 checks
        if config.pas2060_compliance:
            if not config.beneficiary_name:
                errors.append("PAS 2060 requires beneficiary name for retirement")
            if config.retirement_purpose != RetirementPurpose.CARBON_NEUTRALITY:
                warnings.append(
                    "PAS 2060 retirement should state carbon neutrality as purpose"
                )

        # Validate each credit
        records: List[RetirementRecord] = []
        total_volume = 0.0
        for credit in config.credits:
            registry_str = credit.get("registry", "verra")
            try:
                registry = RegistryType(registry_str)
            except ValueError:
                warnings.append(f"Unknown registry: {registry_str}, defaulting to verra")
                registry = RegistryType.VERRA

            volume = float(credit.get("volume_tco2e", 0))
            vintage = int(credit.get("vintage_year", 2024))

            # Vintage check for PAS 2060
            if config.pas2060_compliance and vintage < config.reporting_year - 3:
                warnings.append(
                    f"Credit vintage {vintage} is >3 years old; "
                    "may not meet PAS 2060 vintage requirements"
                )

            record = RetirementRecord(
                record_id=_new_uuid(),
                registry=registry,
                volume_tco2e=volume,
                vintage_year=vintage,
                project_id=credit.get("project_id", ""),
                project_name=credit.get("project_name", ""),
                serial_number_start=credit.get("serial_start", ""),
                serial_number_end=credit.get("serial_end", ""),
                retirement_purpose=config.retirement_purpose,
                beneficiary_name=config.beneficiary_name,
                beneficiary_country=config.beneficiary_country,
                status=RetirementStatus.PENDING,
            )
            records.append(record)
            total_volume += volume

        self._records = records

        outputs["credits_validated"] = len(records)
        outputs["total_volume_tco2e"] = round(total_volume, 2)
        outputs["registries_involved"] = list(set(r.registry.value for r in records))

        status = PhaseStatus.FAILED if errors else PhaseStatus.COMPLETED
        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name=RetirementPhase.PRE_RETIREMENT_VALIDATION.value,
            status=status, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=_compute_hash(outputs),
        )

    async def _phase_registry_execution(self, config: RetirementConfig) -> PhaseResult:
        """Execute retirement on registry platforms."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        submitted = 0
        for record in self._records:
            if record.status != RetirementStatus.PENDING:
                continue

            # Simulate registry submission
            registry_info = REGISTRY_ENDPOINTS.get(record.registry.value, {})
            record.status = RetirementStatus.RETIRED
            record.retirement_date = utcnow()
            record.registry_transaction_id = f"TX-{_new_uuid()[:12]}"
            submitted += 1

            self.logger.info(
                "Retired %.2f tCO2e on %s, tx=%s",
                record.volume_tco2e, record.registry.value,
                record.registry_transaction_id,
            )

        outputs["submitted_count"] = submitted
        outputs["total_retired_tco2e"] = round(
            sum(r.volume_tco2e for r in self._records if r.status == RetirementStatus.RETIRED), 2
        )

        status = PhaseStatus.COMPLETED
        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name=RetirementPhase.REGISTRY_EXECUTION.value,
            status=status, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=_compute_hash(outputs),
        )

    async def _phase_confirmation(self, config: RetirementConfig) -> PhaseResult:
        """Confirm retirement and capture certificates."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        confirmations: List[RegistryConfirmation] = []
        for record in self._records:
            if record.status != RetirementStatus.RETIRED:
                continue

            registry_info = REGISTRY_ENDPOINTS.get(record.registry.value, {})
            cert_url = f"https://{registry_info.get('certificate_url', 'registry.example.com')}/{record.registry_transaction_id}"

            confirmation = RegistryConfirmation(
                confirmation_id=_new_uuid(),
                registry=record.registry,
                transaction_id=record.registry_transaction_id,
                retirement_date=record.retirement_date or utcnow(),
                volume_tco2e=record.volume_tco2e,
                certificate_url=cert_url,
                certificate_hash=_compute_hash(cert_url),
                is_final=True,
                confirmation_timestamp=utcnow(),
            )
            confirmations.append(confirmation)

            record.status = RetirementStatus.CONFIRMED
            record.certificate_url = cert_url

        self._confirmations = confirmations

        outputs["confirmations_received"] = len(confirmations)
        outputs["certificates_generated"] = len(confirmations)
        outputs["all_confirmed"] = all(
            r.status == RetirementStatus.CONFIRMED for r in self._records
        )

        status = PhaseStatus.COMPLETED
        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name=RetirementPhase.CONFIRMATION.value,
            status=status, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=_compute_hash(outputs),
        )
