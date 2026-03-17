# -*- coding: utf-8 -*-
"""
Passport Compilation Workflow
==================================

5-phase workflow for compiling battery digital passports per EU Battery
Regulation 2023/1542, Article 77 and Annex XIII. Implements data gathering
from upstream workflows, validation against the mandatory data schema,
passport assembly, QR code generation, and registry submission preparation.

Phases:
    1. DataGathering         -- Collect data from upstream workflows
    2. Validation            -- Validate against mandatory passport schema
    3. PassportAssembly      -- Assemble the digital battery passport
    4. QRGeneration          -- Generate QR code linking to passport
    5. RegistrySubmission    -- Prepare and submit to ESPR registry

Regulatory references:
    - EU Regulation 2023/1542 Art. 77 (battery passport requirements)
    - EU Regulation 2023/1542 Annex XIII (battery passport data)
    - EU Regulation 2024/3013 (Ecodesign ESPR, digital product passport)
    - Commission Implementing Regulation on battery passport data format

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    """Compute SHA-256 hex digest of *data*."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class WorkflowPhase(str, Enum):
    """Phases of the passport compilation workflow."""
    DATA_GATHERING = "data_gathering"
    VALIDATION = "validation"
    PASSPORT_ASSEMBLY = "passport_assembly"
    QR_GENERATION = "qr_generation"
    REGISTRY_SUBMISSION = "registry_submission"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PhaseStatus(str, Enum):
    """Status of a single phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PassportStatus(str, Enum):
    """Battery passport lifecycle status."""
    DRAFT = "draft"
    VALIDATED = "validated"
    ASSEMBLED = "assembled"
    PUBLISHED = "published"
    SUBMITTED = "submitted"
    ACTIVE = "active"
    REVOKED = "revoked"


class DataCategory(str, Enum):
    """Mandatory data categories per Annex XIII."""
    GENERAL_INFORMATION = "general_information"
    CARBON_FOOTPRINT = "carbon_footprint"
    RECYCLED_CONTENT = "recycled_content"
    PERFORMANCE_DURABILITY = "performance_durability"
    SUPPLY_CHAIN_DUE_DILIGENCE = "supply_chain_due_diligence"
    LABELLING_SYMBOLS = "labelling_symbols"
    MATERIAL_COMPOSITION = "material_composition"
    END_OF_LIFE = "end_of_life"
    STATE_OF_HEALTH = "state_of_health"
    STATE_OF_CHARGE = "state_of_charge"


class AccessLevel(str, Enum):
    """Data access levels within the passport."""
    PUBLIC = "public"
    AUTHORIZED = "authorized"
    NOTIFIED_BODY = "notified_body"
    MANUFACTURER_ONLY = "manufacturer_only"


# =============================================================================
# MANDATORY PASSPORT FIELDS (Annex XIII)
# =============================================================================


MANDATORY_FIELDS: Dict[str, List[str]] = {
    DataCategory.GENERAL_INFORMATION.value: [
        "manufacturer_name",
        "manufacturing_place",
        "manufacturing_date",
        "battery_category",
        "battery_model",
        "battery_weight_kg",
        "unique_identifier",
    ],
    DataCategory.CARBON_FOOTPRINT.value: [
        "carbon_footprint_total_kgco2e",
        "carbon_footprint_per_kwh",
        "performance_class",
        "lifecycle_stages",
    ],
    DataCategory.RECYCLED_CONTENT.value: [
        "cobalt_recycled_pct",
        "lithium_recycled_pct",
        "nickel_recycled_pct",
        "lead_recycled_pct",
    ],
    DataCategory.PERFORMANCE_DURABILITY.value: [
        "rated_capacity_ah",
        "nominal_voltage_v",
        "energy_capacity_kwh",
        "cycle_life_expected",
        "round_trip_efficiency_pct",
    ],
    DataCategory.SUPPLY_CHAIN_DUE_DILIGENCE.value: [
        "due_diligence_policy",
        "third_party_audit",
    ],
    DataCategory.MATERIAL_COMPOSITION.value: [
        "cathode_chemistry",
        "hazardous_substances",
    ],
    DataCategory.END_OF_LIFE.value: [
        "collection_instructions",
        "recycling_information",
    ],
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class PassportDataSection(BaseModel):
    """A data section within the passport."""
    category: str = Field(..., description="Data category name")
    fields: Dict[str, Any] = Field(default_factory=dict)
    completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    access_level: AccessLevel = Field(default=AccessLevel.PUBLIC)
    validated: bool = Field(default=False)
    validation_errors: List[str] = Field(default_factory=list)


class QRCodeInfo(BaseModel):
    """QR code metadata."""
    qr_code_id: str = Field(default_factory=lambda: f"qr-{_new_uuid()[:8]}")
    passport_url: str = Field(default="")
    encoded_data: str = Field(default="")
    format: str = Field(default="URL")
    generated_at: str = Field(default="")


class RegistrySubmission(BaseModel):
    """Registry submission record."""
    submission_id: str = Field(default_factory=lambda: f"sub-{_new_uuid()[:8]}")
    registry_name: str = Field(default="EU Battery Passport Registry")
    submission_status: str = Field(default="prepared")
    passport_id: str = Field(default="")
    submitted_at: str = Field(default="")
    acknowledgement_id: str = Field(default="")


class PassportCompilationInput(BaseModel):
    """Input data model for PassportCompilationWorkflow."""
    battery_id: str = Field(default_factory=lambda: f"bat-{_new_uuid()[:8]}")
    battery_model: str = Field(default="", description="Battery model identifier")
    manufacturer_name: str = Field(default="", description="Manufacturer legal name")
    manufacturing_place: str = Field(default="", description="Manufacturing location")
    manufacturing_date: str = Field(default="", description="ISO date string")
    battery_category: str = Field(default="ev_battery")
    battery_weight_kg: float = Field(default=0.0, ge=0.0)
    general_info: Dict[str, Any] = Field(default_factory=dict)
    carbon_footprint_data: Dict[str, Any] = Field(default_factory=dict)
    recycled_content_data: Dict[str, Any] = Field(default_factory=dict)
    performance_data: Dict[str, Any] = Field(default_factory=dict)
    due_diligence_data: Dict[str, Any] = Field(default_factory=dict)
    material_composition_data: Dict[str, Any] = Field(default_factory=dict)
    end_of_life_data: Dict[str, Any] = Field(default_factory=dict)
    state_of_health_data: Dict[str, Any] = Field(default_factory=dict)
    labelling_data: Dict[str, Any] = Field(default_factory=dict)
    passport_base_url: str = Field(
        default="https://battery-passport.eu/p/",
        description="Base URL for passport public access"
    )
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)


class PassportCompilationResult(BaseModel):
    """Complete result from passport compilation workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="passport_compilation")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    total_duration_seconds: float = Field(default=0.0)
    duration_ms: float = Field(default=0.0)
    passport_id: str = Field(default="")
    passport_status: str = Field(default="draft")
    battery_id: str = Field(default="")
    data_sections: List[PassportDataSection] = Field(default_factory=list)
    overall_completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    validation_passed: bool = Field(default=False)
    validation_error_count: int = Field(default=0, ge=0)
    qr_code: Optional[QRCodeInfo] = Field(default=None)
    registry_submission: Optional[RegistrySubmission] = Field(default=None)
    reporting_year: int = Field(default=2025)
    executed_at: str = Field(default="")
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class PassportCompilationWorkflow:
    """
    5-phase battery passport compilation workflow per EU Battery Regulation.

    Implements end-to-end digital battery passport assembly following
    EU Regulation 2023/1542 Art. 77 and Annex XIII. Gathers data from
    upstream workflows, validates against the mandatory schema, assembles
    the passport document, generates a QR code, and prepares registry
    submission.

    Zero-hallucination: all validation uses deterministic field presence
    checks and completeness calculations. No LLM in validation paths.

    Example:
        >>> wf = PassportCompilationWorkflow()
        >>> inp = PassportCompilationInput(
        ...     battery_id="BAT-001",
        ...     manufacturer_name="ACME Batteries",
        ...     carbon_footprint_data={...},
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.passport_id != ""
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize PassportCompilationWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._passport_id: str = ""
        self._sections: List[PassportDataSection] = []
        self._validation_passed: bool = False
        self._qr_code: Optional[QRCodeInfo] = None
        self._registry_sub: Optional[RegistrySubmission] = None
        self._passport_status: str = PassportStatus.DRAFT.value
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.DATA_GATHERING.value, "description": "Collect data from upstream workflows"},
            {"name": WorkflowPhase.VALIDATION.value, "description": "Validate against mandatory passport schema"},
            {"name": WorkflowPhase.PASSPORT_ASSEMBLY.value, "description": "Assemble the digital battery passport"},
            {"name": WorkflowPhase.QR_GENERATION.value, "description": "Generate QR code linking to passport"},
            {"name": WorkflowPhase.REGISTRY_SUBMISSION.value, "description": "Submit to ESPR registry"},
        ]

    def validate_inputs(self, input_data: PassportCompilationInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.battery_id:
            issues.append("Battery ID is required")
        if not input_data.manufacturer_name:
            issues.append("Manufacturer name is required")
        return issues

    async def execute(
        self,
        input_data: Optional[PassportCompilationInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> PassportCompilationResult:
        """
        Execute the 5-phase passport compilation workflow.

        Args:
            input_data: Full input model (preferred).
            config: Configuration overrides.

        Returns:
            PassportCompilationResult with assembled passport, QR code,
            and registry submission status.
        """
        if input_data is None:
            input_data = PassportCompilationInput(config=config or {})

        started_at = _utcnow()
        self.logger.info("Starting passport compilation workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS
        phases_done = 0

        try:
            phase_results.append(await self._phase_data_gathering(input_data))
            phases_done += 1
            phase_results.append(await self._phase_validation(input_data))
            phases_done += 1
            phase_results.append(await self._phase_passport_assembly(input_data))
            phases_done += 1
            phase_results.append(await self._phase_qr_generation(input_data))
            phases_done += 1
            phase_results.append(await self._phase_registry_submission(input_data))
            phases_done += 1
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error(
                "Passport compilation workflow failed: %s", exc, exc_info=True
            )
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()
        total_completeness = self._compute_overall_completeness()
        total_errors = sum(
            len(s.validation_errors) for s in self._sections
        )

        result = PassportCompilationResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=phases_done,
            total_duration_seconds=elapsed,
            duration_ms=round(elapsed * 1000, 2),
            passport_id=self._passport_id,
            passport_status=self._passport_status,
            battery_id=input_data.battery_id,
            data_sections=self._sections,
            overall_completeness_pct=total_completeness,
            validation_passed=self._validation_passed,
            validation_error_count=total_errors,
            qr_code=self._qr_code,
            registry_submission=self._registry_sub,
            reporting_year=input_data.reporting_year,
            executed_at=_utcnow().isoformat(),
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Passport compilation %s completed in %.2fs: %s, completeness %.1f%%",
            self.workflow_id, elapsed, self._passport_status, total_completeness,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Data Gathering
    # -------------------------------------------------------------------------

    async def _phase_data_gathering(
        self, input_data: PassportCompilationInput,
    ) -> PhaseResult:
        """Collect data from upstream workflows and external sources."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._passport_id = f"BP-{input_data.battery_id}-{_new_uuid()[:6]}"

        # Build general info section from input fields
        general_fields = {
            "manufacturer_name": input_data.manufacturer_name,
            "manufacturing_place": input_data.manufacturing_place,
            "manufacturing_date": input_data.manufacturing_date,
            "battery_category": input_data.battery_category,
            "battery_model": input_data.battery_model,
            "battery_weight_kg": input_data.battery_weight_kg,
            "unique_identifier": input_data.battery_id,
        }
        general_fields.update(input_data.general_info)

        # Map input data sections
        section_data_map: Dict[str, Dict[str, Any]] = {
            DataCategory.GENERAL_INFORMATION.value: general_fields,
            DataCategory.CARBON_FOOTPRINT.value: input_data.carbon_footprint_data,
            DataCategory.RECYCLED_CONTENT.value: input_data.recycled_content_data,
            DataCategory.PERFORMANCE_DURABILITY.value: input_data.performance_data,
            DataCategory.SUPPLY_CHAIN_DUE_DILIGENCE.value: input_data.due_diligence_data,
            DataCategory.MATERIAL_COMPOSITION.value: input_data.material_composition_data,
            DataCategory.END_OF_LIFE.value: input_data.end_of_life_data,
            DataCategory.LABELLING_SYMBOLS.value: input_data.labelling_data,
        }

        # Determine access levels per category
        access_map: Dict[str, AccessLevel] = {
            DataCategory.GENERAL_INFORMATION.value: AccessLevel.PUBLIC,
            DataCategory.CARBON_FOOTPRINT.value: AccessLevel.PUBLIC,
            DataCategory.RECYCLED_CONTENT.value: AccessLevel.AUTHORIZED,
            DataCategory.PERFORMANCE_DURABILITY.value: AccessLevel.PUBLIC,
            DataCategory.SUPPLY_CHAIN_DUE_DILIGENCE.value: AccessLevel.NOTIFIED_BODY,
            DataCategory.MATERIAL_COMPOSITION.value: AccessLevel.AUTHORIZED,
            DataCategory.END_OF_LIFE.value: AccessLevel.PUBLIC,
            DataCategory.LABELLING_SYMBOLS.value: AccessLevel.PUBLIC,
        }

        self._sections = []
        for cat, fields in section_data_map.items():
            self._sections.append(PassportDataSection(
                category=cat,
                fields=fields,
                access_level=access_map.get(cat, AccessLevel.AUTHORIZED),
            ))

        populated = sum(1 for s in self._sections if s.fields)
        empty = sum(1 for s in self._sections if not s.fields)

        outputs["passport_id"] = self._passport_id
        outputs["sections_populated"] = populated
        outputs["sections_empty"] = empty
        outputs["total_sections"] = len(self._sections)
        outputs["data_categories"] = [s.category for s in self._sections]

        if empty > 0:
            empty_cats = [s.category for s in self._sections if not s.fields]
            warnings.append(
                f"{empty} data sections have no data: {', '.join(empty_cats)}"
            )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 DataGathering: %d sections, %d populated",
            len(self._sections), populated,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.DATA_GATHERING.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Validation
    # -------------------------------------------------------------------------

    async def _phase_validation(
        self, input_data: PassportCompilationInput,
    ) -> PhaseResult:
        """Validate passport data against mandatory schema."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        total_errors = 0

        for section in self._sections:
            errors: List[str] = []
            mandatory = MANDATORY_FIELDS.get(section.category, [])

            if not mandatory:
                section.validated = True
                section.completeness_pct = 100.0 if section.fields else 0.0
                continue

            present = 0
            for field_name in mandatory:
                value = section.fields.get(field_name)
                if value is None or value == "" or value == 0:
                    errors.append(
                        f"{section.category}.{field_name}: missing or empty"
                    )
                else:
                    present += 1

            section.completeness_pct = round(
                (present / len(mandatory) * 100) if mandatory else 100.0, 1
            )
            section.validation_errors = errors
            section.validated = len(errors) == 0
            total_errors += len(errors)

        self._validation_passed = total_errors == 0

        section_results: Dict[str, Dict[str, Any]] = {}
        for s in self._sections:
            section_results[s.category] = {
                "completeness_pct": s.completeness_pct,
                "validated": s.validated,
                "error_count": len(s.validation_errors),
            }

        outputs["total_validation_errors"] = total_errors
        outputs["validation_passed"] = self._validation_passed
        outputs["section_results"] = section_results
        outputs["overall_completeness_pct"] = self._compute_overall_completeness()

        if not self._validation_passed:
            warnings.append(
                f"Validation failed with {total_errors} errors across sections"
            )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 Validation: passed=%s, errors=%d",
            self._validation_passed, total_errors,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.VALIDATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Passport Assembly
    # -------------------------------------------------------------------------

    async def _phase_passport_assembly(
        self, input_data: PassportCompilationInput,
    ) -> PhaseResult:
        """Assemble the digital battery passport document."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        passport_document = {
            "passport_id": self._passport_id,
            "schema_version": _MODULE_VERSION,
            "regulation": "EU Regulation 2023/1542",
            "created_at": _utcnow().isoformat(),
            "battery_id": input_data.battery_id,
            "sections": {},
        }

        for section in self._sections:
            passport_document["sections"][section.category] = {
                "data": section.fields,
                "access_level": section.access_level.value,
                "completeness_pct": section.completeness_pct,
                "validated": section.validated,
            }

        # Compute passport data hash for integrity verification
        passport_json = json.dumps(passport_document, sort_keys=True, default=str)
        passport_hash = _compute_hash(passport_json)
        passport_document["integrity_hash"] = passport_hash

        self._passport_status = (
            PassportStatus.ASSEMBLED.value
            if self._validation_passed
            else PassportStatus.DRAFT.value
        )

        outputs["passport_id"] = self._passport_id
        outputs["passport_status"] = self._passport_status
        outputs["schema_version"] = _MODULE_VERSION
        outputs["integrity_hash"] = passport_hash
        outputs["sections_assembled"] = len(passport_document["sections"])
        outputs["total_fields"] = sum(
            len(s.fields) for s in self._sections
        )

        if not self._validation_passed:
            warnings.append(
                "Passport assembled in DRAFT status due to validation errors"
            )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 PassportAssembly: %s assembled, status=%s",
            self._passport_id, self._passport_status,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.PASSPORT_ASSEMBLY.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: QR Generation
    # -------------------------------------------------------------------------

    async def _phase_qr_generation(
        self, input_data: PassportCompilationInput,
    ) -> PhaseResult:
        """Generate QR code linking to the battery passport."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        passport_url = f"{input_data.passport_base_url}{self._passport_id}"

        # Deterministic QR payload construction
        qr_payload = {
            "passport_id": self._passport_id,
            "url": passport_url,
            "battery_id": input_data.battery_id,
            "manufacturer": input_data.manufacturer_name,
            "schema_version": _MODULE_VERSION,
        }
        encoded_data = json.dumps(qr_payload, sort_keys=True)
        qr_hash = _compute_hash(encoded_data)

        self._qr_code = QRCodeInfo(
            passport_url=passport_url,
            encoded_data=encoded_data,
            format="URL",
            generated_at=_utcnow().isoformat(),
        )

        outputs["qr_code_id"] = self._qr_code.qr_code_id
        outputs["passport_url"] = passport_url
        outputs["qr_format"] = "URL"
        outputs["qr_data_hash"] = qr_hash
        outputs["encoded_payload_length"] = len(encoded_data)

        if not input_data.manufacturer_name:
            warnings.append("QR payload missing manufacturer name")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 QRGeneration: %s generated for %s",
            self._qr_code.qr_code_id, passport_url,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.QR_GENERATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Registry Submission
    # -------------------------------------------------------------------------

    async def _phase_registry_submission(
        self, input_data: PassportCompilationInput,
    ) -> PhaseResult:
        """Prepare and record registry submission."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        can_submit = (
            self._validation_passed
            and self._passport_status == PassportStatus.ASSEMBLED.value
        )

        submission_status = "prepared" if can_submit else "blocked"

        self._registry_sub = RegistrySubmission(
            passport_id=self._passport_id,
            submission_status=submission_status,
            submitted_at=_utcnow().isoformat() if can_submit else "",
            acknowledgement_id=f"ACK-{_new_uuid()[:8]}" if can_submit else "",
        )

        if can_submit:
            self._passport_status = PassportStatus.SUBMITTED.value
        else:
            warnings.append(
                "Registry submission blocked: passport validation incomplete"
            )

        completeness = self._compute_overall_completeness()
        readiness_checks = {
            "validation_passed": self._validation_passed,
            "passport_assembled": self._passport_status in (
                PassportStatus.ASSEMBLED.value,
                PassportStatus.SUBMITTED.value,
            ),
            "qr_code_generated": self._qr_code is not None,
            "completeness_above_80": completeness >= 80.0,
        }

        outputs["submission_id"] = self._registry_sub.submission_id
        outputs["submission_status"] = submission_status
        outputs["registry_name"] = self._registry_sub.registry_name
        outputs["acknowledgement_id"] = self._registry_sub.acknowledgement_id
        outputs["readiness_checks"] = readiness_checks
        outputs["passport_final_status"] = self._passport_status

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 5 RegistrySubmission: %s, status=%s",
            self._registry_sub.submission_id, submission_status,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.REGISTRY_SUBMISSION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_overall_completeness(self) -> float:
        """Compute weighted-average completeness across all sections."""
        if not self._sections:
            return 0.0
        total = sum(s.completeness_pct for s in self._sections)
        return round(total / len(self._sections), 1)

    def _compute_provenance(self, result: PassportCompilationResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
