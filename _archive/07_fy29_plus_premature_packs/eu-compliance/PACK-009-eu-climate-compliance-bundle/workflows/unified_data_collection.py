# -*- coding: utf-8 -*-
"""
Unified Data Collection Workflow
====================================

Five-phase workflow that collects data requirements across all four constituent
regulatory packs (CSRD, CBAM, EU Taxonomy, EUDR) in the EU Climate Compliance
Bundle, deduplicates them into a minimal collection set, validates collected
data, routes it to appropriate pack pipelines, and confirms receipt.

Phases:
    1. RequirementsMapping - Scans all 4 regulations for data requirements
    2. DeduplicatedCollection - Builds minimal collection set
    3. Validation - Validates collected data against requirements
    4. Distribution - Routes validated data to pack pipelines
    5. Confirmation - Confirms receipt by all constituent packs

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

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


class RegulationPack(str, Enum):
    """Constituent regulation packs in the bundle."""
    CSRD = "CSRD"
    CBAM = "CBAM"
    EU_TAXONOMY = "EU_TAXONOMY"
    EUDR = "EUDR"


class DataType(str, Enum):
    """Data type classification for requirements."""
    NUMERIC = "NUMERIC"
    TEXT = "TEXT"
    DATE = "DATE"
    BOOLEAN = "BOOLEAN"
    ENUM = "ENUM"
    CURRENCY = "CURRENCY"
    PERCENTAGE = "PERCENTAGE"
    GEOLOCATION = "GEOLOCATION"


class ValidationSeverity(str, Enum):
    """Severity of a validation issue."""
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class DistributionStatus(str, Enum):
    """Status of data distribution to a pack."""
    DELIVERED = "DELIVERED"
    PENDING = "PENDING"
    FAILED = "FAILED"
    REJECTED = "REJECTED"


# =============================================================================
# REGULATION DATA REQUIREMENTS REGISTRY
# =============================================================================


REGULATION_REQUIREMENTS: Dict[str, List[Dict[str, Any]]] = {
    RegulationPack.CSRD.value: [
        {"req_id": "CSRD-001", "field": "total_ghg_emissions_scope1_tco2e", "data_type": "NUMERIC", "unit": "tCO2e", "mandatory": True, "description": "Total Scope 1 GHG emissions"},
        {"req_id": "CSRD-002", "field": "total_ghg_emissions_scope2_tco2e", "data_type": "NUMERIC", "unit": "tCO2e", "mandatory": True, "description": "Total Scope 2 GHG emissions"},
        {"req_id": "CSRD-003", "field": "total_ghg_emissions_scope3_tco2e", "data_type": "NUMERIC", "unit": "tCO2e", "mandatory": True, "description": "Total Scope 3 GHG emissions"},
        {"req_id": "CSRD-004", "field": "energy_consumption_mwh", "data_type": "NUMERIC", "unit": "MWh", "mandatory": True, "description": "Total energy consumption"},
        {"req_id": "CSRD-005", "field": "renewable_energy_pct", "data_type": "PERCENTAGE", "unit": "%", "mandatory": True, "description": "Share of renewable energy"},
        {"req_id": "CSRD-006", "field": "water_consumption_m3", "data_type": "NUMERIC", "unit": "m3", "mandatory": False, "description": "Total water consumption"},
        {"req_id": "CSRD-007", "field": "waste_generated_tonnes", "data_type": "NUMERIC", "unit": "tonnes", "mandatory": False, "description": "Total waste generated"},
        {"req_id": "CSRD-008", "field": "biodiversity_impact_assessment", "data_type": "TEXT", "unit": None, "mandatory": False, "description": "Biodiversity impact narrative"},
        {"req_id": "CSRD-009", "field": "climate_transition_plan", "data_type": "TEXT", "unit": None, "mandatory": True, "description": "Climate transition plan description"},
        {"req_id": "CSRD-010", "field": "internal_carbon_price_eur", "data_type": "CURRENCY", "unit": "EUR", "mandatory": False, "description": "Internal carbon pricing mechanism"},
        {"req_id": "CSRD-011", "field": "reporting_period_start", "data_type": "DATE", "unit": None, "mandatory": True, "description": "Reporting period start date"},
        {"req_id": "CSRD-012", "field": "reporting_period_end", "data_type": "DATE", "unit": None, "mandatory": True, "description": "Reporting period end date"},
    ],
    RegulationPack.CBAM.value: [
        {"req_id": "CBAM-001", "field": "total_ghg_emissions_scope1_tco2e", "data_type": "NUMERIC", "unit": "tCO2e", "mandatory": True, "description": "Scope 1 emissions for embedded emissions"},
        {"req_id": "CBAM-002", "field": "total_ghg_emissions_scope2_tco2e", "data_type": "NUMERIC", "unit": "tCO2e", "mandatory": True, "description": "Scope 2 for electricity in production"},
        {"req_id": "CBAM-003", "field": "embedded_emissions_tco2e_per_tonne", "data_type": "NUMERIC", "unit": "tCO2e/t", "mandatory": True, "description": "Specific embedded emissions"},
        {"req_id": "CBAM-004", "field": "import_volume_tonnes", "data_type": "NUMERIC", "unit": "tonnes", "mandatory": True, "description": "Total import volume"},
        {"req_id": "CBAM-005", "field": "carbon_price_paid_origin_eur", "data_type": "CURRENCY", "unit": "EUR", "mandatory": False, "description": "Carbon price paid in country of origin"},
        {"req_id": "CBAM-006", "field": "goods_category_cn_codes", "data_type": "TEXT", "unit": None, "mandatory": True, "description": "CN codes for imported goods"},
        {"req_id": "CBAM-007", "field": "supplier_installation_id", "data_type": "TEXT", "unit": None, "mandatory": True, "description": "Supplier installation identifier"},
        {"req_id": "CBAM-008", "field": "country_of_origin", "data_type": "TEXT", "unit": None, "mandatory": True, "description": "Country of origin for imports"},
        {"req_id": "CBAM-009", "field": "reporting_period_start", "data_type": "DATE", "unit": None, "mandatory": True, "description": "Quarterly reporting period start"},
        {"req_id": "CBAM-010", "field": "reporting_period_end", "data_type": "DATE", "unit": None, "mandatory": True, "description": "Quarterly reporting period end"},
    ],
    RegulationPack.EU_TAXONOMY.value: [
        {"req_id": "TAX-001", "field": "total_ghg_emissions_scope1_tco2e", "data_type": "NUMERIC", "unit": "tCO2e", "mandatory": True, "description": "Scope 1 for DNSH assessment"},
        {"req_id": "TAX-002", "field": "total_ghg_emissions_scope2_tco2e", "data_type": "NUMERIC", "unit": "tCO2e", "mandatory": True, "description": "Scope 2 for DNSH assessment"},
        {"req_id": "TAX-003", "field": "energy_consumption_mwh", "data_type": "NUMERIC", "unit": "MWh", "mandatory": True, "description": "Energy consumption for alignment"},
        {"req_id": "TAX-004", "field": "renewable_energy_pct", "data_type": "PERCENTAGE", "unit": "%", "mandatory": True, "description": "Renewable share for climate mitigation"},
        {"req_id": "TAX-005", "field": "water_consumption_m3", "data_type": "NUMERIC", "unit": "m3", "mandatory": True, "description": "Water use for DNSH assessment"},
        {"req_id": "TAX-006", "field": "taxonomy_eligible_revenue_eur", "data_type": "CURRENCY", "unit": "EUR", "mandatory": True, "description": "Taxonomy-eligible revenue"},
        {"req_id": "TAX-007", "field": "taxonomy_aligned_revenue_eur", "data_type": "CURRENCY", "unit": "EUR", "mandatory": True, "description": "Taxonomy-aligned revenue"},
        {"req_id": "TAX-008", "field": "taxonomy_eligible_capex_eur", "data_type": "CURRENCY", "unit": "EUR", "mandatory": True, "description": "Taxonomy-eligible CapEx"},
        {"req_id": "TAX-009", "field": "taxonomy_aligned_capex_eur", "data_type": "CURRENCY", "unit": "EUR", "mandatory": True, "description": "Taxonomy-aligned CapEx"},
        {"req_id": "TAX-010", "field": "taxonomy_eligible_opex_eur", "data_type": "CURRENCY", "unit": "EUR", "mandatory": False, "description": "Taxonomy-eligible OpEx"},
        {"req_id": "TAX-011", "field": "reporting_period_start", "data_type": "DATE", "unit": None, "mandatory": True, "description": "Reporting period start"},
        {"req_id": "TAX-012", "field": "reporting_period_end", "data_type": "DATE", "unit": None, "mandatory": True, "description": "Reporting period end"},
    ],
    RegulationPack.EUDR.value: [
        {"req_id": "EUDR-001", "field": "commodity_type", "data_type": "ENUM", "unit": None, "mandatory": True, "description": "Commodity type (soy, palm oil, etc.)"},
        {"req_id": "EUDR-002", "field": "geolocation_of_production", "data_type": "GEOLOCATION", "unit": None, "mandatory": True, "description": "GPS coordinates of production plot"},
        {"req_id": "EUDR-003", "field": "country_of_origin", "data_type": "TEXT", "unit": None, "mandatory": True, "description": "Country of production"},
        {"req_id": "EUDR-004", "field": "supplier_installation_id", "data_type": "TEXT", "unit": None, "mandatory": True, "description": "Supplier/producer identifier"},
        {"req_id": "EUDR-005", "field": "deforestation_free_date", "data_type": "DATE", "unit": None, "mandatory": True, "description": "Cut-off date for deforestation-free"},
        {"req_id": "EUDR-006", "field": "import_volume_tonnes", "data_type": "NUMERIC", "unit": "tonnes", "mandatory": True, "description": "Volume of commodity imported"},
        {"req_id": "EUDR-007", "field": "due_diligence_status", "data_type": "ENUM", "unit": None, "mandatory": True, "description": "Due diligence completion status"},
        {"req_id": "EUDR-008", "field": "risk_assessment_score", "data_type": "NUMERIC", "unit": None, "mandatory": True, "description": "Risk assessment score"},
        {"req_id": "EUDR-009", "field": "reporting_period_start", "data_type": "DATE", "unit": None, "mandatory": True, "description": "Reporting period start"},
        {"req_id": "EUDR-010", "field": "reporting_period_end", "data_type": "DATE", "unit": None, "mandatory": True, "description": "Reporting period end"},
    ],
}

# Fields that appear in multiple regulations (for deduplication)
CROSS_REGULATION_FIELD_MAP: Dict[str, List[str]] = {
    "total_ghg_emissions_scope1_tco2e": ["CSRD-001", "CBAM-001", "TAX-001"],
    "total_ghg_emissions_scope2_tco2e": ["CSRD-002", "CBAM-002", "TAX-002"],
    "energy_consumption_mwh": ["CSRD-004", "TAX-003"],
    "renewable_energy_pct": ["CSRD-005", "TAX-004"],
    "water_consumption_m3": ["CSRD-006", "TAX-005"],
    "reporting_period_start": ["CSRD-011", "CBAM-009", "TAX-011", "EUDR-009"],
    "reporting_period_end": ["CSRD-012", "CBAM-010", "TAX-012", "EUDR-010"],
    "import_volume_tonnes": ["CBAM-004", "EUDR-006"],
    "country_of_origin": ["CBAM-008", "EUDR-003"],
    "supplier_installation_id": ["CBAM-007", "EUDR-004"],
}


# =============================================================================
# DATA MODELS
# =============================================================================


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


class DataRequirement(BaseModel):
    """A single data requirement from a regulation."""
    req_id: str = Field(...)
    field: str = Field(...)
    data_type: str = Field(...)
    unit: Optional[str] = Field(None)
    mandatory: bool = Field(default=True)
    description: str = Field(default="")
    source_regulations: List[str] = Field(default_factory=list)


class CollectedDataPoint(BaseModel):
    """A collected data value satisfying one or more requirements."""
    field: str = Field(...)
    value: Any = Field(...)
    data_type: str = Field(...)
    unit: Optional[str] = Field(None)
    satisfies_requirements: List[str] = Field(default_factory=list)
    collected_at: str = Field(default="")
    source: str = Field(default="manual")


class ValidationIssue(BaseModel):
    """A validation issue found during data validation."""
    field: str = Field(...)
    issue_type: str = Field(...)
    severity: ValidationSeverity = Field(...)
    message: str = Field(...)
    req_ids: List[str] = Field(default_factory=list)


class WorkflowConfig(BaseModel):
    """Configuration for the unified data collection workflow."""
    organization_id: str = Field(...)
    reporting_year: int = Field(..., ge=2024, le=2050)
    target_packs: List[RegulationPack] = Field(
        default_factory=lambda: list(RegulationPack)
    )
    collected_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Pre-collected data values keyed by field name"
    )
    validation_strictness: str = Field(
        default="standard",
        description="Validation mode: strict, standard, or lenient"
    )
    auto_distribute: bool = Field(
        default=True,
        description="Automatically distribute data to packs after validation"
    )
    skip_phases: List[str] = Field(default_factory=list)


class UnifiedDataCollectionResult(WorkflowResult):
    """Result from the unified data collection workflow."""
    total_requirements: int = Field(default=0)
    deduplicated_requirements: int = Field(default=0)
    data_points_collected: int = Field(default=0)
    validation_errors: int = Field(default=0)
    packs_distributed: int = Field(default=0)
    packs_confirmed: int = Field(default=0)


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class UnifiedDataCollectionWorkflow:
    """
    Five-phase unified data collection workflow.

    Scans all four constituent regulation packs for data requirements,
    deduplicates into a minimal collection set, validates collected data,
    distributes to pack pipelines, and confirms receipt.

    Attributes:
        workflow_id: Unique execution identifier.
        config: Workflow configuration.

    Example:
        >>> wf = UnifiedDataCollectionWorkflow()
        >>> config = WorkflowConfig(
        ...     organization_id="org-123",
        ...     reporting_year=2026,
        ...     collected_data={"total_ghg_emissions_scope1_tco2e": 15000.0}
        ... )
        >>> result = wf.execute(config)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    WORKFLOW_NAME = "unified_data_collection"

    PHASE_ORDER = [
        "requirements_mapping",
        "deduplicated_collection",
        "validation",
        "distribution",
        "confirmation",
    ]

    def __init__(self) -> None:
        """Initialize the unified data collection workflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self._phase_outputs: Dict[str, Dict[str, Any]] = {}

    def execute(self, config: WorkflowConfig) -> UnifiedDataCollectionResult:
        """
        Execute the five-phase unified data collection workflow.

        Args:
            config: Validated workflow configuration.

        Returns:
            UnifiedDataCollectionResult with collection outcomes.
        """
        started_at = datetime.utcnow()
        logger.info(
            "Starting unified data collection %s for org=%s year=%d",
            self.workflow_id, config.organization_id, config.reporting_year,
        )

        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING
        phase_methods = {
            "requirements_mapping": self._phase_requirements_mapping,
            "deduplicated_collection": self._phase_deduplicated_collection,
            "validation": self._phase_validation,
            "distribution": self._phase_distribution,
            "confirmation": self._phase_confirmation,
        }

        for phase_name in self.PHASE_ORDER:
            if phase_name in config.skip_phases:
                skip_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.SKIPPED,
                    provenance_hash=_hash_data({"skipped": True}),
                )
                completed_phases.append(skip_result)
                continue

            try:
                phase_result = phase_methods[phase_name](config)
                completed_phases.append(phase_result)

                if phase_result.status == PhaseStatus.COMPLETED:
                    self._phase_outputs[phase_name] = phase_result.outputs
                elif phase_result.status == PhaseStatus.FAILED:
                    overall_status = WorkflowStatus.FAILED
                    break
            except Exception as exc:
                logger.error("Phase '%s' raised: %s", phase_name, exc, exc_info=True)
                error_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    errors=[str(exc)],
                    provenance_hash=_hash_data({"error": str(exc)}),
                )
                completed_phases.append(error_result)
                overall_status = WorkflowStatus.FAILED
                break

        if overall_status == WorkflowStatus.RUNNING:
            all_ok = all(
                p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                for p in completed_phases
            )
            overall_status = WorkflowStatus.COMPLETED if all_ok else WorkflowStatus.PARTIAL

        completed_at = datetime.utcnow()
        summary = self._build_summary()
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })

        return UnifiedDataCollectionResult(
            workflow_id=self.workflow_id,
            workflow_name=self.WORKFLOW_NAME,
            status=overall_status,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=(completed_at - started_at).total_seconds(),
            phases=completed_phases,
            summary=summary,
            provenance_hash=provenance,
            total_requirements=summary.get("total_requirements", 0),
            deduplicated_requirements=summary.get("deduplicated_requirements", 0),
            data_points_collected=summary.get("data_points_collected", 0),
            validation_errors=summary.get("validation_errors", 0),
            packs_distributed=summary.get("packs_distributed", 0),
            packs_confirmed=summary.get("packs_confirmed", 0),
        )

    # -------------------------------------------------------------------------
    # Phase 1: Requirements Mapping
    # -------------------------------------------------------------------------

    def _phase_requirements_mapping(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 1: Scan all 4 regulations for data requirements.

        Builds a unified requirements list from all target packs, recording
        which regulation each requirement originates from.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            all_requirements: List[Dict[str, Any]] = []
            per_pack_counts: Dict[str, int] = {}

            for pack in config.target_packs:
                pack_reqs = REGULATION_REQUIREMENTS.get(pack.value, [])
                per_pack_counts[pack.value] = len(pack_reqs)
                for req in pack_reqs:
                    enriched_req = dict(req)
                    enriched_req["source_regulation"] = pack.value
                    all_requirements.append(enriched_req)

            outputs["all_requirements"] = all_requirements
            outputs["total_requirements"] = len(all_requirements)
            outputs["per_pack_counts"] = per_pack_counts
            outputs["packs_scanned"] = [p.value for p in config.target_packs]

            field_to_regs: Dict[str, List[str]] = {}
            for req in all_requirements:
                field = req["field"]
                reg = req["source_regulation"]
                if field not in field_to_regs:
                    field_to_regs[field] = []
                if reg not in field_to_regs[field]:
                    field_to_regs[field].append(reg)

            shared_fields = {
                f: regs for f, regs in field_to_regs.items() if len(regs) > 1
            }
            outputs["shared_fields"] = shared_fields
            outputs["shared_field_count"] = len(shared_fields)

            logger.info(
                "Requirements mapping complete: %d total, %d shared fields",
                len(all_requirements), len(shared_fields),
            )

            status = PhaseStatus.COMPLETED
            records = len(all_requirements)

        except Exception as exc:
            logger.error("Requirements mapping failed: %s", exc, exc_info=True)
            errors.append(f"Requirements mapping failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="requirements_mapping",
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

    # -------------------------------------------------------------------------
    # Phase 2: Deduplicated Collection
    # -------------------------------------------------------------------------

    def _phase_deduplicated_collection(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 2: Deduplicate requirements into minimal collection set.

        Merges fields that appear in multiple regulations into a single
        collection requirement, tracking all satisfied regulation req_ids.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            mapping_outputs = self._phase_outputs.get("requirements_mapping", {})
            all_requirements = mapping_outputs.get("all_requirements", [])

            deduplicated: Dict[str, Dict[str, Any]] = {}
            for req in all_requirements:
                field = req["field"]
                if field not in deduplicated:
                    deduplicated[field] = {
                        "field": field,
                        "data_type": req["data_type"],
                        "unit": req.get("unit"),
                        "mandatory": req["mandatory"],
                        "description": req["description"],
                        "source_req_ids": [req["req_id"]],
                        "source_regulations": [req["source_regulation"]],
                    }
                else:
                    existing = deduplicated[field]
                    existing["source_req_ids"].append(req["req_id"])
                    if req["source_regulation"] not in existing["source_regulations"]:
                        existing["source_regulations"].append(req["source_regulation"])
                    if req["mandatory"]:
                        existing["mandatory"] = True

            dedup_list = list(deduplicated.values())
            outputs["deduplicated_requirements"] = dedup_list
            outputs["deduplicated_count"] = len(dedup_list)
            outputs["original_count"] = len(all_requirements)
            outputs["deduplication_savings"] = len(all_requirements) - len(dedup_list)

            mandatory_fields = [r for r in dedup_list if r["mandatory"]]
            optional_fields = [r for r in dedup_list if not r["mandatory"]]
            outputs["mandatory_count"] = len(mandatory_fields)
            outputs["optional_count"] = len(optional_fields)

            collected_points: List[Dict[str, Any]] = []
            missing_mandatory: List[str] = []
            for req in dedup_list:
                field = req["field"]
                if field in config.collected_data:
                    collected_points.append({
                        "field": field,
                        "value": config.collected_data[field],
                        "data_type": req["data_type"],
                        "unit": req.get("unit"),
                        "satisfies_requirements": req["source_req_ids"],
                        "collected_at": datetime.utcnow().isoformat(),
                        "source": "provided",
                    })
                elif req["mandatory"]:
                    missing_mandatory.append(field)

            outputs["collected_points"] = collected_points
            outputs["collected_count"] = len(collected_points)
            outputs["missing_mandatory"] = missing_mandatory
            outputs["missing_mandatory_count"] = len(missing_mandatory)

            if missing_mandatory:
                warnings.append(
                    f"{len(missing_mandatory)} mandatory fields not yet collected: "
                    f"{', '.join(missing_mandatory[:5])}"
                    + ("..." if len(missing_mandatory) > 5 else "")
                )

            logger.info(
                "Deduplicated %d -> %d requirements, %d collected, %d missing mandatory",
                len(all_requirements), len(dedup_list),
                len(collected_points), len(missing_mandatory),
            )

            status = PhaseStatus.COMPLETED
            records = len(dedup_list)

        except Exception as exc:
            logger.error("Deduplicated collection failed: %s", exc, exc_info=True)
            errors.append(f"Deduplicated collection failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="deduplicated_collection",
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Validation
    # -------------------------------------------------------------------------

    def _phase_validation(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 3: Validate collected data against requirements.

        Checks data type, range, completeness, and format constraints
        according to the validation strictness level.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            collection_outputs = self._phase_outputs.get("deduplicated_collection", {})
            collected_points = collection_outputs.get("collected_points", [])
            dedup_reqs = collection_outputs.get("deduplicated_requirements", [])
            strictness = config.validation_strictness

            issues: List[Dict[str, Any]] = []
            validated_points: List[Dict[str, Any]] = []

            req_lookup = {r["field"]: r for r in dedup_reqs}

            for point in collected_points:
                field = point["field"]
                value = point["value"]
                req = req_lookup.get(field, {})
                point_issues = self._validate_data_point(field, value, req, strictness)
                issues.extend(point_issues)

                has_errors = any(
                    i["severity"] == ValidationSeverity.ERROR.value for i in point_issues
                )
                validated_points.append({
                    "field": field,
                    "value": value,
                    "valid": not has_errors,
                    "issue_count": len(point_issues),
                })

            error_count = sum(1 for i in issues if i["severity"] == ValidationSeverity.ERROR.value)
            warning_count = sum(1 for i in issues if i["severity"] == ValidationSeverity.WARNING.value)
            info_count = sum(1 for i in issues if i["severity"] == ValidationSeverity.INFO.value)

            outputs["validation_issues"] = issues
            outputs["validated_points"] = validated_points
            outputs["total_issues"] = len(issues)
            outputs["error_count"] = error_count
            outputs["warning_count"] = warning_count
            outputs["info_count"] = info_count
            outputs["validation_passed"] = error_count == 0
            outputs["strictness"] = strictness

            valid_count = sum(1 for v in validated_points if v["valid"])
            outputs["valid_points"] = valid_count
            outputs["invalid_points"] = len(validated_points) - valid_count

            if error_count > 0:
                warnings.append(f"{error_count} validation errors found")

            logger.info(
                "Validation complete: %d points, %d errors, %d warnings",
                len(validated_points), error_count, warning_count,
            )

            status = PhaseStatus.COMPLETED
            records = len(validated_points)

        except Exception as exc:
            logger.error("Validation failed: %s", exc, exc_info=True)
            errors.append(f"Validation failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="validation",
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

    def _validate_data_point(
        self,
        field: str,
        value: Any,
        req: Dict[str, Any],
        strictness: str,
    ) -> List[Dict[str, Any]]:
        """Validate a single data point against its requirement."""
        issues: List[Dict[str, Any]] = []
        data_type = req.get("data_type", "TEXT")
        req_ids = req.get("source_req_ids", [])

        if value is None:
            if req.get("mandatory", False):
                issues.append({
                    "field": field,
                    "issue_type": "missing_value",
                    "severity": ValidationSeverity.ERROR.value,
                    "message": f"Mandatory field '{field}' has no value",
                    "req_ids": req_ids,
                })
            return issues

        if data_type == "NUMERIC":
            if not isinstance(value, (int, float)):
                issues.append({
                    "field": field,
                    "issue_type": "type_mismatch",
                    "severity": ValidationSeverity.ERROR.value,
                    "message": f"Field '{field}' expected numeric, got {type(value).__name__}",
                    "req_ids": req_ids,
                })
            elif value < 0:
                issues.append({
                    "field": field,
                    "issue_type": "range_violation",
                    "severity": ValidationSeverity.WARNING.value if strictness == "lenient" else ValidationSeverity.ERROR.value,
                    "message": f"Field '{field}' has negative value: {value}",
                    "req_ids": req_ids,
                })
        elif data_type == "PERCENTAGE":
            if not isinstance(value, (int, float)):
                issues.append({
                    "field": field,
                    "issue_type": "type_mismatch",
                    "severity": ValidationSeverity.ERROR.value,
                    "message": f"Field '{field}' expected percentage, got {type(value).__name__}",
                    "req_ids": req_ids,
                })
            elif not (0.0 <= value <= 100.0):
                issues.append({
                    "field": field,
                    "issue_type": "range_violation",
                    "severity": ValidationSeverity.ERROR.value,
                    "message": f"Percentage '{field}' out of range [0,100]: {value}",
                    "req_ids": req_ids,
                })
        elif data_type == "CURRENCY":
            if not isinstance(value, (int, float)):
                issues.append({
                    "field": field,
                    "issue_type": "type_mismatch",
                    "severity": ValidationSeverity.ERROR.value,
                    "message": f"Field '{field}' expected currency value, got {type(value).__name__}",
                    "req_ids": req_ids,
                })
        elif data_type == "DATE":
            if isinstance(value, str):
                try:
                    datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    issues.append({
                        "field": field,
                        "issue_type": "format_error",
                        "severity": ValidationSeverity.ERROR.value,
                        "message": f"Field '{field}' has invalid date format: {value}",
                        "req_ids": req_ids,
                    })
            elif not isinstance(value, datetime):
                issues.append({
                    "field": field,
                    "issue_type": "type_mismatch",
                    "severity": ValidationSeverity.ERROR.value,
                    "message": f"Field '{field}' expected date, got {type(value).__name__}",
                    "req_ids": req_ids,
                })
        elif data_type == "TEXT":
            if not isinstance(value, str):
                issues.append({
                    "field": field,
                    "issue_type": "type_mismatch",
                    "severity": ValidationSeverity.WARNING.value,
                    "message": f"Field '{field}' expected text, got {type(value).__name__}",
                    "req_ids": req_ids,
                })
            elif strictness == "strict" and len(str(value).strip()) == 0:
                issues.append({
                    "field": field,
                    "issue_type": "empty_value",
                    "severity": ValidationSeverity.ERROR.value,
                    "message": f"Field '{field}' is empty string",
                    "req_ids": req_ids,
                })

        return issues

    # -------------------------------------------------------------------------
    # Phase 4: Distribution
    # -------------------------------------------------------------------------

    def _phase_distribution(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 4: Route validated data to appropriate pack data pipelines.

        Distributes each data point to the packs that require it based
        on the requirements mapping from Phase 1.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            validation_outputs = self._phase_outputs.get("validation", {})
            collection_outputs = self._phase_outputs.get("deduplicated_collection", {})
            collected_points = collection_outputs.get("collected_points", [])
            validated_points = validation_outputs.get("validated_points", [])
            validation_passed = validation_outputs.get("validation_passed", False)

            if not config.auto_distribute and not validation_passed:
                warnings.append("Auto-distribute disabled and validation has errors; skipping distribution")
                outputs["distribution_skipped"] = True
                outputs["reason"] = "validation_errors_and_auto_distribute_off"
                return PhaseResult(
                    phase_name="distribution",
                    status=PhaseStatus.COMPLETED,
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                    duration_seconds=(datetime.utcnow() - started_at).total_seconds(),
                    outputs=outputs,
                    errors=errors,
                    warnings=warnings,
                    provenance_hash=_hash_data(outputs),
                )

            valid_fields = {
                v["field"] for v in validated_points if v.get("valid", False)
            }

            pack_distributions: Dict[str, Dict[str, Any]] = {}
            for pack in config.target_packs:
                pack_reqs = REGULATION_REQUIREMENTS.get(pack.value, [])
                pack_data: Dict[str, Any] = {}
                missing_fields: List[str] = []

                for req in pack_reqs:
                    field = req["field"]
                    if field in valid_fields and field in config.collected_data:
                        pack_data[field] = config.collected_data[field]
                    elif req["mandatory"]:
                        missing_fields.append(field)

                delivery_status = DistributionStatus.DELIVERED.value
                if missing_fields:
                    if len(missing_fields) == len([r for r in pack_reqs if r["mandatory"]]):
                        delivery_status = DistributionStatus.FAILED.value
                    else:
                        delivery_status = DistributionStatus.DELIVERED.value
                        warnings.append(
                            f"{pack.value}: {len(missing_fields)} mandatory fields missing"
                        )

                pack_distributions[pack.value] = {
                    "pack": pack.value,
                    "data_points_sent": len(pack_data),
                    "total_required": len(pack_reqs),
                    "missing_mandatory": missing_fields,
                    "delivery_status": delivery_status,
                    "distributed_at": datetime.utcnow().isoformat(),
                    "data_payload": pack_data,
                }

            outputs["pack_distributions"] = pack_distributions
            outputs["packs_distributed"] = sum(
                1 for d in pack_distributions.values()
                if d["delivery_status"] == DistributionStatus.DELIVERED.value
            )
            outputs["packs_failed"] = sum(
                1 for d in pack_distributions.values()
                if d["delivery_status"] == DistributionStatus.FAILED.value
            )
            outputs["total_data_points_routed"] = sum(
                d["data_points_sent"] for d in pack_distributions.values()
            )

            logger.info(
                "Distribution complete: %d packs, %d total data points routed",
                len(pack_distributions), outputs["total_data_points_routed"],
            )

            status = PhaseStatus.COMPLETED
            records = len(pack_distributions)

        except Exception as exc:
            logger.error("Distribution failed: %s", exc, exc_info=True)
            errors.append(f"Distribution failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="distribution",
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

    # -------------------------------------------------------------------------
    # Phase 5: Confirmation
    # -------------------------------------------------------------------------

    def _phase_confirmation(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 5: Confirm receipt by all constituent packs.

        Simulates acknowledgment from each pack pipeline that data
        has been received and integrated into pack-level stores.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            distribution_outputs = self._phase_outputs.get("distribution", {})
            pack_distributions = distribution_outputs.get("pack_distributions", {})

            if distribution_outputs.get("distribution_skipped", False):
                outputs["confirmation_skipped"] = True
                outputs["reason"] = "distribution_was_skipped"
                return PhaseResult(
                    phase_name="confirmation",
                    status=PhaseStatus.COMPLETED,
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                    duration_seconds=(datetime.utcnow() - started_at).total_seconds(),
                    outputs=outputs,
                    errors=errors,
                    warnings=warnings,
                    provenance_hash=_hash_data(outputs),
                )

            confirmations: Dict[str, Dict[str, Any]] = {}
            for pack_name, dist_info in pack_distributions.items():
                delivery_status = dist_info.get("delivery_status", "")

                if delivery_status == DistributionStatus.DELIVERED.value:
                    confirmation_status = "ACKNOWLEDGED"
                    data_points_integrated = dist_info.get("data_points_sent", 0)
                elif delivery_status == DistributionStatus.FAILED.value:
                    confirmation_status = "NOT_RECEIVED"
                    data_points_integrated = 0
                    warnings.append(f"{pack_name}: distribution failed, no confirmation")
                else:
                    confirmation_status = "PENDING"
                    data_points_integrated = 0

                confirmations[pack_name] = {
                    "pack": pack_name,
                    "confirmation_id": str(uuid.uuid4()),
                    "confirmation_status": confirmation_status,
                    "data_points_integrated": data_points_integrated,
                    "confirmed_at": datetime.utcnow().isoformat(),
                    "integration_notes": (
                        f"All {data_points_integrated} data points integrated into {pack_name} pipeline"
                        if confirmation_status == "ACKNOWLEDGED"
                        else f"No data integrated for {pack_name}"
                    ),
                }

            outputs["confirmations"] = confirmations
            outputs["packs_confirmed"] = sum(
                1 for c in confirmations.values()
                if c["confirmation_status"] == "ACKNOWLEDGED"
            )
            outputs["packs_pending"] = sum(
                1 for c in confirmations.values()
                if c["confirmation_status"] == "PENDING"
            )
            outputs["packs_not_received"] = sum(
                1 for c in confirmations.values()
                if c["confirmation_status"] == "NOT_RECEIVED"
            )

            collection_receipt = {
                "receipt_id": str(uuid.uuid4()),
                "workflow_id": self.workflow_id,
                "organization_id": config.organization_id,
                "reporting_year": config.reporting_year,
                "timestamp": datetime.utcnow().isoformat(),
                "total_packs": len(confirmations),
                "confirmed_packs": outputs["packs_confirmed"],
                "status": (
                    "COMPLETE" if outputs["packs_confirmed"] == len(confirmations)
                    else "PARTIAL"
                ),
            }
            outputs["collection_receipt"] = collection_receipt

            logger.info(
                "Confirmation complete: %d/%d packs confirmed",
                outputs["packs_confirmed"], len(confirmations),
            )

            status = PhaseStatus.COMPLETED
            records = len(confirmations)

        except Exception as exc:
            logger.error("Confirmation failed: %s", exc, exc_info=True)
            errors.append(f"Confirmation failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="confirmation",
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def _build_summary(self) -> Dict[str, Any]:
        """Build workflow summary from all phase outputs."""
        mapping = self._phase_outputs.get("requirements_mapping", {})
        collection = self._phase_outputs.get("deduplicated_collection", {})
        validation = self._phase_outputs.get("validation", {})
        distribution = self._phase_outputs.get("distribution", {})
        confirmation = self._phase_outputs.get("confirmation", {})

        return {
            "total_requirements": mapping.get("total_requirements", 0),
            "deduplicated_requirements": collection.get("deduplicated_count", 0),
            "deduplication_savings": collection.get("deduplication_savings", 0),
            "data_points_collected": collection.get("collected_count", 0),
            "missing_mandatory": collection.get("missing_mandatory_count", 0),
            "validation_errors": validation.get("error_count", 0),
            "validation_warnings": validation.get("warning_count", 0),
            "validation_passed": validation.get("validation_passed", False),
            "packs_distributed": distribution.get("packs_distributed", 0),
            "total_data_points_routed": distribution.get("total_data_points_routed", 0),
            "packs_confirmed": confirmation.get("packs_confirmed", 0),
        }


# =============================================================================
# UTILITIES
# =============================================================================


def _hash_data(data: Any) -> str:
    """Compute SHA-256 provenance hash of arbitrary data."""
    serialized = str(data).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
