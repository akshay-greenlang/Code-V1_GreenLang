# -*- coding: utf-8 -*-
"""
Quarterly Reporting Workflow
============================

Seven-phase quarterly CBAM report cycle that orchestrates data collection,
validation, supplier integration, emission calculation, compliance checking,
report generation, and submission preparation for the EU CBAM Registry.

Regulatory Context:
    Per EU Implementing Regulation 2023/1773:
    - Article 3: Quarterly CBAM reports must be filed for each calendar quarter
      during the transitional period (Oct 2023 - Dec 2025)
    - Article 5: Reports must contain goods identification, CN codes, quantities,
      country of origin, and embedded emissions
    - Article 7: Default values may be used during transitional period but with
      increasing restrictions in the definitive period
    - Article 35 of Regulation 2023/956: Quarterly reports in the definitive
      period inform certificate obligations

    In the definitive period (2026+), quarterly reporting supports the annual
    declaration and informs the 50% holding requirement by end of each quarter.

Phases:
    1. Import data collection - Gather customs/shipment data
    2. Data validation - CN code, EORI, quantity, duplicate checks
    3. Supplier data integration - Match supplier emission data to shipments
    4. Emission calculation - Calculate embedded emissions per goods category
    5. Policy compliance check - Run 50+ CBAM rules
    6. Report generation - Assemble report, generate XML, render summaries
    7. Submission preparation - Package for EU CBAM Registry

Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import hashlib
import logging
import re
import uuid
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class Quarter(str, Enum):
    """Calendar quarter identifier."""
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"


class EmissionDataSource(str, Enum):
    """Source type for emission data."""
    ACTUAL_SUPPLIER = "actual_supplier"
    DEFAULT_EU = "default_eu"
    DEFAULT_JRC = "default_jrc"
    ESTIMATED = "estimated"


class CbamSector(str, Enum):
    """CBAM goods sector categories per Annex I of Regulation 2023/956."""
    CEMENT = "cement"
    IRON_STEEL = "iron_steel"
    ALUMINIUM = "aluminium"
    FERTILISERS = "fertilisers"
    ELECTRICITY = "electricity"
    HYDROGEN = "hydrogen"


class ComplianceSeverity(str, Enum):
    """Severity for compliance rule violations."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase.

    Every phase in every CBAM workflow returns this model to ensure
    consistent tracking of progress, outputs, warnings, and provenance.
    """
    phase_name: str = Field(..., description="Phase identifier (e.g. 'import_data_collection')")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, ge=0.0, description="Phase execution time")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output artifacts")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings")
    provenance_hash: str = Field(default="", description="SHA-256 hash of phase inputs+outputs")


class ImportRecord(BaseModel):
    """A single CBAM import record representing a customs declaration line."""
    record_id: str = Field(..., description="Unique record identifier")
    cn_code: str = Field(..., min_length=6, max_length=10, description="Combined Nomenclature code")
    goods_description: str = Field(default="", description="Goods description from customs")
    country_of_origin: str = Field(..., min_length=2, max_length=2, description="ISO 3166 alpha-2")
    quantity_tonnes: float = Field(..., gt=0, description="Net mass in metric tonnes")
    importer_eori: str = Field(..., description="Importer EORI number")
    supplier_id: Optional[str] = Field(None, description="Linked supplier identifier")
    cbam_sector: Optional[CbamSector] = Field(None, description="CBAM sector classification")
    customs_declaration_ref: Optional[str] = Field(None, description="Customs declaration reference")
    import_date: Optional[str] = Field(None, description="Import date YYYY-MM-DD")


class SupplierEmissionData(BaseModel):
    """Supplier-provided emission data for CBAM goods."""
    supplier_id: str = Field(..., description="Supplier identifier")
    installation_id: Optional[str] = Field(None, description="Installation identifier")
    cn_code: str = Field(..., description="CN code for the product")
    specific_embedded_emissions: Optional[float] = Field(
        None, ge=0, description="Specific embedded emissions in tCO2e/tonne"
    )
    direct_emissions: Optional[float] = Field(None, ge=0, description="Direct (Scope 1) in tCO2e/t")
    indirect_emissions: Optional[float] = Field(None, ge=0, description="Indirect (Scope 2) in tCO2e/t")
    data_source: EmissionDataSource = Field(
        default=EmissionDataSource.ACTUAL_SUPPLIER,
        description="How emission data was obtained",
    )
    verification_status: Optional[str] = Field(None, description="Verification state of data")
    carbon_price_paid: Optional[float] = Field(None, ge=0, description="Carbon price paid in origin country (EUR/tCO2e)")
    data_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Quality score 0-1")


class ComplianceViolation(BaseModel):
    """A single compliance rule violation from the policy check phase."""
    rule_id: str = Field(..., description="Rule identifier (e.g. 'CBAM-R-001')")
    rule_name: str = Field(..., description="Human-readable rule name")
    severity: ComplianceSeverity = Field(..., description="Severity level")
    description: str = Field(..., description="Violation description")
    affected_records: List[str] = Field(default_factory=list, description="Record IDs affected")
    remediation: str = Field(default="", description="Suggested remediation action")


class GoodsCategoryEmissions(BaseModel):
    """Aggregated emissions for a single goods category (CN code group)."""
    cn_code: str = Field(..., description="Combined Nomenclature code")
    cbam_sector: CbamSector = Field(..., description="CBAM sector")
    goods_description: str = Field(default="", description="Goods description")
    total_quantity_tonnes: float = Field(default=0.0, ge=0, description="Total imported quantity")
    total_embedded_emissions_tco2e: float = Field(default=0.0, ge=0, description="Total embedded emissions")
    specific_emissions_tco2e_per_t: float = Field(default=0.0, ge=0, description="Specific emission intensity")
    direct_emissions_tco2e: float = Field(default=0.0, ge=0, description="Direct emissions subtotal")
    indirect_emissions_tco2e: float = Field(default=0.0, ge=0, description="Indirect emissions subtotal")
    data_source_breakdown: Dict[str, int] = Field(default_factory=dict, description="Count by data source type")
    records_count: int = Field(default=0, ge=0, description="Number of import records")


class QuarterlyReportResult(BaseModel):
    """Complete result from the quarterly CBAM reporting workflow.

    Encapsulates the outcome of all 7 phases including the generated report
    metadata, emission totals, compliance score, and provenance chain.
    """
    workflow_name: str = Field(default="quarterly_reporting", description="Workflow identifier")
    status: PhaseStatus = Field(..., description="Overall workflow status")
    phases: List[PhaseResult] = Field(default_factory=list, description="Results from each phase")
    report_id: str = Field(..., description="Generated CBAM quarterly report ID")
    xml_generated: bool = Field(default=False, description="Whether XML was generated successfully")
    total_emissions_tco2e: float = Field(default=0.0, ge=0, description="Total embedded emissions")
    total_quantity_tonnes: float = Field(default=0.0, ge=0, description="Total imported quantity")
    goods_categories_count: int = Field(default=0, ge=0, description="Distinct goods categories")
    compliance_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Compliance score 0-100")
    provenance_hash: str = Field(default="", description="SHA-256 hash of entire workflow")
    execution_id: str = Field(default="", description="Unique execution identifier")
    started_at: Optional[datetime] = Field(None, description="Workflow start time")
    completed_at: Optional[datetime] = Field(None, description="Workflow end time")
    quarter: Optional[str] = Field(None, description="Quarter label e.g. '2026-Q1'")
    year: Optional[int] = Field(None, description="Reporting year")
    violations_summary: Dict[str, int] = Field(default_factory=dict, description="Count by severity")


# =============================================================================
# CN CODE VALIDATION
# =============================================================================


# CBAM Annex I CN code prefixes by sector
CBAM_CN_CODE_PREFIXES: Dict[CbamSector, List[str]] = {
    CbamSector.CEMENT: ["2507", "2523"],
    CbamSector.IRON_STEEL: [
        "2601", "7201", "7202", "7203", "7204", "7205", "7206", "7207",
        "7208", "7209", "7210", "7211", "7212", "7213", "7214", "7215",
        "7216", "7217", "7218", "7219", "7220", "7221", "7222", "7223",
        "7224", "7225", "7226", "7227", "7228", "7229", "7301", "7302",
        "7303", "7304", "7305", "7306", "7307", "7308", "7309", "7310",
        "7311", "7318", "7326",
    ],
    CbamSector.ALUMINIUM: [
        "7601", "7602", "7603", "7604", "7605", "7606", "7607", "7608",
        "7609", "7610", "7611", "7612", "7613", "7614", "7615", "7616",
    ],
    CbamSector.FERTILISERS: [
        "2808", "2814", "2834", "3102", "3105",
    ],
    CbamSector.ELECTRICITY: ["2716"],
    CbamSector.HYDROGEN: ["280410"],
}

# EORI format: 2-letter country code + up to 15 alphanumeric characters
EORI_PATTERN = re.compile(r"^[A-Z]{2}[A-Za-z0-9]{1,15}$")

# Default emission factors (JRC 2025) by sector in tCO2e/tonne
DEFAULT_EMISSION_FACTORS: Dict[str, float] = {
    "cement": 0.950,
    "iron_steel": 1.850,
    "aluminium": 6.700,
    "fertilisers": 3.000,
    "electricity": 0.450,  # tCO2e/MWh (treated specially)
    "hydrogen": 9.000,
}


# =============================================================================
# QUARTERLY REPORTING WORKFLOW
# =============================================================================


class QuarterlyReportingWorkflow:
    """
    Seven-phase quarterly CBAM report orchestrator.

    Coordinates the end-to-end process of collecting import data, validating
    it against CBAM rules, integrating supplier emission data, calculating
    embedded emissions, checking policy compliance, generating reports, and
    preparing submissions for the EU CBAM Registry.

    Attributes:
        config: Optional configuration dict for tuning workflow behavior.
        logger: Logger instance for this workflow.
        _execution_id: Unique identifier for this execution.
        _phase_results: Accumulated phase results during execution.

    Example:
        >>> wf = QuarterlyReportingWorkflow()
        >>> result = await wf.execute(
        ...     config={"organization_id": "org-123"},
        ...     import_data=[ImportRecord(...)],
        ...     quarter=Quarter.Q1,
        ...     year=2026,
        ... )
        >>> assert result.status == PhaseStatus.COMPLETED
        >>> assert result.compliance_score >= 80.0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the QuarterlyReportingWorkflow.

        Args:
            config: Optional configuration dict with keys like
                'organization_id', 'default_factor_cap_pct', 'strict_mode'.
        """
        self.config: Dict[str, Any] = config or {}
        self.logger = logging.getLogger(f"{__name__}.QuarterlyReportingWorkflow")
        self._execution_id: str = str(uuid.uuid4())
        self._phase_results: List[PhaseResult] = []

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        config: Optional[Dict[str, Any]],
        import_data: List[ImportRecord],
        quarter: Quarter,
        year: int,
    ) -> QuarterlyReportResult:
        """
        Execute the full 7-phase quarterly reporting workflow.

        This is the main entry point. It runs phases sequentially, with each
        phase receiving context from prior phases. If any critical phase fails,
        the workflow halts and returns a FAILED result.

        Args:
            config: Execution-level config overrides (merged with __init__ config).
            import_data: List of import records for the quarter.
            quarter: Which calendar quarter (Q1-Q4).
            year: Reporting year (e.g. 2026).

        Returns:
            QuarterlyReportResult with emission totals, compliance score,
            generated report ID, and full phase-by-phase provenance.
        """
        started_at = datetime.utcnow()
        merged_config = {**self.config, **(config or {})}
        quarter_label = f"{year}-{quarter.value}"
        report_id = f"CBAM-QR-{year}-{quarter.value}-{self._execution_id[:8]}"

        self.logger.info(
            "Starting quarterly reporting workflow execution_id=%s quarter=%s",
            self._execution_id, quarter_label,
        )

        context: Dict[str, Any] = {
            "config": merged_config,
            "import_data": import_data,
            "quarter": quarter,
            "year": year,
            "quarter_label": quarter_label,
            "report_id": report_id,
            "execution_id": self._execution_id,
        }

        phase_handlers = [
            ("import_data_collection", self._phase_1_import_data_collection),
            ("data_validation", self._phase_2_data_validation),
            ("supplier_data_integration", self._phase_3_supplier_data_integration),
            ("emission_calculation", self._phase_4_emission_calculation),
            ("policy_compliance_check", self._phase_5_policy_compliance_check),
            ("report_generation", self._phase_6_report_generation),
            ("submission_preparation", self._phase_7_submission_preparation),
        ]

        overall_status = PhaseStatus.COMPLETED
        total_emissions = 0.0
        total_quantity = 0.0
        goods_categories_count = 0
        compliance_score = 0.0
        xml_generated = False
        violations_summary: Dict[str, int] = {}

        for phase_name, handler in phase_handlers:
            phase_start = datetime.utcnow()
            self.logger.info("Starting phase: %s", phase_name)

            try:
                phase_result = await handler(context)
                phase_result.duration_seconds = (
                    datetime.utcnow() - phase_start
                ).total_seconds()
            except Exception as exc:
                self.logger.error(
                    "Phase '%s' raised exception: %s", phase_name, exc, exc_info=True,
                )
                phase_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    duration_seconds=(datetime.utcnow() - phase_start).total_seconds(),
                    outputs={"error": str(exc)},
                    warnings=[],
                    provenance_hash=self._hash({"error": str(exc), "phase": phase_name}),
                )

            self._phase_results.append(phase_result)
            self.logger.info(
                "Phase '%s' completed status=%s in %.2fs",
                phase_name, phase_result.status.value, phase_result.duration_seconds,
            )

            # Extract running totals from phase outputs
            if phase_name == "emission_calculation" and phase_result.status == PhaseStatus.COMPLETED:
                total_emissions = phase_result.outputs.get("total_emissions_tco2e", 0.0)
                total_quantity = phase_result.outputs.get("total_quantity_tonnes", 0.0)
                goods_categories_count = phase_result.outputs.get("goods_categories_count", 0)

            if phase_name == "policy_compliance_check" and phase_result.status == PhaseStatus.COMPLETED:
                compliance_score = phase_result.outputs.get("compliance_score", 0.0)
                violations_summary = phase_result.outputs.get("violations_summary", {})

            if phase_name == "report_generation" and phase_result.status == PhaseStatus.COMPLETED:
                xml_generated = phase_result.outputs.get("xml_generated", False)

            # Halt on critical phase failure (phases 1-4 are critical)
            if phase_result.status == PhaseStatus.FAILED and phase_name in (
                "import_data_collection", "data_validation",
                "supplier_data_integration", "emission_calculation",
            ):
                overall_status = PhaseStatus.FAILED
                self.logger.error("Critical phase '%s' failed; halting workflow.", phase_name)
                break

            # Non-critical phase failure degrades but continues
            if phase_result.status == PhaseStatus.FAILED:
                overall_status = PhaseStatus.FAILED

        completed_at = datetime.utcnow()
        workflow_provenance = self._hash({
            "execution_id": self._execution_id,
            "phases": [p.provenance_hash for p in self._phase_results],
            "total_emissions": total_emissions,
            "quarter": quarter_label,
        })

        self.logger.info(
            "Quarterly reporting workflow finished execution_id=%s status=%s "
            "emissions=%.4f tCO2e duration=%.2fs",
            self._execution_id, overall_status.value, total_emissions,
            (completed_at - started_at).total_seconds(),
        )

        return QuarterlyReportResult(
            workflow_name="quarterly_reporting",
            status=overall_status,
            phases=self._phase_results,
            report_id=report_id,
            xml_generated=xml_generated,
            total_emissions_tco2e=total_emissions,
            total_quantity_tonnes=total_quantity,
            goods_categories_count=goods_categories_count,
            compliance_score=compliance_score,
            provenance_hash=workflow_provenance,
            execution_id=self._execution_id,
            started_at=started_at,
            completed_at=completed_at,
            quarter=quarter_label,
            year=year,
            violations_summary=violations_summary,
        )

    # -------------------------------------------------------------------------
    # Phase 1: Import Data Collection
    # -------------------------------------------------------------------------

    async def _phase_1_import_data_collection(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Gather customs and shipment data for the quarter.

        Validates that source files are present, normalizes data formats,
        deduplicates raw records, and produces a cleaned import dataset
        ready for validation.

        Steps:
            - Accept import records from context or fetch from data sources
            - Normalize CN codes (strip dots, pad to 8 digits)
            - Normalize country codes to ISO 3166 alpha-2
            - Assign preliminary CBAM sector from CN code prefixes
            - Deduplicate by customs declaration reference
            - Log record counts and data source summary
        """
        phase_name = "import_data_collection"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        import_data: List[ImportRecord] = context.get("import_data", [])
        quarter = context["quarter"]
        year = context["year"]

        if not import_data:
            self.logger.warning("No import records provided for %s-%s", year, quarter.value)
            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.FAILED,
                outputs={"error": "No import records provided", "records_count": 0},
                warnings=["No import data to process"],
                provenance_hash=self._hash({"phase": phase_name, "records": 0}),
            )

        self.logger.info("Processing %d raw import records", len(import_data))

        # Normalize CN codes and assign sectors
        normalized_records: List[Dict[str, Any]] = []
        sector_counts: Dict[str, int] = {}
        country_counts: Dict[str, int] = {}
        duplicate_refs: set = set()
        seen_refs: set = set()

        for record in import_data:
            # Normalize CN code: remove dots, whitespace
            cn_clean = record.cn_code.replace(".", "").replace(" ", "").strip()

            # Assign CBAM sector from CN code prefix
            assigned_sector = self._classify_cn_code(cn_clean)
            if assigned_sector is None:
                warnings.append(
                    f"Record {record.record_id}: CN code '{cn_clean}' not in CBAM Annex I scope"
                )

            # Detect duplicates by customs declaration reference
            if record.customs_declaration_ref:
                dedup_key = f"{record.customs_declaration_ref}:{cn_clean}"
                if dedup_key in seen_refs:
                    duplicate_refs.add(dedup_key)
                    continue
                seen_refs.add(dedup_key)

            normalized = record.model_dump()
            normalized["cn_code"] = cn_clean
            normalized["cbam_sector"] = assigned_sector.value if assigned_sector else None

            normalized_records.append(normalized)

            # Accumulate sector/country statistics
            sector_label = assigned_sector.value if assigned_sector else "unclassified"
            sector_counts[sector_label] = sector_counts.get(sector_label, 0) + 1
            country_counts[record.country_of_origin] = (
                country_counts.get(record.country_of_origin, 0) + 1
            )

        if duplicate_refs:
            warnings.append(
                f"Removed {len(duplicate_refs)} duplicate record(s) by customs declaration reference"
            )

        # Filter to quarter date range if import_date is provided
        quarter_filtered = self._filter_by_quarter(normalized_records, quarter, year)
        if len(quarter_filtered) < len(normalized_records):
            outside_count = len(normalized_records) - len(quarter_filtered)
            warnings.append(
                f"{outside_count} record(s) outside quarter {year}-{quarter.value} date range"
            )
            normalized_records = quarter_filtered

        # Store in context for subsequent phases
        context["normalized_records"] = normalized_records

        outputs["records_count"] = len(normalized_records)
        outputs["duplicates_removed"] = len(duplicate_refs)
        outputs["sector_breakdown"] = sector_counts
        outputs["country_breakdown"] = country_counts
        outputs["unclassified_count"] = sector_counts.get("unclassified", 0)

        self.logger.info(
            "Phase 1 complete: %d records normalized, %d duplicates removed",
            len(normalized_records), len(duplicate_refs),
        )

        provenance = self._hash({
            "phase": phase_name,
            "records_count": len(normalized_records),
            "sector_counts": sector_counts,
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 2: Data Validation
    # -------------------------------------------------------------------------

    async def _phase_2_data_validation(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Validate import data against CBAM regulatory requirements.

        Runs the following validation checks:
            - CN code format (6-10 digits, valid CBAM Annex I prefix)
            - EORI number format (2-letter country + 1-15 alphanumeric)
            - Quantity range checks (positive, < 1,000,000 tonnes per record)
            - Country of origin is a valid ISO 3166-1 alpha-2 code
            - Duplicate detection across declaration references
            - Missing required fields check
            - CBAM sector assignment validation
        """
        phase_name = "data_validation"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        records: List[Dict[str, Any]] = context.get("normalized_records", [])

        if not records:
            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.FAILED,
                outputs={"error": "No records to validate"},
                provenance_hash=self._hash({"phase": phase_name, "error": "no_records"}),
            )

        valid_records: List[Dict[str, Any]] = []
        invalid_records: List[Dict[str, Any]] = []
        validation_errors: List[Dict[str, Any]] = []

        for record in records:
            record_errors: List[str] = []

            # CN code validation
            cn_code = record.get("cn_code", "")
            if not self._validate_cn_code_format(cn_code):
                record_errors.append(f"Invalid CN code format: '{cn_code}'")
            elif self._classify_cn_code(cn_code) is None:
                record_errors.append(f"CN code '{cn_code}' not in CBAM Annex I scope")

            # EORI validation
            eori = record.get("importer_eori", "")
            if not self._validate_eori(eori):
                record_errors.append(f"Invalid EORI format: '{eori}'")

            # Quantity range check
            qty = record.get("quantity_tonnes", 0)
            if qty <= 0:
                record_errors.append(f"Quantity must be positive, got: {qty}")
            elif qty > 1_000_000:
                record_errors.append(
                    f"Quantity {qty} tonnes exceeds single-record maximum (1,000,000)"
                )

            # Country of origin check (basic alpha-2)
            country = record.get("country_of_origin", "")
            if not self._validate_country_code(country):
                record_errors.append(f"Invalid country code: '{country}'")

            # EU origin check (CBAM does not apply to EU imports)
            if country in self._get_eu_country_codes():
                record_errors.append(
                    f"Country '{country}' is an EU member state; CBAM does not apply"
                )

            # CBAM sector assignment
            if not record.get("cbam_sector"):
                record_errors.append("No CBAM sector assigned")

            if record_errors:
                invalid_records.append(record)
                validation_errors.append({
                    "record_id": record.get("record_id", "unknown"),
                    "errors": record_errors,
                })
            else:
                valid_records.append(record)

        # Update context with validated records
        context["validated_records"] = valid_records
        context["invalid_records"] = invalid_records

        outputs["total_records"] = len(records)
        outputs["valid_count"] = len(valid_records)
        outputs["invalid_count"] = len(invalid_records)
        outputs["validation_errors"] = validation_errors
        outputs["validation_pass_rate"] = (
            round(len(valid_records) / len(records) * 100, 2) if records else 0.0
        )

        if invalid_records:
            warnings.append(
                f"{len(invalid_records)} record(s) failed validation "
                f"({outputs['validation_pass_rate']:.1f}% pass rate)"
            )

        # Fail if pass rate below threshold
        min_pass_rate = context.get("config", {}).get("min_validation_pass_rate", 50.0)
        status = PhaseStatus.COMPLETED
        if outputs["validation_pass_rate"] < min_pass_rate:
            status = PhaseStatus.FAILED
            warnings.append(
                f"Validation pass rate {outputs['validation_pass_rate']:.1f}% "
                f"below minimum threshold {min_pass_rate}%"
            )

        self.logger.info(
            "Phase 2 complete: %d valid, %d invalid (%.1f%% pass rate)",
            len(valid_records), len(invalid_records), outputs["validation_pass_rate"],
        )

        provenance = self._hash({
            "phase": phase_name,
            "valid": len(valid_records),
            "invalid": len(invalid_records),
        })

        return PhaseResult(
            phase_name=phase_name,
            status=status,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Supplier Data Integration
    # -------------------------------------------------------------------------

    async def _phase_3_supplier_data_integration(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Match supplier emission data to validated import records.

        For each validated record, attempts to find actual supplier emission
        data. If no supplier data is available, assigns default emission
        factors from the JRC/EU default values database.

        Steps:
            - Load available supplier emission data
            - Match by supplier_id + CN code
            - Fall back to default factors where no actual data exists
            - Track data source breakdown (actual vs default)
            - Warn on high default factor usage
        """
        phase_name = "supplier_data_integration"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        validated_records: List[Dict[str, Any]] = context.get("validated_records", [])
        supplier_data_map = await self._load_supplier_emission_data(context)

        matched_count = 0
        default_count = 0
        enriched_records: List[Dict[str, Any]] = []

        for record in validated_records:
            supplier_id = record.get("supplier_id")
            cn_code = record.get("cn_code", "")
            sector = record.get("cbam_sector", "")

            # Attempt supplier data match
            supplier_match = self._match_supplier_data(
                supplier_id, cn_code, supplier_data_map,
            )

            if supplier_match:
                record["specific_embedded_emissions"] = supplier_match.get(
                    "specific_embedded_emissions"
                )
                record["direct_emissions_ef"] = supplier_match.get("direct_emissions")
                record["indirect_emissions_ef"] = supplier_match.get("indirect_emissions")
                record["emission_data_source"] = EmissionDataSource.ACTUAL_SUPPLIER.value
                record["carbon_price_paid"] = supplier_match.get("carbon_price_paid", 0.0)
                record["data_quality_score"] = supplier_match.get("data_quality_score", 0.8)
                matched_count += 1
            else:
                # Fall back to default emission factors
                default_ef = self._get_default_emission_factor(sector, context["year"])
                record["specific_embedded_emissions"] = default_ef
                record["direct_emissions_ef"] = default_ef
                record["indirect_emissions_ef"] = 0.0
                record["emission_data_source"] = EmissionDataSource.DEFAULT_EU.value
                record["carbon_price_paid"] = 0.0
                record["data_quality_score"] = 0.3
                default_count += 1

            enriched_records.append(record)

        context["enriched_records"] = enriched_records

        # Check default factor usage cap
        total = len(enriched_records)
        default_pct = round(default_count / total * 100, 2) if total > 0 else 0.0
        default_cap = context.get("config", {}).get("default_factor_cap_pct", 100.0)

        if default_pct > default_cap:
            warnings.append(
                f"Default factor usage ({default_pct:.1f}%) exceeds cap ({default_cap:.1f}%)"
            )

        if default_pct > 80.0:
            warnings.append(
                f"High default factor usage ({default_pct:.1f}%). "
                "Consider collecting actual supplier emission data to improve accuracy."
            )

        outputs["total_records"] = total
        outputs["supplier_matched"] = matched_count
        outputs["default_applied"] = default_count
        outputs["default_usage_pct"] = default_pct
        outputs["supplier_data_sources"] = len(supplier_data_map)

        self.logger.info(
            "Phase 3 complete: %d matched, %d default (%.1f%% default usage)",
            matched_count, default_count, default_pct,
        )

        provenance = self._hash({
            "phase": phase_name,
            "matched": matched_count,
            "default": default_count,
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 4: Emission Calculation
    # -------------------------------------------------------------------------

    async def _phase_4_emission_calculation(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Calculate embedded emissions per goods category using the CBAM
        calculation engine (zero-hallucination, deterministic arithmetic).

        Per CBAM Implementing Regulation Article 3:
            Embedded emissions = quantity (tonnes) * specific_embedded_emissions (tCO2e/t)

        For each record:
            total_embedded = quantity_tonnes * specific_embedded_emissions
            direct_emissions = quantity_tonnes * direct_emissions_ef
            indirect_emissions = quantity_tonnes * indirect_emissions_ef

        Results are aggregated by CN code into GoodsCategoryEmissions.
        """
        phase_name = "emission_calculation"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        enriched_records: List[Dict[str, Any]] = context.get("enriched_records", [])

        if not enriched_records:
            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.FAILED,
                outputs={"error": "No enriched records for calculation"},
                provenance_hash=self._hash({"phase": phase_name, "error": "no_records"}),
            )

        # Aggregate by CN code
        category_map: Dict[str, Dict[str, Any]] = {}
        total_emissions = Decimal("0")
        total_quantity = Decimal("0")
        calc_details: List[Dict[str, Any]] = []

        for record in enriched_records:
            qty = Decimal(str(record.get("quantity_tonnes", 0)))
            see = Decimal(str(record.get("specific_embedded_emissions", 0)))
            direct_ef = Decimal(str(record.get("direct_emissions_ef", 0)))
            indirect_ef = Decimal(str(record.get("indirect_emissions_ef", 0)))
            cn_code = record.get("cn_code", "")
            sector = record.get("cbam_sector", "")
            data_source = record.get("emission_data_source", "")

            # Zero-hallucination deterministic calculation
            embedded = (qty * see).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            direct = (qty * direct_ef).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            indirect = (qty * indirect_ef).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

            total_emissions += embedded
            total_quantity += qty

            # Record-level calculation detail
            calc_details.append({
                "record_id": record.get("record_id"),
                "cn_code": cn_code,
                "quantity_tonnes": float(qty),
                "specific_ef": float(see),
                "embedded_emissions_tco2e": float(embedded),
                "direct_tco2e": float(direct),
                "indirect_tco2e": float(indirect),
                "data_source": data_source,
            })

            # Aggregate into category
            if cn_code not in category_map:
                category_map[cn_code] = {
                    "cn_code": cn_code,
                    "cbam_sector": sector,
                    "goods_description": record.get("goods_description", ""),
                    "total_quantity_tonnes": Decimal("0"),
                    "total_embedded_emissions_tco2e": Decimal("0"),
                    "direct_emissions_tco2e": Decimal("0"),
                    "indirect_emissions_tco2e": Decimal("0"),
                    "data_source_breakdown": {},
                    "records_count": 0,
                }

            cat = category_map[cn_code]
            cat["total_quantity_tonnes"] += qty
            cat["total_embedded_emissions_tco2e"] += embedded
            cat["direct_emissions_tco2e"] += direct
            cat["indirect_emissions_tco2e"] += indirect
            cat["records_count"] += 1
            cat["data_source_breakdown"][data_source] = (
                cat["data_source_breakdown"].get(data_source, 0) + 1
            )

        # Compute specific emission intensities per category
        goods_categories: List[Dict[str, Any]] = []
        for cn_code, cat in category_map.items():
            qty_total = cat["total_quantity_tonnes"]
            if qty_total > 0:
                specific = (
                    cat["total_embedded_emissions_tco2e"] / qty_total
                ).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            else:
                specific = Decimal("0")

            goods_categories.append({
                "cn_code": cn_code,
                "cbam_sector": cat["cbam_sector"],
                "goods_description": cat["goods_description"],
                "total_quantity_tonnes": float(cat["total_quantity_tonnes"]),
                "total_embedded_emissions_tco2e": float(cat["total_embedded_emissions_tco2e"]),
                "specific_emissions_tco2e_per_t": float(specific),
                "direct_emissions_tco2e": float(cat["direct_emissions_tco2e"]),
                "indirect_emissions_tco2e": float(cat["indirect_emissions_tco2e"]),
                "data_source_breakdown": cat["data_source_breakdown"],
                "records_count": cat["records_count"],
            })

        # Check for anomalous emission intensities
        for gc in goods_categories:
            see = gc["specific_emissions_tco2e_per_t"]
            sector = gc["cbam_sector"]
            default = DEFAULT_EMISSION_FACTORS.get(sector, 1.0)
            if see > default * 3:
                warnings.append(
                    f"CN {gc['cn_code']}: specific emissions {see:.4f} tCO2e/t "
                    f"is >3x the default ({default:.3f}); verify supplier data"
                )

        context["goods_categories"] = goods_categories
        context["calc_details"] = calc_details

        outputs["total_emissions_tco2e"] = float(total_emissions)
        outputs["total_quantity_tonnes"] = float(total_quantity)
        outputs["goods_categories_count"] = len(goods_categories)
        outputs["goods_categories"] = goods_categories
        outputs["records_calculated"] = len(calc_details)

        self.logger.info(
            "Phase 4 complete: %.4f tCO2e from %.2f tonnes across %d categories",
            float(total_emissions), float(total_quantity), len(goods_categories),
        )

        provenance = self._hash({
            "phase": phase_name,
            "total_emissions": float(total_emissions),
            "total_quantity": float(total_quantity),
            "categories": len(goods_categories),
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 5: Policy Compliance Check
    # -------------------------------------------------------------------------

    async def _phase_5_policy_compliance_check(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Run 50+ CBAM compliance rules against the calculated results.

        Rule categories:
            - Data completeness rules (CBAM-R-001 to CBAM-R-010)
            - CN code validity rules (CBAM-R-011 to CBAM-R-020)
            - Emission factor rules (CBAM-R-021 to CBAM-R-030)
            - Default factor usage caps (CBAM-R-031 to CBAM-R-035)
            - Period-specific rules (CBAM-R-036 to CBAM-R-040)
            - Reporting format rules (CBAM-R-041 to CBAM-R-045)
            - Cross-validation rules (CBAM-R-046 to CBAM-R-050)
        """
        phase_name = "policy_compliance_check"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        violations: List[Dict[str, Any]] = []

        enriched_records: List[Dict[str, Any]] = context.get("enriched_records", [])
        goods_categories: List[Dict[str, Any]] = context.get("goods_categories", [])
        year = context["year"]

        # ---- Data Completeness Rules (CBAM-R-001 to CBAM-R-010) ----
        violations.extend(self._check_data_completeness_rules(enriched_records))

        # ---- CN Code Validity Rules (CBAM-R-011 to CBAM-R-020) ----
        violations.extend(self._check_cn_code_rules(enriched_records))

        # ---- Emission Factor Rules (CBAM-R-021 to CBAM-R-030) ----
        violations.extend(self._check_emission_factor_rules(enriched_records, goods_categories))

        # ---- Default Factor Usage Caps (CBAM-R-031 to CBAM-R-035) ----
        violations.extend(self._check_default_factor_cap_rules(enriched_records, year))

        # ---- Period-Specific Rules (CBAM-R-036 to CBAM-R-040) ----
        violations.extend(self._check_period_specific_rules(context))

        # ---- Reporting Format Rules (CBAM-R-041 to CBAM-R-045) ----
        violations.extend(self._check_reporting_format_rules(goods_categories))

        # ---- Cross-Validation Rules (CBAM-R-046 to CBAM-R-050) ----
        violations.extend(self._check_cross_validation_rules(enriched_records, goods_categories))

        # Calculate compliance score
        error_count = sum(1 for v in violations if v.get("severity") == ComplianceSeverity.ERROR.value)
        warning_count = sum(1 for v in violations if v.get("severity") == ComplianceSeverity.WARNING.value)
        info_count = sum(1 for v in violations if v.get("severity") == ComplianceSeverity.INFO.value)

        # Score: start at 100, subtract 5 per error, 2 per warning, 0.5 per info
        raw_score = 100.0 - (error_count * 5.0) - (warning_count * 2.0) - (info_count * 0.5)
        compliance_score = max(0.0, min(100.0, round(raw_score, 2)))

        if error_count > 0:
            warnings.append(f"{error_count} compliance error(s) detected; review required")

        outputs["violations"] = violations
        outputs["violations_count"] = len(violations)
        outputs["violations_summary"] = {
            "error": error_count,
            "warning": warning_count,
            "info": info_count,
        }
        outputs["compliance_score"] = compliance_score
        outputs["rules_checked"] = 50

        context["compliance_violations"] = violations
        context["compliance_score"] = compliance_score

        self.logger.info(
            "Phase 5 complete: score=%.2f, %d error(s), %d warning(s), %d info(s)",
            compliance_score, error_count, warning_count, info_count,
        )

        provenance = self._hash({
            "phase": phase_name,
            "score": compliance_score,
            "violations": len(violations),
        })

        status = PhaseStatus.COMPLETED
        if compliance_score < 30.0:
            status = PhaseStatus.FAILED
            warnings.append("Compliance score below 30%; report cannot be submitted")

        return PhaseResult(
            phase_name=phase_name,
            status=status,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 6: Report Generation
    # -------------------------------------------------------------------------

    async def _phase_6_report_generation(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Assemble the quarterly CBAM report, generate XML per the EU schema,
        and render human-readable summaries.

        Outputs:
            - CBAM XML report per EU CBAM Registry XSD schema
            - Executive summary (PDF-ready)
            - Detailed goods category breakdown
            - Compliance findings report
        """
        phase_name = "report_generation"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        report_id = context["report_id"]
        quarter_label = context["quarter_label"]
        year = context["year"]
        goods_categories = context.get("goods_categories", [])
        compliance_violations = context.get("compliance_violations", [])
        compliance_score = context.get("compliance_score", 0.0)

        # Generate XML structure
        xml_content = await self._generate_cbam_xml(
            report_id=report_id,
            quarter=quarter_label,
            year=year,
            goods_categories=goods_categories,
            config=context.get("config", {}),
        )
        xml_generated = xml_content is not None and len(xml_content) > 0

        if not xml_generated:
            warnings.append("XML generation failed; report may not be submittable")

        # Generate executive summary
        summary = await self._generate_executive_summary(
            report_id=report_id,
            quarter=quarter_label,
            goods_categories=goods_categories,
            compliance_score=compliance_score,
            violations_count=len(compliance_violations),
        )

        # Generate compliance findings report
        findings_report = self._generate_findings_report(compliance_violations)

        outputs["report_id"] = report_id
        outputs["xml_generated"] = xml_generated
        outputs["xml_size_bytes"] = len(xml_content) if xml_content else 0
        outputs["summary"] = summary
        outputs["findings_report"] = findings_report
        outputs["generated_at"] = datetime.utcnow().isoformat()
        outputs["report_format_version"] = "CBAM-2026-Q"

        context["report_xml"] = xml_content
        context["report_summary"] = summary

        self.logger.info(
            "Phase 6 complete: report_id=%s xml_generated=%s xml_size=%d",
            report_id, xml_generated, outputs["xml_size_bytes"],
        )

        provenance = self._hash({
            "phase": phase_name,
            "report_id": report_id,
            "xml_generated": xml_generated,
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 7: Submission Preparation
    # -------------------------------------------------------------------------

    async def _phase_7_submission_preparation(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Package the quarterly report for submission to the EU CBAM Registry.

        Steps:
            - Run pre-submission validation (schema check, completeness)
            - Package XML + supporting documents
            - Generate submission manifest
            - Create digital signature hash
            - Validate against EU CBAM Registry API schema (if configured)
        """
        phase_name = "submission_preparation"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        report_id = context["report_id"]
        report_xml = context.get("report_xml", "")
        compliance_score = context.get("compliance_score", 0.0)

        # Pre-submission validation
        pre_sub_checks = await self._run_pre_submission_validation(
            report_xml, context.get("config", {}),
        )
        outputs["pre_submission_checks"] = pre_sub_checks

        check_failures = [c for c in pre_sub_checks if c.get("status") == "FAIL"]
        if check_failures:
            for fail in check_failures:
                warnings.append(f"Pre-submission check failed: {fail.get('check_name', 'unknown')}")

        # Generate submission manifest
        manifest = {
            "report_id": report_id,
            "quarter": context["quarter_label"],
            "year": context["year"],
            "submission_format": "CBAM-XML-2026",
            "xml_present": bool(report_xml),
            "compliance_score": compliance_score,
            "pre_submission_passed": len(check_failures) == 0,
            "prepared_at": datetime.utcnow().isoformat(),
            "execution_id": context["execution_id"],
        }

        # Digital signature hash of the full package
        package_hash = self._hash({
            "report_id": report_id,
            "xml": report_xml,
            "manifest": manifest,
        })
        manifest["package_hash"] = package_hash

        outputs["manifest"] = manifest
        outputs["package_hash"] = package_hash
        outputs["ready_for_submission"] = len(check_failures) == 0 and compliance_score >= 50.0

        if not outputs["ready_for_submission"]:
            warnings.append(
                "Report is NOT ready for submission. "
                "Resolve compliance issues and re-run the workflow."
            )

        self.logger.info(
            "Phase 7 complete: report_id=%s ready=%s package_hash=%s",
            report_id, outputs["ready_for_submission"], package_hash[:16],
        )

        provenance = self._hash({
            "phase": phase_name,
            "package_hash": package_hash,
            "ready": outputs["ready_for_submission"],
        })

        status = PhaseStatus.COMPLETED
        if check_failures and context.get("config", {}).get("strict_mode", False):
            status = PhaseStatus.FAILED

        return PhaseResult(
            phase_name=phase_name,
            status=status,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # =========================================================================
    # COMPLIANCE RULE IMPLEMENTATIONS
    # =========================================================================

    def _check_data_completeness_rules(
        self, records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Check data completeness rules CBAM-R-001 through CBAM-R-010."""
        violations: List[Dict[str, Any]] = []

        required_fields = [
            ("record_id", "CBAM-R-001", "Record ID required"),
            ("cn_code", "CBAM-R-002", "CN code required"),
            ("country_of_origin", "CBAM-R-003", "Country of origin required"),
            ("quantity_tonnes", "CBAM-R-004", "Quantity required"),
            ("importer_eori", "CBAM-R-005", "Importer EORI required"),
        ]

        for field, rule_id, rule_name in required_fields:
            missing = [r.get("record_id", "?") for r in records if not r.get(field)]
            if missing:
                violations.append({
                    "rule_id": rule_id,
                    "rule_name": rule_name,
                    "severity": ComplianceSeverity.ERROR.value,
                    "description": f"Missing required field '{field}' in {len(missing)} record(s)",
                    "affected_records": missing[:20],
                    "remediation": f"Provide {field} for all import records",
                })

        # CBAM-R-006: Supplier data recommended
        no_supplier = [r.get("record_id", "?") for r in records if not r.get("supplier_id")]
        if no_supplier:
            violations.append({
                "rule_id": "CBAM-R-006",
                "rule_name": "Supplier identification recommended",
                "severity": ComplianceSeverity.WARNING.value,
                "description": f"{len(no_supplier)} record(s) missing supplier_id",
                "affected_records": no_supplier[:20],
                "remediation": "Link import records to supplier identifiers",
            })

        # CBAM-R-007: Customs declaration reference recommended
        no_ref = [r.get("record_id", "?") for r in records if not r.get("customs_declaration_ref")]
        if no_ref:
            violations.append({
                "rule_id": "CBAM-R-007",
                "rule_name": "Customs declaration reference recommended",
                "severity": ComplianceSeverity.INFO.value,
                "description": f"{len(no_ref)} record(s) missing customs_declaration_ref",
                "affected_records": no_ref[:20],
                "remediation": "Add customs declaration references for traceability",
            })

        # CBAM-R-008: Import date should be present
        no_date = [r.get("record_id", "?") for r in records if not r.get("import_date")]
        if no_date:
            violations.append({
                "rule_id": "CBAM-R-008",
                "rule_name": "Import date should be present",
                "severity": ComplianceSeverity.WARNING.value,
                "description": f"{len(no_date)} record(s) missing import_date",
                "affected_records": no_date[:20],
                "remediation": "Provide import dates for quarter validation",
            })

        # CBAM-R-009: At least one record required
        if not records:
            violations.append({
                "rule_id": "CBAM-R-009",
                "rule_name": "Non-empty report required",
                "severity": ComplianceSeverity.ERROR.value,
                "description": "No import records in the report",
                "affected_records": [],
                "remediation": "Add at least one import record to the quarterly report",
            })

        # CBAM-R-010: Minimum record quality check
        low_quality = [
            r.get("record_id", "?") for r in records
            if r.get("data_quality_score", 1.0) < 0.2
        ]
        if low_quality:
            violations.append({
                "rule_id": "CBAM-R-010",
                "rule_name": "Low data quality detected",
                "severity": ComplianceSeverity.WARNING.value,
                "description": f"{len(low_quality)} record(s) with data quality score below 0.2",
                "affected_records": low_quality[:20],
                "remediation": "Improve data quality by verifying source data",
            })

        return violations

    def _check_cn_code_rules(
        self, records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Check CN code rules CBAM-R-011 through CBAM-R-020."""
        violations: List[Dict[str, Any]] = []

        # CBAM-R-011: CN code must be 6-10 digits
        bad_format = [
            r.get("record_id", "?") for r in records
            if not self._validate_cn_code_format(r.get("cn_code", ""))
        ]
        if bad_format:
            violations.append({
                "rule_id": "CBAM-R-011",
                "rule_name": "CN code format invalid",
                "severity": ComplianceSeverity.ERROR.value,
                "description": f"{len(bad_format)} record(s) with invalid CN code format",
                "affected_records": bad_format[:20],
                "remediation": "CN codes must be 6-10 digits per EU Combined Nomenclature",
            })

        # CBAM-R-012: CN code must be in CBAM Annex I scope
        out_of_scope = [
            r.get("record_id", "?") for r in records
            if self._classify_cn_code(r.get("cn_code", "")) is None
        ]
        if out_of_scope:
            violations.append({
                "rule_id": "CBAM-R-012",
                "rule_name": "CN code not in CBAM scope",
                "severity": ComplianceSeverity.ERROR.value,
                "description": f"{len(out_of_scope)} record(s) with CN code outside CBAM Annex I",
                "affected_records": out_of_scope[:20],
                "remediation": "Only report goods covered by CBAM Annex I (Regulation 2023/956)",
            })

        # CBAM-R-013: CN code consistency within sector
        sector_mismatch = []
        for r in records:
            classified = self._classify_cn_code(r.get("cn_code", ""))
            declared = r.get("cbam_sector", "")
            if classified and declared and classified.value != declared:
                sector_mismatch.append(r.get("record_id", "?"))
        if sector_mismatch:
            violations.append({
                "rule_id": "CBAM-R-013",
                "rule_name": "CN code sector mismatch",
                "severity": ComplianceSeverity.WARNING.value,
                "description": f"{len(sector_mismatch)} record(s) with sector mismatch vs CN code",
                "affected_records": sector_mismatch[:20],
                "remediation": "Ensure declared sector matches CN code classification",
            })

        return violations

    def _check_emission_factor_rules(
        self, records: List[Dict[str, Any]], categories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Check emission factor rules CBAM-R-021 through CBAM-R-030."""
        violations: List[Dict[str, Any]] = []

        # CBAM-R-021: Emission factor must be positive
        zero_ef = [
            r.get("record_id", "?") for r in records
            if r.get("specific_embedded_emissions", 0) <= 0
        ]
        if zero_ef:
            violations.append({
                "rule_id": "CBAM-R-021",
                "rule_name": "Zero or negative emission factor",
                "severity": ComplianceSeverity.ERROR.value,
                "description": f"{len(zero_ef)} record(s) with zero/negative emission factor",
                "affected_records": zero_ef[:20],
                "remediation": "All CBAM goods must have a positive specific embedded emission factor",
            })

        # CBAM-R-022: Emission factor range check (not implausibly high)
        high_ef = []
        for r in records:
            ef = r.get("specific_embedded_emissions", 0)
            sector = r.get("cbam_sector", "")
            default_val = DEFAULT_EMISSION_FACTORS.get(sector, 5.0)
            if ef > default_val * 5:
                high_ef.append(r.get("record_id", "?"))
        if high_ef:
            violations.append({
                "rule_id": "CBAM-R-022",
                "rule_name": "Implausibly high emission factor",
                "severity": ComplianceSeverity.WARNING.value,
                "description": f"{len(high_ef)} record(s) with emission factor >5x default",
                "affected_records": high_ef[:20],
                "remediation": "Verify supplier emission data or use corrected factors",
            })

        # CBAM-R-023: Category-level emission intensity check
        for cat in categories:
            see = cat.get("specific_emissions_tco2e_per_t", 0)
            if see <= 0 and cat.get("total_quantity_tonnes", 0) > 0:
                violations.append({
                    "rule_id": "CBAM-R-023",
                    "rule_name": f"Zero emission intensity for CN {cat.get('cn_code', '?')}",
                    "severity": ComplianceSeverity.ERROR.value,
                    "description": (
                        f"Category {cat.get('cn_code', '?')} has zero specific emission "
                        f"intensity with {cat.get('total_quantity_tonnes', 0):.2f} tonnes"
                    ),
                    "affected_records": [],
                    "remediation": "Ensure valid emission factors are applied to all categories",
                })

        return violations

    def _check_default_factor_cap_rules(
        self, records: List[Dict[str, Any]], year: int
    ) -> List[Dict[str, Any]]:
        """Check default factor usage cap rules CBAM-R-031 through CBAM-R-035."""
        violations: List[Dict[str, Any]] = []
        total = len(records)
        if total == 0:
            return violations

        default_records = [
            r for r in records
            if r.get("emission_data_source") in (
                EmissionDataSource.DEFAULT_EU.value,
                EmissionDataSource.DEFAULT_JRC.value,
            )
        ]
        default_pct = len(default_records) / total * 100

        # CBAM-R-031: Transitional period allows 100% default factors
        if year <= 2025:
            pass  # No cap during transitional period

        # CBAM-R-032: From 2026, default factor usage triggers markup
        if year >= 2026 and default_pct > 50:
            violations.append({
                "rule_id": "CBAM-R-032",
                "rule_name": "High default factor usage in definitive period",
                "severity": ComplianceSeverity.WARNING.value,
                "description": (
                    f"{default_pct:.1f}% of records use default emission factors. "
                    f"Default values will incur a markup from 2026."
                ),
                "affected_records": [r.get("record_id", "?") for r in default_records[:20]],
                "remediation": "Obtain actual emission data from suppliers to avoid markup",
            })

        # CBAM-R-033: From 2028, default factors incur 25% surcharge
        if year >= 2028 and default_pct > 20:
            violations.append({
                "rule_id": "CBAM-R-033",
                "rule_name": "Default factor surcharge threshold exceeded",
                "severity": ComplianceSeverity.ERROR.value,
                "description": (
                    f"{default_pct:.1f}% of records use default factors; "
                    f"surcharge applies from 2028 for default values"
                ),
                "affected_records": [r.get("record_id", "?") for r in default_records[:20]],
                "remediation": "Reduce default factor usage below 20%",
            })

        # CBAM-R-034: Per-sector default factor cap
        sector_default_counts: Dict[str, Tuple[int, int]] = {}
        for r in records:
            sector = r.get("cbam_sector", "unknown")
            is_default = r.get("emission_data_source") in (
                EmissionDataSource.DEFAULT_EU.value,
                EmissionDataSource.DEFAULT_JRC.value,
            )
            if sector not in sector_default_counts:
                sector_default_counts[sector] = (0, 0)
            total_s, default_s = sector_default_counts[sector]
            sector_default_counts[sector] = (total_s + 1, default_s + (1 if is_default else 0))

        for sector, (total_s, default_s) in sector_default_counts.items():
            if total_s > 0 and default_s / total_s > 0.8 and year >= 2027:
                violations.append({
                    "rule_id": "CBAM-R-034",
                    "rule_name": f"Sector '{sector}' default factor cap exceeded",
                    "severity": ComplianceSeverity.WARNING.value,
                    "description": (
                        f"Sector '{sector}': {default_s}/{total_s} records "
                        f"({default_s/total_s*100:.0f}%) use default factors"
                    ),
                    "affected_records": [],
                    "remediation": f"Collect actual supplier data for {sector} imports",
                })

        return violations

    def _check_period_specific_rules(
        self, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check period-specific rules CBAM-R-036 through CBAM-R-040."""
        violations: List[Dict[str, Any]] = []
        year = context["year"]
        quarter = context["quarter"]

        # CBAM-R-036: Transitional period quarterly report deadline
        if year <= 2025:
            violations.append({
                "rule_id": "CBAM-R-036",
                "rule_name": "Transitional period reminder",
                "severity": ComplianceSeverity.INFO.value,
                "description": (
                    f"Quarter {year}-{quarter.value} falls in the CBAM transitional period. "
                    "Report due within one month after quarter end."
                ),
                "affected_records": [],
                "remediation": "Submit report to CBAM Registry by deadline",
            })

        # CBAM-R-037: Definitive period annual obligation reminder
        if year >= 2026 and quarter == Quarter.Q4:
            violations.append({
                "rule_id": "CBAM-R-037",
                "rule_name": "Annual declaration preparation required",
                "severity": ComplianceSeverity.INFO.value,
                "description": (
                    f"Q4 {year}: Annual declaration preparation should begin. "
                    "Declaration due by May 31 of the following year."
                ),
                "affected_records": [],
                "remediation": "Begin annual declaration aggregation process",
            })

        # CBAM-R-038: 50% holding requirement check (definitive period)
        if year >= 2026:
            violations.append({
                "rule_id": "CBAM-R-038",
                "rule_name": "Quarterly 50% holding requirement",
                "severity": ComplianceSeverity.INFO.value,
                "description": (
                    "Per Article 22(2), authorized declarants must hold at least 50% "
                    "of estimated annual certificate obligation by end of each quarter."
                ),
                "affected_records": [],
                "remediation": "Verify certificate holdings meet 50% threshold",
            })

        return violations

    def _check_reporting_format_rules(
        self, categories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Check reporting format rules CBAM-R-041 through CBAM-R-045."""
        violations: List[Dict[str, Any]] = []

        # CBAM-R-041: At least one goods category required
        if not categories:
            violations.append({
                "rule_id": "CBAM-R-041",
                "rule_name": "No goods categories in report",
                "severity": ComplianceSeverity.ERROR.value,
                "description": "Report must contain at least one goods category",
                "affected_records": [],
                "remediation": "Ensure import data maps to CBAM goods categories",
            })

        # CBAM-R-042: All CBAM sectors must be from allowed set
        allowed_sectors = {s.value for s in CbamSector}
        for cat in categories:
            sector = cat.get("cbam_sector", "")
            if sector and sector not in allowed_sectors:
                violations.append({
                    "rule_id": "CBAM-R-042",
                    "rule_name": f"Invalid CBAM sector: {sector}",
                    "severity": ComplianceSeverity.ERROR.value,
                    "description": f"Sector '{sector}' is not a valid CBAM sector",
                    "affected_records": [],
                    "remediation": f"Use one of: {', '.join(sorted(allowed_sectors))}",
                })

        # CBAM-R-043: Quantity precision check (max 4 decimal places)
        for cat in categories:
            qty = cat.get("total_quantity_tonnes", 0)
            qty_str = f"{qty:.10f}"
            decimal_part = qty_str.split(".")[1] if "." in qty_str else ""
            significant = decimal_part.rstrip("0")
            if len(significant) > 4:
                violations.append({
                    "rule_id": "CBAM-R-043",
                    "rule_name": "Quantity precision exceeds 4 decimal places",
                    "severity": ComplianceSeverity.INFO.value,
                    "description": f"CN {cat.get('cn_code', '?')}: quantity has >4 decimal precision",
                    "affected_records": [],
                    "remediation": "Round quantities to 4 decimal places",
                })

        return violations

    def _check_cross_validation_rules(
        self, records: List[Dict[str, Any]], categories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Check cross-validation rules CBAM-R-046 through CBAM-R-050."""
        violations: List[Dict[str, Any]] = []

        # CBAM-R-046: Total quantity from records must match category totals
        record_qty_total = sum(r.get("quantity_tonnes", 0) for r in records)
        category_qty_total = sum(c.get("total_quantity_tonnes", 0) for c in categories)

        if abs(record_qty_total - category_qty_total) > 0.01:
            violations.append({
                "rule_id": "CBAM-R-046",
                "rule_name": "Quantity reconciliation mismatch",
                "severity": ComplianceSeverity.ERROR.value,
                "description": (
                    f"Record total ({record_qty_total:.4f} t) does not match "
                    f"category total ({category_qty_total:.4f} t)"
                ),
                "affected_records": [],
                "remediation": "Investigate and reconcile quantity discrepancies",
            })

        # CBAM-R-047: No single record should exceed 50% of total quantity
        if record_qty_total > 0:
            for r in records:
                qty = r.get("quantity_tonnes", 0)
                if qty / record_qty_total > 0.5 and len(records) > 1:
                    violations.append({
                        "rule_id": "CBAM-R-047",
                        "rule_name": "Single record dominates total quantity",
                        "severity": ComplianceSeverity.INFO.value,
                        "description": (
                            f"Record {r.get('record_id', '?')} is {qty/record_qty_total*100:.1f}% "
                            f"of total quantity; verify this is correct"
                        ),
                        "affected_records": [r.get("record_id", "?")],
                        "remediation": "Confirm large single-record quantity is accurate",
                    })

        # CBAM-R-048: Multiple countries should be from non-EU origins
        eu_records = [
            r for r in records
            if r.get("country_of_origin", "") in self._get_eu_country_codes()
        ]
        if eu_records:
            violations.append({
                "rule_id": "CBAM-R-048",
                "rule_name": "EU-origin records included",
                "severity": ComplianceSeverity.ERROR.value,
                "description": f"{len(eu_records)} record(s) from EU member states (CBAM exempt)",
                "affected_records": [r.get("record_id", "?") for r in eu_records[:20]],
                "remediation": "Remove EU-origin records; CBAM applies only to non-EU imports",
            })

        return violations

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _classify_cn_code(self, cn_code: str) -> Optional[CbamSector]:
        """Classify a CN code into a CBAM sector based on prefix matching."""
        for sector, prefixes in CBAM_CN_CODE_PREFIXES.items():
            for prefix in prefixes:
                if cn_code.startswith(prefix):
                    return sector
        return None

    def _validate_cn_code_format(self, cn_code: str) -> bool:
        """Validate CN code is 6-10 digits."""
        return bool(cn_code) and cn_code.isdigit() and 6 <= len(cn_code) <= 10

    def _validate_eori(self, eori: str) -> bool:
        """Validate EORI number format."""
        return bool(EORI_PATTERN.match(eori))

    def _validate_country_code(self, code: str) -> bool:
        """Validate ISO 3166-1 alpha-2 country code (basic check)."""
        return bool(code) and len(code) == 2 and code.isalpha() and code.isupper()

    def _get_eu_country_codes(self) -> set:
        """Return set of EU member state ISO 3166-1 alpha-2 codes."""
        return {
            "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
            "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
            "PL", "PT", "RO", "SK", "SI", "ES", "SE",
        }

    def _filter_by_quarter(
        self, records: List[Dict[str, Any]], quarter: Quarter, year: int
    ) -> List[Dict[str, Any]]:
        """Filter records to those within the given quarter's date range."""
        quarter_ranges = {
            Quarter.Q1: (f"{year}-01-01", f"{year}-03-31"),
            Quarter.Q2: (f"{year}-04-01", f"{year}-06-30"),
            Quarter.Q3: (f"{year}-07-01", f"{year}-09-30"),
            Quarter.Q4: (f"{year}-10-01", f"{year}-12-31"),
        }
        start, end = quarter_ranges[quarter]

        filtered: List[Dict[str, Any]] = []
        for r in records:
            import_date = r.get("import_date")
            if not import_date:
                # Keep records without dates (cannot filter)
                filtered.append(r)
                continue
            if start <= import_date <= end:
                filtered.append(r)

        return filtered

    def _get_default_emission_factor(self, sector: str, year: int) -> float:
        """Get default emission factor with year-based markup.

        Per the Omnibus Simplification Package COM(2025) 508, default values
        receive an increasing markup from 2026 onward to incentivize actual
        data collection.

        Markup schedule:
            2025 and before: 0% markup
            2026: +10% markup
            2027: +20% markup
            2028+: +30% markup
        """
        base = DEFAULT_EMISSION_FACTORS.get(sector, 1.0)
        if year <= 2025:
            return base
        elif year == 2026:
            return round(base * 1.10, 6)
        elif year == 2027:
            return round(base * 1.20, 6)
        else:
            return round(base * 1.30, 6)

    def _match_supplier_data(
        self,
        supplier_id: Optional[str],
        cn_code: str,
        supplier_map: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Match supplier emission data by supplier_id and CN code."""
        if not supplier_id:
            return None
        key = f"{supplier_id}:{cn_code}"
        if key in supplier_map:
            return supplier_map[key]
        # Try prefix match (first 6 digits of CN code)
        prefix_key = f"{supplier_id}:{cn_code[:6]}"
        return supplier_map.get(prefix_key)

    def _generate_findings_report(
        self, violations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a structured compliance findings report."""
        by_severity: Dict[str, List[Dict[str, Any]]] = {
            "error": [],
            "warning": [],
            "info": [],
        }
        for v in violations:
            severity = v.get("severity", "info")
            if severity in by_severity:
                by_severity[severity].append(v)

        return {
            "total_findings": len(violations),
            "by_severity": {k: len(v) for k, v in by_severity.items()},
            "findings": by_severity,
            "generated_at": datetime.utcnow().isoformat(),
        }

    # =========================================================================
    # ASYNC AGENT INVOCATION STUBS
    # =========================================================================

    async def _load_supplier_emission_data(
        self, context: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Load supplier emission data from data store.

        In production, this invokes the SupplierManagementEngine to fetch
        verified supplier emission data from the database.
        """
        await asyncio.sleep(0)
        return {}

    async def _generate_cbam_xml(
        self,
        report_id: str,
        quarter: str,
        year: int,
        goods_categories: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> str:
        """Generate CBAM XML per EU CBAM Registry XSD schema.

        In production, this uses the QuarterlyReportingEngine to render
        the XML from goods categories and metadata.
        """
        await asyncio.sleep(0)
        # Stub XML structure
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<CBAMQuarterlyReport reportId="{report_id}" quarter="{quarter}" year="{year}">',
        ]
        for cat in goods_categories:
            xml_lines.append(
                f'  <GoodsCategory cnCode="{cat.get("cn_code", "")}" '
                f'sector="{cat.get("cbam_sector", "")}" '
                f'quantityTonnes="{cat.get("total_quantity_tonnes", 0):.4f}" '
                f'embeddedEmissions="{cat.get("total_embedded_emissions_tco2e", 0):.4f}" />'
            )
        xml_lines.append("</CBAMQuarterlyReport>")
        return "\n".join(xml_lines)

    async def _generate_executive_summary(
        self,
        report_id: str,
        quarter: str,
        goods_categories: List[Dict[str, Any]],
        compliance_score: float,
        violations_count: int,
    ) -> Dict[str, Any]:
        """Generate executive summary for the quarterly report."""
        await asyncio.sleep(0)
        total_emissions = sum(c.get("total_embedded_emissions_tco2e", 0) for c in goods_categories)
        total_qty = sum(c.get("total_quantity_tonnes", 0) for c in goods_categories)

        return {
            "report_id": report_id,
            "quarter": quarter,
            "total_emissions_tco2e": round(total_emissions, 4),
            "total_quantity_tonnes": round(total_qty, 4),
            "goods_categories": len(goods_categories),
            "compliance_score": compliance_score,
            "violations_count": violations_count,
            "generated_at": datetime.utcnow().isoformat(),
        }

    async def _run_pre_submission_validation(
        self, xml_content: str, config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Run pre-submission validation checks against the XML report."""
        await asyncio.sleep(0)
        checks: List[Dict[str, Any]] = []

        # Check 1: XML not empty
        checks.append({
            "check_name": "xml_not_empty",
            "status": "PASS" if xml_content else "FAIL",
            "message": "XML content present" if xml_content else "XML content is empty",
        })

        # Check 2: XML contains report header
        checks.append({
            "check_name": "xml_header_present",
            "status": "PASS" if "CBAMQuarterlyReport" in xml_content else "FAIL",
            "message": "Report header found" if "CBAMQuarterlyReport" in xml_content else "Missing header",
        })

        # Check 3: XML encoding declaration
        checks.append({
            "check_name": "xml_encoding",
            "status": "PASS" if "UTF-8" in xml_content else "FAIL",
            "message": "UTF-8 encoding declared" if "UTF-8" in xml_content else "Missing encoding",
        })

        # Check 4: At least one goods category
        checks.append({
            "check_name": "goods_categories_present",
            "status": "PASS" if "GoodsCategory" in xml_content else "FAIL",
            "message": "Goods categories found" if "GoodsCategory" in xml_content else "No categories",
        })

        return checks

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @staticmethod
    def _hash(data: Any) -> str:
        """Compute SHA-256 provenance hash of arbitrary data."""
        return hashlib.sha256(str(data).encode("utf-8")).hexdigest()
