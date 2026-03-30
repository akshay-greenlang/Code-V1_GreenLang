# -*- coding: utf-8 -*-
"""
Bulk Import Workflow
======================

Three-phase bulk data import workflow for EUDR compliance. Handles parsing
of uploaded files (CSV, Excel, JSON, GeoJSON), validation against EUDR
rules, and integration into the EUDR compliance system.

Regulatory Context:
    Per EU Regulation 2023/1115 (EUDR):
    - Article 9: DDS requires comprehensive data on all suppliers, plots,
      and commodities -- bulk import facilitates initial data loading
    - Article 10: Information must be adequate and verifiable; import
      validation ensures data quality from the start
    - Article 12: Supply chain traceability requires complete linkage
      between suppliers, plots, and commodity flows

    Bulk import is critical for organizations with large supplier bases
    (100+ suppliers) and extensive geolocation datasets. It reduces
    manual data entry errors and accelerates compliance readiness.

Phases:
    1. File processing - Parse files, validate format, extract records
    2. Validation and enrichment - Validate, deduplicate, enrich, score
    3. Integration - Load records, link entities, generate summary

Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from greenlang.schemas.enums import FileFormat

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


class RecordType(str, Enum):
    """Type of imported record."""
    SUPPLIER = "supplier"
    PLOT = "plot"
    CERTIFICATION = "certification"
    COMMODITY = "commodity"


class ImportStatus(str, Enum):
    """Status of an individual import record."""
    VALID = "valid"
    INVALID = "invalid"
    DUPLICATE = "duplicate"
    ENRICHED = "enriched"
    LOADED = "loaded"
    FAILED = "failed"


class EUDRCommodity(str, Enum):
    """EUDR-relevant commodities."""
    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


# Country risk benchmarking
HIGH_RISK_COUNTRIES = {
    "BR", "CD", "CM", "CO", "CI", "EC", "GA", "GH", "GT", "GN",
    "HN", "ID", "KH", "LA", "LR", "MG", "MM", "MY", "MZ", "NG",
    "PA", "PE", "PG", "PH", "SL", "TZ", "TH", "UG", "VN",
}

LOW_RISK_COUNTRIES = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
    "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
    "PL", "PT", "RO", "SK", "SI", "ES", "SE",
    "NO", "IS", "CH", "LI", "GB", "AU", "NZ", "JP", "KR", "CA",
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, ge=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class WorkflowContext(BaseModel):
    """Shared context passed between workflow phases."""
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    config: Dict[str, Any] = Field(default_factory=dict)
    phase_results: List[PhaseResult] = Field(default_factory=list)
    checkpoints: Dict[str, Any] = Field(default_factory=dict)
    state: Dict[str, Any] = Field(default_factory=dict)
    started_at: Optional[datetime] = Field(None)
    last_checkpoint_at: Optional[datetime] = Field(None)

    class Config:
        arbitrary_types_allowed = True


class FileUpload(BaseModel):
    """Uploaded file metadata for import."""
    file_name: str = Field(..., description="Original file name")
    file_format: FileFormat = Field(..., description="File format")
    record_type: RecordType = Field(..., description="Type of records in file")
    file_size_bytes: int = Field(default=0, ge=0, description="File size")
    encoding: str = Field(default="utf-8", description="File encoding")
    records: List[Dict[str, Any]] = Field(
        default_factory=list, description="Pre-parsed records (if available)"
    )


class BulkImportInput(BaseModel):
    """Input data for the bulk import workflow."""
    files: List[FileUpload] = Field(
        ..., min_length=1, description="Files to import"
    )
    deduplicate: bool = Field(default=True, description="Enable deduplication")
    enrich_with_risk: bool = Field(
        default=True, description="Enrich records with country risk data"
    )
    dry_run: bool = Field(default=False, description="Validate without loading")
    config: Dict[str, Any] = Field(default_factory=dict)


class BulkImportResult(BaseModel):
    """Complete result from the bulk import workflow."""
    workflow_name: str = Field(default="bulk_import")
    status: PhaseStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    files_processed: int = Field(default=0, ge=0)
    total_records_parsed: int = Field(default=0, ge=0)
    records_valid: int = Field(default=0, ge=0)
    records_invalid: int = Field(default=0, ge=0)
    records_duplicate: int = Field(default=0, ge=0)
    records_loaded: int = Field(default=0, ge=0)
    records_failed: int = Field(default=0, ge=0)
    data_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    suppliers_created: int = Field(default=0, ge=0)
    plots_created: int = Field(default=0, ge=0)
    dry_run: bool = Field(default=False)
    provenance_hash: str = Field(default="")
    execution_id: str = Field(default="")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)


# =============================================================================
# BULK IMPORT WORKFLOW
# =============================================================================


class BulkImportWorkflow:
    """
    Three-phase bulk data import workflow.

    Handles parsing, validation, enrichment, and loading of large datasets
    into the EUDR compliance system. Supports CSV, Excel, JSON, and GeoJSON
    formats.

    Agent Dependencies:
        - DATA-010 (Quality Profiler)
        - DATA-011 (Duplicate Detection)
        - DATA-019 (Validation Rule Engine)

    Attributes:
        config: Workflow configuration.
        logger: Logger instance.
        _execution_id: Unique execution identifier.
        _phase_results: Accumulated phase results.
        _checkpoint_store: Checkpoint data for resume.

    Example:
        >>> wf = BulkImportWorkflow()
        >>> result = await wf.run(BulkImportInput(
        ...     files=[FileUpload(
        ...         file_name="suppliers.csv",
        ...         file_format=FileFormat.CSV,
        ...         record_type=RecordType.SUPPLIER,
        ...         records=[{"supplier_name": "Test", "country_code": "BR"}],
        ...     )],
        ... ))
        >>> assert result.status == PhaseStatus.COMPLETED
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the BulkImportWorkflow.

        Args:
            config: Optional configuration dict.
        """
        self.config: Dict[str, Any] = config or {}
        self.logger = logging.getLogger(f"{__name__}.BulkImportWorkflow")
        self._execution_id: str = str(uuid.uuid4())
        self._phase_results: List[PhaseResult] = []
        self._checkpoint_store: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def run(
        self, input_data: BulkImportInput
    ) -> BulkImportResult:
        """
        Execute the full 3-phase bulk import workflow.

        Args:
            input_data: Import parameters including files and configuration.

        Returns:
            BulkImportResult with import statistics and quality metrics.
        """
        started_at = datetime.utcnow()

        self.logger.info(
            "Starting bulk import workflow execution_id=%s files=%d dry_run=%s",
            self._execution_id, len(input_data.files), input_data.dry_run,
        )

        context = WorkflowContext(
            execution_id=self._execution_id,
            config={**self.config, **input_data.config},
            started_at=started_at,
            state={
                "files": [f.model_dump() for f in input_data.files],
                "deduplicate": input_data.deduplicate,
                "enrich_with_risk": input_data.enrich_with_risk,
                "dry_run": input_data.dry_run,
            },
        )

        phase_handlers = [
            ("file_processing", self._phase_1_file_processing),
            ("validation_and_enrichment", self._phase_2_validation_and_enrichment),
            ("integration", self._phase_3_integration),
        ]

        overall_status = PhaseStatus.COMPLETED

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
                    "Phase '%s' failed: %s", phase_name, exc, exc_info=True,
                )
                phase_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    duration_seconds=(datetime.utcnow() - phase_start).total_seconds(),
                    outputs={"error": str(exc)},
                    provenance_hash=self._hash({"error": str(exc)}),
                )

            self._phase_results.append(phase_result)
            context.phase_results = list(self._phase_results)

            self._checkpoint_store[phase_name] = {
                "result": phase_result.model_dump(),
                "saved_at": datetime.utcnow().isoformat(),
            }

            if phase_result.status == PhaseStatus.FAILED:
                overall_status = PhaseStatus.FAILED
                if phase_name == "file_processing":
                    self.logger.error("File processing failed; halting.")
                    break

        completed_at = datetime.utcnow()

        provenance = self._hash({
            "execution_id": self._execution_id,
            "phases": [p.provenance_hash for p in self._phase_results],
        })

        self.logger.info(
            "Bulk import finished execution_id=%s status=%s",
            self._execution_id, overall_status.value,
        )

        return BulkImportResult(
            status=overall_status,
            phases=self._phase_results,
            files_processed=context.state.get("files_processed", 0),
            total_records_parsed=context.state.get("total_records_parsed", 0),
            records_valid=context.state.get("records_valid", 0),
            records_invalid=context.state.get("records_invalid", 0),
            records_duplicate=context.state.get("records_duplicate", 0),
            records_loaded=context.state.get("records_loaded", 0),
            records_failed=context.state.get("records_failed", 0),
            data_quality_score=context.state.get("data_quality_score", 0.0),
            suppliers_created=context.state.get("suppliers_created", 0),
            plots_created=context.state.get("plots_created", 0),
            dry_run=input_data.dry_run,
            provenance_hash=provenance,
            execution_id=self._execution_id,
            started_at=started_at,
            completed_at=completed_at,
        )

    # -------------------------------------------------------------------------
    # Phase 1: File Processing
    # -------------------------------------------------------------------------

    async def _phase_1_file_processing(
        self, context: WorkflowContext
    ) -> PhaseResult:
        """
        Parse uploaded files (CSV, Excel, JSON, GeoJSON), validate file
        format, extract records, handle encoding issues, and report
        parse errors.
        """
        phase_name = "file_processing"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        files = context.state.get("files", [])

        if not files:
            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.FAILED,
                outputs={"error": "No files provided"},
                provenance_hash=self._hash({"phase": phase_name, "error": "no_files"}),
            )

        self.logger.info("Processing %d file(s)", len(files))

        all_records: List[Dict[str, Any]] = []
        file_summaries: List[Dict[str, Any]] = []
        total_parse_errors = 0

        for file_info in files:
            file_name = file_info.get("file_name", "unknown")
            file_format = file_info.get("file_format", "csv")
            record_type = file_info.get("record_type", "supplier")
            encoding = file_info.get("encoding", "utf-8")
            pre_parsed = file_info.get("records", [])

            self.logger.info(
                "Processing file: %s (format=%s, type=%s)",
                file_name, file_format, record_type,
            )

            # Parse records from file
            if pre_parsed:
                # Records already pre-parsed (e.g., from API)
                parsed_records = pre_parsed
                parse_errors = 0
            else:
                # In production, parse from file bytes via DATA-001/DATA-002
                parsed_records, parse_errors = await self._parse_file(
                    file_name, file_format, encoding,
                )

            if parse_errors > 0:
                warnings.append(
                    f"File '{file_name}': {parse_errors} record(s) failed parsing"
                )
                total_parse_errors += parse_errors

            # Tag records with source metadata
            for idx, record in enumerate(parsed_records):
                record["_import_id"] = f"IMP-{uuid.uuid4().hex[:8]}"
                record["_source_file"] = file_name
                record["_source_format"] = file_format
                record["_record_type"] = record_type
                record["_record_index"] = idx
                record["_imported_at"] = datetime.utcnow().isoformat()

            all_records.extend(parsed_records)

            file_summaries.append({
                "file_name": file_name,
                "file_format": file_format,
                "record_type": record_type,
                "records_extracted": len(parsed_records),
                "parse_errors": parse_errors,
                "encoding": encoding,
            })

        # Categorize records by type
        records_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for record in all_records:
            rt = record.get("_record_type", "unknown")
            if rt not in records_by_type:
                records_by_type[rt] = []
            records_by_type[rt].append(record)

        context.state["all_records"] = all_records
        context.state["records_by_type"] = records_by_type
        context.state["files_processed"] = len(files)
        context.state["total_records_parsed"] = len(all_records)

        outputs["files_processed"] = len(files)
        outputs["total_records_extracted"] = len(all_records)
        outputs["total_parse_errors"] = total_parse_errors
        outputs["file_summaries"] = file_summaries
        outputs["records_by_type"] = {
            k: len(v) for k, v in records_by_type.items()
        }

        if not all_records:
            warnings.append("No records extracted from any file.")

        self.logger.info(
            "Phase 1 complete: %d files, %d records extracted, %d parse errors",
            len(files), len(all_records), total_parse_errors,
        )

        provenance = self._hash({
            "phase": phase_name,
            "files": len(files),
            "records": len(all_records),
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 2: Validation and Enrichment
    # -------------------------------------------------------------------------

    async def _phase_2_validation_and_enrichment(
        self, context: WorkflowContext
    ) -> PhaseResult:
        """
        Validate each record against EUDR rules, deduplicate against existing
        data, enrich with country risk data, classify by commodity, and
        calculate data quality score.

        Uses:
            - DATA-010 (Quality Profiler)
            - DATA-011 (Duplicate Detection)
            - DATA-019 (Validation Rule Engine)
        """
        phase_name = "validation_and_enrichment"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        all_records = context.state.get("all_records", [])
        deduplicate = context.state.get("deduplicate", True)
        enrich_with_risk = context.state.get("enrich_with_risk", True)

        if not all_records:
            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.COMPLETED,
                outputs={"validated": 0},
                warnings=["No records to validate"],
                provenance_hash=self._hash({"phase": phase_name, "records": 0}),
            )

        valid_records: List[Dict[str, Any]] = []
        invalid_records: List[Dict[str, Any]] = []
        duplicate_records: List[Dict[str, Any]] = []

        # Deduplication tracking
        seen_keys: Dict[str, str] = {}

        # EUDR valid commodities
        valid_commodities = {e.value for e in EUDRCommodity}

        for record in all_records:
            record_type = record.get("_record_type", "supplier")
            import_id = record.get("_import_id", "")
            errors: List[str] = []

            # Type-specific validation
            if record_type == "supplier":
                errors.extend(self._validate_supplier_record(record))
            elif record_type == "plot":
                errors.extend(self._validate_plot_record(record))
            elif record_type == "certification":
                errors.extend(self._validate_certification_record(record))
            elif record_type == "commodity":
                errors.extend(self._validate_commodity_record(record))

            # Deduplication
            if deduplicate:
                dedup_key = self._generate_dedup_key(record, record_type)
                if dedup_key in seen_keys:
                    record["_status"] = ImportStatus.DUPLICATE.value
                    record["_duplicate_of"] = seen_keys[dedup_key]
                    duplicate_records.append(record)
                    continue
                seen_keys[dedup_key] = import_id

            # Check against existing records in system
            existing = await self._check_existing_record(record, record_type)
            if existing:
                record["_status"] = ImportStatus.DUPLICATE.value
                record["_duplicate_of"] = existing
                duplicate_records.append(record)
                continue

            if errors:
                record["_status"] = ImportStatus.INVALID.value
                record["_errors"] = errors
                invalid_records.append(record)
                continue

            # Enrichment
            if enrich_with_risk:
                record = self._enrich_record(record, record_type)

            # Calculate per-record quality score
            record["_quality_score"] = self._calculate_record_quality(
                record, record_type,
            )

            record["_status"] = ImportStatus.ENRICHED.value
            valid_records.append(record)

        # Calculate aggregate quality score
        total_quality = sum(r.get("_quality_score", 0.0) for r in valid_records)
        avg_quality = (
            total_quality / len(valid_records) if valid_records else 0.0
        )

        context.state["valid_records"] = valid_records
        context.state["invalid_records"] = invalid_records
        context.state["duplicate_records"] = duplicate_records
        context.state["records_valid"] = len(valid_records)
        context.state["records_invalid"] = len(invalid_records)
        context.state["records_duplicate"] = len(duplicate_records)
        context.state["data_quality_score"] = round(avg_quality, 4)

        outputs["records_validated"] = len(all_records)
        outputs["valid"] = len(valid_records)
        outputs["invalid"] = len(invalid_records)
        outputs["duplicate"] = len(duplicate_records)
        outputs["validation_pass_rate"] = (
            round(len(valid_records) / len(all_records) * 100, 2)
            if all_records else 0.0
        )
        outputs["avg_quality_score"] = round(avg_quality, 4)

        # Quality distribution
        high_quality = sum(1 for r in valid_records if r.get("_quality_score", 0) >= 0.8)
        medium_quality = sum(
            1 for r in valid_records
            if 0.5 <= r.get("_quality_score", 0) < 0.8
        )
        low_quality = sum(1 for r in valid_records if r.get("_quality_score", 0) < 0.5)

        outputs["quality_distribution"] = {
            "high": high_quality,
            "medium": medium_quality,
            "low": low_quality,
        }

        if invalid_records:
            # Group errors by type
            error_types: Dict[str, int] = {}
            for r in invalid_records:
                for err in r.get("_errors", []):
                    error_types[err] = error_types.get(err, 0) + 1
            outputs["error_types"] = error_types

            warnings.append(
                f"{len(invalid_records)} record(s) failed validation. "
                "Review and correct before re-importing."
            )

        if duplicate_records:
            warnings.append(
                f"{len(duplicate_records)} duplicate record(s) detected and excluded."
            )

        self.logger.info(
            "Phase 2 complete: %d valid, %d invalid, %d duplicate, "
            "quality=%.4f",
            len(valid_records), len(invalid_records),
            len(duplicate_records), avg_quality,
        )

        provenance = self._hash({
            "phase": phase_name,
            "valid": len(valid_records),
            "invalid": len(invalid_records),
            "duplicate": len(duplicate_records),
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Integration
    # -------------------------------------------------------------------------

    async def _phase_3_integration(
        self, context: WorkflowContext
    ) -> PhaseResult:
        """
        Load validated records into EUDR system, link suppliers to plots,
        update risk scores, and generate import summary report with
        success/failure counts.
        """
        phase_name = "integration"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        valid_records = context.state.get("valid_records", [])
        dry_run = context.state.get("dry_run", False)

        if not valid_records:
            outputs["loaded"] = 0
            outputs["summary"] = "No valid records to load."
            context.state["records_loaded"] = 0
            context.state["records_failed"] = 0

            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.COMPLETED,
                outputs=outputs,
                warnings=["No valid records available for integration"],
                provenance_hash=self._hash({"phase": phase_name, "loaded": 0}),
            )

        if dry_run:
            self.logger.info(
                "Dry run mode: skipping actual data loading for %d records",
                len(valid_records),
            )
            context.state["records_loaded"] = 0
            context.state["records_failed"] = 0

            outputs["dry_run"] = True
            outputs["would_load"] = len(valid_records)
            outputs["summary"] = (
                f"Dry run complete. {len(valid_records)} record(s) would be "
                "loaded in a live run."
            )

            provenance = self._hash({
                "phase": phase_name, "dry_run": True, "records": len(valid_records),
            })
            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.COMPLETED,
                outputs=outputs,
                provenance_hash=provenance,
            )

        self.logger.info("Loading %d valid record(s)", len(valid_records))

        loaded_count = 0
        failed_count = 0
        suppliers_created = 0
        plots_created = 0
        load_errors: List[Dict[str, Any]] = []

        # Group records by type for batch processing
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for record in valid_records:
            rt = record.get("_record_type", "supplier")
            if rt not in by_type:
                by_type[rt] = []
            by_type[rt].append(record)

        # Load suppliers first (other records may reference them)
        for record in by_type.get("supplier", []):
            success = await self._load_record(record, "supplier")
            if success:
                loaded_count += 1
                suppliers_created += 1
                record["_status"] = ImportStatus.LOADED.value
            else:
                failed_count += 1
                record["_status"] = ImportStatus.FAILED.value
                load_errors.append({
                    "import_id": record.get("_import_id", ""),
                    "type": "supplier",
                    "error": "Load failed",
                })

        # Load plots (link to suppliers)
        for record in by_type.get("plot", []):
            success = await self._load_record(record, "plot")
            if success:
                loaded_count += 1
                plots_created += 1
                record["_status"] = ImportStatus.LOADED.value
            else:
                failed_count += 1
                record["_status"] = ImportStatus.FAILED.value
                load_errors.append({
                    "import_id": record.get("_import_id", ""),
                    "type": "plot",
                    "error": "Load failed",
                })

        # Load certifications
        for record in by_type.get("certification", []):
            success = await self._load_record(record, "certification")
            if success:
                loaded_count += 1
                record["_status"] = ImportStatus.LOADED.value
            else:
                failed_count += 1
                record["_status"] = ImportStatus.FAILED.value
                load_errors.append({
                    "import_id": record.get("_import_id", ""),
                    "type": "certification",
                    "error": "Load failed",
                })

        # Load commodity records
        for record in by_type.get("commodity", []):
            success = await self._load_record(record, "commodity")
            if success:
                loaded_count += 1
                record["_status"] = ImportStatus.LOADED.value
            else:
                failed_count += 1
                record["_status"] = ImportStatus.FAILED.value
                load_errors.append({
                    "import_id": record.get("_import_id", ""),
                    "type": "commodity",
                    "error": "Load failed",
                })

        # Link suppliers to plots
        linkage_results = await self._link_suppliers_to_plots(
            by_type.get("supplier", []), by_type.get("plot", []),
        )

        # Update risk scores for loaded suppliers
        risk_updates = 0
        for supplier_record in by_type.get("supplier", []):
            if supplier_record.get("_status") == ImportStatus.LOADED.value:
                updated = await self._update_risk_score(supplier_record)
                if updated:
                    risk_updates += 1

        context.state["records_loaded"] = loaded_count
        context.state["records_failed"] = failed_count
        context.state["suppliers_created"] = suppliers_created
        context.state["plots_created"] = plots_created

        # Generate import summary
        summary = {
            "execution_id": context.execution_id,
            "completed_at": datetime.utcnow().isoformat(),
            "total_valid": len(valid_records),
            "loaded": loaded_count,
            "failed": failed_count,
            "success_rate": (
                round(loaded_count / len(valid_records) * 100, 2)
                if valid_records else 0.0
            ),
            "suppliers_created": suppliers_created,
            "plots_created": plots_created,
            "linkages_created": linkage_results.get("linkages", 0),
            "risk_scores_updated": risk_updates,
        }

        outputs["loaded"] = loaded_count
        outputs["failed"] = failed_count
        outputs["suppliers_created"] = suppliers_created
        outputs["plots_created"] = plots_created
        outputs["load_errors"] = load_errors[:50]  # Limit error detail output
        outputs["linkages"] = linkage_results
        outputs["risk_updates"] = risk_updates
        outputs["summary"] = summary

        if failed_count > 0:
            warnings.append(
                f"{failed_count} record(s) failed to load. "
                "Review errors and retry."
            )

        self.logger.info(
            "Phase 3 complete: %d loaded, %d failed, %d suppliers, %d plots",
            loaded_count, failed_count, suppliers_created, plots_created,
        )

        provenance = self._hash({
            "phase": phase_name,
            "loaded": loaded_count,
            "failed": failed_count,
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # =========================================================================
    # VALIDATION METHODS
    # =========================================================================

    def _validate_supplier_record(
        self, record: Dict[str, Any]
    ) -> List[str]:
        """Validate a supplier import record."""
        errors: List[str] = []

        if not record.get("supplier_name"):
            errors.append("Missing supplier_name")

        country = record.get("country_code", "")
        if not country or len(country) != 2 or not country.isalpha():
            errors.append(f"Invalid country_code: '{country}'")

        commodity = str(record.get("commodity", "")).lower()
        if commodity and commodity not in {e.value for e in EUDRCommodity}:
            errors.append(f"Invalid commodity: '{commodity}'")

        email = record.get("contact_email", "")
        if email and "@" not in email:
            errors.append(f"Invalid email: '{email}'")

        return errors

    def _validate_plot_record(
        self, record: Dict[str, Any]
    ) -> List[str]:
        """Validate a plot/geolocation import record."""
        errors: List[str] = []

        lat = record.get("latitude")
        lon = record.get("longitude")

        if lat is None or not isinstance(lat, (int, float)):
            errors.append("Missing or invalid latitude")
        elif lat < -90.0 or lat > 90.0:
            errors.append(f"Latitude {lat} out of range [-90, 90]")

        if lon is None or not isinstance(lon, (int, float)):
            errors.append("Missing or invalid longitude")
        elif lon < -180.0 or lon > 180.0:
            errors.append(f"Longitude {lon} out of range [-180, 180]")

        area = record.get("area_hectares", 0)
        if isinstance(area, (int, float)) and area < 0:
            errors.append("Area must be non-negative")

        # Check polygon requirement for large plots
        if isinstance(area, (int, float)) and area >= 4.0:
            polygon = record.get("polygon_points", [])
            if not polygon or len(polygon) < 3:
                errors.append(
                    f"Plot area {area:.2f} ha >= 4 ha requires polygon boundary"
                )

        return errors

    def _validate_certification_record(
        self, record: Dict[str, Any]
    ) -> List[str]:
        """Validate a certification import record."""
        errors: List[str] = []

        if not record.get("cert_id"):
            errors.append("Missing cert_id")

        cert_type = record.get("cert_type", "")
        valid_types = {"FSC", "PEFC", "RSPO", "ISCC", "RA", "UTZ"}
        if cert_type and cert_type not in valid_types:
            errors.append(f"Invalid cert_type: '{cert_type}'")

        if not record.get("supplier_id"):
            errors.append("Missing supplier_id linkage")

        expiry = record.get("expiry_date", "")
        if expiry:
            now_str = datetime.utcnow().strftime("%Y-%m-%d")
            if expiry < now_str:
                errors.append(f"Certificate expired on {expiry}")

        return errors

    def _validate_commodity_record(
        self, record: Dict[str, Any]
    ) -> List[str]:
        """Validate a commodity transaction record."""
        errors: List[str] = []

        commodity = str(record.get("commodity_type", "")).lower()
        if commodity and commodity not in {e.value for e in EUDRCommodity}:
            errors.append(f"Invalid commodity_type: '{commodity}'")

        quantity = record.get("quantity")
        if quantity is not None and (not isinstance(quantity, (int, float)) or quantity <= 0):
            errors.append(f"Invalid quantity: {quantity}")

        return errors

    # =========================================================================
    # ENRICHMENT METHODS
    # =========================================================================

    def _enrich_record(
        self, record: Dict[str, Any], record_type: str
    ) -> Dict[str, Any]:
        """Enrich a record with country risk and commodity classification."""
        if record_type == "supplier":
            country = record.get("country_code", "")
            if country in HIGH_RISK_COUNTRIES:
                record["_country_risk"] = "high"
            elif country in LOW_RISK_COUNTRIES:
                record["_country_risk"] = "low"
            else:
                record["_country_risk"] = "standard"

            commodity = str(record.get("commodity", "")).lower()
            commodity_risk_map = {
                "oil_palm": "high", "soya": "high", "cattle": "high",
                "cocoa": "standard", "rubber": "standard",
                "coffee": "standard", "wood": "standard",
            }
            record["_commodity_risk"] = commodity_risk_map.get(commodity, "standard")

        elif record_type == "plot":
            country = record.get("country_code", "")
            if country in HIGH_RISK_COUNTRIES:
                record["_country_risk"] = "high"
            elif country in LOW_RISK_COUNTRIES:
                record["_country_risk"] = "low"
            else:
                record["_country_risk"] = "standard"

        return record

    def _calculate_record_quality(
        self, record: Dict[str, Any], record_type: str
    ) -> float:
        """Calculate quality score (0-1) for a single record."""
        if record_type == "supplier":
            fields = [
                "supplier_name", "country_code", "commodity",
                "contact_email", "contact_name", "address",
                "eori_number", "certifications",
            ]
        elif record_type == "plot":
            fields = [
                "latitude", "longitude", "area_hectares",
                "country_code", "commodity", "supplier_id",
            ]
        elif record_type == "certification":
            fields = [
                "cert_id", "cert_type", "supplier_id",
                "issue_date", "expiry_date", "scope",
            ]
        else:
            fields = ["commodity_type", "quantity", "origin_country", "supplier_id"]

        populated = 0
        for field in fields:
            value = record.get(field)
            if value is not None and value != "" and value != []:
                populated += 1

        return populated / len(fields) if fields else 0.0

    def _generate_dedup_key(
        self, record: Dict[str, Any], record_type: str
    ) -> str:
        """Generate deduplication key based on record type."""
        if record_type == "supplier":
            name = str(record.get("supplier_name", "")).lower().strip()
            country = str(record.get("country_code", "")).upper()
            return f"SUP:{name}:{country}"

        elif record_type == "plot":
            lat = str(record.get("latitude", ""))
            lon = str(record.get("longitude", ""))
            return f"PLOT:{lat}:{lon}"

        elif record_type == "certification":
            cert_id = str(record.get("cert_id", ""))
            cert_type = str(record.get("cert_type", ""))
            return f"CERT:{cert_id}:{cert_type}"

        else:
            commodity = str(record.get("commodity_type", ""))
            supplier = str(record.get("supplier_id", ""))
            date = str(record.get("transaction_date", ""))
            return f"COMM:{commodity}:{supplier}:{date}"

    # =========================================================================
    # ASYNC STUBS
    # =========================================================================

    async def _parse_file(
        self, file_name: str, file_format: str, encoding: str
    ) -> tuple:
        """Parse file and return (records, parse_errors)."""
        await asyncio.sleep(0)
        return [], 0

    async def _check_existing_record(
        self, record: Dict[str, Any], record_type: str
    ) -> Optional[str]:
        """Check if record already exists. Returns existing ID or None."""
        await asyncio.sleep(0)
        return None

    async def _load_record(
        self, record: Dict[str, Any], record_type: str
    ) -> bool:
        """Load a single record into the system. Returns success."""
        await asyncio.sleep(0)
        return True

    async def _link_suppliers_to_plots(
        self,
        suppliers: List[Dict[str, Any]],
        plots: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Link suppliers to plots based on shared identifiers."""
        await asyncio.sleep(0)
        return {"linkages": 0, "unlinked_plots": 0}

    async def _update_risk_score(
        self, supplier_record: Dict[str, Any]
    ) -> bool:
        """Update risk score for a loaded supplier."""
        await asyncio.sleep(0)
        return True

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @staticmethod
    def _hash(data: Any) -> str:
        """Compute SHA-256 provenance hash of arbitrary data."""
        return hashlib.sha256(str(data).encode("utf-8")).hexdigest()
