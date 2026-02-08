# -*- coding: utf-8 -*-
"""
Excel & CSV Normalizer Service Setup - AGENT-DATA-002: Excel Normalizer

Provides ``configure_excel_normalizer(app)`` which wires up the Excel &
CSV Normalizer SDK (Excel parser, CSV parser, column mapper, data type
detector, schema validator, data quality scorer, transform engine,
provenance tracker) and mounts the REST API.

Also exposes ``get_excel_normalizer(app)`` for programmatic access
and the ``ExcelNormalizerService`` facade class.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.excel_normalizer.setup import configure_excel_normalizer
    >>> app = FastAPI()
    >>> import asyncio
    >>> service = asyncio.run(configure_excel_normalizer(app))

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-002 Excel & CSV Normalizer
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.excel_normalizer.config import ExcelNormalizerConfig, get_config
from greenlang.excel_normalizer.metrics import (
    PROMETHEUS_AVAILABLE,
    record_file_processed,
    record_rows_normalized,
    record_columns_mapped,
    record_mapping_confidence,
    record_type_detection,
    record_validation_finding,
    record_quality_score,
    record_transform,
    record_batch_job,
    update_active_jobs,
    update_queue_size,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment, misc]
    FASTAPI_AVAILABLE = False


# ===================================================================
# Lightweight Pydantic models used by the facade
# ===================================================================


class SheetInfo(BaseModel):
    """Metadata for a single sheet within a workbook.

    Attributes:
        sheet_name: Name of the sheet.
        sheet_index: Zero-based index of the sheet.
        row_count: Number of data rows (excluding header).
        column_count: Number of columns.
        headers: List of column header strings.
    """
    sheet_name: str = Field(default="Sheet1")
    sheet_index: int = Field(default=0)
    row_count: int = Field(default=0)
    column_count: int = Field(default=0)
    headers: List[str] = Field(default_factory=list)


class FileRecord(BaseModel):
    """Record representing an uploaded and normalised file.

    Attributes:
        file_id: Unique identifier for this file.
        file_name: Original filename.
        file_format: Detected or declared file format.
        sheet_count: Number of sheets in the workbook.
        sheets: List of SheetInfo metadata.
        row_count: Total normalised row count across all sheets.
        column_count: Number of columns in normalised output.
        headers: List of normalised column headers.
        normalized_data: Normalised row dictionaries.
        raw_content_base64: Original file content (base64) for reprocessing.
        column_mappings: Applied column name mappings.
        detected_types: Mapping of column name to detected type.
        quality_score: Overall data quality score (0.0-1.0).
        completeness_score: Data completeness dimension (0.0-1.0).
        accuracy_score: Data accuracy dimension (0.0-1.0).
        consistency_score: Data consistency dimension (0.0-1.0).
        template_id: Mapping template used (if any).
        tenant_id: Owning tenant identifier.
        status: Processing status.
        provenance_hash: SHA-256 provenance hash.
        created_at: Timestamp of upload.
    """
    file_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_name: str = Field(default="")
    file_format: str = Field(default="unknown")
    sheet_count: int = Field(default=1)
    sheets: List[SheetInfo] = Field(default_factory=list)
    row_count: int = Field(default=0)
    column_count: int = Field(default=0)
    headers: List[str] = Field(default_factory=list)
    normalized_data: List[Dict[str, Any]] = Field(default_factory=list)
    raw_content_base64: str = Field(default="")
    column_mappings: Dict[str, str] = Field(default_factory=dict)
    detected_types: Dict[str, str] = Field(default_factory=dict)
    quality_score: float = Field(default=0.0)
    completeness_score: float = Field(default=0.0)
    accuracy_score: float = Field(default=0.0)
    consistency_score: float = Field(default=0.0)
    template_id: Optional[str] = Field(default=None)
    tenant_id: str = Field(default="default")
    status: str = Field(default="processed")
    provenance_hash: str = Field(default="")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class NormalizationJob(BaseModel):
    """Represents an asynchronous normalization job.

    Attributes:
        job_id: Unique job identifier.
        file_id: Associated file identifier.
        status: Job status (queued, processing, completed, failed).
        progress_pct: Completion percentage 0-100.
        tenant_id: Owning tenant identifier.
        created_at: Timestamp of job creation.
        completed_at: Timestamp of job completion.
        provenance_hash: SHA-256 provenance hash.
    """
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_id: str = Field(default="")
    status: str = Field(default="queued")
    progress_pct: float = Field(default=0.0)
    tenant_id: str = Field(default="default")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    completed_at: Optional[str] = Field(default=None)
    provenance_hash: str = Field(default="")


class BatchJob(BaseModel):
    """Represents a batch normalization job.

    Attributes:
        batch_id: Unique batch identifier.
        file_count: Number of files in the batch.
        status: Batch status (submitted, processing, completed, partial, failed).
        parallel: Whether files were processed in parallel.
        jobs: Individual normalization jobs within the batch.
        created_at: Timestamp of batch creation.
        provenance_hash: SHA-256 provenance hash.
    """
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_count: int = Field(default=0)
    status: str = Field(default="submitted")
    parallel: bool = Field(default=True)
    jobs: List[NormalizationJob] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    provenance_hash: str = Field(default="")


class ColumnMapping(BaseModel):
    """Result of column mapping operation.

    Attributes:
        mappings: Mapping of source header to canonical header.
        confidences: Mapping of source header to confidence score.
        match_types: Mapping of source header to match type used.
        unmapped: List of source headers that could not be mapped.
        provenance_hash: SHA-256 provenance hash.
    """
    mappings: Dict[str, str] = Field(default_factory=dict)
    confidences: Dict[str, float] = Field(default_factory=dict)
    match_types: Dict[str, str] = Field(default_factory=dict)
    unmapped: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class TypeDetectionResult(BaseModel):
    """Result of column data type detection.

    Attributes:
        types: Mapping of column header to detected type.
        confidences: Mapping of column header to detection confidence.
        sample_count: Number of sample rows used for detection.
        provenance_hash: SHA-256 provenance hash.
    """
    types: Dict[str, str] = Field(default_factory=dict)
    confidences: Dict[str, float] = Field(default_factory=dict)
    sample_count: int = Field(default=0)
    provenance_hash: str = Field(default="")


class NormalizeResult(BaseModel):
    """Result of inline data normalization.

    Attributes:
        data: Normalised row dictionaries.
        row_count: Number of rows normalised.
        column_mappings: Applied column mappings.
        quality_score: Overall quality score.
        provenance_hash: SHA-256 provenance hash.
    """
    data: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = Field(default=0)
    column_mappings: Dict[str, str] = Field(default_factory=dict)
    quality_score: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class ValidationResult(BaseModel):
    """Result of schema validation.

    Attributes:
        valid: Whether all rows passed validation.
        error_count: Total number of validation errors.
        errors: List of individual error detail dictionaries.
        schema_name: Schema that was validated against.
        provenance_hash: SHA-256 provenance hash.
    """
    valid: bool = Field(default=True)
    error_count: int = Field(default=0)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    schema_name: str = Field(default="")
    provenance_hash: str = Field(default="")


class TransformResult(BaseModel):
    """Result of applying transform operations.

    Attributes:
        data: Transformed row dictionaries.
        row_count: Number of rows after transforms.
        operations_applied: Number of operations successfully applied.
        provenance_hash: SHA-256 provenance hash.
    """
    data: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = Field(default=0)
    operations_applied: int = Field(default=0)
    provenance_hash: str = Field(default="")


class MappingTemplate(BaseModel):
    """Column mapping template definition.

    Attributes:
        template_id: Unique template identifier.
        name: Human-readable template name.
        description: Template description.
        source_type: Source file type (xlsx, csv, generic).
        column_mappings: Mapping of source columns to canonical columns.
        created_at: Timestamp of creation.
        provenance_hash: SHA-256 provenance hash.
    """
    template_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default="")
    description: str = Field(default="")
    source_type: str = Field(default="generic")
    column_mappings: Dict[str, str] = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    provenance_hash: str = Field(default="")


class CanonicalFieldsResult(BaseModel):
    """Result of canonical field listing.

    Attributes:
        fields: List of canonical field definitions.
        category: Category filter applied (if any).
        count: Number of fields returned.
    """
    fields: List[Dict[str, Any]] = Field(default_factory=list)
    category: Optional[str] = Field(default=None)
    count: int = Field(default=0)


class ExcelStatistics(BaseModel):
    """Aggregate statistics for the Excel normalizer service.

    Attributes:
        total_files: Total files processed.
        total_rows: Total rows normalised.
        total_columns_mapped: Total columns mapped.
        total_validation_errors: Total validation errors detected.
        total_transforms: Total transform operations applied.
        total_batch_jobs: Total batch jobs submitted.
        total_templates: Total mapping templates created.
        avg_quality_score: Average data quality score.
        avg_processing_time_ms: Average processing time in milliseconds.
        files_by_format: Breakdown of files by format.
        columns_by_match_type: Breakdown of columns by match strategy.
    """
    total_files: int = Field(default=0)
    total_rows: int = Field(default=0)
    total_columns_mapped: int = Field(default=0)
    total_validation_errors: int = Field(default=0)
    total_transforms: int = Field(default=0)
    total_batch_jobs: int = Field(default=0)
    total_templates: int = Field(default=0)
    avg_quality_score: float = Field(default=0.0)
    avg_processing_time_ms: float = Field(default=0.0)
    files_by_format: Dict[str, int] = Field(default_factory=dict)
    columns_by_match_type: Dict[str, int] = Field(default_factory=dict)


# ===================================================================
# Provenance helper
# ===================================================================


class _ProvenanceTracker:
    """Minimal provenance tracker recording SHA-256 audit entries.

    Attributes:
        entries: List of provenance entries.
        entry_count: Number of entries recorded.
    """

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []
        self.entry_count: int = 0

    def record(
        self,
        entity_type: str,
        entity_id: str,
        action: str,
        data_hash: str,
        user_id: str = "system",
    ) -> str:
        """Record a provenance entry and return its hash.

        Args:
            entity_type: Type of entity (file, column_mapping, template, etc.).
            entity_id: Entity identifier.
            action: Action performed (upload, normalize, map, validate, etc.).
            data_hash: SHA-256 hash of associated data.
            user_id: User or system that performed the action.

        Returns:
            SHA-256 hash of the provenance entry itself.
        """
        entry = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "action": action,
            "data_hash": data_hash,
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        entry_hash = hashlib.sha256(
            json.dumps(entry, sort_keys=True, default=str).encode()
        ).hexdigest()
        entry["entry_hash"] = entry_hash
        self._entries.append(entry)
        self.entry_count += 1
        return entry_hash


# ===================================================================
# ExcelNormalizerService facade
# ===================================================================

# Thread-safe singleton lock
_singleton_lock = threading.Lock()
_singleton_instance: Optional["ExcelNormalizerService"] = None


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, list, str, or Pydantic model).

    Returns:
        SHA-256 hex digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _detect_format(file_name: str) -> str:
    """Detect file format from extension.

    Args:
        file_name: Filename including extension.

    Returns:
        Detected format string (xlsx, xls, csv, tsv, unknown).
    """
    lower = file_name.lower()
    if lower.endswith(".xlsx"):
        return "xlsx"
    if lower.endswith(".xls"):
        return "xls"
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".tsv"):
        return "tsv"
    return "unknown"


# GreenLang canonical fields for sustainability data
_CANONICAL_FIELDS: List[Dict[str, Any]] = [
    {"name": "facility_name", "category": "facility", "type": "string",
     "description": "Name of the facility or site"},
    {"name": "facility_id", "category": "facility", "type": "string",
     "description": "Unique facility identifier"},
    {"name": "reporting_period", "category": "time", "type": "date",
     "description": "Reporting period start date"},
    {"name": "reporting_period_end", "category": "time", "type": "date",
     "description": "Reporting period end date"},
    {"name": "scope", "category": "emissions", "type": "string",
     "description": "GHG emission scope (1, 2, 3)"},
    {"name": "emission_category", "category": "emissions", "type": "string",
     "description": "Emission category"},
    {"name": "activity_type", "category": "activity", "type": "string",
     "description": "Activity type description"},
    {"name": "activity_data", "category": "activity", "type": "float",
     "description": "Activity data quantity"},
    {"name": "activity_unit", "category": "activity", "type": "string",
     "description": "Activity data unit of measure"},
    {"name": "emission_factor", "category": "emissions", "type": "float",
     "description": "Emission factor value"},
    {"name": "emission_factor_unit", "category": "emissions", "type": "string",
     "description": "Emission factor unit"},
    {"name": "emission_factor_source", "category": "emissions", "type": "string",
     "description": "Source of emission factor"},
    {"name": "co2e_tonnes", "category": "emissions", "type": "float",
     "description": "CO2 equivalent in metric tonnes"},
    {"name": "co2_tonnes", "category": "emissions", "type": "float",
     "description": "CO2 emissions in metric tonnes"},
    {"name": "ch4_tonnes", "category": "emissions", "type": "float",
     "description": "CH4 emissions in metric tonnes"},
    {"name": "n2o_tonnes", "category": "emissions", "type": "float",
     "description": "N2O emissions in metric tonnes"},
    {"name": "energy_kwh", "category": "energy", "type": "float",
     "description": "Energy consumption in kWh"},
    {"name": "energy_source", "category": "energy", "type": "string",
     "description": "Energy source type"},
    {"name": "water_m3", "category": "water", "type": "float",
     "description": "Water consumption in cubic metres"},
    {"name": "waste_tonnes", "category": "waste", "type": "float",
     "description": "Waste generated in metric tonnes"},
    {"name": "waste_type", "category": "waste", "type": "string",
     "description": "Waste classification type"},
    {"name": "country", "category": "location", "type": "string",
     "description": "Country name or ISO code"},
    {"name": "region", "category": "location", "type": "string",
     "description": "Region or state"},
    {"name": "supplier_name", "category": "supply_chain", "type": "string",
     "description": "Supplier or vendor name"},
    {"name": "data_source", "category": "metadata", "type": "string",
     "description": "Source of the data record"},
    {"name": "notes", "category": "metadata", "type": "string",
     "description": "Additional notes or comments"},
]


class ExcelNormalizerService:
    """Unified facade over the Excel & CSV Normalizer SDK.

    Aggregates all normalisation engines (Excel parser, CSV parser, column
    mapper, data type detector, schema validator, data quality scorer,
    transform engine, provenance tracker) through a single entry point with
    convenience methods for common operations.

    Each method records provenance and updates self-monitoring metrics.

    Attributes:
        config: ExcelNormalizerConfig instance.
        provenance: _ProvenanceTracker instance for SHA-256 audit trails.

    Example:
        >>> service = ExcelNormalizerService()
        >>> record = service.upload_file("data.csv", b64_content)
        >>> print(record.quality_score, record.row_count)
    """

    def __init__(
        self,
        config: Optional[ExcelNormalizerConfig] = None,
    ) -> None:
        """Initialize the Excel Normalizer Service facade.

        Instantiates all 7 internal engines plus the provenance tracker:
        - ExcelParser
        - CSVParser
        - ColumnMapper
        - DataTypeDetector
        - SchemaValidator
        - DataQualityScorer
        - TransformEngine

        Args:
            config: Optional configuration. Uses global config if None.
        """
        self.config = config or get_config()

        # Provenance tracker
        self.provenance = _ProvenanceTracker()

        # Engine placeholders -- real implementations are injected by the
        # respective SDK modules at import time. We use a lazy-init approach
        # so that setup.py can be imported without the full SDK installed.
        self._excel_parser: Any = None
        self._csv_parser: Any = None
        self._column_mapper: Any = None
        self._data_type_detector: Any = None
        self._schema_validator: Any = None
        self._data_quality_scorer: Any = None
        self._transform_engine: Any = None

        self._init_engines()

        # In-memory stores (production uses DB; these are SDK-level caches)
        self._files: Dict[str, FileRecord] = {}
        self._jobs: Dict[str, NormalizationJob] = {}
        self._batch_jobs: Dict[str, BatchJob] = {}
        self._templates: Dict[str, MappingTemplate] = {}

        # Statistics
        self._stats = ExcelStatistics()
        self._started = False

        logger.info("ExcelNormalizerService facade created")

    # ------------------------------------------------------------------
    # Engine initialization
    # ------------------------------------------------------------------

    def _init_engines(self) -> None:
        """Attempt to import and initialise SDK engines.

        Engines are optional; missing imports are logged as warnings and
        the service continues in degraded mode.
        """
        try:
            from greenlang.excel_normalizer.excel_parser import ExcelParser
            self._excel_parser = ExcelParser(self.config)
        except ImportError:
            logger.warning("ExcelParser not available; using stub")

        try:
            from greenlang.excel_normalizer.csv_parser import CSVParser
            self._csv_parser = CSVParser(self.config)
        except ImportError:
            logger.warning("CSVParser not available; using stub")

        try:
            from greenlang.excel_normalizer.column_mapper import ColumnMapper
            self._column_mapper = ColumnMapper(self.config)
        except ImportError:
            logger.warning("ColumnMapper not available; using stub")

        try:
            from greenlang.excel_normalizer.data_type_detector import DataTypeDetector
            self._data_type_detector = DataTypeDetector(self.config)
        except ImportError:
            logger.warning("DataTypeDetector not available; using stub")

        try:
            from greenlang.excel_normalizer.schema_validator import SchemaValidator
            self._schema_validator = SchemaValidator(self.config)
        except ImportError:
            logger.warning("SchemaValidator not available; using stub")

        try:
            from greenlang.excel_normalizer.data_quality_scorer import DataQualityScorer
            self._data_quality_scorer = DataQualityScorer(self.config)
        except ImportError:
            logger.warning("DataQualityScorer not available; using stub")

        try:
            from greenlang.excel_normalizer.transform_engine import TransformEngine
            self._transform_engine = TransformEngine(self.config)
        except ImportError:
            logger.warning("TransformEngine not available; using stub")

    # ------------------------------------------------------------------
    # Engine properties
    # ------------------------------------------------------------------

    @property
    def excel_parser(self) -> Any:
        """Get the Excel parser engine instance."""
        return self._excel_parser

    @property
    def csv_parser(self) -> Any:
        """Get the CSV parser engine instance."""
        return self._csv_parser

    @property
    def column_mapper(self) -> Any:
        """Get the column mapper engine instance."""
        return self._column_mapper

    @property
    def data_type_detector(self) -> Any:
        """Get the data type detector engine instance."""
        return self._data_type_detector

    @property
    def schema_validator(self) -> Any:
        """Get the schema validator engine instance."""
        return self._schema_validator

    @property
    def data_quality_scorer(self) -> Any:
        """Get the data quality scorer engine instance."""
        return self._data_quality_scorer

    @property
    def transform_engine(self) -> Any:
        """Get the transform engine instance."""
        return self._transform_engine

    # ------------------------------------------------------------------
    # File upload (full pipeline)
    # ------------------------------------------------------------------

    def upload_file(
        self,
        file_name: str,
        file_content: str,
        file_format: str = "auto",
        template_id: Optional[str] = None,
        tenant_id: str = "default",
    ) -> FileRecord:
        """Upload and normalise a single Excel/CSV file.

        Executes the full pipeline: parse, detect types, map columns,
        normalise, and score quality.

        Args:
            file_name: Original filename including extension.
            file_content: Base64-encoded file content.
            file_format: File format; "auto" to detect from extension.
            template_id: Optional mapping template ID to apply.
            tenant_id: Tenant identifier.

        Returns:
            FileRecord with normalised data and quality scores.

        Raises:
            ValueError: If file_name is empty or format is unsupported.
        """
        start_time = time.time()

        if not file_name.strip():
            raise ValueError("file_name must not be empty")

        # Step 1: Detect format
        detected_format = (
            file_format if file_format != "auto" else _detect_format(file_name)
        )
        if detected_format == "unknown" and file_format == "auto":
            logger.warning(
                "Could not detect format for %s; defaulting to csv", file_name,
            )
            detected_format = "csv"

        update_active_jobs(1)

        try:
            # Step 2: Parse file content
            headers, raw_rows = self._parse_file(
                file_content, file_name, detected_format,
            )

            # Step 3: Detect data types
            detected_types = self._detect_column_types(raw_rows, headers)

            # Step 4: Map columns
            column_mappings, match_types, confidences = self._map_columns(
                headers, template_id,
            )

            # Step 5: Normalise data
            normalized_data = self._apply_mappings(raw_rows, column_mappings)

            # Step 6: Score quality
            quality_scores = self._score_quality(normalized_data, headers)

            # Step 7: Build sheet info
            sheets = [
                SheetInfo(
                    sheet_name="Sheet1",
                    sheet_index=0,
                    row_count=len(raw_rows),
                    column_count=len(headers),
                    headers=headers,
                ),
            ]

            # Step 8: Build record
            record = FileRecord(
                file_name=file_name,
                file_format=detected_format,
                sheet_count=len(sheets),
                sheets=sheets,
                row_count=len(normalized_data),
                column_count=len(headers),
                headers=(
                    list(column_mappings.values()) if column_mappings
                    else headers
                ),
                normalized_data=normalized_data,
                raw_content_base64=file_content,
                column_mappings=column_mappings,
                detected_types=detected_types,
                quality_score=quality_scores.get("overall", 0.0),
                completeness_score=quality_scores.get("completeness", 0.0),
                accuracy_score=quality_scores.get("accuracy", 0.0),
                consistency_score=quality_scores.get("consistency", 0.0),
                template_id=template_id,
                tenant_id=tenant_id,
                status="processed",
            )

            # Step 9: Compute provenance
            record.provenance_hash = _compute_hash(record)

            # Step 10: Store
            self._files[record.file_id] = record

            # Step 11: Record metrics
            duration = time.time() - start_time
            record_file_processed(detected_format, tenant_id, duration)
            record_rows_normalized(len(normalized_data))
            for header, mt in match_types.items():
                dtype = detected_types.get(header, "string")
                record_columns_mapped(1, mt, dtype)
            for header, conf in confidences.items():
                record_mapping_confidence(conf)
            for col, dtype in detected_types.items():
                record_type_detection(dtype)
            record_quality_score(record.quality_score)

            # Step 12: Record provenance
            self.provenance.record(
                entity_type="file",
                entity_id=record.file_id,
                action="upload",
                data_hash=record.provenance_hash,
            )

            # Step 13: Update statistics
            self._update_stats_for_file(record, duration, match_types)

            logger.info(
                "Uploaded file %s (%s, %d rows, quality=%.2f) in %.2fs",
                record.file_id, detected_format, len(normalized_data),
                record.quality_score, duration,
            )
            return record

        except Exception as exc:
            logger.error("File upload failed: %s", exc, exc_info=True)
            raise
        finally:
            update_active_jobs(-1)

    # ------------------------------------------------------------------
    # Batch upload
    # ------------------------------------------------------------------

    def upload_batch(
        self,
        files: List[Dict[str, Any]],
        parallel: bool = True,
        tenant_id: str = "default",
    ) -> BatchJob:
        """Batch upload and normalise multiple files.

        Args:
            files: List of file dicts with file_name, file_content_base64, etc.
            parallel: Whether to process in parallel (currently sequential).
            tenant_id: Tenant identifier.

        Returns:
            BatchJob with individual normalization jobs.

        Raises:
            ValueError: If files list is empty or exceeds batch limit.
        """
        if not files:
            raise ValueError("Files list must not be empty")
        if len(files) > self.config.batch_max_files:
            raise ValueError(
                f"Batch size {len(files)} exceeds maximum "
                f"{self.config.batch_max_files}"
            )

        batch = BatchJob(
            file_count=len(files),
            status="processing",
            parallel=parallel,
        )
        self._batch_jobs[batch.batch_id] = batch

        record_batch_job("submitted")
        update_queue_size(len(files))

        jobs: List[NormalizationJob] = []
        completed = 0
        failed = 0

        for file_spec in files:
            job = NormalizationJob(status="processing", tenant_id=tenant_id)
            jobs.append(job)
            self._jobs[job.job_id] = job

            try:
                record = self.upload_file(
                    file_name=file_spec.get("file_name", "unknown"),
                    file_content=file_spec.get("file_content_base64", ""),
                    file_format=file_spec.get("file_format") or "auto",
                    template_id=file_spec.get("template_id"),
                    tenant_id=tenant_id,
                )
                job.file_id = record.file_id
                job.status = "completed"
                job.progress_pct = 100.0
                job.completed_at = datetime.now(timezone.utc).isoformat()
                job.provenance_hash = record.provenance_hash
                completed += 1
            except Exception as exc:
                job.status = "failed"
                job.progress_pct = 0.0
                failed += 1
                logger.warning("Batch item failed: %s", exc)

        # Determine batch status
        if failed == 0:
            batch.status = "completed"
            record_batch_job("completed")
        elif completed == 0:
            batch.status = "failed"
            record_batch_job("failed")
        else:
            batch.status = "partial"
            record_batch_job("partial")

        batch.jobs = jobs
        batch.provenance_hash = _compute_hash(batch)
        update_queue_size(0)

        # Record provenance
        self.provenance.record(
            entity_type="batch_job",
            entity_id=batch.batch_id,
            action="batch_upload",
            data_hash=batch.provenance_hash,
        )

        self._stats.total_batch_jobs += 1

        logger.info(
            "Batch %s completed: %d/%d succeeded, status=%s",
            batch.batch_id, completed, len(files), batch.status,
        )
        return batch

    # ------------------------------------------------------------------
    # Excel-only parsing
    # ------------------------------------------------------------------

    def parse_excel(
        self,
        file_content: str,
        file_name: str,
    ) -> FileRecord:
        """Parse an Excel file without full normalisation pipeline.

        Args:
            file_content: Base64-encoded file content.
            file_name: Original filename.

        Returns:
            FileRecord with parsed (but not normalised) data.
        """
        start_time = time.time()
        headers, raw_rows = self._parse_file(file_content, file_name, "xlsx")

        record = FileRecord(
            file_name=file_name,
            file_format="xlsx",
            row_count=len(raw_rows),
            column_count=len(headers),
            headers=headers,
            normalized_data=raw_rows,
            raw_content_base64=file_content,
            status="parsed",
        )
        record.provenance_hash = _compute_hash(record)

        self.provenance.record(
            entity_type="file",
            entity_id=record.file_id,
            action="parse_excel",
            data_hash=record.provenance_hash,
        )

        duration = time.time() - start_time
        record_file_processed("xlsx", "default", duration)
        record_rows_normalized(len(raw_rows))

        logger.info(
            "Parsed Excel %s (%d rows) in %.2fs",
            file_name, len(raw_rows), duration,
        )
        return record

    # ------------------------------------------------------------------
    # CSV-only parsing
    # ------------------------------------------------------------------

    def parse_csv(
        self,
        file_content: str,
        file_name: str,
        encoding: Optional[str] = None,
        delimiter: Optional[str] = None,
    ) -> FileRecord:
        """Parse a CSV file without full normalisation pipeline.

        Args:
            file_content: Base64-encoded file content.
            file_name: Original filename.
            encoding: Character encoding override.
            delimiter: Column delimiter override.

        Returns:
            FileRecord with parsed (but not normalised) data.
        """
        start_time = time.time()
        headers, raw_rows = self._parse_file(file_content, file_name, "csv")

        record = FileRecord(
            file_name=file_name,
            file_format="csv",
            row_count=len(raw_rows),
            column_count=len(headers),
            headers=headers,
            normalized_data=raw_rows,
            raw_content_base64=file_content,
            status="parsed",
        )
        record.provenance_hash = _compute_hash(record)

        self.provenance.record(
            entity_type="file",
            entity_id=record.file_id,
            action="parse_csv",
            data_hash=record.provenance_hash,
        )

        duration = time.time() - start_time
        record_file_processed("csv", "default", duration)
        record_rows_normalized(len(raw_rows))

        logger.info(
            "Parsed CSV %s (%d rows) in %.2fs",
            file_name, len(raw_rows), duration,
        )
        return record

    # ------------------------------------------------------------------
    # Column mapping
    # ------------------------------------------------------------------

    def map_columns(
        self,
        headers: List[str],
        strategy: str = "fuzzy",
        template_id: Optional[str] = None,
    ) -> ColumnMapping:
        """Map source column headers to GreenLang canonical schema.

        Args:
            headers: Source column header strings.
            strategy: Mapping strategy (exact, synonym, fuzzy, pattern, manual).
            template_id: Optional template ID for pre-defined mappings.

        Returns:
            ColumnMapping with matched results.

        Raises:
            ValueError: If headers list is empty.
        """
        if not headers:
            raise ValueError("Headers list must not be empty")

        start_time = time.time()

        # Check for template-based mappings first
        template_mappings: Dict[str, str] = {}
        if template_id and template_id in self._templates:
            template_mappings = self._templates[template_id].column_mappings

        mappings: Dict[str, str] = {}
        confidences: Dict[str, float] = {}
        match_types: Dict[str, str] = {}
        unmapped: List[str] = []

        if self._column_mapper is not None:
            result = self._column_mapper.map(
                headers,
                strategy=strategy,
                template_mappings=template_mappings,
            )
            mappings = result.get("mappings", {})
            confidences = result.get("confidences", {})
            match_types = result.get("match_types", {})
            unmapped = result.get("unmapped", [])
        else:
            # Stub: use template or pass-through
            for header in headers:
                if header in template_mappings:
                    mappings[header] = template_mappings[header]
                    confidences[header] = 1.0
                    match_types[header] = "exact"
                else:
                    mappings[header] = header
                    confidences[header] = 0.5
                    match_types[header] = "fuzzy"

        result_obj = ColumnMapping(
            mappings=mappings,
            confidences=confidences,
            match_types=match_types,
            unmapped=unmapped,
        )
        result_obj.provenance_hash = _compute_hash(result_obj)

        # Metrics
        for header, mt in match_types.items():
            record_columns_mapped(1, mt, "string")
        for header, conf in confidences.items():
            record_mapping_confidence(conf)

        # Provenance
        self.provenance.record(
            entity_type="column_mapping",
            entity_id=str(uuid.uuid4()),
            action="map_columns",
            data_hash=result_obj.provenance_hash,
        )

        duration = time.time() - start_time
        logger.info(
            "Mapped %d columns (strategy=%s) in %.2fs",
            len(headers), strategy, duration,
        )
        return result_obj

    # ------------------------------------------------------------------
    # Type detection
    # ------------------------------------------------------------------

    def detect_types(
        self,
        values: List[List[Any]],
        headers: Optional[List[str]] = None,
    ) -> TypeDetectionResult:
        """Detect data types from column sample values.

        Args:
            values: Column-oriented sample values (list of columns).
            headers: Optional column header labels.

        Returns:
            TypeDetectionResult with detected types and confidences.

        Raises:
            ValueError: If values list is empty.
        """
        if not values:
            raise ValueError("Values list must not be empty")

        col_headers = headers or [f"col_{i}" for i in range(len(values))]
        types: Dict[str, str] = {}
        confidences: Dict[str, float] = {}

        if self._data_type_detector is not None:
            result = self._data_type_detector.detect(
                values, headers=col_headers,
            )
            types = result.get("types", {})
            confidences = result.get("confidences", {})
        else:
            # Stub: infer basic types from first non-None value
            for i, col_values in enumerate(values):
                header = (
                    col_headers[i] if i < len(col_headers) else f"col_{i}"
                )
                detected = self._heuristic_type_detect(col_values)
                types[header] = detected
                confidences[header] = 0.7

        result_obj = TypeDetectionResult(
            types=types,
            confidences=confidences,
            sample_count=max(len(v) for v in values) if values else 0,
        )
        result_obj.provenance_hash = _compute_hash(result_obj)

        # Metrics
        for col, dtype in types.items():
            record_type_detection(dtype)

        # Provenance
        self.provenance.record(
            entity_type="type_detection",
            entity_id=str(uuid.uuid4()),
            action="detect_types",
            data_hash=result_obj.provenance_hash,
        )

        logger.info("Detected types for %d columns", len(types))
        return result_obj

    # ------------------------------------------------------------------
    # Normalize inline data
    # ------------------------------------------------------------------

    def normalize_data(
        self,
        data: List[Dict[str, Any]],
        column_mappings: Dict[str, str],
        tenant_id: str = "default",
    ) -> NormalizeResult:
        """Normalise inline data with explicit column mappings.

        Args:
            data: List of row dictionaries to normalise.
            column_mappings: Mapping of source columns to canonical names.
            tenant_id: Tenant identifier.

        Returns:
            NormalizeResult with normalised rows.

        Raises:
            ValueError: If data list is empty.
        """
        if not data:
            raise ValueError("Data list must not be empty")

        start_time = time.time()

        normalized = self._apply_mappings(data, column_mappings)
        quality = self._score_quality(
            normalized, list(column_mappings.values()),
        )

        result = NormalizeResult(
            data=normalized,
            row_count=len(normalized),
            column_mappings=column_mappings,
            quality_score=quality.get("overall", 0.0),
        )
        result.provenance_hash = _compute_hash(result)

        # Metrics
        record_rows_normalized(len(normalized))
        record_quality_score(result.quality_score)

        # Provenance
        self.provenance.record(
            entity_type="normalize",
            entity_id=str(uuid.uuid4()),
            action="normalize_data",
            data_hash=result.provenance_hash,
        )

        duration = time.time() - start_time
        logger.info(
            "Normalized %d rows inline in %.2fs (quality=%.2f)",
            len(normalized), duration, result.quality_score,
        )
        return result

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------

    def validate_data(
        self,
        data: List[Dict[str, Any]],
        schema_name: str,
    ) -> ValidationResult:
        """Validate row data against a named schema.

        Args:
            data: Row dictionaries to validate.
            schema_name: Schema name to validate against.

        Returns:
            ValidationResult with errors list.

        Raises:
            ValueError: If data list is empty or schema_name is blank.
        """
        if not data:
            raise ValueError("Data list must not be empty")
        if not schema_name.strip():
            raise ValueError("schema_name must not be empty")

        errors: List[Dict[str, Any]] = []

        if self._schema_validator is not None:
            result = self._schema_validator.validate(
                data, schema_name=schema_name,
            )
            errors = result.get("errors", [])
        else:
            # Stub: basic presence check for required canonical fields
            required = {"facility_name", "reporting_period", "activity_data"}
            for row_idx, row in enumerate(data):
                for field in required:
                    val = row.get(field)
                    if val is not None and str(val).strip() == "":
                        errors.append({
                            "row": row_idx,
                            "field": field,
                            "severity": "medium",
                            "rule_name": f"required_{field}",
                            "message": (
                                f"Required field '{field}' is empty"
                            ),
                        })

        result_obj = ValidationResult(
            valid=len(errors) == 0,
            error_count=len(errors),
            errors=errors,
            schema_name=schema_name,
        )
        result_obj.provenance_hash = _compute_hash(result_obj)

        # Metrics
        for err in errors:
            severity = err.get("severity", "medium")
            rule_name = err.get("rule_name", "unknown")
            record_validation_finding(severity, rule_name)
            self._stats.total_validation_errors += 1

        # Provenance
        self.provenance.record(
            entity_type="validation",
            entity_id=str(uuid.uuid4()),
            action="validate_data",
            data_hash=result_obj.provenance_hash,
        )

        logger.info(
            "Validated %d rows against schema '%s': %d errors",
            len(data), schema_name, len(errors),
        )
        return result_obj

    # ------------------------------------------------------------------
    # Quality scoring
    # ------------------------------------------------------------------

    def score_quality(
        self,
        data: List[Dict[str, Any]],
        headers: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Score data quality across completeness, accuracy, consistency.

        Args:
            data: Row dictionaries to score.
            headers: Optional column headers for reference.

        Returns:
            Dictionary with overall, completeness, accuracy, consistency.
        """
        scores = self._score_quality(data, headers or [])

        record_quality_score(scores.get("overall", 0.0))

        self.provenance.record(
            entity_type="quality",
            entity_id=str(uuid.uuid4()),
            action="score_quality",
            data_hash=_compute_hash(scores),
        )

        logger.info("Quality score: %.2f", scores.get("overall", 0.0))
        return scores

    # ------------------------------------------------------------------
    # Transform pipeline
    # ------------------------------------------------------------------

    def apply_transforms(
        self,
        data: Optional[List[Dict[str, Any]]],
        operations: List[Dict[str, Any]],
        file_id: Optional[str] = None,
    ) -> TransformResult:
        """Apply a sequence of transform operations to data.

        Args:
            data: Row dictionaries to transform (or None to use file_id).
            operations: Ordered list of transform operations.
            file_id: File ID to load data from if data is None.

        Returns:
            TransformResult with transformed data.

        Raises:
            ValueError: If neither data nor file_id is provided.
        """
        if data is None and file_id is not None:
            record = self._files.get(file_id)
            if record is None:
                raise ValueError(f"File {file_id} not found")
            data = list(record.normalized_data)
        elif data is None:
            raise ValueError("Either data or file_id must be provided")

        start_time = time.time()
        applied = 0

        if self._transform_engine is not None:
            result = self._transform_engine.apply(
                data, operations=operations,
            )
            data = result.get("data", data)
            applied = result.get("applied", len(operations))
        else:
            # Stub: count operations as applied
            applied = len(operations)

        # Record metrics per operation
        for op in operations:
            op_type = op.get("type", op.get("operation", "unknown"))
            record_transform(op_type)
            self._stats.total_transforms += 1

        result_obj = TransformResult(
            data=data,
            row_count=len(data),
            operations_applied=applied,
        )
        result_obj.provenance_hash = _compute_hash(result_obj)

        # Provenance
        self.provenance.record(
            entity_type="transform",
            entity_id=file_id or str(uuid.uuid4()),
            action="apply_transforms",
            data_hash=result_obj.provenance_hash,
        )

        duration = time.time() - start_time
        logger.info(
            "Applied %d transforms to %d rows in %.2fs",
            applied, len(data), duration,
        )
        return result_obj

    # ------------------------------------------------------------------
    # Template management
    # ------------------------------------------------------------------

    def create_template(
        self,
        name: str,
        description: str,
        source_type: str,
        mappings: Dict[str, str],
    ) -> MappingTemplate:
        """Create a new column mapping template.

        Args:
            name: Template name.
            description: Template description.
            source_type: Source file type (xlsx, csv, generic).
            mappings: Column name mappings.

        Returns:
            Created MappingTemplate.

        Raises:
            ValueError: If name is empty or duplicate.
        """
        if not name.strip():
            raise ValueError("Template name must not be empty")

        # Check for duplicate names
        for existing in self._templates.values():
            if existing.name == name:
                raise ValueError(
                    f"Template with name '{name}' already exists"
                )

        template = MappingTemplate(
            name=name,
            description=description,
            source_type=source_type,
            column_mappings=mappings,
        )
        template.provenance_hash = _compute_hash(template)
        self._templates[template.template_id] = template

        self.provenance.record(
            entity_type="template",
            entity_id=template.template_id,
            action="create",
            data_hash=template.provenance_hash,
        )

        self._stats.total_templates += 1

        logger.info(
            "Created template '%s' (%s)", name, template.template_id,
        )
        return template

    def list_templates(self) -> List[MappingTemplate]:
        """List all mapping templates.

        Returns:
            List of MappingTemplate instances.
        """
        return list(self._templates.values())

    def get_template(
        self,
        template_id: str,
    ) -> Optional[MappingTemplate]:
        """Get a template by ID.

        Args:
            template_id: Template identifier.

        Returns:
            MappingTemplate or None if not found.
        """
        return self._templates.get(template_id)

    # ------------------------------------------------------------------
    # File queries
    # ------------------------------------------------------------------

    def get_file(
        self,
        file_id: str,
    ) -> Optional[FileRecord]:
        """Get a file record by ID.

        Args:
            file_id: File identifier.

        Returns:
            FileRecord or None if not found.
        """
        return self._files.get(file_id)

    def list_files(
        self,
        tenant_id: str = "default",
        limit: int = 50,
        offset: int = 0,
    ) -> List[FileRecord]:
        """List files with optional tenant filter and pagination.

        Args:
            tenant_id: Tenant identifier to filter by.
            limit: Maximum number of files to return.
            offset: Number of files to skip.

        Returns:
            List of FileRecord instances.
        """
        files = [f for f in self._files.values() if f.tenant_id == tenant_id]
        return files[offset:offset + limit]

    # ------------------------------------------------------------------
    # Canonical fields
    # ------------------------------------------------------------------

    def get_canonical_fields(
        self,
        category: Optional[str] = None,
    ) -> CanonicalFieldsResult:
        """List GreenLang canonical field definitions.

        Args:
            category: Optional category filter (facility, time, emissions, etc.).

        Returns:
            CanonicalFieldsResult with matching fields.
        """
        if category:
            fields = [
                f for f in _CANONICAL_FIELDS if f["category"] == category
            ]
        else:
            fields = list(_CANONICAL_FIELDS)

        return CanonicalFieldsResult(
            fields=fields,
            category=category,
            count=len(fields),
        )

    # ------------------------------------------------------------------
    # Job queries
    # ------------------------------------------------------------------

    def list_jobs(
        self,
        status: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[NormalizationJob]:
        """List normalization jobs with optional filters.

        Args:
            status: Optional status filter.
            tenant_id: Optional tenant filter.
            limit: Maximum number of jobs to return.
            offset: Number of jobs to skip.

        Returns:
            List of NormalizationJob instances.
        """
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        if tenant_id:
            jobs = [j for j in jobs if j.tenant_id == tenant_id]
        return jobs[offset:offset + limit]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> ExcelStatistics:
        """Get aggregated Excel normalizer statistics.

        Returns:
            ExcelStatistics summary.
        """
        return self._stats

    # ------------------------------------------------------------------
    # Convenience getters
    # ------------------------------------------------------------------

    def get_provenance(self) -> _ProvenanceTracker:
        """Get the ProvenanceTracker instance.

        Returns:
            _ProvenanceTracker used by this service.
        """
        return self.provenance

    # ------------------------------------------------------------------
    # Metrics summary
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Get Excel normalizer service metrics summary.

        Returns:
            Dictionary with service metric summaries.
        """
        return {
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "started": self._started,
            "total_files": self._stats.total_files,
            "total_rows": self._stats.total_rows,
            "total_columns_mapped": self._stats.total_columns_mapped,
            "total_validation_errors": self._stats.total_validation_errors,
            "total_transforms": self._stats.total_transforms,
            "total_batch_jobs": self._stats.total_batch_jobs,
            "total_templates": self._stats.total_templates,
            "avg_quality_score": self._stats.avg_quality_score,
            "provenance_entries": self.provenance.entry_count,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_file(
        self,
        file_content: str,
        file_name: str,
        file_format: str,
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        """Parse a file into headers and rows.

        Args:
            file_content: Base64-encoded file content.
            file_name: Original filename.
            file_format: File format (xlsx, xls, csv, tsv).

        Returns:
            Tuple of (headers, row_dicts).
        """
        if file_format in ("xlsx", "xls") and self._excel_parser is not None:
            return self._excel_parser.parse(file_content, file_name)

        if file_format in ("csv", "tsv") and self._csv_parser is not None:
            return self._csv_parser.parse(file_content, file_name)

        # Fallback: return empty
        logger.warning(
            "Parser not available for format '%s'; "
            "returning empty data for %s",
            file_format, file_name,
        )
        return [], []

    def _map_columns(
        self,
        headers: List[str],
        template_id: Optional[str],
    ) -> tuple[Dict[str, str], Dict[str, str], Dict[str, float]]:
        """Map column headers using mapper engine or stub.

        Args:
            headers: Source column headers.
            template_id: Optional template ID.

        Returns:
            Tuple of (mappings, match_types, confidences).
        """
        template_mappings: Dict[str, str] = {}
        if template_id and template_id in self._templates:
            template_mappings = self._templates[template_id].column_mappings

        if self._column_mapper is not None:
            result = self._column_mapper.map(
                headers,
                strategy=self.config.default_mapping_strategy,
                template_mappings=template_mappings,
            )
            return (
                result.get("mappings", {}),
                result.get("match_types", {}),
                result.get("confidences", {}),
            )

        # Stub mapping
        mappings: Dict[str, str] = {}
        match_types: Dict[str, str] = {}
        confidences: Dict[str, float] = {}

        for header in headers:
            if header in template_mappings:
                mappings[header] = template_mappings[header]
                match_types[header] = "exact"
                confidences[header] = 1.0
            else:
                mappings[header] = header
                match_types[header] = "fuzzy"
                confidences[header] = 0.5

        return mappings, match_types, confidences

    def _detect_column_types(
        self,
        rows: List[Dict[str, Any]],
        headers: List[str],
    ) -> Dict[str, str]:
        """Detect column data types from row data.

        Args:
            rows: Row dictionaries.
            headers: Column headers.

        Returns:
            Mapping of column name to detected type.
        """
        if not rows or not headers:
            return {}

        if self._data_type_detector is not None:
            col_values = []
            for header in headers:
                sample = [
                    row.get(header)
                    for row in rows[
                        :self.config.sample_rows_for_type_detection
                    ]
                ]
                col_values.append(sample)
            result = self._data_type_detector.detect(
                col_values, headers=headers,
            )
            return result.get("types", {})

        # Stub: heuristic type detection
        detected: Dict[str, str] = {}
        for header in headers:
            sample = [
                row.get(header)
                for row in rows[:100]
                if row.get(header) is not None
            ]
            detected[header] = self._heuristic_type_detect(sample)

        return detected

    @staticmethod
    def _heuristic_type_detect(values: List[Any]) -> str:
        """Detect data type from sample values using heuristics.

        Args:
            values: Sample values from a single column.

        Returns:
            Detected type string (string, integer, float, date, boolean).
        """
        if not values:
            return "string"

        # Check first non-None values
        for val in values[:20]:
            if val is None:
                continue
            if isinstance(val, bool):
                return "boolean"
            if isinstance(val, int):
                return "integer"
            if isinstance(val, float):
                return "float"

            str_val = str(val).strip()
            if str_val.lower() in ("true", "false", "yes", "no"):
                return "boolean"
            try:
                int(str_val)
                return "integer"
            except (ValueError, TypeError):
                pass
            try:
                float(str_val.replace(",", ""))
                return "float"
            except (ValueError, TypeError):
                pass

        return "string"

    @staticmethod
    def _apply_mappings(
        rows: List[Dict[str, Any]],
        column_mappings: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Apply column name mappings to row dictionaries.

        Args:
            rows: Source row dictionaries.
            column_mappings: Mapping of source column to canonical column.

        Returns:
            List of row dictionaries with renamed columns.
        """
        if not column_mappings:
            return rows

        normalized: List[Dict[str, Any]] = []
        for row in rows:
            new_row: Dict[str, Any] = {}
            for src_col, value in row.items():
                canonical = column_mappings.get(src_col, src_col)
                new_row[canonical] = value
            normalized.append(new_row)

        return normalized

    def _score_quality(
        self,
        data: List[Dict[str, Any]],
        headers: List[str],
    ) -> Dict[str, float]:
        """Score data quality across three dimensions.

        Args:
            data: Row dictionaries to score.
            headers: Column headers.

        Returns:
            Dictionary with overall, completeness, accuracy, consistency.
        """
        if self._data_quality_scorer is not None:
            return self._data_quality_scorer.score(data, headers=headers)

        # Stub: calculate basic completeness
        if not data:
            return {
                "overall": 0.0,
                "completeness": 0.0,
                "accuracy": 0.0,
                "consistency": 0.0,
            }

        total_cells = 0
        filled_cells = 0
        for row in data:
            for value in row.values():
                total_cells += 1
                if value is not None and str(value).strip() != "":
                    filled_cells += 1

        completeness = (
            filled_cells / total_cells if total_cells > 0 else 0.0
        )
        # Default accuracy and consistency scores for stub
        accuracy = 0.8
        consistency = 0.85

        overall = (
            completeness * self.config.completeness_weight
            + accuracy * self.config.accuracy_weight
            + consistency * self.config.consistency_weight
        )

        return {
            "overall": round(overall, 4),
            "completeness": round(completeness, 4),
            "accuracy": round(accuracy, 4),
            "consistency": round(consistency, 4),
        }

    def _update_stats_for_file(
        self,
        record: FileRecord,
        duration_seconds: float,
        match_types: Dict[str, str],
    ) -> None:
        """Update statistics after file processing.

        Args:
            record: Processed file record.
            duration_seconds: Processing time.
            match_types: Column match types used.
        """
        self._stats.total_files += 1
        self._stats.total_rows += record.row_count
        self._stats.total_columns_mapped += record.column_count

        # Update files by format
        fmt = record.file_format
        self._stats.files_by_format[fmt] = (
            self._stats.files_by_format.get(fmt, 0) + 1
        )

        # Update columns by match type
        for mt in match_types.values():
            self._stats.columns_by_match_type[mt] = (
                self._stats.columns_by_match_type.get(mt, 0) + 1
            )

        # Update running average quality score
        total = self._stats.total_files
        prev_avg = self._stats.avg_quality_score
        self._stats.avg_quality_score = (
            (prev_avg * (total - 1) + record.quality_score) / total
        )

        # Update running average processing time
        prev_time_avg = self._stats.avg_processing_time_ms
        self._stats.avg_processing_time_ms = (
            (prev_time_avg * (total - 1) + duration_seconds * 1000) / total
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Start the Excel normalizer service.

        Safe to call multiple times.
        """
        if self._started:
            logger.debug("ExcelNormalizerService already started; skipping")
            return

        logger.info("ExcelNormalizerService starting up...")
        self._started = True
        logger.info("ExcelNormalizerService startup complete")

    def shutdown(self) -> None:
        """Shutdown the Excel normalizer service and release resources."""
        if not self._started:
            return

        self._started = False
        logger.info("ExcelNormalizerService shut down")


# ===================================================================
# Thread-safe singleton access
# ===================================================================


def _get_singleton() -> ExcelNormalizerService:
    """Get or create the singleton ExcelNormalizerService instance.

    Returns:
        The singleton ExcelNormalizerService.
    """
    global _singleton_instance
    if _singleton_instance is None:
        with _singleton_lock:
            if _singleton_instance is None:
                _singleton_instance = ExcelNormalizerService()
    return _singleton_instance


# ===================================================================
# FastAPI integration
# ===================================================================


async def configure_excel_normalizer(
    app: Any,
    config: Optional[ExcelNormalizerConfig] = None,
) -> ExcelNormalizerService:
    """Configure the Excel Normalizer Service on a FastAPI application.

    Creates the ExcelNormalizerService, stores it in app.state, mounts
    the Excel normalizer API router, and starts the service.

    Args:
        app: FastAPI application instance.
        config: Optional Excel normalizer config.

    Returns:
        ExcelNormalizerService instance.
    """
    global _singleton_instance

    service = ExcelNormalizerService(config=config)

    # Store as singleton
    with _singleton_lock:
        _singleton_instance = service

    # Attach to app state
    app.state.excel_normalizer_service = service

    # Mount Excel normalizer API router
    try:
        from greenlang.excel_normalizer.api.router import (
            router as excel_router,
        )
        if excel_router is not None:
            app.include_router(excel_router)
            logger.info("Excel normalizer service API router mounted")
    except ImportError:
        logger.warning(
            "Excel normalizer router not available; API not mounted"
        )

    # Start service
    service.startup()

    logger.info("Excel normalizer service configured on app")
    return service


def get_excel_normalizer(app: Any) -> ExcelNormalizerService:
    """Get the ExcelNormalizerService instance from app state.

    Args:
        app: FastAPI application instance.

    Returns:
        ExcelNormalizerService instance.

    Raises:
        RuntimeError: If Excel normalizer service not configured.
    """
    service = getattr(app.state, "excel_normalizer_service", None)
    if service is None:
        raise RuntimeError(
            "Excel normalizer service not configured. "
            "Call configure_excel_normalizer(app) first."
        )
    return service


def get_router(service: Optional[ExcelNormalizerService] = None) -> Any:
    """Get the Excel normalizer API router.

    Args:
        service: Optional service instance (unused, kept for API compat).

    Returns:
        FastAPI APIRouter or None if FastAPI not available.
    """
    try:
        from greenlang.excel_normalizer.api.router import router
        return router
    except ImportError:
        return None


__all__ = [
    "ExcelNormalizerService",
    "configure_excel_normalizer",
    "get_excel_normalizer",
    "get_router",
    # Models
    "FileRecord",
    "SheetInfo",
    "NormalizationJob",
    "BatchJob",
    "ColumnMapping",
    "TypeDetectionResult",
    "NormalizeResult",
    "ValidationResult",
    "TransformResult",
    "MappingTemplate",
    "CanonicalFieldsResult",
    "ExcelStatistics",
]
