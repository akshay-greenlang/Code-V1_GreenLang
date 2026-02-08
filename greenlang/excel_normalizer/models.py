# -*- coding: utf-8 -*-
"""
Excel & CSV Normalizer Service Data Models - AGENT-DATA-002: Excel Normalizer

Pydantic v2 data models for the Excel & CSV Normalizer SDK. This agent has
no Layer 1 foundation agent, so all enumerations and models are defined
fresh here.

Enumerations:
    - FileFormat: Supported spreadsheet file formats
    - DelimiterType: CSV delimiter options
    - DataType: Detected column data types
    - MappingStrategy: Column mapping strategies
    - JobStatus: Normalization job lifecycle statuses
    - QualityLevel: Data quality tiers
    - TransformOperation: Supported data transformations

SDK Models:
    - SpreadsheetFile, SheetMetadata, ColumnMapping
    - NormalizationJob, NormalizedRecord, MappingTemplate
    - DataQualityReport, ValidationFinding, TransformResult
    - ExcelStatistics

Request Models:
    - UploadFileRequest, BatchUploadRequest, MapColumnsRequest
    - NormalizeRequest, TransformRequest, CreateTemplateRequest

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-002 Excel & CSV Normalizer
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# =============================================================================
# Enumerations
# =============================================================================


class FileFormat(str, Enum):
    """Supported spreadsheet and delimited file formats."""

    XLSX = "xlsx"
    XLS = "xls"
    CSV = "csv"
    TSV = "tsv"
    AUTO = "auto"


class DelimiterType(str, Enum):
    """Delimiter types for CSV/delimited file parsing."""

    COMMA = "comma"
    SEMICOLON = "semicolon"
    TAB = "tab"
    PIPE = "pipe"
    SPACE = "space"
    AUTO = "auto"


class DataType(str, Enum):
    """Detected data types for spreadsheet columns."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    DECIMAL = "decimal"
    DATE = "date"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    UNIT_VALUE = "unit_value"
    EMAIL = "email"
    URL = "url"
    EMPTY = "empty"
    UNKNOWN = "unknown"


class MappingStrategy(str, Enum):
    """Strategies for mapping source columns to canonical fields."""

    EXACT = "exact"
    FUZZY = "fuzzy"
    SYNONYM = "synonym"
    PATTERN = "pattern"
    MANUAL = "manual"


class JobStatus(str, Enum):
    """Lifecycle status of a normalization job."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QualityLevel(str, Enum):
    """Data quality tier based on overall quality score.

    Thresholds:
        - excellent: score >= 0.9
        - good: score >= 0.7
        - fair: score >= 0.5
        - poor: score < 0.5
    """

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class TransformOperation(str, Enum):
    """Supported data transformation operations."""

    PIVOT = "pivot"
    UNPIVOT = "unpivot"
    DEDUP = "dedup"
    MERGE = "merge"
    FILTER = "filter"
    AGGREGATE = "aggregate"
    RENAME = "rename"
    SPLIT = "split"
    CAST = "cast"
    FILL_MISSING = "fill_missing"


# =============================================================================
# SDK Data Models
# =============================================================================


class SpreadsheetFile(BaseModel):
    """Persistent record of an uploaded spreadsheet or CSV file.

    Captures metadata about a file that has been uploaded and registered
    in the normalization system for processing and audit purposes.

    Attributes:
        file_id: Unique identifier for this file record.
        file_name: Original file name of the uploaded file.
        file_path: Storage path (local or S3) for the file.
        file_format: Detected or specified file format.
        file_size_bytes: Size of the file in bytes.
        file_hash: SHA-256 hash of the raw file content for deduplication.
        sheet_count: Number of sheets in the workbook (1 for CSV/TSV).
        total_rows: Total number of data rows across all sheets.
        total_columns: Maximum number of columns across all sheets.
        upload_timestamp: Timestamp when the file was uploaded.
        tenant_id: Tenant identifier for multi-tenant isolation.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
        uploaded_by: User or system that uploaded the file.
    """

    file_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this file record",
    )
    file_name: str = Field(
        ..., description="Original file name of the uploaded file",
    )
    file_path: str = Field(
        default="", description="Storage path for the file",
    )
    file_format: FileFormat = Field(
        default=FileFormat.AUTO,
        description="Detected or specified file format",
    )
    file_size_bytes: int = Field(
        default=0, ge=0, description="Size of the file in bytes",
    )
    file_hash: str = Field(
        default="", description="SHA-256 hash of the raw file content",
    )
    sheet_count: int = Field(
        default=1, ge=1, description="Number of sheets in the workbook",
    )
    total_rows: int = Field(
        default=0, ge=0, description="Total number of data rows across all sheets",
    )
    total_columns: int = Field(
        default=0, ge=0,
        description="Maximum number of columns across all sheets",
    )
    upload_timestamp: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the file was uploaded",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )
    uploaded_by: str = Field(
        default="system",
        description="User or system that uploaded the file",
    )

    model_config = {"extra": "forbid"}

    @field_validator("file_name")
    @classmethod
    def validate_file_name(cls, v: str) -> str:
        """Validate file_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("file_name must be non-empty")
        return v


class SheetMetadata(BaseModel):
    """Metadata for a single sheet within a workbook.

    Captures structural information about a sheet including row/column
    counts, header detection, encoding, and delimiter for CSV files.

    Attributes:
        sheet_id: Unique identifier for this sheet metadata record.
        file_id: Parent file identifier.
        sheet_name: Name of the sheet within the workbook.
        sheet_index: Zero-based index of the sheet in the workbook.
        row_count: Number of data rows in the sheet.
        column_count: Number of columns in the sheet.
        header_row_index: Zero-based index of the header row.
        has_headers: Whether the sheet has a detected header row.
        detected_encoding: Detected character encoding for the sheet.
        detected_delimiter: Detected delimiter character for CSV sheets.
    """

    sheet_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this sheet metadata record",
    )
    file_id: str = Field(
        ..., description="Parent file identifier",
    )
    sheet_name: str = Field(
        default="Sheet1", description="Name of the sheet within the workbook",
    )
    sheet_index: int = Field(
        default=0, ge=0,
        description="Zero-based index of the sheet in the workbook",
    )
    row_count: int = Field(
        default=0, ge=0, description="Number of data rows in the sheet",
    )
    column_count: int = Field(
        default=0, ge=0, description="Number of columns in the sheet",
    )
    header_row_index: int = Field(
        default=0, ge=0,
        description="Zero-based index of the header row",
    )
    has_headers: bool = Field(
        default=True,
        description="Whether the sheet has a detected header row",
    )
    detected_encoding: Optional[str] = Field(
        None, description="Detected character encoding for the sheet",
    )
    detected_delimiter: Optional[str] = Field(
        None, description="Detected delimiter character for CSV sheets",
    )

    model_config = {"extra": "forbid"}

    @field_validator("file_id")
    @classmethod
    def validate_file_id(cls, v: str) -> str:
        """Validate file_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("file_id must be non-empty")
        return v


class ColumnMapping(BaseModel):
    """Mapping from a source column to a canonical field.

    Captures the mapping strategy, confidence score, detected data type,
    and any synonym matches used during column resolution.

    Attributes:
        mapping_id: Unique identifier for this column mapping.
        sheet_id: Parent sheet identifier.
        source_column: Original column name from the source file.
        source_index: Zero-based index of the source column.
        canonical_field: Target canonical field name.
        mapping_strategy: Strategy used to produce this mapping.
        confidence: Confidence score of the mapping (0.0 to 1.0).
        detected_data_type: Detected data type for the column values.
        detected_unit: Detected unit of measurement (if applicable).
        synonyms_matched: List of synonym strings that contributed to match.
    """

    mapping_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this column mapping",
    )
    sheet_id: str = Field(
        ..., description="Parent sheet identifier",
    )
    source_column: str = Field(
        ..., description="Original column name from the source file",
    )
    source_index: int = Field(
        default=0, ge=0,
        description="Zero-based index of the source column",
    )
    canonical_field: str = Field(
        default="", description="Target canonical field name",
    )
    mapping_strategy: MappingStrategy = Field(
        default=MappingStrategy.FUZZY,
        description="Strategy used to produce this mapping",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence score of the mapping",
    )
    detected_data_type: DataType = Field(
        default=DataType.UNKNOWN,
        description="Detected data type for the column values",
    )
    detected_unit: Optional[str] = Field(
        None, description="Detected unit of measurement (if applicable)",
    )
    synonyms_matched: List[str] = Field(
        default_factory=list,
        description="List of synonym strings that contributed to match",
    )

    model_config = {"extra": "forbid"}

    @field_validator("sheet_id")
    @classmethod
    def validate_sheet_id(cls, v: str) -> str:
        """Validate sheet_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("sheet_id must be non-empty")
        return v

    @field_validator("source_column")
    @classmethod
    def validate_source_column(cls, v: str) -> str:
        """Validate source_column is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_column must be non-empty")
        return v


class NormalizationJob(BaseModel):
    """Record of a file normalization job execution.

    Tracks the lifecycle, configuration, row counts, errors, and
    provenance of a single normalization job for monitoring and audit.

    Attributes:
        job_id: Unique identifier for this normalization job.
        file_id: File being processed.
        status: Current lifecycle status of the job.
        config: Configuration parameters used for this job.
        rows_processed: Number of rows processed so far.
        rows_normalized: Number of rows successfully normalized.
        rows_skipped: Number of rows skipped due to errors.
        errors: List of error messages encountered during processing.
        started_at: Timestamp when the job started processing.
        completed_at: Timestamp when the job finished (if completed).
        duration_seconds: Total processing duration in seconds.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    job_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this normalization job",
    )
    file_id: str = Field(
        ..., description="File being processed",
    )
    status: JobStatus = Field(
        default=JobStatus.QUEUED,
        description="Current lifecycle status of the job",
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration parameters used for this job",
    )
    rows_processed: int = Field(
        default=0, ge=0,
        description="Number of rows processed so far",
    )
    rows_normalized: int = Field(
        default=0, ge=0,
        description="Number of rows successfully normalized",
    )
    rows_skipped: int = Field(
        default=0, ge=0,
        description="Number of rows skipped due to errors",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="List of error messages encountered during processing",
    )
    started_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the job started processing",
    )
    completed_at: Optional[datetime] = Field(
        None, description="Timestamp when the job finished",
    )
    duration_seconds: Optional[float] = Field(
        None, ge=0.0,
        description="Total processing duration in seconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )

    model_config = {"extra": "forbid"}

    @field_validator("file_id")
    @classmethod
    def validate_file_id(cls, v: str) -> str:
        """Validate file_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("file_id must be non-empty")
        return v


class NormalizedRecord(BaseModel):
    """A single normalized row from a spreadsheet.

    Contains original and normalized values, quality score,
    validation errors, and provenance hash for each row.

    Attributes:
        record_id: Unique identifier for this normalized record.
        job_id: Parent normalization job identifier.
        row_index: Zero-based row index in the source sheet.
        original_values: Original cell values keyed by column name.
        normalized_values: Normalized cell values keyed by canonical field.
        quality_score: Quality score for this row (0.0 to 1.0).
        validation_errors: List of validation error messages for this row.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this normalized record",
    )
    job_id: str = Field(
        ..., description="Parent normalization job identifier",
    )
    row_index: int = Field(
        default=0, ge=0,
        description="Zero-based row index in the source sheet",
    )
    original_values: Dict[str, Any] = Field(
        default_factory=dict,
        description="Original cell values keyed by column name",
    )
    normalized_values: Dict[str, Any] = Field(
        default_factory=dict,
        description="Normalized cell values keyed by canonical field",
    )
    quality_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Quality score for this row",
    )
    validation_errors: List[str] = Field(
        default_factory=list,
        description="List of validation error messages for this row",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, v: str) -> str:
        """Validate job_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("job_id must be non-empty")
        return v


class MappingTemplate(BaseModel):
    """Reusable template defining column mapping patterns.

    Templates allow users to define repeatable column mappings for
    specific source file layouts, enabling consistent normalization
    across recurring file types.

    Attributes:
        template_id: Unique identifier for this template.
        template_name: Human-readable template name.
        description: Detailed description of the template purpose.
        source_type: Type of source system or file layout.
        column_mappings: List of column mapping definitions.
        created_at: Timestamp when the template was created.
        updated_at: Timestamp when the template was last updated.
        usage_count: Number of times this template has been applied.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    template_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this template",
    )
    template_name: str = Field(
        ..., description="Human-readable template name",
    )
    description: str = Field(
        default="",
        description="Detailed description of the template purpose",
    )
    source_type: str = Field(
        default="", description="Type of source system or file layout",
    )
    column_mappings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of column mapping definitions",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the template was created",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the template was last updated",
    )
    usage_count: int = Field(
        default=0, ge=0,
        description="Number of times this template has been applied",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )

    model_config = {"extra": "forbid"}

    @field_validator("template_name")
    @classmethod
    def validate_template_name(cls, v: str) -> str:
        """Validate template_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("template_name must be non-empty")
        return v


class DataQualityReport(BaseModel):
    """Comprehensive data quality assessment report for a file.

    Contains overall and per-dimension quality scores, row-level
    statistics, and detailed issue listings.

    Attributes:
        report_id: Unique identifier for this quality report.
        file_id: File that was assessed.
        overall_score: Weighted overall quality score (0.0 to 1.0).
        completeness_score: Completeness dimension score (0.0 to 1.0).
        accuracy_score: Accuracy dimension score (0.0 to 1.0).
        consistency_score: Consistency dimension score (0.0 to 1.0).
        quality_level: Qualitative tier derived from overall_score.
        total_rows: Total number of rows assessed.
        valid_rows: Number of rows passing all validations.
        invalid_rows: Number of rows with at least one issue.
        null_count: Total number of null/empty cells detected.
        type_mismatch_count: Total number of type mismatch detections.
        duplicate_count: Total number of duplicate rows detected.
        column_scores: Per-column quality scores keyed by column name.
        issues: List of quality issue dictionaries with details.
    """

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this quality report",
    )
    file_id: str = Field(
        ..., description="File that was assessed",
    )
    overall_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Weighted overall quality score",
    )
    completeness_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Completeness dimension score",
    )
    accuracy_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Accuracy dimension score",
    )
    consistency_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Consistency dimension score",
    )
    quality_level: QualityLevel = Field(
        default=QualityLevel.POOR,
        description="Qualitative tier derived from overall_score",
    )
    total_rows: int = Field(
        default=0, ge=0, description="Total number of rows assessed",
    )
    valid_rows: int = Field(
        default=0, ge=0,
        description="Number of rows passing all validations",
    )
    invalid_rows: int = Field(
        default=0, ge=0,
        description="Number of rows with at least one issue",
    )
    null_count: int = Field(
        default=0, ge=0,
        description="Total number of null/empty cells detected",
    )
    type_mismatch_count: int = Field(
        default=0, ge=0,
        description="Total number of type mismatch detections",
    )
    duplicate_count: int = Field(
        default=0, ge=0,
        description="Total number of duplicate rows detected",
    )
    column_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-column quality scores keyed by column name",
    )
    issues: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of quality issue dictionaries with details",
    )

    model_config = {"extra": "forbid"}

    @field_validator("file_id")
    @classmethod
    def validate_file_id(cls, v: str) -> str:
        """Validate file_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("file_id must be non-empty")
        return v


class ValidationFinding(BaseModel):
    """A single validation finding for a specific cell or row.

    Captures the location, severity, rule, and descriptive message
    for a validation issue detected during normalization.

    Attributes:
        finding_id: Unique identifier for this validation finding.
        file_id: File that was validated.
        sheet_name: Name of the sheet containing the finding.
        row_index: Zero-based row index of the finding.
        column_name: Column name where the finding was detected.
        severity: Severity level of the finding.
        rule_name: Name of the validation rule that produced this finding.
        message: Human-readable description of the finding.
        expected_value: Expected value (if applicable).
        actual_value: Actual value that was found.
    """

    finding_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this validation finding",
    )
    file_id: str = Field(
        ..., description="File that was validated",
    )
    sheet_name: str = Field(
        default="", description="Name of the sheet containing the finding",
    )
    row_index: Optional[int] = Field(
        None, ge=0,
        description="Zero-based row index of the finding",
    )
    column_name: str = Field(
        default="", description="Column name where the finding was detected",
    )
    severity: str = Field(
        default="warning",
        description="Severity level of the finding (error, warning, info)",
    )
    rule_name: str = Field(
        default="", description="Name of the validation rule",
    )
    message: str = Field(
        default="",
        description="Human-readable description of the finding",
    )
    expected_value: Optional[str] = Field(
        None, description="Expected value (if applicable)",
    )
    actual_value: Optional[str] = Field(
        None, description="Actual value that was found",
    )

    model_config = {"extra": "forbid"}

    @field_validator("file_id")
    @classmethod
    def validate_file_id(cls, v: str) -> str:
        """Validate file_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("file_id must be non-empty")
        return v


class TransformResult(BaseModel):
    """Result of a data transformation operation.

    Captures the operation type, configuration, row counts,
    output data, and provenance for audit trails.

    Attributes:
        result_id: Unique identifier for this transform result.
        source_file_id: Source file that was transformed.
        operation: Transformation operation that was applied.
        config: Configuration parameters for the transform.
        input_rows: Number of input rows before transformation.
        output_rows: Number of output rows after transformation.
        rows_affected: Number of rows modified by the transformation.
        output_data: List of output row dictionaries.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    result_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this transform result",
    )
    source_file_id: str = Field(
        ..., description="Source file that was transformed",
    )
    operation: TransformOperation = Field(
        ..., description="Transformation operation that was applied",
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration parameters for the transform",
    )
    input_rows: int = Field(
        default=0, ge=0,
        description="Number of input rows before transformation",
    )
    output_rows: int = Field(
        default=0, ge=0,
        description="Number of output rows after transformation",
    )
    rows_affected: int = Field(
        default=0, ge=0,
        description="Number of rows modified by the transformation",
    )
    output_data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of output row dictionaries",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("source_file_id")
    @classmethod
    def validate_source_file_id(cls, v: str) -> str:
        """Validate source_file_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_file_id must be non-empty")
        return v


class ExcelStatistics(BaseModel):
    """Aggregated statistics for the Excel normalizer service.

    Attributes:
        total_files_processed: Total number of files processed.
        total_rows_normalized: Total number of rows normalized.
        total_columns_mapped: Total number of columns mapped.
        avg_quality_score: Average quality score across all files.
        avg_mapping_confidence: Average mapping confidence across columns.
        files_by_format: Breakdown of files processed by format.
        columns_by_type: Breakdown of columns by detected data type.
        quality_distribution: Breakdown of files by quality level.
    """

    total_files_processed: int = Field(
        default=0, ge=0,
        description="Total number of files processed",
    )
    total_rows_normalized: int = Field(
        default=0, ge=0,
        description="Total number of rows normalized",
    )
    total_columns_mapped: int = Field(
        default=0, ge=0,
        description="Total number of columns mapped",
    )
    avg_quality_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Average quality score across all files",
    )
    avg_mapping_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Average mapping confidence across columns",
    )
    files_by_format: Dict[str, int] = Field(
        default_factory=dict,
        description="Breakdown of files processed by format",
    )
    columns_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Breakdown of columns by detected data type",
    )
    quality_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Breakdown of files by quality level",
    )

    model_config = {"extra": "forbid"}


# =============================================================================
# Request Models
# =============================================================================


class UploadFileRequest(BaseModel):
    """Request body for uploading a single spreadsheet or CSV file.

    Attributes:
        file_name: Name of the file to upload.
        file_content_base64: Base64-encoded file content.
        file_format: Expected file format (or auto for detection).
        template_id: Optional template ID for column mapping.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    file_name: str = Field(
        ..., description="Name of the file to upload",
    )
    file_content_base64: str = Field(
        ..., description="Base64-encoded file content",
    )
    file_format: Optional[FileFormat] = Field(
        None, description="Expected file format (or auto for detection)",
    )
    template_id: Optional[str] = Field(
        None, description="Optional template ID for column mapping",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )

    model_config = {"extra": "forbid"}

    @field_validator("file_name")
    @classmethod
    def validate_file_name(cls, v: str) -> str:
        """Validate file_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("file_name must be non-empty")
        return v

    @field_validator("file_content_base64")
    @classmethod
    def validate_file_content_base64(cls, v: str) -> str:
        """Validate file_content_base64 is non-empty."""
        if not v or not v.strip():
            raise ValueError("file_content_base64 must be non-empty")
        return v


class BatchUploadRequest(BaseModel):
    """Request body for uploading a batch of files.

    Attributes:
        files: List of individual upload requests.
        parallel: Whether to process files in parallel.
    """

    files: List[UploadFileRequest] = Field(
        ..., description="List of individual upload requests",
    )
    parallel: bool = Field(
        default=True,
        description="Whether to process files in parallel",
    )

    model_config = {"extra": "forbid"}

    @field_validator("files")
    @classmethod
    def validate_files(cls, v: List[UploadFileRequest]) -> List[UploadFileRequest]:
        """Validate files list is non-empty."""
        if not v:
            raise ValueError("files list must be non-empty")
        return v


class MapColumnsRequest(BaseModel):
    """Request body for mapping columns to canonical fields.

    Attributes:
        sheet_id: Identifier of the sheet to map columns for.
        columns: List of source column names to map.
        strategy: Mapping strategy to use.
        template_id: Optional template ID for predefined mappings.
    """

    sheet_id: str = Field(
        ..., description="Identifier of the sheet to map columns for",
    )
    columns: List[str] = Field(
        ..., description="List of source column names to map",
    )
    strategy: MappingStrategy = Field(
        default=MappingStrategy.FUZZY,
        description="Mapping strategy to use",
    )
    template_id: Optional[str] = Field(
        None, description="Optional template ID for predefined mappings",
    )

    model_config = {"extra": "forbid"}

    @field_validator("sheet_id")
    @classmethod
    def validate_sheet_id(cls, v: str) -> str:
        """Validate sheet_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("sheet_id must be non-empty")
        return v

    @field_validator("columns")
    @classmethod
    def validate_columns(cls, v: List[str]) -> List[str]:
        """Validate columns list is non-empty."""
        if not v:
            raise ValueError("columns list must be non-empty")
        return v


class NormalizeRequest(BaseModel):
    """Request body for normalizing data records.

    Attributes:
        data: List of row dictionaries to normalize.
        column_mappings: Mapping of source columns to canonical fields.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    data: List[Dict[str, Any]] = Field(
        ..., description="List of row dictionaries to normalize",
    )
    column_mappings: Dict[str, str] = Field(
        ..., description="Mapping of source columns to canonical fields",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )

    model_config = {"extra": "forbid"}

    @field_validator("data")
    @classmethod
    def validate_data(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate data list is non-empty."""
        if not v:
            raise ValueError("data list must be non-empty")
        return v

    @field_validator("column_mappings")
    @classmethod
    def validate_column_mappings(
        cls, v: Dict[str, str],
    ) -> Dict[str, str]:
        """Validate column_mappings is non-empty."""
        if not v:
            raise ValueError("column_mappings must be non-empty")
        return v


class TransformRequest(BaseModel):
    """Request body for applying data transformations.

    Attributes:
        file_id: Identifier of the file to transform.
        operations: List of transformation operation definitions.
    """

    file_id: str = Field(
        ..., description="Identifier of the file to transform",
    )
    operations: List[Dict[str, Any]] = Field(
        ..., description="List of transformation operation definitions",
    )

    model_config = {"extra": "forbid"}

    @field_validator("file_id")
    @classmethod
    def validate_file_id(cls, v: str) -> str:
        """Validate file_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("file_id must be non-empty")
        return v

    @field_validator("operations")
    @classmethod
    def validate_operations(
        cls, v: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Validate operations list is non-empty."""
        if not v:
            raise ValueError("operations list must be non-empty")
        return v


class CreateTemplateRequest(BaseModel):
    """Request body for creating a new mapping template.

    Attributes:
        template_name: Human-readable template name.
        description: Detailed description of the template purpose.
        source_type: Type of source system or file layout.
        column_mappings: List of column mapping definitions.
    """

    template_name: str = Field(
        ..., description="Human-readable template name",
    )
    description: str = Field(
        default="",
        description="Detailed description of the template purpose",
    )
    source_type: str = Field(
        default="",
        description="Type of source system or file layout",
    )
    column_mappings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of column mapping definitions",
    )

    model_config = {"extra": "forbid"}

    @field_validator("template_name")
    @classmethod
    def validate_template_name(cls, v: str) -> str:
        """Validate template_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("template_name must be non-empty")
        return v


__all__ = [
    # Enumerations
    "FileFormat",
    "DelimiterType",
    "DataType",
    "MappingStrategy",
    "JobStatus",
    "QualityLevel",
    "TransformOperation",
    # SDK models
    "SpreadsheetFile",
    "SheetMetadata",
    "ColumnMapping",
    "NormalizationJob",
    "NormalizedRecord",
    "MappingTemplate",
    "DataQualityReport",
    "ValidationFinding",
    "TransformResult",
    "ExcelStatistics",
    # Request models
    "UploadFileRequest",
    "BatchUploadRequest",
    "MapColumnsRequest",
    "NormalizeRequest",
    "TransformRequest",
    "CreateTemplateRequest",
]
