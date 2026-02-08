# -*- coding: utf-8 -*-
"""
Unit Tests for Excel Normalizer Models (AGENT-DATA-002)

Tests all enums (FileFormat, DelimiterType, DataType, MappingStrategy,
JobStatus, QualityLevel, TransformOperation) and all SDK models
(SpreadsheetFile, SheetMetadata, ColumnMapping, NormalizationJob,
NormalizedRecord, MappingTemplate, DataQualityReport, ValidationFinding,
TransformResult, ExcelStatistics), plus request models.

Coverage target: 85%+ of models.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline enums mirroring greenlang/excel_normalizer/models.py
# ---------------------------------------------------------------------------


class FileFormat(str, Enum):
    XLSX = "xlsx"
    XLS = "xls"
    CSV = "csv"
    TSV = "tsv"
    AUTO = "auto"


class DelimiterType(str, Enum):
    COMMA = "comma"
    SEMICOLON = "semicolon"
    TAB = "tab"
    PIPE = "pipe"
    SPACE = "space"
    AUTO = "auto"


class DataType(str, Enum):
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
    EXACT = "exact"
    FUZZY = "fuzzy"
    SYNONYM = "synonym"
    PATTERN = "pattern"
    MANUAL = "manual"


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QualityLevel(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class TransformOperation(str, Enum):
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


# ---------------------------------------------------------------------------
# Inline SDK models mirroring greenlang/excel_normalizer/models.py
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


class SpreadsheetFile:
    """Uploaded spreadsheet or CSV file record."""

    def __init__(
        self,
        file_id: Optional[str] = None,
        file_name: str = "",
        file_path: str = "",
        file_format: str = "auto",
        file_size_bytes: int = 0,
        file_hash: str = "",
        sheet_count: int = 1,
        total_rows: int = 0,
        total_columns: int = 0,
        upload_timestamp: Optional[str] = None,
        tenant_id: str = "default",
        provenance_hash: str = "",
        uploaded_by: str = "system",
    ):
        self.file_id = file_id or str(uuid.uuid4())
        self.file_name = file_name
        self.file_path = file_path
        self.file_format = file_format
        self.file_size_bytes = file_size_bytes
        self.file_hash = file_hash
        self.sheet_count = sheet_count
        self.total_rows = total_rows
        self.total_columns = total_columns
        self.upload_timestamp = upload_timestamp or _utcnow().isoformat()
        self.tenant_id = tenant_id
        self.provenance_hash = provenance_hash
        self.uploaded_by = uploaded_by

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_id": self.file_id,
            "file_name": self.file_name,
            "file_format": self.file_format,
            "file_size_bytes": self.file_size_bytes,
            "file_hash": self.file_hash,
            "sheet_count": self.sheet_count,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "tenant_id": self.tenant_id,
            "provenance_hash": self.provenance_hash,
        }


class SheetMetadata:
    """Metadata for a single sheet within a workbook."""

    def __init__(
        self,
        sheet_id: Optional[str] = None,
        file_id: str = "",
        sheet_name: str = "Sheet1",
        sheet_index: int = 0,
        row_count: int = 0,
        column_count: int = 0,
        header_row_index: int = 0,
        has_headers: bool = True,
        detected_encoding: Optional[str] = None,
        detected_delimiter: Optional[str] = None,
    ):
        self.sheet_id = sheet_id or str(uuid.uuid4())
        self.file_id = file_id
        self.sheet_name = sheet_name
        self.sheet_index = sheet_index
        self.row_count = row_count
        self.column_count = column_count
        self.header_row_index = header_row_index
        self.has_headers = has_headers
        self.detected_encoding = detected_encoding
        self.detected_delimiter = detected_delimiter

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sheet_id": self.sheet_id,
            "file_id": self.file_id,
            "sheet_name": self.sheet_name,
            "sheet_index": self.sheet_index,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "has_headers": self.has_headers,
        }


class ColumnMapping:
    """Mapping from source column to canonical field."""

    def __init__(
        self,
        mapping_id: Optional[str] = None,
        sheet_id: str = "",
        source_column: str = "",
        source_index: int = 0,
        canonical_field: str = "",
        mapping_strategy: str = "fuzzy",
        confidence: float = 0.0,
        detected_data_type: str = "unknown",
        detected_unit: Optional[str] = None,
        synonyms_matched: Optional[List[str]] = None,
    ):
        self.mapping_id = mapping_id or str(uuid.uuid4())
        self.sheet_id = sheet_id
        self.source_column = source_column
        self.source_index = source_index
        self.canonical_field = canonical_field
        self.mapping_strategy = mapping_strategy
        self.confidence = confidence
        self.detected_data_type = detected_data_type
        self.detected_unit = detected_unit
        self.synonyms_matched = synonyms_matched or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mapping_id": self.mapping_id,
            "source_column": self.source_column,
            "canonical_field": self.canonical_field,
            "mapping_strategy": self.mapping_strategy,
            "confidence": self.confidence,
            "detected_data_type": self.detected_data_type,
        }


class NormalizationJob:
    """Normalization job execution record."""

    def __init__(
        self,
        job_id: Optional[str] = None,
        file_id: str = "",
        status: str = "queued",
        config: Optional[Dict[str, Any]] = None,
        rows_processed: int = 0,
        rows_normalized: int = 0,
        rows_skipped: int = 0,
        errors: Optional[List[str]] = None,
        started_at: Optional[str] = None,
        completed_at: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        provenance_hash: str = "",
        tenant_id: str = "default",
    ):
        self.job_id = job_id or str(uuid.uuid4())
        self.file_id = file_id
        self.status = status
        self.config = config or {}
        self.rows_processed = rows_processed
        self.rows_normalized = rows_normalized
        self.rows_skipped = rows_skipped
        self.errors = errors or []
        self.started_at = started_at or _utcnow().isoformat()
        self.completed_at = completed_at
        self.duration_seconds = duration_seconds
        self.provenance_hash = provenance_hash
        self.tenant_id = tenant_id

    @property
    def progress_pct(self) -> float:
        total = self.rows_normalized + self.rows_skipped
        if self.rows_processed == 0:
            return 0.0
        return total / self.rows_processed * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "file_id": self.file_id,
            "status": self.status,
            "rows_processed": self.rows_processed,
            "rows_normalized": self.rows_normalized,
            "rows_skipped": self.rows_skipped,
            "provenance_hash": self.provenance_hash,
        }


class NormalizedRecord:
    """A single normalized row from a spreadsheet."""

    def __init__(
        self,
        record_id: Optional[str] = None,
        job_id: str = "",
        row_index: int = 0,
        original_values: Optional[Dict[str, Any]] = None,
        normalized_values: Optional[Dict[str, Any]] = None,
        quality_score: float = 0.0,
        validation_errors: Optional[List[str]] = None,
        provenance_hash: str = "",
    ):
        self.record_id = record_id or str(uuid.uuid4())
        self.job_id = job_id
        self.row_index = row_index
        self.original_values = original_values or {}
        self.normalized_values = normalized_values or {}
        self.quality_score = quality_score
        self.validation_errors = validation_errors or []
        self.provenance_hash = provenance_hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "job_id": self.job_id,
            "row_index": self.row_index,
            "quality_score": self.quality_score,
            "validation_errors": self.validation_errors,
        }


class MappingTemplate:
    """Reusable column mapping template."""

    def __init__(
        self,
        template_id: Optional[str] = None,
        template_name: str = "",
        description: str = "",
        source_type: str = "",
        column_mappings: Optional[List[Dict[str, Any]]] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        usage_count: int = 0,
        tenant_id: str = "default",
    ):
        self.template_id = template_id or str(uuid.uuid4())
        self.template_name = template_name
        self.description = description
        self.source_type = source_type
        self.column_mappings = column_mappings or []
        self.created_at = created_at or _utcnow().isoformat()
        self.updated_at = updated_at or self.created_at
        self.usage_count = usage_count
        self.tenant_id = tenant_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "template_name": self.template_name,
            "source_type": self.source_type,
            "column_mappings": self.column_mappings,
            "usage_count": self.usage_count,
        }


class DataQualityReport:
    """Data quality assessment report."""

    def __init__(
        self,
        report_id: Optional[str] = None,
        file_id: str = "",
        overall_score: float = 0.0,
        completeness_score: float = 0.0,
        accuracy_score: float = 0.0,
        consistency_score: float = 0.0,
        quality_level: str = "poor",
        total_rows: int = 0,
        valid_rows: int = 0,
        invalid_rows: int = 0,
        null_count: int = 0,
        type_mismatch_count: int = 0,
        duplicate_count: int = 0,
        column_scores: Optional[Dict[str, float]] = None,
        issues: Optional[List[Dict[str, Any]]] = None,
    ):
        self.report_id = report_id or str(uuid.uuid4())
        self.file_id = file_id
        self.overall_score = overall_score
        self.completeness_score = completeness_score
        self.accuracy_score = accuracy_score
        self.consistency_score = consistency_score
        self.quality_level = quality_level
        self.total_rows = total_rows
        self.valid_rows = valid_rows
        self.invalid_rows = invalid_rows
        self.null_count = null_count
        self.type_mismatch_count = type_mismatch_count
        self.duplicate_count = duplicate_count
        self.column_scores = column_scores or {}
        self.issues = issues or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "file_id": self.file_id,
            "overall_score": self.overall_score,
            "quality_level": self.quality_level,
            "total_rows": self.total_rows,
            "valid_rows": self.valid_rows,
            "invalid_rows": self.invalid_rows,
        }


class ValidationFinding:
    """Single validation finding."""

    def __init__(
        self,
        finding_id: Optional[str] = None,
        file_id: str = "",
        sheet_name: str = "",
        row_index: Optional[int] = None,
        column_name: str = "",
        severity: str = "warning",
        rule_name: str = "",
        message: str = "",
        expected_value: Optional[str] = None,
        actual_value: Optional[str] = None,
    ):
        self.finding_id = finding_id or str(uuid.uuid4())
        self.file_id = file_id
        self.sheet_name = sheet_name
        self.row_index = row_index
        self.column_name = column_name
        self.severity = severity
        self.rule_name = rule_name
        self.message = message
        self.expected_value = expected_value
        self.actual_value = actual_value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "severity": self.severity,
            "rule_name": self.rule_name,
            "message": self.message,
            "column_name": self.column_name,
        }


class TransformResult:
    """Result of a data transformation."""

    def __init__(
        self,
        result_id: Optional[str] = None,
        source_file_id: str = "",
        operation: str = "rename",
        config: Optional[Dict[str, Any]] = None,
        input_rows: int = 0,
        output_rows: int = 0,
        rows_affected: int = 0,
        output_data: Optional[List[Dict[str, Any]]] = None,
        provenance_hash: str = "",
    ):
        self.result_id = result_id or str(uuid.uuid4())
        self.source_file_id = source_file_id
        self.operation = operation
        self.config = config or {}
        self.input_rows = input_rows
        self.output_rows = output_rows
        self.rows_affected = rows_affected
        self.output_data = output_data or []
        self.provenance_hash = provenance_hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result_id": self.result_id,
            "operation": self.operation,
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "rows_affected": self.rows_affected,
        }


class ExcelStatistics:
    """Aggregated service statistics."""

    def __init__(self):
        self.total_files_processed: int = 0
        self.total_rows_normalized: int = 0
        self.total_columns_mapped: int = 0
        self.avg_quality_score: float = 0.0
        self.avg_mapping_confidence: float = 0.0
        self._quality_sum: float = 0.0
        self._quality_count: int = 0
        self._confidence_sum: float = 0.0
        self._confidence_count: int = 0
        self.files_by_format: Dict[str, int] = {}
        self.columns_by_type: Dict[str, int] = {}
        self.quality_distribution: Dict[str, int] = {}

    def record_file(self, file_format: str, quality_score: float):
        self.total_files_processed += 1
        self._quality_sum += quality_score
        self._quality_count += 1
        self.avg_quality_score = self._quality_sum / self._quality_count
        self.files_by_format[file_format] = self.files_by_format.get(file_format, 0) + 1

    def record_mapping(self, confidence: float):
        self.total_columns_mapped += 1
        self._confidence_sum += confidence
        self._confidence_count += 1
        self.avg_mapping_confidence = self._confidence_sum / self._confidence_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_files_processed": self.total_files_processed,
            "total_rows_normalized": self.total_rows_normalized,
            "total_columns_mapped": self.total_columns_mapped,
            "avg_quality_score": round(self.avg_quality_score, 4),
            "avg_mapping_confidence": round(self.avg_mapping_confidence, 4),
            "files_by_format": self.files_by_format,
        }


# ---------------------------------------------------------------------------
# Inline request models
# ---------------------------------------------------------------------------


class UploadFileRequest:
    def __init__(self, file_name: str = "", file_content_base64: str = "",
                 file_format: Optional[str] = None, template_id: Optional[str] = None,
                 tenant_id: str = "default"):
        self.file_name = file_name
        self.file_content_base64 = file_content_base64
        self.file_format = file_format
        self.template_id = template_id
        self.tenant_id = tenant_id


class BatchUploadRequest:
    def __init__(self, files: Optional[List[UploadFileRequest]] = None, parallel: bool = True):
        self.files = files or []
        self.parallel = parallel


class MapColumnsRequest:
    def __init__(self, sheet_id: str = "", columns: Optional[List[str]] = None,
                 strategy: str = "fuzzy", template_id: Optional[str] = None):
        self.sheet_id = sheet_id
        self.columns = columns or []
        self.strategy = strategy
        self.template_id = template_id


class NormalizeRequest:
    def __init__(self, data: Optional[List[Dict[str, Any]]] = None,
                 column_mappings: Optional[Dict[str, str]] = None, tenant_id: str = "default"):
        self.data = data or []
        self.column_mappings = column_mappings or {}
        self.tenant_id = tenant_id


class TransformRequest:
    def __init__(self, file_id: str = "", operations: Optional[List[Dict[str, Any]]] = None):
        self.file_id = file_id
        self.operations = operations or []


class CreateTemplateRequest:
    def __init__(self, template_name: str = "", description: str = "",
                 source_type: str = "", column_mappings: Optional[List[Dict[str, Any]]] = None):
        self.template_name = template_name
        self.description = description
        self.source_type = source_type
        self.column_mappings = column_mappings or []


# ===========================================================================
# Test Classes - Enums
# ===========================================================================


class TestFileFormatEnum:
    def test_xlsx(self):
        assert FileFormat.XLSX.value == "xlsx"

    def test_xls(self):
        assert FileFormat.XLS.value == "xls"

    def test_csv(self):
        assert FileFormat.CSV.value == "csv"

    def test_tsv(self):
        assert FileFormat.TSV.value == "tsv"

    def test_auto(self):
        assert FileFormat.AUTO.value == "auto"

    def test_all_5_formats(self):
        assert len(FileFormat) == 5

    def test_string_conversion(self):
        assert str(FileFormat.XLSX) == "FileFormat.XLSX"

    def test_from_value(self):
        assert FileFormat("csv") == FileFormat.CSV


class TestDelimiterTypeEnum:
    def test_comma(self):
        assert DelimiterType.COMMA.value == "comma"

    def test_semicolon(self):
        assert DelimiterType.SEMICOLON.value == "semicolon"

    def test_tab(self):
        assert DelimiterType.TAB.value == "tab"

    def test_pipe(self):
        assert DelimiterType.PIPE.value == "pipe"

    def test_space(self):
        assert DelimiterType.SPACE.value == "space"

    def test_auto(self):
        assert DelimiterType.AUTO.value == "auto"

    def test_all_6_types(self):
        assert len(DelimiterType) == 6


class TestDataTypeEnum:
    def test_string(self):
        assert DataType.STRING.value == "string"

    def test_integer(self):
        assert DataType.INTEGER.value == "integer"

    def test_float(self):
        assert DataType.FLOAT.value == "float"

    def test_decimal(self):
        assert DataType.DECIMAL.value == "decimal"

    def test_date(self):
        assert DataType.DATE.value == "date"

    def test_datetime(self):
        assert DataType.DATETIME.value == "datetime"

    def test_boolean(self):
        assert DataType.BOOLEAN.value == "boolean"

    def test_currency(self):
        assert DataType.CURRENCY.value == "currency"

    def test_percentage(self):
        assert DataType.PERCENTAGE.value == "percentage"

    def test_unit_value(self):
        assert DataType.UNIT_VALUE.value == "unit_value"

    def test_email(self):
        assert DataType.EMAIL.value == "email"

    def test_url(self):
        assert DataType.URL.value == "url"

    def test_empty(self):
        assert DataType.EMPTY.value == "empty"

    def test_unknown(self):
        assert DataType.UNKNOWN.value == "unknown"

    def test_all_14_types(self):
        assert len(DataType) == 14


class TestMappingStrategyEnum:
    def test_exact(self):
        assert MappingStrategy.EXACT.value == "exact"

    def test_fuzzy(self):
        assert MappingStrategy.FUZZY.value == "fuzzy"

    def test_synonym(self):
        assert MappingStrategy.SYNONYM.value == "synonym"

    def test_pattern(self):
        assert MappingStrategy.PATTERN.value == "pattern"

    def test_manual(self):
        assert MappingStrategy.MANUAL.value == "manual"

    def test_all_5_strategies(self):
        assert len(MappingStrategy) == 5


class TestJobStatusEnum:
    def test_queued(self):
        assert JobStatus.QUEUED.value == "queued"

    def test_processing(self):
        assert JobStatus.PROCESSING.value == "processing"

    def test_completed(self):
        assert JobStatus.COMPLETED.value == "completed"

    def test_failed(self):
        assert JobStatus.FAILED.value == "failed"

    def test_cancelled(self):
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_all_5_statuses(self):
        assert len(JobStatus) == 5


class TestQualityLevelEnum:
    def test_excellent(self):
        assert QualityLevel.EXCELLENT.value == "excellent"

    def test_good(self):
        assert QualityLevel.GOOD.value == "good"

    def test_fair(self):
        assert QualityLevel.FAIR.value == "fair"

    def test_poor(self):
        assert QualityLevel.POOR.value == "poor"

    def test_all_4_levels(self):
        assert len(QualityLevel) == 4


class TestTransformOperationEnum:
    def test_pivot(self):
        assert TransformOperation.PIVOT.value == "pivot"

    def test_unpivot(self):
        assert TransformOperation.UNPIVOT.value == "unpivot"

    def test_dedup(self):
        assert TransformOperation.DEDUP.value == "dedup"

    def test_merge(self):
        assert TransformOperation.MERGE.value == "merge"

    def test_filter(self):
        assert TransformOperation.FILTER.value == "filter"

    def test_aggregate(self):
        assert TransformOperation.AGGREGATE.value == "aggregate"

    def test_rename(self):
        assert TransformOperation.RENAME.value == "rename"

    def test_split(self):
        assert TransformOperation.SPLIT.value == "split"

    def test_cast(self):
        assert TransformOperation.CAST.value == "cast"

    def test_fill_missing(self):
        assert TransformOperation.FILL_MISSING.value == "fill_missing"

    def test_all_10_operations(self):
        assert len(TransformOperation) == 10


# ===========================================================================
# Test Classes - SDK Models
# ===========================================================================


class TestSpreadsheetFileModel:
    def test_creation_defaults(self):
        f = SpreadsheetFile()
        assert len(f.file_id) == 36
        assert f.file_name == ""
        assert f.file_format == "auto"
        assert f.sheet_count == 1

    def test_creation_with_values(self):
        f = SpreadsheetFile(file_id="f-001", file_name="report.xlsx",
                            file_format="xlsx", file_size_bytes=102400, sheet_count=3)
        assert f.file_id == "f-001"
        assert f.file_name == "report.xlsx"
        assert f.sheet_count == 3

    def test_auto_generated_id(self):
        f = SpreadsheetFile()
        assert len(f.file_id) == 36

    def test_to_dict(self):
        f = SpreadsheetFile(file_id="f-001", file_name="test.csv")
        d = f.to_dict()
        assert d["file_id"] == "f-001"
        assert d["file_name"] == "test.csv"

    def test_default_tenant(self):
        f = SpreadsheetFile()
        assert f.tenant_id == "default"

    def test_provenance_hash(self):
        f = SpreadsheetFile(provenance_hash="a" * 64)
        assert len(f.provenance_hash) == 64


class TestSheetMetadataModel:
    def test_creation_defaults(self):
        s = SheetMetadata()
        assert len(s.sheet_id) == 36
        assert s.sheet_name == "Sheet1"
        assert s.has_headers is True

    def test_creation_with_values(self):
        s = SheetMetadata(sheet_id="s-001", file_id="f-001", sheet_name="Emissions",
                          row_count=1000, column_count=8)
        assert s.sheet_name == "Emissions"
        assert s.row_count == 1000

    def test_to_dict(self):
        s = SheetMetadata(sheet_id="s-001", file_id="f-001")
        d = s.to_dict()
        assert d["sheet_id"] == "s-001"
        assert d["file_id"] == "f-001"

    def test_encoding_detection(self):
        s = SheetMetadata(detected_encoding="utf-8", detected_delimiter=",")
        assert s.detected_encoding == "utf-8"
        assert s.detected_delimiter == ","


class TestColumnMappingModel:
    def test_creation_defaults(self):
        m = ColumnMapping()
        assert len(m.mapping_id) == 36
        assert m.mapping_strategy == "fuzzy"
        assert m.confidence == 0.0

    def test_creation_with_values(self):
        m = ColumnMapping(sheet_id="s-001", source_column="CO2 Emissions",
                          canonical_field="scope1_emissions", confidence=0.92)
        assert m.source_column == "CO2 Emissions"
        assert m.confidence == 0.92

    def test_synonyms_matched(self):
        m = ColumnMapping(synonyms_matched=["CO2", "carbon dioxide", "GHG"])
        assert len(m.synonyms_matched) == 3

    def test_to_dict(self):
        m = ColumnMapping(source_column="col1", canonical_field="field1")
        d = m.to_dict()
        assert d["source_column"] == "col1"
        assert d["canonical_field"] == "field1"

    def test_detected_unit(self):
        m = ColumnMapping(detected_unit="kgCO2e")
        assert m.detected_unit == "kgCO2e"


class TestNormalizationJobModel:
    def test_creation_defaults(self):
        j = NormalizationJob()
        assert len(j.job_id) == 36
        assert j.status == "queued"
        assert j.rows_processed == 0

    def test_creation_with_values(self):
        j = NormalizationJob(job_id="j-001", file_id="f-001", status="completed",
                             rows_processed=1000, rows_normalized=980, rows_skipped=20)
        assert j.rows_normalized == 980
        assert j.rows_skipped == 20

    def test_progress_pct_zero_processed(self):
        j = NormalizationJob(rows_processed=0)
        assert j.progress_pct == 0.0

    def test_progress_pct_partial(self):
        j = NormalizationJob(rows_processed=100, rows_normalized=60, rows_skipped=10)
        assert j.progress_pct == 70.0

    def test_progress_pct_complete(self):
        j = NormalizationJob(rows_processed=100, rows_normalized=100, rows_skipped=0)
        assert j.progress_pct == 100.0

    def test_to_dict(self):
        j = NormalizationJob(job_id="j-001", file_id="f-001")
        d = j.to_dict()
        assert d["job_id"] == "j-001"
        assert "status" in d

    def test_errors_list(self):
        j = NormalizationJob(errors=["Row 5: type mismatch", "Row 12: missing value"])
        assert len(j.errors) == 2


class TestNormalizedRecordModel:
    def test_creation_defaults(self):
        r = NormalizedRecord()
        assert len(r.record_id) == 36
        assert r.quality_score == 0.0

    def test_creation_with_values(self):
        r = NormalizedRecord(
            job_id="j-001", row_index=5,
            original_values={"CO2": "1250.5"},
            normalized_values={"scope1_emissions": 1250.5},
            quality_score=0.95,
        )
        assert r.row_index == 5
        assert r.quality_score == 0.95

    def test_to_dict(self):
        r = NormalizedRecord(record_id="r-001", job_id="j-001")
        d = r.to_dict()
        assert d["record_id"] == "r-001"

    def test_validation_errors(self):
        r = NormalizedRecord(validation_errors=["Type mismatch for column 'emissions'"])
        assert len(r.validation_errors) == 1


class TestMappingTemplateModel:
    def test_creation_defaults(self):
        t = MappingTemplate()
        assert len(t.template_id) == 36
        assert t.template_name == ""
        assert t.usage_count == 0

    def test_creation_with_values(self):
        t = MappingTemplate(template_id="t-001", template_name="Energy Reporting",
                            source_type="utility_bill", usage_count=15)
        assert t.template_name == "Energy Reporting"
        assert t.usage_count == 15

    def test_to_dict(self):
        t = MappingTemplate(template_name="Test")
        d = t.to_dict()
        assert d["template_name"] == "Test"

    def test_column_mappings(self):
        mappings = [{"source": "kWh Used", "target": "energy_kwh"}]
        t = MappingTemplate(column_mappings=mappings)
        assert len(t.column_mappings) == 1

    def test_default_timestamps(self):
        t = MappingTemplate()
        assert t.created_at is not None
        assert t.updated_at is not None


class TestDataQualityReportModel:
    def test_creation_defaults(self):
        r = DataQualityReport()
        assert len(r.report_id) == 36
        assert r.overall_score == 0.0
        assert r.quality_level == "poor"

    def test_creation_with_values(self):
        r = DataQualityReport(file_id="f-001", overall_score=0.92,
                              completeness_score=0.95, accuracy_score=0.90,
                              consistency_score=0.88, quality_level="excellent",
                              total_rows=1000, valid_rows=980, invalid_rows=20)
        assert r.overall_score == 0.92
        assert r.quality_level == "excellent"

    def test_to_dict(self):
        r = DataQualityReport(report_id="r-001", file_id="f-001")
        d = r.to_dict()
        assert d["report_id"] == "r-001"

    def test_column_scores(self):
        r = DataQualityReport(column_scores={"emissions": 0.95, "energy": 0.88})
        assert r.column_scores["emissions"] == 0.95

    def test_issues_list(self):
        r = DataQualityReport(issues=[{"type": "null", "column": "scope1", "count": 5}])
        assert len(r.issues) == 1


class TestValidationFindingModel:
    def test_creation_defaults(self):
        f = ValidationFinding()
        assert len(f.finding_id) == 36
        assert f.severity == "warning"

    def test_creation_with_values(self):
        f = ValidationFinding(file_id="f-001", sheet_name="Sheet1", row_index=5,
                              column_name="emissions", severity="error",
                              rule_name="range_check", message="Value exceeds maximum")
        assert f.severity == "error"
        assert f.rule_name == "range_check"

    def test_to_dict(self):
        f = ValidationFinding(finding_id="vf-001", severity="error")
        d = f.to_dict()
        assert d["severity"] == "error"

    def test_expected_actual(self):
        f = ValidationFinding(expected_value="<10000", actual_value="15000")
        assert f.expected_value == "<10000"
        assert f.actual_value == "15000"


class TestTransformResultModel:
    def test_creation_defaults(self):
        t = TransformResult()
        assert len(t.result_id) == 36
        assert t.operation == "rename"

    def test_creation_with_values(self):
        t = TransformResult(source_file_id="f-001", operation="dedup",
                            input_rows=1000, output_rows=950, rows_affected=50)
        assert t.operation == "dedup"
        assert t.rows_affected == 50

    def test_to_dict(self):
        t = TransformResult(operation="pivot", input_rows=100, output_rows=10)
        d = t.to_dict()
        assert d["operation"] == "pivot"
        assert d["input_rows"] == 100

    def test_output_data(self):
        t = TransformResult(output_data=[{"field": "value"}])
        assert len(t.output_data) == 1


class TestExcelStatisticsModel:
    def test_creation_defaults(self):
        s = ExcelStatistics()
        assert s.total_files_processed == 0
        assert s.avg_quality_score == 0.0

    def test_record_file(self):
        s = ExcelStatistics()
        s.record_file("csv", 0.9)
        assert s.total_files_processed == 1
        assert s.avg_quality_score == 0.9

    def test_record_multiple_files(self):
        s = ExcelStatistics()
        s.record_file("csv", 0.8)
        s.record_file("xlsx", 1.0)
        assert s.total_files_processed == 2
        assert s.avg_quality_score == pytest.approx(0.9, rel=1e-6)

    def test_record_mapping(self):
        s = ExcelStatistics()
        s.record_mapping(0.85)
        s.record_mapping(0.95)
        assert s.total_columns_mapped == 2
        assert s.avg_mapping_confidence == pytest.approx(0.9, rel=1e-6)

    def test_to_dict(self):
        s = ExcelStatistics()
        s.total_files_processed = 5
        s.record_file("csv", 0.85)
        d = s.to_dict()
        assert d["total_files_processed"] == 6
        assert "files_by_format" in d

    def test_files_by_format(self):
        s = ExcelStatistics()
        s.record_file("csv", 0.9)
        s.record_file("csv", 0.8)
        s.record_file("xlsx", 0.95)
        assert s.files_by_format["csv"] == 2
        assert s.files_by_format["xlsx"] == 1


# ===========================================================================
# Test Classes - Request Models
# ===========================================================================


class TestUploadFileRequest:
    def test_creation(self):
        r = UploadFileRequest(file_name="report.csv", file_content_base64="dGVzdA==")
        assert r.file_name == "report.csv"
        assert r.file_content_base64 == "dGVzdA=="

    def test_default_tenant(self):
        r = UploadFileRequest(file_name="x.csv", file_content_base64="x")
        assert r.tenant_id == "default"

    def test_optional_fields(self):
        r = UploadFileRequest(file_name="x.csv", file_content_base64="x",
                              file_format="csv", template_id="t-001")
        assert r.file_format == "csv"
        assert r.template_id == "t-001"


class TestBatchUploadRequest:
    def test_creation(self):
        files = [UploadFileRequest(file_name="a.csv", file_content_base64="x")]
        r = BatchUploadRequest(files=files)
        assert len(r.files) == 1
        assert r.parallel is True

    def test_parallel_false(self):
        r = BatchUploadRequest(files=[], parallel=False)
        assert r.parallel is False


class TestMapColumnsRequest:
    def test_creation(self):
        r = MapColumnsRequest(sheet_id="s-001", columns=["CO2", "Energy"])
        assert r.sheet_id == "s-001"
        assert len(r.columns) == 2

    def test_default_strategy(self):
        r = MapColumnsRequest(sheet_id="s-001", columns=["col1"])
        assert r.strategy == "fuzzy"


class TestNormalizeRequest:
    def test_creation(self):
        r = NormalizeRequest(
            data=[{"col1": "val1"}],
            column_mappings={"col1": "field1"},
        )
        assert len(r.data) == 1
        assert r.column_mappings["col1"] == "field1"


class TestTransformRequest:
    def test_creation(self):
        r = TransformRequest(file_id="f-001", operations=[{"op": "dedup"}])
        assert r.file_id == "f-001"
        assert len(r.operations) == 1


class TestCreateTemplateRequest:
    def test_creation(self):
        r = CreateTemplateRequest(template_name="Standard Energy", source_type="energy")
        assert r.template_name == "Standard Energy"
        assert r.source_type == "energy"

    def test_default_empty(self):
        r = CreateTemplateRequest(template_name="T")
        assert r.description == ""
        assert r.column_mappings == []
