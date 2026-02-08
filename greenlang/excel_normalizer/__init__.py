# -*- coding: utf-8 -*-
"""
GL-DATA-X-016: GreenLang Excel & CSV Normalizer Service SDK
=============================================================

This package provides spreadsheet parsing, CSV/TSV ingestion, column mapping,
data type detection, schema validation, data quality scoring, data
transformation, and provenance tracking SDK for the GreenLang framework. It
supports:

- Excel (.xlsx, .xls) and CSV/TSV file ingestion
- Automatic encoding and delimiter detection
- Column mapping via exact, fuzzy, synonym, and pattern strategies
- Statistical data type detection (14 types including currency, percentage, units)
- Schema validation with configurable rules
- Weighted data quality scoring (completeness, accuracy, consistency)
- Data transformations (pivot, unpivot, dedup, merge, filter, aggregate, etc.)
- Mapping template management for repeatable normalization
- Batch file processing with parallel workers
- SHA-256 provenance chain tracking for complete audit trails
- 12 Prometheus metrics for observability
- FastAPI REST API with 20 endpoints
- Thread-safe configuration with GL_EXCEL_NORMALIZER_ env prefix

Key Components:
    - config: ExcelNormalizerConfig with GL_EXCEL_NORMALIZER_ env prefix
    - models: Pydantic v2 models for all data structures
    - excel_parser: Excel workbook parsing engine (.xlsx, .xls)
    - csv_parser: CSV/TSV parsing engine with encoding detection
    - column_mapper: Column mapping engine with fuzzy matching
    - data_type_detector: Statistical data type detection engine
    - schema_validator: Schema validation rule engine
    - data_quality_scorer: Weighted quality scoring engine
    - transform_engine: Data transformation engine
    - provenance: SHA-256 chain-hashed audit trails
    - metrics: 12 Prometheus metrics
    - api: FastAPI HTTP service
    - setup: ExcelNormalizerService facade

Example:
    >>> from greenlang.excel_normalizer import ExcelNormalizerService
    >>> service = ExcelNormalizerService()
    >>> # Process an uploaded spreadsheet
    >>> from greenlang.excel_normalizer import ExcelParser
    >>> parser = ExcelParser()

Agent ID: GL-DATA-X-016
Agent Name: Excel & CSV Data Normalizer Agent
"""

__version__ = "1.0.0"
__agent_id__ = "GL-DATA-X-016"
__agent_name__ = "Excel & CSV Data Normalizer Agent"

# SDK availability flag
EXCEL_NORMALIZER_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from greenlang.excel_normalizer.config import (
    ExcelNormalizerConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Models (enums, SDK)
# ---------------------------------------------------------------------------
from greenlang.excel_normalizer.models import (
    # Enumerations
    FileFormat,
    DelimiterType,
    DataType,
    MappingStrategy,
    JobStatus,
    QualityLevel,
    TransformOperation,
    # SDK models
    SpreadsheetFile,
    SheetMetadata,
    ColumnMapping,
    NormalizationJob,
    NormalizedRecord,
    MappingTemplate,
    DataQualityReport,
    ValidationFinding,
    TransformResult,
    ExcelStatistics,
    # Request models
    UploadFileRequest,
    BatchUploadRequest,
    MapColumnsRequest,
    NormalizeRequest,
    TransformRequest,
    CreateTemplateRequest,
)

# ---------------------------------------------------------------------------
# Core engines
# ---------------------------------------------------------------------------
from greenlang.excel_normalizer.excel_parser import ExcelParser
from greenlang.excel_normalizer.csv_parser import CSVParser
from greenlang.excel_normalizer.column_mapper import ColumnMapper
from greenlang.excel_normalizer.data_type_detector import DataTypeDetector
from greenlang.excel_normalizer.schema_validator import SchemaValidator
from greenlang.excel_normalizer.data_quality_scorer import DataQualityScorer
from greenlang.excel_normalizer.transform_engine import TransformEngine
from greenlang.excel_normalizer.provenance import ProvenanceTracker

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from greenlang.excel_normalizer.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    excel_files_processed_total,
    excel_processing_duration_seconds,
    excel_rows_normalized_total,
    excel_columns_mapped_total,
    excel_mapping_confidence,
    excel_quality_score,
    excel_validation_findings_total,
    excel_transforms_total,
    excel_type_detections_total,
    excel_batch_jobs_total,
    excel_active_jobs,
    excel_queue_size,
    # Helper functions
    record_file_processed,
    record_rows_normalized,
    record_columns_mapped,
    record_mapping_confidence,
    record_quality_score,
    record_validation_finding,
    record_transform,
    record_type_detection,
    record_batch_job,
    update_active_jobs,
    update_queue_size,
)

# ---------------------------------------------------------------------------
# Service setup facade
# ---------------------------------------------------------------------------
from greenlang.excel_normalizer.setup import (
    ExcelNormalizerService,
    configure_excel_normalizer,
    get_excel_normalizer,
    get_router,
)

__all__ = [
    # Version
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # SDK flag
    "EXCEL_NORMALIZER_SDK_AVAILABLE",
    # Configuration
    "ExcelNormalizerConfig",
    "get_config",
    "set_config",
    "reset_config",
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
    # Core engines
    "ExcelParser",
    "CSVParser",
    "ColumnMapper",
    "DataTypeDetector",
    "SchemaValidator",
    "DataQualityScorer",
    "TransformEngine",
    "ProvenanceTracker",
    # Metric objects
    "PROMETHEUS_AVAILABLE",
    "excel_files_processed_total",
    "excel_processing_duration_seconds",
    "excel_rows_normalized_total",
    "excel_columns_mapped_total",
    "excel_mapping_confidence",
    "excel_quality_score",
    "excel_validation_findings_total",
    "excel_transforms_total",
    "excel_type_detections_total",
    "excel_batch_jobs_total",
    "excel_active_jobs",
    "excel_queue_size",
    # Metric helper functions
    "record_file_processed",
    "record_rows_normalized",
    "record_columns_mapped",
    "record_mapping_confidence",
    "record_quality_score",
    "record_validation_finding",
    "record_transform",
    "record_type_detection",
    "record_batch_job",
    "update_active_jobs",
    "update_queue_size",
    # Service setup facade
    "ExcelNormalizerService",
    "configure_excel_normalizer",
    "get_excel_normalizer",
    "get_router",
]
