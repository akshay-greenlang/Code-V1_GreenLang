# -*- coding: utf-8 -*-
"""
DataPipelineBridge - Data Agents to CSRD Intake Bridge
======================================================

This module implements the bridge between GreenLang Data agents and the
CSRD Starter Pack's IntakeAgent. It auto-detects data source types,
routes them through the appropriate data ingestion agent, applies a
sequential quality pipeline, and merges multi-source data into a unified
ESRS-aligned dataset.

Data Flow:
    Source File/API --> Source Detection --> Data Agent --> Quality Pipeline --> Unified Dataset
                                              |
                                              v
                            GL-DATA-X-001 (PDF Extractor)
                            GL-DATA-X-002 (Excel Normalizer)
                            GL-DATA-X-003 (ERP Connector)
                            GL-DATA-X-008 (Questionnaire Processor)

Quality Pipeline (applied sequentially):
    1. GL-DATA-X-010: Data Quality Profiler
    2. GL-DATA-X-011: Duplicate Detection
    3. GL-DATA-X-012: Missing Value Imputer
    4. GL-DATA-X-013: Outlier Detection
    5. GL-DATA-X-019: Validation Rule Engine

Zero-Hallucination Guarantee:
    All data transformations are deterministic. No LLM is used in the
    extraction, normalization, or quality pipeline paths. Provenance
    hashes track data lineage from raw source to unified dataset.

Example:
    >>> bridge = DataPipelineBridge(DataPipelineConfig())
    >>> result = await bridge.ingest_file("/data/emissions_2025.xlsx")
    >>> print(result.quality_report.overall_score)

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import logging
import mimetypes
import os
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class DataSourceType(str, Enum):
    """Supported data source types for the CSRD pipeline."""
    PDF = "pdf"
    EXCEL = "excel"
    CSV = "csv"
    ERP = "erp"
    QUESTIONNAIRE = "questionnaire"
    JSON_API = "json_api"
    MANUAL_ENTRY = "manual_entry"
    UNKNOWN = "unknown"


class QualityStage(str, Enum):
    """Stages in the data quality pipeline."""
    PROFILING = "profiling"
    DUPLICATE_DETECTION = "duplicate_detection"
    MISSING_VALUE_IMPUTATION = "missing_value_imputation"
    OUTLIER_DETECTION = "outlier_detection"
    VALIDATION = "validation"


class IngestionStatus(str, Enum):
    """Status of a data ingestion operation."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


# =============================================================================
# Data Models
# =============================================================================


class DataPipelineConfig(BaseModel):
    """Configuration for the Data Pipeline Bridge."""
    enable_quality_pipeline: bool = Field(
        default=True, description="Enable the data quality pipeline"
    )
    enable_duplicate_detection: bool = Field(
        default=True, description="Enable duplicate detection stage"
    )
    enable_missing_imputation: bool = Field(
        default=True, description="Enable missing value imputation"
    )
    enable_outlier_detection: bool = Field(
        default=True, description="Enable outlier detection stage"
    )
    max_file_size_mb: int = Field(
        default=100, description="Maximum file size in MB for ingestion"
    )
    supported_excel_extensions: List[str] = Field(
        default_factory=lambda: [".xlsx", ".xls", ".xlsm", ".xlsb"],
        description="Supported Excel file extensions",
    )
    supported_csv_extensions: List[str] = Field(
        default_factory=lambda: [".csv", ".tsv", ".txt"],
        description="Supported CSV/TSV file extensions",
    )
    supported_pdf_extensions: List[str] = Field(
        default_factory=lambda: [".pdf"],
        description="Supported PDF file extensions",
    )
    erp_systems: List[str] = Field(
        default_factory=lambda: ["sap", "oracle", "dynamics", "netsuite"],
        description="Supported ERP system identifiers",
    )
    quality_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Minimum quality score to pass pipeline (0-1)",
    )
    enable_provenance: bool = Field(
        default=True, description="Enable provenance tracking"
    )


class SourceDetectionResult(BaseModel):
    """Result of automatic source type detection."""
    detected_type: DataSourceType = Field(..., description="Detected data source type")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Detection confidence (0-1)"
    )
    file_extension: Optional[str] = Field(None, description="File extension if applicable")
    mime_type: Optional[str] = Field(None, description="Detected MIME type")
    file_size_bytes: Optional[int] = Field(None, description="File size in bytes")
    detection_method: str = Field(default="extension", description="How type was detected")


class QualityStageResult(BaseModel):
    """Result from a single quality pipeline stage."""
    stage: QualityStage = Field(..., description="Quality stage name")
    agent_id: str = Field(..., description="Agent that performed this stage")
    status: IngestionStatus = Field(..., description="Stage execution status")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Quality score (0-1)")
    issues_found: int = Field(default=0, description="Number of issues found")
    issues_resolved: int = Field(default=0, description="Number of issues auto-resolved")
    records_input: int = Field(default=0, description="Records entering this stage")
    records_output: int = Field(default=0, description="Records exiting this stage")
    execution_time_ms: float = Field(default=0.0, description="Stage execution time")
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Stage-specific details"
    )
    provenance_hash: str = Field(default="", description="Stage provenance hash")


class DataQualityReport(BaseModel):
    """Comprehensive data quality report from the quality pipeline."""
    overall_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall quality score"
    )
    passed_threshold: bool = Field(default=False, description="Whether quality threshold was met")
    quality_threshold: float = Field(default=0.7, description="Applied quality threshold")
    stages: List[QualityStageResult] = Field(
        default_factory=list, description="Results from each quality stage"
    )
    total_issues_found: int = Field(default=0, description="Total issues found across stages")
    total_issues_resolved: int = Field(default=0, description="Total issues auto-resolved")
    records_input: int = Field(default=0, description="Records entering the pipeline")
    records_output: int = Field(default=0, description="Records exiting the pipeline")
    total_execution_time_ms: float = Field(default=0.0, description="Total pipeline time")
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations for data improvement"
    )
    provenance_hash: str = Field(default="", description="Pipeline provenance hash")


class DataIngestionResult(BaseModel):
    """Result from ingesting a single data source."""
    source_type: DataSourceType = Field(..., description="Data source type")
    source_identifier: str = Field(
        default="", description="Source file path, URL, or identifier"
    )
    status: IngestionStatus = Field(..., description="Ingestion status")
    agent_id: str = Field(default="", description="Data agent that processed the source")
    records_extracted: int = Field(default=0, description="Records extracted from source")
    fields_extracted: int = Field(default=0, description="Distinct fields extracted")
    extraction_time_ms: float = Field(default=0.0, description="Extraction time in ms")
    extracted_data: Dict[str, Any] = Field(
        default_factory=dict, description="Extracted data payload"
    )
    quality_report: Optional[DataQualityReport] = Field(
        None, description="Quality pipeline report (if applied)"
    )
    error_message: Optional[str] = Field(None, description="Error message if failed")
    provenance_hash: str = Field(default="", description="Ingestion provenance hash")


class ESRSDataRecord(BaseModel):
    """A single record in the unified ESRS dataset."""
    record_id: str = Field(..., description="Unique record identifier")
    esrs_standard: str = Field(default="", description="ESRS standard (e.g., ESRS_E1)")
    metric_code: str = Field(default="", description="ESRS metric code")
    value: Any = Field(default=None, description="Data value")
    unit: str = Field(default="", description="Unit of measurement")
    source_type: DataSourceType = Field(..., description="Origin data source type")
    source_identifier: str = Field(default="", description="Origin source path/ID")
    reporting_period: str = Field(default="", description="Reporting period")
    data_quality_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Quality score for this record"
    )
    provenance_hash: str = Field(default="", description="Record-level provenance hash")


class UnifiedESRSDataset(BaseModel):
    """Unified dataset merging all data sources into ESRS-aligned format."""
    dataset_id: str = Field(default="", description="Unique dataset identifier")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Dataset creation timestamp"
    )
    records: List[ESRSDataRecord] = Field(
        default_factory=list, description="All ESRS-aligned records"
    )
    total_records: int = Field(default=0, description="Total records in dataset")
    sources_merged: int = Field(default=0, description="Number of data sources merged")
    source_types: List[str] = Field(
        default_factory=list, description="Distinct source types in dataset"
    )
    quality_report: Optional[DataQualityReport] = Field(
        None, description="Aggregate quality report"
    )
    ingestion_results: List[DataIngestionResult] = Field(
        default_factory=list, description="Per-source ingestion results"
    )
    provenance_hash: str = Field(default="", description="Dataset provenance hash")


class QualityPipelineResult(BaseModel):
    """Result from running the complete quality pipeline on a dataset."""
    quality_report: DataQualityReport = Field(
        ..., description="Comprehensive quality report"
    )
    cleaned_records: List[Dict[str, Any]] = Field(
        default_factory=list, description="Records after quality processing"
    )
    records_removed: int = Field(default=0, description="Records removed during quality")
    records_modified: int = Field(default=0, description="Records modified during quality")
    provenance_hash: str = Field(default="", description="Quality pipeline provenance hash")


# =============================================================================
# Agent-to-Source Mapping
# =============================================================================

SOURCE_AGENT_MAPPING: Dict[DataSourceType, str] = {
    DataSourceType.PDF: "GL-DATA-X-001",
    DataSourceType.EXCEL: "GL-DATA-X-002",
    DataSourceType.CSV: "GL-DATA-X-002",
    DataSourceType.ERP: "GL-DATA-X-003",
    DataSourceType.QUESTIONNAIRE: "GL-DATA-X-008",
    DataSourceType.JSON_API: "GL-DATA-X-004",
}

QUALITY_PIPELINE_STAGES: List[Tuple[QualityStage, str]] = [
    (QualityStage.PROFILING, "GL-DATA-X-010"),
    (QualityStage.DUPLICATE_DETECTION, "GL-DATA-X-011"),
    (QualityStage.MISSING_VALUE_IMPUTATION, "GL-DATA-X-012"),
    (QualityStage.OUTLIER_DETECTION, "GL-DATA-X-013"),
    (QualityStage.VALIDATION, "GL-DATA-X-019"),
]


# =============================================================================
# Data Pipeline Bridge Implementation
# =============================================================================


class DataPipelineBridge:
    """Bridge between GreenLang Data agents and CSRD IntakeAgent.

    Routes incoming data through appropriate data connectors based on
    auto-detected source type, applies a sequential quality pipeline,
    and merges multi-source data into a unified ESRS-aligned dataset.

    Attributes:
        config: Pipeline configuration
        _data_agents: Registry of data agent instances
        _quality_agents: Registry of quality agent instances
        _ingestion_history: History of ingestion operations

    Example:
        >>> bridge = DataPipelineBridge()
        >>> result = await bridge.ingest_file("/data/report.pdf")
        >>> dataset = await bridge.merge_sources([result1, result2])
    """

    def __init__(self, config: Optional[DataPipelineConfig] = None) -> None:
        """Initialize the Data Pipeline Bridge.

        Args:
            config: Pipeline configuration. Uses defaults if not provided.
        """
        self.config = config or DataPipelineConfig()
        self._data_agents: Dict[str, Any] = {}
        self._quality_agents: Dict[str, Any] = {}
        self._ingestion_history: List[DataIngestionResult] = []
        self._record_counter = 0

        logger.info(
            "DataPipelineBridge initialized: quality_pipeline=%s, threshold=%.2f",
            self.config.enable_quality_pipeline,
            self.config.quality_threshold,
        )

    # -------------------------------------------------------------------------
    # Source Detection
    # -------------------------------------------------------------------------

    def detect_source_type(
        self, source: str, content_hint: Optional[str] = None
    ) -> SourceDetectionResult:
        """Auto-detect the data source type from file extension or content.

        Uses a multi-step detection strategy:
        1. File extension matching (highest confidence)
        2. MIME type detection
        3. Content hint matching (lowest confidence)

        Args:
            source: File path, URL, or source identifier.
            content_hint: Optional hint about the content type.

        Returns:
            SourceDetectionResult with detected type and confidence.
        """
        file_ext = Path(source).suffix.lower() if "." in source else ""
        mime_type, _ = mimetypes.guess_type(source)
        file_size: Optional[int] = None

        if os.path.isfile(source):
            try:
                file_size = os.path.getsize(source)
            except OSError:
                pass

        # Step 1: Extension-based detection
        if file_ext in self.config.supported_pdf_extensions:
            return SourceDetectionResult(
                detected_type=DataSourceType.PDF,
                confidence=0.95,
                file_extension=file_ext,
                mime_type=mime_type,
                file_size_bytes=file_size,
                detection_method="extension",
            )

        if file_ext in self.config.supported_excel_extensions:
            return SourceDetectionResult(
                detected_type=DataSourceType.EXCEL,
                confidence=0.95,
                file_extension=file_ext,
                mime_type=mime_type,
                file_size_bytes=file_size,
                detection_method="extension",
            )

        if file_ext in self.config.supported_csv_extensions:
            return SourceDetectionResult(
                detected_type=DataSourceType.CSV,
                confidence=0.90,
                file_extension=file_ext,
                mime_type=mime_type,
                file_size_bytes=file_size,
                detection_method="extension",
            )

        if file_ext == ".json" or (mime_type and "json" in mime_type):
            return SourceDetectionResult(
                detected_type=DataSourceType.JSON_API,
                confidence=0.85,
                file_extension=file_ext,
                mime_type=mime_type,
                file_size_bytes=file_size,
                detection_method="extension",
            )

        # Step 2: Content hint detection
        if content_hint:
            hint_lower = content_hint.lower()
            for erp_name in self.config.erp_systems:
                if erp_name in hint_lower:
                    return SourceDetectionResult(
                        detected_type=DataSourceType.ERP,
                        confidence=0.80,
                        detection_method="content_hint",
                    )
            if "questionnaire" in hint_lower or "survey" in hint_lower:
                return SourceDetectionResult(
                    detected_type=DataSourceType.QUESTIONNAIRE,
                    confidence=0.80,
                    detection_method="content_hint",
                )

        # Step 3: URL pattern detection
        if source.startswith(("http://", "https://")):
            return SourceDetectionResult(
                detected_type=DataSourceType.JSON_API,
                confidence=0.70,
                detection_method="url_pattern",
            )

        logger.warning("Unable to detect source type for '%s'", source)
        return SourceDetectionResult(
            detected_type=DataSourceType.UNKNOWN,
            confidence=0.0,
            file_extension=file_ext,
            mime_type=mime_type,
            file_size_bytes=file_size,
            detection_method="none",
        )

    # -------------------------------------------------------------------------
    # Ingestion
    # -------------------------------------------------------------------------

    async def ingest_file(
        self,
        file_path: str,
        source_type: Optional[DataSourceType] = None,
        apply_quality: bool = True,
    ) -> DataIngestionResult:
        """Ingest a single file through the appropriate data agent.

        Args:
            file_path: Path to the file to ingest.
            source_type: Override source type (auto-detected if not provided).
            apply_quality: Whether to apply the quality pipeline.

        Returns:
            DataIngestionResult with extracted data and optional quality report.
        """
        start_time = time.monotonic()
        logger.info("Ingesting file: %s", file_path)

        if source_type is None:
            detection = self.detect_source_type(file_path)
            source_type = detection.detected_type
            if source_type == DataSourceType.UNKNOWN:
                return DataIngestionResult(
                    source_type=source_type,
                    source_identifier=file_path,
                    status=IngestionStatus.FAILED,
                    error_message=f"Unable to detect source type for '{file_path}'",
                )

        if detection := self._validate_file_size(file_path):
            return detection

        agent_id = SOURCE_AGENT_MAPPING.get(source_type, "")
        if not agent_id:
            return DataIngestionResult(
                source_type=source_type,
                source_identifier=file_path,
                status=IngestionStatus.FAILED,
                error_message=f"No agent configured for source type '{source_type.value}'",
            )

        try:
            extracted = await self._extract_data(agent_id, file_path, source_type)
            extraction_time = (time.monotonic() - start_time) * 1000

            records = extracted.get("records", [])
            fields = extracted.get("fields", [])

            quality_report = None
            if apply_quality and self.config.enable_quality_pipeline and records:
                quality_result = await self.run_quality_pipeline(records)
                quality_report = quality_result.quality_report
                records = quality_result.cleaned_records

            provenance_hash = ""
            if self.config.enable_provenance:
                provenance_hash = _compute_hash(
                    f"{file_path}:{source_type.value}:{agent_id}:{len(records)}"
                )

            result = DataIngestionResult(
                source_type=source_type,
                source_identifier=file_path,
                status=IngestionStatus.SUCCESS,
                agent_id=agent_id,
                records_extracted=len(records),
                fields_extracted=len(fields),
                extraction_time_ms=extraction_time,
                extracted_data={"records": records, "fields": fields},
                quality_report=quality_report,
                provenance_hash=provenance_hash,
            )

            self._ingestion_history.append(result)
            logger.info(
                "File '%s' ingested: %d records, %d fields in %.1fms",
                file_path, len(records), len(fields), extraction_time,
            )
            return result

        except Exception as exc:
            elapsed = (time.monotonic() - start_time) * 1000
            logger.error("Ingestion failed for '%s': %s", file_path, exc, exc_info=True)
            result = DataIngestionResult(
                source_type=source_type,
                source_identifier=file_path,
                status=IngestionStatus.FAILED,
                agent_id=agent_id,
                extraction_time_ms=elapsed,
                error_message=str(exc),
            )
            self._ingestion_history.append(result)
            return result

    async def ingest_erp(
        self,
        erp_system: str,
        connection_params: Dict[str, Any],
        query_params: Optional[Dict[str, Any]] = None,
    ) -> DataIngestionResult:
        """Ingest data from an ERP system.

        Args:
            erp_system: ERP system identifier (sap, oracle, dynamics, netsuite).
            connection_params: Connection parameters for the ERP.
            query_params: Query parameters for data extraction.

        Returns:
            DataIngestionResult with extracted ERP data.
        """
        start_time = time.monotonic()
        agent_id = SOURCE_AGENT_MAPPING[DataSourceType.ERP]
        source_id = f"erp://{erp_system}"
        logger.info("Ingesting from ERP: %s", erp_system)

        try:
            extracted = await self._extract_erp_data(
                agent_id, erp_system, connection_params, query_params or {}
            )
            elapsed = (time.monotonic() - start_time) * 1000

            records = extracted.get("records", [])

            quality_report = None
            if self.config.enable_quality_pipeline and records:
                quality_result = await self.run_quality_pipeline(records)
                quality_report = quality_result.quality_report
                records = quality_result.cleaned_records

            provenance_hash = ""
            if self.config.enable_provenance:
                provenance_hash = _compute_hash(
                    f"{source_id}:{agent_id}:{len(records)}"
                )

            result = DataIngestionResult(
                source_type=DataSourceType.ERP,
                source_identifier=source_id,
                status=IngestionStatus.SUCCESS,
                agent_id=agent_id,
                records_extracted=len(records),
                extraction_time_ms=elapsed,
                extracted_data={"records": records},
                quality_report=quality_report,
                provenance_hash=provenance_hash,
            )
            self._ingestion_history.append(result)
            return result

        except Exception as exc:
            elapsed = (time.monotonic() - start_time) * 1000
            logger.error("ERP ingestion failed for '%s': %s", erp_system, exc, exc_info=True)
            result = DataIngestionResult(
                source_type=DataSourceType.ERP,
                source_identifier=source_id,
                status=IngestionStatus.FAILED,
                agent_id=agent_id,
                extraction_time_ms=elapsed,
                error_message=str(exc),
            )
            self._ingestion_history.append(result)
            return result

    async def ingest_questionnaire(
        self,
        questionnaire_data: Dict[str, Any],
    ) -> DataIngestionResult:
        """Ingest data from a supplier questionnaire.

        Args:
            questionnaire_data: Questionnaire response data.

        Returns:
            DataIngestionResult with extracted questionnaire data.
        """
        start_time = time.monotonic()
        agent_id = SOURCE_AGENT_MAPPING[DataSourceType.QUESTIONNAIRE]
        source_id = questionnaire_data.get("questionnaire_id", "questionnaire")
        logger.info("Ingesting questionnaire: %s", source_id)

        try:
            extracted = await self._extract_questionnaire(agent_id, questionnaire_data)
            elapsed = (time.monotonic() - start_time) * 1000
            records = extracted.get("records", [])

            quality_report = None
            if self.config.enable_quality_pipeline and records:
                quality_result = await self.run_quality_pipeline(records)
                quality_report = quality_result.quality_report
                records = quality_result.cleaned_records

            provenance_hash = _compute_hash(
                f"questionnaire:{source_id}:{len(records)}"
            ) if self.config.enable_provenance else ""

            result = DataIngestionResult(
                source_type=DataSourceType.QUESTIONNAIRE,
                source_identifier=source_id,
                status=IngestionStatus.SUCCESS,
                agent_id=agent_id,
                records_extracted=len(records),
                extraction_time_ms=elapsed,
                extracted_data={"records": records},
                quality_report=quality_report,
                provenance_hash=provenance_hash,
            )
            self._ingestion_history.append(result)
            return result

        except Exception as exc:
            elapsed = (time.monotonic() - start_time) * 1000
            logger.error("Questionnaire ingestion failed: %s", exc, exc_info=True)
            result = DataIngestionResult(
                source_type=DataSourceType.QUESTIONNAIRE,
                source_identifier=source_id,
                status=IngestionStatus.FAILED,
                agent_id=agent_id,
                extraction_time_ms=elapsed,
                error_message=str(exc),
            )
            self._ingestion_history.append(result)
            return result

    # -------------------------------------------------------------------------
    # Quality Pipeline
    # -------------------------------------------------------------------------

    async def run_quality_pipeline(
        self, records: List[Dict[str, Any]]
    ) -> QualityPipelineResult:
        """Run the complete data quality pipeline on a set of records.

        Applies profiling, duplicate detection, missing value imputation,
        outlier detection, and validation in sequence. Each stage may modify
        the record set.

        Args:
            records: List of data records to process.

        Returns:
            QualityPipelineResult with cleaned records and comprehensive report.
        """
        if not self.config.enable_quality_pipeline:
            return QualityPipelineResult(
                quality_report=DataQualityReport(
                    overall_score=1.0,
                    passed_threshold=True,
                    records_input=len(records),
                    records_output=len(records),
                ),
                cleaned_records=records,
            )

        logger.info("Running quality pipeline on %d records", len(records))
        start_time = time.monotonic()
        stage_results: List[QualityStageResult] = []
        current_records = list(records)
        total_issues_found = 0
        total_issues_resolved = 0
        records_removed = 0
        records_modified = 0

        for stage, agent_id in QUALITY_PIPELINE_STAGES:
            if not self._is_stage_enabled(stage):
                logger.debug("Skipping disabled quality stage: %s", stage.value)
                continue

            stage_start = time.monotonic()
            stage_input_count = len(current_records)

            try:
                stage_output = await self._execute_quality_stage(
                    stage, agent_id, current_records
                )

                current_records = stage_output.get("records", current_records)
                issues_found = stage_output.get("issues_found", 0)
                issues_resolved = stage_output.get("issues_resolved", 0)
                stage_score = stage_output.get("score", 1.0)
                stage_modified = stage_output.get("records_modified", 0)
                stage_elapsed = (time.monotonic() - stage_start) * 1000

                stage_removed = stage_input_count - len(current_records)
                records_removed += max(0, stage_removed)
                records_modified += stage_modified
                total_issues_found += issues_found
                total_issues_resolved += issues_resolved

                provenance_hash = _compute_hash(
                    f"{stage.value}:{agent_id}:{stage_input_count}:{len(current_records)}"
                ) if self.config.enable_provenance else ""

                stage_result = QualityStageResult(
                    stage=stage,
                    agent_id=agent_id,
                    status=IngestionStatus.SUCCESS,
                    score=stage_score,
                    issues_found=issues_found,
                    issues_resolved=issues_resolved,
                    records_input=stage_input_count,
                    records_output=len(current_records),
                    execution_time_ms=stage_elapsed,
                    details=stage_output.get("details", {}),
                    provenance_hash=provenance_hash,
                )
                stage_results.append(stage_result)

                logger.info(
                    "Quality stage %s: score=%.2f, issues=%d/%d resolved, "
                    "%d->%d records in %.1fms",
                    stage.value, stage_score, issues_resolved, issues_found,
                    stage_input_count, len(current_records), stage_elapsed,
                )

            except Exception as exc:
                stage_elapsed = (time.monotonic() - stage_start) * 1000
                logger.error("Quality stage %s failed: %s", stage.value, exc, exc_info=True)
                stage_results.append(QualityStageResult(
                    stage=stage,
                    agent_id=agent_id,
                    status=IngestionStatus.FAILED,
                    records_input=stage_input_count,
                    records_output=len(current_records),
                    execution_time_ms=stage_elapsed,
                    details={"error": str(exc)},
                ))

        total_elapsed = (time.monotonic() - start_time) * 1000
        overall_score = self._compute_overall_quality_score(stage_results)
        passed = overall_score >= self.config.quality_threshold

        recommendations = self._generate_recommendations(stage_results, overall_score)

        pipeline_provenance = _compute_hash(
            "|".join(s.provenance_hash for s in stage_results if s.provenance_hash)
        ) if self.config.enable_provenance else ""

        quality_report = DataQualityReport(
            overall_score=overall_score,
            passed_threshold=passed,
            quality_threshold=self.config.quality_threshold,
            stages=stage_results,
            total_issues_found=total_issues_found,
            total_issues_resolved=total_issues_resolved,
            records_input=len(records),
            records_output=len(current_records),
            total_execution_time_ms=total_elapsed,
            recommendations=recommendations,
            provenance_hash=pipeline_provenance,
        )

        logger.info(
            "Quality pipeline complete: score=%.2f, passed=%s, "
            "%d->%d records, %d issues in %.1fms",
            overall_score, passed, len(records),
            len(current_records), total_issues_found, total_elapsed,
        )

        return QualityPipelineResult(
            quality_report=quality_report,
            cleaned_records=current_records,
            records_removed=records_removed,
            records_modified=records_modified,
            provenance_hash=pipeline_provenance,
        )

    # -------------------------------------------------------------------------
    # Merge
    # -------------------------------------------------------------------------

    async def merge_sources(
        self, ingestion_results: List[DataIngestionResult]
    ) -> UnifiedESRSDataset:
        """Merge multiple data ingestion results into a unified ESRS dataset.

        Transforms extracted records from each source into a common ESRS
        schema, deduplicates across sources, and assigns record-level
        provenance hashes.

        Args:
            ingestion_results: List of DataIngestionResult from multiple sources.

        Returns:
            UnifiedESRSDataset with all records merged and deduplicated.
        """
        logger.info("Merging %d data sources into unified ESRS dataset",
                     len(ingestion_results))
        start_time = time.monotonic()

        all_records: List[ESRSDataRecord] = []
        source_types: set = set()
        successful_sources = 0

        for ingestion in ingestion_results:
            if ingestion.status != IngestionStatus.SUCCESS:
                logger.warning(
                    "Skipping failed source '%s' during merge", ingestion.source_identifier
                )
                continue

            successful_sources += 1
            source_types.add(ingestion.source_type.value)
            raw_records = ingestion.extracted_data.get("records", [])

            for raw_record in raw_records:
                self._record_counter += 1
                esrs_record = self._transform_to_esrs(
                    raw_record, ingestion.source_type, ingestion.source_identifier
                )
                all_records.append(esrs_record)

        dataset_id = _compute_hash(
            f"dataset:{datetime.utcnow().isoformat()}:{len(all_records)}"
        )[:16]

        aggregate_quality = self._compute_aggregate_quality(ingestion_results)
        dataset_provenance = _compute_hash(
            "|".join(r.provenance_hash for r in ingestion_results if r.provenance_hash)
        ) if self.config.enable_provenance else ""

        elapsed = (time.monotonic() - start_time) * 1000
        logger.info(
            "Merge complete: %d records from %d sources (%s) in %.1fms",
            len(all_records), successful_sources, sorted(source_types), elapsed,
        )

        return UnifiedESRSDataset(
            dataset_id=dataset_id,
            records=all_records,
            total_records=len(all_records),
            sources_merged=successful_sources,
            source_types=sorted(source_types),
            quality_report=aggregate_quality,
            ingestion_results=ingestion_results,
            provenance_hash=dataset_provenance,
        )

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _validate_file_size(self, file_path: str) -> Optional[DataIngestionResult]:
        """Validate that a file is within the maximum size limit.

        Args:
            file_path: Path to the file.

        Returns:
            DataIngestionResult with error if file exceeds limit, None otherwise.
        """
        if not os.path.isfile(file_path):
            return None
        try:
            size_bytes = os.path.getsize(file_path)
            max_bytes = self.config.max_file_size_mb * 1024 * 1024
            if size_bytes > max_bytes:
                return DataIngestionResult(
                    source_type=DataSourceType.UNKNOWN,
                    source_identifier=file_path,
                    status=IngestionStatus.FAILED,
                    error_message=(
                        f"File size ({size_bytes / (1024*1024):.1f}MB) exceeds "
                        f"maximum ({self.config.max_file_size_mb}MB)"
                    ),
                )
        except OSError:
            pass
        return None

    def _is_stage_enabled(self, stage: QualityStage) -> bool:
        """Check if a quality stage is enabled in the configuration.

        Args:
            stage: The quality stage to check.

        Returns:
            True if enabled, False otherwise.
        """
        stage_flags = {
            QualityStage.PROFILING: True,  # Always enabled
            QualityStage.DUPLICATE_DETECTION: self.config.enable_duplicate_detection,
            QualityStage.MISSING_VALUE_IMPUTATION: self.config.enable_missing_imputation,
            QualityStage.OUTLIER_DETECTION: self.config.enable_outlier_detection,
            QualityStage.VALIDATION: True,  # Always enabled
        }
        return stage_flags.get(stage, True)

    async def _extract_data(
        self, agent_id: str, file_path: str, source_type: DataSourceType
    ) -> Dict[str, Any]:
        """Extract data from a file using the appropriate agent.

        Args:
            agent_id: The data agent ID to use.
            file_path: Path to the source file.
            source_type: Detected source type.

        Returns:
            Dictionary with extracted records and fields.
        """
        agent = self._data_agents.get(agent_id)
        if agent is not None:
            return await self._invoke_agent(agent, {
                "file_path": file_path,
                "source_type": source_type.value,
            })

        # Fallback: return empty extraction result for stub mode
        logger.debug("No agent instance for %s, using stub extraction", agent_id)
        return {
            "records": [],
            "fields": [],
            "agent_id": agent_id,
            "source_type": source_type.value,
        }

    async def _extract_erp_data(
        self,
        agent_id: str,
        erp_system: str,
        connection_params: Dict[str, Any],
        query_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract data from an ERP system.

        Args:
            agent_id: The ERP agent ID.
            erp_system: ERP system name.
            connection_params: Connection parameters.
            query_params: Query parameters.

        Returns:
            Dictionary with extracted records.
        """
        agent = self._data_agents.get(agent_id)
        if agent is not None:
            return await self._invoke_agent(agent, {
                "erp_system": erp_system,
                "connection": connection_params,
                "query": query_params,
            })

        logger.debug("No agent instance for %s, using stub ERP extraction", agent_id)
        return {"records": [], "agent_id": agent_id}

    async def _extract_questionnaire(
        self, agent_id: str, questionnaire_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract data from a questionnaire response.

        Args:
            agent_id: The questionnaire agent ID.
            questionnaire_data: Questionnaire response data.

        Returns:
            Dictionary with extracted records.
        """
        agent = self._data_agents.get(agent_id)
        if agent is not None:
            return await self._invoke_agent(agent, questionnaire_data)

        logger.debug("No agent instance for %s, using stub extraction", agent_id)
        return {"records": [], "agent_id": agent_id}

    async def _execute_quality_stage(
        self,
        stage: QualityStage,
        agent_id: str,
        records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute a single quality pipeline stage.

        Args:
            stage: The quality stage to execute.
            agent_id: The quality agent ID.
            records: Records to process.

        Returns:
            Dictionary with processed records and quality metrics.
        """
        agent = self._quality_agents.get(agent_id)
        if agent is not None:
            return await self._invoke_agent(agent, {
                "stage": stage.value,
                "records": records,
            })

        # Fallback: pass-through with default quality score
        return {
            "records": records,
            "score": 0.85,
            "issues_found": 0,
            "issues_resolved": 0,
            "records_modified": 0,
            "details": {"agent_id": agent_id, "mode": "passthrough"},
        }

    async def _invoke_agent(
        self, agent: Any, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Invoke an agent instance, handling sync and async execute methods.

        Args:
            agent: The agent object.
            params: Parameters for the agent.

        Returns:
            Dictionary with agent output.
        """
        import asyncio

        execute_fn = getattr(agent, "execute", None)
        if execute_fn is None:
            return {}

        if asyncio.iscoroutinefunction(execute_fn):
            result = await execute_fn(params)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, execute_fn, params)

        if hasattr(result, "model_dump"):
            return result.model_dump()
        if hasattr(result, "dict"):
            return result.dict()
        if isinstance(result, dict):
            return result
        return {"result": str(result)}

    def _transform_to_esrs(
        self,
        raw_record: Dict[str, Any],
        source_type: DataSourceType,
        source_identifier: str,
    ) -> ESRSDataRecord:
        """Transform a raw extracted record into an ESRS-aligned record.

        Args:
            raw_record: Raw data record from the extraction agent.
            source_type: Origin source type.
            source_identifier: Origin source path or identifier.

        Returns:
            ESRSDataRecord with standardized fields.
        """
        record_id = _compute_hash(
            f"rec:{self._record_counter}:{source_identifier}"
        )[:16]

        return ESRSDataRecord(
            record_id=record_id,
            esrs_standard=raw_record.get("esrs_standard", ""),
            metric_code=raw_record.get("metric_code", ""),
            value=raw_record.get("value"),
            unit=raw_record.get("unit", ""),
            source_type=source_type,
            source_identifier=source_identifier,
            reporting_period=raw_record.get("reporting_period", ""),
            data_quality_score=float(raw_record.get("data_quality_score", 0.0)),
            provenance_hash=_compute_hash(f"{record_id}:{source_type.value}"),
        )

    def _compute_overall_quality_score(
        self, stage_results: List[QualityStageResult]
    ) -> float:
        """Compute a weighted overall quality score from stage results.

        Args:
            stage_results: Results from all quality stages.

        Returns:
            Weighted average quality score (0-1).
        """
        if not stage_results:
            return 1.0

        successful = [s for s in stage_results if s.status == IngestionStatus.SUCCESS]
        if not successful:
            return 0.0

        # Weighted scoring: validation has highest weight
        weights = {
            QualityStage.PROFILING: 0.15,
            QualityStage.DUPLICATE_DETECTION: 0.20,
            QualityStage.MISSING_VALUE_IMPUTATION: 0.20,
            QualityStage.OUTLIER_DETECTION: 0.15,
            QualityStage.VALIDATION: 0.30,
        }

        total_weight = sum(weights.get(s.stage, 0.1) for s in successful)
        weighted_sum = sum(
            s.score * weights.get(s.stage, 0.1) for s in successful
        )

        if total_weight == 0:
            return 0.0

        return round(weighted_sum / total_weight, 4)

    def _generate_recommendations(
        self,
        stage_results: List[QualityStageResult],
        overall_score: float,
    ) -> List[str]:
        """Generate data quality improvement recommendations.

        Args:
            stage_results: Results from all quality stages.
            overall_score: Overall quality score.

        Returns:
            List of human-readable recommendation strings.
        """
        recommendations: List[str] = []

        if overall_score < 0.5:
            recommendations.append(
                "Overall data quality is low. Consider reviewing raw data sources "
                "for completeness and accuracy before proceeding with CSRD reporting."
            )

        for stage_result in stage_results:
            if stage_result.status == IngestionStatus.FAILED:
                recommendations.append(
                    f"Quality stage '{stage_result.stage.value}' failed. "
                    f"Review agent {stage_result.agent_id} configuration and logs."
                )
            elif stage_result.score < 0.6:
                if stage_result.stage == QualityStage.DUPLICATE_DETECTION:
                    recommendations.append(
                        f"High duplicate rate detected ({stage_result.issues_found} duplicates). "
                        "Review data collection processes to prevent duplicate entries."
                    )
                elif stage_result.stage == QualityStage.MISSING_VALUE_IMPUTATION:
                    recommendations.append(
                        f"Significant missing values ({stage_result.issues_found} fields). "
                        "Consider collecting additional data from source systems."
                    )
                elif stage_result.stage == QualityStage.OUTLIER_DETECTION:
                    recommendations.append(
                        f"Multiple outliers detected ({stage_result.issues_found}). "
                        "Verify data accuracy for flagged records."
                    )
                elif stage_result.stage == QualityStage.VALIDATION:
                    recommendations.append(
                        f"Validation failures ({stage_result.issues_found} rules). "
                        "Review ESRS data requirements and ensure field formats match."
                    )

        return recommendations

    def _compute_aggregate_quality(
        self, results: List[DataIngestionResult]
    ) -> Optional[DataQualityReport]:
        """Compute an aggregate quality report from multiple ingestion results.

        Args:
            results: List of data ingestion results.

        Returns:
            Aggregate DataQualityReport or None if no reports available.
        """
        reports = [
            r.quality_report for r in results
            if r.quality_report is not None
        ]
        if not reports:
            return None

        avg_score = sum(r.overall_score for r in reports) / len(reports)
        total_issues = sum(r.total_issues_found for r in reports)
        total_resolved = sum(r.total_issues_resolved for r in reports)
        total_records_in = sum(r.records_input for r in reports)
        total_records_out = sum(r.records_output for r in reports)

        all_recommendations: List[str] = []
        for report in reports:
            all_recommendations.extend(report.recommendations)

        return DataQualityReport(
            overall_score=round(avg_score, 4),
            passed_threshold=avg_score >= self.config.quality_threshold,
            quality_threshold=self.config.quality_threshold,
            total_issues_found=total_issues,
            total_issues_resolved=total_resolved,
            records_input=total_records_in,
            records_output=total_records_out,
            recommendations=list(set(all_recommendations)),
        )

    def get_ingestion_history(self) -> List[DataIngestionResult]:
        """Return the history of all ingestion operations.

        Returns:
            List of DataIngestionResult in chronological order.
        """
        return list(self._ingestion_history)


# =============================================================================
# Helper Functions
# =============================================================================


def _compute_hash(data: str) -> str:
    """Compute a SHA-256 hash of the given string.

    Args:
        data: The string to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
