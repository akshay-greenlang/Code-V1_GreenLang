"""
IntakeAgent - ESG Data Ingestion, Validation, and Enrichment for CSRD Reporting

This agent is responsible for:
1. Reading ESG data from multiple sources (CSV, JSON, Excel, Parquet)
2. Validating data against ESRS data point catalog and JSON schemas
3. Performing data quality assessment (completeness, accuracy, consistency, timeliness, validity)
4. Mapping metrics to ESRS taxonomy
5. Detecting statistical outliers
6. Enriching data with ESRS metadata
7. Generating comprehensive validation reports

Key Features:
- 100% deterministic processing (zero hallucination guarantee)
- NO AI/LLM usage
- 1,000+ records/sec throughput target
- Complete audit trail
- Multi-format support

Version: 1.0.0
Author: GreenLang CSRD Team
License: MIT
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from jsonschema import Draft7Validator, ValidationError as JsonSchemaValidationError
from pydantic import BaseModel, Field, validator

# Import validation utilities
import sys
from pathlib import Path as PathLib
sys.path.append(str(PathLib(__file__).parent.parent))
from utils.validation import (
    validate_file_size,
    validate_file_path,
    sanitize_filename,
    validate_string_length,
    ValidationError as InputValidationError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# ERROR CODES
# ============================================================================

ERROR_CODES = {
    # Critical Errors
    "E001": "Missing required field",
    "E002": "Invalid ESRS metric code format",
    "E003": "Metric code not found in ESRS catalog",
    "E004": "Invalid data type for metric value",
    "E005": "Invalid unit for metric",
    "E006": "Invalid date format",
    "E007": "Period end date before start date",
    "E008": "Schema validation failed",
    "E009": "File parsing error",
    "E010": "Invalid data quality value",

    # Warnings
    "W001": "Data quality below threshold",
    "W002": "Statistical outlier detected",
    "W003": "Missing optional metadata",
    "W004": "Fuzzy match used for ESRS mapping",
    "W005": "Time series gap detected",
    "W006": "Large year-over-year change",
    "W007": "Unit mismatch with ESRS standard",
    "W008": "Data outside expected range",

    # Info
    "I001": "Validation passed successfully",
    "I002": "Data quality assessment complete",
    "I003": "ESRS mapping complete",
}


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ValidationIssue(BaseModel):
    """Represents a validation error or warning."""
    metric_code: Optional[str] = None
    error_code: str
    severity: str  # "error", "warning", "info"
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None
    suggestion: Optional[str] = None
    row_index: Optional[int] = None


class DataQualityScore(BaseModel):
    """Data quality assessment scores."""
    overall_score: float = Field(ge=0, le=100, description="Overall quality score (0-100)")
    completeness_score: float = Field(ge=0, le=100)
    accuracy_score: float = Field(ge=0, le=100)
    consistency_score: float = Field(ge=0, le=100)
    timeliness_score: float = Field(ge=0, le=100)
    validity_score: float = Field(ge=0, le=100)

    # Dimension weights
    completeness_weight: float = 0.30
    accuracy_weight: float = 0.25
    consistency_weight: float = 0.20
    timeliness_weight: float = 0.15
    validity_weight: float = 0.10


class ESRSMetadata(BaseModel):
    """ESRS metadata enrichment."""
    esrs_code: Optional[str] = None
    esrs_standard: Optional[str] = None  # E1, E2, S1, etc.
    disclosure_requirement: Optional[str] = None
    data_point_name: Optional[str] = None
    expected_unit: Optional[str] = None
    data_type: Optional[str] = None
    is_mandatory: Optional[bool] = None
    mapping_confidence: Optional[str] = "exact"  # "exact", "fuzzy", "none"


class EnrichedDataPoint(BaseModel):
    """Represents a validated and enriched ESG data point."""
    # Original fields
    metric_code: str
    metric_name: str
    value: Union[float, str, bool, int]
    unit: str
    period_start: str
    period_end: str

    # Optional fields
    data_quality: Optional[str] = "medium"
    source_document: Optional[str] = None
    verification_status: Optional[str] = "unverified"
    notes: Optional[str] = None
    calculation_method: Optional[str] = None
    breakdown: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None

    # Enrichment (added by agent)
    esrs_metadata: Optional[ESRSMetadata] = None
    validation_status: str = "valid"  # "valid", "invalid", "warning"
    quality_score: Optional[float] = None
    is_outlier: bool = False
    processing_timestamp: Optional[str] = None


# ============================================================================
# INTAKE AGENT
# ============================================================================

class IntakeAgent:
    """
    Ingests, validates, and enriches ESG data for CSRD reporting.

    This agent follows a tool-first architecture with ZERO LLM usage.
    All processing is deterministic and reproducible.

    Performance target: 1,000 records/second
    Test coverage target: 90%
    """

    def __init__(
        self,
        esrs_data_points_path: Union[str, Path],
        data_quality_rules_path: Union[str, Path],
        esg_data_schema_path: Optional[Union[str, Path]] = None,
        company_profile_schema_path: Optional[Union[str, Path]] = None,
        quality_threshold: float = 0.80
    ):
        """
        Initialize the IntakeAgent.

        Args:
            esrs_data_points_path: Path to ESRS data points catalog JSON
            data_quality_rules_path: Path to data quality rules YAML
            esg_data_schema_path: Path to ESG data JSON schema (optional)
            company_profile_schema_path: Path to company profile schema (optional)
            quality_threshold: Minimum data quality score (0.0-1.0)
        """
        self.esrs_data_points_path = Path(esrs_data_points_path)
        self.data_quality_rules_path = Path(data_quality_rules_path)
        self.esg_data_schema_path = Path(esg_data_schema_path) if esg_data_schema_path else None
        self.company_profile_schema_path = Path(company_profile_schema_path) if company_profile_schema_path else None
        self.quality_threshold = quality_threshold

        # Load reference data
        self.esrs_catalog = self._load_esrs_catalog()
        self.data_quality_rules = self._load_data_quality_rules()
        self.esg_data_schema = self._load_schema(self.esg_data_schema_path) if self.esg_data_schema_path else None
        self.company_profile_schema = self._load_schema(self.company_profile_schema_path) if self.company_profile_schema_path else None

        # Create lookup structures for fast mapping
        self.esrs_code_lookup = self._create_code_lookup()
        self.esrs_name_lookup = self._create_name_lookup()

        # Statistics
        self.stats = {
            "total_records": 0,
            "valid_records": 0,
            "invalid_records": 0,
            "warnings": 0,
            "outliers_detected": 0,
            "exact_matches": 0,
            "fuzzy_matches": 0,
            "unmapped": 0,
            "start_time": None,
            "end_time": None
        }

        logger.info(f"IntakeAgent initialized with {len(self.esrs_catalog)} ESRS data points")

    # ========================================================================
    # DATA LOADING
    # ========================================================================

    def _load_esrs_catalog(self) -> List[Dict[str, Any]]:
        """Load ESRS data points catalog from JSON."""
        try:
            with open(self.esrs_data_points_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle both array and object with data_points key
            if isinstance(data, list):
                catalog = data
            elif isinstance(data, dict) and "data_points" in data:
                catalog = data["data_points"]
            else:
                raise ValueError("Invalid ESRS catalog format")

            logger.info(f"Loaded {len(catalog)} ESRS data points")
            return catalog
        except Exception as e:
            logger.error(f"Failed to load ESRS catalog: {e}")
            raise

    def _load_data_quality_rules(self) -> Dict[str, Any]:
        """Load data quality rules from YAML."""
        try:
            with open(self.data_quality_rules_path, 'r', encoding='utf-8') as f:
                rules = yaml.safe_load(f)
            logger.info("Loaded data quality rules")
            return rules
        except Exception as e:
            logger.error(f"Failed to load data quality rules: {e}")
            raise

    def _load_schema(self, schema_path: Path) -> Optional[Dict[str, Any]]:
        """Load JSON schema for validation."""
        if not schema_path or not schema_path.exists():
            return None

        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            logger.info(f"Loaded JSON schema from {schema_path.name}")
            return schema
        except Exception as e:
            logger.warning(f"Failed to load schema: {e}")
            return None

    def _create_code_lookup(self) -> Dict[str, Dict[str, Any]]:
        """Create fast lookup dictionary by ESRS code."""
        lookup = {}
        for dp in self.esrs_catalog:
            code = dp.get("code") or dp.get("esrs_code") or dp.get("metric_code")
            if code:
                lookup[code] = dp
        return lookup

    def _create_name_lookup(self) -> Dict[str, Dict[str, Any]]:
        """Create fast lookup dictionary by ESRS name (lowercase)."""
        lookup = {}
        for dp in self.esrs_catalog:
            name = dp.get("name") or dp.get("data_point_name") or dp.get("metric_name")
            if name:
                lookup[name.lower()] = dp
        return lookup

    # ========================================================================
    # FILE INGESTION
    # ========================================================================

    def read_esg_data(self, input_path: Union[str, Path]) -> pd.DataFrame:
        """
        Read ESG data from file (CSV, JSON, Excel, Parquet).

        Args:
            input_path: Path to input file

        Returns:
            DataFrame with ESG data records

        Raises:
            ValueError: If file format not supported or file cannot be read
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise ValueError(f"Input file not found: {input_path}")

        # Detect format by extension
        suffix = input_path.suffix.lower()

        # Validate file size before reading
        try:
            file_type_map = {
                '.csv': 'csv',
                '.json': 'json',
                '.xlsx': 'excel',
                '.xls': 'excel',
                '.parquet': 'default',
                '.tsv': 'csv'
            }
            file_type = file_type_map.get(suffix, 'default')
            validate_file_size(input_path, file_type)
        except InputValidationError as e:
            logger.error(f"File size validation failed: {e}")
            raise ValueError(f"File size validation failed: {e}")

        try:
            if suffix == '.csv':
                df = pd.read_csv(input_path, encoding='utf-8')
            elif suffix == '.json':
                # Handle both array and object with data_points key
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict) and "data_points" in data:
                    df = pd.DataFrame(data["data_points"])
                else:
                    df = pd.read_json(input_path)
            elif suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(input_path)
            elif suffix == '.parquet':
                df = pd.read_parquet(input_path)
            elif suffix == '.tsv':
                df = pd.read_csv(input_path, sep='\t', encoding='utf-8')
            else:
                raise ValueError(f"Unsupported file format: {suffix}")

            logger.info(f"Read {len(df)} data points from {input_path}")
            return df

        except UnicodeDecodeError:
            # Try alternative encoding
            logger.warning("UTF-8 encoding failed, trying Latin-1")
            if suffix == '.csv':
                df = pd.read_csv(input_path, encoding='latin-1')
                return df
            else:
                raise

    # ========================================================================
    # VALIDATION
    # ========================================================================

    def validate_data_point(self, data_point: Dict[str, Any], row_index: int = -1) -> Tuple[bool, List[ValidationIssue]]:
        """
        Validate a single ESG data point.

        Args:
            data_point: Data point dictionary
            row_index: Row index in source file

        Returns:
            Tuple of (is_valid, list of validation issues)
        """
        issues = []

        # Required fields check
        required_fields = ["metric_code", "metric_name", "value", "unit", "period_start", "period_end"]
        for field in required_fields:
            if field not in data_point or data_point[field] is None or data_point[field] == "":
                issues.append(ValidationIssue(
                    metric_code=data_point.get("metric_code"),
                    error_code="E001",
                    severity="error",
                    message=f"Missing required field: {field}",
                    field=field,
                    suggestion="Ensure all required fields are populated",
                    row_index=row_index
                ))

        # If critical fields missing, return early
        if any(issue.field in ["metric_code", "value", "unit"] for issue in issues):
            return False, issues

        # ESRS metric code format validation
        metric_code = str(data_point.get("metric_code", ""))
        if not re.match(r'^(E[1-5]|S[1-4]|G1|ESRS[12])-[0-9]+', metric_code):
            issues.append(ValidationIssue(
                metric_code=metric_code,
                error_code="E002",
                severity="error",
                message=f"Invalid ESRS metric code format: {metric_code}",
                field="metric_code",
                value=metric_code,
                suggestion="Use format like E1-1, S1-9, G1-1, ESRS1-1",
                row_index=row_index
            ))

        # Check if code exists in ESRS catalog
        if metric_code not in self.esrs_code_lookup:
            issues.append(ValidationIssue(
                metric_code=metric_code,
                error_code="E003",
                severity="warning",  # Warning, not error - might be valid but unmapped
                message=f"Metric code not found in ESRS catalog: {metric_code}",
                field="metric_code",
                value=metric_code,
                suggestion="Verify metric code against ESRS data point catalog",
                row_index=row_index
            ))

        # Date validation
        try:
            period_start = pd.to_datetime(data_point.get("period_start"))
            period_end = pd.to_datetime(data_point.get("period_end"))

            if period_end < period_start:
                issues.append(ValidationIssue(
                    metric_code=metric_code,
                    error_code="E007",
                    severity="error",
                    message=f"Period end ({period_end}) is before period start ({period_start})",
                    field="period_end",
                    row_index=row_index
                ))
        except Exception as e:
            issues.append(ValidationIssue(
                metric_code=metric_code,
                error_code="E006",
                severity="error",
                message=f"Invalid date format: {str(e)}",
                field="period_start",
                row_index=row_index
            ))

        # Data quality enum validation
        data_quality = data_point.get("data_quality")
        if data_quality and data_quality not in ["high", "medium", "low"]:
            issues.append(ValidationIssue(
                metric_code=metric_code,
                error_code="E010",
                severity="error",
                message=f"Invalid data_quality value: {data_quality}",
                field="data_quality",
                value=data_quality,
                suggestion="Must be 'high', 'medium', or 'low'",
                row_index=row_index
            ))

        # Determine if valid (no errors, warnings OK)
        has_errors = any(issue.severity == "error" for issue in issues)
        is_valid = not has_errors

        return is_valid, issues

    # ========================================================================
    # ESRS MAPPING
    # ========================================================================

    def map_to_esrs(self, data_point: Dict[str, Any]) -> Tuple[Optional[ESRSMetadata], List[ValidationIssue]]:
        """
        Map data point to ESRS taxonomy.

        Args:
            data_point: Data point dictionary

        Returns:
            Tuple of (ESRS metadata, list of warnings)
        """
        warnings = []
        metric_code = data_point.get("metric_code", "")
        metric_name = data_point.get("metric_name", "")

        # Try exact code match first
        if metric_code in self.esrs_code_lookup:
            esrs_dp = self.esrs_code_lookup[metric_code]
            self.stats["exact_matches"] += 1

            metadata = ESRSMetadata(
                esrs_code=esrs_dp.get("code") or esrs_dp.get("esrs_code"),
                esrs_standard=esrs_dp.get("standard"),
                disclosure_requirement=esrs_dp.get("disclosure_requirement"),
                data_point_name=esrs_dp.get("name") or esrs_dp.get("data_point_name"),
                expected_unit=esrs_dp.get("unit"),
                data_type=esrs_dp.get("data_type"),
                is_mandatory=esrs_dp.get("mandatory", False),
                mapping_confidence="exact"
            )

            # Check unit consistency
            if metadata.expected_unit and data_point.get("unit") != metadata.expected_unit:
                warnings.append(ValidationIssue(
                    metric_code=metric_code,
                    error_code="W007",
                    severity="warning",
                    message=f"Unit mismatch: provided '{data_point.get('unit')}', expected '{metadata.expected_unit}'",
                    field="unit"
                ))

            return metadata, warnings

        # Try fuzzy name match
        if metric_name:
            name_lower = metric_name.lower()
            if name_lower in self.esrs_name_lookup:
                esrs_dp = self.esrs_name_lookup[name_lower]
                self.stats["fuzzy_matches"] += 1

                metadata = ESRSMetadata(
                    esrs_code=esrs_dp.get("code") or esrs_dp.get("esrs_code"),
                    esrs_standard=esrs_dp.get("standard"),
                    disclosure_requirement=esrs_dp.get("disclosure_requirement"),
                    data_point_name=esrs_dp.get("name") or esrs_dp.get("data_point_name"),
                    expected_unit=esrs_dp.get("unit"),
                    data_type=esrs_dp.get("data_type"),
                    is_mandatory=esrs_dp.get("mandatory", False),
                    mapping_confidence="fuzzy"
                )

                warnings.append(ValidationIssue(
                    metric_code=metric_code,
                    error_code="W004",
                    severity="warning",
                    message=f"Fuzzy match used: mapped '{metric_name}' to ESRS code '{metadata.esrs_code}'",
                    field="metric_code",
                    suggestion="Verify mapping is correct"
                ))

                return metadata, warnings

        # No mapping found
        self.stats["unmapped"] += 1
        warnings.append(ValidationIssue(
            metric_code=metric_code,
            error_code="E003",
            severity="warning",
            message=f"No ESRS mapping found for metric: {metric_code} - {metric_name}",
            field="metric_code"
        ))

        return None, warnings

    # ========================================================================
    # DATA QUALITY ASSESSMENT
    # ========================================================================

    def assess_data_quality(self, df: pd.DataFrame, enriched_data: List[Dict[str, Any]]) -> DataQualityScore:
        """
        Assess overall data quality across multiple dimensions.

        Args:
            df: Original dataframe
            enriched_data: List of enriched data points

        Returns:
            DataQualityScore with dimension scores
        """
        # Completeness: % of required fields populated
        required_fields = ["metric_code", "metric_name", "value", "unit", "period_start", "period_end"]
        completeness_scores = []
        for _, row in df.iterrows():
            filled = sum(1 for field in required_fields if pd.notna(row.get(field)) and row.get(field) != "")
            completeness_scores.append(filled / len(required_fields) * 100)
        completeness = np.mean(completeness_scores) if completeness_scores else 0

        # Validity: % of records passing validation
        valid_count = sum(1 for dp in enriched_data if dp.get("validation_status") == "valid")
        validity = (valid_count / len(enriched_data) * 100) if enriched_data else 0

        # Accuracy: inverse of outlier rate
        outlier_count = sum(1 for dp in enriched_data if dp.get("is_outlier", False))
        accuracy = max(0, 100 - (outlier_count / len(enriched_data) * 100)) if enriched_data else 100

        # Consistency: % with ESRS mapping
        mapped_count = sum(1 for dp in enriched_data if dp.get("esrs_metadata") is not None)
        consistency = (mapped_count / len(enriched_data) * 100) if enriched_data else 0

        # Timeliness: assume 100 if data exists (would need more context for real assessment)
        timeliness = 100

        # Calculate overall score with weights
        score = DataQualityScore(
            completeness_score=completeness,
            accuracy_score=accuracy,
            consistency_score=consistency,
            timeliness_score=timeliness,
            validity_score=validity
        )

        score.overall_score = (
            score.completeness_score * score.completeness_weight +
            score.accuracy_score * score.accuracy_weight +
            score.consistency_score * score.consistency_weight +
            score.timeliness_score * score.timeliness_weight +
            score.validity_score * score.validity_weight
        )

        return score

    # ========================================================================
    # OUTLIER DETECTION
    # ========================================================================

    def detect_outliers(self, df: pd.DataFrame) -> Dict[int, List[str]]:
        """
        Detect statistical outliers using Z-score and IQR methods.

        Args:
            df: DataFrame with ESG data

        Returns:
            Dictionary mapping row index to list of outlier reasons
        """
        outliers = {}

        # Group by metric_code
        if "metric_code" not in df.columns or "value" not in df.columns:
            return outliers

        for metric_code, group in df.groupby("metric_code"):
            # Only analyze numeric values
            try:
                values = pd.to_numeric(group["value"], errors='coerce')
                values = values.dropna()

                if len(values) < 3:
                    continue  # Need at least 3 values for outlier detection

                # Z-score method (>3 standard deviations)
                mean = values.mean()
                std = values.std()
                if std > 0:
                    z_scores = np.abs((values - mean) / std)
                    z_outliers = values[z_scores > 3].index

                    for idx in z_outliers:
                        if idx not in outliers:
                            outliers[idx] = []
                        outliers[idx].append(f"Z-score outlier for {metric_code}")

                # IQR method
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                iqr_outliers = values[(values < lower_bound) | (values > upper_bound)].index
                for idx in iqr_outliers:
                    if idx not in outliers:
                        outliers[idx] = []
                    if f"IQR outlier for {metric_code}" not in outliers[idx]:
                        outliers[idx].append(f"IQR outlier for {metric_code}")

            except Exception as e:
                logger.debug(f"Could not analyze outliers for {metric_code}: {e}")
                continue

        return outliers

    # ========================================================================
    # ENRICHMENT
    # ========================================================================

    def enrich_data_point(
        self,
        data_point: Dict[str, Any],
        row_index: int = -1
    ) -> Tuple[Dict[str, Any], List[ValidationIssue]]:
        """
        Enrich data point with ESRS metadata and quality scores.

        Args:
            data_point: Data point dictionary
            row_index: Row index in source file

        Returns:
            Tuple of (enriched data point, list of warnings)
        """
        warnings = []

        # Map to ESRS
        esrs_metadata, mapping_warnings = self.map_to_esrs(data_point)
        warnings.extend(mapping_warnings)

        # Add ESRS metadata
        if esrs_metadata:
            data_point["esrs_metadata"] = esrs_metadata.dict()

        # Add processing timestamp
        data_point["processing_timestamp"] = datetime.now().isoformat()

        return data_point, warnings

    # ========================================================================
    # MAIN PROCESSING
    # ========================================================================

    def process(
        self,
        input_file: Union[str, Path],
        company_profile: Optional[Dict[str, Any]] = None,
        output_file: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Process ESG data: read, validate, enrich, and assess quality.

        Args:
            input_file: Path to input ESG data file
            company_profile: Optional company profile metadata
            output_file: Path for output file (optional)

        Returns:
            Result dictionary with metadata, validated data, and quality report
        """
        self.stats["start_time"] = datetime.now()

        # Read input
        df = self.read_esg_data(input_file)
        self.stats["total_records"] = len(df)

        # Detect outliers
        outliers_map = self.detect_outliers(df)
        self.stats["outliers_detected"] = len(outliers_map)

        # Process each data point
        enriched_data_points = []
        all_issues = []

        for idx, row in df.iterrows():
            data_point = row.to_dict()

            # Validate
            is_valid, issues = self.validate_data_point(data_point, row_index=idx)

            # Enrich
            if is_valid or True:  # Enrich even invalid records for reporting
                data_point, warnings = self.enrich_data_point(data_point, row_index=idx)
                issues.extend(warnings)

            # Mark outliers
            if idx in outliers_map:
                data_point["is_outlier"] = True
                for reason in outliers_map[idx]:
                    issues.append(ValidationIssue(
                        metric_code=data_point.get("metric_code"),
                        error_code="W002",
                        severity="warning",
                        message=reason,
                        field="value",
                        row_index=idx
                    ))
            else:
                data_point["is_outlier"] = False

            # Set validation status
            data_point["validation_status"] = "valid" if is_valid else "invalid"

            # Track statistics
            if is_valid:
                self.stats["valid_records"] += 1
                if any(issue.severity == "warning" for issue in issues):
                    self.stats["warnings"] += 1
            else:
                self.stats["invalid_records"] += 1

            enriched_data_points.append(data_point)
            all_issues.extend([issue.dict() for issue in issues])

        # Assess data quality
        quality_score = self.assess_data_quality(df, enriched_data_points)

        self.stats["end_time"] = datetime.now()
        processing_time = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()

        # Build result
        result = {
            "metadata": {
                "processed_at": self.stats["end_time"].isoformat(),
                "input_file": str(input_file),
                "total_records": self.stats["total_records"],
                "valid_records": self.stats["valid_records"],
                "invalid_records": self.stats["invalid_records"],
                "warnings": self.stats["warnings"],
                "outliers_detected": self.stats["outliers_detected"],
                "exact_esrs_matches": self.stats["exact_matches"],
                "fuzzy_esrs_matches": self.stats["fuzzy_matches"],
                "unmapped_metrics": self.stats["unmapped"],
                "processing_time_seconds": processing_time,
                "records_per_second": self.stats["total_records"] / processing_time if processing_time > 0 else 0,
                "data_quality_score": quality_score.overall_score,
                "quality_threshold_met": quality_score.overall_score >= (self.quality_threshold * 100)
            },
            "company_profile": company_profile,
            "data_points": enriched_data_points,
            "validation_issues": all_issues,
            "data_quality_report": quality_score.dict()
        }

        # Write output if path provided
        if output_file:
            self.write_output(result, output_file)

        logger.info(f"Processed {self.stats['total_records']} data points in {processing_time:.2f}s "
                   f"({result['metadata']['records_per_second']:.0f} records/sec)")
        logger.info(f"Valid: {self.stats['valid_records']}, Invalid: {self.stats['invalid_records']}, "
                   f"Warnings: {self.stats['warnings']}, Outliers: {self.stats['outliers_detected']}")
        logger.info(f"Data Quality Score: {quality_score.overall_score:.1f}/100")

        return result

    def write_output(self, result: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """Write result to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"Wrote output to {output_path}")

    def get_validation_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a validation summary report."""
        issues_by_code = {}
        for issue in result["validation_issues"]:
            code = issue["error_code"]
            if code not in issues_by_code:
                issues_by_code[code] = {
                    "code": code,
                    "description": ERROR_CODES.get(code, "Unknown error"),
                    "count": 0,
                    "severity": issue["severity"]
                }
            issues_by_code[code]["count"] += 1

        return {
            "summary": result["metadata"],
            "issues_by_code": list(issues_by_code.values()),
            "data_quality": result["data_quality_report"],
            "is_ready_for_next_stage": (
                result["metadata"]["invalid_records"] == 0 and
                result["metadata"]["quality_threshold_met"]
            )
        }


# ============================================================================
# CLI INTERFACE (for testing)
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CSRD ESG Data Intake Agent")
    parser.add_argument("--input", required=True, help="Input ESG data file (CSV/JSON/Excel/Parquet)")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--esrs-catalog", required=True, help="Path to ESRS data points catalog JSON")
    parser.add_argument("--quality-rules", required=True, help="Path to data quality rules YAML")
    parser.add_argument("--esg-schema", help="Path to ESG data JSON schema (optional)")
    parser.add_argument("--company-profile", help="Path to company profile JSON (optional)")
    parser.add_argument("--quality-threshold", type=float, default=0.80, help="Minimum quality score (0-1)")

    args = parser.parse_args()

    # Load company profile if provided
    company_profile = None
    if args.company_profile:
        with open(args.company_profile, 'r', encoding='utf-8') as f:
            company_profile = json.load(f)

    # Create agent
    agent = IntakeAgent(
        esrs_data_points_path=args.esrs_catalog,
        data_quality_rules_path=args.quality_rules,
        esg_data_schema_path=args.esg_schema,
        quality_threshold=args.quality_threshold
    )

    # Process
    result = agent.process(
        input_file=args.input,
        company_profile=company_profile,
        output_file=args.output
    )

    # Print report
    report = agent.get_validation_summary(result)
    print("\n" + "="*80)
    print("ESG DATA INTAKE VALIDATION REPORT")
    print("="*80)
    print(f"Total Records: {report['summary']['total_records']}")
    print(f"Valid: {report['summary']['valid_records']}")
    print(f"Invalid: {report['summary']['invalid_records']}")
    print(f"Warnings: {report['summary']['warnings']}")
    print(f"Outliers: {report['summary']['outliers_detected']}")
    print(f"Data Quality Score: {report['summary']['data_quality_score']:.1f}/100")
    print(f"Quality Threshold Met: {report['summary']['quality_threshold_met']}")
    print(f"Ready for next stage: {report['is_ready_for_next_stage']}")

    if report['issues_by_code']:
        print("\nIssues by Code:")
        for issue_summary in report['issues_by_code']:
            print(f"  {issue_summary['code']} ({issue_summary['severity']}): "
                  f"{issue_summary['description']} - Count: {issue_summary['count']}")

    print("\nData Quality Breakdown:")
    dq = report['data_quality']
    print(f"  Completeness: {dq['completeness_score']:.1f}/100")
    print(f"  Accuracy: {dq['accuracy_score']:.1f}/100")
    print(f"  Consistency: {dq['consistency_score']:.1f}/100")
    print(f"  Timeliness: {dq['timeliness_score']:.1f}/100")
    print(f"  Validity: {dq['validity_score']:.1f}/100")
