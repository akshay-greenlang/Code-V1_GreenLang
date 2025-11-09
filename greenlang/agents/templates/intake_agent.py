"""
Intake Agent Template
Multi-format Data Ingestion with Validation

Base agent template for data intake across sustainability applications.
Supports CSV, JSON, Excel, XML, PDF parsing with validation and quality checks.

Version: 1.0.0
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import json

logger = logging.getLogger(__name__)


class DataFormat(str, Enum):
    """Supported data formats."""
    CSV = "csv"
    JSON = "json"
    EXCEL = "excel"
    XML = "xml"
    PDF = "pdf"
    PARQUET = "parquet"
    AVRO = "avro"
    ORC = "orc"
    FEATHER = "feather"


class ValidationSeverity(str, Enum):
    """Validation issue severity."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Validation issue."""
    severity: ValidationSeverity
    message: str
    row_number: Optional[int] = None
    column: Optional[str] = None
    value: Optional[Any] = None


@dataclass
class IntakeResult:
    """Result of data intake operation."""
    success: bool
    data: Optional[pd.DataFrame] = None
    validation_issues: List[ValidationIssue] = None
    rows_read: int = 0
    rows_valid: int = 0
    rows_invalid: int = 0
    metadata: Dict[str, Any] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.validation_issues is None:
            self.validation_issues = []
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class IntakeAgent:
    """
    Base Intake Agent Template.

    Provides common data ingestion patterns for sustainability applications:
    - Multi-format parsing (CSV, JSON, Excel, XML, PDF, Parquet, Avro, ORC, Feather)
    - Schema validation
    - Data quality checks
    - Outlier detection
    - Entity resolution integration
    - Streaming support for large files
    """

    def __init__(
        self,
        schema: Optional[Dict[str, Any]] = None,
        entity_resolver: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Intake Agent.

        Args:
            schema: Data schema for validation
            entity_resolver: EntityResolver instance for entity resolution
            config: Agent configuration
        """
        self.schema = schema or {}
        self.entity_resolver = entity_resolver
        self.config = config or {}

        self._stats = {
            "total_intakes": 0,
            "successful_intakes": 0,
            "failed_intakes": 0,
            "total_rows_processed": 0,
        }

        logger.info("Initialized IntakeAgent")

    async def ingest(
        self,
        file_path: Optional[str] = None,
        data: Optional[Union[str, bytes, pd.DataFrame]] = None,
        format: DataFormat = DataFormat.CSV,
        validate: bool = True,
        resolve_entities: bool = False,
    ) -> IntakeResult:
        """
        Ingest data from file or raw data.

        Args:
            file_path: Path to data file
            data: Raw data (string, bytes, or DataFrame)
            format: Data format
            validate: Whether to validate data
            resolve_entities: Whether to resolve entities

        Returns:
            IntakeResult with ingested data and validation results
        """
        self._stats["total_intakes"] += 1

        try:
            # Step 1: Parse data
            df = await self._parse_data(file_path, data, format)

            if df is None or df.empty:
                return IntakeResult(
                    success=False,
                    validation_issues=[
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            message="No data found or parsing failed"
                        )
                    ]
                )

            rows_read = len(df)

            # Step 2: Validate schema
            validation_issues = []
            if validate:
                validation_issues = self._validate_schema(df)

            # Step 3: Data quality checks
            quality_issues = self._check_data_quality(df)
            validation_issues.extend(quality_issues)

            # Step 4: Outlier detection
            outlier_issues = self._detect_outliers(df)
            validation_issues.extend(outlier_issues)

            # Step 5: Entity resolution
            if resolve_entities and self.entity_resolver:
                df = await self._resolve_entities(df)

            # Count errors
            errors = [v for v in validation_issues if v.severity == ValidationSeverity.ERROR]
            rows_valid = rows_read - len([v for v in errors if v.row_number is not None])
            rows_invalid = len([v for v in errors if v.row_number is not None])

            # Determine success
            success = len(errors) == 0

            if success:
                self._stats["successful_intakes"] += 1
            else:
                self._stats["failed_intakes"] += 1

            self._stats["total_rows_processed"] += rows_read

            return IntakeResult(
                success=success,
                data=df,
                validation_issues=validation_issues,
                rows_read=rows_read,
                rows_valid=rows_valid,
                rows_invalid=rows_invalid,
                metadata={
                    "format": format.value,
                    "has_entity_resolution": resolve_entities,
                }
            )

        except Exception as e:
            logger.error(f"Intake failed: {e}", exc_info=True)
            self._stats["failed_intakes"] += 1

            return IntakeResult(
                success=False,
                validation_issues=[
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Intake failed: {str(e)}"
                    )
                ]
            )

    async def _parse_data(
        self,
        file_path: Optional[str],
        data: Optional[Union[str, bytes, pd.DataFrame]],
        format: DataFormat
    ) -> Optional[pd.DataFrame]:
        """Parse data from file or raw data."""
        try:
            # If DataFrame provided directly
            if isinstance(data, pd.DataFrame):
                return data

            # Parse from file
            if file_path:
                if format == DataFormat.CSV:
                    return pd.read_csv(file_path)
                elif format == DataFormat.JSON:
                    return pd.read_json(file_path)
                elif format == DataFormat.EXCEL:
                    return pd.read_excel(file_path)
                elif format == DataFormat.PARQUET:
                    return pd.read_parquet(file_path)
                elif format == DataFormat.XML:
                    return pd.read_xml(file_path)
                elif format == DataFormat.AVRO:
                    return self._read_avro(file_path)
                elif format == DataFormat.ORC:
                    return pd.read_orc(file_path)
                elif format == DataFormat.FEATHER:
                    return pd.read_feather(file_path)
                else:
                    logger.error(f"Unsupported format: {format}")
                    return None

            # Parse from raw data
            if data:
                if format == DataFormat.CSV:
                    return pd.read_csv(pd.io.common.StringIO(data))
                elif format == DataFormat.JSON:
                    return pd.read_json(pd.io.common.StringIO(data))
                else:
                    logger.error(f"Unsupported format for raw data: {format}")
                    return None

            return None

        except Exception as e:
            logger.error(f"Parsing failed: {e}", exc_info=True)
            return None

    def _validate_schema(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate data against schema."""
        issues = []

        if not self.schema:
            return issues

        # Check required columns
        required_columns = self.schema.get("required", [])
        missing_columns = set(required_columns) - set(df.columns)

        if missing_columns:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Missing required columns: {', '.join(missing_columns)}"
                )
            )

        # Check column types
        column_types = self.schema.get("types", {})
        for column, expected_type in column_types.items():
            if column in df.columns:
                if not self._check_column_type(df[column], expected_type):
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"Column '{column}' has incorrect type (expected: {expected_type})",
                            column=column
                        )
                    )

        return issues

    def _check_column_type(self, series: pd.Series, expected_type: str) -> bool:
        """Check if column matches expected type."""
        type_map = {
            "string": "object",
            "number": ["int64", "float64"],
            "integer": "int64",
            "float": "float64",
            "boolean": "bool",
            "datetime": "datetime64[ns]",
        }

        expected_dtypes = type_map.get(expected_type)
        if expected_dtypes is None:
            return True

        if isinstance(expected_dtypes, list):
            return str(series.dtype) in expected_dtypes
        else:
            return str(series.dtype) == expected_dtypes

    def _check_data_quality(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check data quality."""
        issues = []

        # Check for missing values
        for column in df.columns:
            null_count = df[column].isnull().sum()
            if null_count > 0:
                null_percent = (null_count / len(df)) * 100
                severity = (
                    ValidationSeverity.ERROR if null_percent > 50
                    else ValidationSeverity.WARNING if null_percent > 20
                    else ValidationSeverity.INFO
                )
                issues.append(
                    ValidationIssue(
                        severity=severity,
                        message=f"Column '{column}' has {null_count} missing values ({null_percent:.1f}%)",
                        column=column
                    )
                )

        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Found {duplicate_count} duplicate rows"
                )
            )

        return issues

    def _detect_outliers(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Detect outliers in numeric columns."""
        issues = []

        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

        for column in numeric_columns:
            # Use IQR method
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

            if len(outliers) > 0:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Column '{column}' has {len(outliers)} potential outliers",
                        column=column
                    )
                )

        return issues

    async def _resolve_entities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resolve entities using EntityResolver."""
        if not self.entity_resolver:
            return df

        # This would integrate with the Entity MDM service
        # For now, return df unchanged
        logger.info("Entity resolution would be performed here")
        return df

    def _read_avro(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Read Avro file into DataFrame.

        Args:
            file_path: Path to Avro file

        Returns:
            DataFrame or None if reading fails
        """
        try:
            # Try using pandavro first
            try:
                import pandavro
                return pandavro.read_avro(file_path)
            except ImportError:
                logger.warning("pandavro not available, trying fastavro")

            # Fallback to fastavro
            try:
                from fastavro import reader
                records = []
                with open(file_path, 'rb') as f:
                    avro_reader = reader(f)
                    for record in avro_reader:
                        records.append(record)
                return pd.DataFrame(records)
            except ImportError:
                logger.error("Neither pandavro nor fastavro available for Avro support")
                return None

        except Exception as e:
            logger.error(f"Failed to read Avro file: {e}", exc_info=True)
            return None

    async def ingest_streaming(
        self,
        file_path: str,
        format: DataFormat = DataFormat.CSV,
        chunk_size: int = 10000,
        validate: bool = True,
    ) -> List[IntakeResult]:
        """
        Ingest large file in streaming chunks.

        Args:
            file_path: Path to data file
            format: Data format
            chunk_size: Number of rows per chunk
            validate: Whether to validate each chunk

        Returns:
            List of IntakeResults, one per chunk
        """
        results = []

        try:
            if format == DataFormat.CSV:
                # Read CSV in chunks
                for chunk_num, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
                    logger.info(f"Processing chunk {chunk_num + 1}")

                    # Process chunk as regular intake
                    result = await self.ingest(
                        data=chunk,
                        format=format,
                        validate=validate,
                        resolve_entities=False,
                    )

                    result.metadata["chunk_number"] = chunk_num + 1
                    results.append(result)

            elif format == DataFormat.JSON:
                # Read JSON in chunks
                for chunk_num, chunk in enumerate(pd.read_json(file_path, lines=True, chunksize=chunk_size)):
                    logger.info(f"Processing chunk {chunk_num + 1}")

                    result = await self.ingest(
                        data=chunk,
                        format=format,
                        validate=validate,
                        resolve_entities=False,
                    )

                    result.metadata["chunk_number"] = chunk_num + 1
                    results.append(result)

            elif format == DataFormat.PARQUET:
                # Read Parquet file (already efficient with row groups)
                df = pd.read_parquet(file_path)

                # Split into chunks manually
                total_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size > 0 else 0)

                for chunk_num in range(total_chunks):
                    start_idx = chunk_num * chunk_size
                    end_idx = min((chunk_num + 1) * chunk_size, len(df))
                    chunk = df.iloc[start_idx:end_idx]

                    logger.info(f"Processing chunk {chunk_num + 1}/{total_chunks}")

                    result = await self.ingest(
                        data=chunk,
                        format=format,
                        validate=validate,
                        resolve_entities=False,
                    )

                    result.metadata["chunk_number"] = chunk_num + 1
                    result.metadata["total_chunks"] = total_chunks
                    results.append(result)

            else:
                logger.error(f"Streaming not supported for format: {format}")
                return [IntakeResult(
                    success=False,
                    validation_issues=[
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            message=f"Streaming not supported for {format}"
                        )
                    ]
                )]

        except Exception as e:
            logger.error(f"Streaming intake failed: {e}", exc_info=True)
            return [IntakeResult(
                success=False,
                validation_issues=[
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Streaming intake failed: {str(e)}"
                    )
                ]
            )]

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return self._stats.copy()
