# -*- coding: utf-8 -*-
"""
CSV Parser with Encoding Detection

Robust CSV parsing with automatic encoding detection and error handling.

Features:
- Automatic encoding detection (chardet)
- Configurable delimiter detection
- Header row handling
- Type inference
- Comprehensive error handling

Version: 1.0.0
Phase: 3 (Weeks 7-10)
Date: 2025-10-30
"""

import csv
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import chardet

from greenlang.security.validators import PathTraversalValidator

from ..exceptions import (
    FileParseError,
    EncodingDetectionError,
    UnsupportedFormatError,
)
from ..config import get_config

logger = logging.getLogger(__name__)


# ============================================================================
# CSV PARSER
# ============================================================================

class CSVParser:
    """
    CSV parser with automatic encoding detection.

    Features:
    - Detects file encoding using chardet
    - Supports configurable delimiter
    - Handles headers automatically
    - Type inference for numeric values
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize CSV parser.

        Args:
            config: Optional configuration override
        """
        self.config = get_config().parser if config is None else config
        logger.info("Initialized CSVParser")

    def detect_encoding(self, file_path: Path) -> str:
        """
        Detect file encoding using chardet.

        Args:
            file_path: Path to CSV file

        Returns:
            Detected encoding name

        Raises:
            EncodingDetectionError: If encoding cannot be detected
        """
        try:
            # Validate path for security
            validated_path = PathTraversalValidator.validate_path(file_path, must_exist=True)

            # Read first 100KB for detection
            with open(validated_path, 'rb') as f:
                raw_data = f.read(100000)

            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']

            logger.info(
                f"Detected encoding: {encoding} (confidence: {confidence:.2%})"
            )

            if encoding is None or confidence < 0.5:
                logger.warning(
                    f"Low confidence encoding detection ({confidence:.2%}), "
                    "trying fallbacks"
                )
                # Try fallback encodings
                for fallback in self.config.csv_encoding_fallbacks:
                    try:
                        with open(file_path, 'r', encoding=fallback) as f:
                            f.read(1000)  # Try reading first 1000 chars
                        logger.info(f"Using fallback encoding: {fallback}")
                        return fallback
                    except UnicodeDecodeError:
                        continue

                raise EncodingDetectionError(
                    f"Could not detect encoding with sufficient confidence "
                    f"(confidence: {confidence:.2%})"
                )

            return encoding

        except Exception as e:
            raise EncodingDetectionError(
                f"Encoding detection failed: {str(e)}"
            ) from e

    def detect_delimiter(self, file_path: Path, encoding: str) -> str:
        """
        Detect CSV delimiter by analyzing first few lines.

        Args:
            file_path: Path to CSV file
            encoding: File encoding

        Returns:
            Detected delimiter character
        """
        try:
            # Validate path for security
            validated_path = PathTraversalValidator.validate_path(file_path, must_exist=True)

            with open(validated_path, 'r', encoding=encoding) as f:
                # Read first 3 lines
                sample = ''.join([f.readline() for _ in range(3)])

            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter

            logger.info(f"Detected delimiter: '{delimiter}'")
            return delimiter

        except Exception as e:
            logger.warning(
                f"Could not detect delimiter: {e}, using default: '{self.config.csv_delimiter}'"
            )
            return self.config.csv_delimiter

    def infer_type(self, value: str) -> Any:
        """
        Infer type from string value.

        Args:
            value: String value

        Returns:
            Typed value (int, float, bool, or str)
        """
        value = value.strip()

        # Empty string
        if not value:
            return None

        # Boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False

        # Numeric
        try:
            # Try integer first
            if '.' not in value and 'e' not in value.lower():
                return int(value.replace(',', ''))
            # Try float
            return float(value.replace(',', ''))
        except ValueError:
            pass

        # String
        return value

    def parse(
        self,
        file_path: Path,
        encoding: Optional[str] = None,
        delimiter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Parse CSV file into list of dictionaries with path traversal protection.

        Args:
            file_path: Path to CSV file
            encoding: File encoding (auto-detected if None)
            delimiter: CSV delimiter (auto-detected if None)

        Returns:
            List of dictionaries (one per row)

        Raises:
            FileParseError: If parsing fails
        """
        try:
            # Validate path for security (prevent path traversal)
            validated_path = PathTraversalValidator.validate_path(file_path, must_exist=True)

            logger.info(f"Parsing CSV file: {validated_path}")

            # Detect encoding if not provided
            if encoding is None:
                encoding = self.detect_encoding(validated_path)

            # Detect delimiter if not provided
            if delimiter is None:
                delimiter = self.detect_delimiter(validated_path, encoding)

            # Parse CSV
            records = []
            with open(validated_path, 'r', encoding=encoding, newline='') as f:
                # Skip initial rows if configured
                for _ in range(self.config.csv_skip_rows):
                    next(f, None)

                reader = csv.DictReader(f, delimiter=delimiter)

                for row_num, row in enumerate(reader, start=1):
                    # Type inference
                    typed_row = {
                        key: self.infer_type(value)
                        for key, value in row.items()
                        if key is not None  # Skip None keys from malformed CSV
                    }

                    # Add row metadata
                    typed_row['_row_number'] = row_num + self.config.csv_skip_rows

                    records.append(typed_row)

            logger.info(f"Successfully parsed {len(records)} records from CSV")
            return records

        except EncodingDetectionError:
            raise

        except Exception as e:
            raise FileParseError(
                f"Failed to parse CSV file: {str(e)}",
                details={"file_path": str(file_path), "error": str(e)}
            ) from e

    def parse_with_schema(
        self,
        file_path: Path,
        column_mapping: Dict[str, str],
        encoding: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Parse CSV with column name mapping.

        Args:
            file_path: Path to CSV file
            column_mapping: Mapping from CSV columns to target schema
                           Example: {"Supplier Name": "supplier_name"}
            encoding: File encoding (auto-detected if None)

        Returns:
            List of dictionaries with mapped column names
        """
        records = self.parse(file_path, encoding=encoding)

        # Apply column mapping
        mapped_records = []
        for record in records:
            mapped_record = {}
            for csv_col, target_col in column_mapping.items():
                if csv_col in record:
                    mapped_record[target_col] = record[csv_col]

            # Preserve row metadata
            if '_row_number' in record:
                mapped_record['_row_number'] = record['_row_number']

            mapped_records.append(mapped_record)

        logger.info(
            f"Applied column mapping: {len(column_mapping)} columns mapped"
        )
        return mapped_records

    def validate_headers(
        self,
        file_path: Path,
        required_columns: List[str],
        encoding: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate that CSV has required columns.

        Args:
            file_path: Path to CSV file
            required_columns: List of required column names
            encoding: File encoding (auto-detected if None)

        Returns:
            Validation result dictionary with status and details
        """
        try:
            if encoding is None:
                encoding = self.detect_encoding(file_path)

            delimiter = self.detect_delimiter(file_path, encoding)

            with open(file_path, 'r', encoding=encoding, newline='') as f:
                for _ in range(self.config.csv_skip_rows):
                    next(f, None)

                reader = csv.DictReader(f, delimiter=delimiter)
                headers = reader.fieldnames or []

            missing_columns = [
                col for col in required_columns if col not in headers
            ]

            if missing_columns:
                return {
                    "valid": False,
                    "missing_columns": missing_columns,
                    "found_columns": headers,
                }

            return {
                "valid": True,
                "missing_columns": [],
                "found_columns": headers,
            }

        except Exception as e:
            logger.error(f"Header validation failed: {e}")
            return {
                "valid": False,
                "error": str(e),
            }


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ["CSVParser"]
