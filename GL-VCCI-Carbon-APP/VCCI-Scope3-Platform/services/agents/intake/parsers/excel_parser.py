"""
Excel Parser with Sheet Detection

Excel file parsing with support for multiple sheets and formats.

Version: 1.0.0
Phase: 3 (Weeks 7-10)
Date: 2025-10-30
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import openpyxl
import pandas as pd

from ..exceptions import (
    FileParseError,
    ExcelSheetNotFoundError,
    UnsupportedFormatError,
)
from ..config import get_config

logger = logging.getLogger(__name__)


class ExcelParser:
    """
    Excel parser supporting xlsx and xls formats.

    Features:
    - Multi-sheet support
    - Automatic sheet detection
    - Header row configuration
    - Type preservation
    - Formula evaluation
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize Excel parser."""
        self.config = get_config().parser if config is None else config
        logger.info("Initialized ExcelParser")

    def get_sheet_names(self, file_path: Path) -> List[str]:
        """
        Get all sheet names from Excel file.

        Args:
            file_path: Path to Excel file

        Returns:
            List of sheet names
        """
        try:
            workbook = openpyxl.load_workbook(
                file_path,
                read_only=True,
                data_only=True  # Read values, not formulas
            )
            sheet_names = workbook.sheetnames
            workbook.close()

            logger.info(f"Found {len(sheet_names)} sheets: {sheet_names}")
            return sheet_names

        except Exception as e:
            raise FileParseError(
                f"Failed to read Excel sheets: {str(e)}",
                details={"file_path": str(file_path), "error": str(e)}
            ) from e

    def parse(
        self,
        file_path: Path,
        sheet_name: Optional[str] = None,
        header_row: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Parse Excel file into list of dictionaries.

        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name (None = first sheet)
            header_row: Header row index (default from config)

        Returns:
            List of dictionaries (one per row)

        Raises:
            FileParseError: If parsing fails
            ExcelSheetNotFoundError: If sheet not found
        """
        try:
            logger.info(f"Parsing Excel file: {file_path}")

            # Use pandas for robust Excel parsing
            header_row = header_row or self.config.excel_header_row

            # Determine sheet name
            if sheet_name is None:
                sheet_name = self.config.excel_sheet_name
                if sheet_name is None:
                    # Use first sheet
                    sheet_names = self.get_sheet_names(file_path)
                    sheet_name = sheet_names[0] if sheet_names else 0

            # Parse Excel
            try:
                df = pd.read_excel(
                    file_path,
                    sheet_name=sheet_name,
                    header=header_row,
                    engine='openpyxl'
                )
            except ValueError as e:
                if "Worksheet" in str(e):
                    available_sheets = self.get_sheet_names(file_path)
                    raise ExcelSheetNotFoundError(
                        f"Sheet '{sheet_name}' not found. Available: {available_sheets}",
                        details={
                            "file_path": str(file_path),
                            "requested_sheet": sheet_name,
                            "available_sheets": available_sheets
                        }
                    ) from e
                raise

            # Convert to list of dictionaries
            records = df.to_dict('records')

            # Clean up NaN values
            for record in records:
                for key, value in record.items():
                    if pd.isna(value):
                        record[key] = None

            logger.info(f"Successfully parsed {len(records)} records from Excel")
            return records

        except (FileParseError, ExcelSheetNotFoundError):
            raise

        except Exception as e:
            raise FileParseError(
                f"Failed to parse Excel file: {str(e)}",
                details={"file_path": str(file_path), "error": str(e)}
            ) from e

    def parse_all_sheets(
        self,
        file_path: Path,
        header_row: Optional[int] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Parse all sheets from Excel file.

        Args:
            file_path: Path to Excel file
            header_row: Header row index

        Returns:
            Dictionary mapping sheet names to record lists
        """
        try:
            sheet_names = self.get_sheet_names(file_path)
            all_data = {}

            for sheet_name in sheet_names:
                logger.info(f"Parsing sheet: {sheet_name}")
                records = self.parse(
                    file_path,
                    sheet_name=sheet_name,
                    header_row=header_row
                )
                all_data[sheet_name] = records

            logger.info(
                f"Successfully parsed {len(all_data)} sheets with "
                f"{sum(len(v) for v in all_data.values())} total records"
            )
            return all_data

        except Exception as e:
            raise FileParseError(
                f"Failed to parse all sheets: {str(e)}",
                details={"file_path": str(file_path), "error": str(e)}
            ) from e

    def parse_with_column_types(
        self,
        file_path: Path,
        column_types: Dict[str, str],
        sheet_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Parse Excel with explicit column type casting.

        Args:
            file_path: Path to Excel file
            column_types: Mapping of column names to types
                         Example: {"quantity": "int", "price": "float"}
            sheet_name: Sheet name

        Returns:
            List of dictionaries with typed values
        """
        try:
            # Determine sheet name
            if sheet_name is None:
                sheet_names = self.get_sheet_names(file_path)
                sheet_name = sheet_names[0] if sheet_names else 0

            # Parse with pandas dtype specification
            dtype_map = {}
            for col, dtype_str in column_types.items():
                if dtype_str in ('int', 'integer'):
                    dtype_map[col] = 'Int64'  # Nullable integer
                elif dtype_str in ('float', 'number'):
                    dtype_map[col] = 'float64'
                elif dtype_str in ('str', 'string'):
                    dtype_map[col] = 'str'
                elif dtype_str in ('bool', 'boolean'):
                    dtype_map[col] = 'bool'

            df = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
                header=self.config.excel_header_row,
                dtype=dtype_map,
                engine='openpyxl'
            )

            records = df.to_dict('records')

            # Clean up NaN values
            for record in records:
                for key, value in record.items():
                    if pd.isna(value):
                        record[key] = None

            logger.info(
                f"Parsed {len(records)} records with type casting: {column_types}"
            )
            return records

        except Exception as e:
            raise FileParseError(
                f"Failed to parse Excel with types: {str(e)}",
                details={
                    "file_path": str(file_path),
                    "column_types": column_types,
                    "error": str(e)
                }
            ) from e

    def validate_structure(
        self,
        file_path: Path,
        required_columns: List[str],
        sheet_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate Excel file structure.

        Args:
            file_path: Path to Excel file
            required_columns: List of required column names
            sheet_name: Sheet name to validate

        Returns:
            Validation result dictionary
        """
        try:
            # Parse headers only
            if sheet_name is None:
                sheet_names = self.get_sheet_names(file_path)
                sheet_name = sheet_names[0] if sheet_names else 0

            df = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
                header=self.config.excel_header_row,
                nrows=0,  # Read headers only
                engine='openpyxl'
            )

            found_columns = df.columns.tolist()
            missing_columns = [
                col for col in required_columns if col not in found_columns
            ]

            return {
                "valid": len(missing_columns) == 0,
                "missing_columns": missing_columns,
                "found_columns": found_columns,
                "sheet_name": sheet_name,
            }

        except Exception as e:
            logger.error(f"Structure validation failed: {e}")
            return {
                "valid": False,
                "error": str(e),
            }


__all__ = ["ExcelParser"]
