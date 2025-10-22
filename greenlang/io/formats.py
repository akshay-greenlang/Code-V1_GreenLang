"""
GreenLang Format Registry
Format detection and conversion utilities.
"""

from typing import Dict, Optional, Callable, Any
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Format(str, Enum):
    """Supported data formats."""
    JSON = "json"
    CSV = "csv"
    TSV = "tsv"
    YAML = "yaml"
    EXCEL = "excel"
    PARQUET = "parquet"
    XML = "xml"
    TEXT = "text"


class FormatRegistry:
    """
    Registry for format handlers and detection.

    Provides format detection from file extensions and MIME types.

    Example:
        registry = FormatRegistry()

        # Detect format
        format = registry.detect_format("data.csv")  # Returns Format.CSV

        # Check if format is supported
        if registry.is_supported(".parquet"):
            print("Parquet is supported")
    """

    # Extension to format mapping
    EXTENSION_MAP = {
        ".json": Format.JSON,
        ".csv": Format.CSV,
        ".tsv": Format.TSV,
        ".yaml": Format.YAML,
        ".yml": Format.YAML,
        ".xlsx": Format.EXCEL,
        ".xls": Format.EXCEL,
        ".parquet": Format.PARQUET,
        ".xml": Format.XML,
        ".txt": Format.TEXT,
    }

    # MIME type mapping
    MIME_MAP = {
        "application/json": Format.JSON,
        "text/csv": Format.CSV,
        "text/tab-separated-values": Format.TSV,
        "application/x-yaml": Format.YAML,
        "text/yaml": Format.YAML,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": Format.EXCEL,
        "application/vnd.ms-excel": Format.EXCEL,
        "application/x-parquet": Format.PARQUET,
        "application/xml": Format.XML,
        "text/xml": Format.XML,
        "text/plain": Format.TEXT,
    }

    def __init__(self):
        """Initialize format registry."""
        self._converters: Dict[tuple, Callable] = {}

    def detect_format(self, file_path: str) -> Optional[Format]:
        """
        Detect format from file path.

        Args:
            file_path: Path to file

        Returns:
            Detected format or None
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        return self.EXTENSION_MAP.get(ext)

    def detect_format_from_mime(self, mime_type: str) -> Optional[Format]:
        """
        Detect format from MIME type.

        Args:
            mime_type: MIME type string

        Returns:
            Detected format or None
        """
        return self.MIME_MAP.get(mime_type.lower())

    def is_supported(self, extension: str) -> bool:
        """
        Check if a file extension is supported.

        Args:
            extension: File extension (with or without dot)

        Returns:
            True if supported
        """
        if not extension.startswith('.'):
            extension = '.' + extension

        return extension.lower() in self.EXTENSION_MAP

    def get_extension(self, format: Format) -> str:
        """
        Get default file extension for a format.

        Args:
            format: Format enum

        Returns:
            File extension with dot
        """
        for ext, fmt in self.EXTENSION_MAP.items():
            if fmt == format:
                return ext

        return ".dat"

    def register_converter(
        self,
        from_format: Format,
        to_format: Format,
        converter_func: Callable[[Any], Any]
    ):
        """
        Register a format converter function.

        Args:
            from_format: Source format
            to_format: Target format
            converter_func: Function to convert data
        """
        key = (from_format, to_format)
        self._converters[key] = converter_func
        logger.debug(f"Registered converter: {from_format} -> {to_format}")

    def convert(self, data: Any, from_format: Format, to_format: Format) -> Any:
        """
        Convert data between formats.

        Args:
            data: Data to convert
            from_format: Source format
            to_format: Target format

        Returns:
            Converted data

        Raises:
            ValueError: If conversion not supported
        """
        if from_format == to_format:
            return data

        key = (from_format, to_format)
        if key not in self._converters:
            raise ValueError(f"No converter registered for {from_format} -> {to_format}")

        converter = self._converters[key]
        return converter(data)

    def get_supported_formats(self) -> list:
        """Get list of supported formats."""
        return list(set(self.EXTENSION_MAP.values()))

    def get_supported_extensions(self) -> list:
        """Get list of supported file extensions."""
        return list(self.EXTENSION_MAP.keys())
