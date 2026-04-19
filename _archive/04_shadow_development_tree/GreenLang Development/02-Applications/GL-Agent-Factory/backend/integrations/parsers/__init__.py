"""
File Parser Module.

This package provides parsers for various file formats
used in emissions and sustainability data exchange.

Supported Formats:
- CSV (comma, semicolon, tab delimited)
- Excel (.xlsx, .xls)
- XML
- JSON

Example:
    >>> from integrations.parsers import FileParserFactory
    >>>
    >>> parser = FileParserFactory.get_parser("emissions.csv")
    >>> result = await parser.parse("emissions.csv")
    >>> print(f"Parsed {result.successful_rows} rows")
"""

from .file_parser import (
    BaseFileParser,
    CSVParser,
    ExcelParser,
    XMLParser,
    JSONParser,
    FileParserFactory,
    FileFormat,
    ParseResult,
    ParsedRecord,
    ParserConfig,
    ColumnMapping,
)

__all__ = [
    # Base classes
    "BaseFileParser",
    "FileParserFactory",
    # Parsers
    "CSVParser",
    "ExcelParser",
    "XMLParser",
    "JSONParser",
    # Models
    "FileFormat",
    "ParseResult",
    "ParsedRecord",
    "ParserConfig",
    "ColumnMapping",
]
