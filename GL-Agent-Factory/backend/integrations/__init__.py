"""
External Integrations Module.

This package provides integrations with external systems:
- ERP Connectors (SAP, Oracle)
- File Parsers (CSV, Excel, XML, JSON)

Example:
    >>> from integrations import SAPConnector, FileParserFactory
    >>>
    >>> # ERP Integration
    >>> connector = SAPConnector(config)
    >>> await connector.connect()
    >>> records = await connector.fetch_emissions_data(...)
    >>>
    >>> # File Parsing
    >>> parser = FileParserFactory.get_parser("data.xlsx")
    >>> result = await parser.parse("data.xlsx")
"""

from .erp import (
    BaseERPConnector,
    ConnectionConfig,
    ConnectionStatus,
    DataQuery,
    ERPRecord,
    ERPType,
    SAPConnector,
    SAPConfig,
    OracleConnector,
    OracleConfig,
    get_connector,
)

from .parsers import (
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
    # ERP
    "BaseERPConnector",
    "ConnectionConfig",
    "ConnectionStatus",
    "DataQuery",
    "ERPRecord",
    "ERPType",
    "SAPConnector",
    "SAPConfig",
    "OracleConnector",
    "OracleConfig",
    "get_connector",
    # Parsers
    "BaseFileParser",
    "CSVParser",
    "ExcelParser",
    "XMLParser",
    "JSONParser",
    "FileParserFactory",
    "FileFormat",
    "ParseResult",
    "ParsedRecord",
    "ParserConfig",
    "ColumnMapping",
]
