# -*- coding: utf-8 -*-
"""
ValueChain Intake Agent Parsers

Multi-format data parsers for ingestion pipeline.

Supported Formats:
- CSV (with encoding detection)
- JSON
- Excel (xlsx, xls)
- XML (with XPath)
- PDF (with OCR)

Version: 1.0.0
Phase: 3 (Weeks 7-10)
Date: 2025-10-30
"""

from .csv_parser import CSVParser
from .json_parser import JSONParser
from .excel_parser import ExcelParser
from .xml_parser import XMLParser
from .pdf_ocr_parser import PDFOCRParser

__all__ = [
    "CSVParser",
    "JSONParser",
    "ExcelParser",
    "XMLParser",
    "PDFOCRParser",
]
