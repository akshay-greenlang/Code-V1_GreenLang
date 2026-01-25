# -*- coding: utf-8 -*-
"""
Exporters Tests
GL-VCCI Scope 3 Platform
"""

import pytest
import json
from pathlib import Path
import pandas as pd
from services.agents.reporting.exporters import JSONExporter, ExcelExporter, PDFExporter


# JSON Exporter Tests
def test_json_exporter():
    """Test JSON exporter."""
    exporter = JSONExporter()
    content = {"test": "data", "emissions": 1000.0}
    output_path = "test_export.json"

    result_path = exporter.export(content, output_path)

    assert Path(result_path).exists()

    with open(result_path, 'r') as f:
        loaded = json.load(f)
        assert loaded["test"] == "data"

    # Cleanup
    Path(result_path).unlink()


# Excel Exporter Tests
def test_excel_exporter():
    """Test Excel exporter."""
    exporter = ExcelExporter()
    content = {"total_emissions": 1000.0}
    tables = {
        "Sheet1": pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    }
    output_path = "test_export.xlsx"

    result_path = exporter.export(content, tables, output_path)

    assert Path(result_path).exists()

    # Cleanup
    Path(result_path).unlink()


# PDF Exporter Tests
def test_pdf_exporter():
    """Test PDF exporter (fallback to HTML)."""
    exporter = PDFExporter()
    content = {}
    html_content = "<html><body><h1>Test Report</h1></body></html>"
    output_path = "test_export.pdf"

    result_path = exporter.export(content, html_content, output_path)

    # Will either be PDF or HTML depending on WeasyPrint availability
    assert Path(result_path).exists()

    # Cleanup
    Path(result_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
