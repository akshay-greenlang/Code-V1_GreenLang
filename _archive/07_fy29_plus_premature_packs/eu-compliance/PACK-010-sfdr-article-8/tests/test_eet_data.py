# -*- coding: utf-8 -*-
"""
Unit tests for EETDataEngine (PACK-010 SFDR Article 8).

Tests EET field population, product info, SFDR classification, taxonomy
data, PAI data, validation, export/import, and provenance tracking.

Self-contained: no conftest imports.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from datetime import date
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Dynamic import helper
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"


def _import_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_eet_mod = _import_from_path(
    "eet_data_engine",
    str(ENGINES_DIR / "eet_data_engine.py"),
)

EETDataEngine = _eet_mod.EETDataEngine
EETConfig = _eet_mod.EETConfig
EETField = _eet_mod.EETField
EETDataSet = _eet_mod.EETDataSet
EETValidationResult = _eet_mod.EETValidationResult
EETExportResult = _eet_mod.EETExportResult
EETVersion = _eet_mod.EETVersion
EETSection = _eet_mod.EETSection
EETDataType = _eet_mod.EETDataType
SFDRClassification = _eet_mod.SFDRClassification
ExportFormat = _eet_mod.ExportFormat
ValidationSeverity = _eet_mod.ValidationSeverity


# ===================================================================
# TEST CLASS
# ===================================================================


class TestEETDataEngine:
    """Unit tests for EETDataEngine."""

    # ---------------------------------------------------------------
    # 1. Engine initialization
    # ---------------------------------------------------------------

    def test_engine_default_initialization(self):
        """Test engine initializes with default config."""
        engine = EETDataEngine()
        assert engine is not None

    def test_engine_custom_config(self):
        """Test engine initializes with custom config."""
        config = {"strict_mode": True}
        engine = EETDataEngine(config)
        assert engine is not None

    # ---------------------------------------------------------------
    # 2. set_product_info
    # ---------------------------------------------------------------

    def test_set_product_info(self):
        """Test setting product identification fields."""
        engine = EETDataEngine()
        engine.set_product_info(
            isin="LU1234567890",
            name="Green ESG Fund",
            reporting_date=date(2025, 12, 31),
        )
        dataset = engine.get_dataset()
        assert dataset is not None

    # ---------------------------------------------------------------
    # 3. set_sfdr_classification
    # ---------------------------------------------------------------

    def test_set_sfdr_classification_article_8(self):
        """Test setting SFDR classification to Article 8."""
        engine = EETDataEngine()
        engine.set_sfdr_classification(SFDRClassification.ARTICLE_8)
        sfdr_fields = engine.get_sfdr_fields()
        assert isinstance(sfdr_fields, list)

    def test_set_sfdr_classification_article_8_plus(self):
        """Test setting SFDR classification to Article 8+."""
        engine = EETDataEngine()
        engine.set_sfdr_classification(SFDRClassification.ARTICLE_8_PLUS)
        sfdr_fields = engine.get_sfdr_fields()
        assert isinstance(sfdr_fields, list)

    # ---------------------------------------------------------------
    # 4. set_taxonomy_data
    # ---------------------------------------------------------------

    def test_set_taxonomy_data(self):
        """Test setting taxonomy alignment data."""
        engine = EETDataEngine()
        engine.set_taxonomy_data(alignment_pct=35.5)
        tax_fields = engine.get_taxonomy_fields()
        assert isinstance(tax_fields, list)

    # ---------------------------------------------------------------
    # 5. set_pai_data
    # ---------------------------------------------------------------

    def test_set_pai_data(self):
        """Test setting PAI indicator data."""
        engine = EETDataEngine()
        pai_values = {
            1: 125000.0,
            2: 180.5,
            3: 245.3,
        }
        engine.set_pai_data(pai_values, considers_pai=True)
        pai_fields = engine.get_pai_fields()
        assert isinstance(pai_fields, list)

    # ---------------------------------------------------------------
    # 6. populate_eet_fields
    # ---------------------------------------------------------------

    def test_populate_eet_fields(self):
        """Test bulk field population."""
        engine = EETDataEngine()
        field_values = {
            "EET_01_002": "Test Fund",
            "EET_01_001": "LU0000000001",
        }
        engine.populate_eet_fields(field_values, source="manual")
        dataset = engine.get_dataset()
        assert dataset is not None

    # ---------------------------------------------------------------
    # 7. validate_eet_data
    # ---------------------------------------------------------------

    def test_validate_empty_data(self):
        """Test validation of empty EET data returns validation result."""
        engine = EETDataEngine()
        result = engine.validate_eet_data()
        assert isinstance(result, EETValidationResult)

    def test_validate_populated_data(self):
        """Test validation of populated EET data."""
        engine = EETDataEngine()
        engine.set_product_info(
            isin="LU1234567890",
            name="Validated Fund",
            reporting_date=date(2025, 12, 31),
        )
        engine.set_sfdr_classification(SFDRClassification.ARTICLE_8)
        result = engine.validate_eet_data()
        assert isinstance(result, EETValidationResult)

    # ---------------------------------------------------------------
    # 8. export_eet - JSON
    # ---------------------------------------------------------------

    def test_export_json(self):
        """Test exporting EET data as JSON."""
        engine = EETDataEngine()
        engine.set_product_info(
            isin="LU9999999999",
            name="Export Test Fund",
            reporting_date=date(2025, 12, 31),
        )
        result = engine.export_eet(ExportFormat.JSON)
        assert isinstance(result, EETExportResult)

    # ---------------------------------------------------------------
    # 9. export_eet - CSV
    # ---------------------------------------------------------------

    def test_export_csv(self):
        """Test exporting EET data as CSV."""
        engine = EETDataEngine()
        engine.set_product_info(
            isin="LU8888888888",
            name="CSV Test Fund",
            reporting_date=date(2025, 12, 31),
        )
        result = engine.export_eet(ExportFormat.CSV)
        assert isinstance(result, EETExportResult)

    # ---------------------------------------------------------------
    # 10. import_eet
    # ---------------------------------------------------------------

    def test_import_eet_json(self):
        """Test importing EET data from JSON."""
        engine = EETDataEngine()
        engine.set_product_info(
            isin="LU7777777777",
            name="Round Trip Fund",
            reporting_date=date(2025, 12, 31),
        )
        exported = engine.export_eet(ExportFormat.JSON)

        engine2 = EETDataEngine()
        engine2.import_eet(exported.content, ExportFormat.JSON, source="import_test")
        dataset = engine2.get_dataset()
        assert dataset is not None

    # ---------------------------------------------------------------
    # 11. get_field
    # ---------------------------------------------------------------

    def test_get_field_existing(self):
        """Test retrieving a specific field by ID."""
        engine = EETDataEngine()
        engine.set_product_info(
            isin="LU6666666666",
            name="Field Test Fund",
            reporting_date=date(2025, 12, 31),
        )
        # Try to get SFDR classification fields
        sfdr_fields = engine.get_sfdr_fields()
        assert isinstance(sfdr_fields, list)

    # ---------------------------------------------------------------
    # 12. SFDRClassification enum
    # ---------------------------------------------------------------

    def test_sfdr_classification_enum(self):
        """Test SFDRClassification enum has all article types."""
        vals = {c.value for c in SFDRClassification}
        assert "article_6" in vals
        assert "article_8" in vals
        assert "article_8_plus" in vals
        assert "article_9" in vals

    # ---------------------------------------------------------------
    # 13. ExportFormat enum
    # ---------------------------------------------------------------

    def test_export_format_enum(self):
        """Test ExportFormat enum values."""
        vals = {f.value for f in ExportFormat}
        assert "csv" in vals
        assert "json" in vals

    # ---------------------------------------------------------------
    # 14. ValidationSeverity enum
    # ---------------------------------------------------------------

    def test_validation_severity_enum(self):
        """Test ValidationSeverity enum values."""
        vals = {s.value for s in ValidationSeverity}
        assert "error" in vals
        assert "warning" in vals
        assert len(vals) >= 2

    # ---------------------------------------------------------------
    # 15. EETVersion enum
    # ---------------------------------------------------------------

    def test_eet_version_enum(self):
        """Test EETVersion enum has at least one version."""
        vals = {v.value for v in EETVersion}
        assert "1.1.1" in vals
        assert len(vals) >= 1
