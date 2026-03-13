# -*- coding: utf-8 -*-
"""
Unit tests for DataNormalizationEngine - AGENT-EUDR-027

Tests data normalization operations including coordinate normalization
(decimal and DMS formats), country code normalization (name to ISO,
alpha-3 to alpha-2), date format normalization, unit normalization,
certificate ID normalization, address normalization, product code
normalization, batch normalization, normalize_record dispatch,
confidence scoring, and statistics.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-027 (Engine 6: Data Normalization)
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.agents.eudr.information_gathering.data_normalization_engine import (
    DataNormalizationEngine,
)
from greenlang.agents.eudr.information_gathering.models import (
    NormalizationRecord,
    NormalizationType,
)


@pytest.fixture
def engine(config):
    return DataNormalizationEngine(config)


# ---------------------------------------------------------------------------
# Coordinate Normalization
# ---------------------------------------------------------------------------


class TestNormalizeCoordinate:
    """Test coordinate normalization."""

    def test_normalize_coordinate_decimal_pair(self, engine):
        result = engine.normalize_coordinate("1.2345, 103.456")
        assert "1.234500" in result
        assert "103.456000" in result

    def test_normalize_coordinate_signed_decimal(self, engine):
        result = engine.normalize_coordinate("-1.2345 103.456")
        assert result == "-1.234500,103.456000"

    def test_normalize_coordinate_dms(self, engine):
        result = engine.normalize_coordinate(
            "1\u00b014'04.2\"N, 103\u00b027'21.6\"E"
        )
        # Should convert DMS to decimal degrees
        assert "," in result
        parts = result.split(",")
        assert len(parts) == 2
        lat = float(parts[0])
        lon = float(parts[1])
        assert 1.0 < lat < 2.0
        assert 103.0 < lon < 104.0

    def test_normalize_coordinate_fallback(self, engine):
        # Unparseable input returns cleaned value
        result = engine.normalize_coordinate("some random text")
        assert result == "some random text"


# ---------------------------------------------------------------------------
# Country Code Normalization
# ---------------------------------------------------------------------------


class TestNormalizeCountryCode:
    """Test country code normalization."""

    def test_normalize_country_code_name(self, engine):
        result = engine.normalize_country_code("Brazil")
        assert result == "BR"

    def test_normalize_country_code_alpha3(self, engine):
        result = engine.normalize_country_code("BRA")
        assert result == "BR"

    def test_normalize_country_code_alpha2_passthrough(self, engine):
        result = engine.normalize_country_code("DE")
        assert result == "DE"

    def test_normalize_country_code_case_insensitive(self, engine):
        result = engine.normalize_country_code("germany")
        assert result == "DE"


# ---------------------------------------------------------------------------
# Date Normalization
# ---------------------------------------------------------------------------


class TestNormalizeDate:
    """Test date format normalization."""

    def test_normalize_date_iso(self, engine):
        result = engine.normalize_date("2026-01-15")
        assert result == "2026-01-15"

    def test_normalize_date_european_slash(self, engine):
        result = engine.normalize_date("15/01/2026")
        assert result == "2026-01-15"

    def test_normalize_date_us_dash(self, engine):
        result = engine.normalize_date("01-15-2026")
        assert result == "2026-01-15"

    def test_normalize_date_long_month(self, engine):
        result = engine.normalize_date("January 15, 2026")
        assert result == "2026-01-15"

    def test_normalize_date_fallback(self, engine):
        result = engine.normalize_date("not a date")
        assert result == "not a date"


# ---------------------------------------------------------------------------
# Unit Normalization
# ---------------------------------------------------------------------------


class TestNormalizeUnit:
    """Test unit normalization."""

    def test_normalize_unit_kg(self, engine):
        result = engine.normalize_unit("1000 kilograms")
        assert result == "1000 kg"

    def test_normalize_unit_tonnes(self, engine):
        result = engine.normalize_unit("50 metric tons")
        assert result == "50 t"

    def test_normalize_unit_no_numeric(self, engine):
        result = engine.normalize_unit("hectares")
        assert result == "ha"

    def test_normalize_unit_passthrough(self, engine):
        result = engine.normalize_unit("500 kg")
        assert result == "500 kg"


# ---------------------------------------------------------------------------
# Certificate ID Normalization
# ---------------------------------------------------------------------------


class TestNormalizeCertificateId:
    """Test certificate ID normalization."""

    def test_normalize_certificate_id_fsc(self, engine):
        result = engine.normalize_certificate_id("fsc-c012345")
        assert result == "FSC-C012345"

    def test_normalize_certificate_id_fsc_space(self, engine):
        result = engine.normalize_certificate_id("FSC C012345")
        assert result == "FSC-C012345"

    def test_normalize_certificate_id_pefc(self, engine):
        result = engine.normalize_certificate_id("PEFC 01-23-45")
        assert result == "PEFC/01-23-45"

    def test_normalize_certificate_id_eu_bio(self, engine):
        result = engine.normalize_certificate_id("EU BIO 123")
        assert result == "EU-BIO-123"


# ---------------------------------------------------------------------------
# Address Normalization
# ---------------------------------------------------------------------------


class TestNormalizeAddress:
    """Test address normalization."""

    def test_normalize_address_whitespace(self, engine):
        result = engine.normalize_address("  123  Main   St.,  Berlin  ")
        assert "  " not in result
        assert result.startswith("123")

    def test_normalize_address_abbreviation(self, engine):
        result = engine.normalize_address("123 St. Berlin")
        assert "Street" in result


# ---------------------------------------------------------------------------
# Product Code Normalization
# ---------------------------------------------------------------------------


class TestNormalizeProductCode:
    """Test product code normalization."""

    def test_normalize_product_code_with_dots(self, engine):
        result = engine.normalize_product_code("0901.11")
        assert result == "090111"

    def test_normalize_product_code_short_padded(self, engine):
        result = engine.normalize_product_code("12")
        assert result == "0012"

    def test_normalize_product_code_digits_only(self, engine):
        result = engine.normalize_product_code("090121")
        assert result == "090121"


# ---------------------------------------------------------------------------
# Currency Normalization
# ---------------------------------------------------------------------------


class TestNormalizeCurrency:
    """Test currency normalization."""

    def test_normalize_currency_eur(self, engine):
        result = engine.normalize_currency("$1,234.56")
        assert "1234.56" in result
        assert "EUR" in result

    def test_normalize_currency_target(self, engine):
        result = engine.normalize_currency("1000", target="USD")
        assert "1000.00" in result
        assert "USD" in result


# ---------------------------------------------------------------------------
# normalize_record Dispatch
# ---------------------------------------------------------------------------


class TestNormalizeRecord:
    """Test normalize_record dispatching to correct handler."""

    def test_normalize_record_country_code(self, engine):
        record = engine.normalize_record(
            "country", "Brazil", NormalizationType.COUNTRY_CODE
        )
        assert isinstance(record, NormalizationRecord)
        assert record.normalized_value == "BR"
        assert record.normalization_type == NormalizationType.COUNTRY_CODE
        assert record.confidence == Decimal("1.0")

    def test_normalize_record_date(self, engine):
        record = engine.normalize_record(
            "import_date", "15/01/2026", NormalizationType.DATE
        )
        assert record.normalized_value == "2026-01-15"
        assert record.normalization_type == NormalizationType.DATE
        assert record.confidence == Decimal("1.0")

    def test_normalize_record_coordinate(self, engine):
        record = engine.normalize_record(
            "origin_coords", "1.2345, 103.456", NormalizationType.COORDINATE
        )
        assert "1.234500" in record.normalized_value
        assert record.normalization_type == NormalizationType.COORDINATE
        assert record.confidence == Decimal("1.0")

    def test_normalize_record_product_code(self, engine):
        record = engine.normalize_record(
            "hs_code", "0901.11", NormalizationType.PRODUCT_CODE
        )
        assert record.normalized_value == "090111"
        assert record.confidence == Decimal("1.0")

    def test_normalize_record_unit(self, engine):
        record = engine.normalize_record(
            "weight", "500 kilograms", NormalizationType.UNIT
        )
        assert record.normalized_value == "500 kg"

    def test_normalize_record_certificate_id(self, engine):
        record = engine.normalize_record(
            "cert", "fsc-c012345", NormalizationType.CERTIFICATE_ID
        )
        assert record.normalized_value == "FSC-C012345"

    def test_normalize_record_address(self, engine):
        record = engine.normalize_record(
            "addr", "123  Main St.", NormalizationType.ADDRESS
        )
        assert "  " not in record.normalized_value


# ---------------------------------------------------------------------------
# Batch Normalization
# ---------------------------------------------------------------------------


class TestNormalizeBatch:
    """Test batch normalization."""

    def test_normalize_batch(self, engine):
        records = [
            ("country", "Brazil", NormalizationType.COUNTRY_CODE),
            ("date", "2026-01-15", NormalizationType.DATE),
            ("weight", "100 kilograms", NormalizationType.UNIT),
        ]
        results = engine.normalize_batch(records)
        assert len(results) == 3
        assert results[0].normalized_value == "BR"
        assert results[1].normalized_value == "2026-01-15"
        assert results[2].normalized_value == "100 kg"

    def test_normalize_batch_empty(self, engine):
        results = engine.normalize_batch([])
        assert results == []


# ---------------------------------------------------------------------------
# Normalization Statistics
# ---------------------------------------------------------------------------


class TestNormalizationStats:
    """Test normalization engine statistics."""

    def test_normalization_stats_after_operations(self, engine):
        engine.normalize_record("c", "Brazil", NormalizationType.COUNTRY_CODE)
        engine.normalize_record("d", "2026-01-01", NormalizationType.DATE)
        stats = engine.get_normalization_stats()
        assert stats["total_normalizations"] == 2
        assert stats["error_count"] == 0
        assert "country_code" in stats["type_breakdown"]
        assert "date" in stats["type_breakdown"]
        assert stats["average_confidence"] > 0

    def test_normalization_stats_empty(self, engine):
        stats = engine.get_normalization_stats()
        assert stats["total_normalizations"] == 0

    def test_clear_history(self, engine):
        engine.normalize_record("c", "Brazil", NormalizationType.COUNTRY_CODE)
        engine.clear_history()
        stats = engine.get_normalization_stats()
        assert stats["total_normalizations"] == 0
        assert stats["error_count"] == 0


# ---------------------------------------------------------------------------
# Confidence Scoring
# ---------------------------------------------------------------------------


class TestConfidenceScoring:
    """Test confidence computation for various normalization types."""

    def test_high_confidence_country_code(self, engine):
        record = engine.normalize_record(
            "country", "Colombia", NormalizationType.COUNTRY_CODE
        )
        assert record.confidence == Decimal("1.0")

    def test_half_confidence_when_unchanged_country(self, engine):
        # "DE" is already a valid alpha-2 and returns "DE" unchanged
        record = engine.normalize_record(
            "country", "DE", NormalizationType.COUNTRY_CODE
        )
        assert record.confidence == Decimal("0.5")

    def test_high_confidence_date(self, engine):
        record = engine.normalize_record(
            "date", "15/01/2026", NormalizationType.DATE
        )
        assert record.confidence == Decimal("1.0")
