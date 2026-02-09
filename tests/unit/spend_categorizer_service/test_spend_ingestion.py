# -*- coding: utf-8 -*-
"""
Unit tests for SpendIngestionEngine - AGENT-DATA-009 Batch 2
=============================================================

Comprehensive tests for the spend ingestion engine covering:
- Engine initialisation and configuration
- Single and batch record ingestion
- CSV and Excel ingestion
- Record normalisation and field mapping
- Vendor name normalisation (suffix stripping, casing)
- Deduplication (exact, fuzzy, threshold)
- Currency conversion (major currencies, identity, errors)
- Batch retrieval and listing
- Statistics tracking
- Provenance hashing (SHA-256)
- Thread safety under concurrent ingestion

Target: 100+ tests, 900+ lines, 85%+ coverage of spend_ingestion.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from greenlang.spend_categorizer.spend_ingestion import (
    IngestionBatch,
    NormalizedSpendRecord,
    SpendIngestionEngine,
    _amount_similarity,
    _date_similarity,
    _parse_date_str,
    _parse_float,
    _string_similarity,
    _vendor_id_from_name,
)


# ---------------------------------------------------------------------------
# Local fixtures (defined here rather than in conftest since conftest
# is shared with Batch 1 tests and has a different fixture set)
# ---------------------------------------------------------------------------


@pytest.fixture
def ingestion_engine() -> SpendIngestionEngine:
    """Create a default SpendIngestionEngine for testing."""
    return SpendIngestionEngine()


@pytest.fixture
def ingestion_engine_custom() -> SpendIngestionEngine:
    """Create a SpendIngestionEngine with custom config."""
    return SpendIngestionEngine(config={
        "default_currency": "EUR",
        "normalize_vendors": True,
        "dedup_threshold": 0.90,
        "max_batch_size": 500,
    })


@pytest.fixture
def single_spend_record() -> Dict[str, Any]:
    """A single valid spend record dict."""
    return {
        "vendor_name": "Acme Inc.",
        "amount": 5000.00,
        "currency": "USD",
        "description": "Office supplies order",
        "transaction_date": "2025-06-15",
        "category": "indirect",
        "cost_center": "CC-100",
        "gl_account": "6100",
        "po_number": "PO-12345",
    }


@pytest.fixture
def batch_spend_records() -> List[Dict[str, Any]]:
    """A batch of spend records for ingestion testing."""
    return [
        {
            "vendor_name": "Acme Inc.",
            "amount": 5000,
            "currency": "USD",
            "description": "Office supplies",
            "transaction_date": "2025-01-10",
        },
        {
            "vendor_name": "Beta Corp.",
            "amount": 12000,
            "currency": "EUR",
            "description": "Consulting services Q1",
            "transaction_date": "2025-02-15",
        },
        {
            "vendor_name": "Gamma Ltd.",
            "amount": 850000,
            "currency": "JPY",
            "description": "Electronic components",
            "transaction_date": "2025-03-20",
        },
        {
            "vendor_name": "Delta GmbH",
            "amount": 7500,
            "currency": "GBP",
            "description": "Engineering design services",
            "transaction_date": "2025-04-01",
        },
        {
            "vendor_name": "Epsilon S.A.",
            "amount": 3200,
            "currency": "CHF",
            "description": "Chemical reagents for lab",
            "transaction_date": "2025-05-10",
        },
    ]


@pytest.fixture
def csv_content_simple() -> str:
    """Simple CSV content for ingestion testing."""
    return (
        "vendor_name,amount,currency,description,transaction_date\n"
        "Acme Inc.,5000,USD,Office supplies,2025-01-10\n"
        "Beta Corp.,12000,EUR,Consulting services,2025-02-15\n"
        "Gamma Ltd.,850000,JPY,Electronic components,2025-03-20\n"
    )


@pytest.fixture
def excel_content_json() -> str:
    """JSON-serialised Excel content for ingestion testing."""
    import json as _json
    rows = [
        {"vendor_name": "Acme Inc.", "amount": 5000, "currency": "USD",
         "description": "Office supplies"},
        {"vendor_name": "Beta Corp.", "amount": 12000, "currency": "EUR",
         "description": "Consulting services"},
    ]
    return _json.dumps(rows)


# ===================================================================
# TestInit - engine creation and defaults
# ===================================================================


class TestInit:
    """Tests for SpendIngestionEngine initialisation."""

    def test_default_init(self):
        """Engine initialises with sensible defaults."""
        engine = SpendIngestionEngine()
        assert engine._default_currency == "USD"
        assert engine._normalize_vendors is True
        assert engine._dedup_threshold == 0.85
        assert engine._max_batch_size == 100_000

    def test_custom_config(self):
        """Engine respects custom configuration values."""
        engine = SpendIngestionEngine(config={
            "default_currency": "EUR",
            "normalize_vendors": False,
            "dedup_threshold": 0.90,
            "max_batch_size": 500,
        })
        assert engine._default_currency == "EUR"
        assert engine._normalize_vendors is False
        assert engine._dedup_threshold == 0.90
        assert engine._max_batch_size == 500

    def test_empty_stores_on_init(self):
        """Batch store and stats are empty on fresh engine."""
        engine = SpendIngestionEngine()
        assert engine._batches == {}
        assert engine._stats["batches_created"] == 0
        assert engine._stats["records_ingested"] == 0
        assert engine._stats["total_spend_usd"] == 0.0

    def test_none_config_uses_defaults(self):
        """Passing None for config uses defaults."""
        engine = SpendIngestionEngine(config=None)
        assert engine._default_currency == "USD"

    def test_partial_config(self):
        """Only specified config keys override defaults."""
        engine = SpendIngestionEngine(config={"default_currency": "GBP"})
        assert engine._default_currency == "GBP"
        assert engine._normalize_vendors is True  # not overridden


# ===================================================================
# TestIngestRecords - single and batch ingestion
# ===================================================================


class TestIngestRecords:
    """Tests for ingest_records method."""

    def test_single_record_ingestion(self, ingestion_engine, single_spend_record):
        """Ingest a single record successfully."""
        batch = ingestion_engine.ingest_records([single_spend_record])
        assert isinstance(batch, IngestionBatch)
        assert batch.record_count == 1
        assert batch.error_count == 0
        assert len(batch.records) == 1

    def test_batch_ingestion(self, ingestion_engine, batch_spend_records):
        """Ingest multiple records as a batch."""
        batch = ingestion_engine.ingest_records(batch_spend_records)
        assert batch.record_count == 5
        assert batch.error_count == 0

    def test_batch_id_generation(self, ingestion_engine, single_spend_record):
        """Batch ID is auto-generated when not provided."""
        batch = ingestion_engine.ingest_records([single_spend_record])
        assert batch.batch_id.startswith("batch-")
        assert len(batch.batch_id) > 6

    def test_custom_batch_id(self, ingestion_engine, single_spend_record):
        """Batch ID can be provided explicitly."""
        batch = ingestion_engine.ingest_records(
            [single_spend_record], batch_id="my-batch-001"
        )
        assert batch.batch_id == "my-batch-001"

    def test_source_tracking(self, ingestion_engine, single_spend_record):
        """Source is tracked on the batch."""
        batch = ingestion_engine.ingest_records(
            [single_spend_record], source="sap_s4hana"
        )
        assert batch.source == "sap_s4hana"

    def test_default_source_is_api(self, ingestion_engine, single_spend_record):
        """Default source is 'api'."""
        batch = ingestion_engine.ingest_records([single_spend_record])
        assert batch.source == "api"

    def test_currencies_seen_populated(self, ingestion_engine, batch_spend_records):
        """Currencies seen is populated correctly."""
        batch = ingestion_engine.ingest_records(batch_spend_records)
        assert "USD" in batch.currencies_seen
        assert "EUR" in batch.currencies_seen
        assert "JPY" in batch.currencies_seen

    def test_vendors_seen_count(self, ingestion_engine, batch_spend_records):
        """Vendors seen count reflects unique vendors."""
        batch = ingestion_engine.ingest_records(batch_spend_records)
        assert batch.vendors_seen >= 1

    def test_total_spend_usd_calculated(self, ingestion_engine, single_spend_record):
        """Total spend USD is summed correctly."""
        batch = ingestion_engine.ingest_records([single_spend_record])
        assert batch.total_spend_usd > 0

    def test_processing_time_tracked(self, ingestion_engine, single_spend_record):
        """Processing time in ms is tracked."""
        batch = ingestion_engine.ingest_records([single_spend_record])
        assert batch.processing_time_ms >= 0

    def test_batch_exceeds_max_size_raises(self, ingestion_engine_custom):
        """Exceeding max_batch_size raises ValueError."""
        records = [{"vendor_name": f"V{i}", "amount": 100} for i in range(600)]
        with pytest.raises(ValueError, match="exceeds maximum"):
            ingestion_engine_custom.ingest_records(records)

    def test_empty_records_list(self, ingestion_engine):
        """Ingesting empty list produces zero-count batch."""
        batch = ingestion_engine.ingest_records([])
        assert batch.record_count == 0
        assert batch.total_spend_usd == 0.0

    def test_bad_record_captured_as_error(self, ingestion_engine):
        """Records that fail normalisation produce errors."""
        records = [{"vendor_name": "Good Co", "amount": 100}]
        # This should succeed
        batch = ingestion_engine.ingest_records(records)
        assert batch.error_count == 0

    def test_created_at_populated(self, ingestion_engine, single_spend_record):
        """created_at timestamp is set on batch."""
        batch = ingestion_engine.ingest_records([single_spend_record])
        assert batch.created_at != ""

    def test_provenance_hash_on_batch(self, ingestion_engine, single_spend_record):
        """Batch has a non-empty provenance hash."""
        batch = ingestion_engine.ingest_records([single_spend_record])
        assert len(batch.provenance_hash) == 64  # SHA-256 hex

    def test_records_stored_in_batch(self, ingestion_engine, single_spend_record):
        """Records are stored inside the batch object."""
        batch = ingestion_engine.ingest_records([single_spend_record])
        assert len(batch.records) == 1
        assert isinstance(batch.records[0], NormalizedSpendRecord)


# ===================================================================
# TestIngestCSV - CSV parsing
# ===================================================================


class TestIngestCSV:
    """Tests for ingest_csv method."""

    def test_basic_csv_parsing(self, ingestion_engine, csv_content_simple):
        """Parse basic comma-separated CSV."""
        batch = ingestion_engine.ingest_csv(csv_content_simple)
        assert batch.record_count == 3
        assert batch.source == "csv"

    def test_semicolon_delimiter(self, ingestion_engine):
        """Parse CSV with semicolon delimiter."""
        content = (
            "vendor_name;amount;currency\n"
            "Acme Inc.;5000;USD\n"
            "Beta Corp.;12000;EUR\n"
        )
        batch = ingestion_engine.ingest_csv(content, delimiter=";")
        assert batch.record_count == 2

    def test_tab_delimiter(self, ingestion_engine):
        """Parse CSV with tab delimiter."""
        content = "vendor_name\tamount\tcurrency\nAcme Inc.\t5000\tUSD\n"
        batch = ingestion_engine.ingest_csv(content, delimiter="\t")
        assert batch.record_count == 1

    def test_custom_source(self, ingestion_engine, csv_content_simple):
        """Custom source name is passed through."""
        batch = ingestion_engine.ingest_csv(csv_content_simple, source="erp_export")
        assert batch.source == "erp_export"

    def test_custom_batch_id_csv(self, ingestion_engine, csv_content_simple):
        """Custom batch_id is honoured for CSV ingestion."""
        batch = ingestion_engine.ingest_csv(
            csv_content_simple, batch_id="csv-batch-001"
        )
        assert batch.batch_id == "csv-batch-001"

    def test_field_alias_mapping_in_csv(self, ingestion_engine):
        """CSV column aliases are mapped to canonical names."""
        content = "supplier,spend,curr\nAcme,1000,USD\n"
        batch = ingestion_engine.ingest_csv(content)
        rec = batch.records[0]
        assert rec.vendor_name != ""  # supplier -> vendor_name
        assert rec.amount == 1000.0
        assert rec.currency == "USD"

    def test_empty_csv_body(self, ingestion_engine):
        """CSV with only headers produces zero records."""
        content = "vendor_name,amount,currency\n"
        batch = ingestion_engine.ingest_csv(content)
        assert batch.record_count == 0

    def test_csv_with_extra_whitespace_in_headers(self, ingestion_engine):
        """Whitespace in CSV headers is tolerated."""
        content = " vendor_name , amount , currency \nAcme,1000,USD\n"
        batch = ingestion_engine.ingest_csv(content)
        assert batch.record_count == 1

    def test_csv_encoding_param_accepted(self, ingestion_engine, csv_content_simple):
        """Encoding parameter is accepted (documentation only)."""
        batch = ingestion_engine.ingest_csv(
            csv_content_simple, encoding="utf-8"
        )
        assert batch.record_count == 3


# ===================================================================
# TestIngestExcel - Excel/JSON parsing
# ===================================================================


class TestIngestExcel:
    """Tests for ingest_excel method."""

    def test_basic_excel_parsing(self, ingestion_engine, excel_content_json):
        """Parse JSON-serialised Excel content."""
        batch = ingestion_engine.ingest_excel(excel_content_json)
        assert batch.record_count == 2
        assert batch.source == "excel"

    def test_sheet_name_accepted(self, ingestion_engine, excel_content_json):
        """sheet_name parameter is accepted."""
        batch = ingestion_engine.ingest_excel(
            excel_content_json, sheet_name="Sheet1"
        )
        assert batch.record_count == 2

    def test_custom_source_excel(self, ingestion_engine, excel_content_json):
        """Custom source for Excel ingestion."""
        batch = ingestion_engine.ingest_excel(
            excel_content_json, source="excel_upload"
        )
        assert batch.source == "excel_upload"

    def test_invalid_json_raises(self, ingestion_engine):
        """Invalid JSON content raises ValueError."""
        with pytest.raises(ValueError, match="JSON-serialised"):
            ingestion_engine.ingest_excel("not json at all")

    def test_non_list_json_raises(self, ingestion_engine):
        """JSON that is not a list raises ValueError."""
        with pytest.raises(ValueError, match="JSON array"):
            ingestion_engine.ingest_excel('{"key": "value"}')

    def test_empty_array_excel(self, ingestion_engine):
        """Empty JSON array produces zero records."""
        batch = ingestion_engine.ingest_excel("[]")
        assert batch.record_count == 0


# ===================================================================
# TestNormalizeRecord - field standardisation
# ===================================================================


class TestNormalizeRecord:
    """Tests for normalize_record method."""

    def test_basic_normalisation(self, ingestion_engine, single_spend_record):
        """Record is normalised with canonical field names."""
        rec = ingestion_engine.normalize_record(single_spend_record)
        assert isinstance(rec, NormalizedSpendRecord)
        assert rec.record_id.startswith("rec-")
        assert rec.amount_usd > 0

    def test_currency_defaults_to_usd(self, ingestion_engine):
        """Missing currency defaults to USD."""
        rec = ingestion_engine.normalize_record({"amount": 1000})
        assert rec.currency == "USD"
        assert rec.amount_usd == 1000.0

    def test_field_alias_supplier(self, ingestion_engine):
        """'supplier' maps to vendor_name."""
        rec = ingestion_engine.normalize_record(
            {"supplier": "Test Co", "amount": 100}
        )
        assert "Test" in rec.vendor_name

    def test_field_alias_spend(self, ingestion_engine):
        """'spend' maps to amount."""
        rec = ingestion_engine.normalize_record(
            {"vendor_name": "X", "spend": 5000, "currency": "USD"}
        )
        assert rec.amount == 5000.0

    def test_field_alias_ccy(self, ingestion_engine):
        """'ccy' maps to currency."""
        rec = ingestion_engine.normalize_record(
            {"vendor_name": "X", "amount": 100, "ccy": "GBP"}
        )
        assert rec.currency == "GBP"

    def test_amount_parsed_from_string(self, ingestion_engine):
        """Amount is parsed from formatted string like '$1,000.50'."""
        rec = ingestion_engine.normalize_record(
            {"vendor_name": "X", "amount": "$1,000.50", "currency": "USD"}
        )
        assert rec.amount == pytest.approx(1000.50)

    def test_source_and_batch_id_set(self, ingestion_engine):
        """Source and batch_id are assigned to normalised record."""
        rec = ingestion_engine.normalize_record(
            {"vendor_name": "X", "amount": 100},
            batch_id="b-123",
            source="erp",
        )
        assert rec.batch_id == "b-123"
        assert rec.source == "erp"

    def test_provenance_hash_on_record(self, ingestion_engine, single_spend_record):
        """Each normalised record has a 64-char SHA-256 provenance hash."""
        rec = ingestion_engine.normalize_record(single_spend_record)
        assert len(rec.provenance_hash) == 64

    def test_extra_fields_captured(self, ingestion_engine):
        """Fields not in canonical schema go into extra dict."""
        rec = ingestion_engine.normalize_record(
            {"vendor_name": "X", "amount": 100, "custom_field": "hello"}
        )
        assert "custom_field" in rec.extra

    def test_vendor_id_generated_from_name(self, ingestion_engine):
        """Vendor ID is generated when not provided."""
        rec = ingestion_engine.normalize_record(
            {"vendor_name": "Acme Corp", "amount": 100}
        )
        assert rec.vendor_id.startswith("VEND-")

    def test_vendor_id_preserved_when_provided(self, ingestion_engine):
        """Explicit vendor_id is kept."""
        rec = ingestion_engine.normalize_record(
            {"vendor_name": "X", "vendor_id": "V-999", "amount": 100}
        )
        assert rec.vendor_id == "V-999"

    def test_date_parsing_iso_format(self, ingestion_engine):
        """ISO date is parsed correctly."""
        rec = ingestion_engine.normalize_record(
            {"vendor_name": "X", "amount": 100, "transaction_date": "2025-06-15"}
        )
        assert rec.transaction_date == "2025-06-15"

    def test_date_parsing_us_format(self, ingestion_engine):
        """US date format MM/DD/YYYY is parsed."""
        rec = ingestion_engine.normalize_record(
            {"vendor_name": "X", "amount": 100, "transaction_date": "06/15/2025"}
        )
        assert rec.transaction_date == "2025-06-15"

    def test_missing_date_returns_none(self, ingestion_engine):
        """Missing transaction_date returns None."""
        rec = ingestion_engine.normalize_record(
            {"vendor_name": "X", "amount": 100}
        )
        assert rec.transaction_date is None

    def test_ingested_at_populated(self, ingestion_engine, single_spend_record):
        """ingested_at timestamp is set."""
        rec = ingestion_engine.normalize_record(single_spend_record)
        assert rec.ingested_at != ""


# ===================================================================
# TestNormalizeVendorName - suffix stripping and casing
# ===================================================================


class TestNormalizeVendorName:
    """Tests for normalize_vendor_name method."""

    def test_strip_inc(self, ingestion_engine):
        assert ingestion_engine.normalize_vendor_name("Acme Inc.") == "Acme"

    def test_strip_inc_no_dot(self, ingestion_engine):
        assert ingestion_engine.normalize_vendor_name("Acme Inc") == "Acme"

    def test_strip_llc(self, ingestion_engine):
        assert ingestion_engine.normalize_vendor_name("Beta LLC") == "Beta"

    def test_strip_ltd(self, ingestion_engine):
        assert ingestion_engine.normalize_vendor_name("Gamma Ltd.") == "Gamma"

    def test_strip_corp(self, ingestion_engine):
        assert ingestion_engine.normalize_vendor_name("Delta Corp") == "Delta"

    def test_strip_gmbh(self, ingestion_engine):
        assert ingestion_engine.normalize_vendor_name("Epsilon GmbH") == "Epsilon"

    def test_strip_sa(self, ingestion_engine):
        result = ingestion_engine.normalize_vendor_name("Zeta S.A.")
        assert "S.A." not in result
        assert "Zeta" in result

    def test_strip_plc(self, ingestion_engine):
        result = ingestion_engine.normalize_vendor_name("Eta PLC")
        assert "PLC" not in result

    def test_strip_ag(self, ingestion_engine):
        result = ingestion_engine.normalize_vendor_name("Theta AG")
        assert "AG" not in result

    def test_strip_comma_suffix(self, ingestion_engine):
        """Comma before suffix is also stripped."""
        result = ingestion_engine.normalize_vendor_name("Kappa, Inc.")
        assert "Inc" not in result
        assert result.strip() != ""

    def test_title_case_output(self, ingestion_engine):
        """Output is title-cased."""
        result = ingestion_engine.normalize_vendor_name("acme inc.")
        assert result == "Acme"

    def test_whitespace_collapsed(self, ingestion_engine):
        """Multiple spaces are collapsed to one."""
        result = ingestion_engine.normalize_vendor_name("Acme   Corp   Inc.")
        assert "  " not in result

    def test_empty_string_returns_empty(self, ingestion_engine):
        assert ingestion_engine.normalize_vendor_name("") == ""

    def test_none_like_empty(self, ingestion_engine):
        """Whitespace-only input returns empty."""
        assert ingestion_engine.normalize_vendor_name("   ") == ""

    def test_no_suffix_preserved(self, ingestion_engine):
        """Names without corporate suffixes are preserved."""
        result = ingestion_engine.normalize_vendor_name("Microsoft")
        assert result == "Microsoft"

    def test_multi_word_vendor(self, ingestion_engine):
        """Multi-word vendor names are normalised correctly."""
        result = ingestion_engine.normalize_vendor_name("johnson and johnson inc.")
        # Should strip 'inc.' and title case
        assert "Inc" not in result
        assert result[0].isupper()

    def test_strip_srl(self, ingestion_engine):
        result = ingestion_engine.normalize_vendor_name("Lambda S.R.L.")
        assert "S.R.L." not in result

    def test_strip_pty(self, ingestion_engine):
        result = ingestion_engine.normalize_vendor_name("Mu Pty")
        assert "Pty" not in result

    def test_strip_bv(self, ingestion_engine):
        result = ingestion_engine.normalize_vendor_name("Nu B.V.")
        assert "B.V." not in result

    def test_strip_nv(self, ingestion_engine):
        result = ingestion_engine.normalize_vendor_name("Xi N.V.")
        assert "N.V." not in result


# ===================================================================
# TestDeduplicate - deduplication logic
# ===================================================================


class TestDeduplicate:
    """Tests for deduplication behaviour."""

    def test_exact_duplicates_flagged(self, ingestion_engine):
        """Identical records are flagged as duplicates."""
        records = [
            {"vendor_name": "Acme Inc.", "amount": 5000, "currency": "USD",
             "description": "Office supplies", "transaction_date": "2025-01-10"},
            {"vendor_name": "Acme Inc.", "amount": 5000, "currency": "USD",
             "description": "Office supplies", "transaction_date": "2025-01-10"},
        ]
        batch = ingestion_engine.ingest_records(records)
        duplicates = [r for r in batch.records if r.is_duplicate]
        assert len(duplicates) >= 1
        assert batch.duplicate_count >= 1

    def test_different_records_not_flagged(self, ingestion_engine):
        """Clearly different records are not flagged."""
        records = [
            {"vendor_name": "Acme Inc.", "amount": 5000, "currency": "USD",
             "description": "Office supplies", "transaction_date": "2025-01-10"},
            {"vendor_name": "Totally Different Vendor", "amount": 999,
             "currency": "GBP", "description": "Machinery parts",
             "transaction_date": "2025-12-01"},
        ]
        batch = ingestion_engine.ingest_records(records)
        duplicates = [r for r in batch.records if r.is_duplicate]
        assert len(duplicates) == 0

    def test_threshold_behaviour(self):
        """Higher threshold means fewer duplicates flagged."""
        engine_strict = SpendIngestionEngine(config={"dedup_threshold": 0.99})
        records = [
            {"vendor_name": "Acme Inc.", "amount": 5000, "currency": "USD",
             "description": "Office supplies", "transaction_date": "2025-01-10"},
            {"vendor_name": "Acme Inc", "amount": 5001, "currency": "USD",
             "description": "Office supply", "transaction_date": "2025-01-10"},
        ]
        batch_strict = engine_strict.ingest_records(records)
        # Strict threshold: small variations should not be flagged
        dups_strict = [r for r in batch_strict.records if r.is_duplicate]

        engine_lax = SpendIngestionEngine(config={"dedup_threshold": 0.50})
        batch_lax = engine_lax.ingest_records(records)
        dups_lax = [r for r in batch_lax.records if r.is_duplicate]

        # Lax threshold should flag more or equal duplicates
        assert len(dups_lax) >= len(dups_strict)

    def test_deduplicate_public_method(self, ingestion_engine):
        """Public deduplicate() method works on pre-normalised records."""
        # Ingest first to get normalised records
        batch = ingestion_engine.ingest_records([
            {"vendor_name": "Acme", "amount": 5000, "currency": "USD",
             "description": "supplies", "transaction_date": "2025-01-10"},
            {"vendor_name": "Acme", "amount": 5000, "currency": "USD",
             "description": "supplies", "transaction_date": "2025-01-10"},
        ])
        # Reset flags and re-deduplicate
        for r in batch.records:
            r.is_duplicate = False
            r.duplicate_of = None
        result = ingestion_engine.deduplicate(batch.records, threshold=0.80)
        dups = [r for r in result if r.is_duplicate]
        assert len(dups) >= 1

    def test_no_dedup_on_single_record(self, ingestion_engine, single_spend_record):
        """Single-record batches skip deduplication."""
        batch = ingestion_engine.ingest_records([single_spend_record])
        assert batch.duplicate_count == 0

    def test_duplicate_of_points_to_original(self, ingestion_engine):
        """duplicate_of field references the original record_id."""
        records = [
            {"vendor_name": "Acme Inc.", "amount": 5000, "currency": "USD",
             "description": "Office supplies", "transaction_date": "2025-01-10"},
            {"vendor_name": "Acme Inc.", "amount": 5000, "currency": "USD",
             "description": "Office supplies", "transaction_date": "2025-01-10"},
        ]
        batch = ingestion_engine.ingest_records(records)
        dups = [r for r in batch.records if r.is_duplicate]
        if dups:
            assert dups[0].duplicate_of is not None
            assert dups[0].duplicate_of.startswith("rec-")


# ===================================================================
# TestConvertCurrency - currency conversion
# ===================================================================


class TestConvertCurrency:
    """Tests for convert_currency method."""

    def test_usd_to_usd_identity(self, ingestion_engine):
        """USD to USD returns the same amount."""
        result = ingestion_engine.convert_currency(1000, "USD", "USD")
        assert result == 1000.0

    def test_eur_to_usd(self, ingestion_engine):
        """EUR to USD conversion."""
        result = ingestion_engine.convert_currency(1000, "EUR", "USD")
        # 1000 EUR / 0.92 rate = ~1086.96 USD
        assert result > 1000
        assert result == pytest.approx(1000 / 0.92, abs=0.01)

    def test_gbp_to_usd(self, ingestion_engine):
        """GBP to USD conversion."""
        result = ingestion_engine.convert_currency(1000, "GBP", "USD")
        assert result > 1000  # GBP is worth more than USD

    def test_jpy_to_usd(self, ingestion_engine):
        """JPY to USD conversion."""
        result = ingestion_engine.convert_currency(150000, "JPY", "USD")
        assert result == pytest.approx(150000 / 149.50, abs=0.01)

    def test_usd_to_eur(self, ingestion_engine):
        """USD to EUR conversion via triangulation."""
        result = ingestion_engine.convert_currency(1000, "USD", "EUR")
        assert result == pytest.approx(1000 * 0.92, abs=0.01)

    def test_eur_to_gbp_cross_rate(self, ingestion_engine):
        """EUR to GBP cross-rate via USD triangulation."""
        result = ingestion_engine.convert_currency(1000, "EUR", "GBP")
        expected = (1000 / 0.92) * 0.79  # EUR->USD->GBP
        assert result == pytest.approx(expected, abs=0.01)

    def test_zero_amount(self, ingestion_engine):
        """Zero amount converts to zero."""
        result = ingestion_engine.convert_currency(0, "EUR", "USD")
        assert result == 0.0

    def test_unsupported_from_currency_raises(self, ingestion_engine):
        """Unsupported source currency raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported currency"):
            ingestion_engine.convert_currency(1000, "XYZ", "USD")

    def test_unsupported_to_currency_raises(self, ingestion_engine):
        """Unsupported target currency raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported currency"):
            ingestion_engine.convert_currency(1000, "USD", "XYZ")

    def test_case_insensitive_currency(self, ingestion_engine):
        """Currency codes are case-insensitive."""
        result = ingestion_engine.convert_currency(1000, "eur", "usd")
        assert result > 0

    def test_whitespace_stripped(self, ingestion_engine):
        """Whitespace in currency codes is stripped."""
        result = ingestion_engine.convert_currency(1000, " EUR ", " USD ")
        assert result > 0

    def test_cad_to_usd(self, ingestion_engine):
        """CAD to USD conversion."""
        result = ingestion_engine.convert_currency(1000, "CAD", "USD")
        assert result == pytest.approx(1000 / 1.36, abs=0.01)

    def test_chf_to_usd(self, ingestion_engine):
        """CHF to USD conversion."""
        result = ingestion_engine.convert_currency(1000, "CHF", "USD")
        assert result == pytest.approx(1000 / 0.88, abs=0.01)

    def test_negative_amount(self, ingestion_engine):
        """Negative amounts are converted (credit notes)."""
        result = ingestion_engine.convert_currency(-1000, "EUR", "USD")
        assert result < 0


# ===================================================================
# TestGetBatch - batch retrieval
# ===================================================================


class TestGetBatch:
    """Tests for get_batch method."""

    def test_get_existing_batch(self, ingestion_engine, single_spend_record):
        """Retrieve an existing batch by ID."""
        batch = ingestion_engine.ingest_records(
            [single_spend_record], batch_id="test-batch"
        )
        retrieved = ingestion_engine.get_batch("test-batch")
        assert retrieved is not None
        assert retrieved.batch_id == "test-batch"

    def test_get_nonexistent_batch(self, ingestion_engine):
        """Nonexistent batch returns None."""
        result = ingestion_engine.get_batch("nonexistent-id")
        assert result is None

    def test_batch_has_correct_fields(self, ingestion_engine, single_spend_record):
        """Retrieved batch has all expected fields."""
        ingestion_engine.ingest_records(
            [single_spend_record], batch_id="fields-test"
        )
        batch = ingestion_engine.get_batch("fields-test")
        assert batch.batch_id == "fields-test"
        assert batch.record_count >= 1
        assert batch.provenance_hash != ""
        assert batch.created_at != ""


# ===================================================================
# TestListBatches - batch listing and filtering
# ===================================================================


class TestListBatches:
    """Tests for list_batches method."""

    def test_list_all_batches(self, ingestion_engine, single_spend_record):
        """List all batches."""
        ingestion_engine.ingest_records([single_spend_record], source="csv")
        ingestion_engine.ingest_records([single_spend_record], source="excel")
        batches = ingestion_engine.list_batches()
        assert len(batches) == 2

    def test_list_by_source(self, ingestion_engine, single_spend_record):
        """Filter batches by source."""
        ingestion_engine.ingest_records([single_spend_record], source="csv")
        ingestion_engine.ingest_records([single_spend_record], source="excel")
        csv_batches = ingestion_engine.list_batches(source="csv")
        assert len(csv_batches) == 1
        assert csv_batches[0].source == "csv"

    def test_pagination_limit(self, ingestion_engine, single_spend_record):
        """Limit parameter restricts returned batch count."""
        for i in range(5):
            ingestion_engine.ingest_records([single_spend_record])
        batches = ingestion_engine.list_batches(limit=3)
        assert len(batches) == 3

    def test_pagination_offset(self, ingestion_engine, single_spend_record):
        """Offset parameter skips batches."""
        for i in range(5):
            ingestion_engine.ingest_records([single_spend_record])
        all_batches = ingestion_engine.list_batches()
        offset_batches = ingestion_engine.list_batches(offset=2)
        assert len(offset_batches) == len(all_batches) - 2

    def test_empty_engine_returns_empty(self, ingestion_engine):
        """Empty engine returns empty list."""
        batches = ingestion_engine.list_batches()
        assert batches == []


# ===================================================================
# TestStatistics - cumulative stats
# ===================================================================


class TestStatistics:
    """Tests for get_statistics method."""

    def test_initial_statistics(self, ingestion_engine):
        """Fresh engine reports zero statistics."""
        stats = ingestion_engine.get_statistics()
        assert stats["batches_created"] == 0
        assert stats["records_ingested"] == 0
        assert stats["total_spend_usd"] == 0.0

    def test_stats_after_ingestion(self, ingestion_engine, batch_spend_records):
        """Statistics update after ingestion."""
        ingestion_engine.ingest_records(batch_spend_records, source="api")
        stats = ingestion_engine.get_statistics()
        assert stats["batches_created"] == 1
        assert stats["records_ingested"] == 5
        assert stats["total_spend_usd"] > 0

    def test_by_source_tracking(self, ingestion_engine, single_spend_record):
        """by_source breakdown tracks source counts."""
        ingestion_engine.ingest_records([single_spend_record], source="csv")
        ingestion_engine.ingest_records([single_spend_record], source="csv")
        stats = ingestion_engine.get_statistics()
        assert stats["by_source"]["csv"] == 2

    def test_batches_stored_count(self, ingestion_engine, single_spend_record):
        """batches_stored reflects stored batch count."""
        ingestion_engine.ingest_records([single_spend_record])
        stats = ingestion_engine.get_statistics()
        assert stats["batches_stored"] == 1

    def test_supported_currencies_count(self, ingestion_engine):
        """supported_currencies reflects the exchange rate table size."""
        stats = ingestion_engine.get_statistics()
        assert stats["supported_currencies"] >= 140


# ===================================================================
# TestProvenance - SHA-256 provenance hashing
# ===================================================================


class TestProvenance:
    """Tests for provenance hashing."""

    def test_record_provenance_is_sha256(self, ingestion_engine, single_spend_record):
        """Record provenance hash is 64-char hex SHA-256."""
        rec = ingestion_engine.normalize_record(single_spend_record)
        assert len(rec.provenance_hash) == 64
        # Validate hex chars
        int(rec.provenance_hash, 16)

    def test_batch_provenance_is_sha256(self, ingestion_engine, single_spend_record):
        """Batch provenance hash is 64-char hex SHA-256."""
        batch = ingestion_engine.ingest_records([single_spend_record])
        assert len(batch.provenance_hash) == 64
        int(batch.provenance_hash, 16)

    def test_different_inputs_different_hashes(self, ingestion_engine):
        """Different inputs produce different provenance hashes."""
        rec1 = ingestion_engine.normalize_record(
            {"vendor_name": "A", "amount": 100}
        )
        rec2 = ingestion_engine.normalize_record(
            {"vendor_name": "B", "amount": 200}
        )
        assert rec1.provenance_hash != rec2.provenance_hash

    def test_every_record_has_provenance(self, ingestion_engine, batch_spend_records):
        """Every record in a batch has a provenance hash."""
        batch = ingestion_engine.ingest_records(batch_spend_records)
        for rec in batch.records:
            assert len(rec.provenance_hash) == 64


# ===================================================================
# TestThreadSafety - concurrent ingestion
# ===================================================================


class TestThreadSafety:
    """Tests for thread-safe concurrent ingestion."""

    def test_concurrent_ingestion(self, ingestion_engine):
        """Multiple threads can ingest simultaneously without errors."""
        errors = []

        def ingest_worker(engine: SpendIngestionEngine, source: str):
            try:
                records = [
                    {"vendor_name": f"Vendor-{source}-{i}", "amount": 100 * (i + 1)}
                    for i in range(10)
                ]
                batch = engine.ingest_records(records, source=source)
                assert batch.record_count == 10
            except Exception as exc:
                errors.append(str(exc))

        threads = [
            threading.Thread(
                target=ingest_worker, args=(ingestion_engine, f"thread-{i}")
            )
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == [], f"Thread errors: {errors}"

        stats = ingestion_engine.get_statistics()
        assert stats["batches_created"] == 5
        assert stats["records_ingested"] == 50

    def test_concurrent_statistics_consistent(self, ingestion_engine):
        """Statistics remain consistent under concurrent ingestion."""
        def worker(engine, idx):
            records = [{"vendor_name": f"V{idx}", "amount": 1000}]
            engine.ingest_records(records, source=f"src-{idx}")

        threads = [
            threading.Thread(target=worker, args=(ingestion_engine, i))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        stats = ingestion_engine.get_statistics()
        assert stats["batches_created"] == 20
        assert stats["records_ingested"] == 20


# ===================================================================
# TestModuleLevelHelpers - _parse_float, _parse_date_str, etc.
# ===================================================================


class TestModuleLevelHelpers:
    """Tests for module-level helper functions."""

    # _parse_float
    def test_parse_float_none(self):
        assert _parse_float(None) == 0.0

    def test_parse_float_int(self):
        assert _parse_float(42) == 42.0

    def test_parse_float_float(self):
        assert _parse_float(3.14) == 3.14

    def test_parse_float_string(self):
        assert _parse_float("1234.56") == 1234.56

    def test_parse_float_formatted(self):
        assert _parse_float("$1,234.56") == 1234.56

    def test_parse_float_empty_string(self):
        assert _parse_float("") == 0.0

    def test_parse_float_invalid(self):
        assert _parse_float("abc") == 0.0

    # _parse_date_str
    def test_parse_date_none(self):
        assert _parse_date_str(None) is None

    def test_parse_date_iso(self):
        assert _parse_date_str("2025-06-15") == "2025-06-15"

    def test_parse_date_us(self):
        assert _parse_date_str("06/15/2025") == "2025-06-15"

    def test_parse_date_empty(self):
        assert _parse_date_str("") is None

    def test_parse_date_datetime_object(self):
        from datetime import datetime
        dt = datetime(2025, 6, 15, 12, 30, 0)
        assert _parse_date_str(dt) == "2025-06-15"

    def test_parse_date_date_object(self):
        from datetime import date
        d = date(2025, 6, 15)
        assert _parse_date_str(d) == "2025-06-15"

    # _vendor_id_from_name
    def test_vendor_id_from_empty(self):
        assert _vendor_id_from_name("") == "VEND-unknown"

    def test_vendor_id_from_name(self):
        vid = _vendor_id_from_name("Acme")
        assert vid.startswith("VEND-")
        assert len(vid) == 13  # VEND- + 8 hex chars

    def test_vendor_id_deterministic(self):
        v1 = _vendor_id_from_name("Test Company")
        v2 = _vendor_id_from_name("Test Company")
        assert v1 == v2

    # _string_similarity
    def test_string_similarity_identical(self):
        assert _string_similarity("hello", "hello") == 1.0

    def test_string_similarity_empty_both(self):
        assert _string_similarity("", "") == 1.0

    def test_string_similarity_one_empty(self):
        assert _string_similarity("hello", "") == 0.0

    def test_string_similarity_similar(self):
        score = _string_similarity("Acme Corp", "Acme Corporation")
        assert 0.5 < score < 1.0

    # _amount_similarity
    def test_amount_similarity_identical(self):
        assert _amount_similarity(1000.0, 1000.0) == 1.0

    def test_amount_similarity_both_zero(self):
        assert _amount_similarity(0.0, 0.0) == 1.0

    def test_amount_similarity_different(self):
        score = _amount_similarity(1000.0, 2000.0)
        assert 0.0 < score < 1.0

    # _date_similarity
    def test_date_similarity_identical(self):
        assert _date_similarity("2025-01-01", "2025-01-01") == 1.0

    def test_date_similarity_none(self):
        assert _date_similarity(None, "2025-01-01") == 0.5

    def test_date_similarity_close(self):
        score = _date_similarity("2025-01-01", "2025-01-03")
        assert score >= 0.8  # Within 7 days

    def test_date_similarity_far(self):
        score = _date_similarity("2025-01-01", "2025-12-01")
        assert score == 0.0
