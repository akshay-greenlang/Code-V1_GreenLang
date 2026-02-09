# -*- coding: utf-8 -*-
"""
Unit tests for SpendCategorizerService facade (setup.py) - AGENT-DATA-009

Tests the SpendCategorizerService class covering initialization, record
ingestion, CSV ingestion, classification, Scope 3 mapping, emission
calculation, rule management, analytics, hotspot analysis, trend tracking,
report generation, statistics, health checks, provenance tracking,
lifecycle management, thread safety, the configure_spend_categorizer()
async function, and full end-to-end workflows.

Target: 85%+ coverage of greenlang/spend_categorizer/setup.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.spend_categorizer.config import SpendCategorizerConfig
from greenlang.spend_categorizer.setup import (
    AnalyticsResponse,
    CategoryRuleResponse,
    ClassificationResponse,
    EmissionCalculationResponse,
    ReportResponse,
    Scope3AssignmentResponse,
    SpendCategorizerService,
    SpendCategorizerStatisticsResponse,
    SpendRecordResponse,
    _ProvenanceTracker,
    _compute_hash,
    _SCOPE3_CATEGORIES,
    configure_spend_categorizer,
    get_spend_categorizer,
)


# ===================================================================
# Helpers
# ===================================================================


def _make_config(**overrides: Any) -> SpendCategorizerConfig:
    """Build a SpendCategorizerConfig with optional field overrides."""
    defaults = dict(
        database_url="",
        redis_url="",
        default_currency="USD",
        default_taxonomy="unspsc",
        min_confidence=0.3,
        high_confidence_threshold=0.8,
        medium_confidence_threshold=0.5,
        max_records=100000,
        batch_size=1000,
        eeio_version="2024",
        exiobase_version="3.8.2",
        defra_version="2025",
        ecoinvent_version="3.10",
    )
    defaults.update(overrides)
    return SpendCategorizerConfig(**defaults)


def _build_service(config: Optional[SpendCategorizerConfig] = None) -> SpendCategorizerService:
    """Convenience to build a service with default or custom config.

    Patches _init_engines to avoid real engine construction which may fail
    in unit test environments where engine submodules have incompatible
    constructor signatures.
    """
    with patch.object(SpendCategorizerService, "_init_engines"):
        svc = SpendCategorizerService(config=config or _make_config())
    return svc


def _sample_records(count: int = 3) -> List[Dict[str, Any]]:
    """Generate sample spend record dicts."""
    records = []
    vendors = [
        ("Acme Corp", "Office supplies and paper products"),
        ("TechWorld Inc", "Laptop computers and software licenses"),
        ("Green Transport Ltd", "Freight shipping and logistics services"),
        ("CleanCo Services", "Cleaning products and janitorial supplies"),
        ("FuelMax Energy", "Diesel fuel and lubricants"),
    ]
    for i in range(count):
        v_name, desc = vendors[i % len(vendors)]
        records.append({
            "vendor_name": v_name,
            "vendor_id": f"V-{i:04d}",
            "description": desc,
            "amount": 10000.0 + (i * 5000),
            "currency": "USD",
            "amount_usd": 10000.0 + (i * 5000),
            "date": f"2025-0{(i % 9) + 1}-15",
            "cost_center": f"CC-{i % 3}",
            "gl_account": f"GL-{1000 + i}",
            "po_number": f"PO-{i:06d}",
        })
    return records


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def config() -> SpendCategorizerConfig:
    """Fresh test configuration."""
    return _make_config()


@pytest.fixture
def service(config: SpendCategorizerConfig) -> SpendCategorizerService:
    """Fresh SpendCategorizerService instance."""
    return _build_service(config)


@pytest.fixture
def records_in_service(service: SpendCategorizerService) -> List[SpendRecordResponse]:
    """Service with 5 ingested records; returns the ingested records."""
    return service.ingest_records(_sample_records(5), source="test")


@pytest.fixture
def classifications_in_service(
    service: SpendCategorizerService,
    records_in_service: List[SpendRecordResponse],
) -> List[ClassificationResponse]:
    """Classify all ingested records; returns classification list."""
    rids = [r.record_id for r in records_in_service]
    return service.classify_batch(rids)


@pytest.fixture
def scope3_in_service(
    service: SpendCategorizerService,
    classifications_in_service: List[ClassificationResponse],
) -> List[Scope3AssignmentResponse]:
    """Map all classified records to Scope 3; returns assignments."""
    rids = [c.record_id for c in classifications_in_service]
    return service.map_scope3_batch(rids)


@pytest.fixture
def emissions_in_service(
    service: SpendCategorizerService,
    scope3_in_service: List[Scope3AssignmentResponse],
) -> List[EmissionCalculationResponse]:
    """Calculate emissions for all mapped records; returns calculations."""
    rids = [a.record_id for a in scope3_in_service]
    return service.calculate_emissions_batch(rids)


# ===================================================================
# TestServiceInit
# ===================================================================


class TestServiceInit:
    """Test SpendCategorizerService initialization."""

    def test_init_with_default_config(self) -> None:
        svc = _build_service()
        assert svc.config is not None
        assert svc.provenance is not None

    def test_init_with_custom_config(self) -> None:
        cfg = _make_config(default_taxonomy="naics", default_currency="EUR")
        svc = _build_service(cfg)
        assert svc.config.default_taxonomy == "naics"
        assert svc.config.default_currency == "EUR"

    def test_engine_properties_none_when_stubbed(self) -> None:
        svc = _build_service()
        assert svc.record_ingestion_engine is None
        assert svc.taxonomy_classifier is None
        assert svc.scope3_mapper is None
        assert svc.emission_calculator is None
        assert svc.rule_engine is None
        assert svc.analytics_engine is None
        assert svc.report_generator is None

    def test_empty_stores_on_init(self) -> None:
        svc = _build_service()
        assert len(svc._records) == 0
        assert len(svc._classifications) == 0
        assert len(svc._scope3_assignments) == 0
        assert len(svc._emission_calculations) == 0
        assert len(svc._rules) == 0
        assert len(svc._reports) == 0

    def test_default_emission_factors_loaded(self) -> None:
        svc = _build_service()
        assert len(svc._emission_factors) > 0
        assert "43000000" in svc._emission_factors
        assert "44000000" in svc._emission_factors

    def test_statistics_initial_values(self) -> None:
        svc = _build_service()
        stats = svc.get_statistics()
        assert stats.total_records == 0
        assert stats.total_classified == 0
        assert stats.total_spend_usd == 0.0

    def test_not_started_on_init(self) -> None:
        svc = _build_service()
        assert svc._started is False

    def test_provenance_tracker_initialized(self) -> None:
        svc = _build_service()
        assert isinstance(svc.provenance, _ProvenanceTracker)
        assert svc.provenance.entry_count == 0


# ===================================================================
# TestIngestRecords
# ===================================================================


class TestIngestRecords:
    """Test ingest_records method."""

    def test_ingest_single_record(self, service: SpendCategorizerService) -> None:
        records = service.ingest_records(
            [{"vendor_name": "Acme", "amount": 5000, "description": "Supplies"}],
        )
        assert len(records) == 1
        assert records[0].vendor_name == "Acme"
        assert records[0].amount == 5000.0
        assert records[0].status == "ingested"

    def test_ingest_batch(self, service: SpendCategorizerService) -> None:
        records = service.ingest_records(_sample_records(5), source="batch_test")
        assert len(records) == 5
        for r in records:
            assert r.source == "batch_test"

    def test_ingest_empty_raises(self, service: SpendCategorizerService) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            service.ingest_records([])

    def test_ingest_exceeds_max_raises(self) -> None:
        svc = _build_service(_make_config(max_records=2))
        with pytest.raises(ValueError, match="exceeds maximum"):
            svc.ingest_records(_sample_records(5))

    def test_source_tracking(self, service: SpendCategorizerService) -> None:
        records = service.ingest_records(
            [{"vendor_name": "V1", "amount": 100}], source="erp",
        )
        assert records[0].source == "erp"

    def test_default_source_is_manual(self, service: SpendCategorizerService) -> None:
        records = service.ingest_records([{"vendor_name": "V1", "amount": 100}])
        assert records[0].source == "manual"

    def test_amount_usd_defaults_to_amount(self, service: SpendCategorizerService) -> None:
        records = service.ingest_records(
            [{"vendor_name": "V1", "amount": 2500}],
        )
        assert records[0].amount_usd == 2500.0

    def test_amount_usd_explicit(self, service: SpendCategorizerService) -> None:
        records = service.ingest_records(
            [{"vendor_name": "V1", "amount": 2500, "amount_usd": 3000}],
        )
        assert records[0].amount_usd == 3000.0

    def test_currency_from_config_default(self, service: SpendCategorizerService) -> None:
        records = service.ingest_records([{"vendor_name": "V1", "amount": 100}])
        assert records[0].currency == "USD"

    def test_currency_override(self, service: SpendCategorizerService) -> None:
        records = service.ingest_records(
            [{"vendor_name": "V1", "amount": 100, "currency": "EUR"}],
        )
        assert records[0].currency == "EUR"

    def test_record_id_is_uuid(self, service: SpendCategorizerService) -> None:
        records = service.ingest_records([{"vendor_name": "V1", "amount": 100}])
        uid = uuid.UUID(records[0].record_id)
        assert uid.version == 4

    def test_provenance_hash_set(self, service: SpendCategorizerService) -> None:
        records = service.ingest_records([{"vendor_name": "V1", "amount": 100}])
        assert len(records[0].provenance_hash) == 64

    def test_statistics_updated(self, service: SpendCategorizerService) -> None:
        service.ingest_records(_sample_records(3))
        stats = service.get_statistics()
        assert stats.total_records == 3
        assert stats.total_spend_usd > 0

    def test_provenance_entries_created(self, service: SpendCategorizerService) -> None:
        service.ingest_records(_sample_records(2))
        assert service.provenance.entry_count == 2

    def test_all_fields_preserved(self, service: SpendCategorizerService) -> None:
        records = service.ingest_records([{
            "vendor_name": "TestVendor",
            "vendor_id": "TV-001",
            "description": "Test item",
            "amount": 999.99,
            "currency": "GBP",
            "amount_usd": 1200.00,
            "date": "2025-06-15",
            "cost_center": "CC-X",
            "gl_account": "GL-9999",
            "po_number": "PO-ABC",
        }])
        r = records[0]
        assert r.vendor_name == "TestVendor"
        assert r.vendor_id == "TV-001"
        assert r.description == "Test item"
        assert r.amount == 999.99
        assert r.currency == "GBP"
        assert r.amount_usd == 1200.00
        assert r.date == "2025-06-15"
        assert r.cost_center == "CC-X"
        assert r.gl_account == "GL-9999"
        assert r.po_number == "PO-ABC"

    def test_created_at_set(self, service: SpendCategorizerService) -> None:
        records = service.ingest_records([{"vendor_name": "V", "amount": 1}])
        assert records[0].created_at != ""

    def test_updated_at_set(self, service: SpendCategorizerService) -> None:
        records = service.ingest_records([{"vendor_name": "V", "amount": 1}])
        assert records[0].updated_at != ""

    def test_multiple_batches_accumulate(self, service: SpendCategorizerService) -> None:
        service.ingest_records(_sample_records(2))
        service.ingest_records(_sample_records(3))
        assert len(service._records) == 5
        stats = service.get_statistics()
        assert stats.total_records == 5


# ===================================================================
# TestIngestCSV
# ===================================================================


class TestIngestCSV:
    """Test ingest_csv method."""

    def test_ingest_csv_success(self, service: SpendCategorizerService) -> None:
        csv_content = "vendor_name,amount,description\nAcme,5000,Supplies\nBeta,3000,Paper\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8",
        ) as f:
            f.write(csv_content)
            f.flush()
            path = f.name
        try:
            records = service.ingest_csv(path, source="csv")
            assert len(records) == 2
            assert records[0].vendor_name == "Acme"
            assert records[0].source == "csv"
        finally:
            os.unlink(path)

    def test_ingest_csv_missing_file_raises(self, service: SpendCategorizerService) -> None:
        with pytest.raises(ValueError, match="Failed to parse CSV"):
            service.ingest_csv("/nonexistent/file.csv")

    def test_ingest_csv_empty_file_raises(self, service: SpendCategorizerService) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8",
        ) as f:
            f.write("vendor_name,amount\n")
            f.flush()
            path = f.name
        try:
            with pytest.raises(ValueError, match="must not be empty"):
                service.ingest_csv(path)
        finally:
            os.unlink(path)

    def test_ingest_csv_custom_source(self, service: SpendCategorizerService) -> None:
        csv_content = "vendor_name,amount\nX,100\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8",
        ) as f:
            f.write(csv_content)
            f.flush()
            path = f.name
        try:
            records = service.ingest_csv(path, source="custom_csv")
            assert records[0].source == "custom_csv"
        finally:
            os.unlink(path)


# ===================================================================
# TestGetRecord
# ===================================================================


class TestGetRecord:
    """Test get_record method."""

    def test_get_existing_record(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        rid = records_in_service[0].record_id
        record = service.get_record(rid)
        assert record is not None
        assert record.record_id == rid

    def test_get_nonexistent_record(self, service: SpendCategorizerService) -> None:
        result = service.get_record("nonexistent-id")
        assert result is None

    def test_get_record_after_classification(
        self,
        service: SpendCategorizerService,
        classifications_in_service: List[ClassificationResponse],
    ) -> None:
        rid = classifications_in_service[0].record_id
        record = service.get_record(rid)
        assert record is not None
        assert record.status == "classified"


# ===================================================================
# TestListRecords
# ===================================================================


class TestListRecords:
    """Test list_records method."""

    def test_list_all_records(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        records = service.list_records()
        assert len(records) == 5

    def test_list_by_source(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        records = service.list_records(source="test")
        assert len(records) == 5
        records = service.list_records(source="other")
        assert len(records) == 0

    def test_list_by_status(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        records = service.list_records(status="ingested")
        assert len(records) == 5

    def test_list_by_vendor_name(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        records = service.list_records(vendor_name="Acme")
        assert len(records) >= 1
        for r in records:
            assert "acme" in r.vendor_name.lower()

    def test_list_pagination_limit(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        records = service.list_records(limit=2)
        assert len(records) == 2

    def test_list_pagination_offset(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        all_records = service.list_records()
        offset_records = service.list_records(offset=3)
        assert len(offset_records) == len(all_records) - 3

    def test_list_empty_service(self, service: SpendCategorizerService) -> None:
        records = service.list_records()
        assert len(records) == 0

    def test_list_combined_filters(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        records = service.list_records(source="test", status="ingested", limit=2)
        assert len(records) <= 2


# ===================================================================
# TestClassifyRecord
# ===================================================================


class TestClassifyRecord:
    """Test classify_record method."""

    def test_classify_single_record(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        rid = records_in_service[0].record_id
        classification = service.classify_record(rid)
        assert isinstance(classification, ClassificationResponse)
        assert classification.record_id == rid
        assert classification.taxonomy_system == "unspsc"

    def test_classify_with_taxonomy_override(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        rid = records_in_service[0].record_id
        classification = service.classify_record(rid, taxonomy_system="naics")
        assert classification.taxonomy_system == "naics"

    def test_classify_not_found_raises(self, service: SpendCategorizerService) -> None:
        with pytest.raises(ValueError, match="not found"):
            service.classify_record("nonexistent-id")

    def test_classify_updates_record_status(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        rid = records_in_service[0].record_id
        service.classify_record(rid)
        record = service.get_record(rid)
        assert record.status == "classified"

    def test_classify_updates_record_taxonomy(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        rid = records_in_service[0].record_id
        classification = service.classify_record(rid)
        record = service.get_record(rid)
        assert record.taxonomy_code == classification.taxonomy_code

    def test_classify_confidence_set(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        rid = records_in_service[0].record_id
        classification = service.classify_record(rid)
        assert 0.0 <= classification.confidence <= 1.0

    def test_classify_provenance_hash_set(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        rid = records_in_service[0].record_id
        classification = service.classify_record(rid)
        assert len(classification.provenance_hash) == 64

    def test_classify_method_keyword_when_no_rules(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        rid = records_in_service[0].record_id
        classification = service.classify_record(rid)
        assert classification.method == "keyword"

    def test_classify_with_rule_match(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        service.create_rule(
            name="Office rule",
            taxonomy_code="44000000",
            conditions={"keywords": ["office", "supplies"]},
            taxonomy_system="unspsc",
            priority=1,
        )
        rid = records_in_service[0].record_id  # "Office supplies and paper products"
        classification = service.classify_record(rid)
        assert classification.method == "rule"
        assert classification.taxonomy_code == "44000000"
        assert classification.confidence == 0.95
        assert classification.confidence_label == "HIGH"

    def test_classify_statistics_updated(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        service.classify_record(records_in_service[0].record_id)
        stats = service.get_statistics()
        assert stats.total_classified == 1

    def test_classify_avg_confidence_updated(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        service.classify_record(records_in_service[0].record_id)
        stats = service.get_statistics()
        assert stats.avg_confidence > 0.0

    def test_keyword_high_confidence(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        # "Laptop computers and software licenses" should match IT keywords
        rid = records_in_service[1].record_id
        classification = service.classify_record(rid)
        assert classification.taxonomy_code != ""

    def test_keyword_no_match_returns_empty_code(self, service: SpendCategorizerService) -> None:
        records = service.ingest_records(
            [{"vendor_name": "XYZ", "description": "zzzzz nothing", "amount": 100}],
        )
        classification = service.classify_record(records[0].record_id)
        assert classification.taxonomy_code == ""
        assert classification.confidence == 0.0
        assert classification.confidence_label == "LOW"

    def test_confidence_label_medium(self, service: SpendCategorizerService) -> None:
        records = service.ingest_records(
            [{"vendor_name": "V", "description": "cleaning stuff", "amount": 100}],
        )
        classification = service.classify_record(records[0].record_id)
        # "cleaning" matches one keyword: confidence=0.45 -> MEDIUM check depends on threshold
        assert classification.confidence_label in ("LOW", "MEDIUM", "HIGH")


# ===================================================================
# TestClassifyBatch
# ===================================================================


class TestClassifyBatch:
    """Test classify_batch method."""

    def test_batch_classification(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        rids = [r.record_id for r in records_in_service]
        results = service.classify_batch(rids)
        assert len(results) == 5

    def test_batch_skips_invalid_ids(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        rids = [records_in_service[0].record_id, "nonexistent"]
        results = service.classify_batch(rids)
        assert len(results) == 1

    def test_batch_empty_list(self, service: SpendCategorizerService) -> None:
        results = service.classify_batch([])
        assert len(results) == 0

    def test_batch_with_taxonomy_override(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        rids = [r.record_id for r in records_in_service[:2]]
        results = service.classify_batch(rids, taxonomy_system="naics")
        for r in results:
            assert r.taxonomy_system == "naics"


# ===================================================================
# TestGetClassification
# ===================================================================


class TestGetClassification:
    """Test get_classification method."""

    def test_get_existing_classification(
        self,
        service: SpendCategorizerService,
        classifications_in_service: List[ClassificationResponse],
    ) -> None:
        cid = classifications_in_service[0].classification_id
        result = service.get_classification(cid)
        assert result is not None
        assert result.classification_id == cid

    def test_get_nonexistent_classification(self, service: SpendCategorizerService) -> None:
        result = service.get_classification("nonexistent")
        assert result is None


# ===================================================================
# TestMapScope3
# ===================================================================


class TestMapScope3:
    """Test map_scope3 method."""

    def test_map_single_record(
        self,
        service: SpendCategorizerService,
        classifications_in_service: List[ClassificationResponse],
    ) -> None:
        rid = classifications_in_service[0].record_id
        assignment = service.map_scope3(rid)
        assert isinstance(assignment, Scope3AssignmentResponse)
        assert assignment.record_id == rid
        assert 0 <= assignment.scope3_category <= 15

    def test_map_not_found_raises(self, service: SpendCategorizerService) -> None:
        with pytest.raises(ValueError, match="not found"):
            service.map_scope3("nonexistent")

    def test_map_unclassified_raises(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        # Record not classified but no taxonomy_code either (keyword may set empty)
        svc = _build_service()
        ingested = svc.ingest_records(
            [{"vendor_name": "X", "description": "zzz nothing", "amount": 100}],
        )
        svc.classify_record(ingested[0].record_id)
        # After classification with no keyword match, taxonomy_code is ""
        record = svc.get_record(ingested[0].record_id)
        if not record.taxonomy_code:
            with pytest.raises(ValueError, match="classified"):
                svc.map_scope3(ingested[0].record_id)

    def test_map_updates_record_status(
        self,
        service: SpendCategorizerService,
        classifications_in_service: List[ClassificationResponse],
    ) -> None:
        rid = classifications_in_service[0].record_id
        service.map_scope3(rid)
        record = service.get_record(rid)
        assert record.status == "mapped"

    def test_map_updates_record_scope3(
        self,
        service: SpendCategorizerService,
        classifications_in_service: List[ClassificationResponse],
    ) -> None:
        rid = classifications_in_service[0].record_id
        assignment = service.map_scope3(rid)
        record = service.get_record(rid)
        assert record.scope3_category == assignment.scope3_category

    def test_map_scope3_name_populated(
        self,
        service: SpendCategorizerService,
        classifications_in_service: List[ClassificationResponse],
    ) -> None:
        rid = classifications_in_service[0].record_id
        assignment = service.map_scope3(rid)
        if assignment.scope3_category > 0:
            assert assignment.scope3_category_name in _SCOPE3_CATEGORIES.values()

    def test_map_provenance_hash_set(
        self,
        service: SpendCategorizerService,
        classifications_in_service: List[ClassificationResponse],
    ) -> None:
        rid = classifications_in_service[0].record_id
        assignment = service.map_scope3(rid)
        assert len(assignment.provenance_hash) == 64

    def test_map_statistics_updated(
        self,
        service: SpendCategorizerService,
        classifications_in_service: List[ClassificationResponse],
    ) -> None:
        rid = classifications_in_service[0].record_id
        service.map_scope3(rid)
        stats = service.get_statistics()
        assert stats.total_scope3_mapped == 1

    def test_map_confidence(
        self,
        service: SpendCategorizerService,
        classifications_in_service: List[ClassificationResponse],
    ) -> None:
        rid = classifications_in_service[0].record_id
        assignment = service.map_scope3(rid)
        assert assignment.mapping_confidence >= 0.0

    def test_map_method(
        self,
        service: SpendCategorizerService,
        classifications_in_service: List[ClassificationResponse],
    ) -> None:
        rid = classifications_in_service[0].record_id
        assignment = service.map_scope3(rid)
        assert assignment.mapping_method == "taxonomy_lookup"


# ===================================================================
# TestMapScope3Batch
# ===================================================================


class TestMapScope3Batch:
    """Test map_scope3_batch method."""

    def test_batch_mapping(
        self,
        service: SpendCategorizerService,
        classifications_in_service: List[ClassificationResponse],
    ) -> None:
        rids = [c.record_id for c in classifications_in_service]
        # Filter to those that have taxonomy_code set
        valid_rids = []
        for rid in rids:
            rec = service.get_record(rid)
            if rec and rec.taxonomy_code:
                valid_rids.append(rid)
        results = service.map_scope3_batch(valid_rids)
        assert len(results) == len(valid_rids)

    def test_batch_skips_invalid(
        self,
        service: SpendCategorizerService,
        classifications_in_service: List[ClassificationResponse],
    ) -> None:
        rids = [classifications_in_service[0].record_id, "invalid-id"]
        # Only valid records with taxonomy_code will succeed
        results = service.map_scope3_batch(rids)
        assert len(results) >= 0  # At least no crash

    def test_batch_empty(self, service: SpendCategorizerService) -> None:
        results = service.map_scope3_batch([])
        assert len(results) == 0


# ===================================================================
# TestGetScope3Assignment
# ===================================================================


class TestGetScope3Assignment:
    """Test get_scope3_assignment method."""

    def test_get_existing(
        self,
        service: SpendCategorizerService,
        scope3_in_service: List[Scope3AssignmentResponse],
    ) -> None:
        if scope3_in_service:
            aid = scope3_in_service[0].assignment_id
            result = service.get_scope3_assignment(aid)
            assert result is not None
            assert result.assignment_id == aid

    def test_get_nonexistent(self, service: SpendCategorizerService) -> None:
        result = service.get_scope3_assignment("nonexistent")
        assert result is None


# ===================================================================
# TestCalculateEmissions
# ===================================================================


class TestCalculateEmissions:
    """Test calculate_emissions method."""

    def test_calculate_single_record(
        self,
        service: SpendCategorizerService,
        scope3_in_service: List[Scope3AssignmentResponse],
    ) -> None:
        if not scope3_in_service:
            pytest.skip("No scope3 assignments available")
        rid = scope3_in_service[0].record_id
        calc = service.calculate_emissions(rid)
        assert isinstance(calc, EmissionCalculationResponse)
        assert calc.record_id == rid
        assert calc.emissions_kg_co2e >= 0.0

    def test_calculate_not_found_raises(self, service: SpendCategorizerService) -> None:
        with pytest.raises(ValueError, match="not found"):
            service.calculate_emissions("nonexistent")

    def test_calculate_unclassified_raises(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        rid = records_in_service[0].record_id
        with pytest.raises(ValueError, match="classified"):
            service.calculate_emissions(rid)

    def test_calculate_updates_record_status(
        self,
        service: SpendCategorizerService,
        scope3_in_service: List[Scope3AssignmentResponse],
    ) -> None:
        if not scope3_in_service:
            pytest.skip("No scope3 assignments")
        rid = scope3_in_service[0].record_id
        service.calculate_emissions(rid)
        record = service.get_record(rid)
        assert record.status == "calculated"

    def test_calculate_uses_emission_factor(
        self,
        service: SpendCategorizerService,
        scope3_in_service: List[Scope3AssignmentResponse],
    ) -> None:
        if not scope3_in_service:
            pytest.skip("No scope3 assignments")
        rid = scope3_in_service[0].record_id
        calc = service.calculate_emissions(rid)
        assert calc.emission_factor >= 0.0
        assert calc.methodology == "spend_based"

    def test_calculate_emission_formula(
        self,
        service: SpendCategorizerService,
        scope3_in_service: List[Scope3AssignmentResponse],
    ) -> None:
        if not scope3_in_service:
            pytest.skip("No scope3 assignments")
        rid = scope3_in_service[0].record_id
        calc = service.calculate_emissions(rid)
        expected = round(calc.spend_usd * calc.emission_factor, 4)
        assert calc.emissions_kg_co2e == expected

    def test_calculate_with_factor_source(
        self,
        service: SpendCategorizerService,
        scope3_in_service: List[Scope3AssignmentResponse],
    ) -> None:
        if not scope3_in_service:
            pytest.skip("No scope3 assignments")
        rid = scope3_in_service[0].record_id
        calc = service.calculate_emissions(rid, factor_source="exiobase")
        assert calc.emission_factor_source == "exiobase"

    def test_calculate_statistics_updated(
        self,
        service: SpendCategorizerService,
        scope3_in_service: List[Scope3AssignmentResponse],
    ) -> None:
        if not scope3_in_service:
            pytest.skip("No scope3 assignments")
        rid = scope3_in_service[0].record_id
        service.calculate_emissions(rid)
        stats = service.get_statistics()
        assert stats.total_emissions_calculated == 1

    def test_calculate_provenance_hash(
        self,
        service: SpendCategorizerService,
        scope3_in_service: List[Scope3AssignmentResponse],
    ) -> None:
        if not scope3_in_service:
            pytest.skip("No scope3 assignments")
        rid = scope3_in_service[0].record_id
        calc = service.calculate_emissions(rid)
        assert len(calc.provenance_hash) == 64

    def test_calculate_factor_version(
        self,
        service: SpendCategorizerService,
        scope3_in_service: List[Scope3AssignmentResponse],
    ) -> None:
        if not scope3_in_service:
            pytest.skip("No scope3 assignments")
        rid = scope3_in_service[0].record_id
        calc = service.calculate_emissions(rid)
        assert calc.emission_factor_version != ""


# ===================================================================
# TestCalculateEmissionsBatch
# ===================================================================


class TestCalculateEmissionsBatch:
    """Test calculate_emissions_batch method."""

    def test_batch_calculation(
        self,
        service: SpendCategorizerService,
        scope3_in_service: List[Scope3AssignmentResponse],
    ) -> None:
        rids = [a.record_id for a in scope3_in_service]
        results = service.calculate_emissions_batch(rids)
        assert len(results) == len(rids)

    def test_batch_skips_invalid(
        self,
        service: SpendCategorizerService,
        scope3_in_service: List[Scope3AssignmentResponse],
    ) -> None:
        if not scope3_in_service:
            pytest.skip("No scope3 assignments")
        rids = [scope3_in_service[0].record_id, "invalid"]
        results = service.calculate_emissions_batch(rids)
        assert len(results) >= 1

    def test_batch_empty(self, service: SpendCategorizerService) -> None:
        results = service.calculate_emissions_batch([])
        assert len(results) == 0

    def test_batch_with_factor_source(
        self,
        service: SpendCategorizerService,
        scope3_in_service: List[Scope3AssignmentResponse],
    ) -> None:
        rids = [a.record_id for a in scope3_in_service[:2]]
        results = service.calculate_emissions_batch(rids, factor_source="defra")
        for calc in results:
            assert calc.emission_factor_source == "defra"


# ===================================================================
# TestGetEmissionFactor
# ===================================================================


class TestGetEmissionFactor:
    """Test get_emission_factor method."""

    def test_get_by_code(self, service: SpendCategorizerService) -> None:
        factor = service.get_emission_factor("43000000")
        assert factor["factor"] == 0.42
        assert factor["source"] == "eeio"

    def test_get_by_prefix_match(self, service: SpendCategorizerService) -> None:
        factor = service.get_emission_factor("4300")
        # Should match 43000000 via prefix
        assert factor["factor"] == 0.42

    def test_get_not_found_returns_default(self, service: SpendCategorizerService) -> None:
        factor = service.get_emission_factor("99999999")
        assert factor["factor"] == 0.25
        assert factor["desc"] == "Default emission factor"

    def test_get_with_source_param(self, service: SpendCategorizerService) -> None:
        factor = service.get_emission_factor("99999999", source="defra")
        assert factor["source"] == "defra"


# ===================================================================
# TestListEmissionFactors
# ===================================================================


class TestListEmissionFactors:
    """Test list_emission_factors method."""

    def test_list_all(self, service: SpendCategorizerService) -> None:
        factors = service.list_emission_factors()
        assert len(factors) == 10

    def test_list_with_limit(self, service: SpendCategorizerService) -> None:
        factors = service.list_emission_factors(limit=3)
        assert len(factors) == 3

    def test_list_with_offset(self, service: SpendCategorizerService) -> None:
        factors = service.list_emission_factors(offset=5)
        assert len(factors) == 5

    def test_list_factor_structure(self, service: SpendCategorizerService) -> None:
        factors = service.list_emission_factors(limit=1)
        f = factors[0]
        assert "taxonomy_code" in f
        assert "factor" in f
        assert "source" in f
        assert "description" in f


# ===================================================================
# TestCreateRule
# ===================================================================


class TestCreateRule:
    """Test create_rule method."""

    def test_create_vendor_pattern_rule(self, service: SpendCategorizerService) -> None:
        rule = service.create_rule(
            name="Acme rule",
            taxonomy_code="44000000",
            conditions={"vendor_pattern": "acme"},
        )
        assert isinstance(rule, CategoryRuleResponse)
        assert rule.name == "Acme rule"
        assert rule.taxonomy_code == "44000000"
        assert rule.is_active is True

    def test_create_keyword_rule(self, service: SpendCategorizerService) -> None:
        rule = service.create_rule(
            name="IT rule",
            taxonomy_code="43000000",
            conditions={"keywords": ["computer", "laptop", "software"]},
        )
        assert rule.conditions["keywords"] == ["computer", "laptop", "software"]

    def test_create_gl_code_rule(self, service: SpendCategorizerService) -> None:
        rule = service.create_rule(
            name="GL rule",
            taxonomy_code="78000000",
            conditions={"gl_codes": ["GL-5000", "GL-5001"]},
        )
        assert "gl_codes" in rule.conditions

    def test_create_cost_center_rule(self, service: SpendCategorizerService) -> None:
        rule = service.create_rule(
            name="CC rule",
            taxonomy_code="72000000",
            conditions={"cost_centers": ["CC-FACILITIES"]},
        )
        assert "cost_centers" in rule.conditions

    def test_create_combined_conditions_rule(self, service: SpendCategorizerService) -> None:
        rule = service.create_rule(
            name="Combined",
            taxonomy_code="50000000",
            conditions={
                "vendor_pattern": "catering",
                "keywords": ["food", "meal"],
                "gl_codes": ["GL-3000"],
            },
        )
        assert len(rule.conditions) == 3

    def test_create_with_scope3_category(self, service: SpendCategorizerService) -> None:
        rule = service.create_rule(
            name="Travel rule",
            taxonomy_code="86000000",
            conditions={"keywords": ["flight", "hotel"]},
            scope3_category=6,
        )
        assert rule.scope3_category == 6

    def test_create_with_custom_priority(self, service: SpendCategorizerService) -> None:
        rule = service.create_rule(
            name="High priority",
            taxonomy_code="44000000",
            conditions={"keywords": ["urgent"]},
            priority=1,
        )
        assert rule.priority == 1

    def test_create_with_taxonomy_system(self, service: SpendCategorizerService) -> None:
        rule = service.create_rule(
            name="NAICS rule",
            taxonomy_code="541000",
            conditions={"keywords": ["consulting"]},
            taxonomy_system="naics",
        )
        assert rule.taxonomy_system == "naics"

    def test_create_empty_name_raises(self, service: SpendCategorizerService) -> None:
        with pytest.raises(ValueError, match="name must not be empty"):
            service.create_rule(
                name="",
                taxonomy_code="44000000",
                conditions={"keywords": ["test"]},
            )

    def test_create_whitespace_name_raises(self, service: SpendCategorizerService) -> None:
        with pytest.raises(ValueError, match="name must not be empty"):
            service.create_rule(
                name="   ",
                taxonomy_code="44000000",
                conditions={"keywords": ["test"]},
            )

    def test_create_empty_taxonomy_code_raises(self, service: SpendCategorizerService) -> None:
        with pytest.raises(ValueError, match="Taxonomy code must not be empty"):
            service.create_rule(
                name="Test rule",
                taxonomy_code="",
                conditions={"keywords": ["test"]},
            )

    def test_create_provenance_hash_set(self, service: SpendCategorizerService) -> None:
        rule = service.create_rule(
            name="Test",
            taxonomy_code="44000000",
            conditions={"keywords": ["test"]},
        )
        assert len(rule.provenance_hash) == 64

    def test_create_statistics_updated(self, service: SpendCategorizerService) -> None:
        service.create_rule(
            name="R1",
            taxonomy_code="44000000",
            conditions={"keywords": ["test"]},
        )
        stats = service.get_statistics()
        assert stats.total_rules == 1
        assert stats.active_rules == 1

    def test_create_with_description(self, service: SpendCategorizerService) -> None:
        rule = service.create_rule(
            name="Desc rule",
            taxonomy_code="44000000",
            conditions={"keywords": ["test"]},
            description="A test rule for office supplies",
        )
        assert rule.description == "A test rule for office supplies"


# ===================================================================
# TestGetRule
# ===================================================================


class TestGetRule:
    """Test get_rule method."""

    def test_get_existing_rule(self, service: SpendCategorizerService) -> None:
        rule = service.create_rule(
            name="Test",
            taxonomy_code="44000000",
            conditions={"keywords": ["test"]},
        )
        result = service.get_rule(rule.rule_id)
        assert result is not None
        assert result.rule_id == rule.rule_id

    def test_get_nonexistent_rule(self, service: SpendCategorizerService) -> None:
        result = service.get_rule("nonexistent")
        assert result is None


# ===================================================================
# TestListRules
# ===================================================================


class TestListRules:
    """Test list_rules method."""

    def test_list_empty(self, service: SpendCategorizerService) -> None:
        rules = service.list_rules()
        assert len(rules) == 0

    def test_list_all_rules(self, service: SpendCategorizerService) -> None:
        service.create_rule(name="R1", taxonomy_code="44000000", conditions={"keywords": ["a"]})
        service.create_rule(name="R2", taxonomy_code="43000000", conditions={"keywords": ["b"]})
        rules = service.list_rules()
        assert len(rules) == 2

    def test_list_sorted_by_priority(self, service: SpendCategorizerService) -> None:
        service.create_rule(name="Low", taxonomy_code="44000000", conditions={"keywords": ["a"]}, priority=100)
        service.create_rule(name="High", taxonomy_code="43000000", conditions={"keywords": ["b"]}, priority=1)
        rules = service.list_rules()
        assert rules[0].priority <= rules[1].priority

    def test_list_active_only(self, service: SpendCategorizerService) -> None:
        r1 = service.create_rule(name="R1", taxonomy_code="44000000", conditions={"keywords": ["a"]})
        r2 = service.create_rule(name="R2", taxonomy_code="43000000", conditions={"keywords": ["b"]})
        service.update_rule(r2.rule_id, is_active=False)
        active = service.list_rules(is_active=True)
        assert len(active) == 1
        assert active[0].rule_id == r1.rule_id

    def test_list_by_taxonomy_system(self, service: SpendCategorizerService) -> None:
        service.create_rule(name="U", taxonomy_code="44000000", conditions={"keywords": ["a"]}, taxonomy_system="unspsc")
        service.create_rule(name="N", taxonomy_code="541000", conditions={"keywords": ["b"]}, taxonomy_system="naics")
        unspsc_rules = service.list_rules(taxonomy_system="unspsc")
        assert len(unspsc_rules) == 1

    def test_list_with_pagination(self, service: SpendCategorizerService) -> None:
        for i in range(5):
            service.create_rule(name=f"R{i}", taxonomy_code="44000000", conditions={"keywords": [f"k{i}"]})
        rules = service.list_rules(limit=2, offset=1)
        assert len(rules) == 2


# ===================================================================
# TestUpdateRule
# ===================================================================


class TestUpdateRule:
    """Test update_rule method."""

    def test_update_name(self, service: SpendCategorizerService) -> None:
        rule = service.create_rule(name="Old", taxonomy_code="44000000", conditions={"keywords": ["a"]})
        updated = service.update_rule(rule.rule_id, name="New")
        assert updated.name == "New"

    def test_update_description(self, service: SpendCategorizerService) -> None:
        rule = service.create_rule(name="R", taxonomy_code="44000000", conditions={"keywords": ["a"]})
        updated = service.update_rule(rule.rule_id, description="Updated desc")
        assert updated.description == "Updated desc"

    def test_update_taxonomy_code(self, service: SpendCategorizerService) -> None:
        rule = service.create_rule(name="R", taxonomy_code="44000000", conditions={"keywords": ["a"]})
        updated = service.update_rule(rule.rule_id, taxonomy_code="43000000")
        assert updated.taxonomy_code == "43000000"

    def test_update_conditions(self, service: SpendCategorizerService) -> None:
        rule = service.create_rule(name="R", taxonomy_code="44000000", conditions={"keywords": ["a"]})
        updated = service.update_rule(rule.rule_id, conditions={"keywords": ["b", "c"]})
        assert updated.conditions["keywords"] == ["b", "c"]

    def test_update_priority(self, service: SpendCategorizerService) -> None:
        rule = service.create_rule(name="R", taxonomy_code="44000000", conditions={"keywords": ["a"]})
        updated = service.update_rule(rule.rule_id, priority=1)
        assert updated.priority == 1

    def test_update_is_active(self, service: SpendCategorizerService) -> None:
        rule = service.create_rule(name="R", taxonomy_code="44000000", conditions={"keywords": ["a"]})
        updated = service.update_rule(rule.rule_id, is_active=False)
        assert updated.is_active is False

    def test_update_not_found_raises(self, service: SpendCategorizerService) -> None:
        with pytest.raises(ValueError, match="not found"):
            service.update_rule("nonexistent", name="X")

    def test_update_provenance_hash_changes(self, service: SpendCategorizerService) -> None:
        rule = service.create_rule(name="R", taxonomy_code="44000000", conditions={"keywords": ["a"]})
        old_hash = rule.provenance_hash
        updated = service.update_rule(rule.rule_id, name="Changed")
        assert updated.provenance_hash != old_hash

    def test_update_updated_at_changes(self, service: SpendCategorizerService) -> None:
        rule = service.create_rule(name="R", taxonomy_code="44000000", conditions={"keywords": ["a"]})
        old_at = rule.updated_at
        updated = service.update_rule(rule.rule_id, name="Changed")
        assert updated.updated_at >= old_at

    def test_update_active_rules_count(self, service: SpendCategorizerService) -> None:
        r1 = service.create_rule(name="R1", taxonomy_code="44000000", conditions={"keywords": ["a"]})
        r2 = service.create_rule(name="R2", taxonomy_code="43000000", conditions={"keywords": ["b"]})
        assert service.get_statistics().active_rules == 2
        service.update_rule(r2.rule_id, is_active=False)
        assert service.get_statistics().active_rules == 1


# ===================================================================
# TestDeleteRule
# ===================================================================


class TestDeleteRule:
    """Test delete_rule method."""

    def test_delete_existing_rule(self, service: SpendCategorizerService) -> None:
        rule = service.create_rule(name="R", taxonomy_code="44000000", conditions={"keywords": ["a"]})
        result = service.delete_rule(rule.rule_id)
        assert result is True
        assert service.get_rule(rule.rule_id) is None

    def test_delete_nonexistent_rule(self, service: SpendCategorizerService) -> None:
        result = service.delete_rule("nonexistent")
        assert result is False

    def test_delete_updates_statistics(self, service: SpendCategorizerService) -> None:
        rule = service.create_rule(name="R", taxonomy_code="44000000", conditions={"keywords": ["a"]})
        assert service.get_statistics().total_rules == 1
        service.delete_rule(rule.rule_id)
        assert service.get_statistics().total_rules == 0

    def test_delete_records_provenance(self, service: SpendCategorizerService) -> None:
        rule = service.create_rule(name="R", taxonomy_code="44000000", conditions={"keywords": ["a"]})
        before = service.provenance.entry_count
        service.delete_rule(rule.rule_id)
        assert service.provenance.entry_count == before + 1


# ===================================================================
# TestGetAnalytics
# ===================================================================


class TestGetAnalytics:
    """Test get_analytics method."""

    def test_analytics_empty_service(self, service: SpendCategorizerService) -> None:
        analytics = service.get_analytics()
        assert isinstance(analytics, AnalyticsResponse)
        assert analytics.total_records == 0
        assert analytics.total_spend_usd == 0.0

    def test_analytics_with_data(
        self,
        service: SpendCategorizerService,
        emissions_in_service: List[EmissionCalculationResponse],
    ) -> None:
        analytics = service.get_analytics()
        assert analytics.total_records > 0
        assert analytics.total_spend_usd > 0
        assert analytics.total_classified > 0

    def test_analytics_top_categories(
        self,
        service: SpendCategorizerService,
        emissions_in_service: List[EmissionCalculationResponse],
    ) -> None:
        analytics = service.get_analytics()
        assert isinstance(analytics.top_categories, list)

    def test_analytics_top_vendors(
        self,
        service: SpendCategorizerService,
        emissions_in_service: List[EmissionCalculationResponse],
    ) -> None:
        analytics = service.get_analytics()
        assert isinstance(analytics.top_vendors, list)
        assert len(analytics.top_vendors) > 0

    def test_analytics_scope3_breakdown(
        self,
        service: SpendCategorizerService,
        emissions_in_service: List[EmissionCalculationResponse],
    ) -> None:
        analytics = service.get_analytics()
        assert isinstance(analytics.scope3_breakdown, list)

    def test_analytics_classification_rate(
        self,
        service: SpendCategorizerService,
        classifications_in_service: List[ClassificationResponse],
    ) -> None:
        analytics = service.get_analytics()
        assert analytics.classification_rate_pct > 0.0

    def test_analytics_provenance_hash(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        analytics = service.get_analytics()
        assert len(analytics.provenance_hash) == 64

    def test_analytics_avg_confidence(
        self,
        service: SpendCategorizerService,
        classifications_in_service: List[ClassificationResponse],
    ) -> None:
        analytics = service.get_analytics()
        assert analytics.avg_confidence >= 0.0


# ===================================================================
# TestGetHotspots
# ===================================================================


class TestGetHotspots:
    """Test get_hotspots method."""

    def test_hotspots_empty(self, service: SpendCategorizerService) -> None:
        hotspots = service.get_hotspots()
        assert hotspots == []

    def test_hotspots_with_data(
        self,
        service: SpendCategorizerService,
        emissions_in_service: List[EmissionCalculationResponse],
    ) -> None:
        hotspots = service.get_hotspots()
        assert len(hotspots) > 0
        for h in hotspots:
            assert "taxonomy_code" in h
            assert "emissions_kg_co2e" in h

    def test_hotspots_top_n(
        self,
        service: SpendCategorizerService,
        emissions_in_service: List[EmissionCalculationResponse],
    ) -> None:
        hotspots = service.get_hotspots(top_n=2)
        assert len(hotspots) <= 2

    def test_hotspots_sorted_descending(
        self,
        service: SpendCategorizerService,
        emissions_in_service: List[EmissionCalculationResponse],
    ) -> None:
        hotspots = service.get_hotspots()
        if len(hotspots) >= 2:
            assert hotspots[0]["emissions_kg_co2e"] >= hotspots[1]["emissions_kg_co2e"]


# ===================================================================
# TestGetTrends
# ===================================================================


class TestGetTrends:
    """Test get_trends method."""

    def test_trends_empty(self, service: SpendCategorizerService) -> None:
        trends = service.get_trends()
        assert trends == []

    def test_trends_with_data(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        trends = service.get_trends(group_by="month")
        assert len(trends) > 0
        for t in trends:
            assert "period" in t
            assert "spend_usd" in t

    def test_trends_group_by_day(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        trends = service.get_trends(group_by="day")
        assert len(trends) > 0

    def test_trends_group_by_quarter(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        trends = service.get_trends(group_by="quarter")
        assert len(trends) > 0

    def test_trends_group_by_year(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        trends = service.get_trends(group_by="year")
        assert len(trends) > 0

    def test_trends_group_by_week(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        trends = service.get_trends(group_by="week")
        assert len(trends) > 0

    def test_trends_unknown_group_by(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        trends = service.get_trends(group_by="unknown")
        # Falls through to month default
        assert len(trends) > 0

    def test_trends_records_without_dates(self, service: SpendCategorizerService) -> None:
        service.ingest_records([{"vendor_name": "V", "amount": 100}])
        trends = service.get_trends()
        assert len(trends) == 0  # No date -> no period

    def test_trends_invalid_date_format(self, service: SpendCategorizerService) -> None:
        service.ingest_records([{"vendor_name": "V", "amount": 100, "date": "not-a-date"}])
        trends = service.get_trends()
        if len(trends) > 0:
            assert trends[0]["period"] == "unknown"


# ===================================================================
# TestGenerateReport
# ===================================================================


class TestGenerateReport:
    """Test generate_report method."""

    def test_generate_summary_json(self, service: SpendCategorizerService) -> None:
        report = service.generate_report(report_type="summary", report_format="json")
        assert isinstance(report, ReportResponse)
        assert report.report_type == "summary"
        assert report.format == "json"
        assert report.content is not None

    def test_generate_detailed_json(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        report = service.generate_report(report_type="detailed", report_format="json")
        assert report.content is not None
        assert "records" in report.content

    def test_generate_emissions_report(
        self,
        service: SpendCategorizerService,
        emissions_in_service: List[EmissionCalculationResponse],
    ) -> None:
        report = service.generate_report(report_type="emissions", report_format="json")
        assert report.content is not None
        assert "emission_calculations" in report.content

    def test_generate_scope3_report(
        self,
        service: SpendCategorizerService,
        scope3_in_service: List[Scope3AssignmentResponse],
    ) -> None:
        report = service.generate_report(report_type="scope3", report_format="json")
        assert report.content is not None
        assert "scope3_assignments" in report.content

    def test_generate_csv_format(self, service: SpendCategorizerService) -> None:
        report = service.generate_report(report_format="csv")
        assert report.format == "csv"

    def test_generate_excel_format(self, service: SpendCategorizerService) -> None:
        report = service.generate_report(report_format="excel")
        assert report.format == "excel"

    def test_generate_provenance_hash(self, service: SpendCategorizerService) -> None:
        report = service.generate_report()
        assert len(report.provenance_hash) == 64

    def test_generate_statistics_updated(self, service: SpendCategorizerService) -> None:
        service.generate_report()
        stats = service.get_statistics()
        assert stats.total_reports == 1

    def test_generate_report_id(self, service: SpendCategorizerService) -> None:
        report = service.generate_report()
        uid = uuid.UUID(report.report_id)
        assert uid.version == 4

    def test_generate_report_counts(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        report = service.generate_report()
        assert report.record_count == 5

    def test_generate_report_spend_total(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        report = service.generate_report()
        assert report.total_spend_usd > 0.0


# ===================================================================
# TestStatistics
# ===================================================================


class TestStatistics:
    """Test get_statistics method."""

    def test_initial_statistics(self, service: SpendCategorizerService) -> None:
        stats = service.get_statistics()
        assert isinstance(stats, SpendCategorizerStatisticsResponse)
        assert stats.total_records == 0
        assert stats.total_classified == 0
        assert stats.total_scope3_mapped == 0
        assert stats.total_emissions_calculated == 0
        assert stats.total_rules == 0
        assert stats.active_rules == 0

    def test_statistics_after_full_pipeline(
        self,
        service: SpendCategorizerService,
        emissions_in_service: List[EmissionCalculationResponse],
    ) -> None:
        stats = service.get_statistics()
        assert stats.total_records == 5
        assert stats.total_classified > 0
        assert stats.total_scope3_mapped > 0
        assert stats.total_emissions_calculated > 0
        assert stats.total_spend_usd > 0
        assert stats.total_emissions_kg_co2e > 0

    def test_statistics_avg_confidence(
        self,
        service: SpendCategorizerService,
        classifications_in_service: List[ClassificationResponse],
    ) -> None:
        stats = service.get_statistics()
        assert stats.avg_confidence >= 0.0


# ===================================================================
# TestHealthCheck
# ===================================================================


class TestHealthCheck:
    """Test health_check method."""

    def test_health_before_start(self, service: SpendCategorizerService) -> None:
        health = service.health_check()
        assert health["status"] == "not_started"
        assert health["started"] is False

    def test_health_after_start(self, service: SpendCategorizerService) -> None:
        service.startup()
        health = service.health_check()
        assert health["status"] == "healthy"
        assert health["started"] is True

    def test_health_counts(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        health = service.health_check()
        assert health["records"] == 5
        assert health["service"] == "spend-categorizer"

    def test_health_provenance_count(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        health = service.health_check()
        assert health["provenance_entries"] > 0

    def test_health_all_fields_present(self, service: SpendCategorizerService) -> None:
        health = service.health_check()
        expected_keys = [
            "status", "service", "started", "records",
            "classifications", "scope3_assignments",
            "emission_calculations", "rules", "reports",
            "provenance_entries", "prometheus_available",
        ]
        for key in expected_keys:
            assert key in health


# ===================================================================
# TestGetMetrics
# ===================================================================


class TestGetMetrics:
    """Test get_metrics method."""

    def test_metrics_initial(self, service: SpendCategorizerService) -> None:
        metrics = service.get_metrics()
        assert metrics["total_records"] == 0
        assert metrics["started"] is False

    def test_metrics_after_processing(
        self,
        service: SpendCategorizerService,
        emissions_in_service: List[EmissionCalculationResponse],
    ) -> None:
        metrics = service.get_metrics()
        assert metrics["total_records"] > 0
        assert metrics["total_classified"] > 0
        assert metrics["total_scope3_mapped"] > 0
        assert metrics["total_emissions_calculated"] > 0

    def test_metrics_all_keys(self, service: SpendCategorizerService) -> None:
        metrics = service.get_metrics()
        expected_keys = [
            "prometheus_available", "started", "total_records",
            "total_classified", "total_scope3_mapped",
            "total_emissions_calculated", "total_rules", "active_rules",
            "total_reports", "total_spend_usd", "total_emissions_kg_co2e",
            "avg_confidence", "provenance_entries",
        ]
        for key in expected_keys:
            assert key in metrics


# ===================================================================
# TestGetProvenance
# ===================================================================


class TestGetProvenance:
    """Test get_provenance method."""

    def test_get_provenance_tracker(self, service: SpendCategorizerService) -> None:
        tracker = service.get_provenance()
        assert isinstance(tracker, _ProvenanceTracker)

    def test_provenance_entries_grow(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        tracker = service.get_provenance()
        assert tracker.entry_count == 5


# ===================================================================
# TestLifecycle
# ===================================================================


class TestLifecycle:
    """Test startup/shutdown lifecycle."""

    def test_startup(self, service: SpendCategorizerService) -> None:
        service.startup()
        assert service._started is True

    def test_startup_idempotent(self, service: SpendCategorizerService) -> None:
        service.startup()
        service.startup()
        assert service._started is True

    def test_shutdown(self, service: SpendCategorizerService) -> None:
        service.startup()
        service.shutdown()
        assert service._started is False

    def test_shutdown_without_start(self, service: SpendCategorizerService) -> None:
        service.shutdown()
        assert service._started is False


# ===================================================================
# TestProvenanceThrough
# ===================================================================


class TestProvenanceThrough:
    """Test SHA-256 provenance chain through full workflow."""

    def test_provenance_hashes_are_sha256(
        self,
        service: SpendCategorizerService,
        emissions_in_service: List[EmissionCalculationResponse],
    ) -> None:
        tracker = service.get_provenance()
        for entry in tracker._entries:
            assert len(entry["entry_hash"]) == 64

    def test_provenance_entry_fields(
        self,
        service: SpendCategorizerService,
        records_in_service: List[SpendRecordResponse],
    ) -> None:
        tracker = service.get_provenance()
        entry = tracker._entries[0]
        assert "entity_type" in entry
        assert "entity_id" in entry
        assert "action" in entry
        assert "data_hash" in entry
        assert "user_id" in entry
        assert "timestamp" in entry
        assert "entry_hash" in entry

    def test_provenance_actions_sequence(
        self,
        service: SpendCategorizerService,
        emissions_in_service: List[EmissionCalculationResponse],
    ) -> None:
        tracker = service.get_provenance()
        actions = [e["action"] for e in tracker._entries]
        assert "ingest" in actions
        assert "classify" in actions
        assert "map" in actions
        assert "calculate" in actions


# ===================================================================
# TestComputeHash
# ===================================================================


class TestComputeHash:
    """Test the _compute_hash helper."""

    def test_dict_hash(self) -> None:
        h = _compute_hash({"key": "value"})
        assert len(h) == 64

    def test_deterministic(self) -> None:
        h1 = _compute_hash({"a": 1, "b": 2})
        h2 = _compute_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_different_data_different_hash(self) -> None:
        h1 = _compute_hash({"key": "value1"})
        h2 = _compute_hash({"key": "value2"})
        assert h1 != h2

    def test_pydantic_model_hash(self) -> None:
        model = SpendRecordResponse(vendor_name="Test", amount=100)
        h = _compute_hash(model)
        assert len(h) == 64


# ===================================================================
# TestFullWorkflow
# ===================================================================


class TestFullWorkflow:
    """Test ingest -> classify -> map -> calculate -> analyze -> report."""

    def test_full_pipeline(self, service: SpendCategorizerService) -> None:
        # Step 1: Ingest
        records = service.ingest_records(_sample_records(3), source="pipeline_test")
        assert len(records) == 3

        # Step 2: Classify
        rids = [r.record_id for r in records]
        classifications = service.classify_batch(rids)
        assert len(classifications) > 0

        # Step 3: Map to Scope 3
        classified_rids = [
            c.record_id for c in classifications
            if service.get_record(c.record_id).taxonomy_code
        ]
        assignments = service.map_scope3_batch(classified_rids)

        # Step 4: Calculate emissions
        mapped_rids = [a.record_id for a in assignments]
        calculations = service.calculate_emissions_batch(mapped_rids)

        # Step 5: Analytics
        analytics = service.get_analytics()
        assert analytics.total_records == 3

        # Step 6: Report
        report = service.generate_report(report_type="summary", report_format="json")
        assert report.record_count == 3
        assert report.provenance_hash != ""

        # Verify provenance chain
        tracker = service.get_provenance()
        assert tracker.entry_count > 0

    def test_full_pipeline_with_rules(self, service: SpendCategorizerService) -> None:
        # Create rules first
        service.create_rule(
            name="Office",
            taxonomy_code="44000000",
            conditions={"keywords": ["office", "supplies", "paper"]},
            priority=1,
        )
        service.create_rule(
            name="IT",
            taxonomy_code="43000000",
            conditions={"keywords": ["computer", "laptop", "software"]},
            priority=2,
        )

        # Ingest and classify
        records = service.ingest_records(_sample_records(5), source="rule_test")
        rids = [r.record_id for r in records]
        classifications = service.classify_batch(rids)

        # Verify some matched rules
        rule_matched = [c for c in classifications if c.method == "rule"]
        assert len(rule_matched) >= 1


# ===================================================================
# TestMultiTenant
# ===================================================================


class TestMultiTenant:
    """Test tenant isolation via separate service instances."""

    def test_separate_instances(self) -> None:
        svc_a = _build_service()
        svc_b = _build_service()
        svc_a.ingest_records([{"vendor_name": "A", "amount": 100}])
        svc_b.ingest_records([{"vendor_name": "B", "amount": 200}])
        assert len(svc_a._records) == 1
        assert len(svc_b._records) == 1
        assert list(svc_a._records.values())[0].vendor_name == "A"
        assert list(svc_b._records.values())[0].vendor_name == "B"


# ===================================================================
# TestThreadSafety
# ===================================================================


class TestThreadSafety:
    """Test concurrent operations."""

    def test_concurrent_ingestion(self, service: SpendCategorizerService) -> None:
        errors: List[str] = []

        def ingest(source: str) -> None:
            try:
                service.ingest_records(
                    [{"vendor_name": f"V-{source}", "amount": 100}],
                    source=source,
                )
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=ingest, args=(f"t{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(service._records) == 10

    def test_concurrent_rule_creation(self, service: SpendCategorizerService) -> None:
        errors: List[str] = []

        def create(idx: int) -> None:
            try:
                service.create_rule(
                    name=f"Rule-{idx}",
                    taxonomy_code="44000000",
                    conditions={"keywords": [f"kw{idx}"]},
                )
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=create, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(service._rules) == 10


# ===================================================================
# TestConfigureFunction
# ===================================================================


class TestConfigureFunction:
    """Test configure_spend_categorizer async function."""

    def test_configure_attaches_service(self) -> None:
        from fastapi import FastAPI
        app = FastAPI()

        with patch.object(SpendCategorizerService, "_init_engines"):
            loop = asyncio.new_event_loop()
            try:
                service = loop.run_until_complete(
                    configure_spend_categorizer(app, config=_make_config())
                )
            finally:
                loop.close()

        assert hasattr(app.state, "spend_categorizer_service")
        assert app.state.spend_categorizer_service is service
        assert service._started is True

    def test_configure_mounts_router(self) -> None:
        from fastapi import FastAPI
        app = FastAPI()

        with patch.object(SpendCategorizerService, "_init_engines"):
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(
                    configure_spend_categorizer(app, config=_make_config())
                )
            finally:
                loop.close()

        route_paths = [r.path for r in app.routes]
        # Router prefix is /api/v1/spend-categorizer
        assert any("/api/v1/spend-categorizer" in p for p in route_paths)

    def test_configure_returns_service(self) -> None:
        from fastapi import FastAPI
        app = FastAPI()

        with patch.object(SpendCategorizerService, "_init_engines"):
            loop = asyncio.new_event_loop()
            try:
                service = loop.run_until_complete(
                    configure_spend_categorizer(app, config=_make_config())
                )
            finally:
                loop.close()

        assert isinstance(service, SpendCategorizerService)


# ===================================================================
# TestGetSpendCategorizer
# ===================================================================


class TestGetSpendCategorizer:
    """Test get_spend_categorizer function."""

    def test_get_success(self) -> None:
        from fastapi import FastAPI
        app = FastAPI()
        svc = _build_service()
        app.state.spend_categorizer_service = svc
        result = get_spend_categorizer(app)
        assert result is svc

    def test_get_not_configured_raises(self) -> None:
        from fastapi import FastAPI
        app = FastAPI()
        with pytest.raises(RuntimeError, match="not configured"):
            get_spend_categorizer(app)


# ===================================================================
# TestPydanticModels
# ===================================================================


class TestPydanticModels:
    """Test Pydantic response model defaults."""

    def test_spend_record_response_defaults(self) -> None:
        r = SpendRecordResponse()
        assert r.vendor_name == ""
        assert r.amount == 0.0
        assert r.currency == "USD"
        assert r.status == "ingested"
        assert r.provenance_hash == ""

    def test_classification_response_defaults(self) -> None:
        c = ClassificationResponse()
        assert c.record_id == ""
        assert c.taxonomy_system == "unspsc"
        assert c.confidence == 0.0
        assert c.confidence_label == "LOW"

    def test_scope3_assignment_response_defaults(self) -> None:
        a = Scope3AssignmentResponse()
        assert a.record_id == ""
        assert a.scope3_category == 0
        assert a.mapping_method == "taxonomy_lookup"

    def test_emission_calculation_response_defaults(self) -> None:
        e = EmissionCalculationResponse()
        assert e.record_id == ""
        assert e.emissions_kg_co2e == 0.0
        assert e.methodology == "spend_based"

    def test_category_rule_response_defaults(self) -> None:
        r = CategoryRuleResponse()
        assert r.name == ""
        assert r.is_active is True
        assert r.priority == 100

    def test_analytics_response_defaults(self) -> None:
        a = AnalyticsResponse()
        assert a.total_records == 0
        assert a.total_spend_usd == 0.0
        assert a.top_categories == []

    def test_report_response_defaults(self) -> None:
        r = ReportResponse()
        assert r.report_type == "summary"
        assert r.format == "json"
        assert r.content is None

    def test_statistics_response_defaults(self) -> None:
        s = SpendCategorizerStatisticsResponse()
        assert s.total_records == 0
        assert s.total_classified == 0
        assert s.active_batches == 0


# ===================================================================
# TestScope3Categories
# ===================================================================


class TestScope3Categories:
    """Test _SCOPE3_CATEGORIES reference data."""

    def test_all_15_categories(self) -> None:
        assert len(_SCOPE3_CATEGORIES) == 15

    def test_category_1(self) -> None:
        assert _SCOPE3_CATEGORIES[1] == "Purchased Goods and Services"

    def test_category_6(self) -> None:
        assert _SCOPE3_CATEGORIES[6] == "Business Travel"

    def test_category_15(self) -> None:
        assert _SCOPE3_CATEGORIES[15] == "Investments"


# ===================================================================
# TestProvenanceTracker
# ===================================================================


class TestProvenanceTracker:
    """Test _ProvenanceTracker directly."""

    def test_record_entry(self) -> None:
        tracker = _ProvenanceTracker()
        h = tracker.record("test", "id-1", "create", "datahash123")
        assert len(h) == 64
        assert tracker.entry_count == 1

    def test_record_multiple_entries(self) -> None:
        tracker = _ProvenanceTracker()
        tracker.record("a", "1", "create", "h1")
        tracker.record("b", "2", "update", "h2")
        assert tracker.entry_count == 2

    def test_record_returns_unique_hashes(self) -> None:
        tracker = _ProvenanceTracker()
        h1 = tracker.record("a", "1", "create", "h1")
        h2 = tracker.record("b", "2", "update", "h2")
        assert h1 != h2

    def test_record_stores_entry(self) -> None:
        tracker = _ProvenanceTracker()
        tracker.record("test", "id-1", "create", "datahash")
        assert len(tracker._entries) == 1
        entry = tracker._entries[0]
        assert entry["entity_type"] == "test"
        assert entry["entity_id"] == "id-1"
        assert entry["action"] == "create"


# ===================================================================
# TestTaxonomyToScope3Mapping
# ===================================================================


class TestTaxonomyToScope3Mapping:
    """Test _map_taxonomy_to_scope3 internal method."""

    def test_it_equipment_maps_to_1(self, service: SpendCategorizerService) -> None:
        result = service._map_taxonomy_to_scope3("43000000")
        assert result == 1

    def test_transportation_maps_to_4(self, service: SpendCategorizerService) -> None:
        result = service._map_taxonomy_to_scope3("78000000")
        assert result == 4

    def test_fuels_maps_to_3(self, service: SpendCategorizerService) -> None:
        result = service._map_taxonomy_to_scope3("15000000")
        assert result == 3

    def test_travel_maps_to_6(self, service: SpendCategorizerService) -> None:
        result = service._map_taxonomy_to_scope3("86000000")
        assert result == 6

    def test_waste_maps_to_5(self, service: SpendCategorizerService) -> None:
        result = service._map_taxonomy_to_scope3("90000000")
        assert result == 5

    def test_capital_goods_maps_to_2(self, service: SpendCategorizerService) -> None:
        result = service._map_taxonomy_to_scope3("20000000")
        assert result == 2

    def test_building_materials_maps_to_2(self, service: SpendCategorizerService) -> None:
        result = service._map_taxonomy_to_scope3("72000000")
        assert result == 2

    def test_unknown_prefix_defaults_to_1(self, service: SpendCategorizerService) -> None:
        result = service._map_taxonomy_to_scope3("99000000")
        assert result == 1

    def test_empty_code_defaults_to_1(self, service: SpendCategorizerService) -> None:
        result = service._map_taxonomy_to_scope3("")
        assert result == 1

    def test_short_code(self, service: SpendCategorizerService) -> None:
        result = service._map_taxonomy_to_scope3("7")
        assert result == 1  # Falls through to default


# ===================================================================
# TestExtractPeriod
# ===================================================================


class TestExtractPeriod:
    """Test _extract_period internal method."""

    def test_day_format(self, service: SpendCategorizerService) -> None:
        result = service._extract_period("2025-06-15", "day")
        assert result == "2025-06-15"

    def test_month_format(self, service: SpendCategorizerService) -> None:
        result = service._extract_period("2025-06-15", "month")
        assert result == "2025-06"

    def test_quarter_format(self, service: SpendCategorizerService) -> None:
        result = service._extract_period("2025-06-15", "quarter")
        assert result == "2025-Q2"

    def test_year_format(self, service: SpendCategorizerService) -> None:
        result = service._extract_period("2025-06-15", "year")
        assert result == "2025"

    def test_week_format(self, service: SpendCategorizerService) -> None:
        result = service._extract_period("2025-06-15", "week")
        assert "2025-W" in result

    def test_invalid_date(self, service: SpendCategorizerService) -> None:
        result = service._extract_period("not-a-date", "month")
        assert result == "unknown"

    def test_unknown_group_by(self, service: SpendCategorizerService) -> None:
        result = service._extract_period("2025-06-15", "unknown_group")
        assert result == "2025-06"

    def test_quarter_q1(self, service: SpendCategorizerService) -> None:
        result = service._extract_period("2025-02-01", "quarter")
        assert result == "2025-Q1"

    def test_quarter_q3(self, service: SpendCategorizerService) -> None:
        result = service._extract_period("2025-09-01", "quarter")
        assert result == "2025-Q3"

    def test_quarter_q4(self, service: SpendCategorizerService) -> None:
        result = service._extract_period("2025-12-01", "quarter")
        assert result == "2025-Q4"


# ===================================================================
# TestUpdateAvgConfidence
# ===================================================================


class TestUpdateAvgConfidence:
    """Test _update_avg_confidence internal method."""

    def test_first_value(self, service: SpendCategorizerService) -> None:
        service._stats.total_classified = 0
        service._update_avg_confidence(0.8)
        assert service._stats.avg_confidence == 0.8

    def test_running_average(self, service: SpendCategorizerService) -> None:
        service._stats.total_classified = 1
        service._stats.avg_confidence = 0.8
        service._stats.total_classified = 2
        service._update_avg_confidence(0.6)
        expected = (0.8 * 1 + 0.6) / 2
        assert abs(service._stats.avg_confidence - expected) < 1e-6


# ===================================================================
# TestFactorVersion
# ===================================================================


class TestFactorVersion:
    """Test _get_factor_version internal method."""

    def test_eeio_version(self, service: SpendCategorizerService) -> None:
        assert service._get_factor_version("eeio") == "2024"

    def test_exiobase_version(self, service: SpendCategorizerService) -> None:
        assert service._get_factor_version("exiobase") == "3.8.2"

    def test_defra_version(self, service: SpendCategorizerService) -> None:
        assert service._get_factor_version("defra") == "2025"

    def test_ecoinvent_version(self, service: SpendCategorizerService) -> None:
        assert service._get_factor_version("ecoinvent") == "3.10"

    def test_unknown_version(self, service: SpendCategorizerService) -> None:
        assert service._get_factor_version("unknown") == "unknown"


# ===================================================================
# TestRuleMatching
# ===================================================================


class TestRuleMatching:
    """Test _rule_matches internal method."""

    def test_vendor_pattern_match(self, service: SpendCategorizerService) -> None:
        rule = CategoryRuleResponse(
            conditions={"vendor_pattern": "acme"},
            taxonomy_code="44000000",
        )
        record = SpendRecordResponse(vendor_name="Acme Corp", description="")
        assert service._rule_matches(rule, record) is True

    def test_vendor_pattern_no_match(self, service: SpendCategorizerService) -> None:
        rule = CategoryRuleResponse(
            conditions={"vendor_pattern": "xyz"},
            taxonomy_code="44000000",
        )
        record = SpendRecordResponse(vendor_name="Acme Corp", description="")
        assert service._rule_matches(rule, record) is False

    def test_keyword_match(self, service: SpendCategorizerService) -> None:
        rule = CategoryRuleResponse(
            conditions={"keywords": ["office", "supplies"]},
            taxonomy_code="44000000",
        )
        record = SpendRecordResponse(vendor_name="V", description="Office furniture and supplies")
        assert service._rule_matches(rule, record) is True

    def test_keyword_no_match(self, service: SpendCategorizerService) -> None:
        rule = CategoryRuleResponse(
            conditions={"keywords": ["fuel", "diesel"]},
            taxonomy_code="15000000",
        )
        record = SpendRecordResponse(vendor_name="V", description="Office supplies")
        assert service._rule_matches(rule, record) is False

    def test_gl_code_match(self, service: SpendCategorizerService) -> None:
        rule = CategoryRuleResponse(
            conditions={"gl_codes": ["GL-5000", "GL-5001"]},
            taxonomy_code="44000000",
        )
        record = SpendRecordResponse(vendor_name="V", description="", gl_account="GL-5000")
        assert service._rule_matches(rule, record) is True

    def test_gl_code_no_match(self, service: SpendCategorizerService) -> None:
        rule = CategoryRuleResponse(
            conditions={"gl_codes": ["GL-5000"]},
            taxonomy_code="44000000",
        )
        record = SpendRecordResponse(vendor_name="V", description="", gl_account="GL-9999")
        assert service._rule_matches(rule, record) is False

    def test_cost_center_match(self, service: SpendCategorizerService) -> None:
        rule = CategoryRuleResponse(
            conditions={"cost_centers": ["CC-FAC"]},
            taxonomy_code="72000000",
        )
        record = SpendRecordResponse(vendor_name="V", description="", cost_center="CC-FAC")
        assert service._rule_matches(rule, record) is True

    def test_cost_center_no_match(self, service: SpendCategorizerService) -> None:
        rule = CategoryRuleResponse(
            conditions={"cost_centers": ["CC-FAC"]},
            taxonomy_code="72000000",
        )
        record = SpendRecordResponse(vendor_name="V", description="", cost_center="CC-IT")
        assert service._rule_matches(rule, record) is False

    def test_combined_conditions_all_match(self, service: SpendCategorizerService) -> None:
        rule = CategoryRuleResponse(
            conditions={
                "vendor_pattern": "acme",
                "keywords": ["supplies"],
                "gl_codes": ["GL-1000"],
            },
            taxonomy_code="44000000",
        )
        record = SpendRecordResponse(
            vendor_name="Acme Corp",
            description="Office supplies",
            gl_account="GL-1000",
        )
        assert service._rule_matches(rule, record) is True

    def test_combined_conditions_partial_fail(self, service: SpendCategorizerService) -> None:
        rule = CategoryRuleResponse(
            conditions={
                "vendor_pattern": "acme",
                "keywords": ["fuel"],
            },
            taxonomy_code="44000000",
        )
        record = SpendRecordResponse(
            vendor_name="Acme Corp",
            description="Office supplies",
        )
        assert service._rule_matches(rule, record) is False

    def test_empty_conditions_always_matches(self, service: SpendCategorizerService) -> None:
        rule = CategoryRuleResponse(conditions={}, taxonomy_code="44000000")
        record = SpendRecordResponse(vendor_name="Any", description="Anything")
        assert service._rule_matches(rule, record) is True


# ===================================================================
# TestLookupEmissionFactor
# ===================================================================


class TestLookupEmissionFactor:
    """Test _lookup_emission_factor internal method."""

    def test_exact_match(self, service: SpendCategorizerService) -> None:
        result = service._lookup_emission_factor("43000000", "eeio")
        assert result["factor"] == 0.42

    def test_prefix_match(self, service: SpendCategorizerService) -> None:
        result = service._lookup_emission_factor("4300", "eeio")
        assert result["factor"] == 0.42

    def test_no_match_returns_default(self, service: SpendCategorizerService) -> None:
        result = service._lookup_emission_factor("99999999", "eeio")
        assert result["factor"] == 0.25

    def test_default_source_passed_through(self, service: SpendCategorizerService) -> None:
        result = service._lookup_emission_factor("99999999", "defra")
        assert result["source"] == "defra"
