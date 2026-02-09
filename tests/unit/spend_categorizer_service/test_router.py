# -*- coding: utf-8 -*-
"""
Unit tests for Spend Data Categorizer REST API Router - AGENT-DATA-009

Tests all 20 FastAPI router endpoints using httpx TestClient covering
record ingestion, taxonomy classification, Scope 3 mapping, emission
calculation, rule management, analytics, reporting, and health monitoring.

Each endpoint is tested for success paths, error paths (400, 404, 503),
and response format validation.

Target: 85%+ coverage of greenlang/spend_categorizer/api/router.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from greenlang.spend_categorizer.api.router import router
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
)


# ===================================================================
# Helpers
# ===================================================================

PREFIX = "/api/v1/spend-categorizer"


def _make_config(**overrides: Any) -> SpendCategorizerConfig:
    """Build a SpendCategorizerConfig with defaults for testing."""
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


def _sample_records(count: int = 3) -> List[Dict[str, Any]]:
    """Generate sample spend record dicts for API payloads."""
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
def service() -> SpendCategorizerService:
    """Create a fresh SpendCategorizerService instance for each test."""
    with patch.object(SpendCategorizerService, "_init_engines"):
        svc = SpendCategorizerService(config=_make_config())
    svc.startup()
    return svc


@pytest.fixture
def app(service: SpendCategorizerService) -> FastAPI:
    """Create a FastAPI app with the service attached and router included."""
    application = FastAPI()
    application.state.spend_categorizer_service = service
    application.include_router(router)
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a TestClient from the app."""
    return TestClient(app)


@pytest.fixture
def ingested_records(client: TestClient) -> Dict[str, Any]:
    """POST ingest records and return response JSON."""
    body = {
        "records": _sample_records(5),
        "source": "test",
    }
    resp = client.post(f"{PREFIX}/v1/ingest", json=body)
    assert resp.status_code == 200
    return resp.json()


@pytest.fixture
def classified_records(
    client: TestClient,
    ingested_records: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Classify all ingested records and return classification list."""
    rids = [r["record_id"] for r in ingested_records["records"]]
    resp = client.post(
        f"{PREFIX}/v1/classify/batch",
        json={"record_ids": rids},
    )
    assert resp.status_code == 200
    return resp.json()["classifications"]


@pytest.fixture
def mapped_records(
    client: TestClient,
    service: SpendCategorizerService,
    classified_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Map classified records to Scope 3 and return assignments."""
    # Filter to those with taxonomy_code set
    valid_rids = []
    for c in classified_records:
        rec = service.get_record(c["record_id"])
        if rec and rec.taxonomy_code:
            valid_rids.append(c["record_id"])
    if not valid_rids:
        return []
    resp = client.post(
        f"{PREFIX}/v1/map-scope3/batch",
        json={"record_ids": valid_rids},
    )
    assert resp.status_code == 200
    return resp.json()["assignments"]


@pytest.fixture
def calculated_records(
    client: TestClient,
    mapped_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Calculate emissions for mapped records and return calculations."""
    rids = [a["record_id"] for a in mapped_records]
    if not rids:
        return []
    resp = client.post(
        f"{PREFIX}/v1/calculate-emissions/batch",
        json={"record_ids": rids},
    )
    assert resp.status_code == 200
    return resp.json()["calculations"]


# ===================================================================
# TestIngest
# ===================================================================


class TestIngest:
    """Test POST /v1/ingest endpoint."""

    def test_ingest_success(self, client: TestClient) -> None:
        body = {
            "records": _sample_records(2),
            "source": "api",
        }
        resp = client.post(f"{PREFIX}/v1/ingest", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert data["source"] == "api"
        assert len(data["records"]) == 2

    def test_ingest_single_record(self, client: TestClient) -> None:
        body = {
            "records": [{"vendor_name": "Solo", "amount": 100, "description": "Single"}],
        }
        resp = client.post(f"{PREFIX}/v1/ingest", json=body)
        assert resp.status_code == 200
        assert resp.json()["count"] == 1

    def test_ingest_empty_records_400(self, client: TestClient) -> None:
        body = {"records": [], "source": "test"}
        resp = client.post(f"{PREFIX}/v1/ingest", json=body)
        assert resp.status_code == 400

    def test_ingest_missing_records_422(self, client: TestClient) -> None:
        resp = client.post(f"{PREFIX}/v1/ingest", json={"source": "test"})
        assert resp.status_code == 422

    def test_ingest_default_source(self, client: TestClient) -> None:
        body = {"records": [{"vendor_name": "V", "amount": 1}]}
        resp = client.post(f"{PREFIX}/v1/ingest", json=body)
        assert resp.status_code == 200
        assert resp.json()["source"] == "manual"

    def test_ingest_record_fields(self, client: TestClient) -> None:
        body = {
            "records": [{
                "vendor_name": "TestVendor",
                "amount": 5000,
                "description": "Test item",
                "currency": "EUR",
            }],
            "source": "csv",
        }
        resp = client.post(f"{PREFIX}/v1/ingest", json=body)
        assert resp.status_code == 200
        rec = resp.json()["records"][0]
        assert rec["vendor_name"] == "TestVendor"
        assert rec["amount"] == 5000.0
        assert rec["source"] == "csv"
        assert rec["status"] == "ingested"

    def test_ingest_provenance_hash(self, client: TestClient) -> None:
        body = {"records": [{"vendor_name": "V", "amount": 1}]}
        resp = client.post(f"{PREFIX}/v1/ingest", json=body)
        rec = resp.json()["records"][0]
        assert len(rec["provenance_hash"]) == 64


# ===================================================================
# TestIngestFile
# ===================================================================


class TestIngestFile:
    """Test POST /v1/ingest/file endpoint."""

    def test_ingest_file_csv_not_found(self, client: TestClient) -> None:
        body = {
            "file_path": "/nonexistent/file.csv",
            "file_type": "csv",
            "source": "file",
        }
        resp = client.post(f"{PREFIX}/v1/ingest/file", json=body)
        assert resp.status_code == 400

    def test_ingest_file_excel_not_found(self, client: TestClient) -> None:
        body = {
            "file_path": "/nonexistent/file.xlsx",
            "file_type": "excel",
            "source": "file",
        }
        resp = client.post(f"{PREFIX}/v1/ingest/file", json=body)
        assert resp.status_code == 400

    def test_ingest_file_missing_path_422(self, client: TestClient) -> None:
        resp = client.post(f"{PREFIX}/v1/ingest/file", json={"file_type": "csv"})
        assert resp.status_code == 422


# ===================================================================
# TestListRecords
# ===================================================================


class TestListRecords:
    """Test GET /v1/records endpoint."""

    def test_list_empty(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/records")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["records"] == []

    def test_list_with_data(
        self,
        client: TestClient,
        ingested_records: Dict[str, Any],
    ) -> None:
        resp = client.get(f"{PREFIX}/v1/records")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 5

    def test_list_filter_source(
        self,
        client: TestClient,
        ingested_records: Dict[str, Any],
    ) -> None:
        resp = client.get(f"{PREFIX}/v1/records?source=test")
        assert resp.status_code == 200
        assert resp.json()["count"] == 5

    def test_list_filter_status(
        self,
        client: TestClient,
        ingested_records: Dict[str, Any],
    ) -> None:
        resp = client.get(f"{PREFIX}/v1/records?status=ingested")
        assert resp.status_code == 200
        assert resp.json()["count"] == 5

    def test_list_filter_vendor(
        self,
        client: TestClient,
        ingested_records: Dict[str, Any],
    ) -> None:
        resp = client.get(f"{PREFIX}/v1/records?vendor_name=Acme")
        assert resp.status_code == 200
        assert resp.json()["count"] >= 1

    def test_list_pagination(
        self,
        client: TestClient,
        ingested_records: Dict[str, Any],
    ) -> None:
        resp = client.get(f"{PREFIX}/v1/records?limit=2&offset=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["limit"] == 2
        assert data["offset"] == 1
        assert data["count"] == 2

    def test_list_no_match_source(
        self,
        client: TestClient,
        ingested_records: Dict[str, Any],
    ) -> None:
        resp = client.get(f"{PREFIX}/v1/records?source=nonexistent")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0


# ===================================================================
# TestGetRecord
# ===================================================================


class TestGetRecord:
    """Test GET /v1/records/{record_id} endpoint."""

    def test_get_existing(
        self,
        client: TestClient,
        ingested_records: Dict[str, Any],
    ) -> None:
        rid = ingested_records["records"][0]["record_id"]
        resp = client.get(f"{PREFIX}/v1/records/{rid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["record_id"] == rid

    def test_get_not_found(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/records/nonexistent-id")
        assert resp.status_code == 404


# ===================================================================
# TestClassify
# ===================================================================


class TestClassify:
    """Test POST /v1/classify endpoint."""

    def test_classify_success(
        self,
        client: TestClient,
        ingested_records: Dict[str, Any],
    ) -> None:
        rid = ingested_records["records"][0]["record_id"]
        resp = client.post(
            f"{PREFIX}/v1/classify",
            json={"record_id": rid},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["record_id"] == rid
        assert "taxonomy_code" in data
        assert "confidence" in data

    def test_classify_with_taxonomy(
        self,
        client: TestClient,
        ingested_records: Dict[str, Any],
    ) -> None:
        rid = ingested_records["records"][0]["record_id"]
        resp = client.post(
            f"{PREFIX}/v1/classify",
            json={"record_id": rid, "taxonomy_system": "naics"},
        )
        assert resp.status_code == 200
        assert resp.json()["taxonomy_system"] == "naics"

    def test_classify_not_found_400(self, client: TestClient) -> None:
        resp = client.post(
            f"{PREFIX}/v1/classify",
            json={"record_id": "nonexistent"},
        )
        assert resp.status_code == 400

    def test_classify_missing_record_id_422(self, client: TestClient) -> None:
        resp = client.post(f"{PREFIX}/v1/classify", json={})
        assert resp.status_code == 422


# ===================================================================
# TestClassifyBatch
# ===================================================================


class TestClassifyBatch:
    """Test POST /v1/classify/batch endpoint."""

    def test_batch_success(
        self,
        client: TestClient,
        ingested_records: Dict[str, Any],
    ) -> None:
        rids = [r["record_id"] for r in ingested_records["records"][:3]]
        resp = client.post(
            f"{PREFIX}/v1/classify/batch",
            json={"record_ids": rids},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 3
        assert data["requested"] == 3

    def test_batch_mixed_ids(
        self,
        client: TestClient,
        ingested_records: Dict[str, Any],
    ) -> None:
        rids = [ingested_records["records"][0]["record_id"], "invalid-id"]
        resp = client.post(
            f"{PREFIX}/v1/classify/batch",
            json={"record_ids": rids},
        )
        assert resp.status_code == 200
        assert resp.json()["count"] == 1
        assert resp.json()["requested"] == 2

    def test_batch_with_taxonomy(
        self,
        client: TestClient,
        ingested_records: Dict[str, Any],
    ) -> None:
        rids = [ingested_records["records"][0]["record_id"]]
        resp = client.post(
            f"{PREFIX}/v1/classify/batch",
            json={"record_ids": rids, "taxonomy_system": "naics"},
        )
        assert resp.status_code == 200


# ===================================================================
# TestMapScope3
# ===================================================================


class TestMapScope3:
    """Test POST /v1/map-scope3 endpoint."""

    def test_map_success(
        self,
        client: TestClient,
        service: SpendCategorizerService,
        classified_records: List[Dict[str, Any]],
    ) -> None:
        # Find a record with taxonomy_code set
        for c in classified_records:
            rec = service.get_record(c["record_id"])
            if rec and rec.taxonomy_code:
                resp = client.post(
                    f"{PREFIX}/v1/map-scope3",
                    json={"record_id": c["record_id"]},
                )
                assert resp.status_code == 200
                data = resp.json()
                assert "scope3_category" in data
                return
        pytest.skip("No classified records with taxonomy_code")

    def test_map_not_found_400(self, client: TestClient) -> None:
        resp = client.post(
            f"{PREFIX}/v1/map-scope3",
            json={"record_id": "nonexistent"},
        )
        assert resp.status_code == 400

    def test_map_unclassified_400(
        self,
        client: TestClient,
        ingested_records: Dict[str, Any],
    ) -> None:
        rid = ingested_records["records"][0]["record_id"]
        resp = client.post(
            f"{PREFIX}/v1/map-scope3",
            json={"record_id": rid},
        )
        # Should fail because record is not classified yet
        assert resp.status_code == 400


# ===================================================================
# TestMapScope3Batch
# ===================================================================


class TestMapScope3Batch:
    """Test POST /v1/map-scope3/batch endpoint."""

    def test_batch_success(
        self,
        client: TestClient,
        service: SpendCategorizerService,
        classified_records: List[Dict[str, Any]],
    ) -> None:
        valid_rids = []
        for c in classified_records:
            rec = service.get_record(c["record_id"])
            if rec and rec.taxonomy_code:
                valid_rids.append(c["record_id"])
        if not valid_rids:
            pytest.skip("No valid classified records")
        resp = client.post(
            f"{PREFIX}/v1/map-scope3/batch",
            json={"record_ids": valid_rids},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == len(valid_rids)

    def test_batch_empty(self, client: TestClient) -> None:
        resp = client.post(
            f"{PREFIX}/v1/map-scope3/batch",
            json={"record_ids": []},
        )
        assert resp.status_code == 200
        assert resp.json()["count"] == 0


# ===================================================================
# TestCalculateEmissions
# ===================================================================


class TestCalculateEmissions:
    """Test POST /v1/calculate-emissions endpoint."""

    def test_calculate_success(
        self,
        client: TestClient,
        mapped_records: List[Dict[str, Any]],
    ) -> None:
        if not mapped_records:
            pytest.skip("No mapped records")
        rid = mapped_records[0]["record_id"]
        resp = client.post(
            f"{PREFIX}/v1/calculate-emissions",
            json={"record_id": rid},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "emissions_kg_co2e" in data
        assert data["methodology"] == "spend_based"

    def test_calculate_not_found_400(self, client: TestClient) -> None:
        resp = client.post(
            f"{PREFIX}/v1/calculate-emissions",
            json={"record_id": "nonexistent"},
        )
        assert resp.status_code == 400

    def test_calculate_with_factor_source(
        self,
        client: TestClient,
        mapped_records: List[Dict[str, Any]],
    ) -> None:
        if not mapped_records:
            pytest.skip("No mapped records")
        rid = mapped_records[0]["record_id"]
        resp = client.post(
            f"{PREFIX}/v1/calculate-emissions",
            json={"record_id": rid, "factor_source": "exiobase"},
        )
        assert resp.status_code == 200
        assert resp.json()["emission_factor_source"] == "exiobase"


# ===================================================================
# TestCalculateEmissionsBatch
# ===================================================================


class TestCalculateEmissionsBatch:
    """Test POST /v1/calculate-emissions/batch endpoint."""

    def test_batch_success(
        self,
        client: TestClient,
        mapped_records: List[Dict[str, Any]],
    ) -> None:
        rids = [a["record_id"] for a in mapped_records]
        if not rids:
            pytest.skip("No mapped records")
        resp = client.post(
            f"{PREFIX}/v1/calculate-emissions/batch",
            json={"record_ids": rids},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == len(rids)

    def test_batch_empty(self, client: TestClient) -> None:
        resp = client.post(
            f"{PREFIX}/v1/calculate-emissions/batch",
            json={"record_ids": []},
        )
        assert resp.status_code == 200
        assert resp.json()["count"] == 0


# ===================================================================
# TestListEmissionFactors
# ===================================================================


class TestListEmissionFactors:
    """Test GET /v1/emission-factors endpoint."""

    def test_list_all(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/emission-factors")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 10
        assert len(data["emission_factors"]) == 10

    def test_list_with_limit(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/emission-factors?limit=3")
        assert resp.status_code == 200
        assert resp.json()["count"] == 3

    def test_list_with_offset(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/emission-factors?offset=8")
        assert resp.status_code == 200
        assert resp.json()["count"] == 2

    def test_list_factor_structure(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/emission-factors?limit=1")
        f = resp.json()["emission_factors"][0]
        assert "taxonomy_code" in f
        assert "factor" in f
        assert "source" in f


# ===================================================================
# TestGetEmissionFactor
# ===================================================================


class TestGetEmissionFactor:
    """Test GET /v1/emission-factors/{taxonomy_code} endpoint."""

    def test_get_known_factor(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/emission-factors/43000000")
        assert resp.status_code == 200
        data = resp.json()
        assert data["taxonomy_code"] == "43000000"
        assert data["factor"] == 0.42

    def test_get_unknown_factor_returns_default(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/emission-factors/99999999")
        assert resp.status_code == 200
        data = resp.json()
        assert data["factor"] == 0.25

    def test_get_with_source_param(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/emission-factors/43000000?source=defra")
        assert resp.status_code == 200
        assert resp.json()["source"] == "defra"


# ===================================================================
# TestCreateRule
# ===================================================================


class TestCreateRule:
    """Test POST /v1/rules endpoint."""

    def test_create_success(self, client: TestClient) -> None:
        body = {
            "name": "Office supplies rule",
            "taxonomy_code": "44000000",
            "conditions": {"keywords": ["office", "supplies"]},
        }
        resp = client.post(f"{PREFIX}/v1/rules", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Office supplies rule"
        assert data["taxonomy_code"] == "44000000"
        assert data["is_active"] is True

    def test_create_with_all_fields(self, client: TestClient) -> None:
        body = {
            "name": "Full rule",
            "taxonomy_code": "43000000",
            "conditions": {"keywords": ["laptop"], "vendor_pattern": "tech"},
            "taxonomy_system": "unspsc",
            "scope3_category": 1,
            "description": "Matches IT equipment",
            "priority": 10,
        }
        resp = client.post(f"{PREFIX}/v1/rules", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["priority"] == 10
        assert data["scope3_category"] == 1
        assert data["description"] == "Matches IT equipment"

    def test_create_empty_name_400(self, client: TestClient) -> None:
        body = {
            "name": "",
            "taxonomy_code": "44000000",
            "conditions": {"keywords": ["test"]},
        }
        resp = client.post(f"{PREFIX}/v1/rules", json=body)
        assert resp.status_code == 400

    def test_create_missing_fields_422(self, client: TestClient) -> None:
        resp = client.post(f"{PREFIX}/v1/rules", json={"name": "Test"})
        assert resp.status_code == 422


# ===================================================================
# TestListRules
# ===================================================================


class TestListRules:
    """Test GET /v1/rules endpoint."""

    def test_list_empty(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/rules")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_list_with_data(self, client: TestClient) -> None:
        # Create two rules
        for name in ["R1", "R2"]:
            client.post(
                f"{PREFIX}/v1/rules",
                json={
                    "name": name,
                    "taxonomy_code": "44000000",
                    "conditions": {"keywords": [name.lower()]},
                },
            )
        resp = client.get(f"{PREFIX}/v1/rules")
        assert resp.status_code == 200
        assert resp.json()["count"] == 2

    def test_list_filter_active(self, client: TestClient) -> None:
        resp_create = client.post(
            f"{PREFIX}/v1/rules",
            json={
                "name": "Active",
                "taxonomy_code": "44000000",
                "conditions": {"keywords": ["a"]},
            },
        )
        rule_id = resp_create.json()["rule_id"]
        client.put(
            f"{PREFIX}/v1/rules/{rule_id}",
            json={"is_active": False},
        )
        resp = client.get(f"{PREFIX}/v1/rules?is_active=true")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_list_pagination(self, client: TestClient) -> None:
        for i in range(5):
            client.post(
                f"{PREFIX}/v1/rules",
                json={
                    "name": f"R{i}",
                    "taxonomy_code": "44000000",
                    "conditions": {"keywords": [f"k{i}"]},
                },
            )
        resp = client.get(f"{PREFIX}/v1/rules?limit=2&offset=1")
        assert resp.status_code == 200
        assert resp.json()["count"] == 2


# ===================================================================
# TestUpdateRule
# ===================================================================


class TestUpdateRule:
    """Test PUT /v1/rules/{rule_id} endpoint."""

    def test_update_success(self, client: TestClient) -> None:
        resp_create = client.post(
            f"{PREFIX}/v1/rules",
            json={
                "name": "Original",
                "taxonomy_code": "44000000",
                "conditions": {"keywords": ["test"]},
            },
        )
        rule_id = resp_create.json()["rule_id"]
        resp = client.put(
            f"{PREFIX}/v1/rules/{rule_id}",
            json={"name": "Updated", "priority": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Updated"
        assert data["priority"] == 5

    def test_update_not_found_404(self, client: TestClient) -> None:
        resp = client.put(
            f"{PREFIX}/v1/rules/nonexistent",
            json={"name": "X"},
        )
        assert resp.status_code == 404

    def test_update_deactivate(self, client: TestClient) -> None:
        resp_create = client.post(
            f"{PREFIX}/v1/rules",
            json={
                "name": "Deactivate me",
                "taxonomy_code": "44000000",
                "conditions": {"keywords": ["test"]},
            },
        )
        rule_id = resp_create.json()["rule_id"]
        resp = client.put(
            f"{PREFIX}/v1/rules/{rule_id}",
            json={"is_active": False},
        )
        assert resp.status_code == 200
        assert resp.json()["is_active"] is False


# ===================================================================
# TestDeleteRule
# ===================================================================


class TestDeleteRule:
    """Test DELETE /v1/rules/{rule_id} endpoint."""

    def test_delete_success(self, client: TestClient) -> None:
        resp_create = client.post(
            f"{PREFIX}/v1/rules",
            json={
                "name": "Delete me",
                "taxonomy_code": "44000000",
                "conditions": {"keywords": ["test"]},
            },
        )
        rule_id = resp_create.json()["rule_id"]
        resp = client.delete(f"{PREFIX}/v1/rules/{rule_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] is True
        assert data["rule_id"] == rule_id

    def test_delete_not_found_404(self, client: TestClient) -> None:
        resp = client.delete(f"{PREFIX}/v1/rules/nonexistent")
        assert resp.status_code == 404


# ===================================================================
# TestGetAnalytics
# ===================================================================


class TestGetAnalytics:
    """Test GET /v1/analytics endpoint."""

    def test_analytics_empty(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/analytics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_records"] == 0
        assert data["total_spend_usd"] == 0.0

    def test_analytics_with_data(
        self,
        client: TestClient,
        calculated_records: List[Dict[str, Any]],
    ) -> None:
        resp = client.get(f"{PREFIX}/v1/analytics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_records"] > 0
        assert "top_categories" in data
        assert "top_vendors" in data
        assert "scope3_breakdown" in data


# ===================================================================
# TestGetHotspots
# ===================================================================


class TestGetHotspots:
    """Test GET /v1/analytics/hotspots endpoint."""

    def test_hotspots_empty(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/analytics/hotspots")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["hotspots"] == []

    def test_hotspots_with_data(
        self,
        client: TestClient,
        calculated_records: List[Dict[str, Any]],
    ) -> None:
        resp = client.get(f"{PREFIX}/v1/analytics/hotspots")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] > 0

    def test_hotspots_top_n(
        self,
        client: TestClient,
        calculated_records: List[Dict[str, Any]],
    ) -> None:
        resp = client.get(f"{PREFIX}/v1/analytics/hotspots?top_n=2")
        assert resp.status_code == 200
        assert resp.json()["count"] <= 2


# ===================================================================
# TestGenerateReport
# ===================================================================


class TestGenerateReport:
    """Test POST /v1/reports endpoint."""

    def test_summary_json(self, client: TestClient) -> None:
        body = {"report_type": "summary", "format": "json"}
        resp = client.post(f"{PREFIX}/v1/reports", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["report_type"] == "summary"
        assert data["format"] == "json"
        assert len(data["provenance_hash"]) == 64

    def test_detailed_json(
        self,
        client: TestClient,
        ingested_records: Dict[str, Any],
    ) -> None:
        body = {"report_type": "detailed", "format": "json"}
        resp = client.post(f"{PREFIX}/v1/reports", json=body)
        assert resp.status_code == 200
        content = resp.json()["content"]
        assert content is not None
        assert "records" in content

    def test_csv_format(self, client: TestClient) -> None:
        body = {"report_type": "summary", "format": "csv"}
        resp = client.post(f"{PREFIX}/v1/reports", json=body)
        assert resp.status_code == 200
        assert resp.json()["format"] == "csv"

    def test_default_format(self, client: TestClient) -> None:
        resp = client.post(f"{PREFIX}/v1/reports", json={})
        assert resp.status_code == 200
        assert resp.json()["report_type"] == "summary"
        assert resp.json()["format"] == "json"

    def test_emissions_report(
        self,
        client: TestClient,
        calculated_records: List[Dict[str, Any]],
    ) -> None:
        body = {"report_type": "emissions", "format": "json"}
        resp = client.post(f"{PREFIX}/v1/reports", json=body)
        assert resp.status_code == 200
        content = resp.json()["content"]
        assert "emission_calculations" in content

    def test_scope3_report(
        self,
        client: TestClient,
        mapped_records: List[Dict[str, Any]],
    ) -> None:
        body = {"report_type": "scope3", "format": "json"}
        resp = client.post(f"{PREFIX}/v1/reports", json=body)
        assert resp.status_code == 200
        content = resp.json()["content"]
        assert "scope3_assignments" in content


# ===================================================================
# TestHealthCheck
# ===================================================================


class TestHealthCheck:
    """Test GET /health endpoint."""

    def test_health_200(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/health")
        assert resp.status_code == 200

    def test_health_status_healthy(self, client: TestClient) -> None:
        data = client.get(f"{PREFIX}/health").json()
        assert data["status"] == "healthy"
        assert data["started"] is True

    def test_health_service_name(self, client: TestClient) -> None:
        data = client.get(f"{PREFIX}/health").json()
        assert data["service"] == "spend-categorizer"

    def test_health_counts(
        self,
        client: TestClient,
        ingested_records: Dict[str, Any],
    ) -> None:
        data = client.get(f"{PREFIX}/health").json()
        assert data["records"] == 5

    def test_health_all_fields(self, client: TestClient) -> None:
        data = client.get(f"{PREFIX}/health").json()
        expected_keys = [
            "status", "service", "started", "records",
            "classifications", "scope3_assignments",
            "emission_calculations", "rules", "reports",
            "provenance_entries", "prometheus_available",
        ]
        for key in expected_keys:
            assert key in data


# ===================================================================
# TestServiceNotConfigured
# ===================================================================


class TestServiceNotConfigured:
    """Test endpoints return 503 when service is not configured."""

    @pytest.fixture
    def unconfigured_client(self) -> TestClient:
        """Create a TestClient without service configured."""
        app = FastAPI()
        app.include_router(router)
        return TestClient(app, raise_server_exceptions=False)

    def test_ingest_503(self, unconfigured_client: TestClient) -> None:
        resp = unconfigured_client.post(
            f"{PREFIX}/v1/ingest",
            json={"records": [{"vendor_name": "V", "amount": 1}]},
        )
        assert resp.status_code == 503

    def test_list_records_503(self, unconfigured_client: TestClient) -> None:
        resp = unconfigured_client.get(f"{PREFIX}/v1/records")
        assert resp.status_code == 503

    def test_get_record_503(self, unconfigured_client: TestClient) -> None:
        resp = unconfigured_client.get(f"{PREFIX}/v1/records/some-id")
        assert resp.status_code == 503

    def test_classify_503(self, unconfigured_client: TestClient) -> None:
        resp = unconfigured_client.post(
            f"{PREFIX}/v1/classify",
            json={"record_id": "x"},
        )
        assert resp.status_code == 503

    def test_classify_batch_503(self, unconfigured_client: TestClient) -> None:
        resp = unconfigured_client.post(
            f"{PREFIX}/v1/classify/batch",
            json={"record_ids": ["x"]},
        )
        assert resp.status_code == 503

    def test_map_scope3_503(self, unconfigured_client: TestClient) -> None:
        resp = unconfigured_client.post(
            f"{PREFIX}/v1/map-scope3",
            json={"record_id": "x"},
        )
        assert resp.status_code == 503

    def test_map_scope3_batch_503(self, unconfigured_client: TestClient) -> None:
        resp = unconfigured_client.post(
            f"{PREFIX}/v1/map-scope3/batch",
            json={"record_ids": ["x"]},
        )
        assert resp.status_code == 503

    def test_calculate_emissions_503(self, unconfigured_client: TestClient) -> None:
        resp = unconfigured_client.post(
            f"{PREFIX}/v1/calculate-emissions",
            json={"record_id": "x"},
        )
        assert resp.status_code == 503

    def test_calculate_emissions_batch_503(self, unconfigured_client: TestClient) -> None:
        resp = unconfigured_client.post(
            f"{PREFIX}/v1/calculate-emissions/batch",
            json={"record_ids": ["x"]},
        )
        assert resp.status_code == 503

    def test_list_emission_factors_503(self, unconfigured_client: TestClient) -> None:
        resp = unconfigured_client.get(f"{PREFIX}/v1/emission-factors")
        assert resp.status_code == 503

    def test_get_emission_factor_503(self, unconfigured_client: TestClient) -> None:
        resp = unconfigured_client.get(f"{PREFIX}/v1/emission-factors/43000000")
        assert resp.status_code == 503

    def test_create_rule_503(self, unconfigured_client: TestClient) -> None:
        resp = unconfigured_client.post(
            f"{PREFIX}/v1/rules",
            json={
                "name": "R",
                "taxonomy_code": "44000000",
                "conditions": {"keywords": ["test"]},
            },
        )
        assert resp.status_code == 503

    def test_list_rules_503(self, unconfigured_client: TestClient) -> None:
        resp = unconfigured_client.get(f"{PREFIX}/v1/rules")
        assert resp.status_code == 503

    def test_update_rule_503(self, unconfigured_client: TestClient) -> None:
        resp = unconfigured_client.put(
            f"{PREFIX}/v1/rules/some-id",
            json={"name": "X"},
        )
        assert resp.status_code == 503

    def test_delete_rule_503(self, unconfigured_client: TestClient) -> None:
        resp = unconfigured_client.delete(f"{PREFIX}/v1/rules/some-id")
        assert resp.status_code == 503

    def test_analytics_503(self, unconfigured_client: TestClient) -> None:
        resp = unconfigured_client.get(f"{PREFIX}/v1/analytics")
        assert resp.status_code == 503

    def test_hotspots_503(self, unconfigured_client: TestClient) -> None:
        resp = unconfigured_client.get(f"{PREFIX}/v1/analytics/hotspots")
        assert resp.status_code == 503

    def test_generate_report_503(self, unconfigured_client: TestClient) -> None:
        resp = unconfigured_client.post(
            f"{PREFIX}/v1/reports",
            json={"report_type": "summary"},
        )
        assert resp.status_code == 503

    def test_health_503(self, unconfigured_client: TestClient) -> None:
        resp = unconfigured_client.get(f"{PREFIX}/health")
        assert resp.status_code == 503


# ===================================================================
# TestFullAPIWorkflow
# ===================================================================


class TestFullAPIWorkflow:
    """Test end-to-end workflow through all 20 API endpoints."""

    def test_full_workflow(self, client: TestClient, service: SpendCategorizerService) -> None:
        # 1. Ingest records
        ingest_resp = client.post(
            f"{PREFIX}/v1/ingest",
            json={"records": _sample_records(3), "source": "e2e"},
        )
        assert ingest_resp.status_code == 200
        records = ingest_resp.json()["records"]
        assert len(records) == 3

        # 2. (File ingest tested separately due to file dependency)

        # 3. List records
        list_resp = client.get(f"{PREFIX}/v1/records?source=e2e")
        assert list_resp.status_code == 200
        assert list_resp.json()["count"] == 3

        # 4. Get single record
        rid = records[0]["record_id"]
        get_resp = client.get(f"{PREFIX}/v1/records/{rid}")
        assert get_resp.status_code == 200
        assert get_resp.json()["record_id"] == rid

        # 5. Classify single record
        classify_resp = client.post(
            f"{PREFIX}/v1/classify",
            json={"record_id": rid},
        )
        assert classify_resp.status_code == 200

        # 6. Batch classify remaining
        rids = [r["record_id"] for r in records[1:]]
        batch_classify_resp = client.post(
            f"{PREFIX}/v1/classify/batch",
            json={"record_ids": rids},
        )
        assert batch_classify_resp.status_code == 200

        # 7. Map Scope 3 for records with taxonomy_code
        all_rids = [r["record_id"] for r in records]
        mapped_rids = []
        for r_id in all_rids:
            rec = service.get_record(r_id)
            if rec and rec.taxonomy_code:
                map_resp = client.post(
                    f"{PREFIX}/v1/map-scope3",
                    json={"record_id": r_id},
                )
                if map_resp.status_code == 200:
                    mapped_rids.append(r_id)

        # 8. Batch map Scope 3 (already mapped, so skip or test with empty)
        batch_map_resp = client.post(
            f"{PREFIX}/v1/map-scope3/batch",
            json={"record_ids": []},
        )
        assert batch_map_resp.status_code == 200

        # 9. Calculate emissions for mapped records
        for r_id in mapped_rids[:1]:
            calc_resp = client.post(
                f"{PREFIX}/v1/calculate-emissions",
                json={"record_id": r_id},
            )
            assert calc_resp.status_code == 200

        # 10. Batch calculate emissions
        remaining = mapped_rids[1:]
        if remaining:
            batch_calc_resp = client.post(
                f"{PREFIX}/v1/calculate-emissions/batch",
                json={"record_ids": remaining},
            )
            assert batch_calc_resp.status_code == 200

        # 11. List emission factors
        factors_resp = client.get(f"{PREFIX}/v1/emission-factors")
        assert factors_resp.status_code == 200
        assert factors_resp.json()["count"] > 0

        # 12. Get single emission factor
        factor_resp = client.get(f"{PREFIX}/v1/emission-factors/43000000")
        assert factor_resp.status_code == 200

        # 13. Create rule
        rule_resp = client.post(
            f"{PREFIX}/v1/rules",
            json={
                "name": "E2E Rule",
                "taxonomy_code": "44000000",
                "conditions": {"keywords": ["e2e"]},
            },
        )
        assert rule_resp.status_code == 200
        rule_id = rule_resp.json()["rule_id"]

        # 14. List rules
        rules_resp = client.get(f"{PREFIX}/v1/rules")
        assert rules_resp.status_code == 200
        assert rules_resp.json()["count"] >= 1

        # 15. Update rule
        update_resp = client.put(
            f"{PREFIX}/v1/rules/{rule_id}",
            json={"description": "Updated in E2E test"},
        )
        assert update_resp.status_code == 200

        # 16. Delete rule
        delete_resp = client.delete(f"{PREFIX}/v1/rules/{rule_id}")
        assert delete_resp.status_code == 200

        # 17. Get analytics
        analytics_resp = client.get(f"{PREFIX}/v1/analytics")
        assert analytics_resp.status_code == 200
        assert analytics_resp.json()["total_records"] == 3

        # 18. Get hotspots
        hotspots_resp = client.get(f"{PREFIX}/v1/analytics/hotspots")
        assert hotspots_resp.status_code == 200

        # 19. Generate report
        report_resp = client.post(
            f"{PREFIX}/v1/reports",
            json={"report_type": "summary", "format": "json"},
        )
        assert report_resp.status_code == 200
        assert report_resp.json()["record_count"] == 3

        # 20. Health check
        health_resp = client.get(f"{PREFIX}/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "healthy"
