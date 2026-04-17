# -*- coding: utf-8 -*-
"""Tests for bulk ingestion pipeline (F019)."""

from __future__ import annotations

import json

import pytest

from greenlang.factors.ingestion.bulk_ingest import bulk_ingest, IngestionResult
from greenlang.factors.catalog_repository import SqliteFactorCatalogRepository


@pytest.fixture
def epa_json_file(tmp_path):
    data = {
        "metadata": {"source": "EPA", "version": "2024"},
        "stationary_combustion": [
            {"fuel_type": "Natural Gas", "unit": "scf", "co2_factor": 0.05444, "ch4_factor": 0.00000103, "n2o_factor": 0.0000001},
            {"fuel_type": "Diesel Fuel", "unit": "gallon", "co2_factor": 10.18, "ch4_factor": 0.00041, "n2o_factor": 0.000082},
        ],
    }
    p = tmp_path / "epa_hub.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def egrid_json_file(tmp_path):
    data = {
        "metadata": {"source": "eGRID", "version": "2022"},
        "subregions": [
            {"acronym": "CAMX", "co2_lb_mwh": 496.0, "ch4_lb_mwh": 0.038, "n2o_lb_mwh": 0.005},
            {"acronym": "ERCT", "co2_lb_mwh": 835.5, "ch4_lb_mwh": 0.053, "n2o_lb_mwh": 0.009},
        ],
    }
    p = tmp_path / "egrid.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def cbam_json_file(tmp_path):
    data = {
        "metadata": {"source": "EU Commission", "version": "2024"},
        "products": {
            "cement": {
                "categories": [{"name": "clinker"}],
                "by_country": {"CN": {"direct_emissions_factor": 0.82, "indirect_emissions_factor": 0.05}},
            },
        },
    }
    p = tmp_path / "cbam.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_single_source_ingestion(tmp_path, epa_json_file):
    sqlite_path = tmp_path / "catalog.db"
    result = bulk_ingest(
        [("epa_hub", epa_json_file)],
        sqlite_path,
        "2024.04.0",
        label="Test ingest",
    )
    assert result.total_ingested == 2
    assert result.total_rejected == 0
    assert result.per_source["epa_hub"] == 2
    assert len(result.errors) == 0


def test_multi_source_ingestion(tmp_path, epa_json_file, egrid_json_file):
    sqlite_path = tmp_path / "catalog.db"
    result = bulk_ingest(
        [("epa_hub", epa_json_file), ("egrid", egrid_json_file)],
        sqlite_path,
        "2024.04.0",
    )
    assert result.total_ingested == 4  # 2 EPA + 2 eGRID
    assert result.per_source["epa_hub"] == 2
    assert result.per_source["egrid"] == 2


def test_three_source_ingestion(tmp_path, epa_json_file, egrid_json_file, cbam_json_file):
    sqlite_path = tmp_path / "catalog.db"
    result = bulk_ingest(
        [
            ("epa_hub", epa_json_file),
            ("egrid", egrid_json_file),
            ("eu_cbam", cbam_json_file),
        ],
        sqlite_path,
        "2024.04.0",
    )
    assert result.total_ingested == 5  # 2 + 2 + 1
    assert len(result.per_source) == 3


def test_unknown_source_skipped(tmp_path, epa_json_file):
    sqlite_path = tmp_path / "catalog.db"
    fake_path = tmp_path / "fake.json"
    fake_path.write_text("{}", encoding="utf-8")
    result = bulk_ingest(
        [("unknown_source", fake_path), ("epa_hub", epa_json_file)],
        sqlite_path,
        "2024.04.0",
        skip_unknown_sources=True,
    )
    assert result.total_ingested == 2
    assert len(result.warnings) > 0
    assert "No parser" in result.warnings[0]


def test_missing_file_error(tmp_path):
    sqlite_path = tmp_path / "catalog.db"
    result = bulk_ingest(
        [("epa_hub", tmp_path / "nonexistent.json")],
        sqlite_path,
        "2024.04.0",
    )
    assert result.total_ingested == 0
    assert len(result.errors) > 0


def test_invalid_json_error(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not json!", encoding="utf-8")
    sqlite_path = tmp_path / "catalog.db"
    result = bulk_ingest(
        [("epa_hub", bad_file)],
        sqlite_path,
        "2024.04.0",
    )
    assert result.total_ingested == 0
    assert len(result.errors) > 0


def test_catalog_has_factors_after_ingest(tmp_path, epa_json_file):
    sqlite_path = tmp_path / "catalog.db"
    bulk_ingest(
        [("epa_hub", epa_json_file)],
        sqlite_path,
        "2024.04.0",
    )
    repo = SqliteFactorCatalogRepository(sqlite_path)
    factors = repo.list_factors("2024.04.0")
    assert len(factors) == 2


def test_edition_created_after_ingest(tmp_path, epa_json_file):
    sqlite_path = tmp_path / "catalog.db"
    bulk_ingest(
        [("epa_hub", epa_json_file)],
        sqlite_path,
        "2024.04.0",
        status="preview",
    )
    repo = SqliteFactorCatalogRepository(sqlite_path)
    editions = repo.list_editions()
    assert any(getattr(e, "edition_id", None) == "2024.04.0" for e in editions)


def test_result_to_dict():
    r = IngestionResult(edition_id="test", total_ingested=10, total_rejected=2)
    d = r.to_dict()
    assert d["edition_id"] == "test"
    assert d["total_ingested"] == 10
    assert d["total_rejected"] == 2


def test_empty_sources(tmp_path):
    sqlite_path = tmp_path / "catalog.db"
    result = bulk_ingest([], sqlite_path, "2024.04.0")
    assert result.total_ingested == 0
    assert len(result.errors) == 0
