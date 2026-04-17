# -*- coding: utf-8 -*-
"""Tests for API endpoint logic (F032-F036)."""

from __future__ import annotations

import json

import pytest

from greenlang.factors.api_endpoints import (
    SearchV2Request,
    SearchV2Result,
    VALID_SORT_FIELDS,
    build_audit_bundle,
    bulk_export_factors,
    bulk_export_manifest,
    cache_control_for_status,
    check_etag_match,
    compute_etag,
    compute_etag_from_dict,
    diff_factor_between_editions,
    search_v2,
    _diff_dicts,
)

# ---- Fixtures ----

@pytest.fixture
def emission_db():
    from greenlang.data.emission_factor_database import EmissionFactorDatabase
    return EmissionFactorDatabase(enable_cache=False)


@pytest.fixture
def memory_repo(emission_db):
    from greenlang.factors.catalog_repository import MemoryFactorCatalogRepository
    return MemoryFactorCatalogRepository("test-v1", "test", emission_db)


@pytest.fixture
def edition_id():
    return "test-v1"


@pytest.fixture
def sample_factor_id(memory_repo, edition_id):
    factors, _ = memory_repo.list_factors(edition_id, limit=1)
    return factors[0].factor_id


# ---- F032: Audit bundle ----

def test_audit_bundle_returns_dict(memory_repo, edition_id, sample_factor_id):
    bundle = build_audit_bundle(memory_repo, edition_id, sample_factor_id)
    assert bundle is not None
    assert bundle["factor_id"] == sample_factor_id
    assert bundle["edition_id"] == edition_id


def test_audit_bundle_has_provenance(memory_repo, edition_id, sample_factor_id):
    bundle = build_audit_bundle(memory_repo, edition_id, sample_factor_id)
    assert "provenance" in bundle
    assert "source_org" in bundle["provenance"]
    assert "methodology" in bundle["provenance"]


def test_audit_bundle_has_license(memory_repo, edition_id, sample_factor_id):
    bundle = build_audit_bundle(memory_repo, edition_id, sample_factor_id)
    assert "license_info" in bundle
    assert "redistribution_allowed" in bundle["license_info"]


def test_audit_bundle_has_verification_chain(memory_repo, edition_id, sample_factor_id):
    bundle = build_audit_bundle(memory_repo, edition_id, sample_factor_id)
    chain = bundle["verification_chain"]
    assert chain["algorithm"] == "SHA-256"
    assert len(chain["payload_sha256"]) == 64  # SHA-256 hex length
    assert chain["content_hash"] == bundle["content_hash"]


def test_audit_bundle_has_quality(memory_repo, edition_id, sample_factor_id):
    bundle = build_audit_bundle(memory_repo, edition_id, sample_factor_id)
    assert "quality" in bundle
    assert "dqs_overall" in bundle["quality"]
    assert bundle["quality"]["dqs_overall"] > 0


def test_audit_bundle_not_found(memory_repo, edition_id):
    bundle = build_audit_bundle(memory_repo, edition_id, "EF:NONEXISTENT:xyz")
    assert bundle is None


def test_audit_bundle_includes_optional_fields(memory_repo, edition_id, sample_factor_id):
    bundle = build_audit_bundle(
        memory_repo, edition_id, sample_factor_id,
        raw_artifact_uri="s3://bucket/raw.json",
        parser_log="CBAM parser v1",
        qa_errors=["Q3: missing citation"],
        reviewer_decision="approved",
    )
    assert bundle["raw_artifact_uri"] == "s3://bucket/raw.json"
    assert bundle["parser_log"] == "CBAM parser v1"
    assert bundle["qa_errors"] == ["Q3: missing citation"]
    assert bundle["reviewer_decision"] == "approved"


def test_audit_bundle_payload_sha256_deterministic(memory_repo, edition_id, sample_factor_id):
    b1 = build_audit_bundle(memory_repo, edition_id, sample_factor_id)
    b2 = build_audit_bundle(memory_repo, edition_id, sample_factor_id)
    assert b1["payload_sha256"] == b2["payload_sha256"]


# ---- F033: Bulk export ----

def test_bulk_export_returns_list(memory_repo, edition_id):
    rows = bulk_export_factors(memory_repo, edition_id)
    assert isinstance(rows, list)
    assert len(rows) > 0


def test_bulk_export_rows_are_dicts(memory_repo, edition_id):
    rows = bulk_export_factors(memory_repo, edition_id, max_rows=5)
    for row in rows:
        assert isinstance(row, dict)
        assert "factor_id" in row


def test_bulk_export_max_rows(memory_repo, edition_id):
    rows = bulk_export_factors(memory_repo, edition_id, max_rows=3)
    assert len(rows) <= 3


def test_bulk_export_geography_filter(memory_repo, edition_id):
    rows = bulk_export_factors(memory_repo, edition_id, geography="US")
    for row in rows:
        assert row.get("geography") == "US"


def test_bulk_export_fuel_type_filter(memory_repo, edition_id):
    rows = bulk_export_factors(memory_repo, edition_id, fuel_type="diesel")
    for row in rows:
        assert row.get("fuel_type", "").lower() == "diesel"


def test_bulk_export_manifest(memory_repo, edition_id):
    rows = bulk_export_factors(memory_repo, edition_id, max_rows=10)
    manifest = bulk_export_manifest(memory_repo, edition_id, len(rows))
    assert manifest["edition_id"] == edition_id
    assert manifest["exported_rows"] == len(rows)
    assert manifest["format"] == "json_lines"


def test_bulk_export_empty_edition(memory_repo):
    rows = bulk_export_factors(memory_repo, "nonexistent-edition")
    assert rows == []


# ---- F034: Factor diff ----

def test_diff_same_edition_unchanged(memory_repo, edition_id, sample_factor_id):
    result = diff_factor_between_editions(
        memory_repo, sample_factor_id, edition_id, edition_id
    )
    assert result["status"] == "unchanged"
    assert result["left_exists"]
    assert result["right_exists"]
    assert result["changes"] == []


def test_diff_factor_not_found(memory_repo, edition_id):
    result = diff_factor_between_editions(
        memory_repo, "EF:NONEXISTENT:xyz", edition_id, edition_id
    )
    assert result["status"] == "not_found"


def test_diff_dicts_simple():
    left = {"a": 1, "b": 2}
    right = {"a": 1, "b": 3}
    changes = _diff_dicts(left, right)
    assert len(changes) == 1
    assert changes[0]["field"] == "b"
    assert changes[0]["type"] == "changed"
    assert changes[0]["old_value"] == 2
    assert changes[0]["new_value"] == 3


def test_diff_dicts_added():
    left = {"a": 1}
    right = {"a": 1, "b": 2}
    changes = _diff_dicts(left, right)
    assert any(c["type"] == "added" and c["field"] == "b" for c in changes)


def test_diff_dicts_removed():
    left = {"a": 1, "b": 2}
    right = {"a": 1}
    changes = _diff_dicts(left, right)
    assert any(c["type"] == "removed" and c["field"] == "b" for c in changes)


def test_diff_dicts_nested():
    left = {"top": {"inner": 1}}
    right = {"top": {"inner": 2}}
    changes = _diff_dicts(left, right)
    assert any(c["field"] == "top.inner" for c in changes)


def test_diff_dicts_identical():
    d = {"a": 1, "b": {"c": 3}}
    assert _diff_dicts(d, d) == []


def test_diff_result_structure(memory_repo, edition_id, sample_factor_id):
    result = diff_factor_between_editions(
        memory_repo, sample_factor_id, edition_id, edition_id
    )
    assert "factor_id" in result
    assert "left_edition" in result
    assert "right_edition" in result
    assert "status" in result


# ---- F035: Search v2 ----

def test_search_v2_basic(memory_repo, edition_id):
    req = SearchV2Request(query="diesel")
    result = search_v2(memory_repo, edition_id, req)
    assert isinstance(result, SearchV2Result)
    assert result.total_count > 0
    assert len(result.factors) > 0


def test_search_v2_to_dict(memory_repo, edition_id):
    req = SearchV2Request(query="diesel")
    result = search_v2(memory_repo, edition_id, req)
    d = result.to_dict()
    assert d["query"] == "diesel"
    assert "total_count" in d
    assert "offset" in d


def test_search_v2_limit(memory_repo, edition_id):
    req = SearchV2Request(query="diesel", limit=2)
    result = search_v2(memory_repo, edition_id, req)
    assert len(result.factors) <= 2


def test_search_v2_offset(memory_repo, edition_id):
    req_all = SearchV2Request(query="diesel", limit=50)
    result_all = search_v2(memory_repo, edition_id, req_all)
    if result_all.total_count > 1:
        req_off = SearchV2Request(query="diesel", offset=1, limit=50)
        result_off = search_v2(memory_repo, edition_id, req_off)
        assert result_off.offset == 1
        assert len(result_off.factors) == len(result_all.factors) - 1


def test_search_v2_sort_dqs(memory_repo, edition_id):
    req = SearchV2Request(query="diesel", sort_by="dqs_score", sort_order="desc")
    result = search_v2(memory_repo, edition_id, req)
    if len(result.factors) >= 2:
        scores = [f["dqs_score"] for f in result.factors]
        assert scores == sorted(scores, reverse=True)


def test_search_v2_sort_co2e(memory_repo, edition_id):
    req = SearchV2Request(query="diesel", sort_by="co2e_total", sort_order="asc")
    result = search_v2(memory_repo, edition_id, req)
    if len(result.factors) >= 2:
        vals = [f["co2e_per_unit"] for f in result.factors]
        assert vals == sorted(vals)


def test_search_v2_dqs_min(memory_repo, edition_id):
    req = SearchV2Request(query="diesel", dqs_min=4.0)
    result = search_v2(memory_repo, edition_id, req)
    for f in result.factors:
        assert f["dqs_score"] >= 4.0


def test_search_v2_invalid_sort_defaults_relevance(memory_repo, edition_id):
    req = SearchV2Request(query="diesel", sort_by="invalid_field")
    result = search_v2(memory_repo, edition_id, req)
    assert result.sort_by == "relevance"


def test_search_v2_valid_sort_fields():
    assert "relevance" in VALID_SORT_FIELDS
    assert "dqs_score" in VALID_SORT_FIELDS
    assert "co2e_total" in VALID_SORT_FIELDS
    assert "source_year" in VALID_SORT_FIELDS
    assert "factor_id" in VALID_SORT_FIELDS


def test_search_v2_factor_dict_fields(memory_repo, edition_id):
    req = SearchV2Request(query="diesel", limit=1)
    result = search_v2(memory_repo, edition_id, req)
    if result.factors:
        f = result.factors[0]
        assert "factor_id" in f
        assert "fuel_type" in f
        assert "geography" in f
        assert "co2e_per_unit" in f
        assert "dqs_score" in f


# ---- F036: ETag / Cache-Control ----

def test_compute_etag(emission_db):
    factor = list(emission_db.factors.values())[0]
    etag = compute_etag(factor)
    assert etag.startswith('"')
    assert etag.endswith('"')
    assert len(etag) > 10


def test_compute_etag_deterministic(emission_db):
    factor = list(emission_db.factors.values())[0]
    assert compute_etag(factor) == compute_etag(factor)


def test_compute_etag_from_dict():
    data = {"a": 1, "b": 2}
    etag = compute_etag_from_dict(data)
    assert etag.startswith('"')
    assert len(etag) == 66  # 64 hex chars + 2 quotes


def test_compute_etag_from_dict_deterministic():
    data = {"a": 1, "b": 2}
    assert compute_etag_from_dict(data) == compute_etag_from_dict(data)


def test_cache_control_certified():
    assert cache_control_for_status("certified") == "public, max-age=3600"


def test_cache_control_preview():
    assert cache_control_for_status("preview") == "public, max-age=600"


def test_cache_control_connector():
    assert cache_control_for_status("connector_only") == "private, max-age=600"


def test_cache_control_deprecated():
    assert cache_control_for_status("deprecated") == "no-cache"


def test_cache_control_none_defaults_certified():
    assert cache_control_for_status(None) == "public, max-age=3600"


def test_check_etag_match_true():
    assert check_etag_match('"abc123"', '"abc123"')


def test_check_etag_match_false():
    assert not check_etag_match('"abc123"', '"def456"')


def test_check_etag_match_none():
    assert not check_etag_match(None, '"abc123"')


def test_check_etag_match_weak():
    assert check_etag_match('W/"abc123"', '"abc123"')
