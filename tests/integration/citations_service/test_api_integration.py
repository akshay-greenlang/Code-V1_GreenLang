# -*- coding: utf-8 -*-
"""
API Integration Tests for Citations Service (AGENT-FOUND-005)

Tests the citations API endpoints with a simulated TestClient,
validating CRUD operations, verification, evidence packaging,
export/import, health, and error responses.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Self-contained TestClient implementation
# ---------------------------------------------------------------------------


class CitationRecord:
    def __init__(self, citation_id, citation_type="emission_factor",
                 source_authority="defra", title="", version=None,
                 effective_date="2024-01-01", expiration_date=None,
                 verification_status="unverified",
                 regulatory_frameworks=None, key_values=None):
        self.citation_id = citation_id
        self.citation_type = citation_type
        self.source_authority = source_authority
        self.title = title
        self.version = version
        self.effective_date = effective_date
        self.expiration_date = expiration_date
        self.verification_status = verification_status
        self.regulatory_frameworks = regulatory_frameworks or []
        self.key_values = key_values or {}
        self.record_version = 1


class EvidencePackageRecord:
    def __init__(self, package_id, name, description=""):
        self.package_id = package_id
        self.name = name
        self.description = description
        self.evidence_items = []
        self.citation_ids = []
        self.is_finalized = False
        self.package_hash = None


class CitationsTestClient:
    """Simulated API client for citations service."""

    def __init__(self):
        self._citations: Dict[str, CitationRecord] = {}
        self._packages: Dict[str, EvidencePackageRecord] = {}
        self._pkg_counter = 0

    # ---- Citations CRUD ----

    def post_citation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cid = data.get("citation_id", f"cid-{len(self._citations)+1}")
        if not data.get("citation_type"):
            return {"error": "citation_type required", "status": 400}
        if cid in self._citations:
            return {"error": f"Citation '{cid}' already exists", "status": 409}
        r = CitationRecord(cid, data.get("citation_type", "emission_factor"),
                           data.get("source_authority", "defra"),
                           title=data.get("title", ""),
                           version=data.get("version"),
                           effective_date=data.get("effective_date", "2024-01-01"),
                           expiration_date=data.get("expiration_date"),
                           regulatory_frameworks=data.get("regulatory_frameworks"),
                           key_values=data.get("key_values"))
        self._citations[cid] = r
        return {"data": {"citation_id": cid, "title": r.title}, "status": 201}

    def get_citation(self, cid: str) -> Dict[str, Any]:
        if cid not in self._citations:
            return {"error": f"Citation '{cid}' not found", "status": 404}
        r = self._citations[cid]
        return {"data": {"citation_id": cid, "title": r.title,
                         "verification_status": r.verification_status}, "status": 200}

    def put_citation(self, cid: str, data: Dict[str, Any]) -> Dict[str, Any]:
        if cid not in self._citations:
            return {"error": "not found", "status": 404}
        r = self._citations[cid]
        if "title" in data:
            r.title = data["title"]
        if "verification_status" in data:
            r.verification_status = data["verification_status"]
        if "key_values" in data:
            r.key_values = data["key_values"]
        r.record_version += 1
        return {"data": {"citation_id": cid, "version": r.record_version}, "status": 200}

    def delete_citation(self, cid: str) -> Dict[str, Any]:
        if cid not in self._citations:
            return {"error": "not found", "status": 404}
        del self._citations[cid]
        return {"status": 204}

    def list_citations(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        results = list(self._citations.values())
        if params:
            if params.get("citation_type"):
                results = [r for r in results if r.citation_type == params["citation_type"]]
        return {"data": [{"citation_id": r.citation_id, "title": r.title} for r in results],
                "status": 200}

    # ---- Verification ----

    def post_verify(self, cid: str) -> Dict[str, Any]:
        if cid not in self._citations:
            return {"error": "not found", "status": 404}
        r = self._citations[cid]
        if r.expiration_date and r.expiration_date < "2025-06-15":
            r.verification_status = "expired"
        elif r.source_authority in ("defra", "epa") and not r.version:
            r.verification_status = "unverified"
        else:
            r.verification_status = "verified"
        return {"data": {"citation_id": cid, "status": r.verification_status}, "status": 200}

    # ---- Evidence Packages ----

    def post_package(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self._pkg_counter += 1
        pid = f"pkg-{self._pkg_counter:04d}"
        if not data.get("name"):
            return {"error": "name required", "status": 400}
        pkg = EvidencePackageRecord(pid, data["name"], data.get("description", ""))
        self._packages[pid] = pkg
        return {"data": {"package_id": pid, "name": pkg.name}, "status": 201}

    def get_package(self, pid: str) -> Dict[str, Any]:
        if pid not in self._packages:
            return {"error": "not found", "status": 404}
        pkg = self._packages[pid]
        return {"data": {"package_id": pid, "name": pkg.name,
                         "is_finalized": pkg.is_finalized}, "status": 200}

    def post_add_evidence(self, pid: str, data: Dict[str, Any]) -> Dict[str, Any]:
        if pid not in self._packages:
            return {"error": "not found", "status": 404}
        pkg = self._packages[pid]
        if pkg.is_finalized:
            return {"error": "package is finalized", "status": 409}
        pkg.evidence_items.append(data)
        return {"data": {"evidence_count": len(pkg.evidence_items)}, "status": 200}

    def post_add_citation(self, pid: str, cid: str) -> Dict[str, Any]:
        if pid not in self._packages:
            return {"error": "not found", "status": 404}
        pkg = self._packages[pid]
        if cid not in pkg.citation_ids:
            pkg.citation_ids.append(cid)
        return {"data": {"citation_count": len(pkg.citation_ids)}, "status": 200}

    def post_finalize(self, pid: str) -> Dict[str, Any]:
        if pid not in self._packages:
            return {"error": "not found", "status": 404}
        pkg = self._packages[pid]
        if pkg.is_finalized:
            return {"error": "already finalized", "status": 409}
        content = json.dumps({"name": pkg.name, "items": len(pkg.evidence_items)}, sort_keys=True)
        pkg.package_hash = hashlib.sha256(content.encode()).hexdigest()
        pkg.is_finalized = True
        return {"data": {"package_hash": pkg.package_hash}, "status": 200}

    # ---- Export/Import ----

    def post_export(self, fmt: str = "json") -> Dict[str, Any]:
        data = [{"citation_id": r.citation_id, "title": r.title, "citation_type": r.citation_type}
                for r in self._citations.values()]
        if fmt == "json":
            return {"data": json.dumps(data, indent=2), "status": 200}
        return {"data": data, "status": 200}

    def post_import(self, data: Dict[str, Any]) -> Dict[str, Any]:
        imported = 0
        for item in data.get("citations", []):
            cid = item.get("citation_id", "")
            if cid not in self._citations:
                self._citations[cid] = CitationRecord(cid, item.get("citation_type", "emission_factor"),
                                                       title=item.get("title", ""))
                imported += 1
        return {"data": {"imported": imported}, "status": 200}

    # ---- Health ----

    def get_health(self) -> Dict[str, Any]:
        return {"status": "healthy", "version": "1.0.0",
                "citations_count": len(self._citations),
                "packages_count": len(self._packages)}


@pytest.fixture
def client():
    return CitationsTestClient()


# ===========================================================================
# Test Classes
# ===========================================================================


class TestCitationsCRUD:
    """Test full CRUD via TestClient."""

    def test_create(self, client):
        resp = client.post_citation({"citation_id": "defra-2024", "citation_type": "emission_factor",
                                      "title": "DEFRA 2024"})
        assert resp["status"] == 201

    def test_get(self, client):
        client.post_citation({"citation_id": "defra-2024", "citation_type": "emission_factor",
                               "title": "DEFRA 2024"})
        resp = client.get_citation("defra-2024")
        assert resp["status"] == 200
        assert resp["data"]["title"] == "DEFRA 2024"

    def test_update(self, client):
        client.post_citation({"citation_id": "defra-2024", "citation_type": "emission_factor"})
        resp = client.put_citation("defra-2024", {"title": "Updated Title"})
        assert resp["status"] == 200

    def test_delete(self, client):
        client.post_citation({"citation_id": "defra-2024", "citation_type": "emission_factor"})
        resp = client.delete_citation("defra-2024")
        assert resp["status"] == 204

    def test_list(self, client):
        client.post_citation({"citation_id": "c1", "citation_type": "emission_factor"})
        client.post_citation({"citation_id": "c2", "citation_type": "regulatory"})
        resp = client.list_citations()
        assert resp["status"] == 200
        assert len(resp["data"]) == 2

    def test_list_filtered(self, client):
        client.post_citation({"citation_id": "c1", "citation_type": "emission_factor"})
        client.post_citation({"citation_id": "c2", "citation_type": "regulatory"})
        resp = client.list_citations({"citation_type": "emission_factor"})
        assert len(resp["data"]) == 1


class TestVerificationAPI:
    """Test verification via TestClient."""

    def test_verify_valid(self, client):
        client.post_citation({"citation_id": "defra-2024", "citation_type": "emission_factor",
                               "source_authority": "defra", "version": "2024"})
        resp = client.post_verify("defra-2024")
        assert resp["status"] == 200
        assert resp["data"]["status"] == "verified"

    def test_verify_expired(self, client):
        client.post_citation({"citation_id": "old", "citation_type": "emission_factor",
                               "expiration_date": "2021-12-31", "version": "2020"})
        resp = client.post_verify("old")
        assert resp["data"]["status"] == "expired"

    def test_verify_not_found(self, client):
        resp = client.post_verify("nonexistent")
        assert resp["status"] == 404


class TestEvidencePackageAPI:
    """Test evidence package lifecycle via TestClient."""

    def test_create_package(self, client):
        resp = client.post_package({"name": "Scope 1 Evidence"})
        assert resp["status"] == 201

    def test_add_evidence(self, client):
        pkg_resp = client.post_package({"name": "Test"})
        pid = pkg_resp["data"]["package_id"]
        resp = client.post_add_evidence(pid, {"evidence_type": "calculation", "data": {"result": 26800}})
        assert resp["status"] == 200
        assert resp["data"]["evidence_count"] == 1

    def test_add_citation_to_package(self, client):
        pkg_resp = client.post_package({"name": "Test"})
        pid = pkg_resp["data"]["package_id"]
        resp = client.post_add_citation(pid, "defra-2024")
        assert resp["status"] == 200

    def test_finalize_package(self, client):
        pkg_resp = client.post_package({"name": "Test"})
        pid = pkg_resp["data"]["package_id"]
        resp = client.post_finalize(pid)
        assert resp["status"] == 200
        assert len(resp["data"]["package_hash"]) == 64

    def test_add_evidence_to_finalized_fails(self, client):
        pkg_resp = client.post_package({"name": "Test"})
        pid = pkg_resp["data"]["package_id"]
        client.post_finalize(pid)
        resp = client.post_add_evidence(pid, {"data": {}})
        assert resp["status"] == 409

    def test_double_finalize_fails(self, client):
        pkg_resp = client.post_package({"name": "Test"})
        pid = pkg_resp["data"]["package_id"]
        client.post_finalize(pid)
        resp = client.post_finalize(pid)
        assert resp["status"] == 409


class TestExportImportAPI:
    """Test export/import via TestClient."""

    def test_export(self, client):
        client.post_citation({"citation_id": "c1", "citation_type": "emission_factor", "title": "C1"})
        resp = client.post_export("json")
        assert resp["status"] == 200

    def test_import(self, client):
        data = {"citations": [
            {"citation_id": "c1", "title": "C1", "citation_type": "emission_factor"},
            {"citation_id": "c2", "title": "C2", "citation_type": "regulatory"},
        ]}
        resp = client.post_import(data)
        assert resp["status"] == 200
        assert resp["data"]["imported"] == 2


class TestHealthEndpoint:
    """Test health endpoint."""

    def test_health(self, client):
        resp = client.get_health()
        assert resp["status"] == "healthy"
        assert resp["version"] == "1.0.0"


class TestErrorResponses:
    """Test error response codes."""

    def test_404_citation_not_found(self, client):
        resp = client.get_citation("nonexistent")
        assert resp["status"] == 404

    def test_400_missing_type(self, client):
        resp = client.post_citation({"title": "Test"})
        assert resp["status"] == 400

    def test_409_duplicate(self, client):
        client.post_citation({"citation_id": "c1", "citation_type": "emission_factor"})
        resp = client.post_citation({"citation_id": "c1", "citation_type": "emission_factor"})
        assert resp["status"] == 409

    def test_404_package_not_found(self, client):
        resp = client.get_package("nonexistent")
        assert resp["status"] == 404

    def test_400_package_missing_name(self, client):
        resp = client.post_package({})
        assert resp["status"] == 400
