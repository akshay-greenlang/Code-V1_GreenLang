from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

import cbam_pack.web.app as web_app
from cbam_pack.web.app import create_app


class _FakeV2BackendResult:
    def __init__(self, artifacts: list[str], blocked: bool = False):
        self.success = True
        self.exit_code = 4 if blocked else 0
        self.artifacts = artifacts
        self.errors = []
        self.warnings = ["policy gate blocked export"] if blocked else []
        self.native_backend_used = True
        self.fallback_used = False


def _fake_run_v2_backend(profile_key: str, input_path: Path, output_dir: Path, strict: bool, allow_fallback: bool):
    del strict, allow_fallback
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit").mkdir(parents=True, exist_ok=True)
    if profile_key == "eudr":
        artifact = "due_diligence_statement.json"
        payload = {"app_id": "GL-EUDR-APP", "status": "ok"}
    elif profile_key == "ghg":
        artifact = "ghg_inventory.json"
        payload = {"app_id": "GL-GHG-APP", "status": "ok"}
    elif profile_key == "iso14064":
        artifact = "iso14064_verification_report.json"
        payload = {"app_id": "GL-ISO14064-APP", "status": "ok"}
    elif profile_key == "sb253":
        artifact = "sb253_disclosure.json"
        payload = {"app_id": "GL-SB253-APP", "status": "ok"}
    elif profile_key == "taxonomy":
        artifact = "taxonomy_alignment.json"
        payload = {"app_id": "GL-Taxonomy-APP", "status": "ok"}
    else:
        artifact = "iso14064_verification_report.json"
        payload = {"app_id": "GL-ISO14064-APP", "status": "ok"}
    (output_dir / artifact).write_text(json.dumps(payload), encoding="utf-8")
    (output_dir / "audit" / "run_manifest.json").write_text("{}", encoding="utf-8")
    (output_dir / "audit" / "checksums.json").write_text("{}", encoding="utf-8")
    return _FakeV2BackendResult([artifact, "audit/run_manifest.json", "audit/checksums.json"])


def test_v2_workspace_routes_render() -> None:
    client = TestClient(create_app())
    assert client.get("/apps/eudr").status_code == 200
    assert client.get("/apps/ghg").status_code == 200
    assert client.get("/apps/iso14064").status_code == 200
    assert client.get("/apps/sb253").status_code == 200
    assert client.get("/apps/taxonomy").status_code == 200


def test_v2_run_endpoints_create_runs(monkeypatch) -> None:
    monkeypatch.setattr(web_app, "run_v2_profile_backend", _fake_run_v2_backend)
    client = TestClient(create_app())

    eudr = client.post("/api/v1/apps/eudr/run", files={"input_file": ("eudr.json", b"{}", "application/json")})
    ghg = client.post("/api/v1/apps/ghg/run", files={"input_file": ("ghg.json", b"{}", "application/json")})
    iso = client.post("/api/v1/apps/iso14064/run", files={"input_file": ("iso.json", b"{}", "application/json")})
    sb = client.post("/api/v1/apps/sb253/run", files={"input_file": ("sb.json", b"{}", "application/json")})
    tax = client.post("/api/v1/apps/taxonomy/run", files={"input_file": ("tax.json", b"{}", "application/json")})

    assert eudr.status_code == 200
    assert ghg.status_code == 200
    assert iso.status_code == 200
    assert sb.status_code == 200
    assert tax.status_code == 200
    assert eudr.json()["app_id"] == "eudr"
    assert ghg.json()["app_id"] == "ghg"
    assert iso.json()["app_id"] == "iso14064"
    assert sb.json()["app_id"] == "sb253"
    assert tax.json()["app_id"] == "taxonomy"


def test_v2_blocked_run_disables_bundle_export(monkeypatch) -> None:
    def _blocked_backend(profile_key: str, input_path: Path, output_dir: Path, strict: bool, allow_fallback: bool):
        del profile_key, input_path, strict, allow_fallback
        (output_dir / "audit").mkdir(parents=True, exist_ok=True)
        (output_dir / "due_diligence_statement.json").write_text(
            json.dumps({"app_id": "GL-EUDR-APP", "status": "blocked"}),
            encoding="utf-8",
        )
        (output_dir / "audit" / "run_manifest.json").write_text("{}", encoding="utf-8")
        (output_dir / "audit" / "checksums.json").write_text("{}", encoding="utf-8")
        return _FakeV2BackendResult(
            ["due_diligence_statement.json", "audit/run_manifest.json", "audit/checksums.json"],
            blocked=True,
        )

    monkeypatch.setattr(web_app, "run_v2_profile_backend", _blocked_backend)
    client = TestClient(create_app())
    response = client.post(
        "/api/v1/apps/eudr/run",
        files={"input_file": ("eudr.json", b"{}", "application/json")},
    )
    assert response.status_code == 200
    run_id = response.json()["run_id"]

    bundle = client.get(f"/api/v1/runs/{run_id}/bundle")
    assert bundle.status_code == 409
