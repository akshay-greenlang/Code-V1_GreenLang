from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

import cbam_pack.web.app as web_app
from cbam_pack.web.app import create_app


class _FakeBackendResult:
    def __init__(self, artifacts: list[str], errors: list[str] | None = None, warnings: list[str] | None = None):
        self.success = True
        self.exit_code = 0
        self.artifacts = artifacts
        self.errors = errors or []
        self.warnings = warnings or []
        self.native_backend_used = True
        self.fallback_used = False


def test_csrd_run_endpoint_creates_run_and_downloads(tmp_path: Path, monkeypatch) -> None:
    def _fake_run_csrd_backend(input_path: Path, output_dir: Path, strict: bool, allow_fallback: bool):
        (output_dir / "audit").mkdir(parents=True, exist_ok=True)
        (output_dir / "esrs_report.json").write_text(
            json.dumps({"app_id": "GL-CSRD-APP", "status": "generated"}, indent=2),
            encoding="utf-8",
        )
        (output_dir / "audit" / "run_manifest.json").write_text("{}", encoding="utf-8")
        (output_dir / "audit" / "checksums.json").write_text("{}", encoding="utf-8")
        return _FakeBackendResult(["esrs_report.json", "audit/run_manifest.json", "audit/checksums.json"])

    monkeypatch.setattr(web_app, "run_csrd_backend", _fake_run_csrd_backend)

    client = TestClient(create_app())
    response = client.post(
        "/api/v1/apps/csrd/run",
        files={"input_file": ("input.csv", b"a,b\n1,2\n", "text/csv")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["app_id"] == "csrd"
    assert payload["execution_mode"] in {"native", "fallback", "unknown"}
    run_id = payload["run_id"]

    listed = client.get("/api/v1/runs").json()["runs"]
    assert any(item["run_id"] == run_id for item in listed)

    bundle = client.get(f"/api/v1/runs/{run_id}/bundle")
    assert bundle.status_code == 200
    assert bundle.headers["content-type"].startswith("application/zip")


def test_vcci_run_endpoint_creates_run_and_downloads(tmp_path: Path, monkeypatch) -> None:
    def _fake_run_vcci_backend(input_path: Path, output_dir: Path, strict: bool, allow_fallback: bool):
        (output_dir / "audit").mkdir(parents=True, exist_ok=True)
        (output_dir / "scope3_inventory.json").write_text(
            json.dumps({"app_id": "GL-VCCI-Carbon-APP", "total_emissions_kgco2e": 0.0}, indent=2),
            encoding="utf-8",
        )
        (output_dir / "audit" / "run_manifest.json").write_text("{}", encoding="utf-8")
        (output_dir / "audit" / "checksums.json").write_text("{}", encoding="utf-8")
        return _FakeBackendResult(["scope3_inventory.json", "audit/run_manifest.json", "audit/checksums.json"])

    monkeypatch.setattr(web_app, "run_vcci_backend", _fake_run_vcci_backend)

    client = TestClient(create_app())
    response = client.post(
        "/api/v1/apps/vcci/run",
        files={"input_file": ("input.csv", b"x\n1\n", "text/csv")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["app_id"] == "vcci"
    run_id = payload["run_id"]

    artifact = client.get(f"/api/v1/runs/{run_id}/artifacts/scope3_inventory.json")
    assert artifact.status_code == 200

