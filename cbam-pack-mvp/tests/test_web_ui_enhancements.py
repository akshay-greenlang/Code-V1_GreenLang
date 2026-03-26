from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import cbam_pack.web.app as web_app
from cbam_pack.web.app import create_app


class _FakeBackendResult:
    def __init__(self, artifacts: list[str]) -> None:
        self.success = True
        self.exit_code = 0
        self.artifacts = artifacts
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.native_backend_used = True
        self.fallback_used = False


def test_ui_script_is_served_and_injected() -> None:
    client = TestClient(create_app())
    script = client.get("/ui.js")
    assert script.status_code == 200
    assert "Command Palette" in script.text

    csrd_page = client.get("/apps/csrd")
    assert csrd_page.status_code == 200
    assert "/ui.js" in csrd_page.text


def test_client_error_telemetry_endpoint_accepts_payload() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/api/telemetry/client-error",
        json={"type": "window_error", "message": "test error", "path": "/apps/cbam"},
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_demo_run_endpoints_create_runs(monkeypatch) -> None:
    def _fake_run_csrd_backend(input_path: Path, output_dir: Path, strict: bool, allow_fallback: bool):
        (output_dir / "audit").mkdir(parents=True, exist_ok=True)
        (output_dir / "esrs_report.json").write_text("{}", encoding="utf-8")
        (output_dir / "audit" / "run_manifest.json").write_text("{}", encoding="utf-8")
        (output_dir / "audit" / "checksums.json").write_text("{}", encoding="utf-8")
        return _FakeBackendResult(["esrs_report.json", "audit/run_manifest.json", "audit/checksums.json"])

    def _fake_run_vcci_backend(input_path: Path, output_dir: Path, strict: bool, allow_fallback: bool):
        (output_dir / "audit").mkdir(parents=True, exist_ok=True)
        (output_dir / "scope3_inventory.json").write_text("{}", encoding="utf-8")
        (output_dir / "audit" / "run_manifest.json").write_text("{}", encoding="utf-8")
        (output_dir / "audit" / "checksums.json").write_text("{}", encoding="utf-8")
        return _FakeBackendResult(["scope3_inventory.json", "audit/run_manifest.json", "audit/checksums.json"])

    monkeypatch.setattr(web_app, "run_csrd_backend", _fake_run_csrd_backend)
    monkeypatch.setattr(web_app, "run_vcci_backend", _fake_run_vcci_backend)

    client = TestClient(create_app())
    csrd_demo = client.post("/api/v1/apps/csrd/demo-run")
    assert csrd_demo.status_code == 200
    assert csrd_demo.json().get("run_id")

    vcci_demo = client.post("/api/v1/apps/vcci/demo-run")
    assert vcci_demo.status_code == 200
    assert vcci_demo.json().get("run_id")
