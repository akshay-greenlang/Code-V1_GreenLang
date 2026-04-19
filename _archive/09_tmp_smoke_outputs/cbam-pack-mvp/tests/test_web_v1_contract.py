from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import cbam_pack.web.app as web_app
from cbam_pack.web.app import create_app


def test_cbam_v1_run_endpoint_returns_normalized_shape(monkeypatch, tmp_path: Path) -> None:
    """
    Ensure /api/v1/apps/cbam/run returns the cross-app shape without requiring
    real pipeline execution.
    """

    class FakePipeline:
        def __init__(self, config_path, imports_path, output_dir, verbose, dry_run):
            self.output_dir = output_dir

        def run(self):
            (self.output_dir / "audit").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "cbam_report.xml").write_text("<cbam/>", encoding="utf-8")
            (self.output_dir / "report_summary.xlsx").write_text("binary-placeholder", encoding="utf-8")
            (self.output_dir / "audit" / "run_manifest.json").write_text("{}", encoding="utf-8")
            (self.output_dir / "audit" / "checksums.json").write_text("{}", encoding="utf-8")

            class Result:
                success = True
                statistics = {"total_lines": 1, "total_emissions_tco2e": 0.0, "default_usage_percent": 0.0, "lines_using_defaults": 0}
                artifacts = ["cbam_report.xml", "report_summary.xlsx", "audit/run_manifest.json", "audit/checksums.json"]
                errors = []
                policy_result = {"status": "PASS", "overall_score": 100.0, "can_export": True, "violations": [], "warnings": []}
                xml_validation = {"status": "PASS", "schema_version": "1.0", "schema_date": "2025-01-01", "errors": []}
                gap_summary = {"status": "PASS"}
                lines_using_defaults = []
                can_export = True

            return Result()

    monkeypatch.setattr(web_app, "CBAMPipeline", FakePipeline)

    client = TestClient(create_app())
    response = client.post(
        "/api/v1/apps/cbam/run",
        files={
            "config_file": ("config.yaml", b"declarant:\n  name: A\n", "application/x-yaml"),
            "imports_file": ("imports.csv", b"line_id\n1\n", "text/csv"),
        },
        data={"mode": "transitional", "collect_errors": "true"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["app_id"] == "cbam"
    assert payload["execution_mode"] == "native"
    assert payload["status"] in {"completed", "failed"}
    assert isinstance(payload["artifacts"], list)
    assert "run_id" in payload

