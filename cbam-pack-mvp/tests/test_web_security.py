from fastapi.testclient import TestClient

import cbam_pack.web.app as web_app
from cbam_pack.web.app import MAX_UPLOAD_BYTES, create_app


def test_process_rejects_traversal_filename() -> None:
    client = TestClient(create_app())
    config_bytes = b"declarant:\n  name: A\n"
    imports_bytes = (
        b"line_id,quarter,year,cn_code,product_description,country_of_origin,quantity,unit\n"
        b"L1,Q1,2025,72061000,Steel,CN,1,tonnes\n"
    )
    response = client.post(
        "/api/process",
        files={
            "config_file": ("../../../etc/passwd", config_bytes, "application/x-yaml"),
            "imports_file": ("imports.csv", imports_bytes, "text/csv"),
        },
        data={"mode": "transitional", "collect_errors": "true"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid upload filename"


def test_process_rejects_oversize_upload() -> None:
    client = TestClient(create_app())
    oversize = b"a" * (MAX_UPLOAD_BYTES + 1)
    imports_bytes = (
        b"line_id,quarter,year,cn_code,product_description,country_of_origin,quantity,unit\n"
        b"L1,Q1,2025,72061000,Steel,CN,1,tonnes\n"
    )
    response = client.post(
        "/api/process",
        files={
            "config_file": ("config.yaml", oversize, "application/x-yaml"),
            "imports_file": ("imports.csv", imports_bytes, "text/csv"),
        },
        data={"mode": "transitional", "collect_errors": "true"},
    )

    assert response.status_code == 413
    assert "size limit" in response.json()["detail"].lower()


def test_preview_config_sanitizes_yaml_error_message() -> None:
    client = TestClient(create_app())
    invalid_yaml = b"declarant: ["
    response = client.post(
        "/api/preview-config",
        files={"config_file": ("config.yaml", invalid_yaml, "application/x-yaml")},
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["success"] is False
    assert payload["error"] == "Invalid YAML. Please fix file formatting and retry."
    assert "C:\\" not in payload["error"]
    assert "Traceback" not in payload["error"]


def test_home_page_uses_rendered_checkmark_symbol() -> None:
    client = TestClient(create_app())
    response = client.get("/")
    body = response.text

    assert response.status_code == 200
    assert "nameEl.textContent = '✓ ' + file.name;" in body


def test_api_key_required_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("CBAM_API_KEY", "secret-token")
    client = TestClient(create_app())

    response = client.post(
        "/api/preview-config",
        files={"config_file": ("config.yaml", b"declarant: [", "application/x-yaml")},
    )
    assert response.status_code == 401

    response = client.post(
        "/api/preview-config",
        files={"config_file": ("config.yaml", b"declarant: [", "application/x-yaml")},
        headers={"X-API-Key": "secret-token"},
    )
    assert response.status_code == 200


def test_rate_limit_applies_per_client(monkeypatch) -> None:
    monkeypatch.delenv("CBAM_API_KEY", raising=False)
    monkeypatch.setattr(web_app, "RATE_LIMIT_PER_MINUTE", 2)
    client = TestClient(create_app())
    payload = {"config_file": ("config.yaml", b"declarant: [", "application/x-yaml")}

    first = client.post("/api/preview-config", files=payload)
    second = client.post("/api/preview-config", files=payload)
    third = client.post("/api/preview-config", files=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429


def test_process_sanitizes_path_from_pipeline_errors(monkeypatch) -> None:
    class FakePipeline:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            class Result:
                success = False
                statistics = {}
                artifacts = []
                errors = [
                    r"Validation failed for C:\Users\aksha\AppData\Local\Temp\cbam_abc123\c.yaml"
                ]
                policy_result = None
                xml_validation = None
                gap_summary = None
                lines_using_defaults = None
                can_export = False

            return Result()

    monkeypatch.setattr(web_app, "CBAMPipeline", FakePipeline)
    client = TestClient(create_app())

    response = client.post(
        "/api/process",
        files={
            "config_file": ("config.yaml", b"declarant:\n  name: A\n", "application/x-yaml"),
            "imports_file": ("imports.csv", b"line_id\n1\n", "text/csv"),
        },
        data={"mode": "transitional", "collect_errors": "true"},
    )
    payload = response.json()
    errors = payload.get("errors", [])

    assert response.status_code == 200
    assert payload["success"] is False
    assert errors
    assert "C:\\Users\\aksha" not in errors[0]
    assert "<redacted-path>" in errors[0]


def test_v1_download_bundle_rejects_invalid_run_id() -> None:
    client = TestClient(create_app())
    response = client.get("/api/v1/runs/not-a-valid-id/bundle")
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid run ID"


def test_v1_download_artifact_rejects_invalid_run_id() -> None:
    client = TestClient(create_app())
    response = client.get("/api/v1/runs/not-a-valid-id/artifacts/audit/run_manifest.json")
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid run ID"
