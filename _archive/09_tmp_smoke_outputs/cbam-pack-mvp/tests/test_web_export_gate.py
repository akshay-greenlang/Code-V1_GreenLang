from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from cbam_pack.web.app import create_app


def test_download_blocked_when_export_not_allowed(tmp_path: Path) -> None:
    app = create_app()
    app.state.output_dirs["s1"] = tmp_path
    app.state.session_meta["s1"] = {"can_export": False}

    client = TestClient(app)
    response = client.get("/api/download/s1")

    assert response.status_code == 409
    assert "Export blocked" in response.json()["detail"]


def test_download_allowed_when_export_allowed(tmp_path: Path) -> None:
    app = create_app()
    artifact = tmp_path / "sample.txt"
    artifact.write_text("ok", encoding="utf-8")
    app.state.output_dirs["s2"] = tmp_path
    app.state.session_meta["s2"] = {"can_export": True}

    client = TestClient(app)
    response = client.get("/api/download/s2")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/zip")


def test_e2e_process_blocks_download_when_policy_blocks_export() -> None:
    mvp_root = Path(__file__).resolve().parents[1]
    sample_config = mvp_root / "examples" / "sample_config.yaml"
    sample_imports = mvp_root / "examples" / "sample_imports.csv"

    config_data = yaml.safe_load(sample_config.read_text(encoding="utf-8"))
    config_data["policy"] = {
        "default_usage_cap": 0.0,
        "block_export_on_fail": True,
    }
    strict_config = yaml.safe_dump(config_data, sort_keys=False)

    app = create_app()
    client = TestClient(app)
    process = client.post(
        "/api/process",
        files={
            "config_file": ("config.yaml", strict_config.encode("utf-8"), "application/x-yaml"),
            "imports_file": ("imports.csv", sample_imports.read_bytes(), "text/csv"),
        },
    )
    assert process.status_code == 200
    payload = process.json()
    assert payload["success"] is True
    assert payload["compliance"]["can_export"] is False
    assert payload["compliance"]["policy_status"] == "FAIL"

    blocked = client.get(f"/api/download/{payload['session_id']}")
    assert blocked.status_code == 409
    assert "Export blocked" in blocked.json()["detail"]
