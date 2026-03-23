from pathlib import Path

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
