from fastapi.testclient import TestClient

from cbam_pack.web.app import create_app


def test_v1_bundle_download_blocked_when_export_not_allowed(tmp_path) -> None:
    app = create_app()
    run_id = "a" * 32
    app.state.output_dirs[run_id] = tmp_path
    app.state.session_meta[run_id] = {"can_export": False}

    client = TestClient(app)
    response = client.get(f"/api/v1/runs/{run_id}/bundle")

    assert response.status_code == 409
    assert "Export blocked" in response.json()["detail"]


def test_v1_artifact_download_blocked_when_export_not_allowed(tmp_path) -> None:
    app = create_app()
    run_id = "b" * 32
    app.state.output_dirs[run_id] = tmp_path
    app.state.session_meta[run_id] = {"can_export": False}

    client = TestClient(app)
    response = client.get(f"/api/v1/runs/{run_id}/artifacts/esrs_report.json")

    assert response.status_code == 409
    assert "Export blocked" in response.json()["detail"]

