from fastapi.testclient import TestClient

from cbam_pack.web.app import create_app


def test_shell_home_renders() -> None:
    client = TestClient(create_app())
    response = client.get("/apps")
    assert response.status_code == 200
    assert "GreenLang Compliance Workspace" in response.text


def test_csrd_vcci_workspaces_render() -> None:
    client = TestClient(create_app())
    assert client.get("/apps/csrd").status_code == 200
    assert client.get("/apps/vcci").status_code == 200
    assert client.get("/apps/eudr").status_code == 200
    assert client.get("/apps/ghg").status_code == 200
    assert client.get("/apps/iso14064").status_code == 200


def test_runs_center_renders() -> None:
    client = TestClient(create_app())
    response = client.get("/runs")
    assert response.status_code == 200
    assert "Run Center" in response.text


def test_runs_api_lists_empty_initially() -> None:
    client = TestClient(create_app())
    response = client.get("/api/v1/runs")
    assert response.status_code == 200
    payload = response.json()
    assert payload["runs"] == []

