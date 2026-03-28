from fastapi.testclient import TestClient

from cbam_pack.web.app import create_app


def test_shell_home_contains_visual_baseline_tokens() -> None:
    client = TestClient(create_app())
    response = client.get("/apps")
    body = response.text

    assert response.status_code == 200
    assert "GreenLang Compliance Workspace" in body
    assert "Open CBAM Workspace" in body
    assert "Open CSRD Workspace" in body
    assert "Open VCCI Workspace" in body


def test_workspace_visual_baseline_tokens() -> None:
    client = TestClient(create_app())
    cbam = client.get("/apps/cbam").text
    csrd = client.get("/apps/csrd").text
    vcci = client.get("/apps/vcci").text

    assert "CBAM" in cbam
    assert "CSRD Workspace" in csrd
    assert "VCCI Workspace" in vcci

