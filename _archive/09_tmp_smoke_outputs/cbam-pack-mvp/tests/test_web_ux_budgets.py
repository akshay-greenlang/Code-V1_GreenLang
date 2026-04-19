from fastapi.testclient import TestClient

from cbam_pack.web.app import create_app


def test_shell_html_size_budget() -> None:
    client = TestClient(create_app())
    response = client.get("/apps")
    assert response.status_code == 200
    # Keep shell payload bounded for fast first render.
    assert len(response.text.encode("utf-8")) <= 110_000


def test_shell_navigation_budget_tokens() -> None:
    client = TestClient(create_app())
    body = client.get("/apps").text
    required_links = [
        "/apps/cbam",
        "/apps/csrd",
        "/apps/vcci",
        "/apps/eudr",
        "/apps/ghg",
        "/apps/iso14064",
        "/runs",
    ]
    for link in required_links:
        assert link in body
