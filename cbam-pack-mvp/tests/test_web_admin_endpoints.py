from fastapi.testclient import TestClient

from cbam_pack.web.app import create_app


def test_admin_release_train_endpoint() -> None:
    client = TestClient(create_app())
    response = client.get("/api/v1/admin/release-train")
    assert response.status_code == 200
    payload = response.json()
    assert "available" in payload


def test_admin_connectors_endpoint() -> None:
    client = TestClient(create_app())
    response = client.get("/api/v1/admin/connectors")
    assert response.status_code == 200
    payload = response.json()
    assert "connectors" in payload
    assert isinstance(payload["connectors"], list)
    assert payload["connectors"]
    assert "connector_id" in payload["connectors"][0]


def test_shell_chrome_context_endpoint() -> None:
    client = TestClient(create_app())
    response = client.get("/api/v1/shell/chrome-context")
    assert response.status_code == 200
    payload = response.json()
    assert "compliance_rail" in payload
    assert "connector_incidents" in payload
    assert "managed_pack_count" in payload["compliance_rail"]
    assert isinstance(payload["connector_incidents"], list)
    assert any(i.get("connector_id") == "azure-iot" for i in payload["connector_incidents"])


def test_admin_connectors_health_stub() -> None:
    client = TestClient(create_app())
    response = client.get("/api/v1/admin/connectors/health")
    assert response.status_code == 200
    payload = response.json()
    assert "updated_at_utc" in payload
    assert "probes" in payload
    assert isinstance(payload["probes"], list)
    assert payload["probes"]
    assert "latency_ms" in payload["probes"][0]
